"""Deterministic normalized outputs for the Step 14 classical registry."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from flowcast.modelling.registry_artifacts import read_json, verify_record
from flowcast.settings import Settings


INTERPRETABILITY = {
    "linear_regression": "direct coefficients",
    "decision_tree": "single-tree rules and feature importance",
    "random_forest": "tree-ensemble feature importance",
    "xgboost": "boosted-tree feature importance",
    "svm": "linear decision coefficients",
}


def _scoreboard_map(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["job_id"]): row for row in summary["scoreboard"]}


def _selection_map(
    summary: dict[str, Any],
    settings: Settings,
) -> dict[str, dict[str, Any]]:
    path = verify_record(summary["artifacts"]["selection_manifest"], settings)
    manifest = read_json(path)
    if manifest.get("status") != "frozen_before_test_access":
        raise RuntimeError("Upstream selection was not frozen before test access")
    if manifest.get("test_metrics_present") is not False:
        raise RuntimeError("Upstream selection manifest contains test metrics")
    return {
        str(record["job_id"]): record
        for record in manifest["selections"]
    }


def _acceptance(
    target_config: dict[str, Any],
    test_metrics: dict[str, Any],
) -> dict[str, Any] | None:
    if "acceptance_metric" not in target_config:
        return None
    metric = str(target_config["acceptance_metric"])
    operator = str(target_config["acceptance_operator"])
    threshold = float(target_config["acceptance_value"])
    value = float(test_metrics[metric])
    if operator == "less_than_or_equal":
        met = value <= threshold
    elif operator == "greater_than_or_equal":
        met = value >= threshold
    else:
        raise ValueError(f"Unsupported acceptance operator: {operator}")
    return {
        "metric": metric,
        "operator": operator,
        "threshold": threshold,
        "test_value": value,
        "met": bool(met),
    }


def _runtime(
    source: str,
    selection: dict[str, Any],
    card: dict[str, Any],
) -> dict[str, float]:
    validation = card["metrics"]["validation"]
    if source == "regression":
        fit_seconds = validation["fit_seconds"]
        prediction_seconds = validation["prediction_seconds"]
    else:
        fit_seconds = selection["fit_seconds"]
        prediction_seconds = selection["prediction_seconds"]
    return {
        "validation_fit_seconds": float(fit_seconds),
        "validation_prediction_seconds": float(prediction_seconds),
        "test_prediction_seconds": float(
            card["metrics"]["test"]["prediction_seconds"]
        ),
    }


def _selection_rationale(
    entry: dict[str, Any],
    cv_mean: float,
    cv_std: float,
) -> str:
    metric = str(entry["primary_metric"]).upper()
    validation = float(entry["metrics"]["validation"][entry["primary_metric"]])
    runtime = entry["runtime"]
    return (
        f"Selected before test access because {entry['selected_family']}/"
        f"{entry['candidate_id']} won the frozen validation comparison on "
        f"{metric} ({validation:.10g}) after time-ordered CV "
        f"(mean {cv_mean:.10g}, standard deviation {cv_std:.10g}). "
        f"Validation fit/prediction time was "
        f"{runtime['validation_fit_seconds']:.6g}s/"
        f"{runtime['validation_prediction_seconds']:.6g}s. "
        f"Interpretability context: {entry['interpretability']}. "
        "Runtime and interpretability are governance context only; test metrics "
        "were not selection inputs."
    )


def _entry(
    target_config: dict[str, Any],
    horizon: int,
    summary: dict[str, Any],
    selection: dict[str, Any],
    score: dict[str, Any],
    card: dict[str, Any],
    model_records: dict[str, Any],
    key_template: str,
) -> dict[str, Any]:
    target = str(target_config["key"])
    primary_metric = str(target_config["primary_metric"])
    family = str(card["selection"]["family"])
    model_version = str(card["model_version"])
    cv_mean = float(selection[f"mean_cv_{primary_metric}"])
    cv_std = float(selection[f"std_cv_{primary_metric}"])
    source = str(target_config["source"])
    registry_key = key_template.format(
        target=target,
        horizon=horizon,
        family=family,
        model_version=model_version,
    )
    entry: dict[str, Any] = {
        "registry_key": registry_key,
        "job_id": str(card["job_id"]),
        "target": target,
        "target_column": str(card["target"]["column"]),
        "task_type": str(target_config["task_type"]),
        "source": source,
        "model_version": model_version,
        "horizon_windows": int(horizon),
        "horizon_minutes": int(card["target"]["horizon_minutes"]),
        "seed": int(card["seed"]),
        "selected_family": family,
        "candidate_id": str(card["selection"]["candidate_id"]),
        "hyperparameters": card["selection"]["parameters"],
        "primary_metric": primary_metric,
        "metric_direction": str(target_config["direction"]),
        "selection_evidence": {
            "validation_primary_metric": float(
                card["metrics"]["validation"][primary_metric]
            ),
            "mean_cv_primary_metric": cv_mean,
            "std_cv_primary_metric": cv_std,
            "selection_status": "frozen_before_test_access",
        },
        "metrics": card["metrics"],
        "data": card["data"],
        "features": {
            "input_feature_count": int(
                card["features"]["input_feature_count"]
            ),
            "output_feature_count": int(
                card["features"]["output_feature_count"]
            ),
            "preprocessing_family": str(
                card["features"]["preprocessing_family"]
            ),
            "preprocessing_version": str(
                card["features"]["preprocessing_version"]
            ),
        },
        "probability": card.get("probability"),
        "runtime": _runtime(source, selection, card),
        "interpretability": INTERPRETABILITY[family],
        "acceptance": _acceptance(
            target_config,
            card["metrics"]["test"],
        ),
        "lineage": {
            **card["lineage"],
            "base_config_sha256": summary["configuration"]["base"]["sha256"],
            "models_config_sha256": summary["configuration"]["models"]["sha256"],
        },
        "artifacts": {
            "model": model_records["model"],
            "model_card": model_records["model_card_json"],
            "model_card_markdown": model_records["model_card_markdown"],
            "predictions": summary["artifacts"]["predictions"],
            "selection_manifest": summary["artifacts"]["selection_manifest"],
        },
        "limitations": card["limitations"],
    }
    entry["selection_rationale"] = _selection_rationale(entry, cv_mean, cv_std)
    if score["test"] != card["metrics"]["test"]:
        raise RuntimeError(f"Scoreboard and model card disagree for {card['job_id']}")
    return entry


def build_registry_entries(
    registry_config: dict[str, Any],
    sources: dict[str, dict[str, Any]],
    settings: Settings,
) -> list[dict[str, Any]]:
    """Normalize all frozen source cards into exactly 20 registry entries."""

    if sources["regression"]["input_modeling"] != sources["classification"][
        "input_modeling"
    ]:
        raise RuntimeError("Classical sources do not share modeling lineage")
    if sources["regression"]["configuration"] != sources["classification"][
        "configuration"
    ]:
        raise RuntimeError("Classical sources do not share configuration lineage")
    scoreboards = {
        name: _scoreboard_map(summary)
        for name, summary in sources.items()
    }
    selections = {
        name: _selection_map(summary, settings)
        for name, summary in sources.items()
    }
    entries: list[dict[str, Any]] = []
    for target_config in registry_config["targets"]:
        target = str(target_config["key"])
        source = str(target_config["source"])
        summary = sources[source]
        for horizon in registry_config["horizons"]:
            job_id = f"{target}_h{int(horizon)}"
            model_records = summary["models"].get(job_id)
            if model_records is None:
                raise KeyError(f"Frozen source is missing {job_id}")
            card_path = verify_record(model_records["model_card_json"], settings)
            card = read_json(card_path)
            if card.get("job_id") != job_id:
                raise RuntimeError(f"Model-card identity changed for {job_id}")
            if card["artifacts"]["model"] != model_records["model"]:
                raise RuntimeError(f"Model-card artifact changed for {job_id}")
            if card["lineage"]["selection_sha256"] != summary["artifacts"][
                "selection_manifest"
            ]["sha256"]:
                raise RuntimeError(f"Selection lineage changed for {job_id}")
            entries.append(
                _entry(
                    target_config,
                    int(horizon),
                    summary,
                    selections[source][job_id],
                    scoreboards[source][job_id],
                    card,
                    model_records,
                    str(registry_config["registry_key_template"]),
                )
            )
    if len(entries) != 20:
        raise RuntimeError(f"Expected 20 registry entries, got {len(entries)}")
    return entries


def build_scoreboard(entries: list[dict[str, Any]]) -> pd.DataFrame:
    """Return the normalized task-aware 20-job scoreboard."""

    rows: list[dict[str, Any]] = []
    for entry in entries:
        primary = str(entry["primary_metric"])
        acceptance = entry["acceptance"]
        rows.append(
            {
                "registry_key": entry["registry_key"],
                "job_id": entry["job_id"],
                "target": entry["target"],
                "task_type": entry["task_type"],
                "horizon_windows": entry["horizon_windows"],
                "horizon_minutes": entry["horizon_minutes"],
                "selected_family": entry["selected_family"],
                "candidate_id": entry["candidate_id"],
                "primary_metric": primary,
                "metric_direction": entry["metric_direction"],
                "mean_cv_primary_metric": entry["selection_evidence"][
                    "mean_cv_primary_metric"
                ],
                "std_cv_primary_metric": entry["selection_evidence"][
                    "std_cv_primary_metric"
                ],
                "validation_primary_metric": entry["metrics"]["validation"][
                    primary
                ],
                "test_primary_metric": entry["metrics"]["test"][primary],
                "acceptance_metric": (
                    acceptance["metric"] if acceptance is not None else None
                ),
                "acceptance_threshold": (
                    acceptance["threshold"] if acceptance is not None else None
                ),
                "acceptance_met": (
                    acceptance["met"] if acceptance is not None else None
                ),
                "validation_fit_seconds": entry["runtime"][
                    "validation_fit_seconds"
                ],
                "test_prediction_seconds": entry["runtime"][
                    "test_prediction_seconds"
                ],
                "model_version": entry["model_version"],
                "model_path": entry["artifacts"]["model"]["path"],
                "model_sha256": entry["artifacts"]["model"]["sha256"],
                "model_card_path": entry["artifacts"]["model_card"]["path"],
                "model_card_sha256": entry["artifacts"]["model_card"]["sha256"],
                "prediction_path": entry["artifacts"]["predictions"]["path"],
                "prediction_sha256": entry["artifacts"]["predictions"]["sha256"],
                "preprocessing_version": entry["features"][
                    "preprocessing_version"
                ],
                "processed_sha256": entry["lineage"]["processed_sha256"],
                "feature_schema_sha256": entry["lineage"][
                    "feature_schema_sha256"
                ],
                "selection_sha256": entry["lineage"]["selection_sha256"],
            }
        )
    return pd.DataFrame(rows)


def build_prediction_index(
    entries: list[dict[str, Any]],
    sources: dict[str, dict[str, Any]],
    settings: Settings,
    version: str,
) -> dict[str, Any]:
    """Map every source prediction row to exactly one registry entry."""

    entry_by_job = {str(entry["job_id"]): entry for entry in entries}
    source_payloads: dict[str, Any] = {}
    indexed_rows = 0
    indexed_jobs: set[str] = set()
    for source_name, summary in sources.items():
        record = summary["artifacts"]["predictions"]
        path = verify_record(record, settings)
        frame = pd.read_parquet(path, columns=["job_id", "split"])
        unknown = set(frame["job_id"].astype(str)) - set(entry_by_job)
        if unknown:
            raise RuntimeError(f"Unregistered prediction jobs: {sorted(unknown)}")
        jobs: list[dict[str, Any]] = []
        counts = frame.groupby(["job_id", "split"], sort=True).size()
        for job_id in sorted(frame["job_id"].astype(str).unique()):
            entry = entry_by_job[job_id]
            if entry["source"] != source_name:
                raise RuntimeError(f"Prediction source mismatch for {job_id}")
            split_counts = {
                split: int(counts.get((job_id, split), 0))
                for split in ("validation", "test")
            }
            if any(value <= 0 for value in split_counts.values()):
                raise RuntimeError(f"Prediction split is incomplete for {job_id}")
            if split_counts["validation"] != int(
                entry["data"]["validation_rows"]
            ) or split_counts["test"] != int(entry["data"]["test_rows"]):
                raise RuntimeError(f"Prediction row count changed for {job_id}")
            jobs.append(
                {
                    "job_id": job_id,
                    "registry_key": entry["registry_key"],
                    "rows": {
                        **split_counts,
                        "total": sum(split_counts.values()),
                    },
                }
            )
            indexed_jobs.add(job_id)
        source_payloads[source_name] = {
            "artifact": record,
            "rows": int(len(frame)),
            "jobs": jobs,
        }
        indexed_rows += len(frame)
    if indexed_jobs != set(entry_by_job):
        missing = sorted(set(entry_by_job) - indexed_jobs)
        raise RuntimeError(f"Registry entries lack predictions: {missing}")
    return {
        "contract_version": "classical_prediction_index_v1",
        "registry_version": version,
        "mapping": "indexed_source_manifest",
        "source_count": len(source_payloads),
        "job_count": len(indexed_jobs),
        "prediction_rows": int(indexed_rows),
        "sources": source_payloads,
    }


def canonical_parameters(parameters: dict[str, Any]) -> str:
    """Serialize hyperparameters consistently for reports and tests."""

    return json.dumps(parameters, sort_keys=True, separators=(",", ":"))
