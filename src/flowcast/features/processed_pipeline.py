"""Versioned Step 08 pipeline for multi-horizon processed data."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from flowcast.data.artifacts import (
    artifact_record,
    validate_artifact_version,
    write_json,
    write_parquet,
)
from flowcast.data.quality_report import render_processed_data_markdown
from flowcast.features.config import load_feature_config
from flowcast.features.inputs import load_verified_features
from flowcast.features.targets import engineer_targets
from flowcast.settings import Settings


@dataclass(frozen=True)
class ProcessedArtifacts:
    """Paths, processed table, manifest, and summary for one Step 08 run."""

    version: str
    output_dir: Path
    dataset_path: Path
    manifest_path: Path
    summary_path: Path
    markdown_path: Path
    frame: pd.DataFrame
    manifest: dict[str, Any]
    summary: dict[str, Any]


def _schema(frame: pd.DataFrame, feature_columns: tuple[str, ...]) -> list[dict]:
    feature_set = set(feature_columns)
    records = []
    for name in frame.columns:
        if name in {"road_id", "timestamp"}:
            role = "key"
        elif name.startswith("target_timestamp_"):
            role = "target_timestamp"
        elif name.startswith("target_") and name.endswith("_available"):
            role = "target_availability"
        elif name.startswith("target_"):
            role = "target"
        elif name in feature_set:
            role = "feature_or_lineage"
        else:
            raise RuntimeError(f"Unclassified processed column: {name}")
        records.append({"name": name, "dtype": str(frame[name].dtype), "role": role})
    return records


def _coverage(
    frame: pd.DataFrame,
    definition: dict[str, Any],
) -> dict[str, Any]:
    name = str(definition["name"])
    available = frame[str(definition["availability_column"])].fillna(False).astype(bool)
    target = frame[name]
    record: dict[str, Any] = {
        "available_rows": int(available.sum()),
        "unavailable_rows": int((~available).sum()),
        "null_target_rows": int(target.isna().sum()),
    }
    if definition["task"] == "classification_binary":
        record["positive_rows"] = int(target[available].astype(bool).sum())
        record["negative_rows"] = int(available.sum()) - record["positive_rows"]
    elif definition["task"] == "classification_multiclass":
        record["class_counts"] = {
            str(label): int(count)
            for label, count in target[available].value_counts().sort_index().items()
        }
    return record


def run_processed_data(
    settings: Settings,
    version: str | None = None,
) -> ProcessedArtifacts:
    """Verify features, construct future targets, and persist processed data."""

    config = load_feature_config(settings)
    selected_version = validate_artifact_version(version or settings.processed_version)
    verified = load_verified_features(settings)
    result = engineer_targets(verified.frame, config)
    frame = result.frame
    sorted_features = verified.frame.sort_values(
        ["road_id", "timestamp"], kind="stable"
    ).reset_index(drop=True)
    if not frame[list(result.feature_columns)].equals(sorted_features):
        raise RuntimeError("Target engineering changed an explanatory input column")

    output_dir = settings.processed_dir / selected_version
    dataset_path = output_dir / "dataset.parquet"
    write_parquet(frame, dataset_path)

    target_records = []
    for definition in result.definitions:
        record = asdict(definition)
        record["dtype"] = str(frame[definition.name].dtype)
        record["version"] = selected_version
        target_records.append(record)
    feature_candidate_count = int(verified.manifest["feature_count"])
    manifest_dir = settings.artifacts_dir / "features" / selected_version
    manifest_path = manifest_dir / "manifest.json"
    manifest: dict[str, Any] = {
        "contract_version": str(config["targets"]["contract_version"]),
        "processed_version": selected_version,
        "input_feature_version": settings.feature_version,
        "key_columns": ["road_id", "timestamp"],
        "cadence_minutes": int(config["targets"]["cadence_minutes"]),
        "forecast_horizons": [
            int(value) for value in config["forecast_horizons_reserved"]
        ],
        "row_count": len(frame),
        "feature_column_count": len(result.feature_columns),
        "model_candidate_feature_count": feature_candidate_count,
        "target_count": len(target_records),
        "configuration": {
            "base": artifact_record(settings.config_path, settings),
            "features": artifact_record(settings.features_config_path, settings),
        },
        "input_summary": artifact_record(verified.summary_path, settings),
        "input_manifest": artifact_record(verified.manifest_path, settings),
        "input_artifact": artifact_record(verified.path, settings),
        "output_artifact": artifact_record(dataset_path, settings),
        "columns": _schema(frame, result.feature_columns),
        "targets": target_records,
    }
    write_json(manifest, manifest_path)

    road_count = int(frame["road_id"].nunique())
    timestamp_coverage = {}
    for horizon in manifest["forecast_horizons"]:
        timestamp_column = f"target_timestamp_h{horizon}"
        available = int(frame[timestamp_column].notna().sum())
        timestamp_coverage[str(horizon)] = {
            "horizon_minutes": horizon * manifest["cadence_minutes"],
            "available_rows": available,
            "unavailable_rows": len(frame) - available,
            "expected_trailing_unavailable_rows": road_count * horizon,
        }
    target_coverage = {
        record["name"]: _coverage(frame, record) for record in target_records
    }
    unique_keys = len(frame.drop_duplicates(["road_id", "timestamp"]))
    quality_dir = settings.artifacts_dir / "quality" / selected_version
    summary_path = quality_dir / "summary.json"
    markdown_path = quality_dir / "summary.md"
    summary: dict[str, Any] = {
        "contract_version": manifest["contract_version"],
        "processed_version": selected_version,
        "input_feature_version": settings.feature_version,
        "configuration": manifest["configuration"],
        "input_summary": manifest["input_summary"],
        "input_manifest": manifest["input_manifest"],
        "input_artifact": manifest["input_artifact"],
        "dataset": {
            "input_rows": len(verified.frame),
            "output_rows": len(frame),
            "row_count_change": len(frame) - len(verified.frame),
            "road_count": road_count,
            "output_unique_keys": unique_keys,
            "duplicate_output_keys": len(frame) - unique_keys,
            "feature_column_count": len(result.feature_columns),
            "model_candidate_feature_count": feature_candidate_count,
            "target_count": len(target_records),
            "timestamp_coverage": timestamp_coverage,
            "target_coverage": target_coverage,
            "processed_artifact": artifact_record(dataset_path, settings),
            "manifest_artifact": artifact_record(manifest_path, settings),
        },
    }
    write_json(summary, summary_path)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(
        render_processed_data_markdown(summary),
        encoding="utf-8",
        newline="\n",
    )
    return ProcessedArtifacts(
        version=selected_version,
        output_dir=output_dir,
        dataset_path=dataset_path,
        manifest_path=manifest_path,
        summary_path=summary_path,
        markdown_path=markdown_path,
        frame=frame,
        manifest=manifest,
        summary=summary,
    )
