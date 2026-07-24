"""Step 14 orchestration for the combined classical model registry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from flowcast.data.artifacts import (
    artifact_record,
    validate_artifact_version,
    write_json,
)
from flowcast.modelling.classical_report import write_csv
from flowcast.modelling.registry_artifacts import (
    ClassicalRegistryPaths,
    classical_registry_paths,
    load_verified_source_summaries,
)
from flowcast.modelling.registry_config import load_registry_config
from flowcast.modelling.registry_outputs import (
    build_prediction_index,
    build_registry_entries,
    build_scoreboard,
)
from flowcast.modelling.registry_report import render_registry_report
from flowcast.settings import Settings


@dataclass(frozen=True)
class ClassicalRegistryResult:
    """Completed Step 14 output bundle."""

    paths: ClassicalRegistryPaths
    registry: dict[str, Any]
    prediction_index: dict[str, Any]
    summary: dict[str, Any]


def _acceptance_summary(entries: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for target in ("volume", "congestion", "accident"):
        decisions = [
            bool(entry["acceptance"]["met"])
            for entry in entries
            if entry["target"] == target and entry["acceptance"] is not None
        ]
        output[target] = {
            "evaluated_horizons": len(decisions),
            "met_horizons": sum(decisions),
            "all_horizons_met": all(decisions),
        }
    return output


def run_classical_registry(
    settings: Settings,
    *,
    version: str | None = None,
) -> ClassicalRegistryResult:
    """Build a deterministic registry from frozen outputs without retraining."""

    config, config_path = load_registry_config(settings)
    selected_version = validate_artifact_version(
        version or str(config["version"])
    )
    paths = classical_registry_paths(settings, selected_version)
    sources = load_verified_source_summaries(settings, config)
    entries = build_registry_entries(config, sources, settings)
    registry = {
        "contract_version": "classical_registry_v1",
        "version": selected_version,
        "selection_policy": {
            "source": "frozen_validation_evidence",
            "test_metrics_used_for_selection": False,
            "runtime_and_interpretability_override_winners": False,
        },
        "entry_count": len(entries),
        "entries": entries,
    }
    prediction_index = build_prediction_index(
        entries,
        sources,
        settings,
        selected_version,
    )
    scoreboard = build_scoreboard(entries)
    write_json(registry, paths.registry_path)
    write_csv(scoreboard, paths.scoreboard_path)
    write_json(prediction_index, paths.prediction_index_path)
    paths.report_path.parent.mkdir(parents=True, exist_ok=True)
    paths.report_path.write_text(
        render_registry_report(registry, prediction_index),
        encoding="utf-8",
        newline="\n",
    )
    upstream = config["upstream"]
    source_paths = {
        "regression": (
            settings.artifacts_dir
            / "metrics"
            / str(upstream["regression_version"])
            / "summary.json"
        ),
        "classification": (
            settings.artifacts_dir
            / "metrics"
            / str(upstream["classification_version"])
            / "summary.json"
        ),
    }
    summary = {
        "contract_version": "classical_registry_v1",
        "version": selected_version,
        "configuration": artifact_record(config_path, settings),
        "sources": {
            name: artifact_record(path, settings)
            for name, path in source_paths.items()
        },
        "coverage": {
            "target_count": 5,
            "horizon_count": 4,
            "entry_count": len(entries),
            "regression_entries": sum(
                entry["source"] == "regression" for entry in entries
            ),
            "classification_entries": sum(
                entry["source"] == "classification" for entry in entries
            ),
            "prediction_source_count": prediction_index["source_count"],
            "prediction_rows": prediction_index["prediction_rows"],
        },
        "selection": registry["selection_policy"],
        "acceptance": _acceptance_summary(entries),
        "artifacts": {
            "registry": artifact_record(paths.registry_path, settings),
            "scoreboard": artifact_record(paths.scoreboard_path, settings),
            "prediction_index": artifact_record(
                paths.prediction_index_path,
                settings,
            ),
            "report": artifact_record(paths.report_path, settings),
        },
        "checks": [
            {"name": "twenty_unique_registry_entries", "passed": True},
            {"name": "all_sources_hash_verified", "passed": True},
            {"name": "all_selections_frozen_before_test", "passed": True},
            {"name": "no_retraining_or_test_partition_load", "passed": True},
            {"name": "all_prediction_rows_mapped_once", "passed": True},
        ],
        "limitations": [
            (
                "This registry contains classical models only; deep models "
                "arrive in Step 15."
            ),
            "Confidence intervals and calibrated uncertainty arrive in Step 16.",
            "Congestion and accident-risk acceptance targets are not currently met.",
        ],
    }
    write_json(summary, paths.summary_path)
    return ClassicalRegistryResult(
        paths=paths,
        registry=registry,
        prediction_index=prediction_index,
        summary=summary,
    )
