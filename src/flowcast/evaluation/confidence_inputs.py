"""Verified frozen inputs for confidence and error analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from flowcast.data.artifacts import artifact_record, verify_artifact_record
from flowcast.evaluation.confidence_config import load_confidence_config
from flowcast.features.inputs import VerifiedProcessedInput, load_verified_processed
from flowcast.modelling.classical_artifacts import classical_regression_paths
from flowcast.modelling.classification_artifacts import classification_paths
from flowcast.modelling.recurrent_artifacts import read_json, recurrent_paths
from flowcast.modelling.recurrent_config import load_recurrent_config
from flowcast.modelling.registry_artifacts import (
    classical_registry_paths,
    load_classical_registry,
    record_path,
)
from flowcast.settings import Settings


@dataclass(frozen=True)
class ConfidenceInputs:
    """Frozen prediction tables, processed context, and exact lineage."""

    processed: VerifiedProcessedInput
    regression: pd.DataFrame | None
    classification: pd.DataFrame | None
    recurrent: pd.DataFrame | None
    upstream_records: dict[str, dict[str, Any]]


def _verified_recurrent_summary(
    settings: Settings,
    version: str,
) -> tuple[dict[str, Any], Any]:
    paths = recurrent_paths(settings, version)
    summary = read_json(paths.summary_path)
    if summary.get("contract_version") != "recurrent_volume_v1":
        raise RuntimeError("Unsupported recurrent summary contract")
    if summary.get("version") != version:
        raise RuntimeError("Recurrent summary version changed")
    config, _, config_path = load_recurrent_config(settings)
    if str(config["version"]) != version:
        raise RuntimeError("Recurrent configuration version changed")
    verify_artifact_record(config_path, summary["configuration"], settings)
    for record in summary["upstream"].values():
        verify_artifact_record(record_path(record, settings), record, settings)
    for record in summary["artifacts"].values():
        verify_artifact_record(record_path(record, settings), record, settings)
    return summary, paths


def load_verified_confidence_inputs(
    settings: Settings,
    config: dict[str, Any] | None = None,
    *,
    load_frames: bool = True,
) -> ConfidenceInputs:
    """Verify complete upstream chains and optionally load prediction frames."""

    if config is None:
        config, _ = load_confidence_config(settings)
    upstream = config["upstream"]
    processed_version = str(upstream["processed_version"])
    if processed_version != settings.processed_version:
        raise RuntimeError("Confidence processed version differs from base settings")
    processed = load_verified_processed(settings)

    registry_version = str(upstream["registry_version"])
    _, registry_summary = load_classical_registry(
        settings,
        version=registry_version,
    )
    registry_paths = classical_registry_paths(settings, registry_version)
    regression_version = str(upstream["regression_version"])
    classification_version = str(upstream["classification_version"])
    recurrent_version = str(upstream["recurrent_version"])
    regression_paths = classical_regression_paths(settings, regression_version)
    classification_source_paths = classification_paths(
        settings, classification_version
    )
    recurrent_summary, recurrent_source_paths = _verified_recurrent_summary(
        settings, recurrent_version
    )

    regression_summary = read_json(regression_paths.summary_path)
    classification_summary = read_json(classification_source_paths.summary_path)
    source_records = registry_summary["sources"]
    verify_artifact_record(
        regression_paths.summary_path,
        source_records["regression"],
        settings,
    )
    verify_artifact_record(
        classification_source_paths.summary_path,
        source_records["classification"],
        settings,
    )
    upstream_records = {
        "processed_summary": artifact_record(processed.summary_path, settings),
        "processed_manifest": artifact_record(processed.manifest_path, settings),
        "processed_dataset": artifact_record(processed.path, settings),
        "classical_registry_summary": artifact_record(
            registry_paths.summary_path, settings
        ),
        "classical_regression_summary": artifact_record(
            regression_paths.summary_path, settings
        ),
        "classification_summary": artifact_record(
            classification_source_paths.summary_path, settings
        ),
        "recurrent_summary": artifact_record(
            recurrent_source_paths.summary_path, settings
        ),
    }
    if not load_frames:
        return ConfidenceInputs(processed, None, None, None, upstream_records)

    regression = pd.read_parquet(regression_paths.predictions_path)
    classification = pd.read_parquet(
        classification_source_paths.predictions_path
    )
    recurrent = pd.read_parquet(recurrent_source_paths.predictions_path)
    expected_rows = {
        "regression": int(regression_summary["coverage"]["prediction_rows"]),
        "classification": int(
            classification_summary["coverage"]["prediction_rows"]
        ),
        "recurrent": int(recurrent_summary["coverage"]["total_prediction_rows"]),
    }
    actual_rows = {
        "regression": len(regression),
        "classification": len(classification),
        "recurrent": len(recurrent),
    }
    if actual_rows != expected_rows:
        raise RuntimeError(
            f"Frozen prediction row counts changed: {actual_rows} != {expected_rows}"
        )
    return ConfidenceInputs(
        processed,
        regression,
        classification,
        recurrent,
        upstream_records,
    )
