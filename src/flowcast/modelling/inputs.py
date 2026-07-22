"""Verified Step 09 inputs and sealed Step 10 modelling-artifact access."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from flowcast.analysis.pipeline import run_eda
from flowcast.data.artifacts import verify_artifact_record
from flowcast.features.inputs import VerifiedProcessedInput, load_verified_processed
from flowcast.modelling.config import MODEL_FAMILIES, PARTITIONS, load_model_config
from flowcast.settings import Settings


@dataclass(frozen=True)
class VerifiedModelingInputs:
    """Hash-verified processed data, EDA summary, and feature manifest."""

    processed: VerifiedProcessedInput
    eda_path: Path
    eda: dict[str, Any]
    feature_manifest_path: Path
    feature_manifest: dict[str, Any]


@dataclass(frozen=True)
class VerifiedModelingArtifacts:
    """Verified Step 10 summary, assignments, schemas, folds, and preprocessors."""

    summary_path: Path
    summary: dict[str, Any]
    assignments_path: Path
    assignments: pd.DataFrame
    schema_path: Path
    schema: dict[str, Any]
    folds_path: Path
    folds: dict[str, Any]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required modelling artifact is missing: {path}")
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return payload


def _record_path(record: dict[str, Any], settings: Settings) -> Path:
    path = Path(str(record["path"]))
    return path if path.is_absolute() else settings.root / path


def _verify_record(record: dict[str, Any], settings: Settings) -> Path:
    path = _record_path(record, settings)
    return verify_artifact_record(path, record, settings)


def _verify_eda_outputs(summary: dict[str, Any], settings: Settings) -> None:
    for name, path in {
        "base": settings.config_path,
        "eda": settings.eda_config_path,
        "features": settings.features_config_path,
    }.items():
        verify_artifact_record(path, summary["configuration"][name], settings)
    for record in summary["input_processed"].values():
        _verify_record(record, settings)
    _verify_record(summary["environment_artifact"], settings)
    _verify_record(summary["context_aggregates"]["artifact"], settings)
    _verify_record(summary["correlation"]["correlation_artifact"], settings)
    _verify_record(summary["correlation"]["covariance_artifact"], settings)
    _verify_record(summary["report_artifact"], settings)
    for record in summary["figures"].values():
        _verify_record(record, settings)


def load_verified_modeling_inputs(settings: Settings) -> VerifiedModelingInputs:
    """Verify the complete Step 09 and feature lineage before split creation."""

    processed = load_verified_processed(settings)
    eda_path = (
        settings.artifacts_dir / "reports" / settings.eda_version / "summary.json"
    )
    if not eda_path.is_file():
        run_eda(settings)
    eda = _read_json(eda_path)
    if eda.get("contract_version") != "eda_report_v1":
        raise RuntimeError("Unsupported EDA input contract")
    if eda.get("eda_version") != settings.eda_version:
        raise RuntimeError("EDA input version does not match configuration")
    _verify_eda_outputs(eda, settings)
    verify_artifact_record(
        processed.path,
        eda["input_processed"]["dataset"],
        settings,
    )
    verify_artifact_record(
        processed.manifest_path,
        eda["input_processed"]["manifest"],
        settings,
    )
    verify_artifact_record(
        processed.summary_path,
        eda["input_processed"]["summary"],
        settings,
    )

    feature_manifest_path = (
        settings.artifacts_dir
        / "features"
        / settings.feature_version
        / "manifest.json"
    )
    verify_artifact_record(
        feature_manifest_path,
        processed.manifest["input_manifest"],
        settings,
    )
    feature_manifest = _read_json(feature_manifest_path)
    features = feature_manifest.get("features", [])
    if len(features) != int(processed.manifest["model_candidate_feature_count"]):
        raise RuntimeError("Feature manifest count does not match processed lineage")
    if len({str(record["name"]) for record in features}) != len(features):
        raise RuntimeError("Feature manifest contains duplicate names")
    if any(record.get("leakage_status") != "known_at_origin" for record in features):
        raise RuntimeError("Every model input must be known at prediction origin")
    return VerifiedModelingInputs(
        processed=processed,
        eda_path=eda_path,
        eda=eda,
        feature_manifest_path=feature_manifest_path,
        feature_manifest=feature_manifest,
    )


def ensure_partition_access(
    partition: str,
    purpose: str | None,
    config: dict[str, Any],
) -> str:
    """Reject test access unless final evaluation is requested explicitly."""

    if partition not in PARTITIONS:
        raise ValueError(f"Unknown modelling partition: {partition}")
    selected_purpose = purpose or str(config["access"]["default_purpose"])
    allowed = {
        str(config["access"]["default_purpose"]),
        str(config["access"]["final_evaluation_purpose"]),
    }
    if selected_purpose not in allowed:
        raise ValueError(f"Unknown modelling access purpose: {selected_purpose}")
    if partition == "test" and selected_purpose != (
        config["access"]["final_evaluation_purpose"]
    ):
        raise PermissionError(
            "Test partition is sealed; use purpose='final_evaluation' only after "
            "model selection is frozen"
        )
    return selected_purpose


def load_verified_modeling_artifacts(
    settings: Settings,
) -> VerifiedModelingArtifacts:
    """Verify every persisted Step 10 artifact before modelling use."""

    config = load_model_config(settings)
    summary_path = (
        settings.artifacts_dir
        / "features"
        / settings.modelling_version
        / "summary.json"
    )
    if not summary_path.is_file():
        from flowcast.modelling.pipeline import run_modeling_prep

        run_modeling_prep(settings)
    summary = _read_json(summary_path)
    if summary.get("contract_version") != "split_preprocessing_v1":
        raise RuntimeError("Unsupported persisted modelling-data contract")
    if summary.get("version") != settings.modelling_version:
        raise RuntimeError("Persisted modelling-data version does not match settings")
    for name, path in {
        "base": settings.config_path,
        "models": settings.models_config_path,
        "features": settings.features_config_path,
    }.items():
        verify_artifact_record(path, summary["configuration"][name], settings)
    _verify_record(summary["input_eda_summary"], settings)
    for record in summary["input_processed"].values():
        _verify_record(record, settings)
    assignments_path = _verify_record(summary["artifacts"]["assignments"], settings)
    schema_path = _verify_record(summary["artifacts"]["feature_schema"], settings)
    folds_path = _verify_record(summary["artifacts"]["cv_folds"], settings)
    for family in MODEL_FAMILIES:
        _verify_record(summary["artifacts"]["preprocessors"][family], settings)
    assignments = pd.read_parquet(assignments_path)
    schema = _read_json(schema_path)
    folds = _read_json(folds_path)
    if len(assignments) != int(summary["split"]["total_rows"]):
        raise RuntimeError("Split assignment row count changed")
    if assignments.duplicated(["road_id", "timestamp"]).any():
        raise RuntimeError("Split assignments contain duplicate keys")
    if set(assignments["split"].unique()) != set(PARTITIONS):
        raise RuntimeError("Split assignments omit a required partition")
    if schema.get("feature_count") != summary["preprocessing"]["feature_count"]:
        raise RuntimeError("Persisted feature schema count changed")
    if folds.get("fold_count") != int(config["cross_validation"]["fold_count"]):
        raise RuntimeError("Persisted modelling schemas do not match Step 10")
    return VerifiedModelingArtifacts(
        summary_path=summary_path,
        summary=summary,
        assignments_path=assignments_path,
        assignments=assignments,
        schema_path=schema_path,
        schema=schema,
        folds_path=folds_path,
        folds=folds,
    )


def load_modeling_partition(
    settings: Settings,
    partition: str,
    *,
    purpose: str | None = None,
) -> pd.DataFrame:
    """Load a verified partition while keeping test sealed by default."""

    config = load_model_config(settings)
    ensure_partition_access(partition, purpose, config)
    artifacts = load_verified_modeling_artifacts(settings)
    processed = load_verified_processed(settings)
    keys = ["road_id", "timestamp"]
    if not processed.frame[keys].equals(artifacts.assignments[keys]):
        raise RuntimeError("Split assignments no longer align with processed origins")
    selected = artifacts.assignments["split"].eq(partition)
    return pd.concat(
        [
            processed.frame.loc[selected].reset_index(drop=True),
            artifacts.assignments.loc[
                selected,
                [name for name in artifacts.assignments if name not in keys],
            ].reset_index(drop=True),
        ],
        axis=1,
    )


def load_preprocessor(settings: Settings, family: str) -> Any:
    """Load one verified fitted preprocessor for an approved model family."""

    if family not in MODEL_FAMILIES:
        raise ValueError(f"Unknown preprocessing family: {family}")
    artifacts = load_verified_modeling_artifacts(settings)
    record = artifacts.summary["artifacts"]["preprocessors"][family]
    return joblib.load(_record_path(record, settings))
