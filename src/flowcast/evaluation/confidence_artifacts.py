"""Canonical paths and integrity-checked loading for Step 16 artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import pandas as pd

from flowcast.data.artifacts import verify_artifact_record
from flowcast.evaluation.confidence_config import load_confidence_config
from flowcast.evaluation.confidence_inputs import (
    load_verified_confidence_inputs,
)
from flowcast.settings import Settings


@dataclass(frozen=True)
class ConfidencePaths:
    """Canonical output locations for one confidence-analysis version."""

    version: str
    metrics_dir: Path
    predictions_dir: Path
    summary_path: Path
    report_path: Path
    interval_calibration_path: Path
    regression_coverage_path: Path
    reliability_path: Path
    risk_bands_path: Path
    error_slices_path: Path
    confusions_path: Path
    paired_slices_path: Path
    regression_predictions_path: Path
    classification_predictions_path: Path
    paired_predictions_path: Path


@dataclass(frozen=True)
class VerifiedConfidenceArtifacts:
    """Loaded dashboard-ready outputs with their verified summary."""

    summary: dict[str, Any]
    regression: pd.DataFrame
    classification: pd.DataFrame
    paired_volume: pd.DataFrame


@dataclass(frozen=True)
class VerifiedConfidenceCalibration:
    """Verified Step 16 summary, configuration, and conformal widths."""

    summary: dict[str, Any]
    config: dict[str, Any]
    interval_calibration: pd.DataFrame


def confidence_paths(settings: Settings, version: str) -> ConfidencePaths:
    """Return Step 16 paths without creating directories."""

    metrics = settings.artifacts_dir / "metrics" / version
    predictions = settings.artifacts_dir / "predictions" / version
    return ConfidencePaths(
        version=version,
        metrics_dir=metrics,
        predictions_dir=predictions,
        summary_path=metrics / "summary.json",
        report_path=metrics / "summary.md",
        interval_calibration_path=metrics / "interval_calibration.csv",
        regression_coverage_path=metrics / "regression_coverage.csv",
        reliability_path=metrics / "classification_reliability.csv",
        risk_bands_path=metrics / "accident_risk_bands.csv",
        error_slices_path=metrics / "error_slices.csv",
        confusions_path=metrics / "confusion_matrices.csv",
        paired_slices_path=metrics / "paired_volume_slices.csv",
        regression_predictions_path=predictions / "regression_confidence.parquet",
        classification_predictions_path=predictions
        / "classification_confidence.parquet",
        paired_predictions_path=predictions / "paired_volume_comparison.parquet",
    )


def read_json(path: Path) -> dict[str, Any]:
    """Read a required JSON mapping."""

    if not path.is_file():
        raise FileNotFoundError(f"Required confidence artifact is missing: {path}")
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return payload


def record_path(record: dict[str, Any], settings: Settings) -> Path:
    """Resolve a portable artifact record against the repository root."""

    path = Path(str(record["path"]))
    return path if path.is_absolute() else settings.root / path


def verify_record(record: dict[str, Any], settings: Settings) -> Path:
    """Verify one portable artifact record and return its path."""

    return verify_artifact_record(record_path(record, settings), record, settings)


def _load_verified_summary(
    settings: Settings,
    version: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    paths = confidence_paths(settings, version)
    summary = read_json(paths.summary_path)
    if summary.get("contract_version") != "confidence_error_v1":
        raise RuntimeError("Unsupported confidence-analysis summary contract")
    if summary.get("version") != version:
        raise RuntimeError("Confidence-analysis summary version changed")
    config, config_path = load_confidence_config(settings)
    verify_artifact_record(config_path, summary["configuration"], settings)
    inputs = load_verified_confidence_inputs(
        settings,
        config,
        load_frames=False,
    )
    if inputs.upstream_records != summary["upstream"]:
        raise RuntimeError("Confidence upstream lineage changed")
    for record in summary["artifacts"].values():
        verify_record(record, settings)
    return summary, config


def load_verified_confidence_calibration(
    settings: Settings,
    *,
    version: str = "confidence_error_v1",
) -> VerifiedConfidenceCalibration:
    """Verify the full Step 16 chain while loading only inference-time widths."""

    paths = confidence_paths(settings, version)
    summary, config = _load_verified_summary(settings, version)
    calibration = pd.read_csv(paths.interval_calibration_path)
    expected_groups = int(summary["coverage"]["conformal_group_count"])
    if len(calibration) != expected_groups:
        raise RuntimeError("Confidence calibration group count changed")
    keys = ["model_version", "target", "horizon_windows"]
    if calibration.duplicated(keys).any():
        raise RuntimeError("Confidence calibration keys are no longer unique")
    if set(calibration["calibration_split"]) != {"validation"}:
        raise RuntimeError("Confidence calibration must remain validation-only")
    return VerifiedConfidenceCalibration(summary, config, calibration)


def load_verified_confidence_artifacts(
    settings: Settings,
    *,
    version: str = "confidence_error_v1",
) -> VerifiedConfidenceArtifacts:
    """Verify upstream lineage and load the three dashboard-ready tables."""

    paths = confidence_paths(settings, version)
    summary, _ = _load_verified_summary(settings, version)

    regression = pd.read_parquet(paths.regression_predictions_path)
    classification = pd.read_parquet(paths.classification_predictions_path)
    paired = pd.read_parquet(paths.paired_predictions_path)
    expected = summary["coverage"]
    actual = {
        "regression_prediction_rows": len(regression),
        "classification_prediction_rows": len(classification),
        "paired_volume_rows": len(paired),
    }
    for name, rows in actual.items():
        if rows != int(expected[name]):
            raise RuntimeError(f"Confidence artifact row count changed: {name}")
    if set(regression["split"]) != {"validation", "test"}:
        raise RuntimeError("Regression confidence splits changed")
    if set(classification["split"]) != {"validation", "test"}:
        raise RuntimeError("Classification confidence splits changed")
    return VerifiedConfidenceArtifacts(summary, regression, classification, paired)
