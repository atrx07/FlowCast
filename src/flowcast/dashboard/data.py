"""Integrity-checked loaders for dashboard-ready FlowCast artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from flowcast.evaluation.confidence_artifacts import (
    VerifiedConfidenceArtifacts,
    confidence_paths,
    load_verified_confidence_artifacts,
)
from flowcast.inference.artifacts import (
    LoadedPredictionBatch,
    load_prediction_batch,
)
from flowcast.inference.config import load_inference_config
from flowcast.inference.inputs import (
    VerifiedInferenceContext,
    load_verified_inference_context,
)
from flowcast.reports import verify_prediction_reports
from flowcast.settings import Settings, load_settings


@dataclass(frozen=True)
class DashboardBundle:
    """Read-only, verified data and metadata used across all dashboard pages."""

    settings: Settings
    context: VerifiedInferenceContext
    batch: LoadedPredictionBatch
    report_manifest: dict[str, Any] | None
    confidence: VerifiedConfidenceArtifacts
    registry_scoreboard: pd.DataFrame
    recurrent_comparison: pd.DataFrame
    regression_importance: pd.DataFrame
    classification_importance: pd.DataFrame
    regression_coverage: pd.DataFrame
    classification_reliability: pd.DataFrame
    accident_risk_bands: pd.DataFrame
    error_slices: pd.DataFrame

    @property
    def history(self) -> pd.DataFrame:
        """Return the verified processed history table."""

        return self.context.processed.frame

    @property
    def predictions(self) -> pd.DataFrame:
        """Return the verified latest persisted forecast batch."""

        return self.batch.frame


def _latest_manifest(root: Path) -> Path:
    manifests = sorted(
        root.glob("*/manifest.json"),
        key=lambda path: (path.stat().st_mtime_ns, path.as_posix()),
    )
    if not manifests:
        raise FileNotFoundError(f"No prediction manifest exists under {root}")
    return manifests[-1]


def _report_for_batch(
    settings: Settings,
    batch: LoadedPredictionBatch,
) -> dict[str, Any] | None:
    path = (
        settings.artifacts_dir
        / "reports"
        / batch.paths.version
        / batch.paths.request_id
        / "manifest.json"
    )
    return verify_prediction_reports(settings, path) if path.is_file() else None


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Required dashboard artifact is missing: {path}")
    return pd.read_csv(path)


def load_dashboard_bundle(
    settings: Settings | None = None,
) -> DashboardBundle:
    """Recursively verify and load every canonical dashboard input."""

    resolved = settings or load_settings()
    context = load_verified_inference_context(resolved)
    prediction_root = (
        resolved.artifacts_dir
        / "predictions"
        / str(context.config["version"])
    )
    batch = load_prediction_batch(resolved, _latest_manifest(prediction_root))
    confidence_version = str(context.confidence.summary["version"])
    confidence = load_verified_confidence_artifacts(
        resolved,
        version=confidence_version,
    )
    paths = confidence_paths(resolved, confidence_version)
    registry_paths = context.registry_paths
    recurrent_version = str(context.config["upstream"]["recurrent_version"])
    recurrent_dir = resolved.artifacts_dir / "metrics" / recurrent_version
    regression_dir = (
        resolved.artifacts_dir / "metrics" / "classical_regression_v1"
    )
    classification_dir = (
        resolved.artifacts_dir / "metrics" / "classical_classification_v1"
    )
    return DashboardBundle(
        settings=resolved,
        context=context,
        batch=batch,
        report_manifest=_report_for_batch(resolved, batch),
        confidence=confidence,
        registry_scoreboard=_read_csv(registry_paths.scoreboard_path),
        recurrent_comparison=_read_csv(
            recurrent_dir / "classical_comparison.csv"
        ),
        regression_importance=_read_csv(
            regression_dir / "feature_importance.csv"
        ),
        classification_importance=_read_csv(
            classification_dir / "feature_importance.csv"
        ),
        regression_coverage=_read_csv(paths.regression_coverage_path),
        classification_reliability=_read_csv(paths.reliability_path),
        accident_risk_bands=_read_csv(paths.risk_bands_path),
        error_slices=_read_csv(paths.error_slices_path),
    )


def dashboard_fingerprint(settings: Settings | None = None) -> tuple[int, ...]:
    """Return a compact cache key from dashboard source artifact metadata."""

    resolved = settings or load_settings()
    inference_config, _ = load_inference_config(resolved)
    prediction_manifest = _latest_manifest(
        resolved.artifacts_dir
        / "predictions"
        / str(inference_config["version"])
    )
    report_manifest = (
        resolved.artifacts_dir
        / "reports"
        / str(inference_config["version"])
        / prediction_manifest.parent.name
        / "manifest.json"
    )
    files = [
        resolved.processed_dir
        / resolved.processed_version
        / "dataset.parquet",
        resolved.artifacts_dir
        / "metrics"
        / "classical_registry_v1"
        / "summary.json",
        resolved.artifacts_dir
        / "metrics"
        / "confidence_error_v1"
        / "summary.json",
        resolved.root / "config" / "inference.yaml",
        prediction_manifest,
    ]
    if report_manifest.is_file():
        files.append(report_manifest)
    values: list[int] = []
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(f"Dashboard source is missing: {path}")
        stat = path.stat()
        values.extend((stat.st_mtime_ns, stat.st_size))
    return tuple(values)
