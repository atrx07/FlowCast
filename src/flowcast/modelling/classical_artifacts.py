"""Paths and verified loading for persisted Step 12 regression artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import joblib
from sklearn.pipeline import Pipeline

from flowcast.data.artifacts import verify_artifact_record
from flowcast.settings import Settings


@dataclass(frozen=True)
class ClassicalRegressionPaths:
    """Versioned output locations for one classical-regression run."""

    version: str
    metrics_dir: Path
    models_dir: Path
    predictions_dir: Path
    cards_dir: Path
    summary_path: Path
    report_path: Path
    selection_path: Path
    cv_folds_path: Path
    cv_candidates_path: Path
    family_validation_path: Path
    scoreboard_path: Path
    importance_path: Path
    predictions_path: Path


def classical_regression_paths(
    settings: Settings,
    version: str,
) -> ClassicalRegressionPaths:
    """Return all canonical Step 12 paths without creating them."""

    metrics_dir = settings.artifacts_dir / "metrics" / version
    models_dir = settings.artifacts_dir / "models" / version
    predictions_dir = settings.artifacts_dir / "predictions" / version
    cards_dir = settings.artifacts_dir / "model_cards" / version
    return ClassicalRegressionPaths(
        version=version,
        metrics_dir=metrics_dir,
        models_dir=models_dir,
        predictions_dir=predictions_dir,
        cards_dir=cards_dir,
        summary_path=metrics_dir / "summary.json",
        report_path=metrics_dir / "summary.md",
        selection_path=metrics_dir / "selection_manifest.json",
        cv_folds_path=metrics_dir / "cv_fold_metrics.csv",
        cv_candidates_path=metrics_dir / "cv_candidate_metrics.csv",
        family_validation_path=metrics_dir / "family_validation_metrics.csv",
        scoreboard_path=metrics_dir / "scoreboard.csv",
        importance_path=metrics_dir / "feature_importance.csv",
        predictions_path=predictions_dir / "selected_predictions.parquet",
    )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required classical artifact is missing: {path}")
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return payload


def _record_path(record: dict[str, Any], settings: Settings) -> Path:
    path = Path(str(record["path"]))
    return path if path.is_absolute() else settings.root / path


def _verify_record(record: dict[str, Any], settings: Settings) -> Path:
    path = _record_path(record, settings)
    return verify_artifact_record(path, record, settings)


def load_classical_regression_model(
    settings: Settings,
    target: str,
    horizon: int,
    *,
    version: str = "classical_regression_v1",
) -> tuple[Pipeline, dict[str, Any], dict[str, Any]]:
    """Load one selected pipeline after verifying its complete artifact chain."""

    paths = classical_regression_paths(settings, version)
    summary = _read_json(paths.summary_path)
    if summary.get("contract_version") != "classical_regression_v1":
        raise RuntimeError("Unsupported classical-regression summary contract")
    if summary.get("version") != version:
        raise RuntimeError("Classical-regression summary version changed")
    for name, path in {
        "base": settings.config_path,
        "models": settings.models_config_path,
    }.items():
        verify_artifact_record(path, summary["configuration"][name], settings)
    for record in summary["input_modeling"].values():
        _verify_record(record, settings)
    for record in summary["artifacts"].values():
        _verify_record(record, settings)
    job_id = f"{target}_h{int(horizon)}"
    if job_id not in summary["models"]:
        raise KeyError(f"No selected classical regression model for {job_id}")
    model_entry = summary["models"][job_id]
    model_path = _verify_record(model_entry["model"], settings)
    card_path = _verify_record(model_entry["model_card_json"], settings)
    _verify_record(model_entry["model_card_markdown"], settings)
    card = _read_json(card_path)
    if card.get("job_id") != job_id:
        raise RuntimeError("Model-card job identity changed")
    if card["artifacts"]["model"] != model_entry["model"]:
        raise RuntimeError("Model card no longer identifies the selected pipeline")
    pipeline = joblib.load(model_path)
    if not isinstance(pipeline, Pipeline):
        raise TypeError("Persisted classical artifact is not a sklearn Pipeline")
    return pipeline, card, summary
