"""Versioned paths and verified loading for Step 13 classifier artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from flowcast.data.artifacts import verify_artifact_record
from flowcast.settings import Settings


@dataclass(frozen=True)
class ClassificationPaths:
    """Canonical output paths for one classical-classification run."""

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
    calibration_path: Path
    thresholds_path: Path
    confusions_path: Path
    importance_path: Path
    predictions_path: Path


def classification_paths(
    settings: Settings,
    version: str,
) -> ClassificationPaths:
    """Return Step 13 paths without creating them."""

    metrics = settings.artifacts_dir / "metrics" / version
    models = settings.artifacts_dir / "models" / version
    predictions = settings.artifacts_dir / "predictions" / version
    cards = settings.artifacts_dir / "model_cards" / version
    return ClassificationPaths(
        version=version,
        metrics_dir=metrics,
        models_dir=models,
        predictions_dir=predictions,
        cards_dir=cards,
        summary_path=metrics / "summary.json",
        report_path=metrics / "summary.md",
        selection_path=metrics / "selection_manifest.json",
        cv_folds_path=metrics / "cv_fold_metrics.csv",
        cv_candidates_path=metrics / "cv_candidate_metrics.csv",
        family_validation_path=metrics / "family_validation_metrics.csv",
        scoreboard_path=metrics / "scoreboard.csv",
        calibration_path=metrics / "calibration_metrics.csv",
        thresholds_path=metrics / "accident_thresholds.csv",
        confusions_path=metrics / "confusion_matrices.csv",
        importance_path=metrics / "feature_importance.csv",
        predictions_path=predictions / "selected_predictions.parquet",
    )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required classification artifact is missing: {path}")
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return payload


def _record_path(record: dict[str, Any], settings: Settings) -> Path:
    path = Path(str(record["path"]))
    return path if path.is_absolute() else settings.root / path


def _verify_record(record: dict[str, Any], settings: Settings) -> Path:
    return verify_artifact_record(_record_path(record, settings), record, settings)


def load_classification_model(
    settings: Settings,
    task: str,
    horizon: int,
    *,
    version: str = "classical_classification_v1",
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """Verify the complete chain before loading one probability classifier."""

    paths = classification_paths(settings, version)
    summary = _read_json(paths.summary_path)
    if summary.get("contract_version") != "classical_classification_v1":
        raise RuntimeError("Unsupported classical-classification summary contract")
    if summary.get("version") != version:
        raise RuntimeError("Classical-classification summary version changed")
    for name, path in {
        "base": settings.config_path,
        "models": settings.models_config_path,
    }.items():
        verify_artifact_record(path, summary["configuration"][name], settings)
    for record in summary["input_modeling"].values():
        _verify_record(record, settings)
    for record in summary["artifacts"].values():
        _verify_record(record, settings)

    job_id = f"{task}_h{int(horizon)}"
    if job_id not in summary["models"]:
        raise KeyError(f"No selected classical classifier for {job_id}")
    model_entry = summary["models"][job_id]
    model_path = _verify_record(model_entry["model"], settings)
    card_path = _verify_record(model_entry["model_card_json"], settings)
    _verify_record(model_entry["model_card_markdown"], settings)
    card = _read_json(card_path)
    if card.get("job_id") != job_id:
        raise RuntimeError("Classification model-card job identity changed")
    if card["artifacts"]["model"] != model_entry["model"]:
        raise RuntimeError("Model card no longer identifies the selected classifier")
    estimator = joblib.load(model_path)
    if not hasattr(estimator, "predict_proba") or not hasattr(estimator, "classes_"):
        raise TypeError("Persisted classifier does not expose ordered probabilities")
    expected = np.arange(len(card["target"]["class_order"]), dtype=np.int64)
    if not np.array_equal(np.asarray(estimator.classes_, dtype=np.int64), expected):
        raise RuntimeError("Persisted classifier class order changed")
    return estimator, card, summary
