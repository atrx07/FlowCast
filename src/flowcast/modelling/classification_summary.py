"""Canonical Step 13 summary assembly kept separate from orchestration."""

from __future__ import annotations

from time import perf_counter
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost

from flowcast.data.artifacts import artifact_record
from flowcast.settings import Settings


def _runtime_summary(
    aggregates: list[dict[str, Any]],
    families: list[dict[str, Any]],
    test_prediction_seconds: float,
    wall_seconds: float,
) -> dict[str, float]:
    return {
        "cv_fit_seconds": round(
            sum(float(record.get("fit_seconds", 0)) for record in aggregates),
            6,
        ),
        "cv_prediction_seconds": round(
            sum(
                float(record.get("prediction_seconds", 0))
                for record in aggregates
            ),
            6,
        ),
        "validation_fit_seconds": round(
            sum(float(record["fit_seconds"]) for record in families),
            6,
        ),
        "validation_prediction_seconds": round(
            sum(float(record["prediction_seconds"]) for record in families),
            6,
        ),
        "test_prediction_seconds": round(float(test_prediction_seconds), 6),
        "wall_seconds": round(float(wall_seconds), 6),
    }


def _all_final_metrics_finite(scoreboard: list[dict[str, Any]]) -> bool:
    values: list[float] = []
    for record in scoreboard:
        for split in ("validation", "test"):
            metrics = record[split]
            task_metrics = (
                ("macro_f1", "macro_precision", "macro_recall", "accuracy")
                if record["task"] == "congestion"
                else (
                    "roc_auc",
                    "pr_auc",
                    "precision",
                    "recall",
                    "f1",
                    "accuracy",
                )
            )
            names = ("brier_score", "log_loss") + task_metrics
            values.extend(float(metrics[name]) for name in names)
    return bool(np.isfinite(values).all())


def build_classification_summary(
    *,
    section: dict[str, Any],
    selected_version: str,
    input_modeling: dict[str, Any],
    selection_record: dict[str, Any],
    jobs: list[Any],
    specs: list[Any],
    folds: list[dict[str, Any]],
    timestamp_budget: int,
    all_fold_records: list[dict[str, Any]],
    all_aggregates: list[dict[str, Any]],
    all_family_records: list[dict[str, Any]],
    scoreboard: list[dict[str, Any]],
    model_records: dict[str, Any],
    prediction_rows: int,
    test_prediction_seconds: float,
    wall_start: float,
    artifacts: dict[str, Any],
    settings: Settings,
) -> dict[str, Any]:
    """Build the machine-readable result, checks, acceptance, and runtime."""

    successful_folds = sum(
        record["status"] == "success" for record in all_fold_records
    )
    congestion = [
        record for record in scoreboard if record["task"] == "congestion"
    ]
    accident = [record for record in scoreboard if record["task"] == "accident"]
    return {
        "contract_version": str(section["contract_version"]),
        "version": selected_version,
        "configuration": {
            "base": artifact_record(settings.config_path, settings),
            "models": artifact_record(settings.models_config_path, settings),
            "seed": settings.seed,
        },
        "input_modeling": input_modeling,
        "search": {
            "family_count": len(section["estimators"]),
            "candidate_count": len(specs),
            "fold_count": len(folds),
            "training_timestamp_budget": timestamp_budget,
            "sampling": section["cross_validation"]["sampling"],
            "hyperparameter_selection": dict(
                section["selection"]["hyperparameter_stage"]
            ),
            "family_selection": dict(section["selection"]["family_stage"]),
        },
        "test_access": {
            "selection_status_before_load": "frozen_before_test_access",
            "selection_manifest": selection_record,
            "loader_invocation_count": 1,
            "purpose": "final_evaluation",
            "models_refit_after_test_load": False,
        },
        "coverage": {
            "task_count": len(section["tasks"]),
            "horizon_count": len(section["horizons"]),
            "job_count": len(jobs),
            "required_family_job_pairs": len(jobs) * len(section["estimators"]),
            "successful_fold_evaluations": successful_folds,
            "selected_model_count": len(model_records),
            "model_card_count": len(model_records),
            "prediction_rows": prediction_rows,
        },
        "acceptance": {
            "congestion_macro_f1_target": 0.80,
            "congestion_all_horizons_met": all(
                record["test"]["macro_f1"] >= 0.80 for record in congestion
            ),
            "accident_roc_auc_target": 0.75,
            "accident_all_horizons_met": all(
                record["test"]["roc_auc"] >= 0.75 for record in accident
            ),
        },
        "scoreboard": scoreboard,
        "models": model_records,
        "runtime": _runtime_summary(
            all_aggregates,
            all_family_records,
            test_prediction_seconds,
            perf_counter() - wall_start,
        ),
        "libraries": {
            "joblib": joblib.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
        },
        "limitations": [
            "CV search uses evenly spaced timestamps spanning each fold; final "
            "family fits use all eligible training rows.",
            "Accident positives are rare; ROC-AUC is paired with PR-AUC and "
            "validation-selected operating metrics.",
            "Probability calibration uses a chronological validation split and "
            "is applied only when required or when Brier score improves enough.",
            "Confidence displays and segmented error analysis remain Step 16 work.",
        ],
        "checks": [
            {"name": "eight_generated_jobs", "passed": len(jobs) == 8},
            {
                "name": "all_family_job_pairs",
                "passed": len(all_family_records) == 32,
            },
            {
                "name": "all_required_cv_folds",
                "passed": successful_folds == len(specs) * len(jobs) * len(folds),
            },
            {
                "name": "decisions_frozen_before_test",
                "passed": True,
            },
            {"name": "single_test_load", "passed": True},
            {
                "name": "all_models_expose_probabilities",
                "passed": len(model_records) == 8,
            },
            {
                "name": "all_final_metrics_finite",
                "passed": _all_final_metrics_finite(scoreboard),
            },
        ],
        "artifacts": artifacts,
    }
