"""Validate the bounded Step 13 classical-classification contract."""

from __future__ import annotations

from typing import Any


CLASSIFICATION_TASKS = ("congestion", "accident")
CLASSIFICATION_ESTIMATORS = (
    "decision_tree",
    "random_forest",
    "xgboost",
    "svm",
)
CONGESTION_CLASSES = ("Free-flow", "Moderate", "Heavy", "Severe")
ACCIDENT_CLASSES = ("no_accident", "accident")


def validate_classical_classification(config: dict[str, Any]) -> None:
    """Fail closed when Step 13 scope, metrics, or sealing rules change."""

    section = config.get("classical_classification", {})
    if section.get("contract_version") != "classical_classification_v1":
        raise ValueError("Unsupported classical-classification contract version")
    if section.get("version") != "classical_classification_v1":
        raise ValueError("Unsupported classical-classification artifact version")
    if tuple(section.get("tasks", [])) != CLASSIFICATION_TASKS:
        raise ValueError("Classification must cover congestion and accident risk")
    if tuple(int(value) for value in section.get("horizons", [])) != (1, 2, 3, 4):
        raise ValueError("Classification horizons must be 1, 2, 3, and 4")
    order = section.get("class_order", {})
    if tuple(order.get("congestion", [])) != CONGESTION_CLASSES:
        raise ValueError("Congestion class order changed")
    if tuple(order.get("accident", [])) != ACCIDENT_CLASSES:
        raise ValueError("Accident class order changed")
    if section.get("primary_metrics") != {
        "congestion": "macro_f1",
        "accident": "roc_auc",
    }:
        raise ValueError("Classification primary metrics changed")

    cv = section.get("cross_validation", {})
    if int(cv.get("fold_count", 0)) != int(
        config["cross_validation"]["fold_count"]
    ):
        raise ValueError("Classification must use every frozen CV fold")
    if int(cv.get("training_timestamp_budget", 0)) <= 0:
        raise ValueError("Classification CV timestamp budget must be positive")
    if cv.get("sampling") != "evenly_spaced_timestamps":
        raise ValueError("Unsupported classification CV sampling policy")
    if cv.get("require_all_folds") is not True:
        raise ValueError("Every classification candidate must use all CV folds")

    selection = section.get("selection", {})
    expected_hyperparameters = {
        "congestion": "mean_cv_macro_f1_within_family",
        "accident": "mean_cv_roc_auc_within_family",
    }
    expected_families = {
        "congestion": "validation_macro_f1",
        "accident": "validation_roc_auc",
    }
    if selection.get("hyperparameter_stage") != expected_hyperparameters:
        raise ValueError("Classification CV selection rules changed")
    if selection.get("family_stage") != expected_families:
        raise ValueError("Classification validation selection rules changed")
    if tuple(selection.get("tie_breakers", [])) != ("family", "candidate_id"):
        raise ValueError("Classification tie breakers changed")
    if selection.get("freeze_before_test") is not True:
        raise ValueError("Classification decisions must freeze before test")

    calibration = section.get("probability", {}).get("calibration", {})
    if calibration.get("method") != "sigmoid":
        raise ValueError("Step 13 supports sigmoid calibration only")
    fraction = float(calibration.get("validation_fit_fraction", 0))
    if not 0.0 < fraction < 1.0:
        raise ValueError("Calibration fit fraction must lie strictly between 0 and 1")
    if calibration.get("assessment_metric") != "brier_score":
        raise ValueError("Calibration must be assessed by Brier score")
    if float(calibration.get("minimum_improvement", -1)) < 0:
        raise ValueError("Calibration improvement threshold cannot be negative")

    threshold = section.get("accident_threshold", {})
    if threshold.get("selection_metric") != "f1":
        raise ValueError("Accident threshold must be selected by validation F1")
    if int(threshold.get("candidate_quantiles", 0)) < 3:
        raise ValueError("Accident threshold search needs at least three quantiles")
    default_threshold = float(threshold.get("include_default_threshold", -1))
    if not 0.0 < default_threshold < 1.0:
        raise ValueError("Default accident threshold must lie between zero and one")
    if tuple(threshold.get("tie_breakers", [])) != (
        "recall",
        "precision",
        "lower_threshold",
    ):
        raise ValueError("Accident threshold tie breakers changed")

    estimators = section.get("estimators", {})
    if tuple(estimators) != CLASSIFICATION_ESTIMATORS:
        raise ValueError("Classical-classification estimator coverage changed")
    expected_preprocessing = {
        "decision_tree": "tree",
        "random_forest": "tree",
        "xgboost": "tree",
        "svm": "svm",
    }
    candidate_ids: set[str] = set()
    for family, record in estimators.items():
        if record.get("preprocessing_family") != expected_preprocessing[family]:
            raise ValueError(f"Unexpected preprocessing family for {family}")
        candidates = record.get("candidates", [])
        if not 1 <= len(candidates) <= 2:
            raise ValueError(f"{family} needs one or two bounded candidates")
        for candidate in candidates:
            candidate_id = str(candidate.get("candidate_id", ""))
            if not candidate_id or candidate_id in candidate_ids:
                raise ValueError("Classification candidate IDs must be unique")
            candidate_ids.add(candidate_id)
            if not isinstance(candidate.get("parameters"), dict):
                raise ValueError("Classification parameters must be mappings")
