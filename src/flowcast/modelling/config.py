"""Load and validate the frozen Step 10 modelling-data contract."""

from __future__ import annotations

from math import floor, isclose
from typing import Any, Mapping

import pandas as pd
import yaml

from flowcast.settings import Settings
from flowcast.modelling.classification_config import (
    validate_classical_classification,
)


PARTITIONS = ("train", "validation", "test")
MODEL_FAMILIES = ("linear", "tree", "svm", "recurrent")
_SCALING = {"none", "standard", "minmax"}
REGRESSION_TARGETS = ("volume", "speed", "travel_time")
REGRESSION_ESTIMATORS = (
    "linear_regression",
    "decision_tree",
    "random_forest",
    "xgboost",
)


def allocate_largest_remainder(
    total: int,
    ratios: Mapping[str, float],
) -> dict[str, int]:
    """Allocate an integer total by ratios without losing any observations."""

    if total <= 0:
        raise ValueError("Split allocation total must be positive")
    exact = {name: total * float(ratio) for name, ratio in ratios.items()}
    allocated = {name: floor(value) for name, value in exact.items()}
    remaining = total - sum(allocated.values())
    order = sorted(
        ratios,
        key=lambda name: (-(exact[name] - allocated[name]), list(ratios).index(name)),
    )
    for name in order[:remaining]:
        allocated[name] += 1
    return allocated


def _unique_names(values: list[Any], label: str) -> list[str]:
    names = [str(value) for value in values]
    if len(names) != len(set(names)):
        raise ValueError(f"Modelling {label} must contain unique names")
    return names


def _validate_split(config: dict[str, Any]) -> None:
    split = config.get("split", {})
    if int(split.get("cadence_minutes", 0)) != 30:
        raise ValueError("Modelling split cadence must be 30 minutes")
    if split.get("allocation_method") != "largest_remainder":
        raise ValueError("Modelling split must use largest-remainder allocation")
    ratios = split.get("ratios", {})
    if tuple(ratios) != PARTITIONS:
        raise ValueError("Modelling split ratios must be train/validation/test")
    numeric_ratios = {name: float(value) for name, value in ratios.items()}
    if any(value <= 0 for value in numeric_ratios.values()) or not isclose(
        sum(numeric_ratios.values()), 1.0, abs_tol=1e-12
    ):
        raise ValueError("Modelling split ratios must be positive and sum to one")
    partitions = split.get("partitions", {})
    if tuple(partitions) != PARTITIONS:
        raise ValueError("Modelling split boundaries must be train/validation/test")

    cadence = pd.Timedelta(minutes=30)
    total = 0
    previous_end: pd.Timestamp | None = None
    for name in PARTITIONS:
        record = partitions[name]
        count = int(record.get("timestamp_count", 0))
        start = pd.Timestamp(record.get("start"))
        end = pd.Timestamp(record.get("end"))
        if count <= 0 or start.tzinfo is None or end.tzinfo is None or start > end:
            raise ValueError(f"Invalid configured {name} split boundary")
        actual_count = int((end - start) / cadence) + 1
        if actual_count != count:
            raise ValueError(f"Configured {name} timestamp count is inconsistent")
        if previous_end is not None and start - previous_end != cadence:
            raise ValueError("Configured split boundaries must be contiguous")
        total += count
        previous_end = end
    if allocate_largest_remainder(total, numeric_ratios) != {
        name: int(partitions[name]["timestamp_count"]) for name in PARTITIONS
    }:
        raise ValueError("Configured split counts do not match their ratios")
    if split.get("target_boundary_policy") != (
        "target_timestamp_within_origin_partition"
    ):
        raise ValueError("Unsupported target-boundary policy")


def _validate_cv(config: dict[str, Any]) -> None:
    cv = config.get("cross_validation", {})
    if cv.get("strategy") != "expanding_window":
        raise ValueError("Time-series CV must use an expanding window")
    fold_count = int(cv.get("fold_count", 0))
    validation_windows = int(cv.get("validation_windows", 0))
    gap_windows = int(cv.get("gap_windows", 0))
    train_windows = int(
        config["split"]["partitions"]["train"]["timestamp_count"]
    )
    if fold_count < 2 or validation_windows <= 0:
        raise ValueError("Time-series CV needs at least two non-empty folds")
    if gap_windows < 4:
        raise ValueError("Time-series CV gap must cover the maximum horizon")
    if fold_count * validation_windows + gap_windows >= train_windows:
        raise ValueError("Time-series CV leaves no initial training window")


def _validate_preprocessing(config: dict[str, Any]) -> None:
    preprocessing = config.get("preprocessing", {})
    if preprocessing.get("categorical_encoding") != "one_hot_ignore_unknown":
        raise ValueError("Unsupported categorical encoding policy")
    if preprocessing.get("numeric_imputation") != "median":
        raise ValueError("Numeric preprocessing must use median imputation")
    if preprocessing.get("categorical_imputation") != "most_frequent":
        raise ValueError("Categorical preprocessing must use most-frequent fill")
    if preprocessing.get("binary_imputation") != "most_frequent":
        raise ValueError("Binary preprocessing must use most-frequent fill")
    binary = set(
        _unique_names(
            preprocessing.get("explicit_binary_features", []),
            "explicit binary features",
        )
    )
    bounded = set(
        _unique_names(
            preprocessing.get("bounded_numeric_features", []),
            "bounded numeric features",
        )
    )
    if binary & bounded:
        raise ValueError("Binary and bounded-numeric feature lists must be disjoint")
    families = preprocessing.get("families", {})
    if tuple(families) != MODEL_FAMILIES:
        raise ValueError("Preprocessing families do not match the model contract")
    for name, record in families.items():
        numeric = str(record.get("numeric_scaling"))
        bounded_scaling = str(record.get("bounded_scaling"))
        if numeric not in _SCALING or bounded_scaling not in _SCALING:
            raise ValueError(f"Unsupported scaling policy for {name}")
        if name == "recurrent" and bounded_scaling != "minmax":
            raise ValueError("Recurrent bounded features must use Min-Max scaling")
        if name in {"linear", "svm", "recurrent"} and numeric != "standard":
            raise ValueError(f"{name} numeric features must be standardized")
        if name == "tree" and (numeric != "none" or bounded_scaling != "none"):
            raise ValueError("Tree preprocessing must not scale numeric features")


def _positive_number(record: Mapping[str, Any], name: str) -> float:
    value = float(record.get(name, 0))
    if value <= 0:
        raise ValueError(f"Scratch linear {name} must be positive")
    return value


def _validate_optimizer(record: Mapping[str, Any], label: str) -> None:
    _positive_number(record, "learning_rate")
    _positive_number(record, "tolerance")
    _positive_number(record, "initialization_scale")
    if int(record.get("max_iterations", 0)) <= 0:
        raise ValueError(f"Scratch linear {label} iterations must be positive")
    if int(record.get("patience", 0)) <= 0:
        raise ValueError(f"Scratch linear {label} patience must be positive")


def _validate_scratch_linear(config: dict[str, Any]) -> None:
    scratch = config.get("scratch_linear", {})
    if scratch.get("contract_version") != "scratch_linear_v1":
        raise ValueError("Unsupported scratch-linear contract version")
    if scratch.get("version") != "scratch_linear_v1":
        raise ValueError("Unsupported scratch-linear artifact version")
    if scratch.get("target") != "target_volume_h1":
        raise ValueError("Step 11 must demonstrate next-window volume regression")
    if scratch.get("availability_column") != "target_volume_h1_available":
        raise ValueError("Scratch-linear availability column is inconsistent")
    if scratch.get("horizon_within_split_column") != (
        "target_within_split_h1"
    ):
        raise ValueError("Scratch-linear horizon boundary column is inconsistent")
    subset = scratch.get("training_subset", {})
    if subset.get("method") != "earliest_chronological":
        raise ValueError("Scratch-linear subset must be earliest chronological")
    if int(subset.get("row_limit", 0)) <= 0:
        raise ValueError("Scratch-linear training row limit must be positive")
    _validate_optimizer(scratch.get("optimizer", {}), "optimizer")
    check = scratch.get("gradient_check", {})
    for name in ("epsilon", "absolute_tolerance", "relative_tolerance"):
        _positive_number(check, name)
    proof = scratch.get("synthetic_proof", {})
    _validate_optimizer(proof, "synthetic proof")
    if int(proof.get("rows", 0)) <= int(proof.get("features", 0)):
        raise ValueError("Synthetic proof needs more rows than features")
    if int(proof.get("features", 0)) <= 0:
        raise ValueError("Synthetic proof feature count must be positive")
    _positive_number(proof, "coefficient_tolerance")


def _validate_classical_regression(config: dict[str, Any]) -> None:
    regression = config.get("classical_regression", {})
    if regression.get("contract_version") != "classical_regression_v1":
        raise ValueError("Unsupported classical-regression contract version")
    if regression.get("version") != "classical_regression_v1":
        raise ValueError("Unsupported classical-regression artifact version")
    if tuple(regression.get("targets", [])) != REGRESSION_TARGETS:
        raise ValueError(
            "Classical regression must cover volume, speed, and travel time"
        )
    if tuple(int(value) for value in regression.get("horizons", [])) != (
        1,
        2,
        3,
        4,
    ):
        raise ValueError("Classical regression horizons must be 1, 2, 3, and 4")
    if regression.get("primary_metric") != "rmse":
        raise ValueError("Classical regression must select by RMSE")
    cv = regression.get("cross_validation", {})
    if int(cv.get("fold_count", 0)) != int(
        config["cross_validation"]["fold_count"]
    ):
        raise ValueError("Classical regression must use every frozen CV fold")
    if int(cv.get("training_timestamp_budget", 0)) <= 0:
        raise ValueError("Classical-regression CV budget must be positive")
    if cv.get("sampling") != "evenly_spaced_timestamps":
        raise ValueError("Unsupported classical-regression CV sampling policy")
    if cv.get("require_all_folds") is not True:
        raise ValueError("Every classical-regression candidate must use all folds")
    selection = regression.get("selection", {})
    if selection.get("hyperparameter_stage") != "mean_cv_rmse_within_family":
        raise ValueError("Regression hyperparameters must be selected by mean CV RMSE")
    if selection.get("family_stage") != "validation_rmse":
        raise ValueError("Regression family selection must use validation RMSE")
    if selection.get("freeze_before_test") is not True:
        raise ValueError("Regression selection must be frozen before test access")
    if tuple(selection.get("tie_breakers", [])) != (
        "mean_cv_rmse",
        "family",
        "candidate_id",
    ):
        raise ValueError("Classical-regression tie breakers changed")

    estimators = regression.get("estimators", {})
    if tuple(estimators) != REGRESSION_ESTIMATORS:
        raise ValueError("Classical-regression estimator coverage changed")
    expected_preprocessing = {
        "linear_regression": "linear",
        "decision_tree": "tree",
        "random_forest": "tree",
        "xgboost": "tree",
    }
    candidate_ids: set[str] = set()
    for family, record in estimators.items():
        if record.get("preprocessing_family") != expected_preprocessing[family]:
            raise ValueError(f"Unexpected preprocessing family for {family}")
        candidates = record.get("candidates", [])
        if not 1 <= len(candidates) <= 2:
            raise ValueError(
                f"{family} needs one or two bounded candidate configurations"
            )
        for candidate in candidates:
            candidate_id = str(candidate.get("candidate_id", ""))
            if not candidate_id or candidate_id in candidate_ids:
                raise ValueError("Regression candidate IDs must be unique")
            candidate_ids.add(candidate_id)
            if not isinstance(candidate.get("parameters"), dict):
                raise ValueError("Regression candidate parameters must be mappings")


def load_model_config(settings: Settings) -> dict[str, Any]:
    """Load and fail closed on an invalid Step 10 modelling configuration."""

    with settings.models_config_path.open("r", encoding="utf-8") as handle:
        config: dict[str, Any] = yaml.safe_load(handle)
    if config.get("model_contract_version") != "split_preprocessing_v1":
        raise ValueError("Unsupported modelling-data contract version")
    if config.get("version") != settings.modelling_version:
        raise ValueError("Modelling config version does not match base settings")
    access = config.get("access", {})
    if access.get("default_purpose") != "tuning":
        raise ValueError("Default modelling access must remain tuning-only")
    if access.get("final_evaluation_purpose") != "final_evaluation":
        raise ValueError("Final test access purpose is not explicit")
    _validate_split(config)
    _validate_cv(config)
    _validate_preprocessing(config)
    _validate_scratch_linear(config)
    _validate_classical_regression(config)
    validate_classical_classification(config)
    return config
