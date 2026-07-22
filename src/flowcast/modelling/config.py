"""Load and validate the frozen Step 10 modelling-data contract."""

from __future__ import annotations

from math import floor, isclose
from typing import Any, Mapping

import pandas as pd
import yaml

from flowcast.settings import Settings


PARTITIONS = ("train", "validation", "test")
MODEL_FAMILIES = ("linear", "tree", "svm", "recurrent")
_SCALING = {"none", "standard", "minmax"}


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
    return config
