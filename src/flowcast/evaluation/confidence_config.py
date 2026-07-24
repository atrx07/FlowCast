"""Independent configuration contract for confidence and error analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from flowcast.settings import Settings


CONFIDENCE_CONFIG_PATH = Path("config/confidence.yaml")
EXPECTED_HORIZONS = (1, 2, 3, 4)
EXPECTED_SPLITS = ("validation", "test")
ALLOWED_DIMENSIONS = {
    "road_id",
    "origin_hour",
    "weekday",
    "weekday_type",
    "peak_status",
    "weather_condition",
    "actual_congestion",
}


def confidence_config_path(settings: Settings) -> Path:
    """Return the Step 16 configuration path."""

    return settings.root / CONFIDENCE_CONFIG_PATH


def _positive_int(record: dict[str, Any], name: str) -> int:
    value = int(record[name])
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _validate(section: dict[str, Any]) -> None:
    if section.get("contract_version") != "confidence_error_v1":
        raise ValueError("Unsupported confidence-analysis contract")
    required_upstream = {
        "processed_version",
        "regression_version",
        "classification_version",
        "registry_version",
        "recurrent_version",
    }
    if set(section.get("upstream", {})) != required_upstream:
        raise ValueError("Confidence config must freeze every upstream version")

    regression = section.get("regression", {})
    if regression.get("method") != "split_conformal_absolute_residual":
        raise ValueError("Only split-conformal absolute residuals are supported")
    if regression.get("calibration_split") != "validation":
        raise ValueError("Confidence calibration must use validation only")
    if tuple(regression.get("application_splits", [])) != EXPECTED_SPLITS:
        raise ValueError("Intervals must be applied to validation and test")
    level = float(regression.get("confidence_level", 0.0))
    if not 0.0 < level < 1.0:
        raise ValueError("Regression confidence_level must lie in (0, 1)")
    if regression.get("quantile_method") != "finite_sample_higher":
        raise ValueError("Unsupported conformal quantile method")

    classification = section.get("classification", {})
    _positive_int(classification, "reliability_bins")
    tolerance = float(classification.get("probability_tolerance", 0.0))
    if not 0.0 < tolerance <= 1.0e-3:
        raise ValueError("Probability tolerance must be in (0, 1e-3]")
    bands = classification.get("confidence_bands", {})
    medium = float(bands.get("medium_minimum", 0.0))
    high = float(bands.get("high_minimum", 0.0))
    if not 0.0 < medium < high < 1.0:
        raise ValueError("Confidence band thresholds must be ordered within (0, 1)")

    risk = section.get("accident_risk", {})
    if risk.get("source") != "validation_selected_operating_threshold":
        raise ValueError("Accident risk bands must use frozen validation thresholds")
    multipliers = risk.get("threshold_multipliers", {})
    values = tuple(
        float(multipliers[name])
        for name in ("elevated", "high", "critical")
    )
    if not 0.0 < values[0] < values[1] < values[2]:
        raise ValueError("Accident risk multipliers must be strictly ordered")
    if values[1] != 1.0:
        raise ValueError("The high-risk boundary must equal the frozen threshold")

    slices = section.get("slices", {})
    dimensions = tuple(str(value) for value in slices.get("dimensions", []))
    if not dimensions or len(set(dimensions)) != len(dimensions):
        raise ValueError("Slice dimensions must be non-empty and unique")
    if not set(dimensions).issubset(ALLOWED_DIMENSIONS):
        raise ValueError("Confidence config contains an unsupported slice dimension")
    minimums = slices.get("minimum_rows", {})
    for task in ("regression", "congestion", "accident"):
        _positive_int(minimums, task)
    _positive_int(slices, "minimum_accident_positives")


def load_confidence_config(
    settings: Settings,
) -> tuple[dict[str, Any], Path]:
    """Load and validate the independent Step 16 YAML contract."""

    path = confidence_config_path(settings)
    if not path.is_file():
        raise FileNotFoundError(f"Confidence configuration is missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload: dict[str, Any] = yaml.safe_load(handle)
    section = payload.get("confidence_analysis")
    if not isinstance(section, dict):
        raise ValueError("confidence.yaml must define confidence_analysis")
    _validate(section)
    return section, path
