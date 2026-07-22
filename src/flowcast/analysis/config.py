"""Load and validate the versioned Step 09 EDA contract."""

from __future__ import annotations

from typing import Any

import yaml

from flowcast.settings import Settings


_REQUIRED_DESCRIPTIVE = {
    "traffic_volume",
    "avg_speed",
    "occupancy",
    "travel_time",
}
_REQUIRED_DIMENSIONS = {
    "road_id",
    "local_hour",
    "day_of_week",
    "weather_condition",
    "public_holiday",
    "event_flag",
    "roadwork_flag",
}


def _unique_strings(values: list[Any], label: str) -> list[str]:
    selected = [str(value) for value in values]
    if not selected or len(selected) != len(set(selected)):
        raise ValueError(f"EDA {label} must be unique and non-empty")
    return selected


def load_eda_config(settings: Settings) -> dict[str, Any]:
    """Load and fail closed on an invalid Step 09 EDA configuration."""

    with settings.eda_config_path.open("r", encoding="utf-8") as handle:
        config: dict[str, Any] = yaml.safe_load(handle)
    if config.get("eda_contract_version") != "eda_report_v1":
        raise ValueError("Unsupported EDA report contract version")
    if config.get("version") != settings.eda_version:
        raise ValueError("EDA configuration version does not match base settings")
    descriptive = _unique_strings(
        config.get("descriptive_columns", []), "descriptive columns"
    )
    if not _REQUIRED_DESCRIPTIVE.issubset(descriptive):
        raise ValueError("EDA descriptive columns omit a required traffic measure")
    dimensions = _unique_strings(
        config.get("context_dimensions", []), "context dimensions"
    )
    if set(dimensions) != _REQUIRED_DIMENSIONS:
        raise ValueError("EDA context dimensions do not match the required slices")
    _unique_strings(config.get("correlation_features", []), "correlation features")
    target = config.get("target_correlation", {})
    if target.get("target") != "target_volume_h1":
        raise ValueError("EDA target correlation must use target_volume_h1")
    if target.get("availability") != "target_volume_h1_available":
        raise ValueError("EDA target correlation availability is invalid")
    threshold = float(config.get("redundancy_absolute_correlation", 0.0))
    if not 0.0 < threshold <= 1.0:
        raise ValueError("EDA redundancy threshold must be in (0, 1]")
    if config.get("congestion_order") != [
        "Free-flow",
        "Moderate",
        "Heavy",
        "Severe",
    ]:
        raise ValueError("EDA congestion order is invalid")
    dpi = int(config.get("figures", {}).get("dpi", 0))
    if not 72 <= dpi <= 300:
        raise ValueError("EDA figure DPI must be between 72 and 300")
    return config
