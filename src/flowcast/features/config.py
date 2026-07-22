"""Load and validate the versioned explanatory-feature contract."""

from __future__ import annotations

import re
from datetime import time
from typing import Any

import yaml

from flowcast.settings import Settings


_SAFE_NAME = re.compile(r"^[a-z][a-z0-9_]*$")
_REQUIRED_TARGETS = {
    "volume": ("traffic_volume", "regression"),
    "speed": ("avg_speed", "regression"),
    "travel_time": ("travel_time", "regression"),
    "congestion": ("congestion_level", "classification_multiclass"),
    "accident": ("accident_count", "classification_binary"),
}


def _clock(value: str) -> time:
    try:
        return time.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"Invalid peak-period clock time: {value}") from exc


def load_feature_config(settings: Settings) -> dict[str, Any]:
    """Load and fail closed on an invalid Step 07 feature configuration."""

    with settings.features_config_path.open("r", encoding="utf-8") as handle:
        config: dict[str, Any] = yaml.safe_load(handle)
    if config.get("feature_contract_version") != "explanatory_features_v1":
        raise ValueError("Unsupported explanatory feature contract version")
    if config.get("version") != settings.feature_version:
        raise ValueError("Feature configuration version does not match base settings")
    if config.get("keys") != ["road_id", "timestamp"]:
        raise ValueError("Feature keys must be road_id + timestamp")

    horizons = [int(value) for value in config["forecast_horizons_reserved"]]
    if horizons != [1, 2, 3, 4]:
        raise ValueError("Reserved forecast horizons must be 1, 2, 3, and 4")
    lags = [int(value) for value in config["history"]["lag_windows"]]
    rolls = [int(value) for value in config["history"]["rolling_windows"]]
    if sorted(set(lags)) != [1, 2, 48] or any(value <= 0 for value in lags):
        raise ValueError("Lag windows must be the unique positive values 1, 2, 48")
    if sorted(set(rolls)) != [4, 8] or any(value <= 1 for value in rolls):
        raise ValueError("Rolling windows must be the unique values 4 and 8")

    names: set[str] = set()
    for period in config["temporal"]["peak_periods"]:
        name = str(period["name"])
        start = _clock(str(period["start"]))
        end = _clock(str(period["end"]))
        if not _SAFE_NAME.fullmatch(name) or name in names:
            raise ValueError(f"Invalid or duplicate peak-period name: {name}")
        if start >= end:
            raise ValueError(f"Peak period {name} must not cross midnight")
        names.add(name)

    weather = config["weather"]
    categories = [str(value) for value in weather["categories"]]
    if len(categories) != len(set(categories)) or not categories:
        raise ValueError("Weather categories must be unique and non-empty")
    boundaries = [
        float(value) for value in weather["temperature_boundaries_celsius"]
    ]
    labels = [str(value) for value in weather["temperature_labels"]]
    if boundaries != sorted(set(boundaries)) or len(labels) != len(boundaries) + 1:
        raise ValueError("Temperature boundaries and labels do not form valid bands")
    if float(weather["low_visibility_below_metres"]) <= 0:
        raise ValueError("Low-visibility threshold must be positive")
    if int(config["capacity"]["windows_per_hour"]) <= 0:
        raise ValueError("Capacity windows_per_hour must be positive")
    if int(config["calendar"]["event_proximity_days"]) < 0:
        raise ValueError("Event proximity must not be negative")
    targets = config["targets"]
    if targets.get("contract_version") != "multi_horizon_targets_v1":
        raise ValueError("Unsupported multi-horizon target contract version")
    if targets.get("processed_version") != settings.processed_version:
        raise ValueError("Processed target version does not match base settings")
    if int(targets.get("cadence_minutes", 0)) != 30:
        raise ValueError("Target cadence must be 30 minutes")
    definitions = targets.get("definitions", [])
    observed: dict[str, tuple[str, str]] = {}
    for definition in definitions:
        name = str(definition.get("name", ""))
        if not _SAFE_NAME.fullmatch(name) or name in observed:
            raise ValueError(f"Invalid or duplicate target name: {name}")
        observed[name] = (
            str(definition.get("source_column", "")),
            str(definition.get("task", "")),
        )
    if observed != _REQUIRED_TARGETS:
        raise ValueError("Target definitions do not match the required outputs")
    accident = next(item for item in definitions if item["name"] == "accident")
    if accident.get("availability_source") != "_accident_observed":
        raise ValueError("Accident targets must use observed-incident availability")
    return config
