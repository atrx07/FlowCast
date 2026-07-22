"""Load and validate the versioned explanatory-feature contract."""

from __future__ import annotations

import re
from datetime import time
from typing import Any

import yaml

from flowcast.settings import Settings


_SAFE_NAME = re.compile(r"^[a-z][a-z0-9_]*$")


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
    return config
