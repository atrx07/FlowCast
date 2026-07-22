"""Pure, deterministic transformations for Step 07 explanatory features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureDefinition:
    """Traceable definition for one model-candidate feature."""

    name: str
    group: str
    source_columns: tuple[str, ...]
    transform: str
    leakage_status: str = "known_at_origin"


@dataclass(frozen=True)
class FeatureEngineeringResult:
    """Engineered table and ordered feature definitions."""

    frame: pd.DataFrame
    definitions: tuple[FeatureDefinition, ...]


_REQUIRED_COLUMNS = {
    "road_id",
    "timestamp",
    "traffic_volume",
    "avg_speed",
    "occupancy",
    "signal_timing",
    "road_capacity",
    "share_2w",
    "share_car",
    "share_lcv",
    "share_hcv",
    "weather_condition",
    "temperature",
    "rainfall",
    "visibility",
    "calendar_date",
    "public_holiday",
    "event_flag",
    "roadwork_flag",
    "_inserted_window",
    "traffic_volume_original_missing",
    "traffic_volume_physical_invalid",
    "avg_speed_original_missing",
    "avg_speed_physical_invalid",
    "temperature_was_missing",
    "visibility_was_missing",
    "vehicle_shares_normalized",
}


_PASSTHROUGH = (
    ("traffic_volume", "traffic"),
    ("avg_speed", "traffic"),
    ("occupancy", "traffic"),
    ("signal_timing", "traffic"),
    ("road_capacity", "capacity"),
    ("temperature", "weather"),
    ("rainfall", "weather"),
    ("visibility", "weather"),
    ("share_2w", "vehicle_share"),
    ("share_car", "vehicle_share"),
    ("share_lcv", "vehicle_share"),
    ("share_hcv", "vehicle_share"),
    ("public_holiday", "calendar"),
    ("event_flag", "calendar"),
    ("roadwork_flag", "calendar"),
    ("_inserted_window", "lineage"),
    ("traffic_volume_original_missing", "lineage"),
    ("traffic_volume_physical_invalid", "lineage"),
    ("avg_speed_original_missing", "lineage"),
    ("avg_speed_physical_invalid", "lineage"),
    ("temperature_was_missing", "lineage"),
    ("visibility_was_missing", "lineage"),
    ("vehicle_shares_normalized", "lineage"),
)


def _definition(
    name: str,
    group: str,
    sources: tuple[str, ...],
    transform: str,
) -> FeatureDefinition:
    return FeatureDefinition(name, group, sources, transform)


def _event_distance_days(frame: pd.DataFrame) -> pd.Series:
    event_dates = (
        pd.to_datetime(frame.loc[frame["event_flag"].eq(1), "calendar_date"])
        .drop_duplicates()
        .sort_values()
        .to_numpy(dtype="datetime64[D]")
    )
    if len(event_dates) == 0:
        return pd.Series(pd.NA, index=frame.index, dtype="Int16")
    dates = pd.to_datetime(frame["calendar_date"]).to_numpy(dtype="datetime64[D]")
    distances = np.abs(dates[:, None] - event_dates[None, :]).astype("timedelta64[D]")
    return pd.Series(distances.astype(np.int64).min(axis=1), dtype="Int16")


def engineer_features(
    source: pd.DataFrame,
    config: dict[str, Any],
) -> FeatureEngineeringResult:
    """Return a sorted feature table using only information known by origin time."""

    missing = sorted(_REQUIRED_COLUMNS - set(source.columns))
    if missing:
        raise ValueError(f"Merged input is missing feature columns: {missing}")
    frame = source.sort_values(["road_id", "timestamp"], kind="mergesort").reset_index(
        drop=True
    )
    if frame.duplicated(["road_id", "timestamp"]).any():
        raise ValueError("Feature input contains duplicate road/timestamp keys")
    if not frame.groupby("road_id", sort=False)["timestamp"].apply(
        lambda values: values.is_monotonic_increasing
    ).all():
        raise ValueError("Feature input is not time ordered within road")

    definitions = [
        _definition(name, group, (name,), "trusted source value")
        for name, group in _PASSTHROUGH
    ]
    minute = frame["timestamp"].dt.hour * 60 + frame["timestamp"].dt.minute
    hour = minute / 60.0
    weekday = frame["timestamp"].dt.dayofweek
    frame["hour_of_day"] = pd.array(hour, dtype="Float64")
    frame["hour_sin"] = pd.array(np.sin(2 * np.pi * hour / 24.0), dtype="Float64")
    frame["hour_cos"] = pd.array(np.cos(2 * np.pi * hour / 24.0), dtype="Float64")
    frame["day_of_week"] = pd.array(weekday, dtype="Int8")
    frame["day_of_week_sin"] = pd.array(
        np.sin(2 * np.pi * weekday / 7.0), dtype="Float64"
    )
    frame["day_of_week_cos"] = pd.array(
        np.cos(2 * np.pi * weekday / 7.0), dtype="Float64"
    )
    frame["is_weekend"] = pd.array(weekday.ge(5), dtype="boolean")
    for name, transform in (
        ("hour_of_day", "local hour including half-hour fraction"),
        ("hour_sin", "sin(2*pi*hour_of_day/24)"),
        ("hour_cos", "cos(2*pi*hour_of_day/24)"),
        ("day_of_week", "Monday=0 through Sunday=6"),
        ("day_of_week_sin", "sin(2*pi*day_of_week/7)"),
        ("day_of_week_cos", "cos(2*pi*day_of_week/7)"),
        ("is_weekend", "day_of_week >= 5"),
    ):
        definitions.append(_definition(name, "temporal", ("timestamp",), transform))

    peak_columns: list[str] = []
    for period in config["temporal"]["peak_periods"]:
        start_parts = [int(value) for value in str(period["start"]).split(":")]
        end_parts = [int(value) for value in str(period["end"]).split(":")]
        start = start_parts[0] * 60 + start_parts[1]
        end = end_parts[0] * 60 + end_parts[1]
        column = f"is_{period['name']}_peak"
        frame[column] = pd.array(minute.ge(start) & minute.lt(end), dtype="boolean")
        peak_columns.append(column)
        definitions.append(
            _definition(
                column,
                "temporal",
                ("timestamp",),
                f"local minute in [{period['start']}, {period['end']})",
            )
        )
    frame["is_peak"] = frame[peak_columns].any(axis=1).astype("boolean")
    definitions.append(
        _definition("is_peak", "temporal", tuple(peak_columns), "any configured peak")
    )

    lag_windows = [int(value) for value in config["history"]["lag_windows"]]
    rolling_windows = [int(value) for value in config["history"]["rolling_windows"]]
    history_columns: list[str] = []
    for source_name, prefix in (("traffic_volume", "volume"), ("avg_speed", "speed")):
        grouped = frame.groupby("road_id", sort=False)[source_name]
        for window in lag_windows:
            column = f"{prefix}_lag_{window}"
            frame[column] = grouped.shift(window)
            history_columns.append(column)
            definitions.append(
                _definition(
                    column,
                    "history",
                    ("road_id", source_name),
                    f"within-road shift({window})",
                )
            )
        shifted = grouped.shift(1)
        for window in rolling_windows:
            for statistic in ("mean", "std"):
                column = f"{prefix}_rolling_{statistic}_{window}"
                rolling = shifted.groupby(frame["road_id"], sort=False).transform(
                    lambda values, size=window, stat=statistic: getattr(
                        values.rolling(size, min_periods=size), stat
                    )()
                )
                frame[column] = pd.array(rolling, dtype="Float64")
                history_columns.append(column)
                definitions.append(
                    _definition(
                        column,
                        "history",
                        ("road_id", source_name),
                        f"within-road shift(1) then rolling({window}).{statistic}()",
                    )
                )
    frame["history_available"] = frame[history_columns].notna().all(axis=1).astype(
        "boolean"
    )
    definitions.append(
        _definition(
            "history_available",
            "history",
            tuple(history_columns),
            "all configured lag and rolling features are non-null",
        )
    )

    windows_per_hour = int(config["capacity"]["windows_per_hour"])
    frame["half_hour_capacity"] = pd.array(
        frame["road_capacity"] / windows_per_hour, dtype="Float64"
    )
    frame["volume_capacity_ratio"] = pd.array(
        frame["traffic_volume"] / frame["half_hour_capacity"], dtype="Float64"
    )
    frame["capacity_headroom"] = pd.array(
        frame["half_hour_capacity"] - frame["traffic_volume"], dtype="Float64"
    )
    definitions.extend(
        [
            _definition(
                "half_hour_capacity",
                "capacity",
                ("road_capacity",),
                f"road_capacity / {windows_per_hour}",
            ),
            _definition(
                "volume_capacity_ratio",
                "capacity",
                ("traffic_volume", "half_hour_capacity"),
                "traffic_volume / half_hour_capacity",
            ),
            _definition(
                "capacity_headroom",
                "capacity",
                ("half_hour_capacity", "traffic_volume"),
                "half_hour_capacity - traffic_volume",
            ),
        ]
    )

    weather = config["weather"]
    rain_threshold = float(weather["rain_minimum_mm_exclusive"])
    frame["is_rain"] = pd.array(
        frame["rainfall"].gt(rain_threshold)
        | frame["weather_condition"].eq("Rain"),
        dtype="boolean",
    )
    visibility_threshold = float(weather["low_visibility_below_metres"])
    frame["is_low_visibility"] = pd.array(
        frame["visibility"].lt(visibility_threshold), dtype="boolean"
    )
    definitions.extend(
        [
            _definition(
                "is_rain",
                "weather",
                ("rainfall", "weather_condition"),
                f"rainfall > {rain_threshold} or condition == Rain",
            ),
            _definition(
                "is_low_visibility",
                "weather",
                ("visibility",),
                f"visibility < {visibility_threshold} metres",
            ),
        ]
    )
    for category in weather["categories"]:
        column = f"weather_is_{str(category).lower()}"
        frame[column] = pd.array(
            frame["weather_condition"].eq(category), dtype="boolean"
        )
        definitions.append(
            _definition(
                column,
                "weather",
                ("weather_condition",),
                f"weather_condition == {category}",
            )
        )
    boundaries = [float(value) for value in weather["temperature_boundaries_celsius"]]
    labels = [str(value) for value in weather["temperature_labels"]]
    frame["temperature_band"] = pd.cut(
        frame["temperature"].astype(float),
        bins=[-np.inf, *boundaries, np.inf],
        labels=labels,
        right=False,
    ).astype("string")
    definitions.append(
        _definition(
            "temperature_band",
            "weather",
            ("temperature",),
            f"left-closed bands at {boundaries} Celsius labelled {labels}",
        )
    )

    frame["holiday_peak"] = pd.array(
        frame["public_holiday"].eq(1) & frame["is_peak"], dtype="boolean"
    )
    frame["days_to_nearest_event"] = _event_distance_days(frame)
    proximity = int(config["calendar"]["event_proximity_days"])
    frame["event_within_proximity"] = pd.array(
        frame["days_to_nearest_event"].le(proximity).fillna(False), dtype="boolean"
    )
    definitions.extend(
        [
            _definition(
                "holiday_peak",
                "calendar",
                ("public_holiday", "is_peak"),
                "public_holiday == 1 and is_peak",
            ),
            _definition(
                "days_to_nearest_event",
                "calendar",
                ("calendar_date", "event_flag"),
                "absolute calendar days to nearest scheduled event date",
            ),
            _definition(
                "event_within_proximity",
                "calendar",
                ("days_to_nearest_event",),
                f"days_to_nearest_event <= {proximity}",
            ),
        ]
    )
    return FeatureEngineeringResult(frame=frame, definitions=tuple(definitions))
