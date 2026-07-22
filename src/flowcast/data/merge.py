"""Cardinality-safe alignment and merge of cleaned FlowCast sources."""

from __future__ import annotations

from typing import Any

import pandas as pd

from flowcast.data.cleaning_types import TableCleaningResult


_WEATHER_RENAMES = {
    "date": "weather_date",
    "time": "weather_time",
    "_source_file": "weather_source_file",
    "_source_row": "weather_source_row",
    "_validation_status": "weather_validation_status",
    "_cleaning_status": "weather_cleaning_status",
}
_CALENDAR_RENAMES = {
    "date": "calendar_date",
    "_source_file": "calendar_source_file",
    "_source_row": "calendar_source_row",
    "_validation_status": "calendar_validation_status",
}


def _require_columns(
    frame: pd.DataFrame,
    required: set[str],
    dataset: str,
) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Cleaned {dataset} is missing columns: {missing}")


def merge_cleaned_sources(
    traffic: pd.DataFrame,
    weather: pd.DataFrame,
    calendar: pd.DataFrame,
    config: dict[str, Any],
) -> TableCleaningResult:
    """Return one context-enriched row per trusted traffic key or fail closed."""

    traffic_key = [str(value) for value in config["traffic_key"]]
    weather_config = config["weather"]
    calendar_config = config["calendar"]
    traffic_station = str(weather_config["traffic_station_column"])
    station_column = str(weather_config["station_column"])
    weather_hour = str(weather_config["hour_column"])
    aligned_hour = str(weather_config["aligned_hour_column"])
    traffic_date = str(calendar_config["traffic_date_column"])
    calendar_date = str(calendar_config["date_column"])

    _require_columns(
        traffic,
        set(traffic_key + [traffic_station, "timestamp"]),
        "traffic",
    )
    _require_columns(
        weather,
        {station_column, weather_hour, "weather_condition"},
        "weather",
    )
    _require_columns(calendar, {calendar_date, "public_holiday"}, "calendar")
    if traffic.duplicated(traffic_key).any():
        raise ValueError("Cleaned traffic key is not unique before merging")
    if weather.duplicated([station_column, weather_hour]).any():
        raise ValueError("Cleaned weather key is not unique before merging")
    if calendar.duplicated([calendar_date]).any():
        raise ValueError("Cleaned calendar key is not unique before merging")

    merged = traffic.sort_values(traffic_key, kind="stable").copy(deep=True)
    traffic_rows = len(merged)
    merged[aligned_hour] = merged["timestamp"].dt.floor("h")
    merged[traffic_date] = (
        merged["timestamp"].dt.tz_localize(None).dt.normalize()
    )

    weather_right = weather.rename(columns=_WEATHER_RENAMES).copy(deep=True)
    merged = merged.merge(
        weather_right,
        how="left",
        left_on=[traffic_station, aligned_hour],
        right_on=[station_column, weather_hour],
        validate="many_to_one",
        indicator="weather_join_status",
        sort=False,
    )
    weather_misses = int(merged["weather_join_status"].ne("both").sum())
    if weather_misses:
        raise ValueError(f"Weather join has {weather_misses} unexpected misses")
    merged["weather_join_status"] = merged["weather_join_status"].astype("string")

    calendar_right = calendar.rename(columns=_CALENDAR_RENAMES).copy(deep=True)
    merged = merged.merge(
        calendar_right,
        how="left",
        on=traffic_date,
        validate="many_to_one",
        indicator="calendar_join_status",
        sort=False,
    )
    calendar_misses = int(merged["calendar_join_status"].ne("both").sum())
    if calendar_misses:
        raise ValueError(f"Calendar join has {calendar_misses} unexpected misses")
    merged["calendar_join_status"] = merged["calendar_join_status"].astype(
        "string"
    )

    if len(merged) != traffic_rows:
        raise ValueError("Source merge changed the trusted traffic row count")
    if merged.duplicated(traffic_key).any():
        raise ValueError("Source merge introduced duplicate traffic keys")
    if not merged[aligned_hour].eq(merged["timestamp"].dt.floor("h")).all():
        raise ValueError("Merged weather-hour alignment is inconsistent")
    merged = merged.sort_values(traffic_key, kind="stable").reset_index(drop=True)

    joined_fields = [
        "weather_condition",
        "temperature",
        "rainfall",
        "visibility",
        "public_holiday",
        "event_flag",
        "roadwork_flag",
    ]
    joined_nulls = {
        field: int(merged[field].isna().sum()) for field in joined_fields
    }
    if any(joined_nulls.values()):
        raise ValueError(f"Merged trusted context contains nulls: {joined_nulls}")

    summary = {
        "inputs": {
            "traffic_rows": len(traffic),
            "traffic_unique_keys": int(
                traffic[traffic_key].drop_duplicates().shape[0]
            ),
            "weather_rows": len(weather),
            "weather_unique_keys": int(
                weather[[station_column, weather_hour]].drop_duplicates().shape[0]
            ),
            "calendar_rows": len(calendar),
            "calendar_unique_keys": int(
                calendar[[calendar_date]].drop_duplicates().shape[0]
            ),
        },
        "joins": {
            "weather": {
                "cardinality": "many_to_one",
                "matched": traffic_rows - weather_misses,
                "missing": weather_misses,
            },
            "calendar": {
                "cardinality": "many_to_one",
                "matched": traffic_rows - calendar_misses,
                "missing": calendar_misses,
            },
        },
        "output_rows": len(merged),
        "output_unique_keys": int(merged[traffic_key].drop_duplicates().shape[0]),
        "row_count_change": len(merged) - traffic_rows,
        "duplicate_output_keys": int(merged.duplicated(traffic_key).sum()),
        "joined_context_nulls": joined_nulls,
        "timestamp_start": merged["timestamp"].min().isoformat(),
        "timestamp_end": merged["timestamp"].max().isoformat(),
    }
    return TableCleaningResult(frame=merged, summary=summary)
