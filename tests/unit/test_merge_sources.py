"""Unit tests for cardinality-safe cleaned-source alignment."""

from __future__ import annotations

import pandas as pd
import pytest
import yaml

from flowcast.data.merge import merge_cleaned_sources
from flowcast.settings import load_settings


def _config() -> dict:
    settings = load_settings()
    return yaml.safe_load(
        settings.cleaning_config_path.read_text(encoding="utf-8")
    )["merge"]


def _traffic() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "road_id": ["NL-001", "NL-001"],
            "timestamp": pd.DatetimeIndex(
                ["2025-01-01 00:00", "2025-01-01 00:30"],
                tz="Asia/Kolkata",
            ),
            "weather_station_id": ["WS-NORTH", "WS-NORTH"],
            "traffic_volume": pd.array([100, 110], dtype="Int64"),
            "_source_row": pd.array([2, 3], dtype="Int64"),
            "_inserted_window": [False, False],
            "_accident_observed": [True, True],
        }
    )


def _weather() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "station_id": ["WS-NORTH"],
            "date": ["01/01/2025"],
            "time": ["00:00"],
            "weather_condition": ["Clear"],
            "temperature": pd.array([20.0], dtype="Float64"),
            "rainfall": pd.array([0.0], dtype="Float64"),
            "visibility": pd.array([1000.0], dtype="Float64"),
            "weather_hour": pd.DatetimeIndex(
                ["2025-01-01 00:00"], tz="Asia/Kolkata"
            ),
            "_source_file": ["weather_observations.csv"],
            "_source_row": pd.array([2], dtype="Int64"),
            "_validation_status": ["valid"],
            "_cleaning_status": ["unchanged"],
        }
    )


def _calendar() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-01"]),
            "public_holiday": pd.array([0], dtype="Int8"),
            "holiday_name": pd.array([None], dtype="string"),
            "event_flag": pd.array([0], dtype="Int8"),
            "event_name": pd.array([None], dtype="string"),
            "roadwork_flag": pd.array([0], dtype="Int8"),
            "_source_file": ["calendar_events.csv"],
            "_source_row": pd.array([2], dtype="Int64"),
            "_validation_status": ["valid"],
        }
    )


def test_merge_broadcasts_hourly_weather_without_multiplying_rows() -> None:
    result = merge_cleaned_sources(
        _traffic(), _weather(), _calendar(), _config()
    )
    merged = result.frame

    assert len(merged) == 2
    assert merged["weather_condition"].tolist() == ["Clear", "Clear"]
    assert merged["weather_source_row"].tolist() == [2, 2]
    assert merged["calendar_source_row"].tolist() == [2, 2]
    assert merged["weather_join_status"].tolist() == ["both", "both"]
    assert merged["calendar_join_status"].tolist() == ["both", "both"]
    assert merged["_source_row"].tolist() == [2, 3]
    assert result.summary["row_count_change"] == 0


def test_merge_rejects_duplicate_weather_keys() -> None:
    weather = pd.concat([_weather(), _weather()], ignore_index=True)
    with pytest.raises(ValueError, match="weather key is not unique"):
        merge_cleaned_sources(_traffic(), weather, _calendar(), _config())


def test_merge_rejects_duplicate_calendar_keys() -> None:
    calendar = pd.concat([_calendar(), _calendar()], ignore_index=True)
    with pytest.raises(ValueError, match="calendar key is not unique"):
        merge_cleaned_sources(_traffic(), _weather(), calendar, _config())


def test_merge_rejects_unmatched_context() -> None:
    weather = _weather().assign(station_id="WS-SOUTH")
    with pytest.raises(ValueError, match="Weather join has 2 unexpected misses"):
        merge_cleaned_sources(_traffic(), weather, _calendar(), _config())
