"""Unit and leakage tests for explanatory feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd

from flowcast.features.config import load_feature_config
from flowcast.features.engineering import engineer_features
from flowcast.settings import load_settings


def _source() -> pd.DataFrame:
    timestamps = pd.date_range(
        "2025-01-06 00:00",
        periods=160,
        freq="30min",
        tz="Asia/Kolkata",
    )
    blocks = []
    for road, offset in (("NL-001", 0), ("NL-002", 1000)):
        count = len(timestamps)
        values = np.arange(1, count + 1) + offset
        calendar_dates = timestamps.tz_localize(None).normalize()
        block = pd.DataFrame(
            {
                "road_id": road,
                "timestamp": timestamps,
                "traffic_volume": pd.array(values, dtype="Int64"),
                "avg_speed": pd.array(values / 10 + 20, dtype="Float64"),
                "occupancy": pd.array(np.full(count, 50.0), dtype="Float64"),
                "signal_timing": pd.array(np.full(count, 45), dtype="Int64"),
                "road_capacity": pd.array(np.full(count, 200), dtype="Int64"),
                "share_2w": pd.array(np.full(count, 0.2), dtype="Float64"),
                "share_car": pd.array(np.full(count, 0.6), dtype="Float64"),
                "share_lcv": pd.array(np.full(count, 0.1), dtype="Float64"),
                "share_hcv": pd.array(np.full(count, 0.1), dtype="Float64"),
                "weather_condition": pd.array(["Clear"] * count, dtype="string"),
                "temperature": pd.array(np.full(count, 20.0), dtype="Float64"),
                "rainfall": pd.array(np.zeros(count), dtype="Float64"),
                "visibility": pd.array(np.full(count, 10000.0), dtype="Float64"),
                "calendar_date": calendar_dates,
                "public_holiday": pd.array(
                    (calendar_dates == pd.Timestamp("2025-01-06")).astype(int),
                    dtype="Int8",
                ),
                "event_flag": pd.array(
                    (calendar_dates == pd.Timestamp("2025-01-07")).astype(int),
                    dtype="Int8",
                ),
                "roadwork_flag": pd.array(np.zeros(count), dtype="Int8"),
                "_inserted_window": False,
                "traffic_volume_original_missing": False,
                "traffic_volume_physical_invalid": False,
                "avg_speed_original_missing": False,
                "avg_speed_physical_invalid": False,
                "temperature_was_missing": False,
                "visibility_was_missing": False,
                "vehicle_shares_normalized": False,
            }
        )
        blocks.append(block)
    return pd.concat(blocks, ignore_index=True)


def _config() -> dict:
    return load_feature_config(load_settings())


def test_temporal_capacity_weather_and_calendar_boundaries() -> None:
    source = _source()
    first_road = source["road_id"].eq("NL-001")
    source.loc[first_road & source["timestamp"].eq("2025-01-06 00:00+05:30"), [
        "temperature",
        "visibility",
    ]] = [14.9, 999.0]
    source.loc[first_road & source["timestamp"].eq("2025-01-06 00:30+05:30"), [
        "temperature",
        "visibility",
    ]] = [15.0, 1000.0]
    source.loc[first_road & source["timestamp"].eq("2025-01-06 01:00+05:30"), [
        "temperature",
        "weather_condition",
    ]] = [25.0, "Rain"]
    frame = engineer_features(source, _config()).frame
    road = frame[frame["road_id"].eq("NL-001")].set_index("timestamp")

    assert road.loc["2025-01-06 07:00", "is_morning_peak"]
    assert not road.loc["2025-01-06 10:00", "is_morning_peak"]
    assert road.loc["2025-01-06 17:00", "is_evening_peak"]
    assert not road.loc["2025-01-06 20:00", "is_evening_peak"]
    assert road.loc["2025-01-06 07:00", "holiday_peak"]
    assert road.loc["2025-01-06 00:00", "half_hour_capacity"] == 100.0
    assert road.loc["2025-01-06 00:00", "volume_capacity_ratio"] == 0.01
    assert road.loc["2025-01-06 00:00", "capacity_headroom"] == 99.0
    assert road.loc["2025-01-06 00:00", "temperature_band"] == "cool"
    assert road.loc["2025-01-06 00:30", "temperature_band"] == "mild"
    assert road.loc["2025-01-06 01:00", "temperature_band"] == "warm"
    assert road.loc["2025-01-06 00:00", "is_low_visibility"]
    assert not road.loc["2025-01-06 00:30", "is_low_visibility"]
    assert road.loc["2025-01-06 01:00", "is_rain"]
    assert road.loc["2025-01-06 12:00", "event_within_proximity"]
    assert road.loc["2025-01-08 12:00", "event_within_proximity"]
    assert not road.loc["2025-01-09 06:00", "event_within_proximity"]


def test_lags_and_rolling_are_segment_isolated_and_shifted() -> None:
    frame = engineer_features(_source(), _config()).frame
    first = frame[frame["road_id"].eq("NL-001")].reset_index(drop=True)
    second = frame[frame["road_id"].eq("NL-002")].reset_index(drop=True)

    assert pd.isna(second.loc[0, "volume_lag_1"])
    assert second.loc[1, "volume_lag_1"] == 1001
    assert first.loc[48, "volume_lag_48"] == 1
    assert first.loc[4, "volume_rolling_mean_4"] == 2.5
    assert first.loc[4, "volume_rolling_std_4"] == np.std([1, 2, 3, 4], ddof=1)
    assert first.loc[4, "speed_rolling_mean_4"] == np.mean(
        [20.1, 20.2, 20.3, 20.4]
    )


def test_future_mutation_cannot_change_earlier_features() -> None:
    source = _source()
    original = engineer_features(source, _config())
    mutated_source = source.copy()
    future = mutated_source["road_id"].eq("NL-001") & mutated_source[
        "timestamp"
    ].eq("2025-01-08 02:00+05:30")
    mutated_source.loc[future, ["traffic_volume", "avg_speed"]] = [9999, 199.0]
    mutated = engineer_features(mutated_source, _config())
    feature_names = [definition.name for definition in original.definitions]
    earlier = original.frame["timestamp"].lt("2025-01-08 02:00+05:30")
    pd.testing.assert_frame_equal(
        original.frame.loc[earlier, feature_names].reset_index(drop=True),
        mutated.frame.loc[earlier, feature_names].reset_index(drop=True),
    )


def test_history_unavailability_is_exactly_flagged_per_road() -> None:
    frame = engineer_features(_source(), _config()).frame
    unavailable = (~frame["history_available"]).groupby(frame["road_id"]).sum()

    assert unavailable.to_dict() == {"NL-001": 48, "NL-002": 48}
    for _, road in frame.groupby("road_id", sort=False):
        assert not road.iloc[:48]["history_available"].any()
        assert road.iloc[48:]["history_available"].all()


def test_manifest_definitions_are_unique_and_cover_required_groups() -> None:
    result = engineer_features(_source(), _config())
    names = [definition.name for definition in result.definitions]

    assert len(names) == len(set(names))
    assert set(definition.group for definition in result.definitions) >= {
        "temporal",
        "history",
        "capacity",
        "weather",
        "calendar",
        "vehicle_share",
        "lineage",
    }
    assert all(
        definition.leakage_status == "known_at_origin"
        for definition in result.definitions
    )
