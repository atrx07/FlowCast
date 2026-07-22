"""Unit and leakage tests for causal traffic reconstruction."""

from __future__ import annotations

from copy import deepcopy

import pandas as pd
import yaml

from flowcast.data.clean_traffic import clean_traffic, derive_congestion
from flowcast.settings import load_settings


def _config() -> dict:
    settings = load_settings()
    config = yaml.safe_load(
        settings.cleaning_config_path.read_text(encoding="utf-8")
    )["traffic"]
    config = deepcopy(config)
    config["grid"] = {
        "start": "2025-01-01 00:00",
        "end": "2025-01-01 01:30",
        "frequency": "30min",
        "expected_road_count": 1,
    }
    for policy in config["causal_imputation"].values():
        policy["previous_day_lag_windows"] = 2
        policy["forward_fill_limit_windows"] = 1
    return config


def _frame() -> pd.DataFrame:
    timestamps = pd.DatetimeIndex(
        ["2025-01-01 00:00", "2025-01-01 00:30", "2025-01-01 01:30"],
        tz="Asia/Kolkata",
    )
    return pd.DataFrame(
        {
            "road_id": ["NL-001"] * 3,
            "road_name": ["Test Road"] * 3,
            "latitude": pd.array([12.0] * 3, dtype="Float64"),
            "longitude": pd.array([77.0] * 3, dtype="Float64"),
            "weather_station_id": ["WS-NORTH"] * 3,
            "date": ["2025-01-01"] * 3,
            "time": ["00:00", "00:30", "01:30"],
            "traffic_volume": pd.array([100, None, 130], dtype="Int64"),
            "vehicle_count": pd.array([100, 110, 130], dtype="Int64"),
            "vehicle_type_dist": [
                '{"2W":0.4,"Car":0.4,"LCV":0.1,"HCV":0.09}',
                '{"2W":0.3,"Car":0.5,"LCV":0.1,"HCV":0.1}',
                '{"2W":0.2,"Car":0.6,"LCV":0.1,"HCV":0.1}',
            ],
            "avg_speed": pd.array([50.0, None, 20.0], dtype="Float64"),
            "occupancy": pd.array([20.0, None, 70.0], dtype="Float64"),
            "congestion_level": pd.array(
                ["Moderate", None, "Severe"], dtype="string"
            ),
            "travel_time": pd.array([2.0, 2.2, 5.0], dtype="Float64"),
            "accident_count": pd.array([0, 0, 1], dtype="Int64"),
            "signal_timing": pd.array([40, 40, 50], dtype="Int64"),
            "road_capacity": pd.array([200, 200, 200], dtype="Int64"),
            "_source_file": ["traffic_sensor_log.csv"] * 3,
            "_source_row": pd.array([2, 3, 5], dtype="Int64"),
            "timestamp": timestamps,
            "_validation_status": ["valid", "valid_with_issues", "valid"],
        }
    )


def _issues() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "field": [
                "traffic_volume",
                "avg_speed",
                "occupancy",
                "congestion_level",
                "road_id|timestamp",
            ],
            "reason_code": [
                "missing_value",
                "missing_value",
                "missing_value",
                "missing_value",
                "duplicate_key",
            ],
            "source_row": pd.array([3, 3, 3, 3, 99], dtype="Int64"),
        }
    )


def test_congestion_uses_exact_half_hour_boundaries() -> None:
    result = derive_congestion(
        pd.Series([49, 50, 80, 100], dtype="Int64"),
        pd.Series([200, 200, 200, 200], dtype="Int64"),
    )
    assert result.tolist() == ["Free-flow", "Moderate", "Heavy", "Severe"]


def test_traffic_reconstructs_grid_and_preserves_repair_lineage() -> None:
    result = clean_traffic(_frame(), _issues(), _config())
    cleaned = result.frame

    assert len(cleaned) == 4
    assert cleaned["_inserted_window"].tolist() == [False, False, True, False]
    assert cleaned["traffic_volume"].tolist() == [100, 110, 100, 130]
    assert cleaned["traffic_volume_imputation_method"].tolist() == [
        "observed",
        "same_row_vehicle_count",
        "previous_day_same_window",
        "observed",
    ]
    assert cleaned.loc[1, "traffic_volume_imputation_donor_source_row"] == 3
    assert cleaned.loc[2, "traffic_volume_imputation_donor_source_row"] == 2
    assert cleaned.loc[2, "avg_speed"] == 50.0
    assert cleaned.loc[2, "accident_count"] is pd.NA
    assert not bool(cleaned.loc[2, "_accident_observed"])
    shares = cleaned[["share_2w", "share_car", "share_lcv", "share_hcv"]]
    assert shares.sum(axis=1).round(12).eq(1.0).all()
    assert result.summary["grid"]["inserted_windows"] == 1
    assert result.summary["duplicate_rows_accounted"] == 1


def test_traffic_causal_fill_does_not_depend_on_future_values() -> None:
    original = clean_traffic(_frame(), _issues(), _config()).frame
    mutated = _frame()
    mutated.loc[2, ["traffic_volume", "avg_speed", "occupancy"]] = [999, 1.0, 99.0]
    changed = clean_traffic(mutated, _issues(), _config()).frame

    columns = ["traffic_volume", "avg_speed", "occupancy"]
    pd.testing.assert_frame_equal(original.loc[:2, columns], changed.loc[:2, columns])


def test_traffic_rejects_inconsistent_static_metadata() -> None:
    frame = _frame()
    frame.loc[2, "road_capacity"] = 999
    try:
        clean_traffic(frame, _issues(), _config())
    except ValueError as error:
        assert "metadata is inconsistent" in str(error)
    else:
        raise AssertionError("Inconsistent road metadata was accepted")
