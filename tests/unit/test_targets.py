"""Unit tests for leakage-safe multi-horizon target construction."""

from __future__ import annotations

import pandas as pd
import pytest

from flowcast.features.config import load_feature_config
from flowcast.features.targets import engineer_targets
from flowcast.settings import load_settings


def _features() -> pd.DataFrame:
    timestamps = pd.date_range(
        "2025-01-01 00:00",
        periods=8,
        freq="30min",
        tz="Asia/Kolkata",
    )
    blocks = []
    for road, offset in (("NL-001", 0), ("NL-002", 100)):
        blocks.append(
            pd.DataFrame(
                {
                    "road_id": road,
                    "timestamp": timestamps,
                    "traffic_volume": pd.array(
                        range(offset, offset + 8), dtype="Int64"
                    ),
                    "avg_speed": pd.array(
                        [30.0 + value for value in range(8)], dtype="Float64"
                    ),
                    "travel_time": pd.array(
                        [10.0 + value for value in range(8)], dtype="Float64"
                    ),
                    "congestion_level": pd.array(
                        ["Free-flow", "Moderate", "Heavy", "Severe"] * 2,
                        dtype="string",
                    ),
                    "accident_count": pd.array(
                        [0, pd.NA, 1, 0, 0, 2, 0, 0], dtype="Int64"
                    ),
                    "_accident_observed": pd.array(
                        [True, False, True, True, True, True, True, True],
                        dtype="boolean",
                    ),
                    "feature_marker": pd.array(
                        range(offset + 1000, offset + 1008), dtype="Int64"
                    ),
                }
            )
        )
    return pd.concat(blocks, ignore_index=True)


def _config() -> dict:
    return load_feature_config(load_settings())


def test_targets_shift_within_road_to_exact_future_timestamp() -> None:
    result = engineer_targets(_features(), _config())
    first = result.frame[result.frame["road_id"].eq("NL-001")].reset_index(drop=True)

    assert first.loc[0, "target_timestamp_h1"] == first.loc[1, "timestamp"]
    assert first.loc[0, "target_timestamp_h4"] == first.loc[4, "timestamp"]
    assert first.loc[0, "target_volume_h1"] == 1
    assert first.loc[0, "target_speed_h4"] == 34.0
    assert first.loc[0, "target_travel_time_h4"] == 14.0
    assert first.loc[0, "target_congestion_h4"] == "Free-flow"
    assert pd.isna(first.loc[7, "target_volume_h1"])


def test_trailing_availability_is_explicit_for_every_horizon() -> None:
    frame = engineer_targets(_features(), _config()).frame

    for horizon in range(1, 5):
        expected_unavailable = 2 * horizon
        assert int(frame[f"target_timestamp_h{horizon}"].isna().sum()) == (
            expected_unavailable
        )
        for name in ("volume", "speed", "travel_time", "congestion"):
            available = frame[f"target_{name}_h{horizon}_available"]
            assert int((~available).sum()) == expected_unavailable
            assert (
                frame[f"target_{name}_h{horizon}"].isna().to_numpy()
                == (~available).to_numpy()
            ).all()


def test_unobserved_accident_windows_never_become_negative_labels() -> None:
    frame = engineer_targets(_features(), _config()).frame
    first = frame[frame["road_id"].eq("NL-001")].reset_index(drop=True)

    assert not first.loc[0, "target_accident_h1_available"]
    assert pd.isna(first.loc[0, "target_accident_h1"])
    assert first.loc[1, "target_accident_h1_available"]
    assert first.loc[1, "target_accident_h1"]
    for horizon in range(1, 5):
        target = frame[f"target_accident_h{horizon}"]
        available = frame[f"target_accident_h{horizon}_available"]
        assert target[~available].isna().all()


def test_input_columns_are_preserved_and_manifest_definitions_are_complete() -> None:
    source = _features()
    result = engineer_targets(source, _config())

    pd.testing.assert_frame_equal(
        result.frame[list(source.columns)],
        source,
    )
    assert len(result.definitions) == 20
    assert len({definition.name for definition in result.definitions}) == 20
    assert {definition.horizon_minutes for definition in result.definitions} == {
        30,
        60,
        90,
        120,
    }
    assert all(definition.availability_column for definition in result.definitions)


def test_duplicate_origin_key_is_rejected() -> None:
    source = _features()
    duplicated = pd.concat([source, source.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate road/timestamp"):
        engineer_targets(duplicated, _config())
