"""Unit tests for frozen chronological split and access policies."""

from __future__ import annotations

from copy import deepcopy

import pandas as pd
import pytest

from flowcast.modelling.config import allocate_largest_remainder, load_model_config
from flowcast.modelling.inputs import ensure_partition_access
from flowcast.modelling.split import assign_chronological_splits
from flowcast.settings import load_settings


def test_largest_remainder_allocates_every_origin_timestamp() -> None:
    assert allocate_largest_remainder(
        7_248,
        {"train": 0.70, "validation": 0.15, "test": 0.15},
    ) == {"train": 5_074, "validation": 1_087, "test": 1_087}


def _synthetic_contract() -> tuple[pd.DataFrame, dict, dict]:
    timestamps = pd.date_range(
        "2025-01-01T00:00:00+05:30",
        periods=12,
        freq="30min",
    )
    frame = pd.MultiIndex.from_product(
        [["R1", "R2"], timestamps],
        names=["road_id", "timestamp"],
    ).to_frame(index=False)
    grouped = frame.groupby("road_id", sort=False)
    for horizon in (1, 2):
        frame[f"target_timestamp_h{horizon}"] = grouped["timestamp"].shift(
            -horizon
        )
        frame[f"target_volume_h{horizon}"] = grouped.cumcount().groupby(
            frame["road_id"], sort=False
        ).shift(-horizon).astype("Float64")
        frame[f"target_volume_h{horizon}_available"] = frame[
            f"target_volume_h{horizon}"
        ].notna()
    manifest = {
        "forecast_horizons": [1, 2],
        "targets": [
            {
                "name": f"target_volume_h{horizon}",
                "horizon_windows": horizon,
                "availability_column": f"target_volume_h{horizon}_available",
                "task": "regression",
            }
            for horizon in (1, 2)
        ],
    }
    config = deepcopy(load_model_config(load_settings()))
    config["split"]["partitions"] = {
        "train": {
            "timestamp_count": 6,
            "start": timestamps[0].isoformat(),
            "end": timestamps[5].isoformat(),
        },
        "validation": {
            "timestamp_count": 3,
            "start": timestamps[6].isoformat(),
            "end": timestamps[8].isoformat(),
        },
        "test": {
            "timestamp_count": 3,
            "start": timestamps[9].isoformat(),
            "end": timestamps[11].isoformat(),
        },
    }
    config["cross_validation"].update(
        {"fold_count": 2, "validation_windows": 1, "gap_windows": 2}
    )
    return frame, manifest, config


def test_horizon_targets_cannot_cross_origin_partition() -> None:
    frame, manifest, config = _synthetic_contract()
    assignments, summary, folds = assign_chronological_splits(
        frame, manifest, config
    )

    assert assignments["split"].value_counts().to_dict() == {
        "train": 12,
        "validation": 6,
        "test": 6,
    }
    assert summary["target_coverage"]["target_volume_h1"]["train"][
        "eligible_rows"
    ] == 10
    assert summary["target_coverage"]["target_volume_h2"]["train"][
        "eligible_rows"
    ] == 8
    for horizon in (1, 2):
        inside = assignments[f"target_within_split_h{horizon}"]
        target_partition = assignments["split"].where(inside)
        target_times = frame[f"target_timestamp_h{horizon}"]
        for partition, record in config["split"]["partitions"].items():
            selected = target_partition.eq(partition)
            assert target_times[selected].between(
                pd.Timestamp(record["start"]),
                pd.Timestamp(record["end"]),
            ).all()
    assert len(folds) == 2
    assert folds[-1]["validation_end"] == config["split"]["partitions"][
        "train"
    ]["end"]


def test_test_partition_is_sealed_for_default_tuning_access() -> None:
    config = load_model_config(load_settings())

    assert ensure_partition_access("train", None, config) == "tuning"
    assert ensure_partition_access("validation", "tuning", config) == "tuning"
    with pytest.raises(PermissionError, match="Test partition is sealed"):
        ensure_partition_access("test", None, config)
    assert ensure_partition_access(
        "test", "final_evaluation", config
    ) == "final_evaluation"
