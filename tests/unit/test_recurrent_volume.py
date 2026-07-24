"""Unit tests for Step 15 sequence safety, model shape, and comparison."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import torch

from flowcast.modelling.recurrent_config import (
    RecurrentCandidate,
    load_recurrent_config,
)
from flowcast.modelling.recurrent_model import RecurrentVolumeForecaster
from flowcast.modelling.recurrent_outputs import compare_with_classical
from flowcast.modelling.recurrent_training import seed_torch
from flowcast.modelling.sequence_data import (
    PreparedPartition,
    RecurrentSequenceDataset,
    build_sequence_endpoints,
    fit_target_scaler,
)
from flowcast.settings import load_settings


TARGETS = tuple(f"target_volume_h{value}" for value in range(1, 5))


def _candidate() -> RecurrentCandidate:
    return RecurrentCandidate(
        candidate_id="tiny",
        recurrent_type="lstm",
        sequence_length=3,
        hidden_size=4,
        layer_count=1,
        recurrent_dropout=0.0,
        head_hidden_size=4,
        head_dropout=0.2,
        batch_size=4,
        learning_rate=0.01,
        weight_decay=0.0,
    )


def _partition() -> PreparedPartition:
    records = []
    for road in ("A", "B"):
        timestamps = pd.date_range(
            "2025-01-01",
            periods=8,
            freq="30min",
            tz="Asia/Kolkata",
        )
        if road == "A":
            timestamps = timestamps.delete(4)
        for offset, timestamp in enumerate(timestamps):
            record = {
                "road_id": road,
                "timestamp": timestamp,
            }
            for horizon, target in enumerate(TARGETS, start=1):
                record[target] = float(100 + offset + horizon)
                record[f"{target}_available"] = True
                record[f"target_within_split_h{horizon}"] = True
                record[f"target_timestamp_h{horizon}"] = (
                    timestamp + pd.Timedelta(minutes=30 * horizon)
                )
            records.append(record)
    frame = pd.DataFrame(records).sort_values(
        ["road_id", "timestamp"],
        kind="mergesort",
    ).reset_index(drop=True)
    features = np.arange(len(frame) * 2, dtype=np.float32).reshape(len(frame), 2)
    return PreparedPartition(
        name="synthetic",
        frame=frame,
        features=features,
        feature_names=("x1", "x2"),
        target_columns=TARGETS,
    )


def test_recurrent_config_is_independent_and_bounded() -> None:
    config, candidates, path = load_recurrent_config(load_settings())

    assert path.name == "recurrent.yaml"
    assert config["target"]["horizons"] == [1, 2, 3, 4]
    assert 1 <= len(candidates) <= 3
    assert max(candidate.sequence_length for candidate in candidates) <= 48
    assert all(candidate.recurrent_type in {"lstm", "gru"} for candidate in candidates)


def test_sequence_endpoints_never_cross_roads_or_time_gaps() -> None:
    partition = _partition()
    endpoints = build_sequence_endpoints(
        partition,
        sequence_length=3,
        horizons=(1, 2, 3, 4),
        cadence_minutes=30,
    )
    frame = partition.frame
    for endpoint in endpoints:
        window = frame.iloc[endpoint - 2 : endpoint + 1]
        assert window["road_id"].nunique() == 1
        assert window["timestamp"].diff().dropna().eq(
            pd.Timedelta(minutes=30)
        ).all()
    a_after_gap = frame.index[
        frame["road_id"].eq("A")
        & frame["timestamp"].eq(pd.Timestamp("2025-01-01 02:30", tz="Asia/Kolkata"))
    ][0]
    assert a_after_gap not in endpoints


def test_target_boundary_mask_excludes_invalid_origin() -> None:
    partition = _partition()
    frame = partition.frame.copy()
    invalid = frame.index[frame["road_id"].eq("B")][3]
    frame.loc[invalid, "target_within_split_h4"] = False
    changed = replace(partition, frame=frame)
    endpoints = build_sequence_endpoints(
        changed,
        sequence_length=3,
        horizons=(1, 2, 3, 4),
        cadence_minutes=30,
    )

    assert invalid not in endpoints


def test_target_scaler_uses_only_supplied_training_endpoints() -> None:
    partition = _partition()
    endpoints = build_sequence_endpoints(
        partition,
        sequence_length=3,
        horizons=(1, 2, 3, 4),
        cadence_minutes=30,
    )
    selected = endpoints[:3]
    scaler = fit_target_scaler(partition.frame, selected, TARGETS)
    expected = partition.frame.iloc[selected][list(TARGETS)].to_numpy(dtype=float)

    assert scaler.fitted_rows == 3
    assert scaler.mean == pytest.approx(expected.mean(axis=0))
    assert scaler.transform(expected).mean(axis=0) == pytest.approx(np.zeros(4))


def test_dataset_and_model_have_required_shapes_and_seeded_weights() -> None:
    partition = _partition()
    endpoints = build_sequence_endpoints(
        partition,
        sequence_length=3,
        horizons=(1, 2, 3, 4),
        cadence_minutes=30,
    )
    scaler = fit_target_scaler(partition.frame, endpoints, TARGETS)
    scaled = scaler.transform(partition.frame[list(TARGETS)].to_numpy(dtype=float))
    dataset = RecurrentSequenceDataset(
        partition.features,
        scaled,
        endpoints,
        sequence_length=3,
    )
    features, target = dataset[0]
    seed_torch(42, True, 1)
    first = RecurrentVolumeForecaster(2, _candidate())
    seed_torch(42, True, 1)
    second = RecurrentVolumeForecaster(2, _candidate())

    assert features.shape == (3, 2)
    assert target.shape == (4,)
    assert first(features.unsqueeze(0)).shape == (1, 4)
    for one, two in zip(first.parameters(), second.parameters(), strict=True):
        assert torch.equal(one, two)


def test_classical_comparison_requires_exact_origin_mapping() -> None:
    timestamp = pd.Timestamp("2025-05-10 00:00", tz="Asia/Kolkata")
    deep_rows = []
    classical_rows = []
    for horizon in range(1, 5):
        common = {
            "road_id": "A",
            "timestamp": timestamp,
            "target_timestamp": timestamp + pd.Timedelta(minutes=30 * horizon),
            "horizon_windows": horizon,
            "actual": 100.0 + horizon,
        }
        deep_rows.append({**common, "prediction": 101.0})
        classical_rows.append(
            {
                **common,
                "split": "test",
                "job_id": f"volume_h{horizon}",
                "prediction": 102.0,
            }
        )
    table, records = compare_with_classical(
        pd.DataFrame(deep_rows),
        pd.DataFrame(classical_rows),
    )

    assert len(table) == len(records) == 4
    assert table["origin_mapping_complete"].all()
    broken = pd.DataFrame(classical_rows[:-1])
    with pytest.raises(RuntimeError, match="lacks a classical prediction"):
        compare_with_classical(pd.DataFrame(deep_rows), broken)
