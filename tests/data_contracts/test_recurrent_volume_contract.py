"""Full-artifact contracts for the canonical Step 15 recurrent model."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import torch

from flowcast.modelling.inputs import load_modeling_partition, load_preprocessor
from flowcast.modelling.recurrent_artifacts import (
    load_recurrent_volume_model,
    recurrent_paths,
)
from flowcast.modelling.recurrent_config import load_recurrent_config
from flowcast.modelling.recurrent_support import candidate_endpoints
from flowcast.modelling.recurrent_training import predict_partition
from flowcast.modelling.sequence_data import (
    endpoint_keys,
    prepare_partition,
)
from flowcast.settings import load_settings


@pytest.fixture(scope="module")
def recurrent_artifacts():
    settings = load_settings()
    paths = recurrent_paths(settings, "recurrent_volume_v1")
    if not paths.summary_path.is_file():
        pytest.fail("Canonical recurrent artifacts are missing; run the Step 15 CLI")
    model, scaler, card, summary = load_recurrent_volume_model(settings)
    return settings, paths, model, scaler, card, summary


@pytest.mark.data_contract
def test_recurrent_artifact_coverage_and_freeze(recurrent_artifacts) -> None:
    _, paths, _, _, card, summary = recurrent_artifacts
    selection = json.loads(paths.selection_path.read_text(encoding="utf-8"))
    extension = json.loads(
        paths.registry_extension_path.read_text(encoding="utf-8")
    )

    assert summary["coverage"]["candidate_count"] == 2
    assert summary["coverage"]["horizon_count"] == 4
    assert summary["coverage"]["registry_entry_count"] == 4
    assert selection["status"] == "frozen_before_test_access"
    assert selection["test_metrics_present"] is False
    assert summary["test_access"]["loader_invocation_count"] == 1
    assert summary["test_access"]["models_refit_after_test_load"] is False
    assert extension["entry_count"] == 4
    assert card["selection"]["test_metrics_used"] is False
    assert all(record["passed"] for record in summary["checks"])


@pytest.mark.data_contract
def test_sequences_metrics_and_classical_mapping_are_complete(
    recurrent_artifacts,
) -> None:
    _, paths, _, _, _, summary = recurrent_artifacts
    predictions = pd.read_parquet(paths.predictions_path)
    comparison = pd.read_csv(paths.comparison_path)

    assert summary["sequences"]["train"]["road_count"] == 25
    for split in ("train", "validation", "test"):
        record = summary["sequences"][split]
        assert record["sequence_count"] > 0
        assert record["cross_road_sequences"] == 0
        assert record["cross_partition_sequences"] == 0
        assert record["non_contiguous_sequences"] == 0
        assert record["target_boundary_violations"] == 0
    assert len(summary["metrics"]["validation"]["horizons"]) == 4
    assert len(summary["metrics"]["test"]["horizons"]) == 4
    assert np.isfinite(
        [
            record[metric]
            for record in summary["metrics"]["test"]["horizons"]
            for metric in ("rmse", "mae", "mape_percent", "r_squared")
        ]
    ).all()
    assert predictions["horizon_windows"].nunique() == 4
    assert set(predictions["split"]) == {"validation", "test"}
    assert len(comparison) == 4
    assert comparison["origin_mapping_complete"].all()
    assert comparison["actual_values_identical"].all()
    assert comparison["target_timestamps_identical"].all()


@pytest.mark.data_contract
def test_checkpoint_reload_reproduces_persisted_validation_predictions(
    recurrent_artifacts,
) -> None:
    settings, paths, model, scaler, card, _ = recurrent_artifacts
    config, candidates, _ = load_recurrent_config(settings)
    selected = next(
        value
        for value in candidates
        if value.candidate_id == card["selection"]["candidate_id"]
    )
    longest = max(candidates, key=lambda value: value.sequence_length)
    frame = load_modeling_partition(settings, "validation")
    preprocessor = load_preprocessor(settings, "recurrent")
    features = card["features"]["input_features"]
    targets = tuple(card["target"]["columns"])
    partition = prepare_partition(
        "validation",
        frame,
        preprocessor,
        features,
        targets,
    )
    common = candidate_endpoints(partition, longest, config)
    endpoints = candidate_endpoints(
        partition,
        selected,
        config,
        endpoint_keys(partition, common),
    )
    predicted, _ = predict_partition(
        model,
        partition,
        endpoints,
        selected.sequence_length,
        scaler,
        selected.batch_size,
        int(config["device"]["dataloader_workers"]),
        torch.device("cpu"),
        settings.seed,
    )
    persisted = pd.read_parquet(paths.predictions_path)
    for horizon in range(1, 5):
        observed = persisted.loc[
            persisted["split"].eq("validation")
            & persisted["horizon_windows"].eq(horizon)
        ].sort_values(["road_id", "timestamp"], kind="mergesort")
        expected = pd.DataFrame(
            {
                "road_id": partition.frame.iloc[endpoints]["road_id"].to_numpy(),
                "timestamp": partition.frame.iloc[endpoints][
                    "timestamp"
                ].to_numpy(),
                "prediction": predicted[:, horizon - 1],
            }
        ).sort_values(["road_id", "timestamp"], kind="mergesort")
        assert observed[["road_id", "timestamp"]].reset_index(drop=True).equals(
            expected[["road_id", "timestamp"]].reset_index(drop=True)
        )
        assert observed["prediction"].to_numpy() == pytest.approx(
            expected["prediction"].to_numpy(),
            rel=1e-6,
            abs=1e-5,
        )


@pytest.mark.data_contract
def test_checkpoint_tampering_is_rejected(recurrent_artifacts) -> None:
    settings, paths, _, _, _, _ = recurrent_artifacts
    original = paths.checkpoint_path.read_bytes()
    try:
        paths.checkpoint_path.write_bytes(original + b"tampered")
        with pytest.raises(RuntimeError, match="byte count changed"):
            load_recurrent_volume_model(settings)
    finally:
        paths.checkpoint_path.write_bytes(original)
