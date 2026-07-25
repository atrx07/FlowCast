"""Full-data Step 11 convergence, lineage, and persistence contracts."""

from __future__ import annotations

from dataclasses import replace
import shutil

import numpy as np
import pandas as pd
import pytest

from flowcast.data.audit import sha256_file
from flowcast.modelling.inputs import (
    load_modeling_partition,
    load_preprocessor,
    load_verified_modeling_artifacts,
)
from flowcast.modelling.regression import run_scratch_linear
from flowcast.modelling.scratch_inputs import load_scratch_linear_model
from flowcast.settings import load_settings


@pytest.fixture(scope="module")
def scratch_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("scratch-linear-contract")
    base = load_settings()
    artifacts = root / "artifacts"
    processed = root / "processed"
    source = (
        base.artifacts_dir
        / "features"
        / base.modelling_version
        / "summary.json"
    )
    destination = (
        artifacts / "features" / base.modelling_version / "summary.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    for source, destination in (
        (
            base.processed_dir
            / base.processed_version
            / "dataset.parquet",
            processed / base.processed_version / "dataset.parquet",
        ),
        (
            base.artifacts_dir
            / "quality"
            / base.processed_version
            / "summary.json",
            artifacts
            / "quality"
            / base.processed_version
            / "summary.json",
        ),
        (
            base.artifacts_dir
            / "features"
            / base.processed_version
            / "manifest.json",
            artifacts
            / "features"
            / base.processed_version
            / "manifest.json",
        ),
        (
            base.artifacts_dir
            / "quality"
            / base.feature_version
            / "summary.json",
            artifacts
            / "quality"
            / base.feature_version
            / "summary.json",
        ),
        (
            base.artifacts_dir
            / "features"
            / base.feature_version
            / "manifest.json",
            artifacts
            / "features"
            / base.feature_version
            / "manifest.json",
        ),
    ):
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    settings = replace(
        base,
        artifacts_dir=artifacts,
        processed_dir=processed,
    )
    return run_scratch_linear(settings), settings


@pytest.mark.data_contract
def test_scratch_proofs_and_flowcast_loss_are_reproducible(scratch_run) -> None:
    run, _ = scratch_run
    summary = run.summary

    assert summary["gradient_check"]["passed"]
    assert summary["gradient_check"]["parameter_count"] == 6
    assert summary["gradient_check"]["maximum_absolute_error"] < 1.0e-6
    assert summary["synthetic_proof"]["passed"]
    assert summary["synthetic_proof"]["maximum_coefficient_error"] < 1.0e-5
    assert summary["synthetic_proof"]["bias_absolute_error"] < 1.0e-5
    training = summary["training"]
    assert training["converged"]
    assert training["train_rows"] == 25_000
    assert training["eligible_training_rows"] == 126_825
    assert training["validation_rows"] == 27_150
    assert training["input_feature_count"] == 62
    assert training["output_feature_count"] == 64
    assert training["final_loss"] < training["initial_loss"]


@pytest.mark.data_contract
def test_comparison_uses_identical_validation_rows_without_test_access(
    scratch_run,
) -> None:
    run, _ = scratch_run
    summary = run.summary

    assert summary["purpose"] == "mathematical_verification_not_model_selection"
    assert summary["training"]["same_rows_for_both_estimators"]
    assert summary["test_partition"] == {
        "default_access_rejected": True,
        "rows_loaded": 0,
        "metrics_calculated": False,
    }
    assert summary["metrics"]["split"] == "validation"
    assert summary["metrics"]["scratch"]["rows"] == 27_150
    assert summary["metrics"]["sklearn"]["rows"] == 27_150
    assert summary["metrics"]["scratch"]["rmse"] == pytest.approx(
        86.9331070892,
        rel=1.0e-8,
    )
    assert summary["metrics"]["sklearn"]["rmse"] == pytest.approx(
        80.8723111105,
        rel=1.0e-8,
    )
    assert all(record["passed"] for record in summary["checks"])


@pytest.mark.data_contract
def test_persisted_scratch_model_reproduces_validation_predictions(
    scratch_run,
) -> None:
    run, settings = scratch_run
    model, summary = load_scratch_linear_model(settings)
    artifacts = load_verified_modeling_artifacts(settings)
    features = [record["name"] for record in artifacts.schema["input_features"]]
    validation = load_modeling_partition(settings, "validation")
    selected = validation["target_volume_h1_available"].fillna(False).astype(
        bool
    ) & validation["target_within_split_h1"].fillna(False).astype(bool)
    validation = validation.loc[selected].sort_values(
        ["timestamp", "road_id"], kind="mergesort"
    )
    processor = load_preprocessor(settings, "linear")
    matrix = np.asarray(processor.transform(validation[features]), dtype=float)
    persisted = pd.read_parquet(run.predictions_path)

    assert model.predict(matrix) == pytest.approx(
        persisted["scratch_prediction"].to_numpy()
    )
    assert summary["artifacts"]["model"]["sha256"] == sha256_file(
        run.model_path
    )


@pytest.mark.data_contract
def test_step_11_artifacts_are_deterministic_and_tamper_evident(
    scratch_run,
) -> None:
    first, settings = scratch_run
    paths = {
        "summary": first.summary_path,
        "report": first.report_path,
        "convergence": first.convergence_path,
        "coefficients": first.coefficients_path,
        "predictions": first.predictions_path,
        "model": first.model_path,
    }
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    repeated = run_scratch_linear(settings)
    repeated_paths = {
        "summary": repeated.summary_path,
        "report": repeated.report_path,
        "convergence": repeated.convergence_path,
        "coefficients": repeated.coefficients_path,
        "predictions": repeated.predictions_path,
        "model": repeated.model_path,
    }
    assert repeated.summary == first.summary
    assert {
        name: sha256_file(path) for name, path in repeated_paths.items()
    } == hashes
    assert b"\r\n" not in repeated.summary_path.read_bytes()
    assert b"\r\n" not in repeated.report_path.read_bytes()

    original = repeated.model_path.read_bytes()
    try:
        repeated.model_path.write_bytes(original + b"tampered")
        with pytest.raises(RuntimeError, match="byte count changed"):
            load_scratch_linear_model(settings)
    finally:
        repeated.model_path.write_bytes(original)
