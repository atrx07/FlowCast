"""Full-data Step 12 coverage, freeze, persistence, and metric contracts."""

from __future__ import annotations

from dataclasses import replace
import json
import shutil

import numpy as np
import pandas as pd
import pytest

from flowcast.modelling.classical_artifacts import (
    load_classical_regression_model,
)
from flowcast.modelling.classical_regression import run_classical_regression
from flowcast.modelling.inputs import load_modeling_partition
from flowcast.settings import load_settings


@pytest.fixture(scope="module")
def classical_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("classical-regression-contract")
    base = load_settings()
    artifacts = root / "artifacts"
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
    settings = replace(base, artifacts_dir=artifacts)
    return run_classical_regression(settings), settings


@pytest.mark.data_contract
def test_all_jobs_families_folds_and_artifacts_are_complete(classical_run) -> None:
    run, _ = classical_run
    summary = run.summary
    candidates = pd.read_csv(run.paths.cv_candidates_path)
    fold_metrics = pd.read_csv(run.paths.cv_folds_path)
    families = pd.read_csv(run.paths.family_validation_path)
    importance = pd.read_csv(run.paths.importance_path)

    assert summary["coverage"] == {
        "target_count": 3,
        "horizon_count": 4,
        "job_count": 12,
        "required_family_job_pairs": 48,
        "selected_model_count": 12,
        "model_card_count": 12,
        "prediction_rows": 650_700,
    }
    assert len(candidates) == 84
    assert candidates["status"].eq("success").all()
    assert len(fold_metrics) == 420
    assert fold_metrics["status"].eq("success").all()
    assert len(families) == 48
    assert set(families["family"]) == {
        "linear_regression",
        "decision_tree",
        "random_forest",
        "xgboost",
    }
    assert families.groupby("job_id")["family"].nunique().eq(4).all()
    assert len(importance) == 12 * 64
    assert all(record["passed"] for record in summary["checks"])


@pytest.mark.data_contract
def test_selection_is_frozen_before_the_single_test_load(classical_run) -> None:
    run, _ = classical_run
    selection = json.loads(
        run.paths.selection_path.read_text(encoding="utf-8")
    )
    test_access = run.summary["test_access"]

    assert selection["status"] == "frozen_before_test_access"
    assert selection["job_count"] == len(selection["selections"]) == 12
    assert selection["test_metrics_present"] is False
    assert all("test" not in record for record in selection["selections"])
    assert test_access["selection_status_before_load"] == selection["status"]
    assert test_access["loader_invocation_count"] == 1
    assert test_access["purpose"] == "final_evaluation"
    assert test_access["models_refit_after_test_load"] is False
    assert (
        run.paths.selection_path.stat().st_mtime_ns
        <= run.paths.predictions_path.stat().st_mtime_ns
    )


@pytest.mark.data_contract
def test_holdout_metrics_are_finite_and_volume_target_is_met(classical_run) -> None:
    run, _ = classical_run
    scoreboard = run.summary["scoreboard"]

    assert len(scoreboard) == 12
    for record in scoreboard:
        expected_rows = 27_175 - 25 * int(record["horizon_windows"])
        assert record["validation"]["rows"] == expected_rows
        assert record["test"]["rows"] == expected_rows
        for split in ("validation", "test"):
            assert np.isfinite(
                [
                    record[split]["rmse"],
                    record[split]["mae"],
                    record[split]["mape_percent"],
                    record[split]["r_squared"],
                ]
            ).all()
    volume = [record for record in scoreboard if record["target"] == "volume"]
    assert len(volume) == 4
    assert all(record["test"]["mape_percent"] <= 12.0 for record in volume)


@pytest.mark.data_contract
def test_pipeline_reload_reproduces_persisted_predictions(classical_run) -> None:
    run, settings = classical_run
    pipeline, card, summary = load_classical_regression_model(
        settings,
        "volume",
        1,
    )
    validation = load_modeling_partition(settings, "validation")
    selected = (
        validation["target_volume_h1_available"].fillna(False).astype(bool)
        & validation["target_within_split_h1"].fillna(False).astype(bool)
    )
    validation = validation.loc[selected].sort_values(
        ["timestamp", "road_id"],
        kind="mergesort",
    )
    persisted = pd.read_parquet(run.paths.predictions_path)
    persisted = persisted.loc[
        persisted["job_id"].eq("volume_h1")
        & persisted["split"].eq("validation")
    ].sort_values(["timestamp", "road_id"], kind="mergesort")
    features = card["features"]["input_features"]
    reloaded = pipeline.predict(validation[features])

    assert reloaded == pytest.approx(persisted["prediction"].to_numpy())
    assert card["metrics"]["validation"] == summary["scoreboard"][0]["validation"]
    assert card["metrics"]["test"] == summary["scoreboard"][0]["test"]


@pytest.mark.data_contract
def test_model_cards_are_complete_and_model_hashes_reject_tampering(
    classical_run,
) -> None:
    run, settings = classical_run
    required = {
        "target",
        "selection",
        "data",
        "features",
        "metrics",
        "lineage",
        "artifacts",
        "limitations",
    }
    for job_id, records in run.summary["models"].items():
        card_path = settings.root / records["model_card_json"]["path"]
        if not card_path.is_file():
            card_path = run.paths.cards_dir / f"{job_id}.json"
        card = json.loads(card_path.read_text(encoding="utf-8"))
        assert required.issubset(card)
        assert card["job_id"] == job_id
        assert card["features"]["input_feature_count"] == 62
        assert card["features"]["output_feature_count"] == 64
        assert card["data"]["train_rows"] > card["data"]["validation_rows"]

    model_path = run.paths.models_dir / "volume_h1.joblib"
    original = model_path.read_bytes()
    try:
        model_path.write_bytes(original + b"tampered")
        with pytest.raises(RuntimeError, match="byte count changed"):
            load_classical_regression_model(settings, "volume", 1)
    finally:
        model_path.write_bytes(original)
