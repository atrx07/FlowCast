"""Full-data Step 13 coverage, freeze, probability, and persistence contracts."""

from __future__ import annotations

from dataclasses import replace
import json
import shutil

import numpy as np
import pandas as pd
import pytest

from flowcast.modelling.classification import run_classical_classification
from flowcast.modelling.classification_artifacts import load_classification_model
from flowcast.modelling.inputs import load_modeling_partition
from flowcast.settings import load_settings


@pytest.fixture(scope="module")
def classification_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("classical-classification-contract")
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
    return run_classical_classification(settings), settings


@pytest.mark.data_contract
def test_all_jobs_families_folds_and_artifacts_are_complete(
    classification_run,
) -> None:
    run, _ = classification_run
    summary = run.summary
    candidates = pd.read_csv(run.paths.cv_candidates_path)
    folds = pd.read_csv(run.paths.cv_folds_path)
    families = pd.read_csv(run.paths.family_validation_path)
    calibration = pd.read_csv(run.paths.calibration_path)
    thresholds = pd.read_csv(run.paths.thresholds_path)
    confusions = pd.read_csv(run.paths.confusions_path)
    importance = pd.read_csv(run.paths.importance_path)

    assert summary["coverage"] == {
        "task_count": 2,
        "horizon_count": 4,
        "job_count": 8,
        "required_family_job_pairs": 32,
        "successful_fold_evaluations": 320,
        "selected_model_count": 8,
        "model_card_count": 8,
        "prediction_rows": 428_257,
    }
    assert len(candidates) == 64
    assert candidates["status"].eq("success").all()
    assert len(folds) == 320
    assert folds["status"].eq("success").all()
    assert len(families) == 32
    assert set(families["family"]) == {
        "decision_tree",
        "random_forest",
        "xgboost",
        "svm",
    }
    assert families.groupby("job_id")["family"].nunique().eq(4).all()
    assert len(calibration) == 8
    assert len(thresholds) == 808
    assert thresholds.groupby("job_id")["selected"].sum().eq(1).all()
    assert len(confusions) == 160
    assert len(importance) == 8 * 64
    assert all(record["passed"] for record in summary["checks"])


@pytest.mark.data_contract
def test_all_decisions_freeze_before_the_single_test_load(
    classification_run,
) -> None:
    run, _ = classification_run
    selection = json.loads(
        run.paths.selection_path.read_text(encoding="utf-8")
    )
    access = run.summary["test_access"]

    assert selection["status"] == "frozen_before_test_access"
    assert selection["job_count"] == len(selection["selections"]) == 8
    assert selection["test_metrics_present"] is False
    assert all("test" not in record for record in selection["selections"])
    assert all("calibration" in record for record in selection["selections"])
    accident = [
        record for record in selection["selections"]
        if record["task"] == "accident"
    ]
    assert len(accident) == 4
    assert all(record["operating_threshold"] is not None for record in accident)
    assert all(
        record["calibration"]["threshold_selection"]["source"]
        == "chronologically_later_validation_assessment"
        for record in accident
    )
    assert access["loader_invocation_count"] == 1
    assert access["purpose"] == "final_evaluation"
    assert access["models_refit_after_test_load"] is False
    assert (
        run.paths.selection_path.stat().st_mtime_ns
        <= run.paths.predictions_path.stat().st_mtime_ns
    )


@pytest.mark.data_contract
def test_holdout_metrics_probabilities_and_targets_are_honest(
    classification_run,
) -> None:
    run, _ = classification_run
    summary = run.summary
    predictions = pd.read_parquet(run.paths.predictions_path)

    assert len(summary["scoreboard"]) == 8
    for record in summary["scoreboard"]:
        primary = "macro_f1" if record["task"] == "congestion" else "roc_auc"
        secondary = (
            "macro_recall" if record["task"] == "congestion" else "pr_auc"
        )
        assert np.isfinite(
            [
                record[split][metric]
                for split in ("validation", "test")
                for metric in (
                    primary,
                    secondary,
                    "brier_score",
                    "log_loss",
                )
            ]
        ).all()
        selected = predictions.loc[predictions["job_id"].eq(record["job_id"])]
        probability_columns = [
            f"probability_{name.lower().replace('-', '_').replace(' ', '_')}"
            for name in record["class_order"]
        ]
        probability = selected[probability_columns].to_numpy(dtype=float)
        assert np.isfinite(probability).all()
        assert (probability >= 0).all() and (probability <= 1).all()
        assert probability.sum(axis=1) == pytest.approx(
            np.ones(len(probability)),
            abs=1e-10,
        )

    congestion = [
        record for record in summary["scoreboard"]
        if record["task"] == "congestion"
    ]
    accident = [
        record for record in summary["scoreboard"]
        if record["task"] == "accident"
    ]
    assert summary["acceptance"]["congestion_all_horizons_met"] == all(
        record["test"]["macro_f1"] >= 0.80 for record in congestion
    )
    assert summary["acceptance"]["accident_all_horizons_met"] == all(
        record["test"]["roc_auc"] >= 0.75 for record in accident
    )


@pytest.mark.data_contract
@pytest.mark.parametrize("task", ["congestion", "accident"])
def test_reload_reproduces_ordered_persisted_probabilities(
    classification_run,
    task,
) -> None:
    run, settings = classification_run
    estimator, card, summary = load_classification_model(settings, task, 1)
    validation = load_modeling_partition(settings, "validation")
    selected = (
        validation[f"target_{task}_h1_available"].fillna(False).astype(bool)
        & validation["target_within_split_h1"].fillna(False).astype(bool)
    )
    validation = validation.loc[selected].sort_values(
        ["timestamp", "road_id"],
        kind="mergesort",
    )
    persisted = pd.read_parquet(run.paths.predictions_path)
    persisted = persisted.loc[
        persisted["job_id"].eq(f"{task}_h1")
        & persisted["split"].eq("validation")
    ].sort_values(["timestamp", "road_id"], kind="mergesort")
    probability_columns = [
        f"probability_{name.lower().replace('-', '_').replace(' ', '_')}"
        for name in card["target"]["class_order"]
    ]
    reloaded = estimator.predict_proba(
        validation[card["features"]["input_features"]]
    )

    assert reloaded == pytest.approx(
        persisted[probability_columns].to_numpy()
    )
    score = next(
        record for record in summary["scoreboard"]
        if record["job_id"] == f"{task}_h1"
    )
    assert card["metrics"]["validation"] == score["validation"]
    assert card["metrics"]["test"] == score["test"]


@pytest.mark.data_contract
def test_model_cards_are_complete_and_tampering_is_rejected(
    classification_run,
) -> None:
    run, settings = classification_run
    required = {
        "target",
        "selection",
        "probability",
        "data",
        "features",
        "metrics",
        "lineage",
        "artifacts",
        "limitations",
    }
    for job_id, records in run.summary["models"].items():
        card_path = run.paths.cards_dir / f"{job_id}.json"
        card = json.loads(card_path.read_text(encoding="utf-8"))
        assert required.issubset(card)
        assert card["job_id"] == job_id
        assert card["features"]["input_feature_count"] == 62
        assert card["features"]["output_feature_count"] == 64
        assert card["data"]["train_rows"] > card["data"]["validation_rows"]
        assert records["model"] == card["artifacts"]["model"]

    model_path = run.paths.models_dir / "accident_h1.joblib"
    original = model_path.read_bytes()
    try:
        model_path.write_bytes(original + b"tampered")
        with pytest.raises(RuntimeError, match="byte count changed"):
            load_classification_model(settings, "accident", 1)
    finally:
        model_path.write_bytes(original)
