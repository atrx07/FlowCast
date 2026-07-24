"""Unit tests for Step 13 jobs, metrics, CV safety, and classifiers."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from flowcast.evaluation.classification import (
    binary_metrics,
    multiclass_metrics,
    select_binary_threshold,
    validate_probabilities,
)
from flowcast.modelling.classification_config import CONGESTION_CLASSES
from flowcast.modelling.classifier_cv import (
    choose_classifier_candidate,
    classifier_fold_frames,
)
from flowcast.modelling.classifier_models import (
    ENCODED_TARGET,
    ClassificationJob,
    ClassifierCandidateSpec,
    build_classification_jobs,
    build_classifier_pipeline,
    fit_classifier,
    score_classifier,
)
from flowcast.modelling.config import load_model_config
from flowcast.modelling.preprocessing import build_feature_groups
from flowcast.settings import load_settings


def _target_manifest() -> list[dict[str, object]]:
    records = []
    for horizon in range(1, 5):
        for task, target_task in (
            ("congestion", "classification_multiclass"),
            ("accident", "classification_binary"),
        ):
            records.append(
                {
                    "name": f"target_{task}_h{horizon}",
                    "task": target_task,
                    "horizon_windows": horizon,
                    "horizon_minutes": horizon * 30,
                    "availability_column": (
                        f"target_{task}_h{horizon}_available"
                    ),
                    "target_timestamp_column": f"target_timestamp_h{horizon}",
                }
            )
    return records


def test_config_generates_eight_jobs_with_fixed_class_order() -> None:
    section = load_model_config(load_settings())["classical_classification"]
    jobs = build_classification_jobs(section, _target_manifest())

    assert [job.job_id for job in jobs] == [
        f"{task}_h{horizon}"
        for task in ("congestion", "accident")
        for horizon in range(1, 5)
    ]
    assert all(
        job.class_names == CONGESTION_CLASSES
        for job in jobs
        if job.task == "congestion"
    )
    assert all(job.primary_metric == "roc_auc" for job in jobs[4:])


def test_metrics_preserve_order_and_reject_invalid_probabilities() -> None:
    actual = np.asarray([0, 1, 2, 3, 0, 1, 2, 3])
    predicted = np.asarray([0, 1, 1, 3, 0, 2, 2, 3])
    probabilities = np.full((8, 4), 0.05)
    probabilities[np.arange(8), predicted] = 0.85
    metrics = multiclass_metrics(
        actual,
        predicted,
        CONGESTION_CLASSES,
        probabilities,
    )

    assert list(metrics["per_class"]) == list(CONGESTION_CLASSES)
    assert metrics["confusion_matrix"][2][1] == 1
    assert metrics["rows"] == 8
    invalid = probabilities.copy()
    invalid[0] = [0.5, 0.5, 0.5, 0.5]
    with pytest.raises(ValueError, match="sum to one"):
        validate_probabilities(invalid, 8, 4)


def test_accident_threshold_is_validation_f1_first_and_deterministic() -> None:
    actual = np.asarray([0, 0, 0, 1, 1, 1])
    probability = np.asarray([0.02, 0.10, 0.20, 0.25, 0.70, 0.90])

    first = select_binary_threshold(actual, probability, 21, 0.5)
    second = select_binary_threshold(actual, probability, 21, 0.5)
    threshold, table, selected = first
    metrics = binary_metrics(actual, probability, threshold)

    assert threshold == second[0]
    assert table.equals(second[1])
    assert selected["f1"] == table["f1"].max()
    assert metrics["threshold"] == threshold
    assert metrics["roc_auc"] == 1.0


def test_classifier_fold_labels_stop_before_validation() -> None:
    timestamps = pd.date_range(
        "2025-01-01",
        periods=24,
        freq="30min",
        tz="Asia/Kolkata",
    )
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "road_id": "A",
            "target_timestamp_h4": timestamps + pd.Timedelta(hours=2),
            ENCODED_TARGET: np.tile([0, 1, 2, 3], 6),
        }
    )
    job = ClassificationJob(
        task="congestion",
        target_column="target_congestion_h4",
        availability_column="target_congestion_h4_available",
        target_timestamp_column="target_timestamp_h4",
        within_split_column="target_within_split_h4",
        horizon_windows=4,
        horizon_minutes=120,
        class_names=CONGESTION_CLASSES,
    )
    fold = {
        "train_start": timestamps[0].isoformat(),
        "train_end": timestamps[9].isoformat(),
        "validation_start": timestamps[14].isoformat(),
        "validation_end": timestamps[20].isoformat(),
    }

    training, validation, evidence = classifier_fold_frames(frame, job, fold, 8)

    assert training["target_timestamp_h4"].max() < validation["timestamp"].min()
    assert evidence["sampled_training_timestamps"] == 8
    leaked = frame.copy()
    leaked.loc[leaked["timestamp"].eq(timestamps[9]), "target_timestamp_h4"] = (
        timestamps[14]
    )
    with pytest.raises(RuntimeError, match="labels reach"):
        classifier_fold_frames(leaked, job, fold, 8)


def test_candidate_selection_maximizes_primary_metric_deterministically() -> None:
    records = [
        {
            "family": "random_forest",
            "candidate_id": "z",
            "status": "success",
            "mean_macro_f1": 0.8,
        },
        {
            "family": "random_forest",
            "candidate_id": "a",
            "status": "success",
            "mean_macro_f1": 0.8,
        },
        {
            "family": "random_forest",
            "candidate_id": "failed",
            "status": "failed",
            "mean_macro_f1": 0.9,
        },
    ]

    chosen = choose_classifier_candidate(
        records,
        "random_forest",
        "macro_f1",
    )
    assert chosen["candidate_id"] == "a"


def test_required_classifier_families_are_seeded_and_reproducible() -> None:
    config = deepcopy(load_model_config(load_settings()))
    config["preprocessing"]["explicit_binary_features"] = ["flag"]
    config["preprocessing"]["bounded_numeric_features"] = ["bounded"]
    records = [
        {"name": "numeric", "dtype": "Float64", "leakage_status": "known_at_origin"},
        {"name": "bounded", "dtype": "Float64", "leakage_status": "known_at_origin"},
        {"name": "flag", "dtype": "boolean", "leakage_status": "known_at_origin"},
        {"name": "band", "dtype": "string", "leakage_status": "known_at_origin"},
    ]
    groups = build_feature_groups(records, config)
    generator = np.random.default_rng(42)
    frame = pd.DataFrame(
        {
            "numeric": generator.normal(size=160),
            "bounded": generator.uniform(0, 1, size=160),
            "flag": pd.Series(
                generator.integers(0, 2, size=160).astype(bool),
                dtype="boolean",
            ),
            "band": np.where(np.arange(160) % 2, "warm", "cool"),
        }
    )
    frame[ENCODED_TARGET] = np.tile(np.arange(4), 40)
    job = ClassificationJob(
        task="congestion",
        target_column="target",
        availability_column="available",
        target_timestamp_column="target_timestamp",
        within_split_column="within",
        horizon_windows=1,
        horizon_minutes=30,
        class_names=CONGESTION_CLASSES,
    )
    specs = [
        ClassifierCandidateSpec(
            "decision_tree",
            "tree",
            "tree",
            {"max_depth": 4, "min_samples_leaf": 2},
        ),
        ClassifierCandidateSpec(
            "random_forest",
            "forest",
            "tree",
            {"n_estimators": 5, "max_depth": 4, "min_samples_leaf": 2},
        ),
        ClassifierCandidateSpec(
            "xgboost",
            "xgb",
            "tree",
            {"n_estimators": 5, "max_depth": 3, "learning_rate": 0.1},
        ),
        ClassifierCandidateSpec(
            "svm",
            "svm",
            "svm",
            {"C": 0.1, "max_iter": 2000, "tol": 0.001},
        ),
    ]
    for spec in specs:
        first = build_classifier_pipeline(spec, job, groups, config, 42)
        second = build_classifier_pipeline(spec, job, groups, config, 42)
        fit_classifier(first, spec, job, frame.iloc[:120], list(groups.input_features))
        fit_classifier(second, spec, job, frame.iloc[:120], list(groups.input_features))
        first_output = score_classifier(
            first,
            job,
            frame.iloc[120:],
            list(groups.input_features),
        )
        second_output = score_classifier(
            second,
            job,
            frame.iloc[120:],
            list(groups.input_features),
        )
        assert first_output[0] == pytest.approx(second_output[0])
        assert first_output[2]["macro_f1"] == second_output[2]["macro_f1"]
