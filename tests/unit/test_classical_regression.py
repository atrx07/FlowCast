"""Unit tests for Step 12 job generation, CV safety, and seeded estimators."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from flowcast.modelling.classical_cv import (
    choose_cv_candidate,
    fold_frames,
    sample_evenly_spaced_timestamps,
)
from flowcast.modelling.classical_models import (
    CandidateSpec,
    RegressionJob,
    build_pipeline,
    build_regression_jobs,
    fit_and_score,
)
from flowcast.modelling.config import load_model_config
from flowcast.modelling.preprocessing import build_feature_groups
from flowcast.settings import load_settings


def _target_manifest() -> list[dict[str, object]]:
    records = []
    for target in ("volume", "speed", "travel_time"):
        for horizon in range(1, 5):
            records.append(
                {
                    "name": f"target_{target}_h{horizon}",
                    "task": "regression",
                    "horizon_windows": horizon,
                    "horizon_minutes": horizon * 30,
                    "availability_column": (
                        f"target_{target}_h{horizon}_available"
                    ),
                    "target_timestamp_column": f"target_timestamp_h{horizon}",
                }
            )
    return records


def test_config_generates_exactly_twelve_direct_regression_jobs() -> None:
    config = load_model_config(load_settings())["classical_regression"]
    jobs = build_regression_jobs(config, _target_manifest())

    assert [job.job_id for job in jobs] == [
        f"{target}_h{horizon}"
        for target in ("volume", "speed", "travel_time")
        for horizon in range(1, 5)
    ]
    assert all(
        job.within_split_column.endswith(str(job.horizon_windows))
        for job in jobs
    )


def test_cv_timestamp_budget_is_deterministic_and_spans_fold() -> None:
    timestamps = pd.date_range(
        "2025-01-01",
        periods=20,
        freq="30min",
        tz="Asia/Kolkata",
    )
    frame = pd.DataFrame(
        {
            "timestamp": timestamps.repeat(2),
            "road_id": ["A", "B"] * len(timestamps),
        }
    )

    first = sample_evenly_spaced_timestamps(frame, 5)
    second = sample_evenly_spaced_timestamps(frame, 5)

    assert first.equals(second)
    assert first["timestamp"].nunique() == 5
    assert first["timestamp"].min() == timestamps.min()
    assert first["timestamp"].max() == timestamps.max()
    assert first.groupby("timestamp")["road_id"].nunique().eq(2).all()


def test_fold_training_labels_stop_before_validation() -> None:
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
            "target_volume_h4": np.arange(len(timestamps), dtype=float),
            "target_volume_h4_available": True,
            "target_within_split_h4": True,
            "target_timestamp_h4": timestamps + pd.Timedelta(hours=2),
        }
    )
    job = RegressionJob(
        target_key="volume",
        target_column="target_volume_h4",
        availability_column="target_volume_h4_available",
        target_timestamp_column="target_timestamp_h4",
        within_split_column="target_within_split_h4",
        horizon_windows=4,
        horizon_minutes=120,
    )
    fold = {
        "train_start": timestamps[0].isoformat(),
        "train_end": timestamps[9].isoformat(),
        "validation_start": timestamps[14].isoformat(),
        "validation_end": timestamps[20].isoformat(),
    }

    training, validation, evidence = fold_frames(frame, job, fold, 6)

    assert training["target_timestamp_h4"].max() < validation["timestamp"].min()
    assert evidence["sampled_training_timestamps"] == 6
    assert len(validation) == 7
    leaked = frame.copy()
    leaked.loc[leaked["timestamp"].eq(timestamps[9]), "target_timestamp_h4"] = (
        timestamps[14]
    )
    with pytest.raises(RuntimeError, match="labels reach the validation"):
        fold_frames(leaked, job, fold, 6)


def test_cv_candidate_selection_is_metric_first_and_deterministic() -> None:
    records = [
        {
            "family": "random_forest",
            "candidate_id": "z",
            "status": "success",
            "mean_rmse": 3.0,
        },
        {
            "family": "random_forest",
            "candidate_id": "a",
            "status": "success",
            "mean_rmse": 3.0,
        },
        {
            "family": "random_forest",
            "candidate_id": "failed",
            "status": "failed",
            "mean_rmse": 1.0,
        },
    ]

    assert choose_cv_candidate(records, "random_forest")["candidate_id"] == "a"


def test_all_required_estimators_are_seeded_and_reproducible() -> None:
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
            "numeric": generator.normal(size=80),
            "bounded": generator.uniform(0, 1, size=80),
            "flag": pd.Series(
                generator.integers(0, 2, size=80).astype(bool),
                dtype="boolean",
            ),
            "band": np.where(np.arange(80) % 2, "warm", "cool"),
        }
    )
    frame["target"] = (
        2 * frame["numeric"]
        + frame["bounded"]
        + frame["flag"].astype(float)
    )
    specs = [
        CandidateSpec("linear_regression", "linear", "linear", {}),
        CandidateSpec(
            "decision_tree",
            "tree",
            "tree",
            {"max_depth": 4, "min_samples_leaf": 2},
        ),
        CandidateSpec(
            "random_forest",
            "forest",
            "tree",
            {"n_estimators": 5, "max_depth": 4, "min_samples_leaf": 2},
        ),
        CandidateSpec(
            "xgboost",
            "xgb",
            "tree",
            {"n_estimators": 5, "max_depth": 3, "learning_rate": 0.1},
        ),
    ]
    for spec in specs:
        first = build_pipeline(spec, groups, config, 42)
        second = build_pipeline(spec, groups, config, 42)
        first_predictions, first_metrics = fit_and_score(
            first,
            frame.iloc[:60],
            frame.iloc[60:],
            list(groups.input_features),
            "target",
        )
        second_predictions, second_metrics = fit_and_score(
            second,
            frame.iloc[:60],
            frame.iloc[60:],
            list(groups.input_features),
            "target",
        )
        assert first_predictions == pytest.approx(second_predictions)
        assert first_metrics["rmse"] == second_metrics["rmse"]
        assert np.isfinite(first_predictions).all()
