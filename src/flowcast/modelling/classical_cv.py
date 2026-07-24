"""Time-safe, bounded cross-validation helpers for classical regression."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from flowcast.modelling.classical_models import (
    CandidateSpec,
    RegressionJob,
    build_pipeline,
    fit_and_score,
)
from flowcast.modelling.preprocessing import FeatureGroups


def sample_evenly_spaced_timestamps(
    frame: pd.DataFrame,
    timestamp_budget: int,
) -> pd.DataFrame:
    """Bound CV cost while retaining deterministic coverage of the full fold."""

    if timestamp_budget <= 0:
        raise ValueError("CV timestamp budget must be positive")
    timestamps = pd.DatetimeIndex(frame["timestamp"].drop_duplicates().sort_values())
    if len(timestamps) <= timestamp_budget:
        return frame.reset_index(drop=True)
    indices = np.linspace(
        0,
        len(timestamps) - 1,
        num=timestamp_budget,
        dtype=np.int64,
    )
    selected_timestamps = timestamps[np.unique(indices)]
    sampled = frame.loc[frame["timestamp"].isin(selected_timestamps)]
    return sampled.sort_values(
        ["timestamp", "road_id"],
        kind="mergesort",
    ).reset_index(drop=True)


def fold_frames(
    eligible: pd.DataFrame,
    job: RegressionJob,
    fold: dict[str, Any],
    timestamp_budget: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Materialize one time-safe, horizon-gapped CV fold."""

    train_start = pd.Timestamp(fold["train_start"])
    train_end = pd.Timestamp(fold["train_end"])
    validation_start = pd.Timestamp(fold["validation_start"])
    validation_end = pd.Timestamp(fold["validation_end"])
    full_training = eligible.loc[
        eligible["timestamp"].between(train_start, train_end)
    ]
    validation = eligible.loc[
        eligible["timestamp"].between(validation_start, validation_end)
    ].reset_index(drop=True)
    training = sample_evenly_spaced_timestamps(
        full_training,
        timestamp_budget,
    )
    if training.empty or validation.empty:
        raise RuntimeError(f"{job.job_id} produced an empty CV fold")
    if training["timestamp"].max() >= validation["timestamp"].min():
        raise RuntimeError("CV origin order is not chronological")
    if training[job.target_timestamp_column].max() >= validation_start:
        raise RuntimeError("CV training labels reach the validation period")
    evidence = {
        "full_training_rows": len(full_training),
        "sampled_training_rows": len(training),
        "sampled_training_timestamps": int(training["timestamp"].nunique()),
        "validation_rows": len(validation),
        "training_origin_start": training["timestamp"].min().isoformat(),
        "training_origin_end": training["timestamp"].max().isoformat(),
        "maximum_training_target_timestamp": training[
            job.target_timestamp_column
        ].max().isoformat(),
        "validation_origin_start": validation["timestamp"].min().isoformat(),
        "validation_origin_end": validation["timestamp"].max().isoformat(),
    }
    return training, validation, evidence


def evaluate_candidate_cv(
    spec: CandidateSpec,
    job: RegressionJob,
    eligible_training: pd.DataFrame,
    folds: list[dict[str, Any]],
    timestamp_budget: int,
    groups: FeatureGroups,
    config: dict[str, Any],
    input_features: list[str],
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Evaluate one candidate on every frozen expanding-window fold."""

    fold_records: list[dict[str, Any]] = []
    for fold in folds:
        base = {
            "job_id": job.job_id,
            "target": job.target_key,
            "target_column": job.target_column,
            "horizon_windows": job.horizon_windows,
            "family": spec.family,
            "candidate_id": spec.candidate_id,
            "fold": int(fold["fold"]),
        }
        try:
            training, validation, evidence = fold_frames(
                eligible_training,
                job,
                fold,
                timestamp_budget,
            )
            pipeline = build_pipeline(spec, groups, config, seed)
            _, metrics = fit_and_score(
                pipeline,
                training,
                validation,
                input_features,
                job.target_column,
            )
            fold_records.append(
                {
                    **base,
                    "status": "success",
                    "failure": "",
                    **evidence,
                    **metrics,
                }
            )
        except Exception as error:  # Candidate failures must remain visible.
            fold_records.append(
                {
                    **base,
                    "status": "failed",
                    "failure": f"{type(error).__name__}: {error}",
                }
            )
    successful = [
        record for record in fold_records if record["status"] == "success"
    ]
    aggregate: dict[str, Any] = {
        "job_id": job.job_id,
        "target": job.target_key,
        "target_column": job.target_column,
        "horizon_windows": job.horizon_windows,
        "family": spec.family,
        "candidate_id": spec.candidate_id,
        "preprocessing_family": spec.preprocessing_family,
        "parameters": dict(spec.parameters),
        "fold_count": len(folds),
        "successful_folds": len(successful),
        "status": "success" if len(successful) == len(folds) else "failed",
    }
    if len(successful) == len(folds):
        for metric in ("rmse", "mae", "mape_percent", "r_squared"):
            values = np.asarray(
                [float(record[metric]) for record in successful],
                dtype=np.float64,
            )
            aggregate[f"mean_{metric}"] = round(float(values.mean()), 10)
            aggregate[f"std_{metric}"] = round(float(values.std(ddof=0)), 10)
        aggregate["fit_seconds"] = round(
            sum(float(record["fit_seconds"]) for record in successful),
            6,
        )
        aggregate["prediction_seconds"] = round(
            sum(float(record["prediction_seconds"]) for record in successful),
            6,
        )
        aggregate["failure"] = ""
    else:
        aggregate["failure"] = "One or more required CV folds failed"
    return fold_records, aggregate


def choose_cv_candidate(
    aggregates: list[dict[str, Any]],
    family: str,
) -> dict[str, Any]:
    """Choose one successful family candidate by deterministic mean CV RMSE."""

    eligible = [
        record
        for record in aggregates
        if record["family"] == family and record["status"] == "success"
    ]
    if not eligible:
        raise RuntimeError(f"No successful CV candidate remains for {family}")
    return min(
        eligible,
        key=lambda record: (
            float(record["mean_rmse"]),
            str(record["candidate_id"]),
        ),
    )
