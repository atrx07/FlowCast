"""Time-safe cross-validation for bounded Step 13 classifier search."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from flowcast.modelling.classical_cv import sample_evenly_spaced_timestamps
from flowcast.modelling.classifier_models import (
    ClassificationJob,
    ClassifierCandidateSpec,
    build_classifier_pipeline,
    fit_classifier,
    score_classifier,
)
from flowcast.modelling.preprocessing import FeatureGroups


def classifier_fold_frames(
    eligible: pd.DataFrame,
    job: ClassificationJob,
    fold: dict[str, Any],
    timestamp_budget: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Materialize one chronological, horizon-gapped classifier fold."""

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
        raise RuntimeError("Classifier CV origins are not chronological")
    if training[job.target_timestamp_column].max() >= validation_start:
        raise RuntimeError("Classifier CV labels reach the validation period")
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


def evaluate_classifier_candidate_cv(
    spec: ClassifierCandidateSpec,
    job: ClassificationJob,
    eligible_training: pd.DataFrame,
    folds: list[dict[str, Any]],
    timestamp_budget: int,
    groups: FeatureGroups,
    config: dict[str, Any],
    input_features: list[str],
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Evaluate one classifier candidate on every frozen expanding fold."""

    fold_records: list[dict[str, Any]] = []
    for fold in folds:
        base = {
            "job_id": job.job_id,
            "task": job.task,
            "target_column": job.target_column,
            "horizon_windows": job.horizon_windows,
            "family": spec.family,
            "candidate_id": spec.candidate_id,
            "fold": int(fold["fold"]),
        }
        try:
            training, validation, evidence = classifier_fold_frames(
                eligible_training,
                job,
                fold,
                timestamp_budget,
            )
            pipeline = build_classifier_pipeline(
                spec,
                job,
                groups,
                config,
                seed,
            )
            fit_seconds, weight_evidence = fit_classifier(
                pipeline,
                spec,
                job,
                training,
                input_features,
            )
            _, _, metrics, prediction_seconds = score_classifier(
                pipeline,
                job,
                validation,
                input_features,
            )
            fold_records.append(
                {
                    **base,
                    "status": "success",
                    "failure": "",
                    **evidence,
                    **metrics,
                    "fit_seconds": round(float(fit_seconds), 6),
                    "prediction_seconds": round(float(prediction_seconds), 6),
                    "training_weight_evidence": weight_evidence,
                }
            )
        except Exception as error:  # Candidate failures remain auditable.
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
        "task": job.task,
        "target_column": job.target_column,
        "horizon_windows": job.horizon_windows,
        "family": spec.family,
        "candidate_id": spec.candidate_id,
        "preprocessing_family": spec.preprocessing_family,
        "parameters": dict(spec.parameters),
        "primary_metric": job.primary_metric,
        "fold_count": len(folds),
        "successful_folds": len(successful),
        "status": "success" if len(successful) == len(folds) else "failed",
    }
    if len(successful) == len(folds):
        metrics = (
            ("macro_f1", "macro_precision", "macro_recall", "accuracy")
            if job.task == "congestion"
            else ("roc_auc", "pr_auc")
        )
        for metric in metrics:
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


def choose_classifier_candidate(
    aggregates: list[dict[str, Any]],
    family: str,
    primary_metric: str,
) -> dict[str, Any]:
    """Choose one successful family candidate by descending mean CV metric."""

    eligible = [
        record
        for record in aggregates
        if record["family"] == family and record["status"] == "success"
    ]
    if not eligible:
        raise RuntimeError(f"No successful classifier candidate remains for {family}")
    return min(
        eligible,
        key=lambda record: (
            -float(record[f"mean_{primary_metric}"]),
            str(record["candidate_id"]),
        ),
    )
