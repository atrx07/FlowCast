"""Per-job fitting and prediction-frame assembly for Step 12."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from flowcast.data.artifacts import artifact_record
from flowcast.modelling.classical_cv import (
    choose_cv_candidate,
    evaluate_candidate_cv,
)
from flowcast.modelling.classical_models import (
    CandidateSpec,
    RegressionJob,
    build_pipeline,
    eligible_rows,
    extract_feature_importance,
    fit_and_score,
)
from flowcast.settings import Settings


def prediction_frame(
    frame: pd.DataFrame,
    predictions: np.ndarray,
    job: RegressionJob,
    split: str,
    selection: dict[str, Any],
    version: str,
) -> pd.DataFrame:
    """Build one traceable validation/test prediction table."""

    result = frame[
        ["road_id", "timestamp", job.target_timestamp_column]
    ].copy()
    result = result.rename(
        columns={job.target_timestamp_column: "target_timestamp"}
    )
    result["split"] = split
    result["job_id"] = job.job_id
    result["target"] = job.target_key
    result["target_column"] = job.target_column
    result["horizon_windows"] = job.horizon_windows
    result["horizon_minutes"] = job.horizon_minutes
    result["actual"] = frame[job.target_column].to_numpy(dtype=np.float64)
    result["prediction"] = predictions
    result["residual"] = result["actual"] - result["prediction"]
    result["selected_family"] = selection["family"]
    result["candidate_id"] = selection["candidate_id"]
    result["model_version"] = version
    return result


def _spec_from_selection(
    selection: dict[str, Any],
    specs: list[CandidateSpec],
) -> CandidateSpec:
    for spec in specs:
        if (
            spec.family == selection["family"]
            and spec.candidate_id == selection["candidate_id"]
        ):
            return spec
    raise RuntimeError("Selected candidate is absent from the configured search")


def fit_regression_job(
    job: RegressionJob,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    specs: list[CandidateSpec],
    folds: list[dict[str, Any]],
    timestamp_budget: int,
    groups: Any,
    config: dict[str, Any],
    input_features: list[str],
    settings: Settings,
    model_path: Path,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
    pd.DataFrame,
    pd.DataFrame,
]:
    """Tune, validate, select, and persist one target/horizon pipeline."""

    eligible_train = eligible_rows(train, job)
    eligible_validation = eligible_rows(validation, job)
    fold_records: list[dict[str, Any]] = []
    aggregates: list[dict[str, Any]] = []
    for spec in specs:
        records, aggregate = evaluate_candidate_cv(
            spec,
            job,
            eligible_train,
            folds,
            timestamp_budget,
            groups,
            config,
            input_features,
            settings.seed,
        )
        fold_records.extend(records)
        aggregates.append(aggregate)

    family_records: list[dict[str, Any]] = []
    family_models: list[tuple[dict[str, Any], Any, np.ndarray]] = []
    for family in config["classical_regression"]["estimators"]:
        candidate = choose_cv_candidate(aggregates, family)
        spec = _spec_from_selection(candidate, specs)
        pipeline = build_pipeline(spec, groups, config, settings.seed)
        predictions, metrics = fit_and_score(
            pipeline,
            eligible_train,
            eligible_validation,
            input_features,
            job.target_column,
        )
        record = {
            "job_id": job.job_id,
            "target": job.target_key,
            "target_column": job.target_column,
            "horizon_windows": job.horizon_windows,
            "horizon_minutes": job.horizon_minutes,
            "family": family,
            "candidate_id": spec.candidate_id,
            "preprocessing_family": spec.preprocessing_family,
            "parameters": dict(spec.parameters),
            "mean_cv_rmse": candidate["mean_rmse"],
            "std_cv_rmse": candidate["std_rmse"],
            "validation": metrics,
        }
        family_records.append(record)
        family_models.append((record, pipeline, predictions))
    winner_record, winner_pipeline, winner_predictions = min(
        family_models,
        key=lambda item: (
            float(item[0]["validation"]["rmse"]),
            float(item[0]["mean_cv_rmse"]),
            str(item[0]["family"]),
            str(item[0]["candidate_id"]),
        ),
    )
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(winner_pipeline, model_path, compress=3)
    output_features = [
        str(name)
        for name in winner_pipeline.named_steps[
            "preprocessor"
        ].get_feature_names_out()
    ]
    selection = {
        **{key: value for key, value in winner_record.items() if key != "validation"},
        "validation": winner_record["validation"],
        "selection_primary_metric": "validation_rmse",
        "train_rows": len(eligible_train),
        "train_start": eligible_train["timestamp"].min().isoformat(),
        "train_end": eligible_train["timestamp"].max().isoformat(),
        "validation_rows": len(eligible_validation),
        "validation_start": eligible_validation["timestamp"].min().isoformat(),
        "validation_end": eligible_validation["timestamp"].max().isoformat(),
        "output_feature_count": len(output_features),
        "model": artifact_record(model_path, settings),
    }
    validation_predictions = prediction_frame(
        eligible_validation,
        winner_predictions,
        job,
        "validation",
        selection,
        str(config["classical_regression"]["version"]),
    )
    importance = extract_feature_importance(
        winner_pipeline,
        job,
        str(selection["family"]),
    )
    del family_models, winner_pipeline
    gc.collect()
    return (
        fold_records,
        aggregates,
        family_records,
        selection,
        validation_predictions,
        importance,
    )
