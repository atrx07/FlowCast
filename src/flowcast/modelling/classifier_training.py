"""Per-job tuning, calibration, thresholding, and persistence for Step 13."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.pipeline import Pipeline

from flowcast.data.artifacts import artifact_record
from flowcast.evaluation.classification import (
    binary_metrics,
    multiclass_metrics,
    probability_quality,
    select_binary_threshold,
)
from flowcast.modelling.classifier_cv import (
    choose_classifier_candidate,
    evaluate_classifier_candidate_cv,
)
from flowcast.modelling.classifier_models import (
    ENCODED_TARGET,
    ClassificationJob,
    ClassifierCandidateSpec,
    build_classifier_pipeline,
    eligible_classification_rows,
    extract_classifier_importance,
    fit_classifier,
    ordered_probabilities,
    score_classifier,
)
from flowcast.modelling.classification_outputs import (
    classification_prediction_frame,
)
from flowcast.settings import Settings


def _spec_from_selection(
    candidate: dict[str, Any],
    specs: list[ClassifierCandidateSpec],
) -> ClassifierCandidateSpec:
    for spec in specs:
        if (
            spec.family == candidate["family"]
            and spec.candidate_id == candidate["candidate_id"]
        ):
            return spec
    raise RuntimeError("Selected classifier candidate is absent from configuration")


def _chronological_calibration_frames(
    validation: pd.DataFrame,
    fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    timestamps = pd.DatetimeIndex(
        validation["timestamp"].drop_duplicates().sort_values()
    )
    split_index = int(np.floor(len(timestamps) * fraction))
    split_index = min(max(split_index, 1), len(timestamps) - 1)
    calibration_end = timestamps[split_index - 1]
    assessment_start = timestamps[split_index]
    calibration = validation.loc[
        validation["timestamp"].le(calibration_end)
    ].reset_index(drop=True)
    assessment = validation.loc[
        validation["timestamp"].ge(assessment_start)
    ].reset_index(drop=True)
    if calibration.empty or assessment.empty:
        raise RuntimeError("Validation calibration split produced an empty slice")
    if calibration["timestamp"].max() >= assessment["timestamp"].min():
        raise RuntimeError("Calibration assessment is not chronologically later")
    return calibration, assessment, {
        "fit_rows": len(calibration),
        "fit_start": calibration["timestamp"].min().isoformat(),
        "fit_end": calibration["timestamp"].max().isoformat(),
        "assessment_rows": len(assessment),
        "assessment_start": assessment["timestamp"].min().isoformat(),
        "assessment_end": assessment["timestamp"].max().isoformat(),
    }


def _calibrate_and_decide(
    pipeline: Pipeline,
    job: ClassificationJob,
    validation: pd.DataFrame,
    input_features: list[str],
    calibration_config: dict[str, Any],
) -> tuple[Any, dict[str, Any], pd.DataFrame]:
    calibration, assessment, split_evidence = _chronological_calibration_frames(
        validation,
        float(calibration_config["validation_fit_fraction"]),
    )
    raw_quality: dict[str, float] | None = None
    if hasattr(pipeline, "predict_proba"):
        raw_probabilities = ordered_probabilities(
            pipeline,
            assessment,
            input_features,
            len(job.class_names),
        )
        raw_quality = probability_quality(
            assessment[ENCODED_TARGET],
            raw_probabilities,
            len(job.class_names),
        )

    calibrator = CalibratedClassifierCV(
        FrozenEstimator(pipeline),
        method=str(calibration_config["method"]),
        n_jobs=-1,
    )
    calibrator.fit(
        calibration[input_features],
        calibration[ENCODED_TARGET].to_numpy(dtype=np.int64),
    )
    calibrated_probabilities = ordered_probabilities(
        calibrator,
        assessment,
        input_features,
        len(job.class_names),
    )
    calibrated_quality = probability_quality(
        assessment[ENCODED_TARGET],
        calibrated_probabilities,
        len(job.class_names),
    )
    minimum = float(calibration_config["minimum_improvement"])
    improvement = (
        None
        if raw_quality is None
        else float(raw_quality["brier_score"])
        - float(calibrated_quality["brier_score"])
    )
    applied = raw_quality is None or float(improvement) >= minimum
    reason = (
        "required_for_probability_output"
        if raw_quality is None
        else (
            "validation_brier_improved"
            if applied
            else "validation_brier_improvement_below_minimum"
        )
    )
    decision = {
        "method": str(calibration_config["method"]),
        "assessment_metric": "brier_score",
        "minimum_improvement": minimum,
        "raw_probability_available": raw_quality is not None,
        "raw_quality": raw_quality,
        "calibrated_quality": calibrated_quality,
        "brier_improvement": (
            None if improvement is None else round(float(improvement), 10)
        ),
        "applied": applied,
        "reason": reason,
        **split_evidence,
    }
    return (calibrator if applied else pipeline), decision, assessment


def fit_classification_job(
    job: ClassificationJob,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    specs: list[ClassifierCandidateSpec],
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
    pd.DataFrame,
]:
    """Tune, select, calibrate, threshold, and persist one classifier."""

    section = config["classical_classification"]
    eligible_train = eligible_classification_rows(train, job)
    eligible_validation = eligible_classification_rows(validation, job)
    fold_records: list[dict[str, Any]] = []
    aggregates: list[dict[str, Any]] = []
    for spec in specs:
        records, aggregate = evaluate_classifier_candidate_cv(
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
    family_models: list[tuple[dict[str, Any], Pipeline]] = []
    for family in section["estimators"]:
        candidate = choose_classifier_candidate(
            aggregates,
            family,
            job.primary_metric,
        )
        spec = _spec_from_selection(candidate, specs)
        pipeline = build_classifier_pipeline(
            spec,
            job,
            groups,
            config,
            settings.seed,
        )
        fit_seconds, weight_evidence = fit_classifier(
            pipeline,
            spec,
            job,
            eligible_train,
            input_features,
        )
        _, probabilities, metrics, prediction_seconds = score_classifier(
            pipeline,
            job,
            eligible_validation,
            input_features,
        )
        record = {
            "job_id": job.job_id,
            "task": job.task,
            "target_column": job.target_column,
            "horizon_windows": job.horizon_windows,
            "horizon_minutes": job.horizon_minutes,
            "family": family,
            "candidate_id": spec.candidate_id,
            "preprocessing_family": spec.preprocessing_family,
            "parameters": dict(spec.parameters),
            f"mean_cv_{job.primary_metric}": candidate[
                f"mean_{job.primary_metric}"
            ],
            f"std_cv_{job.primary_metric}": candidate[
                f"std_{job.primary_metric}"
            ],
            "validation": metrics,
            "fit_seconds": round(float(fit_seconds), 6),
            "prediction_seconds": round(float(prediction_seconds), 6),
            "probability_available": probabilities is not None,
            "training_weight_evidence": weight_evidence,
        }
        family_records.append(record)
        family_models.append((record, pipeline))

    winner_record, winner_pipeline = min(
        family_models,
        key=lambda item: (
            -float(item[0]["validation"][job.primary_metric]),
            -float(item[0][f"mean_cv_{job.primary_metric}"]),
            str(item[0]["family"]),
            str(item[0]["candidate_id"]),
        ),
    )
    final_estimator, calibration, assessment = _calibrate_and_decide(
        winner_pipeline,
        job,
        eligible_validation,
        input_features,
        section["probability"]["calibration"],
    )
    threshold_table = pd.DataFrame()
    operating_threshold: float | None = None
    if job.task == "accident":
        assessment_probabilities = ordered_probabilities(
            final_estimator,
            assessment,
            input_features,
            2,
        )
        operating_threshold, threshold_table, threshold_selected = (
            select_binary_threshold(
                assessment[ENCODED_TARGET],
                assessment_probabilities[:, 1],
                int(section["accident_threshold"]["candidate_quantiles"]),
                float(
                    section["accident_threshold"]["include_default_threshold"]
                ),
            )
        )
        calibration["threshold_selection"] = {
            "source": "chronologically_later_validation_assessment",
            **threshold_selected,
        }

    final_probabilities = ordered_probabilities(
        final_estimator,
        eligible_validation,
        input_features,
        len(job.class_names),
    )
    if job.task == "congestion":
        final_metrics = multiclass_metrics(
            eligible_validation[ENCODED_TARGET],
            np.argmax(final_probabilities, axis=1),
            job.class_names,
            final_probabilities,
        )
    else:
        final_metrics = binary_metrics(
            eligible_validation[ENCODED_TARGET],
            final_probabilities[:, 1],
            float(operating_threshold),
        )
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_estimator, model_path, compress=3)
    output_features = [
        str(name)
        for name in winner_pipeline.named_steps[
            "preprocessor"
        ].get_feature_names_out()
    ]
    selection = {
        **{
            key: value
            for key, value in winner_record.items()
            if key not in {"validation", "training_weight_evidence"}
        },
        "selection_validation": winner_record["validation"],
        "validation": final_metrics,
        "training_weight_evidence": winner_record["training_weight_evidence"],
        "selection_primary_metric": f"validation_{job.primary_metric}",
        "class_order": list(job.class_names),
        "calibration": calibration,
        "operating_threshold": operating_threshold,
        "train_rows": len(eligible_train),
        "train_start": eligible_train["timestamp"].min().isoformat(),
        "train_end": eligible_train["timestamp"].max().isoformat(),
        "validation_rows": len(eligible_validation),
        "validation_start": eligible_validation["timestamp"].min().isoformat(),
        "validation_end": eligible_validation["timestamp"].max().isoformat(),
        "output_feature_count": len(output_features),
        "model": artifact_record(model_path, settings),
    }
    validation_predictions = classification_prediction_frame(
        eligible_validation,
        final_probabilities,
        job,
        "validation",
        selection,
        str(section["version"]),
    )
    importance = extract_classifier_importance(
        winner_pipeline,
        job,
        str(selection["family"]),
    )
    if not threshold_table.empty:
        threshold_table.insert(0, "job_id", job.job_id)
        threshold_table.insert(1, "split", "validation_assessment")
        threshold_table["selected"] = threshold_table["threshold"].eq(
            round(float(operating_threshold), 10)
        )
    del family_models, final_estimator, winner_pipeline
    gc.collect()
    return (
        fold_records,
        aggregates,
        family_records,
        selection,
        validation_predictions,
        importance,
        threshold_table,
    )
