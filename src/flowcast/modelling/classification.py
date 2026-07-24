"""Step 13 orchestration for multi-horizon classical classification."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from time import perf_counter
from typing import Any

import joblib
import numpy as np
import pandas as pd

from flowcast.data.artifacts import (
    artifact_record,
    validate_artifact_version,
    write_json,
    write_parquet,
)
from flowcast.evaluation.classification import binary_metrics, multiclass_metrics
from flowcast.features.inputs import load_verified_processed
from flowcast.modelling.classical_report import write_csv
from flowcast.modelling.classification_artifacts import (
    ClassificationPaths,
    classification_paths,
)
from flowcast.modelling.classification_outputs import (
    build_classification_model_card,
    calibration_frame,
    candidate_frame,
    classification_prediction_frame,
    confusion_frame,
    family_frame,
    fold_frame,
    scoreboard_frame,
)
from flowcast.modelling.classification_report import (
    render_classification_model_card,
    render_classification_report,
)
from flowcast.modelling.classification_summary import (
    build_classification_summary,
)
from flowcast.modelling.classifier_models import (
    ENCODED_TARGET,
    build_classification_jobs,
    build_classifier_specs,
    eligible_classification_rows,
    ordered_probabilities,
)
from flowcast.modelling.classifier_training import fit_classification_job
from flowcast.modelling.config import load_model_config
from flowcast.modelling.inputs import (
    load_modeling_partition,
    load_verified_modeling_artifacts,
)
from flowcast.modelling.preprocessing import build_feature_groups
from flowcast.settings import Settings


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ClassicalClassificationArtifacts:
    """Paths and evidence produced by one complete Step 13 run."""

    version: str
    paths: ClassificationPaths
    summary: dict[str, Any]


def _artifact_inputs(modeling: Any, settings: Settings) -> dict[str, Any]:
    return {
        "summary": artifact_record(modeling.summary_path, settings),
        "assignments": artifact_record(modeling.assignments_path, settings),
        "feature_schema": artifact_record(modeling.schema_path, settings),
        "cv_folds": artifact_record(modeling.folds_path, settings),
    }


def run_classical_classification(
    settings: Settings,
    version: str | None = None,
) -> ClassicalClassificationArtifacts:
    """Train, freeze, test once, and persist all eight Step 13 jobs."""

    wall_start = perf_counter()
    config = load_model_config(settings)
    section = config["classical_classification"]
    selected_version = validate_artifact_version(
        version or str(section["version"])
    )
    paths = classification_paths(settings, selected_version)
    modeling = load_verified_modeling_artifacts(settings)
    processed = load_verified_processed(settings)
    train = load_modeling_partition(settings, "train")
    validation = load_modeling_partition(settings, "validation")
    jobs = build_classification_jobs(section, processed.manifest["targets"])
    specs = build_classifier_specs(section)
    groups = build_feature_groups(modeling.schema["input_features"], config)
    input_features = list(groups.input_features)
    folds = list(modeling.folds["folds"])
    timestamp_budget = int(
        section["cross_validation"]["training_timestamp_budget"]
    )

    all_fold_records: list[dict[str, Any]] = []
    all_aggregates: list[dict[str, Any]] = []
    all_family_records: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    importance_frames: list[pd.DataFrame] = []
    threshold_frames: list[pd.DataFrame] = []
    for index, job in enumerate(jobs, start=1):
        LOGGER.info("Classical classification job %s/8: %s", index, job.job_id)
        outputs = fit_classification_job(
            job,
            train,
            validation,
            specs,
            folds,
            timestamp_budget,
            groups,
            config,
            input_features,
            settings,
            paths.models_dir / f"{job.job_id}.joblib",
        )
        folds_out, aggregates, families, selection, predictions, importance, thresholds = (
            outputs
        )
        all_fold_records.extend(folds_out)
        all_aggregates.extend(aggregates)
        all_family_records.extend(families)
        selections.append(selection)
        prediction_frames.append(predictions)
        if not importance.empty:
            importance_frames.append(importance)
        if not thresholds.empty:
            threshold_frames.append(thresholds)

    input_modeling = _artifact_inputs(modeling, settings)
    selection_payload = {
        "contract_version": "classical_classification_selection_v1",
        "version": selected_version,
        "status": "frozen_before_test_access",
        "primary_metrics": dict(section["primary_metrics"]),
        "configuration": {
            "base": artifact_record(settings.config_path, settings),
            "models": artifact_record(settings.models_config_path, settings),
            "seed": settings.seed,
        },
        "input_modeling": input_modeling,
        "job_count": len(selections),
        "selections": selections,
        "test_metrics_present": False,
    }
    write_json(selection_payload, paths.selection_path)
    selection_record = artifact_record(paths.selection_path, settings)

    LOGGER.info("All 8 classifier decisions frozen; opening test once")
    test = load_modeling_partition(
        settings,
        "test",
        purpose="final_evaluation",
    )
    scoreboard: list[dict[str, Any]] = []
    test_prediction_seconds = 0.0
    jobs_by_id = {job.job_id: job for job in jobs}
    for selection in selections:
        job = jobs_by_id[str(selection["job_id"])]
        eligible_test = eligible_classification_rows(test, job)
        estimator = joblib.load(paths.models_dir / f"{job.job_id}.joblib")
        prediction_start = perf_counter()
        probabilities = ordered_probabilities(
            estimator,
            eligible_test,
            input_features,
            len(job.class_names),
        )
        elapsed = perf_counter() - prediction_start
        test_prediction_seconds += elapsed
        if job.task == "congestion":
            test_metrics = multiclass_metrics(
                eligible_test[ENCODED_TARGET],
                np.argmax(probabilities, axis=1),
                job.class_names,
                probabilities,
            )
        else:
            test_metrics = binary_metrics(
                eligible_test[ENCODED_TARGET],
                probabilities[:, 1],
                float(selection["operating_threshold"]),
            )
        test_metrics["prediction_seconds"] = round(float(elapsed), 6)
        score = {
            "job_id": job.job_id,
            "task": job.task,
            "target_column": job.target_column,
            "horizon_windows": job.horizon_windows,
            "horizon_minutes": job.horizon_minutes,
            "class_order": list(job.class_names),
            "selected_family": selection["family"],
            "candidate_id": selection["candidate_id"],
            "model_version": selected_version,
            "calibration": selection["calibration"],
            "operating_threshold": selection["operating_threshold"],
            "validation": selection["validation"],
            "test": test_metrics,
            "test_start": eligible_test["timestamp"].min().isoformat(),
            "test_end": eligible_test["timestamp"].max().isoformat(),
        }
        scoreboard.append(score)
        prediction_frames.append(
            classification_prediction_frame(
                eligible_test,
                probabilities,
                job,
                "test",
                selection,
                selected_version,
            )
        )

    write_csv(fold_frame(all_fold_records), paths.cv_folds_path)
    write_csv(candidate_frame(all_aggregates), paths.cv_candidates_path)
    write_csv(family_frame(all_family_records), paths.family_validation_path)
    write_csv(scoreboard_frame(scoreboard), paths.scoreboard_path)
    write_csv(calibration_frame(selections), paths.calibration_path)
    write_csv(confusion_frame(scoreboard), paths.confusions_path)
    thresholds = pd.concat(threshold_frames, ignore_index=True)
    write_csv(thresholds, paths.thresholds_path)
    importance = pd.concat(importance_frames, ignore_index=True)
    write_csv(importance, paths.importance_path)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    write_parquet(predictions, paths.predictions_path)
    predictions_record = artifact_record(paths.predictions_path, settings)

    model_records: dict[str, Any] = {}
    selections_by_job = {str(item["job_id"]): item for item in selections}
    for score in scoreboard:
        job = jobs_by_id[str(score["job_id"])]
        selection = selections_by_job[job.job_id]
        card = build_classification_model_card(
            selection,
            score,
            job,
            modeling,
            selection_record,
            predictions_record,
            settings,
        )
        json_path = paths.cards_dir / f"{job.job_id}.json"
        markdown_path = paths.cards_dir / f"{job.job_id}.md"
        write_json(card, json_path)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(
            render_classification_model_card(card),
            encoding="utf-8",
            newline="\n",
        )
        model_records[job.job_id] = {
            "model": selection["model"],
            "model_card_json": artifact_record(json_path, settings),
            "model_card_markdown": artifact_record(markdown_path, settings),
        }

    artifacts = {
        "selection_manifest": selection_record,
        "cv_fold_metrics": artifact_record(paths.cv_folds_path, settings),
        "cv_candidate_metrics": artifact_record(paths.cv_candidates_path, settings),
        "family_validation_metrics": artifact_record(
            paths.family_validation_path,
            settings,
        ),
        "scoreboard": artifact_record(paths.scoreboard_path, settings),
        "calibration_metrics": artifact_record(paths.calibration_path, settings),
        "accident_thresholds": artifact_record(paths.thresholds_path, settings),
        "confusion_matrices": artifact_record(paths.confusions_path, settings),
        "feature_importance": artifact_record(paths.importance_path, settings),
        "predictions": predictions_record,
    }
    summary = build_classification_summary(
        section=section,
        selected_version=selected_version,
        input_modeling=input_modeling,
        selection_record=selection_record,
        jobs=jobs,
        specs=specs,
        folds=folds,
        timestamp_budget=timestamp_budget,
        all_fold_records=all_fold_records,
        all_aggregates=all_aggregates,
        all_family_records=all_family_records,
        scoreboard=scoreboard,
        model_records=model_records,
        prediction_rows=len(predictions),
        test_prediction_seconds=test_prediction_seconds,
        wall_start=wall_start,
        artifacts=artifacts,
        settings=settings,
    )
    paths.report_path.parent.mkdir(parents=True, exist_ok=True)
    paths.report_path.write_text(
        render_classification_report(summary),
        encoding="utf-8",
        newline="\n",
    )
    summary["artifacts"]["report"] = artifact_record(paths.report_path, settings)
    write_json(summary, paths.summary_path)
    return ClassicalClassificationArtifacts(
        version=selected_version,
        paths=paths,
        summary=summary,
    )
