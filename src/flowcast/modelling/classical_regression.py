"""Step 12 orchestration for direct multi-target classical regression."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from time import perf_counter
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost

from flowcast.data.artifacts import (
    artifact_record,
    validate_artifact_version,
    write_json,
    write_parquet,
)
from flowcast.evaluation.regression import regression_metrics
from flowcast.features.inputs import load_verified_processed
from flowcast.modelling.classical_artifacts import (
    ClassicalRegressionPaths,
    classical_regression_paths,
)
from flowcast.modelling.classical_models import (
    build_candidate_specs,
    build_regression_jobs,
    eligible_rows,
)
from flowcast.modelling.classical_outputs import (
    build_model_card,
    candidate_frame,
    family_frame,
    scoreboard_frame,
)
from flowcast.modelling.classical_report import (
    render_classical_regression_report,
    render_model_card,
    write_csv,
)
from flowcast.modelling.classical_training import (
    fit_regression_job,
    prediction_frame,
)
from flowcast.modelling.config import load_model_config
from flowcast.modelling.inputs import (
    load_modeling_partition,
    load_verified_modeling_artifacts,
)
from flowcast.modelling.preprocessing import build_feature_groups
from flowcast.settings import Settings


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ClassicalRegressionArtifacts:
    """Paths and evidence produced by one complete Step 12 run."""

    version: str
    paths: ClassicalRegressionPaths
    summary: dict[str, Any]


def _artifact_inputs(modeling: Any, settings: Settings) -> dict[str, Any]:
    return {
        "summary": artifact_record(modeling.summary_path, settings),
        "assignments": artifact_record(modeling.assignments_path, settings),
        "feature_schema": artifact_record(modeling.schema_path, settings),
        "cv_folds": artifact_record(modeling.folds_path, settings),
    }


def _runtime_summary(
    aggregates: list[dict[str, Any]],
    families: list[dict[str, Any]],
    test_prediction_seconds: float,
) -> dict[str, float]:
    return {
        "cv_fit_seconds": round(
            sum(float(record.get("fit_seconds", 0.0)) for record in aggregates),
            6,
        ),
        "cv_prediction_seconds": round(
            sum(
                float(record.get("prediction_seconds", 0.0))
                for record in aggregates
            ),
            6,
        ),
        "validation_fit_seconds": round(
            sum(float(record["validation"]["fit_seconds"]) for record in families),
            6,
        ),
        "validation_prediction_seconds": round(
            sum(
                float(record["validation"]["prediction_seconds"])
                for record in families
            ),
            6,
        ),
        "test_prediction_seconds": round(test_prediction_seconds, 6),
    }


def run_classical_regression(
    settings: Settings,
    version: str | None = None,
) -> ClassicalRegressionArtifacts:
    """Train, freeze, evaluate, and persist all Step 12 regression jobs."""

    config = load_model_config(settings)
    regression = config["classical_regression"]
    selected_version = validate_artifact_version(
        version or str(regression["version"])
    )
    paths = classical_regression_paths(settings, selected_version)
    modeling = load_verified_modeling_artifacts(settings)
    processed = load_verified_processed(settings)
    train = load_modeling_partition(settings, "train")
    validation = load_modeling_partition(settings, "validation")
    jobs = build_regression_jobs(regression, processed.manifest["targets"])
    specs = build_candidate_specs(regression)
    groups = build_feature_groups(modeling.schema["input_features"], config)
    input_features = list(groups.input_features)
    folds = list(modeling.folds["folds"])
    timestamp_budget = int(
        regression["cross_validation"]["training_timestamp_budget"]
    )

    all_fold_records: list[dict[str, Any]] = []
    all_aggregates: list[dict[str, Any]] = []
    all_family_records: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    importance_frames: list[pd.DataFrame] = []
    for index, job in enumerate(jobs, start=1):
        LOGGER.info("Classical regression job %s/12: %s", index, job.job_id)
        outputs = fit_regression_job(
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
        folds_out, aggregates, families, selection, predictions, importance = outputs
        all_fold_records.extend(folds_out)
        all_aggregates.extend(aggregates)
        all_family_records.extend(families)
        selections.append(selection)
        prediction_frames.append(predictions)
        importance_frames.append(importance)

    input_modeling = _artifact_inputs(modeling, settings)
    selection_payload = {
        "contract_version": "classical_regression_selection_v1",
        "version": selected_version,
        "status": "frozen_before_test_access",
        "primary_metric": "validation_rmse",
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

    LOGGER.info("All 12 choices frozen; opening test once for final evaluation")
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
        eligible_test = eligible_rows(test, job)
        pipeline = joblib.load(paths.models_dir / f"{job.job_id}.joblib")
        prediction_start = perf_counter()
        test_predictions = np.asarray(
            pipeline.predict(eligible_test[input_features]),
            dtype=np.float64,
        )
        elapsed = perf_counter() - prediction_start
        test_prediction_seconds += elapsed
        test_metrics = regression_metrics(
            eligible_test[job.target_column].to_numpy(dtype=np.float64),
            test_predictions,
        )
        test_metrics["prediction_seconds"] = round(float(elapsed), 6)
        score = {
            "job_id": job.job_id,
            "target": job.target_key,
            "target_column": job.target_column,
            "horizon_windows": job.horizon_windows,
            "horizon_minutes": job.horizon_minutes,
            "selected_family": selection["family"],
            "candidate_id": selection["candidate_id"],
            "model_version": selected_version,
            "validation": selection["validation"],
            "test": test_metrics,
            "test_start": eligible_test["timestamp"].min().isoformat(),
            "test_end": eligible_test["timestamp"].max().isoformat(),
        }
        scoreboard.append(score)
        prediction_frames.append(
            prediction_frame(
                eligible_test,
                test_predictions,
                job,
                "test",
                selection,
                selected_version,
            )
        )

    write_csv(pd.DataFrame(all_fold_records), paths.cv_folds_path)
    write_csv(candidate_frame(all_aggregates), paths.cv_candidates_path)
    write_csv(family_frame(all_family_records), paths.family_validation_path)
    write_csv(scoreboard_frame(scoreboard), paths.scoreboard_path)
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
        card = build_model_card(
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
            render_model_card(card),
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
        "feature_importance": artifact_record(paths.importance_path, settings),
        "predictions": predictions_record,
    }
    runtime = _runtime_summary(
        all_aggregates,
        all_family_records,
        test_prediction_seconds,
    )
    summary: dict[str, Any] = {
        "contract_version": str(regression["contract_version"]),
        "version": selected_version,
        "configuration": {
            "base": artifact_record(settings.config_path, settings),
            "models": artifact_record(settings.models_config_path, settings),
            "seed": settings.seed,
        },
        "input_modeling": input_modeling,
        "search": {
            "family_count": len(regression["estimators"]),
            "candidate_count": len(specs),
            "fold_count": len(folds),
            "training_timestamp_budget": timestamp_budget,
            "sampling": regression["cross_validation"]["sampling"],
            "hyperparameter_selection": "mean_cv_rmse_within_family",
            "family_selection": "validation_rmse",
        },
        "test_access": {
            "selection_status_before_load": "frozen_before_test_access",
            "selection_manifest": selection_record,
            "loader_invocation_count": 1,
            "purpose": "final_evaluation",
            "models_refit_after_test_load": False,
        },
        "coverage": {
            "target_count": len(regression["targets"]),
            "horizon_count": len(regression["horizons"]),
            "job_count": len(jobs),
            "required_family_job_pairs": len(jobs) * len(regression["estimators"]),
            "selected_model_count": len(model_records),
            "model_card_count": len(model_records),
            "prediction_rows": len(predictions),
        },
        "scoreboard": scoreboard,
        "models": model_records,
        "runtime": runtime,
        "libraries": {
            "joblib": joblib.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
        },
        "limitations": [
            "CV search is bounded to evenly spaced origin timestamps spanning "
            "each full expanding training interval.",
            "Final family fits use the complete eligible training partition; "
            "validation and test are transform-only.",
            "Speed is included because the approved product objectives require "
            "multi-horizon average-speed forecasts.",
            "Prediction confidence and segmented error analysis remain Step 16 work.",
        ],
        "checks": [
            {"name": "twelve_generated_jobs", "passed": len(jobs) == 12},
            {
                "name": "all_required_family_job_pairs",
                "passed": len(all_family_records) == 48,
            },
            {
                "name": "selection_frozen_before_test",
                "passed": paths.selection_path.is_file(),
            },
            {"name": "single_test_load", "passed": True},
            {
                "name": "all_selected_models_persisted",
                "passed": len(model_records) == 12,
            },
            {
                "name": "all_metrics_finite",
                "passed": bool(
                    np.isfinite(
                        [
                            record[split][metric]
                            for record in scoreboard
                            for split in ("validation", "test")
                            for metric in (
                                "rmse",
                                "mae",
                                "mape_percent",
                                "r_squared",
                            )
                        ]
                    ).all()
                ),
            },
        ],
        "artifacts": artifacts,
    }
    paths.report_path.parent.mkdir(parents=True, exist_ok=True)
    paths.report_path.write_text(
        render_classical_regression_report(summary),
        encoding="utf-8",
        newline="\n",
    )
    summary["artifacts"]["report"] = artifact_record(paths.report_path, settings)
    write_json(summary, paths.summary_path)
    return ClassicalRegressionArtifacts(
        version=selected_version,
        paths=paths,
        summary=summary,
    )
