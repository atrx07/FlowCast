"""Modeling, evaluation, and confidence-analysis CLI commands."""

from __future__ import annotations

import argparse
import logging
from typing import Any

from flowcast.evaluation.confidence_pipeline import run_confidence_analysis
from flowcast.modelling.classification import run_classical_classification
from flowcast.modelling.classical_regression import run_classical_regression
from flowcast.modelling.pipeline import run_modeling_prep
from flowcast.modelling.recurrent import run_recurrent_volume
from flowcast.modelling.registry import run_classical_registry
from flowcast.modelling.regression import run_scratch_linear
from flowcast.settings import Settings


def _version_argument(
    parser: argparse.ArgumentParser,
    description: str,
) -> None:
    parser.add_argument("--version", default=None, help=description)


def register_model_parsers(subparsers: Any) -> None:
    """Register modeling and evaluation commands on the root parser."""

    definitions = (
        (
            "prepare-modeling",
            "Freeze chronological splits and fit training-only preprocessors.",
            "Versioned split/preprocessing artifact directory "
            "(default: split_preprocessing_v1).",
        ),
        (
            "train-scratch-linear",
            "Prove NumPy gradient descent against sklearn on frozen data.",
            "Versioned scratch-linear artifact directory "
            "(default: scratch_linear_v1).",
        ),
        (
            "train-classical-regression",
            "Tune, freeze, and evaluate all classical regression jobs.",
            "Versioned classical-regression artifact directory "
            "(default: classical_regression_v1).",
        ),
        (
            "train-classical-classification",
            "Tune, calibrate, freeze, and evaluate all classifier jobs.",
            "Versioned classification artifact directory "
            "(default: classical_classification_v1).",
        ),
        (
            "build-classical-registry",
            "Verify and combine all frozen classical models and scoreboards.",
            "Versioned classical-registry artifact directory "
            "(default: classical_registry_v1).",
        ),
        (
            "train-recurrent-volume",
            "Train and evaluate the multi-horizon recurrent volume model.",
            "Artifact version (default: recurrent_volume_v1).",
        ),
        (
            "analyze-confidence",
            "Calibrate confidence and produce immutable-model error analysis.",
            "Artifact version (default: confidence_error_v1).",
        ),
    )
    for name, help_text, version_help in definitions:
        command = subparsers.add_parser(name, help=help_text)
        _version_argument(command, version_help)


def run_model_command(
    args: argparse.Namespace,
    settings: Settings,
    logger: logging.Logger,
) -> int | None:
    """Dispatch one modeling/evaluation command or return None."""

    if args.command == "prepare-modeling":
        result = run_modeling_prep(settings, version=args.version)
        partitions = result.summary["split"]["partitions"]
        logger.info("Modeling preparation complete: %s", result.summary_path)
        logger.info("Feature schema: %s", result.schema_path)
        logger.info(
            "train=%s validation=%s test=%s features=%s preprocessors=%s",
            partitions["train"]["row_count"],
            partitions["validation"]["row_count"],
            partitions["test"]["row_count"],
            result.summary["preprocessing"]["feature_count"],
            len(result.preprocessor_paths),
        )
        return 0
    if args.command == "train-scratch-linear":
        result = run_scratch_linear(settings, version=args.version)
        metrics = result.summary["metrics"]
        logger.info("Scratch linear proof complete: %s", result.summary_path)
        logger.info("Generated proof report: %s", result.report_path)
        logger.info(
            "train=%s validation=%s iterations=%s scratch_rmse=%.4f "
            "sklearn_rmse=%.4f",
            result.summary["training"]["train_rows"],
            result.summary["training"]["validation_rows"],
            result.summary["training"]["iterations_completed"],
            metrics["scratch"]["rmse"],
            metrics["sklearn"]["rmse"],
        )
        return 0
    if args.command == "train-classical-regression":
        result = run_classical_regression(settings, version=args.version)
        volume = [
            record
            for record in result.summary["scoreboard"]
            if record["target"] == "volume"
        ]
        logger.info("Classical regression complete: %s", result.paths.summary_path)
        logger.info(
            "jobs=%s selected_models=%s prediction_rows=%s",
            result.summary["coverage"]["job_count"],
            result.summary["coverage"]["selected_model_count"],
            result.summary["coverage"]["prediction_rows"],
        )
        logger.info(
            "volume_test_rmse_by_horizon=%s",
            {
                record["horizon_minutes"]: record["test"]["rmse"]
                for record in volume
            },
        )
        return 0
    if args.command == "train-classical-classification":
        result = run_classical_classification(settings, version=args.version)
        logger.info(
            "Classical classification complete: %s", result.paths.summary_path
        )
        logger.info(
            "jobs=%s selected_models=%s prediction_rows=%s",
            result.summary["coverage"]["job_count"],
            result.summary["coverage"]["selected_model_count"],
            result.summary["coverage"]["prediction_rows"],
        )
        logger.info(
            "test_primary_metrics=%s",
            {
                record["job_id"]: record["test"][
                    "macro_f1" if record["task"] == "congestion" else "roc_auc"
                ]
                for record in result.summary["scoreboard"]
            },
        )
        return 0
    if args.command == "build-classical-registry":
        result = run_classical_registry(settings, version=args.version)
        logger.info("Classical registry complete: %s", result.paths.summary_path)
        logger.info(
            "entries=%s prediction_rows=%s acceptance=%s",
            result.summary["coverage"]["entry_count"],
            result.summary["coverage"]["prediction_rows"],
            result.summary["acceptance"],
        )
        return 0
    if args.command == "train-recurrent-volume":
        result = run_recurrent_volume(settings, version=args.version)
        logger.info(
            "Recurrent volume model complete: %s", result.paths.summary_path
        )
        logger.info(
            "candidate=%s test_mean_rmse=%.4f deep_wins=%s/4",
            result.summary["selection"]["selected_candidate_id"],
            result.summary["metrics"]["test"]["mean_rmse"],
            result.summary["acceptance"]["deep_beats_classical_horizons"],
        )
        return 0
    if args.command == "analyze-confidence":
        result = run_confidence_analysis(settings, version=args.version)
        logger.info("Confidence analysis complete: %s", result.paths.summary_path)
        logger.info(
            "regression=%s classification=%s paired=%s conformal_groups=%s",
            result.summary["coverage"]["regression_prediction_rows"],
            result.summary["coverage"]["classification_prediction_rows"],
            result.summary["coverage"]["paired_volume_rows"],
            result.summary["coverage"]["conformal_group_count"],
        )
        return 0
    return None
