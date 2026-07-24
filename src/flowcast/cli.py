"""Command-line interface for reproducible FlowCast workflows."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from flowcast.analysis.pipeline import run_eda
from flowcast.data.audit import run_raw_audit
from flowcast.data.clean_context import run_context_cleaning
from flowcast.data.merge_pipeline import run_source_merge
from flowcast.data.quarantine import run_validation_pipeline
from flowcast.data.traffic_pipeline import run_traffic_cleaning
from flowcast.features.pipeline import run_feature_engineering
from flowcast.features.processed_pipeline import run_processed_data
from flowcast.logging_config import configure_logging
from flowcast.modelling.classification import run_classical_classification
from flowcast.modelling.classical_regression import run_classical_regression
from flowcast.modelling.pipeline import run_modeling_prep
from flowcast.modelling.regression import run_scratch_linear
from flowcast.modelling.registry import run_classical_registry
from flowcast.settings import load_settings


def build_parser() -> argparse.ArgumentParser:
    """Build the FlowCast CLI parser."""

    parser = argparse.ArgumentParser(
        prog="flowcast",
        description="FlowCast reproducible traffic forecasting pipeline.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to the base YAML configuration (default: config/base.yaml).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit = subparsers.add_parser(
        "audit",
        help="Preserve, verify, and audit the delivered raw CSV files.",
    )
    audit.add_argument(
        "--version",
        default=None,
        help="Versioned audit output directory (default: configured raw_v1).",
    )
    validate = subparsers.add_parser(
        "validate",
        help="Validate immutable raw sources and persist quarantine evidence.",
    )
    validate.add_argument(
        "--version",
        default=None,
        help="Versioned validation output directory (default: validated_v1).",
    )
    clean_context = subparsers.add_parser(
        "clean-context",
        help="Clean validated calendar and hourly weather source tables.",
    )
    clean_context.add_argument(
        "--version",
        default=None,
        help="Versioned context output directory (default: cleaned_sources_v1).",
    )
    clean_traffic = subparsers.add_parser(
        "clean-traffic",
        help="Clean validated traffic and reconstruct the half-hour road grid.",
    )
    clean_traffic.add_argument(
        "--version",
        default=None,
        help="Versioned traffic output directory (default: cleaned_sources_v1).",
    )
    merge_sources = subparsers.add_parser(
        "merge-sources",
        help="Align and merge the three verified cleaned source tables.",
    )
    merge_sources.add_argument(
        "--version",
        default=None,
        help="Versioned merged output directory (default: merged_sources_v1).",
    )
    engineer_features = subparsers.add_parser(
        "engineer-features",
        help="Build the verified leakage-safe explanatory feature table.",
    )
    engineer_features.add_argument(
        "--version",
        default=None,
        help=(
            "Versioned feature output directory "
            "(default: engineered_features_v1)."
        ),
    )
    prepare_data = subparsers.add_parser(
        "prepare-data",
        help="Build the verified multi-horizon processed modeling dataset.",
    )
    prepare_data.add_argument(
        "--version",
        default=None,
        help=(
            "Versioned processed output directory "
            "(default: processed_targets_v1)."
        ),
    )
    eda = subparsers.add_parser(
        "eda",
        help="Generate the verified data-quality report and EDA artifacts.",
    )
    eda.add_argument(
        "--version",
        default=None,
        help="Versioned EDA artifact directory (default: eda_v1).",
    )
    prepare_modeling = subparsers.add_parser(
        "prepare-modeling",
        help="Freeze chronological splits and fit training-only preprocessors.",
    )
    prepare_modeling.add_argument(
        "--version",
        default=None,
        help=(
            "Versioned split/preprocessing artifact directory "
            "(default: split_preprocessing_v1)."
        ),
    )
    scratch_linear = subparsers.add_parser(
        "train-scratch-linear",
        help="Prove NumPy gradient descent against sklearn on frozen data.",
    )
    scratch_linear.add_argument(
        "--version",
        default=None,
        help=(
            "Versioned scratch-linear artifact directory "
            "(default: scratch_linear_v1)."
        ),
    )
    classical_regression = subparsers.add_parser(
        "train-classical-regression",
        help="Tune, freeze, and evaluate all classical regression jobs.",
    )
    classical_regression.add_argument(
        "--version",
        default=None,
        help=(
            "Versioned classical-regression artifact directory "
            "(default: classical_regression_v1)."
        ),
    )
    classical_classification = subparsers.add_parser(
        "train-classical-classification",
        help="Tune, calibrate, freeze, and evaluate all classifier jobs.",
    )
    classical_classification.add_argument(
        "--version",
        default=None,
        help=(
            "Versioned classification artifact directory "
            "(default: classical_classification_v1)."
        ),
    )
    classical_registry = subparsers.add_parser(
        "build-classical-registry",
        help="Verify and combine all frozen classical models and scoreboards.",
    )
    classical_registry.add_argument(
        "--version",
        default=None,
        help=(
            "Versioned classical-registry artifact directory "
            "(default: classical_registry_v1)."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run a FlowCast command and return its process exit code."""

    args = build_parser().parse_args(argv)
    settings = load_settings(args.config)
    logger = configure_logging(settings.logs_dir, settings.log_level)
    if args.command == "audit":
        result = run_raw_audit(settings, version=args.version)
        logger.info("Raw audit complete: %s", result.json_path)
        logger.info("Markdown report: %s", result.markdown_path)
        return 0
    if args.command == "validate":
        result = run_validation_pipeline(settings, version=args.version)
        logger.info("Raw validation complete: %s", result.summary_path)
        logger.info("Issue ledger: %s", result.issues_path)
        for dataset, summary in result.summary["datasets"].items():
            logger.info(
                "%s: input=%s valid=%s rejected=%s issues=%s",
                dataset,
                summary["input_rows"],
                summary["valid_rows"],
                summary["rejected_rows"],
                summary["issue_count"],
            )
        return 2 if result.has_dataset_failure else 0
    if args.command == "clean-context":
        result = run_context_cleaning(settings, version=args.version)
        logger.info("Context cleaning complete: %s", result.summary_path)
        logger.info("Generated quality report: %s", result.markdown_path)
        logger.info(
            "calendar=%s weather=%s temperature_imputed=%s visibility_imputed=%s",
            len(result.calendar),
            len(result.weather),
            result.summary["datasets"]["weather"]["imputation"]["temperature"][
                "imputed"
            ],
            result.summary["datasets"]["weather"]["imputation"]["visibility"][
                "imputed"
            ],
        )
        return 0
    if args.command == "clean-traffic":
        result = run_traffic_cleaning(settings, version=args.version)
        traffic = result.summary["dataset"]
        logger.info("Traffic cleaning complete: %s", result.summary_path)
        logger.info("Generated quality report: %s", result.markdown_path)
        logger.info(
            "roads=%s rows=%s inserted=%s congestion_derived=%s",
            traffic["road_count"],
            traffic["output_rows"],
            traffic["grid"]["inserted_windows"],
            traffic["congestion"]["derived_labels"],
        )
        return 0
    if args.command == "merge-sources":
        result = run_source_merge(settings, version=args.version)
        dataset = result.summary["dataset"]
        logger.info("Source merge complete: %s", result.summary_path)
        logger.info("Generated merge report: %s", result.markdown_path)
        logger.info(
            "rows=%s keys=%s weather_missing=%s calendar_missing=%s",
            dataset["output_rows"],
            dataset["output_unique_keys"],
            dataset["joins"]["weather"]["missing"],
            dataset["joins"]["calendar"]["missing"],
        )
        return 0
    if args.command == "engineer-features":
        result = run_feature_engineering(settings, version=args.version)
        dataset = result.summary["dataset"]
        logger.info("Feature engineering complete: %s", result.summary_path)
        logger.info("Feature manifest: %s", result.manifest_path)
        logger.info(
            "rows=%s keys=%s features=%s history_unavailable=%s",
            dataset["output_rows"],
            dataset["output_unique_keys"],
            dataset["feature_count"],
            dataset["history_unavailable_rows"],
        )
        return 0
    if args.command == "prepare-data":
        result = run_processed_data(settings, version=args.version)
        dataset = result.summary["dataset"]
        logger.info("Processed data complete: %s", result.summary_path)
        logger.info("Target and schema manifest: %s", result.manifest_path)
        logger.info(
            "rows=%s keys=%s targets=%s",
            dataset["output_rows"],
            dataset["output_unique_keys"],
            dataset["target_count"],
        )
        return 0
    if args.command == "eda":
        result = run_eda(settings, version=args.version)
        logger.info("EDA complete: %s", result.summary_path)
        logger.info("Data-quality report: %s", result.report_path)
        logger.info(
            "rows=%s context_slices=%s figures=%s",
            result.summary["dataset"]["rows"],
            result.summary["context_aggregates"]["record_count"],
            len(result.figure_paths),
        )
        return 0
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
            "train=%s validation=%s iterations=%s scratch_rmse=%.4f sklearn_rmse=%.4f",
            result.summary["training"]["train_rows"],
            result.summary["training"]["validation_rows"],
            result.summary["training"]["iterations_completed"],
            metrics["scratch"]["rmse"],
            metrics["sklearn"]["rmse"],
        )
        return 0
    if args.command == "train-classical-regression":
        result = run_classical_regression(settings, version=args.version)
        scoreboard = result.summary["scoreboard"]
        volume = [record for record in scoreboard if record["target"] == "volume"]
        logger.info(
            "Classical regression complete: %s",
            result.paths.summary_path,
        )
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
        scoreboard = result.summary["scoreboard"]
        logger.info(
            "Classical classification complete: %s",
            result.paths.summary_path,
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
                for record in scoreboard
            },
        )
        return 0
    if args.command == "build-classical-registry":
        result = run_classical_registry(settings, version=args.version)
        logger.info(
            "Classical registry complete: %s",
            result.paths.summary_path,
        )
        logger.info(
            "entries=%s prediction_rows=%s acceptance=%s",
            result.summary["coverage"]["entry_count"],
            result.summary["coverage"]["prediction_rows"],
            result.summary["acceptance"],
        )
        return 0
    raise RuntimeError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
