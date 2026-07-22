"""Command-line interface for reproducible FlowCast workflows."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from flowcast.data.audit import run_raw_audit
from flowcast.data.clean_context import run_context_cleaning
from flowcast.data.quarantine import run_validation_pipeline
from flowcast.data.merge_pipeline import run_source_merge
from flowcast.data.traffic_pipeline import run_traffic_cleaning
from flowcast.logging_config import configure_logging
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
    raise RuntimeError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
