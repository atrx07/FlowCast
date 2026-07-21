"""Command-line interface for reproducible FlowCast workflows."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from flowcast.data.audit import run_raw_audit
from flowcast.data.quarantine import run_validation_pipeline
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
    raise RuntimeError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
