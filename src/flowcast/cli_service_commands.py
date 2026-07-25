"""Inference and reporting CLI registration and dispatch."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from flowcast.inference.artifacts import persist_prediction_batch
from flowcast.inference.predictor import Predictor
from flowcast.reports.export import build_prediction_reports
from flowcast.settings import Settings


def register_service_parsers(subparsers: Any) -> None:
    """Register the Step 17 inference and reporting commands."""

    predict = subparsers.add_parser(
        "predict",
        help="Load frozen models and persist validated multi-target forecasts.",
    )
    predict.add_argument(
        "--roads",
        nargs="+",
        default=None,
        help="Road IDs (default: all 25 corridor roads).",
    )
    predict.add_argument(
        "--origin",
        default=None,
        help="ISO origin timestamp (default: latest common processed origin).",
    )
    predict.add_argument(
        "--horizons",
        nargs="+",
        type=int,
        default=None,
        help="Forecast windows from 1 to 4 (default: all).",
    )
    predict.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="Inference device (default: cpu).",
    )
    predict.add_argument(
        "--export-reports",
        action="store_true",
        help="Also export verified CSV and self-contained HTML reports.",
    )

    reports = subparsers.add_parser(
        "build-reports",
        help="Export CSV and HTML from a verified prediction manifest.",
    )
    reports.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Path to a persisted inference manifest.json.",
    )


def run_service_command(
    args: argparse.Namespace,
    settings: Settings,
    logger: logging.Logger,
) -> int | None:
    """Dispatch one inference/report command or return None."""

    if args.command == "predict":
        predictor = Predictor(settings, device=args.device)
        request = predictor.build_request(
            road_ids=args.roads,
            origin_timestamp=args.origin,
            horizons=args.horizons,
        )
        result = predictor.predict(request)
        paths = persist_prediction_batch(result, settings)
        logger.info("Prediction batch complete: %s", paths.manifest_path)
        logger.info(
            "roads=%s horizons=%s rows=%s device=%s init=%.3fs "
            "prediction=%.3fs cold_total=%.3fs",
            len(request.road_ids),
            len(request.horizons),
            len(result.frame),
            request.device,
            result.initialization_seconds,
            result.prediction_seconds,
            result.total_seconds,
        )
        if args.export_reports:
            reports = build_prediction_reports(settings, paths.manifest_path)
            logger.info("CSV report: %s", reports.csv_path)
            logger.info("HTML report: %s", reports.html_path)
        return 0
    if args.command == "build-reports":
        reports = build_prediction_reports(settings, args.manifest)
        logger.info("CSV report: %s", reports.csv_path)
        logger.info("HTML report: %s", reports.html_path)
        return 0
    return None
