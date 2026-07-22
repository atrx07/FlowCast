"""Hash-verified access to the three versioned cleaned source tables."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from flowcast.data.artifacts import verify_artifact_record
from flowcast.data.clean_context import run_context_cleaning
from flowcast.data.traffic_pipeline import run_traffic_cleaning
from flowcast.settings import Settings


@dataclass(frozen=True)
class CleanedSources:
    """Verified cleaned inputs and the summaries that record their hashes."""

    traffic: pd.DataFrame
    weather: pd.DataFrame
    calendar: pd.DataFrame
    traffic_path: Path
    weather_path: Path
    calendar_path: Path
    traffic_summary_path: Path
    context_summary_path: Path
    traffic_summary: dict[str, Any]
    context_summary: dict[str, Any]


def _read_summary(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Cleaning summary is missing: {path}")
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return payload


def load_cleaned_sources(settings: Settings) -> CleanedSources:
    """Verify cleaning configuration and table hashes, then load all sources."""

    quality_dir = settings.artifacts_dir / "quality" / settings.cleaning_version
    context_summary_path = quality_dir / "summary.json"
    traffic_summary_path = quality_dir / "traffic_summary.json"
    if not context_summary_path.is_file():
        run_context_cleaning(settings)
    if not traffic_summary_path.is_file():
        run_traffic_cleaning(settings)

    context = _read_summary(context_summary_path)
    traffic = _read_summary(traffic_summary_path)
    if context.get("contract_version") != "context_cleaning_v1":
        raise RuntimeError("Unsupported cleaned context contract")
    if traffic.get("contract_version") != "traffic_cleaning_v1":
        raise RuntimeError("Unsupported cleaned traffic contract")
    if (
        context.get("cleaning_version") != settings.cleaning_version
        or traffic.get("cleaning_version") != settings.cleaning_version
    ):
        raise RuntimeError("Cleaned input version does not match configuration")

    verify_artifact_record(
        settings.cleaning_config_path,
        context["configuration"]["cleaning"],
        settings,
    )
    verify_artifact_record(
        settings.cleaning_config_path,
        traffic["configuration"]["cleaning"],
        settings,
    )
    input_dir = settings.interim_dir / settings.cleaning_version
    calendar_path = input_dir / "calendar.parquet"
    weather_path = input_dir / "weather.parquet"
    traffic_path = input_dir / "traffic.parquet"
    verify_artifact_record(
        calendar_path, context["datasets"]["calendar"]["cleaned_artifact"], settings
    )
    verify_artifact_record(
        weather_path, context["datasets"]["weather"]["cleaned_artifact"], settings
    )
    verify_artifact_record(
        traffic_path, traffic["dataset"]["cleaned_artifact"], settings
    )
    return CleanedSources(
        traffic=pd.read_parquet(traffic_path),
        weather=pd.read_parquet(weather_path),
        calendar=pd.read_parquet(calendar_path),
        traffic_path=traffic_path,
        weather_path=weather_path,
        calendar_path=calendar_path,
        traffic_summary_path=traffic_summary_path,
        context_summary_path=context_summary_path,
        traffic_summary=traffic,
        context_summary=context,
    )
