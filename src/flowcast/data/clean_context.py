"""Versioned Step 04 pipeline for trusted calendar and weather context."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from flowcast.data.artifacts import (
    artifact_record,
    validate_artifact_version,
    write_json,
    write_parquet,
)
from flowcast.data.audit import sha256_file
from flowcast.data.clean_calendar import clean_calendar
from flowcast.data.clean_weather import clean_weather
from flowcast.data.contracts import load_contract_bundle
from flowcast.data.quality_report import render_context_cleaning_markdown
from flowcast.data.quarantine import run_validation_pipeline
from flowcast.settings import Settings


@dataclass(frozen=True)
class ContextCleaningArtifacts:
    """Paths, cleaned frames, and summary for one Step 04 run."""

    version: str
    output_dir: Path
    quality_dir: Path
    calendar_path: Path
    weather_path: Path
    summary_path: Path
    markdown_path: Path
    calendar: pd.DataFrame
    weather: pd.DataFrame
    summary: dict[str, Any]


def load_cleaning_config(settings: Settings) -> dict[str, Any]:
    """Load the versioned cleaning policy configuration."""

    with settings.cleaning_config_path.open("r", encoding="utf-8") as handle:
        config: dict[str, Any] = yaml.safe_load(handle)
    if config.get("contract_version") != "context_cleaning_v1":
        raise ValueError("Unsupported context cleaning contract version")
    return config


def _validated_summary(settings: Settings) -> tuple[Path, dict[str, Any]]:
    summary_path = (
        settings.quarantine_dir / settings.validation_version / "summary.json"
    )
    if not summary_path.is_file():
        run_validation_pipeline(settings)
    payload: dict[str, Any] = json.loads(summary_path.read_text(encoding="utf-8"))
    if payload.get("validation_version") != settings.validation_version:
        raise RuntimeError("Validated input version does not match configuration")
    if payload.get("dataset_failure"):
        raise RuntimeError("Validated input contains a dataset-level failure")
    return summary_path, payload


def _verified_validated_table(
    settings: Settings,
    summary: dict[str, Any],
    dataset: str,
) -> tuple[Path, pd.DataFrame]:
    path = settings.interim_dir / settings.validation_version / f"{dataset}.parquet"
    record = summary["datasets"][dataset]["validated_artifact"]
    if not path.is_file():
        raise FileNotFoundError(f"Validated input artifact is missing: {path}")
    if path.stat().st_size != int(record["bytes"]):
        raise RuntimeError(f"Validated input byte count changed: {path}")
    if sha256_file(path, settings.hash_chunk_size) != str(record["sha256"]):
        raise RuntimeError(f"Validated input SHA-256 changed: {path}")
    return path, pd.read_parquet(path)


def run_context_cleaning(
    settings: Settings,
    version: str | None = None,
) -> ContextCleaningArtifacts:
    """Clean validated calendar/weather data and persist quality evidence."""

    selected_version = validate_artifact_version(
        version or settings.cleaning_version
    )
    cleaning_config = load_cleaning_config(settings)
    validation_summary_path, validation_summary = _validated_summary(settings)
    calendar_input_path, calendar_input = _verified_validated_table(
        settings, validation_summary, "calendar"
    )
    weather_input_path, weather_input = _verified_validated_table(
        settings, validation_summary, "weather"
    )

    contracts = load_contract_bundle(settings)
    normalization_map: dict[str, str] = contracts["datasets"]["weather"][
        "categorical"
    ]["weather_condition"]["normalization_map"]
    calendar_result = clean_calendar(calendar_input, cleaning_config["calendar"])
    weather_result = clean_weather(
        weather_input,
        normalization_map,
        cleaning_config["weather"],
    )

    output_dir = settings.interim_dir / selected_version
    calendar_path = output_dir / "calendar.parquet"
    weather_path = output_dir / "weather.parquet"
    write_parquet(calendar_result.frame, calendar_path)
    write_parquet(weather_result.frame, weather_path)

    quality_dir = settings.artifacts_dir / "quality" / selected_version
    summary_path = quality_dir / "summary.json"
    markdown_path = quality_dir / "summary.md"
    summary: dict[str, Any] = {
        "contract_version": str(cleaning_config["contract_version"]),
        "cleaning_version": selected_version,
        "input_validation_version": settings.validation_version,
        "configuration": {
            "cleaning": artifact_record(settings.cleaning_config_path, settings),
            "data_contracts": artifact_record(
                settings.data_contracts_path, settings
            ),
        },
        "input_validation_summary": artifact_record(
            validation_summary_path, settings
        ),
        "datasets": {
            "calendar": {
                **calendar_result.summary,
                "input_artifact": artifact_record(calendar_input_path, settings),
                "cleaned_artifact": artifact_record(calendar_path, settings),
            },
            "weather": {
                **weather_result.summary,
                "input_artifact": artifact_record(weather_input_path, settings),
                "cleaned_artifact": artifact_record(weather_path, settings),
            },
        },
    }
    write_json(summary, summary_path)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(
        render_context_cleaning_markdown(summary),
        encoding="utf-8",
        newline="\n",
    )
    return ContextCleaningArtifacts(
        version=selected_version,
        output_dir=output_dir,
        quality_dir=quality_dir,
        calendar_path=calendar_path,
        weather_path=weather_path,
        summary_path=summary_path,
        markdown_path=markdown_path,
        calendar=calendar_result.frame,
        weather=weather_result.frame,
        summary=summary,
    )
