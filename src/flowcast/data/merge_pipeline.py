"""Versioned Step 06 pipeline for aligned, merged source data."""

from __future__ import annotations

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
from flowcast.data.cleaned_inputs import load_cleaned_sources
from flowcast.data.merge import merge_cleaned_sources
from flowcast.data.quality_report import render_source_merge_markdown
from flowcast.settings import Settings


@dataclass(frozen=True)
class SourceMergeArtifacts:
    """Paths, merged frame, and summary for one Step 06 run."""

    version: str
    output_dir: Path
    quality_dir: Path
    merged_path: Path
    summary_path: Path
    markdown_path: Path
    merged: pd.DataFrame
    summary: dict[str, Any]


def load_merge_config(settings: Settings) -> dict[str, Any]:
    """Load and validate the configured source-merge policy."""

    with settings.cleaning_config_path.open("r", encoding="utf-8") as handle:
        config: dict[str, Any] = yaml.safe_load(handle)
    if config.get("merge_contract_version") != "source_merge_v1":
        raise ValueError("Unsupported source merge contract version")
    return config


def run_source_merge(
    settings: Settings,
    version: str | None = None,
) -> SourceMergeArtifacts:
    """Hash-verify, align, merge, and persist all three cleaned sources."""

    selected_version = validate_artifact_version(version or settings.merge_version)
    config = load_merge_config(settings)
    sources = load_cleaned_sources(settings)
    result = merge_cleaned_sources(
        sources.traffic,
        sources.weather,
        sources.calendar,
        config["merge"],
    )

    output_dir = settings.interim_dir / selected_version
    merged_path = output_dir / "merged.parquet"
    write_parquet(result.frame, merged_path)
    quality_dir = settings.artifacts_dir / "quality" / selected_version
    summary_path = quality_dir / "summary.json"
    markdown_path = quality_dir / "summary.md"
    summary: dict[str, Any] = {
        "contract_version": str(config["merge_contract_version"]),
        "merge_version": selected_version,
        "input_cleaning_version": settings.cleaning_version,
        "configuration": {
            "base": artifact_record(settings.config_path, settings),
            "cleaning": artifact_record(settings.cleaning_config_path, settings),
        },
        "input_summaries": {
            "context": artifact_record(sources.context_summary_path, settings),
            "traffic": artifact_record(sources.traffic_summary_path, settings),
        },
        "input_artifacts": {
            "traffic": artifact_record(sources.traffic_path, settings),
            "weather": artifact_record(sources.weather_path, settings),
            "calendar": artifact_record(sources.calendar_path, settings),
        },
        "dataset": {
            **result.summary,
            "merged_artifact": artifact_record(merged_path, settings),
        },
    }
    write_json(summary, summary_path)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(
        render_source_merge_markdown(summary),
        encoding="utf-8",
        newline="\n",
    )
    return SourceMergeArtifacts(
        version=selected_version,
        output_dir=output_dir,
        quality_dir=quality_dir,
        merged_path=merged_path,
        summary_path=summary_path,
        markdown_path=markdown_path,
        merged=result.frame,
        summary=summary,
    )
