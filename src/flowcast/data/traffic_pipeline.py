"""Versioned Step 05 pipeline for trusted reconstructed traffic data."""

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
from flowcast.data.clean_traffic import clean_traffic
from flowcast.data.quality_report import render_traffic_cleaning_markdown
from flowcast.data.validated_inputs import (
    validated_summary,
    verified_issue_ledger,
    verified_validated_table,
)
from flowcast.settings import Settings


@dataclass(frozen=True)
class TrafficCleaningArtifacts:
    """Paths, cleaned traffic frame, and summary for one Step 05 run."""

    version: str
    output_dir: Path
    quality_dir: Path
    traffic_path: Path
    summary_path: Path
    markdown_path: Path
    traffic: pd.DataFrame
    summary: dict[str, Any]


def load_traffic_cleaning_config(settings: Settings) -> dict[str, Any]:
    """Load and validate the configured traffic-cleaning policy."""

    with settings.cleaning_config_path.open("r", encoding="utf-8") as handle:
        config: dict[str, Any] = yaml.safe_load(handle)
    if config.get("traffic_contract_version") != "traffic_cleaning_v1":
        raise ValueError("Unsupported traffic cleaning contract version")
    return config


def run_traffic_cleaning(
    settings: Settings,
    version: str | None = None,
) -> TrafficCleaningArtifacts:
    """Clean validated traffic and persist its versioned quality evidence."""

    selected_version = validate_artifact_version(
        version or settings.cleaning_version
    )
    cleaning_config = load_traffic_cleaning_config(settings)
    validation_summary_path, validation = validated_summary(settings)
    traffic_input_path, traffic_input = verified_validated_table(
        settings, validation, "traffic"
    )
    issues_path, issues = verified_issue_ledger(settings, validation)
    traffic_issues = issues.loc[issues["dataset"].eq("traffic")].copy()
    result = clean_traffic(
        traffic_input,
        traffic_issues,
        cleaning_config["traffic"],
    )

    output_dir = settings.interim_dir / selected_version
    traffic_path = output_dir / "traffic.parquet"
    write_parquet(result.frame, traffic_path)

    quality_dir = settings.artifacts_dir / "quality" / selected_version
    summary_path = quality_dir / "traffic_summary.json"
    markdown_path = quality_dir / "traffic_summary.md"
    summary: dict[str, Any] = {
        "contract_version": str(cleaning_config["traffic_contract_version"]),
        "cleaning_version": selected_version,
        "input_validation_version": settings.validation_version,
        "configuration": {
            "cleaning": artifact_record(settings.cleaning_config_path, settings),
        },
        "input_validation_summary": artifact_record(
            validation_summary_path, settings
        ),
        "input_issue_ledger": artifact_record(issues_path, settings),
        "dataset": {
            **result.summary,
            "input_artifact": artifact_record(traffic_input_path, settings),
            "cleaned_artifact": artifact_record(traffic_path, settings),
        },
    }
    write_json(summary, summary_path)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(
        render_traffic_cleaning_markdown(summary),
        encoding="utf-8",
        newline="\n",
    )
    return TrafficCleaningArtifacts(
        version=selected_version,
        output_dir=output_dir,
        quality_dir=quality_dir,
        traffic_path=traffic_path,
        summary_path=summary_path,
        markdown_path=markdown_path,
        traffic=result.frame,
        summary=summary,
    )
