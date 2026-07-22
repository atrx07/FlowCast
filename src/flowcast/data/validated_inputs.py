"""Verified access to versioned validation artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from flowcast.data.artifacts import verify_artifact_record
from flowcast.data.quarantine import run_validation_pipeline
from flowcast.settings import Settings


def validated_summary(settings: Settings) -> tuple[Path, dict[str, Any]]:
    """Load the configured successful validation summary, creating it if absent."""

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


def verified_validated_table(
    settings: Settings,
    summary: dict[str, Any],
    dataset: str,
) -> tuple[Path, pd.DataFrame]:
    """Read one hash-verified validated source table."""

    path = settings.interim_dir / settings.validation_version / f"{dataset}.parquet"
    record = summary["datasets"][dataset]["validated_artifact"]
    verify_artifact_record(path, record, settings)
    return path, pd.read_parquet(path)


def verified_issue_ledger(
    settings: Settings,
    summary: dict[str, Any],
) -> tuple[Path, pd.DataFrame]:
    """Read the hash-verified validation issue ledger."""

    path = settings.quarantine_dir / settings.validation_version / "issues.parquet"
    verify_artifact_record(path, summary["issues_artifact"], settings)
    return path, pd.read_parquet(path)
