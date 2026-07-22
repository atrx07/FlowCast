"""Hash-verified access to the merged Step 06 artifact."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from flowcast.data.artifacts import verify_artifact_record
from flowcast.data.merge_pipeline import run_source_merge
from flowcast.settings import Settings


@dataclass(frozen=True)
class VerifiedMergedInput:
    """Loaded merge artifact with its verified lineage summary."""

    frame: pd.DataFrame
    path: Path
    summary_path: Path
    summary: dict[str, Any]


def load_verified_merged(settings: Settings) -> VerifiedMergedInput:
    """Verify configuration, summary, artifact hash, and key contract before use."""

    summary_path = (
        settings.artifacts_dir / "quality" / settings.merge_version / "summary.json"
    )
    if not summary_path.is_file():
        run_source_merge(settings)
    summary: dict[str, Any] = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("contract_version") != "source_merge_v1":
        raise RuntimeError("Unsupported merged input contract")
    if summary.get("merge_version") != settings.merge_version:
        raise RuntimeError("Merged input version does not match configuration")
    verify_artifact_record(
        settings.config_path,
        summary["configuration"]["base"],
        settings,
    )
    verify_artifact_record(
        settings.cleaning_config_path,
        summary["configuration"]["cleaning"],
        settings,
    )
    path = settings.interim_dir / settings.merge_version / "merged.parquet"
    verify_artifact_record(path, summary["dataset"]["merged_artifact"], settings)
    frame = pd.read_parquet(path)
    expected_rows = int(summary["dataset"]["output_rows"])
    expected_keys = int(summary["dataset"]["output_unique_keys"])
    if len(frame) != expected_rows:
        raise RuntimeError("Merged input row count does not match its summary")
    unique_keys = len(frame.drop_duplicates(["road_id", "timestamp"]))
    if unique_keys != expected_keys or unique_keys != len(frame):
        raise RuntimeError("Merged input road/timestamp key contract failed")
    return VerifiedMergedInput(
        frame=frame,
        path=path,
        summary_path=summary_path,
        summary=summary,
    )
