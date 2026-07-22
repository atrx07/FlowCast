"""Hash-verified access to upstream feature-pipeline artifacts."""

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


@dataclass(frozen=True)
class VerifiedFeatureInput:
    """Loaded Step 07 feature artifact with verified manifest and summary."""

    frame: pd.DataFrame
    path: Path
    manifest_path: Path
    summary_path: Path
    manifest: dict[str, Any]
    summary: dict[str, Any]


@dataclass(frozen=True)
class VerifiedProcessedInput:
    """Loaded Step 08 processed artifact with verified schema and lineage."""

    frame: pd.DataFrame
    path: Path
    manifest_path: Path
    summary_path: Path
    manifest: dict[str, Any]
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


def load_verified_features(settings: Settings) -> VerifiedFeatureInput:
    """Verify the complete Step 07 artifact contract before target creation."""

    summary_path = (
        settings.artifacts_dir / "quality" / settings.feature_version / "summary.json"
    )
    if not summary_path.is_file():
        from flowcast.features.pipeline import run_feature_engineering

        run_feature_engineering(settings)
    summary: dict[str, Any] = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("contract_version") != "explanatory_features_v1":
        raise RuntimeError("Unsupported explanatory feature input contract")
    if summary.get("feature_version") != settings.feature_version:
        raise RuntimeError("Feature input version does not match configuration")
    verify_artifact_record(
        settings.config_path,
        summary["configuration"]["base"],
        settings,
    )
    verify_artifact_record(
        settings.features_config_path,
        summary["configuration"]["features"],
        settings,
    )

    path = settings.interim_dir / settings.feature_version / "features.parquet"
    manifest_path = (
        settings.artifacts_dir / "features" / settings.feature_version / "manifest.json"
    )
    verify_artifact_record(path, summary["dataset"]["feature_artifact"], settings)
    verify_artifact_record(
        manifest_path,
        summary["dataset"]["manifest_artifact"],
        settings,
    )
    manifest: dict[str, Any] = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )
    if manifest.get("contract_version") != summary["contract_version"]:
        raise RuntimeError("Feature manifest contract does not match its summary")
    if manifest.get("feature_version") != settings.feature_version:
        raise RuntimeError("Feature manifest version does not match configuration")
    verify_artifact_record(
        settings.features_config_path,
        manifest["configuration"],
        settings,
    )
    verify_artifact_record(path, manifest["output_artifact"], settings)

    frame = pd.read_parquet(path)
    expected_rows = int(summary["dataset"]["output_rows"])
    expected_keys = int(summary["dataset"]["output_unique_keys"])
    if len(frame) != expected_rows:
        raise RuntimeError("Feature input row count does not match its summary")
    unique_keys = len(frame.drop_duplicates(["road_id", "timestamp"]))
    if unique_keys != expected_keys or unique_keys != len(frame):
        raise RuntimeError("Feature input road/timestamp key contract failed")
    for definition in manifest["features"]:
        name = str(definition["name"])
        if name not in frame or str(frame[name].dtype) != str(definition["dtype"]):
            raise RuntimeError(f"Feature manifest dtype contract failed: {name}")
    return VerifiedFeatureInput(
        frame=frame,
        path=path,
        manifest_path=manifest_path,
        summary_path=summary_path,
        manifest=manifest,
        summary=summary,
    )


def load_verified_processed(settings: Settings) -> VerifiedProcessedInput:
    """Verify the complete Step 08 processed contract before analysis."""

    summary_path = (
        settings.artifacts_dir
        / "quality"
        / settings.processed_version
        / "summary.json"
    )
    if not summary_path.is_file():
        from flowcast.features.processed_pipeline import run_processed_data

        run_processed_data(settings)
    summary: dict[str, Any] = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("contract_version") != "multi_horizon_targets_v1":
        raise RuntimeError("Unsupported processed input contract")
    if summary.get("processed_version") != settings.processed_version:
        raise RuntimeError("Processed input version does not match configuration")
    for name, path in {
        "base": settings.config_path,
        "features": settings.features_config_path,
    }.items():
        verify_artifact_record(path, summary["configuration"][name], settings)

    path = settings.processed_dir / settings.processed_version / "dataset.parquet"
    manifest_path = (
        settings.artifacts_dir
        / "features"
        / settings.processed_version
        / "manifest.json"
    )
    verify_artifact_record(path, summary["dataset"]["processed_artifact"], settings)
    verify_artifact_record(
        manifest_path,
        summary["dataset"]["manifest_artifact"],
        settings,
    )
    manifest: dict[str, Any] = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )
    if manifest.get("contract_version") != summary["contract_version"]:
        raise RuntimeError("Processed manifest contract does not match its summary")
    if manifest.get("processed_version") != settings.processed_version:
        raise RuntimeError("Processed manifest version does not match configuration")
    for name, config_path in {
        "base": settings.config_path,
        "features": settings.features_config_path,
    }.items():
        verify_artifact_record(
            config_path,
            manifest["configuration"][name],
            settings,
        )
    verify_artifact_record(path, manifest["output_artifact"], settings)
    feature_summary_path = (
        settings.artifacts_dir
        / "quality"
        / settings.feature_version
        / "summary.json"
    )
    feature_manifest_path = (
        settings.artifacts_dir
        / "features"
        / settings.feature_version
        / "manifest.json"
    )
    verify_artifact_record(
        feature_summary_path,
        manifest["input_summary"],
        settings,
    )
    verify_artifact_record(
        feature_manifest_path,
        manifest["input_manifest"],
        settings,
    )

    frame = pd.read_parquet(path)
    if len(frame) != int(manifest["row_count"]):
        raise RuntimeError("Processed input row count does not match its manifest")
    unique_keys = len(frame.drop_duplicates(["road_id", "timestamp"]))
    if unique_keys != int(summary["dataset"]["output_unique_keys"]):
        raise RuntimeError("Processed input road/timestamp key contract failed")
    expected_columns = [record["name"] for record in manifest["columns"]]
    if list(frame.columns) != expected_columns:
        raise RuntimeError("Processed input column order does not match its manifest")
    for record in manifest["columns"]:
        name = str(record["name"])
        if str(frame[name].dtype) != str(record["dtype"]):
            raise RuntimeError(f"Processed manifest dtype contract failed: {name}")
    return VerifiedProcessedInput(
        frame=frame,
        path=path,
        manifest_path=manifest_path,
        summary_path=summary_path,
        manifest=manifest,
        summary=summary,
    )
