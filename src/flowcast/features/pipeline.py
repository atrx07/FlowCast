"""Versioned Step 07 pipeline for explanatory feature artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from flowcast.data.artifacts import (
    artifact_record,
    validate_artifact_version,
    write_json,
    write_parquet,
)
from flowcast.data.quality_report import render_feature_engineering_markdown
from flowcast.features.config import load_feature_config
from flowcast.features.engineering import engineer_features
from flowcast.features.inputs import load_verified_merged
from flowcast.settings import Settings


@dataclass(frozen=True)
class FeatureArtifacts:
    """Paths, feature table, manifest, and summary for one Step 07 run."""

    version: str
    output_dir: Path
    feature_path: Path
    manifest_path: Path
    summary_path: Path
    markdown_path: Path
    frame: pd.DataFrame
    manifest: dict[str, Any]
    summary: dict[str, Any]


def _python_scalar(value: Any) -> Any:
    if pd.isna(value):
        return None
    return value.item() if hasattr(value, "item") else value


def _feature_stat(series: pd.Series) -> dict[str, Any]:
    record: dict[str, Any] = {
        "dtype": str(series.dtype),
        "null_count": int(series.isna().sum()),
    }
    present = series.dropna()
    if is_bool_dtype(series.dtype):
        record["true_count"] = int(present.astype(bool).sum())
        record["false_count"] = int(len(present) - record["true_count"])
    elif is_numeric_dtype(series.dtype) and not present.empty:
        record["minimum"] = _python_scalar(present.min())
        record["maximum"] = _python_scalar(present.max())
    else:
        record["value_counts"] = {
            str(key): int(value)
            for key, value in present.value_counts(dropna=False).sort_index().items()
        }
    return record


def run_feature_engineering(
    settings: Settings,
    version: str | None = None,
) -> FeatureArtifacts:
    """Verify merged data, engineer features, and persist traceable artifacts."""

    config = load_feature_config(settings)
    selected_version = validate_artifact_version(version or settings.feature_version)
    merged = load_verified_merged(settings)
    result = engineer_features(merged.frame, config)
    frame = result.frame

    output_dir = settings.interim_dir / selected_version
    feature_path = output_dir / "features.parquet"
    write_parquet(frame, feature_path)
    definitions = []
    for definition in result.definitions:
        record = asdict(definition)
        record["source_columns"] = list(record["source_columns"])
        record["dtype"] = str(frame[definition.name].dtype)
        record["version"] = selected_version
        definitions.append(record)

    feature_dir = settings.artifacts_dir / "features" / selected_version
    manifest_path = feature_dir / "manifest.json"
    manifest: dict[str, Any] = {
        "contract_version": str(config["feature_contract_version"]),
        "feature_version": selected_version,
        "input_merge_version": settings.merge_version,
        "forecast_horizons_reserved": [
            int(value) for value in config["forecast_horizons_reserved"]
        ],
        "key_columns": ["road_id", "timestamp"],
        "feature_count": len(definitions),
        "configuration": artifact_record(settings.features_config_path, settings),
        "input_artifact": artifact_record(merged.path, settings),
        "output_artifact": artifact_record(feature_path, settings),
        "features": definitions,
    }
    write_json(manifest, manifest_path)

    unique_keys = len(frame.drop_duplicates(["road_id", "timestamp"]))
    history_available = frame["history_available"].fillna(False).astype(bool)
    history_by_road = (
        (~history_available).groupby(frame["road_id"], sort=True).sum().astype(int)
    )
    feature_stats = {
        definition.name: _feature_stat(frame[definition.name])
        for definition in result.definitions
    }
    quality_dir = settings.artifacts_dir / "quality" / selected_version
    summary_path = quality_dir / "summary.json"
    markdown_path = quality_dir / "summary.md"
    summary: dict[str, Any] = {
        "contract_version": str(config["feature_contract_version"]),
        "feature_version": selected_version,
        "input_merge_version": settings.merge_version,
        "configuration": {
            "base": artifact_record(settings.config_path, settings),
            "features": artifact_record(settings.features_config_path, settings),
        },
        "input_summary": artifact_record(merged.summary_path, settings),
        "input_artifact": artifact_record(merged.path, settings),
        "dataset": {
            "input_rows": len(merged.frame),
            "output_rows": len(frame),
            "row_count_change": len(frame) - len(merged.frame),
            "road_count": int(frame["road_id"].nunique()),
            "output_unique_keys": unique_keys,
            "duplicate_output_keys": len(frame) - unique_keys,
            "feature_count": len(definitions),
            "history_available_rows": int(history_available.sum()),
            "history_unavailable_rows": int((~history_available).sum()),
            "history_unavailable_by_road": {
                str(key): int(value) for key, value in history_by_road.items()
            },
            "feature_null_counts": {
                name: int(record["null_count"])
                for name, record in feature_stats.items()
            },
            "feature_stats": feature_stats,
            "feature_artifact": artifact_record(feature_path, settings),
            "manifest_artifact": artifact_record(manifest_path, settings),
        },
    }
    write_json(summary, summary_path)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(
        render_feature_engineering_markdown(summary),
        encoding="utf-8",
        newline="\n",
    )
    return FeatureArtifacts(
        version=selected_version,
        output_dir=output_dir,
        feature_path=feature_path,
        manifest_path=manifest_path,
        summary_path=summary_path,
        markdown_path=markdown_path,
        frame=frame,
        manifest=manifest,
        summary=summary,
    )
