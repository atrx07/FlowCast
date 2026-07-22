"""Versioned Step 09 EDA, figure, and data-quality report pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import distributions
from pathlib import Path
import sys
from typing import Any

import pandas as pd

from flowcast.analysis.config import load_eda_config
from flowcast.analysis.figures import generate_figures
from flowcast.analysis.quality import (
    load_verified_quality_sources,
    quality_reconciliation,
)
from flowcast.analysis.report import render_eda_report
from flowcast.analysis.statistics import (
    context_aggregates,
    correlation_analysis,
    descriptive_statistics,
    findings_and_decisions,
    target_distributions,
)
from flowcast.data.artifacts import (
    artifact_record,
    validate_artifact_version,
    write_json,
)
from flowcast.features.inputs import load_verified_processed
from flowcast.settings import Settings


@dataclass(frozen=True)
class EdaArtifacts:
    """Paths, summaries, and aggregate tables produced by one Step 09 run."""

    version: str
    report_dir: Path
    figure_dir: Path
    summary_path: Path
    report_path: Path
    contexts_path: Path
    correlation_path: Path
    covariance_path: Path
    environment_path: Path
    figure_paths: dict[str, Path]
    contexts: pd.DataFrame
    summary: dict[str, Any]


def _write_csv(frame: pd.DataFrame, path: Path, index_label: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        path,
        index=index_label is not None,
        index_label=index_label,
        float_format="%.10f",
        lineterminator="\n",
    )


def _write_environment(path: Path) -> None:
    packages = sorted(
        {
            (distribution.metadata["Name"] or "unknown").lower(): (
                distribution.metadata["Name"] or "unknown",
                distribution.version,
            )
            for distribution in distributions()
        }.values(),
        key=lambda item: item[0].lower(),
    )
    lines = [
        f"python=={sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        *[f"{name}=={version}" for name, version in packages],
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def run_eda(settings: Settings, version: str | None = None) -> EdaArtifacts:
    """Verify processed data, compute EDA, and persist traceable reports."""

    config = load_eda_config(settings)
    selected_version = validate_artifact_version(version or settings.eda_version)
    processed = load_verified_processed(settings)
    quality_sources = load_verified_quality_sources(settings)
    quality = quality_reconciliation(quality_sources)
    frame = processed.frame

    descriptive = descriptive_statistics(frame, config["descriptive_columns"])
    distributions = target_distributions(frame, config["congestion_order"])
    contexts = context_aggregates(
        frame,
        config["context_dimensions"],
        config["congestion_order"],
    )
    correlation = correlation_analysis(
        frame,
        config["correlation_features"],
        config["target_correlation"]["target"],
        config["target_correlation"]["availability"],
        float(config["redundancy_absolute_correlation"]),
    )
    findings, decisions, limitations = findings_and_decisions(
        descriptive,
        distributions,
        contexts,
        correlation,
    )

    report_dir = settings.artifacts_dir / "reports" / selected_version
    figure_dir = settings.artifacts_dir / "figures" / selected_version
    summary_path = report_dir / "summary.json"
    report_path = report_dir / "data_quality.md"
    contexts_path = report_dir / "context_aggregates.csv"
    correlation_path = report_dir / "correlation.csv"
    covariance_path = report_dir / "covariance.csv"
    environment_path = report_dir / "environment.txt"
    _write_csv(contexts, contexts_path)
    _write_csv(correlation.correlation, correlation_path, index_label="feature")
    _write_csv(correlation.covariance, covariance_path, index_label="feature")
    _write_environment(environment_path)
    figure_paths = generate_figures(
        frame,
        contexts,
        correlation.correlation,
        config,
        figure_dir,
    )

    upstream_paths = {
        "raw_audit": quality_sources.audit_path,
        "validation": quality_sources.validation_path,
        "context_cleaning": quality_sources.context_path,
        "traffic_cleaning": quality_sources.traffic_path,
        "merge": quality_sources.merge_path,
        "features": quality_sources.feature_path,
        "processed": quality_sources.processed_path,
    }
    summary: dict[str, Any] = {
        "contract_version": str(config["eda_contract_version"]),
        "eda_version": selected_version,
        "input_processed_version": settings.processed_version,
        "configuration": {
            "base": artifact_record(settings.config_path, settings),
            "eda": artifact_record(settings.eda_config_path, settings),
            "features": artifact_record(settings.features_config_path, settings),
        },
        "input_processed": {
            "dataset": artifact_record(processed.path, settings),
            "manifest": artifact_record(processed.manifest_path, settings),
            "summary": artifact_record(processed.summary_path, settings),
        },
        "input_summaries": {
            name: artifact_record(path, settings)
            for name, path in upstream_paths.items()
        },
        "dataset": {
            "rows": len(frame),
            "columns": len(frame.columns),
            "road_count": int(frame["road_id"].nunique()),
            "timestamp_start": frame["timestamp"].min().isoformat(),
            "timestamp_end": frame["timestamp"].max().isoformat(),
        },
        "environment_artifact": artifact_record(environment_path, settings),
        "quality_reconciliation": quality,
        "descriptive_statistics": descriptive,
        "distributions": distributions,
        "context_aggregates": {
            "dimensions": list(config["context_dimensions"]),
            "record_count": len(contexts),
            "artifact": artifact_record(contexts_path, settings),
        },
        "correlation": {
            "features": list(config["correlation_features"]),
            "feature_count": len(config["correlation_features"]),
            "target": str(config["target_correlation"]["target"]),
            "availability": str(config["target_correlation"]["availability"]),
            "redundancy_threshold": float(
                config["redundancy_absolute_correlation"]
            ),
            "redundant_pairs": correlation.redundant_pairs,
            "target_correlations": correlation.target_correlations,
            "correlation_artifact": artifact_record(correlation_path, settings),
            "covariance_artifact": artifact_record(covariance_path, settings),
        },
        "findings": findings,
        "modelling_decisions": decisions,
        "limitations": limitations,
        "figures": {
            name: artifact_record(path, settings)
            for name, path in figure_paths.items()
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        render_eda_report(summary),
        encoding="utf-8",
        newline="\n",
    )
    summary["report_artifact"] = artifact_record(report_path, settings)
    write_json(summary, summary_path)
    return EdaArtifacts(
        version=selected_version,
        report_dir=report_dir,
        figure_dir=figure_dir,
        summary_path=summary_path,
        report_path=report_path,
        contexts_path=contexts_path,
        correlation_path=correlation_path,
        covariance_path=covariance_path,
        environment_path=environment_path,
        figure_paths=figure_paths,
        contexts=contexts,
        summary=summary,
    )
