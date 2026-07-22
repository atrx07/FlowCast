"""Persist validated rows, rejected rows, and machine-readable issue lineage."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from flowcast.data.artifacts import (
    artifact_record,
    validate_artifact_version,
    write_json,
    write_parquet,
)
from flowcast.data.contracts import (
    ValidationResult,
    load_contract_bundle,
    portable_path,
)
from flowcast.data.ingest import validate_raw_sources
from flowcast.settings import Settings


_ISSUE_COLUMNS = [
    "dataset",
    "source_file",
    "source_row",
    "field",
    "rejected_value",
    "reason_code",
    "disposition",
    "message",
    "retained_source_row",
]


@dataclass(frozen=True)
class ValidationArtifacts:
    """Paths and results produced by one versioned validation run."""

    version: str
    interim_dir: Path
    quarantine_dir: Path
    summary_path: Path
    issues_path: Path
    valid_paths: dict[str, Path]
    rejected_paths: dict[str, Path]
    results: dict[str, ValidationResult]
    summary: dict[str, Any]

    @property
    def has_dataset_failure(self) -> bool:
        """Return whether any complete source dataset failed its schema contract."""

        return any(result.has_dataset_failure for result in self.results.values())


def _issue_frame(results: dict[str, ValidationResult]) -> pd.DataFrame:
    records = [
        issue.as_record()
        for result in results.values()
        for issue in result.issues
    ]
    frame = pd.DataFrame.from_records(records, columns=_ISSUE_COLUMNS)
    frame["source_row"] = pd.array(frame["source_row"], dtype="Int64")
    frame["retained_source_row"] = pd.array(
        frame["retained_source_row"], dtype="Int64"
    )
    return frame


def persist_validation_results(
    settings: Settings,
    results: dict[str, ValidationResult],
    version: str,
) -> ValidationArtifacts:
    """Write a complete versioned validation result with row-accounting evidence."""

    selected_version = validate_artifact_version(version)
    interim_dir = settings.interim_dir / selected_version
    quarantine_dir = settings.quarantine_dir / selected_version
    interim_dir.mkdir(parents=True, exist_ok=True)
    quarantine_dir.mkdir(parents=True, exist_ok=True)

    valid_paths: dict[str, Path] = {}
    rejected_paths: dict[str, Path] = {}
    for dataset, result in results.items():
        if not result.row_accounting_valid:
            raise RuntimeError(f"Row accounting failed for {dataset}")
        valid_path = interim_dir / f"{dataset}.parquet"
        rejected_path = quarantine_dir / f"{dataset}_rejected.parquet"
        write_parquet(result.valid_rows, valid_path)
        write_parquet(result.rejected_rows, rejected_path)
        valid_paths[dataset] = valid_path
        rejected_paths[dataset] = rejected_path

    issues_path = quarantine_dir / "issues.parquet"
    write_parquet(_issue_frame(results), issues_path)

    bundle = load_contract_bundle(settings)
    dataset_summaries: dict[str, Any] = {}
    for dataset, result in results.items():
        contract = bundle["datasets"][dataset]
        dataset_summaries[dataset] = {
            **result.summary(),
            "source": {
                "path": portable_path(
                    settings.raw_dir / str(contract["file"]), settings.root
                ),
                "bytes": int(contract["bytes"]),
                "sha256": str(contract["sha256"]),
            },
            "validated_artifact": artifact_record(valid_paths[dataset], settings),
            "rejected_artifact": artifact_record(
                rejected_paths[dataset], settings
            ),
        }

    summary: dict[str, Any] = {
        "contract_version": str(bundle["contract_version"]),
        "validation_version": selected_version,
        "dataset_order": list(results),
        "dataset_failure": any(
            result.has_dataset_failure for result in results.values()
        ),
        "total_input_rows": sum(result.input_rows for result in results.values()),
        "total_valid_rows": sum(len(result.valid_rows) for result in results.values()),
        "total_rejected_rows": sum(
            len(result.rejected_rows) for result in results.values()
        ),
        "total_issues": sum(len(result.issues) for result in results.values()),
        "issues_artifact": artifact_record(issues_path, settings),
        "datasets": dataset_summaries,
    }
    summary_path = quarantine_dir / "summary.json"
    write_json(summary, summary_path)
    return ValidationArtifacts(
        version=selected_version,
        interim_dir=interim_dir,
        quarantine_dir=quarantine_dir,
        summary_path=summary_path,
        issues_path=issues_path,
        valid_paths=valid_paths,
        rejected_paths=rejected_paths,
        results=results,
        summary=summary,
    )


def run_validation_pipeline(
    settings: Settings,
    version: str | None = None,
) -> ValidationArtifacts:
    """Validate immutable sources and persist every retained and rejected row."""

    selected_version = version or settings.validation_version
    results = validate_raw_sources(settings)
    return persist_validation_results(settings, results, selected_version)
