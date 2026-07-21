"""Persist validated rows, rejected rows, and machine-readable issue lineage."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from flowcast.data.audit import sha256_file
from flowcast.data.contracts import (
    ValidationResult,
    load_contract_bundle,
    portable_path,
)
from flowcast.data.ingest import validate_raw_sources
from flowcast.settings import Settings


_SAFE_VERSION = re.compile(r"^[A-Za-z0-9_.-]+$")
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


def _validate_version(version: str) -> str:
    if not version or not _SAFE_VERSION.fullmatch(version):
        raise ValueError(
            "Validation version must contain only letters, numbers, '.', '_', or '-'"
        )
    return version


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False, engine="pyarrow")


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


def _artifact_record(path: Path, settings: Settings) -> dict[str, Any]:
    return {
        "path": portable_path(path, settings.root),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path, settings.hash_chunk_size),
    }


def persist_validation_results(
    settings: Settings,
    results: dict[str, ValidationResult],
    version: str,
) -> ValidationArtifacts:
    """Write a complete versioned validation result with row-accounting evidence."""

    selected_version = _validate_version(version)
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
        _write_parquet(result.valid_rows, valid_path)
        _write_parquet(result.rejected_rows, rejected_path)
        valid_paths[dataset] = valid_path
        rejected_paths[dataset] = rejected_path

    issues_path = quarantine_dir / "issues.parquet"
    _write_parquet(_issue_frame(results), issues_path)

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
            "validated_artifact": _artifact_record(valid_paths[dataset], settings),
            "rejected_artifact": _artifact_record(
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
        "issues_artifact": _artifact_record(issues_path, settings),
        "datasets": dataset_summaries,
    }
    summary_path = quarantine_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
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
