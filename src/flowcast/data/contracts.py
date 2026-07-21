"""Executable raw-data contract types and timestamp parsers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from flowcast.settings import Settings


class Disposition(str, Enum):
    """How validation handles a detected issue."""

    CELL_INVALIDATED = "cell_invalidated"
    ROW_REJECTED = "row_rejected"
    DATASET_REJECTED = "dataset_rejected"


class ReasonCode(str, Enum):
    """Stable machine-readable validation reason codes."""

    MISSING_REQUIRED_COLUMN = "missing_required_column"
    UNEXPECTED_COLUMN = "unexpected_column"
    MISSING_KEY = "missing_key"
    INVALID_TIMESTAMP = "invalid_timestamp"
    DUPLICATE_KEY = "duplicate_key"
    INVALID_TYPE = "invalid_type"
    MISSING_VALUE = "missing_value"
    INVALID_CATEGORY = "invalid_category"
    NEGATIVE_TRAFFIC_VOLUME = "negative_traffic_volume"
    EXCESSIVE_SPEED = "excessive_speed"
    INVALID_OCCUPANCY = "invalid_occupancy"
    INVALID_JSON = "invalid_json"
    INVALID_FLAG = "invalid_flag"
    INVALID_NUMERIC_RANGE = "invalid_numeric_range"
    MISSING_FLAG_NAME = "missing_flag_name"
    UNEXPECTED_FLAG_NAME = "unexpected_flag_name"


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """One dataset-, row-, or cell-level validation finding."""

    dataset: str
    source_file: str
    source_row: int | None
    field: str | None
    rejected_value: str | None
    reason_code: str
    disposition: str
    message: str
    retained_source_row: int | None = None

    def as_record(self) -> dict[str, Any]:
        """Return a Parquet/JSON-friendly issue record."""

        return {
            "dataset": self.dataset,
            "source_file": self.source_file,
            "source_row": self.source_row,
            "field": self.field,
            "rejected_value": self.rejected_value,
            "reason_code": self.reason_code,
            "disposition": self.disposition,
            "message": self.message,
            "retained_source_row": self.retained_source_row,
        }


@dataclass(frozen=True)
class ValidationResult:
    """Validated rows, rejected rows, and complete issue lineage for one source."""

    dataset: str
    source_file: str
    input_rows: int
    valid_rows: pd.DataFrame
    rejected_rows: pd.DataFrame
    issues: tuple[ValidationIssue, ...]

    @property
    def row_accounting_valid(self) -> bool:
        """Return whether every input row is either retained or rejected."""

        return self.input_rows == len(self.valid_rows) + len(self.rejected_rows)

    @property
    def has_dataset_failure(self) -> bool:
        """Return whether schema failure rejected the complete dataset."""

        return any(
            issue.disposition == Disposition.DATASET_REJECTED.value
            for issue in self.issues
        )

    def summary(self) -> dict[str, Any]:
        """Return deterministic counts suitable for validation manifests."""

        by_reason: dict[str, int] = {}
        by_disposition: dict[str, int] = {}
        for issue in self.issues:
            by_reason[issue.reason_code] = by_reason.get(issue.reason_code, 0) + 1
            by_disposition[issue.disposition] = (
                by_disposition.get(issue.disposition, 0) + 1
            )
        return {
            "source_file": self.source_file,
            "input_rows": self.input_rows,
            "valid_rows": len(self.valid_rows),
            "rejected_rows": len(self.rejected_rows),
            "issue_count": len(self.issues),
            "issues_by_reason": dict(sorted(by_reason.items())),
            "issues_by_disposition": dict(sorted(by_disposition.items())),
            "row_accounting_valid": self.row_accounting_valid,
            "dataset_failure": self.has_dataset_failure,
        }


def load_contract_bundle(settings: Settings) -> dict[str, Any]:
    """Load the complete versioned raw-data contract bundle."""

    with settings.data_contracts_path.open("r", encoding="utf-8") as handle:
        bundle: dict[str, Any] = yaml.safe_load(handle)
    if "datasets" not in bundle or not isinstance(bundle["datasets"], dict):
        raise ValueError("data_contracts.yaml must contain a datasets mapping")
    return bundle


def load_contracts(settings: Settings) -> dict[str, dict[str, Any]]:
    """Load dataset contracts while preserving their configured order."""

    contracts = load_contract_bundle(settings)["datasets"]
    return {str(name): contract for name, contract in contracts.items()}


def parse_traffic_timestamp(date: pd.Series, time: pd.Series) -> pd.Series:
    """Parse traffic date/time columns with the mandated source format."""

    return pd.to_datetime(
        date.astype("string") + " " + time.astype("string"),
        format="%Y-%m-%d %H:%M",
        errors="coerce",
    )


def parse_weather_timestamp(date: pd.Series, time: pd.Series) -> pd.Series:
    """Parse day-first weather date/time columns with the mandated format."""

    return pd.to_datetime(
        date.astype("string") + " " + time.astype("string"),
        format="%d/%m/%Y %H:%M",
        errors="coerce",
    )


def portable_path(path: Path, root: Path) -> str:
    """Return a root-relative path when possible and a portable absolute fallback."""

    resolved = path.resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()
