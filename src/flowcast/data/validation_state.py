"""Shared state, lineage, schema, and result mechanics for raw validation."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd

from flowcast.data.contracts import (
    Disposition,
    ReasonCode,
    ValidationIssue,
    ValidationResult,
)


def _blank(series: pd.Series) -> pd.Series:
    return series.isna() | series.astype("string").str.strip().eq("")


def _value_text(value: Any) -> str | None:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=True)
    except (TypeError, ValueError):
        return str(value)


class _ValidationState:
    """Mutable row state shared by the configured validation rules."""

    def __init__(
        self,
        frame: pd.DataFrame,
        dataset: str,
        contract: dict[str, Any],
        source_file: str,
        timezone_name: str,
    ) -> None:
        self.dataset = dataset
        self.contract = contract
        self.source_file = source_file
        self.timezone_name = timezone_name
        self.frame = frame.reset_index(drop=True).copy(deep=True)
        self.source_columns = list(self.frame.columns)
        self.frame["_source_file"] = source_file
        self.frame["_source_row"] = pd.Series(
            range(2, len(self.frame) + 2), dtype="Int64"
        )
        self.raw = self.frame.copy(deep=True)
        self.issues: list[ValidationIssue] = []
        self.rejected: set[int] = set()
        self.row_reasons: dict[int, set[str]] = defaultdict(set)
        self.cell_issue_rows: set[int] = set()

    def _issue(
        self,
        position: int | None,
        field: str | None,
        reason: str,
        disposition: str,
        message: str,
        retained_source_row: int | None = None,
    ) -> None:
        source_row = None
        value = None
        if position is not None:
            source_row = int(self.frame.at[position, "_source_row"])
            if field in self.raw.columns:
                value = _value_text(self.raw.at[position, field])
            if disposition == Disposition.ROW_REJECTED.value:
                self.rejected.add(position)
                self.row_reasons[position].add(reason)
            elif disposition == Disposition.CELL_INVALIDATED.value:
                self.cell_issue_rows.add(position)
        self.issues.append(
            ValidationIssue(
                dataset=self.dataset,
                source_file=self.source_file,
                source_row=source_row,
                field=field,
                rejected_value=value,
                reason_code=reason,
                disposition=disposition,
                message=message,
                retained_source_row=retained_source_row,
            )
        )

    def _issues_for_mask(
        self,
        mask: pd.Series,
        field: str,
        reason: str,
        disposition: str,
        message: str,
    ) -> None:
        for position in np.flatnonzero(mask.fillna(False).to_numpy(dtype=bool)):
            self._issue(int(position), field, reason, disposition, message)

    def validate_schema(self) -> bool:
        """Reject the full dataset when the delivered column contract differs."""

        required = list(self.contract["required_columns"])
        observed = self.source_columns
        missing = [column for column in required if column not in observed]
        unexpected = [column for column in observed if column not in required]
        if not missing and not unexpected:
            return True

        reasons: list[str] = []
        for field in missing:
            reason = ReasonCode.MISSING_REQUIRED_COLUMN.value
            reasons.append(reason)
            self._issue(
                None,
                field,
                reason,
                Disposition.DATASET_REJECTED.value,
                f"Required column is absent: {field}",
            )
        for field in unexpected:
            reason = ReasonCode.UNEXPECTED_COLUMN.value
            reasons.append(reason)
            self._issue(
                None,
                field,
                reason,
                Disposition.DATASET_REJECTED.value,
                f"Unexpected source column is present: {field}",
            )
        combined = "|".join(sorted(set(reasons)))
        for position in self.frame.index:
            self.rejected.add(int(position))
            self.row_reasons[int(position)].add(combined)
        return False

    def result(self) -> ValidationResult:
        """Build stable retained, rejected, and issue outputs from current state."""

        rejected_positions = sorted(self.rejected)
        valid_positions = [
            int(position)
            for position in self.frame.index
            if int(position) not in self.rejected
        ]
        valid = self.frame.loc[valid_positions].copy()
        valid["_validation_status"] = [
            "valid_with_issues" if position in self.cell_issue_rows else "valid"
            for position in valid_positions
        ]

        rejected = self.raw.loc[rejected_positions].copy()
        output_column = str(self.contract["timestamp"]["output_column"])
        if output_column not in rejected and output_column in self.frame:
            rejected[output_column] = self.frame.loc[rejected_positions, output_column]
        rejected["_validation_status"] = "rejected"
        rejected["_rejection_reason"] = [
            "|".join(sorted(self.row_reasons[position]))
            for position in rejected_positions
        ]

        issues = tuple(
            sorted(
                self.issues,
                key=lambda issue: (
                    -1 if issue.source_row is None else issue.source_row,
                    "" if issue.field is None else issue.field,
                    issue.reason_code,
                    issue.disposition,
                ),
            )
        )
        return ValidationResult(
            dataset=self.dataset,
            source_file=self.source_file,
            input_rows=len(self.frame),
            valid_rows=valid.reset_index(drop=True),
            rejected_rows=rejected.reset_index(drop=True),
            issues=issues,
        )
