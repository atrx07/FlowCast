"""Deterministic raw-source validation with complete issue lineage."""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import pandas as pd

from flowcast.data.contracts import (
    Disposition,
    ReasonCode,
    ValidationResult,
)
from flowcast.data.validation_state import _ValidationState, _blank


class _FrameValidator(_ValidationState):

    def validate_strings(self) -> None:
        for field, rule in self.contract.get("strings", {}).items():
            missing = _blank(self.frame[field])
            if not bool(rule.get("nullable", False)):
                disposition = str(rule["disposition"])
                self._issues_for_mask(
                    missing,
                    field,
                    ReasonCode.MISSING_VALUE.value,
                    disposition,
                    "Required string value is blank",
                )

    def validate_categories(self) -> None:
        for field, rule in self.contract.get("categorical", {}).items():
            values = self.frame[field].astype("string")
            missing = _blank(self.frame[field])
            disposition = str(rule["disposition"])
            nullable = bool(rule.get("nullable", False))
            if not nullable or bool(rule.get("report_missing", False)):
                reason = (
                    ReasonCode.MISSING_KEY.value
                    if disposition == Disposition.ROW_REJECTED.value
                    else ReasonCode.MISSING_VALUE.value
                )
                self._issues_for_mask(
                    missing,
                    field,
                    reason,
                    disposition,
                    "Categorical value is blank",
                )

            if "allowed" in rule:
                allowed = {str(value) for value in rule["allowed"]}
                invalid = ~missing & ~values.isin(allowed)
            else:
                normalized = values.str.strip().str.casefold()
                allowed = {str(value) for value in rule["normalization_map"]}
                invalid = ~missing & ~normalized.isin(allowed)
            self._issues_for_mask(
                invalid,
                field,
                ReasonCode.INVALID_CATEGORY.value,
                disposition,
                "Value is outside the configured categorical vocabulary",
            )
            if disposition == Disposition.CELL_INVALIDATED.value:
                self.frame.loc[missing | invalid, field] = pd.NA

    def validate_numeric(self) -> None:
        for field, rule in self.contract.get("numeric", {}).items():
            original = self.frame[field]
            missing = _blank(original)
            parsed = pd.to_numeric(original, errors="coerce")
            invalid_type = ~missing & parsed.isna()
            invalid_type |= parsed.notna() & ~np.isfinite(parsed)
            if rule["dtype"] == "integer":
                invalid_type |= parsed.notna() & parsed.mod(1).ne(0)

            disposition = str(rule["disposition"])
            invalid_reason = str(
                rule.get("invalid_reason", ReasonCode.INVALID_TYPE.value)
            )
            self._issues_for_mask(
                invalid_type,
                field,
                invalid_reason,
                disposition,
                "Value cannot be coerced to the configured numeric type",
            )
            if not bool(rule.get("nullable", False)) or bool(
                rule.get("report_missing", False)
            ):
                self._issues_for_mask(
                    missing,
                    field,
                    ReasonCode.MISSING_VALUE.value,
                    disposition,
                    "Numeric value is missing",
                )

            invalid_range = pd.Series(False, index=self.frame.index)
            comparable = parsed.notna() & ~invalid_type
            if "allowed" in rule:
                outside = comparable & ~parsed.isin(rule["allowed"])
                invalid_range |= outside
                self._issues_for_mask(
                    outside,
                    field,
                    invalid_reason,
                    disposition,
                    "Numeric value is outside the configured allowed set",
                )
            if "minimum" in rule:
                if bool(rule.get("exclusive_minimum", False)):
                    below = comparable & parsed.le(float(rule["minimum"]))
                else:
                    below = comparable & parsed.lt(float(rule["minimum"]))
                invalid_range |= below
                reason = str(
                    rule.get("minimum_reason", ReasonCode.INVALID_NUMERIC_RANGE.value)
                )
                self._issues_for_mask(
                    below,
                    field,
                    reason,
                    disposition,
                    "Numeric value is below the configured physical minimum",
                )
            if "maximum" in rule:
                above = comparable & parsed.gt(float(rule["maximum"]))
                invalid_range |= above
                reason = str(
                    rule.get("maximum_reason", ReasonCode.INVALID_NUMERIC_RANGE.value)
                )
                self._issues_for_mask(
                    above,
                    field,
                    reason,
                    disposition,
                    "Numeric value exceeds the configured physical maximum",
                )

            parsed = parsed.mask(invalid_type | invalid_range)
            self.frame[field] = (
                parsed.astype("Int64")
                if rule["dtype"] == "integer"
                else parsed.astype("Float64")
            )

    def validate_json_fields(self) -> None:
        for field, rule in self.contract.get("json_fields", {}).items():
            invalid = pd.Series(False, index=self.frame.index)
            required = set(map(str, rule["required_keys"]))
            tolerance = float(rule["sum_tolerance"])
            expected_sum = float(rule["expected_sum"])
            for position, raw_value in self.frame[field].items():
                try:
                    parsed = (
                        json.loads(raw_value)
                        if isinstance(raw_value, str)
                        else raw_value
                    )
                    if not isinstance(parsed, dict) or set(parsed) != required:
                        raise ValueError("unexpected keys")
                    values = [parsed[key] for key in rule["required_keys"]]
                    if any(
                        isinstance(value, bool)
                        or not isinstance(value, (int, float))
                        or value < float(rule["value_minimum"])
                        or value > float(rule["value_maximum"])
                        for value in values
                    ):
                        raise ValueError("invalid share")
                    if not math.isclose(
                        sum(map(float, values)),
                        expected_sum,
                        rel_tol=0.0,
                        abs_tol=tolerance,
                    ):
                        raise ValueError("shares do not sum near one")
                except (TypeError, ValueError):
                    invalid.at[position] = True
            disposition = str(rule["disposition"])
            self._issues_for_mask(
                invalid,
                field,
                ReasonCode.INVALID_JSON.value,
                disposition,
                "JSON object must contain valid near-unit vehicle-class shares",
            )
            if disposition == Disposition.CELL_INVALIDATED.value:
                self.frame.loc[invalid, field] = pd.NA

    def validate_timestamp(self) -> None:
        rule = self.contract["timestamp"]
        date_field = str(rule["date_column"])
        date_text = self.frame[date_field].astype("string")
        date_valid = date_text.str.fullmatch(str(rule["date_regex"]), na=False)

        if "time_column" in rule:
            time_field = str(rule["time_column"])
            time_text = self.frame[time_field].astype("string")
            time_valid = time_text.str.fullmatch(str(rule["time_regex"]), na=False)
            parsed = pd.to_datetime(
                date_text + " " + time_text,
                format=f"{rule['date_format']} {rule['time_format']}",
                errors="coerce",
            )
            frequency = int(rule["frequency_minutes"])
            time_valid &= parsed.notna() & parsed.dt.minute.mod(frequency).eq(0)
            valid = date_valid & time_valid & parsed.notna()
            rejected_value_field = time_field
        else:
            parsed = pd.to_datetime(
                date_text, format=str(rule["date_format"]), errors="coerce"
            )
            valid = date_valid & parsed.notna()
            rejected_value_field = date_field

        invalid = ~valid
        self._issues_for_mask(
            invalid,
            rejected_value_field,
            ReasonCode.INVALID_TIMESTAMP.value,
            Disposition.ROW_REJECTED.value,
            "Date/time does not match the exact source format and frequency",
        )
        if bool(rule.get("timezone_aware", False)):
            parsed = parsed.dt.tz_localize(self.timezone_name)
        else:
            parsed = parsed.dt.normalize()
        self.frame[str(rule["output_column"])] = parsed.mask(invalid)

    def validate_flag_name_pairs(self) -> None:
        for pair in self.contract.get("flag_name_pairs", []):
            flag = str(pair["flag"])
            name = str(pair["name"])
            blank_name = _blank(self.frame[name])
            missing_name = self.frame[flag].eq(1) & blank_name
            unexpected_name = self.frame[flag].eq(0) & ~blank_name
            self._issues_for_mask(
                missing_name,
                name,
                ReasonCode.MISSING_FLAG_NAME.value,
                Disposition.ROW_REJECTED.value,
                f"{name} is required when {flag}=1",
            )
            self._issues_for_mask(
                unexpected_name,
                name,
                ReasonCode.UNEXPECTED_FLAG_NAME.value,
                Disposition.ROW_REJECTED.value,
                f"{name} must be blank when {flag}=0",
            )

    def validate_keys_and_duplicates(self) -> None:
        keys = list(self.contract["key"])
        missing_key = self.frame[keys].isna().any(axis=1)
        new_missing = missing_key & ~self.frame.index.to_series().isin(self.rejected)
        self._issues_for_mask(
            new_missing,
            "|".join(keys),
            ReasonCode.MISSING_KEY.value,
            Disposition.ROW_REJECTED.value,
            "One or more uniqueness-key fields are missing",
        )

        candidates = self.frame.index[~self.frame.index.isin(self.rejected)]
        candidate_frame = self.frame.loc[candidates]
        duplicate_rows = candidate_frame.duplicated(keys, keep=False)
        if not duplicate_rows.any():
            return
        completeness = self.frame[list(self.contract["required_columns"])].notna().sum(
            axis=1
        )
        grouped = candidate_frame.loc[duplicate_rows].groupby(
            keys, sort=True, dropna=False
        )
        for positions in grouped.groups.values():
            ranked = sorted(
                map(int, positions),
                key=lambda position: (
                    -int(completeness.at[position]),
                    int(self.frame.at[position, "_source_row"]),
                ),
            )
            retained = ranked[0]
            retained_source_row = int(self.frame.at[retained, "_source_row"])
            for position in ranked[1:]:
                exact = self.raw.loc[
                    position, self.contract["required_columns"]
                ].equals(
                    self.raw.loc[retained, self.contract["required_columns"]]
                )
                self._issue(
                    position,
                    "|".join(keys),
                    ReasonCode.DUPLICATE_KEY.value,
                    Disposition.ROW_REJECTED.value,
                    f"Duplicate key removed; exact_duplicate={str(exact).lower()}",
                    retained_source_row=retained_source_row,
                )

def validate_frame(
    frame: pd.DataFrame,
    dataset: str,
    contract: dict[str, Any],
    source_file: str,
    timezone_name: str = "Asia/Kolkata",
) -> ValidationResult:
    """Validate one source frame without silently dropping rows or cells."""

    validator = _FrameValidator(
        frame, dataset, contract, source_file, timezone_name
    )
    if not validator.validate_schema():
        return validator.result()
    validator.validate_strings()
    validator.validate_categories()
    validator.validate_numeric()
    validator.validate_json_fields()
    validator.validate_timestamp()
    validator.validate_flag_name_pairs()
    validator.validate_keys_and_duplicates()
    result = validator.result()
    if not result.row_accounting_valid:
        raise RuntimeError(f"Row accounting failed for {dataset}")
    return result
