"""Deterministic cleaning and contract checks for calendar context data."""

from __future__ import annotations

from typing import Any

import pandas as pd

from flowcast.data.cleaning_types import TableCleaningResult


_REQUIRED_COLUMNS = {
    "date",
    "public_holiday",
    "holiday_name",
    "event_flag",
    "event_name",
    "roadwork_flag",
    "_source_file",
    "_source_row",
}


def _blank(series: pd.Series) -> pd.Series:
    return series.isna() | series.astype("string").str.strip().eq("")


def clean_calendar(
    frame: pd.DataFrame,
    config: dict[str, Any],
) -> TableCleaningResult:
    """Return a unique normalized calendar table or fail its trusted contract."""

    missing_columns = sorted(_REQUIRED_COLUMNS.difference(frame.columns))
    if missing_columns:
        raise ValueError(f"Calendar input is missing columns: {missing_columns}")

    cleaned = frame.copy(deep=True)
    cleaned["date"] = pd.to_datetime(cleaned["date"], errors="coerce").dt.normalize()
    if cleaned["date"].isna().any():
        raise ValueError("Calendar contains an invalid normalized date")

    flag_columns = ["public_holiday", "event_flag", "roadwork_flag"]
    for column in flag_columns:
        values = pd.to_numeric(cleaned[column], errors="coerce")
        if values.isna().any() or not values.isin([0, 1]).all():
            raise ValueError(f"Calendar flag must contain only 0/1: {column}")
        cleaned[column] = values.astype("Int8")

    pairs = list(config["flag_name_pairs"])
    for pair in pairs:
        flag = str(pair["flag"])
        name = str(pair["name"])
        names = cleaned[name].astype("string").str.strip()
        names = names.mask(names.eq(""), pd.NA)
        cleaned[name] = names
        missing_name = cleaned[flag].eq(1) & _blank(cleaned[name])
        unexpected_name = cleaned[flag].eq(0) & ~_blank(cleaned[name])
        if missing_name.any() or unexpected_name.any():
            raise ValueError(f"Calendar flag/name relationship failed: {flag}/{name}")

    cleaned = cleaned.sort_values("date", kind="stable").reset_index(drop=True)
    if cleaned["date"].duplicated().any():
        raise ValueError("Calendar date key is not unique")

    summary = {
        "input_rows": len(frame),
        "output_rows": len(cleaned),
        "unique_dates": int(cleaned["date"].nunique()),
        "date_start": cleaned["date"].min().date().isoformat(),
        "date_end": cleaned["date"].max().date().isoformat(),
        "flag_counts": {
            column: int(cleaned[column].sum()) for column in flag_columns
        },
        "duplicate_dates": int(cleaned["date"].duplicated().sum()),
        "invalid_flag_name_pairs": 0,
    }
    return TableCleaningResult(frame=cleaned, summary=summary)
