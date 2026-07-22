"""Causal station-local cleaning for hourly weather context data."""

from __future__ import annotations

from typing import Any

import pandas as pd

from flowcast.data.cleaning_types import TableCleaningResult


_REQUIRED_COLUMNS = {
    "station_id",
    "weather_condition",
    "temperature",
    "rainfall",
    "visibility",
    "weather_hour",
    "_source_file",
    "_source_row",
}


def _maximum_missing_run(mask: pd.Series, station: pd.Series) -> int:
    maximum = 0
    for _, group_mask in mask.groupby(station, sort=True):
        run_ids = group_mask.ne(group_mask.shift()).cumsum()
        lengths = group_mask.groupby(run_ids).sum()
        missing_lengths = lengths[lengths.gt(0)]
        if not missing_lengths.empty:
            maximum = max(maximum, int(missing_lengths.max()))
    return maximum


def _normalization_records(
    source: pd.Series,
    canonical: pd.Series,
) -> list[dict[str, Any]]:
    counts = (
        pd.DataFrame({"source_label": source, "canonical_label": canonical})
        .value_counts(sort=False)
        .rename("count")
        .reset_index()
        .sort_values(["canonical_label", "source_label"], kind="stable")
    )
    return [
        {
            "source_label": str(row.source_label),
            "canonical_label": str(row.canonical_label),
            "count": int(row.count),
        }
        for row in counts.itertuples(index=False)
    ]


def _causal_fill(
    frame: pd.DataFrame,
    field: str,
    policy: dict[str, Any],
) -> dict[str, Any]:
    method = str(policy["method"])
    if method != "station_forward_fill":
        raise ValueError(f"Unsupported weather imputation method: {method}")
    limit = int(policy["max_gap_hours"])
    if limit < 1:
        raise ValueError("Weather max_gap_hours must be positive")

    missing = frame[field].isna()
    maximum_run = _maximum_missing_run(missing, frame["station_id"])
    grouped = frame[field].groupby(frame["station_id"], sort=False)
    filled = grouped.ffill(limit=limit)
    unresolved = missing & filled.isna()
    if unresolved.any():
        rows = frame.loc[unresolved, "_source_row"].astype(int).tolist()
        raise ValueError(
            f"Weather {field} has gaps outside the causal policy at rows: {rows[:10]}"
        )

    donor_rows = (
        frame["_source_row"]
        .where(~missing)
        .groupby(frame["station_id"], sort=False)
        .ffill(limit=limit)
        .where(missing)
    )
    frame[field] = filled.astype("Float64")
    frame[f"{field}_was_missing"] = missing.astype(bool)
    frame[f"{field}_imputation_method"] = pd.Series(
        [method if value else "observed" for value in missing],
        dtype="string",
        index=frame.index,
    )
    frame[f"{field}_imputed_from_source_row"] = pd.array(
        donor_rows, dtype="Int64"
    )

    return {
        "method": method,
        "max_gap_hours": limit,
        "maximum_observed_gap_hours": maximum_run,
        "input_missing": int(missing.sum()),
        "imputed": int(missing.sum()),
        "remaining_missing": int(frame[field].isna().sum()),
        "by_station": {
            str(station): int(value)
            for station, value in missing.groupby(frame["station_id"]).sum().items()
        },
    }


def clean_weather(
    frame: pd.DataFrame,
    normalization_map: dict[str, str],
    config: dict[str, Any],
) -> TableCleaningResult:
    """Normalize and causally impute a complete unique station-hour table."""

    missing_columns = sorted(_REQUIRED_COLUMNS.difference(frame.columns))
    if missing_columns:
        raise ValueError(f"Weather input is missing columns: {missing_columns}")

    cleaned = frame.copy(deep=True)
    cleaned = cleaned.sort_values(
        ["station_id", "weather_hour"], kind="stable"
    ).reset_index(drop=True)
    if cleaned[["station_id", "weather_hour"]].isna().any().any():
        raise ValueError("Weather key contains missing values")
    if cleaned.duplicated(["station_id", "weather_hour"]).any():
        raise ValueError("Weather station/hour key is not unique")

    hourly_delta = cleaned.groupby("station_id")["weather_hour"].diff().dropna()
    if not hourly_delta.eq(pd.Timedelta(hours=1)).all():
        raise ValueError("Weather station grids are not complete hourly sequences")

    source_labels = cleaned["weather_condition"].astype("string")
    normalized_keys = source_labels.str.strip().str.casefold()
    canonical = normalized_keys.map(normalization_map).astype("string")
    if canonical.isna().any():
        labels = sorted(source_labels[canonical.isna()].dropna().unique().tolist())
        raise ValueError(f"Weather labels are outside the controlled map: {labels}")
    cleaned["weather_condition"] = canonical

    imputation: dict[str, Any] = {}
    for field, policy in config["numeric_imputation"].items():
        imputation[str(field)] = _causal_fill(cleaned, str(field), policy)

    cleaned["_cleaning_status"] = "unchanged"
    any_imputed = cleaned[
        [f"{field}_was_missing" for field in config["numeric_imputation"]]
    ].any(axis=1)
    cleaned.loc[any_imputed, "_cleaning_status"] = "imputed"

    if cleaned["rainfall"].isna().any() or cleaned["rainfall"].lt(0).any():
        raise ValueError("Cleaned rainfall must be complete and non-negative")
    if cleaned["visibility"].isna().any() or cleaned["visibility"].lt(0).any():
        raise ValueError("Cleaned visibility must be complete and non-negative")

    station_counts = {
        str(station): int(value)
        for station, value in cleaned.groupby("station_id").size().items()
    }
    summary = {
        "input_rows": len(frame),
        "output_rows": len(cleaned),
        "unique_station_hours": int(
            cleaned[["station_id", "weather_hour"]].drop_duplicates().shape[0]
        ),
        "station_counts": station_counts,
        "weather_hour_start": cleaned["weather_hour"].min().isoformat(),
        "weather_hour_end": cleaned["weather_hour"].max().isoformat(),
        "condition_normalization": _normalization_records(
            source_labels, canonical
        ),
        "condition_counts": {
            str(label): int(value)
            for label, value in canonical.value_counts().sort_index().items()
        },
        "imputation": imputation,
        "remaining_nulls": {
            field: int(cleaned[field].isna().sum())
            for field in ["weather_condition", "temperature", "rainfall", "visibility"]
        },
        "numeric_ranges": {
            field: {
                "minimum": float(cleaned[field].min()),
                "maximum": float(cleaned[field].max()),
            }
            for field in ["temperature", "rainfall", "visibility"]
        },
    }
    return TableCleaningResult(frame=cleaned, summary=summary)
