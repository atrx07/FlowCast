"""Leakage-safe traffic cleaning and complete half-hour grid reconstruction."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from flowcast.data.cleaning_types import TableCleaningResult
from flowcast.data.traffic_recovery import MEASUREMENT_DTYPES, causal_fill


_REQUIRED_COLUMNS = {
    "road_id",
    "road_name",
    "latitude",
    "longitude",
    "weather_station_id",
    "traffic_volume",
    "vehicle_count",
    "vehicle_type_dist",
    "avg_speed",
    "occupancy",
    "congestion_level",
    "travel_time",
    "accident_count",
    "signal_timing",
    "road_capacity",
    "timestamp",
    "_source_file",
    "_source_row",
    "_validation_status",
}
_ISSUE_REASONS = {
    "traffic_volume": ("missing_value", "negative_traffic_volume"),
    "avg_speed": ("missing_value", "excessive_speed"),
    "occupancy": ("missing_value", "invalid_occupancy"),
    "congestion_level": ("missing_value", None),
}


def derive_congestion(
    volume: pd.Series,
    hourly_capacity: pd.Series,
) -> pd.Series:
    """Band volume by exact half-hour volume/capacity boundaries."""

    ratio = volume.astype("Float64") / (hourly_capacity.astype("Float64") / 2.0)
    labels = np.select(
        [ratio.lt(0.50), ratio.lt(0.80), ratio.lt(1.00)],
        ["Free-flow", "Moderate", "Heavy"],
        default="Severe",
    )
    return pd.Series(labels, index=volume.index, dtype="string")


def _issue_rows(
    issues: pd.DataFrame,
    field: str,
    reason: str,
) -> set[int]:
    selected = issues.loc[
        issues["field"].eq(field) & issues["reason_code"].eq(reason),
        "source_row",
    ].dropna()
    return {int(value) for value in selected}


def _add_source_state_flags(
    frame: pd.DataFrame,
    issues: pd.DataFrame,
) -> None:
    source_rows = frame["_source_row"]
    for field in MEASUREMENT_DTYPES:
        missing_reason, invalid_reason = _ISSUE_REASONS.get(field, (None, None))
        missing_rows = (
            _issue_rows(issues, field, missing_reason) if missing_reason else set()
        )
        invalid_rows = (
            _issue_rows(issues, field, invalid_reason) if invalid_reason else set()
        )
        frame[f"{field}_original_missing"] = source_rows.isin(missing_rows)
        frame[f"{field}_physical_invalid"] = source_rows.isin(invalid_rows)

    missing_rows = _issue_rows(issues, "congestion_level", "missing_value")
    frame["congestion_level_original_missing"] = source_rows.isin(missing_rows)


def _parse_vehicle_shares(
    frame: pd.DataFrame,
    policy: dict[str, Any],
) -> dict[str, Any]:
    required = [str(value) for value in policy["required_keys"]]
    expected = float(policy["expected_sum"])
    tolerance = float(policy["sum_tolerance"])
    parsed: list[dict[str, float]] = []
    sums: list[float] = []
    for raw in frame["vehicle_type_dist"]:
        value = json.loads(str(raw))
        if not isinstance(value, dict) or set(value) != set(required):
            raise ValueError("Traffic vehicle distribution keys are invalid")
        numeric = {key: float(value[key]) for key in required}
        if any(not 0 <= share <= 1 for share in numeric.values()):
            raise ValueError("Traffic vehicle shares must remain within 0-1")
        total = sum(numeric.values())
        if abs(total - expected) > tolerance + 1e-12:
            raise ValueError("Traffic vehicle shares exceed the sum tolerance")
        parsed.append(numeric)
        sums.append(total)

    normalize = bool(policy["normalize_to_unit_sum"])
    normalized = np.array([not np.isclose(total, expected) for total in sums])
    for key, column in policy["share_columns"].items():
        shares = [value[str(key)] for value in parsed]
        if normalize:
            shares = [share / total for share, total in zip(shares, sums, strict=True)]
        frame[str(column)] = pd.array(shares, dtype="Float64")
    frame["vehicle_share_original_sum"] = pd.array(sums, dtype="Float64")
    frame["vehicle_shares_normalized"] = normalized
    share_columns = [str(value) for value in policy["share_columns"].values()]
    output_sums = frame[share_columns].sum(axis=1)
    if not np.allclose(output_sums, expected, atol=1e-12):
        raise ValueError("Normalized traffic vehicle shares do not sum to one")
    return {
        "required_keys": required,
        "sum_tolerance": tolerance,
        "input_sum_minimum": float(min(sums)),
        "input_sum_maximum": float(max(sums)),
        "normalized_rows": int(normalized.sum()),
        "output_sum_minimum": float(output_sums.min()),
        "output_sum_maximum": float(output_sums.max()),
    }


def _reconstruct_grid(
    frame: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    cleaned = frame.sort_values(["road_id", "timestamp"], kind="stable").copy()
    if cleaned.duplicated(["road_id", "timestamp"]).any():
        raise ValueError("Validated traffic key is not unique")
    roads = sorted(cleaned["road_id"].unique().tolist())
    expected_roads = int(config["grid"]["expected_road_count"])
    if len(roads) != expected_roads:
        raise ValueError(f"Expected {expected_roads} traffic roads, found {len(roads)}")

    metadata = [str(value) for value in config["static_metadata"]]
    inconsistent = {
        field: sorted(
            cleaned.groupby("road_id")[field]
            .nunique(dropna=False)
            .loc[lambda values: values.ne(1)]
            .index.tolist()
        )
        for field in metadata
    }
    inconsistent = {key: value for key, value in inconsistent.items() if value}
    if inconsistent:
        raise ValueError(f"Traffic road metadata is inconsistent: {inconsistent}")
    road_metadata = cleaned.groupby("road_id", sort=True)[metadata].first()

    timezone = cleaned["timestamp"].dt.tz
    start = pd.Timestamp(config["grid"]["start"], tz=timezone)
    end = pd.Timestamp(config["grid"]["end"], tz=timezone)
    timestamps = pd.date_range(start, end, freq=str(config["grid"]["frequency"]))
    index = pd.MultiIndex.from_product(
        [roads, timestamps], names=["road_id", "timestamp"]
    )
    cleaned = cleaned.set_index(["road_id", "timestamp"]).reindex(index).reset_index()
    cleaned["_inserted_window"] = cleaned["_source_row"].isna()
    for field in metadata:
        cleaned[field] = cleaned[field].fillna(
            cleaned["road_id"].map(road_metadata[field])
        )
    cleaned["date"] = cleaned["timestamp"].dt.strftime("%Y-%m-%d").astype("string")
    cleaned["time"] = cleaned["timestamp"].dt.strftime("%H:%M").astype("string")
    cleaned["_source_file"] = cleaned["_source_file"].astype("string")
    cleaned["_source_row"] = cleaned["_source_row"].astype("Int64")
    cleaned["_validation_status"] = cleaned["_validation_status"].fillna(
        "reconstructed_missing_window"
    ).astype("string")
    cleaned["_accident_observed"] = ~cleaned["_inserted_window"]
    expected_rows = expected_roads * len(timestamps)
    if len(cleaned) != expected_rows:
        raise ValueError("Traffic grid row count does not match its contract")
    return cleaned, {
        "expected_rows": expected_rows,
        "timestamp_count_per_road": len(timestamps),
        "inserted_windows": int(cleaned["_inserted_window"].sum()),
        "start": start.isoformat(),
        "end": end.isoformat(),
        "frequency": str(config["grid"]["frequency"]),
        "metadata_inconsistencies": inconsistent,
    }


def clean_traffic(
    frame: pd.DataFrame,
    issues: pd.DataFrame,
    config: dict[str, Any],
) -> TableCleaningResult:
    """Return a complete, causally repaired traffic table with full lineage."""

    missing_columns = sorted(_REQUIRED_COLUMNS.difference(frame.columns))
    if missing_columns:
        raise ValueError(f"Traffic input is missing columns: {missing_columns}")
    cleaned, grid_summary = _reconstruct_grid(frame, config)
    _add_source_state_flags(cleaned, issues)

    imputation: dict[str, Any] = {}
    for field, policy in config["causal_imputation"].items():
        imputation[str(field)] = causal_fill(cleaned, str(field), policy)
        imputation[str(field)]["original_missing"] = int(
            cleaned[f"{field}_original_missing"].sum()
        )
        imputation[str(field)]["physical_invalid"] = int(
            cleaned[f"{field}_physical_invalid"].sum()
        )

    vehicle_summary = _parse_vehicle_shares(
        cleaned, config["vehicle_distribution"]
    )
    derived = derive_congestion(cleaned["traffic_volume"], cleaned["road_capacity"])
    source_present = cleaned["congestion_level"].notna()
    disagreement = source_present & cleaned["congestion_level"].ne(derived)
    cleaned["congestion_level_derived"] = derived
    cleaned["congestion_level_was_derived"] = ~source_present
    cleaned["congestion_level_imputation_method"] = pd.Series(
        np.where(source_present, "source", "derived_volume_capacity"),
        dtype="string",
        index=cleaned.index,
    )
    cleaned.loc[~source_present, "congestion_level"] = derived.loc[~source_present]
    cleaned["congestion_level"] = cleaned["congestion_level"].astype("string")

    trusted = ["traffic_volume", "avg_speed", "occupancy", "travel_time"]
    if cleaned[trusted + ["congestion_level"]].isna().any().any():
        raise ValueError("Cleaned traffic trusted fields contain null values")
    ranges = config["physical_ranges"]
    if cleaned["traffic_volume"].lt(ranges["traffic_volume"]["minimum"]).any():
        raise ValueError("Cleaned traffic volume is negative")
    if (
        cleaned["avg_speed"].le(ranges["avg_speed"]["exclusive_minimum"]).any()
        or cleaned["avg_speed"].gt(ranges["avg_speed"]["maximum"]).any()
    ):
        raise ValueError("Cleaned traffic speed is outside its physical range")
    if (
        cleaned["occupancy"].lt(ranges["occupancy"]["minimum"]).any()
        or cleaned["occupancy"].gt(ranges["occupancy"]["maximum"]).any()
    ):
        raise ValueError("Cleaned traffic occupancy is outside its physical range")
    if cleaned["travel_time"].le(ranges["travel_time"]["exclusive_minimum"]).any():
        raise ValueError("Cleaned traffic travel time is not positive")

    imputed_columns = [
        f"{field}_imputation_method" for field in config["causal_imputation"]
    ]
    cleaned["_cleaning_status"] = "unchanged"
    repaired = cleaned[imputed_columns].ne("observed").any(axis=1)
    cleaned.loc[repaired, "_cleaning_status"] = "imputed"
    cleaned.loc[cleaned["_inserted_window"], "_cleaning_status"] = "reconstructed"
    cleaned = cleaned.sort_values(["road_id", "timestamp"], kind="stable").reset_index(
        drop=True
    )

    summary = {
        "input_rows": len(frame),
        "output_rows": len(cleaned),
        "road_count": int(cleaned["road_id"].nunique()),
        "unique_road_timestamps": int(
            cleaned[["road_id", "timestamp"]].drop_duplicates().shape[0]
        ),
        "grid": grid_summary,
        "duplicate_rows_accounted": int(
            issues["reason_code"].eq("duplicate_key").sum()
        ),
        "imputation": imputation,
        "vehicle_distribution": vehicle_summary,
        "congestion": {
            "source_labels_preserved": int(source_present.sum()),
            "derived_labels": int((~source_present).sum()),
            "source_disagreements": int(disagreement.sum()),
            "class_counts": {
                str(label): int(count)
                for label, count in cleaned["congestion_level"]
                .value_counts()
                .sort_index()
                .items()
            },
        },
        "unobserved_accident_windows": int(cleaned["accident_count"].isna().sum()),
        "numeric_ranges": {
            field: {
                "minimum": float(cleaned[field].min()),
                "maximum": float(cleaned[field].max()),
            }
            for field in trusted
        },
        "remaining_trusted_nulls": {
            field: int(cleaned[field].isna().sum())
            for field in trusted + ["congestion_level"]
        },
    }
    return TableCleaningResult(frame=cleaned, summary=summary)
