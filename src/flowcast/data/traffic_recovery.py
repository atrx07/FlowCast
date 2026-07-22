"""Causal field-recovery helpers with propagated donor lineage."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd


MEASUREMENT_DTYPES = {
    "traffic_volume": "Int64",
    "vehicle_count": "Int64",
    "avg_speed": "Float64",
    "occupancy": "Float64",
    "travel_time": "Float64",
    "signal_timing": "Int64",
    "vehicle_type_dist": "string",
}


def _maximum_missing_run(mask: pd.Series, roads: pd.Series) -> int:
    maximum = 0
    for _, group_mask in mask.groupby(roads, sort=True):
        run_ids = group_mask.ne(group_mask.shift()).cumsum()
        lengths = group_mask.groupby(run_ids).sum()
        if not lengths.empty:
            maximum = max(maximum, int(lengths.max()))
    return maximum


def _json_lineage(value: Any) -> Any:
    if pd.isna(value):
        return pd.NA
    return json.dumps([int(value)], separators=(",", ":"))


def causal_fill(
    frame: pd.DataFrame,
    field: str,
    policy: dict[str, Any],
) -> dict[str, Any]:
    """Repair one ordered field without consulting a future timestamp."""

    values = frame[field].copy()
    initial_missing = values.isna()
    maximum_run = _maximum_missing_run(initial_missing, frame["road_id"])
    method = pd.Series("observed", index=frame.index, dtype="string")
    method.loc[initial_missing] = "unresolved"

    lineage_row = frame["_source_row"].where(values.notna()).astype("Int64")
    lineage_time = frame["timestamp"].where(values.notna())
    lineage_rows = pd.Series(
        [
            _json_lineage(value) if valid else pd.NA
            for value, valid in zip(
                frame["_source_row"], values.notna(), strict=True
            )
        ],
        index=frame.index,
        dtype="string",
    )

    semantic_source = policy.get("same_row_source")
    if semantic_source:
        use = values.isna() & frame[str(semantic_source)].notna()
        values.loc[use] = frame.loc[use, str(semantic_source)]
        method.loc[use] = f"same_row_{semantic_source}"
        lineage_row.loc[use] = frame.loc[use, "_source_row"]
        lineage_time.loc[use] = frame.loc[use, "timestamp"]
        lineage_rows.loc[use] = frame.loc[use, "_source_row"].map(_json_lineage)

    lag = int(policy["previous_day_lag_windows"])
    if lag < 1:
        raise ValueError(f"Traffic {field} previous-day lag must be positive")
    prior_values = values.groupby(frame["road_id"], sort=False).shift(lag)
    use = values.isna() & prior_values.notna()
    values.loc[use] = prior_values.loc[use]
    method.loc[use] = "previous_day_same_window"
    for target, source in (
        (lineage_row, lineage_row.groupby(frame["road_id"], sort=False).shift(lag)),
        (
            lineage_time,
            lineage_time.groupby(frame["road_id"], sort=False).shift(lag),
        ),
        (
            lineage_rows,
            lineage_rows.groupby(frame["road_id"], sort=False).shift(lag),
        ),
    ):
        target.loc[use] = source.loc[use]

    limit = int(policy["forward_fill_limit_windows"])
    if limit < 1:
        raise ValueError(f"Traffic {field} forward-fill limit must be positive")
    forward_values = values.groupby(frame["road_id"], sort=False).ffill(limit=limit)
    use = values.isna() & forward_values.notna()
    values.loc[use] = forward_values.loc[use]
    method.loc[use] = "same_road_causal_forward_fill"
    for target in (lineage_row, lineage_time, lineage_rows):
        source = target.groupby(frame["road_id"], sort=False).ffill(limit=limit)
        target.loc[use] = source.loc[use]

    if policy.get("leading_fallback"):
        if policy["leading_fallback"] != "same_timestamp_station_median":
            raise ValueError(f"Unsupported traffic leading fallback for {field}")
        peer_available = values.notna()
        peer_median = values.groupby(
            [frame["timestamp"], frame["weather_station_id"]], sort=False
        ).transform("median")
        use = values.isna() & peer_median.notna()
        values.loc[use] = peer_median.loc[use]
        method.loc[use] = "same_timestamp_station_median"
        lineage_time.loc[use] = frame.loc[use, "timestamp"]
        for position in frame.index[use]:
            peers = (
                frame["timestamp"].eq(frame.at[position, "timestamp"])
                & frame["weather_station_id"].eq(
                    frame.at[position, "weather_station_id"]
                )
                & peer_available
                & frame["_source_row"].notna()
            )
            donors = sorted({int(value) for value in frame.loc[peers, "_source_row"]})
            lineage_rows.at[position] = json.dumps(donors, separators=(",", ":"))

    if values.isna().any():
        rows = frame.loc[values.isna(), ["road_id", "timestamp"]].head(10)
        raise ValueError(
            f"Traffic {field} has gaps outside the causal policy: "
            f"{rows.to_dict('records')}"
        )

    frame[field] = values.astype(MEASUREMENT_DTYPES[field])
    imputed = method.ne("observed")
    frame[f"{field}_imputation_method"] = method
    frame[f"{field}_imputation_donor_source_row"] = (
        lineage_row.where(imputed).astype("Int64")
    )
    frame[f"{field}_imputation_donor_timestamp"] = lineage_time.where(imputed)
    frame[f"{field}_imputation_donor_source_rows"] = lineage_rows.where(imputed)
    return {
        "input_missing_after_grid": int(initial_missing.sum()),
        "maximum_missing_run_windows": maximum_run,
        "previous_day_lag_windows": lag,
        "forward_fill_limit_windows": limit,
        "method_counts": {
            str(name): int(count)
            for name, count in method[imputed].value_counts().sort_index().items()
        },
        "imputed": int(imputed.sum()),
        "remaining_missing": int(frame[field].isna().sum()),
    }
