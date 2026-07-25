"""Origin validation and contiguous recurrent feature preparation."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd


def normalize_origin(value: str, timezone: str) -> pd.Timestamp:
    """Parse one origin into the configured timezone and half-hour cadence."""

    try:
        origin = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid origin timestamp: {value}") from exc
    if origin.tzinfo is None:
        origin = origin.tz_localize(timezone)
    else:
        origin = origin.tz_convert(timezone)
    if origin.second or origin.microsecond or origin.minute not in {0, 30}:
        raise ValueError("Origin timestamp must align to a 30-minute boundary")
    return origin


def available_roads(frame: pd.DataFrame) -> tuple[str, ...]:
    """Return the stable sorted corridor road identifiers."""

    roads = tuple(sorted(frame["road_id"].astype(str).unique()))
    if not roads:
        raise RuntimeError("Processed data contains no road identifiers")
    return roads


def latest_common_origin(frame: pd.DataFrame, roads: Sequence[str]) -> pd.Timestamp:
    """Return the latest timestamp containing every requested road."""

    selected = frame.loc[frame["road_id"].isin(roads), ["road_id", "timestamp"]]
    counts = selected.groupby("timestamp", observed=True)["road_id"].nunique()
    matches = counts.loc[counts.eq(len(roads))]
    if matches.empty:
        raise RuntimeError("No common origin exists for the requested roads")
    return pd.Timestamp(matches.index.max())


def select_origin_rows(
    frame: pd.DataFrame,
    roads: Sequence[str],
    origin: pd.Timestamp,
) -> pd.DataFrame:
    """Return one deterministic feature row per requested road and origin."""

    selected = frame.loc[
        frame["road_id"].isin(roads) & frame["timestamp"].eq(origin)
    ].copy()
    if selected["road_id"].nunique() != len(roads) or len(selected) != len(roads):
        missing = sorted(set(roads) - set(selected["road_id"].astype(str)))
        raise ValueError(
            f"Origin is unavailable for requested roads: {missing or list(roads)}"
        )
    if selected.duplicated(["road_id", "timestamp"]).any():
        raise RuntimeError("Origin selection contains duplicate road/timestamp keys")
    return selected.sort_values("road_id", kind="mergesort").reset_index(drop=True)


def recurrent_feature_sequences(
    frame: pd.DataFrame,
    roads: Sequence[str],
    origin: pd.Timestamp,
    sequence_length: int,
    cadence_minutes: int,
    input_features: Sequence[str],
    preprocessor: Any,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Build transformed contiguous road-local sequences ending at origin."""

    sequences: list[np.ndarray] = []
    endpoints: list[pd.DataFrame] = []
    cadence = pd.Timedelta(minutes=int(cadence_minutes))
    expected = pd.date_range(
        end=origin,
        periods=int(sequence_length),
        freq=cadence,
    )
    for road in sorted(str(value) for value in roads):
        history = frame.loc[
            frame["road_id"].eq(road) & frame["timestamp"].le(origin)
        ].sort_values("timestamp", kind="mergesort").tail(int(sequence_length))
        if len(history) != int(sequence_length):
            raise ValueError(
                f"Road {road} lacks {sequence_length} rows of sequence history"
            )
        observed = pd.DatetimeIndex(history["timestamp"])
        if not observed.equals(expected):
            raise ValueError(
                f"Road {road} does not have contiguous 30-minute history at {origin}"
            )
        transformed = np.asarray(
            preprocessor.transform(history[list(input_features)]),
            dtype=np.float32,
        )
        if transformed.ndim != 2 or len(transformed) != int(sequence_length):
            raise RuntimeError("Recurrent preprocessing changed sequence cardinality")
        if not np.isfinite(transformed).all():
            raise RuntimeError("Recurrent preprocessing produced non-finite values")
        sequences.append(transformed)
        endpoints.append(history.tail(1))
    return np.stack(sequences, axis=0), pd.concat(endpoints, ignore_index=True)
