"""Road-isolated, chronological sequence construction for recurrent models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Collection, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


KEY_COLUMNS = ("road_id", "timestamp")


@dataclass(frozen=True)
class TargetScaler:
    """Per-horizon standardization fitted only on eligible training targets."""

    columns: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    fitted_rows: int

    def transform(self, values: Any) -> np.ndarray:
        """Standardize a two-dimensional target matrix."""

        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != len(self.columns):
            raise ValueError("Target matrix does not match scaler columns")
        return ((array - self.mean) / self.scale).astype(np.float32)

    def inverse_transform(self, values: Any) -> np.ndarray:
        """Restore standardized predictions to volume units."""

        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != len(self.columns):
            raise ValueError("Prediction matrix does not match scaler columns")
        return array * self.scale + self.mean

    def metadata(self) -> dict[str, Any]:
        """Return JSON-safe training-only scaling evidence."""

        return {
            "type": "per_horizon_standard",
            "columns": list(self.columns),
            "mean": dict(zip(self.columns, self.mean.tolist(), strict=True)),
            "scale": dict(zip(self.columns, self.scale.tolist(), strict=True)),
            "fitted_rows": self.fitted_rows,
            "source_partition": "train",
        }


@dataclass(frozen=True)
class PreparedPartition:
    """One sorted partition with transformed features and raw targets."""

    name: str
    frame: pd.DataFrame
    features: np.ndarray
    feature_names: tuple[str, ...]
    target_columns: tuple[str, ...]


class RecurrentSequenceDataset(Dataset):
    """Lazy fixed-length sequence views over one transformed partition."""

    def __init__(
        self,
        features: np.ndarray,
        scaled_targets: np.ndarray,
        endpoints: np.ndarray,
        sequence_length: int,
    ) -> None:
        if features.ndim != 2 or scaled_targets.ndim != 2:
            raise ValueError("Features and targets must be two-dimensional")
        if len(features) != len(scaled_targets):
            raise ValueError("Feature and target row counts must match")
        if sequence_length <= 0:
            raise ValueError("Sequence length must be positive")
        endpoint_array = np.asarray(endpoints, dtype=np.int64)
        if endpoint_array.ndim != 1 or not endpoint_array.size:
            raise ValueError("At least one sequence endpoint is required")
        if endpoint_array.min() < sequence_length - 1:
            raise ValueError("A sequence endpoint has insufficient history")
        self.features = np.asarray(features, dtype=np.float32)
        self.targets = np.asarray(scaled_targets, dtype=np.float32)
        self.endpoints = endpoint_array
        self.sequence_length = int(sequence_length)

    def __len__(self) -> int:
        return int(self.endpoints.size)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        end = int(self.endpoints[index])
        start = end - self.sequence_length + 1
        inputs = torch.from_numpy(self.features[start : end + 1])
        target = torch.from_numpy(self.targets[end])
        return inputs, target


def fit_target_scaler(
    frame: pd.DataFrame,
    endpoints: np.ndarray,
    target_columns: Sequence[str],
) -> TargetScaler:
    """Fit per-horizon target statistics on training endpoints only."""

    columns = tuple(str(name) for name in target_columns)
    values = frame.iloc[np.asarray(endpoints, dtype=np.int64)][list(columns)].to_numpy(
        dtype=np.float64
    )
    if not values.size or not np.isfinite(values).all():
        raise ValueError("Training targets for scaling must be finite and non-empty")
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale = np.where(scale > np.finfo(np.float64).eps, scale, 1.0)
    return TargetScaler(
        columns=columns,
        mean=mean,
        scale=scale,
        fitted_rows=int(len(values)),
    )


def prepare_partition(
    name: str,
    frame: pd.DataFrame,
    preprocessor: Any,
    input_features: Sequence[str],
    target_columns: Sequence[str],
) -> PreparedPartition:
    """Sort and transform one already-authorized modelling partition."""

    required = set(KEY_COLUMNS) | set(input_features) | set(target_columns)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Partition is missing recurrent fields: {missing}")
    ordered = frame.sort_values(list(KEY_COLUMNS), kind="mergesort").reset_index(
        drop=True
    )
    if ordered.duplicated(list(KEY_COLUMNS)).any():
        raise RuntimeError("Recurrent partition contains duplicate origin keys")
    transformed = np.asarray(
        preprocessor.transform(ordered[list(input_features)]),
        dtype=np.float32,
    )
    if transformed.ndim != 2 or len(transformed) != len(ordered):
        raise RuntimeError("Recurrent preprocessing changed partition cardinality")
    if not np.isfinite(transformed).all():
        raise RuntimeError("Recurrent preprocessing produced non-finite values")
    names = tuple(str(value) for value in preprocessor.get_feature_names_out())
    if transformed.shape[1] != len(names):
        raise RuntimeError("Recurrent transformed feature schema is inconsistent")
    return PreparedPartition(
        name=name,
        frame=ordered,
        features=transformed,
        feature_names=names,
        target_columns=tuple(str(value) for value in target_columns),
    )


def _eligible_endpoint_mask(
    frame: pd.DataFrame,
    horizons: Sequence[int],
    target_columns: Sequence[str],
    cadence_minutes: int,
) -> np.ndarray:
    mask = np.ones(len(frame), dtype=bool)
    for horizon, target in zip(horizons, target_columns, strict=True):
        available = f"{target}_available"
        within_split = f"target_within_split_h{horizon}"
        target_timestamp = f"target_timestamp_h{horizon}"
        required = {available, within_split, target_timestamp, target}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"Sequence endpoint fields are missing: {missing}")
        mask &= frame[available].fillna(False).astype(bool).to_numpy()
        mask &= frame[within_split].fillna(False).astype(bool).to_numpy()
        values = pd.to_numeric(frame[target], errors="coerce").to_numpy(dtype=float)
        mask &= np.isfinite(values)
        expected = frame["timestamp"] + pd.to_timedelta(
            cadence_minutes * int(horizon),
            unit="min",
        )
        observed = pd.to_datetime(frame[target_timestamp])
        mask &= observed.eq(expected).to_numpy()
    return mask


def build_sequence_endpoints(
    partition: PreparedPartition,
    sequence_length: int,
    horizons: Sequence[int],
    cadence_minutes: int,
    *,
    allowed_keys: Collection[tuple[str, Any]] | None = None,
) -> np.ndarray:
    """Return endpoints whose histories are contiguous and road-local."""

    if sequence_length <= 0:
        raise ValueError("Sequence length must be positive")
    frame = partition.frame
    eligible = _eligible_endpoint_mask(
        frame,
        horizons,
        partition.target_columns,
        cadence_minutes,
    )
    allowed = set(allowed_keys) if allowed_keys is not None else None
    cadence = pd.Timedelta(minutes=int(cadence_minutes))
    endpoints: list[int] = []
    for _, group in frame.groupby("road_id", sort=False, observed=True):
        positions = group.index.to_numpy(dtype=np.int64)
        timestamps = group["timestamp"].reset_index(drop=True)
        run_length = np.ones(len(group), dtype=np.int64)
        if len(group) > 1:
            contiguous = timestamps.diff().iloc[1:].eq(cadence).to_numpy()
            for offset in range(1, len(group)):
                run_length[offset] = (
                    run_length[offset - 1] + 1 if contiguous[offset - 1] else 1
                )
        for offset, position in enumerate(positions):
            if run_length[offset] < sequence_length or not eligible[position]:
                continue
            if allowed is not None:
                key = (
                    str(frame.at[position, "road_id"]),
                    frame.at[position, "timestamp"],
                )
                if key not in allowed:
                    continue
            endpoints.append(int(position))
    if not endpoints:
        raise RuntimeError(f"No eligible {partition.name} recurrent sequences")
    result = np.asarray(endpoints, dtype=np.int64)
    starts = result - int(sequence_length) + 1
    if not frame.iloc[starts]["road_id"].reset_index(drop=True).equals(
        frame.iloc[result]["road_id"].reset_index(drop=True)
    ):
        raise RuntimeError("A recurrent sequence crossed a road boundary")
    return result


def endpoint_keys(
    partition: PreparedPartition,
    endpoints: np.ndarray,
) -> set[tuple[str, Any]]:
    """Return origin keys for candidate-row intersection."""

    rows = partition.frame.iloc[np.asarray(endpoints, dtype=np.int64)]
    return set(zip(rows["road_id"].astype(str), rows["timestamp"], strict=True))


def sequence_manifest(
    partition: PreparedPartition,
    endpoints: np.ndarray,
    sequence_length: int,
    cadence_minutes: int,
) -> dict[str, Any]:
    """Summarize a verified sequence collection for lineage and QA."""

    rows = partition.frame.iloc[np.asarray(endpoints, dtype=np.int64)]
    return {
        "partition": partition.name,
        "sequence_count": int(len(rows)),
        "sequence_length": int(sequence_length),
        "cadence_minutes": int(cadence_minutes),
        "road_count": int(rows["road_id"].nunique()),
        "origin_start": rows["timestamp"].min().isoformat(),
        "origin_end": rows["timestamp"].max().isoformat(),
        "target_columns": list(partition.target_columns),
        "feature_count": int(partition.features.shape[1]),
        "cross_road_sequences": 0,
        "cross_partition_sequences": 0,
        "non_contiguous_sequences": 0,
        "target_boundary_violations": 0,
    }
