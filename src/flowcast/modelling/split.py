"""Chronological split assignment and training-only time-series CV folds."""

from __future__ import annotations

from typing import Any

import pandas as pd

from flowcast.modelling.config import PARTITIONS


def _timestamp_index(frame: pd.DataFrame) -> pd.DatetimeIndex:
    timestamps = pd.DatetimeIndex(frame["timestamp"].drop_duplicates().sort_values())
    if timestamps.empty or timestamps.tz is None:
        raise RuntimeError("Processed timestamps must be non-empty and timezone-aware")
    return timestamps


def _configured_boundaries(
    config: dict[str, Any],
) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    return {
        name: (
            pd.Timestamp(config["split"]["partitions"][name]["start"]),
            pd.Timestamp(config["split"]["partitions"][name]["end"]),
        )
        for name in PARTITIONS
    }


def _validate_origin_geometry(
    frame: pd.DataFrame,
    timestamps: pd.DatetimeIndex,
    config: dict[str, Any],
) -> None:
    cadence = pd.Timedelta(minutes=int(config["split"]["cadence_minutes"]))
    if len(timestamps) > 1 and not (timestamps[1:] - timestamps[:-1] == cadence).all():
        raise RuntimeError("Processed origin timestamps are not a complete cadence")
    road_count = int(frame["road_id"].nunique())
    per_timestamp = frame.groupby("timestamp", sort=False)["road_id"].nunique()
    if not per_timestamp.eq(road_count).all():
        raise RuntimeError("Every origin timestamp must contain every road")
    if frame.duplicated(["road_id", "timestamp"]).any():
        raise RuntimeError("Processed origins contain duplicate road/timestamp keys")

    expected: list[pd.Timestamp] = []
    for name in PARTITIONS:
        record = config["split"]["partitions"][name]
        start = pd.Timestamp(record["start"])
        end = pd.Timestamp(record["end"])
        partition = pd.date_range(start, end, freq=cadence)
        if len(partition) != int(record["timestamp_count"]):
            raise RuntimeError(f"Configured {name} timestamp count changed")
        expected.extend(partition.tolist())
    if list(timestamps) != expected:
        raise RuntimeError("Actual origin coverage does not match frozen boundaries")


def build_cv_folds(
    timestamps: pd.DatetimeIndex,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build deterministic expanding-window folds within the train partition."""

    train_end = pd.Timestamp(config["split"]["partitions"]["train"]["end"])
    train = timestamps[timestamps <= train_end]
    cv = config["cross_validation"]
    fold_count = int(cv["fold_count"])
    validation_windows = int(cv["validation_windows"])
    gap_windows = int(cv["gap_windows"])
    first_validation = len(train) - fold_count * validation_windows
    folds = []
    for index in range(fold_count):
        validation_start = first_validation + index * validation_windows
        validation_end = validation_start + validation_windows - 1
        training_end = validation_start - gap_windows - 1
        if training_end < 0:
            raise RuntimeError("Time-series CV has no initial training window")
        gap_start = training_end + 1
        gap_end = validation_start - 1
        folds.append(
            {
                "fold": index + 1,
                "train_start": train[0].isoformat(),
                "train_end": train[training_end].isoformat(),
                "train_timestamp_count": training_end + 1,
                "gap_start": train[gap_start].isoformat(),
                "gap_end": train[gap_end].isoformat(),
                "gap_timestamp_count": gap_windows,
                "validation_start": train[validation_start].isoformat(),
                "validation_end": train[validation_end].isoformat(),
                "validation_timestamp_count": validation_windows,
            }
        )
    if pd.Timestamp(folds[-1]["validation_end"]) != train_end:
        raise RuntimeError("Final CV validation fold must end at the train boundary")
    return folds


def _target_coverage(
    frame: pd.DataFrame,
    assignments: pd.DataFrame,
    manifest: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    coverage: dict[str, dict[str, Any]] = {}
    for definition in manifest["targets"]:
        name = str(definition["name"])
        horizon = int(definition["horizon_windows"])
        available = frame[str(definition["availability_column"])].fillna(False)
        within = assignments[f"target_within_split_h{horizon}"]
        eligible = available.astype(bool) & within
        target = frame[name]
        records: dict[str, Any] = {}
        for partition in PARTITIONS:
            selected = assignments["split"].eq(partition)
            usable = selected & eligible
            record: dict[str, Any] = {
                "origin_rows": int(selected.sum()),
                "eligible_rows": int(usable.sum()),
                "boundary_excluded_rows": int((selected & ~within).sum()),
                "label_unavailable_rows": int((selected & within & ~available).sum()),
            }
            if definition["task"] == "classification_binary":
                positives = int(target[usable].astype(bool).sum())
                record["positive_rows"] = positives
                record["negative_rows"] = int(usable.sum()) - positives
            elif definition["task"] == "classification_multiclass":
                record["class_counts"] = {
                    str(label): int(count)
                    for label, count in target[usable].value_counts().sort_index().items()
                }
            records[partition] = record
        coverage[name] = records
    return coverage


def assign_chronological_splits(
    frame: pd.DataFrame,
    manifest: dict[str, Any],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    """Assign every origin and mark horizon targets that stay in its partition."""

    timestamps = _timestamp_index(frame)
    _validate_origin_geometry(frame, timestamps, config)
    boundaries = _configured_boundaries(config)
    assignments = frame[["road_id", "timestamp"]].copy()
    assignments["split"] = pd.Series(pd.NA, index=frame.index, dtype="string")
    for name, (start, end) in boundaries.items():
        assignments.loc[frame["timestamp"].between(start, end), "split"] = name
    if assignments["split"].isna().any():
        raise RuntimeError("Every prediction origin must receive one split")

    for horizon in manifest["forecast_horizons"]:
        target_timestamp = frame[f"target_timestamp_h{horizon}"]
        within = pd.Series(False, index=frame.index, dtype=bool)
        for name, (start, end) in boundaries.items():
            selected = assignments["split"].eq(name)
            within.loc[selected] = target_timestamp.loc[selected].between(start, end)
        assignments[f"target_within_split_h{horizon}"] = within

    road_count = int(frame["road_id"].nunique())
    split_summary: dict[str, Any] = {
        "allocation_method": str(config["split"]["allocation_method"]),
        "target_boundary_policy": str(config["split"]["target_boundary_policy"]),
        "total_rows": len(frame),
        "total_timestamps": len(timestamps),
        "road_count": road_count,
        "partitions": {},
    }
    for name, (start, end) in boundaries.items():
        selected = assignments["split"].eq(name)
        timestamp_count = int(assignments.loc[selected, "timestamp"].nunique())
        split_summary["partitions"][name] = {
            "start": start.isoformat(),
            "end": end.isoformat(),
            "timestamp_count": timestamp_count,
            "row_count": int(selected.sum()),
            "ratio_of_timestamps": round(timestamp_count / len(timestamps), 10),
        }
        if int(selected.sum()) != timestamp_count * road_count:
            raise RuntimeError(f"{name} does not preserve all roads per timestamp")
    split_summary["target_coverage"] = _target_coverage(
        frame, assignments, manifest
    )
    folds = build_cv_folds(timestamps, config)
    return assignments, split_summary, folds
