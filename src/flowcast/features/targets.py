"""Leakage-safe multi-horizon target construction for Step 08."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class TargetDefinition:
    """Machine-readable definition for one target and forecast horizon."""

    name: str
    source_column: str
    task: str
    horizon_windows: int
    horizon_minutes: int
    target_timestamp_column: str
    availability_column: str
    transform: str
    availability_source: str | None


@dataclass(frozen=True)
class TargetEngineeringResult:
    """Processed frame and definitions produced from explanatory features."""

    frame: pd.DataFrame
    definitions: tuple[TargetDefinition, ...]
    feature_columns: tuple[str, ...]


def _required_columns(config: dict[str, Any]) -> set[str]:
    required = {"road_id", "timestamp"}
    for definition in config["targets"]["definitions"]:
        required.add(str(definition["source_column"]))
        availability = definition.get("availability_source")
        if availability:
            required.add(str(availability))
    return required


def engineer_targets(
    features: pd.DataFrame,
    config: dict[str, Any],
) -> TargetEngineeringResult:
    """Append exact same-road future targets without removing origin rows."""

    missing = sorted(_required_columns(config) - set(features.columns))
    if missing:
        raise ValueError(f"Target source columns are missing: {missing}")
    if features[["road_id", "timestamp"]].isna().any().any():
        raise ValueError("Target input road/timestamp keys must not be null")
    if features.duplicated(["road_id", "timestamp"]).any():
        raise ValueError("Target input has duplicate road/timestamp keys")

    frame = features.sort_values(
        ["road_id", "timestamp"], kind="stable"
    ).reset_index(drop=True)
    feature_columns = tuple(frame.columns)
    grouped = frame.groupby("road_id", sort=False)
    cadence = int(config["targets"]["cadence_minutes"])
    definitions: list[TargetDefinition] = []

    for horizon in [int(value) for value in config["forecast_horizons_reserved"]]:
        timestamp_column = f"target_timestamp_h{horizon}"
        future_timestamp = grouped["timestamp"].shift(-horizon)
        expected_timestamp = frame["timestamp"] + pd.Timedelta(
            minutes=cadence * horizon
        )
        present_timestamp = future_timestamp.notna()
        if not future_timestamp[present_timestamp].eq(
            expected_timestamp[present_timestamp]
        ).all():
            raise ValueError(
                f"Road timelines are not a complete {cadence}-minute grid"
            )
        frame[timestamp_column] = future_timestamp

        for configured in config["targets"]["definitions"]:
            name = str(configured["name"])
            source = str(configured["source_column"])
            target_column = f"target_{name}_h{horizon}"
            availability_column = f"{target_column}_available"
            future_value = grouped[source].shift(-horizon)
            availability_source = configured.get("availability_source")
            if availability_source:
                future_observed = grouped[str(availability_source)].shift(-horizon)
                available = (
                    present_timestamp
                    & future_observed.fillna(False).astype(bool)
                    & future_value.notna()
                )
                target = future_value.gt(0).astype("boolean")
                transform = "future_count_greater_than_zero"
            else:
                available = present_timestamp & future_value.notna()
                target = future_value
                transform = "same_road_future_shift"
            frame[target_column] = target.where(available, pd.NA)
            frame[availability_column] = pd.array(available, dtype="boolean")
            definitions.append(
                TargetDefinition(
                    name=target_column,
                    source_column=source,
                    task=str(configured["task"]),
                    horizon_windows=horizon,
                    horizon_minutes=cadence * horizon,
                    target_timestamp_column=timestamp_column,
                    availability_column=availability_column,
                    transform=transform,
                    availability_source=(
                        str(availability_source) if availability_source else None
                    ),
                )
            )

    return TargetEngineeringResult(
        frame=frame,
        definitions=tuple(definitions),
        feature_columns=feature_columns,
    )
