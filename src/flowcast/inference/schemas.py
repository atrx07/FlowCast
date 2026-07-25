"""Typed inference requests and strict prediction-output validation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Sequence

import numpy as np
import pandas as pd


CONGESTION_LABELS = ("Free-flow", "Moderate", "Heavy", "Severe")
RISK_BANDS = {"low", "elevated", "high", "critical"}
CONFIDENCE_BANDS = {"low", "medium", "high"}
PROBABILITY_COLUMNS = (
    "congestion_probability_free_flow",
    "congestion_probability_moderate",
    "congestion_probability_heavy",
    "congestion_probability_severe",
)
REQUIRED_OUTPUT_COLUMNS = (
    "request_id",
    "road_id",
    "road_name",
    "latitude",
    "longitude",
    "origin_timestamp",
    "target_timestamp",
    "horizon_windows",
    "horizon_minutes",
    "volume_prediction",
    "volume_interval_lower",
    "volume_interval_upper",
    "volume_confidence_level",
    "volume_model_version",
    "volume_classical_comparator",
    "volume_classical_model_version",
    "speed_prediction",
    "speed_interval_lower",
    "speed_interval_upper",
    "speed_confidence_level",
    "speed_model_version",
    "travel_time_prediction",
    "travel_time_interval_lower",
    "travel_time_interval_upper",
    "travel_time_confidence_level",
    "travel_time_model_version",
    "congestion_prediction",
    *PROBABILITY_COLUMNS,
    "congestion_max_probability",
    "congestion_entropy",
    "congestion_normalized_entropy",
    "congestion_confidence_band",
    "congestion_model_version",
    "accident_probability",
    "no_accident_probability",
    "accident_prediction",
    "accident_operating_threshold",
    "accident_risk_band",
    "accident_max_probability",
    "accident_entropy",
    "accident_normalized_entropy",
    "accident_confidence_band",
    "accident_model_version",
    "data_version",
    "feature_version",
    "preprocessing_version",
    "registry_version",
    "confidence_version",
    "inference_version",
    "processed_data_sha256",
)


@dataclass(frozen=True)
class PredictionRequest:
    """One validated set of roads, origin, horizons, and execution device."""

    road_ids: tuple[str, ...]
    origin_timestamp: str
    horizons: tuple[int, ...]
    device: str = "cpu"

    @classmethod
    def from_values(
        cls,
        road_ids: Sequence[str],
        origin_timestamp: str,
        horizons: Sequence[int],
        *,
        device: str = "cpu",
    ) -> PredictionRequest:
        """Normalize basic request values without consulting project data."""

        roads = tuple(sorted(str(value).strip() for value in road_ids))
        if not roads or any(not value for value in roads):
            raise ValueError("At least one non-empty road_id is required")
        if len(set(roads)) != len(roads):
            raise ValueError("road_ids must be unique")
        selected_horizons = tuple(sorted(int(value) for value in horizons))
        if not selected_horizons or len(set(selected_horizons)) != len(
            selected_horizons
        ):
            raise ValueError("horizons must be non-empty and unique")
        return cls(
            road_ids=roads,
            origin_timestamp=str(origin_timestamp),
            horizons=selected_horizons,
            device=str(device).lower(),
        )

    def payload(self) -> dict[str, Any]:
        """Return a deterministic JSON-safe request representation."""

        return {
            "road_ids": list(self.road_ids),
            "origin_timestamp": self.origin_timestamp,
            "horizons": list(self.horizons),
            "device": self.device,
        }

    def identifier(self, inference_version: str) -> str:
        """Return a stable request identifier from normalized input values."""

        payload = {"inference_version": inference_version, **self.payload()}
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:16]


def _finite(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    for column in columns:
        if not np.isfinite(pd.to_numeric(frame[column], errors="coerce")).all():
            raise RuntimeError(f"Prediction output contains non-finite {column}")


def validate_prediction_frame(
    frame: pd.DataFrame,
    request: PredictionRequest,
) -> pd.DataFrame:
    """Validate cardinality, probabilities, physical bounds, and lineage."""

    missing = [name for name in REQUIRED_OUTPUT_COLUMNS if name not in frame]
    if missing:
        raise RuntimeError(f"Prediction output is missing columns: {missing}")
    output = frame.loc[:, REQUIRED_OUTPUT_COLUMNS].copy()
    expected_rows = len(request.road_ids) * len(request.horizons)
    if len(output) != expected_rows:
        raise RuntimeError("Prediction output row count does not match request")
    keys = ["road_id", "origin_timestamp", "horizon_windows"]
    if output.duplicated(keys).any():
        raise RuntimeError("Prediction output contains duplicate forecast keys")
    if set(output["road_id"]) != set(request.road_ids):
        raise RuntimeError("Prediction output road coverage changed")
    if set(output["horizon_windows"].astype(int)) != set(request.horizons):
        raise RuntimeError("Prediction output horizon coverage changed")

    regression = (
        "volume_prediction",
        "volume_classical_comparator",
        "speed_prediction",
        "travel_time_prediction",
    )
    interval_columns = (
        "volume_interval_lower",
        "volume_interval_upper",
        "speed_interval_lower",
        "speed_interval_upper",
        "travel_time_interval_lower",
        "travel_time_interval_upper",
    )
    _finite(output, (*regression, *interval_columns))
    if output[list(regression) + list(interval_columns)].lt(0.0).any(axis=None):
        raise RuntimeError("Physical regression outputs must be non-negative")
    for prefix in ("volume", "speed", "travel_time"):
        if output[f"{prefix}_interval_lower"].gt(
            output[f"{prefix}_prediction"]
        ).any() or output[f"{prefix}_interval_upper"].lt(
            output[f"{prefix}_prediction"]
        ).any():
            raise RuntimeError(f"{prefix} confidence interval is not ordered")

    _finite(
        output,
        (
            *PROBABILITY_COLUMNS,
            "accident_probability",
            "no_accident_probability",
            "accident_operating_threshold",
        ),
    )
    if output[list(PROBABILITY_COLUMNS)].lt(0.0).any(axis=None):
        raise RuntimeError("Congestion probabilities cannot be negative")
    if output[list(PROBABILITY_COLUMNS)].gt(1.0).any(axis=None):
        raise RuntimeError("Congestion probabilities cannot exceed one")
    if not np.allclose(
        output[list(PROBABILITY_COLUMNS)].sum(axis=1),
        1.0,
        atol=1.0e-6,
        rtol=0.0,
    ):
        raise RuntimeError("Congestion probabilities must sum to one")
    accident_sum = output["accident_probability"] + output[
        "no_accident_probability"
    ]
    if not np.allclose(accident_sum, 1.0, atol=1.0e-6, rtol=0.0):
        raise RuntimeError("Accident probabilities must sum to one")
    if not set(output["congestion_prediction"]).issubset(CONGESTION_LABELS):
        raise RuntimeError("Prediction output contains an unknown congestion label")
    if not set(output["accident_risk_band"]).issubset(RISK_BANDS):
        raise RuntimeError("Prediction output contains an unknown risk band")
    for column in (
        "congestion_confidence_band",
        "accident_confidence_band",
    ):
        if not set(output[column]).issubset(CONFIDENCE_BANDS):
            raise RuntimeError(f"Prediction output contains an unknown {column}")
    lineage = (
        "volume_model_version",
        "volume_classical_model_version",
        "speed_model_version",
        "travel_time_model_version",
        "congestion_model_version",
        "accident_model_version",
        "data_version",
        "feature_version",
        "preprocessing_version",
        "registry_version",
        "confidence_version",
        "inference_version",
        "processed_data_sha256",
    )
    if output[list(lineage)].isna().any(axis=None):
        raise RuntimeError("Prediction output contains missing lineage")
    return output.sort_values(
        ["road_id", "horizon_windows"],
        kind="mergesort",
    ).reset_index(drop=True)
