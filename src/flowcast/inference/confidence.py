"""Frozen Step 16 confidence semantics applied to new predictions."""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
import pandas as pd

from flowcast.evaluation.classification import validate_probabilities


def interval_width(
    calibration: pd.DataFrame,
    model_version: str,
    target: str,
    horizon: int,
) -> tuple[float, float]:
    """Return the one frozen conformal width and confidence level."""

    selected = calibration.loc[
        calibration["model_version"].eq(model_version)
        & calibration["target"].eq(target)
        & calibration["horizon_windows"].eq(int(horizon))
    ]
    if len(selected) != 1:
        raise RuntimeError(
            f"Expected one confidence calibration for "
            f"{model_version}/{target}/h{horizon}"
        )
    row = selected.iloc[0]
    width = float(row["absolute_residual_quantile"])
    level = float(row["confidence_level"])
    if not np.isfinite(width) or width < 0.0 or not 0.0 < level < 1.0:
        raise RuntimeError("Frozen regression confidence values are invalid")
    return width, level


def regression_interval(
    prediction: float,
    width: float,
) -> tuple[float, float]:
    """Apply a non-negative split-conformal interval."""

    value = float(prediction)
    return max(0.0, value - float(width)), value + float(width)


def probability_confidence(
    probabilities: Sequence[float],
    config: dict[str, Any],
) -> tuple[float, float, float, str]:
    """Return maximum probability, entropy, normalized entropy, and band."""

    matrix = validate_probabilities(
        np.asarray([probabilities], dtype=float),
        1,
        len(probabilities),
    )
    values = matrix[0]
    maximum = float(values.max())
    safe = np.clip(values, np.finfo(float).tiny, 1.0)
    entropy = float(-np.sum(values * np.log(safe)))
    normalized = float(entropy / math.log(len(values)))
    bands = config["classification"]["confidence_bands"]
    if maximum >= float(bands["high_minimum"]):
        band = "high"
    elif maximum >= float(bands["medium_minimum"]):
        band = "medium"
    else:
        band = "low"
    return maximum, entropy, normalized, band


def accident_risk_band(
    probability: float,
    threshold: float,
    config: dict[str, Any],
) -> str:
    """Apply the frozen threshold-relative Step 16 accident-risk bands."""

    value = float(probability)
    boundary = float(threshold)
    if not 0.0 <= value <= 1.0 or not 0.0 < boundary <= 1.0:
        raise ValueError("Accident probability and threshold must lie in [0, 1]")
    multipliers = config["accident_risk"]["threshold_multipliers"]
    elevated = boundary * float(multipliers["elevated"])
    critical = min(1.0, boundary * float(multipliers["critical"]))
    if value < elevated:
        return "low"
    if value < boundary:
        return "elevated"
    if value < critical:
        return "high"
    return "critical"
