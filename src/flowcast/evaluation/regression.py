"""Deterministic regression metrics used by classical and deep models."""

from __future__ import annotations

from typing import Any

import numpy as np


def regression_metrics(actual: Any, predicted: Any) -> dict[str, float | int]:
    """Return RMSE, MAE, MAPE, and R-squared with denominator evidence."""

    truth = np.asarray(actual, dtype=np.float64)
    estimates = np.asarray(predicted, dtype=np.float64)
    if truth.ndim != 1 or estimates.ndim != 1 or not truth.size:
        raise ValueError("Regression metrics require non-empty one-dimensional arrays")
    if truth.shape != estimates.shape:
        raise ValueError("Actual and predicted arrays must have the same shape")
    if not np.isfinite(truth).all() or not np.isfinite(estimates).all():
        raise ValueError("Regression metrics require finite values")
    errors = estimates - truth
    absolute_errors = np.abs(errors)
    nonzero = ~np.isclose(truth, 0.0, atol=np.finfo(np.float64).eps)
    if not nonzero.any():
        raise ValueError("MAPE is undefined when every actual value is zero")
    residual_sum = float(np.dot(errors, errors))
    centered = truth - float(truth.mean())
    total_sum = float(np.dot(centered, centered))
    r_squared = 1.0 - residual_sum / total_sum if total_sum else 0.0
    return {
        "rows": int(truth.size),
        "rmse": round(float(np.sqrt(np.mean(errors * errors))), 10),
        "mae": round(float(np.mean(absolute_errors)), 10),
        "mape_percent": round(
            float(np.mean(absolute_errors[nonzero] / np.abs(truth[nonzero])) * 100),
            10,
        ),
        "r_squared": round(r_squared, 10),
        "mape_nonzero_rows": int(nonzero.sum()),
        "mape_zero_actual_rows": int((~nonzero).sum()),
    }
