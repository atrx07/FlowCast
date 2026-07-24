"""Minimum-support error slices and paired-model comparisons."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

import numpy as np
import pandas as pd

from flowcast.evaluation.classification import (
    binary_metrics,
    multiclass_metrics,
)
from flowcast.evaluation.confidence_metrics import CONGESTION_LABELS
from flowcast.evaluation.regression import regression_metrics


def _dimension_groups(
    frame: pd.DataFrame,
    dimensions: Sequence[str],
) -> Iterator[tuple[str, str, pd.DataFrame]]:
    yield "overall", "all", frame
    for dimension in dimensions:
        for value, group in frame.groupby(dimension, sort=True, observed=True):
            yield dimension, str(value), group


def _empty_metrics(names: Sequence[str]) -> dict[str, float]:
    return {name: np.nan for name in names}


def _regression_slice(
    group: pd.DataFrame,
    minimum_rows: int,
) -> dict[str, Any]:
    sufficient = len(group) >= minimum_rows
    names = (
        "rmse",
        "mae",
        "mape_percent",
        "r_squared",
        "bias",
        "residual_standard_deviation",
        "residual_q10",
        "residual_median",
        "residual_q90",
        "interval_coverage",
        "mean_interval_width",
    )
    if not sufficient:
        return {"sufficient_support": False, **_empty_metrics(names)}
    metrics = regression_metrics(group["actual"], group["prediction"])
    errors = group["signed_error"]
    return {
        "sufficient_support": True,
        "rmse": metrics["rmse"],
        "mae": metrics["mae"],
        "mape_percent": metrics["mape_percent"],
        "r_squared": metrics["r_squared"],
        "bias": float(errors.mean()),
        "residual_standard_deviation": float(errors.std(ddof=0)),
        "residual_q10": float(errors.quantile(0.10)),
        "residual_median": float(errors.median()),
        "residual_q90": float(errors.quantile(0.90)),
        "interval_coverage": float(group["interval_covered"].mean()),
        "mean_interval_width": float(group["interval_width"].mean()),
    }


def regression_slices(
    frame: pd.DataFrame,
    dimensions: Sequence[str],
    minimum_rows: int,
) -> pd.DataFrame:
    """Return regression accuracy, bias, residual, and interval slices."""

    rows: list[dict[str, Any]] = []
    base_keys = [
        "model_version",
        "target",
        "horizon_windows",
        "horizon_minutes",
        "split",
    ]
    for key, base in frame.groupby(base_keys, sort=True, observed=True):
        identity = dict(zip(base_keys, key, strict=True))
        for dimension, value, group in _dimension_groups(base, dimensions):
            rows.append(
                {
                    "task_type": "regression",
                    **identity,
                    "dimension": dimension,
                    "slice_value": value,
                    "rows": len(group),
                    "positive_rows": np.nan,
                    **_regression_slice(group, minimum_rows),
                }
            )
    return pd.DataFrame(rows)


def _congestion_slice(
    group: pd.DataFrame,
    minimum_rows: int,
) -> dict[str, Any]:
    sufficient = len(group) >= minimum_rows
    names = (
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "mean_confidence",
        "mean_normalized_entropy",
    )
    if not sufficient:
        return {"sufficient_support": False, **_empty_metrics(names)}
    metrics = multiclass_metrics(
        group["actual_class_index"],
        group["predicted_class_index"],
        CONGESTION_LABELS,
    )
    return {
        "sufficient_support": True,
        "accuracy": metrics["accuracy"],
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
        "macro_f1": metrics["macro_f1"],
        "mean_confidence": float(group["max_probability"].mean()),
        "mean_normalized_entropy": float(group["normalized_entropy"].mean()),
    }


def _accident_slice(
    group: pd.DataFrame,
    minimum_rows: int,
    minimum_positives: int,
) -> dict[str, Any]:
    positives = int(group["actual_class_index"].sum())
    sufficient = (
        len(group) >= minimum_rows
        and positives >= minimum_positives
        and positives < len(group)
    )
    names = (
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
        "prevalence",
        "mean_confidence",
        "mean_normalized_entropy",
    )
    if not sufficient:
        return {"sufficient_support": False, **_empty_metrics(names)}
    threshold = float(group["operating_threshold"].iloc[0])
    if not np.allclose(group["operating_threshold"], threshold):
        raise RuntimeError("An accident slice contains multiple frozen thresholds")
    metrics = binary_metrics(
        group["actual_class_index"],
        group["probability_accident"],
        threshold,
    )
    return {
        "sufficient_support": True,
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "roc_auc": metrics["roc_auc"],
        "pr_auc": metrics["pr_auc"],
        "prevalence": metrics["positive_rate"],
        "mean_confidence": float(group["max_probability"].mean()),
        "mean_normalized_entropy": float(group["normalized_entropy"].mean()),
    }


def classification_slices(
    frame: pd.DataFrame,
    dimensions: Sequence[str],
    minimum_rows: dict[str, int],
    minimum_accident_positives: int,
) -> pd.DataFrame:
    """Return congestion and accident metrics with explicit support flags."""

    rows: list[dict[str, Any]] = []
    keys = [
        "model_version",
        "task",
        "horizon_windows",
        "horizon_minutes",
        "split",
    ]
    for key, base in frame.groupby(keys, sort=True, observed=True):
        identity = dict(zip(keys, key, strict=True))
        task = str(identity["task"])
        for dimension, value, group in _dimension_groups(base, dimensions):
            if task == "congestion":
                metrics = _congestion_slice(group, minimum_rows[task])
                positive_rows: float | int = np.nan
            else:
                metrics = _accident_slice(
                    group,
                    minimum_rows[task],
                    minimum_accident_positives,
                )
                positive_rows = int(group["actual_class_index"].sum())
            rows.append(
                {
                    "task_type": task,
                    **identity,
                    "dimension": dimension,
                    "slice_value": value,
                    "rows": len(group),
                    "positive_rows": positive_rows,
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def confusion_slices(
    frame: pd.DataFrame,
    dimensions: Sequence[str],
    minimum_rows: int,
) -> pd.DataFrame:
    """Persist ordered congestion confusion matrices for supported slices."""

    congestion = frame.loc[frame["task"].eq("congestion")]
    rows: list[dict[str, Any]] = []
    keys = [
        "model_version",
        "horizon_windows",
        "horizon_minutes",
        "split",
    ]
    for key, base in congestion.groupby(keys, sort=True, observed=True):
        identity = dict(zip(keys, key, strict=True))
        for dimension, value, group in _dimension_groups(base, dimensions):
            if len(group) < minimum_rows:
                continue
            matrix = multiclass_metrics(
                group["actual_class_index"],
                group["predicted_class_index"],
                CONGESTION_LABELS,
            )["confusion_matrix"]
            for actual_index, actual_label in enumerate(CONGESTION_LABELS):
                for predicted_index, predicted_label in enumerate(CONGESTION_LABELS):
                    rows.append(
                        {
                            **identity,
                            "dimension": dimension,
                            "slice_value": value,
                            "actual_label": actual_label,
                            "predicted_label": predicted_label,
                            "rows": int(matrix[actual_index][predicted_index]),
                        }
                    )
    return pd.DataFrame(rows)


def accident_risk_bands(
    frame: pd.DataFrame,
    multipliers: dict[str, float],
) -> pd.DataFrame:
    """Aggregate frozen-threshold accident risk bands and observed event rates."""

    accident = frame.loc[frame["task"].eq("accident")]
    base_keys = [
        "model_version",
        "horizon_windows",
        "horizon_minutes",
        "split",
    ]
    rows: list[dict[str, Any]] = []
    for key, base in accident.groupby(base_keys, sort=True, observed=True):
        identity = dict(zip(base_keys, key, strict=True))
        threshold = float(base["operating_threshold"].iloc[0])
        boundaries = (
            0.0,
            threshold * float(multipliers["elevated"]),
            threshold * float(multipliers["high"]),
            min(1.0, threshold * float(multipliers["critical"])),
            1.0,
        )
        for index, band in enumerate(("low", "elevated", "high", "critical")):
            group = base.loc[base["risk_band"].eq(band)]
            rows.append(
                {
                    **identity,
                    "risk_band": band,
                    "probability_lower": boundaries[index],
                    "probability_upper": boundaries[index + 1],
                    "rows": len(group),
                    "event_rows": int(group["actual_class_index"].sum()),
                    "observed_event_rate": (
                        float(group["actual_class_index"].mean())
                        if len(group)
                        else np.nan
                    ),
                    "mean_accident_probability": (
                        float(group["probability_accident"].mean())
                        if len(group)
                        else np.nan
                    ),
                    "operating_threshold": threshold,
                }
            )
    return pd.DataFrame(rows)
