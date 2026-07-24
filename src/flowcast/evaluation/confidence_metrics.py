"""Core confidence calculations and row-level enrichment."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from flowcast.evaluation.classification import validate_probabilities
from flowcast.evaluation.regression import regression_metrics


CONGESTION_LABELS = ("Free-flow", "Moderate", "Heavy", "Severe")
CONGESTION_PROBABILITIES = (
    "probability_free_flow",
    "probability_moderate",
    "probability_heavy",
    "probability_severe",
)
def add_context(frame: pd.DataFrame, processed: pd.DataFrame) -> pd.DataFrame:
    """Join origin-time and future-congestion context to prediction rows."""

    target_columns = [f"target_congestion_h{horizon}" for horizon in range(1, 5)]
    columns = [
        "road_id",
        "timestamp",
        "hour_of_day",
        "day_of_week",
        "is_weekend",
        "is_morning_peak",
        "is_evening_peak",
        "weather_condition",
        *target_columns,
    ]
    context = processed.loc[:, columns].copy()
    context["origin_hour"] = np.floor(
        context.pop("hour_of_day").astype(float)
    ).astype("int8")
    weekday_names = np.asarray(
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
        dtype=object,
    )
    context["weekday"] = weekday_names[context.pop("day_of_week").astype(int)]
    context["weekday_type"] = np.where(
        context.pop("is_weekend").astype(bool), "weekend", "weekday"
    )
    morning = context.pop("is_morning_peak").astype(bool)
    evening = context.pop("is_evening_peak").astype(bool)
    context["peak_status"] = np.select(
        [morning, evening],
        ["morning_peak", "evening_peak"],
        default="off_peak",
    )
    enriched = frame.merge(
        context,
        on=["road_id", "timestamp"],
        how="left",
        validate="many_to_one",
    )
    if len(enriched) != len(frame):
        raise RuntimeError("Context join changed the prediction row count")
    missing = enriched["origin_hour"].isna()
    if missing.any():
        raise RuntimeError(f"Context join missed {int(missing.sum())} predictions")
    horizon = enriched["horizon_windows"].astype(int).to_numpy()
    congestion = np.empty(len(enriched), dtype=object)
    for value, column in enumerate(target_columns, start=1):
        selected = horizon == value
        congestion[selected] = enriched.loc[selected, column].astype(str)
    enriched["actual_congestion"] = congestion
    return enriched.drop(columns=target_columns)


def conformal_calibration(
    predictions: pd.DataFrame,
    confidence_level: float,
) -> pd.DataFrame:
    """Fit finite-sample split-conformal widths on validation residuals only."""

    validation = predictions.loc[predictions["split"].eq("validation")].copy()
    if validation.empty:
        raise ValueError("Split-conformal calibration requires validation rows")
    rows: list[dict[str, Any]] = []
    keys = ["model_version", "target", "horizon_windows", "horizon_minutes"]
    for key, group in validation.groupby(keys, sort=True, observed=True):
        residuals = np.sort(
            np.abs(
                group["actual"].to_numpy(dtype=float)
                - group["prediction"].to_numpy(dtype=float)
            )
        )
        count = len(residuals)
        rank = min(count, int(math.ceil((count + 1) * confidence_level)))
        rows.append(
            {
                **dict(zip(keys, key, strict=True)),
                "calibration_split": "validation",
                "calibration_rows": count,
                "confidence_level": confidence_level,
                "alpha": 1.0 - confidence_level,
                "finite_sample_rank": rank,
                "absolute_residual_quantile": float(residuals[rank - 1]),
                "quantile_method": "finite_sample_higher",
            }
        )
    return pd.DataFrame(rows)


def enrich_regression(
    predictions: pd.DataFrame,
    calibration: pd.DataFrame,
    processed: pd.DataFrame,
    *,
    clip_lower_at_zero: bool,
) -> pd.DataFrame:
    """Apply frozen conformal widths and attach error/context fields."""

    keys = ["model_version", "target", "horizon_windows", "horizon_minutes"]
    columns = keys + [
        "calibration_rows",
        "confidence_level",
        "absolute_residual_quantile",
        "quantile_method",
    ]
    enriched = predictions.merge(
        calibration.loc[:, columns],
        on=keys,
        how="left",
        validate="many_to_one",
    )
    if enriched["absolute_residual_quantile"].isna().any():
        raise RuntimeError("A regression prediction lacks conformal calibration")
    enriched["signed_error"] = enriched["prediction"] - enriched["actual"]
    enriched["absolute_error"] = enriched["signed_error"].abs()
    enriched["interval_lower"] = (
        enriched["prediction"] - enriched["absolute_residual_quantile"]
    )
    if clip_lower_at_zero:
        enriched["interval_lower"] = enriched["interval_lower"].clip(lower=0.0)
    enriched["interval_upper"] = (
        enriched["prediction"] + enriched["absolute_residual_quantile"]
    )
    enriched["interval_width"] = (
        enriched["interval_upper"] - enriched["interval_lower"]
    )
    enriched["interval_covered"] = (
        enriched["actual"].ge(enriched["interval_lower"])
        & enriched["actual"].le(enriched["interval_upper"])
    )
    return add_context(enriched, processed)


def _entropy(probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    safe = np.clip(probabilities, np.finfo(float).tiny, 1.0)
    entropy = -np.sum(probabilities * np.log(safe), axis=1)
    return entropy, entropy / math.log(probabilities.shape[1])


def enrich_classification(
    predictions: pd.DataFrame,
    processed: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Validate frozen probabilities and attach uncertainty and risk bands."""

    output = predictions.copy()
    size = len(output)
    output["max_probability"] = np.nan
    output["entropy"] = np.nan
    output["normalized_entropy"] = np.nan
    output["confidence_band"] = ""
    output["correct"] = output["actual_class_index"].eq(
        output["predicted_class_index"]
    )
    tolerance = float(config["classification"]["probability_tolerance"])
    for task, probability_columns in (
        ("congestion", CONGESTION_PROBABILITIES),
        ("accident", ("probability_no_accident", "probability_accident")),
    ):
        selected = output["task"].eq(task).to_numpy()
        matrix = output.loc[selected, probability_columns].to_numpy(dtype=float)
        validated = validate_probabilities(
            matrix,
            int(selected.sum()),
            len(probability_columns),
        )
        if not np.allclose(matrix, validated, atol=tolerance, rtol=0.0):
            raise RuntimeError(f"{task} probabilities exceed configured tolerance")
        maximum = validated.max(axis=1)
        entropy, normalized = _entropy(validated)
        output.loc[selected, "max_probability"] = maximum
        output.loc[selected, "entropy"] = entropy
        output.loc[selected, "normalized_entropy"] = normalized

    bands = config["classification"]["confidence_bands"]
    medium = float(bands["medium_minimum"])
    high = float(bands["high_minimum"])
    output["confidence_band"] = np.select(
        [output["max_probability"].ge(high), output["max_probability"].ge(medium)],
        ["high", "medium"],
        default="low",
    )
    output["risk_band"] = pd.Series(pd.NA, index=range(size), dtype="string")
    accident = output["task"].eq("accident")
    thresholds = output.loc[accident, "operating_threshold"].to_numpy(dtype=float)
    if not np.isfinite(thresholds).all():
        raise RuntimeError("Accident predictions require frozen operating thresholds")
    probability = output.loc[accident, "probability_accident"].to_numpy(dtype=float)
    multipliers = config["accident_risk"]["threshold_multipliers"]
    elevated = thresholds * float(multipliers["elevated"])
    critical = np.minimum(1.0, thresholds * float(multipliers["critical"]))
    output.loc[accident, "risk_band"] = np.select(
        [probability < elevated, probability < thresholds, probability < critical],
        ["low", "elevated", "high"],
        default="critical",
    )
    return add_context(output, processed)


def regression_coverage(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate regression accuracy, bias, interval coverage, and width."""

    keys = [
        "model_version",
        "target",
        "horizon_windows",
        "horizon_minutes",
        "split",
    ]
    rows: list[dict[str, Any]] = []
    for key, group in frame.groupby(keys, sort=True, observed=True):
        metrics = regression_metrics(group["actual"], group["prediction"])
        rows.append(
            {
                **dict(zip(keys, key, strict=True)),
                **metrics,
                "bias": float(group["signed_error"].mean()),
                "residual_standard_deviation": float(
                    group["signed_error"].std(ddof=0)
                ),
                "interval_coverage": float(group["interval_covered"].mean()),
                "mean_interval_width": float(group["interval_width"].mean()),
                "confidence_level": float(group["confidence_level"].iloc[0]),
            }
        )
    return pd.DataFrame(rows)


def reliability_table(frame: pd.DataFrame, bin_count: int) -> pd.DataFrame:
    """Build fixed-bin reliability data for confidence and event probabilities."""

    rows: list[dict[str, Any]] = []
    keys = ["task", "horizon_windows", "horizon_minutes", "split", "model_version"]
    for key, group in frame.groupby(keys, sort=True, observed=True):
        task = str(key[0])
        if task == "congestion":
            probabilities = group["max_probability"].to_numpy(dtype=float)
            outcomes = group["correct"].to_numpy(dtype=float)
            reliability_type = "confidence_vs_correctness"
        else:
            probabilities = group["probability_accident"].to_numpy(dtype=float)
            outcomes = group["actual_class_index"].to_numpy(dtype=float)
            reliability_type = "accident_probability_vs_event"
        indices = np.minimum((probabilities * bin_count).astype(int), bin_count - 1)
        aggregate_gap = 0.0
        for bin_index in range(bin_count):
            selected = indices == bin_index
            count = int(selected.sum())
            mean_probability = (
                float(probabilities[selected].mean()) if count else np.nan
            )
            observed_rate = float(outcomes[selected].mean()) if count else np.nan
            gap = (
                abs(mean_probability - observed_rate) if count else np.nan
            )
            if count:
                aggregate_gap += count * gap
            rows.append(
                {
                    **dict(zip(keys, key, strict=True)),
                    "reliability_type": reliability_type,
                    "bin_index": bin_index,
                    "bin_lower": bin_index / bin_count,
                    "bin_upper": (bin_index + 1) / bin_count,
                    "rows": count,
                    "mean_probability": mean_probability,
                    "observed_rate": observed_rate,
                    "absolute_gap": gap,
                    "expected_calibration_error": aggregate_gap / len(group)
                    if bin_index == bin_count - 1
                    else np.nan,
                }
            )
        ece = aggregate_gap / len(group)
        for row in rows[-bin_count:]:
            row["expected_calibration_error"] = ece
    return pd.DataFrame(rows)
