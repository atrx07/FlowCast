"""Pure statistical summaries for the Step 09 EDA pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CorrelationResult:
    """Correlation matrices and structured redundancy/target summaries."""

    correlation: pd.DataFrame
    covariance: pd.DataFrame
    redundant_pairs: list[dict[str, Any]]
    target_correlations: list[dict[str, Any]]


def _rounded(value: Any, digits: int = 6) -> float | int | None:
    if pd.isna(value):
        return None
    number = value.item() if hasattr(value, "item") else value
    if isinstance(number, (int, np.integer)):
        return int(number)
    return round(float(number), digits)


def descriptive_statistics(
    frame: pd.DataFrame,
    columns: list[str],
) -> dict[str, dict[str, float | int | None]]:
    """Return stable count, location, spread, range, and skew summaries."""

    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"EDA descriptive columns are missing: {missing}")
    summaries: dict[str, dict[str, float | int | None]] = {}
    for column in columns:
        series = pd.to_numeric(frame[column], errors="coerce")
        present = series.dropna()
        summaries[column] = {
            "count": int(present.count()),
            "null_count": int(series.isna().sum()),
            "mean": _rounded(present.mean()),
            "standard_deviation": _rounded(present.std(ddof=1)),
            "variance": _rounded(present.var(ddof=1)),
            "minimum": _rounded(present.min()),
            "q25": _rounded(present.quantile(0.25)),
            "median": _rounded(present.median()),
            "q75": _rounded(present.quantile(0.75)),
            "maximum": _rounded(present.max()),
            "skewness": _rounded(present.skew()),
        }
    return summaries


def target_distributions(
    frame: pd.DataFrame,
    congestion_order: list[str],
) -> dict[str, Any]:
    """Quantify congestion balance and observed accident imbalance."""

    congestion_counts = frame["congestion_level"].value_counts()
    congestion = {
        label: {
            "rows": int(congestion_counts.get(label, 0)),
            "rate": round(float(congestion_counts.get(label, 0) / len(frame)), 8),
        }
        for label in congestion_order
    }
    observed = frame["_accident_observed"].fillna(False).astype(bool)
    positive = observed & frame["accident_count"].gt(0).fillna(False)
    observed_rows = int(observed.sum())
    positive_rows = int(positive.sum())
    accident = {
        "total_rows": len(frame),
        "observed_rows": observed_rows,
        "unobserved_rows": int((~observed).sum()),
        "positive_rows": positive_rows,
        "negative_rows": observed_rows - positive_rows,
        "positive_rate_observed": round(positive_rows / observed_rows, 8),
        "negative_to_positive_ratio": round(
            (observed_rows - positive_rows) / positive_rows, 4
        ),
    }
    return {"congestion": congestion, "accident": accident}


def _context_frame(frame: pd.DataFrame) -> pd.DataFrame:
    selected = frame[
        [
            "road_id",
            "timestamp",
            "weather_condition",
            "public_holiday",
            "event_flag",
            "roadwork_flag",
            "traffic_volume",
            "avg_speed",
            "occupancy",
            "travel_time",
            "accident_count",
            "_accident_observed",
            "congestion_level",
        ]
    ].copy()
    selected["local_hour"] = selected["timestamp"].dt.hour.astype("Int8")
    selected["day_of_week"] = selected["timestamp"].dt.day_name()
    for column in ("public_holiday", "event_flag", "roadwork_flag"):
        selected[column] = np.where(
            selected[column].fillna(False).astype(bool), "yes", "no"
        )
    return selected


def context_aggregates(
    frame: pd.DataFrame,
    dimensions: list[str],
    congestion_order: list[str],
) -> pd.DataFrame:
    """Build long-form real-data aggregates for every required context slice."""

    working = _context_frame(frame)
    records: list[dict[str, Any]] = []
    for dimension in dimensions:
        if dimension not in working:
            raise ValueError(f"EDA context dimension is missing: {dimension}")
        for value, group in working.groupby(dimension, dropna=False, sort=True):
            observed = group["_accident_observed"].fillna(False).astype(bool)
            positive = observed & group["accident_count"].gt(0).fillna(False)
            observed_rows = int(observed.sum())
            record: dict[str, Any] = {
                "dimension": dimension,
                "dimension_value": str(value),
                "row_count": len(group),
                "traffic_volume_mean": _rounded(group["traffic_volume"].mean()),
                "traffic_volume_median": _rounded(
                    group["traffic_volume"].median()
                ),
                "avg_speed_mean": _rounded(group["avg_speed"].mean()),
                "occupancy_mean": _rounded(group["occupancy"].mean()),
                "travel_time_mean": _rounded(group["travel_time"].mean()),
                "accident_observed_rows": observed_rows,
                "accident_positive_rows": int(positive.sum()),
                "accident_positive_rate": (
                    round(float(positive.sum() / observed_rows), 8)
                    if observed_rows
                    else None
                ),
            }
            counts = group["congestion_level"].value_counts()
            for label in congestion_order:
                key = "congestion_" + label.lower().replace("-", "_") + "_rows"
                record[key] = int(counts.get(label, 0))
            records.append(record)
    return pd.DataFrame.from_records(records)


def correlation_analysis(
    frame: pd.DataFrame,
    features: list[str],
    target: str,
    availability: str,
    redundancy_threshold: float,
) -> CorrelationResult:
    """Compute feature covariance/correlation and volume-target associations."""

    required = set(features) | {target, availability}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"EDA correlation columns are missing: {missing}")
    numeric = frame[features].apply(pd.to_numeric, errors="coerce")
    correlation = numeric.corr()
    covariance = numeric.cov()
    redundant_pairs = []
    for left_index, left in enumerate(features):
        for right in features[left_index + 1 :]:
            value = correlation.loc[left, right]
            if pd.notna(value) and abs(float(value)) >= redundancy_threshold:
                redundant_pairs.append(
                    {
                        "feature_a": left,
                        "feature_b": right,
                        "correlation": round(float(value), 8),
                    }
                )
    redundant_pairs.sort(
        key=lambda record: (
            -abs(record["correlation"]),
            record["feature_a"],
            record["feature_b"],
        )
    )
    target_correlations = []
    available = frame[availability].fillna(False).astype(bool)
    target_values = pd.to_numeric(frame[target], errors="coerce")
    for feature in features:
        feature_values = numeric[feature]
        valid = available & feature_values.notna() & target_values.notna()
        value = feature_values[valid].corr(target_values[valid])
        target_correlations.append(
            {
                "feature": feature,
                "target": target,
                "observations": int(valid.sum()),
                "correlation": _rounded(value, 8),
            }
        )
    target_correlations.sort(
        key=lambda record: (
            -abs(record["correlation"] or 0.0),
            record["feature"],
        )
    )
    return CorrelationResult(
        correlation=correlation,
        covariance=covariance,
        redundant_pairs=redundant_pairs,
        target_correlations=target_correlations,
    )


def findings_and_decisions(
    descriptive: dict[str, dict[str, float | int | None]],
    distributions: dict[str, Any],
    contexts: pd.DataFrame,
    correlations: CorrelationResult,
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[str]]:
    """Translate measured EDA results into bounded modelling implications."""

    roads = contexts[contexts["dimension"].eq("road_id")]
    hours = contexts[contexts["dimension"].eq("local_hour")]
    weather = contexts[contexts["dimension"].eq("weather_condition")]
    highest_road = roads.loc[roads["traffic_volume_mean"].idxmax()]
    peak_hour = hours.loc[hours["traffic_volume_mean"].idxmax()]
    slow_weather = weather.loc[weather["avg_speed_mean"].idxmin()]
    accident = distributions["accident"]
    congestion = distributions["congestion"]
    top_target = correlations.target_correlations[0]
    findings = [
        {
            "id": "highest_volume_road",
            "finding": (
                f"{highest_road['dimension_value']} has the highest mean volume "
                f"at {highest_road['traffic_volume_mean']:.2f} vehicles/window."
            ),
        },
        {
            "id": "peak_hour",
            "finding": (
                f"Local hour {peak_hour['dimension_value']} has the highest mean "
                f"volume at {peak_hour['traffic_volume_mean']:.2f}."
            ),
        },
        {
            "id": "slowest_weather",
            "finding": (
                f"{slow_weather['dimension_value']} has the lowest mean speed by "
                f"weather condition at {slow_weather['avg_speed_mean']:.2f} km/h."
            ),
        },
        {
            "id": "congestion_balance",
            "finding": (
                "Free-flow accounts for "
                f"{100 * congestion['Free-flow']['rate']:.2f}% of origins; Severe "
                f"accounts for {100 * congestion['Severe']['rate']:.2f}%."
            ),
        },
        {
            "id": "accident_imbalance",
            "finding": (
                "Observed accident positives are "
                f"{accident['positive_rate_observed'] * 100:.3f}% "
                f"({accident['negative_to_positive_ratio']:.1f}:1 negatives to "
                "positives)."
            ),
        },
        {
            "id": "volume_predictor",
            "finding": (
                f"{top_target['feature']} has the strongest configured linear "
                "association with next-window volume "
                f"(r={top_target['correlation']:.4f})."
            ),
        },
    ]
    decisions = [
        {
            "area": "split",
            "decision": "Use one chronological split; never use a random split.",
            "evidence": (
                "Strong hour/weekday structure and lag dependence are temporal."
            ),
        },
        {
            "area": "scaling",
            "decision": (
                "Fit scaling on training only for linear, SVM, and recurrent models; "
                "retain unscaled inputs for tree models."
            ),
            "evidence": (
                f"Configured numeric ranges differ materially; volume spans "
                f"{descriptive['traffic_volume']['minimum']} to "
                f"{descriptive['traffic_volume']['maximum']}."
            ),
        },
        {
            "area": "congestion",
            "decision": "Select/tune with Macro-F1 and inspect per-class recall.",
            "evidence": "The four congestion classes are materially imbalanced.",
        },
        {
            "area": "accident",
            "decision": (
                "Use training-only class weights, ROC-AUC plus PR-AUC, and select "
                "the operating threshold on validation data."
            ),
            "evidence": findings[4]["finding"],
        },
        {
            "area": "redundancy",
            "decision": (
                "Review highly correlated pairs inside training folds; do not remove "
                "features from full-data EDA alone."
            ),
            "evidence": (
                f"{len(correlations.redundant_pairs)} configured feature pairs meet "
                "the redundancy threshold."
            ),
        },
        {
            "area": "history",
            "decision": (
                "Keep origins and apply model-specific history availability rather "
                "than globally dropping rows."
            ),
            "evidence": (
                "Leading lag/rolling nulls are expected and explicitly flagged."
            ),
        },
    ]
    limitations = [
        "Associations are observational and must not be interpreted as causal effects.",
        (
            "The data covers one corridor and 151 days, limiting geographic and "
            "seasonal generalization."
        ),
        (
            "Reconstructed traffic windows use documented causal recovery and may "
            "smooth short-lived extremes."
        ),
        (
            "Accident status is unknown for inserted sensor windows and must remain "
            "excluded from classifier labels."
        ),
        (
            "Hourly weather is shared by both half-hour traffic windows and cannot "
            "capture sub-hour variation."
        ),
    ]
    return findings, decisions, limitations
