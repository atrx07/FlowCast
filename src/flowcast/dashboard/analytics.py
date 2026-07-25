"""Pure analytical transformations for FlowCast dashboard views."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from flowcast.dashboard.config import CONGESTION_ORDER


def filter_history(
    frame: pd.DataFrame,
    roads: Sequence[str],
    start_date: object,
    end_date: object,
) -> pd.DataFrame:
    """Filter verified history without mutating the cached source frame."""

    start = pd.Timestamp(start_date).tz_localize("Asia/Kolkata")
    end = (
        pd.Timestamp(end_date).tz_localize("Asia/Kolkata")
        + pd.Timedelta(days=1)
        - pd.Timedelta(microseconds=1)
    )
    selected = frame.loc[
        frame["road_id"].isin(roads)
        & frame["timestamp"].between(start, end)
    ].copy()
    return selected.sort_values(["timestamp", "road_id"], kind="mergesort")


def corridor_snapshot(predictions: pd.DataFrame) -> dict[str, float | int]:
    """Return operational KPIs from a filtered prediction batch."""

    if predictions.empty:
        return {
            "roads": 0,
            "high_congestion": 0,
            "mean_speed": 0.0,
            "max_risk": 0.0,
        }
    return {
        "roads": int(predictions["road_id"].nunique()),
        "high_congestion": int(
            predictions["congestion_prediction"].isin(["Heavy", "Severe"]).sum()
        ),
        "mean_speed": float(predictions["speed_prediction"].mean()),
        "max_risk": float(predictions["accident_probability"].max()),
    }


def congestion_matrix(
    frame: pd.DataFrame,
    *,
    maximum_timestamps: int = 96,
) -> pd.DataFrame:
    """Return a road-by-time matrix encoded in the canonical severity order."""

    if frame.empty:
        return pd.DataFrame()
    recent = frame.loc[
        frame["timestamp"].isin(
            sorted(frame["timestamp"].unique())[-maximum_timestamps:]
        )
    ].copy()
    severity = {label: index for index, label in enumerate(CONGESTION_ORDER)}
    recent["severity"] = recent["congestion_level"].map(severity)
    return recent.pivot_table(
        index="road_id",
        columns="timestamp",
        values="severity",
        aggfunc="first",
        observed=True,
    ).sort_index()


def road_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate real historical performance by road."""

    if frame.empty:
        return pd.DataFrame()
    grouped = frame.groupby(["road_id", "road_name"], observed=True)
    summary = grouped.agg(
        mean_volume=("traffic_volume", "mean"),
        peak_volume=("traffic_volume", "max"),
        mean_speed=("avg_speed", "mean"),
        mean_travel_time=("travel_time", "mean"),
        severe_share=(
            "congestion_level",
            lambda values: float(values.eq("Severe").mean()),
        ),
        accident_windows=(
            "accident_count",
            lambda values: int(values.fillna(0).gt(0).sum()),
        ),
        observed_windows=("timestamp", "size"),
    )
    return summary.reset_index().sort_values(
        "mean_volume",
        ascending=False,
        kind="mergesort",
    )


def feature_importance(
    regression: pd.DataFrame,
    classification: pd.DataFrame,
    target: str,
    horizon: int,
    *,
    top_n: int = 15,
) -> pd.DataFrame:
    """Select ranked feature drivers for one task and horizon."""

    if target in {"volume", "speed", "travel_time"}:
        source = regression
        task_column = "target"
    else:
        source = classification
        task_column = "task"
    selected = source.loc[
        source[task_column].eq(target)
        & source["horizon_windows"].eq(horizon)
    ].copy()
    return selected.sort_values(
        ["rank", "feature"],
        kind="mergesort",
    ).head(top_n)


def hourly_profile(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate historical traffic volume and speed by half-hour."""

    if frame.empty:
        return pd.DataFrame()
    profile = frame.assign(
        time_of_day=frame["timestamp"].dt.strftime("%H:%M")
    ).groupby(["road_id", "time_of_day"], observed=True).agg(
        traffic_volume=("traffic_volume", "mean"),
        avg_speed=("avg_speed", "mean"),
    )
    return profile.reset_index()


def weather_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate traffic behavior by normalized weather condition."""

    if frame.empty:
        return pd.DataFrame()
    grouped = frame.groupby("weather_condition", observed=True)
    return grouped.agg(
        windows=("timestamp", "size"),
        mean_volume=("traffic_volume", "mean"),
        mean_speed=("avg_speed", "mean"),
        mean_travel_time=("travel_time", "mean"),
        mean_rainfall=("rainfall", "mean"),
        mean_visibility=("visibility", "mean"),
        accident_rate=(
            "accident_count",
            lambda values: float(values.fillna(0).gt(0).mean()),
        ),
    ).reset_index()
