"""Deterministic operational insights derived only from prediction rows."""

from __future__ import annotations

from typing import Any

import pandas as pd


def prediction_insights(frame: pd.DataFrame) -> dict[str, Any]:
    """Aggregate one forecast batch into auditable operational findings."""

    if frame.empty:
        raise ValueError("Cannot derive insights from an empty prediction batch")
    ordered = frame.sort_values(
        ["horizon_windows", "road_id"],
        kind="mergesort",
    )
    highest_risk = []
    highest_volume = []
    for horizon, group in ordered.groupby(
        "horizon_windows",
        sort=True,
        observed=True,
    ):
        risk = group.sort_values(
            ["accident_probability", "road_id"],
            ascending=[False, True],
            kind="mergesort",
        ).iloc[0]
        volume = group.sort_values(
            ["volume_prediction", "road_id"],
            ascending=[False, True],
            kind="mergesort",
        ).iloc[0]
        highest_risk.append(
            {
                "horizon_windows": int(horizon),
                "horizon_minutes": int(risk["horizon_minutes"]),
                "road_id": str(risk["road_id"]),
                "road_name": str(risk["road_name"]),
                "probability": float(risk["accident_probability"]),
                "risk_band": str(risk["accident_risk_band"]),
            }
        )
        highest_volume.append(
            {
                "horizon_windows": int(horizon),
                "horizon_minutes": int(volume["horizon_minutes"]),
                "road_id": str(volume["road_id"]),
                "road_name": str(volume["road_name"]),
                "predicted_volume": float(volume["volume_prediction"]),
            }
        )
    congestion_counts = (
        ordered.groupby(
            ["horizon_windows", "congestion_prediction"],
            observed=True,
        )
        .size()
        .rename("roads")
        .reset_index()
        .sort_values(
            ["horizon_windows", "congestion_prediction"],
            kind="mergesort",
        )
    )
    return {
        "road_count": int(ordered["road_id"].nunique()),
        "horizon_count": int(ordered["horizon_windows"].nunique()),
        "forecast_row_count": len(ordered),
        "origin_timestamp": ordered["origin_timestamp"].iloc[0].isoformat(),
        "mean_predicted_volume": float(ordered["volume_prediction"].mean()),
        "mean_predicted_speed": float(ordered["speed_prediction"].mean()),
        "mean_predicted_travel_time": float(
            ordered["travel_time_prediction"].mean()
        ),
        "highest_accident_risk_by_horizon": highest_risk,
        "highest_volume_by_horizon": highest_volume,
        "congestion_counts": congestion_counts.to_dict(orient="records"),
    }
