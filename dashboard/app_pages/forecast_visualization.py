"""Exact-row predicted-versus-actual forecast visualization."""

import streamlit as st

from flowcast.dashboard.cache import get_dashboard_bundle
from flowcast.dashboard.charts import forecast_figure
from flowcast.dashboard.config import HORIZON_LABELS
from flowcast.dashboard.state import current_filters
from flowcast.dashboard.ui import (
    render_empty,
    render_insight_brief,
    render_metric_row,
    render_page_header,
)


bundle = get_dashboard_bundle()
filters = current_filters(bundle)
paired = bundle.confidence.paired_volume
selected = paired.loc[
    paired["split"].eq("test")
    & paired["road_id"].eq(filters.focus_road)
    & paired["horizon_windows"].isin(filters.horizons)
].copy()
render_page_header(
    "Forecasts earn trust on the same held-out road windows.",
    "Actual volume, recurrent and classical predictions, and 90% conformal "
    "intervals share one exact-row test comparison.",
    context="Forecast visualization · sealed test evidence",
)
if selected.empty:
    render_empty("No paired test predictions match the current global filters.")
else:
    horizon = st.segmented_control(
        "Horizon",
        filters.horizons,
        default=filters.horizons[0],
        format_func=HORIZON_LABELS.get,
    )
    model = st.segmented_control(
        "Model",
        ["deep", "classical"],
        default="deep",
        format_func={"deep": "Recurrent", "classical": "Classical"}.get,
    )
    window = selected.loc[selected["horizon_windows"].eq(int(horizon))].tail(192)
    deep_rmse = (
        (window["prediction_deep"] - window["actual"]).pow(2).mean() ** 0.5
    )
    classical_rmse = (
        (window["prediction_classical"] - window["actual"]).pow(2).mean() ** 0.5
    )
    winner = "Recurrent" if deep_rmse < classical_rmse else "Classical"
    rmse_gap = abs(float(deep_rmse - classical_rmse))
    displayed_model = "recurrent" if model == "deep" else "classical"
    render_metric_row(
        [
            {"label": "Road", "value": filters.focus_road},
            {"label": "Visible test rows", "value": f"{len(window):,}"},
            {"label": "Recurrent RMSE", "value": f"{deep_rmse:.2f}"},
            {"label": "Classical RMSE", "value": f"{classical_rmse:.2f}"},
        ]
    )
    render_insight_brief(
        f"For **{filters.focus_road}** at **{HORIZON_LABELS[int(horizon)]}**, "
        f"the **{winner}** forecast has the lower RMSE over the latest "
        f"{len(window):,} held-out rows, by {rmse_gap:.2f} volume units. "
        f"The chart is currently showing the **{displayed_model}** route.",
        guidance=(
            "The observed line is the benchmark; smaller prediction-to-actual "
            "gaps are better, while the shaded interval shows the model's "
            "validation-calibrated 90% uncertainty range."
        ),
        key="forecast",
    )
    with st.container(border=True):
        st.plotly_chart(
            forecast_figure(window, model=model),
            key=f"forecast-{filters.focus_road}-{horizon}-{model}",
        )
        st.caption(
            "The visual window shows the latest 192 eligible test predictions "
            "for readability; summary metrics above use that identical window."
        )
