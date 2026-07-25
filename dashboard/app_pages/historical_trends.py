"""Historical traffic and peak-pattern analysis."""

import streamlit as st

from flowcast.dashboard.analytics import (
    filter_history,
    hourly_profile,
)
from flowcast.dashboard.cache import get_dashboard_bundle
from flowcast.dashboard.charts import (
    history_figure,
    hourly_profile_figure,
)
from flowcast.dashboard.state import current_filters
from flowcast.dashboard.ui import (
    render_empty,
    render_metric_row,
    render_page_header,
)


bundle = get_dashboard_bundle()
filters = current_filters(bundle)
history = filter_history(
    bundle.history,
    filters.roads,
    filters.start_date,
    filters.end_date,
)
render_page_header(
    "Read the corridor in motion.",
    "Observed volume and speed stay anchored to the selected roads and "
    "historical window, with peak patterns derived from verified source data.",
    context="Historical telemetry · cleaned and lineage-preserved",
)
if history.empty:
    render_empty("The selected roads have no observations in this date range.")
else:
    render_metric_row(
        [
            {"label": "Observed windows", "value": f"{len(history):,}"},
            {
                "label": "Mean volume",
                "value": f"{history['traffic_volume'].mean():,.0f}",
            },
            {
                "label": "Mean speed",
                "value": f"{history['avg_speed'].mean():.1f} km/h",
            },
            {
                "label": "Peak-period share",
                "value": f"{history['is_peak'].astype(bool).mean():.1%}",
            },
        ]
    )
    metric = st.segmented_control(
        "Timeline metric",
        ["traffic_volume", "avg_speed"],
        default="traffic_volume",
        format_func={
            "traffic_volume": "Traffic volume",
            "avg_speed": "Average speed",
        }.get,
    )
    with st.container(border=True):
        title = "Traffic volume" if metric == "traffic_volume" else "Average speed"
        st.plotly_chart(
            history_figure(history, metric, title=title),
            key=f"history-{metric}",
        )
    profile = hourly_profile(history)
    left, right = st.columns(2, gap="large")
    with left:
        with st.container(border=True):
            st.subheader("Daily volume signature")
            st.plotly_chart(
                hourly_profile_figure(
                    profile,
                    "traffic_volume",
                    title="Mean volume by half-hour",
                ),
                key="history-volume-profile",
            )
    with right:
        with st.container(border=True):
            st.subheader("Speed recovery curve")
            st.plotly_chart(
                hourly_profile_figure(
                    profile,
                    "avg_speed",
                    title="Mean speed by half-hour",
                ),
                key="history-speed-profile",
            )
