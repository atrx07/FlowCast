"""Road-by-time congestion severity view."""

import streamlit as st

from flowcast.dashboard.analytics import (
    congestion_matrix,
    filter_history,
)
from flowcast.dashboard.cache import get_dashboard_bundle
from flowcast.dashboard.charts import congestion_heatmap_figure
from flowcast.dashboard.config import CONGESTION_ORDER
from flowcast.dashboard.state import current_filters
from flowcast.dashboard.ui import (
    render_empty,
    render_metric_row,
    render_page_header,
)


bundle = get_dashboard_bundle()
filters = current_filters(bundle)
all_roads = tuple(sorted(bundle.history["road_id"].astype(str).unique()))
scope = st.segmented_control(
    "Heatmap scope",
    ["Selected roads", "Full corridor"],
    default="Full corridor",
)
roads = filters.roads if scope == "Selected roads" else all_roads
history = filter_history(
    bundle.history,
    roads,
    filters.start_date,
    filters.end_date,
)
render_page_header(
    "Congestion, compressed into one operational field.",
    "Every cell is an observed road-window severity using the same ordered "
    "mapping carried through training, evaluation, and live forecasts.",
    context="Corridor heatmap · Free-flow to Severe",
)
if history.empty:
    render_empty("No congestion observations match the selected scope.")
else:
    shares = history["congestion_level"].value_counts(normalize=True)
    render_metric_row(
        [
            {
                "label": label,
                "value": f"{shares.get(label, 0.0):.1%}",
                "help": "Share of visible historical road-window observations.",
            }
            for label in CONGESTION_ORDER
        ]
    )
    matrix = congestion_matrix(history)
    with st.container(border=True):
        st.caption(
            "Showing the most recent 96 half-hour timestamps within the "
            "selected range."
        )
        st.plotly_chart(
            congestion_heatmap_figure(matrix),
            key="congestion-heatmap",
        )
