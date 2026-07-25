"""Side-by-side road performance and reliability comparison."""

import streamlit as st

from flowcast.dashboard.analytics import filter_history, road_summary
from flowcast.dashboard.cache import get_dashboard_bundle
from flowcast.dashboard.charts import comparison_figure
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
comparison = road_summary(history)
render_page_header(
    "Compare roads on the conditions that matter.",
    "Volume, speed, traversal time, severe-congestion exposure, and incident "
    "windows are reconciled over one shared historical window.",
    context="Planning analysis · like-for-like segment evidence",
)
if comparison.empty:
    render_empty("Choose at least one road with observations in this date range.")
else:
    busiest = comparison.iloc[0]
    fastest = comparison.sort_values(
        "mean_speed",
        ascending=False,
        kind="mergesort",
    ).iloc[0]
    most_severe = comparison.sort_values(
        "severe_share",
        ascending=False,
        kind="mergesort",
    ).iloc[0]
    render_metric_row(
        [
            {"label": "Roads compared", "value": f"{len(comparison)}"},
            {
                "label": "Highest mean volume",
                "value": busiest["road_id"],
                "delta": f"{busiest['mean_volume']:.0f} vehicles",
                "delta_color": "off",
            },
            {
                "label": "Highest mean speed",
                "value": fastest["road_id"],
                "delta": f"{fastest['mean_speed']:.1f} km/h",
                "delta_color": "off",
            },
            {
                "label": "Most severe exposure",
                "value": most_severe["road_id"],
                "delta": f"{most_severe['severe_share']:.1%}",
                "delta_color": "inverse",
            },
        ]
    )
    metric = st.segmented_control(
        "Comparison metric",
        ["mean_volume", "mean_speed", "mean_travel_time", "severe_share"],
        default="mean_volume",
        format_func={
            "mean_volume": "Mean volume",
            "mean_speed": "Mean speed",
            "mean_travel_time": "Travel time",
            "severe_share": "Severe share",
        }.get,
    )
    with st.container(border=True):
        st.plotly_chart(
            comparison_figure(
                comparison,
                metric,
                title=metric.replace("_", " ").title(),
            ),
            key=f"road-comparison-{metric}",
        )
    st.dataframe(
        comparison,
        hide_index=True,
        column_config={
            "road_id": st.column_config.TextColumn("Road", pinned=True),
            "road_name": "Road name",
            "mean_volume": st.column_config.NumberColumn(
                "Mean volume",
                format="%.1f",
            ),
            "peak_volume": st.column_config.NumberColumn(
                "Peak volume",
                format="%.0f",
            ),
            "mean_speed": st.column_config.NumberColumn(
                "Mean speed",
                format="%.1f km/h",
            ),
            "mean_travel_time": st.column_config.NumberColumn(
                "Mean travel time",
                format="%.2f min",
            ),
            "severe_share": st.column_config.ProgressColumn(
                "Severe share",
                min_value=0.0,
                max_value=max(0.01, float(comparison["severe_share"].max())),
                format="%.2f",
            ),
            "accident_windows": "Incident windows",
            "observed_windows": "Observed windows",
        },
    )
