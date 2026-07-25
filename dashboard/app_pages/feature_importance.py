"""Model-specific feature driver exploration."""

import streamlit as st

from flowcast.dashboard.analytics import feature_importance
from flowcast.dashboard.cache import get_dashboard_bundle
from flowcast.dashboard.charts import feature_importance_figure
from flowcast.dashboard.config import HORIZON_LABELS
from flowcast.dashboard.state import current_filters
from flowcast.dashboard.ui import render_empty, render_page_header


bundle = get_dashboard_bundle()
filters = current_filters(bundle)
render_page_header(
    "Trace each forecast back to its strongest signals.",
    "Feature drivers come directly from the selected tree and ensemble "
    "artifacts, separated by target and forecast horizon.",
    context="Interpretability · persisted model evidence",
)
with st.container(
    horizontal=True,
    horizontal_alignment="left",
    vertical_alignment="bottom",
):
    target = st.selectbox(
        "Prediction target",
        ["volume", "speed", "travel_time", "congestion", "accident"],
        format_func={
            "volume": "Traffic volume",
            "speed": "Average speed",
            "travel_time": "Travel time",
            "congestion": "Congestion",
            "accident": "Accident risk",
        }.get,
    )
    horizon = st.segmented_control(
        "Horizon",
        filters.horizons,
        default=filters.horizons[0],
        format_func=HORIZON_LABELS.get,
    )
ranked = feature_importance(
    bundle.regression_importance,
    bundle.classification_importance,
    target,
    int(horizon),
)
if ranked.empty:
    render_empty(
        "The selected model family does not expose compatible feature importance."
    )
else:
    left, right = st.columns([1.5, 1], gap="large")
    with left:
        with st.container(border=True):
            st.plotly_chart(
                feature_importance_figure(ranked),
                key=f"feature-importance-{target}-{horizon}",
            )
    with right:
        with st.container(border=True, height="stretch"):
            st.subheader("Leading drivers")
            st.dataframe(
                ranked[["rank", "feature", "importance", "family"]],
                hide_index=True,
                column_config={
                    "rank": st.column_config.NumberColumn("Rank", format="%d"),
                    "feature": st.column_config.TextColumn(
                        "Feature",
                        pinned=True,
                    ),
                    "importance": st.column_config.ProgressColumn(
                        "Importance",
                        min_value=0.0,
                        max_value=float(ranked["importance"].max()),
                        format="%.4f",
                    ),
                    "family": "Model family",
                },
            )
            st.caption(
                f"Importance kind: {ranked['importance_kind'].iloc[0]}. "
                "Values explain the frozen selected estimator, not causality."
            )
