"""Regression intervals and classifier reliability diagnostics."""

import plotly.express as px
import streamlit as st

from flowcast.dashboard.cache import get_dashboard_bundle
from flowcast.dashboard.charts import reliability_figure
from flowcast.dashboard.config import HORIZON_LABELS
from flowcast.dashboard.state import current_filters
from flowcast.dashboard.ui import (
    render_insight_brief,
    render_metric_row,
    render_page_header,
)


bundle = get_dashboard_bundle()
filters = current_filters(bundle)
coverage = bundle.regression_coverage.loc[
    bundle.regression_coverage["split"].eq("test")
    & bundle.regression_coverage["horizon_windows"].isin(filters.horizons)
].copy()
render_page_header(
    "Confidence is evidence, not decoration.",
    "Validation-calibrated conformal intervals, classifier reliability, "
    "entropy, and rare-event support remain attached to their frozen models.",
    context="Uncertainty · 90% split-conformal and calibrated probabilities",
)
render_metric_row(
    [
        {
            "label": "Mean interval coverage",
            "value": f"{coverage['interval_coverage'].mean():.1%}",
        },
        {
            "label": "Target confidence",
            "value": f"{coverage['confidence_level'].mean():.0%}",
        },
        {
            "label": "Calibration groups",
            "value": (
                f"{bundle.confidence.summary['coverage']['conformal_group_count']}"
            ),
        },
        {
            "label": "Low-support slices",
            "value": (
                f"{bundle.confidence.summary['coverage']['error_slice_row_count'] - bundle.confidence.summary['coverage']['supported_error_slice_rows']}"
            ),
        },
    ]
)
target = st.segmented_control(
    "Regression target",
    ["volume", "speed", "travel_time"],
    default="volume",
    format_func={
        "volume": "Traffic volume",
        "speed": "Average speed",
        "travel_time": "Travel time",
    }.get,
)
target_coverage = coverage.loc[coverage["target"].eq(target)].copy()
lowest_coverage = target_coverage.sort_values(
    ["interval_coverage", "horizon_windows"],
    kind="mergesort",
).iloc[0]
render_insight_brief(
    f"For **{target.replace('_', ' ')}**, observed test coverage spans "
    f"**{target_coverage['interval_coverage'].min():.1%} to "
    f"{target_coverage['interval_coverage'].max():.1%}** across the selected "
    f"horizons. The lowest group is "
    f"{int(lowest_coverage['horizon_minutes'])} minutes at "
    f"{lowest_coverage['interval_coverage']:.1%}.",
    guidance=(
        "Bars close to the dotted 90% line indicate calibrated intervals; "
        "higher is not automatically better because overly wide intervals can "
        "cover more while being less useful."
    ),
    key="coverage",
)
figure = px.bar(
    target_coverage,
    x="horizon_minutes",
    y="interval_coverage",
    color="model_version",
    barmode="group",
    labels={
        "horizon_minutes": "Horizon (minutes)",
        "interval_coverage": "Observed interval coverage",
        "model_version": "Model",
    },
)
figure.add_hline(y=0.9, line_dash="dot", annotation_text="90% target")
figure.update_layout(
    height=410,
    margin=dict(l=10, r=10, t=25, b=10),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
)
with st.container(border=True):
    st.plotly_chart(figure, key=f"coverage-{target}")

task = st.segmented_control(
    "Classifier reliability",
    ["congestion", "accident"],
    default="congestion",
    format_func={
        "congestion": "Congestion confidence",
        "accident": "Accident probability",
    }.get,
)
reliability = bundle.classification_reliability.loc[
    bundle.classification_reliability["split"].eq("test")
    & bundle.classification_reliability["task"].eq(task)
    & bundle.classification_reliability["horizon_windows"].isin(
        filters.horizons
    )
].copy()
summary = (
    reliability.groupby(
        ["horizon_windows", "horizon_minutes"],
        observed=True,
    )
    .agg(
        expected_calibration_error=(
            "expected_calibration_error",
            "first",
        ),
        supported_bins=("rows", lambda values: int(values.gt(0).sum())),
    )
    .reset_index()
)
worst_ece = summary.sort_values(
    ["expected_calibration_error", "horizon_windows"],
    ascending=[False, True],
    kind="mergesort",
).iloc[0]
render_insight_brief(
    f"For **{task}**, the largest expected calibration error is "
    f"**{worst_ece['expected_calibration_error']:.4f}** at "
    f"{int(worst_ece['horizon_minutes'])} minutes, with "
    f"{int(worst_ece['supported_bins'])} populated probability bins.",
    guidance=(
        "Reliability points closer to the diagonal are better calibrated; "
        "bin counts matter because sparse accident evidence makes apparent "
        "gaps less stable."
    ),
    title="Classifier reading",
    key="reliability",
)
left, right = st.columns([1.45, 1], gap="large")
with left:
    with st.container(border=True):
        st.plotly_chart(
            reliability_figure(reliability),
            key=f"reliability-{task}",
        )
with right:
    with st.container(border=True, height="stretch"):
        st.subheader("Reliability evidence")
        st.dataframe(
            summary,
            hide_index=True,
            column_config={
                "horizon_windows": None,
                "horizon_minutes": st.column_config.NumberColumn(
                    "Horizon",
                    format="%d min",
                ),
                "expected_calibration_error": st.column_config.NumberColumn(
                    "ECE",
                    format="%.4f",
                ),
                "supported_bins": "Supported bins",
            },
        )
        if task == "accident":
            st.caption(
                "Accident events are rare. Empty probability bins remain "
                "visible as support evidence rather than being hidden."
            )
