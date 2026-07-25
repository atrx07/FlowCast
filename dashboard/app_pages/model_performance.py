"""Frozen model scoreboards and deep-versus-classical evidence."""

import streamlit as st

from flowcast.dashboard.cache import get_dashboard_bundle
from flowcast.dashboard.charts import performance_figure
from flowcast.dashboard.ui import (
    render_insight_brief,
    render_metric_row,
    render_page_header,
)


bundle = get_dashboard_bundle()
scoreboard = bundle.registry_scoreboard.copy()
comparison = bundle.recurrent_comparison.copy()
render_page_header(
    "Model evidence without selective memory.",
    "Frozen validation choices, sealed-test results, formal target attainment, "
    "and the recurrent-versus-classical benchmark stay visible together.",
    context="Model governance · 20 registered classical entries",
)
met = scoreboard["acceptance_met"].astype(str).str.lower().eq("true")
volume = scoreboard.loc[scoreboard["target"].eq("volume")]
congestion = scoreboard.loc[scoreboard["target"].eq("congestion")]
accident = scoreboard.loc[scoreboard["target"].eq("accident")]
render_metric_row(
    [
        {"label": "Registered models", "value": f"{len(scoreboard)}"},
        {
            "label": "Formal targets met",
            "value": f"{int(met.sum())} / {len(scoreboard)}",
        },
        {
            "label": "Volume target",
            "value": (
                "Met"
                if volume["acceptance_met"].astype(str).str.lower().eq("true").all()
                else "Not met"
            ),
        },
        {
            "label": "Deep wins",
            "value": f"{int(comparison['deep_beats_classical'].sum())} / 4",
            "help": "Exact shared test origins by forecast horizon.",
        },
    ]
)
if not congestion["acceptance_met"].astype(str).str.lower().eq("true").all():
    st.warning(
        "Congestion Macro-F1 remains below the formal 0.80 target. "
        "The dashboard reports the frozen result without retuning.",
        icon=":material/warning:",
    )
if not accident["acceptance_met"].astype(str).str.lower().eq("true").all():
    st.warning(
        "Accident-risk ROC-AUC remains below the formal 0.75 target. "
        "Low prevalence and threshold limitations remain operational risks.",
        icon=":material/warning:",
    )
unmet_targets = []
if not congestion["acceptance_met"].astype(str).str.lower().eq("true").all():
    unmet_targets.append("congestion Macro-F1")
if not accident["acceptance_met"].astype(str).str.lower().eq("true").all():
    unmet_targets.append("accident-risk ROC-AUC")
unmet_text = (
    " and ".join(unmet_targets) + " remain below their formal targets"
    if unmet_targets
    else "all formal targets are met"
)
deep_wins = int(comparison["deep_beats_classical"].sum())
remaining_horizons = 4 - deep_wins
remaining_text = (
    "the remaining horizon"
    if remaining_horizons == 1
    else f"{remaining_horizons} horizons"
)
deep_result = (
    f"wins on {deep_wins} of 4 exact-row test horizons and trails at "
    f"{remaining_text}"
    if deep_wins < 4
    else "wins on all 4 exact-row test horizons"
)
render_insight_brief(
    f"All **{len(volume)} volume models** meet the MAPE goal, while "
    f"**{unmet_text}**. The recurrent volume model {deep_result}.",
    guidance=(
        "In the attainment chart, compare each frozen test metric with its "
        "target marker; the tables below preserve exact validation, test, and "
        "deep-versus-classical evidence."
    ),
    key="performance",
)
with st.container(border=True):
    st.subheader("Formal target attainment")
    st.plotly_chart(
        performance_figure(scoreboard),
        key="model-target-attainment",
    )

left, right = st.columns([1.15, 1], gap="large")
with left:
    with st.container(border=True, height="stretch"):
        st.subheader("Deep versus classical volume")
        st.dataframe(
            comparison,
            hide_index=True,
            column_config={
                "horizon_windows": None,
                "horizon_minutes": st.column_config.NumberColumn(
                    "Horizon",
                    format="%d min",
                ),
                "rows": st.column_config.NumberColumn(
                    "Shared rows",
                    format="%d",
                ),
                "deep_rmse": st.column_config.NumberColumn(
                    "Deep RMSE",
                    format="%.3f",
                ),
                "classical_rmse": st.column_config.NumberColumn(
                    "Classical RMSE",
                    format="%.3f",
                ),
                "rmse_delta_deep_minus_classical": st.column_config.NumberColumn(
                    "Deep minus classical",
                    format="%+.3f",
                ),
                "deep_beats_classical": "Deep wins",
                "origin_mapping_complete": None,
                "actual_values_identical": None,
                "target_timestamps_identical": None,
            },
        )
with right:
    with st.container(border=True, height="stretch"):
        st.subheader("Known 120-minute limitation")
        trailing = comparison.loc[comparison["horizon_windows"].eq(4)].iloc[0]
        st.metric(
            "RMSE delta",
            f"{trailing['rmse_delta_deep_minus_classical']:+.4f}",
            delta="Deep model trails" if not trailing["deep_beats_classical"] else "Deep model wins",
            delta_color="inverse",
            border=True,
        )
        st.caption(
            "This test-period deficit cannot change the validation-led active "
            "route. The classical volume forecast remains visible as fallback."
        )

st.dataframe(
    scoreboard[
        [
            "job_id",
            "target",
            "horizon_minutes",
            "selected_family",
            "primary_metric",
            "validation_primary_metric",
            "test_primary_metric",
            "acceptance_threshold",
            "acceptance_met",
            "model_version",
        ]
    ],
    hide_index=True,
    column_config={
        "job_id": st.column_config.TextColumn("Job", pinned=True),
        "target": "Target",
        "horizon_minutes": st.column_config.NumberColumn(
            "Horizon",
            format="%d min",
        ),
        "selected_family": "Selected family",
        "primary_metric": "Primary metric",
        "validation_primary_metric": st.column_config.NumberColumn(
            "Validation",
            format="%.4f",
        ),
        "test_primary_metric": st.column_config.NumberColumn(
            "Test",
            format="%.4f",
        ),
        "acceptance_threshold": st.column_config.NumberColumn(
            "Target",
            format="%.4f",
        ),
        "acceptance_met": "Met",
        "model_version": "Model version",
    },
)
