"""Live and user-triggered frozen-model forecasts."""

import pandas as pd
import streamlit as st

from flowcast.dashboard.analytics import corridor_snapshot
from flowcast.dashboard.cache import (
    get_dashboard_bundle,
    get_predictor,
)
from flowcast.dashboard.charts import corridor_figure
from flowcast.dashboard.config import HORIZON_LABELS
from flowcast.dashboard.data import dashboard_fingerprint
from flowcast.dashboard.state import current_filters
from flowcast.dashboard.ui import (
    render_empty,
    render_lineage,
    render_metric_row,
    render_page_header,
)
from flowcast.inference.artifacts import persist_prediction_batch
from flowcast.reports import build_prediction_reports


bundle = get_dashboard_bundle()
filters = current_filters(bundle)
render_page_header(
    "See the corridor before it changes.",
    "Five targets, four horizons, calibrated confidence, and exact model "
    "lineage from one frozen inference route.",
    context="Near-term operations · latest verified origin",
)

generated = st.session_state.get("fc_generated_predictions")
source = generated if isinstance(generated, pd.DataFrame) else bundle.predictions
horizon = st.segmented_control(
    "Displayed horizon",
    filters.horizons,
    default=filters.horizons[0],
    format_func=HORIZON_LABELS.get,
    key="live_horizon",
)
visible = source.loc[
    source["road_id"].isin(filters.roads)
    & source["horizon_windows"].eq(int(horizon))
].copy()
if visible.empty:
    render_empty("No persisted forecast matches the selected roads and horizon.")
else:
    snapshot = corridor_snapshot(visible)
    render_metric_row(
        [
            {"label": "Roads in view", "value": f"{snapshot['roads']}"},
            {
                "label": "Heavy or severe",
                "value": f"{snapshot['high_congestion']}",
                "help": "Road-horizon rows classified Heavy or Severe.",
            },
            {
                "label": "Mean speed",
                "value": f"{snapshot['mean_speed']:.1f} km/h",
            },
            {
                "label": "Highest accident risk",
                "value": f"{snapshot['max_risk']:.2%}",
            },
        ]
    )
    left, right = st.columns([1.55, 1], gap="large")
    with left:
        with st.container(border=True, key="bento-corridor"):
            st.subheader("Corridor signal")
            st.plotly_chart(
                corridor_figure(visible),
                key=f"live-corridor-{horizon}",
            )
    with right:
        with st.container(
            border=True,
            key="bento-queue",
            height="stretch",
        ):
            st.subheader("Priority queue")
            queue = visible.sort_values(
                ["accident_probability", "volume_prediction"],
                ascending=False,
                kind="mergesort",
            )
            st.dataframe(
                queue[
                    [
                        "road_id",
                        "congestion_prediction",
                        "volume_prediction",
                        "speed_prediction",
                        "accident_probability",
                        "accident_risk_band",
                    ]
                ],
                hide_index=True,
                column_config={
                    "road_id": st.column_config.TextColumn("Road", pinned=True),
                    "congestion_prediction": "Congestion",
                    "volume_prediction": st.column_config.NumberColumn(
                        "Volume",
                        format="%.0f",
                    ),
                    "speed_prediction": st.column_config.NumberColumn(
                        "Speed",
                        format="%.1f km/h",
                    ),
                    "accident_probability": st.column_config.ProgressColumn(
                        "Risk probability",
                        min_value=0.0,
                        max_value=max(
                            0.05,
                            float(queue["accident_probability"].max()),
                        ),
                        format="%.3f",
                    ),
                    "accident_risk_band": "Risk band",
                },
            )

st.space("large")
with st.container(border=True):
    st.subheader("Request a frozen-model forecast")
    st.caption(
        "This control loads persisted models and calibration only. It cannot "
        "fit, retune, or switch an active model."
    )
    available_roads = tuple(sorted(bundle.history["road_id"].astype(str).unique()))
    available_origins = (
        bundle.history["timestamp"]
        .drop_duplicates()
        .sort_values()
        .tail(336)
        .tolist()
    )
    with st.form("prediction-request"):
        request_roads = st.multiselect(
            "Roads",
            available_roads,
            default=list(filters.roads),
            max_selections=25,
        )
        request_horizons = st.pills(
            "Horizons",
            tuple(HORIZON_LABELS),
            default=list(filters.horizons),
            selection_mode="multi",
            format_func=HORIZON_LABELS.get,
            width="stretch",
        )
        request_origin = st.selectbox(
            "Prediction origin",
            available_origins,
            index=len(available_origins) - 1,
            format_func=lambda value: pd.Timestamp(value).strftime(
                "%d %b %Y · %H:%M"
            ),
        )
        submitted = st.form_submit_button(
            "Run prediction",
            type="primary",
            icon=":material/play_arrow:",
        )

    if submitted:
        if not request_roads or not request_horizons:
            st.error("Select at least one road and one horizon.")
        else:
            with st.status("Running verified CPU inference...", expanded=True):
                predictor = get_predictor(dashboard_fingerprint())
                request = predictor.build_request(
                    road_ids=request_roads,
                    origin_timestamp=pd.Timestamp(request_origin).isoformat(),
                    horizons=request_horizons,
                )
                result = predictor.predict(request)
                paths = persist_prediction_batch(result, bundle.settings)
                reports = build_prediction_reports(
                    bundle.settings,
                    paths.manifest_path,
                )
            st.session_state["fc_generated_predictions"] = result.frame
            st.session_state["fc_generated_manifest"] = str(paths.manifest_path)
            st.session_state["fc_generated_report_manifest"] = str(
                reports.manifest_path
            )
            st.toast(
                f"Generated {len(result.frame)} verified forecast rows",
                icon=":material/check_circle:",
            )
            st.rerun()

render_lineage(bundle)
