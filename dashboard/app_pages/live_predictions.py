"""Live and user-triggered frozen-model forecasts."""

from datetime import datetime, timedelta

import pandas as pd
import streamlit as st

from flowcast.dashboard.analytics import (
    corridor_snapshot,
    eligible_prediction_origins,
    resolve_prediction_origin,
)
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
    render_insight_brief,
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
available_horizons = tuple(
    value
    for value in filters.horizons
    if value in set(source["horizon_windows"].astype(int))
)
display_horizons = available_horizons or filters.horizons
with st.container(key="live-horizon"):
    horizon = st.segmented_control(
        "Displayed horizon",
        display_horizons,
        default=display_horizons[0],
        format_func=HORIZON_LABELS.get,
        key="live_horizon",
    )
visible = source.loc[
    source["road_id"].isin(filters.roads)
    & source["horizon_windows"].eq(int(horizon))
].copy()

if not visible.empty:
    snapshot = corridor_snapshot(visible)
    queue = visible.sort_values(
        ["accident_probability", "volume_prediction"],
        ascending=False,
        kind="mergesort",
    )
    risk_leader = queue.iloc[0]
    slowest = visible.sort_values(
        ["speed_prediction", "road_id"],
        kind="mergesort",
    ).iloc[0]
    with st.container(key="live-kpis"):
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

available_roads = tuple(sorted(bundle.history["road_id"].astype(str).unique()))
request_config = bundle.context.config["request"]
available_origins = eligible_prediction_origins(
    bundle.history,
    sequence_length=int(request_config["recurrent_sequence_length"]),
    cadence_minutes=int(request_config["cadence_minutes"]),
)
first_origin = pd.Timestamp(available_origins[0])
latest_origin = pd.Timestamp(available_origins[-1])
with st.container(border=True, key="prediction-workflow"):
    st.subheader("Request a frozen-model forecast")
    st.caption(
        "Choose the last observed traffic window. FlowCast predicts the "
        "selected 30–120 minute horizons after it. Eligible origins "
        f"{first_origin.strftime('%d %b %Y · %H:%M')} to "
        f"{latest_origin.strftime('%d %b %Y · %H:%M')} · "
        f"{len(available_origins):,} half-hour slots."
    )
    with st.form("prediction-request", border=False):
        roads_column, date_column, time_column = st.columns(
            [1.6, 0.8, 0.7],
            gap="small",
            vertical_alignment="bottom",
        )
        with roads_column:
            request_roads = st.multiselect(
                "Roads",
                available_roads,
                default=list(filters.roads),
                max_selections=25,
            )
        with date_column:
            request_origin_date = st.date_input(
                "Forecast origin date",
                value=latest_origin.date(),
                min_value=first_origin.date(),
                max_value=latest_origin.date(),
                format="DD/MM/YYYY",
                key="prediction_origin_date",
                help=(
                    "This is the last observed date available to the model, "
                    "not the future target date."
                ),
            )
        with time_column:
            request_origin_time = st.time_input(
                "Forecast origin time",
                value=latest_origin.time(),
                step=timedelta(
                    minutes=int(request_config["cadence_minutes"])
                ),
                key="prediction_origin_time",
                help=(
                    "Forecast origins follow the source data's 30-minute "
                    "cadence."
                ),
            )
        horizon_column, action_column = st.columns(
            [3.15, 0.85],
            gap="small",
            vertical_alignment="bottom",
        )
        with horizon_column:
            request_horizons = st.pills(
                "Horizons",
                tuple(HORIZON_LABELS),
                default=list(filters.horizons),
                selection_mode="multi",
                format_func=HORIZON_LABELS.get,
                width="stretch",
            )
            selected_targets = [
                pd.Timestamp(
                    datetime.combine(
                        request_origin_date,
                        request_origin_time,
                    )
                )
                + pd.Timedelta(
                    minutes=int(request_config["cadence_minutes"]) * horizon
                )
                for horizon in sorted(request_horizons or ())
            ]
            if selected_targets:
                target_dates = {target.date() for target in selected_targets}
                if len(target_dates) == 1:
                    target_preview = (
                        f"{selected_targets[0].strftime('%d %b %Y')} · "
                        + ", ".join(
                            target.strftime("%H:%M")
                            for target in selected_targets
                        )
                    )
                else:
                    target_preview = ", ".join(
                        target.strftime("%d %b %Y · %H:%M")
                        for target in selected_targets
                    )
                st.caption(
                    f"Predicting {target_preview}. "
                    "The origin is the last observed window."
                )
            else:
                st.caption(
                    "Select at least one horizon to preview forecast targets."
                )
        with action_column:
            submitted = st.form_submit_button(
                "Run prediction",
                type="primary",
                icon=":material/play_arrow:",
                width="stretch",
            )

    if submitted:
        if not request_roads or not request_horizons:
            st.error("Select at least one road and one horizon.")
        else:
            try:
                request_origin = resolve_prediction_origin(
                    request_origin_date,
                    request_origin_time,
                    available_origins,
                )
            except ValueError as error:
                st.error(str(error))
            else:
                with st.status(
                    "Running verified CPU inference...",
                    expanded=True,
                ):
                    predictor = get_predictor(dashboard_fingerprint())
                    request = predictor.build_request(
                        road_ids=request_roads,
                        origin_timestamp=request_origin.isoformat(),
                        horizons=request_horizons,
                    )
                    result = predictor.predict(request)
                    paths = persist_prediction_batch(result, bundle.settings)
                    reports = build_prediction_reports(
                        bundle.settings,
                        paths.manifest_path,
                    )
                st.session_state["fc_generated_predictions"] = result.frame
                st.session_state["fc_generated_manifest"] = str(
                    paths.manifest_path
                )
                st.session_state["fc_generated_report_manifest"] = str(
                    reports.manifest_path
                )
                st.toast(
                    f"Generated {len(result.frame)} verified forecast rows",
                    icon=":material/check_circle:",
                )
                st.rerun()

if visible.empty:
    render_empty("No persisted forecast matches the selected roads and horizon.")
    render_insight_brief(
        "No persisted forecast matches the current road and horizon filters.",
        guidance=(
            "Choose a road and horizon covered by the latest persisted batch, "
            "or request a new frozen-model forecast above."
        ),
        key="live-empty",
    )
else:
    congestion_reading = (
        "No selected roads are forecast as Heavy or Severe"
        if snapshot["high_congestion"] == 0
        else (
            f"{snapshot['high_congestion']} selected road"
            f"{'s are' if snapshot['high_congestion'] != 1 else ' is'} "
            "forecast as Heavy or Severe"
        )
    )
    render_insight_brief(
        f"At **{HORIZON_LABELS[int(horizon)]}**, {congestion_reading}. "
        f"**{risk_leader['road_id']}** has the highest modeled accident "
        f"probability at {risk_leader['accident_probability']:.2%}; "
        f"**{slowest['road_id']}** has the lowest predicted speed at "
        f"{slowest['speed_prediction']:.1f} km/h.",
        guidance=(
            "The corridor plot locates each road and sizes it by forecast "
            "volume; the queue ranks the same rows by modeled accident risk."
        ),
        key="live",
    )
    left, right = st.columns([1.6, 1], gap="medium")
    with left:
        with st.container(border=True, key="bento-corridor"):
            st.subheader("Corridor signal")
            st.caption(
                "Each marker is a road at the selected horizon. Position uses "
                "verified corridor coordinates, size shows predicted volume, "
                "and color shows predicted congestion."
            )
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
            st.caption(
                "Highest modeled accident probability first for the same road "
                "and horizon selection."
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

render_lineage(bundle)
