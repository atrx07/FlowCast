"""Shared per-session filters for FlowCast dashboard pages."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd
import streamlit as st

from flowcast.dashboard.config import HORIZON_LABELS
from flowcast.dashboard.data import DashboardBundle


@dataclass(frozen=True)
class DashboardFilters:
    """Normalized filters shared across every dashboard page."""

    focus_road: str
    roads: tuple[str, ...]
    horizons: tuple[int, ...]
    start_date: date
    end_date: date


def initialize_filter_state(bundle: DashboardBundle) -> None:
    """Initialize all cross-page state in one place."""

    roads = tuple(sorted(bundle.history["road_id"].astype(str).unique()))
    minimum = pd.Timestamp(bundle.history["timestamp"].min()).date()
    maximum = pd.Timestamp(bundle.history["timestamp"].max()).date()
    default_start = max(minimum, maximum - pd.Timedelta(days=13))
    st.session_state.setdefault("fc_focus_road", roads[0])
    st.session_state.setdefault("fc_roads", list(roads[:3]))
    st.session_state.setdefault("fc_horizons", [1, 2, 3, 4])
    st.session_state.setdefault("fc_date_range", (default_start, maximum))
    st.session_state.setdefault("fc_generated_predictions", None)
    st.session_state.setdefault("fc_upload_validation", None)


def render_global_filters(bundle: DashboardBundle) -> None:
    """Render navigation-adjacent filters used by all pages."""

    roads = tuple(sorted(bundle.history["road_id"].astype(str).unique()))
    minimum = pd.Timestamp(bundle.history["timestamp"].min()).date()
    maximum = pd.Timestamp(bundle.history["timestamp"].max()).date()
    with st.sidebar:
        st.markdown("### Corridor scope")
        st.selectbox(
            "Focus road",
            roads,
            key="fc_focus_road",
            help="Primary segment used by detailed analytical views.",
        )
        st.multiselect(
            "Comparison roads",
            roads,
            key="fc_roads",
            max_selections=8,
            help="Shared road set used by history and comparison views.",
        )
        st.pills(
            "Forecast horizons",
            tuple(HORIZON_LABELS),
            selection_mode="multi",
            format_func=HORIZON_LABELS.get,
            key="fc_horizons",
            width="stretch",
        )
        st.date_input(
            "Historical range",
            min_value=minimum,
            max_value=maximum,
            key="fc_date_range",
        )
        st.caption(
            "All filters are session-local. Persisted datasets and active "
            "model routing remain unchanged."
        )


def current_filters(bundle: DashboardBundle) -> DashboardFilters:
    """Return safe normalized filter values."""

    available = tuple(sorted(bundle.history["road_id"].astype(str).unique()))
    roads = tuple(
        value
        for value in st.session_state.get("fc_roads", [])
        if value in available
    )
    if not roads:
        roads = (str(st.session_state.get("fc_focus_road", available[0])),)
    horizons = tuple(
        int(value)
        for value in st.session_state.get("fc_horizons", [])
        if int(value) in HORIZON_LABELS
    )
    if not horizons:
        horizons = (1,)
    selected = st.session_state.get("fc_date_range")
    if isinstance(selected, (list, tuple)) and len(selected) == 2:
        start, end = selected
    else:
        start = end = pd.Timestamp(bundle.history["timestamp"].max()).date()
    return DashboardFilters(
        focus_road=str(st.session_state.get("fc_focus_road", available[0])),
        roads=roads,
        horizons=horizons,
        start_date=start,
        end_date=end,
    )
