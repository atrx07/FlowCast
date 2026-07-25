"""Version-aware Streamlit cache boundaries for dashboard services."""

from __future__ import annotations

import streamlit as st

from flowcast.dashboard.data import (
    DashboardBundle,
    dashboard_fingerprint,
    load_dashboard_bundle,
)
from flowcast.inference import Predictor
from flowcast.settings import load_settings


@st.cache_resource(show_spinner="Verifying FlowCast artifacts...")
def _cached_bundle(_fingerprint: tuple[int, ...]) -> DashboardBundle:
    return load_dashboard_bundle()


def get_dashboard_bundle() -> DashboardBundle:
    """Return one shared verified dashboard bundle until an artifact changes."""

    return _cached_bundle(dashboard_fingerprint())


@st.cache_resource(show_spinner="Loading frozen forecast models...")
def get_predictor(_fingerprint: tuple[int, ...]) -> Predictor:
    """Return one CPU predictor shared across sessions without fitting."""

    return Predictor(load_settings(), device="cpu")
