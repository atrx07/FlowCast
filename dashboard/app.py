"""FlowCast Streamlit entry point and shared product shell."""

from pathlib import Path

import streamlit as st

from flowcast.dashboard.cache import get_dashboard_bundle
from flowcast.dashboard.state import (
    initialize_filter_state,
    render_global_filters,
)
from flowcast.dashboard.ui import (
    apply_design_system,
    render_status_strip,
    stop_on_error,
)


st.set_page_config(
    page_title="FlowCast · Northline corridor",
    page_icon=":material/route:",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": (
            "FlowCast v1.0 · verified short-horizon traffic intelligence "
            "for the Northline corridor."
        )
    },
)
apply_design_system()

try:
    bundle = get_dashboard_bundle()
except Exception as error:
    stop_on_error(error)

initialize_filter_state(bundle)
pages = Path(__file__).resolve().parent / "app_pages"
navigation = st.navigation(
    {
        "Operations": [
            st.Page(
                pages / "live_predictions.py",
                title="Live predictions",
                icon=":material/route:",
                default=True,
            ),
            st.Page(
                pages / "congestion_heatmap.py",
                title="Congestion heatmap",
                icon=":material/grid_view:",
            ),
            st.Page(
                pages / "forecast_visualization.py",
                title="Forecast visualization",
                icon=":material/timeline:",
            ),
        ],
        "Analysis": [
            st.Page(
                pages / "historical_trends.py",
                title="Historical trends",
                icon=":material/query_stats:",
            ),
            st.Page(
                pages / "road_comparison.py",
                title="Road comparison",
                icon=":material/compare_arrows:",
            ),
            st.Page(
                pages / "weather_traffic.py",
                title="Weather vs traffic",
                icon=":material/rainy:",
            ),
        ],
        "Models": [
            st.Page(
                pages / "model_performance.py",
                title="Model performance",
                icon=":material/monitoring:",
            ),
            st.Page(
                pages / "feature_importance.py",
                title="Feature importance",
                icon=":material/account_tree:",
            ),
            st.Page(
                pages / "prediction_confidence.py",
                title="Prediction confidence",
                icon=":material/verified:",
            ),
        ],
        "System": [
            st.Page(
                pages / "data_training.py",
                title="Data and training",
                icon=":material/settings_suggest:",
            ),
        ],
    },
    position="sidebar",
    expanded=True,
)
render_global_filters(bundle)
render_status_strip(bundle)
navigation.run()
