"""Observed weather-versus-traffic relationships."""

import plotly.express as px
import streamlit as st

from flowcast.dashboard.analytics import (
    filter_history,
    weather_summary,
)
from flowcast.dashboard.cache import get_dashboard_bundle
from flowcast.dashboard.charts import weather_figure
from flowcast.dashboard.state import current_filters
from flowcast.dashboard.ui import (
    render_empty,
    render_insight_brief,
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
summary = weather_summary(history)
render_page_header(
    "Weather changes the operating envelope.",
    "Rainfall, visibility, normalized condition, observed speed, flow, and "
    "incident rate are compared over the same selected road windows.",
    context="Weather analysis · observed origin conditions only",
)
if history.empty:
    render_empty("No weather-linked observations match the global filters.")
else:
    rain = history["rainfall"].fillna(0).gt(0)
    low_visibility = history["is_low_visibility"].fillna(False).astype(bool)
    render_metric_row(
        [
            {"label": "Observed windows", "value": f"{len(history):,}"},
            {"label": "Rain windows", "value": f"{rain.mean():.1%}"},
            {
                "label": "Low visibility",
                "value": f"{low_visibility.mean():.1%}",
            },
            {
                "label": "Weather conditions",
                "value": f"{history['weather_condition'].nunique()}",
            },
        ]
    )
    slowest_condition = summary.sort_values(
        ["mean_speed", "weather_condition"],
        kind="mergesort",
    ).iloc[0]
    if rain.any() and (~rain).any():
        rain_speed = float(history.loc[rain, "avg_speed"].mean())
        dry_speed = float(history.loc[~rain, "avg_speed"].mean())
        rain_direction = "lower" if rain_speed < dry_speed else "higher"
        rain_reading = (
            f"Rain windows average {rain_speed:.1f} km/h, "
            f"**{abs(rain_speed - dry_speed):.1f} km/h {rain_direction}** than "
            "dry windows in the selected data."
        )
    else:
        rain_reading = (
            "The selected window does not contain both rain and dry observations, "
            "so a direct speed contrast is not available."
        )
    render_insight_brief(
        f"**{slowest_condition['weather_condition']}** has the lowest observed "
        f"mean speed at {slowest_condition['mean_speed']:.1f} km/h. "
        f"{rain_reading}",
        guidance=(
            "Condition bars summarize observed groups; the scatterplots show "
            "spread and overlap. These are associations, not proof that "
            "weather caused the traffic outcome."
        ),
        key="weather",
    )
    with st.container(border=True):
        st.plotly_chart(weather_figure(summary), key="weather-condition-impact")
    left, right = st.columns(2, gap="large")
    with left:
        with st.container(border=True):
            st.subheader("Rainfall and traffic volume")
            sample = history.sample(
                min(5000, len(history)),
                random_state=bundle.settings.seed,
            )
            rain_figure = px.scatter(
                sample,
                x="rainfall",
                y="traffic_volume",
                color="weather_condition",
                opacity=0.5,
                labels={
                    "rainfall": "Rainfall (mm)",
                    "traffic_volume": "Traffic volume",
                    "weather_condition": "Condition",
                },
            )
            rain_figure.update_layout(
                height=380,
                margin=dict(l=10, r=10, t=20, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(rain_figure, key="rain-volume")
    with right:
        with st.container(border=True):
            st.subheader("Visibility and speed")
            visibility_figure = px.scatter(
                sample,
                x="visibility",
                y="avg_speed",
                color="weather_condition",
                opacity=0.5,
                labels={
                    "visibility": "Visibility (m)",
                    "avg_speed": "Average speed (km/h)",
                    "weather_condition": "Condition",
                },
            )
            visibility_figure.update_layout(
                height=380,
                margin=dict(l=10, r=10, t=20, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(visibility_figure, key="visibility-speed")
    st.dataframe(
        summary,
        hide_index=True,
        column_config={
            "weather_condition": st.column_config.TextColumn(
                "Condition",
                pinned=True,
            ),
            "windows": "Windows",
            "mean_volume": st.column_config.NumberColumn(
                "Mean volume",
                format="%.1f",
            ),
            "mean_speed": st.column_config.NumberColumn(
                "Mean speed",
                format="%.1f km/h",
            ),
            "mean_travel_time": st.column_config.NumberColumn(
                "Travel time",
                format="%.2f min",
            ),
            "mean_rainfall": st.column_config.NumberColumn(
                "Rainfall",
                format="%.2f mm",
            ),
            "mean_visibility": st.column_config.NumberColumn(
                "Visibility",
                format="%.0f m",
            ),
            "accident_rate": st.column_config.ProgressColumn(
                "Incident rate",
                min_value=0.0,
                max_value=max(0.02, float(summary["accident_rate"].max())),
                format="%.4f",
            ),
        },
    )
