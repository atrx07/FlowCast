"""Consistent Plotly figures for FlowCast dashboard pages."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from flowcast.dashboard.config import (
    CONGESTION_COLORS,
    CONGESTION_ORDER,
    HAIRLINE,
    INK,
    INK_SUBTLE,
    PRIMARY,
    PRIMARY_HOVER,
)


def _finish(
    figure: go.Figure,
    *,
    height: int = 380,
    legend: bool = True,
) -> go.Figure:
    figure.update_layout(
        height=height,
        margin=dict(l=12, r=12, t=28, b=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Geist, Segoe UI, sans-serif", color=INK, size=13),
        hoverlabel=dict(bgcolor="#141516", font_color=INK),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        showlegend=legend,
    )
    figure.update_xaxes(gridcolor=HAIRLINE, zeroline=False)
    figure.update_yaxes(gridcolor=HAIRLINE, zeroline=False)
    return figure


def history_figure(
    frame: pd.DataFrame,
    metric: str,
    *,
    title: str,
) -> go.Figure:
    """Build a multi-road historical line figure."""

    figure = px.line(
        frame,
        x="timestamp",
        y=metric,
        color="road_id",
        title=title,
        labels={"timestamp": "Time", metric: title, "road_id": "Road"},
    )
    figure.update_traces(line_width=1.7)
    return _finish(figure, height=430)


def hourly_profile_figure(
    frame: pd.DataFrame,
    metric: str,
    *,
    title: str,
) -> go.Figure:
    """Build a half-hour profile figure."""

    figure = px.line(
        frame,
        x="time_of_day",
        y=metric,
        color="road_id",
        title=title,
        labels={"time_of_day": "Time of day", metric: title, "road_id": "Road"},
    )
    figure.update_traces(line_width=2)
    return _finish(figure, height=360)


def congestion_heatmap_figure(matrix: pd.DataFrame) -> go.Figure:
    """Build a discrete congestion heatmap."""

    colors = [CONGESTION_COLORS[label] for label in CONGESTION_ORDER]
    scale: list[list[float | str]] = []
    for index, color in enumerate(colors):
        lower = index / len(colors)
        upper = (index + 1) / len(colors)
        scale.extend([[lower, color], [upper, color]])
    figure = go.Figure(
        go.Heatmap(
            z=matrix.to_numpy(),
            x=[pd.Timestamp(value) for value in matrix.columns],
            y=matrix.index.astype(str),
            zmin=0,
            zmax=3,
            colorscale=scale,
            colorbar=dict(
                tickvals=[0, 1, 2, 3],
                ticktext=list(CONGESTION_ORDER),
                title="Severity",
            ),
            hovertemplate=(
                "Road %{y}<br>Time %{x|%d %b %H:%M}"
                "<br>Severity index %{z}<extra></extra>"
            ),
        )
    )
    figure.update_xaxes(title="Time")
    figure.update_yaxes(title="Road")
    return _finish(figure, height=620, legend=False)


def corridor_figure(predictions: pd.DataFrame) -> go.Figure:
    """Plot the corridor forecast spatially without external map tiles."""

    figure = px.scatter(
        predictions,
        x="longitude",
        y="latitude",
        color="congestion_prediction",
        size="volume_prediction",
        hover_name="road_name",
        hover_data={
            "road_id": True,
            "speed_prediction": ":.1f",
            "accident_probability": ":.2%",
            "volume_prediction": ":.0f",
            "latitude": False,
            "longitude": False,
        },
        category_orders={"congestion_prediction": list(CONGESTION_ORDER)},
        color_discrete_map=CONGESTION_COLORS,
        labels={
            "congestion_prediction": "Congestion",
            "longitude": "Corridor longitude",
            "latitude": "Corridor latitude",
        },
    )
    figure.update_traces(marker=dict(line=dict(width=1, color="#f7f8f8")))
    return _finish(figure, height=500)


def comparison_figure(
    frame: pd.DataFrame,
    metric: str,
    *,
    title: str,
) -> go.Figure:
    """Build a road comparison bar chart."""

    figure = px.bar(
        frame,
        x="road_id",
        y=metric,
        color="road_id",
        title=title,
        labels={"road_id": "Road", metric: title},
    )
    figure.update_layout(colorway=[PRIMARY, PRIMARY_HOVER, INK_SUBTLE])
    return _finish(figure, height=390, legend=False)


def feature_importance_figure(frame: pd.DataFrame) -> go.Figure:
    """Build a ranked horizontal feature-importance figure."""

    ordered = frame.sort_values("importance", ascending=True, kind="mergesort")
    figure = px.bar(
        ordered,
        x="importance",
        y="feature",
        orientation="h",
        color="importance",
        color_continuous_scale=[[0, "#34385a"], [1, PRIMARY_HOVER]],
        labels={"importance": "Importance", "feature": "Feature"},
    )
    figure.update_layout(coloraxis_showscale=False)
    return _finish(figure, height=470, legend=False)


def performance_figure(frame: pd.DataFrame) -> go.Figure:
    """Build task-aware normalized acceptance performance bars."""

    display = frame.copy()
    minimize = display["metric_direction"].eq("minimize")
    display["performance_ratio"] = display["test_primary_metric"]
    display.loc[minimize, "performance_ratio"] = (
        display.loc[minimize, "acceptance_threshold"]
        / display.loc[minimize, "test_primary_metric"]
    )
    display.loc[~minimize, "performance_ratio"] = (
        display.loc[~minimize, "test_primary_metric"]
        / display.loc[~minimize, "acceptance_threshold"]
    )
    display["performance_ratio"] *= 100
    figure = px.bar(
        display,
        x="job_id",
        y="performance_ratio",
        color="acceptance_met",
        color_discrete_map={True: PRIMARY, False: "#ef6461"},
        labels={
            "job_id": "Target and horizon",
            "performance_ratio": "Target attainment (%)",
            "acceptance_met": "Target met",
        },
        hover_data={
            "test_primary_metric": ":.4f",
            "acceptance_threshold": ":.4f",
            "selected_family": True,
        },
    )
    figure.add_hline(
        y=100,
        line_dash="dot",
        line_color=INK_SUBTLE,
        annotation_text="Formal target",
    )
    return _finish(figure, height=430)


def forecast_figure(
    frame: pd.DataFrame,
    *,
    model: str,
) -> go.Figure:
    """Build actual, prediction, and interval overlays."""

    prediction = f"prediction_{model}"
    lower = f"interval_lower_{model}"
    upper = f"interval_upper_{model}"
    ordered = frame.sort_values("target_timestamp", kind="mergesort")
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=ordered["target_timestamp"],
            y=ordered[upper],
            mode="lines",
            line=dict(width=0),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    figure.add_trace(
        go.Scatter(
            x=ordered["target_timestamp"],
            y=ordered[lower],
            mode="lines",
            fill="tonexty",
            fillcolor="rgba(94,106,210,0.18)",
            line=dict(width=0),
            name="90% interval",
            hoverinfo="skip",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=ordered["target_timestamp"],
            y=ordered["actual"],
            mode="lines",
            name="Actual",
            line=dict(color=INK, width=1.8),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=ordered["target_timestamp"],
            y=ordered[prediction],
            mode="lines",
            name=model.title(),
            line=dict(color=PRIMARY_HOVER, width=2),
        )
    )
    figure.update_xaxes(title="Target time")
    figure.update_yaxes(title="Traffic volume")
    return _finish(figure, height=460)


def reliability_figure(frame: pd.DataFrame) -> go.Figure:
    """Build a calibration reliability figure."""

    clean = frame.dropna(subset=["mean_probability", "observed_rate"]).copy()
    figure = px.line(
        clean,
        x="mean_probability",
        y="observed_rate",
        color="horizon_minutes",
        markers=True,
        labels={
            "mean_probability": "Mean predicted probability",
            "observed_rate": "Observed rate",
            "horizon_minutes": "Horizon",
        },
    )
    figure.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Ideal calibration",
            line=dict(color=INK_SUBTLE, dash="dot"),
        )
    )
    return _finish(figure, height=430)


def weather_figure(frame: pd.DataFrame) -> go.Figure:
    """Build weather-condition volume and speed comparison."""

    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            x=frame["weather_condition"],
            y=frame["mean_volume"],
            name="Mean volume",
            marker_color=PRIMARY,
        )
    )
    figure.add_trace(
        go.Scatter(
            x=frame["weather_condition"],
            y=frame["mean_speed"],
            name="Mean speed",
            mode="lines+markers",
            yaxis="y2",
            line=dict(color=INK, width=2),
        )
    )
    figure.update_layout(
        yaxis=dict(title="Mean traffic volume"),
        yaxis2=dict(
            title="Mean speed (km/h)",
            overlaying="y",
            side="right",
            gridcolor="rgba(0,0,0,0)",
        ),
    )
    return _finish(figure, height=420)
