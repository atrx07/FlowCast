"""Deterministic static figures for the Step 09 EDA report."""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None  # type: ignore[assignment]
import numpy as np
import pandas as pd


_NAVY = "#1F3A5F"
_BLUE = "#3B82A0"
_AMBER = "#E09F3E"
_RED = "#C8553D"
_GREEN = "#4F7C65"
_GRID = "#D8DEE8"


def _save(figure: plt.Figure, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        path,
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
        metadata={"Software": "FlowCast"},
    )
    plt.close(figure)


def _style_axes(axis: plt.Axes, grid_axis: str = "y") -> None:
    axis.grid(axis=grid_axis, color=_GRID, linewidth=0.7, alpha=0.8)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)


def _traffic_distributions(frame: pd.DataFrame, path: Path, dpi: int) -> None:
    columns = [
        ("traffic_volume", "Traffic volume", "vehicles / 30 min"),
        ("avg_speed", "Average speed", "km/h"),
        ("occupancy", "Detector occupancy", "%"),
        ("travel_time", "Travel time", "minutes"),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    for axis, (column, title, unit) in zip(axes.flat, columns, strict=True):
        axis.hist(
            frame[column].dropna().astype(float),
            bins=36,
            color=_BLUE,
            edgecolor="white",
            linewidth=0.4,
        )
        axis.axvline(
            float(frame[column].median()),
            color=_AMBER,
            linewidth=1.8,
            label="median",
        )
        axis.set_title(title, loc="left", fontweight="bold")
        axis.set_xlabel(unit)
        axis.set_ylabel("windows")
        axis.legend(frameon=False)
        _style_axes(axis)
    figure.suptitle("Core traffic distributions", fontsize=16, fontweight="bold")
    figure.tight_layout()
    _save(figure, path, dpi)


def _hourly_profiles(contexts: pd.DataFrame, path: Path, dpi: int) -> None:
    hourly = contexts[contexts["dimension"].eq("local_hour")].copy()
    hourly["hour"] = hourly["dimension_value"].astype(int)
    hourly = hourly.sort_values("hour")
    fields = [
        ("traffic_volume_mean", "Mean volume", "vehicles / window", _NAVY),
        ("avg_speed_mean", "Mean speed", "km/h", _GREEN),
        ("travel_time_mean", "Mean travel time", "minutes", _RED),
    ]
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharex=True)
    for axis, (column, title, unit, color) in zip(axes, fields, strict=True):
        axis.plot(hourly["hour"], hourly[column], color=color, linewidth=2.2)
        axis.fill_between(
            hourly["hour"], hourly[column], color=color, alpha=0.12
        )
        axis.set_title(title, loc="left", fontweight="bold")
        axis.set_xlabel("local hour")
        axis.set_ylabel(unit)
        axis.set_xticks([0, 4, 8, 12, 16, 20, 23])
        _style_axes(axis)
    figure.suptitle("Daily traffic profile", fontsize=16, fontweight="bold")
    figure.tight_layout()
    _save(figure, path, dpi)


def _road_comparison(contexts: pd.DataFrame, path: Path, dpi: int) -> None:
    roads = contexts[contexts["dimension"].eq("road_id")].copy()
    roads = roads.sort_values("traffic_volume_mean")
    figure, axis = plt.subplots(figsize=(10, 8))
    colors = np.where(
        roads["traffic_volume_mean"].eq(roads["traffic_volume_mean"].max()),
        _AMBER,
        _NAVY,
    )
    axis.barh(
        roads["dimension_value"],
        roads["traffic_volume_mean"],
        color=colors,
    )
    axis.set_title("Mean traffic volume by road", loc="left", fontweight="bold")
    axis.set_xlabel("vehicles / 30-minute window")
    axis.set_ylabel("road segment")
    _style_axes(axis, grid_axis="x")
    figure.tight_layout()
    _save(figure, path, dpi)


def _class_balance(
    frame: pd.DataFrame,
    congestion_order: list[str],
    path: Path,
    dpi: int,
) -> None:
    congestion = frame["congestion_level"].value_counts().reindex(
        congestion_order, fill_value=0
    )
    observed = frame["_accident_observed"].fillna(False).astype(bool)
    positive = observed & frame["accident_count"].gt(0).fillna(False)
    accident = pd.Series(
        {
            "No accident": int(observed.sum() - positive.sum()),
            "Accident": int(positive.sum()),
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].bar(
        congestion.index,
        congestion.values,
        color=[_GREEN, _BLUE, _AMBER, _RED],
    )
    axes[0].set_title("Congestion class balance", loc="left", fontweight="bold")
    axes[0].set_ylabel("windows")
    axes[0].tick_params(axis="x", rotation=20)
    _style_axes(axes[0])
    axes[1].bar(accident.index, accident.values, color=[_NAVY, _RED])
    axes[1].set_yscale("log")
    axes[1].set_title(
        "Observed accident-label balance (log scale)",
        loc="left",
        fontweight="bold",
    )
    axes[1].set_ylabel("observed windows")
    _style_axes(axes[1])
    figure.tight_layout()
    _save(figure, path, dpi)


def _weather_traffic(contexts: pd.DataFrame, path: Path, dpi: int) -> None:
    weather = contexts[contexts["dimension"].eq("weather_condition")].copy()
    weather = weather.sort_values("traffic_volume_mean", ascending=False)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].bar(
        weather["dimension_value"],
        weather["traffic_volume_mean"],
        color=_BLUE,
    )
    axes[0].set_title("Mean volume by weather", loc="left", fontweight="bold")
    axes[0].set_ylabel("vehicles / window")
    axes[0].tick_params(axis="x", rotation=25)
    _style_axes(axes[0])
    axes[1].bar(
        weather["dimension_value"],
        weather["avg_speed_mean"],
        color=_GREEN,
    )
    axes[1].set_title("Mean speed by weather", loc="left", fontweight="bold")
    axes[1].set_ylabel("km/h")
    axes[1].tick_params(axis="x", rotation=25)
    _style_axes(axes[1])
    figure.tight_layout()
    _save(figure, path, dpi)


def _correlation_heatmap(
    correlation: pd.DataFrame,
    path: Path,
    dpi: int,
) -> None:
    size = len(correlation.columns)
    figure, axis = plt.subplots(figsize=(14, 12))
    image = axis.imshow(correlation.to_numpy(), cmap="coolwarm", vmin=-1, vmax=1)
    axis.set_xticks(range(size), correlation.columns, rotation=90, fontsize=7)
    axis.set_yticks(range(size), correlation.index, fontsize=7)
    axis.set_title(
        "Configured feature correlation matrix", loc="left", fontweight="bold"
    )
    colorbar = figure.colorbar(image, ax=axis, fraction=0.035, pad=0.02)
    colorbar.set_label("Pearson correlation")
    figure.tight_layout()
    _save(figure, path, dpi)


def generate_figures(
    frame: pd.DataFrame,
    contexts: pd.DataFrame,
    correlation: pd.DataFrame,
    config: dict[str, Any],
    output_dir: Path,
) -> dict[str, Path]:
    """Generate all versioned EDA PNGs and return stable named paths."""

    if plt is None:
        from flowcast.analysis.figures_pillow import generate_pillow_figures

        return generate_pillow_figures(
            frame,
            contexts,
            correlation,
            config,
            output_dir,
        )
    plt.style.use(str(config["figures"]["style"]))
    dpi = int(config["figures"]["dpi"])
    paths = {
        "traffic_distributions": output_dir / "traffic_distributions.png",
        "hourly_profiles": output_dir / "hourly_profiles.png",
        "road_comparison": output_dir / "road_comparison.png",
        "class_balance": output_dir / "class_balance.png",
        "weather_traffic": output_dir / "weather_traffic.png",
        "correlation_heatmap": output_dir / "correlation_heatmap.png",
    }
    _traffic_distributions(frame, paths["traffic_distributions"], dpi)
    _hourly_profiles(contexts, paths["hourly_profiles"], dpi)
    _road_comparison(contexts, paths["road_comparison"], dpi)
    _class_balance(frame, config["congestion_order"], paths["class_balance"], dpi)
    _weather_traffic(contexts, paths["weather_traffic"], dpi)
    _correlation_heatmap(correlation, paths["correlation_heatmap"], dpi)
    return paths
