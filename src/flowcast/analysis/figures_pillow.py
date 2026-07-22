"""Pillow fallback for deterministic EDA PNGs on locked-down Windows hosts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


_WHITE = "#FFFFFF"
_INK = "#14213D"
_MUTED = "#64748B"
_GRID = "#D8DEE8"
_NAVY = "#1F3A5F"
_BLUE = "#3B82A0"
_AMBER = "#E09F3E"
_RED = "#C8553D"
_GREEN = "#4F7C65"


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(name, size=size)


def _canvas(
    width: int,
    height: int,
    title: str,
) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (width, height), _WHITE)
    draw = ImageDraw.Draw(image)
    draw.text((48, 28), title, fill=_INK, font=_font(26, bold=True))
    draw.line((48, 70, width - 48, 70), fill=_GRID, width=2)
    return image, draw


def _save(image: Image.Image, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", dpi=(dpi, dpi), optimize=False)


def _axes(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int]) -> None:
    left, top, right, bottom = box
    draw.line((left, top, left, bottom), fill=_MUTED, width=2)
    draw.line((left, bottom, right, bottom), fill=_MUTED, width=2)
    for step in range(1, 5):
        y = bottom - (bottom - top) * step / 5
        draw.line((left, y, right, y), fill=_GRID, width=1)


def _panel_title(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
) -> None:
    draw.text((box[0], box[1] - 34), title, fill=_INK, font=_font(16, bold=True))


def _bars(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    values: Iterable[float],
    colors: Iterable[str],
) -> None:
    values_array = np.asarray(list(values), dtype=float)
    colors_list = list(colors)
    left, top, right, bottom = box
    maximum = max(float(values_array.max()), 1.0)
    slot = (right - left) / max(len(values_array), 1)
    for index, value in enumerate(values_array):
        x0 = left + index * slot + slot * 0.15
        x1 = left + (index + 1) * slot - slot * 0.15
        y0 = bottom - (bottom - top) * float(value) / maximum
        draw.rectangle((x0, y0, x1, bottom), fill=colors_list[index])


def _traffic_distributions(frame: pd.DataFrame, path: Path, dpi: int) -> None:
    image, draw = _canvas(1400, 920, "Core traffic distributions")
    panels = [
        (80, 150, 650, 460),
        (760, 150, 1330, 460),
        (80, 590, 650, 860),
        (760, 590, 1330, 860),
    ]
    fields = [
        ("traffic_volume", "Traffic volume"),
        ("avg_speed", "Average speed"),
        ("occupancy", "Detector occupancy"),
        ("travel_time", "Travel time"),
    ]
    for box, (column, title) in zip(panels, fields, strict=True):
        _panel_title(draw, box, title)
        _axes(draw, box)
        counts, _ = np.histogram(frame[column].dropna().astype(float), bins=28)
        _bars(draw, box, counts, [_BLUE] * len(counts))
        median = float(frame[column].median())
        draw.text(
            (box[0], box[3] + 10),
            f"median {median:.2f}",
            fill=_MUTED,
            font=_font(13),
        )
    _save(image, path, dpi)


def _line_panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    values: np.ndarray,
    title: str,
    color: str,
) -> None:
    _panel_title(draw, box, title)
    _axes(draw, box)
    left, top, right, bottom = box
    low = float(values.min())
    span = max(float(values.max()) - low, 1e-9)
    points = []
    for index, value in enumerate(values):
        x = left + (right - left) * index / max(len(values) - 1, 1)
        y = bottom - (bottom - top) * (float(value) - low) / span
        points.append((x, y))
    draw.line(points, fill=color, width=4, joint="curve")
    for x, y in points:
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)


def _hourly_profiles(contexts: pd.DataFrame, path: Path, dpi: int) -> None:
    hourly = contexts[contexts["dimension"].eq("local_hour")].copy()
    hourly["hour"] = hourly["dimension_value"].astype(int)
    hourly = hourly.sort_values("hour")
    image, draw = _canvas(1500, 560, "Daily traffic profile")
    panels = [(70, 150, 460, 470), (555, 150, 945, 470), (1040, 150, 1430, 470)]
    fields = [
        ("traffic_volume_mean", "Mean volume", _NAVY),
        ("avg_speed_mean", "Mean speed", _GREEN),
        ("travel_time_mean", "Mean travel time", _RED),
    ]
    for box, (field, title, color) in zip(panels, fields, strict=True):
        _line_panel(draw, box, hourly[field].to_numpy(float), title, color)
        draw.text((box[0], box[3] + 12), "00:00", fill=_MUTED, font=_font(12))
        draw.text((box[2] - 38, box[3] + 12), "23:00", fill=_MUTED, font=_font(12))
    _save(image, path, dpi)


def _road_comparison(contexts: pd.DataFrame, path: Path, dpi: int) -> None:
    roads = contexts[contexts["dimension"].eq("road_id")].copy()
    roads = roads.sort_values("traffic_volume_mean")
    image, draw = _canvas(1200, 1000, "Mean traffic volume by road")
    left, top, right, bottom = 180, 120, 1120, 930
    maximum = float(roads["traffic_volume_mean"].max())
    slot = (bottom - top) / len(roads)
    for index, row in enumerate(roads.itertuples(index=False)):
        y0 = top + index * slot + 3
        y1 = top + (index + 1) * slot - 3
        width = (right - left) * float(row.traffic_volume_mean) / maximum
        color = _AMBER if row.traffic_volume_mean == maximum else _NAVY
        draw.rectangle((left, y0, left + width, y1), fill=color)
        draw.text(
            (60, y0 + 2), row.dimension_value, fill=_INK, font=_font(12)
        )
    draw.line((left, top, left, bottom), fill=_MUTED, width=2)
    _save(image, path, dpi)


def _class_balance(
    frame: pd.DataFrame,
    order: list[str],
    path: Path,
    dpi: int,
) -> None:
    image, draw = _canvas(1300, 570, "Classification target balance")
    left_box = (80, 150, 610, 480)
    right_box = (730, 150, 1220, 480)
    congestion = frame["congestion_level"].value_counts().reindex(order, fill_value=0)
    observed = frame["_accident_observed"].fillna(False).astype(bool)
    positive = int((observed & frame["accident_count"].gt(0).fillna(False)).sum())
    accident = [int(observed.sum()) - positive, positive]
    _panel_title(draw, left_box, "Congestion classes")
    _axes(draw, left_box)
    _bars(draw, left_box, congestion.values, [_GREEN, _BLUE, _AMBER, _RED])
    for index, label in enumerate(order):
        draw.text((95 + index * 130, 492), label, fill=_MUTED, font=_font(11))
    _panel_title(draw, right_box, "Accident labels (log count)")
    _axes(draw, right_box)
    _bars(draw, right_box, np.log10(np.asarray(accident) + 1), [_NAVY, _RED])
    draw.text((800, 492), "No accident", fill=_MUTED, font=_font(12))
    draw.text((1080, 492), "Accident", fill=_MUTED, font=_font(12))
    _save(image, path, dpi)


def _weather_traffic(contexts: pd.DataFrame, path: Path, dpi: int) -> None:
    weather = contexts[contexts["dimension"].eq("weather_condition")].copy()
    weather = weather.sort_values("dimension_value")
    image, draw = _canvas(1300, 570, "Weather versus traffic")
    boxes = [(80, 150, 610, 480), (730, 150, 1220, 480)]
    fields = [
        ("traffic_volume_mean", "Mean volume", _BLUE),
        ("avg_speed_mean", "Mean speed", _GREEN),
    ]
    for box, (field, title, color) in zip(boxes, fields, strict=True):
        _panel_title(draw, box, title)
        _axes(draw, box)
        _bars(draw, box, weather[field], [color] * len(weather))
        for index, label in enumerate(weather["dimension_value"]):
            draw.text(
                (box[0] + 20 + index * 100, 492),
                label,
                fill=_MUTED,
                font=_font(10),
            )
    _save(image, path, dpi)


def _correlation_heatmap(correlation: pd.DataFrame, path: Path, dpi: int) -> None:
    size = len(correlation)
    cell = 31
    margin = 260
    extent = margin + size * cell + 80
    image, draw = _canvas(extent, extent, "Configured feature correlation matrix")
    top = 150
    for row, row_name in enumerate(correlation.index):
        draw.text((15, top + row * cell + 7), row_name, fill=_INK, font=_font(10))
        for column, column_name in enumerate(correlation.columns):
            value = float(correlation.loc[row_name, column_name])
            intensity = min(abs(value), 1.0)
            if value >= 0:
                base = (59, 130, 160)
            else:
                base = (200, 85, 61)
            color = tuple(int(255 - (255 - channel) * intensity) for channel in base)
            x = margin + column * cell
            y = top + row * cell
            draw.rectangle((x, y, x + cell - 1, y + cell - 1), fill=color)
    for column, name in enumerate(correlation.columns):
        draw.text(
            (margin + column * cell + 8, 130),
            str(column + 1),
            fill=_MUTED,
            font=_font(10),
        )
    draw.text(
        (margin, extent - 55),
        "Columns follow the numbered matrix order; rows show feature names.",
        fill=_MUTED,
        font=_font(11),
    )
    _save(image, path, dpi)


def generate_pillow_figures(
    frame: pd.DataFrame,
    contexts: pd.DataFrame,
    correlation: pd.DataFrame,
    config: dict[str, Any],
    output_dir: Path,
) -> dict[str, Path]:
    """Generate the complete PNG set without Matplotlib native extensions."""

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
