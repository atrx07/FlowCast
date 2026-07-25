"""Shared dashboard design and product constants."""

from __future__ import annotations

from pathlib import Path


APP_TITLE = "FlowCast"
APP_SUBTITLE = "Northline corridor intelligence"
APP_VERSION = "dashboard_v1"
HORIZON_LABELS = {
    1: "30 min",
    2: "60 min",
    3: "90 min",
    4: "120 min",
}
CONGESTION_ORDER = ("Free-flow", "Moderate", "Heavy", "Severe")
CONGESTION_COLORS = {
    "Free-flow": "#27a644",
    "Moderate": "#c6b458",
    "Heavy": "#d99a48",
    "Severe": "#ef6461",
}
RISK_ORDER = ("low", "elevated", "high", "critical")
RISK_COLORS = {
    "low": "#8a8f98",
    "elevated": "#c6b458",
    "high": "#d99a48",
    "critical": "#ef6461",
}
PRIMARY = "#5e6ad2"
PRIMARY_HOVER = "#828fff"
INK = "#f7f8f8"
INK_MUTED = "#d0d6e0"
INK_SUBTLE = "#8a8f98"
CANVAS = "#010102"
SURFACE_1 = "#0f1011"
SURFACE_2 = "#141516"
HAIRLINE = "#23252a"


def page_root() -> Path:
    """Return the dashboard page directory."""

    return Path(__file__).resolve().parents[3] / "dashboard" / "app_pages"
