"""Unit tests for reproducible Step 09 EDA calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd

from flowcast.analysis.config import load_eda_config
from flowcast.analysis.statistics import (
    context_aggregates,
    correlation_analysis,
    descriptive_statistics,
    target_distributions,
)
from flowcast.settings import load_settings


def _frame() -> pd.DataFrame:
    timestamps = pd.date_range(
        "2025-01-06 07:00",
        periods=8,
        freq="30min",
        tz="Asia/Kolkata",
    )
    return pd.DataFrame(
        {
            "road_id": ["NL-001"] * 4 + ["NL-002"] * 4,
            "timestamp": list(timestamps[:4]) + list(timestamps[:4]),
            "weather_condition": ["Clear", "Rain"] * 4,
            "public_holiday": [0, 0, 1, 1] * 2,
            "event_flag": [0, 1, 0, 1] * 2,
            "roadwork_flag": [0, 0, 0, 1] * 2,
            "traffic_volume": pd.array(range(1, 9), dtype="Int64"),
            "avg_speed": pd.array(np.linspace(20, 27, 8), dtype="Float64"),
            "occupancy": pd.array(np.linspace(30, 37, 8), dtype="Float64"),
            "travel_time": pd.array(np.linspace(8, 15, 8), dtype="Float64"),
            "temperature": pd.array(np.linspace(15, 22, 8), dtype="Float64"),
            "rainfall": pd.array([0, 1] * 4, dtype="Float64"),
            "visibility": pd.array([10000, 800] * 4, dtype="Float64"),
            "congestion_level": pd.array(
                ["Free-flow", "Moderate", "Heavy", "Severe"] * 2,
                dtype="string",
            ),
            "accident_count": pd.array([0, 1, pd.NA, 0, 0, 0, 1, 0], dtype="Int64"),
            "_accident_observed": pd.array(
                [True, True, False, True, True, True, True, True],
                dtype="boolean",
            ),
        }
    )


def test_eda_configuration_covers_required_analysis_contract() -> None:
    config = load_eda_config(load_settings())

    assert config["version"] == "eda_v1"
    assert set(config["descriptive_columns"]) >= {
        "traffic_volume",
        "avg_speed",
        "occupancy",
        "travel_time",
    }
    assert config["congestion_order"] == [
        "Free-flow",
        "Moderate",
        "Heavy",
        "Severe",
    ]


def test_descriptive_statistics_use_real_denominators_and_sample_spread() -> None:
    summary = descriptive_statistics(_frame(), ["traffic_volume", "avg_speed"])

    assert summary["traffic_volume"]["count"] == 8
    assert summary["traffic_volume"]["null_count"] == 0
    assert summary["traffic_volume"]["mean"] == 4.5
    assert summary["traffic_volume"]["median"] == 4.5
    assert summary["traffic_volume"]["standard_deviation"] == round(
        float(np.std(range(1, 9), ddof=1)), 6
    )


def test_target_distributions_exclude_unknown_accident_windows() -> None:
    distributions = target_distributions(
        _frame(), ["Free-flow", "Moderate", "Heavy", "Severe"]
    )

    assert all(
        record["rows"] == 2
        for record in distributions["congestion"].values()
    )
    accident = distributions["accident"]
    assert accident["observed_rows"] == 7
    assert accident["unobserved_rows"] == 1
    assert accident["positive_rows"] == 2
    assert accident["negative_rows"] == 5
    assert accident["positive_rate_observed"] == round(2 / 7, 8)


def test_context_aggregates_cover_every_required_dimension() -> None:
    dimensions = [
        "road_id",
        "local_hour",
        "day_of_week",
        "weather_condition",
        "public_holiday",
        "event_flag",
        "roadwork_flag",
    ]
    aggregates = context_aggregates(
        _frame(), dimensions, ["Free-flow", "Moderate", "Heavy", "Severe"]
    )

    assert set(aggregates["dimension"]) == set(dimensions)
    roads = aggregates[aggregates["dimension"].eq("road_id")]
    assert roads["row_count"].sum() == 8
    assert roads["accident_observed_rows"].sum() == 7
    assert roads["accident_positive_rows"].sum() == 2


def test_correlation_reports_redundancy_and_target_association() -> None:
    frame = pd.DataFrame(
        {
            "x": np.arange(1, 11, dtype=float),
            "x_copy": np.arange(1, 11, dtype=float) * 2,
            "reverse": np.arange(10, 0, -1, dtype=float),
            "target_volume_h1": np.arange(1, 11, dtype=float) * 3,
            "target_volume_h1_available": pd.array([True] * 10, dtype="boolean"),
        }
    )
    result = correlation_analysis(
        frame,
        ["x", "x_copy", "reverse"],
        "target_volume_h1",
        "target_volume_h1_available",
        0.95,
    )

    assert result.correlation.loc["x", "x_copy"] == 1.0
    assert len(result.redundant_pairs) == 3
    assert result.target_correlations[0]["correlation"] in {1.0, -1.0}
    assert all(record["observations"] == 10 for record in result.target_correlations)
