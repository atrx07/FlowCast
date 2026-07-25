"""Unit tests for Step 17 request, confidence, and insight semantics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from flowcast.inference.confidence import (
    accident_risk_band,
    probability_confidence,
    regression_interval,
)
from flowcast.inference.config import load_inference_config
from flowcast.inference.feature_prep import normalize_origin
from flowcast.inference.schemas import PredictionRequest
from flowcast.reports.insights import prediction_insights
from flowcast.settings import load_settings


def test_inference_config_freezes_versions_and_cpu_default() -> None:
    config, path = load_inference_config(load_settings())

    assert path.name == "inference.yaml"
    assert config["contract_version"] == "inference_reporting_v1"
    assert config["request"]["horizons"] == [1, 2, 3, 4]
    assert config["device"]["default"] == "cpu"
    assert config["active_routing"]["volume"]["source"] == "recurrent"
    assert config["active_routing"]["volume"]["fallback"][
        "expose_comparator"
    ] is True


def test_prediction_request_is_canonical_and_rejects_duplicates() -> None:
    request = PredictionRequest.from_values(
        ["NL-002", "NL-001"],
        "2025-05-31T23:30:00+05:30",
        [4, 1],
    )

    assert request.road_ids == ("NL-001", "NL-002")
    assert request.horizons == (1, 4)
    assert len(request.identifier("inference_reporting_v1")) == 16
    assert request.identifier("inference_reporting_v1") == request.identifier(
        "inference_reporting_v1"
    )
    with pytest.raises(ValueError, match="unique"):
        PredictionRequest.from_values(
            ["NL-001", "NL-001"],
            request.origin_timestamp,
            [1],
        )


def test_origin_and_confidence_semantics_match_frozen_contract() -> None:
    settings = load_settings()
    origin = normalize_origin("2025-05-31 23:30", settings.timezone)
    assert origin.isoformat() == "2025-05-31T23:30:00+05:30"
    with pytest.raises(ValueError, match="30-minute"):
        normalize_origin("2025-05-31 23:15", settings.timezone)

    assert regression_interval(2.0, 5.0) == (0.0, 7.0)
    confidence_config = {
        "classification": {
            "confidence_bands": {
                "medium_minimum": 0.55,
                "high_minimum": 0.80,
            }
        }
    }
    maximum, entropy, normalized, band = probability_confidence(
        [0.1, 0.9],
        confidence_config,
    )
    assert maximum == pytest.approx(0.9)
    assert 0.0 < entropy < 1.0
    assert 0.0 < normalized < 1.0
    assert band == "high"
    risk_config = {
        "accident_risk": {
            "threshold_multipliers": {
                "elevated": 0.5,
                "high": 1.0,
                "critical": 2.0,
            }
        }
    }
    assert accident_risk_band(0.004, 0.01, risk_config) == "low"
    assert accident_risk_band(0.007, 0.01, risk_config) == "elevated"
    assert accident_risk_band(0.015, 0.01, risk_config) == "high"
    assert accident_risk_band(0.025, 0.01, risk_config) == "critical"


def test_insights_use_only_supplied_prediction_rows() -> None:
    frame = pd.DataFrame(
        {
            "road_id": ["NL-001", "NL-002", "NL-001", "NL-002"],
            "road_name": ["A", "B", "A", "B"],
            "origin_timestamp": pd.to_datetime(
                ["2025-05-31T23:30:00+05:30"] * 4
            ),
            "horizon_windows": [1, 1, 2, 2],
            "horizon_minutes": [30, 30, 60, 60],
            "volume_prediction": [100.0, 200.0, 110.0, 210.0],
            "speed_prediction": [50.0, 40.0, 48.0, 38.0],
            "travel_time_prediction": [2.0, 3.0, 2.1, 3.1],
            "accident_probability": [0.01, 0.03, 0.02, 0.04],
            "accident_risk_band": ["low", "high", "elevated", "critical"],
            "congestion_prediction": [
                "Free-flow",
                "Moderate",
                "Free-flow",
                "Heavy",
            ],
        }
    )

    insights = prediction_insights(frame)

    assert insights["forecast_row_count"] == 4
    assert insights["road_count"] == 2
    assert insights["mean_predicted_volume"] == pytest.approx(155.0)
    assert [
        item["road_id"] for item in insights["highest_accident_risk_by_horizon"]
    ] == ["NL-002", "NL-002"]
    assert np.isfinite(insights["mean_predicted_speed"])
