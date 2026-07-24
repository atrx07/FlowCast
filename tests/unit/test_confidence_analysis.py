"""Unit tests for Step 16 calibration, uncertainty, and slice rules."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from flowcast.evaluation.confidence_config import load_confidence_config
from flowcast.evaluation.confidence_metrics import (
    conformal_calibration,
    enrich_classification,
)
from flowcast.evaluation.confidence_pairing import paired_volume_frame
from flowcast.evaluation.confidence_slices import regression_slices
from flowcast.settings import load_settings


def test_confidence_config_is_independent_and_validation_only() -> None:
    config, path = load_confidence_config(load_settings())

    assert path.name == "confidence.yaml"
    assert config["regression"]["calibration_split"] == "validation"
    assert config["regression"]["application_splits"] == ["validation", "test"]
    assert config["regression"]["confidence_level"] == pytest.approx(0.90)
    assert config["accident_risk"]["source"] == (
        "validation_selected_operating_threshold"
    )


def test_finite_sample_conformal_quantile_ignores_test_residuals() -> None:
    common = {
        "model_version": "model_v1",
        "target": "volume",
        "horizon_windows": 1,
        "horizon_minutes": 30,
    }
    frame = pd.DataFrame(
        [
            {**common, "split": "validation", "actual": 10.0, "prediction": 9.0},
            {**common, "split": "validation", "actual": 10.0, "prediction": 8.0},
            {**common, "split": "validation", "actual": 10.0, "prediction": 7.0},
            {**common, "split": "validation", "actual": 10.0, "prediction": 6.0},
            {**common, "split": "test", "actual": 10.0, "prediction": -1000.0},
        ]
    )
    calibration = conformal_calibration(frame, 0.80)

    assert len(calibration) == 1
    assert calibration.loc[0, "calibration_rows"] == 4
    assert calibration.loc[0, "finite_sample_rank"] == 4
    assert calibration.loc[0, "absolute_residual_quantile"] == pytest.approx(4.0)

    changed = frame.copy()
    changed.loc[changed["split"].eq("test"), "prediction"] = 1_000_000.0
    assert conformal_calibration(changed, 0.80).equals(calibration)


def _processed_context(timestamp: pd.Timestamp) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "road_id": ["A"],
            "timestamp": [timestamp],
            "hour_of_day": pd.Series([8.5], dtype="Float64"),
            "day_of_week": pd.Series([0], dtype="Int8"),
            "is_weekend": pd.Series([False], dtype="boolean"),
            "is_morning_peak": pd.Series([True], dtype="boolean"),
            "is_evening_peak": pd.Series([False], dtype="boolean"),
            "weather_condition": pd.Series(["Rain"], dtype="string"),
            "target_congestion_h1": pd.Series(["Heavy"], dtype="string"),
            "target_congestion_h2": pd.Series(["Heavy"], dtype="string"),
            "target_congestion_h3": pd.Series(["Heavy"], dtype="string"),
            "target_congestion_h4": pd.Series(["Heavy"], dtype="string"),
        }
    )


def test_classification_uncertainty_and_risk_bands_use_frozen_threshold() -> None:
    settings = load_settings()
    config, _ = load_confidence_config(settings)
    timestamp = pd.Timestamp("2025-05-01 08:30", tz="Asia/Kolkata")
    base = {
        "road_id": "A",
        "timestamp": timestamp,
        "target_timestamp": timestamp + pd.Timedelta(minutes=30),
        "split": "test",
        "horizon_windows": 1,
        "horizon_minutes": 30,
        "model_version": "classical_classification_v1",
        "selected_family": "svm",
        "candidate_id": "candidate",
        "calibration_applied": True,
        "target_column": "target",
        "job_id": "job",
    }
    rows = [
        {
            **base,
            "task": "congestion",
            "actual_class_index": 1,
            "predicted_class_index": 1,
            "actual_label": "Moderate",
            "predicted_label": "Moderate",
            "probability_free_flow": 0.1,
            "probability_moderate": 0.7,
            "probability_heavy": 0.1,
            "probability_severe": 0.1,
            "probability_no_accident": np.nan,
            "probability_accident": np.nan,
            "operating_threshold": np.nan,
        }
    ]
    for probability in (0.004, 0.007, 0.015, 0.03):
        rows.append(
            {
                **base,
                "task": "accident",
                "actual_class_index": 0,
                "predicted_class_index": int(probability >= 0.01),
                "actual_label": "no_accident",
                "predicted_label": "no_accident",
                "probability_free_flow": np.nan,
                "probability_moderate": np.nan,
                "probability_heavy": np.nan,
                "probability_severe": np.nan,
                "probability_no_accident": 1.0 - probability,
                "probability_accident": probability,
                "operating_threshold": 0.01,
            }
        )
    enriched = enrich_classification(
        pd.DataFrame(rows),
        _processed_context(timestamp),
        config,
    )

    congestion = enriched.loc[enriched["task"].eq("congestion")].iloc[0]
    assert congestion["max_probability"] == pytest.approx(0.7)
    assert 0.0 <= congestion["normalized_entropy"] <= 1.0
    assert congestion["confidence_band"] == "medium"
    accident = enriched.loc[enriched["task"].eq("accident")]
    assert accident["risk_band"].tolist() == [
        "low",
        "elevated",
        "high",
        "critical",
    ]
    assert set(enriched["peak_status"]) == {"morning_peak"}
    assert set(enriched["actual_congestion"]) == {"Heavy"}


def test_unsupported_regression_slice_is_retained_without_metrics() -> None:
    frame = pd.DataFrame(
        {
            "model_version": ["m"] * 3,
            "target": ["volume"] * 3,
            "horizon_windows": [1] * 3,
            "horizon_minutes": [30] * 3,
            "split": ["test"] * 3,
            "road_id": ["A", "A", "B"],
            "actual": [10.0, 11.0, 12.0],
            "prediction": [9.0, 12.0, 12.0],
            "signed_error": [-1.0, 1.0, 0.0],
            "interval_covered": [True, True, True],
            "interval_width": [4.0, 4.0, 4.0],
        }
    )
    slices = regression_slices(frame, ("road_id",), minimum_rows=2)
    unsupported = slices.loc[
        slices["dimension"].eq("road_id") & slices["slice_value"].eq("B")
    ].iloc[0]

    assert not bool(unsupported["sufficient_support"])
    assert unsupported["rows"] == 1
    assert np.isnan(unsupported["rmse"])


def test_paired_volume_requires_identical_actual_values() -> None:
    timestamp = pd.Timestamp("2025-05-01", tz="Asia/Kolkata")
    base = {
        "road_id": "A",
        "timestamp": timestamp,
        "target_timestamp": timestamp + pd.Timedelta(minutes=30),
        "split": "test",
        "target": "volume",
        "horizon_windows": 1,
        "horizon_minutes": 30,
        "prediction": 101.0,
        "absolute_error": 1.0,
        "interval_lower": 90.0,
        "interval_upper": 110.0,
        "interval_covered": True,
        "origin_hour": 0,
        "weekday": "Thursday",
        "weekday_type": "weekday",
        "peak_status": "off_peak",
        "weather_condition": "Clear",
        "actual_congestion": "Moderate",
    }
    frame = pd.DataFrame(
        [
            {**base, "actual": 100.0, "model_version": "classical_regression_v1"},
            {**base, "actual": 100.0, "model_version": "recurrent_volume_v1"},
        ]
    )
    assert len(paired_volume_frame(frame)) == 1
    frame.loc[1, "actual"] = 99.0
    with pytest.raises(RuntimeError, match="disagree"):
        paired_volume_frame(frame)
