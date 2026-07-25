"""Unit coverage for dashboard transformations and safety boundaries."""

from __future__ import annotations

import subprocess

import pandas as pd
import pytest

from flowcast.dashboard.analytics import (
    congestion_matrix,
    corridor_snapshot,
    feature_importance,
    road_summary,
)
from flowcast.dashboard.training_service import TrainingService
from flowcast.dashboard.uploads import stage_upload, validate_upload
from flowcast.settings import load_settings


def test_dashboard_analytics_use_real_values() -> None:
    timestamps = pd.date_range(
        "2025-05-01",
        periods=4,
        freq="30min",
        tz="Asia/Kolkata",
    )
    history = pd.DataFrame(
        {
            "road_id": ["NL-001"] * 4,
            "road_name": ["Northline"] * 4,
            "timestamp": timestamps,
            "traffic_volume": [100, 120, 140, 160],
            "avg_speed": [50.0, 48.0, 45.0, 42.0],
            "travel_time": [2.0, 2.1, 2.2, 2.4],
            "congestion_level": [
                "Free-flow",
                "Moderate",
                "Heavy",
                "Severe",
            ],
            "accident_count": [0, 0, 1, 0],
        }
    )
    matrix = congestion_matrix(history)
    assert matrix.loc["NL-001"].tolist() == [0, 1, 2, 3]
    summary = road_summary(history).iloc[0]
    assert summary["mean_volume"] == pytest.approx(130.0)
    assert summary["severe_share"] == pytest.approx(0.25)
    assert summary["accident_windows"] == 1

    predictions = pd.DataFrame(
        {
            "road_id": ["NL-001", "NL-002"],
            "congestion_prediction": ["Heavy", "Free-flow"],
            "speed_prediction": [30.0, 50.0],
            "accident_probability": [0.03, 0.01],
        }
    )
    snapshot = corridor_snapshot(predictions)
    assert snapshot == {
        "roads": 2,
        "high_congestion": 1,
        "mean_speed": 40.0,
        "max_risk": 0.03,
    }


def test_feature_importance_selects_task_and_horizon() -> None:
    regression = pd.DataFrame(
        {
            "target": ["volume", "volume"],
            "horizon_windows": [1, 2],
            "rank": [1, 1],
            "feature": ["traffic_volume", "volume_lag_1"],
            "importance": [0.8, 0.7],
        }
    )
    classification = pd.DataFrame(
        {
            "task": ["congestion"],
            "horizon_windows": [1],
            "rank": [1],
            "feature": ["volume_capacity_ratio"],
            "importance": [0.6],
        }
    )
    selected = feature_importance(
        regression,
        classification,
        "congestion",
        1,
    )
    assert selected["feature"].tolist() == ["volume_capacity_ratio"]


def test_upload_validation_and_isolated_staging(tmp_path) -> None:
    settings = load_settings()
    source = pd.read_csv(
        settings.reference_dir / "calendar_events.csv",
        nrows=3,
    )
    payload = source.to_csv(index=False).encode("utf-8")
    validation = validate_upload(payload, "calendar.csv", settings)
    assert validation.dataset == "calendar"
    assert validation.summary["row_accounting_valid"]
    assert validation.accepted_for_staging
    manifest = stage_upload(validation, settings, output_root=tmp_path)
    assert manifest.is_file()
    assert tmp_path in manifest.parents
    assert settings.raw_dir not in manifest.parents


def test_upload_rejects_unknown_schema() -> None:
    settings = load_settings()
    with pytest.raises(ValueError, match="exactly match"):
        validate_upload(b"a,b\n1,2\n", "unknown.csv", settings)


def test_training_service_requires_confirmation_and_never_switches(
    tmp_path,
) -> None:
    settings = load_settings()

    def runner(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=0,
            stdout="training complete",
            stderr="",
        )

    service = TrainingService(settings, output_root=tmp_path, runner=runner)
    with pytest.raises(ValueError, match="RETRAIN"):
        service.run("Classical regression", "no")
    result = service.run("Classical regression", "RETRAIN")
    assert result.return_code == 0
    payload = result.manifest_path.read_text(encoding="utf-8")
    assert '"active_model_switched": false' in payload
    assert not service.lock_path.exists()
