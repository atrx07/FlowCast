"""Full-artifact contracts for Step 16 confidence and error analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from flowcast.evaluation.confidence_artifacts import (
    confidence_paths,
    load_verified_confidence_artifacts,
)
from flowcast.evaluation.confidence_pipeline import run_confidence_analysis
from flowcast.settings import load_settings


@pytest.fixture(scope="module")
def confidence_artifacts():
    settings = load_settings()
    paths = confidence_paths(settings, "confidence_error_v1")
    if not paths.summary_path.is_file():
        pytest.fail("Canonical confidence artifacts are missing; run the Step 16 CLI")
    loaded = load_verified_confidence_artifacts(settings)
    return settings, paths, loaded


@pytest.mark.data_contract
def test_confidence_artifact_coverage_and_lineage(confidence_artifacts) -> None:
    _, paths, loaded = confidence_artifacts
    summary = loaded.summary
    calibration = pd.read_csv(paths.interval_calibration_path)

    assert summary["coverage"]["regression_prediction_rows"] == 862_700
    assert summary["coverage"]["classification_prediction_rows"] == 428_257
    assert summary["coverage"]["paired_volume_rows"] == 212_000
    assert summary["coverage"]["conformal_group_count"] == 16
    assert set(calibration["calibration_split"]) == {"validation"}
    assert calibration["confidence_level"].eq(0.9).all()
    assert (calibration["finite_sample_rank"] <= calibration["calibration_rows"]).all()
    assert all(summary["checks"].values())
    assert {
        "processed_summary",
        "processed_manifest",
        "processed_dataset",
        "classical_registry_summary",
        "classical_regression_summary",
        "classification_summary",
        "recurrent_summary",
    } == set(summary["upstream"])


@pytest.mark.data_contract
def test_intervals_probabilities_and_risk_bands_are_valid(
    confidence_artifacts,
) -> None:
    _, paths, loaded = confidence_artifacts
    regression = loaded.regression
    classification = loaded.classification
    coverage = pd.read_csv(paths.regression_coverage_path)

    assert set(regression["split"]) == {"validation", "test"}
    assert regression["interval_lower"].ge(0.0).all()
    assert regression["interval_upper"].ge(regression["interval_lower"]).all()
    test_coverage = coverage.loc[coverage["split"].eq("test")]
    assert test_coverage["interval_coverage"].between(0.88, 0.92).all()
    assert np.isfinite(regression["absolute_residual_quantile"]).all()

    assert classification["max_probability"].between(0.0, 1.0).all()
    assert classification["normalized_entropy"].between(0.0, 1.0).all()
    assert set(classification["confidence_band"]) <= {"low", "medium", "high"}
    accident = classification.loc[classification["task"].eq("accident")]
    assert set(accident["risk_band"]) <= {"low", "elevated", "high", "critical"}
    assert accident["operating_threshold"].notna().all()
    assert classification["actual_congestion"].notna().all()


@pytest.mark.data_contract
def test_slices_reconcile_and_unsupported_metrics_are_visible(
    confidence_artifacts,
) -> None:
    _, paths, loaded = confidence_artifacts
    slices = pd.read_csv(paths.error_slices_path)
    dimensions = {
        "road_id",
        "origin_hour",
        "weekday",
        "weekday_type",
        "peak_status",
        "weather_condition",
        "actual_congestion",
    }
    keys = [
        "task_type",
        "model_version",
        "target",
        "task",
        "horizon_windows",
        "split",
    ]
    for _, group in slices.groupby(keys, sort=True, dropna=False):
        overall = group.loc[group["dimension"].eq("overall")]
        assert len(overall) == 1
        expected = int(overall["rows"].iloc[0])
        for dimension in dimensions:
            assert group.loc[group["dimension"].eq(dimension), "rows"].sum() == expected
    unsupported = slices.loc[~slices["sufficient_support"]]
    assert not unsupported.empty
    metric_columns = ["rmse", "macro_f1", "roc_auc"]
    assert unsupported[metric_columns].isna().all(axis=None)
    assert len(loaded.paired_volume) == len(
        loaded.regression.loc[
            loaded.regression["model_version"].eq("recurrent_volume_v1")
        ]
    )


@pytest.mark.data_contract
def test_confidence_outputs_are_deterministic(confidence_artifacts) -> None:
    settings, paths, _ = confidence_artifacts
    tracked = (
        paths.summary_path,
        paths.report_path,
        paths.interval_calibration_path,
        paths.regression_coverage_path,
        paths.reliability_path,
        paths.risk_bands_path,
        paths.error_slices_path,
        paths.confusions_path,
        paths.paired_slices_path,
    )
    before = {path.name: path.read_bytes() for path in tracked}
    run_confidence_analysis(settings)
    after = {path.name: path.read_bytes() for path in tracked}

    assert after == before


@pytest.mark.data_contract
def test_confidence_loader_rejects_tampering(confidence_artifacts) -> None:
    settings, paths, _ = confidence_artifacts
    original = paths.regression_coverage_path.read_bytes()
    try:
        paths.regression_coverage_path.write_bytes(original + b"tampered")
        with pytest.raises(RuntimeError, match="byte count changed"):
            load_verified_confidence_artifacts(settings)
    finally:
        paths.regression_coverage_path.write_bytes(original)
