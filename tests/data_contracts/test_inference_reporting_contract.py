"""Full-artifact inference, persistence, reporting, and runtime contracts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from flowcast.inference.artifacts import (
    load_prediction_batch,
    persist_prediction_batch,
)
from flowcast.inference.predictor import Predictor
from flowcast.inference.schemas import PROBABILITY_COLUMNS
from flowcast.reports.export import (
    build_prediction_reports,
    verify_prediction_reports,
)
from flowcast.settings import load_settings


@pytest.fixture(scope="module")
def predictor() -> Predictor:
    return Predictor(load_settings(), device="cpu")


@pytest.fixture(scope="module")
def four_horizon_result(predictor: Predictor):
    request = predictor.build_request(road_ids=["NL-001"], horizons=[1, 2, 3, 4])
    return predictor.predict(request)


@pytest.mark.data_contract
def test_one_request_returns_all_targets_confidence_and_lineage(
    four_horizon_result,
) -> None:
    frame = four_horizon_result.frame

    assert len(frame) == 4
    assert set(frame["horizon_windows"]) == {1, 2, 3, 4}
    assert frame[
        [
            "volume_prediction",
            "speed_prediction",
            "travel_time_prediction",
            "accident_probability",
        ]
    ].notna().all(axis=None)
    assert frame["congestion_prediction"].notna().all()
    assert frame["volume_interval_lower"].le(frame["volume_prediction"]).all()
    assert frame["volume_interval_upper"].ge(frame["volume_prediction"]).all()
    assert frame[list(PROBABILITY_COLUMNS)].sum(axis=1).round(8).eq(1.0).all()
    assert frame["volume_model_version"].eq("recurrent_volume_v1").all()
    assert frame["volume_classical_model_version"].eq(
        "classical_regression_v1"
    ).all()
    assert frame["confidence_version"].eq("confidence_error_v1").all()
    assert frame["processed_data_sha256"].str.len().eq(64).all()


@pytest.mark.data_contract
def test_repeated_cpu_inference_is_stable(
    predictor: Predictor,
    four_horizon_result,
) -> None:
    repeated = predictor.predict(four_horizon_result.request)

    pd.testing.assert_frame_equal(repeated.frame, four_horizon_result.frame)


@pytest.mark.data_contract
def test_invalid_requests_fail_clearly(predictor: Predictor) -> None:
    with pytest.raises(ValueError, match="Unknown road_ids"):
        predictor.build_request(road_ids=["NL-999"], horizons=[1])
    with pytest.raises(ValueError, match="Horizons"):
        request = predictor.build_request(road_ids=["NL-001"], horizons=[1])
        predictor.predict(
            type(request)(
                request.road_ids,
                request.origin_timestamp,
                (5,),
                request.device,
            )
        )
    early = predictor.build_request(
        road_ids=["NL-001"],
        origin_timestamp="2025-01-01T00:00:00+05:30",
        horizons=[1],
    )
    with pytest.raises(ValueError, match="sequence history"):
        predictor.predict(early)


@pytest.mark.data_contract
def test_batch_persistence_report_and_tamper_rejection(
    predictor: Predictor,
    four_horizon_result,
    tmp_path: Path,
) -> None:
    settings = predictor.settings
    paths = persist_prediction_batch(
        four_horizon_result,
        settings,
        output_root=tmp_path,
    )
    loaded = load_prediction_batch(settings, paths.manifest_path)
    pd.testing.assert_frame_equal(loaded.frame, four_horizon_result.frame)

    reports = build_prediction_reports(
        settings,
        paths.manifest_path,
        output_root=tmp_path,
    )
    exported = pd.read_csv(reports.csv_path)
    assert len(exported) == len(four_horizon_result.frame)
    html = reports.html_path.read_text(encoding="utf-8")
    assert "FlowCast Forecast Report" in html
    assert "does not retrain" in html
    assert "NL-001" in html
    verified_report = verify_prediction_reports(settings, reports.manifest_path)
    assert verified_report["request_id"] == loaded.manifest["request_id"]

    original = paths.predictions_path.read_bytes()
    try:
        paths.predictions_path.write_bytes(original + b"tampered")
        with pytest.raises(RuntimeError, match="byte count changed"):
            load_prediction_batch(settings, paths.manifest_path)
    finally:
        paths.predictions_path.write_bytes(original)

    original_html = reports.html_path.read_bytes()
    try:
        reports.html_path.write_bytes(original_html + b"tampered")
        with pytest.raises(RuntimeError, match="byte count changed"):
            verify_prediction_reports(settings, reports.manifest_path)
    finally:
        reports.html_path.write_bytes(original_html)


@pytest.mark.data_contract
def test_full_corridor_one_horizon_cpu_runtime_is_measured(
    predictor: Predictor,
) -> None:
    request = predictor.build_request(horizons=[1])
    result = predictor.predict(request)
    target = float(
        predictor.context.config["output"][
            "full_corridor_runtime_target_seconds"
        ]
    )

    assert len(result.frame) == 25
    assert result.frame["road_id"].nunique() == 25
    assert result.prediction_seconds > 0.0
    assert result.prediction_seconds <= target
