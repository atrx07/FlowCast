"""Full-source Step 08 processed-data and target-alignment contracts."""

from __future__ import annotations

import json
import shutil
from dataclasses import replace

import pandas as pd
import pytest

from flowcast.data.audit import sha256_file
from flowcast.features.inputs import load_verified_features
from flowcast.features.processed_pipeline import run_processed_data
from flowcast.settings import load_settings


@pytest.fixture(scope="module")
def processed_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("processed-contract")
    base = load_settings()
    interim = root / "interim"
    processed = root / "processed"
    artifacts = root / "artifacts"
    feature_data = interim / base.feature_version
    feature_quality = artifacts / "quality" / base.feature_version
    feature_manifest = artifacts / "features" / base.feature_version
    feature_data.mkdir(parents=True)
    feature_quality.mkdir(parents=True)
    feature_manifest.mkdir(parents=True)
    shutil.copy2(
        base.interim_dir / base.feature_version / "features.parquet",
        feature_data / "features.parquet",
    )
    shutil.copy2(
        base.artifacts_dir / "quality" / base.feature_version / "summary.json",
        feature_quality / "summary.json",
    )
    shutil.copy2(
        base.artifacts_dir / "features" / base.feature_version / "manifest.json",
        feature_manifest / "manifest.json",
    )
    settings = replace(
        base,
        interim_dir=interim,
        processed_dir=processed,
        artifacts_dir=artifacts,
    )
    return run_processed_data(settings), settings


@pytest.mark.data_contract
def test_processed_table_preserves_origins_features_and_cardinality(
    processed_run,
) -> None:
    run, settings = processed_run
    source = pd.read_parquet(
        settings.interim_dir / settings.feature_version / "features.parquet"
    )
    frame = pd.read_parquet(run.dataset_path)

    assert len(frame) == 181_200
    assert frame["road_id"].nunique() == 25
    assert not frame.duplicated(["road_id", "timestamp"]).any()
    pd.testing.assert_frame_equal(frame[list(source.columns)], source)


@pytest.mark.data_contract
def test_all_horizons_are_exact_same_road_future_shifts(processed_run) -> None:
    run, _ = processed_run
    frame = run.frame
    grouped = frame.groupby("road_id", sort=False)

    for horizon in range(1, 5):
        expected_timestamp = grouped["timestamp"].shift(-horizon)
        pd.testing.assert_series_equal(
            frame[f"target_timestamp_h{horizon}"],
            expected_timestamp,
            check_names=False,
        )
        assert int(expected_timestamp.isna().sum()) == 25 * horizon
        for name, source in {
            "volume": "traffic_volume",
            "speed": "avg_speed",
            "travel_time": "travel_time",
            "congestion": "congestion_level",
        }.items():
            expected = grouped[source].shift(-horizon)
            pd.testing.assert_series_equal(
                frame[f"target_{name}_h{horizon}"],
                expected,
                check_names=False,
            )
            available = frame[f"target_{name}_h{horizon}_available"]
            assert available.equals(expected.notna().astype("boolean"))


@pytest.mark.data_contract
def test_accident_targets_require_future_observation(processed_run) -> None:
    run, _ = processed_run
    frame = run.frame
    grouped = frame.groupby("road_id", sort=False)

    for horizon in range(1, 5):
        future_count = grouped["accident_count"].shift(-horizon)
        future_observed = grouped["_accident_observed"].shift(-horizon)
        expected_available = (
            future_observed.fillna(False).astype(bool) & future_count.notna()
        ).astype("boolean")
        available = frame[f"target_accident_h{horizon}_available"]
        expected_target = future_count.gt(0).astype("boolean").where(
            expected_available, pd.NA
        )
        assert available.equals(expected_available)
        pd.testing.assert_series_equal(
            frame[f"target_accident_h{horizon}"],
            expected_target,
            check_names=False,
        )
        assert frame.loc[~available, f"target_accident_h{horizon}"].isna().all()


@pytest.mark.data_contract
def test_processed_manifest_and_coverage_are_complete(processed_run) -> None:
    run, _ = processed_run
    persisted = json.loads(run.manifest_path.read_text(encoding="utf-8"))
    summary = json.loads(run.summary_path.read_text(encoding="utf-8"))

    assert persisted == run.manifest
    assert persisted["contract_version"] == "multi_horizon_targets_v1"
    assert persisted["forecast_horizons"] == [1, 2, 3, 4]
    assert persisted["target_count"] == len(persisted["targets"]) == 20
    assert len({record["name"] for record in persisted["targets"]}) == 20
    assert all(
        set(record) >= {
            "name",
            "source_column",
            "task",
            "horizon_windows",
            "horizon_minutes",
            "target_timestamp_column",
            "availability_column",
            "dtype",
            "transform",
            "version",
        }
        for record in persisted["targets"]
    )
    assert summary == run.summary
    assert summary["dataset"]["output_rows"] == 181_200
    assert summary["dataset"]["row_count_change"] == 0
    for horizon, record in summary["dataset"]["timestamp_coverage"].items():
        assert record["unavailable_rows"] == 25 * int(horizon)
        assert record["unavailable_rows"] == (
            record["expected_trailing_unavailable_rows"]
        )


@pytest.mark.data_contract
def test_processed_artifacts_are_deterministic(processed_run) -> None:
    first, settings = processed_run
    assert b"\r\n" not in first.manifest_path.read_bytes()
    assert b"\r\n" not in first.summary_path.read_bytes()
    assert b"\r\n" not in first.markdown_path.read_bytes()
    first_hashes = {
        "dataset": sha256_file(first.dataset_path),
        "manifest": sha256_file(first.manifest_path),
        "summary": sha256_file(first.summary_path),
        "markdown": sha256_file(first.markdown_path),
    }

    repeated = run_processed_data(settings)
    assert repeated.manifest == first.manifest
    assert repeated.summary == first.summary
    assert {
        "dataset": sha256_file(repeated.dataset_path),
        "manifest": sha256_file(repeated.manifest_path),
        "summary": sha256_file(repeated.summary_path),
        "markdown": sha256_file(repeated.markdown_path),
    } == first_hashes


@pytest.mark.data_contract
def test_feature_input_hash_is_checked_before_read(processed_run) -> None:
    _, settings = processed_run
    path = settings.interim_dir / settings.feature_version / "features.parquet"
    original = path.read_bytes()
    try:
        path.write_bytes(original + b"tampered")
        with pytest.raises(RuntimeError, match="byte count changed"):
            load_verified_features(settings)
    finally:
        path.write_bytes(original)
