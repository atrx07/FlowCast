"""Full-source Step 05 traffic-cleaning artifact contract tests."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from flowcast.data.audit import sha256_file
from flowcast.data.traffic_pipeline import run_traffic_cleaning
from flowcast.settings import load_settings


@pytest.fixture(scope="module")
def cleaned_traffic(tmp_path_factory):
    root = tmp_path_factory.mktemp("cleaned-traffic")
    settings = replace(
        load_settings(),
        raw_dir=root / "raw",
        interim_dir=root / "interim",
        quarantine_dir=root / "quarantine",
        artifacts_dir=root / "artifacts",
    )
    return run_traffic_cleaning(settings), settings


@pytest.mark.data_contract
def test_complete_traffic_grid_contract(cleaned_traffic) -> None:
    run, _ = cleaned_traffic
    traffic = pd.read_parquet(run.traffic_path)

    assert len(traffic) == 181_200
    assert traffic["road_id"].nunique() == 25
    assert not traffic.duplicated(["road_id", "timestamp"]).any()
    assert traffic.groupby("road_id").size().eq(7_248).all()
    assert traffic["_inserted_window"].sum() == 4_499
    assert traffic["_source_row"].isna().sum() == 4_499
    trusted = [
        "traffic_volume",
        "avg_speed",
        "occupancy",
        "travel_time",
        "congestion_level",
    ]
    assert not traffic[trusted].isna().any().any()
    assert traffic["traffic_volume"].ge(0).all()
    assert traffic["avg_speed"].gt(0).all()
    assert traffic["avg_speed"].le(200).all()
    assert traffic["occupancy"].between(0, 100).all()
    assert traffic["travel_time"].gt(0).all()


@pytest.mark.data_contract
def test_vehicle_and_congestion_contract(cleaned_traffic) -> None:
    run, _ = cleaned_traffic
    traffic = run.traffic
    shares = traffic[["share_2w", "share_car", "share_lcv", "share_hcv"]]

    assert shares.ge(0).all().all()
    assert shares.le(1).all().all()
    assert np.allclose(shares.sum(axis=1), 1.0, atol=1e-12)
    assert set(traffic["congestion_level"]) == {
        "Free-flow",
        "Moderate",
        "Heavy",
        "Severe",
    }
    assert run.summary["dataset"]["congestion"] == {
        "source_labels_preserved": 150_077,
        "derived_labels": 31_123,
        "source_disagreements": 0,
        "class_counts": {
            "Free-flow": 111_307,
            "Heavy": 16_721,
            "Moderate": 43_168,
            "Severe": 10_004,
        },
    }


@pytest.mark.data_contract
def test_recovery_lineage_and_unknown_accident_targets(cleaned_traffic) -> None:
    run, _ = cleaned_traffic
    traffic = run.traffic
    fields = [
        "traffic_volume",
        "vehicle_count",
        "avg_speed",
        "occupancy",
        "travel_time",
        "signal_timing",
        "vehicle_type_dist",
    ]
    for field in fields:
        imputed = traffic[f"{field}_imputation_method"].ne("observed")
        assert traffic.loc[
            imputed, f"{field}_imputation_donor_source_rows"
        ].notna().all()
        assert traffic.loc[
            imputed, f"{field}_imputation_donor_timestamp"
        ].notna().all()

    for field in ["avg_speed", "occupancy"]:
        leading = traffic[f"{field}_imputation_method"].eq(
            "same_timestamp_station_median"
        )
        assert leading.sum() == 3
        for row in traffic.loc[
            leading, ["_source_row", f"{field}_imputation_donor_source_rows"]
        ].itertuples(index=False, name=None):
            source_row, donor_rows = row
            assert int(source_row) not in json.loads(donor_rows)

    assert traffic["traffic_volume_original_missing"].sum() == 4_348
    assert traffic["traffic_volume_physical_invalid"].sum() == 239
    assert traffic["avg_speed_physical_invalid"].sum() == 236
    assert traffic["occupancy_physical_invalid"].sum() == 229
    assert traffic["accident_count"].isna().sum() == 4_499
    assert (~traffic["_accident_observed"]).sum() == 4_499


@pytest.mark.data_contract
def test_traffic_quality_summary_and_artifacts_are_deterministic(
    cleaned_traffic,
) -> None:
    first, settings = cleaned_traffic
    persisted = json.loads(first.summary_path.read_text(encoding="utf-8"))
    assert persisted == first.summary
    assert persisted["contract_version"] == "traffic_cleaning_v1"
    assert persisted["dataset"]["duplicate_rows_accounted"] == 1_767
    assert persisted["dataset"]["grid"]["inserted_windows"] == 4_499
    assert persisted["configuration"]["cleaning"]["sha256"] == sha256_file(
        settings.cleaning_config_path
    )
    assert b"\r\n" not in first.summary_path.read_bytes()
    assert b"\r\n" not in first.markdown_path.read_bytes()

    first_hashes = {
        "traffic": sha256_file(first.traffic_path),
        "summary": sha256_file(first.summary_path),
        "markdown": sha256_file(first.markdown_path),
    }
    repeated = run_traffic_cleaning(settings)
    assert repeated.summary == first.summary
    assert {
        "traffic": sha256_file(repeated.traffic_path),
        "summary": sha256_file(repeated.summary_path),
        "markdown": sha256_file(repeated.markdown_path),
    } == first_hashes
