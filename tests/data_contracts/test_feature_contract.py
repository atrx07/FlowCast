"""Full-source Step 07 feature artifact and leakage contracts."""

from __future__ import annotations

import json
import shutil
from dataclasses import replace

import pandas as pd
import pytest

from flowcast.data.audit import sha256_file
from flowcast.features.inputs import load_verified_merged
from flowcast.features.pipeline import run_feature_engineering
from flowcast.settings import load_settings


@pytest.fixture(scope="module")
def feature_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("feature-contract")
    base = load_settings()
    interim = root / "interim"
    artifacts = root / "artifacts"
    merge_dir = interim / base.merge_version
    merge_quality = artifacts / "quality" / base.merge_version
    merge_dir.mkdir(parents=True)
    merge_quality.mkdir(parents=True)
    shutil.copy2(
        base.interim_dir / base.merge_version / "merged.parquet",
        merge_dir / "merged.parquet",
    )
    shutil.copy2(
        base.artifacts_dir / "quality" / base.merge_version / "summary.json",
        merge_quality / "summary.json",
    )
    settings = replace(base, interim_dir=interim, artifacts_dir=artifacts)
    return run_feature_engineering(settings), settings


@pytest.mark.data_contract
def test_feature_table_cardinality_ranges_and_history(feature_run) -> None:
    run, _ = feature_run
    frame = pd.read_parquet(run.feature_path)

    assert len(frame) == 181_200
    assert not frame.duplicated(["road_id", "timestamp"]).any()
    assert frame["road_id"].nunique() == 25
    assert int((~frame["history_available"]).sum()) == 1_200
    assert set(
        (~frame["history_available"]).groupby(frame["road_id"]).sum().unique()
    ) == {48}
    assert frame["half_hour_capacity"].gt(0).all()
    assert frame["volume_capacity_ratio"].ge(0).all()
    assert frame["temperature_band"].isin(["cool", "mild", "warm"]).all()
    assert frame[["share_2w", "share_car", "share_lcv", "share_hcv"]].sum(
        axis=1
    ).between(1 - 1e-9, 1 + 1e-9).all()


@pytest.mark.data_contract
def test_feature_manifest_is_complete_and_traceable(feature_run) -> None:
    run, _ = feature_run
    persisted = json.loads(run.manifest_path.read_text(encoding="utf-8"))

    assert persisted == run.manifest
    assert persisted["contract_version"] == "explanatory_features_v1"
    assert persisted["forecast_horizons_reserved"] == [1, 2, 3, 4]
    assert persisted["feature_count"] == len(persisted["features"]) == 62
    names = [record["name"] for record in persisted["features"]]
    assert len(names) == len(set(names))
    assert all(
        set(record) >= {
            "name",
            "dtype",
            "group",
            "source_columns",
            "transform",
            "version",
            "leakage_status",
        }
        for record in persisted["features"]
    )
    assert all(
        record["leakage_status"] == "known_at_origin"
        for record in persisted["features"]
    )


@pytest.mark.data_contract
def test_feature_summary_and_artifacts_are_deterministic(feature_run) -> None:
    first, settings = feature_run
    persisted = json.loads(first.summary_path.read_text(encoding="utf-8"))

    assert persisted == first.summary
    assert persisted["dataset"]["output_rows"] == 181_200
    assert persisted["dataset"]["output_unique_keys"] == 181_200
    assert persisted["dataset"]["row_count_change"] == 0
    assert persisted["dataset"]["duplicate_output_keys"] == 0
    assert persisted["dataset"]["feature_null_counts"]["volume_lag_48"] == 1_200
    assert b"\r\n" not in first.manifest_path.read_bytes()
    assert b"\r\n" not in first.summary_path.read_bytes()
    assert b"\r\n" not in first.markdown_path.read_bytes()

    first_hashes = {
        "features": sha256_file(first.feature_path),
        "manifest": sha256_file(first.manifest_path),
        "summary": sha256_file(first.summary_path),
        "markdown": sha256_file(first.markdown_path),
    }
    repeated = run_feature_engineering(settings)
    assert repeated.manifest == first.manifest
    assert repeated.summary == first.summary
    assert {
        "features": sha256_file(repeated.feature_path),
        "manifest": sha256_file(repeated.manifest_path),
        "summary": sha256_file(repeated.summary_path),
        "markdown": sha256_file(repeated.markdown_path),
    } == first_hashes


@pytest.mark.data_contract
def test_merged_input_hash_is_checked_before_read(feature_run) -> None:
    _, settings = feature_run
    path = settings.interim_dir / settings.merge_version / "merged.parquet"
    original = path.read_bytes()
    try:
        path.write_bytes(original + b"tampered")
        with pytest.raises(RuntimeError, match="byte count changed"):
            load_verified_merged(settings)
    finally:
        path.write_bytes(original)
