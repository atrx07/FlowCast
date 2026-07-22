"""Full-source Step 06 merge artifact contract tests."""

from __future__ import annotations

import json
from dataclasses import replace

import pandas as pd
import pytest

from flowcast.data.audit import sha256_file
from flowcast.data.merge_pipeline import run_source_merge
from flowcast.settings import load_settings


@pytest.fixture(scope="module")
def merged_sources(tmp_path_factory):
    root = tmp_path_factory.mktemp("merged-sources")
    settings = replace(
        load_settings(),
        raw_dir=root / "raw",
        interim_dir=root / "interim",
        quarantine_dir=root / "quarantine",
        artifacts_dir=root / "artifacts",
    )
    return run_source_merge(settings), settings


@pytest.mark.data_contract
def test_merged_source_cardinality_and_coverage(merged_sources) -> None:
    run, _ = merged_sources
    merged = pd.read_parquet(run.merged_path)

    assert len(merged) == 181_200
    assert not merged.duplicated(["road_id", "timestamp"]).any()
    assert merged["weather_join_status"].eq("both").all()
    assert merged["calendar_join_status"].eq("both").all()
    context = [
        "weather_condition",
        "temperature",
        "rainfall",
        "visibility",
        "public_holiday",
        "event_flag",
        "roadwork_flag",
    ]
    assert not merged[context].isna().any().any()
    assert merged["weather_hour"].eq(merged["timestamp"].dt.floor("h")).all()
    assert merged["calendar_date"].eq(
        merged["timestamp"].dt.tz_localize(None).dt.normalize()
    ).all()


@pytest.mark.data_contract
def test_merge_preserves_traffic_lineage(merged_sources) -> None:
    run, settings = merged_sources
    traffic = pd.read_parquet(
        settings.interim_dir / settings.cleaning_version / "traffic.parquet"
    )
    lineage = [
        "road_id",
        "timestamp",
        "_source_row",
        "_inserted_window",
        "_accident_observed",
        "traffic_volume",
        "traffic_volume_imputation_method",
    ]
    pd.testing.assert_frame_equal(run.merged[lineage], traffic[lineage])
    assert run.merged["weather_source_row"].notna().all()
    assert run.merged["calendar_source_row"].notna().all()


@pytest.mark.data_contract
def test_merge_summary_and_artifacts_are_deterministic(merged_sources) -> None:
    first, settings = merged_sources
    persisted = json.loads(first.summary_path.read_text(encoding="utf-8"))

    assert persisted == first.summary
    assert persisted["contract_version"] == "source_merge_v1"
    assert persisted["dataset"]["output_rows"] == 181_200
    assert persisted["dataset"]["row_count_change"] == 0
    assert persisted["dataset"]["joins"]["weather"]["missing"] == 0
    assert persisted["dataset"]["joins"]["calendar"]["missing"] == 0
    assert persisted["configuration"]["base"]["sha256"] == sha256_file(
        settings.config_path
    )
    assert persisted["configuration"]["cleaning"]["sha256"] == sha256_file(
        settings.cleaning_config_path
    )
    assert b"\r\n" not in first.summary_path.read_bytes()
    assert b"\r\n" not in first.markdown_path.read_bytes()

    first_hashes = {
        "merged": sha256_file(first.merged_path),
        "summary": sha256_file(first.summary_path),
        "markdown": sha256_file(first.markdown_path),
    }
    repeated = run_source_merge(settings)
    assert repeated.summary == first.summary
    assert {
        "merged": sha256_file(repeated.merged_path),
        "summary": sha256_file(repeated.summary_path),
        "markdown": sha256_file(repeated.markdown_path),
    } == first_hashes
