"""Full delivered-source schema, lineage, and baseline audit assertions."""

import json
from dataclasses import replace

import pandas as pd
import pytest
import yaml

from flowcast.data.audit import run_raw_audit, sha256_file
from flowcast.settings import load_settings


@pytest.fixture(scope="module")
def audit_result(tmp_path_factory):
    temporary_root = tmp_path_factory.mktemp("raw-audit")
    settings = replace(
        load_settings(),
        raw_dir=temporary_root / "raw",
        artifacts_dir=temporary_root / "artifacts",
    )
    return run_raw_audit(settings), settings


@pytest.mark.data_contract
def test_raw_copy_hashes_match_contract_and_sources(audit_result) -> None:
    _, settings = audit_result
    with settings.data_contracts_path.open("r", encoding="utf-8") as handle:
        contracts = yaml.safe_load(handle)["datasets"]
    for contract in contracts.values():
        filename = contract["file"]
        expected = contract["sha256"]
        assert sha256_file(settings.reference_dir / filename) == expected
        assert sha256_file(settings.raw_dir / filename) == expected


@pytest.mark.data_contract
def test_required_source_columns(audit_result) -> None:
    _, settings = audit_result
    with settings.data_contracts_path.open("r", encoding="utf-8") as handle:
        contracts = yaml.safe_load(handle)["datasets"]
    for contract in contracts.values():
        observed = list(pd.read_csv(settings.raw_dir / contract["file"], nrows=0).columns)
        assert observed == contract["required_columns"]


@pytest.mark.data_contract
def test_audit_reproduces_known_baseline(audit_result) -> None:
    result, _ = audit_result
    datasets = result.payload["datasets"]
    traffic = datasets["traffic"]
    weather = datasets["weather"]
    calendar = datasets["calendar"]

    assert traffic["shape"] == {"rows": 178_468, "columns": 17}
    assert traffic["exact_duplicate_count"] == 1_767
    assert traffic["key_duplicate_count"] == 1_767
    assert traffic["unique_key_count"] == 176_701
    assert traffic["expected_grid_size"] == 181_200
    assert traffic["missing_window_count"] == 4_499
    assert traffic["null_counts"]["traffic_volume"] == 4_387
    assert traffic["null_counts"]["avg_speed"] == 4_382
    assert traffic["null_counts"]["occupancy"] == 4_383
    assert traffic["blank_congestion_label_count"] == 26_883
    assert traffic["physical_invalid_counts"]["negative_traffic_volume"] == 241
    assert traffic["accident_positive_count"] == 1_669

    assert weather["shape"] == {"rows": 10_872, "columns": 7}
    assert weather["null_counts"]["temperature"] == 167
    assert weather["null_counts"]["visibility"] == 111
    assert weather["missing_window_count"] == 0
    assert calendar["shape"] == {"rows": 151, "columns": 6}


@pytest.mark.data_contract
def test_persisted_json_matches_audit_result(audit_result) -> None:
    result, _ = audit_result
    persisted = json.loads(result.json_path.read_text(encoding="utf-8"))
    assert persisted == result.payload
    assert result.markdown_path.is_file()
    assert b"\r\n" not in result.json_path.read_bytes()
    assert b"\r\n" not in result.markdown_path.read_bytes()
    assert b"\r\n" not in result.manifest_path.read_bytes()


@pytest.mark.data_contract
def test_repeated_audit_preserves_original_copy_timestamps(audit_result) -> None:
    result, settings = audit_result
    before = {
        item["filename"]: item["copied_at_utc"]
        for item in result.payload["raw_manifest"]["files"]
    }
    repeated = run_raw_audit(settings)
    after = {
        item["filename"]: item["copied_at_utc"]
        for item in repeated.payload["raw_manifest"]["files"]
    }
    assert after == before
