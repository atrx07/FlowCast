"""End-to-end validation and quarantine assertions for delivered sources."""

from __future__ import annotations

import json
from dataclasses import replace

import pandas as pd
import pytest

from flowcast.data.audit import sha256_file
from flowcast.data.quarantine import run_validation_pipeline
from flowcast.settings import load_settings


@pytest.fixture(scope="module")
def validation_run(tmp_path_factory):
    temporary_root = tmp_path_factory.mktemp("raw-validation")
    settings = replace(
        load_settings(),
        raw_dir=temporary_root / "raw",
        interim_dir=temporary_root / "interim",
        quarantine_dir=temporary_root / "quarantine",
    )
    return run_validation_pipeline(settings), settings


@pytest.mark.data_contract
def test_validation_reproduces_expected_row_accounting(validation_run) -> None:
    run, _ = validation_run
    summary = run.summary

    assert summary["dataset_failure"] is False
    assert summary["total_input_rows"] == 189_491
    assert summary["total_valid_rows"] == 187_724
    assert summary["total_rejected_rows"] == 1_767
    assert summary["total_issues"] == 42_792
    assert summary["datasets"]["calendar"]["valid_rows"] == 151
    assert summary["datasets"]["weather"]["valid_rows"] == 10_872
    assert summary["datasets"]["traffic"]["valid_rows"] == 176_701
    assert summary["datasets"]["traffic"]["rejected_rows"] == 1_767


@pytest.mark.data_contract
def test_validation_reason_counts_match_source_anomalies(validation_run) -> None:
    run, _ = validation_run
    traffic = run.summary["datasets"]["traffic"]["issues_by_reason"]
    weather = run.summary["datasets"]["weather"]["issues_by_reason"]

    assert traffic == {
        "duplicate_key": 1_767,
        "excessive_speed": 237,
        "invalid_occupancy": 234,
        "missing_value": 40_035,
        "negative_traffic_volume": 241,
    }
    assert weather == {"missing_value": 278}


@pytest.mark.data_contract
def test_validation_artifacts_are_readable_and_complete(validation_run) -> None:
    run, _ = validation_run

    assert json.loads(run.summary_path.read_text(encoding="utf-8")) == run.summary
    issues = pd.read_parquet(run.issues_path)
    assert len(issues) == 42_792
    assert list(issues.columns) == [
        "dataset",
        "source_file",
        "source_row",
        "field",
        "rejected_value",
        "reason_code",
        "disposition",
        "message",
        "retained_source_row",
    ]
    for dataset, result in run.results.items():
        assert len(pd.read_parquet(run.valid_paths[dataset])) == len(result.valid_rows)
        assert len(pd.read_parquet(run.rejected_paths[dataset])) == len(
            result.rejected_rows
        )


@pytest.mark.data_contract
def test_validation_preserves_verified_raw_copies(validation_run) -> None:
    run, settings = validation_run

    for dataset, result in run.results.items():
        source = run.summary["datasets"][dataset]["source"]
        raw_path = settings.raw_dir / result.source_file
        reference_path = settings.reference_dir / result.source_file
        assert sha256_file(raw_path) == source["sha256"]
        assert sha256_file(reference_path) == source["sha256"]


@pytest.mark.data_contract
def test_repeated_validation_is_artifact_deterministic(validation_run) -> None:
    first, settings = validation_run
    first_summary_hash = sha256_file(first.summary_path)
    first_artifact_hashes = {
        dataset: (
            sha256_file(first.valid_paths[dataset]),
            sha256_file(first.rejected_paths[dataset]),
        )
        for dataset in first.results
    }
    first_issues_hash = sha256_file(first.issues_path)
    repeated = run_validation_pipeline(settings)

    assert repeated.summary == first.summary
    assert sha256_file(repeated.summary_path) == first_summary_hash
    assert sha256_file(repeated.issues_path) == first_issues_hash
    assert {
        dataset: (
            sha256_file(repeated.valid_paths[dataset]),
            sha256_file(repeated.rejected_paths[dataset]),
        )
        for dataset in repeated.results
    } == first_artifact_hashes
