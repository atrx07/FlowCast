"""Full-source Step 09 EDA, quality, and artifact contracts."""

from __future__ import annotations

import json
import shutil
from dataclasses import replace

import pytest
from PIL import Image

from flowcast.analysis.pipeline import run_eda
from flowcast.data.audit import sha256_file
from flowcast.features.inputs import load_verified_processed
from flowcast.settings import load_settings


def _copy(source, destination) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


@pytest.fixture(scope="module")
def eda_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("eda-contract")
    base = load_settings()
    artifacts = root / "artifacts"
    quarantine = root / "quarantine"
    processed = root / "processed"
    files = [
        ("audits/raw_v1/audit.json", "audits/raw_v1/audit.json"),
        (
            "quality/cleaned_sources_v1/summary.json",
            "quality/cleaned_sources_v1/summary.json",
        ),
        (
            "quality/cleaned_sources_v1/traffic_summary.json",
            "quality/cleaned_sources_v1/traffic_summary.json",
        ),
        (
            "quality/merged_sources_v1/summary.json",
            "quality/merged_sources_v1/summary.json",
        ),
        (
            "quality/engineered_features_v1/summary.json",
            "quality/engineered_features_v1/summary.json",
        ),
        (
            "quality/processed_targets_v1/summary.json",
            "quality/processed_targets_v1/summary.json",
        ),
        (
            "features/engineered_features_v1/manifest.json",
            "features/engineered_features_v1/manifest.json",
        ),
        (
            "features/processed_targets_v1/manifest.json",
            "features/processed_targets_v1/manifest.json",
        ),
    ]
    for source, destination in files:
        _copy(base.artifacts_dir / source, artifacts / destination)
    _copy(
        base.quarantine_dir / base.validation_version / "summary.json",
        quarantine / base.validation_version / "summary.json",
    )
    _copy(
        base.processed_dir / base.processed_version / "dataset.parquet",
        processed / base.processed_version / "dataset.parquet",
    )
    settings = replace(
        base,
        artifacts_dir=artifacts,
        quarantine_dir=quarantine,
        processed_dir=processed,
    )
    return run_eda(settings), settings


@pytest.mark.data_contract
def test_eda_reconciles_every_completed_stage(eda_run) -> None:
    run, _ = eda_run
    summary = run.summary

    assert summary["dataset"] == {
        "rows": 181_200,
        "columns": 188,
        "road_count": 25,
        "timestamp_start": "2025-01-01T00:00:00+05:30",
        "timestamp_end": "2025-05-31T23:30:00+05:30",
    }
    assert all(
        record["passed"]
        for record in summary["quality_reconciliation"]["checks"]
    )
    assert summary["quality_reconciliation"]["source"]["total_rows"] == 189_491
    assert summary["quality_reconciliation"]["processed"]["output_rows"] == (
        181_200
    )
    assert summary["context_aggregates"]["record_count"] == 67


@pytest.mark.data_contract
def test_eda_distributions_and_findings_use_observed_denominators(eda_run) -> None:
    run, _ = eda_run
    distributions = run.summary["distributions"]

    assert distributions["congestion"]["Free-flow"]["rows"] == 111_307
    assert distributions["congestion"]["Severe"]["rows"] == 10_004
    assert distributions["accident"] == {
        "total_rows": 181_200,
        "observed_rows": 176_701,
        "unobserved_rows": 4_499,
        "positive_rows": 1_652,
        "negative_rows": 175_049,
        "positive_rate_observed": 0.00934913,
        "negative_to_positive_ratio": 105.9619,
    }
    findings = {record["id"]: record["finding"] for record in run.summary["findings"]}
    assert findings["highest_volume_road"].startswith("NL-006")
    assert findings["peak_hour"].startswith("Local hour 8")
    assert findings["slowest_weather"].startswith("Rain")


@pytest.mark.data_contract
def test_eda_correlation_contract_excludes_future_columns(eda_run) -> None:
    run, _ = eda_run
    correlation = run.summary["correlation"]

    assert correlation["feature_count"] == 25
    assert not any(name.startswith("target_") for name in correlation["features"])
    assert correlation["target"] == "target_volume_h1"
    assert len(correlation["redundant_pairs"]) == 3
    assert correlation["target_correlations"][0]["feature"] == "traffic_volume"
    assert correlation["target_correlations"][0]["correlation"] == 0.92952661


@pytest.mark.data_contract
def test_eda_outputs_are_valid_and_deterministic(eda_run) -> None:
    first, settings = eda_run
    paths = {
        "summary": first.summary_path,
        "report": first.report_path,
        "contexts": first.contexts_path,
        "correlation": first.correlation_path,
        "covariance": first.covariance_path,
        "environment": first.environment_path,
        **first.figure_paths,
    }
    for path in paths.values():
        assert path.is_file() and path.stat().st_size > 0
    for path in first.figure_paths.values():
        with Image.open(path) as image:
            assert image.format == "PNG"
            assert image.width >= 1_000
            assert image.height >= 500
    hashes = {name: sha256_file(path) for name, path in paths.items()}

    repeated = run_eda(settings)
    repeated_paths = {
        "summary": repeated.summary_path,
        "report": repeated.report_path,
        "contexts": repeated.contexts_path,
        "correlation": repeated.correlation_path,
        "covariance": repeated.covariance_path,
        "environment": repeated.environment_path,
        **repeated.figure_paths,
    }
    assert repeated.summary == first.summary
    assert {
        name: sha256_file(path) for name, path in repeated_paths.items()
    } == hashes
    assert b"\r\n" not in repeated.summary_path.read_bytes()
    assert b"\r\n" not in repeated.report_path.read_bytes()


@pytest.mark.data_contract
def test_processed_hash_is_checked_before_eda_read(eda_run) -> None:
    _, settings = eda_run
    path = settings.processed_dir / settings.processed_version / "dataset.parquet"
    original = path.read_bytes()
    try:
        path.write_bytes(original + b"tampered")
        with pytest.raises(RuntimeError, match="byte count changed"):
            load_verified_processed(settings)
    finally:
        path.write_bytes(original)
