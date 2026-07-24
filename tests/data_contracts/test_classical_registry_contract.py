"""Full-artifact Step 14 registry, mapping, lineage, and loader contracts."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import shutil

import joblib
import pytest

from flowcast.modelling.registry import run_classical_registry
from flowcast.modelling.registry_artifacts import (
    load_classical_registry,
    load_registered_model,
)
from flowcast.settings import load_settings


@pytest.fixture(scope="module")
def registry_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("classical-registry-contract")
    base = load_settings()
    artifacts = root / "artifacts"
    for version in (
        "classical_regression_v1",
        "classical_classification_v1",
    ):
        source = base.artifacts_dir / "metrics" / version / "summary.json"
        destination = artifacts / "metrics" / version / "summary.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    settings = replace(base, artifacts_dir=artifacts)
    return run_classical_registry(settings), settings


@pytest.mark.data_contract
def test_registry_has_exact_required_coverage_and_task_metrics(registry_run) -> None:
    run, _ = registry_run
    entries = run.registry["entries"]

    assert run.summary["coverage"] == {
        "target_count": 5,
        "horizon_count": 4,
        "entry_count": 20,
        "regression_entries": 12,
        "classification_entries": 8,
        "prediction_source_count": 2,
        "prediction_rows": 1_078_957,
    }
    assert len(entries) == 20
    assert len({entry["registry_key"] for entry in entries}) == 20
    assert len({entry["job_id"] for entry in entries}) == 20
    assert {
        (entry["target"], entry["horizon_windows"])
        for entry in entries
    } == {
        (target, horizon)
        for target in (
            "volume",
            "speed",
            "travel_time",
            "congestion",
            "accident",
        )
        for horizon in range(1, 5)
    }
    metrics = {
        entry["target"]: entry["primary_metric"]
        for entry in entries
    }
    assert metrics == {
        "volume": "rmse",
        "speed": "rmse",
        "travel_time": "rmse",
        "congestion": "macro_f1",
        "accident": "roc_auc",
    }


@pytest.mark.data_contract
def test_registry_metrics_and_lineage_equal_frozen_sources(registry_run) -> None:
    run, settings = registry_run
    entries = {entry["job_id"]: entry for entry in run.registry["entries"]}
    for source_record in run.summary["sources"].values():
        source_path = Path(str(source_record["path"]))
        if not source_path.is_absolute():
            source_path = settings.root / source_path
        source = json.loads(source_path.read_text(encoding="utf-8"))
        for score in source["scoreboard"]:
            entry = entries[score["job_id"]]
            assert entry["metrics"]["validation"] == score["validation"]
            assert entry["metrics"]["test"] == score["test"]
            assert entry["lineage"]["feature_schema_sha256"] == source[
                "input_modeling"
            ]["feature_schema"]["sha256"]
            assert entry["artifacts"]["predictions"] == source["artifacts"][
                "predictions"
            ]
            assert "test metrics were not selection inputs" in entry[
                "selection_rationale"
            ]


@pytest.mark.data_contract
def test_prediction_index_maps_every_row_to_exactly_one_entry(registry_run) -> None:
    run, _ = registry_run
    index = run.prediction_index
    mapped_jobs = []
    mapped_rows = 0
    for source in index["sources"].values():
        mapped_rows += source["rows"]
        for job in source["jobs"]:
            mapped_jobs.append(job["job_id"])
            assert job["rows"]["validation"] > 0
            assert job["rows"]["test"] > 0
            assert job["rows"]["total"] == (
                job["rows"]["validation"] + job["rows"]["test"]
            )

    assert index["job_count"] == len(mapped_jobs) == len(set(mapped_jobs)) == 20
    assert mapped_rows == index["prediction_rows"] == 1_078_957
    assert set(mapped_jobs) == {
        entry["job_id"] for entry in run.registry["entries"]
    }


@pytest.mark.data_contract
def test_registry_is_deterministic_and_acceptance_is_honest(registry_run) -> None:
    first, settings = registry_run
    paths = (
        first.paths.registry_path,
        first.paths.scoreboard_path,
        first.paths.prediction_index_path,
        first.paths.report_path,
        first.paths.summary_path,
    )
    first_bytes = {path.name: path.read_bytes() for path in paths}
    second = run_classical_registry(settings)

    assert {
        path.name: path.read_bytes() for path in paths
    } == first_bytes
    assert second.summary["acceptance"] == {
        "volume": {
            "evaluated_horizons": 4,
            "met_horizons": 4,
            "all_horizons_met": True,
        },
        "congestion": {
            "evaluated_horizons": 4,
            "met_horizons": 0,
            "all_horizons_met": False,
        },
        "accident": {
            "evaluated_horizons": 4,
            "met_horizons": 0,
            "all_horizons_met": False,
        },
    }
    assert all(record["passed"] for record in second.summary["checks"])


@pytest.mark.data_contract
def test_all_registered_models_reload_and_public_loader_resolves_both_tasks(
    registry_run,
) -> None:
    run, settings = registry_run
    for entry in run.registry["entries"]:
        path = settings.root / entry["artifacts"]["model"]["path"]
        estimator = joblib.load(path)
        assert hasattr(estimator, "predict")

    regression, regression_card, regression_entry = load_registered_model(
        settings,
        "volume",
        1,
    )
    classifier, classifier_card, classifier_entry = load_registered_model(
        settings,
        "accident",
        1,
    )
    assert hasattr(regression, "predict")
    assert hasattr(classifier, "predict_proba")
    assert regression_card["job_id"] == regression_entry["job_id"] == "volume_h1"
    assert classifier_card["job_id"] == classifier_entry["job_id"] == "accident_h1"


@pytest.mark.data_contract
def test_loader_rejects_registry_and_upstream_tampering(registry_run) -> None:
    run, settings = registry_run
    registry_path = run.paths.registry_path
    original_registry = registry_path.read_bytes()
    try:
        registry_path.write_bytes(original_registry + b"tampered")
        with pytest.raises(RuntimeError, match="byte count changed"):
            load_classical_registry(settings)
    finally:
        registry_path.write_bytes(original_registry)

    source_path = (
        settings.artifacts_dir
        / "metrics"
        / "classical_regression_v1"
        / "summary.json"
    )
    original_source = source_path.read_bytes()
    try:
        source_path.write_bytes(original_source + b"tampered")
        with pytest.raises(RuntimeError, match="byte count changed"):
            load_classical_registry(settings)
    finally:
        source_path.write_bytes(original_source)
