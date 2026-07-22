"""Full-data Step 10 split, leakage, preprocessing, and artifact contracts."""

from __future__ import annotations

import json
import shutil
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from flowcast.data.audit import sha256_file
from flowcast.modelling.inputs import (
    load_modeling_partition,
    load_preprocessor,
    load_verified_modeling_artifacts,
)
from flowcast.modelling.pipeline import run_modeling_prep
from flowcast.settings import load_settings


def _copy(source, destination) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


@pytest.fixture(scope="module")
def modeling_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("modeling-contract")
    base = load_settings()
    artifacts = root / "artifacts"
    processed = root / "processed"
    models_config = root / "models.yaml"
    _copy(base.models_config_path, models_config)
    copies = [
        (
            base.processed_dir / base.processed_version / "dataset.parquet",
            processed / base.processed_version / "dataset.parquet",
        ),
        (
            base.artifacts_dir
            / "quality"
            / base.processed_version
            / "summary.json",
            artifacts / "quality" / base.processed_version / "summary.json",
        ),
        (
            base.artifacts_dir
            / "features"
            / base.processed_version
            / "manifest.json",
            artifacts / "features" / base.processed_version / "manifest.json",
        ),
        (
            base.artifacts_dir
            / "quality"
            / base.feature_version
            / "summary.json",
            artifacts / "quality" / base.feature_version / "summary.json",
        ),
        (
            base.artifacts_dir
            / "features"
            / base.feature_version
            / "manifest.json",
            artifacts / "features" / base.feature_version / "manifest.json",
        ),
        (
            base.artifacts_dir
            / "reports"
            / base.eda_version
            / "summary.json",
            artifacts / "reports" / base.eda_version / "summary.json",
        ),
    ]
    for source, destination in copies:
        _copy(source, destination)
    settings = replace(
        base,
        artifacts_dir=artifacts,
        processed_dir=processed,
        models_config_path=models_config,
    )
    return run_modeling_prep(settings), settings


@pytest.mark.data_contract
def test_every_origin_has_one_exact_chronological_partition(modeling_run) -> None:
    run, _ = modeling_run
    assignments = run.assignments
    partitions = run.summary["split"]["partitions"]

    assert len(assignments) == 181_200
    assert not assignments.duplicated(["road_id", "timestamp"]).any()
    assert assignments["split"].value_counts().to_dict() == {
        "train": 126_850,
        "validation": 27_175,
        "test": 27_175,
    }
    assert partitions["train"] == {
        "start": "2025-01-01T00:00:00+05:30",
        "end": "2025-04-16T16:30:00+05:30",
        "timestamp_count": 5_074,
        "row_count": 126_850,
        "ratio_of_timestamps": 0.7000551876,
    }
    assert partitions["validation"]["start"] == "2025-04-16T17:00:00+05:30"
    assert partitions["test"]["start"] == "2025-05-09T08:30:00+05:30"
    per_road = assignments.groupby(["road_id", "split"], observed=True).size()
    assert per_road.groupby(level="split").nunique().eq(1).all()


@pytest.mark.data_contract
def test_every_eligible_target_stays_inside_origin_partition(modeling_run) -> None:
    run, settings = modeling_run
    frame = pd.read_parquet(
        settings.processed_dir / settings.processed_version / "dataset.parquet"
    )
    assignments = run.assignments
    timestamp_to_split = (
        assignments[["timestamp", "split"]]
        .drop_duplicates()
        .set_index("timestamp")["split"]
    )
    for horizon in range(1, 5):
        within = assignments[f"target_within_split_h{horizon}"]
        target_split = frame[f"target_timestamp_h{horizon}"].map(
            timestamp_to_split
        )
        assert target_split[within].equals(assignments.loc[within, "split"])
        for partition in ("train", "validation", "test"):
            selected = assignments["split"].eq(partition)
            assert int((selected & ~within).sum()) == 25 * horizon

    coverage = run.summary["split"]["target_coverage"]
    assert coverage["target_volume_h1"]["train"]["eligible_rows"] == 126_825
    assert coverage["target_volume_h4"]["validation"]["eligible_rows"] == 27_075
    assert coverage["target_accident_h1"]["train"]["positive_rows"] == 1_156
    assert coverage["target_accident_h4"]["test"]["positive_rows"] == 258


@pytest.mark.data_contract
def test_time_series_cv_is_training_only_and_horizon_gapped(modeling_run) -> None:
    run, _ = modeling_run
    cv = run.summary["cross_validation"]
    train_end = pd.Timestamp(
        run.summary["split"]["partitions"]["train"]["end"]
    )

    assert cv["fold_count"] == 5
    assert cv["validation_windows"] == 336
    assert cv["gap_windows"] == cv["maximum_horizon_windows"] == 4
    previous_validation_end = None
    for fold in cv["folds"]:
        fold_train_end = pd.Timestamp(fold["train_end"])
        validation_start = pd.Timestamp(fold["validation_start"])
        validation_end = pd.Timestamp(fold["validation_end"])
        assert fold_train_end < validation_start <= validation_end <= train_end
        assert validation_start - fold_train_end == pd.Timedelta(minutes=150)
        if previous_validation_end is not None:
            assert validation_start - previous_validation_end == pd.Timedelta(
                minutes=30
            )
        previous_validation_end = validation_end
    assert previous_validation_end == train_end


@pytest.mark.data_contract
def test_feature_schema_and_statistics_are_training_only(modeling_run) -> None:
    run, settings = modeling_run
    schema = json.loads(run.schema_path.read_text(encoding="utf-8"))
    names = [record["name"] for record in schema["input_features"]]
    frame = pd.read_parquet(
        settings.processed_dir / settings.processed_version / "dataset.parquet"
    )
    train = frame.loc[run.assignments["split"].eq("train")]
    validation = frame.loc[run.assignments["split"].eq("validation")]

    assert schema["feature_count"] == len(names) == 62
    assert "road_id" not in names and "timestamp" not in names
    assert not any(name.startswith("target_") for name in names)
    assert schema["fitted_on"]["row_count"] == 126_850
    assert schema["validation_mode"] == "transform_only"
    assert schema["test_access_default"] == "sealed"
    for family, record in schema["families"].items():
        assert record["input_feature_count"] == 62
        assert record["output_feature_count"] == 64
        assert record["artifact"]["bytes"] > 0
        assert family in {"linear", "tree", "svm", "recurrent"}
    recorded_mean = schema["families"]["linear"]["training_statistics"][
        "numeric"
    ]["scaler"]["mean"]["traffic_volume"]
    assert recorded_mean == pytest.approx(float(train["traffic_volume"].mean()))
    assert recorded_mean != pytest.approx(
        float(validation["traffic_volume"].mean())
    )
    recurrent = schema["families"]["recurrent"]["training_statistics"]
    assert recurrent["bounded_numeric"]["scaler"]["type"] == "minmax"
    assert recurrent["bounded_numeric"]["scaler"]["data_max"][
        "occupancy"
    ] == float(train["occupancy"].max())
    accident = schema["training_class_statistics"]["target_accident_h1"]
    assert accident["source_partition"] == "train"
    assert accident["class_counts"] == {"false": 122_560, "true": 1_156}
    assert accident["scale_pos_weight"] == pytest.approx(122_560 / 1_156)


@pytest.mark.data_contract
def test_verified_loaders_keep_test_sealed_and_load_preprocessors(modeling_run) -> None:
    run, settings = modeling_run
    validation = load_modeling_partition(settings, "validation")

    assert len(validation) == 27_175
    assert validation["split"].eq("validation").all()
    with pytest.raises(PermissionError, match="Test partition is sealed"):
        load_modeling_partition(settings, "test")
    schema = json.loads(run.schema_path.read_text(encoding="utf-8"))
    features = [record["name"] for record in schema["input_features"]]
    for family in ("linear", "tree", "svm", "recurrent"):
        processor = load_preprocessor(settings, family)
        transformed = processor.transform(validation[features].head(32))
        assert transformed.shape == (32, 64)
        assert np.isfinite(np.asarray(transformed, dtype=float)).all()


@pytest.mark.data_contract
def test_step_10_artifacts_are_deterministic_and_tamper_evident(
    modeling_run,
) -> None:
    first, settings = modeling_run
    paths = {
        "assignments": first.assignments_path,
        "folds": first.folds_path,
        "schema": first.schema_path,
        "summary": first.summary_path,
        "report": first.markdown_path,
        **{
            f"preprocessor_{family}": path
            for family, path in first.preprocessor_paths.items()
        },
    }
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    repeated = run_modeling_prep(settings)
    repeated_paths = {
        "assignments": repeated.assignments_path,
        "folds": repeated.folds_path,
        "schema": repeated.schema_path,
        "summary": repeated.summary_path,
        "report": repeated.markdown_path,
        **{
            f"preprocessor_{family}": path
            for family, path in repeated.preprocessor_paths.items()
        },
    }
    assert repeated.summary == first.summary
    assert {
        name: sha256_file(path) for name, path in repeated_paths.items()
    } == hashes
    assert b"\r\n" not in repeated.summary_path.read_bytes()
    assert b"\r\n" not in repeated.markdown_path.read_bytes()

    original = repeated.assignments_path.read_bytes()
    try:
        repeated.assignments_path.write_bytes(original + b"tampered")
        with pytest.raises(RuntimeError, match="byte count changed"):
            load_verified_modeling_artifacts(settings)
    finally:
        repeated.assignments_path.write_bytes(original)

    config_original = settings.models_config_path.read_bytes()
    try:
        settings.models_config_path.write_bytes(config_original + b"\n# tampered\n")
        with pytest.raises(RuntimeError, match="byte count changed"):
            load_verified_modeling_artifacts(settings)
    finally:
        settings.models_config_path.write_bytes(config_original)
