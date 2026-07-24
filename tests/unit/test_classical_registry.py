"""Unit contracts for the independent Step 14 registry configuration."""

from __future__ import annotations

from copy import deepcopy

import pytest

from flowcast.modelling.registry_config import (
    EXPECTED_HORIZONS,
    EXPECTED_TARGETS,
    load_registry_config,
)
from flowcast.modelling.registry_outputs import build_scoreboard
from flowcast.settings import load_settings


def test_registry_config_defines_exactly_twenty_task_horizon_jobs() -> None:
    config, path = load_registry_config(load_settings())

    assert path.name == "registry.yaml"
    assert tuple(config["horizons"]) == EXPECTED_HORIZONS
    assert tuple(record["key"] for record in config["targets"]) == EXPECTED_TARGETS
    assert len(config["horizons"]) * len(config["targets"]) == 20
    assert config["prediction_mapping"] == "indexed_source_manifest"
    assert config["upstream"] == {
        "regression_version": "classical_regression_v1",
        "classification_version": "classical_classification_v1",
    }


def test_registry_primary_metrics_and_directions_are_task_aware() -> None:
    config, _ = load_registry_config(load_settings())
    targets = {record["key"]: record for record in config["targets"]}

    assert targets["volume"]["primary_metric"] == "rmse"
    assert targets["speed"]["direction"] == "minimize"
    assert targets["travel_time"]["direction"] == "minimize"
    assert targets["congestion"]["primary_metric"] == "macro_f1"
    assert targets["congestion"]["direction"] == "maximize"
    assert targets["accident"]["primary_metric"] == "roc_auc"
    assert targets["accident"]["direction"] == "maximize"


def test_scoreboard_uses_each_entry_primary_metric_without_cross_task_ranking() -> None:
    base = {
        "registry_key": "volume/h1/family/version",
        "job_id": "volume_h1",
        "target": "volume",
        "task_type": "regression",
        "horizon_windows": 1,
        "horizon_minutes": 30,
        "selected_family": "family",
        "candidate_id": "candidate",
        "primary_metric": "rmse",
        "metric_direction": "minimize",
        "selection_evidence": {
            "mean_cv_primary_metric": 2.0,
            "std_cv_primary_metric": 0.2,
        },
        "metrics": {
            "validation": {"rmse": 1.5},
            "test": {"rmse": 1.7},
        },
        "acceptance": None,
        "runtime": {
            "validation_fit_seconds": 1.0,
            "test_prediction_seconds": 0.1,
        },
        "model_version": "version",
        "artifacts": {
            "model": {"path": "model", "sha256": "m"},
            "model_card": {"path": "card", "sha256": "c"},
            "predictions": {"path": "predictions", "sha256": "p"},
        },
        "features": {"preprocessing_version": "prep"},
        "lineage": {
            "processed_sha256": "d",
            "feature_schema_sha256": "f",
            "selection_sha256": "s",
        },
    }
    classification = deepcopy(base)
    classification.update(
        {
            "registry_key": "accident/h1/family/version",
            "job_id": "accident_h1",
            "target": "accident",
            "task_type": "classification_binary",
            "primary_metric": "roc_auc",
            "metric_direction": "maximize",
            "metrics": {
                "validation": {"roc_auc": 0.6},
                "test": {"roc_auc": 0.7},
            },
        }
    )

    scoreboard = build_scoreboard([base, classification])

    assert scoreboard["primary_metric"].tolist() == ["rmse", "roc_auc"]
    assert scoreboard["test_primary_metric"].tolist() == pytest.approx([1.7, 0.7])
    assert scoreboard["metric_direction"].tolist() == ["minimize", "maximize"]
