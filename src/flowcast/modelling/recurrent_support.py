"""Preparation and pre-test freeze helpers for recurrent orchestration."""

from __future__ import annotations

from dataclasses import asdict
from importlib import metadata
from pathlib import Path
import platform
from typing import Any

import numpy as np
import torch

from flowcast.data.artifacts import artifact_record, write_json
from flowcast.modelling.classical_report import write_csv
from flowcast.modelling.recurrent_artifacts import RecurrentPaths
from flowcast.modelling.recurrent_config import RecurrentCandidate
from flowcast.modelling.recurrent_model import RecurrentVolumeForecaster
from flowcast.modelling.recurrent_outputs import candidate_frame, curves_frame
from flowcast.modelling.recurrent_training import CandidateTrainingResult
from flowcast.modelling.sequence_data import (
    PreparedPartition,
    build_sequence_endpoints,
    sequence_manifest,
)
from flowcast.settings import Settings


def write_environment_snapshot(path: Path) -> None:
    """Persist the complete installed distribution set for reproduction."""

    distributions = sorted(
        {
            (
                str(distribution.metadata["Name"]).lower(),
                str(distribution.version),
            )
            for distribution in metadata.distributions()
            if distribution.metadata["Name"]
        }
    )
    lines = [
        f"python=={platform.python_version()}",
        f"platform=={platform.platform()}",
        *[f"{name}=={version}" for name, version in distributions],
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def target_columns(config: dict[str, Any]) -> tuple[str, ...]:
    """Return the configured four volume targets in horizon order."""

    return tuple(str(value) for value in config["target"]["columns"])


def input_features(modeling: Any) -> list[str]:
    """Return the frozen Step 10 input feature order."""

    return [
        str(record["name"])
        for record in modeling.schema["input_features"]
    ]


def candidate_endpoints(
    partition: PreparedPartition,
    candidate: RecurrentCandidate,
    config: dict[str, Any],
    common_keys: set[tuple[str, Any]] | None = None,
) -> np.ndarray:
    """Build one candidate's isolated sequence endpoints."""

    return build_sequence_endpoints(
        partition,
        candidate.sequence_length,
        config["target"]["horizons"],
        int(config["sequence"]["cadence_minutes"]),
        allowed_keys=common_keys,
    )


def split_record(
    partition: PreparedPartition,
    endpoints: np.ndarray,
) -> dict[str, Any]:
    """Summarize one sequence-eligible partition."""

    rows = partition.frame.iloc[endpoints]
    return {
        "origin_start": rows["timestamp"].min().isoformat(),
        "origin_end": rows["timestamp"].max().isoformat(),
        "eligible_sequence_rows": int(len(rows)),
        "road_count": int(rows["road_id"].nunique()),
    }


def portable_record_path(record: dict[str, Any], settings: Settings) -> Path:
    """Resolve a portable artifact record against the repository root."""

    path = Path(str(record["path"]))
    return path if path.is_absolute() else settings.root / path


def build_feature_manifest(
    modeling: Any,
    prepared: PreparedPartition,
    features: list[str],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Describe recurrent input/output features and frozen preprocessing."""

    return {
        "contract_version": "recurrent_feature_manifest_v1",
        "preprocessing_version": modeling.summary["version"],
        "preprocessing_family": "recurrent",
        "input_feature_count": len(features),
        "input_features": features,
        "output_feature_count": len(prepared.feature_names),
        "output_features": list(prepared.feature_names),
        "known_at_origin_only": True,
        "training_fitted_preprocessor": modeling.summary["artifacts"][
            "preprocessors"
        ]["recurrent"],
        "source_feature_schema": modeling.summary["artifacts"]["feature_schema"],
        "sequence_candidates": [
            {
                "candidate_id": record["candidate_id"],
                "sequence_length": int(record["sequence_length"]),
            }
            for record in config["candidates"]
        ],
    }


def _pretest_card(
    version: str,
    seed: int,
    selected: CandidateTrainingResult,
    split_summary: dict[str, Any],
    feature_manifest: dict[str, Any],
    scaler: dict[str, Any],
    checkpoint: dict[str, Any],
    selection: dict[str, Any],
) -> dict[str, Any]:
    return {
        "contract_version": "flowcast_recurrent_pretest_card_v1",
        "status": "frozen_before_test_access",
        "model_version": version,
        "seed": seed,
        "target": "volume_multi_horizon",
        "selection": {
            "candidate_id": selected.candidate.candidate_id,
            "architecture": selected.architecture,
            "best_epoch": selected.best_epoch,
            "validation_mean_rmse": selected.best_validation_mean_rmse,
            "test_metrics_present": False,
        },
        "data": split_summary,
        "features": feature_manifest,
        "target_scaling": scaler,
        "artifacts": {
            "checkpoint": checkpoint,
            "selection": selection,
        },
    }


def _save_checkpoint(
    path: Path,
    version: str,
    selected: CandidateTrainingResult,
    scaler: dict[str, Any],
    feature_manifest: dict[str, Any],
    split_summary: dict[str, Any],
    selection_sha256: str,
    seed: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "contract_version": "recurrent_state_dict_v1",
            "version": version,
            "state_dict": selected.best_state,
            "architecture": selected.architecture,
            "target_scaler": scaler,
            "feature_manifest": feature_manifest,
            "split": split_summary,
            "selection_sha256": selection_sha256,
            "seed": seed,
            "validation_metrics": selected.validation_metrics,
        },
        path,
    )


def reload_model(
    checkpoint_path: Path,
    candidate: RecurrentCandidate,
    input_size: int,
    device: torch.device,
) -> RecurrentVolumeForecaster:
    """Reload only the saved state dictionary into a fresh architecture."""

    payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = RecurrentVolumeForecaster(input_size, candidate, output_size=4)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model


def persist_pretest_freeze(
    paths: RecurrentPaths,
    settings: Settings,
    config: dict[str, Any],
    candidates: list[RecurrentCandidate],
    results: list[CandidateTrainingResult],
    selected: CandidateTrainingResult,
    training: PreparedPartition,
    validation: PreparedPartition,
    train_endpoints: np.ndarray,
    validation_endpoints: np.ndarray,
    scaler: Any,
    feature_manifest: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Persist selection, scaling, curves, card, and checkpoint before test."""

    write_csv(candidate_frame(results), paths.candidates_path)
    write_csv(curves_frame(results), paths.curves_path)
    scaler_metadata = scaler.metadata()
    write_json(scaler_metadata, paths.target_scaler_path)
    write_json(feature_manifest, paths.feature_manifest_path)
    cadence = int(config["sequence"]["cadence_minutes"])
    pretest_sequences = {
        "contract_version": "recurrent_sequence_manifest_v1",
        "status": "frozen_before_test_access",
        "selected_candidate_id": selected.candidate.candidate_id,
        "candidate_sequence_lengths": {
            candidate.candidate_id: candidate.sequence_length
            for candidate in candidates
        },
        "train": sequence_manifest(
            training,
            train_endpoints,
            selected.candidate.sequence_length,
            cadence,
        ),
        "validation": sequence_manifest(
            validation,
            validation_endpoints,
            selected.candidate.sequence_length,
            cadence,
        ),
        "test_partition_loaded": False,
    }
    write_json(pretest_sequences, paths.pretest_sequence_manifest_path)
    selection = {
        "contract_version": "recurrent_selection_manifest_v1",
        "version": paths.version,
        "status": "frozen_before_test_access",
        "selected_candidate_id": selected.candidate.candidate_id,
        "primary_metric": "validation_mean_rmse",
        "selected_validation_mean_rmse": selected.best_validation_mean_rmse,
        "test_metrics_present": False,
        "test_partition_loaded": False,
        "candidates": [
            {
                "candidate_id": result.candidate.candidate_id,
                "validation_mean_rmse": result.best_validation_mean_rmse,
                "parameter_count": result.architecture["parameter_count"],
                "best_epoch": result.best_epoch,
            }
            for result in results
        ],
        "artifacts": {
            "candidate_metrics": artifact_record(paths.candidates_path, settings),
            "training_curves": artifact_record(paths.curves_path, settings),
            "target_scaler": artifact_record(paths.target_scaler_path, settings),
            "feature_manifest": artifact_record(paths.feature_manifest_path, settings),
            "sequence_manifest": artifact_record(
                paths.pretest_sequence_manifest_path,
                settings,
            ),
        },
    }
    write_json(selection, paths.selection_path)
    selection_record = artifact_record(paths.selection_path, settings)
    split_summary = {
        "train": split_record(training, train_endpoints),
        "validation": split_record(validation, validation_endpoints),
        "test": {"status": "sealed_not_loaded"},
    }
    _save_checkpoint(
        paths.checkpoint_path,
        paths.version,
        selected,
        scaler_metadata,
        feature_manifest,
        split_summary,
        selection_record["sha256"],
        settings.seed,
    )
    checkpoint_record = artifact_record(paths.checkpoint_path, settings)
    write_json(
        _pretest_card(
            paths.version,
            settings.seed,
            selected,
            split_summary,
            feature_manifest,
            scaler_metadata,
            checkpoint_record,
            selection_record,
        ),
        paths.pretest_card_path,
    )
    return selection_record, scaler_metadata
