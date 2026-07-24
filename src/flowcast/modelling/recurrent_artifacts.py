"""Paths and integrity-checked loading for Step 15 recurrent artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from flowcast.data.artifacts import verify_artifact_record
from flowcast.modelling.recurrent_config import load_recurrent_config
from flowcast.modelling.recurrent_model import (
    RecurrentVolumeForecaster,
    candidate_from_architecture,
)
from flowcast.modelling.sequence_data import TargetScaler
from flowcast.settings import Settings


@dataclass(frozen=True)
class RecurrentPaths:
    """Canonical output paths for one recurrent-volume run."""

    version: str
    metrics_dir: Path
    models_dir: Path
    predictions_dir: Path
    cards_dir: Path
    summary_path: Path
    report_path: Path
    environment_path: Path
    selection_path: Path
    candidates_path: Path
    curves_path: Path
    metrics_path: Path
    comparison_path: Path
    pretest_sequence_manifest_path: Path
    sequence_manifest_path: Path
    pretest_card_path: Path
    feature_manifest_path: Path
    target_scaler_path: Path
    checkpoint_path: Path
    predictions_path: Path
    card_json_path: Path
    card_markdown_path: Path
    registry_extension_path: Path


def recurrent_paths(settings: Settings, version: str) -> RecurrentPaths:
    """Return every Step 15 path without creating directories."""

    metrics = settings.artifacts_dir / "metrics" / version
    models = settings.artifacts_dir / "models" / version
    predictions = settings.artifacts_dir / "predictions" / version
    cards = settings.artifacts_dir / "model_cards" / version
    return RecurrentPaths(
        version=version,
        metrics_dir=metrics,
        models_dir=models,
        predictions_dir=predictions,
        cards_dir=cards,
        summary_path=metrics / "summary.json",
        report_path=metrics / "summary.md",
        environment_path=metrics / "environment.txt",
        selection_path=metrics / "selection_manifest.json",
        candidates_path=metrics / "candidate_metrics.csv",
        curves_path=metrics / "training_curves.csv",
        metrics_path=metrics / "horizon_metrics.csv",
        comparison_path=metrics / "classical_comparison.csv",
        pretest_sequence_manifest_path=metrics / "pretest_sequence_manifest.json",
        sequence_manifest_path=metrics / "sequence_manifest.json",
        pretest_card_path=metrics / "pretest_model_card.json",
        feature_manifest_path=models / "feature_manifest.json",
        target_scaler_path=models / "target_scaler.json",
        checkpoint_path=models / "best_checkpoint.pt",
        predictions_path=predictions / "predictions.parquet",
        card_json_path=cards / "volume_multi_horizon.json",
        card_markdown_path=cards / "volume_multi_horizon.md",
        registry_extension_path=metrics / "registry_extension.json",
    )


def read_json(path: Path) -> dict[str, Any]:
    """Read a required JSON artifact."""

    if not path.is_file():
        raise FileNotFoundError(f"Required recurrent artifact is missing: {path}")
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return payload


def _record_path(record: dict[str, Any], settings: Settings) -> Path:
    path = Path(str(record["path"]))
    return path if path.is_absolute() else settings.root / path


def _verify_record(record: dict[str, Any], settings: Settings) -> Path:
    return verify_artifact_record(_record_path(record, settings), record, settings)


def _target_scaler(payload: dict[str, Any]) -> TargetScaler:
    columns = tuple(str(value) for value in payload["columns"])
    return TargetScaler(
        columns=columns,
        mean=np.asarray([payload["mean"][name] for name in columns], dtype=float),
        scale=np.asarray([payload["scale"][name] for name in columns], dtype=float),
        fitted_rows=int(payload["fitted_rows"]),
    )


def load_recurrent_volume_model(
    settings: Settings,
    *,
    version: str = "recurrent_volume_v1",
    device: str = "cpu",
) -> tuple[
    RecurrentVolumeForecaster,
    TargetScaler,
    dict[str, Any],
    dict[str, Any],
]:
    """Verify the complete chain and reload the selected state dictionary."""

    paths = recurrent_paths(settings, version)
    summary = read_json(paths.summary_path)
    if summary.get("contract_version") != "recurrent_volume_v1":
        raise RuntimeError("Unsupported recurrent summary contract")
    if summary.get("version") != version:
        raise RuntimeError("Recurrent summary version changed")
    _, _, config_path = load_recurrent_config(settings)
    verify_artifact_record(config_path, summary["configuration"], settings)
    for record in summary["upstream"].values():
        _verify_record(record, settings)
    for record in summary["artifacts"].values():
        _verify_record(record, settings)
    card = read_json(_verify_record(summary["model"]["model_card_json"], settings))
    _verify_record(summary["model"]["model_card_markdown"], settings)
    checkpoint_path = _verify_record(summary["model"]["checkpoint"], settings)
    scaler_payload = read_json(
        _verify_record(summary["model"]["target_scaler"], settings)
    )
    checkpoint = torch.load(
        checkpoint_path,
        map_location=torch.device(device),
        weights_only=True,
    )
    architecture = checkpoint["architecture"]
    candidate = candidate_from_architecture(architecture)
    model = RecurrentVolumeForecaster(
        int(architecture["input_size"]),
        candidate,
        output_size=int(architecture["output_size"]),
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(torch.device(device))
    model.eval()
    if checkpoint["selection_sha256"] != summary["selection"]["sha256"]:
        raise RuntimeError("Checkpoint selection lineage changed")
    if card["artifacts"]["checkpoint"] != summary["model"]["checkpoint"]:
        raise RuntimeError("Model card checkpoint lineage changed")
    return model, _target_scaler(scaler_payload), card, summary
