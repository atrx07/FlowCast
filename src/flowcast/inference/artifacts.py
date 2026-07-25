"""Prediction batch persistence and integrity-checked reload."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import pandas as pd

from flowcast.data.artifacts import (
    artifact_record,
    validate_artifact_version,
    verify_artifact_record,
    write_json,
    write_parquet,
)
from flowcast.inference.config import load_inference_config
from flowcast.inference.inputs import load_verified_inference_context
from flowcast.inference.predictor import PredictionResult
from flowcast.inference.schemas import PredictionRequest, validate_prediction_frame
from flowcast.modelling.registry_artifacts import record_path
from flowcast.settings import Settings


@dataclass(frozen=True)
class PredictionBatchPaths:
    """Paths for one deterministic inference request."""

    version: str
    request_id: str
    directory: Path
    predictions_path: Path
    manifest_path: Path


@dataclass(frozen=True)
class LoadedPredictionBatch:
    """One verified persisted prediction batch."""

    manifest: dict[str, Any]
    request: PredictionRequest
    frame: pd.DataFrame
    paths: PredictionBatchPaths


def prediction_batch_paths(
    settings: Settings,
    version: str,
    request_id: str,
    *,
    output_root: Path | None = None,
) -> PredictionBatchPaths:
    """Return prediction paths without creating them."""

    safe_version = validate_artifact_version(version)
    safe_request = validate_artifact_version(request_id)
    root = output_root if output_root is not None else settings.artifacts_dir
    directory = root / "predictions" / safe_version / safe_request
    return PredictionBatchPaths(
        version=safe_version,
        request_id=safe_request,
        directory=directory,
        predictions_path=directory / "predictions.parquet",
        manifest_path=directory / "manifest.json",
    )


def persist_prediction_batch(
    result: PredictionResult,
    settings: Settings,
    *,
    output_root: Path | None = None,
) -> PredictionBatchPaths:
    """Write Parquet plus a JSON request/lineage manifest."""

    config, config_path = load_inference_config(settings)
    request_id = str(result.frame["request_id"].iloc[0])
    paths = prediction_batch_paths(
        settings,
        str(config["version"]),
        request_id,
        output_root=output_root,
    )
    write_parquet(result.frame, paths.predictions_path)
    manifest = {
        "contract_version": "inference_reporting_v1",
        "version": str(config["version"]),
        "schema_version": str(config["output"]["schema_version"]),
        "request_id": request_id,
        "request": result.request.payload(),
        "coverage": {
            "row_count": len(result.frame),
            "road_count": int(result.frame["road_id"].nunique()),
            "horizon_count": int(result.frame["horizon_windows"].nunique()),
        },
        "configuration": artifact_record(config_path, settings),
        "upstream": result.lineage["upstream"],
        "models": result.lineage["models"],
        "artifacts": {
            "predictions": artifact_record(paths.predictions_path, settings),
        },
        "runtime": {
            "initialization_seconds": result.initialization_seconds,
            "prediction_seconds": result.prediction_seconds,
            "cold_total_seconds": result.total_seconds,
            "device": result.request.device,
        },
        "checks": {
            "schema_validated": True,
            "no_retraining": True,
            "confidence_attached": True,
            "lineage_complete": True,
        },
    }
    write_json(manifest, paths.manifest_path)
    return paths


def _verify_models(
    models: dict[str, dict[str, Any]],
    settings: Settings,
) -> None:
    for model in models.values():
        for value in model.values():
            if isinstance(value, dict) and {"path", "bytes", "sha256"} <= set(value):
                verify_artifact_record(record_path(value, settings), value, settings)


def load_prediction_batch(
    settings: Settings,
    manifest_path: Path,
) -> LoadedPredictionBatch:
    """Recursively verify and load one persisted forecast batch."""

    if not manifest_path.is_file():
        raise FileNotFoundError(f"Prediction manifest is missing: {manifest_path}")
    manifest: dict[str, Any] = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )
    if manifest.get("contract_version") != "inference_reporting_v1":
        raise RuntimeError("Unsupported prediction manifest contract")
    config, config_path = load_inference_config(settings)
    if manifest.get("version") != config["version"]:
        raise RuntimeError("Prediction manifest version changed")
    verify_artifact_record(config_path, manifest["configuration"], settings)
    context = load_verified_inference_context(settings)
    if manifest["upstream"] != context.upstream_records:
        raise RuntimeError("Prediction upstream lineage changed")
    _verify_models(manifest["models"], settings)
    prediction_path = verify_artifact_record(
        record_path(manifest["artifacts"]["predictions"], settings),
        manifest["artifacts"]["predictions"],
        settings,
    )
    request_payload = manifest["request"]
    request = PredictionRequest.from_values(
        request_payload["road_ids"],
        request_payload["origin_timestamp"],
        request_payload["horizons"],
        device=request_payload["device"],
    )
    expected_id = request.identifier(str(config["version"]))
    if expected_id != manifest["request_id"]:
        raise RuntimeError("Prediction request identifier changed")
    frame = validate_prediction_frame(pd.read_parquet(prediction_path), request)
    coverage = manifest["coverage"]
    if len(frame) != int(coverage["row_count"]):
        raise RuntimeError("Prediction manifest row count changed")
    paths = PredictionBatchPaths(
        version=str(manifest["version"]),
        request_id=str(manifest["request_id"]),
        directory=manifest_path.parent,
        predictions_path=prediction_path,
        manifest_path=manifest_path,
    )
    return LoadedPredictionBatch(manifest, request, frame, paths)
