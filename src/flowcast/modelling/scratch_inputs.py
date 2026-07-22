"""Verified loading for persisted Step 11 regression artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from flowcast.data.artifacts import (
    validate_artifact_version,
    verify_artifact_record,
)
from flowcast.modelling.config import load_model_config
from flowcast.modelling.scratch_linear import NumpyLinearRegressor
from flowcast.settings import Settings


def _artifact_path(record: Mapping[str, Any], settings: Settings) -> Path:
    path = Path(str(record["path"]))
    return path if path.is_absolute() else settings.root / path


def _verify_records(records: Mapping[str, Any], settings: Settings) -> None:
    for record in records.values():
        if isinstance(record, dict) and {"path", "bytes", "sha256"} <= set(record):
            verify_artifact_record(_artifact_path(record, settings), record, settings)


def load_scratch_linear_model(
    settings: Settings,
    version: str | None = None,
) -> tuple[NumpyLinearRegressor, dict[str, Any]]:
    """Verify and load the persisted Step 11 scratch model and summary."""

    config = load_model_config(settings)
    scratch = config["scratch_linear"]
    selected_version = validate_artifact_version(
        version or str(scratch["version"])
    )
    summary_path = (
        settings.artifacts_dir / "metrics" / selected_version / "summary.json"
    )
    if not summary_path.is_file():
        raise FileNotFoundError(f"Scratch-linear summary is missing: {summary_path}")
    summary: dict[str, Any] = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("contract_version") != scratch["contract_version"]:
        raise RuntimeError("Scratch-linear summary contract changed")
    for name, path in {
        "base": settings.config_path,
        "models": settings.models_config_path,
    }.items():
        verify_artifact_record(path, summary["configuration"][name], settings)
    _verify_records(summary["input_modeling"], settings)
    _verify_records(summary["artifacts"], settings)
    model_record = summary["artifacts"]["model"]
    model_path = _artifact_path(model_record, settings)
    payload = json.loads(model_path.read_text(encoding="utf-8"))
    return NumpyLinearRegressor.from_payload(payload), summary
