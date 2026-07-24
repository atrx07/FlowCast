"""Paths and integrity-checked loading for the Step 14 classical registry."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from flowcast.data.artifacts import verify_artifact_record
from flowcast.modelling.classical_artifacts import (
    classical_regression_paths,
    load_classical_regression_model,
)
from flowcast.modelling.classification_artifacts import (
    classification_paths,
    load_classification_model,
)
from flowcast.modelling.registry_config import load_registry_config
from flowcast.settings import Settings


@dataclass(frozen=True)
class ClassicalRegistryPaths:
    """Canonical output paths for one combined classical registry."""

    version: str
    metrics_dir: Path
    summary_path: Path
    registry_path: Path
    scoreboard_path: Path
    prediction_index_path: Path
    report_path: Path


def classical_registry_paths(
    settings: Settings,
    version: str,
) -> ClassicalRegistryPaths:
    """Return all Step 14 output paths without creating them."""

    metrics = settings.artifacts_dir / "metrics" / version
    return ClassicalRegistryPaths(
        version=version,
        metrics_dir=metrics,
        summary_path=metrics / "summary.json",
        registry_path=metrics / "registry.json",
        scoreboard_path=metrics / "scoreboard.csv",
        prediction_index_path=metrics / "prediction_index.json",
        report_path=metrics / "summary.md",
    )


def read_json(path: Path) -> dict[str, Any]:
    """Read a required JSON mapping."""

    if not path.is_file():
        raise FileNotFoundError(f"Required registry artifact is missing: {path}")
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return payload


def record_path(record: dict[str, Any], settings: Settings) -> Path:
    """Resolve a portable artifact record against the repository root."""

    path = Path(str(record["path"]))
    return path if path.is_absolute() else settings.root / path


def verify_record(record: dict[str, Any], settings: Settings) -> Path:
    """Verify one portable artifact record and return its resolved path."""

    return verify_artifact_record(record_path(record, settings), record, settings)


def _verify_source_summary(
    summary: dict[str, Any],
    settings: Settings,
    *,
    contract: str,
    version: str,
) -> None:
    if summary.get("contract_version") != contract:
        raise RuntimeError(f"Unsupported upstream summary contract: {contract}")
    if summary.get("version") != version:
        raise RuntimeError(f"Upstream summary version changed: {version}")
    for name, path in {
        "base": settings.config_path,
        "models": settings.models_config_path,
    }.items():
        verify_artifact_record(path, summary["configuration"][name], settings)
    for record in summary["input_modeling"].values():
        verify_record(record, settings)
    for record in summary["artifacts"].values():
        verify_record(record, settings)
    for model_records in summary["models"].values():
        for record in model_records.values():
            verify_record(record, settings)


def load_verified_source_summaries(
    settings: Settings,
    registry_config: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Load and recursively verify both frozen classical source summaries."""

    upstream = registry_config["upstream"]
    regression_version = str(upstream["regression_version"])
    classification_version = str(upstream["classification_version"])
    regression = read_json(
        classical_regression_paths(settings, regression_version).summary_path
    )
    classification = read_json(
        classification_paths(settings, classification_version).summary_path
    )
    _verify_source_summary(
        regression,
        settings,
        contract="classical_regression_v1",
        version=regression_version,
    )
    _verify_source_summary(
        classification,
        settings,
        contract="classical_classification_v1",
        version=classification_version,
    )
    return {"regression": regression, "classification": classification}


def _validate_registry_entries(
    registry: dict[str, Any],
    sources: dict[str, dict[str, Any]],
) -> None:
    entries = registry.get("entries", [])
    if len(entries) != 20:
        raise RuntimeError("Classical registry must contain exactly 20 entries")
    keys = [str(entry["registry_key"]) for entry in entries]
    jobs = [str(entry["job_id"]) for entry in entries]
    if len(set(keys)) != 20 or len(set(jobs)) != 20:
        raise RuntimeError("Registry keys and job identities must be unique")
    for entry in entries:
        source = sources[str(entry["source"])]
        records = source["models"][str(entry["job_id"])]
        if entry["artifacts"]["model"] != records["model"]:
            raise RuntimeError("Registry model lineage no longer matches its source")
        if entry["artifacts"]["model_card"] != records["model_card_json"]:
            raise RuntimeError("Registry model-card lineage changed")
        if entry["artifacts"]["predictions"] != source["artifacts"]["predictions"]:
            raise RuntimeError("Registry prediction lineage changed")


def load_classical_registry(
    settings: Settings,
    *,
    version: str = "classical_registry_v1",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load the combined registry after verifying every recorded dependency."""

    paths = classical_registry_paths(settings, version)
    summary = read_json(paths.summary_path)
    if summary.get("contract_version") != "classical_registry_v1":
        raise RuntimeError("Unsupported classical registry summary contract")
    if summary.get("version") != version:
        raise RuntimeError("Classical registry summary version changed")
    config, config_path = load_registry_config(settings)
    verify_artifact_record(config_path, summary["configuration"], settings)
    for record in summary["sources"].values():
        verify_record(record, settings)
    for record in summary["artifacts"].values():
        verify_record(record, settings)
    registry = read_json(paths.registry_path)
    if registry.get("contract_version") != "classical_registry_v1":
        raise RuntimeError("Unsupported registry payload contract")
    if registry.get("version") != version:
        raise RuntimeError("Registry payload version changed")
    sources = load_verified_source_summaries(settings, config)
    _validate_registry_entries(registry, sources)
    return registry, summary


def load_registered_model(
    settings: Settings,
    target: str,
    horizon: int,
    *,
    version: str = "classical_registry_v1",
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """Resolve and load one registered model through its verified source loader."""

    registry, _ = load_classical_registry(settings, version=version)
    matches = [
        entry
        for entry in registry["entries"]
        if entry["target"] == target
        and int(entry["horizon_windows"]) == int(horizon)
    ]
    if len(matches) != 1:
        raise KeyError(f"Expected one registry entry for {target}_h{horizon}")
    entry = matches[0]
    source_version = str(entry["model_version"])
    if entry["source"] == "regression":
        estimator, card, _ = load_classical_regression_model(
            settings,
            target,
            horizon,
            version=source_version,
        )
    else:
        estimator, card, _ = load_classification_model(
            settings,
            target,
            horizon,
            version=source_version,
        )
    if card["artifacts"]["model"] != entry["artifacts"]["model"]:
        raise RuntimeError("Loaded model no longer matches its registry entry")
    if card["job_id"] != entry["job_id"]:
        raise RuntimeError("Loaded model-card identity changed")
    return estimator, card, entry
