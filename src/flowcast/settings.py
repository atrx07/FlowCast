"""Versioned configuration loading and repository path resolution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG = Path("config/base.yaml")


@dataclass(frozen=True)
class Settings:
    """Resolved FlowCast settings used by CLI and pipeline modules."""

    root: Path
    config_path: Path
    name: str
    version: str
    seed: int
    timezone: str
    log_level: str
    data_contracts_path: Path
    cleaning_config_path: Path
    features_config_path: Path
    reference_dir: Path
    raw_dir: Path
    interim_dir: Path
    processed_dir: Path
    quarantine_dir: Path
    artifacts_dir: Path
    logs_dir: Path
    audit_version: str
    validation_version: str
    cleaning_version: str
    merge_version: str
    feature_version: str
    processed_version: str
    hash_chunk_size: int


def repository_root() -> Path:
    """Return the repository root independent of the current working directory."""

    return Path(__file__).resolve().parents[2]


def _resolve(root: Path, configured_path: str) -> Path:
    path = Path(configured_path)
    return path if path.is_absolute() else root / path


def load_settings(config_path: Path | str | None = None) -> Settings:
    """Load YAML settings and resolve all configured paths from the repo root."""

    root = repository_root()
    selected = Path(config_path) if config_path else DEFAULT_CONFIG
    selected = selected if selected.is_absolute() else root / selected
    with selected.open("r", encoding="utf-8") as handle:
        config: dict[str, Any] = yaml.safe_load(handle)

    project = config["project"]
    runtime = config["runtime"]
    paths = config["paths"]
    audit = config["audit"]
    validation = config["validation"]
    cleaning = config["cleaning"]
    merge = config["merge"]
    features = config["features"]
    processed = config["processed"]
    return Settings(
        root=root,
        config_path=selected,
        name=str(project["name"]),
        version=str(project["version"]),
        seed=int(runtime["seed"]),
        timezone=str(runtime["timezone"]),
        log_level=str(runtime["log_level"]),
        data_contracts_path=_resolve(root, paths["data_contracts"]),
        cleaning_config_path=_resolve(root, paths["cleaning"]),
        features_config_path=_resolve(root, paths["features"]),
        reference_dir=_resolve(root, paths["reference"]),
        raw_dir=_resolve(root, paths["raw"]),
        interim_dir=_resolve(root, paths["interim"]),
        processed_dir=_resolve(root, paths["processed"]),
        quarantine_dir=_resolve(root, paths["quarantine"]),
        artifacts_dir=_resolve(root, paths["artifacts"]),
        logs_dir=_resolve(root, paths["logs"]),
        audit_version=str(audit["version"]),
        validation_version=str(validation["version"]),
        cleaning_version=str(cleaning["version"]),
        merge_version=str(merge["version"]),
        feature_version=str(features["version"]),
        processed_version=str(processed["version"]),
        hash_chunk_size=int(audit["chunk_size_bytes"]),
    )
