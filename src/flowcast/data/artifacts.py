"""Shared deterministic artifact writing and lineage helpers."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from flowcast.data.audit import sha256_file
from flowcast.data.contracts import portable_path
from flowcast.settings import Settings


_SAFE_VERSION = re.compile(r"^[A-Za-z0-9_.-]+$")


def validate_artifact_version(version: str) -> str:
    """Return a path-safe artifact version or raise a clear error."""

    if not version or not _SAFE_VERSION.fullmatch(version):
        raise ValueError(
            "Artifact version must contain only letters, numbers, '.', '_', or '-'"
        )
    return version


def write_parquet(frame: pd.DataFrame, path: Path) -> None:
    """Write a dataframe with the approved deterministic Parquet engine."""

    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False, engine="pyarrow")


def write_json(payload: dict[str, Any], path: Path) -> None:
    """Write sorted, indented UTF-8 JSON with platform-independent line endings."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def artifact_record(path: Path, settings: Settings) -> dict[str, Any]:
    """Return path, size, and SHA-256 lineage for an existing artifact."""

    return {
        "path": portable_path(path, settings.root),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path, settings.hash_chunk_size),
    }


def verify_artifact_record(
    path: Path,
    record: dict[str, Any],
    settings: Settings,
) -> Path:
    """Verify one recorded artifact's size and SHA-256 before consumption."""

    if not path.is_file():
        raise FileNotFoundError(f"Recorded artifact is missing: {path}")
    if path.stat().st_size != int(record["bytes"]):
        raise RuntimeError(f"Recorded artifact byte count changed: {path}")
    if sha256_file(path, settings.hash_chunk_size) != str(record["sha256"]):
        raise RuntimeError(f"Recorded artifact SHA-256 changed: {path}")
    return path
