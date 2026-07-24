"""Standalone configuration contract for the Step 14 classical registry."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from flowcast.settings import Settings


REGISTRY_CONFIG_PATH = Path("config/registry.yaml")
EXPECTED_TARGETS = (
    "volume",
    "speed",
    "travel_time",
    "congestion",
    "accident",
)
EXPECTED_HORIZONS = (1, 2, 3, 4)
VALID_SOURCES = {"regression", "classification"}
VALID_DIRECTIONS = {"minimize", "maximize"}
VALID_ACCEPTANCE_OPERATORS = {
    "less_than_or_equal",
    "greater_than_or_equal",
}


def registry_config_path(settings: Settings) -> Path:
    """Return the registry config without changing frozen training config."""

    return settings.root / REGISTRY_CONFIG_PATH


def _validate_target(record: dict[str, Any]) -> None:
    required = {
        "key",
        "source",
        "task_type",
        "primary_metric",
        "direction",
    }
    if not required.issubset(record):
        missing = sorted(required - set(record))
        raise ValueError(f"Registry target is missing fields: {missing}")
    if str(record["source"]) not in VALID_SOURCES:
        raise ValueError(f"Unsupported registry source: {record['source']}")
    if str(record["direction"]) not in VALID_DIRECTIONS:
        raise ValueError(f"Unsupported metric direction: {record['direction']}")
    acceptance_fields = {
        "acceptance_metric",
        "acceptance_operator",
        "acceptance_value",
    }
    present = acceptance_fields.intersection(record)
    if present and present != acceptance_fields:
        raise ValueError("Registry acceptance fields must be configured together")
    if present and record["acceptance_operator"] not in VALID_ACCEPTANCE_OPERATORS:
        raise ValueError(
            f"Unsupported acceptance operator: {record['acceptance_operator']}"
        )


def load_registry_config(
    settings: Settings,
) -> tuple[dict[str, Any], Path]:
    """Load and validate the independent Step 14 registry configuration."""

    path = registry_config_path(settings)
    if not path.is_file():
        raise FileNotFoundError(f"Registry configuration is missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload: dict[str, Any] = yaml.safe_load(handle)
    section = payload.get("classical_registry")
    if not isinstance(section, dict):
        raise ValueError("registry.yaml must define classical_registry")
    if section.get("contract_version") != "classical_registry_v1":
        raise ValueError("Unsupported classical registry contract")
    targets = section.get("targets")
    if not isinstance(targets, list):
        raise ValueError("Registry targets must be a list")
    for record in targets:
        if not isinstance(record, dict):
            raise ValueError("Each registry target must be a mapping")
        _validate_target(record)
    target_keys = tuple(str(record["key"]) for record in targets)
    if target_keys != EXPECTED_TARGETS:
        raise ValueError(
            f"Registry targets must be ordered as {EXPECTED_TARGETS}, got {target_keys}"
        )
    horizons = tuple(int(value) for value in section.get("horizons", []))
    if horizons != EXPECTED_HORIZONS:
        raise ValueError(
            f"Registry horizons must be {EXPECTED_HORIZONS}, got {horizons}"
        )
    upstream = section.get("upstream", {})
    if set(upstream) != {"regression_version", "classification_version"}:
        raise ValueError("Registry must name both frozen upstream versions")
    if section.get("prediction_mapping") != "indexed_source_manifest":
        raise ValueError("Only indexed source prediction mapping is supported")
    return section, path
