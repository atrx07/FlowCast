"""Independent configuration contract for the Step 15 recurrent model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from flowcast.settings import Settings


RECURRENT_CONFIG_PATH = Path("config/recurrent.yaml")
EXPECTED_HORIZONS = (1, 2, 3, 4)
SUPPORTED_RECURRENT_TYPES = {"lstm", "gru"}


@dataclass(frozen=True)
class RecurrentCandidate:
    """One predeclared recurrent architecture and optimizer candidate."""

    candidate_id: str
    recurrent_type: str
    sequence_length: int
    hidden_size: int
    layer_count: int
    recurrent_dropout: float
    head_hidden_size: int
    head_dropout: float
    batch_size: int
    learning_rate: float
    weight_decay: float


def recurrent_config_path(settings: Settings) -> Path:
    """Return the recurrent config without modifying frozen model settings."""

    return settings.root / RECURRENT_CONFIG_PATH


def _positive_int(record: dict[str, Any], name: str) -> int:
    value = int(record[name])
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _probability(record: dict[str, Any], name: str) -> float:
    value = float(record[name])
    if not 0.0 <= value < 1.0:
        raise ValueError(f"{name} must be in [0, 1)")
    return value


def _candidate(record: dict[str, Any]) -> RecurrentCandidate:
    recurrent_type = str(record["recurrent_type"]).lower()
    if recurrent_type not in SUPPORTED_RECURRENT_TYPES:
        raise ValueError(f"Unsupported recurrent type: {recurrent_type}")
    layer_count = _positive_int(record, "layer_count")
    recurrent_dropout = _probability(record, "recurrent_dropout")
    if layer_count == 1 and recurrent_dropout != 0.0:
        raise ValueError("Single-layer recurrent candidates require zero core dropout")
    learning_rate = float(record["learning_rate"])
    weight_decay = float(record["weight_decay"])
    if learning_rate <= 0.0 or weight_decay < 0.0:
        raise ValueError("Learning rate must be positive and weight decay non-negative")
    return RecurrentCandidate(
        candidate_id=str(record["candidate_id"]),
        recurrent_type=recurrent_type,
        sequence_length=_positive_int(record, "sequence_length"),
        hidden_size=_positive_int(record, "hidden_size"),
        layer_count=layer_count,
        recurrent_dropout=recurrent_dropout,
        head_hidden_size=_positive_int(record, "head_hidden_size"),
        head_dropout=_probability(record, "head_dropout"),
        batch_size=_positive_int(record, "batch_size"),
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )


def _validate_section(section: dict[str, Any]) -> list[RecurrentCandidate]:
    if section.get("contract_version") != "recurrent_volume_v1":
        raise ValueError("Unsupported recurrent-volume contract")
    upstream = section.get("upstream", {})
    required_upstream = {
        "modelling_version",
        "classical_regression_version",
        "classical_registry_version",
    }
    if set(upstream) != required_upstream:
        raise ValueError("Recurrent config must name every frozen upstream version")
    target = section.get("target", {})
    horizons = tuple(int(value) for value in target.get("horizons", []))
    if target.get("key") != "volume" or horizons != EXPECTED_HORIZONS:
        raise ValueError("Recurrent target must be volume at horizons 1-4")
    expected_columns = tuple(f"target_volume_h{value}" for value in EXPECTED_HORIZONS)
    if tuple(target.get("columns", [])) != expected_columns:
        raise ValueError("Recurrent target columns do not match horizons")
    sequence = section.get("sequence", {})
    if int(sequence.get("cadence_minutes", 0)) != 30:
        raise ValueError("Recurrent cadence must remain 30 minutes")
    if not all(
        bool(sequence.get(name))
        for name in (
            "require_contiguous",
            "forbid_cross_partition",
            "require_all_horizon_targets",
        )
    ):
        raise ValueError("All recurrent sequence isolation guards are mandatory")
    raw_candidates = section.get("candidates")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise ValueError("At least one recurrent candidate is required")
    candidates = [_candidate(record) for record in raw_candidates]
    identifiers = [candidate.candidate_id for candidate in candidates]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Recurrent candidate identifiers must be unique")
    training = section.get("training", {})
    if training.get("loss") != "mean_squared_error":
        raise ValueError("The primary recurrent loss must be mean squared error")
    if training.get("optimizer") != "adam":
        raise ValueError("The primary recurrent optimizer must be Adam")
    if training.get("target_scaling") != "per_horizon_standard":
        raise ValueError("Only training-fitted per-horizon scaling is supported")
    if int(training["minimum_epochs"]) > int(training["maximum_epochs"]):
        raise ValueError("Minimum epochs cannot exceed maximum epochs")
    selection = section.get("selection", {})
    if selection.get("primary_metric") != "validation_mean_rmse":
        raise ValueError("Recurrent selection must use validation mean RMSE")
    if selection.get("direction") != "minimize":
        raise ValueError("Recurrent selection direction must be minimize")
    if not selection.get("common_validation_origins"):
        raise ValueError("Candidates must share validation origins")
    if not selection.get("freeze_before_test"):
        raise ValueError("Recurrent selection must freeze before test access")
    return candidates


def load_recurrent_config(
    settings: Settings,
) -> tuple[dict[str, Any], list[RecurrentCandidate], Path]:
    """Load and validate the independent Step 15 YAML contract."""

    path = recurrent_config_path(settings)
    if not path.is_file():
        raise FileNotFoundError(f"Recurrent configuration is missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload: dict[str, Any] = yaml.safe_load(handle)
    section = payload.get("recurrent_volume")
    if not isinstance(section, dict):
        raise ValueError("recurrent.yaml must define recurrent_volume")
    candidates = _validate_section(section)
    return section, candidates, path
