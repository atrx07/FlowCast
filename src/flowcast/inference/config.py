"""Independent configuration contract for inference and report generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from flowcast.settings import Settings


INFERENCE_CONFIG_PATH = Path("config/inference.yaml")
EXPECTED_HORIZONS = (1, 2, 3, 4)
REQUIRED_UPSTREAM = {
    "processed_version",
    "modelling_version",
    "registry_version",
    "recurrent_version",
    "confidence_version",
}
CLASSICAL_TARGETS = ("speed", "travel_time", "congestion", "accident")


def inference_config_path(settings: Settings) -> Path:
    """Return the standalone Step 17 configuration path."""

    return settings.root / INFERENCE_CONFIG_PATH


def _validate_routing(section: dict[str, Any]) -> None:
    routing = section.get("active_routing", {})
    if set(routing) != {"volume", *CLASSICAL_TARGETS}:
        raise ValueError("Inference routing must define all five forecast targets")
    volume = routing["volume"]
    if volume.get("source") != "recurrent":
        raise ValueError("Active volume routing must use the recurrent model")
    if volume.get("selection_basis") != "validation_mean_rmse_all_horizons":
        raise ValueError("Volume routing must be frozen from validation evidence")
    fallback = volume.get("fallback", {})
    if fallback.get("source") != "classical_registry":
        raise ValueError("Volume fallback must resolve through the registry")
    if fallback.get("expose_comparator") is not True:
        raise ValueError("Classical volume comparator must remain exposed")
    for target in CLASSICAL_TARGETS:
        if routing[target].get("source") != "classical_registry":
            raise ValueError(f"{target} must resolve through the classical registry")


def _validate(section: dict[str, Any]) -> None:
    if section.get("contract_version") != "inference_reporting_v1":
        raise ValueError("Unsupported inference/reporting contract")
    if section.get("version") != "inference_reporting_v1":
        raise ValueError("Unsupported inference/reporting version")
    if set(section.get("upstream", {})) != REQUIRED_UPSTREAM:
        raise ValueError("Inference config must freeze every upstream version")
    _validate_routing(section)

    request = section.get("request", {})
    horizons = tuple(int(value) for value in request.get("horizons", []))
    if horizons != EXPECTED_HORIZONS:
        raise ValueError(f"Inference horizons must be {EXPECTED_HORIZONS}")
    if int(request.get("cadence_minutes", 0)) != 30:
        raise ValueError("Inference cadence must remain 30 minutes")
    if int(request.get("recurrent_sequence_length", 0)) <= 0:
        raise ValueError("Recurrent sequence length must be positive")
    if int(request.get("maximum_roads", 0)) != 25:
        raise ValueError("Inference maximum_roads must match the 25-road corridor")
    if request.get("default_scope") != "full_corridor":
        raise ValueError("The default inference scope must be the full corridor")

    device = section.get("device", {})
    allowed = tuple(str(value) for value in device.get("allowed", []))
    if allowed != ("cpu", "cuda") or device.get("default") != "cpu":
        raise ValueError("Inference must default to CPU with guarded CUDA support")
    if int(device.get("cpu_threads", 0)) <= 0:
        raise ValueError("Inference CPU thread count must be positive")

    output = section.get("output", {})
    if output.get("schema_version") != "flowcast_prediction_v1":
        raise ValueError("Unsupported prediction output schema")
    if tuple(output.get("prediction_formats", [])) != ("parquet", "json"):
        raise ValueError("Prediction outputs must be Parquet plus JSON")
    if tuple(output.get("report_formats", [])) != ("csv", "html"):
        raise ValueError("Reports must support CSV and HTML")
    if float(output.get("full_corridor_runtime_target_seconds", 0.0)) <= 0.0:
        raise ValueError("Inference runtime target must be positive")


def load_inference_config(
    settings: Settings,
) -> tuple[dict[str, Any], Path]:
    """Load and validate the independent Step 17 YAML contract."""

    path = inference_config_path(settings)
    if not path.is_file():
        raise FileNotFoundError(f"Inference configuration is missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload: dict[str, Any] = yaml.safe_load(handle)
    section = payload.get("inference_reporting")
    if not isinstance(section, dict):
        raise ValueError("inference.yaml must define inference_reporting")
    _validate(section)
    return section, path
