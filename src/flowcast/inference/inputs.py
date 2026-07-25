"""Verified frozen inputs and routing context for Step 17 inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from flowcast.data.artifacts import artifact_record
from flowcast.evaluation.confidence_artifacts import (
    VerifiedConfidenceCalibration,
    load_verified_confidence_calibration,
)
from flowcast.features.inputs import VerifiedProcessedInput, load_verified_processed
from flowcast.inference.config import load_inference_config
from flowcast.modelling.inputs import (
    VerifiedModelingArtifacts,
    load_verified_modeling_artifacts,
)
from flowcast.modelling.registry_artifacts import (
    ClassicalRegistryPaths,
    classical_registry_paths,
    load_classical_registry,
)
from flowcast.settings import Settings


@dataclass(frozen=True)
class VerifiedInferenceContext:
    """All non-model inputs verified before a prediction request runs."""

    config: dict[str, Any]
    config_path: Any
    processed: VerifiedProcessedInput
    modeling: VerifiedModelingArtifacts
    registry: dict[str, Any]
    registry_summary: dict[str, Any]
    registry_paths: ClassicalRegistryPaths
    confidence: VerifiedConfidenceCalibration
    upstream_records: dict[str, dict[str, Any]]


def _validate_versions(
    settings: Settings,
    config: dict[str, Any],
    context: VerifiedInferenceContext,
) -> None:
    upstream = config["upstream"]
    expected = {
        "processed_version": settings.processed_version,
        "modelling_version": settings.modelling_version,
        "registry_version": context.registry["version"],
        "confidence_version": context.confidence.summary["version"],
    }
    for name, value in expected.items():
        if str(upstream[name]) != str(value):
            raise RuntimeError(f"Inference upstream version changed: {name}")
    recurrent = str(upstream["recurrent_version"])
    confidence_recurrent = str(
        context.confidence.config["upstream"]["recurrent_version"]
    )
    if recurrent != confidence_recurrent:
        raise RuntimeError("Inference and confidence recurrent versions differ")


def load_verified_inference_context(
    settings: Settings,
) -> VerifiedInferenceContext:
    """Verify configs, processed data, preprocessing, registry, and confidence."""

    config, config_path = load_inference_config(settings)
    processed = load_verified_processed(settings)
    modeling = load_verified_modeling_artifacts(settings)
    registry_version = str(config["upstream"]["registry_version"])
    registry, registry_summary = load_classical_registry(
        settings,
        version=registry_version,
    )
    paths = classical_registry_paths(settings, registry_version)
    confidence = load_verified_confidence_calibration(
        settings,
        version=str(config["upstream"]["confidence_version"]),
    )
    upstream_records = {
        "processed_summary": artifact_record(processed.summary_path, settings),
        "processed_manifest": artifact_record(processed.manifest_path, settings),
        "processed_dataset": artifact_record(processed.path, settings),
        "modelling_summary": artifact_record(modeling.summary_path, settings),
        "feature_schema": artifact_record(modeling.schema_path, settings),
        "classical_registry_summary": artifact_record(paths.summary_path, settings),
        "confidence_summary": artifact_record(
            settings.artifacts_dir
            / "metrics"
            / confidence.summary["version"]
            / "summary.json",
            settings,
        ),
    }
    context = VerifiedInferenceContext(
        config=config,
        config_path=config_path,
        processed=processed,
        modeling=modeling,
        registry=registry,
        registry_summary=registry_summary,
        registry_paths=paths,
        confidence=confidence,
        upstream_records=upstream_records,
    )
    _validate_versions(settings, config, context)
    return context
