"""Integrity and frozen-metric verification for a completed reproduction."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from flowcast.data.artifacts import verify_artifact_record, write_json
from flowcast.data.audit import sha256_file
from flowcast.inference.artifacts import load_prediction_batch
from flowcast.modelling.registry_artifacts import record_path
from flowcast.reports.export import verify_prediction_reports
from flowcast.settings import Settings


NUMERIC_TOLERANCE = 1.0e-12
RUNTIME_FIELDS = {"fit_seconds", "prediction_seconds"}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required reproduction evidence is missing: {path}")
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return payload


def _semantic(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _semantic(item)
            for key, item in value.items()
            if key not in RUNTIME_FIELDS
        }
    if isinstance(value, list):
        return [_semantic(item) for item in value]
    return value


def _compare(left: Any, right: Any) -> tuple[bool, float]:
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            return False, math.inf
        results = [_compare(left[key], right[key]) for key in left]
    elif isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return False, math.inf
        results = [_compare(one, two) for one, two in zip(left, right, strict=True)]
    elif (
        isinstance(left, (int, float))
        and isinstance(right, (int, float))
        and not isinstance(left, bool)
        and not isinstance(right, bool)
    ):
        delta = abs(float(left) - float(right))
        return math.isclose(
            float(left),
            float(right),
            rel_tol=NUMERIC_TOLERANCE,
            abs_tol=NUMERIC_TOLERANCE,
        ), delta
    else:
        return left == right, 0.0 if left == right else math.inf
    return all(item[0] for item in results), max(
        (item[1] for item in results),
        default=0.0,
    )


def _metric_sections(settings: Settings) -> dict[str, tuple[Any, Any]]:
    canonical = settings.root / "artifacts" / "metrics"
    reproduced = settings.artifacts_dir / "metrics"
    regression_c = _read_json(canonical / "classical_regression_v1/summary.json")
    regression_r = _read_json(reproduced / "classical_regression_v1/summary.json")
    classification_c = _read_json(
        canonical / "classical_classification_v1/summary.json"
    )
    classification_r = _read_json(
        reproduced / "classical_classification_v1/summary.json"
    )
    registry_c = _read_json(canonical / "classical_registry_v1/summary.json")
    registry_r = _read_json(reproduced / "classical_registry_v1/summary.json")
    recurrent_c = _read_json(canonical / "recurrent_volume_v1/summary.json")
    recurrent_r = _read_json(reproduced / "recurrent_volume_v1/summary.json")
    confidence_c = _read_json(canonical / "confidence_error_v1/summary.json")
    confidence_r = _read_json(reproduced / "confidence_error_v1/summary.json")
    selection_keys = (
        "selected_candidate_id",
        "best_epoch",
        "validation_mean_rmse",
    )
    return {
        "classical_regression": (
            _semantic(regression_c["scoreboard"]),
            _semantic(regression_r["scoreboard"]),
        ),
        "classical_classification": (
            _semantic(classification_c["scoreboard"]),
            _semantic(classification_r["scoreboard"]),
        ),
        "classical_registry_acceptance": (
            registry_c["acceptance"],
            registry_r["acceptance"],
        ),
        "recurrent_selection": (
            {key: recurrent_c["selection"][key] for key in selection_keys},
            {key: recurrent_r["selection"][key] for key in selection_keys},
        ),
        "recurrent_metrics": (
            recurrent_c["metrics"],
            recurrent_r["metrics"],
        ),
        "recurrent_classical_comparison": (
            recurrent_c["classical_comparison"],
            recurrent_r["classical_comparison"],
        ),
        "confidence_diagnostics": (
            confidence_c["diagnostics"],
            confidence_r["diagnostics"],
        ),
        "confidence_coverage": (
            confidence_c["coverage"],
            confidence_r["coverage"],
        ),
    }


def verify_reported_metrics(settings: Settings) -> dict[str, Any]:
    """Compare reproduced decisions and metrics with frozen canonical evidence."""

    checks: dict[str, bool] = {}
    maximum_delta = 0.0
    for name, (canonical, reproduced) in _metric_sections(settings).items():
        passed, delta = _compare(canonical, reproduced)
        checks[name] = passed
        maximum_delta = max(maximum_delta, delta)
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "numeric_tolerance": NUMERIC_TOLERANCE,
        "maximum_numeric_delta": maximum_delta,
    }


def verify_completed_reproduction(settings: Settings) -> dict[str, Any]:
    """Verify a run manifest, every primary stage record, and final outputs."""

    evidence_dir = settings.artifacts_dir / "reproduction"
    manifest_path = evidence_dir / "manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("contract_version") != "flowcast_reproduction_v1":
        raise RuntimeError("Unsupported reproduction manifest contract")
    if not all(bool(value) for value in manifest["checks"].values()):
        raise RuntimeError("The reproduction manifest contains a failed check")
    verify_artifact_record(
        settings.config_path,
        manifest["configuration"],
        settings,
    )
    for stage in manifest["stages"]:
        record = stage["evidence"]
        verify_artifact_record(record_path(record, settings), record, settings)
    current_sources = {
        path.name: sha256_file(path, settings.hash_chunk_size)
        for path in sorted(settings.reference_dir.iterdir())
        if path.is_file()
    }
    if current_sources != manifest["source_reference_sha256"]:
        raise RuntimeError("Reference source hashes changed after reproduction")
    prediction_record = manifest["final_outputs"]["prediction_manifest"]
    report_record = manifest["final_outputs"]["report_manifest"]
    prediction_path = verify_artifact_record(
        record_path(prediction_record, settings),
        prediction_record,
        settings,
    )
    report_path = verify_artifact_record(
        record_path(report_record, settings),
        report_record,
        settings,
    )
    load_prediction_batch(settings, prediction_path)
    verify_prediction_reports(settings, report_path)
    metrics = verify_reported_metrics(settings)
    if not metrics["passed"]:
        failed = [name for name, value in metrics["checks"].items() if not value]
        raise RuntimeError(f"Reported metrics failed to reconcile: {failed}")
    result = {
        "contract_version": "flowcast_reproduction_verification_v1",
        "run_id": manifest["run_id"],
        "manifest": {
            "path": manifest_path.resolve().relative_to(settings.root).as_posix(),
            "sha256": sha256_file(manifest_path, settings.hash_chunk_size),
        },
        "stage_record_count": len(manifest["stages"]),
        "source_reference_unchanged": True,
        "prediction_and_report_lineage_verified": True,
        "reported_metrics": metrics,
        "passed": True,
    }
    write_json(result, evidence_dir / "verification.json")
    return result
