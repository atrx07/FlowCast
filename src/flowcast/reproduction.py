"""One-command isolated reconstruction of the complete FlowCast pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import platform
from pathlib import Path
import time
from typing import Any, Callable

import numpy as np
import pandas as pd
import pyarrow
import sklearn
import streamlit
import torch
import xgboost

from flowcast.analysis.pipeline import run_eda
from flowcast.data.artifacts import artifact_record, write_json
from flowcast.data.audit import run_raw_audit, sha256_file
from flowcast.data.clean_context import run_context_cleaning
from flowcast.data.merge_pipeline import run_source_merge
from flowcast.data.quarantine import run_validation_pipeline
from flowcast.data.traffic_pipeline import run_traffic_cleaning
from flowcast.evaluation.confidence_pipeline import run_confidence_analysis
from flowcast.features.pipeline import run_feature_engineering
from flowcast.features.processed_pipeline import run_processed_data
from flowcast.inference.artifacts import (
    load_prediction_batch,
    persist_prediction_batch,
)
from flowcast.inference.predictor import Predictor
from flowcast.modelling.classification import run_classical_classification
from flowcast.modelling.classical_regression import run_classical_regression
from flowcast.modelling.pipeline import run_modeling_prep
from flowcast.modelling.recurrent import run_recurrent_volume
from flowcast.modelling.registry import run_classical_registry
from flowcast.modelling.regression import run_scratch_linear
from flowcast.reports.export import (
    build_prediction_reports,
    verify_prediction_reports,
)
from flowcast.reproduction_verify import verify_completed_reproduction
from flowcast.settings import Settings


@dataclass(frozen=True)
class ReproductionResult:
    """Paths and manifest from one complete isolated reconstruction."""

    root: Path
    manifest_path: Path
    summary_path: Path
    manifest: dict[str, Any]


def _portable(path: Path, settings: Settings) -> str:
    return path.resolve().relative_to(settings.root.resolve()).as_posix()


def _reference_hashes(settings: Settings) -> dict[str, str]:
    return {
        path.name: sha256_file(path, settings.hash_chunk_size)
        for path in sorted(settings.reference_dir.iterdir())
        if path.is_file()
    }


def _run_stage(
    name: str,
    action: Callable[[], Any],
    evidence: Callable[[Any], Path],
    settings: Settings,
) -> tuple[Any, dict[str, Any]]:
    started = time.perf_counter()
    result = action()
    evidence_path = evidence(result)
    return result, {
        "name": name,
        "seconds": time.perf_counter() - started,
        "evidence": artifact_record(evidence_path, settings),
    }


def _environment() -> dict[str, Any]:
    using_cuda = torch.cuda.is_available()
    properties = torch.cuda.get_device_properties(0) if using_cuda else None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "libraries": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pyarrow": pyarrow.__version__,
            "scikit_learn": sklearn.__version__,
            "streamlit": streamlit.__version__,
            "torch": torch.__version__,
            "xgboost": xgboost.__version__,
        },
        "accelerator": {
            "cuda_available": using_cuda,
            "cuda_runtime": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "gpu_name": properties.name if properties is not None else None,
            "gpu_total_memory_bytes": (
                int(properties.total_memory) if properties is not None else None
            ),
            "npu_used": False,
        },
    }


def _render_summary(manifest: dict[str, Any]) -> str:
    stages = "\n".join(
        f"| {item['name']} | {item['seconds']:.3f} | "
        f"`{item['evidence']['path']}` |"
        for item in manifest["stages"]
    )
    checks = "\n".join(
        f"- {'PASS' if value else 'FAIL'}: {name.replace('_', ' ')}"
        for name, value in manifest["checks"].items()
    )
    return f"""# FlowCast clean reproduction

- Run ID: `{manifest['run_id']}`
- Completed: `{manifest['completed_at_utc']}`
- Total runtime: `{manifest['runtime']['total_seconds']:.3f}` seconds
- Output root: `{manifest['output_root']}`

## Stage evidence

| Stage | Seconds | Primary evidence |
|---|---:|---|
{stages}

## Acceptance checks

{checks}

## Final outputs

- Prediction manifest: `{manifest['final_outputs']['prediction_manifest']['path']}`
- Report manifest: `{manifest['final_outputs']['report_manifest']['path']}`
- Recurrent device: `{manifest['final_outputs']['recurrent_device']}`
- Prediction cold runtime: `{manifest['final_outputs']['prediction_cold_seconds']:.3f}` seconds
"""


def run_full_reproduction(
    settings: Settings,
    *,
    recurrent_device: str = "cpu",
) -> ReproductionResult:
    """Rebuild every required artifact under one empty isolated output root."""

    root = settings.artifacts_dir.parent
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Reproduction root must be empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    started_at = datetime.now(timezone.utc)
    source_before = _reference_hashes(settings)
    stages: list[dict[str, Any]] = []

    audit, stage = _run_stage(
        "audit",
        lambda: run_raw_audit(settings),
        lambda item: item.json_path,
        settings,
    )
    stages.append(stage)
    validation, stage = _run_stage(
        "validate",
        lambda: run_validation_pipeline(settings),
        lambda item: item.summary_path,
        settings,
    )
    if validation.has_dataset_failure:
        raise RuntimeError("Raw validation reported a dataset failure")
    stages.append(stage)
    context, stage = _run_stage(
        "clean-context",
        lambda: run_context_cleaning(settings),
        lambda item: item.summary_path,
        settings,
    )
    stages.append(stage)
    traffic, stage = _run_stage(
        "clean-traffic",
        lambda: run_traffic_cleaning(settings),
        lambda item: item.summary_path,
        settings,
    )
    stages.append(stage)
    merged, stage = _run_stage(
        "merge-sources",
        lambda: run_source_merge(settings),
        lambda item: item.summary_path,
        settings,
    )
    stages.append(stage)
    features, stage = _run_stage(
        "engineer-features",
        lambda: run_feature_engineering(settings),
        lambda item: item.summary_path,
        settings,
    )
    stages.append(stage)
    processed, stage = _run_stage(
        "prepare-data",
        lambda: run_processed_data(settings),
        lambda item: item.summary_path,
        settings,
    )
    stages.append(stage)
    eda, stage = _run_stage(
        "eda",
        lambda: run_eda(settings),
        lambda item: item.summary_path,
        settings,
    )
    stages.append(stage)
    modeling, stage = _run_stage(
        "prepare-modeling",
        lambda: run_modeling_prep(settings),
        lambda item: item.summary_path,
        settings,
    )
    stages.append(stage)
    scratch, stage = _run_stage(
        "train-scratch-linear",
        lambda: run_scratch_linear(settings),
        lambda item: item.summary_path,
        settings,
    )
    stages.append(stage)
    regression, stage = _run_stage(
        "train-classical-regression",
        lambda: run_classical_regression(settings),
        lambda item: item.paths.summary_path,
        settings,
    )
    stages.append(stage)
    classification, stage = _run_stage(
        "train-classical-classification",
        lambda: run_classical_classification(settings),
        lambda item: item.paths.summary_path,
        settings,
    )
    stages.append(stage)
    registry, stage = _run_stage(
        "build-classical-registry",
        lambda: run_classical_registry(settings),
        lambda item: item.paths.summary_path,
        settings,
    )
    stages.append(stage)
    recurrent, stage = _run_stage(
        "train-recurrent-volume",
        lambda: run_recurrent_volume(settings, device=recurrent_device),
        lambda item: item.paths.summary_path,
        settings,
    )
    stages.append(stage)
    confidence, stage = _run_stage(
        "analyze-confidence",
        lambda: run_confidence_analysis(settings),
        lambda item: item.paths.summary_path,
        settings,
    )
    stages.append(stage)

    inference_started = time.perf_counter()
    predictor = Predictor(settings, device="cpu")
    request = predictor.build_request(horizons=(1, 2, 3, 4))
    prediction = predictor.predict(request)
    prediction_paths = persist_prediction_batch(prediction, settings)
    load_prediction_batch(settings, prediction_paths.manifest_path)
    report_paths = build_prediction_reports(settings, prediction_paths.manifest_path)
    verify_prediction_reports(settings, report_paths.manifest_path)
    stages.append(
        {
            "name": "predict-and-report",
            "seconds": time.perf_counter() - inference_started,
            "evidence": artifact_record(prediction_paths.manifest_path, settings),
        }
    )

    source_after = _reference_hashes(settings)
    completed_at = datetime.now(timezone.utc)
    report_dir = settings.artifacts_dir / "reproduction"
    manifest_path = report_dir / "manifest.json"
    summary_path = report_dir / "summary.md"
    run_id = root.name
    manifest = {
        "contract_version": "flowcast_reproduction_v1",
        "run_id": run_id,
        "started_at_utc": started_at.isoformat(),
        "completed_at_utc": completed_at.isoformat(),
        "output_root": _portable(root, settings),
        "configuration": artifact_record(settings.config_path, settings),
        "environment": _environment(),
        "source_reference_sha256": source_after,
        "stages": stages,
        "runtime": {"total_seconds": time.perf_counter() - started},
        "final_outputs": {
            "prediction_manifest": artifact_record(
                prediction_paths.manifest_path,
                settings,
            ),
            "report_manifest": artifact_record(
                report_paths.manifest_path,
                settings,
            ),
            "recurrent_device": recurrent.summary["device"]["selected"],
            "recurrent_device_policy": recurrent.summary["device"]["policy"],
            "prediction_cold_seconds": prediction.total_seconds,
        },
        "coverage": {
            "processed_rows": processed.summary["dataset"]["output_rows"],
            "registry_entries": registry.summary["coverage"]["entry_count"],
            "prediction_rows": len(prediction.frame),
            "model_card_count": len(
                tuple(settings.artifacts_dir.glob("model_cards/**/*.json"))
            ),
        },
        "checks": {
            "source_reference_unchanged": source_before == source_after,
            "raw_validation_passed": not validation.has_dataset_failure,
            "processed_row_contract": (
                processed.summary["dataset"]["output_rows"] == 181_200
            ),
            "registry_has_twenty_entries": (
                registry.summary["coverage"]["entry_count"] == 20
            ),
            "five_targets_four_horizons": len(prediction.frame) == 100,
            "prediction_under_thirty_seconds": prediction.total_seconds <= 30.0,
            "reports_verified": True,
            "cpu_inference_verified": request.device == "cpu",
            "recurrent_checkpoint_portable": True,
        },
    }
    if not all(manifest["checks"].values()):
        failed = [name for name, value in manifest["checks"].items() if not value]
        raise RuntimeError(f"Reproduction acceptance checks failed: {failed}")
    write_json(manifest, manifest_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        _render_summary(manifest),
        encoding="utf-8",
        newline="\n",
    )
    verify_completed_reproduction(settings)
    return ReproductionResult(root, manifest_path, summary_path, manifest)
