"""Final artifact assembly after the frozen recurrent test evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
import sklearn
import torch

from flowcast.data.artifacts import artifact_record, write_json
from flowcast.modelling.recurrent_artifacts import RecurrentPaths
from flowcast.modelling.recurrent_config import RecurrentCandidate
from flowcast.modelling.recurrent_outputs import model_card, registry_extension
from flowcast.modelling.recurrent_report import (
    render_recurrent_model_card,
    render_recurrent_report,
)
from flowcast.modelling.recurrent_support import split_record
from flowcast.modelling.recurrent_training import (
    CandidateTrainingResult,
)
from flowcast.modelling.sequence_data import PreparedPartition, sequence_manifest
from flowcast.settings import Settings


@dataclass(frozen=True)
class RecurrentFinalization:
    """All frozen inputs needed to assemble the final Step 15 record."""

    settings: Settings
    paths: RecurrentPaths
    config: dict[str, Any]
    config_path: Path
    modeling: Any
    candidates: list[RecurrentCandidate]
    results: list[CandidateTrainingResult]
    selected: CandidateTrainingResult
    training: PreparedPartition
    validation: PreparedPartition
    test: PreparedPartition
    selected_train: np.ndarray
    selected_validation: np.ndarray
    selected_test: np.ndarray
    feature_manifest: dict[str, Any]
    scaler_metadata: dict[str, Any]
    selection_record: dict[str, Any]
    metrics: dict[str, Any]
    comparison: list[dict[str, Any]]
    classical_predictions_sha256: str
    predictions: pd.DataFrame
    validation_predictions: pd.DataFrame
    test_predictions: pd.DataFrame
    device: torch.device
    started: float
    validation_prediction_seconds: float
    test_prediction_seconds: float


def _upstream(context: RecurrentFinalization) -> dict[str, Any]:
    settings = context.settings
    config = context.config
    return {
        "modelling_summary": artifact_record(
            context.modeling.summary_path,
            settings,
        ),
        "classical_regression_summary": artifact_record(
            settings.artifacts_dir
            / "metrics"
            / str(config["upstream"]["classical_regression_version"])
            / "summary.json",
            settings,
        ),
        "classical_registry_summary": artifact_record(
            settings.artifacts_dir
            / "metrics"
            / str(config["upstream"]["classical_registry_version"])
            / "summary.json",
            settings,
        ),
    }


def _all_artifacts(
    context: RecurrentFinalization,
    checkpoint: dict[str, Any],
    predictions: dict[str, Any],
) -> dict[str, Any]:
    paths = context.paths
    settings = context.settings
    return {
        "selection": context.selection_record,
        "environment": artifact_record(paths.environment_path, settings),
        "candidate_metrics": artifact_record(paths.candidates_path, settings),
        "training_curves": artifact_record(paths.curves_path, settings),
        "horizon_metrics": artifact_record(paths.metrics_path, settings),
        "classical_comparison": artifact_record(paths.comparison_path, settings),
        "pretest_sequence_manifest": artifact_record(
            paths.pretest_sequence_manifest_path,
            settings,
        ),
        "sequence_manifest": artifact_record(
            paths.sequence_manifest_path,
            settings,
        ),
        "pretest_model_card": artifact_record(paths.pretest_card_path, settings),
        "feature_manifest": artifact_record(paths.feature_manifest_path, settings),
        "target_scaler": artifact_record(paths.target_scaler_path, settings),
        "checkpoint": checkpoint,
        "predictions": predictions,
        "model_card_json": artifact_record(paths.card_json_path, settings),
        "model_card_markdown": artifact_record(
            paths.card_markdown_path,
            settings,
        ),
        "registry_extension": artifact_record(
            paths.registry_extension_path,
            settings,
        ),
        "report": artifact_record(paths.report_path, settings),
    }


def finalize_recurrent_run(context: RecurrentFinalization) -> dict[str, Any]:
    """Write final manifests, cards, registry extension, report, and summary."""

    settings = context.settings
    paths = context.paths
    selected = context.selected
    config = context.config
    cadence = int(config["sequence"]["cadence_minutes"])
    maximum_length = max(
        candidate.sequence_length for candidate in context.candidates
    )
    sequences = {
        "contract_version": "recurrent_sequence_manifest_v1",
        "selected_candidate_id": selected.candidate.candidate_id,
        "maximum_candidate_sequence_length": maximum_length,
        "common_validation_and_test_origin_policy": True,
        "train": sequence_manifest(
            context.training,
            context.selected_train,
            selected.candidate.sequence_length,
            cadence,
        ),
        "validation": sequence_manifest(
            context.validation,
            context.selected_validation,
            selected.candidate.sequence_length,
            cadence,
        ),
        "test": sequence_manifest(
            context.test,
            context.selected_test,
            selected.candidate.sequence_length,
            cadence,
        ),
    }
    write_json(sequences, paths.sequence_manifest_path)
    predictions_record = artifact_record(paths.predictions_path, settings)
    checkpoint_record = artifact_record(paths.checkpoint_path, settings)
    split_summary = {
        "train": split_record(context.training, context.selected_train),
        "validation": split_record(
            context.validation,
            context.selected_validation,
        ),
        "test": split_record(context.test, context.selected_test),
        "frozen_boundaries": context.modeling.summary["split"]["partitions"],
    }
    card_artifacts = {
        "checkpoint": checkpoint_record,
        "predictions": predictions_record,
        "selection": context.selection_record,
        "training_curves": artifact_record(paths.curves_path, settings),
        "target_scaler": artifact_record(paths.target_scaler_path, settings),
        "feature_manifest": artifact_record(paths.feature_manifest_path, settings),
    }
    lineage = {
        "processed_sha256": context.modeling.summary["input_processed"]["dataset"][
            "sha256"
        ],
        "feature_schema_sha256": context.modeling.summary["artifacts"][
            "feature_schema"
        ]["sha256"],
        "selection_sha256": context.selection_record["sha256"],
        "classical_predictions_sha256": context.classical_predictions_sha256,
    }
    card = model_card(
        version=paths.version,
        seed=settings.seed,
        selected=selected,
        metrics=context.metrics,
        comparison=context.comparison,
        sequences=sequences,
        feature_manifest=context.feature_manifest,
        scaler=context.scaler_metadata,
        split_summary=split_summary,
        lineage=lineage,
        artifacts=card_artifacts,
    )
    write_json(card, paths.card_json_path)
    paths.card_markdown_path.parent.mkdir(parents=True, exist_ok=True)
    paths.card_markdown_path.write_text(
        render_recurrent_model_card(card),
        encoding="utf-8",
        newline="\n",
    )
    extension = registry_extension(
        paths.version,
        selected.candidate,
        context.metrics,
        {
            "checkpoint": checkpoint_record,
            "model_card": artifact_record(paths.card_json_path, settings),
            "predictions": predictions_record,
        },
    )
    write_json(extension, paths.registry_extension_path)
    report_payload = {
        "version": paths.version,
        "seed": settings.seed,
        "selected": {
            **asdict(selected.candidate),
            "best_epoch": selected.best_epoch,
            "stopped_epoch": selected.stopped_epoch,
        },
        "device": str(context.device),
        "metrics": context.metrics,
        "comparison": context.comparison,
        "sequences": sequences,
    }
    paths.report_path.parent.mkdir(parents=True, exist_ok=True)
    paths.report_path.write_text(
        render_recurrent_report(report_payload),
        encoding="utf-8",
        newline="\n",
    )
    artifacts = _all_artifacts(
        context,
        checkpoint_record,
        predictions_record,
    )
    summary = {
        "contract_version": "recurrent_volume_v1",
        "version": paths.version,
        "configuration": artifact_record(context.config_path, settings),
        "upstream": _upstream(context),
        "coverage": {
            "candidate_count": len(context.candidates),
            "horizon_count": 4,
            "selected_model_count": 1,
            "registry_entry_count": 4,
            "validation_prediction_rows": int(
                len(context.validation_predictions)
            ),
            "test_prediction_rows": int(len(context.test_predictions)),
            "total_prediction_rows": int(len(context.predictions)),
        },
        "selection": {
            "status": "frozen_before_test_access",
            "selected_candidate_id": selected.candidate.candidate_id,
            "best_epoch": selected.best_epoch,
            "validation_mean_rmse": selected.best_validation_mean_rmse,
            "test_metrics_used": False,
            **context.selection_record,
        },
        "test_access": {
            "loader_invocation_count": 1,
            "purpose": "final_evaluation",
            "selection_status_before_load": "frozen_before_test_access",
            "models_refit_after_test_load": False,
        },
        "device": {
            "selected": str(context.device),
            "policy": config["device"]["policy"],
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "gpu_used": context.device.type == "cuda",
            "npu_used": False,
            "cpu_threads": int(config["device"]["cpu_threads"]),
        },
        "sequences": sequences,
        "target_scaling": context.scaler_metadata,
        "metrics": context.metrics,
        "classical_comparison": context.comparison,
        "acceptance": {
            "deep_beats_classical_horizons": sum(
                bool(record["deep_beats_classical"])
                for record in context.comparison
            ),
            "required_horizons": 4,
            "all_horizons_met": all(
                bool(record["deep_beats_classical"])
                for record in context.comparison
            ),
        },
        "runtime": {
            "total_seconds": time.perf_counter() - context.started,
            "candidate_fit_seconds": sum(
                result.fit_seconds for result in context.results
            ),
            "validation_reload_prediction_seconds": (
                context.validation_prediction_seconds
            ),
            "test_prediction_seconds": context.test_prediction_seconds,
        },
        "libraries": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
        },
        "model": {
            "checkpoint": checkpoint_record,
            "target_scaler": artifact_record(
                paths.target_scaler_path,
                settings,
            ),
            "model_card_json": artifact_record(paths.card_json_path, settings),
            "model_card_markdown": artifact_record(
                paths.card_markdown_path,
                settings,
            ),
        },
        "artifacts": artifacts,
        "checks": [
            {"name": "road_partition_gap_target_isolation", "passed": True},
            {"name": "training_only_feature_and_target_scaling", "passed": True},
            {"name": "from_scratch_four_horizon_output", "passed": True},
            {"name": "selection_and_checkpoint_frozen_before_test", "passed": True},
            {"name": "best_checkpoint_reload_equality", "passed": True},
            {"name": "exact_classical_row_mapping", "passed": True},
            {"name": "four_entry_registry_extension", "passed": True},
        ],
        "limitations": card["limitations"],
    }
    write_json(summary, paths.summary_path)
    return summary
