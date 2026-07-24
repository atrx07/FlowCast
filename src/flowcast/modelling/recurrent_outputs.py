"""Deterministic tables, comparison, and model metadata for Step 15."""

from __future__ import annotations

from dataclasses import asdict
import json
from typing import Any

import numpy as np
import pandas as pd

from flowcast.evaluation.regression import regression_metrics
from flowcast.modelling.recurrent_config import RecurrentCandidate
from flowcast.modelling.recurrent_training import (
    CandidateTrainingResult,
    horizon_metrics,
)
from flowcast.modelling.sequence_data import PreparedPartition


def candidate_frame(results: list[CandidateTrainingResult]) -> pd.DataFrame:
    """Flatten validation-led candidate selection evidence."""

    rows = []
    for result in results:
        rows.append(
            {
                **asdict(result.candidate),
                "parameter_count": result.architecture["parameter_count"],
                "best_epoch": result.best_epoch,
                "stopped_epoch": result.stopped_epoch,
                "early_stopped": result.early_stopped,
                "validation_mean_rmse": result.best_validation_mean_rmse,
                "fit_seconds": result.fit_seconds,
                "prediction_seconds": result.prediction_seconds,
                "device": result.device,
            }
        )
    return pd.DataFrame(rows)


def curves_frame(results: list[CandidateTrainingResult]) -> pd.DataFrame:
    """Combine every candidate training/validation curve."""

    return pd.DataFrame(
        [record for result in results for record in result.history]
    )


def choose_candidate(
    results: list[CandidateTrainingResult],
) -> CandidateTrainingResult:
    """Select validation mean RMSE, then size and identifier deterministically."""

    if not results:
        raise ValueError("At least one trained recurrent candidate is required")
    return min(
        results,
        key=lambda result: (
            result.best_validation_mean_rmse,
            int(result.architecture["parameter_count"]),
            result.candidate.candidate_id,
        ),
    )


def prediction_frame(
    partition: PreparedPartition,
    endpoints: np.ndarray,
    predictions: np.ndarray,
    split: str,
    candidate: RecurrentCandidate,
    version: str,
) -> pd.DataFrame:
    """Create long-form, horizon-traceable recurrent predictions."""

    origin = partition.frame.iloc[np.asarray(endpoints, dtype=np.int64)].reset_index(
        drop=True
    )
    if predictions.shape != (len(origin), 4):
        raise ValueError("Recurrent predictions must have four aligned horizons")
    frames = []
    for horizon, target_column in enumerate(partition.target_columns, start=1):
        actual = origin[target_column].to_numpy(dtype=float)
        estimate = predictions[:, horizon - 1]
        frames.append(
            pd.DataFrame(
                {
                    "road_id": origin["road_id"].astype(str),
                    "timestamp": origin["timestamp"],
                    "target_timestamp": origin[f"target_timestamp_h{horizon}"],
                    "split": split,
                    "job_id": f"volume_h{horizon}",
                    "target": "volume",
                    "target_column": target_column,
                    "horizon_windows": horizon,
                    "horizon_minutes": horizon * 30,
                    "actual": actual,
                    "prediction": estimate,
                    "residual": actual - estimate,
                    "selected_family": candidate.recurrent_type,
                    "candidate_id": candidate.candidate_id,
                    "model_version": version,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def metric_frame(
    validation_actual: np.ndarray,
    validation_predictions: np.ndarray,
    test_actual: np.ndarray,
    test_predictions: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build split/horizon metrics and aggregate selection values."""

    validation, validation_mean = horizon_metrics(
        validation_actual,
        validation_predictions,
    )
    test, test_mean = horizon_metrics(test_actual, test_predictions)
    rows = []
    for split, records in (("validation", validation), ("test", test)):
        rows.extend({"split": split, **record} for record in records)
    return pd.DataFrame(rows), {
        "validation": {
            "mean_rmse": validation_mean,
            "horizons": validation,
        },
        "test": {
            "mean_rmse": test_mean,
            "horizons": test,
        },
    }


def compare_with_classical(
    deep_test: pd.DataFrame,
    classical_predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Compare deep and classical predictions on exactly the same origin rows."""

    rows = []
    records = []
    key = ["road_id", "timestamp", "horizon_windows"]
    for horizon in range(1, 5):
        deep = deep_test.loc[
            deep_test["horizon_windows"].eq(horizon),
            key + ["target_timestamp", "actual", "prediction"],
        ].rename(columns={"prediction": "deep_prediction"})
        classical = classical_predictions.loc[
            classical_predictions["split"].eq("test")
            & classical_predictions["job_id"].eq(f"volume_h{horizon}"),
            key + ["target_timestamp", "actual", "prediction"],
        ].rename(
            columns={
                "target_timestamp": "classical_target_timestamp",
                "actual": "classical_actual",
                "prediction": "classical_prediction",
            }
        )
        if deep.duplicated(key).any() or classical.duplicated(key).any():
            raise RuntimeError("Deep/classical comparison keys are not unique")
        merged = deep.merge(classical, on=key, how="left", validate="one_to_one")
        if merged["classical_prediction"].isna().any():
            raise RuntimeError("A deep test origin lacks a classical prediction")
        if not merged["target_timestamp"].equals(
            merged["classical_target_timestamp"]
        ):
            raise RuntimeError("Deep/classical target timestamps do not align")
        if not np.allclose(
            merged["actual"],
            merged["classical_actual"],
            rtol=0.0,
            atol=0.0,
        ):
            raise RuntimeError("Deep/classical actual values do not align")
        deep_metrics = regression_metrics(
            merged["actual"],
            merged["deep_prediction"],
        )
        classical_metrics = regression_metrics(
            merged["actual"],
            merged["classical_prediction"],
        )
        beats = bool(deep_metrics["rmse"] < classical_metrics["rmse"])
        record = {
            "horizon_windows": horizon,
            "horizon_minutes": horizon * 30,
            "rows": len(merged),
            "deep_rmse": deep_metrics["rmse"],
            "classical_rmse": classical_metrics["rmse"],
            "rmse_delta_deep_minus_classical": round(
                float(deep_metrics["rmse"] - classical_metrics["rmse"]),
                10,
            ),
            "deep_beats_classical": beats,
            "origin_mapping_complete": True,
            "actual_values_identical": True,
            "target_timestamps_identical": True,
        }
        rows.append(record)
        records.append(
            {
                **record,
                "deep_metrics": deep_metrics,
                "classical_metrics": classical_metrics,
            }
        )
    return pd.DataFrame(rows), records


def registry_extension(
    version: str,
    candidate: RecurrentCandidate,
    metrics: dict[str, Any],
    artifacts: dict[str, Any],
) -> dict[str, Any]:
    """Create four horizon entries for the shared multi-output checkpoint."""

    entries = []
    for record in metrics["test"]["horizons"]:
        horizon = int(record["horizon_windows"])
        entries.append(
            {
                "registry_key": (
                    f"volume/h{horizon}/{candidate.recurrent_type}/{version}"
                ),
                "job_id": f"volume_h{horizon}",
                "target": "volume",
                "horizon_windows": horizon,
                "horizon_minutes": horizon * 30,
                "family": candidate.recurrent_type,
                "model_version": version,
                "shared_multi_horizon_checkpoint": True,
                "primary_metric": "rmse",
                "test_metrics": record,
                "artifacts": artifacts,
            }
        )
    return {
        "contract_version": "recurrent_registry_extension_v1",
        "version": version,
        "entry_count": 4,
        "entries": entries,
    }


def model_card(
    *,
    version: str,
    seed: int,
    selected: CandidateTrainingResult,
    metrics: dict[str, Any],
    comparison: list[dict[str, Any]],
    sequences: dict[str, Any],
    feature_manifest: dict[str, Any],
    scaler: dict[str, Any],
    split_summary: dict[str, Any],
    lineage: dict[str, Any],
    artifacts: dict[str, Any],
) -> dict[str, Any]:
    """Build the complete recurrent model card after frozen test evaluation."""

    return {
        "contract_version": "flowcast_recurrent_model_card_v1",
        "job_id": "volume_multi_horizon",
        "model_version": version,
        "seed": seed,
        "target": {
            "key": "volume",
            "columns": [f"target_volume_h{value}" for value in range(1, 5)],
            "horizons_windows": [1, 2, 3, 4],
            "horizons_minutes": [30, 60, 90, 120],
        },
        "selection": {
            "candidate_id": selected.candidate.candidate_id,
            "primary_metric": "validation_mean_rmse",
            "validation_mean_rmse": selected.best_validation_mean_rmse,
            "best_epoch": selected.best_epoch,
            "stopped_epoch": selected.stopped_epoch,
            "early_stopped": selected.early_stopped,
            "architecture": selected.architecture,
            "test_metrics_used": False,
        },
        "data": split_summary,
        "sequences": sequences,
        "features": feature_manifest,
        "target_scaling": scaler,
        "metrics": metrics,
        "classical_comparison": comparison,
        "lineage": lineage,
        "artifacts": artifacts,
        "limitations": [
            "The model forecasts volume only; confidence intervals arrive in Step 16.",
            "Observed origin weather is used; future weather forecasts are unavailable.",
            "Sequence isolation removes the first sequence_length-1 origins per road "
            "from each partition, so classical comparison is restricted to the exact "
            "deep-model origin subset.",
            "The installed PyTorch 2.13.0 build is CPU-only on this workstation.",
        ],
    }
