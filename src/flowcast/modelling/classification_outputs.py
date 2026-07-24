"""Tabular outputs, predictions, confusion records, and Step 13 model cards."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from flowcast.modelling.classifier_models import (
    ENCODED_TARGET,
    ClassificationJob,
)
from flowcast.settings import Settings


def classification_prediction_frame(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    job: ClassificationJob,
    split: str,
    selection: dict[str, Any],
    version: str,
) -> pd.DataFrame:
    """Build traceable ordered probability and label predictions."""

    result = frame[
        ["road_id", "timestamp", job.target_timestamp_column]
    ].copy()
    result = result.rename(
        columns={job.target_timestamp_column: "target_timestamp"}
    )
    actual = frame[ENCODED_TARGET].to_numpy(dtype=np.int64)
    if job.task == "accident":
        predicted = (
            probabilities[:, 1] >= float(selection["operating_threshold"])
        ).astype(np.int64)
    else:
        predicted = np.argmax(probabilities, axis=1).astype(np.int64)
    result["split"] = split
    result["job_id"] = job.job_id
    result["task"] = job.task
    result["target_column"] = job.target_column
    result["horizon_windows"] = job.horizon_windows
    result["horizon_minutes"] = job.horizon_minutes
    result["actual_class_index"] = actual
    result["predicted_class_index"] = predicted
    result["actual_label"] = [job.class_names[index] for index in actual]
    result["predicted_label"] = [job.class_names[index] for index in predicted]
    for index, name in enumerate(job.class_names):
        slug = name.lower().replace("-", "_").replace(" ", "_")
        result[f"probability_{slug}"] = probabilities[:, index]
    result["selected_family"] = selection["family"]
    result["candidate_id"] = selection["candidate_id"]
    result["calibration_applied"] = bool(selection["calibration"]["applied"])
    result["operating_threshold"] = selection["operating_threshold"]
    result["model_version"] = version
    return result


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def fold_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Flatten fold evidence while retaining nested audit data as JSON."""

    nested = {"confusion_matrix", "per_class", "training_weight_evidence"}
    rows = []
    for record in records:
        row = {key: value for key, value in record.items() if key not in nested}
        for key in nested & record.keys():
            row[f"{key}_json"] = _json(record[key])
        rows.append(row)
    return pd.DataFrame(rows)


def candidate_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Flatten classifier candidate aggregates."""

    rows = []
    for record in records:
        row = {key: value for key, value in record.items() if key != "parameters"}
        row["parameters_json"] = _json(record["parameters"])
        rows.append(row)
    return pd.DataFrame(rows)


def family_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Flatten family-validation comparisons."""

    excluded = {"parameters", "validation", "training_weight_evidence"}
    rows = []
    for record in records:
        row = {key: value for key, value in record.items() if key not in excluded}
        row["parameters_json"] = _json(record["parameters"])
        row["training_weight_evidence_json"] = _json(
            record["training_weight_evidence"]
        )
        for key, value in record["validation"].items():
            if isinstance(value, (dict, list)):
                row[f"validation_{key}_json"] = _json(value)
            else:
                row[f"validation_{key}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def scoreboard_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Flatten final validation and test classifier metrics."""

    rows = []
    for record in records:
        row = {
            key: value
            for key, value in record.items()
            if key not in {"validation", "test", "calibration"}
        }
        row["calibration_json"] = _json(record["calibration"])
        for split in ("validation", "test"):
            for key, value in record[split].items():
                if isinstance(value, (dict, list)):
                    row[f"{split}_{key}_json"] = _json(value)
                else:
                    row[f"{split}_{key}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def calibration_frame(selections: list[dict[str, Any]]) -> pd.DataFrame:
    """Persist one inspectable probability-calibration decision per job."""

    rows = []
    for selection in selections:
        calibration = selection["calibration"]
        raw = calibration["raw_quality"] or {}
        calibrated = calibration["calibrated_quality"]
        rows.append(
            {
                "job_id": selection["job_id"],
                "task": selection["task"],
                "method": calibration["method"],
                "fit_rows": calibration["fit_rows"],
                "fit_start": calibration["fit_start"],
                "fit_end": calibration["fit_end"],
                "assessment_rows": calibration["assessment_rows"],
                "assessment_start": calibration["assessment_start"],
                "assessment_end": calibration["assessment_end"],
                "raw_probability_available": calibration[
                    "raw_probability_available"
                ],
                "raw_brier_score": raw.get("brier_score"),
                "raw_log_loss": raw.get("log_loss"),
                "calibrated_brier_score": calibrated["brier_score"],
                "calibrated_log_loss": calibrated["log_loss"],
                "brier_improvement": calibration["brier_improvement"],
                "minimum_improvement": calibration["minimum_improvement"],
                "applied": calibration["applied"],
                "reason": calibration["reason"],
            }
        )
    return pd.DataFrame(rows)


def confusion_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Expand ordered validation/test confusion matrices into long form."""

    rows = []
    for record in records:
        names = list(record["class_order"])
        for split in ("validation", "test"):
            matrix = record[split]["confusion_matrix"]
            for actual_index, actual_name in enumerate(names):
                for predicted_index, predicted_name in enumerate(names):
                    rows.append(
                        {
                            "job_id": record["job_id"],
                            "task": record["task"],
                            "horizon_windows": record["horizon_windows"],
                            "split": split,
                            "actual_index": actual_index,
                            "actual_label": actual_name,
                            "predicted_index": predicted_index,
                            "predicted_label": predicted_name,
                            "rows": int(matrix[actual_index][predicted_index]),
                        }
                    )
    return pd.DataFrame(rows)


def build_classification_model_card(
    selection: dict[str, Any],
    score: dict[str, Any],
    job: ClassificationJob,
    modeling: Any,
    selection_record: dict[str, Any],
    predictions_record: dict[str, Any],
    settings: Settings,
) -> dict[str, Any]:
    """Build complete metadata for one selected classifier."""

    return {
        "contract_version": "flowcast_classification_model_card_v1",
        "job_id": job.job_id,
        "model_version": score["model_version"],
        "seed": settings.seed,
        "target": {
            "task": job.task,
            "column": job.target_column,
            "horizon_windows": job.horizon_windows,
            "horizon_minutes": job.horizon_minutes,
            "class_order": list(job.class_names),
        },
        "selection": {
            "family": selection["family"],
            "candidate_id": selection["candidate_id"],
            "parameters": selection["parameters"],
            "parameters_json": json.dumps(
                selection["parameters"],
                indent=2,
                sort_keys=True,
            ),
            "primary_metric": job.primary_metric,
            f"mean_cv_{job.primary_metric}": selection[
                f"mean_cv_{job.primary_metric}"
            ],
            "validation_before_calibration": selection[
                "selection_validation"
            ],
        },
        "probability": {
            "class_order": list(job.class_names),
            "calibration": selection["calibration"],
            "operating_threshold": selection["operating_threshold"],
        },
        "data": {
            "train_start": selection["train_start"],
            "train_end": selection["train_end"],
            "train_rows": selection["train_rows"],
            "validation_start": selection["validation_start"],
            "validation_end": selection["validation_end"],
            "validation_rows": selection["validation_rows"],
            "test_start": score["test_start"],
            "test_end": score["test_end"],
            "test_rows": score["test"]["rows"],
            "training_weight_evidence": selection[
                "training_weight_evidence"
            ],
        },
        "features": {
            "input_feature_count": int(modeling.schema["feature_count"]),
            "input_features": [
                str(record["name"]) for record in modeling.schema["input_features"]
            ],
            "output_feature_count": selection["output_feature_count"],
            "preprocessing_family": selection["preprocessing_family"],
            "preprocessing_version": settings.modelling_version,
        },
        "metrics": {
            "validation": score["validation"],
            "test": score["test"],
        },
        "lineage": {
            "processed_sha256": modeling.summary["input_processed"]["dataset"][
                "sha256"
            ],
            "feature_schema_sha256": modeling.summary["artifacts"][
                "feature_schema"
            ]["sha256"],
            "selection_sha256": selection_record["sha256"],
        },
        "artifacts": {
            "model": selection["model"],
            "predictions": predictions_record,
        },
        "limitations": [
            "This direct classifier is specific to one target and horizon.",
            "CV uses a deterministic timestamp budget; the final family fit uses "
            "all eligible training rows.",
            "Calibration is fit on earlier validation rows and assessed on later "
            "validation rows before test access.",
            "Future weather is not available; weather inputs are observed at origin.",
        ],
    }
