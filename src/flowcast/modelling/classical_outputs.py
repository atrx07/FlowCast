"""Tabular output and model-card assembly for Step 12."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from flowcast.modelling.classical_models import RegressionJob
from flowcast.settings import Settings


def candidate_frame(aggregates: list[dict[str, Any]]) -> pd.DataFrame:
    """Flatten candidate aggregates for deterministic CSV persistence."""

    rows = []
    for record in aggregates:
        row = {key: value for key, value in record.items() if key != "parameters"}
        row["parameters_json"] = json.dumps(
            record["parameters"],
            sort_keys=True,
            separators=(",", ":"),
        )
        rows.append(row)
    return pd.DataFrame(rows)


def family_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Flatten validation metrics for each family winner."""

    rows = []
    for record in records:
        row = {
            key: value
            for key, value in record.items()
            if key not in {"parameters", "validation"}
        }
        row["parameters_json"] = json.dumps(
            record["parameters"],
            sort_keys=True,
            separators=(",", ":"),
        )
        for key, value in record["validation"].items():
            row[f"validation_{key}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def scoreboard_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Flatten final validation/test scoreboard records."""

    rows = []
    for record in records:
        row = {
            key: value
            for key, value in record.items()
            if key not in {"validation", "test"}
        }
        for split in ("validation", "test"):
            for key, value in record[split].items():
                row[f"{split}_{key}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def build_model_card(
    selection: dict[str, Any],
    scoreboard: dict[str, Any],
    job: RegressionJob,
    modeling: Any,
    selection_record: dict[str, Any],
    predictions_record: dict[str, Any],
    settings: Settings,
) -> dict[str, Any]:
    """Build complete machine-readable metadata for one selected model."""

    return {
        "contract_version": "flowcast_model_card_v1",
        "job_id": job.job_id,
        "model_version": scoreboard["model_version"],
        "seed": settings.seed,
        "target": {
            "key": job.target_key,
            "column": job.target_column,
            "horizon_windows": job.horizon_windows,
            "horizon_minutes": job.horizon_minutes,
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
            "mean_cv_rmse": selection["mean_cv_rmse"],
            "validation_primary_metric": "rmse",
        },
        "data": {
            "train_start": selection["train_start"],
            "train_end": selection["train_end"],
            "train_rows": selection["train_rows"],
            "validation_start": selection["validation_start"],
            "validation_end": selection["validation_end"],
            "validation_rows": selection["validation_rows"],
            "test_start": scoreboard["test_start"],
            "test_end": scoreboard["test_end"],
            "test_rows": scoreboard["test"]["rows"],
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
            "validation": scoreboard["validation"],
            "test": scoreboard["test"],
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
            "This direct model is specific to one target and horizon.",
            "CV search uses a deterministic timestamp budget spanning each fold; "
            "the selected family fit uses every eligible training row.",
            "Weather inputs are observed at the origin, not future weather forecasts.",
            "Uncertainty intervals are added in Step 16 and are not part of this card.",
        ],
    }
