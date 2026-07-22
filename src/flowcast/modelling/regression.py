"""Step 11 orchestration for the from-scratch NumPy regression proof."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LinearRegression

from flowcast.data.artifacts import (
    artifact_record,
    validate_artifact_version,
    write_json,
    write_parquet,
)
from flowcast.evaluation.regression import regression_metrics
from flowcast.modelling.config import load_model_config
from flowcast.modelling.inputs import (
    load_modeling_partition,
    load_preprocessor,
    load_verified_modeling_artifacts,
)
from flowcast.modelling.scratch_linear import (
    NumpyLinearRegressor,
)
from flowcast.modelling.scratch_proof import build_optimizer, synthetic_evidence
from flowcast.modelling.scratch_report import render_scratch_linear_report
from flowcast.settings import Settings


@dataclass(frozen=True)
class ScratchLinearArtifacts:
    """Paths and evidence produced by one Step 11 run."""

    version: str
    summary_path: Path
    report_path: Path
    convergence_path: Path
    coefficients_path: Path
    predictions_path: Path
    model_path: Path
    summary: dict[str, Any]


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        frame.to_csv(index=False, lineterminator="\n", float_format="%.12g"),
        encoding="utf-8",
        newline="\n",
    )


def _eligible_rows(
    frame: pd.DataFrame,
    availability: str,
    within_split: str,
) -> pd.DataFrame:
    if availability not in frame or within_split not in frame:
        raise RuntimeError("Required target eligibility columns are missing")
    selected = frame[availability].fillna(False).astype(bool) & frame[
        within_split
    ].fillna(False).astype(bool)
    return frame.loc[selected].sort_values(
        ["timestamp", "road_id"],
        kind="mergesort",
    ).reset_index(drop=True)


def _load_training_data(
    settings: Settings,
    scratch: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str], Any, dict[str, Any]]:
    artifacts = load_verified_modeling_artifacts(settings)
    train = load_modeling_partition(settings, "train")
    validation = load_modeling_partition(settings, "validation")
    try:
        load_modeling_partition(settings, "test")
    except PermissionError:
        test_check = {
            "default_access_rejected": True,
            "rows_loaded": 0,
            "metrics_calculated": False,
        }
    else:
        raise RuntimeError("Test partition was not sealed for Step 11")
    eligible_train = _eligible_rows(
        train,
        str(scratch["availability_column"]),
        str(scratch["horizon_within_split_column"]),
    )
    eligible_validation = _eligible_rows(
        validation,
        str(scratch["availability_column"]),
        str(scratch["horizon_within_split_column"]),
    )
    row_limit = int(scratch["training_subset"]["row_limit"])
    if len(eligible_train) < row_limit:
        raise RuntimeError("Scratch-linear row limit exceeds eligible training data")
    training_subset = eligible_train.head(row_limit).copy()
    input_features = [
        str(record["name"]) for record in artifacts.schema["input_features"]
    ]
    preprocessor = load_preprocessor(settings, "linear")
    output_features = [
        str(name) for name in preprocessor.get_feature_names_out()
    ]
    return (
        training_subset,
        eligible_validation,
        input_features,
        output_features,
        preprocessor,
        {
            "test": test_check,
            "eligible_training_rows": len(eligible_train),
            "modeling_artifacts": artifacts,
        },
    )


def _coefficient_frame(
    output_features: list[str],
    scratch_model: NumpyLinearRegressor,
    sklearn_model: LinearRegression,
) -> pd.DataFrame:
    if scratch_model.weights_ is None or scratch_model.bias_ is None:
        raise RuntimeError("Scratch model coefficients are unavailable")
    frame = pd.DataFrame(
        {
            "parameter": output_features,
            "parameter_type": "weight",
            "scratch_value": scratch_model.weights_,
            "sklearn_value": np.asarray(sklearn_model.coef_, dtype=float),
        }
    )
    bias = pd.DataFrame(
        {
            "parameter": ["__bias__"],
            "parameter_type": ["bias"],
            "scratch_value": [scratch_model.bias_],
            "sklearn_value": [float(sklearn_model.intercept_)],
        }
    )
    return pd.concat([frame, bias], ignore_index=True)


def run_scratch_linear(
    settings: Settings,
    version: str | None = None,
) -> ScratchLinearArtifacts:
    """Run and persist the Step 11 mathematical regression proof."""

    config = load_model_config(settings)
    scratch = config["scratch_linear"]
    selected_version = validate_artifact_version(
        version or str(scratch["version"])
    )
    gradient_summary, synthetic_summary = synthetic_evidence(
        scratch, settings.seed
    )
    (
        training,
        validation,
        input_features,
        output_features,
        preprocessor,
        evidence,
    ) = _load_training_data(settings, scratch)
    target = str(scratch["target"])
    train_matrix = np.asarray(
        preprocessor.transform(training[input_features]), dtype=np.float64
    )
    validation_matrix = np.asarray(
        preprocessor.transform(validation[input_features]), dtype=np.float64
    )
    train_target = training[target].to_numpy(dtype=np.float64)
    validation_target = validation[target].to_numpy(dtype=np.float64)
    scratch_model = build_optimizer(scratch["optimizer"], settings.seed)
    scratch_model.fit(train_matrix, train_target)
    sklearn_model = LinearRegression()
    sklearn_model.fit(train_matrix, train_target)
    scratch_predictions = scratch_model.predict(validation_matrix)
    sklearn_predictions = sklearn_model.predict(validation_matrix)
    scratch_metrics = regression_metrics(validation_target, scratch_predictions)
    sklearn_metrics = regression_metrics(validation_target, sklearn_predictions)

    metrics_dir = settings.artifacts_dir / "metrics" / selected_version
    model_dir = settings.artifacts_dir / "models" / selected_version
    predictions_dir = settings.artifacts_dir / "predictions" / selected_version
    summary_path = metrics_dir / "summary.json"
    report_path = metrics_dir / "summary.md"
    convergence_path = metrics_dir / "convergence.csv"
    coefficients_path = metrics_dir / "coefficients.csv"
    model_path = model_dir / "model.json"
    predictions_path = predictions_dir / "validation.parquet"

    convergence = pd.DataFrame(
        [record.__dict__ for record in scratch_model.history_]
    )
    _write_csv(convergence, convergence_path)
    coefficients = _coefficient_frame(
        output_features, scratch_model, sklearn_model
    )
    _write_csv(coefficients, coefficients_path)
    prediction_frame = validation[
        ["road_id", "timestamp", "target_timestamp_h1"]
    ].copy()
    prediction_frame["actual"] = validation_target
    prediction_frame["scratch_prediction"] = scratch_predictions
    prediction_frame["sklearn_prediction"] = sklearn_predictions
    write_parquet(prediction_frame, predictions_path)

    modeling = evidence["modeling_artifacts"]
    preprocessor_record = modeling.summary["artifacts"]["preprocessors"]["linear"]
    model_payload = scratch_model.to_payload(
        output_features,
        {
            "target": target,
            "training_rows": len(training),
            "training_timestamp_start": training["timestamp"].min().isoformat(),
            "training_timestamp_end": training["timestamp"].max().isoformat(),
            "preprocessor": preprocessor_record,
            "input_modeling_summary": artifact_record(
                modeling.summary_path, settings
            ),
        },
    )
    write_json(model_payload, model_path)
    coefficient_delta = coefficients.loc[
        coefficients["parameter_type"].eq("weight"),
        "scratch_value",
    ].to_numpy() - coefficients.loc[
        coefficients["parameter_type"].eq("weight"),
        "sklearn_value",
    ].to_numpy()
    summary: dict[str, Any] = {
        "contract_version": str(scratch["contract_version"]),
        "version": selected_version,
        "purpose": "mathematical_verification_not_model_selection",
        "configuration": {
            "base": artifact_record(settings.config_path, settings),
            "models": artifact_record(settings.models_config_path, settings),
            "seed": settings.seed,
        },
        "input_modeling": {
            "version": settings.modelling_version,
            "summary": artifact_record(modeling.summary_path, settings),
            "assignments": artifact_record(modeling.assignments_path, settings),
            "feature_schema": artifact_record(modeling.schema_path, settings),
            "linear_preprocessor": preprocessor_record,
        },
        "target": {
            "name": target,
            "horizon_windows": 1,
            "horizon_minutes": 30,
            "availability_column": str(scratch["availability_column"]),
            "within_split_column": str(
                scratch["horizon_within_split_column"]
            ),
        },
        "gradient_check": gradient_summary,
        "synthetic_proof": synthetic_summary,
        "training": {
            "subset_method": str(scratch["training_subset"]["method"]),
            "eligible_training_rows": int(evidence["eligible_training_rows"]),
            "train_rows": len(training),
            "train_timestamp_start": training["timestamp"].min().isoformat(),
            "train_timestamp_end": training["timestamp"].max().isoformat(),
            "train_road_count": int(training["road_id"].nunique()),
            "validation_rows": len(validation),
            "validation_timestamp_start": validation["timestamp"].min().isoformat(),
            "validation_timestamp_end": validation["timestamp"].max().isoformat(),
            "validation_road_count": int(validation["road_id"].nunique()),
            "input_feature_count": len(input_features),
            "output_feature_count": len(output_features),
            "same_rows_for_both_estimators": True,
            "initial_loss": scratch_model.history_[0].loss,
            "final_loss": scratch_model.history_[-1].loss,
            "iterations_completed": scratch_model.history_[-1].iteration,
            "converged": scratch_model.converged_,
        },
        "test_partition": evidence["test"],
        "metrics": {
            "split": "validation",
            "scratch": scratch_metrics,
            "sklearn": sklearn_metrics,
            "coefficient_rmse": round(
                float(np.sqrt(np.mean(coefficient_delta**2))), 10
            ),
            "prediction_rmse_between_estimators": round(
                float(
                    np.sqrt(
                        np.mean((scratch_predictions - sklearn_predictions) ** 2)
                    )
                ),
                10,
            ),
        },
        "libraries": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "limitations": [
            "This Step 11 slice demonstrates the mathematics on next-window "
            "volume only.",
            "The earliest eligible training subset is bounded for an auditable "
            "full-batch gradient loop.",
            "Validation metrics are not final hold-out metrics and do not "
            "select a production model.",
            "The final test partition remains sealed until Step 12 choices are frozen.",
        ],
        "checks": [
            {"name": "central_gradient_check", "passed": True},
            {"name": "synthetic_parameter_recovery", "passed": True},
            {
                "name": "flowcast_loss_decreased",
                "passed": bool(
                    scratch_model.history_[-1].loss
                    < scratch_model.history_[0].loss
                ),
            },
            {"name": "identical_estimator_rows", "passed": True},
            {"name": "test_partition_sealed", "passed": True},
        ],
        "artifacts": {
            "model": artifact_record(model_path, settings),
            "convergence": artifact_record(convergence_path, settings),
            "coefficients": artifact_record(coefficients_path, settings),
            "validation_predictions": artifact_record(predictions_path, settings),
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        render_scratch_linear_report(summary),
        encoding="utf-8",
        newline="\n",
    )
    summary["artifacts"]["report"] = artifact_record(report_path, settings)
    write_json(summary, summary_path)
    return ScratchLinearArtifacts(
        version=selected_version,
        summary_path=summary_path,
        report_path=report_path,
        convergence_path=convergence_path,
        coefficients_path=coefficients_path,
        predictions_path=predictions_path,
        model_path=model_path,
        summary=summary,
    )
