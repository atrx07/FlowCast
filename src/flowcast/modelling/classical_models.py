"""Pure job, estimator, fold, and importance helpers for Step 12 regression."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from flowcast.evaluation.regression import regression_metrics
from flowcast.modelling.preprocessing import (
    FeatureGroups,
    build_preprocessor,
)


@dataclass(frozen=True)
class RegressionJob:
    """One direct target/horizon regression task."""

    target_key: str
    target_column: str
    availability_column: str
    target_timestamp_column: str
    within_split_column: str
    horizon_windows: int
    horizon_minutes: int

    @property
    def job_id(self) -> str:
        """Return the stable task registry key."""

        return f"{self.target_key}_h{self.horizon_windows}"


@dataclass(frozen=True)
class CandidateSpec:
    """One bounded estimator configuration from the versioned YAML contract."""

    family: str
    candidate_id: str
    preprocessing_family: str
    parameters: dict[str, Any]


def build_regression_jobs(
    regression_config: dict[str, Any],
    target_manifest: list[dict[str, Any]],
) -> list[RegressionJob]:
    """Generate and validate all configured direct regression jobs."""

    by_name = {str(record["name"]): record for record in target_manifest}
    jobs: list[RegressionJob] = []
    for target_key in regression_config["targets"]:
        for horizon in regression_config["horizons"]:
            target_column = f"target_{target_key}_h{int(horizon)}"
            if target_column not in by_name:
                raise RuntimeError(f"Processed manifest omits {target_column}")
            record = by_name[target_column]
            if record.get("task") != "regression":
                raise RuntimeError(f"{target_column} is not a regression target")
            if int(record["horizon_windows"]) != int(horizon):
                raise RuntimeError(f"{target_column} horizon metadata changed")
            jobs.append(
                RegressionJob(
                    target_key=str(target_key),
                    target_column=target_column,
                    availability_column=str(record["availability_column"]),
                    target_timestamp_column=str(
                        record["target_timestamp_column"]
                    ),
                    within_split_column=f"target_within_split_h{int(horizon)}",
                    horizon_windows=int(horizon),
                    horizon_minutes=int(record["horizon_minutes"]),
                )
            )
    if len(jobs) != 12 or len({job.job_id for job in jobs}) != len(jobs):
        raise RuntimeError("Classical regression must generate exactly 12 jobs")
    return jobs


def build_candidate_specs(
    regression_config: dict[str, Any],
) -> list[CandidateSpec]:
    """Flatten the bounded estimator search space in configured order."""

    specs: list[CandidateSpec] = []
    for family, record in regression_config["estimators"].items():
        for candidate in record["candidates"]:
            specs.append(
                CandidateSpec(
                    family=str(family),
                    candidate_id=str(candidate["candidate_id"]),
                    preprocessing_family=str(record["preprocessing_family"]),
                    parameters=dict(candidate["parameters"]),
                )
            )
    return specs


def eligible_rows(frame: pd.DataFrame, job: RegressionJob) -> pd.DataFrame:
    """Return finite, boundary-safe rows for one regression job."""

    required = {
        job.target_column,
        job.availability_column,
        job.target_timestamp_column,
        job.within_split_column,
    }
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"Regression input is missing columns: {sorted(missing)}")
    selected = (
        frame[job.availability_column].fillna(False).astype(bool)
        & frame[job.within_split_column].fillna(False).astype(bool)
    )
    eligible = frame.loc[selected].copy()
    targets = pd.to_numeric(eligible[job.target_column], errors="coerce")
    if eligible.empty or targets.isna().any() or not np.isfinite(targets).all():
        raise RuntimeError(f"{job.job_id} contains invalid eligible targets")
    if eligible[job.target_timestamp_column].isna().any():
        raise RuntimeError(f"{job.job_id} contains missing target timestamps")
    return eligible.sort_values(
        ["timestamp", "road_id"],
        kind="mergesort",
    ).reset_index(drop=True)


def build_estimator(spec: CandidateSpec, seed: int) -> RegressorMixin:
    """Build one seeded estimator without fitting it."""

    parameters = dict(spec.parameters)
    if spec.family == "linear_regression":
        return LinearRegression(**parameters)
    if spec.family == "decision_tree":
        return DecisionTreeRegressor(random_state=seed, **parameters)
    if spec.family == "random_forest":
        return RandomForestRegressor(
            random_state=seed,
            n_jobs=-1,
            **parameters,
        )
    if spec.family == "xgboost":
        return XGBRegressor(
            objective="reg:squarederror",
            tree_method="hist",
            random_state=seed,
            n_jobs=-1,
            verbosity=0,
            **parameters,
        )
    raise ValueError(f"Unknown classical regression family: {spec.family}")


def build_pipeline(
    spec: CandidateSpec,
    groups: FeatureGroups,
    config: dict[str, Any],
    seed: int,
) -> Pipeline:
    """Build an unfitted preprocessing-plus-estimator pipeline."""

    policy = config["preprocessing"]["families"][spec.preprocessing_family]
    return Pipeline(
        [
            ("preprocessor", build_preprocessor(groups, policy)),
            ("estimator", build_estimator(spec, seed)),
        ]
    )


def fit_and_score(
    pipeline: Pipeline,
    training: pd.DataFrame,
    evaluation: pd.DataFrame,
    input_features: list[str],
    target_column: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit one pipeline and return predictions, metrics, and elapsed runtime."""

    fit_start = perf_counter()
    pipeline.fit(
        training[input_features],
        training[target_column].to_numpy(dtype=np.float64),
    )
    fit_seconds = perf_counter() - fit_start
    prediction_start = perf_counter()
    predictions = np.asarray(
        pipeline.predict(evaluation[input_features]),
        dtype=np.float64,
    )
    prediction_seconds = perf_counter() - prediction_start
    metrics = regression_metrics(
        evaluation[target_column].to_numpy(dtype=np.float64),
        predictions,
    )
    metrics.update(
        {
            "fit_seconds": round(float(fit_seconds), 6),
            "prediction_seconds": round(float(prediction_seconds), 6),
        }
    )
    return predictions, metrics


def extract_feature_importance(
    pipeline: Pipeline,
    job: RegressionJob,
    family: str,
) -> pd.DataFrame:
    """Return coefficients or tree importance for one fitted selected pipeline."""

    preprocessor = pipeline.named_steps["preprocessor"]
    estimator = pipeline.named_steps["estimator"]
    names = [str(name) for name in preprocessor.get_feature_names_out()]
    if hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_, dtype=np.float64)
        kind = "feature_importance"
    elif hasattr(estimator, "coef_"):
        values = np.abs(np.asarray(estimator.coef_, dtype=np.float64).reshape(-1))
        kind = "absolute_coefficient"
    else:
        return pd.DataFrame(
            columns=[
                "job_id",
                "target",
                "horizon_windows",
                "family",
                "feature",
                "importance",
                "importance_kind",
                "rank",
            ]
        )
    if len(names) != len(values):
        raise RuntimeError("Feature-importance vector does not match feature schema")
    frame = pd.DataFrame({"feature": names, "importance": values})
    frame = frame.sort_values(
        ["importance", "feature"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    frame.insert(0, "job_id", job.job_id)
    frame.insert(1, "target", job.target_key)
    frame.insert(2, "horizon_windows", job.horizon_windows)
    frame.insert(3, "family", family)
    frame["importance_kind"] = kind
    frame["rank"] = np.arange(1, len(frame) + 1)
    return frame
