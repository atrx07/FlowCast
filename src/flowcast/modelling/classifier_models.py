"""Jobs, estimators, weighting, and prediction helpers for Step 13."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from xgboost import XGBClassifier

from flowcast.evaluation.classification import (
    binary_ranking_metrics,
    multiclass_metrics,
    validate_probabilities,
)
from flowcast.modelling.preprocessing import FeatureGroups, build_preprocessor


ENCODED_TARGET = "_classification_target"


@dataclass(frozen=True)
class ClassificationJob:
    """One direct classification target and forecast horizon."""

    task: str
    target_column: str
    availability_column: str
    target_timestamp_column: str
    within_split_column: str
    horizon_windows: int
    horizon_minutes: int
    class_names: tuple[str, ...]

    @property
    def job_id(self) -> str:
        """Return the stable classifier registry key."""

        return f"{self.task}_h{self.horizon_windows}"

    @property
    def primary_metric(self) -> str:
        """Return the formal selection metric for this task."""

        return "macro_f1" if self.task == "congestion" else "roc_auc"


@dataclass(frozen=True)
class ClassifierCandidateSpec:
    """One bounded classifier configuration from YAML."""

    family: str
    candidate_id: str
    preprocessing_family: str
    parameters: dict[str, Any]


def build_classification_jobs(
    section: dict[str, Any],
    target_manifest: list[dict[str, Any]],
) -> list[ClassificationJob]:
    """Generate and validate all eight direct classification jobs."""

    by_name = {str(record["name"]): record for record in target_manifest}
    expected_tasks = {
        "congestion": "classification_multiclass",
        "accident": "classification_binary",
    }
    jobs: list[ClassificationJob] = []
    for task in section["tasks"]:
        for horizon in section["horizons"]:
            target_column = f"target_{task}_h{int(horizon)}"
            if target_column not in by_name:
                raise RuntimeError(f"Processed manifest omits {target_column}")
            record = by_name[target_column]
            if record.get("task") != expected_tasks[task]:
                raise RuntimeError(f"{target_column} task metadata changed")
            if int(record["horizon_windows"]) != int(horizon):
                raise RuntimeError(f"{target_column} horizon metadata changed")
            jobs.append(
                ClassificationJob(
                    task=str(task),
                    target_column=target_column,
                    availability_column=str(record["availability_column"]),
                    target_timestamp_column=str(record["target_timestamp_column"]),
                    within_split_column=f"target_within_split_h{int(horizon)}",
                    horizon_windows=int(horizon),
                    horizon_minutes=int(record["horizon_minutes"]),
                    class_names=tuple(section["class_order"][task]),
                )
            )
    if len(jobs) != 8 or len({job.job_id for job in jobs}) != 8:
        raise RuntimeError("Classical classification must generate exactly eight jobs")
    return jobs


def build_classifier_specs(
    section: dict[str, Any],
) -> list[ClassifierCandidateSpec]:
    """Flatten the configured classifier candidates in stable order."""

    return [
        ClassifierCandidateSpec(
            family=str(family),
            candidate_id=str(candidate["candidate_id"]),
            preprocessing_family=str(record["preprocessing_family"]),
            parameters=dict(candidate["parameters"]),
        )
        for family, record in section["estimators"].items()
        for candidate in record["candidates"]
    ]


def eligible_classification_rows(
    frame: pd.DataFrame,
    job: ClassificationJob,
) -> pd.DataFrame:
    """Return target-available, boundary-safe rows with fixed numeric labels."""

    required = {
        job.target_column,
        job.availability_column,
        job.target_timestamp_column,
        job.within_split_column,
    }
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"Classification input omits columns: {sorted(missing)}")
    selected = (
        frame[job.availability_column].fillna(False).astype(bool)
        & frame[job.within_split_column].fillna(False).astype(bool)
    )
    eligible = frame.loc[selected].copy()
    if eligible.empty or eligible[job.target_timestamp_column].isna().any():
        raise RuntimeError(f"{job.job_id} has no valid target timestamps")
    if job.task == "congestion":
        mapping = {name: index for index, name in enumerate(job.class_names)}
        encoded = eligible[job.target_column].map(mapping)
    else:
        encoded = eligible[job.target_column].astype("boolean").astype("Int64")
    if encoded.isna().any():
        raise RuntimeError(f"{job.job_id} contains unknown eligible class labels")
    eligible[ENCODED_TARGET] = encoded.astype(np.int64)
    return eligible.sort_values(
        ["timestamp", "road_id"],
        kind="mergesort",
    ).reset_index(drop=True)


def build_classifier_estimator(
    spec: ClassifierCandidateSpec,
    job: ClassificationJob,
    seed: int,
) -> Any:
    """Build one seeded classifier without fitting."""

    parameters = dict(spec.parameters)
    if spec.family == "decision_tree":
        return DecisionTreeClassifier(
            class_weight="balanced",
            random_state=seed,
            **parameters,
        )
    if spec.family == "random_forest":
        return RandomForestClassifier(
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
            **parameters,
        )
    if spec.family == "svm":
        return LinearSVC(
            class_weight="balanced",
            random_state=seed,
            dual="auto",
            **parameters,
        )
    if spec.family == "xgboost":
        common = {
            "tree_method": "hist",
            "random_state": seed,
            "n_jobs": -1,
            "verbosity": 0,
        }
        if job.task == "congestion":
            common.update(
                {
                    "objective": "multi:softprob",
                    "num_class": len(job.class_names),
                    "eval_metric": "mlogloss",
                }
            )
        else:
            common.update(
                {
                    "objective": "binary:logistic",
                    "eval_metric": "logloss",
                }
            )
        return XGBClassifier(**common, **parameters)
    raise ValueError(f"Unknown classifier family: {spec.family}")


def build_classifier_pipeline(
    spec: ClassifierCandidateSpec,
    job: ClassificationJob,
    groups: FeatureGroups,
    config: dict[str, Any],
    seed: int,
) -> Pipeline:
    """Build fresh preprocessing plus one classifier."""

    policy = config["preprocessing"]["families"][spec.preprocessing_family]
    return Pipeline(
        [
            ("preprocessor", build_preprocessor(groups, policy)),
            ("estimator", build_classifier_estimator(spec, job, seed)),
        ]
    )


def class_weight_evidence(
    labels: Any,
    class_names: tuple[str, ...],
) -> dict[str, Any]:
    """Compute weights solely from the labels supplied to one fit."""

    target = np.asarray(labels, dtype=np.int64)
    classes = np.arange(len(class_names), dtype=np.int64)
    observed = np.unique(target)
    if not np.array_equal(observed, classes):
        raise RuntimeError("Classifier training rows do not contain every class")
    weights = compute_class_weight("balanced", classes=classes, y=target)
    return {
        "rows": int(len(target)),
        "class_counts": {
            name: int((target == index).sum())
            for index, name in enumerate(class_names)
        },
        "balanced_class_weights": {
            name: round(float(weights[index]), 10)
            for index, name in enumerate(class_names)
        },
    }


def fit_classifier(
    pipeline: Pipeline,
    spec: ClassifierCandidateSpec,
    job: ClassificationJob,
    training: pd.DataFrame,
    input_features: list[str],
) -> tuple[float, dict[str, Any]]:
    """Fit one candidate using only fold-local class information."""

    labels = training[ENCODED_TARGET].to_numpy(dtype=np.int64)
    fit_kwargs: dict[str, Any] = {}
    if spec.family == "xgboost":
        fit_kwargs["estimator__sample_weight"] = compute_sample_weight(
            "balanced",
            labels,
        )
    started = perf_counter()
    pipeline.fit(training[input_features], labels, **fit_kwargs)
    seconds = perf_counter() - started
    return seconds, class_weight_evidence(labels, job.class_names)


def score_classifier(
    pipeline: Pipeline,
    job: ClassificationJob,
    evaluation: pd.DataFrame,
    input_features: list[str],
) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any], float]:
    """Predict one evaluation frame and return the task-primary metrics."""

    features = evaluation[input_features]
    labels = evaluation[ENCODED_TARGET].to_numpy(dtype=np.int64)
    started = perf_counter()
    predicted = np.asarray(pipeline.predict(features), dtype=np.int64)
    probabilities: np.ndarray | None = None
    if hasattr(pipeline, "predict_proba"):
        probabilities = validate_probabilities(
            pipeline.predict_proba(features),
            len(evaluation),
            len(job.class_names),
        )
    prediction_seconds = perf_counter() - started
    if job.task == "congestion":
        metrics = multiclass_metrics(
            labels,
            predicted,
            job.class_names,
            probabilities,
        )
    else:
        if probabilities is not None:
            scores = probabilities[:, 1]
        else:
            scores = np.asarray(pipeline.decision_function(features), dtype=float)
        metrics = binary_ranking_metrics(labels, scores)
    return predicted, probabilities, metrics, prediction_seconds


def ordered_probabilities(
    estimator: Any,
    frame: pd.DataFrame,
    input_features: list[str],
    class_count: int,
) -> np.ndarray:
    """Return probabilities after checking the estimator's fixed class order."""

    classes = np.asarray(estimator.classes_, dtype=np.int64)
    if not np.array_equal(classes, np.arange(class_count)):
        raise RuntimeError("Persisted classifier class order changed")
    return validate_probabilities(
        estimator.predict_proba(frame[input_features]),
        len(frame),
        class_count,
    )


def extract_classifier_importance(
    pipeline: Pipeline,
    job: ClassificationJob,
    family: str,
) -> pd.DataFrame:
    """Return tree importance or mean absolute linear-SVM coefficients."""

    preprocessor = pipeline.named_steps["preprocessor"]
    estimator = pipeline.named_steps["estimator"]
    names = [str(name) for name in preprocessor.get_feature_names_out()]
    if hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_, dtype=np.float64)
        kind = "feature_importance"
    elif hasattr(estimator, "coef_"):
        coefficients = np.asarray(estimator.coef_, dtype=np.float64)
        values = np.mean(np.abs(np.atleast_2d(coefficients)), axis=0)
        kind = "mean_absolute_coefficient"
    else:
        return pd.DataFrame()
    if len(names) != len(values):
        raise RuntimeError("Classifier importance does not match feature schema")
    result = pd.DataFrame({"feature": names, "importance": values})
    result = result.sort_values(
        ["importance", "feature"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    result.insert(0, "job_id", job.job_id)
    result.insert(1, "task", job.task)
    result.insert(2, "horizon_windows", job.horizon_windows)
    result.insert(3, "family", family)
    result["importance_kind"] = kind
    result["rank"] = np.arange(1, len(result) + 1)
    return result
