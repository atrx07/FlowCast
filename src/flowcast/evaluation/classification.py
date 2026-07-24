"""Classification metrics, probability checks, and threshold selection."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)


def validate_probabilities(
    probabilities: Any,
    rows: int,
    class_count: int,
) -> np.ndarray:
    """Return a finite, normalized probability matrix or fail closed."""

    matrix = np.asarray(probabilities, dtype=np.float64)
    if matrix.shape != (rows, class_count):
        raise ValueError(
            f"Expected probability shape {(rows, class_count)}, got {matrix.shape}"
        )
    if not np.isfinite(matrix).all():
        raise ValueError("Class probabilities must be finite")
    if (matrix < -1e-12).any() or (matrix > 1.0 + 1e-12).any():
        raise ValueError("Class probabilities must lie between zero and one")
    if not np.allclose(matrix.sum(axis=1), 1.0, atol=1e-6, rtol=0.0):
        raise ValueError("Class probabilities must sum to one")
    matrix = np.clip(matrix, 0.0, 1.0)
    return matrix / matrix.sum(axis=1, keepdims=True)


def probability_quality(
    actual: Any,
    probabilities: Any,
    class_count: int,
) -> dict[str, float]:
    """Return Brier score and log loss for ordered class probabilities."""

    truth = np.asarray(actual, dtype=np.int64)
    matrix = validate_probabilities(probabilities, len(truth), class_count)
    if truth.ndim != 1 or not truth.size:
        raise ValueError("Probability quality requires non-empty class labels")
    if truth.min() < 0 or truth.max() >= class_count:
        raise ValueError("Actual class labels fall outside the configured order")
    one_hot = np.eye(class_count, dtype=np.float64)[truth]
    brier = float(np.mean(np.sum((matrix - one_hot) ** 2, axis=1)))
    return {
        "brier_score": round(brier, 10),
        "log_loss": round(
            float(log_loss(truth, matrix, labels=np.arange(class_count))),
            10,
        ),
    }


def multiclass_metrics(
    actual: Any,
    predicted: Any,
    class_names: Sequence[str],
    probabilities: Any | None = None,
) -> dict[str, Any]:
    """Return ordered four-class metrics and optional probability quality."""

    truth = np.asarray(actual, dtype=np.int64)
    estimates = np.asarray(predicted, dtype=np.int64)
    class_count = len(class_names)
    if truth.ndim != 1 or estimates.shape != truth.shape or not truth.size:
        raise ValueError("Multiclass metrics require aligned one-dimensional labels")
    labels = np.arange(class_count)
    precision, recall, f1, support = precision_recall_fscore_support(
        truth,
        estimates,
        labels=labels,
        zero_division=0,
    )
    metrics: dict[str, Any] = {
        "rows": int(len(truth)),
        "accuracy": round(float(accuracy_score(truth, estimates)), 10),
        "macro_precision": round(float(precision.mean()), 10),
        "macro_recall": round(float(recall.mean()), 10),
        "macro_f1": round(
            float(f1_score(truth, estimates, labels=labels, average="macro")),
            10,
        ),
        "confusion_matrix": confusion_matrix(
            truth,
            estimates,
            labels=labels,
        ).astype(int).tolist(),
        "per_class": {
            str(name): {
                "precision": round(float(precision[index]), 10),
                "recall": round(float(recall[index]), 10),
                "f1": round(float(f1[index]), 10),
                "support": int(support[index]),
            }
            for index, name in enumerate(class_names)
        },
    }
    if probabilities is not None:
        metrics.update(probability_quality(truth, probabilities, class_count))
    return metrics


def binary_ranking_metrics(actual: Any, scores: Any) -> dict[str, float | int]:
    """Return imbalance-visible ranking metrics from probability or decision scores."""

    truth = np.asarray(actual, dtype=np.int64)
    ranking = np.asarray(scores, dtype=np.float64)
    if truth.ndim != 1 or ranking.shape != truth.shape or not truth.size:
        raise ValueError("Binary ranking metrics require aligned non-empty vectors")
    if not np.isfinite(ranking).all() or set(np.unique(truth)) != {0, 1}:
        raise ValueError("Binary ranking metrics require finite scores and both classes")
    return {
        "rows": int(len(truth)),
        "positive_rows": int(truth.sum()),
        "negative_rows": int(len(truth) - truth.sum()),
        "positive_rate": round(float(truth.mean()), 10),
        "roc_auc": round(float(roc_auc_score(truth, ranking)), 10),
        "pr_auc": round(float(average_precision_score(truth, ranking)), 10),
    }


def binary_metrics(
    actual: Any,
    positive_probability: Any,
    threshold: float,
) -> dict[str, Any]:
    """Return accident ranking, calibration, and operating-point metrics."""

    truth = np.asarray(actual, dtype=np.int64)
    positive = np.asarray(positive_probability, dtype=np.float64)
    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("Binary threshold must lie between zero and one")
    probability_matrix = validate_probabilities(
        np.column_stack([1.0 - positive, positive]),
        len(truth),
        2,
    )
    predictions = (positive >= threshold).astype(np.int64)
    metrics: dict[str, Any] = binary_ranking_metrics(truth, positive)
    metrics.update(
        {
            "threshold": round(float(threshold), 10),
            "accuracy": round(float(accuracy_score(truth, predictions)), 10),
            "precision": round(
                float(precision_score(truth, predictions, zero_division=0)),
                10,
            ),
            "recall": round(
                float(recall_score(truth, predictions, zero_division=0)),
                10,
            ),
            "f1": round(
                float(f1_score(truth, predictions, zero_division=0)),
                10,
            ),
            "predicted_positive_rows": int(predictions.sum()),
            "confusion_matrix": confusion_matrix(
                truth,
                predictions,
                labels=[0, 1],
            ).astype(int).tolist(),
        }
    )
    metrics.update(probability_quality(truth, probability_matrix, 2))
    return metrics


def select_binary_threshold(
    actual: Any,
    positive_probability: Any,
    candidate_quantiles: int,
    default_threshold: float,
) -> tuple[float, pd.DataFrame, dict[str, Any]]:
    """Select validation F1, then recall/precision, then the lower threshold."""

    truth = np.asarray(actual, dtype=np.int64)
    positive = np.asarray(positive_probability, dtype=np.float64)
    validate_probabilities(
        np.column_stack([1.0 - positive, positive]),
        len(truth),
        2,
    )
    quantiles = np.linspace(0.0, 1.0, num=int(candidate_quantiles))
    thresholds = np.unique(
        np.concatenate(
            [
                np.quantile(positive, quantiles),
                np.asarray([float(default_threshold)], dtype=np.float64),
            ]
        )
    )
    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        predicted = (positive >= threshold).astype(np.int64)
        rows.append(
            {
                "threshold": round(float(threshold), 10),
                "precision": round(
                    float(precision_score(truth, predicted, zero_division=0)),
                    10,
                ),
                "recall": round(
                    float(recall_score(truth, predicted, zero_division=0)),
                    10,
                ),
                "f1": round(
                    float(f1_score(truth, predicted, zero_division=0)),
                    10,
                ),
                "predicted_positive_rows": int(predicted.sum()),
            }
        )
    selected = max(
        rows,
        key=lambda row: (
            float(row["f1"]),
            float(row["recall"]),
            float(row["precision"]),
            -float(row["threshold"]),
        ),
    )
    return float(selected["threshold"]), pd.DataFrame(rows), selected
