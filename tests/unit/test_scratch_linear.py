"""Unit tests for explicit NumPy regression mathematics and metrics."""

from __future__ import annotations

import numpy as np
import pytest

from flowcast.evaluation.regression import regression_metrics
from flowcast.modelling.scratch_linear import (
    NumpyLinearRegressor,
    check_mse_gradients,
    mean_squared_error,
    mse_gradients,
    predict_linear,
)


def test_prediction_loss_and_analytical_gradients_are_explicit() -> None:
    features = np.array([[1.0, 2.0], [3.0, 4.0], [-1.0, 1.5]])
    weights = np.array([0.5, -0.25])
    bias = 0.1
    actual = np.array([1.0, 2.0, -0.5])

    predicted = predict_linear(features, weights, bias)
    expected = features @ weights + bias
    assert np.array_equal(predicted, expected)
    assert mean_squared_error(actual, predicted) == pytest.approx(
        float(np.mean((expected - actual) ** 2))
    )
    weight_gradient, bias_gradient = mse_gradients(
        features, actual, weights, bias
    )
    residuals = expected - actual
    assert weight_gradient == pytest.approx(2 * features.T @ residuals / 3)
    assert bias_gradient == pytest.approx(2 * residuals.mean())


def test_every_weight_and_bias_pass_central_gradient_check() -> None:
    generator = np.random.default_rng(42)
    features = generator.normal(size=(24, 5))
    actual = generator.normal(size=24)
    weights = generator.normal(size=5)

    result = check_mse_gradients(
        features,
        actual,
        weights,
        0.25,
        epsilon=1.0e-6,
        absolute_tolerance=1.0e-6,
        relative_tolerance=1.0e-5,
    )

    assert result.passed
    assert result.analytical_weights == pytest.approx(
        result.numerical_weights, abs=1.0e-6, rel=1.0e-5
    )
    assert result.analytical_bias == pytest.approx(
        result.numerical_bias, abs=1.0e-6, rel=1.0e-5
    )


def test_gradient_descent_recovers_synthetic_solution_reproducibly() -> None:
    generator = np.random.default_rng(7)
    features = generator.normal(size=(256, 4))
    true_weights = np.array([1.5, -0.75, 0.25, 2.0])
    true_bias = -0.4
    actual = features @ true_weights + true_bias
    parameters = {
        "learning_rate": 0.1,
        "max_iterations": 1_000,
        "tolerance": 1.0e-12,
        "patience": 25,
        "seed": 42,
        "initialization_scale": 0.01,
    }

    first = NumpyLinearRegressor(**parameters).fit(features, actual)
    second = NumpyLinearRegressor(**parameters).fit(features, actual)

    assert first.converged_
    assert first.history_ == second.history_
    assert first.weights_ == pytest.approx(true_weights, abs=1.0e-5)
    assert first.bias_ == pytest.approx(true_bias, abs=1.0e-5)
    assert first.history_[-1].loss < first.history_[0].loss * 1.0e-10
    payload = first.to_payload(["a", "b", "c", "d"], {"proof": True})
    restored = NumpyLinearRegressor.from_payload(payload)
    assert restored.predict(features) == pytest.approx(first.predict(features))


def test_regression_metrics_report_mape_denominator() -> None:
    metrics = regression_metrics(
        np.array([0.0, 10.0, 20.0]),
        np.array([1.0, 8.0, 24.0]),
    )

    assert metrics["rmse"] == pytest.approx(np.sqrt(7.0))
    assert metrics["mae"] == pytest.approx(7 / 3)
    assert metrics["mape_percent"] == pytest.approx(20.0)
    assert metrics["mape_nonzero_rows"] == 2
    assert metrics["mape_zero_actual_rows"] == 1
