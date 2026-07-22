"""Pure NumPy linear regression and finite-difference gradient verification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class GradientCheckResult:
    """Analytical-versus-numerical gradient comparison evidence."""

    analytical_weights: np.ndarray
    numerical_weights: np.ndarray
    analytical_bias: float
    numerical_bias: float
    maximum_absolute_error: float
    maximum_relative_error: float
    passed: bool


@dataclass(frozen=True)
class IterationRecord:
    """One full-batch gradient-descent observation."""

    iteration: int
    loss: float
    gradient_norm: float


def _matrix(values: Any) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or not matrix.shape[0] or not matrix.shape[1]:
        raise ValueError("Feature matrix must be a non-empty two-dimensional array")
    if not np.isfinite(matrix).all():
        raise ValueError("Feature matrix must contain only finite values")
    return matrix


def _vector(values: Any, expected_rows: int | None = None) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or not vector.size:
        raise ValueError(
            "Target or weight vector must be non-empty and one-dimensional"
        )
    if expected_rows is not None and vector.size != expected_rows:
        raise ValueError("Feature and target row counts must match")
    if not np.isfinite(vector).all():
        raise ValueError("Target or weight vector must contain only finite values")
    return vector


def predict_linear(
    features: Any,
    weights: Any,
    bias: float,
) -> np.ndarray:
    """Return the explicit matrix-vector prediction ``X @ w + b``."""

    matrix = _matrix(features)
    vector = _vector(weights)
    if matrix.shape[1] != vector.size:
        raise ValueError("Feature columns and weights must have the same length")
    if not np.isfinite(float(bias)):
        raise ValueError("Bias must be finite")
    return matrix @ vector + float(bias)


def mean_squared_error(actual: Any, predicted: Any) -> float:
    """Return mean squared error using an explicit NumPy reduction."""

    truth = _vector(actual)
    estimates = _vector(predicted, truth.size)
    residuals = estimates - truth
    return float(np.mean(residuals * residuals))


def mse_gradients(
    features: Any,
    actual: Any,
    weights: Any,
    bias: float,
) -> tuple[np.ndarray, float]:
    """Return analytical MSE gradients for weights and bias."""

    matrix = _matrix(features)
    truth = _vector(actual, matrix.shape[0])
    vector = _vector(weights)
    estimates = predict_linear(matrix, vector, bias)
    residuals = estimates - truth
    scale = 2.0 / matrix.shape[0]
    weight_gradient = scale * (matrix.T @ residuals)
    bias_gradient = float(scale * np.sum(residuals))
    return weight_gradient, bias_gradient


def check_mse_gradients(
    features: Any,
    actual: Any,
    weights: Any,
    bias: float,
    *,
    epsilon: float = 1.0e-6,
    absolute_tolerance: float = 1.0e-6,
    relative_tolerance: float = 1.0e-5,
) -> GradientCheckResult:
    """Check every analytical gradient by central finite differences."""

    if epsilon <= 0 or absolute_tolerance <= 0 or relative_tolerance <= 0:
        raise ValueError("Gradient-check tolerances must be positive")
    matrix = _matrix(features)
    truth = _vector(actual, matrix.shape[0])
    vector = _vector(weights)
    analytical_weights, analytical_bias = mse_gradients(
        matrix, truth, vector, bias
    )
    numerical_weights = np.empty_like(vector)
    for index in range(vector.size):
        above = vector.copy()
        below = vector.copy()
        above[index] += epsilon
        below[index] -= epsilon
        loss_above = mean_squared_error(
            truth, predict_linear(matrix, above, bias)
        )
        loss_below = mean_squared_error(
            truth, predict_linear(matrix, below, bias)
        )
        numerical_weights[index] = (loss_above - loss_below) / (2 * epsilon)
    numerical_bias = (
        mean_squared_error(truth, predict_linear(matrix, vector, bias + epsilon))
        - mean_squared_error(truth, predict_linear(matrix, vector, bias - epsilon))
    ) / (2 * epsilon)
    analytical = np.append(analytical_weights, analytical_bias)
    numerical = np.append(numerical_weights, numerical_bias)
    absolute_errors = np.abs(analytical - numerical)
    relative_errors = absolute_errors / np.maximum(
        np.abs(analytical) + np.abs(numerical),
        np.finfo(np.float64).eps,
    )
    passed = bool(
        np.allclose(
            analytical,
            numerical,
            atol=absolute_tolerance,
            rtol=relative_tolerance,
        )
    )
    return GradientCheckResult(
        analytical_weights=analytical_weights,
        numerical_weights=numerical_weights,
        analytical_bias=analytical_bias,
        numerical_bias=float(numerical_bias),
        maximum_absolute_error=float(absolute_errors.max()),
        maximum_relative_error=float(relative_errors.max()),
        passed=passed,
    )


class NumpyLinearRegressor:
    """Full-batch linear regressor trained by an explicit gradient loop."""

    def __init__(
        self,
        *,
        learning_rate: float,
        max_iterations: int,
        tolerance: float,
        patience: int,
        seed: int,
        initialization_scale: float,
    ) -> None:
        if learning_rate <= 0 or tolerance <= 0 or initialization_scale <= 0:
            raise ValueError("Optimizer rates, tolerance, and scale must be positive")
        if max_iterations <= 0 or patience <= 0:
            raise ValueError("Optimizer iterations and patience must be positive")
        self.learning_rate = float(learning_rate)
        self.max_iterations = int(max_iterations)
        self.tolerance = float(tolerance)
        self.patience = int(patience)
        self.seed = int(seed)
        self.initialization_scale = float(initialization_scale)
        self.weights_: np.ndarray | None = None
        self.bias_: float | None = None
        self.history_: list[IterationRecord] = []
        self.converged_: bool = False

    def fit(self, features: Any, actual: Any) -> "NumpyLinearRegressor":
        """Fit weights with seeded initialization and explicit updates."""

        matrix = _matrix(features)
        truth = _vector(actual, matrix.shape[0])
        generator = np.random.default_rng(self.seed)
        weights = generator.normal(
            loc=0.0,
            scale=self.initialization_scale,
            size=matrix.shape[1],
        )
        bias = float(generator.normal(0.0, self.initialization_scale))
        history: list[IterationRecord] = []
        previous_loss: float | None = None
        stale_iterations = 0
        converged = False
        for iteration in range(self.max_iterations + 1):
            estimates = predict_linear(matrix, weights, bias)
            loss = mean_squared_error(truth, estimates)
            weight_gradient, bias_gradient = mse_gradients(
                matrix, truth, weights, bias
            )
            gradient_norm = float(
                np.sqrt(np.dot(weight_gradient, weight_gradient) + bias_gradient**2)
            )
            if not np.isfinite(loss) or not np.isfinite(gradient_norm):
                raise FloatingPointError("Gradient descent produced a non-finite value")
            history.append(IterationRecord(iteration, loss, gradient_norm))
            if previous_loss is not None:
                improvement = previous_loss - loss
                scale = max(1.0, abs(previous_loss))
                if improvement < -self.tolerance * scale:
                    raise FloatingPointError(
                        "Gradient-descent loss increased; reduce the learning rate"
                    )
                stale_iterations = (
                    stale_iterations + 1
                    if improvement <= self.tolerance * scale
                    else 0
                )
                if stale_iterations >= self.patience:
                    converged = True
                    break
            if iteration == self.max_iterations:
                break
            weights = weights - self.learning_rate * weight_gradient
            bias = bias - self.learning_rate * bias_gradient
            previous_loss = loss
        self.weights_ = weights
        self.bias_ = float(bias)
        self.history_ = history
        self.converged_ = converged
        return self

    def predict(self, features: Any) -> np.ndarray:
        """Predict with fitted coefficients."""

        if self.weights_ is None or self.bias_ is None:
            raise RuntimeError("Scratch linear regressor is not fitted")
        return predict_linear(features, self.weights_, self.bias_)

    def to_payload(
        self,
        output_features: Sequence[str],
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """Return a JSON-safe representation of the fitted scratch model."""

        if self.weights_ is None or self.bias_ is None or not self.history_:
            raise RuntimeError("Scratch linear regressor is not fitted")
        if len(output_features) != self.weights_.size:
            raise ValueError("Output feature names must align with fitted weights")
        return {
            "contract_version": "scratch_linear_model_v1",
            "weights": [float(value) for value in self.weights_],
            "bias": self.bias_,
            "output_features": [str(name) for name in output_features],
            "optimizer": {
                "learning_rate": self.learning_rate,
                "max_iterations": self.max_iterations,
                "tolerance": self.tolerance,
                "patience": self.patience,
                "seed": self.seed,
                "initialization_scale": self.initialization_scale,
            },
            "iterations_completed": self.history_[-1].iteration,
            "converged": self.converged_,
            "metadata": metadata,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "NumpyLinearRegressor":
        """Reconstruct a fitted scratch model from its JSON representation."""

        if payload.get("contract_version") != "scratch_linear_model_v1":
            raise ValueError("Unsupported scratch-linear model artifact")
        optimizer = payload["optimizer"]
        model = cls(
            learning_rate=float(optimizer["learning_rate"]),
            max_iterations=int(optimizer["max_iterations"]),
            tolerance=float(optimizer["tolerance"]),
            patience=int(optimizer["patience"]),
            seed=int(optimizer["seed"]),
            initialization_scale=float(optimizer["initialization_scale"]),
        )
        model.weights_ = _vector(payload["weights"])
        model.bias_ = float(payload["bias"])
        model.converged_ = bool(payload["converged"])
        return model
