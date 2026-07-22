"""Synthetic convergence and gradient evidence for Step 11."""

from __future__ import annotations

from typing import Any

import numpy as np

from flowcast.modelling.scratch_linear import (
    NumpyLinearRegressor,
    check_mse_gradients,
)


def build_optimizer(
    config: dict[str, Any],
    seed: int,
) -> NumpyLinearRegressor:
    """Build the configured pure-NumPy full-batch optimizer."""

    return NumpyLinearRegressor(
        learning_rate=float(config["learning_rate"]),
        max_iterations=int(config["max_iterations"]),
        tolerance=float(config["tolerance"]),
        patience=int(config["patience"]),
        seed=seed,
        initialization_scale=float(config["initialization_scale"]),
    )


def synthetic_evidence(
    scratch: dict[str, Any],
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Prove every gradient and recover a controlled linear solution."""

    proof = scratch["synthetic_proof"]
    generator = np.random.default_rng(seed)
    rows = int(proof["rows"])
    feature_count = int(proof["features"])
    features = generator.normal(size=(rows, feature_count))
    true_weights = generator.normal(size=feature_count)
    true_bias = float(generator.normal())
    actual = features @ true_weights + true_bias
    candidate_weights = generator.normal(size=feature_count)
    candidate_bias = float(generator.normal())
    check = scratch["gradient_check"]
    gradient = check_mse_gradients(
        features[:16],
        actual[:16],
        candidate_weights,
        candidate_bias,
        epsilon=float(check["epsilon"]),
        absolute_tolerance=float(check["absolute_tolerance"]),
        relative_tolerance=float(check["relative_tolerance"]),
    )
    if not gradient.passed:
        raise RuntimeError("Central finite-difference gradient check failed")
    model = build_optimizer(proof, seed)
    model.fit(features, actual)
    if model.weights_ is None or model.bias_ is None:
        raise RuntimeError("Synthetic scratch model did not fit")
    maximum_coefficient_error = float(
        np.max(np.abs(model.weights_ - true_weights))
    )
    bias_error = abs(model.bias_ - true_bias)
    tolerance = float(proof["coefficient_tolerance"])
    if maximum_coefficient_error > tolerance or bias_error > tolerance:
        raise RuntimeError("Synthetic parameter recovery exceeded tolerance")
    gradient_summary = {
        "method": "central_finite_difference",
        "parameter_count": feature_count + 1,
        "epsilon": float(check["epsilon"]),
        "absolute_tolerance": float(check["absolute_tolerance"]),
        "relative_tolerance": float(check["relative_tolerance"]),
        "maximum_absolute_error": gradient.maximum_absolute_error,
        "maximum_relative_error": gradient.maximum_relative_error,
        "passed": True,
    }
    synthetic_summary = {
        "rows": rows,
        "features": feature_count,
        "initial_loss": model.history_[0].loss,
        "final_loss": model.history_[-1].loss,
        "loss_reduction_ratio": (
            model.history_[-1].loss / model.history_[0].loss
        ),
        "iterations_completed": model.history_[-1].iteration,
        "converged": model.converged_,
        "maximum_coefficient_error": maximum_coefficient_error,
        "bias_absolute_error": bias_error,
        "coefficient_tolerance": tolerance,
        "passed": True,
    }
    return gradient_summary, synthetic_summary
