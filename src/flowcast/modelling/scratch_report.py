"""Render the generated Step 11 NumPy regression evidence report."""

from __future__ import annotations

from typing import Any


def _metric_row(name: str, metrics: dict[str, Any]) -> str:
    return (
        f"| {name} | {metrics['rmse']:.4f} | {metrics['mae']:.4f} | "
        f"{metrics['mape_percent']:.4f}% | {metrics['r_squared']:.6f} |"
    )


def render_scratch_linear_report(summary: dict[str, Any]) -> str:
    """Render human-readable evidence from the canonical Step 11 summary."""

    gradient = summary["gradient_check"]
    synthetic = summary["synthetic_proof"]
    training = summary["training"]
    lines = [
        "# FlowCast NumPy Linear Regression Proof",
        "",
        f"- Contract: `{summary['contract_version']}`",
        f"- Version: `{summary['version']}`",
        f"- Demonstration target: `{summary['target']['name']}`",
        "- Purpose: mathematical verification only; no production model selection",
        "- Test partition rows loaded: **0**",
        "",
        "## Mathematics implemented directly",
        "",
        "Prediction uses `X @ w + b`. Mean squared error, its analytical weight "
        "and bias gradients, seeded initialization, and the full-batch update loop "
        "are implemented in `flowcast.modelling.scratch_linear` with NumPy.",
        "",
        "## Gradient and synthetic proofs",
        "",
        f"All {gradient['parameter_count']} parameters passed central finite-"
        f"difference checks. Maximum absolute error: "
        f"`{gradient['maximum_absolute_error']:.3e}`; maximum relative error: "
        f"`{gradient['maximum_relative_error']:.3e}`.",
        "",
        f"Synthetic loss decreased from `{synthetic['initial_loss']:.10f}` to "
        f"`{synthetic['final_loss']:.10f}`. Maximum coefficient error was "
        f"`{synthetic['maximum_coefficient_error']:.3e}` and bias error was "
        f"`{synthetic['bias_absolute_error']:.3e}`.",
        "",
        "## FlowCast data slice",
        "",
        f"The earliest {training['train_rows']:,} eligible training rows "
        f"({training['train_timestamp_start']} through "
        f"{training['train_timestamp_end']}) were selected after chronological "
        "sorting. The unchanged validation partition contributes "
        f"{training['validation_rows']:,} eligible rows. Both estimators consume "
        f"the same {training['input_feature_count']} manifest inputs and "
        f"{training['output_feature_count']} preprocessed columns.",
        "",
        "## Validation comparison",
        "",
        "| Estimator | RMSE | MAE | MAPE | R-squared |",
        "|---|---:|---:|---:|---:|",
        _metric_row("NumPy gradient descent", summary["metrics"]["scratch"]),
        _metric_row("scikit-learn LinearRegression", summary["metrics"]["sklearn"]),
        "",
        f"The scratch loss decreased from `{training['initial_loss']:.6f}` to "
        f"`{training['final_loss']:.6f}` over "
        f"{training['iterations_completed']:,} updates. These validation results "
        "prove the implementation; Step 12 performs model-family training and "
        "selection without changing the frozen split.",
        "",
        "## Limitations",
        "",
    ]
    lines.extend(f"- {item}" for item in summary["limitations"])
    lines.extend(
        [
            "",
            "This report is generated from the canonical JSON, persisted "
            "convergence history, coefficients, validation predictions, and "
            "hash-verified Step 10 lineage; edit the pipeline, not this report.",
            "",
        ]
    )
    return "\n".join(lines)
