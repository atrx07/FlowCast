"""Diagnostics and human-readable reporting for Step 16."""

from __future__ import annotations

from typing import Any

import pandas as pd


def build_diagnostics(
    coverage: pd.DataFrame,
    slices: pd.DataFrame,
    confusions: pd.DataFrame,
    paired: pd.DataFrame,
    reliability: pd.DataFrame,
    risk_bands: pd.DataFrame,
    confidence_level: float,
) -> dict[str, Any]:
    """Summarize hold-out calibration, failure modes, and paired evidence."""

    test_coverage = coverage.loc[coverage["split"].eq("test")]
    overall = slices.loc[
        slices["dimension"].eq("overall") & slices["split"].eq("test")
    ]
    congestion = overall.loc[overall["task_type"].eq("congestion")]
    accident = overall.loc[overall["task_type"].eq("accident")]
    paired_overall = paired.loc[
        paired["dimension"].eq("overall") & paired["split"].eq("test")
    ]
    test_reliability = reliability.loc[
        reliability["split"].eq("test") & reliability["bin_index"].eq(0)
    ]
    test_risk = risk_bands.loc[
        risk_bands["split"].eq("test") & risk_bands["rows"].gt(0)
    ]
    test_confusions = confusions.loc[
        confusions["dimension"].eq("overall")
        & confusions["split"].eq("test")
        & confusions["actual_label"].ne(confusions["predicted_label"])
    ]
    dominant_confusion: dict[str, Any] | None = None
    if not test_confusions.empty:
        record = test_confusions.sort_values(
            ["rows", "horizon_windows"],
            ascending=[False, True],
            kind="stable",
        ).iloc[0]
        dominant_confusion = {
            "horizon_windows": int(record["horizon_windows"]),
            "actual_label": str(record["actual_label"]),
            "predicted_label": str(record["predicted_label"]),
            "rows": int(record["rows"]),
        }
    return {
        "regression_intervals": {
            "nominal_coverage": confidence_level,
            "test_group_count": len(test_coverage),
            "below_nominal_group_count": int(
                test_coverage["interval_coverage"].lt(confidence_level).sum()
            ),
            "minimum_test_coverage": float(
                test_coverage["interval_coverage"].min()
            ),
            "maximum_test_coverage": float(
                test_coverage["interval_coverage"].max()
            ),
        },
        "congestion": {
            "test_macro_f1_by_horizon": {
                str(int(row.horizon_windows)): float(row.macro_f1)
                for row in congestion.itertuples()
            },
            "dominant_off_diagonal_confusion": dominant_confusion,
            "expected_calibration_error_by_horizon": {
                str(int(row.horizon_windows)): float(
                    row.expected_calibration_error
                )
                for row in test_reliability.loc[
                    test_reliability["task"].eq("congestion")
                ].itertuples()
            },
        },
        "accident": {
            "test_roc_auc_by_horizon": {
                str(int(row.horizon_windows)): float(row.roc_auc)
                for row in accident.itertuples()
            },
            "test_pr_auc_by_horizon": {
                str(int(row.horizon_windows)): float(row.pr_auc)
                for row in accident.itertuples()
            },
            "test_prevalence_by_horizon": {
                str(int(row.horizon_windows)): float(row.prevalence)
                for row in accident.itertuples()
            },
            "expected_calibration_error_by_horizon": {
                str(int(row.horizon_windows)): float(
                    row.expected_calibration_error
                )
                for row in test_reliability.loc[
                    test_reliability["task"].eq("accident")
                ].itertuples()
            },
            "risk_band_observed_rates": [
                {
                    "horizon_minutes": int(row.horizon_minutes),
                    "risk_band": str(row.risk_band),
                    "rows": int(row.rows),
                    "observed_event_rate": float(row.observed_event_rate),
                }
                for row in test_risk.itertuples()
            ],
        },
        "paired_volume": {
            "test_horizon_count": len(paired_overall),
            "deep_rmse_win_horizons": int(
                paired_overall["rmse_delta_deep_minus_classical"].lt(0).sum()
            ),
            "rmse_delta_deep_minus_classical_by_horizon": {
                str(int(row.horizon_windows)): float(
                    row.rmse_delta_deep_minus_classical
                )
                for row in paired_overall.itertuples()
            },
            "worst_supported_test_slices": [
                {
                    "horizon_minutes": int(row.horizon_minutes),
                    "dimension": str(row.dimension),
                    "slice_value": str(row.slice_value),
                    "rows": int(row.rows),
                    "rmse_delta_deep_minus_classical": float(
                        row.rmse_delta_deep_minus_classical
                    ),
                }
                for row in paired.loc[
                    paired["split"].eq("test")
                    & paired["dimension"].ne("overall")
                    & paired["sufficient_support"]
                ]
                .nlargest(5, "rmse_delta_deep_minus_classical")
                .itertuples()
            ],
        },
    }


def markdown_report(summary: dict[str, Any], coverage: pd.DataFrame) -> str:
    """Render the concise human-readable confidence report."""

    diagnostics = summary["diagnostics"]
    lines = [
        "# FlowCast Confidence and Error Analysis",
        "",
        "## Contract",
        "",
        "- Regression uncertainty uses validation-only finite-sample "
        "split-conformal absolute residuals.",
        "- Classification uncertainty exposes maximum probability and normalized "
        "entropy from frozen probabilities.",
        "- Accident risk bands are relative to each frozen validation-selected "
        "operating threshold.",
        "- All slices are descriptive; no model, threshold, calibrator, split, or "
        "prediction was changed.",
        "",
        "## Regression test intervals",
        "",
        "| Model | Target | Horizon | RMSE | Coverage | Mean width |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in coverage.loc[coverage["split"].eq("test")].itertuples():
        lines.append(
            f"| {row.model_version} | {row.target} | {row.horizon_minutes} min "
            f"| {row.rmse:.4f} | {row.interval_coverage:.4f} "
            f"| {row.mean_interval_width:.4f} |"
        )
    congestion = diagnostics["congestion"]
    accident = diagnostics["accident"]
    paired = diagnostics["paired_volume"]
    lines.extend(
        [
            "",
            "## Classification and paired-model findings",
            "",
            "- Congestion test Macro-F1 by horizon: "
            f"{congestion['test_macro_f1_by_horizon']}.",
            f"- Accident test ROC-AUC by horizon: "
            f"{accident['test_roc_auc_by_horizon']}.",
            "- Congestion test expected calibration error by horizon: "
            f"{congestion['expected_calibration_error_by_horizon']}.",
            "- Dominant congestion off-diagonal confusion: "
            f"{congestion['dominant_off_diagonal_confusion']}.",
            f"- The recurrent model wins test RMSE on "
            f"{paired['deep_rmse_win_horizons']} of "
            f"{paired['test_horizon_count']} exact paired horizons.",
            f"- Worst supported paired slices: "
            f"{paired['worst_supported_test_slices']}.",
            "",
            "## Interpretation guardrails",
            "",
            "- Subgroups below configured row or positive-event support remain in "
            "the CSV with `sufficient_support=false` and blank metrics.",
            "- Interval coverage is an empirical hold-out diagnostic, not a "
            "guarantee for future distribution shift.",
            "- Low accident prevalence makes PR-AUC, precision, and supported "
            "positive counts essential companions to ROC-AUC.",
            "- Slice differences are associations and must not be treated as causal.",
            "",
        ]
    )
    return "\n".join(lines)
