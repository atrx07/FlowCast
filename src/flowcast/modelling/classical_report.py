"""Generated Markdown and deterministic CSV helpers for Step 12."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    """Write a deterministic, human-readable CSV artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        frame.to_csv(index=False, lineterminator="\n", float_format="%.12g"),
        encoding="utf-8",
        newline="\n",
    )


def render_classical_regression_report(summary: dict[str, Any]) -> str:
    """Render the canonical Step 12 summary into Markdown."""

    lines = [
        "# FlowCast Classical Regression",
        "",
        "## Evaluation contract",
        "",
        (
            f"- Version: `{summary['version']}`; seed: "
            f"`{summary['configuration']['seed']}`."
        ),
        (
            f"- Jobs: {summary['coverage']['job_count']} "
            f"({summary['coverage']['target_count']} targets x "
            f"{summary['coverage']['horizon_count']} horizons)."
        ),
        (
            f"- Search: {summary['search']['candidate_count']} candidate "
            f"configurations across {summary['search']['family_count']} required "
            f"families and {summary['search']['fold_count']} expanding folds."
        ),
        (
            "- Candidate hyperparameters were selected by mean CV RMSE; model "
            "family was selected by validation RMSE."
        ),
        (
            "- The selection manifest was persisted before the single explicit "
            "final-evaluation test load."
        ),
        "",
        "## Frozen hold-out scoreboard",
        "",
        (
            "| Target | Horizon | Selected family | Validation RMSE | Test RMSE | "
            "Test MAE | Test MAPE | Test R-squared |"
        ),
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary["scoreboard"]:
        lines.append(
            "| {target} | {horizon} | {family} | {validation:.4f} | "
            "{rmse:.4f} | {mae:.4f} | {mape:.3f}% | {r2:.4f} |".format(
                target=row["target"],
                horizon=row["horizon_minutes"],
                family=row["selected_family"],
                validation=row["validation"]["rmse"],
                rmse=row["test"]["rmse"],
                mae=row["test"]["mae"],
                mape=row["test"]["mape_percent"],
                r2=row["test"]["r_squared"],
            )
        )
    lines.extend(
        [
            "",
            "## Coverage and persistence",
            "",
            (
                f"- Required family/task CV results: "
                f"{summary['coverage']['required_family_job_pairs']} of "
                f"{summary['coverage']['required_family_job_pairs']}."
            ),
            (
                f"- Selected reloadable pipelines: "
                f"{summary['coverage']['selected_model_count']}."
            ),
            (
                f"- Machine-readable model cards: "
                f"{summary['coverage']['model_card_count']}; Markdown model cards: "
                f"{summary['coverage']['model_card_count']}."
            ),
            (
                f"- Persisted prediction rows: "
                f"{summary['coverage']['prediction_rows']} across validation and test."
            ),
            "",
            "## Runtime",
            "",
            (
                f"- CV fit time: {summary['runtime']['cv_fit_seconds']:.3f}s; "
                f"CV prediction time: "
                f"{summary['runtime']['cv_prediction_seconds']:.3f}s."
            ),
            (
                f"- Full-training family fit time: "
                f"{summary['runtime']['validation_fit_seconds']:.3f}s; "
                f"validation prediction time: "
                f"{summary['runtime']['validation_prediction_seconds']:.3f}s."
            ),
            (
                f"- Frozen test prediction time: "
                f"{summary['runtime']['test_prediction_seconds']:.3f}s."
            ),
            "",
            "## Limitations",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in summary["limitations"])
    return "\n".join(lines) + "\n"


def render_model_card(card: dict[str, Any]) -> str:
    """Render one selected regression model card from JSON metadata."""

    validation = card["metrics"]["validation"]
    test = card["metrics"]["test"]
    lines = [
        f"# Model Card: {card['job_id']}",
        "",
        "## Identity",
        "",
        f"- Model version: `{card['model_version']}`.",
        f"- Target: `{card['target']['column']}`.",
        (
            f"- Horizon: {card['target']['horizon_windows']} windows "
            f"({card['target']['horizon_minutes']} minutes)."
        ),
        f"- Selected family: `{card['selection']['family']}`.",
        f"- Candidate: `{card['selection']['candidate_id']}`.",
        f"- Seed: `{card['seed']}`.",
        "",
        "## Selection and data",
        "",
        (
            "- Hyperparameters were selected by mean RMSE across all five frozen "
            "training-only expanding-window folds."
        ),
        (
            "- The estimator family was selected by validation RMSE before the "
            "test split was opened."
        ),
        (
            f"- Training window: `{card['data']['train_start']}` through "
            f"`{card['data']['train_end']}` "
            f"({card['data']['train_rows']} eligible rows)."
        ),
        (
            f"- Validation window: `{card['data']['validation_start']}` through "
            f"`{card['data']['validation_end']}` "
            f"({card['data']['validation_rows']} eligible rows)."
        ),
        (
            f"- Test window: `{card['data']['test_start']}` through "
            f"`{card['data']['test_end']}` "
            f"({card['data']['test_rows']} eligible rows)."
        ),
        (
            f"- Input features: {card['features']['input_feature_count']}; "
            f"transformed features: {card['features']['output_feature_count']}."
        ),
        (
            f"- Preprocessing version: "
            f"`{card['features']['preprocessing_version']}`."
        ),
        "",
        "## Hyperparameters",
        "",
        "```json",
        card["selection"]["parameters_json"],
        "```",
        "",
        "## Metrics",
        "",
        "| Split | RMSE | MAE | MAPE | R-squared | Rows |",
        "|---|---:|---:|---:|---:|---:|",
        (
            f"| Validation | {validation['rmse']:.4f} | "
            f"{validation['mae']:.4f} | {validation['mape_percent']:.3f}% | "
            f"{validation['r_squared']:.4f} | {validation['rows']} |"
        ),
        (
            f"| Test | {test['rmse']:.4f} | {test['mae']:.4f} | "
            f"{test['mape_percent']:.3f}% | {test['r_squared']:.4f} | "
            f"{test['rows']} |"
        ),
        "",
        "## Lineage and artifacts",
        "",
        f"- Processed data SHA-256: `{card['lineage']['processed_sha256']}`.",
        f"- Feature schema SHA-256: `{card['lineage']['feature_schema_sha256']}`.",
        f"- Selection manifest SHA-256: `{card['lineage']['selection_sha256']}`.",
        f"- Pipeline: `{card['artifacts']['model']['path']}`.",
        f"- Predictions: `{card['artifacts']['predictions']['path']}`.",
        "",
        "## Limitations",
        "",
    ]
    lines.extend(f"- {item}" for item in card["limitations"])
    return "\n".join(lines) + "\n"
