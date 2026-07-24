"""Generated Markdown for Step 13 classification results and model cards."""

from __future__ import annotations

from typing import Any


def render_classification_report(summary: dict[str, Any]) -> str:
    """Render the canonical classification summary into Markdown."""

    lines = [
        "# FlowCast Classical Classification",
        "",
        "## Evaluation contract",
        "",
        (
            f"- Version: `{summary['version']}`; seed: "
            f"`{summary['configuration']['seed']}`."
        ),
        (
            f"- Jobs: {summary['coverage']['job_count']} "
            f"({summary['coverage']['task_count']} tasks x "
            f"{summary['coverage']['horizon_count']} horizons)."
        ),
        (
            f"- Search: {summary['search']['candidate_count']} candidates across "
            f"{summary['search']['family_count']} required families and "
            f"{summary['search']['fold_count']} expanding folds."
        ),
        (
            "- Congestion selection uses Macro-F1; accident selection uses "
            "ROC-AUC with PR-AUC visible."
        ),
        (
            "- Calibration and accident thresholds were frozen from "
            "chronological validation evidence before one explicit test load."
        ),
        "",
        "## Frozen hold-out scoreboard",
        "",
        (
            "| Task | Horizon | Family | Calibration | Threshold | Validation "
            "primary | Test primary | Test secondary |"
        ),
        "|---|---:|---|---|---:|---:|---:|---:|",
    ]
    for row in summary["scoreboard"]:
        primary = "macro_f1" if row["task"] == "congestion" else "roc_auc"
        secondary = "macro_recall" if row["task"] == "congestion" else "pr_auc"
        threshold = (
            "-"
            if row["operating_threshold"] is None
            else f"{row['operating_threshold']:.4f}"
        )
        lines.append(
            "| {task} | {horizon} | {family} | {calibration} | {threshold} | "
            "{validation:.4f} | {test:.4f} | {secondary_value:.4f} |".format(
                task=row["task"],
                horizon=row["horizon_minutes"],
                family=row["selected_family"],
                calibration=(
                    "applied" if row["calibration"]["applied"] else "not applied"
                ),
                threshold=threshold,
                validation=row["validation"][primary],
                test=row["test"][primary],
                secondary_value=row["test"][secondary],
            )
        )
    lines.extend(
        [
            "",
            "## Coverage and persistence",
            "",
            (
                f"- Required family/job comparisons: "
                f"{summary['coverage']['required_family_job_pairs']}."
            ),
            (
                f"- Successful CV fold evaluations: "
                f"{summary['coverage']['successful_fold_evaluations']}."
            ),
            (
                f"- Reloadable selected classifiers/model cards: "
                f"{summary['coverage']['selected_model_count']}."
            ),
            (
                f"- Persisted validation/test predictions: "
                f"{summary['coverage']['prediction_rows']} rows."
            ),
            "",
            "## Acceptance targets",
            "",
            (
                f"- Congestion Macro-F1 target met at all horizons: "
                f"`{summary['acceptance']['congestion_all_horizons_met']}`."
            ),
            (
                f"- Accident ROC-AUC target met at all horizons: "
                f"`{summary['acceptance']['accident_all_horizons_met']}`."
            ),
            "",
            "## Limitations",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in summary["limitations"])
    return "\n".join(lines) + "\n"


def render_classification_model_card(card: dict[str, Any]) -> str:
    """Render one selected classifier card from machine-readable metadata."""

    validation = card["metrics"]["validation"]
    test = card["metrics"]["test"]
    primary = card["selection"]["primary_metric"]
    secondary = "macro_recall" if primary == "macro_f1" else "pr_auc"
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
        f"- Class order: `{card['target']['class_order']}`.",
        f"- Selected family: `{card['selection']['family']}`.",
        f"- Candidate: `{card['selection']['candidate_id']}`.",
        f"- Seed: `{card['seed']}`.",
        "",
        "## Selection and data",
        "",
        (
            f"- Hyperparameters were selected by mean CV `{primary}` across all "
            "five frozen training-only folds."
        ),
        (
            f"- The family was selected by validation `{primary}` before test "
            "access."
        ),
        (
            f"- Training: `{card['data']['train_start']}` through "
            f"`{card['data']['train_end']}` "
            f"({card['data']['train_rows']} rows)."
        ),
        (
            f"- Validation: `{card['data']['validation_start']}` through "
            f"`{card['data']['validation_end']}` "
            f"({card['data']['validation_rows']} rows)."
        ),
        (
            f"- Test: `{card['data']['test_start']}` through "
            f"`{card['data']['test_end']}` "
            f"({card['data']['test_rows']} rows)."
        ),
        "",
        "## Probability and operating decision",
        "",
        (
            f"- Sigmoid calibration applied: "
            f"`{card['probability']['calibration']['applied']}` "
            f"({card['probability']['calibration']['reason']})."
        ),
        (
            f"- Accident operating threshold: "
            f"`{card['probability']['operating_threshold']}`."
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
        f"| Split | {primary} | {secondary} | Rows |",
        "|---|---:|---:|---:|",
        (
            f"| Validation | {validation[primary]:.4f} | "
            f"{validation[secondary]:.4f} | {validation['rows']} |"
        ),
        (
            f"| Test | {test[primary]:.4f} | {test[secondary]:.4f} | "
            f"{test['rows']} |"
        ),
        "",
        "## Lineage and artifacts",
        "",
        f"- Processed data SHA-256: `{card['lineage']['processed_sha256']}`.",
        f"- Feature schema SHA-256: `{card['lineage']['feature_schema_sha256']}`.",
        f"- Selection manifest SHA-256: `{card['lineage']['selection_sha256']}`.",
        f"- Classifier: `{card['artifacts']['model']['path']}`.",
        f"- Predictions: `{card['artifacts']['predictions']['path']}`.",
        "",
        "## Limitations",
        "",
    ]
    lines.extend(f"- {item}" for item in card["limitations"])
    return "\n".join(lines) + "\n"
