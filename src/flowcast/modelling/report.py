"""Render the generated Step 10 split and preprocessing report."""

from __future__ import annotations

from typing import Any


def render_modeling_prep_report(summary: dict[str, Any]) -> str:
    """Render human-readable evidence from the canonical Step 10 summary."""

    split = summary["split"]
    lines = [
        "# FlowCast Frozen Split and Preprocessing Report",
        "",
        f"- Contract: `{summary['contract_version']}`",
        f"- Version: `{summary['version']}`",
        f"- Processed input: `{summary['input_processed_version']}`",
        f"- Feature input: `{summary['input_feature_version']}`",
        "",
        "## Chronological partitions",
        "",
        "| Partition | Start | End | Timestamps | Rows | Share |",
        "|---|---|---|---:|---:|---:|",
    ]
    for name in ("train", "validation", "test"):
        record = split["partitions"][name]
        lines.append(
            f"| {name.title()} | {record['start']} | {record['end']} | "
            f"{record['timestamp_count']:,} | {record['row_count']:,} | "
            f"{record['ratio_of_timestamps']:.2%} |"
        )
    lines.extend(
        [
            "",
            "Every origin is assigned exactly once. A target is eligible only when "
            "its future timestamp stays inside the origin partition and its "
            "target-specific availability mask is true.",
            "",
            "## Time-series cross-validation",
            "",
            "Five expanding-window folds live wholly inside training. Each uses a "
            "four-window gap, covering the maximum 120-minute forecast horizon, "
            "followed by a seven-day validation window.",
            "",
            "| Fold | Train end | Gap | Validation start | Validation end |",
            "|---:|---|---:|---|---|",
        ]
    )
    for record in summary["cross_validation"]["folds"]:
        lines.append(
            f"| {record['fold']} | {record['train_end']} | "
            f"{record['gap_timestamp_count']} windows | "
            f"{record['validation_start']} | {record['validation_end']} |"
        )
    lines.extend(
        [
            "",
            "## Feature and preprocessing contract",
            "",
            f"The schema contains {summary['preprocessing']['feature_count']} "
            "origin-time features from the Step 07 manifest. Keys, raw lineage, "
            "timestamps, targets, and availability masks are excluded.",
            "",
            "| Family | Input features | Output features | Numeric | Bounded |",
            "|---|---:|---:|---|---|",
        ]
    )
    for family, record in summary["preprocessing"]["families"].items():
        policy = record["policy"]
        lines.append(
            f"| {family} | {record['input_feature_count']} | "
            f"{record['output_feature_count']} | {policy['numeric_scaling']} | "
            f"{policy['bounded_scaling']} |"
        )
    lines.extend(
        [
            "",
            "All imputers, encoders, and scalers above were fit on training rows "
            "only. Validation was transform-only. The test partition is sealed by "
            "default and requires the explicit `final_evaluation` purpose.",
            "Training-only class counts, balanced weights, and accident "
            "`scale_pos_weight` values are persisted in the feature schema for "
            "later classifiers; validation and test labels do not influence them.",
            "",
            "## Target eligibility",
            "",
            "| Target | Horizon | Train | Validation | Test |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for target, partitions in split["target_coverage"].items():
        horizon = target.rsplit("_h", 1)[-1]
        lines.append(
            f"| `{target}` | {horizon} | "
            f"{partitions['train']['eligible_rows']:,} | "
            f"{partitions['validation']['eligible_rows']:,} | "
            f"{partitions['test']['eligible_rows']:,} |"
        )
    lines.extend(
        [
            "",
            "This report is generated from the hash-verified processed dataset, "
            "EDA lineage, frozen model config, split assignments, CV folds, and "
            "fitted preprocessing metadata; edit the pipeline, not this report.",
            "",
        ]
    )
    return "\n".join(lines)
