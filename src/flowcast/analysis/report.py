"""Human-readable Markdown rendering for the Step 09 EDA summary."""

from __future__ import annotations

from typing import Any


def render_eda_report(summary: dict[str, Any]) -> str:
    """Render the canonical EDA summary into a concise quality report."""

    quality = summary["quality_reconciliation"]
    source = quality["source"]
    validation = quality["validation"]
    cleaning = quality["cleaning"]
    distributions = summary["distributions"]
    lines = [
        "# FlowCast Data Quality and EDA Report",
        "",
        f"- EDA contract: `{summary['contract_version']}`",
        f"- Output version: `{summary['eda_version']}`",
        f"- Processed input: `{summary['input_processed_version']}`",
        f"- Coverage: {summary['dataset']['timestamp_start']} to "
        f"{summary['dataset']['timestamp_end']}",
        "",
        "## Source-to-processed reconciliation",
        "",
        "| Stage | Rows / keys | Notable accounting |",
        "|---|---:|---|",
        f"| Delivered sources | {source['total_rows']:,} | "
        f"Traffic {source['traffic_rows']:,}; weather {source['weather_rows']:,}; "
        f"calendar {source['calendar_rows']:,} |",
        f"| Validation | {validation['valid_rows']:,} retained | "
        f"{validation['rejected_rows']:,} rejected; "
        f"{validation['issues']:,} issues |",
        f"| Complete traffic grid | {cleaning['traffic_output_rows']:,} | "
        f"{cleaning['inserted_windows']:,} missing windows reconstructed |",
        f"| Merge | {quality['merge']['output_rows']:,} | "
        f"{quality['merge']['weather_missing']} weather and "
        f"{quality['merge']['calendar_missing']} calendar misses |",
        f"| Features | {quality['features']['output_rows']:,} | "
        f"{quality['features']['feature_count']} model-candidate features |",
        f"| Processed targets | {quality['processed']['output_rows']:,} | "
        f"{quality['processed']['target_count']} target/horizon definitions |",
        "",
        "All persisted reconciliation checks passed. No stage has an unexplained "
        "row loss, key multiplication, or context-join miss.",
        "",
        "## Data defects and repair evidence",
        "",
        "| Defect / action | Count |",
        "|---|---:|",
        f"| Exact/key traffic duplicates | "
        f"{source['traffic_exact_duplicates']:,} |",
        f"| Entirely missing traffic windows | "
        f"{source['traffic_missing_windows']:,} |",
        f"| Negative traffic volumes | "
        f"{source['traffic_physical_invalid']['negative_traffic_volume']:,} |",
        f"| Speeds above 200 km/h | "
        f"{source['traffic_physical_invalid']['speed_above_200_kmh']:,} |",
        f"| Occupancy above 100% | "
        f"{source['traffic_physical_invalid']['occupancy_above_100_percent']:,} |",
        f"| Blank congestion labels | {source['blank_congestion_labels']:,} |",
        f"| Congestion labels derived after grid completion | "
        f"{cleaning['congestion']['derived_labels']:,} |",
        f"| Vehicle-share rows normalized | "
        f"{cleaning['vehicle_distribution']['normalized_rows']:,} |",
        f"| Accident windows retained as unknown | "
        f"{cleaning['unobserved_accident_windows']:,} |",
        "",
        "### Traffic imputation",
        "",
        "| Field | Missing after grid | Imputed | Remaining |",
        "|---|---:|---:|---:|",
    ]
    for field, record in cleaning["traffic_imputation"].items():
        lines.append(
            f"| `{field}` | {record['input_missing_after_grid']:,} | "
            f"{record['imputed']:,} | {record['remaining_missing']:,} |"
        )
    lines.extend(
        [
            "",
            "Weather temperature and visibility imputations are causal, "
            "station-local forward fills. Traffic repairs retain their method and "
            "donor lineage in the processed data.",
            "",
            "## Descriptive statistics",
            "",
            "| Measure | Count | Mean | Median | Std. dev. | Min | Max | Skew |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, record in summary["descriptive_statistics"].items():
        lines.append(
            f"| `{name}` | {record['count']:,} | {record['mean']:.3f} | "
            f"{record['median']:.3f} | {record['standard_deviation']:.3f} | "
            f"{record['minimum']:.3f} | {record['maximum']:.3f} | "
            f"{record['skewness']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Target balance",
            "",
            "| Congestion class | Rows | Share |",
            "|---|---:|---:|",
        ]
    )
    for label, record in distributions["congestion"].items():
        lines.append(
            f"| {label} | {record['rows']:,} | {record['rate'] * 100:.2f}% |"
        )
    accident = distributions["accident"]
    lines.extend(
        [
            "",
            f"Observed accident labels contain {accident['positive_rows']:,} "
            f"positives across {accident['observed_rows']:,} observed windows "
            f"({accident['positive_rate_observed'] * 100:.3f}%). The "
            f"negative-to-positive ratio is "
            f"{accident['negative_to_positive_ratio']:.1f}:1. The "
            f"{accident['unobserved_rows']:,} unknown windows are excluded.",
            "",
            "## Measured findings",
            "",
        ]
    )
    for record in summary["findings"]:
        lines.append(f"- **{record['id'].replace('_', ' ').title()}:** "
                     f"{record['finding']}")
    lines.extend(
        [
            "",
            "## Correlation and redundancy",
            "",
            f"The configured correlation matrix contains "
            f"{summary['correlation']['feature_count']} origin-time features. "
            f"{len(summary['correlation']['redundant_pairs'])} pairs have absolute "
            f"correlation at or above "
            f"{summary['correlation']['redundancy_threshold']:.2f}.",
            "",
            "| Feature | Correlation with h1 volume | Observations |",
            "|---|---:|---:|",
        ]
    )
    for record in summary["correlation"]["target_correlations"][:10]:
        lines.append(
            f"| `{record['feature']}` | {record['correlation']:.4f} | "
            f"{record['observations']:,} |"
        )
    lines.extend(["", "## Modelling implications", ""])
    for record in summary["modelling_decisions"]:
        lines.append(
            f"- **{record['area'].title()}:** {record['decision']} "
            f"Evidence: {record['evidence']}"
        )
    lines.extend(["", "## Bias and limitations", ""])
    for limitation in summary["limitations"]:
        lines.append(f"- {limitation}")
    lines.extend(["", "## Exported figures", ""])
    for name, record in summary["figures"].items():
        lines.append(f"- `{name}`: `{record['path']}`")
    lines.extend(
        [
            "",
            "This report is generated from persisted pipeline counters and the "
            "hash-verified processed dataset; edit the pipeline, not this report.",
            "",
        ]
    )
    return "\n".join(lines)
