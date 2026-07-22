"""Render human-readable quality reports from canonical machine summaries."""

from __future__ import annotations

from typing import Any


def render_context_cleaning_markdown(summary: dict[str, Any]) -> str:
    """Render the Step 04 JSON summary as concise generated Markdown."""

    calendar = summary["datasets"]["calendar"]
    weather = summary["datasets"]["weather"]
    temperature = weather["imputation"]["temperature"]
    visibility = weather["imputation"]["visibility"]
    remaining_nulls = sum(weather["remaining_nulls"].values())
    lines = [
        "# FlowCast Context Cleaning Report",
        "",
        f"- Cleaning contract: `{summary['contract_version']}`",
        f"- Output version: `{summary['cleaning_version']}`",
        f"- Input validation version: `{summary['input_validation_version']}`",
        "",
        "## Calendar",
        "",
        "| Check | Result |",
        "|---|---:|",
        (
            "| Rows / unique dates | "
            f"{calendar['output_rows']} / {calendar['unique_dates']} |"
        ),
        f"| Date range | {calendar['date_start']} to {calendar['date_end']} |",
        f"| Public holidays | {calendar['flag_counts']['public_holiday']} |",
        f"| Event days | {calendar['flag_counts']['event_flag']} |",
        f"| Roadwork days | {calendar['flag_counts']['roadwork_flag']} |",
        "",
        "## Weather",
        "",
        "| Check | Result |",
        "|---|---:|",
        (
            "| Rows / unique station-hours | "
            f"{weather['output_rows']} / {weather['unique_station_hours']} |"
        ),
        f"| Stations | {len(weather['station_counts'])} |",
        f"| Temperature imputed | {temperature['imputed']} |",
        f"| Visibility imputed | {visibility['imputed']} |",
        f"| Remaining trusted-field nulls | {remaining_nulls} |",
        "",
        "### Controlled weather vocabulary",
        "",
        "| Label | Rows |",
        "|---|---:|",
    ]
    for label, count in weather["condition_counts"].items():
        lines.append(f"| {label} | {count} |")

    lines.extend(
        [
            "",
            "### Imputation policy",
            "",
            (
                "Temperature and visibility use causal, station-local "
                f"`{temperature['method']}`."
            ),
            (
                "The configured limits are "
                f"{temperature['max_gap_hours']} hours for temperature and "
                f"{visibility['max_gap_hours']} hours for visibility."
            ),
            "Donor source-row lineage is stored beside every imputed value.",
            "No future or cross-station value is used.",
            "",
            (
                "This file is generated from `summary.json`; "
                "edit the pipeline, not this report."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def render_traffic_cleaning_markdown(summary: dict[str, Any]) -> str:
    """Render the Step 05 JSON summary as concise generated Markdown."""

    traffic = summary["dataset"]
    grid = traffic["grid"]
    lines = [
        "# FlowCast Traffic Cleaning Report",
        "",
        f"- Cleaning contract: `{summary['contract_version']}`",
        f"- Output version: `{summary['cleaning_version']}`",
        f"- Input validation version: `{summary['input_validation_version']}`",
        "",
        "## Grid and lineage",
        "",
        "| Check | Result |",
        "|---|---:|",
        f"| Validated input rows | {traffic['input_rows']} |",
        f"| Complete output rows | {traffic['output_rows']} |",
        f"| Roads | {traffic['road_count']} |",
        f"| Inserted missing windows | {grid['inserted_windows']} |",
        f"| Duplicate rows accounted | {traffic['duplicate_rows_accounted']} |",
        (
            "| Metadata inconsistencies | "
            f"{len(grid['metadata_inconsistencies'])} |"
        ),
        "",
        "## Causal recovery",
        "",
        "| Field | Missing after grid | Maximum run | Imputed | Remaining |",
        "|---|---:|---:|---:|---:|",
    ]
    for field, record in traffic["imputation"].items():
        lines.append(
            f"| {field} | {record['input_missing_after_grid']} | "
            f"{record['maximum_missing_run_windows']} | {record['imputed']} | "
            f"{record['remaining_missing']} |"
        )
    lines.extend(
        [
            "",
            "Recovery uses same-row semantic equivalence where configured, then "
            "the previous-day same window, bounded same-road causal forward fill, "
            "and only the configured concurrent station median for unresolved "
            "leading values. Donor rows and timestamps are stored beside repairs.",
            "",
            "## Vehicle distribution and congestion",
            "",
            "| Check | Result |",
            "|---|---:|",
            (
                "| Vehicle-share rows normalized | "
                f"{traffic['vehicle_distribution']['normalized_rows']} |"
            ),
            (
                "| Congestion labels derived | "
                f"{traffic['congestion']['derived_labels']} |"
            ),
            (
                "| Existing-label disagreements | "
                f"{traffic['congestion']['source_disagreements']} |"
            ),
            (
                "| Unobserved accident windows retained as unknown | "
                f"{traffic['unobserved_accident_windows']} |"
            ),
            "",
            "Inserted windows retain a false `_accident_observed` flag and an "
            "unknown accident count; they are not silently relabelled as no incident.",
            "",
            "This file is generated from `traffic_summary.json`; edit the pipeline, "
            "not this report.",
            "",
        ]
    )
    return "\n".join(lines)


def render_source_merge_markdown(summary: dict[str, Any]) -> str:
    """Render the Step 06 JSON summary as concise generated Markdown."""

    dataset = summary["dataset"]
    inputs = dataset["inputs"]
    weather = dataset["joins"]["weather"]
    calendar = dataset["joins"]["calendar"]
    lines = [
        "# FlowCast Source Merge Report",
        "",
        f"- Merge contract: `{summary['contract_version']}`",
        f"- Output version: `{summary['merge_version']}`",
        f"- Input cleaning version: `{summary['input_cleaning_version']}`",
        "",
        "## Cardinality",
        "",
        "| Check | Result |",
        "|---|---:|",
        (
            "| Traffic input rows / keys | "
            f"{inputs['traffic_rows']} / {inputs['traffic_unique_keys']} |"
        ),
        (
            "| Weather input rows / keys | "
            f"{inputs['weather_rows']} / {inputs['weather_unique_keys']} |"
        ),
        (
            "| Calendar input rows / keys | "
            f"{inputs['calendar_rows']} / {inputs['calendar_unique_keys']} |"
        ),
        (
            "| Output rows / keys | "
            f"{dataset['output_rows']} / {dataset['output_unique_keys']} |"
        ),
        f"| Row-count change | {dataset['row_count_change']} |",
        f"| Duplicate output keys | {dataset['duplicate_output_keys']} |",
        "",
        "## Join coverage",
        "",
        "| Context | Cardinality | Matched | Missing |",
        "|---|---|---:|---:|",
        (
            f"| Weather | {weather['cardinality']} | {weather['matched']} | "
            f"{weather['missing']} |"
        ),
        (
            f"| Calendar | {calendar['cardinality']} | {calendar['matched']} | "
            f"{calendar['missing']} |"
        ),
        "",
        "Weather is aligned by station and floored local clock hour. Calendar is "
        "aligned by the normalized local date. Both joins use explicit Pandas "
        "`many_to_one` validation and fail closed on an unexpected miss.",
        "",
        "This file is generated from `summary.json`; edit the pipeline, not "
        "this report.",
        "",
    ]
    return "\n".join(lines)


def render_feature_engineering_markdown(summary: dict[str, Any]) -> str:
    """Render the Step 07 JSON summary as concise generated Markdown."""

    dataset = summary["dataset"]
    null_features = {
        name: count
        for name, count in dataset["feature_null_counts"].items()
        if count
    }
    lines = [
        "# FlowCast Feature Engineering Report",
        "",
        f"- Feature contract: `{summary['contract_version']}`",
        f"- Output version: `{summary['feature_version']}`",
        f"- Input merge version: `{summary['input_merge_version']}`",
        "",
        "## Dataset contract",
        "",
        "| Check | Result |",
        "|---|---:|",
        f"| Input rows | {dataset['input_rows']} |",
        f"| Output rows / keys | {dataset['output_rows']} / "
        f"{dataset['output_unique_keys']} |",
        f"| Row-count change | {dataset['row_count_change']} |",
        f"| Duplicate output keys | {dataset['duplicate_output_keys']} |",
        f"| Model-candidate features | {dataset['feature_count']} |",
        f"| History-available rows | {dataset['history_available_rows']} |",
        f"| History-unavailable rows | {dataset['history_unavailable_rows']} |",
        "",
        "## Expected history nulls",
        "",
        "| Feature | Null rows |",
        "|---|---:|",
    ]
    for name, count in null_features.items():
        lines.append(f"| {name} | {count} |")
    lines.extend(
        [
            "",
            "Lags are computed within each road. Rolling features shift one window "
            "before applying the configured full-width rolling mean or sample "
            "standard deviation, so the current and future rows are excluded.",
            "",
            "All source, imputation, and inserted-window lineage columns remain in "
            "the feature Parquet. The JSON manifest records every model-candidate "
            "feature's dtype, source columns, transform, version, and leakage status.",
            "",
            "This file is generated from `summary.json`; edit the pipeline, not "
            "this report.",
            "",
        ]
    )
    return "\n".join(lines)
