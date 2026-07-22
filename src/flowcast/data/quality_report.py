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
