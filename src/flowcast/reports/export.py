"""CSV and self-contained HTML exports for verified forecast batches."""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
import json
from pathlib import Path
from typing import Any

import pandas as pd

from flowcast.data.artifacts import (
    artifact_record,
    validate_artifact_version,
    verify_artifact_record,
    write_json,
)
from flowcast.inference.artifacts import load_prediction_batch
from flowcast.inference.inputs import load_verified_inference_context
from flowcast.modelling.classical_report import write_csv
from flowcast.modelling.registry_artifacts import record_path
from flowcast.reports.insights import prediction_insights
from flowcast.settings import Settings


@dataclass(frozen=True)
class ReportPaths:
    """Output paths for one report export."""

    directory: Path
    csv_path: Path
    html_path: Path
    manifest_path: Path


def report_paths(
    settings: Settings,
    version: str,
    request_id: str,
    *,
    output_root: Path | None = None,
) -> ReportPaths:
    """Return report paths without creating them."""

    root = output_root if output_root is not None else settings.artifacts_dir
    directory = (
        root
        / "reports"
        / validate_artifact_version(version)
        / validate_artifact_version(request_id)
    )
    return ReportPaths(
        directory=directory,
        csv_path=directory / "predictions.csv",
        html_path=directory / "report.html",
        manifest_path=directory / "manifest.json",
    )


def _metrics_limitations(context: Any) -> tuple[list[str], list[str]]:
    diagnostics = context.confidence.summary["diagnostics"]
    congestion = diagnostics["congestion"]["test_macro_f1_by_horizon"]
    accident = diagnostics["accident"]["test_roc_auc_by_horizon"]
    deltas = diagnostics["paired_volume"][
        "rmse_delta_deep_minus_classical_by_horizon"
    ]
    evidence = [
        (
            "Congestion test Macro-F1 by horizon: "
            + ", ".join(f"h{key}={value:.4f}" for key, value in congestion.items())
        ),
        (
            "Accident-risk test ROC-AUC by horizon: "
            + ", ".join(f"h{key}={value:.4f}" for key, value in accident.items())
        ),
        (
            "Recurrent minus classical volume RMSE by horizon: "
            + ", ".join(f"h{key}={value:+.4f}" for key, value in deltas.items())
        ),
    ]
    return evidence, [str(value) for value in context.confidence.summary["limitations"]]


def _table(headers: list[str], rows: list[list[str]]) -> str:
    head = "".join(f"<th>{escape(value)}</th>" for value in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{escape(value)}</td>" for value in row) + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _render_html(
    manifest: dict[str, Any],
    insights: dict[str, Any],
    evidence: list[str],
    limitations: list[str],
) -> str:
    risk_rows = [
        [
            f"{row['horizon_minutes']} min",
            row["road_id"],
            row["road_name"],
            f"{row['probability']:.4%}",
            row["risk_band"],
        ]
        for row in insights["highest_accident_risk_by_horizon"]
    ]
    volume_rows = [
        [
            f"{row['horizon_minutes']} min",
            row["road_id"],
            row["road_name"],
            f"{row['predicted_volume']:.2f}",
        ]
        for row in insights["highest_volume_by_horizon"]
    ]
    evidence_html = "".join(f"<li>{escape(value)}</li>" for value in evidence)
    limitation_html = "".join(f"<li>{escape(value)}</li>" for value in limitations)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FlowCast Forecast Report</title>
<style>
body {{ font-family: Arial, sans-serif; color: #17202a; margin: 2rem auto;
max-width: 1050px; line-height: 1.45; padding: 0 1rem; }}
h1, h2 {{ color: #14324a; }} .meta {{ background: #eef4f7; padding: 1rem; }}
.metrics {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: .75rem; }}
.metric {{ border: 1px solid #cad6dc; padding: .8rem; }}
table {{ border-collapse: collapse; width: 100%; margin: .75rem 0 1.5rem; }}
th, td {{ border: 1px solid #cad6dc; padding: .55rem; text-align: left; }}
th {{ background: #14324a; color: white; }} small {{ color: #566573; }}
</style>
</head>
<body>
<h1>FlowCast Forecast Report</h1>
<div class="meta">
<strong>Request:</strong> {escape(str(manifest['request_id']))}<br>
<strong>Origin:</strong> {escape(insights['origin_timestamp'])}<br>
<strong>Roads / horizons:</strong> {insights['road_count']} / {insights['horizon_count']}
</div>
<h2>Forecast summary</h2>
<div class="metrics">
<div class="metric"><strong>Mean volume</strong><br>{insights['mean_predicted_volume']:.2f}</div>
<div class="metric"><strong>Mean speed</strong><br>{insights['mean_predicted_speed']:.2f} km/h</div>
<div class="metric"><strong>Mean travel time</strong><br>{insights['mean_predicted_travel_time']:.2f} min</div>
</div>
<h2>Highest accident-risk segment</h2>
{_table(['Horizon', 'Road', 'Name', 'Probability', 'Band'], risk_rows)}
<h2>Highest predicted-volume segment</h2>
{_table(['Horizon', 'Road', 'Name', 'Volume'], volume_rows)}
<h2>Frozen evaluation evidence</h2>
<ul>{evidence_html}</ul>
<h2>Limitations</h2>
<ul>{limitation_html}</ul>
<p><small>All values derive from the verified prediction batch and frozen model
artifacts. This report does not retrain, recalibrate, or retune a model.</small></p>
</body>
</html>
"""


def build_prediction_reports(
    settings: Settings,
    prediction_manifest_path: Path,
    *,
    output_root: Path | None = None,
) -> ReportPaths:
    """Verify a batch and export full CSV plus self-contained HTML."""

    batch = load_prediction_batch(settings, prediction_manifest_path)
    context = load_verified_inference_context(settings)
    insights = prediction_insights(batch.frame)
    evidence, limitations = _metrics_limitations(context)
    paths = report_paths(
        settings,
        batch.manifest["version"],
        batch.manifest["request_id"],
        output_root=output_root,
    )
    write_csv(batch.frame, paths.csv_path)
    paths.html_path.parent.mkdir(parents=True, exist_ok=True)
    paths.html_path.write_text(
        _render_html(batch.manifest, insights, evidence, limitations),
        encoding="utf-8",
        newline="\n",
    )
    report_manifest = {
        "contract_version": "flowcast_prediction_report_v1",
        "version": batch.manifest["version"],
        "request_id": batch.manifest["request_id"],
        "prediction_manifest": artifact_record(prediction_manifest_path, settings),
        "insights": insights,
        "evaluation_evidence": evidence,
        "limitations": limitations,
        "artifacts": {
            "csv": artifact_record(paths.csv_path, settings),
            "html": artifact_record(paths.html_path, settings),
        },
    }
    write_json(report_manifest, paths.manifest_path)
    return paths


def verify_prediction_reports(
    settings: Settings,
    report_manifest_path: Path,
) -> dict[str, Any]:
    """Verify a report manifest, its source batch, and both exported files."""

    if not report_manifest_path.is_file():
        raise FileNotFoundError(
            f"Prediction report manifest is missing: {report_manifest_path}"
        )
    manifest: dict[str, Any] = json.loads(
        report_manifest_path.read_text(encoding="utf-8")
    )
    if manifest.get("contract_version") != "flowcast_prediction_report_v1":
        raise RuntimeError("Unsupported prediction report contract")
    prediction_manifest_path = verify_artifact_record(
        record_path(manifest["prediction_manifest"], settings),
        manifest["prediction_manifest"],
        settings,
    )
    batch = load_prediction_batch(settings, prediction_manifest_path)
    if batch.manifest["request_id"] != manifest.get("request_id"):
        raise RuntimeError("Report request identity changed")
    csv_path = verify_artifact_record(
        record_path(manifest["artifacts"]["csv"], settings),
        manifest["artifacts"]["csv"],
        settings,
    )
    html_path = verify_artifact_record(
        record_path(manifest["artifacts"]["html"], settings),
        manifest["artifacts"]["html"],
        settings,
    )
    exported = pd.read_csv(csv_path)
    if len(exported) != len(batch.frame):
        raise RuntimeError("Report CSV row count changed")
    if list(exported.columns) != list(batch.frame.columns):
        raise RuntimeError("Report CSV schema changed")
    html = html_path.read_text(encoding="utf-8")
    if str(manifest["request_id"]) not in html:
        raise RuntimeError("Report HTML no longer identifies its source request")
    return manifest
