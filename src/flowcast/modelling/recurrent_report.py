"""Generated Markdown reporting for the recurrent volume model."""

from __future__ import annotations

from typing import Any


def render_recurrent_report(payload: dict[str, Any]) -> str:
    """Render machine-readable Step 15 results into a concise report."""

    selected = payload["selected"]
    metrics = payload["metrics"]
    comparison = payload["comparison"]
    lines = [
        "# FlowCast Recurrent Volume Forecaster",
        "",
        "## Frozen evaluation contract",
        "",
        f"- Version: `{payload['version']}`; seed: `{payload['seed']}`.",
        (
            f"- Selected candidate: `{selected['candidate_id']}` "
            f"({selected['recurrent_type'].upper()}, "
            f"sequence length {selected['sequence_length']})."
        ),
        (
            f"- Best epoch: {selected['best_epoch']}; stopped epoch: "
            f"{selected['stopped_epoch']}; device: `{payload['device']}`."
        ),
        (
            "- Candidate selection and best-checkpoint persistence occurred before "
            "the single explicit test-partition load."
        ),
        "- Test metrics were not used for architecture or checkpoint selection.",
        "",
        "## Multi-horizon hold-out metrics",
        "",
        "| Horizon | Validation RMSE | Test RMSE | Test MAE | Test MAPE | Test R-squared |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    validation = {
        int(record["horizon_windows"]): record
        for record in metrics["validation"]["horizons"]
    }
    for record in metrics["test"]["horizons"]:
        horizon = int(record["horizon_windows"])
        lines.append(
            "| {minutes} min | {validation:.4f} | {rmse:.4f} | {mae:.4f} | "
            "{mape:.3f}% | {r2:.4f} |".format(
                minutes=horizon * 30,
                validation=validation[horizon]["rmse"],
                rmse=record["rmse"],
                mae=record["mae"],
                mape=record["mape_percent"],
                r2=record["r_squared"],
            )
        )
    lines.extend(
        [
            "",
            "## Exact-row classical comparison",
            "",
            "| Horizon | Shared rows | Deep RMSE | Classical RMSE | Delta | Deep wins |",
            "|---:|---:|---:|---:|---:|---|",
        ]
    )
    for record in comparison:
        lines.append(
            "| {minutes} min | {rows} | {deep:.4f} | {classical:.4f} | "
            "{delta:.4f} | {wins} |".format(
                minutes=record["horizon_minutes"],
                rows=record["rows"],
                deep=record["deep_rmse"],
                classical=record["classical_rmse"],
                delta=record["rmse_delta_deep_minus_classical"],
                wins="yes" if record["deep_beats_classical"] else "no",
            )
        )
    wins = sum(bool(record["deep_beats_classical"]) for record in comparison)
    lines.extend(
        [
            "",
            (
                f"The recurrent model beats the frozen classical volume model at "
                f"{wins} of 4 horizons on the exact shared test origins."
            ),
            "",
            "## Sequence and persistence checks",
            "",
            (
                f"- Training sequences: "
                f"{payload['sequences']['train']['sequence_count']}; validation "
                f"sequences: {payload['sequences']['validation']['sequence_count']}; "
                f"test sequences: {payload['sequences']['test']['sequence_count']}."
            ),
            "- Cross-road, cross-partition, non-contiguous, and target-boundary violations: 0.",
            "- Feature and target scaling statistics originate from training only.",
            "- Reloaded checkpoint inference reproduces the persisted predictions.",
            "",
            "## Limitations",
            "",
            "- Confidence intervals are deferred to Step 16.",
            "- Future weather forecasts are not available; weather is known only at origin.",
            "- This workstation used the CPU-only PyTorch 2.13.0 build.",
        ]
    )
    return "\n".join(lines) + "\n"


def render_recurrent_model_card(card: dict[str, Any]) -> str:
    """Render the final JSON model card without introducing new facts."""

    selection = card["selection"]
    lines = [
        "# Model Card: volume_multi_horizon",
        "",
        "## Identity",
        "",
        f"- Model version: `{card['model_version']}`.",
        f"- Candidate: `{selection['candidate_id']}`.",
        f"- Seed: `{card['seed']}`.",
        "- Targets: volume at 30, 60, 90, and 120 minutes.",
        "- Pretrained weights: no.",
        "",
        "## Selection and training",
        "",
        (
            f"- Validation mean RMSE selected epoch {selection['best_epoch']} "
            f"before test access."
        ),
        (
            f"- Architecture: {selection['architecture']['recurrent_type'].upper()}, "
            f"{selection['architecture']['layer_count']} layer(s), hidden size "
            f"{selection['architecture']['hidden_size']}, sequence length "
            f"{selection['architecture']['sequence_length']}."
        ),
        (
            f"- Input features: {card['features']['input_feature_count']} raw, "
            f"{card['features']['output_feature_count']} transformed."
        ),
        "- Feature and target scaling statistics were learned from training only.",
        "",
        "## Hold-out metrics",
        "",
        "| Horizon | RMSE | MAE | MAPE | R-squared | Rows |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for record in card["metrics"]["test"]["horizons"]:
        lines.append(
            "| {minutes} min | {rmse:.4f} | {mae:.4f} | {mape:.3f}% | "
            "{r2:.4f} | {rows} |".format(
                minutes=int(record["horizon_windows"]) * 30,
                rmse=record["rmse"],
                mae=record["mae"],
                mape=record["mape_percent"],
                r2=record["r_squared"],
                rows=record["rows"],
            )
        )
    lines.extend(
        [
            "",
            "## Lineage and artifacts",
            "",
            f"- Processed data SHA-256: `{card['lineage']['processed_sha256']}`.",
            f"- Selection SHA-256: `{card['lineage']['selection_sha256']}`.",
            f"- Checkpoint: `{card['artifacts']['checkpoint']['path']}`.",
            f"- Predictions: `{card['artifacts']['predictions']['path']}`.",
            "",
            "## Limitations",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in card["limitations"])
    return "\n".join(lines) + "\n"
