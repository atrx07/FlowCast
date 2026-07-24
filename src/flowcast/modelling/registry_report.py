"""Human-readable report rendering for the Step 14 classical registry."""

from __future__ import annotations

from typing import Any


def render_registry_report(
    registry: dict[str, Any],
    prediction_index: dict[str, Any],
) -> str:
    """Render the task-aware scoreboard and frozen selection rationale."""

    lines = [
        "# FlowCast Classical Model Registry",
        "",
        "## Contract",
        "",
        f"- Version: `{registry['version']}`.",
        (
            f"- Coverage: {len(registry['entries'])} target/horizon entries "
            "across five required outputs and four forecast horizons."
        ),
        (
            f"- Prediction mapping: {prediction_index['prediction_rows']:,} "
            "persisted validation/test rows indexed in place; no data was copied."
        ),
        (
            "- All selections came from frozen validation evidence. Test metrics "
            "are reported honestly and were not used to change a winner."
        ),
        "",
        "## Combined scoreboard",
        "",
        (
            "| Target | Horizon | Family | Primary metric | Validation | Test | "
            "Acceptance |"
        ),
        "|---|---:|---|---|---:|---:|---|",
    ]
    for entry in registry["entries"]:
        metric = entry["primary_metric"]
        acceptance = entry["acceptance"]
        if acceptance is None:
            status = "not specified"
        else:
            status = "met" if acceptance["met"] else "not met"
        lines.append(
            "| {target} | {horizon} min | {family} | {metric} | "
            "{validation:.6g} | {test:.6g} | {status} |".format(
                target=entry["target"],
                horizon=entry["horizon_minutes"],
                family=entry["selected_family"],
                metric=metric,
                validation=entry["metrics"]["validation"][metric],
                test=entry["metrics"]["test"][metric],
                status=status,
            )
        )
    lines.extend(["", "## Selection rationale", ""])
    for entry in registry["entries"]:
        lines.append(f"### {entry['job_id']}")
        lines.append("")
        lines.append(entry["selection_rationale"])
        lines.append("")
    lines.extend(["## Acceptance summary", ""])
    for target in ("volume", "congestion", "accident"):
        selected = [
            entry for entry in registry["entries"] if entry["target"] == target
        ]
        met = sum(bool(entry["acceptance"]["met"]) for entry in selected)
        acceptance = selected[0]["acceptance"]
        operator = (
            "<="
            if acceptance["operator"] == "less_than_or_equal"
            else ">="
        )
        lines.append(
            f"- {target}: {met}/{len(selected)} horizons met "
            f"`{acceptance['metric']} {operator} {acceptance['threshold']}`; "
            "observed test values remain frozen and visible."
        )
    lines.extend(
        [
            "",
            "## Governance",
            "",
            (
                "- Each entry records its model, model card, source predictions, "
                "selection manifest, preprocessing version, feature schema hash, "
                "processed data hash, seed, windows, metrics, and limitations."
            ),
            (
                "- The verified loader checks the registry, both upstream "
                "summaries, every referenced artifact, and the independent "
                "registry configuration before returning a model."
            ),
            (
                "- Runtime and interpretability are supplied as operating context; "
                "they do not override the validation-metric winner."
            ),
            "",
        ]
    )
    return "\n".join(lines)
