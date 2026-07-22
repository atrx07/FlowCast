"""Verified upstream quality lineage and counter reconciliation for Step 09."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from flowcast.data.artifacts import verify_artifact_record
from flowcast.settings import Settings


@dataclass(frozen=True)
class QualitySources:
    """Canonical upstream quality summaries used by the EDA report."""

    audit_path: Path
    validation_path: Path
    context_path: Path
    traffic_path: Path
    merge_path: Path
    feature_path: Path
    processed_path: Path
    audit: dict[str, Any]
    validation: dict[str, Any]
    context: dict[str, Any]
    traffic: dict[str, Any]
    merge: dict[str, Any]
    feature: dict[str, Any]
    processed: dict[str, Any]


def _read(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required quality summary is missing: {path}")
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return payload


def load_verified_quality_sources(settings: Settings) -> QualitySources:
    """Load the complete source-to-processed summary chain and verify hashes."""

    audit_path = (
        settings.artifacts_dir / "audits" / settings.audit_version / "audit.json"
    )
    validation_path = (
        settings.quarantine_dir / settings.validation_version / "summary.json"
    )
    quality = settings.artifacts_dir / "quality"
    context_path = quality / settings.cleaning_version / "summary.json"
    traffic_path = quality / settings.cleaning_version / "traffic_summary.json"
    merge_path = quality / settings.merge_version / "summary.json"
    feature_path = quality / settings.feature_version / "summary.json"
    processed_path = quality / settings.processed_version / "summary.json"

    audit = _read(audit_path)
    validation = _read(validation_path)
    context = _read(context_path)
    traffic = _read(traffic_path)
    merge = _read(merge_path)
    feature = _read(feature_path)
    processed = _read(processed_path)
    verify_artifact_record(feature_path, processed["input_summary"], settings)
    verify_artifact_record(merge_path, feature["input_summary"], settings)
    verify_artifact_record(
        context_path,
        merge["input_summaries"]["context"],
        settings,
    )
    verify_artifact_record(
        traffic_path,
        merge["input_summaries"]["traffic"],
        settings,
    )
    verify_artifact_record(
        validation_path,
        context["input_validation_summary"],
        settings,
    )
    verify_artifact_record(
        validation_path,
        traffic["input_validation_summary"],
        settings,
    )
    manifest_files = audit["raw_manifest"]["files"]
    for record in manifest_files:
        raw_path = settings.root / str(record["copied_path"])
        verify_artifact_record(raw_path, record, settings)
    return QualitySources(
        audit_path=audit_path,
        validation_path=validation_path,
        context_path=context_path,
        traffic_path=traffic_path,
        merge_path=merge_path,
        feature_path=feature_path,
        processed_path=processed_path,
        audit=audit,
        validation=validation,
        context=context,
        traffic=traffic,
        merge=merge,
        feature=feature,
        processed=processed,
    )


def quality_reconciliation(sources: QualitySources) -> dict[str, Any]:
    """Reconcile persisted counters across every completed data stage."""

    raw = sources.audit["datasets"]
    validation = sources.validation
    context = sources.context["datasets"]
    traffic = sources.traffic["dataset"]
    merge = sources.merge["dataset"]
    feature = sources.feature["dataset"]
    processed = sources.processed["dataset"]
    raw_rows = sum(int(dataset["shape"]["rows"]) for dataset in raw.values())
    checks = [
        {
            "name": "raw_validation_input_rows",
            "passed": raw_rows == int(validation["total_input_rows"]),
            "detail": f"{raw_rows} raw rows = validation input rows",
        },
        {
            "name": "validation_row_accounting",
            "passed": int(validation["total_valid_rows"])
            + int(validation["total_rejected_rows"])
            == int(validation["total_input_rows"]),
            "detail": "valid rows + rejected rows = validation input rows",
        },
        {
            "name": "traffic_duplicate_accounting",
            "passed": int(raw["traffic"]["shape"]["rows"])
            - int(traffic["duplicate_rows_accounted"])
            == int(traffic["input_rows"]),
            "detail": "raw traffic - duplicate rows = cleaned traffic input",
        },
        {
            "name": "traffic_grid_accounting",
            "passed": int(traffic["input_rows"])
            + int(traffic["grid"]["inserted_windows"])
            == int(traffic["output_rows"]),
            "detail": "deduplicated traffic + inserted windows = complete grid",
        },
        {
            "name": "context_rows_preserved",
            "passed": int(context["weather"]["input_rows"])
            == int(context["weather"]["output_rows"])
            and int(context["calendar"]["input_rows"])
            == int(context["calendar"]["output_rows"]),
            "detail": "weather and calendar cleaning preserve source rows",
        },
        {
            "name": "merge_cardinality",
            "passed": int(merge["output_rows"]) == int(traffic["output_rows"])
            and int(merge["duplicate_output_keys"]) == 0,
            "detail": "merge retains one row per traffic origin",
        },
        {
            "name": "join_coverage",
            "passed": int(merge["joins"]["weather"]["missing"]) == 0
            and int(merge["joins"]["calendar"]["missing"]) == 0,
            "detail": "weather and calendar joins have zero misses",
        },
        {
            "name": "feature_cardinality",
            "passed": int(feature["input_rows"]) == int(feature["output_rows"])
            == int(merge["output_rows"]),
            "detail": "feature engineering retains every merged origin",
        },
        {
            "name": "processed_cardinality",
            "passed": int(processed["input_rows"])
            == int(processed["output_rows"])
            == int(feature["output_rows"]),
            "detail": "target construction retains every feature origin",
        },
    ]
    failed = [record["name"] for record in checks if not record["passed"]]
    if failed:
        raise RuntimeError(f"Quality reconciliation failed: {failed}")
    return {
        "source": {
            "total_rows": raw_rows,
            "traffic_rows": int(raw["traffic"]["shape"]["rows"]),
            "weather_rows": int(raw["weather"]["shape"]["rows"]),
            "calendar_rows": int(raw["calendar"]["shape"]["rows"]),
            "traffic_exact_duplicates": int(
                raw["traffic"]["exact_duplicate_count"]
            ),
            "traffic_missing_windows": int(raw["traffic"]["missing_window_count"]),
            "traffic_null_counts": raw["traffic"]["null_counts"],
            "weather_null_counts": raw["weather"]["null_counts"],
            "traffic_physical_invalid": raw["traffic"][
                "physical_invalid_counts"
            ],
            "blank_congestion_labels": int(
                raw["traffic"]["blank_congestion_label_count"]
            ),
        },
        "validation": {
            "input_rows": int(validation["total_input_rows"]),
            "valid_rows": int(validation["total_valid_rows"]),
            "rejected_rows": int(validation["total_rejected_rows"]),
            "issues": int(validation["total_issues"]),
            "traffic_issue_reasons": validation["datasets"]["traffic"][
                "issues_by_reason"
            ],
        },
        "cleaning": {
            "traffic_output_rows": int(traffic["output_rows"]),
            "inserted_windows": int(traffic["grid"]["inserted_windows"]),
            "traffic_imputation": traffic["imputation"],
            "weather_imputation": context["weather"]["imputation"],
            "weather_condition_counts": context["weather"]["condition_counts"],
            "weather_normalization": context["weather"][
                "condition_normalization"
            ],
            "congestion": traffic["congestion"],
            "vehicle_distribution": traffic["vehicle_distribution"],
            "unobserved_accident_windows": int(
                traffic["unobserved_accident_windows"]
            ),
            "remaining_trusted_nulls": traffic["remaining_trusted_nulls"],
        },
        "merge": {
            "output_rows": int(merge["output_rows"]),
            "unique_keys": int(merge["output_unique_keys"]),
            "weather_missing": int(merge["joins"]["weather"]["missing"]),
            "calendar_missing": int(merge["joins"]["calendar"]["missing"]),
        },
        "features": {
            "output_rows": int(feature["output_rows"]),
            "feature_count": int(feature["feature_count"]),
            "history_unavailable_rows": int(feature["history_unavailable_rows"]),
        },
        "processed": {
            "output_rows": int(processed["output_rows"]),
            "unique_keys": int(processed["output_unique_keys"]),
            "target_count": int(processed["target_count"]),
            "timestamp_coverage": processed["timestamp_coverage"],
            "target_coverage": processed["target_coverage"],
        },
        "checks": checks,
    }
