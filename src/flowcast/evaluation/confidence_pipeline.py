"""Step 16 confidence calibration, error analysis, and artifact pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from flowcast.data.artifacts import (
    artifact_record,
    validate_artifact_version,
    write_json,
    write_parquet,
)
from flowcast.evaluation.confidence_artifacts import (
    ConfidencePaths,
    confidence_paths,
)
from flowcast.evaluation.confidence_config import load_confidence_config
from flowcast.evaluation.confidence_inputs import (
    ConfidenceInputs,
    load_verified_confidence_inputs,
)
from flowcast.evaluation.confidence_metrics import (
    conformal_calibration,
    enrich_classification,
    enrich_regression,
    regression_coverage,
    reliability_table,
)
from flowcast.evaluation.confidence_pairing import (
    paired_volume_frame,
    paired_volume_slices,
)
from flowcast.evaluation.confidence_report import (
    build_diagnostics,
    markdown_report,
)
from flowcast.evaluation.confidence_slices import (
    accident_risk_bands,
    classification_slices,
    confusion_slices,
    regression_slices,
)
from flowcast.settings import Settings


@dataclass(frozen=True)
class ConfidenceResult:
    """Completed Step 16 artifacts and in-memory summary."""

    paths: ConfidencePaths
    summary: dict[str, Any]


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        frame.to_csv(index=False, lineterminator="\n", float_format="%.12g"),
        encoding="utf-8",
        newline="\n",
    )


def _required_frames(inputs: ConfidenceInputs) -> tuple[pd.DataFrame, ...]:
    frames = (inputs.regression, inputs.classification, inputs.recurrent)
    if any(frame is None for frame in frames):
        raise RuntimeError("Confidence pipeline requires loaded prediction frames")
    return frames  # type: ignore[return-value]


def run_confidence_analysis(
    settings: Settings,
    *,
    version: str | None = None,
) -> ConfidenceResult:
    """Build the complete, immutable-model Step 16 evidence package."""

    config, config_path = load_confidence_config(settings)
    selected_version = validate_artifact_version(
        version or str(config["version"])
    )
    paths = confidence_paths(settings, selected_version)
    inputs = load_verified_confidence_inputs(settings, config, load_frames=True)
    regression_source, classification_source, recurrent_source = _required_frames(
        inputs
    )
    regression_predictions = pd.concat(
        [regression_source, recurrent_source],
        ignore_index=True,
    )
    regression_predictions = regression_predictions.sort_values(
        [
            "model_version",
            "target",
            "horizon_windows",
            "split",
            "road_id",
            "timestamp",
        ],
        kind="stable",
    ).reset_index(drop=True)
    regression_config = config["regression"]
    level = float(regression_config["confidence_level"])
    calibration = conformal_calibration(regression_predictions, level)
    regression = enrich_regression(
        regression_predictions,
        calibration,
        inputs.processed.frame,
        clip_lower_at_zero=bool(regression_config["clip_lower_at_zero"]),
    )
    classification = enrich_classification(
        classification_source.sort_values(
            ["task", "horizon_windows", "split", "road_id", "timestamp"],
            kind="stable",
        ).reset_index(drop=True),
        inputs.processed.frame,
        config,
    )
    paired_predictions = paired_volume_frame(regression)

    coverage = regression_coverage(regression)
    reliability = reliability_table(
        classification,
        int(config["classification"]["reliability_bins"]),
    )
    risk_bands = accident_risk_bands(
        classification,
        config["accident_risk"]["threshold_multipliers"],
    )
    dimensions = tuple(config["slices"]["dimensions"])
    minimums = {
        name: int(value)
        for name, value in config["slices"]["minimum_rows"].items()
    }
    regression_slice_frame = regression_slices(
        regression, dimensions, minimums["regression"]
    )
    classification_slice_frame = classification_slices(
        classification,
        dimensions,
        minimums,
        int(config["slices"]["minimum_accident_positives"]),
    )
    error_slices = pd.concat(
        [regression_slice_frame, classification_slice_frame],
        ignore_index=True,
        sort=False,
    )
    confusions = confusion_slices(
        classification, dimensions, minimums["congestion"]
    )
    paired_slices = paired_volume_slices(
        paired_predictions, dimensions, minimums["regression"]
    )

    write_parquet(regression, paths.regression_predictions_path)
    write_parquet(classification, paths.classification_predictions_path)
    write_parquet(paired_predictions, paths.paired_predictions_path)
    for frame, path in (
        (calibration, paths.interval_calibration_path),
        (coverage, paths.regression_coverage_path),
        (reliability, paths.reliability_path),
        (risk_bands, paths.risk_bands_path),
        (error_slices, paths.error_slices_path),
        (confusions, paths.confusions_path),
        (paired_slices, paths.paired_slices_path),
    ):
        _write_csv(frame, path)

    diagnostics = build_diagnostics(
        coverage,
        error_slices,
        confusions,
        paired_slices,
        reliability,
        risk_bands,
        level,
    )
    source_rows = len(regression_source) + len(recurrent_source)
    checks = {
        "calibration_uses_validation_only": set(
            calibration["calibration_split"]
        )
        == {"validation"},
        "regression_rows_reconciled": len(regression) == source_rows,
        "classification_rows_reconciled": len(classification)
        == len(classification_source),
        "paired_rows_equal_recurrent_rows": len(paired_predictions)
        == len(recurrent_source),
        "all_context_rows_mapped": not (
            regression["actual_congestion"].isna().any()
            or classification["actual_congestion"].isna().any()
        ),
        "all_intervals_ordered": bool(
            regression["interval_lower"].le(regression["interval_upper"]).all()
        ),
        "all_probabilities_finite": bool(
            np.isfinite(classification["max_probability"]).all()
            and np.isfinite(classification["normalized_entropy"]).all()
        ),
        "frozen_source_predictions_unchanged": True,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Confidence analysis checks failed: {checks}")
    summary: dict[str, Any] = {
        "contract_version": "confidence_error_v1",
        "version": selected_version,
        "configuration": artifact_record(config_path, settings),
        "upstream": inputs.upstream_records,
        "methodology": {
            "regression": regression_config,
            "classification": config["classification"],
            "accident_risk": config["accident_risk"],
            "slices": config["slices"],
        },
        "coverage": {
            "regression_prediction_rows": len(regression),
            "classification_prediction_rows": len(classification),
            "paired_volume_rows": len(paired_predictions),
            "conformal_group_count": len(calibration),
            "reliability_row_count": len(reliability),
            "risk_band_row_count": len(risk_bands),
            "error_slice_row_count": len(error_slices),
            "supported_error_slice_rows": int(
                error_slices["sufficient_support"].sum()
            ),
        },
        "diagnostics": diagnostics,
        "checks": checks,
        "limitations": [
            "Intervals assume future residual behavior resembles validation data.",
            "Classification confidence describes frozen model probabilities and "
            "does not establish causal reliability under distribution shift.",
            "Rare accident events limit subgroup estimates; unsupported rows are "
            "retained without computed performance metrics.",
            "The recurrent model trails the classical volume model at 120 minutes.",
        ],
    }
    paths.report_path.parent.mkdir(parents=True, exist_ok=True)
    paths.report_path.write_text(
        markdown_report(summary, coverage),
        encoding="utf-8",
        newline="\n",
    )
    artifact_paths = {
        "report": paths.report_path,
        "interval_calibration": paths.interval_calibration_path,
        "regression_coverage": paths.regression_coverage_path,
        "classification_reliability": paths.reliability_path,
        "accident_risk_bands": paths.risk_bands_path,
        "error_slices": paths.error_slices_path,
        "confusion_matrices": paths.confusions_path,
        "paired_volume_slices": paths.paired_slices_path,
        "regression_confidence_predictions": paths.regression_predictions_path,
        "classification_confidence_predictions": (
            paths.classification_predictions_path
        ),
        "paired_volume_predictions": paths.paired_predictions_path,
    }
    summary["artifacts"] = {
        name: artifact_record(path, settings)
        for name, path in artifact_paths.items()
    }
    write_json(summary, paths.summary_path)
    return ConfidenceResult(paths, summary)
