"""Safe upload validation and isolated staging for dashboard inputs."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from io import BytesIO
import json
from pathlib import Path
from typing import Any

import pandas as pd

from flowcast.data.contracts import load_contracts
from flowcast.data.validation import validate_frame
from flowcast.settings import Settings


MAX_UPLOAD_BYTES = 200 * 1024 * 1024


@dataclass(frozen=True)
class UploadValidation:
    """Validation result and safe previews for one uploaded CSV."""

    dataset: str
    source_name: str
    sha256: str
    input_rows: int
    summary: dict[str, Any]
    valid_preview: pd.DataFrame
    rejected_preview: pd.DataFrame
    issues: pd.DataFrame
    original: bytes

    @property
    def accepted_for_staging(self) -> bool:
        """Return whether schema validation succeeded."""

        return not bool(self.summary["dataset_failure"])


def _detect_dataset(
    columns: set[str],
    contracts: dict[str, dict[str, Any]],
) -> str:
    matches = [
        name
        for name, contract in contracts.items()
        if set(contract["required_columns"]) == columns
    ]
    if len(matches) != 1:
        raise ValueError(
            "Upload columns must exactly match one traffic, weather, or "
            "calendar source contract"
        )
    return matches[0]


def validate_upload(
    payload: bytes,
    source_name: str,
    settings: Settings,
) -> UploadValidation:
    """Validate uploaded CSV bytes without writing to trusted project data."""

    if not payload:
        raise ValueError("Uploaded file is empty")
    if len(payload) > MAX_UPLOAD_BYTES:
        raise ValueError("Uploaded file exceeds the 200 MB product limit")
    if Path(source_name).suffix.lower() != ".csv":
        raise ValueError("FlowCast accepts CSV uploads only")
    try:
        frame = pd.read_csv(BytesIO(payload))
    except Exception as exc:
        raise ValueError("Uploaded file is not a readable CSV") from exc
    contracts = load_contracts(settings)
    dataset = _detect_dataset(set(frame.columns), contracts)
    result = validate_frame(
        frame,
        dataset,
        contracts[dataset],
        Path(source_name).name,
        settings.timezone,
    )
    issues = pd.DataFrame([issue.as_record() for issue in result.issues])
    return UploadValidation(
        dataset=dataset,
        source_name=Path(source_name).name,
        sha256=hashlib.sha256(payload).hexdigest(),
        input_rows=len(frame),
        summary=result.summary(),
        valid_preview=result.valid_rows.head(100).copy(),
        rejected_preview=result.rejected_rows.head(100).copy(),
        issues=issues.head(500).copy(),
        original=payload,
    )


def stage_upload(
    validation: UploadValidation,
    settings: Settings,
    *,
    output_root: Path | None = None,
) -> Path:
    """Stage a validated source outside immutable raw/reference directories."""

    if not validation.accepted_for_staging:
        raise ValueError("Dataset-level validation failures cannot be staged")
    root = output_root or settings.artifacts_dir
    request_id = validation.sha256[:16]
    directory = root / "uploads" / request_id
    directory.mkdir(parents=True, exist_ok=True)
    source_path = directory / validation.source_name
    source_path.write_bytes(validation.original)
    manifest = {
        "contract_version": "dashboard_upload_v1",
        "request_id": request_id,
        "dataset": validation.dataset,
        "source_name": validation.source_name,
        "bytes": len(validation.original),
        "sha256": validation.sha256,
        "validation": validation.summary,
        "trusted_raw_data_changed": False,
        "active_models_changed": False,
    }
    manifest_path = directory / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest_path
