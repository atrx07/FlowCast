"""Load and validate immutable FlowCast raw sources."""

from __future__ import annotations

from typing import Any, Iterable

import pandas as pd

from flowcast.data.audit import preserve_raw_inputs, sha256_file
from flowcast.data.contracts import ValidationResult, load_contract_bundle
from flowcast.data.validation import validate_frame
from flowcast.settings import Settings


def validate_raw_sources(settings: Settings) -> dict[str, ValidationResult]:
    """Verify immutable raw files and validate them in dictionary load order."""

    bundle = load_contract_bundle(settings)
    contracts: dict[str, dict[str, Any]] = bundle["datasets"]
    missing_files = [
        contract["file"]
        for contract in contracts.values()
        if not (settings.raw_dir / contract["file"]).is_file()
    ]
    if missing_files:
        preserve_raw_inputs(settings)

    results: dict[str, ValidationResult] = {}
    load_order: Iterable[str] = bundle.get("load_order", contracts.keys())
    for dataset in load_order:
        contract = contracts[str(dataset)]
        raw_path = settings.raw_dir / str(contract["file"])
        if raw_path.stat().st_size != int(contract["bytes"]):
            raise RuntimeError(f"Raw byte count differs from contract: {raw_path}")
        if sha256_file(raw_path, settings.hash_chunk_size) != str(contract["sha256"]):
            raise RuntimeError(f"Raw SHA-256 differs from contract: {raw_path}")
        frame = pd.read_csv(raw_path, low_memory=False)
        results[str(dataset)] = validate_frame(
            frame,
            str(dataset),
            contract,
            str(contract["file"]),
            settings.timezone,
        )
    return results
