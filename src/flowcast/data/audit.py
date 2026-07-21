"""Immutable raw-input preservation and automated baseline auditing."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import yaml

from flowcast.settings import Settings


@dataclass(frozen=True)
class AuditResult:
    """Paths and machine-readable payload produced by a raw audit."""

    json_path: Path
    markdown_path: Path
    manifest_path: Path
    payload: dict[str, Any]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_utf8(path: Path, content: str) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(content)


def sha256_file(path: Path, chunk_size: int = 1_048_576) -> str:
    """Return a streaming SHA-256 digest for a file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_contracts(settings: Settings) -> dict[str, Any]:
    """Load raw source contracts from the versioned YAML file."""

    with settings.data_contracts_path.open("r", encoding="utf-8") as handle:
        contracts: dict[str, Any] = yaml.safe_load(handle)
    return contracts


def _relative(path: Path, root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _previous_manifest(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    return {item["filename"]: item for item in manifest.get("files", [])}


def _copy_verified(source: Path, destination: Path, expected_hash: str) -> str:
    if destination.exists():
        copied_hash = sha256_file(destination)
        if copied_hash != expected_hash:
            raise RuntimeError(
                f"Immutable raw copy differs from its contract: {destination}"
            )
        return copied_hash

    pending = destination.with_suffix(destination.suffix + ".pending")
    if pending.exists():
        raise RuntimeError(f"Stale pending raw copy requires review: {pending}")
    with source.open("rb") as source_handle, pending.open("xb") as copy_handle:
        shutil.copyfileobj(source_handle, copy_handle)
    copied_hash = sha256_file(pending)
    if copied_hash != expected_hash:
        raise RuntimeError(f"Raw copy verification failed before publish: {source.name}")
    pending.replace(destination)
    return copied_hash


def preserve_raw_inputs(settings: Settings) -> tuple[Path, list[dict[str, Any]]]:
    """Create byte-identical raw copies and a SHA-256 lineage manifest."""

    settings.raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = settings.raw_dir / "raw_manifest.json"
    previous = _previous_manifest(manifest_path)
    contracts = load_contracts(settings)
    entries: list[dict[str, Any]] = []

    for contract in contracts.values():
        filename = str(contract["file"])
        expected_bytes = int(contract["bytes"])
        expected_hash = str(contract["sha256"])
        source = settings.reference_dir / filename
        destination = settings.raw_dir / filename
        if not source.is_file():
            raise FileNotFoundError(f"Missing delivered source file: {source}")
        if source.stat().st_size != expected_bytes:
            raise RuntimeError(f"Delivered source byte count changed: {source}")
        source_hash = sha256_file(source, settings.hash_chunk_size)
        if source_hash != expected_hash:
            raise RuntimeError(f"Delivered source SHA-256 changed: {source}")

        existed = destination.exists()
        copied_hash = _copy_verified(source, destination, expected_hash)
        prior_timestamp = previous.get(filename, {}).get("copied_at_utc")
        entries.append(
            {
                "filename": filename,
                "bytes": destination.stat().st_size,
                "sha256": copied_hash,
                "source_sha256": source_hash,
                "source_path": _relative(source, settings.root),
                "copied_path": _relative(destination, settings.root),
                "copied_at_utc": prior_timestamp if existed and prior_timestamp else _utc_now(),
            }
        )

    manifest = {
        "schema_version": "1.0",
        "generated_at_utc": _utc_now(),
        "files": entries,
    }
    _write_utf8(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest_path, entries


def parse_traffic_timestamp(date: pd.Series, time: pd.Series) -> pd.Series:
    """Parse traffic date/time columns with the mandated source format."""

    return pd.to_datetime(
        date.astype("string") + " " + time.astype("string"),
        format="%Y-%m-%d %H:%M",
        errors="coerce",
    )


def parse_weather_timestamp(date: pd.Series, time: pd.Series) -> pd.Series:
    """Parse day-first weather date/time columns with the mandated format."""

    return pd.to_datetime(
        date.astype("string") + " " + time.astype("string"),
        format="%d/%m/%Y %H:%M",
        errors="coerce",
    )


def _null_counts(frame: pd.DataFrame) -> dict[str, int]:
    return {column: int(value) for column, value in frame.isna().sum().items()}


def _dtypes(frame: pd.DataFrame) -> dict[str, str]:
    return {column: str(dtype) for column, dtype in frame.dtypes.items()}


def _numeric_ranges(
    frame: pd.DataFrame, columns: Iterable[str]
) -> dict[str, dict[str, float | int | None]]:
    ranges: dict[str, dict[str, float | int | None]] = {}
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        ranges[column] = {
            "min": None if values.notna().sum() == 0 else float(values.min()),
            "max": None if values.notna().sum() == 0 else float(values.max()),
            "null_count": int(values.isna().sum()),
        }
    return ranges


def _frequencies(series: pd.Series) -> dict[str, int]:
    labels = series.astype("string").fillna("<BLANK>").replace("", "<BLANK>")
    return {str(key): int(value) for key, value in labels.value_counts().items()}


def audit_traffic(frame: pd.DataFrame) -> dict[str, Any]:
    """Compute source-level traffic structure, quality, and coverage evidence."""

    timestamps = parse_traffic_timestamp(frame["date"], frame["time"])
    key_columns = ["road_id", "date", "time"]
    unique_keys = int(frame[key_columns].drop_duplicates().shape[0])
    road_count = int(frame["road_id"].nunique(dropna=True))
    valid_dates = timestamps.dropna()
    days = int((valid_dates.max().date() - valid_dates.min().date()).days + 1)
    expected_grid = road_count * days * 48
    volume = pd.to_numeric(frame["traffic_volume"], errors="coerce")
    speed = pd.to_numeric(frame["avg_speed"], errors="coerce")
    occupancy = pd.to_numeric(frame["occupancy"], errors="coerce")
    accidents = pd.to_numeric(frame["accident_count"], errors="coerce")
    blank_congestion = frame["congestion_level"].isna() | (
        frame["congestion_level"].astype("string").str.strip() == ""
    )
    numeric_columns = [
        "latitude",
        "longitude",
        "traffic_volume",
        "vehicle_count",
        "avg_speed",
        "occupancy",
        "travel_time",
        "accident_count",
        "signal_timing",
        "road_capacity",
    ]
    return {
        "shape": {"rows": int(frame.shape[0]), "columns": int(frame.shape[1])},
        "columns": list(frame.columns),
        "dtypes": _dtypes(frame),
        "null_counts": _null_counts(frame),
        "timestamp_coverage": {
            "minimum": valid_dates.min().isoformat(),
            "maximum": valid_dates.max().isoformat(),
            "invalid_count": int(timestamps.isna().sum()),
        },
        "unique_counts": {
            "road_id": road_count,
            "weather_station_id": int(frame["weather_station_id"].nunique()),
            "date": int(frame["date"].nunique()),
            "time": int(frame["time"].nunique()),
        },
        "exact_duplicate_count": int(frame.duplicated().sum()),
        "key_duplicate_count": int(frame.duplicated(key_columns).sum()),
        "unique_key_count": unique_keys,
        "expected_grid_size": expected_grid,
        "missing_window_count": expected_grid - unique_keys,
        "numeric_ranges": _numeric_ranges(frame, numeric_columns),
        "physical_invalid_counts": {
            "negative_traffic_volume": int((volume < 0).sum()),
            "speed_above_200_kmh": int((speed > 200).sum()),
            "occupancy_above_100_percent": int((occupancy > 100).sum()),
        },
        "blank_congestion_label_count": int(blank_congestion.sum()),
        "congestion_label_frequencies": _frequencies(frame["congestion_level"]),
        "accident_positive_count": int((accidents > 0).sum()),
        "accident_positive_rate": float((accidents > 0).mean()),
    }


def audit_weather(frame: pd.DataFrame) -> dict[str, Any]:
    """Compute source-level weather structure, quality, and coverage evidence."""

    timestamps = parse_weather_timestamp(frame["date"], frame["time"])
    valid_dates = timestamps.dropna()
    key_columns = ["station_id", "date", "time"]
    station_count = int(frame["station_id"].nunique())
    days = int((valid_dates.max().date() - valid_dates.min().date()).days + 1)
    expected_grid = station_count * days * 24
    unique_keys = int(frame[key_columns].drop_duplicates().shape[0])
    return {
        "shape": {"rows": int(frame.shape[0]), "columns": int(frame.shape[1])},
        "columns": list(frame.columns),
        "dtypes": _dtypes(frame),
        "null_counts": _null_counts(frame),
        "timestamp_coverage": {
            "minimum": valid_dates.min().isoformat(),
            "maximum": valid_dates.max().isoformat(),
            "invalid_count": int(timestamps.isna().sum()),
        },
        "unique_counts": {
            "station_id": station_count,
            "date": int(frame["date"].nunique()),
            "time": int(frame["time"].nunique()),
        },
        "exact_duplicate_count": int(frame.duplicated().sum()),
        "key_duplicate_count": int(frame.duplicated(key_columns).sum()),
        "unique_key_count": unique_keys,
        "expected_grid_size": expected_grid,
        "missing_window_count": expected_grid - unique_keys,
        "numeric_ranges": _numeric_ranges(
            frame, ["temperature", "rainfall", "visibility"]
        ),
        "weather_label_frequencies": _frequencies(frame["weather_condition"]),
    }


def audit_calendar(frame: pd.DataFrame) -> dict[str, Any]:
    """Compute source-level calendar structure and flag evidence."""

    dates = pd.to_datetime(frame["date"], format="%Y-%m-%d", errors="coerce")
    valid_dates = dates.dropna()
    flags = ["public_holiday", "event_flag", "roadwork_flag"]
    return {
        "shape": {"rows": int(frame.shape[0]), "columns": int(frame.shape[1])},
        "columns": list(frame.columns),
        "dtypes": _dtypes(frame),
        "null_counts": _null_counts(frame),
        "date_coverage": {
            "minimum": valid_dates.min().date().isoformat(),
            "maximum": valid_dates.max().date().isoformat(),
            "invalid_count": int(dates.isna().sum()),
        },
        "exact_duplicate_count": int(frame.duplicated().sum()),
        "key_duplicate_count": int(frame.duplicated(["date"]).sum()),
        "unique_date_count": int(frame["date"].nunique()),
        "flag_positive_counts": {
            flag: int(pd.to_numeric(frame[flag], errors="coerce").eq(1).sum())
            for flag in flags
        },
    }


def _markdown(payload: dict[str, Any]) -> str:
    datasets = payload["datasets"]
    traffic = datasets["traffic"]
    weather = datasets["weather"]
    calendar = datasets["calendar"]
    hashes = "\n".join(
        f"| {item['filename']} | {item['bytes']:,} | `{item['sha256']}` |"
        for item in payload["raw_manifest"]["files"]
    )
    return f"""# FlowCast Raw-Data Audit: {payload['audit_version']}

Generated: `{payload['generated_at_utc']}`

## Immutable source copies

| File | Bytes | SHA-256 |
|---|---:|---|
{hashes}

## Baseline summary

| Evidence | Traffic | Weather | Calendar |
|---|---:|---:|---:|
| Rows | {traffic['shape']['rows']:,} | {weather['shape']['rows']:,} | {calendar['shape']['rows']:,} |
| Columns | {traffic['shape']['columns']} | {weather['shape']['columns']} | {calendar['shape']['columns']} |
| Exact duplicates | {traffic['exact_duplicate_count']:,} | {weather['exact_duplicate_count']:,} | {calendar['exact_duplicate_count']:,} |
| Key duplicates | {traffic['key_duplicate_count']:,} | {weather['key_duplicate_count']:,} | {calendar['key_duplicate_count']:,} |

## Traffic coverage and quality

- Unique road/timestamp keys: **{traffic['unique_key_count']:,}**
- Expected 30-minute grid: **{traffic['expected_grid_size']:,}**
- Missing full windows: **{traffic['missing_window_count']:,}**
- Blank congestion labels: **{traffic['blank_congestion_label_count']:,}**
- Negative traffic-volume rows: **{traffic['physical_invalid_counts']['negative_traffic_volume']:,}**
- Accident-positive rows: **{traffic['accident_positive_count']:,}**
- Accident-positive rate: **{traffic['accident_positive_rate']:.6f}**

## Weather coverage and labels

- Unique station/hour keys: **{weather['unique_key_count']:,}**
- Expected station/hour grid: **{weather['expected_grid_size']:,}**
- Missing station/hour windows: **{weather['missing_window_count']:,}**
- Raw labels: `{json.dumps(weather['weather_label_frequencies'], sort_keys=True)}`

## Calendar flags

- Positive counts: `{json.dumps(calendar['flag_positive_counts'], sort_keys=True)}`

The canonical, complete evidence is in `audit.json`; this report is generated from
the same in-memory result and is not maintained independently.
"""


def run_raw_audit(settings: Settings, version: str | None = None) -> AuditResult:
    """Preserve sources, run the complete baseline audit, and persist reports."""

    audit_version = version or settings.audit_version
    manifest_path, manifest_entries = preserve_raw_inputs(settings)
    contracts = load_contracts(settings)
    frames = {
        name: pd.read_csv(settings.raw_dir / contract["file"], low_memory=False)
        for name, contract in contracts.items()
    }
    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "audit_version": audit_version,
        "generated_at_utc": _utc_now(),
        "raw_manifest": {
            "path": _relative(manifest_path, settings.root),
            "files": manifest_entries,
        },
        "datasets": {
            "traffic": audit_traffic(frames["traffic"]),
            "weather": audit_weather(frames["weather"]),
            "calendar": audit_calendar(frames["calendar"]),
        },
    }
    output_dir = settings.artifacts_dir / "audits" / audit_version
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "audit.json"
    markdown_path = output_dir / "audit.md"
    _write_utf8(json_path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _write_utf8(markdown_path, _markdown(payload))
    return AuditResult(json_path, markdown_path, manifest_path, payload)
