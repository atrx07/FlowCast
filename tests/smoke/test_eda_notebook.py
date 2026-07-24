"""Execution smoke test for the package-backed EDA notebook."""

from __future__ import annotations

import asyncio
import shutil
import sys

import nbformat
from nbclient import NotebookClient

from flowcast.settings import repository_root


def _copy(source, destination) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _isolated_paths(tmp_path):
    root = repository_root()
    artifacts = tmp_path / "artifacts"
    quarantine = tmp_path / "quarantine"
    processed = tmp_path / "processed"
    artifact_files = (
        "audits/raw_v1/audit.json",
        "quality/cleaned_sources_v1/summary.json",
        "quality/cleaned_sources_v1/traffic_summary.json",
        "quality/merged_sources_v1/summary.json",
        "quality/engineered_features_v1/summary.json",
        "quality/processed_targets_v1/summary.json",
        "features/engineered_features_v1/manifest.json",
        "features/processed_targets_v1/manifest.json",
    )
    for relative in artifact_files:
        _copy(root / "artifacts" / relative, artifacts / relative)
    _copy(
        root / "data/quarantine/validated_v1/summary.json",
        quarantine / "validated_v1/summary.json",
    )
    _copy(
        root / "data/processed/processed_targets_v1/dataset.parquet",
        processed / "processed_targets_v1/dataset.parquet",
    )
    return artifacts, processed, quarantine, tmp_path / "logs"


def test_eda_notebook_executes_top_to_bottom(tmp_path, monkeypatch) -> None:
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    for name in (
        "IPYTHONDIR",
        "JUPYTER_CONFIG_DIR",
        "JUPYTER_DATA_DIR",
        "JUPYTER_RUNTIME_DIR",
    ):
        directory = tmp_path / name.lower()
        directory.mkdir()
        monkeypatch.setenv(name, str(directory))
    path = repository_root() / "notebooks" / "01_eda.ipynb"
    notebook = nbformat.read(path, as_version=4)
    artifacts, processed, quarantine, logs = _isolated_paths(tmp_path)
    replacement = "\n".join(
        [
            "from dataclasses import replace",
            "from pathlib import Path",
            "settings = replace(",
            "    load_settings(),",
            f"    artifacts_dir=Path({str(artifacts)!r}),",
            f"    processed_dir=Path({str(processed)!r}),",
            f"    quarantine_dir=Path({str(quarantine)!r}),",
            f"    logs_dir=Path({str(logs)!r}),",
            ")",
        ]
    )
    notebook["cells"][1]["source"] = notebook["cells"][1]["source"].replace(
        "settings = load_settings()",
        replacement,
    )
    code = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    assert "run_eda" in code
    assert ".groupby(" not in code
    client = NotebookClient(
        notebook,
        timeout=120,
        kernel_name="python3",
        resources={"metadata": {"path": str(repository_root())}},
    )
    executed = client.execute()
    assert all(
        cell.get("execution_count") is not None
        for cell in executed["cells"]
        if cell["cell_type"] == "code"
    )
