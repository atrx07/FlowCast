"""Execution smoke test for the package-backed EDA notebook."""

from __future__ import annotations

import asyncio
import sys

import nbformat
from nbclient import NotebookClient

from flowcast.settings import repository_root


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
