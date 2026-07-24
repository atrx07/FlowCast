"""Regression tests for repository and pytest-runner safeguards."""

from __future__ import annotations

from pathlib import Path

import pytest

from flowcast.testing.repository_guard import (
    create_repository_snapshot,
    restore_repository_snapshot,
)
from flowcast.testing.test_runner import run_pytest


def test_repository_snapshot_restores_every_tracked_change(tmp_path) -> None:
    root = tmp_path / "repository"
    backup = tmp_path / "backup"
    root.mkdir()
    modified = root / "modified.txt"
    deleted = root / "nested" / "deleted.txt"
    created = root / "created.txt"
    modified.write_bytes(b"original modified bytes")
    deleted.parent.mkdir()
    deleted.write_bytes(b"original deleted bytes")
    snapshot = create_repository_snapshot(
        root,
        (
            Path("modified.txt"),
            Path("nested/deleted.txt"),
            Path("created.txt"),
        ),
        backup,
    )

    modified.write_bytes(b"mutated")
    deleted.unlink()
    created.write_bytes(b"new")
    mutations = restore_repository_snapshot(snapshot)

    assert modified.read_bytes() == b"original modified bytes"
    assert deleted.read_bytes() == b"original deleted bytes"
    assert not created.exists()
    assert {
        (item.relative_path.as_posix(), item.change)
        for item in mutations
    } == {
        ("created.txt", "created"),
        ("modified.txt", "modified"),
        ("nested/deleted.txt", "deleted"),
    }


def test_repository_snapshot_rejects_paths_outside_root(tmp_path) -> None:
    root = tmp_path / "repository"
    root.mkdir()
    with pytest.raises(ValueError, match="escapes"):
        create_repository_snapshot(
            root,
            (Path("../outside.txt"),),
            tmp_path / "backup",
        )


@pytest.mark.parametrize("pytest_exit", [0, 1, 4])
def test_test_runner_propagates_exact_exit_code(
    pytest_exit,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(
        "flowcast.testing.test_runner.pytest.main",
        lambda _args: pytest_exit,
    )

    assert run_pytest(["-q"]) == pytest_exit
    assert f"FLOWCAST_PYTEST_EXIT={pytest_exit}" in capsys.readouterr().out
