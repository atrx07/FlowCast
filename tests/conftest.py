"""Cross-suite safeguards for canonical repository state."""

from __future__ import annotations

import pytest

from flowcast.settings import repository_root
from flowcast.testing.repository_guard import (
    create_repository_snapshot,
    restore_repository_snapshot,
    tracked_repository_files,
)


@pytest.fixture(scope="session", autouse=True)
def preserve_tracked_repository_state(tmp_path_factory):
    """Restore and reject any tracked-file mutation made by a test session."""

    root = repository_root()
    backup_dir = tmp_path_factory.mktemp("flowcast-repository-guard")
    snapshot = create_repository_snapshot(
        root,
        tracked_repository_files(root),
        backup_dir,
    )
    yield
    mutations = restore_repository_snapshot(snapshot)
    if mutations:
        details = ", ".join(
            f"{mutation.relative_path.as_posix()} ({mutation.change})"
            for mutation in mutations
        )
        pytest.fail(
            "Automated tests changed tracked repository files. The pre-test "
            f"bytes were restored; offending paths: {details}",
            pytrace=False,
        )
