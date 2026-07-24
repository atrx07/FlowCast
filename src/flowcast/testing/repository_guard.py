"""Protect the checked-in repository state while automated tests execute."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import shutil
import subprocess
from typing import Iterable


@dataclass(frozen=True)
class SnapshotEntry:
    """One tracked file as it existed before a test session."""

    relative_path: Path
    sha256: str | None
    backup_path: Path | None


@dataclass(frozen=True)
class RepositorySnapshot:
    """Recoverable snapshot of the tracked repository files."""

    root: Path
    entries: tuple[SnapshotEntry, ...]


@dataclass(frozen=True)
class RepositoryMutation:
    """A tracked-file change made after a snapshot was captured."""

    relative_path: Path
    change: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_path(root: Path, relative_path: Path) -> Path:
    if relative_path.is_absolute():
        raise ValueError(f"Snapshot path must be relative: {relative_path}")
    resolved = (root / relative_path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"Snapshot path escapes the repository: {relative_path}")
    return resolved


def tracked_repository_files(root: Path) -> tuple[Path, ...]:
    """Return every Git-tracked path without changing repository state."""

    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=root,
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"Unable to enumerate tracked files: {detail}")
    decoded = result.stdout.decode("utf-8", errors="surrogateescape")
    return tuple(Path(value) for value in decoded.split("\0") if value)


def create_repository_snapshot(
    root: Path,
    relative_paths: Iterable[Path],
    backup_dir: Path,
) -> RepositorySnapshot:
    """Back up the current bytes of selected files, including dirty user edits."""

    resolved_root = root.resolve()
    resolved_backup = backup_dir.resolve()
    if resolved_backup.is_relative_to(resolved_root):
        raise ValueError("Repository snapshots must be stored outside the repository")
    resolved_backup.mkdir(parents=True, exist_ok=True)
    entries: list[SnapshotEntry] = []
    unique_paths = sorted(
        {Path(path) for path in relative_paths},
        key=lambda path: path.as_posix(),
    )
    for relative_path in unique_paths:
        source = _safe_path(resolved_root, relative_path)
        if not source.exists():
            entries.append(SnapshotEntry(relative_path, None, None))
            continue
        if not source.is_file():
            raise ValueError(f"Tracked path is not a regular file: {relative_path}")
        backup_path = resolved_backup / relative_path
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, backup_path)
        entries.append(
            SnapshotEntry(relative_path, _sha256(source), backup_path)
        )
    return RepositorySnapshot(resolved_root, tuple(entries))


def restore_repository_snapshot(
    snapshot: RepositorySnapshot,
) -> tuple[RepositoryMutation, ...]:
    """Restore changed tracked files and report every detected mutation."""

    mutations: list[RepositoryMutation] = []
    for entry in snapshot.entries:
        current = _safe_path(snapshot.root, entry.relative_path)
        if entry.sha256 is None:
            if current.exists():
                if not current.is_file():
                    raise RuntimeError(
                        "A previously absent tracked path became a directory: "
                        f"{entry.relative_path}"
                    )
                current.unlink()
                mutations.append(
                    RepositoryMutation(entry.relative_path, "created")
                )
            continue
        if current.exists() and current.is_file():
            if _sha256(current) == entry.sha256:
                continue
            change = "modified"
        else:
            change = "deleted"
        if entry.backup_path is None:
            raise RuntimeError(f"Missing backup for {entry.relative_path}")
        current.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(entry.backup_path, current)
        mutations.append(RepositoryMutation(entry.relative_path, change))
    return tuple(mutations)
