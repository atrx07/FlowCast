"""Explicit, versioned, duplicate-safe dashboard retraining boundary."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Callable, Sequence

from flowcast.settings import Settings


ALLOWED_COMMANDS = {
    "Classical regression": "train-classical-regression",
    "Congestion and accident classifiers": "train-classical-classification",
    "Recurrent volume model": "train-recurrent-volume",
}
CONFIRMATION = "RETRAIN"


@dataclass(frozen=True)
class TrainingResult:
    """Outcome and immutable run evidence for one explicit training request."""

    run_id: str
    command: str
    version: str
    return_code: int
    run_directory: Path
    log_path: Path
    manifest_path: Path


class TrainingService:
    """Run an approved training command without changing active model routing."""

    def __init__(
        self,
        settings: Settings,
        *,
        output_root: Path | None = None,
        runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
    ) -> None:
        self.settings = settings
        self.root = output_root or settings.artifacts_dir
        self.runner = runner or subprocess.run

    @property
    def lock_path(self) -> Path:
        """Return the single dashboard training lock path."""

        return self.root / "training_runs" / ".active.lock"

    def _acquire_lock(self, run_id: str) -> int:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(
                self.lock_path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            )
        except FileExistsError as exc:
            raise RuntimeError("Another dashboard retraining run is active") from exc
        os.write(descriptor, run_id.encode("utf-8"))
        return descriptor

    def _release_lock(self, descriptor: int) -> None:
        os.close(descriptor)
        self.lock_path.unlink(missing_ok=True)

    def run(self, label: str, confirmation: str) -> TrainingResult:
        """Run one explicitly confirmed command into a new artifact version."""

        if confirmation != CONFIRMATION:
            raise ValueError(f"Type {CONFIRMATION} to confirm retraining")
        if label not in ALLOWED_COMMANDS:
            raise ValueError(f"Unsupported training workflow: {label}")
        command = ALLOWED_COMMANDS[label]
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_id = f"{command}-{stamp}"
        version = f"dashboard_{command.replace('-', '_')}_{stamp.lower()}"
        run_directory = self.root / "training_runs" / run_id
        run_directory.mkdir(parents=True, exist_ok=False)
        log_path = run_directory / "run.log"
        manifest_path = run_directory / "manifest.json"
        descriptor = self._acquire_lock(run_id)
        argv: Sequence[str] = (
            sys.executable,
            "-m",
            "flowcast.cli",
            command,
            "--version",
            version,
        )
        try:
            completed = self.runner(
                argv,
                cwd=self.settings.root,
                capture_output=True,
                text=True,
                check=False,
            )
            output = f"{completed.stdout}\n{completed.stderr}".strip()
            log_path.write_text(output + "\n", encoding="utf-8", newline="\n")
            manifest = {
                "contract_version": "dashboard_training_run_v1",
                "run_id": run_id,
                "command": command,
                "version": version,
                "return_code": int(completed.returncode),
                "explicit_confirmation": True,
                "active_model_switched": False,
                "log_path": log_path.relative_to(self.settings.root).as_posix()
                if log_path.is_relative_to(self.settings.root)
                else log_path.as_posix(),
            }
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
                newline="\n",
            )
        finally:
            self._release_lock(descriptor)
        return TrainingResult(
            run_id=run_id,
            command=command,
            version=version,
            return_code=int(completed.returncode),
            run_directory=run_directory,
            log_path=log_path,
            manifest_path=manifest_path,
        )
