"""Reliable pytest execution with explicit status and elapsed-time evidence."""

from __future__ import annotations

from collections.abc import Sequence
from time import perf_counter

import pytest


def run_pytest(arguments: Sequence[str]) -> int:
    """Execute pytest and return its exact integer exit code."""

    started = perf_counter()
    exit_code = int(pytest.main(list(arguments)))
    elapsed = perf_counter() - started
    print(
        f"FLOWCAST_PYTEST_EXIT={exit_code} "
        f"FLOWCAST_PYTEST_SECONDS={elapsed:.2f}",
        flush=True,
    )
    return exit_code
