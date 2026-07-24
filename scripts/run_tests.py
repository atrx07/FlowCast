"""Run pytest with a visible, faithfully propagated process exit code."""

from __future__ import annotations

import sys

from flowcast.testing.test_runner import run_pytest


if __name__ == "__main__":
    raise SystemExit(run_pytest(sys.argv[1:]))
