"""Small typed results shared by source-cleaning modules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class TableCleaningResult:
    """One cleaned table and its deterministic quality summary."""

    frame: pd.DataFrame
    summary: dict[str, Any]
