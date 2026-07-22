"""Unit tests for trusted calendar cleaning."""

from __future__ import annotations

import pandas as pd
import pytest
import yaml

from flowcast.data.clean_calendar import clean_calendar
from flowcast.settings import load_settings


def _config() -> dict:
    settings = load_settings()
    return yaml.safe_load(settings.cleaning_config_path.read_text(encoding="utf-8"))[
        "calendar"
    ]


def _row(**changes: object) -> dict[str, object]:
    row: dict[str, object] = {
        "date": "2025-01-01",
        "public_holiday": 1,
        "holiday_name": " New Year ",
        "event_flag": 0,
        "event_name": None,
        "roadwork_flag": 0,
        "_source_file": "calendar_events.csv",
        "_source_row": 2,
        "_validation_status": "valid",
    }
    row.update(changes)
    return row


def test_calendar_normalizes_dates_names_and_flags() -> None:
    frame = pd.DataFrame(
        [
            _row(),
            _row(
                date="2025-01-02",
                public_holiday=0,
                holiday_name=None,
                event_flag=1,
                event_name="Concert",
                _source_row=3,
            ),
        ]
    )
    result = clean_calendar(frame, _config())

    assert result.frame["date"].dtype == "datetime64[us]"
    assert str(result.frame["public_holiday"].dtype) == "Int8"
    assert result.frame.loc[0, "holiday_name"] == "New Year"
    assert result.summary["flag_counts"] == {
        "public_holiday": 1,
        "event_flag": 1,
        "roadwork_flag": 0,
    }


def test_calendar_rejects_duplicate_dates() -> None:
    frame = pd.DataFrame([_row(), _row(_source_row=3)])

    with pytest.raises(ValueError, match="not unique"):
        clean_calendar(frame, _config())


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"public_holiday": 2}, "only 0/1"),
        ({"holiday_name": None}, "relationship failed"),
        (
            {"public_holiday": 0, "holiday_name": "Unexpected"},
            "relationship failed",
        ),
    ],
)
def test_calendar_rejects_invalid_flag_semantics(
    changes: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        clean_calendar(pd.DataFrame([_row(**changes)]), _config())
