"""Timestamp parsing tests for the two incompatible delivered date formats."""

import pandas as pd

from flowcast.data.audit import parse_traffic_timestamp, parse_weather_timestamp


def test_traffic_timestamp_uses_year_first_format() -> None:
    parsed = parse_traffic_timestamp(
        pd.Series(["2025-02-03", "invalid"]),
        pd.Series(["07:30", "08:00"]),
    )
    assert parsed.iloc[0] == pd.Timestamp("2025-02-03 07:30:00")
    assert pd.isna(parsed.iloc[1])


def test_weather_timestamp_uses_day_first_format() -> None:
    parsed = parse_weather_timestamp(
        pd.Series(["03/02/2025", "31/02/2025"]),
        pd.Series(["07:00", "08:00"]),
    )
    assert parsed.iloc[0] == pd.Timestamp("2025-02-03 07:00:00")
    assert pd.isna(parsed.iloc[1])
