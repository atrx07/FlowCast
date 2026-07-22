"""Unit and leakage tests for causal hourly weather cleaning."""

from __future__ import annotations

import pandas as pd
import pytest
import yaml

from flowcast.data.clean_weather import clean_weather
from flowcast.data.contracts import load_contracts
from flowcast.settings import load_settings


def _policies() -> tuple[dict[str, str], dict]:
    settings = load_settings()
    normalization = load_contracts(settings)["weather"]["categorical"][
        "weather_condition"
    ]["normalization_map"]
    cleaning = yaml.safe_load(
        settings.cleaning_config_path.read_text(encoding="utf-8")
    )["weather"]
    return normalization, cleaning


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "station_id": ["WS-NORTH"] * 4,
            "weather_condition": ["Clear", " rain ", "RAIN", "foggy"],
            "temperature": [20.0, None, None, 30.0],
            "rainfall": [0.0, 1.0, 2.0, 0.0],
            "visibility": [1000.0, 900.0, None, 800.0],
            "weather_hour": pd.date_range(
                "2025-01-01", periods=4, freq="h", tz="Asia/Kolkata"
            ),
            "_source_file": ["weather_observations.csv"] * 4,
            "_source_row": pd.array([2, 3, 4, 5], dtype="Int64"),
            "_validation_status": ["valid"] * 4,
        }
    )


def test_weather_normalizes_and_causally_fills_with_donor_lineage() -> None:
    normalization, cleaning = _policies()
    result = clean_weather(_frame(), normalization, cleaning)

    assert result.frame["weather_condition"].tolist() == [
        "Clear",
        "Rain",
        "Rain",
        "Fog",
    ]
    assert result.frame["temperature"].tolist() == [20.0, 20.0, 20.0, 30.0]
    assert result.frame["temperature_was_missing"].tolist() == [
        False,
        True,
        True,
        False,
    ]
    assert result.frame["temperature_imputed_from_source_row"].tolist() == [
        pd.NA,
        2,
        2,
        pd.NA,
    ]
    assert result.frame.loc[2, "visibility_imputed_from_source_row"] == 3
    assert result.summary["imputation"]["temperature"]["imputed"] == 2
    assert result.summary["imputation"]["visibility"]["imputed"] == 1


def test_weather_fill_does_not_depend_on_future_value() -> None:
    normalization, cleaning = _policies()
    original = clean_weather(_frame(), normalization, cleaning).frame
    mutated = _frame()
    mutated.loc[3, "temperature"] = -999.0
    changed = clean_weather(mutated, normalization, cleaning).frame

    assert changed.loc[1:2, "temperature"].tolist() == original.loc[
        1:2, "temperature"
    ].tolist()


def test_weather_rejects_gap_longer_than_policy() -> None:
    normalization, cleaning = _policies()
    frame = _frame()
    frame.loc[3, "temperature"] = None

    with pytest.raises(ValueError, match="outside the causal policy"):
        clean_weather(frame, normalization, cleaning)


def test_weather_rejects_leading_missing_value() -> None:
    normalization, cleaning = _policies()
    frame = _frame()
    frame.loc[0, "visibility"] = None

    with pytest.raises(ValueError, match="outside the causal policy"):
        clean_weather(frame, normalization, cleaning)


def test_weather_rejects_uncontrolled_label() -> None:
    normalization, cleaning = _policies()
    frame = _frame()
    frame.loc[0, "weather_condition"] = "Hail"

    with pytest.raises(ValueError, match="outside the controlled map"):
        clean_weather(frame, normalization, cleaning)


def test_weather_rejects_incomplete_hourly_grid() -> None:
    normalization, cleaning = _policies()
    frame = _frame().drop(index=1)

    with pytest.raises(ValueError, match="not complete hourly"):
        clean_weather(frame, normalization, cleaning)


def test_weather_rejects_negative_rainfall() -> None:
    normalization, cleaning = _policies()
    frame = _frame()
    frame.loc[0, "rainfall"] = -0.1

    with pytest.raises(ValueError, match="rainfall"):
        clean_weather(frame, normalization, cleaning)
