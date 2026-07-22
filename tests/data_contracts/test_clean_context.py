"""Full-source Step 04 context-cleaning artifact contract tests."""

from __future__ import annotations

import json
from dataclasses import replace

import pandas as pd
import pytest

from flowcast.data.audit import sha256_file
from flowcast.data.clean_context import run_context_cleaning
from flowcast.settings import load_settings


@pytest.fixture(scope="module")
def cleaned_context(tmp_path_factory):
    root = tmp_path_factory.mktemp("cleaned-context")
    settings = replace(
        load_settings(),
        raw_dir=root / "raw",
        interim_dir=root / "interim",
        quarantine_dir=root / "quarantine",
        artifacts_dir=root / "artifacts",
    )
    return run_context_cleaning(settings), settings


@pytest.mark.data_contract
def test_calendar_cleaning_contract(cleaned_context) -> None:
    run, _ = cleaned_context
    calendar = pd.read_parquet(run.calendar_path)

    assert len(calendar) == 151
    assert calendar["date"].nunique() == 151
    assert calendar["date"].min() == pd.Timestamp("2025-01-01")
    assert calendar["date"].max() == pd.Timestamp("2025-05-31")
    flag_counts = calendar[
        ["public_holiday", "event_flag", "roadwork_flag"]
    ].sum().to_dict()
    assert flag_counts == {
        "public_holiday": 6,
        "event_flag": 6,
        "roadwork_flag": 11,
    }


@pytest.mark.data_contract
def test_weather_cleaning_contract(cleaned_context) -> None:
    run, _ = cleaned_context
    weather = pd.read_parquet(run.weather_path)

    assert len(weather) == 10_872
    assert weather[["station_id", "weather_hour"]].duplicated().sum() == 0
    assert weather.groupby("station_id").size().to_dict() == {
        "WS-CENTRAL": 3_624,
        "WS-NORTH": 3_624,
        "WS-SOUTH": 3_624,
    }
    assert set(weather["weather_condition"]) == {
        "Clear",
        "Cloudy",
        "Fog",
        "Overcast",
        "Rain",
    }
    assert not weather[
        ["weather_condition", "temperature", "rainfall", "visibility"]
    ].isna().any().any()
    assert weather["temperature_was_missing"].sum() == 167
    assert weather["visibility_was_missing"].sum() == 111
    assert weather["rainfall"].ge(0).all()
    assert weather["visibility"].ge(0).all()


@pytest.mark.data_contract
def test_quality_summary_is_complete_and_machine_readable(cleaned_context) -> None:
    run, settings = cleaned_context
    persisted = json.loads(run.summary_path.read_text(encoding="utf-8"))

    assert persisted == run.summary
    assert persisted["contract_version"] == "context_cleaning_v1"
    assert persisted["datasets"]["weather"]["imputation"]["temperature"][
        "imputed"
    ] == 167
    assert persisted["datasets"]["weather"]["imputation"]["visibility"][
        "imputed"
    ] == 111
    assert persisted["configuration"]["cleaning"]["sha256"] == sha256_file(
        settings.cleaning_config_path
    )
    assert persisted["configuration"]["data_contracts"][
        "sha256"
    ] == sha256_file(settings.data_contracts_path)
    assert b"\r\n" not in run.summary_path.read_bytes()
    assert b"\r\n" not in run.markdown_path.read_bytes()


@pytest.mark.data_contract
def test_context_cleaning_is_artifact_deterministic(cleaned_context) -> None:
    first, settings = cleaned_context
    first_hashes = {
        "calendar": sha256_file(first.calendar_path),
        "weather": sha256_file(first.weather_path),
        "summary": sha256_file(first.summary_path),
        "markdown": sha256_file(first.markdown_path),
    }
    repeated = run_context_cleaning(settings)

    assert repeated.summary == first.summary
    assert {
        "calendar": sha256_file(repeated.calendar_path),
        "weather": sha256_file(repeated.weather_path),
        "summary": sha256_file(repeated.summary_path),
        "markdown": sha256_file(repeated.markdown_path),
    } == first_hashes
