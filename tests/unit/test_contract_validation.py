"""Focused raw-contract validation tests using small in-memory frames."""

from __future__ import annotations

import copy
import json

import pandas as pd

from flowcast.data.contracts import load_contracts
from flowcast.data.validation import validate_frame
from flowcast.settings import load_settings


def _traffic_contract() -> dict:
    return load_contracts(load_settings())["traffic"]


def _traffic_row(**changes: object) -> dict[str, object]:
    row: dict[str, object] = {
        "road_id": "NL-001",
        "road_name": "Test Road",
        "latitude": 19.1,
        "longitude": 72.9,
        "weather_station_id": "WS-NORTH",
        "date": "2024-01-01",
        "time": "00:00",
        "traffic_volume": 100,
        "vehicle_count": 100,
        "vehicle_type_dist": json.dumps(
            {"2W": 0.2, "Car": 0.5, "LCV": 0.2, "HCV": 0.1}
        ),
        "avg_speed": 50.0,
        "occupancy": 20.0,
        "congestion_level": "Free-flow",
        "travel_time": 10.0,
        "accident_count": 0,
        "signal_timing": 60,
        "road_capacity": 1000,
    }
    row.update(changes)
    return row


def _validate_traffic(rows: list[dict[str, object]]):
    return validate_frame(
        pd.DataFrame(rows),
        "traffic",
        _traffic_contract(),
        "traffic_sensor_log.csv",
    )


def test_valid_row_has_typed_timestamp_and_lineage() -> None:
    result = _validate_traffic([_traffic_row()])

    assert result.row_accounting_valid
    assert len(result.valid_rows) == 1
    assert result.rejected_rows.empty
    assert not result.issues
    assert str(result.valid_rows["timestamp"].dt.tz) == "Asia/Kolkata"
    assert result.valid_rows.loc[0, "_source_row"] == 2
    assert result.valid_rows.loc[0, "_validation_status"] == "valid"


def test_schema_failure_rejects_dataset_without_losing_rows() -> None:
    row = _traffic_row()
    del row["road_id"]
    result = _validate_traffic([row])

    assert result.has_dataset_failure
    assert result.valid_rows.empty
    assert len(result.rejected_rows) == 1
    assert result.issues[0].reason_code == "missing_required_column"
    assert result.issues[0].disposition == "dataset_rejected"


def test_unexpected_metadata_like_source_column_is_not_silently_accepted() -> None:
    result = _validate_traffic([_traffic_row(_unexpected="value")])

    assert result.has_dataset_failure
    assert result.issues[0].reason_code == "unexpected_column"


def test_invalid_timestamp_is_quarantined() -> None:
    result = _validate_traffic([_traffic_row(time="00:15")])

    assert result.valid_rows.empty
    assert len(result.rejected_rows) == 1
    assert [issue.reason_code for issue in result.issues] == ["invalid_timestamp"]


def test_physical_invalid_values_are_logged_and_invalidated() -> None:
    result = _validate_traffic(
        [_traffic_row(traffic_volume=-1, avg_speed=200.1, occupancy=100.1)]
    )

    assert len(result.valid_rows) == 1
    assert result.valid_rows.loc[0, "_validation_status"] == "valid_with_issues"
    assert pd.isna(result.valid_rows.loc[0, "traffic_volume"])
    assert pd.isna(result.valid_rows.loc[0, "avg_speed"])
    assert pd.isna(result.valid_rows.loc[0, "occupancy"])
    assert {issue.reason_code for issue in result.issues} == {
        "negative_traffic_volume",
        "excessive_speed",
        "invalid_occupancy",
    }


def test_physical_boundary_values_are_allowed() -> None:
    result = _validate_traffic(
        [_traffic_row(traffic_volume=0, avg_speed=200, occupancy=100)]
    )

    assert len(result.valid_rows) == 1
    assert not result.issues


def test_non_finite_numeric_value_is_invalidated() -> None:
    result = _validate_traffic([_traffic_row(avg_speed=float("inf"))])

    assert len(result.valid_rows) == 1
    assert pd.isna(result.valid_rows.loc[0, "avg_speed"])
    assert result.issues[0].reason_code == "invalid_type"


def test_invalid_vehicle_distribution_is_logged_and_invalidated() -> None:
    result = _validate_traffic([_traffic_row(vehicle_type_dist='{"Car": 1}')])

    assert len(result.valid_rows) == 1
    assert pd.isna(result.valid_rows.loc[0, "vehicle_type_dist"])
    assert result.issues[0].reason_code == "invalid_json"


def test_duplicate_retention_prefers_more_complete_row_deterministically() -> None:
    incomplete = _traffic_row(traffic_volume=None)
    complete = copy.deepcopy(_traffic_row())
    result = _validate_traffic([incomplete, complete])

    assert len(result.valid_rows) == 1
    assert len(result.rejected_rows) == 1
    assert result.valid_rows.loc[0, "_source_row"] == 3
    duplicate = next(
        issue for issue in result.issues if issue.reason_code == "duplicate_key"
    )
    assert duplicate.source_row == 2
    assert duplicate.retained_source_row == 3


def test_weather_variants_are_accepted_but_unknown_values_are_invalidated() -> None:
    contract = load_contracts(load_settings())["weather"]
    rows = [
        {
            "station_id": "WS-NORTH",
            "date": "01/01/2024",
            "time": "00:00",
            "weather_condition": " rain ",
            "temperature": 25,
            "rainfall": 1,
            "visibility": 5,
        },
        {
            "station_id": "WS-NORTH",
            "date": "01/01/2024",
            "time": "01:00",
            "weather_condition": "Hail",
            "temperature": 25,
            "rainfall": 0,
            "visibility": 5,
        },
    ]
    result = validate_frame(
        pd.DataFrame(rows), "weather", contract, "weather_observations.csv"
    )

    assert len(result.valid_rows) == 2
    assert result.valid_rows.loc[0, "weather_condition"] == " rain "
    assert pd.isna(result.valid_rows.loc[1, "weather_condition"])
    assert [issue.reason_code for issue in result.issues] == ["invalid_category"]


def test_calendar_flag_name_mismatch_is_rejected() -> None:
    contract = load_contracts(load_settings())["calendar"]
    frame = pd.DataFrame(
        [
            {
                "date": "2024-01-01",
                "public_holiday": 1,
                "holiday_name": None,
                "event_flag": 0,
                "event_name": None,
                "roadwork_flag": 0,
            }
        ]
    )
    result = validate_frame(frame, "calendar", contract, "calendar_events.csv")

    assert result.valid_rows.empty
    assert len(result.rejected_rows) == 1
    assert result.issues[0].reason_code == "missing_flag_name"


def test_calendar_invalid_flag_is_rejected_with_stable_reason() -> None:
    contract = load_contracts(load_settings())["calendar"]
    frame = pd.DataFrame(
        [
            {
                "date": "2024-01-01",
                "public_holiday": 2,
                "holiday_name": None,
                "event_flag": 0,
                "event_name": None,
                "roadwork_flag": 0,
            }
        ]
    )
    result = validate_frame(frame, "calendar", contract, "calendar_events.csv")

    assert result.valid_rows.empty
    assert len(result.rejected_rows) == 1
    assert [issue.reason_code for issue in result.issues] == ["invalid_flag"]
