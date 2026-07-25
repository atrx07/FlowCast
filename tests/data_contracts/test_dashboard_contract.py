"""Full-artifact contracts for the dashboard service boundary."""

from __future__ import annotations

from flowcast.dashboard.analytics import eligible_prediction_origins
from flowcast.dashboard.data import load_dashboard_bundle


def test_dashboard_bundle_reconciles_verified_inputs() -> None:
    bundle = load_dashboard_bundle()
    assert len(bundle.history) == 181_200
    assert bundle.history["road_id"].nunique() == 25
    origins = eligible_prediction_origins(
        bundle.history,
        sequence_length=int(
            bundle.context.config["request"]["recurrent_sequence_length"]
        ),
        cadence_minutes=int(
            bundle.context.config["request"]["cadence_minutes"]
        ),
    )
    assert len(origins) == 7_237
    assert origins[0].isoformat() == "2025-01-01T05:30:00+05:30"
    assert origins[-1].isoformat() == "2025-05-31T23:30:00+05:30"
    request = bundle.batch.manifest["request"]
    coverage = bundle.batch.manifest["coverage"]
    assert len(bundle.predictions) == coverage["row_count"]
    assert bundle.predictions["road_id"].nunique() == coverage["road_count"]
    assert set(bundle.predictions["horizon_windows"]) == set(request["horizons"])
    assert set(bundle.predictions["road_id"]) == set(request["road_ids"])
    assert len(bundle.registry_scoreboard) == 20
    assert len(bundle.recurrent_comparison) == 4
    assert set(bundle.confidence.regression["split"]) == {"validation", "test"}
    assert set(bundle.confidence.classification["split"]) == {
        "validation",
        "test",
    }
    assert bundle.report_manifest is not None
    assert (
        bundle.report_manifest["request_id"]
        == bundle.batch.manifest["request_id"]
    )
