"""Full-artifact contracts for the dashboard service boundary."""

from __future__ import annotations

from flowcast.dashboard.data import load_dashboard_bundle


def test_dashboard_bundle_reconciles_verified_inputs() -> None:
    bundle = load_dashboard_bundle()
    assert len(bundle.history) == 181_200
    assert bundle.history["road_id"].nunique() == 25
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
