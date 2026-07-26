"""Streamlit smoke coverage for the app shell and every required page."""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest


APP = Path(__file__).resolve().parents[2] / "dashboard" / "app.py"
PAGES = (
    "live_predictions.py",
    "historical_trends.py",
    "congestion_heatmap.py",
    "road_comparison.py",
    "model_performance.py",
    "feature_importance.py",
    "forecast_visualization.py",
    "prediction_confidence.py",
    "weather_traffic.py",
    "data_training.py",
)


def test_streamlit_shell_and_all_pages_render() -> None:
    app = AppTest.from_file(APP, default_timeout=90).run()
    assert not app.exception
    for page in PAGES:
        app.switch_page(f"app_pages/{page}").run(timeout=90)
        assert not app.exception, page
        assert app.title, page
        assert any(
            caption.value.startswith("How to read")
            for caption in app.caption
        ), page
        if page == "live_predictions.py":
            assert (
                app.date_input(key="prediction_origin_date").label
                == "Forecast origin date"
            )
            assert (
                app.time_input(key="prediction_origin_time").label
                == "Forecast origin time"
            )
            assert any(
                caption.value.startswith("Predicting ")
                and "last observed window" in caption.value
                for caption in app.caption
            )
            operational_headings = [
                heading.value
                for heading in app.subheader
                if heading.value
                in {
                    "Request a frozen-model forecast",
                    "Corridor signal",
                    "Priority queue",
                }
            ]
            assert operational_headings == [
                "Request a frozen-model forecast",
                "Corridor signal",
                "Priority queue",
            ]
