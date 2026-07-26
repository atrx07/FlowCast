"""Shared Streamlit presentation primitives for the FlowCast product."""

from __future__ import annotations

from collections.abc import Iterable
from html import escape
from typing import Any

import streamlit as st

from flowcast.dashboard.config import APP_VERSION
from flowcast.dashboard.data import DashboardBundle


def apply_design_system() -> None:
    """Inject scoped motion and layout refinements requested by the design brief."""

    st.html(
        """
<style>
:root { --fc-accent: #5e6ad2; --fc-line: #23252a; }
.stApp { overflow-x: hidden; }
[data-testid="stMainBlockContainer"] {
  max-width: 1380px;
  padding-top: 4.25rem;
  padding-bottom: 5rem;
  padding-left: clamp(1rem, 3vw, 3rem);
  padding-right: clamp(1rem, 3vw, 3rem);
}
.st-key-page-hero {
  border: 1px solid var(--fc-line);
  border-radius: 12px;
  min-height: 4rem;
  padding: .55rem clamp(.85rem, 1.8vw, 1.25rem);
  background: #0f1011;
  overflow: hidden;
  animation: fc-rise .55s cubic-bezier(.2,.8,.2,1) both;
  gap: .08rem !important;
}
.st-key-page-hero [data-testid="stHorizontalBlock"] {
  align-items: center;
}
.st-key-page-hero [data-testid="stVerticalBlock"] {
  gap: .08rem;
}
.st-key-page-hero h1 {
  max-width: 1120px;
  font-size: clamp(1.55rem, 2.25vw, 1.95rem);
  letter-spacing: -0.035em;
  line-height: 1.08;
  margin: 0;
  padding: 0;
}
.st-key-page-hero p {
  max-width: 860px;
  color: #d0d6e0;
  font-size: .82rem;
  line-height: 1.32;
  margin: 0;
}
.st-key-page-hero [data-testid="stCaptionContainer"] p {
  color: #8a8f98;
  font-size: .7rem;
  line-height: 1.25;
  letter-spacing: .02em;
  text-align: right;
  padding: 0;
}
.st-key-page-hero code { color: #828fff; background: transparent; }
.st-key-status-strip {
  border-top: 1px solid var(--fc-line);
  border-bottom: 1px solid var(--fc-line);
  min-height: 2.5rem;
  padding: .35rem 0;
  margin-bottom: .2rem;
}
.st-key-status-strip [data-testid="stCaptionContainer"] {
  margin: 0;
}
.st-key-page-brief, [class*="st-key-page-brief-"] {
  border: 1px solid var(--fc-line);
  border-left: 3px solid var(--fc-accent);
  border-radius: 10px;
  padding: .7rem .9rem .75rem;
  background: #0f1011;
}
.st-key-page-brief [data-testid="stVerticalBlock"],
[class*="st-key-page-brief-"] [data-testid="stVerticalBlock"] {
  gap: .2rem;
}
.st-key-page-brief p, [class*="st-key-page-brief-"] p {
  max-width: 1060px;
  margin: 0;
  line-height: 1.45;
}
.fc-brief-label {
  color: #828fff;
  font-size: .74rem;
  font-weight: 600;
  letter-spacing: .045em;
  text-transform: uppercase;
}
.st-key-page-brief [data-testid="stCaptionContainer"] p,
[class*="st-key-page-brief-"] [data-testid="stCaptionContainer"] p {
  color: #a8adb6;
  font-size: .8rem;
  letter-spacing: 0;
}
.st-key-bento-corridor, .st-key-bento-queue {
  transition: transform .35s ease, border-color .35s ease, background .35s ease;
}
.st-key-bento-corridor:hover, .st-key-bento-queue:hover {
  transform: translateY(-3px);
  border-color: #3e3e44;
  background: #141516;
}
.stMetric {
  transition: transform .3s ease, border-color .3s ease;
}
.stMetric:hover { transform: translateY(-2px); border-color: #5e6ad2; }
.st-key-live-horizon > div > [data-testid="stVerticalBlock"] {
  gap: .25rem;
}
.st-key-live-kpis > div > [data-testid="stVerticalBlock"] {
  gap: 0;
}
.st-key-live-kpis [data-testid="stMetric"] {
  min-height: 5.25rem;
  padding: .65rem .75rem;
}
.st-key-prediction-workflow {
  padding: .75rem .9rem .8rem;
}
.st-key-prediction-workflow > div > [data-testid="stVerticalBlock"] {
  gap: .45rem;
}
.st-key-prediction-workflow [data-testid="stForm"] {
  border: 0;
  padding: 0;
}
.st-key-prediction-workflow h3,
.st-key-bento-corridor h3,
.st-key-bento-queue h3 {
  margin-bottom: 0;
}
.st-key-prediction-workflow h3 {
  font-size: 1.35rem;
  line-height: 1.2;
}
.st-key-prediction-workflow [data-testid="stCaptionContainer"] p,
.st-key-bento-corridor [data-testid="stCaptionContainer"] p {
  color: #a8adb6;
  font-size: .78rem;
  line-height: 1.35;
}
[data-testid="stPlotlyChart"], [data-testid="stDataFrame"] {
  animation: fc-reveal .55s ease both;
}
button[kind="primary"] { font-weight: 600; }
@keyframes fc-rise {
  from { opacity: 0; transform: translateY(16px) scale(.985); }
  to { opacity: 1; transform: translateY(0) scale(1); }
}
@keyframes fc-reveal {
  from { opacity: .25; transform: translateY(8px); }
  to { opacity: 1; transform: translateY(0); }
}
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { animation: none !important; transition: none !important; }
}
@media (max-width: 900px) {
  .st-key-page-hero [data-testid="stCaptionContainer"] p {
    text-align: left;
  }
}
@media (max-width: 1399px) {
  .st-key-page-hero [data-testid="stCaptionContainer"] {
    display: none;
  }
}
</style>
        """
    )


def render_page_header(
    title: str,
    description: str,
    *,
    context: str,
) -> None:
    """Render the compact operational page banner."""

    with st.container(key="page-hero"):
        with st.container(
            horizontal=True,
            horizontal_alignment="distribute",
            vertical_alignment="center",
        ):
            st.title(title)
            st.caption(context)
        st.markdown(description)


def render_status_strip(bundle: DashboardBundle) -> None:
    """Render compact verified lineage and runtime status."""

    manifest = bundle.batch.manifest
    runtime = manifest["runtime"]
    with st.container(
        key="status-strip",
        horizontal=True,
        horizontal_alignment="distribute",
        vertical_alignment="center",
    ):
        st.badge(
            "Artifacts verified",
            icon=":material/verified:",
            color="green",
        )
        st.caption(f"Data · `{bundle.settings.processed_version}`")
        st.caption(f"Models · `{bundle.context.registry['version']}`")
        st.caption(f"Inference · `{runtime['cold_total_seconds']:.2f}s` cold")
        st.caption(f"UI · `{APP_VERSION}`")


def render_metric_row(metrics: Iterable[dict[str, Any]]) -> None:
    """Render responsive bordered metrics."""

    with st.container(horizontal=True):
        for item in metrics:
            st.metric(
                item["label"],
                item["value"],
                item.get("delta"),
                delta_color=item.get("delta_color", "normal"),
                border=True,
                help=item.get("help"),
            )


def render_insight_brief(
    summary: str,
    *,
    guidance: str,
    title: str = "Current reading",
    key: str = "summary",
) -> None:
    """Explain verified page evidence in compact, plain language."""

    with st.container(key=f"page-brief-{key}"):
        st.html(f'<p class="fc-brief-label">{escape(title)}</p>')
        st.markdown(summary)
        st.caption(f"How to read · {guidance}")


def render_empty(message: str) -> None:
    """Render a consistent non-fabricated empty state."""

    st.warning(message, icon=":material/filter_alt_off:")


def render_lineage(bundle: DashboardBundle) -> None:
    """Show exact batch, model, data, and configuration lineage."""

    expander = st.expander(
        "Forecast lineage",
        icon=":material/account_tree:",
        on_change="rerun",
    )
    if expander.open:
        with expander:
            prediction = bundle.predictions.iloc[0]
            rows = {
                "Request": bundle.batch.manifest["request_id"],
                "Origin": str(prediction["origin_timestamp"]),
                "Data": prediction["data_version"],
                "Feature set": prediction["feature_version"],
                "Preprocessing": prediction["preprocessing_version"],
                "Registry": prediction["registry_version"],
                "Confidence": prediction["confidence_version"],
                "Volume route": prediction["volume_model_version"],
                "Volume fallback": prediction["volume_classical_model_version"],
                "Processed SHA-256": prediction["processed_data_sha256"],
            }
            st.json(rows)


def stop_on_error(exc: Exception) -> None:
    """Render a clear artifact failure and halt the current page."""

    st.error(
        "FlowCast could not verify the required persisted artifacts. "
        f"{type(exc).__name__}: {exc}",
        icon=":material/error:",
    )
    st.stop()
