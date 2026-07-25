"""Upload validation, report export, audit, and explicit retraining controls."""

from pathlib import Path

import streamlit as st

from flowcast.dashboard.cache import get_dashboard_bundle
from flowcast.dashboard.training_service import (
    ALLOWED_COMMANDS,
    CONFIRMATION,
    TrainingService,
)
from flowcast.dashboard.ui import render_metric_row, render_page_header
from flowcast.dashboard.uploads import stage_upload, validate_upload
from flowcast.reports import verify_prediction_reports


bundle = get_dashboard_bundle()
render_page_header(
    "Operate the pipeline without weakening its boundaries.",
    "Validate and stage new sources, download verified reports, inspect audit "
    "evidence, or explicitly launch a new versioned training run.",
    context="System controls · immutable raw data and frozen active routing",
)
render_metric_row(
    [
        {
            "label": "Processed rows",
            "value": f"{len(bundle.history):,}",
        },
        {
            "label": "Road coverage",
            "value": f"{bundle.history['road_id'].nunique()}",
        },
        {
            "label": "Registry entries",
            "value": f"{len(bundle.context.registry['entries'])}",
        },
        {
            "label": "Latest request",
            "value": bundle.batch.paths.request_id,
        },
    ]
)

upload_tab, report_tab, training_tab, audit_tab = st.tabs(
    [
        "Validate upload",
        "Reports",
        "Retraining",
        "Audit trail",
    ],
    on_change="rerun",
)

if upload_tab.open:
    with upload_tab:
        st.subheader("Validate a source CSV")
        st.caption(
            "Uploads are checked against the traffic, weather, or calendar "
            "contract. They never overwrite FlowCast-project_file or data/raw."
        )
        uploaded = st.file_uploader(
            "Source CSV",
            type=["csv"],
            accept_multiple_files=False,
            help="Maximum 200 MB. Columns must exactly match one raw contract.",
        )
        if uploaded is not None:
            try:
                validation = validate_upload(
                    uploaded.getvalue(),
                    uploaded.name,
                    bundle.settings,
                )
            except Exception as error:
                st.error(
                    f"Upload validation failed: {error}",
                    icon=":material/error:",
                )
            else:
                st.session_state["fc_upload_validation"] = validation
                summary = validation.summary
                render_metric_row(
                    [
                        {"label": "Detected dataset", "value": validation.dataset},
                        {"label": "Input rows", "value": f"{summary['input_rows']:,}"},
                        {"label": "Valid rows", "value": f"{summary['valid_rows']:,}"},
                        {
                            "label": "Rejected rows",
                            "value": f"{summary['rejected_rows']:,}",
                        },
                    ]
                )
                if summary["dataset_failure"]:
                    st.error(
                        "Dataset-level schema validation failed. Nothing can "
                        "be staged.",
                        icon=":material/block:",
                    )
                else:
                    st.success(
                        "Schema recognized and row accounting verified.",
                        icon=":material/check_circle:",
                    )
                    if st.button(
                        "Stage validated upload",
                        type="primary",
                        icon=":material/inventory_2:",
                    ):
                        path = stage_upload(validation, bundle.settings)
                        st.success(f"Staged with manifest: `{path}`")
                preview, issues = st.tabs(
                    ["Valid preview", "Validation issues"],
                    on_change="rerun",
                )
                if preview.open:
                    with preview:
                        st.dataframe(validation.valid_preview, hide_index=True)
                if issues.open:
                    with issues:
                        if validation.issues.empty:
                            st.caption("No validation issues were recorded.")
                        else:
                            st.dataframe(validation.issues, hide_index=True)

if report_tab.open:
    with report_tab:
        st.subheader("Verified forecast exports")
        generated_manifest = st.session_state.get(
            "fc_generated_report_manifest"
        )
        if generated_manifest:
            report_path = Path(generated_manifest)
        else:
            report_path = (
                bundle.settings.artifacts_dir
                / "reports"
                / bundle.batch.paths.version
                / bundle.batch.paths.request_id
                / "manifest.json"
            )
        if not report_path.is_file():
            st.warning(
                "No verified report exists for the current forecast batch.",
                icon=":material/pending:",
            )
        else:
            report = verify_prediction_reports(bundle.settings, report_path)
            csv_path = bundle.settings.root / report["artifacts"]["csv"]["path"]
            html_path = bundle.settings.root / report["artifacts"]["html"]["path"]
            st.caption(
                f"Request `{report['request_id']}` · source batch and both "
                "exports verified before download."
            )
            with st.container(horizontal=True):
                st.download_button(
                    "Download forecast CSV",
                    data=csv_path.read_bytes(),
                    file_name=csv_path.name,
                    mime="text/csv",
                    icon=":material/download:",
                )
                st.download_button(
                    "Download self-contained HTML",
                    data=html_path.read_bytes(),
                    file_name=html_path.name,
                    mime="text/html",
                    icon=":material/download:",
                )
            st.json(
                {
                    "insights": report["insights"],
                    "evaluation_evidence": report["evaluation_evidence"],
                    "limitations": report["limitations"],
                }
            )

if training_tab.open:
    with training_tab:
        st.subheader("Explicit versioned retraining")
        st.warning(
            "Training is computationally expensive. A successful run creates "
            "new versioned artifacts but does not switch active model routing.",
            icon=":material/warning:",
        )
        with st.form("retraining"):
            workflow = st.selectbox(
                "Training workflow",
                tuple(ALLOWED_COMMANDS),
            )
            confirmation = st.text_input(
                f"Type {CONFIRMATION} to confirm",
                placeholder=CONFIRMATION,
            )
            submitted = st.form_submit_button(
                "Start versioned retraining",
                type="primary",
                icon=":material/model_training:",
            )
        if submitted:
            service = TrainingService(bundle.settings)
            with st.status(
                "Running the confirmed training workflow...",
                expanded=True,
            ) as status:
                try:
                    result = service.run(workflow, confirmation)
                except Exception as error:
                    status.update(label="Retraining failed", state="error")
                    st.error(str(error))
                else:
                    state = "complete" if result.return_code == 0 else "error"
                    status.update(
                        label=f"Retraining exited with {result.return_code}",
                        state=state,
                    )
                    st.code(result.log_path.read_text(encoding="utf-8"))
                    st.caption(
                        f"Run `{result.run_id}` · version `{result.version}` · "
                        "active route unchanged."
                    )

if audit_tab.open:
    with audit_tab:
        st.subheader("Traceable system evidence")
        paths = {
            "Raw audit": (
                bundle.settings.artifacts_dir
                / "audits"
                / bundle.settings.audit_version
                / "audit.md"
            ),
            "Processed-data quality": bundle.context.processed.summary_path,
            "Classical registry": bundle.context.registry_paths.summary_path,
            "Confidence and error analysis": (
                bundle.settings.artifacts_dir
                / "metrics"
                / bundle.context.confidence.summary["version"]
                / "summary.md"
            ),
            "DESIGN.md": bundle.settings.root / "DESIGN.md",
        }
        st.dataframe(
            [
                {
                    "Evidence": name,
                    "Path": path.relative_to(bundle.settings.root).as_posix(),
                    "Present": path.is_file(),
                }
                for name, path in paths.items()
            ],
            hide_index=True,
        )
        st.caption(
            "Prediction and report downloads are verified recursively. Audit "
            "links identify local evidence; they do not expose mutable raw data."
        )
