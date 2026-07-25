# NEXT_STEP.md

## Immediate Objective

Execute **Step 19 - Reproducibility, Documentation, and Final Acceptance**.
Prove that a fresh reviewer can rebuild, audit, launch, and use the complete
FlowCast v1.0 pipeline and its verified ten-page Streamlit product surface.

Do not change frozen Step 10-18 model choices, thresholds, calibrators,
recurrent architecture, confidence widths, active routing, evaluation
partitions, source predictions, or reported metrics merely to improve an
acceptance result. Record every limitation honestly.

## Read Before Acting

1. `AGENTS.md`, `TECH_STACK.md`, `STATUS.md`, and this file.
2. `STEPS.md` - Step 19 and every proven command needed by the clean rebuild.
3. Reproduction, deployment, lineage, artifact, and acceptance boundaries in
   `PROJECT.md`, `ROADMAP.md`, and `ARCHITECTURE.md`.
4. The original PRD delivery, reporting, dashboard, usability, and performance
   requirements.
5. All versioned configs, model cards, metric summaries, prediction/report
   manifests, and the Step 18 dashboard acceptance evidence.
6. The build-safety contract in `AGENTS.md` section 14.1.

## Hardware Notice Before Acting

Before execution, obey the workstation-resource disclosure rule in
`AGENTS.md`. Documentation and artifact inventory use the Intel Core Ultra 9
CPU lightly. A clean pipeline reproduction will use substantial CPU, system
RAM, and disk I/O for tabular processing and classical training. Full recurrent
training should use the verified NVIDIA GeForce RTX 5070 Laptop GPU only after
checking CUDA availability, device name, runtime, driver compatibility, and
VRAM; CPU fallback must remain valid. The Intel NPU is not part of v1. Give a
fresh, phase-specific notice before the clean rebuild, full training, or any
other heavy command.

## Single Best Next Action

Perform one isolated clean-environment reproduction and final acceptance audit:

1. Verify pinned installation from the documented setup command on Python 3.11.
2. Redirect every generated output to a fresh test-owned reproduction root;
   never overwrite canonical or pre-existing user artifacts.
3. Run the full pipeline in the proven order from immutable delivered inputs.
4. Reconcile produced row counts, hashes, model cards, metrics, predictions,
   confidence tables, and reports to their documented contracts.
5. Measure runtime and resource-sensitive stages, including CPU fallback and
   the approved CUDA recurrent-training path when available.
6. Launch the Streamlit app against the reproduced artifacts and repeat the
   full navigation, prediction, upload-validation, report-export, lineage, and
   retraining-safety walkthrough.
7. Run the complete test suite through `scripts/run_tests.py`, plus dependency,
   compilation, whitespace, source-size, and repository-mutation checks.
8. Write the final technical report covering data quality, mathematics,
   classical and recurrent models, confidence, error slices, acceptance
   targets, limitations, and recommendations.
9. Finish the README command reference and cross-platform setup/deployment
   notes.
10. Update every dynamic Markdown file with measured evidence before the final
    verified commit and push.

## Acceptance Gate

Step 19 is complete only when:

- A clean reviewer path installs with one documented command.
- One documented pipeline command or exact documented command sequence rebuilds
  all required artifacts from immutable sources.
- Reproduced artifacts load without hidden CUDA or workstation dependencies.
- The reported metrics reconcile to the frozen chronological hold-out.
- Every selected model has a model card, feature schema, preprocessing lineage,
  environment record, and portable load test.
- The ten-page dashboard launches with one command and all nine required views
  use real reproduced outputs.
- Upload validation, prediction, verified report export, audit evidence, and
  explicit versioned retraining controls pass acceptance.
- Formal targets are compared explicitly and unmet classifier/deep-model
  limitations remain visible.
- The final report, README, tests, dependency check, compilation, source-size
  check, and repository guard all pass.

## Current Blockers

None. Step 18 is verified and M7 is complete. Step 19 is intentionally a clean,
resource-intensive reproduction and final delivery gate; it has not yet been
executed.
