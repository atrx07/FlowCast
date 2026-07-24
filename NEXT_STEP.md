# NEXT_STEP.md

## Immediate Objective

Execute **Step 14 - Build the Classical Scoreboard and Registry**. Consolidate
the frozen Step 12 regressors and Step 13 classifiers into one verified,
dashboard-ready classical-model registry without retraining, changing a
selection, or opening test data again.

Do not begin recurrent/deep learning, confidence/error analysis, inference
services, report export, or dashboard work. The registry must preserve the
honest Step 12 and Step 13 results, including the unmet classifier acceptance
goals.

## Read Before Acting

1. `AGENTS.md`, `TECH_STACK.md`, `STATUS.md`, and this file.
2. `STEPS.md` - Step 14 and the proven Step 10/12/13 procedures.
3. Registry, persistence, evaluation, lineage, artifact, naming, and
   reproducibility sections of `PROJECT.md`, `ROADMAP.md`, and
   `ARCHITECTURE.md`.
4. The original PRD sections covering model comparison, model persistence,
   task metrics, SVM, dashboard performance views, and real-output rules.
5. Both canonical selection manifests, scoreboards, prediction manifests,
   model cards, feature schema, processed-target manifest, current Git diff,
   and relevant tests.

## Single Best Next Action

Build the reusable classical registry and complete Step 14:

1. Define a versioned registry contract and paths in configuration. The
   registry must represent all 20 selected classical jobs: volume, speed,
   travel time, congestion, and accident risk across horizons 1-4.
2. Hash-verify the complete Step 12 and Step 13 summaries, selection manifests,
   scoreboards, model/card artifacts, predictions, Step 10 feature schema, and
   processed-data lineage before consolidation. Do not rebuild a model as an
   implicit recovery path.
3. Normalize task/horizon/model/split metrics into one machine-readable
   scoreboard while preserving each task's formal primary metric, supporting
   metrics, runtime, validation evidence, test evidence, class order,
   calibration decision, and accident threshold where applicable.
4. Create exactly one registry entry per selected target/horizon. Each entry
   must identify its model/card, preprocessing and feature versions, data and
   selection hashes, training/validation/test windows, parameters, seed,
   primary metric, limitations, and prediction source.
5. Add explicit selection rationale derived from the already-frozen validation
   evidence. Runtime and interpretability may provide context but must not
   retroactively replace any Step 12 or Step 13 winner.
6. Export dashboard-ready combined predictions or an indexed manifest that
   references the existing versioned prediction artifacts without duplicating
   or fabricating values. Every persisted selected prediction must map to
   exactly one registry entry.
7. Generate canonical JSON/CSV and Markdown registry/scoreboard reports, plus a
   verified loader that rejects missing, stale, or tampered upstream artifacts.
8. Add unit and full-data contracts for 20-job coverage, unique registry keys,
   task-aware metric schemas, lineage completeness, exact prediction mapping,
   deterministic generation, verified model/card resolution, and tamper
   rejection. Add a CLI command for registry construction.

## Acceptance Gate

Step 14 is complete only when:

- Exactly 20 selected classical target/horizon jobs have unique registry
  entries and complete model/card/data/feature/split lineage.
- Regression and classification metrics are normalized without losing their
  task-specific meanings; no classifier is ranked by accuracy and no regression
  model is ranked by a classification metric.
- Existing Step 12 and Step 13 selections, calibration decisions, thresholds,
  predictions, and test results are unchanged.
- Every selected validation/test prediction maps to exactly one registry entry,
  and every registry entry resolves to a verified reloadable artifact.
- The machine-readable registry and scoreboard render into generated Markdown
  and are suitable for later dashboard consumption.
- Focused tests, the full suite, CLI/import smoke, dependency check,
  compilation, whitespace assurance, source-size checks, and artifact
  verification pass.
- `STATUS.md`, `NEXT_STEP.md`, `ROADMAP.md`, `ARCHITECTURE.md`, `STEPS.md`, and
  the README reflect verified Step 14 results before commit/push.

## Current Blockers

None. The unmet Step 13 classifier performance goals are an explicit model-risk
finding, not a blocker to honest registry consolidation.
