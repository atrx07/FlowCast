# NEXT_STEP.md

## Immediate Objective

Execute **Step 10 - Freeze Splits and Preprocessing**. Convert the documented
70/15/15 chronological evaluation plan into exact, persisted timestamp
boundaries and build reusable model-family preprocessing that can learn only
from training data.

Do not train or select models, inspect final-test metrics, implement inference
or confidence services, or begin dashboard work.

## Read Before Acting

1. `AGENTS.md`, `TECH_STACK.md`, `STATUS.md`, and this file.
2. `STEPS.md` - Step 10.
3. Split, preprocessing, target-availability, evaluation, artifact, and leakage
   sections of `PROJECT.md`, `ROADMAP.md`, and `ARCHITECTURE.md`.
4. The original PRD/data dictionary where evaluation or field semantics are
   uncertain.
5. `config/features.yaml`, `config/eda.yaml`, the processed schema manifest,
   EDA summary/report, actual timestamp/target coverage, current Git diff, and
   relevant tests.

## Single Best Next Action

Build one frozen split and preprocessing contract:

1. Hash-verify the Step 09 summary and `processed_targets_v1` lineage before
   deriving any boundary or schema.
2. Convert the planned earliest 70%, next 15%, latest 15% origin-time split into
   exact timestamp boundaries shared by every road and persist them in the
   approved model configuration.
3. Define explicit boundary behavior so an origin and its horizon-specific
   target cannot cross from one split into another; record usable counts per
   target/horizon without silently dropping unavailable labels.
4. Define expanding or rolling time-series CV folds strictly inside training.
5. Freeze model-input order from the feature manifest while excluding IDs,
   timestamps, target/mask columns, lineage strings, and other non-features.
6. Build preprocessing by model family with categorical/numeric/boolean handling
   and fit learned statistics only on training rows.
7. Add a sealed-test access guard so tuning services cannot load test rows.
8. Persist deterministic split assignments, feature schemas, preprocessing
   metadata, config/input hashes, and environment lineage.
9. Add split-order, horizon-boundary, training-only-fit, deterministic-artifact,
   tamper-rejection, and loading tests.

## Acceptance Gate

Step 10 is complete only when:

- Every prediction origin has one deterministic train/validation/test assignment
  and exact boundaries are versioned.
- Train timestamps precede validation timestamps, which precede test timestamps,
  for every road.
- Target use is horizon-compatible and no selected origin/target pair crosses a
  split boundary.
- Time-series CV exists only inside training and the final test period is
  inaccessible to tuning code by default.
- Feature order and exclusions are explicit, stable, and traceable to the Step
  08 manifest.
- Learned imputation, encoding, scaling, weighting inputs, and other statistics
  are fit from training data only; validation/test are transform-only.
- Persisted assignments, schemas, preprocessing metadata, and hashes reproduce
  deterministically and fail closed on input/config tampering.
- Focused tests, full tests, CLI smoke, dependency check, compilation, and
  whitespace assurance pass.
- Project-state documents and README timeline are current and every source file
  remains below 400 lines.

## Current Blockers

None.
