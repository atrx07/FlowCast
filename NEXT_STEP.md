# NEXT_STEP.md

## Immediate Objective

Execute **Step 13 - Train Congestion and Accident Classifiers**. Build one
configuration-driven, time-safe classification pipeline for four-class
congestion and binary accident risk at 30, 60, 90, and 120 minutes.

Required families are Decision Tree, Random Forest, XGBoost, and a scaled SVM
baseline. Do not begin the combined Step 14 registry, deep learning,
confidence/error analysis, inference services, or dashboard work. Step 12 has
final regression test evidence, but no classifier may inspect test labels or
metrics until every Step 13 family, hyperparameter, accident threshold, and
calibration decision is frozen from training/CV/validation evidence.

## Read Before Acting

1. `AGENTS.md`, `TECH_STACK.md`, `STATUS.md`, and this file.
2. `STEPS.md` - Step 13 and the proven Step 10/12 procedures.
3. Classification, imbalance, calibration, evaluation, artifact, registry, and
   reproducibility sections of `PROJECT.md`, `ROADMAP.md`, and
   `ARCHITECTURE.md`.
4. The original PRD sections covering congestion/accident targets, SVM,
   time-series CV, model comparison, persistence, Macro-F1, ROC-AUC, and
   probability outputs.
5. The data dictionary accident/congestion definitions, processed target
   manifest, Step 10 split/class-weight evidence, Step 12 freeze pattern,
   current Git diff, and relevant tests.

## Single Best Next Action

Build the reusable classical-classification engine and complete Step 13:

1. Extend `config/models.yaml` with bounded classifier grids, search budgets,
   probability/calibration policy, accident threshold rule, class ordering,
   artifact version, and seed.
2. Generate eight target/horizon jobs from the processed manifest: congestion
   and accident risk at horizons 1-4. Reuse the exact 62-feature schema,
   horizon-safe eligibility masks, and five frozen expanding CV folds.
3. Fit preprocessing and any class weights only on each fold's training rows.
   Use tree preprocessing for Decision Tree/Random Forest/XGBoost and scaled
   SVM preprocessing for the SVM baseline.
4. Select congestion candidates by mean CV Macro-F1, then validation Macro-F1.
   Persist accuracy, macro precision/recall/F1, per-class metrics, class order,
   confusion matrices, runtime, and probability availability.
5. Select accident candidates by mean CV ROC-AUC, then validation ROC-AUC with
   PR-AUC, precision, recall, F1, confusion matrix, and class imbalance visible.
   Choose the operating threshold from validation probabilities only and record
   threshold analysis. Assess probability calibration and apply it only when
   validation evidence justifies it.
6. Persist all eight frozen classifier choices, accident thresholds, and any
   calibration decisions before one Step 13-scoped
   `purpose="final_evaluation"` test load. Do not refit or change decisions
   after viewing test results.
7. Persist reloadable pipelines, probabilities/predictions, CV and validation
   evidence, final metrics, confusion matrices, threshold/calibration tables,
   hashes, lineage, feature importance where supported, and JSON plus Markdown
   model cards.
8. Add unit, leakage, imbalance, class-order, metric, threshold, probability,
   deterministic training, artifact reload, prediction/probability equality,
   tamper, and full-data contracts. Add a CLI training command and generated
   machine-readable plus Markdown classification scoreboard.

## Acceptance Gate

Step 13 is complete only when:

- All eight classification jobs have Decision Tree, Random Forest, XGBoost, and
  SVM evidence or a precise technical failure.
- Every CV fold and learned preprocessing/weighting statistic is training-only,
  time ordered, horizon safe, and target-availability safe.
- Congestion selection uses Macro-F1 and reports the fixed order Free-flow,
  Moderate, Heavy, Severe with per-class metrics and confusion matrices.
- Accident selection uses ROC-AUC, reports PR-AUC and operating-point metrics,
  and freezes a validation-selected threshold before test access.
- Persisted probabilities are finite, normalized, correctly ordered, and
  reproduce after model reload.
- Step 13 test evaluation occurs only after all classifier decisions are
  frozen; no test result changes a model, threshold, calibration, or feature.
- Every selected classifier has complete lineage and a model card.
- Focused tests, the full suite, CLI smoke, dependency check, compilation,
  whitespace assurance, source-size checks, and artifact verification pass.
- `STATUS.md`, `NEXT_STEP.md`, `ROADMAP.md`, `ARCHITECTURE.md`, `STEPS.md`, and
  the README reflect verified Step 13 results before commit/push.

## Current Blockers

None.
