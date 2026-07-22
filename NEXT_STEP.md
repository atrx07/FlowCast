# NEXT_STEP.md

## Immediate Objective

Execute **Step 12 - Train Classical Regression Models**. Build one
configuration-driven, time-safe training and evaluation pipeline for traffic
volume, average speed, and travel time at 30, 60, 90, and 120 minutes using
scikit-learn Linear Regression, Decision Tree, Random Forest, and XGBoost.

Do not begin congestion/accident classification, deep learning, confidence,
inference services, or dashboard work. Keep final test sealed throughout
hyperparameter search and validation comparison; open it exactly once only
after the winning family/configuration for each target/horizon is frozen.

## Read Before Acting

1. `AGENTS.md`, `TECH_STACK.md`, `STATUS.md`, and this file.
2. `STEPS.md` - Step 12 and the proven Step 10/11 procedures.
3. Classical regression, evaluation, registry, artifact, split, metric, and
   reproducibility sections of `PROJECT.md`, `ROADMAP.md`, and
   `ARCHITECTURE.md`.
4. The original PRD sections covering regression targets, required model
   families, time-series cross-validation, model comparison, persistence, and
   RMSE/MAE/MAPE/R-squared.
5. `config/models.yaml`, Step 10 schema/folds/summary, Step 11 metrics and
   limitations, processed target manifest, current Git diff, and relevant tests.

## Single Best Next Action

Build the reusable classical-regression engine and complete Step 12:

1. Extend `config/models.yaml` with bounded, reviewable hyperparameter search
   spaces, search budgets, selection rules, model/artifact version, and seed for
   Linear Regression, Decision Tree, Random Forest, and XGBoost.
2. Hash-verify the frozen Step 10 inputs and load only training/validation for
   tuning. Generate target/horizon jobs from the processed manifest instead of
   copying code across 12 regression tasks.
3. Reuse the approved linear/tree preprocessors and five horizon-gapped
   expanding CV folds. Fit preprocessing/statistics on training only and ensure
   every fold respects target availability and same-partition boundaries.
4. Evaluate bounded candidate configurations by validation/CV RMSE, with MAE,
   MAPE, R-squared, runtime, and fit/prediction failures visible. Never tune by
   final-test performance.
5. Freeze and persist the selected family and hyperparameters separately for
   each of volume, speed, and travel time at horizons 1-4, including a rationale
   and exact validation evidence.
6. Only after all 12 choices are immutable, use explicit
   `purpose="final_evaluation"` access once to calculate final test metrics and
   persisted predictions for the frozen models.
7. Persist reloadable estimator/preprocessor artifacts, candidate and selected
   metrics, predictions, feature importance where supported, input/output
   hashes, split/feature/preprocessor versions, seeds, library versions, and
   complete model cards.
8. Add unit, time-series-CV, availability/leakage, reproducibility, metric,
   artifact-reload, prediction-equality, tamper, and full-data contract tests.
9. Provide a CLI training command and generated machine-readable plus Markdown
   regression scoreboard suitable for the later Step 14 registry.

## Acceptance Gate

Step 12 is complete only when:

- All three regression targets and all four horizons have required Linear
  Regression, Decision Tree, Random Forest, and XGBoost evidence or a precisely
  documented technical failure.
- Candidate tuning is restricted to training/CV/validation and every selected
  target/horizon choice is frozen before any test access.
- Test is opened exactly once for final evaluation of frozen choices; no test
  result feeds hyperparameters, feature selection, or family selection.
- RMSE, MAE, MAPE, and R-squared are finite, consistently calculated from
  persisted predictions, and include row/denominator evidence.
- Reloaded artifacts reproduce stored predictions within documented tolerance.
- Every selected target/horizon model has complete lineage and a model card.
- Training/search budgets remain practical on the 16 GB CPU acceptance
  workstation and any candidate failure is explicit rather than silently
  skipped.
- Focused tests, full tests, CLI smoke, dependency check, compilation,
  whitespace assurance, source-size checks, and deterministic artifact checks
  pass.
- `STATUS.md`, `NEXT_STEP.md`, `ROADMAP.md`, `ARCHITECTURE.md`, `STEPS.md`, and
  the README timeline reflect verified Step 12 results before commit/push.

## Current Blockers

None.
