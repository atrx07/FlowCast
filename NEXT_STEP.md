# NEXT_STEP.md

## Immediate Objective

Execute **Step 11 - Implement NumPy Linear Regression**. Build and verify the
required regression mathematics directly with NumPy, then compare it against
scikit-learn Linear Regression on the same frozen training/validation data.

Do not open the final test partition, train the remaining classical model
families, select a production winner, implement confidence/inference, or begin
dashboard work.

## Read Before Acting

1. `AGENTS.md`, `TECH_STACK.md`, `STATUS.md`, and this file.
2. `STEPS.md` - Step 11.
3. Regression mathematics, evaluation, split, preprocessing, metric, and
   artifact sections of `PROJECT.md`, `ROADMAP.md`, and `ARCHITECTURE.md`.
4. The original PRD sections covering the build-it-once rule, linear algebra,
   gradient descent, loss functions, chronological validation, and model
   comparison.
5. `config/models.yaml`, Step 10 summary/schema/report, the persisted linear
   preprocessor, train/validation target eligibility, current Git diff, and
   relevant tests.

## Single Best Next Action

Build one reproducible scratch linear-regression slice for next-window volume:

1. Hash-verify all Step 10 artifacts and load only the training and validation
   partitions; prove the test loader remains sealed.
2. Implement pure NumPy prediction `X @ w + b`, MSE loss, analytical weight and
   bias gradients, and an explicit gradient-descent loop with seeded
   initialization, bounded iterations, convergence tolerance, and loss history.
3. Add central finite-difference gradient checks for weights and bias on a
   controlled synthetic problem.
4. Prove loss convergence and parameter recovery on deterministic synthetic
   data before using FlowCast data.
5. Use the persisted linear preprocessor and the boundary-safe
   `target_volume_h1` availability contract on a documented chronological
   training subset or the full eligible training partition.
6. Compare scratch and scikit-learn Linear Regression on the exact same
   preprocessed rows and validation slice using RMSE, MAE, MAPE, and R-squared;
   this is mathematical validation, not final model selection.
7. Persist configuration, convergence history, coefficients, metrics,
   input/preprocessor hashes, seed, row/time coverage, and limitations in
   machine-readable plus generated human-readable artifacts.
8. Add unit, reproducibility, numerical-gradient, convergence, loader, and
   full-data validation tests.

## Acceptance Gate

Step 11 is complete only when:

- Prediction, MSE, analytical gradients, and gradient descent are visibly
  implemented with NumPy rather than delegated to a modelling library.
- Central finite-difference gradient checks pass within a documented tolerance
  for every weight and the bias.
- Synthetic loss decreases reproducibly and fitted parameters recover the
  controlled solution within tolerance.
- Scratch and scikit-learn models consume identical Step 10 features, eligible
  training rows, and validation rows.
- Convergence history and comparison metrics are honest, finite, deterministic,
  and persisted with complete split/preprocessor lineage.
- The final test partition remains sealed and no test metric is calculated.
- Focused tests, full tests, CLI smoke, dependency check, compilation, and
  whitespace assurance pass.
- Project-state documents and README timeline are current and every source file
  remains below 400 lines.

## Current Blockers

None.
