# FlowCast NumPy Linear Regression Proof

- Contract: `scratch_linear_v1`
- Version: `scratch_linear_v1`
- Demonstration target: `target_volume_h1`
- Purpose: mathematical verification only; no production model selection
- Test partition rows loaded: **0**

## Mathematics implemented directly

Prediction uses `X @ w + b`. Mean squared error, its analytical weight and bias gradients, seeded initialization, and the full-batch update loop are implemented in `flowcast.modelling.scratch_linear` with NumPy.

## Gradient and synthetic proofs

All 6 parameters passed central finite-difference checks. Maximum absolute error: `2.642e-09`; maximum relative error: `3.388e-09`.

Synthetic loss decreased from `10.5942952212` to `0.0000000000`. Maximum coefficient error was `1.989e-08` and bias error was `6.850e-09`.

## FlowCast data slice

The earliest 25,000 eligible training rows (2025-01-01T00:00:00+05:30 through 2025-01-21T19:30:00+05:30) were selected after chronological sorting. The unchanged validation partition contributes 27,150 eligible rows. Both estimators consume the same 62 manifest inputs and 64 preprocessed columns.

## Validation comparison

| Estimator | RMSE | MAE | MAPE | R-squared |
|---|---:|---:|---:|---:|
| NumPy gradient descent | 86.9331 | 62.9818 | 18.2032% | 0.902682 |
| scikit-learn LinearRegression | 80.8723 | 59.1634 | 16.5849% | 0.915778 |

The scratch loss decreased from `278398.005590` to `6716.115998` over 4,153 updates. These validation results prove the implementation; Step 12 performs model-family training and selection without changing the frozen split.

## Limitations

- This Step 11 slice demonstrates the mathematics on next-window volume only.
- The earliest eligible training subset is bounded for an auditable full-batch gradient loop.
- Validation metrics are not final hold-out metrics and do not select a production model.
- The final test partition remains sealed until Step 12 choices are frozen.

This report is generated from the canonical JSON, persisted convergence history, coefficients, validation predictions, and hash-verified Step 10 lineage; edit the pipeline, not this report.
