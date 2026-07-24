# Model Card: congestion_h3

## Identity

- Model version: `classical_classification_v1`.
- Target: `target_congestion_h3`.
- Horizon: 3 windows (90 minutes).
- Class order: `['Free-flow', 'Moderate', 'Heavy', 'Severe']`.
- Selected family: `xgboost`.
- Candidate: `xgb_deep`.
- Seed: `42`.

## Selection and data

- Hyperparameters were selected by mean CV `macro_f1` across all five frozen training-only folds.
- The family was selected by validation `macro_f1` before test access.
- Training: `2025-01-01T00:00:00+05:30` through `2025-04-16T15:00:00+05:30` (126775 rows).
- Validation: `2025-04-16T17:00:00+05:30` through `2025-05-09T06:30:00+05:30` (27100 rows).
- Test: `2025-05-09T08:30:00+05:30` through `2025-05-31T22:00:00+05:30` (27100 rows).

## Probability and operating decision

- Sigmoid calibration applied: `True` (validation_brier_improved).
- Accident operating threshold: `None`.

## Hyperparameters

```json
{
  "colsample_bytree": 1.0,
  "learning_rate": 0.06,
  "max_depth": 6,
  "min_child_weight": 3,
  "n_estimators": 64,
  "subsample": 1.0
}
```

## Metrics

| Split | macro_f1 | macro_recall | Rows |
|---|---:|---:|---:|
| Validation | 0.7491 | 0.7751 | 27100 |
| Test | 0.7493 | 0.7762 | 27100 |

## Lineage and artifacts

- Processed data SHA-256: `f5377b7f8969d6b74e850d71a803c91f252ec236d5bceeaa02e3e31dedfa81a4`.
- Feature schema SHA-256: `204d2fc3ab00e18a452e4ef2898826cf9dc0bd05dbaf795f8f203ca26f71f453`.
- Selection manifest SHA-256: `af78be183276cd5d48c9d5eb63148457ebb641395693f8ec25925abf2255f835`.
- Classifier: `artifacts/models/classical_classification_v1/congestion_h3.joblib`.
- Predictions: `artifacts/predictions/classical_classification_v1/selected_predictions.parquet`.

## Limitations

- This direct classifier is specific to one target and horizon.
- CV uses a deterministic timestamp budget; the final family fit uses all eligible training rows.
- Calibration is fit on earlier validation rows and assessed on later validation rows before test access.
- Future weather is not available; weather inputs are observed at origin.
