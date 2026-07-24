# Model Card: congestion_h4

## Identity

- Model version: `classical_classification_v1`.
- Target: `target_congestion_h4`.
- Horizon: 4 windows (120 minutes).
- Class order: `['Free-flow', 'Moderate', 'Heavy', 'Severe']`.
- Selected family: `random_forest`.
- Candidate: `forest_balanced`.
- Seed: `42`.

## Selection and data

- Hyperparameters were selected by mean CV `macro_f1` across all five frozen training-only folds.
- The family was selected by validation `macro_f1` before test access.
- Training: `2025-01-01T00:00:00+05:30` through `2025-04-16T14:30:00+05:30` (126750 rows).
- Validation: `2025-04-16T17:00:00+05:30` through `2025-05-09T06:00:00+05:30` (27075 rows).
- Test: `2025-05-09T08:30:00+05:30` through `2025-05-31T21:30:00+05:30` (27075 rows).

## Probability and operating decision

- Sigmoid calibration applied: `True` (validation_brier_improved).
- Accident operating threshold: `None`.

## Hyperparameters

```json
{
  "max_depth": 12,
  "max_features": 0.7,
  "min_samples_leaf": 6,
  "n_estimators": 24
}
```

## Metrics

| Split | macro_f1 | macro_recall | Rows |
|---|---:|---:|---:|
| Validation | 0.7337 | 0.7684 | 27075 |
| Test | 0.7468 | 0.7751 | 27075 |

## Lineage and artifacts

- Processed data SHA-256: `f5377b7f8969d6b74e850d71a803c91f252ec236d5bceeaa02e3e31dedfa81a4`.
- Feature schema SHA-256: `204d2fc3ab00e18a452e4ef2898826cf9dc0bd05dbaf795f8f203ca26f71f453`.
- Selection manifest SHA-256: `af78be183276cd5d48c9d5eb63148457ebb641395693f8ec25925abf2255f835`.
- Classifier: `artifacts/models/classical_classification_v1/congestion_h4.joblib`.
- Predictions: `artifacts/predictions/classical_classification_v1/selected_predictions.parquet`.

## Limitations

- This direct classifier is specific to one target and horizon.
- CV uses a deterministic timestamp budget; the final family fit uses all eligible training rows.
- Calibration is fit on earlier validation rows and assessed on later validation rows before test access.
- Future weather is not available; weather inputs are observed at origin.
