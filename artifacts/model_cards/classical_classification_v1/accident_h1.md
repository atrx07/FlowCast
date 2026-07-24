# Model Card: accident_h1

## Identity

- Model version: `classical_classification_v1`.
- Target: `target_accident_h1`.
- Horizon: 1 windows (30 minutes).
- Class order: `['no_accident', 'accident']`.
- Selected family: `svm`.
- Candidate: `svm_regularized`.
- Seed: `42`.

## Selection and data

- Hyperparameters were selected by mean CV `roc_auc` across all five frozen training-only folds.
- The family was selected by validation `roc_auc` before test access.
- Training: `2025-01-01T00:00:00+05:30` through `2025-04-16T16:00:00+05:30` (123716 rows).
- Validation: `2025-04-16T17:00:00+05:30` through `2025-05-09T07:30:00+05:30` (26455 rows).
- Test: `2025-05-09T08:30:00+05:30` through `2025-05-31T23:00:00+05:30` (26456 rows).

## Probability and operating decision

- Sigmoid calibration applied: `True` (required_for_probability_output).
- Accident operating threshold: `0.0133277157`.

## Hyperparameters

```json
{
  "C": 0.1,
  "max_iter": 5000,
  "tol": 0.001
}
```

## Metrics

| Split | roc_auc | pr_auc | Rows |
|---|---:|---:|---:|
| Validation | 0.5763 | 0.0135 | 26455 |
| Test | 0.6209 | 0.0209 | 26456 |

## Lineage and artifacts

- Processed data SHA-256: `f5377b7f8969d6b74e850d71a803c91f252ec236d5bceeaa02e3e31dedfa81a4`.
- Feature schema SHA-256: `204d2fc3ab00e18a452e4ef2898826cf9dc0bd05dbaf795f8f203ca26f71f453`.
- Selection manifest SHA-256: `af78be183276cd5d48c9d5eb63148457ebb641395693f8ec25925abf2255f835`.
- Classifier: `artifacts/models/classical_classification_v1/accident_h1.joblib`.
- Predictions: `artifacts/predictions/classical_classification_v1/selected_predictions.parquet`.

## Limitations

- This direct classifier is specific to one target and horizon.
- CV uses a deterministic timestamp budget; the final family fit uses all eligible training rows.
- Calibration is fit on earlier validation rows and assessed on later validation rows before test access.
- Future weather is not available; weather inputs are observed at origin.
