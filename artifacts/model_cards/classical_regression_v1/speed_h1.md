# Model Card: speed_h1

## Identity

- Model version: `classical_regression_v1`.
- Target: `target_speed_h1`.
- Horizon: 1 windows (30 minutes).
- Selected family: `random_forest`.
- Candidate: `forest_deep`.
- Seed: `42`.

## Selection and data

- Hyperparameters were selected by mean RMSE across all five frozen training-only expanding-window folds.
- The estimator family was selected by validation RMSE before the test split was opened.
- Training window: `2025-01-01T00:00:00+05:30` through `2025-04-16T16:00:00+05:30` (126825 eligible rows).
- Validation window: `2025-04-16T17:00:00+05:30` through `2025-05-09T07:30:00+05:30` (27150 eligible rows).
- Test window: `2025-05-09T08:30:00+05:30` through `2025-05-31T23:00:00+05:30` (27150 eligible rows).
- Input features: 62; transformed features: 64.
- Preprocessing version: `split_preprocessing_v1`.

## Hyperparameters

```json
{
  "max_depth": 14,
  "max_features": 0.9,
  "min_samples_leaf": 4,
  "n_estimators": 24
}
```

## Metrics

| Split | RMSE | MAE | MAPE | R-squared | Rows |
|---|---:|---:|---:|---:|---:|
| Validation | 3.7230 | 2.8128 | 8.839% | 0.8928 | 27150 |
| Test | 3.7400 | 2.8214 | 9.029% | 0.8980 | 27150 |

## Lineage and artifacts

- Processed data SHA-256: `f5377b7f8969d6b74e850d71a803c91f252ec236d5bceeaa02e3e31dedfa81a4`.
- Feature schema SHA-256: `204d2fc3ab00e18a452e4ef2898826cf9dc0bd05dbaf795f8f203ca26f71f453`.
- Selection manifest SHA-256: `792df7995558360c99f28f13eb71e8fed182ada164c10582577cd05d5b9e3fd4`.
- Pipeline: `artifacts/models/classical_regression_v1/speed_h1.joblib`.
- Predictions: `artifacts/predictions/classical_regression_v1/selected_predictions.parquet`.

## Limitations

- This direct model is specific to one target and horizon.
- CV search uses a deterministic timestamp budget spanning each fold; the selected family fit uses every eligible training row.
- Weather inputs are observed at the origin, not future weather forecasts.
- Uncertainty intervals are added in Step 16 and are not part of this card.
