# Model Card: speed_h3

## Identity

- Model version: `classical_regression_v1`.
- Target: `target_speed_h3`.
- Horizon: 3 windows (90 minutes).
- Selected family: `random_forest`.
- Candidate: `forest_balanced`.
- Seed: `42`.

## Selection and data

- Hyperparameters were selected by mean RMSE across all five frozen training-only expanding-window folds.
- The estimator family was selected by validation RMSE before the test split was opened.
- Training window: `2025-01-01T00:00:00+05:30` through `2025-04-16T15:00:00+05:30` (126775 eligible rows).
- Validation window: `2025-04-16T17:00:00+05:30` through `2025-05-09T06:30:00+05:30` (27100 eligible rows).
- Test window: `2025-05-09T08:30:00+05:30` through `2025-05-31T22:00:00+05:30` (27100 eligible rows).
- Input features: 62; transformed features: 64.
- Preprocessing version: `split_preprocessing_v1`.

## Hyperparameters

```json
{
  "max_depth": 10,
  "max_features": 0.7,
  "min_samples_leaf": 8,
  "n_estimators": 12
}
```

## Metrics

| Split | RMSE | MAE | MAPE | R-squared | Rows |
|---|---:|---:|---:|---:|---:|
| Validation | 3.8544 | 2.9143 | 9.085% | 0.8844 | 27100 |
| Test | 3.7940 | 2.8715 | 9.145% | 0.8944 | 27100 |

## Lineage and artifacts

- Processed data SHA-256: `f5377b7f8969d6b74e850d71a803c91f252ec236d5bceeaa02e3e31dedfa81a4`.
- Feature schema SHA-256: `204d2fc3ab00e18a452e4ef2898826cf9dc0bd05dbaf795f8f203ca26f71f453`.
- Selection manifest SHA-256: `792df7995558360c99f28f13eb71e8fed182ada164c10582577cd05d5b9e3fd4`.
- Pipeline: `artifacts/models/classical_regression_v1/speed_h3.joblib`.
- Predictions: `artifacts/predictions/classical_regression_v1/selected_predictions.parquet`.

## Limitations

- This direct model is specific to one target and horizon.
- CV search uses a deterministic timestamp budget spanning each fold; the selected family fit uses every eligible training row.
- Weather inputs are observed at the origin, not future weather forecasts.
- Uncertainty intervals are added in Step 16 and are not part of this card.
