# Model Card: volume_h2

## Identity

- Model version: `classical_regression_v1`.
- Target: `target_volume_h2`.
- Horizon: 2 windows (60 minutes).
- Selected family: `random_forest`.
- Candidate: `forest_deep`.
- Seed: `42`.

## Selection and data

- Hyperparameters were selected by mean RMSE across all five frozen training-only expanding-window folds.
- The estimator family was selected by validation RMSE before the test split was opened.
- Training window: `2025-01-01T00:00:00+05:30` through `2025-04-16T15:30:00+05:30` (126800 eligible rows).
- Validation window: `2025-04-16T17:00:00+05:30` through `2025-05-09T07:00:00+05:30` (27125 eligible rows).
- Test window: `2025-05-09T08:30:00+05:30` through `2025-05-31T22:30:00+05:30` (27125 eligible rows).
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
| Validation | 61.1236 | 41.1159 | 10.094% | 0.9518 | 27125 |
| Test | 62.8626 | 42.0109 | 10.263% | 0.9522 | 27125 |

## Lineage and artifacts

- Processed data SHA-256: `f5377b7f8969d6b74e850d71a803c91f252ec236d5bceeaa02e3e31dedfa81a4`.
- Feature schema SHA-256: `204d2fc3ab00e18a452e4ef2898826cf9dc0bd05dbaf795f8f203ca26f71f453`.
- Selection manifest SHA-256: `84c153160c105f73cf49f9e133e9ade982101d355c40f3d51f1567df082828f1`.
- Pipeline: `artifacts/models/classical_regression_v1/volume_h2.joblib`.
- Predictions: `artifacts/predictions/classical_regression_v1/selected_predictions.parquet`.

## Limitations

- This direct model is specific to one target and horizon.
- CV search uses a deterministic timestamp budget spanning each fold; the selected family fit uses every eligible training row.
- Weather inputs are observed at the origin, not future weather forecasts.
- Uncertainty intervals are added in Step 16 and are not part of this card.
