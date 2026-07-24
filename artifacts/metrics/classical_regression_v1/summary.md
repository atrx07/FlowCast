# FlowCast Classical Regression

## Evaluation contract

- Version: `classical_regression_v1`; seed: `42`.
- Jobs: 12 (3 targets x 4 horizons).
- Search: 7 candidate configurations across 4 required families and 5 expanding folds.
- Candidate hyperparameters were selected by mean CV RMSE; model family was selected by validation RMSE.
- The selection manifest was persisted before the single explicit final-evaluation test load.

## Frozen hold-out scoreboard

| Target | Horizon | Selected family | Validation RMSE | Test RMSE | Test MAE | Test MAPE | Test R-squared |
|---|---:|---|---:|---:|---:|---:|---:|
| volume | 30 | random_forest | 61.1926 | 63.4595 | 42.2054 | 10.218% | 0.9514 |
| volume | 60 | random_forest | 61.1236 | 62.8626 | 42.0109 | 10.263% | 0.9522 |
| volume | 90 | random_forest | 65.6565 | 65.3058 | 44.0438 | 10.952% | 0.9483 |
| volume | 120 | random_forest | 60.5924 | 62.0092 | 41.6263 | 10.295% | 0.9533 |
| speed | 30 | random_forest | 3.7230 | 3.7400 | 2.8214 | 9.029% | 0.8980 |
| speed | 60 | random_forest | 3.7591 | 3.7683 | 2.8538 | 9.056% | 0.8960 |
| speed | 90 | random_forest | 3.8544 | 3.7940 | 2.8715 | 9.145% | 0.8944 |
| speed | 120 | random_forest | 3.8419 | 3.7923 | 2.8731 | 9.085% | 0.8945 |
| travel_time | 30 | random_forest | 1.1005 | 1.1426 | 0.4610 | 9.753% | 0.8065 |
| travel_time | 60 | random_forest | 1.0853 | 1.0949 | 0.4393 | 9.316% | 0.8210 |
| travel_time | 90 | random_forest | 1.0870 | 1.0822 | 0.4291 | 9.012% | 0.8247 |
| travel_time | 120 | random_forest | 1.0823 | 1.1016 | 0.4381 | 9.203% | 0.8184 |

## Coverage and persistence

- Required family/task CV results: 48 of 48.
- Selected reloadable pipelines: 12.
- Machine-readable model cards: 12; Markdown model cards: 12.
- Persisted prediction rows: 650700 across validation and test.

## Runtime

- CV fit time: 20.681s; CV prediction time: 10.938s.
- Full-training family fit time: 64.411s; validation prediction time: 2.832s.
- Frozen test prediction time: 0.827s.

## Limitations

- CV search is bounded to evenly spaced origin timestamps spanning each full expanding training interval.
- Final family fits use the complete eligible training partition; validation and test are transform-only.
- Speed is included because the approved product objectives require multi-horizon average-speed forecasts.
- Prediction confidence and segmented error analysis remain Step 16 work.
