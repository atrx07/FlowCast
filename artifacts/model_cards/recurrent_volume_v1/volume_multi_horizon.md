# Model Card: volume_multi_horizon

## Identity

- Model version: `recurrent_volume_v1`.
- Candidate: `lstm_s12_h32`.
- Seed: `42`.
- Targets: volume at 30, 60, 90, and 120 minutes.
- Pretrained weights: no.

## Selection and training

- Validation mean RMSE selected epoch 8 before test access.
- Architecture: LSTM, 1 layer(s), hidden size 32, sequence length 12.
- Input features: 62 raw, 64 transformed.
- Feature and target scaling statistics were learned from training only.

## Hold-out metrics

| Horizon | RMSE | MAE | MAPE | R-squared | Rows |
|---:|---:|---:|---:|---:|---:|
| 30 min | 60.1443 | 40.5448 | 10.210% | 0.9561 | 26500 |
| 60 min | 60.8154 | 41.1086 | 10.388% | 0.9552 | 26500 |
| 90 min | 61.2014 | 41.6956 | 10.979% | 0.9546 | 26500 |
| 120 min | 61.8966 | 42.5216 | 11.535% | 0.9536 | 26500 |

## Lineage and artifacts

- Processed data SHA-256: `f5377b7f8969d6b74e850d71a803c91f252ec236d5bceeaa02e3e31dedfa81a4`.
- Selection SHA-256: `63c8a615bd73312942ac2569f64593dfbc86f883943e131b500a77dfc9240bfa`.
- Checkpoint: `artifacts/models/recurrent_volume_v1/best_checkpoint.pt`.
- Predictions: `artifacts/predictions/recurrent_volume_v1/predictions.parquet`.

## Limitations

- The model forecasts volume only; confidence intervals arrive in Step 16.
- Observed origin weather is used; future weather forecasts are unavailable.
- Sequence isolation removes the first sequence_length-1 origins per road from each partition, so classical comparison is restricted to the exact deep-model origin subset.
- The installed PyTorch 2.13.0 build is CPU-only on this workstation.
