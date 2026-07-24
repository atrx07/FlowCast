# FlowCast Recurrent Volume Forecaster

## Frozen evaluation contract

- Version: `recurrent_volume_v1`; seed: `42`.
- Selected candidate: `lstm_s12_h32` (LSTM, sequence length 12).
- Best epoch: 8; stopped epoch: 11; device: `cpu`.
- Candidate selection and best-checkpoint persistence occurred before the single explicit test-partition load.
- Test metrics were not used for architecture or checkpoint selection.

## Multi-horizon hold-out metrics

| Horizon | Validation RMSE | Test RMSE | Test MAE | Test MAPE | Test R-squared |
|---:|---:|---:|---:|---:|---:|
| 30 min | 58.1037 | 60.1443 | 40.5448 | 10.210% | 0.9561 |
| 60 min | 58.8066 | 60.8154 | 41.1086 | 10.388% | 0.9552 |
| 90 min | 59.0322 | 61.2014 | 41.6956 | 10.979% | 0.9546 |
| 120 min | 59.5989 | 61.8966 | 42.5216 | 11.535% | 0.9536 |

## Exact-row classical comparison

| Horizon | Shared rows | Deep RMSE | Classical RMSE | Delta | Deep wins |
|---:|---:|---:|---:|---:|---|
| 30 min | 26500 | 60.1443 | 63.2354 | -3.0910 | yes |
| 60 min | 26500 | 60.8154 | 62.6833 | -1.8678 | yes |
| 90 min | 26500 | 61.2014 | 65.0565 | -3.8551 | yes |
| 120 min | 26500 | 61.8966 | 61.8495 | 0.0471 | no |

The recurrent model beats the frozen classical volume model at 3 of 4 horizons on the exact shared test origins.

## Sequence and persistence checks

- Training sequences: 126475; validation sequences: 26500; test sequences: 26500.
- Cross-road, cross-partition, non-contiguous, and target-boundary violations: 0.
- Feature and target scaling statistics originate from training only.
- Reloaded checkpoint inference reproduces the persisted predictions.

## Limitations

- Confidence intervals are deferred to Step 16.
- Future weather forecasts are not available; weather is known only at origin.
- This workstation used the CPU-only PyTorch 2.13.0 build.
