# FlowCast Feature Engineering Report

- Feature contract: `explanatory_features_v1`
- Output version: `engineered_features_v1`
- Input merge version: `merged_sources_v1`

## Dataset contract

| Check | Result |
|---|---:|
| Input rows | 181200 |
| Output rows / keys | 181200 / 181200 |
| Row-count change | 0 |
| Duplicate output keys | 0 |
| Model-candidate features | 62 |
| History-available rows | 180000 |
| History-unavailable rows | 1200 |

## Expected history nulls

| Feature | Null rows |
|---|---:|
| volume_lag_1 | 25 |
| volume_lag_2 | 50 |
| volume_lag_48 | 1200 |
| volume_rolling_mean_4 | 100 |
| volume_rolling_std_4 | 100 |
| volume_rolling_mean_8 | 200 |
| volume_rolling_std_8 | 200 |
| speed_lag_1 | 25 |
| speed_lag_2 | 50 |
| speed_lag_48 | 1200 |
| speed_rolling_mean_4 | 100 |
| speed_rolling_std_4 | 100 |
| speed_rolling_mean_8 | 200 |
| speed_rolling_std_8 | 200 |

Lags are computed within each road. Rolling features shift one window before applying the configured full-width rolling mean or sample standard deviation, so the current and future rows are excluded.

All source, imputation, and inserted-window lineage columns remain in the feature Parquet. The JSON manifest records every model-candidate feature's dtype, source columns, transform, version, and leakage status.

This file is generated from `summary.json`; edit the pipeline, not this report.
