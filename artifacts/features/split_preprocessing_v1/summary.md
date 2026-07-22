# FlowCast Frozen Split and Preprocessing Report

- Contract: `split_preprocessing_v1`
- Version: `split_preprocessing_v1`
- Processed input: `processed_targets_v1`
- Feature input: `engineered_features_v1`

## Chronological partitions

| Partition | Start | End | Timestamps | Rows | Share |
|---|---|---|---:|---:|---:|
| Train | 2025-01-01T00:00:00+05:30 | 2025-04-16T16:30:00+05:30 | 5,074 | 126,850 | 70.01% |
| Validation | 2025-04-16T17:00:00+05:30 | 2025-05-09T08:00:00+05:30 | 1,087 | 27,175 | 15.00% |
| Test | 2025-05-09T08:30:00+05:30 | 2025-05-31T23:30:00+05:30 | 1,087 | 27,175 | 15.00% |

Every origin is assigned exactly once. A target is eligible only when its future timestamp stays inside the origin partition and its target-specific availability mask is true.

## Time-series cross-validation

Five expanding-window folds live wholly inside training. Each uses a four-window gap, covering the maximum 120-minute forecast horizon, followed by a seven-day validation window.

| Fold | Train end | Gap | Validation start | Validation end |
|---:|---|---:|---|---|
| 1 | 2025-03-12T14:30:00+05:30 | 4 windows | 2025-03-12T17:00:00+05:30 | 2025-03-19T16:30:00+05:30 |
| 2 | 2025-03-19T14:30:00+05:30 | 4 windows | 2025-03-19T17:00:00+05:30 | 2025-03-26T16:30:00+05:30 |
| 3 | 2025-03-26T14:30:00+05:30 | 4 windows | 2025-03-26T17:00:00+05:30 | 2025-04-02T16:30:00+05:30 |
| 4 | 2025-04-02T14:30:00+05:30 | 4 windows | 2025-04-02T17:00:00+05:30 | 2025-04-09T16:30:00+05:30 |
| 5 | 2025-04-09T14:30:00+05:30 | 4 windows | 2025-04-09T17:00:00+05:30 | 2025-04-16T16:30:00+05:30 |

## Feature and preprocessing contract

The schema contains 62 origin-time features from the Step 07 manifest. Keys, raw lineage, timestamps, targets, and availability masks are excluded.

| Family | Input features | Output features | Numeric | Bounded |
|---|---:|---:|---|---|
| linear | 62 | 64 | standard | standard |
| tree | 62 | 64 | none | none |
| svm | 62 | 64 | standard | standard |
| recurrent | 62 | 64 | standard | minmax |

All imputers, encoders, and scalers above were fit on training rows only. Validation was transform-only. The test partition is sealed by default and requires the explicit `final_evaluation` purpose.
Training-only class counts, balanced weights, and accident `scale_pos_weight` values are persisted in the feature schema for later classifiers; validation and test labels do not influence them.

## Target eligibility

| Target | Horizon | Train | Validation | Test |
|---|---:|---:|---:|---:|
| `target_volume_h1` | 1 | 126,825 | 27,150 | 27,150 |
| `target_speed_h1` | 1 | 126,825 | 27,150 | 27,150 |
| `target_travel_time_h1` | 1 | 126,825 | 27,150 | 27,150 |
| `target_congestion_h1` | 1 | 126,825 | 27,150 | 27,150 |
| `target_accident_h1` | 1 | 123,716 | 26,455 | 26,456 |
| `target_volume_h2` | 2 | 126,800 | 27,125 | 27,125 |
| `target_speed_h2` | 2 | 126,800 | 27,125 | 27,125 |
| `target_travel_time_h2` | 2 | 126,800 | 27,125 | 27,125 |
| `target_congestion_h2` | 2 | 126,800 | 27,125 | 27,125 |
| `target_accident_h2` | 2 | 123,691 | 26,431 | 26,432 |
| `target_volume_h3` | 3 | 126,775 | 27,100 | 27,100 |
| `target_speed_h3` | 3 | 126,775 | 27,100 | 27,100 |
| `target_travel_time_h3` | 3 | 126,775 | 27,100 | 27,100 |
| `target_congestion_h3` | 3 | 126,775 | 27,100 | 27,100 |
| `target_accident_h3` | 3 | 123,668 | 26,407 | 26,409 |
| `target_volume_h4` | 4 | 126,750 | 27,075 | 27,075 |
| `target_speed_h4` | 4 | 126,750 | 27,075 | 27,075 |
| `target_travel_time_h4` | 4 | 126,750 | 27,075 | 27,075 |
| `target_congestion_h4` | 4 | 126,750 | 27,075 | 27,075 |
| `target_accident_h4` | 4 | 123,644 | 26,383 | 26,384 |

This report is generated from the hash-verified processed dataset, EDA lineage, frozen model config, split assignments, CV folds, and fitted preprocessing metadata; edit the pipeline, not this report.
