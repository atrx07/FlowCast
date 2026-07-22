# FlowCast Data Quality and EDA Report

- EDA contract: `eda_report_v1`
- Output version: `eda_v1`
- Processed input: `processed_targets_v1`
- Coverage: 2025-01-01T00:00:00+05:30 to 2025-05-31T23:30:00+05:30

## Source-to-processed reconciliation

| Stage | Rows / keys | Notable accounting |
|---|---:|---|
| Delivered sources | 189,491 | Traffic 178,468; weather 10,872; calendar 151 |
| Validation | 187,724 retained | 1,767 rejected; 42,792 issues |
| Complete traffic grid | 181,200 | 4,499 missing windows reconstructed |
| Merge | 181,200 | 0 weather and 0 calendar misses |
| Features | 181,200 | 62 model-candidate features |
| Processed targets | 181,200 | 20 target/horizon definitions |

All persisted reconciliation checks passed. No stage has an unexplained row loss, key multiplication, or context-join miss.

## Data defects and repair evidence

| Defect / action | Count |
|---|---:|
| Exact/key traffic duplicates | 1,767 |
| Entirely missing traffic windows | 4,499 |
| Negative traffic volumes | 241 |
| Speeds above 200 km/h | 237 |
| Occupancy above 100% | 234 |
| Blank congestion labels | 26,883 |
| Congestion labels derived after grid completion | 31,123 |
| Vehicle-share rows normalized | 60,347 |
| Accident windows retained as unknown | 4,499 |

### Traffic imputation

| Field | Missing after grid | Imputed | Remaining |
|---|---:|---:|---:|
| `avg_speed` | 9,078 | 9,078 | 0 |
| `occupancy` | 9,072 | 9,072 | 0 |
| `signal_timing` | 4,499 | 4,499 | 0 |
| `traffic_volume` | 9,086 | 9,086 | 0 |
| `travel_time` | 4,499 | 4,499 | 0 |
| `vehicle_count` | 4,499 | 4,499 | 0 |
| `vehicle_type_dist` | 4,499 | 4,499 | 0 |

Weather temperature and visibility imputations are causal, station-local forward fills. Traffic repairs retain their method and donor lineage in the processed data.

## Descriptive statistics

| Measure | Count | Mean | Median | Std. dev. | Min | Max | Skew |
|---|---:|---:|---:|---:|---:|---:|---:|
| `traffic_volume` | 181,200 | 431.517 | 362.000 | 283.665 | 41.000 | 2090.000 | 1.005 |
| `avg_speed` | 181,200 | 42.099 | 44.900 | 11.398 | 6.700 | 64.800 | -1.012 |
| `occupancy` | 181,200 | 31.177 | 27.000 | 19.627 | 3.000 | 100.000 | 0.838 |
| `travel_time` | 181,200 | 3.566 | 3.120 | 2.530 | 0.800 | 29.910 | 3.648 |
| `temperature` | 181,200 | 17.973 | 18.000 | 4.371 | 5.800 | 29.800 | -0.002 |
| `rainfall` | 181,200 | 0.191 | 0.000 | 1.057 | 0.000 | 8.900 | 6.029 |
| `visibility` | 181,200 | 9675.872 | 10000.000 | 1495.792 | 301.000 | 10000.000 | -4.569 |

## Target balance

| Congestion class | Rows | Share |
|---|---:|---:|
| Free-flow | 111,307 | 61.43% |
| Moderate | 43,168 | 23.82% |
| Heavy | 16,721 | 9.23% |
| Severe | 10,004 | 5.52% |

Observed accident labels contain 1,652 positives across 176,701 observed windows (0.935%). The negative-to-positive ratio is 106.0:1. The 4,499 unknown windows are excluded.

## Measured findings

- **Highest Volume Road:** NL-006 has the highest mean volume at 565.69 vehicles/window.
- **Peak Hour:** Local hour 8 has the highest mean volume at 886.01.
- **Slowest Weather:** Rain has the lowest mean speed by weather condition at 40.27 km/h.
- **Congestion Balance:** Free-flow accounts for 61.43% of origins; Severe accounts for 5.52%.
- **Accident Imbalance:** Observed accident positives are 0.935% (106.0:1 negatives to positives).
- **Volume Predictor:** traffic_volume has the strongest configured linear association with next-window volume (r=0.9295).

## Correlation and redundancy

The configured correlation matrix contains 25 origin-time features. 3 pairs have absolute correlation at or above 0.95.

| Feature | Correlation with h1 volume | Observations |
|---|---:|---:|
| `traffic_volume` | 0.9295 | 181,175 |
| `volume_capacity_ratio` | 0.8769 | 181,175 |
| `occupancy` | 0.8646 | 181,175 |
| `volume_lag_1` | 0.8374 | 181,150 |
| `volume_lag_48` | 0.8363 | 179,975 |
| `avg_speed` | -0.8173 | 181,175 |
| `capacity_headroom` | -0.7668 | 181,175 |
| `signal_timing` | 0.7303 | 181,175 |
| `speed_lag_48` | -0.7257 | 179,975 |
| `speed_lag_1` | -0.7186 | 181,150 |

## Modelling implications

- **Split:** Use one chronological split; never use a random split. Evidence: Strong hour/weekday structure and lag dependence are temporal.
- **Scaling:** Fit scaling on training only for linear, SVM, and recurrent models; retain unscaled inputs for tree models. Evidence: Configured numeric ranges differ materially; volume spans 41 to 2090.
- **Congestion:** Select/tune with Macro-F1 and inspect per-class recall. Evidence: The four congestion classes are materially imbalanced.
- **Accident:** Use training-only class weights, ROC-AUC plus PR-AUC, and select the operating threshold on validation data. Evidence: Observed accident positives are 0.935% (106.0:1 negatives to positives).
- **Redundancy:** Review highly correlated pairs inside training folds; do not remove features from full-data EDA alone. Evidence: 3 configured feature pairs meet the redundancy threshold.
- **History:** Keep origins and apply model-specific history availability rather than globally dropping rows. Evidence: Leading lag/rolling nulls are expected and explicitly flagged.

## Bias and limitations

- Associations are observational and must not be interpreted as causal effects.
- The data covers one corridor and 151 days, limiting geographic and seasonal generalization.
- Reconstructed traffic windows use documented causal recovery and may smooth short-lived extremes.
- Accident status is unknown for inserted sensor windows and must remain excluded from classifier labels.
- Hourly weather is shared by both half-hour traffic windows and cannot capture sub-hour variation.

## Exported figures

- `traffic_distributions`: `artifacts/figures/eda_v1/traffic_distributions.png`
- `hourly_profiles`: `artifacts/figures/eda_v1/hourly_profiles.png`
- `road_comparison`: `artifacts/figures/eda_v1/road_comparison.png`
- `class_balance`: `artifacts/figures/eda_v1/class_balance.png`
- `weather_traffic`: `artifacts/figures/eda_v1/weather_traffic.png`
- `correlation_heatmap`: `artifacts/figures/eda_v1/correlation_heatmap.png`

This report is generated from persisted pipeline counters and the hash-verified processed dataset; edit the pipeline, not this report.
