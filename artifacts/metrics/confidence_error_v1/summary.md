# FlowCast Confidence and Error Analysis

## Contract

- Regression uncertainty uses validation-only finite-sample split-conformal absolute residuals.
- Classification uncertainty exposes maximum probability and normalized entropy from frozen probabilities.
- Accident risk bands are relative to each frozen validation-selected operating threshold.
- All slices are descriptive; no model, threshold, calibrator, split, or prediction was changed.

## Regression test intervals

| Model | Target | Horizon | RMSE | Coverage | Mean width |
|---|---|---:|---:|---:|---:|
| classical_regression_v1 | speed | 30 min | 3.7400 | 0.8980 | 11.7230 |
| classical_regression_v1 | speed | 60 min | 3.7683 | 0.8966 | 11.8512 |
| classical_regression_v1 | speed | 90 min | 3.7940 | 0.9048 | 12.2148 |
| classical_regression_v1 | speed | 120 min | 3.7923 | 0.9035 | 12.1641 |
| classical_regression_v1 | travel_time | 30 min | 1.1426 | 0.8960 | 1.9934 |
| classical_regression_v1 | travel_time | 60 min | 1.0949 | 0.9008 | 2.0008 |
| classical_regression_v1 | travel_time | 90 min | 1.0822 | 0.9055 | 1.9881 |
| classical_regression_v1 | travel_time | 120 min | 1.1016 | 0.9019 | 1.9507 |
| classical_regression_v1 | volume | 30 min | 63.4595 | 0.8997 | 198.3949 |
| classical_regression_v1 | volume | 60 min | 62.8626 | 0.8997 | 197.1927 |
| classical_regression_v1 | volume | 90 min | 65.3058 | 0.9018 | 209.3654 |
| classical_regression_v1 | volume | 120 min | 62.0092 | 0.9029 | 198.7756 |
| recurrent_volume_v1 | volume | 30 min | 60.1443 | 0.8969 | 185.4537 |
| recurrent_volume_v1 | volume | 60 min | 60.8154 | 0.8951 | 186.9975 |
| recurrent_volume_v1 | volume | 90 min | 61.2014 | 0.8924 | 186.3718 |
| recurrent_volume_v1 | volume | 120 min | 61.8966 | 0.8931 | 190.1864 |

## Classification and paired-model findings

- Congestion test Macro-F1 by horizon: {'1': 0.7539697309, '2': 0.7502815026, '3': 0.7492509117, '4': 0.7467869858}.
- Accident test ROC-AUC by horizon: {'1': 0.6209075489, '2': 0.6236733832, '3': 0.5980192851, '4': 0.5894100267}.
- Congestion test expected calibration error by horizon: {'1': 0.0028863407506180506, '2': 0.05386028010905614, '3': 0.0514892826140485, '4': 0.06023797487768234}.
- Dominant congestion off-diagonal confusion: {'horizon_windows': 2, 'actual_label': 'Free-flow', 'predicted_label': 'Moderate', 'rows': 1263}.
- The recurrent model wins test RMSE on 3 of 4 exact paired horizons.
- Worst supported paired slices: [{'horizon_minutes': 120, 'dimension': 'origin_hour', 'slice_value': '22', 'rows': 1100, 'rmse_delta_deep_minus_classical': 10.0930871341}, {'horizon_minutes': 60, 'dimension': 'origin_hour', 'slice_value': '23', 'rows': 1100, 'rmse_delta_deep_minus_classical': 8.628629840399999}, {'horizon_minutes': 120, 'dimension': 'origin_hour', 'slice_value': '23', 'rows': 1100, 'rmse_delta_deep_minus_classical': 6.852612916000002}, {'horizon_minutes': 120, 'dimension': 'origin_hour', 'slice_value': '0', 'rows': 1100, 'rmse_delta_deep_minus_classical': 5.448936061600001}, {'horizon_minutes': 30, 'dimension': 'origin_hour', 'slice_value': '23', 'rows': 1100, 'rmse_delta_deep_minus_classical': 5.007662693399997}].

## Interpretation guardrails

- Subgroups below configured row or positive-event support remain in the CSV with `sufficient_support=false` and blank metrics.
- Interval coverage is an empirical hold-out diagnostic, not a guarantee for future distribution shift.
- Low accident prevalence makes PR-AUC, precision, and supported positive counts essential companions to ROC-AUC.
- Slice differences are associations and must not be treated as causal.
