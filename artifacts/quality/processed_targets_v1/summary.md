# FlowCast Processed Target Report

- Target contract: `multi_horizon_targets_v1`
- Output version: `processed_targets_v1`
- Input feature version: `engineered_features_v1`

## Dataset contract

| Check | Result |
|---|---:|
| Input rows | 181200 |
| Output rows / keys | 181200 / 181200 |
| Row-count change | 0 |
| Duplicate output keys | 0 |
| Preserved input columns | 144 |
| Model-candidate features | 62 |
| Target/horizon definitions | 20 |

## Horizon coverage

| Horizon | Minutes | Available timestamps | Trailing unavailable |
|---:|---:|---:|---:|
| h1 | 30 | 181175 | 25 |
| h2 | 60 | 181150 | 50 |
| h3 | 90 | 181125 | 75 |
| h4 | 120 | 181100 | 100 |

## Target availability

| Target | Available | Unavailable |
|---|---:|---:|
| `target_volume_h1` | 181175 | 25 |
| `target_speed_h1` | 181175 | 25 |
| `target_travel_time_h1` | 181175 | 25 |
| `target_congestion_h1` | 181175 | 25 |
| `target_accident_h1` | 176676 | 4524 |
| `target_volume_h2` | 181150 | 50 |
| `target_speed_h2` | 181150 | 50 |
| `target_travel_time_h2` | 181150 | 50 |
| `target_congestion_h2` | 181150 | 50 |
| `target_accident_h2` | 176651 | 4549 |
| `target_volume_h3` | 181125 | 75 |
| `target_speed_h3` | 181125 | 75 |
| `target_travel_time_h3` | 181125 | 75 |
| `target_congestion_h3` | 181125 | 75 |
| `target_accident_h3` | 176628 | 4572 |
| `target_volume_h4` | 181100 | 100 |
| `target_speed_h4` | 181100 | 100 |
| `target_travel_time_h4` | 181100 | 100 |
| `target_congestion_h4` | 181100 | 100 |
| `target_accident_h4` | 176604 | 4596 |

Each future value is shifted within its road segment and paired with an exact future timestamp. All origin rows and explanatory columns are preserved; trailing targets remain null with an explicit false availability mask.

Accident-risk labels are available only when the future source window was observed. Inserted, unobserved windows remain null instead of being silently treated as no incident.

This file is generated from `summary.json`; edit the pipeline, not this report.
