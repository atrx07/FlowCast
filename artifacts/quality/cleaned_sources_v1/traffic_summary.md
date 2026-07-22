# FlowCast Traffic Cleaning Report

- Cleaning contract: `traffic_cleaning_v1`
- Output version: `cleaned_sources_v1`
- Input validation version: `validated_v1`

## Grid and lineage

| Check | Result |
|---|---:|
| Validated input rows | 176701 |
| Complete output rows | 181200 |
| Roads | 25 |
| Inserted missing windows | 4499 |
| Duplicate rows accounted | 1767 |
| Metadata inconsistencies | 0 |

## Causal recovery

| Field | Missing after grid | Maximum run | Imputed | Remaining |
|---|---:|---:|---:|---:|
| traffic_volume | 9086 | 4 | 9086 | 0 |
| vehicle_count | 4499 | 2 | 4499 | 0 |
| avg_speed | 9078 | 4 | 9078 | 0 |
| occupancy | 9072 | 4 | 9072 | 0 |
| travel_time | 4499 | 2 | 4499 | 0 |
| signal_timing | 4499 | 2 | 4499 | 0 |
| vehicle_type_dist | 4499 | 2 | 4499 | 0 |

Recovery uses same-row semantic equivalence where configured, then the previous-day same window, bounded same-road causal forward fill, and only the configured concurrent station median for unresolved leading values. Donor rows and timestamps are stored beside repairs.

## Vehicle distribution and congestion

| Check | Result |
|---|---:|
| Vehicle-share rows normalized | 60347 |
| Congestion labels derived | 31123 |
| Existing-label disagreements | 0 |
| Unobserved accident windows retained as unknown | 4499 |

Inserted windows retain a false `_accident_observed` flag and an unknown accident count; they are not silently relabelled as no incident.

This file is generated from `traffic_summary.json`; edit the pipeline, not this report.
