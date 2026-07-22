# FlowCast Context Cleaning Report

- Cleaning contract: `context_cleaning_v1`
- Output version: `cleaned_sources_v1`
- Input validation version: `validated_v1`

## Calendar

| Check | Result |
|---|---:|
| Rows / unique dates | 151 / 151 |
| Date range | 2025-01-01 to 2025-05-31 |
| Public holidays | 6 |
| Event days | 6 |
| Roadwork days | 11 |

## Weather

| Check | Result |
|---|---:|
| Rows / unique station-hours | 10872 / 10872 |
| Stations | 3 |
| Temperature imputed | 167 |
| Visibility imputed | 111 |
| Remaining trusted-field nulls | 0 |

### Controlled weather vocabulary

| Label | Rows |
|---|---:|
| Clear | 8168 |
| Cloudy | 1844 |
| Fog | 76 |
| Overcast | 358 |
| Rain | 426 |

### Imputation policy

Temperature and visibility use causal, station-local `station_forward_fill`.
The configured limits are 2 hours for temperature and 2 hours for visibility.
Donor source-row lineage is stored beside every imputed value.
No future or cross-station value is used.

This file is generated from `summary.json`; edit the pipeline, not this report.
