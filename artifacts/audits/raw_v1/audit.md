# FlowCast Raw-Data Audit: raw_v1

Generated: `2026-07-21T17:04:26.763788+00:00`

## Immutable source copies

| File | Bytes | SHA-256 |
|---|---:|---|
| traffic_sensor_log.csv | 31,231,835 | `8f793f3643c891d4fdda7b66c5c4792d24f4db3f26a07cccb8f1d613e254062a` |
| weather_observations.csv | 540,212 | `63f3dc54a491dfd5d4663d8bf0602779084c30a1396f0c7b4fd177e132bc8a31` |
| calendar_events.csv | 3,114 | `60d3de6b731486e02f6edaa3515af87c2472231a211b48d94d7a3cad38799b9c` |

## Baseline summary

| Evidence | Traffic | Weather | Calendar |
|---|---:|---:|---:|
| Rows | 178,468 | 10,872 | 151 |
| Columns | 17 | 7 | 6 |
| Exact duplicates | 1,767 | 0 | 0 |
| Key duplicates | 1,767 | 0 | 0 |

## Traffic coverage and quality

- Unique road/timestamp keys: **176,701**
- Expected 30-minute grid: **181,200**
- Missing full windows: **4,499**
- Blank congestion labels: **26,883**
- Negative traffic-volume rows: **241**
- Accident-positive rows: **1,669**
- Accident-positive rate: **0.009352**

## Weather coverage and labels

- Unique station/hour keys: **10,872**
- Expected station/hour grid: **10,872**
- Missing station/hour windows: **0**
- Raw labels: `{"CLEAR": 1237, "Clear": 5647, "Cloudy": 1504, "FOG": 7, "Fog": 59, "Overcast": 358, "RAIN": 34, "Rain": 307, "clear": 1284, "cloudy": 340, "foggy": 10, "rain ": 48, "rainy": 37}`

## Calendar flags

- Positive counts: `{"event_flag": 6, "public_holiday": 6, "roadwork_flag": 11}`

The canonical, complete evidence is in `audit.json`; this report is generated from
the same in-memory result and is not maintained independently.
