# FlowCast Source Merge Report

- Merge contract: `source_merge_v1`
- Output version: `merged_sources_v1`
- Input cleaning version: `cleaned_sources_v1`

## Cardinality

| Check | Result |
|---|---:|
| Traffic input rows / keys | 181200 / 181200 |
| Weather input rows / keys | 10872 / 10872 |
| Calendar input rows / keys | 151 / 151 |
| Output rows / keys | 181200 / 181200 |
| Row-count change | 0 |
| Duplicate output keys | 0 |

## Join coverage

| Context | Cardinality | Matched | Missing |
|---|---|---:|---:|
| Weather | many_to_one | 181200 | 0 |
| Calendar | many_to_one | 181200 | 0 |

Weather is aligned by station and floored local clock hour. Calendar is aligned by the normalized local date. Both joins use explicit Pandas `many_to_one` validation and fail closed on an unexpected miss.

This file is generated from `summary.json`; edit the pipeline, not this report.
