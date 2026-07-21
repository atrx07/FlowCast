# FlowCast

FlowCast is a reproducible Python 3.11 traffic forecasting and congestion
intelligence project. The repository is being built milestone by milestone from
the immutable source material in `FlowCast-project_file/`.

The current milestone provides the installable `flowcast` package, versioned
configuration, immutable raw-data preservation, SHA-256 verification, and an
automated baseline audit. It does not yet clean data, train models, or serve the
Streamlit dashboard.

## Bootstrap on Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[test]"
```

On macOS or Linux, replace the last two interpreter paths with
`.venv/bin/python`.

## Verify Milestone M1

```powershell
.\.venv\Scripts\python.exe -m flowcast.cli --help
.\.venv\Scripts\python.exe -m flowcast.cli audit
.\.venv\Scripts\python.exe -m pytest -q
```

The audit copies the three delivered CSV files byte-for-byte into `data/raw/`,
verifies source and destination hashes, and writes machine-readable and Markdown
reports beneath `artifacts/audits/raw_v1/`. Re-running it never overwrites a raw
copy whose hash differs from its reference source.

See `TECH_STACK.md` for the authoritative runtime, dependency, artifact, and
deployment contract. See `STATUS.md` and `NEXT_STEP.md` for verified progress and
the single next action.
