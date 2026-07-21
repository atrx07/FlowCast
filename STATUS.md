# STATUS.md

## Status Metadata

- **Project:** FlowCast v1.0
- **Last updated:** 2026-07-21
- **Current milestone:** M1 - Ingestion and validation (in progress)
- **Current step:** Steps 01 and 02 complete; Step 03 not started
- **Overall state:** Bootstrap and immutable raw-data audit acceptance gate passed
- **Primary blocker:** None

## 1. Verified Current Position

FlowCast is now an initialized Git repository with a project-local CPython 3.11.9
runtime, a Python `src/`-layout package, versioned YAML configuration, a real CLI,
immutable raw copies, SHA-256 lineage, an automated source audit, and baseline tests.

No cleaning, quarantine logic, feature engineering, EDA, model training, inference,
reporting pipeline, or Streamlit dashboard has been implemented. This turn stopped
at the approved Steps 01-02 acceptance gate.

The verified baseline is published on `origin/main`. Initial implementation commit
`07f30ea5390eb48a9ea30f7f9a4c0fdb2ea7390e` was confirmed byte-for-byte equal to
the remote branch head after push.

## 2. Completed Work

### Technology and governance

- Added `TECH_STACK.md` as the authoritative runtime, dependency, artifact,
  deployment, and substitution contract.
- Updated `AGENTS.md` so every agent must read and obey `TECH_STACK.md`.
- Selected CPython 3.11.9, standard-library `venv`, `pip`, setuptools packaging,
  PyTorch for the later recurrent model, and Streamlit for the eventual UI.
- Installed only the bootstrap/data/test dependency group; modelling, deep-learning,
  notebook, and dashboard dependencies remain deferred to their milestones.

### Repository bootstrap (Step 01)

- Initialized Git on branch `main` and configured the canonical remote as
  `https://github.com/atrx07/FlowCast.git`.
- Recorded the user-approved direct-`main`, tested atomic commit-and-push workflow
  in `AGENTS.md`.
- Added `pyproject.toml`, `requirements.txt`, `.gitignore`, and `README.md`.
- Added `config/base.yaml` and `config/data_contracts.yaml`.
- Added the installable `flowcast` package, deterministic root/path settings,
  global seed `42`, `Asia/Kolkata` timezone policy, and console/file logging.
- Added an `argparse` CLI with a functional `audit` subcommand.
- Created only the directories needed for the first pipeline stages.

### Raw preservation and audit (Step 02)

- Copied the three delivered CSV files byte-for-byte from the read-only reference
  directory into `data/raw/`.
- Added known byte counts and SHA-256 values to the versioned data contract.
- Added fail-closed source verification and immutable destination behavior: an
  existing raw file with a different hash is never overwritten.
- Generated `data/raw/raw_manifest.json` with source/copy paths, byte counts,
  hashes, and copy timestamps.
- Generated canonical JSON and derived Markdown audit reports.
- Added unit and full-source data-contract tests.
- Data-contract tests write raw copies and audit artifacts only inside pytest
  temporary directories, so validation does not mutate tracked evidence.

## 3. Files Added or Modified

- Governance: `AGENTS.md`, `TECH_STACK.md`, `ROADMAP.md`, `STATUS.md`,
  `NEXT_STEP.md`.
- Bootstrap: `.gitattributes`, `.gitignore`, `README.md`, `pyproject.toml`, and
  `requirements.txt`.
- Configuration: `config/base.yaml`, `config/data_contracts.yaml`.
- Package: `src/flowcast/__init__.py`, `cli.py`, `settings.py`,
  `logging_config.py`, `data/__init__.py`, and `data/audit.py`.
- Tests: `tests/unit/test_package.py`, `tests/unit/test_timestamp_parsers.py`, and
  `tests/data_contracts/test_raw_audit.py`.
- Generated evidence: `data/raw/raw_manifest.json` and
  `artifacts/audits/raw_v1/{audit.json,audit.md,environment.txt}`.

The copied `data/raw/*.csv`, `.runtime/`, `.venv/`, and runtime logs are intentionally
ignored by Git. Their authoritative sources or reproducible manifests remain in the
repository.

## 4. Commands and Validation Evidence

Executed from the repository root with the project-local Python 3.11.9 runtime:

```text
git init
.runtime/python311/python.exe -m venv .venv
.venv/Scripts/python.exe -m pip install --upgrade pip
.venv/Scripts/python.exe -m pip install -e ".[test]"
.venv/Scripts/python.exe -m flowcast.cli --help
.venv/Scripts/python.exe -m flowcast.cli audit
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m pip check
.venv/Scripts/python.exe -m compileall -q src
git remote add origin https://github.com/atrx07/FlowCast.git
git ls-remote --heads origin
git commit -m "feat: bootstrap FlowCast data audit"
git push -u origin main
soffice.com --version
soffice.com --headless --convert-to pdf FlowCast_PRD.docx
```

Results:

- Python: `3.11.9`.
- Editable package installation: succeeded.
- CLI help: succeeded and exposes `audit`.
- CLI audit: succeeded.
- Tests: `9 passed`.
- Dependency integrity: `No broken requirements found.`
- Package byte-compilation: succeeded.
- GitHub remote inspection: succeeded; no existing branch heads were returned.
- Initial Git commit and push: succeeded; local `HEAD` and `origin/main` both
  resolved to `07f30ea5390eb48a9ea30f7f9a4c0fdb2ea7390e` immediately after push.
- LibreOffice: version and real DOCX-to-PDF conversion succeeded.
- GitHub file-size assurance: no candidate file is 100 MB or larger.
- Largest source module: `src/flowcast/data/audit.py`, 392 lines; all source files
  remain below the 500-line limit.

## 5. Raw-Data Lineage Evidence

| Source file | Bytes | SHA-256 |
|---|---:|---|
| `traffic_sensor_log.csv` | 31,231,835 | `8f793f3643c891d4fdda7b66c5c4792d24f4db3f26a07cccb8f1d613e254062a` |
| `weather_observations.csv` | 540,212 | `63f3dc54a491dfd5d4663d8bf0602779084c30a1396f0c7b4fd177e132bc8a31` |
| `calendar_events.csv` | 3,114 | `60d3de6b731486e02f6edaa3515af87c2472231a211b48d94d7a3cad38799b9c` |

Each copied file matches its delivered source and versioned contract. The original
reference files were read and hashed only; they were not rewritten.

## 6. Automated Audit Results

| Check | Verified result |
|---|---:|
| Traffic shape | 178,468 x 17 |
| Weather shape | 10,872 x 7 |
| Calendar shape | 151 x 6 |
| Traffic exact/key duplicates | 1,767 / 1,767 |
| Traffic unique road/timestamp keys | 176,701 |
| Expected traffic grid | 181,200 |
| Missing traffic windows | 4,499 |
| Null traffic volume / speed / occupancy | 4,387 / 4,382 / 4,383 |
| Blank congestion labels | 26,883 |
| Negative traffic-volume rows | 241 |
| Accident-positive rows | 1,669 |
| Weather unique/expected station-hours | 10,872 / 10,872 |
| Weather null temperature / visibility | 167 / 111 |
| Calendar unique dates | 151 |

All expected baseline assertions were reproduced without variance.

## 7. Decisions and Constraints

- The official Python 3.11.9 Windows installer was verified by its published MD5
  before creating the ignored project-local runtime. This avoids changing the
  machine-wide Python installation.
- The raw CSV copies are ignored because they are reproducible from the reference
  directory and include a committed-sized manifest; the raw data itself remains
  mandatory at runtime and immutable.
- `.gitattributes` treats CSV/DOCX reference artifacts as binary and enforces LF for
  project-authored text. Staged CSV blob hashes were verified equal to their working
  source blobs, preventing Windows line-ending conversion from breaking SHA-256.
- Audit and manifest writers explicitly emit LF on every platform; tests assert that
  generated JSON and Markdown contain no Windows CRLF drift.
- The JSON audit is canonical. `audit.md` is generated from the same result and is
  not independently edited.
- `config/data_contracts.yaml` currently records delivered schemas and hashes; it
  does not yet implement Step 03 validation or quarantine behavior.
- LibreOffice is workstation tooling, not a FlowCast runtime dependency. LibreOffice
  26.2.4.2 was verified by converting the real `FlowCast_PRD.docx` headlessly to a
  valid non-empty 438,493-byte PDF in the ignored local runtime area.
- The GitHub remote was confirmed to have no existing branch heads before the
  initial publication; no history needs to be merged or overwritten.

## 8. Risks and Unresolved Work

- The first M1 slice is complete, but M1 remains in progress until executable data
  contracts and reason-preserving quarantine are implemented and tested in Step 03.
- Weather labels contain casing/spelling/whitespace variants; normalization belongs
  to Step 04.
- Traffic duplicates, missing full windows, physically invalid measurements, blank
  congestion labels, and accident imbalance remain deliberately untouched.
- Deferred dependency pins must be installed and compatibility-tested when their
  corresponding modelling/dashboard milestones begin.

## 9. Next Gate

Proceed in the next implementation turn to **Step 03 - Define Data Contracts and
Quarantine**. The exact bounded action and exit gate are maintained in
`NEXT_STEP.md`.
