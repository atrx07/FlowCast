# TECH_STACK.md

## 1. Authority and Purpose

This file is the single source of truth for FlowCast technologies, runtime versions,
direct dependencies, artifact formats, deployment assumptions, and approved
substitutions. It implements the technology choices in `AGENTS.md`, `PROJECT.md`,
`ARCHITECTURE.md`, `FlowCast_PRD.docx`, and `FlowCast_Data_Dictionary.docx` without
changing their product or acceptance requirements.

No dependency, framework, persistence layer, service, or artifact format may be
introduced outside this contract without documenting the need here and updating
`STATUS.md`. Scope-expanding substitutions require explicit user approval.

## 2. Runtime and Environment

| Concern | Approved choice | Version / policy |
|---|---|---|
| Language | CPython | 3.11.9 for the current Windows build; project compatibility is Python `>=3.11,<3.12` |
| Environment | Standard-library `venv` + `pip` | Repository-local `.venv`; never committed |
| Packaging | `pyproject.toml` with setuptools | PEP 517/518, `src/` package layout |
| Dependency record | `pyproject.toml` + `requirements.txt` | Direct dependencies pinned exactly after compatibility verification |
| Configuration | YAML plus environment variables | YAML is versioned; secrets are never committed |
| Timezone | IANA `Asia/Kolkata` | Raw strings preserved; canonical timestamps use the configured timezone policy |
| Global seed | Integer `42` | Used across Python, NumPy, scikit-learn, XGBoost, and PyTorch |
| Version control | Git | Current workstation Git is acceptable; generated/large artifacts are ignored |

Python 3.11.9 is used because Python 3.11 is mandated and 3.11.9 is the final
3.11 release with official Windows binary installers. Later 3.11 security source
releases are acceptable only when a reproducible binary/runtime is available on
all supported platforms.

## 3. Approved Direct Dependencies

Dependencies are grouped so early data work remains lean. Deferred groups are
installed only when their milestone begins, but their technology choices are fixed
here.

### 3.1 Bootstrap, data, and validation

| Package | Pin | Purpose |
|---|---:|---|
| NumPy | `2.3.3` | Numerical operations and later from-scratch regression |
| Pandas | `3.0.3` | Ingestion, auditing, cleaning, merging, and features |
| PyArrow | `25.0.0` | Parquet read/write engine |
| PyYAML | `6.0.3` | Versioned YAML configuration |
| tzdata | `2026.2` | Cross-platform IANA timezone data, including Windows |
| pytest | `9.0.2` | Automated unit, contract, integration, and smoke tests |

### 3.2 Classical modelling

| Package | Pin | Purpose |
|---|---:|---|
| scikit-learn | `1.9.0` | Linear models, trees, forests, SVM, preprocessing, calibration, metrics |
| XGBoost | `3.2.0` | Gradient-boosted regression and classification compatible with Python 3.11 |
| joblib | `1.5.2` | Persist trusted scikit-learn pipelines and estimators |
| tqdm | `4.67.1` | CLI/notebook progress for bounded training jobs |

### 3.3 Deep learning

| Package | Pin | Purpose |
|---|---:|---|
| PyTorch (`torch`) | `2.13.0` | CPU-first LSTM/GRU training and native checkpoints |

CUDA is optional. CPU execution is the acceptance baseline; GPU wheels or CUDA
toolkits must not become mandatory for reproduction.

### 3.4 Visualisation, dashboard, and notebooks

| Package | Pin | Purpose |
|---|---:|---|
| Plotly | `6.9.0` | Interactive Streamlit charts |
| Matplotlib | `3.11.1` | Static figures, training curves, and reports |
| Seaborn | `0.13.2` | Optional EDA/report statistical plots only |
| Streamlit | `1.59.2` | The single user-facing web application |
| JupyterLab | `4.6.1` | EDA and experiment narrative; never the only pipeline implementation |

Transitive packages are resolved by `pip` from these exact direct pins. A complete
environment snapshot must be recorded after each verified milestone and before final
delivery. Pre-releases and yanked releases are prohibited.

## 4. Standard Library Choices

- `argparse` for the CLI; no additional CLI framework is required.
- `pathlib` for paths and filesystem boundaries.
- `logging` for structured console/file logging.
- `hashlib` for SHA-256 source manifests.
- `json` for machine-readable manifests and audit artifacts.
- `dataclasses` and type hints for small public data structures where appropriate.

## 5. Artifact and Serialization Contract

| Artifact | Canonical format | Notes |
|---|---|---|
| Delivered/raw tables | CSV | Byte-identical immutable copies in `data/raw/` |
| Source checksum manifest | JSON | Filename, bytes, SHA-256, source/copy paths, copy timestamp |
| Interim/processed tables | Parquet via PyArrow | Versioned; never written into `data/raw/` |
| Quarantined records | Parquet + JSON summary | Row/cell reason codes and lineage retained |
| Audits and quality reports | JSON canonical + generated Markdown | Markdown must derive from machine-readable results |
| Feature manifests | JSON | Feature name, dtype, source, transform, version, leakage status |
| Predictions | Parquet + JSON manifest | Includes origin, horizon, data/model versions, and confidence |
| Metrics/scoreboards | JSON and/or CSV | Markdown/figures render from machine-readable metrics |
| scikit-learn models | Joblib | Trusted local artifacts only; preprocessing travels with estimator |
| PyTorch models | `.pt` state dictionary + JSON config | Do not persist opaque whole-model objects as the primary artifact |
| Model cards | Markdown + JSON metadata | Target, horizon, split, features, parameters, metrics, limitations |
| Static figures | PNG | SVG may supplement when portability is verified |
| Notebook narrative | `.ipynb` | Calls package functions; contains no unique production logic |
| Logs | UTF-8 text/JSON-lines style records | Run ID, config, versions, counts, runtime, and failures |
| Reports | Markdown and HTML minimum | PDF styling is optional and must not block v1 acceptance |

## 6. Deployment and Operating Assumptions

- One local Streamlit process calls the `flowcast` Python package directly.
- The CLI is the canonical automation and end-to-end reproduction surface.
- The default target is a workstation with 16 GB RAM and roughly 10 GB free space.
- Windows, macOS, and Linux are supported through Python 3.11 and pinned packages.
- Dashboard pages read persisted artifacts and never retrain on an ordinary rerun.
- Retraining is explicit, confirmed, and may be synchronous for v1.
- Near-real-time means batch refresh/latest persisted or uploaded data; there is no
  streaming ingestion in v1.
- No database, FastAPI service, React frontend, container platform, cloud service,
  paid service, or production orchestrator is part of v1.

## 7. Approved Substitutions

| Preferred choice | Allowed substitution | Conditions |
|---|---|---|
| LSTM | GRU | Same from-scratch, multi-horizon, time-safe evaluation contract |
| PyTorch | TensorFlow/Keras | Only for a documented PyTorch compatibility blocker and an approved architecture update |
| `venv` + `pip` | Conda environment | Only when it preserves Python 3.11, exact direct pins, and one-command setup |
| PyArrow Parquet | `fastparquet` | Only for a reproduced PyArrow blocker and documented cross-platform validation |
| Plotly dashboard charts | Matplotlib static chart | Only when interactivity is not required by the view |
| Joblib | Python pickle | Trusted local scikit-learn artifacts only; document why Joblib failed |
| LSTM/GRU CPU | CUDA acceleration | Optional optimisation; CPU compatibility remains mandatory |

Bidirectional LSTM and TCN comparisons are optional roadmap items and may begin only
after required recurrent-model acceptance gates are met. Transformers, graph neural
networks, databases, API services, and separate frontend frameworks are not approved
substitutions for v1.

## 8. Change Procedure

Before changing this stack:

1. Check the PRD and data dictionary.
2. Demonstrate the compatibility, correctness, or acceptance need.
3. Update this file and any affected architecture/roadmap documents.
4. Record the decision and verification evidence in `STATUS.md`.
5. Re-run the smallest relevant install, import, test, and artifact round-trip checks.
