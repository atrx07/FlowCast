# AGENTS.md

## 1. Purpose

This file is the mandatory operating contract for every coding agent working on FlowCast.
The agent must treat the repository as a long-running, multi-stage engineering build, not as a one-shot prompt.
The goal is to deliver a reproducible Streamlit web application backed by a modular data, machine-learning, deep-learning, evaluation, and reporting pipeline.

## 2. Instruction Priority

When instructions conflict, use this order:

1. The user's latest explicit instruction.
2. `AGENTS.md`.
3. The original project material in `FlowCast-project_file/`:
   - `FlowCast_PRD.docx`
   - `FlowCast_Data_Dictionary.docx`
   - the three source CSV files
4. `TECH_STACK.md` for technologies, runtime versions, dependencies, artifact
   formats, deployment assumptions, and approved substitutions.
5. `PROJECT.md` and `ROADMAP.md`.
6. `ARCHITECTURE.md`.
7. `STEPS.md`.
8. `STATUS.md` and `NEXT_STEP.md`.
9. Existing implementation details and comments.

The original PRD and data dictionary define the formal acceptance requirements. Repository documents may clarify implementation, but must not silently weaken the original requirements.

## 3. Mandatory Reading Protocol

At the beginning of every work session or turn, the agent must:

1. Read `AGENTS.md` completely.
2. Read `TECH_STACK.md` completely and obey its runtime, dependency, artifact,
   deployment, and substitution rules.
3. Read `STATUS.md` to identify the verified current state.
4. Read `NEXT_STEP.md` to identify the immediate task and exit criteria.
5. Read the relevant section of `STEPS.md` before implementing anything.
6. Read `PROJECT.md`, `ROADMAP.md`, and `ARCHITECTURE.md` whenever the task can affect scope, interfaces, folder layout, model behaviour, data contracts, or acceptance criteria.
7. Inspect the current Git diff and relevant tests before changing existing code.
8. Consult `FlowCast-project_file/` whenever the original requirement, field definition, join rule, target, metric, or expected deliverable is uncertain.

Do not begin from chat memory alone. The Markdown files and repository state are the working memory.

## 4. Mandatory End-of-Turn Protocol

After every implementation turn, even if the work is partial, the agent must:

1. Run the smallest relevant validation, test, lint, or smoke command.
2. Record what was actually changed in `STATUS.md`.
3. Record test evidence, generated artifacts, metrics, and unresolved failures in `STATUS.md`.
4. Rewrite `NEXT_STEP.md` so it contains the single best next action based on the new status.
5. Update `ROADMAP.md` if a milestone, scope item, risk, or delivery estimate changed.
6. Update `ARCHITECTURE.md` if a module, interface, folder, data contract, dependency, or runtime flow changed.
7. Update `PROJECT.md` only when the product contract, assumptions, or blueprint changed.
8. Update `STEPS.md` when the proven execution procedure differs from the documented procedure.
9. Never mark a step complete without evidence.

`STATUS.md` and `NEXT_STEP.md` are dynamic files and must be updated after every meaningful turn or repository change.

## 5. Documentation Roles

| File | Role | Update rule |
|---|---|---|
| `AGENTS.md` | Agent operating contract | Change only with explicit user approval or a clearly necessary governance fix |
| `TECH_STACK.md` | Technology, runtime, dependency, artifact, deployment, and substitution contract | Update whenever an approved stack decision changes |
| `PROJECT.md` | Stable product description and blueprint | Update when product assumptions or acceptance contract change |
| `ROADMAP.md` | Scope, milestones, priorities, risks, and release plan | Update when sequencing, scope, or milestone state changes |
| `ARCHITECTURE.md` | Technical structure, boundaries, interfaces, and data flow | Update whenever implementation architecture changes |
| `STEPS.md` | Detailed execution and verification procedure | Update when a better proven procedure is adopted |
| `STATUS.md` | Verified current state and evidence | Update after every turn |
| `NEXT_STEP.md` | Immediate next action and exit gate | Update after every turn |

## 6. Source-of-Truth Rules

- `data/raw/` must contain immutable copies of the delivered CSV files.
- Never edit raw source files in place.
- `FlowCast-project_file/` is a read-only reference directory. Use it whenever original project context is needed.
- Generated datasets belong in `data/interim/`, `data/processed/`, or `artifacts/`, never in `data/raw/`.
- A model shown in the dashboard must be traceable to its model card, feature schema, training window, metric record, and source data version.
- Every displayed dashboard value must originate from real data, a real aggregate, or a persisted model output. No fake analytics or placeholder metrics are allowed.

## 7. Scope Guardrails

FlowCast v1.0 is:

- A single-corridor, batch and near-real-time analytics product.
- A single Streamlit web application as the user-facing surface.
- A modular Python backend called directly by the Streamlit application and CLI.
- A complete pipeline for ingestion, validation, cleaning, feature engineering, EDA, classical ML, recurrent deep learning, evaluation, inference, confidence estimation, and reporting.

FlowCast v1.0 is not:

- A native Android, iOS, or desktop application.
- A React application or a separate FastAPI service unless the user explicitly expands scope.
- A live traffic-signal controller.
- A streaming platform, multi-city platform, graph neural network, transformer system, or automated production MLOps platform.
- A place to add features merely because they are fashionable.

Prefer the simplest architecture that fully satisfies the PRD.

## 8. File-Size and Modularity Rule

- No source-code file should exceed 500 lines unless the user explicitly authorises it or the split is genuinely impractical.
- Begin considering a split before a file reaches 400 lines.
- Split by responsibility, not by arbitrary line count.
- Streamlit pages, model families, validation rules, and reporting code should be separate modules.
- Generated files, datasets, lockfiles, notebooks, and third-party artifacts are exempt from the 500-line source-code limit.
- If an exception is unavoidable, record the reason and intended refactor in `STATUS.md`.

## 9. Engineering Standards

### 9.1 General

- Use Python 3.11.
- Follow PEP 8 and use clear type hints on public functions.
- Add concise docstrings to public modules, classes, and non-obvious functions.
- Use `pathlib.Path`, not hard-coded OS-specific path strings.
- Keep configuration in versioned YAML/TOML files and environment variables where appropriate.
- Do not scatter magic thresholds, seeds, column names, or paths across modules.
- Seed Python, NumPy, scikit-learn, XGBoost, and the deep-learning framework.
- Prefer pure, testable transformations over stateful notebook-only logic.
- Do not hide core pipeline logic inside notebooks.

### 9.2 Changes

- Make small, reviewable changes.
- Preserve existing public interfaces unless the change is intentional and documented.
- Avoid unrelated refactors during a focused step.
- Do not delete working functionality to make a test pass.
- Do not suppress warnings or exceptions without understanding and documenting them.
- Do not silently skip bad records.

### 9.3 Dependencies

- Read and obey `TECH_STACK.md` before installing, upgrading, substituting, or
  removing any runtime or dependency.
- Use the standard open-source stack named in the PRD.
- Prefer one deep-learning framework; the architecture chooses PyTorch unless a documented compatibility problem requires a change.
- Pin direct dependencies after verifying installation compatibility.
- Do not introduce an unlisted dependency or artifact format without updating
  `TECH_STACK.md` and recording the decision in `STATUS.md`.
- Do not add a database, API server, orchestration platform, cloud service, or heavy UI framework without a demonstrated need.

### 9.4 Version Control

- Use `https://github.com/atrx07/FlowCast.git` as the canonical `origin` remote.
- Work directly on `main` unless the user explicitly requests another branch or
  workflow.
- Keep commits atomic: each commit must contain one cohesive, reviewable unit of
  work and its required tests/documentation.
- Before every commit and push, run the smallest relevant tests plus appropriate
  assurance checks. Do not commit or push a failing state.
- Before every push, update the project status and timeline in `README.md` so
  they match the verified state in `STATUS.md` and the immediate gate in
  `NEXT_STEP.md`; never publish a push with stale README progress text.
- Push each verified atomic commit to `origin` after it is created.
- Never force-push or rewrite published history unless the user explicitly asks.

## 10. Data Rules

- Parse traffic dates as `YYYY-MM-DD` and weather dates as `DD/MM/YYYY` before constructing a common timestamp.
- Use `road_id + timestamp` as the traffic uniqueness key.
- Use `station_id + aligned hour` for weather uniqueness.
- Join weather through `weather_station_id -> station_id` and aligned hour.
- Join calendar on date.
- Preserve quarantined rows and reasons; never silently discard invalid data.
- Resolve exact/key duplicates before merging.
- Treat negative traffic volume, speed above 200 km/h, and occupancy above 100% as physically invalid.
- Parse `vehicle_type_dist` from JSON and validate expected keys and near-unit sum.
- Derive missing congestion labels using the dictionary rule:
  - `V/C < 0.50`: Free-flow
  - `0.50 <= V/C < 0.80`: Moderate
  - `0.80 <= V/C < 1.00`: Heavy
  - `V/C >= 1.00`: Severe
- Use `road_capacity / 2` for the 30-minute capacity denominator.
- Compute lags and rolling features within each road segment in strict timestamp order.
- Shift before rolling so the current or future target cannot leak into a feature.
- Fit imputers, encoders, and scalers on the training period only when they learn statistics.

## 11. Forecasting and Target Rules

FlowCast must forecast 1-4 future 30-minute windows, equivalent to 30, 60, 90, and 120 minutes.

Required outputs per road segment and horizon:

- Traffic volume.
- Average speed.
- Congestion class.
- Travel time.
- Accident-risk probability.
- Confidence or uncertainty.

Use time-aware target construction. A row at time `t` may use only information available at or before `t`, while targets are shifted to `t+h`.

Recommended v1 strategy:

- Classical models: direct horizon-specific models or a clearly documented multi-output equivalent.
- Deep model: a recurrent sequence model with a multi-horizon output for volume; an optional congestion head may be added only after the primary volume model is stable.
- Accident risk: binary target `accident_count > 0`, with class weighting and validation-based threshold selection.
- Confidence: split-conformal residual intervals for regression where practical; calibrated probabilities and entropy/max-probability for classification.

Never report random-split metrics for the final system.

## 12. Model and Evaluation Rules

Required classical coverage:

- NumPy linear regression with an explicit gradient-descent loop.
- scikit-learn Linear Regression.
- Decision Tree.
- Random Forest.
- XGBoost.
- SVM baseline for classification.

Required deep-learning coverage:

- LSTM or GRU built and trained from scratch with no pretrained weights.
- Time-ordered train/validation/test sequences.
- Dropout, early stopping, and learning-rate control.
- Training and validation curves.
- Head-to-head comparison on the exact same hold-out period as classical models.

Primary metrics:

- Volume and travel-time regression: RMSE; support with MAE, MAPE, and R².
- Congestion classification: Macro-F1; support with precision, recall, accuracy, and confusion matrix.
- Accident risk: ROC-AUC; support with precision, recall, F1, PR-AUC, and threshold analysis.

Project targets:

- Volume MAPE <= 12%.
- Congestion Macro-F1 >= 0.80.
- Accident-risk ROC-AUC >= 0.75.
- Deep sequence model should beat the best classical volume model on test RMSE.

Targets are acceptance goals, not permission to manipulate splits or hide unfavourable results. Report honest outcomes.

## 13. Dashboard Rules

- The user-facing implementation is Streamlit.
- The dashboard normally reads persisted datasets, metrics, predictions, and models; it must not retrain on every rerun.
- A manual retraining control may invoke an explicit training service only after user confirmation.
- Keep every required view reachable within three clicks.
- Required views:
  1. Live/near-term predictions.
  2. Historical trends.
  3. Congestion heatmap.
  4. Road comparison.
  5. Model performance.
  6. Feature importance.
  7. Forecast visualisation.
  8. Prediction confidence.
  9. Weather versus traffic.
- Also provide upload/validation, explicit retraining, prediction controls, and report export.
- Use consistent congestion severity mapping across all pages.
- Validate uploads before they enter the pipeline.

## 14. Testing and Evidence

Every completed step needs evidence appropriate to its layer:

- Schema tests for ingestion.
- Unit tests for cleaning and feature functions.
- Data-contract tests for outputs.
- Leakage tests for lag and target construction.
- Reproducibility tests for seeded model training where deterministic behaviour is feasible.
- Metric and split assertions for evaluation.
- Smoke tests for model loading and inference.
- Streamlit import/page smoke tests.
- End-to-end clean-environment reproduction before final delivery.

Record commands and outcomes in `STATUS.md`. A statement such as “works” without a command, artifact, metric, or manual verification is not evidence.

## 15. Artifact and Naming Rules

- Use timestamped or versioned artifact directories, not overwritten anonymous files.
- Store metadata beside each model.
- Prefer Parquet for processed tabular data and CSV/JSON for human-readable summaries.
- Store metrics in machine-readable JSON/CSV and render them into Markdown/figures for reports.
- Each model card must contain target, horizon, training/validation/test ranges, features, preprocessing version, hyperparameters, seed, metrics, limitations, and artifact paths.
- Keep large generated artifacts out of Git unless the submission explicitly requires them.

## 16. Decision Discipline

When an ambiguity appears:

1. Check the PRD.
2. Check the data dictionary.
3. Inspect the actual data.
4. Check these Markdown files.
5. Choose the smallest defensible implementation.
6. Record the decision in `STATUS.md` and update architecture/roadmap if persistent.

Do not invent requirements. Do not silently drop required outputs. Ask the user only when the decision changes scope, cost, deadline, or acceptance behaviour and cannot be safely resolved from the repository.

## 17. Completion Standard

FlowCast is complete only when a reviewer can:

1. Clone/open the repository on Windows, macOS, or Linux.
2. Install the pinned environment with one documented setup command.
3. Run the full pipeline from immutable raw inputs with one documented command.
4. Reproduce the reported metrics on the defined time hold-out.
5. Load persisted models without retraining.
6. Open the Streamlit dashboard.
7. Use all nine required views with real outputs.
8. Upload valid data, obtain predictions, and export a report.
9. Trace every displayed prediction to a model version and data version.
