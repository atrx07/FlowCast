# NEXT_STEP.md

## Immediate Objective

Perform one independent cross-platform portability confirmation of the
completed FlowCast v1.0 release. The implementation roadmap is complete; this
is a release-confidence follow-up, not another modelling or UI step.

## Read Before Acting

1. `AGENTS.md`, `TECH_STACK.md`, `STATUS.md`, and this file.
2. The setup and reproduction commands in `README.md`.
3. The final acceptance findings and limitations in `FINAL_REPORT.md`.
4. Step 19 in `STEPS.md` and the reproduction boundary in `ARCHITECTURE.md`.

## Hardware Notice Before Acting

Use CPU as the portability identity. Expect sustained CPU, RAM, and disk use
for approximately nine minutes based on the measured Windows run, though
another platform may differ. GPU, VRAM, and NPU resources are not required and
must not become hidden dependencies.

## Single Best Next Action

On a clean macOS or Linux Python 3.11 workstation:

1. Install `.[classical,deep,eda,dashboard,test]` in a new virtual
   environment and require `pip check` to pass.
2. Run `flowcast.cli run-all` with a fresh child beneath
   `artifacts/reproductions` and `--recurrent-device cpu`.
3. Run `flowcast.cli verify-reproduction` against that root.
4. Run the complete suite through `python scripts/run_tests.py -q`.
5. Launch Streamlit with `FLOWCAST_OUTPUT_ROOT` set to the reproduced root and
   smoke the live-prediction, model-performance, and data/training pages.
6. Record platform, Python/package versions, stage runtimes, verifier result,
   test exit marker, and any reproducible variance in `STATUS.md`.

## Exit Gate

The portability follow-up passes when:

- all 16 stages complete from unchanged delivered sources;
- the permanent verifier reports `passed: true`;
- frozen metric differences remain within `1e-12`;
- the suite prints `FLOWCAST_PYTEST_EXIT=0`;
- portable model loading and CPU inference pass;
- Streamlit reads the reproduced artifacts without a platform-specific path,
  CUDA, or binary dependency failure.

## Current Blockers

None. FlowCast v1.0, M0-M8, and Steps 00-19 are complete on the verified
Windows workstation. No further implementation is required unless the
independent portability run exposes a reproducible defect or the user expands
scope.
