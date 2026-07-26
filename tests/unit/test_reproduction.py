"""Unit coverage for isolated Step 19 reproduction boundaries."""

from pathlib import Path

import pytest

from flowcast.cli import build_parser
from flowcast.reproduction_verify import _compare, _semantic
from flowcast.settings import load_settings


def test_run_all_requires_an_explicit_output_root() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["run-all"])


def test_run_all_parser_accepts_approved_reproduction_root() -> None:
    args = build_parser().parse_args(
        ["run-all", "--output-root", "artifacts/reproductions/final_v1"]
    )
    assert args.command == "run-all"
    assert args.output_root == Path("artifacts/reproductions/final_v1")
    assert args.recurrent_device == "cpu"


def test_run_all_parser_accepts_explicit_cuda_diagnostic() -> None:
    args = build_parser().parse_args(
        [
            "run-all",
            "--output-root",
            "artifacts/reproductions/cuda_diagnostic",
            "--recurrent-device",
            "cuda",
        ]
    )
    assert args.recurrent_device == "cuda"


def test_reproduction_verifier_ignores_runtime_and_tiny_float_noise() -> None:
    canonical = {
        "rmse": 60.0,
        "fit_seconds": 2.0,
        "nested": [{"calibration": 0.0028863407506180506}],
    }
    reproduced = {
        "rmse": 60.0,
        "fit_seconds": 3.0,
        "nested": [{"calibration": 0.0028863407506180398}],
    }
    passed, maximum_delta = _compare(
        _semantic(canonical),
        _semantic(reproduced),
    )
    assert passed is True
    assert maximum_delta < 1.0e-12


def test_reproduction_verifier_rejects_material_metric_drift() -> None:
    passed, maximum_delta = _compare({"rmse": 60.0}, {"rmse": 60.1})
    assert passed is False
    assert maximum_delta == pytest.approx(0.1)


def test_output_root_redirects_every_writable_path() -> None:
    settings = load_settings(
        output_root="artifacts/reproductions/unit_contract"
    )
    run_root = settings.root / "artifacts" / "reproductions" / "unit_contract"
    assert settings.raw_dir == run_root / "data" / "raw"
    assert settings.interim_dir == run_root / "data" / "interim"
    assert settings.processed_dir == run_root / "data" / "processed"
    assert settings.quarantine_dir == run_root / "data" / "quarantine"
    assert settings.artifacts_dir == run_root / "artifacts"
    assert settings.logs_dir == run_root / "logs"
    assert settings.reference_dir == settings.root / "FlowCast-project_file"


@pytest.mark.parametrize(
    "output_root",
    [
        ".",
        "artifacts",
        "artifacts/reproductions",
        "data/raw",
        "FlowCast-project_file",
    ],
)
def test_output_root_rejects_canonical_or_broad_targets(output_root) -> None:
    with pytest.raises(ValueError, match="artifacts/reproductions|name a run"):
        load_settings(output_root=output_root)
