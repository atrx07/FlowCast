"""Baseline package and configuration tests."""

from flowcast import __version__
from flowcast.settings import load_settings, repository_root


def test_package_imports() -> None:
    assert __version__ == "0.1.0"


def test_settings_resolve_from_repository_root() -> None:
    settings = load_settings()
    assert settings.root == repository_root()
    assert settings.config_path == settings.root / "config" / "base.yaml"
    assert settings.data_contracts_path == (
        settings.root / "config" / "data_contracts.yaml"
    )
    assert settings.cleaning_config_path == settings.root / "config" / "cleaning.yaml"
    assert settings.reference_dir == settings.root / "FlowCast-project_file"
    assert settings.raw_dir == settings.root / "data" / "raw"
    assert settings.seed == 42
    assert settings.timezone == "Asia/Kolkata"
    assert settings.validation_version == "validated_v1"
    assert settings.cleaning_version == "cleaned_sources_v1"
    assert settings.merge_version == "merged_sources_v1"
