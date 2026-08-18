from pathlib import Path

import pytest
from menin_discovery.settings import load_settings, resolve_project_path


def test_default_settings_load_and_resolve_paths():
    settings = load_settings()
    assert settings["modeling"]["primary_split"] == "scaffold"
    assert resolve_project_path(settings["paths"]["processed"]).name == "processed"


def test_settings_override_is_recursive(tmp_path: Path):
    override = tmp_path / "override.yaml"
    override.write_text("modeling:\n  test_size: 0.25\n", encoding="utf-8")
    settings = load_settings(override)
    assert settings["modeling"]["test_size"] == 0.25
    assert settings["modeling"]["primary_split"] == "scaffold"


@pytest.mark.parametrize(
    "override_text, message",
    [
        (
            "herg:\n  blocker_max_nm: 30000\n  nonblocker_min_nm: 10000\n",
            "blocker_max_nm",
        ),
        ("modeling:\n  applicability_domain_quantile: 1.2\n", "quantile"),
        ("modeling:\n  fingerprint_radius: -1\n", "fingerprint_radius"),
        ("modeling:\n  evaluation_splits: [scaffold, impossible]\n", "Unsupported"),
    ],
)
def test_semantically_invalid_settings_fail_early(tmp_path: Path, override_text: str, message: str):
    override = tmp_path / "invalid.yaml"
    override.write_text(override_text, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_settings(override)


def test_resolve_project_path_accepts_explicit_installed_project_root(tmp_path: Path):
    assert resolve_project_path("reports", root=tmp_path) == tmp_path / "reports"
