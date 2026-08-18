from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
import yaml
from menin_discovery.platform_non_hpc_governance import (
    build_governance_report,
    load_and_validate_config,
    materialize_governance_bundle,
)


def _config() -> dict[str, object]:
    return {
        "schema_version": "platform-non-hpc-readiness-1.0",
        "evidence_date": "2026-08-05",
        "training": {
            "substantive_large_model_training_authorized": False,
            "substantive_large_model_training_started": False,
            "allowed_smoke_maximum_parameters": 100_000,
            "allowed_smoke_maximum_steps": 2,
        },
        "external_admission": {
            "canonical_observations_admitted": 0,
            "model_labels_admitted": 0,
        },
        "release": {"public_release_approved": False},
        "compute": {"hpc_allocation_approved": False},
    }


def _write_config(root: Path, document: dict[str, object] | None = None) -> Path:
    path = root / "pipeline/config/non_hpc_readiness.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(yaml.safe_dump(document or _config(), sort_keys=True), encoding="utf-8")
    return path


def _git(root: Path, *arguments: str) -> None:
    subprocess.run(["git", *arguments], cwd=root, check=True, capture_output=True)


def _repository(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init")
    _git(root, "config", "user.email", "audit@example.invalid")
    _git(root, "config", "user.name", "Audit Fixture")
    (root / "README.md").write_text("fixture\n", encoding="utf-8")
    _write_config(root)
    _git(root, "add", "README.md", "pipeline/config/non_hpc_readiness.yaml")
    _git(root, "commit", "-m", "fixture")
    return root


def test_policy_validation_rejects_training_authorization(tmp_path: Path) -> None:
    document = _config()
    training = document["training"]
    assert isinstance(training, dict)
    training["substantive_large_model_training_authorized"] = True
    path = _write_config(tmp_path, document)
    with pytest.raises(ValueError, match="must remain unauthorized"):
        load_and_validate_config(path)


def test_policy_validation_rejects_external_label_admission(tmp_path: Path) -> None:
    document = _config()
    external = document["external_admission"]
    assert isinstance(external, dict)
    external["model_labels_admitted"] = 1
    path = _write_config(tmp_path, document)
    with pytest.raises(ValueError, match="label admission is not authorized"):
        load_and_validate_config(path)


def test_report_records_dirty_release_and_redacts_secret_value(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    docs = root / "docs"
    docs.mkdir()
    personal_path = "/" + "Users/example/project"
    fake_access_key = "AKIA" + "ABCDEFGHIJKLMNOP"
    (docs / "unsafe.md").write_text(
        f"developer path {personal_path} and {fake_access_key}\n",
        encoding="utf-8",
    )
    report = build_governance_report(root, root / "pipeline/config/non_hpc_readiness.yaml")
    assert report["git_release_inventory"]["status_record_count"] == 1
    scan = report["release_hygiene"]["text_scan"]
    assert scan["personal_path_findings"] == [{"path": "docs/unsafe.md", "line": 1}]
    assert scan["high_confidence_secret_findings"] == [
        {"category": "aws_access_key", "path": "docs/unsafe.md", "line": 1}
    ]
    assert fake_access_key not in json.dumps(report)
    assert report["decision_gates"]["public_release_ready"] is False
    assert report["substantive_training_started"] is False


def test_materialized_bundle_is_bound_and_never_authorizes_training(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    manifest = materialize_governance_bundle(
        root,
        root / "pipeline/config/non_hpc_readiness.yaml",
        root / "research/reports/platform/non_hpc_completion",
    )
    assert manifest["component_count"] == 3
    assert manifest["substantive_large_model_training_ready"] is False
    assert manifest["substantive_large_model_training_authorized"] is False
    assert manifest["training_actions"] == []
    report = json.loads(
        (root / "research/reports/platform/non_hpc_completion/non_hpc_governance_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["release_hygiene"]["repository_license_present"] is False


def test_output_must_remain_inside_project(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    with pytest.raises(ValueError, match="inside project_root"):
        materialize_governance_bundle(
            root,
            root / "pipeline/config/non_hpc_readiness.yaml",
            tmp_path / "outside",
        )
