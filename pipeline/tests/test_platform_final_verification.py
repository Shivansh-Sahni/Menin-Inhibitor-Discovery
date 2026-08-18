from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest
import yaml
from menin_discovery import platform_final_verification as final
from menin_discovery.platform_final_verification import (
    FinalVerificationPaths,
    compare_exact_artifact_trees,
    run_final_artifact_verification,
)


def _write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(root: Path) -> FinalVerificationPaths:
    paths = FinalVerificationPaths()
    directories = (
        paths.external_raw,
        paths.external_normalized,
        paths.canonical_primary,
        paths.reports_primary,
        paths.canonical_secondary,
        paths.reports_secondary,
        paths.statistical_primary,
        paths.statistical_secondary,
        paths.split_primary,
        paths.split_secondary,
        paths.corpus_readiness,
    )
    for relative in directories:
        (root / relative).mkdir(parents=True, exist_ok=True)
    determinism = {
        "schema_version": "platform_canonical_determinism_verification_v1",
        "status": "passed_content_equivalent",
        "content_equivalent": True,
        "canonical_component_count": 1,
        "canonical_component_bytes": 1,
        "qc_generated_artifact_count": 1,
        "build_manifest_a_sha256": "a",
        "build_manifest_b_sha256": "b",
        "normalized_build_manifest_sha256": "c",
        "qc_report_a_sha256": "d",
        "qc_report_b_sha256": "e",
        "normalized_qc_report_sha256": "f",
        "ignored_nondeterministic_fields": {
            "build_manifest": ["built_at_utc"],
            "qc_report": ["generated_at_utc", "build_manifest_sha256"],
        },
        "large_model_training_started": False,
        "substantive_training_started": False,
    }
    _write(root / paths.canonical_determinism_report, json.dumps(determinism))
    for relative in (
        f"{paths.external_normalized}/external_public_normalized_manifest.json",
        f"{paths.canonical_primary}/build_manifest.json",
        f"{paths.reports_primary}/qc_report.json",
        f"{paths.canonical_secondary}/build_manifest.json",
        f"{paths.reports_secondary}/qc_report.json",
        f"{paths.statistical_primary}/analysis_manifest.json",
        f"{paths.statistical_secondary}/analysis_manifest.json",
        f"{paths.split_primary}/acceptance.json",
        f"{paths.split_secondary}/acceptance.json",
        f"{paths.corpus_readiness}/acceptance.json",
    ):
        _write(
            root / relative,
            json.dumps(
                {
                    "large_model_training_started": False,
                    "substantive_training_started": False,
                },
                sort_keys=True,
            ),
        )
    external_inputs = []
    for source_id, manifest_path in sorted(final._EXTERNAL_SOURCE_BINDINGS.items()):
        raw_manifest = root / paths.external_raw / manifest_path
        _write(raw_manifest, json.dumps({"source_id": source_id}, sort_keys=True))
        external_inputs.append(
            {
                "source_id": source_id,
                "manifest_path": manifest_path,
                "physical_manifest_sha256": _sha(raw_manifest),
            }
        )
    external_manifest = root / paths.external_normalized / "external_public_normalized_manifest.json"
    _write(
        external_manifest,
        json.dumps(
            {
                "inputs": external_inputs,
                "large_model_training_started": False,
                "substantive_training_started": False,
            },
            sort_keys=True,
        ),
    )
    final.materialize_static_readiness_registries(
        feature_directory=root / "research/data/platform/features/static",
        model_directory=root / "research/models/platform",
        evidence_checked_date="2026-08-04",
    )
    config = {
        "schema_version": "protein-molecule-platform-config-1.0",
        "project": {"substantive_large_model_training_authorized": False},
        "pretraining_interface": {"substantive_training_authorized": False},
        "tasks": {"clinical": {"enabled_for_training": False}},
        "release": {
            "public_only": True,
            "allowed_access_classes": ["public_redistributable"],
        },
    }
    _write(root / paths.platform_config, yaml.safe_dump(config, sort_keys=True))
    _write(root / "pipeline/environments/requirements.lock", "fixture==1\n")
    _write(root / paths.dependency_audit, "{}\n")
    return paths


def _valid_external_result() -> dict[str, object]:
    return {
        "status": "passed",
        "schema_version": "platform-external-normalization/1.0",
        "output_root": "test",
        "manifest_declared_sha256": "a",
        "manifest_physical_sha256": "b",
        "manifest_physical_bytes": 1,
        "inventory_entries": 9,
        "parquet_artifacts": 1,
        "aggregate_artifact_bytes": 1,
        "aggregate_parquet_rows": 1,
        "input_verification": "passed_full_recursive_bundle_verification",
        "verified_input_count": 5,
        "semantic_verification": {"all_admission_prohibitions_recomputed": True},
        "zero_label_training_and_identity_replacement_contract": "passed",
    }


def _patch_child_verifiers(
    root: Path, paths: FinalVerificationPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    canonical_result = json.loads((root / paths.canonical_determinism_report).read_text(encoding="utf-8"))
    monkeypatch.setattr(final, "compare_canonical_builds", lambda *args: canonical_result)
    monkeypatch.setattr(final, "PLATFORM_CONFIG_SHA256", _sha(root / paths.platform_config))
    monkeypatch.setattr(
        final,
        "_verify_dependency_audit",
        lambda *args: {
            "status": "verified_no_known_vulnerabilities",
            "dependency_count": 53,
            "vulnerability_count": 0,
            "point_in_time_only": True,
        },
    )
    monkeypatch.setattr(
        final,
        "_EXTERNAL_MANIFEST_SHA256",
        {
            source_id: _sha(root / paths.external_raw / manifest_path)
            for source_id, manifest_path in final._EXTERNAL_SOURCE_BINDINGS.items()
        },
    )
    monkeypatch.setattr(final, "verify_external_normalized_output", lambda *args: _valid_external_result())
    monkeypatch.setattr(
        final,
        "verify_statistical_analysis",
        lambda *args, **kwargs: {
            "analysis_version": "platform-statistical-analysis-v1",
            "status": "verified",
            "analysis_manifest_sha256": "a",
            "artifact_count": 1,
            "canonical_component_count": 1,
            "source_reverified": True,
            "zero_training": True,
            "training_actions": [],
            "scientific_boundaries_verified": True,
        },
    )
    monkeypatch.setattr(
        final,
        "verify_split_suite",
        lambda *args, **kwargs: {
            "schema_version": "platform_split_suite_v1",
            "status": "verified",
            "acceptance_file_sha256": "a",
            "component_count": 1,
            "accounting": {},
            "source_reverified": True,
            "label_values_read": False,
            "test_labels_disclosed": False,
            "large_model_training_started": False,
            "substantive_training_started": False,
        },
    )
    monkeypatch.setattr(
        final,
        "verify_corpus_readiness_bundle",
        lambda *args, **kwargs: {
            "schema_version": "platform_corpus_readiness_bundle_v1",
            "status": "verified",
            "acceptance_file_sha256": "a",
            "component_count": 1,
            "task_counts": {},
            "source_reverified": True,
            "test_lockboxes_opened_or_hashed": False,
            "large_model_training_started": False,
            "substantive_training_started": False,
        },
    )


def test_exact_artifact_tree_comparison_is_byte_and_membership_exact(
    tmp_path: Path,
) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    _write(left / "nested/value.txt", "same")
    _write(right / "nested/value.txt", "same")
    assert compare_exact_artifact_trees(left, right)["status"] == "passed_byte_identical"
    _write(right / "nested/value.txt", "different")
    with pytest.raises(ValueError, match="not byte-identical"):
        compare_exact_artifact_trees(left, right)


def test_exact_artifact_tree_comparison_rejects_shared_hardlink(
    tmp_path: Path,
) -> None:
    left = tmp_path / "left/value.txt"
    right = tmp_path / "right/value.txt"
    _write(left, "same")
    right.parent.mkdir(parents=True)
    os.link(left, right)
    with pytest.raises(ValueError, match="hardlinked|share physical file inodes"):
        compare_exact_artifact_trees(left.parent, right.parent)


def test_final_verifier_replays_all_gates_and_binds_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path)
    _patch_child_verifiers(tmp_path, paths, monkeypatch)

    result = run_final_artifact_verification(tmp_path, paths)
    assert result["mechanical_artifact_verification"] == "passed"
    assert result["readiness_boundary"]["substantive_large_model_training_ready"] is False
    assert result["substantive_training_started"] is False
    assert result["training_actions"] == []
    report = tmp_path / paths.output_report
    assert result["report_physical_sha256"] == _sha(report)
    assert len(result["critical_artifacts"]) == 20


def test_final_verifier_rejects_true_training_flag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _fixture(tmp_path)
    _patch_child_verifiers(tmp_path, paths, monkeypatch)
    _write(
        tmp_path / paths.statistical_primary / "unsafe.json",
        json.dumps({"substantive_training_started": True}),
    )
    # Keep duplicate-tree comparison from masking the explicit no-training gate.
    _write(
        tmp_path / paths.statistical_secondary / "unsafe.json",
        json.dumps({"substantive_training_started": True}),
    )
    with pytest.raises(ValueError, match="Training boundary became true|no-training/public-only policy"):
        run_final_artifact_verification(tmp_path, paths)


def test_final_verifier_rejects_static_artifact_escape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path)
    outside = tmp_path.parent / "outside-static-artifact.txt"
    _write(outside, "outside")
    manifest_path = tmp_path / paths.static_manifest
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["model_candidate_registry_json"] = {
        "path": "../../../../outside-static-artifact.txt",
        "sha256": _sha(outside),
    }
    _write(manifest_path, json.dumps(manifest, sort_keys=True))
    _patch_child_verifiers(tmp_path, paths, monkeypatch)
    with pytest.raises((ValueError, FileNotFoundError)):
        run_final_artifact_verification(tmp_path, paths)


def test_final_verifier_rejects_ambiguous_static_parent_traversal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path)
    manifest_path = tmp_path / paths.static_manifest
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact = manifest_path.parent / "model_candidate_registry.json"
    manifest["artifacts"]["model_candidate_registry_json"] = {
        "path": "alias/../model_candidate_registry.json",
        "sha256": _sha(artifact),
    }
    _write(manifest_path, json.dumps(manifest, sort_keys=True))
    _patch_child_verifiers(tmp_path, paths, monkeypatch)
    with pytest.raises(ValueError, match="deterministic regeneration|path binding"):
        run_final_artifact_verification(tmp_path, paths)


def test_final_verifier_rejects_symlinked_output_parent(tmp_path: Path) -> None:
    base = _fixture(tmp_path)
    outside = tmp_path.parent / f"{tmp_path.name}-redirected-output"
    outside.mkdir()
    namespace = tmp_path / "research/reports/final_verification"
    namespace.mkdir(parents=True, exist_ok=True)
    (namespace / "redirect").symlink_to(outside, target_is_directory=True)
    paths = FinalVerificationPaths(
        **{
            **base.__dict__,
            "output_report": "research/reports/final_verification/redirect/final.json",
        }
    )
    with pytest.raises(ValueError, match="output path chain contains a symlink"):
        run_final_artifact_verification(tmp_path, paths)


def test_final_verifier_rejects_output_aliasing_an_input(tmp_path: Path) -> None:
    base = _fixture(tmp_path)
    paths = FinalVerificationPaths(**{**base.__dict__, "output_report": base.platform_config})
    original = (tmp_path / base.platform_config).read_bytes()
    with pytest.raises(ValueError, match="designated final-report namespace"):
        run_final_artifact_verification(tmp_path, paths)
    assert (tmp_path / base.platform_config).read_bytes() == original


def test_final_verifier_rejects_backslash_in_input_path(tmp_path: Path) -> None:
    base = _fixture(tmp_path)
    paths = FinalVerificationPaths(
        **{
            **base.__dict__,
            "external_normalized": "research/data/platform/interim\\external",
        }
    )
    with pytest.raises(ValueError, match="path is not portable"):
        run_final_artifact_verification(tmp_path, paths)


def test_final_verifier_rejects_preexisting_hardlinked_report(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    report = tmp_path / paths.output_report
    _write(report, "{}\n")
    os.link(report, tmp_path / "external-report-alias.json")
    with pytest.raises(ValueError, match="output is hardlinked"):
        run_final_artifact_verification(tmp_path, paths)


def test_final_verifier_refuses_nonidentical_existing_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path)
    _patch_child_verifiers(tmp_path, paths, monkeypatch)
    run_final_artifact_verification(tmp_path, paths)
    report = tmp_path / paths.output_report
    report.write_text('{"stale":true}\n', encoding="utf-8")
    stale = report.read_bytes()
    with pytest.raises(RuntimeError, match="Refusing to overwrite"):
        run_final_artifact_verification(tmp_path, paths)
    assert report.read_bytes() == stale


def test_final_verifier_rejects_generic_training_authorization_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path)
    _patch_child_verifiers(tmp_path, paths, monkeypatch)
    for relative in (paths.statistical_primary, paths.statistical_secondary):
        _write(
            tmp_path / relative / "unsafe.json",
            json.dumps({"training_authorized": True}, sort_keys=True),
        )
    with pytest.raises(ValueError, match="Training boundary became true|no-training/public-only policy"):
        run_final_artifact_verification(tmp_path, paths)


def test_final_verifier_rejects_regenerated_static_registry_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path)
    _patch_child_verifiers(tmp_path, paths, monkeypatch)
    registry = tmp_path / "research/models/platform/model_candidate_registry.json"
    payload = json.loads(registry.read_text(encoding="utf-8"))
    payload["training_authorized"] = True
    _write(registry, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    manifest_path = tmp_path / paths.static_manifest
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["model_candidate_registry_json"]["sha256"] = _sha(registry)
    _write(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="deterministic regeneration"):
        run_final_artifact_verification(tmp_path, paths)


def test_final_verifier_rejects_nested_yaml_training_enablement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path)
    _patch_child_verifiers(tmp_path, paths, monkeypatch)
    config_path = tmp_path / paths.platform_config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["tasks"]["clinical"]["enabled_for_training"] = True
    _write(config_path, yaml.safe_dump(config, sort_keys=True))
    with pytest.raises(ValueError, match="Training boundary became true|no-training/public-only policy"):
        run_final_artifact_verification(tmp_path, paths)


def test_final_verifier_rejects_duplicate_yaml_policy_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path)
    _patch_child_verifiers(tmp_path, paths, monkeypatch)
    config_path = tmp_path / paths.platform_config
    text = config_path.read_text(encoding="utf-8")
    text = text.replace(
        "  substantive_large_model_training_authorized: false\n",
        "  substantive_large_model_training_authorized: false\n"
        "  substantive_large_model_training_authorized: true\n",
    )
    _write(config_path, text)
    with pytest.raises(ValueError, match="Duplicate YAML key"):
        run_final_artifact_verification(tmp_path, paths)


@pytest.mark.parametrize(
    "policy_key",
    ["training_enabled", "enable_training", "training_allowed", "training_permitted"],
)
def test_final_verifier_rejects_positive_training_policy_synonyms(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, policy_key: str
) -> None:
    paths = _fixture(tmp_path)
    _patch_child_verifiers(tmp_path, paths, monkeypatch)
    config_path = tmp_path / paths.platform_config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config[policy_key] = True
    _write(config_path, yaml.safe_dump(config, sort_keys=True))
    with pytest.raises(ValueError, match="Training boundary became true"):
        run_final_artifact_verification(tmp_path, paths)


def test_final_verifier_rejects_failed_child_status(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _fixture(tmp_path)
    _patch_child_verifiers(tmp_path, paths, monkeypatch)
    monkeypatch.setattr(
        final,
        "verify_external_normalized_output",
        lambda *args: {"status": "failed"},
    )
    with pytest.raises(ValueError, match="external normalization verifier contract failed"):
        run_final_artifact_verification(tmp_path, paths)


def test_final_verifier_rejects_nonstandard_external_semantic_topology(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path)
    _patch_child_verifiers(tmp_path, paths, monkeypatch)
    external = _valid_external_result()
    external["semantic_verification"] = "not_applicable_nonstandard_fixture"
    monkeypatch.setattr(final, "verify_external_normalized_output", lambda *args: external)
    with pytest.raises(ValueError, match="standard-topology semantic replay"):
        run_final_artifact_verification(tmp_path, paths)


def test_final_verifier_rejects_external_source_identity_substitution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path)
    _patch_child_verifiers(tmp_path, paths, monkeypatch)
    manifest_path = tmp_path / paths.external_normalized / "external_public_normalized_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["inputs"][0]["source_id"] = "substituted_source"
    _write(manifest_path, json.dumps(manifest, sort_keys=True))
    with pytest.raises(ValueError, match="exact five sources|source-manifest identity"):
        run_final_artifact_verification(tmp_path, paths)


def test_final_verifier_scans_canonical_qc_reports_for_training_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path)
    _patch_child_verifiers(tmp_path, paths, monkeypatch)
    for relative in (paths.reports_primary, paths.reports_secondary):
        qc = tmp_path / relative / "qc_report.json"
        payload = json.loads(qc.read_text(encoding="utf-8"))
        payload["training_authorized"] = True
        _write(qc, json.dumps(payload, sort_keys=True))
    with pytest.raises(ValueError, match="Training boundary became true"):
        run_final_artifact_verification(tmp_path, paths)


def test_final_verifier_rejects_extra_scientific_claim_from_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path)
    _patch_child_verifiers(tmp_path, paths, monkeypatch)
    external = _valid_external_result()
    external["scientific_task_claim_ready"] = True
    monkeypatch.setattr(final, "verify_external_normalized_output", lambda *args: external)
    with pytest.raises(ValueError, match="result schema drifted"):
        run_final_artifact_verification(tmp_path, paths)


def test_final_verifier_rejects_contradictory_child_training_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path)
    _patch_child_verifiers(tmp_path, paths, monkeypatch)
    monkeypatch.setattr(
        final,
        "verify_external_normalized_output",
        lambda *args: {
            "status": "passed",
            "input_verification": "passed_full_recursive_bundle_verification",
            "verified_input_count": 1,
            "zero_label_training_and_identity_replacement_contract": "passed",
            "training_authorized": True,
        },
    )
    with pytest.raises(ValueError, match="Training boundary became true"):
        run_final_artifact_verification(tmp_path, paths)


def test_final_verifier_requires_explicit_clinical_training_prohibition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path)
    _patch_child_verifiers(tmp_path, paths, monkeypatch)
    config_path = tmp_path / paths.platform_config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["tasks"]["clinical"].pop("enabled_for_training")
    _write(config_path, yaml.safe_dump(config, sort_keys=True))
    with pytest.raises(ValueError, match="no-training/public-only policy"):
        run_final_artifact_verification(tmp_path, paths)


def test_final_verifier_requires_exact_static_artifact_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path)
    _patch_child_verifiers(tmp_path, paths, monkeypatch)
    manifest_path = tmp_path / paths.static_manifest
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"].pop("baseline_robustness_matrix_csv")
    _write(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="exact six artifacts"):
        run_final_artifact_verification(tmp_path, paths)


def test_dependency_audit_rejects_substituted_package_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_root = Path(__file__).resolve().parents[2]
    source_audit = project_root / "research/reports/platform/audit/dependency_vulnerability_audit.json"
    source_lock = project_root / "pipeline/environments/requirements.lock"
    target_lock = tmp_path / "pipeline/environments/requirements.lock"
    target_audit = tmp_path / "research/reports/platform/audit/dependency_vulnerability_audit.json"
    _write(target_lock, source_lock.read_text(encoding="utf-8"))
    document = json.loads(source_audit.read_text(encoding="utf-8"))
    document["results"][0]["name"] = "substituted-package"
    _write(target_audit, json.dumps(document, indent=2, sort_keys=True) + "\n")
    monkeypatch.setattr(final, "DEPENDENCY_AUDIT_SHA256", _sha(target_audit))
    with pytest.raises(ValueError, match="do not exactly match"):
        final._verify_dependency_audit(target_audit, tmp_path)
