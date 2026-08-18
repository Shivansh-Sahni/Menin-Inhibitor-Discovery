from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path

import pytest
from menin_discovery import platform_deep_leakage as deep
from menin_discovery.platform_data_sources import sha256_file
from menin_discovery.platform_features import stable_json_digest


def _minimal_report(config: deep.DeepLeakageConfig) -> dict[str, object]:
    payload = asdict(config)
    return {
        "schema_version": deep.SCHEMA_VERSION,
        "configuration": payload,
        "configuration_sha256": stable_json_digest(payload),
        "source_binding": {"analyzer_code_sha256": sha256_file(Path(deep.__file__).resolve())},
        "label_access_contract": {
            "canonical_label_columns_requested": [],
            "training_labels_newly_read": False,
            "validation_labels_read": False,
            "test_labels_read": False,
            "model_ready_test_lockbox_opened_or_hashed": False,
        },
        "accounting": {"tasks_audited": 0, "materialized_strategies_audited": 0},
        "tasks": [],
        "claim_readiness": False,
        "substantive_training_ready": False,
        "substantive_training_authorized": False,
        "large_model_training_started": False,
        "substantive_training_started": False,
    }


def _write_bound_output(root: Path, config: deep.DeepLeakageConfig) -> None:
    root.mkdir()
    deep._atomic_json(root / "report.json", _minimal_report(config))
    (root / "task_decision_matrix.csv").write_text("rank,dataset_key\n", encoding="utf-8")
    (root / "summary.md").write_text("# Test audit\n", encoding="utf-8")
    inventory = deep._inventory(root)
    manifest = {
        "schema_version": deep.MANIFEST_SCHEMA_VERSION,
        "configuration_sha256": stable_json_digest(asdict(config)),
        "report_sha256": sha256_file(root / "report.json"),
        "source_binding": _minimal_report(config)["source_binding"],
        "label_access_contract": {},
        "component_inventory": inventory,
        "component_inventory_sha256": stable_json_digest(inventory),
        "claim_readiness": False,
        "substantive_training_ready": False,
        "substantive_training_authorized": False,
        "substantive_training_started": False,
    }
    deep._atomic_json(root / "manifest.json", manifest)


def _refresh_report_binding(root: Path) -> None:
    manifest = deep._strict_json(root / "manifest.json")
    entry = manifest["component_inventory"]["report.json"]
    entry["sha256"] = sha256_file(root / "report.json")
    entry["size_bytes"] = (root / "report.json").stat().st_size
    manifest["component_inventory_sha256"] = stable_json_digest(manifest["component_inventory"])
    manifest["report_sha256"] = entry["sha256"]
    deep._atomic_json(root / "manifest.json", manifest)


def test_false_exhaustiveness_and_false_sampling_are_rejected() -> None:
    with pytest.raises(ValueError, match="evaluate every declared"):
        deep._validate_evidence_record(
            {
                "evidence_scope": "exhaustive",
                "declared_cross_pairs": 10,
                "evaluated_cross_pairs": 9,
                "pair_budget": 10,
            }
        )
    with pytest.raises(ValueError, match="exceeds its declared pair budget"):
        deep._validate_evidence_record(
            {
                "evidence_scope": "exhaustive",
                "declared_cross_pairs": 11,
                "evaluated_cross_pairs": 11,
                "pair_budget": 10,
            }
        )
    with pytest.raises(ValueError, match="strict subset"):
        deep._validate_evidence_record(
            {
                "evidence_scope": "sampled",
                "declared_cross_pairs": 10,
                "evaluated_cross_pairs": 10,
            }
        )


@pytest.mark.parametrize(
    "column",
    ["label_value", "label_text", "label_upper_bound", "LABEL_KIND"],
)
def test_label_columns_are_rejected_before_parquet_access(column: str) -> None:
    with pytest.raises(ValueError, match="Label columns are forbidden"):
        deep._guard_columns(("record_id", column), source="poison.parquet")
    assert deep._guard_columns(deep.CONTEXT_COLUMNS, source="safe.parquet") == deep.CONTEXT_COLUMNS


def test_threshold_drift_is_detected_even_when_report_is_self_consistent(tmp_path: Path) -> None:
    frozen = deep.DeepLeakageConfig()
    drifted = replace(frozen, fingerprint_tanimoto_threshold=0.70)
    output = tmp_path / "audit"
    _write_bound_output(output, drifted)
    with pytest.raises(ValueError, match="differs from the expected frozen thresholds"):
        deep.verify_deep_leakage_audit(output, expected_config=frozen)
    verified = deep.verify_deep_leakage_audit(output, expected_config=drifted)
    assert verified["test_labels_read"] is False


def test_strategy_threshold_drift_is_detected_inside_bound_report(tmp_path: Path) -> None:
    output = tmp_path / "audit"
    config = deep.DeepLeakageConfig()
    _write_bound_output(output, config)
    report = deep._strict_json(output / "report.json")
    report["tasks"] = [
        {
            "strategy_audits": [
                {
                    "label_columns_read": [],
                    "test_labels_read": False,
                    "chemical_near_similarity": {
                        "evidence_scope": "not_run",
                        "threshold": 0.70,
                        "fingerprint_bits": config.fingerprint_bits,
                        "fingerprint_radius": config.fingerprint_radius,
                        "pair_budget": config.maximum_exhaustive_chemical_cross_pairs,
                    },
                    "protein_sequence_similarity": {},
                }
            ]
        }
    ]
    deep._atomic_json(output / "report.json", report)
    _refresh_report_binding(output)
    with pytest.raises(ValueError, match="Chemical audit thresholds drifted"):
        deep.verify_deep_leakage_audit(output, expected_config=config)


def test_closed_inventory_detects_content_and_topology_tampering(tmp_path: Path) -> None:
    output = tmp_path / "audit"
    _write_bound_output(output, deep.DeepLeakageConfig())
    assert deep.verify_deep_leakage_audit(output)["status"] == "verified"

    (output / "summary.md").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="component changed"):
        deep.verify_deep_leakage_audit(output)

    output = tmp_path / "audit-extra"
    _write_bound_output(output, deep.DeepLeakageConfig())
    (output / "unbound.txt").write_text("unbound\n", encoding="utf-8")
    with pytest.raises(ValueError, match="topology differs"):
        deep.verify_deep_leakage_audit(output)


def test_verifier_rejects_manifest_report_source_binding_drift(tmp_path: Path) -> None:
    output = tmp_path / "audit"
    _write_bound_output(output, deep.DeepLeakageConfig())
    manifest = deep._strict_json(output / "manifest.json")
    manifest["source_binding"] = {"analyzer_code_sha256": "0" * 64}
    deep._atomic_json(output / "manifest.json", manifest)
    with pytest.raises(ValueError, match="source bindings differ"):
        deep.verify_deep_leakage_audit(output)


def test_unsafe_inventory_paths_are_rejected() -> None:
    for value in ("../report.json", "/tmp/report.json", ""):
        with pytest.raises(ValueError, match="Unsafe"):
            deep._safe_relative(value, "inventory path")


def test_protein_kmers_use_normalized_sequence_not_normalizer_metadata() -> None:
    assert deep._kmers(" ACDE ", 3) == frozenset({"ACD", "CDE"})
    assert deep._kmers("AC", 3) == frozenset({"AC"})
    first = deep._kmer_bitset("ACDE", 3)
    second = deep._kmer_bitset("ACDF", 3)
    assert first.bit_count() == 2
    assert second.bit_count() == 2
    assert (first & second).bit_count() / (first | second).bit_count() == 1 / 3


def test_chemical_pair_accounting_and_sampling_are_deterministic() -> None:
    structures = {
        "train": ["CCO", "c1ccccc1"],
        "validation": ["CCN"],
        "test": ["c1ccncc1"],
    }
    config = replace(
        deep.DeepLeakageConfig(),
        fingerprint_bits=128,
        maximum_exhaustive_chemical_cross_pairs=10,
    )
    first = deep._chemical_similarity(structures, config)
    second = deep._chemical_similarity(structures, config)
    assert first == second
    assert first["evidence_scope"] == "exhaustive"
    assert first["declared_cross_pairs"] == 4
    assert first["evaluated_cross_pairs"] == 4
    assert first["threshold"] == 0.8
    assert first["exact_standardized_smiles_overlap"]["any_overlap"] is False

    values = [f"SEQ-{index:04d}" for index in range(1000)]
    sample_a = deep._stable_sample(values, 17, config.seed, "train")
    sample_b = deep._stable_sample(list(reversed(values)), 17, config.seed, "train")
    assert sample_a == sample_b
    assert len(sample_a) == 17


def test_task_priority_is_explicit_and_penalizes_derived_or_imbalanced_tasks() -> None:
    base = {
        "semantics": {
            "label_kind": "continuous_exact",
            "evidence_domain": "herg",
            "assay_family": "herg_functional",
            "endpoint": "IC50",
        },
        "rows": 137,
        "integrated": True,
        "task_scope": "default",
        "materialized_strategies": ["molecule_grouped", "scaffold"],
        "source_count": 1,
        "protein_count": 1,
        "training_class_ratio_max_to_min": 1.0,
    }
    score, reasons, decision = deep._task_priority(base)
    derived = {**base, "task_scope": "derived_sensitivity"}
    imbalanced = {**base, "training_class_ratio_max_to_min": 5.0}
    assert deep._task_priority(derived)[0] < score
    assert deep._task_priority(imbalanced)[0] < score
    assert decision == "priority_pilot_candidate"
    assert "source holdout impossible" in reasons


def test_verifier_rejects_label_access_and_claim_escalation(tmp_path: Path) -> None:
    output = tmp_path / "audit"
    config = deep.DeepLeakageConfig()
    _write_bound_output(output, config)
    report = deep._strict_json(output / "report.json")
    report["label_access_contract"]["test_labels_read"] = True
    report["claim_readiness"] = True
    deep._atomic_json(output / "report.json", report)
    _refresh_report_binding(output)
    with pytest.raises(ValueError, match="label-access boundary"):
        deep.verify_deep_leakage_audit(output)
