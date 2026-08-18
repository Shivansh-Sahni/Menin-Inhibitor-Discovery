from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from menin_discovery import platform_herg_quality_tasks as quality_tasks
from menin_discovery.platform_herg_master_dataset import (
    ASSAY_OUTPUT,
    CLINICAL_OUTPUT,
    EXCLUSION_OUTPUT,
    MANIFEST_NAME,
    OBSERVATION_OUTPUT,
    PROTOCOL_OUTPUT,
    STRUCTURE_OUTPUT,
    TASK_OUTPUT,
    HergMasterDatasetError,
    build_herg_master_dataset,
    validate_herg_master_dataset,
)


def _write(path: Path, rows: list[dict[str, object]], schema: pa.Schema | None = None) -> None:
    table = pa.Table.from_pylist(rows, schema=schema) if schema is not None else pa.Table.from_pylist(rows)
    pq.write_table(table, path)


def _task_row(**updates: object) -> dict[str, object]:
    row: dict[str, object] = {
        "task_id": "Q0_WEAK_FIXED_DOSE_BINARY",
        "quality_level": "Q0_large_weak_screen",
        "record_id": "R-Q0",
        "observation_id": "W1",
        "structure_id": "S1",
        "standardized_smiles": "CCN",
        "standard_inchi_key": "QUSNBJAOOMFDIB-UHFFFAOYSA-N",
        "target_scope": "wild_type",
        "source_family": "fixture",
        "source_record_ids_json": '["R1"]',
        "assay_id": "A1",
        "assay_family": "functional",
        "measurement_technology": "automated_patch_clamp",
        "measurement_technology_basis": "fixture",
        "native_endpoint": "IC50",
        "native_relation": ">",
        "native_value": 1000.0,
        "native_unit": "nM",
        "target_relation": "<",
        "target_pic50": 6.0,
        "target_class": 0,
        "source_declared_split": None,
        "model_split": "train",
        "scaffold_group_id": "G1",
        "task_role": "fixture",
        "eligible": True,
        "eligibility_reason": "fixture_eligible",
        "exclusion_reason": None,
        "clinical_context_only": False,
        "direct_herg_label": True,
        "use_as_training_label": True,
        "quality_flags": "",
    }
    row.update(updates)
    return row


def _fixture(tmp_path: Path) -> dict[str, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    roots = {name: tmp_path / name for name in ("hierarchy", "scope", "modality", "tasks", "split")}
    for path in roots.values():
        path.mkdir()
    ledger_rows = [
        {
            "observation_id": "W1",
            "source_family": "fixture",
            "source_record_id": "R1",
            "standardized_smiles": "CCN",
            "standard_inchi_key": "QUSNBJAOOMFDIB-UHFFFAOYSA-N",
            "structure_id": "S1",
            "structure_valid": True,
            "target_variant": "wild_type",
            "assay_id": "A1",
            "assay_family": "functional",
            "native_endpoint": "IC50",
            "native_relation": ">",
            "native_value": 1000.0,
            "native_unit": "nM",
            "pic50_value": None,
            "pic50_origin": None,
            "native_aux_json": json.dumps(
                {
                    "assay_description": "Automated whole-cell patch clamp in HEK293 at -80 mV and 37 C for 5 min on QPatch"
                }
            ),
        },
        {
            "observation_id": "W2",
            "source_family": "fixture_quant",
            "source_record_id": "R2",
            "standardized_smiles": "CCO",
            "standard_inchi_key": "LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
            "structure_id": "S2",
            "structure_valid": True,
            "target_variant": "wild_type_or_unspecified",
            "assay_id": None,
            "assay_family": "mixed_unresolved_compilation",
            "native_endpoint": "pIC50",
            "native_relation": ">=",
            "native_value": 6.2,
            "native_unit": "pIC50",
            "pic50_value": 6.2,
            "pic50_origin": "source_reported_pIC50",
            "native_aux_json": "{}",
        },
        {
            "observation_id": "M1",
            "source_family": "fixture",
            "source_record_id": "RM",
            "standardized_smiles": "CCC",
            "standard_inchi_key": "ATUOYWHBWRKTHZ-UHFFFAOYSA-N",
            "structure_id": "SM",
            "structure_valid": True,
            "target_variant": "mutant_or_variant",
            "assay_id": "AM",
            "assay_family": "functional",
            "native_endpoint": "IC50",
            "native_relation": "=",
            "native_value": 10.0,
            "native_unit": "nM",
            "pic50_value": 8.0,
            "pic50_origin": "converted",
            "native_aux_json": "{}",
        },
    ]
    _write(roots["hierarchy"] / "observation_ledger.parquet", ledger_rows)
    _write(
        roots["scope"] / "wildtype_observation_index.parquet",
        [{"observation_id": "W1"}, {"observation_id": "W2"}],
    )
    _write(
        roots["scope"] / "explicit_mutant_exclusions.parquet",
        [
            {
                "observation_id": "M1",
                "structure_id": "SM",
                "source_family": "fixture",
                "target_variant_original": "mutant_or_variant",
                "exclusion_reason": "explicit_mutant_or_variant",
            }
        ],
    )
    modality_rows = []
    for observation_id, structure_id, scope in (
        ("W1", "S1", "confirmed_wild_type"),
        ("W2", "S2", "wild_type_or_unspecified"),
    ):
        modality_rows.append(
            {
                "observation_id": observation_id,
                "wild_type_evidence_scope": scope,
                "measurement_modality": "patch_clamp_electrophysiology"
                if observation_id == "W1"
                else "unresolved",
                "method_detail": "fixture",
                "modality_confidence": "high" if observation_id == "W1" else "unresolved",
                "automation_class": "automated" if observation_id == "W1" else "unresolved",
                "automation_confidence": "high" if observation_id == "W1" else "unresolved",
                "dose_design": "concentration_response_summary",
                "dose_design_confidence": "high",
                "structure_id": structure_id,
            }
        )
    _write(roots["modality"] / "herg_measurement_modality_index.parquet", modality_rows)
    _write(roots["modality"] / "qt_clinical_phenotype_index.parquet", [{"candidate_id": "QT1"}])
    _write(
        roots["split"] / "structure_consensus_binary_scaffold_split.parquet",
        [{"structure_id": "S1", "split": "train", "scaffold_group_id": "G1"}],
    )

    q0 = _task_row()
    q1 = _task_row(
        task_id="Q1_QUANTITATIVE_PIC50",
        quality_level="Q1_quantitative_compilation",
        record_id="R-Q1",
        observation_id="W2",
        structure_id="S2",
        standardized_smiles="CCO",
        standard_inchi_key="LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
        target_scope="wild_type_or_unspecified",
        source_family="fixture_quant",
        assay_id=None,
        assay_family="mixed_unresolved_compilation",
        measurement_technology="compiled_mixed_or_unresolved",
        native_endpoint="pIC50",
        native_relation="=",
        native_value=6.2,
        native_unit="pIC50",
        target_relation="=",
        target_pic50=6.2,
        target_class=2,
        model_split="test",
        scaffold_group_id="G2",
    )
    q2 = _task_row(
        task_id="Q2_FUNCTIONAL_ASSAY_AWARE", quality_level="Q2_functional_assay_aware", record_id="R-Q2"
    )
    c0 = _task_row(
        task_id="C0_CLINICAL_DEVELOPMENT_CONTEXT",
        quality_level="C0_clinical_development_context",
        record_id="R-C0",
        observation_id=None,
        target_scope="clinical_context_not_target_variant",
        source_family="fixture_clinical",
        assay_id=None,
        assay_family="not_applicable_clinical_context",
        measurement_technology="not_applicable_clinical_context",
        native_endpoint="clinical_development_annotation",
        native_relation=None,
        native_value=None,
        native_unit=None,
        target_relation=None,
        target_pic50=None,
        target_class=None,
        task_role="context",
        direct_herg_label=False,
        use_as_training_label=False,
        clinical_context_only=True,
        quality_flags='{"max_phase":4}',
    )
    for name, rows in (
        ("q0_weak_fixed_dose_binary.parquet", [q0]),
        ("q1_quantitative_pic50.parquet", [q1]),
        ("q2_functional_assay_aware.parquet", [q2]),
        ("c0_clinical_development_context.parquet", [c0]),
    ):
        _write(roots["tasks"] / name, rows, quality_tasks._TASK_SCHEMA)
    c1 = {
        "task_id": "C1_QT_CONTEXT_EVALUATION",
        "candidate_id": "QT1",
        "structure_id": "S1",
        "nct_id": "NCT1",
        "endpoint_candidate_id": "E1",
        "record_kind": "reported_outcome_measure",
        "candidate_classification": "qt_qtc_interval_measure_candidate",
        "title_or_term": "QTcF change",
        "description_or_organ_system": "ECG",
        "unit_of_measure": "msec",
        "time_frame": "Day 1",
        "reported_numeric_value_count": 1,
        "value_records_json": "[]",
        "denominator_records_json": "[]",
        "model_split": "train",
        "scaffold_group_id": "G1",
        "context_eligible": True,
        "heldout_evaluation_eligible": False,
        "direct_herg_label": False,
        "use_as_training_label": False,
        "context_semantics": "human_QT_QTc_context_not_hERG_label",
    }
    _write(roots["tasks"] / "c1_qt_context_endpoints.parquet", [c1], quality_tasks._QT_SCHEMA)
    mutant_exclusion = {
        "task_id": "ALL_DIRECT_HERG_TASKS",
        "source_family": "fixture",
        "source_record_id": "RM",
        "observation_id": "M1",
        "structure_id": "SM",
        "target_scope": "mutant_or_variant",
        "exclusion_reason": "explicit_mutant_or_variant_target",
        "exclusion_detail": "fixture",
    }
    _write(roots["tasks"] / "exclusion_ledger.parquet", [mutant_exclusion], quality_tasks._EXCLUSION_SCHEMA)
    return roots


def _build(tmp_path: Path, suffix: str = "") -> tuple[dict[str, object], Path, Path]:
    roots = _fixture(tmp_path)
    output = tmp_path / f"output{suffix}"
    report = tmp_path / f"report{suffix}"
    manifest = build_herg_master_dataset(
        hierarchy_root=roots["hierarchy"],
        wildtype_scope_root=roots["scope"],
        modality_qt_root=roots["modality"],
        quality_tasks_root=roots["tasks"],
        model_ready_root=roots["split"],
        output_root=output,
        report_root=report,
    )
    return manifest, output, report


def test_builds_standardized_wild_type_master_without_inventing_physics(tmp_path: Path) -> None:
    manifest, output, report = _build(tmp_path)
    observations = {
        row["observation_id"]: row for row in pq.read_table(output / OBSERVATION_OUTPUT).to_pylist()
    }
    assert set(observations) == {"W1", "W2"}
    assert observations["W1"]["potency_relation_pic50"] == "<"
    assert observations["W1"]["potency_pic50_point"] is None
    assert observations["W1"]["potency_pic50_lower_bound"] is None
    assert observations["W1"]["potency_pic50_upper_bound"] == pytest.approx(6.0)
    assert observations["W2"]["potency_relation_pic50"] == ">="
    assert observations["W2"]["potency_pic50_point"] is None
    assert observations["W2"]["potency_pic50_lower_bound"] == pytest.approx(6.2)
    assert observations["W2"]["potency_pic50_upper_bound"] is None

    structures = pq.read_table(output / STRUCTURE_OUTPUT).to_pylist()
    assert len(structures) == 2
    assert all(row["feature_status"] == "complete" for row in structures)
    assert all(row["molecular_weight"] is not None for row in structures)
    assert "pKa" in manifest["policies"]["not_computed"]
    assert manifest["model_feature_contract"]["labels_join_only_after_partitioning"]
    assert pq.read_table(output / ASSAY_OUTPUT).num_rows == 2
    protocols = pq.read_table(output / PROTOCOL_OUTPUT).to_pylist()
    explicit = next(row for row in protocols if row["assay_id"] == "A1")
    assert json.loads(explicit["host_systems_json"]) == ["HEK293"]
    assert json.loads(explicit["voltage_values_mv_json"]) == [-80.0]
    assert json.loads(explicit["temperature_values_celsius_json"]) == [37.0]
    assert json.loads(explicit["time_values_seconds_json"]) == [300.0]
    assert json.loads(explicit["recording_configurations_json"]) == ["patch_clamp", "whole_cell"]
    assert json.loads(explicit["named_platforms_json"]) == ["QPatch"]
    assert pq.read_table(output / TASK_OUTPUT).num_rows == 5
    assert pq.read_table(output / CLINICAL_OUTPUT).num_rows == 2
    assert pq.read_table(output / EXCLUSION_OUTPUT).num_rows == 1
    assert (report / "HERG_MASTER_DATASET.md").is_file()
    assert validate_herg_master_dataset(output) == manifest


def test_rebuild_is_byte_deterministic_and_tampering_fails(tmp_path: Path) -> None:
    manifest_a, output_a, _ = _build(tmp_path / "a", "a")
    manifest_b, output_b, _ = _build(tmp_path / "b", "b")
    hashes_a = {name: meta["sha256"] for name, meta in manifest_a["artifacts"].items()}
    hashes_b = {name: meta["sha256"] for name, meta in manifest_b["artifacts"].items()}
    assert hashes_a == hashes_b
    with (output_a / STRUCTURE_OUTPUT).open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(HergMasterDatasetError, match="hash mismatch"):
        validate_herg_master_dataset(output_a)
    assert (output_b / MANIFEST_NAME).is_file()
