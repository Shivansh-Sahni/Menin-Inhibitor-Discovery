from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from menin_discovery.platform_herg_quality_tasks import (
    C0_OUTPUT,
    C1_OUTPUT,
    EXCLUSION_OUTPUT,
    Q0_OUTPUT,
    Q1_OUTPUT,
    Q2_OUTPUT,
    QT_RECORD_OUTPUT,
    HergQualityTaskError,
    _convert_ic50_to_pic50,
    _entity_split_assignments,
    _measurement_technology,
    build_herg_quality_tasks,
    verify_herg_quality_tasks,
)

OBS_SCHEMA = pa.schema(
    [
        pa.field("observation_id", pa.large_string()),
        pa.field("source_family", pa.large_string()),
        pa.field("source_record_id", pa.large_string()),
        pa.field("structure_id", pa.large_string()),
        pa.field("standardized_smiles", pa.large_string()),
        pa.field("standard_inchi_key", pa.large_string()),
        pa.field("structure_valid", pa.bool_()),
        pa.field("target_variant", pa.large_string()),
        pa.field("assay_id", pa.large_string()),
        pa.field("assay_family", pa.large_string()),
        pa.field("native_endpoint", pa.large_string()),
        pa.field("native_relation", pa.large_string()),
        pa.field("native_value", pa.float64()),
        pa.field("native_unit", pa.large_string()),
        pa.field("native_label", pa.large_string()),
        pa.field("pic50_value", pa.float64()),
        pa.field("source_split", pa.large_string()),
        pa.field("native_aux_json", pa.large_string()),
        pa.field("quality_flags", pa.large_string()),
    ]
)


def _obs(
    observation_id: str,
    source: str,
    structure_id: str | None,
    smiles: str | None,
    *,
    variant: str,
    family: str,
    endpoint: str,
    value: float | None = None,
    unit: str | None = None,
    relation: str | None = None,
    label: str | None = None,
    pic50: float | None = None,
    aux: str = "{}",
    assay_id: str | None = None,
) -> dict[str, object]:
    return {
        "observation_id": observation_id,
        "source_family": source,
        "source_record_id": f"SRC:{observation_id}",
        "structure_id": structure_id,
        "standardized_smiles": smiles,
        "standard_inchi_key": f"KEY-{structure_id}" if structure_id else None,
        "structure_valid": structure_id is not None,
        "target_variant": variant,
        "assay_id": assay_id or f"ASSAY-{observation_id}",
        "assay_family": family,
        "native_endpoint": endpoint,
        "native_relation": relation,
        "native_value": value,
        "native_unit": unit,
        "native_label": label,
        "pic50_value": pic50,
        "source_split": "Train" if source == "quantitative_pic50_release" else None,
        "native_aux_json": aux,
        "quality_flags": "",
    }


def _write_fixture(root: Path) -> tuple[Path, Path, Path, Path]:
    hierarchy = root / "hierarchy"
    model = root / "model"
    clinical = root / "clinical"
    operational = root / "operational"
    for directory in (hierarchy, model, clinical, operational):
        directory.mkdir()
    observations = [
        _obs(
            "P1",
            "pubchem_aid720551",
            "S1",
            "CCO",
            variant="wild_type",
            family="source_reported_qhts",
            endpoint="activity_outcome",
            label="Active",
        ),
        _obs(
            "P2",
            "pubchem_aid720551",
            "S2",
            "c1ccccc1",
            variant="wild_type",
            family="source_reported_qhts",
            endpoint="activity_outcome",
            label="Inactive",
        ),
        _obs(
            "P3",
            "pubchem_aid720551",
            "S3",
            "CCN",
            variant="wild_type",
            family="source_reported_qhts",
            endpoint="activity_outcome",
            label="Inconclusive",
        ),
        _obs(
            "Q1",
            "quantitative_pic50_release",
            "S1",
            "CCO",
            variant="wild_type_or_unspecified",
            family="mixed_unresolved_compilation",
            endpoint="pIC50",
            value=5.2,
            unit="pIC50",
            relation="=",
            pic50=5.2,
        ),
        _obs(
            "F1",
            "chembl_herg_specialized_view",
            "S2",
            "c1ccccc1",
            variant="wild_type_or_unspecified",
            family="functional",
            endpoint="IC50",
            value=100.0,
            unit="nM",
            relation="=",
            pic50=7.0,
            aux=json.dumps({"assay_description": "manual whole-cell patch clamp"}),
        ),
        _obs(
            "F2",
            "chembl_herg_specialized_view",
            "S2",
            "c1ccccc1",
            variant="wild_type_or_unspecified",
            family="functional",
            endpoint="IC50",
            value=120.0,
            unit="nM",
            relation=None,
            pic50=None,
        ),
        _obs(
            "F3",
            "chembl_herg_specialized_view",
            "S2",
            "c1ccccc1",
            variant="wild_type_or_unspecified",
            family="functional",
            endpoint="EC10",
            value=800.0,
            unit="nM",
            relation="=",
            assay_id="CHEMBL820994",
        ),
        _obs(
            "M1",
            "chembl_herg_specialized_view",
            "S4",
            "CCC",
            variant="mutant_or_variant",
            family="functional",
            endpoint="IC50",
            value=50.0,
            unit="nM",
            relation="=",
            pic50=7.3,
        ),
    ]
    pq.write_table(
        pa.Table.from_pylist(observations, schema=OBS_SCHEMA), hierarchy / "observation_ledger.parquet"
    )
    split_schema = pa.schema(
        [
            pa.field("structure_id", pa.large_string()),
            pa.field("standardized_smiles", pa.large_string()),
            pa.field("standard_inchi_key", pa.large_string()),
            pa.field("herg_blocker_label", pa.int8()),
            pa.field("split", pa.large_string()),
            pa.field("scaffold_group_id", pa.large_string()),
        ]
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "structure_id": "S1",
                    "standardized_smiles": "CCO",
                    "standard_inchi_key": "KEY-S1",
                    "herg_blocker_label": 1,
                    "split": "train",
                    "scaffold_group_id": "HSCF-A",
                },
                {
                    "structure_id": "S2",
                    "standardized_smiles": "c1ccccc1",
                    "standard_inchi_key": "KEY-S2",
                    "herg_blocker_label": 0,
                    "split": "test",
                    "scaffold_group_id": "HSCF-B",
                },
            ],
            schema=split_schema,
        ),
        model / "structure_consensus_binary_scaffold_split.parquet",
    )
    dev_schema = pa.schema(
        [
            pa.field("molecule_id", pa.string()),
            pa.field("standard_inchi_key", pa.string()),
            pa.field("canonical_smiles", pa.string()),
            pa.field("chembl_max_phase", pa.float64()),
            pa.field("chembl_first_approval", pa.int64()),
            pa.field("chembl_therapeutic_flag", pa.bool_()),
            pa.field("chembl_dosed_ingredient", pa.bool_()),
            pa.field("chembl_withdrawn_flag", pa.bool_()),
            pa.field("drugsfda_exact_name_link_count", pa.int64()),
            pa.field("clinical_development_annotation", pa.bool_()),
        ]
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "molecule_id": "S1",
                    "standard_inchi_key": "KEY-S1",
                    "canonical_smiles": "CCO",
                    "chembl_max_phase": 4.0,
                    "chembl_first_approval": 1990,
                    "chembl_therapeutic_flag": True,
                    "chembl_dosed_ingredient": True,
                    "chembl_withdrawn_flag": False,
                    "drugsfda_exact_name_link_count": 1,
                    "clinical_development_annotation": True,
                }
            ],
            schema=dev_schema,
        ),
        clinical / "structure_development_annotations.parquet",
    )
    qt_schema = pa.schema(
        [
            pa.field("candidate_id", pa.string()),
            pa.field("molecule_id", pa.string()),
            pa.field("nct_id", pa.string()),
            pa.field("endpoint_candidate_id", pa.string()),
            pa.field("record_kind", pa.string()),
            pa.field("candidate_classification", pa.string()),
            pa.field("title_or_term", pa.string()),
            pa.field("description_or_organ_system", pa.string()),
            pa.field("unit_of_measure", pa.string()),
            pa.field("time_frame", pa.string()),
            pa.field("reported_numeric_value_count", pa.int64()),
            pa.field("value_records_json", pa.string()),
            pa.field("denominator_records_json", pa.string()),
            pa.field("candidate_rule_passed", pa.bool_()),
            pa.field("exact_unique_molecule_link", pa.bool_()),
            pa.field("actual_qt_result_present", pa.bool_()),
        ]
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "candidate_id": "QT1",
                    "molecule_id": "S2",
                    "nct_id": "NCT1",
                    "endpoint_candidate_id": "EP1",
                    "record_kind": "outcome",
                    "candidate_classification": "qt_qtc",
                    "title_or_term": "QTc change",
                    "description_or_organ_system": "ECG",
                    "unit_of_measure": "ms",
                    "time_frame": "day 1",
                    "reported_numeric_value_count": 1,
                    "value_records_json": "[]",
                    "denominator_records_json": "[]",
                    "candidate_rule_passed": True,
                    "exact_unique_molecule_link": True,
                    "actual_qt_result_present": True,
                }
            ],
            schema=qt_schema,
        ),
        clinical / "t3_posted_qt_trial_result_candidates.parquet",
    )
    record_schema = pa.schema(
        [
            pa.field("record_id", pa.large_string()),
            pa.field("source_candidate_id", pa.large_string()),
            pa.field("structure_id", pa.large_string()),
            pa.field("nct_id", pa.large_string()),
            pa.field("endpoint_candidate_id", pa.large_string()),
            pa.field("record_ordinal", pa.int64()),
            pa.field("reported_value_is_numeric", pa.bool_()),
            pa.field("source_page_path", pa.large_string()),
            pa.field("raw_json_pointer", pa.large_string()),
        ]
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "record_id": "R1",
                    "source_candidate_id": "QT1",
                    "structure_id": "S2",
                    "nct_id": "NCT1",
                    "endpoint_candidate_id": "EP1",
                    "record_ordinal": 0,
                    "reported_value_is_numeric": True,
                    "source_page_path": "page.json",
                    "raw_json_pointer": "/x/0",
                }
            ],
            schema=record_schema,
        ),
        operational / "operational_qt_record_index.parquet",
    )
    return hierarchy, model, clinical, operational


def test_measurement_technology_and_ic50_conversion() -> None:
    technology, basis = _measurement_technology(
        {
            "source_family": "chembl_herg_specialized_view",
            "assay_family": "functional",
            "native_aux_json": json.dumps({"assay_description": "QPatch automated patch clamp"}),
        }
    )
    assert technology == "automated_patch_clamp"
    assert basis == "assay_description_keyword"
    assert _convert_ic50_to_pic50(100.0, "nM") == pytest.approx(7.0)
    assert _convert_ic50_to_pic50(0.0, "nM") is None


def test_entity_split_assignment_is_order_independent_and_entity_exclusive() -> None:
    observations = [
        {
            "structure_id": "S1",
            "standardized_smiles": "N=C(N)N",
            "target_variant": "wild_type_or_unspecified",
        },
        {
            "structure_id": "S1",
            "standardized_smiles": "NC(=N)N",
            "target_variant": "wild_type_or_unspecified",
        },
        {
            "structure_id": "S1",
            "standardized_smiles": "NC(=N)N",
            "target_variant": "wild_type_or_unspecified",
        },
    ]
    first = _entity_split_assignments(observations, {})
    second = _entity_split_assignments(list(reversed(observations)), {})
    assert first == second
    assert set(first) == {"S1"}


def test_build_quality_tasks_is_wild_type_only_and_context_safe(tmp_path: Path) -> None:
    hierarchy, model, clinical, operational = _write_fixture(tmp_path)
    output, report = tmp_path / "output", tmp_path / "report"
    manifest = build_herg_quality_tasks(
        hierarchy_root=hierarchy,
        model_ready_root=model,
        clinical_links_root=clinical,
        operational_tiers_root=operational,
        output_root=output,
        report_root=report,
    )
    assert manifest["qc"]["explicit_mutant_exclusions"] == 1
    assert pq.ParquetFile(output / Q0_OUTPUT).metadata.num_rows == 3
    assert pq.ParquetFile(output / Q1_OUTPUT).metadata.num_rows == 2
    assert pq.ParquetFile(output / Q2_OUTPUT).metadata.num_rows == 3
    assert pq.ParquetFile(output / C0_OUTPUT).metadata.num_rows == 1
    assert pq.ParquetFile(output / C1_OUTPUT).metadata.num_rows == 1
    assert pq.ParquetFile(output / QT_RECORD_OUTPUT).metadata.num_rows == 1
    exclusions = pq.read_table(output / EXCLUSION_OUTPUT).to_pylist()
    assert any(row["exclusion_reason"] == "explicit_mutant_or_variant_target" for row in exclusions)
    q1_scopes = {row["target_scope"] for row in pq.read_table(output / Q1_OUTPUT).to_pylist()}
    assert q1_scopes == {"wild_type_or_unspecified"}
    q2_rows = pq.read_table(output / Q2_OUTPUT).to_pylist()
    unresolved = [row for row in q2_rows if row["observation_id"] == "F2"]
    assert len(unresolved) == 1
    assert unresolved[0]["eligible"] is False
    assert unresolved[0]["use_as_training_label"] is False
    assert unresolved[0]["exclusion_reason"] == "missing_native_relation_for_ic50"
    clinical = [row for row in q2_rows if row["observation_id"] == "F3"]
    assert len(clinical) == 1
    assert clinical[0]["clinical_context_only"] is True
    assert clinical[0]["direct_herg_label"] is False
    assert clinical[0]["use_as_training_label"] is False
    assert clinical[0]["exclusion_reason"] == "clinical_qt_phenotype_not_direct_herg_potency"
    assert not any(row["use_as_training_label"] for row in pq.read_table(output / C1_OUTPUT).to_pylist())
    assert (report / "HERG_QUALITY_TASKS.md").is_file()
    verify_herg_quality_tasks(output_root=output)


def test_validator_detects_artifact_tampering(tmp_path: Path) -> None:
    hierarchy, model, clinical, operational = _write_fixture(tmp_path)
    output, report = tmp_path / "output", tmp_path / "report"
    build_herg_quality_tasks(
        hierarchy_root=hierarchy,
        model_ready_root=model,
        clinical_links_root=clinical,
        operational_tiers_root=operational,
        output_root=output,
        report_root=report,
    )
    with (output / Q0_OUTPUT).open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(HergQualityTaskError, match="artifact hash mismatch"):
        verify_herg_quality_tasks(output_root=output)


def test_builder_refuses_existing_output(tmp_path: Path) -> None:
    hierarchy, model, clinical, operational = _write_fixture(tmp_path)
    output, report = tmp_path / "output", tmp_path / "report"
    output.mkdir()
    with pytest.raises(HergQualityTaskError, match="must not already exist"):
        build_herg_quality_tasks(
            hierarchy_root=hierarchy,
            model_ready_root=model,
            clinical_links_root=clinical,
            operational_tiers_root=operational,
            output_root=output,
            report_root=report,
        )
