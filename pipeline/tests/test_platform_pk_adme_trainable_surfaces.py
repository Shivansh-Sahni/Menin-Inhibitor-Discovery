from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from menin_discovery.platform_pk_adme_trainable_surfaces import (
    _EXPECTED_OUTPUT_MEMBERS,
    PKADMESurfaceError,
    _bind_file,
    _chembl_spec,
    _source_context_flags,
    parse_numeric_relation,
    relation_bounds,
    self_hash_manifest,
    standardize_structure,
    validate_closed_release_membership,
    verify_input_binding,
    verify_manifest_self_hash,
)


def test_numeric_relation_preserves_only_physically_supported_bounds() -> None:
    assert parse_numeric_relation("12.5") == ("=", 12.5, None, None, False)
    assert parse_numeric_relation("< 12.5") == ("<", 12.5, None, 12.5, True)
    assert parse_numeric_relation(">= 12.5") == (">=", 12.5, 12.5, None, True)
    assert parse_numeric_relation("not measured") is None
    assert relation_bounds("=", 2.0, 3.0) == ("interval", 2.0, 3.0, True)


def test_chembl_contract_is_case_sensitive_and_unit_specific() -> None:
    assert _chembl_spec("Cl", "mL.min-1.kg-1", "chloride", None, None) is None
    assert _chembl_spec("t1/2", "hr", "half life", None, None) is None
    systemic = _chembl_spec(
        "CL",
        "mL.min-1.kg-1",
        "Clearance in human after iv administration",
        None,
        None,
    )
    assert systemic == (
        "clearance",
        "clearance",
        "systemic_or_total",
        "mL/min/kg",
        "regression",
        1.0,
    )
    papp = _chembl_spec("Papp", "nm/s", "PAMPA permeability", None, None)
    assert papp is not None
    assert papp[-1] == pytest.approx(0.1)


def test_standardization_groups_salts_and_stereochemistry_conservatively() -> None:
    neutral = standardize_structure("CCO")
    salt = standardize_structure("CCO.[Na+]")
    assert neutral.status == "standardized"
    assert neutral.standard_inchi_key == salt.standard_inchi_key
    assert neutral.leakage_group_id == salt.leakage_group_id

    left = standardize_structure("N[C@@H](C)C(=O)O")
    right = standardize_structure("N[C@H](C)C(=O)O")
    assert left.standard_inchi_key != right.standard_inchi_key
    assert left.connectivity_key == right.connectivity_key
    assert left.leakage_group_id == right.leakage_group_id


def test_standardization_separates_tautomer_representations_but_groups_leakage() -> None:
    lactam = standardize_structure("O=c1cccc[nH]1")
    lactim = standardize_structure("Oc1ccccn1")
    assert lactam.standard_inchi_key == lactim.standard_inchi_key
    assert lactam.molecule_id != lactim.molecule_id
    assert lactam.leakage_group_id == lactim.leakage_group_id


def test_source_context_flags_do_not_create_labels_or_misread_clinical_isolates() -> None:
    assert _source_context_flags(
        {"description": "Phase I clinical trial in healthy volunteers with QTc ECG monitoring"}
    ) == (True, True)
    assert _source_context_flags({"description": "activity against clinical isolates"}) == (
        False,
        False,
    )
    assert _source_context_flags(
        {"description": "phase I mediated metabolite formation in mouse liver microsomes"}
    ) == (False, False)
    assert _source_context_flags({"description": "PK in healthy human subjects"}) == (True, False)
    assert _source_context_flags({"description": "phase 1/2 trial oral half life"}) == (
        True,
        False,
    )
    assert _source_context_flags({"description": "phase 1 oral half life in rats"}) == (
        False,
        False,
    )
    assert _source_context_flags({"description": "single oral dose in healthy adult male volunteers"}) == (
        True,
        False,
    )
    assert _source_context_flags({"description": "Phase II clinical trial PK study"}) == (
        True,
        False,
    )
    assert _source_context_flags({"description": "healthy human liver microsomes"}) == (
        False,
        False,
    )
    assert _source_context_flags({"description": "intravenous dose in human subjects"}) == (
        True,
        False,
    )
    assert _source_context_flags(
        {"description": "oral administration to healthy individuals"}, "Homo sapiens"
    ) == (True, False)
    assert _source_context_flags({"description": "PK after dosing in healthy boys"}, "Homo sapiens") == (
        True,
        False,
    )
    assert _source_context_flags({"description": "AZT hydrolysis in normal human serum"}) == (
        False,
        False,
    )
    assert _source_context_flags({"description": "male human plasma stability"}) == (
        False,
        False,
    )
    assert _source_context_flags({"description": "phase 2 metabolism in human liver S9 UDPGA"}) == (
        False,
        False,
    )
    assert _source_context_flags(
        {"description": "AUC in healthy human plasma at 800 mg q12h on day 15"},
        "Homo sapiens",
    ) == (True, False)
    assert _source_context_flags(
        {"description": "half life after oral dose in human children"}, "Homo sapiens"
    ) == (True, False)
    assert _source_context_flags(
        {"description": "simulated one-compartment oral dose in human serum"}, "Homo sapiens"
    ) == (False, False)
    assert _source_context_flags(
        {"description": "protein binding in human serum at 1 ug/mL"}, "Homo sapiens"
    ) == (False, False)
    assert _source_context_flags(
        {"description": "metabolite Cmax after oral administration to healthy subjects"},
        "Homo sapiens",
    ) == (True, False)
    assert _source_context_flags(
        {"description": "half life after IV dosing in infants on extracorporeal membrane oxygenation"},
        "Homo sapiens",
    ) == (True, False)
    assert _source_context_flags(
        {"description": "half life in 10% human serum at 1 mM dose"}, "Homo sapiens"
    ) == (False, False)
    assert _source_context_flags(
        {"description": "recombinant protein surface plasmon binding administered for 60 secs"},
        "Homo sapiens",
    ) == (False, False)
    assert _source_context_flags(
        {"description": "Cmax after 236 mg infused over 30 mins in normal-weight adults"},
        "Homo sapiens",
    ) == (True, False)
    assert _source_context_flags(
        {"description": "AUC in infected human at 200 mg/day perorally"}, "Homo sapiens"
    ) == (True, False)
    assert _source_context_flags(
        {"description": "one-compartment model of half life in lymphoma patients"},
        "Homo sapiens",
    ) == (True, False)
    assert _source_context_flags(
        {"description": "intracellular AUC in PBMCs after 600 mg QD in HIV-positive humans"},
        "Homo sapiens",
    ) == (True, False)
    assert _source_context_flags(
        {"description": "Cmax in CYP-genotyped human treated with losartan"}, "Homo sapiens"
    ) == (True, False)
    assert _source_context_flags(
        {"description": "AUC after 0.25 mg SC administration in human"}, "Homo sapiens"
    ) == (True, False)
    assert _source_context_flags(
        {"description": "intranasal bioavailability of a human formulation"}, "Homo sapiens"
    ) == (True, False)


def test_artifact_manifest_payload_self_hash_detects_tamper() -> None:
    manifest = self_hash_manifest({"schema_version": "fixture", "artifacts": [{"path": "a"}]})
    verify_manifest_self_hash(manifest)
    tampered = json.loads(json.dumps(manifest))
    tampered["artifacts"][0]["path"] = "b"
    with pytest.raises(PKADMESurfaceError, match="self-hash mismatch"):
        verify_manifest_self_hash(tampered)


def test_parquet_input_binding_enforces_bytes_rows_schema_and_sha(tmp_path: Path) -> None:
    parquet_path = tmp_path / "input.parquet"
    pq.write_table(pa.table({"value": pa.array([1, 2], type=pa.int64())}), parquet_path)
    binding = _bind_file(tmp_path, parquet_path, "fixture_parquet", None)
    assert binding["row_count"] == 2
    assert len(binding["arrow_schema_sha256"]) == 64
    assert verify_input_binding(tmp_path, binding)["arrow_schema_verified"]

    wrong_rows = dict(binding, row_count=3)
    with pytest.raises(PKADMESurfaceError, match="row-count mismatch"):
        verify_input_binding(tmp_path, wrong_rows)
    wrong_schema = dict(binding, arrow_schema_sha256="0" * 64)
    with pytest.raises(PKADMESurfaceError, match="Arrow schema mismatch"):
        verify_input_binding(tmp_path, wrong_schema)

    parquet_path.write_bytes(parquet_path.read_bytes() + b"tamper")
    with pytest.raises(PKADMESurfaceError, match="byte-size mismatch"):
        verify_input_binding(tmp_path, binding)


def test_release_membership_is_closed_and_rejects_extra_output(tmp_path: Path) -> None:
    output = tmp_path / "release"
    output.mkdir()
    for name in _EXPECTED_OUTPUT_MEMBERS:
        (output / name).write_text("fixture", encoding="utf-8")
    report_dir = tmp_path / "report"
    report_dir.mkdir()
    report = report_dir / "PK_ADME_TRAINABLE_SURFACES.md"
    report.write_text("fixture", encoding="utf-8")
    validate_closed_release_membership(output, report)

    (output / "unexpected.txt").write_text("tamper", encoding="utf-8")
    with pytest.raises(PKADMESurfaceError, match="membership mismatch"):
        validate_closed_release_membership(output, report)


def test_report_location_membership_is_closed(tmp_path: Path) -> None:
    output = tmp_path / "release"
    output.mkdir()
    for name in _EXPECTED_OUTPUT_MEMBERS:
        (output / name).write_text("fixture", encoding="utf-8")
    report_dir = tmp_path / "report"
    report_dir.mkdir()
    report = report_dir / "PK_ADME_TRAINABLE_SURFACES.md"
    report.write_text("fixture", encoding="utf-8")
    (report_dir / "shadow.md").write_text("tamper", encoding="utf-8")
    with pytest.raises(PKADMESurfaceError, match="Report directory membership mismatch"):
        validate_closed_release_membership(output, report)
