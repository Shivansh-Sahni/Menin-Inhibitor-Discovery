from __future__ import annotations

import csv
import gzip
import json
import sqlite3
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from menin_discovery.platform_herg_clinical_links import build_herg_clinical_links
from menin_discovery.platform_herg_hierarchy import build_herg_hierarchy
from menin_discovery.platform_herg_tiers import (
    MANIFEST_NAME,
    STRUCTURE_TIER_OUTPUT,
    T1_CANDIDATE_OUTPUT,
    HergTierIntegrationError,
    build_herg_evidence_tiers,
    main,
    validate_herg_evidence_tiers,
)


def _csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> Path:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _gzip_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> Path:
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _build_hierarchy(tmp_path: Path) -> Path:
    structures = _csv(
        tmp_path / "structures.csv",
        ["CID", "ConnectivitySMILES"],
        [
            {"CID": 1, "ConnectivitySMILES": "CCO"},
            {"CID": 2, "ConnectivitySMILES": "CCN"},
            {"CID": 3, "ConnectivitySMILES": "CCC"},
        ],
    )
    outcomes = _csv(
        tmp_path / "outcomes.csv",
        ["AID", "SID", "CID", "Activity Outcome", "Target GeneID", "Assay Name", "Assay Type"],
        [
            {
                "AID": 720551,
                "SID": 11,
                "CID": 1,
                "Activity Outcome": "Active",
                "Target GeneID": 3757,
                "Assay Name": "wild-type qHTS",
                "Assay Type": "Confirmatory",
            },
            {
                "AID": 720551,
                "SID": 12,
                "CID": 2,
                "Activity Outcome": "Inactive",
                "Target GeneID": 3757,
                "Assay Name": "wild-type qHTS",
                "Assay Type": "Confirmatory",
            },
            {
                "AID": 720551,
                "SID": 13,
                "CID": 3,
                "Activity Outcome": "Active",
                "Target GeneID": 3757,
                "Assay Name": "wild-type qHTS",
                "Assay Type": "Confirmatory",
            },
        ],
    )
    quantitative = _csv(
        tmp_path / "quantitative.csv",
        ["InChl Key", "SMILES", "Source", "pIC50", "USED_AS"],
        [{"InChl Key": "", "SMILES": "CCCC", "Source": "fixture", "pIC50": 5.5, "USED_AS": "Train"}],
    )
    chembl = tmp_path / "chembl_herg.parquet"
    pq.write_table(
        pa.table(
            {
                "activity_id": [101, 102],
                "standard_type": ["IC50", "IC50"],
                "standard_relation": ["=", "="],
                "standard_value": [1000.0, 100.0],
                "standard_units": ["nM", "nM"],
                "assay_type": ["F", "F"],
                "canonical_smiles": ["CCO", "CCN"],
                "target_chembl_id": ["CHEMBL240", "CHEMBL240"],
                "standard_flag": [1, 1],
                "potential_duplicate": [0, 0],
                "data_validity_comment": [None, None],
                "confidence_score": [9, 9],
                "variant_id": [None, None],
                "assay_chembl_id": ["CHEMBL-A1", "CHEMBL-A2"],
                "assay_description": ["fixture functional", "fixture functional"],
            }
        ),
        chembl,
    )
    output = tmp_path / "hierarchy"
    build_herg_hierarchy(
        pubchem_outcomes_path=outcomes,
        pubchem_structures_path=structures,
        quantitative_pic50_paths=[quantitative],
        chembl_parquet_paths=[chembl],
        output_root=output,
    )
    return output


def _build_clinical_links(tmp_path: Path, hierarchy: Path) -> Path:
    chembl = tmp_path / "chembl.db"
    connection = sqlite3.connect(chembl)
    connection.executescript(
        """
        CREATE TABLE compound_structures (
          molregno INTEGER PRIMARY KEY, standard_inchi_key TEXT, canonical_smiles TEXT
        );
        CREATE TABLE molecule_dictionary (
          molregno INTEGER PRIMARY KEY, chembl_id TEXT, pref_name TEXT, max_phase REAL,
          first_approval INTEGER, therapeutic_flag INTEGER, dosed_ingredient INTEGER,
          withdrawn_flag INTEGER
        );
        CREATE TABLE molecule_synonyms (molregno INTEGER, synonyms TEXT);
        """
    )
    consensus = pq.read_table(hierarchy / "structure_consensus_binary.parquet").to_pylist()
    by_smiles = {row["standardized_smiles"]: row for row in consensus}
    for molregno, (smiles, name, phase) in enumerate(
        [("CCO", "Drug A", 4.0), ("CCN", "Drug B", 2.0), ("CCC", "Drug C", None)], start=1
    ):
        row = by_smiles[smiles]
        connection.execute(
            "INSERT INTO compound_structures VALUES (?, ?, ?)",
            (molregno, row["standard_inchi_key"], smiles),
        )
        connection.execute(
            "INSERT INTO molecule_dictionary VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (molregno, f"CHEMBL{molregno}", name, phase, 2005 if phase else None, 1, 1, 0),
        )
    connection.commit()
    connection.close()

    clinical = tmp_path / "clinical"
    clinical.mkdir()
    _gzip_csv(
        clinical / "studies.csv.gz",
        ["nct_id", "has_results_reported"],
        [{"nct_id": "NCT00000001", "has_results_reported": "true"}],
    )
    _gzip_csv(
        clinical / "interventions.csv.gz",
        ["nct_id", "intervention_candidate_id", "intervention_type", "intervention_name"],
        [
            {
                "nct_id": "NCT00000001",
                "intervention_candidate_id": "int-1",
                "intervention_type": "DRUG",
                "intervention_name": "Drug A",
            }
        ],
    )
    _gzip_csv(
        clinical / "endpoint_candidates.csv.gz",
        [
            "nct_id",
            "endpoint_candidate_id",
            "parent_candidate_id",
            "target_domain",
            "candidate_classification",
            "genuine_endpoint_candidate",
            "record_kind",
            "title_or_term",
            "description_or_organ_system",
            "unit_of_measure",
            "time_frame",
            "denominator_records_json",
            "value_records_json",
            "evidence_phrases_json",
            "source_page_path",
            "source_page_sha256",
            "raw_json_pointer",
        ],
        [
            {
                "nct_id": "NCT00000001",
                "endpoint_candidate_id": "qt-1",
                "parent_candidate_id": "outcome-1",
                "target_domain": "qt_qtc",
                "candidate_classification": "qt_qtc_interval_measure_candidate",
                "genuine_endpoint_candidate": "true",
                "record_kind": "outcome_measure",
                "title_or_term": "Change in QTcF",
                "description_or_organ_system": "Corrected QT interval",
                "unit_of_measure": "ms",
                "time_frame": "Day 1",
                "denominator_records_json": json.dumps([{"group_id": "G1", "value": "20"}]),
                "value_records_json": json.dumps([{"group_id": "G1", "value": "4.2"}]),
                "evidence_phrases_json": json.dumps([{"phrase": "QTcF"}]),
                "source_page_path": "record.json",
                "source_page_sha256": "a" * 64,
                "raw_json_pointer": "/resultsSection/outcomeMeasuresModule/0",
            }
        ],
    )
    output = tmp_path / "clinical_links"
    build_herg_clinical_links(
        hierarchy / "structure_consensus_binary.parquet",
        chembl,
        clinical,
        output,
    )
    return output


def test_candidate_states_and_clinical_evidence_never_promote(tmp_path: Path) -> None:
    hierarchy = _build_hierarchy(tmp_path)
    clinical = _build_clinical_links(tmp_path, hierarchy)
    output = tmp_path / "tiers"
    manifest = build_herg_evidence_tiers(hierarchy, clinical, output)

    candidates = {
        row["standardized_smiles"]: row for row in pq.read_table(output / T1_CANDIDATE_OUTPUT).to_pylist()
    }
    assert candidates["CCO"]["candidate_state"] == "concordant_review_candidate"
    assert candidates["CCO"]["concordant_binary_label"] == 1
    assert candidates["CCN"]["candidate_state"] == "discordant_review_candidate"
    assert candidates["CCN"]["concordant_binary_label"] is None
    assert all(row["formal_t1_assigned"] is False for row in candidates.values())
    assert all(row["model_label_admitted"] is False for row in candidates.values())

    structures = {
        row["standardized_smiles"]: row for row in pq.read_table(output / STRUCTURE_TIER_OUTPUT).to_pylist()
    }
    assert structures["CCO"]["clinical_development_annotation"] is True
    assert structures["CCO"]["exact_posted_qt_candidate_count"] == 1
    assert structures["CCO"]["exact_posted_qt_distinct_nct_count"] == 1
    assert structures["CCO"]["clinical_cardiac_candidate_evidence"] is True
    assert structures["CCCC"]["cross_lineage_t1_candidate_state"] == "no_cross_lineage_candidate"
    assert {row["formal_highest_assigned_tier"] for row in structures.values()} == {"T0_reported"}
    assert not any(row["formal_t2_assigned"] or row["formal_t3_assigned"] for row in structures.values())
    assert manifest["counts"]["cross_lineage_T1_candidates"] == 2
    assert manifest["counts"]["formal_T1_assignments"] == 0
    assert validate_herg_evidence_tiers(output)["manifest_sha256"] == manifest["manifest_sha256"]
    assert main(["--output-root", str(output), "--validate-only"]) == 0


def test_determinism_nonempty_output_and_manifest_tamper_fail_closed(tmp_path: Path) -> None:
    hierarchy = _build_hierarchy(tmp_path)
    clinical = _build_clinical_links(tmp_path, hierarchy)
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_manifest = build_herg_evidence_tiers(hierarchy, clinical, first)
    second_manifest = build_herg_evidence_tiers(hierarchy, clinical, second)
    assert first_manifest == second_manifest
    for name in (MANIFEST_NAME, STRUCTURE_TIER_OUTPUT, T1_CANDIDATE_OUTPUT):
        assert (first / name).read_bytes() == (second / name).read_bytes()

    with pytest.raises(HergTierIntegrationError, match="absent or empty"):
        build_herg_evidence_tiers(hierarchy, clinical, first)

    manifest_path = first / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["counts"]["formal_T2_assignments"] = 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(HergTierIntegrationError, match="manifest digest mismatch"):
        validate_herg_evidence_tiers(first)
