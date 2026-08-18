from __future__ import annotations

import csv
import gzip
import json
import sqlite3
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from menin_discovery.platform_herg_clinical_links import (
    LINK_AUDIT_OUTPUT,
    STRUCTURE_OUTPUT,
    T2_OUTPUT,
    T3_OUTPUT,
    HergClinicalLinkError,
    build_herg_clinical_links,
    normalize_exact_name,
    verify_herg_clinical_links,
)

KEY_A = "UHOVQNZJYSORNB-UHFFFAOYSA-N"
KEY_B = "QUSNBJAOOMFDIB-UHFFFAOYSA-N"
KEY_C = "LFQSCWFLJHTTHZ-UHFFFAOYSA-N"


def _write_consensus(path: Path) -> None:
    schema = pa.schema(
        [
            pa.field("molecule_id", pa.string()),
            pa.field("standard_inchi_key", pa.string()),
            pa.field("canonical_smiles", pa.string()),
            pa.field("preferred_name", pa.string()),
            pa.field("aliases_json", pa.string()),
        ]
    )
    rows = [
        {
            "molecule_id": "mol-a",
            "standard_inchi_key": KEY_A,
            "canonical_smiles": "CCO",
            "preferred_name": "Drug A",
            "aliases_json": json.dumps(["Shared Name"]),
        },
        {
            "molecule_id": "mol-b",
            "standard_inchi_key": KEY_B,
            "canonical_smiles": "CCN",
            "preferred_name": "Drug B",
            "aliases_json": json.dumps(["Shared Name"]),
        },
        {
            "molecule_id": "mol-c",
            "standard_inchi_key": KEY_C,
            "canonical_smiles": "CCOCC",
            "preferred_name": "Drug C",
            "aliases_json": "[]",
        },
    ]
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)


def _write_chembl(path: Path) -> None:
    connection = sqlite3.connect(path)
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
    connection.execute("INSERT INTO compound_structures VALUES (1, ?, 'CCO')", (KEY_A,))
    connection.execute("INSERT INTO molecule_dictionary VALUES (1, 'CHEMBL1', 'Drug A', 4.0, 2001, 1, 1, 0)")
    connection.execute("INSERT INTO molecule_synonyms VALUES (1, 'Drug Alpha')")
    connection.commit()
    connection.close()


def _write_gzip_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with gzip.open(path, mode="wt", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _clinical_fixture(root: Path) -> None:
    root.mkdir()
    ncts = ["NCT00000001", "NCT00000002", "NCT00000003", "NCT00000004", "NCT00000005"]
    _write_gzip_csv(
        root / "studies.csv.gz",
        ["nct_id", "has_results_reported"],
        [
            {"nct_id": nct, "has_results_reported": "false" if nct == "NCT00000002" else "true"}
            for nct in ncts
        ],
    )
    _write_gzip_csv(
        root / "interventions.csv.gz",
        ["nct_id", "intervention_candidate_id", "intervention_type", "intervention_name"],
        [
            {
                "nct_id": "NCT00000001",
                "intervention_candidate_id": "i-ambiguous",
                "intervention_type": "DRUG",
                "intervention_name": "Shared Name",
            },
            {
                "nct_id": "NCT00000002",
                "intervention_candidate_id": "i-no-results",
                "intervention_type": "DRUG",
                "intervention_name": "Drug A",
            },
            {
                "nct_id": "NCT00000003",
                "intervention_candidate_id": "i-combination",
                "intervention_type": "DRUG",
                "intervention_name": "Drug A / Drug C",
            },
            {
                "nct_id": "NCT00000004",
                "intervention_candidate_id": "i-posted-qt",
                "intervention_type": "DRUG",
                "intervention_name": "  DRUG   A  ",
            },
            {
                "nct_id": "NCT00000005",
                "intervention_candidate_id": "i-nonnumeric",
                "intervention_type": "DRUG",
                "intervention_name": "Drug A",
            },
        ],
    )
    endpoint_fields = [
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
    ]
    endpoint_rows: list[dict[str, object]] = []
    for index, nct in enumerate(ncts, start=1):
        values = [{"group_id": "OG000", "value": "not reported"}]
        if nct != "NCT00000005":
            values = [{"group_id": "OG000", "value": "4.2"}]
        endpoint_rows.append(
            {
                "nct_id": nct,
                "endpoint_candidate_id": f"e-{index}",
                "parent_candidate_id": f"o-{index}",
                "target_domain": "qt_qtc",
                "candidate_classification": "qt_qtc_interval_measure_candidate",
                "genuine_endpoint_candidate": "true",
                "record_kind": "outcome_measure",
                "title_or_term": "Change in QTcF",
                "description_or_organ_system": "Corrected QT interval",
                "unit_of_measure": "ms",
                "time_frame": "Day 1",
                "denominator_records_json": json.dumps([{"group_id": "OG000", "value": "10"}]),
                "value_records_json": json.dumps(values),
                "evidence_phrases_json": json.dumps([{"phrase": "QTcF"}]),
                "source_page_path": "page.json",
                "source_page_sha256": "a" * 64,
                "raw_json_pointer": "/resultsSection/outcomeMeasuresModule/0",
            }
        )
    _write_gzip_csv(root / "endpoint_candidates.csv.gz", endpoint_fields, endpoint_rows)


def test_name_normalization_is_narrow() -> None:
    assert normalize_exact_name("  Drug   A ") == "drug a"
    assert normalize_exact_name("Drug-A") != normalize_exact_name("Drug A")
    assert normalize_exact_name("Drug A hydrochloride") != normalize_exact_name("Drug A")


def test_fail_closed_hierarchy_links_only_posted_numeric_single_drug(tmp_path: Path) -> None:
    consensus = tmp_path / "consensus.parquet"
    chembl = tmp_path / "chembl.db"
    clinical = tmp_path / "clinical"
    output = tmp_path / "output"
    _write_consensus(consensus)
    _write_chembl(chembl)
    _clinical_fixture(clinical)

    manifest = build_herg_clinical_links(consensus, chembl, clinical, output)
    assert manifest["row_counts"][T2_OUTPUT] == 1
    assert manifest["row_counts"][T3_OUTPUT] == 1

    t3 = pq.read_table(output / T3_OUTPUT).to_pylist()
    assert t3[0]["nct_id"] == "NCT00000004"
    assert t3[0]["molecule_id"] == "mol-a"
    assert t3[0]["actual_qt_result_present"] is True
    assert t3[0]["clinical_herg_label_admitted"] is False
    assert t3[0]["model_label_admitted"] is False
    assert "unknown_not_negative" in t3[0]["absence_semantics"]

    audit = {row["source_record_id"]: row for row in pq.read_table(output / LINK_AUDIT_OUTPUT).to_pylist()}
    assert audit["i-ambiguous"]["link_state"] == "rejected_ambiguous_exact_name"
    assert audit["i-combination"]["link_state"] == "rejected_combination_or_non_drug_name"
    assert audit["i-posted-qt"]["link_is_exact_and_unique"] is True

    development = {row["molecule_id"]: row for row in pq.read_table(output / STRUCTURE_OUTPUT).to_pylist()}
    assert development["mol-a"]["chembl_max_phase"] == 4.0
    assert development["mol-a"]["clinical_development_annotation"] is True
    assert development["mol-a"]["clinical_cardiac_label_admitted"] is False
    assert development["mol-b"]["clinical_development_annotation"] is False

    assert verify_herg_clinical_links(output)["verification_status"] == "pass"


def test_tampering_and_nonempty_output_fail_closed(tmp_path: Path) -> None:
    consensus = tmp_path / "consensus.parquet"
    chembl = tmp_path / "chembl.db"
    clinical = tmp_path / "clinical"
    output = tmp_path / "output"
    _write_consensus(consensus)
    _write_chembl(chembl)
    _clinical_fixture(clinical)
    build_herg_clinical_links(consensus, chembl, clinical, output)

    with pytest.raises(HergClinicalLinkError, match="absent or empty"):
        build_herg_clinical_links(consensus, chembl, clinical, output)

    manifest_path = output / "herg_clinical_links_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["model_labels_admitted"] = 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(HergClinicalLinkError, match="manifest internal"):
        verify_herg_clinical_links(output)
