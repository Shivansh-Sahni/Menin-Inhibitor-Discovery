from __future__ import annotations

import csv
import gzip
import json
import sqlite3
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from menin_discovery.platform_herg_trial_uplift import (
    AUDIT_OUTPUT,
    CANDIDATE_OUTPUT,
    HergTrialUpliftError,
    build_herg_trial_uplift,
    normalize_exact_name,
    normalize_punctuation_name,
    verify_herg_trial_uplift,
)
from rdkit import Chem


def _key(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(smiles)
    assert molecule is not None
    return Chem.MolToInchiKey(molecule)


def _write_gzip_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with gzip.open(path, mode="wt", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_clinical(interventions: Path, studies: Path) -> None:
    intervention_rows = [
        ("i-exact-parent", "NCT00000001", "DRUG", "Drug Alpha"),
        ("i-punctuation", "NCT00000002", "DRUG", "Drug-B"),
        ("i-clean", "NCT00000003", "DRUG", "50 mg Drug C tablet"),
        ("i-code", "NCT00000004", "DRUG", "10 mg Product (ABC-123) tablet"),
        ("i-components", "NCT00000005", "DRUG", "Drug Alpha + Drug B"),
        ("i-ambiguous", "NCT00000006", "DRUG", "Shared Name"),
        ("i-placebo", "NCT00000007", "DRUG", "Placebo"),
        ("i-excipient", "NCT00000008", "DRUG", "0.9% Normal Saline"),
        ("i-biological", "NCT00000009", "BIOLOGICAL", "Drug Alpha"),
        ("i-unresolved", "NCT00000010", "DRUG", "Never Invented Drug"),
    ]
    _write_gzip_csv(
        interventions,
        ["intervention_candidate_id", "nct_id", "intervention_type", "intervention_name"],
        [
            {
                "intervention_candidate_id": candidate_id,
                "nct_id": nct_id,
                "intervention_type": intervention_type,
                "intervention_name": name,
            }
            for candidate_id, nct_id, intervention_type, name in intervention_rows
        ],
    )
    _write_gzip_csv(
        studies,
        ["nct_id", "has_results_reported"],
        [
            {"nct_id": f"NCT{index:08d}", "has_results_reported": str(index % 2 == 0).lower()}
            for index in range(1, 11)
        ],
    )


def _write_chembl(path: Path) -> None:
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE compound_structures (
          molregno INTEGER PRIMARY KEY, canonical_smiles TEXT, standard_inchi_key TEXT
        );
        CREATE TABLE molecule_dictionary (
          molregno INTEGER PRIMARY KEY, chembl_id TEXT, pref_name TEXT
        );
        CREATE TABLE molecule_synonyms (molregno INTEGER, synonyms TEXT);
        CREATE TABLE molecule_hierarchy (
          molregno INTEGER PRIMARY KEY, parent_molregno INTEGER, active_molregno INTEGER
        );
        """
    )
    structures = {
        1: "CCO",
        2: "CCO.Cl",
        3: "CCN",
        4: "CCC",
        5: "c1ccccc1",
        6: "CCCO",
        7: "CCCC",
    }
    names = {
        1: "Drug A",
        2: "Drug Alpha Salt",
        3: "Drug B",
        4: "Drug C",
        5: "Code Product",
        6: "Shared Structure One",
        7: "Shared Structure Two",
    }
    for molregno, smiles in structures.items():
        connection.execute(
            "INSERT INTO compound_structures VALUES (?, ?, ?)",
            (molregno, smiles, _key(smiles)),
        )
        connection.execute(
            "INSERT INTO molecule_dictionary VALUES (?, ?, ?)",
            (molregno, f"CHEMBL{molregno}", names[molregno]),
        )
        parent = 1 if molregno == 2 else molregno
        connection.execute("INSERT INTO molecule_hierarchy VALUES (?, ?, ?)", (molregno, parent, parent))
    connection.executemany(
        "INSERT INTO molecule_synonyms VALUES (?, ?)",
        [
            (2, "Drug Alpha"),
            (3, "Drug B"),
            (4, "Drug C"),
            (5, "ABC-123"),
            (6, "Shared Name"),
            (7, "Shared Name"),
        ],
    )
    connection.commit()
    connection.close()


def _write_hierarchy(path: Path) -> None:
    schema = pa.schema(
        [
            pa.field("structure_id", pa.string()),
            pa.field("standard_inchi_key", pa.string()),
        ]
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"structure_id": "h-ethanol", "standard_inchi_key": _key("CCO")},
                {"structure_id": "h-ethylamine", "standard_inchi_key": _key("CCN")},
            ],
            schema=schema,
        ),
        path,
    )


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    interventions = tmp_path / "interventions.csv.gz"
    studies = tmp_path / "studies.csv.gz"
    chembl = tmp_path / "chembl.db"
    hierarchy = tmp_path / "hierarchy.parquet"
    output = tmp_path / "output"
    _write_clinical(interventions, studies)
    _write_chembl(chembl)
    _write_hierarchy(hierarchy)
    return interventions, studies, chembl, hierarchy, output


def test_normalizers_are_auditable_not_fuzzy() -> None:
    assert normalize_exact_name("  Drug   A ") == "drug a"
    assert normalize_punctuation_name("ABC-123") == "abc 123"
    assert normalize_punctuation_name("Drug-B") == normalize_punctuation_name("Drug B")
    assert normalize_punctuation_name("Drug A") != normalize_punctuation_name("Drug X")


def test_build_preserves_tiers_parent_structures_ambiguity_and_zero_labels(tmp_path: Path) -> None:
    interventions, studies, chembl, hierarchy, output = _fixture(tmp_path)
    manifest = build_herg_trial_uplift(interventions, studies, chembl, hierarchy, output)

    candidates = {
        row["intervention_candidate_id"]: row for row in pq.read_table(output / CANDIDATE_OUTPUT).to_pylist()
    }
    assert set(candidates) == {
        "i-exact-parent",
        "i-punctuation",
        "i-clean",
        "i-code",
        "i-components",
        "i-ambiguous",
    }
    exact = candidates["i-exact-parent"]
    assert exact["first_resolved_rule_name"] == "exact_parent_standardized"
    assert exact["automatic_identity_link"] is True
    assert json.loads(exact["candidate_parent_structure_keys_json"]) == [_key("CCO")]
    assert exact["has_any_local_reported_herg_evidence"] is True

    assert candidates["i-punctuation"]["first_resolved_rule_tier"] == 1
    assert candidates["i-clean"]["first_resolved_rule_tier"] == 2
    assert candidates["i-code"]["first_resolved_rule_tier"] == 2
    component = candidates["i-components"]
    assert component["first_resolved_rule_tier"] == 3
    assert component["is_component_set_candidate"] is True
    assert component["candidate_parent_structure_count"] == 2

    ambiguous = candidates["i-ambiguous"]
    assert ambiguous["ambiguity_preserved"] is True
    assert ambiguous["automatic_identity_link"] is False
    assert ambiguous["candidate_parent_structure_count"] == 2
    for row in candidates.values():
        assert row["herg_label_admitted"] is False
        assert row["clinical_label_admitted"] is False
        assert row["model_label_admitted"] is False

    audit = {
        row["intervention_candidate_id"]: row for row in pq.read_table(output / AUDIT_OUTPUT).to_pylist()
    }
    assert audit["i-placebo"]["exclusion_reason"] == "placebo_vehicle_or_excipient"
    assert audit["i-excipient"]["exclusion_reason"] == "placebo_vehicle_or_excipient"
    assert audit["i-biological"]["exclusion_reason"] == "non_drug_intervention_type"
    assert audit["i-unresolved"]["audit_state"] == "unresolved"

    headline = manifest["headline"]
    assert headline["practical_over_1000_structure_threshold_met"] is False
    assert "local hERG intersection" in headline["disclosure"]
    verified = verify_herg_trial_uplift(output)
    assert verified["verification_status"] == "pass"
    assert verified["herg_labels_admitted"] == 0


def test_nonempty_output_and_tampering_fail_closed(tmp_path: Path) -> None:
    interventions, studies, chembl, hierarchy, output = _fixture(tmp_path)
    build_herg_trial_uplift(interventions, studies, chembl, hierarchy, output)
    with pytest.raises(HergTrialUpliftError, match="absent or empty"):
        build_herg_trial_uplift(interventions, studies, chembl, hierarchy, output)

    manifest_path = output / "herg_trial_uplift_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["model_labels_admitted"] = 1
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(HergTrialUpliftError, match="manifest internal"):
        verify_herg_trial_uplift(output)
