from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from menin_discovery.platform_herg_hierarchy import (
    HergHierarchyError,
    build_herg_hierarchy,
    main,
    validate_herg_hierarchy,
)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> Path:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _base_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    structures = _write_csv(
        tmp_path / "structures.csv",
        ["CID", "ConnectivitySMILES"],
        [
            {"CID": "1", "ConnectivitySMILES": "CC"},
            {"CID": "2", "ConnectivitySMILES": "CC"},
            {"CID": "3", "ConnectivitySMILES": "CCC"},
            {"CID": "4", "ConnectivitySMILES": "CCCC"},
        ],
    )
    outcomes = _write_csv(
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
                "Activity Outcome": "Active",
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
            {
                "AID": 720551,
                "SID": 14,
                "CID": 3,
                "Activity Outcome": "Inactive",
                "Target GeneID": 3757,
                "Assay Name": "wild-type qHTS",
                "Assay Type": "Confirmatory",
            },
            {
                "AID": 720551,
                "SID": 15,
                "CID": 4,
                "Activity Outcome": "Inconclusive",
                "Target GeneID": 3757,
                "Assay Name": "wild-type qHTS",
                "Assay Type": "Confirmatory",
            },
        ],
    )
    quantitative = _write_csv(
        tmp_path / "quantitative.csv",
        ["InChl Key", "SMILES", "Source", "pIC50", "USED_AS"],
        [
            {"InChl Key": "", "SMILES": "CC", "Source": "fixture", "pIC50": 4.0, "USED_AS": "Train"},
            {"InChl Key": "", "SMILES": "CCO", "Source": "fixture", "pIC50": 5.0, "USED_AS": "Train"},
            {
                "InChl Key": "",
                "SMILES": "CCN",
                "Source": "fixture",
                "pIC50": 6.0 - math.log10(30.0),
                "USED_AS": "Validation",
            },
            {"InChl Key": "", "SMILES": "CCCl", "Source": "fixture", "pIC50": 4.8, "USED_AS": "External"},
        ],
    )
    return outcomes, structures, quantitative


def test_conflicts_dedup_threshold_gap_and_secondary_only_provenance(tmp_path: Path) -> None:
    outcomes, structures, quantitative = _base_inputs(tmp_path)
    hergai = _write_csv(
        tmp_path / "hergai.csv",
        ["SID", "partition", "activity", "potency", "smiles"],
        [
            {"SID": 100, "partition": "Training", "activity": "Inactive", "potency": 9.3, "smiles": "CC"},
            {"SID": 101, "partition": "Training", "activity": "Active", "potency": 9.3, "smiles": "CO"},
        ],
    )
    output = tmp_path / "out"
    manifest = build_herg_hierarchy(
        pubchem_outcomes_path=outcomes,
        pubchem_structures_path=structures,
        quantitative_pic50_paths=[quantitative],
        hergai_paths=[hergai],
        output_root=output,
    )

    observations = pq.read_table(output / "observation_ledger.parquet").to_pandas()
    binary = pq.read_table(output / "structure_consensus_binary.parquet").to_pandas()
    hierarchy = pq.read_table(output / "hierarchy_annotations.parquet").to_pandas()
    quantitative_view = pq.read_table(output / "quantitative_pic50.parquet").to_pandas()

    # Two PubChem CIDs standardize to ethane and collapse to one active label.
    ethane = observations[
        (observations["source_family"] == "pubchem_aid720551") & (observations["source_cid"] == "1")
    ].iloc[0]
    assert len(binary[binary["structure_id"] == ethane["structure_id"]]) == 1
    assert (
        int(binary.loc[binary["structure_id"] == ethane["structure_id"], "herg_blocker_label"].iloc[0]) == 1
    )

    # Conflicting source-grade outcomes remain in the ledger but are excluded from one-label output.
    propane = observations[
        (observations["source_family"] == "pubchem_aid720551") & (observations["source_cid"] == "3")
    ].iloc[0]
    assert propane["structure_id"] not in set(binary["structure_id"])
    propane_hierarchy = hierarchy[hierarchy["structure_id"] == propane["structure_id"]].iloc[0]
    assert bool(propane_hierarchy["primary_conflict"])
    assert propane_hierarchy["consensus_status"] == "primary_conflict_excluded"

    # Reported pIC50 thresholds are inclusive and the 10-30 uM interval stays unlabeled.
    by_smiles = quantitative_view.set_index("standardized_smiles")
    assert int(by_smiles.loc["CCO", "derived_binary_label"]) == 1
    assert int(by_smiles.loc["CCN", "derived_binary_label"]) == 0
    assert by_smiles.loc["CCCl", "derived_binary_label"] is None or math.isnan(
        by_smiles.loc["CCCl", "derived_binary_label"]
    )

    # HERGAI potency is preserved only in auxiliary provenance and never becomes pIC50.
    hergai_rows = observations[observations["source_family"] == "hergai_secondary"]
    assert hergai_rows["pic50_value"].isna().all()
    assert not hergai_rows["t1_candidate"].any()
    assert "reported_potency_uninterpreted" in hergai_rows.iloc[0]["native_aux_json"]
    ethane_hierarchy = hierarchy[hierarchy["structure_id"] == ethane["structure_id"]].iloc[0]
    assert bool(ethane_hierarchy["secondary_conflict"])
    assert manifest["scientific_contract"]["binary_to_pic50_conversion_performed"] is False


def test_chembl_exact_functional_nm_um_only_and_censor_preservation(tmp_path: Path) -> None:
    outcomes, structures, quantitative = _base_inputs(tmp_path)
    chembl_path = tmp_path / "chembl.parquet"
    rows = [
        (1, "IC50", "=", 10_000.0, "nM", "F", "CCOCC", 1, 0, None, 9, None),
        (2, "IC50", "=", 30.0, "uM", "F", "CCNCC", 1, 0, None, 9, None),
        (3, "IC50", ">", 30_000.0, "nM", "F", "CCCO", 1, 0, None, 9, None),
        (4, "IC50", "=", 100.0, "nM", "B", "CCCCO", 1, 0, None, 9, None),
        (5, "Ki", "=", 100.0, "nM", "F", "CCCCN", 1, 0, None, 9, None),
        (6, "Inhibition", "=", 75.0, "%", "F", "CCCCCCl", 1, 0, None, 9, None),
        (7, "IC50", "=", 100.0, "nM", "F", "CCCCF", 1, 0, None, 9, 123),
    ]
    table = pa.table(
        {
            "activity_id": [row[0] for row in rows],
            "standard_type": [row[1] for row in rows],
            "standard_relation": [row[2] for row in rows],
            "standard_value": [row[3] for row in rows],
            "standard_units": [row[4] for row in rows],
            "assay_type": [row[5] for row in rows],
            "canonical_smiles": [row[6] for row in rows],
            "target_chembl_id": ["CHEMBL240"] * len(rows),
            "standard_flag": [row[7] for row in rows],
            "potential_duplicate": [row[8] for row in rows],
            "data_validity_comment": [row[9] for row in rows],
            "confidence_score": [row[10] for row in rows],
            "variant_id": [row[11] for row in rows],
            "assay_chembl_id": [f"CHEMBL-A{row[0]}" for row in rows],
            "assay_description": ["fixture"] * len(rows),
        }
    )
    pq.write_table(table, chembl_path)
    output = tmp_path / "out"
    build_herg_hierarchy(
        pubchem_outcomes_path=outcomes,
        pubchem_structures_path=structures,
        quantitative_pic50_paths=[quantitative],
        chembl_parquet_paths=[chembl_path],
        output_root=output,
    )

    observations = pq.read_table(output / "observation_ledger.parquet").to_pandas()
    binary = pq.read_table(output / "structure_consensus_binary.parquet").to_pandas()
    chembl = observations[observations["source_family"] == "chembl_herg_specialized_view"].set_index(
        "source_record_id"
    )
    assert chembl.loc["ACTIVITY:1", "pic50_value"] == pytest.approx(5.0)
    assert int(chembl.loc["ACTIVITY:1", "derived_binary_label"]) == 1
    assert chembl.loc["ACTIVITY:2", "pic50_value"] == pytest.approx(6.0 - math.log10(30.0))
    assert int(chembl.loc["ACTIVITY:2", "derived_binary_label"]) == 0

    # Censoring is retained in the native ledger without an invented point pIC50.
    assert chembl.loc["ACTIVITY:3", "native_relation"] == ">"
    assert math.isnan(chembl.loc["ACTIVITY:3", "pic50_value"])
    for record_id in ("ACTIVITY:4", "ACTIVITY:5", "ACTIVITY:6", "ACTIVITY:7"):
        assert math.isnan(chembl.loc[record_id, "pic50_value"])
        assert not bool(chembl.loc[record_id, "t1_candidate"])
    assert set(chembl["structure_id"].dropna()).isdisjoint(set(binary["structure_id"]))


def test_outputs_are_byte_deterministic_and_cli_validates(tmp_path: Path) -> None:
    outcomes, structures, quantitative = _base_inputs(tmp_path)
    outputs = [tmp_path / "first", tmp_path / "second"]
    manifests = []
    for output in outputs:
        manifests.append(
            build_herg_hierarchy(
                pubchem_outcomes_path=outcomes,
                pubchem_structures_path=structures,
                quantitative_pic50_paths=[quantitative],
                output_root=output,
            )
        )
    assert manifests[0] == manifests[1]
    for name in (
        "manifest.json",
        "observation_ledger.parquet",
        "structure_consensus_binary.parquet",
        "quantitative_pic50.parquet",
        "hierarchy_annotations.parquet",
    ):
        assert (outputs[0] / name).read_bytes() == (outputs[1] / name).read_bytes()
    assert (
        main(
            [
                "--output-root",
                str(outputs[0]),
                "--validate-only",
            ]
        )
        == 0
    )


def test_fail_closed_for_unmatched_cid_and_manifest_tamper(tmp_path: Path) -> None:
    outcomes, structures, quantitative = _base_inputs(tmp_path)
    rows = list(csv.DictReader(outcomes.open(encoding="utf-8")))
    rows[0]["CID"] = "999999"
    _write_csv(outcomes, list(rows[0]), rows)
    with pytest.raises(HergHierarchyError, match="has no supplied structure"):
        build_herg_hierarchy(
            pubchem_outcomes_path=outcomes,
            pubchem_structures_path=structures,
            quantitative_pic50_paths=[quantitative],
            output_root=tmp_path / "failed",
        )
    assert not (tmp_path / "failed").exists()

    outcomes, structures, quantitative = _base_inputs(tmp_path)
    output = tmp_path / "valid"
    build_herg_hierarchy(
        pubchem_outcomes_path=outcomes,
        pubchem_structures_path=structures,
        quantitative_pic50_paths=[quantitative],
        output_root=output,
    )
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    manifest["counts"]["observations"] += 1
    (output / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(HergHierarchyError, match="manifest digest mismatch"):
        validate_herg_hierarchy(output)
