from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from menin_discovery.chemistry import standardize_smiles

from menin_edit.data import (
    build_multi_endpoint_mmp_evidence,
    load_historical_lab_workbook,
    parse_qualified_value,
)


def _write_workbook(path: Path) -> None:
    openpyxl = pytest.importorskip("openpyxl")
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "SMILES"
    sheet.append(
        [
            "Compound",
            "Kekule Canonical SMILES",
            "MW",
            "TPSA",
            "cLogP (RDKit Crippen)",
            "pKa1 (most basic)",
            "pKa2 (2nd most basic)",
            "Binding IC50 (nM)",
            "Slope",
            "MV4;11 IC50 (nM)",
            "MOLM13 IC50 (nM)",
            "HL60 IC50 (nM)",
            "Rat PK: Dose (IV/PO) mg/kg",
            "Rat PK: AUC0-t (PO)",
            "Rat PK: %F",
            "hERG IC50 (µM)",
            "hERG % inhibition",
        ]
    )
    # Two source registrations have the same exact structure but incompatible
    # binding labels.  The loader must aggregate the structure and flag it.
    sheet.append(
        [
            "SECRET-A",
            "CCOc1ccccc1",
            122.17,
            9.23,
            2.0,
            8.5,
            6.1,
            "10",
            "1.1",
            "20",
            "40",
            ">1000",
            "2/5",
            "300",
            "20",
            ">30",
            "5%@10uM and 15%@30uM",
        ]
    )
    sheet.append(
        [
            "SECRET-A-DUP",
            "CCOc1ccccc1",
            122.17,
            9.23,
            2.0,
            8.5,
            6.1,
            "100",
            "1.2",
            "200",
            "400",
            ">1000",
            "2/5 | 1/3",
            "600",
            "10",
            "8",
            "55%@10uM and 85%@30uM",
        ]
    )
    sheet.append(
        [
            "SECRET-B",
            "CCNc1ccccc1",
            121.18,
            12.03,
            1.5,
            8.0,
            5.8,
            "5",
            "1.0",
            "10",
            "20",
            ">1000",
            "2/5",
            "500",
            "25",
            "<0.37",
            "80%@10uM and 95%@30uM",
        ]
    )
    provenance = workbook.create_sheet("Provenance")
    provenance.append(["Compound", "Canonical key", "Parameter", "Source slide(s)", "Raw value(s)"])
    provenance.append(["SECRET-A", "A", "hERG IC50 (µM)", "Deck s1, Deck s2", "4.0 | >30"])
    workbook.save(path)


def test_parse_qualified_value_preserves_censoring():
    greater = parse_qualified_value(" > 30 ")
    less = parse_qualified_value("≤0.37")

    assert greater is not None and greater.relation == ">" and greater.value == 30
    assert greater.is_censored is True
    assert less is not None and less.relation == "<=" and less.value == pytest.approx(0.37)
    assert parse_qualified_value("NT") is None
    assert parse_qualified_value("2/5") is None


def test_historical_loader_is_pseudonymous_long_and_censor_aware(tmp_path):
    workbook_path = tmp_path / "private.xlsx"
    _write_workbook(workbook_path)
    result = load_historical_lab_workbook(
        workbook_path,
        pseudonymization_key="a-long-test-secret-key",
    )

    assert len(result.compounds) == 2
    assert result.summary["source_rows"] == 3
    assert result.summary["standardized_compounds"] == 2
    assert result.summary["raw_identifiers_included"] is False
    assert "source_compound_id" not in result.compounds.columns
    assert "source_compound_ids" not in result.compounds.columns

    serialized = result.compounds.to_csv(index=False) + result.observations.to_csv(index=False)
    assert "SECRET-A" not in serialized
    assert "SECRET-B" not in serialized
    assert result.compounds["compound_id"].str.startswith("ICMP-").all()

    duplicate = result.compounds.loc[result.compounds["duplicate_structure"]].iloc[0]
    assert duplicate["source_record_count"] == 2
    assert bool(duplicate["label_conflict"]) is True
    assert bool(duplicate["conflict_flag"]) is True

    herg = result.observations[result.observations["endpoint"].eq("herg_pIC50")]
    assert set(herg["relation"]) == {"=", "<"}
    safer_bound = herg[herg["relation"].eq("<")].iloc[0]
    assert safer_bound["value"] == pytest.approx(0.37)
    assert np.isfinite(safer_bound["model_lower"])
    assert pd.isna(safer_bound["model_upper"])

    percent = result.observations[result.observations["endpoint"].eq("herg_percent_inhibition")]
    assert set(percent["test_concentration_um"]) == {10.0, 30.0}
    assert {"menin_biochemical_pIC50", "rat_po_auc0_t_log10_ng_h_ml"}.issubset(
        set(result.observations["endpoint"])
    )


def _evidence_tables() -> tuple[pd.DataFrame, pd.DataFrame, str]:
    molecules = {
        "A": "Fc1ccccc1",
        "B": "Clc1ccccc1",
        "LOCKED": "Brc1ccccc1",
    }
    compound_rows = []
    observations = []
    for name, smiles in molecules.items():
        standardized = standardize_smiles(smiles, require_rdkit=True)
        compound_rows.append(
            {
                "structure_id": standardized.structure_id,
                "smiles": standardized.standardized_smiles,
            }
        )
        role = "locked_external" if name == "LOCKED" else "development"
        potency = {"A": 7.0, "B": 8.0, "LOCKED": 9.0}[name]
        herg = {"A": 5.0, "B": 4.0, "LOCKED": 3.0}[name]
        for endpoint, value in (
            ("menin_biochemical_pIC50", potency),
            ("herg_pIC50", herg),
        ):
            observations.append(
                {
                    "structure_id": standardized.structure_id,
                    "endpoint": endpoint,
                    "model_value": value,
                    "model_lower": value,
                    "model_upper": value,
                    "is_exact": True,
                    "label_conflict": False,
                    "relation": "=",
                    "split_role": role,
                    "source_scope": "private",
                    "assay_id": f"ASSAY-{endpoint}",
                }
            )
    locked_id = compound_rows[2]["structure_id"]
    return pd.DataFrame(compound_rows), pd.DataFrame(observations), locked_id


def test_mmp_evidence_is_vector_valued_bidirectional_and_never_uses_locked_rows():
    compounds, observations, locked_id = _evidence_tables()
    pairs = build_multi_endpoint_mmp_evidence(
        compounds,
        observations,
        min_core_heavy_atoms=5,
        max_variable_heavy_atoms=2,
    )

    assert not pairs.empty
    assert set(pairs["endpoint"]) == {"menin_biochemical_pIC50", "herg_pIC50"}
    assert set(pairs["split_role"]) == {"development"}
    assert locked_id not in set(pairs["structure_id_a"])
    assert locked_id not in set(pairs["structure_id_b"])
    assert len(pairs) == 4  # two endpoints, both edit directions

    menin = pairs[pairs["endpoint"].eq("menin_biochemical_pIC50")]
    assert sorted(menin["delta"].round(6).tolist()) == [-1.0, 1.0]
    for row in pairs.itertuples(index=False):
        reverse = pairs[
            pairs["structure_id_a"].eq(row.structure_id_b)
            & pairs["structure_id_b"].eq(row.structure_id_a)
            & pairs["endpoint"].eq(row.endpoint)
        ].iloc[0]
        assert reverse["delta"] == pytest.approx(-row.delta)


def test_mmp_builder_rejects_evaluation_roles():
    compounds, observations, _locked_id = _evidence_tables()
    with pytest.raises(ValueError, match="only 'train' and 'development'"):
        build_multi_endpoint_mmp_evidence(
            compounds,
            observations,
            allowed_split_roles=("development", "locked_external"),
            min_core_heavy_atoms=5,
        )
