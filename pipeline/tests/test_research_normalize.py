from pathlib import Path

import pandas as pd
import pytest
from menin_discovery.research_normalize import (
    normalize_internal_workbook,
    normalize_research_data,
    normalize_sun_herg_workbook,
    parse_concentration_response_panel,
    parse_qualified_value,
)


def _internal_sheets():
    wide = pd.DataFrame(
        {
            "Compound": ["C-1", "C-1-ALIAS", "C-2"],
            "Kekule Canonical SMILES": ["CCO", "OCC", "CCN"],
            "MW": [46.07, 46.07, 45.08],
        }
    )
    rows = []

    def add(compound, parameter, sources, values):
        rows.append(
            {
                "Compound": compound,
                "Canonical key": compound.replace("-", ""),
                "Parameter": parameter,
                "Source slide(s)": sources,
                "Raw value(s)": values,
            }
        )

    # Two explicitly matched rat studies.  Every observation remains separate.
    add("C-1", "Rat PK: Dose (IV/PO) mg/kg", "Slide A, Slide B", "2/5 | 1/3")
    add("C-1", "Rat PK: AUC0-inf (IV)", "Slide A, Slide B", "1000 | 500")
    add("C-1", "Rat PK: AUC0-inf (PO)", "Slide A, Slide B", "500 | 300")
    add("C-1", "Rat PK: CL (IV) mL/kg/min", "Slide A, Slide B", "33.3333 | 33.3333")
    add("C-1", "Rat PK: %F", "Slide A, Slide B", "20 | 20")
    add("C-1", "Rat PK: Cmax (PO) ng/mL", "Slide A, Slide B", "110 | 120")

    # A many-values/one-source case is irreducibly ambiguous and must not be joined.
    add("C-2", "Rat PK: Dose (IV/PO) mg/kg", "Slide X", "2/5 | 1/3")
    add("C-2", "Rat PK: AUC0-inf (IV)", "Slide X", "1000 | 500")

    # hERG one-sided values, concentration panels, and sparse physical-process fields.
    add("C-1", "hERG IC50 (µM)", "Slide H", "< 0.37")
    add("C-1", "hERG % inhibition", "Slide H", "15%@10uM and 45%@30uM")
    add("C-1-ALIAS", "hERG IC50 (µM)", "Slide Alias", "20")
    add("C-1", "MetStab T1/2 (min): Human", "Slide M", "∞")
    add("C-1", "Hepatic Extraction Eh: Human", "Slide M", "0.34")
    add("C-1", "PPB %Bound: Human", "Slide M", "91.4")
    add("C-1", "PPB %Unbound: Human", "Slide M", "8.6")
    add("C-1", "Plasma Stability T1/2 (min): Human", "Slide M", "475")

    provenance = pd.DataFrame(rows)
    pka = pd.DataFrame(
        {
            "Compound Name": ["C-1", "C-2"],
            "All basic pKaH (desc)": ["9.7, 6.5", "8.1"],
            "Acidic pKa < 13": ["11.9", "NA"],
        }
    )
    return {"SMILES": wide, "Provenance": provenance, "pKa_detail": pka}


def _sun_sheets():
    large_alkane = "C" * 50
    return {
        "Classification": pd.DataFrame(
            {
                "Smiles": ["CCO", "OCC", "CCN", large_alkane, "CCC"],
                "hERG Class": [0, 1, 1, 1, 1],
                "IC50(nM)": [100, 20000, ">10000", ">30000", "Not"],
            }
        ),
        "Regression": pd.DataFrame(
            {
                "Smiles": ["CCO", "CCC"],
                "IC50(nM)": [2.0, 4.0],  # Stored log10(IC50 nM), despite header.
                "hERG": [2.1, 3.9],
            }
        ),
        "Validation": pd.DataFrame(
            {
                "Smiles": ["CCO", "CCCl"],
                "hERG (nM)": [100, 100000],
            }
        ),
    }


def test_safe_qualified_and_concentration_panel_parsing():
    left = parse_qualified_value("< 0.37")
    assert (left.value, left.relation, left.censoring) == (0.37, "<", "left")
    right = parse_qualified_value(">10000")
    assert (right.value, right.relation, right.censoring) == (10000.0, ">", "right")
    infinity = parse_qualified_value("∞")
    assert infinity.value is None
    assert infinity.relation == ">"
    assert infinity.status == "unbounded_right_censored"
    interval = parse_qualified_value("1.2-3.4")
    assert (interval.lower_bound, interval.upper_bound, interval.censoring) == (1.2, 3.4, "interval")
    assert parse_qualified_value("Not").relation == "not_reported"

    panel = parse_concentration_response_panel("15%@10uM and 45%@30uM")
    assert [(item.response.value, item.concentration_value) for item in panel] == [(15.0, 10.0), (45.0, 30.0)]


def test_internal_normalization_is_structure_deduplicated_and_process_centred():
    result = normalize_internal_workbook(_internal_sheets(), closure_tolerance=0.02)

    # C-1 and C-1-ALIAS are one standardized structure, with explicit alias lineage.
    assert len(result.tables["compounds"]) == 2
    compounds = result.tables["compounds"]
    assert compounds["series_id"].notna().all()
    assert compounds["scaffold"].notna().all()
    assert compounds["scaffold_method"].notna().all()
    assert set(compounds["stereochemistry_status"]) == {"not_applicable"}
    aliases = result.review_tables["compound_aliases"]
    alias_ids = aliases[aliases["source_compound_name"].isin(["C-1", "C-1-ALIAS"])]["compound_id"]
    assert alias_ids.nunique() == 1
    assert len(aliases) == 3

    # Two well-resolved evidence locations expand to separate IV/PO events.
    studies = result.tables["pk_studies"]
    resolved = studies[
        (studies["compound_id"] == alias_ids.iloc[0]) & (studies["pairing_status"] == "resolved")
    ]
    assert len(resolved) == 4
    assert set(resolved["route"]) == {"IV", "PO"}
    assert resolved["event_pair_id"].nunique() == 2

    # Study labels are never averaged: both Cmax values remain individual records.
    cmax = result.tables["measurements"].query("endpoint == 'cmax'")
    assert sorted(cmax["value"].tolist()) == [110.0, 120.0]
    assert cmax["pk_study_id"].notna().all()

    # Dose/AUC and dose-normalized AUC closure diagnostics are explicit and non-eligible.
    derived = result.tables["derived_pk_parameters"]
    assert len(derived) == 4
    assert set(derived["endpoint"]) == {"clearance", "bioavailability"}
    assert set(derived["closure_status"]) == {"pass"}
    assert not derived["model_eligible"].any()
    reported = result.tables["measurements"].query("endpoint in ['clearance', 'bioavailability']")
    assert set(reported["leakage_role"]) == {"derived_from_exposure"}
    assert not reported["model_eligible"].any()

    # Ambiguous source/value cardinality is retained and explicitly errored.
    assert "unresolved_study_pairing" in set(result.issues["code"])
    unresolved = studies[studies["pairing_status"] == "unresolved"]
    assert len(unresolved) == 4

    measurements = result.tables["measurements"]
    one_sided = measurements.query("endpoint == 'herg_ic50' and submitted_value == '< 0.37'").iloc[0]
    assert (one_sided["relation"], one_sided["censoring"]) == ("<", "left")
    inhibition = measurements.query("endpoint == 'herg_percent_inhibition'")
    assert sorted(inhibition["test_concentration_value"].tolist()) == [10.0, 30.0]
    required_sparse = {
        "microsomal_stability_half_life",
        "hepatic_extraction_ratio",
        "plasma_protein_bound_percent",
        "plasma_protein_unbound_percent",
        "plasma_stability_half_life",
        "basic_pka",
        "acidic_pka_below_13",
    }
    assert required_sparse.issubset(set(measurements["endpoint"]))
    assert "aliased_structure_measurement_conflict" in set(result.quarantine["code"])


def test_sun_normalization_inverts_class_transforms_regression_and_quarantines_qc():
    result = normalize_sun_herg_workbook(_sun_sheets(), domain_max_mw_g_mol=600)
    activity = result.review_tables["activity"]

    first_class = activity.query("dataset_role == 'classification'").iloc[0]
    assert first_class["source_class"] == 0
    assert first_class["canonical_blocker_class"] == 1
    regression = activity.query("dataset_role == 'regression' and stored_log10_nm == 2.0").iloc[0]
    assert regression["pic50_value"] == pytest.approx(7.0)

    # CCO and OCC deduplicate structurally, but their disagreeing source classes remain.
    ethanol = activity[activity["raw_smiles"].isin(["CCO", "OCC"])]
    assert ethanol["structure_id"].nunique() == 1
    assert len(ethanol) >= 2
    codes = set(result.quarantine["code"])
    assert "conflicting_source_classes" in codes
    assert "train_validation_structure_overlap" in codes
    assert "validation_measurement_disagreement" in codes or "conflicting_measurements" in codes

    assert len(result.review_tables["domain_contradictions"]) >= 1
    assert result.summary["computed_mw_above_domain_limit_rows"] >= 1
    assert result.summary["regression_transform"] == "pIC50 = 9 - stored_log10_nM"
    assert result.summary["source_class_transform"] == "canonical_blocker_class = 1 - source_class"

    # A missing source IC50 remains explicitly missing and does not become NaN-as-label.
    missing = (
        result.tables["measurements"]
        .query("source == 'Sun hERG:Classification' and submitted_value == 'Not'")
        .iloc[0]
    )
    assert missing["relation"] == "not_reported"
    assert pd.isna(missing["value"])


def test_high_level_api_emits_all_canonical_parquet_tables(tmp_path: Path):
    workbook = tmp_path / "internal.xlsx"
    with pd.ExcelWriter(workbook) as writer:
        for name, frame in _internal_sheets().items():
            frame.to_excel(writer, sheet_name=name, index=False)
    output = tmp_path / "normalized"
    result = normalize_research_data(workbook, None, output)

    required = {
        "compounds",
        "chemical_states",
        "assay_protocols",
        "measurements",
        "pk_studies",
        "pk_samples",
        "derived_pk_parameters",
        "feature_lineage",
        "validation_issues",
    }
    assert required.issubset(result)
    for name in required:
        path = result[name]
        assert isinstance(path, Path)
        assert path.name == f"{name}.parquet"
        assert path.exists()
    assert pd.read_parquet(result["pk_samples"]).empty
    assert pd.read_parquet(result["feature_lineage"]).empty
