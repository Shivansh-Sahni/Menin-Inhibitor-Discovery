import numpy as np
import pandas as pd
from menin_discovery.curation import (
    aggregate_compounds,
    annotate_cross_source_mirrors,
    assess_pubchem_relevance,
    classify_assay_family,
    classify_pk_admet,
    normalize_activity_table,
    p_value_from_nm,
    parse_numeric,
    pubchem_to_long,
    unit_conversion_status,
    unit_factor_to_nm,
)


def test_parse_numeric_handles_relations_and_commas():
    assert parse_numeric("<1,000") == 1000
    assert parse_numeric("> 3.5e2") == 350
    assert np.isnan(parse_numeric("not reported"))


def test_p_value_from_nm():
    assert round(p_value_from_nm(100), 3) == 7.0
    assert round(p_value_from_nm(1000), 3) == 6.0


def test_units_are_strict_and_unicode_micro_is_supported():
    assert unit_factor_to_nm("nM") == 1
    assert unit_factor_to_nm("nmol/L") == 1
    assert unit_factor_to_nm("uM") == 1000
    assert unit_factor_to_nm("µM") == 1000
    assert np.isnan(unit_factor_to_nm(""))
    assert np.isnan(unit_factor_to_nm(None))
    assert np.isnan(unit_factor_to_nm("mg/mL"))
    assert unit_conversion_status("") == "missing_unit"
    assert unit_conversion_status("mg/mL") == "unsupported_unit"


def test_normalize_activity_and_aggregate():
    df = pd.DataFrame(
        {
            "source": ["x", "x", "x"],
            "source_record_id": [1, 2, 3],
            "compound_id": ["a", "a", "b"],
            "compound_name": ["", "", ""],
            "smiles": ["CCO", "CCO", "CCN"],
            "inchi_key": ["", "", ""],
            "target_name": ["Menin", "Menin", "Menin"],
            "target_id": ["CHEMBL1615381"] * 3,
            "endpoint": ["IC50", "Kd", "IC50"],
            "relation": ["=", "=", ">"],
            "value_raw": ["100", "200", "10000"],
            "standard_units": ["nM", "nM", "nM"],
            "assay_description": ["", "", ""],
            "assay_type": ["", "", ""],
            "document_id": ["", "", ""],
            "document_year": ["", "", ""],
            "reference": ["", "", ""],
            "source_detail": ["", "", ""],
        }
    )
    activity = normalize_activity_table(df)
    assert activity["p_value"].notna().all()
    compounds = aggregate_compounds(activity, exact_only=True)
    assert len(compounds) == 1
    assert compounds.iloc[0]["smiles"] == "CCO"


def test_censoring_bounds_and_per_compound_exact_policy():
    df = pd.DataFrame(
        {
            "source": ["x"] * 4,
            "compound_id": ["a", "a", "b", "c"],
            "smiles": ["CCO", "CCO", "CCN", "CCC"],
            "target_name": ["Menin"] * 4,
            "target_id": ["CHEMBL1615381"] * 4,
            "endpoint": ["IC50"] * 4,
            "relation": ["=", ">", "<", "="],
            "value_raw": [100, 200, 1000, 10],
            "standard_units": ["nM", "nM", "nM", "mg/mL"],
        }
    )
    activity = normalize_activity_table(df)
    right = activity.iloc[1]
    assert right["value_nm_lower_bound"] == 200
    assert np.isnan(right["value_nm_upper_bound"])
    assert np.isnan(right["p_activity_lower_bound"])
    assert right["p_activity_upper_bound"] == p_value_from_nm(200)

    left = activity.iloc[2]
    assert np.isnan(left["value_nm_lower_bound"])
    assert left["value_nm_upper_bound"] == 1000
    assert left["p_activity_lower_bound"] == p_value_from_nm(1000)
    assert np.isnan(left["p_activity_upper_bound"])

    # Strict mode keeps only exact observations. Per-compound mode uses exact
    # rows for compound a and preserves b as an explicitly bound-only record.
    strict = aggregate_compounds(activity, censoring_policy="strict_exact")
    assert set(strict["smiles"]) == {"CCO"}
    preferred = aggregate_compounds(activity, censoring_policy="prefer_exact_per_compound")
    assert set(preferred["smiles"]) == {"CCO", "CCN"}
    b = preferred.loc[preferred["smiles"].eq("CCN")].iloc[0]
    assert b["aggregation_value_semantics"] == "censoring_thresholds_only"


def test_unknown_unit_is_quarantined_not_assumed_nanomolar():
    activity = normalize_activity_table(
        pd.DataFrame(
            {
                "source": ["x"],
                "compound_id": ["a"],
                "smiles": ["CCO"],
                "target_name": ["Menin"],
                "target_id": ["O00255"],
                "endpoint": ["IC50"],
                "relation": ["="],
                "value_raw": [5],
                "standard_units": ["unspecified"],
            }
        )
    )
    row = activity.iloc[0]
    assert np.isnan(row["value_nm"])
    assert row["unit_conversion_status"] == "unsupported_unit"
    assert not row["is_modeling_eligible"]
    assert "unsupported_unit" in row["exclusion_reason"]


def test_source_quality_flags_and_duplicate_records_are_excluded():
    base = {
        "source": ["ChEMBL"] * 4,
        "source_record_id": [1, 2, 3, 3],
        "compound_id": ["a", "b", "c", "c"],
        "smiles": ["CCO", "CCN", "CCC", "CCC"],
        "target_name": ["Menin"] * 4,
        "target_id": ["O00255"] * 4,
        "endpoint": ["IC50"] * 4,
        "relation": ["="] * 4,
        "value_raw": [10, 20, 30, 30],
        "standard_units": ["nM"] * 4,
        "standard_flag": [1, 1, 1, 1],
        "potential_duplicate": [0, 1, 0, 0],
        "data_validity_comment": ["Outside typical range", "", "", ""],
    }
    activity = normalize_activity_table(pd.DataFrame(base))
    assert "chembl_data_validity_warning" in activity.loc[0, "exclusion_reason"]
    assert "chembl_potential_duplicate" in activity.loc[1, "exclusion_reason"]
    assert bool(activity.loc[3, "is_duplicate_measurement"])
    assert "duplicate_measurement" in activity.loc[3, "exclusion_reason"]
    assert activity.loc[2, "measurement_id"] == activity.loc[3, "measurement_id"]
    assert activity["is_modeling_eligible"].sum() == 1


def test_target_relevance_enforcement_can_be_disabled_for_other_tasks():
    raw = pd.DataFrame(
        {
            "source": ["internal"],
            "smiles": ["CCO"],
            "target_name": ["A future selectivity target"],
            "endpoint": ["IC50"],
            "relation": ["="],
            "value_raw": [10],
            "standard_units": ["nM"],
        }
    )
    strict = normalize_activity_table(raw)
    flexible = normalize_activity_table(raw, enforce_target_relevance=False)
    assert not strict.loc[0, "is_modeling_eligible"]
    assert flexible.loc[0, "is_modeling_eligible"]


def test_pubchem_does_not_invent_endpoint_or_unit_and_checks_relevance():
    raw = pd.DataFrame(
        {
            "aid": [1.0, 2.0],
            "PUBCHEM_RESULT_TAG": [10.0, 20.0],
            "PUBCHEM_CID": [100.0, 200.0],
            "PUBCHEM_EXT_DATASOURCE_SMILES": ["CCO", "CCN"],
            "PubChem Standard Value": [2.5, 3.0],
            "Target": ["", "LSD1"],
            "Target Accession(s)": ["", "O60341"],
        }
    )
    catalog = pd.DataFrame(
        {
            "aid": [1, 2],
            "assay_name": ["Menin/MLL interaction assay", "LSD1 inhibition"],
            "assay_description": ["Measures inhibition of menin binding", "LSD1 assay"],
        }
    )
    long = pubchem_to_long(raw, catalog)
    assert long.loc[0, "source_record_id"] == "1:10"
    assert long.loc[0, "compound_id"] == "100"
    assert long.loc[0, "endpoint"] == ""
    assert long.loc[0, "standard_units"] == ""
    assert long.loc[0, "target_relevance"] == "text_supported"
    assert bool(long.loc[0, "is_target_relevant"])
    assert long.loc[1, "target_relevance"] == "off_target"
    normalized = normalize_activity_table(long)
    assert not normalized["is_modeling_eligible"].any()
    assert "unsupported_endpoint" in normalized.loc[0, "exclusion_reason"]
    assert "target_not_relevant" in normalized.loc[1, "exclusion_reason"]


def test_pubchem_relevance_requires_explicit_evidence():
    assert assess_pubchem_relevance("Menin", "", "", "")[1]
    assert assess_pubchem_relevance("", "O00255", "", "")[1]
    assert not assess_pubchem_relevance("", "", "Generic interaction", "")[1]
    assert not assess_pubchem_relevance("LSD1", "O60341", "Menin search hit", "")[1]


def test_pubchem_review_registry_can_override_metadata_and_inclusion():
    raw = pd.DataFrame(
        {
            "aid": [7],
            "PUBCHEM_RESULT_TAG": [1],
            "PUBCHEM_CID": [99],
            "PUBCHEM_EXT_DATASOURCE_SMILES": ["CCO"],
            "PubChem Standard Value": [0.25],
        }
    )
    catalog = pd.DataFrame(
        {
            "aid": [7],
            "assay_name": ["Reviewed internal catalog entry"],
            "assay_description": ["Metadata was incomplete at the source"],
            "curation_decision": ["include"],
            "endpoint_override": ["IC50"],
            "units_override": ["uM"],
            "assay_family_override": ["biochemical_binding"],
        }
    )
    long = pubchem_to_long(raw, catalog)
    assert long.loc[0, "target_relevance"] == "manual_include"
    normalized = normalize_activity_table(long)
    assert normalized.loc[0, "endpoint"] == "IC50"
    assert normalized.loc[0, "value_nm"] == 250
    assert normalized.loc[0, "assay_family"] == "biochemical_binding"
    assert normalized.loc[0, "is_modeling_eligible"]


def test_pubchem_target_gene_lookup_is_explicit_target_evidence():
    raw = pd.DataFrame(
        {
            "aid": [8],
            "PUBCHEM_RESULT_TAG": [1],
            "PUBCHEM_CID": [101],
            "PUBCHEM_EXT_DATASOURCE_SMILES": ["CCO"],
            "IC50": [50],
            "IC50 Units": ["nM"],
        }
    )
    catalog = pd.DataFrame(
        {
            "aid": [8],
            "assay_name": ["Target-linked assay"],
            "assay_description": ["No free-text target name supplied"],
            "search_terms": ["generic text;target_gene:MEN1"],
        }
    )
    long = pubchem_to_long(raw, catalog)
    assert long.loc[0, "target_relevance"] == "confirmed_target_gene_lookup"
    assert bool(long.loc[0, "is_target_relevant"])
    assert normalize_activity_table(long).loc[0, "is_modeling_eligible"]


def test_pk_admet_rules_are_endpoint_specific_and_exclude_keyword_collisions():
    raw = pd.DataFrame(
        {
            "standard_type": ["CL", "MIC", "F", "IC50", "Solubility"],
            "assay_description": [
                "Intrinsic clearance in human liver microsomes",
                "Minimum inhibitory concentration; PK strain",
                "Fluorescence response in a biochemical assay",
                "Inhibition of CYP3A4 activity",
                "Aqueous solubility of the compound",
            ],
            "target_pref_name": ["", "", "", "CYP3A4", ""],
        }
    )
    classified = classify_pk_admet(raw)
    assert classified.loc[0, "admet_endpoint"] == "intrinsic_clearance"
    assert not bool(classified.loc[1, "is_admet_relevant"])
    assert not bool(classified.loc[2, "is_admet_relevant"])
    assert classified.loc[3, "admet_endpoint"] == "cyp_inhibition"
    assert classified.loc[4, "admet_endpoint"] == "solubility"


def test_patch_clamp_assays_are_classified_as_functional_electrophysiology():
    assert (
        classify_assay_family(
            "IC50",
            "T",
            "Automated patch clamp measurement of hERG tail current inhibition",
        )
        == "electrophysiology_functional"
    )


def test_cross_source_mirrors_preserve_same_source_replicates():
    rows = pd.DataFrame(
        {
            "structure_id": ["STR-1"] * 4,
            # Lower-priority sources deliberately precede ChEMBL to ensure
            # preference does not depend on row order.
            "source": ["PubChem", "ChEMBL", "BindingDB", "ChEMBL"],
            "source_record_id": ["p1", "c1", "b1", "c2"],
            "endpoint": ["IC50"] * 4,
            "assay_family": ["biochemical_binding"] * 4,
            "relation": ["="] * 4,
            "p_value": [7.0] * 4,
        }
    )
    annotated = annotate_cross_source_mirrors(rows)
    assert annotated["is_cross_source_mirror_candidate"].all()
    assert annotated["is_cross_source_mirror_redundant"].tolist() == [
        True,
        False,
        True,
        False,
    ]
    assert annotated["cross_source_mirror_group_id"].nunique() == 1
    assert annotated["cross_source_mirror_preferred_source"].eq("ChEMBL").all()


def test_cross_source_mirror_annotation_requires_an_exact_normalized_match():
    rows = pd.DataFrame(
        {
            "structure_id": ["STR-1"] * 4,
            "source": ["ChEMBL", "PubChem", "BindingDB", "PubChem"],
            "endpoint": ["IC50"] * 4,
            "assay_family": [
                "biochemical_binding",
                "biochemical_binding",
                "biochemical_binding",
                "cellular_functional",
            ],
            "relation": ["=", "=", ">", "="],
            "p_value": [7.0, 7.1, 7.0, 7.0],
        }
    )
    annotated = annotate_cross_source_mirrors(rows)
    assert not annotated["is_cross_source_mirror_candidate"].any()
    assert not annotated["is_cross_source_mirror_redundant"].any()
    assert annotated["cross_source_mirror_group_id"].eq("").all()


def test_ineligible_preferred_source_does_not_suppress_an_eligible_row():
    rows = pd.DataFrame(
        {
            "structure_id": ["STR-1", "STR-1"],
            "source": ["ChEMBL", "PubChem"],
            "endpoint": ["IC50", "IC50"],
            "assay_family": ["biochemical_binding", "biochemical_binding"],
            "relation": ["=", "="],
            "p_value": [7.0, 7.0],
            "is_modeling_eligible": [False, True],
        }
    )

    annotated = annotate_cross_source_mirrors(rows)

    assert not annotated["is_cross_source_mirror_candidate"].any()
    assert not annotated["is_cross_source_mirror_redundant"].any()
