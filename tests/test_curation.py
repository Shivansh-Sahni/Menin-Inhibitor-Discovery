import numpy as np
import pandas as pd

from menin_discovery.curation import (
    aggregate_compounds,
    normalize_activity_table,
    parse_numeric,
    p_value_from_nm,
)


def test_parse_numeric_handles_relations_and_commas():
    assert parse_numeric("<1,000") == 1000
    assert parse_numeric("> 3.5e2") == 350
    assert np.isnan(parse_numeric("not reported"))


def test_p_value_from_nm():
    assert round(p_value_from_nm(100), 3) == 7.0
    assert round(p_value_from_nm(1000), 3) == 6.0


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
