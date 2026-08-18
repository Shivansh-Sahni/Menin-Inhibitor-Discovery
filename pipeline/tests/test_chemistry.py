import pandas as pd
import pytest
from menin_discovery.chemistry import (
    rdkit_available,
    standardization_version,
    standardize_smiles,
    standardize_structure_table,
    structure_id_from_smiles,
)


def test_structure_identifier_is_stable_and_namespaced():
    first = structure_id_from_smiles("CCO")
    assert first == structure_id_from_smiles("CCO")
    assert first != structure_id_from_smiles("CCN")
    assert first != structure_id_from_smiles("CCO", namespace="different")
    assert first.startswith("STR-")


def test_structure_standardization_retains_input_and_is_traceable():
    result = standardize_smiles(" CCO ")
    assert result.original_smiles == "CCO"
    assert result.standardized_smiles
    assert result.structure_id
    if rdkit_available():
        assert result.structure_valid is True
        assert result.structure_standardization_status == "standardized"
        assert result.standard_inchi_key
    else:
        assert result.structure_valid is None
        assert result.structure_standardization_status == "rdkit_unavailable"
        assert result.structure_id.startswith("RAW-")


def test_structure_table_preserves_original_columns():
    table = pd.DataFrame({"smiles": [" CCO "], "inchi_key": ["submitted-key"]})
    out = standardize_structure_table(table)
    assert out.loc[0, "original_smiles"] == "CCO"
    assert out.loc[0, "original_inchi_key"] == "submitted-key"
    assert out.loc[0, "structure_id"]
    assert out.loc[0, "smiles"]


def test_missing_structure_has_no_identifier():
    result = standardize_smiles("")
    assert result.structure_id == ""
    assert result.structure_standardization_status == "missing_structure"


@pytest.mark.skipif(not rdkit_available(), reason="policy identity test requires RDKit")
def test_standardization_policy_changes_identity_namespace_and_structure_id():
    stripped = standardize_smiles("CCO.Cl", strip_salts=True)
    salt_retained = standardize_smiles("CCO.Cl", strip_salts=False)
    tautomer_policy = standardize_smiles(
        "CCO.Cl",
        strip_salts=True,
        canonicalize_tautomer=True,
    )

    assert stripped.standardized_smiles == "CCO"
    assert salt_retained.standardized_smiles != stripped.standardized_smiles
    assert (
        len(
            {
                stripped.structure_standardization_version,
                salt_retained.structure_standardization_version,
                tautomer_policy.structure_standardization_version,
            }
        )
        == 3
    )
    assert (
        len(
            {
                stripped.structure_id,
                salt_retained.structure_id,
                tautomer_policy.structure_id,
            }
        )
        == 3
    )
    assert stripped.structure_standardization_version == standardization_version()
