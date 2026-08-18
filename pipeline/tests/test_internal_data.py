import json
import os
import stat

import pandas as pd
import pytest
from menin_discovery.chemistry import rdkit_available
from menin_discovery.internal_data import (
    ingest_internal_data,
    load_internal_data_config,
    pseudonymize_identifier,
    validate_internal_table,
)
from menin_discovery.provenance import create_manifest, verify_manifest

KEY = "test-only-key-material-32-bytes!!"


def _config(**overrides):
    payload = {
        "columns": {
            "smiles": "SMILES",
            "value": "Result",
            "units": "Units",
            "relation": "Relation",
            "endpoint": "Endpoint",
            "compound_id": "CompoundCode",
            "batch_id": "BatchCode",
            "assay_id": "AssayCode",
            "row_id": "MeasurementCode",
            "measurement_date": "Date",
            "replicate": "Replicate",
        },
        "assay_registry": {
            "MENIN_FP": {
                "endpoint": "IC50",
                "target_name": "Menin",
                "target_id": "O00255",
                "assay_family": "biochemical_inhibition",
                "allowed_units": ["nM", "uM"],
            }
        },
        "endpoint_registry": {
            "IC50": {
                "canonical_name": "IC50",
                "family": "inhibitory_potency",
                "allowed_units": ["nM", "uM"],
            }
        },
        "require_rdkit": rdkit_available(),
    }
    payload.update(overrides)
    return payload


def _rows():
    return pd.DataFrame(
        {
            "SMILES": ["CCO", "CCN"],
            "Result": ["10", "0.2"],
            "Units": ["nM", "uM"],
            "Relation": ["=", "<="],
            "Endpoint": ["IC50", "IC50"],
            "CompoundCode": ["SECRET-CMP-1", "SECRET-CMP-2"],
            "BatchCode": ["SECRET-BATCH-A", "SECRET-BATCH-A"],
            "AssayCode": ["MENIN_FP", "MENIN_FP"],
            "MeasurementCode": ["ROW-1", "ROW-2"],
            "Date": ["2026-01-10", "2026-01-10"],
            "Replicate": ["1", "2"],
        }
    )


def test_valid_table_is_pseudonymized_standardized_and_deterministic():
    first = validate_internal_table(_rows(), config=_config(), pseudonymization_key=KEY)
    second = validate_internal_table(
        _rows().iloc[::-1].reset_index(drop=True),
        config=_config(),
        pseudonymization_key=KEY,
    )

    assert len(first.accepted) == 2
    assert first.quarantine.empty
    assert set(first.accepted["value_nm"]) == {10.0, 200.0}
    assert set(first.accepted["relation"]) == {"=", "<="}
    assert first.summary["rows_accepted"] == 2
    assert first.summary == second.summary
    assert first.accepted.to_csv(index=False) == second.accepted.to_csv(index=False)
    assert first.accepted["internal_row_id"].str.startswith("IROW-").all()
    assert first.accepted["internal_compound_id"].str.startswith("ICMP-").all()
    assert first.accepted["internal_source_compound_id"].str.startswith("ISRCMP-").all()
    assert first.accepted["internal_batch_id"].nunique() == 1
    assert first.accepted["internal_assay_id"].nunique() == 1

    serialized = first.accepted.to_csv(index=False) + json.dumps(first.summary)
    assert "SECRET-CMP" not in serialized
    assert "SECRET-BATCH" not in serialized
    assert KEY not in serialized
    assert not any(
        column in first.accepted.columns
        for column in ["CompoundCode", "BatchCode", "AssayCode", "MeasurementCode"]
    )


def test_conflicting_source_ids_are_quarantined():
    table = _rows()
    table.loc[1, "CompoundCode"] = table.loc[0, "CompoundCode"]
    table.loc[1, "MeasurementCode"] = table.loc[0, "MeasurementCode"]
    result = validate_internal_table(table, config=_config(), pseudonymization_key=KEY)
    reordered = validate_internal_table(
        table.iloc[::-1].reset_index(drop=True),
        config=_config(),
        pseudonymization_key=KEY,
    )

    assert result.accepted.empty
    assert len(result.quarantine) == 2
    assert {"compound_structure_conflict", "source_row_id_conflict"}.issubset(set(result.issues["code"]))
    assert result.quarantine.to_csv(index=False) == reordered.quarantine.to_csv(index=False)


def test_unknown_units_and_conflicting_metadata_are_quarantined_with_raw_values():
    table = _rows().iloc[[0]].copy()
    table.loc[0, "Units"] = "mg/mL"
    table.loc[0, "Endpoint"] = "Kd"
    result = validate_internal_table(table, config=_config(), pseudonymization_key=KEY)

    assert result.accepted.empty
    assert len(result.quarantine) == 1
    quarantined = result.quarantine.iloc[0]
    assert quarantined["submitted_value"] == "10"
    assert quarantined["submitted_units"] == "mg/mL"
    assert quarantined["submitted_relation"] == "="
    assert quarantined["submitted_endpoint"] == "Kd"
    codes = set(result.issues["code"])
    assert "unsupported_unit" in codes
    assert "unknown_endpoint" in codes


def test_assay_registry_can_supply_explicit_endpoint_and_units():
    table = _rows().iloc[[0]].drop(columns=["Units", "Endpoint", "Relation"])
    config = _config(
        columns={
            "smiles": "SMILES",
            "value": "Result",
            "compound_id": "CompoundCode",
            "batch_id": "BatchCode",
            "assay_id": "AssayCode",
            "row_id": "MeasurementCode",
        },
        defaults={"relation": "="},
        assay_registry={
            "MENIN_FP": {
                "endpoint": "IC50",
                "units": "nM",
                "target_name": "Menin",
                "target_id": "O00255",
                "assay_family": "biochemical_inhibition",
                "allowed_units": ["nM"],
            }
        },
    )
    result = validate_internal_table(table, config=config, pseudonymization_key=KEY)

    assert len(result.accepted) == 1
    assert result.accepted.loc[0, "submitted_units"] == ""
    assert result.accepted.loc[0, "submitted_relation"] == ""
    assert result.accepted.loc[0, "standard_units"] == "nm"
    assert result.accepted.loc[0, "unit_source"] == "assay_registry"
    assert result.accepted.loc[0, "relation_source"] == "configuration_default"
    assert result.accepted.loc[0, "endpoint"] == "IC50"
    assert result.accepted.loc[0, "endpoint_source"] == "assay_registry"


def test_unregistered_assay_and_missing_batch_are_quarantined_without_echoing_ids():
    table = _rows().iloc[[0]].copy()
    table.loc[0, "AssayCode"] = "UNRELEASED-ASSAY-NAME"
    table.loc[0, "BatchCode"] = ""
    result = validate_internal_table(table, config=_config(), pseudonymization_key=KEY)

    assert {"unknown_assay", "missing_batch_id"}.issubset(set(result.issues["code"]))
    output = result.quarantine.to_csv(index=False) + result.issues.to_csv(index=False)
    assert "UNRELEASED-ASSAY-NAME" not in output


def test_cohort_roles_separate_development_external_and_blind_data():
    table = _rows().copy()
    table["CohortRole"] = ["development", "locked_external"]
    config = _config()
    config["columns"] = {**config["columns"], "cohort_role": "CohortRole"}
    config["required_fields"] = [
        *config.get(
            "required_fields",
            ["smiles", "value", "units", "relation", "endpoint", "batch_id", "assay_id"],
        ),
        "cohort_role",
    ]
    result = validate_internal_table(table, config=config, pseudonymization_key=KEY)
    assert set(result.accepted["cohort_role"]) == {"development", "locked_external"}
    assert result.summary["cohort_role_counts"] == {
        "development": 1,
        "locked_external": 1,
    }

    table.loc[0, "CohortRole"] = "train_and_test"
    invalid = validate_internal_table(table, config=config, pseudonymization_key=KEY)
    assert "invalid_cohort_role" in set(invalid.issues["code"])


def test_missing_required_source_column_produces_quarantine_and_summary():
    table = _rows().drop(columns=["Units"])
    result = validate_internal_table(table, config=_config(), pseudonymization_key=KEY)

    assert result.accepted.empty
    assert len(result.quarantine) == 2
    assert "missing_source_column" in set(result.issues["code"])
    assert result.summary["missing_required_source_fields"] == ["units"]


def test_csv_and_tsv_outputs_are_atomic_deterministic_and_manifest_compatible(tmp_path):
    csv_path = tmp_path / "private.csv"
    tsv_path = tmp_path / "private.tsv"
    _rows().to_csv(csv_path, index=False)
    _rows().to_csv(tsv_path, index=False, sep="\t")

    output = tmp_path / "processed"
    first = ingest_internal_data(
        csv_path,
        config=_config(),
        pseudonymization_key=KEY,
        output_directory=output,
    )
    first_bytes = {path.name: path.read_bytes() for path in first.output_paths.values()}
    second = ingest_internal_data(
        csv_path,
        config=_config(),
        pseudonymization_key=KEY,
        output_directory=output,
    )
    second_bytes = {path.name: path.read_bytes() for path in second.output_paths.values()}
    assert first_bytes == second_bytes
    assert not list(output.glob("*.tmp"))
    if os.name == "posix":
        assert stat.S_IMODE(output.stat().st_mode) == 0o700
        assert all(stat.S_IMODE(path.stat().st_mode) == 0o600 for path in first.output_paths.values())

    tsv_result = ingest_internal_data(tsv_path, config=_config(), pseudonymization_key=KEY)
    assert tsv_result.summary["input_format"] == "tsv"
    assert len(tsv_result.accepted) == 2

    manifest = create_manifest(output, stage="internal-processed", created_at="2026-01-01T00:00:00Z")
    assert manifest["file_count"] == 4
    assert verify_manifest(manifest, root=output).valid


def test_secrets_must_be_runtime_only_and_cryptographically_sized():
    with pytest.raises(ValueError, match="runtime"):
        load_internal_data_config({**_config(), "pseudonymization_key": "must-not-live-here"})
    nested = _config(
        assay_registry={
            "MENIN_FP": {
                **_config()["assay_registry"]["MENIN_FP"],
                "secret": "must-not-live-here",
            }
        }
    )
    with pytest.raises(ValueError, match="runtime"):
        load_internal_data_config(nested)
    deeply_nested = _config(
        assay_registry={
            "MENIN_FP": {
                **_config()["assay_registry"]["MENIN_FP"],
                "instrument": {"authentication": {"salt": "must-not-live-here"}},
            }
        }
    )
    with pytest.raises(ValueError, match="runtime"):
        load_internal_data_config(deeply_nested)
    with pytest.raises(ValueError, match="at least 16 bytes"):
        pseudonymize_identifier("source-id", pseudonymization_key="short", namespace="row")

    first = pseudonymize_identifier("source-id", pseudonymization_key=KEY, namespace="row")
    second = pseudonymize_identifier("source-id", pseudonymization_key=KEY, namespace="assay")
    third = pseudonymize_identifier(
        "source-id",
        pseudonymization_key="different-key-material-32-bytes!",
        namespace="row",
    )
    assert len({first, second, third}) == 3
    assert "source-id" not in first


@pytest.mark.skipif(not rdkit_available(), reason="SDF import requires RDKit")
def test_sdf_import_uses_structure_and_properties(tmp_path):
    from rdkit import Chem

    sdf_path = tmp_path / "internal.sdf"
    writer = Chem.SDWriter(str(sdf_path))
    molecule = Chem.MolFromSmiles("CCO")
    properties = {
        "Result": "15",
        "Units": "nM",
        "Relation": "=",
        "Endpoint": "IC50",
        "CompoundCode": "SDF-SECRET-CMP",
        "BatchCode": "SDF-SECRET-BATCH",
        "AssayCode": "MENIN_FP",
        "MeasurementCode": "SDF-ROW-1",
    }
    for name, value in properties.items():
        molecule.SetProp(name, value)
    writer.write(molecule)
    writer.close()

    config = _config(columns={**_config()["columns"], "smiles": "__sdf_smiles"})
    result = ingest_internal_data(sdf_path, config=config, pseudonymization_key=KEY)

    assert result.summary["input_format"] == "sdf"
    assert len(result.accepted) == 1
    assert result.accepted.loc[0, "standardized_smiles"] == "CCO"
    assert result.accepted.loc[0, "value_nm"] == 15.0
