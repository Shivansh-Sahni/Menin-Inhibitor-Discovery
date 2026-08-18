from pathlib import Path

import pandas as pd
import pandera.errors
import pytest
from menin_discovery.research_contracts import (
    CONTRACT_MODELS,
    AssayProtocol,
    ChemicalState,
    Compound,
    Conformer,
    DerivedPKParameter,
    FeatureLineage,
    Measurement,
    PhysicsObservable,
    PhysicsRun,
    PKSample,
    PKStudy,
    contract_data_dictionary,
    contract_json_schemas,
    records_to_frame,
    validate_contract_frame,
    write_contract_parquet,
    write_review_csv,
)
from pydantic import ValidationError


def _minimal_records():
    return {
        Compound: {"compound_id": "CMP-1"},
        ChemicalState: {"chemical_state_id": "STATE-1", "compound_id": "CMP-1"},
        Conformer: {"conformer_id": "CONF-1", "chemical_state_id": "STATE-1"},
        AssayProtocol: {
            "assay_protocol_id": "AP-1",
            "endpoint": "auc_0_inf",
            "assay_family": "in_vivo_pk",
        },
        Measurement: {
            "measurement_id": "MEAS-1",
            "compound_id": "CMP-1",
            "endpoint": "auc_0_inf",
            "value": 100.0,
            "unit": "ng*h/mL",
        },
        PKStudy: {
            "pk_study_id": "PK-1",
            "compound_id": "CMP-1",
            "species": "Rat",
            "route": "IV",
            "dose_value": 2.0,
            "dose_unit": "mg/kg",
        },
        PKSample: {
            "pk_sample_id": "SAMPLE-1",
            "pk_study_id": "PK-1",
            "compound_id": "CMP-1",
            "time_value": 1.0,
            "time_unit": "h",
            "concentration_value": 10.0,
            "concentration_unit": "ng/mL",
        },
        DerivedPKParameter: {
            "derived_pk_parameter_id": "DPK-1",
            "compound_id": "CMP-1",
            "endpoint": "clearance",
            "value": 3.0,
            "unit": "mL/kg/min",
            "origin": "recomputed",
            "method": "dose/AUC closure",
            "formula": "dose/AUC",
        },
        PhysicsRun: {
            "physics_run_id": "RUN-1",
            "compound_id": "CMP-1",
            "process": "passive_permeation",
            "environment": "water_to_membrane",
            "method": "umbrella_sampling",
        },
        PhysicsObservable: {
            "physics_observable_id": "OBS-1",
            "physics_run_id": "RUN-1",
            "compound_id": "CMP-1",
            "observable": "free_energy_barrier",
            "value": 12.0,
            "unit": "kcal/mol",
        },
        FeatureLineage: {
            "feature_lineage_id": "FL-1",
            "compound_id": "CMP-1",
            "feature_name": "membrane_barrier",
            "process_layer": "passive_permeation",
            "source_entity_type": "physics_observable",
            "source_entity_ids": ("OBS-1",),
            "transform": "identity",
        },
    }


def test_all_pydantic_contracts_have_matching_dataframe_schemas():
    records = _minimal_records()
    assert set(CONTRACT_MODELS.values()) == set(records)
    for model, record in records.items():
        validated = model.model_validate(record)
        frame = records_to_frame(model, [validated])
        assert list(frame.columns) == list(model.model_fields)
        assert len(frame) == 1


def test_contracts_reject_extra_fields_nonfinite_values_and_missing_units():
    with pytest.raises(ValidationError):
        Compound(compound_id="CMP-1", unexpected="no")
    with pytest.raises(ValidationError):
        Measurement(
            measurement_id="M-1",
            compound_id="C-1",
            endpoint="herg_ic50",
            value=float("inf"),
            unit="nM",
        )
    with pytest.raises(ValidationError):
        Measurement(
            measurement_id="M-1",
            compound_id="C-1",
            endpoint="herg_ic50",
            value=10,
        )
    with pytest.raises(ValidationError):
        Measurement(
            measurement_id="M-1",
            compound_id="C-1",
            endpoint="herg_ic50",
            value=10,
            unit="nM",
            relation="not_reported",
        )


def test_dataframe_schema_rejects_duplicate_primary_ids():
    frame = records_to_frame(Compound, [{"compound_id": "CMP-1"}])
    duplicated = pd.concat([frame, frame], ignore_index=True)
    with pytest.raises(pandera.errors.SchemaErrors):
        validate_contract_frame(Compound, duplicated)


def test_typed_parquet_writer_roundtrips_nullable_types(tmp_path: Path):
    frame = records_to_frame(
        Compound,
        [{"compound_id": "CMP-1", "molecular_weight_g_mol": 701.2, "formal_charge": None}],
    )
    path = write_contract_parquet(Compound, frame, tmp_path / "compounds.parquet")
    restored = pd.read_parquet(path)
    assert restored.loc[0, "compound_id"] == "CMP-1"
    assert restored.loc[0, "molecular_weight_g_mol"] == pytest.approx(701.2)
    assert pd.isna(restored.loc[0, "formal_charge"])

    with pytest.raises(ValueError, match="Parquet"):
        write_contract_parquet(Compound, frame, tmp_path / "compounds.csv")


def test_csv_is_explicitly_review_only(tmp_path: Path):
    frame = pd.DataFrame({"code": ["unresolved_pairing"]})
    with pytest.raises(ValueError, match="purpose"):
        write_review_csv(frame, tmp_path / "issues.csv", purpose="")
    path = write_review_csv(frame, tmp_path / "issues.csv", purpose="manual pairing review")
    restored = pd.read_csv(path)
    assert restored.loc[0, "review_purpose"] == "manual pairing review"


def test_data_dictionary_is_derived_from_every_executable_contract():
    dictionary = contract_data_dictionary()
    schemas = contract_json_schemas()
    assert set(dictionary["table"]) == set(CONTRACT_MODELS)
    assert set(schemas) == set(CONTRACT_MODELS)
    for table, model in CONTRACT_MODELS.items():
        fields = set(dictionary.loc[dictionary["table"] == table, "field"])
        assert fields == set(model.model_fields)
    compound_id = dictionary[
        (dictionary["table"] == "compounds") & (dictionary["field"] == "compound_id")
    ].iloc[0]
    assert bool(compound_id["required"])
    assert compound_id["identifier_role"] == "primary key"
