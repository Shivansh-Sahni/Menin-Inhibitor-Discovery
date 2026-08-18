"""Process-centred research data contracts for PK, hERG, and physics work.

The contracts deliberately separate a registered compound, a chemical state,
an assay observation, a PK dosing event, and a computed observable.  That
separation is important for large, flexible molecules: a single flat row can
otherwise silently mix states, protocols, routes, and derived quantities.

Pydantic models validate individual records.  Matching Pandera
``DataFrameSchema`` objects validate table boundaries before typed Parquet is
written.  CSV export is intentionally limited to human-review tables.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import types
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar, Literal, get_args, get_origin

import pandas as pd
import pandera.pandas as pa
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Relation = Literal["=", "<", "<=", ">", ">=", "~", "not_reported"]
Censoring = Literal["none", "left", "right", "interval", "missing", "unknown"]
RecordOrigin = Literal["measured", "reported_derived", "recomputed", "predicted", "unknown"]
LeakageRole = Literal[
    "primary_observation",
    "derived_from_exposure",
    "derived_from_label",
    "physics_feature",
    "descriptor",
    "metadata_only",
    "unknown",
]


def _finite_or_none(value: float | None) -> float | None:
    if value is not None and not math.isfinite(value):
        raise ValueError("numeric values must be finite; represent censoring explicitly")
    return value


class ContractModel(BaseModel):
    """Strict, immutable base model shared by all canonical records."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
        validate_assignment=True,
    )
    contract_name: ClassVar[str]

    @field_validator("*", mode="before")
    @classmethod
    def _blank_optional_text_is_none(cls, value: Any) -> Any:
        return None if isinstance(value, str) and not value.strip() else value


class Compound(ContractModel):
    contract_name = "compounds"

    compound_id: str = Field(min_length=1)
    structure_id: str | None = None
    submitted_smiles: str | None = None
    canonical_smiles: str | None = None
    standardized_smiles: str | None = None
    standard_inchi_key: str | None = None
    molecular_weight_g_mol: float | None = Field(default=None, gt=0)
    formal_charge: int | None = None
    stereochemistry_status: Literal[
        "specified",
        "partially_specified",
        "unspecified",
        "not_applicable",
        "unknown",
    ] = "unknown"
    series_id: str | None = None
    scaffold: str | None = None
    scaffold_method: str | None = None
    source: str | None = None
    source_record_id: str | None = None
    validation_status: Literal["validated", "invalid", "unavailable", "unvalidated"] = "unvalidated"
    standardization_version: str | None = None
    context_note: str | None = None


class ChemicalState(ContractModel):
    contract_name = "chemical_states"

    chemical_state_id: str = Field(min_length=1)
    compound_id: str = Field(min_length=1)
    state_type: Literal[
        "neutral",
        "protomer",
        "tautomer",
        "stereoisomer",
        "microstate",
        "salt",
        "unknown",
    ] = "unknown"
    smiles: str | None = None
    formal_charge: int | None = None
    p_h: float | None = None
    fractional_population: float | None = Field(default=None, ge=0, le=1)
    environment: str | None = None
    method: str | None = None
    uncertainty: float | None = Field(default=None, ge=0)
    source: str | None = None
    context_note: str | None = None


class Conformer(ContractModel):
    contract_name = "conformers"

    conformer_id: str = Field(min_length=1)
    chemical_state_id: str = Field(min_length=1)
    rank: int | None = Field(default=None, ge=0)
    relative_energy_kcal_mol: float | None = None
    population: float | None = Field(default=None, ge=0, le=1)
    environment: str | None = None
    geometry_uri: str | None = None
    method: str | None = None
    source_run_id: str | None = None
    context_note: str | None = None

    _finite_energy = field_validator("relative_energy_kcal_mol")(_finite_or_none)


class AssayProtocol(ContractModel):
    contract_name = "assay_protocols"

    assay_protocol_id: str = Field(min_length=1)
    endpoint: str = Field(min_length=1)
    assay_family: str = Field(min_length=1)
    target_name: str | None = None
    species: str | None = None
    strain: str | None = None
    matrix: str | None = None
    route: str | None = None
    cell_system: str | None = None
    method: str | None = None
    temperature_c: float | None = None
    p_h: float | None = None
    duration_value: float | None = Field(default=None, ge=0)
    duration_unit: str | None = None
    test_concentration_value: float | None = Field(default=None, ge=0)
    test_concentration_unit: str | None = None
    source: str | None = None
    source_locator: str | None = None
    context_note: str | None = None

    _finite_temperature = field_validator("temperature_c")(_finite_or_none)


class Measurement(ContractModel):
    contract_name = "measurements"

    measurement_id: str = Field(min_length=1)
    compound_id: str = Field(min_length=1)
    chemical_state_id: str | None = None
    assay_protocol_id: str | None = None
    pk_study_id: str | None = None
    endpoint: str = Field(min_length=1)
    value: float | None = None
    unit: str | None = None
    relation: Relation = "="
    censoring: Censoring = "none"
    lower_bound: float | None = None
    upper_bound: float | None = None
    test_concentration_value: float | None = Field(default=None, ge=0)
    test_concentration_unit: str | None = None
    species: str | None = None
    matrix: str | None = None
    route: str | None = None
    origin: RecordOrigin = "measured"
    submitted_value: str | None = None
    source: str | None = None
    source_locator: str | None = None
    source_record_id: str | None = None
    pairing_status: Literal["resolved", "unresolved", "not_applicable"] = "not_applicable"
    leakage_role: LeakageRole = "primary_observation"
    model_eligible: bool = True
    context_note: str | None = None

    @field_validator("value", "lower_bound", "upper_bound")
    @classmethod
    def _finite_values(cls, value: float | None) -> float | None:
        return _finite_or_none(value)

    @model_validator(mode="after")
    def _coherent_qualification(self) -> Measurement:
        if self.relation == "not_reported" and self.value is not None:
            raise ValueError("not_reported measurements cannot have a numeric value")
        if self.value is not None and not self.unit:
            raise ValueError("numeric measurements require an explicit unit")
        if self.censoring == "interval":
            if self.lower_bound is None or self.upper_bound is None:
                raise ValueError("interval-censored measurements require both bounds")
            if self.lower_bound > self.upper_bound:
                raise ValueError("lower_bound cannot exceed upper_bound")
        return self


class PKStudy(ContractModel):
    contract_name = "pk_studies"

    pk_study_id: str = Field(min_length=1)
    event_pair_id: str | None = None
    compound_id: str = Field(min_length=1)
    chemical_state_id: str | None = None
    assay_protocol_id: str | None = None
    species: str = Field(min_length=1)
    strain: str | None = None
    sex: str | None = None
    route: Literal["IV", "PO", "SC", "IM", "IP", "other", "unknown"]
    dose_value: float | None = Field(default=None, gt=0)
    dose_unit: str | None = None
    formulation: str | None = None
    vehicle: str | None = None
    matrix: str | None = None
    source: str | None = None
    source_locator: str | None = None
    source_record_id: str | None = None
    pairing_status: Literal["resolved", "unresolved"] = "unresolved"
    context_note: str | None = None

    @model_validator(mode="after")
    def _dose_unit_present(self) -> PKStudy:
        if self.dose_value is not None and not self.dose_unit:
            raise ValueError("dose_value requires dose_unit")
        return self


class PKSample(ContractModel):
    contract_name = "pk_samples"

    pk_sample_id: str = Field(min_length=1)
    pk_study_id: str = Field(min_length=1)
    compound_id: str = Field(min_length=1)
    time_value: float = Field(ge=0)
    time_unit: str = Field(min_length=1)
    concentration_value: float | None = Field(default=None, ge=0)
    concentration_unit: str | None = None
    relation: Relation = "="
    censoring: Censoring = "none"
    lloq_value: float | None = Field(default=None, ge=0)
    lloq_unit: str | None = None
    matrix: str | None = None
    source: str | None = None
    source_locator: str | None = None
    context_note: str | None = None

    @model_validator(mode="after")
    def _concentration_unit_present(self) -> PKSample:
        if self.concentration_value is not None and not self.concentration_unit:
            raise ValueError("concentration_value requires concentration_unit")
        return self


class DerivedPKParameter(ContractModel):
    contract_name = "derived_pk_parameters"

    derived_pk_parameter_id: str = Field(min_length=1)
    compound_id: str = Field(min_length=1)
    pk_study_id: str | None = None
    event_pair_id: str | None = None
    endpoint: str = Field(min_length=1)
    value: float
    unit: str = Field(min_length=1)
    relation: Relation = "="
    origin: Literal["reported_derived", "recomputed"]
    method: str = Field(min_length=1)
    formula: str = Field(min_length=1)
    input_ids: tuple[str, ...] = ()
    reported_value: float | None = None
    recomputed_value: float | None = None
    closure_relative_error: float | None = Field(default=None, ge=0)
    closure_status: Literal["pass", "fail", "not_tested", "unresolved"] = "not_tested"
    leakage_role: LeakageRole = "derived_from_exposure"
    model_eligible: bool = False
    source: str | None = None
    source_locator: str | None = None
    context_note: str | None = None

    @field_validator("value", "reported_value", "recomputed_value", "closure_relative_error")
    @classmethod
    def _finite_parameters(cls, value: float | None) -> float | None:
        return _finite_or_none(value)


class PhysicsRun(ContractModel):
    contract_name = "physics_runs"

    physics_run_id: str = Field(min_length=1)
    compound_id: str = Field(min_length=1)
    chemical_state_id: str | None = None
    conformer_id: str | None = None
    process: str = Field(min_length=1)
    environment: str = Field(min_length=1)
    method: str = Field(min_length=1)
    software: str | None = None
    software_version: str | None = None
    force_field: str | None = None
    temperature_c: float | None = None
    p_h: float | None = None
    replicate: int | None = Field(default=None, ge=0)
    random_seed: int | None = None
    status: Literal["planned", "running", "complete", "failed", "quarantined"] = "planned"
    configuration_uri: str | None = None
    source: str | None = None
    context_note: str | None = None

    _finite_temperature = field_validator("temperature_c")(_finite_or_none)


class PhysicsObservable(ContractModel):
    contract_name = "physics_observables"

    physics_observable_id: str = Field(min_length=1)
    physics_run_id: str = Field(min_length=1)
    compound_id: str = Field(min_length=1)
    chemical_state_id: str | None = None
    conformer_id: str | None = None
    observable: str = Field(min_length=1)
    value: float | None = None
    unit: str | None = None
    relation: Relation = "="
    censoring: Censoring = "none"
    uncertainty: float | None = Field(default=None, ge=0)
    aggregation: str | None = None
    window_start: float | None = None
    window_end: float | None = None
    window_unit: str | None = None
    source: str | None = None
    context_note: str | None = None

    @field_validator("value", "uncertainty", "window_start", "window_end")
    @classmethod
    def _finite_observables(cls, value: float | None) -> float | None:
        return _finite_or_none(value)

    @model_validator(mode="after")
    def _observable_unit_present(self) -> PhysicsObservable:
        if self.value is not None and not self.unit:
            raise ValueError("numeric observables require an explicit unit")
        return self


class FeatureLineage(ContractModel):
    contract_name = "feature_lineage"

    feature_lineage_id: str = Field(min_length=1)
    compound_id: str = Field(min_length=1)
    chemical_state_id: str | None = None
    conformer_id: str | None = None
    feature_name: str = Field(min_length=1)
    feature_value: float | None = None
    feature_unit: str | None = None
    process_layer: str = Field(min_length=1)
    source_entity_type: Literal[
        "compound",
        "chemical_state",
        "conformer",
        "measurement",
        "pk_study",
        "pk_sample",
        "physics_observable",
        "external",
    ]
    source_entity_ids: tuple[str, ...] = ()
    transform: str = Field(min_length=1)
    formula: str | None = None
    leakage_role: LeakageRole = "unknown"
    model_eligible: bool = True
    version: str | None = None
    context_note: str | None = None

    _finite_feature = field_validator("feature_value")(_finite_or_none)


CONTRACT_MODELS: dict[str, type[ContractModel]] = {
    model.contract_name: model
    for model in (
        Compound,
        ChemicalState,
        Conformer,
        AssayProtocol,
        Measurement,
        PKStudy,
        PKSample,
        DerivedPKParameter,
        PhysicsRun,
        PhysicsObservable,
        FeatureLineage,
    )
}


def _unwrap_optional(annotation: Any) -> tuple[Any, bool]:
    origin = get_origin(annotation)
    if origin in (types.UnionType, __import__("typing").Union):
        args = get_args(annotation)
        nullable = type(None) in args
        non_none = tuple(arg for arg in args if arg is not type(None))
        if len(non_none) == 1:
            return non_none[0], nullable
    return annotation, False


def _pandera_dtype(annotation: Any) -> Any:
    annotation, _ = _unwrap_optional(annotation)
    origin = get_origin(annotation)
    if origin is Literal:
        values = get_args(annotation)
        if values and all(isinstance(item, bool) for item in values):
            return pd.BooleanDtype()
        if values and all(isinstance(item, int) and not isinstance(item, bool) for item in values):
            return pd.Int64Dtype()
        return pd.StringDtype()
    if annotation is str:
        return pd.StringDtype()
    if annotation is float:
        return pd.Float64Dtype()
    if annotation is int:
        return pd.Int64Dtype()
    if annotation is bool:
        return pd.BooleanDtype()
    if origin in (tuple, list, Sequence):
        return pa.Object
    return pa.Object


def dataframe_schema_for(model: str | type[ContractModel]) -> pa.DataFrameSchema:
    """Build the strict Pandera table schema corresponding to a contract."""

    model_class = CONTRACT_MODELS[model] if isinstance(model, str) else model
    columns: dict[str, pa.Column] = {}
    for name, field in model_class.model_fields.items():
        _, optional = _unwrap_optional(field.annotation)
        nullable = optional or field.default is None
        columns[name] = pa.Column(_pandera_dtype(field.annotation), nullable=nullable, required=True)
    primary_key = next((name for name in columns if name.endswith("_id")), None)
    checks = None
    if primary_key:
        checks = [pa.Check(lambda frame, key=primary_key: ~frame[key].duplicated(), error="duplicate ID")]
    return pa.DataFrameSchema(
        columns,
        checks=checks,
        strict=True,
        coerce=True,
        name=model_class.contract_name,
    )


DATAFRAME_SCHEMAS: dict[str, pa.DataFrameSchema] = {
    name: dataframe_schema_for(model) for name, model in CONTRACT_MODELS.items()
}

# Named aliases make schema discovery convenient in notebooks and tests.
COMPOUND_SCHEMA = DATAFRAME_SCHEMAS[Compound.contract_name]
CHEMICAL_STATE_SCHEMA = DATAFRAME_SCHEMAS[ChemicalState.contract_name]
CONFORMER_SCHEMA = DATAFRAME_SCHEMAS[Conformer.contract_name]
ASSAY_PROTOCOL_SCHEMA = DATAFRAME_SCHEMAS[AssayProtocol.contract_name]
MEASUREMENT_SCHEMA = DATAFRAME_SCHEMAS[Measurement.contract_name]
PK_STUDY_SCHEMA = DATAFRAME_SCHEMAS[PKStudy.contract_name]
PK_SAMPLE_SCHEMA = DATAFRAME_SCHEMAS[PKSample.contract_name]
DERIVED_PK_PARAMETER_SCHEMA = DATAFRAME_SCHEMAS[DerivedPKParameter.contract_name]
PHYSICS_RUN_SCHEMA = DATAFRAME_SCHEMAS[PhysicsRun.contract_name]
PHYSICS_OBSERVABLE_SCHEMA = DATAFRAME_SCHEMAS[PhysicsObservable.contract_name]
FEATURE_LINEAGE_SCHEMA = DATAFRAME_SCHEMAS[FeatureLineage.contract_name]


def _resolve_model(contract: str | type[ContractModel]) -> type[ContractModel]:
    if isinstance(contract, str):
        try:
            return CONTRACT_MODELS[contract]
        except KeyError as exc:
            raise KeyError(f"unknown research contract: {contract!r}") from exc
    if not issubclass(contract, ContractModel):
        raise TypeError("contract must be a registered contract name or ContractModel subclass")
    return contract


def records_to_frame(
    contract: str | type[ContractModel],
    records: Iterable[ContractModel | Mapping[str, Any]],
) -> pd.DataFrame:
    """Validate records with Pydantic and return a schema-validated dataframe."""

    model = _resolve_model(contract)
    validated = [item if isinstance(item, model) else model.model_validate(item) for item in records]
    rows = [item.model_dump(mode="python") for item in validated]
    frame = pd.DataFrame(rows, columns=list(model.model_fields))
    return DATAFRAME_SCHEMAS[model.contract_name].validate(frame, lazy=True)


def validate_contract_frame(
    contract: str | type[ContractModel],
    frame: pd.DataFrame,
    *,
    lazy: bool = True,
) -> pd.DataFrame:
    """Validate a dataframe and every row against the selected contract."""

    model = _resolve_model(contract)
    schema_frame = DATAFRAME_SCHEMAS[model.contract_name].validate(frame.copy(), lazy=lazy)

    def null_safe(record: Mapping[str, Any]) -> dict[str, Any]:
        cleaned: dict[str, Any] = {}
        for key, value in record.items():
            if isinstance(value, (tuple, list, dict)):
                cleaned[key] = value
                continue
            try:
                cleaned[key] = None if bool(pd.isna(value)) else value
            except (TypeError, ValueError):
                cleaned[key] = value
        return cleaned

    validated = [model.model_validate(null_safe(record)) for record in schema_frame.to_dict("records")]
    normalized = pd.DataFrame(
        [item.model_dump(mode="python") for item in validated],
        columns=list(model.model_fields),
    )
    return DATAFRAME_SCHEMAS[model.contract_name].validate(normalized, lazy=lazy)


def write_contract_parquet(
    contract: str | type[ContractModel],
    frame: pd.DataFrame,
    path: str | os.PathLike[str],
    *,
    compression: str = "zstd",
) -> Path:
    """Atomically validate and write one canonical table as typed Parquet."""

    model = _resolve_model(contract)
    validated = validate_contract_frame(model, frame)
    destination = Path(path)
    if destination.suffix.casefold() not in {".parquet", ".pq"}:
        raise ValueError("canonical contract tables must be written as Parquet")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{destination.stem}-",
            suffix=destination.suffix,
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        validated.to_parquet(temporary, index=False, compression=compression)
        temporary.replace(destination)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise
    return destination


def write_contract_tables(
    tables: Mapping[str, pd.DataFrame],
    directory: str | os.PathLike[str],
    *,
    compression: str = "zstd",
) -> dict[str, Path]:
    """Write a mapping of registered canonical tables to one Parquet directory."""

    root = Path(directory)
    return {
        name: write_contract_parquet(
            name,
            frame,
            root / f"{name}.parquet",
            compression=compression,
        )
        for name, frame in tables.items()
    }


def write_review_csv(
    frame: pd.DataFrame,
    path: str | os.PathLike[str],
    *,
    purpose: str,
) -> Path:
    """Write a non-canonical CSV intended only for a named human-review purpose."""

    if not purpose.strip():
        raise ValueError("review CSV export requires a non-empty purpose")
    destination = Path(path)
    if destination.suffix.casefold() != ".csv":
        raise ValueError("review exports must use a .csv extension")
    destination.parent.mkdir(parents=True, exist_ok=True)
    review = frame.copy()
    review.insert(0, "review_purpose", purpose.strip())
    review.to_csv(destination, index=False)
    return destination


def contract_catalog_json() -> str:
    """Return a deterministic, lightweight description of all table contracts."""

    payload = {
        name: {
            "model": model.__name__,
            "fields": list(model.model_fields),
        }
        for name, model in sorted(CONTRACT_MODELS.items())
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def contract_json_schemas() -> dict[str, dict[str, Any]]:
    """Return the complete machine-readable Pydantic schema for each table."""

    return {name: model.model_json_schema() for name, model in sorted(CONTRACT_MODELS.items())}


def contract_data_dictionary() -> pd.DataFrame:
    """Return a reviewable field-level dictionary for every canonical table.

    This is deliberately derived from the executable contracts so the human
    documentation cannot silently drift away from Pydantic/Pandera validation.
    """

    rows: list[dict[str, Any]] = []
    for table_name, model in sorted(CONTRACT_MODELS.items()):
        schema = model.model_json_schema()
        required = set(schema.get("required", []))
        for field_name, definition in schema.get("properties", {}).items():
            alternatives = definition.get("anyOf", [])
            types = [item.get("type") for item in alternatives if item.get("type")]
            if definition.get("type"):
                types.insert(0, definition["type"])
            allowed = list(definition.get("enum", []))
            for item in alternatives:
                allowed.extend(item.get("enum", []))
            nullable = "null" in types
            value_types = sorted({str(item) for item in types if item != "null"})
            excluded = {"title", "description", "default", "type", "anyOf", "enum"}
            constraints = {key: value for key, value in definition.items() if key not in excluded}
            rows.append(
                {
                    "table": table_name,
                    "field": field_name,
                    "required": field_name in required,
                    "nullable": nullable,
                    "value_type": "|".join(value_types) or "object",
                    "allowed_values": "|".join(str(value) for value in dict.fromkeys(allowed)),
                    "default": definition.get("default"),
                    "description": definition.get("description", ""),
                    "constraints_json": json.dumps(constraints, sort_keys=True, separators=(",", ":")),
                    "storage": "typed Parquet",
                    "identifier_role": (
                        "primary key"
                        if field_name
                        == next(
                            (name for name in model.model_fields if name.endswith("_id")),
                            None,
                        )
                        else "foreign/reference key"
                        if field_name.endswith("_id") or field_name.endswith("_ids")
                        else "attribute"
                    ),
                }
            )
    return pd.DataFrame(rows)


__all__ = [
    "ASSAY_PROTOCOL_SCHEMA",
    "CHEMICAL_STATE_SCHEMA",
    "COMPOUND_SCHEMA",
    "CONFORMER_SCHEMA",
    "CONTRACT_MODELS",
    "DATAFRAME_SCHEMAS",
    "DERIVED_PK_PARAMETER_SCHEMA",
    "FEATURE_LINEAGE_SCHEMA",
    "MEASUREMENT_SCHEMA",
    "PHYSICS_OBSERVABLE_SCHEMA",
    "PHYSICS_RUN_SCHEMA",
    "PK_SAMPLE_SCHEMA",
    "PK_STUDY_SCHEMA",
    "AssayProtocol",
    "ChemicalState",
    "Compound",
    "Conformer",
    "ContractModel",
    "DerivedPKParameter",
    "FeatureLineage",
    "Measurement",
    "PKSample",
    "PKStudy",
    "PhysicsObservable",
    "PhysicsRun",
    "contract_catalog_json",
    "contract_data_dictionary",
    "contract_json_schemas",
    "dataframe_schema_for",
    "records_to_frame",
    "validate_contract_frame",
    "write_contract_parquet",
    "write_contract_tables",
    "write_review_csv",
]
