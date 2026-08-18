"""Offline, privacy-preserving intake for proprietary laboratory measurements.

The intake boundary is intentionally stricter than the public-data adapters:
column meanings and assay metadata must be supplied explicitly, unsupported
units are quarantined, and source identifiers never leave the boundary in
plain text.  Deterministic HMAC identifiers allow repeatable joins without
embedding the caller-supplied pseudonymization key in configuration or output.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .chemistry import rdkit_available, standardization_version, standardize_structure_table
from .curation import (
    VALID_RELATIONS,
    endpoint_family,
    normalize_relation,
    normalize_unit,
    p_value_from_nm,
    parse_numeric,
    unit_factor_to_nm,
)

INTERNAL_INTAKE_VERSION = "1.0"
SUPPORTED_INPUT_FORMATS = frozenset({"csv", "tsv", "sdf"})
FORBIDDEN_CONFIG_KEYS = frozenset({"pseudonymization_key", "pseudonymization_secret", "secret", "salt"})

CANONICAL_INPUT_FIELDS = frozenset(
    {
        "smiles",
        "value",
        "units",
        "relation",
        "endpoint",
        "compound_id",
        "batch_id",
        "assay_id",
        "row_id",
        "measurement_date",
        "replicate",
        "target_name",
        "target_id",
        "assay_family",
        "cohort_role",
    }
)

ALLOWED_COHORT_ROLES = frozenset({"development", "locked_external", "prospective_blind"})

_ID_PREFIXES = {
    "row": "IROW",
    "record_content": "IREC",
    "compound": "ICMP",
    "source_compound": "ISRCMP",
    "batch": "IBAT",
    "assay": "IASSAY",
}


def _clean_text(value: object) -> str:
    if value is None:
        return ""
    try:
        if bool(pd.isna(value)):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_registry(registry: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    normalized: dict[str, dict[str, Any]] = {}
    for raw_key, raw_metadata in registry.items():
        key = _clean_text(raw_key).casefold()
        if not key:
            raise ValueError("Registry keys must not be empty")
        if key in normalized:
            raise ValueError("Registry contains duplicate case-insensitive keys")
        if not isinstance(raw_metadata, Mapping):
            raise TypeError("Registry metadata entries must be mappings")
        normalized[key] = dict(raw_metadata)
    return normalized


def _forbidden_key_paths(value: object, path: str = "") -> list[str]:
    """Locate secret-bearing keys recursively without inspecting/logging values."""

    found: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            name = str(key)
            child_path = f"{path}.{name}" if path else name
            if name.casefold() in FORBIDDEN_CONFIG_KEYS:
                found.append(child_path)
            found.extend(_forbidden_key_paths(child, child_path))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            found.extend(_forbidden_key_paths(child, f"{path}[{index}]"))
    return found


@dataclass(frozen=True)
class InternalDataConfig:
    """Declarative schema and assay registry for one internal data feed.

    ``columns`` maps canonical names (for example ``value`` and ``batch_id``)
    to source column/property names.  No pseudonymization key belongs in this
    object; the key is supplied only to :func:`ingest_internal_data` at runtime.
    """

    columns: Mapping[str, str]
    assay_registry: Mapping[str, Mapping[str, Any]]
    endpoint_registry: Mapping[str, Mapping[str, Any]]
    defaults: Mapping[str, Any] = field(default_factory=dict)
    required_fields: Sequence[str] = (
        "smiles",
        "value",
        "units",
        "relation",
        "endpoint",
        "batch_id",
        "assay_id",
    )
    require_registered_assay: bool = True
    require_registered_endpoint: bool = True
    require_rdkit: bool = True
    strip_salts: bool = True
    canonicalize_tautomer: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.columns, Mapping):
            raise TypeError("columns must be a mapping")
        if not isinstance(self.assay_registry, Mapping):
            raise TypeError("assay_registry must be a mapping")
        if not isinstance(self.endpoint_registry, Mapping):
            raise TypeError("endpoint_registry must be a mapping")
        if not isinstance(self.defaults, Mapping):
            raise TypeError("defaults must be a mapping")
        if isinstance(self.required_fields, (str, bytes)):
            raise TypeError("required_fields must be a sequence of canonical field names")
        unknown = set(self.columns) - CANONICAL_INPUT_FIELDS
        if unknown:
            raise ValueError(f"Unknown canonical column mappings: {sorted(unknown)}")
        required = set(self.required_fields)
        unknown_required = required - CANONICAL_INPUT_FIELDS
        if unknown_required:
            raise ValueError(f"Unknown required fields: {sorted(unknown_required)}")
        forbidden = _forbidden_key_paths(
            {
                "defaults": self.defaults,
                "assay_registry": self.assay_registry,
                "endpoint_registry": self.endpoint_registry,
            }
        )
        if forbidden:
            raise ValueError("Secret material must be supplied at runtime, not stored in configuration")
        _normalized_registry(self.assay_registry)
        _normalized_registry(self.endpoint_registry)

    def fingerprint(self) -> str:
        """Return a deterministic digest of non-secret intake configuration."""

        payload = asdict(self)
        return _sha256_bytes(_canonical_json(payload).encode("utf-8"))


@dataclass(frozen=True)
class InternalIntakeResult:
    """Validated rows, quarantined rows, issues, and deterministic metadata."""

    accepted: pd.DataFrame
    quarantine: pd.DataFrame
    issues: pd.DataFrame
    summary: Mapping[str, Any]
    output_paths: Mapping[str, Path] = field(default_factory=dict)


def load_internal_data_config(
    source: str | os.PathLike[str] | Mapping[str, Any] | InternalDataConfig,
) -> InternalDataConfig:
    """Load and validate an intake configuration without accepting secrets."""

    if isinstance(source, InternalDataConfig):
        return source
    if isinstance(source, Mapping):
        payload = dict(source)
    else:
        path = Path(source)
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(payload, Mapping):
            raise TypeError("Internal data configuration must contain a YAML mapping")
        payload = dict(payload)

    forbidden = _forbidden_key_paths(payload)
    if forbidden:
        names = ", ".join(sorted(forbidden))
        raise ValueError(f"Secret material must be supplied at runtime, not stored in configuration: {names}")
    allowed = {field.name for field in InternalDataConfig.__dataclass_fields__.values()}
    unknown = set(payload) - allowed
    if unknown:
        raise ValueError(f"Unknown internal data configuration keys: {sorted(unknown)}")
    return InternalDataConfig(**payload)


def _secret_bytes(secret: str | bytes) -> bytes:
    if isinstance(secret, str):
        encoded = secret.encode("utf-8")
    elif isinstance(secret, bytes):
        encoded = secret
    else:
        raise TypeError("pseudonymization_key must be text or bytes")
    if len(encoded) < 16:
        raise ValueError("pseudonymization_key must contain at least 16 bytes")
    return encoded


def pseudonymize_identifier(
    value: object,
    *,
    pseudonymization_key: str | bytes,
    namespace: str,
    prefix: str | None = None,
    digest_length: int = 20,
) -> str:
    """Return a deterministic, domain-separated HMAC pseudonym.

    Empty values remain empty.  Neither the source value nor key is included in
    the result or any error message.
    """

    text = _clean_text(value)
    if not text:
        return ""
    if digest_length < 16 or digest_length > 64:
        raise ValueError("digest_length must be between 16 and 64")
    key = _secret_bytes(pseudonymization_key)
    message = f"menin-internal-intake-v{INTERNAL_INTAKE_VERSION}\0{namespace}\0{text}"
    digest = hmac.new(key, message.encode("utf-8"), hashlib.sha256).hexdigest().upper()
    label = prefix or _ID_PREFIXES.get(namespace, "IID")
    return f"{label}-{digest[:digest_length]}"


def _detect_format(path: Path, input_format: str | None) -> str:
    if input_format:
        resolved = input_format.strip().casefold().lstrip(".")
    else:
        suffix = path.suffix.casefold()
        resolved = {".csv": "csv", ".tsv": "tsv", ".tab": "tsv", ".sdf": "sdf"}.get(suffix, "")
    if resolved not in SUPPORTED_INPUT_FORMATS:
        raise ValueError(
            f"Unsupported internal input format {resolved or path.suffix!r}; "
            f"expected one of {sorted(SUPPORTED_INPUT_FORMATS)}"
        )
    return resolved


def _read_sdf(path: Path) -> pd.DataFrame:
    if not rdkit_available():
        raise RuntimeError("RDKit is required to import SDF files")
    from rdkit import Chem  # Imported only at the SDF boundary.

    rows: list[dict[str, str]] = []
    supplier = Chem.SDMolSupplier(str(path), sanitize=True, removeHs=False)
    for position, molecule in enumerate(supplier, start=1):
        if molecule is None:
            rows.append(
                {
                    "__sdf_record_number": str(position),
                    "__sdf_smiles": "",
                    "__input_error": "invalid_sdf_record",
                }
            )
            continue
        row = {name: molecule.GetProp(name) for name in molecule.GetPropNames()}
        row["__sdf_record_number"] = str(position)
        row["__sdf_smiles"] = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
        row["__input_error"] = ""
        rows.append(row)
    return pd.DataFrame(rows, dtype=object).fillna("")


def read_internal_data(
    input_path: str | os.PathLike[str],
    *,
    input_format: str | None = None,
) -> tuple[pd.DataFrame, str]:
    """Read CSV, TSV, or SDF input without type inference or network access."""

    path = Path(input_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    resolved_format = _detect_format(path, input_format)
    if resolved_format == "sdf":
        return _read_sdf(path), resolved_format
    delimiter = "," if resolved_format == "csv" else "\t"
    table = pd.read_csv(
        path,
        sep=delimiter,
        dtype=str,
        keep_default_na=False,
        na_filter=False,
    )
    return table, resolved_format


def _source_series(
    table: pd.DataFrame,
    config: InternalDataConfig,
    canonical_name: str,
) -> pd.Series:
    source_name = _clean_text(config.columns.get(canonical_name, ""))
    if source_name:
        if source_name not in table.columns:
            return pd.Series("", index=table.index, dtype=object)
        return table[source_name].map(_clean_text)
    default = config.defaults.get(canonical_name, "")
    return pd.Series(_clean_text(default), index=table.index, dtype=object)


def _mapped_series(
    table: pd.DataFrame,
    config: InternalDataConfig,
    canonical_name: str,
) -> pd.Series:
    """Return only values physically present in the mapped source field."""

    source_name = _clean_text(config.columns.get(canonical_name, ""))
    if not source_name or source_name not in table.columns:
        return pd.Series("", index=table.index, dtype=object)
    return table[source_name].map(_clean_text)


def _lookup_normalized(
    registry: Mapping[str, Mapping[str, Any]],
    value: object,
) -> dict[str, Any] | None:
    text = _clean_text(value)
    if not text:
        return None
    metadata = registry.get(text.casefold())
    return None if metadata is None else dict(metadata)


def _allowed_units(metadata: Mapping[str, Any]) -> set[str]:
    raw = metadata.get("allowed_units", ())
    if isinstance(raw, str):
        values = [raw]
    elif isinstance(raw, Sequence):
        values = list(raw)
    else:
        values = []
    return {normalize_unit(value) for value in values if normalize_unit(value)}


def _issue(
    records: list[dict[str, str]],
    *,
    row_id: str,
    code: str,
    field: str,
    message: str,
    severity: str = "error",
) -> None:
    records.append(
        {
            "internal_row_id": row_id,
            "severity": severity,
            "code": code,
            "field": field,
            "message": message,
        }
    )


def _row_payload(
    submitted: Mapping[str, str],
    *,
    input_error: str,
) -> str:
    # Source identifiers are safe inside the keyed digest and never emitted.
    return _canonical_json(
        {
            "submitted": dict(sorted(submitted.items())),
            "input_error": input_error,
        }
    )


def validate_internal_table(
    table: pd.DataFrame,
    *,
    config: InternalDataConfig | Mapping[str, Any] | str | os.PathLike[str],
    pseudonymization_key: str | bytes,
    input_format: str = "dataframe",
    input_sha256: str = "",
) -> InternalIntakeResult:
    """Standardize, pseudonymize, and validate an already loaded table."""

    cfg = load_internal_data_config(config)
    key = _secret_bytes(pseudonymization_key)
    source = table.copy().fillna("").reset_index(drop=True)
    if source.columns.duplicated().any():
        raise ValueError("Internal input contains duplicate column/property names")
    normalized_assay_registry = _normalized_registry(cfg.assay_registry)
    normalized_endpoint_registry = _normalized_registry(cfg.endpoint_registry)

    missing_required_source_fields = sorted(
        canonical
        for canonical, source_name in cfg.columns.items()
        if canonical in cfg.required_fields
        and _clean_text(source_name)
        and _clean_text(source_name) not in source.columns
    )

    submitted_fields = {
        canonical: _source_series(source, cfg, canonical) for canonical in CANONICAL_INPUT_FIELDS
    }
    raw_fields = {canonical: _mapped_series(source, cfg, canonical) for canonical in CANONICAL_INPUT_FIELDS}
    if input_format == "sdf" and not submitted_fields["smiles"].astype(bool).any():
        submitted_fields["smiles"] = source.get(
            "__sdf_smiles", pd.Series("", index=source.index, dtype=object)
        ).map(_clean_text)

    base = pd.DataFrame(index=source.index)
    base["submitted_smiles"] = raw_fields["smiles"]
    base["submitted_value"] = raw_fields["value"]
    base["submitted_units"] = raw_fields["units"]
    base["submitted_relation"] = raw_fields["relation"]
    base["submitted_endpoint"] = raw_fields["endpoint"]
    base["measurement_date"] = submitted_fields["measurement_date"]
    base["replicate"] = submitted_fields["replicate"]
    base["cohort_role"] = submitted_fields["cohort_role"].map(lambda value: _clean_text(value).casefold())

    row_payloads: list[str] = []
    for index in source.index:
        submitted = {name: submitted_fields[name].loc[index] for name in sorted(CANONICAL_INPUT_FIELDS)}
        input_error = _clean_text(source.get("__input_error", pd.Series("", index=source.index)).loc[index])
        row_payloads.append(_row_payload(submitted, input_error=input_error))
    content_row_ids = [
        pseudonymize_identifier(
            payload,
            pseudonymization_key=key,
            namespace="record_content",
        )
        for payload in row_payloads
    ]
    base["internal_record_content_id"] = content_row_ids
    source_row_ids = [
        pseudonymize_identifier(value, pseudonymization_key=key, namespace="row")
        for value in submitted_fields["row_id"]
    ]
    base["internal_row_id"] = [
        source_id or content_id
        for source_id, content_id in zip(source_row_ids, content_row_ids, strict=False)
    ]
    base["row_id_basis"] = [
        "source_row_id" if source_id else "record_content" for source_id in source_row_ids
    ]
    base["internal_source_compound_id"] = [
        pseudonymize_identifier(
            value,
            pseudonymization_key=key,
            namespace="source_compound",
        )
        for value in submitted_fields["compound_id"]
    ]
    base["internal_batch_id"] = [
        pseudonymize_identifier(value, pseudonymization_key=key, namespace="batch")
        for value in submitted_fields["batch_id"]
    ]

    assay_metadata = [
        _lookup_normalized(normalized_assay_registry, value) for value in submitted_fields["assay_id"]
    ]
    base["internal_assay_id"] = [
        pseudonymize_identifier(value, pseudonymization_key=key, namespace="assay")
        for value in submitted_fields["assay_id"]
    ]

    structure_input = pd.DataFrame({"smiles": submitted_fields["smiles"]}, index=source.index)
    structures = standardize_structure_table(
        structure_input,
        strip_salts=cfg.strip_salts,
        canonicalize_tautomer=cfg.canonicalize_tautomer,
        require_rdkit=cfg.require_rdkit,
    )
    if "inchi_key" not in structures.columns:
        structures["inchi_key"] = pd.Series("", index=structures.index, dtype=object)
    for column in (
        "original_smiles",
        "canonical_smiles",
        "standardized_smiles",
        "standard_inchi_key",
        "structure_id",
        "full_structure_id",
        "structure_valid",
        "structure_standardization_status",
        "structure_error",
        "structure_standardization_version",
        "rdkit_version",
        "fragment_count",
        "formal_charge",
    ):
        base[column] = structures[column]
    base["smiles"] = structures["smiles"]
    base["inchi_key"] = structures["inchi_key"]
    base["internal_compound_id"] = [
        pseudonymize_identifier(value, pseudonymization_key=key, namespace="compound")
        for value in structures["structure_id"]
    ]

    issue_records: list[dict[str, str]] = []
    endpoint_values: list[str] = []
    endpoint_families: list[str] = []
    endpoint_sources: list[str] = []
    target_names: list[str] = []
    target_ids: list[str] = []
    assay_families: list[str] = []
    unit_values: list[str] = []
    unit_sources: list[str] = []
    relation_values: list[str] = []
    relation_sources: list[str] = []
    unit_status_values: list[str] = []
    numeric_values: list[float] = []
    value_nm_values: list[float] = []

    for position, index in enumerate(source.index):
        row_id = base.loc[index, "internal_row_id"]
        for missing_field in missing_required_source_fields:
            _issue(
                issue_records,
                row_id=row_id,
                code="missing_source_column",
                field=missing_field,
                message="A configured required source column is absent from the input schema",
            )
        assay = assay_metadata[position]
        submitted_endpoint = submitted_fields["endpoint"].loc[index]
        raw_endpoint = raw_fields["endpoint"].loc[index]
        endpoint_meta = _lookup_normalized(normalized_endpoint_registry, submitted_endpoint)
        assay_endpoint = _clean_text((assay or {}).get("endpoint", ""))

        if submitted_endpoint:
            if endpoint_meta is None and cfg.require_registered_endpoint:
                _issue(
                    issue_records,
                    row_id=row_id,
                    code="unknown_endpoint",
                    field="endpoint",
                    message="Submitted endpoint is absent from the configured endpoint registry",
                )
                resolved_endpoint = ""
            else:
                resolved_endpoint = _clean_text(
                    (endpoint_meta or {}).get("canonical_name", submitted_endpoint)
                )
        else:
            resolved_endpoint = assay_endpoint
            endpoint_meta = _lookup_normalized(normalized_endpoint_registry, resolved_endpoint)

        if raw_endpoint:
            resolved_endpoint_source = "submitted"
        elif submitted_endpoint:
            resolved_endpoint_source = "configuration_default"
        elif assay_endpoint:
            resolved_endpoint_source = "assay_registry"
        else:
            resolved_endpoint_source = ""

        if (
            resolved_endpoint
            and endpoint_meta is None
            and cfg.require_registered_endpoint
            and not submitted_endpoint
        ):
            _issue(
                issue_records,
                row_id=row_id,
                code="unknown_endpoint",
                field="endpoint",
                message="Configured assay endpoint is absent from the endpoint registry",
            )

        if not resolved_endpoint:
            _issue(
                issue_records,
                row_id=row_id,
                code="missing_endpoint",
                field="endpoint",
                message="Endpoint was not submitted and is not defined by the assay registry",
            )
        if assay_endpoint and resolved_endpoint and assay_endpoint.casefold() != resolved_endpoint.casefold():
            _issue(
                issue_records,
                row_id=row_id,
                code="assay_endpoint_conflict",
                field="endpoint",
                message="Submitted endpoint conflicts with configured assay metadata",
            )

        submitted_assay = submitted_fields["assay_id"].loc[index]
        if not submitted_assay:
            _issue(
                issue_records,
                row_id=row_id,
                code="missing_assay_id",
                field="assay_id",
                message="A source assay identifier is required",
            )
        elif assay is None and cfg.require_registered_assay:
            _issue(
                issue_records,
                row_id=row_id,
                code="unknown_assay",
                field="assay_id",
                message="Submitted assay is absent from the configured assay registry",
            )

        submitted_batch = submitted_fields["batch_id"].loc[index]
        if not submitted_batch:
            _issue(
                issue_records,
                row_id=row_id,
                code="missing_batch_id",
                field="batch_id",
                message="A source batch identifier is required",
            )

        cohort_role = _clean_text(submitted_fields["cohort_role"].loc[index]).casefold()
        if cohort_role and cohort_role not in ALLOWED_COHORT_ROLES:
            _issue(
                issue_records,
                row_id=row_id,
                code="invalid_cohort_role",
                field="cohort_role",
                message=("Cohort role must be development, locked_external, or prospective_blind"),
            )

        already_validated_required = {
            "smiles",
            "value",
            "units",
            "relation",
            "endpoint",
            "batch_id",
            "assay_id",
        }
        for required_field in sorted(set(cfg.required_fields) - already_validated_required):
            if not submitted_fields[required_field].loc[index]:
                _issue(
                    issue_records,
                    row_id=row_id,
                    code="missing_required_field",
                    field=required_field,
                    message="A configured required field is missing",
                )

        structure_status = _clean_text(base.loc[index, "structure_standardization_status"])
        if structure_status != "standardized":
            _issue(
                issue_records,
                row_id=row_id,
                code="invalid_structure",
                field="smiles",
                message="Structure could not be validated and standardized",
            )

        input_error = _clean_text(source.get("__input_error", pd.Series("", index=source.index)).loc[index])
        if input_error:
            _issue(
                issue_records,
                row_id=row_id,
                code=input_error,
                field="input_record",
                message="Input record could not be parsed",
            )

        raw_value = submitted_fields["value"].loc[index]
        numeric_value = parse_numeric(raw_value)
        if not np.isfinite(numeric_value):
            _issue(
                issue_records,
                row_id=row_id,
                code="invalid_value",
                field="value",
                message="Measurement value is missing or non-numeric",
            )
        elif numeric_value <= 0:
            _issue(
                issue_records,
                row_id=row_id,
                code="nonpositive_value",
                field="value",
                message="Concentration measurements must be positive",
            )

        submitted_unit = submitted_fields["units"].loc[index]
        raw_unit = raw_fields["units"].loc[index]
        assay_unit = _clean_text((assay or {}).get("units", ""))
        if raw_unit:
            resolved_unit = raw_unit
            unit_source = "submitted"
        elif submitted_unit:
            resolved_unit = submitted_unit
            unit_source = "configuration_default"
        elif assay_unit:
            resolved_unit = assay_unit
            unit_source = "assay_registry"
        else:
            resolved_unit = ""
            unit_source = ""
        factor = unit_factor_to_nm(resolved_unit)
        if not resolved_unit:
            _issue(
                issue_records,
                row_id=row_id,
                code="missing_unit",
                field="units",
                message="Units are not submitted or explicitly defined by the assay registry",
            )
        elif not np.isfinite(factor):
            _issue(
                issue_records,
                row_id=row_id,
                code="unsupported_unit",
                field="units",
                message="Submitted unit is not a supported molar concentration unit",
            )
        if not resolved_unit:
            unit_status = "missing_unit"
        elif not np.isfinite(factor):
            unit_status = "unsupported_unit"
        else:
            unit_status = "converted"

        endpoint_allowed = _allowed_units(endpoint_meta or {})
        assay_allowed = _allowed_units(assay or {})
        if endpoint_allowed and normalize_unit(resolved_unit) not in endpoint_allowed:
            _issue(
                issue_records,
                row_id=row_id,
                code="unit_not_allowed_for_endpoint",
                field="units",
                message="Unit is not listed for the configured endpoint",
            )
        if assay_allowed and normalize_unit(resolved_unit) not in assay_allowed:
            _issue(
                issue_records,
                row_id=row_id,
                code="unit_not_allowed_for_assay",
                field="units",
                message="Unit is not listed for the configured assay",
            )

        configured_relation = submitted_fields["relation"].loc[index]
        raw_relation = raw_fields["relation"].loc[index]
        resolved_relation = normalize_relation(configured_relation)
        if raw_relation:
            resolved_relation_source = "submitted"
        elif configured_relation:
            resolved_relation_source = "configuration_default"
        else:
            resolved_relation_source = ""
        if not resolved_relation:
            _issue(
                issue_records,
                row_id=row_id,
                code="missing_relation",
                field="relation",
                message="Measurement relation is missing",
            )
        elif resolved_relation not in VALID_RELATIONS:
            _issue(
                issue_records,
                row_id=row_id,
                code="unsupported_relation",
                field="relation",
                message="Measurement relation is not supported",
            )

        row_target_name = submitted_fields["target_name"].loc[index]
        row_target_id = submitted_fields["target_id"].loc[index]
        assay_target_name = _clean_text((assay or {}).get("target_name", ""))
        assay_target_id = _clean_text((assay or {}).get("target_id", ""))
        target_name = assay_target_name or row_target_name
        target_id = assay_target_id or row_target_id
        if not target_name and not target_id:
            _issue(
                issue_records,
                row_id=row_id,
                code="missing_target_metadata",
                field="target",
                message="Target name or identifier must be explicitly configured",
            )
        if (
            assay_target_name
            and row_target_name
            and assay_target_name.casefold() != row_target_name.casefold()
        ):
            _issue(
                issue_records,
                row_id=row_id,
                code="assay_target_conflict",
                field="target_name",
                message="Submitted target name conflicts with configured assay metadata",
            )
        if assay_target_id and row_target_id and assay_target_id.casefold() != row_target_id.casefold():
            _issue(
                issue_records,
                row_id=row_id,
                code="assay_target_conflict",
                field="target_id",
                message="Submitted target identifier conflicts with configured assay metadata",
            )

        resolved_assay_family = (
            _clean_text((assay or {}).get("assay_family", "")) or submitted_fields["assay_family"].loc[index]
        )
        if not resolved_assay_family:
            _issue(
                issue_records,
                row_id=row_id,
                code="missing_assay_family",
                field="assay_family",
                message="Assay family must be explicitly configured",
            )

        endpoint_values.append(resolved_endpoint)
        endpoint_families.append(
            _clean_text((endpoint_meta or {}).get("family", "")) or endpoint_family(resolved_endpoint)
        )
        endpoint_sources.append(resolved_endpoint_source)
        target_names.append(target_name)
        target_ids.append(target_id)
        assay_families.append(resolved_assay_family)
        unit_values.append(normalize_unit(resolved_unit))
        unit_sources.append(unit_source)
        unit_status_values.append(unit_status)
        relation_values.append(resolved_relation)
        relation_sources.append(resolved_relation_source)
        numeric_values.append(float(numeric_value) if np.isfinite(numeric_value) else np.nan)
        value_nm_values.append(
            float(numeric_value * factor)
            if np.isfinite(numeric_value) and numeric_value > 0 and np.isfinite(factor)
            else np.nan
        )

    base["endpoint"] = endpoint_values
    base["endpoint_family"] = endpoint_families
    base["endpoint_source"] = endpoint_sources
    base["target_name"] = target_names
    base["target_id"] = target_ids
    base["assay_family"] = assay_families
    base["standard_units"] = unit_values
    base["unit_source"] = unit_sources
    base["unit_conversion_status"] = unit_status_values
    base["relation"] = relation_values
    base["relation_source"] = relation_sources
    base["value"] = numeric_values
    base["value_nm"] = value_nm_values
    base["p_value"] = [p_value_from_nm(value) for value in value_nm_values]
    base["is_censored"] = base["relation"].isin({"<", "<=", ">", ">="})
    base["lower_bound_nm"] = np.where(base["relation"].isin({">", ">=", "="}), base["value_nm"], np.nan)
    base["upper_bound_nm"] = np.where(base["relation"].isin({"<", "<=", "="}), base["value_nm"], np.nan)
    base["source"] = "InternalLab"
    base["intake_version"] = INTERNAL_INTAKE_VERSION

    row_payload_series = pd.Series(row_payloads, index=base.index)
    duplicated_ids = base.loc[base["internal_row_id"].duplicated(keep=False), "internal_row_id"]
    for row_id in sorted(set(duplicated_ids)):
        matching_payloads = row_payload_series.loc[base["internal_row_id"].eq(row_id)]
        code = "duplicate_record" if matching_payloads.nunique() == 1 else "source_row_id_conflict"
        message = (
            "An identical submitted record occurs more than once"
            if code == "duplicate_record"
            else "One source row identifier maps to conflicting submitted records"
        )
        _issue(
            issue_records,
            row_id=row_id,
            code=code,
            field="internal_row_id",
            message=message,
        )

    source_compound_ids = base["internal_source_compound_id"].replace("", np.nan)
    for _, group in base.loc[source_compound_ids.notna()].groupby("internal_source_compound_id", sort=True):
        if group["structure_id"].replace("", np.nan).nunique() <= 1:
            continue
        for row_id in sorted(set(group["internal_row_id"])):
            _issue(
                issue_records,
                row_id=row_id,
                code="compound_structure_conflict",
                field="compound_id",
                message="One source compound identifier maps to multiple standardized structures",
            )

    issues = pd.DataFrame(
        issue_records,
        columns=["internal_row_id", "severity", "code", "field", "message"],
    ).sort_values(["internal_row_id", "severity", "code", "field"], kind="stable", ignore_index=True)
    errors_by_row: dict[str, list[str]] = {}
    for row_id, group in issues.loc[issues["severity"].eq("error")].groupby("internal_row_id", sort=True):
        errors_by_row[str(row_id)] = sorted(set(group["code"].astype(str)))
    base["validation_codes"] = ["|".join(errors_by_row.get(row_id, [])) for row_id in base["internal_row_id"]]
    base["validation_status"] = np.where(base["validation_codes"].eq(""), "accepted", "quarantined")
    # Common long-table aliases support direct integration with curation and
    # provenance layers without exposing source identifiers.
    base["source_record_id"] = base["internal_row_id"]
    base["compound_id"] = base["internal_compound_id"]
    base["batch_id"] = base["internal_batch_id"]
    base["assay_id"] = base["internal_assay_id"]
    base["value_raw"] = base["submitted_value"]
    base["quality_eligible"] = base["validation_status"].eq("accepted")

    # Stable column and row ordering makes output byte-for-byte reproducible.
    identifier_columns = [
        "internal_row_id",
        "internal_record_content_id",
        "internal_compound_id",
        "internal_source_compound_id",
        "internal_batch_id",
        "internal_assay_id",
        "structure_id",
        "full_structure_id",
    ]
    ordered_columns = identifier_columns + [
        column for column in base.columns if column not in identifier_columns
    ]
    base = base[ordered_columns]
    stable_sort_columns = ["internal_row_id", "internal_record_content_id"]
    accepted = base.loc[base["validation_status"].eq("accepted")].sort_values(
        stable_sort_columns, kind="stable", ignore_index=True
    )
    quarantine = base.loc[base["validation_status"].eq("quarantined")].sort_values(
        stable_sort_columns, kind="stable", ignore_index=True
    )

    issue_counts = {
        str(code): int(count) for code, count in issues["code"].value_counts().sort_index().items()
    }
    summary: dict[str, Any] = {
        "internal_intake_version": INTERNAL_INTAKE_VERSION,
        "input_format": input_format,
        "input_sha256": input_sha256,
        "configuration_sha256": cfg.fingerprint(),
        "standardization_version": standardization_version(
            strip_salts=cfg.strip_salts,
            canonicalize_tautomer=cfg.canonicalize_tautomer,
        ),
        "rdkit_required": cfg.require_rdkit,
        "rows_received": int(len(base)),
        "rows_accepted": int(len(accepted)),
        "rows_quarantined": int(len(quarantine)),
        "missing_required_source_fields": missing_required_source_fields,
        "unique_internal_compounds": int(accepted["internal_compound_id"].replace("", np.nan).nunique()),
        "unique_internal_batches": int(accepted["internal_batch_id"].replace("", np.nan).nunique()),
        "unique_internal_assays": int(accepted["internal_assay_id"].replace("", np.nan).nunique()),
        "cohort_role_counts": {
            str(role): int(count)
            for role, count in accepted["cohort_role"].value_counts().sort_index().items()
        },
        "issue_counts": issue_counts,
    }
    return InternalIntakeResult(
        accepted=accepted,
        quarantine=quarantine,
        issues=issues,
        summary=summary,
    )


def _atomic_write_csv(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            table.to_csv(handle, index=False, lineterminator="\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        path.chmod(0o600)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_write_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        path.chmod(0o600)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def write_internal_intake_outputs(
    result: InternalIntakeResult,
    output_directory: str | os.PathLike[str],
) -> Mapping[str, Path]:
    """Atomically write deterministic, provenance-manifest-compatible outputs."""

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    output.chmod(0o700)
    paths = {
        "accepted": output / "internal_measurements.csv",
        "quarantine": output / "internal_quarantine.csv",
        "issues": output / "internal_validation_issues.csv",
        "summary": output / "internal_validation_summary.json",
    }
    _atomic_write_csv(result.accepted, paths["accepted"])
    _atomic_write_csv(result.quarantine, paths["quarantine"])
    _atomic_write_csv(result.issues, paths["issues"])
    _atomic_write_json(result.summary, paths["summary"])
    return paths


def ingest_internal_data(
    input_path: str | os.PathLike[str],
    *,
    config: InternalDataConfig | Mapping[str, Any] | str | os.PathLike[str],
    pseudonymization_key: str | bytes,
    output_directory: str | os.PathLike[str] | None = None,
    input_format: str | None = None,
) -> InternalIntakeResult:
    """Run the complete offline internal-data intake workflow.

    The secret is runtime-only and is never persisted, logged, or included in
    returned metadata.  When ``output_directory`` is supplied, all four output
    artifacts are written atomically.
    """

    path = Path(input_path)
    table, resolved_format = read_internal_data(path, input_format=input_format)
    result = validate_internal_table(
        table,
        config=config,
        pseudonymization_key=pseudonymization_key,
        input_format=resolved_format,
        input_sha256=_sha256_file(path),
    )
    if output_directory is None:
        return result
    paths = write_internal_intake_outputs(result, output_directory)
    return InternalIntakeResult(
        accepted=result.accepted,
        quarantine=result.quarantine,
        issues=result.issues,
        summary=result.summary,
        output_paths=paths,
    )


__all__ = [
    "INTERNAL_INTAKE_VERSION",
    "InternalDataConfig",
    "InternalIntakeResult",
    "SUPPORTED_INPUT_FORMATS",
    "ingest_internal_data",
    "load_internal_data_config",
    "pseudonymize_identifier",
    "read_internal_data",
    "validate_internal_table",
    "write_internal_intake_outputs",
]
