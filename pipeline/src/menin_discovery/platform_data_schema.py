"""Canonical public-evidence contract for the protein--molecule platform.

The historical project tables were designed around Menin and hERG.  This
module deliberately defines a separate, protein-agnostic namespace.  The
contract is assertion centred: an experimental observation, a curated label,
and a computational prediction are different observation kinds.  The schema
can represent each kind, while the public experimental-data build implemented
by :mod:`menin_discovery.platform_data_pipeline` rejects prediction rows.

Identifiers are content-derived and contain no local paths or row numbers
that can change when files are moved.  Missing context remains missing; in
particular, absence of a trial link never implies a preclinical development
stage.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa

SCHEMA_VERSION = "platform-evidence-1.0.0"
IDENTIFIER_VERSION = "platform-stable-id-1"

ACCESS_CLASSES = frozenset(
    {
        "public_redistributable",
        "public_access_restricted",
        "licensed",
        "confidential",
        "unknown",
    }
)
EVIDENCE_DOMAINS = frozenset({"binding", "pk_adme", "herg", "qt", "clinical", "other"})
EVIDENCE_STAGES = frozenset(
    {
        "preclinical_in_vitro",
        "preclinical_ex_vivo",
        "preclinical_in_vivo",
        "clinical_registry",
        "clinical_results",
        "regulatory_label",
        "postmarketing",
    }
)
DEVELOPMENT_STAGES = frozenset(
    {
        "discovery",
        "explicit_preclinical",
        "ind_enabling",
        "phase_1",
        "phase_2",
        "phase_3",
        "phase_4",
        "approved",
        "withdrawn",
        "unknown",
    }
)
RESULT_STATUSES = frozenset(
    {
        "reported",
        "not_reported",
        "pending",
        "terminated",
        "withdrawn",
        "not_applicable",
    }
)
QUALITY_GRADES = frozenset(
    {
        "raw",
        "parsable",
        "identity_resolved",
        "protocol_sufficient",
        "gold",
        "quarantined",
    }
)
OBSERVATION_KINDS = frozenset(
    {"experimental_raw", "experimental_summary", "curated_assertion", "derived", "prediction"}
)
INCLUSION_STATUSES = frozenset({"included", "review", "quarantined"})
RELATIONS = frozenset({"=", "~", "<", "<=", ">", ">=", "interval", "not_reported"})

CONCENTRATION_ENDPOINTS = frozenset(
    {
        "ac50",
        "ec50",
        "gi50",
        "ic20",
        "ic25",
        "ic50",
        "ic90",
        "kd",
        "ki",
        "km",
        "potency",
    }
)

_UNIT_TO_NM = {
    "pm": 0.001,
    "pmol/l": 0.001,
    "picomolar": 0.001,
    "nm": 1.0,
    "nmol/l": 1.0,
    "nanomolar": 1.0,
    "um": 1_000.0,
    "umol/l": 1_000.0,
    "micromolar": 1_000.0,
    "mm": 1_000_000.0,
    "mmol/l": 1_000_000.0,
    "millimolar": 1_000_000.0,
    "m": 1_000_000_000.0,
    "mol/l": 1_000_000_000.0,
    "molar": 1_000_000_000.0,
}


def clean_text(value: object) -> str:
    """Return a stripped string while treating scalar NA values as missing."""

    if value is None:
        return ""
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False
    if isinstance(missing, (bool, np.bool_)) and bool(missing):
        return ""
    return str(value).strip()


def stable_id(prefix: str, *parts: object, length: int = 24) -> str:
    """Create a deterministic identifier in a versioned namespace."""

    if not prefix or not re.fullmatch(r"[A-Z][A-Z0-9]{1,11}", prefix):
        raise ValueError("prefix must contain 2--12 uppercase alphanumeric characters")
    if length < 16 or length > 64:
        raise ValueError("identifier digest length must be between 16 and 64")
    payload = "\0".join([IDENTIFIER_VERSION, *(clean_text(part) for part in parts)])
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest().upper()
    return f"{prefix}-{digest[:length]}"


def canonical_relation(value: object) -> str:
    """Normalize a relation without inventing one for a missing token."""

    text = clean_text(value).replace("≤", "<=").replace("≥", ">=").replace("≈", "~")
    text = re.sub(r"\s+", "", text)
    aliases = {"==": "=", "=<": "<=", "=>": ">=", "range": "interval"}
    text = aliases.get(text, text)
    return text if text in RELATIONS else "not_reported"


def relation_from_value(value: object, explicit: object = "") -> str:
    """Use an explicit relation, otherwise parse a leading value qualifier."""

    explicit_text = clean_text(explicit)
    if explicit_text:
        # An unrecognized explicit qualifier carries information: it must be
        # reviewed, never replaced by invented exactness from a bare number.
        return canonical_relation(explicit_text)
    value_text = clean_text(value)
    if re.search(r"(?:\u00b1|\+\s*/\s*-)", value_text) or re.match(
        r"^\s*(?:ca\.?|circa|about|approximately)\s+", value_text, flags=re.IGNORECASE
    ):
        return "~"
    if parse_interval(value) is not None:
        return "interval"
    if _looks_like_range(value_text):
        return "not_reported"
    match = re.match(r"^\s*(<=|>=|<|>|=|~|≤|≥|≈)", value_text)
    return canonical_relation(match.group(1)) if match else "=" if value_text else "not_reported"


_NUMBER_TOKEN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_INTERVAL_PATTERN = re.compile(
    rf"^\s*(?P<lower>{_NUMBER_TOKEN})\s*(?:-|\u2013|\u2014|\bto\b)\s*"
    rf"(?P<upper>{_NUMBER_TOKEN})\s*(?:[A-Za-z%\u00b5\u03bc][A-Za-z0-9%\u00b5\u03bc/.*^_ -]*)?\s*$",
    flags=re.IGNORECASE,
)
_BETWEEN_INTERVAL_PATTERN = re.compile(
    rf"^\s*between\s+(?P<lower>{_NUMBER_TOKEN})\s+and\s+(?P<upper>{_NUMBER_TOKEN})"
    rf"\s*(?:[A-Za-z%\u00b5\u03bc][A-Za-z0-9%\u00b5\u03bc/.*^_ -]*)?\s*$",
    flags=re.IGNORECASE,
)


def parse_interval(value: object) -> tuple[float, float] | None:
    """Parse an unequivocal finite numeric interval without midpointing it.

    The parser is deliberately anchored. Text containing qualifiers, more than
    two numbers, or malformed bounds is not coerced into an interval. Reversed
    bounds are rejected rather than silently repaired; the original text
    remains authoritative in ``value_raw`` for quarantine/review.
    """

    if value is None or isinstance(value, (bool, np.bool_)):
        return None
    text = clean_text(value).replace(",", "").replace("−", "-")
    match = _INTERVAL_PATTERN.fullmatch(text) or _BETWEEN_INTERVAL_PATTERN.fullmatch(text)
    if match is None:
        return None
    try:
        lower = float(match.group("lower"))
        upper = float(match.group("upper"))
    except ValueError:
        return None
    if not (math.isfinite(lower) and math.isfinite(upper)):
        return None
    return (lower, upper) if lower <= upper else None


def _looks_like_range(text: str) -> bool:
    return bool(
        re.search(
            rf"{_NUMBER_TOKEN}\s*(?:-|\u2013|\u2014|\bto\b)\s*{_NUMBER_TOKEN}",
            clean_text(text).replace("−", "-"),
            flags=re.IGNORECASE,
        )
        or _BETWEEN_INTERVAL_PATTERN.fullmatch(clean_text(text)) is not None
    )


def parse_numeric(value: object) -> float:
    """Parse a finite numeric token while preserving qualification separately."""

    if value is None or isinstance(value, (bool, np.bool_)):
        return math.nan
    text = clean_text(value).replace(",", "").replace("−", "-")
    if _looks_like_range(text):
        return math.nan
    match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", text)
    if not match:
        return math.nan
    try:
        parsed = float(match.group(0))
    except ValueError:
        return math.nan
    return parsed if math.isfinite(parsed) else math.nan


def normalize_unit(value: object) -> str:
    """Canonicalize spelling only; never infer an absent or ambiguous unit."""

    text = clean_text(value)
    if not text:
        return ""
    text = text.replace("µ", "u").replace("μ", "u").replace("−", "-").replace("·", ".")
    text = re.sub(r"\s+", "", text).casefold()
    aliases = {
        "nanomolar": "nM",
        "nm": "nM",
        "nmol/l": "nM",
        "micromolar": "uM",
        "um": "uM",
        "umol/l": "uM",
        "picomolar": "pM",
        "pm": "pM",
        "pmol/l": "pM",
        "millimolar": "mM",
        "mm": "mM",
        "mmol/l": "mM",
        "molar": "M",
        "mol/l": "M",
        "m": "M",
        "ml.min-1.kg-1": "mL/min/kg",
        "ml/min/kg": "mL/min/kg",
        "l.kg-1": "L/kg",
        "l/kg": "L/kg",
        "ng.hr.ml-1": "ng*h/mL",
        "ng.h.ml-1": "ng*h/mL",
        "ng*h/ml": "ng*h/mL",
        "ng/ml": "ng/mL",
        "ug.ml-1": "ug/mL",
        "ug/ml": "ug/mL",
        "hr": "h",
        "hour": "h",
        "hours": "h",
        "s-1": "s^-1",
        "m-1-s-1": "M^-1*s^-1",
        "m-1.s-1": "M^-1*s^-1",
        "binary": "binary",
        "%": "%",
    }
    return aliases.get(text, text)


def concentration_to_nm(endpoint: object, value: object, unit: object) -> tuple[float, str, str]:
    """Convert a supported molar concentration endpoint to nM.

    Returns ``(canonical_value, canonical_unit, status)``.  Unsupported or
    absent units remain missing rather than being guessed.
    """

    numeric = parse_numeric(value)
    endpoint_key = re.sub(r"[^a-z0-9]+", "", clean_text(endpoint).casefold())
    if endpoint_key not in CONCENTRATION_ENDPOINTS:
        normalized = normalize_unit(unit)
        return (
            (numeric, normalized, "identity")
            if math.isfinite(numeric) and normalized
            else (
                math.nan,
                normalized,
                "missing_value_or_unit",
            )
        )
    normalized_key = normalize_unit(unit).casefold()
    factor = _UNIT_TO_NM.get(normalized_key)
    if factor is None:
        return math.nan, "", "missing_unit" if not normalized_key else "unsupported_unit"
    if not math.isfinite(numeric):
        return math.nan, "nM", "missing_or_non_numeric_value"
    return numeric * factor, "nM", "converted"


def concentration_interval_to_nm(
    endpoint: object,
    value: object,
    unit: object,
) -> tuple[float, float, str, str]:
    """Convert both bounds of a supported concentration interval to nM."""

    interval = parse_interval(value)
    endpoint_key = re.sub(r"[^a-z0-9]+", "", clean_text(endpoint).casefold())
    normalized = normalize_unit(unit)
    if interval is None:
        canonical_unit = "nM" if endpoint_key in CONCENTRATION_ENDPOINTS else normalized
        return math.nan, math.nan, canonical_unit, "invalid_interval"
    if endpoint_key not in CONCENTRATION_ENDPOINTS:
        if not normalized:
            return math.nan, math.nan, "", "missing_unit"
        return interval[0], interval[1], normalized, "identity"
    factor = _UNIT_TO_NM.get(normalized.casefold())
    if factor is None:
        status = "missing_unit" if not normalized else "unsupported_unit"
        return math.nan, math.nan, "", status
    return interval[0] * factor, interval[1] * factor, "nM", "converted"


def interval_bounds(value: object, relation: object) -> tuple[float, float]:
    """Return numeric lower/upper bounds without midpoint imputation."""

    rel = canonical_relation(relation)
    if rel == "interval":
        parsed = parse_interval(value)
        return parsed if parsed is not None else (math.nan, math.nan)
    numeric = parse_numeric(value)
    if not math.isfinite(numeric):
        return math.nan, math.nan
    if rel in {"=", "~"}:
        return numeric, numeric
    if rel in {"<", "<="}:
        return math.nan, numeric
    if rel in {">", ">="}:
        return numeric, math.nan
    return math.nan, math.nan


def p_activity_from_nm(value_nm: object) -> float:
    numeric = parse_numeric(value_nm)
    if not math.isfinite(numeric) or numeric <= 0:
        return math.nan
    return float(9.0 - math.log10(numeric))


@dataclass(frozen=True)
class FieldSpec:
    table: str
    field_name: str
    definition: str
    dtype: str
    unit: str = ""
    allowed_values: str = ""
    missing_convention: str = "blank/NA"
    origin: str = "observed"
    source_lineage: str = "source record"
    roles: str = "metadata_only"
    expected_range: str = ""
    transformation: str = "none"
    leakage_class: str = "metadata"

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


TABLE_REQUIRED_COLUMNS: dict[str, tuple[str, ...]] = {
    "sources": (
        "source_id",
        "snapshot_id",
        "source_name",
        "source_version",
        "retrieval_date_utc",
        "source_url",
        "license_name",
        "license_status",
        "access_class",
    ),
    "source_files": (
        "source_file_id",
        "source_id",
        "snapshot_id",
        "relative_path",
        "sha256",
        "size_bytes",
        "immutability_status",
    ),
    "observation_lineage": (
        "lineage_id",
        "observation_id",
        "source_id",
        "snapshot_id",
        "source_file_id",
        "lineage_role",
    ),
    "molecules": (
        "molecule_id",
        "structure_id",
        "submitted_smiles",
        "standardized_smiles",
        "standard_inchi_key",
        "standardization_version",
        "identity_resolution_status",
    ),
    "molecule_aliases": (
        "molecule_alias_id",
        "molecule_id",
        "source_id",
        "source_compound_id",
        "source_record_id",
    ),
    "proteins": (
        "protein_id",
        "entity_type",
        "canonical_target_id",
        "target_name",
        "uniprot_accession",
        "sequence",
        "species",
        "identity_resolution_status",
    ),
    "protein_constructs": (
        "construct_id",
        "protein_id",
        "sequence",
        "sequence_sha256",
        "source_id",
        "source_record_id",
    ),
    "assays": (
        "assay_id",
        "source_id",
        "snapshot_id",
        "source_assay_id",
        "protein_id",
        "assay_type",
        "assay_family",
        "description",
        "protocol_completeness",
    ),
    "observations": (
        "observation_id",
        "source_id",
        "snapshot_id",
        "source_record_id",
        "molecule_id",
        "protein_id",
        "assay_id",
        "evidence_domain",
        "endpoint",
        "relation",
        "value_raw",
        "value_numeric",
        "original_unit",
        "canonical_value",
        "canonical_unit",
        "lower_bound",
        "upper_bound",
        "observation_kind",
        "evidence_stage",
        "development_stage",
        "result_status",
        "quality_grade",
        "access_class",
        "inclusion_status",
    ),
    "tasks": (
        "task_id",
        "task_type",
        "observation_id",
        "molecule_id",
        "protein_id",
        "assay_id",
        "source_id",
        "snapshot_id",
        "source_record_id",
        "label_kind",
        "label_value",
        "label_relation",
        "label_lower_bound",
        "label_upper_bound",
        "label_unit",
        "observation_kind",
        "access_class",
        "default_task_eligible",
        "required_modalities",
    ),
}


_SOURCE_REPORTED_FIELDS = frozenset(
    {
        "source_name",
        "source_version",
        "source_url",
        "citation",
        "license_name",
        "license_url",
        "source_record_scope",
        "relative_path",
        "origin_path",
        "source_compound_id",
        "source_record_id",
        "source_assay_id",
        "compound_name",
        "submitted_smiles",
        "submitted_inchi_key",
        "canonical_target_id",
        "target_name",
        "gene_symbol",
        "uniprot_accession",
        "sequence",
        "isoform",
        "species",
        "construct_description",
        "assay_type",
        "assay_format",
        "description",
        "organism",
        "cell_system",
        "matrix",
        "route",
        "temperature_c",
        "ph",
        "value_raw",
        "label_text",
        "original_unit",
        "document_id",
        "document_year",
    }
)

_CALCULATED_FIELDS: dict[str, tuple[str, str]] = {
    "query_json": ("canonical JSON serialization with sorted keys", "source query/snapshot configuration"),
    "source_id": ("versioned SHA-256 over authoritative source name and release", "source/release metadata"),
    "snapshot_id": (
        "versioned SHA-256 over source release and immutable query/snapshot scope",
        "snapshot manifest",
    ),
    "source_file_id": ("versioned SHA-256 over raw-file content SHA-256", "raw snapshot bytes"),
    "lineage_id": (
        "versioned SHA-256 over observation and source-file identities",
        "canonical observation and raw-file manifest",
    ),
    "sha256": ("SHA-256 over unchanged file bytes", "raw snapshot bytes"),
    "size_bytes": ("filesystem byte count after hash-stable read", "raw snapshot file"),
    "row_count": ("parsed physical data-line count excluding the header", "raw snapshot file"),
    "media_type": ("MIME inference from the preserved filename suffix", "raw snapshot file"),
    "molecule_id": (
        "versioned SHA-256 over standard InChIKey or standardized structure namespace",
        "molecule source assertion(s)",
    ),
    "structure_id": (
        "versioned SHA-256 over standardized parent SMILES and policy version",
        "submitted molecular structure",
    ),
    "full_structure_id": (
        "versioned SHA-256 over cleaned full submitted structure",
        "submitted molecular structure",
    ),
    "canonical_smiles": ("RDKit cleanup plus canonical isomeric SMILES serialization", "submitted_smiles"),
    "standardized_smiles": (
        "RDKit cleanup, configured fragment-parent selection and neutralization; no tautomer collapse",
        "submitted_smiles",
    ),
    "standard_inchi_key": ("RDKit/InChI standard InChIKey from standardized parent", "standardized_smiles"),
    "formal_charge": ("sum of RDKit atom formal charges after standardization", "standardized_smiles"),
    "fragment_count": ("RDKit disconnected-fragment count before parent selection", "submitted_smiles"),
    "source_count": ("count distinct contributing source_id values", "molecule aliases"),
    "molecule_alias_id": (
        "versioned SHA-256 over source, source compound and record identifiers",
        "molecule source assertion",
    ),
    "protein_id": (
        "versioned SHA-256 over accession-aware protein or complex identity",
        "source target assertion and explicit accession mapping",
    ),
    "sequence_sha256": (
        "SHA-256 over uppercase whitespace-free amino-acid sequence",
        "source-reported sequence",
    ),
    "component_protein_ids": (
        "sorted semicolon join of resolved component protein identifiers",
        "source target components",
    ),
    "construct_id": (
        "versioned SHA-256 over protein identity and exact sequence/construct annotation",
        "source-reported construct",
    ),
    "assay_id": ("versioned SHA-256 over source release and source assay identity", "source assay record"),
    "assay_family": (
        "conservative rule mapping from endpoint, assay type, format and description",
        "source assay metadata",
    ),
    "protocol_completeness": (
        "reported core protocol fields divided by declared core protocol fields",
        "source assay metadata",
    ),
    "protocol_missing_fields": (
        "sorted semicolon join of absent declared protocol fields",
        "source assay metadata",
    ),
    "observation_id": (
        "versioned SHA-256 over source release, source record and endpoint assertion",
        "source observation assertion",
    ),
    "raw_file_ids": (
        "sorted semicolon join of raw source-file identifiers",
        "raw-file lineage for the source observation",
    ),
    "evidence_domain": (
        "non-pooling endpoint/target/context classification rules",
        "source target, endpoint and assay metadata",
    ),
    "endpoint_family": (
        "conservative endpoint-name mapping without equivalence conversion",
        "source endpoint",
    ),
    "relation": (
        "Unicode/alias normalization or leading value qualifier extraction",
        "source relation and value fields",
    ),
    "value_numeric": ("finite numeric-token parsing from value_raw", "value_raw"),
    "canonical_value": (
        "endpoint-aware unit conversion; molar concentration endpoints converted to nM",
        "value_numeric and original_unit",
    ),
    "canonical_unit": (
        "endpoint-aware unit spelling/conversion; blank if unsupported",
        "original_unit and endpoint",
    ),
    "lower_bound": (
        "relation-preserving canonical bound; no midpoint or cap imputation",
        "canonical_value and relation",
    ),
    "upper_bound": (
        "relation-preserving canonical bound; no midpoint or cap imputation",
        "canonical_value and relation",
    ),
    "p_activity": (
        "9 - log10(canonical_value in nM), only for positive supported molar endpoints",
        "canonical_value, canonical_unit and endpoint",
    ),
    "p_activity_relation": (
        "monotone-decreasing inversion of the source concentration relation",
        "relation and p_activity",
    ),
    "unit_conversion_status": ("deterministic conversion result classification", "endpoint, value and unit"),
    "value_provenance": (
        "deterministic source-field/formula lineage annotation",
        "source value fields and transformation log",
    ),
    "evidence_stage": (
        "rule mapping only from explicit assay/study context; otherwise NA",
        "source assay/study metadata",
    ),
    "quality_grade": (
        "deterministic QC ladder independent of access and stage",
        "identity, value, unit and protocol QC",
    ),
    "inclusion_status": ("deterministic QC gate with retained reason", "canonical validation findings"),
    "exclusion_reason": (
        "sorted join of deterministic QC failure/review reasons",
        "canonical validation findings",
    ),
    "dedup_group_id": (
        "versioned SHA-256 over endpoint-aware repeated-measurement key",
        "canonical observation fields",
    ),
    "conflict_group_id": (
        "versioned SHA-256 over endpoint/identity context when conflict rule fires",
        "canonical observations",
    ),
    "cross_source_mirror": (
        "exact-value/document/identity mirror-candidate rule; never proof of sameness",
        "canonical observations across sources",
    ),
    "potential_leakage": (
        "lineage rule flags prediction/post-outcome/target-derived records",
        "observation lineage",
    ),
    "task_id": ("versioned SHA-256 over intended-use task and observation identity", "canonical observation"),
    "task_type": ("endpoint- and domain-specific task naming rule", "canonical observation"),
    "label_kind": ("relation/value-type mapping to exact, censored or categorical", "canonical observation"),
    "label_value": ("copy of canonical numeric target under task policy", "canonical observation"),
    "label_relation": ("copy of canonical relation under task policy", "canonical observation"),
    "label_lower_bound": ("copy of canonical lower bound under task policy", "canonical observation"),
    "label_upper_bound": ("copy of canonical upper bound under task policy", "canonical observation"),
    "label_unit": ("copy of canonical unit under task policy", "canonical observation"),
}

_INFERRED_FIELDS = frozenset(
    {
        "identity_resolution_status",
        "immutability_status",
        "resolution_method",
        "resolution_status",
        "standardization_status",
        "standardization_version",
        "entity_type",
        "stereochemistry_status",
        "quality_status",
        "development_stage",
        "result_status",
        "access_class",
        "redistribution_status",
        "license_status",
        "retrieval_status",
        "limitations",
        "observation_kind",
    }
)

_FIELD_UNITS: dict[str, str] = {
    "size_bytes": "bytes",
    "row_count": "rows",
    "formal_charge": "elementary-charge units",
    "fragment_count": "count",
    "source_count": "count",
    "temperature_c": "degrees Celsius",
    "ph": "unitless pH",
    "protocol_completeness": "fraction",
    "value_raw": "source-defined per row",
    "value_numeric": "original_unit per row",
    "original_unit": "unit label",
    "canonical_value": "canonical_unit per row",
    "lower_bound": "canonical_unit per row",
    "upper_bound": "canonical_unit per row",
    "p_activity": "-log10(mol/L)",
    "document_year": "calendar year",
    "label_value": "label_unit per row",
    "label_lower_bound": "label_unit per row",
    "label_upper_bound": "label_unit per row",
}

_FIELD_RANGES: dict[str, str] = {
    "size_bytes": "integer >= 0",
    "row_count": "integer >= 0",
    "formal_charge": "integer; chemically plausible range is QC-reviewed",
    "fragment_count": "integer >= 1 when a structure parses",
    "source_count": "integer >= 1",
    "temperature_c": ">= -273.15 when reported",
    "ph": "0..14 conventional assay QC range; values outside are flagged, not silently removed",
    "protocol_completeness": "0..1",
    "document_year": "1800..build year + 1; source exceptions flagged",
    "value_numeric": "finite; endpoint-specific sign/range QC",
    "canonical_value": "finite; positive for molar concentration endpoints",
    "lower_bound": "finite or NA for unbounded; lower <= upper when both present",
    "upper_bound": "finite or NA for unbounded; upper >= lower when both present",
    "p_activity": "finite for positive molar values; 0..14 used as QC review range",
    "label_value": "finite or NA; endpoint-specific",
    "label_lower_bound": "finite or NA",
    "label_upper_bound": "finite or NA",
}


def _field_defaults(table: str, field: str) -> dict[str, str]:
    metadata: dict[str, str] = {
        "unit": _FIELD_UNITS.get(field, "not applicable"),
        "missing_convention": "blank/NA means not reported or not applicable; never imputed",
        "origin": "observed",
        "source_lineage": "source record",
        "expected_range": _FIELD_RANGES.get(field, "source-defined or controlled vocabulary"),
        "transformation": "none",
    }
    if table == "sources":
        metadata.update(
            origin="externally_annotated", source_lineage="authoritative source/release/rights metadata"
        )
    if field in _SOURCE_REPORTED_FIELDS:
        metadata.update(origin="observed", source_lineage="unaltered or whitespace-normalized source field")
    if field in _CALCULATED_FIELDS:
        transformation, lineage = _CALCULATED_FIELDS[field]
        metadata.update(origin="calculated", transformation=transformation, source_lineage=lineage)
    if field in _INFERRED_FIELDS:
        metadata.update(
            origin="inferred",
            transformation=(
                _CALCULATED_FIELDS[field][0]
                if field in _CALCULATED_FIELDS
                else "versioned deterministic classification/review rule"
            ),
            source_lineage=(
                _CALCULATED_FIELDS[field][1]
                if field in _CALCULATED_FIELDS
                else "source metadata plus versioned platform policy"
            ),
        )
    if field.endswith("_id") and field not in {
        "source_record_id",
        "source_compound_id",
        "source_assay_id",
        "canonical_target_id",
        "document_id",
    }:
        metadata["missing_convention"] = "not permitted for the entity/assertion represented by this table"
    if table == "tasks":
        metadata.update(
            origin="calculated",
            source_lineage="canonical observation selected by a versioned task-view policy",
        )
        if metadata["transformation"] == "none":
            metadata["transformation"] = "lossless projection from canonical observation"
    return metadata


def _base_field_specs() -> list[FieldSpec]:
    specs: list[FieldSpec] = []

    def add(table: str, field: str, definition: str, dtype: str, **kwargs: str) -> None:
        metadata = _field_defaults(table, field)
        metadata.update(kwargs)
        specs.append(FieldSpec(table, field, definition, dtype, **metadata))

    add("sources", "source_id", "Stable source/version identity.", "string", roles="identifier")
    add(
        "sources",
        "snapshot_id",
        "Stable identity for one immutable retrieval snapshot.",
        "string",
        roles="identifier",
    )
    add("sources", "source_name", "Human-readable authoritative source name.", "string")
    add("sources", "source_version", "Source release/version; unresolved is explicit.", "string")
    add("sources", "retrieval_date_utc", "UTC retrieval or local snapshot date.", "datetime")
    add("sources", "source_url", "Authoritative access URL or API endpoint.", "string")
    add("sources", "query_json", "Canonical JSON query/scope specification.", "json")
    add("sources", "citation", "Preferred source citation.", "string")
    add("sources", "license_name", "Reported source license name.", "string")
    add("sources", "license_url", "Authoritative license URL.", "string")
    add("sources", "license_status", "Verification state of redistribution terms.", "string")
    add(
        "sources",
        "access_class",
        "Access axis independent of quality.",
        "category",
        allowed_values="|".join(sorted(ACCESS_CLASSES)),
    )
    add("sources", "redistribution_status", "Whether the snapshot may be redistributed.", "string")
    add("sources", "source_record_scope", "Inclusion/query scope, not a completeness claim.", "string")
    add("sources", "retrieval_status", "complete, partial, unavailable, or local_snapshot.", "category")
    add("sources", "limitations", "Known source/snapshot limitations.", "string")

    for field, definition, dtype in (
        ("source_file_id", "Content-derived source-file identifier.", "string"),
        ("source_id", "Parent source identity.", "string"),
        ("snapshot_id", "Parent snapshot identity.", "string"),
        ("relative_path", "Portable path relative to the platform raw root.", "string"),
        ("origin_path", "Original project-relative path when locally snapshotted.", "string"),
        ("sha256", "SHA-256 of unchanged bytes.", "string"),
        ("size_bytes", "File byte size.", "integer"),
        ("row_count", "Parsed data-row count where applicable.", "integer"),
        ("media_type", "Portable media type.", "string"),
        ("immutability_status", "Hash verification state.", "category"),
    ):
        add(
            "source_files",
            field,
            definition,
            dtype,
            roles="identifier" if field.endswith("_id") else "metadata_only",
        )

    for field, definition, dtype in (
        ("lineage_id", "Stable observation-to-source-file lineage edge identity.", "string"),
        ("observation_id", "Canonical observation identity.", "string"),
        ("source_id", "Source identity for this raw lineage edge.", "string"),
        ("snapshot_id", "Singular immutable snapshot identity for this edge.", "string"),
        ("source_file_id", "Immutable raw source-file identity.", "string"),
        ("lineage_role", "primary or mirrored raw assertion.", "category"),
    ):
        add(
            "observation_lineage",
            field,
            definition,
            dtype,
            roles="identifier" if field.endswith("_id") else "metadata_only",
        )

    molecule_fields = (
        ("molecule_id", "Stable standardized-parent molecule identity.", "string", "identifier"),
        ("structure_id", "Versioned standardized-structure identity.", "string", "grouping_variable"),
        ("full_structure_id", "Versioned submitted/full structure identity.", "string", "grouping_variable"),
        (
            "submitted_smiles",
            "One traceable submitted structure representation.",
            "string",
            "feature_candidate",
        ),
        (
            "canonical_smiles",
            "RDKit-cleaned isomeric canonical representation.",
            "string",
            "feature_candidate",
        ),
        (
            "standardized_smiles",
            "Configured parent/neutralized representation.",
            "string",
            "feature_candidate",
        ),
        ("standard_inchi_key", "Generated standard InChIKey when available.", "string", "identifier"),
        (
            "standardization_version",
            "Complete chemical standardization namespace.",
            "string",
            "metadata_only",
        ),
        ("standardization_status", "Validation/standardization result.", "category", "metadata_only"),
        ("identity_resolution_status", "resolved, unresolved, or conflicting.", "category", "metadata_only"),
        ("formal_charge", "Formal charge after configured standardization.", "integer", "feature_candidate"),
        ("fragment_count", "Submitted fragment count.", "integer", "feature_candidate"),
        ("source_count", "Number of represented sources.", "integer", "metadata_only"),
    )
    for field, definition, dtype, roles in molecule_fields:
        add("molecules", field, definition, dtype, roles=roles)

    for field, definition, dtype in (
        ("molecule_alias_id", "Stable source-assertion alias identity.", "string"),
        ("molecule_id", "Resolved canonical molecule identity.", "string"),
        ("source_id", "Source providing the alias.", "string"),
        ("snapshot_id", "Snapshot providing the alias.", "string"),
        ("source_compound_id", "Unmodified source compound identifier.", "string"),
        ("source_record_id", "Traceable source assertion identifier.", "string"),
        ("compound_name", "Source-reported compound name.", "string"),
        ("submitted_smiles", "Unmodified source structure text.", "string"),
        ("submitted_inchi_key", "Unmodified source InChIKey.", "string"),
        ("resolution_method", "Exact identifier/structure rule used.", "string"),
        ("resolution_status", "resolved, unresolved, or conflicting.", "category"),
    ):
        add(
            "molecule_aliases",
            field,
            definition,
            dtype,
            roles="identifier" if field.endswith("_id") else "metadata_only",
        )

    protein_fields = (
        ("protein_id", "Stable canonical protein or protein-complex identity.", "string"),
        ("entity_type", "single_protein or protein_complex.", "category"),
        ("canonical_target_id", "Preferred accession or explicit source target identity.", "string"),
        ("target_name", "Source-supported target name.", "string"),
        ("gene_symbol", "Source-supported gene symbol.", "string"),
        ("uniprot_accession", "UniProt accession; multiple components are semicolon-delimited.", "string"),
        ("sequence", "Source-reported exact sequence only.", "string"),
        ("sequence_sha256", "SHA-256 of the normalized exact sequence.", "string"),
        ("isoform", "Explicit isoform annotation.", "string"),
        ("species", "Source-reported organism.", "string"),
        ("component_protein_ids", "Sorted complex component identities.", "string"),
        ("identity_resolution_status", "resolved, source_assertion, complex, or conflicting.", "category"),
    )
    for field, definition, dtype in protein_fields:
        add(
            "proteins",
            field,
            definition,
            dtype,
            roles="identifier"
            if field.endswith("_id")
            else "feature_candidate"
            if field == "sequence"
            else "metadata_only",
        )

    for field, definition, dtype in (
        ("construct_id", "Stable identity of an exact source-reported construct sequence.", "string"),
        ("protein_id", "Parent canonical protein identity.", "string"),
        ("sequence", "Normalized exact source-reported sequence.", "string"),
        ("sequence_sha256", "SHA-256 of sequence bytes.", "string"),
        ("construct_description", "Source construct/mutation description.", "string"),
        ("source_id", "Source reporting the construct.", "string"),
        ("source_record_id", "Source target/assay/record identifier.", "string"),
        ("quality_status", "Sequence validation status.", "category"),
    ):
        add(
            "protein_constructs",
            field,
            definition,
            dtype,
            roles="identifier"
            if field.endswith("_id")
            else "feature_candidate"
            if field == "sequence"
            else "metadata_only",
        )

    assay_fields = (
        ("assay_id", "Stable source assay/protocol identity.", "string"),
        ("source_id", "Source identity.", "string"),
        ("snapshot_id", "Snapshot identity.", "string"),
        ("source_assay_id", "Unmodified source assay identifier.", "string"),
        ("protein_id", "Resolved protein/complex identity.", "string"),
        ("construct_id", "Exact construct identity when reported.", "string"),
        ("assay_type", "Source assay type code/name.", "string"),
        ("assay_family", "Conservative normalized assay family.", "category"),
        ("assay_format", "Source assay format/BAO label.", "string"),
        ("description", "Unmodified or whitespace-normalized assay description.", "string"),
        ("organism", "Assay/target organism.", "string"),
        ("cell_system", "Explicit cell system only.", "string"),
        ("matrix", "Explicit experimental matrix only.", "string"),
        ("route", "Explicit administration route only.", "string"),
        ("temperature_c", "Explicit assay temperature in degrees C.", "float"),
        ("ph", "Explicit assay pH.", "float"),
        ("protocol_completeness", "Fraction of declared core protocol fields reported.", "float"),
        ("protocol_missing_fields", "Semicolon-delimited absent core fields.", "string"),
    )
    for field, definition, dtype in assay_fields:
        add(
            "assays",
            field,
            definition,
            dtype,
            roles="identifier" if field.endswith("_id") else "grouping_variable",
        )

    observation_fields = (
        ("observation_id", "Stable identity of one source assertion.", "string", "identifier", "metadata"),
        ("source_id", "Source identity.", "string", "identifier", "metadata"),
        (
            "snapshot_id",
            "Snapshot identity or semicolon-delimited mirror snapshots.",
            "string",
            "identifier",
            "metadata",
        ),
        ("source_record_id", "Unmodified source record identity.", "string", "identifier", "metadata"),
        (
            "raw_file_ids",
            "Traceable raw file identities contributing the assertion.",
            "string",
            "metadata_only",
            "metadata",
        ),
        ("molecule_id", "Resolved canonical molecule identity.", "string", "grouping_variable", "safe_group"),
        ("protein_id", "Resolved protein/complex identity.", "string", "grouping_variable", "safe_group"),
        (
            "construct_id",
            "Exact source-reported construct identity.",
            "string",
            "grouping_variable",
            "safe_group",
        ),
        ("assay_id", "Resolved assay identity.", "string", "grouping_variable", "safe_group"),
        (
            "evidence_domain",
            "Non-pooled scientific evidence family.",
            "category",
            "grouping_variable",
            "safe_group",
        ),
        ("endpoint", "Normalized but semantically unmerged endpoint.", "string", "target", "target"),
        (
            "endpoint_family",
            "Coarse endpoint family; not a pooled target.",
            "category",
            "grouping_variable",
            "safe_group",
        ),
        ("relation", "Exact/censored/approximate qualifier.", "category", "target", "target"),
        ("value_raw", "Unmodified source value text.", "string", "target", "target"),
        (
            "label_text",
            "Source categorical label when no numeric measurement exists.",
            "string",
            "target",
            "target",
        ),
        ("value_numeric", "Parsed number in the original unit.", "float", "target", "target"),
        ("original_unit", "Unmodified source unit.", "string", "target", "target"),
        ("canonical_value", "Deterministically unit-normalized value.", "float", "target", "target"),
        (
            "canonical_unit",
            "Canonical unit; blank when conversion is unsupported.",
            "string",
            "target",
            "target",
        ),
        ("lower_bound", "Canonical lower bound; NA means unbounded/unknown.", "float", "target", "target"),
        ("upper_bound", "Canonical upper bound; NA means unbounded/unknown.", "float", "target", "target"),
        (
            "p_activity",
            "Derived -log10 molar value for supported concentration endpoints.",
            "float",
            "target",
            "target_derived",
        ),
        (
            "p_activity_relation",
            "Relation after monotone decreasing p-transform.",
            "category",
            "target",
            "target_derived",
        ),
        (
            "unit_conversion_status",
            "converted, identity, missing, or unsupported.",
            "category",
            "metadata_only",
            "metadata",
        ),
        (
            "observation_kind",
            "Raw/summary experiment, curated assertion, derivation, or prediction.",
            "category",
            "grouping_variable",
            "safe_group",
        ),
        (
            "value_provenance",
            "Source field/formula used for the value.",
            "string",
            "metadata_only",
            "metadata",
        ),
        ("document_id", "Publication/patent/document identity.", "string", "grouping_variable", "safe_group"),
        ("document_year", "Source-reported document year.", "integer", "grouping_variable", "safe_group"),
        (
            "evidence_stage",
            "Stage of the experiment, not asset development stage.",
            "category",
            "grouping_variable",
            "safe_group",
        ),
        (
            "development_stage",
            "Explicit asset stage; unknown unless directly reported.",
            "category",
            "grouping_variable",
            "safe_group",
        ),
        (
            "result_status",
            "Registry/result/regulatory evidence status.",
            "category",
            "grouping_variable",
            "safe_group",
        ),
        (
            "quality_grade",
            "Scientific/data fitness axis independent of access.",
            "category",
            "grouping_variable",
            "safe_group",
        ),
        (
            "access_class",
            "Access/legal axis independent of quality.",
            "category",
            "grouping_variable",
            "safe_group",
        ),
        (
            "inclusion_status",
            "included, review, or quarantined.",
            "category",
            "grouping_variable",
            "safe_group",
        ),
        ("exclusion_reason", "Explicit reason when not included.", "string", "metadata_only", "metadata"),
        (
            "dedup_group_id",
            "Exact/representation/repetition group identity.",
            "string",
            "grouping_variable",
            "safe_group",
        ),
        (
            "conflict_group_id",
            "Measurement conflict group identity when flagged.",
            "string",
            "grouping_variable",
            "safe_group",
        ),
        (
            "cross_source_mirror",
            "Possible mirrored assertion retained across sources.",
            "boolean",
            "metadata_only",
            "metadata",
        ),
        (
            "potential_leakage",
            "Whether the row/field lineage is target-derived or post-outcome.",
            "boolean",
            "metadata_only",
            "leakage_flag",
        ),
    )
    for field, definition, dtype, roles, leakage in observation_fields:
        kwargs: dict[str, str] = {"roles": roles, "leakage_class": leakage}
        if field == "evidence_domain":
            kwargs["allowed_values"] = "|".join(sorted(EVIDENCE_DOMAINS))
        elif field == "evidence_stage":
            kwargs["allowed_values"] = "|".join(sorted(EVIDENCE_STAGES))
        elif field == "development_stage":
            kwargs["allowed_values"] = "|".join(sorted(DEVELOPMENT_STAGES))
        elif field == "result_status":
            kwargs["allowed_values"] = "|".join(sorted(RESULT_STATUSES))
        elif field == "observation_kind":
            kwargs["allowed_values"] = "|".join(sorted(OBSERVATION_KINDS))
        elif field == "access_class":
            kwargs["allowed_values"] = "|".join(sorted(ACCESS_CLASSES))
        elif field == "quality_grade":
            kwargs["allowed_values"] = "|".join(sorted(QUALITY_GRADES))
        elif field == "relation":
            kwargs["allowed_values"] = "|".join(sorted(RELATIONS))
        add("observations", field, definition, dtype, **kwargs)

    for field, definition, dtype, roles in (
        ("task_id", "Stable intended-use task identity.", "string", "identifier"),
        (
            "task_type",
            "Endpoint-specific task; domains are never silently pooled.",
            "string",
            "grouping_variable",
        ),
        ("observation_id", "Traceback to one canonical assertion.", "string", "identifier"),
        ("molecule_id", "Molecule group identity.", "string", "grouping_variable"),
        ("protein_id", "Protein group identity.", "string", "grouping_variable"),
        ("assay_id", "Assay group identity.", "string", "grouping_variable"),
        ("source_id", "Source group identity.", "string", "grouping_variable"),
        ("snapshot_id", "Immutable source snapshot provenance.", "string", "identifier"),
        ("source_record_id", "Source assertion provenance.", "string", "identifier"),
        ("document_year", "Temporal split candidate.", "integer", "grouping_variable"),
        ("label_kind", "continuous_exact, continuous_censored, or categorical.", "category", "target"),
        ("label_value", "Numeric model target when defined.", "float", "target"),
        ("label_text", "Categorical model target when defined.", "string", "target"),
        ("label_relation", "Target qualification.", "category", "target"),
        ("label_lower_bound", "Censored lower target bound.", "float", "target"),
        ("label_upper_bound", "Censored upper target bound.", "float", "target"),
        ("label_unit", "Target unit.", "string", "target"),
        (
            "observation_kind",
            "Normative observation kind; prediction tasks are excluded from public training views.",
            "category",
            "metadata_only",
        ),
        ("quality_grade", "Quality stratum.", "category", "grouping_variable"),
        ("access_class", "Access stratum.", "category", "grouping_variable"),
        (
            "default_task_eligible",
            "Whether the task row is admitted by the conservative default policy.",
            "boolean",
            "metadata_only",
        ),
        (
            "sensitivity_task_eligible",
            "Whether an explicitly enabled sensitivity policy may admit the row.",
            "boolean",
            "metadata_only",
        ),
        (
            "required_modalities",
            "Semicolon-delimited model-input modalities required by this task contract.",
            "string",
            "metadata_only",
        ),
    ):
        add(
            "tasks",
            field,
            definition,
            dtype,
            roles=roles,
            leakage_class="target" if roles == "target" else "safe_group",
        )
    return specs


FIELD_SPECS = tuple(_base_field_specs())


def data_dictionary_frame() -> pd.DataFrame:
    """Return the complete machine-readable field dictionary."""

    return (
        pd.DataFrame([spec.to_dict() for spec in FIELD_SPECS])
        .sort_values(["table", "field_name"], kind="stable")
        .reset_index(drop=True)
    )


def schema_document() -> dict[str, Any]:
    """Return a portable JSON schema summary used by builds and adapters."""

    dictionary = data_dictionary_frame()
    tables: dict[str, Any] = {}
    for table, group in dictionary.groupby("table", sort=True):
        tables[str(table)] = {
            "required_columns": list(TABLE_REQUIRED_COLUMNS.get(str(table), ())),
            "fields": group.drop(columns="table").to_dict("records"),
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "identifier_version": IDENTIFIER_VERSION,
        "axes": {
            "access_class": sorted(ACCESS_CLASSES),
            "evidence_domain": sorted(EVIDENCE_DOMAINS),
            "evidence_stage": sorted(EVIDENCE_STAGES),
            "development_stage": sorted(DEVELOPMENT_STAGES),
            "result_status": sorted(RESULT_STATUSES),
            "quality_grade": sorted(QUALITY_GRADES),
            "observation_kind": sorted(OBSERVATION_KINDS),
        },
        "tables": tables,
        "semantic_guards": [
            "observation_kind distinguishes predictions; public experimental task views exclude them",
            "Ki, Kd, IC50, EC50, kinetics, PK, hERG, QT, and clinical outcomes remain distinct endpoints",
            "IC50 is never converted to binding free energy",
            "absence of clinical evidence never implies an explicit preclinical development stage",
            "censored values retain one-sided bounds and are never midpoint-imputed",
            "access class is independent of evidence quality",
        ],
    }


def validate_table(table: str, frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Return deterministic contract violations for a canonical table."""

    issues: list[dict[str, Any]] = []
    required = TABLE_REQUIRED_COLUMNS.get(table, ())
    for column in required:
        if column not in frame.columns:
            issues.append(
                {"table": table, "code": "missing_required_column", "column": column, "n_rows": len(frame)}
            )
    if issues or frame.empty:
        return issues

    categorical_contracts: dict[str, Mapping[str, Iterable[str]]] = {
        "sources": {"access_class": ACCESS_CLASSES},
        "observations": {
            "evidence_domain": EVIDENCE_DOMAINS,
            "evidence_stage": EVIDENCE_STAGES,
            "development_stage": DEVELOPMENT_STAGES,
            "result_status": RESULT_STATUSES,
            "quality_grade": QUALITY_GRADES,
            "access_class": ACCESS_CLASSES,
            "observation_kind": OBSERVATION_KINDS,
            "inclusion_status": INCLUSION_STATUSES,
            "relation": RELATIONS,
        },
        "tasks": {
            "observation_kind": OBSERVATION_KINDS,
            "access_class": ACCESS_CLASSES,
            "label_kind": {"continuous_exact", "continuous_censored", "categorical"},
            "label_relation": RELATIONS,
        },
    }
    allowed_columns = categorical_contracts.get(table, {})
    for column, allowed in allowed_columns.items():
        if column not in frame.columns:
            continue
        cleaned = frame[column].map(clean_text)
        values = set(cleaned[cleaned != ""]) - set(allowed)
        if values:
            issues.append(
                {
                    "table": table,
                    "code": "unexpected_category",
                    "column": column,
                    "values": sorted(values),
                    "n_rows": int(cleaned.isin(values).sum()),
                }
            )

    nonblank_required = {
        "sources": {"source_id", "snapshot_id", "source_name", "source_version", "access_class"},
        "source_files": {"source_file_id", "source_id", "snapshot_id", "relative_path", "sha256"},
        "observation_lineage": {
            "lineage_id",
            "observation_id",
            "source_id",
            "snapshot_id",
            "source_file_id",
            "lineage_role",
        },
        "molecules": {"molecule_id", "identity_resolution_status"},
        "molecule_aliases": {"molecule_alias_id", "molecule_id", "source_id", "source_record_id"},
        "proteins": {"protein_id", "entity_type", "canonical_target_id", "identity_resolution_status"},
        "assays": {"assay_id", "source_id", "snapshot_id", "source_assay_id", "protein_id"},
        "observations": {
            "observation_id",
            "source_id",
            "snapshot_id",
            "source_record_id",
            "molecule_id",
            "protein_id",
            "assay_id",
            "evidence_domain",
            "endpoint",
            "relation",
            "observation_kind",
            "development_stage",
            "result_status",
            "quality_grade",
            "access_class",
            "default_task_eligible",
            "inclusion_status",
        },
        "tasks": {
            "task_id",
            "task_type",
            "observation_id",
            "molecule_id",
            "protein_id",
            "assay_id",
            "source_id",
            "snapshot_id",
            "source_record_id",
            "label_kind",
            "label_relation",
            "observation_kind",
            "access_class",
            "required_modalities",
        },
    }.get(table, set())
    for column in sorted(nonblank_required):
        if column not in frame.columns:
            continue
        blank = frame[column].map(clean_text).eq("")
        if blank.any():
            issues.append(
                {
                    "table": table,
                    "code": "missing_required_value",
                    "column": column,
                    "n_rows": int(blank.sum()),
                }
            )

    if table == "tasks" and "default_task_eligible" in frame.columns:
        valid_boolean = frame["default_task_eligible"].map(lambda value: isinstance(value, (bool, np.bool_)))
        if not valid_boolean.all():
            issues.append(
                {
                    "table": table,
                    "code": "invalid_boolean",
                    "column": "default_task_eligible",
                    "n_rows": int((~valid_boolean).sum()),
                }
            )

    if table == "observations":
        duplicated = frame["observation_id"].duplicated(keep=False)
        if duplicated.any():
            issues.append(
                {
                    "table": table,
                    "code": "duplicate_primary_key",
                    "column": "observation_id",
                    "n_rows": int(duplicated.sum()),
                }
            )
    else:
        key_columns: tuple[str, ...]
        if table == "sources":
            # One source/version identity legitimately has multiple immutable
            # retrieval/query snapshots.
            key_columns = ("source_id", "snapshot_id")
        elif table == "tasks":
            key_columns = ("task_id", "observation_id")
        else:
            key_columns = (required[0],) if required else ()
        if key_columns and all(column in frame.columns for column in key_columns):
            duplicated = frame[list(key_columns)].astype(str).duplicated(keep=False)
            if duplicated.any():
                issues.append(
                    {
                        "table": table,
                        "code": "duplicate_primary_key",
                        "column": ";".join(key_columns),
                        "n_rows": int(duplicated.sum()),
                    }
                )
    return issues


def canonical_json(value: Any) -> str:
    """Serialize metadata deterministically for IDs, manifests, and logs."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def arrow_schema_contract(schema: pa.Schema) -> dict[str, Any]:
    """Return a portable, metadata-free Arrow field contract and digest."""

    normalized = schema.remove_metadata()
    fields = [
        {
            "name": field.name,
            "type": str(field.type),
            "nullable": bool(field.nullable),
        }
        for field in normalized
    ]
    serialized = {
        "format": "apache-arrow-field-contract-v1",
        "fields": fields,
        "schema_metadata": None,
    }
    serialized["sha256"] = hashlib.sha256(canonical_json(serialized).encode("utf-8")).hexdigest()
    return serialized


def require_arrow_schema_contract(
    manifest: Mapping[str, Any],
    expected_schema: pa.Schema,
    *,
    context: str,
) -> dict[str, Any]:
    """Fail closed unless a manifest declares the exact expected Arrow schema."""

    expected = arrow_schema_contract(expected_schema)
    if manifest.get("arrow_schema") != expected:
        raise RuntimeError(f"Arrow schema contract mismatch: {context}")
    return expected


__all__ = [
    "ACCESS_CLASSES",
    "OBSERVATION_KINDS",
    "DEVELOPMENT_STAGES",
    "EVIDENCE_DOMAINS",
    "EVIDENCE_STAGES",
    "FIELD_SPECS",
    "IDENTIFIER_VERSION",
    "INCLUSION_STATUSES",
    "QUALITY_GRADES",
    "RELATIONS",
    "RESULT_STATUSES",
    "SCHEMA_VERSION",
    "TABLE_REQUIRED_COLUMNS",
    "arrow_schema_contract",
    "canonical_json",
    "canonical_relation",
    "clean_text",
    "concentration_interval_to_nm",
    "concentration_to_nm",
    "data_dictionary_frame",
    "interval_bounds",
    "normalize_unit",
    "p_activity_from_nm",
    "parse_numeric",
    "parse_interval",
    "relation_from_value",
    "require_arrow_schema_contract",
    "schema_document",
    "stable_id",
    "validate_table",
]
