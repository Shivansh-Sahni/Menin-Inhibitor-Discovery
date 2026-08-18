"""Traceable curation utilities for activity, hERG, and PK/ADMET data.

The curation layer is deliberately conservative: missing or unrecognized units
are never guessed, assay-search results are not automatically treated as target
measurements, and censored values retain their interval semantics.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .chemistry import standardize_structure_table
from .config import BIOACTIVITY_ENDPOINTS, HERG_TARGET, MENIN_TARGET

ENDPOINT_CANONICAL = {
    "IC50": "IC50",
    "KI": "Ki",
    "KD": "Kd",
    "EC50": "EC50",
}

ENDPOINT_FAMILY = {
    "IC50": "inhibitory_potency",
    "Ki": "binding_affinity",
    "Kd": "binding_affinity",
    "EC50": "functional_potency",
}

# Keys are normalized by ``normalize_unit``.  Only molar concentration units
# belong here; mass concentration, percentage, rates, and unitless values must
# be handled by endpoint-specific code rather than coerced to nM.
_NORMALIZED_UNIT_TO_NM = {
    "pm": 0.001,
    "pmol/l": 0.001,
    "pmol/liter": 0.001,
    "picomolar": 0.001,
    "nm": 1.0,
    "nmol/l": 1.0,
    "nmol/liter": 1.0,
    "nanomolar": 1.0,
    "um": 1_000.0,
    "umol/l": 1_000.0,
    "umol/liter": 1_000.0,
    "micromolar": 1_000.0,
    "mm": 1_000_000.0,
    "mmol/l": 1_000_000.0,
    "mmol/liter": 1_000_000.0,
    "millimolar": 1_000_000.0,
    "m": 1_000_000_000.0,
    "mol/l": 1_000_000_000.0,
    "mol/liter": 1_000_000_000.0,
    "molar": 1_000_000_000.0,
}

# Backward-compatible exported constant.  Conversion itself goes through the
# stricter normalization function below.
UNIT_TO_NM = {
    "PM": 0.001,
    "PICOMOLAR": 0.001,
    "NM": 1.0,
    "NANOMOLAR": 1.0,
    "UM": 1_000.0,
    "MICROMOLAR": 1_000.0,
    "MM": 1_000_000.0,
    "MILLIMOLAR": 1_000_000.0,
    "M": 1_000_000_000.0,
}

VALID_RELATIONS = frozenset({"=", "<", "<=", ">", ">=", "~"})
EXACT_RELATIONS = frozenset({"="})
CENSORED_RELATIONS = frozenset({"<", "<=", ">", ">="})

CELLULAR_TERMS = (
    "cell proliferation",
    "cell viability",
    "cellular",
    "in cells",
    "cell line",
    "cytotoxic",
    "growth inhibition",
    "colony formation",
)
BIOPHYSICAL_TERMS = (
    "fluorescence polarization",
    "htrf",
    "alpha lisa",
    "alphalisa",
    "surface plasmon",
    "spr assay",
    "thermal shift",
    "isothermal titration",
    "protein binding",
    "interaction assay",
)
IN_VIVO_TERMS = (
    "in vivo",
    "xenograft",
    "mouse model",
    "mice",
    "administered",
    "mg/kg",
)
ELECTROPHYSIOLOGY_TERMS = (
    "patch clamp",
    "voltage clamp",
    "tail current",
    "channel activity",
    "channel blocking",
    "potassium current",
    "k+ channel",
    "herg current",
    "ikr current",
    "electrophysiolog",
)

KNOWN_TARGET_IDS = {
    MENIN_TARGET["chembl_id"].casefold(),
    MENIN_TARGET["uniprot"].casefold(),
    HERG_TARGET["chembl_id"].casefold(),
    HERG_TARGET["uniprot"].casefold(),
}


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return bool(missing) if isinstance(missing, (bool, np.bool_)) else False


def _clean_text(value: object) -> str:
    return "" if _is_missing(value) else str(value).strip()


def col(df: pd.DataFrame, name: str, default: object = "") -> pd.Series:
    """Return a column or an index-aligned default series."""

    if name in df.columns:
        return df[name]
    return pd.Series(default, index=df.index)


def _format_identifier(value: object) -> str:
    """Serialize database identifiers without introducing a ``.0`` suffix."""

    if _is_missing(value):
        return ""
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    text = str(value).strip()
    if re.fullmatch(r"[-+]?\d+\.0", text):
        return text[:-2]
    return text


def normalize_relation(value: object) -> str:
    text = _clean_text(value).replace("≤", "<=").replace("≥", ">=")
    text = re.sub(r"\s+", "", text)
    aliases = {"==": "=", "=<": "<=", "=>": ">=", "≈": "~"}
    return aliases.get(text, text)


def extract_relation(value: object, explicit: object = "") -> str:
    """Return a canonical relation, preferring an explicitly supplied value."""

    explicit_text = normalize_relation(explicit)
    if explicit_text:
        return explicit_text
    match = re.match(r"\s*([<>]=?|=|~|≤|≥|≈)", _clean_text(value))
    return normalize_relation(match.group(1)) if match else "="


def parse_numeric(value: object) -> float:
    """Parse the first numeric token from values such as ``'<10'`` or ``'>1,000'``."""

    if _is_missing(value) or isinstance(value, (bool, np.bool_)):
        return np.nan
    text = str(value).replace(",", "").strip()
    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    if not match:
        return np.nan
    try:
        return float(match.group(0))
    except ValueError:
        return np.nan


def normalize_endpoint(endpoint: object) -> str:
    text = _clean_text(endpoint)
    compact = re.sub(r"[\s_\-]", "", text).upper()
    return ENDPOINT_CANONICAL.get(compact, text)


def normalize_unit(unit: object) -> str:
    """Canonicalize spelling without inferring a missing or ambiguous unit."""

    text = _clean_text(unit)
    if not text:
        return ""
    text = text.replace("µ", "u").replace("μ", "u").replace("Μ", "u")
    text = text.replace("−", "-").replace("·", ".")
    text = re.sub(r"\s+", "", text).casefold()
    # Common equivalent concentration notations.
    text = re.sub(r"(?:\.?)l\^-?1$", "/l", text)
    text = re.sub(r"(?:\.?)l-1$", "/l", text)
    text = text.replace("/litre", "/liter")
    return text


def unit_factor_to_nm(unit: object) -> float:
    """Return an nM conversion factor, or NaN for missing/unsupported units."""

    return float(_NORMALIZED_UNIT_TO_NM.get(normalize_unit(unit), np.nan))


def unit_conversion_status(unit: object) -> str:
    normalized = normalize_unit(unit)
    if not normalized:
        return "missing_unit"
    if normalized in _NORMALIZED_UNIT_TO_NM:
        return "converted"
    return "unsupported_unit"


def p_value_from_nm(value_nm: object) -> float:
    value = parse_numeric(value_nm)
    if not np.isfinite(value) or value <= 0:
        return np.nan
    return float(9.0 - np.log10(value))


def endpoint_family(endpoint: object) -> str:
    return ENDPOINT_FAMILY.get(normalize_endpoint(endpoint), "other")


def classify_assay_family(
    endpoint: object,
    assay_type: object = "",
    description: object = "",
) -> str:
    """Assign a coarse, reviewable assay family from structured and text fields."""

    endpoint_name = normalize_endpoint(endpoint)
    assay_code = _clean_text(assay_type).upper()
    text = _clean_text(description).casefold()
    if any(term in text for term in ELECTROPHYSIOLOGY_TERMS):
        return "electrophysiology_functional"
    if assay_code == "A":
        return "adme"
    if assay_code == "P":
        return "physicochemical"
    if assay_code == "T":
        return "toxicity"
    if any(term in text for term in IN_VIVO_TERMS):
        return "in_vivo"
    if any(term in text for term in CELLULAR_TERMS):
        return "cellular_functional"
    if assay_code == "B" or endpoint_name in {"Ki", "Kd"} or any(term in text for term in BIOPHYSICAL_TERMS):
        return "biochemical_binding"
    if assay_code == "F" or endpoint_name == "EC50":
        return "functional"
    if endpoint_name == "IC50":
        return "biochemical_inhibition"
    return "unclassified"


def _known_target_match(target_name: object, target_id: object) -> bool:
    name = _clean_text(target_name).casefold()
    identifiers = {
        token.strip().casefold() for token in re.split(r"[;,|\s]+", _clean_text(target_id)) if token.strip()
    }
    if identifiers & KNOWN_TARGET_IDS:
        return True
    return bool(re.search(r"\bmenin\b|\bmen1\b|\bkcnh2\b|\bherg\b", name, flags=re.IGNORECASE))


def assess_pubchem_relevance(
    target_name: object,
    target_accession: object,
    assay_name: object,
    assay_description: object,
) -> tuple[str, bool, str]:
    """Conservatively assess whether a PubChem search result measures Menin."""

    target = _clean_text(target_name).casefold()
    accession = _clean_text(target_accession).casefold()
    assay_text = f"{_clean_text(assay_name)} {_clean_text(assay_description)}".casefold()

    known_accessions = {token.strip() for token in re.split(r"[;,|\s]+", accession) if token.strip()}
    menin_pattern = r"\bmenin\b|\bmen1\b"
    off_target_pattern = r"\blsd[- ]?1\b|\bkdm1a\b|lysine[- ]specific demethylase"

    if MENIN_TARGET["uniprot"].casefold() in known_accessions:
        return "confirmed_accession", True, "Menin UniProt accession present"
    if re.search(menin_pattern, target):
        return "confirmed_target_name", True, "explicit Menin/MEN1 target name"
    if target and (re.search(off_target_pattern, target) or not re.search(menin_pattern, target)):
        return "off_target", False, f"explicit non-Menin target: {_clean_text(target_name)}"
    if re.search(menin_pattern, assay_text):
        return "text_supported", True, "assay title/description explicitly names Menin"
    if re.search(off_target_pattern, assay_text):
        return "off_target", False, "assay text identifies an off-target assay"
    return "unresolved", False, "no explicit Menin target evidence"


def chembl_to_long(df: pd.DataFrame, source_detail: str) -> pd.DataFrame:
    """Convert ChEMBL activity rows into the common long schema."""

    if df.empty:
        return pd.DataFrame()

    target_name = col(df, "target_pref_name")
    target_id = col(df, "target_chembl_id")
    target_relevant = [
        _known_target_match(name, identifier)
        for name, identifier in zip(target_name, target_id, strict=False)
    ]
    return pd.DataFrame(
        {
            "source": "ChEMBL",
            "source_record_id": col(df, "activity_id"),
            "compound_id": col(df, "molecule_chembl_id"),
            "compound_name": col(df, "molecule_pref_name"),
            "smiles": col(df, "canonical_smiles"),
            "inchi_key": col(df, "standard_inchi_key"),
            "target_name": target_name,
            "target_id": target_id,
            "endpoint": col(df, "standard_type"),
            "relation": col(df, "standard_relation").combine(
                col(df, "standard_value"), lambda rel, val: extract_relation(val, rel)
            ),
            "value_raw": col(df, "standard_value"),
            "standard_units": col(df, "standard_units"),
            "assay_description": col(df, "assay_description"),
            "assay_type": col(df, "assay_type"),
            "assay_id": col(df, "assay_chembl_id"),
            "assay_format": col(df, "bao_label"),
            "bao_format_id": col(df, "bao_format"),
            "assay_variant_accession": col(df, "assay_variant_accession"),
            "assay_variant_mutation": col(df, "assay_variant_mutation"),
            "target_organism": col(df, "target_organism"),
            "activity_comment": col(df, "activity_comment"),
            "data_validity_comment": col(df, "data_validity_comment"),
            "data_validity_description": col(df, "data_validity_description"),
            "potential_duplicate": col(df, "potential_duplicate"),
            "standard_flag": col(df, "standard_flag"),
            "reported_pchembl_value": col(df, "pchembl_value"),
            "parent_compound_id": col(df, "parent_molecule_chembl_id"),
            "source_record_parent_id": col(df, "record_id"),
            "source_id": col(df, "src_id"),
            "document_id": col(df, "document_chembl_id"),
            "document_year": col(df, "document_year"),
            "date_provenance": "chembl_document_publication_year",
            "reference": col(df, "document_journal"),
            "source_detail": source_detail,
            "measurement_origin": "standard_value",
            "target_relevance": np.where(target_relevant, "confirmed_configured_target", "target_mismatch"),
            "is_target_relevant": target_relevant,
            "target_relevance_reason": np.where(
                target_relevant,
                "target identifier/name matches configured target",
                "target identifier/name does not match Menin or hERG",
            ),
        }
    )


def _merge_pubchem_catalog(df: pd.DataFrame, catalog: pd.DataFrame | None) -> pd.DataFrame:
    work = df.copy()
    if catalog is None or catalog.empty or "aid" not in work.columns or "aid" not in catalog.columns:
        if "assay_name" not in work.columns:
            work["assay_name"] = ""
        if "catalog_assay_description" not in work.columns:
            work["catalog_assay_description"] = ""
        return work

    available = [
        name
        for name in (
            "aid",
            "assay_name",
            "assay_description",
            "assay_source_id",
            "current_source_name",
            "activity_outcome_method",
            "search_terms",
            "n_search_terms",
            "selection_rank",
            "deposit_date",
            "modify_date",
            "curation_decision",
            "include_for_menin",
            "endpoint_override",
            "units_override",
            "assay_family_override",
        )
        if name in catalog.columns
    ]
    cat = catalog[available].drop_duplicates(subset=["aid"]).copy()
    cat = cat.rename(columns={"assay_description": "catalog_assay_description"})
    work["aid"] = pd.to_numeric(work["aid"], errors="coerce").astype("Int64")
    cat["aid"] = pd.to_numeric(cat["aid"], errors="coerce").astype("Int64")
    return work.merge(cat, on="aid", how="left", suffixes=("", "_catalog"))


def pubchem_to_long(df: pd.DataFrame, catalog: pd.DataFrame | None = None) -> pd.DataFrame:
    """Convert PubChem BioAssay rows without guessing endpoint or units."""

    if df.empty:
        return pd.DataFrame()

    work = _merge_pubchem_catalog(df, catalog)
    endpoint = col(work, "Standard Type").map(_clean_text)
    value = pd.Series(np.nan, index=work.index, dtype=object)
    units = pd.Series("", index=work.index, dtype=object)
    measurement_origin = pd.Series("", index=work.index, dtype=object)

    if "Standard Value" in work.columns:
        standard_value = col(work, "Standard Value")
        mask = standard_value.map(parse_numeric).notna()
        value.loc[mask] = standard_value.loc[mask]
        units.loc[mask] = col(work, "Standard Units").loc[mask].map(_clean_text)
        measurement_origin.loc[mask] = "standard_value"

    for assay_endpoint in BIOACTIVITY_ENDPOINTS:
        if assay_endpoint not in work.columns:
            continue
        endpoint_value = col(work, assay_endpoint)
        mask = value.map(parse_numeric).isna() & endpoint_value.map(parse_numeric).notna()
        value.loc[mask] = endpoint_value.loc[mask]
        endpoint.loc[mask & endpoint.eq("")] = assay_endpoint
        unit_column = f"{assay_endpoint} Units"
        if unit_column in work.columns:
            units.loc[mask] = col(work, unit_column).loc[mask].map(_clean_text)
        measurement_origin.loc[mask] = f"assay_column:{assay_endpoint}"

    if "PubChem Standard Value" in work.columns:
        pubchem_value = col(work, "PubChem Standard Value")
        mask = value.map(parse_numeric).isna() & pubchem_value.map(parse_numeric).notna()
        value.loc[mask] = pubchem_value.loc[mask]
        if "PubChem Standard Value Units" in work.columns:
            units.loc[mask] = col(work, "PubChem Standard Value Units").loc[mask].map(_clean_text)
        measurement_origin.loc[mask] = "pubchem_standard_value"

    # A reviewed assay registry can explicitly resolve metadata.  Overrides are
    # opt-in named columns, so raw PubChem metadata is never silently replaced.
    if "endpoint_override" in work.columns:
        override = work["endpoint_override"].map(_clean_text)
        endpoint = endpoint.where(override.eq(""), override)
    if "units_override" in work.columns:
        override = work["units_override"].map(_clean_text)
        units = units.where(override.eq(""), override)

    assay_name = col(work, "assay_name").fillna("")
    catalog_description = col(work, "catalog_assay_description").fillna("")
    row_description = col(work, "assay_description").fillna("")
    descriptions = row_description.where(row_description.astype(str).str.strip() != "", catalog_description)
    automatic_relevance = [
        assess_pubchem_relevance(target, accession, name, description)
        for target, accession, name, description in zip(
            col(work, "Target"),
            col(work, "Target Accession(s)"),
            assay_name,
            descriptions,
            strict=False,
        )
    ]
    relevance: list[tuple[str, bool, str]] = []
    for index, automatic in zip(work.index, automatic_relevance, strict=False):
        decision = _clean_text(col(work, "curation_decision").loc[index]).casefold()
        include_override = _coerce_optional_bool(col(work, "include_for_menin", None).loc[index])
        if include_override is True or decision in {"include", "included", "accept", "accepted"}:
            relevance.append(("manual_include", True, "included by reviewed PubChem assay registry"))
        elif include_override is False or decision in {"exclude", "excluded", "reject", "rejected"}:
            relevance.append(("manual_exclude", False, "excluded by reviewed PubChem assay registry"))
        elif decision in {"review", "pending", "unresolved"}:
            relevance.append(("manual_review", False, "review required by PubChem assay registry"))
        elif automatic[0] == "unresolved" and re.search(
            r"(?:^|;)target_gene:men1(?:;|$)",
            _clean_text(col(work, "search_terms").loc[index]).casefold(),
        ):
            relevance.append(
                (
                    "confirmed_target_gene_lookup",
                    True,
                    "assay returned by PubChem PUG REST MEN1 target-gene lookup",
                )
            )
        else:
            relevance.append(automatic)

    aids = col(work, "aid").map(_format_identifier)
    result_tags = col(work, "PUBCHEM_RESULT_TAG").map(_format_identifier)
    record_ids = aids + ":" + result_tags
    record_ids = record_ids.str.rstrip(":")
    compound_ids = (
        col(work, "PUBCHEM_CID")
        .where(col(work, "PUBCHEM_CID").notna(), col(work, "PUBCHEM_SID"))
        .map(_format_identifier)
    )
    deposit_dates = col(work, "deposit_date").map(_clean_text)
    deposit_years = pd.to_numeric(
        deposit_dates.str.extract(r"((?:19|20)\d{2})", expand=False),
        errors="coerce",
    )

    return pd.DataFrame(
        {
            "source": "PubChem",
            "source_record_id": record_ids,
            "compound_id": compound_ids,
            "compound_name": col(work, "Ligand"),
            "smiles": col(work, "PUBCHEM_EXT_DATASOURCE_SMILES"),
            "inchi_key": col(work, "PUBCHEM_IUPAC_INCHIKEY"),
            "target_name": col(work, "Target"),
            "target_id": col(work, "Target Accession(s)"),
            "endpoint": endpoint,
            "relation": [
                extract_relation(raw, explicit)
                for raw, explicit in zip(value, col(work, "Standard Relation"), strict=False)
            ],
            "value_raw": value,
            "standard_units": units,
            "assay_description": descriptions,
            "assay_name": assay_name,
            "assay_type": col(work, "PUBCHEM_ACTIVITY_OUTCOME"),
            "assay_family_override": col(work, "assay_family_override"),
            "assay_id": aids,
            "document_id": aids,
            "document_year": deposit_years,
            "date_provenance": "pubchem_assay_deposit_date",
            "source_deposit_date": deposit_dates,
            "source_modify_date": col(work, "modify_date"),
            "reference": "PubChem BioAssay AID " + aids,
            "source_detail": "PubChem BioAssay CSV",
            "measurement_origin": measurement_origin,
            "target_relevance": [item[0] for item in relevance],
            "is_target_relevant": [item[1] for item in relevance],
            "target_relevance_reason": [item[2] for item in relevance],
            "assay_relevance": [
                "direct_menin_activity"
                if item[1] and normalize_endpoint(ep) in BIOACTIVITY_ENDPOINTS
                else "menin_assay_endpoint_unresolved"
                if item[1]
                else item[0]
                for item, ep in zip(relevance, endpoint, strict=False)
            ],
        }
    )


def _coerce_optional_bool(value: object) -> bool | None:
    if _is_missing(value) or _clean_text(value) == "":
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = _clean_text(value).casefold()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


def _join_flags(flags: Iterable[str]) -> str:
    return ";".join(sorted({flag for flag in flags if flag}))


def annotate_cross_source_mirrors(activity: pd.DataFrame) -> pd.DataFrame:
    """Link exact normalized observations mirrored by different public sources.

    The rule is deliberately cross-source only: same-source replicates are
    never collapsed.  ChEMBL is preferred over BindingDB and PubChem because it
    generally retains the richest structured assay and document metadata.
    Every source row remains in the measurement table with a stable mirror ID;
    only rows marked ``is_cross_source_mirror_redundant`` are excluded from
    central-label aggregation.
    """

    out = activity.copy()
    if out.empty:
        return out
    identity = "structure_id" if "structure_id" in out.columns else "smiles"
    required = {identity, "source", "endpoint", "relation", "p_value"}
    if not required.issubset(out.columns):
        out["cross_source_mirror_group_id"] = ""
        out["is_cross_source_mirror_candidate"] = False
        out["is_cross_source_mirror_redundant"] = False
        out["cross_source_mirror_preferred_source"] = ""
        return out

    keys = [identity, "endpoint"]
    if "assay_family" in out.columns:
        keys.append("assay_family")
    keys.extend(["relation", "p_value"])
    eligible = (
        out["is_modeling_eligible"].fillna(False).astype(bool)
        if "is_modeling_eligible" in out.columns
        else pd.Series(True, index=out.index)
    )
    grouping_keys = [out[column] for column in keys]
    # A lower-priority row is only redundant when the preferred-source
    # counterpart is itself eligible for the same modeling population.  This
    # prevents a quarantined ChEMBL record from suppressing an otherwise valid
    # PubChem or BindingDB observation.
    eligible_sources = out["source"].where(eligible)
    source_count = eligible_sources.groupby(grouping_keys, dropna=False).transform("nunique")
    candidate = eligible & source_count.gt(1) & pd.to_numeric(out["p_value"], errors="coerce").notna()
    priority = (
        out["source"]
        .map(_clean_text)
        .str.casefold()
        .map({"chembl": 0, "bindingdb": 1, "pubchem": 2})
        .fillna(99)
    )
    preferred_priority = (
        priority.where(eligible, np.inf).groupby(grouping_keys, dropna=False).transform("min")
    )
    redundant = candidate & priority.gt(preferred_priority)
    preferred_source = out["source"].where(priority.eq(preferred_priority), "")
    preferred_source = (
        preferred_source.where(candidate, "").groupby(grouping_keys, dropna=False).transform(_join_unique)
    )
    key_text = out[keys].fillna("").astype(str).agg("\0".join, axis=1)
    group_id = key_text.map(lambda value: "MIR-" + sha256(value.encode("utf-8")).hexdigest()[:20].upper())
    out["cross_source_mirror_group_id"] = group_id.where(candidate, "")
    out["is_cross_source_mirror_candidate"] = candidate
    out["is_cross_source_mirror_redundant"] = redundant
    out["cross_source_mirror_preferred_source"] = preferred_source.where(candidate, "")
    return out


def normalize_activity_table(
    df: pd.DataFrame,
    *,
    standardize_structures: bool = True,
    require_rdkit: bool = False,
    strip_salts: bool = True,
    canonicalize_tautomer: bool = False,
    core_endpoints: Sequence[str] = BIOACTIVITY_ENDPOINTS,
    enforce_target_relevance: bool = True,
    exclude_assay_variants: bool = True,
    accepted_chembl_validity_comments: Sequence[str] = ("",),
    reject_chembl_potential_duplicates: bool = True,
) -> pd.DataFrame:
    """Normalize activity rows and attach validation, bounds, and quality flags."""

    if df.empty:
        return df.copy()

    out = df.copy()
    for required in (
        "source",
        "source_record_id",
        "compound_id",
        "compound_name",
        "smiles",
        "inchi_key",
        "target_name",
        "target_id",
        "endpoint",
        "relation",
        "value_raw",
        "standard_units",
        "assay_description",
        "assay_type",
        "assay_id",
        "document_id",
        "document_year",
        "reference",
        "source_detail",
    ):
        if required not in out.columns:
            out[required] = ""

    out["endpoint_original"] = out["endpoint"]
    out["relation_original"] = out["relation"]
    out["standard_units_original"] = out["standard_units"]
    out["endpoint"] = out["endpoint"].map(normalize_endpoint)
    out["relation"] = [
        extract_relation(value, relation)
        for value, relation in zip(out["value_raw"], out["relation"], strict=False)
    ]
    out["value_numeric"] = out["value_raw"].map(parse_numeric)
    out["unit_normalized"] = out["standard_units"].map(normalize_unit)
    out["unit_conversion_status"] = out["standard_units"].map(unit_conversion_status)
    factors = out["standard_units"].map(unit_factor_to_nm)
    out["value_nm"] = out["value_numeric"] * factors
    valid_positive = out["value_nm"].notna() & (out["value_nm"] > 0)
    out.loc[~valid_positive, "value_nm"] = np.nan
    out["p_value"] = out["value_nm"].map(p_value_from_nm)

    out["is_exact"] = out["relation"].isin(EXACT_RELATIONS)
    out["is_censored"] = out["relation"].isin(CENSORED_RELATIONS)
    out["censoring_direction"] = np.select(
        [
            out["relation"].isin({"<", "<="}),
            out["relation"].isin({">", ">="}),
            out["relation"].eq("~"),
            out["relation"].eq("="),
        ],
        ["left", "right", "approximate", "none"],
        default="invalid",
    )
    out["value_nm_lower_bound"] = np.nan
    out["value_nm_upper_bound"] = np.nan
    lower_mask = out["relation"].isin({"=", "~", ">", ">="}) & valid_positive
    upper_mask = out["relation"].isin({"=", "~", "<", "<="}) & valid_positive
    out.loc[lower_mask, "value_nm_lower_bound"] = out.loc[lower_mask, "value_nm"]
    out.loc[upper_mask, "value_nm_upper_bound"] = out.loc[upper_mask, "value_nm"]

    out["p_activity_lower_bound"] = np.nan
    out["p_activity_upper_bound"] = np.nan
    out.loc[upper_mask, "p_activity_lower_bound"] = out.loc[upper_mask, "p_value"]
    out.loc[lower_mask, "p_activity_upper_bound"] = out.loc[lower_mask, "p_value"]
    out["p_value_semantics"] = np.select(
        [out["is_exact"], out["relation"].eq("~"), out["is_censored"]],
        ["point_estimate", "approximate_point", "censoring_threshold"],
        default="invalid",
    )

    normalized_core_endpoints = {normalize_endpoint(value) for value in core_endpoints}
    out["is_core_endpoint"] = out["endpoint"].isin(normalized_core_endpoints)
    out["endpoint_family"] = out["endpoint"].map(endpoint_family)
    assay_context = (
        out["assay_description"].fillna("").astype(str)
        + " "
        + col(out, "assay_format").fillna("").astype(str)
    )
    automatic_assay_family = pd.Series(
        [
            classify_assay_family(endpoint, assay_type, description)
            for endpoint, assay_type, description in zip(
                out["endpoint"], out["assay_type"], assay_context, strict=False
            )
        ],
        index=out.index,
    )
    assay_family_override = col(out, "assay_family_override").map(_clean_text)
    out["assay_family"] = automatic_assay_family.where(assay_family_override.eq(""), assay_family_override)

    if standardize_structures:
        out = standardize_structure_table(
            out,
            require_rdkit=require_rdkit,
            strip_salts=strip_salts,
            canonicalize_tautomer=canonicalize_tautomer,
        )
    else:
        out["original_smiles"] = out["smiles"].map(_clean_text)
        out["structure_id"] = out["smiles"].map(_clean_text)
        out["structure_valid"] = None
        out["structure_standardization_status"] = "not_requested"

    incoming_relevance = col(out, "is_target_relevant", None).map(_coerce_optional_bool)
    derived_relevance = [
        _known_target_match(name, identifier)
        for name, identifier in zip(out["target_name"], out["target_id"], strict=False)
    ]
    out["is_target_relevant"] = [
        derived if supplied is None else supplied
        for supplied, derived in zip(incoming_relevance, derived_relevance, strict=False)
    ]
    derived_relevance_label = pd.Series(
        np.where(out["is_target_relevant"], "confirmed_target", "unresolved"),
        index=out.index,
    )
    if "target_relevance" not in out.columns:
        out["target_relevance"] = derived_relevance_label
    else:
        supplied_label = out["target_relevance"].map(_clean_text)
        out["target_relevance"] = supplied_label.where(supplied_label.ne(""), derived_relevance_label)
    derived_relevance_reason = pd.Series(
        np.where(
            out["is_target_relevant"],
            "target identifier/name matched",
            "target not established",
        ),
        index=out.index,
    )
    if "target_relevance_reason" not in out.columns:
        out["target_relevance_reason"] = derived_relevance_reason
    else:
        supplied_reason = out["target_relevance_reason"].map(_clean_text)
        out["target_relevance_reason"] = supplied_reason.where(
            supplied_reason.ne(""), derived_relevance_reason
        )

    # Stable observation keys make duplicate source records visible without
    # relying on dataframe row order as an identity.
    identity_fields = (
        "source",
        "source_record_id",
        "compound_id",
        "assay_id",
        "endpoint",
        "relation",
        "value_numeric",
        "unit_normalized",
        "document_id",
    )
    identity_text = out.apply(
        lambda row: "\0".join(_clean_text(row.get(name, "")) for name in identity_fields),
        axis=1,
    )
    out["measurement_id"] = identity_text.map(
        lambda value: "MEA-" + sha256(value.encode("utf-8")).hexdigest()[:24].upper()
    )
    has_source_record = out["source_record_id"].map(_clean_text).ne("")
    out["is_duplicate_measurement"] = has_source_record & out.duplicated(
        subset=list(identity_fields), keep="first"
    )
    record_keys = ["source", "source_record_id", "endpoint"]
    record_groups = out.groupby(record_keys, dropna=False)
    conflicting_values = (
        record_groups["value_numeric"]
        .transform(lambda values: pd.to_numeric(values, errors="coerce").dropna().nunique())
        .gt(1)
    )
    conflicting_structures = (
        record_groups["structure_id"]
        .transform(lambda values: values.map(_clean_text).replace("", np.nan).nunique())
        .gt(1)
    )
    out["source_record_conflict"] = has_source_record & (conflicting_values | conflicting_structures)

    reported_pchembl = pd.to_numeric(col(out, "reported_pchembl_value", np.nan), errors="coerce")
    out["pchembl_delta"] = (reported_pchembl - out["p_value"]).abs()

    accepted_validity = {_clean_text(value).casefold() for value in accepted_chembl_validity_comments}
    quality_flags: list[str] = []
    exclusion_reasons: list[str] = []
    for row in out.itertuples(index=False):
        flags: list[str] = []
        exclusions: list[str] = []
        original_flags = _clean_text(getattr(row, "quality_flags", ""))
        if original_flags:
            flags.extend(original_flags.split(";"))

        structure_status = _clean_text(getattr(row, "structure_standardization_status", ""))
        original_smiles = _clean_text(getattr(row, "original_smiles", ""))
        if not original_smiles:
            flags.append("missing_structure")
            exclusions.append("missing_structure")
        elif structure_status in {"invalid_smiles", "standardization_failed"}:
            flags.append(structure_status)
            exclusions.append("invalid_structure")
        elif structure_status == "rdkit_unavailable":
            flags.append("structure_not_rdkit_validated")

        numeric = row.value_numeric
        if not np.isfinite(numeric):
            flags.append("missing_or_non_numeric_value")
            exclusions.append("missing_or_non_numeric_value")
        elif numeric <= 0:
            flags.append("nonpositive_value")
            exclusions.append("nonpositive_value")

        conversion_status = row.unit_conversion_status
        if conversion_status != "converted":
            flags.append(conversion_status)
            exclusions.append(conversion_status)
        if not row.is_core_endpoint:
            flags.append("unsupported_endpoint")
            exclusions.append("unsupported_endpoint")
        if row.relation not in VALID_RELATIONS:
            flags.append("invalid_relation")
            exclusions.append("invalid_relation")
        if enforce_target_relevance and not bool(row.is_target_relevant):
            relevance = _clean_text(getattr(row, "target_relevance", ""))
            flag = (
                "target_not_relevant"
                if relevance in {"off_target", "manual_exclude"}
                else "target_relevance_unresolved"
            )
            flags.append(flag)
            exclusions.append(flag)
        source = _clean_text(getattr(row, "source", "")).casefold()
        if source == "chembl":
            standard_flag = parse_numeric(getattr(row, "standard_flag", np.nan))
            if not np.isfinite(standard_flag) or standard_flag != 1:
                flags.append("chembl_nonstandard_measurement")
                exclusions.append("chembl_nonstandard_measurement")
            validity_comment = _clean_text(getattr(row, "data_validity_comment", ""))
            if validity_comment.casefold() not in accepted_validity:
                flags.append("chembl_data_validity_warning")
                exclusions.append("chembl_data_validity_warning")
            if (
                reject_chembl_potential_duplicates
                and parse_numeric(getattr(row, "potential_duplicate", 0)) == 1
            ):
                flags.append("chembl_potential_duplicate")
                exclusions.append("chembl_potential_duplicate")
        if bool(getattr(row, "is_duplicate_measurement", False)):
            flags.append("duplicate_measurement")
            exclusions.append("duplicate_measurement")
        if bool(getattr(row, "source_record_conflict", False)):
            flags.append("source_record_conflict")
            exclusions.append("source_record_conflict")
        if exclude_assay_variants and _clean_text(getattr(row, "assay_variant_mutation", "")):
            flags.append("assay_variant_excluded")
            exclusions.append("assay_variant_excluded")
        pchembl_delta = getattr(row, "pchembl_delta", np.nan)
        if np.isfinite(pchembl_delta) and pchembl_delta > 0.05:
            flags.append("pchembl_conversion_mismatch")
            exclusions.append("pchembl_conversion_mismatch")
        if row.relation == "~":
            flags.append("approximate_value")
        quality_flags.append(_join_flags(flags))
        exclusion_reasons.append(_join_flags(exclusions))

    out["quality_flags"] = quality_flags
    out["exclusion_reason"] = exclusion_reasons
    out["is_modeling_eligible"] = out["exclusion_reason"].eq("")
    out["requires_review"] = out["quality_flags"].ne("")
    out = out.replace({np.inf: np.nan, -np.inf: np.nan})
    return out


def _resolve_censoring_policy(exact_only: bool | str, policy: str | None) -> str:
    selected: bool | str = exact_only if policy is None else policy
    if selected is True:
        return "strict_exact"
    if selected is False:
        return "include_all"
    aliases = {
        "strict": "strict_exact",
        "strict_exact": "strict_exact",
        "exact": "strict_exact",
        "prefer_exact": "prefer_exact_per_compound",
        "per_compound": "prefer_exact_per_compound",
        "prefer_exact_per_compound": "prefer_exact_per_compound",
        "all": "include_all",
        "include": "include_all",
        "include_all": "include_all",
    }
    normalized = aliases.get(_clean_text(selected).casefold())
    if normalized is None:
        raise ValueError(
            "censoring_policy must be one of strict_exact, prefer_exact_per_compound, or include_all"
        )
    return normalized


def _join_unique(values: pd.Series, limit: int = 20) -> str:
    items = sorted({_clean_text(value) for value in values if _clean_text(value)})
    return ";".join(items[:limit])


def aggregate_compounds(
    activity: pd.DataFrame,
    *,
    exact_only: bool | str = True,
    censoring_policy: str | None = None,
    stratify_by: Sequence[str] = (),
    eligible_only: bool = True,
    heterogeneity_log_spread_threshold: float = 1.0,
) -> pd.DataFrame:
    """Aggregate measurement-level activity under an explicit censoring policy.

    ``prefer_exact_per_compound`` keeps exact values for compounds that have
    them and retains censored-only compounds as bound-based aggregates.  The
    latter are explicitly marked through ``aggregation_value_semantics``.
    """

    if activity.empty:
        return pd.DataFrame()
    policy = _resolve_censoring_policy(exact_only, censoring_policy)

    data = activity.copy()
    required_mask = (
        col(data, "is_core_endpoint", False).fillna(False).astype(bool)
        & col(data, "smiles").map(_clean_text).ne("")
        & pd.to_numeric(col(data, "value_nm", np.nan), errors="coerce").gt(0)
        & pd.to_numeric(col(data, "p_value", np.nan), errors="coerce").notna()
    )
    if eligible_only and "is_modeling_eligible" in data.columns:
        required_mask &= data["is_modeling_eligible"].fillna(False).astype(bool)
    data = data[required_mask].copy()
    if data.empty:
        return pd.DataFrame()

    identity_column = "structure_id" if "structure_id" in data.columns else "smiles"
    if identity_column == "structure_id":
        data = data[data[identity_column].map(_clean_text).ne("")]
    group_columns = [identity_column] + [
        name for name in stratify_by if name in data.columns and name != identity_column
    ]

    if policy == "strict_exact":
        data = data[col(data, "is_exact", False).fillna(False).astype(bool)]
    elif policy == "prefer_exact_per_compound":
        has_exact = data.groupby(group_columns, dropna=False)["is_exact"].transform("any")
        data = data[data["is_exact"].fillna(False).astype(bool) | ~has_exact]
    if data.empty:
        return pd.DataFrame()

    # Public databases frequently mirror the same reported observation.  Keep
    # every source row in the measurement table and preserve same-source
    # replicates, but remove only explicitly linked lower-priority source mirrors
    # from central-label aggregation.
    provenance_groups = data.groupby(group_columns, dropna=False)
    data["_n_source_rows"] = provenance_groups["p_value"].transform("size")
    data["_n_sources_all"] = provenance_groups["source"].transform("nunique")
    data["_sources_all"] = provenance_groups["source"].transform(_join_unique)
    redundant = (
        data["is_cross_source_mirror_redundant"].fillna(False).astype(bool)
        if "is_cross_source_mirror_redundant" in data.columns
        else pd.Series(False, index=data.index)
    )
    data["_cross_source_mirror_rows_collapsed"] = redundant.groupby(
        [data[column] for column in group_columns], dropna=False
    ).transform("sum")
    data = data.loc[~redundant].copy()
    if data.empty:
        return pd.DataFrame()

    optional_first = {
        name: (name, "first")
        for name in (
            "smiles",
            "original_smiles",
            "canonical_smiles",
            "standardized_smiles",
            "inchi_key",
            "standard_inchi_key",
            "structure_standardization_status",
        )
        if name in data.columns and name not in group_columns
    }
    aggregation: dict[str, tuple[str, object]] = {
        "n_measurements": ("p_value", "size"),
        "n_source_rows": ("_n_source_rows", "first"),
        "n_cross_source_mirror_rows_collapsed": (
            "_cross_source_mirror_rows_collapsed",
            "first",
        ),
        "n_sources": ("_n_sources_all", "first"),
        "n_exact_measurements": ("is_exact", "sum"),
        "n_censored_measurements": ("is_censored", "sum"),
        "n_approximate_measurements": ("relation", lambda values: values.eq("~").sum()),
        "p_activity_median": ("p_value", "median"),
        "p_activity_best": ("p_value", "max"),
        "p_activity_min": ("p_value", "min"),
        "value_nm_median": ("value_nm", "median"),
        "value_nm_best": ("value_nm", "min"),
        "endpoints": ("endpoint", _join_unique),
        "n_endpoints": ("endpoint", "nunique"),
        "endpoint_families": ("endpoint_family", _join_unique),
        "n_endpoint_families": ("endpoint_family", "nunique"),
        "assay_families": ("assay_family", _join_unique),
        "n_assay_families": ("assay_family", "nunique"),
        "sources": ("_sources_all", "first"),
        "compound_ids": ("compound_id", _join_unique),
        "target_names": ("target_name", _join_unique),
        "document_years": ("document_year", _join_unique),
        "relations": ("relation", _join_unique),
    }
    aggregation.update(optional_first)
    # Compatibility for callers passing an older normalized table.
    for field, fallback in (
        ("is_exact", False),
        ("is_censored", False),
        ("endpoint_family", "other"),
        ("assay_family", "unclassified"),
    ):
        if field not in data.columns:
            data[field] = fallback

    compound = data.groupby(group_columns, dropna=False).agg(**aggregation).reset_index()
    compound["activity_range_log10"] = compound["p_activity_best"] - compound["p_activity_min"]
    compound["is_activity_heterogeneous"] = (
        compound["activity_range_log10"].gt(heterogeneity_log_spread_threshold)
        | compound["n_endpoint_families"].gt(1)
        | compound["n_assay_families"].gt(1)
    )
    compound["heterogeneity_log_spread_threshold"] = float(heterogeneity_log_spread_threshold)
    compound["censoring_policy"] = policy
    compound["aggregation_value_semantics"] = np.select(
        [
            compound["n_exact_measurements"].eq(compound["n_measurements"]),
            compound["n_censored_measurements"].eq(compound["n_measurements"]),
            compound["n_approximate_measurements"].eq(compound["n_measurements"]),
        ],
        ["exact_point_estimate", "censoring_thresholds_only", "approximate_points_only"],
        default="mixed_points_and_thresholds",
    )
    compound["potency_class"] = np.select(
        [
            compound["value_nm_median"] <= 100,
            compound["value_nm_median"] <= 1000,
            compound["value_nm_median"] <= 10000,
        ],
        ["high_potency_<=100nM", "moderate_100nM_to_1uM", "weak_1uM_to_10uM"],
        default="low_or_inactive_>10uM",
    )
    compound["active_100nM"] = compound["value_nm_median"] <= 100
    compound["active_1uM"] = compound["value_nm_median"] <= 1000
    return compound.sort_values(
        ["p_activity_median", "n_measurements"], ascending=[False, False]
    ).reset_index(drop=True)


def _normalize_admet_type(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", _clean_text(value).casefold())


def _classify_admet_row(
    standard_type: object,
    description: object,
    target_name: object = "",
) -> tuple[str, str, str, str]:
    """Return category, endpoint, directionality, and rule evidence."""

    kind = _normalize_admet_type(standard_type)
    text = f"{_clean_text(description)} {_clean_text(target_name)}".casefold()

    if kind in {"auc", "auclast", "aucinf", "auc0inf"}:
        return "pharmacokinetics", "exposure_auc", "higher_is_more_exposure", "type:AUC"
    if kind == "cmax":
        return "pharmacokinetics", "maximum_concentration", "higher_is_more_exposure", "type:Cmax"
    if kind == "tmax":
        return "pharmacokinetics", "time_to_cmax", "context_dependent", "type:Tmax"
    if kind in {"cl", "clint", "clrenal"} and re.search(r"clearance|microsom|hepatocyte", text):
        if kind == "clint" or re.search(r"intrinsic|microsom|hepatocyte", text):
            return (
                "metabolic_stability",
                "intrinsic_clearance",
                "lower_is_more_stable",
                "type+description:intrinsic_clearance",
            )
        return (
            "pharmacokinetics",
            "systemic_clearance",
            "lower_is_more_exposure",
            "type+description:clearance",
        )
    if kind in {"t12", "halflife"} and re.search(r"half.?life|microsom|hepatocyte", text):
        if re.search(r"microsom|hepatocyte|plasma stability", text):
            return (
                "metabolic_stability",
                "in_vitro_half_life",
                "higher_is_more_stable",
                "type+description:stability_half_life",
            )
        return (
            "pharmacokinetics",
            "terminal_half_life",
            "higher_is_more_exposure",
            "type+description:half_life",
        )
    if kind in {"f", "frel", "fabs", "fg"} and re.search(
        r"bioavailability|absorbed|intestinal availability", text
    ):
        return "pharmacokinetics", "bioavailability", "higher_is_better", "type+description:bioavailability"
    if kind in {"fu", "ppb", "bpr"} and re.search(
        r"unbound|protein bind|bound to plasma|plasma binding|brain.*plasma", text
    ):
        endpoint_name = "fraction_unbound" if kind == "fu" else "plasma_protein_binding"
        direction = "higher_is_more_unbound" if kind == "fu" else "lower_is_more_unbound"
        return "distribution", endpoint_name, direction, "type+description:protein_binding"
    if kind in {"vd", "vdu", "vdss"}:
        return "distribution", "volume_of_distribution", "context_dependent", "type:volume_distribution"
    if kind in {"papp", "peff", "ratiopapp", "permeability"} and re.search(
        r"permeab|penetration|caco|mdck|pampa|bbb", text
    ):
        return (
            "permeability",
            "apparent_permeability",
            "higher_is_more_permeable",
            "type+description:permeability",
        )
    if kind in {"kp", "kpuucsf"} and re.search(r"tissue|brain|csf|plasma|partition", text):
        return (
            "distribution",
            "tissue_partition",
            "higher_is_more_partitioned",
            "type+description:tissue_partition",
        )
    if kind in {"solubility", "logsolubility"} and "solub" in text:
        return "physicochemical", "solubility", "higher_is_better", "type+description:solubility"
    if kind in {"logd", "logd74"}:
        return "physicochemical", "logd", "context_dependent", "type:LogD"
    if kind == "logp":
        return "physicochemical", "logp", "context_dependent", "type:LogP"
    if kind in {"ic50", "ki"} and re.search(r"\bcyp\s*\d|cytochrome p450", text):
        return "drug_interaction", "cyp_inhibition", "higher_is_safer", "endpoint+description:CYP"
    if kind in {"ic50", "ec50"} and re.search(r"\bherg\b|\bkcnh2\b", text):
        return "safety_pharmacology", "herg_inhibition", "higher_is_safer", "endpoint+target:hERG"
    if kind in {"cc50", "tc50", "ld50", "mtd"}:
        return "toxicity", _clean_text(standard_type), "higher_is_safer", "type:toxicity"
    if kind.startswith("dili") or kind.startswith("hepatotoxicity"):
        return "toxicity", "hepatotoxicity", "lower_is_safer", "type:hepatotoxicity"
    return "", "", "", ""


def _extract_context(description: object) -> tuple[str, str, str, str]:
    text = _clean_text(description).casefold()
    species = next(
        (name for name in ("human", "mouse", "rat", "dog", "monkey", "rabbit") if name in text),
        "",
    )
    matrix = next(
        (
            name
            for name in (
                "liver microsomes",
                "microsomes",
                "hepatocytes",
                "plasma",
                "brain",
                "whole blood",
                "cell homogenate",
            )
            if name in text
        ),
        "",
    )
    route_patterns = (
        (r"\bpo\b|\boral(?:ly)?\b", "oral"),
        (r"\biv\b|intravenous", "intravenous"),
        (r"\bip\b|intraperitoneal", "intraperitoneal"),
        (r"\bim\b|intramuscular", "intramuscular"),
        (r"\bsc\b|subcutaneous", "subcutaneous"),
    )
    route = next((label for pattern, label in route_patterns if re.search(pattern, text)), "")
    if re.search(r"mg/kg|administered|dosing|single dose", text):
        context = "in_vivo"
    elif re.search(r"microsom|hepatocyte|cell|plasma|buffer|dialysis", text):
        context = "in_vitro"
    else:
        context = "unspecified"
    return species, matrix, route, context


def classify_pk_admet(molecule_activity: pd.DataFrame) -> pd.DataFrame:
    """Classify every molecule-activity row under endpoint-specific ADMET rules."""

    if molecule_activity.empty:
        return pd.DataFrame()
    out = molecule_activity.copy()
    classifications = [
        _classify_admet_row(kind, description, target)
        for kind, description, target in zip(
            col(out, "standard_type"),
            col(out, "assay_description"),
            col(out, "target_pref_name"),
            strict=False,
        )
    ]
    contexts = [_extract_context(value) for value in col(out, "assay_description")]
    out["admet_category"] = [item[0] for item in classifications]
    out["admet_endpoint"] = [item[1] for item in classifications]
    out["admet_directionality"] = [item[2] for item in classifications]
    out["admet_inclusion_rule"] = [item[3] for item in classifications]
    out["is_admet_relevant"] = out["admet_category"].ne("")
    out["species"] = [item[0] for item in contexts]
    out["matrix"] = [item[1] for item in contexts]
    out["administration_route"] = [item[2] for item in contexts]
    out["experimental_context"] = [item[3] for item in contexts]
    return out


def curate_pk_admet(molecule_activity: pd.DataFrame) -> pd.DataFrame:
    """Return defensible PK/ADMET observations selected by explicit rules."""

    classified = classify_pk_admet(molecule_activity)
    if classified.empty:
        return pd.DataFrame()
    data = classified[classified["is_admet_relevant"]].copy()
    if data.empty:
        return pd.DataFrame()

    renamed = data.copy()
    submitted_smiles = col(renamed, "smiles").map(_clean_text)
    canonical_input = col(renamed, "canonical_smiles").map(_clean_text)
    renamed["smiles"] = submitted_smiles.where(submitted_smiles.ne(""), canonical_input)
    renamed["relation"] = col(renamed, "standard_relation")
    renamed["value_raw"] = col(renamed, "standard_value")
    if "source" not in renamed.columns:
        renamed["source"] = "ChEMBL"
    renamed = standardize_structure_table(renamed)
    renamed["value_numeric"] = col(renamed, "value_raw").map(parse_numeric)
    renamed["relation"] = [
        extract_relation(value, relation)
        for value, relation in zip(col(renamed, "value_raw"), col(renamed, "relation"), strict=False)
    ]
    # Keep the original public column names as aliases so existing notebooks
    # continue to work while the clearer common-schema names are adopted.
    renamed["standard_relation"] = renamed["relation"]
    renamed["standard_value"] = renamed["value_raw"]
    renamed["admet_quality_flags"] = [
        _join_flags(
            [
                "missing_structure" if not _clean_text(smiles) else "",
                "missing_or_non_numeric_value" if not np.isfinite(value) else "",
                "missing_unit" if not _clean_text(unit) else "",
                "structure_not_rdkit_validated" if status == "rdkit_unavailable" else "",
            ]
        )
        for smiles, value, unit, status in zip(
            col(renamed, "original_smiles"),
            renamed["value_numeric"],
            col(renamed, "standard_units"),
            col(renamed, "structure_standardization_status"),
            strict=False,
        )
    ]
    renamed["is_admet_analysis_ready"] = (
        col(renamed, "structure_id").map(_clean_text).ne("")
        & renamed["value_numeric"].notna()
        & col(renamed, "standard_units").map(_clean_text).ne("")
    )

    preferred = [
        "source",
        "activity_id",
        "molecule_chembl_id",
        "structure_id",
        "smiles",
        "original_smiles",
        "inchi_key",
        "standard_type",
        "standard_relation",
        "standard_value",
        "relation",
        "value_raw",
        "value_numeric",
        "standard_units",
        "admet_category",
        "admet_endpoint",
        "admet_directionality",
        "admet_inclusion_rule",
        "species",
        "matrix",
        "administration_route",
        "experimental_context",
        "assay_description",
        "assay_type",
        "target_chembl_id",
        "target_pref_name",
        "document_chembl_id",
        "document_year",
        "is_admet_analysis_ready",
        "admet_quality_flags",
        "structure_standardization_status",
    ]
    columns = [name for name in preferred if name in renamed.columns]
    return renamed[columns].drop_duplicates().reset_index(drop=True)


def curate_herg_compounds(
    herg_activity: pd.DataFrame,
    *,
    stratify_by: Sequence[str] = ("endpoint", "assay_family"),
    blocker_max_nm: float = 10_000.0,
    nonblocker_min_nm: float = 30_000.0,
    heterogeneity_log_spread_threshold: float = 1.0,
) -> pd.DataFrame:
    """Aggregate exact hERG/KCNH2 activity and add coarse liability labels."""

    compounds = aggregate_compounds(
        herg_activity,
        exact_only=True,
        stratify_by=stratify_by,
        heterogeneity_log_spread_threshold=heterogeneity_log_spread_threshold,
    )
    if compounds.empty:
        return compounds
    compounds = compounds.rename(
        columns={
            "p_activity_median": "p_herg_median",
            "p_activity_best": "p_herg_best",
            "value_nm_median": "herg_value_nm_median",
            "value_nm_best": "herg_value_nm_best",
        }
    )
    compounds["herg_blocker_label"] = np.nan
    compounds.loc[compounds["herg_value_nm_median"] <= blocker_max_nm, "herg_blocker_label"] = 1.0
    compounds.loc[compounds["herg_value_nm_median"] >= nonblocker_min_nm, "herg_blocker_label"] = 0.0
    compounds["herg_label_policy"] = (
        f"<={blocker_max_nm:g} nM blocker, >={nonblocker_min_nm:g} nM non-blocker, "
        "intermediate values omitted"
    )
    compounds["target_names"] = HERG_TARGET["name"]
    return compounds


def source_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["source", "n_measurements", "n_compounds"])
    identity = "structure_id" if "structure_id" in df.columns else "smiles"
    summary = (
        df.groupby("source")
        .agg(n_measurements=("source", "size"), n_compounds=(identity, "nunique"))
        .reset_index()
    )
    if "is_modeling_eligible" in df.columns:
        eligible = (
            df.groupby("source")["is_modeling_eligible"].sum().rename("n_modeling_eligible").reset_index()
        )
        summary = summary.merge(eligible, on="source", how="left")
    return summary.sort_values("n_measurements", ascending=False)


def quality_summary(df: pd.DataFrame, dataset: str = "") -> pd.DataFrame:
    """Summarize modeling eligibility and individual exclusion reasons."""

    if df.empty:
        return pd.DataFrame(columns=["dataset", "reason", "n_rows", "n_structures"])
    identity = "structure_id" if "structure_id" in df.columns else "smiles"
    rows: list[dict[str, object]] = []
    for index, reasons in df.get("exclusion_reason", pd.Series("", index=df.index)).items():
        reason_items = [item for item in _clean_text(reasons).split(";") if item] or ["eligible"]
        for reason in reason_items:
            rows.append({"index": index, "reason": reason})
    exploded = pd.DataFrame(rows).merge(
        df[[identity]].reset_index().rename(columns={"index": "index"}), on="index", how="left"
    )
    summary = (
        exploded.groupby("reason")
        .agg(n_rows=("index", "size"), n_structures=(identity, "nunique"))
        .reset_index()
    )
    summary.insert(0, "dataset", dataset)
    return summary.sort_values(["n_rows", "reason"], ascending=[False, True])


def write_processed_tables(
    *,
    chembl_menin_raw: pd.DataFrame,
    bindingdb_raw: pd.DataFrame,
    pubchem_raw: pd.DataFrame,
    pubchem_catalog: pd.DataFrame,
    herg_raw: pd.DataFrame,
    pk_raw: pd.DataFrame,
    processed_dir: Path,
    menin_stratify_by: Sequence[str] = ("endpoint", "assay_family"),
    herg_stratify_by: Sequence[str] = ("endpoint", "assay_family"),
    normalization_options: Mapping[str, Any] | None = None,
    menin_censoring_policy: str = "strict_exact",
    heterogeneity_log_spread_threshold: float = 1.0,
    herg_blocker_max_nm: float = 10_000.0,
    herg_nonblocker_min_nm: float = 30_000.0,
) -> dict[str, pd.DataFrame]:
    """Create processed tables plus explicit quarantine and QC outputs."""

    processed_dir.mkdir(parents=True, exist_ok=True)
    menin_parts = [
        chembl_to_long(
            chembl_menin_raw,
            source_detail=f"ChEMBL target {MENIN_TARGET['chembl_id']}",
        )
    ]
    if not bindingdb_raw.empty:
        from .bindingdb import bindingdb_to_long

        menin_parts.append(bindingdb_to_long(bindingdb_raw))
    if not pubchem_raw.empty:
        menin_parts.append(pubchem_to_long(pubchem_raw, pubchem_catalog))
    menin_parts = [part for part in menin_parts if not part.empty]
    combined_menin = pd.concat(menin_parts, ignore_index=True, sort=False) if menin_parts else pd.DataFrame()
    normalize_kwargs = dict(normalization_options or {})
    menin_activity = normalize_activity_table(combined_menin, **normalize_kwargs)
    menin_activity = annotate_cross_source_mirrors(menin_activity)
    menin_activity.to_csv(processed_dir / "menin_activity_measurements.csv", index=False)

    menin_compounds = aggregate_compounds(
        menin_activity,
        censoring_policy=menin_censoring_policy,
        stratify_by=menin_stratify_by,
        heterogeneity_log_spread_threshold=heterogeneity_log_spread_threshold,
    )
    menin_compounds.to_csv(processed_dir / "menin_compounds_curated.csv", index=False)
    if not menin_activity.empty:
        menin_activity[~menin_activity["is_modeling_eligible"]].to_csv(
            processed_dir / "menin_activity_quarantine.csv", index=False
        )

    herg_activity = normalize_activity_table(
        chembl_to_long(herg_raw, source_detail=f"ChEMBL target {HERG_TARGET['chembl_id']}"),
        **normalize_kwargs,
    )
    herg_activity = annotate_cross_source_mirrors(herg_activity)
    herg_activity.to_csv(processed_dir / "herg_activity_measurements.csv", index=False)
    herg_compounds = curate_herg_compounds(
        herg_activity,
        stratify_by=herg_stratify_by,
        blocker_max_nm=herg_blocker_max_nm,
        nonblocker_min_nm=herg_nonblocker_min_nm,
        heterogeneity_log_spread_threshold=heterogeneity_log_spread_threshold,
    )
    herg_compounds.to_csv(processed_dir / "herg_compounds_curated.csv", index=False)
    if not herg_activity.empty:
        herg_activity[~herg_activity["is_modeling_eligible"]].to_csv(
            processed_dir / "herg_activity_quarantine.csv", index=False
        )

    pk_admet_all = curate_pk_admet(pk_raw)
    if "is_admet_analysis_ready" in pk_admet_all.columns:
        ready_mask = pk_admet_all["is_admet_analysis_ready"].fillna(False).astype(bool)
        pk_admet = pk_admet_all[ready_mask].copy()
        pk_admet_all.to_csv(processed_dir / "pk_admet_observations_all.csv", index=False)
        pk_admet_all[~ready_mask].to_csv(processed_dir / "pk_admet_quarantine.csv", index=False)
    else:
        pk_admet = pk_admet_all
    pk_admet.to_csv(processed_dir / "pk_admet_observations.csv", index=False)
    source_summary(menin_activity).to_csv(processed_dir / "source_summary.csv", index=False)
    pd.concat(
        [
            quality_summary(menin_activity, "menin"),
            quality_summary(herg_activity, "herg"),
        ],
        ignore_index=True,
    ).to_csv(processed_dir / "data_quality_summary.csv", index=False)

    return {
        "menin_activity": menin_activity,
        "menin_compounds": menin_compounds,
        "herg_activity": herg_activity,
        "herg_compounds": herg_compounds,
        "pk_admet": pk_admet,
    }


__all__ = [
    "ENDPOINT_CANONICAL",
    "ENDPOINT_FAMILY",
    "UNIT_TO_NM",
    "aggregate_compounds",
    "annotate_cross_source_mirrors",
    "assess_pubchem_relevance",
    "chembl_to_long",
    "classify_assay_family",
    "classify_pk_admet",
    "col",
    "curate_herg_compounds",
    "curate_pk_admet",
    "endpoint_family",
    "extract_relation",
    "normalize_activity_table",
    "normalize_endpoint",
    "normalize_relation",
    "normalize_unit",
    "p_value_from_nm",
    "parse_numeric",
    "pubchem_to_long",
    "quality_summary",
    "source_summary",
    "unit_conversion_status",
    "unit_factor_to_nm",
    "write_processed_tables",
]
