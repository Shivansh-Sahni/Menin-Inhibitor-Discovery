"""Build leakage-safe, endpoint-specific PK/ADME modeling surfaces.

This module is deliberately conservative about what constitutes a measurement.
It consumes only local, physically bound source files.  Candidate documents,
catalog rows, paired X/Y file rows, overlapping train/test/raw releases, and
well-level readouts are never counted as molecule-level endpoint measurements.

The release is a modeling-preparation surface, not a canonical clinical/PK
release and not a label for hERG, QT, safety, efficacy, or clinical outcome.
Every endpoint remains source-native and task-specific.  Context that is absent
from the source remains null and is represented by an explicit quality flag.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as pads
import pyarrow.parquet as pq
from rdkit import Chem, rdBase
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold

SCHEMA_VERSION = "platform-pk-adme-trainable-surfaces/1.0"
STANDARDIZATION_VERSION = "rdkit-cleanup-fragment-parent-uncharge-isomeric-v1"
RELEASE_VERSION = "v1_0_trainable_surfaces"
MIN_TASK_CONNECTIVITY_GROUPS = 20


class PKADMESurfaceError(RuntimeError):
    """Raised when a source binding or release invariant fails closed."""


_OBSERVATION_SCHEMA = pa.schema(
    [
        pa.field("observation_id", pa.large_string(), nullable=False),
        pa.field("source_family", pa.large_string(), nullable=False),
        pa.field("source_dataset", pa.large_string(), nullable=False),
        pa.field("source_version", pa.large_string(), nullable=False),
        pa.field("source_file", pa.large_string(), nullable=False),
        pa.field("source_row_number", pa.int64(), nullable=False),
        pa.field("source_record_id", pa.large_string(), nullable=False),
        pa.field("source_split", pa.large_string()),
        pa.field("source_rights_status", pa.large_string(), nullable=False),
        pa.field("source_lineage", pa.large_string(), nullable=False),
        pa.field("raw_smiles", pa.large_string()),
        pa.field("standardized_smiles", pa.large_string()),
        pa.field("standard_inchi_key", pa.large_string()),
        pa.field("molecule_id", pa.large_string()),
        pa.field("connectivity_key", pa.large_string()),
        pa.field("leakage_group_id", pa.large_string()),
        pa.field("scaffold_smiles", pa.large_string()),
        pa.field("scaffold_group_id", pa.large_string()),
        pa.field("structure_status", pa.large_string(), nullable=False),
        pa.field("task_id", pa.large_string(), nullable=False),
        pa.field("endpoint_family", pa.large_string(), nullable=False),
        pa.field("endpoint_name", pa.large_string(), nullable=False),
        pa.field("endpoint_variant", pa.large_string(), nullable=False),
        pa.field("task_type", pa.large_string(), nullable=False),
        pa.field("source_native_endpoint", pa.large_string(), nullable=False),
        pa.field("original_value_text", pa.large_string()),
        pa.field("original_numeric_value", pa.float64()),
        pa.field("original_unit", pa.large_string()),
        pa.field("relation", pa.large_string(), nullable=False),
        pa.field("normalized_value", pa.float64(), nullable=False),
        pa.field("normalized_unit", pa.large_string(), nullable=False),
        pa.field("lower_bound", pa.float64()),
        pa.field("upper_bound", pa.float64()),
        pa.field("is_censored", pa.bool_(), nullable=False),
        pa.field("uncertainty_lower", pa.float64()),
        pa.field("uncertainty_upper", pa.float64()),
        pa.field("species", pa.large_string()),
        pa.field("matrix", pa.large_string()),
        pa.field("route", pa.large_string()),
        pa.field("dose_value", pa.float64()),
        pa.field("dose_unit", pa.large_string()),
        pa.field("time_value", pa.float64()),
        pa.field("time_unit", pa.large_string()),
        pa.field("assay_id", pa.large_string()),
        pa.field("document_id", pa.large_string()),
        pa.field("context_complete", pa.bool_(), nullable=False),
        pa.field("clinical_pk_context", pa.bool_(), nullable=False),
        pa.field("qt_ecg_context", pa.bool_(), nullable=False),
        pa.field("eligible_exact_modeling", pa.bool_(), nullable=False),
        pa.field("eligible_censored_modeling", pa.bool_(), nullable=False),
        pa.field("eligible_cross_source_union", pa.bool_(), nullable=False),
        pa.field("canonical_admission", pa.bool_(), nullable=False),
        pa.field("modeling_status", pa.large_string(), nullable=False),
        pa.field("duplicate_group_id", pa.large_string()),
        pa.field("quality_flags_json", pa.large_string(), nullable=False),
        pa.field("context_json", pa.large_string(), nullable=False),
        pa.field("provenance_json", pa.large_string(), nullable=False),
    ]
)

_TASK_SCHEMA = pa.schema(
    [
        pa.field("task_id", pa.large_string(), nullable=False),
        pa.field("endpoint_family", pa.large_string(), nullable=False),
        pa.field("endpoint_name", pa.large_string(), nullable=False),
        pa.field("endpoint_variant", pa.large_string(), nullable=False),
        pa.field("task_type", pa.large_string(), nullable=False),
        pa.field("normalized_unit", pa.large_string(), nullable=False),
        pa.field("source_families_json", pa.large_string(), nullable=False),
        pa.field("source_datasets_json", pa.large_string(), nullable=False),
        pa.field("observation_count", pa.int64(), nullable=False),
        pa.field("valid_structure_count", pa.int64(), nullable=False),
        pa.field("unique_molecule_count", pa.int64(), nullable=False),
        pa.field("unique_connectivity_group_count", pa.int64(), nullable=False),
        pa.field("exact_modeling_count", pa.int64(), nullable=False),
        pa.field("censored_only_modeling_count", pa.int64(), nullable=False),
        pa.field("context_complete_count", pa.int64(), nullable=False),
        pa.field("cross_source_union_count", pa.int64(), nullable=False),
        pa.field("task_trainable", pa.bool_(), nullable=False),
        pa.field("minimum_connectivity_groups", pa.int64(), nullable=False),
        pa.field("species_json", pa.large_string(), nullable=False),
        pa.field("matrices_json", pa.large_string(), nullable=False),
        pa.field("routes_json", pa.large_string(), nullable=False),
        pa.field("conditioning_contract_json", pa.large_string(), nullable=False),
        pa.field("rights_statuses_json", pa.large_string(), nullable=False),
        pa.field("lineage_overlap_risk", pa.bool_(), nullable=False),
    ]
)

_MOLECULE_SCHEMA = pa.schema(
    [
        pa.field("molecule_id", pa.large_string(), nullable=False),
        pa.field("standardized_smiles", pa.large_string(), nullable=False),
        pa.field("standard_inchi_key", pa.large_string(), nullable=False),
        pa.field("connectivity_key", pa.large_string(), nullable=False),
        pa.field("leakage_group_id", pa.large_string(), nullable=False),
        pa.field("scaffold_smiles", pa.large_string(), nullable=False),
        pa.field("scaffold_group_id", pa.large_string(), nullable=False),
        pa.field("source_families_json", pa.large_string(), nullable=False),
        pa.field("observation_count", pa.int64(), nullable=False),
    ]
)

_DISPOSITION_SCHEMA = pa.schema(
    [
        pa.field("source_family", pa.large_string(), nullable=False),
        pa.field("source_dataset", pa.large_string(), nullable=False),
        pa.field("physical_file_rows", pa.int64(), nullable=False),
        pa.field("candidate_record_rows", pa.int64(), nullable=False),
        pa.field("emitted_measurement_count", pa.int64(), nullable=False),
        pa.field("exact_modeling_count", pa.int64(), nullable=False),
        pa.field("censored_only_modeling_count", pa.int64(), nullable=False),
        pa.field("missing_or_invalid_structure_count", pa.int64(), nullable=False),
        pa.field("excluded_candidate_count", pa.int64(), nullable=False),
        pa.field("disposition", pa.large_string(), nullable=False),
        pa.field("exclusion_reasons_json", pa.large_string(), nullable=False),
        pa.field("double_count_policy", pa.large_string(), nullable=False),
    ]
)


@dataclass(frozen=True)
class StructureIdentity:
    standardized_smiles: str | None
    standard_inchi_key: str | None
    molecule_id: str | None
    connectivity_key: str | None
    leakage_group_id: str | None
    scaffold_smiles: str | None
    scaffold_group_id: str | None
    status: str


@dataclass
class TaskAccumulator:
    endpoint_family: str
    endpoint_name: str
    endpoint_variant: str
    task_type: str
    normalized_unit: str
    source_families: set[str] = field(default_factory=set)
    source_datasets: set[str] = field(default_factory=set)
    rights_statuses: set[str] = field(default_factory=set)
    species: set[str] = field(default_factory=set)
    matrices: set[str] = field(default_factory=set)
    routes: set[str] = field(default_factory=set)
    molecules: set[str] = field(default_factory=set)
    connectivity_groups: set[str] = field(default_factory=set)
    observation_count: int = 0
    valid_structure_count: int = 0
    exact_modeling_count: int = 0
    censored_only_modeling_count: int = 0
    context_complete_count: int = 0
    cross_source_union_count: int = 0
    lineage_overlap_risk: bool = False


@dataclass
class DispositionAccumulator:
    physical_file_rows: int = 0
    candidate_record_rows: int = 0
    emitted_measurement_count: int = 0
    exact_modeling_count: int = 0
    censored_only_modeling_count: int = 0
    missing_or_invalid_structure_count: int = 0
    excluded_candidate_count: int = 0
    reasons: Counter[str] = field(default_factory=Counter)
    disposition: str = "measurement_rows_standardized"
    double_count_policy: str = "one source record per explicit endpoint field"


@dataclass(frozen=True)
class TDCSpec:
    endpoint_family: str
    endpoint_name: str
    endpoint_variant: str
    task_type: str
    normalized_unit: str
    species: str | None = None
    matrix: str | None = None
    lineage: str = "source_native_benchmark_underlying_rights_conditional"


def _tdc_specs() -> dict[str, TDCSpec]:
    classification = "classification"
    regression = "regression"
    return {
        "approved_pampa_ncats": TDCSpec(
            "permeability",
            "pampa_permeability_class",
            "approved_drugs",
            classification,
            "class_label",
            matrix="artificial_membrane",
        ),
        "b3db_classification": TDCSpec(
            "distribution",
            "blood_brain_barrier_class",
            "b3db",
            classification,
            "class_label",
            matrix="blood_brain_barrier",
        ),
        "b3db_regression": TDCSpec(
            "distribution", "logbb", "b3db", regression, "source_native_scale", matrix="blood_brain_barrier"
        ),
        "bbb_martins": TDCSpec(
            "distribution",
            "blood_brain_barrier_class",
            "martins",
            classification,
            "class_label",
            matrix="blood_brain_barrier",
        ),
        "bioavailability_ma": TDCSpec(
            "bioavailability", "bioavailability_class", "ma", classification, "class_label"
        ),
        "caco2_wang": TDCSpec(
            "permeability", "caco2_papp", "wang_log10", regression, "log10(cm/s)", matrix="caco2"
        ),
        "clearance_hepatocyte_az": TDCSpec(
            "clearance",
            "intrinsic_clearance",
            "az_hepatocyte",
            regression,
            "source_native_scale",
            matrix="hepatocyte",
        ),
        "clearance_microsome_az": TDCSpec(
            "clearance",
            "intrinsic_clearance",
            "az_microsome",
            regression,
            "source_native_scale",
            matrix="microsome",
        ),
        "cyp1a2_veith": TDCSpec(
            "metabolism",
            "cyp_inhibition_class",
            "cyp1a2_veith",
            classification,
            "class_label",
            species="Homo sapiens",
        ),
        "cyp2c19_veith": TDCSpec(
            "metabolism",
            "cyp_inhibition_class",
            "cyp2c19_veith",
            classification,
            "class_label",
            species="Homo sapiens",
        ),
        "cyp2c9_substrate_carbonmangels": TDCSpec(
            "metabolism",
            "cyp_substrate_class",
            "cyp2c9_carbonmangels",
            classification,
            "class_label",
            species="Homo sapiens",
        ),
        "cyp2c9_veith": TDCSpec(
            "metabolism",
            "cyp_inhibition_class",
            "cyp2c9_veith",
            classification,
            "class_label",
            species="Homo sapiens",
        ),
        "cyp2d6_substrate_carbonmangels": TDCSpec(
            "metabolism",
            "cyp_substrate_class",
            "cyp2d6_carbonmangels",
            classification,
            "class_label",
            species="Homo sapiens",
        ),
        "cyp2d6_veith": TDCSpec(
            "metabolism",
            "cyp_inhibition_class",
            "cyp2d6_veith",
            classification,
            "class_label",
            species="Homo sapiens",
        ),
        "cyp3a4_substrate_carbonmangels": TDCSpec(
            "metabolism",
            "cyp_substrate_class",
            "cyp3a4_carbonmangels",
            classification,
            "class_label",
            species="Homo sapiens",
        ),
        "cyp3a4_veith": TDCSpec(
            "metabolism",
            "cyp_inhibition_class",
            "cyp3a4_veith",
            classification,
            "class_label",
            species="Homo sapiens",
        ),
        "half_life_obach": TDCSpec(
            "half_life", "reported_half_life", "obach_context_incomplete", regression, "source_native_scale"
        ),
        "hia_hou": TDCSpec(
            "absorption",
            "human_intestinal_absorption_class",
            "hou",
            classification,
            "class_label",
            species="Homo sapiens",
            matrix="intestine",
        ),
        "hlm": TDCSpec(
            "stability",
            "microsome_stability_class",
            "human_liver_microsome",
            classification,
            "class_label",
            species="Homo sapiens",
            matrix="liver_microsome",
        ),
        "hydrationfreeenergy_freesolv": TDCSpec(
            "physicochemical", "hydration_free_energy", "freesolv", regression, "source_native_scale"
        ),
        "lipophilicity_astrazeneca": TDCSpec("lipophilicity", "logd", "astrazeneca", regression, "unitless"),
        "pampa_ncats": TDCSpec(
            "permeability",
            "pampa_permeability_class",
            "ncats",
            classification,
            "class_label",
            matrix="artificial_membrane",
        ),
        "pgp_broccatelli": TDCSpec(
            "transport",
            "pgp_interaction_class",
            "broccatelli",
            classification,
            "class_label",
            species="Homo sapiens",
        ),
        "ppbr_az": TDCSpec(
            "protein_binding",
            "plasma_protein_binding",
            "astrazeneca_source_scale",
            regression,
            "source_native_scale",
            matrix="plasma",
        ),
        "rlm": TDCSpec(
            "stability",
            "microsome_stability_class",
            "rat_liver_microsome",
            classification,
            "class_label",
            species="Rattus norvegicus",
            matrix="liver_microsome",
        ),
        "solubility_aqsoldb": TDCSpec(
            "solubility", "aqueous_solubility", "aqsoldb_log_scale", regression, "source_native_scale"
        ),
        "vdss_lombardo": TDCSpec(
            "distribution", "vdss", "lombardo_context_incomplete", regression, "source_native_scale"
        ),
    }


TDC_SPECS = _tdc_specs()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _stable_id(prefix: str, *parts: object) -> str:
    body = "\x1f".join("" if item is None else str(item) for item in parts)
    return f"{prefix}-{hashlib.sha256(body.encode('utf-8')).hexdigest()[:24].upper()}"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _schema_sha256(schema: pa.Schema) -> str:
    return hashlib.sha256(schema.serialize().to_pybytes()).hexdigest()


def _clean(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).strip()
    return text or None


def _finite(value: object) -> float | None:
    if value is None:
        return None
    try:
        number = float(str(value))
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


_NUMERIC_TEXT_RE = re.compile(
    r"^\s*(?P<relation><=|>=|<|>|~)?\s*"
    r"(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*$"
)


def parse_numeric_relation(value: object) -> tuple[str, float, float | None, float | None, bool] | None:
    """Parse an exact/censored scalar without inventing an interval."""

    text = _clean(value)
    if text is None:
        return None
    match = _NUMERIC_TEXT_RE.match(text)
    if not match:
        return None
    number = float(match.group("value"))
    if not math.isfinite(number):
        return None
    relation = match.group("relation") or "="
    lower = number if relation in {">", ">="} else None
    upper = number if relation in {"<", "<="} else None
    return relation, number, lower, upper, relation in {"<", "<=", ">", ">="}


def relation_bounds(
    relation: object,
    value: float,
    upper_value: object = None,
) -> tuple[str, float | None, float | None, bool]:
    """Normalize a source relation and optional interval upper bound."""

    relation_text = (_clean(relation) or "unknown").replace("≤", "<=").replace("≥", ">=")
    if upper_value is not None and (upper := _finite(upper_value)) is not None and upper >= value:
        return "interval", value, upper, True
    if relation_text in {"=", "<", "<=", ">", ">=", "~"}:
        lower = value if relation_text in {">", ">="} else None
        upper = value if relation_text in {"<", "<="} else None
        return relation_text, lower, upper, relation_text in {"<", "<=", ">", ">="}
    return relation_text, None, None, False


def _slug(value: str) -> str:
    expanded = value.casefold().replace("%", " percent ")
    slug = re.sub(r"[^a-z0-9]+", "_", expanded).strip("_")
    return slug or "unspecified"


def _normalize_species(value: object) -> str | None:
    text = _clean(value)
    if text is None:
        return None
    aliases = {
        "human": "Homo sapiens",
        "mouse": "Mus musculus",
        "rat": "Rattus norvegicus",
        "dog": "Canis lupus familiaris",
        "monkey": "nonhuman primate",
    }
    return aliases.get(text.casefold(), text)


def _context_matrix(*values: object) -> str | None:
    texts = [item for item in (_clean(value) for value in values) if item]
    blob = " ".join(texts).casefold()
    if not blob:
        return None
    ordered = (
        ("liver_microsome", ("liver microsom", "hepatic microsom")),
        ("microsome", ("microsom",)),
        ("hepatocyte", ("hepatocyte",)),
        ("plasma", ("plasma",)),
        ("serum", ("serum",)),
        ("brain", ("brain", "cerebell")),
        ("liver", ("liver", "hepatic")),
        ("caco2", ("caco-2", "caco2")),
        ("mdck", ("mdck",)),
        ("artificial_membrane", ("pampa", "artificial membrane")),
        ("intestinal_fluid", ("intestinal fluid",)),
        ("gastric_fluid", ("gastric fluid",)),
        ("urine", ("urine", "urinary")),
    )
    for label, patterns in ordered:
        if any(pattern in blob for pattern in patterns):
            return label
    # Do not turn a free-text assay description into a matrix category.  Any
    # unrecognized explicit tissue remains available in context_json.
    return None


def _route_from_text(value: object) -> str | None:
    text = (_clean(value) or "").casefold()
    if not text:
        return None
    patterns = {
        "intravenous": r"\b(?:iv|i\.v\.)\b|intravenous",
        "oral": r"\bpo\b|\bp\.o\.\b|\boral(?:ly)?\b",
        "intraperitoneal": r"\bip\b|\bi\.p\.\b|intraperitoneal",
        "subcutaneous": r"\bsc\b|\bs\.c\.\b|subcutaneous",
        "inhaled": r"\binhal(?:ed|ation)\b",
    }
    matches = [label for label, pattern in patterns.items() if re.search(pattern, text)]
    return matches[0] if len(matches) == 1 else ("multiple" if matches else None)


_DOSE_RE = re.compile(
    r"(?<![\d.])(?P<value>\d+(?:\.\d+)?)\s*"
    r"(?P<mass>mg|ug|µg|μg|g)\s*/\s*(?P<basis>kg|m2|m\^2)\b",
    re.IGNORECASE,
)


def _dose_from_text(value: object) -> tuple[float | None, str | None]:
    text = _clean(value)
    if text is None:
        return None, None
    match = _DOSE_RE.search(text)
    if not match:
        return None, None
    number = float(match.group("value"))
    mass = match.group("mass").casefold()
    basis = match.group("basis").casefold().replace("^", "")
    if mass in {"ug", "µg", "μg"}:
        number /= 1000.0
    elif mass == "g":
        number *= 1000.0
    return number, f"mg/{basis}"


_TIME_RE = re.compile(
    r"(?:after|for|up\s*to|upto|incubat(?:ed|ion)\s*(?:for)?)\s*"
    r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>min(?:ute)?s?|h(?:ou)?rs?|days?)\b",
    re.IGNORECASE,
)


def _time_from_text(value: object) -> tuple[float | None, str | None]:
    text = _clean(value)
    if text is None:
        return None, None
    match = _TIME_RE.search(text)
    if not match:
        return None, None
    number = float(match.group("value"))
    unit = match.group("unit").casefold()
    if unit.startswith("min"):
        number /= 60.0
    elif unit.startswith("day"):
        number *= 24.0
    return number, "h"


_DIRECT_CLINICAL_CONTEXT_RE = re.compile(
    r"\b(?:clinical\s+trials?|first[- ]in[- ]human|patients?|human\s+subjects?|volunteers?)\b",
    re.IGNORECASE,
)
_PHASE_CLINICAL_CONTEXT_RE = re.compile(
    r"\bphase\s*(?:i{1,3}(?![a-z])|[1-3]\b)(?:\s*/\s*(?:i{1,3}(?![a-z])|[1-3]\b))?[^.]{0,60}"
    r"(?:trials?|stud(?:y|ies)|oral|dose|humans?|volunteers?|subjects?)\b",
    re.IGNORECASE,
)
_PERSON_CONTEXT_RE = re.compile(
    r"\b(?:patients?|volunteers?|subjects?|individuals?|persons?|adults?|adolescents?|infants?|pediatric|children|elderly|men|women|boys?|girls?|pregnant)\b",
    re.IGNORECASE,
)
_ADMINISTRATION_CONTEXT_RE = re.compile(
    r"\b(?:dose[ds]?|dosing|administration|administered|coadministered|co-administered|co[- ]treat(?:ed|ment)|treated|oral(?:ly)?|perorally|intravenous|intramuscular|intranasal|subcutaneous|i\.v\.|i\.m\.|s\.c\.|\biv\b|\bim\b|\bsc\b|\bpo\b|topical(?:ly)?|infus(?:ion|ed)|formulations?|mg/kg|q\d+h|qd|bid|tid|day\s*\d+)\b",
    re.IGNORECASE,
)
_NONHUMAN_ANIMAL_RE = re.compile(
    r"\b(?:mouse|mice|rat|rats|pig|pigs|dog|dogs|canine|monkey|monkeys|primate|non[- ]?human)\b",
    re.IGNORECASE,
)
_HARD_LAB_CONTEXT_RE = re.compile(
    r"\b(?:microsomes?|hepatocytes?|hydrolysis|(?:pre)?incubat(?:e|ed|es|ion)|stability|s9|udpga|in\s+vitro|caco-?2|permeability|recombinant|cell\s+lines?|surface\s+plasmon|binding\s+assay|dissociation)\b|\b\d+(?:\.\d+)?\s*(?:nM|uM|µM|mM)\b",
    re.IGNORECASE,
)
_AMBIGUOUS_BIOLOGICAL_CONTEXT_RE = re.compile(
    r"\b(?:metabolites?|metabolism|metabolic|membrane)\b",
    re.IGNORECASE,
)
_SIMULATION_CONTEXT_RE = re.compile(
    r"\b(?:simulat(?:e|ed|es|ion)|in\s+silico|one[- ]compartment|pk/pd\s+model)\b",
    re.IGNORECASE,
)
_QT_ECG_CONTEXT_RE = re.compile(
    r"\b(?:qtc?[bf]?|electrocardiograms?|ecg)\b",
    re.IGNORECASE,
)


def _source_context_flags(
    context: Mapping[str, Any] | None,
    species: str | None = None,
) -> tuple[bool, bool]:
    """Identify source context without converting it into an outcome label."""

    text = " ".join(str(value) for value in (context or {}).values() if value is not None)
    # Microbiology "clinical isolate" and biochemical "phase I metabolism" are
    # not human clinical PK; phase-I flags therefore require human study language.
    clinical_text = re.sub(r"\bclinical\s+isolates?\b", "", text, flags=re.IGNORECASE)
    direct_clinical = bool(_DIRECT_CLINICAL_CONTEXT_RE.search(clinical_text))
    phase_clinical = bool(_PHASE_CLINICAL_CONTEXT_RE.search(clinical_text))
    person_context = bool(_PERSON_CONTEXT_RE.search(clinical_text))
    administration_context = bool(_ADMINISTRATION_CONTEXT_RE.search(clinical_text))
    species_text = (species or "").casefold()
    human_species = species_text in {"homo sapiens", "human", "humans"}
    animal_context = bool(_NONHUMAN_ANIMAL_RE.search(clinical_text)) or bool(
        species_text and not human_species and species_text not in {"unspecified", "unknown"}
    )
    hard_lab = bool(_HARD_LAB_CONTEXT_RE.search(clinical_text))
    ambiguous_biology = bool(_AMBIGUOUS_BIOLOGICAL_CONTEXT_RE.search(clinical_text))
    simulation = bool(_SIMULATION_CONTEXT_RE.search(clinical_text))
    clinical = (
        direct_clinical
        or phase_clinical
        or (human_species and administration_context)
        or (human_species and person_context)
    )
    # Human-target lab experiments and simulated regimens are not observed
    # clinical PK.  Explicit patient/volunteer/subject/trial evidence can retain
    # a clinical flag, while generic "human" or dose wording cannot.
    explicit_people_or_trial = bool(
        re.search(
            r"\b(?:patients?|volunteers?|human\s+subjects?|clinical\s+trials?)\b",
            clinical_text,
            re.IGNORECASE,
        )
    )
    if simulation and not (explicit_people_or_trial and human_species):
        clinical = False
    if hard_lab and not (explicit_people_or_trial and administration_context):
        clinical = False
    if ambiguous_biology and not (human_species and administration_context):
        clinical = False
    if animal_context and not explicit_people_or_trial:
        clinical = False
    return clinical, bool(_QT_ECG_CONTEXT_RE.search(text))


def standardize_structure(raw_smiles: object) -> StructureIdentity:
    text = _clean(raw_smiles)
    if text is None:
        return StructureIdentity(None, None, None, None, None, None, None, "missing_smiles")
    try:
        with rdBase.BlockLogs():
            molecule = Chem.MolFromSmiles(text)
            if molecule is None:
                return StructureIdentity(None, None, None, None, None, None, None, "invalid_smiles")
            molecule = rdMolStandardize.Cleanup(molecule)
            molecule = rdMolStandardize.FragmentParent(molecule)
            molecule = rdMolStandardize.Uncharger().uncharge(molecule)
            Chem.SanitizeMol(molecule)
            smiles = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
            inchi_key = Chem.MolToInchiKey(molecule)
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=molecule, includeChirality=False)
    except Exception:  # RDKit exposes several wrapped C++ exception types.
        return StructureIdentity(None, None, None, None, None, None, None, "standardization_failed")
    if not smiles or not inchi_key:
        return StructureIdentity(None, None, None, None, None, None, None, "standardization_failed")
    connectivity = inchi_key.split("-", maxsplit=1)[0]
    scaffold_key = scaffold or f"ACYCLIC:{connectivity}"
    return StructureIdentity(
        smiles,
        inchi_key,
        # Representation identity deliberately includes canonical SMILES because
        # Standard InChI can collapse distinct reported tautomer representations.
        # Connectivity leakage grouping still co-locates those forms.
        _stable_id("PKMOL", inchi_key, smiles),
        connectivity,
        _stable_id("PKLEAK", connectivity),
        scaffold_key,
        _stable_id("PKSCAF", scaffold_key),
        "standardized",
    )


def _checked_file(path: Path, *, role: str, suffixes: frozenset[str] | None = None) -> Path:
    if path.is_symlink() or not path.is_file():
        raise PKADMESurfaceError(f"Missing or symlinked {role}: {path}")
    if suffixes and path.suffix.casefold() not in suffixes:
        raise PKADMESurfaceError(f"Unexpected extension for {role}: {path}")
    return path.resolve()


class SurfaceWriter:
    """Stream observations while retaining only registries and counters."""

    def __init__(self, output_path: Path, *, batch_size: int = 10_000) -> None:
        self.output_path = output_path
        self.batch_size = batch_size
        self._writer = pq.ParquetWriter(output_path, _OBSERVATION_SCHEMA, compression="zstd")
        self._batch: list[dict[str, Any]] = []
        self.structure_cache: dict[str, StructureIdentity] = {}
        self.tasks: dict[str, TaskAccumulator] = {}
        self.molecules: dict[str, dict[str, Any]] = {}
        self.dispositions: dict[tuple[str, str], DispositionAccumulator] = defaultdict(DispositionAccumulator)
        self.observation_ids: set[str] = set()
        self.total_observations = 0
        self.clinical_pk_context_observations = 0
        self.qt_ecg_context_observations = 0

    def structure(self, raw_smiles: object) -> StructureIdentity:
        key = _clean(raw_smiles) or ""
        if key not in self.structure_cache:
            self.structure_cache[key] = standardize_structure(key or None)
        return self.structure_cache[key]

    def disposition(self, source_family: str, source_dataset: str) -> DispositionAccumulator:
        return self.dispositions[(source_family, source_dataset)]

    def emit(
        self,
        *,
        source_family: str,
        source_dataset: str,
        source_version: str,
        source_file: str,
        source_row_number: int,
        source_record_id: str,
        source_split: str | None,
        source_rights_status: str,
        source_lineage: str,
        raw_smiles: object,
        task_id: str,
        endpoint_family: str,
        endpoint_name: str,
        endpoint_variant: str,
        task_type: str,
        source_native_endpoint: str,
        original_value_text: object,
        original_numeric_value: float | None,
        original_unit: str | None,
        relation: str,
        normalized_value: float,
        normalized_unit: str,
        lower_bound: float | None,
        upper_bound: float | None,
        is_censored: bool,
        species: str | None,
        matrix: str | None,
        route: str | None,
        dose_value: float | None,
        dose_unit: str | None,
        time_value: float | None,
        time_unit: str | None,
        assay_id: str | None,
        document_id: str | None,
        context_complete: bool,
        quality_flags: Iterable[str] = (),
        context: Mapping[str, Any] | None = None,
        provenance: Mapping[str, Any] | None = None,
        uncertainty_lower: float | None = None,
        uncertainty_upper: float | None = None,
        cross_source_union_candidate: bool = True,
    ) -> None:
        if not math.isfinite(normalized_value):
            raise PKADMESurfaceError(f"Non-finite normalized value for {source_dataset}:{source_record_id}")
        identity = self.structure(raw_smiles)
        context_payload = dict(context or {})
        clinical_pk_context, qt_ecg_context = _source_context_flags(context_payload, species)
        flags = sorted(set(quality_flags))
        if clinical_pk_context:
            flags.append("clinical_pk_source_context")
        if qt_ecg_context:
            flags.append("qt_or_ecg_context_not_target")
        if identity.status != "standardized":
            flags.append(identity.status)
        recognized_relation = relation in {"=", "<", "<=", ">", ">="}
        fatal_quality = any(
            flag
            in {
                "data_validity_comment",
                "potential_duplicate",
                "nonstandard_chembl_activity",
                "out_of_endpoint_domain",
                "source_value_unparseable",
                "source_qc_fail",
            }
            for flag in flags
        )
        exact = (
            identity.status == "standardized"
            and not fatal_quality
            and relation == "="
            and (task_type != "classification" or normalized_value in {0.0, 1.0})
        )
        censored_eligible = (
            identity.status == "standardized"
            and not fatal_quality
            and recognized_relation
            and task_type == "regression"
        )
        if task_type == "classification":
            censored_eligible = exact
        cross_source = exact and cross_source_union_candidate
        if exact:
            modeling_status = "classification" if task_type == "classification" else "exact_regression"
        elif censored_eligible and is_censored:
            modeling_status = "censored_regression"
        else:
            modeling_status = "quarantine"
        observation_id = _stable_id(
            "PKOBS",
            source_family,
            source_dataset,
            source_file,
            source_row_number,
            source_record_id,
            task_id,
            source_native_endpoint,
        )
        if observation_id in self.observation_ids:
            raise PKADMESurfaceError(f"Duplicate observation ID: {observation_id}")
        self.observation_ids.add(observation_id)
        duplicate_group_id = (
            _stable_id(
                "PKDUP",
                task_id,
                identity.standard_inchi_key,
                relation,
                format(normalized_value, ".15g"),
                normalized_unit,
            )
            if identity.standard_inchi_key
            else None
        )
        record = {
            "observation_id": observation_id,
            "source_family": source_family,
            "source_dataset": source_dataset,
            "source_version": source_version,
            "source_file": source_file,
            "source_row_number": source_row_number,
            "source_record_id": source_record_id,
            "source_split": source_split,
            "source_rights_status": source_rights_status,
            "source_lineage": source_lineage,
            "raw_smiles": _clean(raw_smiles),
            "standardized_smiles": identity.standardized_smiles,
            "standard_inchi_key": identity.standard_inchi_key,
            "molecule_id": identity.molecule_id,
            "connectivity_key": identity.connectivity_key,
            "leakage_group_id": identity.leakage_group_id,
            "scaffold_smiles": identity.scaffold_smiles,
            "scaffold_group_id": identity.scaffold_group_id,
            "structure_status": identity.status,
            "task_id": task_id,
            "endpoint_family": endpoint_family,
            "endpoint_name": endpoint_name,
            "endpoint_variant": endpoint_variant,
            "task_type": task_type,
            "source_native_endpoint": source_native_endpoint,
            "original_value_text": _clean(original_value_text),
            "original_numeric_value": original_numeric_value,
            "original_unit": original_unit,
            "relation": relation,
            "normalized_value": normalized_value,
            "normalized_unit": normalized_unit,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "is_censored": is_censored,
            "uncertainty_lower": uncertainty_lower,
            "uncertainty_upper": uncertainty_upper,
            "species": species,
            "matrix": matrix,
            "route": route,
            "dose_value": dose_value,
            "dose_unit": dose_unit,
            "time_value": time_value,
            "time_unit": time_unit,
            "assay_id": assay_id,
            "document_id": document_id,
            "context_complete": context_complete,
            "clinical_pk_context": clinical_pk_context,
            "qt_ecg_context": qt_ecg_context,
            "eligible_exact_modeling": exact,
            "eligible_censored_modeling": censored_eligible,
            "eligible_cross_source_union": cross_source,
            "canonical_admission": False,
            "modeling_status": modeling_status,
            "duplicate_group_id": duplicate_group_id,
            "quality_flags_json": _canonical_json(sorted(set(flags))),
            "context_json": _canonical_json(context_payload),
            "provenance_json": _canonical_json(dict(provenance or {})),
        }
        self._batch.append(record)
        self.total_observations += 1
        self.clinical_pk_context_observations += int(clinical_pk_context)
        self.qt_ecg_context_observations += int(qt_ecg_context)
        task = self.tasks.setdefault(
            task_id,
            TaskAccumulator(endpoint_family, endpoint_name, endpoint_variant, task_type, normalized_unit),
        )
        if (
            task.endpoint_family,
            task.endpoint_name,
            task.endpoint_variant,
            task.task_type,
            task.normalized_unit,
        ) != (endpoint_family, endpoint_name, endpoint_variant, task_type, normalized_unit):
            raise PKADMESurfaceError(f"Inconsistent task contract for {task_id}")
        task.source_families.add(source_family)
        task.source_datasets.add(source_dataset)
        task.rights_statuses.add(source_rights_status)
        if species:
            task.species.add(species)
        if matrix:
            task.matrices.add(matrix)
        if route:
            task.routes.add(route)
        task.observation_count += 1
        if identity.molecule_id:
            task.valid_structure_count += 1
            task.molecules.add(identity.molecule_id)
            task.connectivity_groups.add(identity.leakage_group_id or "")
            molecule = self.molecules.setdefault(
                identity.molecule_id,
                {
                    "molecule_id": identity.molecule_id,
                    "standardized_smiles": identity.standardized_smiles,
                    "standard_inchi_key": identity.standard_inchi_key,
                    "connectivity_key": identity.connectivity_key,
                    "leakage_group_id": identity.leakage_group_id,
                    "scaffold_smiles": identity.scaffold_smiles,
                    "scaffold_group_id": identity.scaffold_group_id,
                    "source_families": set(),
                    "observation_count": 0,
                },
            )
            molecule["source_families"].add(source_family)
            molecule["observation_count"] += 1
        if exact:
            task.exact_modeling_count += 1
        elif censored_eligible and is_censored:
            task.censored_only_modeling_count += 1
        if context_complete:
            task.context_complete_count += 1
        if cross_source:
            task.cross_source_union_count += 1
        if "chembl" in source_lineage.casefold() and source_family != "chembl_37":
            task.lineage_overlap_risk = True
        disposition = self.disposition(source_family, source_dataset)
        disposition.emitted_measurement_count += 1
        if exact:
            disposition.exact_modeling_count += 1
        elif censored_eligible and is_censored:
            disposition.censored_only_modeling_count += 1
        if identity.status != "standardized":
            disposition.missing_or_invalid_structure_count += 1
            disposition.reasons[identity.status] += 1
        if len(self._batch) >= self.batch_size:
            self.flush()

    def flush(self) -> None:
        if not self._batch:
            return
        self._writer.write_table(pa.Table.from_pylist(self._batch, schema=_OBSERVATION_SCHEMA))
        self._batch.clear()

    def close(self) -> None:
        self.flush()
        self._writer.close()


def _chembl_spec(
    standard_type: object,
    standard_unit: object,
    assay_description: object,
    assay_tissue: object,
    assay_subcellular_fraction: object,
) -> tuple[str, str, str, str, str, float] | None:
    """Return endpoint contract plus unit multiplier for a safe exact type/unit pair."""

    endpoint = _clean(standard_type)
    unit = _clean(standard_unit)
    description = _clean(assay_description) or ""
    matrix = _context_matrix(assay_subcellular_fraction, assay_tissue, description)
    if endpoint == "T1/2" and unit == "hr":
        variant = "microsomal" if matrix in {"microsome", "liver_microsome"} else (matrix or "source_context")
        return "half_life", "half_life", variant, "h", "regression", 1.0
    if endpoint == "CL" and unit == "mL.min-1.kg-1":
        if matrix in {"microsome", "liver_microsome"} or "intrinsic clearance" in description.casefold():
            variant = "intrinsic_scaled_body_weight"
        elif matrix == "hepatocyte":
            variant = "hepatocyte_scaled_body_weight"
        else:
            variant = "systemic_or_total"
        return "clearance", "clearance", variant, "mL/min/kg", "regression", 1.0
    if endpoint == "CL" and unit == "mL.min-1.g-1":
        return "clearance", "intrinsic_clearance", "microsome_or_tissue", "mL/min/g", "regression", 1.0
    if endpoint == "CL" and unit == "uL.min-1.(10^6cells)-1":
        return "clearance", "intrinsic_clearance", "hepatocyte", "uL/min/1e6_cells", "regression", 1.0
    if endpoint in {"Solubility", "solubility"} and unit == "nM":
        return "solubility", "solubility", "molar_source_context", "nM", "regression", 1.0
    if endpoint in {"Solubility", "solubility"} and unit == "ug.mL-1":
        return "solubility", "solubility", "mass_source_context", "ug/mL", "regression", 1.0
    if endpoint == "AUC" and unit == "ng.hr.mL-1":
        return "exposure", "auc", "source_window", "ng*h/mL", "regression", 1.0
    if endpoint == "F" and unit == "%":
        return "bioavailability", "bioavailability", "reported_or_absolute", "%", "regression", 1.0
    if endpoint in {"LogD", "logD"} and unit is None:
        return "lipophilicity", "logd", "source_ph", "unitless", "regression", 1.0
    if endpoint == "Cmax" and unit == "nM":
        return "exposure", "cmax", "molar", "nM", "regression", 1.0
    if endpoint == "Cmax" and unit == "ug.mL-1":
        return "exposure", "cmax", "mass", "ug/mL", "regression", 1.0
    if endpoint == "Papp" and unit in {"10^-6 cm/s", "10'-6 cm/s", "ucm/s", "cm/s * 10E6"}:
        system = matrix or "source_system"
        direction = (
            "btoa"
            if re.search(r"basolateral\s+to\s+apical|b\s*(?:to|->)\s*a", description, re.I)
            else (
                "atob"
                if re.search(r"apical\s+to\s+basolateral|a\s*(?:to|->)\s*b", description, re.I)
                else "unspecified_direction"
            )
        )
        return "permeability", "papp", f"{system}_{direction}", "1e-6_cm/s", "regression", 1.0
    if endpoint == "Papp" and unit == "nm/s":
        return (
            "permeability",
            "papp",
            f"{matrix or 'source_system'}_unspecified_direction",
            "1e-6_cm/s",
            "regression",
            0.1,
        )
    if endpoint == "Papp" and unit == "cm/s":
        return (
            "permeability",
            "papp",
            f"{matrix or 'source_system'}_unspecified_direction",
            "1e-6_cm/s",
            "regression",
            1_000_000.0,
        )
    if endpoint in {"LogP", "logP"} and unit is None:
        return "lipophilicity", "logp", "source_method", "unitless", "regression", 1.0
    if endpoint == "Vdss" and unit == "L.kg-1":
        return "distribution", "vdss", "reported", "L/kg", "regression", 1.0
    if endpoint == "Tmax" and unit == "hr":
        return "exposure", "tmax", "reported", "h", "regression", 1.0
    if endpoint == "PPB" and unit == "%":
        return "protein_binding", "protein_binding", matrix or "source_matrix", "%_bound", "regression", 1.0
    if endpoint == "Fu" and unit is None:
        return "protein_binding", "fraction_unbound", matrix or "source_matrix", "fraction", "regression", 1.0
    if endpoint == "Vd" and unit == "L.kg-1":
        return "distribution", "vd", "reported", "L/kg", "regression", 1.0
    if endpoint == "Peff" and unit in {"10^-6 cm/s", "10'-6 cm/s", "ucm/s"}:
        return "permeability", "peff", matrix or "source_system", "1e-6_cm/s", "regression", 1.0
    if endpoint == "Peff" and unit == "nm/s":
        return "permeability", "peff", matrix or "source_system", "1e-6_cm/s", "regression", 0.1
    if endpoint == "Peff" and unit == "cm/s":
        return "permeability", "peff", matrix or "source_system", "1e-6_cm/s", "regression", 1_000_000.0
    if endpoint == "CL_renal" and unit == "mL.min-1.kg-1":
        return "clearance", "renal_clearance", "reported", "mL/min/kg", "regression", 1.0
    if endpoint == "CLint" and unit == "uL/min/1E6 cells":
        return "clearance", "intrinsic_clearance", "hepatocyte", "uL/min/1e6_cells", "regression", 1.0
    if endpoint == "CLint" and unit == "uL min-1 mg-1":
        return "clearance", "intrinsic_clearance", "microsome", "uL/min/mg", "regression", 1.0
    return None


def _in_endpoint_domain(endpoint_name: str, value: float, normalized_unit: str) -> bool:
    if endpoint_name in {"logd", "logp"}:
        return True
    if value < 0:
        return False
    if normalized_unit in {"%", "%_bound"}:
        return value <= 100.0
    if normalized_unit == "fraction":
        return value <= 1.0
    return True


_CHEMBL_COLUMNS = [
    "activity_id",
    "standard_type",
    "standard_units",
    "standard_value",
    "standard_upper_value",
    "standard_relation",
    "standard_flag",
    "potential_duplicate",
    "data_validity_comment",
    "canonical_smiles",
    "standard_inchi_key",
    "molecule_chembl_id",
    "assay_chembl_id",
    "assay_description",
    "assay_organism",
    "assay_tissue",
    "assay_cell_type",
    "assay_subcellular_fraction",
    "document_chembl_id",
    "document_doi",
    "pubmed_id",
    "patent_id",
    "activity_source_name",
]


def _load_chembl(writer: SurfaceWriter, repo_root: Path, bindings: list[dict[str, Any]]) -> None:
    manifest_path = (
        repo_root
        / "research/data/platform/interim/chembl_37_bulk/specialized_views/pk_adme_candidates_manifest.json"
    )
    manifest = json.loads(_checked_file(manifest_path, role="ChEMBL PK/ADME manifest").read_text())
    bindings.append(_bind_file(repo_root, manifest_path, "input_manifest", None))
    physical_rows = 0
    selected_rows = 0
    emitted = 0
    for part in manifest["parts"]:
        part_path = manifest_path.parent / part["path"]
        binding = _bind_file(repo_root, part_path, "chembl_candidate_parquet", part["sha256"])
        bindings.append(binding)
        physical_rows += int(part["rows"])
        parquet_file = pq.ParquetFile(part_path)
        part_row_number = 1
        for batch in parquet_file.iter_batches(batch_size=25_000, columns=_CHEMBL_COLUMNS):
            columns = batch.to_pydict()
            for offset in range(batch.num_rows):
                part_row_number += 1
                row = {name: columns[name][offset] for name in _CHEMBL_COLUMNS}
                spec = _chembl_spec(
                    row["standard_type"],
                    row["standard_units"],
                    row["assay_description"],
                    row["assay_tissue"],
                    row["assay_subcellular_fraction"],
                )
                if spec is None:
                    continue
                selected_rows += 1
                original_value = _finite(row["standard_value"])
                if original_value is None:
                    continue
                endpoint_family, endpoint_name, endpoint_variant, normalized_unit, task_type, multiplier = (
                    spec
                )
                normalized_value = original_value * multiplier
                scaled_upper = (
                    float(row["standard_upper_value"]) * multiplier
                    if _finite(row["standard_upper_value"]) is not None
                    else None
                )
                relation, lower, upper, censored = relation_bounds(
                    row["standard_relation"], normalized_value, scaled_upper
                )
                flags: list[str] = []
                if row["standard_flag"] != 1:
                    flags.append("nonstandard_chembl_activity")
                if row["potential_duplicate"] == 1:
                    flags.append("potential_duplicate")
                if _clean(row["data_validity_comment"]):
                    flags.append("data_validity_comment")
                if relation not in {"=", "<", "<=", ">", ">="}:
                    flags.append("unsupported_relation")
                if not _in_endpoint_domain(endpoint_name, normalized_value, normalized_unit):
                    flags.append("out_of_endpoint_domain")
                description = _clean(row["assay_description"])
                species = _normalize_species(row["assay_organism"])
                matrix = _context_matrix(
                    row["assay_subcellular_fraction"],
                    row["assay_tissue"],
                    row["assay_cell_type"],
                    description,
                )
                route = _route_from_text(description)
                dose_value, dose_unit = _dose_from_text(description)
                time_value, time_unit = _time_from_text(description)
                if endpoint_family == "exposure" and endpoint_name in {"auc", "cmax", "tmax"}:
                    context_complete = bool(species and route and dose_value is not None)
                elif endpoint_name in {"vd", "vdss", "bioavailability"} or (
                    endpoint_name == "clearance" and endpoint_variant == "systemic_or_total"
                ):
                    context_complete = bool(species and route)
                elif endpoint_family in {"clearance", "half_life", "protein_binding"}:
                    context_complete = bool(species and matrix)
                else:
                    context_complete = True
                if not context_complete:
                    flags.append(
                        "context_incomplete_for_absolute_pk"
                        if endpoint_family in {"exposure", "bioavailability", "distribution"}
                        else "context_partial"
                    )
                task_id = (
                    f"chembl37__{_slug(endpoint_name)}__{_slug(endpoint_variant)}__{_slug(normalized_unit)}"
                )
                source_file = str(part_path.relative_to(repo_root))
                activity_id = str(row["activity_id"])
                writer.emit(
                    source_family="chembl_37",
                    source_dataset="pk_adme_candidates_reclassified",
                    source_version="ChEMBL_37",
                    source_file=source_file,
                    source_row_number=part_row_number,
                    source_record_id=f"activity:{activity_id}",
                    source_split=None,
                    source_rights_status="CC_BY_SA_3_0_attribution_sharealike_required",
                    source_lineage="direct_chembl_37_activity",
                    raw_smiles=row["canonical_smiles"],
                    task_id=task_id,
                    endpoint_family=endpoint_family,
                    endpoint_name=endpoint_name,
                    endpoint_variant=endpoint_variant,
                    task_type=task_type,
                    source_native_endpoint=_clean(row["standard_type"]) or "unknown",
                    original_value_text=str(original_value),
                    original_numeric_value=original_value,
                    original_unit=_clean(row["standard_units"]),
                    relation=relation,
                    normalized_value=normalized_value,
                    normalized_unit=normalized_unit,
                    lower_bound=lower,
                    upper_bound=upper,
                    is_censored=censored,
                    species=species,
                    matrix=matrix,
                    route=route,
                    dose_value=dose_value,
                    dose_unit=dose_unit,
                    time_value=time_value,
                    time_unit=time_unit,
                    assay_id=_clean(row["assay_chembl_id"]),
                    document_id=_clean(row["document_chembl_id"]),
                    context_complete=context_complete,
                    quality_flags=flags,
                    context={
                        "assay_description": description,
                        "assay_tissue": _clean(row["assay_tissue"]),
                        "assay_cell_type": _clean(row["assay_cell_type"]),
                        "assay_subcellular_fraction": _clean(row["assay_subcellular_fraction"]),
                    },
                    provenance={
                        "activity_id": row["activity_id"],
                        "molecule_chembl_id": _clean(row["molecule_chembl_id"]),
                        "reported_standard_inchi_key": _clean(row["standard_inchi_key"]),
                        "document_doi": _clean(row["document_doi"]),
                        "pubmed_id": row["pubmed_id"],
                        "patent_id": _clean(row["patent_id"]),
                        "activity_source_name": _clean(row["activity_source_name"]),
                    },
                )
                emitted += 1
    disposition = writer.disposition("chembl_37", "pk_adme_candidates_reclassified")
    disposition.physical_file_rows = physical_rows
    disposition.candidate_record_rows = selected_rows
    disposition.excluded_candidate_count = selected_rows - emitted
    disposition.reasons["not_selected_by_case_sensitive_endpoint_unit_contract"] += (
        physical_rows - selected_rows
    )
    disposition.reasons["selected_candidate_not_emitted_missing_or_invalid_numeric"] += (
        selected_rows - emitted
    )
    disposition.double_count_policy = (
        "one ChEMBL activity_id and exact standardized endpoint; case-sensitive CL excludes chloride-like Cl"
    )


def _load_tdc(writer: SurfaceWriter, repo_root: Path, bindings: list[dict[str, Any]]) -> None:
    root = repo_root / "research/data/platform/raw/external_public/pk_expansion/avicenna/tdc_adme"
    manifest_path = root / "manifest.json"
    manifest = json.loads(_checked_file(manifest_path, role="TDC manifest").read_text())
    bindings.append(_bind_file(repo_root, manifest_path, "input_manifest", None))
    declared = {item["path"]: item["sha256"] for item in manifest["artifacts"]}
    discovered = {path.stem for path in (root / "files").glob("*.tab")}
    missing_specs = discovered - set(TDC_SPECS)
    if missing_specs:
        raise PKADMESurfaceError(f"TDC task specifications missing: {sorted(missing_specs)}")
    for dataset_name, spec in sorted(TDC_SPECS.items()):
        path = root / "files" / f"{dataset_name}.tab"
        relative = str(path.relative_to(root))
        bindings.append(_bind_file(repo_root, path, "tdc_source_native_table", declared.get(relative)))
        disposition = writer.disposition("tdc_adme", dataset_name)
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            fields = set(reader.fieldnames or [])
            id_field = "Drug_ID" if "Drug_ID" in fields else ("ID" if "ID" in fields else None)
            smiles_field = "Drug" if "Drug" in fields else ("X" if "X" in fields else None)
            if id_field is None or smiles_field is None or "Y" not in fields:
                raise PKADMESurfaceError(f"Unexpected TDC schema for {path}")
            row_count = 0
            for row_number, row in enumerate(reader, start=2):
                row_count += 1
                disposition.candidate_record_rows += 1
                value = _finite(row.get("Y"))
                if value is None:
                    disposition.excluded_candidate_count += 1
                    disposition.reasons["missing_or_nonfinite_y"] += 1
                    continue
                flags = (
                    ["source_native_unit_missing"] if spec.normalized_unit == "source_native_scale" else []
                )
                if spec.task_type == "classification" and value not in {0.0, 1.0}:
                    flags.append("out_of_endpoint_domain")
                species = _normalize_species(row.get("Species")) or spec.species
                context_complete = spec.endpoint_name not in {
                    "bioavailability_class",
                    "reported_half_life",
                    "vdss",
                }
                if not context_complete:
                    flags.append("context_incomplete_for_absolute_pk")
                task_id = f"tdc__{dataset_name}"
                writer.emit(
                    source_family="tdc_adme",
                    source_dataset=dataset_name,
                    source_version="c310c35f27e3f506411018ac43d97b8ba23ca652",
                    source_file=str(path.relative_to(repo_root)),
                    source_row_number=row_number,
                    source_record_id=_clean(row.get(id_field)) or f"row:{row_number}",
                    source_split=None,
                    source_rights_status="TDC_code_MIT_underlying_dataset_rights_conditional",
                    source_lineage=spec.lineage,
                    raw_smiles=row.get(smiles_field),
                    task_id=task_id,
                    endpoint_family=spec.endpoint_family,
                    endpoint_name=spec.endpoint_name,
                    endpoint_variant=spec.endpoint_variant,
                    task_type=spec.task_type,
                    source_native_endpoint="Y",
                    original_value_text=row.get("Y"),
                    original_numeric_value=value,
                    original_unit=None,
                    relation="=",
                    normalized_value=value,
                    normalized_unit=spec.normalized_unit,
                    lower_bound=None,
                    upper_bound=None,
                    is_censored=False,
                    species=species,
                    matrix=spec.matrix,
                    route=None,
                    dose_value=None,
                    dose_unit=None,
                    time_value=None,
                    time_unit=None,
                    assay_id=None,
                    document_id=None,
                    context_complete=context_complete,
                    quality_flags=flags,
                    context={
                        "source_columns": reader.fieldnames,
                        "unit_column_present": False,
                        "route_column_present": False,
                        "dose_column_present": False,
                        "matrix_from_dataset_contract": spec.matrix,
                    },
                    provenance={"drug_id": _clean(row.get(id_field)), "source_id_field": id_field},
                    cross_source_union_candidate=False,
                )
        disposition.physical_file_rows = row_count
        disposition.double_count_policy = "one TDC table row equals one source-native benchmark label; duplicate structures retained but leakage-grouped"


@dataclass(frozen=True)
class PairedEndpointSpec:
    column: str
    endpoint_family: str
    endpoint_name: str
    endpoint_variant: str
    normalized_unit: str
    species: str | None = None
    matrix: str | None = None


def _load_paired_openadmet(
    writer: SurfaceWriter,
    repo_root: Path,
    bindings: list[dict[str, Any]],
    *,
    dataset_name: str,
    x_path: Path,
    y_path: Path,
    endpoint_specs: Sequence[PairedEndpointSpec],
    declared_hashes: Mapping[str, str],
    openadmet_root: Path,
    lineage: str,
    rights: str,
) -> None:
    for path, role in ((x_path, "paired_structure_table"), (y_path, "paired_label_table")):
        bindings.append(
            _bind_file(repo_root, path, role, declared_hashes.get(str(path.relative_to(openadmet_root))))
        )
    disposition = writer.disposition("openadmet", dataset_name)
    with (
        x_path.open("r", encoding="utf-8", newline="") as x_handle,
        y_path.open("r", encoding="utf-8", newline="") as y_handle,
    ):
        x_reader = csv.DictReader(x_handle)
        y_reader = csv.DictReader(y_handle)
        if not x_reader.fieldnames or len(x_reader.fieldnames) != 1:
            raise PKADMESurfaceError(f"Unexpected paired OpenADMET X schema: {x_path}")
        smiles_field = x_reader.fieldnames[0]
        y_fields = set(y_reader.fieldnames or [])
        expected_fields = {spec.column for spec in endpoint_specs}
        if not expected_fields.issubset(y_fields):
            raise PKADMESurfaceError(f"Missing OpenADMET label fields in {y_path}")
        row_count = 0
        for row_number, pair in enumerate(zip(x_reader, y_reader, strict=True), start=2):
            x_row, y_row = pair
            row_count += 1
            disposition.candidate_record_rows += 1
            for spec in endpoint_specs:
                parsed = parse_numeric_relation(y_row.get(spec.column))
                if parsed is None:
                    continue
                relation, value, lower, upper, censored = parsed
                task_id = f"openadmet__{dataset_name}__{_slug(spec.column)}"
                writer.emit(
                    source_family="openadmet",
                    source_dataset=dataset_name,
                    source_version="frozen_huggingface_commit",
                    source_file=str(y_path.relative_to(repo_root)),
                    source_row_number=row_number,
                    source_record_id=f"paired_row:{row_number - 1}",
                    source_split="train_only_as_published",
                    source_rights_status=rights,
                    source_lineage=lineage,
                    raw_smiles=x_row.get(smiles_field),
                    task_id=task_id,
                    endpoint_family=spec.endpoint_family,
                    endpoint_name=spec.endpoint_name,
                    endpoint_variant=spec.endpoint_variant,
                    task_type="regression",
                    source_native_endpoint=spec.column,
                    original_value_text=y_row.get(spec.column),
                    original_numeric_value=value,
                    original_unit=spec.normalized_unit,
                    relation=relation,
                    normalized_value=value,
                    normalized_unit=spec.normalized_unit,
                    lower_bound=lower,
                    upper_bound=upper,
                    is_censored=censored,
                    species=spec.species,
                    matrix=spec.matrix,
                    route=None,
                    dose_value=None,
                    dose_unit=None,
                    time_value=None,
                    time_unit=None,
                    assay_id=None,
                    document_id=None,
                    context_complete=True,
                    context={
                        "paired_structure_file": str(x_path.relative_to(repo_root)),
                        "join_contract": "strict_row_order",
                    },
                    provenance={"paired_row_index": row_number - 1},
                    cross_source_union_candidate=False,
                )
    disposition.physical_file_rows = row_count * 2
    disposition.candidate_record_rows = row_count * len(endpoint_specs)
    disposition.excluded_candidate_count = (
        disposition.candidate_record_rows - disposition.emitted_measurement_count
    )
    disposition.reasons["null_label_field"] += disposition.excluded_candidate_count
    disposition.double_count_policy = (
        "paired X and Y rows count once; only non-null endpoint fields become measurements"
    )


def _load_openadmet_expansion(
    writer: SurfaceWriter,
    repo_root: Path,
    bindings: list[dict[str, Any]],
    openadmet_root: Path,
    declared_hashes: Mapping[str, str],
) -> None:
    path = openadmet_root / "expansionrx_full/expansion_data_raw.csv"
    bindings.append(
        _bind_file(
            repo_root,
            path,
            "openadmet_full_raw_measurements",
            declared_hashes.get(str(path.relative_to(openadmet_root))),
        )
    )
    specs = {
        "LogD": ("lipophilicity", "logd", "expansionrx", "unitless", None, None),
        "KSOL": ("solubility", "kinetic_solubility", "expansionrx", "uM", None, None),
        "HLM CLint": (
            "clearance",
            "intrinsic_clearance",
            "human_liver_microsome",
            "mL/min/kg",
            "Homo sapiens",
            "liver_microsome",
        ),
        "RLM CLint": (
            "clearance",
            "intrinsic_clearance",
            "rat_liver_microsome",
            "mL/min/kg",
            "Rattus norvegicus",
            "liver_microsome",
        ),
        "MLM CLint": (
            "clearance",
            "intrinsic_clearance",
            "mouse_liver_microsome",
            "mL/min/kg",
            "Mus musculus",
            "liver_microsome",
        ),
        "Caco-2 Permeability Papp A>B": (
            "permeability",
            "papp",
            "caco2_atob",
            "1e-6_cm/s",
            "Homo sapiens",
            "caco2",
        ),
        "Caco-2 Permeability Efflux": (
            "permeability",
            "efflux_ratio",
            "caco2",
            "ratio",
            "Homo sapiens",
            "caco2",
        ),
        "MPPB": (
            "protein_binding",
            "fraction_unbound",
            "mouse_plasma_percent",
            "%_unbound",
            "Mus musculus",
            "plasma",
        ),
        "MBPB": (
            "protein_binding",
            "fraction_unbound",
            "mouse_brain_percent",
            "%_unbound",
            "Mus musculus",
            "brain",
        ),
        "MGMB": (
            "protein_binding",
            "fraction_unbound",
            "mouse_muscle_percent",
            "%_unbound",
            "Mus musculus",
            "gastrocnemius_muscle",
        ),
    }
    disposition = writer.disposition("openadmet", "expansionrx_full_raw")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not set(specs).issubset(reader.fieldnames):
            raise PKADMESurfaceError(f"Unexpected ExpansionRx schema: {path}")
        row_count = 0
        for row_number, row in enumerate(reader, start=2):
            row_count += 1
            disposition.candidate_record_rows += 1
            for column, spec in specs.items():
                parsed = parse_numeric_relation(row.get(column))
                if parsed is None:
                    continue
                relation, value, lower, upper, censored = parsed
                endpoint_family, endpoint_name, endpoint_variant, unit, species, matrix = spec
                flags: list[str] = []
                if not _in_endpoint_domain(endpoint_name, value, unit):
                    flags.append("out_of_endpoint_domain")
                task_id = f"openadmet__expansionrx__{_slug(column)}"
                writer.emit(
                    source_family="openadmet",
                    source_dataset="expansionrx_full_raw",
                    source_version="6b898ccc43d10d25b230fb09e22a6e30c30022b5",
                    source_file=str(path.relative_to(repo_root)),
                    source_row_number=row_number,
                    source_record_id=_clean(row.get("Molecule Name")) or f"row:{row_number}",
                    source_split="full_release_raw_no_split",
                    source_rights_status="CC_BY_4_0",
                    source_lineage="direct_expansionrx_experimental_adme",
                    raw_smiles=row.get("SMILES"),
                    task_id=task_id,
                    endpoint_family=endpoint_family,
                    endpoint_name=endpoint_name,
                    endpoint_variant=endpoint_variant,
                    task_type="regression",
                    source_native_endpoint=column,
                    original_value_text=row.get(column),
                    original_numeric_value=value,
                    original_unit=unit,
                    relation=relation,
                    normalized_value=value,
                    normalized_unit=unit,
                    lower_bound=lower,
                    upper_bound=upper,
                    is_censored=censored,
                    species=species,
                    matrix=matrix,
                    route=None,
                    dose_value=None,
                    dose_unit=None,
                    time_value=None,
                    time_unit=None,
                    assay_id=None,
                    document_id=None,
                    context_complete=True,
                    quality_flags=flags,
                    context={"challenge_full_raw": True},
                    provenance={"molecule_name": _clean(row.get("Molecule Name"))},
                )
    disposition.physical_file_rows = row_count
    disposition.candidate_record_rows = row_count * len(specs)
    disposition.excluded_candidate_count = (
        disposition.candidate_record_rows - disposition.emitted_measurement_count
    )
    disposition.reasons["null_endpoint_field"] += disposition.excluded_candidate_count
    disposition.double_count_policy = "full raw release only; overlapping train and test CSVs excluded"


def _octant_structure_map(paths: Sequence[Path]) -> dict[str, str]:
    candidates: dict[str, set[str]] = defaultdict(set)
    for path in paths:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if not reader.fieldnames or not {"ocnt_batch", "standardized_smiles"}.issubset(reader.fieldnames):
                raise PKADMESurfaceError(f"Missing Octant structure mapping columns: {path}")
            for row in reader:
                batch = _clean(row.get("ocnt_batch"))
                smiles = _clean(row.get("standardized_smiles"))
                if batch and smiles:
                    candidates[batch].add(smiles)
    conflicts = {key: values for key, values in candidates.items() if len(values) != 1}
    if conflicts:
        raise PKADMESurfaceError(f"Conflicting Octant structures for {len(conflicts)} batches")
    return {key: next(iter(values)) for key, values in candidates.items()}


def _load_openadmet_octant(
    writer: SurfaceWriter,
    repo_root: Path,
    bindings: list[dict[str, Any]],
    openadmet_root: Path,
    declared_hashes: Mapping[str, str],
) -> None:
    octant = openadmet_root / "octant_cyp"
    mapping_paths = [
        octant / "inhibition.tsv",
        octant / "inhibition_wells.tsv",
        octant / "reactivity_wells.tsv",
        octant / "will_it_fly_in_mass_spec.tsv",
    ]
    reactivity_path = octant / "reactivity.tsv"
    for path in [*mapping_paths, reactivity_path]:
        role = (
            "octant_compound_summary"
            if path.name in {"inhibition.tsv", "reactivity.tsv"}
            else "structure_mapping_only_not_measurement"
        )
        bindings.append(
            _bind_file(repo_root, path, role, declared_hashes.get(str(path.relative_to(openadmet_root))))
        )
    structure_map = _octant_structure_map(mapping_paths)
    for path in mapping_paths[1:]:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = sum(1 for _ in csv.DictReader(handle, delimiter="\t"))
        disposition = writer.disposition("openadmet", f"excluded_{path.stem}")
        disposition.physical_file_rows = rows
        disposition.candidate_record_rows = rows
        disposition.excluded_candidate_count = rows
        disposition.disposition = "excluded_well_or_instrument_level_rows"
        reason = (
            "instrument_compatibility_not_adme_endpoint"
            if "will_it_fly" in path.stem
            else "well_rows_not_compound_level_measurements"
        )
        disposition.reasons[reason] += rows
        disposition.double_count_policy = "file may resolve structures or support QC but contributes zero compound-level endpoint measurements"
    inhibition_path = octant / "inhibition.tsv"
    inhibition_disposition = writer.disposition("openadmet", "octant_cyp3a4_inhibition")
    with inhibition_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        row_count = 0
        for row_number, row in enumerate(reader, start=2):
            row_count += 1
            inhibition_disposition.candidate_record_rows += 1
            parsed = parse_numeric_relation(row.get("CYP3A4_pIC50"))
            if parsed is None:
                continue
            relation, value, lower, upper, censored = parsed
            flags = []
            qc_fields = ("drc_qc_status", "drc_qc_flag", "qc_flag_primary", "plate_qc_status")
            if any((_clean(row.get(field)) or "").casefold() != "pass" for field in qc_fields):
                flags.append("source_qc_fail")
            writer.emit(
                source_family="openadmet",
                source_dataset="octant_cyp3a4_inhibition",
                source_version="96dc1cceaa545a22041d1e16a9c2524a658403f8",
                source_file=str(inhibition_path.relative_to(repo_root)),
                source_row_number=row_number,
                source_record_id=_clean(row.get("ocnt_batch")) or f"row:{row_number}",
                source_split=None,
                source_rights_status="CC_BY_4_0_repository_tag_catalog_conflict_noted",
                source_lineage="direct_octant_compound_level_summary",
                raw_smiles=row.get("standardized_smiles"),
                task_id="openadmet__octant__cyp3a4_pic50_preincubated",
                endpoint_family="metabolism",
                endpoint_name="cyp_inhibition_pic50",
                endpoint_variant="cyp3a4_30min_active_enzyme_preincubation",
                task_type="regression",
                source_native_endpoint="CYP3A4_pIC50",
                original_value_text=row.get("CYP3A4_pIC50"),
                original_numeric_value=value,
                original_unit="-log10(mol/L)",
                relation=relation,
                normalized_value=value,
                normalized_unit="-log10(mol/L)",
                lower_bound=lower,
                upper_bound=upper,
                is_censored=censored,
                uncertainty_lower=_finite(row.get("CYP3A4_pIC50_ci_lower")),
                uncertainty_upper=_finite(row.get("CYP3A4_pIC50_ci_upper")),
                species="Homo sapiens",
                matrix="enzyme_assay",
                route=None,
                dose_value=None,
                dose_unit=None,
                time_value=0.5,
                time_unit="h",
                assay_id="Octant_CYP3A4_inhibition",
                document_id=None,
                context_complete=True,
                quality_flags=flags,
                context={
                    "activity_status": _clean(row.get("activity_status")),
                    "rollover_status": _clean(row.get("rollover_status")),
                    "saturation_status": _clean(row.get("saturation_status")),
                    "direction": _clean(row.get("direction")),
                },
                provenance={"ocnt_batch": _clean(row.get("ocnt_batch"))},
            )
    inhibition_disposition.physical_file_rows = row_count
    inhibition_disposition.excluded_candidate_count = (
        row_count - inhibition_disposition.emitted_measurement_count
    )
    inhibition_disposition.reasons["missing_compound_level_pic50"] += (
        inhibition_disposition.excluded_candidate_count
    )
    inhibition_disposition.double_count_policy = (
        "compound-level fitted pIC50 only; inhibition well rows are mapping/QC inputs, not measurements"
    )

    reactivity_disposition = writer.disposition("openadmet", "octant_cyp_reactivity")
    with reactivity_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        row_count = 0
        for row_number, row in enumerate(reader, start=2):
            row_count += 1
            reactivity_disposition.candidate_record_rows += 1
            parsed = parse_numeric_relation(row.get("pct_remaining"))
            if parsed is None:
                continue
            relation, value, lower, upper, censored = parsed
            batch = _clean(row.get("ocnt_batch"))
            enzyme = _clean(row.get("enzyme")) or "unspecified_enzyme"
            smiles = structure_map.get(batch or "")
            writer.emit(
                source_family="openadmet",
                source_dataset="octant_cyp_reactivity",
                source_version="96dc1cceaa545a22041d1e16a9c2524a658403f8",
                source_file=str(reactivity_path.relative_to(repo_root)),
                source_row_number=row_number,
                source_record_id=f"{batch or row_number}:{enzyme}",
                source_split=None,
                source_rights_status="CC_BY_4_0_repository_tag_catalog_conflict_noted",
                source_lineage="direct_octant_compound_enzyme_summary",
                raw_smiles=smiles,
                task_id=f"openadmet__octant__{_slug(enzyme)}__percent_remaining",
                endpoint_family="metabolism",
                endpoint_name="cyp_substrate_depletion",
                endpoint_variant=_slug(enzyme),
                task_type="regression",
                source_native_endpoint="pct_remaining",
                original_value_text=row.get("pct_remaining"),
                original_numeric_value=value,
                original_unit="%_remaining",
                relation=relation,
                normalized_value=value,
                normalized_unit="%_remaining",
                lower_bound=lower,
                upper_bound=upper,
                is_censored=censored,
                species="Homo sapiens",
                matrix="enzyme_assay",
                route=None,
                dose_value=None,
                dose_unit=None,
                time_value=None,
                time_unit=None,
                assay_id=f"Octant_{enzyme}_reactivity",
                document_id=None,
                context_complete=True,
                quality_flags=[] if smiles else ["missing_local_structure_mapping"],
                context={"log10fc": _finite(row.get("log10fc")), "log2fc": _finite(row.get("log2fc"))},
                provenance={
                    "ocnt_batch": batch,
                    "structure_mapping_files": [str(path.relative_to(repo_root)) for path in mapping_paths],
                },
            )
    reactivity_disposition.physical_file_rows = row_count
    reactivity_disposition.excluded_candidate_count = (
        row_count - reactivity_disposition.emitted_measurement_count
    )
    reactivity_disposition.double_count_policy = (
        "one compound-enzyme summary pct_remaining; correlated log fold changes retained only as context"
    )


def _load_openadmet(writer: SurfaceWriter, repo_root: Path, bindings: list[dict[str, Any]]) -> None:
    root = repo_root / "research/data/platform/raw/external_public/pk_expansion/avicenna/openadmet"
    manifest_path = root / "manifest.json"
    manifest = json.loads(_checked_file(manifest_path, role="OpenADMET manifest").read_text())
    bindings.append(_bind_file(repo_root, manifest_path, "input_manifest", None))
    declared = {item["path"]: item["sha256"] for item in manifest["artifacts"]}
    permeability_root = root / "permeability_logd_ppb/anvil_training/data"
    _load_paired_openadmet(
        writer,
        repo_root,
        bindings,
        dataset_name="chemeleon_permeability_logd_ppb",
        x_path=permeability_root / "X_train.csv",
        y_path=permeability_root / "y_train.csv",
        endpoint_specs=[
            PairedEndpointSpec("logD", "lipophilicity", "logd", "chemeleon", "unitless"),
            PairedEndpointSpec(
                "caco2_atob_LogPapp",
                "permeability",
                "papp",
                "caco2_atob_log10",
                "log10(cm/s)",
                "Homo sapiens",
                "caco2",
            ),
            PairedEndpointSpec(
                "caco2_btoa_LogPapp",
                "permeability",
                "papp",
                "caco2_btoa_log10",
                "log10(cm/s)",
                "Homo sapiens",
                "caco2",
            ),
            PairedEndpointSpec(
                "mppb_LogUnbound",
                "protein_binding",
                "fraction_unbound",
                "mouse_plasma_log10_percent",
                "log10(%_unbound)",
                "Mus musculus",
                "plasma",
            ),
            PairedEndpointSpec(
                "hppb_LogUnbound",
                "protein_binding",
                "fraction_unbound",
                "human_plasma_log10_percent",
                "log10(%_unbound)",
                "Homo sapiens",
                "plasma",
            ),
        ],
        declared_hashes=declared,
        openadmet_root=root,
        lineage="curated_chembl_derived_training_table_overlap_risk",
        rights="Apache_2_0_repository_ChEMBL_attribution_sharealike_review_required",
    )
    cyp_root = root / "cyp_inhibition_chemeleon/anvil_training/data"
    _load_paired_openadmet(
        writer,
        repo_root,
        bindings,
        dataset_name="chemeleon_cyp_inhibition",
        x_path=cyp_root / "X_train.csv",
        y_path=cyp_root / "y_train.csv",
        endpoint_specs=[
            PairedEndpointSpec(
                "OPENADMET_LOGAC50_cyp3a4",
                "metabolism",
                "cyp_inhibition_pic50",
                "cyp3a4_chemeleon",
                "-log10(mol/L)",
                "Homo sapiens",
                "enzyme_assay",
            ),
            PairedEndpointSpec(
                "OPENADMET_LOGAC50_cyp2d6",
                "metabolism",
                "cyp_inhibition_pic50",
                "cyp2d6_chemeleon",
                "-log10(mol/L)",
                "Homo sapiens",
                "enzyme_assay",
            ),
            PairedEndpointSpec(
                "OPENADMET_LOGAC50_cyp2c9",
                "metabolism",
                "cyp_inhibition_pic50",
                "cyp2c9_chemeleon",
                "-log10(mol/L)",
                "Homo sapiens",
                "enzyme_assay",
            ),
            PairedEndpointSpec(
                "OPENADMET_LOGAC50_cyp1a2",
                "metabolism",
                "cyp_inhibition_pic50",
                "cyp1a2_chemeleon",
                "-log10(mol/L)",
                "Homo sapiens",
                "enzyme_assay",
            ),
        ],
        declared_hashes=declared,
        openadmet_root=root,
        lineage="curated_chembl_derived_training_table_overlap_risk",
        rights="Apache_2_0_repository_ChEMBL_attribution_sharealike_review_required",
    )
    _load_openadmet_expansion(writer, repo_root, bindings, root, declared)
    _load_openadmet_octant(writer, repo_root, bindings, root, declared)
    for relative in [
        "permeability_logd_ppb/README.md",
        "cyp_inhibition_chemeleon/README.md",
        "expansionrx_full/README.md",
        "octant_cyp/README.md",
    ]:
        path = root / relative
        bindings.append(_bind_file(repo_root, path, "endpoint_semantics_contract", declared.get(relative)))
    for relative in ["expansionrx_full/expansion_data_train.csv", "expansionrx_full/expansion_data_test.csv"]:
        path = root / relative
        if path.is_file():
            bindings.append(_bind_file(repo_root, path, "excluded_overlap_split", declared.get(relative)))
            rows = sum(1 for _ in path.open("r", encoding="utf-8")) - 1
            disposition = writer.disposition("openadmet", f"excluded_overlap_{path.stem}")
            disposition.physical_file_rows = rows
            disposition.candidate_record_rows = rows
            disposition.excluded_candidate_count = rows
            disposition.disposition = "excluded_overlap"
            disposition.reasons["duplicates_full_raw_release"] += rows
            disposition.double_count_policy = (
                "excluded because the full raw release is the sole ExpansionRx source"
            )


def _ncats_structure_map(path: Path) -> dict[str, str]:
    candidates: dict[str, set[str]] = defaultdict(set)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not {"PUBCHEM_CID", "SMILES_ISO"}.issubset(reader.fieldnames):
            raise PKADMESurfaceError(f"Missing NCATS CID/SMILES mapping fields: {path}")
        for row in reader:
            cid = _clean(row.get("PUBCHEM_CID"))
            smiles = _clean(row.get("SMILES_ISO"))
            if cid and smiles:
                candidates[cid].add(smiles)
    conflicts = {key: values for key, values in candidates.items() if len(values) != 1}
    if conflicts:
        raise PKADMESurfaceError(f"Conflicting NCATS CID structures for {len(conflicts)} CIDs")
    return {key: next(iter(values)) for key, values in candidates.items()}


def _load_ncats(writer: SurfaceWriter, repo_root: Path, bindings: list[dict[str, Any]]) -> None:
    root = repo_root / "research/data/platform/raw/external_public/pk_expansion/avicenna/ncats_adme"
    manifest_path = root / "manifest.json"
    manifest = json.loads(_checked_file(manifest_path, role="NCATS manifest").read_text())
    bindings.append(_bind_file(repo_root, manifest_path, "input_manifest", None))
    declared = {item["path"]: item["sha256"] for item in manifest["artifacts"]}
    mapping_path = root / "data/AID_1508612_datatable_all.csv"
    structure_map = _ncats_structure_map(mapping_path)
    specs: dict[int, dict[str, Any]] = {
        1508591: {
            "column": "Half-life (minutes)",
            "endpoint_family": "stability",
            "endpoint_name": "half_life",
            "endpoint_variant": "rat_liver_microsome",
            "unit": "min",
            "species": "Rattus norvegicus",
            "matrix": "liver_microsome",
        },
        1508603: {
            "column": "Half-life",
            "endpoint_family": "stability",
            "endpoint_name": "half_life",
            "endpoint_variant": "human_liver_cytosol_source_scale",
            "unit": "source_native_scale",
            "species": "Homo sapiens",
            "matrix": "liver_cytosol",
        },
        1508612: {
            "column": "Permeability",
            "endpoint_family": "permeability",
            "endpoint_name": "pampa_permeability",
            "endpoint_variant": "ncats_source_scale",
            "unit": "source_native_scale",
            "species": None,
            "matrix": "artificial_membrane",
        },
        1645848: {
            "column": "Kinetic Aqueous Solubility (ug/mL)",
            "endpoint_family": "solubility",
            "endpoint_name": "kinetic_aqueous_solubility",
            "endpoint_variant": "ncats",
            "unit": "ug/mL",
            "species": None,
            "matrix": "aqueous",
        },
        1645871: {
            "column": "Permeability",
            "endpoint_family": "permeability",
            "endpoint_name": "pampa_permeability",
            "endpoint_variant": "ncats_ph5_source_scale",
            "unit": "source_native_scale",
            "species": None,
            "matrix": "artificial_membrane_ph5",
        },
    }
    for aid, spec in specs.items():
        data_path = root / f"data/AID_{aid}_datatable_all.csv"
        description_path = root / f"assays/AID_{aid}_description.json"
        for path, role in (
            (data_path, "ncats_primary_assay_table"),
            (description_path, "ncats_assay_protocol"),
        ):
            relative = str(path.relative_to(root))
            bindings.append(_bind_file(repo_root, path, role, declared.get(relative)))
        disposition = writer.disposition("ncats_adme", f"AID_{aid}")
        with data_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or spec["column"] not in reader.fieldnames:
                raise PKADMESurfaceError(f"Missing NCATS endpoint column in {data_path}")
            row_count = 0
            for row_number, row in enumerate(reader, start=2):
                row_count += 1
                disposition.candidate_record_rows += 1
                parsed = parse_numeric_relation(row.get(spec["column"]))
                if parsed is None:
                    disposition.excluded_candidate_count += 1
                    disposition.reasons["missing_or_unparseable_primary_endpoint"] += 1
                    continue
                relation, value, lower, upper, censored = parsed
                cid = _clean(row.get("PUBCHEM_CID"))
                smiles = _clean(row.get("SMILES_ISO")) or structure_map.get(cid or "")
                flags: list[str] = []
                if spec["unit"] == "source_native_scale":
                    flags.append("source_native_unit_missing")
                if smiles is None:
                    flags.append("missing_local_structure_mapping")
                writer.emit(
                    source_family="ncats_adme",
                    source_dataset=f"AID_{aid}",
                    source_version=f"PubChem_AID_{aid}_frozen_2026_08_06",
                    source_file=str(data_path.relative_to(repo_root)),
                    source_row_number=row_number,
                    source_record_id=f"SID:{_clean(row.get('PUBCHEM_SID')) or row_number}",
                    source_split=None,
                    source_rights_status="public_NCATS_PubChem_source_rights_review_before_redistribution",
                    source_lineage="direct_ncats_pubchem_primary_assay",
                    raw_smiles=smiles,
                    task_id=f"ncats__aid_{aid}__{_slug(spec['column'])}",
                    endpoint_family=spec["endpoint_family"],
                    endpoint_name=spec["endpoint_name"],
                    endpoint_variant=spec["endpoint_variant"],
                    task_type="regression",
                    source_native_endpoint=spec["column"],
                    original_value_text=row.get(spec["column"]),
                    original_numeric_value=value,
                    original_unit=None if spec["unit"] == "source_native_scale" else spec["unit"],
                    relation=relation,
                    normalized_value=value,
                    normalized_unit=spec["unit"],
                    lower_bound=lower,
                    upper_bound=upper,
                    is_censored=censored,
                    species=spec["species"],
                    matrix=spec["matrix"],
                    route=None,
                    dose_value=None,
                    dose_unit=None,
                    time_value=None,
                    time_unit=None,
                    assay_id=f"PubChem_AID_{aid}",
                    document_id=None,
                    context_complete=True,
                    quality_flags=flags,
                    context={
                        "activity_outcome": _clean(row.get("PUBCHEM_ACTIVITY_OUTCOME")),
                        "phenotype": next(
                            (_clean(value_) for key, value_ in row.items() if key.startswith("Phenotype")),
                            None,
                        ),
                        "compound_qc": _clean(row.get("Compound QC")),
                    },
                    provenance={
                        "pubchem_sid": _clean(row.get("PUBCHEM_SID")),
                        "pubchem_cid": cid,
                        "structure_mapping_aid": 1508612 if not _clean(row.get("SMILES_ISO")) else aid,
                        "assay_description_file": str(description_path.relative_to(repo_root)),
                    },
                )
        disposition.physical_file_rows = row_count
        disposition.double_count_policy = "primary numeric endpoint only; activity outcome and phenotype remain context, not additional labels"

    for aid in (1645840, 1645841, 1645842):
        data_path = root / f"data/AID_{aid}_datatable_all.csv"
        description_path = root / f"assays/AID_{aid}_description.json"
        for path, role in (
            (data_path, "excluded_qhts_replicate_table"),
            (description_path, "ncats_assay_protocol"),
        ):
            relative = str(path.relative_to(root))
            bindings.append(_bind_file(repo_root, path, role, declared.get(relative)))
        with data_path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = sum(1 for _ in csv.DictReader(handle))
        disposition = writer.disposition("ncats_adme", f"AID_{aid}_excluded_qhts")
        disposition.physical_file_rows = rows
        disposition.candidate_record_rows = rows
        disposition.excluded_candidate_count = rows
        disposition.disposition = "excluded_replication_and_structure_contract_unresolved"
        disposition.reasons["replicate_aggregation_and_broad_local_structure_mapping_unresolved"] += rows
        disposition.double_count_policy = "replicate columns are not multiplied into measurements; assay excluded pending an explicit aggregation contract"


def _add_nonmeasurement_blockers(
    writer: SurfaceWriter, repo_root: Path, bindings: list[dict[str, Any]]
) -> None:
    dailymed_root = (
        repo_root
        / "research/data/platform/raw/external_public/pk_expansion/avicenna/dailymed_pk_candidate_evidence"
    )
    dailymed_manifest = dailymed_root / "manifest.json"
    dailymed_summary = dailymed_root / "scan_summary.json"
    bindings.append(_bind_file(repo_root, dailymed_manifest, "zero_label_candidate_manifest", None))
    summary = json.loads(_checked_file(dailymed_summary, role="DailyMed scan summary").read_text())
    declared = {
        item["path"]: item["sha256"] for item in json.loads(dailymed_manifest.read_text())["artifacts"]
    }
    bindings.append(
        _bind_file(
            repo_root, dailymed_summary, "zero_label_candidate_summary", declared.get("scan_summary.json")
        )
    )
    candidate_sections = int(summary["counts"]["candidate_sections"])
    disposition = writer.disposition("dailymed", "pk_candidate_sections")
    disposition.physical_file_rows = candidate_sections
    disposition.candidate_record_rows = candidate_sections
    disposition.excluded_candidate_count = candidate_sections
    disposition.disposition = "zero_measurements_candidate_text_only"
    disposition.reasons["numeric_values_not_normalized_or_validated"] += candidate_sections
    disposition.double_count_policy = "document sections and tables are never measurements"

    pkdb_manifest = (
        repo_root / "research/data/platform/raw/external_public/pkdb_public_2026_08_05/manifest.json"
    )
    pkdb = json.loads(_checked_file(pkdb_manifest, role="PK-DB manifest").read_text())
    bindings.append(_bind_file(repo_root, pkdb_manifest, "zero_output_blocker_manifest", None))
    disposition = writer.disposition("pkdb", "anonymous_api_outputs")
    disposition.physical_file_rows = int(pkdb.get("canonical_observation_count", 0))
    disposition.candidate_record_rows = 0
    disposition.disposition = "zero_rows_authentication_and_rights_blocked"
    disposition.reasons["anonymous_output_probe_returned_zero"] += 1
    disposition.double_count_policy = "official statistics are not acquired output measurements"

    drugsfda_manifest = (
        repo_root
        / "research/data/platform/raw/external_public/drugs_at_fda_bulk/drugs_at_fda_bulk_manifest.json"
    )
    drugsfda = json.loads(_checked_file(drugsfda_manifest, role="Drugs@FDA manifest").read_text())
    bindings.append(_bind_file(repo_root, drugsfda_manifest, "zero_label_regulatory_manifest", None))
    physical_rows = int(drugsfda.get("archive_member_table", {}).get("total_data_rows", 0))
    disposition = writer.disposition("drugs_at_fda", "regulatory_relational_tables")
    disposition.physical_file_rows = physical_rows
    disposition.candidate_record_rows = physical_rows
    disposition.excluded_candidate_count = physical_rows
    disposition.disposition = "zero_measurements_regulatory_metadata_only"
    disposition.reasons["application_product_action_rows_are_not_pk_measurements"] += physical_rows
    disposition.double_count_policy = "regulatory rows are not molecular PK labels"


def _add_governance_bindings(repo_root: Path, bindings: list[dict[str, Any]]) -> None:
    for relative, role in [
        (
            "pipeline/src/menin_discovery/platform_pk_adme_trainable_surfaces.py",
            "builder_implementation",
        ),
        (
            "research/reports/platform/pk_expansion/avicenna/source_ledger.json",
            "source_rights_and_admission_ledger",
        ),
        ("research/reports/platform/pk_expansion/avicenna/tdc_adme_inventory.json", "source_inventory"),
        ("research/reports/platform/pk_expansion/avicenna/ncats_adme_inventory.json", "source_inventory"),
        ("research/reports/platform/pk_expansion/avicenna/openadmet_inventory.json", "source_inventory"),
        (
            "research/data/platform/raw/external_public/pk_expansion/avicenna/tdc_adme/provenance/tdc_metadata.py",
            "tdc_versioned_dataset_metadata",
        ),
    ]:
        bindings.append(_bind_file(repo_root, repo_root / relative, role, None))


def _bind_file(
    repo_root: Path,
    path: Path,
    role: str,
    declared_sha256: str | None,
) -> dict[str, Any]:
    checked = _checked_file(path, role=role)
    actual = _sha256_file(checked)
    if declared_sha256 is not None and actual != declared_sha256:
        raise PKADMESurfaceError(
            f"Input hash mismatch for {checked}: expected {declared_sha256}, observed {actual}"
        )
    binding = {
        "path": str(checked.relative_to(repo_root.resolve())),
        "role": role,
        "bytes": checked.stat().st_size,
        "sha256": actual,
        "declared_sha256": declared_sha256,
        "declared_hash_verified": declared_sha256 is None or declared_sha256 == actual,
    }
    if checked.suffix.casefold() == ".parquet":
        parquet_file = pq.ParquetFile(checked)
        binding["row_count"] = parquet_file.metadata.num_rows
        binding["arrow_schema_sha256"] = _schema_sha256(parquet_file.schema_arrow)
    return binding


def verify_input_binding(repo_root: Path, binding: Mapping[str, Any]) -> dict[str, bool]:
    """Recompute every physical invariant carried by one input binding."""

    path = _checked_file(repo_root / str(binding["path"]), role=str(binding["role"]))
    if path.stat().st_size != int(binding["bytes"]):
        raise PKADMESurfaceError(f"Input byte-size mismatch: {path}")
    actual_sha = _sha256_file(path)
    if actual_sha != binding["sha256"]:
        raise PKADMESurfaceError(f"Input SHA-256 mismatch: {path}")
    declared = binding.get("declared_sha256")
    if declared is not None and declared != actual_sha:
        raise PKADMESurfaceError(f"Input declared SHA-256 mismatch: {path}")
    checks = {"bytes_verified": True, "sha256_verified": True, "declared_hash_verified": True}
    if path.suffix.casefold() == ".parquet":
        if "row_count" not in binding or "arrow_schema_sha256" not in binding:
            raise PKADMESurfaceError(f"Parquet input lacks row/schema binding: {path}")
        parquet_file = pq.ParquetFile(path)
        if parquet_file.metadata.num_rows != int(binding["row_count"]):
            raise PKADMESurfaceError(f"Parquet input row-count mismatch: {path}")
        if _schema_sha256(parquet_file.schema_arrow) != binding["arrow_schema_sha256"]:
            raise PKADMESurfaceError(f"Parquet input Arrow schema mismatch: {path}")
        checks["row_count_verified"] = True
        checks["arrow_schema_verified"] = True
    return checks


def _manifest_payload_sha256(manifest: Mapping[str, Any]) -> str:
    payload = dict(manifest)
    payload.pop("manifest_payload_sha256", None)
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def self_hash_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Add a non-recursive hash over the canonical manifest payload."""

    payload = dict(manifest)
    payload.pop("manifest_payload_sha256", None)
    payload["manifest_payload_sha256"] = _manifest_payload_sha256(payload)
    return payload


def verify_manifest_self_hash(manifest: Mapping[str, Any]) -> None:
    expected = _clean(manifest.get("manifest_payload_sha256"))
    if expected is None or expected != _manifest_payload_sha256(manifest):
        raise PKADMESurfaceError("Artifact manifest canonical-payload self-hash mismatch")


_EXPECTED_OUTPUT_MEMBERS = frozenset(
    {
        "measurement_ledger.parquet",
        "modeling_surface.parquet",
        "task_registry.parquet",
        "molecule_registry.parquet",
        "source_disposition.parquet",
        "input_binding_manifest.json",
        "modeling_contract.json",
        "release_summary.json",
        "artifact_manifest.json",
        "validation.json",
    }
)


def validate_closed_release_membership(output_dir: Path, report_path: Path) -> None:
    if output_dir.is_symlink() or not output_dir.is_dir():
        raise PKADMESurfaceError(f"Release output is not a physical directory: {output_dir}")
    members = {path.name for path in output_dir.iterdir()}
    if members != _EXPECTED_OUTPUT_MEMBERS:
        missing = sorted(_EXPECTED_OUTPUT_MEMBERS - members)
        extra = sorted(members - _EXPECTED_OUTPUT_MEMBERS)
        raise PKADMESurfaceError(f"Release membership mismatch; missing={missing}, extra={extra}")
    if any(path.is_symlink() or not path.is_file() for path in output_dir.iterdir()):
        raise PKADMESurfaceError("Release output contains a symlink or non-file member")
    if report_path.is_symlink() or not report_path.is_file():
        raise PKADMESurfaceError(f"Release report is not a physical file: {report_path}")
    report_members = {path.name for path in report_path.parent.iterdir()}
    if report_members != {report_path.name}:
        raise PKADMESurfaceError(
            f"Report directory membership mismatch: expected only {report_path.name}, observed {sorted(report_members)}"
        )


def _task_rows(writer: SurfaceWriter) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task_id, task in sorted(writer.tasks.items()):
        task_trainable = (
            len(task.connectivity_groups) >= MIN_TASK_CONNECTIVITY_GROUPS
            and (task.exact_modeling_count + task.censored_only_modeling_count) > 0
        )
        if task.endpoint_family in {"exposure", "bioavailability", "distribution"}:
            conditioning = {
                "required_for_absolute_prediction": ["species", "route", "dose"],
                "missing_context_policy": "filter or model explicitly; do not treat null as a reference regimen",
            }
        elif task.endpoint_family in {"clearance", "half_life", "stability", "protein_binding"}:
            conditioning = {
                "required_or_filterable": ["species", "matrix"],
                "missing_context_policy": "retain source-native task but stratify or filter before modeling",
            }
        else:
            conditioning = {
                "required_or_filterable": ["matrix", "species"],
                "missing_context_policy": "use the source-native endpoint variant and available context fields",
            }
        rows.append(
            {
                "task_id": task_id,
                "endpoint_family": task.endpoint_family,
                "endpoint_name": task.endpoint_name,
                "endpoint_variant": task.endpoint_variant,
                "task_type": task.task_type,
                "normalized_unit": task.normalized_unit,
                "source_families_json": _canonical_json(sorted(task.source_families)),
                "source_datasets_json": _canonical_json(sorted(task.source_datasets)),
                "observation_count": task.observation_count,
                "valid_structure_count": task.valid_structure_count,
                "unique_molecule_count": len(task.molecules),
                "unique_connectivity_group_count": len(task.connectivity_groups),
                "exact_modeling_count": task.exact_modeling_count,
                "censored_only_modeling_count": task.censored_only_modeling_count,
                "context_complete_count": task.context_complete_count,
                "cross_source_union_count": task.cross_source_union_count,
                "task_trainable": task_trainable,
                "minimum_connectivity_groups": MIN_TASK_CONNECTIVITY_GROUPS,
                "species_json": _canonical_json(sorted(task.species)),
                "matrices_json": _canonical_json(sorted(task.matrices)),
                "routes_json": _canonical_json(sorted(task.routes)),
                "conditioning_contract_json": _canonical_json(conditioning),
                "rights_statuses_json": _canonical_json(sorted(task.rights_statuses)),
                "lineage_overlap_risk": task.lineage_overlap_risk,
            }
        )
    return rows


def _molecule_rows(writer: SurfaceWriter) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for molecule_id, item in sorted(writer.molecules.items()):
        rows.append(
            {
                "molecule_id": molecule_id,
                "standardized_smiles": item["standardized_smiles"],
                "standard_inchi_key": item["standard_inchi_key"],
                "connectivity_key": item["connectivity_key"],
                "leakage_group_id": item["leakage_group_id"],
                "scaffold_smiles": item["scaffold_smiles"],
                "scaffold_group_id": item["scaffold_group_id"],
                "source_families_json": _canonical_json(sorted(item["source_families"])),
                "observation_count": item["observation_count"],
            }
        )
    return rows


def _disposition_rows(writer: SurfaceWriter) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (source_family, source_dataset), item in sorted(writer.dispositions.items()):
        rows.append(
            {
                "source_family": source_family,
                "source_dataset": source_dataset,
                "physical_file_rows": item.physical_file_rows,
                "candidate_record_rows": item.candidate_record_rows,
                "emitted_measurement_count": item.emitted_measurement_count,
                "exact_modeling_count": item.exact_modeling_count,
                "censored_only_modeling_count": item.censored_only_modeling_count,
                "missing_or_invalid_structure_count": item.missing_or_invalid_structure_count,
                "excluded_candidate_count": item.excluded_candidate_count,
                "disposition": item.disposition,
                "exclusion_reasons_json": _canonical_json(dict(sorted(item.reasons.items()))),
                "double_count_policy": item.double_count_policy,
            }
        )
    return rows


def _write_modeling_surface(
    measurement_path: Path,
    output_path: Path,
    trainable_tasks: set[str],
) -> int:
    writer = pq.ParquetWriter(output_path, _OBSERVATION_SCHEMA, compression="zstd")
    count = 0
    parquet_file = pq.ParquetFile(measurement_path)
    for batch in parquet_file.iter_batches(batch_size=50_000):
        task_mask = pc.is_in(batch.column("task_id"), value_set=pa.array(sorted(trainable_tasks)))
        eligibility_mask = pc.or_(
            batch.column("eligible_exact_modeling"), batch.column("eligible_censored_modeling")
        )
        selected = batch.filter(pc.and_(task_mask, eligibility_mask))
        if selected.num_rows:
            writer.write_batch(selected)
            count += selected.num_rows
    writer.close()
    return count


def _contract_payload() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "release_version": RELEASE_VERSION,
        "standardization_version": STANDARDIZATION_VERSION,
        "scientific_boundary": {
            "canonical_admission": False,
            "clinical_outcome_label": False,
            "source_native_human_clinical_pk_measurements_allowed_and_flagged": True,
            "clinical_context_flag_semantics": "high-precision provenance candidate; deterministic but not exhaustive clinical adjudication",
            "qt_label": False,
            "herg_label": False,
            "safety_or_efficacy_label": False,
            "allowed_use": "source-native PK/ADME endpoint modeling preparation only",
        },
        "measurement_definition": "one physically present, parseable source record and explicit endpoint field; paired X/Y rows are joined once; candidate documents and well rows are not measurements",
        "task_separation": "task_id is mandatory; values from different task_id values must never be pooled into one target without a new reviewed harmonization contract",
        "identity_contract": {
            "molecule_id": "full standardized InChIKey plus canonical-isomeric-SMILES representation hash; this prevents one molecule_id from mapping to multiple scaffolds when Standard InChI collapses reported tautomers",
            "leakage_group_id": "InChIKey connectivity-block hash; all stereochemistry/protonation variants sharing connectivity remain in one split group",
            "scaffold_group_id": "Bemis-Murcko scaffold hash derived from the exact molecule representation identified by molecule_id, with an acyclic connectivity-specific fallback",
            "split_rule": "no leakage_group_id may appear in more than one data split, across all sources and tasks",
        },
        "numeric_contract": {
            "relation_equal": "exact value; lower_bound and upper_bound are null",
            "relation_less": "left-censored upper boundary represented as upper_bound",
            "relation_greater": "right-censored lower boundary represented as lower_bound",
            "interval": "both lower_bound and upper_bound present only when physically supplied",
            "approximate_or_unknown": "retained in the measurement ledger but excluded from default modeling",
            "unit_rule": "only explicit compatible conversions in code are normalized; missing source-native units remain source_native_scale",
            "default_regression_filter": "eligible_exact_modeling == true",
            "censored_objective_filter": "eligible_censored_modeling == true; relation and bounds are mandatory model inputs",
        },
        "rights_contract": "source_rights_status is mandatory; local model-readiness does not grant redistribution or commercial rights",
        "minimum_task_connectivity_groups": MIN_TASK_CONNECTIVITY_GROUPS,
        "observation_schema_sha256": _schema_sha256(_OBSERVATION_SCHEMA),
        "task_schema_sha256": _schema_sha256(_TASK_SCHEMA),
        "molecule_schema_sha256": _schema_sha256(_MOLECULE_SCHEMA),
        "disposition_schema_sha256": _schema_sha256(_DISPOSITION_SCHEMA),
    }


def _write_report(
    path: Path,
    summary: Mapping[str, Any],
    task_rows: Sequence[Mapping[str, Any]],
) -> None:
    source_lines = []
    for source, values in sorted(summary["by_source_family"].items()):
        source_lines.append(
            f"- {source}: {values['observations']:,} standardized endpoint observations; "
            f"{values['exact_modeling']:,} exact and {values['censored_only_modeling']:,} censored-only modeling rows."
        )
    largest = sorted(task_rows, key=lambda row: int(row["exact_modeling_count"]), reverse=True)[:12]
    task_lines = [
        f"- {row['task_id']}: {int(row['exact_modeling_count']):,} exact rows, "
        f"{int(row['censored_only_modeling_count']):,} censored-only rows, and "
        f"{int(row['unique_connectivity_group_count']):,} leakage groups."
        for row in largest
    ]
    text = "\n".join(
        [
            "# PK/ADME trainable surfaces v1.0",
            "",
            "## Outcome",
            "",
            f"The release contains {summary['measurement_observation_count']:,} source-bound endpoint observations and "
            f"{summary['modeling_surface_count']:,} rows in endpoint-specific modeling tasks. "
            f"It covers {summary['unique_molecule_count']:,} standardized molecules, "
            f"{summary['unique_leakage_group_count']:,} connectivity leakage groups, and "
            f"{summary['trainable_task_count']:,} tasks meeting the minimum size contract.",
            "",
            "This is a noncanonical modeling-preparation release. It creates no clinical-outcome, QT, hERG, safety, efficacy, or approval labels, and it does not promote any row into the canonical PK store. Source-native human clinical PK measurements may remain as PK targets and are explicitly flagged; QT or ECG wording is context only and is never a target.",
            "",
            "## Honest source counts",
            "",
            *source_lines,
            "",
            "Counts are endpoint observations, not document hits or physical file rows. A paired structure/label row is counted once per non-null endpoint. ExpansionRx raw/train/test overlap is represented only by the full raw file. Octant and NCATS replicate/well fields are not multiplied into endpoint observations.",
            "",
            "## Largest endpoint-specific tasks",
            "",
            *task_lines,
            "",
            "Task identifiers are scientific boundaries. Training code must filter one task at a time or use an explicitly reviewed multitask objective; it must not concatenate normalized_value across tasks as a common target.",
            "",
            "## Leakage controls",
            "",
            "- The split group hashes the standardized InChIKey connectivity block, conservatively co-locating stereoisomers and protonation variants.",
            "- A scaffold group derived from the exact standardized representation is supplied for holdout evaluation; the connectivity leakage group remains mandatory so tautomeric or protonation representations cannot cross splits.",
            "- ChEMBL-derived OpenADMET CheMeleon tables are flagged as lineage-overlap risks and are not eligible for the default cross-source union.",
            "- Duplicate-group identifiers expose same-task, same-structure, same-value repeats without asserting that separate source records are the same experiment.",
            "",
            "## Context and censoring",
            "",
            "Species, matrix, route, dose, time, assay, and document context are populated only from physically present source fields or narrowly defined text parsers. Missing context remains null. Absolute exposure tasks such as AUC and Cmax carry a context-completeness flag and must not treat missing dose or route as a reference regimen.",
            f"The release flags {summary['clinical_pk_context_observation_count']:,} high-precision candidate observations with explicit human clinical or patient PK context and {summary['qt_ecg_context_observation_count']:,} observations whose source context mentions QT or ECG. The clinical-context flag is deterministic but intentionally non-exhaustive pending human adjudication. These flags stratify provenance; they are not clinical-risk, QT, or hERG labels.",
            "",
            "Exact, less-than, less-than-or-equal, greater-than, and greater-than-or-equal relations remain distinct. Censored observations expose only the applicable bound. Approximate or unknown relations remain in the measurement ledger but are excluded from the default modeling surface.",
            "",
            "## Blockers and limits",
            "",
            "- TDC benchmark files generally lack unit, route, dose, and matrix columns; source-native-scale tasks remain separate and the underlying dataset rights require source-by-source review.",
            "- NCATS CYP qHTS replicate tables are excluded because broad local structure resolution and a reviewed replicate-aggregation contract are absent.",
            "- DailyMed sections and tables are machine-detected candidates, not normalized measurements, so they contribute zero labels.",
            "- Drugs@FDA application, product, and action rows are regulatory metadata, not molecule-level PK measurements, so they contribute zero labels.",
            "- PK-DB reports a large official corpus, but the local anonymous output acquisition contains zero records and its rights/access boundary remains unresolved.",
            "- Rights fields are carried per observation; modeling readiness does not resolve redistribution obligations.",
            "",
            "## Validation",
            "",
            f"All {summary['bound_input_count']:,} consumed input files and {summary['artifact_count']:,} release artifacts are SHA-256 bound. Parquet schemas, row counts, unique observation IDs, task contracts, canonical-admission=false, and the absence of hERG/QT label fields are validated in validation.json.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _artifact_entry(repo_root: Path, path: Path, role: str, row_count: int | None = None) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "path": str(path.resolve().relative_to(repo_root.resolve())),
        "role": role,
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }
    if row_count is not None:
        entry["row_count"] = row_count
    if path.suffix == ".parquet":
        entry["arrow_schema_sha256"] = _schema_sha256(pq.read_schema(path))
    return entry


def validate_release(repo_root: Path, output_dir: Path, report_path: Path) -> dict[str, Any]:
    manifest_path = output_dir / "artifact_manifest.json"
    manifest = json.loads(_checked_file(manifest_path, role="artifact manifest").read_text())
    verify_manifest_self_hash(manifest)
    if set(manifest.get("expected_output_members", [])) != _EXPECTED_OUTPUT_MEMBERS:
        raise PKADMESurfaceError("Artifact manifest expected-output contract mismatch")
    if int(manifest.get("artifact_count", -1)) != len(manifest.get("artifacts", [])):
        raise PKADMESurfaceError("Artifact manifest count mismatch")
    expected_report = str(report_path.resolve().relative_to(repo_root.resolve()))
    if manifest.get("expected_report_path") != expected_report:
        raise PKADMESurfaceError("Artifact manifest report location does not match validated report")
    validate_closed_release_membership(output_dir, report_path)
    expected_artifact_paths = {
        str((output_dir / name).resolve().relative_to(repo_root.resolve()))
        for name in _EXPECTED_OUTPUT_MEMBERS - {"artifact_manifest.json", "validation.json"}
    }
    expected_artifact_paths.add(expected_report)
    observed_artifact_paths = [str(item["path"]) for item in manifest["artifacts"]]
    if len(observed_artifact_paths) != len(set(observed_artifact_paths)):
        raise PKADMESurfaceError("Artifact manifest contains duplicate paths")
    if set(observed_artifact_paths) != expected_artifact_paths:
        raise PKADMESurfaceError("Artifact manifest membership is not the exact closed release payload")
    artifact_checks = []
    for artifact in manifest["artifacts"]:
        path = repo_root / artifact["path"]
        actual = _sha256_file(_checked_file(path, role=artifact["role"]))
        if actual != artifact["sha256"]:
            raise PKADMESurfaceError(f"Artifact hash mismatch: {path}")
        check = {"path": artifact["path"], "sha256_verified": True}
        if path.suffix == ".parquet":
            rows = pq.ParquetFile(path).metadata.num_rows
            if rows != artifact.get("row_count"):
                raise PKADMESurfaceError(f"Artifact row count mismatch: {path}")
            schema_hash = _schema_sha256(pq.read_schema(path))
            if schema_hash != artifact.get("arrow_schema_sha256"):
                raise PKADMESurfaceError(f"Artifact schema mismatch: {path}")
            check["row_count_verified"] = True
            check["schema_verified"] = True
        artifact_checks.append(check)
    input_binding_path = output_dir / "input_binding_manifest.json"
    input_manifest = json.loads(_checked_file(input_binding_path, role="input binding manifest").read_text())
    inputs = input_manifest.get("inputs")
    if not isinstance(inputs, list) or len(inputs) != input_manifest.get("binding_count"):
        raise PKADMESurfaceError("Input binding manifest count mismatch")
    input_paths = [str(item["path"]) for item in inputs]
    if len(input_paths) != len(set(input_paths)):
        raise PKADMESurfaceError("Input binding manifest contains duplicate paths")
    if sum(item.get("role") == "builder_implementation" for item in inputs) != 1:
        raise PKADMESurfaceError("Input binding manifest must bind exactly one builder implementation")
    input_checks = []
    parquet_input_count = 0
    for binding in inputs:
        checks = verify_input_binding(repo_root, binding)
        if str(binding["path"]).endswith(".parquet"):
            parquet_input_count += 1
        input_checks.append({"path": binding["path"], **checks})
    measurement_path = output_dir / "measurement_ledger.parquet"
    modeling_path = output_dir / "modeling_surface.parquet"
    task_path = output_dir / "task_registry.parquet"
    measurement_dataset = pads.dataset(measurement_path, format="parquet")
    measurement_ids = measurement_dataset.to_table(columns=["observation_id"])["observation_id"]
    unique_ids = pc.count_distinct(measurement_ids).as_py()
    measurement_rows = measurement_dataset.count_rows()
    if unique_ids != measurement_rows:
        raise PKADMESurfaceError("Observation IDs are not unique")
    canonical_true = measurement_dataset.count_rows(filter=pads.field("canonical_admission") == True)  # noqa: E712
    if canonical_true:
        raise PKADMESurfaceError("Canonical admission must remain false")
    forbidden = {"herg_label", "qt_label", "clinical_label", "safety_label"}
    if forbidden.intersection(measurement_dataset.schema.names):
        raise PKADMESurfaceError("Forbidden clinical/QT/hERG label field present")
    task_table = pq.read_table(task_path)
    if pc.count_distinct(task_table["task_id"]).as_py() != task_table.num_rows:
        raise PKADMESurfaceError("Task registry IDs are not unique")
    trainable_tasks = set(task_table.filter(task_table["task_trainable"])["task_id"].to_pylist())
    modeling_table = pq.read_table(
        modeling_path,
        columns=["task_id", "eligible_exact_modeling", "eligible_censored_modeling", "modeling_status"],
    )
    modeling_tasks = set(pc.unique(modeling_table["task_id"]).to_pylist())
    if not modeling_tasks.issubset(trainable_tasks):
        raise PKADMESurfaceError("Modeling surface contains a non-trainable task")
    eligibility = pc.or_(
        modeling_table["eligible_exact_modeling"], modeling_table["eligible_censored_modeling"]
    )
    if (
        pc.any(pc.invert(eligibility)).as_py()
        or pc.any(pc.equal(modeling_table["modeling_status"], "quarantine")).as_py()
    ):
        raise PKADMESurfaceError("Modeling surface contains an ineligible or quarantined row")
    molecule_table = pq.read_table(
        output_dir / "molecule_registry.parquet",
        columns=["molecule_id", "scaffold_group_id", "standardized_smiles"],
    )
    if pc.count_distinct(molecule_table["molecule_id"]).as_py() != molecule_table.num_rows:
        raise PKADMESurfaceError("Molecule registry IDs are not unique")
    molecule_identity = {
        str(row["molecule_id"]): (str(row["scaffold_group_id"]), str(row["standardized_smiles"]))
        for row in molecule_table.to_pylist()
    }
    for batch in pq.ParquetFile(measurement_path).iter_batches(
        batch_size=100_000,
        columns=["molecule_id", "scaffold_group_id", "standardized_smiles"],
    ):
        for row in batch.to_pylist():
            molecule_id = row["molecule_id"]
            if molecule_id is None:
                continue
            expected_identity = molecule_identity.get(str(molecule_id))
            observed_identity = (str(row["scaffold_group_id"]), str(row["standardized_smiles"]))
            if expected_identity != observed_identity:
                raise PKADMESurfaceError(
                    f"Molecule identity maps to inconsistent structure/scaffold: {molecule_id}"
                )
    disposition_table = pq.read_table(
        output_dir / "source_disposition.parquet",
        columns=[
            "source_family",
            "physical_file_rows",
            "candidate_record_rows",
            "emitted_measurement_count",
            "excluded_candidate_count",
            "exclusion_reasons_json",
        ],
    )
    if pc.sum(disposition_table["emitted_measurement_count"]).as_py() != measurement_rows:
        raise PKADMESurfaceError("Source-disposition measurements do not sum to the ledger")
    disposition_rows = disposition_table.to_pylist()
    chembl_rows = [row for row in disposition_rows if row["source_family"] == "chembl_37"]
    if len(chembl_rows) != 1:
        raise PKADMESurfaceError("Expected exactly one ChEMBL source-disposition row")
    chembl = chembl_rows[0]
    chembl_reasons = json.loads(str(chembl["exclusion_reasons_json"]))
    if int(chembl["candidate_record_rows"]) - int(chembl["emitted_measurement_count"]) != int(
        chembl["excluded_candidate_count"]
    ):
        raise PKADMESurfaceError("ChEMBL candidate exclusion arithmetic mismatch")
    if int(chembl["physical_file_rows"]) - int(chembl["candidate_record_rows"]) != int(
        chembl_reasons.get("not_selected_by_case_sensitive_endpoint_unit_contract", -1)
    ):
        raise PKADMESurfaceError("ChEMBL noncandidate disposition arithmetic mismatch")
    for zero_source in {"dailymed", "drugs_at_fda", "pkdb"}:
        if (
            sum(
                int(row["emitted_measurement_count"])
                for row in disposition_rows
                if row["source_family"] == zero_source
            )
            != 0
        ):
            raise PKADMESurfaceError(f"Nonmeasurement source emitted labels: {zero_source}")
    if pc.sum(task_table["observation_count"]).as_py() != measurement_rows:
        raise PKADMESurfaceError("Task observations do not sum to the measurement ledger")
    trainable_table = task_table.filter(task_table["task_trainable"])
    task_modeling_sum = (
        pc.sum(trainable_table["exact_modeling_count"]).as_py()
        + pc.sum(trainable_table["censored_only_modeling_count"]).as_py()
    )
    if task_modeling_sum != modeling_table.num_rows:
        raise PKADMESurfaceError("Task modeling counts do not sum to the modeling surface")
    summary = json.loads((output_dir / "release_summary.json").read_text())
    clinical_context_rows = measurement_dataset.count_rows(
        filter=pads.field("clinical_pk_context") == True  # noqa: E712
    )
    qt_ecg_context_rows = measurement_dataset.count_rows(
        filter=pads.field("qt_ecg_context") == True  # noqa: E712
    )
    if qt_ecg_context_rows:
        qt_context = measurement_dataset.to_table(
            columns=["endpoint_family", "endpoint_name", "source_native_endpoint"],
            filter=pads.field("qt_ecg_context") == True,  # noqa: E712
        )
        forbidden_target = re.compile(r"\b(?:herg|qtc?|ecg|electrocardiogram)\b", re.IGNORECASE)
        if any(
            forbidden_target.search(" ".join(str(value or "") for value in row.values()))
            for row in qt_context.to_pylist()
        ):
            raise PKADMESurfaceError("QT/ECG source context was promoted into a modeled target")
    expected_summary_counts = {
        "measurement_observation_count": measurement_rows,
        "modeling_surface_count": modeling_table.num_rows,
        "task_count": task_table.num_rows,
        "trainable_task_count": len(trainable_tasks),
        "unique_molecule_count": molecule_table.num_rows,
        "bound_input_count": len(inputs),
        "artifact_count": len(manifest["artifacts"]),
        "clinical_pk_context_observation_count": clinical_context_rows,
        "qt_ecg_context_observation_count": qt_ecg_context_rows,
    }
    for key, expected_value in expected_summary_counts.items():
        if int(summary.get(key, -1)) != expected_value:
            raise PKADMESurfaceError(f"Release summary count mismatch for {key}")
    report_text = report_path.read_text(encoding="utf-8")
    if any(line.lstrip().startswith("|") for line in report_text.splitlines()):
        raise PKADMESurfaceError("Release report contains a Markdown table")
    if "```mermaid" in report_text or re.search(r"!\[[^]]*\]\(", report_text):
        raise PKADMESurfaceError("Release report contains a figure")
    validation = {
        "schema_version": SCHEMA_VERSION,
        "validated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "artifact_manifest_file_sha256": _sha256_file(manifest_path),
        "artifact_manifest_payload_self_hash_verified": True,
        "artifact_manifest_payload_sha256": manifest["manifest_payload_sha256"],
        "manifest_hash_cycle_contract": "manifest self-hashes its canonical payload excluding manifest_payload_sha256; validation binds the complete manifest file; validation does not hash itself",
        "artifact_checks": artifact_checks,
        "input_binding_count": len(inputs),
        "input_binding_checks": input_checks,
        "builder_implementation_bound": True,
        "parquet_input_binding_count": parquet_input_count,
        "parquet_input_rows_and_arrow_schemas_verified": True,
        "closed_output_membership_verified": True,
        "exact_report_location_and_membership_verified": True,
        "measurement_observation_ids_unique": True,
        "measurement_row_count": measurement_rows,
        "modeling_row_count": pq.ParquetFile(modeling_path).metadata.num_rows,
        "task_count": task_table.num_rows,
        "trainable_task_count": len(trainable_tasks),
        "modeling_tasks_subset_of_trainable_registry": True,
        "modeling_rows_all_eligible_and_nonquarantine": True,
        "task_source_and_summary_counts_reconciled": True,
        "zero_label_sources_enforced": ["dailymed", "drugs_at_fda", "pkdb"],
        "task_and_molecule_registry_ids_unique": True,
        "molecule_structure_and_scaffold_identity_consistent": True,
        "chembl_candidate_and_noncandidate_disposition_arithmetic_verified": True,
        "clinical_pk_context_observation_count": clinical_context_rows,
        "qt_ecg_context_observation_count": qt_ecg_context_rows,
        "qt_ecg_context_not_promoted_to_target": True,
        "report_contains_no_tables_or_figures": True,
        "canonical_admission_false_for_all_rows": True,
        "forbidden_clinical_qt_herg_label_fields_absent": True,
        "report_exists": report_path.is_file(),
        "status": "passed",
    }
    return validation


def build_pk_adme_trainable_surfaces(
    repo_root: Path,
    *,
    output_dir: Path | None = None,
    report_path: Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    output_dir = output_dir or (repo_root / "research/data/platform/processed/pk_adme" / RELEASE_VERSION)
    report_path = report_path or (
        repo_root
        / "research/reports/platform/pk_adme_trainable_surfaces_v1"
        / "PK_ADME_TRAINABLE_SURFACES.md"
    )
    output_dir = output_dir.resolve()
    report_path = report_path.resolve()
    if output_dir.exists() and not overwrite:
        raise PKADMESurfaceError(
            f"Release already exists; pass overwrite=True only for this versioned path: {output_dir}"
        )
    if report_path.exists() and not overwrite:
        raise PKADMESurfaceError(f"Report already exists; pass overwrite=True: {report_path}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{RELEASE_VERSION}.", dir=output_dir.parent))
    temporary_report = temporary / "PK_ADME_TRAINABLE_SURFACES.md"
    bindings: list[dict[str, Any]] = []
    try:
        measurement_path = temporary / "measurement_ledger.parquet"
        writer = SurfaceWriter(measurement_path)
        _add_governance_bindings(repo_root, bindings)
        _load_chembl(writer, repo_root, bindings)
        _load_tdc(writer, repo_root, bindings)
        _load_openadmet(writer, repo_root, bindings)
        _load_ncats(writer, repo_root, bindings)
        _add_nonmeasurement_blockers(writer, repo_root, bindings)
        writer.close()

        binding_paths = [item["path"] for item in bindings]
        if len(binding_paths) != len(set(binding_paths)):
            duplicates = sorted(path for path, count in Counter(binding_paths).items() if count > 1)
            raise PKADMESurfaceError(f"Duplicate physical input bindings: {duplicates}")
        task_rows = _task_rows(writer)
        molecule_rows = _molecule_rows(writer)
        disposition_rows = _disposition_rows(writer)
        task_path = temporary / "task_registry.parquet"
        molecule_path = temporary / "molecule_registry.parquet"
        disposition_path = temporary / "source_disposition.parquet"
        pq.write_table(pa.Table.from_pylist(task_rows, schema=_TASK_SCHEMA), task_path, compression="zstd")
        pq.write_table(
            pa.Table.from_pylist(molecule_rows, schema=_MOLECULE_SCHEMA), molecule_path, compression="zstd"
        )
        pq.write_table(
            pa.Table.from_pylist(disposition_rows, schema=_DISPOSITION_SCHEMA),
            disposition_path,
            compression="zstd",
        )
        trainable_tasks = {row["task_id"] for row in task_rows if row["task_trainable"]}
        modeling_path = temporary / "modeling_surface.parquet"
        modeling_count = _write_modeling_surface(measurement_path, modeling_path, trainable_tasks)
        input_binding_path = temporary / "input_binding_manifest.json"
        input_binding_payload = {
            "schema_version": SCHEMA_VERSION,
            "binding_count": len(bindings),
            "all_declared_hashes_verified": all(item["declared_hash_verified"] for item in bindings),
            "inputs": sorted(bindings, key=lambda item: item["path"]),
        }
        input_binding_path.write_text(_canonical_json(input_binding_payload) + "\n", encoding="utf-8")
        contract_path = temporary / "modeling_contract.json"
        contract_path.write_text(_canonical_json(_contract_payload()) + "\n", encoding="utf-8")

        by_source: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for row in disposition_rows:
            source = row["source_family"]
            by_source[source]["observations"] += int(row["emitted_measurement_count"])
            by_source[source]["exact_modeling"] += int(row["exact_modeling_count"])
            by_source[source]["censored_only_modeling"] += int(row["censored_only_modeling_count"])
        summary = {
            "schema_version": SCHEMA_VERSION,
            "release_version": RELEASE_VERSION,
            "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "measurement_observation_count": writer.total_observations,
            "modeling_surface_count": modeling_count,
            "exact_modeling_observation_count": sum(
                row["exact_modeling_count"] for row in task_rows if row["task_trainable"]
            ),
            "censored_only_modeling_observation_count": sum(
                row["censored_only_modeling_count"] for row in task_rows if row["task_trainable"]
            ),
            "unique_molecule_count": len(molecule_rows),
            "unique_leakage_group_count": len({row["leakage_group_id"] for row in molecule_rows}),
            "task_count": len(task_rows),
            "trainable_task_count": len(trainable_tasks),
            "context_complete_observation_count": sum(row["context_complete_count"] for row in task_rows),
            "cross_source_union_candidate_count": sum(row["cross_source_union_count"] for row in task_rows),
            "clinical_pk_context_observation_count": writer.clinical_pk_context_observations,
            "qt_ecg_context_observation_count": writer.qt_ecg_context_observations,
            "bound_input_count": len(bindings),
            "artifact_count": 9,
            "by_source_family": {key: dict(value) for key, value in sorted(by_source.items())},
            "scientific_boundary": "noncanonical source-native PK/ADME modeling preparation; human clinical PK context may be retained and flagged, but no clinical-outcome/QT/hERG/safety/efficacy labels are created",
        }
        summary_path = temporary / "release_summary.json"
        summary_path.write_text(_canonical_json(summary) + "\n", encoding="utf-8")
        _write_report(temporary_report, summary, task_rows)

        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True)
        data_files = [
            measurement_path,
            modeling_path,
            task_path,
            molecule_path,
            disposition_path,
            input_binding_path,
            contract_path,
            summary_path,
        ]
        for path in data_files:
            shutil.move(str(path), output_dir / path.name)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        if report_path.exists():
            report_path.unlink()
        shutil.move(str(temporary_report), report_path)
        artifact_roles = {
            "measurement_ledger.parquet": "all_parseable_source_bound_endpoint_observations",
            "modeling_surface.parquet": "endpoint_specific_exact_and_censored_modeling_rows",
            "task_registry.parquet": "task_semantics_and_counts",
            "molecule_registry.parquet": "standardized_identity_and_leakage_groups",
            "source_disposition.parquet": "physical_rows_measurements_and_exclusions",
            "input_binding_manifest.json": "physical_input_hash_bindings",
            "modeling_contract.json": "units_censoring_identity_and_scope_contract",
            "release_summary.json": "honest_release_counts",
        }
        row_counts = {
            "measurement_ledger.parquet": writer.total_observations,
            "modeling_surface.parquet": modeling_count,
            "task_registry.parquet": len(task_rows),
            "molecule_registry.parquet": len(molecule_rows),
            "source_disposition.parquet": len(disposition_rows),
        }
        artifacts = [
            _artifact_entry(repo_root, output_dir / name, role, row_counts.get(name))
            for name, role in artifact_roles.items()
        ]
        artifacts.append(_artifact_entry(repo_root, report_path, "release_report_no_tables_or_figures"))
        artifact_manifest = {
            "schema_version": SCHEMA_VERSION,
            "release_version": RELEASE_VERSION,
            "artifact_count": len(artifacts),
            "expected_output_members": sorted(_EXPECTED_OUTPUT_MEMBERS),
            "expected_report_path": str(report_path.relative_to(repo_root)),
            "manifest_payload_self_hash_contract": "SHA-256 of canonical JSON after omitting manifest_payload_sha256",
            "manifest_file_hash_contract": "the complete artifact_manifest.json file is SHA-256 bound by validation.json",
            "validation_hash_cycle_contract": "validation.json is excluded from artifact hashes because it binds the complete artifact manifest and does not self-hash",
            "artifacts": artifacts,
        }
        artifact_manifest = self_hash_manifest(artifact_manifest)
        manifest_path = output_dir / "artifact_manifest.json"
        manifest_path.write_text(_canonical_json(artifact_manifest) + "\n", encoding="utf-8")
        validation_path = output_dir / "validation.json"
        validation_path.write_text(
            _canonical_json(
                {
                    "schema_version": SCHEMA_VERSION,
                    "status": "pending_closed_membership_validation",
                    "self_hash_exclusion": "validation does not hash itself",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        validation = validate_release(repo_root, output_dir, report_path)
        validation_path.write_text(_canonical_json(validation) + "\n", encoding="utf-8")
        # Re-run against the final validation file so closed membership is not
        # merely checked against a pre-publication directory.
        final_validation = validate_release(repo_root, output_dir, report_path)
        if {key: value for key, value in final_validation.items() if key != "validated_at_utc"} != {
            key: value for key, value in validation.items() if key != "validated_at_utc"
        }:
            raise PKADMESurfaceError("Final release validation changed after writing validation.json")
        summary["artifact_count"] = len(artifacts)
        summary["artifact_manifest_sha256"] = _sha256_file(manifest_path)
        summary["validation_sha256"] = _sha256_file(validation_path)
        summary["output_dir"] = str(output_dir)
        summary["report_path"] = str(report_path)
        return summary
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    finally:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--report-path", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    summary = build_pk_adme_trainable_surfaces(
        args.repo_root,
        output_dir=args.output_dir,
        report_path=args.report_path,
        overwrite=args.overwrite,
    )
    print(_canonical_json(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
