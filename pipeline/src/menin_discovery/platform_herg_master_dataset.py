"""Compile the paper-facing, standardized wild-type hERG master dataset.

The compiler is deliberately additive.  It binds immutable upstream artifacts,
excludes explicit hERG variants, preserves every native observation column, and
adds standardized *interpretations* without overwriting reported values.  It
also provides a label-free structure table with an interpretable RDKit 2D
descriptor panel.  pKa, poses, conformations, and three-dimensional physics are
not fabricated when the source data cannot support them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .features import RDKIT_AVAILABLE, scaffold_key

if RDKIT_AVAILABLE:  # pragma: no branch - production dependency is installed.
    from rdkit import Chem
    from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors

SCHEMA_VERSION = "platform-herg-master-dataset/1.0"
DATASET_ID = "wild_type_herg_paper_master_v1_3"
MANIFEST_NAME = "herg_master_manifest.json"
REPORT_NAME = "HERG_MASTER_DATASET.md"

OBSERVATION_OUTPUT = "observation_master.parquet"
STRUCTURE_OUTPUT = "structure_master.parquet"
EVIDENCE_OUTPUT = "structure_evidence_summary.parquet"
FEATURE_SUMMARY_OUTPUT = "fundamental_feature_summary.parquet"
ASSAY_OUTPUT = "assay_catalog.parquet"
METHOD_SUMMARY_OUTPUT = "method_endpoint_summary.parquet"
PROTOCOL_OUTPUT = "assay_protocol_index.parquet"
TASK_OUTPUT = "task_membership.parquet"
CLINICAL_OUTPUT = "clinical_context_master.parquet"
EXCLUSION_OUTPUT = "master_exclusions.parquet"

_TASK_FILES = (
    "q0_weak_fixed_dose_binary.parquet",
    "q1_quantitative_pic50.parquet",
    "q2_functional_assay_aware.parquet",
    "c0_clinical_development_context.parquet",
)
_QT_TASK_FILE = "c1_qt_context_endpoints.parquet"

_OBSERVATION_REQUIRED = (
    "observation_id",
    "source_family",
    "source_record_id",
    "standardized_smiles",
    "standard_inchi_key",
    "structure_id",
    "structure_valid",
    "target_variant",
    "assay_id",
    "assay_family",
    "native_endpoint",
    "native_relation",
    "native_value",
    "native_unit",
    "pic50_value",
    "pic50_origin",
    "native_aux_json",
)

_OBSERVATION_DERIVED_SCHEMA = pa.schema(
    [
        pa.field("master_dataset_id", pa.large_string(), nullable=False),
        pa.field("wild_type_evidence_scope", pa.large_string(), nullable=False),
        pa.field("measurement_modality", pa.large_string(), nullable=False),
        pa.field("method_detail", pa.large_string(), nullable=False),
        pa.field("modality_confidence", pa.large_string(), nullable=False),
        pa.field("automation_class", pa.large_string(), nullable=False),
        pa.field("automation_confidence", pa.large_string(), nullable=False),
        pa.field("dose_design", pa.large_string(), nullable=False),
        pa.field("dose_design_confidence", pa.large_string(), nullable=False),
        pa.field("endpoint_class", pa.large_string(), nullable=False),
        pa.field("endpoint_standardization_status", pa.large_string(), nullable=False),
        pa.field("potency_relation_pic50", pa.large_string()),
        pa.field("potency_pic50_point", pa.float64()),
        pa.field("potency_pic50_lower_bound", pa.float64()),
        pa.field("potency_pic50_upper_bound", pa.float64()),
        pa.field("potency_censoring", pa.large_string(), nullable=False),
        pa.field("potency_standardization_basis", pa.large_string(), nullable=False),
        pa.field("structure_model_eligible", pa.bool_(), nullable=False),
        pa.field("model_split", pa.large_string()),
        pa.field("scaffold_group_id", pa.large_string()),
    ]
)

_STRUCTURE_SCHEMA = pa.schema(
    [
        pa.field("structure_id", pa.large_string(), nullable=False),
        pa.field("standardized_smiles", pa.large_string(), nullable=False),
        pa.field("standard_inchi_key", pa.large_string(), nullable=False),
        pa.field("model_split", pa.large_string(), nullable=False),
        pa.field("scaffold_group_id", pa.large_string(), nullable=False),
        pa.field("split_source", pa.large_string(), nullable=False),
        pa.field("structure_representation_count", pa.int32(), nullable=False),
        pa.field("structure_representations_json", pa.large_string(), nullable=False),
        pa.field("structure_resolution_policy", pa.large_string(), nullable=False),
        pa.field("molecular_formula", pa.large_string()),
        pa.field("molecular_weight", pa.float64()),
        pa.field("exact_molecular_weight", pa.float64()),
        pa.field("heavy_atom_molecular_weight", pa.float64()),
        pa.field("mol_logp", pa.float64()),
        pa.field("molar_refractivity", pa.float64()),
        pa.field("topological_polar_surface_area", pa.float64()),
        pa.field("labute_accessible_surface_area", pa.float64()),
        pa.field("hydrogen_bond_donors", pa.int32()),
        pa.field("hydrogen_bond_acceptors", pa.int32()),
        pa.field("rotatable_bonds", pa.int32()),
        pa.field("ring_count", pa.int32()),
        pa.field("aromatic_ring_count", pa.int32()),
        pa.field("aliphatic_ring_count", pa.int32()),
        pa.field("saturated_ring_count", pa.int32()),
        pa.field("heterocycle_count", pa.int32()),
        pa.field("fraction_csp3", pa.float64()),
        pa.field("formal_charge", pa.int32()),
        pa.field("heavy_atom_count", pa.int32()),
        pa.field("heteroatom_count", pa.int32()),
        pa.field("carbon_atom_count", pa.int32()),
        pa.field("halogen_atom_count", pa.int32()),
        pa.field("stereocenter_count", pa.int32()),
        pa.field("largest_ring_size", pa.int32()),
        pa.field("feature_status", pa.large_string(), nullable=False),
        pa.field("feature_missing_count", pa.int32(), nullable=False),
        pa.field("feature_missing_fields_json", pa.large_string(), nullable=False),
    ]
)

_EVIDENCE_SCHEMA = pa.schema(
    [
        pa.field("structure_id", pa.large_string(), nullable=False),
        pa.field("observation_count", pa.int64(), nullable=False),
        pa.field("confirmed_wild_type_observation_count", pa.int64(), nullable=False),
        pa.field("wild_type_or_unspecified_observation_count", pa.int64(), nullable=False),
        pa.field("source_families_json", pa.large_string(), nullable=False),
        pa.field("assay_count", pa.int64(), nullable=False),
        pa.field("endpoint_classes_json", pa.large_string(), nullable=False),
        pa.field("measurement_modalities_json", pa.large_string(), nullable=False),
        pa.field("task_ids_json", pa.large_string(), nullable=False),
        pa.field("has_clinical_development_context", pa.bool_(), nullable=False),
        pa.field("has_qt_qtc_context", pa.bool_(), nullable=False),
        pa.field("model_feature_eligible", pa.bool_(), nullable=False),
        pa.field("feature_exclusion_reason", pa.large_string(), nullable=False),
    ]
)

_FEATURE_SUMMARY_SCHEMA = pa.schema(
    [
        pa.field("feature_name", pa.large_string(), nullable=False),
        pa.field("semantic_group", pa.large_string(), nullable=False),
        pa.field("value_type", pa.large_string(), nullable=False),
        pa.field("nonmissing_count", pa.int64(), nullable=False),
        pa.field("missing_count", pa.int64(), nullable=False),
        pa.field("mean", pa.float64()),
        pa.field("sample_standard_deviation", pa.float64()),
        pa.field("minimum", pa.float64()),
        pa.field("percentile_05", pa.float64()),
        pa.field("median", pa.float64()),
        pa.field("percentile_95", pa.float64()),
        pa.field("maximum", pa.float64()),
        pa.field("training_feature_eligible", pa.bool_(), nullable=False),
    ]
)

_ASSAY_SCHEMA = pa.schema(
    [
        pa.field("assay_catalog_id", pa.large_string(), nullable=False),
        pa.field("source_family", pa.large_string(), nullable=False),
        pa.field("assay_id", pa.large_string()),
        pa.field("assay_family", pa.large_string(), nullable=False),
        pa.field("observation_count", pa.int64(), nullable=False),
        pa.field("unique_structure_count", pa.int64(), nullable=False),
        pa.field("target_scopes_json", pa.large_string(), nullable=False),
        pa.field("endpoint_classes_json", pa.large_string(), nullable=False),
        pa.field("native_endpoints_json", pa.large_string(), nullable=False),
        pa.field("native_relations_json", pa.large_string(), nullable=False),
        pa.field("native_units_json", pa.large_string(), nullable=False),
        pa.field("measurement_modalities_json", pa.large_string(), nullable=False),
        pa.field("automation_classes_json", pa.large_string(), nullable=False),
        pa.field("dose_designs_json", pa.large_string(), nullable=False),
        pa.field("assay_descriptions_json", pa.large_string(), nullable=False),
        pa.field("catalog_status", pa.large_string(), nullable=False),
    ]
)

_METHOD_SUMMARY_SCHEMA = pa.schema(
    [
        pa.field("wild_type_evidence_scope", pa.large_string(), nullable=False),
        pa.field("measurement_modality", pa.large_string(), nullable=False),
        pa.field("automation_class", pa.large_string(), nullable=False),
        pa.field("dose_design", pa.large_string(), nullable=False),
        pa.field("endpoint_class", pa.large_string(), nullable=False),
        pa.field("observation_count", pa.int64(), nullable=False),
        pa.field("unique_structure_count", pa.int64(), nullable=False),
        pa.field("unique_assay_count", pa.int64(), nullable=False),
        pa.field("exact_pic50_count", pa.int64(), nullable=False),
        pa.field("censored_pic50_count", pa.int64(), nullable=False),
        pa.field("approximate_pic50_count", pa.int64(), nullable=False),
        pa.field("unstandardized_count", pa.int64(), nullable=False),
    ]
)

_PROTOCOL_SCHEMA = pa.schema(
    [
        pa.field("assay_catalog_id", pa.large_string(), nullable=False),
        pa.field("source_family", pa.large_string(), nullable=False),
        pa.field("assay_id", pa.large_string()),
        pa.field("assay_family", pa.large_string(), nullable=False),
        pa.field("observation_count", pa.int64(), nullable=False),
        pa.field("raw_protocol_text_json", pa.large_string(), nullable=False),
        pa.field("source_contract_evidence_json", pa.large_string(), nullable=False),
        pa.field("normalization_evidence_text_json", pa.large_string(), nullable=False),
        pa.field("host_systems_json", pa.large_string(), nullable=False),
        pa.field("host_system_evidence_json", pa.large_string(), nullable=False),
        pa.field("host_system_confidence", pa.large_string(), nullable=False),
        pa.field("voltage_values_mv_json", pa.large_string(), nullable=False),
        pa.field("voltage_evidence_json", pa.large_string(), nullable=False),
        pa.field("voltage_confidence", pa.large_string(), nullable=False),
        pa.field("temperature_values_celsius_json", pa.large_string(), nullable=False),
        pa.field("temperature_condition_terms_json", pa.large_string(), nullable=False),
        pa.field("temperature_evidence_json", pa.large_string(), nullable=False),
        pa.field("temperature_confidence", pa.large_string(), nullable=False),
        pa.field("time_values_seconds_json", pa.large_string(), nullable=False),
        pa.field("time_evidence_json", pa.large_string(), nullable=False),
        pa.field("time_confidence", pa.large_string(), nullable=False),
        pa.field("recording_configurations_json", pa.large_string(), nullable=False),
        pa.field("recording_configuration_evidence_json", pa.large_string(), nullable=False),
        pa.field("recording_configuration_confidence", pa.large_string(), nullable=False),
        pa.field("named_platforms_json", pa.large_string(), nullable=False),
        pa.field("platform_evidence_json", pa.large_string(), nullable=False),
        pa.field("platform_confidence", pa.large_string(), nullable=False),
        pa.field("manual_automation_evidence_json", pa.large_string(), nullable=False),
        pa.field("source_automation_classes_json", pa.large_string(), nullable=False),
        pa.field("unresolved_fields_json", pa.large_string(), nullable=False),
        pa.field("protocol_completeness_score", pa.int8(), nullable=False),
        pa.field("normalization_policy", pa.large_string(), nullable=False),
    ]
)

_TASK_SCHEMA = pa.schema(
    [
        pa.field("membership_id", pa.large_string(), nullable=False),
        pa.field("task_id", pa.large_string(), nullable=False),
        pa.field("quality_level", pa.large_string(), nullable=False),
        pa.field("source_artifact", pa.large_string(), nullable=False),
        pa.field("record_id", pa.large_string(), nullable=False),
        pa.field("observation_id", pa.large_string()),
        pa.field("structure_id", pa.large_string()),
        pa.field("target_scope", pa.large_string(), nullable=False),
        pa.field("source_family", pa.large_string(), nullable=False),
        pa.field("native_endpoint", pa.large_string(), nullable=False),
        pa.field("measurement_technology", pa.large_string(), nullable=False),
        pa.field("model_split", pa.large_string()),
        pa.field("scaffold_group_id", pa.large_string()),
        pa.field("eligible", pa.bool_(), nullable=False),
        pa.field("direct_herg_label", pa.bool_(), nullable=False),
        pa.field("use_as_training_label", pa.bool_(), nullable=False),
        pa.field("clinical_context_only", pa.bool_(), nullable=False),
        pa.field("target_relation_pic50", pa.large_string()),
        pa.field("target_pic50_point", pa.float64()),
        pa.field("target_pic50_lower_bound", pa.float64()),
        pa.field("target_pic50_upper_bound", pa.float64()),
        pa.field("target_class", pa.int8()),
        pa.field("eligibility_reason", pa.large_string(), nullable=False),
        pa.field("exclusion_reason", pa.large_string()),
        pa.field("quality_flags", pa.large_string(), nullable=False),
    ]
)

_CLINICAL_SCHEMA = pa.schema(
    [
        pa.field("clinical_context_id", pa.large_string(), nullable=False),
        pa.field("context_class", pa.large_string(), nullable=False),
        pa.field("structure_id", pa.large_string(), nullable=False),
        pa.field("nct_id", pa.large_string()),
        pa.field("endpoint_candidate_id", pa.large_string()),
        pa.field("model_split", pa.large_string(), nullable=False),
        pa.field("scaffold_group_id", pa.large_string(), nullable=False),
        pa.field("context_eligible", pa.bool_(), nullable=False),
        pa.field("heldout_evaluation_eligible", pa.bool_(), nullable=False),
        pa.field("direct_herg_label", pa.bool_(), nullable=False),
        pa.field("use_as_training_label", pa.bool_(), nullable=False),
        pa.field("title_or_term", pa.large_string()),
        pa.field("description_or_organ_system", pa.large_string()),
        pa.field("unit_of_measure", pa.large_string()),
        pa.field("time_frame", pa.large_string()),
        pa.field("native_context_json", pa.large_string(), nullable=False),
    ]
)

_EXCLUSION_SCHEMA = pa.schema(
    [
        pa.field("master_exclusion_id", pa.large_string(), nullable=False),
        pa.field("exclusion_scope", pa.large_string(), nullable=False),
        pa.field("task_id", pa.large_string()),
        pa.field("source_family", pa.large_string(), nullable=False),
        pa.field("source_record_id", pa.large_string(), nullable=False),
        pa.field("observation_id", pa.large_string()),
        pa.field("structure_id", pa.large_string()),
        pa.field("target_scope", pa.large_string(), nullable=False),
        pa.field("exclusion_reason", pa.large_string(), nullable=False),
        pa.field("exclusion_detail", pa.large_string(), nullable=False),
        pa.field("native_exclusion_context_json", pa.large_string(), nullable=False),
        pa.field("source_artifact", pa.large_string(), nullable=False),
    ]
)

DESCRIPTOR_VALUE_COLUMNS = (
    "molecular_formula",
    "molecular_weight",
    "exact_molecular_weight",
    "heavy_atom_molecular_weight",
    "mol_logp",
    "molar_refractivity",
    "topological_polar_surface_area",
    "labute_accessible_surface_area",
    "hydrogen_bond_donors",
    "hydrogen_bond_acceptors",
    "rotatable_bonds",
    "ring_count",
    "aromatic_ring_count",
    "aliphatic_ring_count",
    "saturated_ring_count",
    "heterocycle_count",
    "fraction_csp3",
    "formal_charge",
    "heavy_atom_count",
    "heteroatom_count",
    "carbon_atom_count",
    "halogen_atom_count",
    "stereocenter_count",
    "largest_ring_size",
)
FEATURE_COLUMNS = DESCRIPTOR_VALUE_COLUMNS[1:]
SPLIT_SALT = "platform-herg-aid720551-scaffold-split-v1"


class HergMasterDatasetError(RuntimeError):
    """Raised when master compilation or validation fails closed."""


_HOST_PATTERNS = {
    "HEK293": re.compile(r"\bHEK[\s-]?293(?:T)?\b|human embryonic kidney", re.IGNORECASE),
    "CHO": re.compile(r"\bCHO(?:[\s-]?K1)?\b|Chinese hamster ovary", re.IGNORECASE),
    "Xenopus_oocyte": re.compile(r"\bXenopus\b|\boocyte(?:s)?\b", re.IGNORECASE),
    "U2OS": re.compile(r"\bU[\s-]?2[\s-]?OS\b", re.IGNORECASE),
    "COS7": re.compile(r"\bCOS[\s-]?7\b", re.IGNORECASE),
}
_RECORDING_PATTERNS = {
    "whole_cell": re.compile(r"\bwhole[\s-]?cell\b", re.IGNORECASE),
    "tail_current": re.compile(r"\btail current(?:s)?\b", re.IGNORECASE),
    "patch_clamp": re.compile(r"\bpatch[\s-]?clamp\b", re.IGNORECASE),
    "voltage_clamp": re.compile(r"\bvoltage[\s-]?clamp\b", re.IGNORECASE),
    "thallium_flux": re.compile(r"\bthallium[\s-]?flux\b|\bFluxOR\b", re.IGNORECASE),
    "fluorescence_polarization": re.compile(r"\bfluorescence polarization\b", re.IGNORECASE),
    "radioligand_binding": re.compile(r"\bradioligand\b|\[3H\]|\[35S\]", re.IGNORECASE),
}
_PLATFORM_PATTERNS = {
    "IonWorks": re.compile(r"\bIonWorks\b", re.IGNORECASE),
    "QPatch": re.compile(r"\bQ[\s-]?Patch\b", re.IGNORECASE),
    "Patchliner": re.compile(r"\bPatchliner\b", re.IGNORECASE),
    "SyncroPatch": re.compile(r"\bSyncroPatch\b", re.IGNORECASE),
    "PatchXpress": re.compile(r"\bPatchXpress\b", re.IGNORECASE),
    "FluxOR": re.compile(r"\bFluxOR\b", re.IGNORECASE),
    "FLIPR": re.compile(r"\bFLIPR\b", re.IGNORECASE),
}
_MANUAL_PATTERN = re.compile(
    r"\bmanual(?:ly)?\b|\bconventional patch|\bglass (?:micro)?pipette", re.IGNORECASE
)
_VOLTAGE_PATTERN = re.compile(r"(?<![\w.])([+-]?\d+(?:\.\d+)?)\s*mV\b", re.IGNORECASE)
_TEMPERATURE_PATTERN = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s*(?:°\s*)?([CK])\b", re.IGNORECASE)
_ROOM_TEMPERATURE_PATTERN = re.compile(
    r"\broom temp(?:erature)?\b|\bambient temp(?:erature)?\b", re.IGNORECASE
)
_TIME_PATTERN = re.compile(
    r"(?<![\w.])(\d+(?:\.\d+)?)\s*(milliseconds?|msec|seconds?|secs?|minutes?|mins?|hours?|hrs?)\b",
    re.IGNORECASE,
)
_SOURCE_PROTOCOL_CONTRACTS = {
    "pubchem_aid720551": [
        "Curated PubChem AID 720551 source contract: confirmatory wild-type KCNH2 U2OS FluxOR thallium-flux qHTS at 0.369 and 1.840 uM."
    ]
}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_id(prefix: str, *parts: object) -> str:
    payload = "\x1f".join("" if value is None else str(value) for value in parts)
    return f"{prefix}-{hashlib.sha256(payload.encode()).hexdigest()[:24].upper()}"


def _checked_parquet(path: Path, role: str, required: Iterable[str] = ()) -> Path:
    if path.is_symlink() or not path.is_file() or path.suffix.casefold() != ".parquet":
        raise HergMasterDatasetError(f"missing, unsafe, or non-Parquet {role}: {path}")
    missing = sorted(set(required) - set(pq.ParquetFile(path).schema_arrow.names))
    if missing:
        raise HergMasterDatasetError(f"{role} is missing required columns: {missing}")
    return path.resolve()


def _input(role: str, path: Path) -> dict[str, Any]:
    return {
        "role": role,
        "path": str(path),
        "rows": pq.ParquetFile(path).metadata.num_rows,
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _artifact(path: Path) -> dict[str, Any]:
    parquet = pq.ParquetFile(path)
    return {
        "path": path.name,
        "rows": parquet.metadata.num_rows,
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
        "arrow_schema_sha256": hashlib.sha256(parquet.schema_arrow.serialize().to_pybytes()).hexdigest(),
    }


def _write(path: Path, table: pa.Table) -> dict[str, Any]:
    pq.write_table(
        table,
        path,
        compression="zstd",
        use_dictionary=False,
        write_statistics=True,
        data_page_version="1.0",
        version="2.6",
        row_group_size=65_536,
    )
    return _artifact(path)


def _finite(value: object) -> float | None:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _endpoint_class(endpoint: object) -> str:
    text = str(endpoint or "").strip().casefold()
    if text == "activity_outcome":
        return "categorical_activity_call"
    if text == "pic50":
        return "potency_pic50"
    if text == "ic50":
        return "potency_ic50"
    if "ic50" in text or text.startswith("ic") or text in {"xc50", "potency"}:
        return "potency_other_or_derived"
    if text == "ac50":
        return "potency_ac50"
    if text == "ec50":
        return "effect_ec50"
    if text in {"ki", "pki"}:
        return "binding_affinity_ki"
    if text in {"kd", "pkd"}:
        return "binding_affinity_kd"
    if text in {"inhibition", "inh", "ip", "% ctrl", "imax", "inhibitory activity"}:
        return "inhibition_or_effect_level"
    if text in {"kon", "k_on", "koff", "k_off", "time"}:
        return "binding_or_effect_kinetics"
    if "qt" in text:
        return "clinical_qt_qtc_phenotype"
    if "ratio" in text or text == "fc":
        return "ratio_or_fold_change"
    return "other_reported_endpoint"


def _pic50_standardization(row: Mapping[str, Any]) -> dict[str, Any]:
    endpoint = str(row.get("native_endpoint") or "").strip().casefold()
    relation = str(row.get("native_relation") or "").strip()
    relation = relation if relation in {"=", "~", ">", ">=", "<", "<="} else ""
    value: float | None = None
    basis = "not_standardized_incompatible_or_insufficient_native_endpoint"
    if endpoint == "pic50":
        value = _finite(row.get("pic50_value"))
        if value is None:
            value = _finite(row.get("native_value"))
        basis = "source_reported_pic50_preserved" if value is not None else "missing_nonfinite_reported_pic50"
    elif endpoint == "ic50":
        native = _finite(row.get("native_value"))
        units = {"pm": 12.0, "nm": 9.0, "um": 6.0, "µm": 6.0, "mm": 3.0, "m": 0.0}
        offset = units.get(str(row.get("native_unit") or "").strip().casefold())
        if native is not None and native > 0 and offset is not None:
            value = offset - math.log10(native)
            basis = "native_ic50_concentration_log10_molar_conversion"
        else:
            basis = "invalid_nonpositive_or_unsupported_ic50_value_unit"
    if value is None:
        return {
            "endpoint_standardization_status": "not_standardized",
            "potency_relation_pic50": None,
            "potency_pic50_point": None,
            "potency_pic50_lower_bound": None,
            "potency_pic50_upper_bound": None,
            "potency_censoring": "not_applicable",
            "potency_standardization_basis": basis,
        }
    inverted = {"=": "=", "~": "~", ">": "<", ">=": "<=", "<": ">", "<=": ">="}
    pic50_relation = relation if endpoint == "pic50" and relation else inverted.get(relation)
    if pic50_relation is None:
        return {
            "endpoint_standardization_status": "value_available_relation_unresolved",
            "potency_relation_pic50": None,
            "potency_pic50_point": value,
            "potency_pic50_lower_bound": None,
            "potency_pic50_upper_bound": None,
            "potency_censoring": "relation_unresolved",
            "potency_standardization_basis": basis,
        }
    if pic50_relation == "=":
        lower, upper, censoring, status = value, value, "exact", "exact_standardized"
    elif pic50_relation == "~":
        lower, upper, censoring, status = None, None, "approximate_unbounded", "approximate_standardized"
    elif pic50_relation in {">", ">="}:
        lower, upper, censoring, status = value, None, "pic50_lower_bounded", "censored_standardized"
    else:
        lower, upper, censoring, status = None, value, "pic50_upper_bounded", "censored_standardized"
    return {
        "endpoint_standardization_status": status,
        "potency_relation_pic50": pic50_relation,
        "potency_pic50_point": value if pic50_relation in {"=", "~"} else None,
        "potency_pic50_lower_bound": lower,
        "potency_pic50_upper_bound": upper,
        "potency_censoring": censoring,
        "potency_standardization_basis": basis,
    }


def _split_draw(group_id: str) -> str:
    draw = int.from_bytes(hashlib.sha256(f"{SPLIT_SALT}\x1f{group_id}".encode()).digest()[:8], "big") / 2**64
    return "train" if draw < 0.8 else ("validation" if draw < 0.9 else "test")


def _computed_split(smiles: str) -> tuple[str, str]:
    key, method = scaffold_key(smiles)
    if not key:
        raise HergMasterDatasetError("cannot compute a scaffold key for a valid standardized structure")
    scaffold_payload = f"{method}\x1f{key}"
    group = f"HSCF-{hashlib.sha256(scaffold_payload.encode()).hexdigest().upper()}"
    return _split_draw(group), group


def _load_split_map(split_path: Path, task_paths: Sequence[Path]) -> dict[str, tuple[str, str, str]]:
    mapping: dict[str, tuple[str, str, str]] = {}
    base = pq.read_table(split_path, columns=["structure_id", "split", "scaffold_group_id"]).to_pylist()
    for row in base:
        mapping[str(row["structure_id"])] = (
            str(row["split"]),
            str(row["scaffold_group_id"]),
            "v1_frozen_split",
        )
    for path in task_paths:
        table = pq.read_table(path, columns=["structure_id", "model_split", "scaffold_group_id"])
        for row in table.to_pylist():
            if not row["structure_id"] or not row["model_split"] or not row["scaffold_group_id"]:
                continue
            sid = str(row["structure_id"])
            candidate = (str(row["model_split"]), str(row["scaffold_group_id"]), "v1_2_quality_task_split")
            existing = mapping.get(sid)
            if existing and existing[:2] != candidate[:2]:
                raise HergMasterDatasetError(f"conflicting frozen split for {sid}")
            mapping.setdefault(sid, candidate)
    return mapping


def _descriptor_row(
    structure_id: str,
    smiles: str,
    inchi_key: str,
    split: tuple[str, str, str],
    representations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    base: dict[str, Any] = {
        "structure_id": structure_id,
        "standardized_smiles": smiles,
        "standard_inchi_key": inchi_key,
        "model_split": split[0],
        "scaffold_group_id": split[1],
        "split_source": split[2],
        "structure_representation_count": len(representations),
        "structure_representations_json": _canonical_json(representations),
        "structure_resolution_policy": "highest_observation_frequency_then_lexical_smiles_and_inchi_key",
    }
    if not RDKIT_AVAILABLE:
        missing = list(DESCRIPTOR_VALUE_COLUMNS)
        return {
            **base,
            **{name: None for name in DESCRIPTOR_VALUE_COLUMNS},
            "feature_status": "rdkit_unavailable",
            "feature_missing_count": len(missing),
            "feature_missing_fields_json": _canonical_json(missing),
        }
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        missing = list(DESCRIPTOR_VALUE_COLUMNS)
        return {
            **base,
            **{name: None for name in DESCRIPTOR_VALUE_COLUMNS},
            "feature_status": "invalid_standardized_smiles",
            "feature_missing_count": len(missing),
            "feature_missing_fields_json": _canonical_json(missing),
        }
    rings = mol.GetRingInfo().AtomRings()
    halogens = sum(atom.GetAtomicNum() in {9, 17, 35, 53, 85} for atom in mol.GetAtoms())
    values: dict[str, Any] = {
        "molecular_formula": rdMolDescriptors.CalcMolFormula(mol),
        "molecular_weight": float(Descriptors.MolWt(mol)),
        "exact_molecular_weight": float(Descriptors.ExactMolWt(mol)),
        "heavy_atom_molecular_weight": float(Descriptors.HeavyAtomMolWt(mol)),
        "mol_logp": float(Crippen.MolLogP(mol)),
        "molar_refractivity": float(Crippen.MolMR(mol)),
        "topological_polar_surface_area": float(rdMolDescriptors.CalcTPSA(mol)),
        "labute_accessible_surface_area": float(rdMolDescriptors.CalcLabuteASA(mol)),
        "hydrogen_bond_donors": int(Lipinski.NumHDonors(mol)),
        "hydrogen_bond_acceptors": int(Lipinski.NumHAcceptors(mol)),
        "rotatable_bonds": int(Lipinski.NumRotatableBonds(mol)),
        "ring_count": int(Lipinski.RingCount(mol)),
        "aromatic_ring_count": int(rdMolDescriptors.CalcNumAromaticRings(mol)),
        "aliphatic_ring_count": int(rdMolDescriptors.CalcNumAliphaticRings(mol)),
        "saturated_ring_count": int(rdMolDescriptors.CalcNumSaturatedRings(mol)),
        "heterocycle_count": int(rdMolDescriptors.CalcNumHeterocycles(mol)),
        "fraction_csp3": float(rdMolDescriptors.CalcFractionCSP3(mol)),
        "formal_charge": int(Chem.GetFormalCharge(mol)),
        "heavy_atom_count": int(mol.GetNumHeavyAtoms()),
        "heteroatom_count": int(rdMolDescriptors.CalcNumHeteroatoms(mol)),
        "carbon_atom_count": sum(atom.GetAtomicNum() == 6 for atom in mol.GetAtoms()),
        "halogen_atom_count": halogens,
        "stereocenter_count": len(Chem.FindMolChiralCenters(mol, includeUnassigned=True)),
        "largest_ring_size": max((len(ring) for ring in rings), default=0),
    }
    missing = [
        name
        for name, value in values.items()
        if value is None or (isinstance(value, float) and not math.isfinite(value))
    ]
    for name in missing:
        values[name] = None
    return {
        **base,
        **values,
        "feature_status": "complete" if not missing else "partial",
        "feature_missing_count": len(missing),
        "feature_missing_fields_json": _canonical_json(missing),
    }


def _build_structures(
    admitted_observations: pa.Table, split_map: Mapping[str, tuple[str, str, str]]
) -> tuple[pa.Table, dict[str, tuple[str, str, str]]]:
    structures: dict[str, Counter[tuple[str, str]]] = defaultdict(Counter)
    for row in admitted_observations.select(
        ["structure_id", "standardized_smiles", "standard_inchi_key", "structure_valid"]
    ).to_pylist():
        if (
            not row["structure_id"]
            or not row["structure_valid"]
            or not row["standardized_smiles"]
            or not row["standard_inchi_key"]
        ):
            continue
        sid = str(row["structure_id"])
        value = (str(row["standardized_smiles"]), str(row["standard_inchi_key"]))
        structures[sid][value] += 1
    final_splits = dict(split_map)
    rows: list[dict[str, Any]] = []
    for sid in sorted(structures):
        ranked = sorted(structures[sid].items(), key=lambda item: (-item[1], item[0][0], item[0][1]))
        (smiles, inchi_key), _ = ranked[0]
        representations = [
            {"standardized_smiles": pair[0], "standard_inchi_key": pair[1], "observation_count": count}
            for pair, count in ranked
        ]
        split = final_splits.get(sid)
        if split is None:
            computed = _computed_split(smiles)
            split = (computed[0], computed[1], "computed_same_frozen_scaffold_policy")
            final_splits[sid] = split
        rows.append(_descriptor_row(sid, smiles, inchi_key, split, representations))
    return pa.Table.from_pylist(rows, schema=_STRUCTURE_SCHEMA), final_splits


def _build_observations(
    ledger_path: Path,
    scope_path: Path,
    modality_path: Path,
    split_map: Mapping[str, tuple[str, str, str]],
) -> pa.Table:
    ledger = pq.read_table(ledger_path)
    admitted_ids = set(pq.read_table(scope_path, columns=["observation_id"]).column(0).to_pylist())
    if len(admitted_ids) != len(set(admitted_ids)):
        raise HergMasterDatasetError("duplicate observation ID in wild-type scope")
    mask = pc.is_in(
        ledger["observation_id"], value_set=pa.array(sorted(admitted_ids), type=pa.large_string())
    )
    admitted = ledger.filter(mask)
    admitted = admitted.take(pc.sort_indices(admitted, sort_keys=[("observation_id", "ascending")]))
    if admitted.num_rows != len(admitted_ids):
        raise HergMasterDatasetError("wild-type scope does not bind one-to-one to the observation ledger")
    if "mutant_or_variant" in set(admitted["target_variant"].to_pylist()):
        raise HergMasterDatasetError("explicit mutant leaked into admitted observations")
    modality_rows = pq.read_table(modality_path).to_pylist()
    modality = {str(row["observation_id"]): row for row in modality_rows}
    if set(modality) != admitted_ids:
        raise HergMasterDatasetError("modality index and wild-type observation scope do not match exactly")
    derived: dict[str, list[Any]] = {field.name: [] for field in _OBSERVATION_DERIVED_SCHEMA}
    selected = admitted.select(list(_OBSERVATION_REQUIRED)).to_pylist()
    for row in selected:
        observation_id = str(row["observation_id"])
        method = modality[observation_id]
        potency = _pic50_standardization(row)
        sid = str(row["structure_id"]) if row["structure_id"] else ""
        split = split_map.get(sid)
        model_eligible = bool(row["structure_valid"] and sid and split)
        values = {
            "master_dataset_id": DATASET_ID,
            "wild_type_evidence_scope": str(method["wild_type_evidence_scope"]),
            "measurement_modality": str(method["measurement_modality"]),
            "method_detail": str(method["method_detail"]),
            "modality_confidence": str(method["modality_confidence"]),
            "automation_class": str(method["automation_class"]),
            "automation_confidence": str(method["automation_confidence"]),
            "dose_design": str(method["dose_design"]),
            "dose_design_confidence": str(method["dose_design_confidence"]),
            "endpoint_class": _endpoint_class(row["native_endpoint"]),
            **potency,
            "structure_model_eligible": model_eligible,
            "model_split": split[0] if model_eligible and split else None,
            "scaffold_group_id": split[1] if model_eligible and split else None,
        }
        for name in derived:
            derived[name].append(values[name])
    for field in _OBSERVATION_DERIVED_SCHEMA:
        admitted = admitted.append_column(field, pa.array(derived[field.name], type=field.type))
    return admitted


def _counter_json(values: Iterable[object]) -> str:
    counts = Counter("<missing>" if value is None or str(value) == "" else str(value) for value in values)
    return _canonical_json(dict(sorted(counts.items())))


def _build_assay_catalog(observations: pa.Table) -> pa.Table:
    columns = [
        "source_family",
        "assay_id",
        "assay_family",
        "structure_id",
        "wild_type_evidence_scope",
        "endpoint_class",
        "native_endpoint",
        "native_relation",
        "native_unit",
        "measurement_modality",
        "automation_class",
        "dose_design",
        "native_aux_json",
    ]
    grouped: dict[tuple[str, str | None, str], list[dict[str, Any]]] = defaultdict(list)
    for row in observations.select(columns).to_pylist():
        key = (
            str(row["source_family"]),
            str(row["assay_id"]) if row["assay_id"] else None,
            str(row["assay_family"]),
        )
        grouped[key].append(row)
    output: list[dict[str, Any]] = []
    for key in sorted(grouped, key=lambda item: tuple("" if part is None else part for part in item)):
        rows = grouped[key]
        descriptions: set[str] = set()
        for row in rows:
            try:
                aux = json.loads(str(row["native_aux_json"]))
            except (TypeError, json.JSONDecodeError):
                aux = {}
            description = aux.get("assay_description") or aux.get("assay_name")
            if description:
                descriptions.add(str(description).strip())
        output.append(
            {
                "assay_catalog_id": _stable_id("HASSAY", *key),
                "source_family": key[0],
                "assay_id": key[1],
                "assay_family": key[2],
                "observation_count": len(rows),
                "unique_structure_count": len(
                    {str(row["structure_id"]) for row in rows if row["structure_id"]}
                ),
                "target_scopes_json": _counter_json(row["wild_type_evidence_scope"] for row in rows),
                "endpoint_classes_json": _counter_json(row["endpoint_class"] for row in rows),
                "native_endpoints_json": _counter_json(row["native_endpoint"] for row in rows),
                "native_relations_json": _counter_json(row["native_relation"] for row in rows),
                "native_units_json": _counter_json(row["native_unit"] for row in rows),
                "measurement_modalities_json": _counter_json(row["measurement_modality"] for row in rows),
                "automation_classes_json": _counter_json(row["automation_class"] for row in rows),
                "dose_designs_json": _counter_json(row["dose_design"] for row in rows),
                "assay_descriptions_json": _canonical_json(sorted(descriptions)),
                "catalog_status": "reported_assay" if key[1] else "source_level_unresolved_assay",
            }
        )
    return pa.Table.from_pylist(output, schema=_ASSAY_SCHEMA)


def _feature_semantic_group(name: str) -> str:
    if name in {"molecular_weight", "exact_molecular_weight", "heavy_atom_molecular_weight"}:
        return "mass_and_size"
    if name in {"mol_logp", "molar_refractivity", "labute_accessible_surface_area"}:
        return "lipophilicity_polarizability_and_surface"
    if name in {"topological_polar_surface_area", "hydrogen_bond_donors", "hydrogen_bond_acceptors"}:
        return "polarity_and_hydrogen_bonding"
    if "ring" in name or name in {"rotatable_bonds", "fraction_csp3", "stereocenter_count"}:
        return "topology_shape_and_flexibility"
    if name in {"formal_charge"}:
        return "formal_ionization_proxy_not_pKa"
    return "elemental_composition"


def _build_feature_summary(structures: pa.Table) -> pa.Table:
    output: list[dict[str, Any]] = []
    for name in FEATURE_COLUMNS:
        array = structures[name]
        nonmissing = len(array) - array.null_count
        quantiles = pc.quantile(array, q=[0.05, 0.5, 0.95], interpolation="linear").to_pylist()
        output.append(
            {
                "feature_name": name,
                "semantic_group": _feature_semantic_group(name),
                "value_type": str(array.type),
                "nonmissing_count": nonmissing,
                "missing_count": array.null_count,
                "mean": _finite(pc.mean(array).as_py()),
                "sample_standard_deviation": _finite(pc.stddev(array, ddof=1).as_py()),
                "minimum": _finite(pc.min(array).as_py()),
                "percentile_05": _finite(quantiles[0]),
                "median": _finite(quantiles[1]),
                "percentile_95": _finite(quantiles[2]),
                "maximum": _finite(pc.max(array).as_py()),
                "training_feature_eligible": True,
            }
        )
    return pa.Table.from_pylist(output, schema=_FEATURE_SUMMARY_SCHEMA)


def _build_method_summary(observations: pa.Table) -> pa.Table:
    columns = [
        "wild_type_evidence_scope",
        "measurement_modality",
        "automation_class",
        "dose_design",
        "endpoint_class",
        "structure_id",
        "assay_id",
        "endpoint_standardization_status",
    ]
    grouped: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    for row in observations.select(columns).to_pylist():
        key = (
            str(row["wild_type_evidence_scope"]),
            str(row["measurement_modality"]),
            str(row["automation_class"]),
            str(row["dose_design"]),
            str(row["endpoint_class"]),
        )
        item = grouped.setdefault(
            key,
            {"count": 0, "structures": set(), "assays": set(), "statuses": Counter()},
        )
        item["count"] += 1
        if row["structure_id"]:
            item["structures"].add(str(row["structure_id"]))
        if row["assay_id"]:
            item["assays"].add(str(row["assay_id"]))
        item["statuses"][str(row["endpoint_standardization_status"])] += 1
    output = []
    for key in sorted(grouped):
        item = grouped[key]
        statuses = item["statuses"]
        output.append(
            {
                "wild_type_evidence_scope": key[0],
                "measurement_modality": key[1],
                "automation_class": key[2],
                "dose_design": key[3],
                "endpoint_class": key[4],
                "observation_count": item["count"],
                "unique_structure_count": len(item["structures"]),
                "unique_assay_count": len(item["assays"]),
                "exact_pic50_count": statuses["exact_standardized"],
                "censored_pic50_count": statuses["censored_standardized"],
                "approximate_pic50_count": statuses["approximate_standardized"],
                "unstandardized_count": statuses["not_standardized"],
            }
        )
    return pa.Table.from_pylist(output, schema=_METHOD_SUMMARY_SCHEMA)


def _pattern_evidence(
    texts: Sequence[str], patterns: Mapping[str, re.Pattern[str]]
) -> tuple[list[str], list[dict[str, Any]]]:
    normalized: set[str] = set()
    evidence: list[dict[str, Any]] = []
    for text_index, text in enumerate(texts):
        for name, pattern in patterns.items():
            for match in pattern.finditer(text):
                normalized.add(name)
                evidence.append(
                    {"normalized": name, "matched_text": match.group(0), "protocol_text_index": text_index}
                )
    evidence.sort(
        key=lambda row: (str(row["normalized"]), int(row["protocol_text_index"]), str(row["matched_text"]))
    )
    return sorted(normalized), evidence


def _numeric_protocol_evidence(
    texts: Sequence[str], pattern: re.Pattern[str], *, kind: str
) -> tuple[list[float], list[dict[str, Any]]]:
    values: set[float] = set()
    evidence: list[dict[str, Any]] = []
    time_scale = {
        "millisecond": 0.001,
        "milliseconds": 0.001,
        "msec": 0.001,
        "second": 1.0,
        "seconds": 1.0,
        "sec": 1.0,
        "secs": 1.0,
        "minute": 60.0,
        "minutes": 60.0,
        "min": 60.0,
        "mins": 60.0,
        "hour": 3600.0,
        "hours": 3600.0,
        "hr": 3600.0,
        "hrs": 3600.0,
    }
    for text_index, text in enumerate(texts):
        for match in pattern.finditer(text):
            value = float(match.group(1))
            if kind == "time_seconds":
                value *= time_scale[match.group(2).casefold()]
            elif kind == "temperature_celsius" and match.group(2).casefold() == "k":
                value -= 273.15
            if kind == "voltage_mv" and not -500.0 <= value <= 500.0:
                continue
            if kind == "temperature_celsius" and not -100.0 <= value <= 100.0:
                continue
            value = round(value, 8)
            values.add(value)
            evidence.append(
                {
                    "normalized_value": value,
                    "normalized_unit": {
                        "voltage_mv": "mV",
                        "temperature_celsius": "degC",
                        "time_seconds": "s",
                    }[kind],
                    "matched_text": match.group(0),
                    "protocol_text_index": text_index,
                }
            )
    evidence.sort(
        key=lambda row: (
            float(row["normalized_value"]),
            int(row["protocol_text_index"]),
            str(row["matched_text"]),
        )
    )
    return sorted(values), evidence


def _build_protocol_index(assays: pa.Table) -> pa.Table:
    output: list[dict[str, Any]] = []
    for row in assays.to_pylist():
        try:
            raw_texts = json.loads(str(row["assay_descriptions_json"]))
        except json.JSONDecodeError as error:
            raise HergMasterDatasetError("assay catalog contains invalid description JSON") from error
        if not isinstance(raw_texts, list):
            raise HergMasterDatasetError("assay description JSON is not a list")
        raw_texts_clean = sorted({str(value).strip() for value in raw_texts if str(value).strip()})
        source_contracts = list(_SOURCE_PROTOCOL_CONTRACTS.get(str(row["source_family"]), []))
        texts = [*raw_texts_clean, *source_contracts]
        hosts, host_evidence = _pattern_evidence(texts, _HOST_PATTERNS)
        configurations, configuration_evidence = _pattern_evidence(texts, _RECORDING_PATTERNS)
        platforms, platform_evidence = _pattern_evidence(texts, _PLATFORM_PATTERNS)
        voltage, voltage_evidence = _numeric_protocol_evidence(texts, _VOLTAGE_PATTERN, kind="voltage_mv")
        temperature, temperature_evidence = _numeric_protocol_evidence(
            texts, _TEMPERATURE_PATTERN, kind="temperature_celsius"
        )
        times, time_evidence = _numeric_protocol_evidence(texts, _TIME_PATTERN, kind="time_seconds")
        temperature_terms: list[str] = []
        for text in texts:
            if _ROOM_TEMPERATURE_PATTERN.search(text):
                temperature_terms.append("room_or_ambient_temperature_reported_without_numeric_conversion")
                break
        manual_evidence = [
            {"matched_text": match.group(0), "protocol_text_index": index}
            for index, text in enumerate(texts)
            for match in _MANUAL_PATTERN.finditer(text)
        ]
        resolved = {
            "host_system": bool(hosts),
            "voltage": bool(voltage),
            "temperature": bool(temperature or temperature_terms),
            "time_or_incubation": bool(times),
            "recording_configuration": bool(configurations),
            "named_platform": bool(platforms),
        }
        output.append(
            {
                "assay_catalog_id": str(row["assay_catalog_id"]),
                "source_family": str(row["source_family"]),
                "assay_id": row["assay_id"],
                "assay_family": str(row["assay_family"]),
                "observation_count": int(row["observation_count"]),
                "raw_protocol_text_json": _canonical_json(raw_texts_clean),
                "source_contract_evidence_json": _canonical_json(source_contracts),
                "normalization_evidence_text_json": _canonical_json(texts),
                "host_systems_json": _canonical_json(hosts),
                "host_system_evidence_json": _canonical_json(host_evidence),
                "host_system_confidence": "high_explicit_text" if hosts else "unresolved",
                "voltage_values_mv_json": _canonical_json(voltage),
                "voltage_evidence_json": _canonical_json(voltage_evidence),
                "voltage_confidence": "high_explicit_value_and_unit" if voltage else "unresolved",
                "temperature_values_celsius_json": _canonical_json(temperature),
                "temperature_condition_terms_json": _canonical_json(temperature_terms),
                "temperature_evidence_json": _canonical_json(temperature_evidence),
                "temperature_confidence": "high_explicit_value_and_unit"
                if temperature
                else ("medium_explicit_non_numeric_term" if temperature_terms else "unresolved"),
                "time_values_seconds_json": _canonical_json(times),
                "time_evidence_json": _canonical_json(time_evidence),
                "time_confidence": "high_explicit_value_and_unit" if times else "unresolved",
                "recording_configurations_json": _canonical_json(configurations),
                "recording_configuration_evidence_json": _canonical_json(configuration_evidence),
                "recording_configuration_confidence": "high_explicit_text"
                if configurations
                else "unresolved",
                "named_platforms_json": _canonical_json(platforms),
                "platform_evidence_json": _canonical_json(platform_evidence),
                "platform_confidence": "high_named_platform" if platforms else "unresolved",
                "manual_automation_evidence_json": _canonical_json(manual_evidence),
                "source_automation_classes_json": str(row["automation_classes_json"]),
                "unresolved_fields_json": _canonical_json(
                    sorted(name for name, status in resolved.items() if not status)
                ),
                "protocol_completeness_score": sum(resolved.values()),
                "normalization_policy": "explicit_text_only_v1; no absent protocol attribute is inferred",
            }
        )
    output.sort(key=lambda row: str(row["assay_catalog_id"]))
    return pa.Table.from_pylist(output, schema=_PROTOCOL_SCHEMA)


def _bounds_from_target(
    relation: object, value: object
) -> tuple[str | None, float | None, float | None, float | None]:
    number = _finite(value)
    rel = str(relation) if relation in {"=", "~", ">", ">=", "<", "<="} else None
    if number is None:
        return rel, None, None, None
    if rel == "=":
        return rel, number, number, number
    if rel in {">", ">="}:
        return rel, None, number, None
    if rel in {"<", "<="}:
        return rel, None, None, number
    return rel, number, None, None


def _build_task_membership(task_paths: Sequence[Path], qt_path: Path) -> pa.Table:
    output: list[dict[str, Any]] = []
    for path in task_paths:
        for row in pq.read_table(path).to_pylist():
            relation, point, lower, upper = _bounds_from_target(row["target_relation"], row["target_pic50"])
            record_id = str(row["record_id"])
            output.append(
                {
                    "membership_id": _stable_id("HTASK", path.name, record_id),
                    "task_id": str(row["task_id"]),
                    "quality_level": str(row["quality_level"]),
                    "source_artifact": path.name,
                    "record_id": record_id,
                    "observation_id": row["observation_id"],
                    "structure_id": row["structure_id"],
                    "target_scope": str(row["target_scope"]),
                    "source_family": str(row["source_family"]),
                    "native_endpoint": str(row["native_endpoint"]),
                    "measurement_technology": str(row["measurement_technology"]),
                    "model_split": row["model_split"],
                    "scaffold_group_id": row["scaffold_group_id"],
                    "eligible": bool(row["eligible"]),
                    "direct_herg_label": bool(row["direct_herg_label"]),
                    "use_as_training_label": bool(row["use_as_training_label"]),
                    "clinical_context_only": bool(row["clinical_context_only"]),
                    "target_relation_pic50": relation,
                    "target_pic50_point": point,
                    "target_pic50_lower_bound": lower,
                    "target_pic50_upper_bound": upper,
                    "target_class": row["target_class"],
                    "eligibility_reason": str(row["eligibility_reason"]),
                    "exclusion_reason": row["exclusion_reason"],
                    "quality_flags": str(row["quality_flags"]),
                }
            )
    for row in pq.read_table(qt_path).to_pylist():
        record_id = str(row["candidate_id"])
        output.append(
            {
                "membership_id": _stable_id("HTASK", qt_path.name, record_id),
                "task_id": str(row["task_id"]),
                "quality_level": "C1_QT_QTc_context",
                "source_artifact": qt_path.name,
                "record_id": record_id,
                "observation_id": None,
                "structure_id": row["structure_id"],
                "target_scope": "clinical_context_not_target_variant",
                "source_family": "ClinicalTrials.gov_posted_results",
                "native_endpoint": str(row["candidate_classification"]),
                "measurement_technology": "clinical_ECG_QT_QTc_context",
                "model_split": row["model_split"],
                "scaffold_group_id": row["scaffold_group_id"],
                "eligible": bool(row["context_eligible"]),
                "direct_herg_label": False,
                "use_as_training_label": False,
                "clinical_context_only": True,
                "target_relation_pic50": None,
                "target_pic50_point": None,
                "target_pic50_lower_bound": None,
                "target_pic50_upper_bound": None,
                "target_class": None,
                "eligibility_reason": str(row["context_semantics"]),
                "exclusion_reason": None,
                "quality_flags": _canonical_json(
                    {"heldout_evaluation_eligible": bool(row["heldout_evaluation_eligible"])}
                ),
            }
        )
    output.sort(key=lambda row: str(row["membership_id"]))
    return pa.Table.from_pylist(output, schema=_TASK_SCHEMA)


def _build_clinical_context(c0_path: Path, c1_path: Path) -> pa.Table:
    output: list[dict[str, Any]] = []
    for row in pq.read_table(c0_path).to_pylist():
        output.append(
            {
                "clinical_context_id": _stable_id("HCTX", "C0", row["record_id"]),
                "context_class": "clinical_development_or_regulatory_annotation",
                "structure_id": str(row["structure_id"]),
                "nct_id": None,
                "endpoint_candidate_id": None,
                "model_split": str(row["model_split"]),
                "scaffold_group_id": str(row["scaffold_group_id"]),
                "context_eligible": bool(row["eligible"]),
                "heldout_evaluation_eligible": False,
                "direct_herg_label": False,
                "use_as_training_label": False,
                "title_or_term": None,
                "description_or_organ_system": None,
                "unit_of_measure": None,
                "time_frame": None,
                "native_context_json": _canonical_json(row),
            }
        )
    for row in pq.read_table(c1_path).to_pylist():
        output.append(
            {
                "clinical_context_id": _stable_id("HCTX", "C1", row["candidate_id"]),
                "context_class": "human_QT_QTc_reported_result_context",
                "structure_id": str(row["structure_id"]),
                "nct_id": str(row["nct_id"]),
                "endpoint_candidate_id": str(row["endpoint_candidate_id"]),
                "model_split": str(row["model_split"]),
                "scaffold_group_id": str(row["scaffold_group_id"]),
                "context_eligible": bool(row["context_eligible"]),
                "heldout_evaluation_eligible": bool(row["heldout_evaluation_eligible"]),
                "direct_herg_label": False,
                "use_as_training_label": False,
                "title_or_term": row["title_or_term"],
                "description_or_organ_system": row["description_or_organ_system"],
                "unit_of_measure": row["unit_of_measure"],
                "time_frame": row["time_frame"],
                "native_context_json": _canonical_json(row),
            }
        )
    output.sort(key=lambda row: str(row["clinical_context_id"]))
    return pa.Table.from_pylist(output, schema=_CLINICAL_SCHEMA)


def _build_exclusions(scope_exclusions: Path, task_exclusions: Path, observation_ledger: Path) -> pa.Table:
    output: list[dict[str, Any]] = []
    source_records = {
        str(row["observation_id"]): str(row["source_record_id"])
        for row in pq.read_table(
            observation_ledger, columns=["observation_id", "source_record_id"]
        ).to_pylist()
    }
    for row in pq.read_table(scope_exclusions).to_pylist():
        source_id = str(row["observation_id"])
        output.append(
            {
                "master_exclusion_id": _stable_id("HEXCL", "scope", source_id),
                "exclusion_scope": "entire_wild_type_master",
                "task_id": None,
                "source_family": str(row["source_family"]),
                "source_record_id": source_records[source_id],
                "observation_id": source_id,
                "structure_id": row["structure_id"],
                "target_scope": str(row["target_variant_original"]),
                "exclusion_reason": str(row["exclusion_reason"]),
                "exclusion_detail": "explicit variant evidence; retained only in this quarantine ledger",
                "native_exclusion_context_json": _canonical_json(row),
                "source_artifact": scope_exclusions.name,
            }
        )
    for row in pq.read_table(task_exclusions).to_pylist():
        if str(row["exclusion_reason"]) == "explicit_mutant_or_variant_target":
            continue
        output.append(
            {
                "master_exclusion_id": _stable_id("HEXCL", row["task_id"], row["source_record_id"]),
                "exclusion_scope": "specific_modeling_task",
                "task_id": str(row["task_id"]),
                "source_family": str(row["source_family"]),
                "source_record_id": str(row["source_record_id"]),
                "observation_id": row["observation_id"],
                "structure_id": row["structure_id"],
                "target_scope": str(row["target_scope"]),
                "exclusion_reason": str(row["exclusion_reason"]),
                "exclusion_detail": str(row["exclusion_detail"]),
                "native_exclusion_context_json": _canonical_json(row),
                "source_artifact": task_exclusions.name,
            }
        )
    output.sort(key=lambda row: str(row["master_exclusion_id"]))
    return pa.Table.from_pylist(output, schema=_EXCLUSION_SCHEMA)


def _build_evidence(
    observations: pa.Table, tasks: pa.Table, clinical: pa.Table, structures: pa.Table
) -> pa.Table:
    evidence: dict[str, dict[str, Any]] = {}
    for sid in structures["structure_id"].to_pylist():
        evidence[str(sid)] = {
            "scopes": Counter(),
            "sources": set(),
            "assays": set(),
            "endpoints": set(),
            "modalities": set(),
            "tasks": set(),
            "observations": 0,
            "clinical": False,
            "qt": False,
        }
    cols = [
        "structure_id",
        "wild_type_evidence_scope",
        "source_family",
        "assay_id",
        "endpoint_class",
        "measurement_modality",
    ]
    for row in observations.select(cols).to_pylist():
        if not row["structure_id"] or str(row["structure_id"]) not in evidence:
            continue
        item = evidence[str(row["structure_id"])]
        item["observations"] += 1
        item["scopes"][str(row["wild_type_evidence_scope"])] += 1
        item["sources"].add(str(row["source_family"]))
        if row["assay_id"]:
            item["assays"].add(str(row["assay_id"]))
        item["endpoints"].add(str(row["endpoint_class"]))
        item["modalities"].add(str(row["measurement_modality"]))
    for row in tasks.select(["structure_id", "task_id", "eligible"]).to_pylist():
        if row["structure_id"] and row["eligible"] and str(row["structure_id"]) in evidence:
            evidence[str(row["structure_id"])]["tasks"].add(str(row["task_id"]))
    for row in clinical.select(["structure_id", "context_class", "context_eligible"]).to_pylist():
        if row["context_eligible"] and str(row["structure_id"]) in evidence:
            item = evidence[str(row["structure_id"])]
            item["clinical"] = True
            item["qt"] = item["qt"] or str(row["context_class"]).startswith("human_QT")
    output = []
    for sid in sorted(evidence):
        item = evidence[sid]
        output.append(
            {
                "structure_id": sid,
                "observation_count": item["observations"],
                "confirmed_wild_type_observation_count": item["scopes"]["confirmed_wild_type"],
                "wild_type_or_unspecified_observation_count": item["scopes"]["wild_type_or_unspecified"],
                "source_families_json": _canonical_json(sorted(item["sources"])),
                "assay_count": len(item["assays"]),
                "endpoint_classes_json": _canonical_json(sorted(item["endpoints"])),
                "measurement_modalities_json": _canonical_json(sorted(item["modalities"])),
                "task_ids_json": _canonical_json(sorted(item["tasks"])),
                "has_clinical_development_context": bool(item["clinical"]),
                "has_qt_qtc_context": bool(item["qt"]),
                "model_feature_eligible": False,
                "feature_exclusion_reason": "evidence density and task membership can encode labels; join only for audit/stratification",
            }
        )
    return pa.Table.from_pylist(output, schema=_EVIDENCE_SCHEMA)


def _write_report(path: Path, manifest: Mapping[str, Any]) -> None:
    counts = manifest["counts"]
    lines = [
        "# Comprehensive wild-type hERG master dataset (v1.3)",
        "",
        "This release is the paper-facing join layer. It does not replace or mutate any native source artifact.",
        "",
        "## Scale",
        "",
        f"- {counts['admitted_observations']:,} admitted wild-type-scope observations.",
        f"- {counts['structures']:,} standardized molecular structures with CPU-feasible RDKit 2D features.",
        f"- {counts['structure_representation_conflicts']:,} structure entities retain multiple reported standardized representations for sensitivity analysis.",
        f"- {counts['assays']:,} source-aware assay catalog entries.",
        f"- {counts['task_memberships']:,} quality/clinical task memberships.",
        f"- {counts['clinical_context_rows']:,} clinical-development or QT/QTc context rows.",
        f"- {counts['master_exclusions']:,} explicit scope or task-specific exclusions retained in quarantine.",
        "",
        "## Established design advantages over existing hERG model datasets",
        "",
        "- Explicit mutants are excluded; confirmed WT and WT-or-unspecified evidence are never conflated.",
        "- Native endpoint, relation, value, unit, assay, source row, auxiliary metadata, and lineage remain recoverable.",
        "- Fixed-dose activity, exact/censored pIC50, other endpoints, measurement modality, and clinical QT/QTc are separate axes.",
        "- Manual/automated evidence, assay technology, source lineage, scaffold partition, and quality-task membership support transport audits that most pooled benchmarks omit.",
        "- Natural censoring is represented with valid pIC50 bounds; no threshold value is fabricated.",
        "- Clinical QT/QTc remains label-disabled, preventing a downstream exposure-dependent phenotype from leaking into a direct hERG target.",
        "These are established data-design superiorities, not a claim of predictive superiority; model superiority still requires locked external and prospective comparisons.",
        "",
        "## Measurement coverage",
        "",
    ]
    for name, count in sorted(
        counts["measurement_modality_counts"].items(), key=lambda item: (-item[1], item[0])
    ):
        lines.append(f"- {name}: {count:,} observations.")
    lines.extend(
        [
            "",
            "## Endpoint coverage",
            "",
        ]
    )
    for name, count in sorted(counts["endpoint_class_counts"].items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"- {name}: {count:,} observations.")
    lines.extend(
        [
            "",
            "The modality and endpoint totals answer different questions: seven rows belong to curated clinical-QT phenotype assays, while only four natively report a `QT interval` endpoint. The other three retain their source-reported EC10 endpoint class and are still clinical-context-only, never direct hERG labels.",
            "",
            "## Standardized potency status",
            "",
        ]
    )
    for name, count in sorted(
        counts["potency_standardization_counts"].items(), key=lambda item: (-item[1], item[0])
    ):
        lines.append(f"- {name}: {count:,} observations.")
    lines.extend(
        [
            "",
            "## Structure features and intentional omissions",
            "",
            "The structure table contains only deterministic 2D physicochemical descriptors plus identifiers and frozen split metadata. `fundamental_feature_summary.parquet` reports coverage and empirical ranges. The manifest lists the only feature-eligible columns. Evidence-density summaries are isolated and explicitly label-ineligible. No pKa, protonation ensemble, docking pose, channel state, 3D conformer, membrane partition coefficient, or binding free energy was invented.",
            "",
            "## Analysis-ready audit tables",
            "",
            "`method_endpoint_summary.parquet` cross-tabulates target certainty, measurement technology, automation, dose design, endpoint semantics, structure coverage, and potency standardization. It supports prespecified method-impact analyses without altering labels. `structure_evidence_summary.parquet` supports coverage audits but is explicitly prohibited as a molecular feature because measurement density can encode outcome and selection bias.",
            "",
            "`assay_protocol_index.parquet` preserves raw assay text and normalizes only explicit host system, voltage, temperature, time, recording configuration, named platform, and manual-operation evidence. Every normalized field carries matched-text evidence and confidence; missing protocol details remain unresolved.",
            "",
            "## Use",
            "",
            "Use `observation_master.parquet` for source-faithful endpoint and method analysis, `structure_master.parquet` for leakage-controlled molecular inputs, `task_membership.parquet` for labels, and `clinical_context_master.parquet` only for clinical stratification or held-out translation analysis.",
            "",
            "No model was trained by this build.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_herg_master_dataset(
    *,
    hierarchy_root: str | os.PathLike[str],
    wildtype_scope_root: str | os.PathLike[str],
    modality_qt_root: str | os.PathLike[str],
    quality_tasks_root: str | os.PathLike[str],
    model_ready_root: str | os.PathLike[str],
    output_root: str | os.PathLike[str],
    report_root: str | os.PathLike[str],
) -> dict[str, Any]:
    """Build the immutable v1.3 join layer and validate it before publication."""

    hierarchy = Path(hierarchy_root).resolve()
    scope = Path(wildtype_scope_root).resolve()
    modality = Path(modality_qt_root).resolve()
    tasks = Path(quality_tasks_root).resolve()
    split = Path(model_ready_root).resolve()
    output = Path(output_root).resolve()
    report = Path(report_root).resolve()
    if output.exists() or report.exists():
        raise HergMasterDatasetError("output_root and report_root must not already exist")
    paths = {
        "observation_ledger": _checked_parquet(
            hierarchy / "observation_ledger.parquet", "observation ledger", _OBSERVATION_REQUIRED
        ),
        "wildtype_index": _checked_parquet(
            scope / "wildtype_observation_index.parquet", "wild-type index", ["observation_id"]
        ),
        "scope_exclusions": _checked_parquet(
            scope / "explicit_mutant_exclusions.parquet", "scope exclusions"
        ),
        "modality_index": _checked_parquet(
            modality / "herg_measurement_modality_index.parquet", "modality index", ["observation_id"]
        ),
        "qt_phenotype": _checked_parquet(
            modality / "qt_clinical_phenotype_index.parquet", "QT phenotype index"
        ),
        "frozen_split": _checked_parquet(
            split / "structure_consensus_binary_scaffold_split.parquet",
            "frozen scaffold split",
            ["structure_id", "split", "scaffold_group_id"],
        ),
        "task_exclusions": _checked_parquet(tasks / "exclusion_ledger.parquet", "task exclusions"),
    }
    task_paths = [_checked_parquet(tasks / name, f"quality task {name}") for name in _TASK_FILES]
    qt_task = _checked_parquet(tasks / _QT_TASK_FILE, "QT context task")
    all_inputs = [*paths.values(), *task_paths, qt_task]
    output.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging.", dir=output.parent))
    report_staging = Path(tempfile.mkdtemp(prefix=f".{report.name}.staging.", dir=report.parent))
    try:
        split_map = _load_split_map(paths["frozen_split"], task_paths)
        ledger = pq.read_table(paths["observation_ledger"])
        scope_ids = set(
            pq.read_table(paths["wildtype_index"], columns=["observation_id"])["observation_id"].to_pylist()
        )
        admitted_raw = ledger.filter(
            pc.is_in(ledger["observation_id"], value_set=pa.array(sorted(scope_ids), type=pa.large_string()))
        )
        structures, final_splits = _build_structures(admitted_raw, split_map)
        observations = _build_observations(
            paths["observation_ledger"], paths["wildtype_index"], paths["modality_index"], final_splits
        )
        task_membership = _build_task_membership(task_paths, qt_task)
        clinical = _build_clinical_context(task_paths[3], qt_task)
        assays = _build_assay_catalog(observations)
        protocols = _build_protocol_index(assays)
        exclusions = _build_exclusions(
            paths["scope_exclusions"], paths["task_exclusions"], paths["observation_ledger"]
        )
        evidence = _build_evidence(observations, task_membership, clinical, structures)
        feature_summary = _build_feature_summary(structures)
        method_summary = _build_method_summary(observations)
        tables = {
            OBSERVATION_OUTPUT: observations,
            STRUCTURE_OUTPUT: structures,
            EVIDENCE_OUTPUT: evidence,
            FEATURE_SUMMARY_OUTPUT: feature_summary,
            ASSAY_OUTPUT: assays,
            PROTOCOL_OUTPUT: protocols,
            METHOD_SUMMARY_OUTPUT: method_summary,
            TASK_OUTPUT: task_membership,
            CLINICAL_OUTPUT: clinical,
            EXCLUSION_OUTPUT: exclusions,
        }
        artifacts = {name: _write(staging / name, table) for name, table in tables.items()}
        endpoint_counts = Counter(str(value) for value in observations["endpoint_class"].to_pylist())
        scope_counts = Counter(str(value) for value in observations["wild_type_evidence_scope"].to_pylist())
        potency_counts = Counter(
            str(value) for value in observations["endpoint_standardization_status"].to_pylist()
        )
        modality_counts = Counter(str(value) for value in observations["measurement_modality"].to_pylist())
        automation_counts = Counter(str(value) for value in observations["automation_class"].to_pylist())
        protocol_host_observations: Counter[str] = Counter()
        protocol_platform_observations: Counter[str] = Counter()
        for row in protocols.select(
            ["host_systems_json", "named_platforms_json", "observation_count"]
        ).to_pylist():
            for host in json.loads(str(row["host_systems_json"])):
                protocol_host_observations[str(host)] += int(row["observation_count"])
            for platform in json.loads(str(row["named_platforms_json"])):
                protocol_platform_observations[str(platform)] += int(row["observation_count"])
        task_rows = task_membership.select(
            ["task_id", "eligible", "structure_id", "use_as_training_label"]
        ).to_pylist()
        task_counts: dict[str, dict[str, int]] = {}
        for task_id in sorted({str(row["task_id"]) for row in task_rows}):
            selected = [row for row in task_rows if str(row["task_id"]) == task_id]
            task_counts[task_id] = {
                "rows": len(selected),
                "eligible_rows": sum(bool(row["eligible"]) for row in selected),
                "training_label_rows": sum(bool(row["use_as_training_label"]) for row in selected),
                "eligible_unique_structures": len(
                    {str(row["structure_id"]) for row in selected if row["eligible"] and row["structure_id"]}
                ),
            }
        manifest_body: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "dataset_id": DATASET_ID,
            "inputs": [_input(path.stem, path) for path in all_inputs],
            "policies": {
                "target": "human wild-type KCNH2/hERG",
                "confirmed_wild_type": "retained as confirmed",
                "wild_type_or_unspecified": "retained with uncertainty; never upgraded",
                "explicit_mutant_or_variant": "excluded and quarantined",
                "native_field_policy": "all source-ledger columns preserved unchanged",
                "endpoint_policy": "native endpoints remain separate; only reported pIC50 and IC50 with recognized concentration units receive pIC50 interpretation",
                "censoring_policy": "monotonic relation inversion creates one-sided pIC50 bounds; approximate values do not receive invented bounds",
                "clinical_policy": "QT/QTc and development annotations are context only and cannot supply direct hERG labels",
                "descriptor_policy": "deterministic RDKit 2D physicochemical features; missing values remain null",
                "not_computed": [
                    "pKa",
                    "protonation microstates",
                    "3D conformers",
                    "docking",
                    "channel-state physics",
                    "binding free energy",
                    "membrane partitioning",
                ],
            },
            "model_feature_contract": {
                "feature_eligible_artifact": STRUCTURE_OUTPUT,
                "eligible_columns": list(FEATURE_COLUMNS),
                "identifier_or_partition_columns_not_features": [
                    "structure_id",
                    "standardized_smiles",
                    "standard_inchi_key",
                    "model_split",
                    "scaffold_group_id",
                    "split_source",
                    "structure_representation_count",
                    "structure_representations_json",
                    "structure_resolution_policy",
                    "molecular_formula",
                    "feature_status",
                    "feature_missing_count",
                    "feature_missing_fields_json",
                ],
                "explicitly_ineligible_artifact": EVIDENCE_OUTPUT,
                "labels_join_only_after_partitioning": True,
            },
            "counts": {
                "source_observations": ledger.num_rows,
                "admitted_observations": observations.num_rows,
                "explicit_mutant_exclusions": pq.ParquetFile(paths["scope_exclusions"]).metadata.num_rows,
                "structures": structures.num_rows,
                "assays": assays.num_rows,
                "assay_protocol_rows": protocols.num_rows,
                "method_endpoint_summary_rows": method_summary.num_rows,
                "fundamental_feature_count": feature_summary.num_rows,
                "task_memberships": task_membership.num_rows,
                "clinical_context_rows": clinical.num_rows,
                "master_exclusions": exclusions.num_rows,
                "scope_counts": dict(sorted(scope_counts.items())),
                "endpoint_class_counts": dict(sorted(endpoint_counts.items())),
                "measurement_modality_counts": dict(sorted(modality_counts.items())),
                "automation_class_counts": dict(sorted(automation_counts.items())),
                "protocol_host_observation_counts": dict(sorted(protocol_host_observations.items())),
                "protocol_platform_observation_counts": dict(sorted(protocol_platform_observations.items())),
                "potency_standardization_counts": dict(sorted(potency_counts.items())),
                "structure_feature_status_counts": dict(
                    sorted(Counter(str(value) for value in structures["feature_status"].to_pylist()).items())
                ),
                "structure_representation_conflicts": sum(
                    int(value) > 1 for value in structures["structure_representation_count"].to_pylist()
                ),
                "tasks": task_counts,
            },
            "artifacts": artifacts,
            "substantive_models_trained": 0,
        }
        manifest = dict(manifest_body)
        manifest["manifest_sha256"] = hashlib.sha256(_canonical_json(manifest_body).encode()).hexdigest()
        (staging / MANIFEST_NAME).write_text(_canonical_json(manifest) + "\n", encoding="utf-8")
        _write_report(report_staging / REPORT_NAME, manifest)
        os.replace(staging, output)
        os.replace(report_staging, report)
        validate_herg_master_dataset(output)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        shutil.rmtree(report_staging, ignore_errors=True)
        if output.exists():
            shutil.rmtree(output, ignore_errors=True)
        if report.exists():
            shutil.rmtree(report, ignore_errors=True)
        raise


def validate_herg_master_dataset(output_root: str | os.PathLike[str]) -> dict[str, Any]:
    """Verify artifact hashes and high-risk semantic contracts."""

    root = Path(output_root).resolve()
    manifest_path = root / MANIFEST_NAME
    if root.is_symlink() or not root.is_dir() or not manifest_path.is_file():
        raise HergMasterDatasetError(f"missing master dataset: {root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    supplied = manifest.pop("manifest_sha256", None)
    expected = hashlib.sha256(_canonical_json(manifest).encode()).hexdigest()
    manifest["manifest_sha256"] = supplied
    if supplied != expected:
        raise HergMasterDatasetError("manifest digest mismatch")
    for binding in manifest["inputs"]:
        source_path = Path(str(binding["path"]))
        if source_path.is_symlink() or not source_path.is_file():
            raise HergMasterDatasetError(f"bound input is missing or unsafe: {source_path}")
        if _sha256_file(source_path) != binding["sha256"]:
            raise HergMasterDatasetError(f"bound input hash mismatch: {source_path.name}")
        if pq.ParquetFile(source_path).metadata.num_rows != binding["rows"]:
            raise HergMasterDatasetError(f"bound input row count mismatch: {source_path.name}")
    schemas: dict[str, pa.Schema | None] = {
        OBSERVATION_OUTPUT: None,
        STRUCTURE_OUTPUT: _STRUCTURE_SCHEMA,
        EVIDENCE_OUTPUT: _EVIDENCE_SCHEMA,
        FEATURE_SUMMARY_OUTPUT: _FEATURE_SUMMARY_SCHEMA,
        ASSAY_OUTPUT: _ASSAY_SCHEMA,
        PROTOCOL_OUTPUT: _PROTOCOL_SCHEMA,
        METHOD_SUMMARY_OUTPUT: _METHOD_SUMMARY_SCHEMA,
        TASK_OUTPUT: _TASK_SCHEMA,
        CLINICAL_OUTPUT: _CLINICAL_SCHEMA,
        EXCLUSION_OUTPUT: _EXCLUSION_SCHEMA,
    }
    for name, schema in schemas.items():
        path = root / name
        metadata = manifest["artifacts"][name]
        if path.is_symlink() or not path.is_file() or _sha256_file(path) != metadata["sha256"]:
            raise HergMasterDatasetError(f"artifact hash mismatch: {name}")
        parquet = pq.ParquetFile(path)
        if parquet.metadata.num_rows != metadata["rows"]:
            raise HergMasterDatasetError(f"artifact row count mismatch: {name}")
        if schema is not None and parquet.schema_arrow != schema:
            raise HergMasterDatasetError(f"artifact schema mismatch: {name}")
        schema_digest = hashlib.sha256(parquet.schema_arrow.serialize().to_pybytes()).hexdigest()
        if schema_digest != metadata["arrow_schema_sha256"]:
            raise HergMasterDatasetError(f"artifact schema digest mismatch: {name}")
    observations = pq.read_table(root / OBSERVATION_OUTPUT)
    required = set(_OBSERVATION_REQUIRED) | set(_OBSERVATION_DERIVED_SCHEMA.names)
    if not required.issubset(observations.column_names):
        raise HergMasterDatasetError("observation master lost required native or standardized fields")
    if "mutant_or_variant" in set(observations["target_variant"].to_pylist()):
        raise HergMasterDatasetError("explicit mutant leaked into observation master")
    ids = observations["observation_id"].to_pylist()
    if ids != sorted(ids) or len(ids) != len(set(ids)):
        raise HergMasterDatasetError("observation IDs are duplicate or non-deterministically ordered")
    bounds = observations.select(["potency_pic50_lower_bound", "potency_pic50_upper_bound"]).to_pylist()
    if any(
        row["potency_pic50_lower_bound"] is not None
        and row["potency_pic50_upper_bound"] is not None
        and row["potency_pic50_lower_bound"] > row["potency_pic50_upper_bound"]
        for row in bounds
    ):
        raise HergMasterDatasetError("invalid pIC50 bounds")
    structures = pq.read_table(root / STRUCTURE_OUTPUT)
    structure_ids = set(structures["structure_id"].to_pylist())
    if len(structure_ids) != structures.num_rows:
        raise HergMasterDatasetError("duplicate structure in structure master")
    for row in structures.select(
        ["structure_representation_count", "structure_representations_json"]
    ).to_pylist():
        representations = json.loads(str(row["structure_representations_json"]))
        if (
            not isinstance(representations, list)
            or len(representations) != row["structure_representation_count"]
        ):
            raise HergMasterDatasetError("structure representation accounting mismatch")
    group_splits: dict[str, set[str]] = defaultdict(set)
    for row in structures.select(["scaffold_group_id", "model_split"]).to_pylist():
        group_splits[str(row["scaffold_group_id"])].add(str(row["model_split"]))
    if any(len(values) != 1 for values in group_splits.values()):
        raise HergMasterDatasetError("scaffold group crosses model partitions")
    clinical = pq.read_table(root / CLINICAL_OUTPUT)
    if any(clinical["direct_herg_label"].to_pylist()) or any(clinical["use_as_training_label"].to_pylist()):
        raise HergMasterDatasetError("clinical context promoted to a direct hERG label")
    tasks = pq.read_table(root / TASK_OUTPUT)
    split_by_structure = {
        str(row["structure_id"]): (str(row["model_split"]), str(row["scaffold_group_id"]))
        for row in structures.select(["structure_id", "model_split", "scaffold_group_id"]).to_pylist()
    }
    task_columns = [
        "structure_id",
        "target_scope",
        "model_split",
        "scaffold_group_id",
        "eligible",
        "clinical_context_only",
        "direct_herg_label",
        "use_as_training_label",
        "target_relation_pic50",
        "target_pic50_point",
        "target_pic50_lower_bound",
        "target_pic50_upper_bound",
        "exclusion_reason",
    ]
    for row in tasks.select(task_columns).to_pylist():
        if row["clinical_context_only"] and (row["direct_herg_label"] or row["use_as_training_label"]):
            raise HergMasterDatasetError("clinical task promoted to a direct hERG label")
        if row["target_scope"] == "mutant_or_variant":
            raise HergMasterDatasetError("mutant task membership leaked into master")
        if bool(row["eligible"]) == bool(row["exclusion_reason"]):
            raise HergMasterDatasetError("task eligibility/exclusion contradiction")
        sid = str(row["structure_id"] or "")
        if sid and sid not in structure_ids:
            raise HergMasterDatasetError("task membership references an unknown master structure")
        if sid and row["model_split"] and row["scaffold_group_id"]:
            if split_by_structure[sid] != (str(row["model_split"]), str(row["scaffold_group_id"])):
                raise HergMasterDatasetError("task membership conflicts with master structure split")
        relation = row["target_relation_pic50"]
        if relation in {"<", "<="} and (
            row["target_pic50_point"] is not None
            or row["target_pic50_lower_bound"] is not None
            or row["target_pic50_upper_bound"] is None
        ):
            raise HergMasterDatasetError("invalid upper-bounded task pIC50 target")
        if relation in {">", ">="} and (
            row["target_pic50_point"] is not None
            or row["target_pic50_lower_bound"] is None
            or row["target_pic50_upper_bound"] is not None
        ):
            raise HergMasterDatasetError("invalid lower-bounded task pIC50 target")
    exclusions = pq.read_table(root / EXCLUSION_OUTPUT)
    master_scope = exclusions.filter(pc.equal(exclusions["exclusion_scope"], "entire_wild_type_master"))
    if master_scope.num_rows != manifest["counts"]["explicit_mutant_exclusions"]:
        raise HergMasterDatasetError("explicit mutant exclusion accounting mismatch")
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hierarchy-root", type=Path)
    parser.add_argument("--wildtype-scope-root", type=Path)
    parser.add_argument("--modality-qt-root", type=Path)
    parser.add_argument("--quality-tasks-root", type=Path)
    parser.add_argument("--model-ready-root", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report-root", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.validate_only:
        validate_herg_master_dataset(args.output_root)
    else:
        required = (
            args.hierarchy_root,
            args.wildtype_scope_root,
            args.modality_qt_root,
            args.quality_tasks_root,
            args.model_ready_root,
            args.report_root,
        )
        if any(path is None for path in required):
            raise HergMasterDatasetError("all input roots and --report-root are required when building")
        build_herg_master_dataset(
            hierarchy_root=args.hierarchy_root,
            wildtype_scope_root=args.wildtype_scope_root,
            modality_qt_root=args.modality_qt_root,
            quality_tasks_root=args.quality_tasks_root,
            model_ready_root=args.model_ready_root,
            output_root=args.output_root,
            report_root=args.report_root,
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
