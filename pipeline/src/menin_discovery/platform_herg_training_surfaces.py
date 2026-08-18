"""Compile deterministic, leakage-aware hERG training surfaces from local releases.

The compiler is deliberately downstream and additive.  It does not alter the
v1.3 master or v1.5 adjudication release, does not train a model, and does not
promote review candidates to validated evidence.  It exposes two complementary
training grains:

* source-faithful observations with native endpoint, relation, unit, method,
  protocol, split, and lineage cautions; and
* one confirmed-wild-type, fixed-dose consensus label per structure.

Clinical QT/QTc context remains a separate, non-label artifact.  Approximate
relations are available only to an explicitly named sensitivity surface.
Explicit mutants are fail-closed and represented only in the exclusion audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from .platform_herg_candidate_adjudication import validate_herg_candidate_adjudication
from .platform_herg_master_dataset import validate_herg_master_dataset
from .platform_herg_tiers import validate_herg_evidence_tiers

SCHEMA_VERSION = "platform-herg-training-surfaces/1.0"
DATASET_ID = "wild_type_herg_training_surfaces_v1_6"
MANIFEST_NAME = "herg_training_surfaces_manifest.json"
REPORT_NAME = "HERG_TRAINING_SURFACES.md"

OBSERVATION_OUTPUT = "herg_training_observations.parquet"
STRUCTURE_OUTPUT = "confirmed_wt_fixed_dose_structure_labels.parquet"
VALIDATION_CANDIDATE_OUTPUT = "preclinical_validation_candidates.parquet"
CLINICAL_OUTPUT = "clinical_context_only.parquet"
REGISTRY_OUTPUT = "training_surface_registry.parquet"
STRATA_OUTPUT = "training_measurement_strata.parquet"
EXCLUSION_OUTPUT = "training_exclusion_audit.parquet"

PRIMARY_RELATIONS = frozenset({"=", "<", "<=", ">", ">="})
SENSITIVITY_RELATIONS = frozenset({"~"})
NUMERIC_SOURCE_FAMILIES = frozenset({"chembl_herg_specialized_view", "quantitative_pic50_release"})
FUNCTIONAL_MODALITIES = frozenset(
    {
        "patch_clamp_electrophysiology",
        "functional_electrophysiology",
        "functional_ion_flux",
        "functional_unspecified",
        "high_throughput_thallium_flux",
    }
)
PARTITIONS = frozenset({"train", "validation", "test"})
STRENGTH_ORDER = {"none": 0, "weak": 1, "moderate": 2, "strong": 3}

SURFACE_REPORTED = "OBS_T0_REPORTED_PUBLIC_ALL"
SURFACE_CLEAN = "OBS_CLEAN_PRIMARY_LABELS"
SURFACE_CONFIRMED_WT = "OBS_CONFIRMED_WT_FIXED_DOSE_SUPPORT"
SURFACE_NUMERIC = "OBS_PRECLINICAL_NATIVE_NUMERIC_PRIMARY"
SURFACE_PIC50 = "OBS_PRECLINICAL_STANDARDIZED_PIC50_PRIMARY"
SURFACE_FUNCTIONAL_NUMERIC = "OBS_FUNCTIONAL_HOW_MEASURED_NATIVE_NUMERIC"
SURFACE_FUNCTIONAL_PIC50 = "OBS_FUNCTIONAL_HOW_MEASURED_PIC50"
SURFACE_APPROXIMATE = "OBS_APPROXIMATE_RELATION_SENSITIVITY_ONLY"
SURFACE_CURATED_REVIEW = "OBS_CURATED_FUNCTIONAL_T1_REVIEW_CANDIDATE"
SURFACE_STRUCTURE_BINARY = "STRUCT_CONFIRMED_WT_FIXED_DOSE_CONSENSUS"
SURFACE_T1_REVIEW = "STRUCT_CROSS_LINEAGE_T1_REVIEW_CANDIDATE"
SURFACE_FORMAL_T1 = "STRUCT_FORMAL_T1_VALIDATED"
SURFACE_CLINICAL = "CONTEXT_CLINICAL_DEVELOPMENT_AND_QT"

OBSERVATION_SURFACES = (
    SURFACE_REPORTED,
    SURFACE_CLEAN,
    SURFACE_CONFIRMED_WT,
    SURFACE_NUMERIC,
    SURFACE_PIC50,
    SURFACE_FUNCTIONAL_NUMERIC,
    SURFACE_FUNCTIONAL_PIC50,
    SURFACE_APPROXIMATE,
    SURFACE_CURATED_REVIEW,
)


class HergTrainingSurfaceError(RuntimeError):
    """Raised when a training-surface build or validation fails closed."""


_OBSERVATION_SCHEMA = pa.schema(
    [
        pa.field("observation_id", pa.large_string(), nullable=False),
        pa.field("structure_id", pa.large_string()),
        pa.field("standardized_smiles", pa.large_string()),
        pa.field("standard_inchi_key", pa.large_string()),
        pa.field("structure_model_eligible", pa.bool_(), nullable=False),
        pa.field("source_family", pa.large_string(), nullable=False),
        pa.field("source_priority", pa.large_string(), nullable=False),
        pa.field("source_record_id", pa.large_string(), nullable=False),
        pa.field("source_aid", pa.large_string()),
        pa.field("source_cid", pa.large_string()),
        pa.field("assay_id", pa.large_string()),
        pa.field("assay_family", pa.large_string(), nullable=False),
        pa.field("target_id", pa.large_string(), nullable=False),
        pa.field("target_variant", pa.large_string(), nullable=False),
        pa.field("wild_type_evidence_scope", pa.large_string(), nullable=False),
        pa.field("master_confirmed_wild_type_scope", pa.bool_(), nullable=False),
        pa.field("measurement_modality", pa.large_string(), nullable=False),
        pa.field("method_detail", pa.large_string(), nullable=False),
        pa.field("modality_confidence", pa.large_string(), nullable=False),
        pa.field("automation_class", pa.large_string(), nullable=False),
        pa.field("dose_design", pa.large_string(), nullable=False),
        pa.field("endpoint_class", pa.large_string(), nullable=False),
        pa.field("native_endpoint", pa.large_string(), nullable=False),
        pa.field("native_relation", pa.large_string()),
        pa.field("native_value", pa.float64()),
        pa.field("native_unit", pa.large_string()),
        pa.field("native_label", pa.large_string()),
        pa.field("upstream_derived_binary_label", pa.int8()),
        pa.field("endpoint_standardization_status", pa.large_string(), nullable=False),
        pa.field("potency_relation_pic50", pa.large_string()),
        pa.field("potency_pic50_point", pa.float64()),
        pa.field("potency_pic50_lower_bound", pa.float64()),
        pa.field("potency_pic50_upper_bound", pa.float64()),
        pa.field("potency_censoring", pa.large_string(), nullable=False),
        pa.field("protocol_completeness_score", pa.int8(), nullable=False),
        pa.field("protocol_unresolved_fields_json", pa.large_string(), nullable=False),
        pa.field("primary_label_kind", pa.large_string(), nullable=False),
        pa.field("primary_binary_label", pa.int8()),
        pa.field("primary_numeric_value", pa.float64()),
        pa.field("primary_numeric_relation", pa.large_string()),
        pa.field("primary_numeric_unit", pa.large_string()),
        pa.field("primary_training_eligible", pa.bool_(), nullable=False),
        pa.field("sensitivity_training_eligible", pa.bool_(), nullable=False),
        pa.field("primary_eligibility_reason", pa.large_string(), nullable=False),
        pa.field("reported_public_observation", pa.bool_(), nullable=False),
        pa.field("confirmed_wt_fixed_dose_primary", pa.bool_(), nullable=False),
        pa.field("preclinical_native_numeric_primary", pa.bool_(), nullable=False),
        pa.field("standardized_pic50_primary", pa.bool_(), nullable=False),
        pa.field("functional_how_measured_numeric_primary", pa.bool_(), nullable=False),
        pa.field("functional_how_measured_pic50_primary", pa.bool_(), nullable=False),
        pa.field("approximate_relation_sensitivity_only", pa.bool_(), nullable=False),
        pa.field("curated_functional_t1_review_candidate", pa.bool_(), nullable=False),
        pa.field("cross_lineage_t1_review_candidate_structure", pa.bool_(), nullable=False),
        pa.field("formal_t1_validated", pa.bool_(), nullable=False),
        pa.field("clinical_qt_context_only", pa.bool_(), nullable=False),
        pa.field("q0_consensus_structure_available", pa.bool_(), nullable=False),
        pa.field("q0_consensus_target_class", pa.int8()),
        pa.field("model_split", pa.large_string()),
        pa.field("scaffold_group_id", pa.large_string()),
        pa.field("source_declared_split", pa.large_string()),
        pa.field("v1_5_evaluation_candidate", pa.bool_(), nullable=False),
        pa.field("v1_5_conflict_review_structure", pa.bool_(), nullable=False),
        pa.field("v1_5_lineage_group_count", pa.int64(), nullable=False),
        pa.field("v1_5_lineage_group_ids_json", pa.large_string(), nullable=False),
        pa.field("v1_5_maximum_lineage_evidence_strength", pa.large_string(), nullable=False),
        pa.field("evaluation_or_lineage_leakage_caution", pa.bool_(), nullable=False),
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
        pa.field("target_scope", pa.large_string(), nullable=False),
        pa.field("measurement_modality", pa.large_string(), nullable=False),
        pa.field("endpoint_semantics", pa.large_string(), nullable=False),
        pa.field("target_class", pa.int8(), nullable=False),
        pa.field("support_observation_count", pa.int64(), nullable=False),
        pa.field("support_observation_ids_json", pa.large_string(), nullable=False),
        pa.field("source_record_ids_json", pa.large_string(), nullable=False),
        pa.field("source_declared_splits_json", pa.large_string(), nullable=False),
        pa.field("direct_herg_label", pa.bool_(), nullable=False),
        pa.field("use_as_training_label", pa.bool_(), nullable=False),
        pa.field("v1_5_evaluation_candidate_structure", pa.bool_(), nullable=False),
        pa.field("v1_5_conflict_review_structure", pa.bool_(), nullable=False),
        pa.field("v1_5_lineage_caution", pa.bool_(), nullable=False),
        pa.field("cross_lineage_t1_review_candidate", pa.bool_(), nullable=False),
        pa.field("formal_t1_validated", pa.bool_(), nullable=False),
    ]
)

_VALIDATION_CANDIDATE_SCHEMA = pa.schema(
    [
        pa.field("candidate_id", pa.large_string(), nullable=False),
        pa.field("structure_id", pa.large_string(), nullable=False),
        pa.field("standardized_smiles", pa.large_string(), nullable=False),
        pa.field("standard_inchi_key", pa.large_string(), nullable=False),
        pa.field("model_split", pa.large_string(), nullable=False),
        pa.field("scaffold_group_id", pa.large_string(), nullable=False),
        pa.field("candidate_state", pa.large_string(), nullable=False),
        pa.field("concordant_binary_label", pa.int8()),
        pa.field("pubchem_qualifying_observation_count", pa.int64(), nullable=False),
        pa.field("pubchem_labels_json", pa.large_string(), nullable=False),
        pa.field("chembl_qualifying_observation_count", pa.int64(), nullable=False),
        pa.field("chembl_labels_json", pa.large_string(), nullable=False),
        pa.field("upstream_lineage_independence_adjudicated", pa.bool_(), nullable=False),
        pa.field("assay_modality_comparability_adjudicated", pa.bool_(), nullable=False),
        pa.field("formal_t1_assigned", pa.bool_(), nullable=False),
        pa.field("model_label_admitted", pa.bool_(), nullable=False),
        pa.field("v1_5_evaluation_candidate_structure", pa.bool_(), nullable=False),
        pa.field("v1_5_conflict_review_structure", pa.bool_(), nullable=False),
        pa.field("v1_5_lineage_caution", pa.bool_(), nullable=False),
        pa.field("candidate_semantics", pa.large_string(), nullable=False),
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
        pa.field("linked_structure_has_primary_training_label", pa.bool_(), nullable=False),
        pa.field("linked_structure_has_confirmed_wt_fixed_dose_label", pa.bool_(), nullable=False),
        pa.field("context_semantics", pa.large_string(), nullable=False),
        pa.field("title_or_term", pa.large_string()),
        pa.field("description_or_organ_system", pa.large_string()),
        pa.field("unit_of_measure", pa.large_string()),
        pa.field("time_frame", pa.large_string()),
        pa.field("native_context_json", pa.large_string(), nullable=False),
    ]
)

_REGISTRY_SCHEMA = pa.schema(
    [
        pa.field("surface_id", pa.large_string(), nullable=False),
        pa.field("parent_surface_id", pa.large_string()),
        pa.field("primary_artifact", pa.large_string(), nullable=False),
        pa.field("artifact_grain", pa.large_string(), nullable=False),
        pa.field("release_status", pa.large_string(), nullable=False),
        pa.field("row_count", pa.int64(), nullable=False),
        pa.field("reported_observation_count", pa.int64(), nullable=False),
        pa.field("unique_structure_count", pa.int64(), nullable=False),
        pa.field("usable_measurement_label_count", pa.int64(), nullable=False),
        pa.field("authorized_training_label_count", pa.int64(), nullable=False),
        pa.field("formal_validated_label_count", pa.int64(), nullable=False),
        pa.field("train_count", pa.int64(), nullable=False),
        pa.field("validation_count", pa.int64(), nullable=False),
        pa.field("test_count", pa.int64(), nullable=False),
        pa.field("confirmed_wild_type_observation_count", pa.int64(), nullable=False),
        pa.field("exact_relation_count", pa.int64(), nullable=False),
        pa.field("censored_relation_count", pa.int64(), nullable=False),
        pa.field("approximate_relation_count", pa.int64(), nullable=False),
        pa.field("direct_herg_label_surface", pa.bool_(), nullable=False),
        pa.field("clinical_context_only", pa.bool_(), nullable=False),
        pa.field("target_semantics", pa.large_string(), nullable=False),
        pa.field("inclusion_rule", pa.large_string(), nullable=False),
        pa.field("model_contract", pa.large_string(), nullable=False),
        pa.field("limitations_json", pa.large_string(), nullable=False),
    ]
)

_STRATA_SCHEMA = pa.schema(
    [
        pa.field("stratum_id", pa.large_string(), nullable=False),
        pa.field("surface_id", pa.large_string(), nullable=False),
        pa.field("source_family", pa.large_string(), nullable=False),
        pa.field("wild_type_evidence_scope", pa.large_string(), nullable=False),
        pa.field("measurement_modality", pa.large_string(), nullable=False),
        pa.field("automation_class", pa.large_string(), nullable=False),
        pa.field("dose_design", pa.large_string(), nullable=False),
        pa.field("endpoint_class", pa.large_string(), nullable=False),
        pa.field("native_endpoint", pa.large_string(), nullable=False),
        pa.field("native_relation", pa.large_string()),
        pa.field("native_unit", pa.large_string()),
        pa.field("potency_censoring", pa.large_string(), nullable=False),
        pa.field("protocol_completeness_score", pa.int8(), nullable=False),
        pa.field("primary_label_kind", pa.large_string(), nullable=False),
        pa.field("model_split", pa.large_string()),
        pa.field("observation_count", pa.int64(), nullable=False),
        pa.field("unique_structure_count", pa.int64(), nullable=False),
        pa.field("primary_eligible_label_count", pa.int64(), nullable=False),
        pa.field("sensitivity_eligible_label_count", pa.int64(), nullable=False),
        pa.field("lineage_caution_observation_count", pa.int64(), nullable=False),
    ]
)

_EXCLUSION_SCHEMA = pa.schema(
    [
        pa.field("audit_id", pa.large_string(), nullable=False),
        pa.field("audit_origin", pa.large_string(), nullable=False),
        pa.field("exclusion_scope", pa.large_string(), nullable=False),
        pa.field("upstream_exclusion_id", pa.large_string()),
        pa.field("source_family", pa.large_string(), nullable=False),
        pa.field("source_record_id", pa.large_string(), nullable=False),
        pa.field("observation_id", pa.large_string()),
        pa.field("structure_id", pa.large_string()),
        pa.field("target_scope", pa.large_string(), nullable=False),
        pa.field("primary_training_excluded", pa.bool_(), nullable=False),
        pa.field("sensitivity_training_excluded", pa.bool_(), nullable=False),
        pa.field("clinical_context_only", pa.bool_(), nullable=False),
        pa.field("exclusion_reason", pa.large_string(), nullable=False),
        pa.field("exclusion_detail", pa.large_string(), nullable=False),
    ]
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _schema_sha256(schema: pa.Schema) -> str:
    return hashlib.sha256(schema.serialize().to_pybytes()).hexdigest()


def _manifest_digest(payload: Mapping[str, Any]) -> str:
    body = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    return hashlib.sha256(_canonical_json(body).encode()).hexdigest()


def _stable_id(prefix: str, *parts: object) -> str:
    body = "\x1f".join("" if part is None else str(part) for part in parts)
    return f"{prefix}-{hashlib.sha256(body.encode()).hexdigest()[:24].upper()}"


def _checked_file(path: Path, *, suffixes: frozenset[str] | None = None) -> Path:
    allowed = suffixes or frozenset({".json", ".parquet"})
    if path.is_symlink() or not path.is_file() or path.suffix.casefold() not in allowed:
        raise HergTrainingSurfaceError(f"missing, unsafe, or unexpected input: {path}")
    return path.resolve()


def _input_binding(path: Path, *, role: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "role": role,
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }
    if path.suffix.casefold() == ".parquet":
        parquet = pq.ParquetFile(path)
        result["rows"] = parquet.metadata.num_rows
        result["arrow_schema_sha256"] = _schema_sha256(parquet.schema_arrow)
    return result


def _implementation_binding(path: Path) -> dict[str, Any]:
    source = _checked_file(path, suffixes=frozenset({".py"}))
    return {
        "path": str(source),
        "bytes": source.stat().st_size,
        "sha256": _sha256_file(source),
    }


def _validate_implementation_binding(manifest: Mapping[str, Any]) -> Path:
    binding = manifest.get("implementation")
    if not isinstance(binding, Mapping):
        raise HergTrainingSurfaceError("implementation binding is missing")
    path = _checked_file(Path(str(binding.get("path", ""))), suffixes=frozenset({".py"}))
    if path.stat().st_size != int(binding.get("bytes", -1)) or _sha256_file(path) != binding.get("sha256"):
        raise HergTrainingSurfaceError("implementation binding mismatch")
    return path


def _artifact(path: Path, schema: pa.Schema, rows: int) -> dict[str, Any]:
    return {
        "path": path.name,
        "rows": rows,
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
        "arrow_schema_sha256": _schema_sha256(schema),
    }


def _write_parquet(path: Path, rows: Sequence[Mapping[str, Any]], schema: pa.Schema) -> dict[str, Any]:
    table = pa.Table.from_pylist(list(rows), schema=schema)
    pq.write_table(
        table,
        path,
        compression="zstd",
        use_dictionary=False,
        row_group_size=65_536,
        data_page_version="1.0",
        version="2.6",
    )
    return _artifact(path, schema, table.num_rows)


def _json_list(value: object) -> list[Any]:
    try:
        parsed = json.loads(str(value))
    except (TypeError, json.JSONDecodeError):
        return []
    return parsed if isinstance(parsed, list) else []


def _finite_number(value: object) -> float | None:
    if not isinstance(value, (int, float, str)):
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def _finite(value: object) -> bool:
    return _finite_number(value) is not None


def _is_clinical_qt(row: Mapping[str, Any]) -> bool:
    return (
        row.get("measurement_modality") == "clinical_qt_in_vivo"
        or row.get("endpoint_class") == "clinical_qt_qtc_phenotype"
    )


def _label_decision(row: Mapping[str, Any], q0_labels: Mapping[str, int]) -> dict[str, Any]:
    """Return a source-specific decision without transforming native numeric labels."""

    source = str(row["source_family"])
    structure_id = str(row.get("structure_id") or "")
    clinical = _is_clinical_qt(row)
    valid = bool(row.get("structure_model_eligible"))
    upstream_binary = row.get("derived_binary_label")
    q0_label = q0_labels.get(structure_id)
    q0_consensus = q0_label is not None
    if row.get("target_variant") == "mutant_or_variant":
        raise HergTrainingSurfaceError("explicit mutant reached admitted master observations")
    if not valid:
        return {
            "kind": "none",
            "binary": None,
            "numeric": None,
            "relation": None,
            "unit": None,
            "primary": False,
            "sensitivity": False,
            "reason": "invalid_or_missing_standardized_structure",
            "q0_consensus": q0_consensus,
            "q0_label": q0_label,
        }
    if clinical:
        return {
            "kind": "clinical_context_not_label",
            "binary": None,
            "numeric": None,
            "relation": None,
            "unit": None,
            "primary": False,
            "sensitivity": False,
            "reason": "clinical_QT_QTc_context_not_direct_hERG_label",
            "q0_consensus": q0_consensus,
            "q0_label": q0_label,
        }
    if source == "pubchem_aid720551" and upstream_binary in {0, 1}:
        if q0_label is None or int(upstream_binary) != q0_label:
            return {
                "kind": "none",
                "binary": None,
                "numeric": None,
                "relation": None,
                "unit": None,
                "primary": False,
                "sensitivity": False,
                "reason": "fixed_dose_structure_lacks_unique_consensus_label",
                "q0_consensus": q0_consensus,
                "q0_label": q0_label,
            }
        return {
            "kind": "fixed_dose_binary",
            "binary": int(upstream_binary),
            "numeric": None,
            "relation": None,
            "unit": None,
            "primary": True,
            "sensitivity": True,
            "reason": "confirmed_WT_fixed_dose_decisive_consensus_support",
            "q0_consensus": True,
            "q0_label": q0_label,
        }
    if source in NUMERIC_SOURCE_FAMILIES:
        value = row.get("native_value")
        number = _finite_number(value)
        relation = str(row.get("native_relation") or "").strip()
        unit = str(row.get("native_unit") or "").strip()
        if number is None:
            reason = "native_numeric_value_missing_or_nonfinite"
        elif relation not in PRIMARY_RELATIONS | SENSITIVITY_RELATIONS:
            reason = "native_numeric_relation_missing_or_unresolved"
        elif not unit:
            reason = "native_numeric_unit_missing_or_unresolved"
        elif relation in SENSITIVITY_RELATIONS:
            return {
                "kind": "native_numeric_approximate_sensitivity",
                "binary": None,
                "numeric": number,
                "relation": relation,
                "unit": unit,
                "primary": False,
                "sensitivity": True,
                "reason": "approximate_native_relation_retained_for_sensitivity_only",
                "q0_consensus": q0_consensus,
                "q0_label": q0_label,
            }
        else:
            return {
                "kind": "native_numeric_relation_preserved",
                "binary": None,
                "numeric": number,
                "relation": relation,
                "unit": unit,
                "primary": True,
                "sensitivity": True,
                "reason": "native_numeric_endpoint_relation_and_unit_complete",
                "q0_consensus": q0_consensus,
                "q0_label": q0_label,
            }
        return {
            "kind": "none",
            "binary": None,
            "numeric": None,
            "relation": None,
            "unit": None,
            "primary": False,
            "sensitivity": False,
            "reason": reason,
            "q0_consensus": q0_consensus,
            "q0_label": q0_label,
        }
    return {
        "kind": "none",
        "binary": None,
        "numeric": None,
        "relation": None,
        "unit": None,
        "primary": False,
        "sensitivity": False,
        "reason": "no_source_specific_usable_training_label",
        "q0_consensus": q0_consensus,
        "q0_label": q0_label,
    }


def _protocol_map(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    result: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (
            str(row["source_family"]),
            str(row.get("assay_id") or ""),
            str(row["assay_family"]),
        )
        if key in result:
            raise HergTrainingSurfaceError(f"duplicate assay protocol key: {key}")
        result[key] = dict(row)
    return result


def _lineage_annotations(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    memberships: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        for observation_id in _json_list(row["observation_ids_json"]):
            memberships[str(observation_id)].append(row)
    result: dict[str, dict[str, Any]] = {}
    for observation_id, groups in memberships.items():
        ordered = sorted(groups, key=lambda row: str(row["lineage_group_id"]))
        maximum = max(
            (str(row["automated_evidence_strength"]) for row in ordered),
            key=lambda strength: STRENGTH_ORDER[strength],
        )
        result[observation_id] = {
            "ids": [str(row["lineage_group_id"]) for row in ordered],
            "maximum": maximum,
        }
    return result


def _surface_membership(row: Mapping[str, Any], surface_id: str) -> bool:
    if surface_id == SURFACE_REPORTED:
        return True
    if surface_id == SURFACE_CLEAN:
        return bool(row["primary_training_eligible"])
    if surface_id == SURFACE_CONFIRMED_WT:
        return bool(row["confirmed_wt_fixed_dose_primary"])
    if surface_id == SURFACE_NUMERIC:
        return bool(row["preclinical_native_numeric_primary"])
    if surface_id == SURFACE_PIC50:
        return bool(row["standardized_pic50_primary"])
    if surface_id == SURFACE_FUNCTIONAL_NUMERIC:
        return bool(row["functional_how_measured_numeric_primary"])
    if surface_id == SURFACE_FUNCTIONAL_PIC50:
        return bool(row["functional_how_measured_pic50_primary"])
    if surface_id == SURFACE_APPROXIMATE:
        return bool(row["approximate_relation_sensitivity_only"])
    if surface_id == SURFACE_CURATED_REVIEW:
        return bool(row["curated_functional_t1_review_candidate"])
    raise HergTrainingSurfaceError(f"unknown observation surface: {surface_id}")


def _observation_rows(
    observations: Sequence[Mapping[str, Any]],
    q0_labels: Mapping[str, int],
    protocols: Mapping[tuple[str, str, str], Mapping[str, Any]],
    candidate_observation_ids: set[str],
    conflict_structure_ids: set[str],
    lineage_by_observation: Mapping[str, Mapping[str, Any]],
    t1_review_structure_ids: set[str],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in observations:
        observation_id = str(row["observation_id"])
        structure_id = str(row.get("structure_id") or "")
        key = (
            str(row["source_family"]),
            str(row.get("assay_id") or ""),
            str(row["assay_family"]),
        )
        protocol = protocols.get(key)
        if protocol is None:
            raise HergTrainingSurfaceError(f"observation lacks assay protocol binding: {observation_id}")
        decision = _label_decision(row, q0_labels)
        lineage = lineage_by_observation.get(observation_id, {"ids": [], "maximum": "none"})
        primary_numeric = decision["kind"] == "native_numeric_relation_preserved"
        functional = str(row["measurement_modality"]) in FUNCTIONAL_MODALITIES
        standardized = bool(
            primary_numeric
            and row["endpoint_standardization_status"] in {"exact_standardized", "censored_standardized"}
            and row.get("potency_relation_pic50") in PRIMARY_RELATIONS
        )
        fixed = decision["kind"] == "fixed_dose_binary"
        conflict = bool(structure_id and structure_id in conflict_structure_ids)
        candidate = observation_id in candidate_observation_ids
        lineage_ids = list(lineage["ids"])
        result.append(
            {
                "observation_id": observation_id,
                "structure_id": row.get("structure_id"),
                "standardized_smiles": row.get("standardized_smiles"),
                "standard_inchi_key": row.get("standard_inchi_key"),
                "structure_model_eligible": bool(row["structure_model_eligible"]),
                "source_family": row["source_family"],
                "source_priority": row["source_priority"],
                "source_record_id": row["source_record_id"],
                "source_aid": str(row["source_aid"]) if row.get("source_aid") is not None else None,
                "source_cid": row.get("source_cid"),
                "assay_id": row.get("assay_id"),
                "assay_family": row["assay_family"],
                "target_id": row["target_id"],
                "target_variant": row["target_variant"],
                "wild_type_evidence_scope": row["wild_type_evidence_scope"],
                "master_confirmed_wild_type_scope": row["wild_type_evidence_scope"] == "confirmed_wild_type",
                "measurement_modality": row["measurement_modality"],
                "method_detail": row["method_detail"],
                "modality_confidence": row["modality_confidence"],
                "automation_class": row["automation_class"],
                "dose_design": row["dose_design"],
                "endpoint_class": row["endpoint_class"],
                "native_endpoint": row["native_endpoint"],
                "native_relation": row.get("native_relation"),
                "native_value": row.get("native_value"),
                "native_unit": row.get("native_unit"),
                "native_label": row.get("native_label"),
                "upstream_derived_binary_label": row.get("derived_binary_label"),
                "endpoint_standardization_status": row["endpoint_standardization_status"],
                "potency_relation_pic50": row.get("potency_relation_pic50"),
                "potency_pic50_point": row.get("potency_pic50_point"),
                "potency_pic50_lower_bound": row.get("potency_pic50_lower_bound"),
                "potency_pic50_upper_bound": row.get("potency_pic50_upper_bound"),
                "potency_censoring": row["potency_censoring"],
                "protocol_completeness_score": int(protocol["protocol_completeness_score"]),
                "protocol_unresolved_fields_json": protocol["unresolved_fields_json"],
                "primary_label_kind": decision["kind"],
                "primary_binary_label": decision["binary"],
                "primary_numeric_value": decision["numeric"],
                "primary_numeric_relation": decision["relation"],
                "primary_numeric_unit": decision["unit"],
                "primary_training_eligible": bool(decision["primary"]),
                "sensitivity_training_eligible": bool(decision["sensitivity"]),
                "primary_eligibility_reason": decision["reason"],
                "reported_public_observation": True,
                "confirmed_wt_fixed_dose_primary": fixed,
                "preclinical_native_numeric_primary": primary_numeric,
                "standardized_pic50_primary": standardized,
                "functional_how_measured_numeric_primary": primary_numeric and functional,
                "functional_how_measured_pic50_primary": standardized and functional,
                "approximate_relation_sensitivity_only": decision["kind"]
                == "native_numeric_approximate_sensitivity",
                "curated_functional_t1_review_candidate": bool(
                    row["source_family"] == "chembl_herg_specialized_view" and row["t1_candidate"]
                ),
                "cross_lineage_t1_review_candidate_structure": structure_id in t1_review_structure_ids,
                "formal_t1_validated": False,
                "clinical_qt_context_only": _is_clinical_qt(row),
                "q0_consensus_structure_available": bool(decision["q0_consensus"]),
                "q0_consensus_target_class": decision["q0_label"],
                "model_split": row.get("model_split"),
                "scaffold_group_id": row.get("scaffold_group_id"),
                "source_declared_split": row.get("source_split"),
                "v1_5_evaluation_candidate": candidate,
                "v1_5_conflict_review_structure": conflict,
                "v1_5_lineage_group_count": len(lineage_ids),
                "v1_5_lineage_group_ids_json": _canonical_json(lineage_ids),
                "v1_5_maximum_lineage_evidence_strength": lineage["maximum"],
                "evaluation_or_lineage_leakage_caution": candidate or conflict or bool(lineage_ids),
            }
        )
    return sorted(result, key=lambda item: item["observation_id"])


def _structure_rows(
    q0_tasks: Sequence[Mapping[str, Any]],
    structures: Mapping[str, Mapping[str, Any]],
    observations: Sequence[Mapping[str, Any]],
    candidate_structure_ids: set[str],
    conflict_structure_ids: set[str],
    lineage_structure_ids: set[str],
    t1_review_structure_ids: set[str],
) -> list[dict[str, Any]]:
    q0 = {
        str(row["structure_id"]): row
        for row in q0_tasks
        if row["task_id"] == "Q0_WEAK_FIXED_DOSE_BINARY" and row["eligible"]
    }
    if len(q0) != sum(
        row["task_id"] == "Q0_WEAK_FIXED_DOSE_BINARY" and bool(row["eligible"]) for row in q0_tasks
    ):
        raise HergTrainingSurfaceError("Q0 eligible task membership is not one row per structure")
    support: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in observations:
        sid = str(row.get("structure_id") or "")
        if (
            sid in q0
            and row["source_family"] == "pubchem_aid720551"
            and row["primary_label_kind"] == "fixed_dose_binary"
        ):
            support[sid].append(row)
    result: list[dict[str, Any]] = []
    for sid in sorted(q0):
        task = q0[sid]
        structure = structures.get(sid)
        rows = sorted(support.get(sid, []), key=lambda row: str(row["observation_id"]))
        if structure is None or not rows:
            raise HergTrainingSurfaceError("eligible Q0 structure lacks master structure or support")
        target = int(task["target_class"])
        if any(int(row["primary_binary_label"]) != target for row in rows):
            raise HergTrainingSurfaceError("Q0 support observation conflicts with consensus target")
        result.append(
            {
                "structure_id": sid,
                "standardized_smiles": structure["standardized_smiles"],
                "standard_inchi_key": structure["standard_inchi_key"],
                "model_split": structure["model_split"],
                "scaffold_group_id": structure["scaffold_group_id"],
                "split_source": structure["split_source"],
                "target_scope": "confirmed_wild_type",
                "measurement_modality": "high_throughput_thallium_flux",
                "endpoint_semantics": "AID720551_fixed_dose_activity_consensus_not_IC50",
                "target_class": target,
                "support_observation_count": len(rows),
                "support_observation_ids_json": _canonical_json([str(row["observation_id"]) for row in rows]),
                "source_record_ids_json": _canonical_json(
                    sorted({str(row["source_record_id"]) for row in rows})
                ),
                "source_declared_splits_json": _canonical_json(
                    sorted(
                        {str(row["source_declared_split"]) for row in rows if row["source_declared_split"]}
                    )
                ),
                "direct_herg_label": True,
                "use_as_training_label": True,
                "v1_5_evaluation_candidate_structure": sid in candidate_structure_ids,
                "v1_5_conflict_review_structure": sid in conflict_structure_ids,
                "v1_5_lineage_caution": sid in lineage_structure_ids,
                "cross_lineage_t1_review_candidate": sid in t1_review_structure_ids,
                "formal_t1_validated": False,
            }
        )
    return result


def _validation_candidate_rows(
    candidates: Sequence[Mapping[str, Any]],
    structures: Mapping[str, Mapping[str, Any]],
    candidate_structure_ids: set[str],
    conflict_structure_ids: set[str],
    lineage_structure_ids: set[str],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in sorted(candidates, key=lambda item: str(item["structure_id"])):
        sid = str(row["structure_id"])
        structure = structures.get(sid)
        if structure is None:
            raise HergTrainingSurfaceError("T1 review candidate lacks master structure")
        result.append(
            {
                "candidate_id": row["candidate_id"],
                "structure_id": sid,
                "standardized_smiles": row["standardized_smiles"],
                "standard_inchi_key": row["standard_inchi_key"],
                "model_split": structure["model_split"],
                "scaffold_group_id": structure["scaffold_group_id"],
                "candidate_state": row["candidate_state"],
                "concordant_binary_label": row.get("concordant_binary_label"),
                "pubchem_qualifying_observation_count": int(row["pubchem_qualifying_observation_count"]),
                "pubchem_labels_json": row["pubchem_labels_json"],
                "chembl_qualifying_observation_count": int(row["chembl_qualifying_observation_count"]),
                "chembl_labels_json": row["chembl_labels_json"],
                "upstream_lineage_independence_adjudicated": bool(
                    row["upstream_lineage_independence_adjudicated"]
                ),
                "assay_modality_comparability_adjudicated": bool(
                    row["assay_modality_comparability_adjudicated"]
                ),
                "formal_t1_assigned": bool(row["formal_t1_assigned"]),
                "model_label_admitted": bool(row["model_label_admitted"]),
                "v1_5_evaluation_candidate_structure": sid in candidate_structure_ids,
                "v1_5_conflict_review_structure": sid in conflict_structure_ids,
                "v1_5_lineage_caution": sid in lineage_structure_ids,
                "candidate_semantics": row["candidate_semantics"],
            }
        )
    return result


def _clinical_rows(
    clinical: Sequence[Mapping[str, Any]], observations: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    primary_structures = {
        str(row["structure_id"])
        for row in observations
        if row["primary_training_eligible"] and row["structure_id"]
    }
    fixed_structures = {
        str(row["structure_id"])
        for row in observations
        if row["confirmed_wt_fixed_dose_primary"] and row["structure_id"]
    }
    result = []
    for row in clinical:
        sid = str(row["structure_id"])
        if row["direct_herg_label"] or row["use_as_training_label"]:
            raise HergTrainingSurfaceError("clinical context was promoted upstream")
        result.append(
            {
                "clinical_context_id": row["clinical_context_id"],
                "context_class": row["context_class"],
                "structure_id": sid,
                "nct_id": row.get("nct_id"),
                "endpoint_candidate_id": row.get("endpoint_candidate_id"),
                "model_split": row["model_split"],
                "scaffold_group_id": row["scaffold_group_id"],
                "context_eligible": bool(row["context_eligible"]),
                "heldout_evaluation_eligible": bool(row["heldout_evaluation_eligible"]),
                "direct_herg_label": False,
                "use_as_training_label": False,
                "linked_structure_has_primary_training_label": sid in primary_structures,
                "linked_structure_has_confirmed_wt_fixed_dose_label": sid in fixed_structures,
                "context_semantics": (
                    "clinical_development_or_QT_QTc_context_only; never_a_direct_molecular_hERG_label"
                ),
                "title_or_term": row.get("title_or_term"),
                "description_or_organ_system": row.get("description_or_organ_system"),
                "unit_of_measure": row.get("unit_of_measure"),
                "time_frame": row.get("time_frame"),
                "native_context_json": row["native_context_json"],
            }
        )
    return sorted(result, key=lambda item: item["clinical_context_id"])


def _relation_counts(rows: Sequence[Mapping[str, Any]]) -> tuple[int, int, int]:
    exact = sum(row.get("native_relation") == "=" for row in rows)
    censored = sum(row.get("native_relation") in {"<", "<=", ">", ">="} for row in rows)
    approximate = sum(row.get("native_relation") == "~" for row in rows)
    return exact, censored, approximate


def _observation_registry_row(
    surface_id: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    parent: str | None,
    status: str,
    direct: bool,
    authorized: bool,
    target_semantics: str,
    inclusion_rule: str,
    model_contract: str,
    limitations: Sequence[str],
) -> dict[str, Any]:
    splits = Counter(str(row["model_split"]) for row in rows if row["model_split"])
    exact, censored, approximate = _relation_counts(rows)
    usable = sum(bool(row["primary_training_eligible"]) for row in rows)
    if surface_id == SURFACE_APPROXIMATE:
        usable = sum(bool(row["sensitivity_training_eligible"]) for row in rows)
    return {
        "surface_id": surface_id,
        "parent_surface_id": parent,
        "primary_artifact": OBSERVATION_OUTPUT,
        "artifact_grain": "observation",
        "release_status": status,
        "row_count": len(rows),
        "reported_observation_count": len(rows),
        "unique_structure_count": len({row["structure_id"] for row in rows if row["structure_id"]}),
        "usable_measurement_label_count": usable,
        "authorized_training_label_count": usable if authorized else 0,
        "formal_validated_label_count": 0,
        "train_count": splits["train"],
        "validation_count": splits["validation"],
        "test_count": splits["test"],
        "confirmed_wild_type_observation_count": sum(row["master_confirmed_wild_type_scope"] for row in rows),
        "exact_relation_count": exact,
        "censored_relation_count": censored,
        "approximate_relation_count": approximate,
        "direct_herg_label_surface": direct,
        "clinical_context_only": False,
        "target_semantics": target_semantics,
        "inclusion_rule": inclusion_rule,
        "model_contract": model_contract,
        "limitations_json": _canonical_json(list(limitations)),
    }


def _registry_rows(
    observations: Sequence[Mapping[str, Any]],
    structures: Sequence[Mapping[str, Any]],
    validation_candidates: Sequence[Mapping[str, Any]],
    clinical: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    memberships = {
        surface: [row for row in observations if _surface_membership(row, surface)]
        for surface in OBSERVATION_SURFACES
    }
    shared_limitations = [
        "clean means schema/structure/relation/unit eligible, not independent experimental lineage",
        "v1.5 candidate, conflict, and lineage flags require leakage review before evaluation use",
        "different endpoints and units require separate or conditioned objectives",
    ]
    rows = [
        _observation_registry_row(
            SURFACE_REPORTED,
            memberships[SURFACE_REPORTED],
            parent=None,
            status="released_reporting_inventory",
            direct=False,
            authorized=False,
            target_semantics="all admitted T0 public observations; not a pooled target",
            inclusion_rule="every v1.3 master observation, including nonlabel and clinical-context rows",
            model_contract="select a nested task-specific surface; do not pool this inventory as one target",
            limitations=shared_limitations,
        ),
        _observation_registry_row(
            SURFACE_CLEAN,
            memberships[SURFACE_CLEAN],
            parent=SURFACE_REPORTED,
            status="released_assay_aware_training_surface",
            direct=True,
            authorized=True,
            target_semantics="decisive fixed-dose binary or native numeric endpoint with relation and unit",
            inclusion_rule="valid structure; nonclinical; Q0 consensus binary or complete primary native numeric",
            model_contract="separate binary and endpoint/unit/relation-conditioned numeric heads",
            limitations=shared_limitations,
        ),
        _observation_registry_row(
            SURFACE_CONFIRMED_WT,
            memberships[SURFACE_CONFIRMED_WT],
            parent=SURFACE_CLEAN,
            status="released_confirmed_WT_training_support",
            direct=True,
            authorized=True,
            target_semantics="AID720551 fixed-dose activity call; never interpreted as IC50",
            inclusion_rule="confirmed-WT PubChem decisive observations supporting an eligible Q0 consensus",
            model_contract="binary classification; prefer the one-row-per-structure companion artifact",
            limitations=shared_limitations,
        ),
        _observation_registry_row(
            SURFACE_NUMERIC,
            memberships[SURFACE_NUMERIC],
            parent=SURFACE_CLEAN,
            status="released_assay_aware_training_surface",
            direct=True,
            authorized=True,
            target_semantics="reported preclinical native numeric measurement with relation and unit preserved",
            inclusion_rule="ChEMBL or quantitative compilation; finite value; resolved nonapproximate relation; unit",
            model_contract="endpoint/unit/relation-specific or conditioned objectives; censor-aware losses",
            limitations=shared_limitations,
        ),
        _observation_registry_row(
            SURFACE_PIC50,
            memberships[SURFACE_PIC50],
            parent=SURFACE_NUMERIC,
            status="released_censor_aware_training_surface",
            direct=True,
            authorized=True,
            target_semantics="exact or censored standardized pIC50 with relation-specific point or bound",
            inclusion_rule="primary numeric plus exact_standardized or censored_standardized pIC50",
            model_contract="censor-aware pIC50 regression; do not replace bounds with equality targets",
            limitations=shared_limitations,
        ),
        _observation_registry_row(
            SURFACE_FUNCTIONAL_NUMERIC,
            memberships[SURFACE_FUNCTIONAL_NUMERIC],
            parent=SURFACE_NUMERIC,
            status="released_method_resolved_training_surface",
            direct=True,
            authorized=True,
            target_semantics="native numeric measurement whose parsed how-measured modality is functional",
            inclusion_rule="primary numeric and modality in the explicit functional-modality allowlist",
            model_contract="method, endpoint, unit, and relation conditioned; report each modality separately",
            limitations=[
                *shared_limitations,
                "modality parsing remains metadata-derived and confidence-tagged",
            ],
        ),
        _observation_registry_row(
            SURFACE_FUNCTIONAL_PIC50,
            memberships[SURFACE_FUNCTIONAL_PIC50],
            parent=SURFACE_FUNCTIONAL_NUMERIC,
            status="released_method_resolved_censor_aware_surface",
            direct=True,
            authorized=True,
            target_semantics="functional how-measured exact or censored standardized pIC50",
            inclusion_rule="functional primary numeric plus standardized pIC50 point or relation-preserving bound",
            model_contract="censor-aware pIC50 regression stratified or conditioned by measurement modality",
            limitations=[
                *shared_limitations,
                "assay-family labels are retained but do not override modality evidence",
            ],
        ),
        _observation_registry_row(
            SURFACE_APPROXIMATE,
            memberships[SURFACE_APPROXIMATE],
            parent=SURFACE_NUMERIC,
            status="released_sensitivity_only_not_primary",
            direct=True,
            authorized=False,
            target_semantics="reported approximate native numeric relation",
            inclusion_rule="finite native numeric value with unit and relation '~'",
            model_contract="sensitivity analysis only; never mix with exact observations without an approximation flag",
            limitations=[
                *shared_limitations,
                "approximate observations are excluded from all primary surfaces",
            ],
        ),
        _observation_registry_row(
            SURFACE_CURATED_REVIEW,
            memberships[SURFACE_CURATED_REVIEW],
            parent=SURFACE_FUNCTIONAL_NUMERIC,
            status="candidate_review_only_not_formal_validation",
            direct=True,
            authorized=False,
            target_semantics="upstream ChEMBL quality-gate candidates; not formal T1 assignments",
            inclusion_rule="ChEMBL observations carrying the upstream t1_candidate quality-gate flag",
            model_contract="may remain in its underlying broad surface; never call validated or gold",
            limitations=[*shared_limitations, "target and lineage adjudication remain incomplete"],
        ),
    ]
    structure_splits = Counter(str(row["model_split"]) for row in structures)
    rows.append(
        {
            "surface_id": SURFACE_STRUCTURE_BINARY,
            "parent_surface_id": SURFACE_CONFIRMED_WT,
            "primary_artifact": STRUCTURE_OUTPUT,
            "artifact_grain": "structure",
            "release_status": "released_canonical_training_surface",
            "row_count": len(structures),
            "reported_observation_count": sum(int(row["support_observation_count"]) for row in structures),
            "unique_structure_count": len(structures),
            "usable_measurement_label_count": len(structures),
            "authorized_training_label_count": len(structures),
            "formal_validated_label_count": 0,
            "train_count": structure_splits["train"],
            "validation_count": structure_splits["validation"],
            "test_count": structure_splits["test"],
            "confirmed_wild_type_observation_count": sum(
                int(row["support_observation_count"]) for row in structures
            ),
            "exact_relation_count": 0,
            "censored_relation_count": 0,
            "approximate_relation_count": 0,
            "direct_herg_label_surface": True,
            "clinical_context_only": False,
            "target_semantics": "one consensus fixed-dose binary label per confirmed-WT structure",
            "inclusion_rule": "eligible Q0 structure with only concordant decisive AID720551 support",
            "model_contract": "one structure entity per row; fixed scaffold split; binary classification",
            "limitations_json": _canonical_json(
                ["fixed-dose activity is not IC50", "class imbalance requires explicit handling"]
            ),
        }
    )
    candidate_splits = Counter(str(row["model_split"]) for row in validation_candidates)
    qualifying = sum(
        int(row["pubchem_qualifying_observation_count"]) + int(row["chembl_qualifying_observation_count"])
        for row in validation_candidates
    )
    rows.append(
        {
            "surface_id": SURFACE_T1_REVIEW,
            "parent_surface_id": SURFACE_STRUCTURE_BINARY,
            "primary_artifact": VALIDATION_CANDIDATE_OUTPUT,
            "artifact_grain": "structure_review_candidate",
            "release_status": "candidate_review_only_not_formal_validation",
            "row_count": len(validation_candidates),
            "reported_observation_count": qualifying,
            "unique_structure_count": len(validation_candidates),
            "usable_measurement_label_count": 0,
            "authorized_training_label_count": 0,
            "formal_validated_label_count": 0,
            "train_count": candidate_splits["train"],
            "validation_count": candidate_splits["validation"],
            "test_count": candidate_splits["test"],
            "confirmed_wild_type_observation_count": 0,
            "exact_relation_count": 0,
            "censored_relation_count": 0,
            "approximate_relation_count": 0,
            "direct_herg_label_surface": False,
            "clinical_context_only": False,
            "target_semantics": "cross-lineage agreement or disagreement requiring human adjudication",
            "inclusion_rule": "upstream cross-lineage T1 review candidate inventory",
            "model_contract": "review only; refreeze lineages and splits after adjudication",
            "limitations_json": _canonical_json(
                ["lineage independence not adjudicated", "assay comparability not adjudicated"]
            ),
        }
    )
    rows.append(
        {
            "surface_id": SURFACE_FORMAL_T1,
            "parent_surface_id": SURFACE_T1_REVIEW,
            "primary_artifact": VALIDATION_CANDIDATE_OUTPUT,
            "artifact_grain": "virtual_blocked_surface",
            "release_status": "blocked_zero_formal_T1_assignments",
            "row_count": 0,
            "reported_observation_count": 0,
            "unique_structure_count": 0,
            "usable_measurement_label_count": 0,
            "authorized_training_label_count": 0,
            "formal_validated_label_count": 0,
            "train_count": 0,
            "validation_count": 0,
            "test_count": 0,
            "confirmed_wild_type_observation_count": 0,
            "exact_relation_count": 0,
            "censored_relation_count": 0,
            "approximate_relation_count": 0,
            "direct_herg_label_surface": False,
            "clinical_context_only": False,
            "target_semantics": "formal T1 validated surface intentionally empty",
            "inclusion_rule": "requires completed lineage and modality adjudication plus formal upstream assignment",
            "model_contract": "unavailable; do not substitute review candidates",
            "limitations_json": _canonical_json(["no formal T1 assignments exist locally"]),
        }
    )
    clinical_splits = Counter(str(row["model_split"]) for row in clinical)
    rows.append(
        {
            "surface_id": SURFACE_CLINICAL,
            "parent_surface_id": None,
            "primary_artifact": CLINICAL_OUTPUT,
            "artifact_grain": "clinical_context",
            "release_status": "released_context_only_not_training_label",
            "row_count": len(clinical),
            "reported_observation_count": 0,
            "unique_structure_count": len({row["structure_id"] for row in clinical}),
            "usable_measurement_label_count": 0,
            "authorized_training_label_count": 0,
            "formal_validated_label_count": 0,
            "train_count": clinical_splits["train"],
            "validation_count": clinical_splits["validation"],
            "test_count": clinical_splits["test"],
            "confirmed_wild_type_observation_count": 0,
            "exact_relation_count": 0,
            "censored_relation_count": 0,
            "approximate_relation_count": 0,
            "direct_herg_label_surface": False,
            "clinical_context_only": True,
            "target_semantics": "development and human QT/QTc context downstream of exposure and many mechanisms",
            "inclusion_rule": "all v1.3 clinical context rows, preserved as context only",
            "model_contract": "stratification or held-out translation analysis; never a molecular hERG target",
            "limitations_json": _canonical_json(
                ["clinical QT is not hERG inhibition", "exposure and non-hERG mechanisms are unresolved"]
            ),
        }
    )
    return sorted(rows, key=lambda row: row["surface_id"])


def _measurement_strata(observations: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], dict[str, Any]] = {}
    for surface in OBSERVATION_SURFACES:
        for row in observations:
            if not _surface_membership(row, surface):
                continue
            key = (
                surface,
                row["source_family"],
                row["wild_type_evidence_scope"],
                row["measurement_modality"],
                row["automation_class"],
                row["dose_design"],
                row["endpoint_class"],
                row["native_endpoint"],
                row["native_relation"],
                row["native_unit"],
                row["potency_censoring"],
                row["protocol_completeness_score"],
                row["primary_label_kind"],
                row["model_split"],
            )
            if key not in groups:
                groups[key] = {
                    "rows": 0,
                    "structures": set(),
                    "primary": 0,
                    "sensitivity": 0,
                    "lineage": 0,
                }
            aggregate = groups[key]
            aggregate["rows"] += 1
            if row["structure_id"]:
                aggregate["structures"].add(str(row["structure_id"]))
            aggregate["primary"] += int(row["primary_training_eligible"])
            aggregate["sensitivity"] += int(row["sensitivity_training_eligible"])
            aggregate["lineage"] += int(row["evaluation_or_lineage_leakage_caution"])
    result = []
    for key in sorted(groups, key=lambda value: tuple("" if part is None else str(part) for part in value)):
        aggregate = groups[key]
        result.append(
            {
                "stratum_id": _stable_id("HSTRAT", *key),
                "surface_id": key[0],
                "source_family": key[1],
                "wild_type_evidence_scope": key[2],
                "measurement_modality": key[3],
                "automation_class": key[4],
                "dose_design": key[5],
                "endpoint_class": key[6],
                "native_endpoint": key[7],
                "native_relation": key[8],
                "native_unit": key[9],
                "potency_censoring": key[10],
                "protocol_completeness_score": key[11],
                "primary_label_kind": key[12],
                "model_split": key[13],
                "observation_count": aggregate["rows"],
                "unique_structure_count": len(aggregate["structures"]),
                "primary_eligible_label_count": aggregate["primary"],
                "sensitivity_eligible_label_count": aggregate["sensitivity"],
                "lineage_caution_observation_count": aggregate["lineage"],
            }
        )
    return result


def _exclusion_rows(
    observations: Sequence[Mapping[str, Any]], upstream: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in observations:
        if row["primary_training_eligible"]:
            continue
        observation_id = str(row["observation_id"])
        result.append(
            {
                "audit_id": _stable_id("HTRAINX", "observation", observation_id),
                "audit_origin": "v1_6_observation_primary_eligibility",
                "exclusion_scope": "primary_training_surface",
                "upstream_exclusion_id": None,
                "source_family": row["source_family"],
                "source_record_id": row["source_record_id"],
                "observation_id": observation_id,
                "structure_id": row.get("structure_id"),
                "target_scope": row["wild_type_evidence_scope"],
                "primary_training_excluded": True,
                "sensitivity_training_excluded": not bool(row["sensitivity_training_eligible"]),
                "clinical_context_only": bool(row["clinical_qt_context_only"]),
                "exclusion_reason": row["primary_eligibility_reason"],
                "exclusion_detail": (
                    "source-specific eligibility decision; observation remains in the reported inventory"
                ),
            }
        )
    for row in upstream:
        result.append(
            {
                "audit_id": _stable_id("HTRAINX", "upstream", row["master_exclusion_id"]),
                "audit_origin": "v1_3_master_exclusion_replayed",
                "exclusion_scope": row["exclusion_scope"],
                "upstream_exclusion_id": row["master_exclusion_id"],
                "source_family": row["source_family"],
                "source_record_id": row["source_record_id"],
                "observation_id": row.get("observation_id"),
                "structure_id": row.get("structure_id"),
                "target_scope": row["target_scope"],
                "primary_training_excluded": True,
                "sensitivity_training_excluded": True,
                "clinical_context_only": False,
                "exclusion_reason": row["exclusion_reason"],
                "exclusion_detail": row["exclusion_detail"],
            }
        )
    return sorted(result, key=lambda row: row["audit_id"])


def _surface_counts(registry: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        str(row["surface_id"]): {
            "artifact_grain": row["artifact_grain"],
            "release_status": row["release_status"],
            "rows": int(row["row_count"]),
            "reported_observations": int(row["reported_observation_count"]),
            "unique_structures": int(row["unique_structure_count"]),
            "usable_measurement_labels": int(row["usable_measurement_label_count"]),
            "authorized_training_labels": int(row["authorized_training_label_count"]),
            "formal_validated_labels": int(row["formal_validated_label_count"]),
        }
        for row in registry
    }


def _report_text(manifest: Mapping[str, Any]) -> str:
    counts = manifest["counts"]
    surfaces = counts["surfaces"]
    clean = surfaces[SURFACE_CLEAN]
    fixed = surfaces[SURFACE_STRUCTURE_BINARY]
    numeric = surfaces[SURFACE_NUMERIC]
    functional = surfaces[SURFACE_FUNCTIONAL_NUMERIC]
    functional_pic50 = surfaces[SURFACE_FUNCTIONAL_PIC50]
    return "\n".join(
        [
            "# hERG training surfaces v1.6",
            "",
            "## Release outcome",
            "",
            f"This downstream release exposes {clean['authorized_training_labels']:,} primary-eligible source-faithful observations spanning {clean['unique_structures']:,} structures. It also provides {fixed['authorized_training_labels']:,} one-row-per-structure confirmed-wild-type fixed-dose consensus labels backed by {fixed['reported_observations']:,} decisive observations.",
            "",
            "No model was trained. No row was promoted to gold or formal T1. No upstream release was modified.",
            "",
            "## Scientific boundary",
            "",
            "The broad primary surface admits only two label forms: decisive AID720551 fixed-dose binary observations whose structure has a unique Q0 consensus, and finite public native numeric measurements whose endpoint, nonapproximate relation, and unit are present. Native numeric endpoints are not pooled. Censored relations and standardized pIC50 bounds remain intact. Approximate relations are isolated in a sensitivity-only surface.",
            "",
            "Clinical QT/QTc and development records are context only. They never supply a molecular hERG training label. Explicit mutants remain excluded. Wild-type-or-unspecified records are never upgraded to confirmed wild type.",
            "",
            "## Nested surfaces",
            "",
            f"The preclinical native numeric surface contains {numeric['authorized_training_labels']:,} observations across {numeric['unique_structures']:,} structures. The how-measured functional subset contains {functional['authorized_training_labels']:,} observations across {functional['unique_structures']:,} structures; {functional_pic50['authorized_training_labels']:,} of those are exact or censored standardized IC50/pIC50 observations. Functional membership follows the parsed measurement modality, not the source assay-family label, because locally available descriptions often identify patch clamp even when the source family field says binding or other.",
            "",
            f"The formal validated T1 surface is intentionally empty: {surfaces[SURFACE_FORMAL_T1]['formal_validated_labels']} formal labels. The {surfaces[SURFACE_T1_REVIEW]['rows']:,} cross-lineage structures and {surfaces[SURFACE_CURATED_REVIEW]['rows']:,} curated functional observations remain review candidates only.",
            "",
            f"Clinical context is separate: {surfaces[SURFACE_CLINICAL]['rows']:,} context rows spanning {surfaces[SURFACE_CLINICAL]['unique_structures']:,} structures, with zero direct hERG labels.",
            "",
            "## Leakage and use",
            "",
            "Every molecular row retains the frozen structure and whole-scaffold split. Candidate, conflict-queue, and automated lineage flags from v1.5 are carried as cautions, not automatic exclusions or proof of duplication. Evaluation panels must be adjudicated and then refrozen across structure, scaffold, assay, document, and measurement lineage before use.",
            "",
            "Use the structure artifact for the large fixed-dose binary task. Use the observation artifact for assay-aware numeric or multitask objectives, conditioning or separating by endpoint, unit, relation, and measurement modality. The reporting inventory itself is not a pooled target.",
            "",
            "## Validation",
            "",
            f"The build validated {counts['reported_observations']:,} admitted observations, {counts['master_structures']:,} master structures, {counts['explicit_mutant_exclusions']:,} replayed explicit-mutant exclusions, {counts['measurement_strata']:,} deterministic measurement strata, all physical input and output hashes, Arrow schemas, row counts, relation semantics, clinical nonpromotion, formal-tier nonpromotion, and structure/scaffold split exclusivity.",
            "",
            "## Limits",
            "",
            "Clean means structurally and semantically eligible under the stated source-specific rules; it does not mean experimentally independent or human-adjudicated. Protocol completeness is inherited from local metadata and may remain unresolved. The quantitative compilation does not resolve original assay modality. Cross-source equal values and reuse signatures remain automated evidence only. No primary papers were newly adjudicated in this build.",
            "",
        ]
    )


def build_herg_training_surfaces(
    *,
    master_root: Path,
    candidate_adjudication_root: Path,
    evidence_tiers_root: Path,
    output_root: Path,
    report_root: Path,
) -> dict[str, Any]:
    """Build the additive v1.6 surfaces and validate them before publication."""

    master = master_root.resolve()
    adjudication = candidate_adjudication_root.resolve()
    evidence = evidence_tiers_root.resolve()
    validate_herg_master_dataset(master)
    validate_herg_candidate_adjudication(adjudication)
    validate_herg_evidence_tiers(evidence)

    input_specs = [
        ("v1_3_master_manifest", master / "herg_master_manifest.json"),
        ("v1_3_observation_master", master / "observation_master.parquet"),
        ("v1_3_structure_master", master / "structure_master.parquet"),
        ("v1_3_task_membership", master / "task_membership.parquet"),
        ("v1_3_assay_protocol_index", master / "assay_protocol_index.parquet"),
        ("v1_3_clinical_context", master / "clinical_context_master.parquet"),
        ("v1_3_master_exclusions", master / "master_exclusions.parquet"),
        (
            "v1_5_candidate_adjudication_manifest",
            adjudication / "herg_candidate_adjudication_manifest.json",
        ),
        ("v1_5_candidate_evidence", adjudication / "candidate_automated_evidence.parquet"),
        ("v1_5_conflict_evidence", adjudication / "conflict_automated_evidence.parquet"),
        ("v1_5_lineage_evidence", adjudication / "lineage_group_evidence.parquet"),
        ("v1_evidence_tiers_manifest", evidence / "herg_evidence_tiers_manifest.json"),
        ("v1_structure_evidence_tiers", evidence / "structure_evidence_tiers.parquet"),
        ("v1_cross_lineage_t1_candidates", evidence / "cross_lineage_t1_candidates.parquet"),
    ]
    inputs = [(role, _checked_file(path)) for role, path in input_specs]
    paths = {role: path for role, path in inputs}
    output = output_root.resolve()
    report = report_root.resolve()
    if output.exists() or report.exists():
        raise HergTrainingSurfaceError("output_root and report_root must not already exist")
    output.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    report_staging = Path(tempfile.mkdtemp(prefix=f".{report.name}.", dir=report.parent))
    try:
        observations = pq.read_table(paths["v1_3_observation_master"]).to_pylist()
        structure_input = pq.read_table(paths["v1_3_structure_master"]).to_pylist()
        structures = {str(row["structure_id"]): row for row in structure_input}
        if len(structures) != len(structure_input):
            raise HergTrainingSurfaceError("master structure IDs are not unique")
        tasks = pq.read_table(paths["v1_3_task_membership"]).to_pylist()
        q0_labels = {
            str(row["structure_id"]): int(row["target_class"])
            for row in tasks
            if row["task_id"] == "Q0_WEAK_FIXED_DOSE_BINARY" and row["eligible"]
        }
        protocols = _protocol_map(pq.read_table(paths["v1_3_assay_protocol_index"]).to_pylist())
        candidate_evidence = pq.read_table(paths["v1_5_candidate_evidence"]).to_pylist()
        conflict_evidence = pq.read_table(paths["v1_5_conflict_evidence"]).to_pylist()
        lineage_evidence = pq.read_table(paths["v1_5_lineage_evidence"]).to_pylist()
        tier_structures = pq.read_table(paths["v1_structure_evidence_tiers"]).to_pylist()
        t1_candidates = pq.read_table(paths["v1_cross_lineage_t1_candidates"]).to_pylist()
        tier_by_structure = {str(row["structure_id"]): row for row in tier_structures}
        if len(tier_by_structure) != len(tier_structures):
            raise HergTrainingSurfaceError("evidence tiers are not one row per structure")
        if any(row["formal_t1_assigned"] or row["model_label_admitted"] for row in tier_structures):
            raise HergTrainingSurfaceError("formal validation exists unexpectedly; policy review required")

        candidate_observation_ids = {str(row["observation_id"]) for row in candidate_evidence}
        candidate_structure_ids = {str(row["structure_id"]) for row in candidate_evidence}
        conflict_structure_ids = {str(row["structure_id"]) for row in conflict_evidence}
        lineage_by_observation = _lineage_annotations(lineage_evidence)
        observation_structure = {
            str(row["observation_id"]): str(row.get("structure_id") or "") for row in observations
        }
        lineage_structure_ids = {
            observation_structure[observation_id]
            for observation_id in lineage_by_observation
            if observation_id in observation_structure and observation_structure[observation_id]
        }
        t1_review_structure_ids = {str(row["structure_id"]) for row in t1_candidates}
        observation_rows = _observation_rows(
            observations,
            q0_labels,
            protocols,
            candidate_observation_ids,
            conflict_structure_ids,
            lineage_by_observation,
            t1_review_structure_ids,
        )
        structure_rows = _structure_rows(
            tasks,
            structures,
            observation_rows,
            candidate_structure_ids,
            conflict_structure_ids,
            lineage_structure_ids,
            t1_review_structure_ids,
        )
        validation_rows = _validation_candidate_rows(
            t1_candidates,
            structures,
            candidate_structure_ids,
            conflict_structure_ids,
            lineage_structure_ids,
        )
        clinical_rows = _clinical_rows(
            pq.read_table(paths["v1_3_clinical_context"]).to_pylist(), observation_rows
        )
        registry_rows = _registry_rows(observation_rows, structure_rows, validation_rows, clinical_rows)
        strata_rows = _measurement_strata(observation_rows)
        exclusions = _exclusion_rows(
            observation_rows, pq.read_table(paths["v1_3_master_exclusions"]).to_pylist()
        )

        artifacts = {
            OBSERVATION_OUTPUT: _write_parquet(
                staging / OBSERVATION_OUTPUT, observation_rows, _OBSERVATION_SCHEMA
            ),
            STRUCTURE_OUTPUT: _write_parquet(staging / STRUCTURE_OUTPUT, structure_rows, _STRUCTURE_SCHEMA),
            VALIDATION_CANDIDATE_OUTPUT: _write_parquet(
                staging / VALIDATION_CANDIDATE_OUTPUT,
                validation_rows,
                _VALIDATION_CANDIDATE_SCHEMA,
            ),
            CLINICAL_OUTPUT: _write_parquet(staging / CLINICAL_OUTPUT, clinical_rows, _CLINICAL_SCHEMA),
            REGISTRY_OUTPUT: _write_parquet(staging / REGISTRY_OUTPUT, registry_rows, _REGISTRY_SCHEMA),
            STRATA_OUTPUT: _write_parquet(staging / STRATA_OUTPUT, strata_rows, _STRATA_SCHEMA),
            EXCLUSION_OUTPUT: _write_parquet(staging / EXCLUSION_OUTPUT, exclusions, _EXCLUSION_SCHEMA),
        }
        explicit_mutants = sum(
            row["exclusion_reason"] == "explicit_mutant_or_variant"
            for row in exclusions
            if row["audit_origin"] == "v1_3_master_exclusion_replayed"
        )
        declared_mutants = int(
            json.loads(paths["v1_3_master_manifest"].read_text(encoding="utf-8"))["counts"][
                "explicit_mutant_exclusions"
            ]
        )
        if explicit_mutants != declared_mutants:
            raise HergTrainingSurfaceError("master explicit-mutant exclusions were not replayed exactly")
        counts: dict[str, Any] = {
            "reported_observations": len(observation_rows),
            "master_structures": len(structures),
            "primary_eligible_observations": sum(
                row["primary_training_eligible"] for row in observation_rows
            ),
            "primary_eligible_unique_structures": len(
                {
                    row["structure_id"]
                    for row in observation_rows
                    if row["primary_training_eligible"] and row["structure_id"]
                }
            ),
            "sensitivity_only_observations": sum(
                row["approximate_relation_sensitivity_only"] for row in observation_rows
            ),
            "explicit_mutant_exclusions": explicit_mutants,
            "clinical_context_rows": len(clinical_rows),
            "formal_T1_assignments": 0,
            "formal_validated_labels": 0,
            "measurement_strata": len(strata_rows),
            "training_exclusion_audit_rows": len(exclusions),
            "primary_eligibility_reason_counts": dict(
                sorted(Counter(str(row["primary_eligibility_reason"]) for row in observation_rows).items())
            ),
            "primary_label_kind_counts": dict(
                sorted(Counter(str(row["primary_label_kind"]) for row in observation_rows).items())
            ),
            "surfaces": _surface_counts(registry_rows),
        }
        body: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "dataset_id": DATASET_ID,
            "implementation": _implementation_binding(Path(__file__)),
            "inputs": [_input_binding(path, role=role) for role, path in inputs],
            "scientific_contract": {
                "master_release_preserved": True,
                "candidate_adjudication_release_preserved": True,
                "explicit_mutants_admitted": 0,
                "clinical_QT_as_hERG_labels": 0,
                "formal_T1_assignments": 0,
                "review_candidates_promoted": 0,
                "primary_numeric_relations": sorted(PRIMARY_RELATIONS),
                "approximate_relations_primary_eligible": False,
                "censored_relations_preserved": True,
                "native_endpoint_and_unit_pooling_allowed": False,
                "confirmed_WT_scope_source": "v1_3_master_scope_and_Q0_consensus_only",
                "functional_membership_basis": "v1_3_parsed_measurement_modality_allowlist",
                "split_policy": "preserve_v1_3_structure_and_whole_scaffold_partitions",
                "lineage_policy": "carry_v1_5_cautions_without_claiming_duplicate_or_independence",
            },
            "counts": counts,
            "artifacts": artifacts,
        }
        report_path = report_staging / REPORT_NAME
        report_path.write_text(_report_text(body), encoding="utf-8")
        body["report_artifact"] = {
            "path": REPORT_NAME,
            "bytes": report_path.stat().st_size,
            "sha256": _sha256_file(report_path),
        }
        manifest = dict(body)
        manifest["manifest_sha256"] = _manifest_digest(body)
        (staging / MANIFEST_NAME).write_text(_canonical_json(manifest) + "\n", encoding="utf-8")
        os.replace(staging, output)
        os.replace(report_staging, report)
        validate_herg_training_surfaces(output, report_root=report)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        shutil.rmtree(report_staging, ignore_errors=True)
        if output.exists():
            shutil.rmtree(output, ignore_errors=True)
        if report.exists():
            shutil.rmtree(report, ignore_errors=True)
        raise


def _validate_bound_inputs(manifest: Mapping[str, Any]) -> dict[str, Path]:
    roles: dict[str, Path] = {}
    for binding in manifest.get("inputs", []):
        role = str(binding["role"])
        path = _checked_file(Path(str(binding["path"])))
        if role in roles:
            raise HergTrainingSurfaceError(f"duplicate input role: {role}")
        roles[role] = path
        if path.stat().st_size != int(binding["bytes"]) or _sha256_file(path) != binding["sha256"]:
            raise HergTrainingSurfaceError(f"input binding mismatch: {role}")
        if path.suffix.casefold() == ".parquet":
            parquet = pq.ParquetFile(path)
            if parquet.metadata.num_rows != int(binding["rows"]):
                raise HergTrainingSurfaceError(f"input row count mismatch: {role}")
            if _schema_sha256(parquet.schema_arrow) != binding["arrow_schema_sha256"]:
                raise HergTrainingSurfaceError(f"input schema mismatch: {role}")
    required = {
        "v1_3_master_manifest",
        "v1_5_candidate_adjudication_manifest",
        "v1_evidence_tiers_manifest",
    }
    if not required.issubset(roles):
        raise HergTrainingSurfaceError("required upstream manifest bindings are missing")
    validate_herg_master_dataset(roles["v1_3_master_manifest"].parent)
    validate_herg_candidate_adjudication(roles["v1_5_candidate_adjudication_manifest"].parent)
    validate_herg_evidence_tiers(roles["v1_evidence_tiers_manifest"].parent)
    return roles


def validate_herg_training_surfaces(output_root: Path, *, report_root: Path | None = None) -> dict[str, Any]:
    """Validate physical bindings and the high-risk scientific contracts."""

    root = output_root.resolve()
    manifest_path = _checked_file(root / MANIFEST_NAME, suffixes=frozenset({".json"}))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("dataset_id") != DATASET_ID:
        raise HergTrainingSurfaceError("unexpected training-surface manifest identity")
    if manifest.get("manifest_sha256") != _manifest_digest(manifest):
        raise HergTrainingSurfaceError("training-surface manifest digest mismatch")
    _validate_implementation_binding(manifest)
    roles = _validate_bound_inputs(manifest)
    schemas = {
        OBSERVATION_OUTPUT: _OBSERVATION_SCHEMA,
        STRUCTURE_OUTPUT: _STRUCTURE_SCHEMA,
        VALIDATION_CANDIDATE_OUTPUT: _VALIDATION_CANDIDATE_SCHEMA,
        CLINICAL_OUTPUT: _CLINICAL_SCHEMA,
        REGISTRY_OUTPUT: _REGISTRY_SCHEMA,
        STRATA_OUTPUT: _STRATA_SCHEMA,
        EXCLUSION_OUTPUT: _EXCLUSION_SCHEMA,
    }
    expected_members = {MANIFEST_NAME, *schemas}
    if root.is_symlink() or not root.is_dir() or {path.name for path in root.iterdir()} != expected_members:
        raise HergTrainingSurfaceError("training-surface output membership mismatch")
    for name, schema in schemas.items():
        path = _checked_file(root / name, suffixes=frozenset({".parquet"}))
        metadata = manifest["artifacts"][name]
        parquet = pq.ParquetFile(path)
        if (
            path.stat().st_size != int(metadata["bytes"])
            or _sha256_file(path) != metadata["sha256"]
            or parquet.metadata.num_rows != int(metadata["rows"])
            or parquet.schema_arrow != schema
            or metadata["arrow_schema_sha256"] != _schema_sha256(schema)
        ):
            raise HergTrainingSurfaceError(f"artifact integrity mismatch: {name}")

    observations = pq.read_table(root / OBSERVATION_OUTPUT).to_pylist()
    observation_ids = [str(row["observation_id"]) for row in observations]
    if observation_ids != sorted(observation_ids) or len(observation_ids) != len(set(observation_ids)):
        raise HergTrainingSurfaceError("observation surface is not unique and sorted")
    if any(row["target_variant"] == "mutant_or_variant" for row in observations):
        raise HergTrainingSurfaceError("mutant observation was admitted")
    structure_splits: dict[str, set[str]] = defaultdict(set)
    scaffold_splits: dict[str, set[str]] = defaultdict(set)
    fixed_observation_ids: set[str] = set()
    fixed_labels: dict[str, int] = {}
    for row in observations:
        if row["model_split"] or row["scaffold_group_id"]:
            if row["model_split"] not in PARTITIONS or not row["scaffold_group_id"]:
                raise HergTrainingSurfaceError("partial or invalid model split assignment")
            if row["structure_id"]:
                structure_splits[str(row["structure_id"])].add(str(row["model_split"]))
            scaffold_splits[str(row["scaffold_group_id"])].add(str(row["model_split"]))
        if row["clinical_qt_context_only"] and (
            row["primary_training_eligible"] or row["sensitivity_training_eligible"]
        ):
            raise HergTrainingSurfaceError("clinical QT row admitted as molecular label")
        kind = row["primary_label_kind"]
        if kind == "fixed_dose_binary":
            if (
                row["source_family"] != "pubchem_aid720551"
                or row["wild_type_evidence_scope"] != "confirmed_wild_type"
                or row["primary_binary_label"] not in {0, 1}
                or not row["q0_consensus_structure_available"]
                or row["primary_binary_label"] != row["q0_consensus_target_class"]
                or not row["primary_training_eligible"]
            ):
                raise HergTrainingSurfaceError("invalid fixed-dose primary label")
            fixed_observation_ids.add(str(row["observation_id"]))
            fixed_labels[str(row["observation_id"])] = int(row["primary_binary_label"])
        elif kind == "native_numeric_relation_preserved":
            if (
                row["source_family"] not in NUMERIC_SOURCE_FAMILIES
                or not _finite(row["primary_numeric_value"])
                or row["primary_numeric_relation"] not in PRIMARY_RELATIONS
                or row["primary_numeric_relation"] != row["native_relation"]
                or row["primary_numeric_unit"] != row["native_unit"]
                or not str(row["primary_numeric_unit"] or "").strip()
                or not row["primary_training_eligible"]
            ):
                raise HergTrainingSurfaceError("invalid native numeric primary label")
        elif kind == "native_numeric_approximate_sensitivity":
            if (
                row["primary_training_eligible"]
                or not row["sensitivity_training_eligible"]
                or row["primary_numeric_relation"] != "~"
                or row["native_relation"] != "~"
            ):
                raise HergTrainingSurfaceError("approximate relation escaped sensitivity-only surface")
        elif row["primary_training_eligible"]:
            raise HergTrainingSurfaceError("eligible observation has no supported label kind")
        if row["standardized_pic50_primary"]:
            relation = row["potency_relation_pic50"]
            if relation not in PRIMARY_RELATIONS:
                raise HergTrainingSurfaceError("standardized primary pIC50 relation is unresolved")
            if relation in {"<", "<="} and (
                row["potency_pic50_point"] is not None
                or row["potency_pic50_lower_bound"] is not None
                or row["potency_pic50_upper_bound"] is None
            ):
                raise HergTrainingSurfaceError("upper-censored pIC50 bound was not preserved")
            if relation in {">", ">="} and (
                row["potency_pic50_point"] is not None
                or row["potency_pic50_lower_bound"] is None
                or row["potency_pic50_upper_bound"] is not None
            ):
                raise HergTrainingSurfaceError("lower-censored pIC50 bound was not preserved")
    if any(len(values) != 1 for values in structure_splits.values()):
        raise HergTrainingSurfaceError("structure crosses model partitions")
    if any(len(values) != 1 for values in scaffold_splits.values()):
        raise HergTrainingSurfaceError("scaffold crosses model partitions")

    structures = pq.read_table(root / STRUCTURE_OUTPUT).to_pylist()
    structure_ids = [str(row["structure_id"]) for row in structures]
    if structure_ids != sorted(structure_ids) or len(structure_ids) != len(set(structure_ids)):
        raise HergTrainingSurfaceError("fixed-dose structure surface is not unique and sorted")
    for row in structures:
        support_ids = [str(value) for value in _json_list(row["support_observation_ids_json"])]
        if (
            row["target_scope"] != "confirmed_wild_type"
            or row["target_class"] not in {0, 1}
            or not row["direct_herg_label"]
            or not row["use_as_training_label"]
            or row["formal_t1_validated"]
            or len(support_ids) != row["support_observation_count"]
            or not set(support_ids).issubset(fixed_observation_ids)
            or any(fixed_labels[item] != int(row["target_class"]) for item in support_ids)
        ):
            raise HergTrainingSurfaceError("invalid confirmed-WT structure label")
        structure_splits[row["structure_id"]].add(row["model_split"])
        scaffold_splits[row["scaffold_group_id"]].add(row["model_split"])
    if any(len(values) != 1 for values in structure_splits.values()) or any(
        len(values) != 1 for values in scaffold_splits.values()
    ):
        raise HergTrainingSurfaceError("structure artifact violates the frozen split")

    candidates = pq.read_table(root / VALIDATION_CANDIDATE_OUTPUT).to_pylist()
    if any(
        row["formal_t1_assigned"]
        or row["model_label_admitted"]
        or row["upstream_lineage_independence_adjudicated"]
        or row["assay_modality_comparability_adjudicated"]
        for row in candidates
    ):
        raise HergTrainingSurfaceError("review candidate was promoted")
    clinical = pq.read_table(root / CLINICAL_OUTPUT).to_pylist()
    if any(row["direct_herg_label"] or row["use_as_training_label"] for row in clinical):
        raise HergTrainingSurfaceError("clinical context promoted to hERG label")
    registry = pq.read_table(root / REGISTRY_OUTPUT).to_pylist()
    registry_ids = [str(row["surface_id"]) for row in registry]
    if registry_ids != sorted(registry_ids) or len(registry_ids) != len(set(registry_ids)):
        raise HergTrainingSurfaceError("surface registry is not unique and sorted")
    formal = next(row for row in registry if row["surface_id"] == SURFACE_FORMAL_T1)
    if any(
        int(formal[field])
        for field in (
            "row_count",
            "usable_measurement_label_count",
            "authorized_training_label_count",
            "formal_validated_label_count",
        )
    ):
        raise HergTrainingSurfaceError("formal T1 surface is not empty")
    computed_surface_counts = _surface_counts(registry)
    if computed_surface_counts != manifest["counts"]["surfaces"]:
        raise HergTrainingSurfaceError("surface registry and manifest counts disagree")
    if len(observations) != int(manifest["counts"]["reported_observations"]):
        raise HergTrainingSurfaceError("reported observation count mismatch")
    if sum(row["primary_training_eligible"] for row in observations) != int(
        manifest["counts"]["primary_eligible_observations"]
    ):
        raise HergTrainingSurfaceError("primary eligible observation count mismatch")
    strata = pq.read_table(root / STRATA_OUTPUT).to_pylist()
    by_surface: Counter[str] = Counter()
    for row in strata:
        by_surface[str(row["surface_id"])] += int(row["observation_count"])
    for surface in OBSERVATION_SURFACES:
        expected = next(row for row in registry if row["surface_id"] == surface)["row_count"]
        if by_surface[surface] != expected:
            raise HergTrainingSurfaceError(f"measurement strata do not cover {surface}")
    exclusions = pq.read_table(root / EXCLUSION_OUTPUT).to_pylist()
    mutant_exclusions = sum(
        row["exclusion_reason"] == "explicit_mutant_or_variant"
        and row["audit_origin"] == "v1_3_master_exclusion_replayed"
        for row in exclusions
    )
    upstream_mutants = int(
        json.loads(roles["v1_3_master_manifest"].read_text(encoding="utf-8"))["counts"][
            "explicit_mutant_exclusions"
        ]
    )
    if (
        mutant_exclusions != int(manifest["counts"]["explicit_mutant_exclusions"])
        or mutant_exclusions != upstream_mutants
    ):
        raise HergTrainingSurfaceError("explicit mutant exclusion accounting mismatch")
    if report_root is not None:
        report_path = _checked_file(report_root.resolve() / REPORT_NAME, suffixes=frozenset({".md"}))
        report_metadata = manifest["report_artifact"]
        if (
            report_path.stat().st_size != int(report_metadata["bytes"])
            or _sha256_file(report_path) != report_metadata["sha256"]
        ):
            raise HergTrainingSurfaceError("report artifact hash mismatch")
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-root", type=Path)
    parser.add_argument("--candidate-adjudication-root", type=Path)
    parser.add_argument("--evidence-tiers-root", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report-root", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.validate_only:
        validate_herg_training_surfaces(args.output_root, report_root=args.report_root)
        return 0
    if (
        args.master_root is None
        or args.candidate_adjudication_root is None
        or args.evidence_tiers_root is None
        or args.report_root is None
    ):
        raise HergTrainingSurfaceError(
            "build mode requires --master-root, --candidate-adjudication-root, "
            "--evidence-tiers-root, and --report-root"
        )
    build_herg_training_surfaces(
        master_root=args.master_root,
        candidate_adjudication_root=args.candidate_adjudication_root,
        evidence_tiers_root=args.evidence_tiers_root,
        output_root=args.output_root,
        report_root=args.report_root,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
