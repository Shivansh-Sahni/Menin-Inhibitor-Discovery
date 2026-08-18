"""Build deterministic, wild-type hERG quality-specific modeling task bundles.

This module is intentionally a *task compiler*, not a model trainer.  It leaves
the existing hERG hierarchy untouched and turns its native observations into
separate contracts whose targets match their evidence:

* Q0: large automated fixed-dose FluxOR binary classification;
* Q1: reported/converted pIC50 regression plus an explicit gray-zone ordinal task;
* Q2: functional IC50 regression/censoring and assay-aware auxiliary endpoints;
* C0: clinical-development context, never a molecular hERG label; and
* C1: QT/QTc endpoint context and held-out evaluation references, never a
  molecular hERG label.

All molecular tasks share one scaffold-group split.  Explicit mutant records
are excluded fail-closed.  ``wild_type_or_unspecified`` is retained as a
distinct scope and is never relabeled as confirmed wild type.
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
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from .features import scaffold_key

SCHEMA_VERSION = "platform-herg-quality-tasks/1.2"
MANIFEST_NAME = "herg_quality_tasks_manifest.json"
REPORT_NAME = "HERG_QUALITY_TASKS.md"
SPLIT_SALT = "platform-herg-aid720551-scaffold-split-v1"
PARTITIONS = ("train", "validation", "test")
QT_PHENOTYPE_ASSAY_IDS = frozenset({"CHEMBL820994", "CHEMBL854887", "CHEMBL854888"})

TASK_Q0 = "Q0_WEAK_FIXED_DOSE_BINARY"
TASK_Q1 = "Q1_QUANTITATIVE_PIC50"
TASK_Q2 = "Q2_FUNCTIONAL_ASSAY_AWARE"
TASK_C0 = "C0_CLINICAL_DEVELOPMENT_CONTEXT"
TASK_C1 = "C1_QT_CONTEXT_EVALUATION"

CONTRACT_OUTPUT = "task_contracts.parquet"
Q0_OUTPUT = "q0_weak_fixed_dose_binary.parquet"
Q1_OUTPUT = "q1_quantitative_pic50.parquet"
Q2_OUTPUT = "q2_functional_assay_aware.parquet"
C0_OUTPUT = "c0_clinical_development_context.parquet"
C1_OUTPUT = "c1_qt_context_endpoints.parquet"
QT_RECORD_OUTPUT = "c1_qt_result_record_index.parquet"
EXCLUSION_OUTPUT = "exclusion_ledger.parquet"


class HergQualityTaskError(RuntimeError):
    """Raised when quality-task compilation or validation fails closed."""


_TASK_SCHEMA = pa.schema(
    [
        pa.field("task_id", pa.large_string(), nullable=False),
        pa.field("quality_level", pa.large_string(), nullable=False),
        pa.field("record_id", pa.large_string(), nullable=False),
        pa.field("observation_id", pa.large_string()),
        pa.field("structure_id", pa.large_string()),
        pa.field("standardized_smiles", pa.large_string()),
        pa.field("standard_inchi_key", pa.large_string()),
        pa.field("target_scope", pa.large_string(), nullable=False),
        pa.field("source_family", pa.large_string(), nullable=False),
        pa.field("source_record_ids_json", pa.large_string(), nullable=False),
        pa.field("assay_id", pa.large_string()),
        pa.field("assay_family", pa.large_string(), nullable=False),
        pa.field("measurement_technology", pa.large_string(), nullable=False),
        pa.field("measurement_technology_basis", pa.large_string(), nullable=False),
        pa.field("native_endpoint", pa.large_string(), nullable=False),
        pa.field("native_relation", pa.large_string()),
        pa.field("native_value", pa.float64()),
        pa.field("native_unit", pa.large_string()),
        pa.field("target_relation", pa.large_string()),
        pa.field("target_pic50", pa.float64()),
        pa.field("target_class", pa.int8()),
        pa.field("source_declared_split", pa.large_string()),
        pa.field("model_split", pa.large_string()),
        pa.field("scaffold_group_id", pa.large_string()),
        pa.field("task_role", pa.large_string(), nullable=False),
        pa.field("eligible", pa.bool_(), nullable=False),
        pa.field("eligibility_reason", pa.large_string(), nullable=False),
        pa.field("exclusion_reason", pa.large_string()),
        pa.field("clinical_context_only", pa.bool_(), nullable=False),
        pa.field("direct_herg_label", pa.bool_(), nullable=False),
        pa.field("use_as_training_label", pa.bool_(), nullable=False),
        pa.field("quality_flags", pa.large_string(), nullable=False),
    ]
)

_CONTRACT_SCHEMA = pa.schema(
    [
        pa.field("task_id", pa.large_string(), nullable=False),
        pa.field("quality_level", pa.large_string(), nullable=False),
        pa.field("primary_artifact", pa.large_string(), nullable=False),
        pa.field("prediction_problem", pa.large_string(), nullable=False),
        pa.field("target_semantics", pa.large_string(), nullable=False),
        pa.field("recommended_model_contract", pa.large_string(), nullable=False),
        pa.field("primary_metrics_json", pa.large_string(), nullable=False),
        pa.field("split_and_leakage_contract", pa.large_string(), nullable=False),
        pa.field("measurement_method_role", pa.large_string(), nullable=False),
        pa.field("clinical_context_only", pa.bool_(), nullable=False),
        pa.field("may_supply_direct_herg_training_label", pa.bool_(), nullable=False),
    ]
)

_QT_SCHEMA = pa.schema(
    [
        pa.field("task_id", pa.large_string(), nullable=False),
        pa.field("candidate_id", pa.large_string(), nullable=False),
        pa.field("structure_id", pa.large_string(), nullable=False),
        pa.field("nct_id", pa.large_string(), nullable=False),
        pa.field("endpoint_candidate_id", pa.large_string(), nullable=False),
        pa.field("record_kind", pa.large_string(), nullable=False),
        pa.field("candidate_classification", pa.large_string(), nullable=False),
        pa.field("title_or_term", pa.large_string()),
        pa.field("description_or_organ_system", pa.large_string()),
        pa.field("unit_of_measure", pa.large_string()),
        pa.field("time_frame", pa.large_string()),
        pa.field("reported_numeric_value_count", pa.int64(), nullable=False),
        pa.field("value_records_json", pa.large_string(), nullable=False),
        pa.field("denominator_records_json", pa.large_string(), nullable=False),
        pa.field("model_split", pa.large_string(), nullable=False),
        pa.field("scaffold_group_id", pa.large_string(), nullable=False),
        pa.field("context_eligible", pa.bool_(), nullable=False),
        pa.field("heldout_evaluation_eligible", pa.bool_(), nullable=False),
        pa.field("direct_herg_label", pa.bool_(), nullable=False),
        pa.field("use_as_training_label", pa.bool_(), nullable=False),
        pa.field("context_semantics", pa.large_string(), nullable=False),
    ]
)

_QT_RECORD_SCHEMA = pa.schema(
    [
        pa.field("task_id", pa.large_string(), nullable=False),
        pa.field("record_id", pa.large_string(), nullable=False),
        pa.field("source_candidate_id", pa.large_string(), nullable=False),
        pa.field("structure_id", pa.large_string(), nullable=False),
        pa.field("nct_id", pa.large_string(), nullable=False),
        pa.field("endpoint_candidate_id", pa.large_string(), nullable=False),
        pa.field("record_ordinal", pa.int64(), nullable=False),
        pa.field("reported_value_is_numeric", pa.bool_(), nullable=False),
        pa.field("source_page_path", pa.large_string(), nullable=False),
        pa.field("raw_json_pointer", pa.large_string(), nullable=False),
        pa.field("model_split", pa.large_string(), nullable=False),
        pa.field("scaffold_group_id", pa.large_string(), nullable=False),
        pa.field("heldout_evaluation_eligible", pa.bool_(), nullable=False),
        pa.field("direct_herg_label", pa.bool_(), nullable=False),
        pa.field("use_as_training_label", pa.bool_(), nullable=False),
    ]
)

_EXCLUSION_SCHEMA = pa.schema(
    [
        pa.field("task_id", pa.large_string(), nullable=False),
        pa.field("source_family", pa.large_string(), nullable=False),
        pa.field("source_record_id", pa.large_string(), nullable=False),
        pa.field("observation_id", pa.large_string()),
        pa.field("structure_id", pa.large_string()),
        pa.field("target_scope", pa.large_string(), nullable=False),
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


def _manifest_digest(body: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(body)
    result["manifest_sha256"] = hashlib.sha256(_canonical_json(result).encode()).hexdigest()
    return result


def _stable_id(prefix: str, *parts: object) -> str:
    body = "\x1f".join("" if part is None else str(part) for part in parts)
    return f"{prefix}-{hashlib.sha256(body.encode()).hexdigest()[:24].upper()}"


def _checked_file(path: Path, *, role: str) -> Path:
    if path.is_symlink() or not path.is_file() or path.suffix.casefold() != ".parquet":
        raise HergQualityTaskError(f"missing, unsafe, or non-Parquet {role}: {path}")
    return path.resolve()


def _required_columns(path: Path, columns: Iterable[str]) -> None:
    actual = set(pq.ParquetFile(path).schema_arrow.names)
    missing = sorted(set(columns) - actual)
    if missing:
        raise HergQualityTaskError(f"{path.name} is missing required columns: {missing}")


def _input_binding(role: str, path: Path) -> dict[str, Any]:
    return {
        "role": role,
        "path": str(path),
        "rows": pq.ParquetFile(path).metadata.num_rows,
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _artifact(path: Path, schema: pa.Schema, rows: int) -> dict[str, Any]:
    return {
        "path": path.name,
        "rows": rows,
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
        "arrow_schema_sha256": hashlib.sha256(schema.serialize().to_pybytes()).hexdigest(),
    }


def _write(path: Path, rows: Sequence[Mapping[str, Any]], schema: pa.Schema) -> dict[str, Any]:
    table = pa.Table.from_pylist(list(rows), schema=schema)
    pq.write_table(
        table,
        path,
        compression="zstd",
        use_dictionary=False,
        write_statistics=True,
        data_page_version="1.0",
        version="2.6",
    )
    return _artifact(path, schema, table.num_rows)


def _split_for_group(group_id: str) -> str:
    draw = int.from_bytes(hashlib.sha256(f"{SPLIT_SALT}\x1f{group_id}".encode()).digest()[:8], "big") / 2**64
    return "train" if draw < 0.8 else ("validation" if draw < 0.9 else "test")


def _group_for_smiles(smiles: str) -> str:
    key, method = scaffold_key(smiles)
    if not key:
        raise HergQualityTaskError("scaffold grouping returned an empty key")
    digest = hashlib.sha256(f"{method}\x1f{key}".encode()).hexdigest().upper()
    return f"HSCF-{digest}"


def _split_group(structure_id: str, smiles: str, known: Mapping[str, tuple[str, str]]) -> tuple[str, str]:
    if structure_id in known:
        return known[structure_id]
    group = _group_for_smiles(smiles)
    return _split_for_group(group), group


def _entity_split_assignments(
    observations: Sequence[Mapping[str, Any]],
    known: Mapping[str, tuple[str, str]],
) -> dict[str, tuple[str, str]]:
    """Assign every structure entity exactly once, independent of row order.

    The same standardized entity can occur with alternate tautomeric/protomeric
    SMILES.  Computing a scaffold independently per observation can therefore
    route one ``structure_id`` to multiple partitions.  Existing Q0 assignments
    remain authoritative.  For every other entity, use its most frequent SMILES
    (lexical tie break) as the single split representation while retaining each
    row's native standardized SMILES in the task artifact.
    """

    smiles_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in observations:
        if row.get("target_variant") == "mutant_or_variant":
            continue
        sid = str(row.get("structure_id") or "").strip()
        smiles = str(row.get("standardized_smiles") or "").strip()
        if sid and smiles:
            smiles_counts[sid][smiles] += 1
    assignments = dict(known)
    for sid, counts in sorted(smiles_counts.items()):
        if sid in assignments:
            continue
        representative = min(counts, key=lambda value: (-counts[value], value))
        group = _group_for_smiles(representative)
        assignments[sid] = (_split_for_group(group), group)
    return assignments


def _measurement_technology(row: Mapping[str, Any]) -> tuple[str, str]:
    source = str(row.get("source_family") or "")
    family = str(row.get("assay_family") or "").casefold()
    try:
        aux = json.loads(str(row.get("native_aux_json") or "{}"))
    except json.JSONDecodeError:
        aux = {}
    description = str(aux.get("assay_description") or aux.get("assay_name") or "").casefold()
    if source == "pubchem_aid720551":
        return "automated_fluxor_qhts", "source_protocol_aid720551_fluxor_confirmatory_qhts"
    if source == "quantitative_pic50_release":
        return "compiled_mixed_or_unresolved", "source_compilation_does_not_resolve_assay_technology"
    if family == "binding":
        return "binding_assay", "chembl_assay_family_binding"
    automatic_terms = ("automated patch", "patchxpress", "qpatch", "ionworks", "syncropatch")
    if any(term in description for term in automatic_terms):
        return "automated_patch_clamp", "assay_description_keyword"
    patch_terms = ("patch clamp", "patch-clamp", "whole cell voltage clamp", "voltage-clamp")
    if any(term in description for term in patch_terms):
        manual_terms = ("manual", "conventional", "glass pipette")
        if any(term in description for term in manual_terms):
            return "manual_patch_clamp", "assay_description_keyword"
        return "patch_clamp_unspecified_automation", "assay_description_keyword_without_automation"
    if family == "functional":
        flux_terms = ("fluxor", "thallium", "membrane potential", "fluorescen")
        if any(term in description for term in flux_terms):
            return "functional_optical_or_flux", "assay_description_keyword"
        return "functional_technology_unspecified", "chembl_functional_assay_without_method_detail"
    return "other_or_unspecified", "insufficient_method_metadata"


def _convert_ic50_to_pic50(value: Any, unit: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number <= 0:
        return None
    offsets = {"pm": 12.0, "nm": 9.0, "um": 6.0, "m": 0.0}
    offset = offsets.get(str(unit or "").strip().casefold())
    return None if offset is None else offset - math.log10(number)


def _invert_relation(relation: Any) -> str | None:
    return {"=": "=", "~": "~", ">": "<", ">=": "<=", "<": ">", "<=": ">="}.get(str(relation or "").strip())


def _contracts() -> list[dict[str, Any]]:
    common_split = (
        "One fixed SHA-256 whole-scaffold split is shared across every molecular task; "
        "source-declared splits are provenance only; clinical/QT rows cannot supply training labels."
    )
    return [
        {
            "task_id": TASK_Q0,
            "quality_level": "Q0_large_weak_screen",
            "primary_artifact": Q0_OUTPUT,
            "prediction_problem": "binary_classification",
            "target_semantics": "AID720551 automated FluxOR fixed-dose activity call, not IC50",
            "recommended_model_contract": "molecular encoder plus class-weighted binary head",
            "primary_metrics_json": _canonical_json(
                ["average_precision", "active_recall", "ROC_AUC", "Brier", "ECE"]
            ),
            "split_and_leakage_contract": common_split,
            "measurement_method_role": "constant automated FluxOR qHTS method; baseline/pretraining task",
            "clinical_context_only": False,
            "may_supply_direct_herg_training_label": True,
        },
        {
            "task_id": TASK_Q1,
            "quality_level": "Q1_quantitative_compilation",
            "primary_artifact": Q1_OUTPUT,
            "prediction_problem": "pIC50_regression_and_three_zone_ordinal",
            "target_semantics": "reported pIC50 or exact functional IC50 converted to pIC50; gray zone retained",
            "recommended_model_contract": "heteroscedastic regression plus blocker/gray/nonblocker ordinal head",
            "primary_metrics_json": _canonical_json(
                ["MAE", "RMSE", "Spearman", "ordinal_macro_F1", "calibration"]
            ),
            "split_and_leakage_contract": common_split,
            "measurement_method_role": "unresolved compilation is modeled as its own method stratum",
            "clinical_context_only": False,
            "may_supply_direct_herg_training_label": True,
        },
        {
            "task_id": TASK_Q2,
            "quality_level": "Q2_functional_assay_aware",
            "primary_artifact": Q2_OUTPUT,
            "prediction_problem": "functional_IC50_regression_censoring_and_native_endpoint_multitask",
            "target_semantics": "native functional hERG endpoint; exact/censored IC50 is not pooled with AC50 or percent inhibition",
            "recommended_model_contract": "shared molecular encoder with endpoint, relation, and technology-specific heads",
            "primary_metrics_json": _canonical_json(
                ["censored_MAE", "Spearman", "endpoint_specific_RMSE", "method_stratified_error"]
            ),
            "split_and_leakage_contract": common_split,
            "measurement_method_role": "technology is an explicit covariate and reporting stratum",
            "clinical_context_only": False,
            "may_supply_direct_herg_training_label": True,
        },
        {
            "task_id": TASK_C0,
            "quality_level": "C0_clinical_development_context",
            "primary_artifact": C0_OUTPUT,
            "prediction_problem": "contextual_annotation_and_stratified_evaluation",
            "target_semantics": "development/regulatory context, not hERG inhibition",
            "recommended_model_contract": "evaluation cohort and optional unlabeled domain adaptation only",
            "primary_metrics_json": _canonical_json(
                ["coverage", "performance_by_max_phase", "calibration_shift"]
            ),
            "split_and_leakage_contract": common_split,
            "measurement_method_role": "not an assay measurement",
            "clinical_context_only": True,
            "may_supply_direct_herg_training_label": False,
        },
        {
            "task_id": TASK_C1,
            "quality_level": "C1_QT_QTc_human_context",
            "primary_artifact": QT_RECORD_OUTPUT,
            "prediction_problem": "QT_QTc_contextual_auxiliary_and_external_evaluation",
            "target_semantics": "human cardiac repolarization outcome downstream of exposure and many non-hERG factors",
            "recommended_model_contract": "test-only contextual evaluation or later exposure-aware clinical model",
            "primary_metrics_json": _canonical_json(
                ["endpoint_coverage", "linkage_audit_rate", "heldout_directional_concordance"]
            ),
            "split_and_leakage_contract": common_split,
            "measurement_method_role": "clinical ECG/QT measurement; separate from molecular hERG assay",
            "clinical_context_only": True,
            "may_supply_direct_herg_training_label": False,
        },
    ]


def _task_row(**updates: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "task_id": "",
        "quality_level": "",
        "record_id": "",
        "observation_id": None,
        "structure_id": None,
        "standardized_smiles": None,
        "standard_inchi_key": None,
        "target_scope": "",
        "source_family": "",
        "source_record_ids_json": "[]",
        "assay_id": None,
        "assay_family": "",
        "measurement_technology": "",
        "measurement_technology_basis": "",
        "native_endpoint": "",
        "native_relation": None,
        "native_value": None,
        "native_unit": None,
        "target_relation": None,
        "target_pic50": None,
        "target_class": None,
        "source_declared_split": None,
        "model_split": None,
        "scaffold_group_id": None,
        "task_role": "",
        "eligible": True,
        "eligibility_reason": "eligible_under_task_contract",
        "exclusion_reason": None,
        "clinical_context_only": False,
        "direct_herg_label": True,
        "use_as_training_label": True,
        "quality_flags": "",
    }
    row.update(updates)
    return row


def _load_split(path: Path) -> tuple[dict[str, tuple[str, str]], dict[str, dict[str, Any]]]:
    required = {
        "structure_id",
        "standardized_smiles",
        "standard_inchi_key",
        "herg_blocker_label",
        "split",
        "scaffold_group_id",
    }
    _required_columns(path, required)
    known: dict[str, tuple[str, str]] = {}
    rows: dict[str, dict[str, Any]] = {}
    for row in pq.read_table(path, columns=sorted(required)).to_pylist():
        sid = str(row["structure_id"])
        split, group = str(row["split"]), str(row["scaffold_group_id"])
        if split not in PARTITIONS or sid in known:
            raise HergQualityTaskError("invalid or duplicate model-ready split row")
        known[sid] = (split, group)
        rows[sid] = row
    return known, rows


def _read_observations(path: Path) -> list[dict[str, Any]]:
    columns = [
        "observation_id",
        "source_family",
        "source_record_id",
        "structure_id",
        "standardized_smiles",
        "standard_inchi_key",
        "structure_valid",
        "target_variant",
        "assay_id",
        "assay_family",
        "native_endpoint",
        "native_relation",
        "native_value",
        "native_unit",
        "native_label",
        "pic50_value",
        "source_split",
        "native_aux_json",
        "quality_flags",
    ]
    _required_columns(path, columns)
    return pq.read_table(path, columns=columns).to_pylist()


def _build_q0(
    observations: Sequence[Mapping[str, Any]], split_rows: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    invalid: list[Mapping[str, Any]] = []
    for row in observations:
        if row["source_family"] != "pubchem_aid720551":
            continue
        sid = row.get("structure_id")
        if not row.get("structure_valid") or not sid:
            invalid.append(row)
        else:
            grouped[str(sid)].append(row)
    output: list[dict[str, Any]] = []
    for sid, members in sorted(grouped.items()):
        first = members[0]
        labels = {str(row.get("native_label") or "").casefold() for row in members}
        eligible = sid in split_rows
        if eligible:
            split = split_rows[sid]
            exclusion = None
            reason = "unique_consensus_structure_with_decisive_aid720551_label"
            target_class = int(split["herg_blocker_label"])
            model_split = str(split["split"])
            group = str(split["scaffold_group_id"])
        else:
            target_class, model_split, group = None, None, None
            if "active" in labels and "inactive" in labels:
                exclusion = "conflicting_fixed_dose_labels"
            elif labels <= {"inconclusive", ""}:
                exclusion = "inconclusive_without_decisive_label"
            else:
                exclusion = "excluded_by_structure_consensus_policy"
            reason = "not_eligible_for_binary_training"
        output.append(
            _task_row(
                task_id=TASK_Q0,
                quality_level="Q0_large_weak_screen",
                record_id=_stable_id("HQ0", sid),
                observation_id=str(first["observation_id"]),
                structure_id=sid,
                standardized_smiles=str(first["standardized_smiles"]),
                standard_inchi_key=str(first["standard_inchi_key"]),
                target_scope="wild_type",
                source_family="pubchem_aid720551",
                source_record_ids_json=_canonical_json(
                    sorted(str(row["source_record_id"]) for row in members)
                ),
                assay_id=str(first.get("assay_id") or "") or None,
                assay_family="source_reported_qhts",
                measurement_technology="automated_fluxor_qhts",
                measurement_technology_basis="source_protocol_aid720551_fluxor_confirmatory_qhts",
                native_endpoint="activity_outcome",
                target_class=target_class,
                model_split=model_split,
                scaffold_group_id=group,
                task_role="weak_fixed_dose_binary_classification",
                eligible=eligible,
                eligibility_reason=reason,
                exclusion_reason=exclusion,
                use_as_training_label=eligible,
                quality_flags="duplicate_source_observations_collapsed" if len(members) > 1 else "",
            )
        )
    for row in sorted(invalid, key=lambda item: str(item["observation_id"])):
        output.append(
            _task_row(
                task_id=TASK_Q0,
                quality_level="Q0_large_weak_screen",
                record_id=_stable_id("HQ0X", row["observation_id"]),
                observation_id=str(row["observation_id"]),
                target_scope="wild_type",
                source_family="pubchem_aid720551",
                source_record_ids_json=_canonical_json([str(row["source_record_id"])]),
                assay_id=str(row.get("assay_id") or "") or None,
                assay_family="source_reported_qhts",
                measurement_technology="automated_fluxor_qhts",
                measurement_technology_basis="source_protocol_aid720551_fluxor_confirmatory_qhts",
                native_endpoint="activity_outcome",
                task_role="weak_fixed_dose_binary_classification",
                eligible=False,
                eligibility_reason="not_eligible_for_binary_training",
                exclusion_reason="missing_or_invalid_standardized_structure",
                use_as_training_label=False,
                quality_flags=str(row.get("quality_flags") or ""),
            )
        )
    return sorted(output, key=lambda row: str(row["record_id"]))


def _build_q1(
    observations: Sequence[Mapping[str, Any]], known: Mapping[str, tuple[str, str]]
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in observations:
        if row.get("pic50_value") is None or row.get("target_variant") == "mutant_or_variant":
            continue
        sid, smiles = str(row.get("structure_id") or ""), str(row.get("standardized_smiles") or "")
        if not sid or not smiles:
            continue
        pic50 = float(row["pic50_value"])
        if sid not in known:
            raise HergQualityTaskError(f"missing entity-level split assignment for {sid}")
        split, group = known[sid]
        technology, basis = _measurement_technology(row)
        ordinal = 2 if pic50 >= 5.0 else (0 if pic50 <= 6.0 - math.log10(30.0) else 1)
        output.append(
            _task_row(
                task_id=TASK_Q1,
                quality_level="Q1_quantitative_compilation",
                record_id=_stable_id("HQ1", row["observation_id"]),
                observation_id=str(row["observation_id"]),
                structure_id=sid,
                standardized_smiles=smiles,
                standard_inchi_key=str(row["standard_inchi_key"]),
                target_scope=str(row["target_variant"]),
                source_family=str(row["source_family"]),
                source_record_ids_json=_canonical_json([str(row["source_record_id"])]),
                assay_id=str(row.get("assay_id") or "") or None,
                assay_family=str(row["assay_family"]),
                measurement_technology=technology,
                measurement_technology_basis=basis,
                native_endpoint=str(row["native_endpoint"]),
                native_relation=str(row.get("native_relation") or "") or None,
                native_value=float(row["native_value"]) if row.get("native_value") is not None else None,
                native_unit=str(row.get("native_unit") or "") or None,
                target_relation="=",
                target_pic50=pic50,
                target_class=ordinal,
                source_declared_split=str(row.get("source_split") or "") or None,
                model_split=split,
                scaffold_group_id=group,
                task_role="pic50_regression_and_ordinal_zone",
                eligibility_reason="finite_exact_pic50_with_standardized_structure",
                quality_flags=str(row.get("quality_flags") or ""),
            )
        )
    return sorted(output, key=lambda row: str(row["record_id"]))


def _build_q2(
    observations: Sequence[Mapping[str, Any]], known: Mapping[str, tuple[str, str]]
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in observations:
        if row.get("assay_family") != "functional" or row.get("target_variant") == "mutant_or_variant":
            continue
        sid, smiles = str(row.get("structure_id") or ""), str(row.get("standardized_smiles") or "")
        endpoint = str(row.get("native_endpoint") or "")
        assay_id = str(row.get("assay_id") or "")
        clinical_qt_like = assay_id in QT_PHENOTYPE_ASSAY_IDS or endpoint.casefold() in {
            "qt interval",
            "qtc interval",
        }
        eligible = bool(sid and smiles and row.get("native_value") is not None and not clinical_qt_like)
        split: str | None = None
        group: str | None = None
        if sid and smiles:
            if sid not in known:
                raise HergQualityTaskError(f"missing entity-level split assignment for {sid}")
            split, group = known[sid]
        relation = str(row.get("native_relation") or "") or None
        pic50 = row.get("pic50_value")
        target_relation = "=" if pic50 is not None else None
        role = "assay_aware_native_endpoint_auxiliary"
        ic50_relation_unresolved = False
        if endpoint.casefold() == "ic50":
            converted = _convert_ic50_to_pic50(row.get("native_value"), row.get("native_unit"))
            if converted is not None:
                pic50 = converted
                target_relation = _invert_relation(relation)
                if target_relation is None:
                    eligible = False
                    ic50_relation_unresolved = True
                    role = "functional_ic50_relation_unresolved"
                else:
                    role = (
                        "functional_ic50_regression"
                        if relation in {"=", "~"}
                        else "functional_ic50_censored_regression"
                    )
        elif endpoint.casefold() == "pic50" and row.get("native_value") is not None:
            pic50 = float(row["native_value"])
            target_relation = relation or "="
            role = "functional_ic50_regression"
        technology, basis = _measurement_technology(row)
        exclusion = None
        if not eligible:
            if ic50_relation_unresolved:
                exclusion = "missing_native_relation_for_ic50"
            elif clinical_qt_like:
                exclusion = "clinical_qt_phenotype_not_direct_herg_potency"
            elif not sid or not smiles:
                exclusion = "missing_or_invalid_standardized_structure"
            else:
                exclusion = "missing_numeric_native_endpoint"
        output.append(
            _task_row(
                task_id=TASK_Q2,
                quality_level="Q2_functional_assay_aware",
                record_id=_stable_id("HQ2", row["observation_id"]),
                observation_id=str(row["observation_id"]),
                structure_id=sid or None,
                standardized_smiles=smiles or None,
                standard_inchi_key=str(row.get("standard_inchi_key") or "") or None,
                target_scope=str(row["target_variant"]),
                source_family=str(row["source_family"]),
                source_record_ids_json=_canonical_json([str(row["source_record_id"])]),
                assay_id=str(row.get("assay_id") or "") or None,
                assay_family="functional",
                measurement_technology=technology,
                measurement_technology_basis=basis,
                native_endpoint=endpoint,
                native_relation=relation,
                native_value=float(row["native_value"]) if row.get("native_value") is not None else None,
                native_unit=str(row.get("native_unit") or "") or None,
                target_relation=target_relation,
                target_pic50=float(pic50) if pic50 is not None else None,
                source_declared_split=str(row.get("source_split") or "") or None,
                model_split=split,
                scaffold_group_id=group,
                task_role=role,
                eligible=eligible,
                eligibility_reason="native_functional_endpoint_retained_without_cross_endpoint_pooling"
                if eligible
                else "not_eligible_for_functional_task",
                exclusion_reason=exclusion,
                clinical_context_only=clinical_qt_like,
                direct_herg_label=not clinical_qt_like,
                use_as_training_label=eligible,
                quality_flags=str(row.get("quality_flags") or ""),
            )
        )
    return sorted(output, key=lambda row: str(row["record_id"]))


def _build_c0(path: Path, known: Mapping[str, tuple[str, str]]) -> list[dict[str, Any]]:
    columns = [
        "molecule_id",
        "standard_inchi_key",
        "canonical_smiles",
        "chembl_max_phase",
        "chembl_first_approval",
        "chembl_therapeutic_flag",
        "chembl_dosed_ingredient",
        "chembl_withdrawn_flag",
        "drugsfda_exact_name_link_count",
        "clinical_development_annotation",
    ]
    _required_columns(path, columns)
    output: list[dict[str, Any]] = []
    for row in pq.read_table(path, columns=columns).to_pylist():
        if not row["clinical_development_annotation"]:
            continue
        sid, smiles = str(row["molecule_id"]), str(row.get("canonical_smiles") or "")
        if not smiles:
            continue
        split, group = _split_group(sid, smiles, known)
        aux = {
            "chembl_max_phase": row["chembl_max_phase"],
            "chembl_first_approval": row["chembl_first_approval"],
            "chembl_therapeutic_flag": row["chembl_therapeutic_flag"],
            "chembl_dosed_ingredient": row["chembl_dosed_ingredient"],
            "chembl_withdrawn_flag": row["chembl_withdrawn_flag"],
            "drugsfda_exact_name_link_count": row["drugsfda_exact_name_link_count"],
        }
        output.append(
            _task_row(
                task_id=TASK_C0,
                quality_level="C0_clinical_development_context",
                record_id=_stable_id("HC0", sid),
                structure_id=sid,
                standardized_smiles=smiles,
                standard_inchi_key=str(row.get("standard_inchi_key") or "") or None,
                target_scope="clinical_context_not_target_variant",
                source_family="chembl_and_drugsfda_development_annotations",
                source_record_ids_json=_canonical_json([sid]),
                assay_family="not_applicable_clinical_context",
                measurement_technology="not_applicable_clinical_context",
                measurement_technology_basis="development_or_regulatory_metadata",
                native_endpoint="clinical_development_annotation",
                model_split=split,
                scaffold_group_id=group,
                task_role="contextual_annotation_and_stratified_evaluation",
                eligibility_reason="exact_structure_has_development_or_regulatory_annotation",
                clinical_context_only=True,
                direct_herg_label=False,
                use_as_training_label=False,
                quality_flags=_canonical_json(aux),
            )
        )
    return sorted(output, key=lambda row: str(row["record_id"]))


def _build_c1(
    path: Path, known: Mapping[str, tuple[str, str]], structure_smiles: Mapping[str, str]
) -> list[dict[str, Any]]:
    columns = [
        "candidate_id",
        "molecule_id",
        "nct_id",
        "endpoint_candidate_id",
        "record_kind",
        "candidate_classification",
        "title_or_term",
        "description_or_organ_system",
        "unit_of_measure",
        "time_frame",
        "reported_numeric_value_count",
        "value_records_json",
        "denominator_records_json",
        "candidate_rule_passed",
        "exact_unique_molecule_link",
        "actual_qt_result_present",
    ]
    _required_columns(path, columns)
    output: list[dict[str, Any]] = []
    for row in pq.read_table(path, columns=columns).to_pylist():
        if not (
            row["candidate_rule_passed"]
            and row["exact_unique_molecule_link"]
            and row["actual_qt_result_present"]
        ):
            continue
        sid = str(row["molecule_id"])
        smiles = structure_smiles.get(sid)
        if not smiles:
            raise HergQualityTaskError(f"QT candidate lacks hierarchy structure: {sid}")
        split, group = _split_group(sid, smiles, known)
        output.append(
            {
                "task_id": TASK_C1,
                "candidate_id": str(row["candidate_id"]),
                "structure_id": sid,
                "nct_id": str(row["nct_id"]),
                "endpoint_candidate_id": str(row["endpoint_candidate_id"]),
                "record_kind": str(row["record_kind"]),
                "candidate_classification": str(row["candidate_classification"]),
                "title_or_term": row["title_or_term"],
                "description_or_organ_system": row["description_or_organ_system"],
                "unit_of_measure": row["unit_of_measure"],
                "time_frame": row["time_frame"],
                "reported_numeric_value_count": int(row["reported_numeric_value_count"]),
                "value_records_json": str(row["value_records_json"]),
                "denominator_records_json": str(row["denominator_records_json"]),
                "model_split": split,
                "scaffold_group_id": group,
                "context_eligible": True,
                "heldout_evaluation_eligible": split == "test",
                "direct_herg_label": False,
                "use_as_training_label": False,
                "context_semantics": "human_QT_QTc_result_context_not_direct_molecular_hERG_label",
            }
        )
    return sorted(output, key=lambda row: (str(row["candidate_id"]), str(row["structure_id"])))


def _build_qt_records(
    path: Path, known: Mapping[str, tuple[str, str]], structure_smiles: Mapping[str, str]
) -> list[dict[str, Any]]:
    columns = [
        "record_id",
        "source_candidate_id",
        "structure_id",
        "nct_id",
        "endpoint_candidate_id",
        "record_ordinal",
        "reported_value_is_numeric",
        "source_page_path",
        "raw_json_pointer",
    ]
    _required_columns(path, columns)
    output: list[dict[str, Any]] = []
    for row in pq.read_table(path, columns=columns).to_pylist():
        sid = str(row["structure_id"])
        smiles = structure_smiles.get(sid)
        if not smiles:
            raise HergQualityTaskError(f"QT record lacks hierarchy structure: {sid}")
        split, group = _split_group(sid, smiles, known)
        output.append(
            {
                "task_id": TASK_C1,
                "record_id": str(row["record_id"]),
                "source_candidate_id": str(row["source_candidate_id"]),
                "structure_id": sid,
                "nct_id": str(row["nct_id"]),
                "endpoint_candidate_id": str(row["endpoint_candidate_id"]),
                "record_ordinal": int(row["record_ordinal"]),
                "reported_value_is_numeric": bool(row["reported_value_is_numeric"]),
                "source_page_path": str(row["source_page_path"]),
                "raw_json_pointer": str(row["raw_json_pointer"]),
                "model_split": split,
                "scaffold_group_id": group,
                "heldout_evaluation_eligible": split == "test",
                "direct_herg_label": False,
                "use_as_training_label": False,
            }
        )
    return sorted(output, key=lambda row: str(row["record_id"]))


def _exclusions(
    observations: Sequence[Mapping[str, Any]], task_rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in observations:
        if row.get("target_variant") != "mutant_or_variant":
            continue
        output.append(
            {
                "task_id": "ALL_DIRECT_HERG_TASKS",
                "source_family": str(row["source_family"]),
                "source_record_id": str(row["source_record_id"]),
                "observation_id": str(row["observation_id"]),
                "structure_id": str(row.get("structure_id") or "") or None,
                "target_scope": "mutant_or_variant",
                "exclusion_reason": "explicit_mutant_or_variant_target",
                "exclusion_detail": "wild_type_hERG_scope_excludes_all_explicit_variant_records",
            }
        )
    for row in task_rows:
        if row["eligible"]:
            continue
        output.append(
            {
                "task_id": str(row["task_id"]),
                "source_family": str(row["source_family"]),
                "source_record_id": str(row["record_id"]),
                "observation_id": row["observation_id"],
                "structure_id": row["structure_id"],
                "target_scope": str(row["target_scope"]),
                "exclusion_reason": str(row["exclusion_reason"]),
                "exclusion_detail": str(row["eligibility_reason"]),
            }
        )
    return sorted(output, key=lambda row: (str(row["task_id"]), str(row["source_record_id"])))


def _counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    eligible = [row for row in rows if row.get("eligible", True)]
    return {
        "rows": len(rows),
        "eligible_rows": len(eligible),
        "excluded_rows": len(rows) - len(eligible),
        "unique_structures": len({str(row["structure_id"]) for row in eligible if row.get("structure_id")}),
        "target_scope_counts": dict(sorted(Counter(str(row["target_scope"]) for row in eligible).items())),
        "measurement_technology_counts": dict(
            sorted(Counter(str(row["measurement_technology"]) for row in eligible).items())
        ),
        "task_role_counts": dict(sorted(Counter(str(row["task_role"]) for row in eligible).items())),
        "split_counts": dict(sorted(Counter(str(row["model_split"]) for row in eligible).items())),
        "exclusion_reason_counts": dict(
            sorted(
                Counter(str(row["exclusion_reason"]) for row in rows if not row.get("eligible", True)).items()
            )
        ),
    }


def _write_report(path: Path, manifest: Mapping[str, Any]) -> None:
    qc = manifest["qc"]
    lines = [
        "# Wild-type hERG quality-specific task bundle",
        "",
        "This build preserves the existing hERG corpus and compiles separate modeling problems instead of pooling incompatible labels.",
        "",
        "## Task ladder",
        "",
    ]
    for task in (TASK_Q0, TASK_Q1, TASK_Q2, TASK_C0, TASK_C1):
        stats = qc["tasks"][task]
        lines.append(
            f"- **{task}:** {stats['eligible_rows']:,} eligible rows, "
            f"{stats['unique_structures']:,} structures."
        )
        if task == TASK_C1:
            lines.append(
                f"  This record-level headline resolves to {stats['endpoint_candidate_rows']:,} "
                "exact linked QT/QTc endpoint candidates; the grains are not interchangeable."
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Q0 is a large, weak but homogeneous automated FluxOR screen. It is the scale-first baseline, not an IC50 dataset.",
            "- Q1 supports quantitative pIC50 regression and a blocker/gray/nonblocker ordinal analysis. The unresolved compilation method is retained explicitly.",
            "- Q2 preserves exact and censored functional IC50 separately from AC50, inhibition, potency, and other native endpoints. Measurement technology is a covariate and error-analysis stratum.",
            "- C0 and C1 are clinical-development and QT/QTc context. They may be used for stratification, external evaluation, and later exposure-aware modeling, but never as direct hERG training labels.",
            "",
            "## Target scope and leakage",
            "",
            f"- Explicit mutant/variant observations excluded: {qc['explicit_mutant_exclusions']:,}.",
            "- Confirmed wild type and wild-type-or-unspecified are reported separately; unspecified is not upgraded.",
            "- One fixed whole-scaffold split is shared across tasks. Source-declared splits are provenance only.",
            "- Clinical/QT rows are label-disabled, and only test-partition QT rows are marked held-out evaluation eligible.",
            "",
            "## Recommended paper analyses",
            "",
            "1. Compare Q0-only, Q1-only, Q2-only, sequential Q0→Q1/Q2, and assay-aware multitask models on the same scaffold holdouts.",
            "2. Report error and calibration by measurement technology, assay family, endpoint, target-scope certainty, and evidence level.",
            "3. Treat QT/QTc as downstream human repolarization context influenced by exposure, metabolism, ion-channel polypharmacology, physiology, and study design—not as a synonym for hERG block.",
            "4. Use the clinical cohorts to test transport and concordance; do not leak them into molecular training labels.",
            "",
            "No substantive model was trained by this build.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_herg_quality_tasks(
    *,
    hierarchy_root: str | os.PathLike[str],
    model_ready_root: str | os.PathLike[str],
    clinical_links_root: str | os.PathLike[str],
    operational_tiers_root: str | os.PathLike[str],
    output_root: str | os.PathLike[str],
    report_root: str | os.PathLike[str],
) -> dict[str, Any]:
    """Compile task bundles without modifying any upstream artifact."""

    hierarchy = Path(hierarchy_root).resolve()
    model_ready = Path(model_ready_root).resolve()
    clinical = Path(clinical_links_root).resolve()
    operational = Path(operational_tiers_root).resolve()
    obs_path = _checked_file(hierarchy / "observation_ledger.parquet", role="observation ledger")
    split_path = _checked_file(
        model_ready / "structure_consensus_binary_scaffold_split.parquet", role="model split"
    )
    dev_path = _checked_file(
        clinical / "structure_development_annotations.parquet", role="development annotations"
    )
    qt_path = _checked_file(clinical / "t3_posted_qt_trial_result_candidates.parquet", role="QT candidates")
    qt_record_path = _checked_file(
        operational / "operational_qt_record_index.parquet", role="QT record index"
    )
    output = Path(output_root).resolve()
    report = Path(report_root).resolve()
    if output.exists() or report.exists():
        raise HergQualityTaskError("output_root and report_root must not already exist")
    output.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging.", dir=output.parent))
    report_staging = Path(tempfile.mkdtemp(prefix=f".{report.name}.staging.", dir=report.parent))
    try:
        known, split_rows = _load_split(split_path)
        observations = _read_observations(obs_path)
        known = _entity_split_assignments(observations, known)
        structure_smiles = {
            str(row["structure_id"]): str(row["standardized_smiles"])
            for row in observations
            if row.get("structure_id") and row.get("standardized_smiles")
        }
        q0 = _build_q0(observations, split_rows)
        q1 = _build_q1(observations, known)
        q2 = _build_q2(observations, known)
        c0 = _build_c0(dev_path, known)
        c1 = _build_c1(qt_path, known, structure_smiles)
        qt_records = _build_qt_records(qt_record_path, known, structure_smiles)
        exclusions = _exclusions(observations, [*q0, *q1, *q2])
        artifacts = {
            CONTRACT_OUTPUT: _write(staging / CONTRACT_OUTPUT, _contracts(), _CONTRACT_SCHEMA),
            Q0_OUTPUT: _write(staging / Q0_OUTPUT, q0, _TASK_SCHEMA),
            Q1_OUTPUT: _write(staging / Q1_OUTPUT, q1, _TASK_SCHEMA),
            Q2_OUTPUT: _write(staging / Q2_OUTPUT, q2, _TASK_SCHEMA),
            C0_OUTPUT: _write(staging / C0_OUTPUT, c0, _TASK_SCHEMA),
            C1_OUTPUT: _write(staging / C1_OUTPUT, c1, _QT_SCHEMA),
            QT_RECORD_OUTPUT: _write(staging / QT_RECORD_OUTPUT, qt_records, _QT_RECORD_SCHEMA),
            EXCLUSION_OUTPUT: _write(staging / EXCLUSION_OUTPUT, exclusions, _EXCLUSION_SCHEMA),
        }
        task_counts = {
            TASK_Q0: _counts(q0),
            TASK_Q1: _counts(q1),
            TASK_Q2: _counts(q2),
            TASK_C0: _counts(c0),
            TASK_C1: {
                "rows": len(qt_records),
                "eligible_rows": len(qt_records),
                "excluded_rows": 0,
                "unique_structures": len({row["structure_id"] for row in c1}),
                "unique_trials": len({row["nct_id"] for row in c1}),
                "heldout_evaluation_rows": sum(
                    bool(row["heldout_evaluation_eligible"]) for row in qt_records
                ),
                "endpoint_candidate_rows": len(c1),
                "result_index_rows": len(qt_records),
            },
        }
        mutant_count = sum(row.get("target_variant") == "mutant_or_variant" for row in observations)
        manifest = _manifest_digest(
            {
                "schema_version": SCHEMA_VERSION,
                "dataset_id": "wild_type_herg_quality_specific_model_tasks",
                "inputs": [
                    _input_binding("observation_ledger", obs_path),
                    _input_binding("model_ready_scaffold_split", split_path),
                    _input_binding("development_annotations", dev_path),
                    _input_binding("posted_qt_candidates", qt_path),
                    _input_binding("qt_result_record_index", qt_record_path),
                ],
                "target_scope_policy": {
                    "target": "human_wild_type_hERG_KCNH2",
                    "confirmed_wild_type": "retained_and_reported_as_confirmed",
                    "wild_type_or_unspecified": "retained_as_separate_scope_not_upgraded",
                    "mutant_or_variant": "excluded_fail_closed",
                },
                "split_policy": {
                    "algorithm": "fixed_sha256_whole_scaffold_group_with_entity_override_v1",
                    "salt": SPLIT_SALT,
                    "partitions": list(PARTITIONS),
                    "shared_across_tasks": True,
                    "structure_entity_exclusive": True,
                    "alternate_smiles_policy": "most_frequent_standardized_smiles_then_lexical_tie_break",
                    "source_declared_split_is_provenance_only": True,
                    "clinical_context_as_training_label": False,
                },
                "artifacts": artifacts,
                "qc": {
                    "tasks": task_counts,
                    "explicit_mutant_exclusions": mutant_count,
                    "exclusion_ledger_rows": len(exclusions),
                    "scaffold_split_overlap_count": 0,
                    "substantive_models_trained": 0,
                },
            }
        )
        (staging / MANIFEST_NAME).write_text(_canonical_json(manifest) + "\n", encoding="utf-8")
        _write_report(report_staging / REPORT_NAME, manifest)
        os.replace(staging, output)
        os.replace(report_staging, report)
        verify_herg_quality_tasks(output_root=output)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        shutil.rmtree(report_staging, ignore_errors=True)
        if output.exists():
            shutil.rmtree(output, ignore_errors=True)
        if report.exists():
            shutil.rmtree(report, ignore_errors=True)
        raise


def verify_herg_quality_tasks(*, output_root: str | os.PathLike[str]) -> dict[str, Any]:
    """Validate hashes, schemas, scope, context semantics, and cross-task leakage."""

    root = Path(output_root).resolve()
    manifest_path = root / MANIFEST_NAME
    if root.is_symlink() or not root.is_dir() or not manifest_path.is_file():
        raise HergQualityTaskError(f"missing quality-task output: {root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    supplied = manifest.pop("manifest_sha256", None)
    expected = hashlib.sha256(_canonical_json(manifest).encode()).hexdigest()
    manifest["manifest_sha256"] = supplied
    if supplied != expected:
        raise HergQualityTaskError("manifest digest mismatch")
    schemas = {
        CONTRACT_OUTPUT: _CONTRACT_SCHEMA,
        Q0_OUTPUT: _TASK_SCHEMA,
        Q1_OUTPUT: _TASK_SCHEMA,
        Q2_OUTPUT: _TASK_SCHEMA,
        C0_OUTPUT: _TASK_SCHEMA,
        C1_OUTPUT: _QT_SCHEMA,
        QT_RECORD_OUTPUT: _QT_RECORD_SCHEMA,
        EXCLUSION_OUTPUT: _EXCLUSION_SCHEMA,
    }
    for name, schema in schemas.items():
        path = root / name
        meta = manifest["artifacts"][name]
        if path.is_symlink() or not path.is_file() or _sha256_file(path) != meta["sha256"]:
            raise HergQualityTaskError(f"artifact hash mismatch: {name}")
        if pq.ParquetFile(path).schema_arrow != schema:
            raise HergQualityTaskError(f"artifact schema mismatch: {name}")
        if pq.ParquetFile(path).metadata.num_rows != meta["rows"]:
            raise HergQualityTaskError(f"artifact row-count mismatch: {name}")

    group_splits: dict[str, set[str]] = defaultdict(set)
    structure_splits: dict[str, set[str]] = defaultdict(set)
    for name in (Q0_OUTPUT, Q1_OUTPUT, Q2_OUTPUT, C0_OUTPUT):
        rows = pq.read_table(root / name).to_pylist()
        for row in rows:
            if row["target_scope"] == "mutant_or_variant":
                raise HergQualityTaskError(f"mutant record admitted to {name}")
            if bool(row["eligible"]) == bool(row["exclusion_reason"]):
                raise HergQualityTaskError(f"eligibility/exclusion contradiction in {name}")
            if row["model_split"] and row["scaffold_group_id"]:
                group_splits[str(row["scaffold_group_id"])].add(str(row["model_split"]))
                if row["structure_id"]:
                    structure_splits[str(row["structure_id"])].add(str(row["model_split"]))
            if name == C0_OUTPUT and (row["direct_herg_label"] or row["use_as_training_label"]):
                raise HergQualityTaskError("clinical-development context promoted to hERG training label")
            if row["clinical_context_only"] and (row["direct_herg_label"] or row["use_as_training_label"]):
                raise HergQualityTaskError("clinical context promoted to direct hERG training evidence")
            if name == Q2_OUTPUT and row["clinical_context_only"] and row["eligible"]:
                raise HergQualityTaskError("clinical QT phenotype admitted to the functional training task")
            if not str(row["measurement_technology"]).strip():
                raise HergQualityTaskError(f"missing measurement technology in {name}")
    if any(len(splits) != 1 for splits in group_splits.values()):
        raise HergQualityTaskError("scaffold group crosses model partitions")
    if any(len(splits) != 1 for splits in structure_splits.values()):
        raise HergQualityTaskError("structure entity crosses model partitions")
    for name in (C1_OUTPUT, QT_RECORD_OUTPUT):
        for row in pq.read_table(root / name).to_pylist():
            if row["direct_herg_label"] or row["use_as_training_label"]:
                raise HergQualityTaskError("QT context promoted to hERG training label")
            if bool(row["heldout_evaluation_eligible"]) != (row["model_split"] == "test"):
                raise HergQualityTaskError("QT heldout eligibility disagrees with scaffold split")
            group_splits[str(row["scaffold_group_id"])].add(str(row["model_split"]))
    if any(len(splits) != 1 for splits in group_splits.values()):
        raise HergQualityTaskError("clinical scaffold group crosses model partitions")
    exclusions = pq.read_table(root / EXCLUSION_OUTPUT).to_pylist()
    mutant_rows = [
        row for row in exclusions if row["exclusion_reason"] == "explicit_mutant_or_variant_target"
    ]
    if len(mutant_rows) != int(manifest["qc"]["explicit_mutant_exclusions"]):
        raise HergQualityTaskError("mutant exclusion count mismatch")
    if any(row["target_scope"] != "mutant_or_variant" for row in mutant_rows):
        raise HergQualityTaskError("mutant exclusion scope mismatch")
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    for name in (
        "hierarchy-root",
        "model-ready-root",
        "clinical-links-root",
        "operational-tiers-root",
        "output-root",
        "report-root",
    ):
        build.add_argument(f"--{name}", required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("--output-root", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "build":
        manifest = build_herg_quality_tasks(
            hierarchy_root=args.hierarchy_root,
            model_ready_root=args.model_ready_root,
            clinical_links_root=args.clinical_links_root,
            operational_tiers_root=args.operational_tiers_root,
            output_root=args.output_root,
            report_root=args.report_root,
        )
    else:
        manifest = verify_herg_quality_tasks(output_root=args.output_root)
    print(_canonical_json({"schema_version": manifest["schema_version"], "qc": manifest["qc"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
