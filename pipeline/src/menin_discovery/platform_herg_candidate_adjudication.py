"""Build deterministic local evidence for pre-HPC hERG adjudication.

The release produced here is an evidence layer, not an adjudication result.
Automated joins, consistency checks, and duplicate-lineage heuristics may
prioritize human work, but they never establish wild-type status, experimental
independence, or gold-standard membership.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from .platform_herg_master_dataset import validate_herg_master_dataset
from .platform_herg_pre_hpc_assets import validate_herg_pre_hpc_assets

SCHEMA_VERSION = "platform-herg-candidate-adjudication/1.0"
DATASET_ID = "wild_type_herg_v1_5_candidate_adjudication"
MANIFEST_NAME = "herg_candidate_adjudication_manifest.json"
REPORT_NAME = "HERG_CANDIDATE_ADJUDICATION.md"
CANDIDATE_OUTPUT = "candidate_automated_evidence.parquet"
CONFLICT_OUTPUT = "conflict_automated_evidence.parquet"
LINEAGE_OUTPUT = "lineage_group_evidence.parquet"
PACKET_OUTPUT = "human_adjudication_packet.csv"
DECISION_CONTRACT_OUTPUT = "human_adjudication_decision_contract.json"
PIC50_EQUIVALENCE_TOLERANCE = 1e-6
AUTOMATED_STATUS = "automated_evidence_only_pending_human_adjudication"
HUMAN_STATUS = "pending_human_adjudication"


class HergCandidateAdjudicationError(RuntimeError):
    """Raised when candidate-adjudication evidence cannot be built safely."""


_CANDIDATE_SCHEMA = pa.schema(
    [
        pa.field("candidate_rank", pa.int64(), nullable=False),
        pa.field("candidate_id", pa.large_string(), nullable=False),
        pa.field("observation_id", pa.large_string(), nullable=False),
        pa.field("structure_id", pa.large_string(), nullable=False),
        pa.field("standardized_smiles", pa.large_string(), nullable=False),
        pa.field("standard_inchi_key", pa.large_string(), nullable=False),
        pa.field("automated_evidence_status", pa.large_string(), nullable=False),
        pa.field("human_adjudication_status", pa.large_string(), nullable=False),
        pa.field("automated_review_priority", pa.large_string(), nullable=False),
        pa.field("automated_evidence_score", pa.float64(), nullable=False),
        pa.field("master_wild_type_evidence_scope", pa.large_string(), nullable=False),
        pa.field("target_evidence_class", pa.large_string(), nullable=False),
        pa.field("target_status_limitations_json", pa.large_string(), nullable=False),
        pa.field("target_chembl_id", pa.large_string()),
        pa.field("target_type", pa.large_string()),
        pa.field("target_organism", pa.large_string()),
        pa.field("target_tax_id", pa.large_string()),
        pa.field("component_accessions", pa.large_string()),
        pa.field("target_relationship_type", pa.large_string()),
        pa.field("target_variant_id", pa.large_string()),
        pa.field("explicit_wild_type_text_evidence", pa.bool_(), nullable=False),
        pa.field("explicit_variant_text_warning", pa.bool_(), nullable=False),
        pa.field("source_binding_status", pa.large_string(), nullable=False),
        pa.field("source_artifact_role", pa.large_string(), nullable=False),
        pa.field("source_artifact_name", pa.large_string(), nullable=False),
        pa.field("source_artifact_row_number", pa.int64(), nullable=False),
        pa.field("source_family", pa.large_string(), nullable=False),
        pa.field("source_record_id", pa.large_string(), nullable=False),
        pa.field("document_id", pa.large_string()),
        pa.field("document_doi", pa.large_string()),
        pa.field("document_pubmed_id", pa.large_string()),
        pa.field("document_patent_id", pa.large_string()),
        pa.field("document_title", pa.large_string()),
        pa.field("document_type", pa.large_string()),
        pa.field("document_year", pa.int64()),
        pa.field("assay_id", pa.large_string()),
        pa.field("assay_family", pa.large_string(), nullable=False),
        pa.field("measurement_modality", pa.large_string(), nullable=False),
        pa.field("automation_class", pa.large_string(), nullable=False),
        pa.field("assay_description", pa.large_string()),
        pa.field("protocol_completeness_score", pa.int8(), nullable=False),
        pa.field("protocol_unresolved_fields_json", pa.large_string(), nullable=False),
        pa.field("protocol_evidence_json", pa.large_string(), nullable=False),
        pa.field("native_endpoint", pa.large_string(), nullable=False),
        pa.field("native_relation", pa.large_string()),
        pa.field("native_value", pa.float64()),
        pa.field("native_unit", pa.large_string()),
        pa.field("potency_relation_pic50", pa.large_string(), nullable=False),
        pa.field("potency_pic50_point", pa.float64()),
        pa.field("potency_pic50_lower_bound", pa.float64()),
        pa.field("potency_pic50_upper_bound", pa.float64()),
        pa.field("potency_censoring", pa.large_string(), nullable=False),
        pa.field("relation_unit_audit_status", pa.large_string(), nullable=False),
        pa.field("expected_pic50_from_native", pa.float64()),
        pa.field("pic50_conversion_absolute_delta", pa.float64()),
        pa.field("structure_exact_pic50_observation_count", pa.int64(), nullable=False),
        pa.field("structure_distinct_source_count", pa.int64(), nullable=False),
        pa.field("structure_distinct_assay_count", pa.int64(), nullable=False),
        pa.field("structure_distinct_known_document_count", pa.int64(), nullable=False),
        pa.field("structure_known_document_ids_json", pa.large_string(), nullable=False),
        pa.field("lineage_group_count", pa.int64(), nullable=False),
        pa.field("lineage_group_ids_json", pa.large_string(), nullable=False),
        pa.field("lineage_classes_json", pa.large_string(), nullable=False),
        pa.field("maximum_lineage_evidence_strength", pa.large_string(), nullable=False),
        pa.field("equal_value_group_size", pa.int64(), nullable=False),
        pa.field("equal_value_group_source_count", pa.int64(), nullable=False),
        pa.field("equal_value_group_known_document_count", pa.int64(), nullable=False),
        pa.field("model_split", pa.large_string(), nullable=False),
        pa.field("source_declared_split", pa.large_string()),
        pa.field("source_input_member", pa.large_string(), nullable=False),
        pa.field("structure_model_splits_json", pa.large_string(), nullable=False),
        pa.field("scaffold_model_splits_json", pa.large_string(), nullable=False),
        pa.field("assay_model_splits_json", pa.large_string(), nullable=False),
        pa.field("document_model_splits_json", pa.large_string(), nullable=False),
        pa.field("lineage_model_splits_json", pa.large_string(), nullable=False),
        pa.field("split_cautions_json", pa.large_string(), nullable=False),
        pa.field("automated_review_reasons_json", pa.large_string(), nullable=False),
        pa.field("required_human_checks_json", pa.large_string(), nullable=False),
        pa.field("automated_evidence_limitations_json", pa.large_string(), nullable=False),
    ]
)

_CONFLICT_SCHEMA = pa.schema(
    [
        pa.field("review_rank", pa.int64(), nullable=False),
        pa.field("review_id", pa.large_string(), nullable=False),
        pa.field("structure_id", pa.large_string(), nullable=False),
        pa.field("standardized_smiles", pa.large_string(), nullable=False),
        pa.field("standard_inchi_key", pa.large_string(), nullable=False),
        pa.field("original_review_priority", pa.large_string(), nullable=False),
        pa.field("original_priority_score", pa.float64(), nullable=False),
        pa.field("exact_replicate_count", pa.int64(), nullable=False),
        pa.field("pic50_minimum", pa.float64(), nullable=False),
        pa.field("pic50_maximum", pa.float64(), nullable=False),
        pa.field("pic50_range", pa.float64(), nullable=False),
        pa.field("automated_evidence_status", pa.large_string(), nullable=False),
        pa.field("human_adjudication_status", pa.large_string(), nullable=False),
        pa.field("automated_review_priority", pa.large_string(), nullable=False),
        pa.field("automated_review_score", pa.float64(), nullable=False),
        pa.field("observation_count_bound_to_local_source", pa.int64(), nullable=False),
        pa.field("distinct_source_count", pa.int64(), nullable=False),
        pa.field("distinct_assay_count", pa.int64(), nullable=False),
        pa.field("distinct_known_document_count", pa.int64(), nullable=False),
        pa.field("unknown_document_observation_count", pa.int64(), nullable=False),
        pa.field("known_document_ids_json", pa.large_string(), nullable=False),
        pa.field("known_document_dois_json", pa.large_string(), nullable=False),
        pa.field("exact_value_cluster_count", pa.int64(), nullable=False),
        pa.field("lineage_group_count", pa.int64(), nullable=False),
        pa.field("lineage_class_counts_json", pa.large_string(), nullable=False),
        pa.field("cross_source_mirror_group_count", pa.int64(), nullable=False),
        pa.field("same_source_duplicate_group_count", pa.int64(), nullable=False),
        pa.field("source_record_reuse_group_count", pa.int64(), nullable=False),
        pa.field("observations_in_any_lineage_group", pa.int64(), nullable=False),
        pa.field("lineage_inflation_caution", pa.bool_(), nullable=False),
        pa.field("target_evidence_class_counts_json", pa.large_string(), nullable=False),
        pa.field("target_relationship_types_json", pa.large_string(), nullable=False),
        pa.field("explicit_wild_type_text_observation_count", pa.int64(), nullable=False),
        pa.field("relation_unit_audit_counts_json", pa.large_string(), nullable=False),
        pa.field("relation_unit_inconsistency_count", pa.int64(), nullable=False),
        pa.field("minimum_protocol_completeness", pa.int8(), nullable=False),
        pa.field("median_protocol_completeness", pa.float64(), nullable=False),
        pa.field("maximum_protocol_completeness", pa.int8(), nullable=False),
        pa.field("fully_complete_protocol_observation_count", pa.int64(), nullable=False),
        pa.field("model_splits_json", pa.large_string(), nullable=False),
        pa.field("source_declared_splits_json", pa.large_string(), nullable=False),
        pa.field("source_input_members_json", pa.large_string(), nullable=False),
        pa.field("lineage_model_splits_json", pa.large_string(), nullable=False),
        pa.field("split_cautions_json", pa.large_string(), nullable=False),
        pa.field("automated_review_reasons_json", pa.large_string(), nullable=False),
        pa.field("required_human_checks_json", pa.large_string(), nullable=False),
        pa.field("automated_evidence_limitations_json", pa.large_string(), nullable=False),
    ]
)

_LINEAGE_SCHEMA = pa.schema(
    [
        pa.field("lineage_group_id", pa.large_string(), nullable=False),
        pa.field("lineage_group_kind", pa.large_string(), nullable=False),
        pa.field("automated_lineage_class", pa.large_string(), nullable=False),
        pa.field("automated_evidence_strength", pa.large_string(), nullable=False),
        pa.field("automated_evidence_status", pa.large_string(), nullable=False),
        pa.field("human_adjudication_status", pa.large_string(), nullable=False),
        pa.field("structure_id", pa.large_string(), nullable=False),
        pa.field("observation_count", pa.int64(), nullable=False),
        pa.field("queue_observation_count", pa.int64(), nullable=False),
        pa.field("source_count", pa.int64(), nullable=False),
        pa.field("source_record_count", pa.int64(), nullable=False),
        pa.field("assay_count", pa.int64(), nullable=False),
        pa.field("known_document_count", pa.int64(), nullable=False),
        pa.field("pic50_minimum", pa.float64()),
        pa.field("pic50_maximum", pa.float64()),
        pa.field("pic50_span", pa.float64()),
        pa.field("source_families_json", pa.large_string(), nullable=False),
        pa.field("reported_sources_json", pa.large_string(), nullable=False),
        pa.field("source_record_ids_json", pa.large_string(), nullable=False),
        pa.field("assay_ids_json", pa.large_string(), nullable=False),
        pa.field("document_ids_json", pa.large_string(), nullable=False),
        pa.field("observation_ids_json", pa.large_string(), nullable=False),
        pa.field("queue_observation_ids_json", pa.large_string(), nullable=False),
        pa.field("model_splits_json", pa.large_string(), nullable=False),
        pa.field("evidence_basis", pa.large_string(), nullable=False),
        pa.field("automated_evidence_limitations_json", pa.large_string(), nullable=False),
    ]
)

_PACKET_FIELDS = [
    "review_rank",
    "item_type",
    "item_id",
    "structure_id",
    "observation_id",
    "automated_review_priority",
    "automated_review_score",
    "automated_evidence_status",
    "human_adjudication_status",
    "source_family",
    "source_record_ids",
    "assay_ids",
    "document_ids",
    "document_dois",
    "target_evidence_summary",
    "measurement_summary",
    "lineage_summary",
    "protocol_summary",
    "split_cautions",
    "required_human_checks",
    "allowed_decisions",
    "human_decision",
    "human_reviewer",
    "human_reviewed_at",
    "human_notes",
]

_OBSERVATION_COLUMNS = [
    "observation_id",
    "source_family",
    "source_priority",
    "source_record_id",
    "source_row_number",
    "raw_smiles",
    "standardized_smiles",
    "standard_inchi_key",
    "structure_id",
    "target_id",
    "target_variant",
    "assay_id",
    "assay_family",
    "native_endpoint",
    "native_relation",
    "native_value",
    "native_unit",
    "native_label",
    "source_split",
    "native_aux_json",
    "reported_evidence_tier",
    "t1_candidate",
    "t1_candidate_reason",
    "quality_flags",
    "wild_type_evidence_scope",
    "measurement_modality",
    "automation_class",
    "endpoint_class",
    "endpoint_standardization_status",
    "potency_relation_pic50",
    "potency_pic50_point",
    "potency_pic50_lower_bound",
    "potency_pic50_upper_bound",
    "potency_censoring",
    "potency_standardization_basis",
    "model_split",
    "scaffold_group_id",
]

_STRENGTH_ORDER = {"none": 0, "weak": 1, "moderate": 2, "strong": 3}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _stable_id(prefix: str, *parts: object) -> str:
    payload = "\x1f".join("" if part is None else str(part) for part in parts)
    return f"{prefix}-{hashlib.sha256(payload.encode()).hexdigest()[:24].upper()}"


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
    text = str(value).strip()
    return text or None


def _json_list(value: object) -> list[Any]:
    try:
        parsed = json.loads(str(value))
    except (TypeError, json.JSONDecodeError):
        return []
    return parsed if isinstance(parsed, list) else []


def _json_object(value: object) -> dict[str, Any]:
    try:
        parsed = json.loads(str(value))
    except (TypeError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _sorted_text(values: Iterable[object]) -> list[str]:
    output: set[str] = set()
    for value in values:
        cleaned = _clean(value)
        if cleaned is not None:
            output.add(cleaned)
    return sorted(output)


def _checked_file(path: Path, *, suffixes: frozenset[str] | None = None) -> Path:
    if path.is_symlink() or not path.is_file():
        raise HergCandidateAdjudicationError(f"missing or unsafe input: {path}")
    if suffixes is not None and path.suffix.casefold() not in suffixes:
        raise HergCandidateAdjudicationError(f"unexpected input extension: {path}")
    return path.resolve()


def _artifact(path: Path, *, rows: int | None = None, schema: pa.Schema | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }
    if rows is not None:
        result["rows"] = rows
    if schema is not None:
        result["arrow_schema_sha256"] = _schema_sha256(schema)
    return result


def _write_parquet(path: Path, rows: Sequence[Mapping[str, Any]], schema: pa.Schema) -> dict[str, Any]:
    table = pa.Table.from_pylist(list(rows), schema=schema)
    pq.write_table(
        table,
        path,
        compression="zstd",
        use_dictionary=False,
        row_group_size=65_536,
        version="2.6",
    )
    return _artifact(path, rows=len(rows), schema=schema)


def _manifest_digest(payload: Mapping[str, Any]) -> str:
    body = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    return hashlib.sha256(_canonical_json(body).encode()).hexdigest()


def _input_binding(path: Path, *, role: str) -> dict[str, Any]:
    binding: dict[str, Any] = {
        "role": role,
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }
    if path.suffix.casefold() == ".parquet":
        binding["rows"] = pq.ParquetFile(path).metadata.num_rows
        binding["arrow_schema_sha256"] = _schema_sha256(pq.read_schema(path))
    return binding


def _source_inputs(
    master_manifest: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], Path | None]:
    allowed = {"chembl_herg_specialized_view", "quantitative_pic50"}
    inputs = [dict(item) for item in master_manifest.get("inputs", []) if item.get("role") in allowed]
    lineage_manifest_path: Path | None = None
    if not inputs:
        ledger_binding = next(
            (item for item in master_manifest.get("inputs", []) if item.get("role") == "observation_ledger"),
            None,
        )
        if ledger_binding is None:
            raise HergCandidateAdjudicationError(
                "master manifest has neither direct source inputs nor an observation-ledger lineage"
            )
        ledger_path = _checked_file(Path(str(ledger_binding["path"])))
        if _sha256_file(ledger_path) != ledger_binding.get("sha256"):
            raise HergCandidateAdjudicationError("master observation-ledger binding mismatch")
        lineage_manifest_path = _checked_file(ledger_path.parent / "manifest.json")
        lineage_manifest = json.loads(lineage_manifest_path.read_text(encoding="utf-8"))
        if lineage_manifest.get("manifest_sha256") != _manifest_digest(lineage_manifest):
            raise HergCandidateAdjudicationError("observation-ledger lineage manifest digest mismatch")
        ledger_artifact = next(
            (item for item in lineage_manifest.get("artifacts", []) if item.get("path") == ledger_path.name),
            None,
        )
        if (
            ledger_artifact is None
            or ledger_artifact.get("sha256") != ledger_binding.get("sha256")
            or int(ledger_artifact.get("rows", -1)) != int(ledger_binding.get("rows", -2))
        ):
            raise HergCandidateAdjudicationError(
                "observation-ledger lineage manifest is not bound to the v1.3 master"
            )
        inputs = [dict(item) for item in lineage_manifest.get("inputs", []) if item.get("role") in allowed]
    if not any(item.get("role") == "chembl_herg_specialized_view" for item in inputs):
        raise HergCandidateAdjudicationError("master manifest has no ChEMBL source binding")
    if not any(item.get("role") == "quantitative_pic50" for item in inputs):
        raise HergCandidateAdjudicationError("master manifest has no quantitative source binding")
    for item in inputs:
        path = _checked_file(Path(str(item["path"])), suffixes=frozenset({".parquet", ".csv"}))
        if _sha256_file(path) != item.get("sha256"):
            raise HergCandidateAdjudicationError(f"master-bound source hash mismatch: {path}")
    return inputs, lineage_manifest_path


def _equivalent_scalar(left: object, right: object, *, tolerance: float = 1e-12) -> bool:
    if left is None or right is None:
        return left is None and right is None
    if isinstance(left, (int, float)) or isinstance(right, (int, float)):
        try:
            return math.isclose(float(str(left)), float(str(right)), rel_tol=0.0, abs_tol=tolerance)
        except (TypeError, ValueError):
            return False
    return str(left).strip() == str(right).strip()


def _base_source_evidence(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "source_binding_status": "unbound_local_source_record",
        "source_artifact_role": "unresolved",
        "source_artifact_name": "",
        "source_artifact_row_number": int(row["source_row_number"]),
        "source_input_member": "unresolved",
        "reported_source": None,
        "provided_inchi_key": None,
        "document_id": None,
        "document_doi": None,
        "document_pubmed_id": None,
        "document_patent_id": None,
        "document_title": None,
        "document_type": None,
        "document_year": None,
        "assay_description": _json_object(row["native_aux_json"]).get("assay_description"),
        "target_chembl_id": None,
        "target_type": None,
        "target_organism": None,
        "target_tax_id": None,
        "component_accessions": None,
        "target_relationship_type": None,
        "target_variant_id": None,
        "assay_organism": None,
        "assay_tax_id": None,
        "assay_cell_type": None,
    }


def _load_chembl_source_evidence(
    paths: Sequence[Path], relevant: Mapping[str, Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    by_record: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in relevant.values():
        if row["source_family"] == "chembl_herg_specialized_view":
            by_record[str(row["source_record_id"])].append(row)
    if not by_record:
        return {}
    if any(len(rows) != 1 for rows in by_record.values()):
        duplicates = sorted(key for key, rows in by_record.items() if len(rows) != 1)[:5]
        raise HergCandidateAdjudicationError(
            f"ambiguous ChEMBL source-record bindings in relevant observations: {duplicates}"
        )
    required = {
        "activity_id",
        "assay_chembl_id",
        "standard_type",
        "standard_relation",
        "standard_value",
        "standard_units",
        "canonical_smiles",
        "document_chembl_id",
        "target_chembl_id",
        "target_type",
        "target_organism",
        "target_tax_id",
        "component_accessions",
        "relationship_type",
        "variant_id",
    }
    output: dict[str, dict[str, Any]] = {}
    row_offset = 0
    for path_index, path in enumerate(paths, start=1):
        table = pq.read_table(path)
        missing = sorted(required - set(table.column_names))
        if missing:
            raise HergCandidateAdjudicationError(f"ChEMBL source is missing columns: {missing}")
        for local_index, source in enumerate(table.to_pylist(), start=1):
            record_id = f"ACTIVITY:{source['activity_id']}"
            if record_id not in by_record:
                continue
            master = by_record[record_id][0]
            checks = [
                _equivalent_scalar(master["source_row_number"], row_offset + local_index),
                _equivalent_scalar(master["assay_id"], source.get("assay_chembl_id")),
                _equivalent_scalar(master["native_endpoint"], source.get("standard_type")),
                _equivalent_scalar(master["native_relation"], source.get("standard_relation")),
                _equivalent_scalar(master["native_value"], source.get("standard_value")),
                _equivalent_scalar(master["native_unit"], source.get("standard_units")),
                _equivalent_scalar(master["raw_smiles"], source.get("canonical_smiles")),
                _equivalent_scalar(
                    _json_object(master["native_aux_json"]).get("document_chembl_id"),
                    source.get("document_chembl_id"),
                ),
            ]
            if not all(checks):
                raise HergCandidateAdjudicationError(
                    f"ChEMBL primary-key binding disagrees with master row: {record_id}"
                )
            observation_id = str(master["observation_id"])
            if observation_id in output:
                raise HergCandidateAdjudicationError(f"duplicate ChEMBL source binding: {observation_id}")
            output[observation_id] = {
                "source_binding_status": "exact_primary_key_and_fields_match",
                "source_artifact_role": "chembl_herg_specialized_view",
                "source_artifact_name": path.name,
                "source_artifact_row_number": row_offset + local_index,
                "source_input_member": f"chembl_input_{path_index:02d}:{path.name}",
                "reported_source": _clean(source.get("activity_source_name")),
                "provided_inchi_key": _clean(source.get("standard_inchi_key")),
                "document_id": _clean(source.get("document_chembl_id")),
                "document_doi": _clean(source.get("document_doi")),
                "document_pubmed_id": _clean(source.get("pubmed_id")),
                "document_patent_id": _clean(source.get("patent_id")),
                "document_title": _clean(source.get("document_title")),
                "document_type": _clean(source.get("document_type")),
                "document_year": int(source["document_year"])
                if source.get("document_year") is not None
                else None,
                "assay_description": _clean(source.get("assay_description")),
                "target_chembl_id": _clean(source.get("target_chembl_id")),
                "target_type": _clean(source.get("target_type")),
                "target_organism": _clean(source.get("target_organism")),
                "target_tax_id": _clean(source.get("target_tax_id")),
                "component_accessions": _clean(source.get("component_accessions")),
                "target_relationship_type": _clean(source.get("relationship_type")),
                "target_variant_id": _clean(source.get("variant_id")),
                "assay_organism": _clean(source.get("assay_organism")),
                "assay_tax_id": _clean(source.get("assay_tax_id")),
                "assay_cell_type": _clean(source.get("assay_cell_type")),
            }
        row_offset += table.num_rows
    missing_observations = sorted(
        str(rows[0]["observation_id"])
        for record_id, rows in by_record.items()
        if str(rows[0]["observation_id"]) not in output
    )
    if missing_observations:
        raise HergCandidateAdjudicationError(
            f"relevant ChEMBL observations lack source bindings: {missing_observations[:5]}"
        )
    return output


def _header(fieldnames: Sequence[str] | None, aliases: Sequence[str], *, required: bool) -> str | None:
    normalized = {str(name).strip().casefold(): str(name) for name in (fieldnames or [])}
    for alias in aliases:
        if alias.casefold() in normalized:
            return normalized[alias.casefold()]
    if required:
        raise HergCandidateAdjudicationError(f"missing CSV column: {aliases[0]}")
    return None


def _load_quantitative_source_evidence(
    paths: Sequence[Path], relevant: Mapping[str, Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    wanted = {
        observation_id: row
        for observation_id, row in relevant.items()
        if row["source_family"] == "quantitative_pic50_release"
    }
    if not wanted:
        return {}
    output: dict[str, dict[str, Any]] = {}
    for path_index, path in enumerate(paths, start=1):
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            pic50_column = _header(reader.fieldnames, ("pIC50",), required=True)
            source_column = _header(reader.fieldnames, ("Source",), required=True)
            key_column = _header(reader.fieldnames, ("InChI Key", "InChl Key", "InChIKey"), required=False)
            split_column = _header(reader.fieldnames, ("USED_AS", "split", "partition"), required=False)
            chembl_column = _header(reader.fieldnames, ("ChEMBL ID",), required=False)
            cid_column = _header(reader.fieldnames, ("PubChem CID", "CID"), required=False)
            smiles_column = _header(reader.fieldnames, ("SMILES",), required=True)
            assert pic50_column is not None and source_column is not None and smiles_column is not None
            for row_number, source in enumerate(reader, start=1):
                value = float(str(source[pic50_column]).strip())
                source_record_id = (
                    (_clean(source.get(chembl_column)) if chembl_column else None)
                    or (_clean(source.get(cid_column)) if cid_column else None)
                    or f"{path_index}:{row_number}"
                )
                observation_id = _stable_id(
                    "HOBS",
                    "quantitative_pic50_release",
                    source_record_id,
                    row_number,
                    "pIC50",
                    "=",
                    value,
                    None,
                )
                if observation_id not in wanted:
                    continue
                master = wanted[observation_id]
                checks = [
                    _equivalent_scalar(master["source_record_id"], source_record_id),
                    _equivalent_scalar(master["source_row_number"], row_number),
                    _equivalent_scalar(master["native_value"], value),
                    _equivalent_scalar(master["raw_smiles"], source.get(smiles_column)),
                    _equivalent_scalar(
                        _json_object(master["native_aux_json"]).get("reported_source"),
                        source.get(source_column),
                    ),
                ]
                if not all(checks):
                    raise HergCandidateAdjudicationError(
                        f"quantitative source binding disagrees with master row: {observation_id}"
                    )
                if observation_id in output:
                    raise HergCandidateAdjudicationError(
                        f"quantitative observation binds to multiple source rows: {observation_id}"
                    )
                output[observation_id] = {
                    "source_binding_status": "exact_reconstructed_observation_id_and_fields_match",
                    "source_artifact_role": "quantitative_pic50",
                    "source_artifact_name": path.name,
                    "source_artifact_row_number": row_number,
                    "source_input_member": f"quantitative_input_{path_index:02d}:{path.name}",
                    "reported_source": _clean(source.get(source_column)),
                    "provided_inchi_key": _clean(source.get(key_column)) if key_column else None,
                    "document_id": None,
                    "document_doi": None,
                    "document_pubmed_id": None,
                    "document_patent_id": None,
                    "document_title": None,
                    "document_type": None,
                    "document_year": None,
                    "assay_description": None,
                    "target_chembl_id": None,
                    "target_type": None,
                    "target_organism": None,
                    "target_tax_id": None,
                    "component_accessions": None,
                    "target_relationship_type": None,
                    "target_variant_id": None,
                    "assay_organism": None,
                    "assay_tax_id": None,
                    "assay_cell_type": None,
                    "raw_source_declared_split": _clean(source.get(split_column)) if split_column else None,
                }
    missing = sorted(set(wanted) - set(output))
    if missing:
        raise HergCandidateAdjudicationError(
            f"relevant quantitative observations lack source bindings: {missing[:5]}"
        )
    return output


def _load_source_evidence(
    source_inputs: Sequence[Mapping[str, Any]], relevant_rows: Sequence[Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    relevant = {str(row["observation_id"]): row for row in relevant_rows}
    chembl_paths = [
        Path(str(item["path"])).resolve()
        for item in source_inputs
        if item["role"] == "chembl_herg_specialized_view"
    ]
    quantitative_paths = [
        Path(str(item["path"])).resolve() for item in source_inputs if item["role"] == "quantitative_pic50"
    ]
    output = {key: _base_source_evidence(row) for key, row in relevant.items()}
    output.update(_load_chembl_source_evidence(chembl_paths, relevant))
    output.update(_load_quantitative_source_evidence(quantitative_paths, relevant))
    unbound = sorted(
        observation_id
        for observation_id, evidence in output.items()
        if evidence["source_binding_status"] == "unbound_local_source_record"
    )
    if unbound:
        raise HergCandidateAdjudicationError(f"unbound relevant source observations: {unbound[:5]}")
    return output


_CONCENTRATION_FACTORS = {
    "m": 1.0,
    "mm": 1e-3,
    "um": 1e-6,
    "nm": 1e-9,
    "pm": 1e-12,
    "fm": 1e-15,
}
_RELATION_TO_PIC50 = {"=": "=", ">": "<", ">=": "<=", "<": ">", "<=": ">="}


def _relation_unit_audit(row: Mapping[str, Any]) -> dict[str, Any]:
    endpoint = str(row["native_endpoint"]).strip().casefold()
    unit = str(row.get("native_unit") or "").replace("µ", "u").replace("μ", "u").casefold()
    value = row.get("native_value")
    native_relation = str(row.get("native_relation") or "")
    relation = str(row.get("potency_relation_pic50") or "")
    expected: float | None = None
    expected_relation: str | None
    if endpoint == "pic50" and value is not None:
        expected = float(value)
        expected_relation = native_relation
    elif endpoint == "ic50" and unit in _CONCENTRATION_FACTORS and value is not None and float(value) > 0:
        expected = -math.log10(float(value) * _CONCENTRATION_FACTORS[unit])
        expected_relation = _RELATION_TO_PIC50.get(native_relation)
    else:
        return {
            "status": "not_machine_checkable_endpoint_or_unit",
            "expected_pic50": None,
            "absolute_delta": None,
        }
    if expected_relation != relation:
        return {
            "status": "inconsistent_relation_direction",
            "expected_pic50": expected,
            "absolute_delta": None,
        }
    coordinate = (
        row.get("potency_pic50_point")
        if relation == "="
        else row.get("potency_pic50_upper_bound")
        if relation in {"<", "<="}
        else row.get("potency_pic50_lower_bound")
    )
    if coordinate is None:
        return {
            "status": "inconsistent_missing_pic50_coordinate",
            "expected_pic50": expected,
            "absolute_delta": None,
        }
    delta = abs(float(coordinate) - expected)
    if delta > PIC50_EQUIVALENCE_TOLERANCE:
        status = "inconsistent_numeric_conversion"
    elif relation == "=":
        status = "consistent_exact_relation_and_unit_conversion"
    else:
        status = "consistent_censored_relation_inversion_and_bound"
    return {"status": status, "expected_pic50": expected, "absolute_delta": delta}


def _target_evidence(row: Mapping[str, Any], source: Mapping[str, Any]) -> tuple[str, bool, bool, list[str]]:
    description = str(source.get("assay_description") or "")
    explicit_wild_type = bool(re.search(r"(?i)\bwild[ -]?type\b", description))
    explicit_variant = bool(re.search(r"(?i)\b(mutant|mutation|variant)\b", description))
    limitations = ["automated_target_metadata_is_not_human_adjudication"]
    relationship = source.get("target_relationship_type")
    if row["source_family"] == "quantitative_pic50_release":
        evidence_class = "compilation_target_assertion_without_assay_level_target_status"
        limitations.extend(
            [
                "secondary_compilation_has_no_assay_level_target_record",
                "wild_type_status_not_explicitly_confirmed",
            ]
        )
    elif (
        source.get("target_chembl_id") == "CHEMBL240"
        and source.get("component_accessions") == "Q12809"
        and source.get("target_type") == "SINGLE PROTEIN"
        and source.get("target_organism") == "Homo sapiens"
        and relationship == "D"
        and not source.get("target_variant_id")
    ):
        evidence_class = "direct_human_kcnh2_single_protein_no_variant_annotation"
        limitations.extend(
            [
                "absence_of_variant_annotation_is_not_explicit_wild_type_confirmation",
                "primary_document_or_protocol_must_confirm_construct",
            ]
        )
    elif (
        source.get("target_chembl_id") == "CHEMBL240"
        and source.get("component_accessions") == "Q12809"
        and source.get("target_type") == "SINGLE PROTEIN"
        and source.get("target_organism") == "Homo sapiens"
        and relationship == "H"
        and not source.get("target_variant_id")
    ):
        evidence_class = "homologue_relationship_to_human_kcnh2_no_variant_annotation"
        limitations.extend(
            [
                "chembl_relationship_is_homologue_not_direct",
                "absence_of_variant_annotation_is_not_explicit_wild_type_confirmation",
                "primary_document_must_resolve_tested_species_and_construct",
            ]
        )
    else:
        evidence_class = "target_metadata_incomplete_or_noncanonical"
        limitations.extend(
            ["target_identity_or_relationship_requires_manual_resolution", "wild_type_status_unresolved"]
        )
    if not explicit_wild_type:
        limitations.append("no_explicit_wild_type_phrase_in_local_assay_description")
    if explicit_variant:
        limitations.append("variant_or_mutant_phrase_requires_manual_review")
    return evidence_class, explicit_wild_type, explicit_variant, sorted(set(limitations))


def _protocol_map(path: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    output: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in pq.read_table(path).to_pylist():
        key = (str(row["source_family"]), str(row["assay_id"] or ""), str(row["assay_family"]))
        if key in output:
            raise HergCandidateAdjudicationError(f"duplicate assay protocol key: {key}")
        output[key] = row
    return output


def _protocol_for(
    row: Mapping[str, Any], protocols: Mapping[tuple[str, str, str], Mapping[str, Any]]
) -> dict[str, Any]:
    key = (str(row["source_family"]), str(row.get("assay_id") or ""), str(row["assay_family"]))
    protocol = dict(protocols.get(key, {}))
    if not protocol:
        return {
            "protocol_completeness_score": 0,
            "unresolved_fields_json": _canonical_json(
                ["host_system", "voltage", "temperature", "time", "recording_configuration", "platform"]
            ),
        }
    return protocol


def _protocol_evidence_json(protocol: Mapping[str, Any]) -> str:
    fields = [
        "host_systems_json",
        "host_system_evidence_json",
        "host_system_confidence",
        "voltage_values_mv_json",
        "voltage_evidence_json",
        "voltage_confidence",
        "temperature_values_celsius_json",
        "temperature_condition_terms_json",
        "temperature_evidence_json",
        "temperature_confidence",
        "time_values_seconds_json",
        "time_evidence_json",
        "time_confidence",
        "recording_configurations_json",
        "recording_configuration_evidence_json",
        "recording_configuration_confidence",
        "named_platforms_json",
        "platform_evidence_json",
        "platform_confidence",
        "manual_automation_evidence_json",
        "normalization_policy",
    ]
    return _canonical_json({field: protocol.get(field) for field in fields if field in protocol})


def _pic50_coordinate(row: Mapping[str, Any]) -> float | None:
    relation = str(row.get("potency_relation_pic50") or "")
    value = (
        row.get("potency_pic50_point")
        if relation == "="
        else row.get("potency_pic50_upper_bound")
        if relation in {"<", "<="}
        else row.get("potency_pic50_lower_bound")
    )
    return float(value) if value is not None else None


def _value_clusters(rows: Sequence[Mapping[str, Any]]) -> list[list[Mapping[str, Any]]]:
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        coordinate = _pic50_coordinate(row)
        if coordinate is None:
            continue
        grouped[
            (
                str(row["structure_id"]),
                str(row.get("potency_relation_pic50") or ""),
                str(row["potency_censoring"]),
            )
        ].append(row)
    output: list[list[Mapping[str, Any]]] = []
    for key in sorted(grouped):
        ordered = sorted(
            grouped[key], key=lambda row: (_pic50_coordinate(row) or -math.inf, str(row["observation_id"]))
        )
        current: list[Mapping[str, Any]] = []
        minimum: float | None = None
        for row in ordered:
            coordinate = _pic50_coordinate(row)
            assert coordinate is not None
            if minimum is None or coordinate - minimum <= PIC50_EQUIVALENCE_TOLERANCE:
                current.append(row)
                minimum = coordinate if minimum is None else minimum
            else:
                output.append(current)
                current = [row]
                minimum = coordinate
        if current:
            output.append(current)
    return output


def _lineage_group_row(
    *,
    group_kind: str,
    lineage_class: str,
    evidence_strength: str,
    rows: Sequence[Mapping[str, Any]],
    source_evidence: Mapping[str, Mapping[str, Any]],
    queue_observation_ids: set[str],
    evidence_basis: str,
    limitations: Sequence[str],
) -> dict[str, Any]:
    observation_ids = sorted(str(row["observation_id"]) for row in rows)
    queue_ids = sorted(set(observation_ids) & queue_observation_ids)
    structure_ids = {str(row["structure_id"]) for row in rows}
    if len(structure_ids) != 1:
        raise HergCandidateAdjudicationError("lineage group crosses standardized structures")
    values = [_pic50_coordinate(row) for row in rows]
    numeric = [value for value in values if value is not None]
    documents = _sorted_text(source_evidence[oid].get("document_id") for oid in observation_ids)
    assays = _sorted_text(row.get("assay_id") for row in rows)
    sources = _sorted_text(row["source_family"] for row in rows)
    source_records = _sorted_text(row["source_record_id"] for row in rows)
    reported_sources = _sorted_text(source_evidence[oid].get("reported_source") for oid in observation_ids)
    signature = _canonical_json(
        {
            "group_kind": group_kind,
            "lineage_class": lineage_class,
            "observation_ids": observation_ids,
        }
    )
    return {
        "lineage_group_id": _stable_id("HLINEAGE", signature),
        "lineage_group_kind": group_kind,
        "automated_lineage_class": lineage_class,
        "automated_evidence_strength": evidence_strength,
        "automated_evidence_status": AUTOMATED_STATUS,
        "human_adjudication_status": HUMAN_STATUS,
        "structure_id": next(iter(structure_ids)),
        "observation_count": len(rows),
        "queue_observation_count": len(queue_ids),
        "source_count": len(sources),
        "source_record_count": len(source_records),
        "assay_count": len(assays),
        "known_document_count": len(documents),
        "pic50_minimum": min(numeric) if numeric else None,
        "pic50_maximum": max(numeric) if numeric else None,
        "pic50_span": max(numeric) - min(numeric) if numeric else None,
        "source_families_json": _canonical_json(sources),
        "reported_sources_json": _canonical_json(reported_sources),
        "source_record_ids_json": _canonical_json(source_records),
        "assay_ids_json": _canonical_json(assays),
        "document_ids_json": _canonical_json(documents),
        "observation_ids_json": _canonical_json(observation_ids),
        "queue_observation_ids_json": _canonical_json(queue_ids),
        "model_splits_json": _canonical_json(_sorted_text(row.get("model_split") for row in rows)),
        "evidence_basis": evidence_basis,
        "automated_evidence_limitations_json": _canonical_json(sorted(set(limitations))),
    }


def _build_lineage_evidence(
    rows: Sequence[Mapping[str, Any]],
    source_evidence: Mapping[str, Mapping[str, Any]],
    queue_observation_ids: set[str],
) -> tuple[
    list[dict[str, Any]],
    dict[str, list[dict[str, Any]]],
    dict[str, dict[str, Any]],
]:
    group_rows: list[dict[str, Any]] = []
    value_cluster_by_observation: dict[str, dict[str, Any]] = {}

    for cluster in _value_clusters(rows):
        observation_ids = [str(row["observation_id"]) for row in cluster]
        sources = {str(row["source_family"]) for row in cluster}
        records = Counter((str(row["source_family"]), str(row["source_record_id"])) for row in cluster)
        documents = {
            _clean(source_evidence[observation_id].get("document_id")) for observation_id in observation_ids
        } - {None}
        assays = {_clean(row.get("assay_id")) for row in cluster} - {None}
        reported_sources = {
            str(source_evidence[observation_id].get("reported_source") or "").casefold()
            for observation_id in observation_ids
        }
        if len(sources) > 1 and "chembl_herg_specialized_view" in sources and "chembl" in reported_sources:
            lineage_class = "cross_source_exact_value_mirror_candidate_reported_chembl"
            strength = "strong"
            basis = (
                "Same standardized structure, relation, censoring semantics, and pIC50 within tolerance "
                "across source families; the secondary compilation names ChEMBL as its source."
            )
        elif len(sources) > 1:
            lineage_class = "cross_source_exact_value_mirror_candidate"
            strength = "moderate"
            basis = (
                "Same standardized structure, relation, censoring semantics, and pIC50 within tolerance "
                "across source families."
            )
        elif any(count > 1 for count in records.values()):
            lineage_class = "same_source_record_equal_value_reuse_candidate"
            strength = "strong"
            basis = "One source primary key is reused for equal standardized pIC50 observations."
        elif len(documents) <= 1 and len(assays) <= 1:
            lineage_class = "within_source_exact_measurement_duplicate_candidate"
            strength = "moderate"
            basis = (
                "Different source records have the same structure, normalized value, assay, and known "
                "document context."
            )
        else:
            lineage_class = "same_structure_equal_value_multiple_contexts"
            strength = "weak"
            basis = (
                "Equal standardized values occur in more than one assay or known document; equality alone "
                "does not establish duplication."
            )
        coordinates = [_pic50_coordinate(row) for row in cluster]
        numeric = [value for value in coordinates if value is not None]
        cluster_id = _stable_id(
            "HVALUE",
            cluster[0]["structure_id"],
            cluster[0].get("potency_relation_pic50"),
            cluster[0]["potency_censoring"],
            min(numeric),
            max(numeric),
            *sorted(observation_ids),
        )
        info = {
            "value_cluster_id": cluster_id,
            "size": len(cluster),
            "source_count": len(sources),
            "known_document_count": len(documents),
            "model_splits": _sorted_text(row.get("model_split") for row in cluster),
        }
        for observation_id in observation_ids:
            value_cluster_by_observation[observation_id] = info
        if len(cluster) < 2:
            continue
        group_rows.append(
            _lineage_group_row(
                group_kind="standardized_equal_value_cluster",
                lineage_class=lineage_class,
                evidence_strength=strength,
                rows=cluster,
                source_evidence=source_evidence,
                queue_observation_ids=queue_observation_ids,
                evidence_basis=basis,
                limitations=[
                    "numeric_equality_does_not_prove_common_experimental_origin",
                    "manual_primary_source_comparison_required",
                    "group_is_not_an_authorized_deduplication",
                ],
            )
        )

    source_record_groups: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    native_groups: dict[tuple[object, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        observation_id = str(row["observation_id"])
        source_record_groups[
            (str(row["structure_id"]), str(row["source_family"]), str(row["source_record_id"]))
        ].append(row)
        document_id = source_evidence[observation_id].get("document_id")
        native_groups[
            (
                str(row["structure_id"]),
                str(row["source_family"]),
                str(row.get("assay_id") or ""),
                str(document_id or ""),
                str(row["native_endpoint"]),
                str(row.get("native_relation") or ""),
                row.get("native_value"),
                str(row.get("native_unit") or ""),
            )
        ].append(row)

    for key in sorted(source_record_groups):
        group = source_record_groups[key]
        if len(group) < 2:
            continue
        group_rows.append(
            _lineage_group_row(
                group_kind="source_record_reuse",
                lineage_class="same_source_primary_key_multiple_observations",
                evidence_strength="strong",
                rows=group,
                source_evidence=source_evidence,
                queue_observation_ids=queue_observation_ids,
                evidence_basis=(
                    "The same source family and source primary key occur more than once for one "
                    "standardized structure."
                ),
                limitations=[
                    "source_key_reuse_can_represent_multiple_reported_measurements",
                    "manual_source_row_review_required_before_collapse",
                    "group_is_not_an_authorized_deduplication",
                ],
            )
        )

    for native_key in sorted(native_groups, key=lambda item: tuple(str(value) for value in item)):
        group = native_groups[native_key]
        if len(group) < 2 or len({str(row["source_record_id"]) for row in group}) < 2:
            continue
        group_rows.append(
            _lineage_group_row(
                group_kind="exact_native_measurement_signature",
                lineage_class="within_source_exact_native_measurement_duplicate_candidate",
                evidence_strength="moderate",
                rows=group,
                source_evidence=source_evidence,
                queue_observation_ids=queue_observation_ids,
                evidence_basis=(
                    "Different source primary keys share structure, source family, assay, known document, "
                    "native endpoint, relation, value, and unit."
                ),
                limitations=[
                    "exact_native_match_can_be_legitimate_replicates",
                    "manual_primary_source_comparison_required",
                    "group_is_not_an_authorized_deduplication",
                ],
            )
        )

    unique: dict[str, dict[str, Any]] = {}
    for row in group_rows:
        group_id = str(row["lineage_group_id"])
        if group_id in unique:
            raise HergCandidateAdjudicationError(f"duplicate lineage group id: {group_id}")
        unique[group_id] = row
    ordered = sorted(
        unique.values(),
        key=lambda row: (
            str(row["structure_id"]),
            str(row["lineage_group_kind"]),
            str(row["lineage_group_id"]),
        ),
    )
    by_observation: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for lineage_group in ordered:
        for observation_id in _json_list(lineage_group["observation_ids_json"]):
            by_observation[str(observation_id)].append(lineage_group)
    return ordered, by_observation, value_cluster_by_observation


def _split_indices(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, set[str]]]:
    indices: dict[str, dict[str, set[str]]] = {
        "structure": defaultdict(set),
        "scaffold": defaultdict(set),
        "assay": defaultdict(set),
        "document": defaultdict(set),
    }
    for row in rows:
        split = _clean(row.get("model_split"))
        if split is None:
            continue
        structure = _clean(row.get("structure_id"))
        scaffold = _clean(row.get("scaffold_group_id"))
        assay = _clean(row.get("assay_id"))
        document = _clean(_json_object(row.get("native_aux_json")).get("document_chembl_id"))
        if structure:
            indices["structure"][structure].add(split)
        if scaffold:
            indices["scaffold"][scaffold].add(split)
        if assay:
            indices["assay"][assay].add(split)
        if document:
            indices["document"][document].add(split)
    return indices


def _maximum_strength(groups: Sequence[Mapping[str, Any]]) -> str:
    return max(
        (str(group["automated_evidence_strength"]) for group in groups),
        key=lambda value: _STRENGTH_ORDER[value],
        default="none",
    )


def _candidate_split_cautions(
    row: Mapping[str, Any],
    source: Mapping[str, Any],
    split_indices: Mapping[str, Mapping[str, set[str]]],
    groups: Sequence[Mapping[str, Any]],
) -> tuple[list[str], dict[str, list[str]]]:
    structure_splits = sorted(split_indices["structure"].get(str(row["structure_id"]), set()))
    scaffold_splits = sorted(split_indices["scaffold"].get(str(row["scaffold_group_id"]), set()))
    assay_splits = sorted(split_indices["assay"].get(str(row.get("assay_id") or ""), set()))
    document_splits = sorted(split_indices["document"].get(str(source.get("document_id") or ""), set()))
    lineage_splits = _sorted_text(
        split for group in groups for split in _json_list(group["model_splits_json"])
    )
    cautions = ["candidate_is_not_sealed_evaluation_membership"]
    if row.get("model_split") == "train":
        cautions.append("candidate_currently_in_training_partition")
    elif row.get("model_split") == "validation":
        cautions.append("candidate_currently_in_validation_partition")
    if len(structure_splits) > 1:
        cautions.append("same_structure_crosses_master_model_partitions")
    if len(scaffold_splits) > 1:
        cautions.append("same_scaffold_crosses_master_model_partitions")
    if len(assay_splits) > 1:
        cautions.append("same_assay_spans_master_model_partitions")
    if len(document_splits) > 1:
        cautions.append("same_document_spans_master_model_partitions")
    if len(lineage_splits) > 1:
        cautions.append("automated_lineage_group_spans_master_model_partitions")
    source_split = _clean(row.get("source_split"))
    model_split = _clean(row.get("model_split"))
    if source_split and model_split and source_split.casefold() != model_split.casefold():
        cautions.append("source_declared_partition_differs_from_master_partition")
    cautions.append("future_acceptance_requires_structure_scaffold_assay_document_lineage_refreeze")
    return sorted(set(cautions)), {
        "structure": structure_splits,
        "scaffold": scaffold_splits,
        "assay": assay_splits,
        "document": document_splits,
        "lineage": lineage_splits,
    }


def _candidate_rows(
    candidates: Sequence[Mapping[str, Any]],
    observations: Mapping[str, Mapping[str, Any]],
    structure_context: Mapping[str, Sequence[Mapping[str, Any]]],
    source_evidence: Mapping[str, Mapping[str, Any]],
    protocols: Mapping[tuple[str, str, str], Mapping[str, Any]],
    split_indices: Mapping[str, Mapping[str, set[str]]],
    groups_by_observation: Mapping[str, Sequence[Mapping[str, Any]]],
    value_clusters: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for candidate in sorted(candidates, key=lambda row: int(row["candidate_rank"])):
        observation_id = str(candidate["observation_id"])
        row = observations[observation_id]
        source = source_evidence[observation_id]
        protocol = _protocol_for(row, protocols)
        relation_audit = _relation_unit_audit(row)
        target_class, explicit_wt, explicit_variant, target_limitations = _target_evidence(row, source)
        groups = list(groups_by_observation.get(observation_id, []))
        lineage_ids = sorted(str(group["lineage_group_id"]) for group in groups)
        lineage_classes = sorted({str(group["automated_lineage_class"]) for group in groups})
        strength = _maximum_strength(groups)
        value_cluster = value_clusters[observation_id]
        context = [
            item
            for item in structure_context[str(row["structure_id"])]
            if item["endpoint_standardization_status"] == "exact_standardized"
            and item.get("potency_pic50_point") is not None
        ]
        context_documents = _sorted_text(
            source_evidence[str(item["observation_id"])].get("document_id") for item in context
        )
        context_sources = _sorted_text(item["source_family"] for item in context)
        context_assays = _sorted_text(item.get("assay_id") for item in context)
        split_cautions, split_surfaces = _candidate_split_cautions(row, source, split_indices, groups)
        score = 0.0
        reasons: list[str] = []
        if target_class == "direct_human_kcnh2_single_protein_no_variant_annotation":
            score += 28.0
            reasons.append("direct_human_kcnh2_single_protein_target_metadata")
        elif target_class == "homologue_relationship_to_human_kcnh2_no_variant_annotation":
            reasons.append("homologue_target_relationship_requires_resolution")
        else:
            score -= 8.0
            reasons.append("target_metadata_incomplete")
        if explicit_wt:
            score += 12.0
            reasons.append("explicit_wild_type_phrase_in_local_assay_text_requires_verification")
        if explicit_variant:
            score -= 25.0
            reasons.append("variant_or_mutant_phrase_requires_resolution")
        if source["source_binding_status"] == "exact_primary_key_and_fields_match":
            score += 6.0
            reasons.append("exact_manifest_bound_source_join")
        if source.get("document_type") == "PUBLICATION":
            score += 8.0
            reasons.append("primary_publication_metadata_available")
        if source.get("document_doi"):
            score += 5.0
        if source.get("document_pubmed_id"):
            score += 3.0
        if row["measurement_modality"] == "patch_clamp_electrophysiology":
            score += 12.0
            reasons.append("patch_clamp_modality")
        if row["automation_class"] == "manual":
            score += 12.0
            reasons.append("manual_automation_class")
        elif row["automation_class"] == "automated":
            score += 5.0
        score += int(protocol.get("protocol_completeness_score", 0)) * 3.0
        if row["potency_censoring"] == "exact":
            score += 10.0
        else:
            score += 3.0
            reasons.append("one_sided_censored_measurement")
        score += min(6.0, len(context_documents) * 2.0)
        if strength == "strong":
            score -= 20.0
            reasons.append("strong_automated_duplicate_or_mirror_lineage_evidence")
        elif strength == "moderate":
            score -= 10.0
            reasons.append("moderate_automated_duplicate_lineage_evidence")
        elif strength == "weak":
            score -= 3.0
            reasons.append("weak_equal_value_lineage_evidence")
        if relation_audit["status"].startswith("inconsistent"):
            score -= 40.0
            reasons.append("relation_or_unit_inconsistency")
        if score >= 75.0:
            priority = "P0_high_yield_primary_source_adjudication"
        elif score >= 50.0:
            priority = "P1_evidence_or_lineage_gap_adjudication"
        else:
            priority = "P2_material_target_or_protocol_gap_adjudication"
        required_checks = [
            "retrieve_primary_document_and_verify_explicit_human_wild_type_kcnh2_construct",
            "verify_assay_modality_cell_system_platform_voltage_temperature_and_timing_from_source",
            "verify_native_ic50_relation_unit_value_and_standardized_pic50_transcription",
            "compare_all_automated_duplicate_or_mirror_lineage_candidates_against_primary_sources",
            "record_human_decision_with_reviewer_date_and_source_citation",
            "if_accepted_refreeze_structure_scaffold_assay_document_and_measurement_lineages_outside_training",
        ]
        if source.get("target_relationship_type") == "H":
            required_checks.append("resolve_chembl_homologue_relationship_and_actual_tested_species")
        limitations = [
            "automated_score_is_for_review_order_only",
            "no_row_is_promoted_to_gold",
            "distinct_assays_documents_or_sources_are_not_assumed_independent",
            "absence_of_duplicate_signal_is_not_proof_of_independence",
            "local_metadata_may_be_incomplete_relative_to_primary_publication",
        ]
        output.append(
            {
                "candidate_rank": int(candidate["candidate_rank"]),
                "candidate_id": str(candidate["candidate_id"]),
                "observation_id": observation_id,
                "structure_id": str(row["structure_id"]),
                "standardized_smiles": str(row["standardized_smiles"]),
                "standard_inchi_key": str(row["standard_inchi_key"]),
                "automated_evidence_status": AUTOMATED_STATUS,
                "human_adjudication_status": HUMAN_STATUS,
                "automated_review_priority": priority,
                "automated_evidence_score": score,
                "master_wild_type_evidence_scope": str(row["wild_type_evidence_scope"]),
                "target_evidence_class": target_class,
                "target_status_limitations_json": _canonical_json(target_limitations),
                "target_chembl_id": source.get("target_chembl_id"),
                "target_type": source.get("target_type"),
                "target_organism": source.get("target_organism"),
                "target_tax_id": source.get("target_tax_id"),
                "component_accessions": source.get("component_accessions"),
                "target_relationship_type": source.get("target_relationship_type"),
                "target_variant_id": source.get("target_variant_id"),
                "explicit_wild_type_text_evidence": explicit_wt,
                "explicit_variant_text_warning": explicit_variant,
                "source_binding_status": str(source["source_binding_status"]),
                "source_artifact_role": str(source["source_artifact_role"]),
                "source_artifact_name": str(source["source_artifact_name"]),
                "source_artifact_row_number": int(source["source_artifact_row_number"]),
                "source_family": str(row["source_family"]),
                "source_record_id": str(row["source_record_id"]),
                "document_id": source.get("document_id"),
                "document_doi": source.get("document_doi"),
                "document_pubmed_id": source.get("document_pubmed_id"),
                "document_patent_id": source.get("document_patent_id"),
                "document_title": source.get("document_title"),
                "document_type": source.get("document_type"),
                "document_year": source.get("document_year"),
                "assay_id": row.get("assay_id"),
                "assay_family": str(row["assay_family"]),
                "measurement_modality": str(row["measurement_modality"]),
                "automation_class": str(row["automation_class"]),
                "assay_description": source.get("assay_description"),
                "protocol_completeness_score": int(protocol.get("protocol_completeness_score", 0)),
                "protocol_unresolved_fields_json": str(protocol.get("unresolved_fields_json", "[]")),
                "protocol_evidence_json": _protocol_evidence_json(protocol),
                "native_endpoint": str(row["native_endpoint"]),
                "native_relation": row.get("native_relation"),
                "native_value": row.get("native_value"),
                "native_unit": row.get("native_unit"),
                "potency_relation_pic50": str(row["potency_relation_pic50"]),
                "potency_pic50_point": row.get("potency_pic50_point"),
                "potency_pic50_lower_bound": row.get("potency_pic50_lower_bound"),
                "potency_pic50_upper_bound": row.get("potency_pic50_upper_bound"),
                "potency_censoring": str(row["potency_censoring"]),
                "relation_unit_audit_status": str(relation_audit["status"]),
                "expected_pic50_from_native": relation_audit["expected_pic50"],
                "pic50_conversion_absolute_delta": relation_audit["absolute_delta"],
                "structure_exact_pic50_observation_count": len(context),
                "structure_distinct_source_count": len(context_sources),
                "structure_distinct_assay_count": len(context_assays),
                "structure_distinct_known_document_count": len(context_documents),
                "structure_known_document_ids_json": _canonical_json(context_documents),
                "lineage_group_count": len(groups),
                "lineage_group_ids_json": _canonical_json(lineage_ids),
                "lineage_classes_json": _canonical_json(lineage_classes),
                "maximum_lineage_evidence_strength": strength,
                "equal_value_group_size": int(value_cluster["size"]),
                "equal_value_group_source_count": int(value_cluster["source_count"]),
                "equal_value_group_known_document_count": int(value_cluster["known_document_count"]),
                "model_split": str(row["model_split"]),
                "source_declared_split": row.get("source_split"),
                "source_input_member": str(source["source_input_member"]),
                "structure_model_splits_json": _canonical_json(split_surfaces["structure"]),
                "scaffold_model_splits_json": _canonical_json(split_surfaces["scaffold"]),
                "assay_model_splits_json": _canonical_json(split_surfaces["assay"]),
                "document_model_splits_json": _canonical_json(split_surfaces["document"]),
                "lineage_model_splits_json": _canonical_json(split_surfaces["lineage"]),
                "split_cautions_json": _canonical_json(split_cautions),
                "automated_review_reasons_json": _canonical_json(sorted(set(reasons))),
                "required_human_checks_json": _canonical_json(sorted(set(required_checks))),
                "automated_evidence_limitations_json": _canonical_json(limitations),
            }
        )
    return output


def _conflict_rows(
    conflicts: Sequence[Mapping[str, Any]],
    observations: Mapping[str, Mapping[str, Any]],
    source_evidence: Mapping[str, Mapping[str, Any]],
    protocols: Mapping[tuple[str, str, str], Mapping[str, Any]],
    groups_by_observation: Mapping[str, Sequence[Mapping[str, Any]]],
    value_clusters: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for conflict in sorted(conflicts, key=lambda row: int(row["review_rank"])):
        observation_ids = [str(value) for value in _json_list(conflict["observation_ids_json"])]
        rows = [observations[observation_id] for observation_id in observation_ids]
        sources = [source_evidence[observation_id] for observation_id in observation_ids]
        audits = [_relation_unit_audit(row) for row in rows]
        target_results = [_target_evidence(row, source) for row, source in zip(rows, sources, strict=True)]
        protocols_for_rows = [_protocol_for(row, protocols) for row in rows]
        group_map: dict[str, Mapping[str, Any]] = {}
        for observation_id in observation_ids:
            for group in groups_by_observation.get(observation_id, []):
                group_map[str(group["lineage_group_id"])] = group
        groups = list(group_map.values())
        lineage_classes = Counter(str(group["automated_lineage_class"]) for group in groups)
        mirror_groups = sum("cross_source" in name for name in lineage_classes.elements())
        source_record_groups = sum(group["lineage_group_kind"] == "source_record_reuse" for group in groups)
        same_source_groups = sum(
            group["lineage_group_kind"] == "exact_native_measurement_signature"
            or str(group["automated_lineage_class"]).startswith("within_source")
            or "same_source_record" in str(group["automated_lineage_class"])
            for group in groups
        )
        grouped_observations = {
            str(observation_id)
            for group in groups
            for observation_id in _json_list(group["observation_ids_json"])
        } & set(observation_ids)
        documents = _sorted_text(source.get("document_id") for source in sources)
        dois = _sorted_text(source.get("document_doi") for source in sources)
        assays = _sorted_text(row.get("assay_id") for row in rows)
        source_families = _sorted_text(row["source_family"] for row in rows)
        target_classes = Counter(result[0] for result in target_results)
        relationship_types = _sorted_text(source.get("target_relationship_type") for source in sources)
        audit_counts = Counter(str(audit["status"]) for audit in audits)
        inconsistent = sum(count for key, count in audit_counts.items() if key.startswith("inconsistent"))
        completeness = [
            int(protocol.get("protocol_completeness_score", 0)) for protocol in protocols_for_rows
        ]
        model_splits = _sorted_text(row.get("model_split") for row in rows)
        source_splits = _sorted_text(row.get("source_split") for row in rows)
        input_members = _sorted_text(source["source_input_member"] for source in sources)
        lineage_splits = _sorted_text(
            split for group in groups for split in _json_list(group["model_splits_json"])
        )
        value_cluster_ids = {
            str(value_clusters[observation_id]["value_cluster_id"]) for observation_id in observation_ids
        }
        cautions: list[str] = []
        if model_splits == ["train"]:
            cautions.append("conflict_structure_currently_in_training_partition")
        elif model_splits == ["validation"]:
            cautions.append("conflict_structure_currently_in_validation_partition")
        if len(model_splits) > 1:
            cautions.append("same_structure_crosses_master_model_partitions")
        if len(lineage_splits) > 1:
            cautions.append("automated_lineage_group_spans_master_model_partitions")
        if len(source_splits) > 1:
            cautions.append("source_compilation_partitions_disagree_within_structure")
        if len(input_members) > 1:
            cautions.append("structure_occurs_in_multiple_manifest_bound_source_inputs")
        cautions.append("human_resolution_must_precede_any_consensus_label_or_split_refreeze")
        reasons = [
            f"exact_pic50_range_{float(conflict['pic50_range']):.6g}",
            f"distinct_exact_value_clusters_{len(value_cluster_ids)}",
        ]
        if mirror_groups:
            reasons.append("cross_source_equal_value_mirror_candidates_present")
        if same_source_groups:
            reasons.append("same_source_duplicate_candidates_present")
        if source_record_groups:
            reasons.append("source_primary_key_reuse_present")
        if inconsistent:
            reasons.append("relation_or_unit_inconsistency_present")
        if len(documents) > 1:
            reasons.append("multiple_known_documents_require_comparability_review")
        if len(documents) < len(rows):
            reasons.append("document_provenance_missing_for_one_or_more_observations")
        score = (
            float(conflict["priority_score"])
            + mirror_groups * 20.0
            + source_record_groups * 15.0
            + same_source_groups * 8.0
            + inconsistent * 50.0
            + min(20.0, len(documents) * 2.0)
        )
        if conflict["review_priority"] in {"critical", "high"} or inconsistent:
            priority = "P0_large_disagreement_or_integrity_review"
        elif conflict["review_priority"] == "moderate" or mirror_groups or source_record_groups:
            priority = "P1_lineage_and_protocol_comparability_review"
        else:
            priority = "P2_lower_range_duplicate_lineage_review"
        required_checks = [
            "compare_primary_documents_and_source_rows_to_determine_common_or_independent_lineage",
            "verify_each_native_value_relation_unit_and_pic50_conversion",
            "verify_human_wild_type_kcnh2_construct_for_each_assay",
            "compare_cell_platform_voltage_temperature_timing_and_recording_configuration",
            "classify_measurements_as_duplicate_comparable_protocol_dependent_incomparable_or_unresolved",
            "do_not_average_or_collapse_until_human_decision_is_recorded",
            "after_resolution_refreeze_all_measurement_document_assay_structure_and_scaffold_lineages",
        ]
        limitations = [
            "automated_lineage_groups_are_candidates_not_proven_duplicates",
            "distinct_documents_sources_and_assays_are_not_assumed_independent",
            "missing_document_metadata_does_not_mean_missing_primary_document",
            "equal_values_can_be_rounding_or_legitimate_replicates",
            "no_consensus_or_exclusion_is_authorized_by_this_table",
        ]
        output.append(
            {
                "review_rank": int(conflict["review_rank"]),
                "review_id": str(conflict["review_id"]),
                "structure_id": str(conflict["structure_id"]),
                "standardized_smiles": str(conflict["standardized_smiles"]),
                "standard_inchi_key": str(conflict["standard_inchi_key"]),
                "original_review_priority": str(conflict["review_priority"]),
                "original_priority_score": float(conflict["priority_score"]),
                "exact_replicate_count": int(conflict["exact_replicate_count"]),
                "pic50_minimum": float(conflict["pic50_minimum"]),
                "pic50_maximum": float(conflict["pic50_maximum"]),
                "pic50_range": float(conflict["pic50_range"]),
                "automated_evidence_status": AUTOMATED_STATUS,
                "human_adjudication_status": HUMAN_STATUS,
                "automated_review_priority": priority,
                "automated_review_score": score,
                "observation_count_bound_to_local_source": sum(
                    source["source_binding_status"] != "unbound_local_source_record" for source in sources
                ),
                "distinct_source_count": len(source_families),
                "distinct_assay_count": len(assays),
                "distinct_known_document_count": len(documents),
                "unknown_document_observation_count": sum(
                    not source.get("document_id") for source in sources
                ),
                "known_document_ids_json": _canonical_json(documents),
                "known_document_dois_json": _canonical_json(dois),
                "exact_value_cluster_count": len(value_cluster_ids),
                "lineage_group_count": len(groups),
                "lineage_class_counts_json": _canonical_json(dict(sorted(lineage_classes.items()))),
                "cross_source_mirror_group_count": mirror_groups,
                "same_source_duplicate_group_count": same_source_groups,
                "source_record_reuse_group_count": source_record_groups,
                "observations_in_any_lineage_group": len(grouped_observations),
                "lineage_inflation_caution": bool(groups),
                "target_evidence_class_counts_json": _canonical_json(dict(sorted(target_classes.items()))),
                "target_relationship_types_json": _canonical_json(relationship_types),
                "explicit_wild_type_text_observation_count": sum(result[1] for result in target_results),
                "relation_unit_audit_counts_json": _canonical_json(dict(sorted(audit_counts.items()))),
                "relation_unit_inconsistency_count": inconsistent,
                "minimum_protocol_completeness": min(completeness),
                "median_protocol_completeness": statistics.median(completeness),
                "maximum_protocol_completeness": max(completeness),
                "fully_complete_protocol_observation_count": sum(value == 6 for value in completeness),
                "model_splits_json": _canonical_json(model_splits),
                "source_declared_splits_json": _canonical_json(source_splits),
                "source_input_members_json": _canonical_json(input_members),
                "lineage_model_splits_json": _canonical_json(lineage_splits),
                "split_cautions_json": _canonical_json(sorted(set(cautions))),
                "automated_review_reasons_json": _canonical_json(sorted(set(reasons))),
                "required_human_checks_json": _canonical_json(required_checks),
                "automated_evidence_limitations_json": _canonical_json(limitations),
            }
        )
    return output


def _decision_contract() -> dict[str, Any]:
    return {
        "schema_version": "platform-herg-human-adjudication-decision/1.0",
        "status": "blank_template_no_human_decisions_recorded",
        "candidate_allowed_decisions": [
            {
                "decision": "accept_for_future_gold_after_required_holdout_refreeze",
                "meaning": (
                    "Primary evidence verifies the label, explicit human wild-type KCNH2 construct, "
                    "assay context, and lineage; acceptance is still conditional on sealed holdout refreezing."
                ),
            },
            {
                "decision": "retain_non_gold_sensitivity_only",
                "meaning": "Useful reported evidence remains, but it is not suitable for a locked gold panel.",
            },
            {
                "decision": "reject_from_evaluation_candidate_pool",
                "meaning": "Primary evidence establishes a material target, assay, value, or lineage defect.",
            },
            {
                "decision": "defer_missing_primary_evidence",
                "meaning": "The available local record is insufficient for a defensible decision.",
            },
        ],
        "conflict_allowed_decisions": [
            {
                "decision": "same_experiment_or_database_mirror_link_lineage",
                "meaning": "Rows represent the same underlying measurement lineage; preserve provenance.",
            },
            {
                "decision": "comparable_independent_measurements_consensus_eligible",
                "meaning": "Primary sources support comparability and distinct experimental origins.",
            },
            {
                "decision": "protocol_dependent_keep_separate",
                "meaning": "Both values are credible but protocol context materially differs.",
            },
            {
                "decision": "incomparable_target_or_endpoint_keep_separate",
                "meaning": "Target construct, modality, or endpoint semantics are not comparable.",
            },
            {
                "decision": "source_transcription_or_conversion_error_requires_correction",
                "meaning": "A verified source or transformation error must be corrected in a future release.",
            },
            {
                "decision": "unresolved_do_not_aggregate",
                "meaning": "Evidence remains insufficient; values must not be collapsed or averaged.",
            },
        ],
        "required_human_fields": [
            "human_decision",
            "human_reviewer",
            "human_reviewed_at",
            "human_notes",
        ],
        "nonpromotion_rule": (
            "This v1.5 release contains no completed human decisions and cannot promote a candidate to gold, "
            "collapse a lineage, correct a source value, or authorize a consensus."
        ),
    }


def _packet_rows(
    candidate_rows: Sequence[Mapping[str, Any]],
    conflict_rows: Sequence[Mapping[str, Any]],
    original_conflicts: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, str]]:
    candidate_decisions = _canonical_json(
        [item["decision"] for item in _decision_contract()["candidate_allowed_decisions"]]
    )
    conflict_decisions = _canonical_json(
        [item["decision"] for item in _decision_contract()["conflict_allowed_decisions"]]
    )
    rows: list[dict[str, str]] = []
    for candidate in candidate_rows:
        rows.append(
            {
                "review_rank": "0",
                "item_type": "evaluation_candidate_observation",
                "item_id": str(candidate["candidate_id"]),
                "structure_id": str(candidate["structure_id"]),
                "observation_id": str(candidate["observation_id"]),
                "automated_review_priority": str(candidate["automated_review_priority"]),
                "automated_review_score": format(float(candidate["automated_evidence_score"]), ".12g"),
                "automated_evidence_status": AUTOMATED_STATUS,
                "human_adjudication_status": HUMAN_STATUS,
                "source_family": str(candidate["source_family"]),
                "source_record_ids": _canonical_json([candidate["source_record_id"]]),
                "assay_ids": _canonical_json([candidate["assay_id"]] if candidate.get("assay_id") else []),
                "document_ids": _canonical_json(
                    [candidate["document_id"]] if candidate.get("document_id") else []
                ),
                "document_dois": _canonical_json(
                    [candidate["document_doi"]] if candidate.get("document_doi") else []
                ),
                "target_evidence_summary": _canonical_json(
                    {
                        "class": candidate["target_evidence_class"],
                        "relationship_type": candidate["target_relationship_type"],
                        "explicit_wild_type_text": candidate["explicit_wild_type_text_evidence"],
                        "master_scope": candidate["master_wild_type_evidence_scope"],
                    }
                ),
                "measurement_summary": _canonical_json(
                    {
                        "native_endpoint": candidate["native_endpoint"],
                        "native_relation": candidate["native_relation"],
                        "native_value": candidate["native_value"],
                        "native_unit": candidate["native_unit"],
                        "pic50_relation": candidate["potency_relation_pic50"],
                        "pic50_point": candidate["potency_pic50_point"],
                        "pic50_lower_bound": candidate["potency_pic50_lower_bound"],
                        "pic50_upper_bound": candidate["potency_pic50_upper_bound"],
                        "conversion_audit": candidate["relation_unit_audit_status"],
                    }
                ),
                "lineage_summary": _canonical_json(
                    {
                        "group_count": candidate["lineage_group_count"],
                        "classes": _json_list(candidate["lineage_classes_json"]),
                        "equal_value_group_size": candidate["equal_value_group_size"],
                    }
                ),
                "protocol_summary": _canonical_json(
                    {
                        "completeness": candidate["protocol_completeness_score"],
                        "unresolved": _json_list(candidate["protocol_unresolved_fields_json"]),
                    }
                ),
                "split_cautions": str(candidate["split_cautions_json"]),
                "required_human_checks": str(candidate["required_human_checks_json"]),
                "allowed_decisions": candidate_decisions,
                "human_decision": "",
                "human_reviewer": "",
                "human_reviewed_at": "",
                "human_notes": "",
            }
        )
    for conflict in conflict_rows:
        original = original_conflicts[str(conflict["review_id"])]
        rows.append(
            {
                "review_rank": "0",
                "item_type": "replicated_pic50_conflict_structure",
                "item_id": str(conflict["review_id"]),
                "structure_id": str(conflict["structure_id"]),
                "observation_id": "",
                "automated_review_priority": str(conflict["automated_review_priority"]),
                "automated_review_score": format(float(conflict["automated_review_score"]), ".12g"),
                "automated_evidence_status": AUTOMATED_STATUS,
                "human_adjudication_status": HUMAN_STATUS,
                "source_family": str(original["source_families_json"]),
                "source_record_ids": str(original["source_record_ids_json"]),
                "assay_ids": str(original["assay_ids_json"]),
                "document_ids": str(conflict["known_document_ids_json"]),
                "document_dois": str(conflict["known_document_dois_json"]),
                "target_evidence_summary": _canonical_json(
                    {
                        "classes": _json_object(conflict["target_evidence_class_counts_json"]),
                        "relationship_types": _json_list(conflict["target_relationship_types_json"]),
                        "explicit_wild_type_text_observations": conflict[
                            "explicit_wild_type_text_observation_count"
                        ],
                    }
                ),
                "measurement_summary": _canonical_json(
                    {
                        "exact_replicates": conflict["exact_replicate_count"],
                        "pic50_minimum": conflict["pic50_minimum"],
                        "pic50_maximum": conflict["pic50_maximum"],
                        "pic50_range": conflict["pic50_range"],
                        "value_clusters": conflict["exact_value_cluster_count"],
                        "relation_unit_audits": _json_object(conflict["relation_unit_audit_counts_json"]),
                    }
                ),
                "lineage_summary": _canonical_json(
                    {
                        "group_count": conflict["lineage_group_count"],
                        "class_counts": _json_object(conflict["lineage_class_counts_json"]),
                        "cross_source_mirror_groups": conflict["cross_source_mirror_group_count"],
                        "source_record_reuse_groups": conflict["source_record_reuse_group_count"],
                    }
                ),
                "protocol_summary": _canonical_json(
                    {
                        "minimum_completeness": conflict["minimum_protocol_completeness"],
                        "median_completeness": conflict["median_protocol_completeness"],
                        "maximum_completeness": conflict["maximum_protocol_completeness"],
                    }
                ),
                "split_cautions": str(conflict["split_cautions_json"]),
                "required_human_checks": str(conflict["required_human_checks_json"]),
                "allowed_decisions": conflict_decisions,
                "human_decision": "",
                "human_reviewer": "",
                "human_reviewed_at": "",
                "human_notes": "",
            }
        )
    priority_order = {"P0": 0, "P1": 1, "P2": 2}
    rows.sort(
        key=lambda row: (
            priority_order.get(row["automated_review_priority"][:2], 9),
            -float(row["automated_review_score"]),
            row["item_type"],
            row["item_id"],
        )
    )
    for rank, row in enumerate(rows, start=1):
        row["review_rank"] = str(rank)
    return rows


def _write_packet(path: Path, rows: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_PACKET_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return _artifact(path, rows=len(rows))


def _counts(
    candidate_rows: Sequence[Mapping[str, Any]],
    conflict_rows: Sequence[Mapping[str, Any]],
    lineage_rows: Sequence[Mapping[str, Any]],
    packet_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    target_classes = Counter(str(row["target_evidence_class"]) for row in candidate_rows)
    candidate_priorities = Counter(str(row["automated_review_priority"]) for row in candidate_rows)
    candidate_splits = Counter(str(row["model_split"]) for row in candidate_rows)
    candidate_strengths = Counter(str(row["maximum_lineage_evidence_strength"]) for row in candidate_rows)
    relation_audits = Counter(str(row["relation_unit_audit_status"]) for row in candidate_rows)
    lineage_classes = Counter(str(row["automated_lineage_class"]) for row in lineage_rows)
    lineage_kinds = Counter(str(row["lineage_group_kind"]) for row in lineage_rows)
    lineage_strengths = Counter(str(row["automated_evidence_strength"]) for row in lineage_rows)
    return {
        "candidate_rows": len(candidate_rows),
        "candidate_structures": len({str(row["structure_id"]) for row in candidate_rows}),
        "candidate_target_evidence_class_counts": dict(sorted(target_classes.items())),
        "candidate_direct_human_kcnh2_no_variant_annotation": target_classes[
            "direct_human_kcnh2_single_protein_no_variant_annotation"
        ],
        "candidate_homologue_relationship_no_variant_annotation": target_classes[
            "homologue_relationship_to_human_kcnh2_no_variant_annotation"
        ],
        "candidate_explicit_wild_type_text_evidence": sum(
            bool(row["explicit_wild_type_text_evidence"]) for row in candidate_rows
        ),
        "candidate_document_known": sum(bool(row["document_id"]) for row in candidate_rows),
        "candidate_doi_known": sum(bool(row["document_doi"]) for row in candidate_rows),
        "candidate_pubmed_known": sum(bool(row["document_pubmed_id"]) for row in candidate_rows),
        "candidate_review_priority_counts": dict(sorted(candidate_priorities.items())),
        "candidate_model_split_counts": dict(sorted(candidate_splits.items())),
        "candidate_lineage_strength_counts": dict(sorted(candidate_strengths.items())),
        "candidate_relation_unit_audit_counts": dict(sorted(relation_audits.items())),
        "candidate_relation_unit_inconsistencies": sum(
            count for key, count in relation_audits.items() if key.startswith("inconsistent")
        ),
        "candidate_in_any_lineage_group": sum(int(row["lineage_group_count"]) > 0 for row in candidate_rows),
        "conflict_rows": len(conflict_rows),
        "conflict_structures_with_any_lineage_group": sum(
            int(row["lineage_group_count"]) > 0 for row in conflict_rows
        ),
        "conflict_structures_with_cross_source_mirror_candidates": sum(
            int(row["cross_source_mirror_group_count"]) > 0 for row in conflict_rows
        ),
        "conflict_structures_with_source_record_reuse": sum(
            int(row["source_record_reuse_group_count"]) > 0 for row in conflict_rows
        ),
        "conflict_relation_unit_inconsistencies": sum(
            int(row["relation_unit_inconsistency_count"]) for row in conflict_rows
        ),
        "conflict_observations_bound_to_local_sources": sum(
            int(row["observation_count_bound_to_local_source"]) for row in conflict_rows
        ),
        "lineage_groups": len(lineage_rows),
        "lineage_group_kind_counts": dict(sorted(lineage_kinds.items())),
        "lineage_class_counts": dict(sorted(lineage_classes.items())),
        "lineage_strength_counts": dict(sorted(lineage_strengths.items())),
        "lineage_groups_touching_queue_observation": sum(
            int(row["queue_observation_count"]) > 0 for row in lineage_rows
        ),
        "human_packet_rows": len(packet_rows),
        "human_decisions_completed": 0,
        "gold_rows_promoted": 0,
    }


def _report_text(counts: Mapping[str, Any]) -> str:
    candidate_targets = counts["candidate_target_evidence_class_counts"]
    lineage_kinds = counts["lineage_group_kind_counts"]
    lineage_strengths = counts["lineage_strength_counts"]
    return "\n".join(
        [
            "# hERG candidate and conflict adjudication evidence",
            "",
            "## Outcome",
            "",
            f"This release adds deterministic local evidence for all {counts['candidate_rows']:,} evaluation candidates and {counts['conflict_rows']:,} pIC50 conflict structures. It records **zero human decisions** and promotes **zero rows to gold**.",
            "",
            "## Candidate evidence",
            "",
            f"- Candidate observations: {counts['candidate_rows']:,} across {counts['candidate_structures']:,} structures.",
            f"- Direct human KCNH2 single-protein target metadata with no variant annotation: {counts['candidate_direct_human_kcnh2_no_variant_annotation']:,}.",
            f"- Homologue relationship to human KCNH2 with no variant annotation: {counts['candidate_homologue_relationship_no_variant_annotation']:,}.",
            f"- Explicit wild-type wording in the locally bound assay description: {counts['candidate_explicit_wild_type_text_evidence']:,}.",
            f"- Known ChEMBL document IDs: {counts['candidate_document_known']:,}; DOI available: {counts['candidate_doi_known']:,}; PubMed ID available: {counts['candidate_pubmed_known']:,}.",
            f"- Candidates participating in at least one automated lineage group: {counts['candidate_in_any_lineage_group']:,}.",
            f"- Automated relation/unit inconsistencies: {counts['candidate_relation_unit_inconsistencies']:,}.",
            f"- Target evidence classes: `{_canonical_json(candidate_targets)}`.",
            f"- Existing model partitions: `{_canonical_json(counts['candidate_model_split_counts'])}`.",
            "",
            "A null ChEMBL variant identifier is only an absence of variant annotation. It is not explicit wild-type confirmation. Likewise, relationship type `H` is a homologue relationship and must be resolved against the primary document before any evaluation use.",
            "",
            "## Conflict and lineage evidence",
            "",
            f"- Conflict structures with at least one automated duplicate/mirror/source-reuse group: {counts['conflict_structures_with_any_lineage_group']:,}.",
            f"- Conflict structures with cross-source exact-value mirror candidates: {counts['conflict_structures_with_cross_source_mirror_candidates']:,}.",
            f"- Conflict structures with source-primary-key reuse: {counts['conflict_structures_with_source_record_reuse']:,}.",
            f"- Conflict observation bindings replayed to manifest-bound local sources: {counts['conflict_observations_bound_to_local_sources']:,}.",
            f"- Lineage groups: {counts['lineage_groups']:,}; groups touching a queued observation: {counts['lineage_groups_touching_queue_observation']:,}.",
            f"- Lineage group kinds: `{_canonical_json(lineage_kinds)}`.",
            f"- Automated evidence strengths: `{_canonical_json(lineage_strengths)}`.",
            "",
            "Equal standardized values, shared source keys, shared assay/document context, and cross-database matches are evidence for review—not proof of a common experiment. No rows were collapsed, averaged, corrected, excluded, or relabeled.",
            "",
            "## Human packet",
            "",
            f"The packet contains {counts['human_packet_rows']:,} pending items. Every decision, reviewer, review date, and note field is intentionally blank. The accompanying decision contract defines allowable dispositions and requires primary-source verification, target/construct resolution, protocol comparison, and a new structure/scaffold/assay/document/measurement-lineage freeze before any accepted candidate can enter an evaluation panel.",
            "",
            "## Limits",
            "",
            "- This is local automated evidence only; no primary paper was manually read in this build.",
            "- Distinct sources, documents, assays, or rows are not assumed independent.",
            "- Absence of an automated duplicate signal is not proof of independence.",
            "- Missing document or protocol fields may be recoverable from primary sources, but are not inferred here.",
            "- Current train/validation/test labels are cautions, not authorization to move a row into a sealed test set.",
            "- Any future gold release must be separately versioned after completed human decisions and leakage-safe refreezing.",
            "",
            "No model, feature generator, smoke test, or HPC job was run.",
            "",
        ]
    )


def build_herg_candidate_adjudication(
    *, master_root: Path, pre_hpc_root: Path, output_root: Path, report_root: Path
) -> dict[str, Any]:
    """Build a versioned evidence layer from frozen v1.3 and v1.4 inputs."""

    master = master_root.resolve()
    pre_hpc = pre_hpc_root.resolve()
    validate_herg_master_dataset(master)
    validate_herg_pre_hpc_assets(pre_hpc)
    master_manifest_path = _checked_file(master / "herg_master_manifest.json")
    pre_hpc_manifest_path = _checked_file(pre_hpc / "herg_pre_hpc_assets_manifest.json")
    observation_path = _checked_file(master / "observation_master.parquet")
    protocol_path = _checked_file(master / "assay_protocol_index.parquet")
    candidate_path = _checked_file(pre_hpc / "gold_standard_evaluation_candidates.parquet")
    conflict_path = _checked_file(pre_hpc / "replicated_pic50_conflict_review_queue.parquet")
    master_manifest = json.loads(master_manifest_path.read_text(encoding="utf-8"))
    pre_manifest = json.loads(pre_hpc_manifest_path.read_text(encoding="utf-8"))
    source_inputs, source_lineage_manifest_path = _source_inputs(master_manifest)

    bound_master_hashes = {
        Path(str(item["path"])).name: str(item["sha256"])
        for item in pre_manifest.get("inputs", [])
        if "sha256" in item
    }
    for path in (observation_path, protocol_path, master_manifest_path):
        if bound_master_hashes.get(path.name) != _sha256_file(path):
            raise HergCandidateAdjudicationError(
                f"v1.4 asset release is not bound to supplied v1.3 input: {path.name}"
            )

    output = output_root.resolve()
    report = report_root.resolve()
    if output.exists() or report.exists():
        raise HergCandidateAdjudicationError("output_root and report_root must not already exist")
    output.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    report_staging = Path(tempfile.mkdtemp(prefix=f".{report.name}.", dir=report.parent))
    try:
        schema_names = set(pq.read_schema(observation_path).names)
        missing = sorted(set(_OBSERVATION_COLUMNS) - schema_names)
        if missing:
            raise HergCandidateAdjudicationError(f"observation master is missing columns: {missing}")
        all_rows = pq.read_table(observation_path, columns=_OBSERVATION_COLUMNS).to_pylist()
        candidates = pq.read_table(candidate_path).to_pylist()
        conflicts = pq.read_table(conflict_path).to_pylist()
        candidate_ids = {str(row["observation_id"]) for row in candidates}
        conflict_ids = {
            str(observation_id)
            for row in conflicts
            for observation_id in _json_list(row["observation_ids_json"])
        }
        queue_ids = candidate_ids | conflict_ids
        observation_by_id = {str(row["observation_id"]): row for row in all_rows}
        if len(observation_by_id) != len(all_rows):
            raise HergCandidateAdjudicationError("observation master has duplicate observation IDs")
        missing_queue = sorted(queue_ids - set(observation_by_id))
        if missing_queue:
            raise HergCandidateAdjudicationError(
                f"queue observations missing from master: {missing_queue[:5]}"
            )
        relevant_structures = {
            str(observation_by_id[observation_id]["structure_id"]) for observation_id in queue_ids
        }
        context_rows = [
            row
            for row in all_rows
            if row.get("structure_id") in relevant_structures
            and row["endpoint_standardization_status"] in {"exact_standardized", "censored_standardized"}
            and _pic50_coordinate(row) is not None
        ]
        context_by_id = {str(row["observation_id"]): row for row in context_rows}
        if not queue_ids.issubset(context_by_id):
            raise HergCandidateAdjudicationError("queue contains a row outside standardized pIC50 context")
        source_evidence = _load_source_evidence(source_inputs, context_rows)
        protocols = _protocol_map(protocol_path)
        split_indices = _split_indices(all_rows)
        structure_context: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in context_rows:
            structure_context[str(row["structure_id"])].append(row)
        lineage, groups_by_observation, value_clusters = _build_lineage_evidence(
            context_rows, source_evidence, queue_ids
        )
        enriched_candidates = _candidate_rows(
            candidates,
            observation_by_id,
            structure_context,
            source_evidence,
            protocols,
            split_indices,
            groups_by_observation,
            value_clusters,
        )
        enriched_conflicts = _conflict_rows(
            conflicts,
            observation_by_id,
            source_evidence,
            protocols,
            groups_by_observation,
            value_clusters,
        )
        original_conflicts = {str(row["review_id"]): row for row in conflicts}
        packet = _packet_rows(enriched_candidates, enriched_conflicts, original_conflicts)
        counts = _counts(enriched_candidates, enriched_conflicts, lineage, packet)
        counts.update(
            {
                "queue_observation_union": len(queue_ids),
                "queue_candidate_observations": len(candidate_ids),
                "queue_conflict_observations": len(conflict_ids),
                "relevant_standardized_context_observations": len(context_rows),
                "relevant_structures": len(relevant_structures),
                "local_source_bindings": len(source_evidence),
                "local_source_binding_status_counts": dict(
                    sorted(
                        Counter(
                            str(item["source_binding_status"]) for item in source_evidence.values()
                        ).items()
                    )
                ),
            }
        )

        artifacts = {
            CANDIDATE_OUTPUT: _write_parquet(
                staging / CANDIDATE_OUTPUT, enriched_candidates, _CANDIDATE_SCHEMA
            ),
            CONFLICT_OUTPUT: _write_parquet(staging / CONFLICT_OUTPUT, enriched_conflicts, _CONFLICT_SCHEMA),
            LINEAGE_OUTPUT: _write_parquet(staging / LINEAGE_OUTPUT, lineage, _LINEAGE_SCHEMA),
            PACKET_OUTPUT: _write_packet(staging / PACKET_OUTPUT, packet),
        }
        contract = _decision_contract()
        contract_path = staging / DECISION_CONTRACT_OUTPUT
        contract_path.write_text(_canonical_json(contract) + "\n", encoding="utf-8")
        artifacts[DECISION_CONTRACT_OUTPUT] = _artifact(contract_path)
        report_text = _report_text(counts)
        report_path = report_staging / REPORT_NAME
        report_path.write_text(report_text, encoding="utf-8")

        inputs = [
            _input_binding(master_manifest_path, role="v1_3_master_manifest"),
            _input_binding(observation_path, role="v1_3_observation_master"),
            _input_binding(protocol_path, role="v1_3_assay_protocol_index"),
            _input_binding(pre_hpc_manifest_path, role="v1_4_pre_hpc_assets_manifest"),
            _input_binding(candidate_path, role="v1_4_evaluation_candidates"),
            _input_binding(conflict_path, role="v1_4_pic50_conflict_queue"),
        ]
        if source_lineage_manifest_path is not None:
            inputs.append(
                _input_binding(
                    source_lineage_manifest_path,
                    role="v1_observation_ledger_lineage_manifest",
                )
            )
        for index, item in enumerate(source_inputs, start=1):
            inputs.append(
                _input_binding(
                    Path(str(item["path"])).resolve(),
                    role=f"master_manifest_bound_{item['role']}_{index:02d}",
                )
            )
        body: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "dataset_id": DATASET_ID,
            "inputs": inputs,
            "policies": {
                "automated_evidence_only": True,
                "human_adjudication_completed": False,
                "gold_promotion_allowed": False,
                "lineage_group_semantics": "review_candidates_not_proven_duplicates",
                "distinct_sources_documents_assays_independent_by_default": False,
                "pic50_numeric_equivalence_tolerance": PIC50_EQUIVALENCE_TOLERANCE,
                "target_variant_null_semantics": "absence_of_annotation_not_explicit_wild_type",
                "split_policy": "no_membership_changes; cautions_only; refreeze_required_after_adjudication",
            },
            "counts": counts,
            "artifacts": artifacts,
            "report_artifact": {
                "path": REPORT_NAME,
                "bytes": report_path.stat().st_size,
                "sha256": _sha256_file(report_path),
            },
        }
        manifest = dict(body)
        manifest["manifest_sha256"] = _manifest_digest(body)
        (staging / MANIFEST_NAME).write_text(_canonical_json(manifest) + "\n", encoding="utf-8")
        os.replace(staging, output)
        os.replace(report_staging, report)
        validate_herg_candidate_adjudication(output, report_root=report)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        shutil.rmtree(report_staging, ignore_errors=True)
        raise


def validate_herg_candidate_adjudication(
    output_root: Path, *, report_root: Path | None = None
) -> dict[str, Any]:
    """Validate input bindings, artifact hashes/schemas, and non-adjudication invariants."""

    root = output_root.resolve()
    manifest_path = _checked_file(root / MANIFEST_NAME)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("manifest_sha256") != _manifest_digest(manifest):
        raise HergCandidateAdjudicationError("manifest digest mismatch")
    roles: dict[str, Path] = {}
    for binding in manifest.get("inputs", []):
        path = _checked_file(Path(str(binding["path"])))
        role = str(binding["role"])
        if role in roles:
            raise HergCandidateAdjudicationError(f"duplicate input role: {role}")
        roles[role] = path
        if path.stat().st_size != int(binding["bytes"]) or _sha256_file(path) != binding["sha256"]:
            raise HergCandidateAdjudicationError(f"input binding mismatch: {path}")
        if path.suffix.casefold() == ".parquet":
            if pq.ParquetFile(path).metadata.num_rows != int(binding["rows"]):
                raise HergCandidateAdjudicationError(f"input row count mismatch: {path}")
            if _schema_sha256(pq.read_schema(path)) != binding["arrow_schema_sha256"]:
                raise HergCandidateAdjudicationError(f"input schema mismatch: {path}")
    validate_herg_master_dataset(roles["v1_3_master_manifest"].parent)
    validate_herg_pre_hpc_assets(roles["v1_4_pre_hpc_assets_manifest"].parent)

    schemas = {
        CANDIDATE_OUTPUT: _CANDIDATE_SCHEMA,
        CONFLICT_OUTPUT: _CONFLICT_SCHEMA,
        LINEAGE_OUTPUT: _LINEAGE_SCHEMA,
    }
    for name, schema in schemas.items():
        path = _checked_file(root / name, suffixes=frozenset({".parquet"}))
        metadata = manifest["artifacts"][name]
        if path.stat().st_size != int(metadata["bytes"]) or _sha256_file(path) != metadata["sha256"]:
            raise HergCandidateAdjudicationError(f"artifact hash mismatch: {name}")
        parquet = pq.ParquetFile(path)
        if parquet.metadata.num_rows != int(metadata["rows"]) or parquet.schema_arrow != schema:
            raise HergCandidateAdjudicationError(f"artifact row count or schema mismatch: {name}")
        if metadata["arrow_schema_sha256"] != _schema_sha256(schema):
            raise HergCandidateAdjudicationError(f"artifact schema digest mismatch: {name}")
    for name in (PACKET_OUTPUT, DECISION_CONTRACT_OUTPUT):
        path = _checked_file(root / name)
        metadata = manifest["artifacts"][name]
        if path.stat().st_size != int(metadata["bytes"]) or _sha256_file(path) != metadata["sha256"]:
            raise HergCandidateAdjudicationError(f"artifact hash mismatch: {name}")
    contract = json.loads((root / DECISION_CONTRACT_OUTPUT).read_text(encoding="utf-8"))
    if contract.get("status") != "blank_template_no_human_decisions_recorded":
        raise HergCandidateAdjudicationError("decision contract does not preserve blank-template status")

    candidates = pq.read_table(root / CANDIDATE_OUTPUT).to_pylist()
    conflicts = pq.read_table(root / CONFLICT_OUTPUT).to_pylist()
    lineage = pq.read_table(root / LINEAGE_OUTPUT).to_pylist()
    upstream_candidates = pq.read_table(roles["v1_4_evaluation_candidates"]).to_pylist()
    upstream_conflicts = pq.read_table(roles["v1_4_pic50_conflict_queue"]).to_pylist()
    if {row["candidate_id"] for row in candidates} != {row["candidate_id"] for row in upstream_candidates}:
        raise HergCandidateAdjudicationError("candidate evidence does not exactly cover v1.4 candidates")
    if {row["review_id"] for row in conflicts} != {row["review_id"] for row in upstream_conflicts}:
        raise HergCandidateAdjudicationError("conflict evidence does not exactly cover v1.4 queue")
    if any(
        row["automated_evidence_status"] != AUTOMATED_STATUS
        or row["human_adjudication_status"] != HUMAN_STATUS
        for row in [*candidates, *conflicts, *lineage]
    ):
        raise HergCandidateAdjudicationError("an automated row was marked adjudicated")
    if any("confirmed_wild_type" in str(row["target_evidence_class"]) for row in candidates):
        raise HergCandidateAdjudicationError("automated target evidence improperly claims confirmed WT")
    if any(row["master_wild_type_evidence_scope"] != "wild_type_or_unspecified" for row in candidates):
        raise HergCandidateAdjudicationError("candidate WT scope was changed from the frozen input")
    if any(
        row["observation_count_bound_to_local_source"] != row["exact_replicate_count"] for row in conflicts
    ):
        raise HergCandidateAdjudicationError("a conflict observation is not bound to its local source")
    if len({row["lineage_group_id"] for row in lineage}) != len(lineage):
        raise HergCandidateAdjudicationError("duplicate lineage group IDs")
    if any(
        row["observation_count"] < 2
        or row["automated_evidence_strength"] not in _STRENGTH_ORDER
        or not _json_list(row["automated_evidence_limitations_json"])
        for row in lineage
    ):
        raise HergCandidateAdjudicationError("invalid lineage evidence row")

    with (root / PACKET_OUTPUT).open("r", encoding="utf-8", newline="") as handle:
        packet = list(csv.DictReader(handle))
    if list(packet[0]) != _PACKET_FIELDS if packet else True:
        raise HergCandidateAdjudicationError("human packet columns mismatch")
    if [int(row["review_rank"]) for row in packet] != list(range(1, len(packet) + 1)):
        raise HergCandidateAdjudicationError("human packet ranks are not deterministic")
    if any(
        row["human_adjudication_status"] != HUMAN_STATUS
        or any(
            row[field] for field in ("human_decision", "human_reviewer", "human_reviewed_at", "human_notes")
        )
        for row in packet
    ):
        raise HergCandidateAdjudicationError("human packet contains a completed decision")
    expected_items = {row["candidate_id"] for row in candidates} | {row["review_id"] for row in conflicts}
    if {row["item_id"] for row in packet} != expected_items or len(packet) != len(expected_items):
        raise HergCandidateAdjudicationError("human packet does not cover each review item exactly once")
    if len(packet) != int(manifest["artifacts"][PACKET_OUTPUT]["rows"]):
        raise HergCandidateAdjudicationError("human packet row count mismatch")

    counts = manifest["counts"]
    if (
        len(candidates) != counts["candidate_rows"]
        or len(conflicts) != counts["conflict_rows"]
        or len(lineage) != counts["lineage_groups"]
        or len(packet) != counts["human_packet_rows"]
        or counts["human_decisions_completed"] != 0
        or counts["gold_rows_promoted"] != 0
    ):
        raise HergCandidateAdjudicationError("manifest counts or nonpromotion invariants mismatch")
    if report_root is not None:
        report_path = _checked_file(report_root.resolve() / REPORT_NAME)
        report_metadata = manifest["report_artifact"]
        if (
            report_path.stat().st_size != int(report_metadata["bytes"])
            or _sha256_file(report_path) != report_metadata["sha256"]
        ):
            raise HergCandidateAdjudicationError("report artifact hash mismatch")
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-root", type=Path)
    parser.add_argument("--pre-hpc-root", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report-root", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    if args.validate_only:
        validate_herg_candidate_adjudication(args.output_root, report_root=args.report_root)
    else:
        if args.master_root is None or args.pre_hpc_root is None or args.report_root is None:
            raise HergCandidateAdjudicationError(
                "--master-root, --pre-hpc-root, and --report-root are required when building"
            )
        build_herg_candidate_adjudication(
            master_root=args.master_root,
            pre_hpc_root=args.pre_hpc_root,
            output_root=args.output_root,
            report_root=args.report_root,
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
