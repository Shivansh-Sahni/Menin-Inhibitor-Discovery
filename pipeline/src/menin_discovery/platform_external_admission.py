"""CPU-only readiness analysis for normalized external public evidence.

This module is deliberately an *admission audit*, not an admission stage.  It
verifies the frozen normalized bundle, measures schema/field completeness,
classifies identity-linkage candidates against the frozen canonical ChEMBL
build, and inventories duplicate, mirror, conflict, clinical, regulatory, and
rights blockers.  It cannot create canonical observations or model labels.

The analysis is deterministic and batch-oriented.  Only candidate external
identifiers and their canonical matches are retained in memory; Parquet inputs
are streamed in fixed batches.  Ambiguity, missing rights evidence, predictions,
and absence of a reported outcome all fail closed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .platform_external_normalization import (
    canonical_json_bytes,
    document_with_sha256,
    sha256_file,
    verify_document_sha256,
    verify_external_normalized_output,
)

SCHEMA_VERSION = "platform-external-admission-readiness/1.0"
ANALYZER_VERSION = "platform_external_admission/1.1"
NORMALIZED_MANIFEST = "external_public_normalized_manifest.json"
CANONICAL_MANIFEST = "build_manifest.json"
DEFAULT_BATCH_ROWS = 8_192
ExternalMeasurement = tuple[str, str, int, str]

RIGHTS_UNRESOLVED = "unresolved_missing_machine_readable_evidence"
RIGHTS_CLEARED = "cleared_for_intended_use_and_redistribution"


class ExternalAdmissionError(RuntimeError):
    """Raised when an input or admission-safety contract fails closed."""


EXPECTED_NORMALIZED_SCHEMAS: dict[str, tuple[str, ...]] = {
    "bindingdb/affinity_observations.parquet": (
        "source_id",
        "source_member",
        "source_row_number_one_based",
        "source_field_map_sha256",
        "reactant_set_id",
        "measurement_key",
        "endpoint_type",
        "endpoint_source_column",
        "raw_value",
        "relation",
        "parsed_numeric_text",
        "value_nm",
        "unit",
        "parse_status",
        "endpoint_pooling_key",
        "candidate_evidence_admitted",
        "model_label_admitted",
    ),
    "bindingdb/article_rows.parquet": (
        "source_id",
        "source_member",
        "source_row_number_one_based",
        "source_column_count",
        "source_field_map_sha256",
        "reactant_set_id",
        "reactant_set_occurrence",
        "reactant_set_total_occurrences",
        "repeated_reactant_set_id",
        "curation_data_source",
        "row_disposition",
        "disposition_reason",
        "candidate_evidence_admitted",
        "model_label_admitted",
        "ligand_smiles",
        "ligand_inchi",
        "ligand_inchi_key",
        "ligand_inchi_key_syntax_valid",
        "structure_representation_status",
        "bindingdb_monomer_id",
        "ligand_name",
        "target_name",
        "target_source_organism",
        "target_chain_count_raw",
        "target_accessions_json",
        "target_sequences_json",
        "article_doi",
        "bindingdb_entry_doi",
        "pmid",
        "publication_date_raw",
        "source_record_json",
    ),
    "clinicaltrials/cohort_membership.parquet": (
        "source_id",
        "cohort_kind",
        "cohort_snapshot_key",
        "source_row_number_one_based",
        "nct_id",
        "page_index",
        "study_index_within_page",
        "study_sha256",
        "heuristic_term_matches_json",
        "has_posted_outcome_measures_module",
        "has_posted_adverse_events_module",
        "evidence_semantics",
        "false_positive_or_context_ambiguity_retained",
        "canonical_observation_admitted",
        "model_label_admitted",
        "source_record_json",
    ),
    "regulatory/archive_inventory.parquet": (
        "source_id",
        "inventory_kind",
        "item_id",
        "row_or_member_count",
        "compressed_or_source_bytes",
        "uncompressed_bytes",
        "sha256",
        "parse_or_verification_status",
        "source_anomaly_count",
        "evidence_semantics",
        "canonical_observation_admitted",
        "model_label_admitted",
        "source_metadata_json",
    ),
    "source_inventory.parquet": (
        "source_id",
        "release_id",
        "snapshot_status",
        "declared_manifest_sha256",
        "physical_manifest_sha256",
        "bundle_entries_sha256",
        "bundle_entry_count",
        "bundle_total_bytes",
        "normalization_scope",
        "canonical_observation_admitted",
        "model_label_admitted",
        "limitations_json",
    ),
    "uniprot/accession_resolution.parquet": (
        "source_id",
        "source_row_number_one_based",
        "requested_accession",
        "resolution_state",
        "normalization_disposition",
        "returned_primary_accession",
        "returned_primary_accessions_json",
        "entry_type",
        "sequence_status",
        "sequence_sha256",
        "replacement_state",
        "silent_identity_replacement_performed",
        "model_label_admitted",
        "source_record_json",
    ),
    "uniprot/returned_entries.parquet": (
        "source_id",
        "source_page_path",
        "source_result_index_zero_based",
        "raw_entry_sha256",
        "returned_primary_accession",
        "entry_type",
        "entry_name",
        "protein_name",
        "taxonomy_id",
        "organism_scientific_name",
        "sequence_status",
        "sequence",
        "sequence_length",
        "sequence_md5",
        "sequence_sha256",
        "sequence_version",
        "entry_version",
        "model_label_admitted",
    ),
    "uniprot/source_membership.parquet": (
        "source_id",
        "source_row_number_one_based",
        "upstream_source_id",
        "upstream_source_file",
        "upstream_source_row_index_zero_based",
        "upstream_source_target_id",
        "upstream_source_component_id",
        "source_accession_value",
        "normalized_identifier",
        "source_admission_state",
        "resolution_state",
        "normalization_disposition",
        "returned_primary_accession",
        "sequence_sha256",
        "silent_identity_replacement_performed",
        "model_label_admitted",
        "source_record_json",
    ),
}

CANONICAL_REQUIRED_COLUMNS: dict[str, tuple[str, ...]] = {
    "molecules": ("standard_inchi_key", "molecule_id"),
    "proteins": ("uniprot_accession", "protein_id"),
    "observations": (
        "molecule_id",
        "protein_id",
        "endpoint",
        "canonical_unit",
        "canonical_value",
        "relation",
    ),
}

SOURCE_POLICIES: dict[str, dict[str, Any]] = {
    "bindingdb_curated_202608": {
        "role": "candidate_external_affinity_evidence",
        "rights_status": RIGHTS_UNRESOLVED,
        "required_rights_evidence": [
            "license_or_terms_version",
            "intended_use_permission",
            "redistribution_permission",
            "required_attribution",
            "legal_or_data_steward_approval",
        ],
    },
    "clinicaltrials_gov_v2": {
        "role": "registry_inventory_not_outcome_labels",
        "rights_status": RIGHTS_UNRESOLVED,
        "required_rights_evidence": [
            "current_terms_or_public_domain_basis",
            "required_disclaimer_and_attribution",
            "participant_level_data_exclusion_confirmation",
            "legal_or_data_steward_approval",
        ],
    },
    "dailymed_spl_v2_human_rx": {
        "role": "regulatory_archive_inventory_not_safety_labels",
        "rights_status": RIGHTS_UNRESOLVED,
        "required_rights_evidence": [
            "current_terms_or_public_domain_basis",
            "label_copyright_and_attribution_review",
            "redistribution_permission",
            "legal_or_data_steward_approval",
        ],
    },
    "drugs_at_fda_bulk": {
        "role": "regulatory_relational_inventory_not_outcome_labels",
        "rights_status": RIGHTS_UNRESOLVED,
        "required_rights_evidence": [
            "current_terms_or_public_domain_basis",
            "redistribution_permission",
            "required_attribution",
            "legal_or_data_steward_approval",
        ],
    },
    "uniprotkb_targeted_2026_02": {
        "role": "protein_identity_and_sequence_enrichment",
        "rights_status": RIGHTS_UNRESOLVED,
        "required_rights_evidence": [
            "license_version",
            "intended_use_permission",
            "redistribution_and_attribution_requirements",
            "legal_or_data_steward_approval",
        ],
    },
}


def _safe_existing_directory(path: str | os.PathLike[str], *, context: str) -> Path:
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_dir():
        raise ExternalAdmissionError(f"Missing or symlinked {context} directory: {candidate}")
    return candidate.resolve()


def _safe_relative(value: str, *, context: str) -> PurePosixPath:
    candidate = PurePosixPath(value)
    if not value or candidate.is_absolute() or ".." in candidate.parts:
        raise ExternalAdmissionError(f"Unsafe {context} path: {value}")
    return candidate


def validate_exact_schema(actual: Iterable[str], expected: Iterable[str], *, context: str) -> None:
    """Reject missing, added, or reordered fields instead of guessing their semantics."""

    actual_tuple = tuple(actual)
    expected_tuple = tuple(expected)
    if actual_tuple != expected_tuple:
        missing = sorted(set(expected_tuple) - set(actual_tuple))
        unexpected = sorted(set(actual_tuple) - set(expected_tuple))
        raise ExternalAdmissionError(
            f"Schema drift for {context}: missing={missing}, unexpected={unexpected}, "
            f"order_changed={not missing and not unexpected}"
        )


def validate_required_columns(actual: Iterable[str], required: Iterable[str], *, context: str) -> None:
    missing = sorted(set(required) - set(actual))
    if missing:
        raise ExternalAdmissionError(f"Canonical schema drift for {context}: missing={missing}")


def _json_object(value: Any, *, context: str) -> dict[str, Any]:
    try:
        parsed = json.loads(str(value))
    except (TypeError, json.JSONDecodeError) as error:
        raise ExternalAdmissionError(f"Invalid JSON object for {context}") from error
    if not isinstance(parsed, dict):
        raise ExternalAdmissionError(f"Expected JSON object for {context}")
    return parsed


def _json_list(value: Any, *, context: str) -> list[Any]:
    try:
        parsed = json.loads(str(value))
    except (TypeError, json.JSONDecodeError) as error:
        raise ExternalAdmissionError(f"Invalid JSON list for {context}") from error
    if not isinstance(parsed, list):
        raise ExternalAdmissionError(f"Expected JSON list for {context}")
    return parsed


def classify_admission_candidate(
    record: Mapping[str, Any],
    *,
    rights_status: str,
    molecule_match_count: int = 0,
    target_match_count: int = 0,
    source_kind: str = "measured_evidence",
) -> dict[str, Any]:
    """Return a fail-closed state; this function never returns an admitted label.

    The helper is intentionally strict enough to exercise the central safety
    rules independently in adversarial tests.
    """

    if record.get("model_label_admitted") is True or record.get("canonical_observation_admitted") is True:
        raise ExternalAdmissionError("Input attempts to pre-admit a label or canonical observation")

    normalized_keys = {str(key).casefold() for key in record}
    prediction_keys = {
        key
        for key in normalized_keys
        if "prediction" in key or key.startswith("predicted_") or key.endswith("_prediction")
    }
    if prediction_keys:
        state = "quarantine_prediction_is_not_observed_evidence"
    elif source_kind == "clinical_inventory" and (
        record.get("has_posted_outcome_measures_module") is False
        or record.get("has_posted_adverse_events_module") is False
    ):
        state = "review_absence_of_posted_module_is_not_a_negative_outcome"
    elif rights_status != RIGHTS_CLEARED:
        state = "blocked_rights_or_access_not_cleared"
    elif molecule_match_count > 1 or target_match_count > 1:
        state = "quarantine_ambiguous_identity_link"
    elif molecule_match_count != 1 or target_match_count != 1:
        state = "review_identity_link_incomplete"
    else:
        state = "review_scientific_assay_unit_conflict_and_provenance_required"

    return {
        "admission_state": state,
        "canonical_observation_admitted": False,
        "model_label_admitted": False,
        "negative_label_inferred_from_absence": False,
        "prediction_treated_as_observation": False,
    }


def _field_completeness(path: Path, expected_fields: Sequence[str]) -> dict[str, Any]:
    parquet = pq.ParquetFile(path)
    validate_exact_schema(parquet.schema_arrow.names, expected_fields, context=path.as_posix())
    counts: dict[str, dict[str, int]] = {
        name: {"non_null": 0, "null": 0, "blank_string": 0} for name in expected_fields
    }
    rows = 0
    for batch in parquet.iter_batches(batch_size=DEFAULT_BATCH_ROWS):
        rows += len(batch)
        for index, field in enumerate(batch.schema):
            array = batch.column(index)
            summary = counts[field.name]
            summary["null"] += array.null_count
            summary["non_null"] += len(array) - array.null_count
            if pa.types.is_string(array.type) or pa.types.is_large_string(array.type):
                filled = pc.fill_null(array, "")
                blanks = pc.sum(pc.cast(pc.equal(pc.utf8_trim_whitespace(filled), ""), pa.int64()))
                summary["blank_string"] += int(blanks.as_py() or 0)
    if rows != parquet.metadata.num_rows:
        raise ExternalAdmissionError(f"Batch/footer row mismatch: {path}")
    for field_counts in counts.values():
        if field_counts["non_null"] + field_counts["null"] != rows:
            raise ExternalAdmissionError(f"Field completeness failed to reconcile: {path}")
    return {"rows": rows, "fields": counts}


def _load_identified_json(path: Path, *, require_internal_sha: bool) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ExternalAdmissionError(f"Missing or symlinked JSON input: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ExternalAdmissionError(f"Unreadable JSON input: {path}") from error
    if not isinstance(value, dict):
        raise ExternalAdmissionError(f"JSON input is not an object: {path}")
    if require_internal_sha and not verify_document_sha256(value):
        raise ExternalAdmissionError(f"JSON internal SHA-256 failed: {path}")
    return value


def _canonical_component_map(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    entries = manifest.get("component_inventory")
    if not isinstance(entries, list) or not all(isinstance(item, dict) for item in entries):
        raise ExternalAdmissionError("Canonical component inventory is missing")
    result: dict[str, dict[str, Any]] = {}
    for item in entries:
        relative = _safe_relative(str(item.get("path", "")), context="canonical component")
        key = relative.as_posix()
        if key in result:
            raise ExternalAdmissionError(f"Duplicate canonical component path: {key}")
        result[key] = dict(item)
    return result


def _verify_bound_component(path: Path, root: Path, entry: Mapping[str, Any]) -> None:
    relative = path.relative_to(root).as_posix()
    if path.is_symlink() or not path.is_file():
        raise ExternalAdmissionError(f"Missing or symlinked canonical component: {relative}")
    if path.stat().st_size != int(entry.get("size_bytes", -1)):
        raise ExternalAdmissionError(f"Canonical component byte drift: {relative}")
    if sha256_file(path) != entry.get("sha256"):
        raise ExternalAdmissionError(f"Canonical component SHA-256 drift: {relative}")
    if path.suffix == ".parquet" and pq.ParquetFile(path).metadata.num_rows != int(entry.get("rows", -1)):
        raise ExternalAdmissionError(f"Canonical component row drift: {relative}")


def _canonical_paths(
    canonical_root: Path, component_map: Mapping[str, Mapping[str, Any]], dataset: str
) -> list[Path]:
    paths = sorted(canonical_root.glob(f"{dataset}/part-*.parquet"))
    if not paths:
        raise ExternalAdmissionError(f"Canonical dataset is empty: {dataset}")
    for path in paths:
        relative = path.relative_to(canonical_root).as_posix()
        entry = component_map.get(relative)
        if entry is None:
            raise ExternalAdmissionError(f"Canonical component is not manifest-bound: {relative}")
        validate_required_columns(
            pq.ParquetFile(path).schema_arrow.names,
            CANONICAL_REQUIRED_COLUMNS[dataset],
            context=relative,
        )
    return paths


def _normalized_artifact_analysis(normalized_root: Path) -> dict[str, Any]:
    artifacts: dict[str, Any] = {}
    for relative, expected in sorted(EXPECTED_NORMALIZED_SCHEMAS.items()):
        path = normalized_root / relative
        if path.is_symlink() or not path.is_file():
            raise ExternalAdmissionError(f"Missing normalized artifact: {relative}")
        artifacts[relative] = _field_completeness(path, expected)
    return artifacts


def _collect_bindingdb_identity_inputs(normalized_root: Path) -> tuple[set[str], set[str]]:
    inchikeys: set[str] = set()
    accessions: set[str] = set()
    path = normalized_root / "bindingdb/article_rows.parquet"
    columns = [
        "ligand_inchi_key",
        "ligand_inchi_key_syntax_valid",
        "target_accessions_json",
        "model_label_admitted",
        "candidate_evidence_admitted",
    ]
    for batch in pq.ParquetFile(path).iter_batches(columns=columns, batch_size=DEFAULT_BATCH_ROWS):
        for row in batch.to_pylist():
            if row["model_label_admitted"]:
                raise ExternalAdmissionError("BindingDB row contains an admitted model label")
            if row["ligand_inchi_key_syntax_valid"] and row["ligand_inchi_key"]:
                inchikeys.add(str(row["ligand_inchi_key"]))
            for item in _json_list(row["target_accessions_json"], context="BindingDB target accessions"):
                if not isinstance(item, dict):
                    raise ExternalAdmissionError("BindingDB target accession entry is not an object")
                accession = str(item.get("raw_primary_id", "")).strip()
                if accession:
                    accessions.add(accession)
    return inchikeys, accessions


def _canonical_identity_maps(
    canonical_root: Path,
    component_map: Mapping[str, Mapping[str, Any]],
    inchikey_candidates: set[str],
    accession_candidates: set[str],
) -> tuple[dict[str, set[str]], dict[str, set[str]], list[dict[str, Any]]]:
    molecule_matches: dict[str, set[str]] = defaultdict(set)
    protein_matches: dict[str, set[str]] = defaultdict(set)
    verified: list[dict[str, Any]] = []
    for dataset, key_column, id_column, candidates, destination in (
        ("molecules", "standard_inchi_key", "molecule_id", inchikey_candidates, molecule_matches),
        ("proteins", "uniprot_accession", "protein_id", accession_candidates, protein_matches),
    ):
        for path in _canonical_paths(canonical_root, component_map, dataset):
            relative = path.relative_to(canonical_root).as_posix()
            entry = component_map[relative]
            _verify_bound_component(path, canonical_root, entry)
            verified.append(
                {
                    "path": relative,
                    "rows": int(entry["rows"]),
                    "bytes": int(entry["size_bytes"]),
                    "sha256": str(entry["sha256"]),
                }
            )
            for batch in pq.ParquetFile(path).iter_batches(
                columns=[key_column, id_column], batch_size=DEFAULT_BATCH_ROWS
            ):
                for row in batch.to_pylist():
                    key = row[key_column]
                    identifier = row[id_column]
                    if key in candidates and identifier:
                        destination[str(key)].add(str(identifier))
    return molecule_matches, protein_matches, verified


def _bindingdb_analysis(
    normalized_root: Path,
    molecule_matches: Mapping[str, set[str]],
    protein_matches: Mapping[str, set[str]],
) -> tuple[dict[str, Any], dict[int, tuple[str, str] | None]]:
    identity_counts: Counter[str] = Counter()
    admission_counts: Counter[str] = Counter()
    origin_counts: Counter[str] = Counter()
    doi_counts: Counter[str] = Counter()
    rows_by_source_number: dict[int, tuple[str, str] | None] = {}
    path = normalized_root / "bindingdb/article_rows.parquet"
    columns = [
        "source_row_number_one_based",
        "row_disposition",
        "curation_data_source",
        "candidate_evidence_admitted",
        "model_label_admitted",
        "ligand_inchi_key",
        "ligand_inchi_key_syntax_valid",
        "target_accessions_json",
        "target_sequences_json",
        "article_doi",
        "pmid",
    ]
    for batch in pq.ParquetFile(path).iter_batches(columns=columns, batch_size=DEFAULT_BATCH_ROWS):
        for row in batch.to_pylist():
            if row["model_label_admitted"]:
                raise ExternalAdmissionError("BindingDB article row admitted a model label")
            source_row = int(row["source_row_number_one_based"])
            origin_counts[str(row["row_disposition"])] += 1
            doi_counts["article_doi_present"] += int(bool(str(row["article_doi"] or "").strip()))
            doi_counts["pmid_present"] += int(bool(str(row["pmid"] or "").strip()))

            inchikey = str(row["ligand_inchi_key"] or "")
            molecule_ids = (
                molecule_matches.get(inchikey, set()) if row["ligand_inchi_key_syntax_valid"] else set()
            )
            accessions = {
                str(item.get("raw_primary_id", "")).strip()
                for item in _json_list(row["target_accessions_json"], context="BindingDB target accessions")
                if isinstance(item, dict) and str(item.get("raw_primary_id", "")).strip()
            }
            target_ids = set().union(*(protein_matches.get(value, set()) for value in accessions))
            sequence_entries = _json_list(row["target_sequences_json"], context="BindingDB target sequences")

            if not row["ligand_inchi_key_syntax_valid"]:
                molecule_state = "molecule_no_valid_inchikey"
            elif not molecule_ids:
                molecule_state = "molecule_exact_inchikey_unmatched"
            elif len(molecule_ids) > 1:
                molecule_state = "molecule_exact_inchikey_ambiguous"
            else:
                molecule_state = "molecule_exact_inchikey_unique_match"

            if len(accessions) > 1:
                target_state = "target_multicomponent_or_multiaccession_review"
            elif not accessions and sequence_entries:
                target_state = "target_sequence_only_review"
            elif not accessions:
                target_state = "target_name_only_or_missing_review"
            elif not target_ids:
                target_state = "target_accession_unmatched"
            elif len(target_ids) > 1:
                target_state = "target_accession_ambiguous"
            else:
                target_state = "target_accession_unique_match"

            identity_state = f"{molecule_state}__{target_state}"
            identity_counts[identity_state] += 1
            disposition = str(row["row_disposition"])
            if disposition == "excluded_chembl_cross_source_mirror":
                admission_state = "excluded_declared_chembl_mirror"
            elif disposition == "quarantine_taylor_origin_rights_pending":
                admission_state = "quarantine_origin_specific_rights_pending"
            else:
                admission_state = classify_admission_candidate(
                    row,
                    rights_status=RIGHTS_UNRESOLVED,
                    molecule_match_count=len(molecule_ids),
                    target_match_count=len(target_ids),
                )["admission_state"]
            admission_counts[str(admission_state)] += 1

            if len(molecule_ids) == 1 and len(accessions) == 1 and len(target_ids) == 1:
                rows_by_source_number[source_row] = (next(iter(molecule_ids)), next(iter(target_ids)))
            else:
                rows_by_source_number[source_row] = None

    return (
        {
            "article_rows": sum(origin_counts.values()),
            "row_disposition_counts": dict(sorted(origin_counts.items())),
            "identity_linkage_state_counts": dict(sorted(identity_counts.items())),
            "admission_state_counts": dict(sorted(admission_counts.items())),
            "provenance_identifier_completeness": dict(sorted(doi_counts.items())),
            "exact_unique_dual_link_rows": sum(value is not None for value in rows_by_source_number.values()),
            "canonical_observations_admitted": 0,
            "model_labels_admitted": 0,
        },
        rows_by_source_number,
    )


def _number_key(value: float) -> str:
    return format(float(value), ".12g")


def _bindingdb_measurement_analysis(
    normalized_root: Path, linked_rows: Mapping[int, tuple[str, str] | None]
) -> tuple[dict[str, Any], dict[tuple[str, str, str], list[ExternalMeasurement]]]:
    exact_signatures: Counter[tuple[str, str, str, str, str]] = Counter()
    values_by_identity_endpoint: dict[tuple[str, str, str], set[tuple[str, str]]] = defaultdict(set)
    external_by_identity_endpoint: dict[tuple[str, str, str], list[ExternalMeasurement]] = defaultdict(list)
    endpoint_counts: Counter[str] = Counter()
    relation_counts: Counter[str] = Counter()
    linked_measurements = 0
    path = normalized_root / "bindingdb/affinity_observations.parquet"
    columns = [
        "source_row_number_one_based",
        "measurement_key",
        "endpoint_type",
        "relation",
        "value_nm",
        "parse_status",
        "candidate_evidence_admitted",
        "model_label_admitted",
    ]
    for batch in pq.ParquetFile(path).iter_batches(columns=columns, batch_size=DEFAULT_BATCH_ROWS):
        for row in batch.to_pylist():
            if row["model_label_admitted"] or not row["candidate_evidence_admitted"]:
                raise ExternalAdmissionError("BindingDB affinity zero-label/candidate contract failed")
            endpoint = str(row["endpoint_type"])
            relation = str(row["relation"] or "")
            endpoint_counts[endpoint] += 1
            relation_counts[relation] += 1
            link = linked_rows.get(int(row["source_row_number_one_based"]))
            if link is None or row["parse_status"] != "parsed_candidate" or row["value_nm"] is None:
                continue
            molecule_id, protein_id = link
            linked_measurements += 1
            value_key = _number_key(float(row["value_nm"]))
            signature = (molecule_id, protein_id, endpoint, relation, value_key)
            exact_signatures[signature] += 1
            identity_endpoint = (molecule_id, protein_id, endpoint)
            values_by_identity_endpoint[identity_endpoint].add((relation, value_key))
            external_by_identity_endpoint[identity_endpoint].append(
                (
                    relation,
                    value_key,
                    int(row["source_row_number_one_based"]),
                    str(row["measurement_key"]),
                )
            )

    local_conflict_groups = sum(1 for values in values_by_identity_endpoint.values() if len(values) > 1)
    local_conflict_measurements = sum(
        len(external_by_identity_endpoint[key])
        for key, values in values_by_identity_endpoint.items()
        if len(values) > 1
    )
    return (
        {
            "affinity_rows": sum(endpoint_counts.values()),
            "endpoint_counts": dict(sorted(endpoint_counts.items())),
            "relation_counts": dict(sorted(relation_counts.items())),
            "exact_unique_dual_link_measurements": linked_measurements,
            "source_local_exact_signature_groups_with_duplicates": sum(
                count > 1 for count in exact_signatures.values()
            ),
            "source_local_exact_signature_duplicate_excess_rows": sum(
                count - 1 for count in exact_signatures.values() if count > 1
            ),
            "source_local_value_disagreement_candidate_groups": local_conflict_groups,
            "source_local_value_disagreement_candidate_measurements": local_conflict_measurements,
            "conflict_interpretation": (
                "candidate only: distinct values may reflect legitimate assay, construct, condition, "
                "or document differences and are never collapsed automatically"
            ),
            "canonical_observations_admitted": 0,
            "model_labels_admitted": 0,
        },
        external_by_identity_endpoint,
    )


def _cross_source_overlap_analysis(
    canonical_root: Path,
    component_map: Mapping[str, Mapping[str, Any]],
    external: Mapping[tuple[str, str, str], list[ExternalMeasurement]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    exact_external_measurements: set[tuple[str, str, str, str, str, int, str]] = set()
    disagreement_external_measurements: set[tuple[str, str, str, str, str, int, str]] = set()
    canonical_rows_on_linked_keys = 0
    exact_canonical_rows = 0
    disagreement_canonical_rows = 0
    verified: list[dict[str, Any]] = []
    for path in _canonical_paths(canonical_root, component_map, "observations"):
        relative = path.relative_to(canonical_root).as_posix()
        entry = component_map[relative]
        _verify_bound_component(path, canonical_root, entry)
        verified.append(
            {
                "path": relative,
                "rows": int(entry["rows"]),
                "bytes": int(entry["size_bytes"]),
                "sha256": str(entry["sha256"]),
            }
        )
        columns = list(CANONICAL_REQUIRED_COLUMNS["observations"])
        for batch in pq.ParquetFile(path).iter_batches(columns=columns, batch_size=DEFAULT_BATCH_ROWS):
            for row in batch.to_pylist():
                key = (
                    str(row["molecule_id"] or ""),
                    str(row["protein_id"] or ""),
                    str(row["endpoint"] or ""),
                )
                candidates = external.get(key)
                if not candidates or row["canonical_unit"] != "nM" or row["canonical_value"] is None:
                    continue
                canonical_rows_on_linked_keys += 1
                canonical_pair = (str(row["relation"] or ""), _number_key(float(row["canonical_value"])))
                row_exact = False
                row_disagreement = False
                for relation, value_key, source_row, measurement_key in candidates:
                    external_id = (*key, relation, value_key, source_row, measurement_key)
                    if canonical_pair == (relation, value_key):
                        exact_external_measurements.add(external_id)
                        row_exact = True
                    else:
                        disagreement_external_measurements.add(external_id)
                        row_disagreement = True
                exact_canonical_rows += int(row_exact)
                disagreement_canonical_rows += int(row_disagreement)
    return (
        {
            "canonical_rows_on_exact_identity_endpoint_candidate_keys": canonical_rows_on_linked_keys,
            "exact_value_relation_match_canonical_rows": exact_canonical_rows,
            "exact_value_relation_match_external_measurements": len(exact_external_measurements),
            "value_or_relation_disagreement_candidate_canonical_rows": disagreement_canonical_rows,
            "value_or_relation_disagreement_candidate_external_measurements": len(
                disagreement_external_measurements
            ),
            "interpretation": (
                "candidate mirror/conflict accounting only; exact identity, endpoint, relation, and "
                "numeric agreement does not establish duplicated experimental provenance, while "
                "disagreement does not establish error without assay/context review"
            ),
            "cross_source_rows_automatically_deduplicated": 0,
            "canonical_observations_admitted": 0,
            "model_labels_admitted": 0,
        },
        verified,
    )


def _clinical_analysis(normalized_root: Path) -> dict[str, Any]:
    cohorts: Counter[str] = Counter()
    module_states: Counter[str] = Counter()
    nct_memberships: dict[str, set[str]] = defaultdict(set)
    path = normalized_root / "clinicaltrials/cohort_membership.parquet"
    columns = [
        "cohort_kind",
        "nct_id",
        "has_posted_outcome_measures_module",
        "has_posted_adverse_events_module",
        "false_positive_or_context_ambiguity_retained",
        "canonical_observation_admitted",
        "model_label_admitted",
    ]
    for batch in pq.ParquetFile(path).iter_batches(columns=columns, batch_size=DEFAULT_BATCH_ROWS):
        for row in batch.to_pylist():
            if row["canonical_observation_admitted"] or row["model_label_admitted"]:
                raise ExternalAdmissionError("Clinical inventory attempts to admit a label")
            cohort = str(row["cohort_kind"])
            cohorts[cohort] += 1
            nct_memberships[str(row["nct_id"])].add(cohort)
            if cohort == "heuristic_cardiac_safety_inventory":
                key = (
                    f"outcomes={row['has_posted_outcome_measures_module']};"
                    f"adverse_events={row['has_posted_adverse_events_module']}"
                )
                module_states[key] += 1
                classify_admission_candidate(
                    row, rights_status=RIGHTS_UNRESOLVED, source_kind="clinical_inventory"
                )
    overlap = sum(len(memberships) > 1 for memberships in nct_memberships.values())
    return {
        "cohort_membership_rows": sum(cohorts.values()),
        "unique_nct_ids": len(nct_memberships),
        "cohort_counts": dict(sorted(cohorts.items())),
        "broad_and_heuristic_duplicate_memberships": overlap,
        "heuristic_posted_module_state_counts": dict(sorted(module_states.items())),
        "studies_with_both_posted_modules": module_states["outcomes=True;adverse_events=True"],
        "normalized_outcome_measure_records_extracted": 0,
        "normalized_adverse_event_records_extracted": 0,
        "normalized_intervention_to_molecule_links": 0,
        "qt_qtc_endpoint_values_extracted": 0,
        "pk_endpoint_values_extracted": 0,
        "outcome_extraction_readiness": "not_ready_inventory_projection_only",
        "required_next_gate": (
            "re-extract exact version-bound raw study modules; normalize arms, interventions, time "
            "frames, units, denominators, adverse-event groups, and molecule identities; then human QC"
        ),
        "absence_semantics": "missing/unposted module is unknown, never a negative outcome",
        "heuristic_semantics": "text-search cohort membership is not a cardiac-safety label",
        "canonical_observations_admitted": 0,
        "model_labels_admitted": 0,
    }


def _regulatory_analysis(normalized_root: Path) -> dict[str, Any]:
    source_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    anomaly_counts: Counter[str] = Counter()
    path = normalized_root / "regulatory/archive_inventory.parquet"
    columns = [
        "source_id",
        "inventory_kind",
        "row_or_member_count",
        "parse_or_verification_status",
        "source_anomaly_count",
        "canonical_observation_admitted",
        "model_label_admitted",
    ]
    for batch in pq.ParquetFile(path).iter_batches(columns=columns, batch_size=DEFAULT_BATCH_ROWS):
        for row in batch.to_pylist():
            if row["canonical_observation_admitted"] or row["model_label_admitted"]:
                raise ExternalAdmissionError("Regulatory inventory attempts to admit a label")
            source = str(row["source_id"])
            source_counts[source] += 1
            status_counts[f"{source}:{row['parse_or_verification_status']}"] += 1
            anomaly_counts[source] += int(row["source_anomaly_count"] or 0)
    return {
        "inventory_rows_by_source": dict(sorted(source_counts.items())),
        "parse_or_verification_status_counts": dict(sorted(status_counts.items())),
        "declared_source_anomalies_by_source": dict(sorted(anomaly_counts.items())),
        "record_level_products_or_labels_normalized": 0,
        "application_product_action_links_normalized": 0,
        "spl_section_texts_normalized": 0,
        "active_ingredient_to_molecule_links": 0,
        "qt_qtc_or_pk_outcomes_extracted": 0,
        "outcome_extraction_readiness": "not_ready_archive_and_table_inventory_only",
        "required_next_gate": (
            "repair/quarantine malformed FDA rows, normalize relational keys and active ingredients, "
            "parse versioned SPL sections, create exact molecule links, and complete human regulatory QC"
        ),
        "approval_semantics": "approval/marketing state is not efficacy, safety, PK, or activity",
        "canonical_observations_admitted": 0,
        "model_labels_admitted": 0,
    }


def _uniprot_analysis(normalized_root: Path) -> dict[str, Any]:
    dispositions: Counter[str] = Counter()
    sequence_states: Counter[str] = Counter()
    path = normalized_root / "uniprot/accession_resolution.parquet"
    for batch in pq.ParquetFile(path).iter_batches(
        columns=[
            "normalization_disposition",
            "sequence_status",
            "silent_identity_replacement_performed",
            "model_label_admitted",
        ],
        batch_size=DEFAULT_BATCH_ROWS,
    ):
        for row in batch.to_pylist():
            if row["silent_identity_replacement_performed"] or row["model_label_admitted"]:
                raise ExternalAdmissionError("UniProt identity/label prohibition failed")
            dispositions[str(row["normalization_disposition"])] += 1
            sequence_states[str(row["sequence_status"])] += 1
    return {
        "resolution_rows": sum(dispositions.values()),
        "normalization_disposition_counts": dict(sorted(dispositions.items())),
        "sequence_status_counts": dict(sorted(sequence_states.items())),
        "identity_enrichment_readiness": "candidate_only_rights_and_identity_review_required",
        "silent_identity_replacements": 0,
        "canonical_observations_admitted": 0,
        "model_labels_admitted": 0,
    }


def _rights_analysis(source_inventory: Mapping[str, Any]) -> dict[str, Any]:
    observed_sources = set(source_inventory.get("source_to_output_reconciliation", {}))
    if observed_sources != set(SOURCE_POLICIES):
        raise ExternalAdmissionError(
            f"Source-policy drift: missing={sorted(set(SOURCE_POLICIES) - observed_sources)}, "
            f"unexpected={sorted(observed_sources - set(SOURCE_POLICIES))}"
        )
    by_source: dict[str, Any] = {}
    for source_id, policy in sorted(SOURCE_POLICIES.items()):
        by_source[source_id] = {
            **policy,
            "canonical_admission_allowed": False,
            "model_training_use_allowed": False,
            "redistribution_allowed": False,
            "gate_reason": "normalized/acquisition manifests contain no machine-readable rights clearance",
        }
    return {
        "sources": by_source,
        "sources_cleared_for_canonical_admission": 0,
        "sources_cleared_for_model_training": 0,
        "sources_cleared_for_redistribution": 0,
        "default_policy": "fail_closed_until_source-specific_evidence_and_human_approval",
    }


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    cwd = Path.cwd().resolve()
    try:
        return resolved.relative_to(cwd).as_posix()
    except ValueError:
        return resolved.as_posix()


def _atomic_json(path: Path, body: Mapping[str, Any]) -> dict[str, Any]:
    document = document_with_sha256(body)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(document, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return document


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _methods_markdown(report: Mapping[str, Any]) -> str:
    binding = report["bindingdb"]
    clinical = report["clinicaltrials"]
    overlap = report["cross_source_overlap_candidates"]
    return f"""# External evidence canonical-admission readiness

## Result

This CPU-only audit is mechanically complete, but **no external source is ready for
canonical admission or model training**. It created zero canonical observations and
zero model labels. It did not train a model.

## What was measured

- Reverified the exact frozen normalized bundle, including its internal manifest,
  file inventory, schemas, row counts, hashes, zero-label flags, and source semantics.
- Streamed every normalized Parquet field and counted non-null, null, and blank-string
  values under fixed schema contracts.
- Matched BindingDB ligand InChIKeys and single target accessions exactly to the frozen
  ChEMBL molecule/protein identities. {binding["exact_unique_dual_link_rows"]:,} article
  rows and {binding["measurements"]["exact_unique_dual_link_measurements"]:,} affinity
  measurements have unique exact dual-link candidates. These are review candidates,
  not admissions.
- Compared linked BindingDB measurements with canonical ChEMBL observations. There are
  {overlap["exact_value_relation_match_external_measurements"]:,} external measurement
  candidates with at least one exact identity/endpoint/relation/value match. This is a
  possible mirror signal, not proof of shared experimental provenance.
- Audited ClinicalTrials.gov inventory coverage: {clinical["unique_nct_ids"]:,} unique
  studies, including {clinical["studies_with_both_posted_modules"]:,} heuristic-cohort
  studies marked as having both posted results modules. The normalized data contain no
  outcome rows, arms, denominators, intervention-to-molecule links, QT/QTc values, PK
  values, or adverse-event counts.
- Audited Drugs@FDA and DailyMed as archive/table inventories only. They contain no
  normalized record-level product/molecule/outcome evidence.

## Conservative decision rules

1. A prediction is never an observed label.
2. Missing or unposted evidence is unknown, never a negative label.
3. One exact molecule and one exact single-protein accession match are only identity
   candidates; multicomponent, missing, and multi-match cases remain review/quarantine.
4. Exact numeric agreement is only a duplicate/mirror candidate; disagreement is only a
   conflict candidate until assay, construct, condition, document, and provenance review.
5. Every source is rights-blocked because the frozen manifests contain no machine-readable
   source-specific clearance for intended use, training, and redistribution.
6. No endpoint pooling, automated deduplication, identity replacement, negative-label
   inference, canonical admission, or model-label creation is performed.

## What must happen before admission

- Obtain and record versioned terms/license, intended-use, redistribution, attribution,
  and human data-steward/legal approval per source.
- For BindingDB, perform molecule standardization, target/construct/assay review, document
  provenance reconciliation, unit/relation review, and explicit mirror/conflict decisions.
- Re-extract version-bound ClinicalTrials.gov results modules and normalize arms,
  interventions, time frames, units, denominators, and adverse-event groups; then link
  interventions to molecules with human review.
- Repair or quarantine FDA relational anomalies, normalize applications/products/actions
  and active ingredients, parse versioned DailyMed SPL sections, and create reviewed
  molecule links.
- Freeze an admission policy and independently reproduce it before rebuilding canonical
  data. Until then, external canonical rows and labels must remain zero.

## Limits

This is a readiness and candidate-accounting analysis, not scientific validation. Exact
identity matching does not resolve salts, mixtures, stereochemistry policies, constructs,
assay context, clinical causality, or regulatory interpretation. Clinical cohort selection
is heuristic and retains false positives. Source-local and cross-source value differences
can be legitimate experimental heterogeneity. The analysis used no HPC and no network data.
"""


def run_external_admission_analysis(
    normalized_root: str | os.PathLike[str],
    canonical_root: str | os.PathLike[str],
    output_root: str | os.PathLike[str],
) -> dict[str, Any]:
    """Run the full deterministic analysis and publish three bound artifacts."""

    normalized = _safe_existing_directory(normalized_root, context="normalized input")
    canonical = _safe_existing_directory(canonical_root, context="canonical input")
    output = Path(output_root)
    if output.exists():
        raise ExternalAdmissionError(f"Output exists and will not be replaced: {output}")
    output_resolved = output.resolve()
    if output_resolved == normalized or normalized in output_resolved.parents:
        raise ExternalAdmissionError("Output must not be inside normalized input")
    if output_resolved == canonical or canonical in output_resolved.parents:
        raise ExternalAdmissionError("Output must not be inside canonical input")

    normalized_manifest_path = normalized / NORMALIZED_MANIFEST
    normalized_manifest = _load_identified_json(normalized_manifest_path, require_internal_sha=True)
    normalized_verification = verify_external_normalized_output(normalized)
    field_completeness = _normalized_artifact_analysis(normalized)

    canonical_manifest_path = canonical / CANONICAL_MANIFEST
    canonical_manifest = _load_identified_json(canonical_manifest_path, require_internal_sha=False)
    if canonical_manifest.get("schema_version") != "platform-evidence-1.0.0":
        raise ExternalAdmissionError("Unexpected canonical schema version")
    component_map = _canonical_component_map(canonical_manifest)

    inchikeys, accessions = _collect_bindingdb_identity_inputs(normalized)
    molecule_matches, protein_matches, identity_components = _canonical_identity_maps(
        canonical, component_map, inchikeys, accessions
    )
    bindingdb, linked_rows = _bindingdb_analysis(normalized, molecule_matches, protein_matches)
    measurements, external_signatures = _bindingdb_measurement_analysis(normalized, linked_rows)
    bindingdb["measurements"] = measurements
    cross_source, observation_components = _cross_source_overlap_analysis(
        canonical, component_map, external_signatures
    )

    used_components = sorted(identity_components + observation_components, key=lambda item: item["path"])
    report_body: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "analyzer_version": ANALYZER_VERSION,
        "analyzer_code_sha256": sha256_file(Path(__file__).resolve()),
        "analysis_id": "external_admission_analysis_full_chembl37",
        "decision": "not_ready_for_external_canonical_admission_or_model_training",
        "mechanical_analysis_status": "passed",
        "normalized_input": {
            "root": _portable_path(normalized),
            "manifest_path": _portable_path(normalized_manifest_path),
            "manifest_internal_sha256": normalized_manifest["manifest_sha256"],
            "manifest_physical_sha256": sha256_file(normalized_manifest_path),
            "manifest_physical_bytes": normalized_manifest_path.stat().st_size,
            "verification": normalized_verification,
        },
        "canonical_input": {
            "root": _portable_path(canonical),
            "manifest_path": _portable_path(canonical_manifest_path),
            "manifest_physical_sha256": sha256_file(canonical_manifest_path),
            "manifest_physical_bytes": canonical_manifest_path.stat().st_size,
            "snapshot_id": canonical_manifest.get("snapshot_id"),
            "verified_used_components": used_components,
            "verified_used_component_count": len(used_components),
            "verification_scope": "every molecule, protein, and observation Parquet component used",
        },
        "field_and_schema_completeness": field_completeness,
        "identity_candidate_universes": {
            "bindingdb_valid_inchikey_candidates": len(inchikeys),
            "bindingdb_target_accession_candidates": len(accessions),
            "inchikey_candidates_with_canonical_match": sum(
                bool(value) for value in molecule_matches.values()
            ),
            "inchikey_candidates_with_ambiguous_canonical_match": sum(
                len(value) > 1 for value in molecule_matches.values()
            ),
            "accession_candidates_with_canonical_match": sum(
                bool(value) for value in protein_matches.values()
            ),
            "accession_candidates_with_ambiguous_canonical_match": sum(
                len(value) > 1 for value in protein_matches.values()
            ),
        },
        "bindingdb": bindingdb,
        "cross_source_overlap_candidates": cross_source,
        "clinicaltrials": _clinical_analysis(normalized),
        "regulatory": _regulatory_analysis(normalized),
        "uniprot": _uniprot_analysis(normalized),
        "rights_and_access": _rights_analysis(normalized_manifest),
        "global_prohibitions": {
            "prediction_treated_as_observed_label": False,
            "absence_treated_as_negative_label": False,
            "ambiguous_identity_silently_resolved": False,
            "endpoint_pooling_performed": False,
            "cross_source_rows_automatically_deduplicated": 0,
            "external_canonical_observations_admitted": 0,
            "external_model_labels_admitted": 0,
            "substantive_model_training_performed": False,
            "substantive_model_training_authorized": False,
        },
        "blocking_gates": [
            "source-specific machine-readable rights, intended-use, attribution, and redistribution clearance",
            "BindingDB molecule standardization plus target/construct/assay/unit/document review",
            "explicit BindingDB mirror, duplicate, and disagreement adjudication",
            "ClinicalTrials.gov version-bound results re-extraction and arm/intervention/time-frame/denominator normalization",
            "human-reviewed intervention-to-molecule identity resolution and clinical endpoint semantics",
            "Drugs@FDA relational anomaly repair and record-level application/product/action normalization",
            "DailyMed SPL section parsing, active-ingredient resolution, and regulatory interpretation",
            "frozen admission policy, independent reproduction, and canonical rebuild QC",
        ],
        "determinism_contract": (
            "no wall-clock fields; fixed ordered schemas and batch sizes; sorted keys/counters/components; "
            "canonical JSON hashing; fail-closed existing-output policy"
        ),
    }
    output.mkdir(parents=True, exist_ok=False)
    report_path = output / "external_admission_readiness_report.json"
    report = _atomic_json(report_path, report_body)
    methods_path = output / "methods_and_limitations.md"
    _atomic_text(methods_path, _methods_markdown(report))

    entries: list[dict[str, Any]] = []
    for path, role in ((report_path, "machine_readable_report"), (methods_path, "methods_and_limitations")):
        entries.append(
            {
                "path": path.name,
                "artifact_role": role,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    entries.sort(key=lambda item: item["path"])
    manifest_body = {
        "schema_version": SCHEMA_VERSION,
        "analyzer_version": ANALYZER_VERSION,
        "analysis_id": report["analysis_id"],
        "decision": report["decision"],
        "mechanical_analysis_status": "passed",
        "report_internal_sha256": report["manifest_sha256"],
        "input_bindings": {
            "analyzer_code_sha256": report["analyzer_code_sha256"],
            "normalized_manifest_physical_sha256": report["normalized_input"]["manifest_physical_sha256"],
            "canonical_manifest_physical_sha256": report["canonical_input"]["manifest_physical_sha256"],
            "canonical_used_component_set_sha256": hashlib.sha256(
                canonical_json_bytes(used_components)
            ).hexdigest(),
        },
        "output_inventory": {
            "entries": entries,
            "entries_sha256": hashlib.sha256(canonical_json_bytes(entries)).hexdigest(),
            "entry_count": len(entries),
            "total_bytes": sum(int(item["bytes"]) for item in entries),
            "excluded_paths": ["external_admission_analysis_manifest.json"],
            "exclusion_reason": "self-referential manifest bound by manifest_sha256",
        },
        "external_canonical_observations_admitted": 0,
        "external_model_labels_admitted": 0,
        "substantive_model_training_performed": False,
        "substantive_model_training_authorized": False,
    }
    manifest = _atomic_json(output / "external_admission_analysis_manifest.json", manifest_body)
    return manifest


def _validate_analysis_bindings(manifest: Mapping[str, Any], report: Mapping[str, Any]) -> None:
    bindings = manifest.get("input_bindings")
    canonical_input = report.get("canonical_input")
    normalized_input = report.get("normalized_input")
    if (
        not isinstance(bindings, Mapping)
        or not isinstance(canonical_input, Mapping)
        or not isinstance(normalized_input, Mapping)
    ):
        raise ExternalAdmissionError("Analysis input bindings are missing")
    used_components = canonical_input.get("verified_used_components")
    if not isinstance(used_components, list):
        raise ExternalAdmissionError("Analysis canonical component binding is missing")
    expected = {
        "analyzer_code_sha256": sha256_file(Path(__file__).resolve()),
        "normalized_manifest_physical_sha256": normalized_input.get("manifest_physical_sha256"),
        "canonical_manifest_physical_sha256": canonical_input.get("manifest_physical_sha256"),
        "canonical_used_component_set_sha256": hashlib.sha256(
            canonical_json_bytes(used_components)
        ).hexdigest(),
    }
    if (
        manifest.get("analyzer_version") != ANALYZER_VERSION
        or report.get("analyzer_version") != ANALYZER_VERSION
        or report.get("analyzer_code_sha256") != expected["analyzer_code_sha256"]
        or any(bindings.get(key) != value for key, value in expected.items())
    ):
        raise ExternalAdmissionError("Analysis code or input binding changed")


def verify_external_admission_analysis(output_root: str | os.PathLike[str]) -> dict[str, Any]:
    """Verify a published analysis without re-reading the source datasets."""

    root = _safe_existing_directory(output_root, context="analysis output")
    manifest_path = root / "external_admission_analysis_manifest.json"
    manifest = _load_identified_json(manifest_path, require_internal_sha=True)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ExternalAdmissionError("Unexpected analysis manifest schema")
    if any(
        (
            int(manifest.get("external_canonical_observations_admitted", -1)) != 0,
            int(manifest.get("external_model_labels_admitted", -1)) != 0,
            manifest.get("substantive_model_training_performed") is not False,
            manifest.get("substantive_model_training_authorized") is not False,
        )
    ):
        raise ExternalAdmissionError("Analysis zero-label/training boundary failed")
    inventory = manifest.get("output_inventory")
    if not isinstance(inventory, dict):
        raise ExternalAdmissionError("Analysis output inventory is missing")
    entries = inventory.get("entries")
    if not isinstance(entries, list) or not all(isinstance(item, dict) for item in entries):
        raise ExternalAdmissionError("Analysis output entries are invalid")
    if hashlib.sha256(canonical_json_bytes(entries)).hexdigest() != inventory.get("entries_sha256"):
        raise ExternalAdmissionError("Analysis output entry digest failed")
    if int(inventory.get("entry_count", -1)) != len(entries):
        raise ExternalAdmissionError("Analysis output entry count failed")
    if inventory.get("excluded_paths") != [manifest_path.name]:
        raise ExternalAdmissionError("Analysis manifest exclusion failed")
    expected: set[str] = set()
    total_bytes = 0
    for item in entries:
        relative = _safe_relative(str(item.get("path", "")), context="analysis artifact")
        expected.add(relative.as_posix())
        path = root / Path(*relative.parts)
        if path.is_symlink() or not path.is_file():
            raise ExternalAdmissionError(f"Missing analysis artifact: {relative}")
        if path.stat().st_size != int(item.get("bytes", -1)) or sha256_file(path) != item.get("sha256"):
            raise ExternalAdmissionError(f"Analysis artifact identity drift: {relative}")
        total_bytes += path.stat().st_size
    observed = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != manifest_path
    }
    if observed != expected:
        raise ExternalAdmissionError(
            f"Analysis artifact membership drift: missing={sorted(expected - observed)}, "
            f"unexpected={sorted(observed - expected)}"
        )
    if total_bytes != int(inventory.get("total_bytes", -1)):
        raise ExternalAdmissionError("Analysis aggregate bytes failed")
    report = _load_identified_json(
        root / "external_admission_readiness_report.json", require_internal_sha=True
    )
    if report.get("manifest_sha256") != manifest.get("report_internal_sha256"):
        raise ExternalAdmissionError("Analysis report/manifest binding failed")
    _validate_analysis_bindings(manifest, report)
    prohibitions = report.get("global_prohibitions")
    if not isinstance(prohibitions, dict) or any(
        (
            prohibitions.get("external_canonical_observations_admitted") != 0,
            prohibitions.get("external_model_labels_admitted") != 0,
            prohibitions.get("substantive_model_training_performed") is not False,
            prohibitions.get("substantive_model_training_authorized") is not False,
        )
    ):
        raise ExternalAdmissionError("Analysis report zero-label/training boundary failed")
    return {
        "status": "passed",
        "manifest_internal_sha256": manifest["manifest_sha256"],
        "manifest_physical_sha256": sha256_file(manifest_path),
        "manifest_physical_bytes": manifest_path.stat().st_size,
        "artifact_count": len(entries),
        "artifact_bytes": total_bytes,
        "decision": manifest["decision"],
        "zero_label_and_training_boundary": "passed",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--normalized-root",
        default="research/data/platform/interim/external_public_normalized",
    )
    parser.add_argument(
        "--canonical-root",
        default="research/data/platform/canonical/full_chembl37",
    )
    parser.add_argument(
        "--output-root",
        default="research/reports/platform/external_admission_analysis",
    )
    parser.add_argument("--verify-existing", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.verify_existing:
        result = verify_external_admission_analysis(args.output_root)
    else:
        result = run_external_admission_analysis(args.normalized_root, args.canonical_root, args.output_root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
