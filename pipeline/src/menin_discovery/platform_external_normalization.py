"""Deterministic normalization of frozen external public evidence.

This stage is intentionally narrower than canonical observation construction.
It preserves source identities and makes independently curated BindingDB rows,
UniProt sequences, registry cohort membership, and regulatory archive metadata
reviewable.  It never creates a model label and never trains a model.

Every input byte is reconciled to its acquisition manifest before parsing.  A
build is written below a transaction-specific staging directory and promoted
only after its output inventory, Parquet schemas, and source-to-output counts
pass.  Existing output and abandoned transaction markers fail closed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import sys
import tempfile
import zipfile
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path, PurePosixPath
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

SCHEMA_VERSION = "platform-external-normalization/1.0"
PARSER_VERSION = "platform_external_normalization/1.0"
SOURCE_SCHEMA_VERSION = "platform-external-acquisition/1.0"
DEFAULT_CHUNK_BYTES = 16 * 1024 * 1024
DEFAULT_BATCH_ROWS = 2_048

BINDINGDB_SOURCE_ID = "bindingdb_curated_202608"
UNIPROT_SOURCE_ID = "uniprotkb_targeted_2026_02"
CLINICALTRIALS_SOURCE_ID = "clinicaltrials_gov_v2"
DRUGSFDA_SOURCE_ID = "drugs_at_fda_bulk"
DAILYMED_SOURCE_ID = "dailymed_spl_v2_human_rx"

SOURCE_MANIFESTS: tuple[tuple[str, str], ...] = (
    (BINDINGDB_SOURCE_ID, "bindingdb_curated_202608_manifest.json"),
    (UNIPROT_SOURCE_ID, "uniprotkb_targeted_2026_02_manifest.json"),
    (CLINICALTRIALS_SOURCE_ID, "clinicaltrials_gov_v2_manifest.json"),
    (DRUGSFDA_SOURCE_ID, "drugs_at_fda_bulk_manifest.json"),
    (DAILYMED_SOURCE_ID, "dailymed_spl_v2_human_rx_manifest.json"),
)

BINDINGDB_ARTICLES_ARCHIVE = "BindingDB_BindingDB_Articles_202608_tsv.zip"
BINDINGDB_ARTICLES_MEMBER = "BindingDB_BindingDB_Articles.tsv"
BINDINGDB_ALLOWED_ORIGIN = "Curated from the literature by BindingDB"
BINDINGDB_EXCLUDED_ORIGIN = "ChEMBL"
BINDINGDB_QUARANTINED_ORIGIN = "Taylor Research Group, UCSD"
AFFINITY_ENDPOINT_COLUMNS: tuple[tuple[str, str], ...] = (
    ("Kd", "Kd (nM)"),
    ("Ki", "Ki (nM)"),
    ("IC50", "IC50 (nM)"),
    ("EC50", "EC50 (nM)"),
)

AFFINITY_RE = re.compile(
    r"^\s*(?P<relation><=|>=|<|>|=|~|≈)?\s*"
    r"(?P<value>\+?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?)\s*$"
)
INCHIKEY_RE = re.compile(r"^[A-Z]{14}-[A-Z]{10}-[A-Z]$")
AMINO_ACID_RE = re.compile(r"^[A-Z]+$")


class NormalizationError(RuntimeError):
    """Raised when immutable input or output contracts do not reconcile."""


@dataclass(frozen=True)
class InputBinding:
    """Verified acquisition-manifest identity and physical bundle binding."""

    source_id: str
    root: Path
    manifest_path: Path
    manifest: dict[str, Any]
    declared_manifest_sha256: str
    physical_manifest_sha256: str
    physical_manifest_bytes: int
    bundle_entries_sha256: str
    bundle_entry_count: int
    bundle_total_bytes: int

    def as_record(self, raw_root: Path) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "manifest_path": self.manifest_path.relative_to(raw_root).as_posix(),
            "declared_manifest_sha256": self.declared_manifest_sha256,
            "physical_manifest_sha256": self.physical_manifest_sha256,
            "physical_manifest_bytes": self.physical_manifest_bytes,
            "bundle_entries_sha256": self.bundle_entries_sha256,
            "bundle_entry_count": self.bundle_entry_count,
            "bundle_total_bytes": self.bundle_total_bytes,
            "release_id": self.manifest.get("release_id"),
            "snapshot_status": self.manifest.get("snapshot_status"),
        }


@dataclass(frozen=True)
class ParquetArtifact:
    """Physical and logical identity of one generated Parquet artifact."""

    path: str
    rows: int
    bytes: int
    sha256: str
    arrow_schema_sha256: str

    def as_record(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "artifact_role": "normalized_parquet",
            "rows": self.rows,
            "bytes": self.bytes,
            "sha256": self.sha256,
            "arrow_schema_sha256": self.arrow_schema_sha256,
        }


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize JSON deterministically."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def canonical_json_text(value: Any) -> str:
    return canonical_json_bytes(value).decode("utf-8")


def sha256_file(path: str | os.PathLike[str], *, chunk_bytes: int = DEFAULT_CHUNK_BYTES) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def document_with_sha256(document: Mapping[str, Any]) -> dict[str, Any]:
    body = {str(key): value for key, value in document.items() if key != "manifest_sha256"}
    body["manifest_sha256"] = hashlib.sha256(canonical_json_bytes(body)).hexdigest()
    return body


def verify_document_sha256(document: Mapping[str, Any]) -> bool:
    expected = document.get("manifest_sha256")
    if not isinstance(expected, str) or not expected:
        return False
    body = {str(key): value for key, value in document.items() if key != "manifest_sha256"}
    return hashlib.sha256(canonical_json_bytes(body)).hexdigest() == expected


def _atomic_write_json(path: Path, document: Mapping[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    identified = document_with_sha256(document)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(identified, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return identified


def _safe_relative_path(value: Any, *, context: str) -> PurePosixPath:
    relative = PurePosixPath(str(value))
    if not str(relative) or relative.is_absolute() or ".." in relative.parts:
        raise NormalizationError(f"Unsafe {context} path: {relative}")
    return relative


def load_and_verify_input(source_root: Path, manifest_name: str) -> InputBinding:
    """Bind a source manifest to its exact recursive physical bundle in one hash pass."""

    if source_root.is_symlink() or not source_root.is_dir():
        raise NormalizationError(f"Missing or symlinked source root: {source_root}")
    root = source_root.resolve()
    manifest_path = root / manifest_name
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise NormalizationError(f"Missing or symlinked source manifest: {manifest_path}")
    try:
        manifest_value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise NormalizationError(f"Unreadable source manifest: {manifest_path}") from error
    if not isinstance(manifest_value, dict):
        raise NormalizationError(f"Source manifest is not an object: {manifest_path}")
    manifest: dict[str, Any] = manifest_value
    if manifest.get("schema_version") != SOURCE_SCHEMA_VERSION:
        raise NormalizationError(f"Unexpected acquisition schema: {manifest_path}")
    if manifest.get("source_id") != source_root.name:
        raise NormalizationError(f"Source identity/root mismatch: {source_root}")
    if not verify_document_sha256(manifest):
        raise NormalizationError(f"Internal manifest digest failed: {manifest_path}")

    bundle = manifest.get("bundle_inventory")
    if not isinstance(bundle, dict):
        raise NormalizationError(f"Missing bundle inventory: {manifest_path}")
    entries = bundle.get("entries")
    if not isinstance(entries, list) or not all(isinstance(item, dict) for item in entries):
        raise NormalizationError(f"Invalid bundle entries: {manifest_path}")
    calculated_entries_sha = hashlib.sha256(canonical_json_bytes(entries)).hexdigest()
    if calculated_entries_sha != bundle.get("entries_sha256"):
        raise NormalizationError(f"Bundle entry digest failed: {manifest_path}")
    if int(bundle.get("entry_count", -1)) != len(entries):
        raise NormalizationError(f"Bundle entry count failed: {manifest_path}")

    exclusions = bundle.get("excluded_paths")
    if exclusions != [manifest_name]:
        raise NormalizationError(f"Bundle must exclude only its own manifest: {manifest_path}")
    declared: dict[str, dict[str, Any]] = {}
    for item in entries:
        relative = _safe_relative_path(item.get("path"), context="bundle")
        key = relative.as_posix()
        if key in declared:
            raise NormalizationError(f"Duplicate bundle path: {key}")
        declared[key] = item

    observed: set[str] = set()
    for candidate in root.rglob("*"):
        if candidate.is_symlink():
            raise NormalizationError(f"Symlink prohibited in source bundle: {candidate}")
        if candidate.is_file():
            observed_relative = candidate.relative_to(root).as_posix()
            if observed_relative != manifest_name:
                observed.add(observed_relative)
    if observed != set(declared):
        raise NormalizationError(
            "Source bundle membership changed: "
            f"missing={sorted(set(declared) - observed)}, "
            f"unexpected={sorted(observed - set(declared))}"
        )

    verified_bytes = 0
    for declared_relative, item in sorted(declared.items()):
        path = root / Path(*PurePosixPath(declared_relative).parts)
        expected_bytes = int(item.get("bytes", -1))
        if path.stat().st_size != expected_bytes:
            raise NormalizationError(f"Source byte count changed: {path}")
        if sha256_file(path) != item.get("sha256"):
            raise NormalizationError(f"Source SHA-256 changed: {path}")
        if "line_count" in item:
            with path.open("rb") as handle:
                lines = sum(1 for _line in handle)
            if lines != int(item["line_count"]):
                raise NormalizationError(f"Source line count changed: {path}")
        verified_bytes += expected_bytes
    if verified_bytes != int(bundle.get("total_bytes", -1)):
        raise NormalizationError(f"Bundle byte total failed: {manifest_path}")
    if len(entries) != int(manifest.get("exact_physical_file_count", -1)):
        raise NormalizationError(f"Manifest physical file total failed: {manifest_path}")
    if verified_bytes != int(manifest.get("exact_physical_bytes", -1)):
        raise NormalizationError(f"Manifest physical byte total failed: {manifest_path}")

    files = manifest.get("files")
    if not isinstance(files, list):
        raise NormalizationError(f"Manifest source files are missing: {manifest_path}")
    source_paths: set[str] = set()
    for item in files:
        if not isinstance(item, dict):
            raise NormalizationError(f"Invalid source file record: {manifest_path}")
        source_relative = _safe_relative_path(
            item.get("local_path", item.get("path")), context="source artifact"
        ).as_posix()
        if source_relative in source_paths:
            raise NormalizationError(f"Duplicate source file record: {source_relative}")
        source_paths.add(source_relative)
        bundled = declared.get(source_relative)
        if bundled is None:
            raise NormalizationError(f"Source artifact absent from bundle inventory: {source_relative}")
        source_bytes = item.get("acquired_bytes", item.get("bytes"))
        if source_bytes is None or int(source_bytes) != int(bundled["bytes"]):
            raise NormalizationError(f"Source/bundle byte mismatch: {source_relative}")
        if item.get("acquired_sha256", item.get("sha256")) != bundled["sha256"]:
            raise NormalizationError(f"Source/bundle digest mismatch: {source_relative}")

    return InputBinding(
        source_id=str(manifest["source_id"]),
        root=root,
        manifest_path=manifest_path,
        manifest=manifest,
        declared_manifest_sha256=str(manifest["manifest_sha256"]),
        physical_manifest_sha256=sha256_file(manifest_path),
        physical_manifest_bytes=manifest_path.stat().st_size,
        bundle_entries_sha256=calculated_entries_sha,
        bundle_entry_count=len(entries),
        bundle_total_bytes=verified_bytes,
    )


def arrow_schema_sha256(schema: pa.Schema) -> str:
    return hashlib.sha256(schema.serialize().to_pybytes()).hexdigest()


def write_parquet_rows(
    path: Path,
    schema: pa.Schema,
    rows: Iterable[Mapping[str, Any]],
    *,
    batch_rows: int = DEFAULT_BATCH_ROWS,
) -> ParquetArtifact:
    """Write deterministic, explicitly typed Parquet from a bounded iterator."""

    path.parent.mkdir(parents=True, exist_ok=True)
    row_count = 0
    batch: list[Mapping[str, Any]] = []
    writer = pq.ParquetWriter(
        path,
        schema,
        compression="zstd",
        compression_level=9,
        use_dictionary=True,
        write_statistics=True,
        version="2.6",
        data_page_version="2.0",
    )
    try:
        for row in rows:
            batch.append(row)
            if len(batch) >= batch_rows:
                table = pa.Table.from_pylist(batch, schema=schema)
                writer.write_table(table, row_group_size=batch_rows)
                row_count += len(batch)
                batch.clear()
        if batch:
            table = pa.Table.from_pylist(batch, schema=schema)
            writer.write_table(table, row_group_size=batch_rows)
            row_count += len(batch)
        elif row_count == 0:
            writer.write_table(pa.Table.from_pylist([], schema=schema))
    finally:
        writer.close()
    physical_schema = pq.ParquetFile(path).schema_arrow
    if not physical_schema.equals(schema, check_metadata=True):
        raise NormalizationError(f"Parquet physical schema changed during write: {path}")
    if pq.ParquetFile(path).metadata.num_rows != row_count:
        raise NormalizationError(f"Parquet footer row count failed: {path}")
    return ParquetArtifact(
        path=path.as_posix(),
        rows=row_count,
        bytes=path.stat().st_size,
        sha256=sha256_file(path),
        arrow_schema_sha256=arrow_schema_sha256(schema),
    )


BINDINGDB_ROWS_SCHEMA = pa.schema(
    [
        pa.field("source_id", pa.string(), nullable=False),
        pa.field("source_member", pa.string(), nullable=False),
        pa.field("source_row_number_one_based", pa.int64(), nullable=False),
        pa.field("source_column_count", pa.int32(), nullable=False),
        pa.field("source_field_map_sha256", pa.string(), nullable=False),
        pa.field("reactant_set_id", pa.string()),
        pa.field("reactant_set_occurrence", pa.int32(), nullable=False),
        pa.field("reactant_set_total_occurrences", pa.int32(), nullable=False),
        pa.field("repeated_reactant_set_id", pa.bool_(), nullable=False),
        pa.field("curation_data_source", pa.string(), nullable=False),
        pa.field("row_disposition", pa.string(), nullable=False),
        pa.field("disposition_reason", pa.string(), nullable=False),
        pa.field("candidate_evidence_admitted", pa.bool_(), nullable=False),
        pa.field("model_label_admitted", pa.bool_(), nullable=False),
        pa.field("ligand_smiles", pa.string()),
        pa.field("ligand_inchi", pa.string()),
        pa.field("ligand_inchi_key", pa.string()),
        pa.field("ligand_inchi_key_syntax_valid", pa.bool_()),
        pa.field("structure_representation_status", pa.string(), nullable=False),
        pa.field("bindingdb_monomer_id", pa.string()),
        pa.field("ligand_name", pa.string()),
        pa.field("target_name", pa.string()),
        pa.field("target_source_organism", pa.string()),
        pa.field("target_chain_count_raw", pa.string()),
        pa.field("target_accessions_json", pa.string(), nullable=False),
        pa.field("target_sequences_json", pa.string(), nullable=False),
        pa.field("article_doi", pa.string()),
        pa.field("bindingdb_entry_doi", pa.string()),
        pa.field("pmid", pa.string()),
        pa.field("publication_date_raw", pa.string()),
        pa.field("source_record_json", pa.large_string(), nullable=False),
    ]
)

AFFINITY_SCHEMA = pa.schema(
    [
        pa.field("source_id", pa.string(), nullable=False),
        pa.field("source_member", pa.string(), nullable=False),
        pa.field("source_row_number_one_based", pa.int64(), nullable=False),
        pa.field("source_field_map_sha256", pa.string(), nullable=False),
        pa.field("reactant_set_id", pa.string()),
        pa.field("measurement_key", pa.string(), nullable=False),
        pa.field("endpoint_type", pa.string(), nullable=False),
        pa.field("endpoint_source_column", pa.string(), nullable=False),
        pa.field("raw_value", pa.string(), nullable=False),
        pa.field("relation", pa.string()),
        pa.field("parsed_numeric_text", pa.string()),
        pa.field("value_nm", pa.float64()),
        pa.field("unit", pa.string(), nullable=False),
        pa.field("parse_status", pa.string(), nullable=False),
        pa.field("endpoint_pooling_key", pa.string(), nullable=False),
        pa.field("candidate_evidence_admitted", pa.bool_(), nullable=False),
        pa.field("model_label_admitted", pa.bool_(), nullable=False),
    ]
)

UNIPROT_ENTRY_SCHEMA = pa.schema(
    [
        pa.field("source_id", pa.string(), nullable=False),
        pa.field("source_page_path", pa.string(), nullable=False),
        pa.field("source_result_index_zero_based", pa.int32(), nullable=False),
        pa.field("raw_entry_sha256", pa.string(), nullable=False),
        pa.field("returned_primary_accession", pa.string(), nullable=False),
        pa.field("entry_type", pa.string(), nullable=False),
        pa.field("entry_name", pa.string()),
        pa.field("protein_name", pa.string()),
        pa.field("taxonomy_id", pa.int64()),
        pa.field("organism_scientific_name", pa.string()),
        pa.field("sequence_status", pa.string(), nullable=False),
        pa.field("sequence", pa.large_string()),
        pa.field("sequence_length", pa.int64()),
        pa.field("sequence_md5", pa.string()),
        pa.field("sequence_sha256", pa.string()),
        pa.field("sequence_version", pa.int64()),
        pa.field("entry_version", pa.int64()),
        pa.field("model_label_admitted", pa.bool_(), nullable=False),
    ]
)

UNIPROT_RESOLUTION_SCHEMA = pa.schema(
    [
        pa.field("source_id", pa.string(), nullable=False),
        pa.field("source_row_number_one_based", pa.int64(), nullable=False),
        pa.field("requested_accession", pa.string(), nullable=False),
        pa.field("resolution_state", pa.string(), nullable=False),
        pa.field("normalization_disposition", pa.string(), nullable=False),
        pa.field("returned_primary_accession", pa.string()),
        pa.field("returned_primary_accessions_json", pa.string(), nullable=False),
        pa.field("entry_type", pa.string()),
        pa.field("sequence_status", pa.string(), nullable=False),
        pa.field("sequence_sha256", pa.string()),
        pa.field("replacement_state", pa.string()),
        pa.field("silent_identity_replacement_performed", pa.bool_(), nullable=False),
        pa.field("model_label_admitted", pa.bool_(), nullable=False),
        pa.field("source_record_json", pa.large_string(), nullable=False),
    ]
)

UNIPROT_MEMBERSHIP_SCHEMA = pa.schema(
    [
        pa.field("source_id", pa.string(), nullable=False),
        pa.field("source_row_number_one_based", pa.int64(), nullable=False),
        pa.field("upstream_source_id", pa.string(), nullable=False),
        pa.field("upstream_source_file", pa.string(), nullable=False),
        pa.field("upstream_source_row_index_zero_based", pa.int64()),
        pa.field("upstream_source_target_id", pa.string()),
        pa.field("upstream_source_component_id", pa.int64()),
        pa.field("source_accession_value", pa.string()),
        pa.field("normalized_identifier", pa.string()),
        pa.field("source_admission_state", pa.string(), nullable=False),
        pa.field("resolution_state", pa.string(), nullable=False),
        pa.field("normalization_disposition", pa.string(), nullable=False),
        pa.field("returned_primary_accession", pa.string()),
        pa.field("sequence_sha256", pa.string()),
        pa.field("silent_identity_replacement_performed", pa.bool_(), nullable=False),
        pa.field("model_label_admitted", pa.bool_(), nullable=False),
        pa.field("source_record_json", pa.large_string(), nullable=False),
    ]
)

CLINICALTRIALS_MEMBERSHIP_SCHEMA = pa.schema(
    [
        pa.field("source_id", pa.string(), nullable=False),
        pa.field("cohort_kind", pa.string(), nullable=False),
        pa.field("cohort_snapshot_key", pa.string(), nullable=False),
        pa.field("source_row_number_one_based", pa.int64(), nullable=False),
        pa.field("nct_id", pa.string(), nullable=False),
        pa.field("page_index", pa.int32(), nullable=False),
        pa.field("study_index_within_page", pa.int32(), nullable=False),
        pa.field("study_sha256", pa.string(), nullable=False),
        pa.field("heuristic_term_matches_json", pa.string(), nullable=False),
        pa.field("has_posted_outcome_measures_module", pa.bool_()),
        pa.field("has_posted_adverse_events_module", pa.bool_()),
        pa.field("evidence_semantics", pa.string(), nullable=False),
        pa.field("false_positive_or_context_ambiguity_retained", pa.bool_(), nullable=False),
        pa.field("canonical_observation_admitted", pa.bool_(), nullable=False),
        pa.field("model_label_admitted", pa.bool_(), nullable=False),
        pa.field("source_record_json", pa.large_string(), nullable=False),
    ]
)

REGULATORY_INVENTORY_SCHEMA = pa.schema(
    [
        pa.field("source_id", pa.string(), nullable=False),
        pa.field("inventory_kind", pa.string(), nullable=False),
        pa.field("item_id", pa.string(), nullable=False),
        pa.field("row_or_member_count", pa.int64()),
        pa.field("compressed_or_source_bytes", pa.int64()),
        pa.field("uncompressed_bytes", pa.int64()),
        pa.field("sha256", pa.string()),
        pa.field("parse_or_verification_status", pa.string(), nullable=False),
        pa.field("source_anomaly_count", pa.int64(), nullable=False),
        pa.field("evidence_semantics", pa.string(), nullable=False),
        pa.field("canonical_observation_admitted", pa.bool_(), nullable=False),
        pa.field("model_label_admitted", pa.bool_(), nullable=False),
        pa.field("source_metadata_json", pa.large_string(), nullable=False),
    ]
)

SOURCE_INVENTORY_SCHEMA = pa.schema(
    [
        pa.field("source_id", pa.string(), nullable=False),
        pa.field("release_id", pa.string(), nullable=False),
        pa.field("snapshot_status", pa.string(), nullable=False),
        pa.field("declared_manifest_sha256", pa.string(), nullable=False),
        pa.field("physical_manifest_sha256", pa.string(), nullable=False),
        pa.field("bundle_entries_sha256", pa.string(), nullable=False),
        pa.field("bundle_entry_count", pa.int64(), nullable=False),
        pa.field("bundle_total_bytes", pa.int64(), nullable=False),
        pa.field("normalization_scope", pa.string(), nullable=False),
        pa.field("canonical_observation_admitted", pa.bool_(), nullable=False),
        pa.field("model_label_admitted", pa.bool_(), nullable=False),
        pa.field("limitations_json", pa.large_string(), nullable=False),
    ]
)


def parse_affinity_value(raw_value: str) -> tuple[str | None, str | None, float | None, str]:
    """Parse one nM endpoint conservatively without transforming endpoint identity."""

    if not raw_value.strip():
        return None, None, None, "blank"
    match = AFFINITY_RE.fullmatch(raw_value)
    if match is None:
        return None, None, None, "unparsed_raw_value_quarantine"
    relation = match.group("relation") or "="
    if relation == "≈":
        relation = "~"
    try:
        numeric = Decimal(match.group("value"))
        value = float(numeric)
    except (InvalidOperation, OverflowError, ValueError):
        return None, None, None, "unparsed_raw_value_quarantine"
    if numeric < 0 or not math.isfinite(value):
        return None, None, None, "unparsed_raw_value_quarantine"
    normalized = format(numeric.normalize(), "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    if normalized in {"", "-0"}:
        normalized = "0"
    return relation, normalized, value, "parsed_candidate"


def _nonblank(value: str) -> str | None:
    stripped = value.strip()
    return stripped if stripped else None


def _bindingdb_disposition(origin: str) -> tuple[str, str, bool]:
    if origin == BINDINGDB_ALLOWED_ORIGIN:
        return (
            "candidate_independent_bindingdb_curated",
            "independently_curated_BindingDB_origin_candidate_pending_scientific_and_rights_review",
            True,
        )
    if origin == BINDINGDB_EXCLUDED_ORIGIN:
        return (
            "excluded_chembl_cross_source_mirror",
            "ChEMBL_origin_is_not_independent_external_evidence",
            False,
        )
    if origin == BINDINGDB_QUARANTINED_ORIGIN:
        return (
            "quarantine_taylor_origin_rights_pending",
            "Taylor_Research_Group_origin_and_rights_review_pending",
            False,
        )
    return ("quarantine_unmapped_origin", "origin_not_in_frozen_disposition_contract", False)


def _structure_status(smiles: str | None, inchi: str | None) -> str:
    if smiles and inchi:
        return "raw_smiles_and_inchi_present_not_canonicalized"
    if smiles:
        return "raw_smiles_only_present_not_canonicalized"
    if inchi:
        return "raw_inchi_only_present_not_canonicalized"
    return "no_structure_representation_quarantine"


def _target_fields(record: Mapping[str, str]) -> tuple[str, str]:
    accessions: list[dict[str, Any]] = []
    sequences: list[dict[str, Any]] = []
    for chain in range(1, 51):
        for tier, prefix in (
            ("Swiss-Prot", "UniProt (SwissProt) Primary ID of Target Chain"),
            ("TrEMBL", "UniProt (TrEMBL) Primary ID of Target Chain"),
        ):
            raw = record.get(f"{prefix} {chain}", "").strip()
            if raw:
                accessions.append({"chain": chain, "source_tier": tier, "raw_primary_id": raw})
        sequence = record.get(f"BindingDB Target Chain Sequence {chain}", "").strip()
        if sequence:
            sequences.append(
                {
                    "chain": chain,
                    "length": len(sequence),
                    "sequence_sha256": hashlib.sha256(sequence.encode("ascii", errors="replace")).hexdigest(),
                    "alphabet_syntax_valid": AMINO_ACID_RE.fullmatch(sequence) is not None,
                }
            )
    return canonical_json_text(accessions), canonical_json_text(sequences)


def normalize_bindingdb(
    binding: InputBinding, output_root: Path
) -> tuple[list[ParquetArtifact], dict[str, Any]]:
    manifest = binding.manifest
    archive_path = binding.root / BINDINGDB_ARTICLES_ARCHIVE
    parse_inventory = manifest.get("parse_inventory")
    if not isinstance(parse_inventory, list):
        raise NormalizationError("BindingDB parse inventory is missing")
    declared_member: dict[str, Any] | None = None
    for archive in parse_inventory:
        if isinstance(archive, dict) and archive.get("file") == BINDINGDB_ARTICLES_ARCHIVE:
            members = archive.get("members")
            if isinstance(members, list) and len(members) == 1 and isinstance(members[0], dict):
                declared_member = members[0]
    if declared_member is None:
        raise NormalizationError("BindingDB article member is absent from parse inventory")

    csv.field_size_limit(sys.maxsize)
    with zipfile.ZipFile(archive_path) as archive:
        names = [name for name in archive.namelist() if not name.endswith("/")]
        if names != [BINDINGDB_ARTICLES_MEMBER]:
            raise NormalizationError(f"BindingDB article archive membership changed: {names}")
        with archive.open(BINDINGDB_ARTICLES_MEMBER) as binary:
            text = io.TextIOWrapper(binary, encoding="utf-8-sig", newline="")
            reader = csv.reader(text, delimiter="\t")
            header = next(reader)
            if len(header) != len(set(header)):
                raise NormalizationError("BindingDB source columns are not unique")
            if header != declared_member.get("columns"):
                raise NormalizationError("BindingDB article columns changed from acquisition inventory")
            reactant_counts: Counter[str] = Counter()
            first_pass_rows = 0
            for values in reader:
                first_pass_rows += 1
                if len(values) != len(header):
                    raise NormalizationError(
                        f"BindingDB row width changed at row {first_pass_rows}: {len(values)}"
                    )
                reactant_counts[values[0].strip()] += 1
    if first_pass_rows != int(declared_member.get("data_row_count", -1)):
        raise NormalizationError("BindingDB physical row count changed")

    header_document = document_with_sha256(
        {
            "schema_version": SCHEMA_VERSION,
            "source_id": BINDINGDB_SOURCE_ID,
            "source_member": BINDINGDB_ARTICLES_MEMBER,
            "source_column_count": len(header),
            "source_columns": header,
            "preservation_contract": (
                "source_record_json is a canonical JSON field map containing every listed source "
                "column and exact parsed string value for each physical row"
            ),
        }
    )
    header_path = output_root / "bindingdb" / "source_columns.json"
    header_path.parent.mkdir(parents=True, exist_ok=True)
    header_path.write_bytes(
        json.dumps(header_document, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8") + b"\n"
    )

    disposition_counts: Counter[str] = Counter()
    origin_counts: Counter[str] = Counter()
    source_endpoint_counts: Counter[str] = Counter()
    candidate_endpoint_counts: Counter[str] = Counter()
    parse_status_counts: Counter[str] = Counter()
    repeated_member_rows = 0
    occurrence: Counter[str] = Counter()

    def _iter_source_records() -> Iterator[tuple[int, dict[str, str]]]:
        with zipfile.ZipFile(archive_path) as archive, archive.open(BINDINGDB_ARTICLES_MEMBER) as binary:
            text = io.TextIOWrapper(binary, encoding="utf-8-sig", newline="")
            reader = csv.reader(text, delimiter="\t")
            current_header = next(reader)
            if current_header != header:
                raise NormalizationError("BindingDB header drifted between deterministic passes")
            for row_number, values in enumerate(reader, start=1):
                if len(values) != len(header):
                    raise NormalizationError(f"BindingDB row width changed at row {row_number}")
                yield row_number, dict(zip(header, values, strict=True))

    def _article_rows() -> Iterator[dict[str, Any]]:
        nonlocal repeated_member_rows
        for row_number, record in _iter_source_records():
            source_json = canonical_json_text(record)
            row_sha = hashlib.sha256(source_json.encode("utf-8")).hexdigest()
            reactant_set_id = record["BindingDB Reactant_set_id"].strip()
            occurrence[reactant_set_id] += 1
            total_occurrences = reactant_counts[reactant_set_id]
            repeated = total_occurrences > 1
            repeated_member_rows += int(repeated)
            origin = record["Curation/DataSource"].strip()
            disposition, reason, admitted = _bindingdb_disposition(origin)
            disposition_counts[disposition] += 1
            origin_counts[origin] += 1
            for endpoint, column in AFFINITY_ENDPOINT_COLUMNS:
                if record[column].strip():
                    source_endpoint_counts[endpoint] += 1
                    if admitted:
                        candidate_endpoint_counts[endpoint] += 1
            smiles = _nonblank(record["Ligand SMILES"])
            inchi = _nonblank(record["Ligand InChI"])
            inchi_key = _nonblank(record["Ligand InChI Key"])
            accessions_json, sequences_json = _target_fields(record)
            yield {
                "source_id": BINDINGDB_SOURCE_ID,
                "source_member": BINDINGDB_ARTICLES_MEMBER,
                "source_row_number_one_based": row_number,
                "source_column_count": len(header),
                "source_field_map_sha256": row_sha,
                "reactant_set_id": reactant_set_id or None,
                "reactant_set_occurrence": occurrence[reactant_set_id],
                "reactant_set_total_occurrences": total_occurrences,
                "repeated_reactant_set_id": repeated,
                "curation_data_source": origin,
                "row_disposition": disposition,
                "disposition_reason": reason,
                "candidate_evidence_admitted": admitted,
                "model_label_admitted": False,
                "ligand_smiles": smiles,
                "ligand_inchi": inchi,
                "ligand_inchi_key": inchi_key,
                "ligand_inchi_key_syntax_valid": (
                    None if inchi_key is None else INCHIKEY_RE.fullmatch(inchi_key) is not None
                ),
                "structure_representation_status": _structure_status(smiles, inchi),
                "bindingdb_monomer_id": _nonblank(record["BindingDB MonomerID"]),
                "ligand_name": _nonblank(record["BindingDB Ligand Name"]),
                "target_name": _nonblank(record["Target Name"]),
                "target_source_organism": _nonblank(
                    record["Target Source Organism According to Curator or DataSource"]
                ),
                "target_chain_count_raw": _nonblank(
                    record["Number of Protein Chains in Target (>1 implies a multichain complex)"]
                ),
                "target_accessions_json": accessions_json,
                "target_sequences_json": sequences_json,
                "article_doi": _nonblank(record["Article DOI"]),
                "bindingdb_entry_doi": _nonblank(record["BindingDB Entry DOI"]),
                "pmid": _nonblank(record["PMID"]),
                "publication_date_raw": _nonblank(record["Date of publication"]),
                "source_record_json": source_json,
            }

    article_path = output_root / "bindingdb" / "article_rows.parquet"
    article_artifact = write_parquet_rows(article_path, BINDINGDB_ROWS_SCHEMA, _article_rows())
    if article_artifact.rows != first_pass_rows:
        raise NormalizationError("BindingDB article output did not reconcile to source rows")

    def _affinity_rows() -> Iterator[dict[str, Any]]:
        for row_number, record in _iter_source_records():
            origin = record["Curation/DataSource"].strip()
            _disposition, _reason, admitted = _bindingdb_disposition(origin)
            if not admitted:
                continue
            source_json = canonical_json_text(record)
            row_sha = hashlib.sha256(source_json.encode("utf-8")).hexdigest()
            reactant_set_id = record["BindingDB Reactant_set_id"].strip() or None
            for endpoint, column in AFFINITY_ENDPOINT_COLUMNS:
                raw = record[column]
                if not raw.strip():
                    continue
                relation, parsed_text, value_nm, parse_status = parse_affinity_value(raw)
                parse_status_counts[f"{endpoint}:{parse_status}"] += 1
                yield {
                    "source_id": BINDINGDB_SOURCE_ID,
                    "source_member": BINDINGDB_ARTICLES_MEMBER,
                    "source_row_number_one_based": row_number,
                    "source_field_map_sha256": row_sha,
                    "reactant_set_id": reactant_set_id,
                    "measurement_key": f"bindingdb:{row_number}:{endpoint}",
                    "endpoint_type": endpoint,
                    "endpoint_source_column": column,
                    "raw_value": raw,
                    "relation": relation,
                    "parsed_numeric_text": parsed_text,
                    "value_nm": value_nm,
                    "unit": "nM",
                    "parse_status": parse_status,
                    "endpoint_pooling_key": endpoint,
                    "candidate_evidence_admitted": True,
                    "model_label_admitted": False,
                }

    affinity_path = output_root / "bindingdb" / "affinity_observations.parquet"
    affinity_artifact = write_parquet_rows(affinity_path, AFFINITY_SCHEMA, _affinity_rows())
    if affinity_artifact.rows != sum(candidate_endpoint_counts.values()):
        raise NormalizationError("BindingDB affinity output did not reconcile to candidate cells")

    audit = manifest.get("articles_origin_and_endpoint_audit")
    if not isinstance(audit, dict):
        raise NormalizationError("BindingDB source audit is missing")
    if dict(origin_counts) != audit.get("curation_data_source_counts"):
        raise NormalizationError("BindingDB origin counts differ from frozen acquisition audit")
    if source_endpoint_counts != Counter(audit.get("endpoint_nonblank_row_counts", {})):
        kinetic_only = {"kon (M-1-s-1)", "koff (s-1)"}
        acquisition_affinity = {
            key.replace(" (nM)", ""): value
            for key, value in audit.get("endpoint_nonblank_row_counts", {}).items()
            if key not in kinetic_only
        }
        if source_endpoint_counts != Counter(acquisition_affinity):
            raise NormalizationError("BindingDB endpoint counts differ from frozen acquisition audit")
    duplicate_excess = first_pass_rows - len(reactant_counts)
    if duplicate_excess != int(audit.get("duplicate_reactant_set_id_rows", -1)):
        raise NormalizationError("BindingDB repeated Reactant_set_id count changed")
    if int(audit.get("physical_measurement_rows", -1)) != first_pass_rows:
        raise NormalizationError("BindingDB source row total changed")

    header_entry = {
        "path": header_path.relative_to(output_root).as_posix(),
        "artifact_role": "source_column_contract",
        "bytes": header_path.stat().st_size,
        "sha256": sha256_file(header_path),
        "rows": len(header),
    }
    summary = {
        "source_rows": first_pass_rows,
        "source_columns": len(header),
        "row_disposition_counts": dict(sorted(disposition_counts.items())),
        "source_origin_counts": dict(sorted(origin_counts.items())),
        "candidate_rows": disposition_counts["candidate_independent_bindingdb_curated"],
        "excluded_chembl_mirror_rows": disposition_counts["excluded_chembl_cross_source_mirror"],
        "quarantined_taylor_rows": disposition_counts["quarantine_taylor_origin_rights_pending"],
        "unmapped_origin_rows": disposition_counts["quarantine_unmapped_origin"],
        "unique_reactant_set_ids": len(reactant_counts),
        "duplicate_reactant_set_id_excess_rows": duplicate_excess,
        "rows_belonging_to_repeated_reactant_set_ids": repeated_member_rows,
        "all_repeated_reactant_set_rows_retained": True,
        "source_endpoint_nonblank_counts": dict(sorted(source_endpoint_counts.items())),
        "candidate_endpoint_nonblank_counts": dict(sorted(candidate_endpoint_counts.items())),
        "candidate_endpoint_parse_status_counts": dict(sorted(parse_status_counts.items())),
        "affinity_observation_rows": affinity_artifact.rows,
        "endpoint_pooling_performed": False,
        "structure_canonicalization_performed": False,
        "model_labels_admitted": 0,
        "extra_output_entries": [header_entry],
    }
    return [article_artifact, affinity_artifact], summary


def _protein_name(entry: Mapping[str, Any]) -> str | None:
    description = entry.get("proteinDescription")
    if not isinstance(description, dict):
        return None
    recommended = description.get("recommendedName")
    if isinstance(recommended, dict):
        full = recommended.get("fullName")
        if isinstance(full, dict) and isinstance(full.get("value"), str):
            return str(full["value"])
    submission = description.get("submissionNames")
    if isinstance(submission, list) and submission and isinstance(submission[0], dict):
        full = submission[0].get("fullName")
        if isinstance(full, dict) and isinstance(full.get("value"), str):
            return str(full["value"])
    return None


def _int_or_none(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise NormalizationError(f"Expected integer-compatible source value, got {value!r}") from error


def normalize_uniprot(
    binding: InputBinding, output_root: Path
) -> tuple[list[ParquetArtifact], dict[str, Any]]:
    manifest = binding.manifest
    page_records = manifest.get("pages")
    if not isinstance(page_records, list):
        raise NormalizationError("UniProt page inventory is missing")
    sorted_pages = sorted(page_records, key=lambda item: int(item["page_index"]))
    if [int(item["page_index"]) for item in sorted_pages] != list(range(len(sorted_pages))):
        raise NormalizationError("UniProt page indexes are not contiguous")

    returned_lookup: dict[str, dict[str, Any]] = {}
    returned_type_counts: Counter[str] = Counter()
    returned_sequence_counts: Counter[str] = Counter()

    def _entry_rows() -> Iterator[dict[str, Any]]:
        for page in sorted_pages:
            relative = _safe_relative_path(page["path"], context="UniProt page")
            page_path = binding.root / Path(*relative.parts)
            payload = json.loads(page_path.read_text(encoding="utf-8"))
            results = payload.get("results")
            if not isinstance(results, list):
                raise NormalizationError(f"UniProt page results are missing: {relative}")
            if len(results) != int(page.get("returned_primary_count", -1)):
                raise NormalizationError(f"UniProt page result count changed: {relative}")
            for result_index, raw_entry in enumerate(results):
                if not isinstance(raw_entry, dict):
                    raise NormalizationError(f"UniProt entry is not an object: {relative}")
                accession = raw_entry.get("primaryAccession")
                entry_type = raw_entry.get("entryType")
                if not isinstance(accession, str) or not accession or not isinstance(entry_type, str):
                    raise NormalizationError(f"UniProt entry identity is missing: {relative}")
                if accession in returned_lookup:
                    raise NormalizationError(f"Duplicate returned UniProt accession: {accession}")
                sequence_value = raw_entry.get("sequence")
                sequence: str | None = None
                sequence_length: int | None = None
                sequence_md5: str | None = None
                sequence_sha: str | None = None
                sequence_version: int | None = None
                if isinstance(sequence_value, dict) and isinstance(sequence_value.get("value"), str):
                    sequence = str(sequence_value["value"])
                    sequence_length = _int_or_none(sequence_value.get("length"))
                    if sequence_length != len(sequence):
                        raise NormalizationError(f"UniProt sequence length failed: {accession}")
                    sequence_md5 = (
                        hashlib.md5(sequence.encode("ascii"), usedforsecurity=False).hexdigest().upper()
                    )
                    declared_md5 = sequence_value.get("md5")
                    if declared_md5 is not None and sequence_md5 != str(declared_md5).upper():
                        raise NormalizationError(f"UniProt sequence MD5 failed: {accession}")
                    sequence_sha = hashlib.sha256(sequence.encode("ascii")).hexdigest()
                    sequence_version = _int_or_none(sequence_value.get("version"))
                    sequence_status = "sequence_ready"
                else:
                    sequence_status = "inactive_sequence_unavailable_quarantine"
                    if entry_type != "Inactive":
                        raise NormalizationError(
                            f"Non-inactive UniProt entry unexpectedly lacks sequence: {accession}"
                        )
                organism = raw_entry.get("organism")
                organism = organism if isinstance(organism, dict) else {}
                audit = raw_entry.get("entryAudit")
                audit = audit if isinstance(audit, dict) else {}
                raw_sha = hashlib.sha256(canonical_json_bytes(raw_entry)).hexdigest()
                returned_lookup[accession] = {
                    "entry_type": entry_type,
                    "sequence_status": sequence_status,
                    "sequence_sha256": sequence_sha,
                }
                returned_type_counts[entry_type] += 1
                returned_sequence_counts[sequence_status] += 1
                yield {
                    "source_id": UNIPROT_SOURCE_ID,
                    "source_page_path": relative.as_posix(),
                    "source_result_index_zero_based": result_index,
                    "raw_entry_sha256": raw_sha,
                    "returned_primary_accession": accession,
                    "entry_type": entry_type,
                    "entry_name": _nonblank(str(raw_entry.get("uniProtkbId", ""))),
                    "protein_name": _protein_name(raw_entry),
                    "taxonomy_id": _int_or_none(organism.get("taxonId")),
                    "organism_scientific_name": (
                        str(organism["scientificName"])
                        if isinstance(organism.get("scientificName"), str)
                        else None
                    ),
                    "sequence_status": sequence_status,
                    "sequence": sequence,
                    "sequence_length": sequence_length,
                    "sequence_md5": sequence_md5,
                    "sequence_sha256": sequence_sha,
                    "sequence_version": sequence_version,
                    "entry_version": _int_or_none(audit.get("entryVersion")),
                    "model_label_admitted": False,
                }

    entry_path = output_root / "uniprot" / "returned_entries.parquet"
    entry_artifact = write_parquet_rows(entry_path, UNIPROT_ENTRY_SCHEMA, _entry_rows(), batch_rows=256)
    inventory = manifest.get("protein_entry_inventory")
    if not isinstance(inventory, dict):
        raise NormalizationError("UniProt protein entry inventory is missing")
    if entry_artifact.rows != int(inventory.get("unique_returned_primary_entries", -1)):
        raise NormalizationError("UniProt returned entry count changed")
    if returned_sequence_counts["sequence_ready"] != int(inventory.get("sequence_ready_entries", -1)):
        raise NormalizationError("UniProt sequence-ready count changed")
    if returned_type_counts != Counter(inventory.get("entry_type_counts", {})):
        raise NormalizationError("UniProt returned entry-type counts changed")

    resolution_path = binding.root / str(manifest["accession_resolution_inventory"]["path"])
    resolution_lookup: dict[str, dict[str, Any]] = {}
    resolution_counts: Counter[str] = Counter()
    disposition_counts: Counter[str] = Counter()

    def _resolution_rows() -> Iterator[dict[str, Any]]:
        with resolution_path.open("r", encoding="utf-8") as handle:
            for row_number, line in enumerate(handle, start=1):
                raw = json.loads(line)
                if not isinstance(raw, dict):
                    raise NormalizationError("UniProt resolution row is not an object")
                requested = str(raw["requested_accession"])
                if requested in resolution_lookup:
                    raise NormalizationError(f"Duplicate requested accession: {requested}")
                state = str(raw["resolution_state"])
                returned = raw.get("returned_primary_accession")
                sequence_status = "not_sequence_ready"
                sequence_sha = None
                if state == "resolved_primary":
                    if not isinstance(returned, str) or returned not in returned_lookup:
                        raise NormalizationError(f"Resolved UniProt entry is absent: {requested}")
                    joined = returned_lookup[returned]
                    sequence_status = str(joined["sequence_status"])
                    sequence_sha = joined["sequence_sha256"]
                    disposition = (
                        "sequence_ready"
                        if sequence_status == "sequence_ready"
                        else "inactive_sequence_unavailable_quarantine"
                    )
                elif state == "ambiguous_multi_mapped_quarantine":
                    disposition = "ambiguous_accession_quarantine"
                    returned = None
                    sequence_status = "ambiguous_not_selected"
                elif state == "non_uniprot_identifier_syntax_quarantine":
                    disposition = "identifier_syntax_quarantine"
                    returned = None
                    sequence_status = "not_applicable_non_uniprot_identifier"
                else:
                    raise NormalizationError(f"Unhandled UniProt resolution state: {state}")
                source_json = canonical_json_text(raw)
                resolution_lookup[requested] = {
                    "resolution_state": state,
                    "normalization_disposition": disposition,
                    "returned_primary_accession": returned,
                    "sequence_sha256": sequence_sha,
                }
                resolution_counts[state] += 1
                disposition_counts[disposition] += 1
                returned_values = raw.get("returned_primary_accessions", [])
                if not isinstance(returned_values, list):
                    raise NormalizationError(f"Invalid returned accession list: {requested}")
                yield {
                    "source_id": UNIPROT_SOURCE_ID,
                    "source_row_number_one_based": row_number,
                    "requested_accession": requested,
                    "resolution_state": state,
                    "normalization_disposition": disposition,
                    "returned_primary_accession": returned,
                    "returned_primary_accessions_json": canonical_json_text(returned_values),
                    "entry_type": raw.get("entry_type"),
                    "sequence_status": sequence_status,
                    "sequence_sha256": sequence_sha,
                    "replacement_state": raw.get("replacement_state"),
                    "silent_identity_replacement_performed": False,
                    "model_label_admitted": False,
                    "source_record_json": source_json,
                }

    normalized_resolution_path = output_root / "uniprot" / "accession_resolution.parquet"
    resolution_artifact = write_parquet_rows(
        normalized_resolution_path, UNIPROT_RESOLUTION_SCHEMA, _resolution_rows()
    )
    declared_resolution_rows = int(manifest["accession_resolution_inventory"]["rows"])
    if resolution_artifact.rows != declared_resolution_rows:
        raise NormalizationError("UniProt resolution row count changed")
    source_resolution_counts = manifest.get("resolution_counts")
    if not isinstance(source_resolution_counts, dict):
        raise NormalizationError("UniProt resolution count audit is missing")
    expected_states = {
        "resolved_primary": int(source_resolution_counts["resolved"]),
        "ambiguous_multi_mapped_quarantine": int(source_resolution_counts["ambiguous_multi_mapped"]),
        "non_uniprot_identifier_syntax_quarantine": int(
            source_resolution_counts["non_uniprot_identifier_syntax_quarantine"]
        ),
    }
    if resolution_counts != Counter(expected_states):
        raise NormalizationError("UniProt resolution states changed")

    membership_path = binding.root / str(manifest["accession_source_membership"]["path"])
    membership_source_counts: Counter[str] = Counter()
    membership_dispositions: Counter[str] = Counter()

    def _membership_rows() -> Iterator[dict[str, Any]]:
        with membership_path.open("r", encoding="utf-8") as handle:
            for row_number, line in enumerate(handle, start=1):
                raw = json.loads(line)
                if not isinstance(raw, dict):
                    raise NormalizationError("UniProt membership row is not an object")
                source_state = str(raw["admission_state"])
                normalized = raw.get("normalized_identifier")
                if source_state == "identifier_syntax_quarantine":
                    resolution_state = "non_uniprot_identifier_syntax_quarantine"
                    disposition = "identifier_syntax_quarantine"
                    returned = None
                    sequence_sha = None
                else:
                    if not isinstance(normalized, str) or normalized not in resolution_lookup:
                        raise NormalizationError(
                            f"UniProt membership identifier lacks resolution: {normalized!r}"
                        )
                    joined = resolution_lookup[normalized]
                    resolution_state = str(joined["resolution_state"])
                    disposition = str(joined["normalization_disposition"])
                    returned = joined["returned_primary_accession"]
                    sequence_sha = joined["sequence_sha256"]
                membership_source_counts[source_state] += 1
                membership_dispositions[disposition] += 1
                yield {
                    "source_id": UNIPROT_SOURCE_ID,
                    "source_row_number_one_based": row_number,
                    "upstream_source_id": str(raw["source_id"]),
                    "upstream_source_file": str(raw["source_file"]),
                    "upstream_source_row_index_zero_based": _int_or_none(
                        raw.get("source_row_index_zero_based")
                    ),
                    "upstream_source_target_id": raw.get("source_target_id"),
                    "upstream_source_component_id": _int_or_none(raw.get("source_component_id")),
                    "source_accession_value": raw.get("source_accession_value"),
                    "normalized_identifier": normalized,
                    "source_admission_state": source_state,
                    "resolution_state": resolution_state,
                    "normalization_disposition": disposition,
                    "returned_primary_accession": returned,
                    "sequence_sha256": sequence_sha,
                    "silent_identity_replacement_performed": False,
                    "model_label_admitted": False,
                    "source_record_json": canonical_json_text(raw),
                }

    normalized_membership_path = output_root / "uniprot" / "source_membership.parquet"
    membership_artifact = write_parquet_rows(
        normalized_membership_path, UNIPROT_MEMBERSHIP_SCHEMA, _membership_rows()
    )
    if membership_artifact.rows != int(manifest["accession_source_membership"]["identifier_reference_rows"]):
        raise NormalizationError("UniProt membership row count changed")
    expected_membership_states = Counter(
        {
            "request_candidate": int(
                manifest["accession_source_membership"]["valid_uniprot_accession_reference_rows"]
            ),
            "identifier_syntax_quarantine": int(
                manifest["accession_source_membership"]["identifier_syntax_quarantine_reference_rows"]
            ),
        }
    )
    if membership_source_counts != expected_membership_states:
        raise NormalizationError("UniProt membership admission states changed")

    summary = {
        "returned_entry_rows": entry_artifact.rows,
        "returned_entry_type_counts": dict(sorted(returned_type_counts.items())),
        "returned_sequence_status_counts": dict(sorted(returned_sequence_counts.items())),
        "sequence_ready_entries": returned_sequence_counts["sequence_ready"],
        "inactive_returned_entries": returned_type_counts["Inactive"],
        "resolution_rows": resolution_artifact.rows,
        "resolution_state_counts": dict(sorted(resolution_counts.items())),
        "resolution_disposition_counts": dict(sorted(disposition_counts.items())),
        "membership_rows": membership_artifact.rows,
        "membership_source_state_counts": dict(sorted(membership_source_counts.items())),
        "membership_disposition_counts": dict(sorted(membership_dispositions.items())),
        "sequence_md5_and_sha256_verified": returned_sequence_counts["sequence_ready"],
        "silent_chembl_identity_replacements": 0,
        "model_labels_admitted": 0,
    }
    return [entry_artifact, resolution_artifact, membership_artifact], summary


def normalize_clinicaltrials(
    binding: InputBinding, output_root: Path
) -> tuple[list[ParquetArtifact], dict[str, Any]]:
    manifest = binding.manifest
    broad = manifest.get("alias_independent_all_drug_cohort")
    heuristic = manifest.get("cardiac_safety_heuristic_cohort")
    if not isinstance(broad, dict) or not isinstance(heuristic, dict):
        raise NormalizationError("ClinicalTrials.gov cohorts are incomplete")
    cohort_specs = (
        (
            "broad_all_drug_registry_inventory",
            broad,
            "registry_and_study_state_inventory_only; no efficacy, PK, QT, safety, or molecular label",
            False,
        ),
        (
            "heuristic_cardiac_safety_inventory",
            heuristic,
            "unreviewed_text_search_membership_and_posted_module_presence_only; no outcome interpretation",
            True,
        ),
    )
    counts: Counter[str] = Counter()
    unique_by_cohort: dict[str, set[str]] = {item[0]: set() for item in cohort_specs}
    overlap: set[str] | None = None

    def _rows() -> Iterator[dict[str, Any]]:
        nonlocal overlap
        for cohort_kind, cohort, semantics, ambiguous in cohort_specs:
            inventory = cohort.get("nct_membership_inventory")
            if not isinstance(inventory, dict):
                raise NormalizationError(f"ClinicalTrials.gov inventory missing: {cohort_kind}")
            relative = _safe_relative_path(inventory["path"], context="ClinicalTrials.gov inventory")
            path = binding.root / Path(*relative.parts)
            with path.open("r", encoding="utf-8") as handle:
                for row_number, line in enumerate(handle, start=1):
                    raw = json.loads(line)
                    if not isinstance(raw, dict):
                        raise NormalizationError("ClinicalTrials.gov inventory row is not an object")
                    nct_id = str(raw["nct_id"])
                    if nct_id in unique_by_cohort[cohort_kind]:
                        raise NormalizationError(f"Duplicate NCT ID in cohort {cohort_kind}: {nct_id}")
                    unique_by_cohort[cohort_kind].add(nct_id)
                    counts[cohort_kind] += 1
                    matches = raw.get("projected_heuristic_term_matches", [])
                    if not isinstance(matches, list):
                        raise NormalizationError(f"Invalid heuristic term matches: {nct_id}")
                    yield {
                        "source_id": CLINICALTRIALS_SOURCE_ID,
                        "cohort_kind": cohort_kind,
                        "cohort_snapshot_key": str(cohort["cohort_snapshot_key"]),
                        "source_row_number_one_based": row_number,
                        "nct_id": nct_id,
                        "page_index": int(raw["page_index"]),
                        "study_index_within_page": int(raw["study_index_within_page"]),
                        "study_sha256": str(raw["study_sha256"]),
                        "heuristic_term_matches_json": canonical_json_text(matches),
                        "has_posted_outcome_measures_module": raw.get("has_posted_outcome_measures_module"),
                        "has_posted_adverse_events_module": raw.get("has_posted_adverse_events_module"),
                        "evidence_semantics": semantics,
                        "false_positive_or_context_ambiguity_retained": ambiguous,
                        "canonical_observation_admitted": False,
                        "model_label_admitted": False,
                        "source_record_json": canonical_json_text(raw),
                    }
            if counts[cohort_kind] != int(inventory.get("rows", -1)):
                raise NormalizationError(f"ClinicalTrials.gov cohort row count changed: {cohort_kind}")
            if len(unique_by_cohort[cohort_kind]) != int(cohort.get("unique_nct_count", -1)):
                raise NormalizationError(f"ClinicalTrials.gov unique membership changed: {cohort_kind}")
        overlap = unique_by_cohort[cohort_specs[0][0]] & unique_by_cohort[cohort_specs[1][0]]

    path = output_root / "clinicaltrials" / "cohort_membership.parquet"
    artifact = write_parquet_rows(path, CLINICALTRIALS_MEMBERSHIP_SCHEMA, _rows())
    if artifact.rows != sum(counts.values()):
        raise NormalizationError("ClinicalTrials.gov output membership did not reconcile")
    summary = {
        "cohort_membership_rows": artifact.rows,
        "cohort_counts": dict(sorted(counts.items())),
        "broad_vs_heuristic_membership_overlap": len(overlap or set()),
        "broad_scope": "registry_and_status_inventory_only",
        "heuristic_scope": "unreviewed_text_search_inventory_with_false_positives_retained",
        "raw_posted_outcomes_interpreted": False,
        "raw_posted_adverse_events_interpreted": False,
        "canonical_observations_admitted": 0,
        "model_labels_admitted": 0,
    }
    return [artifact], summary


def normalize_regulatory_inventories(
    drugs: InputBinding,
    dailymed: InputBinding,
    output_root: Path,
) -> tuple[list[ParquetArtifact], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    drugs_table = drugs.manifest.get("archive_member_table")
    if not isinstance(drugs_table, dict) or not isinstance(drugs_table.get("members"), list):
        raise NormalizationError("Drugs@FDA archive-member table is missing")
    for member in drugs_table["members"]:
        if not isinstance(member, dict):
            raise NormalizationError("Drugs@FDA table metadata is invalid")
        anomalies = int(member.get("malformed_width_rows", 0))
        rows.append(
            {
                "source_id": DRUGSFDA_SOURCE_ID,
                "inventory_kind": "relational_table",
                "item_id": str(member["archive_member_path"]),
                "row_or_member_count": int(member["data_row_count"]),
                "compressed_or_source_bytes": None,
                "uncompressed_bytes": None,
                "sha256": None,
                "parse_or_verification_status": str(member["parse_integrity"]),
                "source_anomaly_count": anomalies,
                "evidence_semantics": (
                    "regulatory application/product/action table inventory only; approval and marketing "
                    "state are not molecular efficacy, safety, PK, or activity labels"
                ),
                "canonical_observation_admitted": False,
                "model_label_admitted": False,
                "source_metadata_json": canonical_json_text(member),
            }
        )
    if len(rows) != int(drugs_table.get("txt_table_count", -1)):
        raise NormalizationError("Drugs@FDA table inventory count changed")
    if sum(int(item["row_or_member_count"] or 0) for item in rows) != int(
        drugs_table.get("total_data_rows", -1)
    ):
        raise NormalizationError("Drugs@FDA table row total changed")

    dailymed_part_count = 0
    dailymed_member_count = 0
    for item in dailymed.manifest.get("files", []):
        if not isinstance(item, dict) or item.get("artifact_role") != "human_prescription_release_part":
            continue
        archive = item.get("archive_integrity")
        if not isinstance(archive, dict):
            raise NormalizationError("DailyMed release part lacks archive integrity")
        dailymed_part_count += 1
        member_count = int(archive["file_member_count"])
        dailymed_member_count += member_count
        rows.append(
            {
                "source_id": DAILYMED_SOURCE_ID,
                "inventory_kind": "archive_release_part",
                "item_id": str(item["local_path"]),
                "row_or_member_count": member_count,
                "compressed_or_source_bytes": int(item["acquired_bytes"]),
                "uncompressed_bytes": int(archive["total_member_uncompressed_bytes"]),
                "sha256": str(item["acquired_sha256"]),
                "parse_or_verification_status": "archive_crc_and_membership_verified_not_extracted",
                "source_anomaly_count": 0,
                "evidence_semantics": (
                    "versioned SPL archive inventory only; prose and section presence are not normalized "
                    "molecule-level safety, PK, efficacy, or cardiotoxicity labels"
                ),
                "canonical_observation_admitted": False,
                "model_label_admitted": False,
                "source_metadata_json": canonical_json_text(item),
            }
        )
    if dailymed_part_count != int(dailymed.manifest.get("release_part_count", -1)):
        raise NormalizationError("DailyMed release-part count changed")
    if dailymed_member_count != int(dailymed.manifest.get("expected_and_verified_file_member_count", -1)):
        raise NormalizationError("DailyMed archive member total changed")

    path = output_root / "regulatory" / "archive_inventory.parquet"
    artifact = write_parquet_rows(path, REGULATORY_INVENTORY_SCHEMA, rows)
    relational = drugs.manifest.get("relational_key_and_join_audit", {})
    summary = {
        "inventory_rows": artifact.rows,
        "drugs_at_fda_table_rows": len(drugs_table["members"]),
        "drugs_at_fda_total_data_rows": int(drugs_table["total_data_rows"]),
        "drugs_at_fda_source_width_anomaly_rows": int(drugs_table["source_width_anomaly_rows"]),
        "drugs_at_fda_blank_primary_key_rows": int(relational.get("total_blank_primary_key_rows", 0)),
        "drugs_at_fda_missing_foreign_key_rows_across_relations": int(
            relational.get("total_missing_foreign_key_rows_across_relations", 0)
        ),
        "dailymed_release_parts": dailymed_part_count,
        "dailymed_verified_archive_members": dailymed_member_count,
        "dailymed_section_extraction_attempted": False,
        "canonical_observations_admitted": 0,
        "model_labels_admitted": 0,
    }
    return [artifact], summary


def normalize_source_inventory(
    bindings: Sequence[InputBinding], raw_root: Path, output_root: Path
) -> ParquetArtifact:
    scopes = {
        BINDINGDB_SOURCE_ID: "row_disposition_plus_conservative_binding_affinity_candidates",
        UNIPROT_SOURCE_ID: "accession_resolution_and_sequence_enrichment_inventory",
        CLINICALTRIALS_SOURCE_ID: "broad_and_heuristic_cohort_membership_inventory_only",
        DRUGSFDA_SOURCE_ID: "relational_archive_table_metadata_only",
        DAILYMED_SOURCE_ID: "verified_archive_part_and_member_inventory_only",
    }
    rows = []
    for binding in sorted(bindings, key=lambda item: item.source_id):
        boundary = binding.manifest.get("semantic_and_rights_boundaries", {})
        rows.append(
            {
                "source_id": binding.source_id,
                "release_id": str(binding.manifest["release_id"]),
                "snapshot_status": str(binding.manifest["snapshot_status"]),
                "declared_manifest_sha256": binding.declared_manifest_sha256,
                "physical_manifest_sha256": binding.physical_manifest_sha256,
                "bundle_entries_sha256": binding.bundle_entries_sha256,
                "bundle_entry_count": binding.bundle_entry_count,
                "bundle_total_bytes": binding.bundle_total_bytes,
                "normalization_scope": scopes[binding.source_id],
                "canonical_observation_admitted": False,
                "model_label_admitted": False,
                "limitations_json": canonical_json_text(boundary),
            }
        )
    path = output_root / "source_inventory.parquet"
    return write_parquet_rows(path, SOURCE_INVENTORY_SCHEMA, rows)


def _inventory_output(output_root: Path, expected_entries: Sequence[dict[str, Any]]) -> dict[str, Any]:
    expected_by_path: dict[str, dict[str, Any]] = {}
    for item in expected_entries:
        relative_key = str(item["path"])
        if relative_key in expected_by_path:
            raise NormalizationError(f"Duplicate output inventory path: {relative_key}")
        expected_by_path[relative_key] = item
    observed: set[str] = set()
    for candidate in output_root.rglob("*"):
        if candidate.is_symlink():
            raise NormalizationError(f"Symlink prohibited in normalized output: {candidate}")
        if candidate.is_file():
            relative = candidate.relative_to(output_root).as_posix()
            if relative != "external_public_normalized_manifest.json":
                observed.add(relative)
    if observed != set(expected_by_path):
        raise NormalizationError(
            f"Output membership failed: missing={sorted(set(expected_by_path) - observed)}, "
            f"unexpected={sorted(observed - set(expected_by_path))}"
        )
    total_bytes = 0
    entries: list[dict[str, Any]] = []
    for relative in sorted(observed):
        artifact_path = output_root / relative
        expected = expected_by_path[relative]
        actual_bytes = artifact_path.stat().st_size
        actual_sha = sha256_file(artifact_path)
        if actual_bytes != int(expected["bytes"]) or actual_sha != expected["sha256"]:
            raise NormalizationError(f"Generated output drifted during inventory: {relative}")
        total_bytes += actual_bytes
        entries.append(expected)
    return {
        "root": ".",
        "included_artifacts": "every regular file recursively below normalized root",
        "excluded_paths": ["external_public_normalized_manifest.json"],
        "exclusion_reason": "output manifest is self-referential and bound by manifest_sha256",
        "symlink_policy": "rejected",
        "entry_count": len(entries),
        "total_bytes": total_bytes,
        "entries": entries,
        "entries_sha256": hashlib.sha256(canonical_json_bytes(entries)).hexdigest(),
    }


def _write_lock(lock_path: Path, transaction: Mapping[str, Any]) -> None:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        descriptor = os.open(lock_path, flags, 0o644)
    except FileExistsError as error:
        raise NormalizationError(
            f"Normalization transaction marker exists; inspect before recovery: {lock_path}"
        ) from error
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(document_with_sha256(transaction), handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _portable_report_path(path: Path, *, project_root: Path | None = None) -> str:
    """Return a non-absolute report path without disclosing a local user root."""

    base = (project_root or Path.cwd()).resolve()
    resolved = path.resolve()
    try:
        return resolved.relative_to(base).as_posix()
    except ValueError:
        return resolved.name


def _verify_standard_semantics(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute key admission, quarantine, sequence, and zero-label invariants."""

    reconciliation = manifest.get("source_to_output_reconciliation")
    if not isinstance(reconciliation, dict):
        raise NormalizationError("Normalized source reconciliation is missing")

    header_path = root / "bindingdb/source_columns.json"
    header = json.loads(header_path.read_text(encoding="utf-8"))
    if not isinstance(header, dict) or not verify_document_sha256(header):
        raise NormalizationError("BindingDB source-column contract identity failed")
    source_column_count = int(header.get("source_column_count", -1))
    if source_column_count != 640 or len(header.get("source_columns", [])) != source_column_count:
        raise NormalizationError("BindingDB source-column preservation contract changed")

    binding_summary = reconciliation.get(BINDINGDB_SOURCE_ID)
    if not isinstance(binding_summary, dict):
        raise NormalizationError("BindingDB normalized reconciliation is missing")
    dispositions: Counter[str] = Counter()
    origins: Counter[str] = Counter()
    candidate_rows = 0
    repeated_group_member_rows = 0
    article_file = pq.ParquetFile(root / "bindingdb/article_rows.parquet")
    for batch in article_file.iter_batches(
        columns=[
            "source_field_map_sha256",
            "source_record_json",
            "source_column_count",
            "row_disposition",
            "curation_data_source",
            "repeated_reactant_set_id",
            "candidate_evidence_admitted",
            "model_label_admitted",
        ],
        batch_size=1_024,
    ):
        for row in batch.to_pylist():
            if row["model_label_admitted"]:
                raise NormalizationError("BindingDB article row admitted a model label")
            source_json = str(row["source_record_json"])
            record = json.loads(source_json)
            if not isinstance(record, dict) or len(record) != source_column_count:
                raise NormalizationError("BindingDB source field-map width failed")
            if hashlib.sha256(source_json.encode("utf-8")).hexdigest() != row["source_field_map_sha256"]:
                raise NormalizationError("BindingDB source field-map identity failed")
            if int(row["source_column_count"]) != source_column_count:
                raise NormalizationError("BindingDB row source-column count failed")
            dispositions[str(row["row_disposition"])] += 1
            origins[str(row["curation_data_source"])] += 1
            candidate_rows += int(bool(row["candidate_evidence_admitted"]))
            repeated_group_member_rows += int(bool(row["repeated_reactant_set_id"]))
    if dispositions != Counter(binding_summary.get("row_disposition_counts", {})):
        raise NormalizationError("BindingDB row disposition reconciliation failed")
    if origins != Counter(binding_summary.get("source_origin_counts", {})):
        raise NormalizationError("BindingDB source-origin reconciliation failed")
    if candidate_rows != int(binding_summary.get("candidate_rows", -1)):
        raise NormalizationError("BindingDB candidate-row reconciliation failed")
    if repeated_group_member_rows != int(
        binding_summary.get("rows_belonging_to_repeated_reactant_set_ids", -1)
    ):
        raise NormalizationError("BindingDB repeated-row reconciliation failed")

    endpoint_counts: Counter[str] = Counter()
    parse_counts: Counter[str] = Counter()
    affinity_file = pq.ParquetFile(root / "bindingdb/affinity_observations.parquet")
    for batch in affinity_file.iter_batches(
        columns=[
            "endpoint_type",
            "endpoint_pooling_key",
            "parse_status",
            "candidate_evidence_admitted",
            "model_label_admitted",
        ],
        batch_size=4_096,
    ):
        for row in batch.to_pylist():
            endpoint = str(row["endpoint_type"])
            if endpoint not in {"Kd", "Ki", "IC50", "EC50"}:
                raise NormalizationError(f"Unexpected BindingDB affinity endpoint: {endpoint}")
            if endpoint != row["endpoint_pooling_key"]:
                raise NormalizationError("BindingDB endpoint pooling contract failed")
            if not row["candidate_evidence_admitted"] or row["model_label_admitted"]:
                raise NormalizationError("BindingDB affinity admission contract failed")
            endpoint_counts[endpoint] += 1
            parse_counts[str(row["parse_status"])] += 1
    if endpoint_counts != Counter(binding_summary.get("candidate_endpoint_nonblank_counts", {})):
        raise NormalizationError("BindingDB candidate endpoint reconciliation failed")
    if sum(endpoint_counts.values()) != int(binding_summary.get("affinity_observation_rows", -1)):
        raise NormalizationError("BindingDB affinity row total failed")

    uniprot_summary = reconciliation.get(UNIPROT_SOURCE_ID)
    if not isinstance(uniprot_summary, dict):
        raise NormalizationError("UniProt normalized reconciliation is missing")
    entry_types: Counter[str] = Counter()
    sequence_states: Counter[str] = Counter()
    sequence_hashes_verified = 0
    entries_file = pq.ParquetFile(root / "uniprot/returned_entries.parquet")
    for batch in entries_file.iter_batches(batch_size=512):
        for row in batch.to_pylist():
            if row["model_label_admitted"]:
                raise NormalizationError("UniProt entry admitted a model label")
            entry_type = str(row["entry_type"])
            sequence_status = str(row["sequence_status"])
            entry_types[entry_type] += 1
            sequence_states[sequence_status] += 1
            sequence = row["sequence"]
            if sequence_status == "sequence_ready":
                if not isinstance(sequence, str):
                    raise NormalizationError("UniProt sequence-ready entry lacks sequence")
                if len(sequence) != int(row["sequence_length"]):
                    raise NormalizationError("UniProt normalized sequence length failed")
                if hashlib.sha256(sequence.encode("ascii")).hexdigest() != row["sequence_sha256"]:
                    raise NormalizationError("UniProt normalized sequence SHA-256 failed")
                observed_md5 = (
                    hashlib.md5(sequence.encode("ascii"), usedforsecurity=False).hexdigest().upper()
                )
                if observed_md5 != row["sequence_md5"]:
                    raise NormalizationError("UniProt normalized sequence MD5 failed")
                sequence_hashes_verified += 1
            elif sequence is not None or entry_type != "Inactive":
                raise NormalizationError("UniProt inactive-sequence quarantine contract failed")
    if entry_types != Counter(uniprot_summary.get("returned_entry_type_counts", {})):
        raise NormalizationError("UniProt entry-type reconciliation failed")
    if sequence_states != Counter(uniprot_summary.get("returned_sequence_status_counts", {})):
        raise NormalizationError("UniProt sequence-state reconciliation failed")
    if sequence_hashes_verified != int(uniprot_summary.get("sequence_ready_entries", -1)):
        raise NormalizationError("UniProt sequence-ready reconciliation failed")

    resolution_dispositions: Counter[str] = Counter()
    resolution_file = pq.ParquetFile(root / "uniprot/accession_resolution.parquet")
    for batch in resolution_file.iter_batches(
        columns=[
            "normalization_disposition",
            "silent_identity_replacement_performed",
            "model_label_admitted",
        ],
        batch_size=4_096,
    ):
        for row in batch.to_pylist():
            if row["silent_identity_replacement_performed"] or row["model_label_admitted"]:
                raise NormalizationError("UniProt resolution prohibition failed")
            resolution_dispositions[str(row["normalization_disposition"])] += 1
    if resolution_dispositions != Counter(uniprot_summary.get("resolution_disposition_counts", {})):
        raise NormalizationError("UniProt resolution disposition reconciliation failed")

    membership_dispositions: Counter[str] = Counter()
    membership_file = pq.ParquetFile(root / "uniprot/source_membership.parquet")
    for batch in membership_file.iter_batches(
        columns=[
            "normalization_disposition",
            "silent_identity_replacement_performed",
            "model_label_admitted",
        ],
        batch_size=4_096,
    ):
        for row in batch.to_pylist():
            if row["silent_identity_replacement_performed"] or row["model_label_admitted"]:
                raise NormalizationError("UniProt membership prohibition failed")
            membership_dispositions[str(row["normalization_disposition"])] += 1
    if membership_dispositions != Counter(uniprot_summary.get("membership_disposition_counts", {})):
        raise NormalizationError("UniProt membership disposition reconciliation failed")

    clinical_summary = reconciliation.get(CLINICALTRIALS_SOURCE_ID)
    if not isinstance(clinical_summary, dict):
        raise NormalizationError("ClinicalTrials.gov normalized reconciliation is missing")
    cohort_counts: Counter[str] = Counter()
    clinical_file = pq.ParquetFile(root / "clinicaltrials/cohort_membership.parquet")
    for batch in clinical_file.iter_batches(
        columns=["cohort_kind", "canonical_observation_admitted", "model_label_admitted"],
        batch_size=8_192,
    ):
        for row in batch.to_pylist():
            if row["canonical_observation_admitted"] or row["model_label_admitted"]:
                raise NormalizationError("ClinicalTrials.gov zero-label contract failed")
            cohort_counts[str(row["cohort_kind"])] += 1
    if cohort_counts != Counter(clinical_summary.get("cohort_counts", {})):
        raise NormalizationError("ClinicalTrials.gov cohort reconciliation failed")

    regulatory_file = pq.ParquetFile(root / "regulatory/archive_inventory.parquet")
    for batch in regulatory_file.iter_batches(
        columns=["canonical_observation_admitted", "model_label_admitted"], batch_size=128
    ):
        for row in batch.to_pylist():
            if row["canonical_observation_admitted"] or row["model_label_admitted"]:
                raise NormalizationError("Regulatory inventory zero-label contract failed")

    return {
        "bindingdb_disposition_counts": dict(sorted(dispositions.items())),
        "bindingdb_candidate_endpoint_counts": dict(sorted(endpoint_counts.items())),
        "bindingdb_affinity_parse_status_counts": dict(sorted(parse_counts.items())),
        "bindingdb_repeated_group_member_rows": repeated_group_member_rows,
        "uniprot_entry_type_counts": dict(sorted(entry_types.items())),
        "uniprot_sequence_status_counts": dict(sorted(sequence_states.items())),
        "uniprot_sequence_hashes_recomputed": sequence_hashes_verified,
        "clinicaltrials_cohort_counts": dict(sorted(cohort_counts.items())),
        "all_admission_prohibitions_recomputed": True,
    }


def verify_external_normalized_output(
    output_root: str | os.PathLike[str],
    raw_root: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Independently verify a promoted normalized dataset without rebuilding it."""

    root_path = Path(output_root)
    if root_path.is_symlink() or not root_path.is_dir():
        raise NormalizationError(f"Missing or symlinked normalized output root: {root_path}")
    root = root_path.resolve()
    manifest_path = root / "external_public_normalized_manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise NormalizationError(f"Missing or symlinked normalized manifest: {manifest_path}")
    manifest_value = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest_value, dict):
        raise NormalizationError("Normalized manifest is not an object")
    manifest: dict[str, Any] = manifest_value
    if manifest.get("schema_version") != SCHEMA_VERSION or not verify_document_sha256(manifest):
        raise NormalizationError("Normalized manifest identity failed")
    if manifest.get("dataset_id") != "external_public_normalized":
        raise NormalizationError("Normalized dataset identity failed")
    if (
        int(manifest.get("canonical_observations_admitted", -1)) != 0
        or int(manifest.get("model_labels_admitted", -1)) != 0
        or manifest.get("substantive_model_training_performed") is not False
        or manifest.get("zero_training_flag") is not True
        or manifest.get("endpoint_pooling_performed") is not False
        or manifest.get("silent_cross_source_identity_replacement_performed") is not False
    ):
        raise NormalizationError("Normalized top-level zero-label/training contract failed")

    inventory = manifest.get("output_inventory")
    if not isinstance(inventory, dict):
        raise NormalizationError("Normalized output inventory is missing")
    entries = inventory.get("entries")
    if not isinstance(entries, list) or not all(isinstance(item, dict) for item in entries):
        raise NormalizationError("Normalized output inventory entries are invalid")
    if hashlib.sha256(canonical_json_bytes(entries)).hexdigest() != inventory.get("entries_sha256"):
        raise NormalizationError("Normalized output inventory entry digest failed")
    if int(inventory.get("entry_count", -1)) != len(entries):
        raise NormalizationError("Normalized output inventory entry count failed")
    if inventory.get("excluded_paths") != [manifest_path.name]:
        raise NormalizationError("Normalized output inventory exclusion failed")

    expected: dict[str, dict[str, Any]] = {}
    for item in entries:
        relative = _safe_relative_path(item.get("path"), context="normalized output")
        key = relative.as_posix()
        if key in expected:
            raise NormalizationError(f"Duplicate normalized output path: {key}")
        expected[key] = item
    observed: set[str] = set()
    for candidate in root.rglob("*"):
        if candidate.is_symlink():
            raise NormalizationError(f"Symlink prohibited in normalized output: {candidate}")
        if candidate.is_file() and candidate != manifest_path:
            observed.add(candidate.relative_to(root).as_posix())
    if observed != set(expected):
        raise NormalizationError(
            f"Normalized output membership changed: missing={sorted(set(expected) - observed)}, "
            f"unexpected={sorted(observed - set(expected))}"
        )

    aggregate_bytes = 0
    aggregate_rows = 0
    parquet_count = 0
    for output_relative, item in sorted(expected.items()):
        path = root / output_relative
        expected_bytes = int(item.get("bytes", -1))
        if path.stat().st_size != expected_bytes:
            raise NormalizationError(f"Normalized output byte count changed: {output_relative}")
        if sha256_file(path) != item.get("sha256"):
            raise NormalizationError(f"Normalized output SHA-256 changed: {output_relative}")
        aggregate_bytes += expected_bytes
        if item.get("artifact_role") == "normalized_parquet":
            parquet_count += 1
            parquet_file = pq.ParquetFile(path)
            rows = int(item.get("rows", -1))
            if parquet_file.metadata.num_rows != rows:
                raise NormalizationError(f"Normalized Parquet footer rows changed: {output_relative}")
            if arrow_schema_sha256(parquet_file.schema_arrow) != item.get("arrow_schema_sha256"):
                raise NormalizationError(f"Normalized Parquet Arrow schema changed: {output_relative}")
            aggregate_rows += rows
            names = set(parquet_file.schema_arrow.names)
            prohibited_columns = [
                column
                for column in (
                    "model_label_admitted",
                    "canonical_observation_admitted",
                    "silent_identity_replacement_performed",
                )
                if column in names
            ]
            for batch in parquet_file.iter_batches(columns=prohibited_columns, batch_size=8_192):
                for column in prohibited_columns:
                    if any(bool(value) for value in batch.column(column).to_pylist()):
                        raise NormalizationError(
                            f"Normalized prohibitive flag became true: {output_relative}:{column}"
                        )
    if aggregate_bytes != int(inventory.get("total_bytes", -1)):
        raise NormalizationError("Normalized output aggregate bytes failed")
    if aggregate_rows != int(manifest.get("output_row_count", -1)):
        raise NormalizationError("Normalized output aggregate rows failed")

    input_status = "not_requested"
    verified_inputs: list[dict[str, Any]] = []
    declared_inputs = manifest.get("inputs")
    if not isinstance(declared_inputs, list):
        raise NormalizationError("Normalized input bindings are missing")
    if raw_root is not None:
        raw = Path(raw_root).resolve()
        for record in declared_inputs:
            if not isinstance(record, dict):
                raise NormalizationError("Normalized input binding is invalid")
            relative = _safe_relative_path(record.get("manifest_path"), context="input manifest")
            source_root = raw / Path(*relative.parts[:-1])
            verified = load_and_verify_input(source_root, relative.name)
            verified_record = verified.as_record(raw)
            if verified_record != record:
                raise NormalizationError(f"Normalized input binding drifted: {verified.source_id}")
            verified_inputs.append(verified_record)
        input_set_sha = hashlib.sha256(canonical_json_bytes(verified_inputs)).hexdigest()
        if input_set_sha != manifest.get("input_manifest_set_sha256"):
            raise NormalizationError("Normalized input manifest-set identity failed")
        input_status = "passed_full_recursive_bundle_verification"

    standard_paths = {
        "bindingdb/source_columns.json",
        "bindingdb/article_rows.parquet",
        "bindingdb/affinity_observations.parquet",
        "uniprot/returned_entries.parquet",
        "uniprot/accession_resolution.parquet",
        "uniprot/source_membership.parquet",
        "clinicaltrials/cohort_membership.parquet",
        "regulatory/archive_inventory.parquet",
        "source_inventory.parquet",
    }
    semantic_verification: dict[str, Any] | str
    if set(expected) == standard_paths:
        semantic_verification = _verify_standard_semantics(root, manifest)
    else:
        semantic_verification = "not_applicable_nonstandard_fixture"

    return {
        "status": "passed",
        "schema_version": SCHEMA_VERSION,
        "output_root": _portable_report_path(root),
        "manifest_declared_sha256": manifest["manifest_sha256"],
        "manifest_physical_sha256": sha256_file(manifest_path),
        "manifest_physical_bytes": manifest_path.stat().st_size,
        "inventory_entries": len(entries),
        "parquet_artifacts": parquet_count,
        "aggregate_artifact_bytes": aggregate_bytes,
        "aggregate_parquet_rows": aggregate_rows,
        "input_verification": input_status,
        "verified_input_count": len(verified_inputs),
        "semantic_verification": semantic_verification,
        "zero_label_training_and_identity_replacement_contract": "passed",
    }


def refresh_external_normalization_report(
    output_root: str | os.PathLike[str],
    report_root: str | os.PathLike[str],
    verification: Mapping[str, Any],
) -> dict[str, Any]:
    """Regenerate the normalization report with portable paths after verification."""

    output = Path(output_root).resolve()
    report_path = Path(report_root).resolve() / "external_public_normalization_report.json"
    if not report_path.is_file():
        raise NormalizationError(f"Normalization report does not exist: {report_path}")
    existing_value = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(existing_value, dict) or not verify_document_sha256(existing_value):
        raise NormalizationError("Existing normalization report identity failed")
    existing = {key: value for key, value in existing_value.items() if key != "manifest_sha256"}
    existing["output_root"] = _portable_report_path(output)
    existing["output_manifest_path"] = _portable_report_path(
        output / "external_public_normalized_manifest.json"
    )
    existing["reproduction_command"] = (
        "PYTHONPATH=pipeline/src .venv/bin/python -m "
        "menin_discovery.platform_external_normalization --verify-existing"
    )
    existing["post_promotion_verification"] = dict(verification)
    return _atomic_write_json(report_path, existing)


def build_external_normalization(
    raw_root: str | os.PathLike[str],
    output_root: str | os.PathLike[str],
    report_root: str | os.PathLike[str],
) -> dict[str, Any]:
    """Verify all frozen inputs, normalize evidence, and atomically promote one build."""

    raw = Path(raw_root).resolve()
    output = Path(output_root).resolve()
    reports = Path(report_root).resolve()
    staging = output.with_name(f".{output.name}.building")
    lock = output.parent / f".{output.name}.normalization.lock.json"
    if output.exists():
        raise NormalizationError(f"Normalized output already exists and will not be replaced: {output}")
    if staging.exists():
        raise NormalizationError(f"Staging output already exists; inspect before recovery: {staging}")

    bindings = [load_and_verify_input(raw / source_id, manifest) for source_id, manifest in SOURCE_MANIFESTS]
    binding_by_id = {binding.source_id: binding for binding in bindings}
    input_records = [binding.as_record(raw) for binding in sorted(bindings, key=lambda x: x.source_id)]
    input_set_sha = hashlib.sha256(canonical_json_bytes(input_records)).hexdigest()
    transaction = {
        "schema_version": SCHEMA_VERSION,
        "transaction_owner": f"external-normalization:{input_set_sha}",
        "raw_root": raw.as_posix(),
        "output_root": output.as_posix(),
        "staging_root": staging.as_posix(),
        "input_manifest_set_sha256": input_set_sha,
        "inputs": input_records,
        "recovery_policy": "fail_closed_manual_inspection_required",
    }
    _write_lock(lock, transaction)
    staging.mkdir(parents=True, exist_ok=False)

    binding_artifacts, binding_summary = normalize_bindingdb(binding_by_id[BINDINGDB_SOURCE_ID], staging)
    uniprot_artifacts, uniprot_summary = normalize_uniprot(binding_by_id[UNIPROT_SOURCE_ID], staging)
    clinical_artifacts, clinical_summary = normalize_clinicaltrials(
        binding_by_id[CLINICALTRIALS_SOURCE_ID], staging
    )
    regulatory_artifacts, regulatory_summary = normalize_regulatory_inventories(
        binding_by_id[DRUGSFDA_SOURCE_ID], binding_by_id[DAILYMED_SOURCE_ID], staging
    )
    source_inventory = normalize_source_inventory(bindings, raw, staging)

    parquet_artifacts = [
        *binding_artifacts,
        *uniprot_artifacts,
        *clinical_artifacts,
        *regulatory_artifacts,
        source_inventory,
    ]
    parquet_entries: list[dict[str, Any]] = []
    for artifact in parquet_artifacts:
        record = artifact.as_record()
        record["path"] = Path(artifact.path).relative_to(staging).as_posix()
        parquet_entries.append(record)
    header_entries = binding_summary.pop("extra_output_entries")
    if not isinstance(header_entries, list):
        raise NormalizationError("BindingDB extra output inventory is invalid")
    output_inventory = _inventory_output(staging, [*parquet_entries, *header_entries])

    source_reconciliation = {
        BINDINGDB_SOURCE_ID: binding_summary,
        UNIPROT_SOURCE_ID: uniprot_summary,
        CLINICALTRIALS_SOURCE_ID: clinical_summary,
        DRUGSFDA_SOURCE_ID: {
            key: value for key, value in regulatory_summary.items() if key.startswith("drugs_at_fda")
        },
        DAILYMED_SOURCE_ID: {
            key: value for key, value in regulatory_summary.items() if key.startswith("dailymed")
        },
    }
    manifest_body = {
        "schema_version": SCHEMA_VERSION,
        "parser_version": PARSER_VERSION,
        "dataset_id": "external_public_normalized",
        "determinism_contract": (
            "no wall-clock fields; sorted canonical JSON; fixed schemas, row order, batching, and Parquet options"
        ),
        "transactional_build": True,
        "input_manifest_set_sha256": input_set_sha,
        "inputs": input_records,
        "source_to_output_reconciliation": source_reconciliation,
        "output_inventory": output_inventory,
        "output_row_count": sum(item.rows for item in parquet_artifacts),
        "candidate_evidence_rows": binding_summary["candidate_rows"],
        "canonical_observations_admitted": 0,
        "model_labels_admitted": 0,
        "substantive_model_training_performed": False,
        "zero_training_flag": True,
        "endpoint_pooling_performed": False,
        "silent_cross_source_identity_replacement_performed": False,
        "limitations": {
            "bindingdb": (
                "candidate evidence remains subject to scientific, duplication, rights, assay, structure, "
                "target, and unit review; repeated Reactant_set_id rows are retained"
            ),
            "uniprot": (
                "sequence enrichment is keyed to requested accessions; inactive, ambiguous, and non-UniProt "
                "identifiers remain quarantined; ChEMBL identities are never silently replaced"
            ),
            "clinicaltrials": (
                "broad cohort is registry/status inventory only; cardiac cohort is heuristic and unreviewed; "
                "posted module presence is not a result or molecular label"
            ),
            "drugs_at_fda": (
                "archive index only; one malformed-width row, blank keys, and source orphans remain declared"
            ),
            "dailymed": "archive-only inventory; SPL sections and molecule mappings were not extracted",
        },
    }
    manifest_path = staging / "external_public_normalized_manifest.json"
    output_manifest = _atomic_write_json(manifest_path, manifest_body)

    if not verify_document_sha256(output_manifest):
        raise NormalizationError("Generated output manifest identity failed")
    os.replace(staging, output)
    promoted_manifest_path = output / manifest_path.name
    promoted_physical_sha = sha256_file(promoted_manifest_path)
    report = {
        "schema_version": SCHEMA_VERSION,
        "report_id": "external_public_normalization_report",
        "normalization_status": "passed_and_promoted",
        "output_root": _portable_report_path(output),
        "output_manifest_path": _portable_report_path(promoted_manifest_path),
        "output_manifest_declared_sha256": output_manifest["manifest_sha256"],
        "output_manifest_physical_sha256": promoted_physical_sha,
        "output_manifest_physical_bytes": promoted_manifest_path.stat().st_size,
        "input_manifest_set_sha256": input_set_sha,
        "inputs": input_records,
        "source_to_output_reconciliation": source_reconciliation,
        "output_inventory": output_inventory,
        "quality_gates": {
            "input_internal_manifest_digests": "passed",
            "input_physical_manifest_digests_recorded": "passed",
            "input_exact_recursive_membership": "passed",
            "input_all_bundle_sha256_and_bytes": "passed",
            "input_symlink_and_extra_file_rejection": "passed",
            "bindingdb_row_level_disposition": "passed",
            "bindingdb_all_source_fields_preserved_as_canonical_field_maps": "passed",
            "bindingdb_no_endpoint_pooling": "passed",
            "uniprot_sequence_hash_and_length_verification": "passed",
            "uniprot_no_silent_identity_replacement": "passed",
            "registry_and_regulatory_zero_label_contract": "passed",
            "output_explicit_arrow_schemas_and_fingerprints": "passed",
            "output_exact_recursive_membership": "passed",
            "transactional_atomic_promotion": "passed",
            "substantive_model_training": "not_performed",
        },
        "canonical_observations_admitted": 0,
        "model_labels_admitted": 0,
        "substantive_model_training_performed": False,
        "zero_training_flag": True,
        "reproduction_command": (
            "PYTHONPATH=pipeline/src .venv/bin/python -m "
            "menin_discovery.platform_external_normalization --verify-existing"
        ),
    }
    report_path = reports / "external_public_normalization_report.json"
    report_document = _atomic_write_json(report_path, report)
    lock.unlink()
    return {
        "status": "passed_and_promoted",
        "output_root": output.as_posix(),
        "manifest": output_manifest,
        "manifest_physical_sha256": promoted_physical_sha,
        "report_path": report_path.as_posix(),
        "report_manifest_sha256": report_document["manifest_sha256"],
        "report_physical_sha256": sha256_file(report_path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-root",
        default="research/data/platform/raw/external_public",
        help="Frozen external-public acquisition bundle root",
    )
    parser.add_argument(
        "--output-root",
        default="research/data/platform/interim/external_public_normalized",
        help="New normalized output root (must not already exist)",
    )
    parser.add_argument(
        "--report-root",
        default="research/reports/platform/external_normalization",
        help="Normalization report directory",
    )
    parser.add_argument(
        "--verify-existing",
        action="store_true",
        help="Verify the promoted output and all frozen input bindings without rebuilding",
    )
    parser.add_argument(
        "--refresh-report",
        action="store_true",
        help="After successful existing-output verification, regenerate the portable report",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.refresh_report and not args.verify_existing:
        raise NormalizationError("--refresh-report requires --verify-existing")
    if args.verify_existing:
        result = verify_external_normalized_output(args.output_root, args.raw_root)
        if args.refresh_report:
            report = refresh_external_normalization_report(args.output_root, args.report_root, result)
            result["refreshed_report_manifest_sha256"] = report["manifest_sha256"]
    else:
        result = build_external_normalization(args.raw_root, args.output_root, args.report_root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
