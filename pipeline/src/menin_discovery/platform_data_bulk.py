"""Deterministic ChEMBL_37 bulk archive verification and SQLite extraction.

This module executes the full official ChEMBL_37 SQLite acquisition path for
the platform readiness build.  The archive is 5.76 GB and expands
substantially, so every expensive operation is resumable and explicit.  It
exports source activity assertions, assay, molecule-identity, target, and
document fields. Exported rows are not model-eligible until canonical
observation-kind and QC gates accept them. ChEMBL's ``compound_properties``
table and source-supplied derived ``pchembl_value`` are omitted, preventing
calculated descriptors or standardized labels from being mistaken for raw
experimental evidence.

The exporter uses keyset pagination over ``activity_id`` and checkpointed,
content-hashed Parquet parts.  Interrupted exports resume without rewriting a
completed immutable part.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import socket
import sqlite3
import tarfile
import tempfile
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests

from .http import build_session, get_response
from .platform_data_schema import (
    SCHEMA_VERSION,
    arrow_schema_contract,
    canonical_json,
    require_arrow_schema_contract,
)
from .platform_data_sources import sha256_file

CHEMBL37_SQLITE_URL = (
    "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_37/chembl_37_sqlite.tar.gz"
)
CHEMBL37_ARCHIVE_BYTES = 5_764_252_857
CHEMBL37_ARCHIVE_SHA256 = "33c203740555f96067710cdfc1c3c55d890660e5908ec5cbf5817492c290d281"
CHEMBL37_CITATION = (
    "Mendez et al. ChEMBL: towards direct deposition of bioassay data. "
    "Nucleic Acids Research 2019; DOI 10.1093/nar/gky1075."
)
CHEMBL37_RELEASE_ROOT = "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_37"
CHEMBL37_RELEASE_METADATA_FILES = (
    "LICENSE",
    "README",
    "REQUIRED.ATTRIBUTION",
    "checksums.txt",
    "chembl_37_release_notes.txt",
    "chembl_37_schema.pdf",
    "chembl_uniprot_mapping.txt",
    "schema_documentation.html",
    "schema_documentation.txt",
)

_ARROW_STRING = pa.large_string()
_ARROW_INTEGER = pa.int64()
_ARROW_FLOAT = pa.float64()
ACTIVITY_ARROW_SCHEMA = pa.schema(
    [
        ("activity_id", _ARROW_INTEGER),
        ("action_type", _ARROW_STRING),
        ("activity_comment", _ARROW_STRING),
        ("bao_endpoint", _ARROW_STRING),
        ("data_validity_comment", _ARROW_STRING),
        ("potential_duplicate", _ARROW_INTEGER),
        ("qudt_units", _ARROW_STRING),
        ("record_id", _ARROW_INTEGER),
        ("relation", _ARROW_STRING),
        ("src_id", _ARROW_INTEGER),
        ("standard_flag", _ARROW_INTEGER),
        ("standard_relation", _ARROW_STRING),
        ("standard_text_value", _ARROW_STRING),
        ("standard_type", _ARROW_STRING),
        ("standard_units", _ARROW_STRING),
        ("standard_upper_value", _ARROW_FLOAT),
        ("standard_value", _ARROW_FLOAT),
        ("text_value", _ARROW_STRING),
        ("toid", _ARROW_INTEGER),
        ("type", _ARROW_STRING),
        ("units", _ARROW_STRING),
        ("uo_units", _ARROW_STRING),
        ("upper_value", _ARROW_FLOAT),
        ("value", _ARROW_FLOAT),
        ("modality", _ARROW_STRING),
        ("assay_chembl_id", _ARROW_STRING),
        ("assay_description", _ARROW_STRING),
        ("assay_type", _ARROW_STRING),
        ("assay_test_type", _ARROW_STRING),
        ("assay_category", _ARROW_STRING),
        ("assay_organism", _ARROW_STRING),
        ("assay_tax_id", _ARROW_INTEGER),
        ("assay_strain", _ARROW_STRING),
        ("assay_tissue", _ARROW_STRING),
        ("assay_cell_type", _ARROW_STRING),
        ("assay_subcellular_fraction", _ARROW_STRING),
        ("relationship_type", _ARROW_STRING),
        ("src_assay_id", _ARROW_STRING),
        ("cell_id", _ARROW_INTEGER),
        ("tissue_id", _ARROW_INTEGER),
        ("variant_id", _ARROW_INTEGER),
        ("assay_group", _ARROW_STRING),
        ("bao_format", _ARROW_STRING),
        ("confidence_score", _ARROW_INTEGER),
        ("target_chembl_id", _ARROW_STRING),
        ("target_pref_name", _ARROW_STRING),
        ("target_type", _ARROW_STRING),
        ("target_organism", _ARROW_STRING),
        ("target_tax_id", _ARROW_INTEGER),
        ("molecule_chembl_id", _ARROW_STRING),
        ("molecule_pref_name", _ARROW_STRING),
        ("canonical_smiles", _ARROW_STRING),
        ("standard_inchi_key", _ARROW_STRING),
        ("document_chembl_id", _ARROW_STRING),
        ("document_journal", _ARROW_STRING),
        ("document_year", _ARROW_INTEGER),
        ("document_doi", _ARROW_STRING),
        ("pubmed_id", _ARROW_INTEGER),
        ("patent_id", _ARROW_STRING),
        ("document_title", _ARROW_STRING),
        ("document_type", _ARROW_STRING),
        ("document_chembl_release_id", _ARROW_INTEGER),
        ("activity_source_name", _ARROW_STRING),
        ("activity_source_description", _ARROW_STRING),
        ("component_accessions", _ARROW_STRING),
        ("component_sequences", _ARROW_STRING),
        ("component_types", _ARROW_STRING),
    ]
)
DEVELOPMENT_ARROW_SCHEMA = pa.schema(
    [
        ("molecule_row_id", _ARROW_INTEGER),
        ("molecule_chembl_id", _ARROW_STRING),
        ("molecule_pref_name", _ARROW_STRING),
        ("molecule_type", _ARROW_STRING),
        ("max_phase", _ARROW_FLOAT),
        ("first_approval", _ARROW_INTEGER),
        ("withdrawn_flag", _ARROW_INTEGER),
        ("black_box_warning", _ARROW_INTEGER),
        ("therapeutic_flag", _ARROW_INTEGER),
        ("annotation_role", _ARROW_STRING),
    ]
)
TARGET_COMPONENT_ARROW_SCHEMA = pa.schema(
    [
        ("target_chembl_id", _ARROW_STRING),
        ("target_name", _ARROW_STRING),
        ("target_type", _ARROW_STRING),
        ("target_organism", _ARROW_STRING),
        ("target_tax_id", _ARROW_INTEGER),
        ("component_id", _ARROW_INTEGER),
        ("homologue", _ARROW_INTEGER),
        ("accession", _ARROW_STRING),
        ("component_type", _ARROW_STRING),
        ("sequence", _ARROW_STRING),
        ("sequence_md5sum", _ARROW_STRING),
        ("component_organism", _ARROW_STRING),
        ("component_tax_id", _ARROW_INTEGER),
    ]
)
_SCHEMA_NORMALIZATION_MARKER = "schema_normalization_transaction.json"


def _assert_schema_normalization_idle(bulk_root: Path) -> None:
    marker = bulk_root / _SCHEMA_NORMALIZATION_MARKER
    if marker.exists():
        raise RuntimeError(
            f"A fail-closed Arrow schema normalization transaction requires inspection: {marker}"
        )


@dataclass(frozen=True)
class ArchivePart:
    """One verified inclusive HTTP byte range of the release archive."""

    path: Path
    start_byte: int
    end_byte: int
    sha256: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _atomic_write_json(path: Path, document: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(document, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json_byte_snapshot(path: Path, *, context: str) -> tuple[dict[str, Any], str, int]:
    """Parse and hash one byte buffer, then reject any concurrent file drift."""

    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    document = json.loads(payload.decode("utf-8"))
    if not isinstance(document, dict) or path.stat().st_size != len(payload) or sha256_file(path) != digest:
        raise RuntimeError(f"{context} changed while read: {path}")
    return document, digest, len(payload)


def _coerce_arrow_table(
    table: pa.Table,
    schema: pa.Schema,
    *,
    allowed_missing: frozenset[str] = frozenset(),
) -> pa.Table:
    expected_names = list(schema.names)
    unexpected = sorted(set(table.column_names) - set(expected_names))
    missing = sorted(set(expected_names) - set(table.column_names))
    if unexpected or set(missing) - allowed_missing:
        raise RuntimeError(f"Arrow field-set mismatch; missing={missing}; unexpected={unexpected}")
    for name in missing:
        table = table.append_column(
            pa.field(name, schema.field(name).type),
            pa.nulls(table.num_rows, type=schema.field(name).type),
        )
    table = table.select(expected_names).cast(schema, safe=True)
    table = table.replace_schema_metadata(None)
    if not table.schema.equals(schema, check_metadata=True):
        raise RuntimeError("Arrow coercion did not produce the declared physical schema")
    return table


def _write_frame_with_schema(frame: pd.DataFrame, destination: Path, schema: pa.Schema) -> None:
    table = pa.Table.from_pandas(frame, preserve_index=False)
    table = _coerce_arrow_table(table, schema)
    temporary = destination.with_suffix(".parquet.part")
    pq.write_table(table, temporary, compression="zstd")
    os.replace(temporary, destination)


def _verify_manifest_arrow_schema(
    manifest: dict[str, Any],
    root: Path,
    expected_schema: pa.Schema,
    *,
    context: str,
) -> None:
    contract = require_arrow_schema_contract(
        manifest,
        expected_schema,
        context=context,
    )
    expected_fingerprint = contract["sha256"]
    for part in manifest.get("parts", []):
        path = (root / str(part["path"])).resolve()
        if not path.is_file():
            raise RuntimeError(f"Missing Parquet part while validating Arrow schema: {path}")
        parquet_file = pq.ParquetFile(path)
        metadata = parquet_file.metadata
        physical_schema = parquet_file.schema_arrow.remove_metadata()
        if (
            metadata is None
            or int(metadata.num_rows) != int(part.get("rows", -1))
            or path.stat().st_size != int(part.get("size_bytes", -1))
            or sha256_file(path) != part.get("sha256")
            or part.get("arrow_schema_sha256") != expected_fingerprint
            or not physical_schema.equals(expected_schema, check_metadata=True)
        ):
            raise RuntimeError(f"Parquet part violates its declared Arrow schema: {path}")


def _verify_schema_normalization_receipt(
    manifest: dict[str, Any],
    bulk_root: Path,
) -> None:
    """Verify the complete normalization ledger against every current export byte.

    A receipt is intentionally an all-or-nothing corpus contract.  Verifying only
    the manifest passed by the caller would allow a self-consistent subset to hide
    an omitted, duplicated, or subsequently changed export part.
    """

    binding = manifest.get("schema_normalization_receipt")
    if binding is None:
        return
    if not isinstance(binding, dict):
        raise RuntimeError("Malformed schema-normalization receipt binding")
    receipt_path = (bulk_root / str(binding.get("path", ""))).resolve()
    try:
        receipt_path.relative_to(bulk_root.resolve())
    except ValueError as error:
        raise RuntimeError("Schema-normalization receipt escapes the bulk root") from error
    if (
        not receipt_path.is_file()
        or sha256_file(receipt_path) != binding.get("sha256")
        or receipt_path.stat().st_size != int(binding.get("size_bytes", -1))
    ):
        raise RuntimeError("Schema-normalization receipt binding failed verification")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if not isinstance(receipt, dict) or receipt.get("status") != "complete":
        raise RuntimeError("Schema-normalization receipt is not a completed ledger")

    full_manifest_path = bulk_root / "activity_export_manifest.json"
    summary_path = bulk_root / "specialized_views" / "specialized_views_manifest.json"
    if not full_manifest_path.is_file() or not summary_path.is_file():
        raise RuntimeError("Schema-normalization receipt lacks its complete manifest set")
    full_manifest = json.loads(full_manifest_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    views = summary.get("views")
    expected_view_names = {
        "single_protein_kd_ki",
        "single_protein_ic50_ec50_candidates",
        "herg_all_endpoints",
        "pk_adme_candidates",
        "cardiac_qt_apd_inventory",
        "molecule_development_annotations",
        "target_components",
    }
    if not isinstance(views, dict) or set(views) != expected_view_names:
        raise RuntimeError("Schema-normalization receipt has an incomplete child-manifest set")

    bound_documents: list[dict[str, Any]] = [full_manifest, summary]
    child_manifests: dict[str, dict[str, Any]] = {}
    specialized_root = bulk_root / "specialized_views"
    for view_name in sorted(expected_view_names):
        child_path = specialized_root / f"{view_name}_manifest.json"
        if not child_path.is_file():
            raise RuntimeError(f"Missing receipt-bound child manifest: {child_path}")
        child = json.loads(child_path.read_text(encoding="utf-8"))
        if child != views[view_name]:
            raise RuntimeError(f"Receipt-bound summary/child manifest drift: {view_name}")
        child_manifests[view_name] = child
        bound_documents.append(child)
    if any(document.get("schema_normalization_receipt") != binding for document in bound_documents):
        raise RuntimeError("Schema-normalization receipt binding differs across manifests")
    source_identity = receipt.get("source_identity")
    manifested_database_sha256 = str(full_manifest.get("database_sha256", ""))
    if (
        not isinstance(source_identity, dict)
        or source_identity.get("source_version") != "ChEMBL_37"
        or source_identity.get("database_sha256") != manifested_database_sha256
        or any(
            document.get("source_version") != "ChEMBL_37"
            or document.get("database_sha256") != manifested_database_sha256
            for document in bound_documents
        )
        or re.fullmatch(
            r"[0-9a-f]{64}",
            str(source_identity.get("pre_normalization_activity_manifest_sha256", "")),
        )
        is None
        or re.fullmatch(
            r"[0-9a-f]{64}",
            str(source_identity.get("pre_normalization_specialized_summary_sha256", "")),
        )
        is None
    ):
        raise RuntimeError("Schema-normalization receipt source identity mismatch")

    activity_contract = arrow_schema_contract(ACTIVITY_ARROW_SCHEMA)
    development_contract = arrow_schema_contract(DEVELOPMENT_ARROW_SCHEMA)
    target_contract = arrow_schema_contract(TARGET_COMPONENT_ARROW_SCHEMA)
    if receipt.get("arrow_schema_contracts") != {
        "activity": activity_contract,
        "molecule_development_annotations": development_contract,
        "target_components": target_contract,
    }:
        raise RuntimeError("Schema-normalization receipt has unexpected Arrow contracts")

    expected: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}

    def add_partitioned(
        source_manifest: dict[str, Any],
        *,
        path_prefix: str,
        schema_contract: dict[str, Any],
    ) -> None:
        if source_manifest.get("arrow_schema") != schema_contract:
            raise RuntimeError("Receipt-bound manifest Arrow contract drift")
        parts = source_manifest.get("parts")
        if not isinstance(parts, list) or int(source_manifest.get("part_count", -1)) != len(parts):
            raise RuntimeError("Receipt-bound manifest part-count drift")
        for part in parts:
            if not isinstance(part, dict):
                raise RuntimeError("Malformed receipt-bound part record")
            relative = f"{path_prefix}{part.get('path', '')}"
            if relative in expected:
                raise RuntimeError(f"Duplicate expected normalization path: {relative}")
            expected[relative] = (part, schema_contract)

    add_partitioned(full_manifest, path_prefix="", schema_contract=activity_contract)
    for view_name in sorted(expected_view_names - {"target_components"}):
        child = child_manifests[view_name]
        contract = (
            development_contract if view_name == "molecule_development_annotations" else activity_contract
        )
        add_partitioned(
            child,
            path_prefix="specialized_views/",
            schema_contract=contract,
        )
    target_manifest = child_manifests["target_components"]
    if target_manifest.get("arrow_schema") != target_contract:
        raise RuntimeError("Target-component Arrow contract drift")
    target_relative = f"specialized_views/{target_manifest.get('path', '')}"
    if target_manifest.get("path") != "target_components.parquet" or target_relative in expected:
        raise RuntimeError("Target-component receipt path is noncanonical or duplicated")
    expected[target_relative] = (target_manifest, target_contract)

    actual_paths = {
        path.relative_to(bulk_root).as_posix()
        for path in bulk_root.rglob("*.parquet")
        if path.is_file()
        and (
            not (bulk_root / _SCHEMA_NORMALIZATION_MARKER).exists()
            or ".schema_normalization_staging" not in path.relative_to(bulk_root).parts
        )
    }
    if actual_paths != set(expected):
        missing = sorted(set(expected) - actual_paths)
        extra = sorted(actual_paths - set(expected))
        raise RuntimeError(
            f"Normalized physical Parquet inventory mismatch; missing={missing}; extra={extra}"
        )

    files = receipt.get("files")
    if not isinstance(files, list):
        raise RuntimeError("Schema-normalization receipt files must be a list")
    by_path: dict[str, dict[str, Any]] = {}
    for record in files:
        if not isinstance(record, dict):
            raise RuntimeError("Malformed schema-normalization file receipt")
        relative = str(record.get("path", ""))
        if not relative or relative in by_path:
            raise RuntimeError(f"Duplicate/blank normalization receipt path: {relative!r}")
        by_path[relative] = record
    if set(by_path) != set(expected):
        missing = sorted(set(expected) - set(by_path))
        extra = sorted(set(by_path) - set(expected))
        raise RuntimeError(
            f"Schema-normalization receipt membership mismatch; missing={missing}; extra={extra}"
        )

    row_total = 0
    old_size_total = 0
    new_size_total = 0
    for relative, (part, contract) in expected.items():
        record = by_path[relative]
        physical = (bulk_root / relative).resolve()
        try:
            physical.relative_to(bulk_root.resolve())
        except ValueError as error:
            raise RuntimeError(f"Normalization receipt path escapes bulk root: {relative}") from error
        if not physical.is_file():
            raise RuntimeError(f"Missing normalized Parquet file: {physical}")
        parquet = pq.ParquetFile(physical)
        metadata = parquet.metadata
        physical_schema = parquet.schema_arrow.remove_metadata()
        proof = record.get("value_preservation")
        if not isinstance(proof, dict):
            raise RuntimeError(f"Missing value-preservation proof: {relative}")
        expected_rows = int(part.get("rows", part.get("row_count", -1)))
        expected_size = int(part.get("size_bytes", -1))
        expected_sha256 = str(part.get("sha256", ""))
        expected_schema_sha256 = str(contract["sha256"])
        physical_null_cells = sum(column.null_count for column in pq.read_table(physical).columns)
        if (
            metadata is None
            or int(metadata.num_rows) != expected_rows
            or physical.stat().st_size != expected_size
            or sha256_file(physical) != expected_sha256
            or not physical_schema.equals(
                ACTIVITY_ARROW_SCHEMA
                if contract == activity_contract
                else DEVELOPMENT_ARROW_SCHEMA
                if contract == development_contract
                else TARGET_COMPONENT_ARROW_SCHEMA,
                check_metadata=True,
            )
            or part.get("arrow_schema_sha256") != expected_schema_sha256
            or int(record.get("rows", -1)) != expected_rows
            or int(record.get("new_size_bytes", -1)) != expected_size
            or record.get("new_sha256") != expected_sha256
            or record.get("declared_arrow_schema_sha256") != expected_schema_sha256
            or re.fullmatch(r"[0-9a-f]{64}", str(record.get("old_sha256", ""))) is None
            or re.fullmatch(r"[0-9a-f]{64}", str(record.get("old_arrow_schema_sha256", ""))) is None
            or not bool(proof.get("coerced_old_equals_new"))
            or int(proof.get("old_coerced_null_cells", -1)) != int(proof.get("new_null_cells", -2))
            or int(proof.get("new_null_cells", -1)) != physical_null_cells
        ):
            raise RuntimeError(f"Normalization receipt/manifest/physical mismatch: {relative}")
        row_total += expected_rows
        old_size_total += int(record.get("old_size_bytes", -1))
        new_size_total += expected_size

    aggregate = receipt.get("aggregate")
    if not isinstance(aggregate, dict) or (
        int(aggregate.get("parquet_files", -1)) != len(expected)
        or int(binding.get("parquet_files", -2)) != len(expected)
        or int(aggregate.get("rows_across_overlapping_exports", -1)) != row_total
        or int(aggregate.get("old_size_bytes", -1)) != old_size_total
        or int(aggregate.get("new_size_bytes", -1)) != new_size_total
        or not bool(aggregate.get("all_value_preservation_checks_passed"))
    ):
        raise RuntimeError("Schema-normalization receipt aggregate does not reconcile")


def _fsync_directory(directory: Path) -> None:
    """Make a completed rename/unlink durable before advancing the journal."""

    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _schema_transaction_entry(
    bulk_root: Path,
    staged_path: Path,
    destination_path: Path,
    *,
    kind: str,
    expected_old_sha256: str | None,
    expected_old_size_bytes: int | None,
) -> dict[str, Any]:
    resolved_root = bulk_root.resolve()
    staged = staged_path.resolve()
    destination = destination_path.resolve()
    try:
        staged_relative = staged.relative_to(resolved_root).as_posix()
        destination_relative = destination.relative_to(resolved_root).as_posix()
    except ValueError as error:
        raise RuntimeError("Schema-normalization transaction path escapes the bulk root") from error
    if not staged.is_file():
        raise RuntimeError(f"Missing staged transaction payload: {staged}")
    with staged.open("rb") as handle:
        os.fsync(handle.fileno())
    _fsync_directory(staged.parent)
    old_sha256 = sha256_file(destination) if destination.is_file() else None
    old_size_bytes = destination.stat().st_size if destination.is_file() else None
    if (old_sha256, old_size_bytes) != (
        expected_old_sha256,
        expected_old_size_bytes,
    ):
        raise RuntimeError(f"Schema-normalization destination drifted after staging: {destination}")
    return {
        "kind": kind,
        "destination_path": destination_relative,
        "staged_path": staged_relative,
        "old_sha256": expected_old_sha256,
        "old_size_bytes": expected_old_size_bytes,
        "new_sha256": sha256_file(staged),
        "new_size_bytes": staged.stat().st_size,
    }


def _resolve_schema_transaction_path(bulk_root: Path, relative: object) -> Path:
    text = str(relative)
    if not text or Path(text).is_absolute():
        raise RuntimeError(f"Invalid schema-normalization transaction path: {text!r}")
    path = (bulk_root / text).resolve()
    try:
        path.relative_to(bulk_root.resolve())
    except ValueError as error:
        raise RuntimeError(f"Schema-normalization transaction path escapes root: {text}") from error
    return path


def _resume_schema_normalization_transaction(
    bulk_root: Path,
    *,
    fail_after_replacements: int | None = None,
    validate_corpus: bool = True,
) -> dict[str, Any]:
    """Idempotently replay a durable old/new-hash journal and validate before cleanup.

    ``fail_after_replacements`` and ``validate_corpus=False`` exist solely for
    deterministic interruption tests of the journal primitive.
    """

    marker = bulk_root / _SCHEMA_NORMALIZATION_MARKER
    if not marker.is_file():
        raise RuntimeError(f"Missing schema-normalization transaction marker: {marker}")
    journal = json.loads(marker.read_text(encoding="utf-8"))
    entries = journal.get("entries")
    owner = journal.get("owner")
    try:
        uuid.UUID(str(journal.get("transaction_id", "")))
    except ValueError as error:
        raise RuntimeError("Schema-normalization transaction ID is malformed") from error
    if not isinstance(owner, dict) or not str(owner.get("hostname", "")) or int(owner.get("pid", -1)) <= 0:
        raise RuntimeError("Schema-normalization transaction owner is malformed")
    if journal.get("status") != "committing" or not isinstance(entries, list) or not entries:
        raise RuntimeError("Schema-normalization transaction is not durably commit-ready")
    if int(journal.get("entry_count", -1)) != len(entries) or int(
        journal.get("staged_part_count", -1)
    ) != sum(isinstance(entry, dict) and entry.get("kind") == "parquet" for entry in entries):
        raise RuntimeError("Schema-normalization transaction count contract mismatch")
    if hashlib.sha256(canonical_json(entries).encode("utf-8")).hexdigest() != journal.get("entries_sha256"):
        raise RuntimeError("Schema-normalization transaction journal digest mismatch")
    destinations: set[Path] = set()
    staged_paths: set[Path] = set()
    for entry in entries:
        if not isinstance(entry, dict) or entry.get("kind") not in {
            "parquet",
            "receipt",
            "manifest",
            "summary",
        }:
            raise RuntimeError("Malformed schema-normalization transaction entry")
        destination = _resolve_schema_transaction_path(bulk_root, entry.get("destination_path"))
        staged = _resolve_schema_transaction_path(bulk_root, entry.get("staged_path"))
        expected_staged = (
            bulk_root / ".schema_normalization_staging" / str(entry.get("destination_path", ""))
        ).resolve()
        if staged != expected_staged:
            raise RuntimeError("Schema-normalization staged/destination path mapping mismatch")
        if destination in destinations or staged in staged_paths:
            raise RuntimeError("Duplicate path in schema-normalization transaction journal")
        destinations.add(destination)
        staged_paths.add(staged)
    staging = _resolve_schema_transaction_path(bulk_root, journal.get("staging_directory", ""))
    if staging != (bulk_root / ".schema_normalization_staging").resolve():
        raise RuntimeError("Unexpected schema-normalization staging directory")

    replacements = 0
    for entry in entries:
        destination = _resolve_schema_transaction_path(bulk_root, entry["destination_path"])
        staged = _resolve_schema_transaction_path(bulk_root, entry["staged_path"])
        new_sha256 = str(entry.get("new_sha256", ""))
        new_size_bytes = int(entry.get("new_size_bytes", -1))
        old_sha256 = entry.get("old_sha256")
        old_size_bytes = entry.get("old_size_bytes")
        if (
            re.fullmatch(r"[0-9a-f]{64}", new_sha256) is None
            or new_size_bytes < 0
            or (old_sha256 is not None and re.fullmatch(r"[0-9a-f]{64}", str(old_sha256)) is None)
        ):
            raise RuntimeError("Malformed old/new state in schema-normalization journal")
        destination_is_new = (
            destination.is_file()
            and destination.stat().st_size == new_size_bytes
            and sha256_file(destination) == new_sha256
        )
        if destination_is_new:
            continue
        destination_is_old = (
            not destination.exists()
            if old_sha256 is None
            else destination.is_file()
            and destination.stat().st_size == int(old_size_bytes)
            and sha256_file(destination) == old_sha256
        )
        if not destination_is_old:
            raise RuntimeError(
                f"Transaction destination matches neither journaled old nor new state: {destination}"
            )
        if (
            not staged.is_file()
            or staged.stat().st_size != new_size_bytes
            or sha256_file(staged) != new_sha256
        ):
            raise RuntimeError(f"Staged transaction payload failed verification: {staged}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staged, destination)
        _fsync_directory(destination.parent)
        replacements += 1
        if fail_after_replacements is not None and replacements >= fail_after_replacements:
            raise RuntimeError("Injected schema-normalization commit interruption")

    for entry in entries:
        destination = _resolve_schema_transaction_path(bulk_root, entry["destination_path"])
        if (
            not destination.is_file()
            or destination.stat().st_size != int(entry["new_size_bytes"])
            or sha256_file(destination) != entry["new_sha256"]
        ):
            raise RuntimeError(f"Committed transaction payload failed verification: {destination}")

    if validate_corpus:
        activity_manifest_path = bulk_root / "activity_export_manifest.json"
        summary_path = bulk_root / "specialized_views" / "specialized_views_manifest.json"
        activity_manifest = json.loads(activity_manifest_path.read_text(encoding="utf-8"))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        views = summary.get("views")
        if not isinstance(views, dict):
            raise RuntimeError("Committed specialized summary is malformed")
        _verify_manifest_arrow_schema(
            activity_manifest,
            bulk_root,
            ACTIVITY_ARROW_SCHEMA,
            context="activity_facts",
        )
        activity_views = (
            "single_protein_kd_ki",
            "single_protein_ic50_ec50_candidates",
            "herg_all_endpoints",
            "pk_adme_candidates",
            "cardiac_qt_apd_inventory",
        )
        specialized_root = bulk_root / "specialized_views"
        for view_name in activity_views:
            _verify_manifest_arrow_schema(
                views[view_name],
                specialized_root,
                ACTIVITY_ARROW_SCHEMA,
                context=view_name,
            )
        _verify_manifest_arrow_schema(
            views["molecule_development_annotations"],
            specialized_root,
            DEVELOPMENT_ARROW_SCHEMA,
            context="molecule_development_annotations",
        )
        target = views["target_components"]
        target_path = specialized_root / str(target.get("path", ""))
        target_metadata = pq.ParquetFile(target_path).metadata if target_path.is_file() else None
        if (
            target_metadata is None
            or int(target_metadata.num_rows) != int(target.get("row_count", -1))
            or not pq.ParquetFile(target_path)
            .schema_arrow.remove_metadata()
            .equals(TARGET_COMPONENT_ARROW_SCHEMA, check_metadata=True)
        ):
            raise RuntimeError("Committed target-component schema verification failed")
        overlap_pairs = {
            "kd_ki_and_herg": ("single_protein_kd_ki", "herg_all_endpoints"),
            "ic50_ec50_and_herg": (
                "single_protein_ic50_ec50_candidates",
                "herg_all_endpoints",
            ),
            "pk_adme_and_herg": ("pk_adme_candidates", "herg_all_endpoints"),
            "cardiac_qt_apd_and_herg": (
                "cardiac_qt_apd_inventory",
                "herg_all_endpoints",
            ),
        }
        exact_overlaps = {
            name: exact_manifest_activity_overlap(views[left], views[right], specialized_root)
            for name, (left, right) in overlap_pairs.items()
        }
        if summary.get("overlap_row_counts") != exact_overlaps:
            raise RuntimeError("Committed overlap counts changed during schema normalization")
        _verify_schema_normalization_receipt(summary, bulk_root)
        receipt_binding = summary.get("schema_normalization_receipt", {})
        receipt_document = json.loads(
            (bulk_root / str(receipt_binding.get("path", ""))).read_text(encoding="utf-8")
        )
        if receipt_document.get("source_identity") != journal.get("source_identity"):
            raise RuntimeError("Committed receipt/journal source identity mismatch")

    if staging.exists():
        shutil.rmtree(staging)
        _fsync_directory(bulk_root)
    marker.unlink()
    _fsync_directory(bulk_root)
    if not validate_corpus:
        return {"status": "complete", "replacements": len(entries)}
    activity_manifest = json.loads((bulk_root / "activity_export_manifest.json").read_text(encoding="utf-8"))
    summary = json.loads(
        (bulk_root / "specialized_views" / "specialized_views_manifest.json").read_text(encoding="utf-8")
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "normalized_at_utc": _utc_now(),
        "activity_export_manifest": activity_manifest,
        "specialized_views_manifest": summary,
        "normalized_part_count": int(summary["schema_normalization_receipt"]["parquet_files"]),
        "schema_normalization_receipt": summary["schema_normalization_receipt"],
        "arrow_schema_contracts": summary["arrow_schema_contracts"],
    }


def _schema_normalization_source_identity(
    database: Path,
    bulk_root: Path,
) -> dict[str, Any]:
    """Bind the actual SQLite bytes to one coherent manifested ChEMBL release."""

    activity_path = bulk_root / "activity_export_manifest.json"
    summary_path = bulk_root / "specialized_views" / "specialized_views_manifest.json"
    if not activity_path.is_file() or not summary_path.is_file():
        raise RuntimeError("Schema normalization requires the complete export manifest set")

    activity, activity_digest, _ = _read_json_byte_snapshot(
        activity_path,
        context="Source-identity manifest",
    )
    summary, summary_digest, _ = _read_json_byte_snapshot(
        summary_path,
        context="Source-identity manifest",
    )
    views = summary.get("views")
    if not isinstance(views, dict):
        raise RuntimeError("Specialized summary lacks child manifests")
    specialized_root = summary_path.parent
    documents = [activity, summary]
    for view_name, embedded in sorted(views.items()):
        child_path = specialized_root / f"{view_name}_manifest.json"
        if not child_path.is_file():
            raise RuntimeError(f"Missing specialized child manifest: {child_path}")
        child, _, _ = _read_json_byte_snapshot(
            child_path,
            context="Source-identity child manifest",
        )
        if child != embedded:
            raise RuntimeError(f"Specialized summary/child drift before normalization: {view_name}")
        documents.append(child)
    database_digests = {str(document.get("database_sha256", "")) for document in documents}
    source_versions = {str(document.get("source_version", "")) for document in documents}
    schema_versions = {str(document.get("schema_version", "")) for document in documents}
    actual_database_sha256 = sha256_file(database)
    if (
        database_digests != {actual_database_sha256}
        or source_versions != {"ChEMBL_37"}
        or schema_versions != {SCHEMA_VERSION}
    ):
        raise RuntimeError("SQLite bytes and export manifests do not share one ChEMBL_37 source identity")
    return {
        "source_version": "ChEMBL_37",
        "database_sha256": actual_database_sha256,
        "pre_normalization_activity_manifest_sha256": activity_digest,
        "pre_normalization_specialized_summary_sha256": summary_digest,
    }


def _verify_schema_normalization_prestate_identity(
    source_identity: dict[str, Any],
    expected_old_states: dict[Path, tuple[str | None, int | None]],
    activity_manifest_path: Path,
    summary_path: Path,
) -> None:
    """Require the staged manifest bytes to be the exact pre-state cited by the receipt."""

    checks = (
        (
            activity_manifest_path.resolve(),
            "pre_normalization_activity_manifest_sha256",
        ),
        (
            summary_path.resolve(),
            "pre_normalization_specialized_summary_sha256",
        ),
    )
    for path, identity_field in checks:
        bound = expected_old_states.get(path)
        if bound is None or bound[0] != source_identity.get(identity_field):
            raise RuntimeError(f"Schema-normalization source identity/pre-state drift: {path}")


def snapshot_chembl37_release_metadata(
    raw_root: str | os.PathLike[str],
    *,
    session: requests.Session | None = None,
) -> dict[str, Any]:
    """Preserve official rights, checksum, release-note, mapping, and schema bytes."""

    destination = Path(raw_root).resolve() / "chembl_37_bulk" / "release_metadata"
    destination.mkdir(parents=True, exist_ok=True)
    client = session or build_session(backoff_factor=0.8)
    manifest_path = destination / "release_metadata_manifest.json"
    previous: dict[str, Any] = {}
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
    previous_files = {str(record.get("filename")): record for record in previous.get("files", [])}
    captured_at = str(previous.get("captured_at_utc", "")) or _utc_now()
    records: list[dict[str, Any]] = []
    for filename in CHEMBL37_RELEASE_METADATA_FILES:
        url = f"{CHEMBL37_RELEASE_ROOT}/{filename}"
        path = destination / filename
        headers: dict[str, str] = {}
        resolved_url = url
        prior = previous_files.get(filename, {})
        retrieved_at = str(prior.get("retrieved_at_utc", "")) or captured_at
        if not path.exists():
            response = get_response(url, timeout=(10, 120), session=client)
            content = response.content
            resolved_url = str(response.url)
            headers = {
                name.casefold(): value
                for name, value in response.headers.items()
                if name.casefold() in {"etag", "last-modified", "content-length", "content-type"}
            }
            retrieved_at = _utc_now()
            descriptor, temporary_name = tempfile.mkstemp(prefix=f".{filename}.", dir=destination)
            temporary = Path(temporary_name)
            try:
                with os.fdopen(descriptor, "wb") as handle:
                    handle.write(content)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)
        else:
            resolved_url = str(prior.get("resolved_url", url))
            headers = {
                str(name): str(value) for name, value in dict(prior.get("response_headers", {})).items()
            }
        records.append(
            {
                "filename": filename,
                "immutable_url": url,
                "resolved_url": resolved_url,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "retrieved_at_utc": retrieved_at,
                "response_headers": headers,
            }
        )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "source_name": "ChEMBL",
        "source_version": "ChEMBL_37",
        "release_root": CHEMBL37_RELEASE_ROOT,
        "captured_at_utc": captured_at,
        "snapshot_status": "complete",
        "files": records,
    }
    _atomic_write_json(manifest_path, manifest)
    return manifest


def assemble_archive_parts(
    parts: Sequence[ArchivePart | str | os.PathLike[str]],
    destination: str | os.PathLike[str],
    *,
    expected_bytes: int = CHEMBL37_ARCHIVE_BYTES,
    expected_sha256: str = CHEMBL37_ARCHIVE_SHA256,
) -> dict[str, Any]:
    """Atomically assemble an explicitly ordered list of archive parts.

    ``ArchivePart`` inputs enforce declared ranges and hashes. Plain paths use
    the caller's sequence order, derive contiguous ranges from file sizes, and
    record freshly computed per-part hashes. The official final byte count and
    SHA-256 remain authoritative in both modes.
    """

    if not parts:
        raise ValueError("At least one archive part is required")
    expected_start = 0
    normalized: list[ArchivePart] = []
    for index, item in enumerate(parts):
        if isinstance(item, ArchivePart):
            part = item
        else:
            path_item = Path(item).resolve()
            size = path_item.stat().st_size if path_item.is_file() else 0
            part = ArchivePart(
                path=path_item,
                start_byte=expected_start,
                end_byte=expected_start + size - 1,
                sha256=sha256_file(path_item) if path_item.is_file() else "",
            )
        path = Path(part.path).resolve()
        if part.start_byte != expected_start:
            raise ValueError(
                f"Archive part {index} has a gap or is out of order: "
                f"expected start {expected_start}, got {part.start_byte}"
            )
        if part.end_byte < part.start_byte:
            raise ValueError(f"Archive part {index} has an invalid byte range")
        expected_part_bytes = part.end_byte - part.start_byte + 1
        if not path.is_file() or path.stat().st_size != expected_part_bytes:
            actual = path.stat().st_size if path.exists() else None
            raise ValueError(
                f"Archive part {index} size mismatch: expected {expected_part_bytes}, got {actual}"
            )
        actual_hash = sha256_file(path)
        if actual_hash.casefold() != part.sha256.casefold():
            raise ValueError(
                f"Archive part {index} SHA-256 mismatch: expected {part.sha256}, got {actual_hash}"
            )
        normalized.append(ArchivePart(path, part.start_byte, part.end_byte, actual_hash))
        expected_start = part.end_byte + 1
    if expected_start != expected_bytes:
        raise ValueError(
            f"Archive parts cover {expected_start} bytes but official archive requires {expected_bytes}"
        )

    output = Path(destination).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        verification = verify_chembl37_archive(
            output,
            expected_bytes=expected_bytes,
            expected_sha256=expected_sha256,
        )
        verification["assembly_status"] = "already_present_verified"
        return verification
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    temporary = Path(temporary_name)
    digest = hashlib.sha256()
    total_written = 0
    try:
        with os.fdopen(descriptor, "wb") as target:
            for part in normalized:
                with part.path.open("rb") as source:
                    for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
                        target.write(chunk)
                        digest.update(chunk)
                        total_written += len(chunk)
            target.flush()
            os.fsync(target.fileno())
        actual_hash = digest.hexdigest()
        if total_written != expected_bytes or actual_hash.casefold() != expected_sha256.casefold():
            raise ValueError(
                f"Assembled archive verification failed: bytes={total_written}, sha256={actual_hash}"
            )
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    verification = verify_chembl37_archive(
        output,
        expected_bytes=expected_bytes,
        expected_sha256=expected_sha256,
    )
    verification.update(
        {
            "assembly_status": "assembled_and_verified",
            "parts": [
                {
                    "file": part.path.name,
                    "start_byte": part.start_byte,
                    "end_byte": part.end_byte,
                    "size_bytes": part.end_byte - part.start_byte + 1,
                    "sha256": part.sha256,
                }
                for part in normalized
            ],
        }
    )
    _atomic_write_json(output.parent / "archive_assembly_manifest.json", verification)
    return verification


def verify_chembl37_archive(
    archive_path: str | os.PathLike[str],
    *,
    expected_bytes: int = CHEMBL37_ARCHIVE_BYTES,
    expected_sha256: str = CHEMBL37_ARCHIVE_SHA256,
) -> dict[str, Any]:
    """Verify the exact official archive size and SHA-256."""

    archive = Path(archive_path).resolve()
    if not archive.is_file():
        raise FileNotFoundError(archive)
    actual_bytes = archive.stat().st_size
    actual_sha256 = sha256_file(archive)
    result = {
        "archive_file": archive.name,
        "source_url": CHEMBL37_SQLITE_URL,
        "expected_bytes": expected_bytes,
        "actual_bytes": actual_bytes,
        "expected_sha256": expected_sha256,
        "actual_sha256": actual_sha256,
        "size_valid": actual_bytes == expected_bytes,
        "sha256_valid": actual_sha256.casefold() == expected_sha256.casefold(),
    }
    result["valid"] = bool(result["size_valid"] and result["sha256_valid"])
    if not result["valid"]:
        raise ValueError(f"ChEMBL_37 archive verification failed: {canonical_json(result)}")
    return result


def stage_chembl37_archive(
    archive_path: str | os.PathLike[str],
    raw_root: str | os.PathLike[str],
    *,
    move: bool = False,
) -> tuple[Path, dict[str, Any]]:
    """Verify and preserve the official archive under the platform raw root."""

    source = Path(archive_path).resolve()
    verification = verify_chembl37_archive(source)
    destination = Path(raw_root).resolve() / "chembl_37_bulk" / "chembl_37_sqlite.tar.gz"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        existing = verify_chembl37_archive(destination)
        if source != destination and move:
            source.unlink(missing_ok=True)
        verification = existing
    elif source == destination:
        pass
    elif move:
        os.replace(source, destination)
        verification = verify_chembl37_archive(destination)
    else:
        descriptor, temporary_name = tempfile.mkstemp(prefix=".chembl_37_sqlite.", dir=destination.parent)
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            shutil.copyfile(source, temporary)
            verify_chembl37_archive(temporary)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        verification = verify_chembl37_archive(destination)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "source_name": "ChEMBL",
        "source_version": "ChEMBL_37",
        "source_url": CHEMBL37_SQLITE_URL,
        "license": "CC BY-SA 3.0",
        "license_url": "https://creativecommons.org/licenses/by-sa/3.0/",
        "citation": CHEMBL37_CITATION,
        "required_attribution": (
            "Preserve ChEMBL identifiers and release number; cite DOI 10.1093/nar/gky1075."
        ),
        "archive_file": destination.name,
        "archive_size_bytes": verification["actual_bytes"],
        "archive_sha256": verification["actual_sha256"],
        "verification_status": "verified",
        "calculated_property_policy": (
            "preserved_unmodified_in_raw_archive; compound_properties, pchembl_value, and other "
            "calculated fields excluded only from platform activity export and canonical/model views"
        ),
    }
    _atomic_write_json(destination.parent / "archive_manifest.json", manifest)
    return destination, manifest


def _safe_archive_members(archive: tarfile.TarFile) -> list[tarfile.TarInfo]:
    members: list[tarfile.TarInfo] = []
    for member in archive.getmembers():
        path = Path(member.name)
        if path.is_absolute() or ".." in path.parts or member.issym() or member.islnk():
            raise ValueError(f"Unsafe path/link in ChEMBL archive: {member.name}")
        if member.isfile():
            members.append(member)
    return members


def extract_chembl37_sqlite(
    archive_path: str | os.PathLike[str],
    raw_root: str | os.PathLike[str],
) -> tuple[Path, dict[str, Any]]:
    """Extract the SQLite database and attribution files without path traversal."""

    archive_path = Path(archive_path).resolve()
    verification = verify_chembl37_archive(archive_path)
    destination_root = Path(raw_root).resolve() / "chembl_37_bulk" / "extracted"
    database_destination = destination_root / "chembl_37.db"
    extraction_manifest_path = destination_root / "extraction_manifest.json"
    if database_destination.exists() and extraction_manifest_path.exists():
        manifest = json.loads(extraction_manifest_path.read_text(encoding="utf-8"))
        if manifest.get("archive_sha256") == verification["actual_sha256"]:
            database_records = [
                record
                for record in manifest.get("files", [])
                if record.get("output_file") == database_destination.name
            ]
            if len(database_records) != 1:
                raise RuntimeError("Cached extraction manifest lacks one database hash record")
            database_record = database_records[0]
            actual_size = database_destination.stat().st_size
            actual_hash = sha256_file(database_destination)
            if (
                int(database_record.get("size_bytes", -1)) != actual_size
                or str(database_record.get("sha256", "")).casefold() != actual_hash.casefold()
            ):
                raise RuntimeError("Cached extracted ChEMBL database failed size/hash verification")
            _verify_sqlite_file(database_destination)
            return database_destination, manifest
        raise RuntimeError("Cached extraction belongs to a different ChEMBL archive")
    if database_destination.exists() != extraction_manifest_path.exists():
        raise RuntimeError("Partial cached ChEMBL extraction exists without its paired manifest")

    destination_root.mkdir(parents=True, exist_ok=True)
    extracted: list[dict[str, Any]] = []
    with tarfile.open(archive_path, mode="r:gz") as archive:
        members = _safe_archive_members(archive)
        database_members = [member for member in members if member.name.casefold().endswith(".db")]
        if len(database_members) != 1:
            raise ValueError(
                f"Expected exactly one .db member in ChEMBL archive; found {[m.name for m in database_members]}"
            )
        selected = database_members + [
            member
            for member in members
            if Path(member.name).name.casefold()
            in {"required.attribution", "license", "license.txt", "readme", "readme.txt"}
        ]
        for member in selected:
            target = (
                database_destination
                if member in database_members
                else destination_root / Path(member.name).name
            )
            descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=destination_root)
            temporary = Path(temporary_name)
            try:
                source_handle = archive.extractfile(member)
                if source_handle is None:
                    raise ValueError(f"Cannot read archive member: {member.name}")
                with os.fdopen(descriptor, "wb") as destination_handle, source_handle:
                    shutil.copyfileobj(source_handle, destination_handle, length=8 * 1024 * 1024)
                    destination_handle.flush()
                    os.fsync(destination_handle.fileno())
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
            extracted.append(
                {
                    "archive_member": member.name,
                    "output_file": target.name,
                    "size_bytes": target.stat().st_size,
                    "sha256": sha256_file(target),
                }
            )
    _verify_sqlite_file(database_destination)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "archive_sha256": verification["actual_sha256"],
        "extracted_at_utc": _utc_now(),
        "database_file": database_destination.name,
        "files": sorted(extracted, key=lambda row: row["output_file"]),
    }
    _atomic_write_json(extraction_manifest_path, manifest)
    return database_destination, manifest


def _verify_sqlite_file(path: Path) -> None:
    with path.open("rb") as handle:
        magic = handle.read(16)
    if magic != b"SQLite format 3\x00":
        raise ValueError(f"Extracted file is not SQLite: {path}")
    connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    try:
        result = connection.execute("PRAGMA quick_check").fetchone()
    finally:
        connection.close()
    if not result or result[0] != "ok":
        raise ValueError(f"SQLite quick_check failed for {path}: {result}")


def inspect_sqlite_schema(database_path: str | os.PathLike[str]) -> dict[str, list[str]]:
    database = Path(database_path).resolve()
    connection = sqlite3.connect(f"file:{database.as_posix()}?mode=ro&immutable=1", uri=True)
    try:
        tables = [
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            ).fetchall()
        ]
        return {
            table: [str(row[1]) for row in connection.execute(f'PRAGMA table_info("{table}")').fetchall()]
            for table in tables
        }
    finally:
        connection.close()


def _column_expression(
    schema: dict[str, list[str]],
    table: str,
    table_alias: str,
    column: str,
    output_name: str | None = None,
) -> str:
    alias = output_name or column
    if column not in schema.get(table, []):
        return f'NULL AS "{alias}"'
    return f'{table_alias}."{column}" AS "{alias}"'


def _source_table_name(schema: dict[str, list[str]]) -> str:
    """Resolve the release-specific ChEMBL activity-source table fail-closed."""

    if "source" in schema:
        return "source"
    if "source_dictionary" in schema:  # Compatibility with older fixtures/releases only.
        return "source_dictionary"
    raise ValueError("ChEMBL SQLite is missing the required activity source table")


def _activity_select(schema: dict[str, list[str]]) -> list[str]:
    source_table = _source_table_name(schema)
    activity_fields = (
        "activity_id",
        "action_type",
        "activity_comment",
        "bao_endpoint",
        "data_validity_comment",
        "potential_duplicate",
        "qudt_units",
        "record_id",
        "relation",
        "src_id",
        "standard_flag",
        "standard_relation",
        "standard_text_value",
        "standard_type",
        "standard_units",
        "standard_upper_value",
        "standard_value",
        "text_value",
        "toid",
        "type",
        "units",
        "uo_units",
        "upper_value",
        "value",
        "modality",
    )
    columns = [_column_expression(schema, "activities", "a", field) for field in activity_fields]
    columns.extend(
        [
            _column_expression(schema, "assays", "s", "chembl_id", "assay_chembl_id"),
            _column_expression(schema, "assays", "s", "description", "assay_description"),
            _column_expression(schema, "assays", "s", "assay_type"),
            _column_expression(schema, "assays", "s", "assay_test_type"),
            _column_expression(schema, "assays", "s", "assay_category"),
            _column_expression(schema, "assays", "s", "assay_organism"),
            _column_expression(schema, "assays", "s", "assay_tax_id"),
            _column_expression(schema, "assays", "s", "assay_strain"),
            _column_expression(schema, "assays", "s", "assay_tissue"),
            _column_expression(schema, "assays", "s", "assay_cell_type"),
            _column_expression(schema, "assays", "s", "assay_subcellular_fraction"),
            _column_expression(schema, "assays", "s", "relationship_type"),
            _column_expression(schema, "assays", "s", "src_assay_id"),
            _column_expression(schema, "assays", "s", "cell_id"),
            _column_expression(schema, "assays", "s", "tissue_id"),
            _column_expression(schema, "assays", "s", "variant_id"),
            _column_expression(schema, "assays", "s", "assay_group"),
            _column_expression(schema, "assays", "s", "bao_format"),
            _column_expression(schema, "assays", "s", "confidence_score"),
            _column_expression(schema, "target_dictionary", "t", "chembl_id", "target_chembl_id"),
            _column_expression(schema, "target_dictionary", "t", "pref_name", "target_pref_name"),
            _column_expression(schema, "target_dictionary", "t", "target_type"),
            _column_expression(schema, "target_dictionary", "t", "organism", "target_organism"),
            _column_expression(schema, "target_dictionary", "t", "tax_id", "target_tax_id"),
            _column_expression(schema, "molecule_dictionary", "m", "chembl_id", "molecule_chembl_id"),
            _column_expression(schema, "molecule_dictionary", "m", "pref_name", "molecule_pref_name"),
            _column_expression(schema, "compound_structures", "cs", "canonical_smiles"),
            _column_expression(schema, "compound_structures", "cs", "standard_inchi_key"),
            _column_expression(schema, "docs", "d", "chembl_id", "document_chembl_id"),
            _column_expression(schema, "docs", "d", "journal", "document_journal"),
            _column_expression(schema, "docs", "d", "year", "document_year"),
            _column_expression(schema, "docs", "d", "doi", "document_doi"),
            _column_expression(schema, "docs", "d", "pubmed_id"),
            _column_expression(schema, "docs", "d", "patent_id"),
            _column_expression(schema, "docs", "d", "title", "document_title"),
            _column_expression(schema, "docs", "d", "doc_type", "document_type"),
            _column_expression(schema, "docs", "d", "chembl_release_id", "document_chembl_release_id"),
            _column_expression(schema, source_table, "sd", "src_short_name", "activity_source_name"),
            _column_expression(
                schema,
                source_table,
                "sd",
                "src_description",
                "activity_source_description",
            ),
        ]
    )
    return columns


def _join_key(schema: dict[str, list[str]], left: str, right: str) -> str:
    return left if left in schema.get("activities", []) else right


def _activity_query(schema: dict[str, list[str]]) -> str:
    required_tables = {
        "activities",
        "assays",
        "target_dictionary",
        "molecule_dictionary",
        "compound_structures",
        "docs",
    }
    missing = sorted(required_tables - set(schema))
    if missing:
        raise ValueError(f"ChEMBL SQLite is missing required tables: {missing}")
    assay_doc = "a.doc_id" if "doc_id" in schema["activities"] else "s.doc_id"
    source_table = _source_table_name(schema)
    source_join = f'LEFT JOIN "{source_table}" AS sd ON a.src_id = sd.src_id'
    fields = [
        *_activity_select(schema),
        'NULL AS "component_accessions"',
        'NULL AS "component_sequences"',
        'NULL AS "component_types"',
    ]
    return f"""
SELECT {", ".join(fields)}
FROM activities AS a
LEFT JOIN assays AS s ON a.assay_id = s.assay_id
LEFT JOIN target_dictionary AS t ON s.tid = t.tid
LEFT JOIN molecule_dictionary AS m ON a.molregno = m.molregno
LEFT JOIN compound_structures AS cs ON a.molregno = cs.molregno
LEFT JOIN docs AS d ON {assay_doc} = d.doc_id
{source_join}
WHERE a.activity_id > ?
ORDER BY a.activity_id
LIMIT ?
""".strip()


def _component_join(schema: dict[str, list[str]]) -> tuple[str, list[str]]:
    required = {"target_components", "component_sequences"}
    if not required.issubset(schema):
        return "", [
            'NULL AS "component_accessions"',
            'NULL AS "component_sequences"',
            'NULL AS "component_types"',
        ]
    component_columns = schema["component_sequences"]
    accession = "COALESCE(cs.accession, '')" if "accession" in component_columns else "''"
    sequence = "COALESCE(cs.sequence, '')" if "sequence" in component_columns else "''"
    component_type = "COALESCE(cs.component_type, '')" if "component_type" in component_columns else "''"
    order_accession = "cs.accession" if "accession" in component_columns else "''"
    join = f"""
LEFT JOIN (
    SELECT ordered.tid,
           GROUP_CONCAT(ordered.accession, ';') AS component_accessions,
           GROUP_CONCAT(ordered.sequence, ';') AS component_sequences,
           GROUP_CONCAT(ordered.component_type, ';') AS component_types
    FROM (
        SELECT tc.tid,
               {accession} AS accession,
               {sequence} AS sequence,
               {component_type} AS component_type
        FROM target_components AS tc
        LEFT JOIN component_sequences AS cs ON tc.component_id = cs.component_id
        ORDER BY tc.tid, tc.component_id, {order_accession}
    ) AS ordered
    GROUP BY ordered.tid
) AS comp ON t.tid = comp.tid
""".strip()
    return join, [
        'comp.component_accessions AS "component_accessions"',
        'comp.component_sequences AS "component_sequences"',
        'comp.component_types AS "component_types"',
    ]


def _specialized_activity_query(schema: dict[str, list[str]], where_clause: str) -> str:
    required_tables = {
        "activities",
        "assays",
        "target_dictionary",
        "molecule_dictionary",
        "compound_structures",
        "docs",
    }
    missing = sorted(required_tables - set(schema))
    if missing:
        raise ValueError(f"ChEMBL SQLite is missing required tables: {missing}")
    assay_doc = "a.doc_id" if "doc_id" in schema["activities"] else "s.doc_id"
    source_table = _source_table_name(schema)
    source_join = f'LEFT JOIN "{source_table}" AS sd ON a.src_id = sd.src_id'
    component_join, component_fields = _component_join(schema)
    fields = [*_activity_select(schema), *component_fields]
    return f"""
SELECT {", ".join(fields)}
FROM activities AS a
LEFT JOIN assays AS s ON a.assay_id = s.assay_id
LEFT JOIN target_dictionary AS t ON s.tid = t.tid
LEFT JOIN molecule_dictionary AS m ON a.molregno = m.molregno
LEFT JOIN compound_structures AS cs ON a.molregno = cs.molregno
LEFT JOIN docs AS d ON {assay_doc} = d.doc_id
{source_join}
{component_join}
WHERE a.activity_id > ? AND ({where_clause})
ORDER BY a.activity_id
LIMIT ?
""".strip()


def _export_keyset_view(
    connection: sqlite3.Connection,
    query: str,
    count_query: str,
    destination: Path,
    *,
    view_name: str,
    database_sha256: str,
    chunk_size: int,
    scientific_boundary: str,
) -> dict[str, Any]:
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination.parent / f"{view_name}_manifest.json"
    query_sha256 = hashlib.sha256(query.encode("utf-8")).hexdigest()
    schema_contract = arrow_schema_contract(ACTIVITY_ARROW_SCHEMA)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("database_sha256") == database_sha256
            and manifest.get("query_sha256") == query_sha256
        ):
            _verify_manifest_arrow_schema(
                manifest,
                destination.parent,
                ACTIVITY_ARROW_SCHEMA,
                context=view_name,
            )
            _verify_schema_normalization_receipt(manifest, destination.parent.parent)
            listed = {
                (destination.parent / str(part["path"])).resolve() for part in manifest.get("parts", [])
            }
            actual = {path.resolve() for path in destination.glob("*.parquet")}
            if actual != listed:
                raise RuntimeError(f"Unmanifested or missing Parquet parts exist for {view_name}")
            return manifest
        raise RuntimeError(f"Existing {view_name} manifest does not match database/query contract")
    checkpoint_path = destination.parent / f"{view_name}_checkpoint.json"
    checkpoint: dict[str, Any] = {
        "database_sha256": database_sha256,
        "query_sha256": query_sha256,
        "last_activity_id": 0,
        "rows_written": 0,
        "parts": [],
        "arrow_schema": schema_contract,
    }
    if checkpoint_path.exists():
        candidate = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if (
            candidate.get("database_sha256") == database_sha256
            and candidate.get("query_sha256") == query_sha256
            and candidate.get("arrow_schema") == schema_contract
        ):
            checkpoint = candidate
            _verify_manifest_arrow_schema(
                checkpoint,
                destination.parent,
                ACTIVITY_ARROW_SCHEMA,
                context=f"{view_name} checkpoint",
            )
        else:
            raise RuntimeError(f"Existing {view_name} checkpoint does not match database/query contract")
    elif any(destination.glob("*.parquet")):
        raise RuntimeError(f"Untracked Parquet parts exist for {view_name}; refusing to overwrite")
    expected_rows = int(connection.execute(count_query).fetchone()[0])
    last_activity_id = int(checkpoint["last_activity_id"])
    while True:
        frame = pd.read_sql_query(query, connection, params=(last_activity_id, chunk_size))
        if frame.empty:
            break
        frame = frame.sort_values("activity_id", kind="stable").reset_index(drop=True)
        first_id = int(frame["activity_id"].iloc[0])
        final_id = int(frame["activity_id"].iloc[-1])
        if first_id <= last_activity_id or frame["activity_id"].duplicated().any():
            raise RuntimeError(f"Non-monotonic activity IDs in {view_name}")
        part_path = destination / f"{view_name}_{first_id:09d}_{final_id:09d}.parquet"
        if part_path.exists():
            raise RuntimeError(f"Unexpected unmanifested specialized view part: {part_path}")
        _write_frame_with_schema(frame, part_path, ACTIVITY_ARROW_SCHEMA)
        checkpoint["parts"].append(
            {
                "path": part_path.relative_to(destination.parent).as_posix(),
                "rows": len(frame),
                "first_activity_id": first_id,
                "last_activity_id": final_id,
                "sha256": sha256_file(part_path),
                "size_bytes": part_path.stat().st_size,
                "arrow_schema_sha256": schema_contract["sha256"],
            }
        )
        checkpoint["last_activity_id"] = final_id
        checkpoint["rows_written"] = int(checkpoint["rows_written"]) + len(frame)
        checkpoint["updated_at_utc"] = _utc_now()
        _atomic_write_json(checkpoint_path, checkpoint)
        last_activity_id = final_id
    if int(checkpoint["rows_written"]) != expected_rows:
        raise RuntimeError(f"{view_name} row-count mismatch: {checkpoint['rows_written']} != {expected_rows}")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "view_name": view_name,
        "source_version": "ChEMBL_37",
        "database_sha256": database_sha256,
        "query_sha256": query_sha256,
        "query": query,
        "count_query": count_query,
        "row_count": expected_rows,
        "part_count": len(checkpoint["parts"]),
        "parts": checkpoint["parts"],
        "arrow_schema": schema_contract,
        "scientific_boundary": scientific_boundary,
        "completed_at_utc": _utc_now(),
    }
    _atomic_write_json(manifest_path, manifest)
    checkpoint_path.unlink(missing_ok=True)
    return manifest


def _verified_manifest_parts(manifest: dict[str, Any], root: Path) -> list[tuple[Path, dict[str, Any]]]:
    view_name = str(manifest.get("view_name", "")).strip()
    database_sha256 = str(manifest.get("database_sha256", "")).strip()
    if not view_name or not re.fullmatch(r"[0-9a-f]{64}", database_sha256):
        raise RuntimeError("Overlap manifest lacks a valid view/database identity")
    records: list[tuple[Path, dict[str, Any]]] = []
    listed: set[Path] = set()
    for part in manifest.get("parts", []):
        path = (root / str(part["path"])).resolve()
        try:
            path.relative_to(root.resolve())
        except ValueError as error:
            raise RuntimeError("Overlap manifest part escapes its artifact root") from error
        if path in listed:
            raise RuntimeError("Overlap manifest lists the same part more than once")
        listed.add(path)
        records.append((path, part))
    actual = {path.resolve() for path in (root / view_name).glob("*.parquet")}
    if listed != actual:
        raise RuntimeError(f"Overlap manifest listed/physical part mismatch for {view_name}")
    return records


def _verified_activity_id_set(manifest: dict[str, Any], root: Path) -> set[int]:
    """Load one manifest-bound activity-ID set, failing on duplicates or drift."""

    activity_ids: set[int] = set()
    physical_rows = 0
    prior_activity_id = -1
    for path, part in _verified_manifest_parts(manifest, root):
        if not path.is_file() or sha256_file(path) != part["sha256"]:
            raise RuntimeError(f"Overlap input failed SHA-256 verification: {path}")
        frame = pd.read_parquet(path, columns=["activity_id"])
        if len(frame) != int(part["rows"]):
            raise RuntimeError(f"Overlap input row-count mismatch: {path}")
        numeric = pd.to_numeric(frame["activity_id"], errors="raise").astype("int64")
        if not numeric.is_monotonic_increasing or (
            not numeric.empty and int(numeric.iloc[0]) <= prior_activity_id
        ):
            raise RuntimeError(f"Overlap input activity IDs are not globally increasing: {path}")
        part_ids = {int(value) for value in numeric}
        if len(part_ids) != len(numeric) or activity_ids.intersection(part_ids):
            raise RuntimeError(f"Overlap input contains duplicate activity IDs: {path}")
        activity_ids.update(part_ids)
        physical_rows += len(frame)
        if not numeric.empty:
            prior_activity_id = int(numeric.iloc[-1])
    if physical_rows != int(manifest.get("row_count", -1)):
        raise RuntimeError("Overlap manifest row count does not reconcile to its parts")
    return activity_ids


def exact_manifest_activity_overlap(
    left_manifest: dict[str, Any],
    right_manifest: dict[str, Any],
    root: str | os.PathLike[str],
) -> int:
    """Count an exact intersection while retaining only the smaller ID set in memory."""

    artifact_root = Path(root).resolve()
    if left_manifest.get("database_sha256") != right_manifest.get("database_sha256"):
        raise RuntimeError("Cannot intersect views exported from different database snapshots")
    if left_manifest.get("view_name") == right_manifest.get("view_name"):
        raise RuntimeError("Overlap contract requires two distinct view identities")
    smaller, larger = sorted(
        (left_manifest, right_manifest),
        key=lambda manifest: int(manifest.get("row_count", -1)),
    )
    smaller_ids = _verified_activity_id_set(smaller, artifact_root)
    overlap = 0
    physical_rows = 0
    prior_activity_id = -1
    for path, part in _verified_manifest_parts(larger, artifact_root):
        if not path.is_file() or sha256_file(path) != part["sha256"]:
            raise RuntimeError(f"Overlap input failed SHA-256 verification: {path}")
        frame = pd.read_parquet(path, columns=["activity_id"])
        if len(frame) != int(part["rows"]):
            raise RuntimeError(f"Overlap input row-count mismatch: {path}")
        numeric = pd.to_numeric(frame["activity_id"], errors="raise").astype("int64")
        if (
            numeric.duplicated().any()
            or not numeric.is_monotonic_increasing
            or (not numeric.empty and int(numeric.iloc[0]) <= prior_activity_id)
        ):
            raise RuntimeError(f"Overlap input is not globally unique/increasing: {path}")
        overlap += int(numeric.isin(smaller_ids).sum())
        physical_rows += len(frame)
        if not numeric.empty:
            prior_activity_id = int(numeric.iloc[-1])
    if physical_rows != int(larger.get("row_count", -1)):
        raise RuntimeError("Overlap manifest row count does not reconcile to its parts")
    return overlap


def _export_target_components(
    connection: sqlite3.Connection,
    schema: dict[str, list[str]],
    destination: Path,
    *,
    database_sha256: str,
) -> dict[str, Any]:
    if not {"target_dictionary", "target_components", "component_sequences"}.issubset(schema):
        raise ValueError("ChEMBL target/component sequence tables are required")
    fields = [
        _column_expression(schema, "target_dictionary", "t", "chembl_id", "target_chembl_id"),
        _column_expression(schema, "target_dictionary", "t", "pref_name", "target_name"),
        _column_expression(schema, "target_dictionary", "t", "target_type"),
        _column_expression(schema, "target_dictionary", "t", "organism", "target_organism"),
        _column_expression(schema, "target_dictionary", "t", "tax_id", "target_tax_id"),
        _column_expression(schema, "target_components", "tc", "component_id"),
        _column_expression(schema, "target_components", "tc", "homologue"),
        _column_expression(schema, "component_sequences", "cs", "accession"),
        _column_expression(schema, "component_sequences", "cs", "component_type"),
        _column_expression(schema, "component_sequences", "cs", "sequence"),
        _column_expression(schema, "component_sequences", "cs", "sequence_md5sum"),
        _column_expression(schema, "component_sequences", "cs", "organism", "component_organism"),
        _column_expression(schema, "component_sequences", "cs", "tax_id", "component_tax_id"),
    ]
    query = f"""
SELECT {", ".join(fields)}
FROM target_dictionary AS t
LEFT JOIN target_components AS tc ON t.tid = tc.tid
LEFT JOIN component_sequences AS cs ON tc.component_id = cs.component_id
ORDER BY t.chembl_id, tc.component_id
""".strip()
    frame = pd.read_sql_query(query, connection)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rewrite = not destination.exists()
    if destination.exists():
        existing = pd.read_parquet(destination)
        try:
            pd.testing.assert_frame_equal(existing, frame, check_dtype=False)
        except AssertionError as error:
            raise RuntimeError("Existing target-component artifact differs from its source query") from error
        rewrite = (
            not pq.ParquetFile(destination)
            .schema_arrow.remove_metadata()
            .equals(
                TARGET_COMPONENT_ARROW_SCHEMA,
                check_metadata=True,
            )
        )
    if rewrite:
        _write_frame_with_schema(frame, destination, TARGET_COMPONENT_ARROW_SCHEMA)
    schema_contract = arrow_schema_contract(TARGET_COMPONENT_ARROW_SCHEMA)
    record = {
        "schema_version": SCHEMA_VERSION,
        "view_name": "target_components",
        "source_version": "ChEMBL_37",
        "database_sha256": database_sha256,
        "path": destination.name,
        "row_count": len(frame),
        "sha256": sha256_file(destination),
        "size_bytes": destination.stat().st_size,
        "arrow_schema": schema_contract,
        "arrow_schema_sha256": schema_contract["sha256"],
        "query": query,
        "query_sha256": hashlib.sha256(query.encode("utf-8")).hexdigest(),
    }
    _atomic_write_json(destination.parent / "target_components_manifest.json", record)
    return record


def _export_development_annotations(
    connection: sqlite3.Connection,
    schema: dict[str, list[str]],
    destination: Path,
    *,
    database_sha256: str,
    chunk_size: int,
) -> dict[str, Any]:
    """Export molecule metadata in resumable molregno-keyset partitions."""

    if "molecule_dictionary" not in schema or "molregno" not in schema["molecule_dictionary"]:
        raise ValueError("ChEMBL molecule_dictionary.molregno is required")
    fields = [
        _column_expression(schema, "molecule_dictionary", "m", "molregno", "molecule_row_id"),
        _column_expression(schema, "molecule_dictionary", "m", "chembl_id", "molecule_chembl_id"),
        _column_expression(schema, "molecule_dictionary", "m", "pref_name", "molecule_pref_name"),
        _column_expression(schema, "molecule_dictionary", "m", "molecule_type"),
        _column_expression(schema, "molecule_dictionary", "m", "max_phase"),
        _column_expression(schema, "molecule_dictionary", "m", "first_approval"),
        _column_expression(schema, "molecule_dictionary", "m", "withdrawn_flag"),
        _column_expression(schema, "molecule_dictionary", "m", "black_box_warning"),
        _column_expression(schema, "molecule_dictionary", "m", "therapeutic_flag"),
    ]
    query = (
        f"SELECT {', '.join(fields)} FROM molecule_dictionary AS m "
        "WHERE m.molregno > ? ORDER BY m.molregno LIMIT ?"
    )
    query_sha256 = hashlib.sha256(query.encode("utf-8")).hexdigest()
    schema_contract = arrow_schema_contract(DEVELOPMENT_ARROW_SCHEMA)
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination.parent / "molecule_development_annotations_manifest.json"
    checkpoint_path = destination.parent / "molecule_development_annotations_checkpoint.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("database_sha256") != database_sha256
            or manifest.get("query_sha256") != query_sha256
            or manifest.get("source_version", "ChEMBL_37") != "ChEMBL_37"
        ):
            raise RuntimeError("Existing development-annotation manifest has a stale contract")
        _verify_manifest_arrow_schema(
            manifest,
            destination.parent,
            DEVELOPMENT_ARROW_SCHEMA,
            context="molecule_development_annotations",
        )
        _verify_schema_normalization_receipt(manifest, destination.parent.parent)
        listed = {(destination.parent / str(part["path"])).resolve() for part in manifest.get("parts", [])}
        actual = {path.resolve() for path in destination.glob("*.parquet")}
        if actual != listed:
            raise RuntimeError("Unmanifested or missing development-annotation parts exist")
        if "source_version" not in manifest:
            manifest["source_version"] = "ChEMBL_37"
            _atomic_write_json(manifest_path, manifest)
        return manifest

    checkpoint: dict[str, Any] = {
        "database_sha256": database_sha256,
        "query_sha256": query_sha256,
        "last_molregno": 0,
        "rows_written": 0,
        "parts": [],
        "arrow_schema": schema_contract,
    }
    if checkpoint_path.exists():
        candidate = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if (
            candidate.get("database_sha256") != database_sha256
            or candidate.get("query_sha256") != query_sha256
            or candidate.get("arrow_schema") != schema_contract
        ):
            raise RuntimeError("Existing development-annotation checkpoint has a stale contract")
        checkpoint = candidate
        _verify_manifest_arrow_schema(
            checkpoint,
            destination.parent,
            DEVELOPMENT_ARROW_SCHEMA,
            context="development checkpoint",
        )
    elif any(destination.glob("*.parquet")):
        raise RuntimeError("Untracked development-annotation parts exist; refusing to overwrite")

    expected_rows = int(connection.execute("SELECT COUNT(*) FROM molecule_dictionary").fetchone()[0])
    last_molregno = int(checkpoint["last_molregno"])
    while True:
        frame = pd.read_sql_query(query, connection, params=(last_molregno, chunk_size))
        if frame.empty:
            break
        frame = frame.sort_values("molecule_row_id", kind="stable").reset_index(drop=True)
        first_id = int(frame["molecule_row_id"].iloc[0])
        final_id = int(frame["molecule_row_id"].iloc[-1])
        if first_id <= last_molregno or frame["molecule_row_id"].duplicated().any():
            raise RuntimeError("Non-monotonic molecule IDs in development metadata export")
        frame["annotation_role"] = "development_metadata_not_outcome_or_model_label"
        part_path = destination / f"molecules_{first_id:09d}_{final_id:09d}.parquet"
        if part_path.exists():
            raise RuntimeError(f"Unexpected unmanifested development-annotation part: {part_path}")
        _write_frame_with_schema(frame, part_path, DEVELOPMENT_ARROW_SCHEMA)
        checkpoint["parts"].append(
            {
                "path": part_path.relative_to(destination.parent).as_posix(),
                "rows": len(frame),
                "first_molregno": first_id,
                "last_molregno": final_id,
                "sha256": sha256_file(part_path),
                "size_bytes": part_path.stat().st_size,
                "arrow_schema_sha256": schema_contract["sha256"],
            }
        )
        checkpoint["last_molregno"] = final_id
        checkpoint["rows_written"] = int(checkpoint["rows_written"]) + len(frame)
        checkpoint["updated_at_utc"] = _utc_now()
        _atomic_write_json(checkpoint_path, checkpoint)
        last_molregno = final_id
    if int(checkpoint["rows_written"]) != expected_rows:
        raise RuntimeError("Development-annotation row-count mismatch")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "view_name": "molecule_development_annotations",
        "source_version": "ChEMBL_37",
        "database_sha256": database_sha256,
        "query": query,
        "query_sha256": query_sha256,
        "row_count": expected_rows,
        "part_count": len(checkpoint["parts"]),
        "parts": checkpoint["parts"],
        "arrow_schema": schema_contract,
        "semantic_role": "metadata_only_not_outcome",
        "completed_at_utc": _utc_now(),
    }
    _atomic_write_json(manifest_path, manifest)
    checkpoint_path.unlink(missing_ok=True)
    return manifest


def export_chembl37_specialized_views(
    database_path: str | os.PathLike[str],
    interim_root: str | os.PathLike[str],
    *,
    chunk_size: int = 200_000,
) -> dict[str, Any]:
    """Export intentionally overlapping, scientifically separated inventories."""

    database = Path(database_path).resolve()
    _verify_sqlite_file(database)
    database_sha256 = sha256_file(database)
    bulk_root = Path(interim_root).resolve() / "chembl_37_bulk"
    _assert_schema_normalization_idle(bulk_root)
    destination = bulk_root / "specialized_views"
    destination.mkdir(parents=True, exist_ok=True)
    schema = inspect_sqlite_schema(database)
    connection = sqlite3.connect(f"file:{database.as_posix()}?mode=ro&immutable=1", uri=True)
    kd_ki_where = (
        "upper(t.target_type) = 'SINGLE PROTEIN' "
        "AND upper(a.standard_type) IN ('KD','KI') "
        "AND a.standard_value IS NOT NULL AND a.standard_value > 0 "
        "AND lower(a.standard_units) IN "
        "('pm','picomolar','nm','nanomolar','um','micromolar','mm','millimolar','m','molar')"
    )
    ic50_ec50_where = (
        "upper(t.target_type) = 'SINGLE PROTEIN' "
        "AND upper(a.standard_type) IN ('IC50','EC50') "
        "AND a.standard_value IS NOT NULL AND a.standard_value > 0 "
        "AND lower(a.standard_units) IN "
        "('pm','picomolar','nm','nanomolar','um','micromolar','mm','millimolar','m','molar')"
    )
    herg_where = "t.chembl_id = 'CHEMBL240'"
    pk_types = (
        "'AUC','AUCINF','AUCLAST','CL','CLINT','CL_RENAL','CMAX','F','FU','HALF-LIFE',"
        "'PAPP','PEFF','PPB','SOLUBILITY','T1/2','TMAX','VD','VDSS','LOGD','LOGP'"
    )
    pk_where = (
        f"(a.src_id = 39 OR upper(a.standard_type) IN ({pk_types}) "
        "OR lower(s.description) LIKE '%microsom%' OR lower(s.description) LIKE '%hepatocyte%' "
        "OR lower(s.description) LIKE '%permeab%' OR lower(s.description) LIKE '%bioavailability%' "
        "OR lower(s.description) LIKE '%plasma protein binding%')"
    )
    cardiac_where = (
        "(upper(a.standard_type) IN ('QT','QTC','QTCF','QTCB','APD','APD50','APD90') "
        "OR upper(a.standard_type) LIKE 'QT%' OR upper(a.standard_type) LIKE 'APD%' "
        "OR lower(s.description) LIKE '%qt interval%' OR lower(s.description) LIKE '%qtc%' "
        "OR lower(s.description) LIKE '%action potential duration%')"
    )
    source_table = _source_table_name(schema)
    source_39 = pd.read_sql_query(
        f'SELECT * FROM "{source_table}" WHERE src_id = 39 ORDER BY src_id',
        connection,
    ).to_dict("records")
    if len(source_39) != 1:
        raise ValueError(f"Expected exactly one {source_table} record for src_id 39; found {len(source_39)}")

    view_specs = {
        "single_protein_kd_ki": (
            kd_ki_where,
            "Kd and Ki are retained as separate endpoints; only Kd exact rows may enter the free-energy derivation.",
        ),
        "single_protein_ic50_ec50_candidates": (
            ic50_ec50_where,
            "IC50 and EC50 potency/activity candidates remain separate and are never labeled binding affinity.",
        ),
        "herg_all_endpoints": (
            herg_where,
            "All CHEMBL240 observations; hERG is not QT, TdP, cardiotoxicity, or clinical risk.",
        ),
        "pk_adme_candidates": (
            pk_where,
            "Candidate inventory only; endpoint/context/unit rules must split admitted and quarantine rows downstream.",
        ),
        "cardiac_qt_apd_inventory": (
            cardiac_where,
            "Explicit QT/QTc/APD naming only; never inferred from hERG and not clinical-task eligible by default.",
        ),
    }
    manifests: dict[str, Any] = {}
    try:
        for name, (where_clause, boundary) in view_specs.items():
            query = _specialized_activity_query(schema, where_clause)
            count_query = f"""
SELECT COUNT(*)
FROM activities AS a
LEFT JOIN assays AS s ON a.assay_id = s.assay_id
LEFT JOIN target_dictionary AS t ON s.tid = t.tid
WHERE {where_clause}
""".strip()
            manifests[name] = _export_keyset_view(
                connection,
                query,
                count_query,
                destination / name,
                view_name=name,
                database_sha256=database_sha256,
                chunk_size=chunk_size,
                scientific_boundary=boundary,
            )
        manifests["target_components"] = _export_target_components(
            connection,
            schema,
            destination / "target_components.parquet",
            database_sha256=database_sha256,
        )
        manifests["molecule_development_annotations"] = _export_development_annotations(
            connection,
            schema,
            destination / "molecule_development_annotations",
            database_sha256=database_sha256,
            chunk_size=chunk_size,
        )
        overlap_pairs = {
            "kd_ki_and_herg": ("single_protein_kd_ki", "herg_all_endpoints"),
            "ic50_ec50_and_herg": (
                "single_protein_ic50_ec50_candidates",
                "herg_all_endpoints",
            ),
            "pk_adme_and_herg": ("pk_adme_candidates", "herg_all_endpoints"),
            "cardiac_qt_apd_and_herg": (
                "cardiac_qt_apd_inventory",
                "herg_all_endpoints",
            ),
        }
        overlaps = {
            name: exact_manifest_activity_overlap(
                manifests[left],
                manifests[right],
                destination,
            )
            for name, (left, right) in overlap_pairs.items()
        }
    finally:
        connection.close()
    summary = {
        "schema_version": SCHEMA_VERSION,
        "source_version": "ChEMBL_37",
        "database_sha256": database_sha256,
        "views": manifests,
        "inventory_relationship": "overlapping_by_design_with_counts_reported",
        "overlap_row_counts": overlaps,
        "overlap_method": "exact activity_id intersection over SHA-256-verified Parquet parts",
        "activity_source_table": source_table,
        "activity_source_src_id_39": source_39[0],
        "clinical_qt_coverage_claim": "absent_unless_explicit_rows_survive_downstream_context_review",
        "endpoint_pooling": "prohibited",
        "completed_at_utc": _utc_now(),
    }
    _atomic_write_json(destination / "specialized_views_manifest.json", summary)
    return summary


def finalize_chembl37_specialized_summary(
    database_path: str | os.PathLike[str],
    interim_root: str | os.PathLike[str],
) -> dict[str, Any]:
    """Finalize a summary from completed immutable exports without rerunning view queries."""

    database = Path(database_path).resolve()
    _verify_sqlite_file(database)
    bulk_root = Path(interim_root).resolve() / "chembl_37_bulk"
    _assert_schema_normalization_idle(bulk_root)
    destination = bulk_root / "specialized_views"
    view_names = (
        "single_protein_kd_ki",
        "single_protein_ic50_ec50_candidates",
        "herg_all_endpoints",
        "pk_adme_candidates",
        "cardiac_qt_apd_inventory",
    )
    manifests: dict[str, Any] = {}
    for view_name in view_names:
        manifest_path = destination / f"{view_name}_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Incomplete specialized export: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("view_name") != view_name:
            raise RuntimeError(f"Specialized view manifest mismatch: {manifest_path}")
        _verify_manifest_arrow_schema(
            manifest,
            destination,
            ACTIVITY_ARROW_SCHEMA,
            context=view_name,
        )
        manifests[view_name] = manifest
    database_hashes = {str(manifest.get("database_sha256", "")) for manifest in manifests.values()}
    if len(database_hashes) != 1 or "" in database_hashes:
        raise RuntimeError("Specialized views do not share one database digest")
    database_sha256 = next(iter(database_hashes))
    extraction_manifest_path = database.parent / "extraction_manifest.json"
    if extraction_manifest_path.is_file():
        extraction = json.loads(extraction_manifest_path.read_text(encoding="utf-8"))
        matching = [
            record for record in extraction.get("files", []) if record.get("output_file") == database.name
        ]
        if (
            len(matching) != 1
            or matching[0].get("sha256") != database_sha256
            or int(matching[0].get("size_bytes", -1)) != database.stat().st_size
        ):
            raise RuntimeError("Extracted database no longer matches its verified extraction manifest")

    schema = inspect_sqlite_schema(database)
    source_table = _source_table_name(schema)
    connection = sqlite3.connect(f"file:{database.as_posix()}?mode=ro&immutable=1", uri=True)
    try:
        source_39 = pd.read_sql_query(
            f'SELECT * FROM "{source_table}" WHERE src_id = 39 ORDER BY src_id',
            connection,
        ).to_dict("records")
        if len(source_39) != 1:
            raise ValueError(
                f"Expected exactly one {source_table} record for src_id 39; found {len(source_39)}"
            )
        manifests["target_components"] = _export_target_components(
            connection,
            schema,
            destination / "target_components.parquet",
            database_sha256=database_sha256,
        )
    finally:
        connection.close()

    development_path = destination / "molecule_development_annotations_manifest.json"
    if not development_path.is_file():
        raise FileNotFoundError(development_path)
    development = json.loads(development_path.read_text(encoding="utf-8"))
    if (
        development.get("database_sha256") != database_sha256
        or development.get("semantic_role") != "metadata_only_not_outcome"
        or development.get("source_version", "ChEMBL_37") != "ChEMBL_37"
    ):
        raise RuntimeError("Development metadata manifest violates its database/semantic contract")
    _verify_manifest_arrow_schema(
        development,
        destination,
        DEVELOPMENT_ARROW_SCHEMA,
        context="molecule_development_annotations",
    )
    development_rows = 0
    for part in development.get("parts", []):
        path = destination / str(part["path"])
        if not path.is_file() or sha256_file(path) != part["sha256"]:
            raise RuntimeError(f"Development metadata part failed verification: {path}")
        development_rows += int(part["rows"])
    if development_rows != int(development.get("row_count", -1)):
        raise RuntimeError("Development metadata part counts do not reconcile")
    if "source_version" not in development:
        development["source_version"] = "ChEMBL_37"
        _atomic_write_json(development_path, development)
    manifests["molecule_development_annotations"] = development

    overlap_pairs = {
        "kd_ki_and_herg": ("single_protein_kd_ki", "herg_all_endpoints"),
        "ic50_ec50_and_herg": (
            "single_protein_ic50_ec50_candidates",
            "herg_all_endpoints",
        ),
        "pk_adme_and_herg": ("pk_adme_candidates", "herg_all_endpoints"),
        "cardiac_qt_apd_and_herg": ("cardiac_qt_apd_inventory", "herg_all_endpoints"),
    }
    overlaps = {
        name: exact_manifest_activity_overlap(manifests[left], manifests[right], destination)
        for name, (left, right) in overlap_pairs.items()
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "source_version": "ChEMBL_37",
        "database_sha256": database_sha256,
        "views": manifests,
        "inventory_relationship": "overlapping_by_design_with_counts_reported",
        "overlap_row_counts": overlaps,
        "overlap_method": "exact activity_id intersection over SHA-256-verified Parquet parts",
        "activity_source_table": source_table,
        "activity_source_src_id_39": source_39[0],
        "clinical_qt_coverage_claim": "absent_unless_explicit_rows_survive_downstream_context_review",
        "endpoint_pooling": "prohibited",
        "completed_at_utc": _utc_now(),
    }
    receipt_bindings = {
        canonical_json(manifest["schema_normalization_receipt"])
        for manifest in manifests.values()
        if manifest.get("schema_normalization_receipt") is not None
    }
    if receipt_bindings:
        if len(receipt_bindings) != 1 or any(
            manifest.get("schema_normalization_receipt") is None for manifest in manifests.values()
        ):
            raise RuntimeError("Specialized manifests have mixed normalization receipt bindings")
        summary["schema_normalization_receipt"] = next(iter(manifests.values()))[
            "schema_normalization_receipt"
        ]
        summary["arrow_schema_contracts"] = {
            "activity": arrow_schema_contract(ACTIVITY_ARROW_SCHEMA),
            "molecule_development_annotations": arrow_schema_contract(DEVELOPMENT_ARROW_SCHEMA),
            "target_components": arrow_schema_contract(TARGET_COMPONENT_ARROW_SCHEMA),
        }
    _atomic_write_json(destination / "specialized_views_manifest.json", summary)
    _verify_schema_normalization_receipt(summary, bulk_root)
    return summary


def normalize_chembl37_export_schemas(
    database_path: str | os.PathLike[str],
    interim_root: str | os.PathLike[str],
) -> dict[str, Any]:
    """Transactionally normalize every existing ChEMBL export to declared Arrow schemas."""

    database = Path(database_path).resolve()
    _verify_sqlite_file(database)
    bulk_root = Path(interim_root).resolve() / "chembl_37_bulk"
    staging = bulk_root / ".schema_normalization_staging"
    marker = bulk_root / _SCHEMA_NORMALIZATION_MARKER
    if marker.exists():
        transaction = json.loads(marker.read_text(encoding="utf-8"))
        if transaction.get("status") == "committing":
            journal_identity = transaction.get("source_identity")
            if not isinstance(journal_identity, dict) or journal_identity.get(
                "database_sha256"
            ) != sha256_file(database):
                raise RuntimeError("Resume database does not match the schema-normalization journal identity")
            return _resume_schema_normalization_transaction(bulk_root)
        owner = transaction.get("owner", {})
        raise RuntimeError(
            "A staging-phase schema normalization may still be active; "
            "inspect it explicitly instead of deleting its state "
            f"(transaction_id={transaction.get('transaction_id')}, "
            f"owner={owner}, started_at_utc={transaction.get('started_at_utc')}, "
            f"marker_mtime_ns={marker.stat().st_mtime_ns})"
        )
    source_identity = _schema_normalization_source_identity(database, bulk_root)
    activity_manifest_path = bulk_root / "activity_export_manifest.json"
    if activity_manifest_path.is_file():
        existing_activity = json.loads(activity_manifest_path.read_text(encoding="utf-8"))
        if existing_activity.get("schema_normalization_receipt") is not None:
            _verify_schema_normalization_receipt(existing_activity, bulk_root)
            summary = json.loads(
                (bulk_root / "specialized_views" / "specialized_views_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            return {
                "schema_version": SCHEMA_VERSION,
                "normalized_at_utc": existing_activity.get("schema_normalized_at_utc"),
                "activity_export_manifest": existing_activity,
                "specialized_views_manifest": summary,
                "normalized_part_count": int(summary["schema_normalization_receipt"]["parquet_files"]),
                "schema_normalization_receipt": summary["schema_normalization_receipt"],
                "arrow_schema_contracts": summary["arrow_schema_contracts"],
            }
    if staging.exists():
        raise RuntimeError(f"Unexpected schema-normalization staging directory: {staging}")
    staging.mkdir(parents=True)
    _atomic_write_json(
        marker,
        {
            "schema_version": SCHEMA_VERSION,
            "status": "staging",
            "started_at_utc": _utc_now(),
            "staging_directory": staging.name,
            "transaction_id": str(uuid.uuid4()),
            "owner": {
                "hostname": socket.gethostname(),
                "pid": os.getpid(),
            },
        },
    )
    replacements: list[tuple[Path, Path]] = []
    staged_manifests: dict[Path, dict[str, Any]] = {}
    rewrite_receipts: list[dict[str, Any]] = []
    expected_old_states: dict[Path, tuple[str | None, int | None]] = {}

    def bind_old_state(
        path: Path,
        expected_sha256: str | None,
        expected_size_bytes: int | None,
    ) -> None:
        resolved = path.resolve()
        state = (expected_sha256, expected_size_bytes)
        prior = expected_old_states.setdefault(resolved, state)
        if prior != state:
            raise RuntimeError(f"Conflicting pre-stage old-state binding: {resolved}")

    def load_manifest(path: Path) -> dict[str, Any]:
        if not path.is_file():
            raise FileNotFoundError(path)
        document, payload_sha256, payload_size = _read_json_byte_snapshot(
            path,
            context="Manifest staged for schema normalization",
        )
        bind_old_state(path, payload_sha256, payload_size)
        return document

    def stage_partitioned(
        manifest_path: Path,
        artifact_root: Path,
        physical_directory: Path,
        schema: pa.Schema,
        *,
        allowed_missing: frozenset[str] = frozenset(),
    ) -> dict[str, Any]:
        manifest = load_manifest(manifest_path)
        contract = arrow_schema_contract(schema)
        listed: set[Path] = set()
        normalized_parts: list[dict[str, Any]] = []
        physical_rows = 0
        for part in manifest.get("parts", []):
            source_path = (artifact_root / str(part["path"])).resolve()
            try:
                source_path.relative_to(bulk_root)
            except ValueError as error:
                raise RuntimeError(f"Manifested export part escapes bulk root: {source_path}") from error
            if source_path in listed:
                raise RuntimeError(f"Duplicate manifested export part: {source_path}")
            listed.add(source_path)
            if not source_path.is_file():
                raise RuntimeError(f"Missing manifested export part: {source_path}")
            parquet_file = pq.ParquetFile(source_path)
            metadata = parquet_file.metadata
            if (
                metadata is None
                or int(metadata.num_rows) != int(part.get("rows", -1))
                or source_path.stat().st_size != int(part.get("size_bytes", -1))
                or sha256_file(source_path) != part.get("sha256")
            ):
                raise RuntimeError(f"Pre-normalization part integrity failure: {source_path}")
            bind_old_state(
                source_path,
                str(part["sha256"]),
                int(part["size_bytes"]),
            )
            old_table = pq.read_table(source_path)
            if sha256_file(source_path) != part["sha256"]:
                raise RuntimeError(f"Export part changed while it was staged: {source_path}")
            old_schema_fingerprint = arrow_schema_contract(old_table.schema.remove_metadata())["sha256"]
            normalized = _coerce_arrow_table(
                old_table,
                schema,
                allowed_missing=allowed_missing,
            )
            staged_path = staging / source_path.relative_to(bulk_root)
            staged_path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(normalized, staged_path, compression="zstd")
            staged_file = pq.ParquetFile(staged_path)
            if not staged_file.schema_arrow.remove_metadata().equals(
                schema,
                check_metadata=True,
            ):
                raise RuntimeError(f"Staged Arrow schema mismatch: {staged_path}")
            staged_table = pq.read_table(staged_path)
            if not normalized.equals(staged_table, check_metadata=True):
                raise RuntimeError(f"Staged values/null masks differ after schema coercion: {staged_path}")
            old_sha256 = str(part.get("sha256", ""))
            new_sha256 = sha256_file(staged_path)
            normalized_record = dict(part)
            normalized_record.update(
                {
                    "rows": int(normalized.num_rows),
                    "sha256": new_sha256,
                    "size_bytes": staged_path.stat().st_size,
                    "arrow_schema_sha256": contract["sha256"],
                }
            )
            normalized_parts.append(normalized_record)
            physical_rows += int(normalized.num_rows)
            replacements.append((staged_path, source_path))
            rewrite_receipts.append(
                {
                    "path": source_path.relative_to(bulk_root).as_posix(),
                    "rows": int(normalized.num_rows),
                    "old_size_bytes": source_path.stat().st_size,
                    "new_size_bytes": staged_path.stat().st_size,
                    "old_sha256": old_sha256,
                    "old_arrow_schema_sha256": old_schema_fingerprint,
                    "new_sha256": new_sha256,
                    "declared_arrow_schema_sha256": contract["sha256"],
                    "value_preservation": {
                        "method": "exact Arrow Table.equals after safe cast to declared schema",
                        "coerced_old_equals_new": True,
                        "old_coerced_null_cells": sum(column.null_count for column in normalized.columns),
                        "new_null_cells": sum(column.null_count for column in staged_table.columns),
                    },
                }
            )
        actual = {path.resolve() for path in physical_directory.glob("*.parquet")}
        if actual != listed:
            raise RuntimeError(
                f"Manifest/physical inventory mismatch before schema normalization: {physical_directory}"
            )
        declared_rows = int(manifest.get("row_count", manifest.get("rows_written", -1)))
        if physical_rows != declared_rows:
            raise RuntimeError(f"Schema-normalization row-count mismatch: {manifest_path}")
        normalized_manifest = dict(manifest)
        normalized_manifest["parts"] = normalized_parts
        normalized_manifest["part_count"] = len(normalized_parts)
        normalized_manifest["arrow_schema"] = contract
        normalized_manifest["schema_normalized_at_utc"] = _utc_now()
        staged_manifests[manifest_path] = normalized_manifest
        return normalized_manifest

    try:
        activity_manifest_path = bulk_root / "activity_export_manifest.json"
        activity_manifest = stage_partitioned(
            activity_manifest_path,
            bulk_root,
            bulk_root / "activity_facts",
            ACTIVITY_ARROW_SCHEMA,
            allowed_missing=frozenset({"component_accessions", "component_sequences", "component_types"}),
        )
        sqlite_schema = inspect_sqlite_schema(database)
        activity_query = _activity_query(sqlite_schema)
        activity_manifest["query"] = activity_query
        activity_manifest["query_sha256"] = hashlib.sha256(activity_query.encode("utf-8")).hexdigest()

        specialized_root = bulk_root / "specialized_views"
        activity_view_names = (
            "single_protein_kd_ki",
            "single_protein_ic50_ec50_candidates",
            "herg_all_endpoints",
            "pk_adme_candidates",
            "cardiac_qt_apd_inventory",
        )
        specialized_manifests: dict[str, dict[str, Any]] = {}
        for view_name in activity_view_names:
            specialized_manifests[view_name] = stage_partitioned(
                specialized_root / f"{view_name}_manifest.json",
                specialized_root,
                specialized_root / view_name,
                ACTIVITY_ARROW_SCHEMA,
            )
        development_manifest = stage_partitioned(
            specialized_root / "molecule_development_annotations_manifest.json",
            specialized_root,
            specialized_root / "molecule_development_annotations",
            DEVELOPMENT_ARROW_SCHEMA,
        )
        specialized_manifests["molecule_development_annotations"] = development_manifest

        target_manifest_path = specialized_root / "target_components_manifest.json"
        target_manifest = load_manifest(target_manifest_path)
        target_source = (specialized_root / str(target_manifest["path"])).resolve()
        if (
            target_manifest.get("path") != "target_components.parquet"
            or target_source != (specialized_root / "target_components.parquet").resolve()
        ):
            raise RuntimeError("Target-component path is not confined to its canonical location")
        target_file = pq.ParquetFile(target_source)
        if (
            target_file.metadata is None
            or int(target_file.metadata.num_rows) != int(target_manifest.get("row_count", -1))
            or target_source.stat().st_size != int(target_manifest.get("size_bytes", -1))
            or sha256_file(target_source) != target_manifest.get("sha256")
        ):
            raise RuntimeError("Target-component pre-normalization integrity failure")
        bind_old_state(
            target_source,
            str(target_manifest["sha256"]),
            int(target_manifest["size_bytes"]),
        )
        old_target_table = pq.read_table(target_source)
        if sha256_file(target_source) != target_manifest["sha256"]:
            raise RuntimeError("Target-component bytes changed while they were staged")
        old_target_schema_fingerprint = arrow_schema_contract(old_target_table.schema.remove_metadata())[
            "sha256"
        ]
        normalized_target = _coerce_arrow_table(
            old_target_table,
            TARGET_COMPONENT_ARROW_SCHEMA,
        )
        staged_target = staging / target_source.relative_to(bulk_root)
        staged_target.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(normalized_target, staged_target, compression="zstd")
        staged_target_table = pq.read_table(staged_target)
        if not normalized_target.equals(staged_target_table, check_metadata=True):
            raise RuntimeError("Staged target-component values/null masks differ")
        target_contract = arrow_schema_contract(TARGET_COMPONENT_ARROW_SCHEMA)
        old_target_sha256 = str(target_manifest.get("sha256", ""))
        new_target_sha256 = sha256_file(staged_target)
        target_manifest.update(
            {
                "row_count": int(normalized_target.num_rows),
                "sha256": new_target_sha256,
                "size_bytes": staged_target.stat().st_size,
                "arrow_schema": target_contract,
                "arrow_schema_sha256": target_contract["sha256"],
                "schema_normalized_at_utc": _utc_now(),
            }
        )
        replacements.append((staged_target, target_source))
        rewrite_receipts.append(
            {
                "path": target_source.relative_to(bulk_root).as_posix(),
                "rows": int(normalized_target.num_rows),
                "old_size_bytes": target_source.stat().st_size,
                "new_size_bytes": staged_target.stat().st_size,
                "old_sha256": old_target_sha256,
                "old_arrow_schema_sha256": old_target_schema_fingerprint,
                "new_sha256": new_target_sha256,
                "declared_arrow_schema_sha256": target_contract["sha256"],
                "value_preservation": {
                    "method": "exact Arrow Table.equals after safe cast to declared schema",
                    "coerced_old_equals_new": True,
                    "old_coerced_null_cells": sum(column.null_count for column in normalized_target.columns),
                    "new_null_cells": sum(column.null_count for column in staged_target_table.columns),
                },
            }
        )
        staged_manifests[target_manifest_path] = target_manifest
        specialized_manifests["target_components"] = target_manifest

        receipt = {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "generated_at_utc": _utc_now(),
            "source_identity": source_identity,
            "proof_boundary": (
                "schema-only rewrite; every new Parquet table exactly equals the safely coerced "
                "old Arrow table, including null masks"
            ),
            "aggregate": {
                "parquet_files": len(rewrite_receipts),
                "rows_across_overlapping_exports": sum(int(record["rows"]) for record in rewrite_receipts),
                "old_size_bytes": sum(int(record["old_size_bytes"]) for record in rewrite_receipts),
                "new_size_bytes": sum(int(record["new_size_bytes"]) for record in rewrite_receipts),
                "all_value_preservation_checks_passed": all(
                    bool(record["value_preservation"]["coerced_old_equals_new"])
                    and int(record["value_preservation"]["old_coerced_null_cells"])
                    == int(record["value_preservation"]["new_null_cells"])
                    for record in rewrite_receipts
                ),
            },
            "arrow_schema_contracts": {
                "activity": arrow_schema_contract(ACTIVITY_ARROW_SCHEMA),
                "molecule_development_annotations": arrow_schema_contract(DEVELOPMENT_ARROW_SCHEMA),
                "target_components": target_contract,
            },
            "files": sorted(rewrite_receipts, key=lambda record: str(record["path"])),
        }
        staged_receipt = staging / "schema_normalization_receipt.json"
        _atomic_write_json(staged_receipt, receipt)
        receipt_binding = {
            "path": "schema_normalization_receipt.json",
            "sha256": sha256_file(staged_receipt),
            "size_bytes": staged_receipt.stat().st_size,
            "parquet_files": len(rewrite_receipts),
        }
        for document in staged_manifests.values():
            document["schema_normalization_receipt"] = receipt_binding
        receipt_destination = bulk_root / "schema_normalization_receipt.json"
        if receipt_destination.exists():
            raise RuntimeError("Unexpected pre-existing schema-normalization receipt")
        bind_old_state(receipt_destination, None, None)
        replacements.append((staged_receipt, receipt_destination))
        prior_summary_path = specialized_root / "specialized_views_manifest.json"
        summary = load_manifest(prior_summary_path)
        prior_views = summary.get("views")
        if not isinstance(prior_views, dict) or any(
            prior_views.get(view_name) != load_manifest(specialized_root / f"{view_name}_manifest.json")
            for view_name in specialized_manifests
        ):
            raise RuntimeError("Pre-normalization specialized summary/child manifest drift")
        _verify_schema_normalization_prestate_identity(
            source_identity,
            expected_old_states,
            activity_manifest_path,
            prior_summary_path,
        )
        summary["views"] = specialized_manifests
        summary["arrow_schema_contracts"] = {
            "activity": arrow_schema_contract(ACTIVITY_ARROW_SCHEMA),
            "molecule_development_annotations": arrow_schema_contract(DEVELOPMENT_ARROW_SCHEMA),
            "target_components": target_contract,
        }
        summary["schema_normalization_receipt"] = receipt_binding
        summary["completed_at_utc"] = _utc_now()

        json_destinations = [
            activity_manifest_path,
            *sorted(path for path in staged_manifests if path != activity_manifest_path),
        ]
        if len(json_destinations) != len(set(json_destinations)):
            raise RuntimeError("Duplicate manifest destination in schema-normalization plan")
        for manifest_path in json_destinations:
            document = staged_manifests[manifest_path]
            staged_manifest = staging / manifest_path.relative_to(bulk_root)
            _atomic_write_json(staged_manifest, document)
            replacements.append((staged_manifest, manifest_path))
        staged_summary = staging / prior_summary_path.relative_to(bulk_root)
        _atomic_write_json(staged_summary, summary)
        replacements.append((staged_summary, prior_summary_path))

        entries: list[dict[str, Any]] = []
        for staged_path, destination_path in replacements:
            if destination_path.suffix == ".parquet":
                kind = "parquet"
            elif destination_path.name == "schema_normalization_receipt.json":
                kind = "receipt"
            elif destination_path == prior_summary_path:
                kind = "summary"
            else:
                kind = "manifest"
            entries.append(
                _schema_transaction_entry(
                    bulk_root,
                    staged_path,
                    destination_path,
                    kind=kind,
                    expected_old_sha256=expected_old_states[destination_path.resolve()][0],
                    expected_old_size_bytes=expected_old_states[destination_path.resolve()][1],
                )
            )
        expected_destinations = len(rewrite_receipts) + 1 + len(staged_manifests) + 1
        if len(entries) != expected_destinations:
            raise RuntimeError("Schema-normalization transaction plan is incomplete")
        receipt_by_path = {str(record["path"]): record for record in rewrite_receipts}
        for entry in entries:
            if entry["kind"] != "parquet":
                continue
            record = receipt_by_path.get(str(entry["destination_path"]))
            if record is None or (
                entry["old_sha256"],
                entry["old_size_bytes"],
                entry["new_sha256"],
                entry["new_size_bytes"],
            ) != (
                record["old_sha256"],
                record["old_size_bytes"],
                record["new_sha256"],
                record["new_size_bytes"],
            ):
                raise RuntimeError("Parquet journal state does not match its rewrite receipt")
        started_at = json.loads(marker.read_text(encoding="utf-8"))["started_at_utc"]
        journal = {
            "schema_version": SCHEMA_VERSION,
            "status": "committing",
            "started_at_utc": started_at,
            "transaction_id": json.loads(marker.read_text(encoding="utf-8"))["transaction_id"],
            "owner": json.loads(marker.read_text(encoding="utf-8"))["owner"],
            "source_identity": source_identity,
            "staging_directory": staging.relative_to(bulk_root).as_posix(),
            "staged_part_count": len(rewrite_receipts),
            "entry_count": len(entries),
            "entries_sha256": hashlib.sha256(canonical_json(entries).encode("utf-8")).hexdigest(),
            "entries": entries,
        }
        _atomic_write_json(marker, journal)
        _fsync_directory(bulk_root)
        return _resume_schema_normalization_transaction(bulk_root)
    except Exception:
        # The marker intentionally remains: every reader/exporter fails closed
        # until the staged transaction is inspected and explicitly recovered.
        raise


def export_chembl37_activity_facts(
    database_path: str | os.PathLike[str],
    interim_root: str | os.PathLike[str],
    *,
    chunk_size: int = 200_000,
) -> dict[str, Any]:
    """Export all ChEMBL source activity assertions as checkpointed parts."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    database = Path(database_path).resolve()
    _verify_sqlite_file(database)
    bulk_root = Path(interim_root).resolve() / "chembl_37_bulk"
    _assert_schema_normalization_idle(bulk_root)
    destination = bulk_root / "activity_facts"
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination.parent / "activity_export_manifest.json"
    checkpoint_path = destination.parent / "activity_export_checkpoint.json"
    database_sha256 = sha256_file(database)
    schema = inspect_sqlite_schema(database)
    query = _activity_query(schema)
    query_sha256 = hashlib.sha256(query.encode("utf-8")).hexdigest()
    schema_contract = arrow_schema_contract(ACTIVITY_ARROW_SCHEMA)
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            existing.get("database_sha256") == database_sha256
            and existing.get("query_sha256") == query_sha256
        ):
            _verify_manifest_arrow_schema(
                existing,
                destination.parent,
                ACTIVITY_ARROW_SCHEMA,
                context="activity_facts",
            )
            _verify_schema_normalization_receipt(existing, bulk_root)
            listed = {
                (destination.parent / str(part["path"])).resolve() for part in existing.get("parts", [])
            }
            actual = {path.resolve() for path in destination.glob("*.parquet")}
            if actual != listed:
                raise RuntimeError("Unmanifested or missing ChEMBL activity parts exist")
            return existing
        raise RuntimeError("Existing ChEMBL activity export manifest does not match database/query contract")

    checkpoint: dict[str, Any] = {
        "database_sha256": database_sha256,
        "query_sha256": query_sha256,
        "last_activity_id": 0,
        "rows_written": 0,
        "parts": [],
        "arrow_schema": schema_contract,
    }
    if checkpoint_path.exists():
        candidate = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if (
            candidate.get("database_sha256") == database_sha256
            and candidate.get("query_sha256") == query_sha256
            and candidate.get("arrow_schema") == schema_contract
        ):
            checkpoint = candidate
            _verify_manifest_arrow_schema(
                checkpoint,
                destination.parent,
                ACTIVITY_ARROW_SCHEMA,
                context="activity_facts checkpoint",
            )
        else:
            raise RuntimeError("Existing ChEMBL activity checkpoint does not match database/query contract")
    elif any(destination.glob("*.parquet")):
        raise RuntimeError("Untracked ChEMBL activity parts exist; refusing to overwrite")

    connection = sqlite3.connect(f"file:{database.as_posix()}?mode=ro&immutable=1", uri=True)
    try:
        total_rows = int(connection.execute("SELECT COUNT(*) FROM activities").fetchone()[0])
        last_activity_id = int(checkpoint["last_activity_id"])
        while True:
            frame = pd.read_sql_query(query, connection, params=(last_activity_id, chunk_size))
            if frame.empty:
                break
            frame = frame.sort_values("activity_id", kind="stable").reset_index(drop=True)
            first_id = int(frame["activity_id"].iloc[0])
            final_id = int(frame["activity_id"].iloc[-1])
            if first_id <= last_activity_id or frame["activity_id"].duplicated().any():
                raise RuntimeError("Non-monotonic or duplicated activity_id in ChEMBL bulk export")
            part_name = f"activities_{first_id:09d}_{final_id:09d}.parquet"
            part_path = destination / part_name
            if part_path.exists():
                raise RuntimeError(f"Unexpected unmanifested export part already exists: {part_path}")
            _write_frame_with_schema(frame, part_path, ACTIVITY_ARROW_SCHEMA)
            part_record = {
                "path": part_path.relative_to(destination.parent).as_posix(),
                "rows": len(frame),
                "first_activity_id": first_id,
                "last_activity_id": final_id,
                "sha256": sha256_file(part_path),
                "size_bytes": part_path.stat().st_size,
                "arrow_schema_sha256": schema_contract["sha256"],
            }
            checkpoint["parts"].append(part_record)
            checkpoint["last_activity_id"] = final_id
            checkpoint["rows_written"] = int(checkpoint["rows_written"]) + len(frame)
            checkpoint["updated_at_utc"] = _utc_now()
            _atomic_write_json(checkpoint_path, checkpoint)
            last_activity_id = final_id
    finally:
        connection.close()

    if int(checkpoint["rows_written"]) != total_rows:
        raise RuntimeError(
            f"ChEMBL bulk row-count mismatch: wrote {checkpoint['rows_written']}, expected {total_rows}"
        )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "export_version": "chembl37-source-activity-assertions-1.0",
        "source_name": "ChEMBL",
        "source_version": "ChEMBL_37",
        "source_url": CHEMBL37_SQLITE_URL,
        "database_path": database.name,
        "database_sha256": database_sha256,
        "query_sha256": query_sha256,
        "query": query,
        "total_activity_rows": total_rows,
        "rows_written": int(checkpoint["rows_written"]),
        "part_count": len(checkpoint["parts"]),
        "parts": checkpoint["parts"],
        "arrow_schema": schema_contract,
        "row_order": "activity_id ascending",
        "selection_policy": "all rows in activities; no endpoint/value/target filtering",
        "field_policy": (
            "source activity/assay/identity/target/document fields; compound_properties and "
            "source-supplied derived pchembl_value excluded; canonical QC determines eligibility"
        ),
        "excluded_tables": ["compound_properties"],
        "excluded_fields": ["activities.pchembl_value"],
        "license": "CC BY-SA 3.0",
        "citation": CHEMBL37_CITATION,
        "completed_at_utc": _utc_now(),
    }
    _atomic_write_json(manifest_path, manifest)
    checkpoint_path.unlink(missing_ok=True)
    return manifest


def bulk_integration_contract() -> dict[str, Any]:
    """Return the exact callable/CLI contract for lead integration."""

    return {
        "archive": {
            "url": CHEMBL37_SQLITE_URL,
            "bytes": CHEMBL37_ARCHIVE_BYTES,
            "sha256": CHEMBL37_ARCHIVE_SHA256,
        },
        "python_api": [
            "snapshot_chembl37_release_metadata(raw_root)",
            "assemble_archive_parts(parts, destination)",
            "verify_chembl37_archive(archive_path)",
            "stage_chembl37_archive(archive_path, raw_root, move=False)",
            "extract_chembl37_sqlite(archive_path, raw_root)",
            "export_chembl37_activity_facts(database_path, interim_root, chunk_size=200000)",
            "export_chembl37_specialized_views(database_path, interim_root, chunk_size=200000)",
            "finalize_chembl37_specialized_summary(database_path, interim_root)",
            "normalize_chembl37_export_schemas(database_path, interim_root)",
        ],
        "deterministic_command": (
            "python -m menin_discovery.platform_data_bulk all "
            "--archive research/data/platform/raw/chembl_37_bulk/chembl_37_sqlite.tar.gz "
            "--raw-root research/data/platform/raw --interim-root research/data/platform/interim "
            "--chunk-size 200000"
        ),
        "output_schema": {
            "raw": [
                "chembl_37_bulk/chembl_37_sqlite.tar.gz",
                "chembl_37_bulk/archive_manifest.json",
                "chembl_37_bulk/release_metadata/*",
                "chembl_37_bulk/extracted/chembl_37.db",
                "chembl_37_bulk/extracted/extraction_manifest.json",
            ],
            "interim": [
                "chembl_37_bulk/activity_facts/activities_<first>_<last>.parquet",
                "chembl_37_bulk/activity_export_manifest.json",
                "chembl_37_bulk/specialized_views/*",
            ],
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify and export the official ChEMBL_37 SQLite release")
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--archive", required=True, type=Path)
    finalize_parser = subparsers.add_parser("finalize-specialized")
    finalize_parser.add_argument("--database", required=True, type=Path)
    finalize_parser.add_argument("--interim-root", required=True, type=Path)
    normalize_parser = subparsers.add_parser("normalize-export-schemas")
    normalize_parser.add_argument("--database", required=True, type=Path)
    normalize_parser.add_argument("--interim-root", required=True, type=Path)
    all_parser = subparsers.add_parser("all")
    all_parser.add_argument("--archive", required=True, type=Path)
    all_parser.add_argument("--raw-root", required=True, type=Path)
    all_parser.add_argument("--interim-root", required=True, type=Path)
    all_parser.add_argument("--chunk-size", type=int, default=200_000)
    arguments = parser.parse_args(argv)
    if arguments.command == "verify":
        print(json.dumps(verify_chembl37_archive(arguments.archive), indent=2, sort_keys=True))
        return 0
    if arguments.command == "finalize-specialized":
        print(
            json.dumps(
                finalize_chembl37_specialized_summary(
                    arguments.database,
                    arguments.interim_root,
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if arguments.command == "normalize-export-schemas":
        print(
            json.dumps(
                normalize_chembl37_export_schemas(
                    arguments.database,
                    arguments.interim_root,
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    snapshot_chembl37_release_metadata(arguments.raw_root)
    staged, _ = stage_chembl37_archive(arguments.archive, arguments.raw_root)
    database, _ = extract_chembl37_sqlite(staged, arguments.raw_root)
    activity_manifest = export_chembl37_activity_facts(
        database,
        arguments.interim_root,
        chunk_size=arguments.chunk_size,
    )
    specialized_manifest = export_chembl37_specialized_views(
        database,
        arguments.interim_root,
        chunk_size=arguments.chunk_size,
    )
    print(
        json.dumps(
            {"activity_facts": activity_manifest, "specialized_views": specialized_manifest},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by CLI smoke tests.
    raise SystemExit(main())


__all__ = [
    "ArchivePart",
    "CHEMBL37_ARCHIVE_BYTES",
    "CHEMBL37_ARCHIVE_SHA256",
    "CHEMBL37_CITATION",
    "CHEMBL37_SQLITE_URL",
    "assemble_archive_parts",
    "bulk_integration_contract",
    "export_chembl37_activity_facts",
    "export_chembl37_specialized_views",
    "exact_manifest_activity_overlap",
    "extract_chembl37_sqlite",
    "inspect_sqlite_schema",
    "finalize_chembl37_specialized_summary",
    "normalize_chembl37_export_schemas",
    "main",
    "snapshot_chembl37_release_metadata",
    "stage_chembl37_archive",
    "verify_chembl37_archive",
]
