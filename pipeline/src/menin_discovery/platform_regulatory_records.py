"""Deterministic record-level normalization of frozen Drugs@FDA metadata.

Approval, marketing, submission, action, and document records are regulatory
metadata.  This module never interprets them as efficacy, safety, QT, PK,
binding, causality, negative outcomes, or model labels.  Malformed, orphaned,
blank-key, and duplicate-key records are explicitly quarantined.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import stat
import tempfile
import zipfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from .platform_external_normalization import (
    arrow_schema_sha256,
    canonical_json_bytes,
    document_with_sha256,
    load_and_verify_input,
    sha256_file,
    verify_document_sha256,
)

SCHEMA_VERSION = "platform-regulatory-record-candidates/1.0"
PARSER_VERSION = "platform_regulatory_records/1.1"
SOURCE_SCHEMA_VERSION = "platform-external-acquisition/1.0"
SOURCE_ID = "drugs_at_fda_bulk"
SOURCE_MANIFEST = "drugs_at_fda_bulk_manifest.json"
ARCHIVE_NAME = "datdaf20260804.zip"
BATCH_ROWS = 4096


class RegulatoryRecordError(RuntimeError):
    """Raised when source, schema, linkage, or zero-label contracts fail."""


TABLE_COLUMNS: dict[str, tuple[str, ...]] = {
    "ActionTypes_Lookup.txt": (
        "ActionTypes_LookupID",
        "ActionTypes_LookupDescription",
        "SupplCategoryLevel1Code",
        "SupplCategoryLevel2Code",
    ),
    "ApplicationDocs.txt": (
        "ApplicationDocsID",
        "ApplicationDocsTypeID",
        "ApplNo",
        "SubmissionType",
        "SubmissionNo",
        "ApplicationDocsTitle",
        "ApplicationDocsURL",
        "ApplicationDocsDate",
    ),
    "Applications.txt": ("ApplNo", "ApplType", "ApplPublicNotes", "SponsorName"),
    "ApplicationsDocsType_Lookup.txt": (
        "ApplicationDocsType_Lookup_ID",
        "ApplicationDocsType_Lookup_Description",
    ),
    "Join_Submission_ActionTypes_Lookup.txt": (
        "SubmissionType",
        "j_submissionActionTypeID",
        "ApplNo",
        "SubmissionNo",
        "ActionTypes_LookupID",
    ),
    "MarketingStatus.txt": ("MarketingStatusID", "ApplNo", "ProductNo"),
    "MarketingStatus_Lookup.txt": ("MarketingStatusID", "MarketingStatusDescription"),
    "Products.txt": (
        "ApplNo",
        "ProductNo",
        "Form",
        "Strength",
        "ReferenceDrug",
        "DrugName",
        "ActiveIngredient",
        "ReferenceStandard",
    ),
    "SubmissionClass_Lookup.txt": (
        "SubmissionClassCodeID",
        "SubmissionClassCode",
        "SubmissionClassCodeDescription",
    ),
    "SubmissionPropertyType.txt": (
        "ApplNo",
        "SubmissionType",
        "SubmissionNo",
        "SubmissionPropertyTypeCode",
        "SubmissionPropertyTypeID",
    ),
    "Submissions.txt": (
        "ApplNo",
        "SubmissionClassCodeID",
        "SubmissionType",
        "SubmissionNo",
        "SubmissionStatus",
        "SubmissionStatusDate",
        "SubmissionsPublicNotes",
        "ReviewPriority",
    ),
    "TE.txt": ("ApplNo", "ProductNo", "MarketingStatusID", "TECode"),
}

TABLE_ENCODINGS = {
    "ApplicationDocs.txt": "cp1252",
    "Submissions.txt": "cp1252",
}


def _common_fields() -> list[pa.Field]:
    return [
        pa.field("source_id", pa.string(), nullable=False),
        pa.field("source_archive", pa.string(), nullable=False),
        pa.field("source_member", pa.string(), nullable=False),
        pa.field("source_row_number_one_based", pa.int64(), nullable=False),
        pa.field("source_field_map_sha256", pa.string(), nullable=False),
    ]


APPLICATION_SCHEMA = pa.schema(
    _common_fields()
    + [
        pa.field("application_number", pa.string(), nullable=False),
        pa.field("application_type", pa.string()),
        pa.field("public_notes", pa.string()),
        pa.field("sponsor_name", pa.string()),
        pa.field("primary_key_state", pa.string(), nullable=False),
        pa.field("product_record_count", pa.int64(), nullable=False),
        pa.field("submission_record_count", pa.int64(), nullable=False),
        pa.field("document_record_count", pa.int64(), nullable=False),
        pa.field("evidence_semantics", pa.string(), nullable=False),
        pa.field("model_label_admitted", pa.bool_(), nullable=False),
    ]
)

PRODUCT_SCHEMA = pa.schema(
    _common_fields()
    + [
        pa.field("application_number", pa.string(), nullable=False),
        pa.field("product_number", pa.string(), nullable=False),
        pa.field("dosage_form_route_raw", pa.string()),
        pa.field("strength_raw", pa.string()),
        pa.field("reference_drug_raw", pa.string()),
        pa.field("drug_name_raw", pa.string()),
        pa.field("active_ingredient_raw", pa.string()),
        pa.field("reference_standard_raw", pa.string()),
        pa.field("primary_key_state", pa.string(), nullable=False),
        pa.field("application_link_state", pa.string(), nullable=False),
        pa.field("marketing_statuses_json", pa.string(), nullable=False),
        pa.field("therapeutic_equivalence_codes_json", pa.string(), nullable=False),
        pa.field("ingredient_component_count", pa.int64(), nullable=False),
        pa.field("molecule_identity_state", pa.string(), nullable=False),
        pa.field("evidence_semantics", pa.string(), nullable=False),
        pa.field("model_label_admitted", pa.bool_(), nullable=False),
    ]
)

SUBMISSION_SCHEMA = pa.schema(
    _common_fields()
    + [
        pa.field("application_number", pa.string(), nullable=False),
        pa.field("submission_type", pa.string(), nullable=False),
        pa.field("submission_number", pa.string(), nullable=False),
        pa.field("submission_class_code_id", pa.string()),
        pa.field("submission_class_code", pa.string()),
        pa.field("submission_class_description", pa.string()),
        pa.field("submission_status_raw", pa.string()),
        pa.field("submission_status_date_raw", pa.string()),
        pa.field("public_notes", pa.string()),
        pa.field("review_priority_raw", pa.string()),
        pa.field("primary_key_state", pa.string(), nullable=False),
        pa.field("application_link_state", pa.string(), nullable=False),
        pa.field("action_record_count", pa.int64(), nullable=False),
        pa.field("property_record_count", pa.int64(), nullable=False),
        pa.field("document_record_count", pa.int64(), nullable=False),
        pa.field("absence_semantics", pa.string(), nullable=False),
        pa.field("evidence_semantics", pa.string(), nullable=False),
        pa.field("model_label_admitted", pa.bool_(), nullable=False),
    ]
)

ACTION_SCHEMA = pa.schema(
    _common_fields()
    + [
        pa.field("submission_action_id", pa.string(), nullable=False),
        pa.field("application_number", pa.string()),
        pa.field("submission_type", pa.string()),
        pa.field("submission_number", pa.string()),
        pa.field("action_type_id", pa.string()),
        pa.field("action_description", pa.string()),
        pa.field("supplement_category_level1", pa.string()),
        pa.field("supplement_category_level2", pa.string()),
        pa.field("primary_key_state", pa.string(), nullable=False),
        pa.field("submission_link_state", pa.string(), nullable=False),
        pa.field("action_lookup_link_state", pa.string(), nullable=False),
        pa.field("evidence_semantics", pa.string(), nullable=False),
        pa.field("model_label_admitted", pa.bool_(), nullable=False),
    ]
)

DOCUMENT_SCHEMA = pa.schema(
    _common_fields()
    + [
        pa.field("application_document_id", pa.string(), nullable=False),
        pa.field("document_type_id", pa.string()),
        pa.field("document_type_description", pa.string()),
        pa.field("application_number", pa.string()),
        pa.field("submission_type", pa.string()),
        pa.field("submission_number", pa.string()),
        pa.field("document_title_raw", pa.string()),
        pa.field("document_url", pa.string()),
        pa.field("document_date_raw", pa.string()),
        pa.field("primary_key_state", pa.string(), nullable=False),
        pa.field("application_link_state", pa.string(), nullable=False),
        pa.field("submission_link_state", pa.string(), nullable=False),
        pa.field("document_type_link_state", pa.string(), nullable=False),
        pa.field("document_content_downloaded", pa.bool_(), nullable=False),
        pa.field("evidence_semantics", pa.string(), nullable=False),
        pa.field("model_label_admitted", pa.bool_(), nullable=False),
    ]
)

INGREDIENT_SCHEMA = pa.schema(
    _common_fields()
    + [
        pa.field("application_number", pa.string(), nullable=False),
        pa.field("product_number", pa.string(), nullable=False),
        pa.field("ingredient_component_index_one_based", pa.int64(), nullable=False),
        pa.field("active_ingredient_raw", pa.string()),
        pa.field("ingredient_component_exact", pa.string(), nullable=False),
        pa.field("ingredient_candidate_key", pa.string(), nullable=False),
        pa.field("split_rule", pa.string(), nullable=False),
        pa.field("product_link_state", pa.string(), nullable=False),
        pa.field("application_link_state", pa.string(), nullable=False),
        pa.field("molecule_link_state", pa.string(), nullable=False),
        pa.field("evidence_semantics", pa.string(), nullable=False),
        pa.field("model_label_admitted", pa.bool_(), nullable=False),
    ]
)

ANOMALY_SCHEMA = pa.schema(
    [
        pa.field("source_id", pa.string(), nullable=False),
        pa.field("source_archive", pa.string(), nullable=False),
        pa.field("source_member", pa.string(), nullable=False),
        pa.field("source_row_number_one_based", pa.int64(), nullable=False),
        pa.field("anomaly_type", pa.string(), nullable=False),
        pa.field("source_key_json", pa.string(), nullable=False),
        pa.field("source_record_sha256", pa.string(), nullable=False),
        pa.field("details", pa.string(), nullable=False),
        pa.field("quarantine_state", pa.string(), nullable=False),
        pa.field("negative_label_inferred_from_absence", pa.bool_(), nullable=False),
        pa.field("model_label_admitted", pa.bool_(), nullable=False),
    ]
)


def canonical_json(value: Any) -> str:
    return canonical_json_bytes(value).decode("utf-8")


def normalize_submission_type(value: str) -> str:
    return " ".join(value.split())


def candidate_ingredient_components(value: str) -> list[str]:
    """Split only the source's semicolon-delimited projection; infer no chemistry."""

    return [" ".join(item.split()) for item in value.split(";") if item.strip()]


def classify_link(key: tuple[str, ...], target: set[tuple[str, ...]]) -> str:
    if not all(part.strip() for part in key):
        return "blank_key_quarantine"
    return "exact_source_key_match" if key in target else "orphan_source_key_quarantine"


def absence_semantics(count: int) -> str:
    if count < 0:
        raise RegulatoryRecordError("Negative linked-record count")
    return "linked_records_present" if count else "no_linked_record_in_snapshot_unknown_not_negative"


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


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _safe_directory(value: str | os.PathLike[str], *, context: str) -> Path:
    path = Path(value)
    if path.is_symlink() or not path.is_dir():
        raise RegulatoryRecordError(f"Missing or symlinked {context}: {path}")
    return path.resolve()


def _safe_relative(value: Any, *, context: str) -> PurePosixPath:
    raw = str(value)
    path = PurePosixPath(raw)
    if not raw or path.is_absolute() or ".." in path.parts:
        raise RegulatoryRecordError(f"Unsafe {context} path: {raw}")
    return path


def _runtime_code_sha256() -> str:
    return sha256_file(Path(__file__).resolve())


def _closed_regular_files(root: Path, *, context: str) -> set[str]:
    """Return a closed regular-file inventory and reject links/special files."""

    observed: set[str] = set()
    for directory, directory_names, file_names in os.walk(root, followlinks=False):
        current = Path(directory)
        for name in directory_names:
            candidate = current / name
            if candidate.is_symlink():
                raise RegulatoryRecordError(f"Symlinked directory in {context}: {candidate}")
        for name in file_names:
            candidate = current / name
            mode = candidate.lstat().st_mode
            if not stat.S_ISREG(mode):
                raise RegulatoryRecordError(f"Non-regular file in {context}: {candidate}")
            observed.add(candidate.relative_to(root).as_posix())
    return observed


def _table_contracts(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    table = manifest.get("archive_member_table")
    if not isinstance(table, dict):
        raise RegulatoryRecordError("Archive table contract missing")
    members = table.get("members")
    if not isinstance(members, list):
        raise RegulatoryRecordError("Archive member contracts missing")
    result = {str(item["archive_member_path"]): dict(item) for item in members if isinstance(item, dict)}
    if set(result) != set(TABLE_COLUMNS):
        raise RegulatoryRecordError("Archive member/schema set drift")
    for name, expected in TABLE_COLUMNS.items():
        if tuple(result[name].get("columns", ())) != expected:
            raise RegulatoryRecordError(f"Source schema drift: {name}")
    return result


def _rows(
    archive: zipfile.ZipFile, member: str, contract: Mapping[str, Any]
) -> Iterator[tuple[int, dict[str, str] | None, dict[str, Any]]]:
    encoding = TABLE_ENCODINGS.get(member, "utf-8-sig")
    with archive.open(member, "r") as binary:
        text = io.TextIOWrapper(binary, encoding=encoding, newline="")
        reader = csv.reader(text, delimiter="\t")
        try:
            header = next(reader)
        except StopIteration as error:
            raise RegulatoryRecordError(f"Empty source table: {member}") from error
        expected = list(TABLE_COLUMNS[member])
        if header != expected:
            raise RegulatoryRecordError(f"Runtime source schema drift: {member}")
        count = 0
        malformed = 0
        for row_number, fields in enumerate(reader, start=1):
            count += 1
            raw_digest = hashlib.sha256(canonical_json_bytes(fields)).hexdigest()
            if len(fields) != len(expected):
                malformed += 1
                yield (
                    row_number,
                    None,
                    {
                        "observed_width": len(fields),
                        "expected_width": len(expected),
                        "source_record_sha256": raw_digest,
                    },
                )
            else:
                mapping = dict(zip(expected, fields, strict=True))
                yield (
                    row_number,
                    mapping,
                    {"source_record_sha256": hashlib.sha256(canonical_json_bytes(mapping)).hexdigest()},
                )
        if count != int(contract.get("data_row_count", -1)):
            raise RegulatoryRecordError(f"Source row-count drift: {member}")
        if malformed != int(contract.get("malformed_width_rows", -1)):
            raise RegulatoryRecordError(f"Source malformed-row drift: {member}")


def _field_map(row: Mapping[str, str]) -> str:
    return hashlib.sha256(canonical_json_bytes(dict(row))).hexdigest()


def _base(member: str, row_number: int, row: Mapping[str, str]) -> dict[str, Any]:
    return {
        "source_id": SOURCE_ID,
        "source_archive": ARCHIVE_NAME,
        "source_member": member,
        "source_row_number_one_based": row_number,
        "source_field_map_sha256": _field_map(row),
    }


def _primary_state(key: tuple[str, ...], seen: set[tuple[str, ...]]) -> str:
    if not all(part.strip() for part in key):
        return "blank_primary_key_quarantine"
    if key in seen:
        return "duplicate_primary_key_quarantine"
    seen.add(key)
    return "unique_primary_key"


def _scan_context(archive: zipfile.ZipFile, contracts: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    applications: set[tuple[str, ...]] = set()
    products: set[tuple[str, ...]] = set()
    submissions: set[tuple[str, ...]] = set()
    app_counts: dict[str, Counter[str]] = defaultdict(Counter)
    submission_counts: dict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)
    marketing: dict[tuple[str, str], list[str]] = defaultdict(list)
    te_codes: dict[tuple[str, str], list[str]] = defaultdict(list)
    action_lookup: dict[str, tuple[str, str, str]] = {}
    doc_lookup: dict[str, str] = {}
    submission_class: dict[str, tuple[str, str]] = {}
    marketing_lookup: dict[str, str] = {}

    for member, key_columns, target in (
        ("Applications.txt", ("ApplNo",), applications),
        ("Products.txt", ("ApplNo", "ProductNo"), products),
        ("Submissions.txt", ("ApplNo", "SubmissionType", "SubmissionNo"), submissions),
    ):
        for _, row, _ in _rows(archive, member, contracts[member]):
            if row is not None:
                key = tuple(normalize_submission_type(row[column]) for column in key_columns)
                if all(key):
                    target.add(key)
                if member == "Products.txt":
                    app_counts[key[0]]["products"] += 1
                elif member == "Submissions.txt":
                    app_counts[key[0]]["submissions"] += 1

    for _, row, _ in _rows(archive, "ActionTypes_Lookup.txt", contracts["ActionTypes_Lookup.txt"]):
        assert row is not None
        action_lookup[row["ActionTypes_LookupID"]] = (
            row["ActionTypes_LookupDescription"],
            row["SupplCategoryLevel1Code"],
            row["SupplCategoryLevel2Code"],
        )
    for _, row, _ in _rows(
        archive, "ApplicationsDocsType_Lookup.txt", contracts["ApplicationsDocsType_Lookup.txt"]
    ):
        assert row is not None
        doc_lookup[row["ApplicationDocsType_Lookup_ID"]] = row["ApplicationDocsType_Lookup_Description"]
    for _, row, _ in _rows(archive, "SubmissionClass_Lookup.txt", contracts["SubmissionClass_Lookup.txt"]):
        assert row is not None
        submission_class[row["SubmissionClassCodeID"]] = (
            row["SubmissionClassCode"],
            row["SubmissionClassCodeDescription"],
        )
    for _, row, _ in _rows(archive, "MarketingStatus_Lookup.txt", contracts["MarketingStatus_Lookup.txt"]):
        assert row is not None
        marketing_lookup[row["MarketingStatusID"]] = row["MarketingStatusDescription"]

    for _, row, _ in _rows(archive, "MarketingStatus.txt", contracts["MarketingStatus.txt"]):
        assert row is not None
        key = (row["ApplNo"].strip(), row["ProductNo"].strip())
        marketing[key].append(
            marketing_lookup.get(row["MarketingStatusID"], f"UNRESOLVED:{row['MarketingStatusID']}")
        )
    for _, row, _ in _rows(archive, "TE.txt", contracts["TE.txt"]):
        assert row is not None
        key = (row["ApplNo"].strip(), row["ProductNo"].strip())
        if row["TECode"].strip():
            te_codes[key].append(row["TECode"].strip())
    for _, row, _ in _rows(
        archive,
        "Join_Submission_ActionTypes_Lookup.txt",
        contracts["Join_Submission_ActionTypes_Lookup.txt"],
    ):
        assert row is not None
        key = (
            row["ApplNo"].strip(),
            normalize_submission_type(row["SubmissionType"]),
            row["SubmissionNo"].strip(),
        )
        submission_counts[key]["actions"] += 1
    for _, row, _ in _rows(archive, "SubmissionPropertyType.txt", contracts["SubmissionPropertyType.txt"]):
        assert row is not None
        key = (
            row["ApplNo"].strip(),
            normalize_submission_type(row["SubmissionType"]),
            row["SubmissionNo"].strip(),
        )
        submission_counts[key]["properties"] += 1
    for _, row, _ in _rows(archive, "ApplicationDocs.txt", contracts["ApplicationDocs.txt"]):
        if row is None:
            continue
        app = row["ApplNo"].strip()
        key = (app, normalize_submission_type(row["SubmissionType"]), row["SubmissionNo"].strip())
        app_counts[app]["documents"] += 1
        submission_counts[key]["documents"] += 1

    return {
        "applications": applications,
        "products": products,
        "submissions": submissions,
        "app_counts": app_counts,
        "submission_counts": submission_counts,
        "marketing": marketing,
        "te_codes": te_codes,
        "action_lookup": action_lookup,
        "doc_lookup": doc_lookup,
        "submission_class": submission_class,
    }


def _write_parquet(path: Path, schema: pa.Schema, rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    writer = pq.ParquetWriter(path, schema, compression="zstd", use_dictionary=True, write_statistics=True)
    count = 0
    buffer: list[dict[str, Any]] = []
    try:
        for row in rows:
            buffer.append(row)
            if len(buffer) == BATCH_ROWS:
                writer.write_table(pa.Table.from_pylist(buffer, schema=schema), row_group_size=BATCH_ROWS)
                count += len(buffer)
                buffer.clear()
        if buffer:
            writer.write_table(pa.Table.from_pylist(buffer, schema=schema), row_group_size=BATCH_ROWS)
            count += len(buffer)
    finally:
        writer.close()
    return {
        "path": path.name,
        "rows": count,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "arrow_schema_sha256": arrow_schema_sha256(schema),
    }


def _applications(
    archive: zipfile.ZipFile, contract: Mapping[str, Any], context: Mapping[str, Any]
) -> Iterator[dict[str, Any]]:
    seen: set[tuple[str, ...]] = set()
    for row_number, row, _ in _rows(archive, "Applications.txt", contract):
        assert row is not None
        app = row["ApplNo"].strip()
        counts = context["app_counts"].get(app, Counter())
        yield {
            **_base("Applications.txt", row_number, row),
            "application_number": app,
            "application_type": row["ApplType"].strip() or None,
            "public_notes": row["ApplPublicNotes"].strip() or None,
            "sponsor_name": row["SponsorName"].strip() or None,
            "primary_key_state": _primary_state((app,), seen),
            "product_record_count": counts["products"],
            "submission_record_count": counts["submissions"],
            "document_record_count": counts["documents"],
            "evidence_semantics": "regulatory_application_identity_not_efficacy_safety_qt_pk_or_activity",
            "model_label_admitted": False,
        }


def _products(
    archive: zipfile.ZipFile, contract: Mapping[str, Any], context: Mapping[str, Any]
) -> Iterator[dict[str, Any]]:
    seen: set[tuple[str, ...]] = set()
    for row_number, row, _ in _rows(archive, "Products.txt", contract):
        assert row is not None
        app, product = row["ApplNo"].strip(), row["ProductNo"].strip()
        key = (app, product)
        ingredients = candidate_ingredient_components(row["ActiveIngredient"])
        yield {
            **_base("Products.txt", row_number, row),
            "application_number": app,
            "product_number": product,
            "dosage_form_route_raw": row["Form"].strip() or None,
            "strength_raw": row["Strength"].strip() or None,
            "reference_drug_raw": row["ReferenceDrug"].strip() or None,
            "drug_name_raw": row["DrugName"].strip() or None,
            "active_ingredient_raw": row["ActiveIngredient"].strip() or None,
            "reference_standard_raw": row["ReferenceStandard"].strip() or None,
            "primary_key_state": _primary_state(key, seen),
            "application_link_state": classify_link((app,), context["applications"]),
            "marketing_statuses_json": canonical_json(sorted(context["marketing"].get(key, []))),
            "therapeutic_equivalence_codes_json": canonical_json(sorted(context["te_codes"].get(key, []))),
            "ingredient_component_count": len(ingredients),
            "molecule_identity_state": "not_attempted_raw_ingredient_string_only",
            "evidence_semantics": "regulatory_product_formulation_metadata_not_molecular_or_clinical_label",
            "model_label_admitted": False,
        }


def _submissions(
    archive: zipfile.ZipFile, contract: Mapping[str, Any], context: Mapping[str, Any]
) -> Iterator[dict[str, Any]]:
    seen: set[tuple[str, ...]] = set()
    for row_number, row, _ in _rows(archive, "Submissions.txt", contract):
        assert row is not None
        key = (
            row["ApplNo"].strip(),
            normalize_submission_type(row["SubmissionType"]),
            row["SubmissionNo"].strip(),
        )
        counts = context["submission_counts"].get(key, Counter())
        class_value = context["submission_class"].get(row["SubmissionClassCodeID"], (None, None))
        yield {
            **_base("Submissions.txt", row_number, row),
            "application_number": key[0],
            "submission_type": key[1],
            "submission_number": key[2],
            "submission_class_code_id": row["SubmissionClassCodeID"].strip() or None,
            "submission_class_code": class_value[0],
            "submission_class_description": class_value[1],
            "submission_status_raw": row["SubmissionStatus"].strip() or None,
            "submission_status_date_raw": row["SubmissionStatusDate"].strip() or None,
            "public_notes": row["SubmissionsPublicNotes"].strip() or None,
            "review_priority_raw": row["ReviewPriority"].strip() or None,
            "primary_key_state": _primary_state(key, seen),
            "application_link_state": classify_link((key[0],), context["applications"]),
            "action_record_count": counts["actions"],
            "property_record_count": counts["properties"],
            "document_record_count": counts["documents"],
            "absence_semantics": absence_semantics(sum(counts.values())),
            "evidence_semantics": "regulatory_submission_state_not_efficacy_safety_qt_pk_causality_or_activity",
            "model_label_admitted": False,
        }


def _actions(
    archive: zipfile.ZipFile, contract: Mapping[str, Any], context: Mapping[str, Any]
) -> Iterator[dict[str, Any]]:
    seen: set[tuple[str, ...]] = set()
    member = "Join_Submission_ActionTypes_Lookup.txt"
    for row_number, row, _ in _rows(archive, member, contract):
        assert row is not None
        submission_key = (
            row["ApplNo"].strip(),
            normalize_submission_type(row["SubmissionType"]),
            row["SubmissionNo"].strip(),
        )
        action_id = row["ActionTypes_LookupID"].strip()
        lookup = context["action_lookup"].get(action_id, (None, None, None))
        primary = (row["j_submissionActionTypeID"].strip(),)
        yield {
            **_base(member, row_number, row),
            "submission_action_id": primary[0],
            "application_number": submission_key[0] or None,
            "submission_type": submission_key[1] or None,
            "submission_number": submission_key[2] or None,
            "action_type_id": action_id or None,
            "action_description": lookup[0],
            "supplement_category_level1": lookup[1],
            "supplement_category_level2": lookup[2],
            "primary_key_state": _primary_state(primary, seen),
            "submission_link_state": classify_link(submission_key, context["submissions"]),
            "action_lookup_link_state": (
                "exact_source_key_match"
                if action_id in context["action_lookup"]
                else "blank_or_orphan_lookup_quarantine"
            ),
            "evidence_semantics": "regulatory_action_category_not_clinical_or_molecular_outcome",
            "model_label_admitted": False,
        }


def _documents(
    archive: zipfile.ZipFile, contract: Mapping[str, Any], context: Mapping[str, Any]
) -> Iterator[dict[str, Any]]:
    seen: set[tuple[str, ...]] = set()
    member = "ApplicationDocs.txt"
    for row_number, row, _ in _rows(archive, member, contract):
        if row is None:
            continue
        app = row["ApplNo"].strip()
        submission_key = (
            app,
            normalize_submission_type(row["SubmissionType"]),
            row["SubmissionNo"].strip(),
        )
        doc_type = row["ApplicationDocsTypeID"].strip()
        primary = (row["ApplicationDocsID"].strip(),)
        yield {
            **_base(member, row_number, row),
            "application_document_id": primary[0],
            "document_type_id": doc_type or None,
            "document_type_description": context["doc_lookup"].get(doc_type),
            "application_number": app or None,
            "submission_type": submission_key[1] or None,
            "submission_number": submission_key[2] or None,
            "document_title_raw": row["ApplicationDocsTitle"].strip() or None,
            "document_url": row["ApplicationDocsURL"].strip() or None,
            "document_date_raw": row["ApplicationDocsDate"].strip() or None,
            "primary_key_state": _primary_state(primary, seen),
            "application_link_state": classify_link((app,), context["applications"]),
            "submission_link_state": classify_link(submission_key, context["submissions"]),
            "document_type_link_state": (
                "exact_source_key_match"
                if doc_type in context["doc_lookup"]
                else "blank_or_orphan_lookup_quarantine"
            ),
            "document_content_downloaded": False,
            "evidence_semantics": "document_pointer_not_document_content_or_biomedical_label",
            "model_label_admitted": False,
        }


def _ingredients(
    archive: zipfile.ZipFile, contract: Mapping[str, Any], context: Mapping[str, Any]
) -> Iterator[dict[str, Any]]:
    member = "Products.txt"
    for row_number, row, _ in _rows(archive, member, contract):
        assert row is not None
        app, product = row["ApplNo"].strip(), row["ProductNo"].strip()
        for index, component in enumerate(candidate_ingredient_components(row["ActiveIngredient"]), start=1):
            yield {
                **_base(member, row_number, row),
                "application_number": app,
                "product_number": product,
                "ingredient_component_index_one_based": index,
                "active_ingredient_raw": row["ActiveIngredient"].strip() or None,
                "ingredient_component_exact": component,
                "ingredient_candidate_key": component.casefold(),
                "split_rule": "semicolon_source_field_projection_whitespace_trim_only",
                "product_link_state": classify_link((app, product), context["products"]),
                "application_link_state": classify_link((app,), context["applications"]),
                "molecule_link_state": "not_attempted_no_structure_or_identifier_in_source_field",
                "evidence_semantics": "ingredient_name_candidate_not_resolved_molecule_or_activity_label",
                "model_label_admitted": False,
            }


def _anomaly(
    member: str, row_number: int, anomaly_type: str, key: Sequence[str], digest: str, details: str
) -> dict[str, Any]:
    return {
        "source_id": SOURCE_ID,
        "source_archive": ARCHIVE_NAME,
        "source_member": member,
        "source_row_number_one_based": row_number,
        "anomaly_type": anomaly_type,
        "source_key_json": canonical_json(list(key)),
        "source_record_sha256": digest,
        "details": details,
        "quarantine_state": "retained_for_human_source_and_relational_review",
        "negative_label_inferred_from_absence": False,
        "model_label_admitted": False,
    }


def _anomalies(
    archive: zipfile.ZipFile, contracts: Mapping[str, Mapping[str, Any]], context: Mapping[str, Any]
) -> Iterator[dict[str, Any]]:
    specifications = (
        ("Applications.txt", ("ApplNo",), None),
        ("Products.txt", ("ApplNo", "ProductNo"), ("applications", ("ApplNo",))),
        (
            "Submissions.txt",
            ("ApplNo", "SubmissionType", "SubmissionNo"),
            ("applications", ("ApplNo",)),
        ),
        (
            "Join_Submission_ActionTypes_Lookup.txt",
            ("j_submissionActionTypeID",),
            ("submissions", ("ApplNo", "SubmissionType", "SubmissionNo")),
        ),
        (
            "SubmissionPropertyType.txt",
            (
                "ApplNo",
                "SubmissionType",
                "SubmissionNo",
                "SubmissionPropertyTypeCode",
                "SubmissionPropertyTypeID",
            ),
            ("submissions", ("ApplNo", "SubmissionType", "SubmissionNo")),
        ),
        ("ApplicationDocs.txt", ("ApplicationDocsID",), ("applications", ("ApplNo",))),
        ("MarketingStatus.txt", ("ApplNo", "ProductNo"), ("products", ("ApplNo", "ProductNo"))),
        (
            "TE.txt",
            ("ApplNo", "ProductNo", "MarketingStatusID", "TECode"),
            ("products", ("ApplNo", "ProductNo")),
        ),
    )
    for member, primary_columns, foreign in specifications:
        seen: set[tuple[str, ...]] = set()
        for row_number, row, metadata in _rows(archive, member, contracts[member]):
            digest = str(metadata["source_record_sha256"])
            if row is None:
                yield _anomaly(
                    member,
                    row_number,
                    "malformed_width",
                    [],
                    digest,
                    f"expected={metadata['expected_width']};observed={metadata['observed_width']}",
                )
                continue
            primary = tuple(normalize_submission_type(row[column]) for column in primary_columns)
            if not all(primary):
                yield _anomaly(
                    member,
                    row_number,
                    "blank_primary_key",
                    primary,
                    digest,
                    "one or more primary-key fields blank",
                )
            elif primary in seen:
                yield _anomaly(
                    member,
                    row_number,
                    "duplicate_primary_key",
                    primary,
                    digest,
                    "duplicate source primary key",
                )
            else:
                seen.add(primary)
            if foreign is not None:
                target_name, columns = foreign
                foreign_key = tuple(normalize_submission_type(row[column]) for column in columns)
                state = classify_link(foreign_key, context[target_name])
                if state != "exact_source_key_match":
                    yield _anomaly(member, row_number, state, foreign_key, digest, f"target={target_name}")

    for row_number, row, metadata in _rows(archive, "ApplicationDocs.txt", contracts["ApplicationDocs.txt"]):
        if row is None:
            continue
        submission_key = (
            row["ApplNo"].strip(),
            normalize_submission_type(row["SubmissionType"]),
            row["SubmissionNo"].strip(),
        )
        state = classify_link(submission_key, context["submissions"])
        if state != "exact_source_key_match":
            yield _anomaly(
                "ApplicationDocs.txt",
                row_number,
                f"document_submission_{state}",
                submission_key,
                str(metadata["source_record_sha256"]),
                "target=submissions",
            )

    for row_number, row, metadata in _rows(
        archive,
        "Join_Submission_ActionTypes_Lookup.txt",
        contracts["Join_Submission_ActionTypes_Lookup.txt"],
    ):
        assert row is not None
        action_type = row["ActionTypes_LookupID"].strip()
        if not action_type:
            yield _anomaly(
                "Join_Submission_ActionTypes_Lookup.txt",
                row_number,
                "blank_action_type_reference",
                [action_type],
                str(metadata["source_record_sha256"]),
                "target=action_type_lookup; absence is unknown",
            )
        elif action_type not in context["action_lookup"]:
            yield _anomaly(
                "Join_Submission_ActionTypes_Lookup.txt",
                row_number,
                "orphan_action_type_reference",
                [action_type],
                str(metadata["source_record_sha256"]),
                "target=action_type_lookup",
            )


def _methods(report: Mapping[str, Any]) -> str:
    counts = report["artifact_row_counts"]
    return f"""# Drugs@FDA record-level candidate normalization

## Result

- Normalized {counts["applications.parquet"]:,} applications, {counts["products.parquet"]:,}
  products, {counts["submissions.parquet"]:,} submissions, {counts["submission_actions.parquet"]:,}
  submission-action links, {counts["application_documents.parquet"]:,} document pointers,
  and {counts["active_ingredient_candidates.parquet"]:,} semicolon-projected ingredient-name candidates.
- Preserved {counts["regulatory_record_anomalies.parquet"]:,} explicit malformed, orphan,
  blank-key, or duplicate-key quarantine records.
- Approval, marketing status, submission status, action category, document presence, and
  ingredient names remain regulatory metadata—not efficacy, safety, QT, PK, binding,
  causality, molecular activity, negative outcomes, or model labels.

## Method

- Reverified the frozen FDA acquisition manifest and every recursive source-bundle hash.
- Parsed the exact 12-table archive under fixed member, column, encoding, width, and row-count
  contracts. One malformed ApplicationDocs row was quarantined rather than repaired.
- Joined only exact FDA relational keys. Missing keys are retained as source orphans.
- Split `ActiveIngredient` only on source semicolons and trimmed whitespace. Commas, `AND`,
  salts, mixtures, stereochemistry, and names were not chemically interpreted.
- Document URLs were retained as pointers; document content was not downloaded.
- DailyMed remains archive-inventory only. Its 17.8 GB source archives were not opened.

## Limits and gates

- Ingredient strings require curated structure/identifier resolution before molecule linkage.
- Regulatory actions and status require indication, formulation, jurisdiction, and temporal
  context and cannot support biomedical labels by themselves.
- Source rights/site-policy and institutional review remain required before redistribution.
- Application-document malformed/orphan records require source-steward adjudication.
- No missing record is interpreted as a negative outcome.
"""


def build_regulatory_records(
    raw_source_root: str | os.PathLike[str],
    output_root: str | os.PathLike[str],
    report_root: str | os.PathLike[str],
) -> dict[str, Any]:
    raw = _safe_directory(raw_source_root, context="Drugs@FDA raw source")
    output = Path(output_root).resolve()
    reports = Path(report_root).resolve()
    if output.exists() or reports.exists():
        raise RegulatoryRecordError("Output/report root exists and will not be replaced")
    binding = load_and_verify_input(raw, SOURCE_MANIFEST)
    manifest = binding.manifest
    if manifest.get("schema_version") != SOURCE_SCHEMA_VERSION or manifest.get("source_id") != SOURCE_ID:
        raise RegulatoryRecordError("Unexpected source identity")
    contracts = _table_contracts(manifest)
    archive_path = raw / ARCHIVE_NAME
    output.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(archive_path) as archive:
        if set(archive.namelist()) != set(TABLE_COLUMNS):
            raise RegulatoryRecordError("Runtime archive membership drift")
        context = _scan_context(archive, contracts)
        generators = (
            (
                "applications.parquet",
                APPLICATION_SCHEMA,
                _applications(archive, contracts["Applications.txt"], context),
            ),
            ("products.parquet", PRODUCT_SCHEMA, _products(archive, contracts["Products.txt"], context)),
            (
                "submissions.parquet",
                SUBMISSION_SCHEMA,
                _submissions(archive, contracts["Submissions.txt"], context),
            ),
            (
                "submission_actions.parquet",
                ACTION_SCHEMA,
                _actions(archive, contracts["Join_Submission_ActionTypes_Lookup.txt"], context),
            ),
            (
                "application_documents.parquet",
                DOCUMENT_SCHEMA,
                _documents(archive, contracts["ApplicationDocs.txt"], context),
            ),
            (
                "active_ingredient_candidates.parquet",
                INGREDIENT_SCHEMA,
                _ingredients(archive, contracts["Products.txt"], context),
            ),
            (
                "regulatory_record_anomalies.parquet",
                ANOMALY_SCHEMA,
                _anomalies(archive, contracts, context),
            ),
        )
        artifacts = [_write_parquet(output / name, schema, rows) for name, schema, rows in generators]
    artifacts.sort(key=lambda item: item["path"])
    normalized = _atomic_json(
        output / "regulatory_record_candidates_manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "parser_version": PARSER_VERSION,
            "dataset_id": "drugs_at_fda_record_candidates_datdaf20260804",
            "input_binding": {
                "parser_code_sha256": _runtime_code_sha256(),
                "source_manifest_internal_sha256": manifest["manifest_sha256"],
                "source_manifest_physical_sha256": sha256_file(raw / SOURCE_MANIFEST),
                "source_bundle_entries_sha256": binding.bundle_entries_sha256,
                "source_bundle_entry_count": binding.bundle_entry_count,
                "source_bundle_total_bytes": binding.bundle_total_bytes,
                "archive_sha256": sha256_file(archive_path),
            },
            "output_inventory": {
                "entries": artifacts,
                "entries_sha256": hashlib.sha256(canonical_json_bytes(artifacts)).hexdigest(),
                "entry_count": len(artifacts),
                "total_bytes": sum(int(item["bytes"]) for item in artifacts),
                "excluded_paths": ["regulatory_record_candidates_manifest.json"],
            },
            "dailymed_scope": "unchanged_archive_inventory_only_no_archive_opened",
            "canonical_molecule_links_created": 0,
            "canonical_observations_admitted": 0,
            "model_labels_admitted": 0,
            "negative_labels_inferred_from_absence": 0,
            "document_contents_downloaded": 0,
            "substantive_model_training_performed": False,
        },
    )
    row_counts = {str(item["path"]): int(item["rows"]) for item in artifacts}
    anomaly_counts: Counter[str] = Counter()
    anomaly_path = output / "regulatory_record_anomalies.parquet"
    for batch in pq.ParquetFile(anomaly_path).iter_batches(columns=["anomaly_type"], batch_size=8192):
        anomaly_counts.update(str(value) for value in batch.column(0).to_pylist())
    reports.mkdir(parents=True, exist_ok=False)
    report_body: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "decision": "record_level_regulatory_candidates_ready_for_review_not_biomedical_label_admission",
        "normalized_manifest_physical_sha256": sha256_file(
            output / "regulatory_record_candidates_manifest.json"
        ),
        "parser_code_sha256": _runtime_code_sha256(),
        "artifact_row_counts": row_counts,
        "anomaly_type_counts": dict(sorted(anomaly_counts.items())),
        "source_release_id": manifest.get("release_id"),
        "source_archive_sha256": sha256_file(archive_path),
        "rights_review_state": manifest.get("semantic_and_rights_boundaries", {}).get("license_review_state"),
        "semantic_prohibitions": [
            "approval_or_status_as_efficacy",
            "approval_or_status_as_safety_or_causality",
            "approval_or_status_as_qt_or_pk",
            "ingredient_name_as_resolved_molecule",
            "document_presence_as_positive_or_absence_as_negative",
            "regulatory_action_as_molecular_activity_or_binding",
        ],
        "canonical_molecule_links_created": 0,
        "canonical_observations_admitted": 0,
        "model_labels_admitted": 0,
        "negative_labels_inferred_from_absence": 0,
        "document_contents_downloaded": 0,
        "dailymed_archives_opened": 0,
        "substantive_model_training_performed": False,
    }
    methods_path = reports / "methods_and_limitations.md"
    _atomic_text(methods_path, _methods(report_body))
    report_body["methods_sha256"] = sha256_file(methods_path)
    report = _atomic_json(reports / "regulatory_record_analysis_report.json", report_body)
    return {"manifest": normalized, "report": report}


def verify_regulatory_records(
    raw_source_root: str | os.PathLike[str],
    output_root: str | os.PathLike[str],
    report_root: str | os.PathLike[str],
) -> dict[str, Any]:
    raw = _safe_directory(raw_source_root, context="Drugs@FDA raw source")
    output = _safe_directory(output_root, context="regulatory output")
    reports = _safe_directory(report_root, context="regulatory reports")
    binding = load_and_verify_input(raw, SOURCE_MANIFEST)
    manifest_path = output / "regulatory_record_candidates_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not verify_document_sha256(manifest):
        raise RegulatoryRecordError("Normalized manifest identity failed")
    input_binding = manifest.get("input_binding", {})
    if input_binding.get("parser_code_sha256") != _runtime_code_sha256():
        raise RegulatoryRecordError("Parser code/normalized manifest binding failed")
    if input_binding.get("source_bundle_entries_sha256") != binding.bundle_entries_sha256:
        raise RegulatoryRecordError("Normalized/source bundle binding failed")
    inventory = manifest.get("output_inventory")
    if not isinstance(inventory, dict) or inventory.get("excluded_paths") != [manifest_path.name]:
        raise RegulatoryRecordError("Normalized inventory contract failed")
    entries = inventory.get("entries")
    if not isinstance(entries, list) or hashlib.sha256(
        canonical_json_bytes(entries)
    ).hexdigest() != inventory.get("entries_sha256"):
        raise RegulatoryRecordError("Normalized inventory digest failed")
    expected: set[str] = set()
    for item in entries:
        relative = _safe_relative(item.get("path"), context="normalized artifact")
        expected.add(relative.as_posix())
        path = output / Path(*relative.parts)
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != int(item.get("bytes", -1))
            or sha256_file(path) != item.get("sha256")
        ):
            raise RegulatoryRecordError(f"Artifact identity drift: {relative}")
        parquet = pq.ParquetFile(path)
        if parquet.metadata.num_rows != int(item.get("rows", -1)) or arrow_schema_sha256(
            parquet.schema_arrow
        ) != item.get("arrow_schema_sha256"):
            raise RegulatoryRecordError(f"Parquet identity drift: {relative}")
        prohibited = [
            name
            for name in ("model_label_admitted", "negative_label_inferred_from_absence")
            if name in parquet.schema_arrow.names
        ]
        for batch in parquet.iter_batches(columns=prohibited, batch_size=8192):
            for index in range(len(prohibited)):
                if any(batch.column(index).to_pylist()):
                    raise RegulatoryRecordError(f"Prohibited inference/admission flag: {relative}")
    observed = _closed_regular_files(output, context="regulatory output")
    if observed != expected | {manifest_path.name}:
        raise RegulatoryRecordError("Normalized output membership drift")
    report_path = reports / "regulatory_record_analysis_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(report, dict) or not verify_document_sha256(report):
        raise RegulatoryRecordError("Report identity failed")
    if _closed_regular_files(reports, context="regulatory reports") != {
        report_path.name,
        "methods_and_limitations.md",
    }:
        raise RegulatoryRecordError("Report output membership drift")
    if report.get("normalized_manifest_physical_sha256") != sha256_file(manifest_path):
        raise RegulatoryRecordError("Report/normalized manifest binding failed")
    if report.get("parser_code_sha256") != _runtime_code_sha256():
        raise RegulatoryRecordError("Parser code/report binding failed")
    if sha256_file(reports / "methods_and_limitations.md") != report.get("methods_sha256"):
        raise RegulatoryRecordError("Methods/report binding failed")
    for document in (manifest, report):
        if any(
            (
                document.get("canonical_observations_admitted") != 0,
                document.get("model_labels_admitted") != 0,
                document.get("negative_labels_inferred_from_absence") != 0,
                document.get("substantive_model_training_performed") is not False,
            )
        ):
            raise RegulatoryRecordError("Zero-label/training contract failed")
    return {
        "status": "passed",
        "artifact_count": len(entries),
        "artifact_rows": sum(int(item["rows"]) for item in entries),
        "normalized_manifest_sha256": sha256_file(manifest_path),
        "report_sha256": sha256_file(report_path),
        "canonical_observations_admitted": 0,
        "model_labels_admitted": 0,
        "negative_labels_inferred_from_absence": 0,
        "substantive_model_training_performed": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", default="research/data/platform/raw/external_public/drugs_at_fda_bulk")
    parser.add_argument(
        "--output-root",
        default="research/data/platform/interim/regulatory_record_candidates/drugs_at_fda_20260804",
    )
    parser.add_argument(
        "--report-root", default="research/reports/platform/regulatory_record_analysis/drugs_at_fda_20260804"
    )
    parser.add_argument("--verify-existing", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.verify_existing:
        result = verify_regulatory_records(args.raw_root, args.output_root, args.report_root)
    else:
        result = build_regulatory_records(args.raw_root, args.output_root, args.report_root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
