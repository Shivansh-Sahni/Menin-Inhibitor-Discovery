from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import pytest
from menin_discovery.platform_external_normalization import canonical_json_bytes, document_with_sha256
from menin_discovery.platform_regulatory_records import (
    ARCHIVE_NAME,
    SOURCE_MANIFEST,
    SOURCE_SCHEMA_VERSION,
    TABLE_COLUMNS,
    RegulatoryRecordError,
    _primary_state,
    _rows,
    _safe_relative,
    absence_semantics,
    build_regulatory_records,
    candidate_ingredient_components,
    classify_link,
    verify_regulatory_records,
)


def _source_rows() -> dict[str, list[str]]:
    return {
        "ActionTypes_Lookup.txt": ["1", "Administrative action", "ADMIN", ""],
        "ApplicationDocs.txt": [
            "10",
            "1",
            "000001",
            "ORIG",
            "1",
            "Approval letter",
            "https://example.invalid/document",
            "2026-01-01 00:00:00",
        ],
        "Applications.txt": ["000001", "NDA", "", "FIXTURE SPONSOR"],
        "ApplicationsDocsType_Lookup.txt": ["1", "Letter"],
        "Join_Submission_ActionTypes_Lookup.txt": ["ORIG", "20", "000001", "1", "1"],
        "MarketingStatus.txt": ["1", "000001", "001"],
        "MarketingStatus_Lookup.txt": ["1", "Prescription"],
        "Products.txt": [
            "000001",
            "001",
            "TABLET;ORAL",
            "1 MG",
            "0",
            "FIXTURE DRUG",
            "INGREDIENT A; INGREDIENT B",
            "0",
        ],
        "SubmissionClass_Lookup.txt": ["1", "ORIG", "Original"],
        "SubmissionPropertyType.txt": ["000001", "ORIG", "1", "Null", "0"],
        "Submissions.txt": [
            "000001",
            "1",
            "ORIG",
            "1",
            "AP",
            "2026-01-01 00:00:00",
            "",
            "STANDARD",
        ],
        "TE.txt": ["000001", "001", "1", "AB"],
    }


def _fixture_source(tmp_path: Path) -> Path:
    root = tmp_path / "drugs_at_fda_bulk"
    root.mkdir(parents=True)
    rows = _source_rows()
    archive_path = root / ARCHIVE_NAME
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for member in sorted(TABLE_COLUMNS):
            text = "\t".join(TABLE_COLUMNS[member]) + "\n" + "\t".join(rows[member]) + "\n"
            archive.writestr(
                member,
                text.encode(
                    "cp1252" if member in {"ApplicationDocs.txt", "Submissions.txt"} else "utf-8-sig"
                ),
            )
    member_contracts = [
        {
            "archive_member_path": member,
            "columns": list(TABLE_COLUMNS[member]),
            "data_row_count": 1,
            "malformed_width_rows": 0,
        }
        for member in sorted(TABLE_COLUMNS)
    ]
    entry = {
        "path": ARCHIVE_NAME,
        "bytes": archive_path.stat().st_size,
        "sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
    }
    manifest = document_with_sha256(
        {
            "schema_version": SOURCE_SCHEMA_VERSION,
            "source_id": root.name,
            "release_id": "fixture",
            "snapshot_status": "fixture",
            "exact_physical_file_count": 1,
            "exact_physical_bytes": archive_path.stat().st_size,
            "files": [
                {
                    "local_path": ARCHIVE_NAME,
                    "acquired_bytes": archive_path.stat().st_size,
                    "acquired_sha256": entry["sha256"],
                }
            ],
            "archive_member_table": {"members": member_contracts},
            "semantic_and_rights_boundaries": {"license_review_state": "human_review_required"},
            "bundle_inventory": {
                "entries": [entry],
                "entries_sha256": hashlib.sha256(canonical_json_bytes([entry])).hexdigest(),
                "entry_count": 1,
                "total_bytes": archive_path.stat().st_size,
                "excluded_paths": [SOURCE_MANIFEST],
            },
        }
    )
    (root / SOURCE_MANIFEST).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return root


def test_ingredient_projection_is_conservative() -> None:
    assert candidate_ingredient_components(" A ; B AND C ; SALT, HYDRATE ") == [
        "A",
        "B AND C",
        "SALT, HYDRATE",
    ]


def test_exact_link_and_orphan_are_distinct() -> None:
    target = {("000001", "001")}
    assert classify_link(("000001", "001"), target) == "exact_source_key_match"
    assert classify_link(("000001", "999"), target) == "orphan_source_key_quarantine"
    assert classify_link(("", "001"), target) == "blank_key_quarantine"


def test_absence_is_unknown_not_negative() -> None:
    assert absence_semantics(0) == "no_linked_record_in_snapshot_unknown_not_negative"
    assert absence_semantics(1) == "linked_records_present"
    with pytest.raises(RegulatoryRecordError, match="Negative"):
        absence_semantics(-1)


def test_duplicate_and_blank_primary_keys_are_quarantined() -> None:
    seen: set[tuple[str, ...]] = set()
    assert _primary_state(("1",), seen) == "unique_primary_key"
    assert _primary_state(("1",), seen) == "duplicate_primary_key_quarantine"
    assert _primary_state(("",), seen) == "blank_primary_key_quarantine"


@pytest.mark.parametrize("value", ["../escape", "/absolute", "a/../../b", ""])
def test_output_path_traversal_fails_closed(value: str) -> None:
    with pytest.raises(RegulatoryRecordError, match="Unsafe"):
        _safe_relative(value, context="fixture")


def test_malformed_width_row_is_explicit(tmp_path: Path) -> None:
    archive_path = tmp_path / "fixture.zip"
    member = "Applications.txt"
    with zipfile.ZipFile(archive_path, "w") as writer:
        writer.writestr(member, "\t".join(TABLE_COLUMNS[member]) + "\nonly\ttwo\n")
    contract = {"data_row_count": 1, "malformed_width_rows": 1}
    with zipfile.ZipFile(archive_path) as archive:
        values = list(_rows(archive, member, contract))
    assert values[0][1] is None
    assert values[0][2]["expected_width"] == 4
    assert values[0][2]["observed_width"] == 2


def test_small_end_to_end_build_verify_and_output_tamper(tmp_path: Path) -> None:
    raw = _fixture_source(tmp_path)
    output = tmp_path / "output"
    reports = tmp_path / "reports"
    result = build_regulatory_records(raw, output, reports)
    assert result["manifest"]["canonical_observations_admitted"] == 0
    assert result["manifest"]["model_labels_admitted"] == 0
    assert verify_regulatory_records(raw, output, reports)["status"] == "passed"
    product_path = output / "products.parquet"
    with product_path.open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(RegulatoryRecordError, match="identity drift"):
        verify_regulatory_records(raw, output, reports)


def test_source_tamper_fails_before_normalization(tmp_path: Path) -> None:
    raw = _fixture_source(tmp_path)
    with (raw / ARCHIVE_NAME).open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(Exception, match="byte count changed|SHA-256 changed"):
        build_regulatory_records(raw, tmp_path / "output", tmp_path / "reports")


def test_symlinked_output_fails_closed(tmp_path: Path) -> None:
    raw = _fixture_source(tmp_path)
    output = tmp_path / "output"
    reports = tmp_path / "reports"
    build_regulatory_records(raw, output, reports)
    product_path = output / "products.parquet"
    external_copy = tmp_path / "products-copy.parquet"
    shutil.copy2(product_path, external_copy)
    product_path.unlink()
    product_path.symlink_to(external_copy)
    with pytest.raises(RegulatoryRecordError, match="Non-regular|identity drift"):
        verify_regulatory_records(raw, output, reports)


def test_parser_code_binding_tamper_fails(tmp_path: Path) -> None:
    raw = _fixture_source(tmp_path)
    output = tmp_path / "output"
    reports = tmp_path / "reports"
    build_regulatory_records(raw, output, reports)
    manifest_path = output / "regulatory_record_candidates_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["input_binding"]["parser_code_sha256"] = "0" * 64
    manifest.pop("manifest_sha256")
    manifest_path.write_text(
        json.dumps(document_with_sha256(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RegulatoryRecordError, match="Parser code"):
        verify_regulatory_records(raw, output, reports)
