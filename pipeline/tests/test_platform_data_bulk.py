from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from menin_discovery.platform_data_bulk import (
    _SCHEMA_NORMALIZATION_MARKER,
    ACTIVITY_ARROW_SCHEMA,
    DEVELOPMENT_ARROW_SCHEMA,
    TARGET_COMPONENT_ARROW_SCHEMA,
    ArchivePart,
    _activity_query,
    _read_json_byte_snapshot,
    _resume_schema_normalization_transaction,
    _schema_normalization_source_identity,
    _schema_transaction_entry,
    _verify_schema_normalization_prestate_identity,
    _verify_schema_normalization_receipt,
    _write_frame_with_schema,
    assemble_archive_parts,
    exact_manifest_activity_overlap,
    export_chembl37_activity_facts,
    inspect_sqlite_schema,
    normalize_chembl37_export_schemas,
)
from menin_discovery.platform_data_schema import (
    SCHEMA_VERSION,
    arrow_schema_contract,
    canonical_json,
)


def test_archive_assembly_accepts_explicit_paths_and_enforces_ranges(tmp_path: Path) -> None:
    first = tmp_path / "first.part"
    second = tmp_path / "second.part"
    first.write_bytes(b"abc")
    second.write_bytes(b"def")
    expected = hashlib.sha256(b"abcdef").hexdigest()
    result = assemble_archive_parts(
        [first, second],
        tmp_path / "archive.bin",
        expected_bytes=6,
        expected_sha256=expected,
    )
    assert result["valid"] is True
    assert result["parts"][0]["file"] == "first.part"
    assert "archive_path" not in result

    bad = ArchivePart(first, 1, 3, hashlib.sha256(b"abc").hexdigest())
    with pytest.raises(ValueError, match="gap or is out of order"):
        assemble_archive_parts([bad], tmp_path / "bad.bin", expected_bytes=3, expected_sha256="0" * 64)


def _tiny_chembl_database(path: Path) -> None:
    connection = sqlite3.connect(path)
    connection.executescript(
        """
CREATE TABLE activities (
 activity_id INTEGER PRIMARY KEY, assay_id INTEGER, molregno INTEGER, doc_id INTEGER, src_id INTEGER,
 standard_type TEXT, standard_value REAL, standard_units TEXT, standard_relation TEXT
);
CREATE TABLE assays (assay_id INTEGER PRIMARY KEY, tid INTEGER, doc_id INTEGER, assay_type TEXT);
CREATE TABLE target_dictionary (tid INTEGER PRIMARY KEY, chembl_id TEXT, target_type TEXT);
CREATE TABLE molecule_dictionary (molregno INTEGER PRIMARY KEY, chembl_id TEXT);
CREATE TABLE compound_structures (molregno INTEGER PRIMARY KEY, canonical_smiles TEXT, standard_inchi_key TEXT);
CREATE TABLE docs (doc_id INTEGER PRIMARY KEY, chembl_id TEXT);
CREATE TABLE source (src_id INTEGER PRIMARY KEY, src_short_name TEXT, src_description TEXT);
INSERT INTO source VALUES (39, 'DRUG_PK', 'Curated Drug Pharmacokinetic Data');
INSERT INTO target_dictionary VALUES (1, 'CHEMBL1', 'SINGLE PROTEIN');
INSERT INTO assays VALUES (1, 1, 1, 'B');
INSERT INTO molecule_dictionary VALUES (1, 'CHEMBL2');
INSERT INTO compound_structures VALUES (1, 'CC', 'OTMSDBZUPAUEDD-UHFFFAOYSA-N');
INSERT INTO docs VALUES (1, 'CHEMBL3');
INSERT INTO activities VALUES (1, 1, 1, 1, 39, 'Kd', 10.0, 'nM', '=');
"""
    )
    connection.commit()
    connection.close()


def test_actual_chembl_source_table_is_required_and_joined(tmp_path: Path) -> None:
    database = tmp_path / "chembl.db"
    _tiny_chembl_database(database)
    schema = inspect_sqlite_schema(database)
    query = _activity_query(schema)
    assert 'LEFT JOIN "source" AS sd' in query
    assert "pchembl_value" not in query
    connection = sqlite3.connect(database)
    frame = pd.read_sql_query(query, connection, params=(0, 10))
    connection.close()
    assert frame.loc[0, "activity_source_name"] == "DRUG_PK"
    assert frame.loc[0, "activity_source_description"] == "Curated Drug Pharmacokinetic Data"


def test_completed_export_rejects_unmanifested_parquet(tmp_path: Path) -> None:
    database = tmp_path / "chembl.db"
    _tiny_chembl_database(database)
    interim = tmp_path / "interim"
    export_chembl37_activity_facts(database, interim, chunk_size=1)
    extra = interim / "chembl_37_bulk" / "activity_facts" / "extra.parquet"
    pd.DataFrame({"x": [1]}).to_parquet(extra, index=False)
    with pytest.raises(RuntimeError, match="Unmanifested or missing"):
        export_chembl37_activity_facts(database, interim, chunk_size=1)


def test_activity_export_uses_one_declared_physical_arrow_schema(tmp_path: Path) -> None:
    database = tmp_path / "chembl.db"
    _tiny_chembl_database(database)
    interim = tmp_path / "interim"
    manifest = export_chembl37_activity_facts(database, interim, chunk_size=1)
    contract = arrow_schema_contract(ACTIVITY_ARROW_SCHEMA)
    assert manifest["arrow_schema"] == contract
    assert manifest["parts"][0]["arrow_schema_sha256"] == contract["sha256"]
    part = interim / "chembl_37_bulk" / manifest["parts"][0]["path"]
    assert (
        pq.ParquetFile(part)
        .schema_arrow.remove_metadata()
        .equals(
            ACTIVITY_ARROW_SCHEMA,
            check_metadata=True,
        )
    )


def test_declared_schemas_are_stable_across_all_null_and_populated_chunks(
    tmp_path: Path,
) -> None:
    for name, schema, populated in (
        ("activity", ACTIVITY_ARROW_SCHEMA, {"activity_id": 1, "document_year": 2024}),
        (
            "development",
            DEVELOPMENT_ARROW_SCHEMA,
            {"molecule_row_id": 1, "first_approval": 2024},
        ),
        (
            "target",
            TARGET_COMPONENT_ARROW_SCHEMA,
            {"target_chembl_id": "CHEMBL1", "component_id": 1},
        ),
    ):
        null_frame = pd.DataFrame({field.name: [None] for field in schema})
        populated_frame = null_frame.copy()
        for column, value in populated.items():
            populated_frame.loc[0, column] = value
        null_path = tmp_path / f"{name}-null.parquet"
        populated_path = tmp_path / f"{name}-populated.parquet"
        _write_frame_with_schema(null_frame, null_path, schema)
        _write_frame_with_schema(populated_frame, populated_path, schema)
        assert (
            pq.ParquetFile(null_path)
            .schema_arrow.remove_metadata()
            .equals(
                schema,
                check_metadata=True,
            )
        )
        assert (
            pq.ParquetFile(populated_path)
            .schema_arrow.remove_metadata()
            .equals(
                schema,
                check_metadata=True,
            )
        )


def _write_interruption_journal(bulk_root: Path) -> list[dict[str, object]]:
    staging = bulk_root / ".schema_normalization_staging"
    staging.mkdir(parents=True)
    destinations = [
        ("parquet", "parts/one.parquet"),
        ("parquet", "parts/two.parquet"),
        ("receipt", "schema_normalization_receipt.json"),
        ("manifest", "activity_export_manifest.json"),
        ("summary", "specialized_views/specialized_views_manifest.json"),
    ]
    entries: list[dict[str, object]] = []
    for index, (kind, relative) in enumerate(destinations):
        destination = bulk_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(f"old-{index}".encode())
        staged = staging / relative
        staged.parent.mkdir(parents=True, exist_ok=True)
        staged.write_bytes(f"new-{index}".encode())
        entries.append(
            _schema_transaction_entry(
                bulk_root,
                staged,
                destination,
                kind=kind,
                expected_old_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
                expected_old_size_bytes=destination.stat().st_size,
            )
        )
    marker = {
        "schema_version": SCHEMA_VERSION,
        "status": "committing",
        "transaction_id": "00000000-0000-4000-8000-000000000001",
        "owner": {"hostname": "test-host", "pid": 1},
        "staging_directory": staging.name,
        "staged_part_count": sum(entry["kind"] == "parquet" for entry in entries),
        "entry_count": len(entries),
        "entries_sha256": hashlib.sha256(canonical_json(entries).encode("utf-8")).hexdigest(),
        "entries": entries,
    }
    (bulk_root / _SCHEMA_NORMALIZATION_MARKER).write_text(
        json.dumps(marker),
        encoding="utf-8",
    )
    return entries


def _assert_interrupted_transaction_resumes(tmp_path: Path, fail_after: int) -> None:
    bulk_root = tmp_path / "chembl_37_bulk"
    bulk_root.mkdir()
    entries = _write_interruption_journal(bulk_root)
    with pytest.raises(RuntimeError, match="Injected"):
        _resume_schema_normalization_transaction(
            bulk_root,
            fail_after_replacements=fail_after,
            validate_corpus=False,
        )
    assert (bulk_root / _SCHEMA_NORMALIZATION_MARKER).is_file()
    _resume_schema_normalization_transaction(bulk_root, validate_corpus=False)
    for index, entry in enumerate(entries):
        assert (bulk_root / str(entry["destination_path"])).read_bytes() == (f"new-{index}".encode())
    assert not (bulk_root / _SCHEMA_NORMALIZATION_MARKER).exists()
    assert not (bulk_root / ".schema_normalization_staging").exists()


def test_schema_transaction_resumes_after_mid_parquet_interruption(tmp_path: Path) -> None:
    _assert_interrupted_transaction_resumes(tmp_path, fail_after=1)


def test_schema_transaction_resumes_after_mid_manifest_interruption(tmp_path: Path) -> None:
    _assert_interrupted_transaction_resumes(tmp_path, fail_after=4)


@pytest.mark.parametrize("field", ["entry_count", "staged_part_count"])
def test_schema_transaction_rejects_tampered_counts(tmp_path: Path, field: str) -> None:
    bulk_root = tmp_path / "chembl_37_bulk"
    bulk_root.mkdir()
    _write_interruption_journal(bulk_root)
    marker_path = bulk_root / _SCHEMA_NORMALIZATION_MARKER
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker[field] = int(marker[field]) + 1
    marker_path.write_text(json.dumps(marker), encoding="utf-8")
    with pytest.raises(RuntimeError, match="count contract mismatch"):
        _resume_schema_normalization_transaction(bulk_root, validate_corpus=False)


def test_schema_transaction_rejects_cross_mapped_staged_path(tmp_path: Path) -> None:
    bulk_root = tmp_path / "chembl_37_bulk"
    bulk_root.mkdir()
    _write_interruption_journal(bulk_root)
    marker_path = bulk_root / _SCHEMA_NORMALIZATION_MARKER
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["entries"][0]["staged_path"], marker["entries"][1]["staged_path"] = (
        marker["entries"][1]["staged_path"],
        marker["entries"][0]["staged_path"],
    )
    marker["entries_sha256"] = hashlib.sha256(canonical_json(marker["entries"]).encode("utf-8")).hexdigest()
    marker_path.write_text(json.dumps(marker), encoding="utf-8")
    with pytest.raises(RuntimeError, match="path mapping mismatch"):
        _resume_schema_normalization_transaction(bulk_root, validate_corpus=False)


def test_schema_transaction_rejects_destination_neither_old_nor_new(tmp_path: Path) -> None:
    bulk_root = tmp_path / "chembl_37_bulk"
    bulk_root.mkdir()
    entries = _write_interruption_journal(bulk_root)
    (bulk_root / str(entries[0]["destination_path"])).write_bytes(b"untracked-state")
    with pytest.raises(RuntimeError, match="neither journaled old nor new"):
        _resume_schema_normalization_transaction(bulk_root, validate_corpus=False)


def test_schema_transaction_rejects_destination_drift_before_journal(
    tmp_path: Path,
) -> None:
    bulk_root = tmp_path / "chembl_37_bulk"
    destination = bulk_root / "parts" / "one.parquet"
    staged = bulk_root / ".schema_normalization_staging" / "parts" / "one.parquet"
    destination.parent.mkdir(parents=True)
    staged.parent.mkdir(parents=True)
    destination.write_bytes(b"old")
    staged.write_bytes(b"new")
    expected_sha256 = hashlib.sha256(destination.read_bytes()).hexdigest()
    expected_size = destination.stat().st_size
    destination.write_bytes(b"concurrent-change")
    with pytest.raises(RuntimeError, match="drifted after staging"):
        _schema_transaction_entry(
            bulk_root,
            staged,
            destination,
            kind="parquet",
            expected_old_sha256=expected_sha256,
            expected_old_size_bytes=expected_size,
        )


def test_staging_marker_is_preserved_fail_closed(tmp_path: Path) -> None:
    database = tmp_path / "chembl.db"
    _tiny_chembl_database(database)
    interim = tmp_path / "interim"
    bulk_root = interim / "chembl_37_bulk"
    staging = bulk_root / ".schema_normalization_staging"
    staging.mkdir(parents=True)
    staged_payload = staging / "active.payload"
    staged_payload.write_bytes(b"active")
    marker = bulk_root / _SCHEMA_NORMALIZATION_MARKER
    marker.write_text(
        json.dumps(
            {
                "status": "staging",
                "transaction_id": "00000000-0000-4000-8000-000000000002",
                "owner": {"hostname": "live-host", "pid": 123},
                "started_at_utc": "2026-01-01T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="may still be active"):
        normalize_chembl37_export_schemas(database, interim)
    assert marker.is_file()
    assert staged_payload.read_bytes() == b"active"


def test_schema_normalization_rejects_database_manifest_identity_mismatch(
    tmp_path: Path,
) -> None:
    database = tmp_path / "chembl.db"
    _tiny_chembl_database(database)
    bulk_root = tmp_path / "chembl_37_bulk"
    specialized = bulk_root / "specialized_views"
    specialized.mkdir(parents=True)
    identity = {
        "schema_version": SCHEMA_VERSION,
        "source_version": "ChEMBL_37",
        "database_sha256": "d" * 64,
    }
    (bulk_root / "activity_export_manifest.json").write_text(
        json.dumps(identity),
        encoding="utf-8",
    )
    child = {**identity, "view_name": "one"}
    (specialized / "one_manifest.json").write_text(
        json.dumps(child),
        encoding="utf-8",
    )
    (specialized / "specialized_views_manifest.json").write_text(
        json.dumps({**identity, "views": {"one": child}}),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="do not share one ChEMBL_37"):
        _schema_normalization_source_identity(database, bulk_root)


def _write_receipt_bound_export_fixture(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    bulk_root = tmp_path / "chembl_37_bulk"
    specialized = bulk_root / "specialized_views"
    specialized.mkdir(parents=True)
    database_sha256 = "d" * 64
    activity_contract = arrow_schema_contract(ACTIVITY_ARROW_SCHEMA)
    development_contract = arrow_schema_contract(DEVELOPMENT_ARROW_SCHEMA)
    target_contract = arrow_schema_contract(TARGET_COMPONENT_ARROW_SCHEMA)
    file_receipts: list[dict[str, object]] = []

    def write_part(
        relative: str,
        schema: pa.Schema,
        values: dict[str, object],
        contract: dict[str, object],
    ) -> tuple[Path, dict[str, object]]:
        path = bulk_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame({field.name: [None] for field in schema})
        for column, value in values.items():
            frame.loc[0, column] = value
        _write_frame_with_schema(frame, path, schema)
        sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        null_cells = sum(column.null_count for column in pq.read_table(path).columns)
        file_receipts.append(
            {
                "path": relative,
                "rows": 1,
                "old_size_bytes": path.stat().st_size,
                "new_size_bytes": path.stat().st_size,
                "old_sha256": "a" * 64,
                "old_arrow_schema_sha256": "b" * 64,
                "new_sha256": sha256,
                "declared_arrow_schema_sha256": contract["sha256"],
                "value_preservation": {
                    "coerced_old_equals_new": True,
                    "old_coerced_null_cells": null_cells,
                    "new_null_cells": null_cells,
                },
            }
        )
        return path, {
            "rows": 1,
            "sha256": sha256,
            "size_bytes": path.stat().st_size,
            "arrow_schema_sha256": contract["sha256"],
        }

    full_path, full_record = write_part(
        "activity_facts/part-00000.parquet",
        ACTIVITY_ARROW_SCHEMA,
        {"activity_id": 1},
        activity_contract,
    )
    full_record["path"] = full_path.relative_to(bulk_root).as_posix()
    full_manifest: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "source_version": "ChEMBL_37",
        "database_sha256": database_sha256,
        "rows_written": 1,
        "part_count": 1,
        "parts": [full_record],
        "arrow_schema": activity_contract,
    }
    activity_views = (
        "single_protein_kd_ki",
        "single_protein_ic50_ec50_candidates",
        "herg_all_endpoints",
        "pk_adme_candidates",
        "cardiac_qt_apd_inventory",
    )
    views: dict[str, dict[str, object]] = {}
    for activity_id, view_name in enumerate(activity_views, start=2):
        relative = f"specialized_views/{view_name}/part-00000.parquet"
        part_path, record = write_part(
            relative,
            ACTIVITY_ARROW_SCHEMA,
            {"activity_id": activity_id},
            activity_contract,
        )
        record["path"] = part_path.relative_to(specialized).as_posix()
        views[view_name] = {
            "schema_version": SCHEMA_VERSION,
            "source_version": "ChEMBL_37",
            "database_sha256": database_sha256,
            "view_name": view_name,
            "row_count": 1,
            "part_count": 1,
            "parts": [record],
            "arrow_schema": activity_contract,
        }
    development_path, development_record = write_part(
        "specialized_views/molecule_development_annotations/part-00000.parquet",
        DEVELOPMENT_ARROW_SCHEMA,
        {"molecule_row_id": 1},
        development_contract,
    )
    development_record["path"] = development_path.relative_to(specialized).as_posix()
    views["molecule_development_annotations"] = {
        "schema_version": SCHEMA_VERSION,
        "source_version": "ChEMBL_37",
        "database_sha256": database_sha256,
        "view_name": "molecule_development_annotations",
        "row_count": 1,
        "part_count": 1,
        "parts": [development_record],
        "arrow_schema": development_contract,
    }
    target_path, target_record = write_part(
        "specialized_views/target_components.parquet",
        TARGET_COMPONENT_ARROW_SCHEMA,
        {"target_chembl_id": "CHEMBL240"},
        target_contract,
    )
    views["target_components"] = {
        "schema_version": SCHEMA_VERSION,
        "source_version": "ChEMBL_37",
        "database_sha256": database_sha256,
        "view_name": "target_components",
        "path": target_path.name,
        "row_count": 1,
        "sha256": target_record["sha256"],
        "size_bytes": target_record["size_bytes"],
        "arrow_schema": target_contract,
        "arrow_schema_sha256": target_contract["sha256"],
    }
    receipt: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "source_identity": {
            "source_version": "ChEMBL_37",
            "database_sha256": database_sha256,
            "pre_normalization_activity_manifest_sha256": "c" * 64,
            "pre_normalization_specialized_summary_sha256": "e" * 64,
        },
        "aggregate": {
            "parquet_files": len(file_receipts),
            "rows_across_overlapping_exports": len(file_receipts),
            "old_size_bytes": sum(int(record["old_size_bytes"]) for record in file_receipts),
            "new_size_bytes": sum(int(record["new_size_bytes"]) for record in file_receipts),
            "all_value_preservation_checks_passed": True,
        },
        "arrow_schema_contracts": {
            "activity": activity_contract,
            "molecule_development_annotations": development_contract,
            "target_components": target_contract,
        },
        "files": file_receipts,
    }
    receipt_path = bulk_root / "schema_normalization_receipt.json"
    receipt_path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
    binding: dict[str, object] = {
        "path": receipt_path.name,
        "sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
        "size_bytes": receipt_path.stat().st_size,
        "parquet_files": len(file_receipts),
    }
    full_manifest["schema_normalization_receipt"] = binding
    (bulk_root / "activity_export_manifest.json").write_text(
        json.dumps(full_manifest, sort_keys=True), encoding="utf-8"
    )
    for view_name, child in views.items():
        child["schema_normalization_receipt"] = binding
        (specialized / f"{view_name}_manifest.json").write_text(
            json.dumps(child, sort_keys=True), encoding="utf-8"
        )
    summary: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "source_version": "ChEMBL_37",
        "database_sha256": database_sha256,
        "views": views,
        "schema_normalization_receipt": binding,
    }
    (specialized / "specialized_views_manifest.json").write_text(
        json.dumps(summary, sort_keys=True), encoding="utf-8"
    )
    return bulk_root, summary


def test_schema_receipt_rejects_extra_physical_parquet(tmp_path: Path) -> None:
    bulk_root, summary = _write_receipt_bound_export_fixture(tmp_path)
    _verify_schema_normalization_receipt(summary, bulk_root)
    source = bulk_root / "activity_facts" / "part-00000.parquet"
    rogue = bulk_root / "unexpected" / "nested" / "rogue.parquet"
    rogue.parent.mkdir(parents=True)
    shutil.copyfile(source, rogue)
    with pytest.raises(RuntimeError, match="physical Parquet inventory mismatch"):
        _verify_schema_normalization_receipt(summary, bulk_root)


def test_source_identity_rejects_manifest_prestate_drift(tmp_path: Path) -> None:
    activity = tmp_path / "activity_export_manifest.json"
    summary = tmp_path / "specialized_views_manifest.json"
    activity.write_bytes(b"activity-v1")
    summary.write_bytes(b"summary-v1")
    source_identity = {
        "pre_normalization_activity_manifest_sha256": hashlib.sha256(activity.read_bytes()).hexdigest(),
        "pre_normalization_specialized_summary_sha256": hashlib.sha256(summary.read_bytes()).hexdigest(),
    }
    staged_states = {
        activity.resolve(): (hashlib.sha256(b"activity-v2").hexdigest(), 11),
        summary.resolve(): (
            source_identity["pre_normalization_specialized_summary_sha256"],
            summary.stat().st_size,
        ),
    }
    with pytest.raises(RuntimeError, match="identity/pre-state drift"):
        _verify_schema_normalization_prestate_identity(
            source_identity,
            staged_states,
            activity,
            summary,
        )


def test_json_byte_snapshot_rejects_change_after_single_buffer_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "manifest.json"
    path.write_text('{"version": 1}', encoding="utf-8")
    original_read_bytes = Path.read_bytes

    def read_then_change(target: Path) -> bytes:
        payload = original_read_bytes(target)
        if target == path:
            target.write_text('{"version": 2}', encoding="utf-8")
        return payload

    monkeypatch.setattr(Path, "read_bytes", read_then_change)
    with pytest.raises(RuntimeError, match="changed while read"):
        _read_json_byte_snapshot(path, context="test manifest")


def test_schema_receipt_rejects_noncanonical_target_path(tmp_path: Path) -> None:
    bulk_root, summary = _write_receipt_bound_export_fixture(tmp_path)
    specialized = bulk_root / "specialized_views"
    target_path = specialized / "target_components_manifest.json"
    target = json.loads(target_path.read_text(encoding="utf-8"))
    target["path"] = "../activity_facts/part-00000.parquet"
    target_path.write_text(json.dumps(target, sort_keys=True), encoding="utf-8")
    summary["views"]["target_components"] = target  # type: ignore[index]
    (specialized / "specialized_views_manifest.json").write_text(
        json.dumps(summary, sort_keys=True), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="noncanonical or duplicated"):
        _verify_schema_normalization_receipt(summary, bulk_root)


def _view_manifest(root: Path, name: str, parts: list[list[int]]) -> dict[str, object]:
    directory = root / name
    directory.mkdir(parents=True)
    records = []
    for index, values in enumerate(parts):
        path = directory / f"part-{index:02d}.parquet"
        pd.DataFrame({"activity_id": values}).to_parquet(path, index=False)
        records.append(
            {
                "path": f"{name}/{path.name}",
                "rows": len(values),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    return {
        "view_name": name,
        "database_sha256": "a" * 64,
        "row_count": sum(len(values) for values in parts),
        "parts": records,
    }


def test_manifest_overlap_is_exact_and_rejects_cross_part_duplicates(tmp_path: Path) -> None:
    left = _view_manifest(tmp_path, "left", [[1, 2], [3, 4]])
    right = _view_manifest(tmp_path, "right", [[2, 4], [6, 8]])
    assert exact_manifest_activity_overlap(left, right, tmp_path) == 2

    bad = _view_manifest(tmp_path, "bad", [[1, 2], [2, 5]])
    with pytest.raises(RuntimeError, match="globally"):
        exact_manifest_activity_overlap(bad, right, tmp_path)
