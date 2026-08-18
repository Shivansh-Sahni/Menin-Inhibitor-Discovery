#!/usr/bin/env python3
"""Independently verify the frozen ChEMBL specialized-export boundary.

This verifier deliberately does not import the exporter or canonical builder.  It
checks the serialized contract and physical Parquet bytes directly so it can be
used as an integration-lead acceptance gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

EXPECTED_VIEWS = {
    "cardiac_qt_apd_inventory",
    "herg_all_endpoints",
    "molecule_development_annotations",
    "pk_adme_candidates",
    "single_protein_ic50_ec50_candidates",
    "single_protein_kd_ki",
    "target_components",
}
ACTIVITY_VIEWS = EXPECTED_VIEWS - {
    "molecule_development_annotations",
    "target_components",
}
EXPECTED_OVERLAP_KEYS = {
    "cardiac_qt_apd_and_herg": "cardiac_qt_apd_inventory",
    "ic50_ec50_and_herg": "single_protein_ic50_ec50_candidates",
    "kd_ki_and_herg": "single_protein_kd_ki",
    "pk_adme_and_herg": "pk_adme_candidates",
}
IDENTITY_FIELDS = ("schema_version", "source_version", "database_sha256")

_S = pa.large_string()
_I = pa.int64()
_F = pa.float64()
ACTIVITY_SCHEMA = pa.schema(
    [
        ("activity_id", _I),
        ("action_type", _S),
        ("activity_comment", _S),
        ("bao_endpoint", _S),
        ("data_validity_comment", _S),
        ("potential_duplicate", _I),
        ("qudt_units", _S),
        ("record_id", _I),
        ("relation", _S),
        ("src_id", _I),
        ("standard_flag", _I),
        ("standard_relation", _S),
        ("standard_text_value", _S),
        ("standard_type", _S),
        ("standard_units", _S),
        ("standard_upper_value", _F),
        ("standard_value", _F),
        ("text_value", _S),
        ("toid", _I),
        ("type", _S),
        ("units", _S),
        ("uo_units", _S),
        ("upper_value", _F),
        ("value", _F),
        ("modality", _S),
        ("assay_chembl_id", _S),
        ("assay_description", _S),
        ("assay_type", _S),
        ("assay_test_type", _S),
        ("assay_category", _S),
        ("assay_organism", _S),
        ("assay_tax_id", _I),
        ("assay_strain", _S),
        ("assay_tissue", _S),
        ("assay_cell_type", _S),
        ("assay_subcellular_fraction", _S),
        ("relationship_type", _S),
        ("src_assay_id", _S),
        ("cell_id", _I),
        ("tissue_id", _I),
        ("variant_id", _I),
        ("assay_group", _S),
        ("bao_format", _S),
        ("confidence_score", _I),
        ("target_chembl_id", _S),
        ("target_pref_name", _S),
        ("target_type", _S),
        ("target_organism", _S),
        ("target_tax_id", _I),
        ("molecule_chembl_id", _S),
        ("molecule_pref_name", _S),
        ("canonical_smiles", _S),
        ("standard_inchi_key", _S),
        ("document_chembl_id", _S),
        ("document_journal", _S),
        ("document_year", _I),
        ("document_doi", _S),
        ("pubmed_id", _I),
        ("patent_id", _S),
        ("document_title", _S),
        ("document_type", _S),
        ("document_chembl_release_id", _I),
        ("activity_source_name", _S),
        ("activity_source_description", _S),
        ("component_accessions", _S),
        ("component_sequences", _S),
        ("component_types", _S),
    ]
)
DEVELOPMENT_SCHEMA = pa.schema(
    [
        ("molecule_row_id", _I),
        ("molecule_chembl_id", _S),
        ("molecule_pref_name", _S),
        ("molecule_type", _S),
        ("max_phase", _F),
        ("first_approval", _I),
        ("withdrawn_flag", _I),
        ("black_box_warning", _I),
        ("therapeutic_flag", _I),
        ("annotation_role", _S),
    ]
)
TARGET_COMPONENT_SCHEMA = pa.schema(
    [
        ("target_chembl_id", _S),
        ("target_name", _S),
        ("target_type", _S),
        ("target_organism", _S),
        ("target_tax_id", _I),
        ("component_id", _I),
        ("homologue", _I),
        ("accession", _S),
        ("component_type", _S),
        ("sequence", _S),
        ("sequence_md5sum", _S),
        ("component_organism", _S),
        ("component_tax_id", _I),
    ]
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def arrow_schema_contract(schema: pa.Schema) -> dict[str, Any]:
    fields = [
        {"name": field.name, "type": str(field.type), "nullable": bool(field.nullable)}
        for field in schema.remove_metadata()
    ]
    contract: dict[str, Any] = {
        "format": "apache-arrow-field-contract-v1",
        "fields": fields,
        "schema_metadata": None,
    }
    contract["sha256"] = hashlib.sha256(canonical_json(contract).encode("utf-8")).hexdigest()
    return contract


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def require_plain_file(path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Expected a regular non-symlink file: {path}")


def verify_identity(top: dict[str, Any], view_name: str, child: dict[str, Any]) -> None:
    if child.get("view_name") != view_name:
        raise ValueError(f"Child view identity mismatch for {view_name}")
    for field in IDENTITY_FIELDS:
        if child.get(field) != top.get(field):
            raise ValueError(f"{view_name} differs from top-level {field}")


def verify_parquet_contract(
    path: Path,
    record: dict[str, Any],
    expected_schema: pa.Schema,
) -> tuple[int, int, pa.Schema]:
    require_plain_file(path)
    size = path.stat().st_size
    if size != int(record["size_bytes"]):
        raise ValueError(f"Byte-size mismatch: {path}")
    if sha256_file(path) != record["sha256"]:
        raise ValueError(f"SHA-256 mismatch: {path}")
    parquet = pq.ParquetFile(path)
    rows = int(parquet.metadata.num_rows)
    if rows != int(record["rows"]):
        raise ValueError(f"Parquet footer row mismatch: {path}")
    physical_schema = parquet.schema_arrow.remove_metadata()
    if not physical_schema.equals(expected_schema, check_metadata=True):
        raise ValueError(f"Physical Arrow schema mismatch: {path}")
    contract = arrow_schema_contract(expected_schema)
    if record.get("arrow_schema_sha256") != contract["sha256"]:
        raise ValueError(f"Part Arrow schema fingerprint mismatch: {path}")
    return size, rows, physical_schema


def scan_strict_identifier_order(parts: list[Path], column: str) -> tuple[int, set[int]]:
    previous: int | None = None
    observed: set[int] = set()
    rows = 0
    for path in parts:
        parquet = pq.ParquetFile(path)
        if column not in parquet.schema_arrow.names:
            raise ValueError(f"{path} lacks required identifier {column}")
        for batch in parquet.iter_batches(columns=[column], batch_size=131_072):
            values = batch.column(0).to_pylist()
            for raw_value in values:
                if raw_value is None:
                    raise ValueError(f"Null {column} in {path}")
                value = int(raw_value)
                if previous is not None and value <= previous:
                    raise ValueError(f"Non-increasing or duplicate {column} in {path}: {value}")
                if value in observed:
                    raise ValueError(f"Duplicate {column} in {path}: {value}")
                observed.add(value)
                previous = value
                rows += 1
    return rows, observed


def verify_target_components(path: Path, expected_rows: int) -> dict[str, int]:
    parquet = pq.ParquetFile(path)
    accessions: set[str] = set()
    sequences: set[str] = set()
    keys: set[tuple[str, int | None]] = set()
    previous: tuple[str, int] | None = None
    rows = 0
    for batch in parquet.iter_batches(
        columns=["target_chembl_id", "component_id", "accession", "sequence"],
        batch_size=131_072,
    ):
        for target, component, accession, sequence in zip(
            *(column.to_pylist() for column in batch.columns), strict=True
        ):
            target_text = str(target or "")
            component_int = int(component) if component is not None else None
            key = (target_text, component_int)
            if key in keys:
                raise ValueError(f"Duplicate target/component row: {key}")
            keys.add(key)
            order_key = (target_text, component_int if component_int is not None else -1)
            if previous is not None and order_key < previous:
                raise ValueError("Target-component rows are not deterministically ordered")
            previous = order_key
            if str(accession or "").strip():
                accessions.add(str(accession).strip())
            if str(sequence or "").strip():
                sequences.add(str(sequence).strip())
            rows += 1
    if rows != expected_rows:
        raise ValueError("Target-component scan count does not match its manifest")
    return {
        "rows": rows,
        "unique_target_component_keys": len(keys),
        "unique_nonblank_accessions": len(accessions),
        "unique_nonblank_sequences": len(sequences),
    }


def verify_specialized_root(root: Path) -> dict[str, Any]:
    root = root.resolve()
    top_path = root / "specialized_views_manifest.json"
    require_plain_file(top_path)
    top = load_json(top_path)
    views = top.get("views")
    if not isinstance(views, dict) or set(views) != EXPECTED_VIEWS:
        raise ValueError("Top manifest has an unexpected specialized-view set")

    expected_files = {top_path}
    physical_bytes = top_path.stat().st_size
    physical_rows = 0
    activity_schemas: set[str] = set()
    activity_ids: dict[str, set[int]] = {}
    view_results: dict[str, Any] = {}

    for name in sorted(EXPECTED_VIEWS):
        child_path = root / f"{name}_manifest.json"
        require_plain_file(child_path)
        expected_files.add(child_path)
        physical_bytes += child_path.stat().st_size
        child = load_json(child_path)
        if child != views[name]:
            raise ValueError(f"Embedded and standalone child manifests differ: {name}")
        verify_identity(top, name, child)

        if name == "target_components":
            target_contract = arrow_schema_contract(TARGET_COMPONENT_SCHEMA)
            if child.get("arrow_schema") != target_contract:
                raise ValueError("Target-component declared Arrow schema mismatch")
            part_path = root / str(child["path"])
            expected_files.add(part_path)
            record = {
                "size_bytes": child["size_bytes"],
                "sha256": child["sha256"],
                "rows": child["row_count"],
                "arrow_schema_sha256": child["arrow_schema_sha256"],
            }
            size, rows, _ = verify_parquet_contract(
                part_path,
                record,
                TARGET_COMPONENT_SCHEMA,
            )
            physical_bytes += size
            physical_rows += rows
            view_results[name] = verify_target_components(part_path, rows)
            continue

        parts = child.get("parts")
        if not isinstance(parts, list) or len(parts) != int(child["part_count"]):
            raise ValueError(f"Part-count mismatch in child manifest: {name}")
        paths: list[Path] = []
        part_rows = 0
        schema: pa.Schema | None = None
        expected_schema = (
            DEVELOPMENT_SCHEMA if name == "molecule_development_annotations" else ACTIVITY_SCHEMA
        )
        if child.get("arrow_schema") != arrow_schema_contract(expected_schema):
            raise ValueError(f"Declared Arrow schema mismatch: {name}")
        for part in parts:
            part_path = root / str(part["path"])
            expected_files.add(part_path)
            size, rows, current_schema = verify_parquet_contract(
                part_path,
                part,
                expected_schema,
            )
            physical_bytes += size
            physical_rows += rows
            part_rows += rows
            paths.append(part_path)
            if schema is None:
                schema = current_schema
            elif not current_schema.equals(schema, check_metadata=True):
                raise ValueError(f"Schema drift within view: {name}")
        if part_rows != int(child["row_count"]):
            raise ValueError(f"Part rows do not sum to child rows: {name}")

        identifier = "molecule_row_id" if name == "molecule_development_annotations" else "activity_id"
        scanned_rows, identifiers = scan_strict_identifier_order(paths, identifier)
        if scanned_rows != part_rows:
            raise ValueError(f"Identifier scan count mismatch: {name}")
        if name in ACTIVITY_VIEWS:
            if schema is None:
                raise ValueError(f"Missing activity schema: {name}")
            activity_schemas.add(str(schema))
            activity_ids[name] = identifiers
        view_results[name] = {
            "part_count": len(parts),
            "rows": part_rows,
            "unique_identifiers": len(identifiers),
            "first_identifier": min(identifiers),
            "last_identifier": max(identifiers),
        }

    if len(activity_schemas) != 1:
        raise ValueError("Activity schemas differ across specialized views")

    actual_files: set[Path] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Specialized root contains a symlink: {path}")
        if path.is_file():
            actual_files.add(path.resolve())
    expected_resolved = {path.resolve() for path in expected_files}
    if actual_files != expected_resolved:
        extra = sorted(path.relative_to(root).as_posix() for path in actual_files - expected_resolved)
        missing = sorted(path.relative_to(root).as_posix() for path in expected_resolved - actual_files)
        raise ValueError(f"Physical membership mismatch; extra={extra}, missing={missing}")

    herg_ids = activity_ids["herg_all_endpoints"]
    overlaps = {
        label: len(herg_ids & activity_ids[view_name]) for label, view_name in EXPECTED_OVERLAP_KEYS.items()
    }
    if overlaps != top.get("overlap_row_counts"):
        raise ValueError("Physical overlap counts differ from the top manifest")

    return {
        "status": "passed",
        "root": root.as_posix(),
        "top_manifest_sha256": sha256_file(top_path),
        "source_version": top["source_version"],
        "database_sha256": top["database_sha256"],
        "exact_physical_file_count": len(actual_files),
        "exact_physical_bytes": physical_bytes,
        "parquet_rows_across_overlapping_views": physical_rows,
        "activity_schema_count": len(activity_schemas),
        "overlap_row_counts": overlaps,
        "views": view_results,
        "substantive_training_started": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=Path("research/data/platform/interim/chembl_37_bulk/specialized_views"),
    )
    args = parser.parse_args()
    print(json.dumps(verify_specialized_root(args.root), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
