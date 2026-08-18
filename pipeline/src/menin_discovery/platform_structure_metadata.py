"""Deterministic, CPU-only PDB structure-metadata coverage analysis.

The stage consumes only the small official PDBe SIFTS UniProt↔PDB chain map
and the wwPDB entry-type table. It never downloads coordinates or predicted
structure models, creates no biological labels, and trains no model.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import re
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from .platform_external_normalization import (
    arrow_schema_sha256,
    canonical_json_bytes,
    document_with_sha256,
    sha256_file,
    verify_document_sha256,
)

SCHEMA_VERSION = "platform-structure-metadata/1.0"
PARSER_VERSION = "platform_structure_metadata/1.1"
SIFTS_URL = "https://ftp.ebi.ac.uk/pub/databases/msd/sifts/flatfiles/tsv/pdb_chain_uniprot.tsv.gz"
ENTRY_TYPE_URL = "https://files.wwpdb.org/pub/pdb/derived_data/pdb_entry_type.txt"
SIFTS_DOC_URL = "https://www.ebi.ac.uk/pdbe/docs/sifts/quick.html"
SIFTS_METHOD_URL = "https://www.ebi.ac.uk/pdbe/docs/sifts/methodology.html"
RIGHTS_URL = "https://www.ebi.ac.uk/pdbe/about/public-data-access-statement"
SIFTS_COLUMNS = (
    "PDB",
    "CHAIN",
    "SP_PRIMARY",
    "RES_BEG",
    "RES_END",
    "PDB_BEG",
    "PDB_END",
    "SP_BEG",
    "SP_END",
)
RELEASE_RE = re.compile(
    r"^# (?P<date>\d{4}/\d{2}/\d{2}) - (?P<time>\d{2}:\d{2}) \| "
    r"PDB: (?P<pdb_version>[^|]+) \| UniProt: (?P<uniprot_version>\S+)$"
)
PDB_ID_RE = re.compile(r"^[0-9][a-z0-9]{3}$")
ACCESSION_RE = re.compile(r"^[A-Z0-9]+(?:-[0-9]+)?$")
KNOWN_METHODS = {"diffraction", "NMR", "EM", "other"}


class StructureMetadataError(RuntimeError):
    """Raised when an input, parser, or zero-training contract fails closed."""


SEGMENT_SCHEMA = pa.schema(
    [
        pa.field("source_id", pa.string(), nullable=False),
        pa.field("pdb_id", pa.string(), nullable=False),
        pa.field("chain_id", pa.string(), nullable=False),
        pa.field("uniprot_accession", pa.string(), nullable=False),
        pa.field("res_beg", pa.int64()),
        pa.field("res_end", pa.int64()),
        pa.field("pdb_beg", pa.string()),
        pa.field("pdb_end", pa.string()),
        pa.field("uniprot_beg", pa.int64()),
        pa.field("uniprot_end", pa.int64()),
        pa.field("entry_molecule_type", pa.string(), nullable=False),
        pa.field("entry_method_class", pa.string(), nullable=False),
        pa.field("structure_source_class", pa.string(), nullable=False),
        pa.field("predicted_structure", pa.bool_(), nullable=False),
        pa.field("model_label_admitted", pa.bool_(), nullable=False),
    ]
)

COVERAGE_SCHEMA = pa.schema(
    [
        pa.field("universe_kind", pa.string(), nullable=False),
        pa.field("protein_id", pa.string(), nullable=False),
        pa.field("uniprot_accession", pa.string()),
        pa.field("sequence_sha256", pa.string()),
        pa.field("sequence_length", pa.int64()),
        pa.field("accession_resolution_status", pa.string(), nullable=False),
        pa.field("sifts_release_accession_exact_match", pa.bool_(), nullable=False),
        pa.field("pdb_entry_count", pa.int64(), nullable=False),
        pa.field("pdb_chain_count", pa.int64(), nullable=False),
        pa.field("mapping_segment_count", pa.int64(), nullable=False),
        pa.field("diffraction_entry_count", pa.int64(), nullable=False),
        pa.field("nmr_entry_count", pa.int64(), nullable=False),
        pa.field("em_entry_count", pa.int64(), nullable=False),
        pa.field("other_method_entry_count", pa.int64(), nullable=False),
        pa.field("mapped_uniprot_min", pa.int64()),
        pa.field("mapped_uniprot_max", pa.int64()),
        pa.field("span_fraction_of_frozen_sequence", pa.float64()),
        pa.field("coverage_interpretation", pa.string(), nullable=False),
        pa.field("construct_identity_verified", pa.bool_(), nullable=False),
        pa.field("sequence_version_verified", pa.bool_(), nullable=False),
        pa.field("predicted_structure_count", pa.int64(), nullable=False),
        pa.field("model_label_admitted", pa.bool_(), nullable=False),
    ]
)


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
        raise StructureMetadataError(f"Missing or symlinked {context}: {path}")
    return path.resolve()


def _safe_relative(value: Any, *, context: str) -> PurePosixPath:
    raw = str(value)
    path = PurePosixPath(raw)
    if not raw or path.is_absolute() or ".." in path.parts:
        raise StructureMetadataError(f"Unsafe {context} path: {path}")
    return path


def _load_json(path: Path, *, identified: bool) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise StructureMetadataError(f"Missing or symlinked JSON: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise StructureMetadataError(f"Unreadable JSON: {path}") from error
    if not isinstance(value, dict) or (identified and not verify_document_sha256(value)):
        raise StructureMetadataError(f"JSON identity failed: {path}")
    return value


def _http_header_values(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        values[key.strip().casefold()] = value.strip()
    return values


def _parse_int(value: str, *, context: str) -> int | None:
    stripped = value.strip()
    if not stripped:
        return None
    try:
        return int(stripped)
    except ValueError as error:
        raise StructureMetadataError(f"Invalid integer {context}: {value!r}") from error


def parse_sifts_header(line: str) -> dict[str, str]:
    match = RELEASE_RE.fullmatch(line.rstrip("\r\n"))
    if not match:
        raise StructureMetadataError(f"Unexpected SIFTS release header: {line!r}")
    return match.groupdict()


def parse_entry_type_line(line: str) -> tuple[str, str, str]:
    fields = line.rstrip("\r\n").split("\t")
    if len(fields) != 3:
        raise StructureMetadataError(f"wwPDB entry-type width drift: {len(fields)}")
    pdb_id, molecule_type, method = fields
    if not PDB_ID_RE.fullmatch(pdb_id) or not molecule_type or method not in KNOWN_METHODS:
        raise StructureMetadataError(f"Invalid wwPDB entry-type record: {fields}")
    return pdb_id, molecule_type, method


def _raw_inventory(raw: Path) -> list[dict[str, Any]]:
    expected = (
        "pdb_chain_uniprot.http_headers.txt",
        "pdb_chain_uniprot.tsv.gz",
        "pdb_entry_type.http_headers.txt",
        "pdb_entry_type.txt",
    )
    observed = {
        path.name for path in raw.iterdir() if path.is_file() and path.name != "acquisition_manifest.json"
    }
    if observed != set(expected):
        raise StructureMetadataError(
            f"Raw membership drift: missing={sorted(set(expected) - observed)}, "
            f"unexpected={sorted(observed - set(expected))}"
        )
    return [
        {"path": name, "bytes": (raw / name).stat().st_size, "sha256": sha256_file(raw / name)}
        for name in expected
    ]


def build_acquisition_manifest(raw_root: str | os.PathLike[str]) -> dict[str, Any]:
    raw = _safe_directory(raw_root, context="raw structure metadata")
    manifest_path = raw / "acquisition_manifest.json"
    if manifest_path.exists():
        raise StructureMetadataError(f"Acquisition manifest exists: {manifest_path}")
    sifts_headers = _http_header_values(raw / "pdb_chain_uniprot.http_headers.txt")
    type_headers = _http_header_values(raw / "pdb_entry_type.http_headers.txt")
    if int(sifts_headers.get("content-length", -1)) != (raw / "pdb_chain_uniprot.tsv.gz").stat().st_size:
        raise StructureMetadataError("SIFTS HTTP content-length mismatch")
    if int(type_headers.get("content-length", -1)) != (raw / "pdb_entry_type.txt").stat().st_size:
        raise StructureMetadataError("wwPDB entry-type HTTP content-length mismatch")
    with gzip.open(raw / "pdb_chain_uniprot.tsv.gz", "rt", encoding="utf-8", newline="") as handle:
        release = parse_sifts_header(handle.readline())
        if tuple(handle.readline().rstrip("\r\n").split("\t")) != SIFTS_COLUMNS:
            raise StructureMetadataError("SIFTS column header drift")
    entries = _raw_inventory(raw)
    body = {
        "schema_version": SCHEMA_VERSION,
        "source_bundle_id": "pdbe_sifts_wwpdb_entry_type_2026_08_03",
        "release": release,
        "sources": [
            {
                "source_id": "pdbe_sifts_chain_uniprot",
                "url": SIFTS_URL,
                "documentation_url": SIFTS_DOC_URL,
                "methodology_url": SIFTS_METHOD_URL,
                "retrieved_http_date": sifts_headers.get("date"),
                "last_modified": sifts_headers.get("last-modified"),
                "etag": sifts_headers.get("etag"),
                "bytes": (raw / "pdb_chain_uniprot.tsv.gz").stat().st_size,
                "sha256": sha256_file(raw / "pdb_chain_uniprot.tsv.gz"),
            },
            {
                "source_id": "wwpdb_entry_type",
                "url": ENTRY_TYPE_URL,
                "retrieved_http_date": type_headers.get("date"),
                "last_modified": type_headers.get("last-modified"),
                "etag": type_headers.get("etag"),
                "bytes": (raw / "pdb_entry_type.txt").stat().st_size,
                "sha256": sha256_file(raw / "pdb_entry_type.txt"),
            },
        ],
        "rights": {
            "status": "public_domain_cc0_statement_located_human_review_still_required_for_release",
            "statement_url": RIGHTS_URL,
            "statement": "PDBe states that primary and derived archive data are available under CC0 and site terms.",
            "release_gate": "retain citation and recheck current source terms before redistribution",
        },
        "citations": [
            "Dana JM et al. SIFTS: updated Structure Integration with Function, Taxonomy and Sequences resource. NAR 2019;47:D482-D489. doi:10.1093/nar/gky1114",
            "Velankar S et al. SIFTS: Structure Integration with Function, Taxonomy and Sequences resource. NAR 2013;41:D483-D489. doi:10.1093/nar/gks1258",
        ],
        "checksum_evidence": {
            "official_sidecar_checksum_found": False,
            "http_etags_preserved": True,
            "local_sha256_computed": True,
            "limitation": "official directory did not expose a cryptographic sidecar checksum for these two endpoints",
        },
        "bundle_inventory": {
            "entries": entries,
            "entries_sha256": hashlib.sha256(canonical_json_bytes(entries)).hexdigest(),
            "entry_count": len(entries),
            "total_bytes": sum(int(item["bytes"]) for item in entries),
            "excluded_paths": ["acquisition_manifest.json"],
        },
        "coordinate_files_downloaded": 0,
        "predicted_structure_files_downloaded": 0,
        "model_labels_admitted": 0,
        "substantive_model_training_performed": False,
    }
    return _atomic_json(manifest_path, body)


def verify_acquisition(raw_root: str | os.PathLike[str]) -> dict[str, Any]:
    raw = _safe_directory(raw_root, context="raw structure metadata")
    manifest = _load_json(raw / "acquisition_manifest.json", identified=True)
    inventory = manifest.get("bundle_inventory")
    if not isinstance(inventory, dict):
        raise StructureMetadataError("Raw inventory missing")
    entries = _raw_inventory(raw)
    if entries != inventory.get("entries") or hashlib.sha256(
        canonical_json_bytes(entries)
    ).hexdigest() != inventory.get("entries_sha256"):
        raise StructureMetadataError("Raw inventory identity failed")
    if (
        manifest.get("coordinate_files_downloaded") != 0
        or manifest.get("predicted_structure_files_downloaded") != 0
        or manifest.get("model_labels_admitted") != 0
    ):
        raise StructureMetadataError("Raw zero-coordinate/label contract failed")
    return manifest


def _validate_runtime_bindings(
    manifest: Mapping[str, Any], acquisition: Mapping[str, Any], raw: Path
) -> None:
    bindings = manifest.get("input_bindings")
    expected = {
        "acquisition_manifest_physical_sha256": sha256_file(raw / "acquisition_manifest.json"),
        "acquisition_manifest_internal_sha256": acquisition.get("manifest_sha256"),
        "analyzer_code_sha256": sha256_file(Path(__file__).resolve()),
    }
    if not isinstance(bindings, Mapping) or any(
        bindings.get(key) != value for key, value in expected.items()
    ):
        raise StructureMetadataError("Structure-metadata runtime input/code binding changed")


def _entry_types(path: Path) -> tuple[dict[str, tuple[str, str]], dict[str, Any]]:
    values: dict[str, tuple[str, str]] = {}
    methods: Counter[str] = Counter()
    molecule_types: Counter[str] = Counter()
    previous = ""
    with path.open("r", encoding="utf-8", newline="") as handle:
        for line in handle:
            pdb_id, molecule_type, method = parse_entry_type_line(line)
            if pdb_id <= previous:
                raise StructureMetadataError("wwPDB entry-type rows are duplicate or unsorted")
            previous = pdb_id
            values[pdb_id] = (molecule_type, method)
            methods[method] += 1
            molecule_types[molecule_type] += 1
    return values, {
        "entry_count": len(values),
        "method_counts": dict(sorted(methods.items())),
        "molecule_type_counts": dict(sorted(molecule_types.items())),
    }


def _canonical_universe(root: Path) -> tuple[list[dict[str, Any]], str, int]:
    manifest_path = root / "build_manifest.json"
    manifest = _load_json(manifest_path, identified=False)
    inventory = {str(item["path"]): item for item in manifest.get("component_inventory", [])}
    rows: list[dict[str, Any]] = []
    for path in sorted((root / "proteins").glob("part-*.parquet")):
        relative = path.relative_to(root).as_posix()
        item = inventory.get(relative)
        if (
            not isinstance(item, dict)
            or path.stat().st_size != int(item.get("size_bytes", -1))
            or sha256_file(path) != item.get("sha256")
        ):
            raise StructureMetadataError(f"Canonical protein component drift: {relative}")
        parquet = pq.ParquetFile(path)
        required = {
            "protein_id",
            "uniprot_accession",
            "sequence",
            "sequence_sha256",
            "identity_resolution_status",
        }
        if not required.issubset(parquet.schema_arrow.names):
            raise StructureMetadataError("Canonical protein schema drift")
        for batch in parquet.iter_batches(columns=sorted(required), batch_size=4096):
            for row in batch.to_pylist():
                sequence = row["sequence"]
                rows.append(
                    {
                        "universe_kind": "canonical_chembl37_protein",
                        "protein_id": str(row["protein_id"]),
                        "uniprot_accession": row["uniprot_accession"],
                        "sequence_sha256": row["sequence_sha256"],
                        "sequence_length": len(sequence) if isinstance(sequence, str) else None,
                        "accession_resolution_status": str(row["identity_resolution_status"] or "unknown"),
                    }
                )
    return (
        rows,
        sha256_file(manifest_path),
        int(manifest.get("entity_counts", {}).get("protein_constructs", -1)),
    )


def _external_uniprot_universe(root: Path) -> tuple[list[dict[str, Any]], str]:
    manifest_path = root / "external_public_normalized_manifest.json"
    _load_json(manifest_path, identified=True)
    path = root / "uniprot/returned_entries.parquet"
    rows: list[dict[str, Any]] = []
    for batch in pq.ParquetFile(path).iter_batches(
        columns=["returned_primary_accession", "sequence", "sequence_sha256", "sequence_status"],
        batch_size=4096,
    ):
        for row in batch.to_pylist():
            sequence = row["sequence"]
            accession = str(row["returned_primary_accession"])
            rows.append(
                {
                    "universe_kind": "frozen_external_uniprot_returned_entry",
                    "protein_id": f"uniprot:{accession}",
                    "uniprot_accession": accession,
                    "sequence_sha256": row["sequence_sha256"],
                    "sequence_length": len(sequence) if isinstance(sequence, str) else None,
                    "accession_resolution_status": str(row["sequence_status"]),
                }
            )
    return rows, sha256_file(manifest_path)


def _write_table(path: Path, schema: pa.Schema, rows: Iterable[dict[str, Any]]) -> tuple[int, str]:
    writer = pq.ParquetWriter(path, schema, compression="zstd", use_dictionary=True, write_statistics=True)
    count = 0
    buffer: list[dict[str, Any]] = []
    try:
        for row in rows:
            buffer.append(row)
            if len(buffer) >= 4096:
                writer.write_table(pa.Table.from_pylist(buffer, schema=schema), row_group_size=4096)
                count += len(buffer)
                buffer.clear()
        if buffer:
            writer.write_table(pa.Table.from_pylist(buffer, schema=schema), row_group_size=4096)
            count += len(buffer)
    finally:
        writer.close()
    return count, arrow_schema_sha256(schema)


def _sifts_rows(
    path: Path,
    entry_types: Mapping[str, tuple[str, str]],
    universe_accessions: set[str],
    summaries: dict[str, dict[str, Any]],
    audit: dict[str, Any],
) -> Iterable[dict[str, Any]]:
    chain_accessions: set[str] = set()
    previous_chain: tuple[str, str] | None = None
    ambiguous_chains = 0
    missing_entry_type = 0
    reversed_uniprot_ranges = 0
    total = 0
    accession_counts: Counter[str] = Counter()
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        parse_sifts_header(handle.readline())
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != SIFTS_COLUMNS:
            raise StructureMetadataError("SIFTS column schema drift")
        for row in reader:
            pdb_id = str(row["PDB"]).casefold()
            chain = str(row["CHAIN"])
            accession = str(row["SP_PRIMARY"])
            if not PDB_ID_RE.fullmatch(pdb_id) or not chain or not ACCESSION_RE.fullmatch(accession):
                raise StructureMetadataError(f"Invalid SIFTS identity: {pdb_id}/{chain}/{accession}")
            chain_key = (pdb_id, chain)
            if previous_chain is not None and chain_key < previous_chain:
                raise StructureMetadataError("SIFTS chain rows are unsorted")
            if previous_chain is not None and chain_key != previous_chain:
                ambiguous_chains += int(len(chain_accessions) > 1)
                chain_accessions.clear()
            previous_chain = chain_key
            chain_accessions.add(accession)
            entry = entry_types.get(pdb_id)
            if entry is None:
                molecule_type, method = "missing", "missing"
                missing_entry_type += 1
            else:
                molecule_type, method = entry
            sp_beg = _parse_int(str(row["SP_BEG"]), context="SP_BEG")
            sp_end = _parse_int(str(row["SP_END"]), context="SP_END")
            if sp_beg is not None and sp_end is not None and sp_beg > sp_end:
                reversed_uniprot_ranges += 1
            if accession in universe_accessions:
                summary = summaries.setdefault(
                    accession,
                    {
                        "pdb": set(),
                        "chains": set(),
                        "segments": 0,
                        "methods": defaultdict(set),
                        "min": None,
                        "max": None,
                    },
                )
                summary["pdb"].add(pdb_id)
                summary["chains"].add((pdb_id, chain))
                summary["segments"] += 1
                summary["methods"][method].add(pdb_id)
                available_bounds = [bound for bound in (sp_beg, sp_end) if bound is not None]
                if available_bounds:
                    lower = min(available_bounds)
                    upper = max(available_bounds)
                    summary["min"] = lower if summary["min"] is None else min(summary["min"], lower)
                    summary["max"] = upper if summary["max"] is None else max(summary["max"], upper)
            total += 1
            accession_counts[accession] += 1
            yield {
                "source_id": "pdbe_sifts_chain_uniprot",
                "pdb_id": pdb_id,
                "chain_id": chain,
                "uniprot_accession": accession,
                "res_beg": _parse_int(str(row["RES_BEG"]), context="RES_BEG"),
                "res_end": _parse_int(str(row["RES_END"]), context="RES_END"),
                "pdb_beg": str(row["PDB_BEG"]).strip() or None,
                "pdb_end": str(row["PDB_END"]).strip() or None,
                "uniprot_beg": sp_beg,
                "uniprot_end": sp_end,
                "entry_molecule_type": molecule_type,
                "entry_method_class": method,
                "structure_source_class": "wwpdb_archive_mapping_not_predicted_model_resource",
                "predicted_structure": False,
                "model_label_admitted": False,
            }
    if previous_chain is not None:
        ambiguous_chains += int(len(chain_accessions) > 1)
    audit.update(
        {
            "segment_rows": total,
            "unique_uniprot_accessions": len(accession_counts),
            "chains_mapping_to_multiple_uniprot_accessions": ambiguous_chains,
            "segments_missing_entry_type_metadata": missing_entry_type,
            "segments_with_reversed_uniprot_endpoint_numbering": reversed_uniprot_ranges,
        }
    )


def _coverage_rows(
    universe: Sequence[dict[str, Any]], summaries: Mapping[str, dict[str, Any]]
) -> Iterable[dict[str, Any]]:
    for protein in sorted(universe, key=lambda item: (item["universe_kind"], item["protein_id"])):
        accession = protein["uniprot_accession"]
        summary = summaries.get(str(accession)) if accession else None
        sequence_length = protein["sequence_length"]
        span = None
        if summary and sequence_length and summary["min"] is not None and summary["max"] is not None:
            span = (summary["max"] - summary["min"] + 1) / sequence_length
        yield {
            **protein,
            "sifts_release_accession_exact_match": summary is not None,
            "pdb_entry_count": len(summary["pdb"]) if summary else 0,
            "pdb_chain_count": len(summary["chains"]) if summary else 0,
            "mapping_segment_count": int(summary["segments"]) if summary else 0,
            "diffraction_entry_count": len(summary["methods"].get("diffraction", set())) if summary else 0,
            "nmr_entry_count": len(summary["methods"].get("NMR", set())) if summary else 0,
            "em_entry_count": len(summary["methods"].get("EM", set())) if summary else 0,
            "other_method_entry_count": len(summary["methods"].get("other", set())) if summary else 0,
            "mapped_uniprot_min": summary["min"] if summary else None,
            "mapped_uniprot_max": summary["max"] if summary else None,
            "span_fraction_of_frozen_sequence": span,
            "coverage_interpretation": "outer_span_proxy_not_observed_residue_coverage"
            if summary
            else "no_exact_accession_mapping_in_release",
            "construct_identity_verified": False,
            "sequence_version_verified": False,
            "predicted_structure_count": 0,
            "model_label_admitted": False,
        }


def _artifact(path: Path, root: Path, rows: int, schema_sha: str) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "rows": rows,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "arrow_schema_sha256": schema_sha,
    }


def build_structure_metadata(
    raw_root: str | os.PathLike[str],
    canonical_root: str | os.PathLike[str],
    external_normalized_root: str | os.PathLike[str],
    output_root: str | os.PathLike[str],
    report_root: str | os.PathLike[str],
) -> dict[str, Any]:
    raw = _safe_directory(raw_root, context="raw structure metadata")
    canonical = _safe_directory(canonical_root, context="canonical input")
    external = _safe_directory(external_normalized_root, context="external normalized input")
    output = Path(output_root).resolve()
    reports = Path(report_root).resolve()
    if output.exists() or reports.exists():
        raise StructureMetadataError("Output/report root exists and will not be replaced")
    acquisition = verify_acquisition(raw)
    entry_types, entry_audit = _entry_types(raw / "pdb_entry_type.txt")
    canonical_rows, canonical_sha, construct_count = _canonical_universe(canonical)
    external_rows, external_sha = _external_uniprot_universe(external)
    universe = [*canonical_rows, *external_rows]
    accessions = {str(row["uniprot_accession"]) for row in universe if row["uniprot_accession"]}
    summaries: dict[str, dict[str, Any]] = {}
    sifts_audit: dict[str, Any] = {}
    output.mkdir(parents=True, exist_ok=False)
    segment_path = output / "sifts_uniprot_pdb_segments.parquet"
    segment_count, segment_schema_sha = _write_table(
        segment_path,
        SEGMENT_SCHEMA,
        _sifts_rows(raw / "pdb_chain_uniprot.tsv.gz", entry_types, accessions, summaries, sifts_audit),
    )
    coverage_path = output / "protein_structure_coverage.parquet"
    coverage_count, coverage_schema_sha = _write_table(
        coverage_path, COVERAGE_SCHEMA, _coverage_rows(universe, summaries)
    )
    artifacts = [
        _artifact(segment_path, output, segment_count, segment_schema_sha),
        _artifact(coverage_path, output, coverage_count, coverage_schema_sha),
    ]
    artifacts.sort(key=lambda item: item["path"])
    manifest = _atomic_json(
        output / "structure_metadata_manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "parser_version": PARSER_VERSION,
            "dataset_id": "structure_metadata_pdbe_sifts_2026_08_03",
            "input_bindings": {
                "acquisition_manifest_physical_sha256": sha256_file(raw / "acquisition_manifest.json"),
                "acquisition_manifest_internal_sha256": acquisition["manifest_sha256"],
                "analyzer_code_sha256": sha256_file(Path(__file__).resolve()),
                "canonical_manifest_physical_sha256": canonical_sha,
                "external_normalized_manifest_physical_sha256": external_sha,
            },
            "release": acquisition["release"],
            "output_inventory": {
                "entries": artifacts,
                "entries_sha256": hashlib.sha256(canonical_json_bytes(artifacts)).hexdigest(),
                "entry_count": len(artifacts),
                "total_bytes": sum(int(item["bytes"]) for item in artifacts),
                "excluded_paths": ["structure_metadata_manifest.json"],
            },
            "sifts_audit": sifts_audit,
            "entry_type_audit": entry_audit,
            "universe_rows": coverage_count,
            "coordinate_files_downloaded": 0,
            "predicted_structure_files_downloaded": 0,
            "model_labels_admitted": 0,
            "substantive_model_training_performed": False,
        },
    )
    coverage_counts: Counter[str] = Counter()
    method_totals: Counter[str] = Counter()
    universe_accession_multiplicity = Counter(
        (str(row["universe_kind"]), str(row["uniprot_accession"]))
        for row in universe
        if row["uniprot_accession"]
    )
    identity_ambiguity: Counter[str] = Counter()
    for (kind, _accession), count in universe_accession_multiplicity.items():
        identity_ambiguity[f"{kind}:accessions_mapping_to_multiple_frozen_records"] += int(count > 1)
        identity_ambiguity[f"{kind}:frozen_records_in_multi_record_accession_groups"] += (
            count if count > 1 else 0
        )
    for batch in pq.ParquetFile(coverage_path).iter_batches(batch_size=4096):
        for row in batch.to_pylist():
            kind = str(row["universe_kind"])
            coverage_counts[f"{kind}:total"] += 1
            coverage_counts[f"{kind}:mapped"] += int(row["pdb_entry_count"] > 0)
            coverage_counts[f"{kind}:accession_missing"] += int(not row["uniprot_accession"])
            coverage_counts[f"{kind}:accession_present_unmapped"] += int(
                bool(row["uniprot_accession"]) and row["pdb_entry_count"] == 0
            )
            coverage_counts[f"{kind}:outer_span_fraction_gt_one"] += int(
                row["span_fraction_of_frozen_sequence"] is not None
                and float(row["span_fraction_of_frozen_sequence"]) > 1.0
            )
            for field in (
                "diffraction_entry_count",
                "nmr_entry_count",
                "em_entry_count",
                "other_method_entry_count",
            ):
                method_totals[f"{kind}:{field}"] += int(row[field])
    report_body = {
        "schema_version": SCHEMA_VERSION,
        "decision": "structure_metadata_coverage_ready_for_candidate_use_not_coordinate_or_construct_ready",
        "normalized_manifest_physical_sha256": sha256_file(output / "structure_metadata_manifest.json"),
        "source_urls": [SIFTS_URL, ENTRY_TYPE_URL],
        "documentation_urls": [SIFTS_DOC_URL, SIFTS_METHOD_URL, RIGHTS_URL],
        "release": acquisition["release"],
        "coverage_counts": dict(sorted(coverage_counts.items())),
        "frozen_universe_identity_ambiguity_counts": dict(sorted(identity_ambiguity.items())),
        "method_entry_count_sums_per_protein": dict(sorted(method_totals.items())),
        "sifts_audit": sifts_audit,
        "entry_type_audit": entry_audit,
        "canonical_construct_rows_not_exactly_reconciled": construct_count,
        "limitations": [
            "SIFTS release uses UniProt 2026.03, while the frozen external UniProt input is 2026_02; accession matches are candidates, not version identity",
            "canonical ChEMBL protein accessions may reflect a different source cutoff and can map ambiguously to multiple canonical protein records",
            "outer mapped UniProt span is not observed-residue coverage and can include gaps/unobserved residues",
            "outer-span fractions above one are retained as sequence-version or mapping-mismatch warning evidence rather than capped",
            "chain mapping does not verify construct boundaries, mutations, tags, assemblies, ligands, quality, or assay relevance",
            "wwPDB entry-type method classes are coarse; method-specific quality requires entry metadata/validation reports",
            "PDB archive mappings are kept separate from predicted structures; no predicted models were acquired",
        ],
        "rights": acquisition["rights"],
        "coordinate_files_downloaded": 0,
        "predicted_structure_files_downloaded": 0,
        "model_labels_admitted": 0,
        "substantive_model_training_performed": False,
    }
    reports.mkdir(parents=True, exist_ok=False)
    methods = reports / "methods_and_limitations.md"
    _atomic_text(methods, _methods_text(report_body))
    report_body["methods_sha256"] = sha256_file(methods)
    report = _atomic_json(reports / "structure_metadata_report.json", report_body)
    return {"manifest": manifest, "report": report}


def _methods_text(report: Mapping[str, Any]) -> str:
    counts = report["coverage_counts"]
    canonical_total = counts.get("canonical_chembl37_protein:total", 0)
    canonical_mapped = counts.get("canonical_chembl37_protein:mapped", 0)
    external_total = counts.get("frozen_external_uniprot_returned_entry:total", 0)
    external_mapped = counts.get("frozen_external_uniprot_returned_entry:mapped", 0)
    return f"""# PDB structure-metadata coverage

## Result

- Exact SIFTS accession mapping candidates exist for {canonical_mapped:,}/{canonical_total:,}
  frozen canonical protein rows and {external_mapped:,}/{external_total:,} frozen returned
  UniProt entries.
- This is **metadata coverage**, not coordinate readiness, construct equivalence, or
  experimentally observed-residue coverage.
- Zero coordinate files and zero predicted structure files were downloaded. Zero labels
  were created and no model was trained.

## Sources and method

- PDBe SIFTS `pdb_chain_uniprot.tsv.gz`: {SIFTS_URL}
- wwPDB `pdb_entry_type.txt`: {ENTRY_TYPE_URL}
- Release: PDB {report["release"]["pdb_version"]}; UniProt {report["release"]["uniprot_version"]};
  generated {report["release"]["date"]} {report["release"]["time"]}.
- Exact raw bytes, HTTP response headers, local SHA-256 hashes, citations, and the PDBe
  public-data statement were preserved. The endpoints did not expose cryptographic
  sidecar checksums; HTTP ETags were preserved and local SHA-256 was computed.
- Every SIFTS segment was parsed into a deterministic Parquet table. Frozen canonical
  and external-UniProt rows were reconciled only by exact accession string.
- `pdb_entry_type.txt` supplies coarse diffraction/NMR/EM/other archive method classes.
  These PDB archive mappings are explicitly separate from predicted-structure resources.

## Limits and next gates

- SIFTS uses UniProt 2026.03 while the frozen external input is 2026_02. Exact accession
  agreement does not prove sequence-version identity.
- The reported span fraction is the outer mapped UniProt range, not observed residues;
  gaps and unobserved residues can occur. Fractions above one are retained as mismatch
  warnings rather than silently capped.
- Chain-level SIFTS mapping cannot prove exact construct boundaries, variants, mutations,
  expression tags, biological assembly, ligand state, resolution, or validation quality.
- The {report["canonical_construct_rows_not_exactly_reconciled"]:,} canonical construct
  records therefore remain unreconciled.
- Coordinate/validation retrieval should be limited to a task-specific, frozen subset
  after construct, ligand, method, quality, and leakage policies are approved.
"""


def verify_structure_metadata(
    raw_root: str | os.PathLike[str], output_root: str | os.PathLike[str], report_root: str | os.PathLike[str]
) -> dict[str, Any]:
    raw = _safe_directory(raw_root, context="raw structure metadata")
    output = _safe_directory(output_root, context="normalized structure metadata")
    reports = _safe_directory(report_root, context="structure metadata reports")
    acquisition = verify_acquisition(raw)
    manifest_path = output / "structure_metadata_manifest.json"
    manifest = _load_json(manifest_path, identified=True)
    _validate_runtime_bindings(manifest, acquisition, raw)
    inventory = manifest.get("output_inventory")
    if not isinstance(inventory, dict) or inventory.get("excluded_paths") != [manifest_path.name]:
        raise StructureMetadataError("Normalized inventory contract failed")
    entries = inventory.get("entries")
    if not isinstance(entries, list) or hashlib.sha256(
        canonical_json_bytes(entries)
    ).hexdigest() != inventory.get("entries_sha256"):
        raise StructureMetadataError("Normalized inventory digest failed")
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
            raise StructureMetadataError(f"Normalized artifact drift: {relative}")
        parquet = pq.ParquetFile(path)
        if parquet.metadata.num_rows != int(item.get("rows", -1)) or arrow_schema_sha256(
            parquet.schema_arrow
        ) != item.get("arrow_schema_sha256"):
            raise StructureMetadataError(f"Normalized Parquet drift: {relative}")
        if "model_label_admitted" in parquet.schema_arrow.names:
            for batch in parquet.iter_batches(columns=["model_label_admitted"], batch_size=8192):
                if any(batch.column(0).to_pylist()):
                    raise StructureMetadataError("Structure metadata admitted a model label")
    observed = {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file() and path != manifest_path
    }
    if observed != expected:
        raise StructureMetadataError("Normalized artifact membership drift")
    report = _load_json(reports / "structure_metadata_report.json", identified=True)
    if report.get("normalized_manifest_physical_sha256") != sha256_file(manifest_path):
        raise StructureMetadataError("Structure metadata report/manifest binding failed")
    methods = reports / "methods_and_limitations.md"
    if sha256_file(methods) != report.get("methods_sha256"):
        raise StructureMetadataError("Methods/report binding failed")
    for document in (acquisition, manifest, report):
        if (
            document.get("model_labels_admitted") != 0
            or document.get("substantive_model_training_performed") is not False
        ):
            raise StructureMetadataError("Zero-label/training contract failed")
    return {
        "status": "passed",
        "raw_manifest_sha256": sha256_file(raw / "acquisition_manifest.json"),
        "normalized_manifest_sha256": sha256_file(manifest_path),
        "report_sha256": sha256_file(reports / "structure_metadata_report.json"),
        "artifact_count": len(entries),
        "coordinate_files_downloaded": 0,
        "predicted_structure_files_downloaded": 0,
        "model_labels_admitted": 0,
        "substantive_model_training_performed": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-root", default="research/data/platform/raw/structure_metadata/sifts_2026_08_03"
    )
    parser.add_argument("--canonical-root", default="research/data/platform/canonical/full_chembl37")
    parser.add_argument(
        "--external-root", default="research/data/platform/interim/external_public_normalized"
    )
    parser.add_argument(
        "--output-root", default="research/data/platform/interim/structure_metadata/full_chembl37"
    )
    parser.add_argument("--report-root", default="research/reports/platform/structure_metadata/full_chembl37")
    parser.add_argument("--create-acquisition-manifest", action="store_true")
    parser.add_argument("--verify-existing", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.create_acquisition_manifest:
        result: Any = build_acquisition_manifest(args.raw_root)
    elif args.verify_existing:
        result = verify_structure_metadata(args.raw_root, args.output_root, args.report_root)
    else:
        result = build_structure_metadata(
            args.raw_root, args.canonical_root, args.external_root, args.output_root, args.report_root
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
