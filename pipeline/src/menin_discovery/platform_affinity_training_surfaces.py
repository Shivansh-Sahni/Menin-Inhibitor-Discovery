"""Build a mirror-aware protein--ligand affinity and potency training release.

The release is deliberately endpoint separated.  Kd, Ki, and IC50 are primary
protein--molecule tasks; EC50 is retained as an explicitly auxiliary potency
task and is never relabelled as affinity.  ChEMBL 37 is the preferred source
when the same documented measurement is present in BindingDB.  Censored
relations and intervals are retained as bounds rather than converted to point
labels.

This module builds data surfaces only.  It does not generate model features,
fit a model, execute an HPC job, or establish prospective performance.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import shutil
import sqlite3
import tempfile
import zipfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from .chemistry import STANDARDIZATION_VERSION, standardize_smiles

SCHEMA_VERSION = "platform-affinity-training-surfaces/1.0"
RELEASE_ID = "chembl37_bindingdb202608_affinity_v1_0"
SPLIT_POLICY_VERSION = "affinity-scaffold-target-cold/1.0"
SPLIT_SEED = "20260809-affinity-v1"
MANIFEST_NAME = "affinity_training_manifest.json"
CONTRACT_NAME = "training_surface_contract.json"
REPORT_NAME = "AFFINITY_TRAINING_SURFACES.md"
ENDPOINTS = ("Kd", "Ki", "IC50", "EC50")
PRIMARY_ENDPOINTS = frozenset(("Kd", "Ki", "IC50"))
SUPPORTED_RELATIONS = frozenset(("=", "<", "<=", ">", ">=", "interval"))
RIGHTS_QUARANTINE_SOURCES = frozenset(("Taylor Research Group, UCSD",))
OUTPUT_PART_ROWS = 200_000


class AffinityTrainingSurfaceError(RuntimeError):
    """Raised when the affinity release cannot be built or proven."""


_LIGAND_SCHEMA = pa.schema(
    [
        pa.field("structure_id", pa.large_string(), nullable=False),
        pa.field("standardized_smiles", pa.large_string(), nullable=False),
        pa.field("standard_inchi_key", pa.large_string(), nullable=False),
        pa.field("scaffold_group_id", pa.large_string(), nullable=False),
        pa.field("scaffold_derivation_status", pa.large_string(), nullable=False),
        pa.field("ligand_cold_split", pa.large_string(), nullable=False),
        pa.field("structure_standardization_version", pa.large_string(), nullable=False),
        pa.field("rdkit_version", pa.large_string(), nullable=False),
        pa.field("formal_charge", pa.int64()),
        pa.field("source_fragment_count", pa.int64()),
    ]
)

_TARGET_SCHEMA = pa.schema(
    [
        pa.field("target_id", pa.large_string(), nullable=False),
        pa.field("target_leakage_group_id", pa.large_string(), nullable=False),
        pa.field("primary_accession", pa.large_string()),
        pa.field("accessions_json", pa.large_string(), nullable=False),
        pa.field("sequence", pa.large_string()),
        pa.field("sequence_sha256", pa.large_string()),
        pa.field("sequence_length", pa.int64()),
        pa.field("sequence_source", pa.large_string(), nullable=False),
        pa.field("target_name", pa.large_string()),
        pa.field("organism", pa.large_string()),
        pa.field("target_aliases_json", pa.large_string(), nullable=False),
        pa.field("organism_aliases_json", pa.large_string(), nullable=False),
        pa.field("source_datasets_json", pa.large_string(), nullable=False),
        pa.field("target_cold_split", pa.large_string(), nullable=False),
        pa.field("sequence_model_eligible", pa.bool_(), nullable=False),
        pa.field("target_group_sequence_conflict", pa.bool_(), nullable=False),
    ]
)

_TARGET_GROUP_SCHEMA = pa.schema(
    [
        pa.field("target_leakage_group_id", pa.large_string(), nullable=False),
        pa.field("target_cold_split", pa.large_string(), nullable=False),
        pa.field("target_record_count", pa.int64(), nullable=False),
        pa.field("primary_accession", pa.large_string()),
        pa.field("accessions_json", pa.large_string(), nullable=False),
        pa.field("sequence_count", pa.int64(), nullable=False),
        pa.field("sequence_sha256s_json", pa.large_string(), nullable=False),
        pa.field("resolved_sequence", pa.large_string()),
        pa.field("resolved_sequence_sha256", pa.large_string()),
        pa.field("sequence_conflict", pa.bool_(), nullable=False),
        pa.field("target_names_json", pa.large_string(), nullable=False),
        pa.field("organisms_json", pa.large_string(), nullable=False),
    ]
)

_RAW_OBSERVATION_SCHEMA = pa.schema(
    [
        pa.field("observation_id", pa.large_string(), nullable=False),
        pa.field("source_dataset", pa.large_string(), nullable=False),
        pa.field("source_snapshot", pa.large_string(), nullable=False),
        pa.field("source_category", pa.large_string(), nullable=False),
        pa.field("source_observation_id", pa.large_string(), nullable=False),
        pa.field("source_record_id", pa.large_string(), nullable=False),
        pa.field("source_row_number", pa.int64()),
        pa.field("source_ligand_id", pa.large_string()),
        pa.field("structure_id", pa.large_string(), nullable=False),
        pa.field("scaffold_group_id", pa.large_string(), nullable=False),
        pa.field("target_id", pa.large_string(), nullable=False),
        pa.field("endpoint", pa.large_string(), nullable=False),
        pa.field("endpoint_role", pa.large_string(), nullable=False),
        pa.field("label_relation", pa.large_string(), nullable=False),
        pa.field("label_value_nM", pa.float64()),
        pa.field("label_lower_bound_nM", pa.float64()),
        pa.field("label_upper_bound_nM", pa.float64()),
        pa.field("label_unit", pa.large_string(), nullable=False),
        pa.field("label_censoring", pa.large_string(), nullable=False),
        pa.field("native_label", pa.large_string()),
        pa.field("assay_id", pa.large_string()),
        pa.field("assay_family", pa.large_string()),
        pa.field("assay_description", pa.large_string()),
        pa.field("assay_matrix", pa.large_string()),
        pa.field("assay_route", pa.large_string()),
        pa.field("assay_ph", pa.large_string()),
        pa.field("assay_temperature", pa.large_string()),
        pa.field("document_doi", pa.large_string()),
        pa.field("document_pubmed_id", pa.large_string()),
        pa.field("document_patent_id", pa.large_string()),
        pa.field("document_year", pa.int64()),
        pa.field("document_identity", pa.large_string()),
        pa.field("rights_status", pa.large_string(), nullable=False),
        pa.field("mirror_status", pa.large_string(), nullable=False),
    ]
)

_OBSERVATION_SCHEMA = pa.schema(
    [
        *_RAW_OBSERVATION_SCHEMA,
        pa.field("target_leakage_group_id", pa.large_string(), nullable=False),
        pa.field("ligand_cold_split", pa.large_string(), nullable=False),
        pa.field("target_cold_split", pa.large_string(), nullable=False),
        pa.field("double_cold_split", pa.large_string()),
        pa.field("double_cold_eligible", pa.bool_(), nullable=False),
        pa.field("sequence_model_eligible", pa.bool_(), nullable=False),
        pa.field("ligand_target_pair_id", pa.large_string(), nullable=False),
        pa.field("target_endpoint_task_id", pa.large_string(), nullable=False),
        pa.field("measurement_lineage_group_id", pa.large_string(), nullable=False),
        pa.field("scientific_training_eligible", pa.bool_(), nullable=False),
        pa.field("eligibility_basis", pa.large_string(), nullable=False),
    ]
)

_PAIR_SCHEMA = pa.schema(
    [
        pa.field("ligand_target_pair_id", pa.large_string(), nullable=False),
        pa.field("structure_id", pa.large_string(), nullable=False),
        pa.field("scaffold_group_id", pa.large_string(), nullable=False),
        pa.field("target_leakage_group_id", pa.large_string(), nullable=False),
        pa.field("ligand_cold_split", pa.large_string(), nullable=False),
        pa.field("target_cold_split", pa.large_string(), nullable=False),
        pa.field("double_cold_split", pa.large_string()),
        pa.field("double_cold_eligible", pa.bool_(), nullable=False),
    ]
)

_TASK_SCHEMA = pa.schema(
    [
        pa.field("target_endpoint_task_id", pa.large_string(), nullable=False),
        pa.field("endpoint", pa.large_string(), nullable=False),
        pa.field("endpoint_role", pa.large_string(), nullable=False),
        pa.field("target_leakage_group_id", pa.large_string(), nullable=False),
        pa.field("target_cold_split", pa.large_string(), nullable=False),
        pa.field("observation_count", pa.int64(), nullable=False),
        pa.field("unique_structure_count", pa.int64(), nullable=False),
        pa.field("exact_count", pa.int64(), nullable=False),
        pa.field("left_censored_count", pa.int64(), nullable=False),
        pa.field("right_censored_count", pa.int64(), nullable=False),
        pa.field("interval_censored_count", pa.int64(), nullable=False),
        pa.field("sequence_model_eligible_count", pa.int64(), nullable=False),
        pa.field("double_cold_eligible_count", pa.int64(), nullable=False),
        pa.field("source_counts_json", pa.large_string(), nullable=False),
        pa.field("at_least_100_observations", pa.bool_(), nullable=False),
        pa.field("at_least_1000_observations", pa.bool_(), nullable=False),
        pa.field("task_semantics", pa.large_string(), nullable=False),
    ]
)

_EXCLUSION_SCHEMA = pa.schema(
    [
        pa.field("source_dataset", pa.large_string(), nullable=False),
        pa.field("endpoint", pa.large_string(), nullable=False),
        pa.field("exclusion_reason", pa.large_string(), nullable=False),
        pa.field("record_count", pa.int64(), nullable=False),
        pa.field("exclusion_scope", pa.large_string(), nullable=False),
    ]
)


@dataclass
class _TargetRecord:
    target_id: str
    accessions: tuple[str, ...]
    sequence: str
    sequence_sha256: str
    names: set[str]
    organisms: set[str]
    sources: set[str]

    @property
    def tokens(self) -> tuple[str, ...]:
        tokens = [f"A:{value}" for value in self.accessions]
        if self.sequence_sha256:
            tokens.append(f"S:{self.sequence_sha256}")
        return tuple(sorted(tokens))


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def add(self, item: str) -> None:
        self.parent.setdefault(item, item)

    def find(self, item: str) -> str:
        self.add(item)
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        keep, merge = sorted((left_root, right_root))
        self.parent[merge] = keep


def _clean(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(*parts: object, prefix: str, length: int = 24) -> str:
    payload = "\0".join(_clean(part) for part in parts).encode()
    return f"{prefix}-{hashlib.sha256(payload).hexdigest()[:length].upper()}"


def _manifest_hash(payload: Mapping[str, Any]) -> str:
    unsigned = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    raw = json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def _schema_hash(schema: pa.Schema) -> str:
    return hashlib.sha256(schema.remove_metadata().serialize().to_pybytes()).hexdigest()


def _file_schema_hash(path: Path) -> str:
    return _schema_hash(pq.read_schema(path))


def _json(values: Iterable[object]) -> str:
    return json.dumps(sorted({_clean(value) for value in values if _clean(value)}), separators=(",", ":"))


def _normalize_accessions(*values: object) -> tuple[str, ...]:
    accessions: set[str] = set()
    for value in values:
        text = _clean(value).upper()
        if not text:
            continue
        for token in re.split(r"[\s,;|]+", text):
            token = token.strip()
            if token and re.fullmatch(r"[A-Z0-9_.-]{3,32}", token):
                accessions.add(token)
    return tuple(sorted(accessions))


def _normalize_sequence(value: object) -> str:
    sequence = re.sub(r"\s+", "", _clean(value)).upper().rstrip("*")
    if not sequence or re.fullmatch(r"[A-Z]+", sequence) is None:
        return ""
    return sequence


def _split(group_id: str, *, dimension: str) -> str:
    value = int(
        hashlib.sha256(f"{SPLIT_SEED}\0{dimension}\0{group_id}".encode()).hexdigest()[:16],
        16,
    )
    bucket = value % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "validation"
    return "test"


def _document_identity(doi: object, pmid: object, patent: object) -> str:
    doi_text = _clean(doi).lower()
    doi_text = re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", "", doi_text)
    if doi_text:
        return f"doi:{doi_text}"
    pmid_text = re.sub(r"\D", "", _clean(pmid))
    if pmid_text:
        return f"pmid:{pmid_text}"
    patent_text = re.sub(r"\s+", "", _clean(patent).upper())
    if patent_text:
        return f"patent:{patent_text}"
    return ""


def _year(value: object) -> int | None:
    match = re.search(r"(?:19|20)\d{2}", _clean(value))
    return int(match.group(0)) if match else None


def _censoring(relation: str) -> str:
    if relation == "=":
        return "exact"
    if relation in {"<", "<="}:
        return "left_censored"
    if relation in {">", ">="}:
        return "right_censored"
    if relation == "interval":
        return "interval_censored"
    raise AffinityTrainingSurfaceError(f"unsupported relation: {relation}")


_BDB_VALUE_PATTERN = re.compile(
    r"^\s*(?P<relation><=|>=|<|>|=)?\s*(?P<value>(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?)\s*$"
)


def _parse_bindingdb_value(value: object) -> tuple[str, float, float | None, float | None] | None:
    text = _clean(value)
    match = _BDB_VALUE_PATTERN.fullmatch(text)
    if match is None:
        return None
    numeric = float(match.group("value"))
    if not math.isfinite(numeric) or numeric <= 0:
        return None
    relation = match.group("relation") or "="
    lower = numeric if relation in {"=", ">", ">="} else None
    upper = numeric if relation in {"=", "<", "<="} else None
    return relation, numeric, lower, upper


def _measurement_key(
    *,
    structure_id: str,
    target_token: str,
    endpoint: str,
    relation: str,
    value: float | None,
    lower: float | None,
    upper: float | None,
    document_identity: str,
) -> bytes:
    fields = (
        structure_id,
        target_token,
        endpoint,
        relation,
        "" if value is None else format(value, ".12g"),
        "" if lower is None else format(lower, ".12g"),
        "" if upper is None else format(upper, ".12g"),
        document_identity,
    )
    return hashlib.sha256("\0".join(fields).encode()).digest()


def _rights_status(source_dataset: str, source_category: str) -> str:
    if source_dataset == "ChEMBL_37":
        return "chembl_cc_by_sa_3_0_source_terms_preserved"
    if source_category == "Curated from the literature by BindingDB":
        return "bindingdb_curated_cc_by_4_0"
    return "bindingdb_archive_source_specific_terms_preserved_distribution_review_required"


class _PartitionWriter:
    def __init__(self, root: Path, schema: pa.Schema, *, part_rows: int = OUTPUT_PART_ROWS) -> None:
        self.root = root
        self.schema = schema
        self.part_rows = part_rows
        self.buffer: list[dict[str, Any]] = []
        self.part_number = 0
        self.root.mkdir(parents=True, exist_ok=True)

    def append(self, row: dict[str, Any]) -> None:
        self.buffer.append(row)
        if len(self.buffer) >= self.part_rows:
            self.flush()

    def extend(self, rows: Iterable[dict[str, Any]]) -> None:
        for row in rows:
            self.append(row)

    def flush(self) -> None:
        if not self.buffer:
            return
        path = self.root / f"part-{self.part_number:05d}.parquet"
        table = pa.Table.from_pylist(self.buffer, schema=self.schema)
        pq.write_table(table, path, compression="zstd", use_dictionary=True, row_group_size=50_000)
        self.buffer.clear()
        self.part_number += 1

    def close(self) -> None:
        self.flush()


def _write_rows(rows: Iterable[dict[str, Any]], path: Path, schema: pa.Schema) -> None:
    materialized = list(rows)
    table = pa.Table.from_pylist(materialized, schema=schema)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd", use_dictionary=True)


def _bind_input(path: Path, **extra: Any) -> dict[str, Any]:
    binding = {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }
    binding.update(extra)
    return binding


def _bind_artifact(path: Path, release_root: Path) -> dict[str, Any]:
    binding: dict[str, Any] = {
        "path": path.relative_to(release_root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }
    if path.suffix == ".parquet":
        binding["rows"] = pq.read_metadata(path).num_rows
        binding["arrow_schema_sha256"] = _file_schema_hash(path)
    return binding


def _register_target(
    records: dict[str, _TargetRecord],
    union: _UnionFind,
    *,
    accessions: Sequence[str],
    sequence: str,
    target_name: object,
    organism: object,
    source_dataset: str,
) -> _TargetRecord | None:
    normalized_accessions = tuple(sorted({_clean(value).upper() for value in accessions if _clean(value)}))
    normalized_sequence = _normalize_sequence(sequence)
    sequence_sha256 = hashlib.sha256(normalized_sequence.encode()).hexdigest() if normalized_sequence else ""
    tokens = [f"A:{value}" for value in normalized_accessions]
    if sequence_sha256:
        tokens.append(f"S:{sequence_sha256}")
    if not tokens:
        return None
    target_id = _digest(
        json.dumps(normalized_accessions, separators=(",", ":")),
        sequence_sha256,
        prefix="TGT",
    )
    record = records.get(target_id)
    if record is None:
        record = _TargetRecord(
            target_id=target_id,
            accessions=normalized_accessions,
            sequence=normalized_sequence,
            sequence_sha256=sequence_sha256,
            names=set(),
            organisms=set(),
            sources=set(),
        )
        records[target_id] = record
    name = _clean(target_name)
    species = _clean(organism)
    if name:
        record.names.add(name)
    if species:
        record.organisms.add(species)
    record.sources.add(source_dataset)
    for token in tokens:
        union.add(token)
    for token in tokens[1:]:
        union.union(tokens[0], token)
    return record


def _resolve_target_groups(
    records: Mapping[str, _TargetRecord],
    union: _UnionFind,
    used_target_ids: set[str],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    root_members: dict[str, set[str]] = defaultdict(set)
    for token in union.parent:
        root_members[union.find(token)].add(token)
    root_group: dict[str, str] = {}
    for root, component_members in root_members.items():
        root_group[root] = _digest(*sorted(component_members), prefix="TGRP")

    group_records: dict[str, list[_TargetRecord]] = defaultdict(list)
    target_group: dict[str, str] = {}
    for target_id in sorted(used_target_ids):
        record = records[target_id]
        roots = {union.find(token) for token in record.tokens}
        if len(roots) != 1:
            raise AffinityTrainingSurfaceError(f"target tokens did not converge: {target_id}")
        group_id = root_group[next(iter(roots))]
        target_group[target_id] = group_id
        group_records[group_id].append(record)

    group_metadata: dict[str, dict[str, Any]] = {}
    group_rows: list[dict[str, Any]] = []
    for group_id, member_records in sorted(group_records.items()):
        accessions = sorted({value for record in member_records for value in record.accessions})
        sequences = {record.sequence_sha256: record.sequence for record in member_records if record.sequence}
        names = sorted({value for record in member_records for value in record.names})
        organisms = sorted({value for record in member_records for value in record.organisms})
        resolved_sequence = next(iter(sequences.values())) if len(sequences) == 1 else ""
        resolved_sha = next(iter(sequences)) if len(sequences) == 1 else ""
        metadata = {
            "group_id": group_id,
            "split": _split(group_id, dimension="target_component"),
            "resolved_sequence": resolved_sequence,
            "resolved_sequence_sha256": resolved_sha,
            "sequence_conflict": len(sequences) > 1,
        }
        group_metadata[group_id] = metadata
        group_rows.append(
            {
                "target_leakage_group_id": group_id,
                "target_cold_split": metadata["split"],
                "target_record_count": len(member_records),
                "primary_accession": accessions[0] if accessions else None,
                "accessions_json": json.dumps(accessions, separators=(",", ":")),
                "sequence_count": len(sequences),
                "sequence_sha256s_json": json.dumps(sorted(sequences), separators=(",", ":")),
                "resolved_sequence": resolved_sequence or None,
                "resolved_sequence_sha256": resolved_sha or None,
                "sequence_conflict": len(sequences) > 1,
                "target_names_json": json.dumps(names, separators=(",", ":")),
                "organisms_json": json.dumps(organisms, separators=(",", ":")),
            }
        )

    target_rows: list[dict[str, Any]] = []
    for target_id in sorted(used_target_ids):
        record = records[target_id]
        group_id = target_group[target_id]
        metadata = group_metadata[group_id]
        sequence = record.sequence
        sequence_source = "reported_by_source" if sequence else "missing"
        if not sequence and metadata["resolved_sequence"]:
            sequence = str(metadata["resolved_sequence"])
            sequence_source = "resolved_from_unambiguous_accession_sequence_component"
        sequence_sha = hashlib.sha256(sequence.encode()).hexdigest() if sequence else ""
        target_rows.append(
            {
                "target_id": target_id,
                "target_leakage_group_id": group_id,
                "primary_accession": record.accessions[0] if record.accessions else None,
                "accessions_json": json.dumps(record.accessions, separators=(",", ":")),
                "sequence": sequence or None,
                "sequence_sha256": sequence_sha or None,
                "sequence_length": len(sequence) if sequence else None,
                "sequence_source": sequence_source,
                "target_name": sorted(record.names)[0] if record.names else None,
                "organism": sorted(record.organisms)[0] if record.organisms else None,
                "target_aliases_json": _json(record.names),
                "organism_aliases_json": _json(record.organisms),
                "source_datasets_json": _json(record.sources),
                "target_cold_split": metadata["split"],
                "sequence_model_eligible": bool(sequence),
                "target_group_sequence_conflict": bool(metadata["sequence_conflict"]),
            }
        )
    target_lookup = {row["target_id"]: row for row in target_rows}
    return target_lookup, target_rows, group_rows


def _initialize_work_database(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA temp_store=FILE")
    connection.executescript(
        """
        CREATE TABLE structure_cache (
            raw_smiles TEXT PRIMARY KEY,
            structure_id TEXT NOT NULL,
            standardized_smiles TEXT NOT NULL,
            standard_inchi_key TEXT NOT NULL,
            scaffold_group_id TEXT NOT NULL,
            scaffold_status TEXT NOT NULL,
            ligand_cold_split TEXT NOT NULL,
            standardization_version TEXT NOT NULL,
            rdkit_version TEXT NOT NULL,
            formal_charge INTEGER,
            fragment_count INTEGER
        ) WITHOUT ROWID;
        CREATE TABLE ligands (
            structure_id TEXT PRIMARY KEY,
            standardized_smiles TEXT NOT NULL,
            standard_inchi_key TEXT NOT NULL,
            scaffold_group_id TEXT NOT NULL,
            scaffold_status TEXT NOT NULL,
            ligand_cold_split TEXT NOT NULL,
            standardization_version TEXT NOT NULL,
            rdkit_version TEXT NOT NULL,
            formal_charge INTEGER,
            fragment_count INTEGER
        ) WITHOUT ROWID;
        CREATE TABLE strong_mirror_keys (
            mirror_key BLOB PRIMARY KEY,
            source_dataset TEXT NOT NULL
        ) WITHOUT ROWID;
        CREATE TABLE ligand_target_pairs (
            pair_id TEXT PRIMARY KEY,
            structure_id TEXT NOT NULL,
            scaffold_group_id TEXT NOT NULL,
            target_group_id TEXT NOT NULL,
            ligand_split TEXT NOT NULL,
            target_split TEXT NOT NULL,
            double_split TEXT
        ) WITHOUT ROWID;
        CREATE TABLE endpoint_pairs (
            endpoint_pair_id TEXT PRIMARY KEY,
            endpoint TEXT NOT NULL,
            structure_id TEXT NOT NULL,
            target_group_id TEXT NOT NULL
        ) WITHOUT ROWID;
        """
    )
    return connection


def _resolve_structure(connection: sqlite3.Connection, raw_smiles: object) -> dict[str, Any] | None:
    text = _clean(raw_smiles)
    if not text:
        return None
    cached = connection.execute(
        """SELECT structure_id, standardized_smiles, standard_inchi_key,
                  scaffold_group_id, scaffold_status, ligand_cold_split, standardization_version,
                  rdkit_version, formal_charge, fragment_count
           FROM structure_cache WHERE raw_smiles = ?""",
        (text,),
    ).fetchone()
    if cached is not None:
        return {
            "structure_id": cached[0],
            "standardized_smiles": cached[1],
            "standard_inchi_key": cached[2],
            "scaffold_group_id": cached[3],
            "scaffold_derivation_status": cached[4],
            "ligand_cold_split": cached[5],
            "structure_standardization_version": cached[6],
            "rdkit_version": cached[7],
            "formal_charge": cached[8],
            "source_fragment_count": cached[9],
        }
    standardized = standardize_smiles(text, require_rdkit=True)
    if (
        not standardized.structure_valid
        or not standardized.structure_id
        or not standardized.standardized_smiles
        or not standardized.standard_inchi_key
    ):
        return None
    molecule = Chem.MolFromSmiles(standardized.standardized_smiles)
    if molecule is None:
        return None
    try:
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=molecule, includeChirality=False)
        scaffold_key = scaffold or standardized.standardized_smiles
        scaffold_status = "bemis_murcko" if scaffold else "acyclic_exact_structure_fallback"
    except (RuntimeError, ValueError):
        scaffold_key = standardized.standardized_smiles
        scaffold_status = "murcko_failure_exact_structure_fallback"
    scaffold_group_id = _digest(scaffold_key, prefix="SCF")
    ligand_split = _split(scaffold_group_id, dimension="ligand_scaffold")
    values = (
        standardized.structure_id,
        standardized.standardized_smiles,
        standardized.standard_inchi_key,
        scaffold_group_id,
        scaffold_status,
        ligand_split,
        standardized.structure_standardization_version,
        standardized.rdkit_version,
        standardized.formal_charge,
        standardized.fragment_count,
    )
    connection.execute(
        "INSERT INTO structure_cache VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (text, *values),
    )
    return {
        "structure_id": values[0],
        "standardized_smiles": values[1],
        "standard_inchi_key": values[2],
        "scaffold_group_id": values[3],
        "scaffold_derivation_status": values[4],
        "ligand_cold_split": values[5],
        "structure_standardization_version": values[6],
        "rdkit_version": values[7],
        "formal_charge": values[8],
        "source_fragment_count": values[9],
    }


def _admit_ligand(connection: sqlite3.Connection, ligand: Mapping[str, Any]) -> None:
    connection.execute(
        """INSERT OR IGNORE INTO ligands VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            ligand["structure_id"],
            ligand["standardized_smiles"],
            ligand["standard_inchi_key"],
            ligand["scaffold_group_id"],
            ligand["scaffold_derivation_status"],
            ligand["ligand_cold_split"],
            ligand["structure_standardization_version"],
            ligand["rdkit_version"],
            ligand["formal_charge"],
            ligand["source_fragment_count"],
        ),
    )


def _task_keys(task_datasets: Mapping[str, Any]) -> dict[str, list[str]]:
    selected: dict[str, list[str]] = {endpoint: [] for endpoint in ENDPOINTS}
    for key in task_datasets:
        for endpoint in ENDPOINTS:
            token = endpoint.casefold()
            if key.startswith(f"default::default__binding__{token}__binding__nm__continuous_") and key.rsplit(
                "_", 1
            )[-1] in {"exact", "censored"}:
                selected[endpoint].append(key)
    for endpoint, keys in selected.items():
        if len(keys) != 2:
            raise AffinityTrainingSurfaceError(
                f"expected exact and censored ChEMBL tasks for {endpoint}; found {keys}"
            )
        keys.sort()
    return selected


def _verified_chembl_parts(
    canonical_root: Path,
    task_datasets: Mapping[str, Any],
    keys: Mapping[str, Sequence[str]],
) -> tuple[list[tuple[str, Path]], list[dict[str, Any]]]:
    parts: list[tuple[str, Path]] = []
    bindings: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for endpoint in ENDPOINTS:
        for key in keys[endpoint]:
            dataset = task_datasets[key]
            declared_parts = dataset.get("parts")
            if not isinstance(declared_parts, list) or int(dataset.get("part_count", -1)) != len(
                declared_parts
            ):
                raise AffinityTrainingSurfaceError(f"malformed ChEMBL task dataset: {key}")
            rows = 0
            for part in declared_parts:
                path = (canonical_root / _clean(part.get("path"))).resolve()
                try:
                    path.relative_to(canonical_root.resolve())
                except ValueError as error:
                    raise AffinityTrainingSurfaceError(f"ChEMBL part escapes root: {path}") from error
                if not path.is_file() or path in seen:
                    raise AffinityTrainingSurfaceError(f"missing or reused ChEMBL part: {path}")
                seen.add(path)
                metadata = pq.read_metadata(path)
                actual_rows = metadata.num_rows
                if actual_rows != int(part.get("rows", -1)) or _sha256(path) != part.get("sha256"):
                    raise AffinityTrainingSurfaceError(f"ChEMBL part binding failed: {path}")
                rows += actual_rows
                parts.append((endpoint, path))
                bindings.append(
                    _bind_input(
                        path,
                        rows=actual_rows,
                        arrow_schema_sha256=_file_schema_hash(path),
                        source_declared_arrow_schema_sha256=part.get("arrow_schema_sha256"),
                        source_task_key=key,
                    )
                )
            if rows != int(dataset.get("row_count", -1)):
                raise AffinityTrainingSurfaceError(f"ChEMBL dataset row count failed: {key}")
    return parts, bindings


_CHEMBL_COLUMNS = [
    "observation_id",
    "source_record_id",
    "source_id",
    "molecule_id",
    "structure_id",
    "standardized_smiles",
    "canonical_target_id",
    "protein_id",
    "sequence",
    "target_name",
    "species",
    "endpoint",
    "label_relation",
    "label_value",
    "label_lower_bound",
    "label_upper_bound",
    "label_unit",
    "label_text",
    "label_kind",
    "assay_id",
    "assay_family",
    "description",
    "matrix",
    "route",
    "document_doi",
    "document_pubmed_id",
    "document_patent_id",
    "document_year",
    "default_task_eligible",
    "inclusion_status",
]


def _float_or_none(value: object) -> float | None:
    if value is None:
        return None
    numeric = float(_clean(value))
    return numeric if math.isfinite(numeric) else None


def _chembl_observations(
    *,
    parts: Sequence[tuple[str, Path]],
    connection: sqlite3.Connection,
    writers: Mapping[str, _PartitionWriter],
    targets: dict[str, _TargetRecord],
    union: _UnionFind,
    used_target_ids: set[str],
    exclusions: Counter[tuple[str, str, str]],
    physical: Counter[str],
) -> None:
    pending = 0
    for declared_endpoint, path in parts:
        parquet = pq.ParquetFile(path)
        missing = set(_CHEMBL_COLUMNS) - set(parquet.schema_arrow.names)
        if missing:
            raise AffinityTrainingSurfaceError(f"ChEMBL task columns missing in {path}: {sorted(missing)}")
        for batch in parquet.iter_batches(batch_size=20_000, columns=_CHEMBL_COLUMNS):
            for row in batch.to_pylist():
                physical["chembl_task_rows_considered"] += 1
                endpoint = _clean(row["endpoint"])
                if endpoint != declared_endpoint:
                    raise AffinityTrainingSurfaceError(
                        f"ChEMBL endpoint partition mismatch: {declared_endpoint} != {endpoint}"
                    )
                if not row["default_task_eligible"] or _clean(row["inclusion_status"]) != "included":
                    exclusions[("ChEMBL_37", endpoint, "upstream_task_not_eligible")] += 1
                    continue
                relation = _clean(row["label_relation"])
                value = _float_or_none(row["label_value"])
                lower = _float_or_none(row["label_lower_bound"])
                upper = _float_or_none(row["label_upper_bound"])
                if relation not in SUPPORTED_RELATIONS:
                    exclusions[("ChEMBL_37", endpoint, "unsupported_relation")] += 1
                    continue
                if relation == "=" and (value is None or value <= 0 or lower != value or upper != value):
                    exclusions[("ChEMBL_37", endpoint, "invalid_exact_bounds")] += 1
                    continue
                if relation == "interval" and (
                    lower is None or upper is None or lower <= 0 or lower >= upper
                ):
                    exclusions[("ChEMBL_37", endpoint, "invalid_interval_bounds")] += 1
                    continue
                ligand = _resolve_structure(connection, row["standardized_smiles"])
                if ligand is None:
                    exclusions[("ChEMBL_37", endpoint, "current_policy_structure_failure")] += 1
                    continue
                sequence = _normalize_sequence(row["sequence"])
                target = _register_target(
                    targets,
                    union,
                    accessions=_normalize_accessions(row["canonical_target_id"]),
                    sequence=sequence,
                    target_name=row["target_name"],
                    organism=row["species"],
                    source_dataset="ChEMBL_37",
                )
                if target is None:
                    exclusions[("ChEMBL_37", endpoint, "missing_accession_and_sequence")] += 1
                    continue
                document_id = _document_identity(
                    row["document_doi"], row["document_pubmed_id"], row["document_patent_id"]
                )
                if document_id:
                    for token in target.tokens:
                        key = _measurement_key(
                            structure_id=ligand["structure_id"],
                            target_token=token,
                            endpoint=endpoint,
                            relation=relation,
                            value=value,
                            lower=lower,
                            upper=upper,
                            document_identity=document_id,
                        )
                        connection.execute(
                            "INSERT OR IGNORE INTO strong_mirror_keys VALUES (?, ?)",
                            (key, "ChEMBL_37"),
                        )
                _admit_ligand(connection, ligand)
                used_target_ids.add(target.target_id)
                source_record_id = _clean(row["source_record_id"])
                source_observation_id = _clean(row["observation_id"])
                writers[endpoint].append(
                    {
                        "observation_id": _digest(
                            "ChEMBL_37", source_observation_id, endpoint, prefix="AFFOBS"
                        ),
                        "source_dataset": "ChEMBL_37",
                        "source_snapshot": "ChEMBL_37",
                        "source_category": "ChEMBL",
                        "source_observation_id": source_observation_id,
                        "source_record_id": source_record_id,
                        "source_row_number": None,
                        "source_ligand_id": _clean(row["molecule_id"]) or None,
                        "structure_id": ligand["structure_id"],
                        "scaffold_group_id": ligand["scaffold_group_id"],
                        "target_id": target.target_id,
                        "endpoint": endpoint,
                        "endpoint_role": "primary" if endpoint in PRIMARY_ENDPOINTS else "auxiliary_potency",
                        "label_relation": relation,
                        "label_value_nM": value,
                        "label_lower_bound_nM": lower,
                        "label_upper_bound_nM": upper,
                        "label_unit": "nM",
                        "label_censoring": _censoring(relation),
                        "native_label": _clean(row["label_text"]) or None,
                        "assay_id": _clean(row["assay_id"]) or None,
                        "assay_family": _clean(row["assay_family"]) or None,
                        "assay_description": _clean(row["description"]) or None,
                        "assay_matrix": _clean(row["matrix"]) or None,
                        "assay_route": _clean(row["route"]) or None,
                        "assay_ph": None,
                        "assay_temperature": None,
                        "document_doi": _clean(row["document_doi"]) or None,
                        "document_pubmed_id": _clean(row["document_pubmed_id"]) or None,
                        "document_patent_id": _clean(row["document_patent_id"]) or None,
                        "document_year": _year(row["document_year"]),
                        "document_identity": document_id or None,
                        "rights_status": _rights_status("ChEMBL_37", "ChEMBL"),
                        "mirror_status": "preferred_canonical_source",
                    }
                )
                pending += 1
                if pending % 20_000 == 0:
                    connection.commit()
    connection.commit()


_BDB_REQUIRED_COLUMNS = {
    "reactant_set": "BindingDB Reactant_set_id",
    "smiles": "Ligand SMILES",
    "monomer_id": "BindingDB MonomerID",
    "target_name": "Target Name",
    "organism": "Target Source Organism According to Curator or DataSource",
    "Ki": "Ki (nM)",
    "IC50": "IC50 (nM)",
    "Kd": "Kd (nM)",
    "EC50": "EC50 (nM)",
    "ph": "pH",
    "temperature": "Temp (C)",
    "source": "Curation/DataSource",
    "doi": "Article DOI",
    "pmid": "PMID",
    "pubchem_aid": "PubChem AID",
    "patent": "Patent Number",
    "publication_date": "Date of publication",
    "pubchem_cid": "PubChem CID",
    "chembl_ligand_id": "ChEMBL ID of Ligand",
    "chain_count": "Number of Protein Chains in Target (>1 implies a multichain complex)",
    "sequence": "BindingDB Target Chain Sequence 1",
    "swissprot": "UniProt (SwissProt) Primary ID of Target Chain 1",
    "trembl": "UniProt (TrEMBL) Primary ID of Target Chain 1",
}


def _bindingdb_member(archive: zipfile.ZipFile) -> str:
    members = [name for name in archive.namelist() if name.casefold().endswith(".tsv")]
    if len(members) != 1:
        raise AffinityTrainingSurfaceError(f"expected exactly one BindingDB TSV member, found {members}")
    return members[0]


def _bindingdb_observations(
    *,
    archive_path: Path,
    connection: sqlite3.Connection,
    writers: Mapping[str, _PartitionWriter],
    targets: dict[str, _TargetRecord],
    union: _UnionFind,
    used_target_ids: set[str],
    exclusions: Counter[tuple[str, str, str]],
    physical: Counter[str],
    source_counts: Counter[str],
) -> None:
    csv.field_size_limit(100_000_000)
    accepted_since_commit = 0
    with zipfile.ZipFile(archive_path) as archive:
        member = _bindingdb_member(archive)
        with archive.open(member) as raw_handle:
            text_handle = io.TextIOWrapper(raw_handle, encoding="utf-8-sig", errors="replace", newline="")
            reader = csv.reader(text_handle, delimiter="\t")
            try:
                header = next(reader)
            except StopIteration as error:
                raise AffinityTrainingSurfaceError("BindingDB archive has no header") from error
            duplicate_headers = [name for name, count in Counter(header).items() if count > 1]
            if duplicate_headers:
                raise AffinityTrainingSurfaceError(
                    f"BindingDB header has duplicate names: {duplicate_headers[:5]}"
                )
            index = {name: position for position, name in enumerate(header)}
            missing = set(_BDB_REQUIRED_COLUMNS.values()) - set(index)
            if missing:
                raise AffinityTrainingSurfaceError(f"BindingDB columns missing: {sorted(missing)}")
            column = {key: index[name] for key, name in _BDB_REQUIRED_COLUMNS.items()}

            for source_row_number, fields in enumerate(reader, start=2):
                physical["bindingdb_physical_rows"] += 1
                if len(fields) < len(header):
                    fields.extend([""] * (len(header) - len(fields)))
                elif len(fields) > len(header):
                    physical["bindingdb_rows_with_extra_fields"] += 1
                    fields = fields[: len(header)]
                populated = [endpoint for endpoint in ENDPOINTS if _clean(fields[column[endpoint]])]
                if not populated:
                    physical["bindingdb_rows_without_selected_endpoint"] += 1
                    continue
                source_category = _clean(fields[column["source"]]) or "unspecified"
                source_counts[source_category] += 1
                if source_category == "ChEMBL":
                    physical["bindingdb_explicit_chembl_mirror_rows"] += 1
                    for endpoint in populated:
                        exclusions[("BindingDB_202608", endpoint, "explicit_chembl_source_mirror")] += 1
                    continue
                physical["bindingdb_non_chembl_rows_with_selected_endpoint"] += 1
                if source_category in RIGHTS_QUARANTINE_SOURCES:
                    physical["bindingdb_rights_quarantined_rows"] += 1
                    for endpoint in populated:
                        exclusions[("BindingDB_202608", endpoint, "rights_pending_source")] += 1
                    continue

                parsed: dict[str, tuple[str, float, float | None, float | None]] = {}
                for endpoint in populated:
                    result = _parse_bindingdb_value(fields[column[endpoint]])
                    if result is None:
                        exclusions[("BindingDB_202608", endpoint, "unparseable_or_nonpositive_label")] += 1
                    else:
                        parsed[endpoint] = result
                if not parsed:
                    continue

                chain_count_text = _clean(fields[column["chain_count"]])
                try:
                    chain_count = int(float(chain_count_text))
                except ValueError:
                    chain_count = 0
                if chain_count != 1:
                    physical["bindingdb_non_single_chain_rows"] += 1
                    for endpoint in parsed:
                        exclusions[("BindingDB_202608", endpoint, "not_single_protein_chain")] += 1
                    continue

                ligand = _resolve_structure(connection, fields[column["smiles"]])
                if ligand is None:
                    physical["bindingdb_current_policy_structure_failures"] += 1
                    for endpoint in parsed:
                        exclusions[("BindingDB_202608", endpoint, "current_policy_structure_failure")] += 1
                    continue

                accessions = _normalize_accessions(fields[column["swissprot"]], fields[column["trembl"]])
                sequence = _normalize_sequence(fields[column["sequence"]])
                target = _register_target(
                    targets,
                    union,
                    accessions=accessions,
                    sequence=sequence,
                    target_name=fields[column["target_name"]],
                    organism=fields[column["organism"]],
                    source_dataset="BindingDB_202608",
                )
                if target is None:
                    physical["bindingdb_rows_missing_accession_and_sequence"] += 1
                    for endpoint in parsed:
                        exclusions[("BindingDB_202608", endpoint, "missing_accession_and_sequence")] += 1
                    continue

                doi = _clean(fields[column["doi"]])
                pmid = _clean(fields[column["pmid"]])
                patent = _clean(fields[column["patent"]])
                document_id = _document_identity(doi, pmid, patent)
                reactant_set = _clean(fields[column["reactant_set"]]) or f"row:{source_row_number}"
                ph = _clean(fields[column["ph"]])
                temperature = _clean(fields[column["temperature"]])
                pubchem_aid = _clean(fields[column["pubchem_aid"]])
                assay_id = (
                    f"PubChem:AID:{pubchem_aid}"
                    if pubchem_aid
                    else _digest(
                        "BindingDB_202608",
                        target.target_id,
                        document_id,
                        ph,
                        temperature,
                        prefix="ASSAYCTX",
                    )
                )
                source_ligand_id = (
                    _clean(fields[column["monomer_id"]])
                    or _clean(fields[column["pubchem_cid"]])
                    or _clean(fields[column["chembl_ligand_id"]])
                )

                for endpoint, (relation, value, lower, upper) in parsed.items():
                    mirror_keys: list[bytes] = []
                    if document_id:
                        mirror_keys = [
                            _measurement_key(
                                structure_id=ligand["structure_id"],
                                target_token=token,
                                endpoint=endpoint,
                                relation=relation,
                                value=value,
                                lower=lower,
                                upper=upper,
                                document_identity=document_id,
                            )
                            for token in target.tokens
                        ]
                    matched_sources: set[str] = set()
                    if mirror_keys:
                        placeholders = ",".join("?" for _ in mirror_keys)
                        matched_sources.update(
                            row[0]
                            for row in connection.execute(
                                f"SELECT source_dataset FROM strong_mirror_keys "
                                f"WHERE mirror_key IN ({placeholders})",
                                mirror_keys,
                            )
                        )
                    if "ChEMBL_37" in matched_sources:
                        exclusions[("BindingDB_202608", endpoint, "same_document_chembl_mirror")] += 1
                        continue
                    if "BindingDB_202608" in matched_sources:
                        exclusions[("BindingDB_202608", endpoint, "same_document_internal_exact_mirror")] += 1
                        continue

                    for key in mirror_keys:
                        connection.execute(
                            "INSERT OR IGNORE INTO strong_mirror_keys VALUES (?, ?)",
                            (key, "BindingDB_202608"),
                        )
                    _admit_ligand(connection, ligand)
                    used_target_ids.add(target.target_id)
                    source_observation_id = f"BindingDB:reactant_set:{reactant_set}:{endpoint}"
                    writers[endpoint].append(
                        {
                            "observation_id": _digest(
                                "BindingDB_202608", reactant_set, endpoint, prefix="AFFOBS"
                            ),
                            "source_dataset": "BindingDB_202608",
                            "source_snapshot": "BindingDB_All_202608",
                            "source_category": source_category,
                            "source_observation_id": source_observation_id,
                            "source_record_id": f"BindingDB:reactant_set:{reactant_set}",
                            "source_row_number": source_row_number,
                            "source_ligand_id": source_ligand_id or None,
                            "structure_id": ligand["structure_id"],
                            "scaffold_group_id": ligand["scaffold_group_id"],
                            "target_id": target.target_id,
                            "endpoint": endpoint,
                            "endpoint_role": (
                                "primary" if endpoint in PRIMARY_ENDPOINTS else "auxiliary_potency"
                            ),
                            "label_relation": relation,
                            "label_value_nM": value,
                            "label_lower_bound_nM": lower,
                            "label_upper_bound_nM": upper,
                            "label_unit": "nM",
                            "label_censoring": _censoring(relation),
                            "native_label": _clean(fields[column[endpoint]]) or None,
                            "assay_id": assay_id,
                            "assay_family": "bindingdb_reported_context",
                            "assay_description": None,
                            "assay_matrix": None,
                            "assay_route": None,
                            "assay_ph": ph or None,
                            "assay_temperature": temperature or None,
                            "document_doi": doi or None,
                            "document_pubmed_id": pmid or None,
                            "document_patent_id": patent or None,
                            "document_year": _year(fields[column["publication_date"]]),
                            "document_identity": document_id or None,
                            "rights_status": _rights_status("BindingDB_202608", source_category),
                            "mirror_status": "independent_after_strong_mirror_screen",
                        }
                    )
                    accepted_since_commit += 1
                    if accepted_since_commit % 20_000 == 0:
                        connection.commit()
    connection.commit()


def _copy_ligand_registry(connection: sqlite3.Connection, root: Path) -> int:
    writer = _PartitionWriter(root, _LIGAND_SCHEMA)
    count = 0
    cursor = connection.execute(
        """SELECT structure_id, standardized_smiles, standard_inchi_key,
                  scaffold_group_id, scaffold_status, ligand_cold_split, standardization_version,
                  rdkit_version, formal_charge, fragment_count
           FROM ligands ORDER BY structure_id"""
    )
    while True:
        rows = cursor.fetchmany(20_000)
        if not rows:
            break
        for values in rows:
            writer.append(
                {
                    "structure_id": values[0],
                    "standardized_smiles": values[1],
                    "standard_inchi_key": values[2],
                    "scaffold_group_id": values[3],
                    "scaffold_derivation_status": values[4],
                    "ligand_cold_split": values[5],
                    "structure_standardization_version": values[6],
                    "rdkit_version": values[7],
                    "formal_charge": values[8],
                    "source_fragment_count": values[9],
                }
            )
            count += 1
    writer.close()
    return count


def _lineage_id(row: Mapping[str, Any], target_group: str) -> str:
    document = _clean(row["document_identity"])
    if not document:
        document = f"unknown:{row['source_dataset']}:{row['source_observation_id']}"
    return _digest(
        row["structure_id"],
        target_group,
        row["endpoint"],
        row["label_relation"],
        row["label_value_nM"],
        row["label_lower_bound_nM"],
        row["label_upper_bound_nM"],
        document,
        prefix="MLIN",
    )


def _finalize_observations(
    *,
    raw_root: Path,
    final_root: Path,
    connection: sqlite3.Connection,
    target_lookup: Mapping[str, Mapping[str, Any]],
) -> tuple[
    Counter[tuple[str, str, str]],
    Counter[tuple[str, str, str, str]],
    Counter[tuple[str, str, str]],
]:
    endpoint_counts: Counter[tuple[str, str, str]] = Counter()
    task_counts: Counter[tuple[str, str, str, str]] = Counter()
    split_counts: Counter[tuple[str, str, str]] = Counter()
    pair_batch: list[tuple[Any, ...]] = []
    endpoint_pair_batch: list[tuple[Any, ...]] = []

    for endpoint in ENDPOINTS:
        writer = _PartitionWriter(final_root / endpoint.casefold(), _OBSERVATION_SCHEMA)
        for path in sorted((raw_root / endpoint.casefold()).glob("part-*.parquet")):
            parquet = pq.ParquetFile(path)
            if not parquet.schema_arrow.remove_metadata().equals(
                _RAW_OBSERVATION_SCHEMA, check_metadata=True
            ):
                raise AffinityTrainingSurfaceError(f"raw observation schema drift: {path}")
            for batch in parquet.iter_batches(batch_size=20_000):
                for row in batch.to_pylist():
                    target = target_lookup.get(row["target_id"])
                    if target is None:
                        raise AffinityTrainingSurfaceError(
                            f"observation has unresolved target: {row['target_id']}"
                        )
                    target_group = str(target["target_leakage_group_id"])
                    target_split = str(target["target_cold_split"])
                    ligand_split = _split(str(row["scaffold_group_id"]), dimension="ligand_scaffold")
                    double_split = ligand_split if ligand_split == target_split else None
                    pair_id = _digest(row["structure_id"], target_group, prefix="PAIR")
                    endpoint_pair_id = _digest(pair_id, endpoint, prefix="EPAIR")
                    task_id = _digest(endpoint, target_group, prefix="TASK")
                    lineage_id = _lineage_id(row, target_group)
                    final = dict(row)
                    final.update(
                        {
                            "target_leakage_group_id": target_group,
                            "ligand_cold_split": ligand_split,
                            "target_cold_split": target_split,
                            "double_cold_split": double_split,
                            "double_cold_eligible": double_split is not None,
                            "sequence_model_eligible": bool(target["sequence_model_eligible"]),
                            "ligand_target_pair_id": pair_id,
                            "target_endpoint_task_id": task_id,
                            "measurement_lineage_group_id": lineage_id,
                            "scientific_training_eligible": True,
                            "eligibility_basis": (
                                "positive_nM_endpoint_separated_single_protein_parent_structure_"
                                "accession_or_sequence_mirror_screened"
                            ),
                        }
                    )
                    writer.append(final)
                    source = str(row["source_dataset"])
                    censoring = str(row["label_censoring"])
                    endpoint_counts[(endpoint, "observations", "all")] += 1
                    endpoint_counts[(endpoint, "source", source)] += 1
                    endpoint_counts[(endpoint, "censoring", censoring)] += 1
                    endpoint_counts[
                        (endpoint, "sequence_model_eligible", str(bool(target["sequence_model_eligible"])))
                    ] += 1
                    split_counts[(endpoint, "ligand_cold", ligand_split)] += 1
                    split_counts[(endpoint, "target_cold", target_split)] += 1
                    split_counts[(endpoint, "double_cold", double_split or "mixed_ineligible")] += 1
                    task_counts[(endpoint, target_group, "observations", "all")] += 1
                    task_counts[(endpoint, target_group, "source", source)] += 1
                    task_counts[(endpoint, target_group, "censoring", censoring)] += 1
                    if bool(target["sequence_model_eligible"]):
                        task_counts[(endpoint, target_group, "eligibility", "sequence")] += 1
                    if double_split:
                        task_counts[(endpoint, target_group, "eligibility", "double_cold")] += 1
                    pair_batch.append(
                        (
                            pair_id,
                            row["structure_id"],
                            row["scaffold_group_id"],
                            target_group,
                            ligand_split,
                            target_split,
                            double_split,
                        )
                    )
                    endpoint_pair_batch.append(
                        (endpoint_pair_id, endpoint, row["structure_id"], target_group)
                    )
                    if len(pair_batch) >= 20_000:
                        connection.executemany(
                            "INSERT OR IGNORE INTO ligand_target_pairs VALUES (?, ?, ?, ?, ?, ?, ?)",
                            pair_batch,
                        )
                        connection.executemany(
                            "INSERT OR IGNORE INTO endpoint_pairs VALUES (?, ?, ?, ?)",
                            endpoint_pair_batch,
                        )
                        connection.commit()
                        pair_batch.clear()
                        endpoint_pair_batch.clear()
        writer.close()
    if pair_batch:
        connection.executemany(
            "INSERT OR IGNORE INTO ligand_target_pairs VALUES (?, ?, ?, ?, ?, ?, ?)", pair_batch
        )
        connection.executemany(
            "INSERT OR IGNORE INTO endpoint_pairs VALUES (?, ?, ?, ?)", endpoint_pair_batch
        )
        connection.commit()
    return endpoint_counts, task_counts, split_counts


def _copy_pair_registry(connection: sqlite3.Connection, root: Path) -> int:
    writer = _PartitionWriter(root, _PAIR_SCHEMA)
    count = 0
    cursor = connection.execute(
        """SELECT pair_id, structure_id, scaffold_group_id, target_group_id,
                  ligand_split, target_split, double_split
           FROM ligand_target_pairs ORDER BY pair_id"""
    )
    while True:
        rows = cursor.fetchmany(20_000)
        if not rows:
            break
        for row in rows:
            writer.append(
                {
                    "ligand_target_pair_id": row[0],
                    "structure_id": row[1],
                    "scaffold_group_id": row[2],
                    "target_leakage_group_id": row[3],
                    "ligand_cold_split": row[4],
                    "target_cold_split": row[5],
                    "double_cold_split": row[6],
                    "double_cold_eligible": row[6] is not None,
                }
            )
            count += 1
    writer.close()
    return count


def _task_registry(
    *,
    connection: sqlite3.Connection,
    task_counts: Counter[tuple[str, str, str, str]],
    target_lookup: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    group_splits = {
        str(row["target_leakage_group_id"]): str(row["target_cold_split"]) for row in target_lookup.values()
    }
    unique_structures = {
        (str(endpoint), str(target_group)): int(count)
        for endpoint, target_group, count in connection.execute(
            """SELECT endpoint, target_group_id, COUNT(*)
               FROM endpoint_pairs GROUP BY endpoint, target_group_id"""
        )
    }
    tasks = sorted(
        {
            (endpoint, target_group)
            for endpoint, target_group, dimension, category in task_counts
            if dimension == "observations" and category == "all"
        }
    )
    rows: list[dict[str, Any]] = []
    for endpoint, target_group in tasks:
        observation_count = task_counts[(endpoint, target_group, "observations", "all")]
        source_counts = {
            category: count
            for (candidate_endpoint, candidate_target, dimension, category), count in task_counts.items()
            if candidate_endpoint == endpoint and candidate_target == target_group and dimension == "source"
        }
        rows.append(
            {
                "target_endpoint_task_id": _digest(endpoint, target_group, prefix="TASK"),
                "endpoint": endpoint,
                "endpoint_role": "primary" if endpoint in PRIMARY_ENDPOINTS else "auxiliary_potency",
                "target_leakage_group_id": target_group,
                "target_cold_split": group_splits[target_group],
                "observation_count": observation_count,
                "unique_structure_count": unique_structures[(endpoint, target_group)],
                "exact_count": task_counts[(endpoint, target_group, "censoring", "exact")],
                "left_censored_count": task_counts[(endpoint, target_group, "censoring", "left_censored")],
                "right_censored_count": task_counts[(endpoint, target_group, "censoring", "right_censored")],
                "interval_censored_count": task_counts[
                    (endpoint, target_group, "censoring", "interval_censored")
                ],
                "sequence_model_eligible_count": task_counts[
                    (endpoint, target_group, "eligibility", "sequence")
                ],
                "double_cold_eligible_count": task_counts[
                    (endpoint, target_group, "eligibility", "double_cold")
                ],
                "source_counts_json": json.dumps(source_counts, sort_keys=True, separators=(",", ":")),
                "at_least_100_observations": observation_count >= 100,
                "at_least_1000_observations": observation_count >= 1000,
                "task_semantics": (
                    "protein_conditioned_positive_nM_continuous_exact_and_censored;endpoint_not_pooled"
                ),
            }
        )
    return rows


def _endpoint_summary(
    *,
    connection: sqlite3.Connection,
    endpoint_counts: Counter[tuple[str, str, str]],
    split_counts: Counter[tuple[str, str, str]],
    task_rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    unique_pairs = {
        str(endpoint): int(count)
        for endpoint, count in connection.execute(
            "SELECT endpoint, COUNT(*) FROM endpoint_pairs GROUP BY endpoint"
        )
    }
    unique_structures = {
        str(endpoint): int(count)
        for endpoint, count in connection.execute(
            "SELECT endpoint, COUNT(DISTINCT structure_id) FROM endpoint_pairs GROUP BY endpoint"
        )
    }
    tasks_per_endpoint = Counter(str(row["endpoint"]) for row in task_rows)
    summary: dict[str, dict[str, Any]] = {}
    for endpoint in ENDPOINTS:
        observations = endpoint_counts[(endpoint, "observations", "all")]
        sources = {
            category: count
            for (candidate, dimension, category), count in endpoint_counts.items()
            if candidate == endpoint and dimension == "source"
        }
        censoring = {
            category: count
            for (candidate, dimension, category), count in endpoint_counts.items()
            if candidate == endpoint and dimension == "censoring"
        }
        splits: dict[str, dict[str, int]] = {}
        for dimension in ("ligand_cold", "target_cold", "double_cold"):
            splits[dimension] = {
                category: count
                for (candidate, candidate_dimension, category), count in split_counts.items()
                if candidate == endpoint and candidate_dimension == dimension
            }
        summary[endpoint] = {
            "endpoint_role": "primary" if endpoint in PRIMARY_ENDPOINTS else "auxiliary_potency",
            "observations": observations,
            "source_counts": dict(sorted(sources.items())),
            "censoring_counts": dict(sorted(censoring.items())),
            "unique_structures": unique_structures.get(endpoint, 0),
            "unique_ligand_target_pairs": unique_pairs.get(endpoint, 0),
            "target_endpoint_tasks": tasks_per_endpoint[endpoint],
            "sequence_model_eligible_observations": endpoint_counts[
                (endpoint, "sequence_model_eligible", "True")
            ],
            "split_counts": splits,
        }
    return summary


def _contract() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "release_id": RELEASE_ID,
        "prediction_unit": "one_documented_ligand_target_endpoint_measurement",
        "endpoint_contract": {
            "Kd": "primary_equilibrium_dissociation_constant_nM",
            "Ki": "primary_inhibition_constant_nM",
            "IC50": "primary_assay_dependent_half_maximal_inhibition_nM",
            "EC50": "auxiliary_assay_dependent_half_maximal_effect_nM_not_affinity",
            "pooling_across_endpoints_allowed": False,
            "binding_free_energy_precomputed": False,
            "binding_free_energy_reason": (
                "Delta-G derivation requires an explicit thermodynamic endpoint, standard state, and "
                "temperature; IC50, Ki, and EC50 are not silently converted"
            ),
        },
        "label_contract": {
            "unit": "nM",
            "positive_values_only": True,
            "relations": sorted(SUPPORTED_RELATIONS),
            "exact": "lower=value=upper",
            "left_censored": "relation < or <=; upper bound retained",
            "right_censored": "relation > or >=; lower bound retained",
            "interval_censored": "both bounds retained; no midpoint imputation",
            "censored_values_promoted_to_points": False,
        },
        "identity_contract": {
            "ligand_parent_policy": STANDARDIZATION_VERSION,
            "tautomers_collapsed": False,
            "scaffold": (
                "Bemis-Murcko; acyclic structures and rare RDKit scaffold-extraction failures use "
                "the exact standardized parent as a conservative no-leak fallback"
            ),
            "target_record": "reported accession set plus exact reported sequence hash",
            "target_leakage_group": (
                "connected component over exact accession tokens and exact sequence SHA-256 tokens"
            ),
            "unambiguous_sequence_resolution": (
                "an accession-only record receives a sequence only when its connected component has "
                "exactly one reported sequence"
            ),
            "homology_clustering_completed": False,
            "homology_boundary": (
                "exact accession/sequence leakage is blocked now; sequence-similarity cluster splits "
                "remain a later HPC preprocessing step"
            ),
        },
        "mirror_contract": {
            "bindingdb_rows_declared_as_chembl": "excluded_without_counting_as_independent",
            "strong_cross_source_key": (
                "current-policy parent structure + exact accession-or-sequence token + endpoint + "
                "relation/bounds + normalized DOI/PMID/patent"
            ),
            "same_document_exact_non_chembl_mirrors": "one retained; later rows excluded",
            "unknown_document_replicates": "retained because experimental identity is not established",
            "equal_values_without_shared_document": "not treated as duplicates",
            "measurement_lineage_group": "materialized on every retained observation",
        },
        "split_contract": {
            "policy_version": SPLIT_POLICY_VERSION,
            "seed": SPLIT_SEED,
            "ratios": {"train": 80, "validation": 10, "test": 10},
            "label_columns_read_for_assignment": False,
            "ligand_cold_group": "scaffold_group_id",
            "target_cold_group": "target_leakage_group_id",
            "double_cold": (
                "eligible only when independently assigned ligand-scaffold and target-component "
                "partitions agree; mixed assignments are explicitly null"
            ),
            "exact_structure_cross_ligand_cold_split_allowed": False,
            "scaffold_cross_ligand_cold_split_allowed": False,
            "exact_accession_or_sequence_component_cross_target_cold_split_allowed": False,
        },
        "eligibility_contract": {
            "scientific_training_eligible": (
                "valid current-policy parent structure, single reported protein chain, accession or "
                "sequence, positive parseable endpoint-specific nM label, supported relation, and "
                "strong mirror screen"
            ),
            "sequence_model_eligible": "scientific eligibility plus reported or unambiguously resolved sequence",
            "ec50_primary_affinity_eligible": False,
            "rights_quarantined_sources": sorted(RIGHTS_QUARANTINE_SOURCES),
            "redistribution": (
                "source terms and row provenance are preserved; non-curated BindingDB imports require "
                "source-specific distribution review"
            ),
        },
        "execution_boundary": {
            "production_features_generated": False,
            "model_fitting_performed": False,
            "hyperparameter_search_performed": False,
            "hpc_executed": False,
            "predictive_superiority_established": False,
        },
    }


def _report_text(manifest_counts: Mapping[str, Any], exclusions: Mapping[str, Any]) -> str:
    endpoints = manifest_counts["endpoints"]
    total = manifest_counts["total"]
    lines = [
        "# Protein--ligand affinity and potency training surfaces",
        "",
        "## Outcome",
        "",
        (
            "This release creates a local, endpoint-separated protein--molecule training surface from "
            "the canonical ChEMBL 37 tasks and the independent portion of BindingDB 2026-08. It keeps "
            "real censoring, standardized parent structures, accession/sequence target identity, assay "
            "and document provenance, and leakage-safe split memberships."
        ),
        "",
        "## Scale",
        "",
        f"- Retained observations: {total['observations']:,}.",
        f"- Primary Kd/Ki/IC50 observations: {total['primary_observations']:,}.",
        f"- Auxiliary EC50 observations: {total['auxiliary_ec50_observations']:,}.",
        f"- Unique standardized ligand parents: {total['unique_structures']:,}.",
        f"- Unique leakage-grouped targets: {total['unique_target_groups']:,}.",
        f"- Unique ligand--target pairs: {total['unique_ligand_target_pairs']:,}.",
        f"- Target-by-endpoint tasks: {total['target_endpoint_tasks']:,}.",
        (
            "- Exact-structure scaffold fallbacks after a rare RDKit Murcko failure: "
            f"{total['scaffold_derivation_status_counts'].get('murcko_failure_exact_structure_fallback', 0):,}."
        ),
        "",
    ]
    for endpoint in ENDPOINTS:
        item = endpoints[endpoint]
        lines.extend(
            [
                f"### {endpoint}",
                "",
                f"- Observations: {item['observations']:,}.",
                f"- Unique ligand--target pairs: {item['unique_ligand_target_pairs']:,}.",
                f"- Unique structures: {item['unique_structures']:,}.",
                f"- Target-specific tasks: {item['target_endpoint_tasks']:,}.",
                (
                    "- Endpoint role: primary protein--molecule task."
                    if endpoint in PRIMARY_ENDPOINTS
                    else "- Endpoint role: auxiliary potency only; it is not affinity."
                ),
                "",
            ]
        )
    lines.extend(
        [
            "## What was deliberately removed",
            "",
            f"- Explicit BindingDB rows sourced from ChEMBL: {exclusions.get('explicit_chembl_source_mirror', 0):,} endpoint records.",
            f"- Same-document exact ChEMBL mirrors found in non-ChEMBL BindingDB rows: {exclusions.get('same_document_chembl_mirror', 0):,}.",
            f"- Same-document exact internal BindingDB mirrors: {exclusions.get('same_document_internal_exact_mirror', 0):,}.",
            f"- Rights-pending source records: {exclusions.get('rights_pending_source', 0):,}.",
            "- Multi-chain targets, missing target identity, invalid parent structures, nonpositive labels, and unsupported label syntax were excluded and counted in the manifest.",
            "",
            "## Scientific boundaries",
            "",
            "- Kd, Ki, IC50, and EC50 remain separate. No endpoint substitution or silent pooling occurred.",
            "- Less-than, greater-than, inclusive inequalities, and ChEMBL intervals remain bounds. No censored threshold was converted to an exact point.",
            "- Binding free energy was not manufactured from IC50, Ki, or EC50. Kd-to-Delta-G also remains deferred where temperature and standard-state evidence are absent.",
            "- Exact structure and scaffold groups cannot cross ligand-cold splits. Exact accession/sequence connected components cannot cross target-cold splits.",
            "- Double-cold membership is available only when the independently assigned ligand and target splits agree; mixed assignments are explicitly ineligible rather than forced.",
            "- Exact target identity leakage is controlled. Homology-level clustering is not claimed and remains a later sequence preprocessing step.",
            "- The release is trainable data preparation. It contains no generated production features, fitted models, HPC execution, or evidence of predictive superiority.",
            "",
            "## Recommended first training order",
            "",
            "- Begin with IC50 because it is the largest primary surface, while retaining assay context and evaluating target-cold and double-cold generalization.",
            "- Train Ki as the cleaner equilibrium-style primary complement, then Kd as the most direct affinity endpoint despite its smaller scale.",
            "- Use EC50 only as an auxiliary potency task with its own head and metrics.",
            "- Compare exact-only baselines against censor-aware objectives. Never use a censored threshold as an exact regression target.",
            "- Report random-like performance only as a diagnostic; scaffold-cold, target-cold, and double-cold results are the meaningful generalization tests.",
            "",
        ]
    )
    return "\n".join(lines)


def _verify_bindingdb_acquisition(archive_path: Path, acquisition_manifest_path: Path) -> None:
    manifest = json.loads(acquisition_manifest_path.read_text(encoding="utf-8"))
    files = manifest.get("files")
    if not isinstance(files, list):
        raise AffinityTrainingSurfaceError("external acquisition manifest lacks files")
    matches = [
        item
        for item in files
        if isinstance(item, dict)
        and item.get("source") == "BindingDB"
        and Path(_clean(item.get("path"))).name == archive_path.name
    ]
    if len(matches) != 1:
        raise AffinityTrainingSurfaceError(
            "BindingDB archive is not uniquely declared by acquisition manifest"
        )
    declared = matches[0]
    if (
        archive_path.stat().st_size != int(declared.get("bytes", -1))
        or _sha256(archive_path) != declared.get("sha256")
        or not bool(declared.get("checksum_verified"))
    ):
        raise AffinityTrainingSurfaceError("BindingDB acquisition binding failed")


def build_affinity_training_surfaces(
    *,
    canonical_chembl_root: Path,
    bindingdb_archive: Path,
    acquisition_manifest: Path,
    output_root: Path,
    report_mirror_path: Path | None = None,
) -> dict[str, Any]:
    """Build the immutable ChEMBL + independent-BindingDB training release."""

    if output_root.exists():
        raise AffinityTrainingSurfaceError(f"refusing to overwrite existing release: {output_root}")
    for path in (
        canonical_chembl_root / "build_manifest.json",
        canonical_chembl_root / "task_datasets.json",
        bindingdb_archive,
        acquisition_manifest,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    _verify_bindingdb_acquisition(bindingdb_archive, acquisition_manifest)
    task_datasets_path = canonical_chembl_root / "task_datasets.json"
    task_datasets = json.loads(task_datasets_path.read_text(encoding="utf-8"))
    if not isinstance(task_datasets, dict):
        raise AffinityTrainingSurfaceError("ChEMBL task_datasets.json is not an object")
    selected_keys = _task_keys(task_datasets)
    chembl_parts, chembl_bindings = _verified_chembl_parts(
        canonical_chembl_root, task_datasets, selected_keys
    )

    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent))
    work_db_path = staging / ".affinity_build.sqlite"
    raw_root = staging / ".raw_observations"
    connection = _initialize_work_database(work_db_path)
    targets: dict[str, _TargetRecord] = {}
    union = _UnionFind()
    used_target_ids: set[str] = set()
    exclusions: Counter[tuple[str, str, str]] = Counter()
    physical: Counter[str] = Counter()
    bindingdb_sources: Counter[str] = Counter()
    raw_writers = {
        endpoint: _PartitionWriter(raw_root / endpoint.casefold(), _RAW_OBSERVATION_SCHEMA)
        for endpoint in ENDPOINTS
    }
    try:
        _chembl_observations(
            parts=chembl_parts,
            connection=connection,
            writers=raw_writers,
            targets=targets,
            union=union,
            used_target_ids=used_target_ids,
            exclusions=exclusions,
            physical=physical,
        )
        _bindingdb_observations(
            archive_path=bindingdb_archive,
            connection=connection,
            writers=raw_writers,
            targets=targets,
            union=union,
            used_target_ids=used_target_ids,
            exclusions=exclusions,
            physical=physical,
            source_counts=bindingdb_sources,
        )
        for writer in raw_writers.values():
            writer.close()

        target_lookup, target_rows, target_group_rows = _resolve_target_groups(
            targets, union, used_target_ids
        )
        _write_rows(target_rows, staging / "targets" / "target_registry.parquet", _TARGET_SCHEMA)
        _write_rows(
            target_group_rows,
            staging / "targets" / "target_leakage_group_registry.parquet",
            _TARGET_GROUP_SCHEMA,
        )
        endpoint_counts, task_counts, split_counts = _finalize_observations(
            raw_root=raw_root,
            final_root=staging / "observations",
            connection=connection,
            target_lookup=target_lookup,
        )
        ligand_count = _copy_ligand_registry(connection, staging / "ligands")
        scaffold_status_counts = {
            str(status): int(count)
            for status, count in connection.execute(
                "SELECT scaffold_status, COUNT(*) FROM ligands GROUP BY scaffold_status"
            )
        }
        pair_count = _copy_pair_registry(connection, staging / "ligand_target_pairs")
        task_rows = _task_registry(
            connection=connection, task_counts=task_counts, target_lookup=target_lookup
        )
        _write_rows(
            task_rows,
            staging / "target_endpoint_task_registry.parquet",
            _TASK_SCHEMA,
        )
        exclusion_rows = [
            {
                "source_dataset": source,
                "endpoint": endpoint,
                "exclusion_reason": reason,
                "record_count": count,
                "exclusion_scope": "endpoint_record",
            }
            for (source, endpoint, reason), count in sorted(exclusions.items())
        ]
        _write_rows(
            exclusion_rows,
            staging / "exclusion_summary.parquet",
            _EXCLUSION_SCHEMA,
        )

        endpoint_summary = _endpoint_summary(
            connection=connection,
            endpoint_counts=endpoint_counts,
            split_counts=split_counts,
            task_rows=task_rows,
        )
        observations = sum(item["observations"] for item in endpoint_summary.values())
        primary_observations = sum(
            endpoint_summary[endpoint]["observations"] for endpoint in PRIMARY_ENDPOINTS
        )
        total_counts = {
            "observations": observations,
            "primary_observations": primary_observations,
            "auxiliary_ec50_observations": endpoint_summary["EC50"]["observations"],
            "unique_structures": ligand_count,
            "unique_target_records": len(target_rows),
            "unique_target_groups": len(target_group_rows),
            "unique_ligand_target_pairs": pair_count,
            "unique_endpoint_ligand_target_pairs": sum(
                int(row["unique_structure_count"]) for row in task_rows
            ),
            "target_endpoint_tasks": len(task_rows),
            "tasks_with_at_least_100_observations": sum(
                bool(row["at_least_100_observations"]) for row in task_rows
            ),
            "tasks_with_at_least_1000_observations": sum(
                bool(row["at_least_1000_observations"]) for row in task_rows
            ),
            "sequence_model_eligible_observations": sum(
                item["sequence_model_eligible_observations"] for item in endpoint_summary.values()
            ),
            "double_cold_eligible_observations": sum(
                item["split_counts"]["double_cold"].get("train", 0)
                + item["split_counts"]["double_cold"].get("validation", 0)
                + item["split_counts"]["double_cold"].get("test", 0)
                for item in endpoint_summary.values()
            ),
            "scaffold_derivation_status_counts": dict(sorted(scaffold_status_counts.items())),
        }
        exclusion_totals: Counter[str] = Counter()
        for (_, _, reason), count in exclusions.items():
            exclusion_totals[reason] += count

        contract_path = staging / CONTRACT_NAME
        contract_path.write_text(json.dumps(_contract(), indent=2, sort_keys=True) + "\n")
        report_path = staging / REPORT_NAME
        report_path.write_text(
            _report_text({"total": total_counts, "endpoints": endpoint_summary}, exclusion_totals)
        )

        shutil.rmtree(raw_root)
        connection.close()
        for sidecar in (
            work_db_path,
            work_db_path.with_name(work_db_path.name + "-wal"),
            work_db_path.with_name(work_db_path.name + "-shm"),
        ):
            if sidecar.exists():
                sidecar.unlink()

        module_path = Path(__file__).resolve()
        chemistry_path = module_path.with_name("chemistry.py")
        inputs = [
            _bind_input(
                canonical_chembl_root / "build_manifest.json", role="chembl_canonical_build_manifest"
            ),
            _bind_input(task_datasets_path, role="chembl_task_dataset_registry"),
            _bind_input(acquisition_manifest, role="external_acquisition_manifest"),
            _bind_input(bindingdb_archive, role="bindingdb_full_archive"),
            _bind_input(module_path, role="implementation"),
            _bind_input(chemistry_path, role="structure_standardization_implementation"),
            *chembl_bindings,
        ]
        artifacts: dict[str, dict[str, Any]] = {}
        for path in sorted(staging.rglob("*")):
            if path.is_file() and path.name != MANIFEST_NAME:
                binding = _bind_artifact(path, staging)
                artifacts[binding["path"]] = binding
        manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "release_id": RELEASE_ID,
            "inputs": inputs,
            "selected_chembl_task_keys": selected_keys,
            "artifacts": artifacts,
            "counts": {
                "total": total_counts,
                "endpoints": endpoint_summary,
                "exclusions": dict(sorted(exclusion_totals.items())),
                "physical_source_rows": dict(sorted(physical.items())),
                "bindingdb_source_row_counts": dict(sorted(bindingdb_sources.items())),
            },
            "status": {
                "endpoint_separated": True,
                "censoring_preserved": True,
                "explicit_chembl_bindingdb_mirrors_excluded": True,
                "strong_same_document_mirrors_excluded": True,
                "ligand_scaffold_split_exclusive": True,
                "target_exact_component_split_exclusive": True,
                "ec50_is_auxiliary_not_affinity": True,
                "binding_free_energy_computed": False,
                "production_features_generated": False,
                "model_fitting_performed": False,
                "hpc_executed": False,
                "figures_generated": False,
                "presentation_tables_generated": False,
            },
        }
        manifest["manifest_sha256"] = _manifest_hash(manifest)
        (staging / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        staging.rename(output_root)
        validated = validate_affinity_training_surfaces(output_root)
        if report_mirror_path is not None:
            report_mirror_path.parent.mkdir(parents=True, exist_ok=True)
            report_bytes = (output_root / REPORT_NAME).read_bytes()
            if report_mirror_path.exists() and report_mirror_path.read_bytes() != report_bytes:
                raise AffinityTrainingSurfaceError(
                    f"refusing to overwrite a different report mirror: {report_mirror_path}"
                )
            if not report_mirror_path.exists():
                report_mirror_path.write_bytes(report_bytes)
        return validated
    except Exception:
        try:
            connection.close()
        except Exception:
            pass
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise


def _iter_parquet_batches(
    paths: Sequence[Path], columns: Sequence[str] | None = None
) -> Iterator[pa.RecordBatch]:
    for path in paths:
        yield from pq.ParquetFile(path).iter_batches(batch_size=50_000, columns=columns)


def _assert_schema(paths: Sequence[Path], schema: pa.Schema, label: str) -> int:
    if not paths:
        raise AffinityTrainingSurfaceError(f"no {label} Parquet parts")
    rows = 0
    for path in paths:
        actual = pq.read_schema(path).remove_metadata()
        if not actual.equals(schema.remove_metadata(), check_metadata=True):
            raise AffinityTrainingSurfaceError(f"{label} schema mismatch: {path}")
        rows += pq.read_metadata(path).num_rows
    return rows


def _assert_sorted_unique(paths: Sequence[Path], key: str, label: str) -> int:
    previous = ""
    count = 0
    for batch in _iter_parquet_batches(paths, [key]):
        for value in batch.column(0).to_pylist():
            text = _clean(value)
            if not text or (previous and text <= previous):
                raise AffinityTrainingSurfaceError(f"{label} is not globally sorted and unique")
            previous = text
            count += 1
    return count


def validate_affinity_training_surfaces(output_root: Path) -> dict[str, Any]:
    """Replay physical, semantic, count, mirror, and leakage contracts."""

    manifest_path = output_root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise AffinityTrainingSurfaceError(f"missing manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("manifest_sha256") != _manifest_hash(manifest):
        raise AffinityTrainingSurfaceError("manifest self-hash mismatch")
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("release_id") != RELEASE_ID:
        raise AffinityTrainingSurfaceError("release identity mismatch")

    expected_files = {MANIFEST_NAME, *manifest.get("artifacts", {})}
    actual_files = {
        path.relative_to(output_root).as_posix() for path in output_root.rglob("*") if path.is_file()
    }
    if actual_files != expected_files:
        raise AffinityTrainingSurfaceError(
            f"unexpected output membership: {sorted(actual_files ^ expected_files)}"
        )
    for binding in manifest.get("inputs", []):
        path = Path(binding["path"])
        if (
            not path.is_file()
            or path.stat().st_size != int(binding["bytes"])
            or _sha256(path) != binding["sha256"]
        ):
            raise AffinityTrainingSurfaceError(f"input binding failed: {path}")
        if path.suffix == ".parquet":
            if (
                pq.read_metadata(path).num_rows != int(binding["rows"])
                or _file_schema_hash(path) != binding["arrow_schema_sha256"]
            ):
                raise AffinityTrainingSurfaceError(f"input Parquet binding failed: {path}")
    for relative, binding in manifest["artifacts"].items():
        if relative != binding.get("path"):
            raise AffinityTrainingSurfaceError(f"artifact key/path mismatch: {relative}")
        path = output_root / relative
        if (
            not path.is_file()
            or path.stat().st_size != int(binding["bytes"])
            or _sha256(path) != binding["sha256"]
        ):
            raise AffinityTrainingSurfaceError(f"artifact binding failed: {relative}")
        if path.suffix == ".parquet" and (
            pq.read_metadata(path).num_rows != int(binding["rows"])
            or _file_schema_hash(path) != binding["arrow_schema_sha256"]
        ):
            raise AffinityTrainingSurfaceError(f"artifact Parquet binding failed: {relative}")

    ligand_paths = sorted((output_root / "ligands").glob("part-*.parquet"))
    pair_paths = sorted((output_root / "ligand_target_pairs").glob("part-*.parquet"))
    target_path = output_root / "targets" / "target_registry.parquet"
    target_group_path = output_root / "targets" / "target_leakage_group_registry.parquet"
    task_path = output_root / "target_endpoint_task_registry.parquet"
    exclusion_path = output_root / "exclusion_summary.parquet"
    ligand_rows = _assert_schema(ligand_paths, _LIGAND_SCHEMA, "ligand registry")
    pair_rows = _assert_schema(pair_paths, _PAIR_SCHEMA, "pair registry")
    _assert_schema([target_path], _TARGET_SCHEMA, "target registry")
    _assert_schema([target_group_path], _TARGET_GROUP_SCHEMA, "target group registry")
    _assert_schema([task_path], _TASK_SCHEMA, "task registry")
    _assert_schema([exclusion_path], _EXCLUSION_SCHEMA, "exclusion summary")
    if _assert_sorted_unique(ligand_paths, "structure_id", "ligand registry") != ligand_rows:
        raise AffinityTrainingSurfaceError("ligand registry uniqueness replay failed")
    if _assert_sorted_unique(pair_paths, "ligand_target_pair_id", "pair registry") != pair_rows:
        raise AffinityTrainingSurfaceError("pair registry uniqueness replay failed")

    counts = manifest["counts"]
    total = counts["total"]
    if ligand_rows != int(total["unique_structures"]) or pair_rows != int(
        total["unique_ligand_target_pairs"]
    ):
        raise AffinityTrainingSurfaceError("registry count replay failed")
    scaffold_status_counts: Counter[str] = Counter()
    for batch in _iter_parquet_batches(ligand_paths, ["scaffold_derivation_status"]):
        scaffold_status_counts.update(str(value) for value in batch.column(0).to_pylist())
    if dict(sorted(scaffold_status_counts.items())) != total["scaffold_derivation_status_counts"]:
        raise AffinityTrainingSurfaceError("scaffold derivation status count replay failed")
    target_table = pq.read_table(target_path)
    target_group_table = pq.read_table(target_group_path)
    target_rows = target_table.to_pylist()
    target_group_rows = target_group_table.to_pylist()
    if len(target_rows) != int(total["unique_target_records"]) or len(target_group_rows) != int(
        total["unique_target_groups"]
    ):
        raise AffinityTrainingSurfaceError("target count replay failed")
    target_ids = {row["target_id"] for row in target_rows}
    if len(target_ids) != len(target_rows):
        raise AffinityTrainingSurfaceError("duplicate target_id")
    group_ids = {row["target_leakage_group_id"] for row in target_group_rows}
    if len(group_ids) != len(target_group_rows):
        raise AffinityTrainingSurfaceError("duplicate target leakage group")
    target_lookup = {row["target_id"]: row for row in target_rows}
    if any(row["target_leakage_group_id"] not in group_ids for row in target_rows):
        raise AffinityTrainingSurfaceError("target references unknown leakage group")

    observation_counter: Counter[tuple[str, str, str]] = Counter()
    split_counter: Counter[tuple[str, str, str]] = Counter()
    observed_rows = 0
    with tempfile.TemporaryDirectory(prefix="affinity-validator-") as temporary:
        uniqueness = sqlite3.connect(Path(temporary) / "observation_ids.sqlite")
        uniqueness.executescript(
            """
            PRAGMA journal_mode=OFF;
            PRAGMA synchronous=OFF;
            PRAGMA temp_store=FILE;
            CREATE TABLE ligand_lookup (
                structure_id TEXT PRIMARY KEY,
                scaffold_group_id TEXT NOT NULL,
                ligand_split TEXT NOT NULL
            ) WITHOUT ROWID;
            CREATE TABLE pair_lookup (
                pair_id TEXT PRIMARY KEY,
                structure_id TEXT NOT NULL,
                scaffold_group_id TEXT NOT NULL,
                target_group_id TEXT NOT NULL,
                ligand_split TEXT NOT NULL,
                target_split TEXT NOT NULL,
                double_split TEXT
            ) WITHOUT ROWID;
            CREATE TABLE task_lookup (task_id TEXT PRIMARY KEY) WITHOUT ROWID;
            CREATE TABLE ids (
                id TEXT PRIMARY KEY,
                structure_id TEXT NOT NULL,
                scaffold_group_id TEXT NOT NULL,
                pair_id TEXT NOT NULL,
                task_id TEXT NOT NULL,
                target_group_id TEXT NOT NULL,
                ligand_split TEXT NOT NULL,
                target_split TEXT NOT NULL,
                double_split TEXT
            ) WITHOUT ROWID;
            """
        )
        for batch in _iter_parquet_batches(
            ligand_paths, ["structure_id", "scaffold_group_id", "ligand_cold_split"]
        ):
            uniqueness.executemany(
                "INSERT INTO ligand_lookup VALUES (?, ?, ?)",
                zip(*batch.to_pydict().values(), strict=True),
            )
        for batch in _iter_parquet_batches(pair_paths):
            pair_rows_for_db: list[tuple[Any, ...]] = []
            for row in batch.to_pylist():
                expected_pair_id = _digest(row["structure_id"], row["target_leakage_group_id"], prefix="PAIR")
                expected_ligand_split = _split(row["scaffold_group_id"], dimension="ligand_scaffold")
                expected_target_split = _split(row["target_leakage_group_id"], dimension="target_component")
                expected_double = (
                    expected_ligand_split if expected_ligand_split == expected_target_split else None
                )
                if (
                    row["ligand_target_pair_id"] != expected_pair_id
                    or row["ligand_cold_split"] != expected_ligand_split
                    or row["target_cold_split"] != expected_target_split
                    or row["double_cold_split"] != expected_double
                    or row["double_cold_eligible"] != (expected_double is not None)
                ):
                    raise AffinityTrainingSurfaceError("pair identity/split contract failed")
                pair_rows_for_db.append(
                    (
                        row["ligand_target_pair_id"],
                        row["structure_id"],
                        row["scaffold_group_id"],
                        row["target_leakage_group_id"],
                        row["ligand_cold_split"],
                        row["target_cold_split"],
                        row["double_cold_split"],
                    )
                )
            uniqueness.executemany("INSERT INTO pair_lookup VALUES (?, ?, ?, ?, ?, ?, ?)", pair_rows_for_db)
        uniqueness.executemany(
            "INSERT INTO task_lookup VALUES (?)",
            ((row["target_endpoint_task_id"],) for row in pq.read_table(task_path).to_pylist()),
        )
        uniqueness.commit()
        for endpoint in ENDPOINTS:
            paths = sorted((output_root / "observations" / endpoint.casefold()).glob("part-*.parquet"))
            endpoint_rows = _assert_schema(paths, _OBSERVATION_SCHEMA, f"{endpoint} observations")
            if endpoint_rows != int(counts["endpoints"][endpoint]["observations"]):
                raise AffinityTrainingSurfaceError(f"{endpoint} physical row count mismatch")
            id_batch: list[tuple[Any, ...]] = []
            for batch in _iter_parquet_batches(paths):
                for row in batch.to_pylist():
                    if row["endpoint"] != endpoint or row["label_unit"] != "nM":
                        raise AffinityTrainingSurfaceError("endpoint/unit partition mismatch")
                    relation = str(row["label_relation"])
                    if relation not in SUPPORTED_RELATIONS or row["label_censoring"] != _censoring(relation):
                        raise AffinityTrainingSurfaceError("relation/censoring mismatch")
                    value = row["label_value_nM"]
                    lower = row["label_lower_bound_nM"]
                    upper = row["label_upper_bound_nM"]
                    if relation == "=" and not (
                        value is not None and value > 0 and lower == value and upper == value
                    ):
                        raise AffinityTrainingSurfaceError("exact label bounds mismatch")
                    if relation in {"<", "<="} and not (upper is not None and upper > 0 and lower is None):
                        raise AffinityTrainingSurfaceError("left-censored bounds mismatch")
                    if relation in {">", ">="} and not (lower is not None and lower > 0 and upper is None):
                        raise AffinityTrainingSurfaceError("right-censored bounds mismatch")
                    if relation == "interval" and not (
                        value is None and lower is not None and upper is not None and 0 < lower < upper
                    ):
                        raise AffinityTrainingSurfaceError("interval bounds mismatch")
                    target = target_lookup.get(str(row["target_id"]))
                    if target is None:
                        raise AffinityTrainingSurfaceError("observation target is absent from registry")
                    if (
                        row["target_leakage_group_id"] != target["target_leakage_group_id"]
                        or row["target_cold_split"] != target["target_cold_split"]
                        or row["sequence_model_eligible"] != target["sequence_model_eligible"]
                    ):
                        raise AffinityTrainingSurfaceError("observation target join mismatch")
                    expected_ligand_split = _split(str(row["scaffold_group_id"]), dimension="ligand_scaffold")
                    expected_target_split = _split(
                        str(row["target_leakage_group_id"]), dimension="target_component"
                    )
                    expected_double = (
                        expected_ligand_split if expected_ligand_split == expected_target_split else None
                    )
                    if (
                        row["ligand_cold_split"] != expected_ligand_split
                        or row["target_cold_split"] != expected_target_split
                        or row["double_cold_split"] != expected_double
                        or row["double_cold_eligible"] != (expected_double is not None)
                    ):
                        raise AffinityTrainingSurfaceError("identifier-only split contract failed")
                    pair_id = _digest(row["structure_id"], row["target_leakage_group_id"], prefix="PAIR")
                    task_id = _digest(endpoint, row["target_leakage_group_id"], prefix="TASK")
                    if (
                        row["ligand_target_pair_id"] != pair_id
                        or row["target_endpoint_task_id"] != task_id
                        or row["measurement_lineage_group_id"]
                        != _lineage_id(row, str(row["target_leakage_group_id"]))
                        or not row["scientific_training_eligible"]
                    ):
                        raise AffinityTrainingSurfaceError("observation identity/eligibility failed")
                    source = str(row["source_dataset"])
                    if source == "BindingDB_202608" and row["source_category"] == "ChEMBL":
                        raise AffinityTrainingSurfaceError("explicit ChEMBL BindingDB mirror was retained")
                    observation_counter[(endpoint, "observations", "all")] += 1
                    observation_counter[(endpoint, "source", source)] += 1
                    observation_counter[(endpoint, "censoring", row["label_censoring"])] += 1
                    observation_counter[
                        (endpoint, "sequence_model_eligible", str(row["sequence_model_eligible"]))
                    ] += 1
                    split_counter[(endpoint, "ligand_cold", expected_ligand_split)] += 1
                    split_counter[(endpoint, "target_cold", expected_target_split)] += 1
                    split_counter[(endpoint, "double_cold", expected_double or "mixed_ineligible")] += 1
                    id_batch.append(
                        (
                            str(row["observation_id"]),
                            str(row["structure_id"]),
                            str(row["scaffold_group_id"]),
                            str(row["ligand_target_pair_id"]),
                            str(row["target_endpoint_task_id"]),
                            str(row["target_leakage_group_id"]),
                            str(row["ligand_cold_split"]),
                            str(row["target_cold_split"]),
                            row["double_cold_split"],
                        )
                    )
                    observed_rows += 1
                    if len(id_batch) >= 50_000:
                        uniqueness.executemany(
                            "INSERT OR IGNORE INTO ids VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                            id_batch,
                        )
                        uniqueness.commit()
                        id_batch.clear()
            if id_batch:
                uniqueness.executemany(
                    "INSERT OR IGNORE INTO ids VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", id_batch
                )
                uniqueness.commit()
                id_batch.clear()
        unique_observations = int(uniqueness.execute("SELECT COUNT(*) FROM ids").fetchone()[0])
        missing_ligand_or_mismatch = int(
            uniqueness.execute(
                """SELECT COUNT(*) FROM ids AS o
                   LEFT JOIN ligand_lookup AS l ON l.structure_id = o.structure_id
                   WHERE l.structure_id IS NULL
                      OR l.scaffold_group_id != o.scaffold_group_id
                      OR l.ligand_split != o.ligand_split"""
            ).fetchone()[0]
        )
        missing_pair_or_mismatch = int(
            uniqueness.execute(
                """SELECT COUNT(*) FROM ids AS o
                   LEFT JOIN pair_lookup AS p ON p.pair_id = o.pair_id
                   WHERE p.pair_id IS NULL
                      OR p.structure_id != o.structure_id
                      OR p.scaffold_group_id != o.scaffold_group_id
                      OR p.target_group_id != o.target_group_id
                      OR p.ligand_split != o.ligand_split
                      OR p.target_split != o.target_split
                      OR COALESCE(p.double_split, '') != COALESCE(o.double_split, '')"""
            ).fetchone()[0]
        )
        missing_task = int(
            uniqueness.execute(
                """SELECT COUNT(*) FROM ids AS o
                   LEFT JOIN task_lookup AS t ON t.task_id = o.task_id
                   WHERE t.task_id IS NULL"""
            ).fetchone()[0]
        )
        uniqueness.close()
    if (
        observed_rows != int(total["observations"])
        or unique_observations != observed_rows
        or missing_ligand_or_mismatch
        or missing_pair_or_mismatch
        or missing_task
    ):
        raise AffinityTrainingSurfaceError("observation total or uniqueness replay failed")

    for endpoint in ENDPOINTS:
        expected = counts["endpoints"][endpoint]
        if observation_counter[(endpoint, "observations", "all")] != int(expected["observations"]):
            raise AffinityTrainingSurfaceError(f"{endpoint} observation replay failed")
        for source, value in expected["source_counts"].items():
            if observation_counter[(endpoint, "source", source)] != int(value):
                raise AffinityTrainingSurfaceError(f"{endpoint} source count replay failed")
        for censoring, value in expected["censoring_counts"].items():
            if observation_counter[(endpoint, "censoring", censoring)] != int(value):
                raise AffinityTrainingSurfaceError(f"{endpoint} censor count replay failed")
        for dimension, categories in expected["split_counts"].items():
            for category, value in categories.items():
                if split_counter[(endpoint, dimension, category)] != int(value):
                    raise AffinityTrainingSurfaceError(f"{endpoint} split count replay failed")

    tasks = pq.read_table(task_path).to_pylist()
    if len(tasks) != int(total["target_endpoint_tasks"]):
        raise AffinityTrainingSurfaceError("task count replay failed")
    if len({row["target_endpoint_task_id"] for row in tasks}) != len(tasks):
        raise AffinityTrainingSurfaceError("duplicate target endpoint task")
    for endpoint in ENDPOINTS:
        endpoint_tasks = [row for row in tasks if row["endpoint"] == endpoint]
        if len(endpoint_tasks) != int(counts["endpoints"][endpoint]["target_endpoint_tasks"]):
            raise AffinityTrainingSurfaceError(f"{endpoint} task registry count mismatch")
        if sum(row["observation_count"] for row in endpoint_tasks) != int(
            counts["endpoints"][endpoint]["observations"]
        ):
            raise AffinityTrainingSurfaceError(f"{endpoint} task observation sum mismatch")
        if sum(row["unique_structure_count"] for row in endpoint_tasks) != int(
            counts["endpoints"][endpoint]["unique_ligand_target_pairs"]
        ):
            raise AffinityTrainingSurfaceError(f"{endpoint} endpoint-pair sum mismatch")
        for row in endpoint_tasks:
            if (
                row["exact_count"]
                + row["left_censored_count"]
                + row["right_censored_count"]
                + row["interval_censored_count"]
                != row["observation_count"]
            ):
                raise AffinityTrainingSurfaceError("task censor counts do not sum")

    exclusion_rows = pq.read_table(exclusion_path).to_pylist()
    exclusion_totals: Counter[str] = Counter()
    for row in exclusion_rows:
        exclusion_totals[row["exclusion_reason"]] += int(row["record_count"])
    if dict(sorted(exclusion_totals.items())) != counts["exclusions"]:
        raise AffinityTrainingSurfaceError("exclusion summary replay failed")
    contract = json.loads((output_root / CONTRACT_NAME).read_text(encoding="utf-8"))
    if contract["endpoint_contract"]["pooling_across_endpoints_allowed"]:
        raise AffinityTrainingSurfaceError("endpoint pooling was enabled")
    if contract["label_contract"]["censored_values_promoted_to_points"]:
        raise AffinityTrainingSurfaceError("censored values were promoted")
    if any(
        contract["execution_boundary"][key]
        for key in (
            "production_features_generated",
            "model_fitting_performed",
            "hyperparameter_search_performed",
            "hpc_executed",
            "predictive_superiority_established",
        )
    ):
        raise AffinityTrainingSurfaceError("execution boundary violated")
    if any(
        manifest["status"][key]
        for key in (
            "binding_free_energy_computed",
            "production_features_generated",
            "model_fitting_performed",
            "hpc_executed",
            "figures_generated",
            "presentation_tables_generated",
        )
    ):
        raise AffinityTrainingSurfaceError("manifest claim boundary violated")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-chembl-root", type=Path)
    parser.add_argument("--bindingdb-archive", type=Path)
    parser.add_argument("--acquisition-manifest", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report-mirror-path", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        validate_affinity_training_surfaces(args.output_root)
    else:
        required = {
            "--canonical-chembl-root": args.canonical_chembl_root,
            "--bindingdb-archive": args.bindingdb_archive,
            "--acquisition-manifest": args.acquisition_manifest,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            parser.error(f"required when building: {', '.join(missing)}")
        build_affinity_training_surfaces(
            canonical_chembl_root=args.canonical_chembl_root,
            bindingdb_archive=args.bindingdb_archive,
            acquisition_manifest=args.acquisition_manifest,
            output_root=args.output_root,
            report_mirror_path=args.report_mirror_path,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
