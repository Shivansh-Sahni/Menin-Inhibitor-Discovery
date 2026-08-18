"""Deterministic, label-blind deep leakage and task-selection audit.

The audit consumes the accepted canonical task inventory and the official
feature-only split suite.  It may inspect test *features* for leakage auditing,
but it never requests a canonical label column and never opens a model-ready
test lockbox.  Evidence scopes are deliberately strict:

* ``exhaustive`` means every distinct item or cross-partition pair in the
  declared population was evaluated;
* ``sampled`` means a fixed, deterministic subset was evaluated; and
* ``not_run`` records an explicit feasibility blocker.

This module does not alter official splits, train models, or make performance
claims.  It publishes a supplemental, inventory-bound audit directory.
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
import stat
import tempfile
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pyarrow.parquet as pq

from .features import RDKIT_AVAILABLE, nearest_neighbor_tanimoto, scaffold_key
from .platform_corpus_readiness import _task_parts, _verify_canonical_corpus
from .platform_data_schema import clean_text
from .platform_data_sources import sha256_file
from .platform_features import PROTEIN_ALPHABET, normalize_protein_sequence, stable_json_digest
from .platform_split_suite import verify_split_suite

SCHEMA_VERSION = "platform_deep_leakage_audit_v1"
MANIFEST_SCHEMA_VERSION = "platform_deep_leakage_manifest_v1"
DEFAULT_OUTPUT_DIRECTORY = Path("research/reports/platform/deep_leakage_analysis")
DEFAULT_CANONICAL_ROOT = Path("research/data/platform/canonical/full_chembl37")
DEFAULT_QC_REPORT = Path("research/reports/platform/qc_report.json")
DEFAULT_SPLIT_ROOT = Path("research/data/platform/splits/full_chembl37")
DEFAULT_CORPUS_ACCEPTANCE = Path("research/models/platform/corpus_readiness/full_chembl37/acceptance.json")
DEFAULT_MODEL_REGISTRY = Path("research/models/platform/model_candidate_registry.json")
PARTITIONS = ("train", "validation", "test")
ROUTES = (*PARTITIONS, "excluded_mixed")
_KMER_SYMBOLS = tuple(sorted(PROTEIN_ALPHABET))
_KMER_SYMBOL_INDEX = {symbol: index for index, symbol in enumerate(_KMER_SYMBOLS)}
FORBIDDEN_LABEL_COLUMNS = frozenset(
    {
        "label_kind",
        "label_value",
        "label_text",
        "label_relation",
        "label_lower_bound",
        "label_upper_bound",
        "label_unit",
        "threshold_source_value_nm",
    }
)
CONTEXT_COLUMNS = (
    "observation_id",
    "assay_id",
    "document_id",
    "document_year",
)
FEATURE_COLUMNS = (
    "record_id",
    "molecule_id",
    "protein_id",
    "target_id",
    "source_id",
    "smiles",
    "sequence",
)
SPLIT_COLUMNS = (
    "record_id",
    "split",
    "molecule_id",
    "protein_id",
    "task_id",
    "source_id",
)
REPORT_FILES = frozenset({"report.json", "task_decision_matrix.csv", "summary.md"})
EvidenceScope = Literal["exhaustive", "sampled", "not_run", "not_applicable"]


@dataclass(frozen=True)
class DeepLeakageConfig:
    """Frozen CPU audit settings; changing any value changes the report hash."""

    seed: int = 20260805
    analysis_date: str = "2026-08-05"
    fingerprint_bits: int = 2048
    fingerprint_radius: int = 2
    fingerprint_tanimoto_threshold: float = 0.80
    maximum_exhaustive_chemical_cross_pairs: int = 30_000_000
    protein_kmer_size: int = 3
    protein_jaccard_threshold: float = 0.80
    maximum_exhaustive_protein_cross_pairs: int = 5_000_000
    maximum_sampled_proteins_per_partition: int = 256
    parquet_batch_size: int = 50_000

    def validate(self) -> None:
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", self.analysis_date):
            raise ValueError("analysis_date must use YYYY-MM-DD")
        if self.fingerprint_bits < 128 or self.fingerprint_bits & (self.fingerprint_bits - 1):
            raise ValueError("fingerprint_bits must be a power of two >=128")
        if not 1 <= self.fingerprint_radius <= 8:
            raise ValueError("fingerprint_radius must be between 1 and 8")
        if not 0 < self.fingerprint_tanimoto_threshold <= 1:
            raise ValueError("fingerprint_tanimoto_threshold must be in (0, 1]")
        if self.maximum_exhaustive_chemical_cross_pairs < 1:
            raise ValueError("chemical pair budget must be positive")
        if not 1 <= self.protein_kmer_size <= 12:
            raise ValueError("protein_kmer_size must be between 1 and 12")
        if not 0 < self.protein_jaccard_threshold <= 1:
            raise ValueError("protein_jaccard_threshold must be in (0, 1]")
        if self.maximum_exhaustive_protein_cross_pairs < 1:
            raise ValueError("protein pair budget must be positive")
        if self.maximum_sampled_proteins_per_partition < 1:
            raise ValueError("protein sample cap must be positive")
        if not 1 <= self.parquet_batch_size <= 250_000:
            raise ValueError("parquet_batch_size must be between 1 and 250000")


def _strict_json(path: Path) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key in {path}: {key}")
            result[key] = value
        return result

    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _safe_relative(value: object, field: str) -> str:
    text = clean_text(value)
    path = Path(text)
    if not text or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe {field}: {text!r}")
    return path.as_posix()


def _guard_columns(columns: Iterable[str], *, source: str) -> tuple[str, ...]:
    normalized = tuple(str(column) for column in columns)
    forbidden = {column.casefold() for column in normalized} & FORBIDDEN_LABEL_COLUMNS
    if forbidden or any(column.casefold().startswith("label") for column in normalized):
        raise ValueError(f"Label columns are forbidden in {source}: {sorted(forbidden)}")
    return normalized


def _iter_batches(path: Path, columns: Sequence[str], batch_size: int):
    requested = _guard_columns(columns, source=path.as_posix())
    parquet = pq.ParquetFile(path)
    missing = set(requested) - set(parquet.schema_arrow.names)
    if missing:
        raise ValueError(f"Required feature columns are missing from {path}: {sorted(missing)}")
    yield from parquet.iter_batches(batch_size=batch_size, columns=list(requested))


def _task_slug_to_record(split_acceptance: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    tasks = split_acceptance.get("tasks")
    if not isinstance(tasks, list):
        raise ValueError("Split acceptance lacks tasks")
    result: dict[str, dict[str, Any]] = {}
    for raw in tasks:
        if not isinstance(raw, dict):
            raise ValueError("Malformed split task")
        slug = _safe_relative(raw.get("task_slug"), "task_slug")
        if "/" in slug or slug in result:
            raise ValueError(f"Invalid or duplicate task slug: {slug}")
        result[slug] = raw
    return result


def _load_partition_map(path: Path, batch_size: int) -> dict[str, str]:
    result: dict[str, str] = {}
    for batch in _iter_batches(path, ("record_id", "split"), batch_size):
        record_ids = batch.column(0).to_pylist()
        partitions = batch.column(1).to_pylist()
        for record_id, partition in zip(record_ids, partitions, strict=True):
            key = clean_text(record_id)
            value = clean_text(partition)
            if not key or value not in ROUTES or key in result:
                raise ValueError(f"Malformed or duplicate split row in {path}")
            result[key] = value
    return result


def _partition_sets(path: Path, columns: Sequence[str], batch_size: int) -> dict[str, dict[str, set[str]]]:
    requested = ("split", *columns)
    sets: dict[str, dict[str, set[str]]] = {
        partition: {column: set() for column in columns} for partition in ROUTES
    }
    for batch in _iter_batches(path, requested, batch_size):
        values = batch.to_pydict()
        for index, raw_partition in enumerate(values["split"]):
            partition = clean_text(raw_partition)
            if partition not in sets:
                raise ValueError(f"Unknown partition in {path}: {partition!r}")
            for column in columns:
                value = clean_text(values[column][index])
                if value:
                    sets[partition][column].add(value)
    return sets


def _pairwise_overlap(sets: Mapping[str, set[str]]) -> dict[str, Any]:
    pairs: dict[str, int] = {}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        pairs[f"{left}__{right}"] = len(sets[left] & sets[right])
    return {
        "evidence_scope": "exhaustive",
        "population": "all nonblank distinct values in train/validation/test; excluded_mixed rows are omitted from cross-partition claims",
        "unique_counts": {partition: len(sets[partition]) for partition in PARTITIONS},
        "pairwise_overlap_counts": pairs,
        "any_overlap": any(pairs.values()),
    }


def _canonical_context(
    task_parts: Sequence[Path],
    partition_map: Mapping[str, str],
    batch_size: int,
) -> dict[str, Any]:
    assays: dict[str, set[str]] = {partition: set() for partition in ROUTES}
    documents: dict[str, set[str]] = {partition: set() for partition in ROUTES}
    years: dict[str, list[int]] = {partition: [] for partition in ROUTES}
    row_counts: Counter[str] = Counter()
    seen: set[str] = set()
    missing: Counter[str] = Counter()
    for path in task_parts:
        for batch in _iter_batches(path, CONTEXT_COLUMNS, batch_size):
            columns = batch.to_pydict()
            for index, raw_record_id in enumerate(columns["observation_id"]):
                record_id = clean_text(raw_record_id)
                if record_id in seen:
                    raise ValueError(f"Duplicate canonical observation ID: {record_id}")
                seen.add(record_id)
                partition = partition_map.get(record_id)
                if partition is None:
                    raise ValueError(f"Official split does not contain canonical row: {record_id}")
                row_counts[partition] += 1
                assay = clean_text(columns["assay_id"][index])
                if assay:
                    assays[partition].add(assay)
                else:
                    missing["assay_id"] += 1
                document = clean_text(columns["document_id"][index])
                if document:
                    documents[partition].add(document)
                else:
                    missing["document_id"] += 1
                raw_year = columns["document_year"][index]
                if raw_year is None:
                    missing["document_year"] += 1
                else:
                    try:
                        year = int(raw_year)
                    except (TypeError, ValueError) as exc:
                        raise ValueError(f"Invalid document year for {record_id}") from exc
                    years[partition].append(year)
    if seen != set(partition_map):
        raise ValueError("Canonical context and official split membership differ")
    assay_overlap = _pairwise_overlap({partition: assays[partition] for partition in PARTITIONS})
    document_overlap = _pairwise_overlap({partition: documents[partition] for partition in PARTITIONS})
    year_ranges = {
        partition: {
            "nonmissing_rows": len(years[partition]),
            "minimum": min(years[partition], default=None),
            "maximum": max(years[partition], default=None),
        }
        for partition in PARTITIONS
    }
    train_max = max(years["train"], default=None)
    validation_min = min(years["validation"], default=None)
    validation_max = max(years["validation"], default=None)
    test_min = min(years["test"], default=None)
    chronological = bool(
        train_max is not None
        and validation_min is not None
        and validation_max is not None
        and test_min is not None
        and train_max < validation_min
        and validation_max <= test_min
    )
    all_years = {year for partition_years in years.values() for year in partition_years}
    all_assays = {value for partition_assays in assays.values() for value in partition_assays}
    return {
        "evidence_scope": "exhaustive",
        "columns_read": list(CONTEXT_COLUMNS),
        "label_columns_read": [],
        "rows": len(seen),
        "partition_rows": dict(sorted(row_counts.items())),
        "missing_rows": dict(sorted(missing.items())),
        "assay_overlap": assay_overlap,
        "document_overlap": document_overlap,
        "document_year": {
            "partition_ranges": year_ranges,
            "distinct_years": len(all_years),
            "complete_rows": sum(len(item) for item in years.values()),
            "coverage_fraction": sum(len(item) for item in years.values()) / len(seen) if seen else 0.0,
            "strictly_chronological_current_split": chronological,
            "complete_case_temporal_split_feasible_without_missing_year_policy": len(all_years) >= 3
            and missing["document_year"] == 0,
        },
        "assay_holdout_feasible_from_available_identifier": len(all_assays) >= 3 and missing["assay_id"] == 0,
    }


def _routed_feature_sets(
    feature_path: Path,
    partition_map: Mapping[str, str],
    fields: Sequence[str],
    batch_size: int,
) -> dict[str, dict[str, set[str]]]:
    """Join routed features by record ID without retaining a row-sized feature map."""

    requested = ("record_id", *fields)
    result: dict[str, dict[str, set[str]]] = {
        partition: {field: set() for field in fields} for partition in ROUTES
    }
    seen: set[str] = set()
    for batch in _iter_batches(feature_path, requested, batch_size):
        columns = batch.to_pydict()
        for index, raw_record_id in enumerate(columns["record_id"]):
            record_id = clean_text(raw_record_id)
            if not record_id or record_id in seen:
                raise ValueError(f"Malformed or duplicate feature row in {feature_path}")
            seen.add(record_id)
            partition = partition_map.get(record_id)
            if partition is None:
                raise ValueError(f"Feature row absent from official split: {record_id}")
            for field in fields:
                value = clean_text(columns[field][index])
                if value:
                    result[partition][field].add(value)
    if seen != set(partition_map):
        raise ValueError("Feature projection and official split membership differ")
    return result


def _cross_pair_count(values: Mapping[str, Sequence[str]]) -> int:
    return len(values["train"]) * (len(values["validation"]) + len(values["test"]))


def _chemical_similarity(smiles: Mapping[str, Sequence[str]], config: DeepLeakageConfig) -> dict[str, Any]:
    pair_count = _cross_pair_count(smiles)
    base: dict[str, Any] = {
        "representation": "RDKit Morgan bit fingerprint",
        "fingerprint_bits": config.fingerprint_bits,
        "fingerprint_radius": config.fingerprint_radius,
        "threshold": config.fingerprint_tanimoto_threshold,
        "distinct_structure_counts": {key: len(values) for key, values in smiles.items()},
        "declared_cross_pairs": pair_count,
        "pair_budget": config.maximum_exhaustive_chemical_cross_pairs,
        "exact_standardized_smiles_overlap": _pairwise_overlap(
            {partition: set(values) for partition, values in smiles.items()}
        ),
    }
    if not RDKIT_AVAILABLE:
        return {**base, "evidence_scope": "not_run", "reason": "RDKit unavailable"}
    if not smiles["train"] or not (smiles["validation"] or smiles["test"]):
        return {**base, "evidence_scope": "not_applicable", "reason": "empty comparison population"}
    if pair_count > config.maximum_exhaustive_chemical_cross_pairs:
        return {
            **base,
            "evidence_scope": "not_run",
            "reason": "declared cross-pair population exceeds the frozen CPU pair budget",
            "evaluated_cross_pairs": 0,
        }
    partitions: dict[str, Any] = {}
    for partition in ("validation", "test"):
        query = list(smiles[partition])
        maxima, _, backend = nearest_neighbor_tanimoto(
            query,
            list(smiles["train"]),
            backend="rdkit",
            n_bits=config.fingerprint_bits,
            radius=config.fingerprint_radius,
            chunk_size=256,
        )
        partitions[partition] = {
            "query_structures": len(query),
            "reference_structures": len(smiles["train"]),
            "evaluated_cross_pairs": len(query) * len(smiles["train"]),
            "maximum_nearest_train_similarity": float(np.max(maxima)) if len(maxima) else None,
            "mean_nearest_train_similarity": float(np.mean(maxima)) if len(maxima) else None,
            "queries_at_or_above_threshold": int(np.sum(maxima >= config.fingerprint_tanimoto_threshold)),
            "queries_with_zero_nearest_train_similarity": int(np.sum(maxima == 0.0)),
            "backend": backend,
        }
    evaluated = sum(item["evaluated_cross_pairs"] for item in partitions.values())
    result = {
        **base,
        "evidence_scope": "exhaustive",
        "population": "every distinct nonblank standardized SMILES in validation/test versus every distinct nonblank standardized SMILES in train",
        "evaluated_cross_pairs": evaluated,
        "partitions": partitions,
    }
    _validate_evidence_record(result)
    return result


def _scaffold_overlap(
    smiles: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    scaffolds: dict[str, set[str]] = {partition: set() for partition in PARTITIONS}
    methods: Counter[str] = Counter()
    for partition, structures in smiles.items():
        for structure in structures:
            key, method = scaffold_key(structure)
            scaffolds[partition].add(key)
            methods[method] += 1
    return {
        **_pairwise_overlap(scaffolds),
        "population": "all distinct nonblank standardized SMILES mapped with the recorded scaffold implementation",
        "method_counts": dict(sorted(methods.items())),
        "true_bemis_murcko_only": set(methods) <= {"bemis_murcko", "bemis_murcko_with_exact_acyclic"},
    }


def _kmers(sequence: str, size: int) -> frozenset[str]:
    normalized, invalid_residues = normalize_protein_sequence(sequence)
    if invalid_residues:
        raise ValueError(f"Protein sequence contains invalid residues: {invalid_residues}")
    if len(normalized) < size:
        return frozenset({normalized}) if normalized else frozenset()
    return frozenset(normalized[index : index + size] for index in range(len(normalized) - size + 1))


def _kmer_bitset(sequence: str, size: int) -> int:
    """Encode an exact k-mer set as a collision-free integer bitset."""

    kmers = _kmers(sequence, size)
    base = len(_KMER_SYMBOLS)
    bitset = 0
    for kmer in kmers:
        offset = sum(base**length for length in range(1, len(kmer)))
        encoded = 0
        for symbol in kmer:
            encoded = encoded * base + _KMER_SYMBOL_INDEX[symbol]
        index = offset + encoded
        bitset |= 1 << index
    return bitset


def _stable_sample(values: Sequence[str], cap: int, seed: int, namespace: str) -> list[str]:
    return sorted(
        values,
        key=lambda value: hashlib.sha256(f"{seed}|{namespace}|{value}".encode()).hexdigest(),
    )[:cap]


def _protein_similarity(sequences: Mapping[str, Sequence[str]], config: DeepLeakageConfig) -> dict[str, Any]:
    full_pairs = _cross_pair_count(sequences)
    if not sequences["train"] or not (sequences["validation"] or sequences["test"]):
        return {
            "evidence_scope": "not_applicable",
            "reason": "empty comparison population",
            "distinct_sequence_counts": {key: len(value) for key, value in sequences.items()},
        }
    scope: EvidenceScope = (
        "exhaustive" if full_pairs <= config.maximum_exhaustive_protein_cross_pairs else "sampled"
    )
    selected = {
        partition: (
            list(values)
            if scope == "exhaustive"
            else _stable_sample(
                values,
                config.maximum_sampled_proteins_per_partition,
                config.seed,
                partition,
            )
        )
        for partition, values in sequences.items()
    }
    train_kmers = [_kmer_bitset(value, config.protein_kmer_size) for value in selected["train"]]
    outputs: dict[str, Any] = {}
    evaluated = 0
    for partition in ("validation", "test"):
        maxima: list[float] = []
        for sequence in selected[partition]:
            query = _kmer_bitset(sequence, config.protein_kmer_size)
            best = 0.0
            for reference in train_kmers:
                union = (query | reference).bit_count()
                similarity = (query & reference).bit_count() / union if union else 0.0
                best = max(best, similarity)
            maxima.append(best)
        comparisons = len(selected[partition]) * len(selected["train"])
        evaluated += comparisons
        outputs[partition] = {
            "query_sequences": len(selected[partition]),
            "reference_sequences": len(selected["train"]),
            "evaluated_cross_pairs": comparisons,
            "maximum_nearest_train_jaccard": max(maxima, default=None),
            "mean_nearest_train_jaccard": float(np.mean(maxima)) if maxima else None,
            "queries_at_or_above_threshold": sum(
                value >= config.protein_jaccard_threshold for value in maxima
            ),
        }
    result = {
        "evidence_scope": scope,
        "representation": f"set Jaccard of normalized sequence {config.protein_kmer_size}-mers",
        "threshold": config.protein_jaccard_threshold,
        "distinct_sequence_counts": {key: len(value) for key, value in sequences.items()},
        "declared_cross_pairs": full_pairs,
        "evaluated_cross_pairs": evaluated,
        "pair_budget": config.maximum_exhaustive_protein_cross_pairs,
        "sample_cap_per_partition": (
            None if scope == "exhaustive" else config.maximum_sampled_proteins_per_partition
        ),
        "partitions": outputs,
        "family_proxy_status": "unavailable_no_validated_protein_family_hierarchy_in_split_features",
    }
    _validate_evidence_record(result)
    return result


def _validate_evidence_record(record: Mapping[str, Any]) -> None:
    scope = record.get("evidence_scope")
    if scope not in {"exhaustive", "sampled", "not_run", "not_applicable"}:
        raise ValueError(f"Unknown evidence scope: {scope!r}")
    if scope == "exhaustive" and "declared_cross_pairs" in record:
        if int(record.get("evaluated_cross_pairs", -1)) != int(record["declared_cross_pairs"]):
            raise ValueError("Exhaustive evidence must evaluate every declared cross pair")
        budget = record.get("pair_budget")
        if budget is not None and int(record["declared_cross_pairs"]) > int(budget):
            raise ValueError("Exhaustive evidence exceeds its declared pair budget")
    if scope == "sampled" and int(record.get("evaluated_cross_pairs", 0)) >= int(
        record.get("declared_cross_pairs", -1)
    ):
        raise ValueError("Sampled evidence must be a strict subset of the declared population")


def _registry_overlap_readiness(path: Path) -> dict[str, Any]:
    registry = _strict_json(path)
    candidates = registry.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("Model registry lacks candidates")
    audited: list[dict[str, Any]] = []
    ready = 0
    not_required = 0
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise ValueError("Malformed model candidate")
        identifier = clean_text(candidate.get("pretrained_identifier"))
        cutoff = clean_text(candidate.get("training_cutoff_status"))
        requirements = [clean_text(item) for item in candidate.get("required_preflight", [])]
        from_scratch = identifier == "none_supervised_from_scratch"
        immutable_checkpoint = bool(re.search(r"\b[0-9a-f]{64}\b", identifier.casefold()))
        overlap_required = any("overlap" in item.casefold() for item in requirements)
        executable = not from_scratch and immutable_checkpoint and cutoff not in {"", "unresolved"}
        if from_scratch:
            not_required += 1
        if executable:
            ready += 1
        audited.append(
            {
                "key": clean_text(candidate.get("key")),
                "from_scratch": from_scratch,
                "immutable_checkpoint_hash_present": immutable_checkpoint,
                "training_cutoff_status": cutoff,
                "overlap_preflight_required": overlap_required,
                "overlap_audit_executable_now": executable,
                "disposition": (
                    "not_applicable_no_pretraining_corpus"
                    if from_scratch
                    else "blocked_exact_checkpoint_and_training_corpus_not_frozen"
                ),
            }
        )
    return {
        "evidence_scope": "exhaustive",
        "population": "every candidate in the inventory-bound local model registry",
        "registry_sha256": sha256_file(path),
        "candidate_count": len(audited),
        "pretrained_candidates_with_executable_overlap_audit": ready,
        "candidates_not_requiring_pretrained_overlap": not_required,
        "actual_pretraining_corpus_overlap_measured": False,
        "candidates": audited,
        "claim": "readiness audit only; no pretrained-corpus membership or overlap was measured",
    }


def _task_priority(task: Mapping[str, Any]) -> tuple[int, list[str], str]:
    """Return a transparent pilot score, reasons, and decision label."""

    semantics = task["semantics"]
    rows = int(task["rows"])
    score = 0
    reasons: list[str] = []
    if task["integrated"]:
        score += 15
        reasons.append("integrated fixed-split corpus")
    if semantics["label_kind"] == "continuous_exact":
        score += 20
        reasons.append("exact continuous endpoint; no censoring model required")
    elif semantics["label_kind"] == "categorical":
        score += 10
        reasons.append("classification endpoint has train-only support evidence")
    elif "censored" in semantics["label_kind"]:
        score -= 10
        reasons.append("requires a censoring-aware loss")
    if semantics["evidence_domain"] == "herg" and semantics["assay_family"] == "herg_functional":
        score += 25
        reasons.append("narrow safety-relevant hERG functional domain")
    if semantics["endpoint"] == "IC50":
        score += 8
        reasons.append("endpoint-specific IC50 task")
    if task["task_scope"] == "derived_sensitivity":
        score -= 20
        reasons.append("derived transform is not independent evidence")
    if rows >= 100:
        score += min(15, int(math.log10(rows) * 4))
    else:
        score -= 20
        reasons.append("fewer than 100 observations")
    materialized = set(task["materialized_strategies"])
    if {"molecule_grouped", "scaffold"} <= materialized:
        score += 15
        reasons.append("both molecule and scaffold candidates materialized")
    elif "molecule_grouped" in materialized:
        score += 5
    if task["source_count"] < 3:
        score -= 8
        reasons.append("source holdout impossible")
    if task["protein_count"] < 3:
        score -= 5
        reasons.append("cross-protein claim impossible")
    if task.get("training_class_ratio_max_to_min", 1.0) > 4.0:
        score -= 5
        reasons.append("train-only class imbalance exceeds 4:1")
    decision = "defer"
    if score >= 65:
        decision = "priority_pilot_candidate"
    elif score >= 45:
        decision = "secondary_candidate"
    elif not task["integrated"] or rows < 100:
        decision = "insufficient_support_or_not_integrated"
    return score, reasons, decision


def _write_csv(path: Path, tasks: Sequence[Mapping[str, Any]]) -> None:
    fields = [
        "rank",
        "dataset_key",
        "task_scope",
        "evidence_domain",
        "endpoint",
        "assay_family",
        "label_kind",
        "rows",
        "molecules",
        "proteins",
        "sources",
        "integrated",
        "molecule_split",
        "scaffold_split",
        "pilot_score",
        "decision",
        "limitations",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for task in tasks:
            writer.writerow(
                {
                    "rank": task["rank"],
                    "dataset_key": task["dataset_key"],
                    "task_scope": task["task_scope"],
                    "evidence_domain": task["semantics"]["evidence_domain"],
                    "endpoint": task["semantics"]["endpoint"],
                    "assay_family": task["semantics"]["assay_family"],
                    "label_kind": task["semantics"]["label_kind"],
                    "rows": task["rows"],
                    "molecules": task["molecule_count"],
                    "proteins": task["protein_count"],
                    "sources": task["source_count"],
                    "integrated": str(task["integrated"]).lower(),
                    "molecule_split": str("molecule_grouped" in task["materialized_strategies"]).lower(),
                    "scaffold_split": str("scaffold" in task["materialized_strategies"]).lower(),
                    "pilot_score": task["pilot_score"],
                    "decision": task["decision"],
                    "limitations": "; ".join(task["priority_reasons"]),
                }
            )


def _write_summary(path: Path, report: Mapping[str, Any]) -> None:
    selected = report["first_task_decision"]
    accounting = report["accounting"]
    text = f"""# Deep leakage audit and first-task decision

## Decision

The first defensible **CPU pilot**, not a production or clinical claim, is
`{selected["dataset_key"]}`. It was selected because it is a narrow hERG
functional IC50 task with exact continuous measurements and both molecule- and
scaffold-based split candidates. It contains only {selected["rows"]} rows from
{selected["sources"]} source and {selected["proteins"]} protein, so it cannot
support cross-source, cross-protein, clinical, or prospective claims.

Use the **scaffold split as the primary evaluation**: its supplemental audit
found {selected["scaffold_audit"]["queries_at_or_above_morgan_threshold"]}
validation/test structures at or above the frozen Morgan threshold and
cross-partition scaffold overlap was
`{str(selected["scaffold_audit"]["cross_partition_scaffold_overlap"]).lower()}`.
Keep the molecule-grouped split as sensitivity analysis only: it had
{selected["molecule_grouped_warning"]["queries_at_or_above_morgan_threshold"]}
high-similarity queries and maximum nearest-train similarity
{selected["molecule_grouped_warning"]["maximum_nearest_train_similarity"]}.

The recommended second-stage scale benchmark is the exact binding Kd task. Kd
is closer to a direct equilibrium-affinity quantity than IC50, but its assay
heterogeneity and unavailable validated protein-family hierarchy still require
careful stratification. Derived binding-free-energy views are transformations
of the same measurements, not independent validation data.

## What was actually checked

- {accounting["tasks_audited"]} tasks and {accounting["materialized_strategies_audited"]} materialized strategies were enumerated.
- Exact molecule, protein, target/source identity overlaps were recomputed over every published split row.
- Assay, document, and document-year context were read from canonical Parquet using an explicit label-free projection and joined exhaustively by observation ID.
- {accounting["chemical_audits_exhaustive"]} chemical audits evaluated every declared validation/test-versus-train Morgan-fingerprint pair; {accounting["chemical_audits_not_run"]} larger audits were not run because they exceeded the frozen CPU pair budget.
- Protein sequence identity was exhaustive. K-mer Jaccard evidence is exhaustive only below the pair budget and otherwise is a deterministic sample.
- The complete local model-candidate registry was reviewed for pretrained-overlap audit readiness. No actual pretrained-corpus overlap could be measured because exact checkpoint hashes and training corpora are not frozen.
- No canonical label column, validation label, test label, or model-ready test lockbox was opened. No model was trained and no official split was modified.

## Hard limitations and next decisions

1. A random molecule/scaffold partition is not temporal or prospective. Use a separately frozen temporal split where year completeness permits it.
2. A single ChEMBL source makes source holdout impossible. Admit independently governed external evidence before cross-source claims.
3. Assay and document overlap is reported, not wished away. Add assay/document-grouped sensitivity splits before publication.
4. Fingerprint similarity is representation- and threshold-dependent. The report binds RDKit, radius, bit length, threshold, population, and pair count.
5. A target ID is not a protein-family annotation. A reviewed family hierarchy or sequence-cluster contract is still missing.
6. The task score is a transparent engineering prioritization heuristic, not a learned scientific result.

## Bottom line

This closes the strongest feasible label-blind CPU audit for the accepted split
suite. It improves evidence about leakage and selects a bounded first pilot,
but it does **not** establish performance, clinical safety, scientific claim
readiness, or authorization for substantive training.
"""
    path.write_text(text, encoding="utf-8")


def _transactional_directory(target: Path):
    class Transaction:
        def __enter__(self) -> Path:
            target.parent.mkdir(parents=True, exist_ok=True)
            self.temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.building-", dir=target.parent))
            return self.temporary

        def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
            if exc_type is not None:
                shutil.rmtree(self.temporary, ignore_errors=True)
                return
            if target.exists():
                raise FileExistsError(f"Refusing to replace existing audit directory: {target}")
            os.replace(self.temporary, target)

    return Transaction()


def _inventory(root: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*")):
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode):
            raise ValueError(f"Output contains a symlink: {path}")
        if stat.S_ISREG(mode):
            relative = path.relative_to(root).as_posix()
            result[relative] = {
                "relative_path": relative,
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        elif not stat.S_ISDIR(mode):
            raise ValueError(f"Output contains a special entry: {path}")
    return result


def materialize_deep_leakage_audit(
    canonical_root: str | os.PathLike[str] = DEFAULT_CANONICAL_ROOT,
    qc_report: str | os.PathLike[str] = DEFAULT_QC_REPORT,
    split_root: str | os.PathLike[str] = DEFAULT_SPLIT_ROOT,
    corpus_acceptance_path: str | os.PathLike[str] = DEFAULT_CORPUS_ACCEPTANCE,
    model_registry_path: str | os.PathLike[str] = DEFAULT_MODEL_REGISTRY,
    output_directory: str | os.PathLike[str] = DEFAULT_OUTPUT_DIRECTORY,
    config: DeepLeakageConfig | None = None,
) -> dict[str, Any]:
    """Publish a deterministic supplemental audit without reading labels."""

    config = config or DeepLeakageConfig()
    config.validate()
    canonical = _verify_canonical_corpus(Path(canonical_root), Path(qc_report))
    split_verification = verify_split_suite(split_root)
    split_path = Path(split_root).resolve()
    split_acceptance_path = split_path / "acceptance.json"
    split_acceptance = _strict_json(split_acceptance_path)
    source = split_acceptance.get("source_binding")
    if (
        not isinstance(source, Mapping)
        or source.get("canonical_build_manifest_sha256") != canonical.build_manifest_sha256
    ):
        raise ValueError("Split suite is not bound to the current canonical build")
    if source.get("canonical_component_inventory_sha256") != canonical.component_inventory_sha256:
        raise ValueError("Split suite canonical inventory binding changed")
    corpus_path = Path(corpus_acceptance_path).resolve()
    corpus = _strict_json(corpus_path)
    if corpus.get("substantive_training_started") is not False:
        raise ValueError("Corpus acceptance violates the no-training boundary")
    if corpus.get("source_binding") != source:
        raise ValueError("Corpus and split suite source bindings differ")
    corpus_tasks = {
        clean_text(task.get("dataset_key")): task
        for task in corpus.get("tasks", [])
        if isinstance(task, Mapping)
    }
    split_tasks = _task_slug_to_record(split_acceptance)
    if {task["dataset_key"] for task in split_tasks.values()} != set(canonical.task_datasets):
        raise ValueError("Deep-audit task enumeration differs from canonical tasks")

    task_records: list[dict[str, Any]] = []
    strategy_audits: list[dict[str, Any]] = []
    accounting: Counter[str] = Counter()
    for slug, task in sorted(split_tasks.items(), key=lambda item: item[1]["dataset_key"]):
        dataset_key = clean_text(task["dataset_key"])
        canonical_entry = canonical.task_datasets[dataset_key]
        corpus_task = corpus_tasks.get(dataset_key)
        if not isinstance(corpus_task, Mapping):
            raise ValueError(f"Corpus acceptance lacks task: {dataset_key}")
        preflight = corpus_task.get("preflight")
        if not isinstance(preflight, Mapping):
            raise ValueError(f"Corpus task lacks preflight: {dataset_key}")
        semantics = preflight.get("task_semantics")
        if not isinstance(semantics, Mapping):
            raise ValueError(f"Corpus task lacks semantics: {dataset_key}")
        feature_summary = task["feature_projection"]
        entity_counts = feature_summary["unique_entity_counts"]
        strategies = [item for item in task["strategies"] if item["status"] == "materialized"]
        feature_path = split_path / _safe_relative(task["feature_projection_path"], "feature path")
        per_task_audits: list[dict[str, Any]] = []
        for strategy in strategies:
            strategy_name = clean_text(strategy["strategy"])
            split_file = split_path / _safe_relative(strategy["split_path"], "split path")
            partition_map = _load_partition_map(split_file, config.parquet_batch_size)
            if len(partition_map) != int(task["canonical_declared_rows"]):
                raise ValueError(f"Split row count differs from task declaration: {dataset_key}")
            identity_sets = _partition_sets(
                split_file,
                ("molecule_id", "protein_id", "source_id"),
                config.parquet_batch_size,
            )
            identity = {
                field: _pairwise_overlap(
                    {partition: identity_sets[partition][field] for partition in PARTITIONS}
                )
                for field in ("molecule_id", "protein_id", "source_id")
            }
            context = _canonical_context(
                _task_parts(canonical, canonical_entry), partition_map, config.parquet_batch_size
            )
            chemical: dict[str, Any] = {
                "evidence_scope": "not_applicable",
                "reason": "chemical near-similarity is prioritized for molecule/scaffold strategies",
            }
            scaffolds: dict[str, Any] = {
                "evidence_scope": "not_applicable",
                "reason": "scaffold audit is prioritized for molecule/scaffold strategies",
            }
            proteins: dict[str, Any] = {
                "evidence_scope": "not_applicable",
                "reason": "sequence near-similarity is prioritized for protein/target/double-cold strategies",
            }
            routed_fields = ["target_id"]
            if strategy_name in {"molecule_grouped", "scaffold"}:
                routed_fields.append("smiles")
            if strategy_name in {"protein_holdout", "target_holdout", "double_cold"}:
                routed_fields.append("sequence")
            routed = _routed_feature_sets(
                feature_path,
                partition_map,
                routed_fields,
                config.parquet_batch_size,
            )
            identity["target_id"] = _pairwise_overlap(
                {partition: routed[partition]["target_id"] for partition in PARTITIONS}
            )
            if strategy_name in {"molecule_grouped", "scaffold"}:
                smiles = {partition: sorted(routed[partition]["smiles"]) for partition in PARTITIONS}
                chemical = _chemical_similarity(smiles, config)
                if chemical["evidence_scope"] == "exhaustive":
                    scaffolds = _scaffold_overlap(smiles)
                else:
                    scaffolds = {
                        "evidence_scope": "not_run",
                        "reason": "supplemental scaffold recomputation shares the frozen chemical CPU population gate",
                    }
                accounting[f"chemical_audits_{chemical['evidence_scope']}"] += 1
            if strategy_name in {"protein_holdout", "target_holdout", "double_cold"}:
                sequences = {partition: sorted(routed[partition]["sequence"]) for partition in PARTITIONS}
                exact_sequence = _pairwise_overlap(
                    {partition: set(values) for partition, values in sequences.items()}
                )
                proteins = {
                    "exact_sequence_identity": exact_sequence,
                    "near_sequence_similarity": _protein_similarity(sequences, config),
                }
                accounting[f"protein_audits_{proteins['near_sequence_similarity']['evidence_scope']}"] += 1
            audit = {
                "dataset_key": dataset_key,
                "task_slug": slug,
                "strategy": strategy_name,
                "split_sha256": strategy["split_sha256"],
                "split_rows": len(partition_map),
                "partition_counts": strategy["partition_counts"],
                "exact_identity_overlap": identity,
                "assay_document_time_context": context,
                "chemical_near_similarity": chemical,
                "scaffold_overlap": scaffolds,
                "protein_sequence_similarity": proteins,
                "label_columns_read": [],
                "test_features_inspected": True,
                "test_labels_read": False,
                "substantive_training_started": False,
            }
            per_task_audits.append(audit)
            strategy_audits.append(audit)
            accounting["materialized_strategies_audited"] += 1

        training_support = corpus_task.get("training_label_support", {})
        training_counts = training_support.get("training_label_counts", {})
        positive_counts = [int(value) for value in training_counts.values() if int(value) > 0]
        class_ratio = max(positive_counts) / min(positive_counts) if positive_counts else 1.0
        materialized_names = [clean_text(item["strategy"]) for item in strategies]
        record = {
            "dataset_key": dataset_key,
            "task_scope": clean_text(task["task_scope"]),
            "semantics": {
                "evidence_domain": clean_text(semantics.get("evidence_domain")),
                "endpoint": clean_text(semantics.get("endpoint")),
                "assay_family": clean_text(semantics.get("assay_family")),
                "label_kind": clean_text(semantics.get("label_kind")),
                "observation_kind": clean_text(semantics.get("observation_kind")),
            },
            "rows": int(task["canonical_declared_rows"]),
            "molecule_count": int(entity_counts["molecule"]),
            "protein_count": int(entity_counts["protein"]),
            "target_count": int(entity_counts["target"]),
            "source_count": int(entity_counts["source"]),
            "integrated": bool(corpus_task.get("integration_materialized")),
            "corpus_status": clean_text(corpus_task.get("status")),
            "materialized_strategies": materialized_names,
            "training_label_support_reused_not_reread": training_support,
            "training_class_ratio_max_to_min": class_ratio,
            "strategy_audits": per_task_audits,
        }
        score, reasons, decision = _task_priority(record)
        record.update({"pilot_score": score, "priority_reasons": reasons, "decision": decision})
        task_records.append(record)
        accounting["tasks_audited"] += 1

    task_records.sort(key=lambda row: (-int(row["pilot_score"]), str(row["dataset_key"])))
    for rank, task in enumerate(task_records, start=1):
        task["rank"] = rank
    selected = next(
        (
            task
            for task in task_records
            if task["semantics"]
            == {
                "evidence_domain": "herg",
                "endpoint": "IC50",
                "assay_family": "herg_functional",
                "label_kind": "continuous_exact",
                "observation_kind": "experimental_summary",
            }
        ),
        task_records[0],
    )
    selected_audits = {
        audit["strategy"]: audit
        for audit in selected["strategy_audits"]
        if audit["strategy"] in {"molecule_grouped", "scaffold"}
    }
    scaffold_chemical = selected_audits.get("scaffold", {}).get("chemical_near_similarity", {})
    scaffold_scaffolds = selected_audits.get("scaffold", {}).get("scaffold_overlap", {})
    molecule_chemical = selected_audits.get("molecule_grouped", {}).get("chemical_near_similarity", {})
    scaffold_high_similarity = sum(
        int(item.get("queries_at_or_above_threshold", 0))
        for item in scaffold_chemical.get("partitions", {}).values()
    )
    molecule_high_similarity = sum(
        int(item.get("queries_at_or_above_threshold", 0))
        for item in molecule_chemical.get("partitions", {}).values()
    )
    molecule_max_similarity = max(
        (
            float(item["maximum_nearest_train_similarity"])
            for item in molecule_chemical.get("partitions", {}).values()
            if item.get("maximum_nearest_train_similarity") is not None
        ),
        default=None,
    )
    registry = _registry_overlap_readiness(Path(model_registry_path).resolve())
    config_payload = asdict(config)
    report = {
        "schema_version": SCHEMA_VERSION,
        "analysis_date": config.analysis_date,
        "configuration": config_payload,
        "configuration_sha256": stable_json_digest(config_payload),
        "source_binding": {
            "analyzer_code_sha256": sha256_file(Path(__file__).resolve()),
            "canonical_build_manifest_sha256": canonical.build_manifest_sha256,
            "canonical_component_inventory_sha256": canonical.component_inventory_sha256,
            "canonical_qc_report_sha256": canonical.qc_report_sha256,
            "split_acceptance_sha256": sha256_file(split_acceptance_path),
            "split_component_inventory_sha256": split_acceptance["component_inventory_sha256"],
            "corpus_acceptance_sha256": sha256_file(corpus_path),
            "model_registry_sha256": sha256_file(Path(model_registry_path).resolve()),
        },
        "input_verification": split_verification,
        "evidence_taxonomy": {
            "exhaustive": "all declared distinct items or cross-partition pairs were evaluated",
            "sampled": "a deterministic fixed-seed strict subset was evaluated",
            "not_run": "analysis was feasible in principle but exceeded a frozen CPU/input-readiness gate",
            "not_applicable": "the comparison does not answer the strategy's generalization question",
        },
        "label_access_contract": {
            "canonical_columns_requested": list(CONTEXT_COLUMNS),
            "canonical_label_columns_requested": [],
            "split_feature_columns_requested": list(FEATURE_COLUMNS),
            "test_features_inspected_for_leakage": True,
            "training_labels_newly_read": False,
            "validation_labels_read": False,
            "test_labels_read": False,
            "model_ready_test_lockbox_opened_or_hashed": False,
        },
        "accounting": dict(sorted(accounting.items())),
        "first_task_decision": {
            "dataset_key": selected["dataset_key"],
            "rows": selected["rows"],
            "sources": selected["source_count"],
            "proteins": selected["protein_count"],
            "role": "bounded CPU pilot only",
            "claim_boundary": "not a production, clinical, prospective, cross-source, or cross-protein claim",
            "selection_method": "predeclared transparent heuristic plus explicit narrow-task override",
            "primary_evaluation_strategy": "scaffold",
            "molecule_grouped_role": "sensitivity_analysis_only",
            "scaffold_audit": {
                "cross_partition_scaffold_overlap": scaffold_scaffolds.get("any_overlap"),
                "queries_at_or_above_morgan_threshold": scaffold_high_similarity,
                "threshold": config.fingerprint_tanimoto_threshold,
            },
            "molecule_grouped_warning": {
                "queries_at_or_above_morgan_threshold": molecule_high_similarity,
                "maximum_nearest_train_similarity": molecule_max_similarity,
                "interpretation": "distinct molecule identifiers do not guarantee fingerprint dissimilarity",
            },
        },
        "second_stage_recommendation": {
            "task": "default exact binding Kd",
            "role": "scale and cross-target baseline after assay/document leakage controls",
            "derived_free_energy_warning": "derived free-energy views reuse the same observations and are not independent validation",
        },
        "pretrained_corpus_overlap_readiness": registry,
        "tasks": task_records,
        "strategy_audits": strategy_audits,
        "claim_readiness": False,
        "substantive_training_ready": False,
        "substantive_training_authorized": False,
        "large_model_training_started": False,
        "substantive_training_started": False,
    }
    output = Path(output_directory)
    with _transactional_directory(output) as staging:
        _atomic_json(staging / "report.json", report)
        _write_csv(staging / "task_decision_matrix.csv", task_records)
        _write_summary(staging / "summary.md", report)
        inventory = _inventory(staging)
        manifest = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "configuration_sha256": report["configuration_sha256"],
            "report_sha256": inventory["report.json"]["sha256"],
            "source_binding": report["source_binding"],
            "label_access_contract": report["label_access_contract"],
            "component_inventory": inventory,
            "component_inventory_sha256": stable_json_digest(inventory),
            "claim_readiness": False,
            "substantive_training_ready": False,
            "substantive_training_authorized": False,
            "substantive_training_started": False,
        }
        _atomic_json(staging / "manifest.json", manifest)
    return verify_deep_leakage_audit(output, expected_config=config)


def verify_deep_leakage_audit(
    output_directory: str | os.PathLike[str] = DEFAULT_OUTPUT_DIRECTORY,
    *,
    expected_config: DeepLeakageConfig | None = None,
) -> dict[str, Any]:
    """Verify exact topology, hashes, thresholds, evidence scopes, and boundaries."""

    root = Path(output_directory).resolve()
    manifest_path = root / "manifest.json"
    manifest = _strict_json(manifest_path)
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("Unrecognized deep-leakage manifest schema")
    actual_files: set[str] = set()
    for path in root.rglob("*"):
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode):
            raise ValueError(f"Audit contains symlink: {path}")
        if stat.S_ISREG(mode):
            actual_files.add(path.relative_to(root).as_posix())
        elif not stat.S_ISDIR(mode):
            raise ValueError(f"Audit contains special entry: {path}")
    if actual_files != REPORT_FILES | {"manifest.json"}:
        raise ValueError("Deep-leakage output topology differs from its closed contract")
    inventory = manifest.get("component_inventory")
    if not isinstance(inventory, Mapping) or set(inventory) != REPORT_FILES:
        raise ValueError("Deep-leakage component inventory membership mismatch")
    if stable_json_digest(inventory) != manifest.get("component_inventory_sha256"):
        raise ValueError("Deep-leakage component inventory digest mismatch")
    for relative, raw in inventory.items():
        if not isinstance(raw, Mapping) or set(raw) != {"relative_path", "sha256", "size_bytes"}:
            raise ValueError(f"Malformed inventory entry: {relative}")
        safe = _safe_relative(relative, "inventory path")
        if raw.get("relative_path") != safe:
            raise ValueError(f"Inventory path binding mismatch: {relative}")
        path = root / safe
        if path.stat().st_size != int(raw["size_bytes"]) or sha256_file(path) != raw["sha256"]:
            raise ValueError(f"Inventory component changed: {relative}")
    report = _strict_json(root / "report.json")
    if report.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unrecognized deep-leakage report schema")
    config = report.get("configuration")
    if not isinstance(config, Mapping) or stable_json_digest(config) != report.get("configuration_sha256"):
        raise ValueError("Report configuration digest mismatch")
    parsed_config = DeepLeakageConfig(**config)
    parsed_config.validate()
    if expected_config is not None and asdict(expected_config) != dict(config):
        raise ValueError("Report configuration differs from the expected frozen thresholds")
    if manifest.get("configuration_sha256") != report.get("configuration_sha256"):
        raise ValueError("Manifest/report configuration bindings differ")
    if manifest.get("report_sha256") != sha256_file(root / "report.json"):
        raise ValueError("Manifest report binding differs")
    source_binding = report.get("source_binding")
    if not isinstance(source_binding, Mapping) or manifest.get("source_binding") != source_binding:
        raise ValueError("Manifest/report source bindings differ")
    if source_binding.get("analyzer_code_sha256") != sha256_file(Path(__file__).resolve()):
        raise ValueError("Deep-leakage analyzer code binding changed")
    label_contract = report.get("label_access_contract")
    if not isinstance(label_contract, Mapping) or any(
        label_contract.get(key) is not False
        for key in (
            "training_labels_newly_read",
            "validation_labels_read",
            "test_labels_read",
            "model_ready_test_lockbox_opened_or_hashed",
        )
    ):
        raise ValueError("Deep-leakage label-access boundary changed")
    if label_contract.get("canonical_label_columns_requested") != []:
        raise ValueError("Deep-leakage report claims canonical label access")
    for task in report.get("tasks", []):
        for audit in task.get("strategy_audits", []):
            if audit.get("label_columns_read") != [] or audit.get("test_labels_read") is not False:
                raise ValueError("Strategy audit violates label-access boundary")
            chemical = audit.get("chemical_near_similarity", {})
            if isinstance(chemical, Mapping):
                _validate_evidence_record(chemical)
                if "threshold" in chemical and (
                    float(chemical["threshold"]) != parsed_config.fingerprint_tanimoto_threshold
                    or int(chemical.get("fingerprint_bits", -1)) != parsed_config.fingerprint_bits
                    or int(chemical.get("fingerprint_radius", -1)) != parsed_config.fingerprint_radius
                    or int(chemical.get("pair_budget", -1))
                    != parsed_config.maximum_exhaustive_chemical_cross_pairs
                ):
                    raise ValueError("Chemical audit thresholds drifted from frozen configuration")
            protein = audit.get("protein_sequence_similarity", {})
            if isinstance(protein, Mapping) and isinstance(protein.get("near_sequence_similarity"), Mapping):
                near_protein = protein["near_sequence_similarity"]
                _validate_evidence_record(near_protein)
                if "threshold" in near_protein and (
                    float(near_protein["threshold"]) != parsed_config.protein_jaccard_threshold
                    or int(near_protein.get("pair_budget", -1))
                    != parsed_config.maximum_exhaustive_protein_cross_pairs
                    or near_protein.get("representation")
                    != f"set Jaccard of normalized sequence {parsed_config.protein_kmer_size}-mers"
                ):
                    raise ValueError("Protein audit thresholds drifted from frozen configuration")
    for key in (
        "claim_readiness",
        "substantive_training_ready",
        "substantive_training_authorized",
        "large_model_training_started",
        "substantive_training_started",
    ):
        if report.get(key) is not False:
            raise ValueError(f"Deep-leakage boundary changed: {key}")
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "status": "verified",
        "report_sha256": sha256_file(root / "report.json"),
        "manifest_sha256": sha256_file(manifest_path),
        "component_count": len(inventory),
        "configuration_sha256": report["configuration_sha256"],
        "tasks_audited": report["accounting"]["tasks_audited"],
        "strategies_audited": report["accounting"]["materialized_strategies_audited"],
        "test_labels_read": False,
        "substantive_training_started": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build", help="materialize the deterministic audit")
    build.add_argument("--canonical-root", type=Path, default=DEFAULT_CANONICAL_ROOT)
    build.add_argument("--qc-report", type=Path, default=DEFAULT_QC_REPORT)
    build.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT_ROOT)
    build.add_argument("--corpus-acceptance", type=Path, default=DEFAULT_CORPUS_ACCEPTANCE)
    build.add_argument("--model-registry", type=Path, default=DEFAULT_MODEL_REGISTRY)
    build.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    verify = subparsers.add_parser("verify", help="rehash and validate an existing audit")
    verify.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "build":
        result = materialize_deep_leakage_audit(
            args.canonical_root,
            args.qc_report,
            args.split_root,
            args.corpus_acceptance,
            args.model_registry,
            args.output,
        )
    else:
        result = verify_deep_leakage_audit(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "DeepLeakageConfig",
    "materialize_deep_leakage_audit",
    "verify_deep_leakage_audit",
]
