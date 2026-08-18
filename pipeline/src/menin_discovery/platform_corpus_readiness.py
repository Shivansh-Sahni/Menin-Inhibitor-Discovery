"""Transactional orchestration of every accepted default canonical task.

This module is intentionally an orchestration layer.  It verifies the complete
canonical build and its QC binding, performs one exact fixed-seed split
preflight per task, and then calls the existing task integration and capped
diagnostic APIs.  It never searches for a favorable seed and never starts
substantive training.  Physically routed test lockboxes are transitively bound
from their integration acceptances and are not reopened or rehashed here.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import os
import re
import shutil
import sqlite3
import tempfile
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import pyarrow.parquet as pq

from .platform_data_schema import canonical_json, clean_text
from .platform_data_sources import sha256_file
from .platform_features import stable_json_digest
from .platform_model_integration import (
    DiagnosticConfig,
    TaskIntegrationConfig,
    materialize_capped_diagnostic_bundle,
    materialize_task_integration_bundle,
)
from .platform_pretraining import JsonlIterableDataset

CORPUS_READINESS_SCHEMA_VERSION = "platform_corpus_readiness_bundle_v1"
NO_SUBSTANTIVE_TRAINING = False
PARTITIONS = ("train", "validation", "test")
STRUCTURAL_PREFLIGHT_LABEL_ACCESS: dict[str, Any] = {
    "label_columns_read": [],
    "training_labels_read": False,
    "validation_labels_read": False,
    "test_labels_read": False,
    "policy": "structural preflight never requests label_value or label_text columns",
}


def _label_access_policy(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "structural_preflight_label_columns_read": [],
        "categorical_mapping_and_support_partition": "physically_routed_train_only",
        "validation_labels_read_for_preflight_or_mapping": False,
        "test_labels_read_for_preflight_or_mapping": False,
        "continuous_censored_ordinal_label_histograms_materialized": False,
        "categorical_cardinality_cap": int(config["maximum_categorical_label_cardinality"]),
    }


@dataclass(frozen=True)
class CorpusReadinessConfig:
    """Fixed orchestration, split-preflight, and capped-diagnostic settings."""

    seed: int = 20260804
    preflight_batch_size: int = 50_000
    split_batch_size: int = 50_000
    serialization_batch_size: int = 8_192
    loader_batch_size: int = 8
    loader_maximum_batches: int = 4
    minimum_rows: int = 4
    minimum_unique_molecules: int = 3
    minimum_unique_proteins: int = 1
    diagnostic_maximum_train_examples: int = 10_000
    diagnostic_maximum_validation_examples: int = 2_500
    diagnostic_fingerprint_bits: int = 2_048
    diagnostic_fingerprint_radius: int = 2
    maximum_categorical_label_cardinality: int = 32

    def validate(self) -> None:
        if not 1 <= self.preflight_batch_size <= 250_000:
            raise ValueError("preflight_batch_size must be between 1 and 250000")
        integration = self.integration_config()
        integration.validate()
        if self.minimum_rows < 4:
            raise ValueError("minimum_rows must permit train>=2 plus validation and test")
        if self.minimum_unique_molecules < 3 or self.minimum_unique_proteins < 1:
            raise ValueError("minimum unique-group thresholds are invalid")
        self.diagnostic_config().validate()
        if not 2 <= self.maximum_categorical_label_cardinality <= 256:
            raise ValueError("maximum_categorical_label_cardinality must be between 2 and 256")

    def integration_config(
        self,
        *,
        task_eligibility_mode: Literal["default", "derived_sensitivity"] = "default",
    ) -> TaskIntegrationConfig:
        if task_eligibility_mode not in {"default", "derived_sensitivity"}:
            raise ValueError("Unsupported task eligibility mode")
        return TaskIntegrationConfig(
            split_name="molecule_hash_stream_v1",
            split_strategy="molecule_grouped",
            intended_use="new molecule within the observed public task domain at platform scale",
            seed=self.seed,
            split_batch_size=self.split_batch_size,
            serialization_batch_size=self.serialization_batch_size,
            task_eligibility_mode=task_eligibility_mode,
            loader_batch_size=self.loader_batch_size,
            loader_maximum_batches=self.loader_maximum_batches,
            require_manifest_bound_directory=True,
        )

    def diagnostic_config(
        self,
        *,
        binary_label_mapping: tuple[tuple[str, int], ...] = (),
    ) -> DiagnosticConfig:
        return DiagnosticConfig(
            seed=self.seed,
            maximum_train_examples=self.diagnostic_maximum_train_examples,
            maximum_validation_examples=self.diagnostic_maximum_validation_examples,
            fingerprint_bits=self.diagnostic_fingerprint_bits,
            fingerprint_radius=self.diagnostic_fingerprint_radius,
            binary_label_mapping=binary_label_mapping,
        )


@dataclass(frozen=True)
class CanonicalCorpusBinding:
    root: Path
    build_manifest: dict[str, Any]
    build_manifest_sha256: str
    component_inventory_sha256: str
    qc_report_sha256: str
    task_datasets_path: Path
    task_datasets_sha256: str
    task_datasets: dict[str, Any]


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    if payload.get("substantive_training_started") is not NO_SUBSTANTIVE_TRAINING:
        raise ValueError("Every corpus-readiness JSON must state substantive_training_started=false")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _safe_relative(value: object, field: str) -> str:
    text = clean_text(value)
    path = Path(text)
    if not text or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe or blank {field}: {text!r}")
    return path.as_posix()


def _hex_digest(value: object, field: str) -> str:
    text = clean_text(value).casefold()
    if not re.fullmatch(r"[0-9a-f]{64}", text):
        raise ValueError(f"{field} must be a SHA-256 digest")
    return text


def _verify_canonical_corpus(
    canonical_build_root: Path,
    qc_report_path: Path,
) -> CanonicalCorpusBinding:
    root = canonical_build_root.resolve()
    if root.name.startswith(".") or root.name.endswith(".building"):
        raise ValueError("Refusing a provisional canonical build")
    sibling = root.parent / f".{root.name}.building"
    if sibling.exists():
        raise ValueError(f"Refusing canonical data while provisional sibling exists: {sibling}")
    manifest_path = root / "build_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("Canonical build manifest must be an object")
    inventory = manifest.get("component_inventory")
    if not isinstance(inventory, list) or not inventory:
        raise ValueError("Canonical build manifest lacks a component inventory")
    declared: dict[str, Mapping[str, Any]] = {}
    normalized_inventory: list[dict[str, Any]] = []
    for index, entry in enumerate(inventory):
        if not isinstance(entry, Mapping):
            raise ValueError(f"Invalid component inventory entry {index}")
        relative = _safe_relative(entry.get("path"), f"component_inventory[{index}].path")
        if relative in declared:
            raise ValueError(f"Duplicate canonical component path: {relative}")
        declared[relative] = entry
        normalized_inventory.append(dict(entry))
    actual: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Canonical build contains a symlink: {path}")
        if path.is_file() and path != manifest_path:
            actual.add(path.relative_to(root).as_posix())
    if actual != set(declared):
        raise ValueError(
            "Canonical component membership mismatch; "
            f"unbound={sorted(actual - set(declared))}, missing={sorted(set(declared) - actual)}"
        )
    for relative, entry in sorted(declared.items()):
        path = root / relative
        if path.stat().st_size != int(entry.get("size_bytes", -1)):
            raise ValueError(f"Canonical component size mismatch: {relative}")
        if sha256_file(path) != _hex_digest(entry.get("sha256"), f"{relative}.sha256"):
            raise ValueError(f"Canonical component SHA-256 mismatch: {relative}")
        if path.suffix.casefold() == ".parquet" and "rows" in entry:
            metadata = pq.ParquetFile(path).metadata
            if metadata is None or int(metadata.num_rows) != int(entry["rows"]):
                raise ValueError(f"Canonical component Parquet row mismatch: {relative}")
    build_sha = sha256_file(manifest_path)
    qc_path = qc_report_path.resolve()
    if not qc_path.is_file():
        raise FileNotFoundError(qc_path)
    qc = json.loads(qc_path.read_text(encoding="utf-8"))
    if not isinstance(qc, dict) or qc.get("qc_passed") is not True:
        raise ValueError("Canonical QC report is not accepted")
    if clean_text(qc.get("build_manifest_sha256")) != build_sha:
        raise ValueError("Canonical QC report does not bind the current build manifest")
    task_path = root / "task_datasets.json"
    if "task_datasets.json" not in declared or not task_path.is_file():
        raise ValueError("Canonical component inventory does not bind task_datasets.json")
    task_document = json.loads(task_path.read_text(encoding="utf-8"))
    if not isinstance(task_document, dict) or not task_document:
        raise ValueError("task_datasets.json must be a non-empty object")
    task_payload_digest = hashlib.sha256(canonical_json(task_document).encode("utf-8")).hexdigest()
    if task_payload_digest != clean_text(manifest.get("task_datasets_manifest_sha256")):
        raise ValueError("task_datasets.json payload digest does not match build manifest")
    _validate_task_datasets(root, manifest, task_document, declared)
    return CanonicalCorpusBinding(
        root=root,
        build_manifest=manifest,
        build_manifest_sha256=build_sha,
        component_inventory_sha256=hashlib.sha256(
            canonical_json(sorted(normalized_inventory, key=lambda row: clean_text(row["path"]))).encode(
                "utf-8"
            )
        ).hexdigest(),
        qc_report_sha256=sha256_file(qc_path),
        task_datasets_path=task_path,
        task_datasets_sha256=sha256_file(task_path),
        task_datasets=task_document,
    )


def _validate_task_datasets(
    root: Path,
    manifest: Mapping[str, Any],
    task_datasets: Mapping[str, Any],
    components: Mapping[str, Mapping[str, Any]],
) -> None:
    shard_records = manifest.get("shard_artifacts")
    if not isinstance(shard_records, list):
        raise ValueError("Canonical build manifest lacks shard_artifacts")
    shards = {
        clean_text(record.get("relative_path")): record
        for record in shard_records
        if isinstance(record, Mapping) and clean_text(record.get("relative_path"))
    }
    seen_parts: set[str] = set()
    for key, entry in sorted(task_datasets.items()):
        if not isinstance(entry, Mapping):
            raise ValueError(f"Invalid task dataset entry: {key}")
        scope = clean_text(entry.get("task_scope"))
        task_type = clean_text(entry.get("task_type"))
        if key != f"{scope}::{task_type}" or scope not in {"default", "derived_sensitivity"}:
            raise ValueError(f"Task dataset key/semantics mismatch: {key}")
        parts = entry.get("parts")
        if not isinstance(parts, list) or not parts:
            raise ValueError(f"Task dataset has no parts: {key}")
        digest_payload: list[dict[str, Any]] = []
        parents: set[str] = set()
        row_count = 0
        for index, part in enumerate(parts):
            if not isinstance(part, Mapping):
                raise ValueError(f"Invalid task part {key}[{index}]")
            relative = _safe_relative(
                part.get("path", part.get("relative_path")),
                f"{key}.parts[{index}].path",
            )
            if relative in seen_parts:
                raise ValueError(f"Task part belongs to multiple datasets: {relative}")
            seen_parts.add(relative)
            parents.add(Path(relative).parent.as_posix())
            component = components.get(relative)
            shard = shards.get(relative)
            if component is None or shard is None or not (root / relative).is_file():
                raise ValueError(f"Task part is not bound by canonical inventories: {relative}")
            rows = int(part.get("rows", -1))
            digest = _hex_digest(part.get("sha256"), f"{relative}.sha256")
            schema_digest = _hex_digest(part.get("arrow_schema_sha256"), f"{relative}.arrow_schema_sha256")
            if (
                rows < 0
                or rows != int(component.get("rows", -1))
                or rows != int(shard.get("rows", -1))
                or digest != clean_text(component.get("sha256"))
                or digest != clean_text(shard.get("sha256"))
            ):
                raise ValueError(f"Task part bindings disagree: {relative}")
            row_count += rows
            digest_payload.append(
                {
                    "path": relative,
                    "rows": rows,
                    "sha256": digest,
                    "arrow_schema_sha256": schema_digest,
                }
            )
        if len(parents) != 1:
            raise ValueError(f"Task dataset spans multiple directories: {key}")
        if row_count != int(entry.get("row_count", -1)) or len(parts) != int(entry.get("part_count", -1)):
            raise ValueError(f"Task dataset counts do not reconcile: {key}")
        expected_digest = hashlib.sha256(canonical_json(digest_payload).encode("utf-8")).hexdigest()
        if expected_digest != clean_text(entry.get("dataset_sha256")):
            raise ValueError(f"Task dataset aggregate digest mismatch: {key}")


def _partition(group_id: str, seed: int) -> str:
    digest = hashlib.sha256(f"{seed}|{group_id}".encode()).digest()
    unit = int.from_bytes(digest[:8], "big") / 2**64
    if unit < 0.70:
        return "train"
    if unit < 0.85:
        return "validation"
    return "test"


class _CappedLabelSample:
    def __init__(self, caps: Mapping[str, int], seed: int) -> None:
        self.caps = dict(caps)
        self.seed = seed
        self.heaps: dict[str, list[tuple[int, str, str]]] = {partition: [] for partition in caps}

    def add(self, partition: str, record_id: str, label: str) -> None:
        if partition not in self.heaps:
            return
        rank = int.from_bytes(hashlib.sha256(f"{self.seed}|{record_id}".encode()).digest(), "big")
        item = (-rank, record_id, label)
        heap = self.heaps[partition]
        cap = self.caps[partition]
        if len(heap) < cap:
            heapq.heappush(heap, item)
        elif item > heap[0]:
            heapq.heapreplace(heap, item)

    def label_counts(self, partition: str) -> dict[str, int]:
        return dict(sorted(Counter(item[2] for item in self.heaps[partition]).items()))


def _json_label_key(example: Mapping[str, Any]) -> str:
    label = example.get("label")
    if not isinstance(label, Mapping):
        raise ValueError("Model-ready training example lacks a label object")
    value = label.get("value")
    if value is not None:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return clean_text(value)
        if math.isfinite(numeric) and numeric.is_integer():
            return str(int(numeric))
        return repr(numeric)
    return clean_text(label.get("text"))


def _task_parts(binding: CanonicalCorpusBinding, entry: Mapping[str, Any]) -> list[Path]:
    return [
        binding.root / _safe_relative(part.get("path", part.get("relative_path")), "task part")
        for part in entry["parts"]
    ]


def _preflight_task(
    binding: CanonicalCorpusBinding,
    dataset_key: str,
    entry: Mapping[str, Any],
    config: CorpusReadinessConfig,
) -> dict[str, Any]:
    columns = [
        "observation_id",
        "molecule_id",
        "protein_id",
        "task_id",
        "task_type",
        "label_kind",
        "observation_kind",
        "evidence_domain",
        "endpoint",
        "assay_family",
    ]
    semantics_columns = (
        "task_id",
        "task_type",
        "label_kind",
        "observation_kind",
        "evidence_domain",
        "endpoint",
        "assay_family",
    )
    semantics: dict[str, set[str]] = {column: set() for column in semantics_columns}
    row_partitions: Counter[str] = Counter()
    rows_seen = 0
    with tempfile.TemporaryDirectory(prefix="corpus-readiness-preflight-") as temporary:
        connection = sqlite3.connect(str(Path(temporary) / "identities.sqlite"))
        connection.executescript(
            """
            CREATE TABLE records (record_id TEXT PRIMARY KEY) WITHOUT ROWID;
            CREATE TABLE molecules (value TEXT PRIMARY KEY, rows INTEGER NOT NULL) WITHOUT ROWID;
            CREATE TABLE proteins (value TEXT PRIMARY KEY, rows INTEGER NOT NULL) WITHOUT ROWID;
            """
        )
        try:
            for path in _task_parts(binding, entry):
                parquet = pq.ParquetFile(path)
                missing = sorted(set(columns) - set(parquet.schema_arrow.names))
                if missing:
                    raise ValueError(f"Task preflight columns missing from {path.name}: {missing}")
                for batch in parquet.iter_batches(
                    batch_size=config.preflight_batch_size,
                    columns=columns,
                ):
                    frame = batch.to_pandas()
                    if frame.empty:
                        continue
                    rows_seen += len(frame)
                    for column in semantics_columns:
                        values = set(frame[column].fillna("").astype(str).str.strip())
                        semantics[column].update(values)
                        if "" in semantics[column] or len(semantics[column]) > 1:
                            raise ValueError(f"Task {dataset_key} is blank or heterogeneous in {column}")
                    record_ids = frame["observation_id"].fillna("").astype(str).str.strip()
                    molecules = frame["molecule_id"].fillna("").astype(str).str.strip()
                    proteins = frame["protein_id"].fillna("").astype(str).str.strip()
                    if record_ids.eq("").any() or molecules.eq("").any() or proteins.eq("").any():
                        raise ValueError(f"Task {dataset_key} has blank split identities")
                    try:
                        connection.executemany(
                            "INSERT INTO records(record_id) VALUES (?)",
                            [(value,) for value in record_ids],
                        )
                    except sqlite3.IntegrityError as error:
                        raise ValueError(f"Task {dataset_key} has duplicate observation IDs") from error
                    for table, values in (("molecules", molecules), ("proteins", proteins)):
                        counts = values.value_counts()
                        connection.executemany(
                            f"""
                            INSERT INTO {table}(value, rows) VALUES (?, ?)
                            ON CONFLICT(value) DO UPDATE SET rows=rows+excluded.rows
                            """,
                            [(str(value), int(count)) for value, count in counts.items()],
                        )
                    for molecule in molecules:
                        partition = _partition(f"molecule:{molecule}", config.seed)
                        row_partitions[partition] += 1
                    connection.commit()
            unique_molecules = int(connection.execute("SELECT COUNT(*) FROM molecules").fetchone()[0])
            unique_proteins = int(connection.execute("SELECT COUNT(*) FROM proteins").fetchone()[0])
            group_partitions: Counter[str] = Counter()
            for (molecule,) in connection.execute("SELECT value FROM molecules ORDER BY value"):
                group_partitions[_partition(f"molecule:{molecule}", config.seed)] += 1
        finally:
            connection.close()
    declared_rows = int(entry.get("row_count", -1))
    if rows_seen != declared_rows:
        raise ValueError(f"Task preflight row mismatch for {dataset_key}: {rows_seen} != {declared_rows}")
    resolved_semantics = {column: next(iter(values)) for column, values in semantics.items()}
    reasons: list[str] = []
    if rows_seen < config.minimum_rows:
        reasons.append("rows_below_configured_minimum")
    if unique_molecules < config.minimum_unique_molecules:
        reasons.append("unique_molecule_groups_below_configured_minimum")
    if unique_proteins < config.minimum_unique_proteins:
        reasons.append("unique_protein_groups_below_configured_minimum")
    for partition in PARTITIONS:
        if row_partitions[partition] < (2 if partition == "train" else 1):
            reasons.append(f"predicted_{partition}_rows_insufficient")
        if group_partitions[partition] < 1:
            reasons.append(f"predicted_{partition}_molecule_groups_empty")
    label_kind = resolved_semantics["label_kind"]
    if label_kind not in {"categorical", "continuous_exact", "continuous_censored", "ordinal"}:
        reasons.append("unsupported_label_kind")
    return {
        "dataset_key": dataset_key,
        "declared_rows": declared_rows,
        "rows_scanned": rows_seen,
        "unique_observation_ids": rows_seen,
        "unique_molecule_groups": unique_molecules,
        "unique_protein_groups": unique_proteins,
        "predicted_partition_rows": {partition: int(row_partitions[partition]) for partition in PARTITIONS},
        "predicted_partition_molecule_groups": {
            partition: int(group_partitions[partition]) for partition in PARTITIONS
        },
        "task_semantics": resolved_semantics,
        "label_access": dict(STRUCTURAL_PREFLIGHT_LABEL_ACCESS),
        "fixed_split": {
            "strategy": "molecule_grouped",
            "algorithm": "stable_sha256_group_hash_bounded_memory_v1",
            "seed": config.seed,
            "fractions": {"train": 0.70, "validation": 0.15, "test": 0.15},
            "seed_search_performed": False,
        },
        "support_decision": "skip" if reasons else "integrate",
        "skip_reasons": sorted(set(reasons)),
        "substantive_training_started": NO_SUBSTANTIVE_TRAINING,
    }


def _training_categorical_support(
    integration_root: Path,
    integration_acceptance: Mapping[str, Any],
    config: CorpusReadinessConfig,
) -> tuple[dict[str, Any], tuple[tuple[str, int], ...]]:
    """Inspect only the physically routed train lockbox for categorical support."""

    semantics = integration_acceptance.get("task_semantics")
    if not isinstance(semantics, Mapping):
        raise ValueError("Integration acceptance lacks task semantics")
    label_kind = clean_text(semantics.get("label_kind"))
    base = {
        "label_kind": label_kind,
        "selection_seed": config.seed,
        "maximum_categorical_label_cardinality": config.maximum_categorical_label_cardinality,
        "validation_labels_read": False,
        "test_labels_read": False,
        "test_partition_opened_or_hashed": False,
        "substantive_training_started": NO_SUBSTANTIVE_TRAINING,
    }
    if label_kind != "categorical":
        return (
            {
                **base,
                "status": "not_applicable_non_categorical_no_label_scan",
                "training_labels_read": False,
                "training_rows_scanned": 0,
                "training_label_counts": {},
                "capped_training_label_counts": {},
                "skip_reasons": [],
            },
            (),
        )
    partition_manifest_path = integration_root / "model_ready" / "partitions" / "partition_manifest.json"
    if sha256_file(partition_manifest_path) != clean_text(
        integration_acceptance.get("physical_partition_manifest_file_sha256_external")
    ):
        raise ValueError("Physical partition manifest changed before training-label support scan")
    partition_manifest = json.loads(partition_manifest_path.read_text(encoding="utf-8"))
    partitions = partition_manifest.get("partitions")
    if not isinstance(partitions, Mapping):
        raise ValueError("Physical partition manifest lacks partition bindings")
    train = partitions.get("train")
    if not isinstance(train, Mapping) or clean_text(train.get("path")) != "train.jsonl":
        raise ValueError("Physical training partition binding is invalid")
    train_path = partition_manifest_path.parent / "train.jsonl"
    sample = _CappedLabelSample({"train": config.diagnostic_maximum_train_examples}, config.seed)
    counts: Counter[str] = Counter()
    rows = 0
    cardinality_exceeded = False
    for example in JsonlIterableDataset(
        train_path,
        expected_sha256=_hex_digest(train.get("sha256"), "train partition sha256"),
    ):
        record_id = clean_text(example.get("record_id"))
        label = _json_label_key(example)
        if not record_id or not label:
            raise ValueError("Categorical training rows require nonblank record IDs and labels")
        rows += 1
        counts[label] += 1
        if len(counts) > config.maximum_categorical_label_cardinality:
            cardinality_exceeded = True
            break
        sample.add("train", record_id, label)
    declared_rows = int(train.get("record_count", -1))
    reasons: list[str] = []
    if cardinality_exceeded:
        reasons.append("training_categorical_cardinality_exceeds_declared_cap")
        reported_counts: dict[str, int] = {}
        sampled_counts: dict[str, int] = {}
        exact_rows_scanned = False
    else:
        if rows != declared_rows:
            raise ValueError("Categorical training-label scan row count mismatch")
        reported_counts = dict(sorted(counts.items()))
        sampled_counts = sample.label_counts("train")
        exact_rows_scanned = True
        if len(counts) != 2:
            reasons.append("training_partition_does_not_have_exactly_two_labels")
        if len(sampled_counts) != 2:
            reasons.append("capped_diagnostic_training_sample_lacks_both_classes")
    mapping: tuple[tuple[str, int], ...] = ()
    if not reasons and not set(counts).issubset({"0", "1"}):
        mapping = tuple((label, encoded) for encoded, label in enumerate(sorted(counts)))
    return (
        {
            **base,
            "status": "supported" if not reasons else "insufficient_training_label_support",
            "training_labels_read": True,
            "training_rows_scanned": rows,
            "training_rows_declared": declared_rows,
            "training_scan_complete": exact_rows_scanned,
            "training_label_counts": reported_counts,
            "capped_training_label_counts": sampled_counts,
            "binary_label_mapping": [list(item) for item in mapping],
            "skip_reasons": reasons,
        },
        mapping,
    )


def _task_slug(dataset_key: str, task_type: str) -> str:
    stem = re.sub(r"[^a-z0-9]+", "_", task_type.casefold()).strip("_")[:80] or "task"
    return f"{stem}-{hashlib.sha256(dataset_key.encode()).hexdigest()[:12]}"


@contextmanager
def _transactional_directory(destination: Path) -> Iterator[Path]:
    final = destination.resolve()
    if final.exists():
        raise FileExistsError(f"Immutable corpus-readiness output already exists: {final}")
    final.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{final.name}.building-", dir=final.parent)).resolve()
    try:
        yield staging
        if final.exists():
            raise FileExistsError(f"Immutable corpus-readiness output appeared: {final}")
        os.replace(staging, final)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _locked_test_binding(
    task_root_relative: str,
    integration_acceptance: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    source_relative = "model_ready/partitions/test.jsonl"
    inventory = integration_acceptance.get("component_inventory")
    if not isinstance(inventory, Mapping) or source_relative not in inventory:
        raise RuntimeError("Integration acceptance lacks its physically routed test binding")
    source = inventory[source_relative]
    if not isinstance(source, Mapping):
        raise RuntimeError("Integration test binding is malformed")
    top_relative = f"{task_root_relative}/integration/{source_relative}"
    acceptance_relative = f"{task_root_relative}/integration/acceptance.json"
    return top_relative, {
        "relative_path": top_relative,
        "sha256": _hex_digest(source.get("sha256"), "integration test sha256"),
        "size_bytes": int(source.get("size_bytes", -1)),
        "verification_source": "transitively_bound_during_physical_routing_no_reopen",
        "source_integration_acceptance_relative_path": acceptance_relative,
        "source_component_relative_path": source_relative,
    }


def _component_inventory(
    root: Path,
    locked: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    inventory: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*"), key=lambda candidate: candidate.as_posix()):
        if path.is_symlink():
            raise ValueError(f"Corpus-readiness output contains a symlink: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative == "acceptance.json":
            continue
        if relative in locked:
            inventory[relative] = dict(locked[relative])
            continue
        inventory[relative] = {
            "relative_path": relative,
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
            "verification_source": "post_materialization_sha256",
        }
    if set(locked) - set(inventory):
        raise RuntimeError(f"Locked test artifacts are absent: {sorted(set(locked) - set(inventory))}")
    return dict(sorted(inventory.items()))


def _assert_portable(value: Any, path: str = "acceptance") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _assert_portable(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_portable(item, f"{path}[{index}]")
    elif isinstance(value, str) and (value.startswith("/") or re.match(r"^[A-Za-z]:[\\/]", value)):
        raise ValueError(f"Absolute path persisted in portable corpus acceptance: {path}")


def _verify_json_training_flags(root: Path) -> None:
    for path in root.rglob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("substantive_training_started") is not False:
            raise RuntimeError(f"JSON artifact violates the no-training declaration: {path}")
        _assert_portable(payload, path.relative_to(root).as_posix())


def _verify_task_accounting(root: Path, acceptance: Mapping[str, Any]) -> None:
    tasks = acceptance.get("tasks")
    task_order = acceptance.get("task_order")
    counts = acceptance.get("task_counts")
    if not isinstance(tasks, list) or not isinstance(task_order, list) or not isinstance(counts, Mapping):
        raise ValueError("Corpus acceptance task accounting is malformed")
    dataset_keys = [clean_text(record.get("dataset_key")) for record in tasks if isinstance(record, Mapping)]
    if len(dataset_keys) != len(tasks) or dataset_keys != sorted(dataset_keys) or dataset_keys != task_order:
        raise ValueError("Corpus task enumeration is incomplete or nondeterministic")
    observed: Counter[str] = Counter()
    observed["listed_task_datasets"] = len(tasks)
    inventory = acceptance.get("component_inventory")
    if not isinstance(inventory, Mapping):
        raise ValueError("Corpus task accounting lacks the output inventory")
    seen_task_paths: set[str] = set()
    for record in tasks:
        if not isinstance(record, Mapping):
            raise ValueError("Corpus task record is not an object")
        scope = clean_text(record.get("task_scope"))
        if scope not in {"default", "derived_sensitivity"}:
            raise ValueError(f"Corpus task has an invalid scope: {scope}")
        observed[f"scope_{scope}"] += 1
        observed[f"{scope}_tasks_enumerated"] += 1
        task_output_path = _safe_relative(record.get("task_output_path"), "task output path")
        if not task_output_path.startswith("tasks/") or task_output_path in seen_task_paths:
            raise ValueError("Corpus task output paths are invalid or duplicated")
        seen_task_paths.add(task_output_path)
        status_relative = f"{task_output_path}/task_status.json"
        if status_relative not in inventory:
            raise ValueError("Corpus task status is not bound by the output inventory")
        status_path = root / status_relative
        persisted_status = json.loads(status_path.read_text(encoding="utf-8"))
        if canonical_json(persisted_status) != canonical_json(dict(record)):
            raise ValueError("Top task record diverges from its inventory-bound task_status.json")
        preflight = record.get("preflight")
        if not isinstance(preflight, Mapping):
            raise ValueError("Corpus task lacks structural preflight evidence")
        label_access = preflight.get("label_access")
        if not isinstance(label_access, Mapping) or dict(label_access) != (STRUCTURAL_PREFLIGHT_LABEL_ACCESS):
            raise ValueError("Structural preflight inspected or persisted task labels")
        status = clean_text(record.get("status"))
        if status == "skipped_insufficient_fixed_split_support":
            reasons = record.get("skip_reasons")
            if not isinstance(reasons, list) or not reasons:
                raise ValueError("Skipped task lacks explicit preflight reasons")
            observed[f"{scope}_tasks_preflight_skipped"] += 1
            continue
        if status not in {
            "integrated_and_diagnostic_completed",
            "integrated_with_diagnostic_skipped",
            "integrated_with_diagnostic_not_run_insufficient_training_label_support",
        }:
            raise ValueError(f"Corpus task has an invalid terminal status: {status}")
        observed[f"{scope}_tasks_integrated"] += 1
        integration_relative = _safe_relative(
            record.get("integration_acceptance_path"), "integration acceptance path"
        )
        diagnostic_relative = _safe_relative(
            record.get("diagnostic_acceptance_path"), "diagnostic acceptance path"
        )
        if sha256_file(root / integration_relative) != clean_text(
            record.get("integration_acceptance_sha256")
        ):
            raise ValueError("Task integration acceptance hash does not match top accounting")
        if sha256_file(root / diagnostic_relative) != clean_text(record.get("diagnostic_acceptance_sha256")):
            raise ValueError("Task diagnostic acceptance hash does not match top accounting")
        integration = json.loads((root / integration_relative).read_text(encoding="utf-8"))
        if {key: int(value) for key, value in integration["split_partition_counts"].items()} != preflight.get(
            "predicted_partition_rows"
        ):
            raise ValueError("Task preflight and integration split counts do not reconcile")
        training_support = record.get("training_label_support")
        if (
            not isinstance(training_support, Mapping)
            or training_support.get("validation_labels_read") is not False
            or training_support.get("test_labels_read") is not False
            or training_support.get("test_partition_opened_or_hashed") is not False
        ):
            raise ValueError("Training-label support crossed the validation/test boundary")
        diagnostic_status = clean_text(record.get("diagnostic_status"))
        expected_terminal = {
            "completed_diagnostic_only": "integrated_and_diagnostic_completed",
            "skipped": "integrated_with_diagnostic_skipped",
            "not_run": ("integrated_with_diagnostic_not_run_insufficient_training_label_support"),
        }.get(diagnostic_status)
        if status != expected_terminal:
            raise ValueError("Task terminal status misstates its diagnostic disposition")
        observed[f"diagnostic_status_{diagnostic_status}"] += 1
        if record.get("test_partition_opened_by_diagnostics") is not False:
            raise ValueError("Task diagnostics do not prove the test lockbox stayed closed")
    normalized_counts = {str(key): int(value) for key, value in counts.items()}
    if dict(sorted(observed.items())) != dict(sorted(normalized_counts.items())):
        raise ValueError("Corpus top task counts do not reconcile to task records")


def _verify_bundle_root(root: Path, acceptance: Mapping[str, Any]) -> None:
    if acceptance.get("schema_version") != CORPUS_READINESS_SCHEMA_VERSION:
        raise ValueError("Unrecognized corpus-readiness acceptance schema")
    if acceptance.get("substantive_training_started") is not False:
        raise ValueError("Corpus acceptance violates the no-training boundary")
    configuration = acceptance.get("configuration")
    if not isinstance(configuration, Mapping) or stable_json_digest(configuration) != clean_text(
        acceptance.get("configuration_sha256")
    ):
        raise ValueError("Corpus configuration digest mismatch")
    expected_label_policy = _label_access_policy(configuration)
    label_policy = acceptance.get("label_access_policy")
    if not isinstance(label_policy, Mapping) or dict(label_policy) != expected_label_policy:
        raise ValueError("Corpus top label-access policy does not match its fixed configuration")
    inventory = acceptance.get("component_inventory")
    if not isinstance(inventory, Mapping):
        raise ValueError("Corpus acceptance lacks a component inventory")
    if stable_json_digest(inventory) != clean_text(acceptance.get("component_inventory_sha256")):
        raise ValueError("Corpus component inventory digest mismatch")
    actual: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Corpus-readiness output contains a symlink: {path}")
        if path.is_file() and path != root / "acceptance.json":
            actual.add(path.relative_to(root).as_posix())
    if actual != set(inventory):
        raise ValueError(
            "Corpus output membership mismatch; "
            f"unbound={sorted(actual - set(inventory))}, missing={sorted(set(inventory) - actual)}"
        )
    for relative, raw_entry in sorted(inventory.items()):
        if not isinstance(raw_entry, Mapping) or raw_entry.get("relative_path") != relative:
            raise ValueError(f"Invalid corpus component record: {relative}")
        path = root / relative
        if raw_entry.get("verification_source") == ("transitively_bound_during_physical_routing_no_reopen"):
            acceptance_relative = _safe_relative(
                raw_entry.get("source_integration_acceptance_relative_path"),
                "source integration acceptance",
            )
            source_component = _safe_relative(
                raw_entry.get("source_component_relative_path"), "source component"
            )
            nested_path = root / acceptance_relative
            nested = json.loads(nested_path.read_text(encoding="utf-8"))
            nested_inventory = nested.get("component_inventory")
            if not isinstance(nested_inventory, Mapping) or source_component not in nested_inventory:
                raise ValueError(f"Locked test lacks transitive integration binding: {relative}")
            source = nested_inventory[source_component]
            if (
                not isinstance(source, Mapping)
                or clean_text(source.get("sha256")) != clean_text(raw_entry.get("sha256"))
                or int(source.get("size_bytes", -1)) != int(raw_entry.get("size_bytes", -2))
            ):
                raise ValueError(f"Locked test transitive binding changed: {relative}")
            # Deliberately no open, read, hash, or byte-size inspection of path.
            if relative not in actual:
                raise ValueError(f"Locked test is absent: {relative}")
            continue
        if path.stat().st_size != int(raw_entry.get("size_bytes", -1)):
            raise ValueError(f"Corpus component size mismatch: {relative}")
        if sha256_file(path) != clean_text(raw_entry.get("sha256")):
            raise ValueError(f"Corpus component SHA-256 mismatch: {relative}")
    _assert_portable(acceptance)
    _verify_json_training_flags(root)
    _verify_task_accounting(root, acceptance)


def materialize_corpus_readiness_bundle(
    canonical_build_root: str | os.PathLike[str],
    qc_report_path: str | os.PathLike[str],
    output_directory: str | os.PathLike[str],
    config: CorpusReadinessConfig | None = None,
) -> dict[str, Any]:
    """Materialize integration and capped diagnostics for every supported default task."""

    config = config or CorpusReadinessConfig()
    config.validate()
    binding = _verify_canonical_corpus(Path(canonical_build_root), Path(qc_report_path))
    tasks: list[dict[str, Any]] = []
    locked: dict[str, dict[str, Any]] = {}
    counts: Counter[str] = Counter()
    counts["listed_task_datasets"] = len(binding.task_datasets)
    with _transactional_directory(Path(output_directory)) as staging:
        for dataset_key, raw_entry in sorted(binding.task_datasets.items()):
            if not isinstance(raw_entry, Mapping):
                raise ValueError(f"Invalid task dataset: {dataset_key}")
            entry = dict(raw_entry)
            scope = clean_text(entry.get("task_scope"))
            task_type = clean_text(entry.get("task_type"))
            slug = _task_slug(dataset_key, task_type)
            task_relative = f"tasks/{slug}"
            task_root = staging / task_relative
            counts[f"scope_{scope}"] += 1
            base_record: dict[str, Any] = {
                "dataset_key": dataset_key,
                "task_scope": scope,
                "task_type": task_type,
                "task_dataset_sha256": clean_text(entry.get("dataset_sha256")),
                "declared_rows": int(entry.get("row_count", -1)),
                "task_output_path": task_relative,
            }
            counts[f"{scope}_tasks_enumerated"] += 1
            preflight = _preflight_task(binding, dataset_key, entry, config)
            if preflight["support_decision"] == "skip":
                status = {
                    **base_record,
                    "status": "skipped_insufficient_fixed_split_support",
                    "skip_reasons": preflight["skip_reasons"],
                    "preflight": preflight,
                    "integration_materialized": False,
                    "diagnostic_materialized": False,
                    "large_model_training_started": False,
                    "substantive_training_started": False,
                }
                _atomic_json(task_root / "task_status.json", status)
                tasks.append(status)
                counts[f"{scope}_tasks_preflight_skipped"] += 1
                continue
            integration_path = task_root / "integration"
            eligibility_mode: Literal["default", "derived_sensitivity"] = (
                "derived_sensitivity" if scope == "derived_sensitivity" else "default"
            )
            integration = materialize_task_integration_bundle(
                _task_parts(binding, entry)[0].parent,
                integration_path,
                config.integration_config(task_eligibility_mode=eligibility_mode),
            )
            if integration.get("substantive_training_started") is not False:
                raise RuntimeError("Task integration violated the no-training boundary")
            if {key: int(value) for key, value in integration["split_partition_counts"].items()} != preflight[
                "predicted_partition_rows"
            ]:
                raise RuntimeError(f"Integration split diverged from exact preflight: {dataset_key}")
            integration_acceptance_sha = clean_text(integration["acceptance_file_sha256_external"])
            locked_relative, locked_entry = _locked_test_binding(task_relative, integration)
            locked[locked_relative] = locked_entry
            training_label_support, binary_mapping = _training_categorical_support(
                integration_path,
                integration,
                config,
            )
            diagnostic_path = task_root / "diagnostics"
            training_support_reasons = list(training_label_support["skip_reasons"])
            if training_support_reasons:
                diagnostic_name = "diagnostic_disposition.json"
                diagnostic_disposition = {
                    "schema_version": "corpus_diagnostic_disposition_v1",
                    "status": "not_run",
                    "reason": ";".join(training_support_reasons),
                    "integration_acceptance_sha256": integration_acceptance_sha,
                    "training_label_support": training_label_support,
                    "opened_partitions": ["train"],
                    "validation_partition_opened": False,
                    "test_partition_opened": False,
                    "model_selection_performed": False,
                    "hyperparameter_sweep_performed": False,
                    "large_model_training_started": False,
                    "substantive_training_started": False,
                }
                diagnostic_file = diagnostic_path / diagnostic_name
                _atomic_json(diagnostic_file, diagnostic_disposition)
                diagnostic_status = "not_run"
                diagnostic_reason = clean_text(diagnostic_disposition["reason"])
                diagnostic_acceptance_sha = sha256_file(diagnostic_file)
                diagnostic_materialized = False
                terminal_status = "integrated_with_diagnostic_not_run_insufficient_training_label_support"
            else:
                diagnostic = materialize_capped_diagnostic_bundle(
                    integration_path,
                    diagnostic_path,
                    integration_acceptance_sha256=integration_acceptance_sha,
                    config=config.diagnostic_config(binary_label_mapping=binary_mapping),
                )
                if diagnostic.get("substantive_training_started") is not False:
                    raise RuntimeError("Capped diagnostics violated the no-training boundary")
                diagnostic_status = clean_text(diagnostic.get("status"))
                diagnostic_reason = clean_text(diagnostic.get("reason"))
                diagnostic_name = (
                    "diagnostic_status.json" if diagnostic_status == "skipped" else "acceptance.json"
                )
                diagnostic_acceptance_sha = clean_text(diagnostic["acceptance_file_sha256_external"])
                diagnostic_materialized = True
                terminal_status = (
                    "integrated_with_diagnostic_skipped"
                    if diagnostic_status == "skipped"
                    else "integrated_and_diagnostic_completed"
                )
            status = {
                **base_record,
                "status": terminal_status,
                "skip_reasons": [],
                "preflight": preflight,
                "training_label_support": training_label_support,
                "integration_materialized": True,
                "integration_path": f"{task_relative}/integration",
                "integration_acceptance_path": f"{task_relative}/integration/acceptance.json",
                "integration_acceptance_sha256": integration_acceptance_sha,
                "diagnostic_materialized": diagnostic_materialized,
                "diagnostic_disposition_materialized": not diagnostic_materialized,
                "diagnostic_status": diagnostic_status,
                "diagnostic_reason": diagnostic_reason,
                "diagnostic_path": f"{task_relative}/diagnostics",
                "diagnostic_acceptance_path": f"{task_relative}/diagnostics/{diagnostic_name}",
                "diagnostic_acceptance_sha256": diagnostic_acceptance_sha,
                "test_partition_opened_by_diagnostics": False,
                "large_model_training_started": False,
                "substantive_training_started": False,
            }
            _atomic_json(task_root / "task_status.json", status)
            tasks.append(status)
            counts[f"{scope}_tasks_integrated"] += 1
            counts[f"diagnostic_status_{diagnostic_status}"] += 1
        for scope in ("default", "derived_sensitivity"):
            if counts[f"{scope}_tasks_enumerated"] != (
                counts[f"{scope}_tasks_preflight_skipped"] + counts[f"{scope}_tasks_integrated"]
            ):
                raise RuntimeError(f"{scope} task accounting is incomplete")
        inventory = _component_inventory(staging, locked)
        acceptance = {
            "schema_version": CORPUS_READINESS_SCHEMA_VERSION,
            "configuration": asdict(config),
            "configuration_sha256": stable_json_digest(asdict(config)),
            "source_binding": {
                "canonical_build_manifest_sha256": binding.build_manifest_sha256,
                "canonical_component_inventory_sha256": binding.component_inventory_sha256,
                "canonical_qc_report_sha256": binding.qc_report_sha256,
                "task_datasets_file_sha256": binding.task_datasets_sha256,
                "task_datasets_payload_sha256": clean_text(
                    binding.build_manifest.get("task_datasets_manifest_sha256")
                ),
                "source_id": binding.build_manifest.get("source_id"),
                "snapshot_id": binding.build_manifest.get("snapshot_id"),
            },
            "task_counts": dict(sorted(counts.items())),
            "tasks": tasks,
            "task_order": [record["dataset_key"] for record in tasks],
            "unexpected_error_policy": "abort_complete_transaction_and_publish_nothing",
            "insufficient_support_policy": (
                "single fixed-seed exact preflight with explicit skip reasons; no seed search"
            ),
            "test_lockbox_policy": (
                "transitively bind routing-time SHA-256 and size; top orchestrator and verifier "
                "do not open, read, iterate, hash, or byte-size-inspect test.jsonl"
            ),
            "label_access_policy": _label_access_policy(asdict(config)),
            "near_duplicate_claim_status": (
                "not_claim_ready_until_separate_cross_partition_near-duplicate audits are accepted"
            ),
            "component_inventory": inventory,
            "component_inventory_sha256": stable_json_digest(inventory),
            "component_inventory_policy": (
                "all output files except top acceptance.json; routed test JSONL entries are "
                "transitively bound without rehash"
            ),
            "large_model_training_started": False,
            "substantive_training_started": False,
        }
        _assert_portable(acceptance)
        _atomic_json(staging / "acceptance.json", acceptance)
        _verify_bundle_root(staging, acceptance)
    final_root = Path(output_directory).resolve()
    final_acceptance = json.loads((final_root / "acceptance.json").read_text(encoding="utf-8"))
    _verify_bundle_root(final_root, final_acceptance)
    final_acceptance["acceptance_file_sha256_external"] = sha256_file(final_root / "acceptance.json")
    return final_acceptance


def verify_corpus_readiness_bundle(
    output_directory: str | os.PathLike[str],
    *,
    canonical_build_root: str | os.PathLike[str] | None = None,
    qc_report_path: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Verify an existing bundle without opening or hashing routed test lockboxes."""

    root = Path(output_directory).resolve()
    acceptance_path = root / "acceptance.json"
    if not acceptance_path.is_file():
        raise FileNotFoundError(acceptance_path)
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    if not isinstance(acceptance, dict):
        raise ValueError("Corpus acceptance must be a JSON object")
    _verify_bundle_root(root, acceptance)
    source_verified = False
    if (canonical_build_root is None) != (qc_report_path is None):
        raise ValueError("canonical_build_root and qc_report_path must be supplied together")
    if canonical_build_root is not None and qc_report_path is not None:
        binding = _verify_canonical_corpus(Path(canonical_build_root), Path(qc_report_path))
        source = acceptance.get("source_binding")
        if not isinstance(source, Mapping):
            raise ValueError("Corpus acceptance lacks source binding")
        expected = {
            "canonical_build_manifest_sha256": binding.build_manifest_sha256,
            "canonical_component_inventory_sha256": binding.component_inventory_sha256,
            "canonical_qc_report_sha256": binding.qc_report_sha256,
            "task_datasets_file_sha256": binding.task_datasets_sha256,
        }
        if any(clean_text(source.get(key)) != value for key, value in expected.items()):
            raise ValueError("Corpus source binding no longer matches canonical inputs")
        source_verified = True
    return {
        "schema_version": CORPUS_READINESS_SCHEMA_VERSION,
        "status": "verified",
        "acceptance_file_sha256": sha256_file(acceptance_path),
        "component_count": len(acceptance["component_inventory"]),
        "task_counts": acceptance["task_counts"],
        "source_reverified": source_verified,
        "test_lockboxes_opened_or_hashed": False,
        "large_model_training_started": False,
        "substantive_training_started": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Materialize or verify corpus-wide task readiness")
    subparsers = parser.add_subparsers(dest="command", required=True)
    materialize = subparsers.add_parser("materialize")
    materialize.add_argument(
        "--canonical-build-root", type=Path, default=Path("research/data/platform/canonical/full_chembl37")
    )
    materialize.add_argument(
        "--qc-report", type=Path, default=Path("research/reports/platform/qc_report.json")
    )
    materialize.add_argument(
        "--output-directory", type=Path, default=Path("research/models/platform/corpus_readiness")
    )
    verify = subparsers.add_parser("verify-existing")
    verify.add_argument(
        "--output-directory", type=Path, default=Path("research/models/platform/corpus_readiness")
    )
    verify.add_argument("--canonical-build-root", type=Path)
    verify.add_argument("--qc-report", type=Path)
    arguments = parser.parse_args(argv)
    if arguments.command == "materialize":
        result = materialize_corpus_readiness_bundle(
            arguments.canonical_build_root,
            arguments.qc_report,
            arguments.output_directory,
        )
    else:
        result = verify_corpus_readiness_bundle(
            arguments.output_directory,
            canonical_build_root=arguments.canonical_build_root,
            qc_report_path=arguments.qc_report,
        )
    print(canonical_json(result))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "CORPUS_READINESS_SCHEMA_VERSION",
    "CorpusReadinessConfig",
    "main",
    "materialize_corpus_readiness_bundle",
    "verify_corpus_readiness_bundle",
]
