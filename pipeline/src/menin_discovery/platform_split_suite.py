"""Transactional, label-blind official split-suite materialization.

The suite consumes only feature and identity columns from a QC-accepted,
manifest-bound canonical corpus.  It delegates partition construction to the
existing bounded-memory ``stream_hash_group_split_manifest`` implementation,
but never passes canonical label values to that implementation.  A transient
compatibility projection supplies explicit non-label sentinels for the legacy
semantic columns required by the splitter and is deleted before publication.

Published split manifests contain record/entity identifiers and partitions,
never labels.  Exact overlap audits are exhaustive and disk backed.  Chemical
and protein near-similarity audits are deterministic capped samples and are
therefore always described as non-exhaustive and not claim-ready.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import math
import os
import re
import shutil
import sqlite3
import stat
import tempfile
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Literal

import pyarrow as pa
import pyarrow.parquet as pq

from .features import is_exact_smiles_proxy_method, nearest_neighbor_tanimoto, scaffold_key
from .platform_corpus_readiness import (
    CanonicalCorpusBinding,
    _task_parts,
    _verify_canonical_corpus,
)
from .platform_data_sources import sha256_file
from .platform_features import normalize_protein_sequence, stable_json_digest
from .platform_splits import SplitConfig, stream_hash_group_split_manifest

SPLIT_SUITE_SCHEMA_VERSION = "platform_split_suite_v1"
FEATURE_PROJECTION_SCHEMA_VERSION = "platform_split_feature_projection_v1"
LEAKAGE_DIAGNOSTIC_SCHEMA_VERSION = "platform_split_feature_leakage_v1"
NO_SUBSTANTIVE_TRAINING = False
DEFAULT_OUTPUT_DIRECTORY = Path("research/data/platform/splits/full_chembl37")
PARTITIONS = ("train", "validation", "test")
OFFICIAL_STRATEGIES = (
    "molecule_grouped",
    "scaffold",
    "source_holdout",
    "protein_holdout",
    "target_holdout",
    "double_cold",
)
MANDATORY_CANDIDATES = frozenset({"molecule_grouped", "scaffold"})
EXACT_OVERLAP_EVIDENCE = "exhaustive_disk_backed_for_every_materialized_split"
NEAR_SIMILARITY_EVIDENCE = "deterministic_capped_sample_non_exhaustive"
TOP_CLAIM_READINESS = (
    "not_claim_ready_until_sampled_near_similarity_is_replaced_or_accepted_with_an "
    "independent scalable audit and every applicable split is reviewed"
)
TASK_CLAIM_READINESS = "not_claim_ready_sampled_near_similarity_and_human_leakage_acceptance_required"
LEAKAGE_CLAIM_READINESS = (
    "not_claim_ready_sampled_near_similarity_requires_independent_exhaustive_or_accepted_indexed_audit"
)
TEST_FEATURE_ACCESS_POLICY = (
    "test features may be inspected only for split/leakage auditing; no test label is "
    "present in any suite input projection or output"
)
FORBIDDEN_LABEL_COLUMNS = frozenset(
    {
        "label_kind",
        "label_value",
        "label_text",
        "label_relation",
        "label_lower_bound",
        "label_upper_bound",
        "label_unit",
    }
)
TOP_ACCEPTANCE_KEYS = frozenset(
    {
        "schema_version",
        "configuration",
        "configuration_sha256",
        "source_binding",
        "task_order",
        "tasks",
        "strategy_order",
        "mandatory_candidates",
        "accounting",
        "label_access_contract",
        "test_feature_access_policy",
        "exact_overlap_evidence",
        "near_similarity_evidence",
        "claim_readiness",
        "no_seed_search",
        "component_inventory",
        "component_inventory_sha256",
        "large_model_training_started",
        "substantive_training_started",
    }
)
SOURCE_BINDING_KEYS = frozenset(
    {
        "canonical_build_manifest_sha256",
        "canonical_component_inventory_sha256",
        "canonical_qc_report_sha256",
        "task_datasets_file_sha256",
        "task_datasets_payload_sha256",
        "source_id",
        "snapshot_id",
    }
)
TASK_STATUS_KEYS = frozenset(
    {
        "schema_version",
        "dataset_key",
        "task_scope",
        "task_type",
        "task_slug",
        "canonical_task_dataset_sha256",
        "canonical_declared_rows",
        "feature_projection_path",
        "feature_projection",
        "strategies",
        "strategy_order",
        "mandatory_candidates",
        "mandatory_candidates_materialized",
        "claim_readiness",
        "label_values_read",
        "test_labels_disclosed",
        "large_model_training_started",
        "substantive_training_started",
    }
)
FEATURE_PROJECTION_KEYS = frozenset(
    {
        "schema_version",
        "canonical_columns_requested",
        "canonical_label_columns_requested",
        "label_values_read",
        "rows",
        "maximum_batch_rows",
        "feature_schema_sha256",
        "feature_file_sha256",
        "feature_file_bytes",
        "unique_entity_counts",
        "missing_feature_row_counts",
        "scaffold_method_row_counts",
        "transient_routing_projection",
        "large_model_training_started",
        "substantive_training_started",
    }
)
BASE_STRATEGY_STATUS_KEYS = frozenset(
    {
        "strategy",
        "mandatory_candidate",
        "fixed_seed",
        "seed_search_performed",
        "applicability",
        "status",
        "reasons",
        "split_materialized",
        "label_values_read",
        "test_labels_disclosed",
        "large_model_training_started",
        "substantive_training_started",
    }
)
MATERIALIZED_STRATEGY_KEYS = BASE_STRATEGY_STATUS_KEYS | frozenset(
    {
        "split_path",
        "split_sha256",
        "split_rows",
        "partition_counts",
        "sidecar_path",
        "sidecar_sha256",
        "leakage_diagnostics_path",
        "leakage_diagnostics_sha256",
        "exact_overlap_status",
        "near_similarity_status",
    }
)
SIDECAR_KEYS = frozenset(
    {
        "algorithm",
        "bounded_memory",
        "canonical_task_dataset_sha256",
        "claim_readiness",
        "config",
        "config_sha256",
        "dataset_key",
        "exact_group_exclusion",
        "feature_projection_path",
        "feature_projection_sha256",
        "fraction_policy",
        "group_method_row_counts",
        "label_values_read",
        "large_model_training_started",
        "manifest_path",
        "manifest_sha256",
        "near_duplicate_audit",
        "observation_kind_counts",
        "partition_counts",
        "record_count",
        "record_id_uniqueness",
        "routing_input_policy",
        "row_order_binding",
        "schema_version",
        "source_dataset_sha256",
        "split_name",
        "strategy",
        "substantive_training_started",
        "suite_schema_version",
        "task_signature",
        "test_labels_disclosed",
    }
)
LEAKAGE_KEYS = frozenset(
    {
        "chemical_near_similarity",
        "claim_readiness",
        "exact_overlap_audit",
        "label_values_read",
        "large_model_training_started",
        "partition_counts",
        "protein_near_similarity",
        "record_coverage",
        "schema_version",
        "strategy",
        "substantive_training_started",
        "test_feature_access_policy",
        "test_labels_disclosed",
    }
)
Strategy = Literal[
    "molecule_grouped",
    "scaffold",
    "source_holdout",
    "protein_holdout",
    "target_holdout",
    "double_cold",
]

_FEATURE_FIELDS = (
    "record_id",
    "molecule_id",
    "protein_id",
    "target_id",
    "source_id",
    "smiles",
    "sequence",
)
_FEATURE_SCHEMA = pa.schema([(name, pa.string()) for name in _FEATURE_FIELDS])
_PUBLISHED_SPLIT_SCHEMA = pa.schema(
    [
        ("record_id", pa.string()),
        ("split", pa.string()),
        ("group_id", pa.string()),
        ("strategy", pa.string()),
        ("split_name", pa.string()),
        ("seed", pa.int64()),
        ("molecule_id", pa.string()),
        ("protein_id", pa.string()),
        ("task_id", pa.string()),
        ("source_id", pa.string()),
        ("document_year", pa.string()),
        ("observation_kind", pa.string()),
        ("source_row_index", pa.int64()),
    ]
)
_ROUTING_SCHEMA = pa.schema(
    [
        ("record_id", pa.string()),
        ("molecule_id", pa.string()),
        ("protein_id", pa.string()),
        ("canonical_target_id", pa.string()),
        ("source_id", pa.string()),
        ("standardized_smiles", pa.string()),
        ("sequence", pa.string()),
        ("assay_id", pa.string()),
        ("snapshot_id", pa.string()),
        ("source_record_id", pa.string()),
        ("task_id", pa.string()),
        ("task_type", pa.string()),
        ("label_kind", pa.string()),
        ("label_value", pa.string()),
        ("label_text", pa.string()),
        ("label_relation", pa.string()),
        ("label_lower_bound", pa.string()),
        ("label_upper_bound", pa.string()),
        ("label_unit", pa.string()),
        ("observation_kind", pa.string()),
        ("access_class", pa.string()),
        ("inclusion_status", pa.string()),
        ("default_task_eligible", pa.bool_()),
        ("evidence_domain", pa.string()),
        ("endpoint", pa.string()),
        ("assay_family", pa.string()),
        ("document_year", pa.string()),
    ]
)


@dataclass(frozen=True)
class SplitSuiteConfig:
    """Fixed, serializable controls for official feature-only splits."""

    seed: int = 20260804
    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    batch_size: int = 50_000
    near_sample_cap_per_partition: int = 256
    chemical_tanimoto_threshold: float = 0.80
    chemical_fingerprint_bits: int = 2_048
    chemical_fingerprint_radius: int = 2
    protein_kmer_size: int = 3
    protein_jaccard_threshold: float = 0.80

    def validate(self) -> None:
        fractions = (self.train_fraction, self.validation_fraction, self.test_fraction)
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer and not a boolean")
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in fractions):
            raise ValueError("Split fractions must be numeric and not booleans")
        if any(not 0 < value < 1 for value in fractions) or not math.isclose(
            sum(fractions), 1.0, rel_tol=0.0, abs_tol=1e-9
        ):
            raise ValueError("Split fractions must be positive and sum exactly to one")
        if type(self.batch_size) is not int or not 1 <= self.batch_size <= 250_000:
            raise ValueError("batch_size must be between 1 and 250000")
        if (
            type(self.near_sample_cap_per_partition) is not int
            or not 1 <= self.near_sample_cap_per_partition <= 2_048
        ):
            raise ValueError("near_sample_cap_per_partition must be between 1 and 2048")
        if (
            isinstance(self.chemical_tanimoto_threshold, bool)
            or not isinstance(self.chemical_tanimoto_threshold, (int, float))
            or not 0 < self.chemical_tanimoto_threshold <= 1
        ):
            raise ValueError("chemical_tanimoto_threshold must be in (0, 1]")
        if (
            type(self.chemical_fingerprint_bits) is not int
            or not 64 <= self.chemical_fingerprint_bits <= 8_192
        ):
            raise ValueError("chemical_fingerprint_bits must be between 64 and 8192")
        if (
            type(self.chemical_fingerprint_radius) is not int
            or not 1 <= self.chemical_fingerprint_radius <= 4
        ):
            raise ValueError("chemical_fingerprint_radius must be between 1 and 4")
        if type(self.protein_kmer_size) is not int or not 1 <= self.protein_kmer_size <= 8:
            raise ValueError("protein_kmer_size must be between 1 and 8")
        if (
            isinstance(self.protein_jaccard_threshold, bool)
            or not isinstance(self.protein_jaccard_threshold, (int, float))
            or not 0 < self.protein_jaccard_threshold <= 1
        ):
            raise ValueError("protein_jaccard_threshold must be in (0, 1]")


def _validated_configuration(value: object) -> SplitSuiteConfig:
    if not isinstance(value, Mapping):
        raise ValueError("Split-suite configuration must be an object")
    expected_keys = {field.name for field in fields(SplitSuiteConfig)}
    if set(value) != expected_keys:
        raise ValueError(
            "Split-suite configuration keys differ from the exact contract; "
            f"extra={sorted(set(value) - expected_keys)}, missing={sorted(expected_keys - set(value))}"
        )
    try:
        config = SplitSuiteConfig(**dict(value))
        config.validate()
    except (TypeError, ValueError) as exc:
        raise ValueError("Split-suite configuration values are invalid") from exc
    if asdict(config) != dict(value):
        raise ValueError("Split-suite configuration did not round-trip exactly")
    return config


def _clean(value: object) -> str:
    return "" if value is None else str(value).strip()


def _safe_relative(value: object, field: str) -> str:
    text = _clean(value)
    path = Path(text)
    if not text or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe or blank {field}: {text!r}")
    return path.as_posix()


def _absolute_without_resolving(path: Path) -> Path:
    return Path(os.path.abspath(path.expanduser()))


def _reject_symlink_path_chain(path: Path, field: str) -> None:
    current = _absolute_without_resolving(path)
    while True:
        if current.is_symlink():
            raise ValueError(f"{field} contains a symlink in its path chain: {current}")
        parent = current.parent
        if parent == current:
            break
        current = parent


def _label_access_contract() -> dict[str, Any]:
    return {
        "canonical_label_columns_requested": [],
        "label_values_read": False,
        "test_labels_disclosed": False,
        "published_parquet_contains_label_columns": False,
    }


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f"Duplicate JSON object key is prohibited: {key!r}")
        payload[key] = value
    return payload


def _canonical_json_text(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _load_strict_json(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(text, object_pairs_hook=_reject_duplicate_keys)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid split-suite JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Split-suite JSON is not an object: {path}")
    if text != _canonical_json_text(payload):
        raise ValueError(f"Split-suite JSON is not in canonical sorted representation: {path}")
    return payload


def _require_exact_keys(value: Mapping[str, Any], expected: frozenset[str], field: str) -> None:
    observed = set(value)
    if observed != expected:
        raise ValueError(
            f"{field} keys differ from the exact schema; "
            f"extra={sorted(observed - expected)}, missing={sorted(expected - observed)}"
        )


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    if payload.get("substantive_training_started") is not NO_SUBSTANTIVE_TRAINING:
        raise ValueError("Every split-suite JSON must state substantive_training_started=false")
    if payload.get("large_model_training_started") is not False:
        raise ValueError("Every split-suite JSON must state large_model_training_started=false")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(_canonical_json_text(payload))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def _transactional_directory(destination: Path) -> Iterator[Path]:
    _reject_symlink_path_chain(destination.parent, "split-suite destination")
    final = destination.resolve()
    if not final.name or final == Path(final.anchor):
        raise ValueError("Refusing a broad or root split-suite destination")
    if final.exists():
        raise FileExistsError(f"Immutable split-suite output already exists: {final}")
    final.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{final.name}.building-", dir=final.parent)).resolve()
    committed = False
    try:
        yield staging
        if final.exists():
            raise FileExistsError(f"Split-suite destination appeared during transaction: {final}")
        os.replace(staging, final)
        committed = True
    finally:
        if not committed:
            shutil.rmtree(staging, ignore_errors=True)


def _assert_portable(value: Any, path: str = "acceptance") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _assert_portable(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_portable(item, f"{path}[{index}]")
    elif isinstance(value, str) and (value.startswith("/") or re.match(r"^[A-Za-z]:[\\/]", value)):
        raise ValueError(f"Absolute path persisted in split-suite JSON: {path}")


def _task_slug(dataset_key: str) -> str:
    readable = re.sub(r"[^a-z0-9]+", "-", dataset_key.casefold()).strip("-")[:72]
    digest = hashlib.sha256(dataset_key.encode("utf-8")).hexdigest()[:12]
    return f"{readable or 'task'}-{digest}"


def _resolve_feature_columns(parts: Sequence[Path]) -> dict[str, str | None]:
    candidates: dict[str, tuple[str, ...]] = {
        "record_id": ("observation_id", "record_id", "measurement_id", "source_record_id"),
        "molecule_id": ("molecule_id", "structure_id", "standard_inchi_key"),
        "protein_id": ("protein_id", "canonical_target_id", "target_id"),
        "target_id": ("canonical_target_id", "target_id", "protein_id"),
        "source_id": ("source_id", "source", "snapshot_id"),
        "smiles": ("standardized_smiles", "canonical_smiles", "submitted_smiles", "smiles"),
        "sequence": ("sequence", "protein_sequence"),
    }
    available_by_part = [set(pq.ParquetFile(path).schema_arrow.names) for path in parts]
    common = set.intersection(*available_by_part)
    resolved = {
        output: next((candidate for candidate in options if candidate in common), None)
        for output, options in candidates.items()
    }
    for mandatory in ("record_id", "molecule_id", "smiles"):
        if resolved[mandatory] is None:
            raise ValueError(f"Task lacks mandatory feature-only column for {mandatory}")
    return resolved


def _selected_source_columns(columns: Mapping[str, str | None]) -> list[str]:
    return list(dict.fromkeys(value for value in columns.values() if value is not None))


def _feature_values(
    source: Mapping[str, list[Any]],
    columns: Mapping[str, str | None],
    row_count: int,
) -> dict[str, list[str]]:
    values: dict[str, list[str]] = {}
    for output in _FEATURE_FIELDS:
        source_name = columns.get(output)
        if source_name is None:
            values[output] = [""] * row_count
        else:
            values[output] = [_clean(value) for value in source[source_name]]
    return values


def _routing_values(
    features: Mapping[str, list[str]],
    *,
    dataset_key: str,
    task_type: str,
) -> dict[str, list[Any]]:
    count = len(features["record_id"])
    task_id = f"feature-routing-{hashlib.sha256(dataset_key.encode()).hexdigest()[:16]}"
    protein = [value or "feature-missing:protein" for value in features["protein_id"]]
    target = [value or "feature-missing:target" for value in features["target_id"]]
    source = [value or "feature-missing:source" for value in features["source_id"]]
    records = features["record_id"]
    return {
        "record_id": records,
        "molecule_id": features["molecule_id"],
        "protein_id": protein,
        "canonical_target_id": target,
        "source_id": source,
        "standardized_smiles": features["smiles"],
        "sequence": features["sequence"],
        "assay_id": ["feature-only-routing"] * count,
        "snapshot_id": ["canonical-manifest-bound"] * count,
        "source_record_id": records,
        "task_id": [task_id] * count,
        "task_type": [task_type] * count,
        "label_kind": ["feature_only_placeholder_not_a_label"] * count,
        "label_value": [""] * count,
        "label_text": [""] * count,
        "label_relation": ["not_read"] * count,
        "label_lower_bound": [""] * count,
        "label_upper_bound": [""] * count,
        "label_unit": ["feature_only_not_a_label"] * count,
        "observation_kind": ["curated_assertion"] * count,
        "access_class": ["public_redistributable"] * count,
        "inclusion_status": ["included"] * count,
        "default_task_eligible": [True] * count,
        "evidence_domain": ["feature_only_routing"] * count,
        "endpoint": [task_type] * count,
        "assay_family": ["feature_only_routing"] * count,
        "document_year": [""] * count,
    }


def _materialize_feature_projection(
    binding: CanonicalCorpusBinding,
    entry: Mapping[str, Any],
    dataset_key: str,
    feature_path: Path,
    routing_path: Path,
    *,
    batch_size: int,
) -> dict[str, Any]:
    parts = _task_parts(binding, entry)
    columns = _resolve_feature_columns(parts)
    source_columns = _selected_source_columns(columns)
    feature_writer: Any = None
    routing_writer: Any = None
    rows = 0
    maximum_batch_rows = 0
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    routing_path.parent.mkdir(parents=True, exist_ok=True)
    support_db = routing_path.with_suffix(".support.sqlite3")
    connection = sqlite3.connect(support_db)
    connection.executescript(
        """
        CREATE TABLE distinct_values (
          entity_type TEXT NOT NULL,
          value TEXT NOT NULL,
          PRIMARY KEY (entity_type, value)
        ) WITHOUT ROWID;
        CREATE TABLE counters (name TEXT PRIMARY KEY, value INTEGER NOT NULL) WITHOUT ROWID;
        """
    )
    counters: Counter[str] = Counter()
    task_type = _clean(entry.get("task_type"))
    try:
        for part in parts:
            parquet = pq.ParquetFile(part)
            for batch in parquet.iter_batches(batch_size=batch_size, columns=source_columns):
                row_count = batch.num_rows
                if row_count == 0:
                    continue
                maximum_batch_rows = max(maximum_batch_rows, row_count)
                source = batch.to_pydict()
                features = _feature_values(source, columns, row_count)
                if any(not value for value in features["record_id"]):
                    raise ValueError(f"Task {dataset_key} has blank stable record IDs")
                feature_table = pa.Table.from_pydict(features, schema=_FEATURE_SCHEMA)
                routing_table = pa.Table.from_pydict(
                    _routing_values(features, dataset_key=dataset_key, task_type=task_type),
                    schema=_ROUTING_SCHEMA,
                )
                if feature_writer is None:
                    feature_writer = pq.ParquetWriter(feature_path, _FEATURE_SCHEMA, compression="zstd")
                    routing_writer = pq.ParquetWriter(routing_path, _ROUTING_SCHEMA, compression="zstd")
                feature_writer.write_table(feature_table, row_group_size=batch_size)
                assert routing_writer is not None
                routing_writer.write_table(routing_table, row_group_size=batch_size)
                for entity_type, field in (
                    ("molecule", "molecule_id"),
                    ("protein", "protein_id"),
                    ("target", "target_id"),
                    ("source", "source_id"),
                ):
                    values = features[field]
                    counters[f"missing_{entity_type}_rows"] += sum(not value for value in values)
                    connection.executemany(
                        "INSERT OR IGNORE INTO distinct_values(entity_type,value) VALUES (?,?)",
                        ((entity_type, value) for value in values if value),
                    )
                counters["missing_smiles_rows"] += sum(not value for value in features["smiles"])
                scaffold_rows: list[tuple[str, str]] = []
                for value in features["smiles"]:
                    if not value:
                        continue
                    scaffold, method = scaffold_key(value)
                    scaffold_rows.append(("scaffold", scaffold))
                    counters[f"scaffold_method_{method}"] += 1
                connection.executemany(
                    "INSERT OR IGNORE INTO distinct_values(entity_type,value) VALUES (?,?)",
                    scaffold_rows,
                )
                rows += row_count
            connection.commit()
        if feature_writer is None or routing_writer is None or rows == 0:
            raise ValueError(f"Task {dataset_key} is empty")
    finally:
        if feature_writer is not None:
            feature_writer.close()
        if routing_writer is not None:
            routing_writer.close()
    try:
        unique_counts = {
            entity: int(
                connection.execute(
                    "SELECT COUNT(*) FROM distinct_values WHERE entity_type=?", (entity,)
                ).fetchone()[0]
            )
            for entity in ("molecule", "scaffold", "source", "protein", "target")
        }
    finally:
        connection.close()
        support_db.unlink(missing_ok=True)
    declared_rows = int(entry.get("row_count", -1))
    if rows != declared_rows:
        raise ValueError(
            f"Feature-only projection row count differs from task manifest: {rows} != {declared_rows}"
        )
    schema_digest = hashlib.sha256(str(_FEATURE_SCHEMA).encode("utf-8")).hexdigest()
    return {
        "schema_version": FEATURE_PROJECTION_SCHEMA_VERSION,
        "canonical_columns_requested": source_columns,
        "canonical_label_columns_requested": [],
        "label_values_read": False,
        "rows": rows,
        "maximum_batch_rows": maximum_batch_rows,
        "feature_schema_sha256": schema_digest,
        "feature_file_sha256": sha256_file(feature_path),
        "feature_file_bytes": feature_path.stat().st_size,
        "unique_entity_counts": unique_counts,
        "missing_feature_row_counts": {
            key: int(value) for key, value in sorted(counters.items()) if key.startswith("missing_")
        },
        "scaffold_method_row_counts": {
            key.removeprefix("scaffold_method_"): int(value)
            for key, value in sorted(counters.items())
            if key.startswith("scaffold_method_")
        },
        "transient_routing_projection": (
            "derived only from listed feature columns plus constant non-label compatibility sentinels; "
            "deleted before atomic publication"
        ),
        "large_model_training_started": False,
        "substantive_training_started": False,
    }


def _applicability(strategy: Strategy, support: Mapping[str, Any]) -> tuple[bool, list[str]]:
    unique = support["unique_entity_counts"]
    missing = support["missing_feature_row_counts"]
    reasons: list[str] = []
    if int(unique["molecule"]) < 3:
        reasons.append("fewer_than_three_unique_molecules")
    if int(missing.get("missing_molecule_rows", 0)):
        reasons.append("missing_molecule_identity")
    if strategy == "scaffold":
        if int(unique["scaffold"]) < 3:
            reasons.append("fewer_than_three_unique_scaffolds")
        if int(missing.get("missing_smiles_rows", 0)):
            reasons.append("missing_smiles")
        proxy_methods = {
            str(method): int(count)
            for method, count in support["scaffold_method_row_counts"].items()
            if int(count) and is_exact_smiles_proxy_method(method)
        }
        if proxy_methods.get("exact_smiles_proxy", 0):
            reasons.append("invalid_structure_requires_exact_smiles_proxy")
        if proxy_methods.get("exact_smiles_proxy_rdkit_exception", 0):
            reasons.append("rdkit_scaffold_exception_requires_exact_smiles_proxy")
        if set(proxy_methods) - {
            "exact_smiles_proxy",
            "exact_smiles_proxy_rdkit_exception",
        }:
            reasons.append("unrecognized_exact_smiles_proxy_is_not_true_scaffold")
    required_entity = {
        "source_holdout": "source",
        "protein_holdout": "protein",
        "target_holdout": "target",
    }.get(strategy)
    if required_entity is not None:
        if int(unique[required_entity]) < 3:
            reasons.append(f"fewer_than_three_unique_{required_entity}_entities")
        if int(missing.get(f"missing_{required_entity}_rows", 0)):
            reasons.append(f"missing_{required_entity}_identity")
    if strategy == "double_cold":
        for entity in ("protein",):
            if int(unique[entity]) < 3:
                reasons.append(f"fewer_than_three_unique_{entity}_entities")
            if int(missing.get(f"missing_{entity}_rows", 0)):
                reasons.append(f"missing_{entity}_identity")
    return not reasons, reasons


def _split_config(strategy: Strategy, config: SplitSuiteConfig, dataset_key: str) -> SplitConfig:
    intended_uses = {
        "molecule_grouped": "unseen molecule identity within the accepted task domain",
        "scaffold": "unseen Bemis-Murcko scaffold within the accepted task domain",
        "source_holdout": "source identity absent from training",
        "protein_holdout": "protein identity absent from training",
        "target_holdout": "canonical target identity absent from training",
        "double_cold": "both molecule and protein identity absent from training; mixed pairs excluded",
    }
    return SplitConfig(
        name=f"{strategy}_fixed_v1",
        strategy=strategy,
        intended_use=intended_uses[strategy],
        seed=config.seed,
        train_fraction=config.train_fraction,
        validation_fraction=config.validation_fraction,
        test_fraction=config.test_fraction,
        record_id_column="record_id",
        molecule_id_column="molecule_id",
        smiles_column="standardized_smiles",
        protein_id_column="protein_id",
        target_id_column="canonical_target_id",
        source_id_column="source_id",
        label_column="label_value",
        task_type="regression",
    )


def _rewrite_split_sidecar(
    sidecar_path: Path,
    metadata: Mapping[str, Any],
    *,
    dataset_key: str,
    canonical_dataset_sha256: str,
    feature_relative_path: str,
    feature_sha256: str,
) -> dict[str, Any]:
    retained = {
        key: value
        for key, value in metadata.items()
        if key
        not in {
            "source_task_path",
            "source_dataset",
            "source_dataset_manifest_path",
            "sidecar_path",
            "sidecar_sha256",
        }
    }
    payload = {
        **retained,
        "suite_schema_version": SPLIT_SUITE_SCHEMA_VERSION,
        "dataset_key": dataset_key,
        "canonical_task_dataset_sha256": canonical_dataset_sha256,
        "feature_projection_path": feature_relative_path,
        "feature_projection_sha256": feature_sha256,
        "routing_input_policy": (
            "transient feature-derived compatibility projection; canonical labels never requested, "
            "read, copied, partitioned, summarized, or published"
        ),
        "label_values_read": False,
        "test_labels_disclosed": False,
        "large_model_training_started": False,
        "substantive_training_started": False,
    }
    _assert_portable(payload)
    _atomic_json(sidecar_path, payload)
    return payload


def _insert_rows(connection: sqlite3.Connection, sql: str, rows: Sequence[tuple[Any, ...]]) -> None:
    try:
        connection.executemany(sql, rows)
    except sqlite3.IntegrityError as exc:
        raise ValueError("Duplicate or inconsistent stable record ID in split audit") from exc


def _ranked_sample(
    rows: Iterator[tuple[str, str, str]], *, seed: int, cap: int, salt: str
) -> tuple[dict[str, list[tuple[str, str]]], dict[str, int]]:
    heaps: dict[str, list[tuple[int, str, str]]] = {partition: [] for partition in PARTITIONS}
    populations: Counter[str] = Counter()
    for partition, record_id, value in rows:
        if partition not in heaps or not value:
            continue
        populations[partition] += 1
        rank = int.from_bytes(
            hashlib.sha256(f"{seed}|{salt}|{partition}|{record_id}".encode()).digest(), "big"
        )
        item = (-rank, record_id, value)
        heap = heaps[partition]
        if len(heap) < cap:
            heapq.heappush(heap, item)
        elif item > heap[0]:
            heapq.heapreplace(heap, item)
    samples = {
        partition: [(record_id, value) for _, record_id, value in sorted(heap, reverse=True)]
        for partition, heap in heaps.items()
    }
    return samples, {partition: int(populations[partition]) for partition in PARTITIONS}


def _chemical_sample_audit(
    samples: Mapping[str, Sequence[tuple[str, str]]],
    populations: Mapping[str, int],
    config: SplitSuiteConfig,
) -> dict[str, Any]:
    pairs: dict[str, Any] = {}
    backend_names: set[str] = set()
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        reference = list(samples[left])
        query = list(samples[right])
        key = f"{left}_vs_{right}"
        if not reference or not query:
            pairs[key] = {
                "status": "not_computable_empty_sample",
                "sampled_pair_comparisons": 0,
                "population_cartesian_pairs": int(populations[left] * populations[right]),
            }
            continue
        maxima, _, backend = nearest_neighbor_tanimoto(
            [value for _, value in query],
            [value for _, value in reference],
            backend="rdkit",
            n_bits=config.chemical_fingerprint_bits,
            radius=config.chemical_fingerprint_radius,
            chunk_size=min(256, len(query)),
        )
        backend_names.add(backend)
        pairs[key] = {
            "status": "completed_deterministic_capped_sample",
            "sampled_query_entities": len(query),
            "sampled_reference_entities": len(reference),
            "sampled_pair_comparisons": len(query) * len(reference),
            "population_query_entities": int(populations[right]),
            "population_reference_entities": int(populations[left]),
            "population_cartesian_pairs": int(populations[left] * populations[right]),
            "query_entities_at_or_above_threshold": int(
                sum(float(value) >= config.chemical_tanimoto_threshold for value in maxima)
            ),
            "maximum_sampled_similarity": float(maxima.max()) if len(maxima) else None,
        }
    return {
        "method": (
            f"Morgan radius {config.chemical_fingerprint_radius}, "
            f"{config.chemical_fingerprint_bits} bits; exact Tanimoto within capped samples"
        ),
        "backend": sorted(backend_names),
        "threshold": config.chemical_tanimoto_threshold,
        "sampling": "smallest_sha256_rank_per_partition_without_seed_search",
        "sample_cap_per_partition": config.near_sample_cap_per_partition,
        "population_unique_entities": dict(populations),
        "sampled_unique_entities": {partition: len(samples[partition]) for partition in PARTITIONS},
        "completeness": "sampled_non_exhaustive",
        "claim_status": "not_claim_ready_from_sampled_near_similarity",
        "partition_pairs": pairs,
    }


def _kmer_set(value: str, size: int) -> frozenset[str]:
    sequence, invalid = normalize_protein_sequence(value)
    if invalid or len(sequence) < size:
        return frozenset()
    return frozenset(sequence[index : index + size] for index in range(len(sequence) - size + 1))


def _protein_sample_audit(
    samples: Mapping[str, Sequence[tuple[str, str]]],
    populations: Mapping[str, int],
    config: SplitSuiteConfig,
) -> dict[str, Any]:
    prepared = {
        partition: [(record, _kmer_set(value, config.protein_kmer_size)) for record, value in rows]
        for partition, rows in samples.items()
    }
    pairs: dict[str, Any] = {}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        reference = [(record, kmers) for record, kmers in prepared[left] if kmers]
        query = [(record, kmers) for record, kmers in prepared[right] if kmers]
        key = f"{left}_vs_{right}"
        maxima: list[float] = []
        for _, query_kmers in query:
            maxima.append(
                max(
                    (
                        len(query_kmers & reference_kmers) / len(query_kmers | reference_kmers)
                        for _, reference_kmers in reference
                    ),
                    default=0.0,
                )
            )
        pairs[key] = {
            "status": (
                "completed_deterministic_capped_sample"
                if reference and query
                else "not_computable_empty_or_invalid_sample"
            ),
            "sampled_query_entities_with_valid_sequence": len(query),
            "sampled_reference_entities_with_valid_sequence": len(reference),
            "sampled_pair_comparisons": len(query) * len(reference),
            "population_query_entities_with_nonblank_sequence": int(populations[right]),
            "population_reference_entities_with_nonblank_sequence": int(populations[left]),
            "population_cartesian_pairs": int(populations[left] * populations[right]),
            "query_entities_at_or_above_threshold": int(
                sum(value >= config.protein_jaccard_threshold for value in maxima)
            ),
            "maximum_sampled_similarity": max(maxima) if maxima else None,
        }
    return {
        "method": f"normalized_protein_{config.protein_kmer_size}mer_Jaccard_screen",
        "interpretation": "screening similarity, not aligned percent sequence identity",
        "threshold": config.protein_jaccard_threshold,
        "sampling": "smallest_sha256_rank_per_partition_without_seed_search",
        "sample_cap_per_partition": config.near_sample_cap_per_partition,
        "population_unique_entities_with_nonblank_sequence": dict(populations),
        "sampled_unique_entities": {partition: len(samples[partition]) for partition in PARTITIONS},
        "completeness": "sampled_non_exhaustive",
        "claim_status": "not_claim_ready_from_sampled_near_similarity_or_without_alignment",
        "partition_pairs": pairs,
    }


def _leakage_diagnostics(
    feature_path: Path,
    split_path: Path,
    strategy: Strategy,
    config: SplitSuiteConfig,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="split-suite-audit-", dir=split_path.parent) as temporary:
        connection = sqlite3.connect(str(Path(temporary) / "audit.sqlite3"))
        connection.executescript(
            """
            CREATE TABLE assignments (
              record_id TEXT PRIMARY KEY,
              split TEXT NOT NULL,
              group_id TEXT NOT NULL
            ) WITHOUT ROWID;
            CREATE TABLE features (
              record_id TEXT PRIMARY KEY,
              molecule_id TEXT NOT NULL,
              protein_id TEXT NOT NULL,
              target_id TEXT NOT NULL,
              source_id TEXT NOT NULL,
              smiles TEXT NOT NULL,
              sequence TEXT NOT NULL
            ) WITHOUT ROWID;
            CREATE TABLE entity_partition (
              entity_type TEXT NOT NULL,
              entity_id TEXT NOT NULL,
              split TEXT NOT NULL,
              PRIMARY KEY(entity_type, entity_id, split)
            ) WITHOUT ROWID;
            """
        )
        split_rows = 0
        for batch in pq.ParquetFile(split_path).iter_batches(
            batch_size=config.batch_size, columns=["record_id", "split", "group_id"]
        ):
            values = batch.to_pydict()
            rows = list(zip(values["record_id"], values["split"], values["group_id"], strict=True))
            _insert_rows(
                connection,
                "INSERT INTO assignments(record_id,split,group_id) VALUES (?,?,?)",
                rows,
            )
            split_rows += len(rows)
        feature_rows = 0
        for batch in pq.ParquetFile(feature_path).iter_batches(batch_size=config.batch_size):
            values = batch.to_pydict()
            rows = list(
                zip(
                    values["record_id"],
                    values["molecule_id"],
                    values["protein_id"],
                    values["target_id"],
                    values["source_id"],
                    values["smiles"],
                    values["sequence"],
                    strict=True,
                )
            )
            _insert_rows(
                connection,
                "INSERT INTO features VALUES (?,?,?,?,?,?,?)",
                rows,
            )
            feature_rows += len(rows)
        connection.commit()
        if split_rows != feature_rows:
            raise ValueError("Split and feature projection row counts differ")
        missing = int(
            connection.execute(
                "SELECT COUNT(*) FROM assignments a LEFT JOIN features f USING(record_id) "
                "WHERE f.record_id IS NULL"
            ).fetchone()[0]
        )
        extra = int(
            connection.execute(
                "SELECT COUNT(*) FROM features f LEFT JOIN assignments a USING(record_id) "
                "WHERE a.record_id IS NULL"
            ).fetchone()[0]
        )
        if missing or extra:
            raise ValueError(f"Split/feature record coverage mismatch: missing={missing}, extra={extra}")
        connection.execute(
            "INSERT OR IGNORE INTO entity_partition "
            "SELECT 'group',group_id,split FROM assignments WHERE split IN ('train','validation','test')"
        )
        for entity, column in (
            ("molecule", "molecule_id"),
            ("protein", "protein_id"),
            ("target", "target_id"),
            ("source", "source_id"),
        ):
            connection.execute(
                "INSERT OR IGNORE INTO entity_partition "
                f"SELECT ?,f.{column},a.split FROM features f JOIN assignments a USING(record_id) "
                f"WHERE a.split IN ('train','validation','test') AND f.{column}<>''",
                (entity,),
            )
        connection.commit()
        exact: dict[str, Any] = {}
        for entity in ("group", "molecule", "protein", "target", "source"):
            population, overlap = connection.execute(
                "SELECT COUNT(*), COALESCE(SUM(CASE WHEN n>1 THEN 1 ELSE 0 END),0) FROM "
                "(SELECT entity_id,COUNT(*) AS n FROM entity_partition "
                "WHERE entity_type=? GROUP BY entity_id)",
                (entity,),
            ).fetchone()
            exact[entity] = {
                "unique_entities_in_official_partitions": int(population),
                "entities_present_in_multiple_official_partitions": int(overlap),
                "completeness": "exhaustive_disk_backed",
            }
        required_zero = {
            "molecule_grouped": ("group", "molecule"),
            "scaffold": ("group", "molecule"),
            "source_holdout": ("group", "source"),
            "protein_holdout": ("group", "protein"),
            "target_holdout": ("group", "target"),
            "double_cold": ("group", "molecule", "protein"),
        }[strategy]
        violations = [
            entity
            for entity in required_zero
            if exact[entity]["entities_present_in_multiple_official_partitions"] != 0
        ]
        if violations:
            raise ValueError(f"Exact entity exclusion failed for {strategy}: {violations}")
        chemical_rows = connection.execute(
            "SELECT a.split,MIN(f.record_id),MIN(f.smiles) FROM features f "
            "JOIN assignments a USING(record_id) WHERE a.split IN ('train','validation','test') "
            "AND f.molecule_id<>'' AND f.smiles<>'' GROUP BY a.split,f.molecule_id "
            "ORDER BY a.split,f.molecule_id"
        )
        chemical_samples, chemical_populations = _ranked_sample(
            iter(chemical_rows),
            seed=config.seed,
            cap=config.near_sample_cap_per_partition,
            salt=f"{strategy}|chemical",
        )
        protein_rows = connection.execute(
            "SELECT a.split,MIN(f.record_id),MIN(f.sequence) FROM features f "
            "JOIN assignments a USING(record_id) WHERE a.split IN ('train','validation','test') "
            "AND f.protein_id<>'' AND f.sequence<>'' GROUP BY a.split,f.protein_id "
            "ORDER BY a.split,f.protein_id"
        )
        protein_samples, protein_populations = _ranked_sample(
            iter(protein_rows),
            seed=config.seed,
            cap=config.near_sample_cap_per_partition,
            salt=f"{strategy}|protein",
        )
        connection.close()
    partition_counts: Counter[str] = Counter()
    for batch in pq.ParquetFile(split_path).iter_batches(batch_size=config.batch_size, columns=["split"]):
        partition_counts.update(_clean(value) for value in batch.column(0).to_pylist())
    return {
        "schema_version": LEAKAGE_DIAGNOSTIC_SCHEMA_VERSION,
        "strategy": strategy,
        "record_coverage": {
            "feature_records": feature_rows,
            "split_records": split_rows,
            "missing_assignments": 0,
            "extra_assignments": 0,
            "completeness": "exhaustive_disk_backed_primary_key_join",
        },
        "partition_counts": dict(sorted(partition_counts.items())),
        "exact_overlap_audit": {
            "backend": "temporary_SQLite_distinct_entity_partition_primary_keys",
            "official_partitions": list(PARTITIONS),
            "excluded_rows_not_treated_as_official_partitions": True,
            "required_zero_overlap_entities": list(required_zero),
            "entities": exact,
            "completeness": "exhaustive",
        },
        "chemical_near_similarity": _chemical_sample_audit(chemical_samples, chemical_populations, config),
        "protein_near_similarity": _protein_sample_audit(protein_samples, protein_populations, config),
        "test_feature_access_policy": (
            "test feature identities/structures/sequences may be inspected only for predeclared "
            "leakage auditing; test labels are absent from the projection and never accessed"
        ),
        "label_values_read": False,
        "test_labels_disclosed": False,
        "claim_readiness": LEAKAGE_CLAIM_READINESS,
        "large_model_training_started": False,
        "substantive_training_started": False,
    }


def _component_inventory(root: Path) -> dict[str, dict[str, Any]]:
    inventory: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_symlink():
            raise ValueError(f"Split-suite output contains a symlink: {path}")
        if not path.is_file() or path == root / "acceptance.json":
            continue
        relative = path.relative_to(root).as_posix()
        record: dict[str, Any] = {
            "relative_path": relative,
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        if path.suffix.casefold() == ".parquet":
            metadata = pq.ParquetFile(path).metadata
            record["rows"] = int(metadata.num_rows if metadata is not None else -1)
        inventory[relative] = record
    return inventory


def _verify_json_boundaries(root: Path) -> None:
    for path in root.rglob("*.json"):
        payload = _load_strict_json(path)
        if payload.get("substantive_training_started") is not False:
            raise ValueError(f"Split-suite JSON violates no-training boundary: {path}")
        if payload.get("large_model_training_started") is not False:
            raise ValueError(f"Split-suite JSON violates large-model no-training boundary: {path}")
        _assert_portable(payload, path.relative_to(root).as_posix())


def _bound_json(
    root: Path,
    inventory: Mapping[str, Any],
    relative: str,
) -> dict[str, Any]:
    safe = _safe_relative(relative, "bound JSON path")
    if safe not in inventory:
        raise ValueError(f"Split-suite accounting references an unbound JSON: {safe}")
    return _load_strict_json(root / safe)


def _verify_published_parquet_schema(path: Path, *, feature: bool) -> None:
    schema = pq.ParquetFile(path).schema_arrow
    forbidden = sorted(
        column
        for column in schema.names
        if column in FORBIDDEN_LABEL_COLUMNS or column.casefold().startswith("label_")
    )
    if forbidden:
        raise ValueError(f"Published split-suite Parquet exposes label columns: {forbidden}")
    expected = _FEATURE_SCHEMA if feature else _PUBLISHED_SPLIT_SCHEMA
    if not schema.equals(expected, check_metadata=False):
        role = "feature" if feature else "split"
        raise ValueError(f"Published {role} Parquet schema differs from its fixed contract")


def _verify_materialized_strategy(
    root: Path,
    inventory: Mapping[str, Any],
    task: Mapping[str, Any],
    item: Mapping[str, Any],
    configuration: Mapping[str, Any],
) -> None:
    _require_exact_keys(item, MATERIALIZED_STRATEGY_KEYS, "materialized strategy status")
    slug = _clean(task.get("task_slug"))
    strategy = _clean(item.get("strategy"))
    base = f"tasks/{slug}/strategies/{strategy}"
    expected_paths = {
        "split_path": f"{base}/split.parquet",
        "sidecar_path": f"{base}/split.parquet.manifest.json",
        "leakage_diagnostics_path": f"{base}/leakage_diagnostics.json",
    }
    applicability = item.get("applicability")
    if (
        item.get("split_materialized") is not True
        or item.get("reasons") != []
        or not isinstance(applicability, Mapping)
        or applicability != {"evaluated": True, "supported": True, "reasons": []}
        or "failure_class" in item
    ):
        raise ValueError(f"Materialized {strategy} has inconsistent terminal semantics")
    for field, expected in expected_paths.items():
        if item.get(field) != expected or expected not in inventory:
            raise ValueError(f"Materialized {strategy} has an unbound or unexpected {field}")
    split_path = root / expected_paths["split_path"]
    _verify_published_parquet_schema(split_path, feature=False)
    if sha256_file(split_path) != _clean(item.get("split_sha256")):
        raise ValueError(f"Materialized {strategy} split hash differs from status")
    sidecar_path = root / expected_paths["sidecar_path"]
    if sha256_file(sidecar_path) != _clean(item.get("sidecar_sha256")):
        raise ValueError(f"Materialized {strategy} sidecar hash differs from status")
    leakage_path = root / expected_paths["leakage_diagnostics_path"]
    if sha256_file(leakage_path) != _clean(item.get("leakage_diagnostics_sha256")):
        raise ValueError(f"Materialized {strategy} leakage hash differs from status")
    if (
        item.get("exact_overlap_status") != "passed_exhaustive"
        or item.get("near_similarity_status") != "sampled_non_exhaustive_not_claim_ready"
    ):
        raise ValueError(f"Materialized {strategy} overstates or changes leakage status")

    sidecar = _bound_json(root, inventory, expected_paths["sidecar_path"])
    _require_exact_keys(sidecar, SIDECAR_KEYS, "materialized split sidecar")
    if (
        sidecar.get("suite_schema_version") != SPLIT_SUITE_SCHEMA_VERSION
        or sidecar.get("dataset_key") != task.get("dataset_key")
        or sidecar.get("canonical_task_dataset_sha256") != task.get("canonical_task_dataset_sha256")
        or sidecar.get("feature_projection_path") != task.get("feature_projection_path")
        or sidecar.get("strategy") != strategy
        or sidecar.get("label_values_read") is not False
        or sidecar.get("test_labels_disclosed") is not False
        or sidecar.get("large_model_training_started") is not False
        or sidecar.get("substantive_training_started") is not False
    ):
        raise ValueError(f"Materialized {strategy} sidecar semantic binding mismatch")
    feature_path = root / _safe_relative(task.get("feature_projection_path"), "feature path")
    if sidecar.get("feature_projection_sha256") != sha256_file(feature_path):
        raise ValueError(f"Materialized {strategy} sidecar feature binding mismatch")
    if sidecar.get("manifest_sha256") != sha256_file(split_path):
        raise ValueError(f"Materialized {strategy} sidecar split binding mismatch")
    sidecar_config = sidecar.get("config")
    if (
        not isinstance(sidecar_config, Mapping)
        or sidecar_config.get("strategy") != strategy
        or int(sidecar_config.get("seed", -1)) != int(configuration.get("seed", -2))
    ):
        raise ValueError(f"Materialized {strategy} sidecar fixed configuration mismatch")
    if not _clean(sidecar.get("claim_readiness")).startswith("not_claim_ready"):
        raise ValueError(f"Materialized {strategy} sidecar overstates claim readiness")

    leakage = _bound_json(root, inventory, expected_paths["leakage_diagnostics_path"])
    _require_exact_keys(leakage, LEAKAGE_KEYS, "materialized leakage diagnostics")
    expected_zero = {
        "molecule_grouped": ["group", "molecule"],
        "scaffold": ["group", "molecule"],
        "source_holdout": ["group", "source"],
        "protein_holdout": ["group", "protein"],
        "target_holdout": ["group", "target"],
        "double_cold": ["group", "molecule", "protein"],
    }[strategy]
    exact = leakage.get("exact_overlap_audit")
    chemical = leakage.get("chemical_near_similarity")
    protein = leakage.get("protein_near_similarity")
    if (
        leakage.get("schema_version") != LEAKAGE_DIAGNOSTIC_SCHEMA_VERSION
        or leakage.get("strategy") != strategy
        or leakage.get("label_values_read") is not False
        or leakage.get("test_labels_disclosed") is not False
        or leakage.get("large_model_training_started") is not False
        or leakage.get("claim_readiness") != LEAKAGE_CLAIM_READINESS
        or not isinstance(exact, Mapping)
        or exact.get("completeness") != "exhaustive"
        or exact.get("required_zero_overlap_entities") != expected_zero
        or not isinstance(chemical, Mapping)
        or chemical.get("completeness") != "sampled_non_exhaustive"
        or chemical.get("claim_status") != "not_claim_ready_from_sampled_near_similarity"
        or not isinstance(protein, Mapping)
        or protein.get("completeness") != "sampled_non_exhaustive"
        or protein.get("claim_status") != "not_claim_ready_from_sampled_near_similarity_or_without_alignment"
    ):
        raise ValueError(f"Materialized {strategy} leakage policy mismatch")
    entities = exact.get("entities")
    if not isinstance(entities, Mapping) or any(
        not isinstance(entities.get(entity), Mapping)
        or entities[entity].get("entities_present_in_multiple_official_partitions") != 0
        for entity in expected_zero
    ):
        raise ValueError(f"Materialized {strategy} exact exclusion evidence is not accepted")
    if (
        int(chemical.get("sample_cap_per_partition", -1))
        != int(configuration.get("near_sample_cap_per_partition", -2))
        or not math.isclose(
            float(chemical.get("threshold", -1)),
            float(configuration.get("chemical_tanimoto_threshold", -2)),
        )
        or int(protein.get("sample_cap_per_partition", -1))
        != int(configuration.get("near_sample_cap_per_partition", -2))
        or not math.isclose(
            float(protein.get("threshold", -1)),
            float(configuration.get("protein_jaccard_threshold", -2)),
        )
    ):
        raise ValueError(f"Materialized {strategy} sampled-audit configuration mismatch")
    sidecar_counts = sidecar.get("partition_counts")
    leakage_counts = leakage.get("partition_counts")
    status_counts = item.get("partition_counts")
    if (
        not isinstance(sidecar_counts, Mapping)
        or dict(sidecar_counts) != status_counts
        or leakage_counts != status_counts
        or int(sidecar.get("record_count", -1)) != int(item.get("split_rows", -2))
        or sum(int(value) for value in sidecar_counts.values()) != int(item.get("split_rows", -2))
    ):
        raise ValueError(f"Materialized {strategy} row/partition accounting mismatch")


def _verify_task_strategy_accounting(root: Path, acceptance: Mapping[str, Any]) -> None:
    tasks = acceptance.get("tasks")
    task_order = acceptance.get("task_order")
    inventory = acceptance.get("component_inventory")
    configuration = acceptance.get("configuration")
    if (
        not isinstance(tasks, list)
        or not isinstance(task_order, list)
        or not isinstance(inventory, Mapping)
        or not isinstance(configuration, Mapping)
    ):
        raise ValueError("Split-suite task accounting is malformed")
    if (
        acceptance.get("strategy_order") != list(OFFICIAL_STRATEGIES)
        or acceptance.get("mandatory_candidates") != sorted(MANDATORY_CANDIDATES)
        or acceptance.get("label_access_contract") != _label_access_contract()
        or acceptance.get("no_seed_search") is not True
        or acceptance.get("exact_overlap_evidence") != EXACT_OVERLAP_EVIDENCE
        or acceptance.get("near_similarity_evidence") != NEAR_SIMILARITY_EVIDENCE
        or acceptance.get("claim_readiness") != TOP_CLAIM_READINESS
        or acceptance.get("test_feature_access_policy") != TEST_FEATURE_ACCESS_POLICY
    ):
        raise ValueError("Split-suite fixed top-level scientific policy changed")
    keys = [_clean(task.get("dataset_key")) for task in tasks if isinstance(task, Mapping)]
    if len(keys) != len(tasks) or keys != sorted(keys) or keys != task_order:
        raise ValueError("Split-suite task enumeration is incomplete or nondeterministic")
    observed: Counter[str] = Counter()
    for task in tasks:
        if not isinstance(task, Mapping):
            raise ValueError("Split-suite task record is not an object")
        _require_exact_keys(task, TASK_STATUS_KEYS, "task status")
        dataset_key = _clean(task.get("dataset_key"))
        slug = _task_slug(dataset_key)
        if task.get("task_slug") != slug:
            raise ValueError(f"Split-suite task slug is not stable: {dataset_key}")
        task_status_relative = f"tasks/{slug}/task_status.json"
        if _bound_json(root, inventory, task_status_relative) != dict(task):
            raise ValueError(f"Top task record differs from bound task status: {dataset_key}")
        feature_relative = f"tasks/{slug}/features.parquet"
        if task.get("feature_projection_path") != feature_relative or feature_relative not in inventory:
            raise ValueError(f"Task feature projection path is not exactly bound: {dataset_key}")
        feature_path = root / feature_relative
        _verify_published_parquet_schema(feature_path, feature=True)
        projection = task.get("feature_projection")
        if (
            not isinstance(projection, Mapping)
            or projection.get("canonical_label_columns_requested") != []
            or projection.get("label_values_read") is not False
            or projection.get("large_model_training_started") is not False
            or projection.get("substantive_training_started") is not False
            or set(projection.get("canonical_columns_requested", [])) & FORBIDDEN_LABEL_COLUMNS
            or projection.get("feature_file_sha256") != sha256_file(feature_path)
            or int(projection.get("rows", -1)) != int(task.get("canonical_declared_rows", -2))
        ):
            raise ValueError(f"Task feature projection violates label-blind binding: {dataset_key}")
        _require_exact_keys(projection, FEATURE_PROJECTION_KEYS, "feature projection status")
        strategies = task.get("strategies")
        if not isinstance(strategies, list):
            raise ValueError("Split-suite task lacks strategy accounting")
        names = [_clean(item.get("strategy")) for item in strategies if isinstance(item, Mapping)]
        if names != list(OFFICIAL_STRATEGIES):
            raise ValueError("Split-suite strategy enumeration is incomplete or reordered")
        if task.get("strategy_order") != list(OFFICIAL_STRATEGIES) or task.get(
            "mandatory_candidates"
        ) != sorted(MANDATORY_CANDIDATES):
            raise ValueError(f"Task strategy policy changed: {dataset_key}")
        mandatory_materialized = all(
            item.get("status") == "materialized"
            for item in strategies
            if isinstance(item, Mapping) and item.get("strategy") in MANDATORY_CANDIDATES
        )
        if task.get("mandatory_candidates_materialized") is not mandatory_materialized:
            raise ValueError(f"Task mandatory-candidate accounting changed: {dataset_key}")
        if task.get("claim_readiness") != TASK_CLAIM_READINESS:
            raise ValueError(f"Task claim-readiness policy changed: {dataset_key}")
        if (
            task.get("label_values_read") is not False
            or task.get("test_labels_disclosed") is not False
            or task.get("large_model_training_started") is not False
            or task.get("substantive_training_started") is not False
        ):
            raise ValueError(f"Task no-training/label boundary changed: {dataset_key}")
        observed["tasks_enumerated"] += 1
        for item in strategies:
            assert isinstance(item, Mapping)
            strategy = _clean(item.get("strategy"))
            if strategy not in OFFICIAL_STRATEGIES:
                raise ValueError(f"Unsupported strategy in task accounting: {strategy}")
            typed_strategy: Strategy = strategy  # type: ignore[assignment]
            strategy_status_relative = f"tasks/{slug}/strategies/{strategy}/status.json"
            if _bound_json(root, inventory, strategy_status_relative) != dict(item):
                raise ValueError(
                    f"Top strategy record differs from bound strategy status: {dataset_key}/{strategy}"
                )
            if (
                item.get("fixed_seed") != configuration.get("seed")
                or item.get("seed_search_performed") is not False
                or item.get("mandatory_candidate") is not (strategy in MANDATORY_CANDIDATES)
                or item.get("label_values_read") is not False
                or item.get("test_labels_disclosed") is not False
                or item.get("large_model_training_started") is not False
                or item.get("substantive_training_started") is not False
            ):
                raise ValueError(f"Strategy fixed scientific boundary changed: {dataset_key}/{strategy}")
            status = _clean(item.get("status"))
            if status == "materialized":
                recorded_applicable, recorded_reasons = _applicability(typed_strategy, projection)
                if not recorded_applicable or recorded_reasons:
                    raise ValueError(
                        f"Materialized strategy conflicts with recorded feature support: "
                        f"{dataset_key}/{strategy}"
                    )
                observed["strategies_materialized"] += 1
                _verify_materialized_strategy(root, inventory, task, item, configuration)
            elif status in {
                "skipped_inapplicable",
                "skipped_inapplicable_mandatory_candidate",
                "skipped_fixed_seed_support",
                "skipped_fixed_seed_mandatory_candidate",
            }:
                expected_skip_keys = (
                    BASE_STRATEGY_STATUS_KEYS | {"failure_class"}
                    if status.startswith("skipped_fixed_seed")
                    else BASE_STRATEGY_STATUS_KEYS
                )
                _require_exact_keys(
                    item,
                    frozenset(expected_skip_keys),
                    "skipped strategy status",
                )
                reasons = item.get("reasons")
                applicability = item.get("applicability")
                materialized_fields = {
                    "split_path",
                    "split_sha256",
                    "split_rows",
                    "partition_counts",
                    "sidecar_path",
                    "sidecar_sha256",
                    "leakage_diagnostics_path",
                    "leakage_diagnostics_sha256",
                    "exact_overlap_status",
                    "near_similarity_status",
                }
                if (
                    item.get("split_materialized") is not False
                    or not isinstance(reasons, list)
                    or not reasons
                    or not isinstance(applicability, Mapping)
                    or applicability.get("evaluated") is not True
                    or materialized_fields & set(item)
                ):
                    raise ValueError(f"Skipped strategy terminal semantics changed: {dataset_key}/{strategy}")
                recorded_applicable, recorded_reasons = _applicability(typed_strategy, projection)
                if status.startswith("skipped_inapplicable"):
                    expected_status = (
                        "skipped_inapplicable_mandatory_candidate"
                        if strategy in MANDATORY_CANDIDATES
                        else "skipped_inapplicable"
                    )
                    if (
                        status != expected_status
                        or recorded_applicable
                        or reasons != recorded_reasons
                        or dict(applicability)
                        != {"evaluated": True, "supported": False, "reasons": recorded_reasons}
                        or "failure_class" in item
                    ):
                        raise ValueError(
                            f"Skipped strategy applicability is not independently supported: {dataset_key}/{strategy}"
                        )
                else:
                    expected_status = (
                        "skipped_fixed_seed_mandatory_candidate"
                        if strategy in MANDATORY_CANDIDATES
                        else "skipped_fixed_seed_support"
                    )
                    if (
                        status != expected_status
                        or not recorded_applicable
                        or reasons != ["fixed_seed_partition_support_empty_no_seed_retry"]
                        or dict(applicability) != {"evaluated": True, "supported": True, "reasons": []}
                        or item.get("failure_class") != "expected_fixed_seed_support_limitation"
                    ):
                        raise ValueError(
                            f"Fixed-seed skip is not exactly supported: {dataset_key}/{strategy}"
                        )
                observed["strategies_skipped"] += 1
            else:
                raise ValueError(f"Invalid strategy terminal state: {status}")
            observed[f"strategy_{strategy}_{status}"] += 1
    if dict(sorted(observed.items())) != acceptance.get("accounting"):
        raise ValueError("Split-suite top accounting does not reconcile")


def _expected_output_topology(
    acceptance: Mapping[str, Any],
) -> tuple[set[str], set[str]]:
    tasks = acceptance.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("Split-suite topology requires at least one task")
    files = {"acceptance.json"}
    directories = {"tasks"}
    for raw_task in tasks:
        if not isinstance(raw_task, Mapping):
            raise ValueError("Split-suite topology contains a malformed task")
        dataset_key = _clean(raw_task.get("dataset_key"))
        slug = _task_slug(dataset_key)
        if raw_task.get("task_slug") != slug:
            raise ValueError(f"Split-suite topology has an unstable task slug: {dataset_key}")
        task_root = f"tasks/{slug}"
        directories.update({task_root, f"{task_root}/strategies"})
        files.update({f"{task_root}/features.parquet", f"{task_root}/task_status.json"})
        strategies = raw_task.get("strategies")
        if not isinstance(strategies, list):
            raise ValueError(f"Split-suite topology task lacks strategies: {dataset_key}")
        names = [_clean(item.get("strategy")) for item in strategies if isinstance(item, Mapping)]
        if names != list(OFFICIAL_STRATEGIES):
            raise ValueError(f"Split-suite topology strategy set changed: {dataset_key}")
        for raw_item in strategies:
            assert isinstance(raw_item, Mapping)
            strategy = _clean(raw_item.get("strategy"))
            strategy_root = f"{task_root}/strategies/{strategy}"
            directories.add(strategy_root)
            files.add(f"{strategy_root}/status.json")
            if raw_item.get("status") == "materialized":
                files.update(
                    {
                        f"{strategy_root}/split.parquet",
                        f"{strategy_root}/split.parquet.manifest.json",
                        f"{strategy_root}/leakage_diagnostics.json",
                    }
                )
    return files, directories


def _verify_root(root: Path, acceptance: Mapping[str, Any]) -> None:
    _reject_symlink_path_chain(root, "split-suite output")
    if acceptance.get("schema_version") != SPLIT_SUITE_SCHEMA_VERSION:
        raise ValueError("Unrecognized split-suite schema")
    _require_exact_keys(acceptance, TOP_ACCEPTANCE_KEYS, "top acceptance")
    if acceptance.get("substantive_training_started") is not False:
        raise ValueError("Split-suite acceptance violates no-training boundary")
    if acceptance.get("large_model_training_started") is not False:
        raise ValueError("Split-suite acceptance violates large-model no-training boundary")
    configuration = acceptance.get("configuration")
    _validated_configuration(configuration)
    source_binding = acceptance.get("source_binding")
    if not isinstance(source_binding, Mapping):
        raise ValueError("Split-suite source binding must be an object")
    _require_exact_keys(source_binding, SOURCE_BINDING_KEYS, "source binding")
    if not isinstance(configuration, Mapping) or stable_json_digest(configuration) != _clean(
        acceptance.get("configuration_sha256")
    ):
        raise ValueError("Split-suite configuration digest mismatch")
    inventory = acceptance.get("component_inventory")
    if not isinstance(inventory, Mapping) or stable_json_digest(inventory) != _clean(
        acceptance.get("component_inventory_sha256")
    ):
        raise ValueError("Split-suite component inventory digest mismatch")
    expected_files, expected_directories = _expected_output_topology(acceptance)
    actual: set[str] = set()
    actual_directories: set[str] = set()
    for path in root.rglob("*"):
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode):
            raise ValueError(f"Split-suite output contains a symlink: {path}")
        relative = path.relative_to(root).as_posix()
        if stat.S_ISREG(mode):
            actual.add(relative)
        elif stat.S_ISDIR(mode):
            actual_directories.add(relative)
        else:
            raise ValueError(f"Split-suite output contains a special filesystem entry: {path}")
    if actual != expected_files or actual_directories != expected_directories:
        raise ValueError(
            "Split-suite exact output topology mismatch; "
            f"unexpected_files={sorted(actual - expected_files)}, missing_files={sorted(expected_files - actual)}, "
            f"unexpected_dirs={sorted(actual_directories - expected_directories)}, "
            f"missing_dirs={sorted(expected_directories - actual_directories)}"
        )
    actual_components = actual - {"acceptance.json"}
    if actual_components != set(inventory):
        raise ValueError(
            "Split-suite recursive membership mismatch; "
            f"unbound={sorted(actual_components - set(inventory))}, "
            f"missing={sorted(set(inventory) - actual_components)}"
        )
    for relative, raw_entry in sorted(inventory.items()):
        if not isinstance(raw_entry, Mapping):
            raise ValueError(f"Malformed split-suite inventory entry: {relative}")
        parquet_component = Path(relative).suffix.casefold() == ".parquet"
        expected_entry_keys = frozenset(
            {"relative_path", "sha256", "size_bytes", "rows"}
            if parquet_component
            else {"relative_path", "sha256", "size_bytes"}
        )
        _require_exact_keys(
            raw_entry,
            expected_entry_keys,
            f"component inventory entry {relative}",
        )
        if raw_entry.get("relative_path") != relative:
            raise ValueError(f"Component inventory relative_path differs from its key: {relative}")
        digest = raw_entry.get("sha256")
        size = raw_entry.get("size_bytes")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"Component inventory SHA-256 is invalid: {relative}")
        if type(size) is not int or size < 0:
            raise ValueError(f"Component inventory byte size is invalid: {relative}")
        if parquet_component and (type(raw_entry.get("rows")) is not int or int(raw_entry["rows"]) < 0):
            raise ValueError(f"Component inventory Parquet row count is invalid: {relative}")
        path = root / _safe_relative(relative, "component path")
        if path.stat().st_size != size:
            raise ValueError(f"Split-suite component size mismatch: {relative}")
        if sha256_file(path) != digest:
            raise ValueError(f"Split-suite component SHA-256 mismatch: {relative}")
        if "rows" in raw_entry:
            metadata = pq.ParquetFile(path).metadata
            if metadata is None or int(metadata.num_rows) != int(raw_entry["rows"]):
                raise ValueError(f"Split-suite Parquet row mismatch: {relative}")
    _assert_portable(acceptance)
    _verify_json_boundaries(root)
    _verify_task_strategy_accounting(root, acceptance)


def materialize_split_suite(
    canonical_build_root: str | os.PathLike[str],
    qc_report_path: str | os.PathLike[str],
    output_directory: str | os.PathLike[str] = DEFAULT_OUTPUT_DIRECTORY,
    config: SplitSuiteConfig | None = None,
) -> dict[str, Any]:
    """Materialize all fixed-seed, feature-only split candidates transactionally."""

    config = config or SplitSuiteConfig()
    config.validate()
    binding = _verify_canonical_corpus(Path(canonical_build_root), Path(qc_report_path))
    tasks: list[dict[str, Any]] = []
    accounting: Counter[str] = Counter()
    with _transactional_directory(Path(output_directory)) as staging:
        for dataset_key, raw_entry in sorted(binding.task_datasets.items()):
            if not isinstance(raw_entry, Mapping):
                raise ValueError(f"Invalid canonical task entry: {dataset_key}")
            entry = dict(raw_entry)
            slug = _task_slug(dataset_key)
            task_root = staging / "tasks" / slug
            feature_path = task_root / "features.parquet"
            task_type = _clean(entry.get("task_type"))
            task_scope = _clean(entry.get("task_scope"))
            accounting["tasks_enumerated"] += 1
            strategies: list[dict[str, Any]] = []
            task_root.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix="routing-", dir=task_root.parent) as transient:
                routing_path = Path(transient) / "routing_projection.parquet"
                support = _materialize_feature_projection(
                    binding,
                    entry,
                    dataset_key,
                    feature_path,
                    routing_path,
                    batch_size=config.batch_size,
                )
                for raw_strategy in OFFICIAL_STRATEGIES:
                    strategy = raw_strategy  # narrowed below for static checkers
                    assert strategy in OFFICIAL_STRATEGIES
                    typed_strategy: Strategy = strategy  # type: ignore[assignment]
                    strategy_root = task_root / "strategies" / strategy
                    status_path = strategy_root / "status.json"
                    applicable, reasons = _applicability(typed_strategy, support)
                    base = {
                        "strategy": strategy,
                        "mandatory_candidate": strategy in MANDATORY_CANDIDATES,
                        "fixed_seed": config.seed,
                        "seed_search_performed": False,
                        "applicability": {
                            "evaluated": True,
                            "supported": applicable,
                            "reasons": reasons,
                        },
                    }
                    if not applicable:
                        status = {
                            **base,
                            "status": (
                                "skipped_inapplicable_mandatory_candidate"
                                if strategy in MANDATORY_CANDIDATES
                                else "skipped_inapplicable"
                            ),
                            "reasons": reasons,
                            "split_materialized": False,
                            "label_values_read": False,
                            "test_labels_disclosed": False,
                            "large_model_training_started": False,
                            "substantive_training_started": False,
                        }
                        _atomic_json(status_path, status)
                        strategies.append(status)
                        accounting["strategies_skipped"] += 1
                        accounting[f"strategy_{strategy}_{status['status']}"] += 1
                        continue
                    split_path = strategy_root / "split.parquet"
                    split_config = _split_config(typed_strategy, config, dataset_key)
                    try:
                        metadata = stream_hash_group_split_manifest(
                            routing_path,
                            split_path,
                            split_config,
                            batch_size=config.batch_size,
                        )
                    except ValueError as exc:
                        message = str(exc)
                        expected = (
                            "Stable hash split produced an empty partition" in message
                            or "Double-cold support is insufficient" in message
                        )
                        if not expected:
                            raise
                        reasons = ["fixed_seed_partition_support_empty_no_seed_retry"]
                        status = {
                            **base,
                            "status": (
                                "skipped_fixed_seed_mandatory_candidate"
                                if strategy in MANDATORY_CANDIDATES
                                else "skipped_fixed_seed_support"
                            ),
                            "reasons": reasons,
                            "split_materialized": False,
                            "failure_class": "expected_fixed_seed_support_limitation",
                            "label_values_read": False,
                            "test_labels_disclosed": False,
                            "large_model_training_started": False,
                            "substantive_training_started": False,
                        }
                        _atomic_json(status_path, status)
                        strategies.append(status)
                        accounting["strategies_skipped"] += 1
                        accounting[f"strategy_{strategy}_{status['status']}"] += 1
                        continue
                    sidecar_path = split_path.with_suffix(split_path.suffix + ".manifest.json")
                    sidecar = _rewrite_split_sidecar(
                        sidecar_path,
                        metadata,
                        dataset_key=dataset_key,
                        canonical_dataset_sha256=_clean(entry.get("dataset_sha256")),
                        feature_relative_path=feature_path.relative_to(staging).as_posix(),
                        feature_sha256=_clean(support["feature_file_sha256"]),
                    )
                    leakage = _leakage_diagnostics(feature_path, split_path, typed_strategy, config)
                    leakage_path = strategy_root / "leakage_diagnostics.json"
                    _atomic_json(leakage_path, leakage)
                    status = {
                        **base,
                        "status": "materialized",
                        "reasons": [],
                        "split_materialized": True,
                        "split_path": split_path.relative_to(staging).as_posix(),
                        "split_sha256": sha256_file(split_path),
                        "split_rows": int(sidecar["record_count"]),
                        "partition_counts": sidecar["partition_counts"],
                        "sidecar_path": sidecar_path.relative_to(staging).as_posix(),
                        "sidecar_sha256": sha256_file(sidecar_path),
                        "leakage_diagnostics_path": leakage_path.relative_to(staging).as_posix(),
                        "leakage_diagnostics_sha256": sha256_file(leakage_path),
                        "exact_overlap_status": "passed_exhaustive",
                        "near_similarity_status": "sampled_non_exhaustive_not_claim_ready",
                        "label_values_read": False,
                        "test_labels_disclosed": False,
                        "large_model_training_started": False,
                        "substantive_training_started": False,
                    }
                    _atomic_json(status_path, status)
                    strategies.append(status)
                    accounting["strategies_materialized"] += 1
                    accounting[f"strategy_{strategy}_materialized"] += 1
            task_status = {
                "schema_version": SPLIT_SUITE_SCHEMA_VERSION,
                "dataset_key": dataset_key,
                "task_scope": task_scope,
                "task_type": task_type,
                "task_slug": slug,
                "canonical_task_dataset_sha256": _clean(entry.get("dataset_sha256")),
                "canonical_declared_rows": int(entry.get("row_count", -1)),
                "feature_projection_path": feature_path.relative_to(staging).as_posix(),
                "feature_projection": support,
                "strategies": strategies,
                "strategy_order": list(OFFICIAL_STRATEGIES),
                "mandatory_candidates": sorted(MANDATORY_CANDIDATES),
                "mandatory_candidates_materialized": all(
                    record["status"] == "materialized"
                    for record in strategies
                    if record["strategy"] in MANDATORY_CANDIDATES
                ),
                "claim_readiness": TASK_CLAIM_READINESS,
                "label_values_read": False,
                "test_labels_disclosed": False,
                "large_model_training_started": False,
                "substantive_training_started": False,
            }
            _atomic_json(task_root / "task_status.json", task_status)
            tasks.append(task_status)
        inventory = _component_inventory(staging)
        acceptance = {
            "schema_version": SPLIT_SUITE_SCHEMA_VERSION,
            "configuration": asdict(config),
            "configuration_sha256": stable_json_digest(asdict(config)),
            "source_binding": {
                "canonical_build_manifest_sha256": binding.build_manifest_sha256,
                "canonical_component_inventory_sha256": binding.component_inventory_sha256,
                "canonical_qc_report_sha256": binding.qc_report_sha256,
                "task_datasets_file_sha256": binding.task_datasets_sha256,
                "task_datasets_payload_sha256": _clean(
                    binding.build_manifest.get("task_datasets_manifest_sha256")
                ),
                "source_id": binding.build_manifest.get("source_id"),
                "snapshot_id": binding.build_manifest.get("snapshot_id"),
            },
            "task_order": [record["dataset_key"] for record in tasks],
            "tasks": tasks,
            "strategy_order": list(OFFICIAL_STRATEGIES),
            "mandatory_candidates": sorted(MANDATORY_CANDIDATES),
            "accounting": dict(sorted(accounting.items())),
            "label_access_contract": _label_access_contract(),
            "test_feature_access_policy": TEST_FEATURE_ACCESS_POLICY,
            "exact_overlap_evidence": EXACT_OVERLAP_EVIDENCE,
            "near_similarity_evidence": NEAR_SIMILARITY_EVIDENCE,
            "claim_readiness": TOP_CLAIM_READINESS,
            "no_seed_search": True,
            "component_inventory": inventory,
            "component_inventory_sha256": stable_json_digest(inventory),
            "large_model_training_started": False,
            "substantive_training_started": False,
        }
        _assert_portable(acceptance)
        _atomic_json(staging / "acceptance.json", acceptance)
        _verify_root(staging, acceptance)
    root = Path(output_directory).resolve()
    result = _load_strict_json(root / "acceptance.json")
    _verify_root(root, result)
    result["acceptance_file_sha256_external"] = sha256_file(root / "acceptance.json")
    return result


def _rebind_feature_projections_and_fixed_skips(
    root: Path,
    binding: CanonicalCorpusBinding,
    acceptance: Mapping[str, Any],
    config: SplitSuiteConfig,
) -> None:
    tasks = {
        _clean(record.get("dataset_key")): record
        for record in acceptance["tasks"]
        if isinstance(record, Mapping)
    }
    with tempfile.TemporaryDirectory(prefix="split-suite-source-rebind-") as temporary:
        temporary_root = Path(temporary)
        for dataset_key, raw_entry in sorted(binding.task_datasets.items()):
            if not isinstance(raw_entry, Mapping):
                raise ValueError(f"Canonical task entry is malformed during projection rebind: {dataset_key}")
            task = tasks[dataset_key]
            slug = _task_slug(dataset_key)
            task_temporary = temporary_root / slug
            expected_feature = task_temporary / "features.parquet"
            routing = task_temporary / "routing.parquet"
            support = _materialize_feature_projection(
                binding,
                raw_entry,
                dataset_key,
                expected_feature,
                routing,
                batch_size=config.batch_size,
            )
            published_feature = root / _safe_relative(
                task.get("feature_projection_path"), "published feature projection"
            )
            if sha256_file(expected_feature) != sha256_file(published_feature) or support != task.get(
                "feature_projection"
            ):
                raise ValueError(
                    f"Published feature projection differs from canonical feature-only regeneration: {dataset_key}"
                )
            strategies = task.get("strategies")
            assert isinstance(strategies, list)
            for raw_item in strategies:
                assert isinstance(raw_item, Mapping)
                strategy_text = _clean(raw_item.get("strategy"))
                assert strategy_text in OFFICIAL_STRATEGIES
                strategy: Strategy = strategy_text  # type: ignore[assignment]
                applicable, reasons = _applicability(strategy, support)
                status = _clean(raw_item.get("status"))
                if status == "materialized":
                    if not applicable or reasons:
                        raise ValueError(
                            f"Materialized split is inapplicable under regenerated canonical features: "
                            f"{dataset_key}/{strategy}"
                        )
                    trial_output = task_temporary / f"materialized-check-{strategy}.parquet"
                    trial_metadata = stream_hash_group_split_manifest(
                        routing,
                        trial_output,
                        _split_config(strategy, config, dataset_key),
                        batch_size=config.batch_size,
                    )
                    published_split = root / _safe_relative(
                        raw_item.get("split_path"), "published materialized split"
                    )
                    if sha256_file(trial_output) != sha256_file(published_split):
                        raise ValueError(
                            f"Published materialized split differs from deterministic fixed-seed "
                            f"regeneration: {dataset_key}/{strategy}"
                        )
                    trial_sidecar_path = trial_output.with_suffix(trial_output.suffix + ".manifest.json")
                    trial_sidecar = _rewrite_split_sidecar(
                        trial_sidecar_path,
                        trial_metadata,
                        dataset_key=dataset_key,
                        canonical_dataset_sha256=_clean(raw_entry.get("dataset_sha256")),
                        feature_relative_path=_clean(task.get("feature_projection_path")),
                        feature_sha256=sha256_file(expected_feature),
                    )
                    trial_sidecar["manifest_path"] = "split.parquet"
                    published_sidecar = _load_strict_json(
                        root / _safe_relative(raw_item.get("sidecar_path"), "published materialized sidecar")
                    )
                    if trial_sidecar != published_sidecar:
                        raise ValueError(
                            f"Published split sidecar differs from deterministic regeneration: "
                            f"{dataset_key}/{strategy}"
                        )
                    regenerated_leakage = _leakage_diagnostics(
                        expected_feature,
                        trial_output,
                        strategy,
                        config,
                    )
                    published_leakage = _load_strict_json(
                        root
                        / _safe_relative(
                            raw_item.get("leakage_diagnostics_path"),
                            "published leakage diagnostics",
                        )
                    )
                    if regenerated_leakage != published_leakage:
                        raise ValueError(
                            f"Published leakage diagnostics differ from deterministic regeneration: "
                            f"{dataset_key}/{strategy}"
                        )
                    continue
                if status.startswith("skipped_inapplicable"):
                    if applicable or raw_item.get("reasons") != reasons:
                        raise ValueError(
                            f"Inapplicable split reason differs under canonical regeneration: "
                            f"{dataset_key}/{strategy}"
                        )
                    continue
                if status not in {
                    "skipped_fixed_seed_support",
                    "skipped_fixed_seed_mandatory_candidate",
                }:
                    raise ValueError(f"Unrecognized skip during canonical rebind: {status}")
                if not applicable:
                    raise ValueError(
                        f"Fixed-seed skip is actually feature-inapplicable: {dataset_key}/{strategy}"
                    )
                trial_output = task_temporary / f"fixed-seed-check-{strategy}.parquet"
                try:
                    stream_hash_group_split_manifest(
                        routing,
                        trial_output,
                        _split_config(strategy, config, dataset_key),
                        batch_size=config.batch_size,
                    )
                except ValueError as exc:
                    message = str(exc)
                    if not (
                        "Stable hash split produced an empty partition" in message
                        or "Double-cold support is insufficient" in message
                    ):
                        raise ValueError(
                            f"Fixed-seed skip failed for an unexpected reason: {dataset_key}/{strategy}"
                        ) from exc
                else:
                    raise ValueError(
                        f"Fixed-seed skipped strategy is materializable without seed search: "
                        f"{dataset_key}/{strategy}"
                    )


def verify_split_suite(
    output_directory: str | os.PathLike[str] = DEFAULT_OUTPUT_DIRECTORY,
    *,
    canonical_build_root: str | os.PathLike[str] | None = None,
    qc_report_path: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Rehash a split suite and optionally rebind it to accepted canonical inputs."""

    requested_root = Path(output_directory)
    _reject_symlink_path_chain(requested_root, "split-suite output")
    root = requested_root.resolve()
    acceptance_path = root / "acceptance.json"
    if not acceptance_path.is_file():
        raise FileNotFoundError(acceptance_path)
    acceptance = _load_strict_json(acceptance_path)
    _verify_root(root, acceptance)
    if (canonical_build_root is None) != (qc_report_path is None):
        raise ValueError("canonical_build_root and qc_report_path must be supplied together")
    source_reverified = False
    if canonical_build_root is not None and qc_report_path is not None:
        binding = _verify_canonical_corpus(Path(canonical_build_root), Path(qc_report_path))
        config = _validated_configuration(acceptance.get("configuration"))
        source = acceptance.get("source_binding")
        if not isinstance(source, Mapping):
            raise ValueError("Split-suite acceptance lacks source binding")
        expected = {
            "canonical_build_manifest_sha256": binding.build_manifest_sha256,
            "canonical_component_inventory_sha256": binding.component_inventory_sha256,
            "canonical_qc_report_sha256": binding.qc_report_sha256,
            "task_datasets_file_sha256": binding.task_datasets_sha256,
            "task_datasets_payload_sha256": _clean(
                binding.build_manifest.get("task_datasets_manifest_sha256")
            ),
            "source_id": _clean(binding.build_manifest.get("source_id")),
            "snapshot_id": _clean(binding.build_manifest.get("snapshot_id")),
        }
        if any(_clean(source.get(key)) != value for key, value in expected.items()):
            raise ValueError("Split-suite source binding no longer matches canonical inputs")
        accepted_keys = [record["dataset_key"] for record in acceptance["tasks"]]
        if accepted_keys != sorted(binding.task_datasets):
            raise ValueError("Split-suite task enumeration no longer matches canonical manifest")
        accepted_tasks = {
            _clean(record.get("dataset_key")): record
            for record in acceptance["tasks"]
            if isinstance(record, Mapping)
        }
        for dataset_key, raw_entry in sorted(binding.task_datasets.items()):
            if not isinstance(raw_entry, Mapping):
                raise ValueError(f"Canonical task entry is malformed during rebind: {dataset_key}")
            record = accepted_tasks.get(dataset_key)
            if record is None or (
                record.get("task_scope") != raw_entry.get("task_scope")
                or record.get("task_type") != raw_entry.get("task_type")
                or record.get("canonical_task_dataset_sha256") != raw_entry.get("dataset_sha256")
                or int(record.get("canonical_declared_rows", -1)) != int(raw_entry.get("row_count", -2))
                or int(record.get("feature_projection", {}).get("rows", -1))
                != int(raw_entry.get("row_count", -2))
            ):
                raise ValueError(
                    f"Split-suite task semantics no longer match canonical manifest: {dataset_key}"
                )
        _rebind_feature_projections_and_fixed_skips(root, binding, acceptance, config)
        source_reverified = True
    return {
        "schema_version": SPLIT_SUITE_SCHEMA_VERSION,
        "status": "verified",
        "acceptance_file_sha256": sha256_file(acceptance_path),
        "component_count": len(acceptance["component_inventory"]),
        "accounting": acceptance["accounting"],
        "source_reverified": source_reverified,
        "label_values_read": False,
        "test_labels_disclosed": False,
        "large_model_training_started": False,
        "substantive_training_started": False,
    }


__all__ = [
    "DEFAULT_OUTPUT_DIRECTORY",
    "SPLIT_SUITE_SCHEMA_VERSION",
    "SplitSuiteConfig",
    "materialize_split_suite",
    "verify_split_suite",
]
