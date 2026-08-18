"""Transactional DATA-to-MODEL readiness bundles and capped diagnostics.

The APIs in this module perform deterministic preprocessing, loader smoke
checks, and explicitly lightweight diagnostic baselines.  They never launch
substantive pretrained-model training and never evaluate a diagnostic model on
the physically separated test partition.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import math
import os
import shutil
import tempfile
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
from scipy import sparse

from .features import fingerprint_matrix
from .platform_baselines import (
    BaselineConfig,
    build_error_analysis,
    class_imbalance_options,
    numeric_target_leakage_scan,
    run_diagnostic_baselines,
    run_identifier_hash_control,
    run_label_permutation_control,
)
from .platform_features import (
    deterministic_descriptor_frame,
    feature_failure_summary,
    stable_json_digest,
    tokenize_smiles,
)
from .platform_pretraining import (
    CollatorConfig,
    JsonlIterableDataset,
    MultimodalCollator,
    Vocabulary,
    build_training_vocabulary,
    estimate_model_ready_materialization_memory,
    file_sha256,
    serialize_model_ready_jsonl_streaming,
    streaming_loader_smoke_test,
)
from .platform_splits import (
    ResolvedParquetDataset,
    SplitConfig,
    resolve_manifest_bound_parquet_dataset,
    stream_hash_group_split_manifest,
)

INTEGRATION_BUNDLE_SCHEMA_VERSION = "platform_model_integration_bundle_v1"
DIAGNOSTIC_BUNDLE_SCHEMA_VERSION = "platform_model_diagnostic_bundle_v1"
NO_SUBSTANTIVE_TRAINING = False
TaskEligibilityMode = Literal["default", "derived_sensitivity"]
TaskFamily = Literal["regression", "classification"]


@dataclass(frozen=True)
class TaskIntegrationConfig:
    """Configuration for one immutable task-readiness bundle."""

    split_name: str = "molecule_hash_stream_v1"
    split_strategy: Literal[
        "molecule_grouped",
        "scaffold",
        "source_holdout",
        "protein_holdout",
        "target_holdout",
        "double_cold",
    ] = "molecule_grouped"
    intended_use: str = "new molecule within the observed public task domain at platform scale"
    seed: int = 20260804
    split_batch_size: int = 50_000
    serialization_batch_size: int = 8_192
    task_eligibility_mode: TaskEligibilityMode = "default"
    loader_batch_size: int = 8
    loader_maximum_batches: int = 4
    require_manifest_bound_directory: bool = True

    def validate(self) -> None:
        if not self.split_name.strip() or not self.intended_use.strip():
            raise ValueError("Split name and intended use must be nonblank")
        if not 1 <= self.split_batch_size <= 250_000:
            raise ValueError("split_batch_size must be between 1 and 250000")
        if not 1 <= self.serialization_batch_size <= 65_536:
            raise ValueError("serialization_batch_size must be between 1 and 65536")
        if (
            self.loader_batch_size < 1
            or self.loader_maximum_batches < 1
            or self.loader_batch_size * self.loader_maximum_batches > 32
        ):
            raise ValueError("Loader smoke must process between 1 and 32 examples")


@dataclass(frozen=True)
class DiagnosticConfig:
    """Fixed, capped diagnostic settings; no hyperparameter search is implied."""

    seed: int = 20260804
    maximum_train_examples: int = 50_000
    maximum_validation_examples: int = 10_000
    fingerprint_bits: int = 2_048
    fingerprint_radius: int = 2
    binary_label_mapping: tuple[tuple[str, int], ...] = ()

    def validate(self) -> None:
        if not 2 <= self.maximum_train_examples <= 50_000:
            raise ValueError("maximum_train_examples must be between 2 and the fixed 50000 cap")
        if not 1 <= self.maximum_validation_examples <= 10_000:
            raise ValueError("maximum_validation_examples must be between 1 and the fixed 10000 cap")
        if not 64 <= self.fingerprint_bits <= 8_192 or not 1 <= self.fingerprint_radius <= 4:
            raise ValueError("Fingerprint width/radius are invalid")
        mapping = dict(self.binary_label_mapping)
        if len(mapping) != len(self.binary_label_mapping):
            raise ValueError("binary_label_mapping keys must be unique")
        if mapping and set(mapping.values()) != {0, 1}:
            raise ValueError("An explicit binary mapping must define both encoded classes 0 and 1")


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return value.as_posix()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    if payload.get("substantive_training_started") is not NO_SUBSTANTIVE_TRAINING:
        raise ValueError("Every integration JSON must state substantive_training_started=false")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(
                payload,
                handle,
                indent=2,
                sort_keys=True,
                allow_nan=False,
                default=_json_default,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        if path.suffix == ".parquet":
            frame.to_parquet(temporary, index=False, compression="zstd")
        elif path.suffix == ".csv":
            frame.to_csv(temporary, index=False)
        else:
            raise ValueError(f"Unsupported frame artifact suffix: {path.suffix}")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_sparse_matrix(matrix: sparse.csr_matrix, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp.npz")
    try:
        sparse.save_npz(temporary, matrix, compressed=True)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_joblib(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        joblib.dump(value, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def _transactional_directory(output_directory: Path) -> Iterator[Path]:
    """Build a new directory beside its destination and commit with one rename."""

    output = output_directory.resolve()
    if not output.name or output == Path(output.anchor):
        raise ValueError("Refusing a broad or root integration output directory")
    if output.exists():
        raise FileExistsError(f"Immutable output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.building-", dir=output.parent)).resolve()
    committed = False
    try:
        yield staging
        if output.exists():
            raise FileExistsError(f"Output appeared during transaction: {output}")
        os.replace(staging, output)
        committed = True
    finally:
        if not committed and staging.exists():
            shutil.rmtree(staging)


def _verify_custom_json_flags(root: Path) -> None:
    for path in root.rglob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("substantive_training_started") is not NO_SUBSTANTIVE_TRAINING:
            raise RuntimeError(f"Generated JSON omits the no-training declaration: {path}")


def _component_inventory(
    root: Path,
    *,
    excluded_relative_paths: Sequence[str] = (),
    prebound_entries: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Inventory every component without reopening explicitly prebound lockboxes."""

    excluded = set(excluded_relative_paths)
    prebound = dict(prebound_entries or {})
    inventory: dict[str, dict[str, Any]] = {}
    actual: set[str] = set()
    for path in sorted(root.rglob("*"), key=lambda candidate: candidate.as_posix()):
        if path.is_symlink():
            raise ValueError(f"Artifact bundles may not contain symlinks: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative in excluded:
            continue
        actual.add(relative)
        if relative in prebound:
            entry = dict(prebound[relative])
            if entry.get("relative_path") != relative:
                raise ValueError(f"Prebound artifact path is inconsistent: {relative}")
            digest = str(entry.get("sha256", ""))
            if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
                raise ValueError(f"Prebound artifact digest is invalid: {relative}")
            if int(entry.get("size_bytes", -1)) < 0:
                raise ValueError(f"Prebound artifact size is invalid: {relative}")
            inventory[relative] = entry
            continue
        inventory[relative] = {
            "relative_path": relative,
            "sha256": file_sha256(path),
            "size_bytes": path.stat().st_size,
            "verification_source": "post_materialization_sha256",
        }
    if set(prebound) - actual:
        raise ValueError(f"Prebound artifacts are absent from the bundle: {sorted(set(prebound) - actual)}")
    return dict(sorted(inventory.items()))


def _verify_component_inventory(
    root: Path,
    inventory: Mapping[str, Mapping[str, Any]],
    *,
    inventory_sha256: str,
    excluded_relative_paths: Sequence[str] = (),
    do_not_rehash: Sequence[str] = (),
) -> None:
    """Require exact file membership and digest/size agreement before commit."""

    excluded = set(excluded_relative_paths)
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.relative_to(root).as_posix() not in excluded
    }
    if actual != set(inventory):
        raise RuntimeError(
            "Component inventory membership mismatch; "
            f"unbound={sorted(actual - set(inventory))}, missing={sorted(set(inventory) - actual)}"
        )
    if stable_json_digest(inventory) != inventory_sha256:
        raise RuntimeError("Component inventory digest does not match its payload")
    locked = set(do_not_rehash)
    for relative, entry in inventory.items():
        if entry.get("relative_path") != relative:
            raise RuntimeError(f"Component inventory relative path mismatch: {relative}")
        if relative in locked:
            if entry.get("verification_source") != "bound_during_physical_routing_no_reopen":
                raise RuntimeError(f"Locked component lacks its routing-time binding: {relative}")
            continue
        path = root / relative
        if path.stat().st_size != int(entry["size_bytes"]):
            raise RuntimeError(f"Component size mismatch before commit: {relative}")
        if file_sha256(path) != str(entry["sha256"]):
            raise RuntimeError(f"Component digest mismatch before commit: {relative}")


def _assert_final_manifest_bound_source(
    task_dataset_path: Path,
    *,
    required: bool,
) -> ResolvedParquetDataset:
    source = resolve_manifest_bound_parquet_dataset(task_dataset_path)
    if required and source.input_kind != "manifest_bound_directory":
        raise ValueError("Task integration requires a manifest-bound partitioned directory")
    if source.manifest_path is not None and any(
        part.startswith(".") and ".building" in part for part in source.manifest_path.parts
    ):
        raise ValueError("Refusing a provisional .building canonical dataset")
    if source.manifest_path is not None:
        final_root = source.manifest_path.parent
        sibling_build = final_root.parent / f".{final_root.name}.building"
        if sibling_build.exists():
            raise ValueError(f"Refusing final DATA while a provisional sibling build exists: {sibling_build}")
    if any(part.startswith(".") and ".building" in part for part in source.input_path.parts):
        raise ValueError("Refusing a provisional .building task directory")
    return source


def _derive_task_semantics(
    source: ResolvedParquetDataset,
    *,
    batch_size: int,
) -> dict[str, Any]:
    """Scan only semantic columns and require one task contract across all parts."""

    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - platform dependency profile.
        raise ImportError("pyarrow is required for task semantic inspection") from exc
    columns = (
        "task_id",
        "task_type",
        "label_kind",
        "label_unit",
        "observation_kind",
        "evidence_domain",
        "endpoint",
        "assay_family",
    )
    values: dict[str, set[str]] = {column: set() for column in columns}
    rows = 0
    for part in source.parts:
        parquet = pq.ParquetFile(part.path)
        missing = sorted(set(columns) - set(parquet.schema_arrow.names))
        if missing:
            raise ValueError(f"Task part {part.relative_path!r} lacks semantic columns: {missing}")
        for batch in parquet.iter_batches(batch_size=batch_size, columns=list(columns)):
            frame = batch.to_pandas()
            rows += len(frame)
            for column in columns:
                observed = set(frame[column].fillna("").astype(str).str.strip().tolist())
                values[column].update(observed)
                if len(values[column]) > 1 or "" in values[column]:
                    raise ValueError(
                        f"Canonical task is heterogeneous or blank in {column}: {sorted(values[column])[:5]}"
                    )
    if rows != source.total_rows or rows < 1:
        raise ValueError(f"Semantic scan row count mismatch: expected={source.total_rows}, observed={rows}")
    signature = {column: next(iter(items)) for column, items in values.items()}
    label_kind = signature["label_kind"]
    if label_kind in {"categorical", "ordinal"}:
        family: TaskFamily = "classification"
    elif label_kind in {"continuous_exact", "continuous_censored"}:
        family = "regression"
    else:
        raise ValueError(f"Unsupported canonical label_kind: {label_kind!r}")
    return {
        **signature,
        "resolved_task_family": family,
        "decision_rule": ("categorical_or_ordinal=>classification; continuous_exact_or_censored=>regression"),
        "semantic_scan_record_count": rows,
        "semantic_scan_complete": True,
    }


def _amend_split_sidecar_no_training(split_path: Path) -> dict[str, Any]:
    sidecar = split_path.with_suffix(split_path.suffix + ".manifest.json")
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    payload["substantive_training_started"] = NO_SUBSTANTIVE_TRAINING
    _atomic_json(sidecar, payload)
    return {
        "path": sidecar,
        "sha256": file_sha256(sidecar),
        "payload": payload,
    }


def _partition_model_ready_jsonl(
    source_path: Path,
    output_directory: Path,
    *,
    expected_sha256: str,
) -> dict[str, Any]:
    """Physically route immutable canonical JSONL lines into three lockbox files."""

    if file_sha256(source_path) != expected_sha256:
        raise ValueError("Combined model-ready JSONL digest changed before physical routing")
    output_directory.mkdir(parents=True, exist_ok=False)
    names = ("train", "validation", "test")
    temporary = {name: output_directory / f".{name}.jsonl.tmp" for name in names}
    final = {name: output_directory / f"{name}.jsonl" for name in names}
    handles = {name: temporary[name].open("w", encoding="utf-8", newline="\n") for name in names}
    counts: Counter[str] = Counter()
    try:
        with source_path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                payload = json.loads(line)
                split = payload.get("split")
                partition = split.get("partition") if isinstance(split, Mapping) else None
                if partition not in handles:
                    raise ValueError(f"Unexpected partition at model-ready line {line_number}: {partition!r}")
                handles[partition].write(line if line.endswith("\n") else line + "\n")
                counts[str(partition)] += 1
        for handle in handles.values():
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
        if any(counts[name] == 0 for name in names):
            raise ValueError(f"Physical JSONL routing produced an empty partition: {dict(counts)}")
        for name in names:
            os.replace(temporary[name], final[name])
    finally:
        for handle in handles.values():
            if not handle.closed:
                handle.close()
        for path in temporary.values():
            path.unlink(missing_ok=True)

    metadata = {
        "schema_version": "physical_model_ready_partitions_v1",
        "transient_combined_source": {
            "retained_in_final_bundle": False,
            "sha256": expected_sha256,
            "record_count": sum(counts.values()),
        },
        "partition_counts": dict(sorted(counts.items())),
        "partitions": {
            name: {
                "path": final[name].name,
                "sha256": file_sha256(final[name]),
                "size_bytes": final[name].stat().st_size,
                "record_count": counts[name],
            }
            for name in names
        },
        "lockbox_policy": (
            "training receives train; model selection receives train+validation; "
            "test remains physically separate"
        ),
        "substantive_training_started": NO_SUBSTANTIVE_TRAINING,
    }
    manifest_path = output_directory / "partition_manifest.json"
    _atomic_json(manifest_path, metadata)
    return {
        **metadata,
        "external_manifest_file_sha256": file_sha256(manifest_path),
    }


def _retire_transient_model_ready_corpus(
    model_path: Path,
    model_metadata: Mapping[str, Any],
    *,
    partition_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Delete the routed all-partition corpus and replace metadata with a receipt."""

    sidecar_path = model_path.with_suffix(model_path.suffix + ".manifest.json")
    if not model_path.is_file() or not sidecar_path.is_file():
        raise RuntimeError("Transient combined model-ready corpus or sidecar is missing")
    if str(model_metadata.get("manifest_sha256", "")) != file_sha256(sidecar_path):
        raise RuntimeError("Transient model-ready sidecar changed before retirement")
    if int(model_metadata["record_count"]) != sum(
        int(value) for value in partition_metadata["partition_counts"].values()
    ):
        raise RuntimeError("Cannot retire a combined corpus before partition counts reconcile")
    receipt = {
        "schema_version": "transient_model_ready_serialization_receipt_v1",
        "transient_combined_artifact": {
            "retained_in_final_bundle": False,
            "removed_after_verified_physical_routing": True,
            "sha256": str(model_metadata["file_sha256"]),
            "record_count": int(model_metadata["record_count"]),
            "ordered_line_digest_sha256": str(model_metadata["ordered_line_digest_sha256"]),
            "serialization_sidecar_sha256_before_removal": str(model_metadata["manifest_sha256"]),
        },
        "source_dataset": model_metadata["source_dataset"],
        "source_dataset_sha256": str(model_metadata["source_dataset_sha256"]),
        "split_manifest_sha256": str(model_metadata["split_manifest_sha256"]),
        "split_sidecar_sha256": str(model_metadata["split_sidecar_sha256"]),
        "build_config": model_metadata["build_config"],
        "build_config_sha256": str(model_metadata["build_config_sha256"]),
        "serialization_implementation": str(model_metadata["serialization_implementation"]),
        "serialization_format": str(model_metadata["serialization_format"]),
        "partition_counts": model_metadata["partition_counts"],
        "excluded_partition_counts": model_metadata["excluded_partition_counts"],
        "outcome_kind_counts": model_metadata["outcome_kind_counts"],
        "bounded_memory": model_metadata["bounded_memory"],
        "physical_partition_manifest_file_sha256": str(partition_metadata["external_manifest_file_sha256"]),
        "final_payload_policy": (
            "only physically separated train validation and sealed test JSONL payloads are retained"
        ),
        "substantive_training_started": NO_SUBSTANTIVE_TRAINING,
    }
    model_path.unlink()
    sidecar_path.unlink()
    receipt_path = model_path.parent / "combined_serialization_receipt.json"
    _atomic_json(receipt_path, receipt)
    return {
        "path": receipt_path,
        "sha256": file_sha256(receipt_path),
        "payload": receipt,
    }


def _length_quantile(counter: Counter[int], fraction: float) -> int:
    total = sum(counter.values())
    if total < 1:
        raise ValueError("Cannot calculate a length quantile from an empty partition")
    target = max(1, math.ceil(total * fraction))
    cumulative = 0
    for length, count in sorted(counter.items()):
        cumulative += count
        if cumulative >= target:
            return int(length)
    raise RuntimeError("Length quantile calculation did not terminate")


def _length_inventory(
    partition_metadata: Mapping[str, Any],
    partition_directory: Path,
) -> dict[str, Any]:
    counters: dict[str, dict[str, Counter[int]]] = {
        partition: {modality: Counter() for modality in ("smiles", "protein", "text")}
        for partition in ("train", "validation")
    }
    for partition in ("train", "validation"):
        artifact = partition_metadata["partitions"][partition]
        dataset = JsonlIterableDataset(
            partition_directory / str(artifact["path"]),
            expected_sha256=str(artifact["sha256"]),
        )
        for example in dataset:
            inputs = example["inputs"]
            counters[str(partition)]["smiles"][len(tokenize_smiles(inputs.get("smiles", "")))] += 1
            counters[str(partition)]["protein"][len(str(inputs.get("protein_sequence", "")))] += 1
            counters[str(partition)]["text"][len(str(inputs.get("text", "")).split())] += 1

    candidate_limits = {
        "smiles": (128, 256, 512, 1024),
        "protein": (512, 1024, 2048, 4096),
        "text": (8, 32, 128, 256),
    }
    report: dict[str, Any] = {}
    for partition, modalities in counters.items():
        report[partition] = {}
        for modality, counter in modalities.items():
            total = sum(counter.values())
            report[partition][modality] = {
                "n": total,
                "n_nonempty": total - counter.get(0, 0),
                "minimum": min(counter),
                "median": _length_quantile(counter, 0.50),
                "p90": _length_quantile(counter, 0.90),
                "p95": _length_quantile(counter, 0.95),
                "p99": _length_quantile(counter, 0.99),
                "maximum": max(counter),
                "candidate_limit_affected": {
                    str(limit): sum(count for length, count in counter.items() if length + 2 > limit)
                    for limit in candidate_limits[modality]
                },
            }
    report["test"] = {
        "status": "not_inspected_locked_test",
        "record_count_from_routing_manifest": int(partition_metadata["partitions"]["test"]["record_count"]),
        "length_statistics": None,
        "selection_use": "prohibited",
    }
    return {
        "schema_version": "streaming_length_inventory_v1",
        "special_tokens_counted_in_candidate_limit_affected": 2,
        "selection_policy": (
            "train_statistics_may_inform_smoke_wiring; validation_is_report_only; "
            "test_is_not_opened_hashed_or_iterated_after_physical_routing"
        ),
        "opened_partitions": ["train", "validation"],
        "test_partition_opened_after_routing": False,
        "test_partition_hashed_after_routing": False,
        "test_partition_iterated_after_routing": False,
        "partitions": report,
        "substantive_training_started": NO_SUBSTANTIVE_TRAINING,
    }


def _training_vocabularies(
    train_path: Path,
    *,
    train_sha256: str,
    source_model_ready_sha256: str,
    output_directory: Path,
) -> dict[str, Vocabulary]:
    dataset = JsonlIterableDataset(train_path, expected_sha256=train_sha256)
    vocabularies: dict[str, Vocabulary] = {}
    modalities: tuple[
        tuple[Literal["smiles", "protein", "text"], str],
        ...,
    ] = (
        ("smiles", "smiles"),
        ("protein", "protein_sequence"),
        ("text", "text"),
    )
    for modality, key in modalities:
        vocabulary = build_training_vocabulary(
            (example["inputs"].get(key, "") for example in dataset),
            modality=modality,
            fitted_partition="train",
        )
        vocabularies[modality] = vocabulary
        _atomic_json(
            output_directory / f"vocabulary_{modality}.json",
            {
                "vocabulary": asdict(vocabulary),
                "vocabulary_sha256": vocabulary.digest(),
                "training_partition_sha256": train_sha256,
                "source_model_ready_sha256": source_model_ready_sha256,
                "substantive_training_started": NO_SUBSTANTIVE_TRAINING,
            },
        )
    return vocabularies


def materialize_task_integration_bundle(
    task_dataset_path: Path,
    output_directory: Path,
    config: TaskIntegrationConfig | None = None,
) -> dict[str, Any]:
    """Atomically materialize one fixed-split, train-ready task bundle."""

    config = config or TaskIntegrationConfig()
    config.validate()
    source = _assert_final_manifest_bound_source(
        task_dataset_path,
        required=config.require_manifest_bound_directory,
    )
    semantics = _derive_task_semantics(
        source,
        batch_size=min(config.split_batch_size, 65_536),
    )
    allow_derived = config.task_eligibility_mode == "derived_sensitivity"
    outcome_kind = str(semantics["observation_kind"])
    if allow_derived and outcome_kind != "derived":
        raise ValueError("Derived-sensitivity integration requires a derived-only task")
    if not allow_derived and outcome_kind == "derived":
        raise ValueError("Default integration refuses derived labels")

    with _transactional_directory(output_directory) as staging:
        split_directory = staging / "split"
        model_directory = staging / "model_ready"
        readiness_directory = staging / "readiness"
        split_path = split_directory / f"{config.split_name}.parquet"
        model_path = model_directory / f"{config.split_name}.jsonl"
        split_config = SplitConfig(
            name=config.split_name,
            strategy=config.split_strategy,
            intended_use=config.intended_use,
            seed=config.seed,
            task_type=str(semantics["resolved_task_family"]),
            allow_derived_labels=allow_derived,
        )
        split_metadata = stream_hash_group_split_manifest(
            task_dataset_path,
            split_path,
            split_config,
            batch_size=config.split_batch_size,
        )
        if split_metadata["source_dataset_sha256"] != source.dataset_sha256:
            raise RuntimeError("Split does not bind the resolved source dataset")
        if int(split_metadata["record_count"]) != source.total_rows:
            raise RuntimeError("Split/source record counts do not reconcile")
        if sum(int(value) for value in split_metadata["partition_counts"].values()) != source.total_rows:
            raise RuntimeError("Split partition counts do not reconcile")
        if not str(split_metadata["claim_readiness"]).startswith("not_claim_ready"):
            raise RuntimeError("Scalable split must retain the near-duplicate claim blocker")
        amended_sidecar = _amend_split_sidecar_no_training(split_path)
        split_metadata["sidecar_sha256"] = amended_sidecar["sha256"]

        sample_part = next((part for part in source.parts if part.rows), None)
        if sample_part is None:
            raise ValueError("Canonical task contains no rows")
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:  # pragma: no cover - platform dependency profile.
            raise ImportError("pyarrow is required for integration memory estimation") from exc
        sample_parquet = pq.ParquetFile(sample_part.path)
        sample = next(sample_parquet.iter_batches(batch_size=min(1_000, sample_part.rows))).to_pandas()
        memory_estimate = estimate_model_ready_materialization_memory(
            sample,
            target_row_count=source.total_rows,
            allow_derived_labels=allow_derived,
            task_eligibility_mode=config.task_eligibility_mode,
        )
        model_metadata = serialize_model_ready_jsonl_streaming(
            task_dataset_path,
            split_path,
            model_path,
            source_dataset_sha256=str(split_metadata["source_dataset_sha256"]),
            split_manifest_sha256=str(split_metadata["manifest_sha256"]),
            split_sidecar_sha256=str(split_metadata["sidecar_sha256"]),
            build_config={
                "source_build_manifest_sha256": source.manifest_sha256,
                "split_config_sha256": split_metadata["config_sha256"],
                "near_duplicate_audit": "pending_separate_required",
                "task_eligibility_mode": config.task_eligibility_mode,
                "task_semantics": semantics,
            },
            batch_size=config.serialization_batch_size,
            allow_derived_labels=allow_derived,
            task_eligibility_mode=config.task_eligibility_mode,
        )
        if int(model_metadata["source_record_count"]) != source.total_rows:
            raise RuntimeError("Model-ready/source record counts do not reconcile")
        serialized_plus_excluded = int(model_metadata["record_count"]) + sum(
            int(value) for value in model_metadata["excluded_partition_counts"].values()
        )
        if serialized_plus_excluded != source.total_rows:
            raise RuntimeError("Serialized plus excluded record counts do not reconcile")

        partition_directory = model_directory / "partitions"
        partition_metadata = _partition_model_ready_jsonl(
            model_path,
            partition_directory,
            expected_sha256=str(model_metadata["file_sha256"]),
        )
        if sum(int(value) for value in partition_metadata["partition_counts"].values()) != int(
            model_metadata["record_count"]
        ):
            raise RuntimeError("Physical JSONL partitions do not reconcile")
        serialization_receipt = _retire_transient_model_ready_corpus(
            model_path,
            model_metadata,
            partition_metadata=partition_metadata,
        )
        if model_path.exists() or model_path.with_suffix(model_path.suffix + ".manifest.json").exists():
            raise RuntimeError("Combined all-partition corpus survived physical routing")

        lengths = _length_inventory(partition_metadata, partition_directory)
        _atomic_json(readiness_directory / "length_inventory.json", lengths)
        train_artifact = partition_metadata["partitions"]["train"]
        train_path = partition_directory / str(train_artifact["path"])
        vocabularies = _training_vocabularies(
            train_path,
            train_sha256=str(train_artifact["sha256"]),
            source_model_ready_sha256=str(model_metadata["file_sha256"]),
            output_directory=readiness_directory,
        )

        train_lengths = lengths["partitions"]["train"]

        def smoke_limit(modality: str) -> int:
            observed = int(train_lengths[modality]["maximum"])
            return max(8, ((observed + 2 + 7) // 8) * 8)

        collator_config = CollatorConfig(
            max_smiles_tokens=smoke_limit("smiles"),
            max_protein_tokens=smoke_limit("protein"),
            max_text_tokens=smoke_limit("text"),
            truncation_policy="error",
            pad_to_multiple_of=8,
        )
        collator = MultimodalCollator(
            smiles_vocabulary=vocabularies["smiles"],
            protein_vocabulary=vocabularies["protein"],
            text_vocabulary=vocabularies["text"],
            config=collator_config,
        )
        smoke = streaming_loader_smoke_test(
            JsonlIterableDataset(
                train_path,
                expected_sha256=str(train_artifact["sha256"]),
            ),
            collator,
            batch_size=config.loader_batch_size,
            maximum_batches=config.loader_maximum_batches,
        )
        smoke.update(
            {
                "collator_config": asdict(collator_config),
                "collator_config_sha256": stable_json_digest(asdict(collator_config)),
                "limit_policy": ("training_maximum_plus_special_tokens_rounded_to_8_for_smoke_only"),
                "substantive_training_started": NO_SUBSTANTIVE_TRAINING,
            }
        )
        if smoke["status"] != "passed" or int(smoke["examples_seen"]) > 32:
            raise RuntimeError("Streaming loader smoke failed its capped acceptance contract")
        _atomic_json(readiness_directory / "streaming_loader_smoke.json", smoke)

        _atomic_json(
            readiness_directory / "task_semantics.json",
            {
                "schema_version": "task_semantics_decision_v1",
                **semantics,
                "task_eligibility_mode": config.task_eligibility_mode,
                "substantive_training_started": NO_SUBSTANTIVE_TRAINING,
            },
        )
        partition_manifest_path = partition_directory / "partition_manifest.json"
        if file_sha256(partition_manifest_path) != partition_metadata["external_manifest_file_sha256"]:
            raise RuntimeError("Physical partition manifest digest changed before acceptance")
        test_relative_path = "model_ready/partitions/test.jsonl"
        test_artifact = partition_metadata["partitions"]["test"]
        component_inventory = _component_inventory(
            staging,
            prebound_entries={
                test_relative_path: {
                    "relative_path": test_relative_path,
                    "sha256": str(test_artifact["sha256"]),
                    "size_bytes": int(test_artifact["size_bytes"]),
                    "verification_source": "bound_during_physical_routing_no_reopen",
                }
            },
        )
        component_inventory_sha256 = stable_json_digest(component_inventory)
        acceptance = {
            "schema_version": INTEGRATION_BUNDLE_SCHEMA_VERSION,
            "integration_config": asdict(config),
            "integration_config_sha256": stable_json_digest(asdict(config)),
            "bundle_layout": {
                "split": f"split/{config.split_name}.parquet",
                "serialization_receipt": "model_ready/combined_serialization_receipt.json",
                "combined_model_ready_corpus": None,
                "partitions": "model_ready/partitions",
                "readiness": "readiness",
            },
            "source_dataset_sha256": source.dataset_sha256,
            "source_build_manifest_sha256": source.manifest_sha256,
            "source_part_count": len(source.parts),
            "source_record_count": source.total_rows,
            "task_semantics": semantics,
            "task_eligibility_mode": config.task_eligibility_mode,
            "split_config": asdict(split_config),
            "split_config_sha256": split_metadata["config_sha256"],
            "split_manifest_sha256": split_metadata["manifest_sha256"],
            "split_sidecar_sha256": split_metadata["sidecar_sha256"],
            "split_partition_counts": split_metadata["partition_counts"],
            "near_duplicate_audit": split_metadata["near_duplicate_audit"],
            "claim_readiness": split_metadata["claim_readiness"],
            "model_ready_sha256": model_metadata["file_sha256"],
            "model_ready_manifest_sha256": model_metadata["manifest_sha256"],
            "model_ready_record_count": model_metadata["record_count"],
            "model_ready_combined_corpus_retained": False,
            "model_ready_combined_corpus_retired_after_routing": True,
            "serialization_receipt_file_sha256": serialization_receipt["sha256"],
            "physical_partition_manifest_file_sha256_external": partition_metadata[
                "external_manifest_file_sha256"
            ],
            "physical_partition_counts": partition_metadata["partition_counts"],
            "vocabulary_digests": {key: vocabulary.digest() for key, vocabulary in vocabularies.items()},
            "memory_estimate": memory_estimate,
            "loader_smoke_status": smoke["status"],
            "loader_smoke_examples": smoke["examples_seen"],
            "component_inventory": component_inventory,
            "component_inventory_sha256": component_inventory_sha256,
            "component_inventory_policy": (
                "all bundle files except acceptance.json; acceptance is externally hashed on return"
            ),
            "test_lockbox_after_physical_routing": {
                "relative_path": test_relative_path,
                "open_calls": 0,
                "hash_calls": 0,
                "iteration_calls": 0,
                "length_status": "not_inspected_locked_test",
                "binding_source": "physical routing manifest",
            },
            "large_model_training_started": NO_SUBSTANTIVE_TRAINING,
            "substantive_training_started": NO_SUBSTANTIVE_TRAINING,
        }
        _atomic_json(staging / "acceptance.json", acceptance)
        _verify_custom_json_flags(staging)
        _verify_component_inventory(
            staging,
            component_inventory,
            inventory_sha256=component_inventory_sha256,
            excluded_relative_paths=("acceptance.json",),
            do_not_rehash=(test_relative_path,),
        )

    acceptance_path = output_directory.resolve() / "acceptance.json"
    final = json.loads(acceptance_path.read_text(encoding="utf-8"))
    final["output_directory"] = output_directory.resolve().as_posix()
    final["acceptance_file_sha256_external"] = file_sha256(acceptance_path)
    return final


def _stable_capped_sample(
    path: Path,
    *,
    expected_sha256: str,
    maximum_examples: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Keep the lowest seeded record hashes with memory bounded by the cap."""

    heap: list[tuple[int, str, dict[str, Any]]] = []
    for example in JsonlIterableDataset(path, expected_sha256=expected_sha256):
        record_id = str(example["record_id"])
        rank = int.from_bytes(
            hashlib.sha256(f"{seed}|{record_id}".encode()).digest(),
            "big",
        )
        item = (-rank, record_id, example)
        if len(heap) < maximum_examples:
            heapq.heappush(heap, item)
        elif item > heap[0]:
            heapq.heapreplace(heap, item)
    return [item[2] for item in sorted(heap, key=lambda entry: (-entry[0], entry[1]))]


def _raw_label_key(example: Mapping[str, Any]) -> str:
    label = example["label"]
    value = label.get("value")
    if value is not None:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return str(value).strip()
        if math.isfinite(numeric) and numeric.is_integer():
            return str(int(numeric))
        return repr(numeric)
    return str(label.get("text", "")).strip()


def _diagnostic_targets(
    train_examples: Sequence[Mapping[str, Any]],
    validation_examples: Sequence[Mapping[str, Any]],
    *,
    semantics: Mapping[str, Any],
    config: DiagnosticConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    label_kind = str(semantics["label_kind"])
    if label_kind == "continuous_exact":
        train = np.asarray([float(example["label"]["value"]) for example in train_examples])
        validation = np.asarray([float(example["label"]["value"]) for example in validation_examples])
        if not np.all(np.isfinite(train)) or not np.all(np.isfinite(validation)):
            raise ValueError("Exact regression diagnostic labels must be finite")
        return (
            train,
            validation,
            {
                "task_family": "regression",
                "label_encoding": "identity_continuous_exact",
            },
        )
    if label_kind != "categorical":
        raise ValueError(f"Diagnostic targets do not support label_kind={label_kind!r}")

    train_keys = [_raw_label_key(example) for example in train_examples]
    validation_keys = [_raw_label_key(example) for example in validation_examples]
    explicit_mapping = dict(config.binary_label_mapping)
    if explicit_mapping:
        unknown = sorted((set(train_keys) | set(validation_keys)) - set(explicit_mapping))
        if unknown:
            raise ValueError(f"Binary label mapping does not cover observed labels: {unknown}")
        mapping = explicit_mapping
        encoding_policy = "explicit_user_supplied_binary_mapping"
    else:
        if not (set(train_keys) | set(validation_keys)).issubset({"0", "1"}):
            raise ValueError(
                "Categorical diagnostics require explicit binary_label_mapping unless labels are numeric 0/1"
            )
        mapping = {"0": 0, "1": 1}
        encoding_policy = "explicit_numeric_identity_0_1"
    train = np.asarray([mapping[key] for key in train_keys], dtype=float)
    validation = np.asarray([mapping[key] for key in validation_keys], dtype=float)
    if set(np.unique(train)) != {0.0, 1.0}:
        raise ValueError("Capped classification training sample must contain both encoded classes")
    if not set(np.unique(validation)).issubset({0.0, 1.0}):
        raise ValueError("Validation labels must use the frozen binary encoding")
    return (
        train,
        validation,
        {
            "task_family": "classification",
            "label_encoding": encoding_policy,
            "binary_label_mapping": dict(sorted(mapping.items())),
        },
    )


def _selected_row_frame(
    train_examples: Sequence[Mapping[str, Any]],
    validation_examples: Sequence[Mapping[str, Any]],
    y_train: np.ndarray,
    y_validation: np.ndarray,
    *,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for partition, examples, targets in (
        ("train", train_examples, y_train),
        ("validation", validation_examples, y_validation),
    ):
        for example, target in zip(examples, targets, strict=True):
            record_id = str(example["record_id"])
            rows.append(
                {
                    "record_id": record_id,
                    "selection_sha256": hashlib.sha256(f"{seed}|{record_id}".encode()).hexdigest(),
                    "partition": partition,
                    "task_id": str(example["task_id"]),
                    "molecule_id": str(example["molecule_id"]),
                    "protein_id": str(example["protein_id"]),
                    "standardized_smiles": str(example["inputs"].get("smiles", "")),
                    "protein_sequence_status": str(example["inputs"].get("protein_sequence_status", "")),
                    "protein_sequence": str(example["inputs"].get("protein_sequence", "")),
                    "text": str(example["inputs"].get("text", "")),
                    "label_value_encoded": float(target),
                    "label_kind": str(example["label"]["label_kind"]),
                    "label_relation": str(example["label"]["relation"]),
                    "label_unit": str(example["label"]["unit"]),
                    "outcome_kind": str(example["label"]["outcome_kind"]),
                    "label_lineage_digest": str(example["label"].get("lineage_digest", "")),
                }
            )
    return pd.DataFrame(rows)


def _write_skipped_diagnostic_bundle(
    staging: Path,
    *,
    reason: str,
    integration_acceptance_sha256: str,
    semantics: Mapping[str, Any],
    config: DiagnosticConfig,
) -> dict[str, Any]:
    component_inventory: dict[str, dict[str, Any]] = {}
    payload = {
        "schema_version": DIAGNOSTIC_BUNDLE_SCHEMA_VERSION,
        "status": "skipped",
        "reason": reason,
        "task_semantics": dict(semantics),
        "integration_acceptance_sha256": integration_acceptance_sha256,
        "diagnostic_config": asdict(config),
        "diagnostic_config_sha256": stable_json_digest(asdict(config)),
        "component_inventory": component_inventory,
        "component_inventory_sha256": stable_json_digest(component_inventory),
        "component_inventory_policy": (
            "no generated components; diagnostic_status.json is externally hashed on return"
        ),
        "test_partition_opened": False,
        "model_selection_performed": False,
        "hyperparameter_sweep_performed": False,
        "large_model_training_started": NO_SUBSTANTIVE_TRAINING,
        "substantive_training_started": NO_SUBSTANTIVE_TRAINING,
    }
    _atomic_json(staging / "diagnostic_status.json", payload)
    return payload


def materialize_capped_diagnostic_bundle(
    integration_bundle: Path,
    output_directory: Path,
    *,
    integration_acceptance_sha256: str,
    config: DiagnosticConfig | None = None,
) -> dict[str, Any]:
    """Create capped train/validation diagnostics without opening the test JSONL."""

    config = config or DiagnosticConfig()
    config.validate()
    acceptance_path = integration_bundle.resolve() / "acceptance.json"
    if not acceptance_path.is_file():
        raise FileNotFoundError(acceptance_path)
    if file_sha256(acceptance_path) != integration_acceptance_sha256:
        raise ValueError("Integration acceptance digest does not match the supplied bundle")
    integration_acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    if integration_acceptance.get("schema_version") != INTEGRATION_BUNDLE_SCHEMA_VERSION:
        raise ValueError("Diagnostic input is not a recognized integration acceptance contract")
    if integration_acceptance.get("substantive_training_started") is not False:
        raise ValueError("Integration bundle does not preserve the no-training boundary")
    if integration_acceptance.get("large_model_training_started") is not False:
        raise ValueError("Integration bundle reports that large-model training has started")
    semantics = integration_acceptance.get("task_semantics")
    if not isinstance(semantics, Mapping):
        raise ValueError("Integration acceptance lacks bound task semantics")

    label_kind = str(semantics.get("label_kind", ""))
    if label_kind == "continuous_censored":
        reason = "skipped_requires_censored_model_no_midpoint_imputation"
    elif label_kind == "ordinal":
        reason = "skipped_requires_multiclass_or_ordinal_model"
    elif str(semantics.get("observation_kind", "")) == "derived":
        reason = "skipped_derived_sensitivity_not_primary_diagnostic"
    else:
        reason = ""
    with _transactional_directory(output_directory) as staging:
        if reason:
            result = _write_skipped_diagnostic_bundle(
                staging,
                reason=reason,
                integration_acceptance_sha256=integration_acceptance_sha256,
                semantics=semantics,
                config=config,
            )
            _verify_custom_json_flags(staging)
            _verify_component_inventory(
                staging,
                result["component_inventory"],
                inventory_sha256=str(result["component_inventory_sha256"]),
                excluded_relative_paths=("diagnostic_status.json",),
            )
        else:
            partition_directory = integration_bundle.resolve() / "model_ready" / "partitions"
            partition_manifest_path = partition_directory / "partition_manifest.json"
            expected_partition_manifest_sha256 = str(
                integration_acceptance["physical_partition_manifest_file_sha256_external"]
            )
            if file_sha256(partition_manifest_path) != expected_partition_manifest_sha256:
                raise ValueError("Physical partition manifest digest does not match acceptance")
            partition_manifest = json.loads(partition_manifest_path.read_text(encoding="utf-8"))
            artifacts = partition_manifest.get("partitions")
            if not isinstance(artifacts, Mapping):
                raise ValueError("Physical partition manifest lacks partition artifacts")
            train_artifact = artifacts["train"]
            validation_artifact = artifacts["validation"]
            for partition, artifact in (
                ("train", train_artifact),
                ("validation", validation_artifact),
            ):
                if not isinstance(artifact, Mapping):
                    raise ValueError(f"Physical {partition} partition metadata is invalid")
                if str(artifact.get("path", "")) != f"{partition}.jsonl":
                    raise ValueError(f"Physical {partition} partition path is not the canonical lockbox name")
                if int(artifact.get("record_count", 0)) < 1:
                    raise ValueError(f"Physical {partition} partition is empty")
            train_path = partition_directory / str(train_artifact["path"])
            validation_path = partition_directory / str(validation_artifact["path"])
            # Deliberately do not resolve, hash, open, or iterate artifacts["test"].
            train_examples = _stable_capped_sample(
                train_path,
                expected_sha256=str(train_artifact["sha256"]),
                maximum_examples=config.maximum_train_examples,
                seed=config.seed,
            )
            validation_examples = _stable_capped_sample(
                validation_path,
                expected_sha256=str(validation_artifact["sha256"]),
                maximum_examples=config.maximum_validation_examples,
                seed=config.seed,
            )
            if len(train_examples) < 2 or not validation_examples:
                raise ValueError("Insufficient capped train/validation examples for diagnostics")
            y_train, y_validation, target_metadata = _diagnostic_targets(
                train_examples,
                validation_examples,
                semantics=semantics,
                config=config,
            )
            task_ids = {str(example["task_id"]) for example in (*train_examples, *validation_examples)}
            if len(task_ids) != 1 or next(iter(task_ids)) != str(semantics["task_id"]):
                raise ValueError("Diagnostic sample does not preserve one bound task ID")
            prohibited_kinds = {
                str(example["label"]["outcome_kind"]) for example in (*train_examples, *validation_examples)
            } - {"experimental_raw", "experimental_summary", "curated_assertion"}
            if prohibited_kinds:
                raise ValueError(
                    "Default diagnostic baselines refuse non-observed/curated outcomes: "
                    f"{sorted(prohibited_kinds)}"
                )

            rows = _selected_row_frame(
                train_examples,
                validation_examples,
                y_train,
                y_validation,
                seed=config.seed,
            )
            features_directory = staging / "features"
            baselines_directory = staging / "baselines"
            selected_rows_path = features_directory / "selected_rows.parquet"
            _atomic_frame(rows, selected_rows_path)
            descriptors = deterministic_descriptor_frame(
                rows[["molecule_id", "standardized_smiles"]],
                smiles_column="standardized_smiles",
                id_column="molecule_id",
            )
            descriptors.insert(0, "record_id", rows["record_id"].to_numpy())
            descriptors.insert(1, "partition", rows["partition"].to_numpy())
            descriptors_path = features_directory / "descriptors.parquet"
            _atomic_frame(descriptors, descriptors_path)
            failures = feature_failure_summary(rows)
            _atomic_frame(failures, features_directory / "feature_failure_summary.csv")
            fingerprints, backend = fingerprint_matrix(
                rows["standardized_smiles"],
                backend="rdkit",
                n_bits=config.fingerprint_bits,
                radius=config.fingerprint_radius,
            )
            if backend != "rdkit_morgan":
                raise RuntimeError("Diagnostic feature artifact silently changed fingerprint backend")
            fingerprints = fingerprints.tocsr()
            fingerprint_path = features_directory / (
                f"fingerprints_rdkit_morgan_r{config.fingerprint_radius}_{config.fingerprint_bits}.npz"
            )
            _atomic_sparse_matrix(fingerprints, fingerprint_path)
            feature_config = {
                "schema_version": "capped_diagnostic_feature_config_v1",
                "diagnostic_config": asdict(config),
                "diagnostic_config_sha256": stable_json_digest(asdict(config)),
                "selection": {
                    "algorithm": "lowest_full_256_bit_seeded_sha256_record_id",
                    "rank_bits": 256,
                    "seed": config.seed,
                    "maximum_train_examples": config.maximum_train_examples,
                    "maximum_validation_examples": config.maximum_validation_examples,
                    "selected_train_examples": len(train_examples),
                    "selected_validation_examples": len(validation_examples),
                },
                "fingerprint": {
                    "backend": backend,
                    "radius": config.fingerprint_radius,
                    "bits": config.fingerprint_bits,
                },
                "task_semantics": dict(semantics),
                "target": target_metadata,
                "test_partition_opened": False,
                "substantive_training_started": NO_SUBSTANTIVE_TRAINING,
            }
            feature_config_path = features_directory / "feature_config.json"
            _atomic_json(feature_config_path, feature_config)
            component_hashes = {
                "selected_rows.parquet": file_sha256(selected_rows_path),
                "descriptors.parquet": file_sha256(descriptors_path),
                fingerprint_path.name: file_sha256(fingerprint_path),
                "feature_config.json": file_sha256(feature_config_path),
                "feature_failure_summary.csv": file_sha256(
                    features_directory / "feature_failure_summary.csv"
                ),
            }
            feature_bundle_sha256 = stable_json_digest(
                {
                    "component_hashes": component_hashes,
                    "configuration": feature_config,
                }
            )
            feature_manifest = {
                "schema_version": "capped_diagnostic_feature_manifest_v1",
                "component_hashes": component_hashes,
                "feature_bundle_sha256": feature_bundle_sha256,
                "record_count": len(rows),
                "test_partition_opened": False,
                "substantive_training_started": NO_SUBSTANTIVE_TRAINING,
            }
            feature_manifest_path = features_directory / "feature_manifest.json"
            _atomic_json(feature_manifest_path, feature_manifest)

            numeric_descriptor_columns = [
                column
                for column in descriptors.columns
                if column
                not in {
                    "record_id",
                    "partition",
                    "molecule_id",
                    "feature_origin",
                    "leakage_role",
                    "feature_registry_version",
                    "input_structure_sha256",
                }
            ]
            leakage = numeric_target_leakage_scan(
                descriptors[numeric_descriptor_columns],
                np.concatenate([y_train, y_validation]),
            )
            _atomic_frame(
                leakage,
                features_directory / "numeric_target_leakage_scan.csv",
            )

            train_mask = rows["partition"].eq("train").to_numpy()
            validation_mask = rows["partition"].eq("validation").to_numpy()
            X_train = fingerprints[train_mask]
            X_validation = fingerprints[validation_mask]
            task_family: TaskFamily = target_metadata["task_family"]
            baseline_config = BaselineConfig(
                task_type=task_family,
                seed=config.seed,
                class_imbalance=("balanced_class_weight" if task_family == "classification" else "none"),
                include_tree_baseline=False,
            )
            baseline_result = run_diagnostic_baselines(
                X_train,
                y_train,
                X_validation,
                y_validation,
                eval_record_ids=rows.loc[validation_mask, "record_id"].tolist(),
                config=baseline_config,
                task_id=next(iter(task_ids)),
                split_name=str(integration_acceptance["split_config"]["name"]),
                feature_set_name=(
                    f"capped_rdkit_morgan_r{config.fingerprint_radius}_{config.fingerprint_bits}"
                ),
                train_outcome_kinds=rows.loc[train_mask, "outcome_kind"].tolist(),
                eval_outcome_kinds=rows.loc[validation_mask, "outcome_kind"].tolist(),
                train_label_lineage_digests=rows.loc[train_mask, "label_lineage_digest"].tolist(),
                eval_label_lineage_digests=rows.loc[validation_mask, "label_lineage_digest"].tolist(),
                dataset_sha256=str(integration_acceptance["source_dataset_sha256"]),
                split_manifest_sha256=str(integration_acceptance["split_manifest_sha256"]),
                feature_artifact_sha256=feature_bundle_sha256,
                eval_partition="validation",
            )
            _atomic_frame(baseline_result.metrics, baselines_directory / "metrics.csv")
            _atomic_frame(
                baseline_result.predictions,
                baselines_directory / "validation_predictions.parquet",
            )
            _atomic_frame(
                build_error_analysis(baseline_result.predictions),
                baselines_directory / "validation_error_analysis.parquet",
            )
            permutation = run_label_permutation_control(
                X_train,
                y_train,
                X_validation,
                y_validation,
                config=baseline_config,
            )
            permutation["substantive_training_started"] = NO_SUBSTANTIVE_TRAINING
            _atomic_json(
                baselines_directory / "label_permutation_control.json",
                permutation,
            )
            identifier = run_identifier_hash_control(
                rows.loc[train_mask, "record_id"].tolist(),
                y_train,
                rows.loc[validation_mask, "record_id"].tolist(),
                y_validation,
                config=baseline_config,
            )
            identifier["substantive_training_started"] = NO_SUBSTANTIVE_TRAINING
            _atomic_json(
                baselines_directory / "identifier_hash_control.json",
                identifier,
            )
            if task_family == "classification":
                imbalance = class_imbalance_options(y_train)
                imbalance.update(
                    {
                        "validation_class_counts": {
                            str(int(label)): int(np.sum(y_validation == label)) for label in (0, 1)
                        },
                        "validation_prevalence": float(np.mean(y_validation)),
                        "label_encoding": target_metadata,
                        "threshold_policy": "fixed_0.5_diagnostic_not_selected",
                        "substantive_training_started": NO_SUBSTANTIVE_TRAINING,
                    }
                )
                _atomic_json(
                    baselines_directory / "class_imbalance_and_prevalence.json",
                    imbalance,
                )
            model_artifacts: dict[str, str] = {}
            for name, model in sorted(baseline_result.fitted_models.items()):
                model_path = baselines_directory / f"{name}.joblib"
                _atomic_joblib(model, model_path)
                model_artifacts[model_path.name] = file_sha256(model_path)
            baseline_metadata = {
                **baseline_result.metadata,
                "schema_version": "capped_diagnostic_baseline_metadata_v1",
                "integration_acceptance_sha256": integration_acceptance_sha256,
                "diagnostic_config": asdict(config),
                "diagnostic_config_sha256": stable_json_digest(asdict(config)),
                "feature_bundle_sha256": feature_bundle_sha256,
                "feature_manifest_file_sha256": file_sha256(feature_manifest_path),
                "model_artifacts": model_artifacts,
                "target": target_metadata,
                "opened_partitions": ["train", "validation"],
                "test_partition_opened": False,
                "model_selection_performed": False,
                "hyperparameter_sweep_performed": False,
                "large_model_training_started": NO_SUBSTANTIVE_TRAINING,
                "substantive_training_started": NO_SUBSTANTIVE_TRAINING,
            }
            _atomic_json(
                baselines_directory / "baseline_metadata.json",
                baseline_metadata,
            )
            component_inventory = _component_inventory(staging)
            component_inventory_sha256 = stable_json_digest(component_inventory)
            result = {
                "schema_version": DIAGNOSTIC_BUNDLE_SCHEMA_VERSION,
                "status": "completed_diagnostic_only",
                "integration_acceptance_sha256": integration_acceptance_sha256,
                "diagnostic_config": asdict(config),
                "diagnostic_config_sha256": stable_json_digest(asdict(config)),
                "task_semantics": dict(semantics),
                "target": target_metadata,
                "selection": feature_config["selection"],
                "feature_bundle_sha256": feature_bundle_sha256,
                "feature_manifest_file_sha256": file_sha256(feature_manifest_path),
                "baseline_metadata_file_sha256": file_sha256(baselines_directory / "baseline_metadata.json"),
                "metrics_file_sha256": file_sha256(baselines_directory / "metrics.csv"),
                "predictions_file_sha256": file_sha256(
                    baselines_directory / "validation_predictions.parquet"
                ),
                "component_inventory": component_inventory,
                "component_inventory_sha256": component_inventory_sha256,
                "component_inventory_policy": (
                    "all diagnostic bundle files except acceptance.json; acceptance is externally "
                    "hashed on return"
                ),
                "opened_partitions": ["train", "validation"],
                "test_partition_opened": False,
                "interpretation": "lightweight_diagnostic_not_scientific_or_prospective_evidence",
                "model_selection_performed": False,
                "hyperparameter_sweep_performed": False,
                "large_model_training_started": NO_SUBSTANTIVE_TRAINING,
                "substantive_training_started": NO_SUBSTANTIVE_TRAINING,
            }
            _atomic_json(staging / "acceptance.json", result)
            _verify_custom_json_flags(staging)
            _verify_component_inventory(
                staging,
                component_inventory,
                inventory_sha256=component_inventory_sha256,
                excluded_relative_paths=("acceptance.json",),
            )

    final_path = output_directory.resolve() / ("diagnostic_status.json" if reason else "acceptance.json")
    final = json.loads(final_path.read_text(encoding="utf-8"))
    final["output_directory"] = output_directory.resolve().as_posix()
    final["acceptance_file_sha256_external"] = file_sha256(final_path)
    return final


__all__ = [
    "DiagnosticConfig",
    "TaskIntegrationConfig",
    "materialize_capped_diagnostic_bundle",
    "materialize_task_integration_bundle",
]
