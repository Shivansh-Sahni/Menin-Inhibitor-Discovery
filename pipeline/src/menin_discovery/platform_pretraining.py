"""Large-model input preparation and guarded fine-tuning readiness utilities.

No function in this module launches substantive model training.  Serialization,
tokenization, collation, checkpoint contracts, analytic resource estimates,
and a strictly capped tiny wiring smoke test are provided so an authorized
training run can begin from frozen artifacts later.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import random
import re
import sys
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, overload

import numpy as np
import pandas as pd

from .platform_baselines import robustness_configuration_matrix
from .platform_data_schema import ACCESS_CLASSES
from .platform_features import (
    FEATURE_REGISTRY_VERSION,
    FeatureRegistry,
    normalize_protein_sequence,
    prepare_molecular_graph,
    stable_json_digest,
    tokenize_smiles,
)
from .platform_metrics import metric_reporting_registry
from .platform_splits import resolve_manifest_bound_parquet_dataset

MODEL_READY_SCHEMA_VERSION = "1.0.0"
SERIALIZATION_FORMAT = "canonical_jsonl_utf8_v1"
ALLOWED_OUTCOME_KINDS = {
    "experimental_raw",
    "experimental_summary",
    "curated_assertion",
    "derived",
}
OUTCOME_KIND_ALIASES = {
    "observed": "experimental_raw",
    "curated": "curated_assertion",
    "experimental_observation": "experimental_raw",
    "curated_label": "curated_assertion",
}
PROHIBITED_LABEL_KINDS = {"prediction", "computational_prediction", "model_prediction"}
PUBLIC_ACCESS_CLASS = "public_redistributable"

if PUBLIC_ACCESS_CLASS not in ACCESS_CLASSES:  # pragma: no cover - cross-module contract guard.
    raise RuntimeError("Canonical platform schema no longer defines public_redistributable")


def _canonical_outcome_kind(value: object) -> str:
    normalized = str(value).strip().lower()
    return OUTCOME_KIND_ALIASES.get(normalized, normalized)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _canonical_json(payload: Any) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=_json_default,
    )


def _require_sha256(value: str, field_name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value.lower()):
        raise ValueError(f"{field_name} must be a 64-character SHA-256 hexadecimal digest")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _cell(value: object) -> Any:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _first_column(frame: pd.DataFrame, candidates: Sequence[str], *, required: bool) -> str | None:
    column = next((item for item in candidates if item in frame.columns), None)
    if required and column is None:
        raise ValueError(f"None of the required canonical columns is available: {tuple(candidates)}")
    return column


def validate_model_ready_example(example: Mapping[str, Any]) -> None:
    """Validate scientific type boundaries and minimum multimodal lineage."""

    required = {
        "schema_version",
        "record_id",
        "task_id",
        "task_type",
        "molecule_id",
        "protein_id",
        "assay_id",
        "inputs",
        "label",
        "context",
        "provenance",
    }
    missing = sorted(required - set(example))
    if missing:
        raise ValueError(f"Model-ready example is missing fields: {missing}")
    if str(example["schema_version"]) != MODEL_READY_SCHEMA_VERSION:
        raise ValueError("Unsupported model-ready schema version")
    for key in ("record_id", "task_id", "task_type", "molecule_id", "protein_id", "assay_id"):
        if not str(example[key]).strip():
            raise ValueError(f"Model-ready field {key!r} may not be blank")
    inputs = example["inputs"]
    if not isinstance(inputs, Mapping):
        raise ValueError("inputs must be an object")
    if not any(str(inputs.get(key, "") or "").strip() for key in ("smiles", "protein_sequence", "text")):
        if not inputs.get("molecular_graph"):
            raise ValueError("At least one molecular, protein, graph, or text input is required")
    label = example["label"]
    if not isinstance(label, Mapping):
        raise ValueError("label must be an object")
    required_label_fields = {
        "label_kind",
        "value",
        "text",
        "relation",
        "lower_bound",
        "upper_bound",
        "unit",
        "outcome_kind",
        "lineage_digest",
    }
    missing_label_fields = sorted(required_label_fields - set(label))
    if missing_label_fields:
        raise ValueError(f"Model-ready label is missing fields: {missing_label_fields}")
    outcome_kind = _canonical_outcome_kind(label.get("outcome_kind", ""))
    if outcome_kind in PROHIBITED_LABEL_KINDS or outcome_kind not in ALLOWED_OUTCOME_KINDS:
        raise ValueError(f"Prohibited or unknown label outcome_kind: {outcome_kind!r}")
    if outcome_kind == "derived":
        lineage_digest = str(label.get("lineage_digest", "")).strip()
        if not lineage_digest:
            raise ValueError("Derived labels require a non-empty lineage_digest")
        _require_sha256(lineage_digest, "derived label lineage_digest")
    label_kind = str(label.get("label_kind", ""))
    if label_kind not in {"continuous_exact", "continuous_censored", "categorical", "ordinal"}:
        raise ValueError(f"Unsupported label_kind: {label_kind!r}")
    if not str(label.get("relation", "")).strip() or not str(label.get("unit", "")).strip():
        raise ValueError("Every model-ready label requires explicit nonblank relation and unit")
    if label_kind == "continuous_exact" and label.get("value") is None:
        raise ValueError("continuous_exact labels require value")
    if label_kind == "continuous_exact":
        try:
            exact_value = float(label["value"])
        except (TypeError, ValueError) as exc:
            raise ValueError("continuous_exact value must be numeric") from exc
        if not math.isfinite(exact_value):
            raise ValueError("continuous_exact value must be finite")
        if str(label.get("relation", "=")) != "=":
            raise ValueError("continuous_exact labels require relation='='")
    if (
        label_kind in {"categorical", "ordinal"}
        and label.get("value") is None
        and not str(label.get("text", "")).strip()
    ):
        raise ValueError("categorical/ordinal labels require numeric value or label text")
    if label_kind == "continuous_censored":
        relation = str(label.get("relation", ""))
        if relation not in {"<", "<=", ">", ">=", "interval"}:
            raise ValueError("continuous_censored labels require a censoring relation")
        if label.get("lower_bound") is None and label.get("upper_bound") is None:
            raise ValueError("continuous_censored labels require at least one bound")
        lower_bound = label.get("lower_bound")
        upper_bound = label.get("upper_bound")
        for name, bound in (("lower_bound", lower_bound), ("upper_bound", upper_bound)):
            if bound is not None and not math.isfinite(float(bound)):
                raise ValueError(f"Censoring {name} must be finite when present")
        if relation == "interval":
            if lower_bound is None or upper_bound is None:
                raise ValueError("interval labels require both lower and upper bounds")
            if float(lower_bound) > float(upper_bound):
                raise ValueError("interval lower bound must not exceed upper bound")
        elif relation in {"<", "<="} and upper_bound is None:
            raise ValueError("left-censored labels require an upper bound")
        elif relation in {">", ">="} and lower_bound is None:
            raise ValueError("right-censored labels require a lower bound")
    context = example["context"]
    if not isinstance(context, Mapping):
        raise ValueError("context must be an object")
    missing_context = [
        key
        for key in ("evidence_domain", "endpoint", "assay_family")
        if not str(context.get(key, "")).strip()
    ]
    if missing_context:
        raise ValueError(f"Model-ready task context requires nonblank fields: {missing_context}")
    provenance = example["provenance"]
    if not isinstance(provenance, Mapping):
        raise ValueError("provenance must be an object")
    missing_provenance = [
        key
        for key in ("source_id", "snapshot_id", "source_record_id")
        if not str(provenance.get(key, "")).strip()
    ]
    if missing_provenance:
        raise ValueError(f"Every model-ready example requires nonblank provenance: {missing_provenance}")
    if str(provenance.get("access_class", "")).lower() != PUBLIC_ACCESS_CLASS:
        raise ValueError("Public model-ready artifacts require access_class=public_redistributable exactly")


def model_ready_examples_from_task_view(
    frame: pd.DataFrame,
    *,
    include_graph: bool = False,
    allow_derived_labels: bool = False,
    task_eligibility_mode: Literal["default", "derived_sensitivity"] = "default",
) -> list[dict[str, Any]]:
    """Adapt a canonical task view without pooling incompatible endpoint tasks."""

    required = {
        "observation_id",
        "molecule_id",
        "protein_id",
        "assay_id",
        "source_id",
        "snapshot_id",
        "source_record_id",
        "task_id",
        "task_type",
        "label_kind",
        "label_value",
        "label_text",
        "label_relation",
        "label_lower_bound",
        "label_upper_bound",
        "label_unit",
        "observation_kind",
        "access_class",
        "inclusion_status",
        "evidence_domain",
        "endpoint",
        "assay_family",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Canonical task view is missing columns: {missing}")
    record_column = "observation_id"
    molecule_column = "molecule_id"
    source_column = "source_id"
    smiles_column = _first_column(
        frame, ("standardized_smiles", "canonical_smiles", "submitted_smiles", "smiles"), required=False
    )
    sequence_column = _first_column(frame, ("sequence", "protein_sequence"), required=False)
    outcome_column = "observation_kind"
    access_column = "access_class"
    if frame.empty:
        raise ValueError("Canonical task view may not be empty")
    homogeneous_fields = [
        field
        for field in (
            "task_id",
            "task_type",
            "evidence_domain",
            "endpoint",
            "assay_family",
            "label_kind",
            "label_unit",
        )
    ]
    heterogeneous = {
        field: sorted(frame[field].fillna("").astype(str).str.strip().unique().tolist())
        for field in homogeneous_fields
        if frame[field].fillna("").astype(str).str.strip().nunique(dropna=False) != 1
    }
    if heterogeneous:
        raise ValueError(
            "One model-ready task view must have one shared homogeneous task signature; "
            f"heterogeneous={heterogeneous}"
        )
    if not str(frame["task_id"].iloc[0]).strip():
        raise ValueError("A shared task_id may not be blank")
    nonblank_columns = (
        "observation_id",
        "molecule_id",
        "protein_id",
        "assay_id",
        "source_id",
        "snapshot_id",
        "source_record_id",
        "task_id",
        "task_type",
        "label_kind",
        "label_relation",
        "label_unit",
        "observation_kind",
        "access_class",
        "evidence_domain",
        "endpoint",
        "assay_family",
    )
    blank_columns = [
        column for column in nonblank_columns if frame[column].fillna("").astype(str).str.strip().eq("").any()
    ]
    if blank_columns:
        raise ValueError(f"Canonical task view has blank required identity/semantic fields: {blank_columns}")

    normalized_inclusion = frame["inclusion_status"].fillna("").astype(str).str.strip().str.lower()
    if not normalized_inclusion.eq("included").all():
        raise ValueError("Only inclusion_status=included rows may enter model-ready artifacts")
    if task_eligibility_mode == "default":
        if allow_derived_labels:
            raise ValueError(
                "Generic derived-label opt-in cannot override default eligibility; use the explicit "
                "derived_sensitivity mode"
            )
        eligibility_column = next(
            (column for column in ("default_task_eligible", "task_eligible") if column in frame.columns),
            None,
        )
        if eligibility_column is None:
            raise ValueError("Default task views require explicit default_task_eligible/task_eligible")
        normalized_eligibility = frame[eligibility_column].fillna("").astype(str).str.strip().str.lower()
        if not normalized_eligibility.isin({"true", "1"}).all():
            raise ValueError("Default-ineligible rows are prohibited from default model-ready artifacts")
    elif task_eligibility_mode == "derived_sensitivity":
        if not allow_derived_labels:
            raise ValueError("Derived sensitivity mode requires allow_derived_labels=True")
        if "sensitivity_task_eligible" not in frame.columns:
            raise ValueError("Derived sensitivity mode requires sensitivity_task_eligible")
        sensitivity_eligible = (
            frame["sensitivity_task_eligible"].fillna("").astype(str).str.strip().str.lower()
        )
        if not sensitivity_eligible.isin({"true", "1"}).all():
            raise ValueError("Every derived sensitivity row must be explicitly sensitivity-task eligible")
        if "default_task_eligible" not in frame.columns:
            raise ValueError("Derived sensitivity rows must explicitly retain default_task_eligible=False")
        default_eligible = frame["default_task_eligible"].fillna("").astype(str).str.strip().str.lower()
        if not default_eligible.isin({"false", "0"}).all():
            raise ValueError("Derived sensitivity rows must remain ineligible for the default task path")
        sensitivity_outcomes = frame["observation_kind"].map(_canonical_outcome_kind)
        if not sensitivity_outcomes.eq("derived").all():
            raise ValueError("Derived sensitivity task views may contain derived labels only")
    else:  # pragma: no cover - typed callers cannot select another mode.
        raise ValueError(f"Unsupported task_eligibility_mode: {task_eligibility_mode}")

    examples: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str]] = set()
    for _, row in frame.iterrows():
        record_id = str(row[record_column]).strip()
        task_id = str(row["task_id"]).strip()
        key = (record_id, task_id)
        if key in seen_keys:
            raise ValueError(f"Duplicate record/task key in task view: {key}")
        seen_keys.add(key)
        outcome_kind = _canonical_outcome_kind(row[outcome_column])
        if outcome_kind == "derived" and not allow_derived_labels:
            raise ValueError("Derived labels are disabled by default and require explicit opt-in")
        access_class = str(row[access_column]).strip().lower()
        if access_class != PUBLIC_ACCESS_CLASS:
            raise ValueError(
                f"Non-public row cannot enter public model-ready artifact: record_id={record_id}"
            )
        smiles = str(row[smiles_column]).strip() if smiles_column and not pd.isna(row[smiles_column]) else ""
        sequence = (
            str(row[sequence_column]).strip() if sequence_column and not pd.isna(row[sequence_column]) else ""
        )
        normalized_sequence, invalid_sequence = normalize_protein_sequence(sequence)
        if invalid_sequence:
            normalized_sequence = ""
            protein_sequence_status = "invalid"
            protein_sequence_failure_reason = "invalid_characters:" + ",".join(invalid_sequence)
        elif normalized_sequence:
            protein_sequence_status = "valid"
            protein_sequence_failure_reason = ""
        else:
            protein_sequence_status = "missing"
            protein_sequence_failure_reason = "sequence_unavailable"
        inputs: dict[str, Any] = {
            "smiles": smiles,
            "protein_sequence": normalized_sequence,
            "protein_sequence_status": protein_sequence_status,
            "protein_sequence_failure_reason": protein_sequence_failure_reason,
            "text": "",
            "molecular_graph_status": "not_requested",
            "molecular_graph_failure_reason": "",
        }
        if include_graph and smiles:
            graph = prepare_molecular_graph(smiles)
            inputs["molecular_graph"] = graph if graph["valid"] else None
            inputs["molecular_graph_status"] = "valid" if graph["valid"] else "invalid"
            inputs["molecular_graph_failure_reason"] = str(graph.get("error", ""))
        elif include_graph:
            inputs["molecular_graph"] = None
            inputs["molecular_graph_status"] = "missing"
            inputs["molecular_graph_failure_reason"] = "smiles_unavailable"
        label_value = _cell(row["label_value"])
        label_text = str(_cell(row["label_text"]) or "")
        label = {
            "label_kind": str(row["label_kind"]),
            "value": label_value,
            "text": label_text,
            "relation": str(_cell(row["label_relation"])),
            "lower_bound": _cell(row["label_lower_bound"]),
            "upper_bound": _cell(row["label_upper_bound"]),
            "unit": str(_cell(row["label_unit"])),
            "outcome_kind": outcome_kind,
            "lineage_digest": str(_cell(row.get("label_lineage_digest", "")) or ""),
        }
        example = {
            "schema_version": MODEL_READY_SCHEMA_VERSION,
            "record_id": record_id,
            "task_id": task_id,
            "task_type": str(row["task_type"]),
            "molecule_id": str(row[molecule_column]).strip(),
            "protein_id": str(row["protein_id"]).strip(),
            "assay_id": str(row["assay_id"]).strip(),
            "inputs": inputs,
            "label": label,
            "context": {
                "evidence_domain": str(_cell(row.get("evidence_domain", "")) or ""),
                "endpoint": str(_cell(row.get("endpoint", "")) or ""),
                "endpoint_family": str(_cell(row.get("endpoint_family", "")) or ""),
                "assay_family": str(row["assay_family"]).strip(),
                "species": str(_cell(row.get("species", "")) or ""),
                "matrix": str(_cell(row.get("matrix", "")) or ""),
                "route": str(_cell(row.get("route", "")) or ""),
            },
            "provenance": {
                "source_id": str(row[source_column]).strip(),
                "snapshot_id": str(_cell(row.get("snapshot_id", "")) or ""),
                "source_record_id": str(row["source_record_id"]).strip(),
                "access_class": access_class,
                "document_year": _cell(row.get("document_year")),
            },
        }
        validate_model_ready_example(example)
        examples.append(example)
    return sorted(examples, key=lambda item: (item["record_id"], item["task_id"]))


def attach_fixed_split(examples: Sequence[Mapping[str, Any]], manifest: pd.DataFrame) -> list[dict[str, Any]]:
    """Join a frozen manifest by record ID; never regenerate a split."""

    required = {"record_id", "split", "split_name", "strategy"}
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"Split manifest is missing columns: {missing}")
    if manifest["record_id"].astype(str).duplicated().any():
        raise ValueError("Split manifest record IDs must be unique")
    mapping = manifest.set_index(manifest["record_id"].astype(str)).to_dict(orient="index")
    output: list[dict[str, Any]] = []
    for original in examples:
        record_id = str(original["record_id"])
        if record_id not in mapping:
            raise ValueError(f"Model-ready record missing from split manifest: {record_id}")
        copied = json.loads(_canonical_json(original))
        copied["split"] = {
            "partition": str(mapping[record_id]["split"]),
            "split_name": str(mapping[record_id]["split_name"]),
            "strategy": str(mapping[record_id]["strategy"]),
            "group_id": str(mapping[record_id].get("group_id", "")),
        }
        output.append(copied)
    if len(mapping) != len({str(item["record_id"]) for item in examples}):
        raise ValueError("Split manifest contains records absent from the model-ready examples")
    return output


def serialize_model_ready_jsonl(
    examples: Sequence[Mapping[str, Any]],
    output_path: Path,
    *,
    source_dataset_sha256: str,
    split_manifest_sha256: str,
    build_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Write deterministic JSONL plus an integrity manifest using atomic replacement."""

    _require_sha256(source_dataset_sha256, "source_dataset_sha256")
    _require_sha256(split_manifest_sha256, "split_manifest_sha256")
    normalized: list[dict[str, Any]] = []
    if not examples:
        raise ValueError("Train-ready serialization refuses an empty example collection")
    keys: set[tuple[str, str]] = set()
    for example in examples:
        validate_model_ready_example(example)
        copied = json.loads(_canonical_json(example))
        split = copied.get("split")
        if not isinstance(split, Mapping):
            raise ValueError("Train-ready serialization requires an attached fixed split object")
        if split.get("partition") not in {"train", "validation", "test"}:
            raise ValueError(
                "Train-ready serialization rejects unassigned/excluded partitions; keep them in split audit artifacts"
            )
        if not str(split.get("split_name", "")).strip() or not str(split.get("strategy", "")).strip():
            raise ValueError("Attached split requires nonblank split_name and strategy")
        key = (str(copied["record_id"]), str(copied["task_id"]))
        if key in keys:
            raise ValueError(f"Duplicate serialized record/task key: {key}")
        keys.add(key)
        normalized.append(copied)
    normalized.sort(key=lambda item: (item["record_id"], item["task_id"]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    line_digests: list[str] = []
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for example in normalized:
            line = _canonical_json(example)
            handle.write(line + "\n")
            line_digests.append(hashlib.sha256(line.encode()).hexdigest())
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output_path)
    partitions = Counter(
        str(example.get("split", {}).get("partition", "unassigned")) for example in normalized
    )
    outcome_kinds = Counter(str(example["label"]["outcome_kind"]) for example in normalized)
    metadata = {
        "schema_version": MODEL_READY_SCHEMA_VERSION,
        "serialization_format": SERIALIZATION_FORMAT,
        "path": output_path.name,
        "file_sha256": file_sha256(output_path),
        "record_count": len(normalized),
        "ordered_line_digest_sha256": hashlib.sha256("\n".join(line_digests).encode()).hexdigest(),
        "source_dataset_sha256": source_dataset_sha256,
        "split_manifest_sha256": split_manifest_sha256,
        "build_config": dict(build_config),
        "build_config_sha256": stable_json_digest(build_config),
        "partition_counts": dict(sorted(partitions.items())),
        "outcome_kind_counts": dict(sorted(outcome_kinds.items())),
        "label_policy": "prediction outcome kinds prohibited; derived requires lineage digest",
        "access_policy": "access_class=public_redistributable only",
    }
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    _atomic_write_text(manifest_path, json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    metadata["manifest_path"] = manifest_path.name
    metadata["manifest_sha256"] = file_sha256(manifest_path)
    return metadata


class _ParquetBatchCursor:
    """Consume exactly requested row counts while retaining at most one Arrow batch."""

    def __init__(self, parquet_file: Any, *, batch_size: int):
        self._iterator = iter(parquet_file.iter_batches(batch_size=batch_size))
        self._buffer = pd.DataFrame()

    def take(self, count: int) -> pd.DataFrame:
        parts: list[pd.DataFrame] = []
        remaining = count
        while remaining:
            if self._buffer.empty:
                try:
                    self._buffer = next(self._iterator).to_pandas()
                except StopIteration as exc:
                    raise ValueError("Split manifest ended before the source task artifact") from exc
            take = min(remaining, len(self._buffer))
            parts.append(self._buffer.iloc[:take].copy())
            self._buffer = self._buffer.iloc[take:].reset_index(drop=True)
            remaining -= take
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    def assert_exhausted(self) -> None:
        if not self._buffer.empty:
            raise ValueError("Split manifest contains rows beyond the source task artifact")
        try:
            extra = next(self._iterator)
        except StopIteration:
            return
        if extra.num_rows:
            raise ValueError("Split manifest contains rows beyond the source task artifact")


def serialize_model_ready_jsonl_streaming(
    task_parquet_path: Path,
    split_manifest_path: Path,
    output_path: Path,
    *,
    source_dataset_sha256: str,
    split_manifest_sha256: str,
    split_sidecar_sha256: str,
    build_config: Mapping[str, Any],
    batch_size: int = 8_192,
    allow_derived_labels: bool = False,
    task_eligibility_mode: Literal["default", "derived_sensitivity"] = "default",
) -> dict[str, Any]:
    """Stream a task and row-order-bound split to deterministic model-ready JSONL.

    This path never stores the full task, manifest, examples, offsets, or line
    digest list in memory.  It only accepts manifests produced with a complete
    disk-backed record-ID uniqueness audit and an exact source-row binding.
    """

    _require_sha256(source_dataset_sha256, "source_dataset_sha256")
    _require_sha256(split_manifest_sha256, "split_manifest_sha256")
    _require_sha256(split_sidecar_sha256, "split_sidecar_sha256")
    if not 1 <= batch_size <= 65_536:
        raise ValueError("Streaming serialization batch_size must be between 1 and 65536")
    if not task_parquet_path.exists():
        raise FileNotFoundError(task_parquet_path)
    if not split_manifest_path.is_file():
        raise FileNotFoundError(split_manifest_path)
    source_dataset = resolve_manifest_bound_parquet_dataset(task_parquet_path)
    actual_source_sha256 = source_dataset.dataset_sha256
    actual_split_sha256 = file_sha256(split_manifest_path)
    if actual_source_sha256 != source_dataset_sha256:
        raise ValueError("Source task digest does not match source_dataset_sha256")
    if actual_split_sha256 != split_manifest_sha256:
        raise ValueError("Split artifact digest does not match split_manifest_sha256")
    split_sidecar_path = split_manifest_path.with_suffix(split_manifest_path.suffix + ".manifest.json")
    if not split_sidecar_path.is_file():
        raise ValueError("Streaming serialization requires the split integrity sidecar")
    if file_sha256(split_sidecar_path) != split_sidecar_sha256:
        raise ValueError("Split sidecar digest does not match split_sidecar_sha256")
    split_sidecar = json.loads(split_sidecar_path.read_text(encoding="utf-8"))
    if split_sidecar.get("manifest_sha256") != split_manifest_sha256:
        raise ValueError("Split sidecar does not bind the supplied split manifest")
    if split_sidecar.get("source_dataset_sha256") != source_dataset_sha256:
        raise ValueError("Split sidecar does not bind the supplied source task")
    sidecar_dataset_binding = split_sidecar.get("source_dataset")
    expected_dataset_binding = source_dataset.binding_payload()
    if source_dataset.input_kind == "manifest_bound_directory" and sidecar_dataset_binding is None:
        raise ValueError("Partitioned source tasks require a split-sidecar dataset-part binding")
    if sidecar_dataset_binding is not None and sidecar_dataset_binding != expected_dataset_binding:
        raise ValueError("Split sidecar dataset-part binding does not match the supplied source task")
    if split_sidecar.get("record_id_uniqueness") != "complete_disk_backed_primary_key_audit":
        raise ValueError("Streaming serialization requires a complete disk-backed ID uniqueness audit")
    if not str(split_sidecar.get("row_order_binding", "")).startswith("manifest source_row_index"):
        raise ValueError("Streaming serialization requires an exact source-row-order binding")
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - platform dependency profile.
        raise ImportError("pyarrow is required for streaming model-ready serialization") from exc

    split_parquet = pq.ParquetFile(split_manifest_path)
    if source_dataset.total_rows != split_parquet.metadata.num_rows:
        raise ValueError("Source task and split manifest row counts differ")
    required_manifest = {"record_id", "split", "split_name", "strategy", "group_id", "source_row_index"}
    missing_manifest = sorted(required_manifest - set(split_parquet.schema_arrow.names))
    if missing_manifest:
        raise ValueError(f"Streaming split manifest is missing columns: {missing_manifest}")
    source_parquets: list[Any] = []
    reference_source_schema: Any = None
    for part in source_dataset.parts:
        source_parquet = pq.ParquetFile(part.path)
        if reference_source_schema is None:
            reference_source_schema = source_parquet.schema_arrow
        elif not reference_source_schema.equals(
            source_parquet.schema_arrow,
            check_metadata=False,
        ):
            raise ValueError(f"Streaming task Parquet schema changed across parts: {part.relative_path}")
        source_parquets.append(source_parquet)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    manifest_cursor = _ParquetBatchCursor(split_parquet, batch_size=batch_size)
    serialized_partitions: Counter[str] = Counter()
    excluded_partitions: Counter[str] = Counter()
    outcome_kinds: Counter[str] = Counter()
    source_rows = 0
    serialized_rows = 0
    maximum_batch_rows = 0
    maximum_input_batch_deep_bytes = 0
    maximum_example_batch_recursive_bytes = 0
    ordered_line_digest = hashlib.sha256()
    expected_signature: tuple[str, ...] | None = None
    signature_columns = (
        "task_id",
        "task_type",
        "evidence_domain",
        "endpoint",
        "assay_family",
        "label_kind",
        "label_unit",
    )
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            source_batches = (
                batch
                for source_parquet in source_parquets
                for batch in source_parquet.iter_batches(batch_size=batch_size)
            )
            for source_batch in source_batches:
                task_frame = source_batch.to_pandas()
                split_frame = manifest_cursor.take(len(task_frame))
                maximum_batch_rows = max(maximum_batch_rows, len(task_frame))
                maximum_input_batch_deep_bytes = max(
                    maximum_input_batch_deep_bytes,
                    int(
                        task_frame.memory_usage(index=True, deep=True).sum()
                        + split_frame.memory_usage(index=True, deep=True).sum()
                    ),
                )
                source_ids = task_frame["observation_id"].fillna("").astype(str).tolist()
                manifest_ids = split_frame["record_id"].fillna("").astype(str).tolist()
                expected_indices = np.arange(source_rows, source_rows + len(task_frame), dtype=np.int64)
                if source_ids != manifest_ids or not np.array_equal(
                    pd.to_numeric(split_frame["source_row_index"], errors="coerce").to_numpy(),
                    expected_indices,
                ):
                    raise ValueError("Split manifest row binding does not match the source task")
                current_signature = tuple(
                    str(task_frame[column].iloc[0]).strip() for column in signature_columns
                )
                if expected_signature is None:
                    expected_signature = current_signature
                elif expected_signature != current_signature:
                    raise ValueError("Task signature changed across streaming serialization batches")
                examples = model_ready_examples_from_task_view(
                    task_frame,
                    include_graph=False,
                    allow_derived_labels=allow_derived_labels,
                    task_eligibility_mode=task_eligibility_mode,
                )
                maximum_example_batch_recursive_bytes = max(
                    maximum_example_batch_recursive_bytes,
                    _recursive_size_bytes(examples),
                )
                attached = attach_fixed_split(examples, split_frame)
                by_id = {str(example["record_id"]): example for example in attached}
                for record_id in source_ids:
                    example = by_id[record_id]
                    split = str(example["split"]["partition"])
                    if split not in {"train", "validation", "test"}:
                        excluded_partitions[split] += 1
                        continue
                    line = _canonical_json(example)
                    handle.write(line + "\n")
                    ordered_line_digest.update(hashlib.sha256(line.encode()).hexdigest().encode())
                    ordered_line_digest.update(b"\n")
                    serialized_partitions[split] += 1
                    outcome_kinds[str(example["label"]["outcome_kind"])] += 1
                    serialized_rows += 1
                source_rows += len(task_frame)
            if source_rows != source_dataset.total_rows:
                raise ValueError(
                    "Resolved source task row count changed during streaming: "
                    f"expected={source_dataset.total_rows}, observed={source_rows}"
                )
            manifest_cursor.assert_exhausted()
            if not serialized_rows:
                raise ValueError("Streaming serialization produced no train/validation/test records")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)

    metadata = {
        "schema_version": MODEL_READY_SCHEMA_VERSION,
        "serialization_format": SERIALIZATION_FORMAT,
        "serialization_implementation": "bounded_memory_parquet_lockstep_v1",
        "path": output_path.name,
        "source_task_path": Path(
            os.path.relpath(source_dataset.input_path, start=output_path.parent)
        ).as_posix(),
        "source_dataset": source_dataset.binding_payload(),
        "source_dataset_manifest_path": (
            Path(os.path.relpath(source_dataset.manifest_path, start=output_path.parent)).as_posix()
            if source_dataset.manifest_path is not None
            else None
        ),
        "split_manifest_path": Path(
            os.path.relpath(split_manifest_path, start=output_path.parent)
        ).as_posix(),
        "split_sidecar_path": Path(os.path.relpath(split_sidecar_path, start=output_path.parent)).as_posix(),
        "file_sha256": file_sha256(output_path),
        "source_record_count": source_rows,
        "record_count": serialized_rows,
        "excluded_partition_counts": dict(sorted(excluded_partitions.items())),
        "partition_counts": dict(sorted(serialized_partitions.items())),
        "outcome_kind_counts": dict(sorted(outcome_kinds.items())),
        "ordered_line_digest_sha256": ordered_line_digest.hexdigest(),
        "ordered_line_digest_algorithm": "sha256_over_newline_delimited_per_line_sha256_hex",
        "source_dataset_sha256": source_dataset_sha256,
        "split_manifest_sha256": split_manifest_sha256,
        "split_sidecar_sha256": split_sidecar_sha256,
        "build_config": dict(build_config),
        "build_config_sha256": stable_json_digest(build_config),
        "label_policy": (
            "prediction prohibited; default path excludes derived; derived sensitivity requires independent "
            "eligibility and SHA-256 lineage"
        ),
        "access_policy": "access_class=public_redistributable only",
        "bounded_memory": {
            "configured_batch_rows": batch_size,
            "maximum_observed_batch_rows": maximum_batch_rows,
            "maximum_observed_input_and_manifest_pandas_deep_bytes": maximum_input_batch_deep_bytes,
            "maximum_observed_example_batch_recursive_bytes": maximum_example_batch_recursive_bytes,
            "full_record_or_line_digest_lists_materialized": False,
        },
        "substantive_training_started": False,
    }
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    _atomic_write_text(manifest_path, json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    metadata["manifest_path"] = manifest_path.name
    metadata["manifest_sha256"] = file_sha256(manifest_path)
    return metadata


def _recursive_size_bytes(value: Any, seen: set[int] | None = None) -> int:
    seen = seen or set()
    identifier = id(value)
    if identifier in seen:
        return 0
    seen.add(identifier)
    size = sys.getsizeof(value)
    if isinstance(value, Mapping):
        size += sum(
            _recursive_size_bytes(key, seen) + _recursive_size_bytes(item, seen)
            for key, item in value.items()
        )
    elif isinstance(value, (list, tuple, set, frozenset)):
        size += sum(_recursive_size_bytes(item, seen) for item in value)
    return size


def estimate_model_ready_materialization_memory(
    frame: pd.DataFrame,
    *,
    target_row_count: int,
    sample_rows: int = 1_000,
    allow_derived_labels: bool = False,
    task_eligibility_mode: Literal["default", "derived_sensitivity"] = "default",
) -> dict[str, Any]:
    """Sample measured Python-object cost and extrapolate the in-memory risk."""

    if target_row_count < 1 or sample_rows < 1 or frame.empty:
        raise ValueError("A nonempty sample and positive row counts are required")
    sample = frame.iloc[: min(sample_rows, len(frame))].copy()
    examples = model_ready_examples_from_task_view(
        sample,
        allow_derived_labels=allow_derived_labels,
        task_eligibility_mode=task_eligibility_mode,
    )
    input_bytes = int(sample.memory_usage(index=True, deep=True).sum())
    example_bytes = _recursive_size_bytes(examples)
    measured_rows = len(sample)
    combined_per_row = (input_bytes + example_bytes) / measured_rows
    return {
        "estimate_type": "sample_measured_python_object_extrapolation_not_peak_rss",
        "sample_rows": measured_rows,
        "target_row_count": target_row_count,
        "sample_input_pandas_deep_bytes": input_bytes,
        "sample_model_ready_recursive_bytes": example_bytes,
        "combined_measured_bytes_per_row": combined_per_row,
        "extrapolated_combined_gib": combined_per_row * target_row_count / 1024**3,
        "risk": "unbounded_full_list_materialization_prohibited_for_platform_scale",
        "required_path": "serialize_model_ready_jsonl_streaming",
    }


class JsonlDataset(Sequence[dict[str, Any]]):
    """Random-access JSONL reader with optional whole-file integrity check."""

    def __init__(
        self,
        path: Path,
        *,
        expected_sha256: str | None = None,
        partition: Literal["train", "validation", "test"] | None = None,
    ):
        self.path = path
        if not path.is_file():
            raise FileNotFoundError(path)
        if expected_sha256 and file_sha256(path) != expected_sha256:
            raise ValueError("JSONL file digest does not match the expected artifact")
        self._offsets: list[int] = []
        self.partition = partition
        with path.open("rb") as handle:
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if line.strip():
                    payload = json.loads(line)
                    split = payload.get("split")
                    if not isinstance(split, Mapping) or split.get("partition") not in {
                        "train",
                        "validation",
                        "test",
                    }:
                        raise ValueError("JSONL contains an unassigned or excluded train-ready record")
                    if partition is None or split["partition"] == partition:
                        self._offsets.append(offset)

    def __len__(self) -> int:
        return len(self._offsets)

    @overload
    def __getitem__(self, index: int) -> dict[str, Any]: ...

    @overload
    def __getitem__(self, index: slice) -> list[dict[str, Any]]: ...

    def __getitem__(self, index: int | slice) -> dict[str, Any] | list[dict[str, Any]]:
        if isinstance(index, slice):
            return [self[item] for item in range(*index.indices(len(self)))]
        resolved = index + len(self) if index < 0 else index
        if resolved < 0 or resolved >= len(self):
            raise IndexError(index)
        with self.path.open("rb") as handle:
            handle.seek(self._offsets[resolved])
            payload = json.loads(handle.readline())
        validate_model_ready_example(payload)
        return payload

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for index in range(len(self)):
            item = self[index]
            assert isinstance(item, dict)
            yield item


class JsonlIterableDataset(Iterable[dict[str, Any]]):
    """Bounded-memory JSONL iterator for platform-scale training readers."""

    def __init__(
        self,
        path: Path,
        *,
        expected_sha256: str | None = None,
        partition: Literal["train", "validation", "test"] | None = None,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
        if expected_sha256 is not None:
            _require_sha256(expected_sha256, "expected_sha256")
            if file_sha256(path) != expected_sha256:
                raise ValueError("JSONL file digest does not match the expected artifact")
        self.path = path
        self.partition = partition

    def __iter__(self) -> Iterator[dict[str, Any]]:
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                payload = json.loads(line)
                validate_model_ready_example(payload)
                split = payload.get("split")
                if not isinstance(split, Mapping) or split.get("partition") not in {
                    "train",
                    "validation",
                    "test",
                }:
                    raise ValueError(
                        f"JSONL line {line_number} contains an unassigned/excluded train-ready record"
                    )
                if self.partition is None or split["partition"] == self.partition:
                    yield payload


@dataclass(frozen=True)
class Vocabulary:
    modality: str
    token_to_id: dict[str, int]
    fitted_partition: str
    minimum_frequency: int
    training_corpus_sha256: str
    schema_version: str = "static_vocabulary_v1"

    @property
    def pad_id(self) -> int:
        return self.token_to_id["<PAD>"]

    @property
    def unknown_id(self) -> int:
        return self.token_to_id["<UNK>"]

    def digest(self) -> str:
        return stable_json_digest(asdict(self))

    def encode(self, tokens: Sequence[str], *, add_special_tokens: bool = True) -> list[int]:
        ids = [self.token_to_id.get(token, self.unknown_id) for token in tokens]
        if add_special_tokens:
            return [self.token_to_id["<BOS>"], *ids, self.token_to_id["<EOS>"]]
        return ids


def _tokenize_for_modality(value: object, modality: str) -> list[str]:
    if modality == "smiles":
        return tokenize_smiles(value)
    if modality == "protein":
        sequence, invalid = normalize_protein_sequence(value)
        if invalid:
            raise ValueError(f"Invalid protein symbols for vocabulary: {invalid}")
        return list(sequence)
    if modality == "text":
        return str(value).split()
    raise ValueError("modality must be smiles, protein, or text")


def build_training_vocabulary(
    values: Iterable[object],
    *,
    modality: Literal["smiles", "protein", "text"],
    fitted_partition: str,
    minimum_frequency: int = 1,
    maximum_size: int | None = None,
) -> Vocabulary:
    """Fit a static vocabulary on training input only in one bounded-memory pass."""

    if fitted_partition != "train":
        raise ValueError("Vocabulary fitting is permitted on the training partition only")
    if minimum_frequency < 1:
        raise ValueError("minimum_frequency must be positive")
    counts: Counter[str] = Counter()
    corpus_hasher = hashlib.sha256()
    first_document = True
    for value in values:
        tokens = _tokenize_for_modality(value, modality)
        counts.update(tokens)
        if not first_document:
            corpus_hasher.update(b"\n")
        corpus_hasher.update(" ".join(tokens).encode())
        first_document = False
    candidates = [
        token
        for token, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        if count >= minimum_frequency
    ]
    reserved = ["<PAD>", "<UNK>", "<BOS>", "<EOS>", "<MASK>"]
    candidates = [token for token in candidates if token not in set(reserved)]
    if maximum_size is not None:
        if maximum_size < len(reserved):
            raise ValueError("maximum_size is smaller than the reserved vocabulary")
        candidates = candidates[: maximum_size - len(reserved)]
    token_to_id = {token: index for index, token in enumerate([*reserved, *candidates])}
    return Vocabulary(
        modality=modality,
        token_to_id=token_to_id,
        fitted_partition=fitted_partition,
        minimum_frequency=minimum_frequency,
        training_corpus_sha256=corpus_hasher.hexdigest(),
    )


@dataclass(frozen=True)
class CollatorConfig:
    max_smiles_tokens: int = 256
    max_protein_tokens: int = 1024
    max_text_tokens: int = 256
    truncation_policy: Literal["error", "right"] = "error"
    pad_to_multiple_of: int = 8
    include_graph: bool = False

    def validate(self) -> None:
        if min(self.max_smiles_tokens, self.max_protein_tokens, self.max_text_tokens) < 2:
            raise ValueError("Modality maxima must leave room for special tokens")
        if self.pad_to_multiple_of < 1:
            raise ValueError("pad_to_multiple_of must be positive")


def _padded_ids(
    sequences: Sequence[Sequence[int]],
    *,
    pad_id: int,
    maximum: int,
    truncation_policy: str,
    pad_to_multiple_of: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    affected = sum(len(values) > maximum for values in sequences)
    if affected and truncation_policy == "error":
        raise ValueError(f"{affected} sequences exceed the configured maximum {maximum}")
    clipped: list[list[int]] = []
    for values in sequences:
        materialized = list(values)
        if len(materialized) > maximum:
            # Encoded sequences are BOS/content/EOS. Preserve both boundary
            # tokens while hard-truncating content under the explicit policy.
            materialized = [materialized[0], *materialized[1 : maximum - 1], materialized[-1]]
        clipped.append(materialized)
    longest = max((len(values) for values in clipped), default=0)
    width = min(maximum, int(math.ceil(max(longest, 1) / pad_to_multiple_of) * pad_to_multiple_of))
    ids = np.full((len(clipped), width), pad_id, dtype=np.int64)
    mask = np.zeros((len(clipped), width), dtype=np.int8)
    for index, values in enumerate(clipped):
        ids[index, : len(values)] = values
        mask[index, : len(values)] = 1
    return ids, mask, affected


def collate_molecular_graphs(graphs: Sequence[Mapping[str, Any] | None]) -> dict[str, np.ndarray]:
    """Concatenate deterministic molecular graphs with graph membership indices."""

    node_features: list[list[float]] = []
    edge_indices: list[tuple[int, int]] = []
    edge_features: list[list[float]] = []
    node_batch: list[int] = []
    graph_present: list[bool] = []
    graph_failure_reasons: list[str] = []
    node_offset = 0
    hybridization_vocabulary = {
        "UNSPECIFIED": 0,
        "S": 1,
        "SP": 2,
        "SP2": 3,
        "SP3": 4,
        "SP3D": 5,
        "SP3D2": 6,
    }
    bond_vocabulary = {"SINGLE": 1, "DOUBLE": 2, "TRIPLE": 3, "AROMATIC": 4}
    for batch_index, graph in enumerate(graphs):
        if not graph or not graph.get("valid"):
            graph_present.append(False)
            graph_failure_reasons.append(
                str(graph.get("error", "missing_graph")) if graph else "missing_graph"
            )
            continue
        graph_present.append(True)
        graph_failure_reasons.append("")
        nodes = list(graph.get("nodes", []))
        for node in nodes:
            node_features.append(
                [
                    float(node["atomic_number"]),
                    float(node["formal_charge"]),
                    float(node["total_degree"]),
                    float(node["total_hydrogens"]),
                    float(bool(node["is_aromatic"])),
                    float(bool(node["is_in_ring"])),
                    float(hybridization_vocabulary.get(str(node["hybridization"]), 0)),
                ]
            )
            node_batch.append(batch_index)
        for edge in graph.get("edges", []):
            edge_indices.append((node_offset + int(edge["source"]), node_offset + int(edge["target"])))
            edge_features.append(
                [
                    float(bond_vocabulary.get(str(edge["bond_type"]), 0)),
                    float(bool(edge["is_aromatic"])),
                    float(bool(edge["is_conjugated"])),
                    float(bool(edge["is_in_ring"])),
                ]
            )
        node_offset += len(nodes)
    return {
        "node_features": np.asarray(node_features, dtype=np.float32).reshape((-1, 7)),
        "edge_index": np.asarray(edge_indices, dtype=np.int64).reshape((-1, 2)).T,
        "edge_features": np.asarray(edge_features, dtype=np.float32).reshape((-1, 4)),
        "node_batch": np.asarray(node_batch, dtype=np.int64),
        "graph_present_mask": np.asarray(graph_present, dtype=bool),
        "graph_failure_reasons": np.asarray(graph_failure_reasons, dtype=object),
        "n_graphs": np.asarray([sum(graph_present)], dtype=np.int64),
        "batch_size": np.asarray([len(graphs)], dtype=np.int64),
    }


class MultimodalCollator:
    """Dependency-light collator returning NumPy arrays and truncation counts."""

    def __init__(
        self,
        *,
        smiles_vocabulary: Vocabulary,
        protein_vocabulary: Vocabulary,
        text_vocabulary: Vocabulary,
        config: CollatorConfig | None = None,
    ):
        config = config or CollatorConfig()
        config.validate()
        if smiles_vocabulary.modality != "smiles":
            raise ValueError("smiles_vocabulary has the wrong modality")
        if protein_vocabulary.modality != "protein":
            raise ValueError("protein_vocabulary has the wrong modality")
        if text_vocabulary.modality != "text":
            raise ValueError("text_vocabulary has the wrong modality")
        self.vocabularies = {
            "smiles": smiles_vocabulary,
            "protein": protein_vocabulary,
            "text": text_vocabulary,
        }
        self.config = config

    def __call__(self, examples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        if not examples:
            raise ValueError("Cannot collate an empty batch")
        for example in examples:
            validate_model_ready_example(example)
        encoded: dict[str, list[list[int]]] = {"smiles": [], "protein": [], "text": []}
        for example in examples:
            inputs = example["inputs"]
            for modality, key in (
                ("smiles", "smiles"),
                ("protein", "protein_sequence"),
                ("text", "text"),
            ):
                tokens = _tokenize_for_modality(inputs.get(key, ""), modality)
                encoded[modality].append(self.vocabularies[modality].encode(tokens))
        modality_present = {
            "smiles": np.asarray(
                [bool(str(item["inputs"].get("smiles", "")).strip()) for item in examples], dtype=bool
            ),
            "protein": np.asarray(
                [
                    item["inputs"].get("protein_sequence_status", "valid") == "valid"
                    and bool(str(item["inputs"].get("protein_sequence", "")).strip())
                    for item in examples
                ],
                dtype=bool,
            ),
            "text": np.asarray(
                [bool(str(item["inputs"].get("text", "")).strip()) for item in examples], dtype=bool
            ),
        }
        batch: dict[str, Any] = {
            "record_ids": np.asarray([str(item["record_id"]) for item in examples], dtype=object),
            "task_ids": np.asarray([str(item["task_id"]) for item in examples], dtype=object),
            "outcome_kinds": np.asarray(
                [_canonical_outcome_kind(item["label"]["outcome_kind"]) for item in examples],
                dtype=object,
            ),
        }
        maxima = {
            "smiles": self.config.max_smiles_tokens,
            "protein": self.config.max_protein_tokens,
            "text": self.config.max_text_tokens,
        }
        truncation_counts: dict[str, int] = {}
        for modality in ("smiles", "protein", "text"):
            ids, mask, affected = _padded_ids(
                encoded[modality],
                pad_id=self.vocabularies[modality].pad_id,
                maximum=maxima[modality],
                truncation_policy=self.config.truncation_policy,
                pad_to_multiple_of=self.config.pad_to_multiple_of,
            )
            batch[f"{modality}_input_ids"] = ids
            batch[f"{modality}_attention_mask"] = mask
            batch[f"{modality}_present_mask"] = modality_present[modality]
            batch[f"{modality}_attention_mask"][~modality_present[modality], :] = 0
            truncation_counts[modality] = affected
        labels: list[float] = []
        lower: list[float] = []
        upper: list[float] = []
        relations: list[str] = []
        for example in examples:
            label = example["label"]
            value = label.get("value")
            labels.append(float(value) if value is not None else np.nan)
            lo = label.get("lower_bound")
            hi = label.get("upper_bound")
            lower.append(float(lo) if lo is not None else -np.inf)
            upper.append(float(hi) if hi is not None else np.inf)
            relations.append(str(label.get("relation", "=")))
        batch["labels"] = np.asarray(labels, dtype=np.float32)
        batch["label_texts"] = np.asarray(
            [str(example["label"].get("text", "")) for example in examples], dtype=object
        )
        batch["label_mask"] = np.isfinite(batch["labels"])
        batch["label_lower_bound"] = np.asarray(lower, dtype=np.float32)
        batch["label_upper_bound"] = np.asarray(upper, dtype=np.float32)
        batch["label_relations"] = np.asarray(relations, dtype=object)
        batch["truncation_counts"] = truncation_counts
        batch["collator_config_sha256"] = stable_json_digest(asdict(self.config))
        if self.config.include_graph:
            graphs: list[Mapping[str, Any] | None] = []
            for item in examples:
                inputs = item["inputs"]
                graph = inputs.get("molecular_graph")
                if graph is None:
                    graph = {
                        "valid": False,
                        "error": str(inputs.get("molecular_graph_failure_reason", "missing_graph")),
                    }
                graphs.append(graph)
            batch["graph"] = collate_molecular_graphs(graphs)
        return batch


@dataclass(frozen=True)
class ContrastivePairBuild:
    pairs: tuple[dict[str, Any], ...]
    metadata: dict[str, Any]


def observed_contrastive_pairs(
    examples: Sequence[Mapping[str, Any]],
    *,
    seed: int = 20260804,
    negatives_per_positive: int = 1,
    positive_values: Sequence[object] = (1,),
    negative_values: Sequence[object] = (0,),
) -> ContrastivePairBuild:
    """Pair observed positives only with observed negatives from the same task."""

    if negatives_per_positive < 1:
        raise ValueError("negatives_per_positive must be positive")
    if not positive_values or not negative_values:
        raise ValueError("Explicit positive_values and negative_values are required")
    positive_keys = {str(value) for value in positive_values}
    negative_keys = {str(value) for value in negative_values}
    if positive_keys & negative_keys:
        raise ValueError("Positive and negative label mappings must be disjoint")
    by_task: dict[str, dict[int, list[Mapping[str, Any]]]] = {}
    exclusions: Counter[str] = Counter()
    for example in examples:
        validate_model_ready_example(example)
        split = example.get("split")
        if not isinstance(split, Mapping) or split.get("partition") != "train":
            raise ValueError("Observed contrastive pairs require examples from one fixed train partition")
        if _canonical_outcome_kind(example["label"]["outcome_kind"]) not in {
            "experimental_raw",
            "experimental_summary",
            "curated_assertion",
        }:
            exclusions["non_observed_or_curated_assertion"] += 1
            continue
        if str(example["label"]["label_kind"]) != "categorical":
            exclusions["non_categorical"] += 1
            continue
        value = example["label"].get("value")
        if value is None:
            value = example["label"].get("text", "")
        key = str(value)
        if key in negative_keys:
            encoded_class = 0
        elif key in positive_keys:
            encoded_class = 1
        else:
            exclusions["unmapped_categorical_value"] += 1
            continue
        by_task.setdefault(str(example["task_id"]), {0: [], 1: []})[encoded_class].append(example)
    rng = random.Random(seed)
    pairs: list[dict[str, Any]] = []
    for task_id, classes in sorted(by_task.items()):
        negatives = sorted(classes[0], key=lambda item: str(item["record_id"]))
        positives = sorted(classes[1], key=lambda item: str(item["record_id"]))
        if not negatives:
            continue
        for positive in positives:
            candidates = negatives.copy()
            rng.shuffle(candidates)
            for negative in candidates[:negatives_per_positive]:
                pairs.append(
                    {
                        "task_id": task_id,
                        "positive_record_id": str(positive["record_id"]),
                        "negative_record_id": str(negative["record_id"]),
                        "positive_outcome_kind": positive["label"]["outcome_kind"],
                        "negative_outcome_kind": negative["label"]["outcome_kind"],
                        "sampling_semantics": "both_classes_are_observed_or_curated_assertions",
                        "seed": seed,
                    }
                )
    return ContrastivePairBuild(
        pairs=tuple(pairs),
        metadata={
            "seed": seed,
            "negatives_per_positive": negatives_per_positive,
            "positive_values": [str(value) for value in positive_values],
            "negative_values": [str(value) for value in negative_values],
            "n_input_examples": len(examples),
            "n_pairs": len(pairs),
            "exclusion_counts": dict(sorted(exclusions.items())),
            "negative_semantics": "observed_or_curated_class_only; no unlabeled pair is called negative",
        },
    )


def unlabeled_in_batch_candidates(examples: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Prepare nonmatching pairs without mislabeling them as experimental negatives."""

    for example in examples:
        split = example.get("split")
        if not isinstance(split, Mapping) or split.get("partition") != "train":
            raise ValueError("Unlabeled in-batch candidates require examples from one fixed train partition")
    ordered = sorted(examples, key=lambda item: str(item["record_id"]))
    if len(ordered) < 2:
        return []
    candidates: list[dict[str, Any]] = []
    for index, anchor in enumerate(ordered):
        other = ordered[(index + 1) % len(ordered)]
        if anchor.get("protein_id") == other.get("protein_id") and anchor.get("molecule_id") == other.get(
            "molecule_id"
        ):
            continue
        candidates.append(
            {
                "anchor_record_id": str(anchor["record_id"]),
                "candidate_record_id": str(other["record_id"]),
                "sampling_semantics": "unlabeled_nonmatching_candidate_not_a_negative_observation",
            }
        )
    return candidates


@dataclass(frozen=True)
class CandidateModelProfile:
    key: str
    architecture_family: str
    modalities: tuple[str, ...]
    compatible_task_types: tuple[str, ...]
    parameter_count: int | None
    pretrained_identifier: str
    identifier_status: str
    peft_support: tuple[str, ...]
    precision_support: tuple[str, ...]
    maximum_lengths: dict[str, int | None]
    license_status: str
    training_cutoff_status: str
    required_preflight: tuple[str, ...]
    scientific_role: str
    reference_url: str = ""
    use_mode: str = "future_training_candidate"


def candidate_model_registry() -> tuple[CandidateModelProfile, ...]:
    """Compatibility registry; unresolved identifiers cannot pass preflight."""

    common_tasks = ("regression", "binary_classification", "multiclass_classification", "ranking")
    return (
        CandidateModelProfile(
            key="chemprop_dmpnn",
            architecture_family="directed_message_passing_neural_network",
            modalities=("molecular_graph",),
            compatible_task_types=common_tasks,
            parameter_count=None,
            pretrained_identifier="none_supervised_from_scratch",
            identifier_status="architecture_verified_version_to_pin",
            peft_support=("full",),
            precision_support=("fp32", "bf16", "fp16"),
            maximum_lengths={"molecule_atoms": None},
            license_status="verify_package_and_checkpoint_before_run",
            training_cutoff_status="not_applicable_without_pretrained_checkpoint",
            required_preflight=("pin_chemprop_version", "graph_schema_compatibility", "license_review"),
            scientific_role="graph baseline beyond fingerprints",
        ),
        CandidateModelProfile(
            key="molecule_transformer",
            architecture_family="pretrained_smiles_or_molecular_transformer",
            modalities=("smiles",),
            compatible_task_types=common_tasks,
            parameter_count=None,
            pretrained_identifier="selection_pending_model_card_license_and_cutoff_audit",
            identifier_status="unresolved_blocks_training",
            peft_support=("linear_probe", "frozen", "lora", "full"),
            precision_support=("fp32", "bf16", "fp16"),
            maximum_lengths={"smiles_tokens": None},
            license_status="unresolved",
            training_cutoff_status="unresolved",
            required_preflight=(
                "select_exact_checkpoint",
                "license_review",
                "training_overlap_audit",
                "token_length_check",
            ),
            scientific_role="learned molecular sequence representation",
        ),
        CandidateModelProfile(
            key="protein_transformer",
            architecture_family="pretrained_protein_language_model",
            modalities=("protein_sequence",),
            compatible_task_types=common_tasks,
            parameter_count=None,
            pretrained_identifier="selection_pending_model_card_license_and_cutoff_audit",
            identifier_status="unresolved_blocks_training",
            peft_support=("linear_probe", "frozen", "lora", "full"),
            precision_support=("fp32", "bf16", "fp16"),
            maximum_lengths={"protein_tokens": None},
            license_status="unresolved",
            training_cutoff_status="unresolved",
            required_preflight=(
                "select_exact_checkpoint",
                "license_review",
                "sequence_length_check",
                "training_overlap_audit",
            ),
            scientific_role="protein representation for cross-target generalization",
        ),
        CandidateModelProfile(
            key="dual_encoder_fusion",
            architecture_family="molecule_and_protein_encoder_with_task_heads",
            modalities=("smiles_or_graph", "protein_sequence", "assay_context_optional"),
            compatible_task_types=common_tasks,
            parameter_count=None,
            pretrained_identifier="composed_from_separately_audited_encoders",
            identifier_status="design_ready_components_unresolved",
            peft_support=("frozen", "linear_probe", "lora", "full"),
            precision_support=("fp32", "bf16", "fp16"),
            maximum_lengths={"smiles_tokens": None, "protein_tokens": None},
            license_status="intersection_of_component_licenses_unresolved",
            training_cutoff_status="component_specific_unresolved",
            required_preflight=(
                "component_audits",
                "fusion_ablation",
                "double_cold_split_support",
                "missing_modality_policy",
            ),
            scientific_role="primary protein-agnostic supervised architecture",
        ),
        CandidateModelProfile(
            key="structure_conditioned_complex",
            architecture_family="protein_ligand_complex_or_cofolding_model",
            modalities=("molecular_graph", "protein_sequence", "protein_coordinates"),
            compatible_task_types=("regression", "ranking", "pose_confidence"),
            parameter_count=None,
            pretrained_identifier="selection_pending_exact_model_and_checkpoint_audit",
            identifier_status="unresolved_blocks_training",
            peft_support=("frozen", "linear_probe", "lora_if_supported"),
            precision_support=("bf16", "fp16", "fp32_smoke_only"),
            maximum_lengths={"protein_tokens": None, "molecule_atoms": None},
            license_status="unresolved",
            training_cutoff_status="critical_unresolved",
            required_preflight=(
                "exact_model_identification",
                "license_review",
                "PDB_and_ligand_overlap_audit",
                "pose_validity_gate",
            ),
            scientific_role="incremental structural value after simpler baselines",
        ),
        CandidateModelProfile(
            key="chai_1_external_evaluation",
            architecture_family="protein_ligand_structure_prediction",
            modalities=("smiles_or_molecular_graph", "protein_sequence"),
            compatible_task_types=("pose_confidence", "external_structure_evaluation"),
            parameter_count=None,
            pretrained_identifier=(
                "official chai-lab Chai-1 distribution (repository reports chai_lab==0.6.1); "
                "exact package, weights revision, and SHA-256 remain to pin"
            ),
            identifier_status="named_external_candidate_revision_unpinned_blocks_evaluation",
            peft_support=("frozen",),
            precision_support=("bf16", "fp16"),
            maximum_lengths={"protein_tokens": None, "molecule_atoms": None},
            license_status=(
                "Apache-2.0 reported by the official repository for code/model; retain an explicit "
                "release-specific legal review and immutable license snapshot"
            ),
            training_cutoff_status="overlap_audit_required",
            required_preflight=(
                "pin_exact_chai_1_release_and_weights",
                "record_license_review",
                "audit_structure_and_ligand_training_overlap",
                "freeze_external_evaluation_protocol",
            ),
            scientific_role="frozen external pose/structure comparator, not project center",
            reference_url="https://github.com/chaidiscovery/chai-lab",
            use_mode="frozen_external_evaluation_only",
        ),
        CandidateModelProfile(
            key="boltz_2_external_evaluation",
            architecture_family="biomolecular_complex_and_affinity_prediction",
            modalities=("smiles_or_molecular_graph", "protein_sequence"),
            compatible_task_types=(
                "regression",
                "ranking",
                "pose_confidence",
                "external_structure_evaluation",
            ),
            parameter_count=None,
            pretrained_identifier=(
                "official Boltz-2 distribution; exact repository release, weights revision, and SHA-256 remain to pin"
            ),
            identifier_status="named_external_candidate_revision_unpinned_blocks_evaluation",
            peft_support=("frozen",),
            precision_support=("bf16", "fp16"),
            maximum_lengths={"protein_tokens": None, "molecule_atoms": None},
            license_status=(
                "MIT reported by the official repository for code and weights; retain release-specific "
                "legal review and immutable license snapshots"
            ),
            training_cutoff_status="overlap_audit_required",
            required_preflight=(
                "pin_exact_boltz_2_release_and_weights",
                "record_code_and_weight_license_review",
                "audit_structure_affinity_and_ligand_training_overlap",
                "freeze_external_evaluation_protocol",
            ),
            scientific_role=(
                "frozen cofolding comparator plus its documented binder probability and log10(IC50 in micromolar) "
                "outputs; never reinterpret that value as Kd or standard binding free energy"
            ),
            reference_url="https://github.com/jwohlwend/boltz",
            use_mode="frozen_external_evaluation_only",
        ),
        CandidateModelProfile(
            key="nesso_1_external_evaluation",
            architecture_family="biomolecular_structure_and_affinity_prediction",
            modalities=("smiles_or_molecular_graph", "protein_sequence"),
            compatible_task_types=(
                "regression",
                "ranking",
                "pose_confidence",
                "external_structure_evaluation",
            ),
            parameter_count=None,
            pretrained_identifier=(
                "official recursionpharma/nesso Nesso-1 distribution; exact checkpoint revision and SHA-256 remain to pin"
            ),
            identifier_status="named_external_candidate_revision_unpinned_blocks_evaluation",
            peft_support=("frozen",),
            precision_support=("bf16", "fp16"),
            maximum_lengths={"protein_tokens": None, "molecule_atoms": None},
            license_status=(
                "Apache-2.0 reported by the official model card; retain release-specific code/weight "
                "legal review and immutable license snapshots"
            ),
            training_cutoff_status="overlap_audit_required",
            required_preflight=(
                "pin_exact_release_and_weights",
                "record_code_and_weight_license_review",
                "audit_structure_affinity_and_ligand_training_overlap",
                "freeze_external_evaluation_protocol",
            ),
            scientific_role=(
                "frozen mixed potency/affinity-score comparator; never reinterpret mixed Ki/Kd/IC50/EC50 output "
                "as Kd, standard binding free energy, or endpoint-specific ground truth"
            ),
            reference_url="https://huggingface.co/recursionpharma/nesso",
            use_mode="frozen_external_evaluation_only_if_verified",
        ),
    )


@dataclass(frozen=True)
class LoraConfig:
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: tuple[str, ...] = ("query", "key", "value", "dense")
    bias: Literal["none", "all", "lora_only"] = "none"

    def validate(self) -> None:
        if self.rank < 1 or self.alpha < 1 or not 0 <= self.dropout < 1:
            raise ValueError("Invalid LoRA rank, alpha, or dropout")
        if not self.target_modules:
            raise ValueError("LoRA target_modules must be verified against the selected model")


@dataclass(frozen=True)
class FineTuningConfig:
    experiment_name: str
    candidate_key: str
    task_ids: tuple[str, ...]
    dataset_sha256: str
    split_manifest_sha256: str
    tokenizer_sha256: str
    loss_name: str
    censoring_policy: Literal["censored_likelihood", "exact_only", "not_applicable"]
    class_weight_policy: Literal[
        "natural_prevalence",
        "balanced_training_only",
        "focal_training_only",
        "not_applicable",
    ]
    model_selection_metric: str
    model_selection_direction: Literal["minimize", "maximize"]
    seed: int = 20260804
    precision: Literal["fp32", "bf16", "fp16"] = "bf16"
    parameter_strategy: Literal["full", "frozen", "linear_probe", "lora"] = "lora"
    lora: LoraConfig | None = field(default_factory=LoraConfig)
    per_device_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    optimizer_name: Literal["adamw"] = "adamw"
    scheduler_name: Literal[
        "cosine_with_warmup",
        "linear_with_warmup",
        "constant_with_warmup",
    ] = "cosine_with_warmup"
    maximum_epochs: int = 20
    warmup_fraction: float = 0.05
    evaluation_interval_steps: int = 250
    checkpoint_interval_steps: int = 250
    keep_last_checkpoints: int = 3
    early_stopping_patience_evaluations: int = 8
    logging_backend: Literal["jsonl", "mlflow", "weights_and_biases"] = "jsonl"
    logging_interval_steps: int = 25
    evaluation_partition: Literal["validation"] = "validation"
    locked_test_policy: Literal["once_after_frozen_model_selection"] = "once_after_frozen_model_selection"
    save_optimizer_scheduler_rng_state: bool = True
    dataloader_workers: int = 4
    deterministic_algorithms: bool = True
    gradient_checkpointing: bool = True
    clip_gradient_norm: float = 1.0
    resume_policy: Literal["never", "exact_contract_only"] = "exact_contract_only"
    substantive_training_authorized: bool = False

    def validate(self, candidate: CandidateModelProfile) -> None:
        if not self.experiment_name.strip() or not self.task_ids:
            raise ValueError("experiment_name and task_ids are required")
        if any(not task_id.strip() for task_id in self.task_ids) or len(set(self.task_ids)) != len(
            self.task_ids
        ):
            raise ValueError("task_ids must be nonblank and unique")
        for digest in (self.dataset_sha256, self.split_manifest_sha256, self.tokenizer_sha256):
            if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest.lower()):
                raise ValueError("Dataset, split, and tokenizer digests must be SHA-256 hex strings")
        if self.candidate_key != candidate.key:
            raise ValueError("Fine-tuning config candidate does not match profile")
        if self.precision not in candidate.precision_support:
            raise ValueError("Requested precision is incompatible with candidate profile")
        if self.parameter_strategy not in candidate.peft_support:
            raise ValueError("Requested parameter strategy is incompatible with candidate profile")
        if self.parameter_strategy == "lora":
            if self.lora is None:
                raise ValueError("LoRA strategy requires a LoRA config")
            self.lora.validate()
        if not self.loss_name.strip() or not self.model_selection_metric.strip():
            raise ValueError("Task loss and model-selection metric must be declared")
        if self.loss_name in {"unresolved", "task_specific_unresolved"}:
            raise ValueError("Task loss must be resolved before training readiness")
        if self.censoring_policy == "censored_likelihood" and "censor" not in self.loss_name.lower():
            raise ValueError("Censored tasks require an explicitly censored loss")
        if (
            min(
                self.per_device_batch_size,
                self.gradient_accumulation_steps,
                self.maximum_epochs,
                self.evaluation_interval_steps,
                self.checkpoint_interval_steps,
                self.keep_last_checkpoints,
                self.early_stopping_patience_evaluations,
                self.logging_interval_steps,
            )
            < 1
        ):
            raise ValueError("Batch, epoch, evaluation, checkpoint, and stopping values must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0 or not 0 <= self.warmup_fraction < 1:
            raise ValueError("Invalid optimizer or warmup settings")
        if self.dataloader_workers < 0 or self.clip_gradient_norm <= 0:
            raise ValueError("dataloader_workers must be non-negative and clip_gradient_norm positive")
        if self.substantive_training_authorized:
            raise PermissionError(
                "This readiness program prohibits substantive training; authorization must occur in a later run"
            )


def checkpoint_contract(
    config: FineTuningConfig,
    *,
    base_model_revision: str,
    code_commit: str,
) -> dict[str, Any]:
    """Required checkpoint contents and exact-match resume keys."""

    if not base_model_revision.strip() or not code_commit.strip():
        raise ValueError("Base-model revision and code commit are required")
    config_payload = asdict(config)
    return {
        "schema_version": "checkpoint_contract_v1",
        "experiment_name": config.experiment_name,
        "base_model_revision": base_model_revision,
        "code_commit": code_commit,
        "dataset_sha256": config.dataset_sha256,
        "split_manifest_sha256": config.split_manifest_sha256,
        "tokenizer_sha256": config.tokenizer_sha256,
        "training_config_sha256": stable_json_digest(config_payload),
        "required_state": [
            "model_or_adapter_weights",
            "optimizer_state",
            "scheduler_state",
            "gradient_scaler_state_if_fp16",
            "epoch_global_step_and_best_metric",
            "python_numpy_and_framework_rng_states",
            "sampler_state_or_deterministic_epoch_seed",
        ],
        "resume_policy": "all_digest_and_revision_keys_must_match_exactly",
    }


def validate_resume_checkpoint(expected: Mapping[str, Any], actual: Mapping[str, Any]) -> None:
    """Reject partial or cross-dataset resume attempts."""

    keys = (
        "schema_version",
        "experiment_name",
        "base_model_revision",
        "code_commit",
        "dataset_sha256",
        "split_manifest_sha256",
        "tokenizer_sha256",
        "training_config_sha256",
    )
    mismatches = [key for key in keys if expected.get(key) != actual.get(key)]
    missing_state = sorted(set(expected.get("required_state", ())) - set(actual.get("saved_state", ())))
    if mismatches or missing_state:
        raise ValueError(
            f"Checkpoint resume contract failed: mismatches={mismatches}, missing_state={missing_state}"
        )


def estimate_training_resources(
    *,
    parameter_count: int,
    trainable_parameter_count: int,
    precision: Literal["fp32", "bf16", "fp16"],
    batch_size_per_device: int,
    sequence_tokens_per_example: int,
    hidden_size: int,
    layer_count: int,
    gradient_checkpointing: bool,
    device_count: int = 1,
) -> dict[str, Any]:
    """Analytic memory/storage estimate with components and assumptions exposed."""

    values = (
        parameter_count,
        trainable_parameter_count,
        batch_size_per_device,
        sequence_tokens_per_example,
        hidden_size,
        layer_count,
        device_count,
    )
    if any(value < 1 for value in values) or trainable_parameter_count > parameter_count:
        raise ValueError("Resource estimate inputs must be positive and trainable <= total parameters")
    weight_bytes = 4 if precision == "fp32" else 2
    model_weights = parameter_count * weight_bytes
    gradients = trainable_parameter_count * weight_bytes
    # AdamW moments (8 bytes) plus fp32 master weights for mixed precision (4 bytes).
    optimizer = trainable_parameter_count * (8 + (4 if precision != "fp32" else 0))
    activation_multiplier = 5.0 if gradient_checkpointing else 12.0
    activations = (
        batch_size_per_device
        * sequence_tokens_per_example
        * hidden_size
        * layer_count
        * weight_bytes
        * activation_multiplier
    )
    subtotal = model_weights + gradients + optimizer + activations
    framework_overhead = subtotal * 0.20
    total_per_device = (
        model_weights + (gradients + optimizer) / device_count + activations + framework_overhead
    )
    checkpoint_bytes = model_weights + gradients + optimizer + max(16 * 1024**2, model_weights * 0.01)
    return {
        "estimate_type": "analytic_scenario_not_measured_runtime_or_allocation",
        "parameter_count": parameter_count,
        "trainable_parameter_count": trainable_parameter_count,
        "trainable_fraction": trainable_parameter_count / parameter_count,
        "precision": precision,
        "device_count": device_count,
        "component_gib": {
            "model_weights_replicated": model_weights / 1024**3,
            "gradients_total": gradients / 1024**3,
            "adamw_and_master_weights_total": optimizer / 1024**3,
            "activations_per_device": activations / 1024**3,
            "framework_fragmentation_margin": framework_overhead / 1024**3,
        },
        "estimated_peak_gib_per_device": total_per_device / 1024**3,
        "estimated_full_checkpoint_gib": checkpoint_bytes / 1024**3,
        "activation_formula_multiplier": activation_multiplier,
        "assumptions": [
            "data-parallel replicated base weights",
            "optimizer and gradients evenly sharded across devices for multi-device scenario",
            "AdamW optimizer",
            "20 percent framework and fragmentation margin",
            "does not include model-specific attention workspaces or dataloader cache",
        ],
    }


def estimate_runtime_scenario(
    *,
    n_training_examples: int,
    maximum_epochs: int,
    effective_batch_size: int,
    assumed_examples_per_second_low: float,
    assumed_examples_per_second_high: float,
    checkpoint_count: int,
    checkpoint_size_gib: float,
) -> dict[str, Any]:
    """Scenario range only; throughput must come from a later hardware dry run."""

    if min(n_training_examples, maximum_epochs, effective_batch_size, checkpoint_count) < 1:
        raise ValueError("Runtime scenario counts must be positive")
    if not 0 < assumed_examples_per_second_low <= assumed_examples_per_second_high:
        raise ValueError("Throughput assumptions must be positive and ordered")
    examples_seen = n_training_examples * maximum_epochs
    steps = math.ceil(examples_seen / effective_batch_size)
    lower_seconds = examples_seen / assumed_examples_per_second_high
    upper_seconds = examples_seen / assumed_examples_per_second_low
    return {
        "estimate_type": "uncalibrated_scenario_range_not_measured_runtime",
        "n_training_examples": n_training_examples,
        "maximum_epochs": maximum_epochs,
        "effective_batch_size": effective_batch_size,
        "optimizer_steps_upper_bound": steps,
        "examples_seen_upper_bound": examples_seen,
        "assumed_examples_per_second_range": [
            assumed_examples_per_second_low,
            assumed_examples_per_second_high,
        ],
        "runtime_hours_range": [lower_seconds / 3600, upper_seconds / 3600],
        "retained_checkpoint_storage_gib": checkpoint_count * checkpoint_size_gib,
        "calibration_requirement": "replace throughput assumptions with tiny dry-run measurement on exact hardware/model",
    }


def training_readiness_check(
    candidate: CandidateModelProfile,
    config: FineTuningConfig,
    *,
    available_modalities: Iterable[str],
    available_hardware: Mapping[str, Any],
    completed_preflight: Iterable[str] = (),
) -> dict[str, Any]:
    """Return blocking preflight conditions without launching training."""

    blockers: list[str] = []
    configuration_errors: list[str] = []
    try:
        config.validate(candidate)
    except (ValueError, PermissionError) as exc:
        configuration_errors.append(str(exc))
        blockers.append(f"invalid_training_config:{exc}")
    completed = set(completed_preflight)
    blockers.extend(
        f"unmet_preflight:{item}" for item in candidate.required_preflight if item not in completed
    )
    modalities = set(available_modalities)
    for required in candidate.modalities:
        if required.endswith("_optional"):
            continue
        if required == "smiles_or_molecular_graph":
            alternatives = {"smiles", "molecular_graph"}
        elif required == "smiles_or_graph":
            alternatives = {"smiles", "molecular_graph", "graph"}
        else:
            alternatives = {required}
        if not alternatives & modalities and required not in modalities:
            blockers.append(f"missing_modality:{required}")
    if "unresolved" in candidate.identifier_status:
        blockers.append("exact_pretrained_identifier_unresolved")
    if candidate.license_status == "unresolved" or "unresolved" in candidate.license_status:
        blockers.append("license_unresolved")
    if "unresolved" in candidate.training_cutoff_status:
        blockers.append("training_cutoff_or_overlap_unresolved")
    if config.precision == "bf16" and not bool(available_hardware.get("bf16_supported", False)):
        blockers.append("bf16_not_confirmed_on_hardware")
    if int(available_hardware.get("device_count", 0)) < 1:
        blockers.append("no_accelerator_inventory")
    return {
        "candidate_key": candidate.key,
        "experiment_name": config.experiment_name,
        "configuration_complete": not configuration_errors,
        "ready_for_substantive_training": False,
        "blockers": sorted(set(blockers)),
        "required_preflight": list(candidate.required_preflight),
        "training_status": "prohibited_during_current_pretraining_readiness_program",
        "interpretation": (
            "absence of a blocker means configuration completeness only, not scientific or operational authorization"
        ),
    }


def loader_smoke_test(
    dataset: Sequence[Mapping[str, Any]],
    collator: MultimodalCollator,
    *,
    batch_size: int = 4,
    maximum_batches: int = 2,
) -> dict[str, Any]:
    """Strictly capped data-loader smoke; produces no model-performance metric."""

    if batch_size < 1 or maximum_batches < 1 or batch_size * maximum_batches > 32:
        raise ValueError("Loader smoke is capped at 32 examples")
    batches = 0
    examples_seen = 0
    shapes: list[dict[str, Any]] = []
    for start in range(0, min(len(dataset), batch_size * maximum_batches), batch_size):
        items = [dataset[index] for index in range(start, min(start + batch_size, len(dataset)))]
        batch = collator(items)
        batches += 1
        examples_seen += len(items)
        shapes.append(
            {key: list(value.shape) for key, value in batch.items() if isinstance(value, np.ndarray)}
        )
    return {
        "status": "passed" if batches else "failed_empty_dataset",
        "smoke_only": True,
        "performance_evidence": False,
        "batches": batches,
        "examples_seen": examples_seen,
        "batch_shapes": shapes,
        "maximum_examples_guard": 32,
    }


def streaming_loader_smoke_test(
    dataset: Iterable[Mapping[str, Any]],
    collator: MultimodalCollator,
    *,
    batch_size: int = 4,
    maximum_batches: int = 2,
) -> dict[str, Any]:
    """Strictly capped smoke test that never indexes or counts the full dataset."""

    if batch_size < 1 or maximum_batches < 1 or batch_size * maximum_batches > 32:
        raise ValueError("Streaming loader smoke is capped at 32 examples")
    iterator = iter(dataset)
    batches = 0
    examples_seen = 0
    shapes: list[dict[str, Any]] = []
    for _ in range(maximum_batches):
        items: list[Mapping[str, Any]] = []
        for _ in range(batch_size):
            try:
                items.append(next(iterator))
            except StopIteration:
                break
        if not items:
            break
        batch = collator(items)
        batches += 1
        examples_seen += len(items)
        shapes.append(
            {key: list(value.shape) for key, value in batch.items() if isinstance(value, np.ndarray)}
        )
    return {
        "status": "passed" if batches else "failed_empty_dataset",
        "smoke_only": True,
        "performance_evidence": False,
        "streaming": True,
        "batches": batches,
        "examples_seen": examples_seen,
        "batch_shapes": shapes,
        "maximum_examples_guard": 32,
    }


def tiny_training_interface_smoke(
    batch: Mapping[str, Any],
    *,
    maximum_steps: int = 2,
    seed: int = 20260804,
) -> dict[str, Any]:
    """Two-step tiny Torch wiring check, guarded against meaningful training."""

    if maximum_steps < 1 or maximum_steps > 2:
        raise ValueError("Tiny training smoke is capped at two optimizer steps")
    labels = np.asarray(batch.get("labels"), dtype=float)
    ids = np.asarray(batch.get("smiles_input_ids"), dtype=np.int64)
    mask = np.asarray(batch.get("smiles_attention_mask"), dtype=np.float32)
    finite = np.isfinite(labels)
    if ids.ndim != 2 or mask.shape != ids.shape or labels.shape != (ids.shape[0],):
        raise ValueError("Batch does not satisfy the tiny smoke interface")
    if ids.shape[0] > 32 or ids.size > 8192:
        raise ValueError("Tiny training smoke tensor cap exceeded")
    if not np.any(finite):
        raise ValueError("Tiny training smoke needs at least one finite label")
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency profile.
        return {
            "status": "skipped_optional_torch_unavailable",
            "smoke_only": True,
            "performance_evidence": False,
            "reason": str(exc),
        }
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    vocabulary_size = int(ids.max()) + 1
    embedding_width = 8
    model = torch.nn.Sequential(torch.nn.Embedding(vocabulary_size, embedding_width))
    head = torch.nn.Linear(embedding_width, 1)
    parameter_count = sum(item.numel() for item in model.parameters()) + sum(
        item.numel() for item in head.parameters()
    )
    if parameter_count >= 100_000:
        raise RuntimeError("Tiny smoke parameter guard exceeded")
    optimizer = torch.optim.SGD([*model.parameters(), *head.parameters()], lr=1e-3)
    tensor_ids = torch.as_tensor(ids, dtype=torch.long)
    tensor_mask = torch.as_tensor(mask, dtype=torch.float32)
    tensor_labels = torch.as_tensor(labels[finite], dtype=torch.float32)
    finite_tensor = torch.as_tensor(finite, dtype=torch.bool)
    losses: list[float] = []
    for _ in range(maximum_steps):
        optimizer.zero_grad(set_to_none=True)
        embedded = model(tensor_ids)
        pooled = (embedded * tensor_mask.unsqueeze(-1)).sum(dim=1) / tensor_mask.sum(dim=1).clamp_min(
            1
        ).unsqueeze(-1)
        prediction = head(pooled).squeeze(-1)[finite_tensor]
        loss = torch.nn.functional.mse_loss(prediction, tensor_labels)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    return {
        "status": "passed",
        "smoke_only": True,
        "performance_evidence": False,
        "steps": maximum_steps,
        "examples": int(np.sum(finite)),
        "parameter_count": parameter_count,
        "losses_for_numerical_wiring_only": losses,
        "warning": "loss values are not model results and must not enter scientific comparisons",
    }


def environment_snapshot() -> dict[str, Any]:
    """Minimal runtime identity for serialization/smoke manifests."""

    versions = {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "executable_name": Path(sys.executable).name,
    }
    try:
        import rdkit

        versions["rdkit"] = rdkit.__version__
    except ImportError:
        versions["rdkit"] = "not_installed"
    try:
        import torch

        versions["torch"] = torch.__version__
        versions["torch_cuda_available"] = bool(torch.cuda.is_available())
        versions["torch_mps_available"] = bool(
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        )
    except ImportError:
        versions["torch"] = "not_installed"
    versions["feature_registry_sha256"] = FeatureRegistry().digest()
    return versions


def materialize_static_readiness_registries(
    *,
    feature_directory: Path,
    model_directory: Path,
    evidence_checked_date: str,
) -> dict[str, Any]:
    """Write deterministic feature/model registries without launching a model.

    ``evidence_checked_date`` is supplied by the caller so the artifact records
    when mutable upstream model cards were reviewed.  An exact checkpoint,
    immutable license snapshot, and overlap audit remain explicit preflight
    requirements; this registry is not training authorization.
    """

    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", evidence_checked_date):
        raise ValueError("evidence_checked_date must use YYYY-MM-DD")
    feature_directory.mkdir(parents=True, exist_ok=True)
    model_directory.mkdir(parents=True, exist_ok=True)

    feature_registry = FeatureRegistry()
    feature_frame = feature_registry.frame()
    feature_csv = feature_directory / "feature_registry.csv"
    _atomic_write_text(feature_csv, feature_frame.to_csv(index=False, lineterminator="\n"))
    feature_metadata = {
        "schema_version": FEATURE_REGISTRY_VERSION,
        "feature_registry_sha256": feature_registry.digest(),
        "file": feature_csv.name,
        "file_sha256": file_sha256(feature_csv),
        "n_feature_rules": int(len(feature_frame)),
        "default_model_inputs": sorted(
            feature_frame.loc[feature_frame["default_model_input"], "name"].astype(str).tolist()
        ),
        "free_text_default_enabled": False,
        "unregistered_feature_policy": "fail_closed",
    }
    feature_metadata_path = feature_directory / "feature_registry_metadata.json"
    _atomic_write_text(
        feature_metadata_path,
        json.dumps(feature_metadata, indent=2, sort_keys=True) + "\n",
    )

    candidate_payload = [asdict(candidate) for candidate in candidate_model_registry()]
    candidate_json_path = model_directory / "model_candidate_registry.json"
    candidate_json = {
        "schema_version": "model_candidate_registry_v1",
        "evidence_checked_date": evidence_checked_date,
        "training_authorized": False,
        "candidate_count": len(candidate_payload),
        "candidates": candidate_payload,
        "preflight_policy": (
            "exact checkpoint revision/hash, license snapshot/review, input compatibility, and training-overlap "
            "audit are required before use"
        ),
    }
    _atomic_write_text(candidate_json_path, json.dumps(candidate_json, indent=2, sort_keys=True) + "\n")
    candidate_rows: list[dict[str, Any]] = []
    for candidate in candidate_payload:
        candidate_rows.append(
            {
                key: _canonical_json(value) if isinstance(value, (dict, list, tuple)) else value
                for key, value in candidate.items()
            }
        )
    candidate_csv_path = model_directory / "model_candidate_registry.csv"
    _atomic_write_text(
        candidate_csv_path,
        pd.DataFrame(candidate_rows).to_csv(index=False, lineterminator="\n"),
    )
    metric_registry_path = model_directory / "model_metric_registry.csv"
    _atomic_write_text(
        metric_registry_path,
        metric_reporting_registry().to_csv(index=False, lineterminator="\n"),
    )
    robustness_matrix_path = model_directory / "baseline_robustness_matrix.csv"
    _atomic_write_text(
        robustness_matrix_path,
        robustness_configuration_matrix().to_csv(index=False, lineterminator="\n"),
    )

    def manifest_relative_path(path: Path) -> str:
        return Path(os.path.relpath(path, start=model_directory)).as_posix()

    artifacts = {
        "feature_registry_csv": {
            "path": manifest_relative_path(feature_csv),
            "sha256": file_sha256(feature_csv),
        },
        "feature_registry_metadata": {
            "path": manifest_relative_path(feature_metadata_path),
            "sha256": file_sha256(feature_metadata_path),
        },
        "model_candidate_registry_json": {
            "path": manifest_relative_path(candidate_json_path),
            "sha256": file_sha256(candidate_json_path),
        },
        "model_candidate_registry_csv": {
            "path": manifest_relative_path(candidate_csv_path),
            "sha256": file_sha256(candidate_csv_path),
        },
        "model_metric_registry_csv": {
            "path": manifest_relative_path(metric_registry_path),
            "sha256": file_sha256(metric_registry_path),
        },
        "baseline_robustness_matrix_csv": {
            "path": manifest_relative_path(robustness_matrix_path),
            "sha256": file_sha256(robustness_matrix_path),
        },
    }
    static_manifest = {
        "schema_version": "static_pretraining_readiness_manifest_v1",
        "evidence_checked_date": evidence_checked_date,
        "artifacts": artifacts,
        "environment": environment_snapshot(),
        "substantive_training_started": False,
    }
    manifest_path = model_directory / "pretraining_static_manifest.json"
    _atomic_write_text(manifest_path, json.dumps(static_manifest, indent=2, sort_keys=True) + "\n")
    return {
        **static_manifest,
        "manifest": {"path": manifest_path.name, "sha256": file_sha256(manifest_path)},
    }
