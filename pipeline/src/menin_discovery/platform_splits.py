"""Fixed intended-use splits and leakage audits for platform task views.

Official partitions are materialized as record-level manifests.  They are never
regenerated implicitly by a training loop.  Split strategies answer different
questions, so this module does not designate one universally "best" split.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans

from .features import (
    fingerprint_matrix,
    is_exact_smiles_proxy_method,
    nearest_neighbor_tanimoto,
    scaffold_key,
)
from .platform_data_schema import OBSERVATION_KINDS
from .platform_features import normalize_protein_sequence, stable_json_digest

SPLIT_SCHEMA_VERSION = "1.0.0"
PARTITIONS = ("train", "validation", "test")
SplitStrategy = Literal[
    "molecule_grouped",
    "scaffold",
    "chemical_cluster",
    "temporal",
    "source_holdout",
    "protein_holdout",
    "target_holdout",
    "double_cold",
]


@dataclass(frozen=True)
class SplitConfig:
    """Serializable fixed-split specification."""

    name: str
    strategy: SplitStrategy
    intended_use: str
    seed: int = 20260804
    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    record_id_column: str = "observation_id"
    molecule_id_column: str = "molecule_id"
    smiles_column: str = "standardized_smiles"
    protein_id_column: str = "protein_id"
    target_id_column: str = "canonical_target_id"
    source_id_column: str = "source_id"
    date_column: str = "document_year"
    label_column: str = "label_value"
    task_type: str = "regression"
    chemical_cluster_bits: int = 1024
    chemical_cluster_count: int | None = None
    near_duplicate_tanimoto: float = 0.80
    sequence_kmer_size: int = 3
    near_duplicate_sequence_jaccard: float = 0.80
    allow_derived_labels: bool = False
    derived_label_lineage_column: str = "label_lineage_digest"
    near_duplicate_max_pair_comparisons: int = 20_000_000
    near_duplicate_max_dense_bytes: int = 512 * 1024 * 1024
    near_duplicate_max_protein_pair_comparisons: int = 5_000_000

    def validate(self) -> None:
        fractions = (self.train_fraction, self.validation_fraction, self.test_fraction)
        if any(not 0 < value < 1 for value in fractions):
            raise ValueError("All train/validation/test fractions must be between zero and one")
        if not math.isclose(sum(fractions), 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("Split fractions must sum to one")
        if self.task_type not in {"regression", "classification"}:
            raise ValueError("task_type must be regression or classification")
        if not 0 < self.near_duplicate_tanimoto <= 1:
            raise ValueError("near_duplicate_tanimoto must be in (0, 1]")
        if (
            min(
                self.near_duplicate_max_pair_comparisons,
                self.near_duplicate_max_dense_bytes,
                self.near_duplicate_max_protein_pair_comparisons,
            )
            < 1
        ):
            raise ValueError("Near-duplicate work and memory guards must be positive")


def _normalized_cell(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _stable_hash(value: object, seed: int) -> str:
    return hashlib.sha256(f"{seed}|{_normalized_cell(value)}".encode()).hexdigest()


def _resolve_column(frame: pd.DataFrame, preferred: str, fallbacks: Sequence[str]) -> str:
    for column in (preferred, *fallbacks):
        if column in frame.columns:
            return column
    raise ValueError(f"None of the required columns is available: {(preferred, *fallbacks)}")


def _stable_record_ids(frame: pd.DataFrame, requested: str) -> tuple[pd.Series, str]:
    candidates = (requested, "record_id", "measurement_id", "source_record_id")
    column = next((item for item in candidates if item in frame.columns), None)
    if column is None:
        raise ValueError(f"A stable record ID is required; tried {candidates}")
    ids = frame[column].map(_normalized_cell)
    if ids.eq("").any():
        raise ValueError(f"Stable record ID column {column!r} contains missing/blank values")
    if ids.duplicated().any():
        examples = ids[ids.duplicated(keep=False)].unique()[:5].tolist()
        raise ValueError(f"Stable record IDs must be unique; duplicate examples={examples}")
    return ids, column


def _stable_group_values(frame: pd.DataFrame, column: str, *, prefix: str) -> np.ndarray:
    values = frame[column].map(_normalized_cell).to_numpy(dtype=object)
    if any(not value for value in values):
        raise ValueError(
            f"Grouping column {column!r} has missing values; missing cannot be treated as one entity"
        )
    return np.asarray([f"{prefix}:{value}" for value in values], dtype=object)


def _label_categories(frame: pd.DataFrame, config: SplitConfig) -> np.ndarray | None:
    if config.task_type != "classification" or config.label_column not in frame.columns:
        return None
    labels = frame[config.label_column]
    if labels.isna().any() or labels.nunique(dropna=True) < 2:
        return None
    return labels.astype(str).to_numpy(dtype=object)


def _greedy_group_partition(
    groups: np.ndarray,
    *,
    fractions: Mapping[str, float],
    seed: int,
    labels: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Assign whole groups with deterministic size and optional class balancing."""

    if len(groups) < 3:
        raise ValueError("At least three records are required for a three-way split")
    group_rows: dict[str, list[int]] = defaultdict(list)
    for index, group in enumerate(groups):
        group_rows[str(group)].append(index)
    if len(group_rows) < 3:
        raise ValueError("At least three unique groups are required for a three-way split")
    partitions = tuple(fractions)
    targets = {name: float(fractions[name]) * len(groups) for name in partitions}
    assigned_counts = {name: 0 for name in partitions}
    unique_labels = sorted(set(labels.tolist())) if labels is not None else []
    global_label_counts = (
        {label: int(np.sum(labels == label)) for label in unique_labels} if labels is not None else {}
    )
    assigned_labels = {name: {label: 0 for label in unique_labels} for name in partitions}
    assignments = np.full(len(groups), "unassigned", dtype=object)
    ordered_groups = sorted(
        group_rows, key=lambda group: (-len(group_rows[group]), _stable_hash(group, seed))
    )

    for position, group in enumerate(ordered_groups):
        indices = np.asarray(group_rows[group], dtype=int)
        remaining_groups = len(ordered_groups) - position
        empty_partitions = [name for name in partitions if assigned_counts[name] == 0]
        candidates = empty_partitions if remaining_groups <= len(empty_partitions) else list(partitions)
        best: tuple[float, str] | None = None
        for name in candidates:
            count_after = assigned_counts[name] + len(indices)
            size_score = abs(count_after - targets[name]) / max(targets[name], 1.0)
            overflow = max(0.0, count_after - targets[name]) / max(targets[name], 1.0)
            score = size_score + 0.35 * overflow
            if labels is not None:
                for label in unique_labels:
                    target_label_count = fractions[name] * global_label_counts[label]
                    after = assigned_labels[name][label] + int(np.sum(labels[indices] == label))
                    score += 0.20 * abs(after - target_label_count) / max(target_label_count, 1.0)
            score += int(_stable_hash(f"{group}|{name}", seed)[:8], 16) / 16**8 * 1e-8
            candidate = (float(score), name)
            if best is None or candidate < best:
                best = candidate
        assert best is not None
        chosen = best[1]
        assignments[indices] = chosen
        assigned_counts[chosen] += len(indices)
        if labels is not None:
            for label in unique_labels:
                assigned_labels[chosen][label] += int(np.sum(labels[indices] == label))

    if set(assignments) != set(partitions):
        raise RuntimeError(f"Failed to create all partitions; observed={sorted(set(assignments))}")
    group_partition_counts = {name: int(len(set(groups[assignments == name]))) for name in partitions}
    return assignments, {
        "algorithm": "deterministic_greedy_whole_group_v1",
        "n_groups": int(len(group_rows)),
        "row_counts": {name: int(np.sum(assignments == name)) for name in partitions},
        "group_counts": group_partition_counts,
        "class_balance_used": bool(labels is not None),
    }


def _scaffold_groups(frame: pd.DataFrame, smiles_column: str) -> tuple[np.ndarray, dict[str, Any]]:
    keys: list[str] = []
    methods: list[str] = []
    for value in frame[smiles_column]:
        key, method = scaffold_key(value)
        keys.append(f"scaffold:{key}")
        methods.append(method)
    fallback_count = sum(is_exact_smiles_proxy_method(method) for method in methods)
    return np.asarray(keys, dtype=object), {
        "group_definition": "bemis_murcko_with_exact_acyclic",
        "method_counts": dict(sorted(pd.Series(methods).value_counts().astype(int).to_dict().items())),
        "fallback_count": int(fallback_count),
        "is_true_scaffold_split": fallback_count == 0,
    }


def _chemical_cluster_groups(
    frame: pd.DataFrame, config: SplitConfig, smiles_column: str
) -> tuple[np.ndarray, dict[str, Any]]:
    """Deterministic fingerprint-space clusters; similarity is audited separately."""

    matrix, backend = fingerprint_matrix(
        frame[smiles_column].fillna("").astype(str),
        backend="rdkit",
        n_bits=config.chemical_cluster_bits,
    )
    n_rows = len(frame)
    requested = config.chemical_cluster_count
    n_clusters = requested or min(max(3, int(round(math.sqrt(n_rows)))), min(512, n_rows - 1))
    n_clusters = min(max(3, int(n_clusters)), n_rows - 1)
    model = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=config.seed,
        n_init=10,
        batch_size=min(2048, max(64, n_rows)),
        reassignment_ratio=0.0,
    )
    cluster = model.fit_predict(matrix)
    return np.asarray([f"chemical_cluster:{item}" for item in cluster], dtype=object), {
        "group_definition": "minibatch_kmeans_morgan_centroid",
        "fingerprint_backend": backend,
        "fingerprint_bits": int(config.chemical_cluster_bits),
        "n_clusters": int(n_clusters),
        "near_duplicate_guarantee": False,
        "near_duplicate_policy": "quantify_with_cross_partition_tanimoto_audit",
    }


def _extract_year(value: object) -> float:
    text = _normalized_cell(value)
    matches = re_year.findall(text)
    if matches:
        return float(min(int(value) for value in matches))
    try:
        year = float(text)
    except ValueError:
        return np.nan
    return year if 1900 <= year <= 2100 else np.nan


re_year = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")


def _temporal_partition(
    frame: pd.DataFrame,
    config: SplitConfig,
    molecule_column: str,
    date_column: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    molecules = frame[molecule_column].map(_normalized_cell).to_numpy(dtype=object)
    years = frame[date_column].map(_extract_year).to_numpy(dtype=float)
    first_year: dict[str, float] = {}
    for molecule in sorted(set(molecules)):
        observed = years[molecules == molecule]
        observed = observed[np.isfinite(observed)]
        first_year[molecule] = float(np.min(observed)) if len(observed) else np.nan
    group_years = np.asarray([first_year[item] for item in molecules], dtype=float)
    known_unique_years = np.unique(group_years[np.isfinite(group_years)])
    if len(known_unique_years) < 3:
        raise ValueError("Temporal split needs at least three distinct known first-observation years")

    target_test = config.test_fraction * len(frame)
    test_candidates = [
        (abs(np.sum(group_years >= cutoff) - target_test), cutoff)
        for cutoff in known_unique_years[1:]
        if 0 < np.sum(group_years >= cutoff) < len(frame)
    ]
    test_cutoff = float(min(test_candidates, key=lambda item: (item[0], -item[1]))[1])
    earlier = group_years[group_years < test_cutoff]
    target_validation = config.validation_fraction * len(frame)
    validation_candidates = [
        (abs(np.sum((group_years >= cutoff) & (group_years < test_cutoff)) - target_validation), cutoff)
        for cutoff in np.unique(earlier)[1:]
        if 0 < np.sum((group_years >= cutoff) & (group_years < test_cutoff)) < len(earlier)
    ]
    if not validation_candidates:
        raise ValueError("Temporal split could not identify a non-empty validation period")
    validation_cutoff = float(min(validation_candidates, key=lambda item: (item[0], -item[1]))[1])
    assignments = np.full(len(frame), "excluded_unknown_date", dtype=object)
    assignments[np.isfinite(group_years) & (group_years < validation_cutoff)] = "train"
    assignments[np.isfinite(group_years) & (group_years >= validation_cutoff)] = "validation"
    assignments[np.isfinite(group_years) & (group_years >= test_cutoff)] = "test"
    groups = np.asarray([f"molecule:{item}" for item in molecules], dtype=object)
    return (
        assignments,
        groups,
        {
            "algorithm": "first_observation_year_grouped_v1",
            "validation_cutoff_year": int(validation_cutoff),
            "test_cutoff_year": int(test_cutoff),
            "unknown_date_rows_excluded": int(np.sum(~np.isfinite(group_years))),
            "unknown_date_policy": "fail_closed_excluded_unknown_date",
            "known_year_range": [int(known_unique_years.min()), int(known_unique_years.max())],
            "time_order_assertion": "test_first_year >= test_cutoff > validation_first_year >= validation_cutoff",
        },
    )


def _double_cold_partition(
    molecule_groups: np.ndarray,
    protein_groups: np.ndarray,
    *,
    config: SplitConfig,
    labels: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    fractions = dict(
        zip(
            PARTITIONS,
            (config.train_fraction, config.validation_fraction, config.test_fraction),
            strict=True,
        )
    )
    molecule_assignment, molecule_metadata = _greedy_group_partition(
        molecule_groups, fractions=fractions, seed=config.seed, labels=labels
    )
    protein_assignment, protein_metadata = _greedy_group_partition(
        protein_groups, fractions=fractions, seed=config.seed + 7919, labels=labels
    )
    assignment = np.where(molecule_assignment == protein_assignment, molecule_assignment, "excluded_mixed")
    if any(np.sum(assignment == name) == 0 for name in PARTITIONS):
        raise ValueError(
            "Double-cold support is insufficient: an aligned molecule/protein partition is empty"
        )
    combined = np.asarray(
        [f"{molecule}|{protein}" for molecule, protein in zip(molecule_groups, protein_groups, strict=True)],
        dtype=object,
    )
    return (
        assignment,
        combined,
        {
            "algorithm": "independent_entity_partition_intersection_v1",
            "molecule_partition": molecule_metadata,
            "protein_partition": protein_metadata,
            "excluded_mixed_rows": int(np.sum(assignment == "excluded_mixed")),
            "excluded_mixed_reason": "molecule and protein were assigned to different cold partitions",
        },
    )


def make_split_manifest(frame: pd.DataFrame, config: SplitConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build one immutable split manifest from a canonical task view."""

    config.validate()
    if len(frame) < 3:
        raise ValueError("At least three records are required")
    record_ids, resolved_record_column = _stable_record_ids(frame, config.record_id_column)
    observation_kind_column = next(
        (
            column
            for column in ("observation_kind", "outcome_kind", "outcome_origin", "assertion_type")
            if column in frame.columns
        ),
        None,
    )
    if observation_kind_column is None:
        raise ValueError("A canonical observation_kind is required for every split manifest")
    observation_kind_aliases = {
        "observed": "experimental_raw",
        "curated": "curated_assertion",
        "experimental_observation": "experimental_raw",
        "curated_label": "curated_assertion",
    }
    normalized_observation_kinds = (
        frame[observation_kind_column]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .map(lambda value: observation_kind_aliases.get(value, value))
    )
    invalid_kinds = sorted(set(normalized_observation_kinds) - set(OBSERVATION_KINDS))
    if invalid_kinds or normalized_observation_kinds.eq("").any():
        raise ValueError(
            f"Every task row requires a canonical nonblank observation_kind; invalid={invalid_kinds}"
        )
    if normalized_observation_kinds.eq("prediction").any():
        raise ValueError("Prediction rows are prohibited from split manifests used for model labels")
    derived_mask = normalized_observation_kinds.eq("derived")
    if derived_mask.any():
        if not config.allow_derived_labels:
            raise ValueError("Derived labels require allow_derived_labels=True and explicit lineage")
        if config.derived_label_lineage_column not in frame.columns:
            raise ValueError("Derived labels require a lineage-digest column")
        lineage_present = (
            frame.loc[derived_mask, config.derived_label_lineage_column]
            .fillna("")
            .astype(str)
            .str.strip()
            .ne("")
        )
        if not lineage_present.all():
            raise ValueError("Every derived label requires a non-empty lineage digest")
        invalid_lineage = [
            str(value)
            for value in frame.loc[derived_mask, config.derived_label_lineage_column]
            if not re.fullmatch(r"[0-9a-fA-F]{64}", str(value).strip())
        ]
        if invalid_lineage:
            raise ValueError("Every derived label lineage digest must be a SHA-256 hexadecimal value")
    molecule_column = _resolve_column(
        frame, config.molecule_id_column, ("structure_id", "standard_inchi_key")
    )
    labels = _label_categories(frame, config)
    fractions = dict(
        zip(
            PARTITIONS,
            (config.train_fraction, config.validation_fraction, config.test_fraction),
            strict=True,
        )
    )
    strategy_metadata: dict[str, Any]
    split_driving_columns = [molecule_column]

    if config.strategy == "molecule_grouped":
        groups = _stable_group_values(frame, molecule_column, prefix="molecule")
        assignments, strategy_metadata = _greedy_group_partition(
            groups, fractions=fractions, seed=config.seed, labels=labels
        )
    elif config.strategy == "scaffold":
        smiles_column = _resolve_column(
            frame, config.smiles_column, ("canonical_smiles", "submitted_smiles", "smiles")
        )
        groups, group_metadata = _scaffold_groups(frame, smiles_column)
        if int(group_metadata["fallback_count"]):
            proxy_methods = sorted(
                method
                for method, count in group_metadata["method_counts"].items()
                if int(count) and is_exact_smiles_proxy_method(method)
            )
            raise ValueError(
                f"A true scaffold split cannot admit exact-SMILES proxy groups; methods={proxy_methods}"
            )
        split_driving_columns.append(smiles_column)
        assignments, strategy_metadata = _greedy_group_partition(
            groups, fractions=fractions, seed=config.seed, labels=labels
        )
        strategy_metadata.update(group_metadata)
    elif config.strategy == "chemical_cluster":
        smiles_column = _resolve_column(
            frame, config.smiles_column, ("canonical_smiles", "submitted_smiles", "smiles")
        )
        groups, group_metadata = _chemical_cluster_groups(frame, config, smiles_column)
        split_driving_columns.append(smiles_column)
        assignments, strategy_metadata = _greedy_group_partition(
            groups, fractions=fractions, seed=config.seed, labels=labels
        )
        strategy_metadata.update(group_metadata)
    elif config.strategy == "temporal":
        date_column = _resolve_column(
            frame, config.date_column, ("document_date", "measurement_date", "year")
        )
        assignments, groups, strategy_metadata = _temporal_partition(
            frame, config, molecule_column, date_column
        )
        split_driving_columns.append(date_column)
    elif config.strategy == "source_holdout":
        source_column = _resolve_column(frame, config.source_id_column, ("source", "snapshot_id"))
        split_driving_columns.append(source_column)
        groups = _stable_group_values(frame, source_column, prefix="source")
        assignments, strategy_metadata = _greedy_group_partition(
            groups, fractions=fractions, seed=config.seed, labels=labels
        )
    elif config.strategy in {"protein_holdout", "target_holdout"}:
        requested = (
            config.protein_id_column if config.strategy == "protein_holdout" else config.target_id_column
        )
        fallbacks = (
            ("canonical_target_id", "target_id", "target_name")
            if config.strategy == "protein_holdout"
            else ("protein_id", "target_id", "target_name")
        )
        protein_column = _resolve_column(frame, requested, fallbacks)
        split_driving_columns.append(protein_column)
        groups = _stable_group_values(frame, protein_column, prefix=config.strategy)
        assignments, strategy_metadata = _greedy_group_partition(
            groups, fractions=fractions, seed=config.seed, labels=labels
        )
    elif config.strategy == "double_cold":
        protein_column = _resolve_column(
            frame,
            config.protein_id_column,
            ("canonical_target_id", "target_id", "target_name"),
        )
        split_driving_columns.append(protein_column)
        molecule_groups = _stable_group_values(frame, molecule_column, prefix="molecule")
        protein_groups = _stable_group_values(frame, protein_column, prefix="protein")
        assignments, groups, strategy_metadata = _double_cold_partition(
            molecule_groups, protein_groups, config=config, labels=labels
        )
    else:  # pragma: no cover - Literal and config validation constrain callers.
        raise ValueError(f"Unsupported strategy: {config.strategy}")

    manifest = pd.DataFrame(
        {
            "record_id": record_ids,
            "split": assignments,
            "group_id": groups,
            "strategy": config.strategy,
            "split_name": config.name,
            "seed": config.seed,
            "molecule_id": frame[molecule_column].map(_normalized_cell),
        }
    )
    for output, candidates in {
        "protein_id": (config.protein_id_column, "canonical_target_id", "target_id"),
        "task_id": ("task_id",),
        "source_id": (config.source_id_column, "source", "snapshot_id"),
        "document_year": (config.date_column, "year"),
        "observation_kind": (
            "observation_kind",
            "outcome_kind",
            "outcome_origin",
            "assertion_type",
        ),
    }.items():
        column = next((item for item in candidates if item in frame.columns), None)
        manifest[output] = frame[column].map(_normalized_cell) if column else ""
    manifest["observation_kind"] = (
        manifest["observation_kind"]
        .astype(str)
        .str.lower()
        .map(lambda value: observation_kind_aliases.get(value, value))
    )
    manifest = manifest.sort_values("record_id", kind="mergesort").reset_index(drop=True)
    dataset_binding_candidates = [
        resolved_record_column,
        *split_driving_columns,
        config.protein_id_column,
        "canonical_target_id",
        "sequence",
        "protein_sequence",
        "assay_id",
        config.source_id_column,
        "snapshot_id",
        "task_id",
        "task_type",
        "label_kind",
        config.label_column,
        "label_text",
        "label_relation",
        "label_lower_bound",
        "label_upper_bound",
        "label_unit",
        observation_kind_column,
        config.derived_label_lineage_column,
        "access_class",
        "quality_grade",
        "document_year",
    ]
    dataset_binding_columns = list(
        dict.fromkeys(column for column in dataset_binding_candidates if column in frame.columns)
    )
    binding = (
        frame[dataset_binding_columns]
        .fillna("")
        .astype(str)
        .sort_values(resolved_record_column, kind="mergesort")
    )
    dataset_sha256 = hashlib.sha256(
        binding.to_csv(index=False, lineterminator="\n").encode("utf-8")
    ).hexdigest()
    manifest_sha256 = hashlib.sha256(
        manifest.to_csv(index=False, lineterminator="\n").encode("utf-8")
    ).hexdigest()
    metadata = {
        "schema_version": SPLIT_SCHEMA_VERSION,
        "config": asdict(config),
        "resolved_record_id_column": resolved_record_column,
        "resolved_molecule_id_column": molecule_column,
        "n_records": int(len(frame)),
        "partition_counts": manifest["split"].value_counts().sort_index().astype(int).to_dict(),
        "dataset_binding_columns": dataset_binding_columns,
        "dataset_sha256": dataset_sha256,
        "manifest_sha256": manifest_sha256,
        "strategy_metadata": strategy_metadata,
        "official_split_policy": "materialize_manifest_and_load_by_record_id_only",
    }
    return manifest, metadata


def _cross_partition_overlap(manifest: pd.DataFrame, values: pd.Series) -> dict[str, Any]:
    normalized = values.map(_normalized_cell)
    table = pd.DataFrame({"value": normalized, "split": manifest["split"].to_numpy()})
    table = table[(table["value"] != "") & table["split"].isin(PARTITIONS)]
    counts = table.groupby("value")["split"].nunique()
    overlapping = counts[counts > 1].index.astype(str).tolist()
    return {
        "n_values": int(table["value"].nunique()),
        "n_cross_partition": int(len(overlapping)),
        "examples_sha256": [hashlib.sha256(item.encode("utf-8")).hexdigest() for item in overlapping[:20]],
    }


def _unique_value_representatives(
    indices: Sequence[int] | np.ndarray,
    values: pd.Series,
) -> tuple[np.ndarray, dict[str, list[int]]]:
    """Collapse repeated entity representations without losing record support."""

    members: dict[str, list[int]] = defaultdict(list)
    for raw_index in indices:
        index = int(raw_index)
        members[_normalized_cell(values.iloc[index])].append(index)
    representatives = np.asarray([rows[0] for _, rows in sorted(members.items())], dtype=int)
    return representatives, members


def _chemical_near_duplicate_audit(
    frame: pd.DataFrame,
    manifest: pd.DataFrame,
    *,
    smiles_column: str,
    threshold: float,
    n_bits: int,
    maximum_pair_comparisons: int,
    maximum_dense_bytes: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    pair_metadata: dict[str, Any] = {}
    smiles = frame[smiles_column].fillna("").astype(str).reset_index(drop=True)
    aligned = manifest.set_index("record_id")
    record_column = next(
        column
        for column in ("observation_id", "record_id", "measurement_id", "source_record_id")
        if column in frame.columns
    )
    splits = frame[record_column].astype(str).map(aligned["split"]).reset_index(drop=True)
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        left_indices = np.flatnonzero(splits.to_numpy() == left)
        right_indices = np.flatnonzero(splits.to_numpy() == right)
        key = f"{left}_vs_{right}"
        if not len(left_indices) or not len(right_indices):
            pair_metadata[key] = {"status": "not_computable_empty_partition"}
            continue
        left_unique, left_members = _unique_value_representatives(left_indices, smiles)
        right_unique, right_members = _unique_value_representatives(right_indices, smiles)
        left_fingerprints, backend = fingerprint_matrix(
            smiles.iloc[left_unique], backend="rdkit", n_bits=n_bits
        )
        right_fingerprints, _ = fingerprint_matrix(smiles.iloc[right_unique], backend=backend, n_bits=n_bits)
        left_valid_mask = np.asarray(left_fingerprints.sum(axis=1)).ravel() > 0
        right_valid_mask = np.asarray(right_fingerprints.sum(axis=1)).ravel() > 0
        left_valid = left_unique[left_valid_mask]
        right_valid = right_unique[right_valid_mask]
        invalid_left_records = sum(
            len(left_members[_normalized_cell(smiles.iloc[index])]) for index in left_unique[~left_valid_mask]
        )
        invalid_right_records = sum(
            len(right_members[_normalized_cell(smiles.iloc[index])])
            for index in right_unique[~right_valid_mask]
        )
        if not len(left_valid) or not len(right_valid):
            pair_metadata[key] = {
                "status": "not_computable_no_valid_fingerprints",
                "n_reference_records": int(len(left_indices)),
                "n_query_records": int(len(right_indices)),
                "n_reference_unique_structures": int(len(left_unique)),
                "n_query_unique_structures": int(len(right_unique)),
                "n_reference_invalid_fingerprint_records": int(invalid_left_records),
                "n_query_invalid_fingerprint_records": int(invalid_right_records),
            }
            continue
        pair_comparisons = int(len(left_valid) * len(right_valid))
        conservative_dense_bytes = int(min(256, len(right_valid)) * len(left_valid) * 8)
        if pair_comparisons > maximum_pair_comparisons or conservative_dense_bytes > maximum_dense_bytes:
            pair_metadata[key] = {
                "status": "requires_scalable_indexed_or_partitioned_audit",
                "n_query_records": int(len(right_indices)),
                "n_reference_records": int(len(left_indices)),
                "n_query_unique_valid_structures": int(len(right_valid)),
                "n_reference_unique_valid_structures": int(len(left_valid)),
                "pair_comparisons": pair_comparisons,
                "conservative_dense_chunk_bytes": conservative_dense_bytes,
                "maximum_pair_comparisons": maximum_pair_comparisons,
                "maximum_dense_bytes": maximum_dense_bytes,
                "completeness": "not_run_fail_closed_work_guard",
            }
            continue
        maxima, nearest, backend = nearest_neighbor_tanimoto(
            smiles.iloc[right_valid],
            smiles.iloc[left_valid],
            backend="rdkit",
            n_bits=n_bits,
        )
        hit = maxima >= threshold
        hit_record_count = sum(
            len(right_members[_normalized_cell(smiles.iloc[int(right_valid[index])])])
            for index in np.flatnonzero(hit)
        )
        invalid_records = invalid_left_records + invalid_right_records
        pair_metadata[key] = {
            "status": (
                "complete_exact_fingerprint_comparison"
                if not invalid_records
                else "incomplete_invalid_fingerprints"
            ),
            "fingerprint_backend": backend,
            "n_query_records": int(len(right_indices)),
            "n_reference_records": int(len(left_indices)),
            "n_query_unique_valid_structures": int(len(right_valid)),
            "n_reference_unique_valid_structures": int(len(left_valid)),
            "n_query_invalid_fingerprint_records": int(invalid_right_records),
            "n_reference_invalid_fingerprint_records": int(invalid_left_records),
            "n_query_unique_structures_at_or_above_threshold": int(np.sum(hit)),
            "n_query_records_at_or_above_threshold": int(hit_record_count),
            "fraction_query_records_at_or_above_threshold": float(hit_record_count / len(right_indices)),
            "maximum_similarity": float(np.max(maxima)),
            "pair_comparisons": pair_comparisons,
            "conservative_dense_chunk_bytes": conservative_dense_bytes,
            "completeness": (
                "exhaustive_unique_valid_structure_query_to_reference_maxima"
                if not invalid_records
                else "incomplete_due_to_invalid_or_empty_structure_fingerprints"
            ),
        }
        for local_index in np.flatnonzero(hit)[:100]:
            query_index = int(right_valid[local_index])
            reference_index = int(left_valid[int(nearest[local_index])])
            rows.append(
                {
                    "partition_pair": key,
                    "query_record_id": str(frame.iloc[query_index][record_column]),
                    "reference_record_id": str(frame.iloc[reference_index][record_column]),
                    "similarity": float(maxima[local_index]),
                    "threshold": float(threshold),
                    "audit_method": f"exact_max_tanimoto_morgan_{n_bits}",
                }
            )
    return {
        "threshold": float(threshold),
        "fingerprint": f"Morgan radius 2, {n_bits} bits",
        "pair_audits": pair_metadata,
        "example_cap_per_pair": 100,
        "platform_scale_policy": (
            "work/memory guards fail closed; build an indexed candidate generator plus exact verification before large-corpus claims"
        ),
    }, pd.DataFrame(rows)


def _kmer_set(sequence: object, size: int) -> frozenset[str]:
    normalized, invalid = normalize_protein_sequence(sequence)
    if invalid or len(normalized) < size:
        return frozenset()
    return frozenset(normalized[index : index + size] for index in range(len(normalized) - size + 1))


def _protein_near_duplicate_audit(
    frame: pd.DataFrame,
    manifest: pd.DataFrame,
    *,
    sequence_column: str,
    kmer_size: int,
    threshold: float,
    maximum_pair_comparisons: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    record_column = next(
        column
        for column in ("observation_id", "record_id", "measurement_id", "source_record_id")
        if column in frame.columns
    )
    aligned = manifest.set_index("record_id")
    splits = frame[record_column].astype(str).map(aligned["split"]).to_numpy(dtype=object)
    sequences = frame[sequence_column].map(_normalized_cell).reset_index(drop=True)
    kmers = [_kmer_set(value, kmer_size) for value in sequences]
    rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        left_records = [int(item) for item in np.flatnonzero(splits == left)]
        right_records = [int(item) for item in np.flatnonzero(splits == right)]
        left_unique, left_members = _unique_value_representatives(left_records, sequences)
        right_unique, right_members = _unique_value_representatives(right_records, sequences)
        left_indices = [int(item) for item in left_unique if kmers[int(item)]]
        right_indices = [int(item) for item in right_unique if kmers[int(item)]]
        invalid_left_records = sum(
            len(left_members[_normalized_cell(sequences.iloc[int(index)])])
            for index in left_unique
            if not kmers[int(index)]
        )
        invalid_right_records = sum(
            len(right_members[_normalized_cell(sequences.iloc[int(index)])])
            for index in right_unique
            if not kmers[int(index)]
        )
        key = f"{left}_vs_{right}"
        pair_comparisons = int(len(left_indices) * len(right_indices))
        if pair_comparisons > maximum_pair_comparisons:
            summaries[key] = {
                "status": "requires_scalable_indexed_or_partitioned_audit",
                "n_query_records": len(right_records),
                "n_reference_records": len(left_records),
                "n_query_unique_with_sequence": len(right_indices),
                "n_reference_unique_with_sequence": len(left_indices),
                "pair_comparisons": pair_comparisons,
                "maximum_pair_comparisons": maximum_pair_comparisons,
                "completeness": "not_run_fail_closed_work_guard",
            }
            continue
        hits = 0
        hit_records = 0
        maximum = 0.0
        for query_index in right_indices:
            best_similarity = 0.0
            best_reference = -1
            for reference_index in left_indices:
                union = kmers[query_index] | kmers[reference_index]
                similarity = len(kmers[query_index] & kmers[reference_index]) / len(union) if union else 0.0
                if similarity > best_similarity:
                    best_similarity, best_reference = similarity, reference_index
            maximum = max(maximum, best_similarity)
            if best_similarity >= threshold:
                hits += 1
                hit_records += len(right_members[_normalized_cell(sequences.iloc[query_index])])
                if sum(item["partition_pair"] == key for item in rows) < 100:
                    rows.append(
                        {
                            "partition_pair": key,
                            "query_record_id": str(frame.iloc[query_index][record_column]),
                            "reference_record_id": str(frame.iloc[best_reference][record_column]),
                            "similarity": float(best_similarity),
                            "threshold": float(threshold),
                            "audit_method": f"protein_{kmer_size}mer_jaccard_screen",
                        }
                    )
        summaries[key] = {
            "status": (
                "complete_kmer_screen"
                if left_indices and right_indices and not (invalid_left_records + invalid_right_records)
                else "incomplete_missing_or_invalid_sequence"
                if left_indices and right_indices
                else "not_computable_missing_sequence"
            ),
            "n_query_records": len(right_records),
            "n_reference_records": len(left_records),
            "n_query_unique_with_sequence": len(right_indices),
            "n_reference_unique_with_sequence": len(left_indices),
            "n_query_missing_or_invalid_sequence_records": invalid_right_records,
            "n_reference_missing_or_invalid_sequence_records": invalid_left_records,
            "n_query_unique_at_or_above_threshold": hits,
            "n_query_records_at_or_above_threshold": hit_records,
            "maximum_similarity": maximum if left_indices and right_indices else None,
            "pair_comparisons": pair_comparisons,
            "completeness": "exhaustive_kmer_set_comparison"
            if left_indices and right_indices
            else "unavailable",
        }
    return {
        "threshold": threshold,
        "kmer_size": kmer_size,
        "interpretation": "screening similarity, not aligned percent sequence identity",
        "pair_audits": summaries,
        "example_cap_per_pair": 100,
        "platform_scale_policy": (
            "quadratic screen is guarded; use indexed sequence candidates and exact alignment verification at scale"
        ),
    }, pd.DataFrame(rows)


def audit_split_leakage(
    frame: pd.DataFrame,
    manifest: pd.DataFrame,
    config: SplitConfig,
    *,
    run_chemical_near_duplicate_audit: bool = True,
    run_protein_near_duplicate_audit: bool = True,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Audit exact identities exhaustively and near identities with declared methods."""

    required_manifest = {"record_id", "split", "group_id", "strategy"}
    missing = sorted(required_manifest - set(manifest.columns))
    if missing:
        raise ValueError(f"Split manifest is missing columns: {missing}")
    record_ids, record_column = _stable_record_ids(frame, config.record_id_column)
    if set(record_ids) != set(manifest["record_id"].astype(str)):
        raise ValueError("Manifest record IDs do not exactly match the dataset")
    if len(manifest) != len(frame):
        raise ValueError("Manifest must contain exactly one row per dataset record")
    aligned_manifest = manifest.set_index("record_id").loc[record_ids].reset_index()
    exact: dict[str, Any] = {}
    column_families = {
        "record": (record_column,),
        "molecule": (config.molecule_id_column, "structure_id", "standard_inchi_key"),
        "canonical_structure": (config.smiles_column, "canonical_smiles", "smiles"),
        "protein": (config.protein_id_column, "canonical_target_id", "target_id"),
        "assay": ("assay_id",),
        "source": (config.source_id_column, "source"),
        "dedup_group": ("dedup_group_id",),
    }
    for family, candidates in column_families.items():
        column = next((item for item in candidates if item in frame.columns), None)
        if column is None:
            exact[family] = {"status": "column_unavailable"}
        else:
            exact[family] = {
                "status": "complete",
                "column": column,
                **_cross_partition_overlap(aligned_manifest, frame[column].reset_index(drop=True)),
            }
    exact["manifest_group"] = {
        "status": "complete",
        **_cross_partition_overlap(aligned_manifest, aligned_manifest["group_id"]),
    }

    outcome_column = next(
        (
            column
            for column in ("observation_kind", "outcome_kind", "outcome_origin", "assertion_type")
            if column in frame.columns
        ),
        None,
    )
    if outcome_column:
        normalization = {
            "observed": "experimental_raw",
            "curated": "curated_assertion",
            "experimental_observation": "experimental_raw",
            "curated_label": "curated_assertion",
        }
        normalized_series = (
            frame[outcome_column]
            .fillna("")
            .astype(str)
            .str.lower()
            .map(lambda value: normalization.get(value, value))
        )
        normalized_origins = set(normalized_series) - {""}
        allowed = {"experimental_raw", "experimental_summary", "curated_assertion"}
        if config.allow_derived_labels:
            allowed.add("derived")
        prohibited_origins = sorted(normalized_origins - allowed)
        derived_mask = normalized_series.eq("derived")
        if derived_mask.any() and config.allow_derived_labels:
            if config.derived_label_lineage_column not in frame.columns:
                prohibited_origins.append("derived_missing_lineage_column")
            else:
                lineage_present = (
                    frame.loc[derived_mask, config.derived_label_lineage_column]
                    .fillna("")
                    .astype(str)
                    .str.strip()
                    .ne("")
                )
                if not lineage_present.all():
                    prohibited_origins.append("derived_missing_lineage_digest")
                invalid_lineage = [
                    str(value)
                    for value in frame.loc[derived_mask, config.derived_label_lineage_column]
                    if not re.fullmatch(r"[0-9a-fA-F]{64}", str(value).strip())
                ]
                if invalid_lineage:
                    prohibited_origins.append("derived_invalid_lineage_digest")
    else:
        prohibited_origins = ["missing_observation_kind_column"]

    near_frames: list[pd.DataFrame] = []
    near: dict[str, Any] = {}
    smiles_column = next(
        (
            column
            for column in (config.smiles_column, "canonical_smiles", "submitted_smiles", "smiles")
            if column in frame.columns
        ),
        None,
    )
    if run_chemical_near_duplicate_audit and smiles_column:
        chemical, examples = _chemical_near_duplicate_audit(
            frame.reset_index(drop=True),
            manifest,
            smiles_column=smiles_column,
            threshold=config.near_duplicate_tanimoto,
            n_bits=config.chemical_cluster_bits,
            maximum_pair_comparisons=config.near_duplicate_max_pair_comparisons,
            maximum_dense_bytes=config.near_duplicate_max_dense_bytes,
        )
        near["chemical"] = chemical
        if not examples.empty:
            examples.insert(0, "modality", "chemical")
            near_frames.append(examples)
    else:
        near["chemical"] = {
            "status": "not_run" if smiles_column else "column_unavailable",
            "reason": "disabled_by_caller" if smiles_column else "no_smiles_column",
        }
    sequence_column = next(
        (column for column in ("sequence", "protein_sequence") if column in frame.columns), None
    )
    if run_protein_near_duplicate_audit and sequence_column:
        protein, examples = _protein_near_duplicate_audit(
            frame.reset_index(drop=True),
            manifest,
            sequence_column=sequence_column,
            kmer_size=config.sequence_kmer_size,
            threshold=config.near_duplicate_sequence_jaccard,
            maximum_pair_comparisons=config.near_duplicate_max_protein_pair_comparisons,
        )
        near["protein"] = protein
        if not examples.empty:
            examples.insert(0, "modality", "protein")
            near_frames.append(examples)
    else:
        near["protein"] = {
            "status": "not_run" if sequence_column else "column_unavailable",
            "reason": "disabled_by_caller" if sequence_column else "no_sequence_column",
        }

    exclusive_families = {
        "molecule_grouped": ("molecule",),
        "scaffold": ("molecule", "manifest_group"),
        "chemical_cluster": ("molecule", "manifest_group"),
        "temporal": ("molecule",),
        "source_holdout": ("source",),
        "protein_holdout": ("protein",),
        "target_holdout": ("protein",),
        "double_cold": ("molecule", "protein"),
    }[config.strategy]
    failures: list[str] = []
    for family in exclusive_families:
        result = exact.get(family, {})
        if result.get("status") != "complete":
            failures.append(f"required_exact_audit_unavailable:{family}")
        elif result.get("n_cross_partition", 0):
            failures.append(f"exact_cross_partition_overlap:{family}")
    if prohibited_origins:
        failures.append("prohibited_unknown_or_unlineaged_observation_kind_used_as_label")
    examples_frame = (
        pd.concat(near_frames, ignore_index=True)
        if near_frames
        else pd.DataFrame(
            columns=[
                "modality",
                "partition_pair",
                "query_record_id",
                "reference_record_id",
                "similarity",
                "threshold",
                "audit_method",
            ]
        )
    )
    requested_modalities = {
        "chemical": bool(run_chemical_near_duplicate_audit),
        "protein": bool(run_protein_near_duplicate_audit),
    }
    modality_completeness: dict[str, dict[str, Any]] = {}
    for modality, requested in requested_modalities.items():
        details = near[modality]
        if not requested:
            modality_completeness[modality] = {
                "requested": False,
                "complete": False,
                "reason": "not_requested_by_caller",
            }
            continue
        if details.get("status") in {"column_unavailable", "not_run"}:
            modality_completeness[modality] = {
                "requested": True,
                "complete": False,
                "reason": str(details.get("reason", details.get("status"))),
            }
            continue
        pair_audits = details.get("pair_audits", {})
        expected_pairs = {"train_vs_validation", "train_vs_test", "validation_vs_test"}
        incomplete = {
            pair: str(pair_audits.get(pair, {}).get("status", "missing_pair_audit"))
            for pair in expected_pairs
            if not str(pair_audits.get(pair, {}).get("status", "")).startswith("complete_")
        }
        modality_completeness[modality] = {
            "requested": True,
            "complete": not incomplete and set(pair_audits) >= expected_pairs,
            "incomplete_partition_pairs": incomplete,
        }
    requested_results = [value["complete"] for value in modality_completeness.values() if value["requested"]]
    audit = {
        "schema_version": SPLIT_SCHEMA_VERSION,
        "split_name": config.name,
        "strategy": config.strategy,
        "intended_use": config.intended_use,
        "n_records": int(len(frame)),
        "partition_counts": aligned_manifest["split"].value_counts().sort_index().astype(int).to_dict(),
        "exact_overlap": exact,
        "near_duplicate": near,
        "near_duplicate_modality_completeness": modality_completeness,
        "near_duplicate_audit_complete": bool(requested_results) and all(requested_results),
        "label_origin_column": outcome_column,
        "prohibited_label_origins": prohibited_origins,
        "required_exclusive_families": exclusive_families,
        "failures": failures,
        "exact_leakage_gate_passed": not failures,
        "near_duplicate_gate_policy": (
            "reported_separately; threshold acceptance depends on the declared intended-use claim"
        ),
        "audit_sha256": "",
    }
    audit["audit_sha256"] = stable_json_digest({**audit, "audit_sha256": ""})
    return audit, examples_frame


def default_split_suite(*, task_type: str, seed: int = 20260804) -> tuple[SplitConfig, ...]:
    """Predeclare the platform's distinct generalization questions."""

    return (
        SplitConfig(
            name="molecule_grouped_v1",
            strategy="molecule_grouped",
            intended_use="new observation for a molecule never present in training",
            task_type=task_type,
            seed=seed,
        ),
        SplitConfig(
            name="scaffold_v1",
            strategy="scaffold",
            intended_use="new chemical scaffold for targets represented in the task view",
            task_type=task_type,
            seed=seed,
        ),
        SplitConfig(
            name="chemical_cluster_v1",
            strategy="chemical_cluster",
            intended_use="new fingerprint-space chemical cluster; near neighbors audited explicitly",
            task_type=task_type,
            seed=seed,
        ),
        SplitConfig(
            name="temporal_v1",
            strategy="temporal",
            intended_use="molecules first observed after frozen calendar cutoffs",
            task_type=task_type,
            seed=seed,
        ),
        SplitConfig(
            name="source_holdout_v1",
            strategy="source_holdout",
            intended_use="evidence from a source absent from training",
            task_type=task_type,
            seed=seed,
        ),
        SplitConfig(
            name="protein_holdout_v1",
            strategy="protein_holdout",
            intended_use="protein or construct absent from training, when at least three are supported",
            task_type=task_type,
            seed=seed,
        ),
        SplitConfig(
            name="target_holdout_v1",
            strategy="target_holdout",
            intended_use="canonical target identity absent from training, when at least three are supported",
            task_type=task_type,
            seed=seed,
        ),
        SplitConfig(
            name="double_cold_v1",
            strategy="double_cold",
            intended_use="both molecule and protein absent from training; mixed pairs excluded",
            task_type=task_type,
            seed=seed,
        ),
    )


STREAMING_SPLIT_SCHEMA_VERSION = "stream_hash_group_split_v1"
MANIFEST_BOUND_PARQUET_DATASET_SCHEMA_VERSION = "manifest_bound_parquet_dataset_v1"


@dataclass(frozen=True)
class ParquetDatasetPart:
    """One hash- and row-count-bound member of a resolved task dataset."""

    path: Path
    relative_path: str
    sha256: str
    rows: int


@dataclass(frozen=True)
class ResolvedParquetDataset:
    """Deterministic single-file or manifest-bound partitioned Parquet input."""

    input_path: Path
    input_kind: Literal["single_file", "manifest_bound_directory"]
    parts: tuple[ParquetDatasetPart, ...]
    dataset_sha256: str
    total_rows: int
    manifest_path: Path | None = None
    manifest_sha256: str | None = None

    def binding_payload(self) -> dict[str, Any]:
        return {
            "schema_version": MANIFEST_BOUND_PARQUET_DATASET_SCHEMA_VERSION,
            "input_kind": self.input_kind,
            "dataset_sha256": self.dataset_sha256,
            "total_rows": self.total_rows,
            "manifest_sha256": self.manifest_sha256,
            "parts": [
                {
                    "relative_path": part.relative_path,
                    "sha256": part.sha256,
                    "rows": part.rows,
                }
                for part in self.parts
            ],
        }


def _path_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_hex_sha256(value: object, field_name: str) -> str:
    normalized = str(value).strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", normalized):
        raise ValueError(f"{field_name} must be a SHA-256 hexadecimal digest")
    return normalized


def _nearest_bulk_build_manifest(directory: Path) -> Path:
    for candidate_root in (directory, *directory.parents):
        candidate = candidate_root / "build_manifest.json"
        if candidate.is_file():
            return candidate
    raise ValueError("Partitioned Parquet task directories require an ancestor build_manifest.json")


def resolve_manifest_bound_parquet_dataset(input_path: Path) -> ResolvedParquetDataset:
    """Resolve a single file or an exactly manifest-bound partitioned task dataset.

    Directory inputs are fail closed: all top-level Parquet files must be listed
    in the nearest ancestor build manifest, and every listed digest and row
    count is verified before a dataset-level contract digest is calculated.
    """

    resolved_input = input_path.resolve()
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - platform dependency profile.
        raise ImportError("pyarrow is required for Parquet task datasets") from exc

    if resolved_input.is_file():
        if resolved_input.suffix.casefold() != ".parquet":
            raise ValueError("A single task dataset input must be a Parquet file")
        digest = _path_sha256(resolved_input)
        rows = int(pq.ParquetFile(resolved_input).metadata.num_rows)
        return ResolvedParquetDataset(
            input_path=resolved_input,
            input_kind="single_file",
            parts=(
                ParquetDatasetPart(
                    path=resolved_input,
                    relative_path=resolved_input.name,
                    sha256=digest,
                    rows=rows,
                ),
            ),
            dataset_sha256=digest,
            total_rows=rows,
        )
    if not resolved_input.is_dir():
        raise FileNotFoundError(input_path)

    actual_paths = sorted(
        (path.resolve() for path in resolved_input.glob("*.parquet") if path.is_file()),
        key=lambda path: path.name,
    )
    if not actual_paths:
        raise ValueError("Partitioned task dataset directory contains no Parquet parts")
    invalid_names = [
        path.name for path in actual_paths if not re.fullmatch(r"part-[0-9]+\.parquet", path.name)
    ]
    if invalid_names:
        raise ValueError(f"Partitioned task dataset has noncanonical part filenames: {invalid_names}")

    manifest_path = _nearest_bulk_build_manifest(resolved_input)
    manifest_sha256 = _path_sha256(manifest_path)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read partitioned task build manifest: {manifest_path}") from exc
    shard_artifacts = manifest.get("shard_artifacts")
    if not isinstance(shard_artifacts, list):
        raise ValueError("Partitioned task build manifest requires shard_artifacts")

    manifest_root = manifest_path.parent.resolve()
    if not resolved_input.is_relative_to(manifest_root):
        raise ValueError("Task directory must be contained by its build manifest root")
    dataset_relative_path = resolved_input.relative_to(manifest_root).as_posix()
    listed: dict[Path, ParquetDatasetPart] = {}
    for index, artifact in enumerate(shard_artifacts):
        if not isinstance(artifact, Mapping):
            raise ValueError(f"Invalid shard_artifacts entry at index {index}")
        relative_text = str(artifact.get("relative_path", "")).strip()
        if not relative_text:
            continue
        relative = Path(relative_text)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe shard relative_path in build manifest: {relative_text!r}")
        candidate = (manifest_root / relative).resolve()
        if not candidate.is_relative_to(manifest_root) or candidate.parent != resolved_input:
            continue
        if candidate in listed:
            raise ValueError(f"Duplicate task part entry in build manifest: {relative_text}")
        digest = _require_hex_sha256(
            artifact.get("sha256", ""),
            f"shard_artifacts[{index}].sha256",
        )
        try:
            rows = int(artifact["rows"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid row count for task part {relative_text}") from exc
        if rows < 0:
            raise ValueError(f"Negative row count for task part {relative_text}")
        listed[candidate] = ParquetDatasetPart(
            path=candidate,
            relative_path=relative.as_posix(),
            sha256=digest,
            rows=rows,
        )

    actual_set = set(actual_paths)
    listed_set = set(listed)
    if actual_set != listed_set:
        unmanifested = sorted(path.name for path in actual_set - listed_set)
        missing = sorted(path.name for path in listed_set - actual_set)
        raise ValueError(
            "Partitioned task parts do not exactly match build_manifest.json; "
            f"unmanifested={unmanifested}, missing={missing}"
        )
    parts = tuple(sorted(listed.values(), key=lambda part: part.relative_path))
    for part in parts:
        if not part.path.is_file():
            raise ValueError(f"Manifest-listed task part is missing: {part.relative_path}")
        if _path_sha256(part.path) != part.sha256:
            raise ValueError(f"Task part SHA-256 mismatch: {part.relative_path}")
        actual_rows = int(pq.ParquetFile(part.path).metadata.num_rows)
        if actual_rows != part.rows:
            raise ValueError(
                f"Task part row-count mismatch: {part.relative_path}; "
                f"manifest={part.rows}, actual={actual_rows}"
            )

    contract = {
        "schema_version": MANIFEST_BOUND_PARQUET_DATASET_SCHEMA_VERSION,
        "manifest_sha256": manifest_sha256,
        "dataset_relative_path": dataset_relative_path,
        "parts": [
            {
                "relative_path": part.relative_path,
                "sha256": part.sha256,
                "rows": part.rows,
            }
            for part in parts
        ],
    }
    return ResolvedParquetDataset(
        input_path=resolved_input,
        input_kind="manifest_bound_directory",
        parts=parts,
        dataset_sha256=stable_json_digest(contract),
        total_rows=sum(part.rows for part in parts),
        manifest_path=manifest_path.resolve(),
        manifest_sha256=manifest_sha256,
    )


def _atomic_json_document(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _hash_group_partition(group_id: str, config: SplitConfig, *, seed_offset: int = 0) -> str:
    digest = hashlib.sha256(f"{config.seed + seed_offset}|{group_id}".encode()).digest()
    unit_interval = int.from_bytes(digest[:8], "big") / 2**64
    if unit_interval < config.train_fraction:
        return "train"
    if unit_interval < config.train_fraction + config.validation_fraction:
        return "validation"
    return "test"


def _stream_group_and_partition(
    row: pd.Series,
    config: SplitConfig,
) -> tuple[str, str, str]:
    molecule = _normalized_cell(row[config.molecule_id_column])
    if not molecule:
        raise ValueError("Streaming split requires nonblank molecule identity")
    if config.strategy == "molecule_grouped":
        group = f"molecule:{molecule}"
        return group, _hash_group_partition(group, config), "stable_sha256_group_hash"
    if config.strategy == "scaffold":
        smiles = _normalized_cell(row[config.smiles_column])
        if not smiles:
            raise ValueError("Streaming scaffold split requires nonblank canonical SMILES")
        scaffold, method = scaffold_key(smiles)
        group = f"scaffold:{scaffold}"
        return group, _hash_group_partition(group, config), method
    if config.strategy == "source_holdout":
        source = _normalized_cell(row[config.source_id_column])
        if not source:
            raise ValueError("Streaming source holdout requires nonblank source identity")
        group = f"source:{source}"
        return group, _hash_group_partition(group, config), "stable_sha256_group_hash"
    if config.strategy in {"protein_holdout", "target_holdout"}:
        column = config.protein_id_column if config.strategy == "protein_holdout" else config.target_id_column
        value = _normalized_cell(row[column])
        if not value:
            raise ValueError(f"Streaming {config.strategy} requires nonblank grouping identity")
        group = f"{config.strategy}:{value}"
        return group, _hash_group_partition(group, config), "stable_sha256_group_hash"
    if config.strategy == "double_cold":
        protein = _normalized_cell(row[config.protein_id_column])
        if not protein:
            raise ValueError("Streaming double-cold split requires nonblank protein identity")
        molecule_group = f"molecule:{molecule}"
        protein_group = f"protein:{protein}"
        molecule_split = _hash_group_partition(molecule_group, config)
        protein_split = _hash_group_partition(protein_group, config, seed_offset=7919)
        split = molecule_split if molecule_split == protein_split else "excluded_mixed"
        return f"{molecule_group}|{protein_group}", split, "independent_sha256_hash_intersection"
    raise ValueError(
        f"Strategy {config.strategy!r} has no out-of-core implementation; "
        "materialize a separately audited scalable strategy rather than falling back silently"
    )


def stream_hash_group_split_manifest(
    task_parquet_path: Path,
    output_path: Path,
    config: SplitConfig,
    *,
    batch_size: int = 50_000,
) -> dict[str, Any]:
    """Write an immutable group split in bounded memory from a Parquet task view.

    The scalable path intentionally uses a stable hash of the grouping entity,
    not the small-data greedy balancer.  Fractions are therefore approximate.
    Exact entity exclusion is guaranteed by construction and record-ID
    uniqueness is verified with a temporary disk-backed primary key.
    """

    config.validate()
    if config.strategy not in {
        "molecule_grouped",
        "scaffold",
        "source_holdout",
        "protein_holdout",
        "target_holdout",
        "double_cold",
    }:
        raise ValueError(f"Strategy {config.strategy!r} is not implemented for bounded-memory manifests")
    if not 1 <= batch_size <= 250_000:
        raise ValueError("batch_size must be between 1 and 250000 rows")
    if task_parquet_path.resolve() == output_path.resolve():
        raise ValueError("Input task and output manifest paths must differ")
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - platform dependency profile.
        raise ImportError("pyarrow is required for bounded-memory Parquet splitting") from exc

    dataset = resolve_manifest_bound_parquet_dataset(task_parquet_path)
    required_columns = {
        config.record_id_column,
        config.molecule_id_column,
        config.protein_id_column,
        config.source_id_column,
        "assay_id",
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
        "default_task_eligible",
        "evidence_domain",
        "endpoint",
        "assay_family",
    }
    if config.strategy == "scaffold":
        required_columns.add(config.smiles_column)
    if config.strategy == "target_holdout":
        required_columns.add(config.target_id_column)
    if config.allow_derived_labels:
        required_columns.update({config.derived_label_lineage_column, "sensitivity_task_eligible"})
    parquets: list[Any] = []
    reference_schema: Any = None
    for part in dataset.parts:
        parquet = pq.ParquetFile(part.path)
        missing = sorted(required_columns - set(parquet.schema_arrow.names))
        if missing:
            raise ValueError(
                f"Streaming split task part {part.relative_path!r} is missing columns: {missing}"
            )
        if reference_schema is None:
            reference_schema = parquet.schema_arrow
        elif not reference_schema.equals(parquet.schema_arrow, check_metadata=False):
            raise ValueError(f"Streaming task Parquet schema changed across parts: {part.relative_path}")
        parquets.append(parquet)

    source_sha256 = dataset.dataset_sha256
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_output.unlink(missing_ok=True)
    writer: Any = None
    partition_counts: Counter[str] = Counter()
    method_counts: Counter[str] = Counter()
    outcome_counts: Counter[str] = Counter()
    task_signature: tuple[str, ...] | None = None
    rows_seen = 0
    maximum_batch_rows = 0
    maximum_batch_deep_bytes = 0
    manifest_schema = pa.schema(
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
    signature_columns = (
        "task_id",
        "task_type",
        "evidence_domain",
        "endpoint",
        "assay_family",
        "label_kind",
        "label_unit",
    )
    sqlite_directory: tempfile.TemporaryDirectory[str] | None = None
    connection: sqlite3.Connection | None = None
    try:
        sqlite_directory = tempfile.TemporaryDirectory(prefix="platform-split-", dir=output_path.parent)
        connection = sqlite3.connect(str(Path(sqlite_directory.name) / "record_ids.sqlite3"))
        connection.execute("CREATE TABLE record_ids (record_id TEXT PRIMARY KEY) WITHOUT ROWID")
        batches = (batch for parquet in parquets for batch in parquet.iter_batches(batch_size=batch_size))
        for batch in batches:
            frame = batch.to_pandas()
            if frame.empty:
                continue
            maximum_batch_rows = max(maximum_batch_rows, len(frame))
            maximum_batch_deep_bytes = max(
                maximum_batch_deep_bytes,
                int(frame.memory_usage(index=True, deep=True).sum()),
            )
            current_signature: list[str] = []
            for column in signature_columns:
                values = frame[column].fillna("").astype(str).str.strip().unique().tolist()
                if len(values) != 1 or not values[0]:
                    raise ValueError(f"Streaming task is heterogeneous or blank in {column}: {values[:5]}")
                current_signature.append(values[0])
            signature_tuple = tuple(current_signature)
            if task_signature is None:
                task_signature = signature_tuple
            elif task_signature != signature_tuple:
                raise ValueError("Task signature changed across Parquet batches")

            access = frame["access_class"].fillna("").astype(str).str.strip().str.lower()
            if not access.eq("public_redistributable").all():
                raise ValueError("Streaming public split contains non-public rows")
            inclusion = frame["inclusion_status"].fillna("").astype(str).str.strip().str.lower()
            if not inclusion.eq("included").all():
                raise ValueError("Streaming splits require inclusion_status=included for every row")
            required_nonblank = (
                config.record_id_column,
                config.molecule_id_column,
                config.protein_id_column,
                config.source_id_column,
                "assay_id",
                "snapshot_id",
                "source_record_id",
                "label_relation",
                "observation_kind",
            )
            blank_columns = [
                column
                for column in required_nonblank
                if frame[column].fillna("").astype(str).str.strip().eq("").any()
            ]
            if blank_columns:
                raise ValueError(
                    f"Streaming task has blank required identity/semantic fields: {blank_columns}"
                )
            kinds = (
                frame["observation_kind"]
                .fillna("")
                .astype(str)
                .str.strip()
                .str.lower()
                .map(
                    lambda value: {
                        "observed": "experimental_raw",
                        "curated": "curated_assertion",
                        "experimental_observation": "experimental_raw",
                        "curated_label": "curated_assertion",
                    }.get(value, value)
                )
            )
            outcome_counts.update(kinds.tolist())
            canonical_kinds = {
                "experimental_raw",
                "experimental_summary",
                "curated_assertion",
                "derived",
                "prediction",
            }
            unknown_kinds = sorted(set(kinds) - canonical_kinds)
            if kinds.eq("").any() or kinds.eq("prediction").any() or unknown_kinds:
                raise ValueError(
                    f"Blank, prediction, or unknown observation kinds are prohibited: {unknown_kinds}"
                )
            if config.allow_derived_labels:
                if not kinds.eq("derived").all():
                    raise ValueError("Derived sensitivity streaming splits require derived-only rows")
                sensitivity = frame["sensitivity_task_eligible"].fillna("").astype(str).str.lower()
                default = frame["default_task_eligible"].fillna("").astype(str).str.lower()
                if not sensitivity.isin({"true", "1"}).all() or not default.isin({"false", "0"}).all():
                    raise ValueError("Derived sensitivity eligibility conjunction failed")
                lineage = frame[config.derived_label_lineage_column].fillna("").astype(str).str.strip()
                if not lineage.map(lambda value: bool(re.fullmatch(r"[0-9a-fA-F]{64}", value))).all():
                    raise ValueError("Derived sensitivity rows require SHA-256 lineage")
            else:
                if kinds.eq("derived").any():
                    raise ValueError("Derived rows are prohibited in the default streaming split path")
                eligible = frame["default_task_eligible"].fillna("").astype(str).str.lower()
                if not eligible.isin({"true", "1"}).all():
                    raise ValueError("Default-ineligible rows are prohibited")

            record_ids = frame[config.record_id_column].fillna("").astype(str).str.strip().tolist()
            if any(not value for value in record_ids):
                raise ValueError("Streaming split record IDs may not be blank")
            before = connection.total_changes
            connection.executemany(
                "INSERT OR IGNORE INTO record_ids(record_id) VALUES (?)",
                ((value,) for value in record_ids),
            )
            if connection.total_changes - before != len(record_ids):
                raise ValueError("Duplicate record ID detected by disk-backed streaming audit")
            connection.commit()

            groups: list[str] = []
            splits: list[str] = []
            for _, row in frame.iterrows():
                group, split, method = _stream_group_and_partition(row, config)
                groups.append(group)
                splits.append(split)
                partition_counts[split] += 1
                method_counts[method] += 1
            count = len(frame)
            manifest_frame = pd.DataFrame(
                {
                    "record_id": record_ids,
                    "split": splits,
                    "group_id": groups,
                    "strategy": [config.strategy] * count,
                    "split_name": [config.name] * count,
                    "seed": np.full(count, config.seed, dtype=np.int64),
                    "molecule_id": frame[config.molecule_id_column].fillna("").astype(str),
                    "protein_id": frame[config.protein_id_column].fillna("").astype(str),
                    "task_id": frame["task_id"].fillna("").astype(str),
                    "source_id": frame[config.source_id_column].fillna("").astype(str),
                    "document_year": (
                        frame[config.date_column].fillna("").astype(str)
                        if config.date_column in frame.columns
                        else [""] * count
                    ),
                    "observation_kind": kinds,
                    "source_row_index": np.arange(rows_seen, rows_seen + count, dtype=np.int64),
                }
            )
            table = pa.Table.from_pandas(
                manifest_frame,
                schema=manifest_schema,
                preserve_index=False,
                safe=True,
            )
            if writer is None:
                writer = pq.ParquetWriter(temporary_output, manifest_schema, compression="zstd")
            writer.write_table(table, row_group_size=batch_size)
            rows_seen += count
        if writer is None or rows_seen == 0:
            raise ValueError("Streaming split refuses an empty task artifact")
        if rows_seen != dataset.total_rows:
            raise ValueError(
                f"Resolved task row count changed during streaming: expected={dataset.total_rows}, "
                f"observed={rows_seen}"
            )
        writer.close()
        writer = None
        proxy_method_counts = {
            method: int(count)
            for method, count in method_counts.items()
            if int(count) and is_exact_smiles_proxy_method(method)
        }
        if config.strategy == "scaffold" and proxy_method_counts:
            raise ValueError(
                "A true streaming scaffold split cannot admit exact-SMILES proxy groups; "
                f"methods={sorted(proxy_method_counts)}"
            )
        if any(partition_counts[name] == 0 for name in PARTITIONS):
            raise ValueError(
                f"Stable hash split produced an empty partition: {dict(sorted(partition_counts.items()))}"
            )
        os.replace(temporary_output, output_path)
    finally:
        if writer is not None:
            writer.close()
        temporary_output.unlink(missing_ok=True)
        if connection is not None:
            connection.close()
        if sqlite_directory is not None:
            sqlite_directory.cleanup()

    metadata = {
        "schema_version": STREAMING_SPLIT_SCHEMA_VERSION,
        "split_name": config.name,
        "strategy": config.strategy,
        "algorithm": "stable_sha256_group_hash_bounded_memory_v1",
        "fraction_policy": "approximate_under_stable_group_hash_not_greedy_or_stratified",
        "source_task_path": Path(os.path.relpath(dataset.input_path, start=output_path.parent)).as_posix(),
        "source_dataset_sha256": source_sha256,
        "source_dataset": dataset.binding_payload(),
        "source_dataset_manifest_path": (
            Path(os.path.relpath(dataset.manifest_path, start=output_path.parent)).as_posix()
            if dataset.manifest_path is not None
            else None
        ),
        "manifest_path": output_path.name,
        "manifest_sha256": _path_sha256(output_path),
        "record_count": rows_seen,
        "partition_counts": dict(sorted(partition_counts.items())),
        "group_method_row_counts": dict(sorted(method_counts.items())),
        "observation_kind_counts": dict(sorted(outcome_counts.items())),
        "task_signature": dict(zip(signature_columns, task_signature or (), strict=True)),
        "record_id_uniqueness": "complete_disk_backed_primary_key_audit",
        "row_order_binding": (
            "manifest source_row_index exactly matches source Parquet row order across the "
            "deterministically sorted source part contract"
        ),
        "exact_group_exclusion": "guaranteed_by_deterministic_group_to_partition_function",
        "near_duplicate_audit": "separate_required_not_implied_by_group_hash",
        "claim_readiness": (
            "not_claim_ready_until_separate_cross_partition_near_duplicate_audit_is_complete_and_accepted"
        ),
        "bounded_memory": {
            "configured_batch_rows": batch_size,
            "maximum_observed_batch_rows": maximum_batch_rows,
            "maximum_observed_input_pandas_deep_bytes": maximum_batch_deep_bytes,
            "record_id_uniqueness_state": "temporary_sqlite_on_disk",
        },
        "config": asdict(config),
        "config_sha256": stable_json_digest(asdict(config)),
    }
    sidecar = output_path.with_suffix(output_path.suffix + ".manifest.json")
    _atomic_json_document(sidecar, metadata)
    metadata["sidecar_path"] = sidecar.name
    metadata["sidecar_sha256"] = _path_sha256(sidecar)
    return metadata
