"""Deterministic holdout and cross-validation strategies for chemical data."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    StratifiedGroupKFold,
    TimeSeriesSplit,
)

from .features import fingerprint_matrix, scaffold_key

SUPPORTED_SPLIT_STRATEGIES = ("scaffold", "chemical", "temporal", "random")


@dataclass(frozen=True)
class SplitResult:
    """Positional train/test indices plus a JSON-serializable audit record."""

    train_indices: np.ndarray
    test_indices: np.ndarray
    metadata: dict[str, Any]

    def assignments(self, n_rows: int) -> np.ndarray:
        labels = np.full(n_rows, "unassigned", dtype=object)
        labels[self.train_indices] = "train"
        labels[self.test_indices] = "test"
        return labels


def _valid_stratification(y: np.ndarray | None, test_size: float) -> bool:
    if y is None or len(y) < 4:
        return False
    values, counts = np.unique(y, return_counts=True)
    if len(values) < 2 or counts.min() < 2:
        return False
    n_test = int(math.ceil(len(y) * test_size))
    n_train = len(y) - n_test
    return n_test >= len(values) and n_train >= len(values)


def structure_groups(
    data: pd.DataFrame,
    *,
    smiles_column: str = "smiles",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Resolve the most stable available compound identity for leakage control."""

    candidates = (
        "structure_id",
        "standard_inchi_key",
        "standardized_smiles",
        "canonical_smiles",
        smiles_column,
    )
    identity_column = next(
        (
            column
            for column in candidates
            if column in data.columns and data[column].fillna("").astype(str).str.strip().ne("").any()
        ),
        smiles_column,
    )
    raw = data[identity_column].fillna("").astype(str).str.strip()
    fallback = data[smiles_column].fillna("").astype(str).str.strip()
    values = raw.where(raw.ne(""), fallback)
    # Missing identities must not collapse all structureless rows into one group.
    values = values.where(
        values.ne(""), pd.Series([f"missing:{index}" for index in range(len(data))], index=data.index)
    )
    return values.to_numpy(dtype=object), {
        "identity_column": identity_column,
        "n_unique_structures": int(values.nunique()),
    }


def _random_holdout(
    groups: np.ndarray,
    *,
    test_size: float,
    random_state: int,
    y: np.ndarray | None,
    task_type: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    train, test, metadata = _candidate_group_holdouts(
        groups,
        test_size=test_size,
        random_state=random_state,
        y=y,
        task_type=task_type,
    )
    metadata["stratified"] = task_type == "classification" and _valid_stratification(y, test_size)
    metadata["grouping"] = "compound_random"
    return train, test, metadata


def _candidate_group_holdouts(
    groups: np.ndarray,
    *,
    test_size: float,
    random_state: int,
    y: np.ndarray | None,
    task_type: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        raise ValueError("At least two structure groups are required for a group holdout")

    splitter = GroupShuffleSplit(
        n_splits=min(128, max(24, len(unique_groups) * 2)), test_size=test_size, random_state=random_state
    )
    target_fraction = float(test_size)
    global_mean = float(np.mean(y)) if y is not None and len(y) else 0.0
    global_scale = float(np.std(y)) if y is not None and np.std(y) > 0 else 1.0
    best: tuple[float, np.ndarray, np.ndarray] | None = None
    for train, test in splitter.split(np.zeros(len(groups)), y, groups):
        if not len(train) or not len(test):
            continue
        score = abs(len(test) / len(groups) - target_fraction)
        if y is not None:
            if task_type == "classification":
                train_classes = np.unique(y[train])
                test_classes = np.unique(y[test])
                all_classes = np.unique(y)
                if len(train_classes) != len(all_classes) or len(test_classes) != len(all_classes):
                    score += 10.0
                else:
                    score += 0.35 * abs(float(np.mean(y[test])) - global_mean)
            else:
                score += 0.10 * abs(float(np.mean(y[test])) - global_mean) / global_scale
        candidate = (float(score), np.sort(train), np.sort(test))
        if best is None or candidate[0] < best[0]:
            best = candidate
    if best is None:
        raise ValueError("Could not construct a non-empty group holdout")
    return (
        best[1],
        best[2],
        {
            "n_structure_groups": int(len(unique_groups)),
            "structure_group_overlap": int(len(set(groups[best[1]]) & set(groups[best[2]]))),
            "group_selection_score": best[0],
        },
    )


def scaffold_groups(smiles: pd.Series) -> tuple[np.ndarray, dict[str, Any]]:
    keys: list[str] = []
    methods: list[str] = []
    for value in smiles.fillna("").astype(str):
        key, method = scaffold_key(value)
        keys.append(key)
        methods.append(method)
    method_counts = pd.Series(methods, dtype=str).value_counts().sort_index().to_dict()
    return np.asarray(keys, dtype=object), {
        "grouping": "bemis_murcko"
        if all(method.startswith("bemis_murcko") for method in methods)
        else "mixed_or_exact_smiles_proxy",
        "grouping_method_counts": {str(key): int(value) for key, value in method_counts.items()},
        "chemistry_fallback_used": any(method == "exact_smiles_proxy" for method in methods),
    }


def chemical_cluster_groups(
    smiles: pd.Series,
    *,
    random_state: int,
    n_bits: int = 1024,
) -> tuple[np.ndarray, dict[str, Any]]:
    n_rows = len(smiles)
    if n_rows < 3:
        return np.arange(n_rows), {"grouping": "singleton", "fingerprint_backend": "none"}
    matrix, backend = fingerprint_matrix(smiles, backend="auto", n_bits=n_bits)
    n_clusters = min(n_rows - 1, max(3, min(64, int(round(math.sqrt(n_rows))))))
    model = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10,
        batch_size=min(1024, max(32, n_rows)),
        reassignment_ratio=0.0,
    )
    labels = model.fit_predict(matrix)
    return labels.astype(int), {
        "grouping": "minibatch_kmeans_fingerprint_clusters",
        "fingerprint_backend": backend,
        "fingerprint_bits": int(n_bits),
        "n_structure_groups": int(len(np.unique(labels))),
        "chemistry_fallback_used": backend != "rdkit_morgan",
    }


def extract_year(value: object) -> float:
    """Extract the earliest plausible year from numbers or delimited provenance."""

    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    years = [int(item) for item in re.findall(r"(?<!\d)(?:19|20)\d{2}(?!\d)", str(value))]
    if years:
        return float(min(years))
    try:
        numeric = float(str(value))
    except (TypeError, ValueError):
        return np.nan
    return numeric if 1900 <= numeric <= 2100 else np.nan


def infer_time_column(data: pd.DataFrame, requested: str | None = None) -> str | None:
    if requested and requested in data.columns:
        return requested
    candidates = (
        "document_year",
        "document_years",
        "assay_year",
        "year",
        "measurement_date",
        "assay_date",
        "date",
    )
    return next((column for column in candidates if column in data.columns), None)


def _temporal_holdout(
    data: pd.DataFrame,
    *,
    test_size: float,
    y: np.ndarray | None,
    task_type: str,
    time_column: str | None,
    structure_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    column = infer_time_column(data, time_column)
    if column is None:
        raise ValueError("No supported time/year column was found")
    row_years = data[column].map(extract_year).to_numpy(dtype=float)
    group_first_year: dict[object, float] = {}
    for group in np.unique(structure_ids):
        observed = row_years[structure_ids == group]
        observed = observed[np.isfinite(observed)]
        group_first_year[group] = float(np.min(observed)) if len(observed) else np.nan
    years = np.asarray([group_first_year[group] for group in structure_ids], dtype=float)
    n_dated = int(np.sum(np.isfinite(years)))
    minimum_dated = max(4, int(np.ceil(0.5 * len(data))))
    if n_dated < minimum_dated:
        raise ValueError(f"Insufficient dated structures for temporal holdout: {n_dated}/{len(data)}")
    valid_years = np.unique(years[np.isfinite(years)])
    if len(valid_years) < 2:
        raise ValueError("Temporal splitting requires at least two distinct known years")

    target_fraction = float(test_size)
    minimum_test_rows = max(2, int(np.ceil(len(data) * target_fraction * 0.5)))
    candidates: list[tuple[float, float, np.ndarray, np.ndarray]] = []
    for cutoff in valid_years[1:]:
        test = np.flatnonzero(np.isfinite(years) & (years >= cutoff))
        train = np.flatnonzero(~np.isin(np.arange(len(data)), test))
        if len(test) < minimum_test_rows or len(train) < 2:
            continue
        score = abs(len(test) / len(data) - target_fraction)
        if task_type == "classification" and y is not None:
            n_classes = len(np.unique(y))
            if len(np.unique(y[train])) != n_classes or len(np.unique(y[test])) != n_classes:
                score += 10.0
        candidates.append((score, float(cutoff), train, test))
    if not candidates:
        raise ValueError("No viable temporal cutoff was found")
    score, cutoff, train, test = min(candidates, key=lambda item: (item[0], -item[1]))
    return (
        np.sort(train),
        np.sort(test),
        {
            "time_column": column,
            "cutoff_year": int(cutoff),
            "n_missing_time_assigned_to_train": int(np.sum(~np.isfinite(years))),
            "known_year_range": [int(valid_years.min()), int(valid_years.max())],
            "selection_score": float(score),
            "time_basis": "earliest known year per structure",
        },
    )


def _split_digest(
    data: pd.DataFrame, assignments: np.ndarray, smiles_column: str, target_column: str | None
) -> str:
    columns = [smiles_column] + ([target_column] if target_column and target_column in data.columns else [])
    records = data[columns].copy()
    records["split"] = assignments
    records = records.fillna("").astype(str).sort_values(columns + ["split"], kind="mergesort")
    payload = records.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def make_split(
    data: pd.DataFrame,
    *,
    strategy: str = "scaffold",
    test_size: float = 0.2,
    random_state: int = 13,
    smiles_column: str = "smiles",
    target_column: str | None = None,
    task_type: str = "regression",
    time_column: str | None = None,
) -> SplitResult:
    """Construct an audited holdout without compound/scaffold group leakage."""

    requested = strategy.strip().lower()
    if requested not in SUPPORTED_SPLIT_STRATEGIES:
        raise ValueError(f"strategy must be one of {SUPPORTED_SPLIT_STRATEGIES}, got {strategy!r}")
    if task_type not in {"regression", "classification"}:
        raise ValueError("task_type must be 'regression' or 'classification'")
    if not 0.05 <= test_size <= 0.5:
        raise ValueError("test_size must be between 0.05 and 0.5")
    if len(data) < 4:
        raise ValueError("At least four rows are required to split a dataset")
    if smiles_column not in data.columns:
        raise KeyError(f"Missing structure column: {smiles_column}")

    y = data[target_column].to_numpy() if target_column and target_column in data.columns else None
    identities, identity_metadata = structure_groups(data, smiles_column=smiles_column)
    metadata: dict[str, Any] = {
        "requested_strategy": requested,
        "strategy": requested,
        "random_state": int(random_state),
        "requested_test_fraction": float(test_size),
        "fallback_reason": None,
    }

    try:
        if requested == "random":
            train, test, details = _random_holdout(
                identities, test_size=test_size, random_state=random_state, y=y, task_type=task_type
            )
        elif requested == "temporal":
            train, test, details = _temporal_holdout(
                data,
                test_size=test_size,
                y=y,
                task_type=task_type,
                time_column=time_column,
                structure_ids=identities,
            )
        else:
            if requested == "scaffold":
                groups, grouping = scaffold_groups(data[smiles_column])
            else:
                groups, grouping = chemical_cluster_groups(data[smiles_column], random_state=random_state)
            # One registered structure must map to one group even if source rows
            # contain alternate representations.
            identity_to_group = {
                identity: sorted({str(group) for group in groups[identities == identity]})[0]
                for identity in np.unique(identities)
            }
            groups = np.asarray([identity_to_group[identity] for identity in identities], dtype=object)
            train, test, details = _candidate_group_holdouts(
                groups, test_size=test_size, random_state=random_state, y=y, task_type=task_type
            )
            details.update(grouping)
    except ValueError as exc:
        train, test, details = _random_holdout(
            identities, test_size=test_size, random_state=random_state, y=y, task_type=task_type
        )
        metadata["strategy"] = "random"
        metadata["fallback_reason"] = str(exc)

    assignments = np.full(len(data), "unassigned", dtype=object)
    assignments[train] = "train"
    assignments[test] = "test"
    metadata.update(identity_metadata)
    metadata.update(details)
    metadata.update(
        {
            "n_rows": int(len(data)),
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "actual_test_fraction": float(len(test) / len(data)),
            "structure_identity_overlap": int(len(set(identities[train]) & set(identities[test]))),
            "split_sha256": _split_digest(data, assignments, smiles_column, target_column),
        }
    )
    return SplitResult(np.asarray(train, dtype=int), np.asarray(test, dtype=int), metadata)


def _classification_folds_are_valid(folds: list[tuple[np.ndarray, np.ndarray]], y: np.ndarray) -> bool:
    classes = len(np.unique(y))
    return bool(folds) and all(
        len(np.unique(y[train])) == classes and len(np.unique(y[test])) == classes for train, test in folds
    )


def make_cv_folds(
    data: pd.DataFrame,
    *,
    strategy: str,
    n_splits: int = 3,
    random_state: int = 13,
    smiles_column: str = "smiles",
    target_column: str,
    task_type: str,
    time_column: str | None = None,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], dict[str, Any]]:
    """Build deterministic model-selection folds aligned to the holdout strategy."""

    if len(data) < 6:
        raise ValueError("At least six rows are required for cross-validation")
    y = data[target_column].to_numpy()
    identities, identity_metadata = structure_groups(data, smiles_column=smiles_column)
    requested_splits = max(2, int(n_splits))
    details: dict[str, Any] = {"requested_strategy": strategy, "strategy": strategy, "fallback_reason": None}
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    try:
        if strategy == "random":
            if task_type == "classification":
                min_class = int(pd.Series(y).value_counts().min())
                count = min(requested_splits, min_class, len(np.unique(identities)))
                if count < 2:
                    raise ValueError("Insufficient class support for stratified cross-validation")
                splitter = StratifiedGroupKFold(n_splits=count, shuffle=True, random_state=random_state)
                folds = [(train, test) for train, test in splitter.split(data, y, identities)]
                if not _classification_folds_are_valid(folds, y):
                    raise ValueError("Compound-grouped folds do not contain all classes")
            else:
                count = min(requested_splits, len(np.unique(identities)))
                if count < 2:
                    raise ValueError("Insufficient compounds for grouped cross-validation")
                splitter = GroupKFold(n_splits=count)
                folds = [(train, test) for train, test in splitter.split(data, y, identities)]
            details["grouping"] = "compound_random"
        elif strategy in {"scaffold", "chemical"}:
            if strategy == "scaffold":
                groups, grouping = scaffold_groups(data[smiles_column])
            else:
                groups, grouping = chemical_cluster_groups(data[smiles_column], random_state=random_state)
            identity_to_group = {
                identity: sorted({str(group) for group in groups[identities == identity]})[0]
                for identity in np.unique(identities)
            }
            groups = np.asarray([identity_to_group[identity] for identity in identities], dtype=object)
            count = min(requested_splits, len(np.unique(groups)))
            if count < 2:
                raise ValueError("Insufficient structure groups for grouped cross-validation")
            if task_type == "classification":
                splitter = StratifiedGroupKFold(n_splits=count, shuffle=True, random_state=random_state)
                folds = [(train, test) for train, test in splitter.split(data, y, groups)]
                if not _classification_folds_are_valid(folds, y):
                    raise ValueError("Grouped folds do not contain all classes")
            else:
                splitter = GroupKFold(n_splits=count)
                folds = [(train, test) for train, test in splitter.split(data, y, groups)]
            details.update(grouping)
            details["n_structure_groups"] = int(len(np.unique(groups)))
        elif strategy == "temporal":
            column = infer_time_column(data, time_column)
            if column is None:
                raise ValueError("No supported time/year column was found")
            row_years = data[column].map(extract_year).to_numpy(dtype=float)
            group_years: dict[object, float] = {}
            for identity in np.unique(identities):
                observed = row_years[identities == identity]
                observed = observed[np.isfinite(observed)]
                group_years[identity] = float(np.min(observed)) if len(observed) else np.nan
            finite_group_years = [value for value in group_years.values() if np.isfinite(value)]
            if not finite_group_years:
                raise ValueError("Insufficient dated structures for temporal cross-validation")
            missing_year = min(finite_group_years) - 1
            ordered_groups = sorted(
                np.unique(identities),
                key=lambda identity: (
                    group_years.get(identity, np.nan)
                    if np.isfinite(group_years.get(identity, np.nan))
                    else missing_year,
                    str(identity),
                ),
            )
            if len(ordered_groups) < requested_splits + 2:
                raise ValueError("Insufficient dated rows for temporal cross-validation")
            splitter = TimeSeriesSplit(n_splits=min(requested_splits, len(ordered_groups) - 1))
            folds = []
            ordered_groups_array = np.asarray(ordered_groups, dtype=object)
            for train_groups, test_groups in splitter.split(ordered_groups_array):
                train = np.flatnonzero(np.isin(identities, ordered_groups_array[train_groups]))
                test = np.flatnonzero(np.isin(identities, ordered_groups_array[test_groups]))
                folds.append((train, test))
            if task_type == "classification" and not _classification_folds_are_valid(folds, y):
                raise ValueError("Temporal folds do not contain all classes")
            details["time_column"] = column
        else:
            raise ValueError(f"Unsupported CV strategy: {strategy}")
    except ValueError as exc:
        details["strategy"] = "random"
        details["fallback_reason"] = str(exc)
        if task_type == "classification":
            count = min(
                requested_splits,
                int(pd.Series(y).value_counts().min()),
                len(np.unique(identities)),
            )
            if count < 2:
                raise ValueError("Insufficient class support for fallback cross-validation") from exc
            splitter = StratifiedGroupKFold(n_splits=count, shuffle=True, random_state=random_state)
            folds = [(train, test) for train, test in splitter.split(data, y, identities)]
        else:
            count = min(requested_splits, len(np.unique(identities)))
            if count < 2:
                raise ValueError("Insufficient compounds for fallback cross-validation") from exc
            splitter = GroupKFold(n_splits=count)
            folds = [(train, test) for train, test in splitter.split(data, y, identities)]

    fold_signature = [
        {"train": [int(value) for value in train], "test": [int(value) for value in test]}
        for train, test in folds
    ]
    details.update(identity_metadata)
    details["n_splits"] = int(len(folds))
    details["maximum_structure_overlap"] = int(
        max((len(set(identities[train]) & set(identities[test])) for train, test in folds), default=0)
    )
    details["cv_sha256"] = hashlib.sha256(
        json.dumps(fold_signature, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return folds, details
