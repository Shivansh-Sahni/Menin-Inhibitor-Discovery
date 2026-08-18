"""Exact, disk-backed QC, attrition, missingness, and EDA for canonical data."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sqlite3
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .platform_data_pipeline import PUBLIC_ACCESS_CLASS, _atomic_frame, _atomic_json, _utc_now
from .platform_data_schema import (
    SCHEMA_VERSION,
    arrow_schema_contract,
    canonical_json,
    clean_text,
    validate_table,
)
from .platform_data_sources import sha256_file

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if "MPLCONFIGDIR" not in os.environ:
    matplotlib_cache = _PROJECT_ROOT / ".matplotlib_cache"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(matplotlib_cache)
if "XDG_CACHE_HOME" not in os.environ:
    xdg_cache = _PROJECT_ROOT / ".cache"
    xdg_cache.mkdir(parents=True, exist_ok=True)
    os.environ["XDG_CACHE_HOME"] = str(xdg_cache)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _parquet_parts(root: Path, name: str) -> list[Path]:
    single = root / f"{name}.parquet"
    if single.exists():
        return [single]
    directory = root / name
    return sorted(directory.glob("*.parquet")) if directory.exists() else []


def _read_small_dataset(root: Path, name: str) -> pd.DataFrame:
    parts = _parquet_parts(root, name)
    if not parts:
        raise FileNotFoundError(f"Missing canonical dataset: {name}")
    return pd.concat([pd.read_parquet(path) for path in parts], ignore_index=True, sort=False)


def _issue(
    issues: list[dict[str, Any]],
    severity: str,
    code: str,
    table: str,
    count: int,
    detail: str,
) -> None:
    if count <= 0:
        return
    issues.append(
        {
            "severity": severity,
            "code": code,
            "table": table,
            "count": int(count),
            "detail": detail,
        }
    )


def _safe_missing(series: pd.Series) -> pd.Series:
    if series.dtype == object or isinstance(series.dtype, pd.StringDtype):
        return series.isna() | series.map(clean_text).eq("")
    return series.isna()


def _accumulate_missingness(
    table: str,
    frame: pd.DataFrame,
    missing_counts: defaultdict[tuple[str, str, str], int],
    denominators: defaultdict[tuple[str, str, str], int],
    *,
    stratify_by_evidence_domain: bool = False,
) -> None:
    """Accumulate exact per-field missingness, including zero-row schemas."""

    if stratify_by_evidence_domain and "evidence_domain" in frame.columns and not frame.empty:
        groups = [
            (clean_text(domain) or "<NA>", group)
            for domain, group in frame.groupby("evidence_domain", dropna=False)
        ]
    else:
        groups = [("<all>", frame)]
    for domain, group in groups:
        for column in frame.columns:
            key = (table, clean_text(column), domain)
            denominators[key] += len(group)
            missing_counts[key] += int(_safe_missing(group[column]).sum())


def _sql_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False
    if isinstance(missing, (bool, np.bool_)) and bool(missing):
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _insert_unique(
    connection: sqlite3.Connection,
    statement: str,
    rows: list[tuple[Any, ...]],
) -> int:
    before = connection.total_changes
    connection.executemany(statement, rows)
    connection.commit()
    return len(rows) - (connection.total_changes - before)


def _plot_counts(composition: pd.DataFrame, reports: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    domain = composition[composition["dimension"] == "evidence_domain"].sort_values("count", ascending=False)
    if not domain.empty:
        figure, axis = plt.subplots(figsize=(8, 4.5))
        axis.bar(domain["value"], domain["count"], color="#31688e")
        axis.set_ylabel("Canonical observations")
        axis.set_title("Evidence-domain composition")
        axis.tick_params(axis="x", rotation=30)
        figure.tight_layout()
        path = reports / "eda_domain_counts.png"
        figure.savefig(path, dpi=180)
        plt.close(figure)
        records.append({"path": path.name, "sha256": sha256_file(path), "size_bytes": path.stat().st_size})
    return records


def _create_qc_state(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=FULL;
        CREATE TABLE sources (
            source_id TEXT NOT NULL,
            snapshot_id TEXT NOT NULL,
            PRIMARY KEY(source_id, snapshot_id)
        );
        CREATE TABLE source_files (
            source_file_id TEXT PRIMARY KEY,
            source_id TEXT NOT NULL,
            snapshot_id TEXT NOT NULL
        );
        CREATE TABLE molecules (
            molecule_id TEXT PRIMARY KEY,
            has_standardized_smiles INTEGER NOT NULL,
            has_standard_inchi_key INTEGER NOT NULL,
            usable_structure INTEGER NOT NULL
        );
        CREATE TABLE proteins (
            protein_id TEXT PRIMARY KEY,
            has_sequence INTEGER NOT NULL,
            canonical_target_id TEXT NOT NULL
        );
        CREATE TABLE assays (
            assay_id TEXT PRIMARY KEY,
            assay_family TEXT NOT NULL
        );
        CREATE TABLE molecule_aliases (
            molecule_alias_id TEXT PRIMARY KEY,
            molecule_id TEXT NOT NULL,
            source_id TEXT NOT NULL,
            snapshot_id TEXT NOT NULL
        );
        CREATE TABLE protein_constructs (
            construct_id TEXT PRIMARY KEY,
            protein_id TEXT NOT NULL,
            source_id TEXT NOT NULL
        );
        CREATE TABLE development_metadata (
            development_metadata_id TEXT PRIMARY KEY,
            molecule_id TEXT NOT NULL
        );
        CREATE TABLE observations (
            observation_id TEXT PRIMARY KEY,
            source_id TEXT NOT NULL,
            snapshot_id TEXT NOT NULL,
            source_record_id TEXT NOT NULL,
            molecule_id TEXT NOT NULL,
            protein_id TEXT NOT NULL,
            assay_id TEXT NOT NULL,
            observation_kind TEXT NOT NULL,
            evidence_domain TEXT NOT NULL,
            endpoint TEXT NOT NULL,
            inclusion_status TEXT NOT NULL,
            relation TEXT NOT NULL,
            canonical_value REAL,
            canonical_unit TEXT,
            lower_bound REAL,
            upper_bound REAL,
            dedup_group_id TEXT,
            conflict_group_id TEXT,
            document_id TEXT,
            document_year TEXT,
            activity_origin_name TEXT
        );
        CREATE TABLE lineage (
            lineage_id TEXT PRIMARY KEY,
            observation_id TEXT NOT NULL,
            source_id TEXT NOT NULL,
            snapshot_id TEXT NOT NULL,
            source_file_id TEXT NOT NULL,
            lineage_role TEXT NOT NULL
        );
        CREATE TABLE tasks (
            task_id TEXT NOT NULL,
            observation_id TEXT NOT NULL,
            task_scope TEXT NOT NULL,
            task_type TEXT NOT NULL,
            source_id TEXT NOT NULL,
            snapshot_id TEXT NOT NULL,
            source_record_id TEXT NOT NULL,
            molecule_id TEXT NOT NULL,
            protein_id TEXT NOT NULL,
            assay_id TEXT NOT NULL,
            evidence_domain TEXT NOT NULL,
            endpoint TEXT NOT NULL,
            assay_family TEXT NOT NULL,
            label_kind TEXT NOT NULL,
            label_unit TEXT NOT NULL,
            observation_kind TEXT NOT NULL,
            default_task_eligible INTEGER NOT NULL,
            sensitivity_task_eligible INTEGER NOT NULL,
            required_modalities TEXT NOT NULL,
            label_relation TEXT NOT NULL,
            label_value REAL,
            label_text TEXT NOT NULL,
            label_lower_bound REAL,
            label_upper_bound REAL,
            threshold_low_nM REAL,
            threshold_high_nM REAL,
            threshold_source_value_nM REAL,
            PRIMARY KEY(task_id, observation_id)
        );
        CREATE TABLE task_input_exclusions (
            task_id TEXT NOT NULL,
            observation_id TEXT NOT NULL,
            task_scope TEXT NOT NULL,
            task_type TEXT NOT NULL,
            source_id TEXT NOT NULL,
            snapshot_id TEXT NOT NULL,
            source_record_id TEXT NOT NULL,
            molecule_id TEXT NOT NULL,
            protein_id TEXT NOT NULL,
            assay_id TEXT NOT NULL,
            canonical_target_id TEXT NOT NULL,
            required_modalities TEXT NOT NULL,
            exclusion_reason TEXT NOT NULL,
            missing_standardized_smiles INTEGER NOT NULL,
            missing_standard_inchi_key INTEGER NOT NULL,
            missing_protein_sequence INTEGER NOT NULL,
            PRIMARY KEY(task_id, observation_id, task_scope)
        );
        """
    )
    connection.commit()


def _sql_count(connection: sqlite3.Connection, query: str) -> int:
    row = connection.execute(query).fetchone()
    return int(row[0]) if row else 0


def _parquet_row_count(path: Path) -> int:
    metadata = pq.ParquetFile(path).metadata
    if metadata is None:
        raise RuntimeError(f"Parquet file lacks footer metadata: {path}")
    return int(metadata.num_rows)


def _verify_declared_arrow_schema(
    path: Path,
    declared: object,
    part_fingerprint: object,
    *,
    table: str,
    issues: list[dict[str, Any]],
) -> None:
    actual = arrow_schema_contract(pq.ParquetFile(path).schema_arrow.remove_metadata())
    if not isinstance(declared, dict) or actual != declared or part_fingerprint != actual["sha256"]:
        _issue(
            issues,
            "error",
            "arrow_schema_contract_mismatch",
            table,
            1,
            str(path),
        )


def _task_paths(canonical: Path) -> tuple[list[Path], list[Path]]:
    default = sorted((canonical / "tasks" / "default").rglob("*.parquet"))
    if not default:
        aggregate = canonical / "tasks" / "public_model_tasks.parquet"
        default = [aggregate] if aggregate.exists() else []
    sensitivity = sorted((canonical / "tasks" / "derived_sensitivity").rglob("*.parquet"))
    if not sensitivity:
        aggregate = canonical / "tasks" / "binding_free_energy_sensitivity.parquet"
        sensitivity = [aggregate] if aggregate.exists() else []
    return default, sensitivity


def _verify_task_dataset_manifest(
    canonical: Path,
    manifest: dict[str, Any],
    issues: list[dict[str, Any]],
) -> None:
    datasets = manifest.get("task_datasets", {})
    if not datasets:
        if "full_specialized" in clean_text(manifest.get("build_type", "")):
            _issue(
                issues,
                "error",
                "missing_task_dataset_manifest",
                "tasks",
                1,
                "Full build requires task_datasets",
            )
        return
    listed: set[Path] = set()
    shard_schemas = manifest.get("shard_dataset_schemas", {})
    for dataset_key, dataset in sorted(datasets.items()):
        declared_schema = dataset.get("arrow_schema")
        parts = dataset.get("parts", [])
        if not isinstance(parts, list) or int(dataset.get("part_count", -1)) != len(parts):
            _issue(issues, "error", "task_dataset_part_count_mismatch", "tasks", 1, dataset_key)
            continue
        digest_payload: list[dict[str, Any]] = []
        rows = 0
        parent_groups = {
            Path(clean_text(part.get("path"))).parent.as_posix() for part in parts if isinstance(part, dict)
        }
        if (
            len(parent_groups) != 1
            or not isinstance(shard_schemas, dict)
            or shard_schemas.get(next(iter(parent_groups), "")) != declared_schema
        ):
            _issue(
                issues,
                "error",
                "task_dataset_shard_schema_binding_mismatch",
                "tasks",
                1,
                dataset_key,
            )
        for part in parts:
            path = (canonical / str(part["path"])).resolve()
            try:
                path.relative_to(canonical)
            except ValueError:
                _issue(issues, "error", "task_part_escapes_root", "tasks", 1, dataset_key)
                continue
            listed.add(path)
            if not path.is_file():
                _issue(issues, "error", "missing_task_part", "tasks", 1, str(path))
                continue
            actual_sha = sha256_file(path)
            actual_rows = _parquet_row_count(path)
            if actual_sha != part.get("sha256") or actual_rows != int(part.get("rows", -1)):
                _issue(issues, "error", "task_part_manifest_drift", "tasks", 1, str(path))
            _verify_declared_arrow_schema(
                path,
                declared_schema,
                part.get("arrow_schema_sha256"),
                table=dataset_key,
                issues=issues,
            )
            digest_payload.append(
                {
                    "path": part["path"],
                    "rows": int(part["rows"]),
                    "sha256": part["sha256"],
                    "arrow_schema_sha256": part.get("arrow_schema_sha256"),
                }
            )
            rows += actual_rows
        digest = hashlib_sha256(canonical_json(digest_payload))
        if digest != dataset.get("dataset_sha256") or rows != int(dataset.get("row_count", -1)):
            _issue(issues, "error", "task_dataset_digest_or_count_mismatch", "tasks", 1, dataset_key)
    actual = {
        path.resolve()
        for scope in (canonical / "tasks" / "default", canonical / "tasks" / "derived_sensitivity")
        if scope.exists()
        for path in scope.rglob("*.parquet")
    }
    if listed != actual:
        _issue(
            issues,
            "error",
            "task_dataset_listed_physical_mismatch",
            "tasks",
            len(listed.symmetric_difference(actual)),
            "Manifest and physical multipart task inventories differ",
        )
    aggregate = hashlib_sha256(canonical_json(datasets))
    if aggregate != manifest.get("task_datasets_manifest_sha256"):
        _issue(issues, "error", "task_datasets_aggregate_digest_mismatch", "tasks", 1, "build_manifest")


def _verify_partitioned_dataset(
    canonical: Path,
    dataset_name: str,
    record: dict[str, Any],
    issues: list[dict[str, Any]],
) -> list[Path]:
    listed: set[Path] = set()
    digest_payload: list[dict[str, Any]] = []
    rows = 0
    parts = record.get("parts", [])
    if not isinstance(parts, list) or int(record.get("part_count", -1)) != len(parts):
        _issue(issues, "error", "dataset_part_count_mismatch", dataset_name, 1, "build_manifest")
        return []
    declared_schema = record.get("arrow_schema")
    for part in parts:
        path = (canonical / str(part["relative_path"])).resolve()
        try:
            path.relative_to(canonical)
        except ValueError:
            _issue(issues, "error", "dataset_part_escapes_root", dataset_name, 1, str(path))
            continue
        listed.add(path)
        if not path.is_file():
            _issue(issues, "error", "missing_dataset_part", dataset_name, 1, str(path))
            continue
        actual_rows = _parquet_row_count(path)
        if actual_rows != int(part.get("rows", -1)) or sha256_file(path) != part.get("sha256"):
            _issue(issues, "error", "dataset_part_manifest_drift", dataset_name, 1, str(path))
        _verify_declared_arrow_schema(
            path,
            declared_schema,
            part.get("arrow_schema_sha256"),
            table=dataset_name,
            issues=issues,
        )
        rows += actual_rows
        digest_payload.append(
            {
                "path": part["relative_path"],
                "rows": int(part["rows"]),
                "sha256": part["sha256"],
                "arrow_schema_sha256": part.get("arrow_schema_sha256"),
            }
        )
    actual = {path.resolve() for path in (canonical / dataset_name).glob("*.parquet")}
    if listed != actual:
        _issue(
            issues,
            "error",
            "dataset_listed_physical_mismatch",
            dataset_name,
            len(listed.symmetric_difference(actual)),
            "Manifest and physical part inventories differ",
        )
    if rows != int(record.get("rows", record.get("linked_rows", -1))) or hashlib_sha256(
        canonical_json(digest_payload)
    ) != record.get("dataset_sha256"):
        _issue(issues, "error", "dataset_digest_or_count_mismatch", dataset_name, 1, "build_manifest")
    return sorted(listed)


def _verify_full_build_artifacts(
    canonical: Path,
    manifest: dict[str, Any],
    issues: list[dict[str, Any]],
) -> None:
    if "full_specialized" not in clean_text(manifest.get("build_type", "")):
        return
    listed: set[Path] = set()
    shard_records = manifest.get("shard_artifacts", [])
    declared_groups = manifest.get("shard_dataset_schemas")
    if not isinstance(shard_records, list) or not isinstance(declared_groups, dict):
        _issue(
            issues,
            "error",
            "missing_shard_arrow_schema_contracts",
            "build_manifest",
            1,
            "shard_artifacts/shard_dataset_schemas",
        )
        return
    actual_groups = {
        Path(clean_text(record.get("relative_path"))).parent.as_posix()
        for record in shard_records
        if isinstance(record, dict)
    }
    if set(declared_groups) != actual_groups:
        _issue(
            issues,
            "error",
            "shard_arrow_schema_group_mismatch",
            "build_manifest",
            max(1, len(set(declared_groups).symmetric_difference(actual_groups))),
            "Declared schema groups must exactly match shard parent directories",
        )
    for record in shard_records:
        path = (canonical / str(record.get("relative_path", ""))).resolve()
        try:
            path.relative_to(canonical)
        except ValueError:
            _issue(issues, "error", "shard_artifact_escapes_root", "build_manifest", 1, str(path))
            continue
        if path in listed:
            _issue(issues, "error", "duplicate_shard_artifact_manifest_path", "build_manifest", 1, str(path))
            continue
        listed.add(path)
        if not path.is_file():
            _issue(issues, "error", "missing_shard_artifact", "build_manifest", 1, str(path))
            continue
        rows = _parquet_row_count(path)
        if rows != int(record.get("rows", -1)) or sha256_file(path) != record.get("sha256"):
            _issue(issues, "error", "shard_artifact_manifest_drift", "build_manifest", 1, str(path))
        group = Path(clean_text(record.get("relative_path"))).parent.as_posix()
        _verify_declared_arrow_schema(
            path,
            declared_groups.get(group),
            record.get("arrow_schema_sha256"),
            table=group,
            issues=issues,
        )
    actual: set[Path] = set()
    for relative in (
        "observations",
        "observation_lineage",
        "views/binding_free_energy_standard",
        "derived_observations",
        "tasks/default",
        "tasks/derived_sensitivity",
        "task_exclusions",
    ):
        directory = canonical / relative
        if directory.exists():
            actual.update(path.resolve() for path in directory.rglob("*.parquet"))
    if listed != actual:
        _issue(
            issues,
            "error",
            "shard_artifact_listed_physical_mismatch",
            "build_manifest",
            len(listed.symmetric_difference(actual)),
            "Unlisted or missing observation/lineage/derivation/task shard",
        )
    entity_artifacts = manifest.get("entity_artifacts", {})
    expected_root_artifacts = {"sources", "source_files", "task_registry"}
    if not isinstance(entity_artifacts, dict):
        _issue(issues, "error", "malformed_entity_artifact_manifest", "build_manifest", 1, "entity_artifacts")
        entity_artifacts = {}
    artifact_names = set(entity_artifacts)
    if artifact_names != expected_root_artifacts:
        _issue(
            issues,
            "error",
            "root_entity_artifact_name_set_mismatch",
            "build_manifest",
            max(1, len(artifact_names.symmetric_difference(expected_root_artifacts))),
            "Full build requires exactly sources, source_files, and task_registry root artifacts",
        )
    listed_root = {(canonical / f"{name}.parquet").resolve() for name in artifact_names}
    physical_root = {path.resolve() for path in canonical.glob("*.parquet")}
    if listed_root != physical_root:
        _issue(
            issues,
            "error",
            "root_parquet_listed_physical_mismatch",
            "build_manifest",
            max(1, len(listed_root.symmetric_difference(physical_root))),
            "Root-level Parquet inventory must exactly match entity_artifacts",
        )
    for name, record in entity_artifacts.items():
        if not isinstance(record, dict):
            _issue(issues, "error", "malformed_entity_artifact_record", name, 1, "build_manifest")
            continue
        path = canonical / f"{name}.parquet"
        if not path.is_file():
            _issue(issues, "error", "missing_entity_artifact", name, 1, str(path))
            continue
        rows = _parquet_row_count(path)
        if (
            record.get("path") != path.name
            or record.get("relative_path") != path.name
            or rows != int(record.get("rows", -1))
            or path.stat().st_size != int(record.get("size_bytes", -1))
            or sha256_file(path) != record.get("sha256")
        ):
            _issue(issues, "error", "entity_artifact_manifest_drift", name, 1, str(path))
        _verify_declared_arrow_schema(
            path,
            record.get("arrow_schema"),
            record.get("arrow_schema_sha256"),
            table=name,
            issues=issues,
        )


def _verify_component_inventory(
    canonical: Path,
    manifest: dict[str, Any],
    issues: list[dict[str, Any]],
) -> None:
    """Recompute the complete non-manifest file inventory for a full build."""

    if "full_specialized" not in clean_text(manifest.get("build_type", "")):
        return
    expected = manifest.get("component_inventory")
    if not isinstance(expected, list):
        _issue(
            issues,
            "error",
            "missing_or_malformed_component_inventory",
            "build_manifest",
            1,
            "Full build requires a list-valued component_inventory",
        )
        return
    from .platform_data_bulk_canonical import _component_inventory

    try:
        actual = _component_inventory(canonical)
    except (OSError, RuntimeError, ValueError) as error:
        _issue(
            issues,
            "error",
            "component_inventory_recompute_failure",
            "build_manifest",
            1,
            str(error),
        )
        return
    expected_by_path = {
        clean_text(record.get("path")): record
        for record in expected
        if isinstance(record, dict) and clean_text(record.get("path"))
    }
    actual_by_path = {clean_text(record["path"]): record for record in actual}
    if len(expected_by_path) != len(expected) or set(expected_by_path) != set(actual_by_path):
        _issue(
            issues,
            "error",
            "component_inventory_membership_mismatch",
            "build_manifest",
            max(1, len(set(expected_by_path).symmetric_difference(actual_by_path))),
            "Every committed file except build_manifest.json must be listed exactly once",
        )
    drift = sum(
        int(expected_by_path[path] != actual_by_path[path])
        for path in set(expected_by_path).intersection(actual_by_path)
    )
    _issue(
        issues,
        "error",
        "component_inventory_record_mismatch",
        "build_manifest",
        drift,
        "Component size/hash/row metadata drift",
    )
    acceptance = manifest.get("qc_acceptance")
    expected_acceptance = {
        "required_before_promotion": True,
        "report_path": "qc_report.json",
        "binding": "qc_report.build_manifest_sha256 == SHA256(build_manifest.json)",
    }
    if acceptance != expected_acceptance:
        _issue(
            issues,
            "error",
            "qc_acceptance_contract_mismatch",
            "build_manifest",
            1,
            "Full build must require manifest-bound QC before promotion",
        )


def _verify_model_readiness_accounting(
    canonical: Path,
    manifest: dict[str, Any],
    issues: list[dict[str, Any]],
    connection: sqlite3.Connection,
    *,
    task_counts: dict[str, int],
    task_dimension_counts: dict[str, dict[str, Counter[str]]],
) -> dict[str, Any]:
    """Reconcile candidate -> admitted/excluded task-input readiness exactly."""

    if "full_specialized" not in clean_text(manifest.get("build_type", "")):
        return {}
    policy = manifest.get("model_readiness_policy")
    datasets = manifest.get("model_readiness_exclusion_datasets")
    if not isinstance(policy, dict) or not isinstance(datasets, dict):
        _issue(
            issues,
            "error",
            "missing_model_readiness_accounting",
            "task_exclusions",
            1,
            "build_manifest",
        )
        return {}
    scopes = ("default", "derived_sensitivity")
    reason_order = (
        "missing_standardized_smiles",
        "missing_standard_inchi_key",
        "missing_protein_sequence",
    )
    allowed_declarations = {
        "small_molecule_structure",
        "small_molecule_structure;protein_sequence",
    }
    stage_counts: dict[str, dict[str, int]] = {}
    reason_counts: dict[str, dict[str, int]] = {}
    combination_counts: dict[str, dict[str, int]] = {}
    dimension_counts: dict[str, Any] = {}
    expected_dataset_scopes: set[str] = set()
    for scope in scopes:
        declared_stage = policy.get("stage_counts", {}).get(scope, {})
        try:
            declared_excluded = int(declared_stage.get("excluded", -1))
        except (AttributeError, TypeError, ValueError):
            declared_excluded = -1
        record = datasets.get(scope)
        if declared_excluded > 0:
            expected_dataset_scopes.add(scope)
            if not isinstance(record, dict):
                _issue(
                    issues,
                    "error",
                    "missing_model_readiness_exclusion_dataset",
                    "task_exclusions",
                    1,
                    scope,
                )
                paths: list[Path] = []
            else:
                paths = _verify_partitioned_dataset(
                    canonical,
                    f"task_exclusions/{scope}",
                    record,
                    issues,
                )
        else:
            paths = []
            physical = canonical / "task_exclusions" / scope
            if isinstance(record, dict) or any(physical.glob("*.parquet")):
                _issue(
                    issues,
                    "error",
                    "unexpected_model_readiness_exclusion_dataset",
                    "task_exclusions",
                    1,
                    scope,
                )
        scope_reasons: Counter[str] = Counter()
        scope_combinations: Counter[str] = Counter()
        excluded_dimensions = {
            dimension: Counter[str]()
            for dimension in (
                "task_type",
                "source_id",
                "protein_id",
                "canonical_target_id",
            )
        }
        excluded_rows = 0
        for path in paths:
            frame = pd.read_parquet(path)
            excluded_rows += len(frame)
            required_columns = {
                "observation_id",
                "task_id",
                "task_type",
                "task_scope",
                "source_id",
                "snapshot_id",
                "source_record_id",
                "molecule_id",
                "protein_id",
                "assay_id",
                "canonical_target_id",
                "required_modalities",
                "model_readiness_exclusion_reason",
                *reason_order,
            }
            missing_columns = sorted(required_columns - set(frame.columns))
            if missing_columns:
                _issue(
                    issues,
                    "error",
                    "model_readiness_exclusion_schema_missing",
                    "task_exclusions",
                    len(frame) or 1,
                    f"{path}: {missing_columns}",
                )
                continue
            invalid_scope = frame["task_scope"].map(clean_text).ne(scope)
            invalid_declaration = ~frame["required_modalities"].map(clean_text).isin(allowed_declarations)
            invalid_reason_rows = 0
            sql_records: list[tuple[Any, ...]] = []
            for row in frame.to_dict("records"):
                flags = {reason: bool(row[reason]) for reason in reason_order}
                expected_reason = ";".join(reason for reason in reason_order if flags[reason])
                actual_reason = clean_text(row["model_readiness_exclusion_reason"])
                if not expected_reason or actual_reason != expected_reason:
                    invalid_reason_rows += 1
                scope_combinations[actual_reason] += 1
                for reason in actual_reason.split(";"):
                    if reason:
                        scope_reasons[reason] += 1
                for dimension in excluded_dimensions:
                    excluded_dimensions[dimension][clean_text(row[dimension])] += 1
                sql_records.append(
                    (
                        clean_text(row["task_id"]),
                        clean_text(row["observation_id"]),
                        scope,
                        clean_text(row["task_type"]),
                        clean_text(row["source_id"]),
                        clean_text(row["snapshot_id"]),
                        clean_text(row["source_record_id"]),
                        clean_text(row["molecule_id"]),
                        clean_text(row["protein_id"]),
                        clean_text(row["assay_id"]),
                        clean_text(row["canonical_target_id"]),
                        clean_text(row["required_modalities"]),
                        actual_reason,
                        int(flags["missing_standardized_smiles"]),
                        int(flags["missing_standard_inchi_key"]),
                        int(flags["missing_protein_sequence"]),
                    )
                )
            _issue(
                issues,
                "error",
                "model_readiness_exclusion_row_contract_violation",
                "task_exclusions",
                int(invalid_scope.sum()) + int(invalid_declaration.sum()) + invalid_reason_rows,
                str(path),
            )
            duplicates = _insert_unique(
                connection,
                "INSERT OR IGNORE INTO task_input_exclusions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                sql_records,
            )
            _issue(
                issues,
                "error",
                "duplicate_task_input_exclusion",
                "task_exclusions",
                duplicates,
                str(path),
            )
        stage_counts[scope] = {
            "candidate": int(task_counts.get(scope, 0)) + excluded_rows,
            "eligible": int(task_counts.get(scope, 0)),
            "excluded": excluded_rows,
        }
        reason_counts[scope] = {reason: int(count) for reason, count in sorted(scope_reasons.items())}
        combination_counts[scope] = {
            reason: int(count) for reason, count in sorted(scope_combinations.items())
        }
        dimension_counts[scope] = {}
        for dimension, eligible_counts in task_dimension_counts[scope].items():
            keys = sorted(set(eligible_counts) | set(excluded_dimensions[dimension]))
            dimension_counts[scope][dimension] = {
                key: {
                    "candidate": int(eligible_counts[key]) + int(excluded_dimensions[dimension][key]),
                    "eligible": int(eligible_counts[key]),
                    "excluded": int(excluded_dimensions[dimension][key]),
                }
                for key in keys
            }
    if set(datasets) != expected_dataset_scopes:
        _issue(
            issues,
            "error",
            "model_readiness_exclusion_dataset_scope_mismatch",
            "task_exclusions",
            max(1, len(set(datasets).symmetric_difference(expected_dataset_scopes))),
            "build_manifest",
        )
    expected_static = {
        "policy_version": "platform-model-readiness-v1",
        "allowed_modality_declarations": sorted(allowed_declarations),
        "reason_order": list(reason_order),
        "exclusion_artifact_root": "task_exclusions",
        "evidence_layer_policy": (
            "source observations and lineage remain unchanged; only model-task admission is gated"
        ),
    }
    for key, expected in expected_static.items():
        if policy.get(key) != expected:
            _issue(
                issues,
                "error",
                "model_readiness_policy_contract_mismatch",
                "task_exclusions",
                1,
                key,
            )
    for key, actual in (
        ("stage_counts", stage_counts),
        ("reason_counts", reason_counts),
        ("reason_combination_counts", combination_counts),
        ("dimension_counts", dimension_counts),
    ):
        if policy.get(key) != actual:
            _issue(
                issues,
                "error",
                "model_readiness_accounting_mismatch",
                "task_exclusions",
                1,
                key,
            )
    _issue(
        issues,
        "error",
        "task_input_exclusion_observation_or_entity_mismatch",
        "task_exclusions",
        _sql_count(
            connection,
            """
            SELECT COUNT(*) FROM task_input_exclusions e
            LEFT JOIN observations o USING(observation_id)
            LEFT JOIN molecules m ON e.molecule_id=m.molecule_id
            LEFT JOIN proteins p ON e.protein_id=p.protein_id
            WHERE o.observation_id IS NULL OR m.molecule_id IS NULL OR p.protein_id IS NULL
               OR e.source_id<>o.source_id OR e.snapshot_id<>o.snapshot_id
               OR e.source_record_id<>o.source_record_id OR e.molecule_id<>o.molecule_id
               OR e.protein_id<>o.protein_id OR e.assay_id<>o.assay_id
               OR e.canonical_target_id<>p.canonical_target_id
            """,
        ),
        "Exact source-observation/entity reconciliation",
    )
    _issue(
        issues,
        "error",
        "task_input_exclusion_reason_entity_mismatch",
        "task_exclusions",
        _sql_count(
            connection,
            """
            SELECT COUNT(*) FROM task_input_exclusions e
            JOIN molecules m USING(molecule_id)
            JOIN proteins p USING(protein_id)
            WHERE e.missing_standardized_smiles <>
                  (instr(e.required_modalities,'small_molecule_structure')>0 AND m.has_standardized_smiles=0)
               OR e.missing_standard_inchi_key <>
                  (instr(e.required_modalities,'small_molecule_structure')>0 AND m.has_standard_inchi_key=0)
               OR e.missing_protein_sequence <>
                  (instr(e.required_modalities,'protein_sequence')>0 AND p.has_sequence=0)
            """,
        ),
        "Exclusion flags must equal the bound entity input state",
    )
    _issue(
        issues,
        "error",
        "task_input_exclusion_without_preserved_lineage",
        "task_exclusions",
        _sql_count(
            connection,
            """
            SELECT COUNT(*) FROM task_input_exclusions e
            WHERE NOT EXISTS (
                SELECT 1 FROM lineage l WHERE l.observation_id=e.observation_id
            )
            """,
        ),
        "Excluded task candidates must retain evidence-layer lineage",
    )
    _issue(
        issues,
        "error",
        "excluded_task_leaked_into_model_dataset",
        "task_exclusions",
        _sql_count(
            connection,
            """
            SELECT COUNT(*) FROM task_input_exclusions e
            JOIN tasks t ON e.task_id=t.task_id AND e.observation_id=t.observation_id
                        AND e.task_scope=t.task_scope
            """,
        ),
        "Excluded candidate must not appear in sequence-dependent task artifacts",
    )
    return {
        "stage_counts": stage_counts,
        "reason_counts": reason_counts,
        "reason_combination_counts": combination_counts,
        "dimension_counts": dimension_counts,
    }


def _verify_full_build_count_conservation(
    manifest: dict[str, Any],
    issues: list[dict[str, Any]],
    *,
    observation_rows: int,
    lineage_rows: int,
    derived_observation_rows: int,
    derivation_rows: int,
    entity_counts: dict[str, int],
    default_task_rows: int,
    sensitivity_task_rows: int,
    task_registry_rows: int,
) -> None:
    """Reconcile every full-build stage counter to exact audited physical counts."""

    if "full_specialized" not in clean_text(manifest.get("build_type", "")):
        return

    def mismatch(code: str, expected: object, actual: object, detail: str) -> None:
        if expected != actual:
            _issue(
                issues,
                "error",
                code,
                "build_manifest",
                1,
                f"{detail}; manifest={expected!r}; audited={actual!r}",
            )

    def manifest_int(value: object) -> int:
        try:
            return int(value)  # type: ignore[call-overload]
        except (TypeError, ValueError):
            return -1

    unique_activity_rows = manifest_int(manifest.get("unique_activity_rows", -1))
    derived_rows = manifest_int(manifest.get("derived_binding_free_energy_rows", -1))
    mismatch(
        "observation_stage_count_nonconservation",
        unique_activity_rows + derived_rows,
        observation_rows,
        "observations must equal unique activity rows plus derived free-energy rows",
    )
    mismatch(
        "unique_activity_count_mismatch",
        unique_activity_rows,
        observation_rows - derived_observation_rows,
        "unique activity rows must equal audited non-derived observations",
    )
    mismatch(
        "derived_observation_count_mismatch",
        derived_rows,
        derived_observation_rows,
        "derived manifest counter must equal audited derived observations",
    )
    mismatch(
        "derivation_count_mismatch",
        derived_rows,
        derivation_rows,
        "derived manifest counter must equal verified derivation records",
    )
    mismatch(
        "lineage_stage_count_nonconservation",
        observation_rows,
        lineage_rows,
        "current full-build design requires exactly one lineage edge per observation",
    )

    input_summary = manifest.get("input_summary", {})
    expected_membership = input_summary.get("view_row_counts", {}) if isinstance(input_summary, dict) else {}
    actual_membership = manifest.get("inventory_membership_counts_before_cross_view_dedup", {})
    try:
        expected_membership = {clean_text(name): int(count) for name, count in expected_membership.items()}
        actual_membership = {clean_text(name): int(count) for name, count in actual_membership.items()}
    except (AttributeError, TypeError, ValueError):
        expected_membership = {"<malformed>": -1}
        actual_membership = {"<malformed-actual>": -1}
    mismatch(
        "input_inventory_membership_count_mismatch",
        expected_membership,
        actual_membership,
        "pre-dedup view membership must reproduce the bound input manifests",
    )

    manifest_entity_counts = manifest.get("entity_counts", {})
    if not isinstance(manifest_entity_counts, dict):
        manifest_entity_counts = {}
    for dataset_name, audited_rows in sorted(entity_counts.items()):
        mismatch(
            "entity_count_mismatch",
            manifest_int(manifest_entity_counts.get(dataset_name, -1)),
            int(audited_rows),
            dataset_name,
        )
    mismatch(
        "default_task_count_mismatch",
        manifest_int(manifest_entity_counts.get("tasks", -1)),
        default_task_rows,
        "default task rows",
    )
    mismatch(
        "sensitivity_task_count_mismatch",
        manifest_int(manifest_entity_counts.get("sensitivity_tasks", -1)),
        sensitivity_task_rows,
        "derived-sensitivity task rows",
    )
    mismatch(
        "task_registry_manifest_count_mismatch",
        manifest_int(manifest.get("task_registry_rows", -1)),
        task_registry_rows,
        "task registry rows",
    )


def hashlib_sha256(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _audit_binding_free_energy(
    canonical: Path,
    issues: list[dict[str, Any]],
) -> dict[str, Any]:
    from .platform_data_bulk_canonical import _validate_derivation_bundle

    derivation_parts = _parquet_parts(
        canonical / "views",
        "binding_free_energy_standard",
    )
    if not derivation_parts:
        return {"derivation_rows": 0, "max_roundtrip_relative_error": None}
    observation_parts = {path.name: path for path in _parquet_parts(canonical, "observations")}
    derived_parts = {path.name: path for path in _parquet_parts(canonical, "derived_observations")}
    derivation_rows = 0
    maximum_error = 0.0
    for derivation_path in derivation_parts:
        derivations = pd.read_parquet(derivation_path)
        derivation_rows += len(derivations)
        if "roundtrip_relative_error" in derivations.columns and not derivations.empty:
            maximum_error = max(
                maximum_error,
                float(pd.to_numeric(derivations["roundtrip_relative_error"], errors="raise").max()),
            )
        if derivation_path.name in observation_parts:
            observations = pd.read_parquet(observation_parts[derivation_path.name])
        elif len(observation_parts) == 1:
            observations = pd.read_parquet(next(iter(observation_parts.values())))
        else:
            _issue(
                issues,
                "error",
                "derivation_observation_shard_missing",
                "binding_free_energy",
                len(derivations),
                derivation_path.name,
            )
            continue
        if derivation_path.name in derived_parts:
            derived = pd.read_parquet(derived_parts[derivation_path.name])
        else:
            derived_ids = set(derivations["observation_id"].map(clean_text))
            derived = observations[observations["observation_id"].map(clean_text).isin(derived_ids)].copy()
        source_ids = set(derivations["source_observation_id"].map(clean_text))
        source = observations[observations["observation_id"].map(clean_text).isin(source_ids)].copy()
        try:
            _validate_derivation_bundle(source, derivations, derived)
        except RuntimeError as error:
            _issue(
                issues,
                "error",
                "binding_free_energy_integrity_failure",
                "binding_free_energy",
                max(len(derivations), 1),
                f"{derivation_path.name}: {error}",
            )
    if set(derived_parts) - {path.name for path in derivation_parts}:
        _issue(
            issues,
            "error",
            "derived_observation_without_derivation_shard",
            "binding_free_energy",
            len(set(derived_parts) - {path.name for path in derivation_parts}),
            "Shard inventory mismatch",
        )
    return {
        "derivation_rows": derivation_rows,
        "max_roundtrip_relative_error": maximum_error if derivation_rows else None,
    }


def _run_platform_qc_inner(
    canonical: Path,
    reports: Path,
    state_path: Path,
) -> dict[str, Any]:
    """Run exact schema/FK/rights/task QC with disk-backed global state."""

    reports.mkdir(parents=True, exist_ok=True)
    issues: list[dict[str, Any]] = []
    missing_counts: defaultdict[tuple[str, str, str], int] = defaultdict(int)
    missing_denominators: defaultdict[tuple[str, str, str], int] = defaultdict(int)
    sources = _read_small_dataset(canonical, "sources")
    source_files = _read_small_dataset(canonical, "source_files")
    task_registry_path = canonical / "task_registry.parquet"
    task_registry = pd.read_parquet(task_registry_path) if task_registry_path.exists() else pd.DataFrame()
    build_manifest_path = canonical / "build_manifest.json"
    build_manifest = (
        json.loads(build_manifest_path.read_text(encoding="utf-8")) if build_manifest_path.exists() else {}
    )
    _verify_component_inventory(canonical, build_manifest, issues)
    _accumulate_missingness(
        "task_registry",
        task_registry,
        missing_counts,
        missing_denominators,
    )
    for table, frame in (("sources", sources), ("source_files", source_files)):
        _accumulate_missingness(table, frame, missing_counts, missing_denominators)
        for finding in validate_table(table, frame):
            _issue(
                issues,
                "error",
                finding["code"],
                table,
                finding.get("n_rows", 0),
                json.dumps(finding, sort_keys=True),
            )
    _issue(
        issues,
        "error",
        "nonpublic_source_in_public_canonical",
        "sources",
        int(sources["access_class"].ne(PUBLIC_ACCESS_CLASS).sum()),
        "Only public_redistributable sources are allowed",
    )

    connection = sqlite3.connect(state_path)
    _create_qc_state(connection)
    duplicate_sources = _insert_unique(
        connection,
        "INSERT OR IGNORE INTO sources(source_id, snapshot_id) VALUES (?, ?)",
        [(clean_text(row.source_id), clean_text(row.snapshot_id)) for row in sources.itertuples()],
    )
    duplicate_files = _insert_unique(
        connection,
        "INSERT OR IGNORE INTO source_files(source_file_id, source_id, snapshot_id) VALUES (?, ?, ?)",
        [
            (clean_text(row.source_file_id), clean_text(row.source_id), clean_text(row.snapshot_id))
            for row in source_files.itertuples()
        ],
    )
    _issue(issues, "error", "duplicate_source_snapshot", "sources", duplicate_sources, "Composite key")
    _issue(issues, "error", "duplicate_source_file", "source_files", duplicate_files, "source_file_id")

    entity_counts: dict[str, int] = {}
    for table, key, insert_sql in (
        ("molecules", "molecule_id", "INSERT OR IGNORE INTO molecules VALUES (?, ?, ?, ?)"),
        ("proteins", "protein_id", "INSERT OR IGNORE INTO proteins VALUES (?, ?, ?)"),
        ("assays", "assay_id", "INSERT OR IGNORE INTO assays VALUES (?, ?)"),
    ):
        count = 0
        for part in _parquet_parts(canonical, table):
            frame = pd.read_parquet(part)
            count += len(frame)
            _accumulate_missingness(table, frame, missing_counts, missing_denominators)
            for finding in validate_table(table, frame):
                _issue(issues, "error", finding["code"], table, finding.get("n_rows", 0), part.name)
            entity_rows: list[tuple[Any, ...]]
            if table == "molecules":
                entity_rows = [
                    (
                        clean_text(row[key]),
                        int(bool(clean_text(row.get("standardized_smiles", "")))),
                        int(bool(clean_text(row.get("standard_inchi_key", "")))),
                        int(
                            bool(clean_text(row.get("standardized_smiles", "")))
                            and bool(clean_text(row.get("standard_inchi_key", "")))
                            and clean_text(row.get("identity_resolution_status", "")) == "resolved"
                        ),
                    )
                    for row in frame.to_dict("records")
                ]
            elif table == "proteins":
                entity_rows = [
                    (
                        clean_text(row[key]),
                        int(bool(clean_text(row.get("sequence", "")))),
                        clean_text(row.get("canonical_target_id", "")),
                    )
                    for row in frame.to_dict("records")
                ]
            else:
                entity_rows = [
                    (clean_text(row[key]), clean_text(row.get("assay_family", "")))
                    for row in frame.to_dict("records")
                ]
            duplicates = _insert_unique(connection, insert_sql, entity_rows)
            _issue(issues, "error", f"duplicate_{key}_across_parts", table, duplicates, part.name)
        if count == 0:
            _issue(issues, "error", f"missing_{table}_dataset", table, 1, "No Parquet parts")
        entity_counts[table] = count

    for table, insert_sql in (
        (
            "molecule_aliases",
            "INSERT OR IGNORE INTO molecule_aliases VALUES (?,?,?,?)",
        ),
        (
            "protein_constructs",
            "INSERT OR IGNORE INTO protein_constructs VALUES (?,?,?)",
        ),
    ):
        count = 0
        parts = _parquet_parts(canonical, table)
        if not parts:
            _issue(issues, "error", f"missing_{table}_dataset", table, 1, "No Parquet parts")
        for part in parts:
            frame = pd.read_parquet(part)
            count += len(frame)
            _accumulate_missingness(table, frame, missing_counts, missing_denominators)
            for finding in validate_table(table, frame):
                _issue(issues, "error", finding["code"], table, finding.get("n_rows", 0), part.name)
            relation_rows: list[tuple[Any, ...]]
            if table == "molecule_aliases":
                relation_rows = [
                    (
                        clean_text(row["molecule_alias_id"]),
                        clean_text(row["molecule_id"]),
                        clean_text(row["source_id"]),
                        clean_text(row.get("snapshot_id", "")),
                    )
                    for row in frame.to_dict("records")
                ]
            else:
                relation_rows = [
                    (
                        clean_text(row["construct_id"]),
                        clean_text(row["protein_id"]),
                        clean_text(row["source_id"]),
                    )
                    for row in frame.to_dict("records")
                ]
            duplicates = _insert_unique(connection, insert_sql, relation_rows)
            _issue(issues, "error", f"duplicate_{table}_id", table, duplicates, part.name)
        entity_counts[table] = count

    group_counts: Counter[tuple[str, str]] = Counter()
    endpoint_numeric: defaultdict[tuple[str, str, str], list[float]] = defaultdict(list)
    document_years: Counter[str] = Counter()
    attrition_counts: Counter[tuple[str, str, str]] = Counter()
    observation_rows = 0
    local_conflict_rows = 0
    observation_parts = _parquet_parts(canonical, "observations")
    if not observation_parts:
        _issue(
            issues, "error", "missing_observation_dataset", "observations", 1, "No observations Parquet found"
        )
    for part in observation_parts:
        frame = pd.read_parquet(part)
        observation_rows += len(frame)
        for finding in validate_table("observations", frame):
            _issue(issues, "error", finding["code"], "observations", finding.get("n_rows", 0), part.name)
        invalid_kind = frame["observation_kind"].map(clean_text).eq("") | frame["observation_kind"].eq(
            "prediction"
        )
        _issue(
            issues,
            "error",
            "blank_or_prediction_observation_kind",
            "observations",
            int(invalid_kind.sum()),
            part.name,
        )
        _issue(
            issues,
            "error",
            "nonpublic_observation",
            "observations",
            int(frame["access_class"].ne(PUBLIC_ACCESS_CLASS).sum()),
            part.name,
        )
        both_bounds = frame["lower_bound"].notna() & frame["upper_bound"].notna()
        reversed_bounds = both_bounds & (
            pd.to_numeric(frame["lower_bound"], errors="coerce")
            > pd.to_numeric(frame["upper_bound"], errors="coerce")
        )
        _issue(issues, "error", "reversed_bounds", "observations", int(reversed_bounds.sum()), part.name)
        _accumulate_missingness(
            "observations",
            frame,
            missing_counts,
            missing_denominators,
            stratify_by_evidence_domain=True,
        )
        for dimension in (
            "evidence_domain",
            "inclusion_status",
            "quality_grade",
            "observation_kind",
            "endpoint",
        ):
            for value, count in frame[dimension].fillna("<NA>").astype(str).value_counts().items():
                group_counts[(dimension, value)] += int(count)
        for (domain, endpoint, unit), group in frame.groupby(
            ["evidence_domain", "endpoint", "canonical_unit"], dropna=False
        ):
            key_tuple = (clean_text(domain), clean_text(endpoint), clean_text(unit))
            remaining = max(0, 200_000 - len(endpoint_numeric[key_tuple]))
            if remaining:
                values = pd.to_numeric(group["canonical_value"], errors="coerce").dropna().head(remaining)
                endpoint_numeric[key_tuple].extend(values.astype(float))
        for year, count in frame["document_year"].fillna("<NA>").astype(str).value_counts().items():
            document_years[year] += int(count)
        for _, row in frame[["evidence_domain", "inclusion_status", "exclusion_reason"]].iterrows():
            reasons = [reason for reason in clean_text(row["exclusion_reason"]).split(";") if reason] or [
                "<none>"
            ]
            for reason in reasons:
                attrition_counts[
                    (clean_text(row["evidence_domain"]), clean_text(row["inclusion_status"]), reason)
                ] += 1
        local_conflict_rows += int(frame["conflict_group_id"].map(clean_text).ne("").sum())
        observation_records = [
            (
                clean_text(row["observation_id"]),
                clean_text(row["source_id"]),
                clean_text(row["snapshot_id"]),
                clean_text(row["source_record_id"]),
                clean_text(row["molecule_id"]),
                clean_text(row["protein_id"]),
                clean_text(row["assay_id"]),
                clean_text(row["observation_kind"]),
                clean_text(row["evidence_domain"]),
                clean_text(row["endpoint"]),
                clean_text(row["inclusion_status"]),
                clean_text(row["relation"]),
                _sql_value(row.get("canonical_value")),
                clean_text(row.get("canonical_unit", "")),
                _sql_value(row.get("lower_bound")),
                _sql_value(row.get("upper_bound")),
                clean_text(row.get("dedup_group_id", "")),
                clean_text(row.get("conflict_group_id", "")),
                clean_text(row.get("document_id", "")),
                clean_text(row.get("document_year", "")),
                clean_text(row.get("activity_origin_name", "")),
            )
            for row in frame.to_dict("records")
        ]
        duplicates = _insert_unique(
            connection,
            "INSERT OR IGNORE INTO observations VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            observation_records,
        )
        _issue(issues, "error", "duplicate_observation_across_shards", "observations", duplicates, part.name)

    lineage_rows = 0
    for part in _parquet_parts(canonical, "observation_lineage"):
        frame = pd.read_parquet(part)
        lineage_rows += len(frame)
        _accumulate_missingness(
            "observation_lineage",
            frame,
            missing_counts,
            missing_denominators,
        )
        for finding in validate_table("observation_lineage", frame):
            _issue(
                issues, "error", finding["code"], "observation_lineage", finding.get("n_rows", 0), part.name
            )
        lineage_records = [
            (
                clean_text(row["lineage_id"]),
                clean_text(row["observation_id"]),
                clean_text(row["source_id"]),
                clean_text(row["snapshot_id"]),
                clean_text(row["source_file_id"]),
                clean_text(row["lineage_role"]),
            )
            for row in frame.to_dict("records")
        ]
        duplicates = _insert_unique(
            connection,
            "INSERT OR IGNORE INTO lineage VALUES (?,?,?,?,?,?)",
            lineage_records,
        )
        _issue(issues, "error", "duplicate_lineage_edge", "observation_lineage", duplicates, part.name)
    if lineage_rows == 0:
        _issue(issues, "error", "missing_lineage_dataset", "observation_lineage", 1, "No lineage parts")

    fk_queries = {
        "orphan_source_file_snapshot": "SELECT COUNT(*) FROM source_files f LEFT JOIN sources s ON f.source_id=s.source_id AND f.snapshot_id=s.snapshot_id WHERE s.source_id IS NULL",
        "orphan_observation_source_snapshot": "SELECT COUNT(*) FROM observations o LEFT JOIN sources s ON o.source_id=s.source_id AND o.snapshot_id=s.snapshot_id WHERE s.source_id IS NULL",
        "orphan_observation_molecule": "SELECT COUNT(*) FROM observations o LEFT JOIN molecules m USING(molecule_id) WHERE m.molecule_id IS NULL",
        "orphan_observation_protein": "SELECT COUNT(*) FROM observations o LEFT JOIN proteins p USING(protein_id) WHERE p.protein_id IS NULL",
        "orphan_observation_assay": "SELECT COUNT(*) FROM observations o LEFT JOIN assays a USING(assay_id) WHERE a.assay_id IS NULL",
        "orphan_molecule_alias": "SELECT COUNT(*) FROM molecule_aliases a LEFT JOIN molecules m USING(molecule_id) WHERE m.molecule_id IS NULL",
        "orphan_molecule_alias_source_snapshot": "SELECT COUNT(*) FROM molecule_aliases a LEFT JOIN sources s ON a.source_id=s.source_id AND a.snapshot_id=s.snapshot_id WHERE s.source_id IS NULL",
        "orphan_protein_construct": "SELECT COUNT(*) FROM protein_constructs c LEFT JOIN proteins p USING(protein_id) WHERE p.protein_id IS NULL",
        "orphan_protein_construct_source": "SELECT COUNT(*) FROM protein_constructs c LEFT JOIN sources s ON c.source_id=s.source_id WHERE s.source_id IS NULL",
        "orphan_lineage_file": "SELECT COUNT(*) FROM lineage l LEFT JOIN source_files f USING(source_file_id) WHERE f.source_file_id IS NULL",
        "orphan_lineage_file_source_snapshot_tuple": "SELECT COUNT(*) FROM lineage l LEFT JOIN source_files f ON l.source_file_id=f.source_file_id AND l.source_id=f.source_id AND l.snapshot_id=f.snapshot_id WHERE f.source_file_id IS NULL",
        "orphan_lineage_observation": "SELECT COUNT(*) FROM lineage l LEFT JOIN observations o USING(observation_id) WHERE o.observation_id IS NULL",
        "orphan_lineage_snapshot": "SELECT COUNT(*) FROM lineage l LEFT JOIN sources s ON l.source_id=s.source_id AND l.snapshot_id=s.snapshot_id WHERE s.source_id IS NULL",
        "observation_without_lineage": "SELECT COUNT(*) FROM observations o LEFT JOIN lineage l USING(observation_id) WHERE l.observation_id IS NULL",
        "primary_source_snapshot_mismatch": "SELECT COUNT(*) FROM observations o JOIN lineage l USING(observation_id) WHERE l.lineage_role='primary' AND (o.source_id<>l.source_id OR o.snapshot_id<>l.snapshot_id)",
        "experimental_primary_lineage_cardinality": "SELECT COUNT(*) FROM observations o LEFT JOIN (SELECT observation_id, SUM(lineage_role='primary') n FROM lineage GROUP BY observation_id) l USING(observation_id) WHERE o.observation_kind<>'derived' AND COALESCE(l.n,0)<>1",
        "derived_support_lineage_missing": "SELECT COUNT(*) FROM observations o LEFT JOIN (SELECT observation_id, SUM(lineage_role='derived_support') n FROM lineage GROUP BY observation_id) l USING(observation_id) WHERE o.observation_kind='derived' AND COALESCE(l.n,0)<1",
    }
    for code, query in fk_queries.items():
        _issue(issues, "error", code, "global_contract", _sql_count(connection, query), "Exact SQLite audit")

    default_parts, sensitivity_parts = _task_paths(canonical)
    if not default_parts:
        _issue(issues, "error", "missing_default_task_dataset", "tasks", 1, "No default tasks")
    task_counts = {"default": 0, "derived_sensitivity": 0}
    task_dimension_counts = {
        scope: {
            dimension: Counter[str]()
            for dimension in (
                "task_type",
                "source_id",
                "protein_id",
                "canonical_target_id",
            )
        }
        for scope in ("default", "derived_sensitivity")
    }
    for scope, paths in (("default", default_parts), ("derived_sensitivity", sensitivity_parts)):
        for part in paths:
            frame = pd.read_parquet(part)
            task_counts[scope] += len(frame)
            for dimension in (
                "task_type",
                "source_id",
                "protein_id",
                "canonical_target_id",
            ):
                if dimension in frame.columns:
                    task_dimension_counts[scope][dimension].update(frame[dimension].map(clean_text))
                elif "full_specialized" in clean_text(build_manifest.get("build_type", "")):
                    _issue(
                        issues,
                        "error",
                        "task_readiness_dimension_missing",
                        "tasks",
                        len(frame) or 1,
                        f"{part}: {dimension}",
                    )
            _accumulate_missingness(
                "tasks",
                frame,
                missing_counts,
                missing_denominators,
                stratify_by_evidence_domain=True,
            )
            for finding in validate_table("tasks", frame):
                _issue(issues, "error", finding["code"], "tasks", finding.get("n_rows", 0), str(part))
            bool_default = frame["default_task_eligible"].map(
                lambda value: isinstance(value, (bool, np.bool_))
            )
            _issue(
                issues,
                "error",
                "invalid_default_task_boolean",
                "tasks",
                int((~bool_default).sum()),
                part.name,
            )
            sensitivity_values = frame.get("sensitivity_task_eligible", pd.Series(False, index=frame.index))
            bool_sensitivity = sensitivity_values.map(lambda value: isinstance(value, (bool, np.bool_)))
            _issue(
                issues,
                "error",
                "invalid_sensitivity_task_boolean",
                "tasks",
                int((~bool_sensitivity).sum()),
                part.name,
            )
            label_kind = frame["label_kind"].map(clean_text)
            label_relation = frame["label_relation"].map(clean_text)
            label_value_numeric = pd.to_numeric(frame["label_value"], errors="coerce")
            lower_numeric = pd.to_numeric(frame["label_lower_bound"], errors="coerce")
            upper_numeric = pd.to_numeric(frame["label_upper_bound"], errors="coerce")
            finite_value = pd.Series(np.isfinite(label_value_numeric), index=frame.index)
            finite_lower = pd.Series(np.isfinite(lower_numeric), index=frame.index)
            finite_upper = pd.Series(np.isfinite(upper_numeric), index=frame.index)
            exact_rows = label_kind.eq("continuous_exact")
            valid_exact = (
                label_relation.eq("=")
                & finite_value
                & finite_lower
                & finite_upper
                & lower_numeric.eq(label_value_numeric)
                & upper_numeric.eq(label_value_numeric)
            )
            _issue(
                issues,
                "error",
                "continuous_exact_cross_column_violation",
                "tasks",
                int((exact_rows & ~valid_exact).sum()),
                part.name,
            )
            censored_rows = label_kind.eq("continuous_censored")
            valid_censored = label_value_numeric.isna() & (
                (label_relation.isin({"<", "<="}) & ~finite_lower & finite_upper)
                | (label_relation.isin({">", ">="}) & finite_lower & ~finite_upper)
                | (
                    label_relation.eq("interval")
                    & finite_lower
                    & finite_upper
                    & lower_numeric.le(upper_numeric)
                )
            )
            _issue(
                issues,
                "error",
                "continuous_censored_cross_column_violation",
                "tasks",
                int((censored_rows & ~valid_censored).sum()),
                part.name,
            )
            categorical_rows = label_kind.eq("categorical")
            threshold_source = pd.to_numeric(
                frame.get("threshold_source_value_nM", pd.Series(np.nan, index=frame.index)),
                errors="coerce",
            )
            threshold_low = pd.to_numeric(
                frame.get("threshold_low_nM", pd.Series(np.nan, index=frame.index)),
                errors="coerce",
            )
            threshold_high = pd.to_numeric(
                frame.get("threshold_high_nM", pd.Series(np.nan, index=frame.index)),
                errors="coerce",
            )
            endpoint_key = frame["endpoint"].map(
                lambda value: re.sub(r"[^a-z0-9]+", "", clean_text(value).casefold())
            )
            label_text = frame.get("label_text", pd.Series("", index=frame.index)).map(clean_text)
            threshold_policy = frame.get(
                "threshold_policy",
                pd.Series("", index=frame.index),
            ).map(clean_text)
            valid_binary_label = (
                threshold_source.le(10_000.0) & label_value_numeric.eq(1.0) & label_text.eq("blocker")
            ) | (threshold_source.ge(30_000.0) & label_value_numeric.eq(0.0) & label_text.eq("nonblocker"))
            valid_categorical = (
                frame["evidence_domain"].eq("herg")
                & endpoint_key.eq("ic50")
                & frame["assay_family"].eq("herg_functional")
                & label_relation.eq("=")
                & frame["label_unit"].eq("class")
                & threshold_low.eq(10_000.0)
                & threshold_high.eq(30_000.0)
                & pd.Series(np.isfinite(threshold_source), index=frame.index)
                & ~finite_lower
                & ~finite_upper
                & threshold_policy.ne("")
                & valid_binary_label
            )
            _issue(
                issues,
                "error",
                "categorical_herg_threshold_provenance_violation",
                "tasks",
                int((categorical_rows & ~valid_categorical).sum()),
                part.name,
            )
            declarations = frame.get(
                "required_modalities",
                pd.Series("", index=frame.index),
            ).map(clean_text)
            allowed_declarations = {
                "small_molecule_structure",
                "small_molecule_structure;protein_sequence",
            }
            declaration_valid = declarations.isin(allowed_declarations)
            model_ready = declaration_valid.copy()
            for column in ("standardized_smiles", "standard_inchi_key"):
                present = frame.get(column, pd.Series("", index=frame.index)).map(clean_text).ne("")
                requires = declarations.map(
                    lambda declaration: "small_molecule_structure" in declaration.split(";")
                )
                model_ready &= ~requires | present
            sequence_present = frame.get("sequence", pd.Series("", index=frame.index)).map(clean_text).ne("")
            requires_sequence = declarations.map(
                lambda declaration: "protein_sequence" in declaration.split(";")
            )
            model_ready &= ~requires_sequence | sequence_present
            _issue(
                issues,
                "error",
                "task_required_modality_or_input_violation",
                "tasks",
                int((~model_ready).sum()),
                part.name,
            )
            if scope == "default":
                base = (
                    frame["access_class"].eq(PUBLIC_ACCESS_CLASS)
                    & frame["inclusion_status"].eq("included")
                    & frame["default_task_eligible"].map(bool)
                    & ~frame["observation_kind"].isin({"derived", "prediction"})
                    & model_ready
                )
                endpoint = endpoint_key
                continuous = (
                    frame["label_kind"].isin({"continuous_exact", "continuous_censored"})
                    & frame["label_unit"].eq("nM")
                    & endpoint.isin({"kd", "ki", "ic50", "ec50"})
                    & ~frame["evidence_domain"].isin({"pk_adme", "qt"})
                    & (
                        ~frame["evidence_domain"].eq("herg")
                        | (endpoint.eq("ic50") & frame["assay_family"].eq("herg_functional"))
                    )
                )
                binary_label = (
                    threshold_source.le(10_000.0) & label_value_numeric.eq(1.0) & label_text.eq("blocker")
                ) | (
                    threshold_source.ge(30_000.0) & label_value_numeric.eq(0.0) & label_text.eq("nonblocker")
                )
                binary = (
                    frame["label_kind"].eq("categorical")
                    & frame["label_unit"].eq("class")
                    & frame["label_relation"].eq("=")
                    & frame["evidence_domain"].eq("herg")
                    & endpoint.eq("ic50")
                    & frame["assay_family"].eq("herg_functional")
                    & threshold_low.eq(10_000.0)
                    & threshold_high.eq(30_000.0)
                    & binary_label
                )
                invalid = ~(base & (continuous | binary))
                _issue(
                    issues, "error", "default_task_policy_violation", "tasks", int(invalid.sum()), part.name
                )
            else:
                digest = frame.get("label_lineage_digest", pd.Series("", index=frame.index)).map(clean_text)
                valid = (
                    frame["access_class"].eq(PUBLIC_ACCESS_CLASS)
                    & ~frame["default_task_eligible"].map(bool)
                    & sensitivity_values.map(bool)
                    & frame["observation_kind"].eq("derived")
                    & frame["endpoint"].eq("standard_binding_free_energy")
                    & frame["label_unit"].eq("kcal/mol")
                    & digest.str.fullmatch(r"[0-9a-f]{64}", na=False)
                    & model_ready
                )
                _issue(
                    issues,
                    "error",
                    "sensitivity_task_policy_violation",
                    "tasks",
                    int((~valid).sum()),
                    part.name,
                )
            signature_columns = [
                "task_id",
                "task_type",
                "evidence_domain",
                "endpoint",
                "assay_family",
                "label_kind",
                "label_unit",
                "observation_kind",
                "default_task_eligible",
                "sensitivity_task_eligible",
                "required_modalities",
            ]
            if len(frame[signature_columns].drop_duplicates()) != 1:
                _issue(issues, "error", "heterogeneous_task_shard", "tasks", len(frame), str(part))
            task_records = [
                (
                    clean_text(row["task_id"]),
                    clean_text(row["observation_id"]),
                    scope,
                    clean_text(row["task_type"]),
                    clean_text(row["source_id"]),
                    clean_text(row["snapshot_id"]),
                    clean_text(row["source_record_id"]),
                    clean_text(row["molecule_id"]),
                    clean_text(row["protein_id"]),
                    clean_text(row["assay_id"]),
                    clean_text(row["evidence_domain"]),
                    clean_text(row["endpoint"]),
                    clean_text(row["assay_family"]),
                    clean_text(row["label_kind"]),
                    clean_text(row["label_unit"]),
                    clean_text(row["observation_kind"]),
                    int(bool(row["default_task_eligible"])),
                    int(bool(row.get("sensitivity_task_eligible", False))),
                    clean_text(row["required_modalities"]),
                    clean_text(row["label_relation"]),
                    _sql_value(row.get("label_value")),
                    clean_text(row.get("label_text", "")),
                    _sql_value(row.get("label_lower_bound")),
                    _sql_value(row.get("label_upper_bound")),
                    _sql_value(row.get("threshold_low_nM")),
                    _sql_value(row.get("threshold_high_nM")),
                    _sql_value(row.get("threshold_source_value_nM")),
                )
                for row in frame.to_dict("records")
            ]
            duplicates = _insert_unique(
                connection,
                "INSERT OR IGNORE INTO tasks VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                task_records,
            )
            _issue(issues, "error", "duplicate_task_observation_across_parts", "tasks", duplicates, str(part))
    _issue(
        issues,
        "error",
        "orphan_task_observation",
        "tasks",
        _sql_count(
            connection,
            "SELECT COUNT(*) FROM tasks t LEFT JOIN observations o USING(observation_id) WHERE o.observation_id IS NULL",
        ),
        "Exact SQLite audit",
    )
    _issue(
        issues,
        "error",
        "task_observation_identity_or_provenance_mismatch",
        "tasks",
        _sql_count(
            connection,
            """
            SELECT COUNT(*) FROM tasks t JOIN observations o USING(observation_id)
            WHERE t.source_id<>o.source_id OR t.snapshot_id<>o.snapshot_id
               OR t.source_record_id<>o.source_record_id OR t.molecule_id<>o.molecule_id
               OR t.protein_id<>o.protein_id OR t.assay_id<>o.assay_id
               OR t.evidence_domain<>o.evidence_domain OR t.endpoint<>o.endpoint
               OR t.observation_kind<>o.observation_kind
            """,
        ),
        "Task identity/provenance must equal its canonical observation",
    )
    _issue(
        issues,
        "error",
        "continuous_task_label_mismatch",
        "tasks",
        _sql_count(
            connection,
            """
            SELECT COUNT(*) FROM tasks t JOIN observations o USING(observation_id)
            WHERE t.label_kind IN ('continuous_exact','continuous_censored')
              AND (t.label_relation<>o.relation OR t.label_unit<>o.canonical_unit
                   OR t.label_lower_bound IS NOT o.lower_bound
                   OR t.label_upper_bound IS NOT o.upper_bound
                   OR (t.label_kind='continuous_exact' AND t.label_value IS NOT o.canonical_value)
                   OR (t.label_kind='continuous_censored' AND t.label_value IS NOT NULL))
            """,
        ),
        "Continuous task labels must be an exact projection of canonical bounds/value",
    )
    _issue(
        issues,
        "error",
        "herg_binary_task_derivation_mismatch",
        "tasks",
        _sql_count(
            connection,
            """
            SELECT COUNT(*) FROM tasks t JOIN observations o USING(observation_id)
            WHERE t.label_kind='categorical'
              AND NOT (
                o.evidence_domain='herg' AND lower(o.endpoint)='ic50'
                AND o.relation='=' AND o.canonical_unit='nM'
                AND t.label_relation='=' AND t.label_unit='class'
                AND t.threshold_low_nM=10000.0 AND t.threshold_high_nM=30000.0
                AND t.threshold_source_value_nM IS o.canonical_value
                AND ((o.canonical_value<=10000.0 AND t.label_value=1.0 AND t.label_text='blocker')
                     OR (o.canonical_value>=30000.0 AND t.label_value=0.0 AND t.label_text='nonblocker'))
              )
            """,
        ),
        "Binary hERG labels must derive from exact frozen-threshold IC50 values",
    )

    registry_ids = set(task_registry["task_id"].map(clean_text)) if not task_registry.empty else set()
    actual_ids = {clean_text(row[0]) for row in connection.execute("SELECT DISTINCT task_id FROM tasks")}
    _issue(
        issues,
        "error",
        "task_registry_id_set_mismatch",
        "task_registry",
        len(registry_ids.symmetric_difference(actual_ids)),
        "Exact task ID set",
    )
    registry_lookup = (
        {clean_text(row["task_id"]): row for row in task_registry.to_dict("records")}
        if not task_registry.empty
        else {}
    )
    for task_id in sorted(actual_ids):
        registry_row = registry_lookup.get(task_id)
        if registry_row is None:
            continue
        rows = connection.execute(
            """
            SELECT task_type,evidence_domain,endpoint,assay_family,label_kind,label_unit,
                   observation_kind,default_task_eligible,sensitivity_task_eligible,
                   required_modalities,label_relation
            FROM tasks WHERE task_id=?
            """,
            (task_id,),
        ).fetchall()
        task_signatures = {tuple(row[:-1]) for row in rows}
        if len(task_signatures) != 1:
            _issue(issues, "error", "task_signature_not_unique", "task_registry", len(rows), task_id)
            continue
        task_signature = next(iter(task_signatures))
        columns = (
            "task_type",
            "evidence_domain",
            "endpoint",
            "assay_family",
            "label_kind",
            "label_unit",
            "observation_kind",
            "default_task_eligible",
            "sensitivity_task_eligible",
            "required_modalities",
        )
        expected = tuple(
            int(bool(registry_row[column]))
            if column in {"default_task_eligible", "sensitivity_task_eligible"}
            else clean_text(registry_row[column])
            for column in columns
        )
        if task_signature != expected or len(rows) != int(registry_row["row_count"]):
            _issue(issues, "error", "task_registry_signature_or_count_mismatch", "task_registry", 1, task_id)
        relations = Counter(clean_text(row[-1]) for row in rows)
        try:
            registered_relations = Counter(json.loads(clean_text(registry_row["relation_counts_json"])))
        except json.JSONDecodeError:
            registered_relations = Counter()
        if relations != registered_relations:
            _issue(issues, "error", "task_registry_relation_counts_mismatch", "task_registry", 1, task_id)

    _verify_full_build_artifacts(canonical, build_manifest, issues)
    _verify_task_dataset_manifest(canonical, build_manifest, issues)
    entity_dataset_manifest = build_manifest.get("entity_datasets", {})
    if "full_specialized" in clean_text(build_manifest.get("build_type", "")):
        for dataset_name in (
            "molecules",
            "molecule_aliases",
            "proteins",
            "protein_constructs",
            "assays",
        ):
            record = entity_dataset_manifest.get(dataset_name)
            if not isinstance(record, dict):
                _issue(issues, "error", "missing_entity_dataset_manifest", dataset_name, 1, "build_manifest")
            else:
                _verify_partitioned_dataset(canonical, dataset_name, record, issues)
    development_audit: dict[str, Any] = {
        "present": False,
        "linked_rows": 0,
        "unlinked_source_rows": None,
        "semantic_role": None,
    }
    development_record = build_manifest.get("molecule_development_annotations")
    if isinstance(development_record, dict):
        development_audit = {
            "present": True,
            "linked_rows": int(development_record.get("linked_rows", -1)),
            "unlinked_source_rows": int(development_record.get("unlinked_rows", -1)),
            "semantic_role": development_record.get("semantic_role"),
        }
        if development_record.get("semantic_role") != "development_metadata_not_outcome_or_model_label":
            _issue(
                issues,
                "error",
                "development_metadata_semantic_role_mismatch",
                "development_metadata",
                1,
                "build_manifest",
            )
        development_paths = _verify_partitioned_dataset(
            canonical,
            "molecule_development_annotations",
            development_record,
            issues,
        )
        linked_rows = 0
        for part in development_paths:
            frame = pd.read_parquet(part)
            linked_rows += len(frame)
            _accumulate_missingness(
                "molecule_development_annotations",
                frame,
                missing_counts,
                missing_denominators,
            )
            invalid_semantic = frame["semantic_role"].ne("development_metadata_not_outcome_or_model_label")
            _issue(
                issues,
                "error",
                "development_metadata_row_semantic_mismatch",
                "development_metadata",
                int(invalid_semantic.sum()),
                part.name,
            )
            records = [
                (
                    clean_text(row["development_metadata_id"]),
                    clean_text(row["molecule_id"]),
                )
                for row in frame.to_dict("records")
            ]
            duplicates = _insert_unique(
                connection,
                "INSERT OR IGNORE INTO development_metadata VALUES (?,?)",
                records,
            )
            _issue(
                issues,
                "error",
                "duplicate_development_metadata_id",
                "development_metadata",
                duplicates,
                part.name,
            )
        if linked_rows != int(development_record.get("linked_rows", -1)):
            _issue(
                issues,
                "error",
                "development_metadata_link_count_mismatch",
                "development_metadata",
                1,
                "build_manifest",
            )
        _issue(
            issues,
            "error",
            "orphan_development_metadata_molecule",
            "development_metadata",
            _sql_count(
                connection,
                "SELECT COUNT(*) FROM development_metadata d LEFT JOIN molecules m USING(molecule_id) WHERE m.molecule_id IS NULL",
            ),
            "Exact SQLite FK audit",
        )
    elif "full_specialized" in clean_text(build_manifest.get("build_type", "")):
        _issue(
            issues, "error", "missing_development_metadata_manifest", "development_metadata", 1, "Full build"
        )

    derivation_audit = _audit_binding_free_energy(canonical, issues)
    derived_observation_rows = _sql_count(
        connection,
        "SELECT COUNT(*) FROM observations WHERE observation_kind='derived'",
    )
    model_readiness_audit = _verify_model_readiness_accounting(
        canonical,
        build_manifest,
        issues,
        connection,
        task_counts=task_counts,
        task_dimension_counts=task_dimension_counts,
    )
    _verify_full_build_count_conservation(
        build_manifest,
        issues,
        observation_rows=observation_rows,
        lineage_rows=lineage_rows,
        derived_observation_rows=derived_observation_rows,
        derivation_rows=int(derivation_audit["derivation_rows"]),
        entity_counts=entity_counts,
        default_task_rows=task_counts["default"],
        sensitivity_task_rows=task_counts["derived_sensitivity"],
        task_registry_rows=len(task_registry),
    )

    kdki_row = connection.execute(
        """
        SELECT COUNT(*),
               SUM(COALESCE(m.usable_structure,0)),
               SUM(COALESCE(p.has_sequence,0)),
               SUM(CASE WHEN o.document_year<>'' THEN 1 ELSE 0 END)
        FROM observations o
        LEFT JOIN molecules m USING(molecule_id)
        LEFT JOIN proteins p USING(protein_id)
        WHERE lower(replace(o.endpoint,' ','')) IN ('kd','ki')
        """
    ).fetchone()
    kdki_coverage = {
        "rows": int(kdki_row[0] or 0),
        "with_usable_structure": int(kdki_row[1] or 0),
        "with_sequence": int(kdki_row[2] or 0),
        "with_year": int(kdki_row[3] or 0),
    }
    dedup_row = connection.execute(
        """
        SELECT COUNT(*), SUM(CASE WHEN n>1 THEN 1 ELSE 0 END)
        FROM (SELECT dedup_group_id, COUNT(*) n FROM observations WHERE dedup_group_id<>'' GROUP BY dedup_group_id)
        """
    ).fetchone()
    global_conflict = connection.execute(
        """
        SELECT COUNT(*), COALESCE(SUM(n),0)
        FROM (
            SELECT molecule_id,protein_id,assay_id,endpoint,canonical_unit,COUNT(*) n
            FROM observations
            WHERE relation='=' AND canonical_value>0
            GROUP BY molecule_id,protein_id,assay_id,endpoint,canonical_unit
            HAVING MAX(canonical_value)/MIN(canonical_value)>=10.0
        )
        """
    ).fetchone()
    dedup_frame = pd.DataFrame(
        {
            "metric": [
                "dedup_groups",
                "groups_with_repeats",
                "rows_in_shard_local_conflict_groups",
                "global_tenfold_conflict_groups",
                "global_rows_in_tenfold_conflict_groups",
            ],
            "value": [
                int(dedup_row[0] or 0),
                int(dedup_row[1] or 0),
                local_conflict_rows,
                int(global_conflict[0] or 0),
                int(global_conflict[1] or 0),
            ],
        }
    )
    concentration_rows: list[dict[str, Any]] = []
    for dimension in (
        "source_id",
        "activity_origin_name",
        "protein_id",
        "molecule_id",
        "assay_id",
        "document_id",
    ):
        query = f"""
        SELECT COALESCE(NULLIF({dimension},''),'<NA>') value, COUNT(*) n
        FROM observations GROUP BY value ORDER BY n DESC, value LIMIT 50
        """
        for value, count in connection.execute(query):
            concentration_rows.append(
                {
                    "dimension": dimension,
                    "value": clean_text(value),
                    "count": int(count),
                    "fraction_of_observations": float(count) / observation_rows
                    if observation_rows
                    else math.nan,
                }
            )
    concentration = pd.DataFrame(concentration_rows)
    connection.close()
    state_path.unlink(missing_ok=True)
    state_path.with_name(f"{state_path.name}-wal").unlink(missing_ok=True)
    state_path.with_name(f"{state_path.name}-shm").unlink(missing_ok=True)

    missingness = pd.DataFrame(
        [
            {
                "table": table,
                "field": field,
                "evidence_domain": domain,
                "missing_rows": missing_counts[(table, field, domain)],
                "denominator_rows": denominator,
                "missing_rate": missing_counts[(table, field, domain)] / denominator
                if denominator
                else math.nan,
            }
            for (table, field, domain), denominator in sorted(missing_denominators.items())
        ]
    )
    composition = pd.DataFrame(
        [
            {"dimension": dimension, "value": value, "count": count}
            for (dimension, value), count in sorted(group_counts.items())
        ]
    )
    distributions: list[dict[str, Any]] = []
    for (domain, endpoint, unit), values in sorted(endpoint_numeric.items()):
        series = pd.Series(values, dtype=float)
        distributions.append(
            {
                "evidence_domain": domain,
                "endpoint": endpoint,
                "canonical_unit": unit,
                "sampled_numeric_rows": len(series),
                "min": float(series.min()) if not series.empty else math.nan,
                "p01": float(series.quantile(0.01)) if not series.empty else math.nan,
                "median": float(series.median()) if not series.empty else math.nan,
                "p99": float(series.quantile(0.99)) if not series.empty else math.nan,
                "max": float(series.max()) if not series.empty else math.nan,
                "sampling_note": "exact when group <=200000; deterministic first-200000 otherwise",
            }
        )
    distribution_frame = pd.DataFrame(distributions)
    attrition = pd.DataFrame(
        [
            {
                "evidence_domain": domain,
                "inclusion_status": status,
                "reason": reason,
                "rows": count,
            }
            for (domain, status, reason), count in sorted(attrition_counts.items())
        ]
    )
    issue_frame = pd.DataFrame(issues, columns=["severity", "code", "table", "count", "detail"])
    report_artifacts = {
        "issues": _atomic_frame(issue_frame, reports / "qc_issues.csv"),
        "missingness": _atomic_frame(missingness, reports / "qc_missingness.csv"),
        "attrition": _atomic_frame(attrition, reports / "qc_attrition.csv"),
        "composition": _atomic_frame(composition, reports / "eda_composition.csv"),
        "distributions": _atomic_frame(distribution_frame, reports / "eda_endpoint_distributions.csv"),
        "concentration": _atomic_frame(concentration, reports / "eda_top_concentrations.csv"),
        "dedup": _atomic_frame(dedup_frame, reports / "dedup_analysis.csv"),
    }
    figures = _plot_counts(composition, reports)
    error_count = sum(int(row["count"]) for row in issues if row["severity"] == "error")
    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": _utc_now(),
        "build_manifest_sha256": (
            sha256_file(build_manifest_path) if build_manifest_path.is_file() else None
        ),
        "qc_passed": error_count == 0,
        "error_finding_rows": error_count,
        "issue_record_count": len(issues),
        "counts": {
            "observations": observation_rows,
            "lineage_edges": lineage_rows,
            "default_task_rows": task_counts["default"],
            "sensitivity_task_rows": task_counts["derived_sensitivity"],
            **entity_counts,
            "task_registry_rows": len(task_registry),
        },
        "kd_ki_coverage": kdki_coverage,
        "binding_free_energy_integrity": derivation_audit,
        "model_readiness_audit": model_readiness_audit,
        "development_metadata_audit": development_audit,
        "document_year_counts": dict(sorted(document_years.items())),
        "scale_method": (
            "exact disk-backed SQLite identity/FK/lineage/task/global-conflict audit; "
            "numeric EDA alone is capped deterministically at 200000 rows per endpoint/domain/unit"
        ),
        "scientific_boundaries": [
            "hERG is not QT, TdP, cardiotoxicity, or clinical risk",
            "PK/QT inventories are not default tasks without endpoint-specific context",
            "derived free energy is exact-Kd-only and opt-in sensitivity evidence",
            "development metadata is not a clinical result or model label",
            "missing clinical-results rows are reported as absent, never inferred preclinical",
        ],
        "artifacts": report_artifacts,
        "figures": figures,
    }
    _atomic_json(reports / "qc_report.json", report)
    _atomic_json(
        reports / "eda_summary.json",
        {
            "schema_version": SCHEMA_VERSION,
            "observation_rows": observation_rows,
            "composition": composition.to_dict("records"),
            "distribution_policy": "endpoint/domain/unit separated; no incompatible pooling",
            "temporal_coverage": dict(sorted(document_years.items())),
        },
    )
    if error_count:
        raise RuntimeError(
            f"Platform QC failed with {error_count} error rows; see {reports / 'qc_report.json'}"
        )
    return report


def run_platform_qc(
    canonical_root: str | Path,
    reports_root: str | Path,
) -> dict[str, Any]:
    """Run QC with exception-safe lifecycle management for disk-backed state."""

    canonical = Path(canonical_root).resolve()
    reports = Path(reports_root).resolve()
    with tempfile.TemporaryDirectory(prefix="platform_qc_") as state_directory:
        return _run_platform_qc_inner(
            canonical,
            reports,
            Path(state_directory) / "audit.sqlite",
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run canonical platform QC/EDA")
    parser.add_argument("--canonical-root", type=Path, required=True)
    parser.add_argument("--reports-root", type=Path, default=Path("research/reports/platform"))
    arguments = parser.parse_args(argv)
    report = run_platform_qc(arguments.canonical_root, arguments.reports_root)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["main", "run_platform_qc"]
