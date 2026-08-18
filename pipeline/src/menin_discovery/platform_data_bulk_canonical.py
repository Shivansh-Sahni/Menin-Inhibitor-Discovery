"""Streaming canonicalization of the complete ChEMBL_37 task inventories.

The full 24.5-million activity export remains a lossless interim evidence
store.  This module materializes the scientifically scoped, approximately
three-million-row Kd/Ki/IC50/EC50 + hERG/PK/QT inventories in immutable shards
without loading the corpus into one dataframe.  Default model tasks remain
strictly narrower than the retained evidence inventory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .platform_data_bulk import (
    ACTIVITY_ARROW_SCHEMA,
    DEVELOPMENT_ARROW_SCHEMA,
    TARGET_COMPONENT_ARROW_SCHEMA,
    _assert_schema_normalization_idle,
    _coerce_arrow_table,
    _verify_manifest_arrow_schema,
    _verify_schema_normalization_receipt,
)
from .platform_data_pipeline import (
    CHEMBL_SOURCE_ID,
    GAS_CONSTANT_KCAL_MOL_K,
    PUBLIC_ACCESS_CLASS,
    _assay_entities,
    _atomic_frame,
    _atomic_json,
    _join_model_inputs,
    _molecule_entities,
    _observations,
    _protein_entities,
    _task_view,
    _utc_now,
    _validate_observation_lineage,
    binding_free_energy_view,
)
from .platform_data_schema import (
    FIELD_SPECS,
    SCHEMA_VERSION,
    TABLE_REQUIRED_COLUMNS,
    arrow_schema_contract,
    canonical_json,
    clean_text,
    data_dictionary_frame,
    require_arrow_schema_contract,
    schema_document,
    stable_id,
    validate_table,
)
from .platform_data_sources import sha256_file, source_file_inventory, source_registry

_VIEW_ORDER = (
    "cardiac_qt_apd_inventory",
    "pk_adme_candidates",
    "herg_all_endpoints",
    "single_protein_kd_ki",
    "single_protein_ic50_ec50_candidates",
)
_SUMMARY_VIEW_NAMES = frozenset((*_VIEW_ORDER, "molecule_development_annotations", "target_components"))

_DICTIONARY_ARROW_TYPES: dict[str, pa.DataType] = {
    "string": pa.large_string(),
    "category": pa.large_string(),
    "json": pa.large_string(),
    "datetime": pa.large_string(),
    "float": pa.float64(),
    "integer": pa.int64(),
    "boolean": pa.bool_(),
}
_NORMATIVE_ARROW_TYPES: dict[str, pa.DataType] = {
    spec.field_name: _DICTIONARY_ARROW_TYPES[spec.dtype] for spec in FIELD_SPECS
}
_NORMATIVE_ARROW_TYPES.update(
    {
        # Canonical extensions preserved from the ChEMBL evidence layer.
        "parent_target_id": pa.large_string(),
        "component_id": pa.large_string(),
        "component_order": pa.int64(),
        "component_accession": pa.large_string(),
        "component_type": pa.large_string(),
        "tissue": pa.large_string(),
        "strain": pa.large_string(),
        "subcellular_fraction": pa.large_string(),
        "assay_test_type": pa.large_string(),
        "assay_category": pa.large_string(),
        "assay_tax_id": pa.large_string(),
        "relationship_type": pa.large_string(),
        "source_assay_external_id": pa.large_string(),
        "cell_id": pa.large_string(),
        "tissue_id": pa.large_string(),
        "variant_id": pa.large_string(),
        "assay_group": pa.large_string(),
        "bao_format_id": pa.large_string(),
        "confidence_score": pa.int64(),
        "chembl_standard_flag": pa.large_string(),
        "chembl_potential_duplicate": pa.bool_(),
        "chembl_data_validity_comment": pa.large_string(),
        "document_doi": pa.large_string(),
        "document_pubmed_id": pa.large_string(),
        "document_patent_id": pa.large_string(),
        "document_title": pa.large_string(),
        "document_type": pa.large_string(),
        "document_chembl_release_id": pa.large_string(),
        "activity_origin_id": pa.large_string(),
        "activity_origin_name": pa.large_string(),
        "activity_origin_description": pa.large_string(),
        "chembl_activity_id": pa.int64(),
        # Standard-state binding free-energy derivation contract.
        "source_observation_id": pa.large_string(),
        "source_snapshot_id": pa.large_string(),
        "source_relation": pa.large_string(),
        "source_kd_value": pa.float64(),
        "source_kd_unit": pa.large_string(),
        "kd_nM": pa.float64(),
        "kd_molar": pa.float64(),
        "temperature_k": pa.float64(),
        "temperature_source": pa.large_string(),
        "standard_state": pa.large_string(),
        "gas_constant_kcal_mol_k": pa.float64(),
        "formula": pa.large_string(),
        "delta_g_kcal_mol": pa.float64(),
        "roundtrip_kd_molar": pa.float64(),
        "roundtrip_relative_error": pa.float64(),
        "label_lineage_digest": pa.large_string(),
        # Threshold audit columns carried by task rows.
        "threshold_low_nM": pa.float64(),
        "threshold_high_nM": pa.float64(),
        "threshold_source_value_nM": pa.float64(),
        "threshold_policy": pa.large_string(),
        # Metadata-only development annotation extension.
        "molecule_row_id": pa.int64(),
        "molecule_chembl_id": pa.large_string(),
        "molecule_pref_name": pa.large_string(),
        "molecule_type": pa.large_string(),
        "max_phase": pa.float64(),
        "first_approval": pa.int64(),
        "withdrawn_flag": pa.int64(),
        "black_box_warning": pa.int64(),
        "therapeutic_flag": pa.int64(),
        "annotation_role": pa.large_string(),
        "development_metadata_id": pa.large_string(),
        "semantic_role": pa.large_string(),
        # Task-registry-only audit fields.
        "policy_version": pa.large_string(),
        "relation_counts_json": pa.large_string(),
        "intended_use": pa.large_string(),
        "prohibited_claim": pa.large_string(),
        # Task-input readiness and exclusion audit fields.
        "required_modalities": pa.large_string(),
        "task_scope": pa.large_string(),
        "model_readiness_exclusion_reason": pa.large_string(),
        "missing_standardized_smiles": pa.bool_(),
        "missing_standard_inchi_key": pa.bool_(),
        "missing_protein_sequence": pa.bool_(),
    }
)

_MODEL_READINESS_POLICY_VERSION = "platform-model-readiness-v1"
_ALLOWED_MODALITY_DECLARATIONS = frozenset(
    {
        "small_molecule_structure",
        "small_molecule_structure;protein_sequence",
    }
)
_MODEL_READINESS_REASONS = (
    ("missing_standardized_smiles", "standardized_smiles", "small_molecule_structure"),
    ("missing_standard_inchi_key", "standard_inchi_key", "small_molecule_structure"),
    ("missing_protein_sequence", "sequence", "protein_sequence"),
)


def bulk_canonicalization_contract() -> dict[str, Any]:
    """Return the frozen bulk-to-canonical callable and artifact contract."""

    return {
        "input": "chembl_37_bulk/specialized_views/*_manifest.json plus hash-verified Parquet parts",
        "callable": (
            "materialize_chembl37_specialized_canonical(raw_root, interim_root, canonical_root, reports_root)"
        ),
        "scientific_scope": {
            "retained_inventory": [
                "all positive molar single-protein Kd/Ki",
                "all positive molar single-protein IC50/EC50 candidates",
                "all CHEMBL240/hERG endpoints",
                "PK/ADME candidates including source-table-pinned src_id=39",
                "explicit QT/QTc/APD candidates",
            ],
            "default_tasks": (
                "Kd/Ki/IC50/EC50 nM contracts separated by domain, assay family, unit, and label policy; "
                "hERG default limited to herg_functional IC50; PK/QT excluded by default"
            ),
            "derived_sensitivity": "exact positive Kd only; explicit opt-in and SHA-256 lineage",
        },
        "output": {
            "root": "canonical/full_chembl37",
            "observations": "observations/part-*.parquet",
            "lineage": "observation_lineage/part-*.parquet",
            "default_model_tasks": "tasks/default/<task_type>/part-*.parquet",
            "derived_sensitivity_tasks": "tasks/derived_sensitivity/<task_type>/part-*.parquet",
            "entities": [
                "molecules/part-*.parquet",
                "proteins/part-*.parquet",
                "protein_constructs/part-*.parquet",
                "assays/part-*.parquet",
            ],
            "development_metadata": "molecule_development_annotations/part-*.parquet (metadata only)",
            "task_datasets": "task_datasets.json with ordered parts, hashes, and aggregate digests",
            "manifest": "build_manifest.json",
        },
    }


def _component_inventory(root: Path) -> list[dict[str, Any]]:
    """Hash every committed build component except the self-referential manifest."""

    resolved_root = root.resolve()
    records: list[dict[str, Any]] = []
    for path in sorted(
        (candidate for candidate in resolved_root.rglob("*") if candidate.is_file()),
        key=lambda candidate: candidate.relative_to(resolved_root).as_posix(),
    ):
        relative_path = path.relative_to(resolved_root).as_posix()
        if relative_path == "build_manifest.json":
            continue
        if path.is_symlink():
            raise RuntimeError(f"Symlinks are prohibited in a canonical component inventory: {path}")
        record: dict[str, Any] = {
            "path": relative_path,
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        if path.suffix.casefold() == ".parquet":
            metadata = pq.ParquetFile(path).metadata
            if metadata is None:
                raise RuntimeError(f"Parquet component lacks footer metadata: {path}")
            record["rows"] = int(metadata.num_rows)
        elif path.suffix.casefold() == ".csv":
            record["rows"] = len(pd.read_csv(path))
        records.append(record)
    return records


def _load_build_manifest(build_root: Path) -> tuple[dict[str, Any], Path]:
    manifest_path = build_root / "build_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("Canonical build lacks build_manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise RuntimeError("Canonical build manifest must be a JSON object")
    return manifest, manifest_path


def _run_bound_qc(build_root: Path, reports_root: Path) -> dict[str, Any]:
    """Require a fresh QC acceptance cryptographically bound to this manifest."""

    manifest, manifest_path = _load_build_manifest(build_root)
    expected_inventory = manifest.get("component_inventory")
    if not isinstance(expected_inventory, list) or _component_inventory(build_root) != expected_inventory:
        raise RuntimeError("Canonical component inventory failed exact pre-QC verification")
    from .platform_data_qc import run_platform_qc

    report = run_platform_qc(build_root, reports_root)
    manifest_sha256 = sha256_file(manifest_path)
    if not bool(report.get("qc_passed")) or report.get("build_manifest_sha256") != manifest_sha256:
        raise RuntimeError("QC acceptance is not bound to the current build manifest")
    return report


def _promote_qc_accepted_build(
    building: Path,
    destination: Path,
    reports_root: Path,
) -> None:
    """Promote atomically only after component verification and bound exact QC."""

    if destination.exists():
        raise RuntimeError(f"Canonical destination already exists: {destination}")
    _run_bound_qc(building, reports_root)
    os.replace(building, destination)


def _load_input_parts(
    interim_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], Path, dict[str, Any]]:
    bulk_root = interim_root / "chembl_37_bulk"
    _assert_schema_normalization_idle(bulk_root)
    root = bulk_root / "specialized_views"
    summary_path = root / "specialized_views_manifest.json"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary_views = summary.get("views")
    if not isinstance(summary_views, dict) or set(summary_views) != _SUMMARY_VIEW_NAMES:
        raise RuntimeError("Specialized summary has an incomplete or unexpected child-manifest set")
    database_sha256 = clean_text(summary.get("database_sha256"))
    if (
        summary.get("schema_version") != SCHEMA_VERSION
        or summary.get("source_version") != "ChEMBL_37"
        or re.fullmatch(r"[0-9a-f]{64}", database_sha256) is None
    ):
        raise RuntimeError("Specialized summary violates its schema/source/database identity contract")
    _verify_schema_normalization_receipt(summary, bulk_root)
    full_manifest_path = bulk_root / "activity_export_manifest.json"
    if not full_manifest_path.is_file():
        raise FileNotFoundError(full_manifest_path)
    full_manifest = json.loads(full_manifest_path.read_text(encoding="utf-8"))
    if (
        full_manifest.get("schema_version") != SCHEMA_VERSION
        or full_manifest.get("source_version") != "ChEMBL_37"
        or full_manifest.get("database_sha256") != database_sha256
    ):
        raise RuntimeError("Full activity manifest identity does not match specialized inputs")
    _verify_manifest_arrow_schema(
        full_manifest,
        bulk_root,
        ACTIVITY_ARROW_SCHEMA,
        context="activity_facts",
    )

    def load_bound_child(view_name: str) -> dict[str, Any]:
        manifest_path = root / f"{view_name}_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Required specialized view is incomplete: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest != summary_views[view_name]:
            raise RuntimeError(f"Specialized summary/standalone child manifest drift: {view_name}")
        if (
            manifest.get("schema_version") != SCHEMA_VERSION
            or manifest.get("view_name") != view_name
            or manifest.get("source_version") != "ChEMBL_37"
            or manifest.get("database_sha256") != database_sha256
        ):
            raise RuntimeError(f"Specialized child identity mismatch: {manifest_path}")
        return manifest

    def resolved_part_path(relative_path: object, directory: Path) -> Path:
        path = (root / clean_text(relative_path)).resolve()
        try:
            path.relative_to(directory.resolve())
        except ValueError as error:
            raise RuntimeError(f"Specialized part escapes its declared view directory: {path}") from error
        return path

    def verify_partitioned_child(
        view_name: str,
        manifest: dict[str, Any],
    ) -> list[dict[str, Any]]:
        expected_schema = (
            DEVELOPMENT_ARROW_SCHEMA
            if view_name == "molecule_development_annotations"
            else ACTIVITY_ARROW_SCHEMA
        )
        schema_contract = require_arrow_schema_contract(
            manifest,
            expected_schema,
            context=view_name,
        )
        directory = root / view_name
        parts = manifest.get("parts")
        if not isinstance(parts, list) or int(manifest.get("part_count", -1)) != len(parts):
            raise RuntimeError(f"Specialized part-count contract mismatch: {view_name}")
        listed: set[Path] = set()
        physical_rows = 0
        verified: list[dict[str, Any]] = []
        for part in parts:
            if not isinstance(part, dict):
                raise RuntimeError(f"Malformed specialized part record: {view_name}")
            path = resolved_part_path(part.get("path"), directory)
            if path in listed:
                raise RuntimeError(f"Duplicate specialized part path: {path}")
            listed.add(path)
            if not path.is_file():
                raise RuntimeError(f"Missing specialized input part: {path}")
            metadata = pq.ParquetFile(path).metadata
            physical_schema = pq.ParquetFile(path).schema_arrow.remove_metadata()
            if metadata is None:
                raise RuntimeError(f"Specialized Parquet part lacks footer metadata: {path}")
            actual_rows = int(metadata.num_rows)
            if (
                sha256_file(path) != part.get("sha256")
                or path.stat().st_size != int(part.get("size_bytes", -1))
                or actual_rows != int(part.get("rows", -1))
                or part.get("arrow_schema_sha256") != schema_contract["sha256"]
                or not physical_schema.equals(expected_schema, check_metadata=True)
            ):
                raise RuntimeError(f"Specialized input part failed physical verification: {path}")
            physical_rows += actual_rows
            verified.append(
                {
                    "view_name": view_name,
                    "path": path,
                    "sha256": part["sha256"],
                    "rows": actual_rows,
                }
            )
        actual = {path.resolve() for path in directory.glob("*.parquet")}
        if actual != listed:
            raise RuntimeError(f"Unmanifested/missing specialized parts for {view_name}")
        if physical_rows != int(manifest.get("row_count", -1)):
            raise RuntimeError(f"Specialized row-count contract mismatch: {view_name}")
        return verified

    records: list[dict[str, Any]] = []
    view_counts: dict[str, int] = {}
    for view_name in _VIEW_ORDER:
        manifest = load_bound_child(view_name)
        view_counts[view_name] = int(manifest.get("row_count", -1))
        records.extend(verify_partitioned_child(view_name, manifest))

    component_record = load_bound_child("target_components")
    component_schema_contract = require_arrow_schema_contract(
        component_record,
        TARGET_COMPONENT_ARROW_SCHEMA,
        context="target_components",
    )
    component_path = (root / clean_text(component_record.get("path"))).resolve()
    try:
        component_path.relative_to(root.resolve())
    except ValueError as error:
        raise RuntimeError("Target-component artifact escapes the specialized root") from error
    component_metadata = pq.ParquetFile(component_path).metadata if component_path.is_file() else None
    component_schema = (
        pq.ParquetFile(component_path).schema_arrow.remove_metadata() if component_path.is_file() else None
    )
    if (
        component_path.name != "target_components.parquet"
        or component_metadata is None
        or sha256_file(component_path) != component_record.get("sha256")
        or component_path.stat().st_size != int(component_record.get("size_bytes", -1))
        or int(component_metadata.num_rows) != int(component_record.get("row_count", -1))
        or component_record.get("arrow_schema_sha256") != component_schema_contract["sha256"]
        or component_schema is None
        or not component_schema.equals(TARGET_COMPONENT_ARROW_SCHEMA, check_metadata=True)
    ):
        raise RuntimeError("Target-component artifact failed exact physical verification")

    development = load_bound_child("molecule_development_annotations")
    if development.get("semantic_role") != "metadata_only_not_outcome":
        raise RuntimeError("Development annotations lack the metadata-only semantic contract")
    verified_development = verify_partitioned_child(
        "molecule_development_annotations",
        development,
    )
    development_rows = sum(int(part["rows"]) for part in verified_development)
    return (
        records,
        {
            "view_row_counts": view_counts,
            "summary_sha256": sha256_file(summary_path),
            "database_sha256": database_sha256,
            "full_activity_export": {
                "rows": int(full_manifest.get("rows_written", -1)),
                "part_count": int(full_manifest.get("part_count", -1)),
                "manifest_sha256": sha256_file(full_manifest_path),
                "arrow_schema_sha256": full_manifest["arrow_schema"]["sha256"],
            },
            "target_components": {
                "path": component_path.name,
                "rows": int(component_metadata.num_rows),
                "sha256": component_record.get("sha256"),
                "size_bytes": component_path.stat().st_size,
                "query_sha256": component_record.get("query_sha256"),
            },
            "molecule_development_annotations": {
                "rows": development_rows,
                "part_count": len(verified_development),
                "manifest_digest": hashlib.sha256(canonical_json(development).encode("utf-8")).hexdigest(),
                "semantic_role": "development_metadata_not_outcome_or_model_label",
            },
        },
        component_path,
        development,
    )


def _prepare_bulk_rows(frame: pd.DataFrame, *, snapshot_id: str, source_file_id: str) -> pd.DataFrame:
    out = frame.copy()
    out["_activity_key"] = out["activity_id"].map(lambda value: str(int(value)) if pd.notna(value) else "")
    out["_source_compound_key"] = out.get("molecule_chembl_id", "").fillna("").astype(str).str.strip()
    out["_source_target_key"] = out.get("target_chembl_id", "").fillna("").astype(str).str.strip()
    out["_source_assay_key"] = out.get("assay_chembl_id", "").fillna("").astype(str).str.strip()
    for column, prefix in (
        ("_source_compound_key", "CMPD"),
        ("_source_target_key", "TGT"),
        ("_source_assay_key", "SRCASSAY"),
    ):
        missing = out[column].eq("")
        out.loc[missing, column] = [
            stable_id(prefix, "missing", activity_id) for activity_id in out.loc[missing, "_activity_key"]
        ]
    out["_raw_file_id"] = source_file_id
    out["_raw_file_ids"] = source_file_id
    out["_snapshot_id"] = snapshot_id
    out["_snapshot_ids"] = snapshot_id
    out["_snapshot_id_primary"] = snapshot_id
    out["_raw_duplicate_conflict"] = False
    out["_raw_mirror_count"] = 1
    return out


def _partition_model_ready_tasks(
    model_tasks: pd.DataFrame,
    *,
    task_scope: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split task candidates by their explicitly declared input modalities."""

    if model_tasks.empty:
        return model_tasks.copy(), pd.DataFrame()
    required_columns = {
        "observation_id",
        "task_id",
        "task_type",
        "source_id",
        "snapshot_id",
        "source_record_id",
        "molecule_id",
        "protein_id",
        "assay_id",
        "required_modalities",
        "standardized_smiles",
        "standard_inchi_key",
        "sequence",
        "canonical_target_id",
    }
    missing_columns = sorted(required_columns - set(model_tasks.columns))
    if missing_columns:
        raise RuntimeError(f"Task readiness input columns are missing: {missing_columns}")
    declarations = model_tasks["required_modalities"].map(clean_text)
    invalid_declarations = sorted(set(declarations) - _ALLOWED_MODALITY_DECLARATIONS)
    if invalid_declarations:
        raise RuntimeError(
            f"Task candidates have undeclared model modality contracts: {invalid_declarations}"
        )
    missing_flags: dict[str, pd.Series] = {}
    for reason, column, modality in _MODEL_READINESS_REASONS:
        requires_modality = declarations.map(
            lambda declaration, required=modality: required in declaration.split(";")
        )
        missing_flags[reason] = requires_modality & model_tasks[column].map(clean_text).eq("")
    reason_values = [
        ";".join(
            reason for reason, _, _ in _MODEL_READINESS_REASONS if bool(missing_flags[reason].iloc[index])
        )
        for index in range(len(model_tasks))
    ]
    excluded_mask = pd.Series(reason_values, index=model_tasks.index).ne("")
    eligible = model_tasks.loc[~excluded_mask].copy().reset_index(drop=True)
    exclusion_columns = [
        "observation_id",
        "task_id",
        "task_type",
        "source_id",
        "snapshot_id",
        "source_record_id",
        "molecule_id",
        "protein_id",
        "assay_id",
        "canonical_target_id",
        "required_modalities",
    ]
    exclusions = model_tasks.loc[excluded_mask, exclusion_columns].copy()
    exclusions["task_scope"] = task_scope
    exclusions["model_readiness_exclusion_reason"] = pd.Series(
        reason_values,
        index=model_tasks.index,
    ).loc[excluded_mask]
    for reason, _, _ in _MODEL_READINESS_REASONS:
        exclusions[reason] = missing_flags[reason].loc[excluded_mask].astype(bool)
    exclusions = exclusions.reset_index(drop=True)
    if len(eligible) + len(exclusions) != len(model_tasks):
        raise RuntimeError("Task readiness partition does not conserve candidate rows")
    return eligible, exclusions


def _select_bulk_source_files(files: pd.DataFrame, snapshot_id: str) -> tuple[pd.DataFrame, str]:
    """Select the exact full-release snapshot dependency set and its one SQLite file."""

    selected = files[
        files["source_id"].eq(CHEMBL_SOURCE_ID) & files["snapshot_id"].eq(snapshot_id)
    ].reset_index(drop=True)
    if selected.empty or set(selected["snapshot_id"]) != {snapshot_id}:
        raise RuntimeError("Full-release source-file inventory is missing or snapshot-mixed")
    database_files = selected[selected["relative_path"].str.endswith("/extracted/chembl_37.db", na=False)]
    if len(database_files) != 1:
        raise RuntimeError("Exactly one manifested extracted ChEMBL database source file is required")
    return selected, clean_text(database_files.iloc[0]["source_file_id"])


def _json_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False
    if isinstance(missing, (bool, np.bool_)) and bool(missing):
        return None
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        # Pandas promotes nullable integer columns to float on shards that
        # contain a missing value.  Registry identity is semantic, so 1 and
        # 1.0 must serialize identically; otherwise the surrounding shard's
        # null pattern creates a false cross-shard entity collision.
        if value.is_integer():
            return int(value)
    return value


def _record_payload(record: dict[str, Any]) -> str:
    normalized = {str(key): _json_scalar(value) for key, value in record.items()}
    return json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


class _DiskRegistry:
    """Disk-backed exact-ID registry with fail-closed collision detection."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute(
            """
            CREATE TABLE records (
                dataset TEXT NOT NULL,
                record_key TEXT NOT NULL,
                payload TEXT NOT NULL,
                digest TEXT NOT NULL,
                conflict INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (dataset, record_key)
            )
            """
        )
        self.connection.execute(
            "CREATE TABLE seen_activities (activity_id INTEGER PRIMARY KEY, first_view TEXT NOT NULL)"
        )
        self.connection.execute(
            """
            CREATE TABLE molecule_map (
                source_compound_id TEXT PRIMARY KEY,
                molecule_id TEXT NOT NULL,
                conflict INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        self.connection.commit()

    def add_frame(self, dataset: str, key: str, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        if key not in frame.columns:
            raise RuntimeError(f"{dataset} registry frame lacks key {key}")
        payloads: list[tuple[str, str, str, str]] = []
        for record in frame.to_dict("records"):
            record_key = clean_text(record[key])
            if not record_key:
                raise RuntimeError(f"{dataset} registry received a blank key")
            payload = _record_payload(record)
            payloads.append(
                (
                    dataset,
                    record_key,
                    payload,
                    hashlib.sha256(payload.encode("utf-8")).hexdigest(),
                )
            )
        self.connection.executemany(
            """
            INSERT INTO records(dataset, record_key, payload, digest)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(dataset, record_key) DO UPDATE SET
                conflict = CASE
                    WHEN records.digest <> excluded.digest THEN 1
                    ELSE records.conflict
                END
            """,
            payloads,
        )
        if dataset == "molecule_aliases":
            mappings = [
                (clean_text(row["source_compound_id"]), clean_text(row["molecule_id"]))
                for row in frame.to_dict("records")
            ]
            self.connection.executemany(
                """
                INSERT INTO molecule_map(source_compound_id, molecule_id)
                VALUES (?, ?)
                ON CONFLICT(source_compound_id) DO UPDATE SET
                    conflict = CASE
                        WHEN molecule_map.molecule_id <> excluded.molecule_id THEN 1
                        ELSE molecule_map.conflict
                    END
                """,
                mappings,
            )
        self.connection.commit()

    def claim_activity_ids(self, activity_ids: pd.Series, view_name: str) -> np.ndarray:
        numeric = pd.to_numeric(activity_ids, errors="raise").astype("int64")
        if numeric.duplicated().any():
            raise RuntimeError(f"Duplicate activity IDs within a {view_name} input shard")
        self.connection.execute("DROP TABLE IF EXISTS temp.incoming_activities")
        self.connection.execute(
            "CREATE TEMP TABLE incoming_activities (ordinal INTEGER PRIMARY KEY, activity_id INTEGER UNIQUE)"
        )
        self.connection.executemany(
            "INSERT INTO incoming_activities(ordinal, activity_id) VALUES (?, ?)",
            [(ordinal, int(value)) for ordinal, value in enumerate(numeric)],
        )
        kept_ordinals = {
            int(row[0])
            for row in self.connection.execute(
                """
                SELECT incoming.ordinal
                FROM incoming_activities AS incoming
                LEFT JOIN seen_activities AS seen USING(activity_id)
                WHERE seen.activity_id IS NULL
                """
            )
        }
        self.connection.execute(
            """
            INSERT OR IGNORE INTO seen_activities(activity_id, first_view)
            SELECT activity_id, ? FROM incoming_activities
            """,
            (view_name,),
        )
        self.connection.execute("DROP TABLE temp.incoming_activities")
        self.connection.commit()
        return np.fromiter(
            (ordinal in kept_ordinals for ordinal in range(len(numeric))),
            dtype=bool,
            count=len(numeric),
        )

    def assert_consistent(self) -> None:
        collisions = self.connection.execute(
            "SELECT dataset, COUNT(*) FROM records WHERE conflict = 1 GROUP BY dataset"
        ).fetchall()
        molecule_collisions = int(
            self.connection.execute("SELECT COUNT(*) FROM molecule_map WHERE conflict = 1").fetchone()[0]
        )
        if molecule_collisions:
            collisions.append(("source_compound_to_molecule", molecule_collisions))
        if collisions:
            raise RuntimeError(f"Conflicting duplicate canonical entity IDs: {collisions}")

    def count(self, dataset: str) -> int:
        row = self.connection.execute("SELECT COUNT(*) FROM records WHERE dataset = ?", (dataset,)).fetchone()
        return int(row[0]) if row else 0

    def resolve_molecules(self, source_compound_ids: pd.Series) -> dict[str, str]:
        values = sorted({clean_text(value) for value in source_compound_ids if clean_text(value)})
        self.connection.execute("DROP TABLE IF EXISTS temp.requested_compounds")
        self.connection.execute("CREATE TEMP TABLE requested_compounds (source_compound_id TEXT PRIMARY KEY)")
        self.connection.executemany(
            "INSERT INTO requested_compounds(source_compound_id) VALUES (?)",
            [(value,) for value in values],
        )
        resolved = {
            clean_text(row[0]): clean_text(row[1])
            for row in self.connection.execute(
                """
                SELECT mapping.source_compound_id, mapping.molecule_id
                FROM molecule_map AS mapping
                INNER JOIN requested_compounds AS requested USING(source_compound_id)
                """
            )
        }
        self.connection.execute("DROP TABLE temp.requested_compounds")
        return resolved

    def iter_frames(self, dataset: str, *, chunk_size: int = 100_000) -> Any:
        cursor = self.connection.execute(
            "SELECT payload FROM records WHERE dataset = ? ORDER BY record_key", (dataset,)
        )
        while True:
            rows = cursor.fetchmany(chunk_size)
            if not rows:
                break
            yield pd.DataFrame([json.loads(str(row[0])) for row in rows])

    def close(self) -> None:
        self.connection.close()


_TASK_SIGNATURE_COLUMNS = (
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


class _TaskRegistryAccumulator:
    """Constant-in-task-count registry aggregation without retaining task rows."""

    def __init__(self) -> None:
        self.entries: dict[str, dict[str, Any]] = {}

    def add(self, tasks: pd.DataFrame) -> None:
        if tasks.empty:
            return
        for task_id, group in tasks.groupby("task_id", sort=True):
            for column in _TASK_SIGNATURE_COLUMNS:
                if group[column].nunique(dropna=False) != 1:
                    raise RuntimeError(f"Task {task_id} pools incompatible {column} values")
            first = group.iloc[0]
            signature = {column: _json_scalar(first[column]) for column in _TASK_SIGNATURE_COLUMNS}
            entry = self.entries.setdefault(
                clean_text(task_id),
                {"signature": signature, "row_count": 0, "relations": Counter()},
            )
            if entry["signature"] != signature:
                raise RuntimeError(f"Task {task_id} changed signature across shards")
            entry["row_count"] = int(entry["row_count"]) + len(group)
            entry["relations"].update(clean_text(value) for value in group["label_relation"])

    def frame(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for task_id, entry in sorted(self.entries.items()):
            signature = entry["signature"]
            rows.append(
                {
                    "task_id": task_id,
                    **signature,
                    "policy_version": "platform-task-contract-v1",
                    "row_count": int(entry["row_count"]),
                    "relation_counts_json": canonical_json(
                        {key: int(value) for key, value in sorted(entry["relations"].items())}
                    ),
                    "intended_use": "endpoint-specific supervised modeling under frozen group/temporal splits",
                    "prohibited_claim": "not clinical efficacy, QT/TdP risk, or cross-endpoint equivalence",
                }
            )
        return pd.DataFrame(rows)


def _validate_frame(table: str, frame: pd.DataFrame) -> None:
    issues = validate_table(table, frame)
    if issues:
        raise RuntimeError(f"{table} schema validation failed: {canonical_json(issues)}")


def _component_metadata(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    for target_id, group in frame.groupby("target_chembl_id", sort=True, dropna=False):
        ordered = group.sort_values(["component_id", "accession"], kind="stable", na_position="last")
        components: list[dict[str, Any]] = []
        for _, row in ordered.iterrows():
            component_id = clean_text(row.get("component_id", ""))
            accession = clean_text(row.get("accession", ""))
            sequence = re.sub(r"\s+", "", clean_text(row.get("sequence", ""))).upper()
            component_type = clean_text(row.get("component_type", ""))
            if component_id or accession or sequence:
                components.append(
                    {
                        "component_id": component_id,
                        "accession": accession,
                        "sequence": sequence,
                        "component_type": component_type,
                        "organism": clean_text(row.get("component_organism", "")),
                    }
                )
        metadata[clean_text(target_id)] = {
            "target_type": clean_text(ordered.iloc[0].get("target_type", "")),
            "target_name": clean_text(ordered.iloc[0].get("target_name", "")),
            "species": clean_text(ordered.iloc[0].get("target_organism", "")),
            "accessions": [component["accession"] for component in components if component["accession"]],
            "components": components,
        }
    return metadata


def _bulk_protein_entities(
    rows: pd.DataFrame,
    target_metadata: dict[str, dict[str, Any]],
) -> tuple[pd.DataFrame, dict[str, str], pd.DataFrame]:
    """Preserve target components without inventing a concatenated complex sequence."""

    proteins, target_map = _protein_entities(rows, target_metadata)
    protein_records = {clean_text(row["protein_id"]): row for row in proteins.to_dict("records")}
    constructs: list[dict[str, Any]] = []
    for target_id in sorted(set(rows["_source_target_key"].map(clean_text))):
        parent_id = target_map[target_id]
        parent = protein_records[parent_id]
        components = list(target_metadata.get(target_id, {}).get("components", []))
        if not components:
            first = rows.loc[rows["_source_target_key"].map(clean_text).eq(target_id)].iloc[0]
            accessions = clean_text(first.get("component_accessions", "")).split(";")
            sequences = clean_text(first.get("component_sequences", "")).split(";")
            types = clean_text(first.get("component_types", "")).split(";")
            width = max(len(accessions), len(sequences), len(types))
            components = [
                {
                    "component_id": str(index + 1),
                    "accession": clean_text(accessions[index] if index < len(accessions) else ""),
                    "sequence": re.sub(
                        r"\s+", "", clean_text(sequences[index] if index < len(sequences) else "")
                    ).upper(),
                    "component_type": clean_text(types[index] if index < len(types) else ""),
                    "organism": clean_text(first.get("target_organism", "")),
                }
                for index in range(width)
                if any(
                    clean_text(values[index] if index < len(values) else "")
                    for values in (accessions, sequences, types)
                )
            ]
        components = sorted(
            components,
            key=lambda component: (
                clean_text(component.get("component_id", "")),
                clean_text(component.get("accession", "")),
                hashlib.sha256(clean_text(component.get("sequence", "")).encode("utf-8")).hexdigest(),
            ),
        )
        if len(components) == 1 and clean_text(parent["entity_type"]) == "single_protein":
            component = components[0]
            sequence = clean_text(component.get("sequence", ""))
            accession = clean_text(component.get("accession", ""))
            parent["sequence"] = sequence
            parent["sequence_sha256"] = (
                hashlib.sha256(sequence.encode("utf-8")).hexdigest() if sequence else ""
            )
            parent["uniprot_accession"] = accession
            parent["component_protein_ids"] = parent_id
            component_protein_id = parent_id
            component_ids = [parent_id]
        else:
            component_ids = []
            for component in components:
                accession = clean_text(component.get("accession", ""))
                sequence = clean_text(component.get("sequence", ""))
                sequence_sha = hashlib.sha256(sequence.encode("utf-8")).hexdigest() if sequence else ""
                component_protein_id = stable_id(
                    "PROT",
                    "ChEMBL_37.component",
                    accession or target_id,
                    sequence_sha or clean_text(component.get("component_id", "")),
                )
                component_ids.append(component_protein_id)
                protein_records.setdefault(
                    component_protein_id,
                    {
                        "protein_id": component_protein_id,
                        "entity_type": "single_protein",
                        "canonical_target_id": accession
                        or f"{target_id}:component:{clean_text(component.get('component_id', ''))}",
                        "target_name": accession or "ChEMBL target component",
                        "gene_symbol": "",
                        "uniprot_accession": accession,
                        "sequence": sequence,
                        "sequence_sha256": sequence_sha,
                        "isoform": "",
                        "species": clean_text(component.get("organism", "")),
                        "component_protein_ids": "",
                        "identity_resolution_status": "resolved" if accession else "source_assertion",
                    },
                )
            parent["sequence"] = ""
            parent["sequence_sha256"] = ""
            parent["component_protein_ids"] = ";".join(component_ids)
        for component_order, (component, component_protein_id) in enumerate(
            zip(components, component_ids, strict=True), start=1
        ):
            sequence = clean_text(component.get("sequence", ""))
            if not sequence:
                continue
            accession = clean_text(component.get("accession", ""))
            component_source_id = clean_text(component.get("component_id", "")) or str(component_order)
            constructs.append(
                {
                    "construct_id": stable_id(
                        "CONSTRUCT", parent_id, component_source_id, accession, sequence
                    ),
                    "protein_id": component_protein_id,
                    "sequence": sequence,
                    "sequence_sha256": hashlib.sha256(sequence.encode("utf-8")).hexdigest(),
                    "construct_description": "ChEMBL target component canonical sequence",
                    "source_id": CHEMBL_SOURCE_ID,
                    "source_record_id": f"ChEMBL:target:{target_id}:component:{component_source_id}",
                    "quality_status": "source_component_sequence",
                    "parent_target_id": target_id,
                    "component_id": component_source_id,
                    "component_order": component_order,
                    "component_accession": accession,
                    "component_type": clean_text(component.get("component_type", "")),
                }
            )
    protein_frame = (
        pd.DataFrame(protein_records.values()).sort_values("protein_id", kind="stable").reset_index(drop=True)
    )
    construct_columns = list(TABLE_REQUIRED_COLUMNS["protein_constructs"]) + [
        "construct_description",
        "quality_status",
        "parent_target_id",
        "component_id",
        "component_order",
        "component_accession",
        "component_type",
    ]
    construct_frame = pd.DataFrame(constructs, columns=construct_columns)
    return protein_frame, target_map, construct_frame


def _validate_derivation_bundle(
    source_observations: pd.DataFrame,
    derivations: pd.DataFrame,
    derived_observations: pd.DataFrame,
) -> None:
    if derivations.empty and derived_observations.empty:
        return
    if len(derivations) != len(derived_observations):
        raise RuntimeError("Derived observation and derivation cardinalities differ")
    sources = source_observations.set_index("observation_id", drop=False)
    if not sources.index.is_unique:
        raise RuntimeError("Derivation source observations are not unique")
    derived_ids = set(derived_observations["observation_id"])
    if set(derivations["observation_id"]) != derived_ids:
        raise RuntimeError("Derived observation IDs do not match derivation records")
    for _, derivation in derivations.iterrows():
        source_id = clean_text(derivation["source_observation_id"])
        if source_id not in sources.index:
            raise RuntimeError("Derived free energy has an orphan source observation")
        source = sources.loc[source_id]
        if not (
            clean_text(source["endpoint"]).casefold() == "kd"
            and source["relation"] == "="
            and source["canonical_unit"] == "nM"
            and source["inclusion_status"] == "included"
            and float(source["canonical_value"]) > 0
        ):
            raise RuntimeError("Derived free energy source is not an included exact positive Kd")
        payload = {
            "source_observation_id": source["observation_id"],
            "source_record_id": source["source_record_id"],
            "source_snapshot_id": source["snapshot_id"],
            "source_relation": source["relation"],
            "source_kd_value": float(source["canonical_value"]),
            "source_kd_unit": source["canonical_unit"],
            "formula": derivation["formula"],
            "R": float(derivation["gas_constant_kcal_mol_k"]),
            "temperature_k": float(derivation["temperature_k"]),
            "temperature_source": derivation["temperature_source"],
            "standard_state": derivation["standard_state"],
        }
        expected_digest = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
        if expected_digest != clean_text(derivation["label_lineage_digest"]):
            raise RuntimeError("Derived free-energy lineage digest mismatch")
        kd_molar = float(source["canonical_value"]) * 1e-9
        roundtrip = math.exp(
            float(derivation["delta_g_kcal_mol"])
            / (GAS_CONSTANT_KCAL_MOL_K * float(derivation["temperature_k"]))
        )
        if abs(roundtrip - kd_molar) / kd_molar > 1e-12:
            raise RuntimeError("Derived free-energy round-trip tolerance exceeded")


def _unified_partition_schema(paths: list[Path]) -> pa.Schema:
    """Apply normative dictionary types to one logical shard dataset."""

    if not paths:
        raise RuntimeError("Cannot declare an Arrow schema for an empty partition set")
    physical_schemas = [pq.ParquetFile(path).schema_arrow.remove_metadata() for path in sorted(paths)]
    if any(len(schema.names) != len(set(schema.names)) for schema in physical_schemas):
        raise RuntimeError("Canonical partitions cannot contain duplicate Arrow fields")
    field_names = set().union(*(schema.names for schema in physical_schemas))
    unknown_fields = sorted(field_names - set(_NORMATIVE_ARROW_TYPES))
    if unknown_fields:
        raise RuntimeError(f"Canonical fields lack normative Arrow type declarations: {unknown_fields}")
    fields = [pa.field(name, _NORMATIVE_ARROW_TYPES[name], nullable=True) for name in sorted(field_names)]
    return pa.schema(fields)


def _compatible_physical_type(source: pa.DataType, target: pa.DataType) -> bool:
    """Allow only lossless physical type families before the safe Arrow cast."""

    if source.equals(target):
        return True
    if pa.types.is_large_string(target):
        return pa.types.is_string(source) or pa.types.is_large_string(source)
    if pa.types.is_floating(target):
        return pa.types.is_integer(source) or pa.types.is_floating(source)
    if pa.types.is_integer(target):
        return pa.types.is_integer(source) or pa.types.is_floating(source)
    if pa.types.is_boolean(target):
        return pa.types.is_boolean(source)
    return False


def _assert_no_symlink_path(build_root: Path, path: Path) -> None:
    """Reject lexical escape and every symlink in a private-build path."""

    try:
        relative_path = path.relative_to(build_root)
    except ValueError as error:
        raise RuntimeError(f"Canonical partition escapes build root: {path}") from error
    cursor = build_root
    if cursor.is_symlink():
        raise RuntimeError(f"Canonical build path contains a symlink: {cursor}")
    for part in relative_path.parts:
        cursor /= part
        if cursor.is_symlink():
            raise RuntimeError(f"Canonical build path contains a symlink: {cursor}")


def _assert_regular_build_file(build_root: Path, path: Path) -> None:
    _assert_no_symlink_path(build_root, path)
    if not path.is_file():
        raise RuntimeError(f"Canonical partition is not a regular file: {path}")
    if path.stat().st_nlink != 1:
        raise RuntimeError(f"Canonical partition cannot be hard-linked: {path}")


def _normalize_partition_records(
    building: Path,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Normalize private-build shards; only the enclosing directory is published.

    A commit interruption may make this private ``.building`` tree internally
    inconsistent.  The materializer refuses to resume such a tree, and only a
    later whole-directory rename can expose a QC-accepted canonical build.
    """

    if building.is_symlink() or not building.is_dir():
        raise RuntimeError("Canonical build root must be a real directory")
    build_root = building.resolve()
    paths: list[Path] = []
    for record in records:
        raw_path = record.get("relative_path")
        if not isinstance(raw_path, str):
            raise RuntimeError("Canonical partition path must be a string")
        text_path = clean_text(raw_path)
        relative_path = Path(text_path)
        if (
            not text_path
            or raw_path != text_path
            or "\\" in text_path
            or any(ord(character) < 32 or ord(character) == 127 for character in text_path)
            or relative_path.is_absolute()
            or relative_path.as_posix() != text_path
            or any(part in {"", ".", ".."} for part in relative_path.parts)
            or relative_path.suffix != ".parquet"
        ):
            raise RuntimeError(f"Non-canonical partition path: {text_path}")
        path = build_root / relative_path
        _assert_regular_build_file(build_root, path)
        paths.append(path)
    if len(paths) != len(set(paths)):
        raise RuntimeError("Duplicate canonical partition path")
    schema = _unified_partition_schema(paths)
    contract = arrow_schema_contract(schema)
    temporary_paths = [path.with_suffix(".parquet.schema.part") for path in paths]
    if any(path.exists() or path.is_symlink() for path in temporary_paths):
        raise RuntimeError("Refusing to overwrite an existing schema-normalization staging file")
    try:
        for record, path, temporary in zip(records, paths, temporary_paths, strict=True):
            _assert_regular_build_file(build_root, path)
            _assert_no_symlink_path(build_root, temporary)
            if temporary.exists() or temporary.is_symlink():
                raise RuntimeError("Refusing to overwrite a schema-normalization staging file")
            old_table = pq.read_table(path)
            try:
                for name in old_table.column_names:
                    source_type = old_table.schema.field(name).type
                    target_type = schema.field(name).type
                    column = old_table[name]
                    if column.null_count != old_table.num_rows and not _compatible_physical_type(
                        source_type, target_type
                    ):
                        raise pa.ArrowTypeError(
                            f"non-null {name} values have incompatible physical type "
                            f"{source_type}; expected {target_type}"
                        )
                normalized = _coerce_arrow_table(
                    old_table,
                    schema,
                    allowed_missing=frozenset(schema.names),
                )
            except (pa.ArrowInvalid, pa.ArrowTypeError) as error:
                raise RuntimeError(f"Canonical values violate a normative Arrow type: {path}") from error
            pq.write_table(normalized, temporary, compression="zstd")
            staged = pq.read_table(temporary)
            metadata = pq.ParquetFile(temporary).metadata
            if not staged.schema.remove_metadata().equals(
                schema, check_metadata=True
            ) or not normalized.equals(staged, check_metadata=True):
                raise RuntimeError(f"Canonical partition schema rewrite changed values: {path}")
            if metadata is None or int(metadata.num_rows) != int(record.get("rows", -1)):
                raise RuntimeError(f"Canonical partition row count changed: {path}")
    except Exception:
        for temporary in temporary_paths:
            temporary.unlink(missing_ok=True)
        raise
    try:
        for record, path, temporary in zip(records, paths, temporary_paths, strict=True):
            _assert_regular_build_file(build_root, path)
            _assert_regular_build_file(build_root, temporary)
            os.replace(temporary, path)
            metadata = pq.ParquetFile(path).metadata
            if metadata is None:
                raise RuntimeError(f"Canonical partition metadata is missing: {path}")
            record.update(
                {
                    "rows": int(metadata.num_rows),
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                    "arrow_schema_sha256": contract["sha256"],
                }
            )
    finally:
        for temporary in temporary_paths:
            temporary.unlink(missing_ok=True)
    return contract


def _partition_dataset_manifest(
    records: list[dict[str, Any]],
    contract: dict[str, Any],
) -> dict[str, Any]:
    digest_payload = [
        {
            "path": part["relative_path"],
            "rows": int(part["rows"]),
            "sha256": part["sha256"],
            "arrow_schema_sha256": part["arrow_schema_sha256"],
        }
        for part in records
    ]
    return {
        "rows": sum(int(part["rows"]) for part in records),
        "part_count": len(records),
        "parts": records,
        "arrow_schema": contract,
        "dataset_sha256": hashlib.sha256(canonical_json(digest_payload).encode("utf-8")).hexdigest(),
    }


def _write_normative_singleton(
    frame: pd.DataFrame,
    path: Path,
    building: Path,
) -> dict[str, Any]:
    """Write one root Parquet with a normative schema bound into its manifest record."""

    record = _atomic_frame(frame, path)
    record["relative_path"] = path.relative_to(building).as_posix()
    contract = _normalize_partition_records(building, [record])
    record["arrow_schema"] = contract
    return record


def _write_registry_dataset(
    registry: _DiskRegistry,
    dataset: str,
    building: Path,
    *,
    chunk_size: int = 100_000,
) -> dict[str, Any]:
    parts: list[dict[str, Any]] = []
    for part_number, frame in enumerate(registry.iter_frames(dataset, chunk_size=chunk_size)):
        _validate_frame(dataset, frame)
        record = _atomic_frame(
            frame,
            building / dataset / f"part-{part_number:05d}.parquet",
        )
        record["relative_path"] = (
            (building / dataset / f"part-{part_number:05d}.parquet").relative_to(building).as_posix()
        )
        parts.append(record)
    contract = _normalize_partition_records(building, parts)
    return _partition_dataset_manifest(parts, contract)


def _write_linked_development_metadata(
    registry: _DiskRegistry,
    manifest: dict[str, Any],
    interim_root: Path,
    building: Path,
    *,
    snapshot_id: str,
) -> dict[str, Any]:
    """Link metadata-only development annotations to retained molecule entities."""

    root = interim_root / "chembl_37_bulk" / "specialized_views"
    parts: list[dict[str, Any]] = []
    input_rows = 0
    linked_rows = 0
    for part_number, source_part in enumerate(manifest.get("parts", [])):
        source_path = root / str(source_part["path"])
        frame = pd.read_parquet(source_path)
        if len(frame) != int(source_part["rows"]):
            raise RuntimeError(f"Development metadata row-count mismatch: {source_path}")
        input_rows += len(frame)
        mapping = registry.resolve_molecules(frame["molecule_chembl_id"])
        frame["molecule_id"] = frame["molecule_chembl_id"].map(mapping).fillna("")
        linked = frame[frame["molecule_id"].map(clean_text).ne("")].copy()
        if linked.empty:
            continue
        linked["development_metadata_id"] = [
            stable_id("DEVMETA", snapshot_id, source_id) for source_id in linked["molecule_chembl_id"]
        ]
        linked["source_id"] = CHEMBL_SOURCE_ID
        linked["snapshot_id"] = snapshot_id
        linked["semantic_role"] = "development_metadata_not_outcome_or_model_label"
        if linked["development_metadata_id"].duplicated().any():
            raise RuntimeError("Duplicate development metadata IDs within an input part")
        output_path = building / "molecule_development_annotations" / f"part-{part_number:05d}.parquet"
        record = _atomic_frame(linked, output_path)
        record["relative_path"] = output_path.relative_to(building).as_posix()
        parts.append(record)
        linked_rows += len(linked)
    contract = _normalize_partition_records(building, parts)
    dataset = _partition_dataset_manifest(parts, contract)
    return {
        "input_rows": input_rows,
        "linked_rows": linked_rows,
        "unlinked_rows": input_rows - linked_rows,
        "part_count": dataset["part_count"],
        "parts": dataset["parts"],
        "arrow_schema": dataset["arrow_schema"],
        "dataset_sha256": dataset["dataset_sha256"],
        "semantic_role": "development_metadata_not_outcome_or_model_label",
        "prohibited_use": "not a clinical result, efficacy outcome, safety outcome, or default model label",
    }


def _write_shard(
    frame: pd.DataFrame,
    path: Path,
    artifacts: list[dict[str, Any]],
) -> dict[str, Any]:
    record = _atomic_frame(frame, path)
    build_root = next(parent for parent in path.parents if parent.name.endswith(".building"))
    record["relative_path"] = path.relative_to(build_root).as_posix()
    artifacts.append(record)
    return record


def materialize_chembl37_specialized_canonical(
    raw_root: str | os.PathLike[str],
    interim_root: str | os.PathLike[str],
    canonical_root: str | os.PathLike[str],
    reports_root: str | os.PathLike[str],
) -> dict[str, Any]:
    """Stream every completed specialized part into a large canonical corpus."""

    raw = Path(raw_root).resolve()
    interim = Path(interim_root).resolve()
    destination = Path(canonical_root).resolve() / "full_chembl37"
    building = destination.parent / f".{destination.name}.building"
    if destination.exists():
        manifest, _ = _load_build_manifest(destination)
        _run_bound_qc(destination, Path(reports_root).resolve())
        return manifest
    if building.exists():
        raise RuntimeError(
            f"Incomplete bulk canonical build exists; inspect before retrying: {building.name}"
        )
    building.mkdir(parents=True)

    input_parts, input_summary, component_path, development_manifest = _load_input_parts(interim)
    target_metadata = _component_metadata(pd.read_parquet(component_path))
    registry_all = source_registry(raw)
    bulk_sources = registry_all[
        registry_all["source_record_scope"].str.startswith("full official", na=False)
        & registry_all["access_class"].eq(PUBLIC_ACCESS_CLASS)
    ]
    if len(bulk_sources) != 1:
        raise RuntimeError("Exactly one rights-verified full ChEMBL_37 snapshot is required")
    source_row = bulk_sources.iloc[0]
    snapshot_id = clean_text(source_row["snapshot_id"])
    files = source_file_inventory(raw, registry_all)
    bulk_source_files, database_file_id = _select_bulk_source_files(files, snapshot_id)

    disk_registry = _DiskRegistry(building / ".canonical_registry.sqlite")
    task_registry_accumulator = _TaskRegistryAccumulator()
    artifacts: list[dict[str, Any]] = []
    task_dataset_parts: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    task_slugs: dict[str, str] = {}
    inventory_membership_counts = {name: 0 for name in _VIEW_ORDER}
    unique_rows = 0
    derived_rows = 0
    default_task_rows = 0
    sensitivity_task_rows = 0
    readiness_stage_counts = {scope: Counter[str]() for scope in ("default", "derived_sensitivity")}
    readiness_reason_counts = {scope: Counter[str]() for scope in ("default", "derived_sensitivity")}
    readiness_combination_counts = {scope: Counter[str]() for scope in ("default", "derived_sensitivity")}
    readiness_dimension_counts: dict[str, dict[str, dict[str, Counter[str]]]] = {
        scope: {
            dimension: {stage: Counter[str]() for stage in ("candidate", "eligible", "excluded")}
            for dimension in (
                "task_type",
                "source_id",
                "protein_id",
                "canonical_target_id",
            )
        }
        for scope in ("default", "derived_sensitivity")
    }
    readiness_exclusion_parts: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    maximum_input_part_rows = 0
    inclusion_counts: Counter[str] = Counter()
    exclusion_reason_counts: Counter[str] = Counter()
    standard_flag_counts: Counter[str] = Counter()
    potential_duplicate_rows = 0
    validity_comment_rows = 0

    def admit_model_ready_tasks(
        candidates: pd.DataFrame,
        *,
        scope: str,
        part_number: int,
    ) -> pd.DataFrame:
        eligible, exclusions = _partition_model_ready_tasks(
            candidates,
            task_scope=scope,
        )
        readiness_stage_counts[scope].update(
            {
                "candidate": len(candidates),
                "eligible": len(eligible),
                "excluded": len(exclusions),
            }
        )
        for dimension in (
            "task_type",
            "source_id",
            "protein_id",
            "canonical_target_id",
        ):
            readiness_dimension_counts[scope][dimension]["candidate"].update(
                candidates[dimension].map(clean_text)
            )
            readiness_dimension_counts[scope][dimension]["eligible"].update(
                eligible[dimension].map(clean_text)
            )
            readiness_dimension_counts[scope][dimension]["excluded"].update(
                exclusions[dimension].map(clean_text)
            )
        for combination in exclusions.get(
            "model_readiness_exclusion_reason",
            pd.Series(dtype="object"),
        ).map(clean_text):
            readiness_combination_counts[scope][combination] += 1
            for reason in combination.split(";"):
                if reason:
                    readiness_reason_counts[scope][reason] += 1
        if not exclusions.empty:
            record = _write_shard(
                exclusions,
                building / "task_exclusions" / scope / f"part-{part_number:05d}.parquet",
                artifacts,
            )
            record["task_scope"] = scope
            readiness_exclusion_parts[scope].append(record)
        return eligible

    for part_number, part in enumerate(input_parts):
        frame = pd.read_parquet(part["path"])
        if len(frame) != part["rows"]:
            raise RuntimeError(f"Specialized input row-count mismatch: {part['path']}")
        maximum_input_part_rows = max(maximum_input_part_rows, len(frame))
        activity_ids = pd.to_numeric(frame["activity_id"], errors="raise").astype("int64")
        inventory_membership_counts[part["view_name"]] += len(frame)
        keep = disk_registry.claim_activity_ids(activity_ids, part["view_name"])
        frame = frame.loc[keep].copy()
        activity_ids = activity_ids.loc[keep]
        if frame.empty:
            continue
        unique_rows += len(frame)
        prepared = _prepare_bulk_rows(
            frame,
            snapshot_id=snapshot_id,
            source_file_id=database_file_id,
        )
        proteins, target_map, constructs = _bulk_protein_entities(
            prepared,
            target_metadata,
        )
        molecules, aliases, molecule_map, conflicting = _molecule_entities(
            prepared,
            require_rdkit=False,
        )
        assays, assay_map = _assay_entities(prepared, target_map)
        observations = _observations(
            prepared,
            molecule_map,
            conflicting,
            target_map,
            assay_map,
            assays,
            require_rdkit=False,
        )
        observations["chembl_activity_id"] = activity_ids.to_numpy()
        inclusion_counts.update(observations["inclusion_status"].map(clean_text))
        for reasons in observations["exclusion_reason"].map(clean_text):
            exclusion_reason_counts.update(reason for reason in reasons.split(";") if reason)
        standard_flag_counts.update(
            clean_text(value) or "<blank>" for value in observations["chembl_standard_flag"]
        )
        potential_duplicate_rows += int(observations["chembl_potential_duplicate"].astype(bool).sum())
        validity_comment_rows += int(
            observations["chembl_data_validity_comment"].map(clean_text).ne("").sum()
        )
        default_tasks = _task_view(observations, assays)
        derivations, derived_observations = binding_free_energy_view(observations, assays, proteins)
        _validate_derivation_bundle(observations, derivations, derived_observations)
        sensitivity_tasks = (
            _task_view(
                derived_observations,
                assays,
                derived_sensitivity=True,
            )
            if not derived_observations.empty
            else pd.DataFrame()
        )
        derived_rows += len(derived_observations)
        for dataset, key, entity_frame in (
            ("molecules", "molecule_id", molecules),
            ("molecule_aliases", "molecule_alias_id", aliases),
            ("proteins", "protein_id", proteins),
            ("protein_constructs", "construct_id", constructs),
            ("assays", "assay_id", assays),
        ):
            disk_registry.add_frame(dataset, key, entity_frame)

        if not derived_observations.empty:
            activity_lookup = observations.set_index("observation_id")["chembl_activity_id"].to_dict()
            source_lookup = derivations.set_index("observation_id")["source_observation_id"].to_dict()
            derived_observations["chembl_activity_id"] = [
                activity_lookup[source_lookup[observation_id]]
                for observation_id in derived_observations["observation_id"]
            ]
            shard_observations = pd.concat(
                [observations, derived_observations],
                ignore_index=True,
                sort=False,
            )
        else:
            shard_observations = observations
        _validate_frame("observations", shard_observations)
        obs_path = building / "observations" / f"part-{part_number:05d}.parquet"
        _write_shard(shard_observations, obs_path, artifacts)
        lineage = pd.DataFrame(
            {
                "lineage_id": [
                    stable_id("LINEAGE", obs, database_file_id) for obs in observations["observation_id"]
                ],
                "observation_id": observations["observation_id"],
                "source_id": CHEMBL_SOURCE_ID,
                "snapshot_id": snapshot_id,
                "source_file_id": database_file_id,
                "lineage_role": "primary",
            }
        )
        if not derived_observations.empty:
            derived_lineage = pd.DataFrame(
                {
                    "lineage_id": [
                        stable_id("LINEAGE", observation_id, database_file_id)
                        for observation_id in derived_observations["observation_id"]
                    ],
                    "observation_id": derived_observations["observation_id"],
                    "source_id": CHEMBL_SOURCE_ID,
                    "snapshot_id": snapshot_id,
                    "source_file_id": database_file_id,
                    "lineage_role": "derived_support",
                }
            )
            lineage = pd.concat([lineage, derived_lineage], ignore_index=True)
        _validate_frame("observation_lineage", lineage)
        _validate_observation_lineage(
            shard_observations,
            lineage,
            bulk_sources,
            bulk_source_files,
        )
        _write_shard(lineage, building / "observation_lineage" / f"part-{part_number:05d}.parquet", artifacts)
        if not derivations.empty:
            _write_shard(
                derivations,
                building / "views" / "binding_free_energy_standard" / f"part-{part_number:05d}.parquet",
                artifacts,
            )
            _write_shard(
                derived_observations,
                building / "derived_observations" / f"part-{part_number:05d}.parquet",
                artifacts,
            )
        if not default_tasks.empty:
            model_tasks = _join_model_inputs(default_tasks, molecules, proteins, assays, derivations)
            _validate_frame("tasks", model_tasks)
            model_tasks = admit_model_ready_tasks(
                model_tasks,
                scope="default",
                part_number=part_number,
            )
            task_registry_accumulator.add(model_tasks)
            default_task_rows += len(model_tasks)
            for task_type, task_frame in model_tasks.groupby("task_type", sort=True):
                slug = re.sub(r"[^a-z0-9]+", "_", task_type.casefold()).strip("_")
                slug_key = f"default:{slug}"
                prior = task_slugs.setdefault(slug_key, clean_text(task_type))
                if prior != clean_text(task_type):
                    raise RuntimeError("Task-type slug collision")
                record = _write_shard(
                    task_frame,
                    building / "tasks" / "default" / slug / f"part-{part_number:05d}.parquet",
                    artifacts,
                )
                record["task_type"] = clean_text(task_type)
                record["task_scope"] = "default"
                task_dataset_parts[f"default::{task_type}"].append(record)
        if not sensitivity_tasks.empty:
            model_sensitivity = _join_model_inputs(
                sensitivity_tasks,
                molecules,
                proteins,
                assays,
                derivations,
            )
            _validate_frame("tasks", model_sensitivity)
            model_sensitivity = admit_model_ready_tasks(
                model_sensitivity,
                scope="derived_sensitivity",
                part_number=part_number,
            )
            task_registry_accumulator.add(model_sensitivity)
            sensitivity_task_rows += len(model_sensitivity)
            for task_type, task_frame in model_sensitivity.groupby("task_type", sort=True):
                slug = re.sub(r"[^a-z0-9]+", "_", task_type.casefold()).strip("_")
                slug_key = f"derived_sensitivity:{slug}"
                prior = task_slugs.setdefault(slug_key, clean_text(task_type))
                if prior != clean_text(task_type):
                    raise RuntimeError("Sensitivity task-type slug collision")
                record = _write_shard(
                    task_frame,
                    building / "tasks" / "derived_sensitivity" / slug / f"part-{part_number:05d}.parquet",
                    artifacts,
                )
                record["task_type"] = clean_text(task_type)
                record["task_scope"] = "derived_sensitivity"
                task_dataset_parts[f"derived_sensitivity::{task_type}"].append(record)

    disk_registry.assert_consistent()
    shard_groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in artifacts:
        relative_path = Path(clean_text(record.get("relative_path")))
        shard_groups[relative_path.parent.as_posix()].append(record)
    shard_dataset_schemas = {
        group: _normalize_partition_records(
            building,
            sorted(
                records,
                key=lambda record: clean_text(record.get("relative_path")),
            ),
        )
        for group, records in sorted(shard_groups.items())
    }
    model_readiness_exclusion_datasets: dict[str, dict[str, Any]] = {}
    for scope, parts in sorted(readiness_exclusion_parts.items()):
        ordered_parts = sorted(
            parts,
            key=lambda record: clean_text(record.get("relative_path")),
        )
        group_names = {
            Path(clean_text(record["relative_path"])).parent.as_posix() for record in ordered_parts
        }
        if len(group_names) != 1:
            raise RuntimeError("Model-readiness exclusions span multiple scope directories")
        contract = shard_dataset_schemas[next(iter(group_names))]
        model_readiness_exclusion_datasets[scope] = _partition_dataset_manifest(
            ordered_parts,
            contract,
        )

    readiness_dimensions_document: dict[str, Any] = {}
    for scope, dimensions in readiness_dimension_counts.items():
        readiness_dimensions_document[scope] = {}
        for dimension, stages in dimensions.items():
            keys = sorted(set().union(*(set(counter) for counter in stages.values())))
            readiness_dimensions_document[scope][dimension] = {
                key: {
                    stage: int(stages[stage].get(key, 0)) for stage in ("candidate", "eligible", "excluded")
                }
                for key in keys
            }
            if any(
                counts["candidate"] != counts["eligible"] + counts["excluded"]
                for counts in readiness_dimensions_document[scope][dimension].values()
            ):
                raise RuntimeError("Model-readiness dimension counts do not conserve rows")
    for _scope, counts in readiness_stage_counts.items():
        if counts["candidate"] != counts["eligible"] + counts["excluded"]:
            raise RuntimeError("Model-readiness stage counts do not conserve rows")
    model_readiness_policy = {
        "policy_version": _MODEL_READINESS_POLICY_VERSION,
        "allowed_modality_declarations": sorted(_ALLOWED_MODALITY_DECLARATIONS),
        "reason_order": [reason for reason, _, _ in _MODEL_READINESS_REASONS],
        "stage_counts": {
            scope: {stage: int(counts[stage]) for stage in ("candidate", "eligible", "excluded")}
            for scope, counts in readiness_stage_counts.items()
        },
        "reason_counts": {
            scope: {reason: int(count) for reason, count in sorted(counts.items())}
            for scope, counts in readiness_reason_counts.items()
        },
        "reason_combination_counts": {
            scope: {reason: int(count) for reason, count in sorted(counts.items())}
            for scope, counts in readiness_combination_counts.items()
        },
        "dimension_counts": readiness_dimensions_document,
        "exclusion_artifact_root": "task_exclusions",
        "evidence_layer_policy": (
            "source observations and lineage remain unchanged; only model-task admission is gated"
        ),
    }
    _validate_frame("sources", bulk_sources.reset_index(drop=True))
    _validate_frame("source_files", bulk_source_files)
    entity_datasets = {
        dataset: _write_registry_dataset(disk_registry, dataset, building)
        for dataset in (
            "molecules",
            "molecule_aliases",
            "proteins",
            "protein_constructs",
            "assays",
        )
    }
    entity_artifacts = {
        "sources": _write_normative_singleton(
            bulk_sources.reset_index(drop=True),
            building / "sources.parquet",
            building,
        ),
        "source_files": _write_normative_singleton(
            bulk_source_files,
            building / "source_files.parquet",
            building,
        ),
    }
    task_registry = task_registry_accumulator.frame()
    if task_registry.empty or task_registry["task_id"].duplicated().any():
        raise RuntimeError("Task registry is empty or has duplicate task IDs")
    if int(task_registry["row_count"].sum()) != default_task_rows + sensitivity_task_rows:
        raise RuntimeError("Task registry row counts do not reconcile to emitted task rows")
    entity_artifacts["task_registry"] = _write_normative_singleton(
        task_registry,
        building / "task_registry.parquet",
        building,
    )
    _atomic_frame(task_registry, building / "task_registry.csv")
    task_datasets: dict[str, Any] = {}
    for dataset_key, parts in sorted(task_dataset_parts.items()):
        scope, task_type = dataset_key.split("::", maxsplit=1)
        ordered_parts = sorted(parts, key=lambda record: clean_text(record["relative_path"]))
        parent_groups = {
            Path(clean_text(record["relative_path"])).parent.as_posix() for record in ordered_parts
        }
        if len(parent_groups) != 1:
            raise RuntimeError("One logical task dataset spans multiple shard directories")
        parent_group = next(iter(parent_groups))
        task_schema = shard_dataset_schemas[parent_group]
        digest_payload = [
            {
                "path": record["relative_path"],
                "rows": int(record["rows"]),
                "sha256": record["sha256"],
                "arrow_schema_sha256": record["arrow_schema_sha256"],
            }
            for record in ordered_parts
        ]
        task_datasets[dataset_key] = {
            "task_scope": scope,
            "task_type": task_type,
            "row_count": sum(record["rows"] for record in ordered_parts),
            "part_count": len(ordered_parts),
            "parts": digest_payload,
            "arrow_schema": task_schema,
            "dataset_sha256": hashlib.sha256(canonical_json(digest_payload).encode("utf-8")).hexdigest(),
        }
    emitted_default = sum(
        int(record["row_count"]) for record in task_datasets.values() if record["task_scope"] == "default"
    )
    emitted_sensitivity = sum(
        int(record["row_count"])
        for record in task_datasets.values()
        if record["task_scope"] == "derived_sensitivity"
    )
    if (emitted_default, emitted_sensitivity) != (
        default_task_rows,
        sensitivity_task_rows,
    ):
        raise RuntimeError("Task dataset manifests do not reconcile to task accumulators")
    _atomic_json(building / "task_datasets.json", task_datasets)
    development_artifact = _write_linked_development_metadata(
        disk_registry,
        development_manifest,
        interim,
        building,
        snapshot_id=snapshot_id,
    )
    proteins_with_sequence = 0
    for protein_chunk in disk_registry.iter_frames("proteins"):
        proteins_with_sequence += int(protein_chunk["sequence"].map(clean_text).ne("").sum())
    registry_path = disk_registry.path
    disk_registry.close()
    registry_path.unlink(missing_ok=True)
    registry_path.with_name(f"{registry_path.name}-wal").unlink(missing_ok=True)
    registry_path.with_name(f"{registry_path.name}-shm").unlink(missing_ok=True)
    _atomic_frame(data_dictionary_frame(), building / "data_dictionary.csv")
    _atomic_json(building / "schema.json", schema_document())

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "build_type": "public_chembl37_full_specialized_canonical",
        "built_at_utc": _utc_now(),
        "source_id": CHEMBL_SOURCE_ID,
        "snapshot_id": snapshot_id,
        "rights_gate": "public_redistributable ChEMBL_37 only",
        "input_summary": input_summary,
        "inventory_membership_counts_before_cross_view_dedup": inventory_membership_counts,
        "unique_activity_rows": unique_rows,
        "derived_binding_free_energy_rows": derived_rows,
        "canonical_attrition": {
            "inclusion_status_counts": dict(sorted(inclusion_counts.items())),
            "exclusion_reason_counts": dict(sorted(exclusion_reason_counts.items())),
            "chembl_standard_flag_counts": dict(sorted(standard_flag_counts.items())),
            "chembl_potential_duplicate_rows": potential_duplicate_rows,
            "chembl_data_validity_comment_rows": validity_comment_rows,
        },
        "entity_counts": {
            "molecules": entity_datasets["molecules"]["rows"],
            "molecule_aliases": entity_datasets["molecule_aliases"]["rows"],
            "proteins": entity_datasets["proteins"]["rows"],
            "protein_constructs": entity_datasets["protein_constructs"]["rows"],
            "assays": entity_datasets["assays"]["rows"],
            "tasks": default_task_rows,
            "sensitivity_tasks": sensitivity_task_rows,
        },
        "protein_sequence_coverage": {
            "with_sequence": proteins_with_sequence,
            "total": entity_datasets["proteins"]["rows"],
            "complex_parent_policy": "parent sequence blank; ordered component entities/constructs retained",
        },
        "entity_artifacts": entity_artifacts,
        "entity_datasets": entity_datasets,
        "molecule_development_annotations": development_artifact,
        "shard_artifacts": artifacts,
        "shard_dataset_schemas": shard_dataset_schemas,
        "model_readiness_policy": model_readiness_policy,
        "model_readiness_exclusion_datasets": model_readiness_exclusion_datasets,
        "task_datasets": task_datasets,
        "task_datasets_manifest_sha256": hashlib.sha256(
            canonical_json(task_datasets).encode("utf-8")
        ).hexdigest(),
        "task_registry_rows": len(task_registry),
        "qc_acceptance": {
            "required_before_promotion": True,
            "report_path": "qc_report.json",
            "binding": "qc_report.build_manifest_sha256 == SHA256(build_manifest.json)",
        },
        "operational_scale_contract": {
            "maximum_input_part_rows": maximum_input_part_rows,
            "global_identity_and_activity_state": "disk-backed SQLite with fail-closed collisions",
            "entity_output": "partitioned Parquet; no all-entity dataframe concat",
            "task_registry_state": "bounded by distinct task signatures",
        },
        "conflict_detection_boundary": (
            "activity-ID overlap deduplication is global; 10-fold biological repeat conflict IDs are "
            "shard-local candidates and must be recomputed globally by QC before analytical claims"
        ),
        "default_policy": bulk_canonicalization_contract()["scientific_scope"]["default_tasks"],
    }
    manifest["component_inventory"] = _component_inventory(building)
    _atomic_json(building / "build_manifest.json", manifest)
    _promote_qc_accepted_build(
        building,
        destination,
        Path(reports_root).resolve(),
    )
    report_path = Path(reports_root).resolve() / "data_bulk_canonical_manifest.json"
    _atomic_json(report_path, manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Materialize the large ChEMBL_37 specialized canonical corpus"
    )
    parser.add_argument("--raw-root", type=Path, default=Path("research/data/platform/raw"))
    parser.add_argument("--interim-root", type=Path, default=Path("research/data/platform/interim"))
    parser.add_argument("--canonical-root", type=Path, default=Path("research/data/platform/canonical"))
    parser.add_argument("--reports-root", type=Path, default=Path("research/reports/platform"))
    arguments = parser.parse_args(argv)
    manifest = materialize_chembl37_specialized_canonical(
        arguments.raw_root,
        arguments.interim_root,
        arguments.canonical_root,
        arguments.reports_root,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "bulk_canonicalization_contract",
    "main",
    "materialize_chembl37_specialized_canonical",
]
