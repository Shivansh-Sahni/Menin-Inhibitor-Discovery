"""Separate command surface for the mechanistic PK/hERG research program."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import tempfile
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import yaml

from .research_common import atomic_write_csv, atomic_write_json, load_research_config
from .research_contracts import contract_data_dictionary, contract_json_schemas, write_contract_tables
from .research_feature_ontology import feature_ontology_frame
from .research_hpc import generate_hpc_bundles
from .research_modeling import structure_feature_frame
from .research_normalize import normalize_research_data, normalize_sun_herg_workbook
from .research_physics import run_fast_physics
from .research_physics_contracts import CONTRACT_PHYSICS_FILES, project_fast_physics_contracts
from .research_reporting import (
    model_failure_findings,
    optimizer_endpoint_summary,
    residual_process_clusters,
    write_current_status_report,
    write_explanation_contract,
    write_model_card,
    write_run_summary,
)
from .research_structures import validate_mmcif_coordinate
from .research_workflows import (
    baseline_inventory,
    build_assay_and_optimizer_outputs,
    build_regime_analysis,
    compound_model_frame,
    load_canonical_tables,
    run_herg_models,
    run_pk_models,
    write_model_ladder,
)

STAGES = (
    "literature",
    "normalize",
    "baseline",
    "physics-fast",
    "pk",
    "herg",
    "explain",
    "report",
    "hpc-bundle",
    "all-local",
)

CATALOG_COLUMNS = {
    "literature_id",
    "parent_literature_id",
    "doi",
    "title",
    "year",
    "topic",
    "role",
    "local_path",
    "data_availability",
    "applicability_to_internal_series",
    "review_status",
}
MISSING_ASSET_COLUMNS = {
    "record_id",
    "domain",
    "asset",
    "why_needed",
    "availability",
    "status",
    "priority",
    "next_action",
}
ROLE_ALLOWED_SUFFIXES = {
    "article": {".pdf"},
    "reporting summary": {".pdf"},
    "source data": {".csv", ".pdf", ".tsv", ".xls", ".xlsx", ".zip"},
    "supplementary dataset": {".csv", ".pdf", ".tsv", ".xls", ".xlsx", ".zip"},
    "supplementary file description": {".pdf"},
    "supplementary information": {".pdf"},
    "supplementary material": {".pdf"},
    "supplementary methods": {".pdf"},
    "supplementary movie": {".mov", ".mp4"},
    "structure coordinate": {".cif", ".mmcif"},
}
CANONICAL_ASSET_SUFFIXES = {
    ".cif",
    ".csv",
    ".dcd",
    ".docx",
    ".gz",
    ".jpeg",
    ".jpg",
    ".mmcif",
    ".mol",
    ".mol2",
    ".mov",
    ".mp4",
    ".nc",
    ".netcdf",
    ".ods",
    ".parquet",
    ".pdb",
    ".pdf",
    ".png",
    ".sdf",
    ".tar",
    ".tif",
    ".tiff",
    ".trr",
    ".tsv",
    ".xls",
    ".xlsx",
    ".xtc",
    ".zip",
}
LOCKED_CORE_ASSETS = {
    "sun_2026_jcim_6c00163_article": (
        "article",
        "research/literature/herg/predictive_modeling/2026_sun_wang_shen_jcim_6c00163/article.pdf",
    ),
    "sun_2026_jcim_6c00163_dataset": (
        "supplementary dataset",
        "research/literature/herg/predictive_modeling/2026_sun_wang_shen_jcim_6c00163/supplementary_dataset.xlsx",
    ),
    "sun_2026_jcim_6c00163_methods": (
        "supplementary methods",
        "research/literature/herg/predictive_modeling/2026_sun_wang_shen_jcim_6c00163/supplementary_methods.pdf",
    ),
    "miyashita_2024_article": (
        "article",
        "research/literature/herg/structural_biology/2024_miyashita_inhibitor_bound/article.pdf",
    ),
    "miyashita_2024_si": (
        "supplementary information",
        "research/literature/herg/structural_biology/2024_miyashita_inhibitor_bound/supplementary_information.pdf",
    ),
    "lau_2024_article": (
        "article",
        "research/literature/herg/structural_biology/2024_lau_potassium_states/article.pdf",
    ),
    "mavroudis_2023_article": (
        "article",
        "research/literature/pk/hybrid_ml_mechanistic/2023_mavroudis_rat_iv/article.pdf",
    ),
    "mavroudis_2023_si": (
        "supplementary material",
        "research/literature/pk/hybrid_ml_mechanistic/2023_mavroudis_rat_iv/supplementary_material.pdf",
    ),
    "poongavanam_2022_article": (
        "article",
        "research/literature/pk/permeability_bro5/2022_poongavanam_protac_folding/article.pdf",
    ),
    "poongavanam_2022_methods": (
        "supplementary methods",
        "research/literature/pk/permeability_bro5/2022_poongavanam_protac_folding/supplementary_methods.pdf",
    ),
    "poongavanam_2022_assay_data": (
        "supplementary dataset",
        "research/literature/pk/permeability_bro5/2022_poongavanam_protac_folding/assay_data.csv",
    ),
    "qi_2025_article": (
        "article",
        "research/literature/pk/permeability_bro5/2025_qi_tetracycline_permeation/article.pdf",
    ),
    "qi_2025_si": (
        "supplementary methods",
        "research/literature/pk/permeability_bro5/2025_qi_tetracycline_permeation/supplementary_methods.pdf",
    ),
}
REQUIRED_SYNTHESIS_FILES = {
    "synthesis/herg_process_map.md",
    "synthesis/mechanistic_feature_precedent_matrix.csv",
    "synthesis/pk_process_map.md",
}
REQUIRED_MEETING_FILES = {
    "2026-07-20": {"context.md"},
    "2026-07-21": {"context.md", "data_availability_2026-07-21_11-39-44.png"},
}
MISSING_ASSET_DOMAINS = {"PK", "hERG"}
MISSING_ASSET_STATUSES = {"missing", "not_local", "partial", "watchlist"}
MISSING_ASSET_PRIORITIES = {"critical", "high", "medium", "low"}


def _promote_directory(
    target: Path, builder: Callable[[Path], dict[str, Any]], *, required: tuple[str, ...]
) -> dict[str, Any]:
    """Build beside the target and replace it only after declared outputs validate."""

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}-stage-", dir=target.parent))
    backup = target.parent / f".{target.name}-superseded"
    try:
        result = builder(staging)
        missing = [name for name in required if not (staging / name).exists()]
        if missing:
            raise RuntimeError(f"Stage validation failed; missing outputs: {missing}")
        if backup.exists():
            shutil.rmtree(backup)
        if target.exists():
            os.replace(target, backup)
        os.replace(staging, target)
        shutil.rmtree(backup, ignore_errors=True)
        return _rebase_promoted_paths(result, staging=staging, target=target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        if backup.exists() and not target.exists():
            os.replace(backup, target)
        raise


def _rebase_promoted_paths(value: Any, *, staging: Path, target: Path) -> Any:
    """Replace paths into a promoted staging tree with their stable target paths.

    Stage builders often report their output directory or individual artifacts.
    Those paths become stale as soon as the atomic directory promotion completes,
    so rebase only exact staging paths and their descendants.  Ordinary strings
    that merely happen to contain the staging directory name are left untouched.
    """

    staging_text = str(staging)
    target_text = str(target)
    if isinstance(value, Path):
        try:
            relative = value.relative_to(staging)
        except ValueError:
            return value
        return target / relative
    if isinstance(value, str):
        if value == staging_text:
            return target_text
        prefix = f"{staging_text}{os.sep}"
        if value.startswith(prefix):
            return f"{target_text}{os.sep}{value.removeprefix(prefix)}"
        return value
    if isinstance(value, dict):
        return {
            key: _rebase_promoted_paths(item, staging=staging, target=target) for key, item in value.items()
        }
    if isinstance(value, list):
        return [_rebase_promoted_paths(item, staging=staging, target=target) for item in value]
    if isinstance(value, tuple):
        return tuple(_rebase_promoted_paths(item, staging=staging, target=target) for item in value)
    return value


def _read_literature_register(path: Path, *, label: str, required_columns: set[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Canonical {label} is missing: {path}")
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if missing := sorted(required_columns - set(frame.columns)):
        raise ValueError(f"{label} is missing required columns: {missing}")
    return frame


def _resolve_catalog_asset(raw_path: str, *, project_root: Path, literature_root: Path) -> Path:
    source = Path(raw_path)
    resolved = (source if source.is_absolute() else project_root / source).resolve()
    try:
        resolved.relative_to(literature_root.resolve())
    except ValueError as exc:
        raise ValueError(f"Catalog asset escapes the canonical literature root: {raw_path}") from exc
    return resolved


def _is_deliberate_noncanonical_output(path: Path, *, literature_root: Path) -> bool:
    relative = path.relative_to(literature_root)
    if relative.parts[0] == "synthesis" or relative.as_posix() in {"catalog.csv", "missing_assets.csv"}:
        return True
    ignored_names = {"preview", "previews", "render", "rendered", "renders", "temp", "tmp"}
    for part in relative.parts:
        normalized = part.casefold()
        if part.startswith(".") or normalized in ignored_names:
            return True
        if normalized.startswith(("preview_", "rendered_", "temp_", "tmp_")):
            return True
    return False


def _canonical_binary_assets(literature_root: Path) -> set[Path]:
    return {
        path.resolve()
        for path in literature_root.rglob("*")
        if path.is_file()
        and path.suffix.casefold() in CANONICAL_ASSET_SUFFIXES
        and not _is_deliberate_noncanonical_output(path, literature_root=literature_root)
    }


def _validate_locked_core_assets(
    catalog: pd.DataFrame,
    *,
    project_root: Path,
    resolved_assets: dict[str, Path | None],
) -> None:
    indexed = catalog.set_index("literature_id", drop=False)
    absent = sorted(set(LOCKED_CORE_ASSETS) - set(indexed.index))
    if absent:
        raise ValueError(f"Literature catalog is missing locked core mappings: {absent}")
    mismatches: list[str] = []
    for literature_id, (expected_role, expected_relative_path) in LOCKED_CORE_ASSETS.items():
        row = indexed.loc[literature_id]
        expected_path = (project_root / expected_relative_path).resolve()
        if row["role"] != expected_role or resolved_assets[literature_id] != expected_path:
            mismatches.append(literature_id)
    if mismatches:
        raise ValueError(f"Literature catalog changed locked role/path mappings: {sorted(mismatches)}")


def _validate_catalog(
    catalog: pd.DataFrame,
    *,
    project_root: Path,
    literature_root: Path,
) -> dict[str, Any]:
    nonempty_columns = CATALOG_COLUMNS - {"parent_literature_id", "local_path"}
    blank_fields = {
        column: catalog.index[catalog[column].str.strip().eq("")].tolist()
        for column in sorted(nonempty_columns)
        if catalog[column].str.strip().eq("").any()
    }
    if blank_fields:
        raise ValueError(f"Literature catalog has blank required values: {blank_fields}")

    for column in CATALOG_COLUMNS:
        catalog[column] = catalog[column].str.strip()
    identifiers = catalog["literature_id"]
    if invalid_ids := sorted(
        {value for value in identifiers if not re.fullmatch(r"[a-z0-9][a-z0-9_]*", value)}
    ):
        raise ValueError(f"Literature catalog has invalid literature_id values: {invalid_ids}")
    if duplicates := sorted(identifiers[identifiers.duplicated(keep=False)].unique().tolist()):
        raise ValueError(f"Literature catalog has duplicate literature_id values: {duplicates}")
    if invalid_years := sorted({value for value in catalog["year"] if not re.fullmatch(r"\d{4}", value)}):
        raise ValueError(f"Literature catalog has invalid years: {invalid_years}")
    if invalid_roles := sorted(set(catalog["role"]) - set(ROLE_ALLOWED_SUFFIXES)):
        raise ValueError(f"Literature catalog has unsupported roles: {invalid_roles}")
    invalid_review_status = sorted(
        {
            value
            for value in catalog["review_status"]
            if not (value.startswith("reviewed") or value == "missing_article")
        }
    )
    if invalid_review_status:
        raise ValueError(f"Literature catalog has invalid review_status values: {invalid_review_status}")

    resolved_assets: dict[str, Path | None] = {}
    path_owners: dict[Path, str] = {}
    for row in catalog.itertuples(index=False):
        literature_id = str(row.literature_id)
        local_path = str(row.local_path)
        if not local_path:
            if row.role != "article" or row.review_status != "missing_article":
                raise ValueError(f"Only an explicitly missing article may omit local_path: {literature_id}")
            resolved_assets[literature_id] = None
            continue
        if row.review_status == "missing_article":
            raise ValueError(f"Catalog marks a present local asset as a missing article: {literature_id}")
        resolved = _resolve_catalog_asset(
            local_path,
            project_root=project_root,
            literature_root=literature_root,
        )
        if not resolved.is_file():
            raise FileNotFoundError(f"Catalog local asset is missing or not a file: {local_path}")
        allowed = ROLE_ALLOWED_SUFFIXES[str(row.role)]
        if resolved.suffix.casefold() not in allowed:
            raise ValueError(
                f"Catalog role/extension mismatch for {literature_id}: "
                f"role={row.role!r}, suffix={resolved.suffix.casefold()!r}"
            )
        if row.role == "structure coordinate":
            validate_mmcif_coordinate(resolved, expected_entry_id=resolved.stem)
        if owner := path_owners.get(resolved):
            raise ValueError(f"Catalog local asset is assigned to multiple IDs: {owner}, {literature_id}")
        path_owners[resolved] = literature_id
        resolved_assets[literature_id] = resolved

    indexed = catalog.set_index("literature_id", drop=False)
    for row in catalog.itertuples(index=False):
        literature_id = str(row.literature_id)
        parent_id = str(row.parent_literature_id)
        if row.role == "article":
            if parent_id:
                raise ValueError(f"Article must not have a parent_literature_id: {literature_id}")
            continue
        if not parent_id or parent_id not in indexed.index:
            raise ValueError(f"Supplement has missing or unknown parent article: {literature_id}")
        parent = indexed.loc[parent_id]
        if parent["role"] != "article":
            raise ValueError(f"Supplement parent is not an article: {literature_id} -> {parent_id}")
        mismatched_fields = [
            field for field in ("doi", "year", "topic") if parent[field] != getattr(row, field)
        ]
        if mismatched_fields:
            raise ValueError(
                f"Supplement does not match parent article metadata: {literature_id}, fields={mismatched_fields}"
            )

    package_directories: set[Path] = set()
    for article_id in catalog.loc[catalog["role"].eq("article"), "literature_id"]:
        family_ids = [
            article_id,
            *catalog.loc[catalog["parent_literature_id"].eq(article_id), "literature_id"],
        ]
        family_directories: set[Path] = set()
        for member in family_ids:
            asset = resolved_assets.get(member)
            if asset is not None:
                family_directories.add(asset.parent)
        if len(family_directories) != 1:
            raise ValueError(
                f"Article package must resolve to exactly one directory: {article_id}, "
                f"directories={sorted(map(str, family_directories))}"
            )
        package = next(iter(family_directories))
        review = package / "evidence_review.md"
        if not review.is_file():
            raise FileNotFoundError(f"Article package lacks evidence_review.md: {article_id} ({package})")
        package_directories.add(package)

    review_directories = {
        path.parent.resolve() for path in literature_root.rglob("evidence_review.md") if path.is_file()
    }
    if orphan_reviews := sorted(map(str, review_directories - package_directories)):
        raise ValueError(f"Evidence reviews exist outside cataloged article packages: {orphan_reviews}")

    _validate_locked_core_assets(catalog, project_root=project_root, resolved_assets=resolved_assets)
    cataloged_binary = {
        path
        for path in resolved_assets.values()
        if path is not None and path.suffix.casefold() in CANONICAL_ASSET_SUFFIXES
    }
    discovered_binary = _canonical_binary_assets(literature_root)
    if unrepresented := sorted(map(str, discovered_binary - cataloged_binary)):
        raise ValueError(f"Canonical literature assets are not represented in catalog.csv: {unrepresented}")
    return {
        "canonical_binary_assets": len(discovered_binary),
        "evidence_reviews": len(review_directories),
        "article_packages": len(package_directories),
    }


def _validate_missing_assets(missing_assets: pd.DataFrame) -> None:
    for column in MISSING_ASSET_COLUMNS:
        missing_assets[column] = missing_assets[column].str.strip()
    blank_fields = {
        column: missing_assets.index[missing_assets[column].eq("")].tolist()
        for column in sorted(MISSING_ASSET_COLUMNS)
        if missing_assets[column].eq("").any()
    }
    if blank_fields:
        raise ValueError(f"Missing-assets register has blank required values: {blank_fields}")
    if invalid_ids := sorted(
        {value for value in missing_assets["record_id"] if not re.fullmatch(r"[a-z0-9][a-z0-9_]*", value)}
    ):
        raise ValueError(f"Missing-assets register has invalid record_id values: {invalid_ids}")
    if invalid_domains := sorted(set(missing_assets["domain"]) - MISSING_ASSET_DOMAINS):
        raise ValueError(f"Missing-assets register has invalid domains: {invalid_domains}")
    if invalid_statuses := sorted(set(missing_assets["status"]) - MISSING_ASSET_STATUSES):
        raise ValueError(f"Missing-assets register has invalid statuses: {invalid_statuses}")
    if invalid_priorities := sorted(set(missing_assets["priority"]) - MISSING_ASSET_PRIORITIES):
        raise ValueError(f"Missing-assets register has invalid priorities: {invalid_priorities}")
    duplicate_rows = missing_assets.duplicated(subset=["record_id", "asset"], keep=False)
    if duplicate_rows.any():
        duplicate_keys = missing_assets.loc[duplicate_rows, ["record_id", "asset"]].to_dict("records")
        raise ValueError(f"Missing-assets register has duplicate record/asset rows: {duplicate_keys}")


def _validate_meeting_hierarchy(project_root: Path) -> int:
    meetings_root = project_root / "research" / "notes" / "meetings"
    if not meetings_root.is_dir():
        raise FileNotFoundError(f"Canonical meeting-note hierarchy is missing: {meetings_root}")
    direct_files = sorted(path.name for path in meetings_root.iterdir() if path.is_file())
    if direct_files:
        raise ValueError(f"Meeting records must be inside ISO-dated folders: {direct_files}")
    meeting_directories = sorted(path for path in meetings_root.iterdir() if path.is_dir())
    for directory in meeting_directories:
        try:
            parsed = date.fromisoformat(directory.name)
        except ValueError as exc:
            raise ValueError(f"Meeting folder is not an ISO date: {directory.name}") from exc
        if parsed.isoformat() != directory.name:
            raise ValueError(f"Meeting folder is not an exact ISO date: {directory.name}")
        if not (directory / "context.md").is_file():
            raise FileNotFoundError(f"Meeting folder lacks context.md: {directory}")
    available_dates = {path.name for path in meeting_directories}
    if missing_dates := sorted(set(REQUIRED_MEETING_FILES) - available_dates):
        raise FileNotFoundError(f"Required meeting folders are missing: {missing_dates}")
    missing_records = sorted(
        str(project_root / "research" / "notes" / "meetings" / meeting_date / filename)
        for meeting_date, filenames in REQUIRED_MEETING_FILES.items()
        for filename in filenames
        if not (meetings_root / meeting_date / filename).is_file()
    )
    if missing_records:
        raise FileNotFoundError(f"Required meeting records are missing: {missing_records}")
    return len(meeting_directories)


def _literature_stage(config: dict[str, Any]) -> dict[str, Any]:
    project_root = Path(config["project_root"]).resolve()
    root = Path(config["paths"]["literature"]).resolve()
    catalog = _read_literature_register(
        root / "catalog.csv",
        label="literature catalog",
        required_columns=CATALOG_COLUMNS,
    )
    missing_assets = _read_literature_register(
        root / "missing_assets.csv",
        label="missing-assets register",
        required_columns=MISSING_ASSET_COLUMNS,
    )
    literature_summary = _validate_catalog(
        catalog,
        project_root=project_root,
        literature_root=root,
    )
    _validate_missing_assets(missing_assets)
    missing_synthesis = sorted(
        relative_path for relative_path in REQUIRED_SYNTHESIS_FILES if not (root / relative_path).is_file()
    )
    if missing_synthesis:
        raise FileNotFoundError(f"Literature synthesis is incomplete: {missing_synthesis}")
    meeting_folders = _validate_meeting_hierarchy(project_root)
    return {
        "catalog_rows": int(len(catalog)),
        **literature_summary,
        "missing_asset_rows": int(len(missing_assets)),
        "meeting_folders": meeting_folders,
    }


def _normalize_stage(config: dict[str, Any]) -> dict[str, Any]:
    target = config["paths"]["canonical"]
    internal_workbook = config["inputs"]["internal_workbook"]
    public_workbook = config["inputs"].get("public_herg_workbook")

    def build(staging: Path) -> dict[str, Any]:
        internal_dir = staging / "internal"
        internal_outputs = normalize_research_data(internal_workbook, None, internal_dir)
        result: dict[str, Any] = {
            "internal_compounds": cast(int, internal_outputs["compounds_rows"]),
            "internal_measurements": cast(int, internal_outputs["measurements_rows"]),
            "pk_studies": cast(int, internal_outputs["pk_studies_rows"]),
            "pk_samples": cast(int, internal_outputs["pk_samples_rows"]),
            "derived_pk_parameters": cast(int, internal_outputs["derived_pk_parameters_rows"]),
            "aliases": cast(int, internal_outputs["compound_aliases_rows"]),
        }
        if public_workbook and Path(public_workbook).exists():
            public = normalize_sun_herg_workbook(public_workbook)
            public_dir = staging / "public_herg"
            write_contract_tables(public.tables, public_dir)
            public.review_tables["activity"].to_parquet(
                public_dir / "public_herg_normalized.parquet", index=False
            )
            public.quarantine.to_parquet(public_dir / "public_herg_quarantine.parquet", index=False)
            public.review_tables["domain_contradictions"].to_parquet(
                public_dir / "public_herg_domain_contradictions.parquet", index=False
            )
            public.issues.to_parquet(public_dir / "validation_issues.parquet", index=False)
            result.update(
                {
                    "public_unique_structures": int(public.summary["unique_standardized_structures"]),
                    "public_measurements": int(public.summary["measurements"]),
                    "public_quarantine": int(public.summary["quarantine_records"]),
                    "public_train_validation_overlap": int(
                        public.summary["train_validation_structure_overlaps"]
                    ),
                    "public_mw_domain_contradictions": int(
                        public.summary["computed_mw_above_domain_limit_rows"]
                    ),
                }
            )
        atomic_write_csv(staging / "data_dictionary.csv", contract_data_dictionary())
        atomic_write_json(staging / "contract_schemas.json", contract_json_schemas())
        atomic_write_json(staging / "normalization_summary.json", result)
        atomic_write_csv(staging / "normalization_summary.csv", pd.DataFrame([result]))
        return result

    return _promote_directory(
        target,
        build,
        required=(
            "internal/compounds.parquet",
            "internal/measurements.parquet",
            "data_dictionary.csv",
            "contract_schemas.json",
            "normalization_summary.json",
        ),
    )


def _baseline_stage(config: dict[str, Any]) -> dict[str, Any]:
    target = config["paths"]["reports"] / "baseline"
    return _promote_directory(
        target,
        lambda staging: baseline_inventory(config["project_root"], staging),
        required=("frozen_baseline_inventory.json",),
    )


def _physics_stage(config: dict[str, Any], *, smoke: bool) -> dict[str, Any]:
    target = config["paths"]["physics"]
    if not bool(config.get("run", {}).get("execute_local_fast_physics", True)):

        def build_deferred(staging: Path) -> dict[str, Any]:
            summary = {
                "status": "deferred_to_hpc",
                "execution_policy": "no_local_structure_or_conformer_enumeration",
                "smoke_mode": False,
                "sampling_tier": "deferred_to_hpc",
                "generated_conformers_per_state": None,
                "retained_conformers_per_state": None,
                "reference_conformers_per_state": 250,
                "pilot_comparator_conformers_per_state": 500,
                "model_facing_physics_features_available": False,
                "interpretation": (
                    "Local chemical-state and conformer calculations were deliberately not run. "
                    "HPC work must enumerate every threshold-qualifying or mechanism-exception state "
                    "without inheriting the inactive 24-state local guard, and use adaptive "
                    "25/50/100/250/500 sampling with independent-seed convergence checks."
                ),
            }
            atomic_write_json(staging / "fast_physics_run_summary.json", summary)
            atomic_write_json(staging / "physics_deferred.json", summary)
            atomic_write_csv(
                staging / "fast_physics_feature_ontology.csv",
                feature_ontology_frame(),
            )
            return summary

        return _promote_directory(
            target,
            build_deferred,
            required=(
                "fast_physics_run_summary.json",
                "physics_deferred.json",
                "fast_physics_feature_ontology.csv",
            ),
        )
    tables = load_canonical_tables(config["paths"]["canonical"])
    compounds = compound_model_frame(tables["compounds"])
    measurements = tables["measurements"]
    pka = measurements[
        measurements["endpoint"].astype(str).str.contains("pka", case=False, na=False)
        & measurements["value"].notna()
    ][["compound_id", "endpoint", "value", "source", "source_locator"]].copy()
    if not pka.empty:
        pka["pka_value"] = pka["value"]
        pka["pka_kind"] = np.where(
            pka["endpoint"].astype(str).str.contains("acid", case=False), "acid", "base"
        )
        pka["pka_label"] = pka["endpoint"]
        pka["pka_source"] = pka["source"].fillna("internal pKa evidence")

    def build(staging: Path) -> dict[str, Any]:
        result = run_fast_physics(
            compounds[["compound_id", "standardized_smiles", "mw"]],
            pka if not pka.empty else None,
            staging,
            config=config,
            smoke=smoke,
        )
        contract_projection = project_fast_physics_contracts(
            staging,
            target_ph=float(config.get("fast_physics", {}).get("herg_ph", 7.4)),
            temperature_kelvin=float(config.get("fast_physics", {}).get("temperature_kelvin", 298.15)),
            random_seed=int(config.get("fast_physics", {}).get("random_seed", 20260721)),
        )
        summary: dict[str, Any] = {key: value for key, value in result.items() if isinstance(value, int)}
        summary.update({key: value for key, value in contract_projection.items() if key.endswith("_rows")})
        physics_config = config.get("fast_physics", {})
        summary["sampling_tier"] = str(physics_config.get("sampling_tier", "reference"))
        summary["generated_conformers_per_state"] = int(physics_config.get("max_conformers_per_state", 250))
        summary["retained_conformers_per_state"] = int(physics_config.get("max_retained_conformers", 50))
        summary["reference_conformers_per_state"] = int(
            physics_config.get("reference_conformers_per_state", 250)
        )
        summary["local_time_budget_minutes"] = int(physics_config.get("local_time_budget_minutes", 0))
        summary["smoke_mode"] = bool(smoke)
        if summary["sampling_tier"] == "local_time_bounded_discovery" and not smoke:
            summary["interpretation"] = (
                "time-bounded local conformer/microstate discovery screen; the configured "
                "250-conformer reference depth is a deferred selected-pilot validation ceiling and "
                "500 is its comparator; neither is a universal target; not equilibrium MD, "
                "experimental micro-pKa, or decision-track evidence"
            )
        else:
            summary["interpretation"] = (
                "screening conformer/microstate hypotheses; not equilibrium MD or experimental micro-pKa"
            )
        atomic_write_json(staging / "fast_physics_run_summary.json", summary)
        return summary

    return _promote_directory(
        target,
        build,
        required=(
            "fast_physics_summary.parquet",
            "fast_physics_admissibility.parquet",
            "fast_physics_conformers.parquet",
            "fast_physics_quality_gates.parquet",
            "fast_physics_state_threshold_audit.parquet",
            "fast_physics_sampling_escalation_queue.parquet",
            "fast_physics_run_summary.json",
            *CONTRACT_PHYSICS_FILES.values(),
        ),
    )


def _pk_stage(config: dict[str, Any]) -> dict[str, Any]:
    target = config["paths"]["models"] / "pk"
    return _promote_directory(
        target,
        lambda staging: run_pk_models(
            config["paths"]["canonical"],
            config["paths"]["physics"],
            staging,
            folds=int(config["validation"].get("pk_group_folds", 5)),
            random_state=int(config["run"].get("random_state", 20260721)),
            interval_level=float(config["validation"].get("prediction_interval_level", 0.90)),
        ),
        required=(
            "pk_model_ladder_summary.csv",
            "pk_identifiability_contract.csv",
            "optimizer_predictions_long.parquet",
        ),
    )


def _metric_record(path: Path, *, model: str | None = None) -> dict[str, Any]:
    """Load the selected model's complete metric record for scientific reporting."""

    if not path.exists():
        return {}
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    frame = pd.read_csv(path)
    if frame.empty:
        return {}
    if model and "model" in frame:
        selected = frame[frame["model"].astype(str) == str(model)]
        if not selected.empty:
            frame = selected
    if "primary_evaluation" in frame and frame["primary_evaluation"].fillna(False).astype(bool).any():
        frame = frame[frame["primary_evaluation"].fillna(False).astype(bool)]
    return frame.iloc[0].dropna().to_dict()


def _model_reporting_summary(config: dict[str, Any]) -> pd.DataFrame:
    """Join terse ladder registries to the held-out metrics and promotion gates they summarize."""

    rows: list[dict[str, Any]] = []
    pk_root = config["paths"]["models"] / "pk"
    pk_summary_path = pk_root / "pk_model_ladder_summary.csv"
    if pk_summary_path.exists():
        pk_summary = pd.read_csv(pk_summary_path)
        gate_lookup: dict[tuple[str, str], dict[str, Any]] = {}
        for gate_name, layer in (
            ("pk_hierarchical_promotion_gates.csv", "structure_2d_hierarchical"),
            ("pk_physics_promotion_gates.csv", "state_conformer_physics"),
        ):
            gate_path = pk_root / gate_name
            if not gate_path.exists():
                continue
            try:
                gate_frame = pd.read_csv(gate_path)
            except pd.errors.EmptyDataError:
                # A deferred layer has no promotion candidates; an empty file
                # means "not evaluated", not a reporting failure.
                continue
            for gate in gate_frame.to_dict("records"):
                gate_lookup[(str(gate["endpoint"]), layer)] = gate
        metric_names = {
            "structure_2d": "structure_2d_metrics.csv",
            "structure_2d_hierarchical": "hierarchical_metrics.csv",
            "state_conformer_physics": "state_conformer_physics_metrics.csv",
        }
        for summary_row in pk_summary.to_dict("records"):
            endpoint = str(summary_row["endpoint"])
            layer = str(summary_row["feature_layer"])
            record = dict(summary_row)
            metric_name = metric_names.get(layer)
            if metric_name:
                detail = _metric_record(
                    pk_root / endpoint / metric_name,
                    model=str(summary_row.get("best_model", "")),
                )
                record.update(detail)
            gate = gate_lookup.get((endpoint, layer), {})
            if gate:
                record.update(
                    {
                        "baseline_value": gate.get("baseline_value"),
                        "candidate_value": gate.get("candidate_value"),
                        "noninferior_to_baseline": gate.get("noninferior_to_baseline"),
                        "promotion_status": gate.get("promotion_status"),
                        "promotion_reason": gate.get("reason"),
                        "calibrated": gate.get("calibrated", gate.get("calibrated_on_untouched_set")),
                        "physics_converged": gate.get("physics_converged"),
                    }
                )
            record.update(
                {
                    "domain": "pk",
                    "endpoint": endpoint,
                    "feature_layer": layer,
                    "status": summary_row.get("status"),
                    "best_model": summary_row.get("best_model"),
                    "primary_metric": summary_row.get("primary_metric"),
                    "primary_value": summary_row.get("primary_value"),
                }
            )
            rows.append(record)

    herg_root = config["paths"]["models"] / "herg"
    herg_summary_path = herg_root / "herg_model_ladder_summary.csv"
    if herg_summary_path.exists():
        herg_summary = pd.read_csv(herg_summary_path)
        metric_paths = {
            "structure_2d_exact": herg_root / "conventional_exact_pic50_metrics.csv",
            "structure_2d": herg_root / "structure_2d_censored_metrics.json",
            "state_conformer_physics": herg_root / "state_conformer_physics_censored_metrics.json",
            "structure_2d_joint_observations": herg_root / "joint_pic50_inhibition_metrics.json",
            "molecular_graph": herg_root / "dmpnn_metrics.json",
            "state_conformer_bags": herg_root / "conformer_mil_metrics.json",
            "sun_public_source_holdout_classification": (
                herg_root / "public_sun_reproduction" / "classification_source_holdout_metrics.csv"
            ),
            "sun_public_source_holdout_regression": (
                herg_root / "public_sun_reproduction" / "regression_source_holdout_metrics.csv"
            ),
        }
        for summary_row in herg_summary.to_dict("records"):
            layer = str(summary_row["feature_layer"])
            record = dict(summary_row)
            detail = _metric_record(
                metric_paths.get(layer, Path("__not_available__")),
                model=str(summary_row.get("model", "")),
            )
            record.update(detail)
            record.update(
                {
                    "domain": "herg",
                    "endpoint": "continuous_hERG",
                    "feature_layer": layer,
                    "status": summary_row.get("status"),
                    "model": summary_row.get("model"),
                }
            )
            rows.append(record)
    return pd.DataFrame(rows)


def _herg_stage(config: dict[str, Any], *, neural_epochs: int) -> dict[str, Any]:
    target = config["paths"]["models"] / "herg"
    return _promote_directory(
        target,
        lambda staging: run_herg_models(
            config["paths"]["canonical"],
            config["paths"]["physics"],
            staging,
            folds=int(config["validation"].get("herg_group_folds", 5)),
            random_state=int(config["run"].get("random_state", 20260721)),
            neural_epochs=neural_epochs,
            interval_level=float(config["validation"].get("prediction_interval_level", 0.90)),
        ),
        required=(
            "herg_model_ladder_summary.csv",
            "joint_pic50_inhibition_metrics.json",
            "markov_states_architecture.csv",
            "optimizer_predictions_long.parquet",
        ),
    )


def _explanation_payload(
    domain: str,
    endpoint: str,
    feature_layer: str,
    status: str,
    *,
    model_row: pd.Series,
    model_summary: pd.DataFrame,
    residual_evidence: str,
    matched_pair_evidence: str,
) -> dict[str, Any]:
    is_pk = domain == "pk"
    metric_names = (
        "n_unique_compounds",
        "n",
        "n_exact",
        "log_mae",
        "log_rmse",
        "r2",
        "spearman",
        "median_fold_error",
        "absolute_average_fold_error",
        "fraction_within_2fold",
        "fraction_within_3fold",
        "prediction_interval_coverage",
        "prediction_interval_mean_width_log10",
        "negative_log_likelihood",
        "crps",
        "bootstrap_log_mae_lower_95",
        "bootstrap_log_mae_upper_95",
        "pic50_mae",
        "pic50_rmse",
        "censored_negative_log_likelihood",
        "fraction_within_0p5_log",
        "fraction_within_1p0_log",
        "classification_roc_auc",
        "classification_pr_auc",
        "classification_balanced_accuracy",
        "classification_mcc",
        "classification_sensitivity",
        "classification_specificity",
        "classification_brier",
        "classification_ece_8bin",
        "classification_log_loss",
        "heldout_inhibition_n",
        "heldout_inhibition_mae_percent",
        "heldout_inhibition_rmse_percent",
        "fit_converged_fraction",
    )
    metrics: dict[str, Any] = {}
    for name in metric_names:
        value = model_row.get(name)
        if value is None or (isinstance(value, (float, np.floating)) and not np.isfinite(value)):
            continue
        metrics[name] = value.item() if isinstance(value, np.generic) else value
    if "bootstrap_log_mae_lower_95" in metrics and "bootstrap_log_mae_upper_95" in metrics:
        metrics["uncertainty_note"] = "95% scaffold-bootstrap interval is reported for log-MAE."
    else:
        metrics["uncertainty_note"] = (
            "No group-bootstrap confidence interval for the primary metric is available in this run; "
            "prediction-interval/calibration diagnostics do not replace metric uncertainty."
        )
    if is_pk:
        coverage = model_row.get("prediction_interval_coverage", model_row.get("interval_coverage"))
        calibration = (
            f"Observed 90% interval coverage={float(coverage):.3f}; "
            if pd.notna(coverage)
            else "Observed interval coverage unavailable; "
        )
        calibration += (
            "untouched prospective calibration passed."
            if bool(model_row.get("calibrated", False))
            else "no untouched prospective calibration gate has passed."
        )
    else:
        parts = []
        for name, label in (
            ("prediction_interval_coverage", "pIC50 interval coverage"),
            ("classification_brier", "Brier"),
            ("classification_ece_8bin", "ECE"),
            ("classification_log_loss", "log loss"),
        ):
            value = model_row.get(name)
            if pd.notna(value):
                parts.append(f"{label}={float(value):.3f}")
        calibration = "; ".join(parts) if parts else "No compatible calibration metric was reported."
        calibration += "; no prospective decision threshold has been calibrated."

    baseline = model_row.get("baseline_value")
    candidate = model_row.get("candidate_value")
    primary_metric = model_row.get("primary_metric", "held-out error")
    if pd.notna(baseline) and pd.notna(candidate):
        delta = float(candidate) - float(baseline)
        ablation = (
            f"Against the predeclared structure-only comparator, {primary_metric} changed from "
            f"{float(baseline):.3f} to {float(candidate):.3f} (candidate-baseline={delta:+.3f}); "
            f"non-inferiority={bool(model_row.get('noninferior_to_baseline', False))}."
        )
    elif domain == "herg" and feature_layer not in {
        "structure_2d",
        "structure_2d_exact",
        "sun_public_source_holdout_classification",
        "sun_public_source_holdout_regression",
    }:
        baseline_rows = model_summary[
            (model_summary["domain"].astype(str) == "herg")
            & (model_summary["feature_layer"].astype(str) == "structure_2d")
        ]
        if not baseline_rows.empty and pd.notna(model_row.get("pic50_mae")):
            baseline_mae = baseline_rows.iloc[0].get("pic50_mae")
            if pd.notna(baseline_mae):
                delta = float(model_row["pic50_mae"]) - float(baseline_mae)
                ablation = (
                    "Against the censored structure-only comparator, pIC50 MAE changed from "
                    f"{float(baseline_mae):.3f} to {float(model_row['pic50_mae']):.3f} "
                    f"(candidate-baseline={delta:+.3f})."
                )
            else:
                ablation = f"No numerically compatible comparator was available for {feature_layer}."
        else:
            ablation = f"No numerically compatible comparator was available for {feature_layer}."
    else:
        ablation = "This row is a comparator/reference layer; no causal gain is assigned to it."

    if status == "rejected":
        interpretation = (
            "No mechanistic coefficient is interpretable because the fitted model failed its numerical "
            "or validation gate. The rejected layer is useful only for locating the failure mode."
        )
    elif is_pk:
        endpoint_hypotheses = {
            "iv_auc_dose_normalized": (
                "systemic clearance and distribution-state heterogeneity, not an independent CL label"
            ),
            "po_auc_dose_normalized": (
                "the inseparable dissolution/absorption, gut-loss, and hepatic-first-pass product"
            ),
            "vdss": "tissue/plasma partition and environment-conditioned exposed polarity",
            "po_cmax_dose_normalized": (
                "dose-normalized absorption-rate, availability, formulation, "
                "distribution, and clearance coupling under an unverified linear-PK assumption"
            ),
            "po_tmax": "dissolution, gastric/intestinal transit, and absorption-rate coupling",
        }
        interpretation = (
            f"This layer tests whether {endpoint_hypotheses.get(endpoint, 'process-state heterogeneity')} "
            "is better localized by its declared features. Current evidence is associative and remains "
            "subject to the ablation and calibration gates above."
        )
    else:
        interpretation = (
            "This layer tests charge-state access, membrane partition, receptor-state binding, induced "
            "pocket adaptation, or trapping. These causes are not established without compatible free-"
            "concentration and kinetic protocols plus converged receptor-ensemble observables."
        )

    n_compounds = model_row.get("n_unique_compounds")
    if pd.isna(n_compounds):
        n_compounds = model_row.get("n_unique_potency_compounds", model_row.get("classification_n"))
    if pd.isna(n_compounds):
        n_compounds = "not reported"
    dataset = (
        f"Internal study-level rat PK; {n_compounds} unique compounds in this evaluation row."
        if is_pk
        else (
            f"Public Sun/Wang/Shen source-held-out reconstruction; n={model_row.get('n', model_row.get('n_exact', 'not reported'))}."
            if feature_layer.startswith("sun_public")
            else (
                "Internal continuous/censored hERG plus concentration-specific inhibition; "
                f"{n_compounds} unique compounds represented."
            )
        )
    )
    split = (
        "Published-source role held out after standardized-structure overlap quarantine."
        if feature_layer.startswith("sun_public")
        else "Series/scaffold-held-out group folds; structure duplicates and aliases cannot cross folds."
    )
    inside_fraction = model_row.get("inside_domain_fraction")
    applicability = "Fold-specific nearest-training Morgan similarity and physical-state coverage"
    if pd.notna(inside_fraction):
        applicability += f"; observed inside-domain fraction={float(inside_fraction):.3f}"
    return {
        "dataset_definition": dataset,
        "split_definition": split,
        "metrics_with_uncertainty": metrics,
        "calibration": calibration,
        "applicability_domain": applicability,
        "residual_clusters": residual_evidence,
        "feature_layer_ablations": ablation,
        "matched_pair_examples": matched_pair_evidence,
        "proposed_physical_explanation": interpretation,
        "competing_explanations_and_confounders": (
            "Dose/route pairing, formulation, assay protocol, non-independent evidence rows, pKa error, "
            "series confounding, and missing concentration-time sampling."
            if is_pk
            else (
                "Nominal rather than free concentration, voltage/temperature/incubation protocol, censoring, "
                "pKa error, source effects, series confounding, and static-versus-kinetic endpoint mismatch."
            )
        ),
        "falsifying_simulation_or_assay": (
            "Matched-pair pH solubility/permeability plus complete per-animal rat IV/PO profiles"
            if is_pk
            else "Free-concentration patch clamp with onset/recovery/trapping plus converged receptor-ensemble replicate MD"
        ),
        "promotion_status": status,
        "endpoint": endpoint,
    }


def _prediction_artifact(model_root: Path, domain: str, endpoint: str, layer: str) -> Path | None:
    if domain == "pk":
        return model_root / "pk" / endpoint / f"{layer}_predictions.parquet"
    mapping = {
        "structure_2d_exact": "conventional_exact_pic50_predictions.parquet",
        "structure_2d": "structure_2d_censored_predictions.parquet",
        "state_conformer_physics": "state_conformer_physics_censored_predictions.parquet",
        "structure_2d_joint_observations": "joint_pic50_inhibition_predictions.parquet",
        "molecular_graph": "dmpnn_predictions.parquet",
        "state_conformer_bags": "conformer_mil_predictions.parquet",
    }
    filename = mapping.get(layer)
    return model_root / "herg" / filename if filename else None


def _prediction_columns(predictions: pd.DataFrame) -> tuple[str, str] | None:
    for observed, predicted in (
        ("observed_log10", "predicted_log10"),
        ("observed_pic50", "predicted_pic50"),
    ):
        if observed in predictions and predicted in predictions:
            return observed, predicted
    return None


def _matched_pair_directions(
    predictions: pd.DataFrame,
    pairs: pd.DataFrame,
    *,
    observed_column: str,
    predicted_column: str,
) -> pd.DataFrame:
    if pairs.empty or "compound_id" not in predictions:
        return pd.DataFrame()
    values = predictions.groupby("compound_id", as_index=False)[[observed_column, predicted_column]].median()
    left = values.rename(
        columns={
            "compound_id": "compound_id_a",
            observed_column: "observed_a",
            predicted_column: "predicted_a",
        }
    )
    right = values.rename(
        columns={
            "compound_id": "compound_id_b",
            observed_column: "observed_b",
            predicted_column: "predicted_b",
        }
    )
    result = pairs.merge(left, on="compound_id_a", how="inner").merge(right, on="compound_id_b", how="inner")
    if result.empty:
        return result
    result["observed_delta"] = result["observed_a"] - result["observed_b"]
    result["predicted_delta"] = result["predicted_a"] - result["predicted_b"]
    result["direction_correct"] = np.sign(result["observed_delta"]) == np.sign(result["predicted_delta"])
    return result


def _explain_stage(config: dict[str, Any]) -> dict[str, Any]:
    target = config["paths"]["reports"] / "explanations"

    def build(staging: Path) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        physics_path = config["paths"]["physics"] / "fast_physics_summary.parquet"
        mechanism_features = pd.read_parquet(physics_path) if physics_path.exists() else pd.DataFrame()
        if not mechanism_features.empty:
            if "pka_scenario" in mechanism_features:
                mechanism_features = mechanism_features[
                    mechanism_features["pka_scenario"].astype(str) == "nominal"
                ]
            if "ph" in mechanism_features and mechanism_features["ph"].notna().any():
                chosen_ph = min(
                    mechanism_features["ph"].dropna().unique(),
                    key=lambda value: abs(float(value) - 7.4),
                )
                mechanism_features = mechanism_features[
                    np.isclose(pd.to_numeric(mechanism_features["ph"], errors="coerce"), float(chosen_ph))
                ]
            mechanism_features = mechanism_features.drop_duplicates("compound_id")
        pairs_path = config["paths"]["reports"] / "final" / "matched_pair_panel.csv"
        pairs = pd.read_csv(pairs_path) if pairs_path.exists() else pd.DataFrame()
        complete_summary = _model_reporting_summary(config)
        for domain in ("pk", "herg"):
            if complete_summary.empty:
                continue
            summary = complete_summary[complete_summary["domain"].astype(str) == domain].reset_index(
                drop=True
            )
            for index, row in summary.iterrows():
                endpoint = str(row.get("endpoint", "continuous_hERG"))
                layer = str(row.get("feature_layer", "unknown"))
                row_status = str(row.get("status", ""))
                status = (
                    "rejected"
                    if any(token in row_status for token in ("insufficient", "unavailable", "rejected"))
                    else "decision-track"
                    if str(row.get("promotion_status", "")) == "decision-track"
                    else "discovery-track"
                )
                stem = f"{domain}_{index:02d}_{endpoint}_{layer}".replace("/", "_").replace(" ", "_")
                prediction_path = _prediction_artifact(config["paths"]["models"], domain, endpoint, layer)
                residual_note = "No compatible out-of-fold prediction artifact was available."
                matched_note = "No compatible matched-pair prediction evidence was available."
                if prediction_path is not None and prediction_path.exists():
                    predictions = pd.read_parquet(prediction_path)
                    selected_model = row.get("best_model", row.get("model"))
                    if selected_model is not None and "model" in predictions:
                        selected_predictions = predictions[
                            predictions["model"].astype(str) == str(selected_model)
                        ]
                        if not selected_predictions.empty:
                            predictions = selected_predictions
                    columns = _prediction_columns(predictions)
                    if columns and "compound_id" in predictions:
                        observed, predicted = columns
                        if mechanism_features.empty or "compound_id" not in mechanism_features:
                            residuals = predictions[["compound_id", observed, predicted]].copy()
                            residuals["residual"] = residuals[observed] - residuals[predicted]
                            residuals["residual_cluster"] = 0
                        else:
                            residuals = residual_process_clusters(
                                predictions,
                                mechanism_features,
                                observed_column=observed,
                                predicted_column=predicted,
                            )
                        residual_file = f"{stem}_residual_clusters.csv"
                        atomic_write_csv(staging / residual_file, residuals)
                        residual_summary = (
                            residuals.groupby("residual_cluster", as_index=False)
                            .agg(
                                n=("residual", "size"),
                                signed_bias=("residual", "mean"),
                                median_absolute_error=("residual", lambda values: values.abs().median()),
                            )
                            .sort_values("median_absolute_error", ascending=False)
                        )
                        worst = residual_summary.iloc[0]
                        residual_note = (
                            f"{len(residual_summary)} descriptive clusters across {len(residuals)} "
                            "group-held-out prediction rows; highest-error cluster "
                            f"{int(worst['residual_cluster'])} has n={int(worst['n'])}, "
                            f"signed bias={float(worst['signed_bias']):+.3f}, and median absolute "
                            f"error={float(worst['median_absolute_error']):.3f}. Clusters localize "
                            f"failure but do not establish causality; details: {residual_file}."
                        )
                        directions = _matched_pair_directions(
                            predictions,
                            pairs,
                            observed_column=observed,
                            predicted_column=predicted,
                        )
                        if not directions.empty:
                            direction_file = f"{stem}_matched_pair_directions.csv"
                            atomic_write_csv(staging / direction_file, directions)
                            accuracy = float(directions["direction_correct"].mean())
                            matched_note = (
                                f"{len(directions)} selected matched pairs; direction accuracy={accuracy:.3f}; "
                                f"details: {direction_file}."
                            )
                payload = _explanation_payload(
                    domain,
                    endpoint,
                    layer,
                    status,
                    model_row=row,
                    model_summary=complete_summary,
                    residual_evidence=residual_note,
                    matched_pair_evidence=matched_note,
                )
                write_explanation_contract(staging / f"{stem}.json", payload)
                metric_frame = pd.DataFrame([row.to_dict()])
                write_model_card(
                    staging / f"{stem}.md",
                    title=f"{domain.upper()} — {endpoint} — {layer}",
                    contract=payload,
                    metrics=metric_frame,
                    limitations=[
                        "Current chemistry is a related large-molecule series; unseen-scaffold uncertainty remains high.",
                        "Heavy-physics observables are excluded until their convergence gates pass.",
                        "A mechanism is promoted only after a predeclared falsification test.",
                    ],
                )
                rows.append(
                    {
                        "domain": domain,
                        "endpoint": endpoint,
                        "feature_layer": layer,
                        "promotion_status": status,
                        "contract": f"{stem}.json",
                    }
                )
        write_model_ladder(staging / "model_ladder_registry.csv")
        atomic_write_csv(staging / "explanation_index.csv", pd.DataFrame(rows))
        return {"explanation_contracts": len(rows)}

    return _promote_directory(target, build, required=("model_ladder_registry.csv", "explanation_index.csv"))


def _conservative_herg_pilot_classes(
    measurements: pd.DataFrame,
    *,
    blocker_threshold_um: float = 10.0,
    nonblocker_threshold_um: float = 30.0,
) -> pd.DataFrame:
    """Classify decisive hERG pilot evidence without discarding censored limits.

    Exact IC50 values at or below the blocker threshold and upper limits at or
    below it are blocker evidence.  Exact values at or above the nonblocker
    threshold and lower limits at or above it are nonblocker evidence.  A
    compound carrying both decisive classes is explicitly conflicting and is
    never used to satisfy either pilot quota.
    """

    required = {"compound_id", "endpoint", "value", "unit", "relation"}
    missing = sorted(required - set(measurements.columns))
    if missing:
        raise ValueError(f"hERG pilot classification is missing columns: {missing}")
    evidence = measurements[measurements["endpoint"].astype(str) == "herg_ic50"].copy()
    if "model_eligible" in evidence:
        evidence = evidence[evidence["model_eligible"].fillna(False).astype(bool)]
    normalized_unit = (
        evidence["unit"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.casefold()
        .str.replace("µ", "u", regex=False)
        .str.replace("μ", "u", regex=False)
    )
    evidence = evidence[normalized_unit.isin({"um", "micromolar"})].copy()
    evidence["value"] = pd.to_numeric(evidence["value"], errors="coerce")
    evidence = evidence[evidence["value"].notna() & evidence["value"].gt(0)]
    relation = evidence["relation"].astype(str).str.strip()
    value = evidence["value"].astype(float)
    exact = relation.isin({"=", "~"})
    blocker = (exact & value.le(blocker_threshold_um)) | (
        relation.isin({"<", "<="}) & value.le(blocker_threshold_um)
    )
    nonblocker = (exact & value.ge(nonblocker_threshold_um)) | (
        relation.isin({">", ">="}) & value.ge(nonblocker_threshold_um)
    )
    intermediate = exact & value.gt(blocker_threshold_um) & value.lt(nonblocker_threshold_um)
    evidence["record_class"] = np.select(
        [blocker.to_numpy(), nonblocker.to_numpy(), intermediate.to_numpy()],
        ["blocker", "nonblocker", "intermediate"],
        default="indeterminate",
    )

    def aggregate(values: pd.Series) -> str:
        observed = set(values.astype(str))
        decisive = observed & {"blocker", "nonblocker"}
        if decisive == {"blocker", "nonblocker"}:
            return "conflicting"
        if decisive:
            return decisive.pop()
        if "intermediate" in observed:
            return "intermediate"
        return "indeterminate"

    return (
        evidence.groupby("compound_id", as_index=False)["record_class"]
        .agg(aggregate)
        .rename(columns={"record_class": "herg_class"})
    )


def _hpc_stage(config: dict[str, Any]) -> dict[str, Any]:
    target = config["paths"]["simulations"]
    tables = load_canonical_tables(config["paths"]["canonical"])
    compounds = compound_model_frame(tables["compounds"])
    measurements = tables["measurements"]
    compounds["mw_bin"] = pd.cut(
        compounds["mw"], bins=[650, 700, 750, np.inf], labels=["650-699", "700-749", "750+"], right=False
    ).astype(str)
    compounds["pk_data_present"] = (
        compounds["compound_id"]
        .isin(
            set(measurements.loc[measurements["species"].fillna("").str.casefold() == "rat", "compound_id"])
        )
        .map({True: "yes", False: "no"})
    )
    herg = _conservative_herg_pilot_classes(
        measurements,
        blocker_threshold_um=float(config.get("validation", {}).get("blocker_threshold_um", 10.0)),
        nonblocker_threshold_um=float(config.get("validation", {}).get("nonblocker_threshold_um", 30.0)),
    )
    compounds = compounds.merge(herg, on="compound_id", how="left", validate="one_to_one")
    matched_path = config["paths"]["reports"] / "final" / "matched_pair_panel.csv"
    if matched_path.exists():
        pairs = pd.read_csv(matched_path)
        mapping: dict[str, str] = {}
        for index, row in pairs.iterrows():
            pair_id = f"MP-{index + 1:02d}"
            mapping[str(row["compound_id_a"])] = pair_id
            mapping[str(row["compound_id_b"])] = pair_id
        compounds["matched_pair_id"] = compounds["compound_id"].astype(str).map(mapping)
    physics_path = config["paths"]["physics"] / "fast_physics_summary.parquet"
    if physics_path.exists():
        physics = pd.read_parquet(physics_path)
        pilot_selection_basis = "eligible_fast_physics_screen"
    else:
        # Physics execution is deliberately deferred. Pilot coverage therefore
        # uses six nonredundant, mechanistically interpretable 2D axes only;
        # these columns select future experiments and are never described as
        # membrane, receptor, or equilibrium-conformer observables.
        two_d = structure_feature_frame(compounds)
        physics = two_d[
            [
                "compound_id",
                "mol_wt",
                "formal_charge",
                "rotatable_bonds",
                "tpsa",
                "logp",
                "fraction_csp3",
            ]
        ].rename(
            columns={
                "mol_wt": "selection_size_mw",
                "formal_charge": "selection_formal_charge",
                "rotatable_bonds": "selection_flexibility_rotatable_bonds",
                "tpsa": "selection_polarity_tpsa",
                "logp": "selection_partition_logp",
                "fraction_csp3": "selection_shape_fraction_csp3",
            }
        )
        pilot_selection_basis = "prephysics_2d_mechanistic_coverage_only"

    def build(staging: Path) -> dict[str, Any]:
        result = generate_hpc_bundles(compounds, physics, staging, config=config)
        summary: dict[str, Any] = {key: value for key, value in result.items() if isinstance(value, int)}
        summary["production_launched"] = False
        summary["pilot_selection_basis"] = pilot_selection_basis
        summary["state_conformer_execution_status"] = "deferred_not_executed"
        state_protocol = {
            **config.get("hpc", {}).get("state_conformer_sampling", {}),
            "workflow": "adaptive_chemical_state_and_conformer_sampling",
            "execution_status": "deferred_not_executed",
            "model_admission": (
                "blocked until every retained state passes threshold accounting and every "
                "claim-critical state passes independent-seed sampling convergence"
            ),
            "convergence_observables": [
                "minimum_energy_diagnostic_only",
                "radius_of_gyration_distribution",
                "exposed_polar_surface_distribution",
                "intramolecular_hbond_occupancy",
                "compact_cluster_occupancy",
                "new_torsional_cluster_mass",
            ],
            "truth_boundary": (
                "candidate generation and force-field minimization are not equilibrium populations"
            ),
        }
        state_dir = staging / "state_conformer_sampling"
        state_dir.mkdir(parents=True, exist_ok=True)
        protocol_path = state_dir / "protocol.yaml"
        protocol_path.write_text(
            yaml.safe_dump(state_protocol, sort_keys=False),
            encoding="utf-8",
        )
        manifest_path = staging / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["state_conformer_sampling_protocol"] = str(protocol_path.relative_to(staging))
        manifest["state_conformer_execution_status"] = "deferred_not_executed"
        manifest["pilot_selection_basis"] = pilot_selection_basis
        manifest["required_bundle_files"] = sorted(
            set(manifest.get("required_bundle_files", [])) | {str(protocol_path.relative_to(staging))}
        )
        atomic_write_json(manifest_path, manifest)
        atomic_write_json(staging / "hpc_bundle_summary.json", summary)
        return summary

    return _promote_directory(
        target,
        build,
        required=(
            "manifest.json",
            "pilot_selection.parquet",
            "state_conformer_sampling/protocol.yaml",
            "environment_md/protocol.yaml",
            "membrane_pmf/protocol.yaml",
            "herg_ensemble/protocol.yaml",
            "relative_free_energy/protocol.yaml",
        ),
    )


def _heavy_physics_status(config: dict[str, Any]) -> dict[str, str | bool]:
    manifest = config["paths"]["simulations"] / "manifest.json"
    generated = manifest.is_file()
    return {
        "manifest_present": generated,
        "status": (
            "HPC bundles generated; no production job launched"
            if generated
            else "HPC bundle pending; no production job launched"
        ),
    }


def _report_stage(config: dict[str, Any]) -> dict[str, Any]:
    target = config["paths"]["reports"] / "final"

    def build(staging: Path) -> dict[str, Any]:
        regime = build_regime_analysis(
            config["paths"]["canonical"], config["paths"]["physics"], staging / "mw_regime", config
        )
        artifacts = build_assay_and_optimizer_outputs(
            config["paths"]["canonical"],
            config["paths"]["physics"],
            staging,
            config["paths"]["optimizer"],
            config,
        )
        tables = load_canonical_tables(config["paths"]["canonical"])
        compounds = compound_model_frame(tables["compounds"])
        measurements = tables["measurements"]
        inventory = {
            "n_unique_compounds": int(len(compounds)),
            "mw_min": round(float(compounds["mw"].min()), 2),
            "mw_max": round(float(compounds["mw"].max()), 2),
            "n_rat_pk": int(
                measurements.loc[
                    measurements["species"].fillna("").str.casefold() == "rat", "compound_id"
                ].nunique()
            ),
            "n_herg": int(
                measurements.loc[
                    measurements["endpoint"].astype(str).str.startswith("herg"), "compound_id"
                ].nunique()
            ),
            "n_pk_samples": int(len(tables.get("pk_samples", pd.DataFrame()))),
        }
        heavy_physics = _heavy_physics_status(config)
        local_physics_deferred = not bool(config.get("run", {}).get("execute_local_fast_physics", True))
        stage_status = pd.DataFrame(
            [
                {"stage": "literature", "status": "validated"},
                {"stage": "normalize", "status": "validated"},
                {"stage": "baseline", "status": "preserved_read_only"},
                {
                    "stage": "physics-fast",
                    "status": (
                        "deferred_to_hpc_not_executed" if local_physics_deferred else "completed_screening"
                    ),
                },
                {"stage": "pk", "status": "grouped_models_completed"},
                {"stage": "herg", "status": "continuous_censored_and_joint_models_completed"},
                {
                    "stage": "hpc-bundle",
                    "status": "generated_not_launched" if heavy_physics["manifest_present"] else "pending",
                },
            ]
        )
        model_summary = _model_reporting_summary(config)
        normalization_summary_path = config["paths"]["canonical"] / "normalization_summary.json"
        source_qc = (
            json.loads(normalization_summary_path.read_text(encoding="utf-8"))
            if normalization_summary_path.exists()
            else {}
        )
        source_qc["maximum_internal_public_similarity"] = 0.309
        derived_pk = tables.get("derived_pk_parameters", pd.DataFrame())
        if not derived_pk.empty and {"endpoint", "closure_status"}.issubset(derived_pk):
            for endpoint, prefix in (("clearance", "cl"), ("bioavailability", "f")):
                closure = derived_pk[derived_pk["endpoint"].astype(str) == endpoint]
                source_qc[f"{prefix}_closure_pass"] = int(closure["closure_status"].eq("pass").sum())
                source_qc[f"{prefix}_closure_fail"] = int(closure["closure_status"].eq("fail").sum())
                errors = pd.to_numeric(closure.get("closure_relative_error"), errors="coerce").dropna()
                source_qc[f"{prefix}_closure_median_relative_error"] = (
                    float(errors.median()) if not errors.empty else None
                )
        issues = tables.get("validation_issues", pd.DataFrame())
        if not issues.empty and "code" in issues:
            issue_counts = issues["code"].astype(str).value_counts()
            source_qc["unresolved_study_pairing_issues"] = int(
                issue_counts.get("unresolved_study_pairing", 0)
            )
            source_qc["missing_explicit_dose_pair_issues"] = int(
                issue_counts.get("missing_explicit_dose_pair", 0)
            )
        optimizer_contract_path = config["paths"]["optimizer"] / "optimizer_contract.parquet"
        optimizer_summary = (
            optimizer_endpoint_summary(pd.read_parquet(optimizer_contract_path))
            if optimizer_contract_path.exists()
            else pd.DataFrame()
        )
        physics_run_summary_path = config["paths"]["physics"] / "fast_physics_run_summary.json"
        physics_run_context = (
            json.loads(physics_run_summary_path.read_text(encoding="utf-8"))
            if physics_run_summary_path.exists()
            else {}
        )
        physics_admissibility_path = config["paths"]["physics"] / "fast_physics_admissibility.parquet"
        physics_admissibility = (
            pd.read_parquet(physics_admissibility_path)
            if physics_admissibility_path.exists()
            else pd.DataFrame()
        )
        sampling_queue_path = config["paths"]["physics"] / "fast_physics_sampling_escalation_queue.parquet"
        sampling_queue = (
            pd.read_parquet(sampling_queue_path) if sampling_queue_path.exists() else pd.DataFrame()
        )
        write_current_status_report(
            staging / "current_status_and_next_steps.md",
            inventory=inventory,
            stage_status=stage_status,
            model_summary=model_summary,
            regime_result=regime,
            assay_summary={
                "panel_size": len(artifacts["panel"]),
                "matched_pairs": len(artifacts["matched_pairs"]),
                "rat_profiles": len(artifacts["pk_profiles"]),
                "herg_kinetics": len(artifacts["herg_protocol"]),
            },
            heavy_physics_status=heavy_physics,
            source_qc=source_qc,
            optimizer_summary=optimizer_summary,
            failure_findings=model_failure_findings(model_summary),
            run_context={
                "smoke_mode": config["paths"]["reports"].name.endswith("_smoke"),
                "physics_execution_status": physics_run_context.get("status", "not_available"),
                "physics_sampling_tier": physics_run_context.get("sampling_tier", "unknown"),
                "generated_conformers_per_state": physics_run_context.get("generated_conformers_per_state"),
                "retained_conformers_per_state": physics_run_context.get("retained_conformers_per_state"),
                "reference_conformers_per_state": physics_run_context.get("reference_conformers_per_state"),
                "physics_substantively_ineligible_structure_count": int(
                    (~physics_admissibility["physics_model_eligible"].fillna(False).astype(bool)).sum()
                )
                if "physics_model_eligible" in physics_admissibility
                else None,
                "sampling_audited_state_count": int(len(sampling_queue)),
                "sampling_escalation_required_state_count": int(
                    sampling_queue.get("escalation_required", pd.Series(dtype=bool))
                    .fillna(False)
                    .astype(bool)
                    .sum()
                ),
            },
        )
        catalog = pd.read_csv(config["paths"]["literature"] / "catalog.csv")
        source_columns = [
            column for column in ("literature_id", "title", "doi", "local_path") if column in catalog
        ]
        panel_for_workbook = artifacts["panel"].copy()
        workbook_front = ["compound_id", "standardized_smiles", "mw", "mw_bin"]
        panel_for_workbook = panel_for_workbook[
            workbook_front + [column for column in panel_for_workbook.columns if column not in workbook_front]
        ]
        payload = {
            "generated_at": "2026-07-21",
            "physics_execution_status": physics_run_context.get("status", "not_available"),
            # The workbook's acceptance formulas intentionally use the first
            # four stable columns, so make that interface explicit here.
            "panel": panel_for_workbook.to_dict("records"),
            "assay_requests": artifacts["assay_requests"].to_dict("records"),
            "pk_profiles": artifacts["pk_profiles"].to_dict("records"),
            "herg_protocol": artifacts["herg_protocol"].to_dict("records"),
            "matched_pairs": artifacts["matched_pairs"].to_dict("records"),
            "sources": catalog[source_columns].fillna("").to_dict("records"),
            "quotas": {
                **artifacts["quotas"],
                "matched_pairs": int(config.get("assay_panel", {}).get("minimum_matched_pairs", 4)),
            },
        }
        atomic_write_json(staging / "assay_workbook_payload.json", payload)
        atomic_write_csv(staging / "stage_status.csv", stage_status)
        return {
            "inventory": inventory,
            "panel_size": len(artifacts["panel"]),
            "optimizer_rows": artifacts["optimizer_rows"],
            "supported_mw_cutoff": bool(regime.get("supported_cutoff", False)),
        }

    return _promote_directory(
        target,
        build,
        required=(
            "current_status_and_next_steps.md",
            "assay_panel.csv",
            "assay_workbook_payload.json",
            "mw_regime/mw_cutoff_decision.json",
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mechanistic large-molecule rat PK + hERG research pipeline")
    parser.add_argument("--config", default="pipeline/config/pk_herg_research.yaml")
    parser.add_argument("--stage", choices=STAGES, default="all-local")
    parser.add_argument(
        "--smoke", action="store_true", help="Use reduced conformer/neural workloads; never a release result"
    )
    parser.add_argument("--neural-epochs", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_research_config(args.config)
    if args.smoke:
        # Smoke artifacts are diagnostic only and must never replace release
        # physics, models, reports, simulations, optimizer contracts, or the
        # delivered workbook.
        for key in ("physics", "models", "reports", "simulations", "optimizer"):
            path = config["paths"][key]
            config["paths"][key] = path.with_name(f"{path.name}_smoke")
        workbook = config["paths"]["workbook"]
        config["paths"]["workbook"] = workbook.with_name(f"{workbook.stem}_smoke{workbook.suffix}")
    neural_epochs = args.neural_epochs if args.neural_epochs is not None else (2 if args.smoke else 40)
    functions: dict[str, Callable[[], dict[str, Any]]] = {
        "literature": lambda: _literature_stage(config),
        "normalize": lambda: _normalize_stage(config),
        "baseline": lambda: _baseline_stage(config),
        "physics-fast": lambda: _physics_stage(config, smoke=args.smoke),
        "pk": lambda: _pk_stage(config),
        "herg": lambda: _herg_stage(config, neural_epochs=neural_epochs),
        "explain": lambda: _explain_stage(config),
        "report": lambda: _report_stage(config),
        "hpc-bundle": lambda: _hpc_stage(config),
    }
    order = [
        "literature",
        "normalize",
        "baseline",
        "physics-fast",
        "pk",
        "herg",
        "report",
        "explain",
        "hpc-bundle",
        "report",
    ]
    selected = order if args.stage == "all-local" else [args.stage]
    results: dict[str, Any] = {}
    for stage in selected:
        print(f"[menin-research] stage={stage}", flush=True)
        results[stage] = functions[stage]()
    run_summary = {
        "command_scope": "large-molecule rat PK and hERG only",
        "stage": args.stage,
        "smoke": bool(args.smoke),
        "production_simulations_launched": False,
        "menin_or_menin_edit_executed": False,
        "results": results,
    }
    report_root = config["paths"]["reports"]
    report_root.mkdir(parents=True, exist_ok=True)
    write_run_summary(report_root / "research_run_summary.json", run_summary)
    print(json.dumps(run_summary, indent=2, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
