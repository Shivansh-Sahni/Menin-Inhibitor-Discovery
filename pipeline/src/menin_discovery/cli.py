"""Command-line orchestration for the reproducible Menin workflow."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .bindingdb import download_bindingdb_sources, load_bindingdb_tsvs
from .chembl import (
    fetch_chembl_status,
    fetch_molecule_activities,
    fetch_target_activities,
    fetch_target_search,
)
from .config import HERG_TARGET, MENIN_TARGET
from .curation import write_processed_tables
from .modeling import (
    prepare_menin_task,
    train_herg_classifier_and_predict,
    train_menin_activity_model,
)
from .provenance import (
    ManifestVerification,
    VerificationIssue,
    create_data_manifests,
    create_manifest,
    sha256_file,
    verify_data_manifests,
    verify_manifest,
    write_manifest,
)
from .pubchem import collect_pubchem_assays, load_pubchem_assay_csvs
from .quality import QualityConfig, audit_tables, write_quality_outputs
from .settings import ROOT, load_settings, resolve_project_path, settings_snapshot

STAGES = (
    "all",
    "collect",
    "curate",
    "data",  # Backward-compatible alias for curate.
    "quality",
    "models",
    "analyze",
    "report",
    "manifest",
    "verify",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False) if path.exists() else pd.DataFrame()


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return path


def _git_revision(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def _git_dirty(root: Path) -> bool | None:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    return bool(result.stdout.strip())


def _project_root(settings: dict[str, Any]) -> Path:
    configured = settings.get("project", {}).get("root")
    if configured is None or not str(configured).strip():
        return ROOT
    path = Path(str(configured)).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _analysis_enabled(settings: dict[str, Any]) -> bool:
    """Return whether the optional analysis stage participates in the release DAG."""

    return settings.get("analysis", {}).get("enabled", False) is True


def _paths(settings: dict[str, Any]) -> dict[str, Path]:
    configured = settings["paths"]
    root = _project_root(settings)
    paths = {
        key: resolve_project_path(configured[key], root=root)
        for key in ("raw", "processed", "models", "reports")
    }
    if _analysis_enabled(settings):
        configured_analysis = configured.get("analysis")
        if configured_analysis is None or not str(configured_analysis).strip():
            raise ValueError("paths.analysis is required when analysis.enabled is true")
        paths["analysis"] = resolve_project_path(configured_analysis, root=root)
    return paths


def collect_public_data(args: argparse.Namespace, settings: dict[str, Any]) -> dict[str, int]:
    """Refresh public-source snapshots without deleting a previous good download."""

    if args.skip_network:
        print("Network collection skipped; reusing the existing raw snapshot.")
        return {}

    raw_target = _paths(settings)["raw"]
    raw_target.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(tempfile.mkdtemp(prefix=".menin-collection-", dir=raw_target.parent))
    raw = staging_root / "raw"
    if raw_target.exists():
        shutil.copytree(raw_target, raw, dirs_exist_ok=True)
    else:
        raw.mkdir(parents=True, exist_ok=True)
    chembl_raw = raw / "chembl"
    bindingdb_raw = raw / "bindingdb"
    pubchem_raw = raw / "pubchem"
    chembl_raw.mkdir(parents=True, exist_ok=True)

    status = fetch_chembl_status(chembl_raw / "chembl_status.json")
    fetch_target_search("menin", chembl_raw / "chembl_target_search_menin.json")
    fetch_target_search("KCNH2 hERG", chembl_raw / "chembl_target_search_herg.json")
    menin = fetch_target_activities(
        MENIN_TARGET["chembl_id"],
        chembl_raw / "chembl_menin_activities.csv",
        max_records=args.max_menin_records,
    )
    herg = fetch_target_activities(
        HERG_TARGET["chembl_id"],
        chembl_raw / "chembl_herg_activities.csv",
        max_records=args.max_herg_records,
    )
    binding_paths = download_bindingdb_sources(bindingdb_raw)
    pubchem_catalog = collect_pubchem_assays(
        pubchem_raw,
        max_aids=args.max_pubchem_aids,
        retmax_per_term=args.pubchem_retmax_per_term,
        overwrite=args.overwrite_raw,
    )

    pk = pd.DataFrame()
    if not args.skip_pk:
        molecule_ids = (
            menin.get("molecule_chembl_id", pd.Series(dtype=str))
            .dropna()
            .astype(str)
            .drop_duplicates()
            .tolist()
        )
        pk = fetch_molecule_activities(
            molecule_ids,
            chembl_raw / "chembl_menin_molecule_all_activities.csv",
            max_chunks=args.max_pk_chunks,
        )

    counts = {
        "chembl_menin_rows": int(len(menin)),
        "chembl_herg_rows": int(len(herg)),
        "bindingdb_files": int(len(binding_paths)),
        "pubchem_assays": int(len(pubchem_catalog)),
        "molecule_activity_rows": int(len(pk)),
    }
    _write_json(
        raw / "collection_metadata.json",
        {
            "collected_at": _utc_now(),
            "chembl_status": status,
            "counts": counts,
            "collection_policy": {
                "max_menin_records": args.max_menin_records,
                "max_herg_records": args.max_herg_records,
                "max_pubchem_aids": args.max_pubchem_aids,
                "pubchem_retmax_per_term": args.pubchem_retmax_per_term,
                "max_pk_chunks": args.max_pk_chunks,
                "skip_pk": bool(args.skip_pk),
                "overwrite_raw": bool(args.overwrite_raw),
            },
            "targets": {"menin": MENIN_TARGET, "herg": HERG_TARGET},
            "snapshot_promotion": "all-source staging promoted after successful collection",
        },
    )
    previous_raw = staging_root / "previous_raw"
    promoted = False
    try:
        if raw_target.exists():
            os.replace(raw_target, previous_raw)
        os.replace(raw, raw_target)
        promoted = True
    except Exception:
        if promoted and raw_target.exists():
            shutil.rmtree(raw_target)
        if previous_raw.exists() and not raw_target.exists():
            os.replace(previous_raw, raw_target)
        raise
    shutil.rmtree(staging_root, ignore_errors=True)
    return counts


def curate_data(settings: dict[str, Any]) -> dict[str, int]:
    """Normalize all available public data into auditable project contracts."""

    paths = _paths(settings)
    raw = paths["raw"]
    chembl_raw = raw / "chembl"
    bindingdb_raw = raw / "bindingdb"
    pubchem_raw = raw / "pubchem"
    curation = settings.get("curation", {})
    herg = settings.get("herg", {})
    processed_target = paths["processed"]
    processed_target.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(tempfile.mkdtemp(prefix=".menin-curation-", dir=processed_target.parent))

    try:
        tables = write_processed_tables(
            chembl_menin_raw=_read_csv(chembl_raw / "chembl_menin_activities.csv"),
            bindingdb_raw=load_bindingdb_tsvs(bindingdb_raw),
            pubchem_raw=load_pubchem_assay_csvs(pubchem_raw),
            pubchem_catalog=_read_csv(pubchem_raw / "pubchem_assay_catalog.csv"),
            herg_raw=_read_csv(chembl_raw / "chembl_herg_activities.csv"),
            pk_raw=_read_csv(chembl_raw / "chembl_menin_molecule_all_activities.csv"),
            processed_dir=staging_dir,
            menin_stratify_by=tuple(curation.get("menin_stratify_by", ("endpoint", "assay_family"))),
            herg_stratify_by=tuple(curation.get("herg_stratify_by", ("endpoint", "assay_family"))),
            normalization_options={
                "standardize_structures": bool(curation.get("standardize_structures", True)),
                "require_rdkit": bool(curation.get("require_rdkit", True)),
                "strip_salts": bool(curation.get("strip_salts", True)),
                "canonicalize_tautomer": bool(curation.get("canonicalize_tautomers", False)),
                "core_endpoints": tuple(curation.get("core_endpoints", ("IC50", "Ki", "Kd", "EC50"))),
                "enforce_target_relevance": bool(curation.get("pubchem_require_menin_relevance", True)),
                "exclude_assay_variants": bool(curation.get("exclude_assay_variants", True)),
                "accepted_chembl_validity_comments": tuple(
                    curation.get("accepted_chembl_validity_comments", ("",))
                ),
                "reject_chembl_potential_duplicates": bool(curation.get("reject_potential_duplicates", True)),
            },
            menin_censoring_policy=str(curation.get("exact_policy", "strict_exact")),
            heterogeneity_log_spread_threshold=float(curation.get("max_within_compound_log_spread", 1.0)),
            herg_blocker_max_nm=float(herg.get("blocker_max_nm", 10_000.0)),
            herg_nonblocker_min_nm=float(herg.get("nonblocker_min_nm", 30_000.0)),
        )
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise
    counts = {name: int(len(table)) for name, table in tables.items()}
    _write_json(
        staging_dir / "build_summary.json",
        {"processed_schema_version": "2.0", "counts": counts},
    )
    previous_processed = staging_dir.with_name(f"{staging_dir.name}-previous")
    promoted = False
    backed_up = False
    try:
        if processed_target.exists():
            os.replace(processed_target, previous_processed)
            backed_up = True
        os.replace(staging_dir, processed_target)
        promoted = True
    except Exception:
        if promoted and processed_target.exists():
            shutil.rmtree(processed_target)
        if backed_up and previous_processed.exists() and not processed_target.exists():
            os.replace(previous_processed, processed_target)
        raise
    shutil.rmtree(previous_processed, ignore_errors=True)
    for name, count in counts.items():
        print(f"{name}: {count:,} rows")
    return counts


def run_quality_stage(settings: dict[str, Any], *, fail_on_errors: bool = False) -> dict[str, dict[str, Any]]:
    """Audit the complete source inventory and gate only analysis-eligible rows."""

    paths = _paths(settings)
    processed = paths["processed"]
    reports_dir = paths["reports"] / "quality"
    threshold = float(settings.get("curation", {}).get("max_within_compound_log_spread", 1.0))
    config = QualityConfig(conflict_log10_threshold=threshold)
    menin_inventory = _read_csv(processed / "menin_activity_measurements.csv")
    herg_inventory = _read_csv(processed / "herg_activity_measurements.csv")
    pk_inventory = _read_csv(processed / "pk_admet_observations_all.csv")
    pk_analysis = _read_csv(processed / "pk_admet_observations.csv")

    def eligible_rows(table: pd.DataFrame) -> pd.DataFrame:
        if "is_modeling_eligible" not in table.columns:
            return table.copy()
        mask = table["is_modeling_eligible"].astype(str).str.strip().str.casefold().isin({"true", "1", "yes"})
        return table.loc[mask].copy()

    inventory_reports = audit_tables(
        menin=menin_inventory,
        herg=herg_inventory,
        pk=pk_inventory,
        config=config,
    )
    reports = audit_tables(
        menin=eligible_rows(menin_inventory),
        herg=eligible_rows(herg_inventory),
        pk=pk_analysis,
        config=config,
    )
    summary: dict[str, dict[str, Any]] = {}
    inventory_summary: dict[str, dict[str, Any]] = {}
    for name, report in inventory_reports.items():
        write_quality_outputs(report, reports_dir, prefix=f"{name}_inventory")
        severity = report.summary_frame()
        severity_counts = (
            severity.groupby("severity")["finding_count"].sum().astype(int).to_dict()
            if not severity.empty
            else {}
        )
        inventory_summary[name] = {
            "passed": report.passed,
            "row_count": report.row_count,
            "finding_count": report.finding_count,
            "severity_counts": severity_counts,
        }
    for name, report in reports.items():
        write_quality_outputs(report, reports_dir, prefix=name)
        severity = report.summary_frame()
        severity_counts = (
            severity.groupby("severity")["finding_count"].sum().astype(int).to_dict()
            if not severity.empty
            else {}
        )
        summary[name] = {
            "passed": report.passed,
            "row_count": report.row_count,
            "finding_count": report.finding_count,
            "severity_counts": severity_counts,
        }
    gate_passed = all(item["passed"] for item in summary.values())
    _write_json(
        reports_dir / "quality_gate.json",
        {
            "generated_at": _utc_now(),
            "passed": gate_passed,
            "gate_scope": "analysis-eligible rows; source-inventory findings remain visible",
            "tables": summary,
            "source_inventory": inventory_summary,
        },
    )
    if fail_on_errors and any(not item["passed"] for item in summary.values()):
        failed = ", ".join(name for name, item in summary.items() if not item["passed"])
        raise RuntimeError(f"Quality gate failed for: {failed}")
    return summary


def _metric_row(task: str, split: str, metrics: dict[str, Any]) -> dict[str, Any]:
    test = metrics.get("test_metrics", {})
    row: dict[str, Any] = {
        "task": task,
        "requested_split": split,
        "actual_split": metrics.get("split", {}).get("strategy"),
        "status": metrics.get("status"),
        "model": metrics.get("model"),
        "n_compounds": metrics.get("n_compounds"),
        "n_train": metrics.get("n_train"),
        "n_test": metrics.get("n_test"),
    }
    for key in (
        "mae",
        "rmse",
        "r2",
        "spearman_r",
        "roc_auc",
        "pr_auc",
        "balanced_accuracy",
        "brier_score",
        "expected_calibration_error_10bin",
    ):
        row[key] = test.get(key)
    return row


def run_model_stage(
    settings: dict[str, Any],
    *,
    split_override: str | None = None,
    endpoint_override: str | None = None,
    assay_family_override: str | None = None,
    fast: bool = False,
    models_output_dir: Path | None = None,
    reports_output_dir: Path | None = None,
    provenance_reports_dir: Path | None = None,
) -> dict[str, Any]:
    """Train endpoint-specific Menin and calibrated hERG models under audited splits."""

    paths = _paths(settings)
    release_models_dir = paths["models"]
    manifest_reports_dir = provenance_reports_dir or paths["reports"]
    if models_output_dir is not None:
        paths["models"] = models_output_dir
    if reports_output_dir is not None:
        paths["reports"] = reports_output_dir
    processed = paths["processed"]
    model_cfg = settings.get("modeling", {})
    herg_cfg = settings.get("herg", {})
    random_state = int(settings.get("project", {}).get("random_state", 13))
    primary_split = split_override or str(model_cfg.get("primary_split", "scaffold"))
    primary_endpoint = endpoint_override or str(model_cfg.get("primary_menin_endpoint", "IC50"))
    primary_herg_endpoint = str(herg_cfg.get("primary_endpoint", "IC50"))
    primary_herg_assay_family = str(herg_cfg.get("primary_assay_family", "electrophysiology_functional"))
    task_families = {
        str(key).casefold(): str(value)
        for key, value in model_cfg.get("menin_task_assay_families", {}).items()
    }
    if assay_family_override:
        primary_assay_family = assay_family_override
    elif endpoint_override:
        primary_assay_family = task_families.get(primary_endpoint.casefold(), "")
    else:
        primary_assay_family = str(
            model_cfg.get(
                "primary_menin_assay_family",
                task_families.get(primary_endpoint.casefold(), "biochemical_inhibition"),
            )
        )
    evaluation_splits = list(dict.fromkeys(model_cfg.get("evaluation_splits", [primary_split])))
    if primary_split not in evaluation_splits:
        evaluation_splits.insert(0, primary_split)
    if fast:
        evaluation_splits = [primary_split]

    menin_measurements = _read_csv(processed / "menin_activity_measurements.csv")
    menin_compounds = _read_csv(processed / "menin_compounds_curated.csv")
    herg_compounds = _read_csv(processed / "herg_compounds_curated.csv")
    if menin_measurements.empty or herg_compounds.empty:
        raise FileNotFoundError("Curated Menin and hERG tables are required before modeling.")

    settings_yaml = settings_snapshot(settings)
    processed_manifest_path = manifest_reports_dir / "manifests" / "processed_manifest.json"
    processed_manifest = (
        json.loads(processed_manifest_path.read_text(encoding="utf-8"))
        if processed_manifest_path.exists()
        else {}
    )
    software_manifest_path = manifest_reports_dir / "manifests" / "software_manifest.json"
    software_manifest = (
        json.loads(software_manifest_path.read_text(encoding="utf-8"))
        if software_manifest_path.exists()
        else {}
    )
    provenance_context = {
        "git_revision": _git_revision(_project_root(settings)),
        "git_dirty": _git_dirty(_project_root(settings)),
        "fast_mode": bool(fast),
        "resolved_settings": settings,
        "resolved_settings_sha256": hashlib.sha256(settings_yaml.encode("utf-8")).hexdigest(),
        "processed_build_id": processed_manifest.get("build_id"),
        "processed_dataset_sha256": processed_manifest.get("dataset_sha256"),
        "processed_manifest_sha256": (
            hashlib.sha256(processed_manifest_path.read_bytes()).hexdigest()
            if processed_manifest_path.exists()
            else None
        ),
        "software_dataset_sha256": software_manifest.get("dataset_sha256"),
        "software_manifest_sha256": (
            hashlib.sha256(software_manifest_path.read_bytes()).hexdigest()
            if software_manifest_path.exists()
            else None
        ),
    }

    for directory in (
        paths["models"] / "evaluations",
        paths["models"] / "endpoints",
        paths["models"] / "additional_tasks",
        paths["models"] / "herg_sensitivity",
        paths["models"] / "sensitivity",
        paths["reports"] / "model_evaluations",
        paths["reports"] / "endpoint_models",
        paths["reports"] / "additional_task_models",
        paths["reports"] / "herg_sensitivity",
        paths["reports"] / "sensitivity",
    ):
        if directory.exists():
            shutil.rmtree(directory)
    for root, patterns in (
        (
            paths["models"],
            ("menin_activity_*", "herg_classifier_*", "herg_liability_*"),
        ),
        (
            paths["reports"],
            (
                "menin_activity_*",
                "herg_classifier_*",
                "menin_with_predicted_herg_risk.csv",
                "model_validation_summary.*",
            ),
        ),
    ):
        for pattern in patterns:
            for artifact in root.glob(pattern):
                if artifact.is_file():
                    artifact.unlink()

    for legacy in (
        paths["models"] / "menin_activity_ridge.pkl",
        paths["models"] / "herg_liability_logistic.pkl",
        paths["reports"] / "menin_activity_model_metrics.json",
        paths["reports"] / "menin_activity_model_test_predictions.csv",
    ):
        legacy.unlink(missing_ok=True)

    common: dict[str, Any] = {
        "random_state": random_state,
        "test_size": float(model_cfg.get("test_size", 0.2)),
        "feature_backend": str(model_cfg.get("feature_backend", "rdkit")),
        "feature_n_bits": int(512 if fast else model_cfg.get("fingerprint_bits", 2048)),
        "feature_radius": int(model_cfg.get("fingerprint_radius", 2)),
        "applicability_domain_quantile": float(model_cfg.get("applicability_domain_quantile", 0.05)),
        "cv_folds": int(model_cfg.get("cv_folds", 3)),
        "bootstrap_iterations": int(50 if fast else model_cfg.get("bootstrap_iterations", 500)),
        "tree_estimators": int(50 if fast else model_cfg.get("tree_estimators", 200)),
        "provenance_context": provenance_context,
        "artifact_build_root": paths["models"],
        "artifact_release_root": release_models_dir,
        "artifact_project_root": _project_root(settings),
    }
    menin_common: dict[str, Any] = {
        **common,
        "prediction_interval_coverage": float(model_cfg.get("uncertainty_coverage", 0.90)),
        "min_samples": int(model_cfg.get("min_regression_compounds", 80)),
        "heterogeneity_log_spread_threshold": float(
            settings.get("curation", {}).get("max_within_compound_log_spread", 2.0)
        ),
    }
    herg_common: dict[str, Any] = {
        **common,
        "min_samples": int(model_cfg.get("min_classification_compounds", 120)),
    }

    all_metrics: dict[str, Any] = {"primary_endpoint": primary_endpoint, "evaluations": {}}
    comparison_rows: list[dict[str, Any]] = []
    for split in evaluation_splits:
        is_primary = split == primary_split
        model_dir = paths["models"] if is_primary else paths["models"] / "evaluations" / split
        report_dir = paths["reports"] if is_primary else paths["reports"] / "model_evaluations" / split
        menin_metrics = train_menin_activity_model(
            menin_measurements,
            model_dir,
            report_dir,
            split_strategy=split,
            endpoint=primary_endpoint,
            assay_family=primary_assay_family,
            **menin_common,
        )
        herg_metrics = train_herg_classifier_and_predict(
            herg_compounds,
            menin_compounds,
            model_dir,
            report_dir,
            split_strategy=split,
            endpoint=primary_herg_endpoint,
            assay_family=primary_herg_assay_family,
            menin_endpoint=primary_endpoint,
            menin_assay_family=primary_assay_family,
            **herg_common,
        )
        all_metrics["evaluations"][split] = {
            "menin_activity": menin_metrics,
            "herg_liability": herg_metrics,
        }
        comparison_rows.extend(
            [
                _metric_row(f"menin_{primary_endpoint}_{primary_assay_family}", split, menin_metrics),
                _metric_row(
                    f"herg_{primary_herg_endpoint}_{primary_herg_assay_family}",
                    split,
                    herg_metrics,
                ),
            ]
        )

    endpoint_metrics: dict[str, Any] = {}
    trained_tasks: set[tuple[str, str]] = {(primary_endpoint.casefold(), primary_assay_family.casefold())}
    endpoints = [str(value) for value in model_cfg.get("menin_endpoints", ["IC50", "Ki", "Kd", "EC50"])]
    if fast:
        endpoints = [primary_endpoint]
    for endpoint in endpoints:
        assay_family = task_families.get(endpoint.casefold())
        if endpoint.casefold() == primary_endpoint.casefold():
            endpoint_metrics[endpoint] = all_metrics["evaluations"][primary_split]["menin_activity"]
            continue
        metrics = train_menin_activity_model(
            menin_measurements,
            paths["models"] / "endpoints" / endpoint.lower(),
            paths["reports"] / "endpoint_models" / endpoint.lower(),
            split_strategy=primary_split,
            endpoint=endpoint,
            assay_family=assay_family,
            **menin_common,
        )
        endpoint_metrics[endpoint] = metrics
        if assay_family:
            trained_tasks.add((endpoint.casefold(), assay_family.casefold()))
        task_name = f"menin_{endpoint}_{assay_family}" if assay_family else f"menin_{endpoint}"
        comparison_rows.append(_metric_row(task_name, primary_split, metrics))
    all_metrics["endpoint_models"] = endpoint_metrics

    additional_metrics: dict[str, Any] = {}
    if not fast and model_cfg.get("analyze_all_eligible_endpoint_assay_tasks", True):
        eligible = menin_measurements.copy()
        if "is_modeling_eligible" in eligible:
            eligible = eligible[
                eligible["is_modeling_eligible"].astype(str).str.casefold().isin({"true", "1", "yes"})
            ]
        pairs = (
            eligible[["endpoint", "assay_family"]]
            .dropna()
            .drop_duplicates()
            .sort_values(["endpoint", "assay_family"])
            .itertuples(index=False, name=None)
        )
        for endpoint, assay_family in pairs:
            key = (str(endpoint).casefold(), str(assay_family).casefold())
            if key in trained_tasks:
                continue
            task = prepare_menin_task(
                menin_measurements,
                endpoint=str(endpoint),
                assay_family=str(assay_family),
            )
            if len(task) < int(model_cfg.get("min_regression_compounds", 80)):
                continue
            slug = f"{str(endpoint).lower()}_{str(assay_family).lower()}"
            metrics = train_menin_activity_model(
                menin_measurements,
                paths["models"] / "additional_tasks" / slug,
                paths["reports"] / "additional_task_models" / slug,
                split_strategy=primary_split,
                endpoint=str(endpoint),
                assay_family=str(assay_family),
                **menin_common,
            )
            name = f"{endpoint}_{assay_family}"
            additional_metrics[name] = metrics
            comparison_rows.append(_metric_row(f"menin_{name}", primary_split, metrics))
    all_metrics["additional_task_models"] = additional_metrics

    if not fast and model_cfg.get("run_cross_source_mirror_sensitivity", True):
        mirror_sensitivity = train_menin_activity_model(
            menin_measurements,
            paths["models"] / "sensitivity" / "cross_source_mirrors_retained",
            paths["reports"] / "sensitivity" / "cross_source_mirrors_retained",
            split_strategy=primary_split,
            endpoint=primary_endpoint,
            assay_family=primary_assay_family,
            collapse_cross_source_mirrors=False,
            **menin_common,
        )
        all_metrics["menin_cross_source_mirror_sensitivity"] = mirror_sensitivity
        comparison_rows.append(
            _metric_row(
                "menin_cross_source_mirrors_retained_sensitivity",
                primary_split,
                mirror_sensitivity,
            )
        )

    if not fast and model_cfg.get("run_clean_label_sensitivity", True):
        clean_label_sensitivity = train_menin_activity_model(
            menin_measurements,
            paths["models"] / "sensitivity" / "clean_labels",
            paths["reports"] / "sensitivity" / "clean_labels",
            split_strategy=primary_split,
            endpoint=primary_endpoint,
            assay_family=primary_assay_family,
            exclude_heterogeneous_labels=True,
            **menin_common,
        )
        all_metrics["menin_clean_label_sensitivity"] = clean_label_sensitivity
        comparison_rows.append(
            _metric_row(
                "menin_clean_label_sensitivity",
                primary_split,
                clean_label_sensitivity,
            )
        )

    if not fast and herg_cfg.get("run_pooled_sensitivity", True):
        pooled_metrics = train_herg_classifier_and_predict(
            herg_compounds,
            menin_compounds,
            paths["models"] / "herg_sensitivity" / "pooled",
            paths["reports"] / "herg_sensitivity" / "pooled",
            split_strategy=primary_split,
            endpoint=None,
            assay_family=None,
            menin_endpoint=primary_endpoint,
            menin_assay_family=primary_assay_family,
            **herg_common,
        )
        all_metrics["herg_pooled_sensitivity"] = pooled_metrics
        comparison_rows.append(_metric_row("herg_pooled_sensitivity", primary_split, pooled_metrics))

    paths["reports"].mkdir(parents=True, exist_ok=True)
    pd.DataFrame(comparison_rows).to_csv(paths["reports"] / "model_validation_summary.csv", index=False)
    _write_json(paths["reports"] / "model_validation_summary.json", all_metrics)
    return all_metrics


def run_model_stage_transactionally(
    settings: dict[str, Any],
    *,
    split_override: str | None = None,
    endpoint_override: str | None = None,
    assay_family_override: str | None = None,
    fast: bool = False,
) -> dict[str, Any]:
    """Build models/reports in staging and promote only a complete model stage."""

    paths = _paths(settings)
    staging_root = Path(tempfile.mkdtemp(prefix=".menin-model-build-", dir=_project_root(settings)))
    staged_models = staging_root / "models"
    staged_reports = staging_root / "reports"
    try:
        metrics = run_model_stage(
            settings,
            split_override=split_override,
            endpoint_override=endpoint_override,
            assay_family_override=assay_family_override,
            fast=fast,
            models_output_dir=staged_models,
            reports_output_dir=staged_reports,
            provenance_reports_dir=paths["reports"],
        )
    except Exception:
        shutil.rmtree(staging_root, ignore_errors=True)
        raise

    previous_models = staging_root / "previous_models"
    previous_reports = staging_root / "previous_reports"
    previous_reports.mkdir(parents=True, exist_ok=True)
    promoted_reports: list[str] = []
    backed_up_reports: list[str] = []
    models_promoted = False
    models_backed_up = False
    try:
        paths["models"].parent.mkdir(parents=True, exist_ok=True)
        if paths["models"].exists():
            os.replace(paths["models"], previous_models)
            models_backed_up = True
        os.replace(staged_models, paths["models"])
        models_promoted = True

        paths["reports"].mkdir(parents=True, exist_ok=True)
        model_report_names = {path.name for path in staged_reports.iterdir()}
        for directory_name in (
            "model_evaluations",
            "endpoint_models",
            "additional_task_models",
            "herg_sensitivity",
            "sensitivity",
        ):
            if (paths["reports"] / directory_name).exists():
                model_report_names.add(directory_name)
        for pattern in (
            "menin_activity_*",
            "herg_classifier_*",
            "herg_liability_*",
            "menin_with_predicted_herg_risk.csv",
            "model_validation_summary.*",
        ):
            model_report_names.update(path.name for path in paths["reports"].glob(pattern))
        # Remove the complete previous model-report surface, including outputs
        # for analyses disabled in the new configuration, while retaining
        # quality, descriptive, and provenance reports owned by other stages.
        for name in sorted(model_report_names):
            destination = paths["reports"] / name
            if destination.exists():
                os.replace(destination, previous_reports / name)
                backed_up_reports.append(name)
        for staged in sorted(staged_reports.iterdir(), key=lambda path: path.name):
            destination = paths["reports"] / staged.name
            try:
                os.replace(staged, destination)
            except Exception:
                raise
            promoted_reports.append(staged.name)
    except Exception:
        for name in reversed(promoted_reports):
            destination = paths["reports"] / name
            if destination.is_dir():
                shutil.rmtree(destination)
            elif destination.exists():
                destination.unlink()
        for name in reversed(backed_up_reports):
            backup = previous_reports / name
            destination = paths["reports"] / name
            if backup.exists() and not destination.exists():
                os.replace(backup, destination)
        if models_promoted:
            if paths["models"].exists():
                shutil.rmtree(paths["models"])
            if previous_models.exists():
                os.replace(previous_models, paths["models"])
        elif models_backed_up and previous_models.exists() and not paths["models"].exists():
            os.replace(previous_models, paths["models"])
        raise
    shutil.rmtree(staging_root, ignore_errors=True)
    return metrics


SOFTWARE_MANIFEST_EXCLUDES = (
    "*/__pycache__/*",
    "*/.pytest_cache/*",
    "*/.ruff_cache/*",
    "*/.mypy_cache/*",
    "*/.cache/*",
    "*.egg-info/*",
    "*.pyc",
    ".DS_Store",
)
MODEL_MANIFEST_EXCLUDES = ("smoke/*", ".DS_Store")
ANALYSIS_MANIFEST_EXCLUDES = (
    "analysis_build_metadata.json",
    "smoke/*",
    ".DS_Store",
)
REPORT_MANIFEST_EXCLUDES = (
    "manifests/*",
    "verification/*",
    "run_metadata.json",
    "run_metadata/*",
    "report_build_metadata.json",
    "smoke/*",
    ".DS_Store",
)


def _load_json_document(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    document = json.loads(path.read_text(encoding="utf-8"))
    return document if isinstance(document, dict) else {}


def _software_paths(project_root: Path) -> list[Path]:
    candidates = (
        "pipeline/src/menin_discovery",
        "pipeline/scripts",
        "pipeline/config",
        "pipeline/tests",
        "pipeline/environments",
        "packages/menin-edit",
        "docs",
        ".github",
        ".gitignore",
        ".gitattributes",
        ".pre-commit-config.yaml",
        "CITATION.cff",
        "LICENSE",
        "Makefile",
        "README.md",
        "pyproject.toml",
    )
    return [project_root / name for name in candidates if (project_root / name).exists()]


def _stable_data_manifests(paths: dict[str, Path]) -> dict[str, dict[str, Any]]:
    manifest_dir = paths["reports"] / "manifests"
    candidate = create_data_manifests(paths["raw"], paths["processed"])
    existing_raw = _load_json_document(manifest_dir / "raw_manifest.json")
    existing_processed = _load_json_document(manifest_dir / "processed_manifest.json")
    if (
        existing_raw.get("dataset_sha256") == candidate["raw"].get("dataset_sha256")
        and existing_processed.get("dataset_sha256") == candidate["processed"].get("dataset_sha256")
        and existing_processed.get("build_id") == candidate["processed"].get("build_id")
    ):
        created_at = existing_processed.get("created_at") or existing_raw.get("created_at")
        if created_at:
            candidate = create_data_manifests(paths["raw"], paths["processed"], created_at=str(created_at))
    write_manifest(candidate["raw"], manifest_dir / "raw_manifest.json")
    write_manifest(candidate["processed"], manifest_dir / "processed_manifest.json")
    return candidate


def _stable_software_manifest(
    settings: dict[str, Any], processed: dict[str, Any], manifest_dir: Path
) -> dict[str, Any]:
    project_root = _project_root(settings)
    existing = _load_json_document(manifest_dir / "software_manifest.json")
    kwargs: dict[str, Any] = {
        "paths": _software_paths(project_root),
        "root": project_root,
        "stage": "software",
        "build_id": str(processed["build_id"]),
        "exclude": SOFTWARE_MANIFEST_EXCLUDES,
    }
    candidate = create_manifest(**kwargs)
    if (
        existing.get("dataset_sha256") == candidate.get("dataset_sha256")
        and existing.get("build_id") == candidate.get("build_id")
        and existing.get("created_at")
    ):
        candidate = create_manifest(**kwargs, created_at=str(existing["created_at"]))
    write_manifest(candidate, manifest_dir / "software_manifest.json")
    return candidate


def _model_upstreams(
    processed: dict[str, Any], software: dict[str, Any]
) -> tuple[dict[str, str], dict[str, str]]:
    return (
        {"stage": "processed", "dataset_sha256": str(processed["dataset_sha256"])},
        {"stage": "software", "dataset_sha256": str(software["dataset_sha256"])},
    )


def _validate_model_lineage(
    settings: dict[str, Any],
    processed: dict[str, Any],
    software: dict[str, Any],
    *,
    processed_manifest_path: Path,
) -> None:
    paths = _paths(settings)
    models_dir = paths["models"]
    project_root = _project_root(settings)
    manifest_paths = [
        path
        for path in models_dir.rglob("*_manifest.json")
        if "smoke" not in path.relative_to(models_dir).parts
    ]
    if not manifest_paths:
        raise RuntimeError("No release model manifests were found; train models before manifesting.")
    referenced_artifacts: set[Path] = set()
    expected_processed_manifest_sha = sha256_file(processed_manifest_path)
    software_manifest_path = processed_manifest_path.with_name("software_manifest.json")
    expected_software_manifest_sha = sha256_file(software_manifest_path)
    for manifest_path in sorted(manifest_paths):
        document = _load_json_document(manifest_path)
        provenance = document.get("provenance", {})
        mismatches = {
            "processed_build_id": (
                provenance.get("processed_build_id"),
                processed.get("build_id"),
            ),
            "processed_dataset_sha256": (
                provenance.get("processed_dataset_sha256"),
                processed.get("dataset_sha256"),
            ),
            "processed_manifest_sha256": (
                provenance.get("processed_manifest_sha256"),
                expected_processed_manifest_sha,
            ),
            "software_dataset_sha256": (
                provenance.get("software_dataset_sha256"),
                software.get("dataset_sha256"),
            ),
            "software_manifest_sha256": (
                provenance.get("software_manifest_sha256"),
                expected_software_manifest_sha,
            ),
        }
        failed = {
            key: {"actual": actual, "expected": expected}
            for key, (actual, expected) in mismatches.items()
            if actual != expected
        }
        if failed:
            raise RuntimeError(
                f"Model lineage mismatch in {manifest_path.relative_to(models_dir)}: "
                f"{json.dumps(failed, sort_keys=True)}"
            )
        artifact = document.get("artifact", {})
        filename = str(artifact.get("filename", ""))
        artifact_path = manifest_path.parent / filename
        if not filename or not artifact_path.is_file():
            raise RuntimeError(f"Missing model artifact declared by {manifest_path}.")
        if sha256_file(artifact_path) != artifact.get("sha256"):
            raise RuntimeError(f"Model artifact hash mismatch for {artifact_path}.")
        try:
            expected_path = artifact_path.resolve().relative_to(project_root.resolve()).as_posix()
            expected_relative = True
        except ValueError:
            expected_path = artifact_path.name
            expected_relative = False
        if (
            artifact.get("path") != expected_path
            or bool(artifact.get("path_is_repository_relative")) != expected_relative
        ):
            raise RuntimeError(
                f"Non-portable or stale artifact path in {manifest_path.relative_to(models_dir)}."
            )
        referenced_artifacts.add(artifact_path.resolve())
    artifact_files = {
        path.resolve()
        for pattern in ("*.skops", "*.joblib")
        for path in models_dir.rglob(pattern)
        if "smoke" not in path.relative_to(models_dir).parts
    }
    if artifact_files != referenced_artifacts:
        missing = sorted(str(path) for path in artifact_files - referenced_artifacts)
        absent = sorted(str(path) for path in referenced_artifacts - artifact_files)
        raise RuntimeError(
            "Every release model artifact must be covered by exactly one model manifest; "
            f"unreferenced={missing}, missing={absent}."
        )
    unsupported = [
        path.relative_to(models_dir).as_posix()
        for path in models_dir.rglob("*")
        if path.is_file()
        and "smoke" not in path.relative_to(models_dir).parts
        and path.suffix.casefold() not in {".json", ".skops", ".joblib"}
    ]
    if unsupported:
        raise RuntimeError(f"Unsupported or stale release-model files: {sorted(unsupported)}")


def _model_manifest_candidate(
    paths: dict[str, Path], processed: dict[str, Any], software: dict[str, Any]
) -> dict[str, Any]:
    return create_manifest(
        paths["models"],
        stage="models",
        build_id=str(processed["build_id"]),
        exclude=MODEL_MANIFEST_EXCLUDES,
        upstream=_model_upstreams(processed, software),
    )


def _analysis_manifest_candidate(
    analysis_dir: Path,
    processed: dict[str, Any],
    software: dict[str, Any],
    models: dict[str, Any],
) -> dict[str, Any]:
    return create_manifest(
        analysis_dir,
        stage="analysis",
        build_id=str(processed["build_id"]),
        exclude=ANALYSIS_MANIFEST_EXCLUDES,
        upstream=(
            {"stage": "processed", "dataset_sha256": processed["dataset_sha256"]},
            {"stage": "models", "dataset_sha256": models["dataset_sha256"]},
            {"stage": "software", "dataset_sha256": software["dataset_sha256"]},
        ),
    )


def _report_manifest_candidate(
    reports_dir: Path,
    processed: dict[str, Any],
    software: dict[str, Any],
    models: dict[str, Any],
    analysis: dict[str, Any] | None = None,
) -> dict[str, Any]:
    upstream = [
        {"stage": "processed", "dataset_sha256": processed["dataset_sha256"]},
        {"stage": "models", "dataset_sha256": models["dataset_sha256"]},
        {"stage": "software", "dataset_sha256": software["dataset_sha256"]},
    ]
    if analysis is not None:
        upstream.append({"stage": "analysis", "dataset_sha256": analysis["dataset_sha256"]})
    return create_manifest(
        reports_dir,
        stage="reports",
        build_id=str(processed["build_id"]),
        exclude=REPORT_MANIFEST_EXCLUDES,
        upstream=upstream,
    )


def _settings_sha256(settings: dict[str, Any]) -> str:
    return hashlib.sha256(settings_snapshot(settings).encode("utf-8")).hexdigest()


def _expected_analysis_metadata(
    settings: dict[str, Any],
    processed: dict[str, Any],
    software: dict[str, Any],
    models: dict[str, Any],
    analysis: dict[str, Any],
) -> dict[str, Any]:
    return {
        "build_id": processed["build_id"],
        "processed_dataset_sha256": processed["dataset_sha256"],
        "software_dataset_sha256": software["dataset_sha256"],
        "models_dataset_sha256": models["dataset_sha256"],
        "analysis_dataset_sha256": analysis["dataset_sha256"],
        "resolved_settings_sha256": _settings_sha256(settings),
    }


def _expected_report_metadata(
    settings: dict[str, Any],
    processed: dict[str, Any],
    software: dict[str, Any],
    models: dict[str, Any],
    reports: dict[str, Any],
    analysis: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = {
        "build_id": processed["build_id"],
        "processed_dataset_sha256": processed["dataset_sha256"],
        "software_dataset_sha256": software["dataset_sha256"],
        "models_dataset_sha256": models["dataset_sha256"],
        "reports_dataset_sha256": reports["dataset_sha256"],
        "resolved_settings_sha256": _settings_sha256(settings),
    }
    if analysis is not None:
        metadata["analysis_dataset_sha256"] = analysis["dataset_sha256"]
    return metadata


def _metadata_mismatches(actual: dict[str, Any], expected: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        key: {"actual": actual.get(key), "expected": value}
        for key, value in expected.items()
        if actual.get(key) != value
    }


def _validate_analysis_lineage(
    settings: dict[str, Any],
    processed: dict[str, Any],
    software: dict[str, Any],
    models: dict[str, Any],
) -> dict[str, Any]:
    """Validate the analysis directory against every declared upstream digest."""

    if not _analysis_enabled(settings):
        raise RuntimeError("The analysis stage is not enabled in the resolved settings.")
    paths = _paths(settings)
    analysis_dir = paths["analysis"]
    if not analysis_dir.is_dir():
        raise RuntimeError("Analysis outputs are missing; run the analyze stage first.")
    analysis = _analysis_manifest_candidate(
        analysis_dir,
        processed,
        software,
        models,
    )
    metadata = _load_json_document(analysis_dir / "analysis_build_metadata.json")
    expected = _expected_analysis_metadata(
        settings,
        processed,
        software,
        models,
        analysis,
    )
    mismatches = _metadata_mismatches(metadata, expected)
    summary = _load_json_document(analysis_dir / "analysis_summary.json")
    recorded_input_sha256 = summary.get("input_sha256", {})
    if not isinstance(recorded_input_sha256, dict):
        recorded_input_sha256 = {}
    direct_inputs = {
        "scored_menin_herg": paths["reports"] / "menin_with_predicted_herg_risk.csv",
        "menin_measurements": paths["processed"] / "menin_activity_measurements.csv",
        "observed_herg": paths["processed"] / "herg_compounds_curated.csv",
        "pk_admet": paths["processed"] / "pk_admet_observations.csv",
    }
    current_input_sha256 = {
        name: sha256_file(path) if path.is_file() else None for name, path in sorted(direct_inputs.items())
    }
    if recorded_input_sha256 != current_input_sha256:
        mismatches["input_sha256"] = {
            "actual": recorded_input_sha256,
            "expected": current_input_sha256,
        }
    if mismatches:
        raise RuntimeError(
            "Analysis lineage is missing or stale; rerun the analyze stage: "
            + json.dumps(mismatches, sort_keys=True)
        )
    return analysis


def run_analysis_stage(settings: dict[str, Any]) -> Path:
    """Build chemical-intelligence outputs and promote the whole directory atomically."""

    if not _analysis_enabled(settings):
        raise RuntimeError("The analyze stage requires analysis.enabled: true.")

    from .analysis import write_chemical_intelligence

    paths = _paths(settings)
    manifest_dir = paths["reports"] / "manifests"
    manifests = run_manifest_stage(settings, include_analysis_artifacts=False)
    processed = manifests["processed"]
    software = manifests["software"]
    _validate_model_lineage(
        settings,
        processed,
        software,
        processed_manifest_path=manifest_dir / "processed_manifest.json",
    )
    models = _model_manifest_candidate(paths, processed, software)

    analysis_target = paths["analysis"]
    analysis_target.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(tempfile.mkdtemp(prefix=".menin-analysis-build-", dir=analysis_target.parent))
    staged_analysis = staging_root / "analysis"
    staged_analysis.mkdir()
    try:
        write_chemical_intelligence(
            paths["processed"],
            paths["reports"],
            staged_analysis,
            settings=settings,
        )
        analysis = _analysis_manifest_candidate(
            staged_analysis,
            processed,
            software,
            models,
        )
        _write_json(
            staged_analysis / "analysis_build_metadata.json",
            _expected_analysis_metadata(
                settings,
                processed,
                software,
                models,
                analysis,
            ),
        )
        previous = staging_root / "previous"
        promoted = False
        backed_up = False
        try:
            if analysis_target.exists():
                os.replace(analysis_target, previous)
                backed_up = True
            os.replace(staged_analysis, analysis_target)
            promoted = True
        except Exception:
            if promoted and analysis_target.exists():
                if analysis_target.is_dir():
                    shutil.rmtree(analysis_target)
                else:
                    analysis_target.unlink()
            if backed_up and previous.exists() and not analysis_target.exists():
                os.replace(previous, analysis_target)
            raise
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)
    return analysis_target


def run_report_stage(settings: dict[str, Any]) -> Path:
    """Generate descriptive/publication reports in staging and promote atomically."""

    from .reporting import write_summary_report

    paths = _paths(settings)
    manifest_dir = paths["reports"] / "manifests"
    run_manifest_stage(settings, include_analysis_artifacts=False)
    processed = _load_json_document(manifest_dir / "processed_manifest.json")
    software = _load_json_document(manifest_dir / "software_manifest.json")
    _validate_model_lineage(
        settings,
        processed,
        software,
        processed_manifest_path=manifest_dir / "processed_manifest.json",
    )
    model_snapshot = _model_manifest_candidate(paths, processed, software)
    analysis_snapshot = (
        _validate_analysis_lineage(settings, processed, software, model_snapshot)
        if _analysis_enabled(settings)
        else None
    )
    staging_root = Path(tempfile.mkdtemp(prefix=".menin-report-build-", dir=_project_root(settings)))
    staged_reports = staging_root / "reports"
    try:
        if paths["reports"].exists():
            shutil.copytree(paths["reports"], staged_reports)
        else:
            staged_reports.mkdir(parents=True)
        for name in ("tables", "figures", "publication_summary.md", "summary.md"):
            stale = staged_reports / name
            if stale.is_dir():
                shutil.rmtree(stale)
            elif stale.exists():
                stale.unlink()
        report_kwargs: dict[str, Any] = {"settings": settings}
        if analysis_snapshot is not None:
            report_kwargs["analysis_dir"] = paths["analysis"]
        write_summary_report(
            paths["processed"],
            staged_reports,
            paths["models"],
            **report_kwargs,
        )
        report_snapshot = _report_manifest_candidate(
            staged_reports,
            processed,
            software,
            model_snapshot,
            analysis_snapshot,
        )
        _write_json(
            staged_reports / "report_build_metadata.json",
            _expected_report_metadata(
                settings,
                processed,
                software,
                model_snapshot,
                report_snapshot,
                analysis_snapshot,
            ),
        )
    except Exception:
        shutil.rmtree(staging_root, ignore_errors=True)
        raise
    report_owned = (
        "tables",
        "figures",
        "publication_summary.md",
        "summary.md",
        "report_build_metadata.json",
    )
    previous = staging_root / "previous"
    previous.mkdir()
    promoted: list[str] = []
    backed_up: list[str] = []
    try:
        paths["reports"].mkdir(parents=True, exist_ok=True)
        for name in report_owned:
            destination = paths["reports"] / name
            if destination.exists():
                os.replace(destination, previous / name)
                backed_up.append(name)
        for name in report_owned:
            staged = staged_reports / name
            if staged.exists():
                os.replace(staged, paths["reports"] / name)
                promoted.append(name)
    except Exception:
        for name in reversed(promoted):
            destination = paths["reports"] / name
            if destination.is_dir():
                shutil.rmtree(destination)
            elif destination.exists():
                destination.unlink()
        for name in reversed(backed_up):
            if (previous / name).exists():
                os.replace(previous / name, paths["reports"] / name)
        raise
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)
    return paths["reports"] / "publication_summary.md"


def run_manifest_stage(
    settings: dict[str, Any], *, include_analysis_artifacts: bool = True
) -> dict[str, dict[str, Any]]:
    """Write data/software manifests and, optionally, the complete release DAG.

    ``include_analysis_artifacts`` is retained as a public compatibility name;
    historically it selected the downstream model/report manifests.
    """

    paths = _paths(settings)
    manifest_dir = paths["reports"] / "manifests"
    manifests = _stable_data_manifests(paths)
    processed = manifests["processed"]
    software_manifest = _stable_software_manifest(settings, processed, manifest_dir)
    manifests["software"] = software_manifest
    if not include_analysis_artifacts:
        return manifests
    _validate_model_lineage(
        settings,
        processed,
        software_manifest,
        processed_manifest_path=manifest_dir / "processed_manifest.json",
    )
    model_manifest = _model_manifest_candidate(paths, processed, software_manifest)
    analysis_manifest = (
        _validate_analysis_lineage(
            settings,
            processed,
            software_manifest,
            model_manifest,
        )
        if _analysis_enabled(settings)
        else None
    )
    report_manifest = _report_manifest_candidate(
        paths["reports"],
        processed,
        software_manifest,
        model_manifest,
        analysis_manifest,
    )
    report_metadata = _load_json_document(paths["reports"] / "report_build_metadata.json")
    expected_report_metadata = _expected_report_metadata(
        settings,
        processed,
        software_manifest,
        model_manifest,
        report_manifest,
        analysis_manifest,
    )
    mismatches = _metadata_mismatches(report_metadata, expected_report_metadata)
    if mismatches:
        raise RuntimeError(
            "Report lineage is missing or stale; rerun the report stage before manifesting: "
            + json.dumps(mismatches, sort_keys=True)
        )
    write_manifest(model_manifest, manifest_dir / "models_manifest.json")
    if analysis_manifest is not None:
        write_manifest(analysis_manifest, manifest_dir / "analysis_manifest.json")
    write_manifest(report_manifest, manifest_dir / "reports_manifest.json")
    manifests["models"] = model_manifest
    if analysis_manifest is not None:
        manifests["analysis"] = analysis_manifest
    manifests["reports"] = report_manifest
    return manifests


def run_verify_stage(settings: dict[str, Any]) -> dict[str, bool]:
    paths = _paths(settings)
    manifest_dir = paths["reports"] / "manifests"
    required_stages = (
        "raw",
        "processed",
        "software",
        "models",
        *(("analysis",) if _analysis_enabled(settings) else ()),
        "reports",
    )
    manifest_paths = {stage: manifest_dir / f"{stage}_manifest.json" for stage in required_stages}
    missing = [stage for stage, path in manifest_paths.items() if not path.exists()]
    if missing:
        verification_dir = paths["reports"] / "verification"
        for stage in missing:
            result = ManifestVerification(
                stage=stage,
                issues=[
                    VerificationIssue(
                        code="missing_manifest",
                        message=f"Required {stage} release manifest is missing.",
                        path=str(manifest_paths[stage]),
                    )
                ],
            )
            result.write_json(verification_dir / f"{stage}_verification.json")
            result.write_csv(verification_dir / f"{stage}_verification_issues.csv")
        raise RuntimeError("Missing required release manifests: " + ", ".join(missing))

    results = verify_data_manifests(
        {"raw": manifest_paths["raw"], "processed": manifest_paths["processed"]},
        raw_root=paths["raw"],
        processed_root=paths["processed"],
        allow_extra=False,
    )
    manifest_roots: list[tuple[str, Path]] = [
        ("software", _project_root(settings)),
        ("models", paths["models"]),
    ]
    if _analysis_enabled(settings):
        manifest_roots.append(("analysis", paths["analysis"]))
    manifest_roots.append(("reports", paths["reports"]))
    for stage, root in manifest_roots:
        results[stage] = verify_manifest(
            manifest_paths[stage],
            root=root,
            allow_extra=True,
        )
    manifest_documents = {stage: _load_json_document(path) for stage, path in manifest_paths.items()}
    processed_document = manifest_documents["processed"]
    software_document = manifest_documents["software"]
    models_document = manifest_documents["models"]
    analysis_document = manifest_documents.get("analysis")
    reports_document = manifest_documents["reports"]
    if software_document.get("build_id") != processed_document.get("build_id"):
        results["software"].issues.append(
            VerificationIssue(
                code="build_id_mismatch",
                message="Software manifest build ID does not match processed data.",
                expected=processed_document.get("build_id"),
                actual=software_document.get("build_id"),
            )
        )
    downstream_stages = ["models"]
    if analysis_document is not None:
        downstream_stages.append("analysis")
    downstream_stages.append("reports")
    expected_upstreams: dict[str, tuple[tuple[str, Any], ...]] = {
        "models": (
            ("processed", processed_document.get("dataset_sha256")),
            ("software", software_document.get("dataset_sha256")),
        ),
        "reports": (
            ("processed", processed_document.get("dataset_sha256")),
            ("models", models_document.get("dataset_sha256")),
            ("software", software_document.get("dataset_sha256")),
            *(
                (("analysis", analysis_document.get("dataset_sha256")),)
                if analysis_document is not None
                else ()
            ),
        ),
    }
    if analysis_document is not None:
        expected_upstreams["analysis"] = (
            ("processed", processed_document.get("dataset_sha256")),
            ("models", models_document.get("dataset_sha256")),
            ("software", software_document.get("dataset_sha256")),
        )

    for stage in downstream_stages:
        document = manifest_documents[stage]
        if document.get("build_id") != processed_document.get("build_id"):
            results[stage].issues.append(
                VerificationIssue(
                    code="build_id_mismatch",
                    message=f"{stage} manifest build ID does not match processed data.",
                    expected=processed_document.get("build_id"),
                    actual=document.get("build_id"),
                )
            )
        upstream = document.get("upstream", [])
        for upstream_stage, upstream_digest in expected_upstreams[stage]:
            if not any(
                isinstance(item, dict)
                and item.get("stage") == upstream_stage
                and item.get("dataset_sha256") == upstream_digest
                for item in upstream
            ):
                results[stage].issues.append(
                    VerificationIssue(
                        code="upstream_manifest_mismatch",
                        message=(f"{stage} manifest is not linked to the {upstream_stage} dataset digest."),
                        expected=upstream_digest,
                        actual=upstream,
                    )
                )

    current_software = create_manifest(
        _software_paths(_project_root(settings)),
        root=_project_root(settings),
        stage="software",
        build_id=str(processed_document["build_id"]),
        exclude=SOFTWARE_MANIFEST_EXCLUDES,
    )
    if current_software.get("dataset_sha256") != software_document.get("dataset_sha256"):
        results["software"].issues.append(
            VerificationIssue(
                code="manifest_scope_changed",
                message="The current software release scope differs from its manifest.",
                expected=software_document.get("dataset_sha256"),
                actual=current_software.get("dataset_sha256"),
            )
        )
    current_models = _model_manifest_candidate(paths, processed_document, software_document)
    if current_models.get("dataset_sha256") != models_document.get("dataset_sha256"):
        results["models"].issues.append(
            VerificationIssue(
                code="manifest_scope_changed",
                message="The current release model bundle differs from its manifest.",
                expected=models_document.get("dataset_sha256"),
                actual=current_models.get("dataset_sha256"),
            )
        )
    try:
        _validate_model_lineage(
            settings,
            processed_document,
            software_document,
            processed_manifest_path=manifest_paths["processed"],
        )
    except RuntimeError as exc:
        results["models"].issues.append(VerificationIssue(code="model_lineage_invalid", message=str(exc)))
    current_analysis: dict[str, Any] | None = None
    if analysis_document is not None:
        if paths["analysis"].is_dir():
            current_analysis = _analysis_manifest_candidate(
                paths["analysis"],
                processed_document,
                software_document,
                models_document,
            )
            if current_analysis.get("dataset_sha256") != analysis_document.get("dataset_sha256"):
                results["analysis"].issues.append(
                    VerificationIssue(
                        code="manifest_scope_changed",
                        message="The current analysis bundle differs from its manifest.",
                        expected=analysis_document.get("dataset_sha256"),
                        actual=current_analysis.get("dataset_sha256"),
                    )
                )
        try:
            _validate_analysis_lineage(
                settings,
                processed_document,
                software_document,
                models_document,
            )
        except RuntimeError as exc:
            results["analysis"].issues.append(
                VerificationIssue(code="analysis_lineage_invalid", message=str(exc))
            )
    current_reports = _report_manifest_candidate(
        paths["reports"],
        processed_document,
        software_document,
        models_document,
        analysis_document,
    )
    if current_reports.get("dataset_sha256") != reports_document.get("dataset_sha256"):
        results["reports"].issues.append(
            VerificationIssue(
                code="manifest_scope_changed",
                message="The current release report bundle differs from its manifest.",
                expected=reports_document.get("dataset_sha256"),
                actual=current_reports.get("dataset_sha256"),
            )
        )
    report_metadata = _load_json_document(paths["reports"] / "report_build_metadata.json")
    expected_metadata = _expected_report_metadata(
        settings,
        processed_document,
        software_document,
        models_document,
        reports_document,
        analysis_document,
    )
    metadata_mismatches = _metadata_mismatches(report_metadata, expected_metadata)
    if metadata_mismatches:
        results["reports"].issues.append(
            VerificationIssue(
                code="report_lineage_invalid",
                message="Report build metadata does not match the verified release bundle.",
                expected=expected_metadata,
                actual=report_metadata,
            )
        )
    verification_dir = paths["reports"] / "verification"
    valid: dict[str, bool] = {}
    for stage, result in results.items():
        result.write_json(verification_dir / f"{stage}_verification.json")
        result.write_csv(verification_dir / f"{stage}_verification_issues.csv")
        valid[stage] = result.valid
    if not all(valid.values()):
        failed = ", ".join(stage for stage, passed in valid.items() if not passed)
        raise RuntimeError(f"Manifest verification failed for: {failed}")
    return valid


def _write_run_metadata(
    settings: dict[str, Any],
    args: argparse.Namespace,
    *,
    status: str,
    started_at: str,
    error: BaseException | None = None,
) -> None:
    paths = _paths(settings)
    metadata = {
        "generated_at": started_at,
        "started_at": started_at,
        "finished_at": _utc_now() if status != "started" else None,
        "status": status,
        "command": ["menin-pipeline", *sys.argv[1:]],
        "stage": args.stage,
        "git_revision": _git_revision(_project_root(settings)),
        "git_dirty": _git_dirty(_project_root(settings)),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "settings": settings,
        "settings_yaml": settings_snapshot(settings),
    }
    if error is not None:
        metadata["error"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
    _write_json(paths["reports"] / "run_metadata.json", metadata)
    _write_json(paths["reports"] / "run_metadata" / f"{args.stage}.json", metadata)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES, default="all", help="Pipeline stage to run.")
    parser.add_argument("--config", type=Path, help="YAML overrides merged over defaults.")
    parser.add_argument(
        "--project-root",
        type=Path,
        help="Resolve configured relative paths from this directory.",
    )
    parser.add_argument("--skip-network", action="store_true", help="Reuse the raw snapshot.")
    parser.add_argument("--overwrite-raw", action="store_true")
    parser.add_argument("--skip-pk", action="store_true")
    parser.add_argument("--fast", action="store_true", help="Reduced settings for a smoke test.")
    parser.add_argument(
        "--split-strategy",
        choices=("scaffold", "chemical", "temporal", "random"),
    )
    parser.add_argument("--menin-endpoint")
    parser.add_argument("--menin-assay-family")
    parser.add_argument("--fail-on-quality-errors", action="store_true")
    parser.add_argument("--max-menin-records", type=int, default=None)
    parser.add_argument("--max-herg-records", type=int, default=None)
    parser.add_argument("--max-pubchem-aids", type=int, default=None)
    parser.add_argument("--pubchem-retmax-per-term", type=int, default=250)
    parser.add_argument("--max-pk-chunks", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    settings = load_settings(args.config)
    if args.project_root is not None:
        settings.setdefault("project", {})["root"] = str(args.project_root.resolve())
    if args.fast:
        output_paths = settings.setdefault("paths", {})
        isolated_paths = ["models", "reports"]
        if _analysis_enabled(settings):
            isolated_paths.append("analysis")
        for name in isolated_paths:
            if name not in output_paths:
                continue
            base = Path(str(output_paths[name]))
            if base.name != "smoke":
                output_paths[name] = str(base / "smoke")
    paths = _paths(settings)
    project_root = _project_root(settings)
    os.environ.setdefault("MPLCONFIGDIR", str(project_root / ".matplotlib_cache"))
    os.environ.setdefault("XDG_CACHE_HOME", str(project_root / ".cache"))
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)

    stage = "curate" if args.stage == "data" else args.stage
    started_at = _utc_now()
    _write_run_metadata(settings, args, status="started", started_at=started_at)
    try:
        if stage in {"all", "collect"}:
            collect_public_data(args, settings)
        if stage in {"all", "curate"}:
            curate_data(settings)
        if stage in {"all", "quality"}:
            run_quality_stage(
                settings,
                fail_on_errors=(stage == "all" or args.fail_on_quality_errors),
            )
        if stage == "all":
            run_manifest_stage(settings, include_analysis_artifacts=False)
        elif stage == "manifest":
            run_manifest_stage(settings)
        if stage in {"all", "models"}:
            if stage == "models":
                # Standalone model runs still receive a fresh, content-linked
                # data manifest (under the isolated smoke path when --fast).
                run_manifest_stage(settings, include_analysis_artifacts=False)
            run_model_stage_transactionally(
                settings,
                split_override=args.split_strategy,
                endpoint_override=args.menin_endpoint,
                assay_family_override=args.menin_assay_family,
                fast=args.fast,
            )
        if stage == "analyze" or (stage == "all" and _analysis_enabled(settings)):
            output = run_analysis_stage(settings)
            print(f"Wrote analysis outputs to {output}")
        if stage in {"all", "report"}:
            output = run_report_stage(settings)
            print(f"Wrote {output}")
        if stage == "all":
            run_manifest_stage(settings)
            run_verify_stage(settings)
            # Refresh semantic readiness from verified artifacts, then freeze
            # and verify that final report bundle once more.
            output = run_report_stage(settings)
            print(f"Wrote {output}")
            run_manifest_stage(settings)
            run_verify_stage(settings)
        elif stage == "verify":
            run_verify_stage(settings)
    except Exception as exc:
        _write_run_metadata(
            settings,
            args,
            status="failed",
            started_at=started_at,
            error=exc,
        )
        raise
    _write_run_metadata(settings, args, status="complete", started_at=started_at)


if __name__ == "__main__":
    main()
