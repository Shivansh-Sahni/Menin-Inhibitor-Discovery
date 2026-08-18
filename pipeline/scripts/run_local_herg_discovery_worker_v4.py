#!/usr/bin/env python3
"""Governed train-only hERG evidence worker for the fourth local campaign.

The worker deliberately has no code path for repository validation or test
labels.  Model selection occurs on scaffold-disjoint folds inside the training
partition.  Fixed-dose broad hERG activity remains an auxiliary binary endpoint
and is never converted to, or pooled with, quantitative pIC50.

V4 imports immutable, tested V3 data/model helpers, but writes a separate V4
artifact contract.  Every unit spec is strict: unknown keys are rejected and
the exact executed/material spec is self-hashed in ``unit.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.optimize import nnls
from scipy.stats import spearmanr, wilcoxon
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
v3 = importlib.import_module("run_local_herg_discovery_worker_v3")

SCHEMA = "platform-local-herg-discovery-worker-v4/1.0"
CAPABILITY_SCHEMA = "platform-local-herg-discovery-worker-v4-capabilities/1.0"
EXACT_ROWS = 18_801
SEED = 20260812
HASH_CONTRACT = "compact_sorted_json_plus_newline"

GOVERNANCE_KEYS = {
    "source_partition",
    "repository_validation_labels_opened",
    "repository_test_labels_opened",
    "broad_fixed_dose_pooled_into_pic50",
}
COMMON_KEYS = {"operation", "seed", *GOVERNANCE_KEYS}
CANDIDATE_KEYS = {"candidate_id", "engine", "feature_set", "params"}

EXACT_FEATURE_SETS = {
    "rdkit2d",
    "morgan",
    "morgan_rdkit2d",
    "morgan_rdkit2d_maccs",
    "candidate_primary",
    "physics_selected",
    "all_scalable",
    "fundamental_core",
    "fundamental_interactions",
    "fingerprint_plus_fundamental",
    "all_scalable_v3",
}
BROAD_FEATURE_SETS = {"broad_compact_19", "broad_all_rdkit", "broad_sparse_morgan_rdkit"}
EXACT_ENGINES = {"xgboost", "lightgbm", "extratrees"}
BROAD_ENGINES = {"xgboost", "lightgbm", "logistic_saga"}


def _op(
    allowed: set[str], required: set[str], *, optional: bool = False, score_required: bool = False
) -> dict[str, Any]:
    return {
        "allowed_spec_keys": sorted(COMMON_KEYS | allowed),
        "required_spec_keys": sorted({"operation"} | required),
        "optional": optional,
        "score_required": score_required,
    }


OPERATIONS: dict[str, dict[str, Any]] = {
    "prepare": _op(set(), set()),
    "analyze": {
        **_op(set(), set()),
        "required_artifacts": [
            "validation.json",
            "analysis.md",
            "manifest.json",
            "decision_ledger.json",
            "feature_relationships.json",
            "uncertainty_ad.json",
            "assay_quality_report.json",
            "censored_report.json",
            "mmp_cliff_report.json",
            "broad_transfer_report.json",
            "model_cards.json",
            "final_models_manifest.json",
        ],
    },
    "baseline_reproduction": _op({"outer_folds", "anchor_source"}, {"anchor_source"}, score_required=True),
    "classical_hpo": _op(
        {
            "candidate",
            "evaluation_stage",
            "outer_fold",
            "inner_folds",
            "budget_fraction",
            "retain_model_artifact",
        },
        {"candidate", "evaluation_stage", "outer_fold", "inner_folds", "budget_fraction"},
        score_required=True,
    ),
    "chemprop_hpo": _op(
        {
            "evaluation_stage",
            "outer_fold",
            "inner_folds",
            "epochs",
            "batch_size",
            "depth",
            "hidden_dim",
            "dropout",
            "ffn_num_layers",
            "patience",
            "loss",
            "warmup_epochs",
            "init_lr",
            "max_lr",
            "final_lr",
        },
        {
            "evaluation_stage",
            "outer_fold",
            "inner_folds",
            "epochs",
            "batch_size",
            "depth",
            "hidden_dim",
            "dropout",
            "ffn_num_layers",
            "patience",
            "loss",
            "warmup_epochs",
            "init_lr",
            "max_lr",
            "final_lr",
        },
        score_required=True,
    ),
    "nested_outer_evaluation": _op(
        {
            "candidate",
            "resolved_candidate",
            "outer_fold",
            "selection_source_unit_ids",
            "selected_source_unit_id",
            "retain_model_artifact",
        },
        {"outer_fold", "selection_source_unit_ids"},
        score_required=True,
    ),
    "aggregate_nested_oof": _op({"source_unit_ids"}, {"source_unit_ids"}, score_required=True),
    "nested_stack": _op(
        {"member_unit_ids", "outer_fold", "meta_engine", "selection_source_unit_ids"},
        {"outer_fold", "meta_engine", "selection_source_unit_ids"},
        score_required=True,
    ),
    "feature_family_ablation": _op(
        {"candidate", "base_feature_set", "outer_folds", "source_unit_id"},
        {"candidate", "base_feature_set", "outer_folds"},
        score_required=True,
    ),
    "assay_hierarchical": _op(
        {"source_unit_id", "dimensions", "bootstrap_replicates"},
        {"source_unit_id", "dimensions", "bootstrap_replicates"},
    ),
    "censored_interval": _op(
        {"treatment", "alpha", "max_iter", "outer_folds"},
        {"treatment", "alpha", "max_iter", "outer_folds"},
        score_required=True,
    ),
    "broad_auxiliary": _op(
        {"candidate", "outer_folds", "fixed_fpr", "final_refit"},
        {"candidate", "outer_folds", "fixed_fpr"},
        score_required=True,
    ),
    "broad_transfer": _op(
        {
            "broad_source_unit_id",
            "exact_source_unit_id",
            "broad_selection_source_unit_ids",
            "exact_selection_source_unit_ids",
            "exact_candidate",
            "transfer_mode",
            "outer_folds",
        },
        {
            "broad_selection_source_unit_ids",
            "exact_selection_source_unit_ids",
            "transfer_mode",
            "outer_folds",
        },
        score_required=True,
    ),
    "mmp_cliff": _op(
        {"source_unit_id", "analysis_id", "minimum_pair_support", "cliff_threshold"},
        {"source_unit_id", "analysis_id", "minimum_pair_support", "cliff_threshold"},
    ),
    "uncertainty_ad": _op(
        {"member_unit_ids", "coverage_levels", "ad_method", "similarity_source"},
        {"member_unit_ids", "coverage_levels", "ad_method", "similarity_source"},
        score_required=True,
    ),
    "microstate_conformer": _op(
        {
            "conformer_feature_set",
            "requested_conformers",
            "retained_conformers",
            "max_iterations",
            "panel_selection",
            "panel_size",
            "levels",
            "source_unit_id",
            "outer_folds",
            "aggregation",
            "shard_size",
        },
        {
            "conformer_feature_set",
            "requested_conformers",
            "retained_conformers",
            "max_iterations",
            "panel_selection",
            "panel_size",
            "levels",
            "aggregation",
            "shard_size",
        },
    ),
    "receptor_pilot": _op(
        {"receptor_artifact_manifest", "compound_panel_manifest", "protocol_id"},
        {"receptor_artifact_manifest", "compound_panel_manifest", "protocol_id"},
        optional=True,
    ),
    "final_refit_exact": _op(
        {
            "candidate",
            "resolved_candidate",
            "source_unit_id",
            "selection_source_unit_ids",
            "retain_model_artifact",
        },
        {"selection_source_unit_ids", "retain_model_artifact"},
        score_required=True,
    ),
    "final_refit_broad": _op(
        {"candidate", "source_unit_id", "selection_source_unit_ids", "retain_model_artifact"},
        {"selection_source_unit_ids", "retain_model_artifact"},
        score_required=True,
    ),
}

# The first production launch is intentionally compute-only and capped at
# fifteen hours.  Secondary diagnostics/deployment code below is retained for
# later review, but cannot enter a governed plan until independently audited.
_COMPUTE_ONLY_OPERATIONS = {
    "prepare",
    "baseline_reproduction",
    "classical_hpo",
    "chemprop_hpo",
    "nested_outer_evaluation",
    "aggregate_nested_oof",
    "microstate_conformer",
}
OPERATIONS = {key: value for key, value in OPERATIONS.items() if key in _COMPUTE_ONLY_OPERATIONS}
OPERATIONS["chemprop_hpo"]["adaptive_selection_eligible"] = False
OPERATIONS["chemprop_hpo"]["selection_limitation"] = "inner-only standalone neural evidence"


class WorkerError(RuntimeError):
    """The scientific or artifact contract was violated."""


class Unavailable(WorkerError):
    """An explicitly optional capability is not locally available."""


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic(path: Path, writer: Callable[[Path], None]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        writer(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json(path: Path, value: dict[str, Any], hash_key: str) -> dict[str, Any]:
    document = dict(value)
    document.pop(hash_key, None)
    document[hash_key] = _digest(document)

    def writer(temporary: Path) -> None:
        temporary.write_bytes(_canonical(document))

    _atomic(path, writer)
    return document


def _read_json(path: Path, hash_key: str) -> dict[str, Any]:
    document = json.loads(path.read_text("utf-8"))
    expected = document.pop(hash_key, None)
    actual = _digest(document)
    document[hash_key] = expected
    if not isinstance(expected, str) or expected != actual:
        raise WorkerError(f"self-hash mismatch: {path}")
    return document


def _write_parquet(path: Path, frame: pd.DataFrame) -> None:
    def writer(temporary: Path) -> None:
        frame.to_parquet(temporary, index=False, compression="zstd")

    _atomic(path, writer)


def _binding(path: Path, role: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "role": role,
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _sha(path),
    }
    if path.suffix == ".parquet":
        metadata = pq.read_metadata(path)
        result.update(rows=metadata.num_rows, columns=metadata.num_columns)
    return result


def _verify_binding(binding: dict[str, Any], root: Path | None = None) -> None:
    path = Path(str(binding["path"])).resolve()
    if root is not None and root.resolve() not in (path, *path.parents):
        raise WorkerError(f"artifact escapes results root: {path}")
    if not path.is_file() or path.stat().st_size != int(binding["bytes"]):
        raise WorkerError(f"bound artifact missing or changed size: {path}")
    if _sha(path) != str(binding["sha256"]):
        raise WorkerError(f"bound artifact changed hash: {path}")
    if path.suffix == ".parquet" and pq.read_metadata(path).num_rows != int(binding["rows"]):
        raise WorkerError(f"bound artifact changed rows: {path}")


def _scope(endpoint: str) -> dict[str, Any]:
    return {
        "source_partition": "train",
        "repository_validation_labels_opened": False,
        "repository_test_labels_opened": False,
        "target_scope": (
            "confirmed wild-type hERG"
            if endpoint.startswith("confirmed-WT")
            else "wild-type-or-unspecified hERG"
        ),
        "endpoint": endpoint,
        "broad_fixed_dose_pooled_into_pic50": False,
        "causal_interpretation_allowed": False,
    }


def _safe_id(value: str) -> str:
    if not value or any(
        ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_." for ch in value
    ):
        raise WorkerError("unsafe or empty unit id")
    return value


def _capabilities() -> dict[str, Any]:
    return {
        "schema_version": CAPABILITY_SCHEMA,
        "unit_document_hash_contract": HASH_CONTRACT,
        "operations": OPERATIONS,
        "feature_sets": {
            "exact": sorted(EXACT_FEATURE_SETS),
            "broad": sorted(BROAD_FEATURE_SETS),
            "microstate_conformer": [
                "conformer_converged_subset",
                "conformer_ensemble_full",
                "conformer_scalar_core",
            ],
        },
        "engines": {"exact": sorted(EXACT_ENGINES), "broad": sorted(BROAD_ENGINES)},
    }


def _validate_governance(spec: dict[str, Any]) -> None:
    expected = {
        "source_partition": "train",
        "repository_validation_labels_opened": False,
        "repository_test_labels_opened": False,
        "broad_fixed_dose_pooled_into_pic50": False,
    }
    for key, value in expected.items():
        if key in spec and spec[key] != value:
            raise WorkerError(f"forbidden governance value {key}={spec[key]!r}")


def _validate_candidate(candidate: Any, *, broad: bool = False) -> dict[str, Any]:
    if not isinstance(candidate, dict):
        raise WorkerError("candidate must be an object")
    unknown = set(candidate) - CANDIDATE_KEYS
    if unknown:
        raise WorkerError(f"unknown candidate keys: {sorted(unknown)}")
    missing = {"candidate_id", "engine", "feature_set", "params"} - set(candidate)
    if missing:
        raise WorkerError(f"candidate missing keys: {sorted(missing)}")
    engines = BROAD_ENGINES if broad else EXACT_ENGINES
    features = BROAD_FEATURE_SETS if broad else EXACT_FEATURE_SETS
    if str(candidate["engine"]).lower() not in engines:
        raise WorkerError(f"unsupported {'broad' if broad else 'exact'} engine: {candidate['engine']}")
    if candidate["feature_set"] not in features:
        raise WorkerError(
            f"unsupported {'broad' if broad else 'exact'} feature_set: {candidate['feature_set']}"
        )
    if not isinstance(candidate["params"], dict):
        raise WorkerError("candidate params must be an object")
    return dict(candidate)


def _validate_spec(spec: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(spec, dict) or not isinstance(spec.get("operation"), str):
        raise WorkerError("unit spec must be an object with operation")
    operation = spec["operation"]
    if operation not in OPERATIONS:
        raise WorkerError(f"unsupported operation: {operation}")
    contract = OPERATIONS[operation]
    unknown = set(spec) - set(contract["allowed_spec_keys"])
    missing = set(contract["required_spec_keys"]) - set(spec)
    if unknown:
        raise WorkerError(f"{operation} unknown spec keys: {sorted(unknown)}")
    if missing:
        raise WorkerError(f"{operation} missing required keys: {sorted(missing)}")
    _validate_governance(spec)
    if "candidate" in spec:
        _validate_candidate(spec["candidate"], broad=operation in {"broad_auxiliary", "final_refit_broad"})
    if "resolved_candidate" in spec:
        _validate_candidate(spec["resolved_candidate"])
    if "exact_candidate" in spec:
        _validate_candidate(spec["exact_candidate"])
    if operation == "chemprop_hpo":
        if spec["evaluation_stage"] != "inner":
            raise WorkerError("chemprop_hpo is restricted to inner selection folds")
        if str(spec["loss"]) not in {"mae", "mse", "rmse"}:
            raise WorkerError("Chemprop loss must be mae, mse, or rmse")
        if not 1 <= int(spec["ffn_num_layers"]) <= 8:
            raise WorkerError("Chemprop ffn_num_layers must be in [1,8]")
        if not (0 < float(spec["init_lr"]) <= float(spec["max_lr"])):
            raise WorkerError("Chemprop requires 0 < init_lr <= max_lr")
        if not (0 < float(spec["final_lr"]) <= float(spec["max_lr"])):
            raise WorkerError("Chemprop requires 0 < final_lr <= max_lr")
    if operation == "aggregate_nested_oof":
        ids = [str(item) for item in spec["source_unit_ids"]]
        if len(ids) != 5 or len(set(ids)) != 5:
            raise WorkerError("aggregate_nested_oof requires five unique source units")
    if operation == "microstate_conformer":
        requested = int(spec["requested_conformers"])
        retained = int(spec["retained_conformers"])
        levels = [int(item) for item in spec["levels"]]
        if not 6 <= requested <= 64 or not 1 <= retained <= requested:
            raise WorkerError("invalid requested/retained conformer counts")
        if not levels or sorted(set(levels)) != levels or levels[-1] > requested or levels[0] < 6:
            raise WorkerError("conformer levels must be unique ascending values in [6, requested]")
        if retained < levels[-1]:
            raise WorkerError("retained_conformers must cover the largest convergence level")
        if spec["panel_selection"] not in {
            "all_exact_train",
            "nested_residual_stratified",
            "cliff_and_domain_stratified",
        }:
            raise WorkerError("unsupported conformer panel selection")
        if spec["aggregation"] not in {"boltzmann", "mean_sd_range", "minimum_energy"}:
            raise WorkerError("unsupported conformer aggregation")
        panel_size = int(spec["panel_size"])
        if spec["panel_selection"] == "all_exact_train":
            if panel_size != EXACT_ROWS or requested != 24 or retained < 8:
                raise WorkerError(
                    "whole-exact conformer run requires panel_size=18801, requested_conformers=24, retained>=8"
                )
        elif requested != 50 or retained < 50 or not {6, 20, 50}.issubset(levels):
            raise WorkerError(
                "convergence panel requires requested=50, retained>=50, and levels including 6,20,50"
            )
    return spec


def _write_unit(
    directory: Path,
    *,
    unit_id: str,
    operation: str,
    spec: dict[str, Any],
    metrics: dict[str, Any],
    artifacts: list[dict[str, Any]],
    limitations: list[str] | None = None,
    endpoint: str = "exact quantitative pIC50",
) -> dict[str, Any]:
    executed_spec = json.loads(json.dumps(spec))
    material_spec = {key: value for key, value in executed_spec.items() if key not in GOVERNANCE_KEYS}
    candidate = executed_spec.get("resolved_candidate") or executed_spec.get("candidate") or {}
    document = {
        "schema_version": SCHEMA,
        "created_at": _now(),
        "status": "passed",
        "operation": operation,
        "unit_id": unit_id,
        "unit_spec": spec,
        "unit_spec_sha256": _digest(spec),
        "executed_spec": executed_spec,
        "executed_spec_sha256": _digest(executed_spec),
        "material_spec_sha256": _digest(material_spec),
        "candidate_id": candidate.get("candidate_id", unit_id),
        "metrics": metrics,
        "artifacts": artifacts,
        "model_artifacts": [item for item in artifacts if str(item["role"]).startswith("model")],
        "scientific_scope": _scope(endpoint),
        "limitations": limitations or [],
    }
    return _write_json(directory / "unit.json", document, "unit_json_sha256")


def _existing_unit(path: Path, spec: dict[str, Any], results_root: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        unit = _read_json(path, "unit_json_sha256")
        if unit.get("status") != "passed" or unit.get("unit_spec") != spec:
            return None
        for artifact in unit.get("artifacts", []):
            _verify_binding(artifact, results_root)
        return unit
    except Exception:
        return None


def _context(prepared: Path) -> tuple[Path, Path, dict[str, Any]]:
    validation = _read_json(prepared / "validation.json", "validation_sha256")
    if validation.get("status") != "passed":
        raise WorkerError("v4 prepared validation did not pass")
    return Path(validation["base_v2_root"]), Path(validation["base_v3_root"]), validation


def _prepare(args: argparse.Namespace) -> dict[str, Any]:
    repo = Path(args.repo_root).resolve()
    base_v2 = Path(args.base_v2_root).resolve()
    base_v3 = Path(args.base_v3_root).resolve()
    output = Path(args.output_root).resolve()
    if (output / "validation.json").is_file():
        return _read_json(output / "validation.json", "validation_sha256")
    required = [
        base_v2 / "prepared/exact_train_cache.parquet",
        base_v2 / "prepared/nested_scaffold_splits.parquet",
        base_v2 / "prepared/feature_registry.json",
        base_v2 / "analysis/outer_model_summary.parquet",
        base_v3 / "prepared/validation.json",
        base_v3 / "prepared/expanded_interactions.parquet",
        base_v3 / "prepared/broad_wt_train_routing.parquet",
        base_v3 / "prepared/censored_train_routing.parquet",
        base_v3 / "analysis/nested_model_selection_oof.parquet",
    ]
    for path in required:
        if not path.is_file():
            raise WorkerError(f"required immutable input missing: {path}")
    if pq.read_metadata(required[0]).num_rows != EXACT_ROWS:
        raise WorkerError("exact quantitative cache must contain 18,801 train structures")
    if pq.read_metadata(required[1]).num_rows != EXACT_ROWS * 5:
        raise WorkerError("nested scaffold registry must contain five complete outer assignments")
    if output.exists() and any(output.iterdir()):
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    bindings = [_binding(path, f"immutable_input_{index:02d}") for index, path in enumerate(required)]
    manifest = _write_json(
        output / "manifest.json",
        {
            "schema_version": SCHEMA,
            "status": "passed",
            "created_at": _now(),
            "repo_root": str(repo),
            "base_v2_root": str(base_v2),
            "base_v3_root": str(base_v3),
            "bindings": bindings,
            "capabilities_sha256": _digest(_capabilities()),
        },
        "manifest_sha256",
    )
    return _write_json(
        output / "validation.json",
        {
            "schema_version": SCHEMA,
            "status": "passed",
            "created_at": _now(),
            "repo_root": str(repo),
            "base_v2_root": str(base_v2),
            "base_v3_root": str(base_v3),
            "exact_rows": EXACT_ROWS,
            "broad_train_rows": pq.read_metadata(
                base_v3 / "prepared/broad_wt_train_routing.parquet"
            ).num_rows,
            "repository_validation_labels_opened": False,
            "repository_test_labels_opened": False,
            "broad_fixed_dose_pooled_into_pic50": False,
            "manifest": _binding(output / "manifest.json", "prepared_manifest"),
            "manifest_sha256_bound": manifest["manifest_sha256"],
        },
        "validation_sha256",
    )


def _load_exact(base_v2: Path, base_v3: Path, feature_set: str) -> tuple[pd.DataFrame, list[str]]:
    return v3._load_exact(base_v2, base_v3 / "prepared", feature_set)


def _candidate(spec: dict[str, Any]) -> dict[str, Any]:
    return dict(spec.get("resolved_candidate") or spec.get("candidate") or {})


def _exact_model(candidate: dict[str, Any], workers: int, seed: int) -> Any:
    normalized = dict(candidate)
    if str(normalized["engine"]).lower() == "extratrees":
        normalized["engine"] = "extra_trees"
    return v3._model(normalized, workers, seed)


def _metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    return v3._regression_metrics(np.asarray(observed, dtype=float), np.asarray(predicted, dtype=float))


def _split_registry(base_v2: Path) -> pd.DataFrame:
    return pq.read_table(base_v2 / "prepared/nested_scaffold_splits.parquet").to_pandas()


def _regression_eval(
    base_v2: Path,
    base_v3: Path,
    directory: Path,
    candidate: dict[str, Any],
    *,
    stage: str,
    outer_fold: int | None,
    inner_folds: Sequence[int] | None,
    outer_folds: Sequence[int] | None,
    workers: int,
    seed: int,
    retain_models: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], pd.DataFrame]:
    frame, features = _load_exact(base_v2, base_v3, str(candidate["feature_set"]))
    splits = _split_registry(base_v2)
    fold_specs: list[tuple[int, set[str], set[str]]] = []
    if stage == "inner":
        if outer_fold is None:
            raise WorkerError("inner evaluation requires outer_fold")
        assignments = splits.loc[(splits.outer_fold == outer_fold) & (splits.outer_role == "fit")]
        for inner in list(inner_folds or []):
            fold_specs.append(
                (
                    int(inner),
                    set(assignments.loc[assignments.inner_fold != inner, "structure_id"].astype(str)),
                    set(assignments.loc[assignments.inner_fold == inner, "structure_id"].astype(str)),
                )
            )
    elif stage == "outer":
        for outer in list(outer_folds or []):
            assignments = splits.loc[splits.outer_fold == int(outer)]
            fold_specs.append(
                (
                    int(outer),
                    set(assignments.loc[assignments.outer_role == "fit", "structure_id"].astype(str)),
                    set(assignments.loc[assignments.outer_role == "heldout", "structure_id"].astype(str)),
                )
            )
    else:
        raise WorkerError(f"unsupported regression stage: {stage}")
    if not fold_specs:
        raise WorkerError("no evaluation folds requested")
    records: list[pd.DataFrame] = []
    importances: list[pd.DataFrame] = []
    model_paths: list[Path] = []
    started = time.monotonic()
    for fold, fit_ids, eval_ids in fold_specs:
        fit = frame.loc[frame.structure_id.astype(str).isin(fit_ids)].copy()
        evaluation = frame.loc[frame.structure_id.astype(str).isin(eval_ids)].copy()
        if set(fit.scaffold_group_id.astype(str)) & set(evaluation.scaffold_group_id.astype(str)):
            raise WorkerError("scaffold leakage in regression evaluation")
        model = v3._fit_regression_model(
            _exact_model(candidate, workers, seed + fold),
            fit[features],
            fit.target_pic50.to_numpy(dtype=float),
        )
        predicted = v3._predict_model(model, evaluation[features])
        records.append(
            pd.DataFrame(
                {
                    "structure_id": evaluation.structure_id.astype(str),
                    "scaffold_group_id": evaluation.scaffold_group_id.astype(str),
                    "candidate_id": candidate["candidate_id"],
                    "stage": stage,
                    "fold": fold,
                    "observed_pic50": evaluation.target_pic50.to_numpy(dtype=float),
                    "predicted_pic50": predicted,
                }
            )
        )
        importances.append(v3._importance(model, features, fold, str(candidate["candidate_id"])))
        if retain_models:
            path = directory / "models" / f"fold_{fold}.joblib"
            path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump({"model": model, "features": features}, path)
            model_paths.append(path)
    predictions = pd.concat(records, ignore_index=True)
    if predictions.structure_id.duplicated().any():
        raise WorkerError("evaluation produced duplicate OOF identities")
    predictions["residual_observed_minus_predicted"] = (
        predictions.observed_pic50 - predictions.predicted_pic50
    )
    output_metrics = _metrics(predictions.observed_pic50, predictions.predicted_pic50)
    output_metrics.update(
        selection_score=output_metrics["mae"],
        evaluation_stage=stage,
        folds=len(fold_specs),
        feature_count=len(features),
        fit_elapsed_seconds=time.monotonic() - started,
    )
    prediction_path = directory / "oof_predictions.parquet"
    importance_path = directory / "feature_importance.parquet"
    schema_path = directory / "feature_schema.json"
    _write_parquet(prediction_path, predictions)
    _write_parquet(importance_path, pd.concat(importances, ignore_index=True))
    _write_json(
        schema_path,
        {
            "schema_version": SCHEMA,
            "feature_set": candidate["feature_set"],
            "features": features,
            "label_blind_features": True,
        },
        "feature_schema_sha256",
    )
    artifacts = [
        _binding(prediction_path, "oof_predictions"),
        _binding(importance_path, "feature_importance"),
        _binding(schema_path, "feature_schema"),
        *[_binding(path, "model_fold") for path in model_paths],
    ]
    return output_metrics, artifacts, predictions


def _baseline_unit(
    base_v2: Path,
    base_v3: Path,
    directory: Path,
    unit_id: str,
    spec: dict[str, Any],
) -> dict[str, Any]:
    source = str(spec["anchor_source"])
    if source == "v2_nested":
        frame = pq.read_table(base_v2 / "analysis/outer_oof_predictions.parquet").to_pandas()
        frame = frame.loc[frame.model_id.eq("nested_selected")].copy()
        frame = frame.rename(columns={"outer_fold": "fold"})
    elif source == "v3_nested":
        frame = pq.read_table(base_v3 / "analysis/nested_model_selection_oof.parquet").to_pandas()
    else:
        raise WorkerError("anchor_source must be v2_nested or v3_nested")
    if len(frame) != EXACT_ROWS or frame.structure_id.duplicated().any():
        raise WorkerError("baseline anchor lacks complete unique 18,801-row OOF coverage")
    if set(pd.to_numeric(frame.fold, errors="raise").astype(int)) != set(range(5)):
        raise WorkerError("baseline anchor lacks all five outer folds")
    keep = [
        "structure_id",
        "scaffold_group_id",
        "fold",
        "observed_pic50",
        "predicted_pic50",
    ]
    output = frame[keep].copy()
    output["anchor_source"] = source
    path = directory / "oof_predictions.parquet"
    _write_parquet(path, output)
    values = _metrics(output.observed_pic50, output.predicted_pic50)
    values.update(selection_score=values["mae"], evaluation_stage="outer_anchor_reproduction", folds=5)
    return _write_unit(
        directory,
        unit_id=unit_id,
        operation="baseline_reproduction",
        spec=spec,
        metrics=values,
        artifacts=[_binding(path, "oof_predictions")],
        limitations=[
            "This reproduces a hash-bound completed train-only anchor; it is not external validation."
        ],
    )


def _classical_unit(
    base_v2: Path,
    base_v3: Path,
    directory: Path,
    unit_id: str,
    spec: dict[str, Any],
    workers: int,
) -> dict[str, Any]:
    operation = str(spec["operation"])
    candidate = _candidate(spec)
    if operation == "feature_family_ablation":
        if candidate["feature_set"] != spec["base_feature_set"]:
            raise WorkerError("ablation candidate feature_set must equal base_feature_set")
        stage, outer_fold, inner_folds = "outer", None, None
        outer_folds = [int(item) for item in spec["outer_folds"]]
    elif operation == "nested_outer_evaluation":
        stage, outer_fold, inner_folds = "outer", None, None
        outer_folds = [int(spec["outer_fold"])]
        _verify_selection_sources(directory.parents[1], spec["selection_source_unit_ids"], candidate)
    else:
        if spec["evaluation_stage"] != "inner":
            raise WorkerError("classical_hpo evaluation_stage must be inner")
        fraction = float(spec["budget_fraction"])
        if not 0 < fraction <= 1:
            raise WorkerError("budget_fraction must lie in (0,1]")
        stage = "inner"
        outer_fold = int(spec["outer_fold"])
        requested = [int(item) for item in spec["inner_folds"]]
        keep = max(1, int(math.ceil(len(requested) * fraction)))
        inner_folds, outer_folds = requested[:keep], None
    metrics, artifacts, _ = _regression_eval(
        base_v2,
        base_v3,
        directory,
        candidate,
        stage=stage,
        outer_fold=outer_fold,
        inner_folds=inner_folds,
        outer_folds=outer_folds,
        workers=workers,
        seed=int(spec.get("seed", SEED)),
        retain_models=bool(spec.get("retain_model_artifact", False)),
    )
    return _write_unit(
        directory,
        unit_id=unit_id,
        operation=operation,
        spec=spec,
        metrics=metrics,
        artifacts=artifacts,
        limitations=[
            "All model selection and evaluation are inside repository train.",
            "Feature relationships are noncausal, multiple-testing-controlled hypotheses.",
        ],
    )


def _verify_selection_sources(
    results_root: Path, source_ids: Sequence[str], candidate: dict[str, Any]
) -> None:
    if not source_ids:
        raise WorkerError("selection source IDs cannot be empty")
    sources: list[dict[str, Any]] = []
    for source_id in source_ids:
        unit = _source_unit(results_root, str(source_id))
        if unit.get("status") != "passed" or unit.get("operation") not in {"classical_hpo", "chemprop_hpo"}:
            raise WorkerError(f"invalid inner selection source: {source_id}")
        if unit.get("executed_spec", {}).get("evaluation_stage") != "inner":
            raise WorkerError("selection source is not inner-only")
        sources.append(unit)
    outers = {int(unit["executed_spec"]["outer_fold"]) for unit in sources}
    if len(outers) != 1:
        raise WorkerError("selection sources must belong to one outer fold")
    # Select by complete inner-fold evidence only.  If candidates have several
    # promoted units, aggregate their finite scores before ranking.
    scored: dict[str, list[float]] = {}
    for unit in sources:
        cid = str(unit.get("candidate_id"))
        score = unit.get("metrics", {}).get("selection_score")
        if isinstance(score, (int, float)) and math.isfinite(float(score)):
            scored.setdefault(cid, []).append(float(score))
    if not scored:
        raise WorkerError("selection pool has no finite inner score")
    winner = min(scored, key=lambda key: (float(np.mean(scored[key])), key))
    if str(candidate.get("candidate_id")) != winner:
        raise WorkerError(f"resolved candidate {candidate.get('candidate_id')} is not inner winner {winner}")


def _bh_qvalues(pvalues: np.ndarray) -> np.ndarray:
    pvalues = np.asarray(pvalues, dtype=float)
    order = np.argsort(pvalues)
    ranked = pvalues[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    output = np.empty_like(adjusted)
    output[order] = np.clip(adjusted, 0, 1)
    return output


def _relationship_artifacts(
    base_v2: Path,
    base_v3: Path,
    directory: Path,
    candidate: dict[str, Any],
    workers: int,
    spec: dict[str, Any],
) -> list[dict[str, Any]]:
    frame, features = _load_exact(base_v2, base_v3, candidate["feature_set"])
    features = [item for item in features if not item.startswith(("morgan__", "maccs__"))]
    splits = _split_registry(base_v2)
    rows: list[dict[str, Any]] = []
    ale_rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(int(spec.get("seed", SEED)))
    for fold in [int(item) for item in spec["outer_folds"]]:
        assignments = splits.loc[splits.outer_fold == fold]
        fit_ids = set(assignments.loc[assignments.outer_role.eq("fit"), "structure_id"].astype(str))
        eval_ids = set(assignments.loc[assignments.outer_role.eq("heldout"), "structure_id"].astype(str))
        fit = frame.loc[frame.structure_id.astype(str).isin(fit_ids)].copy()
        evaluation = frame.loc[frame.structure_id.astype(str).isin(eval_ids)].copy()
        model = v3._fit_regression_model(
            _exact_model(candidate, workers, int(spec.get("seed", SEED)) + fold),
            fit[features],
            fit.target_pic50.to_numpy(dtype=float),
        )
        baseline = v3._predict_model(model, evaluation[features])
        baseline_ae = np.abs(evaluation.target_pic50.to_numpy(dtype=float) - baseline)
        for feature in features:
            values = pd.to_numeric(evaluation[feature], errors="coerce")
            finite = values.notna() & np.isfinite(values)
            if finite.sum() < 50 or values[finite].nunique() < 4:
                continue
            # Conditional permutation within coarse MolWt/logP strata reduces
            # the most obvious physicochemical extrapolation artifact.
            condition = pd.DataFrame(index=evaluation.index)
            for name in ("rdkit2d__MolWt", "rdkit2d__MolLogP"):
                if name in evaluation:
                    condition[name] = pd.qcut(
                        pd.to_numeric(evaluation[name], errors="coerce").rank(method="first"),
                        5,
                        labels=False,
                        duplicates="drop",
                    )
            permuted = evaluation[features].copy()
            if condition.empty:
                permuted.loc[finite, feature] = rng.permutation(values[finite].to_numpy())
            else:
                labels = condition.fillna(-1).astype(str).agg("|".join, axis=1)
                for _, index in labels[finite].groupby(labels[finite]).groups.items():
                    permuted.loc[index, feature] = rng.permutation(values.loc[index].to_numpy())
            changed = v3._predict_model(model, permuted)
            delta = np.abs(evaluation.target_pic50.to_numpy(dtype=float) - changed) - baseline_ae
            scaffold_delta = (
                pd.DataFrame({"scaffold": evaluation.scaffold_group_id.astype(str), "delta": delta})
                .groupby("scaffold", as_index=False)
                .delta.mean()
            )
            pvalue = (
                float(wilcoxon(scaffold_delta, alternative="two-sided").pvalue)
                if len(scaffold_delta) >= 10 and np.any(scaffold_delta != 0)
                else 1.0
            )
            rho = spearmanr(values[finite], evaluation.loc[finite, "target_pic50"]).statistic
            rows.append(
                {
                    "fold": fold,
                    "feature": feature,
                    "process_hypothesis": v3._process(feature),
                    "conditional_permutation_mae_delta": float(np.mean(delta)),
                    "signed_spearman_target": float(np.nan_to_num(rho)),
                    "p_value": float(pvalue),
                    "structures": int(finite.sum()),
                    "causal_interpretation_allowed": False,
                }
            )
            bins = pd.qcut(values[finite].rank(method="first"), 8, labels=False, duplicates="drop")
            reference = float(np.mean(baseline[finite]))
            for bin_id, index in bins.groupby(bins).groups.items():
                ale_rows.append(
                    {
                        "fold": fold,
                        "feature": feature,
                        "quantile_bin": int(bin_id),
                        "mean_feature": float(values.loc[index].mean()),
                        "mean_prediction_centered": float(
                            np.mean(baseline[evaluation.index.get_indexer(index)]) - reference
                        ),
                        "structures": len(index),
                        "ale_like_not_causal": True,
                    }
                )
    relationships = pd.DataFrame(rows)
    if relationships.empty:
        raise WorkerError("no interpretable relationships could be evaluated")
    relationships["bh_q_value_within_campaign"] = _bh_qvalues(relationships.p_value.to_numpy())
    relationship_path = directory / "conditional_permutation_relationships.parquet"
    ale_path = directory / "ale_like_signed_effects.parquet"
    _write_parquet(relationship_path, relationships)
    _write_parquet(ale_path, pd.DataFrame(ale_rows))
    return [_binding(relationship_path, "conditional_permutation"), _binding(ale_path, "ale_like_effects")]


def _source_unit(results_root: Path, unit_id: str) -> dict[str, Any]:
    path = results_root / "units" / _safe_id(unit_id) / "unit.json"
    unit = _read_json(path, "unit_json_sha256")
    if unit.get("status") != "passed":
        raise WorkerError(f"source unit is not passed: {unit_id}")
    for artifact in unit.get("artifacts", []):
        _verify_binding(artifact, results_root)
    return unit


def _artifact(unit: dict[str, Any], role: str) -> Path:
    matches = [Path(item["path"]) for item in unit.get("artifacts", []) if item.get("role") == role]
    if len(matches) != 1:
        raise WorkerError(f"unit {unit.get('unit_id')} requires exactly one {role} artifact")
    return matches[0]


def _stack_unit(
    base_v2: Path,
    base_v3: Path,
    results_root: Path,
    directory: Path,
    unit_id: str,
    spec: dict[str, Any],
    workers: int,
) -> dict[str, Any]:
    outer = int(spec["outer_fold"])
    member_ids = [str(item) for item in spec["member_unit_ids"]]
    if len(member_ids) < 2 or len(set(member_ids)) != len(member_ids):
        raise WorkerError("nested_stack needs at least two unique member units")
    selection_ids = set(str(item) for item in spec["selection_source_unit_ids"])
    if not set(member_ids).issubset(selection_ids):
        raise WorkerError("stack members must be inner-only selection sources")
    inner: pd.DataFrame | None = None
    candidates: list[dict[str, Any]] = []
    for member_id in member_ids:
        unit = _source_unit(results_root, member_id)
        if unit["operation"] not in {"classical_hpo", "chemprop_hpo"}:
            raise WorkerError("stack member is not an inner HPO unit")
        executed = unit["executed_spec"]
        if int(executed["outer_fold"]) != outer or executed["evaluation_stage"] != "inner":
            raise WorkerError("stack member was not selected inside the requested outer fold")
        frame = pd.read_parquet(_artifact(unit, "oof_predictions"))
        selected = frame[["structure_id", "scaffold_group_id", "observed_pic50", "predicted_pic50"]].rename(
            columns={"predicted_pic50": f"prediction__{member_id}"}
        )
        inner = (
            selected
            if inner is None
            else inner.merge(
                selected.drop(columns=["scaffold_group_id", "observed_pic50"]),
                on="structure_id",
                validate="one_to_one",
            )
        )
        if unit["operation"] == "classical_hpo":
            candidates.append(dict(executed["candidate"]))
        else:
            raise Unavailable(
                "Chemprop outer refit for nested stacking requires completed V4 Chemprop adapter"
            )
    assert inner is not None
    prediction_columns = [item for item in inner if item.startswith("prediction__")]
    matrix = inner[prediction_columns].to_numpy(dtype=float)
    observed = inner.observed_pic50.to_numpy(dtype=float)
    if spec["meta_engine"] == "ridge":
        meta = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(matrix, observed)
        meta_predict: Callable[[np.ndarray], np.ndarray] = meta.predict
        meta_artifact: Any = meta
    elif spec["meta_engine"] == "nnls":
        weights, _ = nnls(np.column_stack([matrix, np.ones(len(matrix))]), observed)
        coef, intercept = weights[:-1], weights[-1]

        def nnls_predict(values: np.ndarray) -> np.ndarray:
            return values @ coef + intercept

        meta_predict = nnls_predict
        meta_artifact = {"weights": coef, "intercept": intercept}
    else:
        raise WorkerError("meta_engine must be ridge or nnls")
    # Refit each selected member on the complete outer-fit partition only.
    assignments = _split_registry(base_v2)
    assignment = assignments.loc[assignments.outer_fold == outer]
    fit_ids = set(assignment.loc[assignment.outer_role.eq("fit"), "structure_id"].astype(str))
    eval_ids = set(assignment.loc[assignment.outer_role.eq("heldout"), "structure_id"].astype(str))
    outer_predictions: list[np.ndarray] = []
    outer_frame: pd.DataFrame | None = None
    member_models: list[Any] = []
    for index, candidate in enumerate(candidates):
        frame, features = _load_exact(base_v2, base_v3, candidate["feature_set"])
        fit = frame.loc[frame.structure_id.astype(str).isin(fit_ids)]
        evaluation = frame.loc[frame.structure_id.astype(str).isin(eval_ids)]
        model = v3._fit_regression_model(
            _exact_model(candidate, workers, int(spec.get("seed", SEED)) + index),
            fit[features],
            fit.target_pic50.to_numpy(dtype=float),
        )
        outer_predictions.append(v3._predict_model(model, evaluation[features]))
        member_models.append({"candidate": candidate, "features": features, "model": model})
        if outer_frame is None:
            outer_frame = evaluation[["structure_id", "scaffold_group_id", "target_pic50"]].copy()
    assert outer_frame is not None
    predicted = meta_predict(np.column_stack(outer_predictions))
    output = outer_frame.rename(columns={"target_pic50": "observed_pic50"})
    output["predicted_pic50"] = predicted
    output["fold"] = outer
    prediction_path = directory / "oof_predictions.parquet"
    model_path = directory / "stack_bundle.joblib"
    _write_parquet(prediction_path, output)
    joblib.dump(
        {"members": member_models, "meta": meta_artifact, "meta_engine": spec["meta_engine"]}, model_path
    )
    values = _metrics(output.observed_pic50, output.predicted_pic50)
    values.update(selection_score=values["mae"], evaluation_stage="outer", folds=1, members=len(member_ids))
    return _write_unit(
        directory,
        unit_id=unit_id,
        operation="nested_stack",
        spec=spec,
        metrics=values,
        artifacts=[
            _binding(prediction_path, "oof_predictions"),
            _binding(model_path, "model_stack_outer_fold"),
        ],
        limitations=["The meta-model used only inner OOF predictions inside this outer fold."],
    )


def _aggregate_nested_unit(
    base_v2: Path, results_root: Path, directory: Path, unit_id: str, spec: dict[str, Any]
) -> dict[str, Any]:
    frames: list[pd.DataFrame] = []
    for source_id in [str(item) for item in spec["source_unit_ids"]]:
        unit = _source_unit(results_root, source_id)
        if unit["operation"] not in {"nested_outer_evaluation", "nested_stack"}:
            raise WorkerError("nested aggregate source must be a held-out outer evaluation")
        frame = pd.read_parquet(_artifact(unit, "oof_predictions"))
        if frame["fold"].nunique() != 1 or frame["structure_id"].duplicated().any():
            raise WorkerError("nested source is not exactly one unique outer fold")
        frame = frame.copy()
        frame["source_unit_id"] = source_id
        frames.append(frame)
    composite = pd.concat(frames, ignore_index=True)
    if (
        len(composite) != EXACT_ROWS
        or composite.structure_id.nunique() != EXACT_ROWS
        or composite.structure_id.duplicated().any()
        or set(pd.to_numeric(composite.fold, errors="raise").astype(int)) != set(range(5))
    ):
        raise WorkerError("nested aggregate must cover exactly 18,801 structures across folds 0..4")
    canonical = pd.read_parquet(
        base_v2 / "prepared/exact_train_cache.parquet",
        columns=["structure_id", "scaffold_group_id", "target_pic50"],
    ).rename(columns={"target_pic50": "canonical_observed"})
    check = composite.merge(canonical, on="structure_id", suffixes=("", "_canonical"), validate="one_to_one")
    if (
        len(check) != EXACT_ROWS
        or not check.scaffold_group_id.astype(str).eq(check.scaffold_group_id_canonical.astype(str)).all()
        or not np.isclose(check.observed_pic50, check.canonical_observed).all()
    ):
        raise WorkerError("nested aggregate does not match canonical identities/scaffolds/targets")
    path = directory / "nested_model_selection_oof.parquet"
    _write_parquet(path, composite)
    values = _metrics(composite.observed_pic50, composite.predicted_pic50)
    values.update(
        selection_score=values["mae"],
        evaluation_stage="unbiased_nested_outer_composite",
        folds=5,
        candidate_may_differ_by_outer_fold=True,
    )
    return _write_unit(
        directory,
        unit_id=unit_id,
        operation="aggregate_nested_oof",
        spec=spec,
        metrics=values,
        artifacts=[_binding(path, "oof_predictions")],
        limitations=[
            "This is an unbiased internal nested model-selection estimate, not external validation."
        ],
    )


def _chemprop_argv(
    executable: Path, csv_path: Path, output: Path, spec: dict[str, Any], seed: int
) -> list[str]:
    """Build an argv in which every advertised neural knob is material."""
    return [
        str(executable),
        "train",
        "--data-path",
        str(csv_path),
        "--output-dir",
        str(output),
        "--smiles-columns",
        "smiles",
        "--target-columns",
        "target_pic50",
        "--ignore-columns",
        "structure_id",
        "scaffold_group_id",
        "--splits-column",
        "split",
        "--task-type",
        "regression",
        "--loss-function",
        str(spec["loss"]),
        "--metrics",
        "mae",
        "rmse",
        "r2",
        "--epochs",
        str(int(spec["epochs"])),
        "--patience",
        str(int(spec["patience"])),
        "--batch-size",
        str(int(spec["batch_size"])),
        "--depth",
        str(int(spec["depth"])),
        "--message-hidden-dim",
        str(int(spec["hidden_dim"])),
        "--ffn-hidden-dim",
        str(int(spec["hidden_dim"])),
        "--ffn-num-layers",
        str(int(spec["ffn_num_layers"])),
        "--dropout",
        str(float(spec["dropout"])),
        "--warmup-epochs",
        str(float(spec["warmup_epochs"])),
        "--init-lr",
        str(float(spec["init_lr"])),
        "--max-lr",
        str(float(spec["max_lr"])),
        "--final-lr",
        str(float(spec["final_lr"])),
        "--data-seed",
        str(seed),
        "--pytorch-seed",
        str(seed),
        "--accelerator",
        "cpu",
        "--devices",
        "1",
        "--num-workers",
        "0",
        "--ensemble-size",
        "1",
    ]


def _chemprop_fold(
    repo: Path,
    directory: Path,
    data: pd.DataFrame,
    spec: dict[str, Any],
    *,
    fold: int,
    seed: int,
    workers: int,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    executable = repo / ".venv/bin/chemprop"
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise Unavailable("Chemprop 2 executable is unavailable in the bound environment")
    counts = data["split"].value_counts().to_dict()
    if any(int(counts.get(role, 0)) == 0 for role in ("train", "val", "test")):
        raise WorkerError("Chemprop split must have nonempty train, val, and test roles")
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        if set(data.loc[data.split.eq(left), "scaffold_group_id"]) & set(
            data.loc[data.split.eq(right), "scaffold_group_id"]
        ):
            raise WorkerError("Chemprop roles share scaffolds")
    fold_root = directory / "chemprop" / f"fold_{fold}"
    fold_root.mkdir(parents=True, exist_ok=True)
    csv_path = fold_root / "train_only_split.csv"
    data[["structure_id", "scaffold_group_id", "smiles", "target_pic50", "split"]].to_csv(
        csv_path, index=False
    )
    training = fold_root / "training"
    argv = _chemprop_argv(executable, csv_path, training, spec, seed)
    command_path = fold_root / "command.json"
    _write_json(
        command_path,
        {"argv": argv, "argv_sha256": _digest(argv), "role_counts": counts},
        "command_sha256",
    )
    log_path = fold_root / "chemprop.log"
    environment = {
        **os.environ,
        "PYTHONHASHSEED": str(seed),
        "OMP_NUM_THREADS": str(workers),
        "OPENBLAS_NUM_THREADS": str(workers),
        "VECLIB_MAXIMUM_THREADS": str(workers),
        "MPLCONFIGDIR": str(fold_root / "mpl-cache"),
        "XDG_CACHE_HOME": str(fold_root / "cache"),
    }
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(
            argv,
            cwd=repo,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=max(1, int(spec["epochs"])) * 900,
        )
    if process.returncode != 0:
        raise WorkerError(f"Chemprop fold {fold} failed; see {log_path}")
    prediction_files = sorted(training.rglob("test_predictions.csv"))
    if len(prediction_files) != 1:
        raise WorkerError(f"Chemprop fold {fold} emitted {len(prediction_files)} prediction files")
    raw = pd.read_csv(prediction_files[0])
    prediction_column = "target_pic50" if "target_pic50" in raw else None
    if prediction_column is None:
        candidates = [item for item in raw if item not in {"smiles", "structure_id"}]
        if len(candidates) != 1:
            raise WorkerError("cannot resolve Chemprop prediction column")
        prediction_column = candidates[0]
    heldout = data.loc[data.split.eq("test")].reset_index(drop=True)
    if len(raw) != len(heldout):
        raise WorkerError("Chemprop test prediction count mismatch")
    if "smiles" in raw and not raw.smiles.astype(str).equals(heldout.smiles.astype(str)):
        raise WorkerError("Chemprop prediction order differs from explicit heldout SMILES")
    predicted = pd.to_numeric(raw[prediction_column], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(predicted).all():
        raise WorkerError("Chemprop emitted nonfinite predictions")
    output = heldout[["structure_id", "scaffold_group_id", "target_pic50"]].rename(
        columns={"target_pic50": "observed_pic50"}
    )
    output["predicted_pic50"] = predicted
    output["fold"] = fold
    output["fit_elapsed_seconds"] = time.monotonic() - started
    artifacts = [_binding(command_path, "chemprop_command"), _binding(log_path, "chemprop_log")]
    for model in sorted(training.rglob("*.ckpt")) + sorted(training.rglob("*.pt")):
        artifacts.append(_binding(model, "model_chemprop_fold"))
    return output, artifacts


def _chemprop_data(
    base_v2: Path,
    *,
    outer_fold: int,
    test_inner_fold: int | None,
    seed: int,
) -> pd.DataFrame:
    exact = pd.read_parquet(
        base_v2 / "prepared/exact_train_cache.parquet",
        columns=["structure_id", "standardized_smiles", "scaffold_group_id", "target_pic50"],
    ).rename(columns={"standardized_smiles": "smiles"})
    assignments = _split_registry(base_v2)
    selected = assignments.loc[assignments.outer_fold.eq(outer_fold)].copy()
    if test_inner_fold is None:
        selected["split"] = np.where(selected.outer_role.eq("heldout"), "test", "train")
    else:
        selected = selected.loc[selected.outer_role.eq("fit")]
        selected["split"] = np.where(selected.inner_fold.eq(test_inner_fold), "test", "train")
    fit_scaffolds = sorted(selected.loc[selected.split.eq("train"), "scaffold_group_id"].astype(str).unique())
    rng = np.random.default_rng(seed)
    rng.shuffle(fit_scaffolds)
    calibration = set(fit_scaffolds[: max(1, int(math.ceil(len(fit_scaffolds) * 0.12)))])
    selected.loc[
        selected.split.eq("train") & selected.scaffold_group_id.astype(str).isin(calibration), "split"
    ] = "val"
    data = exact.merge(
        selected[["structure_id", "split"]], on="structure_id", how="inner", validate="one_to_one"
    )
    return data


def _chemprop_unit(
    repo: Path,
    base_v2: Path,
    directory: Path,
    unit_id: str,
    spec: dict[str, Any],
    workers: int,
) -> dict[str, Any]:
    outer = int(spec["outer_fold"])
    frames: list[pd.DataFrame] = []
    artifacts: list[dict[str, Any]] = []
    for inner in [int(item) for item in spec["inner_folds"]]:
        data = _chemprop_data(
            base_v2, outer_fold=outer, test_inner_fold=inner, seed=int(spec.get("seed", SEED)) + inner
        )
        frame, bound = _chemprop_fold(
            repo,
            directory,
            data,
            spec,
            fold=inner,
            seed=int(spec.get("seed", SEED)) + inner,
            workers=workers,
        )
        frames.append(frame)
        artifacts.extend(bound)
    oof = pd.concat(frames, ignore_index=True)
    if oof.structure_id.duplicated().any():
        raise WorkerError("Chemprop inner OOF identities overlap")
    path = directory / "oof_predictions.parquet"
    _write_parquet(path, oof)
    values = _metrics(oof.observed_pic50, oof.predicted_pic50)
    values.update(
        selection_score=values["mae"],
        evaluation_stage="inner",
        folds=len(frames),
        candidate_family="chemprop_mpnn",
        all_advertised_hyperparameters_in_argv=True,
    )
    return _write_unit(
        directory,
        unit_id=unit_id,
        operation="chemprop_hpo",
        spec=spec,
        metrics=values,
        artifacts=[_binding(path, "oof_predictions"), *artifacts],
        limitations=["Chemprop selection uses explicit scaffold-disjoint inner folds inside train only."],
    )


def _prediction_source(results_root: Path, source_id: str) -> tuple[dict[str, Any], pd.DataFrame]:
    unit = _source_unit(results_root, source_id)
    return unit, pd.read_parquet(_artifact(unit, "oof_predictions"))


def _scaffold_bootstrap_mae(frame: pd.DataFrame, *, replicates: int, seed: int) -> tuple[float, float]:
    grouped = list(frame.groupby("scaffold_group_id", sort=False))
    if len(grouped) < 3:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates, dtype=float)
    for repeat in range(replicates):
        selected = rng.integers(0, len(grouped), size=len(grouped))
        sampled = pd.concat([grouped[index][1] for index in selected], ignore_index=True)
        estimates[repeat] = mean_absolute_error(sampled.observed_pic50, sampled.predicted_pic50)
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def _assay_unit(
    base_v2: Path,
    results_root: Path,
    directory: Path,
    unit_id: str,
    spec: dict[str, Any],
) -> dict[str, Any]:
    _, prediction = _prediction_source(results_root, str(spec["source_unit_id"]))
    if len(prediction) != EXACT_ROWS or prediction.structure_id.duplicated().any():
        raise WorkerError("assay analysis requires one complete 18,801-row nested OOF source")
    available = {
        "measurement_modality",
        "automation_class",
        "assay_family",
        "source_family",
        "protocol_completeness_mean",
        "wild_type_evidence_scope",
        "master_confirmed_wild_type_scope",
    }
    dimensions = [str(item) for item in spec["dimensions"]]
    if not dimensions or not set(dimensions).issubset(available):
        raise WorkerError(f"unsupported assay dimensions: {sorted(set(dimensions) - available)}")
    metadata = pd.read_parquet(
        base_v2 / "prepared/exact_train_cache.parquet", columns=["structure_id", *sorted(available)]
    )
    frame = prediction.merge(metadata, on="structure_id", how="inner", validate="one_to_one")
    rows: list[dict[str, Any]] = []
    replicates = int(spec["bootstrap_replicates"])
    if not 100 <= replicates <= 20_000:
        raise WorkerError("assay bootstrap_replicates must be in [100,20000]")
    for dimension in dimensions:
        values = frame[dimension]
        if dimension == "protocol_completeness_mean":
            values = pd.cut(pd.to_numeric(values, errors="coerce"), [-np.inf, 0, 2, 4, np.inf]).astype(str)
        for index, (level, group) in enumerate(
            frame.assign(_stratum=values).groupby("_stratum", dropna=False)
        ):
            if len(group) < 50 or group.scaffold_group_id.nunique() < 3:
                continue
            metrics = _metrics(group.observed_pic50, group.predicted_pic50)
            lower, upper = _scaffold_bootstrap_mae(
                group, replicates=replicates, seed=int(spec.get("seed", SEED)) + index
            )
            rows.append(
                {
                    "dimension": dimension,
                    "level": str(level),
                    **metrics,
                    "mae_scaffold_bootstrap_ci95_lower": lower,
                    "mae_scaffold_bootstrap_ci95_upper": upper,
                    "observational_confounding_acknowledged": True,
                }
            )
    output = pd.DataFrame(rows)
    if output.empty:
        raise WorkerError("assay analysis produced no supported strata")
    path = directory / "assay_hierarchical_strata.parquet"
    _write_parquet(path, output)
    return _write_unit(
        directory,
        unit_id=unit_id,
        operation="assay_hierarchical",
        spec=spec,
        metrics={"strata": len(output), "structures": len(frame), "dimensions": len(dimensions)},
        artifacts=[_binding(path, "assay_quality_strata")],
        limitations=[
            "Assay/source strata are observational and remain confounded by chemistry, protocol, and source."
        ],
    )


def _censored_unit(
    repo: Path,
    base_v3: Path,
    directory: Path,
    unit_id: str,
    spec: dict[str, Any],
    workers: int,
) -> dict[str, Any]:
    mapped = {
        "operation": "censored_sensitivity",
        "treatment": spec["treatment"],
        "alpha": spec["alpha"],
        "max_iter": spec["max_iter"],
        "outer_folds": spec["outer_folds"],
        "source_partition": "train",
    }
    legacy = v3._censored_unit(repo, base_v3 / "prepared", directory, unit_id, mapped, workers)
    metrics = dict(legacy["metrics"])
    artifacts = list(legacy["artifacts"])
    return _write_unit(
        directory,
        unit_id=unit_id,
        operation="censored_interval",
        spec=spec,
        metrics=metrics,
        artifacts=artifacts,
        endpoint="censored quantitative sensitivity",
        limitations=[
            "Censored interval likelihood is a separate sensitivity analysis and is not pooled into the exact primary label."
        ],
    )


def _mmp_unit(
    repo: Path,
    results_root: Path,
    directory: Path,
    unit_id: str,
    spec: dict[str, Any],
) -> dict[str, Any]:
    _, predictions = _prediction_source(results_root, str(spec["source_unit_id"]))
    if len(predictions) != EXACT_ROWS or predictions.structure_id.duplicated().any():
        raise WorkerError("MMP analysis requires complete nested OOF predictions")
    effects = pd.read_parquet(repo / v3.DEFAULT_MMP)
    prediction = predictions.set_index("structure_id").predicted_pic50
    effects = effects.loc[
        effects.structure_id_a.isin(prediction.index) & effects.structure_id_b.isin(prediction.index)
    ].copy()
    if len(effects) < int(spec["minimum_pair_support"]):
        raise Unavailable("insufficient train-only matched pairs")
    effects["predicted_delta_pic50_b_minus_a"] = effects.structure_id_b.map(
        prediction
    ) - effects.structure_id_a.map(prediction)
    effects["observed_cliff_at_requested_threshold"] = effects.absolute_delta_pic50.ge(
        float(spec["cliff_threshold"])
    )
    observed = effects.delta_pic50_b_minus_a.to_numpy(dtype=float)
    predicted = effects.predicted_delta_pic50_b_minus_a.to_numpy(dtype=float)
    effects["delta_residual"] = observed - predicted
    cliff = effects.observed_cliff_at_requested_threshold.to_numpy(dtype=bool)
    detected = np.abs(predicted) >= float(spec["cliff_threshold"])
    metrics = {
        "pairs": len(effects),
        "delta_mae": float(np.mean(np.abs(observed - predicted))),
        "delta_spearman": float(np.nan_to_num(spearmanr(observed, predicted).statistic)),
        "direction_accuracy": float(np.mean(np.sign(observed) == np.sign(predicted))),
        "observed_activity_cliffs": int(cliff.sum()),
        "cliff_mae": float(np.mean(np.abs(observed[cliff] - predicted[cliff]))) if cliff.any() else None,
        "cliff_recall": float(detected[cliff].mean()) if cliff.any() else None,
        "analysis_id": spec["analysis_id"],
    }
    path = directory / "mmp_cliff_residuals.parquet"
    _write_parquet(path, effects)
    return _write_unit(
        directory,
        unit_id=unit_id,
        operation="mmp_cliff",
        spec=spec,
        metrics=metrics,
        artifacts=[_binding(path, "mmp_cliff_residuals")],
        limitations=[
            "Matched-pair residuals are train-only predictive diagnostics, not causal transformations."
        ],
    )


def _uncertainty_unit(
    base_v2: Path,
    results_root: Path,
    directory: Path,
    unit_id: str,
    spec: dict[str, Any],
) -> dict[str, Any]:
    member_ids = [str(item) for item in spec["member_unit_ids"]]
    if not member_ids:
        raise WorkerError("uncertainty requires at least one full nested/OOF member")
    combined: pd.DataFrame | None = None
    for member_id in member_ids:
        _, frame = _prediction_source(results_root, member_id)
        required = ["structure_id", "scaffold_group_id", "observed_pic50", "predicted_pic50", "fold"]
        if (
            not set(required).issubset(frame)
            or len(frame) != EXACT_ROWS
            or frame.structure_id.duplicated().any()
        ):
            raise WorkerError(f"uncertainty member {member_id} is not complete five-fold OOF")
        selected = frame[required].rename(columns={"predicted_pic50": f"prediction__{member_id}"})
        if combined is None:
            combined = selected
        else:
            aligned = combined.merge(
                selected,
                on="structure_id",
                suffixes=("_left", "_right"),
                validate="one_to_one",
            )
            for key in ("scaffold_group_id", "fold"):
                if not aligned[f"{key}_left"].astype(str).eq(aligned[f"{key}_right"].astype(str)).all():
                    raise WorkerError(f"uncertainty members disagree on {key}")
            if not np.isclose(aligned.observed_pic50_left, aligned.observed_pic50_right).all():
                raise WorkerError("uncertainty members disagree on observed target")
            combined = combined.merge(
                selected.drop(columns=["scaffold_group_id", "observed_pic50", "fold"]),
                on="structure_id",
                validate="one_to_one",
            )
    assert combined is not None
    prediction_columns = [item for item in combined if item.startswith("prediction__")]
    combined["ensemble_prediction"] = combined[prediction_columns].mean(axis=1)
    combined["ensemble_sd"] = combined[prediction_columns].std(axis=1, ddof=0)
    absolute_error = np.abs(combined.observed_pic50 - combined.ensemble_prediction).to_numpy()
    levels = [float(item) for item in spec["coverage_levels"]]
    if not levels or any(not 0 < item < 1 for item in levels):
        raise WorkerError("coverage levels must lie strictly inside (0,1)")
    calibration_rows: list[dict[str, Any]] = []
    for level in levels:
        lower = np.full(len(combined), np.nan)
        upper = np.full(len(combined), np.nan)
        for fold in range(5):
            evaluation = combined.fold.to_numpy(dtype=int) == fold
            quantile = float(np.quantile(absolute_error[~evaluation], level))
            lower[evaluation] = combined.loc[evaluation, "ensemble_prediction"] - quantile
            upper[evaluation] = combined.loc[evaluation, "ensemble_prediction"] + quantile
        covered = (combined.observed_pic50 >= lower) & (combined.observed_pic50 <= upper)
        combined[f"lower_{int(level * 100)}"] = lower
        combined[f"upper_{int(level * 100)}"] = upper
        calibration_rows.append(
            {
                "nominal_coverage": level,
                "empirical_cross_conformal_coverage": float(covered.mean()),
                "mean_width": float(np.mean(upper - lower)),
            }
        )
    similarity_path = base_v2 / "analysis/outer_oof_predictions.parquet"
    similarity = pd.read_parquet(similarity_path)
    if "model_id" in similarity:
        similarity = similarity.loc[similarity.model_id.eq(str(spec["similarity_source"]))]
    if "maximum_train_tanimoto" in similarity and similarity.structure_id.nunique() == EXACT_ROWS:
        combined = combined.merge(
            similarity[["structure_id", "maximum_train_tanimoto"]].drop_duplicates("structure_id"),
            on="structure_id",
            how="left",
            validate="one_to_one",
        )
    else:
        raise Unavailable("fold-local Tanimoto applicability source is incomplete")
    combined["absolute_error"] = absolute_error
    risk_rows: list[dict[str, Any]] = []
    for retained_fraction in (0.25, 0.5, 0.75, 0.9, 1.0):
        n = max(1, int(math.ceil(len(combined) * retained_fraction)))
        selected = combined.nlargest(n, "maximum_train_tanimoto")
        risk_rows.append(
            {
                "retained_fraction": retained_fraction,
                "mae": float(selected.absolute_error.mean()),
                "minimum_similarity": float(selected.maximum_train_tanimoto.min()),
            }
        )
    prediction_path = directory / "uncertainty_ad_oof.parquet"
    calibration_path = directory / "cross_conformal_calibration.parquet"
    risk_path = directory / "applicability_risk_coverage.parquet"
    _write_parquet(prediction_path, combined)
    _write_parquet(calibration_path, pd.DataFrame(calibration_rows))
    _write_parquet(risk_path, pd.DataFrame(risk_rows))
    metrics = _metrics(combined.observed_pic50, combined.ensemble_prediction)
    metrics.update(selection_score=metrics["mae"], members=len(member_ids), ad_method=spec["ad_method"])
    return _write_unit(
        directory,
        unit_id=unit_id,
        operation="uncertainty_ad",
        spec=spec,
        metrics=metrics,
        artifacts=[
            _binding(prediction_path, "oof_predictions"),
            _binding(calibration_path, "calibration"),
            _binding(risk_path, "applicability_risk_coverage"),
        ],
        limitations=["Cross-conformal coverage and applicability are internal train-only estimates."],
    )


def _broad_unit(
    repo: Path,
    base_v3: Path,
    directory: Path,
    unit_id: str,
    spec: dict[str, Any],
    workers: int,
    *,
    operation: str = "broad_auxiliary",
) -> dict[str, Any]:
    candidate = _candidate(spec)
    if candidate["feature_set"] == "broad_sparse_morgan_rdkit":
        raise Unavailable(
            "packed sparse-Morgan broad training is gated until a memory-bounded CSR implementation is validated"
        )
    if candidate["engine"] == "logistic_saga":
        raise Unavailable("logistic_saga broad engine requires sparse-Morgan implementation")
    mapped = {
        "operation": "broad_wt_auxiliary",
        "candidate": candidate,
        "outer_folds": spec.get("outer_folds", 5),
        "fixed_fpr": spec.get("fixed_fpr", 0.01),
        "final_refit": operation == "final_refit_broad",
        "retain_model_artifact": bool(spec.get("retain_model_artifact", False)),
        "source_partition": "train",
    }
    legacy = v3._broad_unit(repo, base_v3 / "prepared", directory, unit_id, mapped, workers)
    artifacts = list(legacy["artifacts"])
    metrics = dict(legacy["metrics"])
    return _write_unit(
        directory,
        unit_id=unit_id,
        operation=operation,
        spec=spec,
        metrics=metrics,
        artifacts=artifacts,
        endpoint="confirmed-WT fixed-dose auxiliary classification",
        limitations=[
            "The 339,373-structure surface is represented by its governed 265,625-row train partition.",
            "This fixed-dose binary endpoint is never treated as exact pIC50 or pooled into it.",
        ],
    )


def _final_exact_unit(
    base_v2: Path,
    base_v3: Path,
    results_root: Path,
    directory: Path,
    unit_id: str,
    spec: dict[str, Any],
    workers: int,
) -> dict[str, Any]:
    source_id = str(spec.get("source_unit_id") or "")
    if not source_id:
        pool = [str(item) for item in spec["selection_source_unit_ids"]]
        ranked = sorted(
            (_source_unit(results_root, item) for item in pool),
            key=lambda unit: (float(unit["metrics"]["selection_score"]), unit["unit_id"]),
        )
        if not ranked:
            raise WorkerError("final exact selection pool is empty")
        source = ranked[0]
        source_id = str(source["unit_id"])
    else:
        source = _source_unit(results_root, source_id)
    candidate = _candidate(spec)
    if not candidate:
        candidate = _candidate(source["executed_spec"])
    if not candidate:
        raise WorkerError("final exact source does not expose a refittable classical candidate")
    _validate_candidate(candidate)
    source_frame = pd.read_parquet(_artifact(source, "oof_predictions"))
    if len(source_frame) != EXACT_ROWS or source_frame.structure_id.duplicated().any():
        raise WorkerError("final exact source lacks complete 18,801-row OOF evidence")
    frame, features = _load_exact(base_v2, base_v3, candidate["feature_set"])
    model = v3._fit_regression_model(
        _exact_model(candidate, workers, int(spec.get("seed", SEED))),
        frame[features],
        frame.target_pic50.to_numpy(dtype=float),
    )
    model_path = directory / "model.joblib"
    joblib.dump({"model": model, "features": features, "candidate": candidate}, model_path)
    schema_path = directory / "feature_schema.json"
    _write_json(
        schema_path,
        {
            "schema_version": SCHEMA,
            "features": features,
            "feature_set": candidate["feature_set"],
            "training_structures": len(frame),
            "target": "exact quantitative pIC50",
        },
        "feature_schema_sha256",
    )
    smoke = v3._predict_model(model, frame.iloc[:16][features])
    smoke_path = directory / "inference_smoke.json"
    _write_json(
        smoke_path,
        {
            "status": "passed",
            "rows": len(smoke),
            "all_finite": bool(np.isfinite(smoke).all()),
            "model_loadable": bool(joblib.load(model_path)),
        },
        "inference_smoke_sha256",
    )
    metrics = _metrics(source_frame.observed_pic50, source_frame.predicted_pic50)
    metrics.update(selection_score=metrics["mae"], source_unit_id=source_id, refit_uses_all_exact_train=True)
    return _write_unit(
        directory,
        unit_id=unit_id,
        operation="final_refit_exact",
        spec=spec,
        metrics=metrics,
        artifacts=[
            _binding(model_path, "model"),
            _binding(schema_path, "feature_schema"),
            _binding(smoke_path, "inference_smoke"),
            _binding(results_root / "units" / source_id / "unit.json", "selection_source_unit"),
        ],
        limitations=[
            "Persisted model is full-train refit; reported performance comes from bound OOF evidence."
        ],
    )


def _receptor_unit(directory: Path, unit_id: str, spec: dict[str, Any]) -> dict[str, Any]:
    paths = [
        Path(str(spec[key])).resolve() for key in ("receptor_artifact_manifest", "compound_panel_manifest")
    ]
    if not all(path.is_file() for path in paths):
        raise Unavailable("receptor pilot inputs are absent; no docking/OpenMM claim is made")
    documents = [json.loads(path.read_text("utf-8")) for path in paths]
    if not all(item.get("status") == "passed" for item in documents):
        raise Unavailable("receptor pilot inputs are not validated passed artifacts")
    report_path = directory / "receptor_pilot_gate.json"
    _write_json(
        report_path,
        {
            "status": "passed_input_gate_only",
            "protocol_id": spec["protocol_id"],
            "docking_or_dynamics_executed_by_this_unit": False,
            "claim_allowed": False,
        },
        "receptor_pilot_gate_sha256",
    )
    return _write_unit(
        directory,
        unit_id=unit_id,
        operation="receptor_pilot",
        spec=spec,
        metrics={"input_manifests": 2, "physics_execution": 0},
        artifacts=[_binding(path, "receptor_input_manifest") for path in paths]
        + [_binding(report_path, "receptor_pilot_gate")],
        limitations=[
            "Input validation only; no docking, OpenMM, affinity, or mechanistic claim is produced."
        ],
    )


def _select_panel(exact: pd.DataFrame, results_root: Path, spec: dict[str, Any]) -> pd.DataFrame:
    size = min(int(spec["panel_size"]), len(exact))
    if size < 1:
        raise WorkerError("panel_size must be positive")
    mode = str(spec["panel_selection"])
    if mode == "all_exact_train":
        ranked = exact.assign(
            _rank=exact.structure_id.astype(str).map(
                lambda value: hashlib.sha256(f"{spec.get('seed', SEED)}:{value}".encode()).hexdigest()
            )
        ).sort_values("_rank")
        return ranked.head(size).drop(columns="_rank")
    source_id = str(spec.get("source_unit_id") or "")
    if not source_id:
        raise WorkerError(f"{mode} requires a nested OOF source_unit_id")
    _, predictions = _prediction_source(results_root, source_id)
    if len(predictions) != EXACT_ROWS or predictions.structure_id.duplicated().any():
        raise WorkerError("residual panel selection requires complete nested OOF predictions")
    residual = predictions.assign(_error=np.abs(predictions.observed_pic50 - predictions.predicted_pic50))[
        ["structure_id", "_error"]
    ]
    ranked = exact.merge(residual, on="structure_id", validate="one_to_one")
    if mode == "nested_residual_stratified":
        ranked["_band"] = pd.qcut(ranked._error.rank(method="first"), 10, labels=False)
    else:
        mmp = pd.read_parquet(Path(__file__).resolve().parents[2] / v3.DEFAULT_MMP)
        cliff_ids = set(mmp.loc[mmp.activity_cliff_ge_1_pic50, "structure_id_a"]) | set(
            mmp.loc[mmp.activity_cliff_ge_1_pic50, "structure_id_b"]
        )
        ranked["_band"] = ranked.structure_id.isin(cliff_ids).astype(int) * 10 + pd.qcut(
            ranked._error.rank(method="first"), 5, labels=False
        )
    chunks: list[pd.DataFrame] = []
    per = max(1, math.ceil(size / ranked._band.nunique()))
    for band, group in ranked.groupby("_band"):
        band_value = band
        group = group.assign(
            _rank=group.structure_id.astype(str).map(
                lambda value, fixed_band=band_value: hashlib.sha256(
                    f"{fixed_band}:{value}".encode()
                ).hexdigest()
            )
        )
        chunks.append(group.sort_values("_rank").head(per))
    return pd.concat(chunks, ignore_index=True).head(size).drop(columns=["_rank", "_band", "_error"])


def _conformer_record(row: Any, spec: dict[str, Any]) -> tuple[list[dict[str, Any]], str | None]:
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, Descriptors3D, rdMolAlign
        from rdkit.Chem.MolStandardize import rdMolStandardize
    except ImportError as error:  # pragma: no cover - governed environment has RDKit
        raise Unavailable("RDKit 3D is unavailable") from error
    mol = Chem.MolFromSmiles(str(row.standardized_smiles))
    if mol is None:
        return [], "smiles_parse_failed"
    mol = Chem.AddHs(mol)
    requested = int(spec["requested_conformers"])
    params = AllChem.ETKDGv3()
    structure_seed = int.from_bytes(
        hashlib.sha256(f"{spec.get('seed', SEED)}:{row.structure_id}".encode()).digest()[:4], "little"
    )
    params.randomSeed = structure_seed & 0x7FFFFFFF
    params.numThreads = 1
    ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=requested, params=params))
    if not ids:
        return [], "embedding_failed"
    energies: dict[int, float] = {}
    force_field = "MMFF94s"
    properties = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94s")
    for conformer in ids:
        try:
            if properties is not None:
                field = AllChem.MMFFGetMoleculeForceField(mol, properties, confId=int(conformer))
            else:
                force_field = "UFF"
                field = AllChem.UFFGetMoleculeForceField(mol, confId=int(conformer))
            if field is None:
                continue
            field.Minimize(maxIts=int(spec["max_iterations"]))
            energies[int(conformer)] = float(field.CalcEnergy())
        except Exception:
            continue
    if not energies:
        return [], "force_field_failed"
    ordered = sorted(energies, key=lambda key: energies[key])[: int(spec["retained_conformers"])]
    rows: list[dict[str, Any]] = []
    tautomer_count = len(rdMolStandardize.TautomerEnumerator().Enumerate(Chem.RemoveHs(mol)))
    try:
        AllChem.ComputeGasteigerCharges(mol)
        charges = np.asarray(
            [float(atom.GetProp("_GasteigerCharge")) for atom in mol.GetAtoms()], dtype=float
        )
        charges[~np.isfinite(charges)] = 0.0
    except Exception:
        charges = np.zeros(mol.GetNumAtoms(), dtype=float)
    polar_atoms = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() in {7, 8, 15, 16}]
    for level in [int(item) for item in spec["levels"]]:
        active = ordered[: min(len(ordered), level, int(spec["retained_conformers"]))]
        energy = np.asarray([energies[item] for item in active], dtype=float)
        weights = np.exp(-(energy - energy.min()) / (0.0019872041 * 298.15))
        weights /= weights.sum()
        radius_of_gyration = np.asarray([Descriptors3D.RadiusOfGyration(mol, confId=item) for item in active])
        asphericity = np.asarray([Descriptors3D.Asphericity(mol, confId=item) for item in active])
        npr1 = np.asarray([Descriptors3D.NPR1(mol, confId=item) for item in active])
        dipoles: list[float] = []
        charge_radii: list[float] = []
        polar_exposure: list[float] = []
        internal_polar_contacts: list[float] = []
        for conformer_id in active:
            coordinates = np.asarray(mol.GetConformer(conformer_id).GetPositions(), dtype=float)
            center = coordinates.mean(axis=0)
            atom_radius = np.linalg.norm(coordinates - center, axis=1)
            dipoles.append(float(np.linalg.norm(np.sum(charges[:, None] * (coordinates - center), axis=0))))
            charge_radii.append(
                float(np.sum(np.abs(charges) * atom_radius) / max(np.abs(charges).sum(), 1e-12))
            )
            polar_exposure.append(float(np.mean(atom_radius[polar_atoms])) if polar_atoms else 0.0)
            contacts = 0
            for left, atom_left in enumerate(polar_atoms):
                for atom_right in polar_atoms[left + 1 :]:
                    if (
                        mol.GetBondBetweenAtoms(atom_left, atom_right) is None
                        and np.linalg.norm(coordinates[atom_left] - coordinates[atom_right]) <= 3.5
                    ):
                        contacts += 1
            internal_polar_contacts.append(float(contacts))
        rmsd: list[float] = []
        for left in range(len(active)):
            for right in range(left + 1, len(active)):
                rmsd.append(float(rdMolAlign.GetBestRMS(mol, mol, prbId=active[left], refId=active[right])))
        rows.append(
            {
                "structure_id": str(row.structure_id),
                "level": level,
                "embedded_conformers": len(ids),
                "retained_conformers": len(active),
                "force_field": force_field,
                "energy_min_kcal_mol": float(energy.min()),
                "energy_range_kcal_mol": float(energy.max() - energy.min()),
                "effective_conformer_count": float(1.0 / np.square(weights).sum()),
                "radius_gyration_boltzmann": float(np.sum(weights * radius_of_gyration)),
                "asphericity_boltzmann": float(np.sum(weights * asphericity)),
                "npr1_boltzmann": float(np.sum(weights * npr1)),
                "gasteiger_dipole_proxy_eA_boltzmann": float(np.sum(weights * dipoles)),
                "absolute_charge_radius_A_boltzmann": float(np.sum(weights * charge_radii)),
                "polar_radial_exposure_A_boltzmann": float(np.sum(weights * polar_exposure)),
                "internal_polar_contact_count_boltzmann": float(np.sum(weights * internal_polar_contacts)),
                "pairwise_rmsd_mean_A": float(np.mean(rmsd)) if rmsd else 0.0,
                "tautomer_count_hypothesis_only": tautomer_count,
                "protomer_population_model": "unavailable",
            }
        )
    return rows, None


def _conformer_unit(
    base_v2: Path,
    results_root: Path,
    directory: Path,
    unit_id: str,
    spec: dict[str, Any],
) -> dict[str, Any]:
    exact = pd.read_parquet(
        base_v2 / "prepared/exact_train_cache.parquet",
        columns=["structure_id", "standardized_smiles", "scaffold_group_id"],
    )
    panel = _select_panel(exact, results_root, spec)
    shard_size = int(spec["shard_size"])
    if not 25 <= shard_size <= 2_000:
        raise WorkerError("shard_size must be in [25,2000]")
    shard_root = directory / "shards"
    shard_root.mkdir(parents=True, exist_ok=True)
    shard_spec_path = directory / "conformer_shard_spec.json"
    shard_spec = {
        "unit_spec_sha256": _digest(spec),
        "panel_identities_sha256": _digest(sorted(panel.structure_id.astype(str))),
        "panel_size": len(panel),
        "levels": spec["levels"],
        "requested_conformers": spec["requested_conformers"],
        "retained_conformers": spec["retained_conformers"],
    }
    if shard_spec_path.is_file():
        existing_spec = _read_json(shard_spec_path, "conformer_shard_spec_sha256")
        if any(existing_spec.get(key) != value for key, value in shard_spec.items()):
            shutil.rmtree(shard_root)
            shard_root.mkdir(parents=True)
    _write_json(shard_spec_path, shard_spec, "conformer_shard_spec_sha256")
    rows = list(panel.itertuples(index=False))
    chunks = [rows[index : index + shard_size] for index in range(0, len(rows), shard_size)]
    pending: list[tuple[int, list[Any], Path]] = []
    for index, chunk in enumerate(chunks):
        path = shard_root / f"part-{index:05d}.parquet"
        if path.is_file():
            found = pd.read_parquet(path)
            expected = {str(row.structure_id) for row in chunk}
            succeeded = found.loc[found.level.ne(-1)]
            valid_levels = (
                set(pd.to_numeric(succeeded.level, errors="coerce").dropna().astype(int))
                == set(int(item) for item in spec["levels"])
                if not succeeded.empty
                else True
            )
            if set(found.structure_id.astype(str)) != expected or not valid_levels:
                path.unlink()
                pending.append((index, chunk, path))
        else:
            pending.append((index, chunk, path))
    if pending:
        with ProcessPoolExecutor(max_workers=max(1, min(6, os.cpu_count() or 1))) as pool:
            for _index, chunk, path in pending:
                results = list(pool.map(_conformer_record_for_pool, [(row, spec) for row in chunk]))
                block: list[dict[str, Any]] = []
                for row, (values, error) in zip(chunk, results, strict=True):
                    if values:
                        block.extend(values)
                    else:
                        block.append(
                            {
                                "structure_id": str(row.structure_id),
                                "level": -1,
                                "feature_status": "failed",
                                "error": error,
                            }
                        )
                _write_parquet(path, pd.DataFrame(block))
    shard_frames = [pd.read_parquet(shard_root / f"part-{index:05d}.parquet") for index in range(len(chunks))]
    all_rows = pd.concat(shard_frames, ignore_index=True)
    failures_frame = all_rows.loc[all_rows.level.eq(-1), ["structure_id", "error"]].copy()
    features = all_rows.loc[all_rows.level.ne(-1)].drop(columns=["feature_status", "error"], errors="ignore")
    failures = failures_frame.to_dict("records")
    if features.empty:
        raise WorkerError("fresh conformer calculation produced no features")
    path = directory / "fresh_conformer_convergence.parquet"
    failure_path = directory / "fresh_conformer_failures.parquet"
    _write_parquet(path, features)
    _write_parquet(failure_path, failures_frame)
    levels = sorted(features.level.unique())
    convergence: dict[str, Any] = {}
    if len(levels) > 1:
        pivot = features.pivot(index="structure_id", columns="level", values="radius_gyration_boltzmann")
        common = pivot.dropna(subset=[levels[0], levels[-1]])
        convergence["radius_gyration_first_to_last_median_abs_delta"] = float(
            np.median(np.abs(common[levels[-1]] - common[levels[0]]))
        )
    metrics = {
        "panel_requested": len(panel),
        "structures_succeeded": int(features.structure_id.nunique()),
        "structures_failed": len(failures),
        "requested_conformers": int(spec["requested_conformers"]),
        "levels": levels,
        "fresh_calculation": True,
        "checkpoint_shards": len(chunks),
        "requested_identity_coverage": int(features.structure_id.nunique() + len(failures)),
        "protomer_population_model": "unavailable",
        **convergence,
    }
    return _write_unit(
        directory,
        unit_id=unit_id,
        operation="microstate_conformer",
        spec=spec,
        metrics=metrics,
        artifacts=[_binding(path, "fresh_conformer_features"), _binding(failure_path, "conformer_failures")]
        + [_binding(shard_spec_path, "conformer_shard_spec")]
        + [
            _binding(shard_root / f"part-{index:05d}.parquet", "conformer_checkpoint_shard")
            for index in range(len(chunks))
        ],
        limitations=[
            "Tautomer counts are hypothesis-only; no protomer population or pH population model is available.",
            "Fresh RDKit conformers are ligand-only physics proxies and do not represent receptor binding thermodynamics.",
        ],
    )


def _conformer_record_for_pool(
    payload: tuple[Any, dict[str, Any]],
) -> tuple[list[dict[str, Any]], str | None]:
    return _conformer_record(*payload)


def _run_unit(args: argparse.Namespace) -> dict[str, Any]:
    repo = Path(args.repo_root).resolve()
    prepared = Path(args.prepared_root).resolve()
    results_root = Path(args.results_root).resolve()
    unit_id = _safe_id(str(args.unit_id))
    spec = _validate_spec(json.loads(args.unit_spec_json))
    base_v2, base_v3, _ = _context(prepared)
    directory = results_root / "units" / unit_id
    existing = _existing_unit(directory / "unit.json", spec, results_root)
    if existing is not None:
        return existing
    operation = str(spec["operation"])
    if directory.exists():
        if operation != "microstate_conformer":
            shutil.rmtree(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if operation == "baseline_reproduction":
        return _baseline_unit(base_v2, base_v3, directory, unit_id, spec)
    if operation in {"classical_hpo", "nested_outer_evaluation", "feature_family_ablation"}:
        if operation == "nested_outer_evaluation" and not _candidate(spec):
            raise WorkerError("orchestrator must resolve a classical candidate before outer evaluation")
        return _classical_unit(base_v2, base_v3, directory, unit_id, spec, args.workers)
    if operation == "chemprop_hpo":
        return _chemprop_unit(repo, base_v2, directory, unit_id, spec, args.workers)
    if operation == "aggregate_nested_oof":
        return _aggregate_nested_unit(base_v2, results_root, directory, unit_id, spec)
    if operation == "nested_stack":
        if not spec.get("member_unit_ids"):
            raise WorkerError("orchestrator must resolve model-diverse inner member_unit_ids")
        return _stack_unit(base_v2, base_v3, results_root, directory, unit_id, spec, args.workers)
    if operation == "assay_hierarchical":
        return _assay_unit(base_v2, results_root, directory, unit_id, spec)
    if operation == "censored_interval":
        return _censored_unit(repo, base_v3, directory, unit_id, spec, args.workers)
    if operation == "mmp_cliff":
        return _mmp_unit(repo, results_root, directory, unit_id, spec)
    if operation == "uncertainty_ad":
        return _uncertainty_unit(base_v2, results_root, directory, unit_id, spec)
    if operation == "microstate_conformer":
        return _conformer_unit(base_v2, results_root, directory, unit_id, spec)
    if operation == "receptor_pilot":
        return _receptor_unit(directory, unit_id, spec)
    if operation == "broad_auxiliary":
        return _broad_unit(repo, base_v3, directory, unit_id, spec, args.workers)
    if operation == "broad_transfer":
        raise Unavailable(
            "broad transfer remains gated until cross-fitted broad scores cover all exact structures without fold leakage"
        )
    if operation == "final_refit_exact":
        return _final_exact_unit(base_v2, base_v3, results_root, directory, unit_id, spec, args.workers)
    if operation == "final_refit_broad":
        if not spec.get("candidate"):
            raise WorkerError("orchestrator must resolve broad candidate before final refit")
        return _broad_unit(
            repo, base_v3, directory, unit_id, spec, args.workers, operation="final_refit_broad"
        )
    raise WorkerError(f"operation dispatch missing: {operation}")


def _report_from_units(units: list[dict[str, Any]], operation: str) -> dict[str, Any]:
    selected = [unit for unit in units if unit["operation"] == operation]
    return {
        "schema_version": SCHEMA,
        "created_at": _now(),
        "operation": operation,
        "units": [
            {
                "unit_id": unit["unit_id"],
                "metrics": unit["metrics"],
                "limitations": unit.get("limitations", []),
                "artifacts": unit.get("artifacts", []),
            }
            for unit in selected
        ],
        "repository_validation_labels_opened": False,
        "repository_test_labels_opened": False,
    }


def _analyze(args: argparse.Namespace) -> dict[str, Any]:
    prepared = Path(args.prepared_root).resolve()
    results_root = Path(args.results_root).resolve()
    output = Path(args.output_root).resolve()
    base_v2, base_v3, prepared_validation = _context(prepared)
    if (output / "validation.json").is_file():
        return _read_json(output / "validation.json", "validation_sha256")
    if output.exists() and any(output.iterdir()):
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    units: list[dict[str, Any]] = []
    rejected: list[dict[str, str]] = []
    for path in sorted((results_root / "units").glob("*/unit.json")):
        try:
            unit = _read_json(path, "unit_json_sha256")
            if unit.get("status") != "passed":
                raise WorkerError(f"status={unit.get('status')}")
            scope = unit.get("scientific_scope", {})
            if (
                scope.get("source_partition") != "train"
                or scope.get("repository_validation_labels_opened") is not False
                or scope.get("repository_test_labels_opened") is not False
                or scope.get("broad_fixed_dose_pooled_into_pic50") is not False
            ):
                raise WorkerError("scientific scope failed")
            for artifact in unit.get("artifacts", []):
                _verify_binding(artifact, results_root)
            units.append(unit)
        except Exception as error:
            rejected.append({"path": str(path), "reason": str(error)})
    if not units:
        raise WorkerError("analysis found no valid passed V4 units")
    nested = [unit for unit in units if unit["operation"] == "aggregate_nested_oof"]
    if len(nested) != 1:
        raise WorkerError("analysis requires exactly one governed nested OOF aggregate")
    nested_frame = pd.read_parquet(_artifact(nested[0], "oof_predictions"))
    if len(nested_frame) != EXACT_ROWS or nested_frame.structure_id.duplicated().any():
        raise WorkerError("nested aggregate coverage failed during final analysis")
    v2 = pd.read_parquet(base_v2 / "analysis/outer_oof_predictions.parquet")
    v2 = v2.loc[v2.model_id.eq("nested_selected")]
    comparison = nested_frame[["structure_id", "observed_pic50", "predicted_pic50"]].merge(
        v2[["structure_id", "observed_pic50", "predicted_pic50"]],
        on=["structure_id", "observed_pic50"],
        suffixes=("_v4", "_v2"),
        validate="one_to_one",
    )
    paired_delta = np.abs(comparison.observed_pic50 - comparison.predicted_pic50_v4) - np.abs(
        comparison.observed_pic50 - comparison.predicted_pic50_v2
    )
    v2_metrics = _metrics(comparison.observed_pic50, comparison.predicted_pic50_v2)
    v4_metrics = _metrics(comparison.observed_pic50, comparison.predicted_pic50_v4)
    decisions = {
        "schema_version": SCHEMA,
        "created_at": _now(),
        "primary_internal_estimate": v4_metrics,
        "v2_anchor": v2_metrics,
        "paired_v4_minus_v2_mae_delta": float(paired_delta.mean()),
        "v4_better_than_v2_on_internal_nested_mae": bool(paired_delta.mean() < 0),
        "selection_basis": "five disjoint nested outer folds covering exact train once",
        "chemprop_status": (
            "inner-only model-diverse evidence; deliberately ineligible for outer finalist/stack in V4"
        ),
        "predictive_superiority_established": False,
        "next_gate": "freeze finalist then evaluate locked repository validation or external prospective data",
    }
    final_exact = [unit for unit in units if unit["operation"] == "final_refit_exact"]
    final_broad = [unit for unit in units if unit["operation"] == "final_refit_broad"]
    if len(final_exact) != 1 or len(final_broad) != 1:
        raise WorkerError("analysis requires exactly one exact and one broad final refit")
    required_roles = {"model", "feature_schema", "inference_smoke"}
    final_entries: dict[str, Any] = {}
    for key, unit in (("quantitative_exact", final_exact[0]), ("broad_fixed_dose", final_broad[0])):
        artifacts = {item["role"]: item for item in unit["artifacts"]}
        if not required_roles.issubset(artifacts):
            raise WorkerError(f"{key} finalist lacks deployable artifact roles")
        final_entries[key] = {
            "unit_id": unit["unit_id"],
            "metrics": unit["metrics"],
            "artifacts": [artifacts[role] for role in sorted(required_roles)],
        }
    reports = {
        "assay_quality_report.json": _report_from_units(units, "assay_hierarchical"),
        "censored_report.json": _report_from_units(units, "censored_interval"),
        "mmp_cliff_report.json": _report_from_units(units, "mmp_cliff"),
        "broad_transfer_report.json": {
            **_report_from_units(units, "broad_transfer"),
            "blocked_if_absent": "cross-fitted broad-to-exact structure coverage gate",
        },
        "uncertainty_ad.json": _report_from_units(units, "uncertainty_ad"),
    }
    for name, report in reports.items():
        _write_json(output / name, report, name.replace(".json", "_sha256"))
    relationships: list[dict[str, Any]] = []
    for unit in units:
        for artifact in unit["artifacts"]:
            if artifact["role"] == "conditional_permutation":
                relationships.extend(pd.read_parquet(artifact["path"]).to_dict("records"))
    _write_json(
        output / "feature_relationships.json",
        {
            "schema_version": SCHEMA,
            "created_at": _now(),
            "relationships": relationships,
            "causal_interpretation_allowed": False,
            "interpretation": "Held-out fold relationships are exploratory hypotheses, not direct channel mechanisms.",
        },
        "feature_relationships_sha256",
    )
    _write_json(output / "decision_ledger.json", decisions, "decision_ledger_sha256")
    _write_json(
        output / "model_cards.json",
        {
            "schema_version": SCHEMA,
            "created_at": _now(),
            "models": [
                {
                    "unit_id": unit["unit_id"],
                    "operation": unit["operation"],
                    "metrics": unit["metrics"],
                    "limitations": unit.get("limitations", []),
                    "external_validation_completed": False,
                }
                for unit in units
                if unit["operation"] in {"aggregate_nested_oof", "final_refit_exact", "final_refit_broad"}
            ],
        },
        "model_cards_sha256",
    )
    _write_json(
        output / "final_models_manifest.json",
        {"schema_version": SCHEMA, "created_at": _now(), **final_entries},
        "final_models_manifest_sha256",
    )
    lines = [
        "# hERG discovery campaign V4",
        "",
        f"- Valid passed units: {len(units)}; rejected units: {len(rejected)}.",
        f"- Unbiased internal nested MAE: {v4_metrics['mae']:.6f} pIC50 log units.",
        f"- V2 anchor nested MAE: {v2_metrics['mae']:.6f}.",
        f"- Paired V4 minus V2 MAE: {float(paired_delta.mean()):+.6f}.",
        "- Exact quantitative and broad fixed-dose endpoints remain strictly separate.",
        "- Chemprop is inner-only evidence in this release and is not represented as an outer finalist.",
        "- Repository validation and test outcomes remain sealed.",
        "- Feature and assay findings are hypothesis-generating and noncausal.",
    ]
    analysis_path = output / "analysis.md"
    analysis_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    dossier = [
        "analysis.md",
        "decision_ledger.json",
        "feature_relationships.json",
        "uncertainty_ad.json",
        "assay_quality_report.json",
        "censored_report.json",
        "mmp_cliff_report.json",
        "broad_transfer_report.json",
        "model_cards.json",
        "final_models_manifest.json",
    ]
    manifest = _write_json(
        output / "manifest.json",
        {
            "schema_version": SCHEMA,
            "created_at": _now(),
            "status": "passed",
            "prepared_validation_sha256": prepared_validation["validation_sha256"],
            "v3_worker_source_sha256": _sha(Path(str(v3.__file__)).resolve()),
            "artifacts": [_binding(output / name, name) for name in dossier],
            "unit_count": len(units),
            "rejected_units": rejected,
        },
        "manifest_sha256",
    )
    return _write_json(
        output / "validation.json",
        {
            "schema_version": SCHEMA,
            "created_at": _now(),
            "status": "passed",
            "manifest": _binding(output / "manifest.json", "analysis_manifest"),
            "manifest_sha256_bound": manifest["manifest_sha256"],
            "exact_nested_rows": len(nested_frame),
            "repository_validation_labels_opened": False,
            "repository_test_labels_opened": False,
            "broad_fixed_dose_pooled_into_pic50": False,
        },
        "validation_sha256",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("capabilities")
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--repo-root", required=True)
    prepare.add_argument("--base-v2-root", required=True)
    prepare.add_argument("--base-v3-root", required=True)
    prepare.add_argument("--output-root", required=True)
    prepare.add_argument("--workers", type=int, default=6)
    unit = subparsers.add_parser("run-unit")
    unit.add_argument("--repo-root", required=True)
    unit.add_argument("--prepared-root", required=True)
    unit.add_argument("--results-root", required=True)
    unit.add_argument("--unit-id", required=True)
    unit.add_argument("--unit-spec-json", required=True)
    unit.add_argument("--workers", type=int, default=6)
    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("--repo-root", required=True)
    analyze.add_argument("--prepared-root", required=True)
    analyze.add_argument("--results-root", required=True)
    analyze.add_argument("--output-root", required=True)
    analyze.add_argument("--workers", type=int, default=6)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        if args.command == "capabilities":
            print(json.dumps(_capabilities(), sort_keys=True, separators=(",", ":"), allow_nan=False))
            return 0
        if args.command == "prepare":
            result = _prepare(args)
        elif args.command == "run-unit":
            result = _run_unit(args)
        elif args.command == "analyze":
            raise Unavailable("analysis is deferred from the 15-hour compute-only launch")
        else:  # pragma: no cover
            raise WorkerError(f"unsupported command: {args.command}")
        print(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False))
        return 0
    except Unavailable as error:
        print(json.dumps({"status": "unavailable", "error": str(error)}, sort_keys=True), file=sys.stderr)
        return 3
    except Exception as error:
        print(json.dumps({"status": "failed", "error": str(error)}, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
