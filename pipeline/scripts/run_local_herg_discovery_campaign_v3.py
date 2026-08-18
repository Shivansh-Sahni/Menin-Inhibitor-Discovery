#!/usr/bin/env python3
"""Run the extended, train-only hERG discovery campaign (v3).

This is a new campaign layered on the completed v2 campaign.  It never edits
v2.  The primary task is exact-pIC50 regression on 18,801 train structures
(wild-type-or-target-unspecified, not confirmed-WT-only).  A separate
confirmed-WT fixed-dose binary task uses only the train partition of a
339,373-structure surface; its labels are never pooled into pIC50 regression.

The campaign performs prespecified HPO, adaptive promotion, multi-seed nested
scaffold confirmation, mechanistic ablation and interaction analyses,
assay-quality/censoring/activity-cliff diagnostics, uncertainty calibration,
optional Chemprop ensembles, and final full-train refits.  Repository
validation and test labels remain sealed.  Results are local diagnostics, not
prospective validation or evidence of superiority.
"""

from __future__ import annotations

import argparse
import collections
import copy
import fcntl
import hashlib
import json
import math
import os
import shutil
import signal
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Any

SCHEMA_VERSION = "platform-local-herg-discovery-campaign-v3/1.0"
DEFAULT_OUTPUT = Path("research/local_runs/herg_discovery_campaign_v3")
DEFAULT_BASE = Path("research/local_runs/herg_discovery_campaign_v1")
DEFAULT_BROAD = Path(
    "research/data/platform/processed/herg_hierarchy/v1_6_training_surfaces/"
    "confirmed_wt_fixed_dose_structure_labels.parquet"
)
DEFAULT_WORKERS = 6
MAX_ATTEMPTS = 2
RECOVERY_MIGRATION_ID = "v3-worker-newline-hash-censored-writability/1"
RECOVERY_OLD_IMPLEMENTATION_HASHES = {
    "v3_orchestrator": "16e4e334d7159e70a3c90046d7aa484c8e383ff70df87e3c282de9b7a78c0ae3",
    "v3_worker": "6052a21ee2c2476c3ad0ff9eac27c512bbc274bbdba052f83c48b7ad3fa195a5",
}
RECOVERY_EXPECTED_STATUS_COUNTS = {"failed_noncritical": 125, "pending": 45, "passed": 1}
RECOVERY_EXPECTED_STAGE_COUNTS = {
    "hpo_coarse_o0": 24,
    "hpo_coarse_o1": 24,
    "hpo_coarse_o2": 24,
    "hpo_coarse_o3": 24,
    "hpo_coarse_o4": 24,
    "broad_wt_hpo": 4,
}
RECOVERY_CENSORED_UNIT_ID = "censored_interval_likelihood"
ANALYSIS_FILES = (
    "validation.json",
    "analysis.md",
    "manifest.json",
    "model_cards.json",
    "decision_ledger.json",
    "feature_relationships.json",
    "final_models_manifest.json",
)
TERMINAL = {
    "passed",
    "failed_noncritical",
    "failed_critical",
    "skipped_unavailable",
    "skipped_time",
}


class CampaignError(RuntimeError):
    """Raised when campaign governance or reproducibility validation fails."""


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _self_hashed(value: dict[str, Any], key: str) -> dict[str, Any]:
    result = copy.deepcopy(value)
    result.pop(key, None)
    result[key] = hashlib.sha256(_canonical(result)).hexdigest()
    return result


def _verify_self_hash(value: dict[str, Any], key: str) -> None:
    if value.get(key) != _self_hashed(value, key)[key]:
        raise CampaignError(f"{key} mismatch")


def _atomic_json(path: Path, value: dict[str, Any], hash_key: str) -> dict[str, Any]:
    result = _self_hashed(value, hash_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return result


def _immutable_snapshot(source: Path, destination: Path) -> dict[str, Any]:
    """Atomically create a no-overwrite, byte-exact snapshot and bind it."""
    source_bytes = source.read_bytes()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.read_bytes() != source_bytes:
            raise CampaignError(f"immutable snapshot differs from source: {destination}")
    else:
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
        try:
            with temporary.open("xb") as handle:
                handle.write(source_bytes)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temporary, destination)
            except FileExistsError:
                if destination.read_bytes() != source_bytes:
                    raise CampaignError(f"immutable snapshot collision: {destination}") from None
        finally:
            temporary.unlink(missing_ok=True)
    if destination.read_bytes() != source_bytes:
        raise CampaignError(f"immutable snapshot verification failed: {destination}")
    binding = _binding(destination, "pre_migration_checkpoint_snapshot")
    if binding["sha256"] != hashlib.sha256(source_bytes).hexdigest():
        raise CampaignError("snapshot hash differs from original checkpoint bytes")
    return binding


def _read_json(path: Path, hash_key: str | None = None) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CampaignError(f"expected JSON object: {path}")
    if hash_key is not None:
        _verify_self_hash(value, hash_key)
    return value


def _worker_canonical(value: Any) -> bytes:
    """Canonical form used by the v3 worker for its own JSON artifacts."""
    return _canonical(value) + b"\n"


def _worker_hash(value: Any) -> str:
    return hashlib.sha256(_worker_canonical(value)).hexdigest()


def _read_worker_json(path: Path, hash_key: str) -> dict[str, Any]:
    """Read a worker artifact without changing checkpoint hash semantics.

    Checkpoints deliberately continue to use :func:`_canonical`.  Worker JSON
    documents were historically written and self-hashed with one trailing
    newline, so their verifier must use that distinct, explicit contract.
    """
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CampaignError(f"expected worker JSON object: {path}")
    expected = value.get(hash_key)
    unhashed = dict(value)
    unhashed.pop(hash_key, None)
    actual = _worker_hash(unhashed)
    if not isinstance(expected, str) or expected != actual:
        raise CampaignError(f"{hash_key} mismatch")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _binding(path: Path, role: str) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise CampaignError(f"missing bound {role}: {path}")
    return {"role": role, "path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}


def _verify_bindings(bindings: list[dict[str, Any]]) -> None:
    for binding in bindings:
        path = Path(str(binding["path"]))
        if not path.is_file() or path.stat().st_size != int(binding["bytes"]):
            raise CampaignError(f"bound file missing or changed: {path}")
        if _sha256(path) != binding["sha256"]:
            raise CampaignError(f"bound file hash changed: {path}")


def _candidate(
    candidate_id: str, engine: str, feature_set: str, params: dict[str, Any], rationale: str
) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "engine": engine,
        "feature_set": feature_set,
        "params": params,
        "rationale": rationale,
    }


def _hpo_candidates() -> list[dict[str, Any]]:
    """Nonredundant, prespecified XGBoost/LightGBM search regimes."""
    feature_sets = (
        "rdkit2d",
        "fundamental_core",
        "fundamental_interactions",
        "fingerprint_plus_fundamental",
        "morgan_rdkit2d",
        "all_scalable_v3",
    )
    xgb_regimes = (
        ("shallow", 4, 0.035, 700, 5, 0.10, 2.0),
        ("regularized", 5, 0.025, 1000, 8, 0.50, 4.0),
        ("deeper", 7, 0.020, 1200, 8, 0.30, 3.0),
    )
    candidates: list[dict[str, Any]] = []
    for index, features in enumerate(feature_sets):
        # Two materially distinct regimes per surface avoid repetitive sweeps.
        for name, depth, rate, trees, child, alpha, ridge in (
            xgb_regimes[index % 3],
            xgb_regimes[(index + 1) % 3],
        ):
            candidates.append(
                _candidate(
                    f"xgb_{features}_{name}",
                    "xgboost",
                    features,
                    {
                        "n_estimators": trees,
                        "max_depth": depth,
                        "learning_rate": rate,
                        "min_child_weight": child,
                        "subsample": 0.82,
                        "colsample_bytree": 0.78,
                        "reg_alpha": alpha,
                        "reg_lambda": ridge,
                        "objective": "reg:squarederror",
                    },
                    f"{features} under the {name} bias-variance regime",
                )
            )
    lgbm_regimes = (
        ("compact", 15, 30, 0.035, 700, 0.1, 2.0),
        ("balanced", 31, 25, 0.025, 950, 0.2, 3.0),
        ("wide", 63, 45, 0.020, 1150, 0.5, 4.0),
    )
    for index, features in enumerate(feature_sets):
        for name, leaves, child, rate, trees, alpha, ridge in (
            lgbm_regimes[index % 3],
            lgbm_regimes[(index + 1) % 3],
        ):
            candidates.append(
                _candidate(
                    f"lgbm_{features}_{name}",
                    "lightgbm",
                    features,
                    {
                        "n_estimators": trees,
                        "num_leaves": leaves,
                        "min_child_samples": child,
                        "learning_rate": rate,
                        "subsample": 0.82,
                        "colsample_bytree": 0.78,
                        "reg_alpha": alpha,
                        "reg_lambda": ridge,
                    },
                    f"{features} under the {name} leaf-complexity regime",
                )
            )
    return candidates


def _common_spec(operation: str, task: str = "exact_pic50_regression") -> dict[str, Any]:
    return {
        "operation": operation,
        "task": task,
        "source_partition": "train",
        "repository_validation_labels_opened": False,
        "repository_test_labels_opened": False,
        "broad_fixed_dose_pooled_into_pic50": False,
    }


def _unit(
    unit_id: str,
    stage: str,
    operation: str,
    expected_minutes: float,
    *,
    critical: bool = False,
    required: bool = False,
    dependencies: list[dict[str, Any]] | None = None,
    priority_tier: str = "core",
    **spec: Any,
) -> dict[str, Any]:
    base = _common_spec(operation, str(spec.pop("task", "exact_pic50_regression")))
    base.update(spec)
    return {
        "unit_id": unit_id,
        "stage": stage,
        "operation": operation,
        "expected_minutes": expected_minutes,
        "critical": critical,
        "required_for_completion": required,
        "dependencies": dependencies or [],
        "priority_tier": priority_tier,
        "status": "pending",
        "attempts": 0,
        "spec": base,
    }


def _stage_dependency(stage: str, minimum_passed: int, require_terminal: bool = True) -> dict[str, Any]:
    return {
        "stage": stage,
        "minimum_passed": minimum_passed,
        "require_terminal": require_terminal,
    }


def _plan() -> list[dict[str, Any]]:
    """Build a material, adaptive plan bounded well below 29 active hours."""
    units = [
        _unit("prepare_v3", "prepare", "prepare", 10, critical=True, required=True),
    ]
    prepare = [_stage_dependency("prepare", 1)]

    # Candidate selection is repeated independently inside every outer context.
    # The held-out outer scaffold labels never influence that fold's recipe.
    for outer_fold in range(5):
        coarse_stage = f"hpo_coarse_o{outer_fold}"
        promoted_stage = f"hpo_promoted_o{outer_fold}"
        for candidate in _hpo_candidates():
            units.append(
                _unit(
                    f"hpo_coarse_o{outer_fold}__{candidate['candidate_id']}",
                    coarse_stage,
                    "expanded_hpo_tree",
                    1.5,
                    dependencies=prepare,
                    candidate=candidate,
                    evaluation_stage="inner",
                    outer_fold=outer_fold,
                    inner_folds=[0],
                    seed=2026081201,
                    retain_model_artifact=False,
                )
            )
        for rank in range(4):
            units.append(
                _unit(
                    f"hpo_promoted_o{outer_fold}_rank_{rank}",
                    promoted_stage,
                    "expanded_hpo_tree",
                    4,
                    dependencies=[_stage_dependency(coarse_stage, 16)],
                    selection={"source_stage": coarse_stage, "rank": rank},
                    evaluation_stage="inner",
                    outer_fold=outer_fold,
                    inner_folds=[0, 1, 2],
                    seed=2026081211 + rank,
                    retain_model_artifact=False,
                )
            )
        units.append(
            _unit(
                f"nested_outer_{outer_fold}",
                "nested_confirmation",
                "nested_robustness",
                6,
                dependencies=[_stage_dependency(promoted_stage, 3)],
                selection={"source_stage": promoted_stage, "rank": 0},
                evaluation_stage="outer",
                outer_folds=[outer_fold],
                seed=2026081301 + outer_fold,
                retain_model_artifact=True,
                scaffold_entity_exclusive=True,
            )
        )

    # Global train-only comparison aggregates each candidate's inner scores
    # across all five outer contexts; repeats then quantify seed sensitivity.
    # Every candidate has one complete coarse score in each of the five outer
    # contexts, so global repeat recipes require five material observations.
    coarse_stages = [f"hpo_coarse_o{outer}" for outer in range(5)]
    for rank in range(2):
        for seed_index, seed in enumerate((2026081501, 2026081601)):
            units.append(
                _unit(
                    f"repeat_global_rank_{rank}_seed_{seed_index}",
                    "repeated_seed_confirmation",
                    "repeated_seed_tree",
                    20,
                    dependencies=[_stage_dependency("nested_confirmation", 5)],
                    selection={
                        "source_stages": coarse_stages,
                        "rank": rank,
                        "aggregate": "mean",
                        "minimum_material_repeats": 5,
                    },
                    evaluation_stage="outer",
                    outer_folds=[0, 1, 2, 3, 4],
                    seed=seed + rank,
                    retain_model_artifact=True,
                )
            )

    # Six explicitly different, supported surfaces form the ablation ladder.
    ablations = (
        "rdkit2d",
        "fundamental_core",
        "fundamental_interactions",
        "fingerprint_plus_fundamental",
        "morgan_rdkit2d",
        "all_scalable_v3",
    )
    standard_candidate = {
        "engine": "xgboost",
        "params": {
            "n_estimators": 700,
            "max_depth": 5,
            "learning_rate": 0.035,
            "min_child_weight": 5,
            "subsample": 0.82,
            "colsample_bytree": 0.78,
            "reg_alpha": 0.1,
            "reg_lambda": 2.0,
        },
    }
    for feature_set in ablations:
        candidate = {
            **standard_candidate,
            "candidate_id": f"ablation_{feature_set}",
            "feature_set": feature_set,
            "rationale": "fixed-capacity feature-family comparison",
        }
        units.append(
            _unit(
                f"ablation_{feature_set}",
                "feature_ablation",
                "feature_ablation",
                8,
                dependencies=[_stage_dependency("nested_confirmation", 5)],
                candidate=candidate,
                evaluation_stage="outer",
                outer_folds=[0, 1, 2, 3, 4],
                ablation_id=feature_set,
                seed=2026081801,
            )
        )

    units.extend(
        [
            _unit(
                "interaction_all_prespecified",
                "interaction_stability",
                "interaction_stability",
                25,
                dependencies=[_stage_dependency("repeated_seed_confirmation", 4)],
                candidate={
                    **standard_candidate,
                    "candidate_id": "interaction_fundamental_interactions",
                    "feature_set": "fundamental_interactions",
                    "rationale": "fixed-capacity prespecified interaction surface",
                },
                evaluation_stage="outer",
                outer_folds=[0, 1, 2, 3, 4],
                feature_set="fundamental_interactions",
                seed=2026081901,
            ),
            _unit(
                "assay_quality_all_supported",
                "assay_quality_strata",
                "assay_quality_strata",
                10,
                dependencies=[_stage_dependency("repeated_seed_confirmation", 4)],
                selection={"source_stage": "repeated_seed_confirmation", "rank": 0},
                stratum="all",
                minimum_subgroup_size=100,
            ),
            _unit(
                "uncertainty_repeated_ensemble",
                "uncertainty_calibration",
                "uncertainty_calibration",
                15,
                dependencies=[_stage_dependency("repeated_seed_confirmation", 4)],
                member_source_stage="repeated_seed_confirmation",
                method="ensemble_variance_conformal",
                coverage_levels=[0.5, 0.8, 0.9, 0.95],
            ),
            _unit(
                "censored_interval_likelihood",
                "censored_sensitivity",
                "censored_sensitivity",
                20,
                dependencies=prepare,
                priority_tier="target",
                treatment="interval_likelihood",
                never_point_impute_bounds=True,
                outer_folds=5,
                seed=2026082001,
            ),
            _unit(
                "mmp_activity_cliff_residual",
                "mmp_cliff_residual",
                "mmp_cliff_residual",
                10,
                dependencies=[_stage_dependency("repeated_seed_confirmation", 4)],
                selection={"source_stage": "repeated_seed_confirmation", "rank": 0},
                analysis_id="matched_pairs_and_activity_cliffs",
                minimum_pair_support=20,
            ),
        ]
    )

    # Confirmed-WT fixed-dose labels are a separate imbalanced classification
    # task.  The full surface has 339,373 rows; only its train split is fitted.
    broad_candidates = (
        _candidate(
            "broad_xgb_compact", "xgboost", "broad_compact_19", {"max_depth": 4}, "compact properties"
        ),
        _candidate(
            "broad_xgb_all", "xgboost", "broad_all_rdkit", {"max_depth": 6}, "all cached RDKit descriptors"
        ),
        _candidate(
            "broad_lgbm_compact", "lightgbm", "broad_compact_19", {"num_leaves": 15}, "compact leaf-wise"
        ),
        _candidate(
            "broad_lgbm_all", "lightgbm", "broad_all_rdkit", {"num_leaves": 31}, "all-descriptor leaf-wise"
        ),
    )
    for candidate in broad_candidates:
        units.append(
            _unit(
                f"broad_hpo__{candidate['candidate_id']}",
                "broad_wt_hpo",
                "broad_wt_auxiliary",
                25,
                dependencies=prepare,
                task="confirmed_wt_fixed_dose_binary",
                candidate=candidate,
                final_refit=False,
                full_surface_rows=339373,
                fit_partition="train",
                imbalance_strategy="scale_pos_weight_plus_calibration",
                selection_metric="pr_auc",
                report_metrics=["pr_auc", "roc_auc", "mcc", "brier", "enrichment_at_1pct"],
                outer_folds=3,
                seed=2026082101,
            )
        )
    units.append(
        _unit(
            "broad_wt_final_refit",
            "broad_wt_final",
            "broad_wt_auxiliary",
            60,
            critical=True,
            required=True,
            dependencies=[_stage_dependency("broad_wt_hpo", 3)],
            task="confirmed_wt_fixed_dose_binary",
            selection={"source_stage": "broad_wt_hpo", "rank": 0},
            final_refit=True,
            full_surface_rows=339373,
            fit_partition="train",
            imbalance_strategy="scale_pos_weight_plus_calibration",
            selection_metric="pr_auc",
            retain_model_artifact=True,
            require_inference_smoke=True,
        )
    )

    # Existing complete train-only Chemprop OOF predictions are combined at
    # three genuinely different prespecified weights; no fake retraining knobs.
    for tree_weight in (0.25, 0.50, 0.75):
        units.append(
            _unit(
                f"chemprop_tree_weight_{str(tree_weight).replace('.', 'p')}",
                "chemprop_ensemble",
                "chemprop_ensemble",
                5,
                dependencies=[_stage_dependency("repeated_seed_confirmation", 4)],
                priority_tier="stretch",
                selection={"source_stage": "repeated_seed_confirmation", "rank": 0},
                tree_weight=tree_weight,
            )
        )

    units.append(
        _unit(
            "exact_pic50_final_refit",
            "exact_final",
            "finalist_refit_artifact",
            60,
            critical=True,
            required=True,
            dependencies=[
                _stage_dependency("nested_confirmation", 5),
                _stage_dependency("repeated_seed_confirmation", 4),
            ],
            selection={"source_stage": "repeated_seed_confirmation", "rank": 0},
            fit_partition="train",
            retain_model_artifact=True,
            require_feature_schema=True,
            require_inference_smoke=True,
            ensemble_with_repeated_seeds=True,
        )
    )
    units.append(
        _unit(
            "analyze_v3",
            "analyze",
            "analyze",
            45,
            critical=True,
            required=True,
            dependencies=[
                _stage_dependency("exact_final", 1),
                _stage_dependency("broad_wt_final", 1),
                _stage_dependency("feature_ablation", 6),
                _stage_dependency("interaction_stability", 1),
                _stage_dependency("assay_quality_strata", 1),
                _stage_dependency("uncertainty_calibration", 1),
                _stage_dependency("mmp_cliff_residual", 1),
            ],
        )
    )
    return units


class _Lock:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.handle: IO[str] | None = None

    def __enter__(self) -> _Lock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise CampaignError(f"campaign lock already held: {self.path}") from error
        self.handle.seek(0)
        self.handle.truncate()
        self.handle.write(json.dumps({"pid": os.getpid(), "acquired_utc": _utc()}) + "\n")
        self.handle.flush()
        return self

    def __exit__(self, *_: object) -> None:
        if self.handle is not None:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            self.handle.close()


def _resource_gate(output: Path, disk_gib: float, memory_gib: float) -> dict[str, Any]:
    free_disk = shutil.disk_usage(output).free
    if free_disk < disk_gib * 1024**3:
        raise CampaignError(f"disk gate: only {free_disk / 1024**3:.1f} GiB free")
    available: int | None = None
    method = "unavailable_best_effort"
    try:
        query = subprocess.run(
            ["memory_pressure", "-Q"], check=False, capture_output=True, text=True, timeout=10
        )
        line = next(line for line in query.stdout.splitlines() if "memory free percentage" in line)
        percent = float(line.rsplit(":", 1)[1].strip().rstrip("%"))
        total = int(
            subprocess.check_output(
                ["sysctl", "-n", "hw.memsize"], text=True, stderr=subprocess.DEVNULL
            ).strip()
        )
        available = int(total * percent / 100)
        method = "macos_memory_pressure"
    except (FileNotFoundError, StopIteration, OSError, ValueError, subprocess.SubprocessError):
        pass
    if available is not None and available < memory_gib * 1024**3:
        raise CampaignError(f"memory gate: only {available / 1024**3:.1f} GiB available")
    return {
        "status": "passed",
        "checked_utc": _utc(),
        "disk_free_bytes": free_disk,
        "memory_available_bytes": available,
        "memory_method": method,
    }


def _checkpoint(output: Path, value: dict[str, Any]) -> dict[str, Any]:
    value["updated_utc"] = _utc()
    return _atomic_json(output / "checkpoint.json", value, "checkpoint_sha256")


def _stage_units(checkpoint: dict[str, Any], stage: str) -> list[dict[str, Any]]:
    return [unit for unit in checkpoint["units"] if unit["stage"] == stage]


def _dependencies_ready(checkpoint: dict[str, Any], unit: dict[str, Any]) -> bool:
    for dependency in unit["dependencies"]:
        source = _stage_units(checkpoint, str(dependency["stage"]))
        passed = sum(item["status"] == "passed" for item in source)
        if passed < int(dependency["minimum_passed"]):
            return False
        if dependency.get("require_terminal", True) and any(
            item["status"] not in TERMINAL for item in source
        ):
            return False
    return True


def _unit_result(output: Path, unit_id: str) -> dict[str, Any]:
    return _read_worker_json(output / "units" / unit_id / "unit.json", "unit_json_sha256")


def _finite_score(result: dict[str, Any]) -> float:
    value = result.get("metrics", {}).get("selection_score")
    if not isinstance(value, (float, int)) or not math.isfinite(float(value)):
        raise CampaignError("unit lacks a finite metrics.selection_score")
    return float(value)


def _rank_stages(checkpoint: dict[str, Any], stages: list[str]) -> list[dict[str, Any]]:
    """Rank candidates by the mean score over all passed material repeats."""
    output = Path(checkpoint["output_root"])
    grouped: dict[str, dict[str, Any]] = {}
    for stage in stages:
        for unit in _stage_units(checkpoint, stage):
            if unit["status"] != "passed":
                continue
            candidate = unit["spec"].get("resolved_candidate", unit["spec"].get("candidate"))
            if not isinstance(candidate, dict):
                continue
            candidate_id = str(candidate.get("candidate_id", ""))
            if not candidate_id:
                continue
            score = _finite_score(_unit_result(output, unit["unit_id"]))
            group = grouped.setdefault(
                candidate_id,
                {"candidate": candidate, "scores": [], "unit_ids": []},
            )
            group["scores"].append(score)
            group["unit_ids"].append(unit["unit_id"])
    ranked: list[dict[str, Any]] = []
    for candidate_id, group in grouped.items():
        scores = [float(value) for value in group["scores"]]
        unit_ids = [str(value) for value in group["unit_ids"]]
        best_index = min(range(len(scores)), key=lambda index: (scores[index], unit_ids[index]))
        ranked.append(
            {
                "candidate_id": candidate_id,
                "candidate": group["candidate"],
                "mean_selection_score": sum(scores) / len(scores),
                "selection_score_sd": float(statistics.pstdev(scores)),
                "material_repeats": len(scores),
                "source_unit_id": unit_ids[best_index],
                "member_unit_ids": unit_ids,
            }
        )
    return sorted(
        ranked,
        key=lambda row: (row["mean_selection_score"], row["candidate_id"]),
    )


def _resolve_selection(checkpoint: dict[str, Any], unit: dict[str, Any]) -> None:
    selection = unit["spec"].get("selection")
    if not isinstance(selection, dict) or "resolved_candidate" in unit["spec"]:
        # The uncertainty unit consumes every repeated full-OOF member.
        member_stage = unit["spec"].get("member_source_stage")
        if member_stage and "member_unit_ids" not in unit["spec"]:
            unit["spec"]["member_unit_ids"] = [
                source["unit_id"]
                for source in _stage_units(checkpoint, str(member_stage))
                if source["status"] == "passed"
            ]
        return
    stages = selection.get("source_stages")
    if not isinstance(stages, list):
        stages = [str(selection["source_stage"])]
    ranking = _rank_stages(checkpoint, [str(stage) for stage in stages])
    minimum_repeats = int(selection.get("minimum_material_repeats", 1))
    ranking = [choice for choice in ranking if int(choice["material_repeats"]) >= minimum_repeats]
    rank = int(selection["rank"])
    if rank >= len(ranking):
        raise CampaignError(
            f"{unit['unit_id']}: rank {rank} unavailable from {stages} ({len(ranking)} candidates)"
        )
    choice = ranking[rank]
    unit["spec"]["resolved_candidate"] = choice["candidate"]
    unit["spec"]["candidate"] = choice["candidate"]
    unit["spec"]["source_unit_id"] = choice["source_unit_id"]
    unit["spec"]["selection_evidence"] = {
        key: choice[key]
        for key in (
            "candidate_id",
            "mean_selection_score",
            "selection_score_sd",
            "material_repeats",
            "member_unit_ids",
        )
    }


def _command(checkpoint: dict[str, Any], unit: dict[str, Any]) -> list[str]:
    worker = checkpoint["worker_path"]
    python = checkpoint["python_path"]
    output = Path(checkpoint["output_root"])
    common = [
        "--repo-root",
        checkpoint["repo_root"],
        "--base-campaign-root",
        checkpoint["base_campaign_root"],
    ]
    if unit["operation"] == "prepare":
        return [
            python,
            worker,
            "prepare",
            *common,
            "--output-root",
            str(output / "prepared"),
            "--workers",
            str(checkpoint["workers"]),
        ]
    if unit["operation"] == "analyze":
        return [
            python,
            worker,
            "analyze",
            *common,
            "--prepared-root",
            str(output / "prepared"),
            "--results-root",
            str(output),
            "--output-root",
            str(output / "analysis"),
            "--workers",
            str(checkpoint["workers"]),
        ]
    return [
        python,
        worker,
        "run-unit",
        *common,
        "--prepared-root",
        str(output / "prepared"),
        "--output-root",
        str(output),
        "--unit-id",
        unit["unit_id"],
        "--unit-spec",
        json.dumps(unit["spec"], sort_keys=True, separators=(",", ":"), allow_nan=False),
        "--workers",
        str(checkpoint["workers"]),
    ]


def _bounded_artifact(output: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = output / path
    path = path.resolve()
    if not path.is_relative_to(output.resolve()):
        raise CampaignError(f"artifact escapes v3 output root: {path}")
    if not path.is_file():
        raise CampaignError(f"missing artifact: {path}")
    return path


def _verify_artifact_binding(output: Path, binding: dict[str, Any]) -> None:
    path = _bounded_artifact(output, str(binding["path"]))
    if "bytes" in binding and path.stat().st_size != int(binding["bytes"]):
        raise CampaignError(f"artifact size mismatch: {path}")
    if "sha256" in binding and _sha256(path) != binding["sha256"]:
        raise CampaignError(f"artifact hash mismatch: {path}")


def _validate_scientific_scope(scope: Any, operation: str) -> None:
    if not isinstance(scope, dict) or scope.get("source_partition") != "train":
        raise CampaignError("worker result is not bound to train partition")
    if scope.get("repository_validation_labels_opened") is not False:
        raise CampaignError("worker did not affirm validation-label blindness")
    if scope.get("repository_test_labels_opened") is not False:
        raise CampaignError("worker did not affirm test-label blindness")
    if operation == "broad_wt_auxiliary" and scope.get("broad_fixed_dose_pooled_into_pic50") is not False:
        raise CampaignError("broad fixed-dose labels were not kept separate from pIC50")


def _validate_completed_unit(output: Path, unit: dict[str, Any]) -> None:
    if unit["operation"] == "prepare":
        for name in ("validation.json", "manifest.json"):
            if not (output / "prepared" / name).is_file():
                raise CampaignError(f"prepare did not create {name}")
        return
    if unit["operation"] == "analyze":
        missing = [name for name in ANALYSIS_FILES if not (output / "analysis" / name).is_file()]
        if missing:
            raise CampaignError(f"analysis missing required files: {', '.join(missing)}")
        return
    result = _unit_result(output, unit["unit_id"])
    if result.get("status") != "passed" or result.get("unit_id") != unit["unit_id"]:
        raise CampaignError(f"invalid unit result identity/status: {unit['unit_id']}")
    if result.get("operation") != unit["operation"]:
        raise CampaignError(f"operation mismatch: {unit['unit_id']}")
    if result.get("unit_spec") != unit["spec"]:
        raise CampaignError(f"unit specification mismatch: {unit['unit_id']}")
    if result.get("unit_spec_sha256") != _worker_hash(unit["spec"]):
        raise CampaignError(f"unit specification hash mismatch: {unit['unit_id']}")
    executed_spec = result.get("executed_spec")
    if not isinstance(executed_spec, dict):
        raise CampaignError(f"missing executed specification: {unit['unit_id']}")
    if result.get("executed_spec_sha256") != _worker_hash(executed_spec):
        raise CampaignError(f"executed specification hash mismatch: {unit['unit_id']}")
    _validate_scientific_scope(result.get("scientific_scope"), unit["operation"])
    if unit["operation"] not in {"assay_quality_strata", "mmp_cliff_residual"}:
        _finite_score(result)
    artifacts = result.get("artifacts", [])
    if not isinstance(artifacts, list):
        raise CampaignError(f"invalid artifact ledger: {unit['unit_id']}")
    for binding in artifacts:
        if not isinstance(binding, dict):
            raise CampaignError(f"invalid artifact binding: {unit['unit_id']}")
        _verify_artifact_binding(output, binding)


def _binding_by_role(bindings: list[dict[str, Any]], role: str) -> dict[str, Any]:
    matches = [binding for binding in bindings if binding.get("role") == role]
    if len(matches) != 1:
        raise CampaignError(f"expected exactly one {role} implementation binding")
    return matches[0]


def _recovery_log_path(output: Path, unit: dict[str, Any], attempt: int) -> Path:
    cycle = int(unit.get("recovery_attempt_cycle", 0))
    suffix = f".recovery_{cycle}" if cycle else ""
    return output / "logs" / f"{unit['unit_id']}{suffix}.attempt_{attempt}.log"


def _recovery_migration(output: Path, worker_path: Path, confirmation: str) -> dict[str, Any]:
    """Transactionally recover the single observed v3 hash-contract failure.

    This is intentionally not a general checkpoint editor.  It refuses to run
    unless the checkpoint and every affected artifact exactly match the
    audited August 12 failure signature.
    """
    if confirmation != RECOVERY_MIGRATION_ID:
        raise CampaignError("recovery confirmation does not match the governed migration id")
    output = output.resolve()
    checkpoint_path = output / "checkpoint.json"
    with _Lock(output / ".campaign.lock"):
        checkpoint = _read_json(checkpoint_path, "checkpoint_sha256")
        existing = [
            record
            for record in checkpoint.get("migrations", [])
            if isinstance(record, dict) and record.get("migration_id") == RECOVERY_MIGRATION_ID
        ]
        if existing:
            _verify_bindings(checkpoint["bindings"])
            snapshot_binding = existing[-1].get("pre_checkpoint_snapshot_binding")
            if not isinstance(snapshot_binding, dict):
                raise CampaignError("migration lacks immutable pre-checkpoint snapshot binding")
            _verify_bindings([snapshot_binding])
            return {
                "status": "already_applied",
                "migration_id": RECOVERY_MIGRATION_ID,
                "checkpoint_sha256": checkpoint["checkpoint_sha256"],
                "adopted_units": int(existing[-1]["adopted_unit_count"]),
                "active_elapsed_seconds": checkpoint["active_elapsed_seconds"],
            }

        if checkpoint.get("schema_version") != SCHEMA_VERSION:
            raise CampaignError("recovery checkpoint schema mismatch")
        if checkpoint.get("status") != "incomplete" or len(checkpoint.get("units", [])) != 171:
            raise CampaignError("checkpoint does not match the observed incomplete campaign")
        status_counts = collections.Counter(str(unit.get("status")) for unit in checkpoint["units"])
        if dict(status_counts) != RECOVERY_EXPECTED_STATUS_COUNTS:
            raise CampaignError(f"unexpected pre-recovery status counts: {dict(status_counts)}")

        implementation_roles = set(RECOVERY_OLD_IMPLEMENTATION_HASHES)
        for binding in checkpoint["bindings"]:
            role = str(binding.get("role"))
            if role in implementation_roles:
                if binding.get("sha256") != RECOVERY_OLD_IMPLEMENTATION_HASHES[role]:
                    raise CampaignError(f"unexpected historical {role} binding")
                continue
            _verify_bindings([binding])
        old_implementation_bindings = [
            copy.deepcopy(_binding_by_role(checkpoint["bindings"], role))
            for role in sorted(implementation_roles)
        ]
        expected_orchestrator_path = Path(
            str(_binding_by_role(checkpoint["bindings"], "v3_orchestrator")["path"])
        ).resolve()
        if expected_orchestrator_path != Path(__file__).resolve():
            raise CampaignError("historical orchestrator path does not identify this migration code")
        expected_worker_path = Path(str(checkpoint["worker_path"])).resolve()
        if worker_path.resolve() != expected_worker_path:
            raise CampaignError("recovery worker path differs from the checkpoint")

        expected_stage_counts = collections.Counter(RECOVERY_EXPECTED_STAGE_COUNTS)
        adopted_stage_counts: collections.Counter[str] = collections.Counter()
        adopted: list[str] = []
        verified_artifact_bindings = 0
        allowed_stages = set(RECOVERY_EXPECTED_STAGE_COUNTS)
        unit_document_ids = {path.parent.name for path in (output / "units").glob("*/unit.json")}
        for unit in checkpoint["units"]:
            if unit["stage"] not in allowed_stages:
                continue
            if unit.get("status") != "failed_noncritical" or int(unit.get("attempts", -1)) != 2:
                raise CampaignError(f"unexpected recoverable unit state: {unit['unit_id']}")
            attempt_results = unit.get("attempt_results")
            if not isinstance(attempt_results, list) or len(attempt_results) != 2:
                raise CampaignError(f"unexpected attempt history: {unit['unit_id']}")
            if any(
                result.get("artifact_validation_error") != "unit_json_sha256 mismatch"
                for result in attempt_results
            ):
                raise CampaignError(f"unit did not fail solely at the known hash gate: {unit['unit_id']}")
            _validate_completed_unit(output, unit)
            result = _unit_result(output, unit["unit_id"])
            verified_artifact_bindings += len(result.get("artifacts", []))
            adopted.append(unit["unit_id"])
            adopted_stage_counts[unit["stage"]] += 1
        if adopted_stage_counts != expected_stage_counts or set(adopted) != unit_document_ids:
            raise CampaignError(
                "recoverable unit set differs from the 120 coarse plus four broad audited artifacts"
            )
        if len(adopted) != 124 or verified_artifact_bindings != 376:
            raise CampaignError("recovery artifact cardinality mismatch")

        censored = next(
            (unit for unit in checkpoint["units"] if unit["unit_id"] == RECOVERY_CENSORED_UNIT_ID),
            None,
        )
        if not isinstance(censored, dict):
            raise CampaignError("censored recovery unit is missing")
        censored_artifact = output / "units" / RECOVERY_CENSORED_UNIT_ID / "unit.json"
        if censored_artifact.exists():
            raise CampaignError("refusing to reset censored unit because an artifact exists")
        if censored.get("status") != "failed_noncritical" or int(censored.get("attempts", -1)) != 2:
            raise CampaignError("censored recovery unit state differs from the audited failure")
        censored_results = censored.get("attempt_results")
        if not isinstance(censored_results, list) or len(censored_results) != 2:
            raise CampaignError("censored recovery attempt history is incomplete")
        censored_logs = [Path(str(result.get("log_path", ""))) for result in censored_results]
        if not all(
            path.is_file() and "assignment destination is read-only" in path.read_text()
            for path in censored_logs
        ):
            raise CampaignError("censored unit logs do not match the audited writability failure")

        pre_hash = str(checkpoint["checkpoint_sha256"])
        pre_active_seconds = float(checkpoint["active_elapsed_seconds"])
        snapshot_path = output / "recovery" / f"checkpoint.pre_{pre_hash}.json"
        snapshot_binding = _immutable_snapshot(checkpoint_path, snapshot_path)
        snapshot_document = _read_json(snapshot_path, "checkpoint_sha256")
        if snapshot_document["checkpoint_sha256"] != pre_hash:
            raise CampaignError("immutable snapshot does not contain the pre-migration checkpoint")
        new_implementation_bindings = {
            "v3_orchestrator": _binding(Path(__file__), "v3_orchestrator"),
            "v3_worker": _binding(worker_path, "v3_worker"),
        }
        checkpoint["bindings"] = [
            new_implementation_bindings.get(str(binding.get("role")), binding)
            for binding in checkpoint["bindings"]
        ]
        for unit in checkpoint["units"]:
            if unit["unit_id"] in set(adopted):
                unit["status"] = "passed"
                unit["recovered_worker_artifact"] = {
                    "migration_id": RECOVERY_MIGRATION_ID,
                    "previous_status": "failed_noncritical",
                    "attempts_preserved": int(unit["attempts"]),
                    "reason": "orchestrator omitted worker canonical trailing newline",
                }
        censored["status"] = "pending"
        censored["attempts"] = 0
        censored["recovery_attempt_cycle"] = int(censored.get("recovery_attempt_cycle", 0)) + 1
        censored["recovered_for_retry"] = {
            "migration_id": RECOVERY_MIGRATION_ID,
            "previous_status": "failed_noncritical",
            "previous_attempts": 2,
            "previous_attempt_results_preserved": len(censored_results),
            "reason": "worker arrays were read-only; implementation now copies writable arrays",
        }
        checkpoint["status"] = "recovered_ready_to_resume"
        checkpoint.pop("incomplete_reasons", None)
        migration = _self_hashed(
            {
                "migration_id": RECOVERY_MIGRATION_ID,
                "applied_utc": _utc(),
                "pre_checkpoint_sha256": pre_hash,
                "pre_checkpoint_snapshot_binding": snapshot_binding,
                "pre_active_elapsed_seconds": pre_active_seconds,
                "post_active_elapsed_seconds": pre_active_seconds,
                "adopted_unit_count": len(adopted),
                "adopted_unit_ids": adopted,
                "adopted_stage_counts": dict(adopted_stage_counts),
                "verified_artifact_bindings": verified_artifact_bindings,
                "censored_unit_reset": RECOVERY_CENSORED_UNIT_ID,
                "old_implementation_bindings": old_implementation_bindings,
                "new_implementation_bindings": [
                    new_implementation_bindings[role] for role in sorted(new_implementation_bindings)
                ],
                "locked_repository_labels_opened": False,
            },
            "migration_sha256",
        )
        checkpoint.setdefault("migrations", []).append(migration)
        migrated = _checkpoint(output, checkpoint)
        if float(migrated["active_elapsed_seconds"]) != pre_active_seconds:
            raise CampaignError("migration changed cumulative active time")
        _verify_bindings(migrated["bindings"])
        return {
            "status": "migration_applied",
            "migration_id": RECOVERY_MIGRATION_ID,
            "checkpoint_sha256": migrated["checkpoint_sha256"],
            "adopted_units": len(adopted),
            "verified_artifact_bindings": verified_artifact_bindings,
            "censored_unit_reset": RECOVERY_CENSORED_UNIT_ID,
            "active_elapsed_seconds": migrated["active_elapsed_seconds"],
            "next_step": "rerun the identical governed v3 launcher",
        }


def _run_process(
    command: list[str], log_path: Path, environment: dict[str, str], deadline_epoch: float
) -> tuple[int, bool, float]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    timed_out = False
    with log_path.open("a", encoding="utf-8") as log:
        log.write(json.dumps({"started_utc": _utc(), "argv": command}) + "\n")
        log.flush()
        process = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            env=environment,
            start_new_session=True,
        )
        while process.poll() is None:
            if time.time() >= deadline_epoch:
                timed_out = True
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                break
            time.sleep(1)
        returncode = process.wait()
        elapsed = time.monotonic() - started
        log.write(
            json.dumps(
                {
                    "finished_utc": _utc(),
                    "returncode": returncode,
                    "timed_out": timed_out,
                    "elapsed_seconds": elapsed,
                }
            )
            + "\n"
        )
    return returncode, timed_out, elapsed


def _run_unit_with_retry(
    command: list[str],
    unit: dict[str, Any],
    output: Path,
    checkpoint: dict[str, Any],
    environment: dict[str, str],
    deadline_epoch: float,
) -> tuple[int, bool, float, Path]:
    if int(unit["attempts"]) >= MAX_ATTEMPTS:
        raise CampaignError(f"attempt budget already exhausted: {unit['unit_id']}")
    elapsed_total = 0.0
    last_returncode = 2
    last_timeout = False
    last_log = _recovery_log_path(output, unit, int(unit["attempts"]) + 1)
    while int(unit["attempts"]) < MAX_ATTEMPTS:
        unit["attempts"] = int(unit["attempts"]) + 1
        attempt = int(unit["attempts"])
        last_log = _recovery_log_path(output, unit, attempt)
        unit.setdefault("attempt_logs", []).append(str(last_log))
        unit["log_path"] = str(last_log)
        unit["started_utc"] = _utc()
        _checkpoint(output, checkpoint)
        last_returncode, last_timeout, elapsed = _run_process(command, last_log, environment, deadline_epoch)
        elapsed_total += elapsed
        checkpoint["active_elapsed_seconds"] += elapsed
        checkpoint["invocations"][-1]["active_elapsed_seconds"] += elapsed
        validation_error: str | None = None
        if last_returncode == 0:
            try:
                _validate_completed_unit(output, unit)
            except (CampaignError, OSError, ValueError, KeyError, json.JSONDecodeError) as error:
                last_returncode = 2
                validation_error = str(error)
        unit.setdefault("attempt_results", []).append(
            {
                "attempt": attempt,
                "returncode": last_returncode,
                "timed_out": last_timeout,
                "elapsed_seconds": elapsed,
                "log_path": str(last_log),
                "artifact_validation_error": validation_error,
                "recovery_attempt_cycle": int(unit.get("recovery_attempt_cycle", 0)),
            }
        )
        _checkpoint(output, checkpoint)
        if last_returncode in {0, 3} or time.time() >= deadline_epoch:
            break
    return last_returncode, last_timeout, elapsed_total, last_log


def _initial(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.repo_root).resolve()
    output = Path(args.output_root).resolve()
    base = Path(args.base_campaign_root).resolve()
    worker = Path(args.worker).resolve()
    broad = Path(args.broad_surface).resolve()
    bindings = [
        _binding(Path(__file__), "v3_orchestrator"),
        _binding(worker, "v3_worker"),
        _binding(base / "DONE.json", "completed_v2_done"),
        _binding(base / "final_summary.json", "completed_v2_summary"),
        _binding(base / "prepared" / "exact_train_cache.parquet", "exact_pic50_train_cache"),
        _binding(base / "prepared" / "nested_scaffold_splits.parquet", "nested_scaffold_splits"),
        _binding(base / "prepared" / "feature_registry.json", "feature_registry"),
        _binding(base / "analysis" / "validation.json", "v2_analysis_validation"),
        _binding(broad, "confirmed_wt_fixed_dose_surface"),
    ]
    plan = _plan()
    expected = sum(float(unit["expected_minutes"]) for unit in plan)
    if expected > args.hard_hours * 60 - args.finalization_reserve_minutes:
        raise CampaignError("prespecified nominal schedule exceeds hard work budget")
    return {
        "schema_version": SCHEMA_VERSION,
        "created_utc": _utc(),
        "updated_utc": _utc(),
        "status": "running",
        "repo_root": str(root),
        "output_root": str(output),
        "base_campaign_root": str(base),
        "broad_surface": str(broad),
        "worker_path": str(worker),
        # Preserve the virtual-environment symlink rather than resolving it.
        "python_path": str(Path(args.python).absolute()),
        "workers": args.workers,
        "target_hours": args.target_hours,
        "hard_hours": args.hard_hours,
        "reserve_minutes": args.finalization_reserve_minutes,
        "active_elapsed_seconds": 0.0,
        "nominal_expected_minutes": expected,
        "v2_empirical_runtime_hours": 4.1436,
        "duration_interpretation": (
            "Nominal values are conservative scheduling bounds. v2 completed in 4.14 h; "
            "v3 completes early when prespecified science finishes and never burns time artificially."
        ),
        "bindings": bindings,
        "invocations": [],
        "resource_checks": [],
        "stop_requests": [],
        "scientific_contract": {
            "exact_pic50_train_structures": 18801,
            "exact_target_scope": "wild_type_or_target_unspecified_not_confirmed_wt_only",
            "broad_full_surface_structures": 339373,
            "broad_train_structures_expected": 265625,
            "broad_target_scope": "confirmed_wild_type_fixed_dose_binary_auxiliary",
            "broad_fixed_dose_pooled_into_pic50": False,
            "source_partition": "train",
            "repository_validation_labels_opened": False,
            "repository_test_labels_opened": False,
            "fixed_validation_previously_examined_elsewhere": True,
            "locked_test_remains_sealed": True,
            "local_diagnostic_not_production_or_superiority_evidence": True,
        },
        "units": plan,
    }


def _resume(args: argparse.Namespace, output: Path) -> dict[str, Any]:
    checkpoint = _read_json(output / "checkpoint.json", "checkpoint_sha256")
    _verify_bindings(checkpoint["bindings"])
    expected = {
        "repo_root": str(Path(args.repo_root).resolve()),
        "output_root": str(output.resolve()),
        "base_campaign_root": str(Path(args.base_campaign_root).resolve()),
        "broad_surface": str(Path(args.broad_surface).resolve()),
        "worker_path": str(Path(args.worker).resolve()),
        "workers": args.workers,
    }
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise CampaignError(f"resume argument changed: {key}")
    for unit in checkpoint["units"]:
        if unit["status"] != "running":
            continue
        # A worker may have atomically completed its governed artifact just
        # before the orchestrator was interrupted. Validate and adopt that
        # result before spending or exhausting another attempt.
        try:
            _validate_completed_unit(output, unit)
        except (CampaignError, OSError, ValueError, KeyError, json.JSONDecodeError):
            pass
        else:
            unit["status"] = "passed"
            unit["recovered_completed_artifact_after_interruption"] = True
            continue
        if int(unit["attempts"]) < MAX_ATTEMPTS:
            unit["status"] = "pending"
            unit["interrupted_attempt_recovered"] = True
        else:
            unit["status"] = "failed_critical" if unit["critical"] else "failed_noncritical"
            unit["interrupted_after_attempt_budget_exhausted"] = True
    checkpoint["status"] = "running"
    return _checkpoint(output, checkpoint)


def _required_thresholds_met(checkpoint: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    for unit in checkpoint["units"]:
        if unit["required_for_completion"] and unit["status"] != "passed":
            reasons.append(f"required unit not passed: {unit['unit_id']}")
    thresholds = {
        "nested_confirmation": 5,
        "repeated_seed_confirmation": 4,
        "feature_ablation": 6,
        "interaction_stability": 1,
        "assay_quality_strata": 1,
        "uncertainty_calibration": 1,
        "mmp_cliff_residual": 1,
        "broad_wt_hpo": 3,
    }
    for stage, required in thresholds.items():
        passed = sum(unit["status"] == "passed" for unit in _stage_units(checkpoint, stage))
        if passed < required:
            reasons.append(f"stage {stage}: {passed}/{required} required units passed")
    return not reasons, reasons


def _finalist_roles(output: Path) -> dict[str, set[str]]:
    manifest = _read_json(output / "analysis" / "final_models_manifest.json")
    finalists = manifest.get("finalists")
    if not isinstance(finalists, list):
        raise CampaignError("final_models_manifest.json lacks finalists list")
    roles: dict[str, set[str]] = {}
    for finalist in finalists:
        if not isinstance(finalist, dict):
            raise CampaignError("invalid finalist record")
        operation = str(finalist.get("operation", ""))
        bindings = finalist.get("artifacts", [])
        if not isinstance(bindings, list):
            raise CampaignError("invalid finalist artifact ledger")
        for binding in bindings:
            if not isinstance(binding, dict):
                raise CampaignError("invalid finalist artifact binding")
            _verify_artifact_binding(output, binding)
            roles.setdefault(operation, set()).add(str(binding.get("role", "")))
    return roles


def _validate_terminal_artifacts(output: Path) -> None:
    for name in ANALYSIS_FILES:
        if not (output / "analysis" / name).is_file():
            raise CampaignError(f"missing terminal analysis artifact: {name}")
    validation = _read_json(output / "analysis" / "validation.json")
    if validation.get("status") != "passed":
        raise CampaignError("analysis validation did not pass")
    roles = _finalist_roles(output)
    exact_required = {"model", "feature_schema", "inference_smoke"}
    broad_required = exact_required | {"calibration", "enrichment"}
    if not exact_required <= roles.get("finalist_refit_artifact", set()):
        raise CampaignError("exact-pIC50 finalist lacks model/schema/inference smoke artifacts")
    if not broad_required <= roles.get("broad_wt_auxiliary", set()):
        raise CampaignError("broad fixed-dose finalist lacks model/schema/smoke/calibration/enrichment")


def _finalize(output: Path, checkpoint: dict[str, Any]) -> dict[str, Any]:
    complete, reasons = _required_thresholds_met(checkpoint)
    if complete:
        try:
            _validate_terminal_artifacts(output)
        except (CampaignError, OSError, ValueError, KeyError, json.JSONDecodeError) as error:
            complete = False
            reasons.append(str(error))
    if not complete:
        checkpoint["status"] = "incomplete"
        checkpoint["incomplete_reasons"] = reasons
        _checkpoint(output, checkpoint)
        return {"status": "incomplete", "reasons": reasons, "resume": "rerun identical launcher"}
    counts: dict[str, int] = {}
    failures: list[dict[str, Any]] = []
    for unit in checkpoint["units"]:
        counts[unit["status"]] = counts.get(unit["status"], 0) + 1
        if unit["status"].startswith("failed"):
            failures.append(
                {key: unit.get(key) for key in ("unit_id", "stage", "operation", "status", "log_path")}
            )
    checkpoint["status"] = "complete_with_noncritical_failures" if failures else "complete"
    checkpoint["finished_utc"] = _utc()
    checkpoint = _checkpoint(output, checkpoint)
    summary = _atomic_json(
        output / "final_summary.json",
        {
            "schema_version": SCHEMA_VERSION,
            "status": checkpoint["status"],
            "created_utc": _utc(),
            "active_elapsed_seconds": checkpoint["active_elapsed_seconds"],
            "completed_early_because_prespecified_science_finished": checkpoint["active_elapsed_seconds"]
            < checkpoint["target_hours"] * 3600,
            "unit_status_counts": counts,
            "failure_ledger": failures,
            "scientific_contract": checkpoint["scientific_contract"],
            "checkpoint_sha256": checkpoint["checkpoint_sha256"],
            "analysis_path": str(output / "analysis" / "analysis.md"),
        },
        "summary_sha256",
    )
    return _atomic_json(
        output / "DONE.json",
        {
            "schema_version": SCHEMA_VERSION,
            "status": checkpoint["status"],
            "finished_utc": _utc(),
            "checkpoint_sha256": checkpoint["checkpoint_sha256"],
            "summary_sha256": summary["summary_sha256"],
        },
        "done_sha256",
    )


def _start(args: argparse.Namespace) -> dict[str, Any]:
    if args.workers != DEFAULT_WORKERS:
        raise CampaignError("v3 campaign governance requires exactly six workers")
    if not 0 < args.target_hours <= args.hard_hours <= 30:
        raise CampaignError("require 0 < target-hours <= hard-hours <= 30")
    if not 0 < args.finalization_reserve_minutes < args.hard_hours * 60:
        raise CampaignError("invalid finalization reserve")
    output = Path(args.output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "checkpoint.json"
    if checkpoint_path.is_file():
        checkpoint_preview = _read_json(checkpoint_path, "checkpoint_sha256")
        already_migrated = any(
            isinstance(record, dict) and record.get("migration_id") == RECOVERY_MIGRATION_ID
            for record in checkpoint_preview.get("migrations", [])
        )
        historical_hashes = {
            str(binding.get("role")): str(binding.get("sha256"))
            for binding in checkpoint_preview.get("bindings", [])
            if binding.get("role") in RECOVERY_OLD_IMPLEMENTATION_HASHES
        }
        if (
            not already_migrated
            and checkpoint_preview.get("status") == "incomplete"
            and historical_hashes == RECOVERY_OLD_IMPLEMENTATION_HASHES
        ):
            _recovery_migration(output, Path(args.worker), RECOVERY_MIGRATION_ID)
    with _Lock(output / ".campaign.lock"):
        checkpoint = (
            _resume(args, output)
            if (output / "checkpoint.json").is_file()
            else _checkpoint(output, _initial(args))
        )
        if checkpoint["status"].startswith("complete"):
            return {"status": checkpoint["status"], "message": "v3 already complete"}
        hard_remaining = max(0.0, checkpoint["hard_hours"] * 3600 - checkpoint["active_elapsed_seconds"])
        work_remaining = max(
            0.0,
            checkpoint["hard_hours"] * 3600
            - checkpoint["reserve_minutes"] * 60
            - checkpoint["active_elapsed_seconds"],
        )
        now = time.time()
        invocation = {
            "invocation_index": len(checkpoint["invocations"]) + 1,
            "started_utc": _utc(),
            "active_elapsed_seconds_before": checkpoint["active_elapsed_seconds"],
            "active_elapsed_seconds": 0.0,
            "work_deadline_epoch": now + work_remaining,
            "hard_deadline_epoch": now + hard_remaining,
            "downtime_excluded_from_active_budget": True,
        }
        checkpoint["invocations"].append(invocation)
        _checkpoint(output, checkpoint)
        stop_requested = False

        def request_stop(signum: int, _frame: object) -> None:
            nonlocal stop_requested
            stop_requested = True
            checkpoint["stop_requests"].append(
                {"signal": signum, "received_utc": _utc(), "policy": "safe_stop_after_current_unit"}
            )

        old_handlers = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)}
        environment = os.environ.copy()
        environment.update(
            {
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONUNBUFFERED": "1",
                "OMP_NUM_THREADS": str(args.workers),
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "VECLIB_MAXIMUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
                "HERG_CAMPAIGN_FORBID_REPOSITORY_VALIDATION_TEST": "1",
            }
        )
        try:
            progress = True
            while progress:
                progress = False
                for unit in checkpoint["units"]:
                    if unit["status"] in TERMINAL:
                        continue
                    if stop_requested:
                        checkpoint["status"] = "safely_stopped"
                        _checkpoint(output, checkpoint)
                        return {"status": "safely_stopped", "resume": "rerun identical launcher"}
                    if not _dependencies_ready(checkpoint, unit):
                        unit["dependency_wait_count"] = int(unit.get("dependency_wait_count", 0)) + 1
                        continue
                    final = unit["operation"] == "analyze"
                    deadline = (
                        invocation["hard_deadline_epoch"] if final else invocation["work_deadline_epoch"]
                    )
                    remaining_minutes = (deadline - time.time()) / 60
                    if remaining_minutes <= 0 or (not final and unit["expected_minutes"] > remaining_minutes):
                        if unit["required_for_completion"]:
                            checkpoint["status"] = "safely_stopped_time_budget"
                            _checkpoint(output, checkpoint)
                            return {
                                "status": checkpoint["status"],
                                "unit": unit["unit_id"],
                                "resume": "no hard active-time budget remains",
                            }
                        unit["status"] = "skipped_time"
                        _checkpoint(output, checkpoint)
                        progress = True
                        continue
                    try:
                        gate = _resource_gate(
                            output, args.minimum_free_disk_gib, args.minimum_available_memory_gib
                        )
                    except CampaignError as error:
                        checkpoint["status"] = "safely_stopped_resource_gate"
                        checkpoint["resource_error"] = str(error)
                        _checkpoint(output, checkpoint)
                        return {"status": checkpoint["status"], "error": str(error)}
                    gate["before_unit"] = unit["unit_id"]
                    checkpoint["resource_checks"].append(gate)
                    try:
                        _resolve_selection(checkpoint, unit)
                        command = _command(checkpoint, unit)
                    except (CampaignError, KeyError, ValueError) as error:
                        unit["status"] = "failed_critical" if unit["critical"] else "failed_noncritical"
                        unit["error"] = str(error)
                        _checkpoint(output, checkpoint)
                        progress = True
                        continue
                    unit["status"] = "running"
                    _checkpoint(output, checkpoint)
                    returncode, timed_out, elapsed, log_path = _run_unit_with_retry(
                        command, unit, output, checkpoint, environment, deadline
                    )
                    unit.update(
                        returncode=returncode,
                        timed_out=timed_out,
                        elapsed_seconds=elapsed,
                        finished_utc=_utc(),
                        log_path=str(log_path),
                    )
                    if returncode == 0:
                        unit["status"] = "passed"
                    elif returncode == 3 and not unit["critical"]:
                        unit["status"] = "skipped_unavailable"
                    else:
                        unit["status"] = "failed_critical" if unit["critical"] else "failed_noncritical"
                    _checkpoint(output, checkpoint)
                    progress = True
                    if unit["critical"] and unit["status"] == "failed_critical":
                        checkpoint["status"] = "failed"
                        _checkpoint(output, checkpoint)
                        return {"status": "failed", "unit": unit["unit_id"], "log": str(log_path)}
            return _finalize(output, checkpoint)
        finally:
            for sig, handler in old_handlers.items():
                signal.signal(sig, handler)


def _status(output: Path) -> dict[str, Any]:
    checkpoint = _read_json(output.resolve() / "checkpoint.json", "checkpoint_sha256")
    counts: dict[str, int] = {}
    for unit in checkpoint["units"]:
        counts[unit["status"]] = counts.get(unit["status"], 0) + 1
    active = next((unit["unit_id"] for unit in checkpoint["units"] if unit["status"] == "running"), None)
    return {
        "status": checkpoint["status"],
        "active_unit": active,
        "unit_status_counts": counts,
        "active_elapsed_hours": checkpoint["active_elapsed_seconds"] / 3600,
        "hard_active_hours_remaining": max(
            0.0, checkpoint["hard_hours"] - checkpoint["active_elapsed_seconds"] / 3600
        ),
        "updated_utc": checkpoint["updated_utc"],
        "output_root": checkpoint["output_root"],
    }


def _validate(output: Path, require_done: bool = True) -> dict[str, Any]:
    output = output.resolve()
    checkpoint = _read_json(output / "checkpoint.json", "checkpoint_sha256")
    _verify_bindings(checkpoint["bindings"])
    if checkpoint["schema_version"] != SCHEMA_VERSION or checkpoint["workers"] != DEFAULT_WORKERS:
        raise CampaignError("schema/worker governance mismatch")
    contract = checkpoint["scientific_contract"]
    if contract["repository_validation_labels_opened"] or contract["repository_test_labels_opened"]:
        raise CampaignError("sealed-label contract failed")
    if contract["broad_fixed_dose_pooled_into_pic50"]:
        raise CampaignError("broad and quantitative tasks were pooled")
    for unit in checkpoint["units"]:
        if unit["status"] == "passed":
            if not Path(str(unit.get("log_path", ""))).is_file():
                raise CampaignError(f"missing log: {unit['unit_id']}")
            _validate_completed_unit(output, unit)
    if require_done:
        if checkpoint["status"] not in {"complete", "complete_with_noncritical_failures"}:
            raise CampaignError(f"campaign is not complete: {checkpoint['status']}")
        complete, reasons = _required_thresholds_met(checkpoint)
        if not complete:
            raise CampaignError("; ".join(reasons))
        _validate_terminal_artifacts(output)
        done = _read_json(output / "DONE.json", "done_sha256")
        summary = _read_json(output / "final_summary.json", "summary_sha256")
        if done["checkpoint_sha256"] != checkpoint["checkpoint_sha256"]:
            raise CampaignError("DONE does not bind final checkpoint")
        if done["summary_sha256"] != summary["summary_sha256"]:
            raise CampaignError("DONE does not bind final summary")
    return {
        "status": "passed",
        "campaign_status": checkpoint["status"],
        "units": len(checkpoint["units"]),
        "validation_labels_opened": False,
        "test_labels_opened": False,
        "broad_pooled_into_pic50": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    start = commands.add_parser("start")
    start.add_argument("--repo-root", default=".")
    start.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    start.add_argument("--base-campaign-root", default=str(DEFAULT_BASE))
    start.add_argument("--broad-surface", default=str(DEFAULT_BROAD))
    start.add_argument(
        "--worker", default=str(Path(__file__).with_name("run_local_herg_discovery_worker_v3.py"))
    )
    start.add_argument("--python", default=sys.executable)
    start.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    start.add_argument("--target-hours", type=float, default=24.0)
    start.add_argument("--hard-hours", type=float, default=30.0)
    start.add_argument("--finalization-reserve-minutes", type=float, default=60.0)
    start.add_argument("--minimum-free-disk-gib", type=float, default=15.0)
    start.add_argument("--minimum-available-memory-gib", type=float, default=1.5)
    for name in ("status", "validate"):
        command = commands.add_parser(name)
        command.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    commands.choices["validate"].add_argument("--allow-incomplete", action="store_true")
    recover = commands.add_parser("recover-production")
    recover.add_argument("--output-root", required=True)
    recover.add_argument("--worker", required=True)
    recover.add_argument("--confirm", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        if args.action == "start":
            result = _start(args)
        elif args.action == "status":
            result = _status(Path(args.output_root))
        elif args.action == "recover-production":
            result = _recovery_migration(Path(args.output_root), Path(args.worker), str(args.confirm))
        else:
            result = _validate(Path(args.output_root), require_done=not args.allow_incomplete)
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except (
        CampaignError,
        OSError,
        ValueError,
        KeyError,
        json.JSONDecodeError,
        subprocess.SubprocessError,
    ) as error:
        print(json.dumps({"status": "failed", "error": str(error)}, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
