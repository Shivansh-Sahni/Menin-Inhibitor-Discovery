#!/usr/bin/env python3
"""Run the governed, resumable, train-only hERG discovery campaign v4.

The scientific plan is owned by a versioned YAML file.  This module owns only
orchestration: immutable input and implementation bindings, strict validation
against the worker's machine-readable capabilities, resource and active-time
budgets, retry/resume/adoption, per-unit isolation, and terminal deliverable
verification.  Repository validation and test labels are forbidden inputs.
"""

from __future__ import annotations

import argparse
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
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Any

import yaml

SCHEMA_VERSION = "platform-local-herg-discovery-campaign-v4/1.0"
CONFIG_SCHEMA_VERSION = "platform-herg-discovery-v4-config/1.0"
CAPABILITIES_SCHEMA_VERSION = "platform-local-herg-discovery-worker-v4-capabilities/1.0"
DEFAULT_OUTPUT = Path("research/local_runs/herg_discovery_campaign_v4")
DEFAULT_CONFIG = Path("pipeline/config/herg_discovery_campaign_v4.yaml")
DEFAULT_PROTOCOL = Path("research/reports/platform/herg_paper/v4/HERG_V4_PREREGISTERED_PROTOCOL.md")
DEFAULT_V2 = Path("research/local_runs/herg_discovery_campaign_v1")
DEFAULT_V3 = Path("research/local_runs/herg_discovery_campaign_v3")
DEFAULT_WORKERS = 6
MAX_ATTEMPTS = 2
TERMINAL = {
    "passed",
    "failed_noncritical",
    "failed_critical",
    "skipped_unavailable",
    "skipped_time",
}
TOP_LEVEL_CONFIG_KEYS = {
    "schema_version",
    "campaign",
    "scientific_contract",
    "inputs",
    "plan",
    "completion",
    "empirical_basis",
}
EMPIRICAL_BASIS_KEYS = {
    "v2_active_hours",
    "v3_active_hours",
    "chemprop_five_fold_40_epoch_minutes",
    "nominal_useful_minutes",
    "estimate_method",
}
CAMPAIGN_KEYS = {
    "workers",
    "target_active_hours",
    "hard_active_hours",
    "finalization_reserve_minutes",
    "minimum_free_disk_gib",
    "minimum_available_memory_gib",
    "maximum_output_gib",
}
UNIT_KEYS = {
    "unit_id",
    "stage",
    "operation",
    "expected_minutes",
    "critical",
    "required_for_completion",
    "priority_tier",
    "dependencies",
    "spec",
}
DEPENDENCY_KEYS = {"stage", "minimum_passed", "require_terminal"}
COMPLETION_KEYS = {
    "stage_minimum_passed",
    "required_artifact_roles",
    "exact_nested_rows",
}
MINIMUM_COMPLETION_ARTIFACT_ROLES = {
    "classical_hpo": {"oof_predictions", "feature_importance", "feature_schema"},
    "chemprop_hpo": {"oof_predictions", "model_chemprop_fold", "chemprop_command"},
    "nested_outer_evaluation": {"oof_predictions"},
    "aggregate_nested_oof": {"oof_predictions"},
    "microstate_conformer": {"fresh_conformer_features", "conformer_checkpoint_shard"},
}


class CampaignError(RuntimeError):
    """Raised when campaign governance or reproducibility validation fails."""


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical(value: Any, *, newline: bool = False) -> bytes:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return data + (b"\n" if newline else b"")


def _hash_value(value: Any, *, worker: bool = False) -> str:
    return hashlib.sha256(_canonical(value, newline=worker)).hexdigest()


def _self_hashed(value: dict[str, Any], key: str) -> dict[str, Any]:
    result = copy.deepcopy(value)
    result.pop(key, None)
    result[key] = _hash_value(result)
    return result


def _verify_self_hash(value: dict[str, Any], key: str) -> None:
    if value.get(key) != _self_hashed(value, key)[key]:
        raise CampaignError(f"{key} mismatch")


def _atomic_json(path: Path, value: dict[str, Any], hash_key: str) -> dict[str, Any]:
    document = _self_hashed(value, hash_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return document


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _read_json(path: Path, hash_key: str | None = None) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CampaignError(f"expected JSON object: {path}")
    if hash_key:
        _verify_self_hash(value, hash_key)
    return value


def _read_worker_json(path: Path, hash_key: str = "unit_json_sha256") -> dict[str, Any]:
    value = _read_json(path)
    expected = value.get(hash_key)
    unhashed = dict(value)
    unhashed.pop(hash_key, None)
    if not isinstance(expected, str) or expected != _hash_value(unhashed, worker=True):
        raise CampaignError(f"{hash_key} mismatch: {path}")
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
        raise CampaignError(f"missing bound file for {role}: {path}")
    return {
        "role": role,
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _verify_bindings(bindings: list[dict[str, Any]]) -> None:
    for binding in bindings:
        path = Path(str(binding["path"]))
        if not path.is_file() or path.stat().st_size != int(binding["bytes"]):
            raise CampaignError(f"bound file missing or changed: {path}")
        if _sha256(path) != binding["sha256"]:
            raise CampaignError(f"bound file hash changed: {path}")


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


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CampaignError("v4 config must be a YAML mapping")
    return value


def _worker_capabilities(python: Path, worker: Path) -> dict[str, Any]:
    result = subprocess.run(
        [str(python), str(worker), "capabilities"],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    if result.returncode != 0:
        raise CampaignError(f"worker capabilities failed: {result.stderr.strip()}")
    try:
        capabilities = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise CampaignError("worker capabilities did not return one JSON object") from error
    if not isinstance(capabilities, dict):
        raise CampaignError("worker capabilities must be a JSON object")
    if capabilities.get("schema_version") != CAPABILITIES_SCHEMA_VERSION:
        raise CampaignError("worker capabilities schema mismatch")
    if capabilities.get("unit_document_hash_contract") != "compact_sorted_json_plus_newline":
        raise CampaignError("unsupported worker unit hash contract")
    if not isinstance(capabilities.get("operations"), dict):
        raise CampaignError("worker capabilities lack operations")
    return capabilities


def _configured_input_bindings(config: dict[str, Any], repo_root: Path) -> list[dict[str, Any]]:
    raw = config.get("inputs")
    records: list[dict[str, Any]] = []
    if isinstance(raw, dict):
        iterable = [{"role": role, "path": path} for role, path in raw.items()]
    elif isinstance(raw, list):
        iterable = raw
    else:
        raise CampaignError("config inputs must map roles to file paths or be a role/path list")
    roles: set[str] = set()
    for record in iterable:
        if not isinstance(record, dict) or set(record) != {"role", "path"}:
            raise CampaignError("every configured input must contain exactly role and path")
        role = str(record["role"])
        if not role or role in roles:
            raise CampaignError(f"duplicate/empty configured input role: {role}")
        roles.add(role)
        path = Path(str(record["path"]))
        if not path.is_absolute():
            path = repo_root / path
        records.append(_binding(path, role))
    return records


def _implementation_import_bindings(worker: Path) -> list[dict[str, Any]]:
    """Bind local versioned worker modules imported by the v4 implementation."""
    required = [worker.with_name("run_local_herg_discovery_worker_v3.py")]
    return [_binding(path, f"imported_implementation_{path.stem}") for path in required]


def _validate_capability_spec(unit: dict[str, Any], capabilities: dict[str, Any]) -> tuple[str, bool]:
    operation = str(unit["operation"])
    capability = capabilities["operations"].get(operation)
    if not isinstance(capability, dict):
        raise CampaignError(f"worker does not support operation: {operation}")
    allowed = set(capability.get("allowed_spec_keys", []))
    required = set(capability.get("required_spec_keys", []))
    if not allowed or not all(isinstance(key, str) for key in allowed | required):
        raise CampaignError(f"invalid capability declaration for {operation}")
    spec = unit["spec"]
    unknown = set(spec) - allowed
    missing = required - set(spec)
    if unknown:
        raise CampaignError(f"{unit['unit_id']}: unsupported spec keys: {sorted(unknown)}")
    if missing:
        raise CampaignError(f"{unit['unit_id']}: missing spec keys: {sorted(missing)}")
    candidate = spec.get("candidate")
    if isinstance(candidate, dict):
        engines = capability.get("engines")
        feature_sets = capability.get("feature_sets")
        if isinstance(engines, list) and candidate.get("engine") not in engines:
            raise CampaignError(f"{unit['unit_id']}: unsupported engine")
        if isinstance(feature_sets, list) and candidate.get("feature_set") not in feature_sets:
            raise CampaignError(f"{unit['unit_id']}: unsupported feature set")
    return operation, bool(capability.get("score_required", False))


def _validate_dependency_sources(units: list[dict[str, Any]]) -> None:
    """Require fixed upstream sources and inner-only nested selection evidence."""
    by_id = {unit["unit_id"]: unit for unit in units}
    positions = {unit["unit_id"]: index for index, unit in enumerate(units)}
    for position, unit in enumerate(units):
        spec = unit["spec"]
        source_ids: list[str] = []
        for key in ("source_unit_id", "broad_source_unit_id", "exact_source_unit_id"):
            if key in spec:
                source_ids.append(str(spec[key]))
        for key in (
            "source_unit_ids",
            "member_unit_ids",
            "selection_source_unit_ids",
            "broad_selection_source_unit_ids",
            "exact_selection_source_unit_ids",
        ):
            if key not in spec:
                continue
            value = spec[key]
            if not isinstance(value, list):
                raise CampaignError(f"{unit['unit_id']}: {key} must be a list")
            source_ids.extend(str(item) for item in value)
        for source_id in source_ids:
            if source_id not in by_id:
                raise CampaignError(f"{unit['unit_id']}: unknown source unit {source_id}")
            if positions[source_id] >= position:
                raise CampaignError(f"{unit['unit_id']}: source is not upstream: {source_id}")
        if unit["operation"] in {"nested_outer_evaluation", "nested_stack"}:
            outer = int(spec["outer_fold"])
            selection_ids = spec.get("selection_source_unit_ids", [])
            if not isinstance(selection_ids, list) or not selection_ids:
                raise CampaignError(f"{unit['unit_id']}: no fixed inner selection sources")
            for source_id in selection_ids:
                if by_id[str(source_id)]["operation"] != "classical_hpo":
                    raise CampaignError(
                        f"{unit['unit_id']}: only classical HPO may drive adaptive outer selection"
                    )
                source_spec = by_id[str(source_id)]["spec"]
                if source_spec.get("evaluation_stage") != "inner":
                    raise CampaignError(f"{unit['unit_id']}: selection source is not inner-only")
                if int(source_spec.get("outer_fold", -1)) != outer:
                    raise CampaignError(f"{unit['unit_id']}: selection source outer context mismatch")
                if list(source_spec.get("inner_folds", [])) != [0, 1, 2]:
                    raise CampaignError(f"{unit['unit_id']}: selection source lacks all inner folds")
                if float(source_spec.get("budget_fraction", 0.0)) != 1.0:
                    raise CampaignError(f"{unit['unit_id']}: selection source is not full budget")


def _validate_scientific_plan_shape(units: list[dict[str, Any]]) -> None:
    """Require a material compute-first plan without padding or optional branches."""
    operations = Counter(unit["operation"] for unit in units)
    mandatory = {
        "prepare": 1,
        "baseline_reproduction": 2,
        "classical_hpo": 30,
        "chemprop_hpo": 3,
        "nested_outer_evaluation": 5,
        "aggregate_nested_oof": 1,
        "microstate_conformer": 2,
    }
    for operation, minimum in mandatory.items():
        if operations[operation] < minimum:
            raise CampaignError(f"scientific plan lacks {minimum} material {operation} units")
    outer = {
        int(unit["spec"]["outer_fold"]) for unit in units if unit["operation"] == "nested_outer_evaluation"
    }
    if outer != set(range(5)):
        raise CampaignError("nested outer evaluation does not cover folds 0..4")
    classical = [unit for unit in units if unit["operation"] == "classical_hpo"]
    candidate_sets: dict[int, set[str]] = {}
    engines: set[str] = set()
    feature_sets: set[str] = set()
    for unit in classical:
        spec = unit["spec"]
        if (
            spec.get("evaluation_stage") != "inner"
            or list(spec.get("inner_folds", [])) != [0, 1, 2]
            or float(spec.get("budget_fraction", 0.0)) != 1.0
        ):
            raise CampaignError("classical HPO must execute all three inner folds at full budget")
        candidate = spec.get("candidate", {})
        outer_fold = int(spec.get("outer_fold", -1))
        candidate_sets.setdefault(outer_fold, set()).add(str(candidate.get("candidate_id")))
        engines.add(str(candidate.get("engine")))
        feature_sets.add(str(candidate.get("feature_set")))
    if set(candidate_sets) != set(range(5)) or any(len(value) < 6 for value in candidate_sets.values()):
        raise CampaignError("classical HPO requires at least six candidates in every outer context")
    if len(engines) < 3 or len(feature_sets) < 3:
        raise CampaignError("classical HPO lacks material engine/feature-family diversity")
    for unit in units:
        if unit["operation"] == "chemprop_hpo":
            spec = unit["spec"]
            if spec.get("evaluation_stage") != "inner" or list(spec.get("inner_folds", [])) != [0, 1, 2]:
                raise CampaignError("Chemprop is restricted to standalone three-inner-fold evidence")
    disallowed = {
        "nested_stack",
        "assay_hierarchical",
        "censored_interval",
        "broad_auxiliary",
        "broad_transfer",
        "mmp_cliff",
        "uncertainty_ad",
        "receptor_pilot",
        "final_refit_exact",
        "final_refit_broad",
        "analyze",
    }
    present = sorted(disallowed & set(operations))
    if present:
        raise CampaignError(f"15-hour compute-only plan contains deferred operations: {present}")
    conformers = [unit["spec"] for unit in units if unit["operation"] == "microstate_conformer"]
    if not any(
        int(spec.get("requested_conformers", 0)) >= 24 and int(spec.get("panel_size", 0)) == 18801
        for spec in conformers
    ):
        raise CampaignError("plan lacks fresh >=24-conformer whole-exact-surface unit")
    if not any(
        int(spec.get("requested_conformers", 0)) >= 50 and 100 <= int(spec.get("panel_size", 0)) <= 500
        for spec in conformers
    ):
        raise CampaignError("plan lacks bounded >=50-conformer convergence panel")


def _validate_config(
    config_path: Path,
    repo_root: Path,
    capabilities: dict[str, Any],
) -> dict[str, Any]:
    config = _load_yaml(config_path)
    unknown_top = set(config) - TOP_LEVEL_CONFIG_KEYS
    if unknown_top:
        raise CampaignError(f"unknown top-level config keys: {sorted(unknown_top)}")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise CampaignError("v4 config schema mismatch")
    campaign = config.get("campaign")
    if not isinstance(campaign, dict) or set(campaign) != CAMPAIGN_KEYS:
        raise CampaignError("campaign config keys do not match the governed schema")
    if int(campaign["workers"]) != DEFAULT_WORKERS:
        raise CampaignError("v4 governance requires exactly six workers")
    target = float(campaign["target_active_hours"])
    hard = float(campaign["hard_active_hours"])
    reserve = float(campaign["finalization_reserve_minutes"])
    if target != 13.5 or hard != 15.0 or reserve != 60.0:
        raise CampaignError("v4 compute run requires target=13.5h, hard=15h, reserve=60m")
    empirical = config.get("empirical_basis")
    if not isinstance(empirical, dict) or set(empirical) != EMPIRICAL_BASIS_KEYS:
        raise CampaignError("empirical_basis keys do not match the governed schema")
    expected_empirical = {
        "v2_active_hours": 4.1436,
        "v3_active_hours": 0.996683,
        "chemprop_five_fold_40_epoch_minutes": 32.8,
        "estimate_method": "empirical_pilots_and_conservative_stage_bounds",
    }
    for key, expected in expected_empirical.items():
        if empirical.get(key) != expected:
            raise CampaignError(f"empirical runtime basis mismatch: {key}")
    contract = config.get("scientific_contract")
    if not isinstance(contract, dict):
        raise CampaignError("scientific_contract must be a mapping")
    required_contract = {
        "source_partition": "train",
        "repository_validation_labels_opened": False,
        "repository_test_labels_opened": False,
        "broad_fixed_dose_pooled_into_pic50": False,
        "exact_train_structures": 18801,
        "broad_full_structures": 339373,
        "broad_train_structures": 265625,
        "fixed_outer_folds": 5,
        "fixed_inner_folds": 3,
    }
    for key, expected in required_contract.items():
        if contract.get(key) != expected:
            raise CampaignError(f"scientific contract mismatch: {key}")
    plan = config.get("plan")
    if not isinstance(plan, dict) or set(plan) != {"units"}:
        raise CampaignError("plan must contain exactly an explicit units list")
    raw_units = plan["units"]
    if not isinstance(raw_units, list) or not raw_units:
        raise CampaignError("plan.units must be a nonempty list")
    units: list[dict[str, Any]] = []
    ids: set[str] = set()
    stages_seen: set[str] = set()
    material_hashes: dict[str, str] = {}
    for index, raw in enumerate(raw_units):
        if not isinstance(raw, dict) or set(raw) != UNIT_KEYS:
            raise CampaignError(f"unit {index} keys do not match the governed schema")
        unit = copy.deepcopy(raw)
        unit_id = str(unit["unit_id"])
        stage = str(unit["stage"])
        operation = str(unit["operation"])
        if not unit_id or unit_id in ids or not stage:
            raise CampaignError(f"duplicate/empty unit identity: {unit_id}")
        ids.add(unit_id)
        if operation != unit["spec"].get("operation"):
            raise CampaignError(f"{unit_id}: operation/spec mismatch")
        _validate_capability_spec(unit, capabilities)
        dependencies = unit["dependencies"]
        if not isinstance(dependencies, list):
            raise CampaignError(f"{unit_id}: dependencies must be a list")
        for dependency in dependencies:
            if not isinstance(dependency, dict) or set(dependency) != DEPENDENCY_KEYS:
                raise CampaignError(f"{unit_id}: invalid dependency")
            if dependency["stage"] not in stages_seen:
                raise CampaignError(f"{unit_id}: dependency is forward or absent: {dependency['stage']}")
            if int(dependency["minimum_passed"]) < 1:
                raise CampaignError(f"{unit_id}: invalid dependency threshold")
        expected = float(unit["expected_minutes"])
        if not math.isfinite(expected) or expected <= 0:
            raise CampaignError(f"{unit_id}: expected_minutes must be positive")
        material_hash = _hash_value(unit["spec"], worker=True)
        prior = material_hashes.get(material_hash)
        if prior is not None:
            raise CampaignError(f"pseudo-varied duplicate material specs: {prior}, {unit_id}")
        material_hashes[material_hash] = unit_id
        unit.update(
            unit_id=unit_id,
            stage=stage,
            operation=operation,
            expected_minutes=expected,
            status="pending",
            attempts=0,
            planned_material_spec_sha256=material_hash,
        )
        units.append(unit)
        stages_seen.add(stage)
    _validate_dependency_sources(units)
    _validate_scientific_plan_shape(units)
    nominal = sum(float(unit["expected_minutes"]) for unit in units)
    if not math.isclose(nominal, float(empirical["nominal_useful_minutes"]), rel_tol=0.0, abs_tol=1e-6):
        raise CampaignError("empirical_basis.nominal_useful_minutes does not equal unit sum")
    work_budget = hard * 60 - reserve
    if nominal > work_budget:
        raise CampaignError(
            f"nominal useful work {nominal:.1f} min exceeds active work budget {work_budget:.1f} min"
        )
    completion = config.get("completion")
    if not isinstance(completion, dict):
        raise CampaignError("completion must be a mapping")
    if set(completion) != COMPLETION_KEYS:
        raise CampaignError("completion keys do not match the governed schema")
    thresholds = completion["stage_minimum_passed"]
    if not isinstance(thresholds, dict):
        raise CampaignError("stage_minimum_passed must be a mapping")
    stage_counts = Counter(unit["stage"] for unit in units)
    for stage, minimum in thresholds.items():
        if stage not in stage_counts or not 1 <= int(minimum) <= stage_counts[stage]:
            raise CampaignError(f"impossible completion threshold: {stage}")
    required_roles = completion["required_artifact_roles"]
    if not isinstance(required_roles, dict) or not required_roles:
        raise CampaignError("completion.required_artifact_roles must be a nonempty mapping")
    for operation, roles in required_roles.items():
        if operation not in {unit["operation"] for unit in units}:
            raise CampaignError(f"artifact requirement names absent operation: {operation}")
        if not isinstance(roles, list) or not roles or len(roles) != len(set(roles)):
            raise CampaignError(f"invalid required artifact roles for {operation}")
    present_operations = {unit["operation"] for unit in units}
    for operation, minimum_roles in MINIMUM_COMPLETION_ARTIFACT_ROLES.items():
        if operation not in present_operations:
            continue
        configured = set(required_roles.get(operation, []))
        if not minimum_roles <= configured:
            raise CampaignError(f"completion roles for {operation} omit {sorted(minimum_roles - configured)}")
    if int(completion["exact_nested_rows"]) != 18801:
        raise CampaignError("completion exact nested row requirement must be 18,801")
    _configured_input_bindings(config, repo_root)
    return {
        "config": config,
        "units": units,
        "nominal_expected_minutes": nominal,
        "material_specs": len(material_hashes),
        "stage_counts": dict(stage_counts),
    }


def _checkpoint(output: Path, value: dict[str, Any]) -> dict[str, Any]:
    value["updated_utc"] = _utc()
    return _atomic_json(output / "checkpoint.json", value, "checkpoint_sha256")


def _stage_units(checkpoint: dict[str, Any], stage: str) -> list[dict[str, Any]]:
    return [unit for unit in checkpoint["units"] if unit["stage"] == stage]


def _dependencies_ready(checkpoint: dict[str, Any], unit: dict[str, Any]) -> bool:
    for dependency in unit["dependencies"]:
        source = _stage_units(checkpoint, str(dependency["stage"]))
        if sum(item["status"] == "passed" for item in source) < int(dependency["minimum_passed"]):
            return False
        if dependency["require_terminal"] and any(item["status"] not in TERMINAL for item in source):
            return False
    return True


def _result(output: Path, unit_id: str) -> dict[str, Any]:
    return _read_worker_json(output / "units" / unit_id / "unit.json")


def _candidate_from_result(result: dict[str, Any]) -> dict[str, Any] | None:
    executed = result.get("executed_spec", {})
    if not isinstance(executed, dict):
        return None
    candidate = executed.get("resolved_candidate") or executed.get("candidate")
    return dict(candidate) if isinstance(candidate, dict) else None


def _rank_sources(output: Path, source_ids: list[str], *, aggregate_candidates: bool) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for source_id in source_ids:
        result = _result(output, source_id)
        score = result.get("metrics", {}).get("selection_score")
        if not isinstance(score, (int, float)) or not math.isfinite(float(score)):
            raise CampaignError(f"selection source has no finite score: {source_id}")
        candidate = _candidate_from_result(result)
        candidate_id = str(candidate.get("candidate_id")) if candidate else source_id
        records.append(
            {
                "source_unit_id": source_id,
                "candidate_id": candidate_id,
                "candidate": candidate,
                "engine": (
                    str(candidate.get("engine"))
                    if candidate
                    else ("chemprop" if result.get("operation") == "chemprop_hpo" else "")
                ),
                "score": float(score),
            }
        )
    if not aggregate_candidates:
        return sorted(records, key=lambda row: (row["score"], row["source_unit_id"]))
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(record["candidate_id"], []).append(record)
    ranked: list[dict[str, Any]] = []
    for group in grouped.values():
        scores = [record["score"] for record in group]
        best = min(group, key=lambda row: (row["score"], row["source_unit_id"]))
        ranked.append(
            {
                **best,
                "mean_score": statistics.mean(scores),
                "score_sd": statistics.pstdev(scores),
                "material_repeats": len(scores),
                "member_unit_ids": [record["source_unit_id"] for record in group],
            }
        )
    return sorted(ranked, key=lambda row: (row["mean_score"], row["candidate_id"]))


def _resolve_adaptive_spec(checkpoint: dict[str, Any], unit: dict[str, Any]) -> None:
    """Resolve only preregistered inner-source pools at a unit boundary."""
    spec = unit["spec"]
    if unit.get("adaptive_resolution"):
        return
    output = Path(checkpoint["output_root"])
    operation = unit["operation"]
    resolution: dict[str, Any] = {"resolved_utc": _utc(), "rule": "lowest_inner_selection_score"}
    if operation in {"nested_outer_evaluation", "nested_stack"}:
        ids = [str(value) for value in spec["selection_source_unit_ids"]]
        ranked = _rank_sources(output, ids, aggregate_candidates=True)
        if operation == "nested_outer_evaluation":
            winner = ranked[0]
            if not isinstance(winner["candidate"], dict):
                raise CampaignError("nested winner lacks candidate")
            spec["resolved_candidate"] = winner["candidate"]
            spec["candidate"] = winner["candidate"]
            spec["selected_source_unit_id"] = winner["source_unit_id"]
            resolution.update(
                selected_source_unit_id=winner["source_unit_id"],
                selected_candidate_id=winner["candidate_id"],
                mean_inner_score=winner["mean_score"],
            )
        else:
            # One best candidate per engine, then the strongest remaining
            # candidate, keeps the stack genuinely model-diverse.
            selected: list[dict[str, Any]] = []
            engines: set[str] = set()
            for record in ranked:
                engine = str(record["engine"])
                if engine and engine not in engines:
                    selected.append(record)
                    engines.add(engine)
            if len(selected) < 2:
                raise CampaignError("nested stack lacks two model-diverse inner candidates")
            spec["member_unit_ids"] = [record["source_unit_id"] for record in selected[:3]]
            resolution["selected_member_unit_ids"] = spec["member_unit_ids"]
    elif operation == "final_refit_exact":
        ids = [str(value) for value in spec["selection_source_unit_ids"]]
        winner = _rank_sources(output, ids, aggregate_candidates=False)[0]
        candidate = _candidate_from_result(_result(output, winner["source_unit_id"]))
        if not isinstance(candidate, dict):
            raise CampaignError("exact finalist source lacks candidate")
        spec["source_unit_id"] = winner["source_unit_id"]
        spec["candidate"] = candidate
        resolution.update(selected_source_unit_id=winner["source_unit_id"])
    elif operation == "final_refit_broad":
        ids = [str(value) for value in spec["selection_source_unit_ids"]]
        winner = _rank_sources(output, ids, aggregate_candidates=True)[0]
        if not isinstance(winner["candidate"], dict):
            raise CampaignError("broad finalist source lacks candidate")
        spec["source_unit_id"] = winner["source_unit_id"]
        spec["candidate"] = winner["candidate"]
        resolution.update(selected_source_unit_id=winner["source_unit_id"])
    elif operation == "broad_transfer":
        broad = _rank_sources(
            output,
            [str(value) for value in spec["broad_selection_source_unit_ids"]],
            aggregate_candidates=True,
        )[0]
        exact = _rank_sources(
            output,
            [str(value) for value in spec["exact_selection_source_unit_ids"]],
            aggregate_candidates=False,
        )[0]
        exact_candidate = _candidate_from_result(_result(output, exact["source_unit_id"]))
        if not isinstance(exact_candidate, dict):
            raise CampaignError("transfer exact source lacks candidate")
        spec["broad_source_unit_id"] = broad["source_unit_id"]
        spec["exact_source_unit_id"] = exact["source_unit_id"]
        spec["exact_candidate"] = exact_candidate
        resolution.update(
            selected_broad_source_unit_id=broad["source_unit_id"],
            selected_exact_source_unit_id=exact["source_unit_id"],
        )
    else:
        return
    unit["adaptive_resolution"] = resolution
    unit["planned_material_spec_sha256_before_resolution"] = unit["planned_material_spec_sha256"]
    unit["resolved_material_spec_sha256"] = _hash_value(spec, worker=True)


def _bounded_artifact(output: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = output / path
    path = path.resolve()
    if not path.is_relative_to(output.resolve()):
        raise CampaignError(f"artifact escapes v4 output root: {path}")
    if not path.is_file():
        raise CampaignError(f"missing artifact: {path}")
    return path


def _verify_artifact(output: Path, binding: dict[str, Any]) -> None:
    path = _bounded_artifact(output, str(binding["path"]))
    if path.stat().st_size != int(binding["bytes"]) or _sha256(path) != binding["sha256"]:
        raise CampaignError(f"artifact binding mismatch: {path}")


def _validate_scope(scope: Any) -> None:
    if not isinstance(scope, dict) or scope.get("source_partition") != "train":
        raise CampaignError("worker result is not train-only")
    if scope.get("repository_validation_labels_opened") is not False:
        raise CampaignError("worker did not affirm validation-label blindness")
    if scope.get("repository_test_labels_opened") is not False:
        raise CampaignError("worker did not affirm test-label blindness")
    if scope.get("broad_fixed_dose_pooled_into_pic50") is not False:
        raise CampaignError("broad fixed-dose labels were pooled into pIC50")


def _validate_unit(output: Path, unit: dict[str, Any], capabilities: dict[str, Any]) -> None:
    operation = unit["operation"]
    if operation == "prepare":
        prepared = output / "prepared"
        validation = _read_worker_json(prepared / "validation.json", "validation_sha256")
        manifest = _read_worker_json(prepared / "manifest.json", "manifest_sha256")
        if validation.get("status") != "passed" or manifest.get("status") != "passed":
            raise CampaignError("prepare validation/manifest did not pass")
        if int(validation.get("exact_rows", -1)) != 18801:
            raise CampaignError("prepare exact-row count mismatch")
        if int(validation.get("broad_train_rows", -1)) != 265625:
            raise CampaignError("prepare broad-train count mismatch")
        if validation.get("repository_validation_labels_opened") is not False:
            raise CampaignError("prepare opened validation labels")
        if validation.get("repository_test_labels_opened") is not False:
            raise CampaignError("prepare opened test labels")
        if validation.get("broad_fixed_dose_pooled_into_pic50") is not False:
            raise CampaignError("prepare pooled broad and exact endpoints")
        bindings = manifest.get("bindings")
        if not isinstance(bindings, list) or not bindings:
            raise CampaignError("prepare manifest lacks immutable input bindings")
        _verify_bindings(bindings)
        return
    if operation == "analyze":
        required = capabilities["operations"]["analyze"].get("required_artifacts", [])
        for name in required:
            if not (output / "analysis" / str(name)).is_file():
                raise CampaignError(f"analysis missing required artifact: {name}")
        validation = _read_worker_json(output / "analysis" / "validation.json", "validation_sha256")
        if validation.get("status") != "passed":
            raise CampaignError("analysis validation did not pass")
        if validation.get("repository_validation_labels_opened") is not False:
            raise CampaignError("analysis opened validation labels")
        if validation.get("repository_test_labels_opened") is not False:
            raise CampaignError("analysis opened test labels")
        if validation.get("broad_fixed_dose_pooled_into_pic50") is not False:
            raise CampaignError("analysis pooled broad and exact endpoints")
        manifest = _read_worker_json(output / "analysis" / "manifest.json", "manifest_sha256")
        bindings = manifest.get("bindings")
        if not isinstance(bindings, list) or not bindings:
            raise CampaignError("analysis manifest lacks artifact bindings")
        for binding in bindings:
            _verify_artifact(output, binding)
        return
    result = _read_worker_json(output / "units" / unit["unit_id"] / "unit.json")
    if result.get("status") != "passed" or result.get("unit_id") != unit["unit_id"]:
        raise CampaignError(f"unit identity/status mismatch: {unit['unit_id']}")
    if result.get("operation") != operation or result.get("unit_spec") != unit["spec"]:
        raise CampaignError(f"unit operation/spec mismatch: {unit['unit_id']}")
    if result.get("unit_spec_sha256") != _hash_value(unit["spec"], worker=True):
        raise CampaignError(f"unit spec hash mismatch: {unit['unit_id']}")
    executed = result.get("executed_spec")
    if not isinstance(executed, dict):
        raise CampaignError(f"unit lacks executed_spec: {unit['unit_id']}")
    executed_hash = _hash_value(executed, worker=True)
    if result.get("executed_spec_sha256") != executed_hash:
        raise CampaignError(f"executed spec hash mismatch: {unit['unit_id']}")
    material = {
        key: value
        for key, value in executed.items()
        if key
        not in {
            "source_partition",
            "repository_validation_labels_opened",
            "repository_test_labels_opened",
            "broad_fixed_dose_pooled_into_pic50",
        }
    }
    if result.get("material_spec_sha256") != _hash_value(material, worker=True):
        raise CampaignError(f"material spec was not executed exactly: {unit['unit_id']}")
    _validate_scope(result.get("scientific_scope"))
    _, score_required = _validate_capability_spec(unit, capabilities)
    if score_required:
        score = result.get("metrics", {}).get("selection_score")
        if not isinstance(score, (int, float)) or not math.isfinite(float(score)):
            raise CampaignError(f"unit lacks finite selection_score: {unit['unit_id']}")
    artifacts = result.get("artifacts")
    if not isinstance(artifacts, list):
        raise CampaignError(f"invalid artifact ledger: {unit['unit_id']}")
    for binding in artifacts:
        if not isinstance(binding, dict):
            raise CampaignError(f"invalid artifact binding: {unit['unit_id']}")
        _verify_artifact(output, binding)


def _output_bytes(output: Path) -> int:
    return sum(path.stat().st_size for path in output.rglob("*") if path.is_file())


def _resource_gate(output: Path, campaign: dict[str, Any]) -> dict[str, Any]:
    free_disk = shutil.disk_usage(output).free
    minimum_disk = float(campaign["minimum_free_disk_gib"]) * 1024**3
    if free_disk < minimum_disk:
        raise CampaignError(f"disk gate: only {free_disk / 1024**3:.1f} GiB free")
    output_bytes = _output_bytes(output)
    if output_bytes > float(campaign["maximum_output_gib"]) * 1024**3:
        raise CampaignError(f"output-size gate: {output_bytes / 1024**3:.1f} GiB")
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
    minimum_memory = float(campaign["minimum_available_memory_gib"]) * 1024**3
    if available is not None and available < minimum_memory:
        raise CampaignError(f"memory gate: only {available / 1024**3:.1f} GiB available")
    return {
        "status": "passed",
        "checked_utc": _utc(),
        "disk_free_bytes": free_disk,
        "memory_available_bytes": available,
        "memory_method": method,
        "output_bytes": output_bytes,
    }


def _command(checkpoint: dict[str, Any], unit: dict[str, Any]) -> list[str]:
    python = checkpoint["python_path"]
    worker = checkpoint["worker_path"]
    output = Path(checkpoint["output_root"])
    common = ["--repo-root", checkpoint["repo_root"]]
    if unit["operation"] == "prepare":
        return [
            python,
            worker,
            "prepare",
            *common,
            "--base-v2-root",
            checkpoint["base_v2_root"],
            "--base-v3-root",
            checkpoint["base_v3_root"],
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
        "--results-root",
        str(output),
        "--unit-id",
        unit["unit_id"],
        "--unit-spec-json",
        json.dumps(unit["spec"], sort_keys=True, separators=(",", ":"), allow_nan=False),
        "--workers",
        str(checkpoint["workers"]),
    ]


def _run_process(
    command: list[str], log_path: Path, environment: dict[str, str], deadline: float
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
            if time.time() >= deadline:
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


def _run_with_retry(
    command: list[str],
    unit: dict[str, Any],
    output: Path,
    checkpoint: dict[str, Any],
    capabilities: dict[str, Any],
    environment: dict[str, str],
    deadline: float,
) -> tuple[int, bool, float, Path]:
    elapsed_total = 0.0
    returncode = 2
    timed_out = False
    log_path = output / "logs" / f"{unit['unit_id']}.attempt_1.log"
    while int(unit["attempts"]) < MAX_ATTEMPTS:
        unit["attempts"] = int(unit["attempts"]) + 1
        attempt = int(unit["attempts"])
        log_path = output / "logs" / f"{unit['unit_id']}.attempt_{attempt}.log"
        unit["log_path"] = str(log_path)
        _checkpoint(output, checkpoint)
        returncode, timed_out, elapsed = _run_process(command, log_path, environment, deadline)
        elapsed_total += elapsed
        checkpoint["active_elapsed_seconds"] += elapsed
        checkpoint["invocations"][-1]["active_elapsed_seconds"] += elapsed
        validation_error: str | None = None
        if returncode == 0:
            try:
                _validate_unit(output, unit, capabilities)
            except (CampaignError, OSError, ValueError, KeyError, json.JSONDecodeError) as error:
                validation_error = str(error)
                returncode = 2
        unit.setdefault("attempt_results", []).append(
            {
                "attempt": attempt,
                "returncode": returncode,
                "timed_out": timed_out,
                "elapsed_seconds": elapsed,
                "log_path": str(log_path),
                "artifact_validation_error": validation_error,
            }
        )
        _checkpoint(output, checkpoint)
        # A unit may emit more than forecast. Enforce the campaign cap after
        # every isolated attempt, before that result can become terminal.
        maximum_output = float(checkpoint["campaign_governance"]["maximum_output_gib"]) * 1024**3
        if _output_bytes(output) > maximum_output:
            returncode = 2
            unit["post_attempt_resource_error"] = "output-size gate exceeded"
            unit.setdefault("attempt_results", [])[-1]["post_attempt_resource_error"] = (
                "output-size gate exceeded"
            )
            _checkpoint(output, checkpoint)
            break
        if returncode in {0, 3} or time.time() >= deadline:
            break
    return returncode, timed_out, elapsed_total, log_path


def _initial(args: argparse.Namespace) -> dict[str, Any]:
    repo = Path(args.repo_root).resolve()
    output = Path(args.output_root).resolve()
    config_path = Path(args.config).resolve()
    protocol = Path(args.protocol).resolve()
    worker = Path(args.worker).resolve()
    python = Path(args.python).absolute()
    capabilities = _worker_capabilities(python, worker)
    validated = _validate_config(config_path, repo, capabilities)
    config = validated["config"]
    capability_path = output / "governance" / "worker_capabilities.json"
    _atomic_bytes(capability_path, _canonical(capabilities, newline=True))
    bindings = [
        _binding(Path(__file__), "v4_orchestrator"),
        _binding(worker, "v4_worker"),
        _binding(config_path, "v4_config"),
        _binding(protocol, "v4_preregistered_protocol"),
        _binding(capability_path, "v4_worker_capabilities"),
        *_implementation_import_bindings(worker),
        *_configured_input_bindings(config, repo),
    ]
    campaign = config["campaign"]
    return {
        "schema_version": SCHEMA_VERSION,
        "created_utc": _utc(),
        "status": "running",
        "repo_root": str(repo),
        "output_root": str(output),
        "config_path": str(config_path),
        "protocol_path": str(protocol),
        "base_v2_root": str(Path(args.base_v2_root).resolve()),
        "base_v3_root": str(Path(args.base_v3_root).resolve()),
        "worker_path": str(worker),
        "python_path": str(python),
        "workers": int(campaign["workers"]),
        "target_hours": float(campaign["target_active_hours"]),
        "hard_hours": float(campaign["hard_active_hours"]),
        "reserve_minutes": float(campaign["finalization_reserve_minutes"]),
        "nominal_expected_minutes": validated["nominal_expected_minutes"],
        "active_elapsed_seconds": 0.0,
        "bindings": bindings,
        "capabilities": capabilities,
        "campaign_governance": campaign,
        "scientific_contract": config["scientific_contract"],
        "completion": config["completion"],
        "units": validated["units"],
        "invocations": [],
        "resource_checks": [],
        "stop_requests": [],
        "artifact_adoptions": [],
    }


def _resume(args: argparse.Namespace, output: Path) -> dict[str, Any]:
    checkpoint = _read_json(output / "checkpoint.json", "checkpoint_sha256")
    _verify_bindings(checkpoint["bindings"])
    expected = {
        "repo_root": str(Path(args.repo_root).resolve()),
        "output_root": str(output),
        "config_path": str(Path(args.config).resolve()),
        "protocol_path": str(Path(args.protocol).resolve()),
        "base_v2_root": str(Path(args.base_v2_root).resolve()),
        "base_v3_root": str(Path(args.base_v3_root).resolve()),
        "worker_path": str(Path(args.worker).resolve()),
        "python_path": str(Path(args.python).absolute()),
    }
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise CampaignError(f"resume argument changed: {key}")
    capabilities = checkpoint["capabilities"]
    for unit in checkpoint["units"]:
        if unit["status"] == "passed":
            continue
        artifact = output / "units" / unit["unit_id"] / "unit.json"
        prepare_ready = unit["operation"] == "prepare" and (output / "prepared/manifest.json").is_file()
        analyze_ready = unit["operation"] == "analyze" and (output / "analysis/validation.json").is_file()
        if artifact.is_file() or prepare_ready or analyze_ready:
            try:
                _validate_unit(output, unit, capabilities)
            except (CampaignError, OSError, ValueError, KeyError, json.JSONDecodeError):
                pass
            else:
                unit["status"] = "passed"
                checkpoint["artifact_adoptions"].append(
                    {
                        "unit_id": unit["unit_id"],
                        "adopted_utc": _utc(),
                        "attempts_preserved": unit["attempts"],
                    }
                )
                continue
        if unit["status"] == "running":
            if int(unit["attempts"]) < MAX_ATTEMPTS:
                unit["status"] = "pending"
                unit["interrupted_attempt_recovered"] = True
            else:
                unit["status"] = "failed_critical" if unit["critical"] else "failed_noncritical"
                unit["interrupted_after_attempt_budget_exhausted"] = True
    checkpoint["status"] = "running"
    return _checkpoint(output, checkpoint)


def _completion_reasons(checkpoint: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    for unit in checkpoint["units"]:
        if unit["required_for_completion"] and unit["status"] != "passed":
            reasons.append(f"required unit not passed: {unit['unit_id']}")
    for stage, minimum in checkpoint["completion"]["stage_minimum_passed"].items():
        passed = sum(unit["status"] == "passed" for unit in _stage_units(checkpoint, stage))
        if passed < int(minimum):
            reasons.append(f"stage {stage}: {passed}/{minimum} required units passed")
    return reasons


def _write_compute_manifest(output: Path, checkpoint: dict[str, Any]) -> dict[str, Any]:
    """Freeze the successful compute surface without performing interpretation."""
    records: list[dict[str, Any]] = []
    for unit in checkpoint["units"]:
        if unit["status"] != "passed" or unit["operation"] == "prepare":
            continue
        result = _result(output, unit["unit_id"])
        records.append(
            {
                "unit_id": unit["unit_id"],
                "operation": unit["operation"],
                "metrics": result.get("metrics", {}),
                "artifacts": result.get("artifacts", []),
                "unit_json_sha256": result["unit_json_sha256"],
            }
        )
    return _atomic_json(
        output / "compute_manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "status": "passed",
            "created_utc": _utc(),
            "analysis_deferred": True,
            "repository_validation_labels_opened": False,
            "repository_test_labels_opened": False,
            "broad_fixed_dose_pooled_into_pic50": False,
            "units": records,
        },
        "compute_manifest_sha256",
    )


def _validate_terminal(output: Path, checkpoint: dict[str, Any]) -> None:
    requirements = checkpoint["completion"]["required_artifact_roles"]
    for operation, required_roles in requirements.items():
        matched = [unit for unit in checkpoint["units"] if unit["operation"] == operation]
        if not matched:
            raise CampaignError(f"terminal artifact operation absent: {operation}")
        passed = 0
        for unit in matched:
            if unit["status"] != "passed":
                continue
            passed += 1
            result = _result(output, unit["unit_id"])
            unit_roles = {str(item.get("role")) for item in result.get("artifacts", [])}
            missing = set(required_roles) - unit_roles
            if missing:
                raise CampaignError(f"{unit['unit_id']} lacks terminal artifact roles: {sorted(missing)}")
        if passed == 0:
            raise CampaignError(f"terminal artifact operation has no passed unit: {operation}")
    aggregate = [
        unit
        for unit in checkpoint["units"]
        if unit["operation"] == "aggregate_nested_oof" and unit["status"] == "passed"
    ]
    if len(aggregate) != 1:
        raise CampaignError("completion requires exactly one passed nested OOF aggregate")
    result = _result(output, aggregate[0]["unit_id"])
    metrics = result.get("metrics", {})
    if int(metrics.get("folds", -1)) != 5:
        raise CampaignError("nested OOF aggregate does not cover five folds")
    artifact = next(
        (item for item in result.get("artifacts", []) if item.get("role") == "oof_predictions"),
        None,
    )
    if not isinstance(artifact, dict):
        raise CampaignError("nested OOF aggregate lacks predictions")
    try:
        import pyarrow.parquet as parquet

        rows = parquet.read_metadata(_bounded_artifact(output, str(artifact["path"]))).num_rows
    except (ImportError, OSError, ValueError) as error:
        raise CampaignError(f"cannot validate nested OOF row count: {error}") from error
    expected_rows = int(checkpoint["completion"]["exact_nested_rows"])
    if rows != expected_rows:
        raise CampaignError(f"nested OOF row mismatch: {rows} != {expected_rows}")


def _finalize(output: Path, checkpoint: dict[str, Any]) -> dict[str, Any]:
    reasons = _completion_reasons(checkpoint)
    if not reasons:
        try:
            _validate_terminal(output, checkpoint)
        except (CampaignError, OSError, ValueError, KeyError, json.JSONDecodeError) as error:
            reasons.append(str(error))
    if reasons:
        checkpoint["status"] = "incomplete"
        checkpoint["incomplete_reasons"] = reasons
        _checkpoint(output, checkpoint)
        return {"status": "incomplete", "reasons": reasons, "resume": "rerun identical launcher"}
    counts = Counter(str(unit["status"]) for unit in checkpoint["units"])
    failures = [
        {key: unit.get(key) for key in ("unit_id", "stage", "operation", "status", "log_path")}
        for unit in checkpoint["units"]
        if str(unit["status"]).startswith("failed")
    ]
    checkpoint["status"] = "complete_with_noncritical_failures" if failures else "complete"
    checkpoint["finished_utc"] = _utc()
    checkpoint = _checkpoint(output, checkpoint)
    compute_manifest = _write_compute_manifest(output, checkpoint)
    summary = _atomic_json(
        output / "final_summary.json",
        {
            "schema_version": SCHEMA_VERSION,
            "status": checkpoint["status"],
            "created_utc": _utc(),
            "active_elapsed_seconds": checkpoint["active_elapsed_seconds"],
            "completed_early_because_prespecified_science_finished": checkpoint["active_elapsed_seconds"]
            < checkpoint["target_hours"] * 3600,
            "unit_status_counts": dict(counts),
            "failure_ledger": failures,
            "scientific_contract": checkpoint["scientific_contract"],
            "checkpoint_sha256": checkpoint["checkpoint_sha256"],
            "compute_results_root": str(output / "units"),
            "compute_manifest_sha256": compute_manifest["compute_manifest_sha256"],
            "analysis_deferred": True,
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
            "compute_manifest_sha256": compute_manifest["compute_manifest_sha256"],
        },
        "done_sha256",
    )


def _start(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with _Lock(output / ".campaign.lock"):
        checkpoint = (
            _resume(args, output)
            if (output / "checkpoint.json").is_file()
            else _checkpoint(output, _initial(args))
        )
        if str(checkpoint["status"]).startswith("complete"):
            return {"status": checkpoint["status"], "message": "v4 already complete"}
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
                {"signal": signum, "received_utc": _utc(), "policy": "unit_boundary"}
            )

        old_handlers = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)}
        environment = os.environ.copy()
        environment.update(
            {
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONUNBUFFERED": "1",
                "OMP_NUM_THREADS": str(checkpoint["workers"]),
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
                        continue
                    finalization = unit["operation"] == "analyze"
                    deadline = (
                        invocation["hard_deadline_epoch"]
                        if finalization
                        else invocation["work_deadline_epoch"]
                    )
                    remaining_minutes = (deadline - time.time()) / 60
                    if remaining_minutes <= 0 or (
                        not finalization and float(unit["expected_minutes"]) > remaining_minutes
                    ):
                        if unit["required_for_completion"]:
                            checkpoint["status"] = "safely_stopped_time_budget"
                            _checkpoint(output, checkpoint)
                            return {
                                "status": checkpoint["status"],
                                "unit": unit["unit_id"],
                                "resume": "active-time budget exhausted",
                            }
                        unit["status"] = "skipped_time"
                        _checkpoint(output, checkpoint)
                        progress = True
                        continue
                    try:
                        gate = _resource_gate(output, checkpoint["campaign_governance"])
                    except CampaignError as error:
                        checkpoint["status"] = "safely_stopped_resource_gate"
                        checkpoint["resource_error"] = str(error)
                        _checkpoint(output, checkpoint)
                        return {"status": checkpoint["status"], "error": str(error)}
                    gate["before_unit"] = unit["unit_id"]
                    checkpoint["resource_checks"].append(gate)
                    try:
                        _resolve_adaptive_spec(checkpoint, unit)
                        _validate_capability_spec(unit, checkpoint["capabilities"])
                    except (CampaignError, KeyError, ValueError, TypeError) as error:
                        unit["status"] = "failed_critical" if unit["critical"] else "failed_noncritical"
                        unit["adaptive_resolution_error"] = str(error)
                        _checkpoint(output, checkpoint)
                        progress = True
                        continue
                    unit["status"] = "running"
                    _checkpoint(output, checkpoint)
                    returncode, timed_out, elapsed, log_path = _run_with_retry(
                        _command(checkpoint, unit),
                        unit,
                        output,
                        checkpoint,
                        checkpoint["capabilities"],
                        environment,
                        deadline,
                    )
                    unit.update(
                        returncode=returncode,
                        timed_out=timed_out,
                        elapsed_seconds=elapsed,
                        finished_utc=_utc(),
                        log_path=str(log_path),
                    )
                    if unit.get("post_attempt_resource_error"):
                        unit["status"] = "pending"
                        checkpoint["status"] = "safely_stopped_resource_gate"
                        checkpoint["resource_error"] = unit["post_attempt_resource_error"]
                        _checkpoint(output, checkpoint)
                        return {
                            "status": checkpoint["status"],
                            "unit": unit["unit_id"],
                            "error": checkpoint["resource_error"],
                        }
                    if returncode == 0:
                        unit["status"] = "passed"
                    elif returncode == 3 and not unit["critical"]:
                        unit["status"] = "skipped_unavailable"
                    else:
                        unit["status"] = "failed_critical" if unit["critical"] else "failed_noncritical"
                    _checkpoint(output, checkpoint)
                    progress = True
                    if unit["status"] == "failed_critical":
                        checkpoint["status"] = "failed"
                        _checkpoint(output, checkpoint)
                        return {"status": "failed", "unit": unit["unit_id"], "log": str(log_path)}
            return _finalize(output, checkpoint)
        finally:
            for sig, handler in old_handlers.items():
                signal.signal(sig, handler)


def _status(output: Path) -> dict[str, Any]:
    checkpoint = _read_json(output.resolve() / "checkpoint.json", "checkpoint_sha256")
    counts = Counter(str(unit["status"]) for unit in checkpoint["units"])
    active = next((unit["unit_id"] for unit in checkpoint["units"] if unit["status"] == "running"), None)
    return {
        "status": checkpoint["status"],
        "active_unit": active,
        "unit_status_counts": dict(counts),
        "active_elapsed_hours": checkpoint["active_elapsed_seconds"] / 3600,
        "hard_active_hours_remaining": max(
            0.0, checkpoint["hard_hours"] - checkpoint["active_elapsed_seconds"] / 3600
        ),
        "updated_utc": checkpoint["updated_utc"],
        "output_root": checkpoint["output_root"],
    }


def _validate(output: Path, require_done: bool) -> dict[str, Any]:
    output = output.resolve()
    checkpoint = _read_json(output / "checkpoint.json", "checkpoint_sha256")
    _verify_bindings(checkpoint["bindings"])
    if checkpoint.get("schema_version") != SCHEMA_VERSION or checkpoint.get("workers") != 6:
        raise CampaignError("campaign governance mismatch")
    _validate_scope(checkpoint["scientific_contract"])
    maximum_output = float(checkpoint["campaign_governance"]["maximum_output_gib"]) * 1024**3
    if _output_bytes(output) > maximum_output:
        raise CampaignError("campaign output exceeds governed size cap")
    for unit in checkpoint["units"]:
        if unit["status"] == "passed":
            if not Path(str(unit.get("log_path", ""))).is_file():
                raise CampaignError(f"missing unit log: {unit['unit_id']}")
            _validate_unit(output, unit, checkpoint["capabilities"])
    if require_done:
        if checkpoint["status"] not in {"complete", "complete_with_noncritical_failures"}:
            raise CampaignError(f"campaign is not complete: {checkpoint['status']}")
        reasons = _completion_reasons(checkpoint)
        if reasons:
            raise CampaignError("; ".join(reasons))
        _validate_terminal(output, checkpoint)
        done = _read_json(output / "DONE.json", "done_sha256")
        summary = _read_json(output / "final_summary.json", "summary_sha256")
        compute_manifest = _read_json(output / "compute_manifest.json", "compute_manifest_sha256")
        if done["checkpoint_sha256"] != checkpoint["checkpoint_sha256"]:
            raise CampaignError("DONE does not bind the final checkpoint")
        if done["summary_sha256"] != summary["summary_sha256"]:
            raise CampaignError("DONE does not bind the final summary")
        if done["compute_manifest_sha256"] != compute_manifest["compute_manifest_sha256"]:
            raise CampaignError("DONE does not bind the compute manifest")
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
    for name in ("start", "validate-config"):
        command = commands.add_parser(name)
        command.add_argument("--repo-root", default=".")
        command.add_argument("--config", default=str(DEFAULT_CONFIG))
        command.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
        command.add_argument("--base-v2-root", default=str(DEFAULT_V2))
        command.add_argument("--base-v3-root", default=str(DEFAULT_V3))
        command.add_argument(
            "--worker", default=str(Path(__file__).with_name("run_local_herg_discovery_worker_v4.py"))
        )
        command.add_argument("--python", default=sys.executable)
        command.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    for name in ("status", "validate"):
        command = commands.add_parser(name)
        command.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    commands.choices["validate"].add_argument("--allow-incomplete", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        if args.action == "start":
            result = _start(args)
        elif args.action == "validate-config":
            capabilities = _worker_capabilities(Path(args.python), Path(args.worker))
            validated = _validate_config(Path(args.config), Path(args.repo_root).resolve(), capabilities)
            result = {
                "status": "passed",
                "units": len(validated["units"]),
                "nominal_expected_minutes": validated["nominal_expected_minutes"],
                "material_specs": validated["material_specs"],
                "stage_counts": validated["stage_counts"],
            }
        elif args.action == "status":
            result = _status(Path(args.output_root))
        else:
            result = _validate(Path(args.output_root), require_done=not args.allow_incomplete)
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except (
        CampaignError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
        json.JSONDecodeError,
        subprocess.SubprocessError,
        yaml.YAMLError,
    ) as error:
        print(json.dumps({"status": "failed", "error": str(error)}, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
