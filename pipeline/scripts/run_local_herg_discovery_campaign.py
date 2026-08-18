#!/usr/bin/env python3
"""Run a bounded, resumable, train-only hERG discovery campaign.

The orchestrator schedules only concrete commands implemented by
``run_local_herg_discovery_worker.py``.  Coarse candidates are compared on one
inner scaffold fold per outer fold, the best candidates are promoted to the
remaining inner folds, and one recipe per outer fold is selected without
reading that outer fold.  Thus every final ``nested_selected`` prediction is
made for a scaffold group excluded from both fitting and model selection.

Repository validation and test outcomes are never requested.  The target is
exact pIC50 for wild-type-or-target-unspecified hERG, which is not the same as
an experimentally confirmed-WT-only target.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Any

SCHEMA_VERSION = "platform-local-herg-discovery-campaign/2.0"
DEFAULT_OUTPUT = Path("research/local_runs/herg_discovery_campaign_v1")
DEFAULT_MATRIX = Path("research/local_runs/herg_quantitative_feature_matrix_v1")
DEFAULT_OBSERVATIONS = Path(
    "research/data/platform/processed/herg_hierarchy/v1_6_training_surfaces/"
    "herg_training_observations.parquet"
)
DEFAULT_WORKERS = 6
TERMINAL = {
    "passed",
    "failed_noncritical",
    "failed_critical",
    "skipped_time",
    "skipped_unavailable",
}


class CampaignError(RuntimeError):
    """Raised when a campaign governance or reproducibility contract fails."""


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


def _read_json(path: Path, hash_key: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CampaignError(f"expected a JSON object: {path}")
    _verify_self_hash(value, hash_key)
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
        raise CampaignError(f"missing {role}: {path}")
    return {"role": role, "path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}


def _verify_bindings(bindings: list[dict[str, Any]]) -> None:
    for binding in bindings:
        path = Path(binding["path"])
        if not path.is_file() or path.stat().st_size != int(binding["bytes"]):
            raise CampaignError(f"bound file missing or changed: {path}")
        if _sha256(path) != binding["sha256"]:
            raise CampaignError(f"bound file hash changed: {path}")


def _candidate(
    key: str,
    engine: str,
    groups: str,
    parameters: dict[str, Any],
    hypothesis: str,
) -> dict[str, Any]:
    return {
        "candidate_key": key,
        "engine": engine,
        "groups": groups,
        "params": parameters,
        "hypothesis": hypothesis,
    }


def _candidates() -> list[dict[str, Any]]:
    """Prespecified feature and parameter candidates for nested halving."""
    xgb_default = {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.04,
        "min_child_weight": 3,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.05,
        "reg_lambda": 1.0,
    }
    candidates = [
        _candidate("xgb_rdkit2d", "xgboost", "rdkit2d", xgb_default, "2D reference"),
        _candidate("xgb_morgan", "xgboost", "morgan", xgb_default, "substructure reference"),
        _candidate(
            "xgb_morgan_2d",
            "xgboost",
            "morgan_rdkit2d",
            xgb_default,
            "substructure plus physicochemistry",
        ),
        _candidate(
            "xgb_morgan_2d_maccs",
            "xgboost",
            "morgan_rdkit2d_maccs",
            xgb_default,
            "added interpretable structural keys",
        ),
        _candidate(
            "xgb_selected_physics",
            "xgboost",
            "physics_selected",
            xgb_default,
            "spatial electrostatics without fingerprint identity",
        ),
        _candidate(
            "xgb_candidate_primary",
            "xgboost",
            "candidate_primary",
            xgb_default,
            "Morgan, 2D, spatial charge/polarity, and label-blind process interactions",
        ),
        _candidate(
            "xgb_all_scalable",
            "xgboost",
            "all_scalable",
            xgb_default,
            "test whether untargeted descriptors dilute selected physics",
        ),
    ]
    xgb_variants = [
        ("shallow_slow", 4, 0.025, 1000, 5, 0.9, 0.8, 0.1, 2.0),
        ("shallow_regularized", 3, 0.035, 800, 8, 0.8, 0.7, 0.5, 4.0),
        ("medium_robust", 5, 0.025, 1100, 5, 0.85, 0.75, 0.2, 3.0),
        ("deep_sparse", 8, 0.02, 1200, 8, 0.75, 0.6, 0.5, 5.0),
        ("huber_like", 5, 0.03, 900, 4, 0.85, 0.8, 0.1, 2.0),
        ("absolute_error", 5, 0.03, 900, 4, 0.85, 0.8, 0.1, 2.0),
    ]
    for name, depth, rate, trees, child, rows, columns, alpha, ridge in xgb_variants:
        params: dict[str, Any] = {
            "n_estimators": trees,
            "max_depth": depth,
            "learning_rate": rate,
            "min_child_weight": child,
            "subsample": rows,
            "colsample_bytree": columns,
            "reg_alpha": alpha,
            "reg_lambda": ridge,
        }
        if name == "huber_like":
            params["objective"] = "reg:pseudohubererror"
        if name == "absolute_error":
            params["objective"] = "reg:absoluteerror"
        candidates.append(
            _candidate(
                f"xgb_primary_{name}",
                "xgboost",
                "candidate_primary",
                params,
                f"XGBoost parameter regime {name}",
            )
        )
    lgbm_variants = [
        ("lgbm_primary_default", 31, 20, 0.03, 700, 0.05, 1.0),
        ("lgbm_primary_small_leaves", 15, 30, 0.025, 1000, 0.1, 2.0),
        ("lgbm_primary_large_leaves", 63, 40, 0.02, 1100, 0.3, 3.0),
        ("lgbm_morgan_2d", 31, 20, 0.03, 700, 0.05, 1.0),
        ("lgbm_all_scalable", 31, 30, 0.025, 900, 0.2, 2.0),
    ]
    for name, leaves, child, rate, trees, alpha, ridge in lgbm_variants:
        groups = (
            "morgan_rdkit2d"
            if name.endswith("morgan_2d")
            else "all_scalable"
            if name.endswith("all_scalable")
            else "candidate_primary"
        )
        candidates.append(
            _candidate(
                name,
                "lightgbm",
                groups,
                {
                    "n_estimators": trees,
                    "num_leaves": leaves,
                    "min_child_samples": child,
                    "learning_rate": rate,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "reg_alpha": alpha,
                    "reg_lambda": ridge,
                },
                f"LightGBM parameter regime {name}",
            )
        )
    return candidates


def _unit(unit_id: str, campaign_stage: str, kind: str, expected: float, **spec: Any) -> dict[str, Any]:
    return {
        "unit_id": unit_id,
        "campaign_stage": campaign_stage,
        "kind": kind,
        "expected_minutes": expected,
        "critical": bool(spec.pop("critical", False)),
        "status": "pending",
        "attempts": 0,
        "spec": spec,
    }


def _plan() -> list[dict[str, Any]]:
    units = [_unit("prepare", "prepare", "prepare", 10, critical=True)]
    for outer_fold in range(5):
        for candidate in _candidates():
            units.append(
                _unit(
                    f"coarse_o{outer_fold}__{candidate['candidate_key']}",
                    "coarse_halving",
                    "tree",
                    2,
                    candidate=candidate,
                    stage="inner",
                    outer_fold=outer_fold,
                    inner_fold=0,
                    seed=20260811,
                )
            )
    # Slots resolve to the six best available coarse candidates for each outer
    # fold.  Both remaining inner folds are evaluated before outer selection.
    for outer_fold in range(5):
        for rank in range(6):
            for inner_fold in (1, 2):
                units.append(
                    _unit(
                        f"promoted_o{outer_fold}_r{rank}_i{inner_fold}",
                        "promoted_halving",
                        "tree",
                        3,
                        select="coarse_rank",
                        rank=rank,
                        stage="inner",
                        outer_fold=outer_fold,
                        inner_fold=inner_fold,
                        seed=20260811,
                    )
                )
        units.append(
            _unit(
                f"nested_selected_outer_{outer_fold}",
                "nested_confirmation",
                "tree",
                5,
                select="promoted_mean_winner",
                stage="outer",
                outer_fold=outer_fold,
                seed=20260811,
                critical=True,
            )
        )

    # Fixed outer-fold diagnostics quantify what helps and hurts without using
    # them to alter the nested-selected prediction for that fold.
    diagnostic_sets = {
        "diag_2d": "rdkit2d",
        "diag_morgan_2d": "morgan_rdkit2d",
        "diag_selected_physics": "physics_selected",
        "diag_primary": "candidate_primary",
        "diag_all_scalable": "all_scalable",
        "diag_no_interactions": "morgan,rdkit2d,autocorr3d,polarity_charge_scalars",
        "diag_no_autocorr": "morgan,rdkit2d,polarity_charge_scalars,targeted_interactions",
        "diag_no_polarity_charge": "morgan,rdkit2d,autocorr3d,targeted_interactions",
        "diag_no_fingerprint": "rdkit2d,autocorr3d,polarity_charge_scalars,targeted_interactions",
        "diag_shape_added": "candidate_primary,shape_scalars",
        "diag_whim_added": "candidate_primary,whim3d",
        "diag_energy_added": "candidate_primary,energy_flexibility",
    }
    diagnostic_params = next(
        item["params"] for item in _candidates() if item["candidate_key"] == "xgb_candidate_primary"
    )
    for model_id, groups in diagnostic_sets.items():
        for outer_fold in range(5):
            units.append(
                _unit(
                    f"{model_id}_outer_{outer_fold}",
                    "feature_relationships",
                    "tree",
                    4,
                    candidate=_candidate(model_id, "xgboost", groups, diagnostic_params, model_id),
                    stage="outer",
                    outer_fold=outer_fold,
                    seed=20260821,
                )
            )
    for model, model_id in (("ridge", "classical_ridge"), ("extratrees", "classical_extratrees")):
        for outer_fold in range(5):
            units.append(
                _unit(
                    f"{model_id}_outer_{outer_fold}",
                    "fixed_alternatives",
                    "classical",
                    5,
                    model=model,
                    model_id=model_id,
                    groups="safe_classical",
                    params={},
                    stage="outer",
                    outer_fold=outer_fold,
                    seed=20260831,
                )
            )
    for outer_fold in range(5):
        units.append(
            _unit(
                f"similarity_outer_{outer_fold}",
                "fixed_alternatives",
                "similarity",
                5,
                model_id="similarity_tanimoto_knn",
                stage="outer",
                outer_fold=outer_fold,
                seed=20260831,
                neighbors=10,
                similarity_floor=0.0,
                weight_power=2.0,
            )
        )
    units.append(_unit("chemprop_export", "optional_chemprop", "chemprop_export", 5))
    for outer_fold in range(5):
        units.append(
            _unit(
                f"chemprop_outer_{outer_fold}",
                "optional_chemprop",
                "chemprop",
                150,
                outer_fold=outer_fold,
                seed=20260901 + outer_fold,
                epochs=40,
                patience=8,
                maximum_minutes=150,
            )
        )
    units.append(_unit("analyze", "finalize", "analyze", 30, critical=True))
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
            raise CampaignError(f"campaign lock is already held: {self.path}") from error
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
        raise CampaignError(f"disk safety gate: only {free_disk / 1024**3:.1f} GiB free")
    available: int | None = None
    method = "unavailable_best_effort"
    try:
        query = subprocess.run(
            ["memory_pressure", "-Q"], check=False, capture_output=True, text=True, timeout=10
        )
        line = next(
            item for item in query.stdout.splitlines() if "System-wide memory free percentage:" in item
        )
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
        raise CampaignError(f"memory-pressure safety gate: {available / 1024**3:.1f} GiB available")
    return {
        "checked_utc": _utc(),
        "status": "passed",
        "disk_free_bytes": free_disk,
        "memory_available_bytes": available,
        "memory_method": method,
    }


def _metric(output: Path, unit_id: str) -> float | None:
    path = output / "units" / unit_id / "metrics.json"
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != "passed":
        return None
    metric = value.get("metrics", {}).get("mae")
    return float(metric) if metric is not None else None


def _coarse_ranking(checkpoint: dict[str, Any], outer_fold: int) -> list[dict[str, Any]]:
    output = Path(checkpoint["output_root"])
    scored: list[tuple[float, str, dict[str, Any]]] = []
    for unit in checkpoint["units"]:
        spec = unit["spec"]
        if unit["campaign_stage"] != "coarse_halving" or spec.get("outer_fold") != outer_fold:
            continue
        score = _metric(output, unit["unit_id"])
        if unit["status"] == "passed" and score is not None:
            candidate = spec["candidate"]
            scored.append((score, candidate["candidate_key"], candidate))
    return [candidate for _, _, candidate in sorted(scored, key=lambda row: (row[0], row[1]))]


def _resolve_candidate(checkpoint: dict[str, Any], unit: dict[str, Any]) -> dict[str, Any]:
    spec = unit["spec"]
    if "candidate" in spec:
        return spec["candidate"]
    if "resolved_candidate" in spec:
        return spec["resolved_candidate"]
    outer = int(spec["outer_fold"])
    if spec.get("select") == "coarse_rank":
        ranking = _coarse_ranking(checkpoint, outer)
        rank = int(spec["rank"])
        if rank >= len(ranking):
            raise CampaignError(f"only {len(ranking)} coarse candidates passed for outer fold {outer}")
        candidate = ranking[rank]
    elif spec.get("select") == "promoted_mean_winner":
        output = Path(checkpoint["output_root"])
        coarse = {item["candidate_key"]: item for item in _coarse_ranking(checkpoint, outer)[:6]}
        scores: dict[str, list[float]] = {key: [] for key in coarse}
        for candidate_key in scores:
            initial_id = f"coarse_o{outer}__{candidate_key}"
            value = _metric(output, initial_id)
            if value is not None:
                scores[candidate_key].append(value)
        for other in checkpoint["units"]:
            other_spec = other["spec"]
            resolved = other_spec.get("resolved_candidate")
            if (
                other["campaign_stage"] == "promoted_halving"
                and other_spec.get("outer_fold") == outer
                and isinstance(resolved, dict)
                and resolved["candidate_key"] in scores
            ):
                value = _metric(output, other["unit_id"])
                if other["status"] == "passed" and value is not None:
                    scores[resolved["candidate_key"]].append(value)
        complete = [(sum(values) / len(values), key) for key, values in scores.items() if len(values) == 3]
        if not complete:
            raise CampaignError(f"no candidate completed all inner folds for outer fold {outer}")
        candidate = coarse[min(complete)[1]]
    else:
        raise CampaignError(f"tree unit has no concrete candidate: {unit['unit_id']}")
    spec["resolved_candidate"] = candidate
    return candidate


def _chemprop_assignment(checkpoint: dict[str, Any], outer_fold: int) -> Path:
    import pandas as pd

    output = Path(checkpoint["output_root"])
    destination = output / "chemprop_assignments" / f"outer_{outer_fold}.csv"
    if destination.is_file():
        return destination
    splits = pd.read_parquet(output / "prepared" / "nested_scaffold_splits.parquet")
    selected = splits[splits["outer_fold"].eq(outer_fold)].copy()
    selected["inner_role"] = "train"
    selected.loc[selected["outer_role"].eq("heldout"), "inner_role"] = "holdout"
    selected.loc[selected["outer_role"].eq("fit") & selected["inner_fold"].eq(0), "inner_role"] = "validation"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    selected[["structure_id", "scaffold_group_id", "inner_role"]].to_csv(temporary, index=False)
    os.replace(temporary, destination)
    return destination


def _command(checkpoint: dict[str, Any], unit: dict[str, Any]) -> list[str]:
    python = checkpoint["python_path"]
    worker = checkpoint["worker_path"]
    output = Path(checkpoint["output_root"])
    prepared = output / "prepared"
    spec = unit["spec"]
    kind = unit["kind"]
    if kind == "prepare":
        return [
            python,
            worker,
            "prepare",
            "--repo-root",
            checkpoint["repo_root"],
            "--matrix-root",
            checkpoint["matrix_root"],
            "--observations",
            checkpoint["observations"],
            "--output-root",
            str(prepared),
            "--outer-folds",
            "5",
            "--inner-folds",
            "3",
        ]
    if kind in {"tree", "classical", "similarity"}:
        subcommand = {"tree": "tree-unit", "classical": "classical-unit", "similarity": "similarity-unit"}[
            kind
        ]
        command = [
            python,
            worker,
            subcommand,
            "--prepared-root",
            str(prepared),
            "--output-root",
            str(output),
            "--unit-id",
            unit["unit_id"],
            "--model-id",
            spec.get("model_id", "nested_selected"),
            "--stage",
            spec["stage"],
            "--outer-fold",
            str(spec["outer_fold"]),
            "--seed",
            str(spec["seed"]),
        ]
        if spec["stage"] == "inner":
            command.extend(["--inner-fold", str(spec["inner_fold"])])
        if kind == "tree":
            candidate = _resolve_candidate(checkpoint, unit)
            command[command.index("--model-id") + 1] = (
                "nested_selected"
                if spec.get("select") == "promoted_mean_winner"
                else candidate["candidate_key"]
            )
            command.extend(
                [
                    "--engine",
                    candidate["engine"],
                    "--groups",
                    candidate["groups"],
                    "--params-json",
                    json.dumps(candidate["params"], sort_keys=True, separators=(",", ":")),
                    "--workers",
                    str(checkpoint["workers"]),
                ]
            )
        elif kind == "classical":
            command.extend(
                [
                    "--model",
                    spec["model"],
                    "--groups",
                    spec["groups"],
                    "--params-json",
                    json.dumps(spec["params"], sort_keys=True),
                    "--workers",
                    str(checkpoint["workers"]),
                ]
            )
        else:
            command.extend(
                [
                    "--neighbors",
                    str(spec["neighbors"]),
                    "--similarity-floor",
                    str(spec["similarity_floor"]),
                    "--weight-power",
                    str(spec["weight_power"]),
                ]
            )
        return command
    if kind == "chemprop_export":
        return [
            python,
            worker,
            "chemprop-prepare",
            "--prepared-root",
            str(prepared),
            "--output-root",
            str(output / "chemprop_prepared"),
        ]
    if kind == "chemprop":
        assignment = _chemprop_assignment(checkpoint, int(spec["outer_fold"]))
        unit_spec = {
            "data_path": str(prepared / "exact_train_cache.parquet"),
            "assignments_path": str(assignment),
            "outer_fold": spec["outer_fold"],
            "seed": spec["seed"],
            "epochs": spec["epochs"],
            "patience": spec["patience"],
            "maximum_minutes": spec["maximum_minutes"],
        }
        return [
            python,
            checkpoint["chemprop_runner_path"],
            "--repo-root",
            checkpoint["repo_root"],
            "--prepared-root",
            str(prepared),
            "--output-root",
            str(output),
            "--unit-id",
            unit["unit_id"],
            "--unit-spec",
            json.dumps(unit_spec, sort_keys=True, separators=(",", ":")),
            "--workers",
            str(checkpoint["workers"]),
        ]
    if kind == "analyze":
        return [
            python,
            worker,
            "analyze",
            "--prepared-root",
            str(prepared),
            "--results-root",
            str(output),
            "--output-root",
            str(output / "analysis"),
            "--baseline-model",
            "diag_2d",
            "--bootstrap-replicates",
            "10000",
            "--minimum-subgroup-size",
            "50",
        ]
    raise CampaignError(f"unknown unit kind: {kind}")


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
            command, stdout=log, stderr=subprocess.STDOUT, text=True, env=environment, start_new_session=True
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


def _run_unit_with_retry(
    command: list[str],
    unit: dict[str, Any],
    output: Path,
    checkpoint: dict[str, Any],
    environment: dict[str, str],
    deadline: float,
) -> tuple[int, bool, float, Path]:
    """Run a computational unit at most twice, retaining both attempt logs."""
    elapsed_total = 0.0
    last_log = output / "logs" / f"{unit['unit_id']}.attempt_1.log"
    while unit["attempts"] < 2:
        unit["attempts"] += 1
        last_log = output / "logs" / f"{unit['unit_id']}.attempt_{unit['attempts']}.log"
        unit.setdefault("attempt_logs", []).append(str(last_log))
        unit["log_path"] = str(last_log)
        unit["started_utc"] = _utc()
        _checkpoint(output, checkpoint)
        returncode, timed_out, elapsed = _run_process(command, last_log, environment, deadline)
        elapsed_total += elapsed
        checkpoint["active_elapsed_seconds"] += elapsed
        checkpoint["invocations"][-1]["active_elapsed_seconds"] += elapsed
        unit.setdefault("attempt_results", []).append(
            {
                "attempt": unit["attempts"],
                "returncode": returncode,
                "timed_out": timed_out,
                "elapsed_seconds": elapsed,
                "log_path": str(last_log),
            }
        )
        # Code 3 is the governed Chemprop-unavailable result, not a transient
        # failure.  Successful work and exhausted hard deadlines also stop.
        if returncode in {0, 3} or time.time() >= deadline:
            return returncode, timed_out, elapsed_total, last_log
    return returncode, timed_out, elapsed_total, last_log


def _checkpoint(output: Path, value: dict[str, Any]) -> dict[str, Any]:
    value["updated_utc"] = _utc()
    return _atomic_json(output / "checkpoint.json", value, "checkpoint_sha256")


def _initial(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.repo_root).resolve()
    worker = Path(args.worker).resolve()
    chemprop = Path(args.chemprop_runner).resolve()
    matrix = Path(args.matrix_root).resolve()
    observations = Path(args.observations).resolve()
    bindings = [
        _binding(Path(__file__), "orchestrator"),
        _binding(worker, "worker"),
        _binding(chemprop, "optional_chemprop_runner"),
        _binding(observations, "exact_pic50_observations"),
        _binding(matrix / "combined_feature_matrix.parquet", "feature_matrix"),
        _binding(matrix / "manifest.json", "feature_matrix_manifest"),
        _binding(matrix / "validation.json", "feature_matrix_validation"),
    ]
    now = time.time()
    return {
        "schema_version": SCHEMA_VERSION,
        "created_utc": _utc(),
        "updated_utc": _utc(),
        "status": "running",
        "repo_root": str(root),
        "output_root": str(Path(args.output_root).resolve()),
        "worker_path": str(worker),
        "chemprop_runner_path": str(chemprop),
        # Do not resolve this symlink: on macOS, ``.venv/bin/python`` points at
        # the framework interpreter, and resolving it discards the virtual
        # environment context for every child worker.
        "python_path": str(Path(args.python).absolute()),
        "matrix_root": str(matrix),
        "observations": str(observations),
        "workers": args.workers,
        "target_hours": args.target_hours,
        "hard_hours": args.hard_hours,
        "reserve_minutes": args.finalization_reserve_minutes,
        "started_epoch": now,
        "active_elapsed_seconds": 0.0,
        "nominal_expected_minutes": sum(unit["expected_minutes"] for unit in _plan()),
        "invocations": [],
        "bindings": bindings,
        "scientific_contract": {
            "source_partition": "train_only",
            "target": "wild_type_or_target_unspecified_exact_pic50",
            "confirmed_wild_type_only": False,
            "repository_validation_outcomes_loaded": False,
            "repository_test_outcomes_loaded": False,
            "fixed_validation_previously_examined_elsewhere": True,
            "locked_test_remains_sealed": True,
            "local_diagnostic_not_production_or_superiority_evidence": True,
        },
        "resource_checks": [],
        "stop_requests": [],
        "units": _plan(),
    }


def _resume(args: argparse.Namespace, output: Path) -> dict[str, Any]:
    value = _read_json(output / "checkpoint.json", "checkpoint_sha256")
    _verify_bindings(value["bindings"])
    required = {
        "repo_root": str(Path(args.repo_root).resolve()),
        "output_root": str(output),
        "worker_path": str(Path(args.worker).resolve()),
        "chemprop_runner_path": str(Path(args.chemprop_runner).resolve()),
        "workers": args.workers,
    }
    for key, expected in required.items():
        if value.get(key) != expected:
            raise CampaignError(f"resume argument changed: {key}")
    for unit in value["units"]:
        if unit["status"] == "running":
            unit["status"] = "pending"
            unit["interrupted_before_checkpoint"] = True
        elif unit["status"] == "skipped_dependency":
            unit["status"] = "pending"
            unit["dependency_wait_reconsidered_on_resume"] = True
    return _checkpoint(output, value)


def _dependencies_passed(checkpoint: dict[str, Any], unit: dict[str, Any]) -> bool:
    kind = unit["kind"]
    if kind == "prepare":
        return True
    if not any(item["unit_id"] == "prepare" and item["status"] == "passed" for item in checkpoint["units"]):
        return False
    if unit["campaign_stage"] == "promoted_halving":
        outer = unit["spec"]["outer_fold"]
        return bool(_coarse_ranking(checkpoint, outer))
    if unit["campaign_stage"] == "nested_confirmation":
        outer = unit["spec"]["outer_fold"]
        return (
            sum(
                item["status"] == "passed"
                for item in checkpoint["units"]
                if item["campaign_stage"] == "promoted_halving" and item["spec"]["outer_fold"] == outer
            )
            >= 3
        )
    if kind == "chemprop":
        return any(
            item["unit_id"] == "chemprop_export" and item["status"] == "passed"
            for item in checkpoint["units"]
        )
    if kind == "analyze":
        nested = [item for item in checkpoint["units"] if item["campaign_stage"] == "nested_confirmation"]
        return len(nested) == 5 and all(item["status"] == "passed" for item in nested)
    return True


def _finalize(output: Path, checkpoint: dict[str, Any]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    failures: list[dict[str, Any]] = []
    for unit in checkpoint["units"]:
        counts[unit["status"]] = counts.get(unit["status"], 0) + 1
        if unit["status"].startswith("failed"):
            failures.append(
                {
                    key: unit.get(key)
                    for key in ("unit_id", "campaign_stage", "status", "returncode", "log_path")
                }
            )
    critical = [unit for unit in checkpoint["units"] if unit["critical"]]
    analyze = next(unit for unit in checkpoint["units"] if unit["kind"] == "analyze")
    analysis_artifacts = [output / "analysis" / "analysis.md", output / "analysis" / "validation.json"]
    terminal_complete = (
        all(unit["status"] == "passed" for unit in critical)
        and analyze["status"] == "passed"
        and all(path.is_file() for path in analysis_artifacts)
    )
    if not terminal_complete:
        checkpoint["status"] = "incomplete"
        checkpoint["incomplete_reason"] = (
            "completion requires every critical unit, all five nested confirmations, "
            "and validated analysis artifacts"
        )
        _checkpoint(output, checkpoint)
        return {
            "status": "incomplete",
            "reason": checkpoint["incomplete_reason"],
            "resume": "rerun the identical start command",
        }
    checkpoint["status"] = "complete_with_noncritical_failures" if failures else "complete"
    checkpoint["finished_utc"] = _utc()
    checkpoint = _checkpoint(output, checkpoint)
    summary = _atomic_json(
        output / "final_summary.json",
        {
            "schema_version": SCHEMA_VERSION,
            "status": checkpoint["status"],
            "created_utc": _utc(),
            "unit_status_counts": counts,
            "failure_ledger": failures,
            "scientific_contract": checkpoint["scientific_contract"],
            "checkpoint_sha256": checkpoint["checkpoint_sha256"],
            "completed_early_because_prespecified_work_finished": checkpoint["active_elapsed_seconds"]
            < checkpoint["target_hours"] * 3600,
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
        raise CampaignError("campaign governance requires exactly six workers")
    if not 0 < args.target_hours <= args.hard_hours:
        raise CampaignError("require 0 < target-hours <= hard-hours")
    if args.finalization_reserve_minutes < 0 or args.finalization_reserve_minutes >= args.hard_hours * 60:
        raise CampaignError("invalid finalization reserve")
    output = Path(args.output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with _Lock(output / ".campaign.lock"):
        checkpoint = (
            _resume(args, output)
            if (output / "checkpoint.json").is_file()
            else _checkpoint(output, _initial(args))
        )
        if checkpoint["status"].startswith("complete"):
            return {"status": checkpoint["status"], "message": "already complete"}
        active_remaining = max(0.0, checkpoint["hard_hours"] * 3600 - checkpoint["active_elapsed_seconds"])
        work_remaining = max(
            0.0,
            checkpoint["hard_hours"] * 3600
            - checkpoint["reserve_minutes"] * 60
            - checkpoint["active_elapsed_seconds"],
        )
        invocation_started = time.time()
        invocation = {
            "invocation_index": len(checkpoint["invocations"]) + 1,
            "started_utc": _utc(),
            "active_elapsed_seconds_before": checkpoint["active_elapsed_seconds"],
            "active_elapsed_seconds": 0.0,
            "work_deadline_epoch": invocation_started + work_remaining,
            "hard_deadline_epoch": invocation_started + active_remaining,
            "downtime_excluded_from_active_budget": True,
        }
        checkpoint["invocations"].append(invocation)
        _checkpoint(output, checkpoint)
        stop = False

        def request_stop(signum: int, _frame: object) -> None:
            nonlocal stop
            stop = True
            checkpoint["stop_requests"].append(
                {"signal": signum, "received_utc": _utc(), "policy": "after_current_unit"}
            )

        old = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)}
        environment = os.environ.copy()
        environment.update(
            {
                "PYTHONDONTWRITEBYTECODE": "1",
                "OMP_NUM_THREADS": str(args.workers),
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "VECLIB_MAXIMUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
                "HERG_CAMPAIGN_FORBID_REPOSITORY_VALIDATION_TEST": "1",
            }
        )
        try:
            # Dependency waits remain pending and are reconsidered within this
            # invocation and on every resume.  Normally one pass suffices;
            # repeated passes protect against reordered or recovered units.
            progress = True
            while progress:
                progress = False
                for unit in checkpoint["units"]:
                    if unit["status"] == "skipped_dependency":
                        unit["status"] = "pending"
                    if unit["status"] in TERMINAL:
                        continue
                    if stop:
                        checkpoint["status"] = "safely_stopped"
                        _checkpoint(output, checkpoint)
                        return {"status": "safely_stopped", "resume": "rerun the identical start command"}
                    if not _dependencies_passed(checkpoint, unit):
                        unit["dependency_wait_count"] = unit.get("dependency_wait_count", 0) + 1
                        continue
                    final = unit["kind"] == "analyze"
                    deadline = (
                        invocation["hard_deadline_epoch"] if final else invocation["work_deadline_epoch"]
                    )
                    remaining = (deadline - time.time()) / 60
                    if remaining <= 0 or (not final and unit["expected_minutes"] > remaining):
                        unit["status"] = "skipped_time"
                        _checkpoint(output, checkpoint)
                        continue
                    try:
                        resource = _resource_gate(
                            output, args.minimum_free_disk_gib, args.minimum_available_memory_gib
                        )
                    except CampaignError as error:
                        checkpoint["status"] = "safely_stopped_resource_gate"
                        checkpoint["resource_error"] = str(error)
                        _checkpoint(output, checkpoint)
                        return {"status": checkpoint["status"], "error": str(error)}
                    resource["before_unit"] = unit["unit_id"]
                    checkpoint["resource_checks"].append(resource)
                    try:
                        command = _command(checkpoint, unit)
                    except CampaignError as error:
                        unit["status"] = "failed_critical" if unit["critical"] else "failed_noncritical"
                        unit["error"] = str(error)
                        _checkpoint(output, checkpoint)
                        progress = True
                        continue
                    unit["status"] = "running"
                    unit["started_utc"] = _utc()
                    _checkpoint(output, checkpoint)
                    returncode, timed_out, elapsed, log = _run_unit_with_retry(
                        command, unit, output, checkpoint, environment, deadline
                    )
                    unit.update(
                        returncode=returncode,
                        timed_out=timed_out,
                        elapsed_seconds=elapsed,
                        finished_utc=_utc(),
                    )
                    if returncode == 0:
                        unit["status"] = "passed"
                    elif unit["kind"] == "chemprop" and returncode == 3:
                        unit["status"] = "skipped_unavailable"
                    else:
                        unit["status"] = "failed_critical" if unit["critical"] else "failed_noncritical"
                    _checkpoint(output, checkpoint)
                    progress = True
                    if unit["unit_id"] == "prepare" and unit["status"] == "failed_critical":
                        checkpoint["status"] = "failed"
                        _checkpoint(output, checkpoint)
                        return {"status": "failed", "unit": "prepare", "log": str(log)}
            return _finalize(output, checkpoint)
        finally:
            for sig, handler in old.items():
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
        "updated_utc": checkpoint["updated_utc"],
        "output_root": checkpoint["output_root"],
    }


def _validate(output: Path, require_done: bool = True) -> dict[str, Any]:
    output = output.resolve()
    checkpoint = _read_json(output / "checkpoint.json", "checkpoint_sha256")
    _verify_bindings(checkpoint["bindings"])
    if checkpoint["schema_version"] != SCHEMA_VERSION or checkpoint["workers"] != DEFAULT_WORKERS:
        raise CampaignError("campaign schema or worker governance mismatch")
    contract = checkpoint["scientific_contract"]
    if contract["repository_validation_outcomes_loaded"] or contract["repository_test_outcomes_loaded"]:
        raise CampaignError("label-blindness contract failed")
    for unit in checkpoint["units"]:
        if unit["status"] in {"passed", "failed_noncritical", "failed_critical", "skipped_unavailable"}:
            if not Path(unit.get("log_path", "")).is_file():
                raise CampaignError(f"missing unit log: {unit['unit_id']}")
    if require_done:
        if checkpoint["status"] not in {"complete", "complete_with_noncritical_failures"}:
            raise CampaignError(f"campaign is not complete: {checkpoint['status']}")
        nested = [unit for unit in checkpoint["units"] if unit["campaign_stage"] == "nested_confirmation"]
        if len(nested) != 5 or not all(unit["status"] == "passed" for unit in nested):
            raise CampaignError("all five nested confirmations must pass before terminal validation")
        analyze = next(unit for unit in checkpoint["units"] if unit["kind"] == "analyze")
        if analyze["status"] != "passed" or not (output / "analysis" / "analysis.md").is_file():
            raise CampaignError("successful final analysis artifact is required")
        done = _read_json(output / "DONE.json", "done_sha256")
        summary = _read_json(output / "final_summary.json", "summary_sha256")
        if (
            done["checkpoint_sha256"] != checkpoint["checkpoint_sha256"]
            or done["summary_sha256"] != summary["summary_sha256"]
        ):
            raise CampaignError("terminal files do not bind the final checkpoint")
    return {
        "status": "passed",
        "campaign_status": checkpoint["status"],
        "units": len(checkpoint["units"]),
        "validation_labels_opened": False,
        "test_labels_opened": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    start = commands.add_parser("start")
    start.add_argument("--repo-root", default=".")
    start.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    start.add_argument(
        "--worker", default=str(Path(__file__).with_name("run_local_herg_discovery_worker.py"))
    )
    start.add_argument(
        "--chemprop-runner", default=str(Path(__file__).with_name("run_local_herg_chemprop_unit.py"))
    )
    start.add_argument("--python", default=sys.executable)
    start.add_argument("--matrix-root", default=str(DEFAULT_MATRIX))
    start.add_argument("--observations", default=str(DEFAULT_OBSERVATIONS))
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
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        if args.action == "start":
            result = _start(args)
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
        json.JSONDecodeError,
        subprocess.SubprocessError,
    ) as error:
        print(json.dumps({"status": "failed", "error": str(error)}, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
