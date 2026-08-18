#!/usr/bin/env python3
"""Run one isolated, resumable, train-only Chemprop hERG CV unit.

The adapter deliberately never reads repository validation or test outcomes.  It
expects a prepared exact-pIC50 *training* surface and explicit scaffold-fold (or
inner train/validation/holdout) assignments.  Chemprop's internal ``test`` name
is used only for the selected training-only OOF holdout.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

SCHEMA_VERSION = "platform-local-herg-chemprop-unit/1.0"
PREDICTION_SCHEMA_VERSION = "platform-local-herg-chemprop-oof/1.0"
DATA_CANDIDATES = (
    "prepared_exact_train.parquet",
    "prepared_exact_train.csv",
    "chemprop_prepared_exact_train.parquet",
    "chemprop_prepared_exact_train.csv",
    "chemprop_train.parquet",
    "chemprop_train.csv",
)
ASSIGNMENT_CANDIDATES = (
    "scaffold_inner_assignments.parquet",
    "scaffold_inner_assignments.csv",
    "scaffold_inner_assignments.json",
    "inner_fold_assignments.parquet",
    "inner_fold_assignments.csv",
    "inner_fold_assignments.json",
    "fold_assignments.parquet",
    "fold_assignments.csv",
    "fold_assignments.json",
)
ID_ALIASES = ("structure_id", "molecule_id", "compound_id")
SMILES_ALIASES = ("standardized_smiles", "smiles", "canonical_smiles")
TARGET_ALIASES = ("target_pic50", "potency_pic50_point", "pic50", "pIC50")
SCAFFOLD_ALIASES = ("scaffold_group_id", "scaffold_id", "scaffold_group", "murcko_scaffold")
FOLD_ALIASES = ("inner_fold", "scaffold_fold", "fold_id", "fold")
ROLE_ALIASES = ("inner_role", "cv_role", "assignment", "partition")
TERMINAL_SUCCESS = {"passed", "skipped_validated_complete"}


class ChempropUnitError(RuntimeError):
    """Raised when the governed Chemprop unit contract is violated."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _self_hashed(value: dict[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(value)
    payload.pop("unit_json_sha256", None)
    payload["unit_json_sha256"] = _digest(payload)
    return payload


def _atomic_json(path: Path, value: dict[str, Any], self_hash: bool = False) -> dict[str, Any]:
    payload = _self_hashed(value) if self_hash else value
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return payload


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    table = pa.Table.from_pandas(frame, preserve_index=False)
    pq.write_table(table, temporary, compression="zstd")
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _binding(path: Path, role: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "role": role,
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }
    if path.suffix == ".parquet":
        result["rows"] = pq.read_metadata(path).num_rows
        result["arrow_schema_sha256"] = hashlib.sha256(
            pq.read_schema(path).serialize().to_pybytes()
        ).hexdigest()
    return result


def _verify_binding(binding: dict[str, Any]) -> bool:
    path = Path(str(binding["path"]))
    if not path.is_file() or path.stat().st_size != int(binding["bytes"]):
        return False
    if _sha256(path) != binding["sha256"]:
        return False
    return not (path.suffix == ".parquet" and pq.read_metadata(path).num_rows != binding.get("rows"))


def _read_frame(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix in {".csv", ".tsv"}:
        return pd.read_csv(path, sep="\t" if path.suffix == ".tsv" else ",")
    if path.suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, list):
            return pd.DataFrame(value)
        if isinstance(value, dict):
            for key in ("assignments", "fold_assignments", "rows"):
                if isinstance(value.get(key), list):
                    return pd.DataFrame(value[key])
            if value and all(not isinstance(item, (dict, list)) for item in value.values()):
                return pd.DataFrame({"structure_id": list(value), "inner_fold": list(value.values())})
        raise ChempropUnitError(f"unsupported assignment JSON structure: {path}")
    raise ChempropUnitError(f"unsupported prepared input format: {path}")


def _specified_path(prepared: Path, spec: dict[str, Any], key: str) -> Path | None:
    raw = spec.get(key)
    if raw is None:
        return None
    path = Path(str(raw))
    return path.resolve() if path.is_absolute() else (prepared / path).resolve()


def _find_input(prepared: Path, spec: dict[str, Any], key: str, candidates: tuple[str, ...]) -> Path:
    specified = _specified_path(prepared, spec, key)
    if specified is not None:
        if not specified.is_file():
            raise ChempropUnitError(f"{key} does not exist: {specified}")
        return specified
    matches = [prepared / candidate for candidate in candidates if (prepared / candidate).is_file()]
    if len(matches) != 1:
        raise ChempropUnitError(
            f"expected exactly one {key}; found {[path.name for path in matches]}. Set {key} in --unit-spec."
        )
    return matches[0].resolve()


def _column(frame: pd.DataFrame, aliases: tuple[str, ...], purpose: str) -> str:
    matches = [name for name in aliases if name in frame.columns]
    if not matches:
        raise ChempropUnitError(f"prepared input has no {purpose} column; accepted aliases={aliases}")
    return matches[0]


def _normalize_roles(values: pd.Series) -> pd.Series:
    aliases = {
        "train": "train",
        "inner_train": "train",
        "val": "validation",
        "validation": "validation",
        "inner_val": "validation",
        "inner_validation": "validation",
        "holdout": "holdout",
        "inner_holdout": "holdout",
        "oof": "holdout",
    }
    lowered = values.astype(str).str.strip().str.lower()
    forbidden = sorted(set(lowered) & {"test", "repo_test", "repo_validation"})
    if forbidden:
        raise ChempropUnitError(f"repository-style validation/test roles are forbidden: {forbidden}")
    unknown = sorted(set(lowered) - set(aliases))
    if unknown:
        raise ChempropUnitError(f"unknown inner assignment roles: {unknown}")
    return lowered.map(aliases)


def _resolve_roles(frame: pd.DataFrame, spec: dict[str, Any]) -> tuple[pd.Series, dict[str, Any]]:
    role_column = next((name for name in ROLE_ALIASES if name in frame.columns), None)
    if role_column is not None:
        roles = _normalize_roles(frame[role_column])
        if set(roles) != {"train", "validation", "holdout"}:
            raise ChempropUnitError("explicit roles must contain inner train, validation, and holdout")
        return roles, {"method": "explicit_inner_roles", "column": role_column}

    fold_column = next((name for name in FOLD_ALIASES if name in frame.columns), None)
    if fold_column is None:
        raise ChempropUnitError("explicit scaffold fold or inner-role assignments are required")
    folds = pd.to_numeric(frame[fold_column], errors="raise").astype(int)
    unique = sorted(set(folds))
    if len(unique) < 3:
        raise ChempropUnitError("at least three explicit scaffold folds are required for train/val/OOF")
    holdout = int(spec.get("outer_fold", spec.get("holdout_fold", unique[0])))
    if holdout not in unique:
        raise ChempropUnitError(f"holdout fold {holdout} is absent from explicit assignments")
    remaining = [fold for fold in unique if fold != holdout]
    validation = int(spec.get("validation_fold", remaining[holdout % len(remaining)]))
    if validation not in remaining:
        raise ChempropUnitError("validation_fold must differ from and coexist with holdout fold")
    roles = pd.Series("train", index=frame.index, dtype="object")
    roles.loc[folds.eq(validation)] = "validation"
    roles.loc[folds.eq(holdout)] = "holdout"
    return roles, {
        "method": "explicit_scaffold_folds",
        "column": fold_column,
        "holdout_fold": holdout,
        "validation_fold": validation,
        "available_folds": unique,
    }


def _prepare_inputs(
    prepared_root: Path, spec: dict[str, Any]
) -> tuple[pd.DataFrame, Path, Path | None, dict[str, Any]]:
    data_path = _find_input(prepared_root, spec, "data_path", DATA_CANDIDATES)
    data = _read_frame(data_path)
    assignment_path = _specified_path(prepared_root, spec, "assignments_path")
    if assignment_path is None:
        matches = [
            prepared_root / candidate
            for candidate in ASSIGNMENT_CANDIDATES
            if (prepared_root / candidate).is_file()
        ]
        if len(matches) > 1:
            raise ChempropUnitError("multiple assignment files found; set assignments_path")
        assignment_path = matches[0].resolve() if matches else None

    id_column = _column(data, ID_ALIASES, "structure identifier")
    smiles_column = _column(data, SMILES_ALIASES, "SMILES")
    target_column = _column(data, TARGET_ALIASES, "exact pIC50 target")
    scaffold_column = next((name for name in SCAFFOLD_ALIASES if name in data.columns), None)
    if "model_split" in data.columns and set(data["model_split"].astype(str).str.lower()) != {"train"}:
        raise ChempropUnitError("prepared surface contains nontraining model_split rows")
    if "source_partition" in data.columns and set(data["source_partition"].astype(str).str.lower()) != {
        "train"
    }:
        raise ChempropUnitError("prepared surface contains nontraining source_partition rows")

    frame = data.copy()
    if assignment_path is not None:
        assignments = _read_frame(assignment_path)
        assignment_id = _column(assignments, ID_ALIASES, "assignment structure identifier")
        if "unit_id" in assignments.columns and spec.get("unit_id") is not None:
            assignments = assignments.loc[assignments["unit_id"].astype(str).eq(str(spec["unit_id"]))]
        keep = [assignment_id]
        keep.extend(name for name in (*ROLE_ALIASES, *FOLD_ALIASES, *SCAFFOLD_ALIASES) if name in assignments)
        assignments = assignments[keep].copy()
        if assignments[assignment_id].duplicated().any():
            raise ChempropUnitError("assignment file has duplicate structure identifiers")
        frame = frame.merge(
            assignments,
            left_on=id_column,
            right_on=assignment_id,
            how="left",
            validate="one_to_one",
            suffixes=("", "__assignment"),
        )
        if frame[assignment_id].isna().any():
            raise ChempropUnitError("not every prepared structure has an explicit assignment")
        if scaffold_column is None:
            scaffold_column = next((name for name in SCAFFOLD_ALIASES if name in frame), None)

    if scaffold_column is None:
        raise ChempropUnitError("a scaffold group is required to verify scaffold separation")
    frame["structure_id"] = frame[id_column].astype(str)
    frame["smiles"] = frame[smiles_column].astype(str).str.strip()
    frame["target_pic50"] = pd.to_numeric(frame[target_column], errors="coerce")
    frame["scaffold_group_id"] = frame[scaffold_column].astype(str)
    if frame["structure_id"].duplicated().any():
        raise ChempropUnitError("prepared surface must have one row per structure")
    if frame["smiles"].eq("").any() or frame["target_pic50"].isna().any():
        raise ChempropUnitError("SMILES and exact pIC50 targets must be complete")
    if not np.isfinite(frame["target_pic50"].to_numpy(dtype=float)).all():
        raise ChempropUnitError("nonfinite target detected")

    roles, assignment = _resolve_roles(frame, spec)
    frame["inner_role"] = roles
    scaffold_roles = frame.groupby("scaffold_group_id")["inner_role"].nunique()
    if int(scaffold_roles.max()) != 1:
        raise ChempropUnitError("a scaffold group crosses inner train/validation/holdout roles")
    counts = frame["inner_role"].value_counts().to_dict()
    if any(int(counts.get(role, 0)) < 1 for role in ("train", "validation", "holdout")):
        raise ChempropUnitError("all three inner roles must be nonempty")
    selected = frame[["structure_id", "smiles", "target_pic50", "scaffold_group_id", "inner_role"]].copy()
    metadata = {
        "data_path": str(data_path),
        "assignments_path": str(assignment_path) if assignment_path else None,
        "assignment": assignment,
        "role_counts": {key: int(value) for key, value in counts.items()},
        "scaffold_counts": {
            key: int(selected.loc[selected["inner_role"].eq(key), "scaffold_group_id"].nunique())
            for key in ("train", "validation", "holdout")
        },
    }
    return selected, data_path, assignment_path, metadata


def _bounded_spec(raw: dict[str, Any], unit_id: str) -> dict[str, Any]:
    spec = dict(raw)
    spec["unit_id"] = unit_id
    values: dict[str, Any] = {
        "seed": int(spec.get("seed", 20260811)),
        "outer_fold": int(spec.get("outer_fold", spec.get("holdout_fold", 0))),
        "epochs": int(spec.get("epochs", 60)),
        "patience": int(spec.get("patience", 10)),
        "batch_size": int(spec.get("batch_size", 64)),
        "depth": int(spec.get("depth", 3)),
        "hidden_dim": int(spec.get("hidden_dim", 300)),
        "dropout": float(spec.get("dropout", 0.1)),
        "loss": str(spec.get("loss", "mse")).lower(),
        "maximum_minutes": float(spec.get("maximum_minutes", 300.0)),
    }
    bounds: dict[str, tuple[float, float]] = {
        "epochs": (1, 200),
        "patience": (1, 50),
        "batch_size": (8, 1024),
        "depth": (1, 8),
        "hidden_dim": (32, 2048),
        "dropout": (0.0, 0.7),
        "maximum_minutes": (1.0, 480.0),
    }
    for key, (lower, upper) in bounds.items():
        if not lower <= values[key] <= upper:
            raise ChempropUnitError(f"{key} must be between {lower} and {upper}")
    if values["patience"] > values["epochs"]:
        raise ChempropUnitError("patience cannot exceed epochs")
    if values["loss"] not in {"mse", "mae", "rmse"}:
        raise ChempropUnitError("loss must be mse, mae, or rmse")
    passthrough: dict[str, Any] = {
        key: spec[key] for key in ("data_path", "assignments_path", "validation_fold") if key in spec
    }
    return {**values, **passthrough, "unit_id": unit_id}


def _chemprop_executable(repo_root: Path) -> Path | None:
    local = repo_root / ".venv" / "bin" / "chemprop"
    if local.is_file() and os.access(local, os.X_OK):
        return local.resolve()
    found = shutil.which("chemprop")
    return Path(found).resolve() if found else None


def _packages() -> list[dict[str, str]]:
    return sorted(
        (
            {"name": dist.metadata["Name"] or "unknown", "version": dist.version}
            for dist in importlib.metadata.distributions()
        ),
        key=lambda row: row["name"].lower(),
    )


def _capabilities(repo_root: Path) -> dict[str, Any]:
    executable = _chemprop_executable(repo_root)
    try:
        version = importlib.metadata.version("chemprop")
    except importlib.metadata.PackageNotFoundError:
        version = None
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "available" if executable is not None and version is not None else "unavailable",
        "operation": "isolated_chemprop_train_only_scaffold_oof_unit",
        "chemprop_executable": str(executable) if executable else None,
        "chemprop_version": version,
        "cpu_only": True,
        "macos_num_workers": 0,
        "accepted_data_formats": ["csv", "parquet"],
        "required_scientific_scope": "repository_train_partition_only",
        "outputs": ["unit.json", "oof_predictions.parquet"],
    }


def _metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    error = predicted - observed
    residual_sum = float(np.square(error).sum())
    total_sum = float(np.square(observed - observed.mean()).sum())
    return {
        "n": int(len(observed)),
        "mae": float(np.abs(error).mean()),
        "rmse": float(math.sqrt(np.square(error).mean())),
        "r2": float(1.0 - residual_sum / total_sum) if total_sum else 0.0,
        "pearson": float(pd.Series(observed).corr(pd.Series(predicted), method="pearson")),
        "spearman": float(pd.Series(observed).corr(pd.Series(predicted), method="spearman")),
        "within_0_5_log_fraction": float((np.abs(error) <= 0.5).mean()),
        "within_1_0_log_fraction": float((np.abs(error) <= 1.0).mean()),
    }


def _validated_complete(unit_json: Path, spec_sha: str, inputs: list[dict[str, Any]]) -> bool:
    if not unit_json.is_file():
        return False
    try:
        payload = json.loads(unit_json.read_text(encoding="utf-8"))
        if payload.get("unit_json_sha256") != _self_hashed(payload)["unit_json_sha256"]:
            return False
        if payload.get("status") != "passed" or payload.get("resolved_spec_sha256") != spec_sha:
            return False
        recorded = {(item["role"], item["sha256"]) for item in payload.get("inputs", [])}
        if recorded != {(item["role"], item["sha256"]) for item in inputs}:
            return False
        return all(_verify_binding(item) for item in payload.get("artifacts", []))
    except (OSError, KeyError, ValueError, json.JSONDecodeError):
        return False


def _write_terminal(
    unit_json: Path,
    *,
    status: str,
    unit_id: str,
    spec: dict[str, Any],
    started: str,
    inputs: list[dict[str, Any]],
    error: str | None = None,
    extras: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "unit_id": unit_id,
        "started_utc": started,
        "finished_utc": _utc_now(),
        "resolved_spec": spec,
        "resolved_spec_sha256": _digest(spec),
        "inputs": inputs,
        "scientific_contract": {
            "data_scope": "exact_pic50_repository_train_partition_only",
            "scaffold_separated_inner_oof": True,
            "repository_validation_labels_opened": False,
            "repository_test_labels_opened": False,
            "chemprop_test_name_means_training_only_inner_holdout": True,
        },
    }
    if error is not None:
        payload["error"] = error
    if extras:
        payload.update(extras)
    return _atomic_json(unit_json, payload, self_hash=True)


def _run(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    started = _utc_now()
    root = Path(args.repo_root).resolve()
    prepared = Path(args.prepared_root).resolve()
    output = Path(args.output_root).resolve()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.unit_id):
        raise ChempropUnitError("unit-id may contain only letters, numbers, dot, underscore, and hyphen")
    raw_spec = json.loads(args.unit_spec)
    if not isinstance(raw_spec, dict):
        raise ChempropUnitError("unit-spec must be a JSON object")
    spec = _bounded_spec(raw_spec, args.unit_id)
    unit_root = output / "chemprop_units" / args.unit_id
    unit_root.mkdir(parents=True, exist_ok=True)
    unit_json = unit_root / "unit.json"

    try:
        frame, data_path, assignment_path, preparation = _prepare_inputs(prepared, spec)
        inputs = [_binding(data_path, "prepared_exact_pic50_train_surface")]
        if assignment_path is not None and assignment_path != data_path:
            inputs.append(_binding(assignment_path, "explicit_scaffold_inner_assignments"))
        inputs.append(_binding(Path(__file__).resolve(), "chemprop_unit_runner"))
    except (ChempropUnitError, OSError, ValueError, KeyError) as error:
        payload = _write_terminal(
            unit_json,
            status="failed",
            unit_id=args.unit_id,
            spec=spec,
            started=started,
            inputs=[],
            error=str(error),
            extras={"failure_phase": "input_contract"},
        )
        return payload, 2

    spec_sha = _digest(spec)
    if _validated_complete(unit_json, spec_sha, inputs):
        return {
            "status": "skipped_validated_complete",
            "unit_id": args.unit_id,
            "unit_json": str(unit_json),
            "oof_predictions": str(unit_root / "oof_predictions.parquet"),
        }, 0

    executable = _chemprop_executable(root)
    capabilities = _capabilities(root)
    if executable is None or capabilities["chemprop_version"] is None:
        payload = _write_terminal(
            unit_json,
            status="unavailable",
            unit_id=args.unit_id,
            spec=spec,
            started=started,
            inputs=inputs,
            error="Chemprop package and executable are required in the existing environment",
            extras={"capabilities": capabilities, "preparation": preparation},
        )
        return payload, 3

    attempt = unit_root / "attempts" / f"{int(time.time())}-{os.getpid()}"
    training_root = attempt / "training"
    attempt.mkdir(parents=True, exist_ok=False)
    staged_csv = attempt / "chemprop_train_only_inner_cv.csv"
    chemprop_frame = frame[["structure_id", "smiles", "target_pic50", "inner_role"]].copy()
    chemprop_frame["__chemprop_split"] = chemprop_frame["inner_role"].map(
        {"train": "train", "validation": "val", "holdout": "test"}
    )
    chemprop_frame.to_csv(staged_csv, index=False)
    log_path = attempt / "chemprop.log"
    environment_path = attempt / "environment.json"
    package_inventory = _packages()
    controlled_environment = {
        "PYTHONHASHSEED": str(spec["seed"]),
        "OMP_NUM_THREADS": str(args.workers),
        "OPENBLAS_NUM_THREADS": str(args.workers),
        "MKL_NUM_THREADS": str(args.workers),
        "VECLIB_MAXIMUM_THREADS": str(args.workers),
        "NUMEXPR_NUM_THREADS": str(args.workers),
        "MPLCONFIGDIR": str(attempt / "cache" / "matplotlib"),
        "XDG_CACHE_HOME": str(attempt / "cache"),
        "TOKENIZERS_PARALLELISM": "false",
    }
    environment_record = {
        "created_utc": _utc_now(),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "packages": package_inventory,
        "packages_sha256": _digest(package_inventory),
        "controlled_environment": controlled_environment,
        "chemprop_capabilities": capabilities,
    }
    _atomic_json(environment_path, environment_record)
    command = [
        str(executable),
        "train",
        "--data-path",
        str(staged_csv),
        "--output-dir",
        str(training_root),
        "--smiles-columns",
        "smiles",
        "--target-columns",
        "target_pic50",
        "--ignore-columns",
        "structure_id",
        "inner_role",
        "--splits-column",
        "__chemprop_split",
        "--task-type",
        "regression",
        "--loss-function",
        str(spec["loss"]),
        "--metrics",
        "mae",
        "rmse",
        "r2",
        "--epochs",
        str(spec["epochs"]),
        "--patience",
        str(spec["patience"]),
        "--batch-size",
        str(spec["batch_size"]),
        "--depth",
        str(spec["depth"]),
        "--message-hidden-dim",
        str(spec["hidden_dim"]),
        "--dropout",
        str(spec["dropout"]),
        "--data-seed",
        str(spec["seed"]),
        "--pytorch-seed",
        str(spec["seed"]),
        "--accelerator",
        "cpu",
        "--devices",
        "1",
        "--num-workers",
        "0",
        "--ensemble-size",
        "1",
    ]
    command_record = {
        "argv": command,
        "argv_sha256": _digest(command),
        "cwd": str(root),
        "timeout_seconds": spec["maximum_minutes"] * 60.0,
    }
    _atomic_json(attempt / "command.json", command_record)
    env = os.environ.copy()
    env.update(controlled_environment)
    timed_out = False
    started_monotonic = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=root,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        try:
            returncode = process.wait(timeout=spec["maximum_minutes"] * 60.0)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                returncode = process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                returncode = process.wait()
    elapsed = time.monotonic() - started_monotonic
    runtime = {"returncode": returncode, "timed_out": timed_out, "elapsed_seconds": elapsed}
    if returncode != 0:
        payload = _write_terminal(
            unit_json,
            status="failed",
            unit_id=args.unit_id,
            spec=spec,
            started=started,
            inputs=inputs,
            error=f"Chemprop exited with return code {returncode}",
            extras={
                "failure_phase": "chemprop_subprocess",
                "capabilities": capabilities,
                "preparation": preparation,
                "command": command_record,
                "runtime": runtime,
                "artifacts": [_binding(log_path, "chemprop_log"), _binding(environment_path, "environment")],
            },
        )
        return payload, 2

    prediction_files = sorted(training_root.rglob("test_predictions.csv"))
    if len(prediction_files) != 1:
        payload = _write_terminal(
            unit_json,
            status="failed",
            unit_id=args.unit_id,
            spec=spec,
            started=started,
            inputs=inputs,
            error=f"expected one Chemprop test_predictions.csv, found {len(prediction_files)}",
            extras={"failure_phase": "prediction_collection", "runtime": runtime},
        )
        return payload, 2
    raw_predictions = pd.read_csv(prediction_files[0])
    prediction_column = "target_pic50"
    if prediction_column not in raw_predictions:
        candidates = [name for name in raw_predictions if name not in {"smiles", "structure_id"}]
        if len(candidates) != 1:
            raise ChempropUnitError("cannot identify the Chemprop prediction column")
        prediction_column = candidates[0]
    holdout = frame.loc[frame["inner_role"].eq("holdout")].reset_index(drop=True)
    if len(raw_predictions) != len(holdout):
        raise ChempropUnitError("Chemprop prediction row count differs from explicit inner holdout")
    if "smiles" in raw_predictions and not raw_predictions["smiles"].astype(str).equals(holdout["smiles"]):
        raise ChempropUnitError("Chemprop prediction order/SMILES differs from explicit holdout")
    predicted = pd.to_numeric(raw_predictions[prediction_column], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(predicted).all():
        raise ChempropUnitError("Chemprop emitted nonfinite predictions")
    observed = holdout["target_pic50"].to_numpy(dtype=float)
    oof = pd.DataFrame(
        {
            "schema_version": PREDICTION_SCHEMA_VERSION,
            "unit_id": args.unit_id,
            "structure_id": holdout["structure_id"],
            "scaffold_group_id": holdout["scaffold_group_id"],
            "source_partition": "train",
            "inner_role": "holdout",
            "outer_fold": spec["outer_fold"],
            "observed_pic50": observed,
            "predicted_pic50": predicted,
            "error_pic50": predicted - observed,
            "absolute_error_pic50": np.abs(predicted - observed),
        }
    )
    oof_path = unit_root / "oof_predictions.parquet"
    _atomic_parquet(oof_path, oof)
    artifact_paths = [oof_path, log_path, environment_path, attempt / "command.json", prediction_files[0]]
    artifact_paths.extend(sorted(training_root.rglob("*.ckpt")))
    artifact_paths.extend(sorted(training_root.rglob("*.pt")))
    artifacts = [_binding(path, "artifact") for path in dict.fromkeys(artifact_paths)]
    payload = _write_terminal(
        unit_json,
        status="passed",
        unit_id=args.unit_id,
        spec=spec,
        started=started,
        inputs=inputs,
        extras={
            "capabilities": capabilities,
            "preparation": preparation,
            "command": command_record,
            "runtime": runtime,
            "metrics": _metrics(observed, predicted),
            "artifacts": artifacts,
            "oof_predictions_path": str(oof_path),
        },
    )
    return payload, 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--capabilities-json",
        nargs="?",
        const="-",
        help="write capability JSON to PATH, or stdout when passed without PATH",
    )
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--prepared-root")
    parser.add_argument("--output-root")
    parser.add_argument("--unit-id")
    parser.add_argument("--unit-spec")
    parser.add_argument("--workers", type=int, default=6)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.capabilities_json is not None:
        payload = _capabilities(Path(args.repo_root).resolve())
        if args.capabilities_json == "-":
            print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
        else:
            _atomic_json(Path(args.capabilities_json).resolve(), payload)
        return 0
    missing = [
        name for name in ("prepared_root", "output_root", "unit_id", "unit_spec") if not getattr(args, name)
    ]
    if missing:
        print(
            json.dumps({"status": "failed", "error": f"missing required arguments: {missing}"}),
            file=sys.stderr,
        )
        return 2
    if args.workers < 1:
        print(json.dumps({"status": "failed", "error": "workers must be positive"}), file=sys.stderr)
        return 2
    try:
        result, returncode = _run(args)
        stream = sys.stdout if returncode in {0, 3} else sys.stderr
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False), file=stream)
        return returncode
    except (ChempropUnitError, OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(json.dumps({"status": "failed", "error": str(error)}, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
