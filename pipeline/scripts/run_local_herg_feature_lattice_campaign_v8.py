#!/usr/bin/env python3
"""Run the exhaustive, train-only hERG feature-coalition campaign V8.

V8 evaluates every coalition of eleven prespecified molecular feature blocks
(2**11 = 2,048 coalitions) inside each of five outer scaffold contexts.  A
fast, fixed LightGBM screen is followed by deterministic diversity-aware
promotion, material XGBoost/LightGBM parameter search, and one untouched outer
evaluation per context.  The complete lattice supports exact block Shapley
values, pairwise Banzhaf synergies, and molecule-level feature benefit maps.

All outcomes come only from the canonical exact-pIC50 training partition.
Repository validation and test labels are never opened.  Results are internal
nested-scaffold evidence, not prospective validation or causal biology.
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
import sys
import time
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from lightgbm import LGBMRegressor
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import DMatrix, XGBRegressor

SCHEMA_VERSION = "platform-local-herg-feature-lattice-campaign-v8/1.0"
SEED = 20260815
EXACT_ROWS = 18_801
OUTERS = tuple(range(5))
INNERS = tuple(range(3))
SCREEN_SHARD_SIZE = 32
MAX_ACTIVE_HOURS = 48.0
FINALIZATION_RESERVE_SECONDS = 90 * 60
MIN_FREE_DISK_GIB = 20.0
MAX_OUTPUT_GIB = 14.0
PERMUTATION_REPEATS = 30
PROMOTED_COALITIONS = 48

DEFAULT_MATRIX_ROOT = Path("research/local_runs/herg_fundamental_optimization_v6")
DEFAULT_BASE_ROOT = Path("research/local_runs/herg_discovery_campaign_v1")
DEFAULT_V7_ROOT = Path("research/local_runs/herg_honest_measurement_campaign_v7_1")
DEFAULT_OUTPUT = Path("research/local_runs/herg_feature_lattice_campaign_v8")

BLOCKS = (
    "rdkit2d",
    "morgan",
    "maccs",
    "polarity_charge_internal_contacts",
    "energy_flexibility",
    "shape",
    "autocorr3d",
    "whim",
    "old3d_stable",
    "new3d_stable_misc",
    "selected_interactions",
)


class CampaignError(RuntimeError):
    """Raised when a scientific, resource, or integrity contract fails."""


@dataclass(frozen=True)
class Profile:
    profile_id: str
    engine: str
    params: dict[str, Any]

    def payload(self) -> dict[str, Any]:
        return {"profile_id": self.profile_id, "engine": self.engine, "params": self.params}


def _profiles() -> tuple[Profile, ...]:
    return (
        Profile(
            "xgb_v2_anchor",
            "xgboost",
            {
                "n_estimators": 1200,
                "max_depth": 8,
                "learning_rate": 0.02,
                "min_child_weight": 8.0,
                "subsample": 0.75,
                "colsample_bytree": 0.60,
                "reg_alpha": 0.5,
                "reg_lambda": 5.0,
                "max_bin": 96,
            },
        ),
        Profile(
            "xgb_shallow",
            "xgboost",
            {
                "n_estimators": 900,
                "max_depth": 5,
                "learning_rate": 0.03,
                "min_child_weight": 8.0,
                "subsample": 0.80,
                "colsample_bytree": 0.70,
                "reg_alpha": 0.5,
                "reg_lambda": 6.0,
                "max_bin": 96,
            },
        ),
        Profile(
            "xgb_medium",
            "xgboost",
            {
                "n_estimators": 850,
                "max_depth": 6,
                "learning_rate": 0.03,
                "min_child_weight": 4.0,
                "subsample": 0.85,
                "colsample_bytree": 0.82,
                "reg_alpha": 0.1,
                "reg_lambda": 3.0,
                "max_bin": 128,
            },
        ),
        Profile(
            "xgb_deep_sparse",
            "xgboost",
            {
                "n_estimators": 1100,
                "max_depth": 7,
                "learning_rate": 0.023,
                "min_child_weight": 12.0,
                "subsample": 0.72,
                "colsample_bytree": 0.50,
                "reg_alpha": 1.0,
                "reg_lambda": 8.0,
                "max_bin": 64,
            },
        ),
        Profile(
            "xgb_strong_regularization",
            "xgboost",
            {
                "n_estimators": 1000,
                "max_depth": 6,
                "learning_rate": 0.025,
                "min_child_weight": 18.0,
                "subsample": 0.80,
                "colsample_bytree": 0.68,
                "reg_alpha": 2.0,
                "reg_lambda": 15.0,
                "max_bin": 96,
            },
        ),
        Profile(
            "xgb_fast_local",
            "xgboost",
            {
                "n_estimators": 650,
                "max_depth": 5,
                "learning_rate": 0.05,
                "min_child_weight": 3.0,
                "subsample": 0.90,
                "colsample_bytree": 0.90,
                "reg_alpha": 0.0,
                "reg_lambda": 2.0,
                "max_bin": 128,
            },
        ),
        Profile(
            "lgb_balanced",
            "lightgbm",
            {
                "n_estimators": 1100,
                "num_leaves": 31,
                "learning_rate": 0.027,
                "min_child_samples": 25,
                "subsample": 0.82,
                "colsample_bytree": 0.75,
                "reg_alpha": 0.1,
                "reg_lambda": 3.0,
                "max_bin": 127,
            },
        ),
        Profile(
            "lgb_small_leaves",
            "lightgbm",
            {
                "n_estimators": 1300,
                "num_leaves": 15,
                "learning_rate": 0.022,
                "min_child_samples": 35,
                "subsample": 0.78,
                "colsample_bytree": 0.68,
                "reg_alpha": 0.5,
                "reg_lambda": 6.0,
                "max_bin": 127,
            },
        ),
        Profile(
            "lgb_large_regularized",
            "lightgbm",
            {
                "n_estimators": 800,
                "num_leaves": 63,
                "learning_rate": 0.035,
                "min_child_samples": 50,
                "subsample": 0.85,
                "colsample_bytree": 0.60,
                "reg_alpha": 1.0,
                "reg_lambda": 10.0,
                "max_bin": 127,
            },
        ),
        Profile(
            "lgb_huber",
            "lightgbm",
            {
                "objective": "huber",
                "alpha": 0.9,
                "n_estimators": 1000,
                "num_leaves": 31,
                "learning_rate": 0.03,
                "min_child_samples": 30,
                "subsample": 0.82,
                "colsample_bytree": 0.72,
                "reg_alpha": 0.25,
                "reg_lambda": 5.0,
                "max_bin": 127,
            },
        ),
    )


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _schema_sha(path: Path) -> str:
    return hashlib.sha256(pq.read_schema(path).serialize().to_pybytes()).hexdigest()


def _binding(path: Path, role: str) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise CampaignError(f"missing {role}: {path}")
    result: dict[str, Any] = {
        "role": role,
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha(path),
    }
    if path.suffix == ".parquet":
        result.update(rows=pq.read_metadata(path).num_rows, arrow_schema_sha256=_schema_sha(path))
    return result


def _verify_binding(binding: dict[str, Any], root: Path | None = None) -> None:
    path = Path(str(binding["path"])).resolve()
    if root is not None and not path.is_relative_to(root.resolve()):
        raise CampaignError(f"artifact escapes output root: {path}")
    if not path.is_file() or path.stat().st_size != int(binding["bytes"]) or _sha(path) != binding["sha256"]:
        raise CampaignError(f"artifact binding changed: {path}")
    if path.suffix == ".parquet":
        if (
            pq.read_metadata(path).num_rows != int(binding["rows"])
            or _schema_sha(path) != binding["arrow_schema_sha256"]
        ):
            raise CampaignError(f"Parquet physical binding changed: {path}")


def _self_hash(value: dict[str, Any], key: str) -> dict[str, Any]:
    result = copy.deepcopy(value)
    result.pop(key, None)
    result[key] = _digest(result)
    return result


def _atomic_json(path: Path, value: dict[str, Any], key: str) -> dict[str, Any]:
    document = _self_hash(value, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(json.dumps(document, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n")
    os.replace(temporary, path)
    return document


def _read_json(path: Path, key: str | None = None) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict) or (key and value.get(key) != _self_hash(value, key)[key]):
        raise CampaignError(f"invalid/self-hash-mismatched JSON: {path}")
    return value


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


class _Lock:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.handle: Any = None

    def __enter__(self) -> _Lock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+")
        try:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise CampaignError(f"campaign already running: {self.path}") from error
        return self

    def __exit__(self, *_: object) -> None:
        fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        self.handle.close()


def _mask_blocks(mask: int) -> tuple[str, ...]:
    return tuple(block for index, block in enumerate(BLOCKS) if mask & (1 << index))


def _coalition_columns(mask: int, families: dict[str, list[str]]) -> list[str]:
    return sorted({column for block in _mask_blocks(mask) for column in families[block]})


def _all_masks() -> tuple[int, ...]:
    return tuple(range(1 << len(BLOCKS)))


def _metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    error = np.abs(observed - predicted)
    pearson = pearsonr(observed, predicted).statistic if np.std(predicted) > 0 else np.nan
    spearman = spearmanr(observed, predicted).statistic if np.std(predicted) > 0 else np.nan
    return {
        "n": len(observed),
        "mae": float(error.mean()),
        "rmse": float(math.sqrt(mean_squared_error(observed, predicted))),
        "r2": float(r2_score(observed, predicted)),
        "pearson": float(pearson) if np.isfinite(pearson) else 0.0,
        "spearman": float(spearman) if np.isfinite(spearman) else 0.0,
        "fraction_within_0p5": float(np.mean(error <= 0.5)),
        "fraction_within_1p0": float(np.mean(error <= 1.0)),
        "tail_mae": float(error[(observed < 4.0) | (observed >= 7.0)].mean()),
    }


def _model(profile: Profile, workers: int, seed: int) -> Any:
    params = dict(profile.params)
    if profile.engine == "xgboost":
        return XGBRegressor(
            objective="reg:squarederror",
            tree_method="hist",
            n_jobs=workers,
            random_state=seed,
            verbosity=0,
            **params,
        )
    objective = str(params.pop("objective", "regression_l1"))
    return LGBMRegressor(
        objective=objective,
        n_jobs=workers,
        random_state=seed,
        verbosity=-1,
        subsample_freq=1,
        **params,
    )


def _screen_model(workers: int, seed: int) -> LGBMRegressor:
    return LGBMRegressor(
        objective="regression_l1",
        n_estimators=140,
        num_leaves=15,
        learning_rate=0.055,
        min_child_samples=45,
        subsample=0.82,
        subsample_freq=1,
        colsample_bytree=0.72,
        reg_alpha=0.5,
        reg_lambda=6.0,
        max_bin=63,
        n_jobs=workers,
        random_state=seed,
        verbosity=-1,
    )


def _fit_predict(
    frame: pd.DataFrame,
    columns: list[str],
    fit_ids: set[str],
    eval_ids: set[str],
    model: Any,
    *,
    constant_if_empty: bool = False,
) -> tuple[Any, pd.DataFrame, float]:
    fit = frame.loc[frame.structure_id.astype(str).isin(fit_ids)]
    evaluation = frame.loc[frame.structure_id.astype(str).isin(eval_ids)]
    if set(fit.scaffold_group_id.astype(str)) & set(evaluation.scaffold_group_id.astype(str)):
        raise CampaignError("scaffold leakage in fit/evaluation")
    started = time.monotonic()
    if not columns and constant_if_empty:
        predicted = np.full(len(evaluation), float(fit.target_pic50.median()))
        fitted: Any = {"constant": float(fit.target_pic50.median())}
    else:
        model.fit(fit[columns], fit.target_pic50.to_numpy(dtype=float))
        predicted = np.asarray(model.predict(evaluation[columns]), dtype=float)
        fitted = model
    elapsed = time.monotonic() - started
    result = evaluation[
        [
            "structure_id",
            "scaffold_group_id",
            "target_pic50",
            "measurement_modality",
            "automation_class",
            "assay_family",
            "source_family",
        ]
    ].copy()
    result["predicted_pic50"] = predicted
    return fitted, result, elapsed


def _free_disk_gib(path: Path) -> float:
    return shutil.disk_usage(path).free / 1024**3


def _output_gib(path: Path) -> float:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file()) / 1024**3


def _resource_gate(output: Path) -> None:
    if _free_disk_gib(output) < MIN_FREE_DISK_GIB:
        raise CampaignError(f"free disk below {MIN_FREE_DISK_GIB:.0f} GiB")
    if _output_gib(output) > MAX_OUTPUT_GIB:
        raise CampaignError(f"campaign output exceeds {MAX_OUTPUT_GIB:.0f} GiB")


def _prepare(repo: Path, matrix_root: Path, base_root: Path, output: Path) -> dict[str, Any]:
    prepared = output / "prepared"
    validation_path = prepared / "validation.json"
    if validation_path.is_file():
        return _read_json(validation_path, "validation_sha256")
    source_validation = _read_json(matrix_root / "validation.json", "validation_sha256")
    if source_validation.get("status") != "passed":
        raise CampaignError("V6 matrix validation did not pass")
    matrix = pd.read_parquet(matrix_root / "prepared/training_matrix.parquet")
    splits = pd.read_parquet(matrix_root / "prepared/fixed_nested_scaffold_splits.parquet")
    schema = _read_json(matrix_root / "prepared/feature_schemas.json", "feature_schema_sha256")
    cache_path = base_root / "prepared/exact_train_cache.parquet"
    maccs_columns = [f"maccs__{index:03d}" for index in range(167)]
    maccs = pd.read_parquet(cache_path, columns=["structure_id", *maccs_columns])
    matrix = matrix.merge(maccs, on="structure_id", how="left", validate="one_to_one")
    if (
        len(matrix) != EXACT_ROWS
        or matrix.structure_id.duplicated().any()
        or matrix[maccs_columns].isna().any().any()
    ):
        raise CampaignError("canonical matrix/MACCS join failed")
    if set(schema["families"]) != set(BLOCKS) - {"maccs"}:
        raise CampaignError("source feature-family contract changed")
    families = {key: list(value) for key, value in schema["families"].items()}
    families["maccs"] = maccs_columns
    if any(not families[block] for block in BLOCKS):
        raise CampaignError("one or more V8 feature blocks are empty")
    if len(splits) != EXACT_ROWS * 5 or set(splits.outer_fold.astype(int)) != set(OUTERS):
        raise CampaignError("canonical nested split registry changed")
    prepared.mkdir(parents=True, exist_ok=True)
    matrix_path = prepared / "training_matrix.parquet"
    splits_path = prepared / "fixed_nested_scaffold_splits.parquet"
    blocks_path = prepared / "feature_blocks.json"
    _atomic_parquet(matrix_path, matrix)
    _atomic_parquet(splits_path, splits)
    blocks = _atomic_json(
        blocks_path,
        {
            "schema_version": SCHEMA_VERSION,
            "blocks": families,
            "block_order": list(BLOCKS),
            "coalitions": 2 ** len(BLOCKS),
            "interpretation": {
                "rdkit2d_morgan_maccs": "common ligand representations",
                "remaining_blocks": "ligand-only fundamental/conformational descriptors",
                "selected_interactions": "engineered derived descriptors assessed as an explicit independent block",
            },
        },
        "feature_blocks_sha256",
    )
    validation = _atomic_json(
        validation_path,
        {
            "schema_version": SCHEMA_VERSION,
            "status": "passed",
            "created_utc": _utc(),
            "exact_train_structures": len(matrix),
            "split_rows": len(splits),
            "feature_blocks": len(BLOCKS),
            "coalitions": 2 ** len(BLOCKS),
            "repository_validation_labels_opened": False,
            "repository_test_labels_opened": False,
            "artifacts": [
                _binding(matrix_path, "prepared_matrix"),
                _binding(splits_path, "prepared_splits"),
                _binding(blocks_path, "feature_blocks"),
            ],
            "source_bindings": [
                _binding(matrix_root / "manifest.json", "v6_manifest"),
                _binding(matrix_root / "validation.json", "v6_validation"),
                _binding(matrix_root / "prepared/training_matrix.parquet", "v6_matrix"),
                _binding(matrix_root / "prepared/fixed_nested_scaffold_splits.parquet", "v6_splits"),
                _binding(matrix_root / "prepared/feature_schemas.json", "v6_schema"),
                _binding(cache_path, "exact_train_cache"),
                _binding(Path(__file__), "v8_implementation"),
            ],
            "feature_blocks_sha256": blocks["feature_blocks_sha256"],
        },
        "validation_sha256",
    )
    return validation


def _load(output: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[str]]]:
    matrix = pd.read_parquet(output / "prepared/training_matrix.parquet")
    splits = pd.read_parquet(output / "prepared/fixed_nested_scaffold_splits.parquet")
    blocks = _read_json(output / "prepared/feature_blocks.json", "feature_blocks_sha256")
    return matrix, splits, {key: list(value) for key, value in blocks["blocks"].items()}


def _unit_document(
    directory: Path, spec: dict[str, Any], metrics: dict[str, Any], artifacts: list[dict[str, Any]]
) -> dict[str, Any]:
    return _atomic_json(
        directory / "unit.json",
        {
            "schema_version": SCHEMA_VERSION,
            "status": "passed",
            "created_utc": _utc(),
            "unit_id": directory.name,
            "unit_spec": spec,
            "unit_spec_sha256": _digest(spec),
            "metrics": metrics,
            "artifacts": artifacts,
            "scientific_scope": {
                "source_partition": "train",
                "scaffold_held_out": True,
                "repository_validation_labels_opened": False,
                "repository_test_labels_opened": False,
            },
        },
        "unit_json_sha256",
    )


def _existing(directory: Path, spec: dict[str, Any], output: Path) -> dict[str, Any] | None:
    try:
        unit = _read_json(directory / "unit.json", "unit_json_sha256")
        if unit.get("status") != "passed" or unit.get("unit_spec") != spec:
            return None
        for artifact in unit["artifacts"]:
            _verify_binding(artifact, output)
        return unit
    except (OSError, ValueError, KeyError, json.JSONDecodeError, CampaignError):
        return None


def _screen_shard(
    output: Path,
    matrix: pd.DataFrame,
    splits: pd.DataFrame,
    families: dict[str, list[str]],
    outer: int,
    masks: list[int],
    workers: int,
) -> dict[str, Any]:
    shard = masks[0] // SCREEN_SHARD_SIZE
    directory = output / "units" / f"screen_o{outer}_s{shard:03d}"
    spec = {
        "operation": "complete_lattice_screen",
        "outer_fold": outer,
        "masks": masks,
        "screen_profile": "fixed_lgbm_140",
        "selection_inner_fold": 0,
    }
    existing = _existing(directory, spec, output)
    if existing is not None:
        return existing
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    assignments = splits.loc[(splits.outer_fold.eq(outer)) & splits.outer_role.eq("fit")]
    fit_ids = set(assignments.loc[assignments.inner_fold.ne(0), "structure_id"].astype(str))
    eval_ids = set(assignments.loc[assignments.inner_fold.eq(0), "structure_id"].astype(str))
    eval_identity: pd.DataFrame | None = None
    prediction_columns: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    for mask in masks:
        columns = _coalition_columns(mask, families)
        model = _screen_model(workers, SEED + outer * 10_000 + mask)
        _, prediction, elapsed = _fit_predict(
            matrix, columns, fit_ids, eval_ids, model, constant_if_empty=True
        )
        if eval_identity is None:
            eval_identity = prediction[["structure_id", "scaffold_group_id", "target_pic50"]].copy()
        prediction_columns[f"c{mask:04d}"] = prediction.predicted_pic50.to_numpy(dtype=np.float32)
        rows.append(
            {
                "outer_fold": outer,
                "coalition_mask": mask,
                "coalition_size": len(_mask_blocks(mask)),
                "blocks_json": json.dumps(_mask_blocks(mask)),
                "feature_count": len(columns),
                "fit_seconds": elapsed,
                **_metrics(prediction.target_pic50, prediction.predicted_pic50),
            }
        )
    assert eval_identity is not None
    prediction_frame = eval_identity.assign(**prediction_columns)
    metrics_path = directory / "screen_metrics.parquet"
    predictions_path = directory / "screen_predictions.parquet"
    _atomic_parquet(metrics_path, pd.DataFrame(rows))
    _atomic_parquet(predictions_path, prediction_frame)
    return _unit_document(
        directory,
        spec,
        {"coalitions": len(masks), "rows": len(prediction_frame)},
        [_binding(metrics_path, "screen_metrics"), _binding(predictions_path, "screen_predictions")],
    )


def _role(unit: dict[str, Any], role: str) -> Path:
    for artifact in unit["artifacts"]:
        if artifact["role"] == role:
            return Path(artifact["path"])
    raise CampaignError(f"unit lacks {role}: {unit['unit_id']}")


def _promote(screen: pd.DataFrame) -> list[int]:
    screen = screen.sort_values(["mae", "coalition_size", "coalition_mask"])
    chosen: list[int] = []

    def add(values: Iterable[int]) -> None:
        for value in values:
            if value not in chosen:
                chosen.append(int(value))

    add(screen.head(24).coalition_mask)
    for size in range(len(BLOCKS) + 1):
        add(screen.loc[screen.coalition_size.eq(size)].head(2).coalition_mask)
    for index in range(len(BLOCKS)):
        bit = 1 << index
        add(
            screen.loc[
                screen.coalition_mask.astype(int).map(lambda value, selected=bit: bool(value & selected))
            ]
            .head(2)
            .coalition_mask
        )
    common = (1 << BLOCKS.index("rdkit2d")) | (1 << BLOCKS.index("morgan"))
    add([0, common, (1 << len(BLOCKS)) - 1])
    return chosen[:PROMOTED_COALITIONS]


def _hpo_unit(
    output: Path,
    matrix: pd.DataFrame,
    splits: pd.DataFrame,
    families: dict[str, list[str]],
    outer: int,
    mask: int,
    profile: Profile,
    workers: int,
) -> dict[str, Any]:
    directory = output / "units" / f"hpo_o{outer}_c{mask:04d}_{profile.profile_id}"
    spec = {
        "operation": "promoted_model_hpo",
        "outer_fold": outer,
        "coalition_mask": mask,
        "blocks": list(_mask_blocks(mask)),
        "profile": profile.payload(),
        "evaluation_inner_folds": [1, 2],
    }
    existing = _existing(directory, spec, output)
    if existing is not None:
        return existing
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    assignments = splits.loc[(splits.outer_fold.eq(outer)) & splits.outer_role.eq("fit")]
    columns = _coalition_columns(mask, families)
    rows: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    for inner in (1, 2):
        fit_ids = set(assignments.loc[assignments.inner_fold.ne(inner), "structure_id"].astype(str))
        eval_ids = set(assignments.loc[assignments.inner_fold.eq(inner), "structure_id"].astype(str))
        model = _model(profile, workers, SEED + outer * 1000 + inner * 100 + mask)
        _, pred, elapsed = _fit_predict(matrix, columns, fit_ids, eval_ids, model, constant_if_empty=True)
        rows.append(
            {
                "outer_fold": outer,
                "inner_fold": inner,
                "coalition_mask": mask,
                "profile_id": profile.profile_id,
                "engine": profile.engine,
                "feature_count": len(columns),
                "fit_seconds": elapsed,
                **_metrics(pred.target_pic50, pred.predicted_pic50),
            }
        )
        pred = pred.rename(columns={"target_pic50": "observed_pic50"})
        pred["inner_fold"] = inner
        predictions.append(pred)
    metrics_path = directory / "hpo_metrics.parquet"
    predictions_path = directory / "hpo_predictions.parquet"
    frame = pd.DataFrame(rows)
    _atomic_parquet(metrics_path, frame)
    _atomic_parquet(predictions_path, pd.concat(predictions, ignore_index=True))
    return _unit_document(
        directory,
        spec,
        {"selection_score": float(frame.mae.mean()), "folds": 2},
        [_binding(metrics_path, "hpo_metrics"), _binding(predictions_path, "hpo_predictions")],
    )


def _block_contributions(
    model: Any, evaluation: pd.DataFrame, columns: list[str], families: dict[str, list[str]]
) -> pd.DataFrame:
    x = evaluation[columns]
    if isinstance(model, XGBRegressor):
        contributions = model.get_booster().predict(DMatrix(x, feature_names=columns), pred_contribs=True)
    else:
        contributions = np.asarray(model.predict(x, pred_contrib=True), dtype=float)
    if contributions.shape[1] != len(columns) + 1:
        raise CampaignError("tree contribution width mismatch")
    mapping: dict[str, str] = {}
    for block, block_columns in families.items():
        for column in block_columns:
            mapping[column] = block
    data: dict[str, Any] = {
        "structure_id": evaluation.structure_id.astype(str).to_numpy(),
        "scaffold_group_id": evaluation.scaffold_group_id.astype(str).to_numpy(),
        "bias": contributions[:, -1],
    }
    for block in BLOCKS:
        indices = [index for index, column in enumerate(columns) if mapping.get(column) == block]
        data[f"contribution__{block}"] = (
            contributions[:, indices].sum(axis=1) if indices else np.zeros(len(evaluation))
        )
    return pd.DataFrame(data)


def _outer_unit(
    output: Path,
    matrix: pd.DataFrame,
    splits: pd.DataFrame,
    families: dict[str, list[str]],
    outer: int,
    promoted: list[int],
    hpo_units: list[dict[str, Any]],
    workers: int,
) -> dict[str, Any]:
    best = min(hpo_units, key=lambda unit: (float(unit["metrics"]["selection_score"]), unit["unit_id"]))
    profile_payload = best["unit_spec"]["profile"]
    profile = next(item for item in _profiles() if item.profile_id == profile_payload["profile_id"])
    mask = int(best["unit_spec"]["coalition_mask"])
    directory = output / "units" / f"outer_o{outer}"
    spec = {
        "operation": "nested_outer_evaluation_and_local_relationships",
        "outer_fold": outer,
        "promoted_masks": promoted,
        "winner_mask": mask,
        "winner_profile": profile.payload(),
        "selection_source_unit": best["unit_id"],
        "conditional_permutation_repeats": PERMUTATION_REPEATS,
    }
    existing = _existing(directory, spec, output)
    if existing is not None:
        return existing
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    assignments = splits.loc[splits.outer_fold.eq(outer)]
    fit_ids = set(assignments.loc[assignments.outer_role.eq("fit"), "structure_id"].astype(str))
    eval_ids = set(assignments.loc[assignments.outer_role.eq("heldout"), "structure_id"].astype(str))
    columns = _coalition_columns(mask, families)
    model = _model(profile, workers, SEED + 100_000 + outer)
    model, pred, elapsed = _fit_predict(matrix, columns, fit_ids, eval_ids, model, constant_if_empty=True)
    pred = pred.rename(columns={"target_pic50": "observed_pic50"})
    pred["outer_fold"] = outer
    pred["coalition_mask"] = mask
    pred["profile_id"] = profile.profile_id
    evaluation = matrix.loc[matrix.structure_id.astype(str).isin(eval_ids)].copy()
    contribution = _block_contributions(model, evaluation, columns, families)
    # Paired direct block effects: refit without included blocks and add each omitted block.
    paired_rows: list[pd.DataFrame] = []
    base_abs = np.abs(pred.observed_pic50.to_numpy() - pred.predicted_pic50.to_numpy())
    for index, block in enumerate(BLOCKS):
        changed_mask = mask ^ (1 << index)
        changed_columns = _coalition_columns(changed_mask, families)
        changed_profile = Profile(profile.profile_id, profile.engine, profile.params)
        changed_model = _model(changed_profile, workers, SEED + 110_000 + outer * 100 + index)
        _, changed, _ = _fit_predict(
            matrix, changed_columns, fit_ids, eval_ids, changed_model, constant_if_empty=True
        )
        changed_abs = np.abs(changed.target_pic50.to_numpy() - changed.predicted_pic50.to_numpy())
        paired_rows.append(
            pd.DataFrame(
                {
                    "structure_id": pred.structure_id.astype(str),
                    "scaffold_group_id": pred.scaffold_group_id.astype(str),
                    "outer_fold": outer,
                    "block": block,
                    "baseline_abs_error": base_abs,
                    "changed_abs_error": changed_abs,
                    "delta_mae": changed_abs - base_abs,
                    "operation": "remove_included" if mask & (1 << index) else "add_omitted",
                }
            )
        )
    # Conditional permutation of included blocks, retaining molecule-level effects.
    fit = matrix.loc[matrix.structure_id.astype(str).isin(fit_ids)]
    strata = _conditioning_strata(fit, evaluation)
    for index, block in enumerate(BLOCKS):
        if not mask & (1 << index):
            continue
        block_columns = [column for column in families[block] if column in columns]
        for repeat in range(PERMUTATION_REPEATS):
            perturbed = evaluation[columns].copy()
            rng = np.random.default_rng(SEED + 200_000 + outer * 10_000 + index * 100 + repeat)
            for cell in np.unique(strata):
                indices = np.flatnonzero(strata == cell)
                if len(indices) > 1:
                    donor = rng.permutation(indices)
                    perturbed.iloc[indices, perturbed.columns.get_indexer(block_columns)] = evaluation.iloc[
                        donor
                    ][block_columns].to_numpy()
            changed_prediction = np.asarray(model.predict(perturbed), dtype=float)
            changed_abs = np.abs(pred.observed_pic50.to_numpy() - changed_prediction)
            paired_rows.append(
                pd.DataFrame(
                    {
                        "structure_id": pred.structure_id.astype(str),
                        "scaffold_group_id": pred.scaffold_group_id.astype(str),
                        "outer_fold": outer,
                        "block": block,
                        "baseline_abs_error": base_abs,
                        "changed_abs_error": changed_abs,
                        "delta_mae": changed_abs - base_abs,
                        "operation": "conditional_permutation",
                        "repeat": repeat,
                    }
                )
            )
    paths = {
        "nested_oof": directory / "nested_oof.parquet",
        "block_contributions": directory / "block_contributions.parquet",
        "paired_block_effects": directory / "paired_block_effects.parquet",
        "model": directory / "nested_model.joblib",
    }
    _atomic_parquet(paths["nested_oof"], pred)
    _atomic_parquet(paths["block_contributions"], contribution)
    _atomic_parquet(paths["paired_block_effects"], pd.concat(paired_rows, ignore_index=True))
    temporary = paths["model"].with_suffix(".joblib.tmp")
    joblib.dump(
        {"model": model, "features": columns, "profile": profile.payload(), "coalition_mask": mask}, temporary
    )
    os.replace(temporary, paths["model"])
    metric = _metrics(pred.observed_pic50, pred.predicted_pic50)
    metric.update(
        fit_seconds=elapsed,
        selection_score=float(best["metrics"]["selection_score"]),
        feature_count=len(columns),
    )
    return _unit_document(directory, spec, metric, [_binding(path, role) for role, path in paths.items()])


def _conditioning_strata(fit: pd.DataFrame, evaluation: pd.DataFrame) -> np.ndarray:
    codes: list[np.ndarray] = []
    for column in ("rdkit2d__MolWt", "rdkit2d__MolLogP", "rdkit2d__TPSA"):
        fit_values = pd.to_numeric(fit[column], errors="coerce").to_numpy(dtype=float, copy=True)
        finite = fit_values[np.isfinite(fit_values)]
        edges = np.unique(np.quantile(finite, [0.25, 0.5, 0.75]))
        values = pd.to_numeric(evaluation[column], errors="coerce").to_numpy(dtype=float, copy=True)
        values[~np.isfinite(values)] = float(np.median(finite))
        codes.append(np.digitize(values, edges))
    return codes[0] * 16 + codes[1] * 4 + codes[2]


def _screen_frames(output: Path, outer: int) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
    units = [
        _read_json(path, "unit_json_sha256")
        for path in sorted((output / "units").glob(f"screen_o{outer}_*/unit.json"))
    ]
    metrics = pd.concat([pd.read_parquet(_role(unit, "screen_metrics")) for unit in units], ignore_index=True)
    predictions = [pd.read_parquet(_role(unit, "screen_predictions")) for unit in units]
    if len(metrics) != 2 ** len(BLOCKS) or metrics.coalition_mask.nunique() != len(metrics):
        raise CampaignError(f"outer {outer} lattice is incomplete")
    return metrics, predictions


def _shapley_values(values: np.ndarray, n_blocks: int) -> np.ndarray:
    """Exact Shapley values for a complete coalition value vector."""
    if values.shape[-1] != 1 << n_blocks:
        raise ValueError("complete power-set values are required")
    result = np.zeros(values.shape[:-1] + (n_blocks,), dtype=float)
    factorial = math.factorial
    denominator = factorial(n_blocks)
    for index in range(n_blocks):
        bit = 1 << index
        for mask in range(1 << n_blocks):
            if mask & bit:
                continue
            size = int(mask).bit_count()
            weight = factorial(size) * factorial(n_blocks - size - 1) / denominator
            result[..., index] += weight * (values[..., mask | bit] - values[..., mask])
    return result


def _pairwise_banzhaf(values: np.ndarray, n_blocks: int) -> np.ndarray:
    result = np.zeros((n_blocks, n_blocks), dtype=float)
    for first, second in combinations(range(n_blocks), 2):
        a, b = 1 << first, 1 << second
        deltas = [
            values[mask | a | b] - values[mask | a] - values[mask | b] + values[mask]
            for mask in range(1 << n_blocks)
            if not mask & (a | b)
        ]
        result[first, second] = result[second, first] = float(np.mean(deltas))
    return result


def _bootstrap_delta(
    frame: pd.DataFrame, first: str, second: str, replicates: int = 10_000
) -> dict[str, Any]:
    grouped = (
        frame.assign(
            delta=np.abs(frame.observed_pic50 - frame[first]) - np.abs(frame.observed_pic50 - frame[second])
        )
        .groupby("scaffold_group_id")
        .delta.agg(["sum", "count"])
    )
    rng = np.random.default_rng(SEED)
    sums = grouped["sum"].to_numpy()
    counts = grouped["count"].to_numpy()
    n = len(grouped)
    estimates = np.empty(replicates)
    for start in range(0, replicates, 250):
        stop = min(start + 250, replicates)
        sample = rng.integers(0, n, size=(stop - start, n))
        estimates[start:stop] = sums[sample].sum(axis=1) / counts[sample].sum(axis=1)
    return {
        "point_estimate": float(grouped["sum"].sum() / grouped["count"].sum()),
        "ci95_lower": float(np.quantile(estimates, 0.025)),
        "ci95_upper": float(np.quantile(estimates, 0.975)),
        "probability_first_better": float(np.mean(estimates < 0)),
        "replicates": replicates,
        "resampling_unit": "scaffold_group_id",
        "delta": f"abs_error({first})-abs_error({second})",
    }


def _load_v7_accuracy_predictions(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    prediction_columns = [
        column for column in ("accuracy_prediction", "predicted_pic50") if column in frame.columns
    ]
    if len(prediction_columns) != 1:
        raise CampaignError("V7 accuracy artifact must contain exactly one supported prediction column")
    required = {"structure_id", prediction_columns[0]}
    if not required.issubset(frame.columns):
        raise CampaignError("V7 accuracy artifact is missing required identity columns")
    result = frame[["structure_id", prediction_columns[0]]].rename(
        columns={prediction_columns[0]: "accuracy_prediction"}
    )
    if (
        result.structure_id.duplicated().any()
        or not np.isfinite(result.accuracy_prediction.to_numpy(dtype=float)).all()
    ):
        raise CampaignError("V7 accuracy predictions are not unique and finite")
    return result


def _select_final_recipe(hpo_units: list[dict[str, Any]]) -> tuple[int, Profile, pd.DataFrame]:
    rows = pd.DataFrame(
        [
            {
                "outer_fold": int(unit["unit_spec"]["outer_fold"]),
                "coalition_mask": int(unit["unit_spec"]["coalition_mask"]),
                "profile_id": str(unit["unit_spec"]["profile"]["profile_id"]),
                "selection_score": float(unit["metrics"]["selection_score"]),
            }
            for unit in hpo_units
        ]
    )
    summary = (
        rows.groupby(["coalition_mask", "profile_id"], as_index=False)
        .agg(mean_inner_mae=("selection_score", "mean"), outer_contexts=("outer_fold", "nunique"))
        .loc[lambda frame: frame.outer_contexts.eq(5)]
        .sort_values(["mean_inner_mae", "coalition_mask", "profile_id"], ignore_index=True)
    )
    if summary.empty:
        raise CampaignError("no final recipe was evaluated inside all five outer contexts")
    winner = summary.iloc[0]
    profile = next(item for item in _profiles() if item.profile_id == winner.profile_id)
    return int(winner.coalition_mask), profile, summary


def _final_refit(
    output: Path,
    matrix: pd.DataFrame,
    families: dict[str, list[str]],
    hpo_units: list[dict[str, Any]],
    workers: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    mask, profile, summary = _select_final_recipe(hpo_units)
    columns = _coalition_columns(mask, families)
    models: list[Any] = []
    fit_seconds: list[float] = []
    for seed_offset in range(5):
        model = _model(profile, workers, SEED + 400_000 + seed_offset)
        started = time.monotonic()
        model.fit(matrix[columns], matrix.target_pic50.to_numpy(dtype=float))
        fit_seconds.append(time.monotonic() - started)
        models.append(model)
    directory = output / "final_model"
    directory.mkdir(parents=True, exist_ok=True)
    model_path = directory / "model_ensemble.joblib"
    temporary = model_path.with_suffix(".joblib.tmp")
    joblib.dump(
        {
            "models": models,
            "features": columns,
            "feature_blocks": list(_mask_blocks(mask)),
            "coalition_mask": mask,
            "profile": profile.payload(),
            "target": "wild-type-or-unspecified exact hERG pIC50",
            "training_partition": "train",
        },
        temporary,
        compress=3,
    )
    os.replace(temporary, model_path)
    recipe_path = directory / "model_schema.json"
    recipe = _atomic_json(
        recipe_path,
        {
            "schema_version": SCHEMA_VERSION,
            "created_utc": _utc(),
            "coalition_mask": mask,
            "feature_blocks": list(_mask_blocks(mask)),
            "feature_columns": columns,
            "feature_count": len(columns),
            "profile": profile.payload(),
            "ensemble_seeds": 5,
            "fit_seconds": fit_seconds,
            "selection_contract": "lowest mean inner-fold MAE among recipes present in all five outer contexts",
            "performance_contract": "report nested outer OOF only; full-data refit has no independent performance estimate",
        },
        "model_schema_sha256",
    )
    ranking_path = directory / "eligible_recipe_ranking.parquet"
    _atomic_parquet(ranking_path, summary)
    smoke = matrix.head(12)
    prediction = np.mean([np.asarray(model.predict(smoke[columns]), dtype=float) for model in models], axis=0)
    smoke_path = directory / "inference_smoke.parquet"
    _atomic_parquet(
        smoke_path,
        pd.DataFrame(
            {
                "structure_id": smoke.structure_id.astype(str),
                "predicted_pic50": prediction,
                "finite_prediction": np.isfinite(prediction),
            }
        ),
    )
    if not np.isfinite(prediction).all():
        raise CampaignError("final model smoke predictions are not finite")
    bindings = [
        _binding(model_path, "final_model_ensemble"),
        _binding(recipe_path, "final_model_schema"),
        _binding(ranking_path, "final_recipe_ranking"),
        _binding(smoke_path, "final_inference_smoke"),
    ]
    return recipe, bindings


def _aggregate(
    output: Path,
    checkpoint: dict[str, Any],
    outer_units: list[dict[str, Any]],
    matrix: pd.DataFrame,
    families: dict[str, list[str]],
    v7_root: Path,
    hpo_units: list[dict[str, Any]],
    workers: int,
) -> dict[str, Any]:
    nested = pd.concat(
        [pd.read_parquet(_role(unit, "nested_oof")) for unit in outer_units], ignore_index=True
    )
    if (
        len(nested) != EXACT_ROWS
        or nested.structure_id.duplicated().any()
        or set(nested.outer_fold) != set(OUTERS)
    ):
        raise CampaignError("nested OOF coverage is not exactly 18,801 structures across five folds")
    paired = pd.concat(
        [pd.read_parquet(_role(unit, "paired_block_effects")) for unit in outer_units], ignore_index=True
    )
    contributions = pd.concat(
        [pd.read_parquet(_role(unit, "block_contributions")) for unit in outer_units], ignore_index=True
    )
    lattice_rows: list[pd.DataFrame] = []
    global_shapley_rows: list[dict[str, Any]] = []
    synergy_rows: list[dict[str, Any]] = []
    local_shapley_rows: list[pd.DataFrame] = []
    for outer in OUTERS:
        metrics, prediction_shards = _screen_frames(output, outer)
        metrics = metrics.sort_values("coalition_mask")
        value = -metrics.mae.to_numpy(dtype=float)
        shapley = _shapley_values(value, len(BLOCKS))
        for index, block in enumerate(BLOCKS):
            global_shapley_rows.append(
                {"outer_fold": outer, "block": block, "shapley_mae_improvement": float(shapley[index])}
            )
        synergy = _pairwise_banzhaf(value, len(BLOCKS))
        for first, second in combinations(range(len(BLOCKS)), 2):
            synergy_rows.append(
                {
                    "outer_fold": outer,
                    "block_a": BLOCKS[first],
                    "block_b": BLOCKS[second],
                    "banzhaf_synergy": float(synergy[first, second]),
                }
            )
        lattice_rows.append(metrics)
        identity = prediction_shards[0][["structure_id", "scaffold_group_id", "target_pic50"]].copy()
        prediction_matrix = np.empty((len(identity), 1 << len(BLOCKS)), dtype=np.float32)
        for shard in prediction_shards:
            if not identity.structure_id.equals(shard.structure_id):
                raise CampaignError("screen prediction shard identity mismatch")
            for column in shard.columns:
                if column.startswith("c") and column[1:].isdigit():
                    prediction_matrix[:, int(column[1:])] = shard[column]
        individual_value = -np.abs(identity.target_pic50.to_numpy()[:, None] - prediction_matrix)
        local = _shapley_values(individual_value, len(BLOCKS))
        for index, block in enumerate(BLOCKS):
            local_shapley_rows.append(
                pd.DataFrame(
                    {
                        "outer_context": outer,
                        "structure_id": identity.structure_id,
                        "scaffold_group_id": identity.scaffold_group_id,
                        "block": block,
                        "local_shapley_abs_error_improvement": local[:, index],
                    }
                )
            )
    lattice = pd.concat(lattice_rows, ignore_index=True)
    local_shapley = pd.concat(local_shapley_rows, ignore_index=True)
    overall = _metrics(nested.observed_pic50, nested.predicted_pic50)
    v7_path = v7_root / "nested_accuracy_oof_predictions.parquet"
    comparison: dict[str, Any] | None = None
    if v7_path.is_file():
        v7 = _load_v7_accuracy_predictions(v7_path)
        joined = nested.merge(v7, on="structure_id", validate="one_to_one")
        comparison = _bootstrap_delta(joined, "predicted_pic50", "accuracy_prediction")
    block_effect = paired.groupby(["operation", "block"], as_index=False).agg(
        mean_delta_mae=("delta_mae", "mean"),
        median_delta_mae=("delta_mae", "median"),
        rows=("delta_mae", "size"),
    )
    final_recipe, final_bindings = _final_refit(output, matrix, families, hpo_units, workers)
    outputs = {
        "nested_oof_predictions": nested,
        "complete_lattice_metrics": lattice,
        "global_block_shapley": pd.DataFrame(global_shapley_rows),
        "pairwise_block_synergy": pd.DataFrame(synergy_rows),
        "individual_block_shapley": local_shapley,
        "nested_block_contributions": contributions,
        "nested_paired_block_effects": paired,
        "block_effect_summary": block_effect,
    }
    bindings: list[dict[str, Any]] = []
    for role, frame in outputs.items():
        path = output / f"{role}.parquet"
        _atomic_parquet(path, frame)
        bindings.append(_binding(path, role))
    unit_paths = sorted((output / "units").glob("*/unit.json"))
    unit_bindings = [_binding(path, f"unit::{path.parent.name}") for path in unit_paths]
    analysis = _atomic_json(
        output / "analysis.json",
        {
            "schema_version": SCHEMA_VERSION,
            "status": "passed",
            "created_utc": _utc(),
            "metrics": overall,
            "v8_vs_v7_scaffold_bootstrap": comparison,
            "final_recipe": final_recipe,
            "outer_winners": [
                {
                    "outer_fold": unit["unit_spec"]["outer_fold"],
                    "winner_mask": unit["unit_spec"]["winner_mask"],
                    "winner_blocks": list(_mask_blocks(int(unit["unit_spec"]["winner_mask"]))),
                    "winner_profile": unit["unit_spec"]["winner_profile"],
                    "metrics": unit["metrics"],
                }
                for unit in outer_units
            ],
            "evidence_contract": {
                "complete_lattice": "all 2,048 coalitions evaluated in every outer context on an inner screening fold",
                "global_shapley": "exact screen-level allocation of coalition predictive value; exploratory selection evidence",
                "local_shapley": "molecule-level allocation of absolute-error improvement in screening contexts",
                "nested_effects": "outer-heldout refit toggles and conditional permutations; associative, not causal",
                "validation_test": "repository validation and test labels remained sealed",
            },
        },
        "analysis_sha256",
    )
    bindings.append(_binding(output / "analysis.json", "analysis"))
    manifest = _atomic_json(
        output / "manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "status": "passed",
            "created_utc": _utc(),
            "metrics": overall,
            "analysis_sha256": analysis["analysis_sha256"],
            "scientific_scope": {
                "source_partition": "train",
                "nested_scaffold_evaluation": True,
                "external_or_prospective_validation": False,
                "repository_validation_labels_opened": False,
                "repository_test_labels_opened": False,
            },
            "input_bindings": checkpoint["input_bindings"],
            "artifacts": bindings + final_bindings + unit_bindings,
            "unit_documents": unit_bindings,
        },
        "manifest_sha256",
    )
    validation = _validate(output)
    checkpoint.update(status="complete", finished_utc=_utc())
    checkpoint = _write_checkpoint(output, checkpoint)
    _atomic_json(
        output / "DONE.json",
        {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "manifest_sha256": manifest["manifest_sha256"],
            "validation_sha256": validation["validation_sha256"],
            "checkpoint_sha256": checkpoint["checkpoint_sha256"],
        },
        "done_sha256",
    )
    return validation


def _initial_checkpoint(
    repo: Path, matrix_root: Path, base_root: Path, v7_root: Path, output: Path, workers: int
) -> dict[str, Any]:
    bindings = [
        _binding(Path(__file__), "implementation"),
        _binding(matrix_root / "manifest.json", "v6_manifest"),
        _binding(matrix_root / "validation.json", "v6_validation"),
        _binding(base_root / "prepared/exact_train_cache.parquet", "exact_train_cache"),
    ]
    if (v7_root / "manifest.json").is_file():
        bindings.append(_binding(v7_root / "manifest.json", "v7_manifest"))
    return _atomic_json(
        output / "checkpoint.json",
        {
            "schema_version": SCHEMA_VERSION,
            "status": "running",
            "created_utc": _utc(),
            "repo_root": str(repo),
            "matrix_root": str(matrix_root),
            "base_root": str(base_root),
            "v7_root": str(v7_root),
            "output_root": str(output),
            "workers": workers,
            "block_order": list(BLOCKS),
            "profiles": [profile.payload() for profile in _profiles()],
            "input_bindings": bindings,
            "completed_units": [],
            "active_seconds": 0.0,
            "hard_active_hours": MAX_ACTIVE_HOURS,
        },
        "checkpoint_sha256",
    )


def _write_checkpoint(output: Path, checkpoint: dict[str, Any]) -> dict[str, Any]:
    checkpoint["updated_utc"] = _utc()
    return _atomic_json(output / "checkpoint.json", checkpoint, "checkpoint_sha256")


def _resume(args: argparse.Namespace, output: Path) -> dict[str, Any]:
    checkpoint = _read_json(output / "checkpoint.json", "checkpoint_sha256")
    expected = {
        "repo_root": str(Path(args.repo_root).resolve()),
        "matrix_root": str(Path(args.matrix_root).resolve()),
        "base_root": str(Path(args.base_root).resolve()),
        "v7_root": str(Path(args.v7_root).resolve()),
        "output_root": str(output),
        "workers": int(args.workers),
    }
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise CampaignError(f"resume argument changed: {key}")
    if checkpoint.get("block_order") != list(BLOCKS) or checkpoint.get("profiles") != [
        p.payload() for p in _profiles()
    ]:
        raise CampaignError("V8 plan changed after launch")
    for binding in checkpoint["input_bindings"]:
        _verify_binding(binding)
    if checkpoint.get("status") == "complete":
        return checkpoint
    checkpoint["status"] = "running"
    return _write_checkpoint(output, checkpoint)


def _time_remaining(checkpoint: dict[str, Any], invocation_start: float) -> float:
    active = float(checkpoint.get("active_seconds", 0.0)) + (time.monotonic() - invocation_start)
    return MAX_ACTIVE_HOURS * 3600 - active


def _record_unit(checkpoint: dict[str, Any], output: Path, unit: dict[str, Any]) -> dict[str, Any]:
    if unit["unit_id"] not in checkpoint["completed_units"]:
        checkpoint["completed_units"].append(unit["unit_id"])
        checkpoint = _write_checkpoint(output, checkpoint)
    return checkpoint


def _run(args: argparse.Namespace) -> dict[str, Any]:
    repo = Path(args.repo_root).resolve()
    matrix_root = Path(args.matrix_root).resolve()
    base_root = Path(args.base_root).resolve()
    v7_root = Path(args.v7_root).resolve()
    output = Path(args.output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with _Lock(output / ".campaign.lock"):
        checkpoint = (
            _resume(args, output)
            if (output / "checkpoint.json").is_file()
            else _initial_checkpoint(repo, matrix_root, base_root, v7_root, output, args.workers)
        )
        if checkpoint.get("status") == "complete":
            return _validate(output)
        _resource_gate(output)
        _prepare(repo, matrix_root, base_root, output)
        matrix, splits, families = _load(output)
        stop = False

        def request_stop(_signum: int, _frame: object) -> None:
            nonlocal stop
            stop = True

        handlers = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)}
        started = time.monotonic()
        try:
            screen_by_outer: dict[int, list[dict[str, Any]]] = {}
            for outer in OUTERS:
                screen_by_outer[outer] = []
                for start in range(0, len(_all_masks()), SCREEN_SHARD_SIZE):
                    if stop or _time_remaining(checkpoint, started) < FINALIZATION_RESERVE_SECONDS:
                        checkpoint["active_seconds"] += time.monotonic() - started
                        checkpoint["status"] = "safely_stopped"
                        _write_checkpoint(output, checkpoint)
                        return {"status": "safely_stopped", "resume": "rerun identical command"}
                    masks = list(_all_masks()[start : start + SCREEN_SHARD_SIZE])
                    unit = _screen_shard(output, matrix, splits, families, outer, masks, args.workers)
                    screen_by_outer[outer].append(unit)
                    checkpoint = _record_unit(checkpoint, output, unit)
                    if len(checkpoint["completed_units"]) % 8 == 0:
                        _resource_gate(output)
            promoted_by_outer: dict[int, list[int]] = {}
            hpo_by_outer: dict[int, list[dict[str, Any]]] = {}
            for outer in OUTERS:
                screen, _ = _screen_frames(output, outer)
                promoted_by_outer[outer] = _promote(screen)
                hpo_by_outer[outer] = []
                for mask in promoted_by_outer[outer]:
                    for profile in _profiles():
                        if stop or _time_remaining(checkpoint, started) < FINALIZATION_RESERVE_SECONDS:
                            checkpoint["active_seconds"] += time.monotonic() - started
                            checkpoint["status"] = "safely_stopped"
                            _write_checkpoint(output, checkpoint)
                            return {"status": "safely_stopped", "resume": "rerun identical command"}
                        unit = _hpo_unit(output, matrix, splits, families, outer, mask, profile, args.workers)
                        hpo_by_outer[outer].append(unit)
                        checkpoint = _record_unit(checkpoint, output, unit)
            outer_units: list[dict[str, Any]] = []
            for outer in OUTERS:
                unit = _outer_unit(
                    output,
                    matrix,
                    splits,
                    families,
                    outer,
                    promoted_by_outer[outer],
                    hpo_by_outer[outer],
                    args.workers,
                )
                outer_units.append(unit)
                checkpoint = _record_unit(checkpoint, output, unit)
            checkpoint["active_seconds"] += time.monotonic() - started
            return _aggregate(
                output,
                checkpoint,
                outer_units,
                matrix,
                families,
                v7_root,
                [unit for units in hpo_by_outer.values() for unit in units],
                args.workers,
            )
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)


def _validate(output: Path) -> dict[str, Any]:
    manifest = _read_json(output / "manifest.json", "manifest_sha256")
    if manifest.get("status") != "passed":
        raise CampaignError("manifest did not pass")
    for binding in manifest["input_bindings"]:
        _verify_binding(binding)
    for binding in manifest["artifacts"]:
        _verify_binding(binding, output)
        if str(binding["role"]).startswith("unit::"):
            unit = _read_json(Path(binding["path"]), "unit_json_sha256")
            if unit.get("status") != "passed":
                raise CampaignError("unit did not pass")
    nested = pd.read_parquet(output / "nested_oof_predictions.parquet")
    splits = pd.read_parquet(output / "prepared/fixed_nested_scaffold_splits.parquet")
    if len(nested) != EXACT_ROWS or nested.structure_id.duplicated().any():
        raise CampaignError("nested identity coverage failed")
    for outer in OUTERS:
        expected = set(
            splits.loc[splits.outer_fold.eq(outer) & splits.outer_role.eq("heldout"), "structure_id"].astype(
                str
            )
        )
        actual = set(nested.loc[nested.outer_fold.eq(outer), "structure_id"].astype(str))
        if expected != actual:
            raise CampaignError(f"nested fold {outer} differs from canonical split")
    lattice = pd.read_parquet(output / "complete_lattice_metrics.parquet")
    if (
        len(lattice) != 5 * (1 << len(BLOCKS))
        or lattice.groupby("outer_fold").coalition_mask.nunique().ne(1 << len(BLOCKS)).any()
    ):
        raise CampaignError("complete feature lattice is incomplete")
    scope = manifest["scientific_scope"]
    if (
        scope.get("repository_validation_labels_opened") is not False
        or scope.get("repository_test_labels_opened") is not False
    ):
        raise CampaignError("label-blind scope failed")
    return _atomic_json(
        output / "validation.json",
        {
            "schema_version": SCHEMA_VERSION,
            "status": "passed",
            "created_utc": _utc(),
            "exact_train_structures": EXACT_ROWS,
            "outer_folds": 5,
            "feature_blocks": len(BLOCKS),
            "coalitions_per_outer": 1 << len(BLOCKS),
            "total_screen_coalitions": 5 * (1 << len(BLOCKS)),
            "repository_validation_labels_opened": False,
            "repository_test_labels_opened": False,
            "manifest_sha256_bound": manifest["manifest_sha256"],
        },
        "validation_sha256",
    )


def _status(output: Path) -> dict[str, Any]:
    checkpoint = _read_json(output.resolve() / "checkpoint.json", "checkpoint_sha256")
    expected_screen = 5 * math.ceil((1 << len(BLOCKS)) / SCREEN_SHARD_SIZE)
    expected_hpo = 5 * PROMOTED_COALITIONS * len(_profiles())
    return {
        "status": checkpoint["status"],
        "completed_units": len(checkpoint.get("completed_units", [])),
        "planned_units": expected_screen + expected_hpo + 5,
        "active_hours": float(checkpoint.get("active_seconds", 0.0)) / 3600,
        "updated_utc": checkpoint.get("updated_utc"),
        "output_root": checkpoint["output_root"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--repo-root", default=".")
    run.add_argument("--matrix-root", default=str(DEFAULT_MATRIX_ROOT))
    run.add_argument("--base-root", default=str(DEFAULT_BASE_ROOT))
    run.add_argument("--v7-root", default=str(DEFAULT_V7_ROOT))
    run.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    run.add_argument("--workers", type=int, default=6)
    for name in ("status", "validate"):
        command = commands.add_parser(name)
        command.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        if args.command == "run":
            if not 1 <= args.workers <= 6:
                raise CampaignError("workers must be in [1,6]")
            result = _run(args)
        elif args.command == "status":
            result = _status(Path(args.output_root))
        else:
            result = _validate(Path(args.output_root).resolve())
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except (CampaignError, OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        print(json.dumps({"status": "failed", "error": str(error)}, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "6")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    raise SystemExit(main())
