#!/usr/bin/env python3
"""Run the train-only hERG domain-mixture and MMP campaign V9.

V9 is deliberately narrower than the preceding feature-lattice campaigns.  It
uses the frozen V8 matrix and scaffold folds to test whether cross-fitted
domain specialists, measurement-noise handling, selective ligand physics, and
matched-molecular-pair (MMP) delta learning improve the honest V8 benchmark.

Repository validation and test outcomes are never opened.  All selection,
stacking, gating, calibration, and thresholds are fitted inside the training
partition.  The result is internal scaffold-held-out evidence, not external or
prospective validation and not proof of clinical utility or biological cause.
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
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from lightgbm import LGBMRegressor
from rdkit import DataStructs
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

SCHEMA_VERSION = "platform-local-herg-domain-mixture-campaign-v9/1.0"
EXACT_ROWS = 18_801
OUTERS = tuple(range(5))
INNERS = tuple(range(3))
SEED = 20260817
V8_MAE = 0.444135
FINALIZER_RECOVERY_CONFIRMATION = "v9-complete-prediction-columns-finalizer/1"
ORIGINAL_IMPLEMENTATION_BYTES = 68_317
ORIGINAL_IMPLEMENTATION_SHA256 = "93ea0549ae1313c8e48b15e2e59843640aeaae9babb8ceeb7cf8ad1723a4a8fb"
MMP_COLUMNS = (
    "delta_molecular_weight_b_minus_a",
    "delta_mol_logp_b_minus_a",
    "delta_topological_polar_surface_area_b_minus_a",
    "delta_hydrogen_bond_donors_b_minus_a",
    "delta_hydrogen_bond_acceptors_b_minus_a",
    "delta_rotatable_bonds_b_minus_a",
    "delta_fraction_csp3_b_minus_a",
    "delta_standardized_logp_x_tpsa_b_minus_a",
)


class CampaignError(RuntimeError):
    """Scientific, leakage, resource, or integrity contract failure."""


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    engine: str
    surface: str
    params: dict[str, Any]
    objective: str = "squared"
    target_mode: str = "canonical"
    weight_mode: str = "uniform"


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
    if not path.is_file() or path.stat().st_size != int(binding["bytes"]):
        raise CampaignError(f"artifact missing or size changed: {path}")
    if _sha(path) != binding["sha256"]:
        raise CampaignError(f"artifact hash changed: {path}")
    if path.suffix == ".parquet":
        if pq.read_metadata(path).num_rows != int(binding["rows"]):
            raise CampaignError(f"row count changed: {path}")
        if _schema_sha(path) != binding["arrow_schema_sha256"]:
            raise CampaignError(f"schema changed: {path}")


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
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CampaignError(f"expected JSON object: {path}")
    if key and value.get(key) != _self_hash(value, key)[key]:
        raise CampaignError(f"self-hash mismatch: {path}")
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
        self.handle = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise CampaignError(f"campaign already running: {self.path}") from error
        return self

    def __exit__(self, *_: object) -> None:
        fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        self.handle.close()


def _candidate_plan(smoke: bool = False) -> list[Candidate]:
    anchor = {
        "n_estimators": 120 if smoke else 1200,
        "max_depth": 8,
        "learning_rate": 0.02,
        "min_child_weight": 8.0,
        "subsample": 0.75,
        "colsample_bytree": 0.60,
        "reg_alpha": 0.50,
        "reg_lambda": 5.0,
        "max_bin": 96,
    }

    def xgb(identifier: str, **changes: Any) -> Candidate:
        params = dict(anchor)
        params.update(changes)
        return Candidate(identifier, "xgboost", "anchor", params)

    result = [
        xgb("xgb_v2_anchor"),
        xgb("xgb_depth5", max_depth=5),
        xgb("xgb_depth10", max_depth=10),
        xgb("xgb_lr015", learning_rate=0.015, n_estimators=160 if smoke else 1600),
        xgb("xgb_lr035", learning_rate=0.035, n_estimators=90 if smoke else 800),
        xgb("xgb_child4", min_child_weight=4.0),
        xgb("xgb_child16", min_child_weight=16.0),
        xgb("xgb_col05", colsample_bytree=0.50),
        xgb("xgb_col08", colsample_bytree=0.80),
        xgb("xgb_row09", subsample=0.90),
        xgb("xgb_regularized", max_depth=6, reg_alpha=1.5, reg_lambda=12.0),
        xgb("xgb_weak_regularization", reg_alpha=0.05, reg_lambda=1.0),
        Candidate(
            "lgb_l1",
            "lightgbm",
            "anchor",
            {
                "n_estimators": 120 if smoke else 900,
                "num_leaves": 31,
                "learning_rate": 0.025,
                "min_child_samples": 30,
                "subsample": 0.8,
                "colsample_bytree": 0.7,
                "reg_alpha": 0.5,
                "reg_lambda": 5.0,
                "max_bin": 127,
            },
            objective="absolute",
        ),
        Candidate(
            "lgb_huber",
            "lightgbm",
            "anchor",
            {
                "n_estimators": 120 if smoke else 900,
                "num_leaves": 31,
                "learning_rate": 0.025,
                "min_child_samples": 30,
                "subsample": 0.8,
                "colsample_bytree": 0.7,
                "reg_alpha": 0.5,
                "reg_lambda": 5.0,
                "max_bin": 127,
                "alpha": 0.85,
            },
            objective="huber",
        ),
        Candidate(
            "lgb_l2",
            "lightgbm",
            "anchor",
            {
                "n_estimators": 120 if smoke else 900,
                "num_leaves": 31,
                "learning_rate": 0.025,
                "min_child_samples": 30,
                "subsample": 0.8,
                "colsample_bytree": 0.7,
                "reg_alpha": 0.5,
                "reg_lambda": 5.0,
                "max_bin": 127,
            },
        ),
        Candidate(
            "extratrees_rdkit2d",
            "extratrees",
            "rdkit2d",
            {
                "n_estimators": 80 if smoke else 600,
                "max_features": 0.75,
                "min_samples_leaf": 2,
                "max_depth": None,
            },
        ),
        Candidate("xgb_selective_physics", "xgboost", "physics", dict(anchor)),
        Candidate("xgb_heavy_flexible", "xgboost", "physics", dict(anchor), weight_mode="heavy_flexible"),
        Candidate("xgb_potency_tail", "xgboost", "anchor", dict(anchor), weight_mode="potency_tail"),
        Candidate("xgb_cliff_risk", "xgboost", "anchor", dict(anchor), weight_mode="cliff_risk"),
        Candidate("xgb_reliability_weighted", "xgboost", "anchor", dict(anchor), weight_mode="reliability"),
        Candidate(
            "xgb_hierarchical_target",
            "xgboost",
            "anchor",
            dict(anchor),
            target_mode="hierarchical",
            weight_mode="reliability",
        ),
    ]
    return result[:4] if smoke else result


def _surface_columns(blocks: dict[str, list[str]]) -> dict[str, list[str]]:
    anchor = blocks["rdkit2d"] + blocks["morgan"]
    physics = anchor + blocks["polarity_charge_internal_contacts"] + blocks["energy_flexibility"]
    physics += blocks["shape"] + blocks["autocorr3d"] + blocks["selected_interactions"]
    return {
        "anchor": list(dict.fromkeys(anchor)),
        "rdkit2d": blocks["rdkit2d"],
        "physics": list(dict.fromkeys(physics)),
    }


def _metrics(observed: Iterable[float], predicted: Iterable[float]) -> dict[str, Any]:
    y = np.asarray(observed, dtype=float)
    p = np.asarray(predicted, dtype=float)
    absolute = np.abs(y - p)
    correlation = spearmanr(y, p).statistic if len(y) > 1 else math.nan
    return {
        "n": int(len(y)),
        "mae": float(np.mean(absolute)),
        "rmse": float(math.sqrt(mean_squared_error(y, p))),
        "median_absolute_error": float(np.median(absolute)),
        "spearman": None if not math.isfinite(float(correlation)) else float(correlation),
        "fraction_within_0p5": float(np.mean(absolute <= 0.5)),
        "fraction_within_1p0": float(np.mean(absolute <= 1.0)),
        "tail_mae": float(np.mean(absolute[(y < 4.5) | (y >= 6.5)])),
    }


def _clean_matrix(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    values = frame[columns].to_numpy(dtype=np.float32, copy=True)
    values[~np.isfinite(values)] = np.nan
    values[np.abs(values) > 1e30] = np.nan
    return pd.DataFrame(values, columns=columns, index=frame.index)


def _new_model(candidate: Candidate, workers: int, seed: int) -> Any:
    if candidate.engine == "xgboost":
        objective = "reg:absoluteerror" if candidate.objective == "absolute" else "reg:squarederror"
        return XGBRegressor(
            objective=objective,
            tree_method="hist",
            n_jobs=workers,
            random_state=seed,
            verbosity=0,
            **candidate.params,
        )
    if candidate.engine == "lightgbm":
        objective = {"absolute": "regression_l1", "huber": "huber"}.get(candidate.objective, "regression")
        return LGBMRegressor(
            objective=objective,
            n_jobs=workers,
            random_state=seed,
            verbosity=-1,
            subsample_freq=1,
            **candidate.params,
        )
    if candidate.engine == "extratrees":
        model = ExtraTreesRegressor(n_jobs=workers, random_state=seed, **candidate.params)
        return Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=True)), ("model", model)])
    raise CampaignError(f"unsupported engine: {candidate.engine}")


def _measurement_targets(observations: pd.DataFrame, fit_ids: set[str], mode: str) -> pd.DataFrame:
    part = observations.loc[observations.structure_id.astype(str).isin(fit_ids)].copy()
    if mode == "hierarchical":
        centers = part.groupby("structure_id", observed=True).potency_pic50_point.transform("median")
        residual = part.potency_pic50_point.to_numpy(float) - centers.to_numpy(float)
        correction = np.zeros(len(part), dtype=float)
        for column, prior in (
            ("measurement_modality", 35.0),
            ("assay_family", 45.0),
            ("source_family", 50.0),
            ("automation_class", 35.0),
        ):
            keys = part[column].astype("string").fillna("<missing>")
            table = pd.DataFrame({"key": keys, "residual": residual - correction})
            grouped = table.groupby("key", observed=True).residual.agg(["mean", "count"])
            grouped["offset"] = grouped["mean"] * grouped["count"] / (grouped["count"] + prior)
            correction += keys.map(grouped.offset).fillna(0.0).to_numpy(float)
        part["working_target"] = part.potency_pic50_point.to_numpy(float) - correction
    else:
        part["working_target"] = part.potency_pic50_point
    rows: list[dict[str, Any]] = []
    for sid, group in part.groupby("structure_id", observed=True, sort=False):
        values = group.working_target.to_numpy(float)
        raw = group.potency_pic50_point.to_numpy(float)
        spread = float(np.ptp(raw)) if len(raw) > 1 else 0.0
        weight = math.sqrt(len(raw)) / (1.0 + spread)
        if group.v1_5_conflict_review_structure.any():
            weight *= 0.45
        if group.evaluation_or_lineage_leakage_caution.any():
            weight *= 0.65
        rows.append(
            {
                "structure_id": str(sid),
                "training_target": float(np.median(values)),
                "replicate_range": spread,
                "observation_count": len(raw),
                "reliability_weight": float(np.clip(weight, 0.15, 3.0)),
            }
        )
    return pd.DataFrame(rows)


def _fit_predict(
    matrix: pd.DataFrame,
    observations: pd.DataFrame,
    mmp: pd.DataFrame,
    columns: list[str],
    fit_ids: set[str],
    eval_ids: set[str],
    candidate: Candidate,
    workers: int,
    seed: int,
) -> tuple[Any, pd.DataFrame, float]:
    fit = matrix.loc[matrix.structure_id.astype(str).isin(fit_ids)].copy()
    evaluation = matrix.loc[matrix.structure_id.astype(str).isin(eval_ids)].copy()
    if set(fit.scaffold_group_id.astype(str)) & set(evaluation.scaffold_group_id.astype(str)):
        raise CampaignError("scaffold leakage in model fit")
    targets = _measurement_targets(observations, fit_ids, candidate.target_mode)
    joined = fit.merge(targets, on="structure_id", validate="one_to_one")
    y = joined.training_target.to_numpy(float)
    weights = np.ones(len(joined), dtype=float)
    if candidate.weight_mode == "reliability":
        weights *= joined.reliability_weight.to_numpy(float)
    elif candidate.weight_mode == "potency_tail":
        weights *= 1.0 + 1.25 * ((y < 4.5) | (y >= 6.5))
    elif candidate.weight_mode == "heavy_flexible":
        mw = joined["rdkit2d__MolWt"].fillna(0).to_numpy(float)
        rotors = joined["rdkit2d__NumRotatableBonds"].fillna(0).to_numpy(float)
        weights *= 1.0 + 0.75 * ((mw >= 500) | (rotors >= 8))
    elif candidate.weight_mode == "cliff_risk":
        contained = mmp.loc[
            mmp.structure_id_a.astype(str).isin(fit_ids)
            & mmp.structure_id_b.astype(str).isin(fit_ids)
            & mmp.activity_cliff_ge_1_pic50
        ]
        cliff_ids = set(contained.structure_id_a.astype(str)) | set(contained.structure_id_b.astype(str))
        weights *= 1.0 + 1.0 * joined.structure_id.astype(str).isin(cliff_ids).to_numpy(float)
    model = _new_model(candidate, workers, seed)
    started = time.monotonic()
    fit_kwargs = (
        {"model__sample_weight": weights} if isinstance(model, Pipeline) else {"sample_weight": weights}
    )
    model.fit(_clean_matrix(joined, columns), y, **fit_kwargs)
    prediction = np.asarray(model.predict(_clean_matrix(evaluation, columns)), dtype=float)
    elapsed = time.monotonic() - started
    result = evaluation[["structure_id", "scaffold_group_id", "target_pic50"]].copy()
    result = result.rename(columns={"target_pic50": "observed_pic50"})
    result["predicted_pic50"] = prediction
    return model, result, elapsed


def _static_gate(frame: pd.DataFrame) -> pd.DataFrame:
    names = [
        "rdkit2d__MolWt",
        "rdkit2d__MolLogP",
        "rdkit2d__TPSA",
        "rdkit2d__NumRotatableBonds",
        "rdkit2d__HeavyAtomCount",
        "new3d__formal_charge",
        "new3d__feature_failed_indicator",
        "new3d__all_retained_unconverged_indicator",
        "new3d__convergence_fraction",
    ]
    available = [name for name in names if name in frame]
    result = frame[available].astype(float).copy()
    result.columns = [f"gate__{name}" for name in available]
    return result


def _unit(
    directory: Path,
    unit_id: str,
    spec: dict[str, Any],
    metrics: dict[str, Any],
    artifacts: list[dict[str, Any]],
) -> dict[str, Any]:
    return _atomic_json(
        directory / "unit.json",
        {
            "schema_version": SCHEMA_VERSION,
            "status": "passed",
            "created_utc": _utc(),
            "unit_id": unit_id,
            "unit_spec": spec,
            "unit_spec_sha256": _digest(spec),
            "metrics": metrics,
            "artifacts": artifacts,
            "scientific_scope": {
                "source_partition": "train",
                "fixed_scaffold_folds": True,
                "validation_labels_opened": False,
                "test_labels_opened": False,
                "external_or_prospective_validation": False,
            },
        },
        "unit_json_sha256",
    )


def _existing(directory: Path, spec: dict[str, Any], output: Path) -> dict[str, Any] | None:
    path = directory / "unit.json"
    if not path.is_file():
        return None
    try:
        value = _read_json(path, "unit_json_sha256")
        if value.get("status") != "passed" or value.get("unit_spec") != spec:
            return None
        for binding in value["artifacts"]:
            _verify_binding(binding, output)
        return value
    except Exception:
        return None


def _artifact(unit: dict[str, Any], role: str) -> Path:
    matches = [Path(row["path"]) for row in unit["artifacts"] if row["role"] == role]
    if len(matches) != 1:
        raise CampaignError(f"unit {unit['unit_id']} lacks unique role {role}")
    return matches[0]


def _prepare(repo: Path, v8: Path, v81: Path, mmp_root: Path, output: Path) -> dict[str, Any]:
    target = output / "prepared" / "validation.json"
    if target.is_file():
        value = _read_json(target, "validation_sha256")
        for row in value["inputs"]:
            _verify_binding(row)
        for row in value["artifacts"]:
            _verify_binding(row, output)
        return value
    source_validation = _read_json(v8 / "validation.json", "validation_sha256")
    if source_validation.get("status") != "passed":
        raise CampaignError("V8 validation did not pass")
    matrix = pd.read_parquet(v8 / "prepared/training_matrix.parquet")
    splits = pd.read_parquet(v8 / "prepared/fixed_nested_scaffold_splits.parquet")
    blocks_doc = _read_json(v8 / "prepared/feature_blocks.json", "feature_blocks_sha256")
    observations = pd.read_parquet(
        repo / "research/local_runs/herg_honest_measurement_campaign_v7_1/prepared/"
        "exact_train_observations.parquet"
    )
    mmp = pd.read_parquet(mmp_root / "training_mmp_effects.parquet")
    v1_oof = pd.read_parquet(
        repo / "research/local_runs/herg_discovery_campaign_v1/analysis/outer_oof_predictions.parquet"
    )
    ad = v1_oof.loc[
        v1_oof.model_id.eq("similarity_tanimoto_knn"),
        [
            "structure_id",
            "outer_fold",
            "maximum_train_tanimoto",
            "mean_neighbor_tanimoto",
            "effective_neighbors",
        ],
    ].copy()
    if len(matrix) != EXACT_ROWS or matrix.structure_id.duplicated().any():
        raise CampaignError("V8 matrix is not 18,801 unique structures")
    ids = set(matrix.structure_id.astype(str))
    if set(observations.structure_id.astype(str)) != ids:
        raise CampaignError("observation identities do not match matrix")
    if len(splits) != EXACT_ROWS * 5 or len(ad) != EXACT_ROWS:
        raise CampaignError("split or applicability-domain coverage is invalid")
    for outer in OUTERS:
        part = splits.loc[splits.outer_fold.eq(outer)]
        if set(part.structure_id.astype(str)) != ids or part.structure_id.duplicated().any():
            raise CampaignError(f"outer {outer} identity coverage failed")
        fit_scaf = set(part.loc[part.outer_role.eq("fit"), "scaffold_group_id"].astype(str))
        held_scaf = set(part.loc[part.outer_role.eq("heldout"), "scaffold_group_id"].astype(str))
        if fit_scaf & held_scaf:
            raise CampaignError(f"outer {outer} scaffold leakage")
    prepared = output / "prepared"
    prepared.mkdir(parents=True, exist_ok=True)
    paths = {
        "training_matrix": prepared / "training_matrix.parquet",
        "splits": prepared / "fixed_nested_scaffold_splits.parquet",
        "observations": prepared / "exact_train_observations.parquet",
        "mmp": prepared / "training_mmp_effects.parquet",
        "ad": prepared / "outer_applicability_domain.parquet",
        "blocks": prepared / "feature_blocks.json",
    }
    _atomic_parquet(paths["training_matrix"], matrix)
    _atomic_parquet(paths["splits"], splits)
    _atomic_parquet(paths["observations"], observations)
    _atomic_parquet(paths["mmp"], mmp)
    _atomic_parquet(paths["ad"], ad)
    _atomic_json(
        paths["blocks"],
        {"schema_version": SCHEMA_VERSION, "blocks": blocks_doc["blocks"]},
        "feature_blocks_sha256",
    )
    implementation = Path(__file__).resolve()
    input_paths = [
        v8 / "validation.json",
        v8 / "nested_oof_predictions.parquet",
        v81 / "validation.json",
        mmp_root / "mmp_analysis_manifest.json",
    ]
    return _atomic_json(
        target,
        {
            "schema_version": SCHEMA_VERSION,
            "status": "passed",
            "created_utc": _utc(),
            "exact_train_rows": EXACT_ROWS,
            "outer_folds": 5,
            "inner_folds": 3,
            "source_partition": "train",
            "validation_labels_opened": False,
            "test_labels_opened": False,
            "inputs": [_binding(path, f"input_{index}") for index, path in enumerate(input_paths)]
            + [_binding(implementation, "implementation")],
            "artifacts": [_binding(path, role) for role, path in paths.items()],
        },
        "validation_sha256",
    )


def _load(
    output: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, list[str]]]:
    validation = _read_json(output / "prepared/validation.json", "validation_sha256")
    for row in validation["artifacts"]:
        _verify_binding(row, output)
    blocks = _read_json(output / "prepared/feature_blocks.json", "feature_blocks_sha256")["blocks"]
    return (
        pd.read_parquet(output / "prepared/training_matrix.parquet"),
        pd.read_parquet(output / "prepared/fixed_nested_scaffold_splits.parquet"),
        pd.read_parquet(output / "prepared/exact_train_observations.parquet"),
        pd.read_parquet(output / "prepared/training_mmp_effects.parquet"),
        pd.read_parquet(output / "prepared/outer_applicability_domain.parquet"),
        blocks,
    )


def _inner_unit(
    output: Path,
    matrix: pd.DataFrame,
    splits: pd.DataFrame,
    observations: pd.DataFrame,
    mmp: pd.DataFrame,
    surfaces: dict[str, list[str]],
    candidate: Candidate,
    outer: int,
    workers: int,
) -> dict[str, Any]:
    unit_id = f"inner_o{outer}__{candidate.candidate_id}"
    directory = output / "units" / unit_id
    spec = {
        "operation": "inner_scaffold_oof",
        "outer_fold": outer,
        "inner_folds": list(INNERS),
        "candidate": asdict(candidate),
        "feature_count": len(surfaces[candidate.surface]),
    }
    existing = _existing(directory, spec, output)
    if existing is not None:
        return existing
    registry = splits.loc[splits.outer_fold.eq(outer) & splits.outer_role.eq("fit")]
    frames: list[pd.DataFrame] = []
    elapsed = 0.0
    for inner in INNERS:
        fit_ids = set(registry.loc[registry.inner_fold.ne(inner), "structure_id"].astype(str))
        eval_ids = set(registry.loc[registry.inner_fold.eq(inner), "structure_id"].astype(str))
        _model, pred, seconds = _fit_predict(
            matrix,
            observations,
            mmp,
            surfaces[candidate.surface],
            fit_ids,
            eval_ids,
            candidate,
            workers,
            SEED + outer * 1000 + inner,
        )
        pred["inner_fold"] = inner
        frames.append(pred)
        elapsed += seconds
    predictions = pd.concat(frames, ignore_index=True)
    expected = set(registry.structure_id.astype(str))
    if set(predictions.structure_id.astype(str)) != expected or predictions.structure_id.duplicated().any():
        raise CampaignError(f"inner OOF coverage failed for {unit_id}")
    path = directory / "inner_oof_predictions.parquet"
    _atomic_parquet(path, predictions)
    metrics = _metrics(predictions.observed_pic50, predictions.predicted_pic50)
    metrics["fit_elapsed_seconds"] = elapsed
    return _unit(directory, unit_id, spec, metrics, [_binding(path, "inner_oof_predictions")])


def _top_candidates(units: list[dict[str, Any]], minimum: int = 8) -> list[str]:
    rows = sorted(
        (float(unit["metrics"]["mae"]), unit["unit_spec"]["candidate"]["candidate_id"]) for unit in units
    )
    selected = [identifier for _, identifier in rows[:minimum]]
    forced = [
        "xgb_v2_anchor",
        "xgb_selective_physics",
        "xgb_heavy_flexible",
        "xgb_potency_tail",
        "xgb_cliff_risk",
        "xgb_reliability_weighted",
        "xgb_hierarchical_target",
        "xgb_regularized",
    ]
    return list(dict.fromkeys(selected + forced))


def _nearest_analogs(matrix: pd.DataFrame, fit_ids: set[str], eval_ids: set[str]) -> pd.DataFrame:
    bit_columns = [name for name in matrix if name.startswith("morgan__")]
    fit = matrix.loc[matrix.structure_id.astype(str).isin(fit_ids), ["structure_id", *bit_columns]]
    evaluation = matrix.loc[matrix.structure_id.astype(str).isin(eval_ids), ["structure_id", *bit_columns]]
    fit_fps = [
        DataStructs.CreateFromBitString("".join("1" if value else "0" for value in row))
        for row in fit[bit_columns].fillna(0).to_numpy(dtype=np.uint8)
    ]
    records: list[dict[str, Any]] = []
    fit_names = fit.structure_id.astype(str).tolist()
    for sid, row in zip(
        evaluation.structure_id.astype(str),
        evaluation[bit_columns].fillna(0).to_numpy(dtype=np.uint8),
        strict=True,
    ):
        query = DataStructs.CreateFromBitString("".join("1" if value else "0" for value in row))
        similarities = np.asarray(DataStructs.BulkTanimotoSimilarity(query, fit_fps), dtype=float)
        take = np.argpartition(similarities, -min(3, len(similarities)))[-3:]
        take = take[np.argsort(similarities[take])[::-1]]
        records.append(
            {
                "structure_id": sid,
                "nearest_analog_ids": json.dumps([fit_names[i] for i in take]),
                "nearest_analog_similarities": json.dumps([float(similarities[i]) for i in take]),
                "computed_maximum_train_tanimoto": float(similarities[take[0]]),
            }
        )
    return pd.DataFrame(records)


def _outer_unit(
    output: Path,
    matrix: pd.DataFrame,
    splits: pd.DataFrame,
    observations: pd.DataFrame,
    mmp: pd.DataFrame,
    ad: pd.DataFrame,
    surfaces: dict[str, list[str]],
    candidates: list[Candidate],
    outer: int,
    workers: int,
    inner_units: list[dict[str, Any]],
) -> dict[str, Any]:
    selected_ids = _top_candidates(inner_units)
    selected = [candidate for candidate in candidates if candidate.candidate_id in selected_ids]
    unit_id = f"outer_o{outer}"
    directory = output / "units" / unit_id
    spec = {
        "operation": "nested_outer_domain_mixture",
        "outer_fold": outer,
        "selected_candidates": [asdict(candidate) for candidate in selected],
        "selection_unit_hashes": [unit["unit_json_sha256"] for unit in inner_units],
        "gating_inputs_label_blind": True,
        "stack_fitted_on_inner_oof_only": True,
    }
    existing = _existing(directory, spec, output)
    if existing is not None:
        return existing
    registry = splits.loc[splits.outer_fold.eq(outer)]
    fit_ids = set(registry.loc[registry.outer_role.eq("fit"), "structure_id"].astype(str))
    held_ids = set(registry.loc[registry.outer_role.eq("heldout"), "structure_id"].astype(str))
    inner_frames: list[pd.DataFrame] = []
    for unit in inner_units:
        cid = unit["unit_spec"]["candidate"]["candidate_id"]
        if cid not in selected_ids:
            continue
        frame = pd.read_parquet(_artifact(unit, "inner_oof_predictions"))
        inner_frames.append(
            frame[["structure_id", "observed_pic50", "predicted_pic50"]].rename(
                columns={"predicted_pic50": f"pred__{cid}"}
            )
        )
    inner = inner_frames[0]
    for frame in inner_frames[1:]:
        inner = inner.merge(frame, on=["structure_id", "observed_pic50"], validate="one_to_one")
    outer_predictions: list[pd.DataFrame] = []
    models: dict[str, Any] = {}
    elapsed = 0.0
    for index, candidate in enumerate(selected):
        model, frame, seconds = _fit_predict(
            matrix,
            observations,
            mmp,
            surfaces[candidate.surface],
            fit_ids,
            held_ids,
            candidate,
            workers,
            SEED + outer * 1000 + 500 + index,
        )
        models[candidate.candidate_id] = model
        frame = frame.rename(columns={"predicted_pic50": f"pred__{candidate.candidate_id}"})
        outer_predictions.append(frame)
        elapsed += seconds
    outer_frame = outer_predictions[0]
    for frame in outer_predictions[1:]:
        outer_frame = outer_frame.merge(
            frame, on=["structure_id", "scaffold_group_id", "observed_pic50"], validate="one_to_one"
        )
    prediction_columns = [name for name in outer_frame if name.startswith("pred__")]
    inner_prediction_columns = [name for name in inner if name.startswith("pred__")]
    if prediction_columns != inner_prediction_columns:
        outer_frame = outer_frame[
            ["structure_id", "scaffold_group_id", "observed_pic50", *inner_prediction_columns]
        ]
        prediction_columns = inner_prediction_columns
    ridge = RidgeCV(alphas=np.logspace(-4, 3, 30), fit_intercept=True)
    ridge.fit(inner[prediction_columns], inner.observed_pic50)
    outer_frame["pred__honest_stack"] = ridge.predict(outer_frame[prediction_columns])
    inner["mean_prediction"] = inner[prediction_columns].mean(axis=1)
    inner["prediction_spread"] = inner[prediction_columns].std(axis=1)
    outer_frame["mean_prediction"] = outer_frame[prediction_columns].mean(axis=1)
    outer_frame["prediction_spread"] = outer_frame[prediction_columns].std(axis=1)
    matrix_index = matrix.set_index("structure_id")
    inner_static = _static_gate(matrix_index.loc[inner.structure_id].reset_index(drop=True))
    outer_static = _static_gate(matrix_index.loc[outer_frame.structure_id].reset_index(drop=True))
    gate_columns = [*prediction_columns, "mean_prediction", "prediction_spread"]
    gate_inner = pd.concat([inner[gate_columns].reset_index(drop=True), inner_static], axis=1)
    gate_outer = pd.concat([outer_frame[gate_columns].reset_index(drop=True), outer_static], axis=1)
    expected_errors: dict[str, np.ndarray] = {}
    for index, column in enumerate(prediction_columns):
        error_model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                (
                    "model",
                    ExtraTreesRegressor(
                        n_estimators=300,
                        min_samples_leaf=15,
                        max_features=0.8,
                        n_jobs=workers,
                        random_state=SEED + outer * 100 + index,
                    ),
                ),
            ]
        )
        error_model.fit(gate_inner, np.abs(inner.observed_pic50 - inner[column]))
        expected_errors[column] = np.maximum(0.03, error_model.predict(gate_outer))
    error_matrix = np.column_stack([expected_errors[column] for column in prediction_columns])
    selected_index = np.argmin(error_matrix, axis=1)
    pred_matrix = outer_frame[prediction_columns].to_numpy(float)
    outer_frame["pred__learned_gate"] = pred_matrix[np.arange(len(outer_frame)), selected_index]
    outer_frame["selected_expert"] = [
        prediction_columns[index].removeprefix("pred__") for index in selected_index
    ]
    sorted_error = np.sort(error_matrix, axis=1)
    outer_frame["predicted_expert_error"] = sorted_error[:, 0]
    outer_frame["predicted_expert_advantage"] = sorted_error[:, 1] - sorted_error[:, 0]
    anchor_column = "pred__xgb_v2_anchor"
    regularized = "pred__xgb_regularized"
    physics = "pred__xgb_selective_physics"
    if regularized not in outer_frame:
        regularized = anchor_column
    if physics not in outer_frame:
        physics = anchor_column
    domain = ad.loc[ad.outer_fold.eq(outer)].copy()
    outer_frame = outer_frame.merge(domain, on="structure_id", validate="one_to_one")
    outer_frame["pred__fixed_domain_gate"] = np.where(
        outer_frame.maximum_train_tanimoto.lt(0.60), outer_frame[regularized], outer_frame[anchor_column]
    )
    outer_frame["physics_correction"] = np.where(
        outer_frame.selected_expert.eq("xgb_selective_physics"),
        outer_frame[physics] - outer_frame[anchor_column],
        0.0,
    )
    inner_predicted = ridge.predict(inner[prediction_columns])
    conformal = float(np.quantile(np.abs(inner.observed_pic50 - inner_predicted), 0.90, method="higher"))
    outer_frame["interval90_lower"] = outer_frame.pred__honest_stack - conformal
    outer_frame["interval90_upper"] = outer_frame.pred__honest_stack + conformal
    outer_frame["interval90_half_width"] = conformal
    outer_frame["extrapolation_or_abstention_flag"] = outer_frame.maximum_train_tanimoto.lt(
        0.50
    ) | outer_frame.predicted_expert_error.gt(np.quantile(error_matrix[:, 0], 0.90))
    outer_frame["outer_fold"] = outer
    analogs = _nearest_analogs(matrix, fit_ids, held_ids)
    outer_frame = outer_frame.merge(analogs, on="structure_id", validate="one_to_one")
    if (
        np.max(np.abs(outer_frame.maximum_train_tanimoto - outer_frame.computed_maximum_train_tanimoto))
        > 1e-6
    ):
        raise CampaignError(f"outer {outer} independent similarity replay mismatch")
    predictions_path = directory / "outer_predictions.parquet"
    _atomic_parquet(predictions_path, outer_frame)
    model_path = directory / "outer_models.joblib"
    joblib.dump(
        {"models": models, "stack": ridge, "columns": prediction_columns, "conformal90": conformal},
        model_path,
        compress=3,
    )
    metrics: dict[str, Any] = {
        column.removeprefix("pred__"): _metrics(outer_frame.observed_pic50, outer_frame[column])
        for column in [
            *prediction_columns,
            "pred__honest_stack",
            "pred__learned_gate",
            "pred__fixed_domain_gate",
        ]
    }
    metrics["fit_elapsed_seconds"] = elapsed
    return _unit(
        directory,
        unit_id,
        spec,
        metrics,
        [_binding(predictions_path, "outer_predictions"), _binding(model_path, "outer_models")],
    )


def _components(mmp: pd.DataFrame) -> dict[str, str]:
    parent: dict[str, str] = {}

    def find(value: str) -> str:
        parent.setdefault(value, value)
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        a, b = find(left), find(right)
        if a != b:
            parent[b] = a

    for left, right in zip(mmp.structure_id_a.astype(str), mmp.structure_id_b.astype(str), strict=True):
        union(left, right)
    return {value: find(value) for value in parent}


def _mmp_unit(
    output: Path,
    matrix: pd.DataFrame,
    splits: pd.DataFrame,
    mmp: pd.DataFrame,
    outer: int,
    workers: int,
    outer_unit: dict[str, Any],
) -> dict[str, Any]:
    unit_id = f"mmp_o{outer}"
    directory = output / "units" / unit_id
    spec = {
        "operation": "component_exclusive_mmp_delta",
        "outer_fold": outer,
        "numeric_delta_features": list(MMP_COLUMNS),
        "connected_components_exclusive": True,
        "analog_correction_alpha": 0.5,
        "analog_correction_scope": "cross_fitted_training_side_potency_plus_label_blind_pair",
        "outer_prediction_unit_sha256": outer_unit["unit_json_sha256"],
    }
    existing = _existing(directory, spec, output)
    if existing is not None:
        return existing
    registry = splits.loc[splits.outer_fold.eq(outer)]
    fit_ids = set(registry.loc[registry.outer_role.eq("fit"), "structure_id"].astype(str))
    held_ids = set(registry.loc[registry.outer_role.eq("heldout"), "structure_id"].astype(str))
    component = _components(mmp)
    members: dict[str, set[str]] = {}
    for sid, group in component.items():
        members.setdefault(group, set()).add(sid)
    fit_components = {group for group, ids in members.items() if ids <= fit_ids}
    held_components = {group for group, ids in members.items() if ids <= held_ids}
    groups = mmp.structure_id_a.astype(str).map(component)
    training = mmp.loc[groups.isin(fit_components)].copy()
    evaluation = mmp.loc[groups.isin(held_components)].copy()
    if not training.empty and not evaluation.empty:
        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                (
                    "model",
                    ExtraTreesRegressor(
                        n_estimators=700,
                        min_samples_leaf=3,
                        max_features=0.9,
                        n_jobs=workers,
                        random_state=SEED + outer,
                    ),
                ),
            ]
        )
        model.fit(training[list(MMP_COLUMNS)], training.delta_pic50_b_minus_a)
        evaluation["predicted_delta_pic50_b_minus_a"] = model.predict(evaluation[list(MMP_COLUMNS)])
        evaluation["predicted_direction_correct"] = np.sign(
            evaluation.predicted_delta_pic50_b_minus_a
        ) == np.sign(evaluation.delta_pic50_b_minus_a)
    else:
        raise CampaignError(f"outer {outer} lacks component-exclusive MMP support")
    path = directory / "mmp_delta_predictions.parquet"
    _atomic_parquet(path, evaluation)

    # Component-exclusive pairs above provide the unbiased delta-model diagnostic.
    # Analog assistance is a distinct cross-fitted prediction mode: it may use a
    # label-blind MMP edge between a fit molecule and a held-out molecule, but it
    # never uses the held-out potency to construct that prediction.
    left_fit = mmp.structure_id_a.astype(str).isin(fit_ids)
    right_fit = mmp.structure_id_b.astype(str).isin(fit_ids)
    left_held = mmp.structure_id_a.astype(str).isin(held_ids)
    right_held = mmp.structure_id_b.astype(str).isin(held_ids)
    cross = mmp.loc[(left_fit & right_held) | (right_fit & left_held)].copy()
    if cross.empty:
        raise CampaignError(f"outer {outer} lacks cross-fitted MMP analog support")
    cross["predicted_delta_pic50_b_minus_a"] = model.predict(cross[list(MMP_COLUMNS)])
    a_is_fit = cross.structure_id_a.astype(str).isin(fit_ids)
    cross["heldout_structure_id"] = np.where(
        a_is_fit, cross.structure_id_b.astype(str), cross.structure_id_a.astype(str)
    )
    cross["training_analog_structure_id"] = np.where(
        a_is_fit, cross.structure_id_a.astype(str), cross.structure_id_b.astype(str)
    )
    cross["analog_estimated_pic50"] = np.where(
        a_is_fit,
        cross.pic50_median_a + cross.predicted_delta_pic50_b_minus_a,
        cross.pic50_median_b - cross.predicted_delta_pic50_b_minus_a,
    )
    analog_summary = (
        cross.groupby("heldout_structure_id", observed=True)
        .agg(
            mmp_analog_estimated_pic50=("analog_estimated_pic50", "median"),
            mmp_training_analog_count=("training_analog_structure_id", "nunique"),
            mmp_cross_pair_count=("pair_id", "nunique"),
        )
        .reset_index()
        .rename(columns={"heldout_structure_id": "structure_id"})
    )
    analog_ids = cross.groupby("heldout_structure_id", observed=True).training_analog_structure_id.apply(
        lambda values: json.dumps(sorted(set(map(str, values)))[:10])
    )
    analog_summary["mmp_training_analog_ids"] = analog_summary.structure_id.map(analog_ids)
    outer_predictions = pd.read_parquet(_artifact(outer_unit, "outer_predictions"))[
        ["structure_id", "observed_pic50", "pred__xgb_v2_anchor"]
    ]
    assisted = outer_predictions.merge(analog_summary, on="structure_id", how="left", validate="one_to_one")
    assisted["mmp_analog_covered"] = assisted.mmp_analog_estimated_pic50.notna()
    assisted["predicted_pic50_mmp_assisted"] = np.where(
        assisted.mmp_analog_covered,
        0.5 * assisted.pred__xgb_v2_anchor + 0.5 * assisted.mmp_analog_estimated_pic50,
        assisted.pred__xgb_v2_anchor,
    )
    assisted_path = directory / "mmp_analog_assisted_predictions.parquet"
    _atomic_parquet(assisted_path, assisted)
    delta_metrics = _metrics(evaluation.delta_pic50_b_minus_a, evaluation.predicted_delta_pic50_b_minus_a)
    delta_metrics["direction_accuracy"] = float(evaluation.predicted_direction_correct.mean())
    delta_metrics["cliff_delta_mae"] = float(
        np.mean(
            np.abs(
                evaluation.loc[evaluation.activity_cliff_ge_1_pic50, "delta_pic50_b_minus_a"]
                - evaluation.loc[evaluation.activity_cliff_ge_1_pic50, "predicted_delta_pic50_b_minus_a"]
            )
        )
    )
    delta_metrics["training_pairs"] = len(training)
    delta_metrics["heldout_pairs"] = len(evaluation)
    delta_metrics["cross_fitted_analog_pairs"] = len(cross)
    delta_metrics["analog_covered_structures"] = int(assisted.mmp_analog_covered.sum())
    delta_metrics["analog_coverage_fraction"] = float(assisted.mmp_analog_covered.mean())
    covered = assisted.loc[assisted.mmp_analog_covered]
    if not covered.empty:
        delta_metrics["covered_anchor_mae"] = float(
            np.mean(np.abs(covered.observed_pic50 - covered.pred__xgb_v2_anchor))
        )
        delta_metrics["covered_assisted_mae"] = float(
            np.mean(np.abs(covered.observed_pic50 - covered.predicted_pic50_mmp_assisted))
        )
    return _unit(
        directory,
        unit_id,
        spec,
        delta_metrics,
        [
            _binding(path, "mmp_delta_predictions"),
            _binding(assisted_path, "mmp_analog_assisted_predictions"),
        ],
    )


def _bootstrap_delta(
    frame: pd.DataFrame, challenger: str, reference: str, replicates: int, seed: int
) -> dict[str, Any]:
    work = frame[["scaffold_group_id", "observed_pic50", challenger, reference]].copy()
    work["delta"] = np.abs(work.observed_pic50 - work[challenger]) - np.abs(
        work.observed_pic50 - work[reference]
    )
    grouped = work.groupby("scaffold_group_id", observed=True).delta.agg(["sum", "count"])
    sums, counts = grouped["sum"].to_numpy(float), grouped["count"].to_numpy(float)
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=float)
    flips = np.empty(replicates, dtype=float)
    for start in range(0, replicates, 500):
        n = min(500, replicates - start)
        indices = rng.integers(0, len(grouped), size=(n, len(grouped)))
        draws[start : start + n] = sums[indices].sum(axis=1) / counts[indices].sum(axis=1)
        signs = rng.choice((-1.0, 1.0), size=(n, len(grouped)))
        flips[start : start + n] = (signs * sums).sum(axis=1) / counts.sum()
    estimate = float(sums.sum() / counts.sum())
    return {
        "challenger": challenger,
        "reference": reference,
        "delta_mae": estimate,
        "ci95_lower": float(np.quantile(draws, 0.025)),
        "ci95_upper": float(np.quantile(draws, 0.975)),
        "sign_flip_p_value": float((1 + np.sum(np.abs(flips) >= abs(estimate))) / (replicates + 1)),
        "replicates": replicates,
    }


def _bh(rows: list[dict[str, Any]]) -> None:
    order = np.argsort([row["sign_flip_p_value"] for row in rows])
    adjusted = np.ones(len(rows), dtype=float)
    running = 1.0
    for position in range(len(rows) - 1, -1, -1):
        index = int(order[position])
        value = min(1.0, rows[index]["sign_flip_p_value"] * len(rows) / (position + 1))
        running = min(running, value)
        adjusted[index] = running
    for row, q in zip(rows, adjusted, strict=True):
        row["bh_q_value"] = float(q)


def _complete_prediction_columns(frame: pd.DataFrame) -> list[str]:
    """Return only prediction columns defined and finite for the full OOF union.

    Inner selection can promote a small number of fold-specific candidates.  Their
    columns are retained in the per-fold artifacts for auditability, but concatenating
    the five outer folds necessarily leaves missing values outside the fold where a
    candidate was promoted.  Aggregate comparisons must therefore be limited to the
    prespecified predictions available for every held-out structure.
    """
    return [
        name
        for name in frame
        if name.startswith("pred__")
        and frame[name].notna().all()
        and np.isfinite(frame[name].to_numpy(float)).all()
    ]


def _aggregate(
    repo: Path,
    v8: Path,
    output: Path,
    outer_units: list[dict[str, Any]],
    mmp_units: list[dict[str, Any]],
    inner_units: list[dict[str, Any]],
    candidates: list[Candidate],
    surfaces: dict[str, list[str]],
    workers: int,
    bootstrap_replicates: int,
) -> dict[str, Any]:
    oof = pd.concat(
        [pd.read_parquet(_artifact(unit, "outer_predictions")) for unit in outer_units], ignore_index=True
    )
    if len(oof) != EXACT_ROWS or oof.structure_id.duplicated().any() or set(oof.outer_fold) != set(OUTERS):
        raise CampaignError("nested OOF union is not exactly 18,801 once")
    v8_oof = pd.read_parquet(v8 / "nested_oof_predictions.parquet")[
        ["structure_id", "predicted_pic50"]
    ].rename(columns={"predicted_pic50": "pred__v8"})
    oof = oof.merge(v8_oof, on="structure_id", validate="one_to_one")
    mmp_assisted = pd.concat(
        [
            pd.read_parquet(_artifact(unit, "mmp_analog_assisted_predictions"))[
                [
                    "structure_id",
                    "predicted_pic50_mmp_assisted",
                    "mmp_analog_covered",
                    "mmp_training_analog_count",
                    "mmp_cross_pair_count",
                    "mmp_training_analog_ids",
                ]
            ].assign(outer_fold=unit["unit_spec"]["outer_fold"])
            for unit in mmp_units
        ],
        ignore_index=True,
    )
    if len(mmp_assisted) != EXACT_ROWS or mmp_assisted.structure_id.duplicated().any():
        raise CampaignError("MMP analog-assisted OOF union is not exactly 18,801 once")
    oof = oof.merge(mmp_assisted.drop(columns="outer_fold"), on="structure_id", validate="one_to_one")
    oof["pred__mmp_analog_assisted"] = oof.predicted_pic50_mmp_assisted
    model_columns = _complete_prediction_columns(oof)
    if not model_columns:
        raise CampaignError("no complete finite OOF prediction columns were produced")
    metrics_rows = [
        {"model_id": column.removeprefix("pred__"), **_metrics(oof.observed_pic50, oof[column])}
        for column in model_columns
    ]
    metrics = pd.DataFrame(metrics_rows).sort_values("mae")
    comparisons = [
        _bootstrap_delta(oof, column, "pred__v8", bootstrap_replicates, SEED + index)
        for index, column in enumerate(model_columns)
        if column != "pred__v8"
    ]
    _bh(comparisons)
    comparisons_frame = pd.DataFrame(comparisons)
    v9_metrics = metrics.loc[metrics.model_id.ne("v8")]
    if v9_metrics.empty:
        raise CampaignError("no V9 finalist predictions were produced")
    best = str(v9_metrics.iloc[0].model_id)
    diagnostics = oof.copy()
    diagnostic_context = pd.read_parquet(output / "prepared/training_matrix.parquet")[
        [
            "structure_id",
            "measurement_modality",
            "automation_class",
            "assay_family",
            "source_family",
        ]
    ]
    diagnostics = diagnostics.merge(diagnostic_context, on="structure_id", validate="one_to_one")
    diagnostics["finalist_model_id"] = best
    diagnostics["finalist_prediction"] = diagnostics[f"pred__{best}"]
    diagnostics["finalist_residual"] = diagnostics.observed_pic50 - diagnostics.finalist_prediction
    diagnostics["molecular_weight_regime"] = pd.cut(
        diagnostics.structure_id.map(
            pd.read_parquet(output / "prepared/training_matrix.parquet").set_index("structure_id")[
                "rdkit2d__MolWt"
            ]
        ),
        [-np.inf, 500, 700, 1000, np.inf],
        labels=["lt500", "500_700", "700_1000", "gt1000"],
    ).astype(str)
    cliff_ids: set[str] = set()
    mmp = pd.read_parquet(output / "prepared/training_mmp_effects.parquet")
    cliffs = mmp.loc[mmp.activity_cliff_ge_1_pic50]
    cliff_ids |= set(cliffs.structure_id_a.astype(str)) | set(cliffs.structure_id_b.astype(str))
    diagnostics["training_mmp_cliff_member"] = diagnostics.structure_id.astype(str).isin(cliff_ids)
    subgroup_rows: list[dict[str, Any]] = []
    for dimension in (
        "measurement_modality",
        "source_family",
        "molecular_weight_regime",
        "training_mmp_cliff_member",
        "extrapolation_or_abstention_flag",
    ):
        for value, part in diagnostics.groupby(dimension, observed=True, dropna=False):
            if len(part) >= 25:
                subgroup_rows.append(
                    {
                        "dimension": dimension,
                        "value": str(value),
                        "model_id": best,
                        **_metrics(part.observed_pic50, part.finalist_prediction),
                    }
                )
    mmp_predictions = pd.concat(
        [
            pd.read_parquet(_artifact(unit, "mmp_delta_predictions")).assign(
                outer_fold=unit["unit_spec"]["outer_fold"]
            )
            for unit in mmp_units
        ],
        ignore_index=True,
    )
    analysis = output / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    paths = {
        "nested_oof_predictions": analysis / "nested_oof_predictions.parquet",
        "individual_compound_diagnostic_atlas": analysis / "individual_compound_diagnostic_atlas.parquet",
        "model_metrics": analysis / "model_metrics.parquet",
        "paired_statistical_comparisons": analysis / "paired_statistical_comparisons.parquet",
        "subgroup_report": analysis / "subgroup_report.parquet",
        "mmp_delta_predictions": analysis / "mmp_delta_predictions.parquet",
        "mmp_analog_assisted_predictions": analysis / "mmp_analog_assisted_predictions.parquet",
    }
    _atomic_parquet(paths["nested_oof_predictions"], oof)
    _atomic_parquet(paths["individual_compound_diagnostic_atlas"], diagnostics)
    _atomic_parquet(paths["model_metrics"], metrics)
    _atomic_parquet(paths["paired_statistical_comparisons"], comparisons_frame)
    _atomic_parquet(paths["subgroup_report"], pd.DataFrame(subgroup_rows))
    _atomic_parquet(paths["mmp_delta_predictions"], mmp_predictions)
    _atomic_parquet(paths["mmp_analog_assisted_predictions"], mmp_assisted)
    # Freeze the deployable recipe from mean inner OOF rank, not repository validation/test outcomes.
    inner_rank: dict[str, list[float]] = {}
    for unit in inner_units:
        cid = unit["unit_spec"]["candidate"]["candidate_id"]
        inner_rank.setdefault(cid, []).append(float(unit["metrics"]["mae"]))
    chosen_id = min(inner_rank, key=lambda cid: float(np.mean(inner_rank[cid])))
    chosen = next(candidate for candidate in candidates if candidate.candidate_id == chosen_id)
    matrix, _splits, observations, mmp, _ad, _blocks = _load(output)
    all_ids = set(matrix.structure_id.astype(str))
    final_targets = _measurement_targets(observations, all_ids, chosen.target_mode)
    final_training = matrix.merge(final_targets, on="structure_id", validate="one_to_one")
    final_weights = np.ones(len(final_training), dtype=float)
    final_y = final_training.training_target.to_numpy(float)
    if chosen.weight_mode == "reliability":
        final_weights *= final_training.reliability_weight.to_numpy(float)
    elif chosen.weight_mode == "potency_tail":
        final_weights *= 1.0 + 1.25 * ((final_y < 4.5) | (final_y >= 6.5))
    elif chosen.weight_mode == "heavy_flexible":
        final_weights *= 1.0 + 0.75 * (
            final_training.rdkit2d__MolWt.ge(500) | final_training.rdkit2d__NumRotatableBonds.ge(8)
        ).to_numpy(float)
    elif chosen.weight_mode == "cliff_risk":
        cliff_rows = mmp.loc[mmp.activity_cliff_ge_1_pic50]
        cliff_ids = set(cliff_rows.structure_id_a.astype(str)) | set(cliff_rows.structure_id_b.astype(str))
        final_weights *= 1.0 + final_training.structure_id.astype(str).isin(cliff_ids).to_numpy(float)
    final_model = _new_model(chosen, workers, SEED + 9999)
    final_fit_kwargs = (
        {"model__sample_weight": final_weights}
        if isinstance(final_model, Pipeline)
        else {"sample_weight": final_weights}
    )
    final_model.fit(_clean_matrix(final_training, surfaces[chosen.surface]), final_y, **final_fit_kwargs)
    model_root = output / "final_model"
    model_root.mkdir(parents=True, exist_ok=True)
    model_path = model_root / "molecular_model.joblib"
    joblib.dump(
        {
            "schema_version": SCHEMA_VERSION,
            "candidate": asdict(chosen),
            "feature_columns": surfaces[chosen.surface],
            "model": final_model,
            "training_partition": "train",
            "validation_test_labels_opened": False,
        },
        model_path,
        compress=3,
    )
    schema_path = model_root / "feature_preprocessing_schema.json"
    _atomic_json(
        schema_path,
        {
            "schema_version": SCHEMA_VERSION,
            "candidate": asdict(chosen),
            "feature_columns": surfaces[chosen.surface],
            "nonfinite_or_abs_gt_1e30": "missing",
            "target_scope": "wild-type-or-unspecified exact pIC50",
            "validation_test_labels_opened": False,
        },
        "schema_sha256",
    )
    smoke = matrix.iloc[:12].copy()
    bundle = joblib.load(model_path)
    smoke_pred = bundle["model"].predict(_clean_matrix(smoke, bundle["feature_columns"]))
    smoke_path = model_root / "inference_smoke.parquet"
    _atomic_parquet(
        smoke_path,
        pd.DataFrame(
            {
                "structure_id": smoke.structure_id,
                "predicted_pic50": smoke_pred,
                "finite_prediction": np.isfinite(smoke_pred),
            }
        ),
    )
    model_card_path = model_root / "MODEL_CARD.md"
    best_row = v9_metrics.iloc[0]
    model_card_path.write_text(
        "# hERG V9 molecular model\n\n"
        f"Frozen train-only recipe: `{chosen_id}`. Internal nested scaffold OOF best model: `{best}`.\n\n"
        f"Internal MAE: {best_row.mae:.6f}; RMSE: {best_row.rmse:.6f}.\n\n"
        "The quantitative target is wild-type-or-unspecified hERG pIC50, not confirmed WT. "
        "Repository validation and test labels remained sealed. This is not external/prospective "
        "validation, clinical utility, superiority, or biological causality evidence. MMP analog "
        "assistance is reported separately and is not required by the molecular model.\n",
        encoding="utf-8",
    )
    report_path = output / "REPORT.md"
    improvement = comparisons_frame.loc[comparisons_frame.challenger.eq(f"pred__{best}")].iloc[0]
    report_path.write_text(
        "# hERG V9 domain-mixture campaign\n\n"
        f"The primary internal nested scaffold result used 18,801 train structures. Best MAE was "
        f"{best_row.mae:.6f} for {best}; V8 was {V8_MAE:.6f}. The paired delta was "
        f"{improvement.delta_mae:+.6f} with 95% CI [{improvement.ci95_lower:+.6f}, "
        f"{improvement.ci95_upper:+.6f}].\n\n"
        "The campaign tests domain specialization, measurement handling, selective ligand physics, "
        "MMP delta prediction, honest stacking, gating, uncertainty, and abstention. Associations and "
        "feature importances are not biological causal evidence. Validation and test outcomes were not "
        "opened. See the Parquet artifacts for all models, subgroups, molecules, and MMP pairs.\n",
        encoding="utf-8",
    )
    decision_path = output / "decision_ledger.json"
    _atomic_json(
        decision_path,
        {
            "schema_version": SCHEMA_VERSION,
            "best_nested_model": best,
            "frozen_deployable_candidate": chosen_id,
            "v8_reference_mae": V8_MAE,
            "improvement_claim_supported": bool(improvement.ci95_upper < 0),
            "molecular_only_deployment": True,
            "analog_assistance_separate": True,
            "biological_causality_claimed": False,
        },
        "decision_sha256",
    )
    artifacts = [_binding(path, role) for role, path in paths.items()]
    artifacts += [
        _binding(model_path, "deployable_molecular_model"),
        _binding(schema_path, "feature_preprocessing_schema"),
        _binding(smoke_path, "inference_smoke"),
        _binding(model_card_path, "model_card"),
        _binding(report_path, "report"),
        _binding(decision_path, "decision_ledger"),
    ]
    all_units = [*inner_units, *outer_units, *mmp_units]
    manifest_path = output / "manifest.json"
    manifest = _atomic_json(
        manifest_path,
        {
            "schema_version": SCHEMA_VERSION,
            "status": "passed",
            "created_utc": _utc(),
            "source_partition": "train",
            "exact_nested_oof_rows": EXACT_ROWS,
            "validation_labels_opened": False,
            "test_labels_opened": False,
            "external_or_prospective_validation": False,
            "unit_documents": [
                _binding(output / "units" / unit["unit_id"] / "unit.json", f"unit_{unit['unit_id']}")
                for unit in all_units
            ],
            "artifacts": artifacts,
        },
        "manifest_sha256",
    )
    validation = _atomic_json(
        output / "validation.json",
        {
            "schema_version": SCHEMA_VERSION,
            "status": "passed",
            "manifest_sha256": manifest["manifest_sha256"],
            "exact_nested_oof_rows": EXACT_ROWS,
            "outer_folds": 5,
            "validation_labels_opened": False,
            "test_labels_opened": False,
            "all_predictions_finite": bool(np.isfinite(oof[model_columns].to_numpy(float)).all()),
            "deployable_model_smoke_passed": bool(np.isfinite(smoke_pred).all()),
        },
        "validation_sha256",
    )
    return validation


def _resource_gate(output: Path) -> None:
    free = shutil.disk_usage(output.parent if output.parent.exists() else Path.cwd()).free / 2**30
    if free < 20:
        raise CampaignError(f"need at least 20 GiB free; found {free:.1f}")


def _validate(output: Path) -> dict[str, Any]:
    prepared = _read_json(output / "prepared/validation.json", "validation_sha256")
    for row in prepared["inputs"]:
        _verify_binding(row)
    for row in prepared["artifacts"]:
        _verify_binding(row, output)
    validation = _read_json(output / "validation.json", "validation_sha256")
    manifest = _read_json(output / "manifest.json", "manifest_sha256")
    if validation.get("status") != "passed" or manifest.get("status") != "passed":
        raise CampaignError("campaign did not pass")
    for row in [*manifest["unit_documents"], *manifest["artifacts"]]:
        _verify_binding(row, output)
    oof = pd.read_parquet(output / "analysis/nested_oof_predictions.parquet")
    if len(oof) != EXACT_ROWS or oof.structure_id.duplicated().any() or set(oof.outer_fold) != set(OUTERS):
        raise CampaignError("nested OOF validation failed")
    if validation.get("validation_labels_opened") or validation.get("test_labels_opened"):
        raise CampaignError("sealed-label contract failed")
    return validation


def _recover_finalizer(output: Path, confirmation: str) -> dict[str, Any]:
    """Govern the one-time implementation-binding migration for the V9 finalizer fix."""
    if confirmation != FINALIZER_RECOVERY_CONFIRMATION:
        raise CampaignError("incorrect finalizer recovery confirmation")
    with _Lock(output / ".campaign.lock"):
        prepared_path = output / "prepared/validation.json"
        prepared = _read_json(prepared_path, "validation_sha256")
        implementation = Path(__file__).resolve()
        current_binding = _binding(implementation, "implementation")
        implementation_rows = [row for row in prepared["inputs"] if row["role"] == "implementation"]
        if len(implementation_rows) != 1:
            raise CampaignError("prepared validation lacks one implementation binding")
        old_binding = implementation_rows[0]
        if old_binding == current_binding:
            return {"status": "already_applied", "implementation_sha256": current_binding["sha256"]}
        if (
            int(old_binding["bytes"]) != ORIGINAL_IMPLEMENTATION_BYTES
            or old_binding["sha256"] != ORIGINAL_IMPLEMENTATION_SHA256
        ):
            raise CampaignError("prepared validation does not match the frozen pre-fix implementation")
        for row in prepared["inputs"]:
            if row["role"] != "implementation":
                _verify_binding(row)
        for row in prepared["artifacts"]:
            _verify_binding(row, output)
        checkpoint = _read_json(output / "checkpoint.json", "checkpoint_sha256")
        expected_counts = (110, 5, 5)
        actual_counts = (
            int(checkpoint.get("completed_inner_units", -1)),
            int(checkpoint.get("completed_outer_units", -1)),
            int(checkpoint.get("completed_mmp_units", -1)),
        )
        if actual_counts != expected_counts:
            raise CampaignError(f"unexpected V9 checkpoint counts: {actual_counts}")
        unit_paths = sorted((output / "units").glob("*/unit.json"))
        if len(unit_paths) != sum(expected_counts):
            raise CampaignError(f"expected 120 completed unit documents; found {len(unit_paths)}")
        artifact_count = 0
        for path in unit_paths:
            unit = _read_json(path, "unit_json_sha256")
            if unit.get("status") != "passed":
                raise CampaignError(f"unit did not pass: {path}")
            for row in unit["artifacts"]:
                _verify_binding(row, output)
                artifact_count += 1
        if artifact_count != 130:
            raise CampaignError(f"expected 130 bound unit artifacts; found {artifact_count}")
        recovery_root = output / "recovery"
        recovery_root.mkdir(parents=True, exist_ok=True)
        snapshot_path = recovery_root / "prepared_validation_before_finalizer_fix.json"
        if not snapshot_path.exists():
            snapshot_path.write_bytes(prepared_path.read_bytes())
        snapshot_binding = _binding(snapshot_path, "pre_recovery_prepared_validation")
        if snapshot_binding["sha256"] != _sha(prepared_path):
            raise CampaignError("pre-recovery snapshot differs from current prepared validation")
        recovery_path = recovery_root / "finalizer_recovery.json"
        recovery = _atomic_json(
            recovery_path,
            {
                "schema_version": SCHEMA_VERSION,
                "status": "passed",
                "created_utc": _utc(),
                "confirmation": confirmation,
                "reason": "final aggregation now excludes fold-specific incomplete prediction columns",
                "old_implementation": old_binding,
                "new_implementation": current_binding,
                "completed_unit_documents_verified": len(unit_paths),
                "completed_unit_artifacts_verified": artifact_count,
                "checkpoint_counts": {
                    "inner": actual_counts[0],
                    "outer": actual_counts[1],
                    "mmp": actual_counts[2],
                },
                "snapshot": snapshot_binding,
            },
            "recovery_sha256",
        )
        prepared["inputs"] = [row for row in prepared["inputs"] if row["role"] != "implementation"] + [
            current_binding,
            _binding(recovery_path, "finalizer_recovery"),
        ]
        prepared["implementation_migration"] = {
            "status": "passed",
            "recovery_sha256": recovery["recovery_sha256"],
        }
        migrated = _atomic_json(prepared_path, prepared, "validation_sha256")
        return {
            "status": "recovered_ready_to_finalize",
            "prepared_validation_sha256": migrated["validation_sha256"],
            "implementation_sha256": current_binding["sha256"],
            "unit_documents_verified": len(unit_paths),
            "unit_artifacts_verified": artifact_count,
        }


def _run(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.repo_root.resolve()
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with _Lock(output / ".campaign.lock"):
        if (output / "validation.json").is_file():
            return _validate(output)
        _resource_gate(output)
        _prepare(repo, args.v8_root.resolve(), args.v81_root.resolve(), args.mmp_root.resolve(), output)
        matrix, splits, observations, mmp, ad, blocks = _load(output)
        surfaces = _surface_columns(blocks)
        candidates = _candidate_plan(args.smoke)
        inner_units: list[dict[str, Any]] = []
        outer_units: list[dict[str, Any]] = []
        mmp_units: list[dict[str, Any]] = []
        checkpoint_path = output / "checkpoint.json"
        stop = False

        def request_stop(_signum: int, _frame: Any) -> None:
            nonlocal stop
            stop = True

        signal.signal(signal.SIGINT, request_stop)
        signal.signal(signal.SIGTERM, request_stop)
        for outer in OUTERS[:1] if args.smoke else OUTERS:
            current: list[dict[str, Any]] = []
            for candidate in candidates:
                unit = _inner_unit(
                    output, matrix, splits, observations, mmp, surfaces, candidate, outer, args.workers
                )
                current.append(unit)
                inner_units.append(unit)
                _atomic_json(
                    checkpoint_path,
                    {
                        "schema_version": SCHEMA_VERSION,
                        "status": "running",
                        "completed_inner_units": len(inner_units),
                        "completed_outer_units": len(outer_units),
                        "completed_mmp_units": len(mmp_units),
                        "last_completed_unit": unit["unit_id"],
                    },
                    "checkpoint_sha256",
                )
                if stop:
                    return {
                        "status": "interrupted_at_safe_unit_boundary",
                        "resume": "rerun identical command",
                    }
            outer_unit = _outer_unit(
                output,
                matrix,
                splits,
                observations,
                mmp,
                ad,
                surfaces,
                candidates,
                outer,
                args.workers,
                current,
            )
            outer_units.append(outer_unit)
            mmp_units.append(_mmp_unit(output, matrix, splits, mmp, outer, args.workers, outer_unit))
            _atomic_json(
                checkpoint_path,
                {
                    "schema_version": SCHEMA_VERSION,
                    "status": "running",
                    "completed_inner_units": len(inner_units),
                    "completed_outer_units": len(outer_units),
                    "completed_mmp_units": len(mmp_units),
                    "last_completed_unit": mmp_units[-1]["unit_id"],
                },
                "checkpoint_sha256",
            )
            if stop:
                return {"status": "interrupted_at_safe_unit_boundary", "resume": "rerun identical command"}
        if args.smoke:
            return {
                "status": "smoke_passed",
                "inner_units": len(inner_units),
                "outer_units": len(outer_units),
                "mmp_units": len(mmp_units),
            }
        result = _aggregate(
            repo,
            args.v8_root.resolve(),
            output,
            outer_units,
            mmp_units,
            inner_units,
            candidates,
            surfaces,
            args.workers,
            args.bootstrap_replicates,
        )
        _atomic_json(
            checkpoint_path,
            {
                "schema_version": SCHEMA_VERSION,
                "status": "complete",
                "completed_inner_units": len(inner_units),
                "completed_outer_units": len(outer_units),
                "completed_mmp_units": len(mmp_units),
            },
            "checkpoint_sha256",
        )
        return result


def _predict(args: argparse.Namespace) -> dict[str, Any]:
    bundle = joblib.load(args.model_root.resolve() / "molecular_model.joblib")
    frame = pd.read_parquet(args.input.resolve())
    missing = set(bundle["feature_columns"]) - set(frame)
    if missing:
        raise CampaignError(f"input lacks {len(missing)} required features")
    prediction = bundle["model"].predict(_clean_matrix(frame, bundle["feature_columns"]))
    result = pd.DataFrame(
        {
            "structure_id": frame.structure_id.astype(str),
            "predicted_pic50": prediction,
            "finite_prediction": np.isfinite(prediction),
        }
    )
    _atomic_parquet(args.output.resolve(), result)
    return {"status": "passed", "rows": len(result), "output": str(args.output.resolve())}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--repo-root", type=Path, required=True)
    run.add_argument("--v8-root", type=Path, required=True)
    run.add_argument("--v81-root", type=Path, required=True)
    run.add_argument("--mmp-root", type=Path, required=True)
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--workers", type=int, default=6, choices=range(1, 7))
    run.add_argument("--bootstrap-replicates", type=int, default=10_000)
    run.add_argument("--smoke", action="store_true")
    validate = sub.add_parser("validate")
    validate.add_argument("--output-root", type=Path, required=True)
    status = sub.add_parser("status")
    status.add_argument("--output-root", type=Path, required=True)
    predict = sub.add_parser("predict")
    predict.add_argument("--model-root", type=Path, required=True)
    predict.add_argument("--input", type=Path, required=True)
    predict.add_argument("--output", type=Path, required=True)
    recover = sub.add_parser("recover-finalizer")
    recover.add_argument("--output-root", type=Path, required=True)
    recover.add_argument("--confirm", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        if args.command == "run":
            result = _run(args)
        elif args.command == "validate":
            result = _validate(args.output_root.resolve())
        elif args.command == "status":
            root = args.output_root.resolve()
            if (root / "validation.json").is_file():
                result = _validate(root)
            elif (root / "checkpoint.json").is_file():
                result = _read_json(root / "checkpoint.json", "checkpoint_sha256")
            else:
                result = {"status": "not_started"}
        elif args.command == "recover-finalizer":
            result = _recover_finalizer(args.output_root.resolve(), args.confirm)
        else:
            result = _predict(args)
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except (CampaignError, ValueError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
