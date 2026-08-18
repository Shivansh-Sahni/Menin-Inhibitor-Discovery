#!/usr/bin/env python3
"""Run the train-only hERG feature/model relationship campaign V5.

The campaign joins the immutable 18,801-structure exact-pIC50 train surface to
the completed 24-conformer cache, uses the already-fixed five outer/three inner
scaffold folds, performs material XGBoost/LightGBM model selection, and writes
one prediction for every structure from a model that did not train on its
scaffold.  Group ablations and conditional held-out permutations are saved as
row-level paired effects so later inference can resample scaffolds rather than
pretend molecules are independent.

Repository validation/test labels are never loaded.  Conformer energies are
physics-QC covariates only: nonfinite or implausibly extreme energies are
masked, absolute energy is excluded, and explicit convergence/pathology
indicators are retained.
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
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from lightgbm import LGBMRegressor
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

SCHEMA_VERSION = "platform-local-herg-feature-relationship-campaign-v5/1.0"
DEFAULT_FEATURE_ROOT = Path("research/local_runs/herg_quantitative_24conformer_v4")
DEFAULT_BASE_ROOT = Path("research/local_runs/herg_discovery_campaign_v1")
DEFAULT_OUTPUT = Path("research/local_runs/herg_feature_relationship_campaign_v5")
EXACT_ROWS = 18_801
OUTER_FOLDS = tuple(range(5))
INNER_FOLDS = tuple(range(3))
SEED = 20260814
MAX_ABSOLUTE_NUMERIC = 1.0e30
MAX_ABSOLUTE_ENERGY_KCAL_MOL = 100_000.0
MAX_ENERGY_RANGE_KCAL_MOL = 10_000.0
PERMUTATION_REPEATS = 20
GOVERNANCE_COLUMNS = {
    "structure_id",
    "standardized_smiles",
    "standard_inchi_key",
    "scaffold_group_id",
    "target_pic50",
    "exact_observation_count",
    "measurement_modality",
    "automation_class",
    "assay_family",
    "source_family",
    "protocol_completeness_mean",
    "wild_type_evidence_scope",
    "master_confirmed_wild_type_scope",
    "feature_order",
    "f2d_feature_id",
    "descriptor_missing_count",
    "feature_error",
    "morgan_r2_2048_raw",
    "maccs_167_raw",
}


class CampaignError(RuntimeError):
    """Raised when a scientific or reproducibility contract is violated."""


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    engine: str
    surface: str
    params: dict[str, Any]

    def payload(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "engine": self.engine,
            "surface": self.surface,
            "params": self.params,
        }


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical(value: Any, *, newline: bool = True) -> bytes:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return data + (b"\n" if newline else b"")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha256(path: Path) -> str:
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
        raise CampaignError(f"missing artifact for {role}: {path}")
    record: dict[str, Any] = {
        "role": role,
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }
    if path.suffix == ".parquet":
        record.update(rows=pq.read_metadata(path).num_rows, arrow_schema_sha256=_schema_sha(path))
    return record


def _verify_binding(binding: dict[str, Any], root: Path | None = None) -> None:
    path = Path(str(binding["path"])).resolve()
    if root is not None and not path.is_relative_to(root.resolve()):
        raise CampaignError(f"artifact escapes campaign root: {path}")
    if not path.is_file() or path.stat().st_size != int(binding["bytes"]):
        raise CampaignError(f"artifact size/path changed: {path}")
    if _sha256(path) != binding["sha256"]:
        raise CampaignError(f"artifact hash changed: {path}")
    if path.suffix == ".parquet":
        if pq.read_metadata(path).num_rows != int(binding["rows"]):
            raise CampaignError(f"artifact row count changed: {path}")
        if _schema_sha(path) != binding["arrow_schema_sha256"]:
            raise CampaignError(f"artifact schema changed: {path}")


def _self_hashed(value: dict[str, Any], key: str) -> dict[str, Any]:
    result = copy.deepcopy(value)
    result.pop(key, None)
    result[key] = _digest(result)
    return result


def _atomic_json(path: Path, value: dict[str, Any], key: str) -> dict[str, Any]:
    document = _self_hashed(value, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(json.dumps(document, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n")
    os.replace(temporary, path)
    return document


def _read_json(path: Path, key: str | None = None) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CampaignError(f"expected JSON object: {path}")
    if key and value.get(key) != _self_hashed(value, key)[key]:
        raise CampaignError(f"self hash mismatch: {path}")
    return value


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_parquet(temporary, index=False)
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
            raise CampaignError(f"campaign lock held: {self.path}") from error
        return self

    def __exit__(self, *_: object) -> None:
        fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        self.handle.close()


def _feature_family(column: str) -> str | None:
    if column.startswith("rdkit2d__"):
        return "rdkit2d"
    if column.startswith("morgan__"):
        return "morgan"
    if column.startswith("interaction__") or column.startswith("v5interaction__"):
        return "selected_interactions"
    if column.startswith("f3d__"):
        return "old3d_stable"
    if not column.startswith("new3d__"):
        return None
    if "dominant_autocorr3d" in column:
        return "autocorr3d"
    if "dominant_whim" in column:
        return "whim"
    energy_terms = (
        "rotatable_bond",
        "conformer_count",
        "unconverged",
        "energy_range",
        "effective_conformer",
        "dominant_conformer_weight",
        "pairwise_rmsd",
        "energy_polar",
        "energy_extreme",
        "energy_nonfinite",
        "convergence_fraction",
        "all_retained_unconverged",
        "feature_failed",
    )
    polarity_terms = (
        "formal_charge",
        "basic_site",
        "acidic_site",
        "tautomer",
        "polar_radial",
        "internal_polar",
        "gasteiger_dipole",
        "absolute_charge_radius",
        "sasa",
    )
    shape_terms = (
        "pmi",
        "npr",
        "asphericity",
        "eccentricity",
        "inertial_shape",
        "radius_of_gyration",
        "spherocity",
        "pbf",
        "heavy_pair_distance",
        "heavy_contact_density",
    )
    if any(term in column for term in energy_terms):
        return "energy_flexibility"
    if any(term in column for term in polarity_terms):
        return "polarity_charge_internal_contacts"
    if any(term in column for term in shape_terms):
        return "shape"
    return "new3d_stable_misc"


def _candidate_plan() -> list[Candidate]:
    xgb_anchor = {
        "n_estimators": 900,
        "max_depth": 6,
        "learning_rate": 0.035,
        "min_child_weight": 3.0,
        "subsample": 0.85,
        "colsample_bytree": 0.75,
        "reg_alpha": 0.05,
        "reg_lambda": 2.0,
    }
    lgb_anchor = {
        "n_estimators": 1000,
        "num_leaves": 31,
        "max_depth": -1,
        "learning_rate": 0.03,
        "min_child_samples": 25,
        "subsample": 0.85,
        "colsample_bytree": 0.75,
        "reg_alpha": 0.05,
        "reg_lambda": 2.0,
    }
    return [
        Candidate("xgb_2d", "xgboost", "2d", xgb_anchor),
        Candidate("xgb_morgan", "xgboost", "morgan", xgb_anchor),
        Candidate("xgb_2d_morgan", "xgboost", "2d_morgan", xgb_anchor),
        Candidate("xgb_old3d", "xgboost", "old3d", xgb_anchor),
        Candidate("xgb_new3d", "xgboost", "new3d", xgb_anchor),
        Candidate(
            "xgb_full_memory_bounded",
            "xgboost",
            "full",
            {
                **xgb_anchor,
                "n_estimators": 400,
                "max_depth": 5,
                "min_child_weight": 5.0,
                "learning_rate": 0.035,
                "max_bin": 128,
            },
        ),
        Candidate("lgb_2d_morgan", "lightgbm", "2d_morgan", lgb_anchor),
        Candidate("lgb_old3d", "lightgbm", "old3d", lgb_anchor),
        Candidate("lgb_new3d", "lightgbm", "new3d", lgb_anchor),
        Candidate("lgb_fundamental", "lightgbm", "fundamental", lgb_anchor),
        Candidate(
            "lgb_full_memory_bounded",
            "lightgbm",
            "full",
            {
                **lgb_anchor,
                "n_estimators": 600,
                "num_leaves": 31,
                "min_child_samples": 35,
                "learning_rate": 0.03,
                "max_bin": 127,
            },
        ),
    ]


def _validate_candidate_plan(candidates: list[Candidate]) -> None:
    payloads = [_digest(candidate.payload()) for candidate in candidates]
    if len(payloads) != len(set(payloads)) or len(candidates) < 10:
        raise CampaignError("candidate plan is duplicated or too shallow")
    if {candidate.engine for candidate in candidates} != {"xgboost", "lightgbm"}:
        raise CampaignError("both XGBoost and LightGBM are required")
    required = {"2d", "morgan", "2d_morgan", "old3d", "new3d", "fundamental", "full"}
    if not required <= {candidate.surface for candidate in candidates}:
        raise CampaignError("candidate plan omits a required feature comparison")


def _load_new3d(feature_root: Path) -> pd.DataFrame:
    shards = sorted((feature_root / "features").glob("part-*.parquet"))
    if not shards:
        raise CampaignError("24-conformer feature shards are missing")
    frames = [pd.read_parquet(path) for path in shards]
    frame = pd.concat(frames, ignore_index=True, sort=False)
    if len(frame) != 24_901 or frame["structure_id"].duplicated().any():
        raise CampaignError("24-conformer feature identity contract failed")
    rename = {
        column: f"new3d__{column}"
        for column in frame.columns
        if column not in {"structure_id", "feature_order", "f2d_feature_id"}
    }
    return frame.rename(columns=rename)


def _qc_new3d(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = frame.copy()
    energy_min = pd.to_numeric(result["new3d__energy_min_kcal_mol"], errors="coerce")
    energy_range = pd.to_numeric(result["new3d__energy_range_kcal_mol"], errors="coerce")
    energy_nonfinite = ~np.isfinite(energy_min) | ~np.isfinite(energy_range)
    energy_extreme = (~energy_nonfinite) & (
        energy_min.abs().gt(MAX_ABSOLUTE_ENERGY_KCAL_MOL) | energy_range.abs().gt(MAX_ENERGY_RANGE_KCAL_MOL)
    )
    invalid_energy = energy_nonfinite | energy_extreme
    result["new3d__energy_nonfinite_indicator"] = energy_nonfinite.astype(np.int8)
    result["new3d__energy_extreme_indicator"] = energy_extreme.astype(np.int8)
    result.loc[invalid_energy, "new3d__energy_range_kcal_mol"] = np.nan
    # Absolute force-field energy has no cross-molecule physical reference and
    # is deliberately excluded from every model, even when finite.
    result["new3d__energy_min_kcal_mol"] = np.nan
    retained = pd.to_numeric(result["new3d__retained_conformer_count"], errors="coerce")
    unconverged = pd.to_numeric(result["new3d__unconverged_retained_count"], errors="coerce")
    result["new3d__convergence_fraction"] = np.where(
        retained.gt(0), np.maximum(0.0, 1.0 - unconverged / retained), np.nan
    )
    result["new3d__all_retained_unconverged_indicator"] = (retained.gt(0) & unconverged.ge(retained)).astype(
        np.int8
    )
    result["new3d__feature_failed_indicator"] = (result["new3d__feature_status"].astype(str) != "ok").astype(
        np.int8
    )
    qc = pd.DataFrame(
        [
            {"qc_measure": "structures", "count": len(result)},
            {"qc_measure": "feature_failed", "count": int(result["new3d__feature_failed_indicator"].sum())},
            {"qc_measure": "energy_nonfinite", "count": int(energy_nonfinite.sum())},
            {"qc_measure": "energy_extreme", "count": int(energy_extreme.sum())},
            {
                "qc_measure": "all_retained_unconverged",
                "count": int(result["new3d__all_retained_unconverged_indicator"].sum()),
            },
            {"qc_measure": "absolute_energy_feature_excluded", "count": len(result)},
        ]
    )
    return result, qc


def _add_interactions(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    terms = {
        "logp_x_polar_exposure": (
            "rdkit2d__MolLogP",
            "new3d__ensemble_polar_radial_exposure__mean",
        ),
        "logp_x_internal_polar_contacts": (
            "rdkit2d__MolLogP",
            "new3d__ensemble_internal_polar_contact_count__mean",
        ),
        "abs_charge_x_charge_radius": (
            "new3d__formal_charge",
            "new3d__ensemble_absolute_charge_radius_A__mean",
        ),
        "rotors_x_effective_conformers": (
            "new3d__rotatable_bond_count",
            "new3d__effective_conformer_count",
        ),
        "energy_range_x_polar_exposure": (
            "new3d__energy_range_kcal_mol",
            "new3d__ensemble_polar_radial_exposure__mean",
        ),
        "shape_x_logp": (
            "new3d__ensemble_radius_of_gyration__mean",
            "rdkit2d__MolLogP",
        ),
    }
    for name, (left, right) in terms.items():
        if left not in result or right not in result:
            raise CampaignError(f"interaction input missing: {left}, {right}")
        a = pd.to_numeric(result[left], errors="coerce")
        if name.startswith("abs_charge"):
            a = a.abs()
        result[f"v5interaction__{name}"] = a * pd.to_numeric(result[right], errors="coerce")
    return result


def _prepare(repo: Path, feature_root: Path, base_root: Path, output: Path) -> dict[str, Any]:
    prepared = output / "prepared"
    validation_path = prepared / "validation.json"
    if validation_path.is_file():
        return _read_json(validation_path, "validation_sha256")
    feature_validation = json.loads((feature_root / "validation.json").read_text())
    if feature_validation.get("status") != "passed":
        raise CampaignError("24-conformer source validation did not pass")
    exact_path = base_root / "prepared/exact_train_cache.parquet"
    split_path = base_root / "prepared/nested_scaffold_splits.parquet"
    exact = pd.read_parquet(exact_path)
    if len(exact) != EXACT_ROWS or exact.structure_id.duplicated().any():
        raise CampaignError("exact train surface is not 18,801 unique structures")
    new3d, qc = _qc_new3d(_load_new3d(feature_root))
    frame = exact.merge(new3d, on="structure_id", how="left", validate="one_to_one")
    if len(frame) != EXACT_ROWS:
        raise CampaignError("exact/new-3D join lost identities")
    physics_excluded = (
        frame["new3d__energy_nonfinite_indicator"].fillna(1).astype(bool)
        | frame["new3d__energy_extreme_indicator"].fillna(1).astype(bool)
        | frame["new3d__all_retained_unconverged_indicator"].fillna(1).astype(bool)
        | frame["new3d__feature_failed_indicator"].fillna(1).astype(bool)
    )
    indicator_columns = {
        "new3d__energy_nonfinite_indicator",
        "new3d__energy_extreme_indicator",
        "new3d__all_retained_unconverged_indicator",
        "new3d__feature_failed_indicator",
    }
    physical_columns = [
        column
        for column in frame.columns
        if column.startswith("new3d__")
        and column not in indicator_columns
        and pd.api.types.is_numeric_dtype(frame[column])
        and not pd.api.types.is_bool_dtype(frame[column])
    ]
    for column in physical_columns:
        values = pd.to_numeric(frame[column], errors="coerce").astype(float)
        values.loc[physics_excluded] = np.nan
        frame[column] = values
    frame["new3d__physics_qc_excluded_indicator"] = physics_excluded.astype(np.int8)
    qc = pd.concat(
        [
            qc,
            pd.DataFrame(
                [{"qc_measure": "exact_join_physics_qc_excluded", "count": int(physics_excluded.sum())}]
            ),
        ],
        ignore_index=True,
    )
    old_min = pd.to_numeric(frame["f3d__energy_min_kcal_mol"], errors="coerce")
    old_range = pd.to_numeric(frame["f3d__energy_range_kcal_mol"], errors="coerce")
    old_bad = (
        (~np.isfinite(old_min)) | (~np.isfinite(old_range)) | old_range.abs().gt(MAX_ENERGY_RANGE_KCAL_MOL)
    )
    frame["f3d__energy_min_kcal_mol"] = np.nan
    frame.loc[old_bad, "f3d__energy_range_kcal_mol"] = np.nan
    frame["f3d__energy_pathology_indicator"] = old_bad.astype(np.int8)
    frame = _add_interactions(frame)
    families: dict[str, list[str]] = {}
    sanitization: list[dict[str, Any]] = []
    for column in frame.columns:
        family = _feature_family(column)
        if family is None or column == "new3d__energy_min_kcal_mol":
            continue
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64, copy=True)
        invalid = ~np.isfinite(values) | (np.abs(values) > MAX_ABSOLUTE_NUMERIC)
        values[invalid] = np.nan
        finite = values[np.isfinite(values)]
        if finite.size < 100 or float(np.nanstd(finite)) <= 1e-12:
            continue
        frame[column] = values.astype(np.float32)
        families.setdefault(family, []).append(column)
        sanitization.append(
            {
                "feature": column,
                "family": family,
                "nonfinite_or_overflow_to_missing": int(invalid.sum()),
                "finite_count": int(finite.size),
            }
        )
    required_families = {
        "rdkit2d",
        "morgan",
        "old3d_stable",
        "energy_flexibility",
        "polarity_charge_internal_contacts",
        "shape",
        "autocorr3d",
        "whim",
        "selected_interactions",
    }
    if not required_families <= set(families):
        raise CampaignError(f"required feature families missing: {sorted(required_families - set(families))}")
    keep = [
        "structure_id",
        "scaffold_group_id",
        "target_pic50",
        "measurement_modality",
        "automation_class",
        "assay_family",
        "source_family",
        *sorted({column for values in families.values() for column in values}),
    ]
    matrix_path = prepared / "training_matrix.parquet"
    split_copy = prepared / "fixed_nested_scaffold_splits.parquet"
    qc_path = prepared / "conformer_qc.parquet"
    sanitize_path = prepared / "sanitization_report.parquet"
    schema_path = prepared / "feature_schemas.json"
    _atomic_parquet(matrix_path, frame[keep])
    splits = pd.read_parquet(split_path)
    expected_ids = set(frame.structure_id.astype(str))
    if len(splits) != EXACT_ROWS * len(OUTER_FOLDS):
        raise CampaignError("fixed split registry row count is invalid")
    for outer in OUTER_FOLDS:
        part = splits.loc[splits.outer_fold.eq(outer)]
        if set(part.structure_id.astype(str)) != expected_ids or part.structure_id.duplicated().any():
            raise CampaignError(f"fixed split registry identity coverage failed for outer {outer}")
        fit_scaffolds = set(part.loc[part.outer_role.eq("fit"), "scaffold_group_id"].astype(str))
        held_scaffolds = set(part.loc[part.outer_role.eq("heldout"), "scaffold_group_id"].astype(str))
        if fit_scaffolds & held_scaffolds:
            raise CampaignError(f"fixed split scaffold leakage for outer {outer}")
    _atomic_parquet(split_copy, splits)
    _atomic_parquet(qc_path, qc)
    _atomic_parquet(sanitize_path, pd.DataFrame(sanitization))
    surfaces = _surfaces(families)
    schema = _atomic_json(
        schema_path,
        {
            "schema_version": SCHEMA_VERSION,
            "created_utc": _utc(),
            "families": {key: sorted(value) for key, value in families.items()},
            "surfaces": surfaces,
            "absolute_energy_excluded": True,
            "extreme_energy_thresholds": {
                "absolute_energy_kcal_mol": MAX_ABSOLUTE_ENERGY_KCAL_MOL,
                "energy_range_kcal_mol": MAX_ENERGY_RANGE_KCAL_MOL,
            },
        },
        "feature_schema_sha256",
    )
    return _atomic_json(
        validation_path,
        {
            "schema_version": SCHEMA_VERSION,
            "status": "passed",
            "created_utc": _utc(),
            "exact_train_rows": len(frame),
            "source_partition": "train",
            "repository_validation_labels_opened": False,
            "repository_test_labels_opened": False,
            "feature_schema_sha256_bound": schema["feature_schema_sha256"],
            "artifacts": [
                _binding(matrix_path, "training_matrix"),
                _binding(split_copy, "fixed_scaffold_splits"),
                _binding(qc_path, "conformer_qc"),
                _binding(sanitize_path, "sanitization_report"),
                _binding(schema_path, "feature_schemas"),
            ],
        },
        "validation_sha256",
    )


def _surfaces(families: dict[str, list[str]]) -> dict[str, list[str]]:
    def union(*names: str) -> list[str]:
        return sorted({column for name in names for column in families.get(name, [])})

    new_names = (
        "energy_flexibility",
        "polarity_charge_internal_contacts",
        "shape",
        "autocorr3d",
        "whim",
        "new3d_stable_misc",
        "selected_interactions",
    )
    all_names = tuple(families)
    return {
        "2d": union("rdkit2d"),
        "morgan": union("morgan"),
        "2d_morgan": union("rdkit2d", "morgan"),
        "old3d": union("rdkit2d", "morgan", "old3d_stable"),
        "new3d": union("rdkit2d", "morgan", *new_names),
        "fundamental": union(
            "rdkit2d",
            "morgan",
            "energy_flexibility",
            "polarity_charge_internal_contacts",
            "shape",
            "selected_interactions",
        ),
        "full": union(*all_names),
    }


def _load_prepared(output: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    validation = _read_json(output / "prepared/validation.json", "validation_sha256")
    for binding in validation["artifacts"]:
        _verify_binding(binding, output)
    frame = pd.read_parquet(output / "prepared/training_matrix.parquet")
    splits = pd.read_parquet(output / "prepared/fixed_nested_scaffold_splits.parquet")
    schema = _read_json(output / "prepared/feature_schemas.json", "feature_schema_sha256")
    return frame, splits, schema


def _model(candidate: Candidate, workers: int, seed: int) -> Any:
    if candidate.engine == "xgboost":
        return XGBRegressor(
            objective="reg:squarederror",
            tree_method="hist",
            n_jobs=workers,
            random_state=seed,
            verbosity=0,
            **candidate.params,
        )
    if candidate.engine == "lightgbm":
        return LGBMRegressor(
            objective="regression_l1",
            n_jobs=workers,
            random_state=seed,
            verbosity=-1,
            subsample_freq=1,
            **candidate.params,
        )
    raise CampaignError(f"unsupported engine: {candidate.engine}")


def _metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    absolute = np.abs(observed - predicted)
    return {
        "n": len(observed),
        "mae": float(np.mean(absolute)),
        "rmse": float(math.sqrt(mean_squared_error(observed, predicted))),
        "r2": float(r2_score(observed, predicted)),
        "pearson": float(pearsonr(observed, predicted).statistic),
        "spearman": float(spearmanr(observed, predicted).statistic),
        "fraction_within_0p5": float(np.mean(absolute <= 0.5)),
    }


def _fit_predict(
    frame: pd.DataFrame,
    columns: list[str],
    fit_ids: set[str],
    eval_ids: set[str],
    candidate: Candidate,
    workers: int,
    seed: int,
) -> tuple[Any, pd.DataFrame, float]:
    fit = frame.loc[frame.structure_id.astype(str).isin(fit_ids)]
    evaluation = frame.loc[frame.structure_id.astype(str).isin(eval_ids)]
    if set(fit.scaffold_group_id.astype(str)) & set(evaluation.scaffold_group_id.astype(str)):
        raise CampaignError("scaffold leakage in fit/evaluation")
    model = _model(candidate, workers, seed)
    started = time.monotonic()
    model.fit(fit[columns], fit.target_pic50.to_numpy(dtype=float))
    elapsed = time.monotonic() - started
    predicted = np.asarray(model.predict(evaluation[columns]), dtype=float)
    prediction = evaluation[
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
    prediction["predicted_pic50"] = predicted
    return model, prediction, elapsed


def _unit_document(
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
                "repository_validation_labels_opened": False,
                "repository_test_labels_opened": False,
            },
        },
        "unit_json_sha256",
    )


def _existing_unit(directory: Path, spec: dict[str, Any], output: Path) -> dict[str, Any] | None:
    path = directory / "unit.json"
    if not path.is_file():
        return None
    try:
        unit = _read_json(path, "unit_json_sha256")
        if unit.get("status") != "passed" or unit.get("unit_spec") != spec:
            return None
        for binding in unit.get("artifacts", []):
            _verify_binding(binding, output)
        return unit
    except Exception:
        return None


def _inner_unit(
    output: Path,
    frame: pd.DataFrame,
    splits: pd.DataFrame,
    surfaces: dict[str, list[str]],
    candidate: Candidate,
    outer: int,
    workers: int,
) -> dict[str, Any]:
    unit_id = f"inner_o{outer}_{candidate.candidate_id}"
    directory = output / "units" / unit_id
    spec = {"operation": "inner_hpo", "outer_fold": outer, "candidate": candidate.payload()}
    existing = _existing_unit(directory, spec, output)
    if existing is not None:
        return existing
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    assignments = splits.loc[(splits.outer_fold.eq(outer)) & (splits.outer_role.eq("fit"))]
    rows: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    for inner in INNER_FOLDS:
        fit_ids = set(assignments.loc[assignments.inner_fold.ne(inner), "structure_id"].astype(str))
        eval_ids = set(assignments.loc[assignments.inner_fold.eq(inner), "structure_id"].astype(str))
        _, prediction, elapsed = _fit_predict(
            frame,
            surfaces[candidate.surface],
            fit_ids,
            eval_ids,
            candidate,
            workers,
            SEED + outer * 100 + inner,
        )
        rows.append(
            {
                "outer_fold": outer,
                "inner_fold": inner,
                "candidate_id": candidate.candidate_id,
                "engine": candidate.engine,
                "feature_surface": candidate.surface,
                "feature_count": len(surfaces[candidate.surface]),
                "fit_seconds": elapsed,
                **_metrics(prediction.target_pic50, prediction.predicted_pic50),
            }
        )
        prediction = prediction.rename(columns={"target_pic50": "observed_pic50"})
        prediction["outer_fold"] = outer
        prediction["inner_fold"] = inner
        prediction["candidate_id"] = candidate.candidate_id
        predictions.append(prediction)
    metrics_path = directory / "inner_metrics.parquet"
    prediction_path = directory / "inner_oof_predictions.parquet"
    _atomic_parquet(metrics_path, pd.DataFrame(rows))
    _atomic_parquet(prediction_path, pd.concat(predictions, ignore_index=True))
    mean_mae = float(np.mean([row["mae"] for row in rows]))
    return _unit_document(
        directory,
        unit_id,
        spec,
        {"selection_score": mean_mae, "mean_inner_mae": mean_mae, "folds": 3},
        [
            _binding(metrics_path, "inner_metrics"),
            _binding(prediction_path, "inner_oof_predictions"),
        ],
    )


def _importance(model: Any, columns: list[str], outer: int, candidate: Candidate, role: str) -> pd.DataFrame:
    values = np.asarray(getattr(model, "feature_importances_", np.zeros(len(columns))), dtype=float)
    if len(values) != len(columns):
        values = np.zeros(len(columns), dtype=float)
    return pd.DataFrame(
        {
            "outer_fold": outer,
            "candidate_id": candidate.candidate_id,
            "engine": candidate.engine,
            "model_role": role,
            "feature": columns,
            "family": [_feature_family(column) for column in columns],
            "importance": values,
        }
    )


def _conditional_strata(fit: pd.DataFrame, evaluation: pd.DataFrame) -> np.ndarray:
    columns = ["rdkit2d__MolWt", "rdkit2d__MolLogP", "rdkit2d__TPSA"]
    codes: list[np.ndarray] = []
    for column in columns:
        fit_values = pd.to_numeric(fit[column], errors="coerce").to_numpy(dtype=float)
        finite = fit_values[np.isfinite(fit_values)]
        edges = np.unique(np.quantile(finite, [0.25, 0.5, 0.75])) if finite.size else np.array([])
        values = pd.to_numeric(evaluation[column], errors="coerce").to_numpy(dtype=float)
        values[~np.isfinite(values)] = float(np.median(finite)) if finite.size else 0.0
        codes.append(np.digitize(values, edges))
    return codes[0] * 16 + codes[1] * 4 + codes[2]


def _reference_effects(
    frame: pd.DataFrame,
    fit_ids: set[str],
    eval_ids: set[str],
    surfaces: dict[str, list[str]],
    families: dict[str, list[str]],
    candidate: Candidate,
    outer: int,
    workers: int,
    directory: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Any]:
    columns = surfaces["full"]
    model, baseline, _ = _fit_predict(
        frame, columns, fit_ids, eval_ids, candidate, workers, SEED + 5000 + outer
    )
    observed = baseline.target_pic50.to_numpy(dtype=float)
    baseline_pred = baseline.predicted_pic50.to_numpy(dtype=float)
    baseline_abs = np.abs(observed - baseline_pred)
    baseline_sq = np.square(observed - baseline_pred)
    paired: list[pd.DataFrame] = []
    ablation_rows: list[dict[str, Any]] = []
    permutation_rows: list[dict[str, Any]] = []
    fit = frame.loc[frame.structure_id.astype(str).isin(fit_ids)]
    evaluation = frame.loc[frame.structure_id.astype(str).isin(eval_ids)].copy()
    strata = _conditional_strata(fit, evaluation)
    for family in sorted(families):
        block = [column for column in families[family] if column in columns]
        if family != "selected_interactions":
            block = sorted(set(block) | set(families.get("selected_interactions", [])))
        if not block:
            continue
        ablated_columns = [column for column in columns if column not in set(block)]
        ablated_candidate = Candidate(
            f"{candidate.candidate_id}_minus_{family}",
            candidate.engine,
            "full_minus_family",
            candidate.params,
        )
        _, ablated, elapsed = _fit_predict(
            frame,
            ablated_columns,
            fit_ids,
            eval_ids,
            ablated_candidate,
            workers,
            SEED + 10_000 + outer,
        )
        ablated_pred = ablated.predicted_pic50.to_numpy(dtype=float)
        ablation_rows.append(
            {
                "outer_fold": outer,
                "candidate_id": candidate.candidate_id,
                "removed_family": family,
                "baseline_mae": float(np.mean(baseline_abs)),
                "ablated_mae": float(np.mean(np.abs(observed - ablated_pred))),
                "delta_mae": float(np.mean(np.abs(observed - ablated_pred) - baseline_abs)),
                "feature_count_removed": len(block),
                "fit_seconds": elapsed,
            }
        )
        paired.append(
            _paired_frame(
                baseline,
                baseline_abs,
                baseline_sq,
                ablated_pred,
                family,
                "grouped_ablation",
                outer,
                -1,
            )
        )
        for repeat in range(PERMUTATION_REPEATS):
            perturbed = evaluation[columns].copy()
            rng = np.random.default_rng(SEED + outer * 1000 + repeat * 100 + len(block))
            for stratum in np.unique(strata):
                indices = np.flatnonzero(strata == stratum)
                if len(indices) > 1:
                    donor = rng.permutation(indices)
                    perturbed.iloc[indices, perturbed.columns.get_indexer(block)] = evaluation.iloc[donor][
                        block
                    ].to_numpy()
            predicted = np.asarray(model.predict(perturbed), dtype=float)
            delta = np.abs(observed - predicted) - baseline_abs
            permutation_rows.append(
                {
                    "outer_fold": outer,
                    "family": family,
                    "repeat": repeat,
                    "mae_delta": float(np.mean(delta)),
                    "rmse_delta": float(
                        math.sqrt(np.mean(np.square(observed - predicted))) - math.sqrt(np.mean(baseline_sq))
                    ),
                    "feature_count_permuted": len(block),
                    "conditioning": "fit-derived MolWt/MolLogP/TPSA quartile cells",
                }
            )
            paired.append(
                _paired_frame(
                    baseline,
                    baseline_abs,
                    baseline_sq,
                    predicted,
                    family,
                    "conditional_permutation",
                    outer,
                    repeat,
                )
            )
    return (
        baseline,
        pd.DataFrame(ablation_rows),
        pd.DataFrame(permutation_rows),
        pd.concat(paired, ignore_index=True),
        model,
    )


def _paired_frame(
    baseline: pd.DataFrame,
    baseline_abs: np.ndarray,
    baseline_sq: np.ndarray,
    perturbed_prediction: np.ndarray,
    family: str,
    evidence_type: str,
    outer: int,
    repeat: int,
) -> pd.DataFrame:
    observed = baseline.target_pic50.to_numpy(dtype=float)
    return pd.DataFrame(
        {
            "hypothesis_id": f"{evidence_type}::{family}",
            "evidence_type": evidence_type,
            "outer_fold": outer,
            "structure_id": baseline.structure_id.astype(str).to_numpy(),
            "scaffold_group_id": baseline.scaffold_group_id.astype(str).to_numpy(),
            "family": family,
            "repeat": repeat,
            "baseline_abs_error": baseline_abs,
            "perturbed_abs_error": np.abs(observed - perturbed_prediction),
            "baseline_sq_error": baseline_sq,
            "perturbed_sq_error": np.square(observed - perturbed_prediction),
        }
    )


def _outer_unit(
    output: Path,
    frame: pd.DataFrame,
    splits: pd.DataFrame,
    schema: dict[str, Any],
    candidates: list[Candidate],
    inner_units: list[dict[str, Any]],
    outer: int,
    workers: int,
) -> dict[str, Any]:
    scores = {
        str(unit["unit_spec"]["candidate"]["candidate_id"]): float(unit["metrics"]["selection_score"])
        for unit in inner_units
    }
    winner = min(candidates, key=lambda candidate: (scores[candidate.candidate_id], candidate.candidate_id))
    full_candidates = [candidate for candidate in candidates if candidate.surface == "full"]
    reference = min(
        full_candidates, key=lambda candidate: (scores[candidate.candidate_id], candidate.candidate_id)
    )
    unit_id = f"outer_o{outer}"
    directory = output / "units" / unit_id
    spec = {
        "operation": "nested_outer_and_relationships",
        "outer_fold": outer,
        "winner": winner.payload(),
        "reference": reference.payload(),
        "selection_scores": scores,
        "permutation_repeats": PERMUTATION_REPEATS,
    }
    existing = _existing_unit(directory, spec, output)
    if existing is not None:
        return existing
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    assignments = splits.loc[splits.outer_fold.eq(outer)]
    fit_ids = set(assignments.loc[assignments.outer_role.eq("fit"), "structure_id"].astype(str))
    eval_ids = set(assignments.loc[assignments.outer_role.eq("heldout"), "structure_id"].astype(str))
    surfaces = schema["surfaces"]
    families = schema["families"]
    model, nested, elapsed = _fit_predict(
        frame,
        surfaces[winner.surface],
        fit_ids,
        eval_ids,
        winner,
        workers,
        SEED + outer,
    )
    nested = nested.rename(columns={"target_pic50": "observed_pic50"})
    nested["outer_fold"] = outer
    nested["engine"] = winner.engine
    nested["candidate_id"] = winner.candidate_id
    nested["feature_surface"] = winner.surface
    nested["residual"] = nested.observed_pic50 - nested.predicted_pic50
    reference_oof, ablation, permutation, paired, reference_model = _reference_effects(
        frame,
        fit_ids,
        eval_ids,
        surfaces,
        families,
        reference,
        outer,
        workers,
        directory,
    )
    reference_oof = reference_oof.rename(columns={"target_pic50": "observed_pic50"})
    reference_oof["outer_fold"] = outer
    reference_oof["candidate_id"] = reference.candidate_id
    importance = pd.concat(
        [
            _importance(model, surfaces[winner.surface], outer, winner, "nested_winner"),
            _importance(reference_model, surfaces["full"], outer, reference, "relationship_reference"),
        ],
        ignore_index=True,
    )
    paths = {
        "nested_oof": directory / "nested_oof.parquet",
        "relationship_reference_oof": directory / "relationship_reference_oof.parquet",
        "grouped_ablation": directory / "grouped_ablation.parquet",
        "conditional_permutation": directory / "conditional_permutation.parquet",
        "paired_effects": directory / "paired_effects.parquet",
        "feature_importance": directory / "feature_importance.parquet",
        "model": directory / "nested_model.joblib",
    }
    _atomic_parquet(paths["nested_oof"], nested)
    _atomic_parquet(paths["relationship_reference_oof"], reference_oof)
    _atomic_parquet(paths["grouped_ablation"], ablation)
    _atomic_parquet(paths["conditional_permutation"], permutation)
    _atomic_parquet(paths["paired_effects"], paired)
    _atomic_parquet(paths["feature_importance"], importance)
    model_tmp = paths["model"].with_suffix(".joblib.tmp")
    joblib.dump(
        {"model": model, "features": surfaces[winner.surface], "candidate": winner.payload()},
        model_tmp,
    )
    os.replace(model_tmp, paths["model"])
    metric = _metrics(nested.observed_pic50, nested.predicted_pic50)
    metric.update(selection_score=scores[winner.candidate_id], fit_seconds=elapsed)
    return _unit_document(
        directory,
        unit_id,
        spec,
        metric,
        [_binding(path, role) for role, path in paths.items()],
    )


def _input_bindings(feature_root: Path, base_root: Path) -> list[dict[str, Any]]:
    paths = [
        Path(__file__),
        feature_root / "manifest.json",
        feature_root / "validation.json",
        feature_root / "feature_index.parquet",
        feature_root / "routing.parquet",
        base_root / "prepared/exact_train_cache.parquet",
        base_root / "prepared/nested_scaffold_splits.parquet",
        *sorted((feature_root / "features").glob("part-*.parquet")),
    ]
    return [_binding(path, f"immutable_input_{index:03d}") for index, path in enumerate(paths)]


def _initial_checkpoint(
    repo: Path, feature_root: Path, base_root: Path, output: Path, workers: int
) -> dict[str, Any]:
    candidates = _candidate_plan()
    _validate_candidate_plan(candidates)
    return _atomic_json(
        output / "checkpoint.json",
        {
            "schema_version": SCHEMA_VERSION,
            "status": "running",
            "created_utc": _utc(),
            "repo_root": str(repo),
            "feature_root": str(feature_root),
            "base_root": str(base_root),
            "output_root": str(output),
            "workers": workers,
            "candidate_plan": [candidate.payload() for candidate in candidates],
            "input_bindings": _input_bindings(feature_root, base_root),
            "completed_units": [],
            "active_seconds": 0.0,
        },
        "checkpoint_sha256",
    )


def _checkpoint(output: Path, checkpoint: dict[str, Any]) -> dict[str, Any]:
    checkpoint["updated_utc"] = _utc()
    return _atomic_json(output / "checkpoint.json", checkpoint, "checkpoint_sha256")


def _resume(args: argparse.Namespace, output: Path) -> dict[str, Any]:
    checkpoint = _read_json(output / "checkpoint.json", "checkpoint_sha256")
    expected = {
        "repo_root": str(Path(args.repo_root).resolve()),
        "feature_root": str(Path(args.feature_root).resolve()),
        "base_root": str(Path(args.base_root).resolve()),
        "output_root": str(output),
        "workers": int(args.workers),
    }
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise CampaignError(f"resume argument changed: {key}")
    for binding in checkpoint["input_bindings"]:
        _verify_binding(binding)
    if checkpoint["candidate_plan"] != [candidate.payload() for candidate in _candidate_plan()]:
        raise CampaignError("candidate implementation changed")
    checkpoint["status"] = "running"
    return _checkpoint(output, checkpoint)


def _aggregate(output: Path, checkpoint: dict[str, Any], outer_units: list[dict[str, Any]]) -> dict[str, Any]:
    nested = pd.concat(
        [pd.read_parquet(_role_path(unit, "nested_oof")) for unit in outer_units], ignore_index=True
    )
    if (
        len(nested) != EXACT_ROWS
        or nested.structure_id.duplicated().any()
        or set(nested.outer_fold.astype(int)) != set(OUTER_FOLDS)
    ):
        raise CampaignError("nested OOF does not cover exactly 18,801 structures once")
    artifacts: dict[str, pd.DataFrame] = {
        "nested_oof_predictions": nested,
        "fold_metrics": pd.DataFrame(
            [
                {
                    "outer_fold": int(unit["unit_spec"]["outer_fold"]),
                    "candidate_id": unit["unit_spec"]["winner"]["candidate_id"],
                    "engine": unit["unit_spec"]["winner"]["engine"],
                    "feature_surface": unit["unit_spec"]["winner"]["surface"],
                    **unit["metrics"],
                }
                for unit in outer_units
            ]
        ),
        "candidate_metrics": pd.concat(
            [
                pd.read_parquet(path)
                for path in sorted((output / "units").glob("inner_*/inner_metrics.parquet"))
            ],
            ignore_index=True,
        ),
        "relationship_reference_oof": pd.concat(
            [pd.read_parquet(_role_path(unit, "relationship_reference_oof")) for unit in outer_units],
            ignore_index=True,
        ),
        "grouped_ablation": pd.concat(
            [pd.read_parquet(_role_path(unit, "grouped_ablation")) for unit in outer_units],
            ignore_index=True,
        ),
        "conditional_permutation": pd.concat(
            [pd.read_parquet(_role_path(unit, "conditional_permutation")) for unit in outer_units],
            ignore_index=True,
        ),
        "paired_effects": pd.concat(
            [pd.read_parquet(_role_path(unit, "paired_effects")) for unit in outer_units],
            ignore_index=True,
        ),
        "feature_importance": pd.concat(
            [pd.read_parquet(_role_path(unit, "feature_importance")) for unit in outer_units],
            ignore_index=True,
        ),
    }
    bindings: list[dict[str, Any]] = []
    for role, frame in artifacts.items():
        path = output / f"{role}.parquet"
        _atomic_parquet(path, frame)
        bindings.append(_binding(path, role))
    for role, path in (
        ("feature_schemas", output / "prepared/feature_schemas.json"),
        ("sanitization_report", output / "prepared/sanitization_report.parquet"),
        ("conformer_qc", output / "prepared/conformer_qc.parquet"),
    ):
        bindings.append(_binding(path, role))
    unit_paths = sorted((output / "units").glob("*/unit.json"))
    expected_unit_count = len(_candidate_plan()) * len(OUTER_FOLDS) + len(OUTER_FOLDS)
    if len(unit_paths) != expected_unit_count:
        raise CampaignError(f"unit document count mismatch: {len(unit_paths)} != {expected_unit_count}")
    unit_bindings = [_binding(path, f"unit_document::{path.parent.name}") for path in unit_paths]
    bindings.extend(unit_bindings)
    overall = _metrics(nested.observed_pic50, nested.predicted_pic50)
    manifest = _atomic_json(
        output / "manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "status": "passed",
            "created_utc": _utc(),
            "scientific_scope": {
                "source_partition": "train",
                "exact_structures": EXACT_ROWS,
                "nested_scaffold_evaluation": True,
                "repository_validation_labels_opened": False,
                "repository_test_labels_opened": False,
                "predictive_superiority_established": False,
            },
            "metrics": overall,
            "input_bindings": checkpoint["input_bindings"],
            "artifacts": bindings,
            "unit_documents": unit_bindings,
        },
        "manifest_sha256",
    )
    validation = _validate(output)
    checkpoint["status"] = "complete"
    checkpoint["finished_utc"] = _utc()
    checkpoint = _checkpoint(output, checkpoint)
    _atomic_json(
        output / "DONE.json",
        {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "finished_utc": _utc(),
            "manifest_sha256": manifest["manifest_sha256"],
            "validation_sha256": validation["validation_sha256"],
            "checkpoint_sha256": checkpoint["checkpoint_sha256"],
        },
        "done_sha256",
    )
    return validation


def _role_path(unit: dict[str, Any], role: str) -> Path:
    for artifact in unit["artifacts"]:
        if artifact["role"] == role:
            return Path(artifact["path"])
    raise CampaignError(f"unit {unit['unit_id']} lacks role {role}")


def _validate(output: Path) -> dict[str, Any]:
    existing_path = output / "validation.json"
    if existing_path.is_file():
        existing = _read_json(existing_path, "validation_sha256")
        if existing.get("status") == "passed":
            manifest = _read_json(output / "manifest.json", "manifest_sha256")
            if existing.get("manifest_sha256_bound") != manifest["manifest_sha256"]:
                raise CampaignError("validation is not bound to the current manifest")
            for binding in manifest["input_bindings"]:
                _verify_binding(binding)
            for binding in manifest["artifacts"]:
                _verify_binding(binding, output)
            return existing
    manifest = _read_json(output / "manifest.json", "manifest_sha256")
    if manifest.get("status") != "passed":
        raise CampaignError("manifest status did not pass")
    scope = manifest["scientific_scope"]
    if (
        scope.get("source_partition") != "train"
        or scope.get("repository_validation_labels_opened") is not False
        or scope.get("repository_test_labels_opened") is not False
    ):
        raise CampaignError("train-only scientific scope failed")
    for binding in manifest["input_bindings"]:
        _verify_binding(binding)
    for binding in manifest["artifacts"]:
        _verify_binding(binding, output)
        if str(binding.get("role", "")).startswith("unit_document::"):
            unit = _read_json(Path(binding["path"]), "unit_json_sha256")
            if unit.get("status") != "passed":
                raise CampaignError(f"unit document is not passed: {binding['path']}")
    nested = pd.read_parquet(output / "nested_oof_predictions.parquet")
    if len(nested) != EXACT_ROWS or nested.structure_id.duplicated().any():
        raise CampaignError("nested OOF identity validation failed")
    if set(nested.outer_fold.astype(int)) != set(OUTER_FOLDS):
        raise CampaignError("nested OOF fold validation failed")
    reference = pd.read_parquet(output / "relationship_reference_oof.parquet")
    if len(reference) != EXACT_ROWS or reference.structure_id.duplicated().any():
        raise CampaignError("relationship reference OOF identity validation failed")
    splits = pd.read_parquet(output / "prepared/fixed_nested_scaffold_splits.parquet")
    for outer in OUTER_FOLDS:
        split_ids = set(
            splits.loc[splits.outer_fold.eq(outer) & splits.outer_role.eq("heldout"), "structure_id"].astype(
                str
            )
        )
        prediction_ids = set(nested.loc[nested.outer_fold.eq(outer), "structure_id"].astype(str))
        if prediction_ids != split_ids:
            raise CampaignError(f"OOF predictions do not match canonical heldout fold {outer}")
        reference_ids = set(reference.loc[reference.outer_fold.eq(outer), "structure_id"].astype(str))
        if reference_ids != split_ids:
            raise CampaignError(f"relationship reference does not match canonical heldout fold {outer}")
    nested_identity = nested[
        ["structure_id", "scaffold_group_id", "observed_pic50", "outer_fold"]
    ].sort_values("structure_id", ignore_index=True)
    reference_identity = reference[
        ["structure_id", "scaffold_group_id", "observed_pic50", "outer_fold"]
    ].sort_values("structure_id", ignore_index=True)
    if not nested_identity.equals(reference_identity):
        raise CampaignError(
            "relationship reference identities, scaffolds, targets, or folds differ from canonical nested OOF"
        )
    validation = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "created_utc": _utc(),
        "exact_train_structures": len(nested),
        "nested_outer_folds": 5,
        "source_partition": "train",
        "repository_validation_labels_opened": False,
        "repository_test_labels_opened": False,
        "manifest_sha256_bound": manifest["manifest_sha256"],
    }
    return _atomic_json(output / "validation.json", validation, "validation_sha256")


def _run(args: argparse.Namespace) -> dict[str, Any]:
    repo = Path(args.repo_root).resolve()
    feature_root = Path(args.feature_root).resolve()
    base_root = Path(args.base_root).resolve()
    output = Path(args.output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with _Lock(output / ".campaign.lock"):
        checkpoint = (
            _resume(args, output)
            if (output / "checkpoint.json").is_file()
            else _initial_checkpoint(repo, feature_root, base_root, output, args.workers)
        )
        if checkpoint.get("status") == "complete":
            validation = _validate(output)
            return {
                "status": "complete",
                "message": "V5 already complete and revalidated",
                "validation_sha256": validation["validation_sha256"],
            }
        _prepare(repo, feature_root, base_root, output)
        frame, splits, schema = _load_prepared(output)
        candidates = _candidate_plan()
        stop_requested = False

        def request_stop(_signum: int, _frame: object) -> None:
            nonlocal stop_requested
            stop_requested = True

        old_handlers = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)}
        started = time.monotonic()
        try:
            inner_by_outer: dict[int, list[dict[str, Any]]] = {}
            for outer in OUTER_FOLDS:
                inner_by_outer[outer] = []
                for candidate in candidates:
                    if stop_requested:
                        checkpoint["status"] = "safely_stopped"
                        _checkpoint(output, checkpoint)
                        return {"status": "safely_stopped", "resume": "rerun identical command"}
                    unit = _inner_unit(
                        output, frame, splits, schema["surfaces"], candidate, outer, args.workers
                    )
                    inner_by_outer[outer].append(unit)
                    if unit["unit_id"] not in checkpoint["completed_units"]:
                        checkpoint["completed_units"].append(unit["unit_id"])
                        _checkpoint(output, checkpoint)
            outer_units: list[dict[str, Any]] = []
            for outer in OUTER_FOLDS:
                if stop_requested:
                    checkpoint["status"] = "safely_stopped"
                    _checkpoint(output, checkpoint)
                    return {"status": "safely_stopped", "resume": "rerun identical command"}
                unit = _outer_unit(
                    output,
                    frame,
                    splits,
                    schema,
                    candidates,
                    inner_by_outer[outer],
                    outer,
                    args.workers,
                )
                outer_units.append(unit)
                if unit["unit_id"] not in checkpoint["completed_units"]:
                    checkpoint["completed_units"].append(unit["unit_id"])
                    _checkpoint(output, checkpoint)
            checkpoint["active_seconds"] = float(checkpoint.get("active_seconds", 0.0)) + (
                time.monotonic() - started
            )
            return _aggregate(output, checkpoint, outer_units)
        finally:
            for sig, handler in old_handlers.items():
                signal.signal(sig, handler)


def _status(output: Path) -> dict[str, Any]:
    checkpoint = _read_json(output.resolve() / "checkpoint.json", "checkpoint_sha256")
    return {
        "status": checkpoint["status"],
        "completed_units": len(checkpoint.get("completed_units", [])),
        "planned_units": len(_candidate_plan()) * 5 + 5,
        "updated_utc": checkpoint.get("updated_utc"),
        "output_root": checkpoint["output_root"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--repo-root", default=".")
    run.add_argument("--feature-root", default=str(DEFAULT_FEATURE_ROOT))
    run.add_argument("--base-root", default=str(DEFAULT_BASE_ROOT))
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
            if not 1 <= int(args.workers) <= 6:
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
    raise SystemExit(main())
