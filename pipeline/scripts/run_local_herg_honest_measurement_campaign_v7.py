#!/usr/bin/env python3
"""Run the train-only hERG honest measurement-model campaign V7.

V7 addresses the dominant failure modes found in V2--V6 instead of adding
another undirected descriptor sweep.  It preserves the immutable five-fold
scaffold split, models assay/source offsets from observation-level data using
fit-fold records only, compares conventional and tail-aware regressors, and
produces both accuracy-first and safety-oriented nested OOF estimates.

The repository validation and test labels are never returned to this analysis
frame.  This is internal scaffold-held-out evidence, not external or
prospective validation and not proof of clinical utility.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import math
import os
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
from scipy.special import expit
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score
from xgboost import XGBRegressor

SCHEMA_VERSION = "platform-local-herg-honest-measurement-campaign-v7/1.0"
EXACT_STRUCTURES = 18_801
EXACT_OBSERVATIONS = 27_728
OUTER_FOLDS = tuple(range(5))
INNER_FOLDS = tuple(range(3))
SEED = 20260814

DEFAULT_MATRIX_ROOT = Path("research/local_runs/herg_fundamental_optimization_v6")
DEFAULT_OBSERVATIONS = Path(
    "research/data/platform/processed/herg_hierarchy/v1_6_training_surfaces/"
    "herg_training_observations.parquet"
)
DEFAULT_OUTPUT = Path("research/local_runs/herg_honest_measurement_campaign_v7")

CONTEXT_COLUMNS = (
    "source_family",
    "assay_family",
    "measurement_modality",
    "automation_class",
    "modality_confidence",
    "endpoint_class",
    "protocol_completeness_score",
)
OFFSET_PRIOR = {
    "source_family": 40.0,
    "assay_family": 35.0,
    "measurement_modality": 30.0,
    "automation_class": 30.0,
    "modality_confidence": 30.0,
    "endpoint_class": 40.0,
    "protocol_completeness_score": 25.0,
}


class CampaignError(RuntimeError):
    """Scientific, integrity, or reproducibility contract failure."""


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    engine: str
    surface: str
    params: dict[str, Any]
    measurement_correction: bool = False
    quality_weighting: bool = False
    tail_weight: float = 0.0
    mixture: bool = False

    def payload(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "engine": self.engine,
            "surface": self.surface,
            "params": self.params,
            "measurement_correction": self.measurement_correction,
            "quality_weighting": self.quality_weighting,
            "tail_weight": self.tail_weight,
            "mixture": self.mixture,
        }


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"


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
        raise CampaignError(f"missing {role}: {path}")
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
        raise CampaignError(f"artifact escapes output root: {path}")
    if not path.is_file() or path.stat().st_size != int(binding["bytes"]):
        raise CampaignError(f"artifact path or size changed: {path}")
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
            raise CampaignError(f"campaign lock is already held: {self.path}") from error
        return self

    def __exit__(self, *_: object) -> None:
        fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        self.handle.close()


def _candidate_plan() -> list[Candidate]:
    xgb_anchor = {
        "n_estimators": 1200,
        "max_depth": 8,
        "learning_rate": 0.02,
        "min_child_weight": 8.0,
        "subsample": 0.75,
        "colsample_bytree": 0.60,
        "reg_alpha": 0.50,
        "reg_lambda": 5.0,
        "max_bin": 96,
    }
    xgb_robust = {
        "n_estimators": 950,
        "max_depth": 6,
        "learning_rate": 0.028,
        "min_child_weight": 12.0,
        "subsample": 0.78,
        "colsample_bytree": 0.65,
        "reg_alpha": 1.0,
        "reg_lambda": 8.0,
        "max_bin": 96,
    }
    lgb = {
        "n_estimators": 1100,
        "num_leaves": 31,
        "learning_rate": 0.025,
        "min_child_samples": 30,
        "subsample": 0.80,
        "colsample_bytree": 0.72,
        "reg_alpha": 0.25,
        "reg_lambda": 5.0,
        "max_bin": 127,
    }
    return [
        Candidate("xgb_median_v2_anchor", "xgboost", "2d_morgan", xgb_anchor),
        Candidate(
            "xgb_measurement_corrected",
            "xgboost",
            "2d_morgan",
            xgb_anchor,
            measurement_correction=True,
        ),
        Candidate(
            "xgb_measurement_quality",
            "xgboost",
            "2d_morgan",
            xgb_anchor,
            measurement_correction=True,
            quality_weighting=True,
        ),
        Candidate(
            "xgb_measurement_tail2",
            "xgboost",
            "2d_morgan",
            xgb_robust,
            measurement_correction=True,
            quality_weighting=True,
            tail_weight=2.0,
        ),
        Candidate(
            "xgb_measurement_tail4_mixture",
            "xgboost",
            "2d_morgan",
            xgb_robust,
            measurement_correction=True,
            quality_weighting=True,
            tail_weight=4.0,
            mixture=True,
        ),
        Candidate(
            "xgb_qcphysics_measurement",
            "xgboost",
            "qc_physics",
            xgb_robust,
            measurement_correction=True,
            quality_weighting=True,
        ),
        Candidate("lgb_median_l1", "lightgbm", "2d_morgan", lgb),
        Candidate(
            "lgb_measurement_l1",
            "lightgbm",
            "2d_morgan",
            lgb,
            measurement_correction=True,
            quality_weighting=True,
        ),
        Candidate(
            "lgb_measurement_tail2",
            "lightgbm",
            "2d_morgan",
            lgb,
            measurement_correction=True,
            quality_weighting=True,
            tail_weight=2.0,
        ),
        Candidate(
            "lgb_qcphysics_measurement",
            "lightgbm",
            "qc_physics",
            lgb,
            measurement_correction=True,
            quality_weighting=True,
        ),
    ]


def _safe_correlation(function: Any, observed: np.ndarray, predicted: np.ndarray) -> float | None:
    if np.std(observed) <= 1e-12 or np.std(predicted) <= 1e-12:
        return None
    value = float(function(observed, predicted).statistic)
    return value if math.isfinite(value) else None


def _threshold_metrics(observed: np.ndarray, predicted: np.ndarray, threshold: float) -> dict[str, Any]:
    actual = observed >= threshold
    called = predicted >= threshold
    tp = int(np.sum(actual & called))
    tn = int(np.sum(~actual & ~called))
    fp = int(np.sum(~actual & called))
    fn = int(np.sum(actual & ~called))
    recall = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    precision = tp / (tp + fp) if tp + fp else None
    roc = float(roc_auc_score(actual, predicted)) if len(np.unique(actual)) == 2 else None
    return {
        "threshold_pic50": threshold,
        "positive_prevalence": float(np.mean(actual)),
        "recall": recall,
        "specificity": specificity,
        "balanced_accuracy": (recall + specificity) / 2
        if recall is not None and specificity is not None
        else None,
        "precision": precision,
        "roc_auc": roc,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def _metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    absolute = np.abs(observed - predicted)
    edges = (-np.inf, 4.0, 5.0, 6.0, 7.0, np.inf)
    bin_rows: list[dict[str, Any]] = []
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        mask = (observed >= lower) & (observed < upper)
        if mask.any():
            bin_rows.append(
                {
                    "lower": None if not math.isfinite(lower) else lower,
                    "upper": None if not math.isfinite(upper) else upper,
                    "n": int(mask.sum()),
                    "mae": float(np.mean(absolute[mask])),
                }
            )
    balanced_bin = float(np.mean([row["mae"] for row in bin_rows]))
    tail_mask = (observed < 4.0) | (observed >= 7.0)
    mae = float(np.mean(absolute))
    return {
        "n": len(observed),
        "mae": mae,
        "rmse": float(math.sqrt(mean_squared_error(observed, predicted))),
        "r2": float(r2_score(observed, predicted)),
        "pearson": _safe_correlation(pearsonr, observed, predicted),
        "spearman": _safe_correlation(spearmanr, observed, predicted),
        "fraction_within_0p5": float(np.mean(absolute <= 0.5)),
        "fraction_within_1p0": float(np.mean(absolute <= 1.0)),
        "balanced_potency_bin_mae": balanced_bin,
        "tail_mae": float(np.mean(absolute[tail_mask])) if tail_mask.any() else None,
        "accuracy_selection_score": mae,
        "safety_selection_score": 0.70 * mae + 0.30 * balanced_bin,
        "potency_bins": bin_rows,
        "thresholds": {
            "20uM": _threshold_metrics(observed, predicted, -math.log10(20e-6)),
            "10uM": _threshold_metrics(observed, predicted, 5.0),
            "1uM": _threshold_metrics(observed, predicted, 6.0),
        },
    }


def _load_train_observations(path: Path) -> pd.DataFrame:
    columns = [
        "observation_id",
        "structure_id",
        "scaffold_group_id",
        "potency_pic50_point",
        *CONTEXT_COLUMNS,
        "v1_5_conflict_review_structure",
        "evaluation_or_lineage_leakage_caution",
    ]
    table = pq.read_table(
        path,
        columns=columns,
        filters=[
            ("model_split", "=", "train"),
            ("standardized_pic50_primary", "=", True),
            ("potency_relation_pic50", "=", "="),
        ],
    )
    frame = table.to_pandas()
    frame = frame.loc[frame.potency_pic50_point.notna()].copy()
    if len(frame) != EXACT_OBSERVATIONS or frame.observation_id.duplicated().any():
        raise CampaignError(
            f"expected {EXACT_OBSERVATIONS:,} unique exact train observations; got {len(frame):,}"
        )
    return frame


def _prepare(matrix_root: Path, observations_path: Path, output: Path) -> dict[str, Any]:
    prepared = output / "prepared"
    validation_path = prepared / "validation.json"
    if validation_path.is_file():
        validation = _read_json(validation_path, "validation_sha256")
        for binding in validation["artifacts"]:
            _verify_binding(binding, output)
        return validation
    source_validation = _read_json(matrix_root / "validation.json", "validation_sha256")
    if source_validation.get("status") != "passed":
        raise CampaignError("source feature campaign validation did not pass")
    matrix = pd.read_parquet(matrix_root / "prepared/training_matrix.parquet")
    splits = pd.read_parquet(matrix_root / "prepared/fixed_nested_scaffold_splits.parquet")
    schema = _read_json(matrix_root / "prepared/feature_schemas.json", "feature_schema_sha256")
    observations = _load_train_observations(observations_path)
    if len(matrix) != EXACT_STRUCTURES or matrix.structure_id.duplicated().any():
        raise CampaignError("molecular matrix is not 18,801 unique structures")
    expected_ids = set(matrix.structure_id.astype(str))
    if set(observations.structure_id.astype(str)) != expected_ids:
        raise CampaignError("observation and molecular identity sets differ")
    if len(splits) != EXACT_STRUCTURES * 5:
        raise CampaignError("fixed nested split registry row count is invalid")
    for outer in OUTER_FOLDS:
        part = splits.loc[splits.outer_fold.eq(outer)]
        if set(part.structure_id.astype(str)) != expected_ids or part.structure_id.duplicated().any():
            raise CampaignError(f"split identity coverage failed for outer fold {outer}")
        fit_scaffolds = set(part.loc[part.outer_role.eq("fit"), "scaffold_group_id"].astype(str))
        held_scaffolds = set(part.loc[part.outer_role.eq("heldout"), "scaffold_group_id"].astype(str))
        if fit_scaffolds & held_scaffolds:
            raise CampaignError(f"scaffold leakage in outer fold {outer}")
    required_surfaces = {"2d_morgan", "qc_physics"}
    if not required_surfaces <= set(schema["surfaces"]):
        raise CampaignError(
            f"source schema lacks surfaces: {sorted(required_surfaces - set(schema['surfaces']))}"
        )
    prepared.mkdir(parents=True, exist_ok=True)
    matrix_path = prepared / "molecular_matrix.parquet"
    observations_copy = prepared / "exact_train_observations.parquet"
    split_copy = prepared / "fixed_nested_scaffold_splits.parquet"
    schema_copy = prepared / "feature_schemas.json"
    _atomic_parquet(matrix_path, matrix)
    _atomic_parquet(observations_copy, observations)
    _atomic_parquet(split_copy, splits)
    _atomic_json(
        schema_copy,
        {
            "schema_version": SCHEMA_VERSION,
            "source_feature_schema_sha256": schema["feature_schema_sha256"],
            "surfaces": {key: schema["surfaces"][key] for key in sorted(required_surfaces)},
            "context_columns": list(CONTEXT_COLUMNS),
        },
        "feature_schema_sha256",
    )
    return _atomic_json(
        validation_path,
        {
            "schema_version": SCHEMA_VERSION,
            "status": "passed",
            "created_utc": _utc(),
            "exact_train_structures": len(matrix),
            "exact_train_observations": len(observations),
            "source_partition": "train",
            "nontraining_label_values_returned_to_analysis_frame": False,
            "repository_validation_labels_opened": False,
            "repository_test_labels_opened": False,
            "artifacts": [
                _binding(matrix_path, "molecular_matrix"),
                _binding(observations_copy, "exact_train_observations"),
                _binding(split_copy, "fixed_nested_scaffold_splits"),
                _binding(schema_copy, "feature_schemas"),
            ],
        },
        "validation_sha256",
    )


def _load_prepared(output: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    validation = _read_json(output / "prepared/validation.json", "validation_sha256")
    for binding in validation["artifacts"]:
        _verify_binding(binding, output)
    matrix = pd.read_parquet(output / "prepared/molecular_matrix.parquet")
    observations = pd.read_parquet(output / "prepared/exact_train_observations.parquet")
    splits = pd.read_parquet(output / "prepared/fixed_nested_scaffold_splits.parquet")
    schema = _read_json(output / "prepared/feature_schemas.json", "feature_schema_sha256")
    return matrix, observations, splits, schema


def _fit_measurement_offsets(
    observations: pd.DataFrame, fit_ids: set[str]
) -> tuple[dict[str, dict[str, float]], pd.DataFrame]:
    fit = observations.loc[observations.structure_id.astype(str).isin(fit_ids)].copy()
    if not set(fit.structure_id.astype(str)) <= fit_ids:
        raise CampaignError("measurement offset fitter received non-fit identities")
    y = fit.potency_pic50_point.to_numpy(dtype=float)
    structure_center = fit.groupby("structure_id", observed=True).potency_pic50_point.transform("median")
    residual = y - structure_center.to_numpy(dtype=float)
    offset_tables: dict[str, dict[str, float]] = {}
    total_offset = np.zeros(len(fit), dtype=float)
    for column in CONTEXT_COLUMNS:
        keys = fit[column].astype("string").fillna("<missing>")
        work = pd.DataFrame({"key": keys, "residual": residual - total_offset})
        grouped = work.groupby("key", observed=True).residual.agg(["mean", "count"])
        prior = OFFSET_PRIOR[column]
        grouped["offset"] = grouped["mean"] * grouped["count"] / (grouped["count"] + prior)
        table = {str(key): float(value) for key, value in grouped.offset.items()}
        offset_tables[column] = table
        total_offset += keys.map(table).fillna(0.0).to_numpy(dtype=float)
    fit["estimated_measurement_offset"] = total_offset
    fit["measurement_corrected_pic50"] = y - total_offset
    return offset_tables, fit


def _structure_training_targets(
    matrix: pd.DataFrame,
    observations: pd.DataFrame,
    fit_ids: set[str],
    candidate: Candidate,
) -> pd.DataFrame:
    base = matrix.loc[matrix.structure_id.astype(str).isin(fit_ids), ["structure_id", "target_pic50"]].copy()
    base = base.rename(columns={"target_pic50": "canonical_target"})
    obs = observations.loc[observations.structure_id.astype(str).isin(fit_ids)].copy()
    if candidate.measurement_correction:
        _tables, obs = _fit_measurement_offsets(observations, fit_ids)
        target_column = "measurement_corrected_pic50"
    else:
        target_column = "potency_pic50_point"
    summaries: list[dict[str, Any]] = []
    for structure_id, part in obs.groupby("structure_id", sort=False, observed=True):
        values = part[target_column].to_numpy(dtype=float)
        center = float(np.median(values))
        spread = float(np.max(values) - np.min(values)) if len(values) > 1 else 0.0
        if candidate.quality_weighting:
            protocol = part.protocol_completeness_score.to_numpy(dtype=float)
            confidence = (
                part.modality_confidence.astype(str)
                .map({"high": 1.0, "medium": 0.85, "low": 0.65, "unresolved": 0.55})
                .fillna(0.5)
                .to_numpy(dtype=float)
            )
            row_weights = (0.5 + protocol / 6.0) * confidence
            clipped = np.clip(values, center - 1.5, center + 1.5)
            target = float(np.average(clipped, weights=row_weights))
        else:
            target = center
        conflict = bool(part.v1_5_conflict_review_structure.any())
        caution = bool(part.evaluation_or_lineage_leakage_caution.any())
        quality_weight = math.sqrt(len(values)) / (1.0 + spread)
        if conflict:
            quality_weight *= 0.45
        if caution:
            quality_weight *= 0.65
        summaries.append(
            {
                "structure_id": str(structure_id),
                "training_target": target,
                "observation_count": len(values),
                "replicate_range": spread,
                "conflict": conflict,
                "lineage_or_evaluation_caution": caution,
                "quality_weight": float(np.clip(quality_weight, 0.15, 3.0)),
            }
        )
    result = base.merge(pd.DataFrame(summaries), on="structure_id", validate="one_to_one")
    if len(result) != len(base):
        raise CampaignError("structure training-target aggregation lost identities")
    if not candidate.measurement_correction:
        result["training_target"] = result["canonical_target"]
    weights = (
        result.quality_weight.to_numpy(dtype=float, copy=True)
        if candidate.quality_weighting
        else np.ones(len(result))
    )
    if candidate.tail_weight > 0:
        tail = (result.training_target.lt(4.5) | result.training_target.ge(6.5)).to_numpy(dtype=float)
        weights *= 1.0 + candidate.tail_weight * tail
    result["sample_weight"] = weights
    return result


def _new_model(candidate: Candidate, workers: int, seed: int) -> Any:
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


def _fit_models(
    fit: pd.DataFrame,
    columns: list[str],
    targets: pd.DataFrame,
    candidate: Candidate,
    workers: int,
    seed: int,
) -> dict[str, Any]:
    joined = fit.merge(
        targets[["structure_id", "training_target", "sample_weight"]],
        on="structure_id",
        validate="one_to_one",
    )
    x = joined[columns]
    y = joined.training_target.to_numpy(dtype=float)
    weights = joined.sample_weight.to_numpy(dtype=float)
    base = _new_model(candidate, workers, seed)
    base.fit(x, y, sample_weight=weights)
    models: dict[str, Any] = {"base": base}
    if candidate.mixture:
        low = _new_model(candidate, workers, seed + 101)
        high = _new_model(candidate, workers, seed + 202)
        low_weights = weights * (1.0 + candidate.tail_weight * (y < 4.5))
        high_weights = weights * (1.0 + candidate.tail_weight * (y >= 6.5))
        low.fit(x, y, sample_weight=low_weights)
        high.fit(x, y, sample_weight=high_weights)
        models.update(low=low, high=high)
    return models


def _predict_models(models: dict[str, Any], evaluation: pd.DataFrame, columns: list[str]) -> np.ndarray:
    base = np.asarray(models["base"].predict(evaluation[columns]), dtype=float)
    if "low" not in models:
        return base
    low = np.asarray(models["low"].predict(evaluation[columns]), dtype=float)
    high = np.asarray(models["high"].predict(evaluation[columns]), dtype=float)
    low_gate = expit((4.5 - base) / 0.35)
    high_gate = expit((base - 6.5) / 0.35)
    central = np.maximum(0.0, 1.0 - low_gate - high_gate)
    normalizer = low_gate + high_gate + central
    return (low_gate * low + high_gate * high + central * base) / normalizer


def _fit_predict(
    matrix: pd.DataFrame,
    observations: pd.DataFrame,
    columns: list[str],
    fit_ids: set[str],
    eval_ids: set[str],
    candidate: Candidate,
    workers: int,
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame, float]:
    fit = matrix.loc[matrix.structure_id.astype(str).isin(fit_ids)]
    evaluation = matrix.loc[matrix.structure_id.astype(str).isin(eval_ids)]
    if set(fit.scaffold_group_id.astype(str)) & set(evaluation.scaffold_group_id.astype(str)):
        raise CampaignError("scaffold leakage in fit/evaluation")
    targets = _structure_training_targets(matrix, observations, fit_ids, candidate)
    started = time.monotonic()
    models = _fit_models(fit, columns, targets, candidate, workers, seed)
    predicted = _predict_models(models, evaluation, columns)
    elapsed = time.monotonic() - started
    result = evaluation[["structure_id", "scaffold_group_id", "target_pic50"]].copy()
    result = result.rename(columns={"target_pic50": "observed_pic50"})
    result["predicted_pic50"] = predicted
    return models, result, elapsed


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
                "external_or_prospective_validation": False,
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
        for binding in unit["artifacts"]:
            _verify_binding(binding, output)
        return unit
    except Exception:
        return None


def _inner_unit(
    output: Path,
    matrix: pd.DataFrame,
    observations: pd.DataFrame,
    splits: pd.DataFrame,
    surfaces: dict[str, list[str]],
    candidate: Candidate,
    outer_fold: int,
    workers: int,
) -> dict[str, Any]:
    unit_id = f"inner_o{outer_fold}__{candidate.candidate_id}"
    directory = output / "units" / unit_id
    spec = {
        "operation": "inner_scaffold_selection",
        "outer_fold": outer_fold,
        "inner_folds": list(INNER_FOLDS),
        "candidate": candidate.payload(),
        "feature_count": len(surfaces[candidate.surface]),
    }
    existing = _existing_unit(directory, spec, output)
    if existing is not None:
        return existing
    split = splits.loc[splits.outer_fold.eq(outer_fold) & splits.outer_role.eq("fit")]
    rows: list[pd.DataFrame] = []
    elapsed = 0.0
    for inner in INNER_FOLDS:
        fit_ids = set(split.loc[split.inner_fold.ne(inner), "structure_id"].astype(str))
        eval_ids = set(split.loc[split.inner_fold.eq(inner), "structure_id"].astype(str))
        _models, prediction, seconds = _fit_predict(
            matrix,
            observations,
            surfaces[candidate.surface],
            fit_ids,
            eval_ids,
            candidate,
            workers,
            SEED + outer_fold * 100 + inner,
        )
        prediction["outer_fold"] = outer_fold
        prediction["inner_fold"] = inner
        rows.append(prediction)
        elapsed += seconds
    oof = pd.concat(rows, ignore_index=True)
    if oof.structure_id.duplicated().any() or set(oof.structure_id.astype(str)) != set(
        split.structure_id.astype(str)
    ):
        raise CampaignError(f"inner OOF coverage failed: {unit_id}")
    metrics = _metrics(oof.observed_pic50.to_numpy(), oof.predicted_pic50.to_numpy())
    metrics["fit_elapsed_seconds"] = elapsed
    prediction_path = directory / "inner_oof_predictions.parquet"
    _atomic_parquet(prediction_path, oof)
    return _unit_document(
        directory,
        unit_id,
        spec,
        metrics,
        [_binding(prediction_path, "inner_oof_predictions")],
    )


def _outer_unit(
    output: Path,
    matrix: pd.DataFrame,
    observations: pd.DataFrame,
    splits: pd.DataFrame,
    surfaces: dict[str, list[str]],
    candidate: Candidate,
    outer_fold: int,
    role: str,
    workers: int,
) -> dict[str, Any]:
    unit_id = f"outer_o{outer_fold}__{role}__{candidate.candidate_id}"
    directory = output / "units" / unit_id
    spec = {
        "operation": "outer_scaffold_evaluation",
        "outer_fold": outer_fold,
        "selection_role": role,
        "candidate": candidate.payload(),
        "feature_count": len(surfaces[candidate.surface]),
    }
    existing = _existing_unit(directory, spec, output)
    if existing is not None:
        return existing
    split = splits.loc[splits.outer_fold.eq(outer_fold)]
    fit_ids = set(split.loc[split.outer_role.eq("fit"), "structure_id"].astype(str))
    eval_ids = set(split.loc[split.outer_role.eq("heldout"), "structure_id"].astype(str))
    models, prediction, elapsed = _fit_predict(
        matrix,
        observations,
        surfaces[candidate.surface],
        fit_ids,
        eval_ids,
        candidate,
        workers,
        SEED + 10_000 + outer_fold,
    )
    prediction["outer_fold"] = outer_fold
    prediction["selection_role"] = role
    prediction["candidate_id"] = candidate.candidate_id
    metrics = _metrics(prediction.observed_pic50.to_numpy(), prediction.predicted_pic50.to_numpy())
    metrics["fit_elapsed_seconds"] = elapsed
    prediction_path = directory / "outer_predictions.parquet"
    model_path = directory / "models.joblib"
    _atomic_parquet(prediction_path, prediction)
    joblib.dump(models, model_path, compress=3)
    return _unit_document(
        directory,
        unit_id,
        spec,
        metrics,
        [_binding(prediction_path, "outer_predictions"), _binding(model_path, "outer_models")],
    )


def _artifact_path(unit: dict[str, Any], role: str) -> Path:
    for binding in unit["artifacts"]:
        if binding["role"] == role:
            return Path(binding["path"])
    raise CampaignError(f"unit {unit['unit_id']} lacks artifact role {role}")


def _bootstrap_difference(
    frame: pd.DataFrame, first: str, second: str, replicates: int = 10_000
) -> dict[str, Any]:
    grouped = []
    for _scaffold, part in frame.groupby("scaffold_group_id", observed=True):
        grouped.append(
            (
                float(np.sum(np.abs(part.observed_pic50 - part[first]))),
                float(np.sum(np.abs(part.observed_pic50 - part[second]))),
                len(part),
            )
        )
    values = np.asarray(grouped, dtype=float)
    rng = np.random.default_rng(SEED)
    draws = np.empty(replicates, dtype=float)
    for index in range(replicates):
        chosen = values[rng.integers(0, len(values), len(values))]
        draws[index] = chosen[:, 0].sum() / chosen[:, 2].sum() - chosen[:, 1].sum() / chosen[:, 2].sum()
    point = float(
        np.mean(np.abs(frame.observed_pic50 - frame[first]))
        - np.mean(np.abs(frame.observed_pic50 - frame[second]))
    )
    return {
        "delta_definition": f"{first}_mae_minus_{second}_mae; negative favors {first}",
        "point_estimate": point,
        "ci95_lower": float(np.quantile(draws, 0.025)),
        "ci95_upper": float(np.quantile(draws, 0.975)),
        "probability_first_better": float(np.mean(draws < 0.0)),
        "replicates": replicates,
        "resampling_unit": "scaffold_group_id",
    }


def _full_measurement_report(observations: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    ids = observations.groupby("structure_id", observed=True).potency_pic50_point.agg(["count", "min", "max"])
    ids["range"] = ids["max"] - ids["min"]
    offset_tables, adjusted = _fit_measurement_offsets(
        observations, set(observations.structure_id.astype(str))
    )
    rows = []
    for column, table in offset_tables.items():
        counts = adjusted[column].astype("string").fillna("<missing>").value_counts()
        for level, offset in table.items():
            rows.append(
                {
                    "context_field": column,
                    "context_level": level,
                    "estimated_pic50_offset": offset,
                    "observation_count": int(counts.get(level, 0)),
                }
            )
    report = {
        "exact_observations": len(observations),
        "structures": observations.structure_id.nunique(),
        "replicated_structures": int((ids["count"] >= 2).sum()),
        "replicate_range_gt_0p5": int((ids["range"] > 0.5).sum()),
        "replicate_range_gt_1p0": int((ids["range"] > 1.0).sum()),
        "conflict_structures": int(
            observations.loc[observations.v1_5_conflict_review_structure].structure_id.nunique()
        ),
        "lineage_or_evaluation_caution_structures": int(
            observations.loc[observations.evaluation_or_lineage_leakage_caution].structure_id.nunique()
        ),
        "offsets_are_within_structure_fit_only_estimates": True,
        "offsets_are_not_causal_assay_effects": True,
    }
    return report, pd.DataFrame(rows)


def _final_refit(
    output: Path,
    matrix: pd.DataFrame,
    observations: pd.DataFrame,
    surfaces: dict[str, list[str]],
    candidate: Candidate,
    role: str,
    workers: int,
) -> dict[str, Any]:
    directory = output / "final_models" / role
    directory.mkdir(parents=True, exist_ok=True)
    ids = set(matrix.structure_id.astype(str))
    targets = _structure_training_targets(matrix, observations, ids, candidate)
    models = _fit_models(matrix, surfaces[candidate.surface], targets, candidate, workers, SEED + 50_000)
    offset_tables, _adjusted = _fit_measurement_offsets(observations, ids)
    model_path = directory / "model.joblib"
    schema_path = directory / "model_schema.json"
    joblib.dump(models, model_path, compress=3)
    schema = _atomic_json(
        schema_path,
        {
            "schema_version": SCHEMA_VERSION,
            "model_role": role,
            "candidate": candidate.payload(),
            "feature_columns": surfaces[candidate.surface],
            "feature_count": len(surfaces[candidate.surface]),
            "measurement_offset_tables": offset_tables,
            "training_structures": len(matrix),
            "training_observations": len(observations),
            "source_partition": "train",
            "prediction_contract": {
                "latent_potency": "molecular model output without an assay offset",
                "context_conditioned_potency": "latent potency plus matching measurement offset",
                "unknown_context_offset": 0.0,
            },
        },
        "model_schema_sha256",
    )
    smoke = matrix.iloc[:3]
    prediction = _predict_models(models, smoke, surfaces[candidate.surface])
    smoke_path = directory / "inference_smoke.parquet"
    _atomic_parquet(
        smoke_path,
        pd.DataFrame(
            {
                "structure_id": smoke.structure_id.astype(str),
                "predicted_latent_pic50": prediction,
            }
        ),
    )
    return {
        "role": role,
        "candidate_id": candidate.candidate_id,
        "model_schema_sha256": schema["model_schema_sha256"],
        "artifacts": [
            _binding(model_path, f"{role}_model"),
            _binding(schema_path, f"{role}_schema"),
            _binding(smoke_path, f"{role}_inference_smoke"),
        ],
    }


def _input_bindings(matrix_root: Path, observations_path: Path) -> list[dict[str, Any]]:
    return [
        _binding(Path(__file__), "implementation"),
        _binding(matrix_root / "prepared/training_matrix.parquet", "source_molecular_matrix"),
        _binding(matrix_root / "prepared/fixed_nested_scaffold_splits.parquet", "source_split_registry"),
        _binding(matrix_root / "prepared/feature_schemas.json", "source_feature_schema"),
        _binding(matrix_root / "validation.json", "source_validation"),
        _binding(observations_path, "source_training_observations"),
    ]


def _checkpoint(output: Path, document: dict[str, Any]) -> dict[str, Any]:
    document["updated_utc"] = _utc()
    return _atomic_json(output / "checkpoint.json", document, "checkpoint_sha256")


def _initial_checkpoint(matrix_root: Path, observations: Path, output: Path, workers: int) -> dict[str, Any]:
    return _checkpoint(
        output,
        {
            "schema_version": SCHEMA_VERSION,
            "status": "running",
            "created_utc": _utc(),
            "output_root": str(output),
            "workers": workers,
            "completed_units": [],
            "active_seconds": 0.0,
            "input_bindings": _input_bindings(matrix_root, observations),
            "candidate_plan_sha256": _digest([candidate.payload() for candidate in _candidate_plan()]),
        },
    )


def _resume(output: Path, matrix_root: Path, observations: Path, workers: int) -> dict[str, Any]:
    checkpoint = _read_json(output / "checkpoint.json", "checkpoint_sha256")
    if Path(checkpoint["output_root"]).resolve() != output.resolve() or int(checkpoint["workers"]) != workers:
        raise CampaignError("resume path or worker count differs from the original run")
    expected = _input_bindings(matrix_root, observations)
    if checkpoint["input_bindings"] != expected:
        raise CampaignError("resume input or implementation bindings changed")
    if checkpoint["candidate_plan_sha256"] != _digest(
        [candidate.payload() for candidate in _candidate_plan()]
    ):
        raise CampaignError("candidate plan changed")
    return checkpoint


def _aggregate(
    output: Path,
    checkpoint: dict[str, Any],
    matrix: pd.DataFrame,
    observations: pd.DataFrame,
    surfaces: dict[str, list[str]],
    outer_units: dict[str, list[dict[str, Any]]],
    winners: dict[str, list[Candidate]],
    workers: int,
) -> dict[str, Any]:
    predictions: dict[str, pd.DataFrame] = {}
    metrics: dict[str, Any] = {}
    for role, units in outer_units.items():
        frame = pd.concat([pd.read_parquet(_artifact_path(unit, "outer_predictions")) for unit in units])
        if len(frame) != EXACT_STRUCTURES or frame.structure_id.duplicated().any():
            raise CampaignError(f"nested OOF coverage failed for {role}")
        frame = frame.sort_values("structure_id", ignore_index=True)
        predictions[role] = frame
        metrics[role] = _metrics(frame.observed_pic50.to_numpy(), frame.predicted_pic50.to_numpy())
        _atomic_parquet(output / f"nested_{role}_oof_predictions.parquet", frame)
    joined = (
        predictions["accuracy"]
        .rename(columns={"predicted_pic50": "accuracy_prediction"})[
            ["structure_id", "scaffold_group_id", "observed_pic50", "outer_fold", "accuracy_prediction"]
        ]
        .merge(
            predictions["safety"][["structure_id", "predicted_pic50"]].rename(
                columns={"predicted_pic50": "safety_prediction"}
            ),
            on="structure_id",
            validate="one_to_one",
        )
    )
    bootstrap = _bootstrap_difference(joined, "safety_prediction", "accuracy_prediction")
    measurement_report, offsets = _full_measurement_report(observations)
    offsets_path = output / "measurement_offsets.parquet"
    _atomic_parquet(offsets_path, offsets)
    accuracy_counts = pd.Series([candidate.candidate_id for candidate in winners["accuracy"]]).value_counts()
    safety_counts = pd.Series([candidate.candidate_id for candidate in winners["safety"]]).value_counts()
    accuracy_final = next(
        candidate for candidate in _candidate_plan() if candidate.candidate_id == accuracy_counts.index[0]
    )
    safety_final = next(
        candidate for candidate in _candidate_plan() if candidate.candidate_id == safety_counts.index[0]
    )
    final_models = [
        _final_refit(output, matrix, observations, surfaces, accuracy_final, "accuracy", workers),
        _final_refit(output, matrix, observations, surfaces, safety_final, "safety", workers),
    ]
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "created_utc": _utc(),
        "scientific_scope": {
            "endpoint": "wild-type-or-unspecified hERG exact quantitative potency",
            "explicit_human_wt_claim_supported": False,
            "source_partition": "train",
            "repository_validation_labels_opened": False,
            "repository_test_labels_opened": False,
            "external_or_prospective_validation": False,
            "clinical_utility_established": False,
        },
        "metrics": metrics,
        "safety_vs_accuracy_scaffold_bootstrap": bootstrap,
        "outer_winners": {
            "accuracy": [candidate.candidate_id for candidate in winners["accuracy"]],
            "safety": [candidate.candidate_id for candidate in winners["safety"]],
        },
        "final_recipes": {
            "accuracy": accuracy_final.payload(),
            "safety": safety_final.payload(),
        },
        "measurement_report": measurement_report,
        "interpretation_contract": {
            "assay_offsets": "exploratory within-structure measurement associations, not causal effects",
            "accuracy_track": "selected by inner-scaffold MAE",
            "safety_track": "selected by 70% overall MAE plus 30% equal-potency-bin MAE",
            "published_tool_comparison": "only valid for matched endpoint, threshold, split, and population",
        },
    }
    report_path = output / "analysis.json"
    _atomic_json(report_path, report, "analysis_sha256")
    markdown_path = output / "ANALYSIS.md"
    markdown = f"""# hERG honest measurement campaign V7

Status: passed internal train-only nested scaffold evaluation.

The campaign evaluated {EXACT_STRUCTURES:,} structures and {EXACT_OBSERVATIONS:,} exact observations.
Repository validation and test labels were not opened. This is not external or prospective validation.

Accuracy track MAE: {metrics["accuracy"]["mae"]:.6f}
Safety track MAE: {metrics["safety"]["mae"]:.6f}
Accuracy track balanced potency-bin MAE: {metrics["accuracy"]["balanced_potency_bin_mae"]:.6f}
Safety track balanced potency-bin MAE: {metrics["safety"]["balanced_potency_bin_mae"]:.6f}

The endpoint is wild-type-or-unspecified hERG potency. Explicit human-WT status is not established for
the quantitative surface. Assay/source offsets are descriptive measurement associations and must not be
interpreted causally. Threshold metrics, regression metrics, and published tools are comparable only
when endpoint definitions, populations, splitting, and operating thresholds match.
"""
    markdown_path.write_text(markdown, encoding="utf-8")
    artifacts = [
        _binding(output / "nested_accuracy_oof_predictions.parquet", "nested_accuracy_oof"),
        _binding(output / "nested_safety_oof_predictions.parquet", "nested_safety_oof"),
        _binding(offsets_path, "measurement_offsets"),
        _binding(report_path, "analysis"),
        _binding(markdown_path, "analysis_markdown"),
    ]
    for unit_id in sorted(set(checkpoint["completed_units"])):
        artifacts.append(_binding(output / "units" / unit_id / "unit.json", f"unit::{unit_id}"))
    for record in final_models:
        artifacts.extend(record["artifacts"])
    manifest = _atomic_json(
        output / "manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "status": "passed",
            "created_utc": _utc(),
            "input_bindings": checkpoint["input_bindings"],
            "artifacts": artifacts,
            "final_models": final_models,
            "candidate_plan": [candidate.payload() for candidate in _candidate_plan()],
            "scientific_scope": report["scientific_scope"],
        },
        "manifest_sha256",
    )
    checkpoint["status"] = "complete"
    checkpoint["manifest_sha256"] = manifest["manifest_sha256"]
    _checkpoint(output, checkpoint)
    return _validate(output)


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
                raise CampaignError(f"bound unit did not pass: {binding['path']}")
            for artifact in unit["artifacts"]:
                _verify_binding(artifact, output)
    scope = manifest["scientific_scope"]
    if (
        scope.get("source_partition") != "train"
        or scope.get("repository_validation_labels_opened") is not False
        or scope.get("repository_test_labels_opened") is not False
        or scope.get("external_or_prospective_validation") is not False
    ):
        raise CampaignError("scientific scope validation failed")
    splits = pd.read_parquet(output / "prepared/fixed_nested_scaffold_splits.parquet")
    expected_ids = set(
        pd.read_parquet(
            output / "prepared/molecular_matrix.parquet", columns=["structure_id"]
        ).structure_id.astype(str)
    )
    for role in ("accuracy", "safety"):
        oof = pd.read_parquet(output / f"nested_{role}_oof_predictions.parquet")
        if (
            len(oof) != EXACT_STRUCTURES
            or oof.structure_id.duplicated().any()
            or set(oof.structure_id.astype(str)) != expected_ids
        ):
            raise CampaignError(f"nested {role} identity coverage failed")
        for outer in OUTER_FOLDS:
            expected = set(
                splits.loc[
                    splits.outer_fold.eq(outer) & splits.outer_role.eq("heldout"), "structure_id"
                ].astype(str)
            )
            actual = set(oof.loc[oof.outer_fold.eq(outer), "structure_id"].astype(str))
            if actual != expected:
                raise CampaignError(f"nested {role} fold {outer} differs from canonical split")
    for model in manifest["final_models"]:
        for binding in model["artifacts"]:
            _verify_binding(binding, output)
    return _atomic_json(
        output / "validation.json",
        {
            "schema_version": SCHEMA_VERSION,
            "status": "passed",
            "created_utc": _utc(),
            "manifest_sha256_bound": manifest["manifest_sha256"],
            "exact_train_structures": EXACT_STRUCTURES,
            "exact_train_observations": EXACT_OBSERVATIONS,
            "nested_outer_folds": 5,
            "accuracy_and_safety_models_present": True,
            "source_partition": "train",
            "repository_validation_labels_opened": False,
            "repository_test_labels_opened": False,
            "external_or_prospective_validation": False,
        },
        "validation_sha256",
    )


def _run(args: argparse.Namespace) -> dict[str, Any]:
    matrix_root = Path(args.matrix_root).resolve()
    observations_path = Path(args.observations).resolve()
    output = Path(args.output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with _Lock(output / ".campaign.lock"):
        checkpoint = (
            _resume(output, matrix_root, observations_path, args.workers)
            if (output / "checkpoint.json").is_file()
            else _initial_checkpoint(matrix_root, observations_path, output, args.workers)
        )
        if checkpoint.get("status") == "complete":
            validation = _validate(output)
            return {"status": "complete", "message": "V7 already complete and revalidated", **validation}
        _prepare(matrix_root, observations_path, output)
        matrix, observations, splits, schema = _load_prepared(output)
        candidates = _candidate_plan()
        stop_requested = False

        def request_stop(_signum: int, _frame: object) -> None:
            nonlocal stop_requested
            stop_requested = True

        handlers = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)}
        started = time.monotonic()
        try:
            inner_by_outer: dict[int, list[dict[str, Any]]] = {}
            for outer in OUTER_FOLDS:
                inner_by_outer[outer] = []
                for candidate in candidates:
                    if stop_requested:
                        checkpoint["status"] = "safely_stopped"
                        checkpoint["active_seconds"] += time.monotonic() - started
                        _checkpoint(output, checkpoint)
                        return {"status": "safely_stopped", "resume": "rerun identical command"}
                    unit = _inner_unit(
                        output,
                        matrix,
                        observations,
                        splits,
                        schema["surfaces"],
                        candidate,
                        outer,
                        args.workers,
                    )
                    inner_by_outer[outer].append(unit)
                    if unit["unit_id"] not in checkpoint["completed_units"]:
                        checkpoint["completed_units"].append(unit["unit_id"])
                        _checkpoint(output, checkpoint)
            winners: dict[str, list[Candidate]] = {"accuracy": [], "safety": []}
            outer_units: dict[str, list[dict[str, Any]]] = {"accuracy": [], "safety": []}
            for outer in OUTER_FOLDS:
                ranked_accuracy = sorted(
                    inner_by_outer[outer], key=lambda unit: unit["metrics"]["accuracy_selection_score"]
                )
                ranked_safety = sorted(
                    inner_by_outer[outer], key=lambda unit: unit["metrics"]["safety_selection_score"]
                )
                for role, ranked in (("accuracy", ranked_accuracy), ("safety", ranked_safety)):
                    winner_id = ranked[0]["unit_spec"]["candidate"]["candidate_id"]
                    winner = next(
                        candidate for candidate in candidates if candidate.candidate_id == winner_id
                    )
                    winners[role].append(winner)
                    unit = _outer_unit(
                        output,
                        matrix,
                        observations,
                        splits,
                        schema["surfaces"],
                        winner,
                        outer,
                        role,
                        args.workers,
                    )
                    outer_units[role].append(unit)
                    if unit["unit_id"] not in checkpoint["completed_units"]:
                        checkpoint["completed_units"].append(unit["unit_id"])
                        _checkpoint(output, checkpoint)
            checkpoint["active_seconds"] += time.monotonic() - started
            return _aggregate(
                output,
                checkpoint,
                matrix,
                observations,
                schema["surfaces"],
                outer_units,
                winners,
                args.workers,
            )
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)


def _status(output: Path) -> dict[str, Any]:
    checkpoint = _read_json(output.resolve() / "checkpoint.json", "checkpoint_sha256")
    return {
        "status": checkpoint["status"],
        "completed_units": len(checkpoint.get("completed_units", [])),
        "planned_units": len(_candidate_plan()) * 5 + 10,
        "active_seconds": checkpoint.get("active_seconds", 0.0),
        "updated_utc": checkpoint.get("updated_utc"),
        "output_root": checkpoint["output_root"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--matrix-root", default=str(DEFAULT_MATRIX_ROOT))
    run.add_argument("--observations", default=str(DEFAULT_OBSERVATIONS))
    run.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    run.add_argument("--workers", type=int, default=6, choices=range(1, 9))
    for name in ("status", "validate"):
        command = commands.add_parser(name)
        command.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        if args.command == "run":
            result = _run(args)
        elif args.command == "status":
            result = _status(Path(args.output_root))
        else:
            result = _validate(Path(args.output_root).resolve())
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except CampaignError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
