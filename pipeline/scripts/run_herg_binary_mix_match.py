#!/usr/bin/env python3
"""Run interval-aware hERG liability mix-and-match experiments.

This companion analysis exists because exact-value pIC50 regression omits many
right-censored nonblockers.  It makes binary labels only when the measurement
interval is decisive:

* blocker: the entire interval is at or above pIC50 5 (IC50 <= 10 uM);
* nonblocker: the entire interval is at or below pIC50 4.522879
  (IC50 >= 30 uM); and
* intermediate/ambiguous intervals: excluded from binary fitting and scoring.

Internal Menin-inhibitor evidence is included in every candidate training
regime.  Angelo/Ascentage rows are a retrospective same-series extension, and
the Sun data are heterogeneous public evidence rather than protocol-matched
ground truth.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/menin-binary-mix-match-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/menin-binary-mix-match-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    log_loss,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.svm import SVC

warnings.filterwarnings(
    "ignore",
    message="The `probability` parameter was deprecated",
    category=FutureWarning,
)

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "pipeline/scripts"
SRC = ROOT / "pipeline/src"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SRC))

import run_herg_pk_mix_match as base  # noqa: E402
from menin_discovery.research_common import (  # noqa: E402
    atomic_write_csv,
    atomic_write_json,
    atomic_write_parquet,
)
from menin_discovery.research_public_herg import prepare_sun_source_holdout  # noqa: E402

SEED = base.SEED + 1
BOOTSTRAPS = 1200
CHECKPOINT_VERSION = "2026-07-30-binary-v2"
DEFAULT_OUTPUT = base.DEFAULT_OUTPUT

FEATURE_LAYERS = ("compact_proxies", "morgan_latent", "hybrid")
MODEL_NAMES = ("logistic", "svc_rbf", "random_forest", "extra_trees")
REGIMES = (
    "internal_only",
    "internal_plus_extension",
    "internal_plus_public_naive",
    "internal_plus_public_balanced",
    "internal_plus_public_nearest",
    "internal_plus_public_large",
    "internal_plus_extension_plus_public_balanced",
)


def _decisive_interval_frame(
    frame: pd.DataFrame,
    *,
    dataset_role: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return decisive rows plus a transparent excluded-row audit."""
    work = frame.copy()
    conflict = work.get("source_conflict", pd.Series(False, index=work.index))
    conflict = conflict.fillna(False).astype(bool)
    lower = pd.to_numeric(work["pic50_lower"], errors="coerce")
    upper = pd.to_numeric(work["pic50_upper"], errors="coerce")
    blocker = lower.ge(base.PIC50_BLOCKER)
    nonblocker = upper.le(base.PIC50_NONBLOCKER)
    work["target_class"] = np.select(
        [blocker & ~conflict, nonblocker & ~conflict],
        [1.0, 0.0],
        default=np.nan,
    )
    work["binary_label_basis"] = np.select(
        [conflict, blocker, nonblocker],
        [
            "excluded_source_conflict",
            "blocker_interval_entirely_at_or_above_pic50_5",
            "nonblocker_interval_entirely_at_or_below_30uM_boundary",
        ],
        default="excluded_intermediate_or_ambiguous_interval",
    )
    work["dataset_role"] = dataset_role
    audit = work[
        [
            "record_id",
            "compound_id",
            "source_group",
            "pic50_lower",
            "pic50_upper",
            "target_class",
            "binary_label_basis",
            "dataset_role",
        ]
    ].copy()
    decisive = work[work["target_class"].notna()].copy()
    decisive["target_class"] = decisive["target_class"].astype(int)
    keep = [
        "record_id",
        "compound_id",
        "standardized_smiles",
        "scaffold",
        "target_class",
        "source_group",
        "dataset_role",
    ]
    if "structure_id" in decisive:
        keep.append("structure_id")
    return base._attach_features(decisive[keep].reset_index(drop=True)), audit


def _public_classification_frames(
    excluded_structure_ids: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prepared = prepare_sun_source_holdout(base.PUBLIC_ROOT)
    result: list[pd.DataFrame] = []
    for key, role in (
        ("classification_train", "public_classification_train"),
        ("classification_validation", "public_classification_source_holdout"),
    ):
        source = prepared[key].copy()
        source = source[
            source["canonical_blocker_class"].notna()
            & ~source["structure_id"].astype(str).isin(excluded_structure_ids)
        ].copy()
        source["record_id"] = "public-class:" + source["structure_id"].astype(str)
        source["compound_id"] = source["structure_id"].astype(str)
        source["standardized_smiles"] = source["raw_smiles"].astype(str)
        source["target_class"] = source["canonical_blocker_class"].astype(int)
        source["source_group"] = role
        source["dataset_role"] = role
        minimal = source[
            [
                "record_id",
                "compound_id",
                "structure_id",
                "standardized_smiles",
                "scaffold",
                "target_class",
                "computed_mw_g_mol",
                "source_group",
                "dataset_role",
            ]
        ]
        result.append(base._attach_features(minimal))
    return result[0], result[1]


def _class_balanced_weights(
    frame: pd.DataFrame,
    *,
    source_balance: bool,
) -> pd.Series:
    """Balance classes, optionally after equalizing internal-like/public mass."""
    weights = pd.Series(np.ones(len(frame), dtype=float), index=frame.index)
    if source_balance:
        source_bucket = np.where(
            frame["source_group"].isin({"internal", "angelo_same_series_extension"}),
            "internal_like",
            "public",
        )
        cells = pd.Series(
            [
                f"{bucket}:{label}"
                for bucket, label in zip(
                    source_bucket,
                    frame["target_class"].astype(int),
                    strict=True,
                )
            ],
            index=frame.index,
        )
        cell_counts = cells.value_counts()
        if set(cell_counts.index) != {
            "internal_like:0",
            "internal_like:1",
            "public:0",
            "public:1",
        }:
            raise ValueError("Source-balanced pool lacks a source-by-class cell")
        weights = cells.map(1.0 / cell_counts).astype(float)
    else:
        labels = frame["target_class"].to_numpy(dtype=int)
        for label in (0, 1):
            mask = labels == label
            mass = float(weights.loc[mask].sum())
            if mass <= 0:
                raise ValueError("Training pool lacks one decisive hERG class")
            weights.loc[mask] *= float(weights.sum()) / (2.0 * mass)
    weights *= len(weights) / float(weights.sum())
    return weights


def _regime_pool(
    fold: base.Fold,
    public_train: pd.DataFrame,
    regime: str,
) -> tuple[pd.DataFrame, str]:
    test_scaffolds = set(fold.test["scaffold"].astype(str))
    public = public_train[~public_train["scaffold"].astype(str).isin(test_scaffolds)].reset_index(drop=True)
    internal = fold.base_train.copy()
    extension = fold.extension_train.copy()
    source_balance = False
    if regime == "internal_only":
        train = internal
        role = "internal_anchor"
    elif regime == "internal_plus_extension":
        train = pd.concat([internal, extension], ignore_index=True)
        role = "same_series_augmentation"
    elif regime == "internal_plus_public_naive":
        train = pd.concat([internal, public], ignore_index=True)
        role = "negative_control_external_row_dominance"
    elif regime == "internal_plus_public_balanced":
        train = pd.concat([internal, public], ignore_index=True)
        source_balance = True
        role = "source_and_class_mass_balanced_public_pool"
    elif regime == "internal_plus_public_nearest":
        selected = base._select_nearest_public(public, internal)
        train = pd.concat([internal, selected], ignore_index=True)
        source_balance = True
        role = "outcome_blind_nearest_public_pool_source_and_class_balanced"
    elif regime == "internal_plus_public_large":
        selected = public[public["computed_mw_g_mol"].ge(650)].copy()
        train = pd.concat([internal, selected], ignore_index=True)
        source_balance = True
        role = "mw_matched_public_pool_source_and_class_balanced"
    elif regime == "internal_plus_extension_plus_public_balanced":
        train = pd.concat([internal, extension, public], ignore_index=True)
        source_balance = True
        role = "same_series_plus_source_and_class_mass_balanced_public_pool"
    else:
        raise ValueError(f"Unknown classification regime: {regime}")
    train = train.reset_index(drop=True)
    train["sample_weight"] = _class_balanced_weights(
        train,
        source_balance=source_balance,
    )
    return train, role


def _classifier(name: str) -> Any:
    if name == "logistic":
        return LogisticRegression(C=3.0, max_iter=5000, random_state=SEED)
    if name == "svc_rbf":
        return SVC(
            C=3.0,
            gamma="scale",
            probability=True,
            random_state=SEED,
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=240,
            min_samples_leaf=3,
            max_features="sqrt",
            n_jobs=-1,
            random_state=SEED,
        )
    if name == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=240,
            min_samples_leaf=3,
            max_features="sqrt",
            n_jobs=-1,
            random_state=SEED,
        )
    raise ValueError(f"Unknown classifier: {name}")


def _fit_predict_probability(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    feature_layer: str,
    model_name: str,
) -> np.ndarray:
    transformer = base.FeatureTransformer(feature_layer)
    train_x = transformer.fit_transform(train)
    test_x = transformer.transform(test)
    estimator = _classifier(model_name)
    estimator.fit(
        train_x,
        train["target_class"].to_numpy(dtype=int),
        sample_weight=train["sample_weight"].to_numpy(dtype=float),
    )
    return np.asarray(estimator.predict_proba(test_x)[:, 1], dtype=float)


def _tanimoto_probability(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    neighbors: int = 3,
) -> np.ndarray:
    proxy = train.copy()
    proxy["target_pic50"] = proxy["target_class"].astype(float)
    return np.clip(base._tanimoto_predict(proxy, test, neighbors=neighbors), 0.0, 1.0)


def _prediction_frame(
    fold: base.Fold,
    probability: np.ndarray,
    train: pd.DataFrame,
    *,
    regime: str,
    feature_layer: str,
    model: str,
    role: str,
    max_internal: np.ndarray,
    max_any: np.ndarray,
) -> pd.DataFrame:
    internal_like = train[
        train["source_group"].isin({"internal", "angelo_same_series_extension"})
    ].drop_duplicates("record_id")
    result = fold.test[["record_id", "compound_id", "scaffold", "source_group", "target_class"]].copy()
    result = result.rename(
        columns={
            "source_group": "test_source_group",
            "target_class": "observed_class",
        }
    )
    result["evaluation"] = fold.evaluation
    result["fold"] = fold.fold
    result["evidence_status"] = fold.evidence_status
    result["predicted_blocker_probability"] = np.clip(probability, 1e-6, 1 - 1e-6)
    result["predicted_class"] = result["predicted_blocker_probability"].ge(0.5).astype(int)
    result["brier_component"] = np.square(result["predicted_blocker_probability"] - result["observed_class"])
    result["data_regime"] = regime
    result["external_data_role"] = role
    result["feature_layer"] = feature_layer
    result["model"] = model
    result["training_structures"] = train["record_id"].nunique()
    result["training_internal_like_structures"] = internal_like["record_id"].nunique()
    result["training_public_structures"] = train["source_group"].astype(str).str.startswith("public").sum()
    result["max_internal_like_train_tanimoto"] = max_internal
    result["max_any_train_tanimoto"] = max_any
    return result


def _run_matrix(
    internal: pd.DataFrame,
    extension: pd.DataFrame,
    public_train: pd.DataFrame,
    public_validation: pd.DataFrame,
    output: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[pd.DataFrame] = []
    failures: list[dict[str, Any]] = []
    checkpoint_dir = output / "binary_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for fold in base._evaluation_folds(internal, extension, public_validation):
        stem = f"{fold.evaluation}__fold_{fold.fold}"
        prediction_path = checkpoint_dir / f"{stem}.parquet"
        failure_path = checkpoint_dir / f"{stem}_failures.csv"
        if prediction_path.exists():
            cached = pd.read_parquet(prediction_path)
            if not cached.empty and cached["checkpoint_version"].eq(CHECKPOINT_VERSION).all():
                rows.append(cached.drop(columns="checkpoint_version"))
                if failure_path.exists():
                    try:
                        cached_failures = pd.read_csv(failure_path)
                    except pd.errors.EmptyDataError:
                        cached_failures = pd.DataFrame()
                    failures.extend(cached_failures.to_dict("records"))
                print(f"reused binary checkpoint {stem}", flush=True)
                continue
        print(f"running binary {stem}", flush=True)
        fold_rows: list[pd.DataFrame] = []
        fold_failures: list[dict[str, Any]] = []
        for regime in REGIMES:
            if fold.evaluation == "angelo_fixed_nonoverlap" and "extension" in regime:
                continue
            try:
                train, role = _regime_pool(fold, public_train, regime)
                internal_like = train[
                    train["source_group"].isin({"internal", "angelo_same_series_extension"})
                ].drop_duplicates("record_id")
                max_internal = base._max_tanimoto(fold.test, internal_like)
                max_any = base._max_tanimoto(
                    fold.test,
                    train.drop_duplicates("record_id"),
                )
            except Exception as exc:
                fold_failures.append(
                    {
                        "domain": "herg_binary",
                        "evaluation": fold.evaluation,
                        "fold": fold.fold,
                        "data_regime": regime,
                        "feature_layer": "all",
                        "model": "all",
                        "failure": f"pool_preparation: {type(exc).__name__}: {exc}",
                    }
                )
                continue
            for feature_layer in FEATURE_LAYERS:
                for model_name in MODEL_NAMES:
                    try:
                        probability = _fit_predict_probability(
                            train,
                            fold.test,
                            feature_layer=feature_layer,
                            model_name=model_name,
                        )
                        fold_rows.append(
                            _prediction_frame(
                                fold,
                                probability,
                                train,
                                regime=regime,
                                feature_layer=feature_layer,
                                model=model_name,
                                role=role,
                                max_internal=max_internal,
                                max_any=max_any,
                            )
                        )
                    except Exception as exc:
                        fold_failures.append(
                            {
                                "domain": "herg_binary",
                                "evaluation": fold.evaluation,
                                "fold": fold.fold,
                                "data_regime": regime,
                                "feature_layer": feature_layer,
                                "model": model_name,
                                "failure": f"{type(exc).__name__}: {exc}",
                            }
                        )
            try:
                prior = float(
                    np.average(
                        train["target_class"],
                        weights=train["sample_weight"],
                    )
                )
                for feature_layer, model_name, probability in (
                    ("none", "train_prior", np.full(len(fold.test), prior)),
                    (
                        "morgan_tanimoto",
                        "tanimoto_3nn",
                        _tanimoto_probability(train, fold.test),
                    ),
                ):
                    fold_rows.append(
                        _prediction_frame(
                            fold,
                            probability,
                            train,
                            regime=regime,
                            feature_layer=feature_layer,
                            model=model_name,
                            role=role,
                            max_internal=max_internal,
                            max_any=max_any,
                        )
                    )
            except Exception as exc:
                fold_failures.append(
                    {
                        "domain": "herg_binary",
                        "evaluation": fold.evaluation,
                        "fold": fold.fold,
                        "data_regime": regime,
                        "feature_layer": "fixed_comparator",
                        "model": "train_prior_or_tanimoto_3nn",
                        "failure": f"{type(exc).__name__}: {exc}",
                    }
                )
        checkpoint = pd.concat(fold_rows, ignore_index=True) if fold_rows else pd.DataFrame()
        checkpoint["checkpoint_version"] = CHECKPOINT_VERSION
        atomic_write_parquet(prediction_path, checkpoint)
        atomic_write_csv(
            failure_path,
            pd.DataFrame(
                fold_failures,
                columns=[
                    "domain",
                    "evaluation",
                    "fold",
                    "data_regime",
                    "feature_layer",
                    "model",
                    "failure",
                ],
            ),
        )
        rows.append(checkpoint.drop(columns="checkpoint_version"))
        failures.extend(fold_failures)
    return pd.concat(rows, ignore_index=True), pd.DataFrame(failures)


def _ece(observed: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    indices = np.clip(np.digitize(probability, edges[1:-1]), 0, bins - 1)
    value = 0.0
    for index in range(bins):
        mask = indices == index
        if mask.any():
            value += float(mask.mean()) * abs(float(probability[mask].mean()) - float(observed[mask].mean()))
    return value


def _metrics(frame: pd.DataFrame) -> dict[str, float]:
    observed = frame["observed_class"].to_numpy(dtype=int)
    probability = frame["predicted_blocker_probability"].to_numpy(dtype=float)
    called = (probability >= 0.5).astype(int)
    both = np.unique(observed).size == 2
    return {
        "n": float(len(frame)),
        "n_scaffolds": float(frame["scaffold"].nunique()),
        "n_blockers": float(observed.sum()),
        "n_nonblockers": float((observed == 0).sum()),
        "blocker_prevalence": float(observed.mean()),
        "roc_auc": float(roc_auc_score(observed, probability)) if both else float("nan"),
        "pr_auc": float(average_precision_score(observed, probability)) if both else float("nan"),
        "balanced_accuracy": float(balanced_accuracy_score(observed, called)) if both else float("nan"),
        "mcc": float(matthews_corrcoef(observed, called)) if both else float("nan"),
        "sensitivity": float(np.mean(called[observed == 1] == 1)) if np.any(observed == 1) else float("nan"),
        "specificity": float(np.mean(called[observed == 0] == 0)) if np.any(observed == 0) else float("nan"),
        "brier": float(np.mean(np.square(probability - observed))),
        "log_loss": float(log_loss(observed, probability, labels=[0, 1])),
        "ece_10bin": _ece(observed, probability),
        "median_max_internal_like_tanimoto": float(frame["max_internal_like_train_tanimoto"].median()),
    }


def _bootstrap_metrics(frame: pd.DataFrame, *, seed: int) -> dict[str, float]:
    point = _metrics(frame)
    work = frame.copy()
    work["_positive"] = work["observed_class"].eq(1).astype(int)
    work["_negative"] = work["observed_class"].eq(0).astype(int)
    work["_tp"] = (work["observed_class"].eq(1) & work["predicted_class"].eq(1)).astype(int)
    work["_tn"] = (work["observed_class"].eq(0) & work["predicted_class"].eq(0)).astype(int)
    grouped = (
        work.groupby(work["scaffold"].astype(str), sort=True)
        .agg(
            count=("brier_component", "size"),
            brier_sum=("brier_component", "sum"),
            positive=("_positive", "sum"),
            negative=("_negative", "sum"),
            tp=("_tp", "sum"),
            tn=("_tn", "sum"),
        )
        .reset_index(drop=True)
    )
    if len(grouped) < 2:
        return point
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(grouped), size=(BOOTSTRAPS, len(grouped)))
    count = grouped["count"].to_numpy(dtype=float)[sampled].sum(axis=1)
    brier = grouped["brier_sum"].to_numpy(dtype=float)[sampled].sum(axis=1) / count
    positive = grouped["positive"].to_numpy(dtype=float)[sampled].sum(axis=1)
    negative = grouped["negative"].to_numpy(dtype=float)[sampled].sum(axis=1)
    sensitivity = np.divide(
        grouped["tp"].to_numpy(dtype=float)[sampled].sum(axis=1),
        positive,
        out=np.full(BOOTSTRAPS, np.nan),
        where=positive > 0,
    )
    specificity = np.divide(
        grouped["tn"].to_numpy(dtype=float)[sampled].sum(axis=1),
        negative,
        out=np.full(BOOTSTRAPS, np.nan),
        where=negative > 0,
    )
    balanced = (sensitivity + specificity) / 2.0
    for metric, values in (("brier", brier), ("balanced_accuracy", balanced)):
        finite = values[np.isfinite(values)]
        point[f"{metric}_lower_95"] = float(np.quantile(finite, 0.025))
        point[f"{metric}_upper_95"] = float(np.quantile(finite, 0.975))
    return point


def _summarize(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    columns = ["evaluation", "data_regime", "feature_layer", "model"]
    for index, (key, frame) in enumerate(predictions.groupby(columns, sort=True)):
        rows.append(
            {
                **dict(zip(columns, key, strict=True)),
                "evidence_status": " | ".join(sorted(frame["evidence_status"].unique())),
                "external_data_role": " | ".join(sorted(frame["external_data_role"].unique())),
                "training_structures_median": float(frame["training_structures"].median()),
                "training_public_structures_median": float(frame["training_public_structures"].median()),
                **_bootstrap_metrics(frame, seed=SEED + index),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["evaluation", "brier", "feature_layer", "model"])
        .reset_index(drop=True)
    )


def _paired_external_gain(predictions: pd.DataFrame) -> pd.DataFrame:
    baseline = predictions[predictions["data_regime"].eq("internal_only")][
        [
            "evaluation",
            "record_id",
            "scaffold",
            "feature_layer",
            "model",
            "observed_class",
            "predicted_class",
            "brier_component",
        ]
    ].rename(
        columns={
            "predicted_class": "baseline_call",
            "brier_component": "baseline_brier",
        }
    )
    candidate = predictions[~predictions["data_regime"].eq("internal_only")].merge(
        baseline,
        on=[
            "evaluation",
            "record_id",
            "scaffold",
            "feature_layer",
            "model",
            "observed_class",
        ],
        how="inner",
        validate="many_to_one",
    )
    rows: list[dict[str, Any]] = []
    columns = ["evaluation", "data_regime", "feature_layer", "model"]
    for index, (key, frame) in enumerate(candidate.groupby(columns, sort=True)):
        work = frame.copy()
        work["_brier_delta"] = work["brier_component"] - work["baseline_brier"]
        work["_positive"] = work["observed_class"].eq(1).astype(int)
        work["_negative"] = work["observed_class"].eq(0).astype(int)
        work["_candidate_tp"] = (work["observed_class"].eq(1) & work["predicted_class"].eq(1)).astype(int)
        work["_candidate_tn"] = (work["observed_class"].eq(0) & work["predicted_class"].eq(0)).astype(int)
        work["_baseline_tp"] = (work["observed_class"].eq(1) & work["baseline_call"].eq(1)).astype(int)
        work["_baseline_tn"] = (work["observed_class"].eq(0) & work["baseline_call"].eq(0)).astype(int)
        grouped = (
            work.groupby(work["scaffold"].astype(str), sort=True)
            .agg(
                count=("_brier_delta", "size"),
                brier_delta_sum=("_brier_delta", "sum"),
                positive=("_positive", "sum"),
                negative=("_negative", "sum"),
                candidate_tp=("_candidate_tp", "sum"),
                candidate_tn=("_candidate_tn", "sum"),
                baseline_tp=("_baseline_tp", "sum"),
                baseline_tn=("_baseline_tn", "sum"),
            )
            .reset_index(drop=True)
        )
        brier_delta_samples = np.array([], dtype=float)
        balanced_delta_samples = np.array([], dtype=float)
        if len(grouped) >= 2:
            rng = np.random.default_rng(SEED + 10000 + index)
            sampled = rng.integers(0, len(grouped), size=(BOOTSTRAPS, len(grouped)))
            counts = grouped["count"].to_numpy(dtype=float)[sampled].sum(axis=1)
            brier_delta_samples = (
                grouped["brier_delta_sum"].to_numpy(dtype=float)[sampled].sum(axis=1) / counts
            )
            positive = grouped["positive"].to_numpy(dtype=float)[sampled].sum(axis=1)
            negative = grouped["negative"].to_numpy(dtype=float)[sampled].sum(axis=1)
            candidate_sensitivity = np.divide(
                grouped["candidate_tp"].to_numpy(dtype=float)[sampled].sum(axis=1),
                positive,
                out=np.full(BOOTSTRAPS, np.nan),
                where=positive > 0,
            )
            candidate_specificity = np.divide(
                grouped["candidate_tn"].to_numpy(dtype=float)[sampled].sum(axis=1),
                negative,
                out=np.full(BOOTSTRAPS, np.nan),
                where=negative > 0,
            )
            baseline_sensitivity = np.divide(
                grouped["baseline_tp"].to_numpy(dtype=float)[sampled].sum(axis=1),
                positive,
                out=np.full(BOOTSTRAPS, np.nan),
                where=positive > 0,
            )
            baseline_specificity = np.divide(
                grouped["baseline_tn"].to_numpy(dtype=float)[sampled].sum(axis=1),
                negative,
                out=np.full(BOOTSTRAPS, np.nan),
                where=negative > 0,
            )
            candidate_balanced = 0.5 * (candidate_sensitivity + candidate_specificity)
            baseline_balanced = 0.5 * (baseline_sensitivity + baseline_specificity)
            balanced_delta_samples = candidate_balanced - baseline_balanced
        observed = work["observed_class"].to_numpy(dtype=int)
        candidate_call = work["predicted_class"].to_numpy(dtype=int)
        baseline_call = work["baseline_call"].to_numpy(dtype=int)
        candidate_balanced_point = balanced_accuracy_score(observed, candidate_call)
        baseline_balanced_point = balanced_accuracy_score(observed, baseline_call)
        rows.append(
            {
                **dict(zip(columns, key, strict=True)),
                "n": int(len(work)),
                "n_scaffolds": int(len(grouped)),
                "external_minus_internal_brier": float(work["_brier_delta"].mean()),
                "brier_delta_lower_95": (
                    float(np.quantile(brier_delta_samples, 0.025))
                    if len(brier_delta_samples)
                    else float("nan")
                ),
                "brier_delta_upper_95": (
                    float(np.quantile(brier_delta_samples, 0.975))
                    if len(brier_delta_samples)
                    else float("nan")
                ),
                "bootstrap_probability_external_brier_better": (
                    float(np.mean(brier_delta_samples < 0)) if len(brier_delta_samples) else float("nan")
                ),
                "external_minus_internal_balanced_accuracy": float(
                    candidate_balanced_point - baseline_balanced_point
                ),
                "balanced_delta_lower_95": (
                    float(np.nanquantile(balanced_delta_samples, 0.025))
                    if len(balanced_delta_samples)
                    else float("nan")
                ),
                "balanced_delta_upper_95": (
                    float(np.nanquantile(balanced_delta_samples, 0.975))
                    if len(balanced_delta_samples)
                    else float("nan")
                ),
            }
        )
    return (
        pd.DataFrame(rows).sort_values(["evaluation", "external_minus_internal_brier"]).reset_index(drop=True)
    )


def _figures(summary: pd.DataFrame, output: Path) -> None:
    panel = summary[
        summary["evaluation"].isin(["internal_scaffold_cv", "angelo_fixed_nonoverlap"])
        & summary["feature_layer"].isin(FEATURE_LAYERS)
        & summary["model"].isin(MODEL_NAMES)
    ].copy()
    for evaluation in panel["evaluation"].unique():
        selected = panel[panel["evaluation"].eq(evaluation)].nsmallest(16, "brier")
        labels = (
            selected["data_regime"].str.replace("internal_plus_", "I+")
            + " | "
            + selected["feature_layer"].str.replace("_", " ")
            + " | "
            + selected["model"].str.replace("_", " ")
        )
        figure, axis = plt.subplots(figsize=(11, 7))
        order = np.arange(len(selected))[::-1]
        axis.barh(order, selected["brier"], color="#386641")
        axis.set_yticks(order, labels)
        axis.set_xlabel("Brier score (lower is better)")
        axis.set_title(f"{evaluation}: interval-decisive hERG classification")
        axis.grid(axis="x", alpha=0.2)
        figure.tight_layout()
        base._atomic_figure(
            figure,
            output / f"{evaluation}_binary_model_comparison.png",
            dpi=220,
        )
        base._atomic_figure(
            figure,
            output / f"{evaluation}_binary_model_comparison.pdf",
        )
        plt.close(figure)


def run(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    _, internal_intervals, tables = base._internal_herg_frames()
    internal, internal_audit = _decisive_interval_frame(
        internal_intervals,
        dataset_role="internal_interval_decisive",
    )
    internal_structure_ids = set(tables["compounds"]["structure_id"].astype(str))
    _, extension_intervals = base._extension_frames(internal_structure_ids)
    extension, extension_audit = _decisive_interval_frame(
        extension_intervals,
        dataset_role="angelo_same_series_interval_decisive",
    )
    excluded = internal_structure_ids | set(extension_intervals["structure_id"].astype(str))
    public_train, public_validation = _public_classification_frames(excluded)
    predictions, failures = _run_matrix(
        internal,
        extension,
        public_train,
        public_validation,
        output,
    )
    summary = _summarize(predictions)
    gain = _paired_external_gain(predictions)
    audit = pd.concat([internal_audit, extension_audit], ignore_index=True)

    atomic_write_parquet(output / "herg_binary_predictions.parquet", predictions)
    atomic_write_csv(output / "herg_binary_summary.csv", summary)
    atomic_write_csv(output / "herg_binary_external_gain.csv", gain)
    atomic_write_csv(output / "herg_binary_failure_ledger.csv", failures)
    atomic_write_csv(output / "herg_binary_label_audit.csv", audit)
    _figures(summary, output)
    payload = {
        "status": "completed",
        "prediction_rows": int(len(predictions)),
        "experiment_summaries": int(len(summary)),
        "fit_failures": int(len(failures)),
        "internal_decisive_structures": int(len(internal)),
        "internal_blockers": int(internal["target_class"].sum()),
        "internal_nonblockers": int(internal["target_class"].eq(0).sum()),
        "extension_decisive_structures": int(len(extension)),
        "extension_blockers": int(extension["target_class"].sum()),
        "extension_nonblockers": int(extension["target_class"].eq(0).sum()),
        "public_train_structures": int(len(public_train)),
        "public_validation_structures": int(len(public_validation)),
        "intermediate_or_conflicting_rows_excluded": int(audit["target_class"].isna().sum()),
        "decision_boundary": (
            "blocker only when interval lower bound >= pIC50 5; nonblocker only "
            "when interval upper bound <= pIC50 4.522879; intermediate excluded"
        ),
    }
    atomic_write_json(output / "herg_binary_run_summary.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2))


if __name__ == "__main__":
    main()
