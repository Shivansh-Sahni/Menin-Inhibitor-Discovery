#!/usr/bin/env python3
"""Run leakage-aware hERG/PK dataset-feature-model mix-and-match experiments.

The analysis has two purposes:

1. quantify when external hERG evidence helps the internal Menin-inhibitor
   program and when it dilutes or miscalibrates the internal signal; and
2. compare compact process-motivated molecular proxies with associative
   fingerprints and hybrids without promoting unavailable physics quantities.

The primary hERG endpoint is continuous pIC50. Binary liability summaries are
derived only for interval-decisive compounds (IC50 <=10 uM or >=30 uM). Public
hERG data are never called protocol-matched validation. The Angelo/Ascentage
set is explicitly labeled a retrospective, same-series extension.

PK remains internal-only because the currently curated external PK table has
too few protocol-compatible rat IV/PO structures for a defensible transfer
fit. The rejected external merge is retained as a quantitative compatibility
audit rather than silently omitted.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/menin-mix-match-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/menin-mix-match-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from scipy.stats import spearmanr
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "pipeline/src"))

from menin_discovery.research_common import (  # noqa: E402
    atomic_write_csv,
    atomic_write_json,
    atomic_write_parquet,
    atomic_write_text,
)
from menin_discovery.research_modeling import (  # noqa: E402
    merge_feature_layers,
    structure_feature_frame,
)
from menin_discovery.research_public_herg import prepare_sun_source_holdout  # noqa: E402
from menin_discovery.research_reviewer_audit import (  # noqa: E402
    _source_collapsed_pk_frames,
)
from menin_discovery.research_workflows import (  # noqa: E402
    compound_model_frame,
    load_canonical_tables,
    prepare_herg_evidence,
)

SEED = 20260730
BOOTSTRAPS = 1200
MORGAN_BITS = 2048
MORGAN_RADIUS = 2
MORGAN_COMPONENTS = 8
PUBLIC_NEAREST_MULTIPLIER = 5
PIC50_BLOCKER = 5.0
PIC50_NONBLOCKER = 6.0 - math.log10(30.0)

CANONICAL = ROOT / "research/data/pk_herg/canonical"
EXTENSION_PATH = CANONICAL / "ascentage_herg_2026_07_28/normalized_records.parquet"
PUBLIC_ROOT = CANONICAL / "public_herg"
DEFAULT_OUTPUT = ROOT / "research/reports/pk_herg/mix_match"
CHECKPOINT_VERSION = "2026-07-30-v1"

PROXY_COLUMNS = [
    "mol_wt",
    "logp",
    "tpsa",
    "h_bond_donors",
    "h_bond_acceptors",
    "rotatable_bonds",
    "ring_count",
    "aromatic_ring_count",
    "fraction_csp3",
]

FEATURE_LAYERS = ("compact_proxies", "morgan_latent", "hybrid")
MODEL_NAMES = ("ridge", "svr", "random_forest", "extra_trees")
RESIDUAL_MODEL_NAMES = ("ridge", "extra_trees")
RESIDUAL_FEATURE_LAYERS = ("compact_proxies", "hybrid")


def _atomic_figure(figure: plt.Figure, path: Path, **kwargs: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.stem}.tmp{path.suffix}")
    figure.savefig(temporary, **kwargs)
    os.replace(temporary, path)


def _safe_spearman(observed: np.ndarray, predicted: np.ndarray) -> float:
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    finite = np.isfinite(observed) & np.isfinite(predicted)
    if finite.sum() < 3 or np.unique(observed[finite]).size < 2 or np.unique(predicted[finite]).size < 2:
        return float("nan")
    return float(spearmanr(observed[finite], predicted[finite]).statistic)


def _fingerprints(smiles: Iterable[str]) -> np.ndarray:
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=MORGAN_RADIUS,
        fpSize=MORGAN_BITS,
    )
    rows: list[np.ndarray] = []
    for value in smiles:
        molecule = Chem.MolFromSmiles(str(value))
        if molecule is None:
            raise ValueError(f"Invalid standardized SMILES in mix-and-match input: {value}")
        rows.append(generator.GetFingerprintAsNumPy(molecule).astype(np.float32))
    return np.asarray(rows, dtype=np.float32)


def _attach_features(frame: pd.DataFrame) -> pd.DataFrame:
    compounds = pd.DataFrame(
        {
            "compound_id": frame["record_id"].astype(str),
            "standardized_smiles": frame["standardized_smiles"].astype(str),
        }
    )
    descriptors = structure_feature_frame(compounds)
    descriptors = descriptors.rename(columns={"compound_id": "record_id"})
    result = frame.merge(
        descriptors[["record_id", *PROXY_COLUMNS]],
        on="record_id",
        how="inner",
        validate="one_to_one",
    )
    result["fingerprint"] = list(_fingerprints(result["standardized_smiles"]))
    return result


def _aggregate_internal_pka(measurements: pd.DataFrame) -> pd.DataFrame:
    basic = measurements[
        measurements["endpoint"].eq("basic_pka")
        & pd.to_numeric(measurements["value"], errors="coerce").notna()
    ].copy()
    basic["value"] = pd.to_numeric(basic["value"], errors="coerce")
    summary = basic.groupby("compound_id", as_index=False).agg(
        maximum_basic_pka=("value", "max"),
        reported_basic_pka_count=("value", "size"),
    )
    # This is one declared proxy, not a new independent parameter. It converts
    # the strongest reported basic pKa into a single-site population estimate
    # for an assay-pH sensitivity ablation.
    summary["single_site_cation_fraction_pH7p4_proxy"] = 1.0 / (
        1.0 + np.power(10.0, 7.4 - summary["maximum_basic_pka"])
    )
    return summary


def _internal_herg_frames() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    tables = load_canonical_tables(CANONICAL)
    compounds = compound_model_frame(tables["compounds"], tables.get("compound_aliases"))
    features, layers = merge_feature_layers(compounds)
    _, potency, _ = prepare_herg_evidence(
        compounds,
        tables["measurements"],
        features,
    )
    controls = [
        column for column in layers["structure_2d"] if column in potency and potency[column].notna().any()
    ]
    if controls != PROXY_COLUMNS:
        raise ValueError(f"Compact proxy contract changed: expected {PROXY_COLUMNS}; observed {controls}")

    exact_mask = (
        np.isfinite(potency["pic50_lower"])
        & np.isfinite(potency["pic50_upper"])
        & np.isclose(potency["pic50_lower"], potency["pic50_upper"])
    )
    exact_rows = potency.loc[exact_mask].copy()
    exact = exact_rows.groupby("compound_id", as_index=False).agg(
        target_pic50=("pic50_lower", "median"),
        exact_measurement_count=("pic50_lower", "size"),
        exact_measurement_min=("pic50_lower", "min"),
        exact_measurement_max=("pic50_lower", "max"),
        standardized_smiles=("standardized_smiles", "first"),
        scaffold=("scaffold", "first"),
    )
    exact["measurement_spread_log"] = exact["exact_measurement_max"] - exact["exact_measurement_min"]
    exact["record_id"] = "internal:" + exact["compound_id"].astype(str)
    exact["source_group"] = "internal"
    exact["dataset_role"] = "internal_original_exact_compound_collapsed"

    interval_rows: list[dict[str, Any]] = []
    for compound_id, group in potency.groupby("compound_id", sort=True):
        exact_group = group[
            np.isfinite(group["pic50_lower"])
            & np.isfinite(group["pic50_upper"])
            & np.isclose(group["pic50_lower"], group["pic50_upper"])
        ]
        if len(exact_group):
            lower = upper = float(exact_group["pic50_lower"].median())
            conflict = bool(
                (
                    np.isfinite(group["pic50_lower"]) & (group["pic50_lower"].astype(float) > upper + 1e-9)
                ).any()
                or (
                    np.isfinite(group["pic50_upper"]) & (group["pic50_upper"].astype(float) < lower - 1e-9)
                ).any()
            )
            role = "exact_median_with_source_conflict_flag" if conflict else "exact_median"
        else:
            finite_lower = (
                pd.to_numeric(group["pic50_lower"], errors="coerce")
                .replace([-np.inf, np.inf], np.nan)
                .dropna()
            )
            finite_upper = (
                pd.to_numeric(group["pic50_upper"], errors="coerce")
                .replace([-np.inf, np.inf], np.nan)
                .dropna()
            )
            lower = float(finite_lower.max()) if len(finite_lower) else -np.inf
            upper = float(finite_upper.min()) if len(finite_upper) else np.inf
            conflict = bool(lower > upper)
            role = "censored_interval_intersection" if not conflict else "quarantined_interval_conflict"
        first = group.iloc[0]
        interval_rows.append(
            {
                "record_id": f"internal:{compound_id}",
                "compound_id": compound_id,
                "standardized_smiles": first["standardized_smiles"],
                "scaffold": first["scaffold"],
                "pic50_lower": lower,
                "pic50_upper": upper,
                "source_conflict": conflict,
                "collapse_role": role,
                "source_group": "internal",
            }
        )
    intervals = pd.DataFrame(interval_rows)
    pka = _aggregate_internal_pka(tables["measurements"])
    exact = exact.merge(pka, on="compound_id", how="left", validate="one_to_one")
    return _attach_features(exact), intervals, tables


def _extension_frames(internal_structure_ids: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = pd.read_parquet(EXTENSION_PATH)
    novel = source[
        ~source["structure_id"].astype(str).isin(internal_structure_ids)
        & source["synthesis_status"].eq("synthesized_by_cro")
        & source["herg_ic50_censoring"].ne("missing")
    ].copy()
    novel["record_id"] = "extension:" + novel["structure_id"].astype(str)
    novel["compound_id"] = novel["structure_id"].astype(str)
    novel["pic50_lower"] = pd.to_numeric(novel["herg_pic50_lower_bound"], errors="coerce")
    novel["pic50_upper"] = pd.to_numeric(novel["herg_pic50_upper_bound"], errors="coerce")
    novel["source_group"] = "angelo_same_series_extension"
    novel["dataset_role"] = "retrospective_same_series_nonoverlap"
    exact_mask = (
        np.isfinite(novel["pic50_lower"])
        & np.isfinite(novel["pic50_upper"])
        & np.isclose(novel["pic50_lower"], novel["pic50_upper"])
    )
    exact = novel.loc[exact_mask].copy()
    exact["target_pic50"] = exact["pic50_lower"]
    keep = [
        "record_id",
        "compound_id",
        "structure_id",
        "standardized_smiles",
        "scaffold",
        "target_pic50",
        "source_group",
        "dataset_role",
    ]
    interval_keep = [
        "record_id",
        "compound_id",
        "structure_id",
        "standardized_smiles",
        "scaffold",
        "pic50_lower",
        "pic50_upper",
        "source_group",
        "dataset_role",
    ]
    return _attach_features(exact[keep]), novel[interval_keep].reset_index(drop=True)


def _public_frames(excluded_structure_ids: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    prepared = prepare_sun_source_holdout(PUBLIC_ROOT)
    train = prepared["regression_train"].copy()
    validation = prepared["regression_validation"].copy()
    for frame, role in (
        (train, "public_regression_train"),
        (validation, "public_source_held_out"),
    ):
        frame.rename(
            columns={
                "structure_id": "compound_id",
                "raw_smiles": "standardized_smiles",
                "pic50_value": "target_pic50",
            },
            inplace=True,
        )
        frame["record_id"] = "public:" + frame["compound_id"].astype(str)
        frame["source_group"] = role
        frame["dataset_role"] = role
        frame["structure_id"] = frame["compound_id"].astype(str)
    train = train[~train["structure_id"].isin(excluded_structure_ids)].copy()
    validation = validation[~validation["structure_id"].isin(excluded_structure_ids)].copy()
    keep = [
        "record_id",
        "compound_id",
        "structure_id",
        "standardized_smiles",
        "scaffold",
        "target_pic50",
        "computed_mw_g_mol",
        "source_group",
        "dataset_role",
    ]
    return _attach_features(train[keep]), _attach_features(validation[keep])


@dataclass
class FeatureTransformer:
    layer: str
    components: int = MORGAN_COMPONENTS
    imputer: SimpleImputer | None = None
    scaler: StandardScaler | None = None
    svd: TruncatedSVD | None = None

    def _raw(self, frame: pd.DataFrame, *, fit: bool) -> np.ndarray:
        blocks: list[np.ndarray] = []
        if self.layer in {"compact_proxies", "hybrid", "hybrid_pka"}:
            blocks.append(frame[PROXY_COLUMNS].to_numpy(dtype=float))
        if self.layer in {"morgan_latent", "hybrid", "hybrid_pka"}:
            bits = np.vstack(frame["fingerprint"].to_numpy()).astype(np.float32)
            if fit:
                n_components = max(1, min(self.components, len(frame) - 1, bits.shape[1] - 1))
                self.svd = TruncatedSVD(n_components=n_components, random_state=SEED)
                latent = self.svd.fit_transform(bits)
            else:
                if self.svd is None:
                    raise RuntimeError("Feature transformer SVD is not fitted")
                latent = self.svd.transform(bits)
            blocks.append(latent)
        if self.layer == "hybrid_pka":
            blocks.append(frame[["single_site_cation_fraction_pH7p4_proxy"]].to_numpy(dtype=float))
        if not blocks:
            raise ValueError(f"Unknown feature layer: {self.layer}")
        return np.column_stack(blocks)

    def fit_transform(self, frame: pd.DataFrame) -> np.ndarray:
        raw = self._raw(frame, fit=True)
        self.imputer = SimpleImputer()
        self.scaler = StandardScaler()
        return self.scaler.fit_transform(self.imputer.fit_transform(raw))

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if self.imputer is None or self.scaler is None:
            raise RuntimeError("Feature transformer is not fitted")
        raw = self._raw(frame, fit=False)
        return self.scaler.transform(self.imputer.transform(raw))


def _model(name: str) -> Any:
    if name == "ridge":
        return Ridge(alpha=10.0)
    if name == "svr":
        return SVR(C=3.0, epsilon=0.15, gamma="scale")
    if name == "random_forest":
        return RandomForestRegressor(
            n_estimators=240,
            min_samples_leaf=3,
            max_features="sqrt",
            n_jobs=-1,
            random_state=SEED,
        )
    if name == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=240,
            min_samples_leaf=3,
            max_features="sqrt",
            n_jobs=-1,
            random_state=SEED,
        )
    raise ValueError(f"Unknown model: {name}")


def _fit_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    feature_layer: str,
    model_name: str,
) -> np.ndarray:
    transformer = FeatureTransformer(feature_layer)
    train_x = transformer.fit_transform(train)
    test_x = transformer.transform(test)
    estimator = _model(model_name)
    fit_kwargs: dict[str, Any] = {}
    if "sample_weight" in train and not np.allclose(train["sample_weight"], 1.0):
        fit_kwargs["sample_weight"] = train["sample_weight"].to_numpy(dtype=float)
    estimator.fit(train_x, train["target_pic50"].to_numpy(dtype=float), **fit_kwargs)
    return np.asarray(estimator.predict(test_x), dtype=float)


def _tanimoto_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    neighbors: int = 3,
) -> np.ndarray:
    train_bits = np.vstack(train["fingerprint"].to_numpy()).astype(bool)
    test_bits = np.vstack(test["fingerprint"].to_numpy()).astype(bool)
    train_y = train["target_pic50"].to_numpy(dtype=float)
    predictions = np.empty(len(test), dtype=float)
    for index, query in enumerate(test_bits):
        intersections = np.logical_and(train_bits, query).sum(axis=1)
        unions = np.logical_or(train_bits, query).sum(axis=1)
        similarity = np.divide(
            intersections,
            unions,
            out=np.zeros_like(intersections, dtype=float),
            where=unions > 0,
        )
        count = min(neighbors, len(train))
        nearest = np.argpartition(-similarity, kth=count - 1)[:count]
        weight = np.maximum(similarity[nearest], 1e-6)
        predictions[index] = float(np.sum(weight * train_y[nearest]) / np.sum(weight))
    return predictions


def _max_tanimoto(query: pd.DataFrame, reference: pd.DataFrame) -> np.ndarray:
    if query.empty or reference.empty:
        return np.full(len(query), np.nan)
    reference_fps: list[Any] = []
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=MORGAN_RADIUS,
        fpSize=MORGAN_BITS,
    )
    for smiles in reference["standardized_smiles"].astype(str):
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise ValueError(f"Invalid reference SMILES: {smiles}")
        reference_fps.append(generator.GetFingerprint(molecule))
    result = np.empty(len(query), dtype=float)
    for index, smiles in enumerate(query["standardized_smiles"].astype(str)):
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise ValueError(f"Invalid query SMILES: {smiles}")
        fingerprint = generator.GetFingerprint(molecule)
        result[index] = max(DataStructs.BulkTanimotoSimilarity(fingerprint, reference_fps))
    return result


def _select_nearest_public(
    public: pd.DataFrame,
    internal_like: pd.DataFrame,
    *,
    multiplier: int = PUBLIC_NEAREST_MULTIPLIER,
) -> pd.DataFrame:
    if public.empty:
        return public.copy()
    work = public.copy()
    work["_internal_similarity"] = _max_tanimoto(work, internal_like)
    count = min(len(work), max(len(internal_like) * multiplier, len(internal_like)))
    return work.nlargest(count, "_internal_similarity").drop(columns="_internal_similarity")


def _source_balanced_weights(
    frame: pd.DataFrame,
    *,
    internal_sources: set[str],
) -> pd.Series:
    internal_mask = frame["source_group"].isin(internal_sources)
    internal_n = int(internal_mask.sum())
    public_n = int((~internal_mask).sum())
    weights = pd.Series(np.ones(len(frame), dtype=float), index=frame.index)
    if internal_n and public_n:
        weights.loc[~internal_mask] = internal_n / public_n
    return weights


@dataclass(frozen=True)
class Fold:
    evaluation: str
    fold: int
    base_train: pd.DataFrame
    extension_train: pd.DataFrame
    test: pd.DataFrame
    evidence_status: str


def _group_folds(
    frame: pd.DataFrame,
    *,
    folds: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    unique = frame["scaffold"].astype(str).nunique()
    n_splits = min(folds, unique)
    if n_splits < 2:
        raise ValueError("Need at least two scaffold groups")
    return list(
        GroupKFold(n_splits=n_splits).split(
            frame,
            groups=frame["scaffold"].astype(str),
        )
    )


def _evaluation_folds(
    internal: pd.DataFrame,
    extension: pd.DataFrame,
    public_validation: pd.DataFrame,
) -> list[Fold]:
    folds: list[Fold] = []
    for fold, (train_index, test_index) in enumerate(_group_folds(internal, folds=5)):
        test = internal.iloc[test_index].reset_index(drop=True)
        test_scaffolds = set(test["scaffold"].astype(str))
        folds.append(
            Fold(
                evaluation="internal_scaffold_cv",
                fold=fold,
                base_train=internal.iloc[train_index].reset_index(drop=True),
                extension_train=extension[
                    ~extension["scaffold"].astype(str).isin(test_scaffolds)
                ].reset_index(drop=True),
                test=test,
                evidence_status="retrospective_internal_group_held_out",
            )
        )
    folds.append(
        Fold(
            evaluation="angelo_fixed_nonoverlap",
            fold=0,
            base_train=internal.reset_index(drop=True),
            extension_train=extension.iloc[0:0].copy(),
            test=extension.reset_index(drop=True),
            evidence_status=(
                "retrospective_protocol_unmatched_same_series_extension; outcomes previously inspected"
            ),
        )
    )
    for fold, (train_index, test_index) in enumerate(_group_folds(extension, folds=5)):
        test = extension.iloc[test_index].reset_index(drop=True)
        test_scaffolds = set(test["scaffold"].astype(str))
        folds.append(
            Fold(
                evaluation="angelo_augmented_scaffold_cv",
                fold=fold,
                base_train=internal[~internal["scaffold"].astype(str).isin(test_scaffolds)].reset_index(
                    drop=True
                ),
                extension_train=extension.iloc[train_index][
                    ~extension.iloc[train_index]["scaffold"].astype(str).isin(test_scaffolds)
                ].reset_index(drop=True),
                test=test,
                evidence_status=("retrospective_same_series_group_held_out_after_extension_augmentation"),
            )
        )
    folds.append(
        Fold(
            evaluation="public_source_holdout",
            fold=0,
            base_train=internal.reset_index(drop=True),
            extension_train=extension.reset_index(drop=True),
            test=public_validation.reset_index(drop=True),
            evidence_status=("external_source_holdout_but_not_protocol_matched_to_internal_program"),
        )
    )
    return folds


def _regime_pool(
    fold: Fold,
    public_train: pd.DataFrame,
    regime: str,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    test_scaffolds = set(fold.test["scaffold"].astype(str))
    public = public_train[~public_train["scaffold"].astype(str).isin(test_scaffolds)].reset_index(drop=True)
    base = fold.base_train.copy()
    extension = fold.extension_train.copy()
    if regime == "internal_only":
        train = base
        role = "internal_anchor"
    elif regime == "internal_plus_extension":
        train = pd.concat([base, extension], ignore_index=True)
        role = "same_series_augmentation"
    elif regime == "internal_plus_public_naive":
        train = pd.concat([base, public], ignore_index=True)
        role = "negative_control_external_row_dominance"
    elif regime == "internal_plus_public_balanced":
        train = pd.concat([base, public], ignore_index=True)
        train["sample_weight"] = _source_balanced_weights(
            train,
            internal_sources={"internal"},
        )
        role = "source_mass_balanced_public_pool"
    elif regime == "internal_plus_public_nearest":
        selected = _select_nearest_public(public, base)
        train = pd.concat([base, selected], ignore_index=True)
        train["sample_weight"] = _source_balanced_weights(
            train,
            internal_sources={"internal"},
        )
        role = "outcome_blind_nearest_public_pool_source_balanced"
    elif regime == "internal_plus_public_large":
        selected = public[public["computed_mw_g_mol"].ge(650)].copy()
        train = pd.concat([base, selected], ignore_index=True)
        train["sample_weight"] = _source_balanced_weights(
            train,
            internal_sources={"internal"},
        )
        role = "mw_matched_but_chemically_distant_public_pool"
    elif regime == "internal_plus_extension_plus_public_balanced":
        train = pd.concat([base, extension, public], ignore_index=True)
        train["sample_weight"] = _source_balanced_weights(
            train,
            internal_sources={"internal", "angelo_same_series_extension"},
        )
        role = "same_series_plus_source_mass_balanced_public_pool"
    elif regime == "public_prior_internal_residual":
        train = pd.concat([base, extension], ignore_index=True)
        role = "public_pretrain_then_internal_like_residual_calibration"
    else:
        raise ValueError(f"Unknown data regime: {regime}")
    if "sample_weight" not in train:
        train["sample_weight"] = 1.0
    return train.reset_index(drop=True), public, role


def _predict_regime(
    fold: Fold,
    public_train: pd.DataFrame,
    *,
    regime: str,
    feature_layer: str,
    model_name: str,
    prepared_pool: tuple[pd.DataFrame, pd.DataFrame, str] | None = None,
) -> tuple[np.ndarray, pd.DataFrame, str]:
    train, public, role = (
        prepared_pool if prepared_pool is not None else _regime_pool(fold, public_train, regime)
    )
    if train.empty:
        raise ValueError("Training pool is empty")
    if regime == "public_prior_internal_residual":
        if model_name not in RESIDUAL_MODEL_NAMES or feature_layer not in RESIDUAL_FEATURE_LAYERS:
            raise ValueError("unsupported_residual_combination")
        prior_prediction_train = _fit_predict(
            public.assign(sample_weight=1.0),
            train,
            feature_layer=feature_layer,
            model_name=model_name,
        )
        prior_prediction_test = _fit_predict(
            public.assign(sample_weight=1.0),
            fold.test,
            feature_layer=feature_layer,
            model_name=model_name,
        )
        residual_train = train.copy()
        residual_train["target_pic50"] = (
            residual_train["target_pic50"].to_numpy(dtype=float) - prior_prediction_train
        )
        residual = _fit_predict(
            residual_train,
            fold.test,
            feature_layer=feature_layer,
            model_name=model_name,
        )
        prediction = prior_prediction_test + residual
    else:
        prediction = _fit_predict(
            train,
            fold.test,
            feature_layer=feature_layer,
            model_name=model_name,
        )
    return prediction, train, role


def _prediction_rows(
    fold: Fold,
    prediction: np.ndarray,
    train: pd.DataFrame,
    *,
    regime: str,
    feature_layer: str,
    model_name: str,
    role: str,
    max_internal: np.ndarray | None = None,
    max_all: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    internal_like = train[
        train["source_group"].isin({"internal", "angelo_same_series_extension"})
    ].drop_duplicates("record_id")
    if max_internal is None:
        max_internal = _max_tanimoto(fold.test, internal_like)
    if max_all is None:
        max_all = _max_tanimoto(fold.test, train.drop_duplicates("record_id"))
    rows: list[dict[str, Any]] = []
    for position, record in enumerate(fold.test.itertuples(index=False)):
        rows.append(
            {
                "evaluation": fold.evaluation,
                "fold": fold.fold,
                "evidence_status": fold.evidence_status,
                "record_id": record.record_id,
                "compound_id": record.compound_id,
                "scaffold": record.scaffold,
                "test_source_group": record.source_group,
                "observed_pic50": float(record.target_pic50),
                "predicted_pic50": float(prediction[position]),
                "residual": float(prediction[position] - record.target_pic50),
                "absolute_error": float(abs(prediction[position] - record.target_pic50)),
                "data_regime": regime,
                "external_data_role": role,
                "feature_layer": feature_layer,
                "model": model_name,
                "training_rows": int(len(train)),
                "training_structures": int(train["record_id"].nunique()),
                "training_internal_like_structures": int(internal_like["record_id"].nunique()),
                "training_public_structures": int(
                    train["source_group"].astype(str).str.startswith("public").sum()
                ),
                "max_internal_like_train_tanimoto": float(max_internal[position]),
                "max_any_train_tanimoto": float(max_all[position]),
            }
        )
    return rows


def _run_herg_matrix(
    internal: pd.DataFrame,
    extension: pd.DataFrame,
    public_train: pd.DataFrame,
    public_validation: pd.DataFrame,
    *,
    checkpoint_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    regimes = (
        "internal_only",
        "internal_plus_extension",
        "internal_plus_public_naive",
        "internal_plus_public_balanced",
        "internal_plus_public_nearest",
        "internal_plus_public_large",
        "internal_plus_extension_plus_public_balanced",
        "public_prior_internal_residual",
    )
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for fold in _evaluation_folds(internal, extension, public_validation):
        checkpoint_stem = f"{fold.evaluation}__fold_{fold.fold}"
        checkpoint_predictions = (
            checkpoint_dir / "checkpoints" / f"{checkpoint_stem}.parquet"
            if checkpoint_dir is not None
            else None
        )
        checkpoint_failures = (
            checkpoint_dir / "checkpoints" / f"{checkpoint_stem}_failures.csv"
            if checkpoint_dir is not None
            else None
        )
        if checkpoint_predictions is not None and checkpoint_predictions.exists():
            cached = pd.read_parquet(checkpoint_predictions)
            if not cached.empty and cached["checkpoint_version"].eq(CHECKPOINT_VERSION).all():
                rows.extend(cached.drop(columns="checkpoint_version").to_dict("records"))
                if checkpoint_failures is not None and checkpoint_failures.exists():
                    try:
                        failure_frame = pd.read_csv(checkpoint_failures)
                    except pd.errors.EmptyDataError:
                        failure_frame = pd.DataFrame()
                    if not failure_frame.empty:
                        failures.extend(failure_frame.to_dict("records"))
                print(f"reused checkpoint {checkpoint_stem}", flush=True)
                continue
        print(f"running {checkpoint_stem}", flush=True)
        fold_rows_start = len(rows)
        fold_failures_start = len(failures)
        for regime in regimes:
            if fold.evaluation == "angelo_fixed_nonoverlap" and "extension" in regime:
                continue
            try:
                prepared_pool = _regime_pool(fold, public_train, regime)
                cached_train, _, cached_role = prepared_pool
                cached_internal_like = cached_train[
                    cached_train["source_group"].isin({"internal", "angelo_same_series_extension"})
                ].drop_duplicates("record_id")
                cached_max_internal = _max_tanimoto(
                    fold.test,
                    cached_internal_like,
                )
                cached_max_all = _max_tanimoto(
                    fold.test,
                    cached_train.drop_duplicates("record_id"),
                )
            except Exception as exc:
                failures.append(
                    {
                        "domain": "herg",
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
                    if regime == "public_prior_internal_residual" and (
                        model_name not in RESIDUAL_MODEL_NAMES or feature_layer not in RESIDUAL_FEATURE_LAYERS
                    ):
                        continue
                    try:
                        prediction, train, role = _predict_regime(
                            fold,
                            public_train,
                            regime=regime,
                            feature_layer=feature_layer,
                            model_name=model_name,
                            prepared_pool=prepared_pool,
                        )
                        rows.extend(
                            _prediction_rows(
                                fold,
                                prediction,
                                train,
                                regime=regime,
                                feature_layer=feature_layer,
                                model_name=model_name,
                                role=role,
                                max_internal=cached_max_internal,
                                max_all=cached_max_all,
                            )
                        )
                    except Exception as exc:  # preserve failed experiments
                        failures.append(
                            {
                                "domain": "herg",
                                "evaluation": fold.evaluation,
                                "fold": fold.fold,
                                "data_regime": regime,
                                "feature_layer": feature_layer,
                                "model": model_name,
                                "failure": f"{type(exc).__name__}: {exc}",
                            }
                        )
        # Fixed comparators are intentionally not crossed with feature layers.
        for regime in regimes:
            if regime == "public_prior_internal_residual":
                continue
            if fold.evaluation == "angelo_fixed_nonoverlap" and "extension" in regime:
                continue
            try:
                train, _, role = _regime_pool(fold, public_train, regime)
                internal_like = train[
                    train["source_group"].isin({"internal", "angelo_same_series_extension"})
                ].drop_duplicates("record_id")
                max_internal = _max_tanimoto(fold.test, internal_like)
                max_all = _max_tanimoto(
                    fold.test,
                    train.drop_duplicates("record_id"),
                )
                mean = np.average(
                    train["target_pic50"].to_numpy(dtype=float),
                    weights=train["sample_weight"].to_numpy(dtype=float),
                )
                rows.extend(
                    _prediction_rows(
                        fold,
                        np.full(len(fold.test), mean),
                        train,
                        regime=regime,
                        feature_layer="none",
                        model_name="train_mean",
                        role=role,
                        max_internal=max_internal,
                        max_all=max_all,
                    )
                )
                rows.extend(
                    _prediction_rows(
                        fold,
                        _tanimoto_predict(train, fold.test, neighbors=3),
                        train,
                        regime=regime,
                        feature_layer="morgan_tanimoto",
                        model_name="tanimoto_3nn",
                        role=role,
                        max_internal=max_internal,
                        max_all=max_all,
                    )
                )
            except Exception as exc:
                failures.append(
                    {
                        "domain": "herg",
                        "evaluation": fold.evaluation,
                        "fold": fold.fold,
                        "data_regime": regime,
                        "feature_layer": "fixed_comparator",
                        "model": "train_mean_or_tanimoto_3nn",
                        "failure": f"{type(exc).__name__}: {exc}",
                    }
                )
        if checkpoint_predictions is not None:
            fold_rows = pd.DataFrame(rows[fold_rows_start:])
            fold_rows["checkpoint_version"] = CHECKPOINT_VERSION
            atomic_write_parquet(checkpoint_predictions, fold_rows)
            atomic_write_csv(
                checkpoint_failures,
                pd.DataFrame(
                    failures[fold_failures_start:],
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
            print(
                f"saved checkpoint {checkpoint_stem}: {len(fold_rows)} predictions",
                flush=True,
            )
    return pd.DataFrame(rows), pd.DataFrame(failures)


def _regression_metrics(frame: pd.DataFrame) -> dict[str, float]:
    observed = frame["observed_pic50"].to_numpy(dtype=float)
    predicted = frame["predicted_pic50"].to_numpy(dtype=float)
    error = predicted - observed
    absolute = np.abs(error)
    metrics: dict[str, float] = {
        "n": float(len(frame)),
        "n_scaffolds": float(frame["scaffold"].nunique()),
        "mae": float(mean_absolute_error(observed, predicted)),
        "rmse": float(math.sqrt(mean_squared_error(observed, predicted))),
        "r2": float(r2_score(observed, predicted)) if len(observed) >= 2 else float("nan"),
        "spearman": _safe_spearman(observed, predicted),
        "mean_signed_error": float(np.mean(error)),
        "fraction_within_0p5_log": float(np.mean(absolute <= 0.5)),
        "fraction_within_1p0_log": float(np.mean(absolute <= 1.0)),
        "median_max_internal_like_tanimoto": float(frame["max_internal_like_train_tanimoto"].median()),
    }
    decisive = (observed >= PIC50_BLOCKER) | (observed <= PIC50_NONBLOCKER)
    if decisive.sum() >= 2:
        labels = (observed[decisive] >= PIC50_BLOCKER).astype(int)
        scores = predicted[decisive]
        calls = (scores >= PIC50_BLOCKER).astype(int)
        both = np.unique(labels).size == 2
        metrics.update(
            {
                "decisive_n": float(decisive.sum()),
                "n_blockers": float(labels.sum()),
                "n_nonblockers": float((labels == 0).sum()),
                "roc_auc": float(roc_auc_score(labels, scores)) if both else float("nan"),
                "pr_auc": float(average_precision_score(labels, scores)) if both else float("nan"),
                "balanced_accuracy": float(balanced_accuracy_score(labels, calls)) if both else float("nan"),
                "mcc": float(matthews_corrcoef(labels, calls)) if both else float("nan"),
                "sensitivity": float(np.mean(calls[labels == 1] == 1))
                if np.any(labels == 1)
                else float("nan"),
                "specificity": float(np.mean(calls[labels == 0] == 0))
                if np.any(labels == 0)
                else float("nan"),
            }
        )
    return metrics


def _bootstrap_metrics(frame: pd.DataFrame, *, seed: int) -> dict[str, float]:
    point = _regression_metrics(frame)
    work = frame.copy()
    work["_abs"] = np.abs(work["predicted_pic50"] - work["observed_pic50"])
    work["_sq"] = np.square(work["predicted_pic50"] - work["observed_pic50"])
    work["_error"] = work["predicted_pic50"] - work["observed_pic50"]
    work["_within"] = work["_abs"].le(0.5).astype(float)
    grouped = (
        work.groupby(work["scaffold"].astype(str), sort=True)
        .agg(
            count=("_abs", "size"),
            absolute_sum=("_abs", "sum"),
            squared_sum=("_sq", "sum"),
            error_sum=("_error", "sum"),
            within_sum=("_within", "sum"),
        )
        .reset_index(drop=True)
    )
    if len(grouped) < 2:
        return point
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(grouped), size=(BOOTSTRAPS, len(grouped)))
    counts = grouped["count"].to_numpy(dtype=float)[sampled].sum(axis=1)
    absolute_sum = grouped["absolute_sum"].to_numpy(dtype=float)[sampled].sum(axis=1)
    squared_sum = grouped["squared_sum"].to_numpy(dtype=float)[sampled].sum(axis=1)
    error_sum = grouped["error_sum"].to_numpy(dtype=float)[sampled].sum(axis=1)
    within_sum = grouped["within_sum"].to_numpy(dtype=float)[sampled].sum(axis=1)
    bootstrap = {
        "mae": absolute_sum / counts,
        "rmse": np.sqrt(squared_sum / counts),
        "mean_signed_error": error_sum / counts,
        "fraction_within_0p5_log": within_sum / counts,
    }
    for metric, values in bootstrap.items():
        point[f"{metric}_lower_95"] = float(np.quantile(values, 0.025))
        point[f"{metric}_upper_95"] = float(np.quantile(values, 0.975))
    return point


def _summarize_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_columns = ["evaluation", "data_regime", "feature_layer", "model"]
    for index, (key, frame) in enumerate(predictions.groupby(group_columns, sort=True)):
        metrics = _bootstrap_metrics(frame, seed=SEED + index)
        rows.append(
            {
                **dict(zip(group_columns, key, strict=True)),
                "evidence_status": " | ".join(sorted(frame["evidence_status"].unique())),
                "external_data_role": " | ".join(sorted(frame["external_data_role"].unique())),
                "training_structures_median": float(frame["training_structures"].median()),
                "training_public_structures_median": float(frame["training_public_structures"].median()),
                **metrics,
            }
        )
    return (
        pd.DataFrame(rows).sort_values(["evaluation", "mae", "feature_layer", "model"]).reset_index(drop=True)
    )


def _paired_external_gain(predictions: pd.DataFrame) -> pd.DataFrame:
    baseline = predictions[predictions["data_regime"].eq("internal_only")][
        [
            "evaluation",
            "record_id",
            "feature_layer",
            "model",
            "absolute_error",
            "scaffold",
        ]
    ].rename(columns={"absolute_error": "internal_only_absolute_error"})
    candidates = predictions[~predictions["data_regime"].eq("internal_only")].merge(
        baseline,
        on=["evaluation", "record_id", "feature_layer", "model", "scaffold"],
        how="inner",
        validate="many_to_one",
    )
    rows: list[dict[str, Any]] = []
    group_columns = ["evaluation", "data_regime", "feature_layer", "model"]
    for index, (key, frame) in enumerate(candidates.groupby(group_columns, sort=True)):
        frame = frame.copy()
        frame["delta"] = frame["absolute_error"] - frame["internal_only_absolute_error"]
        grouped = (
            frame.groupby(frame["scaffold"].astype(str), sort=True)
            .agg(count=("delta", "size"), delta_sum=("delta", "sum"))
            .reset_index(drop=True)
        )
        rng = np.random.default_rng(SEED + 10000 + index)
        deltas = np.array([], dtype=float)
        if len(grouped) >= 2:
            sampled = rng.integers(0, len(grouped), size=(BOOTSTRAPS, len(grouped)))
            counts = grouped["count"].to_numpy(dtype=float)[sampled].sum(axis=1)
            sums = grouped["delta_sum"].to_numpy(dtype=float)[sampled].sum(axis=1)
            deltas = sums / counts
        rows.append(
            {
                **dict(zip(group_columns, key, strict=True)),
                "n": int(len(frame)),
                "n_scaffolds": int(len(grouped)),
                "external_minus_internal_only_mae": float(frame["delta"].mean()),
                "delta_lower_95": (float(np.quantile(deltas, 0.025)) if len(deltas) else float("nan")),
                "delta_upper_95": (float(np.quantile(deltas, 0.975)) if len(deltas) else float("nan")),
                "bootstrap_probability_external_better": (
                    float(np.mean(deltas < 0)) if len(deltas) else float("nan")
                ),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["evaluation", "external_minus_internal_only_mae"])
        .reset_index(drop=True)
    )


def _pk_feature_matrix(
    tables: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    compounds = compound_model_frame(tables["compounds"], tables.get("compound_aliases"))
    frames, target_audit = _source_collapsed_pk_frames(
        compounds,
        tables["measurements"],
        tables["pk_studies"],
    )
    descriptor = structure_feature_frame(compounds[["compound_id", "standardized_smiles"]])
    pka = _aggregate_internal_pka(tables["measurements"])
    prediction_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    endpoints = [
        "iv_auc_dose_normalized",
        "po_auc_dose_normalized",
        "vdss",
        "po_cmax_dose_normalized",
        "po_tmax",
    ]
    for endpoint in endpoints:
        frame = (
            frames[endpoint]
            .merge(
                descriptor[["compound_id", *PROXY_COLUMNS]],
                on="compound_id",
                how="inner",
                validate="one_to_one",
            )
            .merge(pka, on="compound_id", how="left", validate="one_to_one")
        )
        frame["record_id"] = "internal:" + frame["compound_id"].astype(str)
        frame["target_pic50"] = frame["target_log10"]
        frame["source_group"] = "internal"
        frame["fingerprint"] = list(_fingerprints(frame["standardized_smiles"]))
        for fold, (train_index, test_index) in enumerate(_group_folds(frame, folds=5)):
            train = frame.iloc[train_index].copy()
            test = frame.iloc[test_index].copy()
            train["sample_weight"] = 1.0
            for layer in (*FEATURE_LAYERS, "hybrid_pka"):
                for model_name in MODEL_NAMES:
                    try:
                        predicted = _fit_predict(
                            train,
                            test,
                            feature_layer=layer,
                            model_name=model_name,
                        )
                        for position, record in enumerate(test.itertuples(index=False)):
                            prediction_rows.append(
                                {
                                    "endpoint": endpoint,
                                    "fold": fold,
                                    "record_id": record.record_id,
                                    "compound_id": record.compound_id,
                                    "scaffold": record.scaffold,
                                    "observed_log10": float(record.target_log10),
                                    "predicted_log10": float(predicted[position]),
                                    "feature_layer": layer,
                                    "model": model_name,
                                    "data_regime": "internal_only",
                                }
                            )
                    except Exception as exc:
                        failures.append(
                            {
                                "domain": "pk",
                                "endpoint": endpoint,
                                "fold": fold,
                                "data_regime": "internal_only",
                                "feature_layer": layer,
                                "model": model_name,
                                "failure": f"{type(exc).__name__}: {exc}",
                            }
                        )
            mean = float(train["target_log10"].mean())
            nearest = _tanimoto_predict(train, test, neighbors=3)
            for position, record in enumerate(test.itertuples(index=False)):
                for model_name, predicted in (
                    ("train_mean", mean),
                    ("tanimoto_3nn", nearest[position]),
                ):
                    prediction_rows.append(
                        {
                            "endpoint": endpoint,
                            "fold": fold,
                            "record_id": record.record_id,
                            "compound_id": record.compound_id,
                            "scaffold": record.scaffold,
                            "observed_log10": float(record.target_log10),
                            "predicted_log10": float(predicted),
                            "feature_layer": ("none" if model_name == "train_mean" else "morgan_tanimoto"),
                            "model": model_name,
                            "data_regime": "internal_only",
                        }
                    )
    predictions = pd.DataFrame(prediction_rows)
    summaries: list[dict[str, Any]] = []
    for index, (key, frame) in enumerate(
        predictions.groupby(["endpoint", "feature_layer", "model"], sort=True)
    ):
        observed = frame["observed_log10"].to_numpy(dtype=float)
        predicted = frame["predicted_log10"].to_numpy(dtype=float)
        absolute = np.abs(predicted - observed)
        work = frame.copy()
        work["_abs"] = np.abs(work["predicted_log10"] - work["observed_log10"])
        grouped = (
            work.groupby(work["scaffold"].astype(str), sort=True)
            .agg(count=("_abs", "size"), absolute_sum=("_abs", "sum"))
            .reset_index(drop=True)
        )
        rng = np.random.default_rng(SEED + 20000 + index)
        sampled = rng.integers(0, len(grouped), size=(BOOTSTRAPS, len(grouped)))
        counts = grouped["count"].to_numpy(dtype=float)[sampled].sum(axis=1)
        sums = grouped["absolute_sum"].to_numpy(dtype=float)[sampled].sum(axis=1)
        boot_mae = sums / counts
        summaries.append(
            {
                "endpoint": key[0],
                "feature_layer": key[1],
                "model": key[2],
                "n": int(len(frame)),
                "n_scaffolds": int(len(grouped)),
                "log_mae": float(np.mean(absolute)),
                "log_mae_lower_95": float(np.quantile(boot_mae, 0.025)),
                "log_mae_upper_95": float(np.quantile(boot_mae, 0.975)),
                "log_rmse": float(math.sqrt(np.mean((predicted - observed) ** 2))),
                "spearman": _safe_spearman(observed, predicted),
                "median_fold_error": float(np.median(np.power(10.0, absolute))),
                "fraction_within_2fold": float(np.mean(np.power(10.0, absolute) <= 2.0)),
                "fraction_within_3fold": float(np.mean(np.power(10.0, absolute) <= 3.0)),
            }
        )
    return (
        predictions,
        pd.DataFrame(summaries).sort_values(["endpoint", "log_mae"]).reset_index(drop=True),
        pd.concat(
            [pd.DataFrame(failures), target_audit.assign(domain="pk_target_audit")],
            sort=False,
        ),
    )


def _external_pk_compatibility() -> pd.DataFrame:
    path = ROOT / "research/data/processed/pk_admet_observations.csv"
    public = pd.read_csv(path)
    compatible_definitions = [
        ("iv_auc_dose_normalized", "exposure_auc", "rat", "intravenous"),
        ("po_auc_dose_normalized", "exposure_auc", "rat", "oral"),
        ("vdss", "volume_of_distribution", "rat", "intravenous"),
        ("po_cmax_dose_normalized", "maximum_concentration", "rat", "oral"),
        ("po_tmax", "time_to_cmax", "rat", "oral"),
    ]
    rows: list[dict[str, Any]] = []
    for internal_endpoint, external_endpoint, species, route in compatible_definitions:
        subset = public[
            public["admet_endpoint"].eq(external_endpoint)
            & public["species"].fillna("").str.lower().eq(species)
            & public["administration_route"].fillna("").str.lower().eq(route)
        ].copy()
        rows.append(
            {
                "internal_endpoint": internal_endpoint,
                "external_endpoint": external_endpoint,
                "required_species": species,
                "required_route": route,
                "external_rows": int(len(subset)),
                "external_structures": int(subset["structure_id"].nunique()),
                "external_documents": int(subset["document_chembl_id"].nunique()),
                "integration_decision": (
                    "rejected_too_few_protocol_compatible_structures"
                    if subset["structure_id"].nunique() < 20
                    else "candidate_for_source_held_out_transfer"
                ),
                "reason": (
                    "A model is not fit when fewer than 20 external structures share "
                    "endpoint, species, route, and interpretable units; sparse rows remain "
                    "context evidence."
                ),
            }
        )
    return pd.DataFrame(rows)


def _feature_contract() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "feature_layer": "compact_proxies",
                "inputs": "|".join(PROXY_COLUMNS),
                "scientific_role": "interpretable conventional physical-property proxies",
                "allowed_claim": (
                    "tests whether size, neutral-parent hydrophobic drive, polarity, "
                    "hydrogen-bonding capacity, flexibility, and topology carry signal"
                ),
                "prohibited_claim": "not fundamental free energies, rates, or fluxes",
            },
            {
                "feature_layer": "morgan_latent",
                "inputs": "ECFP4 2048-bit fingerprint compressed to 8 fold-fitted components",
                "scientific_role": "associative local-substructure baseline",
                "allowed_claim": "tests analogue and motif signal",
                "prohibited_claim": "not a physical mechanism",
            },
            {
                "feature_layer": "hybrid",
                "inputs": "compact proxies plus Morgan latent components",
                "scientific_role": "associative structure plus interpretable proxy control",
                "allowed_claim": "tests complementarity of global properties and local motifs",
                "prohibited_claim": "not a state/path/flux model",
            },
            {
                "feature_layer": "hybrid_pka",
                "inputs": "hybrid plus single-site cation-fraction proxy from maximum reported basic pKa",
                "scientific_role": "internal-only chemical-state sensitivity ablation",
                "allowed_claim": "tests whether available protonation evidence adds held-scaffold signal",
                "prohibited_claim": (
                    "not a microscopic-state population; coupled sites and site assignment remain unresolved"
                ),
            },
            {
                "feature_layer": "state_path_flux",
                "inputs": (
                    "microstate free energies and rates; membrane PMF/diffusivity; "
                    "channel-state binding and kinetic rates"
                ),
                "scientific_role": "genuinely fundamental mechanistic layer",
                "allowed_claim": "none until experimental/HPC admission gates pass",
                "prohibited_claim": "must not be imputed from 2D descriptors",
            },
        ]
    )


def _build_report(
    hsummary: pd.DataFrame,
    gains: pd.DataFrame,
    pk_summary: pd.DataFrame,
    pk_compatibility: pd.DataFrame,
    failures: pd.DataFrame,
) -> str:
    lines = [
        "# Menin-inhibitor hERG/PK mix-and-match analysis",
        "",
        "## Scientific boundary",
        "",
        "The prediction target is hERG liability (continuous pIC50 and derived screening "
        "class) for Menin inhibitors. Menin potency is not modeled. Internal-only models "
        "remain the anchor. External rows are tested as same-series augmentation, pooled "
        "training evidence, covariate-selected support, large-molecule support, or a "
        "pretrained prior followed by internal residual calibration.",
        "",
        "The nine compact molecular properties are interpretable proxies, not fundamental "
        "physics. Genuine state/path/flux quantities remain excluded because no converged "
        "microstate, membrane, receptor-kinetic, or free-concentration evidence is admitted.",
        "",
        "## hERG headline comparisons",
        "",
    ]
    for evaluation in hsummary["evaluation"].unique():
        subset = hsummary[hsummary["evaluation"].eq(evaluation)]
        internal = subset[subset["data_regime"].eq("internal_only")].nsmallest(1, "mae")
        external = subset[~subset["data_regime"].eq("internal_only")].nsmallest(1, "mae")
        if not internal.empty:
            row = internal.iloc[0]
            lines.append(
                f"- **{evaluation}, internal anchor:** {row['model']} + "
                f"{row['feature_layer']}, MAE {row['mae']:.3f}, Spearman "
                f"{row['spearman']:.3f}, n={int(row['n'])}."
            )
        if not external.empty:
            row = external.iloc[0]
            lines.append(
                f"- **{evaluation}, best external strategy (descriptive/post hoc):** "
                f"{row['data_regime']} with {row['model']} + {row['feature_layer']}, "
                f"MAE {row['mae']:.3f}, Spearman {row['spearman']:.3f}."
            )
    lines.extend(
        [
            "",
            "These minima summarize the completed matrix; they are not prospective model "
            "selection evidence. The paired external-minus-internal table is the relevant "
            "test of whether adding external evidence helps a fixed model/representation.",
            "",
            "## External-data decisions",
            "",
        ]
    )
    for evaluation in ("internal_scaffold_cv", "angelo_fixed_nonoverlap"):
        subset = gains[gains["evaluation"].eq(evaluation)].sort_values("external_minus_internal_only_mae")
        if not subset.empty:
            best = subset.iloc[0]
            worst = subset.iloc[-1]
            lines.append(
                f"- On **{evaluation}**, the most favorable paired external strategy was "
                f"{best['data_regime']} ({best['feature_layer']} + {best['model']}): "
                f"ΔMAE {best['external_minus_internal_only_mae']:+.3f} "
                f"[{best['delta_lower_95']:+.3f}, {best['delta_upper_95']:+.3f}]."
            )
            lines.append(
                f"- The least favorable was {worst['data_regime']} "
                f"({worst['feature_layer']} + {worst['model']}): "
                f"ΔMAE {worst['external_minus_internal_only_mae']:+.3f}."
            )
    lines.extend(
        [
            "",
            "Angelo/Ascentage augmentation remains same-series development evidence. Broad "
            "public data remain assay- and chemistry-shifted. No external strategy becomes "
            "decision-track without an untouched protocol-matched Menin series.",
            "",
            "## PK result",
            "",
        ]
    )
    for endpoint in pk_summary["endpoint"].unique():
        best = pk_summary[pk_summary["endpoint"].eq(endpoint)].nsmallest(1, "log_mae").iloc[0]
        lines.append(
            f"- **{endpoint}:** {best['model']} + {best['feature_layer']}; "
            f"log-MAE {best['log_mae']:.3f}, median fold error "
            f"{best['median_fold_error']:.2f}, Spearman {best['spearman']:.3f}."
        )
    rejected = pk_compatibility[pk_compatibility["integration_decision"].str.startswith("rejected")]
    lines.extend(
        [
            "",
            f"External PK pooling was rejected for {len(rejected)}/{len(pk_compatibility)} "
            "target definitions because compatible rat endpoint/route groups contain too "
            "few distinct structures. This negative result is retained rather than replacing "
            "the intended analysis with incompatible species, routes, or endpoints.",
            "",
            "## Failure preservation",
            "",
            f"- Executed model-fit failures: {int((failures['domain'].isin(['herg', 'pk'])).sum()) if not failures.empty and 'domain' in failures else 0}.",
            "- Scientifically rejected integrations and unsupported physical layers are "
            "recorded separately from software failures.",
            "",
            "## What can be presented",
            "",
            "1. A frozen internal hERG anchor and a same-series augmented discovery model.",
            "2. A direct test of naive pooling, source balancing, chemical-neighborhood "
            "selection, size matching, and public-prior residual transfer.",
            "3. Basic associative fingerprints versus compact mechanism-motivated proxies and their hybrid.",
            "4. Continuous pIC50, rank/error metrics, decisive liability metrics, paired "
            "external-data gains, domain similarity, negative controls, and uncertainty.",
            "5. Internal PK feature/model comparisons plus a quantitative reason why current "
            "external PK data cannot yet be pooled.",
            "",
            "## Promotion rule",
            "",
            "A model is eligible for future decision use only if it improves a fixed "
            "internal comparator on an untouched new Menin series, preserves calibration, "
            "and reports applicability. Until then, outputs are analogue-series discovery "
            "hypotheses and assay-prioritization evidence.",
            "",
        ]
    )
    return "\n".join(lines)


def _figures(summary: pd.DataFrame, gains: pd.DataFrame, output: Path) -> None:
    selected = summary[
        summary["evaluation"].isin(["internal_scaffold_cv", "angelo_fixed_nonoverlap"])
        & summary["feature_layer"].isin(["compact_proxies", "morgan_latent", "hybrid"])
        & summary["model"].isin(["ridge", "svr", "random_forest", "extra_trees"])
    ].copy()
    for evaluation in selected["evaluation"].unique():
        panel = selected[selected["evaluation"].eq(evaluation)].nsmallest(18, "mae")
        labels = (
            panel["data_regime"].str.replace("internal_plus_", "I+")
            + " | "
            + panel["feature_layer"].str.replace("_", " ")
            + " | "
            + panel["model"].str.replace("_", " ")
        )
        figure, axis = plt.subplots(figsize=(11, 7))
        order = np.arange(len(panel))[::-1]
        axis.barh(order, panel["mae"], color="#2A6F97")
        axis.set_yticks(order, labels)
        axis.set_xlabel("pIC50 MAE (lower is better)")
        axis.set_title(f"{evaluation}: best completed mix-and-match combinations")
        axis.grid(axis="x", alpha=0.2)
        figure.tight_layout()
        stem = f"{evaluation}_model_comparison"
        _atomic_figure(figure, output / f"{stem}.png", dpi=220)
        _atomic_figure(figure, output / f"{stem}.pdf")
        plt.close(figure)

    gain_panel = gains[
        gains["evaluation"].isin(["internal_scaffold_cv", "angelo_fixed_nonoverlap"])
        & gains["feature_layer"].isin(["compact_proxies", "hybrid"])
        & gains["model"].isin(["ridge", "extra_trees"])
    ].copy()
    if not gain_panel.empty:
        gain_panel = gain_panel.sort_values("external_minus_internal_only_mae")
        labels = (
            gain_panel["evaluation"].str.replace("_", " ")
            + " | "
            + gain_panel["data_regime"].str.replace("internal_plus_", "I+")
            + " | "
            + gain_panel["feature_layer"].str.replace("_", " ")
            + " | "
            + gain_panel["model"].str.replace("_", " ")
        )
        figure, axis = plt.subplots(figsize=(12, max(6, 0.28 * len(gain_panel))))
        y = np.arange(len(gain_panel))
        x = gain_panel["external_minus_internal_only_mae"].to_numpy(dtype=float)
        lower = x - gain_panel["delta_lower_95"].to_numpy(dtype=float)
        upper = gain_panel["delta_upper_95"].to_numpy(dtype=float) - x
        axis.errorbar(x, y, xerr=np.vstack([lower, upper]), fmt="o", color="#014F86", ecolor="#61A5C2")
        axis.axvline(0, color="#9B2226", linestyle="--", linewidth=1)
        axis.set_yticks(y, labels)
        axis.set_xlabel("External strategy minus internal-only MAE (negative helps)")
        axis.set_title("Paired external-data contribution with scaffold bootstrap")
        axis.grid(axis="x", alpha=0.2)
        figure.tight_layout()
        _atomic_figure(figure, output / "external_data_gain.png", dpi=220)
        _atomic_figure(figure, output / "external_data_gain.pdf")
        plt.close(figure)


def run(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    internal, intervals, tables = _internal_herg_frames()
    internal_structure_ids = set(tables["compounds"]["structure_id"].astype(str))
    extension, extension_intervals = _extension_frames(internal_structure_ids)
    excluded = internal_structure_ids | set(extension["structure_id"].astype(str))
    public_train, public_validation = _public_frames(excluded)

    predictions, herg_failures = _run_herg_matrix(
        internal,
        extension,
        public_train,
        public_validation,
        checkpoint_dir=output,
    )
    atomic_write_parquet(output / "herg_mix_match_predictions.parquet", predictions)
    atomic_write_csv(output / "herg_fit_failures.csv", herg_failures)
    summary = _summarize_predictions(predictions)
    gains = _paired_external_gain(predictions)
    pk_predictions, pk_summary, pk_failures = _pk_feature_matrix(tables)
    pk_compatibility = _external_pk_compatibility()
    feature_contract = _feature_contract()
    failures = pd.concat([herg_failures, pk_failures], ignore_index=True, sort=False)

    dataset_audit = pd.DataFrame(
        [
            {
                "dataset": "internal_hERG_exact_compound_collapsed",
                "structures": len(internal),
                "scaffolds": internal["scaffold"].nunique(),
                "role": "required anchor and internal scaffold CV",
                "protocol_status": "internal source; incomplete dynamic/free-concentration metadata",
            },
            {
                "dataset": "internal_hERG_interval_compound_collapsed",
                "structures": len(intervals),
                "scaffolds": intervals["scaffold"].nunique(),
                "role": "censoring and source-conflict audit",
                "protocol_status": "internal source",
            },
            {
                "dataset": "Angelo_Ascentage_nonoverlap_exact",
                "structures": len(extension),
                "scaffolds": extension["scaffold"].nunique(),
                "role": "retrospective same-series extension and augmentation",
                "protocol_status": "protocol metadata missing",
            },
            {
                "dataset": "Angelo_Ascentage_nonoverlap_all_measured",
                "structures": len(extension_intervals),
                "scaffolds": extension_intervals["scaffold"].nunique(),
                "role": "exact plus >30 uM censoring audit",
                "protocol_status": "protocol metadata missing",
            },
            {
                "dataset": "Sun_public_regression_train",
                "structures": len(public_train),
                "scaffolds": public_train["scaffold"].nunique(),
                "role": "external pooling/pretraining experiments",
                "protocol_status": "public-source heterogeneous",
            },
            {
                "dataset": "Sun_public_source_holdout",
                "structures": len(public_validation),
                "scaffolds": public_validation["scaffold"].nunique(),
                "role": "external-source diagnostic",
                "protocol_status": "not protocol matched to internal",
            },
        ]
    )

    atomic_write_csv(output / "herg_mix_match_summary.csv", summary)
    atomic_write_csv(output / "external_gain_vs_internal.csv", gains)
    atomic_write_parquet(output / "pk_feature_model_predictions.parquet", pk_predictions)
    atomic_write_csv(output / "pk_feature_model_summary.csv", pk_summary)
    atomic_write_csv(output / "external_pk_compatibility.csv", pk_compatibility)
    atomic_write_csv(output / "feature_contract.csv", feature_contract)
    atomic_write_csv(output / "dataset_audit.csv", dataset_audit)
    atomic_write_csv(output / "failure_ledger.csv", failures)
    atomic_write_csv(output / "internal_interval_collapse_audit.csv", intervals)
    atomic_write_csv(output / "extension_interval_audit.csv", extension_intervals)
    report = _build_report(summary, gains, pk_summary, pk_compatibility, failures)
    atomic_write_text(output / "professor_briefing.md", report)
    _figures(summary, gains, output)

    payload = {
        "status": "completed",
        "hERG_prediction_rows": int(len(predictions)),
        "hERG_experiment_summaries": int(len(summary)),
        "hERG_fit_failures": int(len(herg_failures)),
        "internal_exact_structures": int(len(internal)),
        "extension_exact_nonoverlap_structures": int(len(extension)),
        "public_train_structures": int(len(public_train)),
        "public_validation_structures": int(len(public_validation)),
        "pk_experiment_summaries": int(len(pk_summary)),
        "pk_prediction_rows": int(len(pk_predictions)),
        "external_pk_endpoints_rejected": int(
            pk_compatibility["integration_decision"].str.startswith("rejected").sum()
        ),
        "feature_policy": (
            "compact proxies, Morgan association, hybrids, and one internal-only pKa "
            "sensitivity; no unadmitted physics quantities"
        ),
        "claim_boundary": (
            "retrospective discovery analysis; requires untouched protocol-matched "
            "multi-series Menin hERG evaluation"
        ),
    }
    atomic_write_json(output / "run_summary.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2))


if __name__ == "__main__":
    main()
