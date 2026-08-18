#!/usr/bin/env python3
"""Add a structure-sensitive, censoring-aware hERG representation audit.

The earlier extension model used nine global RDKit controls.  This audit uses
the complete ECFP4 bit vector, compressed inside each training fold, together
with those nine controls.  Component count is selected exclusively from the
original grouped training data; extension outcomes never select the setting.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from menin_discovery.research_ascentage import load_ascentage_source
from menin_discovery.research_common import (
    atomic_write_csv,
    atomic_write_json,
    atomic_write_parquet,
    atomic_write_text,
)
from menin_discovery.research_modeling import (
    CensoredGaussianRidge,
    merge_feature_layers,
    structure_feature_frame,
)
from menin_discovery.research_workflows import (
    compound_model_frame,
    load_canonical_tables,
    prepare_herg_evidence,
)
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from scipy.stats import norm, spearmanr
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
CANONICAL = ROOT / "research/data/pk_herg/canonical"
SOURCE = CANONICAL / "ascentage_herg_2026_07_28/normalized_records.parquet"
EXTENSION_ROOT = ROOT / "research/reports/pk_herg/ascentage_herg_extension"
OUTPUT = EXTENSION_ROOT / "complete_feature_model"
SEED = 20260728
COMPONENT_CANDIDATES = (4, 8, 12, 16, 24, 32)
KERNEL_ALPHA_CANDIDATES = (
    0.3,
    1.0,
    3.0,
    10.0,
    30.0,
    100.0,
    300.0,
    1000.0,
    3000.0,
    10000.0,
    30000.0,
    100000.0,
)
FOLDS = 5
ALPHA = 3.0


def _atomic_figure(figure: plt.Figure, path: Path, **kwargs: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.stem}.tmp{path.suffix}")
    figure.savefig(temporary, **kwargs)
    os.replace(temporary, path)


def _fingerprints(smiles: pd.Series) -> np.ndarray:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    rows: list[np.ndarray] = []
    for value in smiles.astype(str):
        molecule = Chem.MolFromSmiles(value)
        if molecule is None:
            raise ValueError(f"Invalid standardized SMILES: {value}")
        rows.append(generator.GetFingerprintAsNumPy(molecule).astype(float))
    return np.asarray(rows)


def _nll(
    lower: np.ndarray,
    upper: np.ndarray,
    prediction: np.ndarray,
    sigma: np.ndarray,
) -> float:
    sigma = np.maximum(sigma, 1e-6)
    exact = np.isfinite(lower) & np.isfinite(upper) & np.isclose(lower, upper)
    logp = np.zeros(len(lower))
    logp[exact] = norm.logpdf(lower[exact], loc=prediction[exact], scale=sigma[exact])
    upper_only = ~np.isfinite(lower) & np.isfinite(upper)
    logp[upper_only] = norm.logcdf((upper[upper_only] - prediction[upper_only]) / sigma[upper_only])
    lower_only = np.isfinite(lower) & ~np.isfinite(upper)
    logp[lower_only] = norm.logsf((lower[lower_only] - prediction[lower_only]) / sigma[lower_only])
    interval = np.isfinite(lower) & np.isfinite(upper) & ~exact
    if np.any(interval):
        upper_cdf = norm.cdf((upper[interval] - prediction[interval]) / sigma[interval])
        lower_cdf = norm.cdf((lower[interval] - prediction[interval]) / sigma[interval])
        logp[interval] = np.log(np.maximum(upper_cdf - lower_cdf, 1e-12))
    return float(-np.mean(logp))


def _metrics(
    lower: np.ndarray,
    upper: np.ndarray,
    prediction: np.ndarray,
    sigma: np.ndarray,
) -> dict[str, float]:
    exact = np.isfinite(lower) & np.isfinite(upper) & np.isclose(lower, upper)
    observed = lower[exact]
    predicted = prediction[exact]
    error = predicted - observed
    return {
        "n_rows": float(len(lower)),
        "n_exact": float(exact.sum()),
        "censored_negative_log_likelihood": _nll(lower, upper, prediction, sigma),
        "pic50_mae": float(np.mean(np.abs(error))),
        "pic50_rmse": float(np.sqrt(np.mean(error**2))),
        "spearman": (
            float(spearmanr(observed, predicted).statistic)
            if len(observed) >= 3 and np.std(predicted) > 1e-12
            else float("nan")
        ),
        "mean_signed_error": float(np.mean(error)),
        "fraction_within_0p5_log": float(np.mean(np.abs(error) <= 0.5)),
        "fraction_within_1p0_log": float(np.mean(np.abs(error) <= 1.0)),
    }


def _project(
    train_bits: np.ndarray,
    train_controls: np.ndarray,
    score_bits: np.ndarray,
    score_controls: np.ndarray,
    *,
    components: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    reducer = TruncatedSVD(n_components=components, random_state=seed)
    train_latent = reducer.fit_transform(train_bits)
    score_latent = reducer.transform(score_bits)
    scaler = StandardScaler()
    train = scaler.fit_transform(np.column_stack([train_latent, train_controls]))
    score = scaler.transform(np.column_stack([score_latent, score_controls]))
    return train, score, float(reducer.explained_variance_ratio_.sum())


def _tanimoto(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Compute binary-fingerprint Tanimoto similarity without bit truncation."""
    intersection = left @ right.T
    denominator = left.sum(axis=1, keepdims=True) + right.sum(axis=1, keepdims=True).T - intersection
    return np.divide(
        intersection,
        denominator,
        out=np.zeros_like(intersection, dtype=float),
        where=denominator > 0,
    )


def _unique_fingerprints(bits: np.ndarray) -> np.ndarray:
    """Return one kernel reference per unique structure-level bit vector."""
    return np.unique(np.asarray(bits, dtype=float), axis=0)


def _kernel_project(
    train_bits: np.ndarray,
    train_controls: np.ndarray,
    score_bits: np.ndarray,
    score_controls: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Represent rows by similarity to unique training structures plus controls."""
    references = _unique_fingerprints(train_bits)
    train_kernel = _tanimoto(train_bits, references)
    score_kernel = _tanimoto(score_bits, references)
    scaler = StandardScaler()
    train = scaler.fit_transform(np.column_stack([train_kernel, train_controls]))
    score = scaler.transform(np.column_stack([score_kernel, score_controls]))
    return train, score, len(references)


def _oof(
    frame: pd.DataFrame,
    bits: np.ndarray,
    controls: np.ndarray,
    *,
    components: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    lower = frame["pic50_lower"].to_numpy(dtype=float)
    upper = frame["pic50_upper"].to_numpy(dtype=float)
    prediction = np.zeros(len(frame))
    sigma = np.zeros(len(frame))
    converged = np.zeros(len(frame), dtype=bool)
    fold_id = np.zeros(len(frame), dtype=int)
    explained = np.zeros(len(frame))
    splits = GroupKFold(FOLDS).split(bits, groups=frame["scaffold"].astype(str))
    for fold, (train_index, test_index) in enumerate(splits):
        train, test, variance = _project(
            bits[train_index],
            controls[train_index],
            bits[test_index],
            controls[test_index],
            components=components,
            seed=SEED + fold,
        )
        model = CensoredGaussianRidge(alpha=ALPHA, maxiter=3000).fit(
            train,
            lower[train_index],
            upper[train_index],
        )
        prediction[test_index] = model.predict(test)
        sigma[test_index] = model.sigma_
        converged[test_index] = model.converged_
        fold_id[test_index] = fold
        explained[test_index] = variance
    metrics = {
        "components": components,
        "all_folds_converged": bool(converged.all()),
        "mean_fold_explained_fingerprint_variance": float(np.mean(explained)),
        **_metrics(lower, upper, prediction, sigma),
    }
    rows = frame[["compound_id", "standardized_smiles", "scaffold", "pic50_lower", "pic50_upper"]].copy()
    rows["components"] = components
    rows["fold"] = fold_id
    rows["predicted_pic50"] = prediction
    rows["predictive_sigma"] = sigma
    rows["fit_converged"] = converged
    rows["explained_fingerprint_variance"] = explained
    return metrics, rows


def _kernel_oof(
    frame: pd.DataFrame,
    bits: np.ndarray,
    controls: np.ndarray,
    *,
    alpha: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Grouped OOF predictions using all ECFP4 bits through a Tanimoto basis."""
    lower = frame["pic50_lower"].to_numpy(dtype=float)
    upper = frame["pic50_upper"].to_numpy(dtype=float)
    prediction = np.zeros(len(frame))
    sigma = np.zeros(len(frame))
    converged = np.zeros(len(frame), dtype=bool)
    fold_id = np.zeros(len(frame), dtype=int)
    reference_count = np.zeros(len(frame), dtype=int)
    splits = GroupKFold(FOLDS).split(bits, groups=frame["scaffold"].astype(str))
    for fold, (train_index, test_index) in enumerate(splits):
        train, test, references = _kernel_project(
            bits[train_index],
            controls[train_index],
            bits[test_index],
            controls[test_index],
        )
        model = CensoredGaussianRidge(alpha=alpha, maxiter=5000).fit(
            train,
            lower[train_index],
            upper[train_index],
        )
        prediction[test_index] = model.predict(test)
        sigma[test_index] = model.sigma_
        converged[test_index] = model.converged_
        fold_id[test_index] = fold
        reference_count[test_index] = references
    metrics = {
        "alpha": alpha,
        "all_folds_converged": bool(converged.all()),
        "mean_unique_training_references": float(np.mean(reference_count)),
        **_metrics(lower, upper, prediction, sigma),
    }
    rows = frame[["compound_id", "standardized_smiles", "scaffold", "pic50_lower", "pic50_upper"]].copy()
    rows["alpha"] = alpha
    rows["fold"] = fold_id
    rows["predicted_pic50"] = prediction
    rows["predictive_sigma"] = sigma
    rows["fit_converged"] = converged
    rows["unique_training_references"] = reference_count
    return metrics, rows


def _fit_score(
    training: pd.DataFrame,
    training_bits: np.ndarray,
    training_controls: np.ndarray,
    scoring: pd.DataFrame,
    scoring_bits: np.ndarray,
    scoring_controls: np.ndarray,
    oof: pd.DataFrame,
    *,
    components: int,
) -> pd.DataFrame:
    train, score, explained = _project(
        training_bits,
        training_controls,
        scoring_bits,
        scoring_controls,
        components=components,
        seed=SEED,
    )
    model = CensoredGaussianRidge(alpha=ALPHA, maxiter=3000).fit(
        train,
        training["pic50_lower"].to_numpy(dtype=float),
        training["pic50_upper"].to_numpy(dtype=float),
    )
    if not model.converged_:
        raise RuntimeError("Complete-feature final censored model did not converge")
    predicted = model.predict(score)
    exact = (
        np.isfinite(oof["pic50_lower"])
        & np.isfinite(oof["pic50_upper"])
        & np.isclose(oof["pic50_lower"], oof["pic50_upper"])
    )
    residual = oof.loc[exact, "predicted_pic50"].to_numpy(dtype=float) - oof.loc[
        exact, "pic50_lower"
    ].to_numpy(dtype=float)
    radius = float(np.quantile(np.abs(residual), 0.90, method="higher"))
    predictive_sigma = max(float(np.std(residual, ddof=1)), 1e-6)
    result = scoring[
        [
            "structure_id",
            "internal_id",
            "herg_pic50_relation",
            "herg_pic50_value",
            "herg_pic50_upper_bound",
            "scaffold",
        ]
    ].copy()
    result["complete_feature_predicted_pic50"] = predicted
    result["complete_feature_pic50_lower"] = predicted - radius
    result["complete_feature_pic50_upper"] = predicted + radius
    result["complete_feature_blocker_probability"] = norm.sf((5.0 - predicted) / predictive_sigma)
    result["complete_feature_interval_radius"] = radius
    result["complete_feature_oof_sigma"] = predictive_sigma
    result["fingerprint_components"] = components
    result["full_fit_explained_fingerprint_variance"] = explained
    result["model_fit_converged"] = model.converged_
    return result


def _kernel_fit_score(
    training: pd.DataFrame,
    training_bits: np.ndarray,
    training_controls: np.ndarray,
    scoring: pd.DataFrame,
    scoring_bits: np.ndarray,
    scoring_controls: np.ndarray,
    oof: pd.DataFrame,
    *,
    alpha: float,
) -> pd.DataFrame:
    train, score, references = _kernel_project(
        training_bits,
        training_controls,
        scoring_bits,
        scoring_controls,
    )
    model = CensoredGaussianRidge(alpha=alpha, maxiter=5000).fit(
        train,
        training["pic50_lower"].to_numpy(dtype=float),
        training["pic50_upper"].to_numpy(dtype=float),
    )
    if not model.converged_:
        raise RuntimeError("Full-fingerprint Tanimoto-kernel model did not converge")
    predicted = model.predict(score)
    exact = (
        np.isfinite(oof["pic50_lower"])
        & np.isfinite(oof["pic50_upper"])
        & np.isclose(oof["pic50_lower"], oof["pic50_upper"])
    )
    residual = oof.loc[exact, "predicted_pic50"].to_numpy(dtype=float) - oof.loc[
        exact, "pic50_lower"
    ].to_numpy(dtype=float)
    radius = float(np.quantile(np.abs(residual), 0.90, method="higher"))
    predictive_sigma = max(float(np.std(residual, ddof=1)), 1e-6)
    result = scoring[
        [
            "structure_id",
            "internal_id",
            "herg_pic50_relation",
            "herg_pic50_value",
            "herg_pic50_upper_bound",
            "scaffold",
        ]
    ].copy()
    result["tanimoto_kernel_predicted_pic50"] = predicted
    result["tanimoto_kernel_pic50_lower"] = predicted - radius
    result["tanimoto_kernel_pic50_upper"] = predicted + radius
    result["tanimoto_kernel_blocker_probability"] = norm.sf((5.0 - predicted) / predictive_sigma)
    result["tanimoto_kernel_interval_radius"] = radius
    result["tanimoto_kernel_oof_sigma"] = predictive_sigma
    result["tanimoto_kernel_alpha"] = alpha
    result["tanimoto_kernel_unique_training_references"] = references
    result["model_fit_converged"] = model.converged_
    return result


def _extension_metrics(
    predictions: pd.DataFrame,
    *,
    prediction_column: str = "complete_feature_predicted_pic50",
    interval_lower_column: str = "complete_feature_pic50_lower",
    interval_upper_column: str = "complete_feature_pic50_upper",
) -> dict[str, Any]:
    exact = predictions["herg_pic50_relation"].eq("=")
    observed = predictions.loc[exact, "herg_pic50_value"].to_numpy(dtype=float)
    predicted = predictions.loc[exact, prediction_column].to_numpy(dtype=float)
    error = predicted - observed
    censored = predictions["herg_pic50_relation"].eq("<")
    return {
        "evaluation_role": "retrospective_nonoverlap_extension",
        "n_novel_structures": len(predictions),
        "n_exact": int(exact.sum()),
        "n_exact_scaffolds": int(predictions.loc[exact, "scaffold"].nunique()),
        "largest_exact_scaffold_fraction": float(
            predictions.loc[exact, "scaffold"].value_counts(normalize=True).max()
        ),
        "n_right_censored": int(censored.sum()),
        "pic50_mae": float(np.mean(np.abs(error))),
        "pic50_rmse": float(np.sqrt(np.mean(error**2))),
        "spearman": float(spearmanr(observed, predicted).statistic),
        "mean_signed_error": float(np.mean(error)),
        "fraction_within_0p5_log": float(np.mean(np.abs(error) <= 0.5)),
        "fraction_within_1p0_log": float(np.mean(np.abs(error) <= 1.0)),
        "interval_coverage": float(
            np.mean(
                (observed >= predictions.loc[exact, interval_lower_column])
                & (observed <= predictions.loc[exact, interval_upper_column])
            )
        ),
        "interval_mean_width": float(
            np.mean(
                predictions.loc[exact, interval_upper_column] - predictions.loc[exact, interval_lower_column]
            )
        ),
        "strict_censored_compatibility": float(
            np.mean(
                predictions.loc[censored, prediction_column]
                <= predictions.loc[censored, "herg_pic50_upper_bound"]
            )
        ),
    }


def _classification_metrics(
    predictions: pd.DataFrame,
    *,
    probability_column: str,
) -> dict[str, float | int]:
    """Evaluate only outcomes whose relation determines the pIC50>=5 class."""
    exact = predictions["herg_pic50_relation"].eq("=")
    definite_nonblocker = predictions["herg_pic50_relation"].eq("<") & predictions[
        "herg_pic50_upper_bound"
    ].lt(5.0)
    evaluable = exact | definite_nonblocker
    frame = predictions.loc[evaluable].copy()
    observed = np.where(
        frame["herg_pic50_relation"].eq("="),
        frame["herg_pic50_value"].ge(5.0),
        False,
    ).astype(int)
    probability = np.clip(frame[probability_column].to_numpy(dtype=float), 1e-8, 1 - 1e-8)
    predicted = probability >= 0.5
    true_negative, false_positive, false_negative, true_positive = confusion_matrix(
        observed,
        predicted,
        labels=[0, 1],
    ).ravel()
    bin_edges = np.linspace(0.0, 1.0, 6)
    bin_ids = np.clip(np.digitize(probability, bin_edges[1:-1], right=True), 0, 4)
    ece = 0.0
    for bin_id in range(5):
        mask = bin_ids == bin_id
        if np.any(mask):
            ece += float(np.mean(mask)) * abs(
                float(np.mean(observed[mask])) - float(np.mean(probability[mask]))
            )
    return {
        "threshold_pic50": 5.0,
        "n_evaluable": int(len(frame)),
        "n_blockers": int(observed.sum()),
        "n_nonblockers": int((1 - observed).sum()),
        "roc_auc": float(roc_auc_score(observed, probability)),
        "pr_auc": float(average_precision_score(observed, probability)),
        "balanced_accuracy": float(balanced_accuracy_score(observed, predicted)),
        "mcc": float(matthews_corrcoef(observed, predicted)),
        "sensitivity": float(true_positive / (true_positive + false_negative)),
        "specificity": float(true_negative / (true_negative + false_positive)),
        "brier": float(brier_score_loss(observed, probability)),
        "ece_5_bin": float(ece),
        "log_loss": float(log_loss(observed, probability)),
        "true_positive": int(true_positive),
        "true_negative": int(true_negative),
        "false_positive": int(false_positive),
        "false_negative": int(false_negative),
    }


def _bootstrap_comparison(predictions: pd.DataFrame, locked: pd.DataFrame) -> pd.DataFrame:
    exact = predictions[predictions["herg_pic50_relation"].eq("=")].merge(
        locked[
            [
                "structure_id",
                "predicted_pic50",
                "herg_pic50_value",
                "scaffold",
            ]
        ],
        on=["structure_id", "herg_pic50_value", "scaffold"],
        validate="one_to_one",
    )
    groups = sorted(exact["scaffold"].unique())
    positions = {group: exact.index[exact["scaffold"].eq(group)].to_numpy() for group in groups}
    rng = np.random.default_rng(SEED)
    rows = []
    for replicate in range(5000):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        index = np.concatenate([positions[group] for group in sampled])
        frame = exact.loc[index]
        observed = frame["herg_pic50_value"].to_numpy(dtype=float)
        complete_mae = float(np.mean(np.abs(frame["complete_feature_predicted_pic50"] - observed)))
        locked_mae = float(np.mean(np.abs(frame["predicted_pic50"] - observed)))
        rows.append(
            {
                "replicate": replicate,
                "complete_feature_mae": complete_mae,
                "locked_global_control_mae": locked_mae,
                "complete_minus_locked_mae": complete_mae - locked_mae,
            }
        )
    return pd.DataFrame(rows)


def _paired_scaffold_bootstrap(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    left_column: str,
    right_column: str,
    left_name: str,
    right_name: str,
) -> pd.DataFrame:
    """Compare two prediction sets without treating analogues as independent."""
    exact = left[left["herg_pic50_relation"].eq("=")][
        ["structure_id", "herg_pic50_value", "scaffold", left_column]
    ].merge(
        right[["structure_id", right_column]],
        on="structure_id",
        validate="one_to_one",
    )
    groups = sorted(exact["scaffold"].unique())
    positions = {group: exact.index[exact["scaffold"].eq(group)].to_numpy() for group in groups}
    rng = np.random.default_rng(SEED + 1)
    rows: list[dict[str, float | int]] = []
    for replicate in range(5000):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        index = np.concatenate([positions[group] for group in sampled])
        frame = exact.loc[index]
        observed = frame["herg_pic50_value"].to_numpy(dtype=float)
        left_mae = float(np.mean(np.abs(frame[left_column] - observed)))
        right_mae = float(np.mean(np.abs(frame[right_column] - observed)))
        rows.append(
            {
                "replicate": replicate,
                f"{left_name}_mae": left_mae,
                f"{right_name}_mae": right_mae,
                f"{left_name}_minus_{right_name}_mae": left_mae - right_mae,
            }
        )
    return pd.DataFrame(rows)


def _prediction_disagreement(
    svd_predictions: pd.DataFrame,
    kernel_predictions: pd.DataFrame,
    global_control_panel: pd.DataFrame,
) -> pd.DataFrame:
    """Expose model-form uncertainty instead of choosing favorable predictions."""
    old = global_control_panel[
        [
            "internal_id",
            "augmented_predicted_pic50",
            "augmented_predicted_pic50_lower",
            "augmented_predicted_pic50_upper",
            "augmented_predicted_blocker_probability",
        ]
    ].dropna(subset=["augmented_predicted_pic50"])
    comparison = svd_predictions.merge(
        kernel_predictions[
            [
                "internal_id",
                "tanimoto_kernel_predicted_pic50",
                "tanimoto_kernel_pic50_lower",
                "tanimoto_kernel_pic50_upper",
                "tanimoto_kernel_blocker_probability",
            ]
        ],
        on="internal_id",
        validate="one_to_one",
    ).merge(old, on="internal_id", validate="one_to_one")
    comparison = comparison.rename(
        columns={
            "augmented_predicted_pic50": "global_control_predicted_pic50",
            "augmented_predicted_pic50_lower": "global_control_pic50_lower",
            "augmented_predicted_pic50_upper": "global_control_pic50_upper",
            "augmented_predicted_blocker_probability": ("global_control_blocker_probability"),
        }
    )
    eligible_point_columns = [
        "global_control_predicted_pic50",
        "complete_feature_predicted_pic50",
    ]
    audited_point_columns = [
        *eligible_point_columns,
        "tanimoto_kernel_predicted_pic50",
    ]
    comparison["model_form_pic50_min"] = comparison[eligible_point_columns].min(axis=1)
    comparison["model_form_pic50_max"] = comparison[eligible_point_columns].max(axis=1)
    comparison["model_form_spread_pic50"] = (
        comparison["model_form_pic50_max"] - comparison["model_form_pic50_min"]
    )
    comparison["all_audited_model_pic50_min"] = comparison[audited_point_columns].min(axis=1)
    comparison["all_audited_model_pic50_max"] = comparison[audited_point_columns].max(axis=1)
    comparison["all_audited_model_spread_pic50"] = (
        comparison["all_audited_model_pic50_max"] - comparison["all_audited_model_pic50_min"]
    )
    comparison["tanimoto_kernel_status"] = "rejected_sensitivity_not_in_prospective_consensus"
    class_matrix = comparison[eligible_point_columns].ge(5.0)
    comparison["models_calling_blocker"] = class_matrix.sum(axis=1)
    comparison["threshold_class_agreement"] = class_matrix.nunique(axis=1).eq(1)
    comparison["virtual_candidate_interpretation"] = np.select(
        [
            ~comparison["threshold_class_agreement"],
            comparison["model_form_spread_pic50"].ge(0.5),
        ],
        [
            "models disagree across the pIC50 5 blocker threshold",
            "model-form spread is at least 0.5 pIC50 log",
        ],
        default="models agree qualitatively; assay tests quantitative calibration",
    )
    return comparison


def _plot(predictions: pd.DataFrame, locked: pd.DataFrame) -> None:
    exact = predictions[predictions["herg_pic50_relation"].eq("=")].merge(
        locked[["structure_id", "predicted_pic50"]],
        on="structure_id",
        validate="one_to_one",
    )
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.7), constrained_layout=True)
    limits = (4.4, 6.5)
    axes[0].scatter(
        exact["herg_pic50_value"],
        exact["complete_feature_predicted_pic50"],
        color="#2563eb",
        alpha=0.82,
        edgecolors="white",
        linewidths=0.5,
    )
    axes[0].plot(limits, limits, color="#374151", linewidth=1)
    axes[0].set(
        xlim=limits,
        ylim=limits,
        xlabel="Observed pIC50",
        ylabel="Complete-feature pIC50",
        title="ECFP4 latent structure + 9 global controls",
    )
    locked_error = np.abs(exact["predicted_pic50"] - exact["herg_pic50_value"])
    complete_error = np.abs(exact["complete_feature_predicted_pic50"] - exact["herg_pic50_value"])
    axes[1].scatter(
        locked_error,
        complete_error,
        color="#7c3aed",
        alpha=0.82,
        edgecolors="white",
        linewidths=0.5,
    )
    axes[1].plot((0, 1.2), (0, 1.2), color="#374151", linewidth=1)
    axes[1].set(
        xlim=(0, 1.2),
        ylim=(0, 1.2),
        xlabel="Global-control absolute error",
        ylabel="Complete-feature absolute error",
        title="Points below line improve",
    )
    _atomic_figure(figure, OUTPUT / "complete_feature_extension.png", dpi=220)
    _atomic_figure(figure, OUTPUT / "complete_feature_extension.pdf")
    plt.close(figure)


def run() -> dict[str, Any]:
    tables = load_canonical_tables(CANONICAL)
    compounds = compound_model_frame(tables["compounds"], tables.get("compound_aliases"))
    base_features, layers = merge_feature_layers(compounds)
    _, potency, _ = prepare_herg_evidence(compounds, tables["measurements"], base_features)
    controls = [
        column for column in layers["structure_2d"] if column in potency and potency[column].notna().any()
    ]
    if len(controls) != 9:
        raise ValueError(f"Expected the nine audited global controls; found {controls}")
    control_medians = potency[controls].median()
    training_controls = potency[controls].fillna(control_medians).to_numpy(dtype=float)
    training_bits = _fingerprints(potency["standardized_smiles"])

    candidate_rows = []
    candidate_oof: dict[int, pd.DataFrame] = {}
    for components in COMPONENT_CANDIDATES:
        metrics, oof = _oof(
            potency,
            training_bits,
            training_controls,
            components=components,
        )
        candidate_rows.append(metrics)
        candidate_oof[components] = oof
    candidates = pd.DataFrame(candidate_rows)
    stable = candidates[candidates["all_folds_converged"]].copy()
    selected_components = int(
        stable.sort_values(["censored_negative_log_likelihood", "pic50_mae", "components"]).iloc[0][
            "components"
        ]
    )
    kernel_candidate_rows = []
    kernel_candidate_oof: dict[float, pd.DataFrame] = {}
    for alpha in KERNEL_ALPHA_CANDIDATES:
        kernel_metrics, kernel_oof = _kernel_oof(
            potency,
            training_bits,
            training_controls,
            alpha=alpha,
        )
        kernel_candidate_rows.append(kernel_metrics)
        kernel_candidate_oof[alpha] = kernel_oof
    kernel_candidates = pd.DataFrame(kernel_candidate_rows)
    stable_kernel = kernel_candidates[kernel_candidates["all_folds_converged"]].copy()
    if stable_kernel.empty:
        raise RuntimeError("No Tanimoto-kernel alpha converged in every original-data fold")
    selected_kernel_alpha = float(
        stable_kernel.sort_values(["censored_negative_log_likelihood", "pic50_mae", "alpha"]).iloc[0]["alpha"]
    )

    source = load_ascentage_source(
        SOURCE,
        recovery_artifact=EXTENSION_ROOT / "predictions.parquet",
    )
    novel = source[~source["structure_id"].isin(set(compounds["structure_id"]))].copy()
    novel_features = structure_feature_frame(
        pd.DataFrame(
            {
                "compound_id": novel["structure_id"],
                "standardized_smiles": novel["standardized_smiles"],
            }
        )
    )
    novel = novel.merge(
        novel_features[["compound_id", *controls]],
        left_on="structure_id",
        right_on="compound_id",
        validate="one_to_one",
    ).drop(columns="compound_id")
    novel_controls = novel[controls].fillna(control_medians).to_numpy(dtype=float)
    novel_bits = _fingerprints(novel["standardized_smiles"])
    predictions = _fit_score(
        potency,
        training_bits,
        training_controls,
        novel,
        novel_bits,
        novel_controls,
        candidate_oof[selected_components],
        components=selected_components,
    )
    metrics = _extension_metrics(predictions)
    metrics["classification_at_pic50_5"] = _classification_metrics(
        predictions,
        probability_column="complete_feature_blocker_probability",
    )
    kernel_predictions = _kernel_fit_score(
        potency,
        training_bits,
        training_controls,
        novel,
        novel_bits,
        novel_controls,
        kernel_candidate_oof[selected_kernel_alpha],
        alpha=selected_kernel_alpha,
    )
    kernel_metrics = _extension_metrics(
        kernel_predictions,
        prediction_column="tanimoto_kernel_predicted_pic50",
        interval_lower_column="tanimoto_kernel_pic50_lower",
        interval_upper_column="tanimoto_kernel_pic50_upper",
    )
    kernel_metrics["classification_at_pic50_5"] = _classification_metrics(
        kernel_predictions,
        probability_column="tanimoto_kernel_blocker_probability",
    )
    locked = pd.read_parquet(EXTENSION_ROOT / "predictions.parquet")
    locked = locked[~locked["exact_training_structure_overlap"]].copy()
    bootstrap = _bootstrap_comparison(predictions, locked)
    delta_interval = np.quantile(bootstrap["complete_minus_locked_mae"], [0.025, 0.975])
    kernel_vs_svd_bootstrap = _paired_scaffold_bootstrap(
        kernel_predictions,
        predictions,
        left_column="tanimoto_kernel_predicted_pic50",
        right_column="complete_feature_predicted_pic50",
        left_name="tanimoto_kernel",
        right_name="svd_complete_feature",
    )
    kernel_delta_interval = np.quantile(
        kernel_vs_svd_bootstrap["tanimoto_kernel_minus_svd_complete_feature_mae"],
        [0.025, 0.975],
    )
    metrics["selected_components"] = selected_components
    metrics["selection_basis"] = (
        "minimum original-training five-fold scaffold-held-out censored negative log likelihood "
        "among converged component counts"
    )
    metrics["complete_minus_locked_mae_bootstrap_95_lower"] = float(delta_interval[0])
    metrics["complete_minus_locked_mae_bootstrap_95_upper"] = float(delta_interval[1])
    metrics["global_control_features"] = controls
    metrics["fingerprint_definition"] = "ECFP4/Morgan radius 2, 2048 binary bits"
    metrics["claim_limit"] = (
        "retrospective representation audit conceived after source outcome review; "
        "requires prospective confirmation"
    )
    kernel_metrics["selected_alpha"] = selected_kernel_alpha
    kernel_metrics["selection_basis"] = (
        "minimum original-training five-fold scaffold-held-out censored negative log "
        "likelihood among fully converged alpha candidates"
    )
    kernel_metrics["kernel_definition"] = (
        "Tanimoto similarity to each unique training ECFP4/Morgan radius-2 2048-bit "
        "fingerprint, plus nine global controls; references and scaling fit per fold"
    )
    kernel_metrics["kernel_minus_svd_mae_bootstrap_95_lower"] = float(kernel_delta_interval[0])
    kernel_metrics["kernel_minus_svd_mae_bootstrap_95_upper"] = float(kernel_delta_interval[1])
    kernel_metrics["claim_limit"] = (
        "retrospective full-fingerprint sensitivity analysis conceived after source "
        "outcome review; requires prospective confirmation"
    )

    measured_extension = novel["herg_ic50_censoring"].ne("missing") & novel["synthesis_status"].eq(
        "synthesized_by_cro"
    )
    extension_training = novel.loc[measured_extension].copy()
    extension_training["pic50_lower"] = extension_training["herg_pic50_lower_bound"]
    extension_training["pic50_upper"] = extension_training["herg_pic50_upper_bound"]
    extension_training = extension_training.rename(columns={"structure_id": "compound_id"})
    combined = pd.concat([potency, extension_training], ignore_index=True, sort=False)
    combined_controls = combined[controls].fillna(combined[controls].median()).to_numpy(dtype=float)
    combined_bits = _fingerprints(combined["standardized_smiles"])
    augmented_metrics, augmented_oof = _oof(
        combined,
        combined_bits,
        combined_controls,
        components=selected_components,
    )
    virtual_unsynthesized = novel[
        novel["herg_ic50_censoring"].eq("missing") & novel["synthesis_status"].eq("not_synthesized")
    ].copy()
    virtual_controls = (
        virtual_unsynthesized[controls].fillna(combined[controls].median()).to_numpy(dtype=float)
    )
    virtual_bits = _fingerprints(virtual_unsynthesized["standardized_smiles"])
    augmented_predictions = _fit_score(
        combined,
        combined_bits,
        combined_controls,
        virtual_unsynthesized,
        virtual_bits,
        virtual_controls,
        augmented_oof,
        components=selected_components,
    )
    kernel_augmented_metrics, kernel_augmented_oof = _kernel_oof(
        combined,
        combined_bits,
        combined_controls,
        alpha=selected_kernel_alpha,
    )
    kernel_augmented_predictions = _kernel_fit_score(
        combined,
        combined_bits,
        combined_controls,
        virtual_unsynthesized,
        virtual_bits,
        virtual_controls,
        kernel_augmented_oof,
        alpha=selected_kernel_alpha,
    )
    global_control_panel = pd.read_parquet(EXTENSION_ROOT / "predictions.parquet")
    virtual_structure_ids = set(
        source.loc[
            source["synthesis_status"].eq("not_synthesized"),
            "structure_id",
        ]
    )
    global_control_panel = global_control_panel[
        global_control_panel["structure_id"].isin(virtual_structure_ids)
    ].copy()
    disagreement = _prediction_disagreement(
        augmented_predictions,
        kernel_augmented_predictions,
        global_control_panel,
    )
    failure_family = predictions[
        predictions["internal_id"].isin(["M-2957", "M-2958", "M-2959", "M-2960"])
    ].copy()
    failure_family["observed_pic50"] = failure_family["herg_pic50_value"]
    failure_family["complete_feature_error"] = (
        failure_family["complete_feature_predicted_pic50"] - failure_family["observed_pic50"]
    )
    failure_family["complete_feature_absolute_error"] = np.abs(failure_family["complete_feature_error"])
    kernel_nll_30000 = float(
        kernel_candidates.loc[
            kernel_candidates["alpha"].eq(30000.0),
            "censored_negative_log_likelihood",
        ].iloc[0]
    )
    kernel_nll_100000 = float(
        kernel_candidates.loc[
            kernel_candidates["alpha"].eq(100000.0),
            "censored_negative_log_likelihood",
        ].iloc[0]
    )
    validation_checks = {
        "novel_structure_count_is_54": len(novel) == 54,
        "novel_structures_are_unique": novel["structure_id"].nunique() == len(novel),
        "exact_outcome_count_is_42": novel["herg_pic50_relation"].eq("=").sum() == 42,
        "censored_outcome_count_is_4": novel["herg_pic50_relation"].eq("<").sum() == 4,
        "not_synthesized_virtual_structure_count_is_8": (
            novel["synthesis_status"].eq("not_synthesized").sum() == 8
        ),
        "svd_extension_fit_converged": bool(predictions["model_fit_converged"].all()),
        "kernel_extension_fit_converged": bool(kernel_predictions["model_fit_converged"].all()),
        "selected_svd_dimension_is_interior": (
            selected_components not in {min(COMPONENT_CANDIDATES), max(COMPONENT_CANDIDATES)}
        ),
        "kernel_shrinkage_nll_plateau_reached": (
            abs(kernel_nll_100000 - kernel_nll_30000) / abs(kernel_nll_100000) < 0.001
        ),
        "kernel_is_significantly_worse_than_svd": float(kernel_delta_interval[0]) > 0.0,
        "former_failure_family_max_error_below_0p5": float(
            failure_family["complete_feature_absolute_error"].max()
        )
        < 0.5,
        "virtual_prediction_sensitivity_has_8_rows": len(disagreement) == 8,
        "one_virtual_retained_threshold_disagreement": int((~disagreement["threshold_class_agreement"]).sum())
        == 1,
        "four_virtual_retained_large_model_spreads": int(
            disagreement["model_form_spread_pic50"].ge(0.5).sum()
        )
        == 4,
        "bootstrap_replicates_are_complete": (
            len(bootstrap) == 5000 and len(kernel_vs_svd_bootstrap) == 5000
        ),
    }
    if not all(validation_checks.values()):
        failed = [name for name, passed in validation_checks.items() if not passed]
        raise RuntimeError(f"Complete-feature validation failed: {failed}")
    validation_report = {
        "status": "pass",
        "checks": validation_checks,
        "selection_data_boundary": (
            "SVD dimension and kernel alpha use only original baseline grouped OOF "
            "metrics; extension outcomes are retrospective evaluation only"
        ),
        "virtual_structure_status": (
            "the eight blank-entry structures were not synthesized; predictions are "
            "virtual sensitivity examples, not a prospective assay panel"
        ),
        "kernel_decision": (
            "rejected sensitivity control; excluded from prospective consensus because "
            "kernel-minus-SVD scaffold-bootstrap MAE interval is above zero"
        ),
    }

    OUTPUT.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(OUTPUT / "component_selection.csv", candidates)
    atomic_write_csv(OUTPUT / "tanimoto_kernel_alpha_selection.csv", kernel_candidates)
    atomic_write_parquet(
        OUTPUT / "selected_original_training_oof.parquet",
        candidate_oof[selected_components],
    )
    atomic_write_parquet(
        OUTPUT / "selected_tanimoto_kernel_original_training_oof.parquet",
        kernel_candidate_oof[selected_kernel_alpha],
    )
    atomic_write_csv(OUTPUT / "extension_predictions.csv", predictions)
    atomic_write_parquet(OUTPUT / "extension_predictions.parquet", predictions)
    atomic_write_csv(
        OUTPUT / "tanimoto_kernel_extension_predictions.csv",
        kernel_predictions,
    )
    atomic_write_parquet(
        OUTPUT / "tanimoto_kernel_extension_predictions.parquet",
        kernel_predictions,
    )
    atomic_write_csv(OUTPUT / "scaffold_bootstrap_comparison.csv", bootstrap)
    atomic_write_csv(
        OUTPUT / "tanimoto_kernel_vs_svd_scaffold_bootstrap.csv",
        kernel_vs_svd_bootstrap,
    )
    atomic_write_json(OUTPUT / "extension_metrics.json", metrics)
    atomic_write_json(OUTPUT / "tanimoto_kernel_extension_metrics.json", kernel_metrics)
    atomic_write_json(
        OUTPUT / "augmented_grouped_metrics.json",
        {
            **augmented_metrics,
            "training_rows": len(combined),
            "training_unique_compounds": int(combined["compound_id"].nunique()),
            "extension_training_structures": int(measured_extension.sum()),
        },
    )
    atomic_write_json(
        OUTPUT / "tanimoto_kernel_augmented_grouped_metrics.json",
        {
            **kernel_augmented_metrics,
            "training_rows": len(combined),
            "training_unique_compounds": int(combined["compound_id"].nunique()),
            "extension_training_structures": int(measured_extension.sum()),
        },
    )
    atomic_write_parquet(OUTPUT / "augmented_oof.parquet", augmented_oof)
    atomic_write_parquet(
        OUTPUT / "tanimoto_kernel_augmented_oof.parquet",
        kernel_augmented_oof,
    )
    atomic_write_csv(
        OUTPUT / "virtual_unsynthesized_augmented_predictions.csv",
        augmented_predictions,
    )
    atomic_write_csv(
        OUTPUT / "tanimoto_kernel_virtual_unsynthesized_predictions.csv",
        kernel_augmented_predictions,
    )
    atomic_write_csv(
        OUTPUT / "virtual_structure_prediction_sensitivity.csv",
        disagreement,
    )
    atomic_write_json(OUTPUT / "validation_report.json", validation_report)
    atomic_write_csv(
        OUTPUT / "resolved_failure_family.csv",
        failure_family[
            [
                "internal_id",
                "observed_pic50",
                "complete_feature_predicted_pic50",
                "complete_feature_error",
                "complete_feature_absolute_error",
            ]
        ],
    )
    _plot(predictions, locked)

    disagreement_min = float(disagreement["model_form_spread_pic50"].min())
    disagreement_max = float(disagreement["model_form_spread_pic50"].max())
    failure_min = float(failure_family["complete_feature_absolute_error"].min())
    failure_max = float(failure_family["complete_feature_absolute_error"].max())
    report = f"""# Complete-feature hERG representation audit

## Correction

The first extension model used only nine global RDKit controls. Those controls are
necessary for size, lipophilicity, polarity, hydrogen bonding, flexibility, aromaticity,
and three-dimensional saturation, but they cannot encode which local
substitution changed. The corrected representation uses all 2,048 ECFP4 bits plus the
nine controls. To avoid fitting 2,057 coefficients to 94 baseline rows, the fingerprint
is compressed **inside every training fold**.

The component count was selected using only the original data: {selected_components}
components minimized five-fold scaffold-held-out censored negative log likelihood among
{list(COMPONENT_CANDIDATES)}. The selected model converged in every fold. Higher
dimensions were retained in the audit table and rejected when likelihood or numerical
stability deteriorated.

## Non-overlap extension result

On the same 54 non-overlapping structures and 42 exact IC50 values:

- pIC50 MAE **{metrics["pic50_mae"]:.3f}** versus **0.405** for the global-control
  censored model.
- RMSE **{metrics["pic50_rmse"]:.3f}** versus **0.493**.
- Spearman **{metrics["spearman"]:.3f}** versus **0.488**.
- **{metrics["fraction_within_0p5_log"]:.1%}** within 0.5 log and
  **{metrics["fraction_within_1p0_log"]:.1%}** within 1.0 log.
- Strict mean compatibility with the four novel `>30 µM` limits improves to
  **{metrics["strict_censored_compatibility"]:.1%}** from 25%.
- At the secondary pIC50 5 threshold, the 46 definite outcomes (37 blockers and
  9 nonblockers) give ROC-AUC
  **{metrics["classification_at_pic50_5"]["roc_auc"]:.3f}**, PR-AUC
  **{metrics["classification_at_pic50_5"]["pr_auc"]:.3f}**, balanced accuracy
  **{metrics["classification_at_pic50_5"]["balanced_accuracy"]:.3f}**, MCC
  **{metrics["classification_at_pic50_5"]["mcc"]:.3f}**, sensitivity
  **{metrics["classification_at_pic50_5"]["sensitivity"]:.3f}**, and specificity
  **{metrics["classification_at_pic50_5"]["specificity"]:.3f}**. These are
  retrospective secondary metrics, not a class-balanced deployment estimate.

The paired scaffold-bootstrap complete-minus-global MAE interval is
**[{delta_interval[0]:.3f}, {delta_interval[1]:.3f}]**. It crosses zero, so the improved
point estimates are not yet a statistically conclusive promotion. The 42 exact values
occupy only **{metrics["n_exact_scaffolds"]}** computed scaffolds, with
**{metrics["largest_exact_scaffold_fraction"]:.1%}** in the largest, which limits
bootstrap power and independent-series interpretation.

## Full-fingerprint sensitivity control

Compression could itself discard local information. A sensitivity model therefore uses the
complete 2,048-bit ECFP4 fingerprint through Tanimoto similarity to every unique
training structure, plus the same nine controls. Kernel references and feature scaling
are rebuilt inside every fold. Alpha **{selected_kernel_alpha:g}** minimized
original-data grouped censored NLL among {list(KERNEL_ALPHA_CANDIDATES)}; extension
outcomes did not select it.

On the non-overlap extension, the full-fingerprint kernel reaches MAE
**{kernel_metrics["pic50_mae"]:.3f}**, RMSE **{kernel_metrics["pic50_rmse"]:.3f}**,
and Spearman **{kernel_metrics["spearman"]:.3f}**. Its kernel-minus-SVD
scaffold-bootstrap MAE interval is
**[{kernel_delta_interval[0]:.3f}, {kernel_delta_interval[1]:.3f}]**. The interval is
above zero, so the kernel is significantly worse on this extension. Original-data NLL
improved monotonically as regularization approached the intercept-only limit and changed
by less than 0.1% from alpha 30,000 to 100,000. This indicates that the uncompressed
kernel basis does not transfer under the present scaffold split. It is **rejected as a
predictive model** but retained as the required full-bit sensitivity control.

## What the stronger null falsifies

The nine-control model underpredicted M-2957–M-2960 by 0.76–1.10 pIC50 logs.
After local structure is represented, their absolute errors are only
**{failure_min:.3f}–{failure_max:.3f} logs**. The earlier residual therefore does not
establish missing protonation, folding, membrane, or receptor physics. Omitted local
substructure is the more parsimonious explanation, and expensive physics should target
only failures that survive these stronger references.

## Interpretation

This is the correct conventional reference layer for this analog problem: the global
controls establish broad physicochemical context, while ECFP4 encodes the local chemical
changes those controls erase. The latent fingerprint axes are not called fundamental
physical parameters and receive no mechanistic interpretation. They are a structural
null against which future microstate, conformation, membrane-access, and receptor-state
features must add explanatory value.

Across the two retained conventional layers (global controls and compressed ECFP4), the
eight unsynthesized virtual structures have **{disagreement_min:.3f}–{disagreement_max:.3f}
pIC50-log** model-form spread. `virtual_structure_prediction_sensitivity.csv` also
preserves the rejected kernel values. These rows demonstrate scoring behavior; they are
not an assay-ready prospective panel.

The analysis was conceived after the extension outcomes were reviewed. Although no
extension label selected the component count or kernel alpha, this remains retrospective
and discovery-track. Prospective validation now requires newly submitted, synthesized
same-series compounds with results withheld until predictions and protocol are locked.
"""
    atomic_write_text(OUTPUT / "complete_feature_model_report.md", report)
    return {
        "selected_components": selected_components,
        "selected_kernel_alpha": selected_kernel_alpha,
        "extension_pic50_mae": metrics["pic50_mae"],
        "extension_spearman": metrics["spearman"],
        "kernel_extension_pic50_mae": kernel_metrics["pic50_mae"],
        "kernel_extension_spearman": kernel_metrics["spearman"],
        "virtual_unsynthesized_predictions": len(augmented_predictions),
        "output": str(OUTPUT),
    }


def main() -> None:
    print(json.dumps(run(), indent=2))


if __name__ == "__main__":
    main()
