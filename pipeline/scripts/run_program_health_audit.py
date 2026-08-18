#!/usr/bin/env python3
"""Publication-oriented health audit for the mechanistic PK/hERG program.

This script extends, rather than replaces, the existing skeptical-review
audit.  It uses fixed retrospective predictions and prespecified conventional
controls to quantify:

* scaffold-bootstrap uncertainty and advantage over trivial/local baselines;
* hERG threshold, censoring, representation-seed, and calibration sensitivity;
* PK fold-error, interval-efficiency, and chemical-group learning behavior;
* the amount and type of additional independent data likely to be informative;
* which mechanistic/physics claims remain gated rather than model inputs.

No production molecular simulation is launched and no extension outcome is
used to select a representation, hyperparameter, or compound subset.
"""

from __future__ import annotations

import json
import math
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/menin-program-health-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/menin-program-health-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components
from scipy.stats import norm, spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    matthews_corrcoef,
    r2_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "pipeline/scripts"))

from menin_discovery.features import fingerprint_matrix  # noqa: E402
from menin_discovery.research_common import (  # noqa: E402
    atomic_write_csv,
    atomic_write_json,
    atomic_write_parquet,
    atomic_write_text,
)
from menin_discovery.research_local_analysis import (  # noqa: E402
    _load_balanced_predictions,
)
from menin_discovery.research_modeling import (  # noqa: E402
    CensoredGaussianRidge,
    merge_feature_layers,
    structure_feature_frame,
)
from menin_discovery.research_reviewer_audit import (  # noqa: E402
    _source_collapsed_pk_frames,
)
from menin_discovery.research_workflows import (  # noqa: E402
    compound_model_frame,
    load_canonical_tables,
    prepare_herg_evidence,
)
from run_ascentage_complete_feature_model import (  # noqa: E402
    ALPHA,
    _fingerprints,
    _project,
)

SEED = 20260729
BOOTSTRAPS = 5000
PERMUTATIONS = 2000
OUTPUT = ROOT / "research/reports/pk_herg/program_health"
CANONICAL = ROOT / "research/data/pk_herg/canonical"
EXTENSION_ROOT = ROOT / "research/reports/pk_herg/ascentage_herg_extension"
COMPLETE_ROOT = EXTENSION_ROOT / "complete_feature_model"
REVIEW_ROOT = ROOT / "research/reports/pk_herg/reviewer_audit"
LOCAL_ROOT = ROOT / "research/reports/pk_herg/local_m3"


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


def _ece(labels: np.ndarray, probabilities: np.ndarray, bins: int = 5) -> float:
    labels = np.asarray(labels, dtype=int)
    probabilities = np.clip(np.asarray(probabilities, dtype=float), 1e-7, 1 - 1e-7)
    edges = np.linspace(0, 1, bins + 1)
    ids = np.clip(np.digitize(probabilities, edges[1:-1]), 0, bins - 1)
    value = 0.0
    for bin_id in range(bins):
        mask = ids == bin_id
        if np.any(mask):
            value += float(mask.mean()) * abs(float(labels[mask].mean()) - float(probabilities[mask].mean()))
    return float(value)


def _wilson(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return float("nan"), float("nan")
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def _regression_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    finite = np.isfinite(observed) & np.isfinite(predicted)
    observed, predicted = observed[finite], predicted[finite]
    error = predicted - observed
    absolute = np.abs(error)
    folds = np.power(10.0, absolute)
    return {
        "n": float(len(observed)),
        "mae": float(np.mean(absolute)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mean_signed_error": float(np.mean(error)),
        "r2": float(r2_score(observed, predicted)) if len(observed) >= 2 else float("nan"),
        "spearman": _safe_spearman(observed, predicted),
        "median_fold_error": float(np.median(folds)),
        "absolute_average_fold_error": float(np.mean(folds)),
        "fraction_within_0p5_log": float(np.mean(absolute <= 0.5)),
        "fraction_within_1p0_log": float(np.mean(absolute <= 1.0)),
        "fraction_within_2fold": float(np.mean(folds <= 2.0)),
        "fraction_within_3fold": float(np.mean(folds <= 3.0)),
    }


def _classification_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=int)
    probabilities = np.clip(np.asarray(probabilities, dtype=float), 1e-7, 1 - 1e-7)
    predictions = (probabilities >= 0.5).astype(int)
    blockers = labels == 1
    nonblockers = labels == 0
    both = blockers.any() and nonblockers.any()
    tp = int(np.sum(predictions[blockers] == 1))
    tn = int(np.sum(predictions[nonblockers] == 0))
    fp = int(np.sum(predictions[nonblockers] == 1))
    fn = int(np.sum(predictions[blockers] == 0))
    sensitivity = tp / int(blockers.sum()) if blockers.any() else float("nan")
    specificity = tn / int(nonblockers.sum()) if nonblockers.any() else float("nan")
    sens_lower, sens_upper = _wilson(tp, int(blockers.sum()))
    spec_lower, spec_upper = _wilson(tn, int(nonblockers.sum()))
    return {
        "n": float(len(labels)),
        "n_blockers": float(blockers.sum()),
        "n_nonblockers": float(nonblockers.sum()),
        "roc_auc": float(roc_auc_score(labels, probabilities)) if both else float("nan"),
        "pr_auc": float(average_precision_score(labels, probabilities)) if both else float("nan"),
        "balanced_accuracy": (float(balanced_accuracy_score(labels, predictions)) if both else float("nan")),
        "mcc": float(matthews_corrcoef(labels, predictions)) if both else float("nan"),
        "sensitivity": float(sensitivity),
        "sensitivity_wilson_lower_95": float(sens_lower),
        "sensitivity_wilson_upper_95": float(sens_upper),
        "specificity": float(specificity),
        "specificity_wilson_lower_95": float(spec_lower),
        "specificity_wilson_upper_95": float(spec_upper),
        "brier": float(brier_score_loss(labels, probabilities)),
        "ece_5_bin": _ece(labels, probabilities, bins=5),
        "log_loss": float(log_loss(labels, probabilities, labels=[0, 1])),
        "true_positive": float(tp),
        "true_negative": float(tn),
        "false_positive": float(fp),
        "false_negative": float(fn),
    }


def _group_bootstrap(
    frame: pd.DataFrame,
    metric: Callable[[pd.DataFrame], dict[str, float]],
    *,
    group_column: str = "scaffold",
    replicates: int = BOOTSTRAPS,
    seed: int = SEED,
) -> tuple[dict[str, float], pd.DataFrame]:
    work = frame.dropna(subset=[group_column]).copy()
    groups = sorted(work[group_column].astype(str).unique())
    if len(groups) < 2:
        raise ValueError(f"Need at least two {group_column} groups")
    pieces = {group: work[work[group_column].astype(str).eq(group)] for group in groups}
    point = metric(work)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float]] = []
    for replicate in range(replicates):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        sample = pd.concat([pieces[group] for group in sampled], ignore_index=True)
        rows.append({"replicate": float(replicate), **metric(sample)})
    bootstrap = pd.DataFrame(rows)
    ci_metrics = {
        "mae",
        "rmse",
        "mean_signed_error",
        "r2",
        "spearman",
        "median_fold_error",
        "absolute_average_fold_error",
        "fraction_within_0p5_log",
        "fraction_within_1p0_log",
        "fraction_within_2fold",
        "fraction_within_3fold",
        "interval_coverage",
        "interval_mean_width_log10",
        "interval_width_to_observed_range",
        "interval_score_90",
        "roc_auc",
        "pr_auc",
        "balanced_accuracy",
        "mcc",
        "sensitivity",
        "specificity",
        "brier",
        "ece_5_bin",
        "log_loss",
    }
    for name in list(point):
        if name not in ci_metrics:
            continue
        values = pd.to_numeric(bootstrap.get(name), errors="coerce").dropna()
        if len(values):
            point[f"{name}_lower_95"] = float(values.quantile(0.025))
            point[f"{name}_upper_95"] = float(values.quantile(0.975))
    return point, bootstrap


def _paired_group_bootstrap(
    frame: pd.DataFrame,
    *,
    left: str,
    right: str,
    observed: str,
    group_column: str = "scaffold",
    replicates: int = BOOTSTRAPS,
    seed: int = SEED,
) -> dict[str, float]:
    work = frame[[observed, left, right, group_column]].dropna().copy()
    work["_left_error"] = np.abs(work[left] - work[observed])
    work["_right_error"] = np.abs(work[right] - work[observed])
    groups = sorted(work[group_column].astype(str).unique())
    pieces = {group: work[work[group_column].astype(str).eq(group)] for group in groups}
    rng = np.random.default_rng(seed)
    deltas = np.empty(replicates, dtype=float)
    for replicate in range(replicates):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        sample = pd.concat([pieces[group] for group in sampled], ignore_index=True)
        deltas[replicate] = float((sample["_left_error"] - sample["_right_error"]).mean())
    point = float((work["_left_error"] - work["_right_error"]).mean())
    return {
        "n": float(len(work)),
        "n_groups": float(len(groups)),
        "mean_delta_mae": point,
        "delta_mae_lower_95": float(np.quantile(deltas, 0.025)),
        "delta_mae_upper_95": float(np.quantile(deltas, 0.975)),
        "bootstrap_probability_left_better": float(np.mean(deltas < 0)),
    }


def _load_core() -> dict[str, Any]:
    tables = load_canonical_tables(CANONICAL)
    compounds = compound_model_frame(tables["compounds"], tables.get("compound_aliases"))
    base_features, layers = merge_feature_layers(compounds)
    _, potency, inhibition = prepare_herg_evidence(
        compounds,
        tables["measurements"],
        base_features,
    )
    controls = [
        column for column in layers["structure_2d"] if column in potency and potency[column].notna().any()
    ]
    if len(controls) != 9:
        raise ValueError(f"Expected nine audited conventional controls, observed {controls}")
    extension_context = pd.read_parquet(EXTENSION_ROOT / "predictions.parquet")
    complete = pd.read_parquet(COMPLETE_ROOT / "extension_predictions.parquet")
    novel = complete.merge(
        extension_context[
            [
                "structure_id",
                "standardized_smiles",
                "max_train_tanimoto",
                "applicability_domain",
                "predicted_pic50",
                "predicted_pic50_lower",
                "predicted_pic50_upper",
                "audit_logp",
                "audit_tpsa",
                "audit_rotatable_bonds",
                "audit_fraction_csp3",
                "audit_formal_charge",
            ]
        ],
        on="structure_id",
        how="left",
        validate="one_to_one",
    )
    if len(novel) != 54 or novel["standardized_smiles"].isna().any():
        raise ValueError("Extension reconstruction failed the expected 54-structure boundary")
    novel_feature_frame = structure_feature_frame(
        pd.DataFrame(
            {
                "compound_id": novel["structure_id"],
                "standardized_smiles": novel["standardized_smiles"],
            }
        )
    )
    novel = novel.merge(
        novel_feature_frame[["compound_id", *controls]],
        left_on="structure_id",
        right_on="compound_id",
        how="left",
        validate="one_to_one",
    ).drop(columns="compound_id")
    return {
        "tables": tables,
        "compounds": compounds,
        "potency": potency.reset_index(drop=True),
        "inhibition": inhibition,
        "controls": controls,
        "novel": novel,
    }


def _tanimoto(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    intersection = left @ right.T
    denominator = left.sum(axis=1, keepdims=True) + right.sum(axis=1, keepdims=True).T - intersection
    return np.divide(
        intersection,
        denominator,
        out=np.zeros_like(intersection, dtype=float),
        where=denominator > 0,
    )


def _herg_extension_baselines(core: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    potency = core["potency"]
    novel = core["novel"].copy()
    train_exact = potency[
        np.isfinite(potency["pic50_lower"])
        & np.isfinite(potency["pic50_upper"])
        & np.isclose(potency["pic50_lower"], potency["pic50_upper"])
    ].copy()
    train_exact = train_exact.groupby(["compound_id", "standardized_smiles"], as_index=False).agg(
        observed=("pic50_lower", "mean"), scaffold=("scaffold", "first")
    )
    train_bits = _fingerprints(train_exact["standardized_smiles"])
    score_bits = _fingerprints(novel["standardized_smiles"])
    similarities = _tanimoto(score_bits, train_bits)
    train_values = train_exact["observed"].to_numpy(dtype=float)
    novel["train_mean"] = float(train_values.mean())
    for neighbors in (1, 3, 5):
        count = min(neighbors, len(train_exact))
        nearest = np.argpartition(-similarities, kth=count - 1, axis=1)[:, :count]
        weights = np.maximum(np.take_along_axis(similarities, nearest, axis=1), 1e-6)
        values = train_values[nearest]
        novel[f"morgan_{neighbors}nn"] = np.sum(weights * values, axis=1) / np.sum(
            weights,
            axis=1,
        )
    novel["nearest_exact_train_tanimoto"] = similarities.max(axis=1)
    novel["equal_weight_consensus"] = 0.5 * (
        novel["predicted_pic50"] + novel["complete_feature_predicted_pic50"]
    )
    novel = novel.rename(
        columns={
            "herg_pic50_value": "observed",
            "predicted_pic50": "global_controls",
            "complete_feature_predicted_pic50": "complete_feature",
        }
    )
    exact = novel[novel["herg_pic50_relation"].eq("=")].copy()
    model_columns = [
        "train_mean",
        "morgan_1nn",
        "morgan_3nn",
        "morgan_5nn",
        "global_controls",
        "equal_weight_consensus",
        "complete_feature",
    ]
    summary_rows: list[dict[str, Any]] = []
    bootstrap_tables: list[pd.DataFrame] = []
    for index, model in enumerate(model_columns):
        point, bootstrap = _group_bootstrap(
            exact,
            lambda sample, column=model: _regression_metrics(
                sample["observed"].to_numpy(),
                sample[column].to_numpy(),
            ),
            seed=SEED + index,
        )
        summary_rows.append(
            {
                "model": model,
                "n_exact": int(len(exact)),
                "n_scaffolds": int(exact["scaffold"].nunique()),
                **point,
                "selection_role": (
                    "retained_representation_selected_on_original_training_only"
                    if model == "complete_feature"
                    else "post_outcome_fixed_equal_weight_sensitivity_not_promoted"
                    if model == "equal_weight_consensus"
                    else "fixed_comparator_no_extension_selection"
                ),
            }
        )
        bootstrap.insert(1, "model", model)
        bootstrap_tables.append(bootstrap)
    for index, comparator in enumerate(model_columns[:-1]):
        delta = _paired_group_bootstrap(
            exact,
            left="complete_feature",
            right=comparator,
            observed="observed",
            seed=SEED + 100 + index,
        )
        summary_rows.append(
            {
                "model": f"complete_feature_minus_{comparator}",
                "n_exact": int(len(exact)),
                "n_scaffolds": int(exact["scaffold"].nunique()),
                **delta,
                "selection_role": "paired_scaffold_bootstrap_comparison",
            }
        )
    return pd.DataFrame(summary_rows), pd.concat(bootstrap_tables, ignore_index=True)


def _interval_score(
    observed: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    alpha: float = 0.10,
) -> np.ndarray:
    observed = np.asarray(observed, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    below = np.maximum(lower - observed, 0.0)
    above = np.maximum(observed - upper, 0.0)
    return (upper - lower) + (2.0 / alpha) * (below + above)


def _herg_interval_audit(core: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    novel = core["novel"]
    exact = novel[novel["herg_pic50_relation"].eq("=")].copy()
    exact = exact.rename(columns={"herg_pic50_value": "observed"})
    model_definitions = {
        "global_controls": ("predicted_pic50_lower", "predicted_pic50_upper"),
        "complete_feature": (
            "complete_feature_pic50_lower",
            "complete_feature_pic50_upper",
        ),
    }
    summary_rows: list[dict[str, Any]] = []
    conditional_rows: list[dict[str, Any]] = []
    exact["similarity_quartile"] = pd.qcut(
        exact["max_train_tanimoto"].rank(method="first"),
        q=4,
        labels=["Q1_lowest", "Q2", "Q3", "Q4_highest"],
    ).astype(str)
    for index, (model, (lower_column, upper_column)) in enumerate(model_definitions.items()):

        def metric(
            sample: pd.DataFrame,
            lower: str = lower_column,
            upper: str = upper_column,
        ) -> dict[str, float]:
            covered = (sample["observed"] >= sample[lower]) & (sample["observed"] <= sample[upper])
            width = sample[upper] - sample[lower]
            score = _interval_score(
                sample["observed"].to_numpy(),
                sample[lower].to_numpy(),
                sample[upper].to_numpy(),
            )
            return {
                "interval_coverage": float(covered.mean()),
                "interval_mean_width_log10": float(width.mean()),
                "interval_width_to_observed_range": float(
                    width.mean() / max(float(sample["observed"].max() - sample["observed"].min()), 1e-12)
                ),
                "interval_score_90": float(score.mean()),
            }

        point, _ = _group_bootstrap(
            exact,
            metric,
            seed=SEED + 250 + index,
        )
        covered = (exact["observed"] >= exact[lower_column]) & (exact["observed"] <= exact[upper_column])
        empirical_radius_90 = float(
            np.quantile(
                np.abs(exact["observed"] - 0.5 * (exact[lower_column] + exact[upper_column])),
                0.90,
                method="higher",
            )
        )
        current_radius = float((0.5 * (exact[upper_column] - exact[lower_column])).mean())
        lower, upper = _wilson(int(covered.sum()), int(len(exact)))
        summary_rows.append(
            {
                "model": model,
                "n_exact": int(len(exact)),
                "n_scaffolds": int(exact["scaffold"].nunique()),
                "nominal_interval_level": 0.90,
                **point,
                "coverage_wilson_lower_95": lower,
                "coverage_wilson_upper_95": upper,
                "post_outcome_empirical_radius_90": empirical_radius_90,
                "current_mean_radius": current_radius,
                "current_to_post_outcome_radius_ratio": current_radius / max(empirical_radius_90, 1e-12),
                "interval_status": "retrospective_oof_radius_not_prospectively_calibrated",
            }
        )
        for quartile, group in exact.groupby("similarity_quartile", sort=True):
            conditional_rows.append(
                {
                    "model": model,
                    "stratum": quartile,
                    "n": int(len(group)),
                    "scaffolds": int(group["scaffold"].nunique()),
                    "similarity_min": float(group["max_train_tanimoto"].min()),
                    "similarity_max": float(group["max_train_tanimoto"].max()),
                    **metric(group),
                    "interpretation": "Descriptive conditional coverage; bins are too small for certification.",
                }
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(conditional_rows)


def _herg_extension_topology(core: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    exact = core["novel"][core["novel"]["herg_pic50_relation"].eq("=")].copy()
    exact = exact.rename(
        columns={
            "herg_pic50_value": "observed",
            "complete_feature_predicted_pic50": "complete_feature",
            "predicted_pic50": "global_controls",
        }
    )
    bits = _fingerprints(exact["standardized_smiles"])
    similarity = _tanimoto(bits, bits)
    topology_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    for index, threshold in enumerate((0.70, 0.75, 0.80, 0.85, 0.90, 0.95)):
        adjacency = (similarity >= threshold).astype(np.int8)
        group_count, labels = connected_components(adjacency, directed=False)
        counts = pd.Series(labels).value_counts()
        fractions = counts.to_numpy(dtype=float) / len(exact)
        topology_rows.append(
            {
                "tanimoto_threshold": threshold,
                "n_components": int(group_count),
                "largest_component_n": int(counts.max()),
                "largest_component_fraction": float(counts.max() / len(exact)),
                "singleton_components": int(counts.eq(1).sum()),
                "effective_component_count": float(1.0 / np.sum(fractions**2)),
                "interpretation": (
                    "Outcome-blind connectivity topology of the fixed extension set; "
                    "Dr. Aguilar identifies all structures as one medicinal-chemistry series."
                ),
            }
        )
        if group_count >= 2:
            group_column = f"component_{index}"
            work = exact.copy()
            work[group_column] = [f"C{label:03d}" for label in labels]
            delta = _paired_group_bootstrap(
                work,
                left="complete_feature",
                right="global_controls",
                observed="observed",
                group_column=group_column,
                seed=SEED + 350 + index,
            )
            comparison_rows.append(
                {
                    "group_definition": f"tanimoto_component_{threshold:.2f}",
                    "tanimoto_threshold": threshold,
                    **delta,
                    "uncertainty_scope": (
                        "Within the one confirmed medicinal-chemistry series; cannot estimate "
                        "between-series generalization."
                    ),
                }
            )
    return pd.DataFrame(topology_rows), pd.DataFrame(comparison_rows)


def _calibration_audit(core: dict[str, Any], threshold: float = 5.0) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = core["novel"].copy()
    exact = frame["herg_pic50_relation"].eq("=")
    censored = frame["herg_pic50_relation"].eq("<") & frame["herg_pic50_upper_bound"].le(threshold + 1e-12)
    frame = frame[exact | censored].copy()
    frame["label"] = np.where(
        frame["herg_pic50_relation"].eq("="),
        (frame["herg_pic50_value"] >= threshold).astype(int),
        0,
    )
    sigma = float(frame["complete_feature_oof_sigma"].iloc[0])
    frame["probability"] = norm.sf((threshold - frame["complete_feature_predicted_pic50"]) / sigma)
    frame["calibration_bin"] = pd.qcut(
        frame["probability"].rank(method="first"),
        q=5,
        labels=False,
    )
    rows: list[dict[str, Any]] = []
    for bin_id, group in frame.groupby("calibration_bin", sort=True):
        blockers = int(group["label"].sum())
        lower, upper = _wilson(blockers, len(group))
        rows.append(
            {
                "calibration_bin": int(bin_id) + 1,
                "n": int(len(group)),
                "mean_predicted_probability": float(group["probability"].mean()),
                "observed_blocker_fraction": float(group["label"].mean()),
                "observed_wilson_lower_95": lower,
                "observed_wilson_upper_95": upper,
                "probability_min": float(group["probability"].min()),
                "probability_max": float(group["probability"].max()),
            }
        )
    bins = pd.DataFrame(rows)
    prevalence = float(frame["label"].mean())
    weights = bins["n"] / len(frame)
    reliability = float(
        np.sum(weights * (bins["mean_predicted_probability"] - bins["observed_blocker_fraction"]) ** 2)
    )
    resolution = float(np.sum(weights * (bins["observed_blocker_fraction"] - prevalence) ** 2))
    uncertainty = prevalence * (1 - prevalence)
    metrics = _classification_metrics(frame["label"].to_numpy(), frame["probability"].to_numpy())
    summary = {
        "threshold_pic50": threshold,
        "n": int(len(frame)),
        "observed_prevalence": prevalence,
        "mean_predicted_probability": float(frame["probability"].mean()),
        "calibration_in_the_large_observed_minus_predicted": float(prevalence - frame["probability"].mean()),
        "brier_reliability": reliability,
        "brier_resolution": resolution,
        "brier_uncertainty": uncertainty,
        "binned_brier_decomposition_approximation": reliability - resolution + uncertainty,
        "brier_direct": metrics["brier"],
        "interpretation": (
            "Five equal-count bins are descriptive only; all probabilities use the "
            "original-training OOF residual scale and no extension recalibration."
        ),
    }
    return bins, summary


def _threshold_metrics(core: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    novel = core["novel"].copy()
    novel = novel.rename(
        columns={
            "herg_pic50_value": "observed",
            "complete_feature_predicted_pic50": "predicted",
        }
    )
    sigma = float(novel["complete_feature_oof_sigma"].iloc[0])
    thresholds = [
        (1.0, 6.0),
        (3.0, 6.0 - math.log10(3.0)),
        (10.0, 5.0),
        (30.0, 6.0 - math.log10(30.0)),
    ]
    summary_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[pd.DataFrame] = []
    for index, (ic50_um, threshold) in enumerate(thresholds):
        exact = novel["herg_pic50_relation"].eq("=")
        # An observed IC50 > C transforms to pIC50 < threshold(C), so equality
        # of the stored upper bound and decision threshold is already a
        # definite nonblocker (the underlying pIC50 is strictly lower).
        censored_nonblocker = novel["herg_pic50_relation"].eq("<") & novel["herg_pic50_upper_bound"].le(
            threshold + 1e-12
        )
        evaluable = exact | censored_nonblocker
        frame = novel[evaluable].copy()
        frame["label"] = np.where(
            frame["herg_pic50_relation"].eq("="),
            (frame["observed"] >= threshold).astype(int),
            0,
        )
        frame["probability"] = norm.sf((threshold - frame["predicted"]) / sigma)
        point, bootstrap = _group_bootstrap(
            frame,
            lambda sample: _classification_metrics(
                sample["label"].to_numpy(),
                sample["probability"].to_numpy(),
            ),
            seed=SEED + 300 + index,
        )
        summary_rows.append(
            {
                "ic50_threshold_um": ic50_um,
                "pic50_threshold": threshold,
                "n_exact": int(exact.sum()),
                "n_censored_definite_nonblockers": int(censored_nonblocker.sum()),
                "n_scaffolds": int(frame["scaffold"].nunique()),
                **point,
                "interpretation": (
                    "Retrospective threshold sensitivity; probabilities reuse the "
                    "original-training OOF residual scale and are not recalibrated "
                    "on extension outcomes."
                ),
            }
        )
        bootstrap.insert(1, "ic50_threshold_um", ic50_um)
        bootstrap_rows.append(bootstrap)
    return pd.DataFrame(summary_rows), pd.concat(bootstrap_rows, ignore_index=True)


def _fit_censored_extension(
    training: pd.DataFrame,
    scoring: pd.DataFrame,
    controls: list[str],
    *,
    seed: int,
    components: int = 8,
) -> tuple[np.ndarray, float, bool, float]:
    medians = training[controls].median()
    training_controls = training[controls].fillna(medians).to_numpy(dtype=float)
    scoring_controls = scoring[controls].fillna(medians).to_numpy(dtype=float)
    training_bits = _fingerprints(training["standardized_smiles"])
    scoring_bits = _fingerprints(scoring["standardized_smiles"])
    train_matrix, score_matrix, explained = _project(
        training_bits,
        training_controls,
        scoring_bits,
        scoring_controls,
        components=components,
        seed=seed,
    )
    model = CensoredGaussianRidge(alpha=ALPHA, maxiter=5000).fit(
        train_matrix,
        training["pic50_lower"].to_numpy(dtype=float),
        training["pic50_upper"].to_numpy(dtype=float),
    )
    return model.predict(score_matrix), float(model.sigma_), bool(model.converged_), explained


def _censoring_sensitivity(core: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    potency = core["potency"].copy()
    novel = core["novel"].copy()
    controls = core["controls"]
    exact_extension = novel[novel["herg_pic50_relation"].eq("=")].copy()
    exact_extension = exact_extension.rename(columns={"herg_pic50_value": "observed"})
    exact_training = np.isfinite(potency["pic50_lower"]) & np.isfinite(potency["pic50_upper"])
    exact_training &= np.isclose(potency["pic50_lower"], potency["pic50_upper"])
    scenarios: list[tuple[str, pd.DataFrame]] = []
    scenarios.append(("correct_interval_censoring", potency.copy()))
    scenarios.append(("drop_censored_rows", potency.loc[exact_training].copy()))
    boundary = potency.copy()
    upper_only = ~np.isfinite(boundary["pic50_lower"]) & np.isfinite(boundary["pic50_upper"])
    lower_only = np.isfinite(boundary["pic50_lower"]) & ~np.isfinite(boundary["pic50_upper"])
    boundary.loc[upper_only, "pic50_lower"] = boundary.loc[upper_only, "pic50_upper"]
    boundary.loc[lower_only, "pic50_upper"] = boundary.loc[lower_only, "pic50_lower"]
    scenarios.append(("incorrect_boundary_as_exact_sensitivity", boundary))
    rows: list[dict[str, Any]] = []
    predictions = exact_extension[["structure_id", "internal_id", "scaffold", "observed"]].copy()
    for name, training in scenarios:
        predicted, sigma, converged, explained = _fit_censored_extension(
            training,
            exact_extension,
            controls,
            seed=SEED,
        )
        predictions[name] = predicted
        metrics = _regression_metrics(exact_extension["observed"].to_numpy(), predicted)
        rows.append(
            {
                "censoring_scenario": name,
                "training_rows": int(len(training)),
                "training_exact_rows": int(
                    (
                        np.isfinite(training["pic50_lower"])
                        & np.isfinite(training["pic50_upper"])
                        & np.isclose(training["pic50_lower"], training["pic50_upper"])
                    ).sum()
                ),
                "training_censored_rows": int(
                    (
                        ~(
                            np.isfinite(training["pic50_lower"])
                            & np.isfinite(training["pic50_upper"])
                            & np.isclose(training["pic50_lower"], training["pic50_upper"])
                        )
                    ).sum()
                ),
                "fit_converged": converged,
                "predictive_sigma": sigma,
                "explained_fingerprint_variance": explained,
                **metrics,
                "primary": name == "correct_interval_censoring",
            }
        )
    reference = "correct_interval_censoring"
    for scenario in [name for name, _ in scenarios if name != reference]:
        delta = _paired_group_bootstrap(
            predictions,
            left=scenario,
            right=reference,
            observed="observed",
            seed=SEED + 400 + len(rows),
        )
        rows.append(
            {
                "censoring_scenario": f"{scenario}_minus_{reference}",
                **delta,
                "primary": False,
                "interpretation": "Positive delta means the sensitivity treatment is worse.",
            }
        )
    return pd.DataFrame(rows), predictions


def _representation_seed_stability(core: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    potency = core["potency"]
    novel = core["novel"].copy()
    controls = core["controls"]
    exact = novel[novel["herg_pic50_relation"].eq("=")].copy()
    exact = exact.rename(columns={"herg_pic50_value": "observed"})
    rows: list[dict[str, Any]] = []
    prediction_columns: dict[int, np.ndarray] = {}
    for offset in range(24):
        seed = SEED + 1000 + offset
        predicted, sigma, converged, explained = _fit_censored_extension(
            potency,
            exact,
            controls,
            seed=seed,
        )
        prediction_columns[seed] = predicted
        rows.append(
            {
                "seed": seed,
                "fit_converged": converged,
                "predictive_sigma": sigma,
                "explained_fingerprint_variance": explained,
                **_regression_metrics(exact["observed"].to_numpy(), predicted),
            }
        )
    matrix = np.column_stack(list(prediction_columns.values()))
    compound_rows = exact[["structure_id", "internal_id", "scaffold", "observed"]].copy()
    compound_rows["seed_mean_prediction"] = matrix.mean(axis=1)
    compound_rows["seed_prediction_sd"] = matrix.std(axis=1, ddof=1)
    compound_rows["seed_prediction_range"] = matrix.max(axis=1) - matrix.min(axis=1)
    compound_rows["seed_mean_absolute_error"] = np.abs(
        compound_rows["seed_mean_prediction"] - compound_rows["observed"]
    )
    return pd.DataFrame(rows), compound_rows


def _within_group_permutation_correlation(
    frame: pd.DataFrame,
    *,
    x: str,
    y: str,
    group: str = "scaffold",
    permutations: int = PERMUTATIONS,
    seed: int = SEED,
) -> dict[str, float]:
    work = frame[[x, y, group]].dropna().copy()
    informative = work.groupby(group).filter(lambda value: len(value) >= 2).copy()
    if len(informative) < 6 or informative[group].nunique() < 2:
        return {
            "n": float(len(work)),
            "n_informative": float(len(informative)),
            "overall_spearman": _safe_spearman(work[x], work[y]),
            "within_group_spearman": float("nan"),
            "permutation_p": float("nan"),
        }
    informative["_x"] = informative[x] - informative.groupby(group)[x].transform("mean")
    informative["_y"] = informative[y] - informative.groupby(group)[y].transform("mean")
    observed = _safe_spearman(informative["_x"].to_numpy(), informative["_y"].to_numpy())
    groups = [value.index.to_numpy() for _, value in informative.groupby(group, sort=False)]
    positions = {index: position for position, index in enumerate(informative.index)}
    group_positions = [np.asarray([positions[index] for index in indices], dtype=int) for indices in groups]
    centered_x = informative["_x"].to_numpy(dtype=float)
    centered_y = informative["_y"].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    null = np.empty(permutations, dtype=float)
    for replicate in range(permutations):
        permuted = centered_y.copy()
        for group_indices in group_positions:
            permuted[group_indices] = rng.permutation(permuted[group_indices])
        null[replicate] = _safe_spearman(centered_x, permuted)
    finite = null[np.isfinite(null)]
    return {
        "n": float(len(work)),
        "n_informative": float(len(informative)),
        "n_informative_groups": float(informative[group].nunique()),
        "overall_spearman": _safe_spearman(work[x], work[y]),
        "within_group_spearman": observed,
        "permutation_p": float((1 + np.sum(np.abs(finite) >= abs(observed))) / (1 + len(finite))),
    }


def _benjamini_hochberg(values: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")
    finite = values.notna()
    adjusted = pd.Series(np.nan, index=values.index, dtype=float)
    if not finite.any():
        return adjusted
    ordered = values[finite].sort_values()
    count = len(ordered)
    raw = ordered.to_numpy(dtype=float) * count / np.arange(1, count + 1)
    corrected = np.minimum.accumulate(raw[::-1])[::-1]
    adjusted.loc[ordered.index] = np.minimum(corrected, 1.0)
    return adjusted


def _herg_residual_bias(core: dict[str, Any]) -> pd.DataFrame:
    novel = core["novel"].copy()
    exact = novel[novel["herg_pic50_relation"].eq("=")].copy()
    exact["absolute_error"] = np.abs(exact["complete_feature_predicted_pic50"] - exact["herg_pic50_value"])
    exact["signed_error"] = exact["complete_feature_predicted_pic50"] - exact["herg_pic50_value"]
    candidate_features = [
        *core["controls"],
        "nearest_exact_train_tanimoto",
        "max_train_tanimoto",
        "herg_pic50_value",
    ]
    if "nearest_exact_train_tanimoto" not in exact:
        train = core["potency"]
        exact_train = train[
            np.isfinite(train["pic50_lower"])
            & np.isfinite(train["pic50_upper"])
            & np.isclose(train["pic50_lower"], train["pic50_upper"])
        ].drop_duplicates("standardized_smiles")
        exact["nearest_exact_train_tanimoto"] = _tanimoto(
            _fingerprints(exact["standardized_smiles"]),
            _fingerprints(exact_train["standardized_smiles"]),
        ).max(axis=1)
    rows: list[dict[str, Any]] = []
    for feature_index, feature in enumerate(candidate_features):
        if feature not in exact:
            continue
        for error in ("absolute_error", "signed_error"):
            result = _within_group_permutation_correlation(
                exact,
                x=feature,
                y=error,
                seed=SEED + 2000 + feature_index,
            )
            rows.append(
                {
                    "feature": feature,
                    "error_definition": error,
                    **result,
                    "role": "diagnostic_only_not_feature_selection",
                }
            )
    result = pd.DataFrame(rows)
    result["within_group_fdr_q"] = _benjamini_hochberg(result["permutation_p"])
    return result


def _herg_extension_learning_curve(core: dict[str, Any]) -> pd.DataFrame:
    potency = core["potency"]
    novel = core["novel"].copy()
    exact = novel[novel["herg_pic50_relation"].eq("=")].copy()
    exact = exact.rename(columns={"herg_pic50_value": "observed"})
    controls = core["controls"]
    groups = np.asarray(sorted(potency["scaffold"].astype(str).unique()))
    fractions = (0.40, 0.60, 0.80, 1.00)
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(SEED + 3000)
    for fraction in fractions:
        repetitions = 20 if fraction < 1 else 12
        for replicate in range(repetitions):
            count = max(5, int(math.ceil(fraction * len(groups))))
            selected = rng.choice(groups, size=count, replace=False) if count < len(groups) else groups.copy()
            training = potency[potency["scaffold"].astype(str).isin(selected)].copy()
            if len(training) < 20:
                continue
            predicted, sigma, converged, explained = _fit_censored_extension(
                training,
                exact,
                controls,
                seed=SEED + 4000 + replicate,
            )
            metrics = _regression_metrics(exact["observed"].to_numpy(), predicted)
            rows.append(
                {
                    "training_scaffold_fraction": fraction,
                    "replicate": replicate,
                    "training_rows": int(len(training)),
                    "training_unique_structures": int(training["compound_id"].nunique()),
                    "training_scaffolds": int(training["scaffold"].nunique()),
                    "fit_converged": converged,
                    "predictive_sigma": sigma,
                    "explained_fingerprint_variance": explained,
                    **metrics,
                    "selection_boundary": (
                        "Outcome-blind random training-scaffold subsets; fixed 8-component "
                        "representation and fixed extension evaluation set."
                    ),
                }
            )
    return pd.DataFrame(rows)


def _pk_compound_metrics() -> tuple[pd.DataFrame, pd.DataFrame]:
    pk_frames, herg_frame = _load_balanced_predictions(ROOT)
    frames = {**pk_frames, "herg_pic50": herg_frame}
    summary_rows: list[dict[str, Any]] = []
    bootstraps: list[pd.DataFrame] = []
    for index, (endpoint, frame) in enumerate(frames.items()):
        work = frame.copy()

        def metric(sample: pd.DataFrame) -> dict[str, float]:
            result = _regression_metrics(
                sample["observed"].to_numpy(),
                sample["predicted"].to_numpy(),
            )
            if {"interval_lower", "interval_upper"}.issubset(sample):
                covered = (sample["observed"] >= sample["interval_lower"]) & (
                    sample["observed"] <= sample["interval_upper"]
                )
                lower, upper = _wilson(int(covered.sum()), int(len(covered)))
                result.update(
                    {
                        "interval_coverage": float(covered.mean()),
                        "interval_coverage_wilson_lower_95": lower,
                        "interval_coverage_wilson_upper_95": upper,
                        "interval_mean_width_log10": float(
                            (sample["interval_upper"] - sample["interval_lower"]).mean()
                        ),
                        "interval_width_to_observed_range": float(
                            (sample["interval_upper"] - sample["interval_lower"]).mean()
                            / max(
                                float(sample["observed"].max() - sample["observed"].min()),
                                1e-12,
                            )
                        ),
                        "interval_score_90": float(
                            _interval_score(
                                sample["observed"].to_numpy(),
                                sample["interval_lower"].to_numpy(),
                                sample["interval_upper"].to_numpy(),
                            ).mean()
                        ),
                    }
                )
            return result

        point, bootstrap = _group_bootstrap(
            work,
            metric,
            seed=SEED + 5000 + index,
        )
        domain_column = (
            "inside_applicability_domain"
            if "inside_applicability_domain" in work
            else "inside_domain"
            if "inside_domain" in work
            else None
        )
        inside = work[work[domain_column].fillna(False)] if domain_column else work.iloc[0:0]
        outside = work[~work[domain_column].fillna(False)] if domain_column else work.iloc[0:0]
        summary_rows.append(
            {
                "endpoint": endpoint,
                "model": str(work["model"].iloc[0]) if "model" in work else "unknown",
                "n_compounds": int(len(work)),
                "n_scaffolds": int(work["scaffold"].nunique()),
                "inside_domain_n": int(len(inside)),
                "outside_domain_n": int(len(outside)),
                "inside_domain_mae": (
                    float(np.mean(np.abs(inside["predicted"] - inside["observed"])))
                    if len(inside)
                    else float("nan")
                ),
                "outside_domain_mae": (
                    float(np.mean(np.abs(outside["predicted"] - outside["observed"])))
                    if len(outside)
                    else float("nan")
                ),
                **point,
                "evidence_role": "fixed_compound_balanced_group_held_out_retrospective",
            }
        )
        bootstrap.insert(1, "endpoint", endpoint)
        bootstraps.append(bootstrap)
    return pd.DataFrame(summary_rows), pd.concat(bootstraps, ignore_index=True)


def _fixed_loco_summary() -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = pd.read_csv(REVIEW_ROOT / "fixed_baseline_loco_metrics.csv")
    rows: list[dict[str, Any]] = []
    for (endpoint, definition), group in metrics.groupby(
        ["endpoint", "group_definition"],
        sort=True,
    ):
        mean_row = group[group["model"].eq("train_mean")].iloc[0]
        candidates = group[~group["model"].eq("train_mean")].sort_values(
            ["mae", "model"],
            kind="stable",
        )
        best = candidates.iloc[0]
        conclusive = bool(best["delta_mae_upper_95"] < 0)
        rows.append(
            {
                "endpoint": endpoint,
                "group_definition": definition,
                "n_compounds": int(best["n_compounds"]),
                "n_groups": int(best["n_held_out_groups"]),
                "train_mean_mae": float(mean_row["mae"]),
                "best_fixed_model": best["model"],
                "best_fixed_model_mae": float(best["mae"]),
                "best_fixed_model_spearman": float(best["spearman"]),
                "delta_mae_vs_train_mean": float(best["mean_delta_mae_vs_train_mean"]),
                "delta_mae_lower_95": float(best["delta_mae_lower_95"]),
                "delta_mae_upper_95": float(best["delta_mae_upper_95"]),
                "conclusive_advantage_over_train_mean": conclusive,
                "interpretation": (
                    "Prespecified comparator only; choosing the lowest row here is a "
                    "descriptive best-of-fixed-baselines summary, not a final model claim."
                ),
            }
        )
    return metrics, pd.DataFrame(rows)


def _selected_vs_fixed_baselines() -> pd.DataFrame:
    pk_frames, herg_frame = _load_balanced_predictions(ROOT)
    selected_frames = {**pk_frames, "herg_pic50": herg_frame}
    fixed = pd.read_parquet(REVIEW_ROOT / "fixed_baseline_loco_predictions.parquet")
    fixed = fixed[fixed["group_definition"].eq("scaffold")].copy()
    rows: list[dict[str, Any]] = []
    for endpoint_index, (endpoint, selected) in enumerate(selected_frames.items()):
        endpoint_fixed = fixed[fixed["endpoint"].eq(endpoint)]
        for model_index, (model, comparator) in enumerate(endpoint_fixed.groupby("model")):
            merged = (
                selected[["compound_id", "scaffold", "observed", "predicted", "model"]]
                .rename(columns={"predicted": "selected_prediction", "model": "selected_model"})
                .merge(
                    comparator[["compound_id", "observed", "predicted"]].rename(
                        columns={
                            "observed": "fixed_observed",
                            "predicted": "fixed_prediction",
                        }
                    ),
                    on="compound_id",
                    how="inner",
                    validate="one_to_one",
                )
            )
            target_delta = np.abs(merged["observed"] - merged["fixed_observed"])
            target_mismatch_count = int((target_delta > 1e-10).sum())
            maximum_target_delta = float(target_delta.max())
            # One hERG compound differs by 0.0359 log because the production
            # and reviewer audits collapse repeated exact measurements slightly
            # differently. Evaluate both predictions against the production OOF
            # target, expose the discrepancy, and fail if it is ever material.
            if maximum_target_delta > 0.05:
                raise ValueError(
                    f"Material observed target mismatch for {endpoint}/{model}: "
                    f"{maximum_target_delta:.6f} log"
                )
            selected_metrics = _regression_metrics(
                merged["observed"].to_numpy(), merged["selected_prediction"].to_numpy()
            )
            fixed_metrics = _regression_metrics(
                merged["observed"].to_numpy(), merged["fixed_prediction"].to_numpy()
            )
            delta = _paired_group_bootstrap(
                merged,
                left="selected_prediction",
                right="fixed_prediction",
                observed="observed",
                seed=SEED + 5500 + endpoint_index * 20 + model_index,
            )
            rows.append(
                {
                    "endpoint": endpoint,
                    "selected_model": str(merged["selected_model"].iloc[0]),
                    "fixed_comparator": model,
                    "n_compounds": int(len(merged)),
                    "n_scaffolds": int(merged["scaffold"].nunique()),
                    "target_definition_mismatch_compounds": target_mismatch_count,
                    "maximum_target_definition_delta_log": maximum_target_delta,
                    "selected_mae": selected_metrics["mae"],
                    "selected_spearman": selected_metrics["spearman"],
                    "fixed_mae": fixed_metrics["mae"],
                    "fixed_spearman": fixed_metrics["spearman"],
                    "selected_minus_fixed_mae": delta["mean_delta_mae"],
                    "delta_mae_lower_95": delta["delta_mae_lower_95"],
                    "delta_mae_upper_95": delta["delta_mae_upper_95"],
                    "selected_conclusively_better": delta["delta_mae_upper_95"] < 0,
                    "fixed_conclusively_better": delta["delta_mae_lower_95"] > 0,
                    "interpretation": (
                        "Retrospective paired comparison on identical compounds and scaffold "
                        "grouping; selected production model has selection optimism, while fixed "
                        "comparator was prespecified in the reviewer audit. Both predictions are "
                        "evaluated against the production OOF target; any small reviewer-target "
                        "aggregation discrepancy is reported explicitly."
                    ),
                }
            )
    return pd.DataFrame(rows)


def _pk_learning_curve(core: dict[str, Any]) -> pd.DataFrame:
    frames, _ = _source_collapsed_pk_frames(
        core["compounds"],
        core["tables"]["measurements"],
        core["tables"]["pk_studies"],
    )
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(SEED + 6000)
    for endpoint in [
        "iv_auc_dose_normalized",
        "po_auc_dose_normalized",
        "vdss",
        "po_cmax_dose_normalized",
        "po_tmax",
    ]:
        frame = frames[endpoint].reset_index(drop=True)
        fingerprints, _ = fingerprint_matrix(
            frame["standardized_smiles"],
            backend="rdkit",
            n_bits=2048,
            radius=2,
        )
        bits = fingerprints.toarray().astype(float)
        y = frame["target_log10"].to_numpy(dtype=float)
        groups = np.asarray(sorted(frame["scaffold"].astype(str).unique()))
        for fraction in (0.35, 0.55, 0.75, 0.85):
            for replicate in range(50):
                train_group_count = min(
                    len(groups) - 2,
                    max(3, int(math.ceil(fraction * len(groups)))),
                )
                train_groups = rng.choice(groups, size=train_group_count, replace=False)
                train_mask = frame["scaffold"].astype(str).isin(train_groups).to_numpy()
                test_mask = ~train_mask
                train = np.flatnonzero(train_mask)
                test = np.flatnonzero(test_mask)
                if len(train) < 8 or len(test) < 3:
                    continue
                model = Ridge(alpha=10.0)
                model.fit(bits[train], y[train])
                predicted = model.predict(bits[test])
                mean_predicted = np.full(len(test), float(y[train].mean()))
                model_metrics = _regression_metrics(y[test], predicted)
                mean_metrics = _regression_metrics(y[test], mean_predicted)
                rows.append(
                    {
                        "endpoint": endpoint,
                        "training_scaffold_fraction": fraction,
                        "replicate": replicate,
                        "training_compounds": int(len(train)),
                        "test_compounds": int(len(test)),
                        "training_scaffolds": int(train_group_count),
                        "test_scaffolds": int(len(groups) - train_group_count),
                        "morgan_ridge_mae": model_metrics["mae"],
                        "morgan_ridge_spearman": model_metrics["spearman"],
                        "train_mean_mae": mean_metrics["mae"],
                        "delta_mae_vs_train_mean": (model_metrics["mae"] - mean_metrics["mae"]),
                        "split_seed_family": SEED + 6000,
                        "selection_boundary": (
                            "Outcome-blind random scaffold subsets; fixed Morgan ridge; "
                            "no model or hyperparameter selection."
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _learning_curve_summary(
    hERG_curve: pd.DataFrame,
    pk_curve: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for fraction, group in hERG_curve.groupby("training_scaffold_fraction"):
        rows.append(
            {
                "domain": "herg_same_series_extension",
                "endpoint": "herg_pic50",
                "training_scaffold_fraction": fraction,
                "replicates": int(len(group)),
                "median_training_compounds": float(group["training_unique_structures"].median()),
                "median_training_scaffolds": float(group["training_scaffolds"].median()),
                "median_mae": float(group["mae"].median()),
                "mae_lower_10": float(group["mae"].quantile(0.10)),
                "mae_upper_90": float(group["mae"].quantile(0.90)),
                "median_spearman": float(group["spearman"].median()),
                "fraction_converged": float(group["fit_converged"].mean()),
                "interpretation": (
                    "Fixed external retrospective extension; estimates chemical coverage "
                    "value, not prospective performance."
                ),
            }
        )
    for (endpoint, fraction), group in pk_curve.groupby(["endpoint", "training_scaffold_fraction"]):
        rows.append(
            {
                "domain": "pk_internal_random_scaffold_holdout",
                "endpoint": endpoint,
                "training_scaffold_fraction": fraction,
                "replicates": int(len(group)),
                "median_training_compounds": float(group["training_compounds"].median()),
                "median_training_scaffolds": float(group["training_scaffolds"].median()),
                "median_mae": float(group["morgan_ridge_mae"].median()),
                "mae_lower_10": float(group["morgan_ridge_mae"].quantile(0.10)),
                "mae_upper_90": float(group["morgan_ridge_mae"].quantile(0.90)),
                "median_spearman": float(group["morgan_ridge_spearman"].median()),
                "median_delta_mae_vs_train_mean": float(group["delta_mae_vs_train_mean"].median()),
                "probability_improves_on_train_mean": float(np.mean(group["delta_mae_vs_train_mean"] < 0)),
                "interpretation": (
                    "Repeated outcome-blind scaffold holdouts; descriptive learning "
                    "behavior under current narrow chemistry."
                ),
            }
        )
    return pd.DataFrame(rows)


def _learning_curve_trends(
    herg_curve: pd.DataFrame,
    pk_curve: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(SEED + 6500)
    definitions = [("herg_same_series_extension", "herg_pic50", herg_curve, "mae")]
    definitions.extend(
        (
            "pk_internal_random_scaffold_holdout",
            endpoint,
            group,
            "morgan_ridge_mae",
        )
        for endpoint, group in pk_curve.groupby("endpoint", sort=True)
    )
    for analysis, endpoint, frame, metric in definitions:
        work = frame[["training_scaffold_fraction", metric]].dropna().copy()
        fractions = sorted(work["training_scaffold_fraction"].unique())
        low = work[work["training_scaffold_fraction"].eq(fractions[0])][metric].to_numpy()
        high = work[work["training_scaffold_fraction"].eq(fractions[-1])][metric].to_numpy()
        deltas = np.empty(BOOTSTRAPS, dtype=float)
        for replicate in range(BOOTSTRAPS):
            deltas[replicate] = float(
                rng.choice(high, size=len(high), replace=True).mean()
                - rng.choice(low, size=len(low), replace=True).mean()
            )
        observed_rho = _safe_spearman(work["training_scaffold_fraction"].to_numpy(), work[metric].to_numpy())
        null = np.empty(PERMUTATIONS, dtype=float)
        values = work[metric].to_numpy()
        fraction_values = work["training_scaffold_fraction"].to_numpy()
        for replicate in range(PERMUTATIONS):
            null[replicate] = _safe_spearman(
                fraction_values,
                rng.permutation(values),
            )
        rows.append(
            {
                "analysis": analysis,
                "endpoint": endpoint,
                "n_runs": int(len(work)),
                "lowest_training_fraction": float(fractions[0]),
                "highest_training_fraction": float(fractions[-1]),
                "median_mae_at_lowest_fraction": float(np.median(low)),
                "median_mae_at_highest_fraction": float(np.median(high)),
                "mean_high_minus_low_mae": float(high.mean() - low.mean()),
                "high_minus_low_mae_lower_95": float(np.quantile(deltas, 0.025)),
                "high_minus_low_mae_upper_95": float(np.quantile(deltas, 0.975)),
                "spearman_training_fraction_vs_mae": observed_rho,
                "permutation_p_two_sided": float(
                    (1 + np.sum(np.abs(null) >= abs(observed_rho))) / (1 + len(null))
                ),
                "interpretation": (
                    "Descriptive repeated-split trend; repeats share compounds and are not "
                    "independent prospective experiments. Negative high-minus-low favors more "
                    "chemical-group coverage."
                ),
            }
        )
    return pd.DataFrame(rows)


def _sample_size_sensitivity(
    pk_metrics: pd.DataFrame,
    extension_predictions: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    z95 = 1.959963984540054
    z80 = 0.8416212335729143
    pk_frames, _ = _load_balanced_predictions(ROOT)
    for endpoint, frame in pk_frames.items():
        scaffold_errors = (
            frame.assign(absolute_error=np.abs(frame["predicted"] - frame["observed"]))
            .groupby("scaffold")["absolute_error"]
            .mean()
        )
        sd = float(scaffold_errors.std(ddof=1))
        for half_width in (0.10, 0.05):
            needed = int(math.ceil((z95 * sd / half_width) ** 2)) if sd > 0 else 1
            rows.append(
                {
                    "domain": "pk",
                    "endpoint": endpoint,
                    "planning_quantity": "independent_scaffold_mean_absolute_error",
                    "target_half_width": half_width,
                    "current_independent_groups": int(len(scaffold_errors)),
                    "estimated_independent_groups_needed": needed,
                    "additional_groups_needed": max(0, needed - len(scaffold_errors)),
                    "basis": (
                        "Normal-approximation sensitivity using the observed SD of "
                        "scaffold-level OOF absolute error; not a formal trial power analysis."
                    ),
                }
            )
    exact = extension_predictions[extension_predictions["herg_pic50_relation"].eq("=")].copy()
    exact["complete_error"] = np.abs(exact["complete_feature_predicted_pic50"] - exact["herg_pic50_value"])
    exact["global_error"] = np.abs(exact["predicted_pic50"] - exact["herg_pic50_value"])
    differences = (
        exact.assign(delta=exact["complete_error"] - exact["global_error"])
        .groupby("scaffold")["delta"]
        .mean()
    )
    effect = abs(float(differences.mean()))
    sd = float(differences.std(ddof=1))
    needed = int(math.ceil(((z95 + z80) * sd / effect) ** 2)) if effect > 1e-12 and sd > 0 else 999
    rows.append(
        {
            "domain": "herg",
            "endpoint": "complete_feature_vs_global_controls",
            "planning_quantity": "paired_scaffold_mae_difference_80pct_power",
            "target_half_width": float("nan"),
            "current_independent_groups": int(len(differences)),
            "estimated_independent_groups_needed": needed,
            "additional_groups_needed": max(0, needed - len(differences)),
            "observed_scaffold_mean_effect": float(differences.mean()),
            "observed_scaffold_effect_sd": sd,
            "basis": (
                "Normal-approximation sensitivity at two-sided alpha 0.05 and 80% "
                "power; exploratory because the observed effect was estimated after "
                "retrospective outcome review."
            ),
        }
    )
    threshold = _threshold_metrics_from_predictions(extension_predictions, 5.0)
    for class_name, probability, current in (
        ("blocker_sensitivity", threshold["sensitivity"], int(threshold["n_blockers"])),
        ("nonblocker_specificity", threshold["specificity"], int(threshold["n_nonblockers"])),
    ):
        half_width = 0.10
        needed = int(math.ceil(z95 * z95 * probability * (1 - probability) / half_width**2))
        rows.append(
            {
                "domain": "herg",
                "endpoint": class_name,
                "planning_quantity": "binomial_precision",
                "target_half_width": half_width,
                "current_independent_groups": current,
                "estimated_independent_groups_needed": needed,
                "additional_groups_needed": max(0, needed - current),
                "basis": (
                    "Simple binomial precision sensitivity; compounds are not truly "
                    "independent within series, so new scaffolds/series are more valuable "
                    "than the same number of additional close analogs."
                ),
            }
        )
    return pd.DataFrame(rows)


def _threshold_metrics_from_predictions(
    predictions: pd.DataFrame,
    threshold: float,
) -> dict[str, float]:
    frame = predictions.copy()
    exact = frame["herg_pic50_relation"].eq("=")
    censored = frame["herg_pic50_relation"].eq("<") & frame["herg_pic50_upper_bound"].le(threshold + 1e-12)
    frame = frame[exact | censored].copy()
    labels = np.where(
        frame["herg_pic50_relation"].eq("="),
        (frame["herg_pic50_value"] >= threshold).astype(int),
        0,
    )
    sigma = float(frame["complete_feature_oof_sigma"].iloc[0])
    probability = norm.sf((threshold - frame["complete_feature_predicted_pic50"].to_numpy()) / sigma)
    return _classification_metrics(labels, probability)


def _data_inventory(core: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    table_rows = []
    for name, frame in sorted(core["tables"].items()):
        table_rows.append(
            {
                "canonical_table": name,
                "rows": int(len(frame)),
                "columns": int(len(frame.columns)),
                "nonempty": bool(len(frame)),
            }
        )
    measurements = core["tables"]["measurements"].copy()
    endpoint_rows: list[dict[str, Any]] = []
    for endpoint, frame in measurements.groupby("endpoint", dropna=False):
        relations = frame["relation"].fillna("").astype(str)
        endpoint_rows.append(
            {
                "endpoint": endpoint,
                "rows": int(len(frame)),
                "unique_compounds": int(frame["compound_id"].nunique()),
                "model_eligible_rows": int(frame["model_eligible"].fillna(False).sum()),
                "exact_or_approximate_rows": int(relations.isin(["=", "~"]).sum()),
                "left_or_right_censored_rows": int(relations.isin(["<", "<=", ">", ">="]).sum()),
                "species": "|".join(sorted(frame["species"].dropna().astype(str).unique())),
                "routes": "|".join(sorted(frame["route"].dropna().astype(str).unique())),
            }
        )
    return pd.DataFrame(table_rows), pd.DataFrame(endpoint_rows)


def _physics_gate_summary(core: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    admission = pd.read_csv(
        ROOT / "research/reports/pk_herg/manuscript/local_physics_observable_admission.csv"
    )
    hpc = pd.read_csv(ROOT / "research/reports/pk_herg/manuscript/local_physics_hpc_handoff.csv")
    admitted = admission["current_decision"].astype(str).str.match(r"^(admitted|decision_track)")
    physics_observables = core["tables"]["physics_observables"]
    physics_runs = core["tables"]["physics_runs"]
    summary = {
        "candidate_observables_reviewed": int(len(admission)),
        "observables_admitted_to_models": int(admitted.sum()),
        "canonical_physics_observable_rows": int(len(physics_observables)),
        "canonical_physics_run_rows": int(len(physics_runs)),
        "hpc_workflows_specified": int(len(hpc)),
        "highest_priority_hpc_workflows": hpc.sort_values("priority").head(3)["workflow"].tolist(),
        "interpretation": (
            "Fail-closed physics admission is working: local proxies generated "
            "falsifiable hypotheses and preparation gates, but none is mislabeled "
            "as an equilibrium, kinetic, or predictive observable."
        ),
    }
    return admission, summary


def _claim_ledger(
    *,
    core: dict[str, Any],
    pk_metrics: pd.DataFrame,
    herg_comparison: pd.DataFrame,
    threshold_metrics: pd.DataFrame,
    seed_metrics: pd.DataFrame,
    censor_metrics: pd.DataFrame,
    learning_summary: pd.DataFrame,
    physics_summary: dict[str, Any],
) -> pd.DataFrame:
    complete = herg_comparison[herg_comparison["model"].eq("complete_feature")].iloc[0]
    delta = herg_comparison[herg_comparison["model"].eq("complete_feature_minus_global_controls")].iloc[0]
    threshold = threshold_metrics.loc[np.isclose(threshold_metrics["pic50_threshold"], 5.0)].iloc[0]
    pk_lookup = pk_metrics.set_index("endpoint")
    external = json.loads(
        (ROOT / "research/reports/pk_herg/herg_external_challenge/challenge_summary.json").read_text(
            encoding="utf-8"
        )
    )
    rows = [
        {
            "claim": "Canonical data and leakage-aware target construction are operational.",
            "status": "strong",
            "evidence": (
                f"{len(core['compounds'])} unique internal structures; typed canonical "
                "tables; CL/F retained as closure diagnostics rather than independent labels."
            ),
            "limitation": (
                "130 unique normalization/pairing errors remain and PK availability is "
                "chemically selected; the Ascentage DOCX/CDX raw binaries require reacquisition."
            ),
            "allowed_use": "reproducible retrospective research",
        },
        {
            "claim": "Same-series hERG chemistry contains a strong rank-order signal.",
            "status": "promising_not_confirmatory",
            "evidence": (
                f"42 non-overlap exact values across 8 computed scaffolds: complete-feature "
                f"MAE {complete['mae']:.3f} log, Spearman {complete['spearman']:.3f}, "
                f"{complete['fraction_within_0p5_log']:.1%} within 0.5 log."
            ),
            "limitation": (
                "Representation was conceived after outcome review; all compounds are from "
                "the same medicinal-chemistry series."
            ),
            "allowed_use": "analogue-series hypothesis generation",
        },
        {
            "claim": "The complete hERG representation conclusively beats global controls.",
            "status": ("supported" if float(delta["delta_mae_upper_95"]) < 0 else "not_yet_supported"),
            "evidence": (
                f"Paired scaffold-bootstrap complete-minus-global MAE "
                f"{delta['mean_delta_mae']:.3f}, 95% CI "
                f"[{delta['delta_mae_lower_95']:.3f}, {delta['delta_mae_upper_95']:.3f}]."
            ),
            "limitation": "Only eight computed extension scaffolds and post-outcome conception.",
            "allowed_use": "retain both models and expose disagreement",
        },
        {
            "claim": "The 10 µM hERG threshold has useful retrospective discrimination.",
            "status": "promising_not_calibrated",
            "evidence": (
                f"n={int(threshold['n'])}, ROC-AUC {threshold['roc_auc']:.3f}, "
                f"balanced accuracy {threshold['balanced_accuracy']:.3f}, "
                f"sensitivity {threshold['sensitivity']:.1%}, "
                f"specificity {threshold['specificity']:.1%}."
            ),
            "limitation": (
                f"Only {int(threshold['n_nonblockers'])} definite nonblockers; specificity "
                f"Wilson 95% CI [{threshold['specificity_wilson_lower_95']:.2f}, "
                f"{threshold['specificity_wilson_upper_95']:.2f}]; calibration is retrospective."
            ),
            "allowed_use": "probability plus uncertainty, never an unqualified binary call",
        },
        {
            "claim": "hERG latent-representation randomness is negligible.",
            "status": (
                "supported"
                if float(seed_metrics["mae"].max() - seed_metrics["mae"].min()) < 0.03
                else "sensitivity_detected"
            ),
            "evidence": (
                f"Across {len(seed_metrics)} SVD seeds, MAE range "
                f"{seed_metrics['mae'].min():.3f}-{seed_metrics['mae'].max():.3f}; "
                f"all-fit convergence {seed_metrics['fit_converged'].mean():.1%}."
            ),
            "limitation": "Numerical stability does not establish biological validity.",
            "allowed_use": "implementation stability evidence",
        },
        {
            "claim": "Correct censoring treatment materially changes hERG conclusions.",
            "status": "sensitivity_quantified",
            "evidence": (
                "Correct interval-censoring, exact-only, and boundary-as-exact scenarios "
                f"were compared; primary MAE "
                f"{censor_metrics.loc[censor_metrics['censoring_scenario'].eq('correct_interval_censoring'), 'mae'].iloc[0]:.3f}."
            ),
            "limitation": "Only four non-overlap and nineteen baseline censored rows.",
            "allowed_use": "correct censoring remains mandatory",
        },
        {
            "claim": "Rat PK contains useful retrospective analogue signal.",
            "status": "endpoint_dependent",
            "evidence": (
                f"Vdss MAE {pk_lookup.loc['vdss', 'mae']:.3f} log "
                f"({pk_lookup.loc['vdss', 'median_fold_error']:.2f}-fold median error); "
                f"PO AUC/dose MAE {pk_lookup.loc['po_auc_dose_normalized', 'mae']:.3f} log; "
                f"Cmax/dose MAE {pk_lookup.loc['po_cmax_dose_normalized', 'mae']:.3f} log."
            ),
            "limitation": (
                "46 compounds, about 15 computed scaffolds, no concentration-time profiles, "
                "formulation gaps, and biased measurement availability."
            ),
            "allowed_use": "discovery-track analogue prioritization with wide uncertainty",
        },
        {
            "claim": "Current PK intervals are calibrated tightly enough for decisions.",
            "status": "not_supported",
            "evidence": (
                f"Compound-level coverages span {pk_metrics['interval_coverage'].min():.1%}-"
                f"{pk_metrics['interval_coverage'].max():.1%}; width/range ratios span "
                f"{pk_metrics['interval_width_to_observed_range'].min():.2f}-"
                f"{pk_metrics['interval_width_to_observed_range'].max():.2f}."
            ),
            "limitation": "Small scaffold calibration sets and retrospective cross-conformal intervals.",
            "allowed_use": "uncertainty warning, not decision certification",
        },
        {
            "claim": "The current internal hERG model transfers to unrelated large molecules.",
            "status": "falsified",
            "evidence": (
                f"Public large-molecule challenge maximum internal similarity "
                f"{external['maximum_public_internal_tanimoto']:.3f}; "
                f"all outside internal domain={external['all_public_outside_internal_domain']}."
            ),
            "limitation": "Public assays and selection are heterogeneous.",
            "allowed_use": "none outside the analogue domain without new calibration",
        },
        {
            "claim": "A single MW cutoff or unavoidable PK-hERG tradeoff is established.",
            "status": "rejected",
            "evidence": (
                "No stable cross-outcome MW boundary; zero joint PK-hERG associations "
                "survived within-scaffold FDR."
            ),
            "limitation": "Only 35 compounds overlap exact PK and hERG.",
            "allowed_use": "model continuous physical-state interactions instead",
        },
        {
            "claim": "Physics work is appropriately gated and ready for HPC escalation.",
            "status": "strong_workflow_not_yet_physical_result",
            "evidence": (
                f"{physics_summary['candidate_observables_reviewed']} local observables/proxies "
                f"audited, {physics_summary['observables_admitted_to_models']} admitted; "
                f"{physics_summary['hpc_workflows_specified']} fail-closed HPC workflows specified."
            ),
            "limitation": "No converged equilibrium, transition, PMF, or receptor-binding observable.",
            "allowed_use": "hypothesis selection and simulation design only",
        },
    ]
    return pd.DataFrame(rows)


def _format_metric(value: Any, digits: int = 3) -> str:
    try:
        if pd.isna(value):
            return "NA"
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _write_figures(
    pk_metrics: pd.DataFrame,
    herg_comparison: pd.DataFrame,
    learning_summary: pd.DataFrame,
) -> None:
    comparison = herg_comparison[
        herg_comparison["model"].isin(
            [
                "train_mean",
                "morgan_1nn",
                "morgan_3nn",
                "morgan_5nn",
                "global_controls",
                "equal_weight_consensus",
                "complete_feature",
            ]
        )
    ].copy()
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    axes[0].barh(
        comparison["model"],
        comparison["mae"],
        color=[
            "#9ca3af",
            "#94a3b8",
            "#94a3b8",
            "#94a3b8",
            "#f59e0b",
            "#7c3aed",
            "#2563eb",
        ],
    )
    axes[0].errorbar(
        comparison["mae"],
        np.arange(len(comparison)),
        xerr=np.vstack(
            [
                comparison["mae"] - comparison["mae_lower_95"],
                comparison["mae_upper_95"] - comparison["mae"],
            ]
        ),
        fmt="none",
        ecolor="#111827",
        capsize=3,
    )
    axes[0].set(
        xlabel="pIC50 MAE (log10 µM transform)",
        title="Same-series extension: fixed comparators",
    )
    pk_plot = pk_metrics[~pk_metrics["endpoint"].eq("herg_pic50")].copy()
    axes[1].barh(pk_plot["endpoint"], pk_plot["median_fold_error"], color="#0f766e")
    axes[1].axvline(2.0, color="#dc2626", linestyle="--", linewidth=1, label="2-fold")
    axes[1].set(xlabel="Median absolute fold error", title="Compound-balanced PK OOF")
    axes[1].legend(frameon=False)
    _atomic_figure(figure, OUTPUT / "model_health_overview.png", dpi=220)
    _atomic_figure(figure, OUTPUT / "model_health_overview.pdf")
    plt.close(figure)

    curve = learning_summary.copy()
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    h = curve[curve["domain"].eq("herg_same_series_extension")]
    axes[0].plot(
        h["median_training_scaffolds"],
        h["median_mae"],
        marker="o",
        color="#2563eb",
    )
    axes[0].fill_between(
        h["median_training_scaffolds"].to_numpy(dtype=float),
        h["mae_lower_10"].to_numpy(dtype=float),
        h["mae_upper_90"].to_numpy(dtype=float),
        alpha=0.2,
        color="#2563eb",
    )
    axes[0].set(
        xlabel="Training computed scaffolds",
        ylabel="Fixed extension MAE",
        title="hERG chemical-coverage sensitivity",
    )
    pk = curve[curve["domain"].eq("pk_internal_random_scaffold_holdout")]
    for endpoint, group in pk.groupby("endpoint"):
        axes[1].plot(
            group["median_training_scaffolds"],
            group["median_mae"],
            marker="o",
            label=endpoint,
        )
    axes[1].set(
        xlabel="Training computed scaffolds",
        ylabel="Held-scaffold MAE",
        title="PK repeated scaffold learning curves",
    )
    axes[1].legend(fontsize=7, frameon=False)
    _atomic_figure(figure, OUTPUT / "learning_curve_overview.png", dpi=220)
    _atomic_figure(figure, OUTPUT / "learning_curve_overview.pdf")
    plt.close(figure)


def _write_calibration_figure(calibration_bins: pd.DataFrame) -> None:
    figure, axis = plt.subplots(figsize=(5.6, 5.2), constrained_layout=True)
    axis.plot([0, 1], [0, 1], linestyle="--", color="#6b7280", label="Perfect calibration")
    y = calibration_bins["observed_blocker_fraction"].to_numpy(dtype=float)
    lower = calibration_bins["observed_wilson_lower_95"].to_numpy(dtype=float)
    upper = calibration_bins["observed_wilson_upper_95"].to_numpy(dtype=float)
    axis.errorbar(
        calibration_bins["mean_predicted_probability"],
        y,
        yerr=np.vstack([y - lower, upper - y]),
        marker="o",
        capsize=4,
        color="#2563eb",
        label="Equal-count bins (Wilson 95% CI)",
    )
    ceiling_index = 0
    for _, row in calibration_bins.iterrows():
        at_ceiling = row["observed_blocker_fraction"] >= 0.98
        at_right_edge = row["mean_predicted_probability"] >= 0.95
        vertical_offset = -7 - 12 * ceiling_index if at_ceiling else 5
        axis.annotate(
            f"n={int(row['n'])}",
            (row["mean_predicted_probability"], row["observed_blocker_fraction"]),
            xytext=(-4 if at_right_edge else 4, vertical_offset),
            textcoords="offset points",
            fontsize=8,
            va="top" if at_ceiling else "bottom",
            ha="right" if at_right_edge else "left",
        )
        if at_ceiling:
            ceiling_index += 1
    axis.set(
        xlim=(-0.03, 1.03),
        ylim=(-0.03, 1.03),
        xlabel="Mean predicted blocker probability",
        ylabel="Observed blocker fraction",
        title="10 µM hERG retrospective calibration",
    )
    axis.legend(frameon=False, fontsize=8)
    _atomic_figure(figure, OUTPUT / "herg_calibration.png", dpi=220)
    _atomic_figure(figure, OUTPUT / "herg_calibration.pdf")
    plt.close(figure)


def _markdown_table(frame: pd.DataFrame, columns: list[str], labels: list[str]) -> str:
    header = "| " + " | ".join(labels) + " |"
    divider = "|" + "|".join(["---"] * len(columns)) + "|"
    rows = []
    for _, row in frame[columns].iterrows():
        values = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                values.append(_format_metric(value))
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, divider, *rows])


def _write_report(
    *,
    core: dict[str, Any],
    data_tables: pd.DataFrame,
    data_endpoints: pd.DataFrame,
    pk_metrics: pd.DataFrame,
    herg_comparison: pd.DataFrame,
    herg_interval: pd.DataFrame,
    herg_topology: pd.DataFrame,
    herg_group_sensitivity: pd.DataFrame,
    threshold_metrics: pd.DataFrame,
    calibration_summary: dict[str, Any],
    censor_metrics: pd.DataFrame,
    seed_metrics: pd.DataFrame,
    seed_compounds: pd.DataFrame,
    residual_bias: pd.DataFrame,
    loco_best: pd.DataFrame,
    selected_vs_fixed: pd.DataFrame,
    learning_summary: pd.DataFrame,
    learning_trends: pd.DataFrame,
    sample_size: pd.DataFrame,
    physics_summary: dict[str, Any],
    claim_ledger: pd.DataFrame,
) -> None:
    exact_models = herg_comparison[
        herg_comparison["model"].isin(
            [
                "train_mean",
                "morgan_1nn",
                "morgan_3nn",
                "morgan_5nn",
                "global_controls",
                "equal_weight_consensus",
                "complete_feature",
            ]
        )
    ].copy()
    complete = exact_models[exact_models["model"].eq("complete_feature")].iloc[0]
    consensus = exact_models[exact_models["model"].eq("equal_weight_consensus")].iloc[0]
    delta_global = herg_comparison[
        herg_comparison["model"].eq("complete_feature_minus_global_controls")
    ].iloc[0]
    threshold_10 = threshold_metrics.loc[np.isclose(threshold_metrics["ic50_threshold_um"], 10.0)].iloc[0]
    interval_complete = herg_interval[herg_interval["model"].eq("complete_feature")].iloc[0]
    topology_070 = herg_topology.loc[np.isclose(herg_topology["tanimoto_threshold"], 0.70)].iloc[0]
    censor_primary = censor_metrics[
        censor_metrics["censoring_scenario"].eq("correct_interval_censoring")
    ].iloc[0]
    pk_only = pk_metrics[~pk_metrics["endpoint"].eq("herg_pic50")].copy()
    nulls = pd.read_csv(REVIEW_ROOT / "y_scrambling_summary.csv")
    significant_null_controls = nulls[
        nulls["permutation_mode"].eq("within_scaffold") & nulls["spearman_monte_carlo_p"].lt(0.05)
    ][["endpoint", "observed_spearman", "spearman_monte_carlo_p"]]
    residuals = pd.read_csv(LOCAL_ROOT / "oof_residual_correlations.csv")
    residual_survivors = residuals[residuals["within_scaffold_fdr_005"]].copy()
    joint = pd.read_csv(LOCAL_ROOT / "joint_pk_herg_associations.csv")
    closure = pd.read_csv(REVIEW_ROOT / "pk_closure_summary.csv")
    selection = pd.read_csv(REVIEW_ROOT / "measurement_selection_bias.csv")
    selection_survivors = selection[selection["fdr_q"].lt(0.05)]
    measurements = core["tables"]["measurements"]
    internal_hERG_compounds = int(
        measurements.loc[measurements["endpoint"].eq("herg_ic50"), "compound_id"].nunique()
    )
    pk_endpoint_names = {"auc_0_inf", "auc_0_last", "cmax", "tmax", "vdss", "clearance"}
    internal_pk_compounds = int(
        measurements.loc[measurements["endpoint"].isin(pk_endpoint_names), "compound_id"].nunique()
    )
    external_summary = json.loads(
        (ROOT / "research/reports/pk_herg/herg_external_challenge/challenge_summary.json").read_text(
            encoding="utf-8"
        )
    )

    herg_table = _markdown_table(
        exact_models,
        ["model", "mae", "mae_lower_95", "mae_upper_95", "spearman", "fraction_within_0p5_log"],
        ["Model", "MAE", "MAE 95% low", "MAE 95% high", "Spearman", "Within 0.5 log"],
    )
    pk_table = _markdown_table(
        pk_only,
        [
            "endpoint",
            "n_compounds",
            "mae",
            "mae_lower_95",
            "mae_upper_95",
            "median_fold_error",
            "fraction_within_2fold",
            "spearman",
            "interval_coverage",
        ],
        [
            "Endpoint",
            "n",
            "log MAE",
            "MAE 95% low",
            "MAE 95% high",
            "Median fold error",
            "Within 2-fold",
            "Spearman",
            "PI coverage",
        ],
    )
    threshold_table = _markdown_table(
        threshold_metrics,
        [
            "ic50_threshold_um",
            "n",
            "n_blockers",
            "n_nonblockers",
            "roc_auc",
            "balanced_accuracy",
            "sensitivity",
            "specificity",
            "brier",
        ],
        [
            "IC50 cutoff (uM)",
            "n",
            "Blockers",
            "Nonblockers",
            "ROC-AUC",
            "Balanced accuracy",
            "Sensitivity",
            "Specificity",
            "Brier",
        ],
    )
    censor_table = _markdown_table(
        censor_metrics[
            censor_metrics["censoring_scenario"].isin(
                [
                    "correct_interval_censoring",
                    "drop_censored_rows",
                    "incorrect_boundary_as_exact_sensitivity",
                ]
            )
        ],
        ["censoring_scenario", "training_rows", "mae", "spearman", "mean_signed_error"],
        ["Treatment", "Training rows", "Extension MAE", "Spearman", "Bias"],
    )
    claim_table = _markdown_table(
        claim_ledger,
        ["status", "claim", "allowed_use"],
        ["Status", "Claim tested", "Allowed use"],
    )
    learning_latest = (
        learning_summary.sort_values("training_scaffold_fraction")
        .groupby(["domain", "endpoint"], as_index=False)
        .tail(1)
    )
    learning_table = _markdown_table(
        learning_latest,
        [
            "domain",
            "endpoint",
            "median_training_compounds",
            "median_training_scaffolds",
            "median_mae",
            "median_spearman",
        ],
        ["Analysis", "Endpoint", "Training compounds", "Training scaffolds", "Median MAE", "Median Spearman"],
    )
    sample_top = sample_size.sort_values(["domain", "additional_groups_needed"], ascending=[True, False])
    sample_table = _markdown_table(
        sample_top,
        [
            "domain",
            "endpoint",
            "planning_quantity",
            "current_independent_groups",
            "estimated_independent_groups_needed",
            "additional_groups_needed",
        ],
        ["Domain", "Endpoint", "Planning target", "Current groups", "Estimated groups", "Additional"],
    )
    selected_best = (
        selected_vs_fixed.sort_values(["endpoint", "fixed_mae"]).groupby("endpoint", as_index=False).first()
    )
    herg_target_mismatch = selected_vs_fixed[selected_vs_fixed["endpoint"].eq("herg_pic50")][
        "target_definition_mismatch_compounds"
    ].max()
    herg_target_delta = selected_vs_fixed[selected_vs_fixed["endpoint"].eq("herg_pic50")][
        "maximum_target_definition_delta_log"
    ].max()
    selected_vs_mean = selected_vs_fixed[selected_vs_fixed["fixed_comparator"].eq("train_mean")]
    selected_beats_mean_count = int(selected_vs_mean["selected_conclusively_better"].sum())
    fixed_beats_selected_count = int(selected_vs_fixed["fixed_conclusively_better"].sum())
    selected_table = _markdown_table(
        selected_best,
        [
            "endpoint",
            "selected_model",
            "selected_mae",
            "fixed_comparator",
            "fixed_mae",
            "selected_minus_fixed_mae",
            "delta_mae_lower_95",
            "delta_mae_upper_95",
        ],
        [
            "Endpoint",
            "Selected model",
            "Selected MAE",
            "Best fixed comparator",
            "Fixed MAE",
            "Selected-fixed delta",
            "Delta 95% low",
            "Delta 95% high",
        ],
    )
    trend_table = _markdown_table(
        learning_trends,
        [
            "analysis",
            "endpoint",
            "mean_high_minus_low_mae",
            "high_minus_low_mae_lower_95",
            "high_minus_low_mae_upper_95",
            "permutation_p_two_sided",
        ],
        ["Analysis", "Endpoint", "High-low MAE", "95% low", "95% high", "Permutation p"],
    )
    residual_bias_survivors = residual_bias["within_group_fdr_q"].lt(0.05).sum()

    report = f"""# PK/hERG program health audit

## Direct answer: what is going well?

The program is going well as a **rigorous analogue-series research program**, not yet as a
general PK/hERG prediction platform. The strongest accomplishments are the scientific
boundaries: derived-label leakage is controlled, censored hERG values are retained correctly,
compound/scaffold-held-out predictions replace random-row validation, trivial and nearest-
neighbor comparators are explicit, and every local physics proxy is fail-closed until it has
equilibrium or kinetic support. This makes the positive results substantially more credible.

The strongest quantitative result is the non-overlapping, same-series hERG extension. On 42
exact IC50 values across 8 computed scaffolds, the complete structure representation has MAE
**{complete["mae"]:.3f} log**, scaffold-bootstrap 95% CI
**[{complete["mae_lower_95"]:.3f}, {complete["mae_upper_95"]:.3f}]**, Spearman
**{complete["spearman"]:.3f}**, and **{complete["fraction_within_0p5_log"]:.1%}** of predictions
within 0.5 log unit. At the 10 uM threshold, retrospective ROC-AUC is
**{threshold_10["roc_auc"]:.3f}** and balanced accuracy is
**{threshold_10["balanced_accuracy"]:.3f}**. The essential caveat is that this representation
was conceived after the extension outcomes were inspected, so these are strong
representation-development results, not prospective validation.

Rat PK also contains real analogue signal. The most accurate compound-balanced endpoint is
Vdss (MAE **{pk_only.set_index("endpoint").loc["vdss", "mae"]:.3f} log**, median absolute error
**{pk_only.set_index("endpoint").loc["vdss", "median_fold_error"]:.2f}-fold**). Dose-normalized
PO Cmax and PO AUC show useful rank ordering, although PO Cmax remains discovery-only until
within-compound dose proportionality is available. The negative controls support genuine
structure signal for {", ".join(significant_null_controls["endpoint"].astype(str))}; IV AUC
and Tmax remain materially weaker.

## Evidence breadth actually used

- **Internal chemistry:** {len(core["compounds"])} unique structures and
  {len(measurements)} normalized measurement rows.
- **Rat PK:** {internal_pk_compounds} compounds have some PK evidence; the balanced endpoint models
  use 46 compounds because unresolved links and ineligible records are excluded.
- **Internal hERG:** {internal_hERG_compounds} compounds have IC50 evidence; exact compound-balanced
  OOF analysis uses 63 structures.
- **Angelo/Ascentage extension:** 76 structures total, including 22 exact training overlaps;
  the non-overlap evaluation contains 42 exact, four `>30 uM` censored, and eight unsynthesized
  structures.
- **Public hERG stress test:** {external_summary["public_exact_structures"]} exact public structures,
  including {external_summary["large_molecule_ambiguity_quarantined_structures"]} curated
  MW>=650 structures after ambiguity quarantine.
- **Mechanistic depth:** zero raw PK concentration-time samples, zero dynamic hERG onset/recovery
  records, and zero physics observables admitted to models. These are explicit absence findings,
  not silently imputed data.

## What the new tests show

### hERG fixed comparators

{herg_table}

The complete representation's paired scaffold-bootstrap MAE difference versus global controls
is **{delta_global["mean_delta_mae"]:.3f}**, 95% CI
**[{delta_global["delta_mae_lower_95"]:.3f}, {delta_global["delta_mae_upper_95"]:.3f}]**.
An interval crossing zero means the incremental advantage is not yet conclusive even when the
point estimate is better. The local-neighbor rows show how much of the extension is explainable
by close analogue interpolation; this is the correct comparator for Angelo's same-series set.
The equal-weight consensus reaches MAE **{consensus["mae"]:.3f}** with Spearman
**{consensus["spearman"]:.3f}**, consistent with partially complementary global and local errors,
but it was examined after outcomes were available and is therefore a prospective candidate, not
a promoted model.

The apparent eight-scaffold sample is not eight independent medicinal-chemistry series. At
Tanimoto 0.70, the 42 extension structures form **{int(topology_070["n_components"])}** connected
component(s), with **{topology_070["largest_component_fraction"]:.1%}** in the largest; Dr. Aguilar
also confirms one series. Consequently, all scaffold/component bootstrap intervals here estimate
within-series heterogeneity only. They cannot estimate new-series generalization.

### hERG threshold and calibration stress test

{threshold_table}

Threshold performance is not a single immutable number. Class balance changes sharply across
1, 3, 10, and 30 uM, and the small nonblocker counts produce wide Wilson intervals. At 10 uM,
sensitivity is **{threshold_10["sensitivity"]:.1%}** (Wilson 95%
**[{threshold_10["sensitivity_wilson_lower_95"]:.1%}, {threshold_10["sensitivity_wilson_upper_95"]:.1%}]**)
and specificity is **{threshold_10["specificity"]:.1%}** (Wilson 95%
**[{threshold_10["specificity_wilson_lower_95"]:.1%}, {threshold_10["specificity_wilson_upper_95"]:.1%}]**).
This supports continuous pIC50 plus probability/uncertainty output, not a bare blocker label.
The mean predicted blocker probability is **{calibration_summary["mean_predicted_probability"]:.3f}**
versus observed prevalence **{calibration_summary["observed_prevalence"]:.3f}**. The complete
90% interval covers **{interval_complete["interval_coverage"]:.1%}**, but its mean width is
**{interval_complete["interval_mean_width_log10"]:.3f} log**
(**{interval_complete["interval_width_to_observed_range"]:.2f}x** the observed extension range).
Its half-width is **{interval_complete["current_to_post_outcome_radius_ratio"]:.2f}x** the
post-outcome empirical 90th-percentile error radius. That ratio is diagnostic only—it cannot be
used to recalibrate on the same extension—but confirms that the interval is conservative,
inefficient, and not decision-grade calibration.

### Censoring and numerical stability

{censor_table}

The correct primary treatment uses the likelihood contribution of each limit; dropping limits
discards information, while treating a limit as an exact value invents a measurement. The
primary treatment's same-series extension MAE is **{censor_primary["mae"]:.3f}**.
Across {len(seed_metrics)} randomized SVD seeds, extension MAE ranges from
**{seed_metrics["mae"].min():.3f}** to **{seed_metrics["mae"].max():.3f}** and the median
per-compound prediction SD is **{seed_compounds["seed_prediction_sd"].median():.4f} log**.
This separates numerical representation uncertainty from biological/model uncertainty.

### PK compound-balanced performance

{pk_table}

These are fixed group-held-out compound predictions, not repeated evidence rows. The bootstrap
resamples computed scaffolds. Median fold error is interpretable, but it is not a claim of
prospective accuracy. Applicability-domain performance is generally better inside the current
chemical neighborhood, except Tmax; the small number of inside-domain compounds prevents an
accuracy guarantee.

### Does model complexity add value over fixed simple baselines?

{selected_table}

A positive selected-minus-fixed delta means the selected nonlinear model is worse. The table is
important because strong absolute performance can still be explained by local fingerprint
interpolation. Model complexity is scientifically useful only where it adds stable information or
mechanistic diagnostics; it is not rewarded merely for being complex. The chosen “best fixed” row
is descriptive across prespecified comparators and is not another model-selection exercise.
The selected models conclusively beat their held-scaffold train-mean baseline for
**{selected_beats_mean_count}/{len(selected_vs_mean)}** endpoints, so there is real structure
signal. However, **{fixed_beats_selected_count}** prespecified simple-model comparison(s)
conclusively favor the fixed baseline, and no selected model conclusively beats the point-best
fixed comparator in the table. The appropriate success claim is therefore “predictive local
structure signal,” not “complex ML superiority.”
For internal hERG, {int(herg_target_mismatch)} compound has a small target-collapse difference
(maximum **{herg_target_delta:.4f} log**) between artifacts; both predictions are evaluated against
the production OOF target and the discrepancy is retained as an audit field.

### Learning behavior and value of more data

{learning_table}

{trend_table}

The learning curves use outcome-blind training-scaffold subsets and fixed models. They answer
whether adding chemically distinct training groups improves a held-scaffold or fixed-extension
test, not whether simply increasing the number of nearly identical analogues will solve the
problem. The planning sensitivities below are deliberately expressed in **independent chemical
groups**, because 1000 compounds from one connected series do not provide 1000 independent
tests.

{sample_table}

These estimates are sensitivities, not promises: they use observed scaffold-level variability,
normal approximations, and computed rather than chemist-validated series labels. Their main
message is robust: additional series/protocol strata are more valuable than more close analogs.

## Mechanistic findings that are genuinely useful

- **Shared PK residual structure:** {len(residual_survivors)} residual associations survive
  within-scaffold FDR. The strongest is PO AUC/dose versus Cmax/dose
  (within-scaffold rho **{residual_survivors.iloc[0]["within_scaffold_spearman"]:.3f}**), followed
  by IV AUC versus Cmax/dose. This localizes a shared unmodeled exposure/study process, but does
  not identify permeability, clearance, or formulation as the cause.
- **No demonstrated PK-hERG tradeoff:** among {int(joint["n"].max())} overlapping compounds,
  **{int(joint["within_scaffold_fdr_005"].sum())}** joint associations survive within-scaffold
  FDR. The apparent overall Tmax-hERG association collapses after scaffold control.
- **Closure catches source problems:** clearance has {int(closure.set_index("endpoint").loc["clearance", "failed_compounds"])}
  failing compounds and reported bioavailability has
  {int(closure.set_index("endpoint").loc["bioavailability", "failed_compounds"])}; these are QC
  diagnostics, not independent model endpoints.
- **Selection bias is visible:** {len(selection_survivors)} descriptor/availability comparisons
  survive FDR, all in PK availability. Current PK performance therefore describes the historically
  measured chemistry, not all 110 structures.
- **No obvious residual hERG subgroup failure survived multiplicity control:**
  {int(residual_bias_survivors)} of {len(residual_bias)} tested error-feature relationships survive
  within-scaffold FDR. This is reassuring but low-powered, especially because only six extension
  scaffolds contain within-scaffold contrasts.
- **Physics governance is a success:** {physics_summary["candidate_observables_reviewed"]} candidate
  observables/proxies were audited and **{physics_summary["observables_admitted_to_models"]}** were
  admitted. Refusing to promote unconverged xTB minima, arbitrary microstates, or coordinate RMSDs
  prevents false mechanism claims.

## What is not yet going well

- Unrelated large-molecule hERG transfer is falsified: all 56 curated public large structures are
  outside the internal domain and maximum internal similarity is only 0.309.
- Prediction intervals are not prospectively calibrated; several are inefficiently wide and some
  under-cover. They are warnings, not guarantees.
- PK is summary-only: there are zero concentration-time samples, so absorption, distribution,
  clearance, and first-pass components are not separately identifiable.
- Static hERG IC50 does not identify access, onset, recovery, state dependence, or trapping.
- The normalized Ascentage table is reproducible, but its original DOCX and embedded CDX binaries
  were removed from staging and must be reacquired for a complete raw-evidence publication archive.
- There is no defensible single MW cutoff and no admitted equilibrium/kinetic physics feature.
- The internal chemistry is one Tanimoto-0.70 connected component and lacks chemist-validated
  series labels/synthesis chronology. Nominal scaffold CV is therefore less stringent than a true
  new-series test.

## Immediate decisions

1. Keep the hERG complete model and global-control model together. Report their envelope and
   disagreement; do not promote the complete model on this retrospective extension alone.
2. Blind-lock the next Angelo set before labels are exposed. Preserve assay protocol, exact limits,
   and explicit series membership. Evaluate continuous pIC50 first.
3. Prioritize new nonblockers/intermediate compounds and genuinely distinct series; they improve
   specificity/calibration and transfer evidence more than more strong blockers from the same core.
4. For PK, request raw rat IV/PO curves, formulation, animal metadata, LLOQ, PPB/fu, blood:plasma,
   solubility, permeability, and hepatocyte/microsomal CLint. Without these, PBPK components remain
   sensitivity hypotheses.
5. On HPC, start with site-resolved chemical-state calibration and adaptive conformer free-energy
   sampling. Only then launch replicated environment MD/transition-network tests; PMF and receptor
   MD remain downstream of those gates.

## Claim ledger

{claim_table}

## Reproducibility and truth boundary

This audit uses all currently usable internal measurements, the non-overlap Ascentage extension,
the public large-molecule stress test, fixed OOF artifacts, closure/protocol audits, and local
physics admission records. It does not execute Menin-Edit, generate molecules, treat extension
outcomes as prospective, infer missing assay values, or substitute laptop approximations for HPC
physics. Detailed machine-readable results are indexed in `README.md` in this directory.
"""
    atomic_write_text(OUTPUT / "program_health_audit.md", report)


def run() -> dict[str, Any]:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    core = _load_core()
    data_tables, data_endpoints = _data_inventory(core)
    herg_comparison, herg_bootstrap = _herg_extension_baselines(core)
    herg_interval, herg_interval_conditional = _herg_interval_audit(core)
    herg_topology, herg_group_sensitivity = _herg_extension_topology(core)
    threshold_metrics, threshold_bootstrap = _threshold_metrics(core)
    calibration_bins, calibration_summary = _calibration_audit(core)
    censor_metrics, censor_predictions = _censoring_sensitivity(core)
    seed_metrics, seed_compounds = _representation_seed_stability(core)
    residual_bias = _herg_residual_bias(core)
    herg_curve = _herg_extension_learning_curve(core)
    pk_metrics, pk_bootstrap = _pk_compound_metrics()
    loco_metrics, loco_best = _fixed_loco_summary()
    selected_vs_fixed = _selected_vs_fixed_baselines()
    pk_curve = _pk_learning_curve(core)
    learning_summary = _learning_curve_summary(herg_curve, pk_curve)
    learning_trends = _learning_curve_trends(herg_curve, pk_curve)
    sample_size = _sample_size_sensitivity(pk_metrics, core["novel"])
    physics_admission, physics_summary = _physics_gate_summary(core)
    claims = _claim_ledger(
        core=core,
        pk_metrics=pk_metrics,
        herg_comparison=herg_comparison,
        threshold_metrics=threshold_metrics,
        seed_metrics=seed_metrics,
        censor_metrics=censor_metrics,
        learning_summary=learning_summary,
        physics_summary=physics_summary,
    )

    atomic_write_csv(OUTPUT / "canonical_table_inventory.csv", data_tables)
    atomic_write_csv(OUTPUT / "measurement_endpoint_inventory.csv", data_endpoints)
    atomic_write_csv(OUTPUT / "herg_fixed_comparator_metrics.csv", herg_comparison)
    atomic_write_parquet(OUTPUT / "herg_fixed_comparator_bootstrap.parquet", herg_bootstrap)
    atomic_write_csv(OUTPUT / "herg_interval_efficiency.csv", herg_interval)
    atomic_write_csv(OUTPUT / "herg_interval_conditional.csv", herg_interval_conditional)
    atomic_write_csv(OUTPUT / "herg_extension_group_topology.csv", herg_topology)
    atomic_write_csv(OUTPUT / "herg_comparator_group_sensitivity.csv", herg_group_sensitivity)
    atomic_write_csv(OUTPUT / "herg_threshold_sensitivity.csv", threshold_metrics)
    atomic_write_parquet(OUTPUT / "herg_threshold_bootstrap.parquet", threshold_bootstrap)
    atomic_write_csv(OUTPUT / "herg_calibration_bins.csv", calibration_bins)
    atomic_write_json(OUTPUT / "herg_calibration_summary.json", calibration_summary)
    atomic_write_csv(OUTPUT / "herg_censoring_sensitivity.csv", censor_metrics)
    atomic_write_parquet(OUTPUT / "herg_censoring_predictions.parquet", censor_predictions)
    atomic_write_csv(OUTPUT / "herg_svd_seed_stability.csv", seed_metrics)
    atomic_write_csv(OUTPUT / "herg_svd_seed_compound_stability.csv", seed_compounds)
    atomic_write_csv(OUTPUT / "herg_extension_residual_bias.csv", residual_bias)
    atomic_write_parquet(OUTPUT / "herg_learning_curve_detail.parquet", herg_curve)
    atomic_write_csv(OUTPUT / "pk_compound_balanced_metrics.csv", pk_metrics)
    atomic_write_parquet(OUTPUT / "pk_compound_balanced_bootstrap.parquet", pk_bootstrap)
    atomic_write_csv(OUTPUT / "fixed_loco_all_metrics.csv", loco_metrics)
    atomic_write_csv(OUTPUT / "fixed_loco_best_summary.csv", loco_best)
    atomic_write_csv(OUTPUT / "selected_vs_fixed_baselines.csv", selected_vs_fixed)
    atomic_write_parquet(OUTPUT / "pk_learning_curve_detail.parquet", pk_curve)
    atomic_write_csv(OUTPUT / "learning_curve_summary.csv", learning_summary)
    atomic_write_csv(OUTPUT / "learning_curve_trends.csv", learning_trends)
    atomic_write_csv(OUTPUT / "independent_data_planning_sensitivity.csv", sample_size)
    atomic_write_csv(OUTPUT / "physics_admission_snapshot.csv", physics_admission)
    atomic_write_csv(OUTPUT / "claim_ledger.csv", claims)

    _write_figures(pk_metrics, herg_comparison, learning_summary)
    _write_calibration_figure(calibration_bins)
    _write_report(
        core=core,
        data_tables=data_tables,
        data_endpoints=data_endpoints,
        pk_metrics=pk_metrics,
        herg_comparison=herg_comparison,
        herg_interval=herg_interval,
        herg_topology=herg_topology,
        herg_group_sensitivity=herg_group_sensitivity,
        threshold_metrics=threshold_metrics,
        calibration_summary=calibration_summary,
        censor_metrics=censor_metrics,
        seed_metrics=seed_metrics,
        seed_compounds=seed_compounds,
        residual_bias=residual_bias,
        loco_best=loco_best,
        selected_vs_fixed=selected_vs_fixed,
        learning_summary=learning_summary,
        learning_trends=learning_trends,
        sample_size=sample_size,
        physics_summary=physics_summary,
        claim_ledger=claims,
    )

    exact_extension = core["novel"][core["novel"]["herg_pic50_relation"].eq("=")]
    censored_extension = core["novel"][core["novel"]["herg_pic50_relation"].eq("<")]
    validation_checks = {
        "canonical_internal_structure_count_is_110": len(core["compounds"]) == 110,
        "baseline_hERG_evidence_rows_are_94": len(core["potency"]) == 94,
        "extension_nonoverlap_structures_are_54": len(core["novel"]) == 54,
        "extension_exact_values_are_42": len(exact_extension) == 42,
        "extension_right_censored_values_are_4": len(censored_extension) == 4,
        "extension_unsynthesized_values_are_8": int(
            core["novel"]["herg_pic50_relation"].fillna("").astype(str).str.strip().eq("").sum()
        )
        == 8,
        "all_svd_seed_models_converged": bool(seed_metrics["fit_converged"].all()),
        "all_censoring_sensitivity_models_converged": bool(
            censor_metrics.loc[censor_metrics["fit_converged"].notna(), "fit_converged"].all()
        ),
        "all_hERG_thresholds_evaluable": bool(threshold_metrics["n"].ge(10).all()),
        "thirty_uM_includes_four_definite_censored_nonblockers": int(
            threshold_metrics.loc[
                np.isclose(threshold_metrics["ic50_threshold_um"], 30.0),
                "n_censored_definite_nonblockers",
            ].iloc[0]
        )
        == 4,
        "all_pk_endpoints_have_compound_balanced_predictions": set(
            [
                "iv_auc_dose_normalized",
                "po_auc_dose_normalized",
                "vdss",
                "po_cmax_dose_normalized",
                "po_tmax",
            ]
        ).issubset(set(pk_metrics["endpoint"])),
        "no_physics_observable_admitted": physics_summary["observables_admitted_to_models"] == 0,
        "hERG_interval_audit_has_both_retained_models": set(herg_interval["model"])
        == {"global_controls", "complete_feature"},
        "hERG_extension_is_one_component_at_tanimoto_0p70": int(
            herg_topology.loc[np.isclose(herg_topology["tanimoto_threshold"], 0.70), "n_components"].iloc[0]
        )
        == 1,
        "selected_vs_fixed_comparisons_cover_all_endpoints": set(pk_metrics["endpoint"])
        == set(selected_vs_fixed["endpoint"]),
        "calibration_summary_is_finite": bool(
            np.isfinite(calibration_summary["brier_direct"])
            and np.isfinite(calibration_summary["calibration_in_the_large_observed_minus_predicted"])
        ),
        "canonical_physics_tables_remain_empty": (
            physics_summary["canonical_physics_observable_rows"] == 0
            and physics_summary["canonical_physics_run_rows"] == 0
        ),
        "claim_ledger_has_no_unqualified_general_model_claim": not claims["status"].eq("generalizable").any(),
    }
    validation = {
        "status": "pass" if all(validation_checks.values()) else "fail",
        "checks": validation_checks,
        "random_seed": SEED,
        "scaffold_bootstrap_replicates": BOOTSTRAPS,
        "within_scaffold_permutations": PERMUTATIONS,
        "selection_boundary": (
            "No extension outcome selected a representation, hyperparameter, threshold, "
            "learning subset, or promoted model. The equal-weight consensus is explicitly "
            "post-outcome exploratory and requires a newly locked prospective set."
        ),
        "physics_boundary": (
            "No local proxy was admitted as equilibrium, kinetic, membrane, receptor-binding, "
            "or optimizer feature."
        ),
    }
    atomic_write_json(OUTPUT / "validation_report.json", validation)
    if validation["status"] != "pass":
        failed = [name for name, passed in validation_checks.items() if not passed]
        raise RuntimeError(f"Program-health validation failed: {failed}")

    complete = herg_comparison[herg_comparison["model"].eq("complete_feature")].iloc[0]
    consensus = herg_comparison[herg_comparison["model"].eq("equal_weight_consensus")].iloc[0]
    threshold_10 = threshold_metrics.loc[np.isclose(threshold_metrics["ic50_threshold_um"], 10.0)].iloc[0]
    summary = {
        "status": "program_health_audit_complete",
        "overall_conclusion": (
            "Strong, carefully bounded analogue-series signal and research infrastructure; "
            "not yet a generalizable or decision-track PK/hERG platform."
        ),
        "internal_structures": int(len(core["compounds"])),
        "pk_compound_count": int(
            pk_metrics.loc[pk_metrics["endpoint"].eq("iv_auc_dose_normalized"), "n_compounds"].iloc[0]
        ),
        "hERG_nonoverlap_extension_exact_count": int(len(exact_extension)),
        "hERG_complete_feature_mae": float(complete["mae"]),
        "hERG_complete_feature_mae_95ci": [
            float(complete["mae_lower_95"]),
            float(complete["mae_upper_95"]),
        ],
        "hERG_complete_feature_spearman": float(complete["spearman"]),
        "hERG_equal_weight_consensus_mae_post_outcome_sensitivity": float(consensus["mae"]),
        "hERG_10uM_roc_auc": float(threshold_10["roc_auc"]),
        "hERG_10uM_balanced_accuracy": float(threshold_10["balanced_accuracy"]),
        "physics_features_admitted": int(physics_summary["observables_admitted_to_models"]),
        "hERG_extension_tanimoto_0p70_components": int(
            herg_topology.loc[np.isclose(herg_topology["tanimoto_threshold"], 0.70), "n_components"].iloc[0]
        ),
        "claim_status_counts": claims["status"].value_counts().to_dict(),
    }
    atomic_write_json(OUTPUT / "program_health_summary.json", summary)

    readme = """# Program-health audit artifacts

- `program_health_audit.md`: integrated scientific interpretation and next decisions.
- `program_health_summary.json`: machine-readable headline results.
- `claim_ledger.csv`: claim-by-claim evidence boundary and allowed use.
- `herg_fixed_comparator_metrics.csv`: complete representation versus mean, 1/3/5-NN, and global controls.
- `herg_interval_efficiency.csv`: coverage, sharpness, width/range, and 90% interval score.
- `herg_extension_group_topology.csv`: outcome-blind similarity-component sensitivity.
- `herg_comparator_group_sensitivity.csv`: model-difference uncertainty under alternate chemical groups.
- `herg_threshold_sensitivity.csv`: 1/3/10/30 uM classification and calibration stress test.
- `herg_calibration_bins.csv`: equal-count 10 uM reliability bins with Wilson intervals.
- `herg_censoring_sensitivity.csv`: correct, dropped, and boundary-as-exact treatments.
- `herg_svd_seed_stability.csv`: numerical representation stability.
- `herg_extension_residual_bias.csv`: overall and within-scaffold error correlations.
- `pk_compound_balanced_metrics.csv`: log, fold-error, rank, and interval metrics.
- `fixed_loco_best_summary.csv`: fixed baseline advantage over train-mean under scaffold/component splits.
- `selected_vs_fixed_baselines.csv`: selected nonlinear models versus all fixed simple baselines.
- `learning_curve_summary.csv`: chemical-group learning behavior.
- `learning_curve_trends.csv`: high-versus-low coverage effects and permutation sensitivities.
- `independent_data_planning_sensitivity.csv`: independent-series/scaffold planning sensitivities.
- `physics_admission_snapshot.csv`: explicit fail-closed physics evidence boundary.
- `validation_report.json`: acceptance checks and selection boundaries.
- `local_execution_validation.md`: independently executed full-suite and blind-input smoke checks.
- `model_health_overview.png`, `learning_curve_overview.png`, and `herg_calibration.png`: publication-ready overview figures.

Bootstrap details are stored as Parquet to avoid treating resampled rows as new observations.
"""
    atomic_write_text(OUTPUT / "README.md", readme)
    return summary


def main() -> None:
    print(json.dumps(run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
