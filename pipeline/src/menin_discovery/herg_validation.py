"""Nested scaffold validation for the private/public hERG benchmark.

Outer folds estimate performance on unseen confidential scaffolds.  Model,
feature, hyperparameter, calibration, threshold, and ensemble choices are made
using inner confidential-scaffold folds only.  Public hERG structures are
available to hybrid regimes but never enter the private-domain evaluation set.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold

from .herg_benchmark import (
    REGIMES,
    ModelSpec,
    _feature_slice,
    _fit_model,
    _jsonable,
    _predict_model,
    _regime_training_indices,
    _save_model,
    _selection_score,
    _sha256_file,
    build_model_specs,
    calculate_feature_registry,
    load_private_workbook,
    load_public_herg,
)


@dataclass(frozen=True)
class Candidate:
    spec: ModelSpec
    feature_set: str

    @property
    def key(self) -> str:
        return f"{self.spec.key}__{self.feature_set}"


@dataclass
class SigmoidCalibrator:
    estimator: LogisticRegression | None

    def transform(self, probability: np.ndarray) -> np.ndarray:
        probability = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
        if self.estimator is None:
            return probability
        logits = np.log(probability / (1 - probability)).reshape(-1, 1)
        return np.clip(self.estimator.predict_proba(logits)[:, 1], 1e-7, 1 - 1e-7)

    @property
    def slope(self) -> float:
        return 1.0 if self.estimator is None else float(self.estimator.coef_[0, 0])

    @property
    def intercept(self) -> float:
        return 0.0 if self.estimator is None else float(self.estimator.intercept_[0])


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(dict(payload)), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _candidate_rows(specs: Sequence[ModelSpec]) -> list[Candidate]:
    return [
        Candidate(spec=spec, feature_set=feature_set) for spec in specs for feature_set in spec.feature_sets
    ]


def _scaffold_folds(
    data: pd.DataFrame,
    *,
    n_splits: int,
    random_state: int,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], dict[str, Any]]:
    """Find deterministic viable stratified scaffold folds without row-level fallback."""

    y = data["herg_blocker_label"].to_numpy(dtype=int)
    groups = data["scaffold"].fillna("").astype(str).to_numpy()
    attempts: list[dict[str, Any]] = []
    for resolved_splits in range(min(n_splits, len(np.unique(groups))), 1, -1):
        for seed_offset in range(50):
            seed = random_state + seed_offset
            splitter = StratifiedGroupKFold(
                n_splits=resolved_splits,
                shuffle=True,
                random_state=seed,
            )
            folds = list(splitter.split(np.zeros(len(data)), y, groups=groups))
            class_counts = [
                {
                    "fold": fold_index,
                    "n_train": int(len(train)),
                    "n_test": int(len(test)),
                    "test_blockers": int(np.sum(y[test] == 1)),
                    "test_nonblockers": int(np.sum(y[test] == 0)),
                    "test_scaffolds": int(len(np.unique(groups[test]))),
                }
                for fold_index, (train, test) in enumerate(folds)
            ]
            viable = all(
                len(np.unique(y[train])) == 2 and len(np.unique(y[test])) == 2 for train, test in folds
            )
            attempts.append(
                {
                    "n_splits": resolved_splits,
                    "seed": seed,
                    "viable": viable,
                    "class_counts": class_counts,
                }
            )
            if viable:
                return folds, {
                    "requested_splits": int(n_splits),
                    "resolved_splits": int(resolved_splits),
                    "random_state": int(seed),
                    "n_scaffolds": int(len(np.unique(groups))),
                    "folds": class_counts,
                    "n_attempts": int(len(attempts)),
                }
    raise ValueError(f"Unable to create viable scaffold folds after {len(attempts)} attempts")


def _fit_sigmoid_calibrator(y_true: np.ndarray, raw_probability: np.ndarray) -> SigmoidCalibrator:
    y_true = np.asarray(y_true, dtype=int)
    probability = np.clip(np.asarray(raw_probability, dtype=float), 1e-6, 1 - 1e-6)
    if len(np.unique(y_true)) < 2 or float(np.std(probability)) < 1e-8:
        return SigmoidCalibrator(estimator=None)
    logits = np.log(probability / (1 - probability)).reshape(-1, 1)
    estimator = LogisticRegression(C=1e3, solver="lbfgs", max_iter=2000, random_state=13)
    estimator.fit(logits, y_true)
    return SigmoidCalibrator(estimator=estimator)


def _optimal_threshold(y_true: np.ndarray, probability: np.ndarray) -> tuple[float, float]:
    y_true = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    candidates = np.unique(
        np.concatenate(
            [
                np.linspace(0.05, 0.95, 181),
                np.quantile(probability, np.linspace(0.05, 0.95, 37)),
            ]
        )
    )
    scored = [
        (float(balanced_accuracy_score(y_true, probability >= threshold)), float(threshold))
        for threshold in candidates
    ]
    best_score, best_threshold = max(scored, key=lambda item: (item[0], -abs(item[1] - 0.5)))
    return best_threshold, best_score


def _expected_calibration_error(y_true: np.ndarray, probability: np.ndarray, n_bins: int = 8) -> float:
    edges = np.linspace(0, 1, n_bins + 1)
    ids = np.clip(np.digitize(probability, edges[1:-1], right=True), 0, n_bins - 1)
    result = 0.0
    for bin_id in range(n_bins):
        mask = ids == bin_id
        if np.any(mask):
            result += float(np.mean(mask)) * abs(
                float(np.mean(y_true[mask])) - float(np.mean(probability[mask]))
            )
    return float(result)


def _probability_calibration_parameters(
    y_true: np.ndarray,
    probability: np.ndarray,
) -> tuple[float, float]:
    probability = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    logits = np.log(probability / (1 - probability))
    if len(np.unique(y_true)) < 2 or float(np.std(logits)) < 1e-8:
        return np.nan, np.nan
    model = LogisticRegression(C=1e3, solver="lbfgs", max_iter=2000, random_state=13)
    model.fit(logits.reshape(-1, 1), np.asarray(y_true, dtype=int))
    return float(model.coef_[0, 0]), float(model.intercept_[0])


def validation_metrics(
    y_true: np.ndarray,
    probability: np.ndarray,
    *,
    predicted_label: np.ndarray | None = None,
    threshold: float = 0.5,
) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=int)
    probability = np.clip(np.asarray(probability, dtype=float), 1e-7, 1 - 1e-7)
    predicted = (
        np.asarray(predicted_label, dtype=int)
        if predicted_label is not None
        else (probability >= threshold).astype(int)
    )
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    both = len(np.unique(y_true)) == 2
    slope, intercept = _probability_calibration_parameters(y_true, probability)
    return {
        "roc_auc": float(roc_auc_score(y_true, probability)) if both else np.nan,
        "pr_auc": float(average_precision_score(y_true, probability)) if both else np.nan,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "mcc": float(matthews_corrcoef(y_true, predicted)),
        "sensitivity": float(recall_score(y_true, predicted, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else np.nan,
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "brier": float(brier_score_loss(y_true, probability)),
        "ece_8bin": _expected_calibration_error(y_true, probability, n_bins=8),
        "calibration_slope": slope,
        "calibration_intercept": intercept,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "n": int(len(y_true)),
    }


def _prediction_frame_metrics(group: pd.DataFrame) -> dict[str, Any]:
    """Score calibrated decisions while preserving raw-score discrimination.

    Fold-specific calibration is required for Brier/ECE and probability-based
    decisions, but it can alter the ordering of observations *between* outer
    folds.  ROC/PR discrimination is therefore reported from each fitted
    model's raw score, with calibrated ROC/PR retained as sensitivity fields.
    """

    y_true = group["observed_label"].to_numpy(dtype=int)
    probability = group["probability"].to_numpy(dtype=float)
    predicted = group["predicted_label"].to_numpy(dtype=int)
    raw_probability = (
        group["raw_probability"].to_numpy(dtype=float) if "raw_probability" in group else probability
    )
    calibrated = validation_metrics(y_true, probability, predicted_label=predicted)
    raw = validation_metrics(y_true, raw_probability, predicted_label=predicted)
    calibrated["calibrated_roc_auc"] = calibrated["roc_auc"]
    calibrated["calibrated_pr_auc"] = calibrated["pr_auc"]
    calibrated["raw_roc_auc"] = raw["roc_auc"]
    calibrated["raw_pr_auc"] = raw["pr_auc"]
    calibrated["roc_auc"] = raw["roc_auc"]
    calibrated["pr_auc"] = raw["pr_auc"]
    return calibrated


def _nearest_tanimoto(
    query: sparse.csr_matrix,
    reference: sparse.csr_matrix,
    *,
    exclude_diagonal: bool = False,
) -> np.ndarray:
    query = query.tocsr().astype(np.float32)
    reference = reference.tocsr().astype(np.float32)
    intersections = (query @ reference.T).toarray()
    query_counts = np.asarray(query.sum(axis=1)).ravel()[:, None]
    reference_counts = np.asarray(reference.sum(axis=1)).ravel()[None, :]
    unions = query_counts + reference_counts - intersections
    similarities = np.divide(
        intersections,
        unions,
        out=np.zeros_like(intersections, dtype=float),
        where=unions > 0,
    )
    if exclude_diagonal and similarities.shape[0] == similarities.shape[1]:
        np.fill_diagonal(similarities, -np.inf)
    return np.maximum(0.0, np.max(similarities, axis=1))


def _applicability_values(
    fingerprints: sparse.csr_matrix,
    *,
    train_private: np.ndarray,
    test_private: np.ndarray,
    public_indices: np.ndarray,
    quantile: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    private_train_fp = fingerprints[train_private]
    threshold_values = _nearest_tanimoto(private_train_fp, private_train_fp, exclude_diagonal=True)
    threshold = float(np.quantile(threshold_values, quantile))
    private_similarity = _nearest_tanimoto(fingerprints[test_private], private_train_fp)
    public_similarity = (
        _nearest_tanimoto(fingerprints[test_private], fingerprints[public_indices])
        if len(public_indices)
        else np.zeros(len(test_private), dtype=float)
    )
    return private_similarity, public_similarity, threshold


def _bootstrap_intervals(
    predictions: pd.DataFrame,
    *,
    iterations: int,
    random_state: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    metric_names = (
        "roc_auc",
        "pr_auc",
        "raw_roc_auc",
        "raw_pr_auc",
        "calibrated_roc_auc",
        "calibrated_pr_auc",
        "balanced_accuracy",
        "mcc",
        "sensitivity",
        "specificity",
        "brier",
        "ece_8bin",
    )
    rows: list[dict[str, Any]] = []
    for (regime, strategy), group in predictions.groupby(["regime", "strategy"]):
        scaffolds = group["scaffold"].astype(str).unique()
        samples: dict[str, list[float]] = {metric: [] for metric in metric_names}
        for _ in range(iterations):
            sampled = rng.choice(scaffolds, size=len(scaffolds), replace=True)
            sample = pd.concat([group[group["scaffold"].astype(str) == scaffold] for scaffold in sampled])
            if sample["observed_label"].nunique() < 2:
                continue
            metrics = _prediction_frame_metrics(sample)
            for metric in metric_names:
                value = metrics[metric]
                if value is not None and np.isfinite(value):
                    samples[metric].append(float(value))
        point = _prediction_frame_metrics(group)
        for metric in metric_names:
            values = samples[metric]
            rows.append(
                {
                    "regime": regime,
                    "strategy": strategy,
                    "metric": metric,
                    "point_estimate": point[metric],
                    "lower_95": float(np.quantile(values, 0.025)) if values else np.nan,
                    "upper_95": float(np.quantile(values, 0.975)) if values else np.nan,
                    "successful_resamples": int(len(values)),
                    "resampling_unit": "confidential_scaffold",
                    "n_scaffolds": int(len(scaffolds)),
                }
            )
    return pd.DataFrame(rows)


def _calibration_curve_rows(predictions: pd.DataFrame, n_bins: int = 8) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    edges = np.linspace(0, 1, n_bins + 1)
    for (regime, strategy), group in predictions.groupby(["regime", "strategy"]):
        probability = group["probability"].to_numpy(dtype=float)
        observed = group["observed_label"].to_numpy(dtype=int)
        ids = np.clip(np.digitize(probability, edges[1:-1], right=True), 0, n_bins - 1)
        for bin_id in range(n_bins):
            mask = ids == bin_id
            rows.append(
                {
                    "regime": regime,
                    "strategy": strategy,
                    "bin": bin_id,
                    "lower": float(edges[bin_id]),
                    "upper": float(edges[bin_id + 1]),
                    "n": int(np.sum(mask)),
                    "mean_probability": float(np.mean(probability[mask])) if np.any(mask) else np.nan,
                    "observed_fraction": float(np.mean(observed[mask])) if np.any(mask) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _retained_candidates(
    active: Sequence[Candidate],
    scores: Mapping[str, float],
    *,
    keep_fraction: float,
    minimum_candidates: int,
) -> list[Candidate]:
    ranked = sorted(active, key=lambda candidate: scores.get(candidate.key, -np.inf), reverse=True)
    keep_n = min(len(ranked), max(minimum_candidates, int(np.ceil(len(ranked) * keep_fraction))))
    retained = {candidate.key: candidate for candidate in ranked[:keep_n]}
    for family in sorted({candidate.spec.family for candidate in active} - {"dummy"}):
        family_ranked = [candidate for candidate in ranked if candidate.spec.family == family]
        if family_ranked:
            retained[family_ranked[0].key] = family_ranked[0]
    return sorted(retained.values(), key=lambda candidate: scores.get(candidate.key, -np.inf), reverse=True)


def _candidate_summary(
    rows: Sequence[dict[str, Any]],
    candidates: Sequence[Candidate],
) -> tuple[dict[str, float], dict[str, dict[str, Any]]]:
    frame = pd.DataFrame(rows)
    scores: dict[str, float] = {}
    summaries: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        group = frame[frame["candidate_key"] == candidate.key]
        if group.empty:
            scores[candidate.key] = -np.inf
            continue
        metrics = validation_metrics(
            group["observed_label"].to_numpy(dtype=int),
            group["raw_probability"].to_numpy(dtype=float),
        )
        score = _selection_score(metrics)
        scores[candidate.key] = score
        summaries[candidate.key] = {**metrics, "selection_score": score, "n_inner_predictions": len(group)}
    return scores, summaries


def _model_prediction_for_indices(
    candidate: Candidate,
    *,
    matrices: Mapping[str, sparse.csr_matrix | np.ndarray],
    smiles_values: np.ndarray,
    labels: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    source_weights: np.ndarray,
    random_state: int,
) -> tuple[Any, np.ndarray]:
    fitted = _fit_model(
        candidate.spec,
        _feature_slice(candidate.feature_set, matrices, smiles_values, train_indices),
        labels[train_indices].astype(int),
        source_weights,
        random_state=random_state,
    )
    probability = _predict_model(
        fitted,
        _feature_slice(candidate.feature_set, matrices, smiles_values, test_indices),
    )
    return fitted, probability


def _plot_validation(
    predictions: pd.DataFrame,
    calibration: pd.DataFrame,
    output_dir: Path,
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True, sharey=True)
    for axis, regime in zip(axes, REGIMES, strict=True):
        group = predictions[(predictions["regime"] == regime) & (predictions["strategy"] == "selected")]
        axis.scatter(
            group["private_similarity"],
            group["probability"],
            c=group["observed_label"],
            cmap="coolwarm",
            edgecolor="black",
            linewidth=0.4,
            alpha=0.8,
        )
        axis.set_title(regime.replace("_", " "))
        axis.set_xlabel("Nearest private-training Tanimoto")
    axes[0].set_ylabel("Calibrated blocker probability")
    fig.suptitle("Nested scaffold predictions and private-chemistry similarity")
    fig.tight_layout()
    path = figure_dir / "nested_probability_vs_similarity.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path))

    fig, ax = plt.subplots(figsize=(7, 6))
    selected = calibration[(calibration["strategy"] == "selected") & (calibration["n"] > 0)]
    sns.lineplot(
        data=selected,
        x="mean_probability",
        y="observed_fraction",
        hue="regime",
        marker="o",
        ax=ax,
    )
    ax.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Nested scaffold calibration")
    fig.tight_layout()
    path = figure_dir / "nested_calibration.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path))
    return paths


def _summary_markdown(
    path: Path,
    *,
    metrics: pd.DataFrame,
    intervals: pd.DataFrame,
    selections: pd.DataFrame,
    private_audit: Mapping[str, Any],
    outer_metadata: Sequence[Mapping[str, Any]],
) -> None:
    selected = metrics[metrics["strategy"] == "selected"].copy()
    lines = [
        "# Nested scaffold hERG validation",
        "",
        "This report uses outer confidential-scaffold folds. Every model, feature, parameter, calibration, threshold, and ensemble decision was made using inner scaffold folds only.",
        "",
        "## Dataset",
        "",
        f"- Decisively labeled confidential structures: {private_audit['n_labeled_unique_structures']} "
        f"({private_audit['n_blockers']} blockers; {private_audit['n_nonblockers']} nonblockers).",
        f"- Unique confidential scaffolds in the labeled task: {outer_metadata[0]['n_scaffolds']}.",
        "",
        "## Outer-fold performance",
        "",
        "Raw model scores are used for discrimination; calibrated probabilities are used for Brier/ECE and inner-selected classification thresholds.",
        "",
        "| Regime | Strategy | Raw ROC AUC | Calibrated ROC AUC | Balanced accuracy | MCC | Sensitivity | Specificity | Brier | ECE |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in metrics.sort_values(["regime", "strategy"]).iterrows():
        lines.append(
            f"| {row['regime']} | {row['strategy']} | {row['raw_roc_auc']:.3f} | "
            f"{row['calibrated_roc_auc']:.3f} | {row['balanced_accuracy']:.3f} | "
            f"{row['mcc']:.3f} | {row['sensitivity']:.3f} | "
            f"{row['specificity']:.3f} | {row['brier']:.3f} | {row['ece_8bin']:.3f} |"
        )
    lines.extend(["", "## Selected-model 95% scaffold-bootstrap intervals", ""])
    for _, metric_row in selected.iterrows():
        regime = metric_row["regime"]
        interval = intervals[
            (intervals["regime"] == regime)
            & (intervals["strategy"] == "selected")
            & (intervals["metric"] == "roc_auc")
        ].iloc[0]
        balanced = intervals[
            (intervals["regime"] == regime)
            & (intervals["strategy"] == "selected")
            & (intervals["metric"] == "balanced_accuracy")
        ].iloc[0]
        lines.append(
            f"- {regime}: ROC AUC {interval['point_estimate']:.3f} "
            f"[{interval['lower_95']:.3f}, {interval['upper_95']:.3f}]; "
            f"balanced accuracy {balanced['point_estimate']:.3f} "
            f"[{balanced['lower_95']:.3f}, {balanced['upper_95']:.3f}]."
        )
    lines.extend(["", "## Fold selections", ""])
    selection_counts = (
        selections.groupby(["regime", "selected_family", "selected_feature_set"])
        .size()
        .reset_index(name="folds")
        .sort_values(["regime", "folds"], ascending=[True, False])
    )
    for regime, group in selection_counts.groupby("regime"):
        choices = "; ".join(
            f"{row.selected_family}/{row.selected_feature_set} ({int(row.folds)} folds)"
            for row in group.itertuples()
        )
        lines.append(f"- {regime}: {choices}.")
    lines.extend(
        [
            "",
            "These estimates remain development-stage because the confidential sample is small and no later locked prospective cohort exists. The intervals quantify scaffold sampling uncertainty but cannot replace independent experimental validation.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_nested_validation(
    *,
    workbook_path: Path,
    public_path: Path,
    benchmark_config_path: Path,
    validation_config_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    started = time.time()
    benchmark_config = yaml.safe_load(benchmark_config_path.read_text(encoding="utf-8"))
    config = yaml.safe_load(validation_config_path.read_text(encoding="utf-8"))
    seed = int(config.get("random_state", 29))
    profile = str(config.get("candidate_profile", "quick"))
    private_weight = float(config.get("confidential_priority_weight", 5.0))
    output_dir.mkdir(parents=True, exist_ok=True)

    private_rows, private_unique, private_audit = load_private_workbook(
        workbook_path,
        blocker_max_um=float(config.get("blocker_max_um", 10.0)),
        nonblocker_min_um=float(config.get("nonblocker_min_um", 30.0)),
    )
    public, public_audit = load_public_herg(public_path)
    private_keys = set(private_unique["standard_inchi_key"])
    overlap = int(public["standard_inchi_key"].isin(private_keys).sum())
    public = public[~public["standard_inchi_key"].isin(private_keys)].reset_index(drop=True)
    private_unique = private_unique.reset_index(drop=True)

    private_labeled_local = np.flatnonzero(private_unique["herg_blocker_label"].notna().to_numpy())
    private_labeled = private_unique.iloc[private_labeled_local].reset_index(drop=True)
    universe = pd.concat(
        [
            public[["standard_inchi_key", "smiles", "scaffold", "herg_blocker_label", "source"]],
            private_unique[["standard_inchi_key", "smiles", "scaffold", "herg_blocker_label", "source"]],
        ],
        ignore_index=True,
    )
    n_public = len(public)
    public_indices = np.arange(n_public, dtype=int)
    private_indices = n_public + np.arange(len(private_unique), dtype=int)
    private_labeled_indices = private_indices[private_labeled_local]
    smiles_values = universe["smiles"].to_numpy(dtype=object)
    labels = pd.to_numeric(universe["herg_blocker_label"], errors="coerce").to_numpy(dtype=float)

    print(f"[nested:features] calculating features for {len(universe):,} structures", flush=True)
    matrices, _, feature_metadata = calculate_feature_registry(smiles_values.tolist())
    specs = build_model_specs(benchmark_config, profile)
    candidates = _candidate_rows(specs)
    candidate_table = pd.DataFrame(
        [
            {
                "candidate_key": candidate.key,
                "model_key": candidate.spec.key,
                "family": candidate.spec.family,
                "complexity": candidate.spec.complexity,
                "feature_set": candidate.feature_set,
                "parameters_json": json.dumps(candidate.spec.parameters, sort_keys=True),
            }
            for candidate in candidates
        ]
    )
    candidate_table.to_csv(output_dir / "nested_candidate_registry.csv", index=False)

    outer_repeats = int(config.get("outer_repeats", 1))
    outer_splits = int(config.get("outer_splits", 5))
    inner_splits = int(config.get("inner_splits", 3))
    keep_fraction = float(config.get("successive_keep_fraction", 0.55))
    minimum_candidates = int(config.get("successive_minimum_candidates", 12))
    ensemble_size = int(config.get("ensemble_size", 3))
    ad_quantile = float(config.get("applicability_quantile", 0.05))

    inner_prediction_rows: list[dict[str, Any]] = []
    inner_fit_rows: list[dict[str, Any]] = []
    halving_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    calibration_audit_rows: list[dict[str, Any]] = []
    outer_prediction_rows: list[dict[str, Any]] = []
    outer_metadata: list[dict[str, Any]] = []
    fitted_model_count = 0

    for repeat in range(outer_repeats):
        outer_folds, fold_metadata = _scaffold_folds(
            private_labeled,
            n_splits=outer_splits,
            random_state=seed + repeat * 1000,
        )
        fold_metadata = {"repeat": repeat, **fold_metadata}
        outer_metadata.append(fold_metadata)
        print(
            f"[nested] repeat {repeat + 1}/{outer_repeats}: {len(outer_folds)} outer scaffold folds; "
            f"{len(candidates)} starting candidates",
            flush=True,
        )
        for outer_fold, (outer_train_local, outer_test_local) in enumerate(outer_folds):
            outer_train_data = private_labeled.iloc[outer_train_local].reset_index(drop=True)
            inner_folds, inner_metadata = _scaffold_folds(
                outer_train_data,
                n_splits=inner_splits,
                random_state=seed + repeat * 1000 + outer_fold * 100,
            )
            outer_train_global = private_labeled_indices[outer_train_local]
            outer_test_global = private_labeled_indices[outer_test_local]
            private_similarity, public_similarity, ad_threshold = _applicability_values(
                matrices["morgan_1024_r2"],
                train_private=outer_train_global,
                test_private=outer_test_global,
                public_indices=public_indices,
                quantile=ad_quantile,
            )

            for regime_index, regime in enumerate(REGIMES):
                print(
                    f"[nested] repeat={repeat} outer={outer_fold + 1}/{len(outer_folds)} regime={regime}",
                    flush=True,
                )
                active = list(candidates)
                regime_inner_rows: list[dict[str, Any]] = []
                failed_candidates: set[str] = set()
                final_scores: dict[str, float] = {}
                for inner_stage, (inner_train_position, inner_test_position) in enumerate(inner_folds):
                    inner_train_local = outer_train_local[inner_train_position]
                    inner_test_local = outer_train_local[inner_test_position]
                    inner_train_global = private_labeled_indices[inner_train_local]
                    inner_test_global = private_labeled_indices[inner_test_local]
                    for candidate_index, candidate in enumerate(active):
                        train_indices, source_weights = _regime_training_indices(
                            regime,
                            public_indices=public_indices,
                            private_train_indices=inner_train_global,
                            private_weight=private_weight,
                        )
                        fit_started = time.time()
                        status = "ok"
                        error = ""
                        try:
                            _, probability = _model_prediction_for_indices(
                                candidate,
                                matrices=matrices,
                                smiles_values=smiles_values,
                                labels=labels,
                                train_indices=train_indices,
                                test_indices=inner_test_global,
                                source_weights=source_weights,
                                random_state=(
                                    seed
                                    + repeat * 100_000
                                    + outer_fold * 10_000
                                    + regime_index * 1000
                                    + inner_stage * 100
                                    + candidate_index
                                ),
                            )
                            for position, global_index, observed, score in zip(
                                inner_test_local,
                                inner_test_global,
                                labels[inner_test_global].astype(int),
                                probability,
                                strict=True,
                            ):
                                row = {
                                    "repeat": repeat,
                                    "outer_fold": outer_fold,
                                    "regime": regime,
                                    "inner_stage": inner_stage,
                                    "candidate_key": candidate.key,
                                    "family": candidate.spec.family,
                                    "complexity": candidate.spec.complexity,
                                    "feature_set": candidate.feature_set,
                                    "private_labeled_position": int(position),
                                    "private_global_index": int(global_index),
                                    "observed_label": int(observed),
                                    "raw_probability": float(score),
                                }
                                regime_inner_rows.append(row)
                                inner_prediction_rows.append(row)
                        except Exception as exc:
                            status = "failed"
                            error = f"{type(exc).__name__}: {exc}"
                            failed_candidates.add(candidate.key)
                        inner_fit_rows.append(
                            {
                                "repeat": repeat,
                                "outer_fold": outer_fold,
                                "regime": regime,
                                "inner_stage": inner_stage,
                                "candidate_key": candidate.key,
                                "family": candidate.spec.family,
                                "complexity": candidate.spec.complexity,
                                "feature_set": candidate.feature_set,
                                "n_train": int(len(train_indices)),
                                "n_test": int(len(inner_test_global)),
                                "status": status,
                                "error": error,
                                "runtime_seconds": time.time() - fit_started,
                            }
                        )
                        if (candidate_index + 1) % 10 == 0 or candidate_index + 1 == len(active):
                            print(
                                f"[nested:fits] outer={outer_fold + 1} {regime} inner={inner_stage + 1} "
                                f"completed={candidate_index + 1}/{len(active)}",
                                flush=True,
                            )
                    scores, summaries = _candidate_summary(regime_inner_rows, active)
                    for failed in failed_candidates:
                        scores[failed] = -np.inf
                    is_final_stage = inner_stage == len(inner_folds) - 1
                    retained = (
                        list(active)
                        if is_final_stage
                        else _retained_candidates(
                            active,
                            scores,
                            keep_fraction=keep_fraction,
                            minimum_candidates=minimum_candidates,
                        )
                    )
                    retained_keys = {candidate.key for candidate in retained}
                    for candidate in active:
                        summary = summaries.get(candidate.key, {})
                        halving_rows.append(
                            {
                                "repeat": repeat,
                                "outer_fold": outer_fold,
                                "regime": regime,
                                "inner_stage": inner_stage,
                                "candidate_key": candidate.key,
                                "family": candidate.spec.family,
                                "complexity": candidate.spec.complexity,
                                "feature_set": candidate.feature_set,
                                "retained": candidate.key in retained_keys,
                                "selection_score": scores.get(candidate.key, -np.inf),
                                **summary,
                            }
                        )
                    active = retained
                    final_scores = scores
                    pd.DataFrame(inner_fit_rows).to_csv(
                        output_dir / "inner_fit_audit.checkpoint.csv", index=False
                    )
                    pd.DataFrame(inner_prediction_rows).to_csv(
                        output_dir / "inner_predictions.checkpoint.csv", index=False
                    )
                    print(
                        f"[nested] outer={outer_fold + 1} {regime} inner={inner_stage + 1}/"
                        f"{len(inner_folds)} retained={len(active)}",
                        flush=True,
                    )

                ranked = sorted(
                    [
                        candidate
                        for candidate in active
                        if np.isfinite(final_scores.get(candidate.key, -np.inf))
                    ],
                    key=lambda candidate: final_scores[candidate.key],
                    reverse=True,
                )
                if not ranked:
                    raise RuntimeError(f"No nested candidates survived for {regime}, outer fold {outer_fold}")
                winner = ranked[0]
                diverse: list[Candidate] = []
                seen_families: set[str] = set()
                for candidate in ranked:
                    if candidate.spec.family == "dummy" or candidate.spec.family in seen_families:
                        continue
                    diverse.append(candidate)
                    seen_families.add(candidate.spec.family)
                    if len(diverse) >= ensemble_size:
                        break
                if winner.key not in {candidate.key for candidate in diverse}:
                    diverse.insert(0, winner)
                    diverse = diverse[:ensemble_size]

                inner_frame = pd.DataFrame(regime_inner_rows)
                inner_calibrated: dict[str, pd.DataFrame] = {}
                outer_raw: dict[str, np.ndarray] = {}
                outer_calibrated: dict[str, np.ndarray] = {}
                for ensemble_rank, candidate in enumerate(diverse, start=1):
                    candidate_inner = inner_frame[inner_frame["candidate_key"] == candidate.key].copy()
                    candidate_inner = candidate_inner.drop_duplicates("private_global_index")
                    calibrator = _fit_sigmoid_calibrator(
                        candidate_inner["observed_label"].to_numpy(dtype=int),
                        candidate_inner["raw_probability"].to_numpy(dtype=float),
                    )
                    candidate_inner["calibrated_probability"] = calibrator.transform(
                        candidate_inner["raw_probability"].to_numpy(dtype=float)
                    )
                    inner_calibrated[candidate.key] = candidate_inner
                    calibration_audit_rows.append(
                        {
                            "repeat": repeat,
                            "outer_fold": outer_fold,
                            "regime": regime,
                            "ensemble_rank": ensemble_rank,
                            "candidate_key": candidate.key,
                            "family": candidate.spec.family,
                            "feature_set": candidate.feature_set,
                            "n_inner_predictions": int(len(candidate_inner)),
                            "calibrator_slope": calibrator.slope,
                            "calibrator_intercept": calibrator.intercept,
                        }
                    )
                    train_indices, source_weights = _regime_training_indices(
                        regime,
                        public_indices=public_indices,
                        private_train_indices=outer_train_global,
                        private_weight=private_weight,
                    )
                    fitted, raw_outer = _model_prediction_for_indices(
                        candidate,
                        matrices=matrices,
                        smiles_values=smiles_values,
                        labels=labels,
                        train_indices=train_indices,
                        test_indices=outer_test_global,
                        source_weights=source_weights,
                        random_state=(
                            seed
                            + repeat * 100_000
                            + outer_fold * 10_000
                            + regime_index * 1000
                            + 900
                            + ensemble_rank
                        ),
                    )
                    fitted_model_count += 1
                    _save_model(
                        output_dir
                        / "fold_models"
                        / f"repeat{repeat}__outer{outer_fold}__{regime}__rank{ensemble_rank}__{candidate.spec.key}",
                        fitted,
                    )
                    joblib.dump(
                        calibrator,
                        output_dir
                        / "fold_models"
                        / f"repeat{repeat}__outer{outer_fold}__{regime}__rank{ensemble_rank}__calibrator.joblib",
                        compress=3,
                    )
                    outer_raw[candidate.key] = raw_outer
                    outer_calibrated[candidate.key] = calibrator.transform(raw_outer)

                winner_inner = inner_calibrated[winner.key]
                winner_threshold, winner_inner_balanced = _optimal_threshold(
                    winner_inner["observed_label"].to_numpy(dtype=int),
                    winner_inner["calibrated_probability"].to_numpy(dtype=float),
                )
                ensemble_inner = pd.concat(
                    [
                        frame.set_index("private_global_index")["calibrated_probability"].rename(
                            candidate.key
                        )
                        for candidate, frame in (
                            (candidate, inner_calibrated[candidate.key]) for candidate in diverse
                        )
                    ],
                    axis=1,
                    join="inner",
                )
                ensemble_inner_probability = ensemble_inner.mean(axis=1).to_numpy(dtype=float)
                ensemble_inner_y = labels[ensemble_inner.index.to_numpy(dtype=int)].astype(int)
                ensemble_threshold, ensemble_inner_balanced = _optimal_threshold(
                    ensemble_inner_y,
                    ensemble_inner_probability,
                )
                winner_inner_metrics = validation_metrics(
                    winner_inner["observed_label"].to_numpy(dtype=int),
                    winner_inner["calibrated_probability"].to_numpy(dtype=float),
                    predicted_label=(
                        winner_inner["calibrated_probability"].to_numpy(dtype=float) >= winner_threshold
                    ),
                )
                ensemble_inner_metrics = validation_metrics(
                    ensemble_inner_y,
                    ensemble_inner_probability,
                    predicted_label=ensemble_inner_probability >= ensemble_threshold,
                )
                chosen_strategy = (
                    "diverse_ensemble"
                    if _selection_score(ensemble_inner_metrics) > _selection_score(winner_inner_metrics)
                    else "selected"
                )
                selection_rows.append(
                    {
                        "repeat": repeat,
                        "outer_fold": outer_fold,
                        "regime": regime,
                        "selected_candidate_key": winner.key,
                        "selected_family": winner.spec.family,
                        "selected_complexity": winner.spec.complexity,
                        "selected_feature_set": winner.feature_set,
                        "selected_parameters_json": json.dumps(winner.spec.parameters, sort_keys=True),
                        "selected_inner_score": final_scores[winner.key],
                        "selected_inner_balanced_accuracy": winner_inner_balanced,
                        "selected_threshold": winner_threshold,
                        "ensemble_candidate_keys": " | ".join(candidate.key for candidate in diverse),
                        "ensemble_families": " | ".join(candidate.spec.family for candidate in diverse),
                        "ensemble_inner_balanced_accuracy": ensemble_inner_balanced,
                        "ensemble_threshold": ensemble_threshold,
                        "inner_chosen_strategy": chosen_strategy,
                        "ad_threshold": ad_threshold,
                        "inner_fold_metadata_json": json.dumps(inner_metadata, sort_keys=True),
                    }
                )

                strategy_probabilities = {
                    "selected": outer_calibrated[winner.key],
                    "diverse_ensemble": np.mean(
                        np.vstack([outer_calibrated[candidate.key] for candidate in diverse]), axis=0
                    ),
                }
                strategy_raw_probabilities = {
                    "selected": outer_raw[winner.key],
                    "diverse_ensemble": np.mean(
                        np.vstack([outer_raw[candidate.key] for candidate in diverse]), axis=0
                    ),
                }
                strategy_thresholds = {
                    "selected": winner_threshold,
                    "diverse_ensemble": ensemble_threshold,
                }
                for strategy, probability in strategy_probabilities.items():
                    raw_probability = strategy_raw_probabilities[strategy]
                    threshold = strategy_thresholds[strategy]
                    for (
                        test_position,
                        global_index,
                        observed,
                        raw_score,
                        score,
                        private_sim,
                        public_sim,
                    ) in zip(
                        outer_test_local,
                        outer_test_global,
                        labels[outer_test_global].astype(int),
                        raw_probability,
                        probability,
                        private_similarity,
                        public_similarity,
                        strict=True,
                    ):
                        outer_prediction_rows.append(
                            {
                                "repeat": repeat,
                                "outer_fold": outer_fold,
                                "regime": regime,
                                "strategy": strategy,
                                "private_labeled_position": int(test_position),
                                "private_global_index": int(global_index),
                                "standard_inchi_key": universe.iloc[global_index]["standard_inchi_key"],
                                "compound_id": private_labeled.iloc[test_position]["compound_id"],
                                "scaffold": private_labeled.iloc[test_position]["scaffold"],
                                "observed_label": int(observed),
                                "raw_probability": float(raw_score),
                                "probability": float(score),
                                "threshold": float(threshold),
                                "predicted_label": int(score >= threshold),
                                "private_similarity": float(private_sim),
                                "public_similarity": float(public_sim),
                                "ad_threshold": float(ad_threshold),
                                "in_private_domain": bool(private_sim >= ad_threshold),
                                "inner_chosen_strategy": chosen_strategy,
                            }
                        )

            pd.DataFrame(inner_fit_rows).to_csv(output_dir / "inner_fit_audit.checkpoint.csv", index=False)
            pd.DataFrame(outer_prediction_rows).to_csv(
                output_dir / "outer_predictions.checkpoint.csv", index=False
            )

    inner_predictions = pd.DataFrame(inner_prediction_rows)
    inner_fits = pd.DataFrame(inner_fit_rows)
    halving = pd.DataFrame(halving_rows)
    selections = pd.DataFrame(selection_rows)
    calibration_audit = pd.DataFrame(calibration_audit_rows)
    outer_predictions = pd.DataFrame(outer_prediction_rows)
    inner_predictions.to_csv(output_dir / "inner_predictions.csv", index=False)
    inner_fits.to_csv(output_dir / "inner_fit_audit.csv", index=False)
    halving.to_csv(output_dir / "successive_halving_results.csv", index=False)
    selections.to_csv(output_dir / "outer_fold_selections.csv", index=False)
    calibration_audit.to_csv(output_dir / "calibration_audit.csv", index=False)
    outer_predictions.to_csv(output_dir / "outer_predictions.csv", index=False)
    (output_dir / "inner_fit_audit.checkpoint.csv").unlink(missing_ok=True)
    (output_dir / "inner_predictions.checkpoint.csv").unlink(missing_ok=True)
    (output_dir / "outer_predictions.checkpoint.csv").unlink(missing_ok=True)

    metric_rows: list[dict[str, Any]] = []
    outer_fold_metric_rows: list[dict[str, Any]] = []
    ad_rows: list[dict[str, Any]] = []
    for (regime, strategy), group in outer_predictions.groupby(["regime", "strategy"]):
        metrics = _prediction_frame_metrics(group)
        metric_rows.append({"regime": regime, "strategy": strategy, **metrics})
        for (repeat, outer_fold), fold_group in group.groupby(["repeat", "outer_fold"]):
            outer_fold_metric_rows.append(
                {
                    "repeat": int(repeat),
                    "outer_fold": int(outer_fold),
                    "regime": regime,
                    "strategy": strategy,
                    **_prediction_frame_metrics(fold_group),
                }
            )
        for domain_status, domain_group in group.groupby("in_private_domain"):
            base = {
                "regime": regime,
                "strategy": strategy,
                "in_private_domain": bool(domain_status),
                "n": int(len(domain_group)),
                "n_blockers": int((domain_group["observed_label"] == 1).sum()),
                "n_nonblockers": int((domain_group["observed_label"] == 0).sum()),
                "mean_private_similarity": float(domain_group["private_similarity"].mean()),
                "mean_public_similarity": float(domain_group["public_similarity"].mean()),
            }
            if domain_group["observed_label"].nunique() == 2:
                domain_metrics = _prediction_frame_metrics(domain_group)
            else:
                domain_metrics = {
                    key: np.nan
                    for key in validation_metrics(np.array([0, 1]), np.array([0.25, 0.75]))
                    if key != "n"
                }
            ad_rows.append({**base, **domain_metrics})
    metrics = pd.DataFrame(metric_rows)
    outer_fold_metrics = pd.DataFrame(outer_fold_metric_rows)
    ad_performance = pd.DataFrame(ad_rows)
    intervals = _bootstrap_intervals(
        outer_predictions,
        iterations=int(config.get("bootstrap_iterations", 2000)),
        random_state=seed + 50_000,
    )
    calibration = _calibration_curve_rows(outer_predictions, n_bins=int(config.get("calibration_bins", 8)))
    metrics.to_csv(output_dir / "nested_metrics.csv", index=False)
    outer_fold_metrics.to_csv(output_dir / "outer_fold_metrics.csv", index=False)
    ad_performance.to_csv(output_dir / "applicability_performance.csv", index=False)
    intervals.to_csv(output_dir / "bootstrap_intervals.csv", index=False)
    calibration.to_csv(output_dir / "calibration_curves.csv", index=False)
    figures = _plot_validation(outer_predictions, calibration, output_dir)
    _summary_markdown(
        output_dir / "validation_summary.md",
        metrics=metrics,
        intervals=intervals,
        selections=selections,
        private_audit=private_audit,
        outer_metadata=outer_metadata,
    )

    manifest = {
        "status": "complete",
        "runtime_seconds": time.time() - started,
        "workbook_sha256": _sha256_file(workbook_path),
        "public_data_sha256": _sha256_file(public_path),
        "benchmark_config_sha256": _sha256_file(benchmark_config_path),
        "validation_config_sha256": _sha256_file(validation_config_path),
        "candidate_profile": profile,
        "n_candidates": len(candidates),
        "n_inner_fits": int(len(inner_fits)),
        "n_successful_inner_fits": int((inner_fits["status"] == "ok").sum()),
        "n_failed_inner_fits": int((inner_fits["status"] != "ok").sum()),
        "n_fitted_outer_models": int(fitted_model_count),
        "n_outer_predictions": int(len(outer_predictions)),
        "private_audit": private_audit,
        "public_audit": public_audit,
        "private_public_exact_overlap_removed": overlap,
        "outer_fold_metadata": outer_metadata,
        "feature_registry": feature_metadata,
        "config": config,
        "figures": figures,
        "output_files": sorted(path.name for path in output_dir.iterdir() if path.is_file()),
    }
    _write_json(output_dir / "validation_manifest.json", manifest)
    print(f"[nested:complete] results written to {output_dir.resolve()}", flush=True)
    return manifest
