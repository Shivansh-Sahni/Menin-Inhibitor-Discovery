"""Task-aware metrics with deterministic, hand-checkable behavior.

The functions return plain dictionaries/data frames so reports can distinguish
point estimates, resampling uncertainty, subgroup diagnostics, and smoke-only
pipeline checks.  Undefined statistics are represented as ``None`` rather
than silently replaced by zero.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

METRIC_SCHEMA_VERSION = "1.0.0"


def metric_reporting_registry() -> pd.DataFrame:
    """Predeclared primary/secondary/diagnostic roles by task family."""

    rows = [
        ("binary_classification", "average_precision_pr_auc", "primary", "discrimination under imbalance"),
        ("binary_classification", "brier_score", "secondary", "probabilistic accuracy and calibration"),
        ("binary_classification", "balanced_accuracy", "secondary", "thresholded class balance"),
        ("binary_classification", "roc_auc", "secondary", "threshold-free ranking; can obscure imbalance"),
        ("binary_classification", "ece_uniform", "diagnostic", "bin-dependent calibration screen"),
        ("binary_classification", "matthews_correlation_coefficient", "diagnostic", "thresholded summary"),
        ("binding_log_regression", "mae", "primary", "absolute error on declared log-affinity scale"),
        ("binding_log_regression", "spearman_r", "secondary", "within-set ranking"),
        ("binding_log_regression", "rmse", "secondary", "penalizes large errors"),
        ("binding_log_regression", "r2", "diagnostic", "set-distribution-dependent explained variance"),
        ("pk_log_regression", "mae", "primary", "absolute error on declared log-PK scale"),
        ("pk_raw_positive_regression", "median_fold_error", "primary", "multiplicative PK error"),
        ("pk_regression", "prediction_interval_coverage", "primary", "uncertainty validity"),
        (
            "censored_regression",
            "mean_interval_violation",
            "primary",
            "respects bounds without midpoint imputation",
        ),
        (
            "censored_regression",
            "fraction_predictions_consistent_with_bounds",
            "secondary",
            "bound consistency",
        ),
        ("virtual_screening", "enrichment_factor_0p01", "primary", "early retrieval at a fixed fraction"),
    ]
    return pd.DataFrame(rows, columns=["task_family", "metric", "role", "rationale"]).assign(
        registry_version=METRIC_SCHEMA_VERSION,
        reporting_rule="task_must_predeclare_one_registry_family_before_locked_test_evaluation",
    )


def _float_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _paired_arrays(y_true: Iterable[object], y_pred: Iterable[object]) -> tuple[np.ndarray, np.ndarray]:
    true = np.asarray(list(y_true), dtype=float)
    pred = np.asarray(list(y_pred), dtype=float)
    if true.ndim != 1 or pred.ndim != 1:
        raise ValueError("Inputs must be one-dimensional")
    if len(true) != len(pred):
        raise ValueError("Inputs must have equal length")
    if not len(true):
        raise ValueError("At least one observation is required")
    if not np.all(np.isfinite(true)) or not np.all(np.isfinite(pred)):
        raise ValueError("Metrics require finite inputs; filter with an explicit exclusion ledger")
    return true, pred


def regression_metrics(y_true: Iterable[object], y_pred: Iterable[object]) -> dict[str, Any]:
    """Regression metrics on the modeled scale with explicit calibration terms."""

    true, pred = _paired_arrays(y_true, y_pred)
    residual = pred - true
    absolute = np.abs(residual)
    squared = residual**2
    sst = float(np.sum((true - np.mean(true)) ** 2))
    r2 = 1.0 - float(np.sum(squared)) / sst if sst > 0 else None
    pearson = (
        float(stats.pearsonr(true, pred).statistic)
        if len(true) >= 2 and np.std(true) > 1e-12 and np.std(pred) > 1e-12
        else None
    )
    spearman = (
        float(stats.spearmanr(true, pred).statistic)
        if len(true) >= 2 and np.ptp(true) > 1e-12 and np.ptp(pred) > 1e-12
        else None
    )
    pred_variance = float(np.var(pred))
    calibration_slope: float | None
    calibration_intercept: float | None
    if len(true) >= 2 and pred_variance > 0:
        covariance = float(np.mean((pred - np.mean(pred)) * (true - np.mean(true))))
        calibration_slope = covariance / pred_variance
        calibration_intercept = float(np.mean(true) - calibration_slope * np.mean(pred))
    else:
        calibration_slope = calibration_intercept = None
    covariance = float(np.mean((true - np.mean(true)) * (pred - np.mean(pred))))
    ccc_denominator = float(np.var(true) + np.var(pred) + (np.mean(true) - np.mean(pred)) ** 2)
    ccc = 2.0 * covariance / ccc_denominator if ccc_denominator > 0 else None
    return {
        "metric_schema_version": METRIC_SCHEMA_VERSION,
        "n": int(len(true)),
        "mae": float(np.mean(absolute)),
        "median_absolute_error": float(np.median(absolute)),
        "mse": float(np.mean(squared)),
        "rmse": float(np.sqrt(np.mean(squared))),
        "r2": r2,
        "pearson_r": pearson,
        "spearman_r": spearman,
        "concordance_correlation_coefficient": _float_or_none(ccc),
        "mean_signed_error_prediction_minus_observation": float(np.mean(residual)),
        "calibration_slope_observation_on_prediction": _float_or_none(calibration_slope),
        "calibration_intercept_observation_on_prediction": _float_or_none(calibration_intercept),
    }


def fold_error_metrics(observed: Iterable[object], predicted: Iterable[object]) -> dict[str, Any]:
    """Multiplicative error for strictly positive, untransformed quantities."""

    true, pred = _paired_arrays(observed, predicted)
    if np.any(true <= 0) or np.any(pred <= 0):
        raise ValueError("Fold error requires strictly positive observed and predicted values")
    ratio = np.maximum(pred / true, true / pred)
    return {
        "n": int(len(true)),
        "median_fold_error": float(np.median(ratio)),
        "geometric_mean_fold_error": float(np.exp(np.mean(np.log(ratio)))),
        "fraction_within_2_fold": float(np.mean(ratio <= 2.0)),
        "fraction_within_3_fold": float(np.mean(ratio <= 3.0)),
        "fraction_within_10_fold": float(np.mean(ratio <= 10.0)),
    }


def calibration_bins(
    y_true: Iterable[object],
    probability: Iterable[object],
    *,
    n_bins: int = 10,
    strategy: Literal["uniform", "quantile"] = "uniform",
) -> pd.DataFrame:
    """Return non-empty calibration bins and their weighted contributions."""

    true, prob = _paired_arrays(y_true, probability)
    if not np.all(np.isin(true, [0.0, 1.0])):
        raise ValueError("Binary calibration labels must be 0 or 1")
    if np.any((prob < 0) | (prob > 1)):
        raise ValueError("Probabilities must lie in [0, 1]")
    if n_bins < 2:
        raise ValueError("n_bins must be at least two")
    if strategy == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    elif strategy == "quantile":
        edges = np.unique(np.quantile(prob, np.linspace(0.0, 1.0, n_bins + 1)))
        if len(edges) < 2:
            edges = np.asarray([0.0, 1.0])
        edges[0], edges[-1] = 0.0, 1.0
    else:
        raise ValueError("strategy must be uniform or quantile")
    bin_ids = np.clip(np.searchsorted(edges[1:-1], prob, side="right"), 0, len(edges) - 2)
    rows: list[dict[str, Any]] = []
    for bin_id in range(len(edges) - 1):
        mask = bin_ids == bin_id
        if not np.any(mask):
            continue
        confidence = float(np.mean(prob[mask]))
        observed = float(np.mean(true[mask]))
        rows.append(
            {
                "bin_id": int(bin_id),
                "lower": float(edges[bin_id]),
                "upper": float(edges[bin_id + 1]),
                "n": int(np.sum(mask)),
                "fraction": float(np.mean(mask)),
                "mean_probability": confidence,
                "observed_fraction": observed,
                "absolute_gap": abs(confidence - observed),
                "weighted_absolute_gap": float(np.mean(mask)) * abs(confidence - observed),
            }
        )
    return pd.DataFrame(rows)


def _operating_point_at_specificity(
    true: np.ndarray, probability: np.ndarray, minimum_specificity: float
) -> tuple[float | None, float | None, float | None]:
    false_positive_rate, sensitivity, thresholds = roc_curve(true, probability)
    specificity = 1.0 - false_positive_rate
    eligible = np.flatnonzero(specificity >= minimum_specificity)
    if not len(eligible):
        return None, None, None
    index = int(eligible[np.argmax(sensitivity[eligible])])
    return float(sensitivity[index]), float(specificity[index]), float(thresholds[index])


def _operating_point_at_sensitivity(
    true: np.ndarray, probability: np.ndarray, minimum_sensitivity: float
) -> tuple[float | None, float | None, float | None]:
    false_positive_rate, sensitivity, thresholds = roc_curve(true, probability)
    specificity = 1.0 - false_positive_rate
    eligible = np.flatnonzero(sensitivity >= minimum_sensitivity)
    if not len(eligible):
        return None, None, None
    index = int(eligible[np.argmax(specificity[eligible])])
    return float(sensitivity[index]), float(specificity[index]), float(thresholds[index])


def binary_classification_metrics(
    y_true: Iterable[object],
    probability: Iterable[object],
    *,
    threshold: float = 0.5,
    calibration_bins_count: int = 10,
    fixed_specificity: float = 0.90,
    fixed_sensitivity: float = 0.90,
) -> dict[str, Any]:
    """Discrimination, threshold, and calibration metrics for a binary task."""

    true_float, prob = _paired_arrays(y_true, probability)
    if not np.all(np.isin(true_float, [0.0, 1.0])):
        raise ValueError("Binary labels must be 0 or 1")
    if np.any((prob < 0) | (prob > 1)):
        raise ValueError("Probabilities must lie in [0, 1]")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must lie in [0, 1]")
    true = true_float.astype(int)
    prediction = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(true, prediction, labels=[0, 1]).ravel()
    both_classes = np.unique(true).size == 2
    bins_uniform = calibration_bins(true, prob, n_bins=calibration_bins_count, strategy="uniform")
    bins_quantile = calibration_bins(true, prob, n_bins=calibration_bins_count, strategy="quantile")
    sensitivity_at_spec, achieved_spec, threshold_at_spec = (
        _operating_point_at_specificity(true, prob, fixed_specificity) if both_classes else (None, None, None)
    )
    achieved_sens, specificity_at_sens, threshold_at_sens = (
        _operating_point_at_sensitivity(true, prob, fixed_sensitivity) if both_classes else (None, None, None)
    )
    return {
        "metric_schema_version": METRIC_SCHEMA_VERSION,
        "n": int(len(true)),
        "n_events": int(np.sum(true == 1)),
        "n_non_events": int(np.sum(true == 0)),
        "estimability_status": "estimable" if both_classes else "one_class_probability_metrics_only",
        "prevalence": float(np.mean(true)),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(true, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(true, prediction)) if both_classes else None,
        "precision": float(precision_score(true, prediction, zero_division=0)),
        "recall_sensitivity": float(recall_score(true, prediction, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else None,
        "negative_predictive_value": float(tn / (tn + fn)) if tn + fn else None,
        "f1": float(f1_score(true, prediction, zero_division=0)),
        "matthews_correlation_coefficient": (
            _float_or_none(matthews_corrcoef(true, prediction)) if both_classes else None
        ),
        "cohen_kappa": _float_or_none(cohen_kappa_score(true, prediction)) if both_classes else None,
        "roc_auc": float(roc_auc_score(true, prob)) if both_classes else None,
        "average_precision_pr_auc": float(average_precision_score(true, prob)) if both_classes else None,
        "brier_score": float(brier_score_loss(true, prob)),
        "log_loss": float(log_loss(true, np.clip(prob, 1e-15, 1 - 1e-15), labels=[0, 1])),
        "ece_uniform": float(bins_uniform["weighted_absolute_gap"].sum()),
        "ece_quantile": float(bins_quantile["weighted_absolute_gap"].sum()),
        "maximum_calibration_error_uniform": float(bins_uniform["absolute_gap"].max()),
        f"sensitivity_at_specificity_{fixed_specificity:.2f}": sensitivity_at_spec,
        f"achieved_specificity_for_{fixed_specificity:.2f}": achieved_spec,
        f"threshold_at_specificity_{fixed_specificity:.2f}": threshold_at_spec,
        f"specificity_at_sensitivity_{fixed_sensitivity:.2f}": specificity_at_sens,
        f"achieved_sensitivity_for_{fixed_sensitivity:.2f}": achieved_sens,
        f"threshold_at_sensitivity_{fixed_sensitivity:.2f}": threshold_at_sens,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def multiclass_classification_metrics(
    y_true: Sequence[object],
    probabilities: np.ndarray,
    *,
    class_labels: Sequence[object],
) -> dict[str, Any]:
    """Macro/micro/weighted metrics for explicitly ordered multiclass outputs."""

    true = np.asarray(list(y_true), dtype=object)
    probability = np.asarray(probabilities, dtype=float)
    labels = np.asarray(list(class_labels), dtype=object)
    if not len(true) or not len(labels):
        raise ValueError("Multiclass metrics require non-empty observations and class labels")
    if len(set(labels.tolist())) != len(labels):
        raise ValueError("class_labels must be unique")
    if probability.shape != (len(true), len(labels)):
        raise ValueError("Probability matrix shape must be (n_records, n_class_labels)")
    if not np.all(np.isfinite(probability)) or np.any(probability < 0):
        raise ValueError("Probabilities must be finite and non-negative")
    if not np.allclose(probability.sum(axis=1), 1.0, atol=1e-7):
        raise ValueError("Each multiclass probability row must sum to one")
    unknown = set(true) - set(labels)
    if unknown:
        raise ValueError(f"Observed labels are absent from class_labels: {sorted(map(str, unknown))}")
    prediction = labels[np.argmax(probability, axis=1)]
    result: dict[str, Any] = {
        "metric_schema_version": METRIC_SCHEMA_VERSION,
        "n": int(len(true)),
        "n_classes": int(len(labels)),
        "accuracy": float(accuracy_score(true, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(true, prediction)),
        "cohen_kappa": _float_or_none(cohen_kappa_score(true, prediction)),
        "log_loss": float(log_loss(true, probability, labels=list(labels))),
        "confusion_matrix": confusion_matrix(true, prediction, labels=labels).astype(int).tolist(),
        "class_labels": [str(item) for item in labels],
    }
    for average in ("macro", "micro", "weighted"):
        precision, recall, f1, _ = precision_recall_fscore_support(
            true, prediction, labels=labels, average=average, zero_division=0
        )
        result[f"precision_{average}"] = float(precision)
        result[f"recall_{average}"] = float(recall)
        result[f"f1_{average}"] = float(f1)
    try:
        result["roc_auc_ovr_macro"] = float(
            roc_auc_score(true, probability, labels=labels, multi_class="ovr", average="macro")
        )
    except ValueError:
        result["roc_auc_ovr_macro"] = None
    return result


def censored_regression_metrics(
    lower_bound: Iterable[object],
    upper_bound: Iterable[object],
    prediction: Iterable[object],
) -> dict[str, Any]:
    """Interval-consistency metrics without midpoint imputation.

    Use ``-inf``/``inf`` for one-sided censoring.  Exact labels have equal
    finite lower and upper bounds.
    """

    lower = np.asarray(list(lower_bound), dtype=float)
    upper = np.asarray(list(upper_bound), dtype=float)
    pred = np.asarray(list(prediction), dtype=float)
    if lower.shape != upper.shape or lower.shape != pred.shape or lower.ndim != 1:
        raise ValueError("Bounds and predictions must be equal-length one-dimensional arrays")
    if not len(pred):
        raise ValueError("Censored regression metrics require at least one observation")
    if not np.all(np.isfinite(pred)):
        raise ValueError("Predictions must be finite")
    if np.any(np.isnan(lower)) or np.any(np.isnan(upper)):
        raise ValueError("Use explicit infinite bounds rather than missing censoring bounds")
    if np.any(lower > upper):
        raise ValueError("Every lower bound must be <= its upper bound")
    below = np.maximum(lower - pred, 0.0)
    above = np.maximum(pred - upper, 0.0)
    violation = below + above
    exact = np.isfinite(lower) & np.isfinite(upper) & (lower == upper)
    return {
        "n": int(len(pred)),
        "n_exact": int(np.sum(exact)),
        "n_interval_or_censored": int(np.sum(~exact)),
        "fraction_predictions_consistent_with_bounds": float(np.mean(violation == 0)),
        "mean_interval_violation": float(np.mean(violation)),
        "median_interval_violation": float(np.median(violation)),
        "exact_subset_mae": float(np.mean(np.abs(pred[exact] - lower[exact]))) if np.any(exact) else None,
    }


def prediction_interval_metrics(
    y_true: Iterable[object],
    lower: Iterable[object],
    upper: Iterable[object],
    *,
    nominal_coverage: float,
) -> dict[str, Any]:
    """Coverage, width, and interval score for prediction intervals."""

    true = np.asarray(list(y_true), dtype=float)
    lo = np.asarray(list(lower), dtype=float)
    hi = np.asarray(list(upper), dtype=float)
    if true.shape != lo.shape or true.shape != hi.shape or true.ndim != 1:
        raise ValueError("Inputs must be equal-length one-dimensional arrays")
    if not len(true):
        raise ValueError("Prediction interval metrics require at least one observation")
    if not np.all(np.isfinite(true)) or not np.all(np.isfinite(lo)) or not np.all(np.isfinite(hi)):
        raise ValueError("Prediction interval inputs must be finite")
    if np.any(lo > hi):
        raise ValueError("Prediction interval lower bound exceeds upper bound")
    if not 0 < nominal_coverage < 1:
        raise ValueError("nominal_coverage must lie in (0, 1)")
    alpha = 1.0 - nominal_coverage
    width = hi - lo
    interval_score = width + 2 / alpha * (lo - true) * (true < lo) + 2 / alpha * (true - hi) * (true > hi)
    covered = (true >= lo) & (true <= hi)
    return {
        "n": int(len(true)),
        "nominal_coverage": nominal_coverage,
        "empirical_coverage": float(np.mean(covered)),
        "coverage_error": float(np.mean(covered) - nominal_coverage),
        "mean_interval_width": float(np.mean(width)),
        "median_interval_width": float(np.median(width)),
        "mean_interval_score": float(np.mean(interval_score)),
    }


def enrichment_metrics(
    y_true: Iterable[object],
    score: Iterable[object],
    *,
    fractions: Sequence[float] = (0.01, 0.05, 0.10),
) -> dict[str, Any]:
    """Top-fraction enrichment for binary hit retrieval (higher score is better)."""

    true_float, values = _paired_arrays(y_true, score)
    if not np.all(np.isin(true_float, [0.0, 1.0])):
        raise ValueError("Enrichment labels must be binary")
    true = true_float.astype(int)
    prevalence = float(np.mean(true))
    order = np.lexsort((np.arange(len(values)), -values))
    output: dict[str, Any] = {"n": len(true), "prevalence": prevalence}
    for fraction in fractions:
        if not 0 < fraction <= 1:
            raise ValueError("Enrichment fractions must lie in (0, 1]")
        top_n = max(1, int(math.ceil(fraction * len(true))))
        top_rate = float(np.mean(true[order[:top_n]]))
        suffix = str(fraction).replace(".", "p")
        output[f"top_{suffix}_n"] = top_n
        output[f"top_{suffix}_hit_rate"] = top_rate
        output[f"enrichment_factor_{suffix}"] = top_rate / prevalence if prevalence > 0 else None
    return output


MetricFunction = Callable[[np.ndarray, np.ndarray], Mapping[str, Any]]


def bootstrap_metric_intervals(
    y_true: Iterable[object],
    prediction: Iterable[object],
    metric_function: MetricFunction,
    *,
    groups: Iterable[object] | None = None,
    iterations: int = 1000,
    confidence: float = 0.95,
    seed: int = 20260804,
) -> dict[str, Any]:
    """Paired row or cluster bootstrap intervals with successful-draw counts."""

    true, pred = _paired_arrays(y_true, prediction)
    if iterations < 1:
        raise ValueError("iterations must be positive")
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie in (0, 1)")
    if groups is None:
        group_values = np.arange(len(true), dtype=object)
        resampling_unit = "row"
    else:
        group_values = np.asarray(list(groups), dtype=object)
        if group_values.shape != true.shape:
            raise ValueError("groups must match y_true length")
        if any(pd.isna(value) or str(value).strip() == "" for value in group_values):
            raise ValueError("Bootstrap groups may not be missing")
        resampling_unit = "group"
    unique_groups = np.unique(group_values)
    rng = np.random.default_rng(seed)
    point = dict(metric_function(true, pred))
    numeric_names = [
        name for name, value in point.items() if _float_or_none(value) is not None and name != "n"
    ]
    draws: dict[str, list[float]] = {name: [] for name in numeric_names}
    successful_draws = 0
    for _ in range(iterations):
        sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        indices = np.concatenate([np.flatnonzero(group_values == group) for group in sampled])
        try:
            result = metric_function(true[indices], pred[indices])
        except ValueError:
            continue
        successful_draws += 1
        for name in numeric_names:
            value = _float_or_none(result.get(name))
            if value is not None:
                draws[name].append(value)
    alpha = 1.0 - confidence
    intervals = {}
    for name, values in draws.items():
        intervals[name] = {
            "point_estimate": _float_or_none(point[name]),
            "lower": float(np.quantile(values, alpha / 2)) if values else None,
            "upper": float(np.quantile(values, 1 - alpha / 2)) if values else None,
            "n_finite_draws": int(len(values)),
        }
    return {
        "confidence": confidence,
        "requested_draws": iterations,
        "successful_draws": successful_draws,
        "seed": seed,
        "resampling_unit": resampling_unit,
        "n_resampling_units": int(len(unique_groups)),
        "intervals": intervals,
    }


def paired_bootstrap_difference(
    y_true: Iterable[object],
    prediction_a: Iterable[object],
    prediction_b: Iterable[object],
    *,
    metric: Literal["mae", "rmse", "brier_score"],
    groups: Iterable[object] | None = None,
    iterations: int = 1000,
    seed: int = 20260804,
) -> dict[str, Any]:
    """Paired bootstrap of A-minus-B error; negative favors model A."""

    true, pred_a = _paired_arrays(y_true, prediction_a)
    _, pred_b = _paired_arrays(y_true, prediction_b)
    if iterations < 1:
        raise ValueError("iterations must be positive")
    if metric == "brier_score":
        if not np.all(np.isin(true, [0.0, 1.0])):
            raise ValueError("Brier comparison requires binary observed labels")
        if np.any((pred_a < 0) | (pred_a > 1)) or np.any((pred_b < 0) | (pred_b > 1)):
            raise ValueError("Brier comparison requires probabilities in [0, 1]")
    if groups is None:
        group_values = np.arange(len(true), dtype=object)
        unit = "row"
    else:
        group_values = np.asarray(list(groups), dtype=object)
        if group_values.shape != true.shape:
            raise ValueError("groups must match y_true length")
        if any(pd.isna(value) or str(value).strip() == "" for value in group_values):
            raise ValueError("Paired-bootstrap groups may not be missing or blank")
        unit = "group"
    unique = np.unique(group_values)

    def evaluate(observed: np.ndarray, predicted: np.ndarray) -> float:
        if metric == "mae":
            return float(np.mean(np.abs(observed - predicted)))
        if metric == "rmse":
            return float(np.sqrt(np.mean((observed - predicted) ** 2)))
        return float(np.mean((observed - predicted) ** 2))

    point = evaluate(true, pred_a) - evaluate(true, pred_b)
    rng = np.random.default_rng(seed)
    differences: list[float] = []
    for _ in range(iterations):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([np.flatnonzero(group_values == group) for group in sampled])
        differences.append(
            evaluate(true[indices], pred_a[indices]) - evaluate(true[indices], pred_b[indices])
        )
    return {
        "metric": metric,
        "difference_definition": "model_a_minus_model_b; negative_favors_a",
        "point_difference": point,
        "lower_95": float(np.quantile(differences, 0.025)),
        "upper_95": float(np.quantile(differences, 0.975)),
        "probability_a_better_bootstrap": float(np.mean(np.asarray(differences) < 0)),
        "iterations": iterations,
        "seed": seed,
        "resampling_unit": unit,
    }


def subgroup_metrics(
    frame: pd.DataFrame,
    *,
    subgroup_columns: Sequence[str],
    target_column: str,
    prediction_column: str,
    task_type: Literal["regression", "classification"],
    minimum_size: int = 30,
    group_column: str | None = None,
    minimum_independent_groups: int = 30,
    minimum_events_per_class: int = 20,
) -> pd.DataFrame:
    """Long-form subgroup metrics with support and suppression reasons."""

    required = {target_column, prediction_column, *subgroup_columns}
    if group_column:
        required.add(group_column)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Subgroup metric input is missing columns: {missing}")
    rows: list[dict[str, Any]] = []
    for column in subgroup_columns:
        normalized = frame[column].fillna("__MISSING__").astype(str)
        for value in sorted(normalized.unique()):
            subset = frame.loc[normalized == value]
            base = {
                "subgroup_column": column,
                "subgroup_value": value,
                "n": int(len(subset)),
                "fraction": float(len(subset) / len(frame)) if len(frame) else None,
                "n_independent_groups": (
                    int(subset[group_column].nunique(dropna=True)) if group_column else None
                ),
                "minimum_size_policy": minimum_size,
                "minimum_independent_groups_policy": minimum_independent_groups,
                "minimum_events_per_class_policy": minimum_events_per_class,
            }
            if task_type == "classification":
                target_values = pd.to_numeric(subset[target_column], errors="coerce")
                base["n_events"] = int((target_values == 1).sum())
                base["n_non_events"] = int((target_values == 0).sum())
            if group_column is None:
                rows.append(
                    {
                        **base,
                        "status": "representation_only_missing_independent_group_column",
                        "metric": None,
                        "value": None,
                    }
                )
                continue
            if len(subset) < minimum_size or base["n_independent_groups"] < minimum_independent_groups:
                rows.append(
                    {
                        **base,
                        "status": "suppressed_insufficient_independent_support",
                        "metric": None,
                        "value": None,
                    }
                )
                continue
            if task_type == "classification":
                if min(base["n_events"], base["n_non_events"]) < minimum_events_per_class:
                    rows.append(
                        {
                            **base,
                            "status": "suppressed_insufficient_event_or_nonevent_support",
                            "metric": None,
                            "value": None,
                        }
                    )
                    continue
            try:
                metrics = (
                    regression_metrics(subset[target_column], subset[prediction_column])
                    if task_type == "regression"
                    else binary_classification_metrics(subset[target_column], subset[prediction_column])
                )
            except ValueError as exc:
                rows.append({**base, "status": f"not_estimable:{exc}", "metric": None, "value": None})
                continue
            for metric, metric_value in metrics.items():
                if metric in {"metric_schema_version", "confusion_matrix", "n"}:
                    continue
                rows.append(
                    {
                        **base,
                        "status": "estimated",
                        "metric": metric,
                        "value": _float_or_none(metric_value),
                    }
                )
    return pd.DataFrame(rows)


def applicability_domain_metrics(
    y_true: Iterable[object],
    prediction: Iterable[object],
    nearest_training_similarity: Iterable[object],
    *,
    task_type: Literal["regression", "classification"],
    bin_edges: Sequence[float] = (0.0, 0.3, 0.5, 0.7, 0.85, 1.0),
) -> pd.DataFrame:
    """Performance as a function of similarity to the training domain."""

    true, pred = _paired_arrays(y_true, prediction)
    similarity = np.asarray(list(nearest_training_similarity), dtype=float)
    if similarity.shape != true.shape or np.any(~np.isfinite(similarity)):
        raise ValueError("Similarity must be finite and match target length")
    edges = np.asarray(bin_edges, dtype=float)
    if len(edges) < 2 or np.any(np.diff(edges) <= 0):
        raise ValueError("bin_edges must be strictly increasing")
    if np.any(similarity < edges[0]) or np.any(similarity > edges[-1]):
        raise ValueError("Similarity values fall outside the declared applicability-domain bin coverage")
    bin_ids = np.searchsorted(edges[1:-1], similarity, side="right")
    rows: list[dict[str, Any]] = []
    for index in range(len(edges) - 1):
        mask = bin_ids == index
        if not np.any(mask):
            continue
        try:
            metrics = (
                regression_metrics(true[mask], pred[mask])
                if task_type == "regression"
                else binary_classification_metrics(true[mask], pred[mask])
            )
        except ValueError:
            metrics = {"n": int(np.sum(mask))}
        rows.append(
            {
                "similarity_lower": float(edges[index]),
                "similarity_upper": float(edges[index + 1]),
                "n": int(np.sum(mask)),
                "mean_similarity": float(np.mean(similarity[mask])),
                **{key: value for key, value in metrics.items() if key not in {"n", "confusion_matrix"}},
            }
        )
    return pd.DataFrame(rows)


def benjamini_hochberg(p_values: Iterable[object]) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values, preserving missing entries."""

    values = np.asarray(list(p_values), dtype=float)
    result = np.full(len(values), np.nan, dtype=float)
    finite_indices = np.flatnonzero(np.isfinite(values))
    finite = values[finite_indices]
    if np.any((finite < 0) | (finite > 1)):
        raise ValueError("p-values must lie in [0, 1]")
    if not len(finite):
        return result
    order = np.argsort(finite, kind="mergesort")
    ranked = finite[order]
    adjusted_ranked = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = np.minimum(adjusted_ranked, 1.0)
    result[finite_indices] = adjusted
    return result


def repeated_seed_summary(
    results: pd.DataFrame,
    *,
    metric_columns: Sequence[str],
    group_columns: Sequence[str] = ("model_name", "split_name"),
) -> pd.DataFrame:
    """Mean, SD, extrema, and seed count without treating repeats as independent evidence."""

    required = {*metric_columns, *group_columns, "seed"}
    missing = sorted(required - set(results.columns))
    if missing:
        raise ValueError(f"Repeated-seed results are missing columns: {missing}")
    rows: list[dict[str, Any]] = []
    for keys, subset in results.groupby(list(group_columns), dropna=False, sort=True):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        base = dict(zip(group_columns, key_values, strict=True))
        for metric in metric_columns:
            values = pd.to_numeric(subset[metric], errors="coerce").dropna()
            rows.append(
                {
                    **base,
                    "metric": metric,
                    "n_seeds": int(subset["seed"].nunique()),
                    "mean": float(values.mean()) if len(values) else None,
                    "standard_deviation": float(values.std(ddof=1)) if len(values) > 1 else None,
                    "minimum": float(values.min()) if len(values) else None,
                    "maximum": float(values.max()) if len(values) else None,
                    "interpretation": "algorithmic_seed_sensitivity_not_independent_external_validation",
                }
            )
    return pd.DataFrame(rows)
