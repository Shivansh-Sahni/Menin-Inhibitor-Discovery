"""Grouped, uncertainty-aware PK and hERG research models.

The module intentionally keeps two scientific tracks separate.  A conventional
2D representation is the decision baseline.  Mechanistic feature layers are
evaluated as controlled additions and remain discovery-only unless they are
calibrated and non-inferior under the same group-held-out splits.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, log_ndtr
from scipy.stats import norm, spearmanr
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from .features import nearest_neighbor_tanimoto, rdkit_descriptors, scaffold_key
from .research_feature_ontology import (
    CONVENTIONAL_DESCRIPTOR_COLUMNS,
    MODEL_PHYSICS_FEATURE_BLOCKS,
    selected_model_physics_features,
)

EPSILON = 1e-12


def _safe_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 3 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")
    return float(spearmanr(y_true, y_pred).statistic)


def _ece(y_true: np.ndarray, probability: np.ndarray, bins: int = 8) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    ids = np.clip(np.digitize(probability, edges[1:-1], right=True), 0, bins - 1)
    total = 0.0
    for bin_id in range(bins):
        mask = ids == bin_id
        if np.any(mask):
            total += float(np.mean(mask)) * abs(
                float(np.mean(y_true[mask])) - float(np.mean(probability[mask]))
            )
    return float(total)


def regression_metrics(
    y_true_log10: np.ndarray,
    y_pred_log10: np.ndarray,
    *,
    interval_lower: np.ndarray | None = None,
    interval_upper: np.ndarray | None = None,
    predictive_sigma: np.ndarray | None = None,
) -> dict[str, float]:
    """PK metrics on log10 values plus interpretable multiplicative errors."""

    y_true = np.asarray(y_true_log10, dtype=float)
    y_pred = np.asarray(y_pred_log10, dtype=float)
    finite = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[finite], y_pred[finite]
    error = y_pred - y_true
    fold = np.power(10.0, np.abs(error))
    metrics = {
        "n": float(len(y_true)),
        "log_mae": float(mean_absolute_error(y_true, y_pred)),
        "log_rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else float("nan"),
        "spearman": _safe_spearman(y_true, y_pred),
        "median_fold_error": float(np.median(fold)),
        "absolute_average_fold_error": float(np.mean(fold)),
        "fraction_within_2fold": float(np.mean(fold <= 2.0 * (1.0 + 1e-7))),
        "fraction_within_3fold": float(np.mean(fold <= 3.0 * (1.0 + 1e-7))),
    }
    if interval_lower is not None and interval_upper is not None:
        lower = np.asarray(interval_lower, dtype=float)[finite]
        upper = np.asarray(interval_upper, dtype=float)[finite]
        metrics["prediction_interval_coverage"] = float(np.mean((y_true >= lower) & (y_true <= upper)))
        metrics["prediction_interval_mean_width_log10"] = float(np.mean(upper - lower))
    if predictive_sigma is not None:
        sigma = np.maximum(np.asarray(predictive_sigma, dtype=float)[finite], 1e-6)
        z = error / sigma
        metrics["negative_log_likelihood"] = float(
            np.mean(np.log(sigma) + 0.5 * z**2 + 0.5 * np.log(2 * np.pi))
        )
        # Closed form CRPS for a Normal predictive distribution.
        metrics["crps"] = float(
            np.mean(sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi)))
        )
    return metrics


def herg_classification_metrics(y_true: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    probability = np.clip(np.asarray(probability, dtype=float), 1e-7, 1 - 1e-7)
    predicted = (probability >= 0.5).astype(int)
    both = len(np.unique(y)) == 2
    return {
        "n": float(len(y)),
        "roc_auc": float(roc_auc_score(y, probability)) if both else float("nan"),
        "pr_auc": float(average_precision_score(y, probability)) if both else float("nan"),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)) if both else float("nan"),
        "mcc": float(matthews_corrcoef(y, predicted)) if both else float("nan"),
        "sensitivity": float(np.mean(predicted[y == 1] == 1)) if np.any(y == 1) else float("nan"),
        "specificity": float(np.mean(predicted[y == 0] == 0)) if np.any(y == 0) else float("nan"),
        "brier": float(brier_score_loss(y, probability)),
        "ece_8bin": _ece(y, probability),
        "log_loss": float(log_loss(y, probability, labels=[0, 1])),
    }


def continuous_herg_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    error = np.abs(y_pred - y_true)
    return {
        "n_exact": float(len(y_true)),
        "pic50_mae": float(np.mean(error)),
        "pic50_rmse": float(np.sqrt(np.mean((y_pred - y_true) ** 2))),
        "spearman": _safe_spearman(y_true, y_pred),
        "fraction_within_0p5_log": float(np.mean(error <= 0.5)),
        "fraction_within_1p0_log": float(np.mean(error <= 1.0)),
    }


def structure_feature_frame(compounds: pd.DataFrame) -> pd.DataFrame:
    """Return interpretable 2D features keyed by compound and scaffold."""

    if "compound_id" not in compounds or "standardized_smiles" not in compounds:
        raise ValueError("Compounds require compound_id and standardized_smiles")
    smiles = compounds["standardized_smiles"].fillna("").astype(str)
    features = rdkit_descriptors(smiles)
    result = pd.concat([compounds[["compound_id"]].reset_index(drop=True), features], axis=1)
    scaffolds = [scaffold_key(value)[0] for value in smiles]
    result["scaffold"] = scaffolds
    return result


def merge_feature_layers(
    compounds: pd.DataFrame,
    physics: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Build named feature layers without descriptor proliferation or leakage.

    Rich physics outputs remain available for explanation and falsification,
    but only the small, predeclared causal set in the feature ontology enters
    this benchmark.  This is deliberately fail-closed: an unknown future
    numeric column is not silently promoted to a model input.
    """

    base = structure_feature_frame(compounds)
    layers: dict[str, list[str]] = {"structure_2d": list(CONVENTIONAL_DESCRIPTOR_COLUMNS)}
    if physics is not None and not physics.empty:
        if "compound_id" not in physics:
            raise ValueError("Physics summary requires compound_id")
        numeric = selected_model_physics_features(physics.columns)
        safe_physics = physics[["compound_id", *numeric]].copy()
        if "physics_model_eligible" in physics.columns:
            eligible = physics["physics_model_eligible"].fillna(False).astype(bool).to_numpy()
            safe_physics.loc[~eligible, numeric] = np.nan
            numeric = [
                column
                for column in numeric
                if np.isfinite(pd.to_numeric(safe_physics[column], errors="coerce")).any()
            ]
            safe_physics = safe_physics[["compound_id", *numeric]]
        if numeric:
            base = base.merge(safe_physics, on="compound_id", how="left", validate="one_to_one")
            for layer_name, block in MODEL_PHYSICS_FEATURE_BLOCKS.items():
                block_columns = [column for column in block if column in numeric]
                if block_columns:
                    layers[layer_name] = layers["structure_2d"] + block_columns
            layers["state_conformer_physics"] = layers["structure_2d"] + numeric
    return base, layers


def _group_splits(groups: np.ndarray, requested: int) -> list[tuple[np.ndarray, np.ndarray]]:
    unique = np.unique(groups)
    n_splits = min(int(requested), len(unique))
    if n_splits < 2:
        raise ValueError("At least two distinct series/scaffolds are required")
    return list(GroupKFold(n_splits=n_splits).split(np.zeros(len(groups)), groups=groups))


def _model_families(random_state: int, *, include_optional_boosters: bool = False) -> dict[str, Any]:
    families: dict[str, Any] = {
        "ridge": Pipeline(
            [("impute", SimpleImputer()), ("scale", StandardScaler()), ("model", Ridge(alpha=10.0))]
        ),
        "random_forest": Pipeline(
            [
                ("impute", SimpleImputer()),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=500,
                        min_samples_leaf=3,
                        max_features="sqrt",
                        n_jobs=-1,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "extra_trees": Pipeline(
            [
                ("impute", SimpleImputer()),
                (
                    "model",
                    ExtraTreesRegressor(
                        n_estimators=500,
                        min_samples_leaf=3,
                        max_features="sqrt",
                        n_jobs=-1,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "svr": Pipeline(
            [
                ("impute", SimpleImputer()),
                ("scale", StandardScaler()),
                ("model", SVR(C=3.0, epsilon=0.15, gamma="scale")),
            ]
        ),
    }
    # XGBoost/LightGBM are kept as conventional comparators, but loaded only
    # when explicitly requested.  On macOS their OpenMP runtimes can conflict
    # with PyTorch in the same process; production orchestration runs optional
    # boosters separately from the D-MPNN comparator.
    if include_optional_boosters:
        try:
            from xgboost import XGBRegressor

            families["xgboost"] = Pipeline(
                [
                    ("impute", SimpleImputer()),
                    (
                        "model",
                        XGBRegressor(
                            n_estimators=300,
                            max_depth=3,
                            learning_rate=0.03,
                            min_child_weight=3,
                            subsample=0.8,
                            colsample_bytree=0.8,
                            n_jobs=-1,
                            random_state=random_state,
                        ),
                    ),
                ]
            )
        except ImportError:
            pass
        try:
            from lightgbm import LGBMRegressor

            families["lightgbm"] = Pipeline(
                [
                    ("impute", SimpleImputer()),
                    (
                        "model",
                        LGBMRegressor(
                            n_estimators=300,
                            max_depth=4,
                            num_leaves=15,
                            learning_rate=0.03,
                            min_child_samples=10,
                            verbosity=-1,
                            random_state=random_state,
                        ),
                    ),
                ]
            )
        except ImportError:
            pass
    return families


def grouped_regression_benchmark(
    data: pd.DataFrame,
    *,
    feature_columns: list[str],
    target_column: str,
    group_column: str = "scaffold",
    folds: int = 5,
    random_state: int = 20260721,
    interval_level: float = 0.90,
    include_optional_boosters: bool = False,
    model_names: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate fixed conventional models with group-held-out split conformal intervals."""

    required = {target_column, group_column, "compound_id", *feature_columns}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Regression benchmark is missing columns: {missing}")
    frame = data.dropna(subset=[target_column, group_column]).reset_index(drop=True)
    y = np.log10(frame[target_column].astype(float).to_numpy())
    if not np.all(np.isfinite(y)):
        raise ValueError(f"{target_column} must contain positive finite values")
    X = frame[feature_columns].replace([np.inf, -np.inf], np.nan)
    groups = frame[group_column].astype(str).to_numpy()
    splits = _group_splits(groups, folds)
    prediction_rows: list[dict[str, Any]] = []

    families = _model_families(
        random_state,
        include_optional_boosters=include_optional_boosters,
    )
    if model_names is not None:
        requested = set(model_names)
        missing_models = sorted(requested - set(families))
        if missing_models:
            raise ValueError(f"Requested regression models are unavailable: {missing_models}")
        families = {name: model for name, model in families.items() if name in requested}
    for model_name, prototype in families.items():
        for fold_index, (train, test) in enumerate(splits):
            outer_groups = groups[train]
            if len(np.unique(outer_groups)) >= 3:
                calibration_split = GroupShuffleSplit(
                    n_splits=1, test_size=0.25, random_state=random_state + fold_index
                )
                fit_rel, cal_rel = next(calibration_split.split(train, groups=outer_groups))
                fit_indices, calibration_indices = train[fit_rel], train[cal_rel]
            else:
                fit_indices, calibration_indices = train, train
            model = clone(prototype)
            model.fit(X.iloc[fit_indices], y[fit_indices])
            calibration_prediction = np.asarray(model.predict(X.iloc[calibration_indices]), dtype=float)
            residual = np.abs(y[calibration_indices] - calibration_prediction)
            quantile_level = min(1.0, math.ceil((len(residual) + 1) * interval_level) / max(len(residual), 1))
            radius = (
                float(np.quantile(residual, quantile_level, method="higher"))
                if len(residual)
                else float("nan")
            )
            predicted = np.asarray(model.predict(X.iloc[test]), dtype=float)
            sigma = (
                max(float(np.std(y[calibration_indices] - calibration_prediction, ddof=1)), 1e-6)
                if len(residual) > 1
                else radius / 1.645
            )
            if "standardized_smiles" in frame:
                train_similarity, _, _ = nearest_neighbor_tanimoto(
                    frame.iloc[fit_indices]["standardized_smiles"],
                    frame.iloc[fit_indices]["standardized_smiles"],
                    exclude_identical_positions=True,
                )
                domain_threshold = float(np.quantile(train_similarity[np.isfinite(train_similarity)], 0.05))
                test_similarity, _, _ = nearest_neighbor_tanimoto(
                    frame.iloc[test]["standardized_smiles"],
                    frame.iloc[fit_indices]["standardized_smiles"],
                )
            else:
                domain_threshold = float("nan")
                test_similarity = np.full(len(test), np.nan)
            for position, index in enumerate(test):
                prediction_rows.append(
                    {
                        "compound_id": frame.loc[index, "compound_id"],
                        "model": model_name,
                        "fold": fold_index,
                        "group": groups[index],
                        "observed_log10": y[index],
                        "predicted_log10": predicted[position],
                        "interval_lower_log10": predicted[position] - radius,
                        "interval_upper_log10": predicted[position] + radius,
                        "predictive_sigma_log10": sigma,
                        "fit_rows": int(len(fit_indices)),
                        "fit_groups": int(len(np.unique(groups[fit_indices]))),
                        "calibration_rows": int(len(calibration_indices)),
                        "calibration_groups": int(len(np.unique(groups[calibration_indices]))),
                        "interval_radius_log10": radius,
                        "max_train_tanimoto": test_similarity[position],
                        "domain_threshold": domain_threshold,
                        "inside_applicability_domain": bool(test_similarity[position] >= domain_threshold)
                        if np.isfinite(domain_threshold)
                        else False,
                    }
                )
    predictions = pd.DataFrame(prediction_rows)
    metric_rows: list[dict[str, Any]] = []
    for model_name, group in predictions.groupby("model", sort=True):
        metric_rows.append(
            {
                "model": model_name,
                "evaluation_unit": "evidence_row",
                "n_unique_compounds": int(group["compound_id"].nunique()),
                **regression_metrics(
                    group["observed_log10"].to_numpy(),
                    group["predicted_log10"].to_numpy(),
                    interval_lower=group["interval_lower_log10"].to_numpy(),
                    interval_upper=group["interval_upper_log10"].to_numpy(),
                    predictive_sigma=group["predictive_sigma_log10"].to_numpy(),
                ),
            }
        )
    return pd.DataFrame(metric_rows).sort_values(["log_mae", "model"]).reset_index(drop=True), predictions


def _oof_residual_uncertainty(
    observed: np.ndarray,
    predicted: np.ndarray,
    *,
    interval_level: float,
) -> dict[str, float | int]:
    """Estimate predictive dispersion only from held-out residuals.

    Final-fit models are trained on every eligible observation, so their own
    residuals are necessarily optimistic.  This helper deliberately accepts
    only the group-held-out predictions emitted by the benchmark functions.
    """

    observed_array = np.asarray(observed, dtype=float)
    predicted_array = np.asarray(predicted, dtype=float)
    finite = np.isfinite(observed_array) & np.isfinite(predicted_array)
    residual = observed_array[finite] - predicted_array[finite]
    if not len(residual):
        raise ValueError("At least one finite group-held-out residual is required")
    quantile_level = min(
        1.0,
        math.ceil((len(residual) + 1) * interval_level) / len(residual),
    )
    radius = float(np.quantile(np.abs(residual), quantile_level, method="higher"))
    if len(residual) > 1:
        sigma = max(float(np.std(residual, ddof=1)), 1e-6)
    else:
        sigma = max(radius / 1.645, 1e-6)
    return {
        "oof_n": int(len(residual)),
        "oof_mae": float(np.mean(np.abs(residual))),
        "oof_sigma": sigma,
        "conformal_radius": radius,
    }


def _applicability_domain_for_final_fit(
    training: pd.DataFrame,
    scoring: pd.DataFrame,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Return nearest-training similarity, threshold, and inside-domain mask."""

    if "standardized_smiles" not in training or "standardized_smiles" not in scoring:
        return np.full(len(scoring), np.nan), float("nan"), np.zeros(len(scoring), dtype=bool)
    references = training[["compound_id", "standardized_smiles"]].drop_duplicates("compound_id")
    if len(references) < 2:
        return np.full(len(scoring), np.nan), float("nan"), np.zeros(len(scoring), dtype=bool)
    train_similarity, _, _ = nearest_neighbor_tanimoto(
        references["standardized_smiles"],
        references["standardized_smiles"],
        exclude_identical_positions=True,
    )
    finite_reference = train_similarity[np.isfinite(train_similarity)]
    threshold = float(np.quantile(finite_reference, 0.05)) if len(finite_reference) else float("nan")
    score_similarity, _, _ = nearest_neighbor_tanimoto(
        scoring["standardized_smiles"],
        references["standardized_smiles"],
    )
    inside = np.isfinite(score_similarity) & np.isfinite(threshold) & (score_similarity >= threshold)
    return score_similarity, threshold, inside


def final_fit_regression_predictions(
    training: pd.DataFrame,
    scoring: pd.DataFrame,
    oof_predictions: pd.DataFrame,
    *,
    feature_columns: list[str],
    target_column: str,
    selected_model: str,
    endpoint: str,
    unit: str,
    feature_layer: str = "structure_2d",
    group_column: str = "scaffold",
    interval_level: float = 0.90,
    random_state: int = 20260721,
    promotion_status: str = "provisional-discovery",
) -> pd.DataFrame:
    """Fit a selected grouped-CV regressor to all evidence and score a library.

    The target is modeled in log10 space.  Central estimates and intervals are
    returned in the endpoint's original units, while uncertainty is the
    conformal interval half-width.  No in-sample residual enters uncertainty.
    """

    required_training = {"compound_id", target_column, group_column, *feature_columns}
    if missing := sorted(required_training - set(training.columns)):
        raise ValueError(f"Final regression fit is missing training columns: {missing}")
    required_scoring = {"compound_id", *feature_columns}
    if missing := sorted(required_scoring - set(scoring.columns)):
        raise ValueError(f"Final regression fit is missing scoring columns: {missing}")
    frame = training.dropna(subset=[target_column, group_column]).reset_index(drop=True)
    target = frame[target_column].to_numpy(dtype=float)
    if not np.all(np.isfinite(target) & (target > 0)):
        raise ValueError(f"{target_column} must contain positive finite values")
    selected_oof = oof_predictions[oof_predictions["model"].astype(str) == selected_model]
    uncertainty = _oof_residual_uncertainty(
        selected_oof["observed_log10"].to_numpy(dtype=float),
        selected_oof["predicted_log10"].to_numpy(dtype=float),
        interval_level=interval_level,
    )
    family = _model_families(random_state, include_optional_boosters=False).get(selected_model)
    if family is None and selected_model in {"xgboost", "lightgbm"}:
        family = _model_families(random_state, include_optional_boosters=True).get(selected_model)
    if family is None:
        raise ValueError(f"Selected final-fit model is unavailable: {selected_model}")
    model = clone(family)
    X_train = frame[feature_columns].replace([np.inf, -np.inf], np.nan)
    score = scoring.drop_duplicates("compound_id").reset_index(drop=True)
    X_score = score[feature_columns].replace([np.inf, -np.inf], np.nan)
    model.fit(X_train, np.log10(target))
    predicted_log10 = np.asarray(model.predict(X_score), dtype=float)
    radius = float(uncertainty["conformal_radius"])
    central = np.power(10.0, predicted_log10)
    lower = np.power(10.0, predicted_log10 - radius)
    upper = np.power(10.0, predicted_log10 + radius)
    similarity, domain_threshold, inside = _applicability_domain_for_final_fit(frame, score)
    rows = pd.DataFrame(
        {
            "compound_id": score["compound_id"].astype(str),
            "endpoint": endpoint,
            "mean": central,
            "lower": lower,
            "upper": upper,
            "uncertainty": (upper - lower) / 2.0,
            "unit": unit,
            "domain_status": np.where(inside, "inside", "outside"),
            "promotion_status": promotion_status,
            "model_name": selected_model,
            "feature_layer": feature_layer,
            "max_train_tanimoto": similarity,
            "domain_threshold": domain_threshold,
            "training_rows": len(frame),
            "training_compounds": frame["compound_id"].nunique(),
            "oof_residual_rows": int(uncertainty["oof_n"]),
            "oof_mae_log10": float(uncertainty["oof_mae"]),
            "oof_sigma_log10": float(uncertainty["oof_sigma"]),
            "interval_level": interval_level,
            "uncertainty_method": "group-held-out OOF absolute-residual interval (cross-conformal heuristic)",
            "estimate_semantics": "back-transformed log10 location (median on original scale)",
            "lineage_role": "modeled_endpoint",
            "promotion_reason": (
                "Provisional discovery output; model-family selection and OOF uncertainty currently reuse the "
                "same grouped CV, and no untouched prospective calibration/final set is available."
            ),
        }
    )
    return rows


@dataclass
class CensoredGaussianRidge:
    """Normal linear model fitted by exact/interval/one-sided censored likelihood."""

    alpha: float = 1.0
    maxiter: int = 3000
    coefficients_: np.ndarray | None = None
    intercept_: float = 0.0
    sigma_: float = 1.0
    converged_: bool = False
    optimization_message_: str = "not_fitted"
    optimization_iterations_: int = 0
    objective_value_: float = float("nan")

    def fit(self, X: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> CensoredGaussianRidge:
        X = np.asarray(X, dtype=float)
        lower = np.asarray(lower, dtype=float)
        upper = np.asarray(upper, dtype=float)
        if len(X) != len(lower) or len(lower) != len(upper):
            raise ValueError("Censoring bounds and features must have equal row counts")
        exact = np.isfinite(lower) & np.isfinite(upper) & np.isclose(lower, upper)
        midpoint = np.where(
            exact, lower, np.where(np.isfinite(lower), lower, np.where(np.isfinite(upper), upper, 5.0))
        )
        initial_beta = np.linalg.lstsq(np.column_stack([np.ones(len(X)), X]), midpoint, rcond=None)[0]
        initial_sigma = max(
            float(np.std(midpoint - np.column_stack([np.ones(len(X)), X]) @ initial_beta)), 0.25
        )
        initial = np.concatenate([initial_beta, [np.log(initial_sigma)]])

        def objective_and_gradient(parameters: np.ndarray) -> tuple[float, np.ndarray]:
            intercept = parameters[0]
            beta = parameters[1:-1]
            exp_log_sigma = np.exp(parameters[-1])
            sigma = exp_log_sigma + 1e-6
            mu = intercept + X @ beta
            log_likelihood = np.zeros(len(X), dtype=float)
            gradient_mu = np.zeros(len(X), dtype=float)
            gradient_sigma = np.zeros(len(X), dtype=float)
            exact_mask = exact
            log_likelihood[exact_mask] = norm.logpdf(lower[exact_mask], loc=mu[exact_mask], scale=sigma)
            exact_delta = mu[exact_mask] - lower[exact_mask]
            gradient_mu[exact_mask] = exact_delta / sigma**2
            gradient_sigma[exact_mask] = 1.0 / sigma - exact_delta**2 / sigma**3
            lower_only = np.isfinite(lower) & ~np.isfinite(upper)
            lower_z = (lower[lower_only] - mu[lower_only]) / sigma
            lower_log_probability = norm.logsf(lower_z)
            log_likelihood[lower_only] = lower_log_probability
            lower_mills = np.exp(norm.logpdf(lower_z) - lower_log_probability)
            gradient_mu[lower_only] = -lower_mills / sigma
            gradient_sigma[lower_only] = -lower_mills * lower_z / sigma
            upper_only = ~np.isfinite(lower) & np.isfinite(upper)
            upper_z = (upper[upper_only] - mu[upper_only]) / sigma
            upper_log_probability = log_ndtr(upper_z)
            log_likelihood[upper_only] = upper_log_probability
            upper_mills = np.exp(norm.logpdf(upper_z) - upper_log_probability)
            gradient_mu[upper_only] = upper_mills / sigma
            gradient_sigma[upper_only] = upper_mills * upper_z / sigma
            interval = np.isfinite(lower) & np.isfinite(upper) & ~exact_mask
            if np.any(interval):
                interval_upper_z = (upper[interval] - mu[interval]) / sigma
                interval_lower_z = (lower[interval] - mu[interval]) / sigma
                upper_cdf = norm.cdf(interval_upper_z)
                lower_cdf = norm.cdf(interval_lower_z)
                probability = np.maximum(upper_cdf - lower_cdf, EPSILON)
                upper_density = norm.pdf(interval_upper_z)
                lower_density = norm.pdf(interval_lower_z)
                log_likelihood[interval] = np.log(probability)
                gradient_mu[interval] = (upper_density - lower_density) / (sigma * probability)
                gradient_sigma[interval] = (
                    interval_upper_z * upper_density - interval_lower_z * lower_density
                ) / (sigma * probability)
            objective = float(-np.sum(log_likelihood) + 0.5 * self.alpha * np.sum(beta**2))
            gradient = np.concatenate(
                [
                    [np.sum(gradient_mu)],
                    X.T @ gradient_mu + self.alpha * beta,
                    [np.sum(gradient_sigma) * exp_log_sigma],
                ]
            )
            return objective, gradient

        result = minimize(
            objective_and_gradient,
            initial,
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": self.maxiter},
        )
        if not np.isfinite(result.fun):
            raise RuntimeError(f"Censored likelihood failed: {result.message}")
        self.converged_ = bool(result.success)
        self.optimization_message_ = str(result.message)
        self.optimization_iterations_ = int(result.nit)
        self.objective_value_ = float(result.fun)
        self.intercept_ = float(result.x[0])
        self.coefficients_ = np.asarray(result.x[1:-1], dtype=float)
        self.sigma_ = float(np.exp(result.x[-1]) + 1e-6)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.coefficients_ is None:
            raise RuntimeError("Model is not fitted")
        return self.intercept_ + np.asarray(X, dtype=float) @ self.coefficients_

    def blocker_probability(self, X: np.ndarray, *, pic50_threshold: float = 5.0) -> np.ndarray:
        mu = self.predict(X)
        return norm.sf((pic50_threshold - mu) / self.sigma_)


@dataclass
class JointCensoredHergRidge(CensoredGaussianRidge):
    """Continuous pIC50 model jointly constrained by fixed-concentration inhibition."""

    inhibition_sigma_: float = 15.0
    hill_slope_: float = 1.0

    def fit_joint(
        self,
        potency_X: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        inhibition_X: np.ndarray,
        concentrations_um: np.ndarray,
        inhibition_percent: np.ndarray,
    ) -> JointCensoredHergRidge:
        potency_X = np.asarray(potency_X, dtype=float)
        inhibition_X = np.asarray(inhibition_X, dtype=float)
        lower, upper = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
        concentrations = np.asarray(concentrations_um, dtype=float)
        response = np.asarray(inhibition_percent, dtype=float)
        n_features = potency_X.shape[1] if len(potency_X) else inhibition_X.shape[1]
        exact = np.isfinite(lower) & np.isfinite(upper) & np.isclose(lower, upper)
        if exact.any():
            design = np.column_stack([np.ones(exact.sum()), potency_X[exact]])
            initial_beta = np.linalg.lstsq(design, lower[exact], rcond=None)[0]
        else:
            initial_beta = np.concatenate([[5.0], np.zeros(n_features)])
        initial = np.concatenate([initial_beta, [np.log(0.7), np.log(15.0), 0.0]])

        def potency_log_likelihood(mu: np.ndarray, sigma: float) -> np.ndarray:
            values = np.zeros(len(mu), dtype=float)
            values[exact] = norm.logpdf(lower[exact], loc=mu[exact], scale=sigma)
            lower_only = np.isfinite(lower) & ~np.isfinite(upper)
            values[lower_only] = norm.logsf((lower[lower_only] - mu[lower_only]) / sigma)
            upper_only = ~np.isfinite(lower) & np.isfinite(upper)
            values[upper_only] = norm.logcdf((upper[upper_only] - mu[upper_only]) / sigma)
            interval = np.isfinite(lower) & np.isfinite(upper) & ~exact
            if interval.any():
                probability = norm.cdf((upper[interval] - mu[interval]) / sigma) - norm.cdf(
                    (lower[interval] - mu[interval]) / sigma
                )
                values[interval] = np.log(np.maximum(probability, EPSILON))
            return values

        def objective(parameters: np.ndarray) -> float:
            intercept, beta = parameters[0], parameters[1 : 1 + n_features]
            sigma = np.exp(parameters[-3]) + 1e-6
            inhibition_sigma = np.exp(parameters[-2]) + 1e-6
            hill = 0.25 + 2.75 * expit(parameters[-1])
            logp = 0.0
            if len(potency_X):
                mu = intercept + potency_X @ beta
                logp += float(np.sum(potency_log_likelihood(mu, sigma)))
            if len(inhibition_X):
                mu_inhibition = intercept + inhibition_X @ beta
                # Algebraically identical to the Hill equation but evaluated
                # in log space so extreme trial parameters cannot overflow.
                log_concentration = np.log(np.maximum(concentrations, 1e-12))
                log_ic50_um = np.log(10.0) * (6.0 - mu_inhibition)
                expected = 100.0 * expit(hill * (log_concentration - log_ic50_um))
                logp += float(np.sum(norm.logpdf(response, loc=expected, scale=inhibition_sigma)))
            return float(-logp + 0.5 * self.alpha * np.sum(beta**2))

        result = minimize(objective, initial, method="L-BFGS-B", options={"maxiter": self.maxiter})
        if not np.isfinite(result.fun):
            raise RuntimeError(f"Joint hERG likelihood failed: {result.message}")
        self.converged_ = bool(result.success)
        self.optimization_message_ = str(result.message)
        self.optimization_iterations_ = int(result.nit)
        self.objective_value_ = float(result.fun)
        self.intercept_ = float(result.x[0])
        self.coefficients_ = np.asarray(result.x[1 : 1 + n_features])
        self.sigma_ = float(np.exp(result.x[-3]) + 1e-6)
        self.inhibition_sigma_ = float(np.exp(result.x[-2]) + 1e-6)
        self.hill_slope_ = float(0.25 + 2.75 / (1.0 + np.exp(-result.x[-1])))
        return self


def _prepare_numeric_features(train: pd.DataFrame, test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    imputer = SimpleImputer()
    scaler = StandardScaler()
    train_i = imputer.fit_transform(train)
    test_i = imputer.transform(test)
    return scaler.fit_transform(train_i), scaler.transform(test_i)


def grouped_censored_herg_benchmark(
    data: pd.DataFrame,
    *,
    feature_columns: list[str],
    lower_column: str = "pic50_lower",
    upper_column: str = "pic50_upper",
    group_column: str = "scaffold",
    folds: int = 5,
    blocker_threshold_pic50: float = 5.0,
    alpha: float = 3.0,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Nested-group-safe continuous hERG fit with one-sided censoring retained."""

    required = {"compound_id", lower_column, upper_column, group_column, *feature_columns}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"hERG benchmark is missing columns: {missing}")
    frame = data.copy().reset_index(drop=True)
    lower = pd.to_numeric(frame[lower_column], errors="coerce").to_numpy(dtype=float)
    upper = pd.to_numeric(frame[upper_column], errors="coerce").to_numpy(dtype=float)
    informative = np.isfinite(lower) | np.isfinite(upper)
    frame, lower, upper = (
        frame.loc[informative].reset_index(drop=True),
        lower[informative],
        upper[informative],
    )
    groups = frame[group_column].fillna("").astype(str).to_numpy()
    splits = _group_splits(groups, folds)
    rows: list[dict[str, Any]] = []
    X = frame[feature_columns].replace([np.inf, -np.inf], np.nan)
    for fold_index, (train, test) in enumerate(splits):
        X_train, X_test = _prepare_numeric_features(X.iloc[train], X.iloc[test])
        model = CensoredGaussianRidge(alpha=alpha).fit(X_train, lower[train], upper[train])
        prediction = model.predict(X_test)
        probability = model.blocker_probability(X_test, pic50_threshold=blocker_threshold_pic50)
        if "standardized_smiles" in frame:
            train_similarity, _, _ = nearest_neighbor_tanimoto(
                frame.iloc[train]["standardized_smiles"],
                frame.iloc[train]["standardized_smiles"],
                exclude_identical_positions=True,
            )
            domain_threshold = float(np.quantile(train_similarity[np.isfinite(train_similarity)], 0.05))
            test_similarity, _, _ = nearest_neighbor_tanimoto(
                frame.iloc[test]["standardized_smiles"], frame.iloc[train]["standardized_smiles"]
            )
        else:
            domain_threshold = float("nan")
            test_similarity = np.full(len(test), np.nan)
        for position, index in enumerate(test):
            lo, hi = lower[index], upper[index]
            exact = np.isfinite(lo) and np.isfinite(hi) and np.isclose(lo, hi)
            decisive_label: int | None = None
            # Strong blocker is support entirely at/above pIC50 5.  A nonblocker
            # is support entirely at/below pIC50 for 30 uM (4.5229).
            if np.isfinite(lo) and lo >= blocker_threshold_pic50:
                decisive_label = 1
            elif np.isfinite(hi) and hi <= 6.0 - np.log10(30.0):
                decisive_label = 0
            rows.append(
                {
                    "compound_id": frame.loc[index, "compound_id"],
                    "fold": fold_index,
                    "group": groups[index],
                    "pic50_lower": lo,
                    "pic50_upper": hi,
                    "is_exact": exact,
                    "observed_pic50": lo if exact else np.nan,
                    "predicted_pic50": prediction[position],
                    "predictive_sigma": model.sigma_,
                    "blocker_probability": probability[position],
                    "decisive_label": decisive_label,
                    "fit_converged": model.converged_,
                    "optimization_iterations": model.optimization_iterations_,
                    "max_train_tanimoto": test_similarity[position],
                    "domain_threshold": domain_threshold,
                    "inside_applicability_domain": bool(test_similarity[position] >= domain_threshold)
                    if np.isfinite(domain_threshold)
                    else False,
                }
            )
    predictions = pd.DataFrame(rows)
    exact = predictions["is_exact"].astype(bool)
    metrics: dict[str, Any] = {
        "evaluation_unit": "measurement_row",
        "n_unique_compounds": int(predictions["compound_id"].nunique()),
        "censored_negative_log_likelihood": _heldout_censored_nll(predictions),
        "fit_converged_fraction": float(predictions["fit_converged"].mean()),
        **continuous_herg_metrics(
            predictions.loc[exact, "observed_pic50"].to_numpy(),
            predictions.loc[exact, "predicted_pic50"].to_numpy(),
        ),
    }
    decisive = predictions["decisive_label"].notna()
    if decisive.any():
        metrics.update(
            {
                f"classification_{key}": value
                for key, value in herg_classification_metrics(
                    predictions.loc[decisive, "decisive_label"].astype(int).to_numpy(),
                    predictions.loc[decisive, "blocker_probability"].to_numpy(),
                ).items()
            }
        )
    return metrics, predictions


def final_fit_censored_herg_predictions(
    training: pd.DataFrame,
    scoring: pd.DataFrame,
    oof_predictions: pd.DataFrame,
    *,
    feature_columns: list[str],
    lower_column: str = "pic50_lower",
    upper_column: str = "pic50_upper",
    interval_level: float = 0.90,
    blocker_threshold_pic50: float = 5.0,
    alpha: float = 3.0,
    feature_layer: str = "structure_2d",
    promotion_status: str = "provisional-discovery",
) -> pd.DataFrame:
    """Fit the censored hERG model to all evidence and score every compound.

    Both pIC50 and blocker probability are views of one fitted continuous
    endpoint.  Probability is therefore derived rather than independently
    classified.  Intervals and probability dispersion come exclusively from
    exact, group-held-out pIC50 residuals.
    """

    required_training = {"compound_id", lower_column, upper_column, *feature_columns}
    if missing := sorted(required_training - set(training.columns)):
        raise ValueError(f"Final censored hERG fit is missing training columns: {missing}")
    required_scoring = {"compound_id", *feature_columns}
    if missing := sorted(required_scoring - set(scoring.columns)):
        raise ValueError(f"Final censored hERG fit is missing scoring columns: {missing}")
    frame = training.copy().reset_index(drop=True)
    lower_bounds = pd.to_numeric(frame[lower_column], errors="coerce").to_numpy(dtype=float)
    upper_bounds = pd.to_numeric(frame[upper_column], errors="coerce").to_numpy(dtype=float)
    informative = np.isfinite(lower_bounds) | np.isfinite(upper_bounds)
    frame = frame.loc[informative].reset_index(drop=True)
    lower_bounds, upper_bounds = lower_bounds[informative], upper_bounds[informative]
    exact_oof = oof_predictions[oof_predictions["is_exact"].astype(bool)].copy()
    uncertainty = _oof_residual_uncertainty(
        exact_oof["observed_pic50"].to_numpy(dtype=float),
        exact_oof["predicted_pic50"].to_numpy(dtype=float),
        interval_level=interval_level,
    )
    score = scoring.drop_duplicates("compound_id").reset_index(drop=True)
    X_train, X_score = _prepare_numeric_features(
        frame[feature_columns].replace([np.inf, -np.inf], np.nan),
        score[feature_columns].replace([np.inf, -np.inf], np.nan),
    )
    model = CensoredGaussianRidge(alpha=alpha).fit(X_train, lower_bounds, upper_bounds)
    predicted_pic50 = model.predict(X_score)
    radius = float(uncertainty["conformal_radius"])
    predictive_sigma = float(uncertainty["oof_sigma"])
    pic50_lower = predicted_pic50 - radius
    pic50_upper = predicted_pic50 + radius
    probability = norm.sf((blocker_threshold_pic50 - predicted_pic50) / predictive_sigma)
    probability_lower = norm.sf((blocker_threshold_pic50 - pic50_lower) / predictive_sigma)
    probability_upper = norm.sf((blocker_threshold_pic50 - pic50_upper) / predictive_sigma)
    similarity, domain_threshold, inside = _applicability_domain_for_final_fit(frame, score)

    final_promotion_status = promotion_status if model.converged_ else "rejected"
    common: dict[str, Any] = {
        "promotion_status": final_promotion_status,
        "model_name": "censored_gaussian_ridge",
        "feature_layer": feature_layer,
        "domain_threshold": domain_threshold,
        "training_rows": len(frame),
        "training_compounds": frame["compound_id"].nunique(),
        "oof_residual_rows": int(uncertainty["oof_n"]),
        "oof_mae_pic50": float(uncertainty["oof_mae"]),
        "oof_sigma_pic50": predictive_sigma,
        "interval_level": interval_level,
        "fit_converged": model.converged_,
        "optimization_iterations": model.optimization_iterations_,
        "optimization_message": model.optimization_message_,
        "uncertainty_method": "exact group-held-out OOF pIC50 residual interval (cross-conformal heuristic)",
        "lineage_role": "modeled_endpoint",
        "promotion_reason": (
            "Rejected because the final censored-likelihood optimization did not converge."
            if not model.converged_
            else "Provisional discovery output; censoring is retained, but OOF uncertainty is cross-validated "
            "rather than prospectively calibrated and no untouched final set is available."
        ),
    }
    pic50 = pd.DataFrame(
        {
            "compound_id": score["compound_id"].astype(str),
            "endpoint": "herg_pic50",
            "mean": predicted_pic50,
            "lower": pic50_lower,
            "upper": pic50_upper,
            "uncertainty": (pic50_upper - pic50_lower) / 2.0,
            "unit": "pIC50",
            "domain_status": np.where(inside, "inside", "outside"),
            "max_train_tanimoto": similarity,
            "estimate_semantics": "censored-likelihood pIC50 location",
            **common,
        }
    )
    blocker = pd.DataFrame(
        {
            "compound_id": score["compound_id"].astype(str),
            "endpoint": "herg_blocker_probability",
            "mean": probability,
            "lower": probability_lower,
            "upper": probability_upper,
            "uncertainty": (probability_upper - probability_lower) / 2.0,
            "unit": "probability",
            "domain_status": np.where(inside, "inside", "outside"),
            "max_train_tanimoto": similarity,
            "estimate_semantics": "P(pIC50 >= 5) derived from the continuous pIC50 distribution",
            **common,
        }
    )
    return pd.concat([pic50, blocker], ignore_index=True)


def grouped_joint_herg_benchmark(
    compounds: pd.DataFrame,
    potency: pd.DataFrame,
    inhibition: pd.DataFrame,
    *,
    feature_columns: list[str],
    folds: int = 5,
    alpha: float = 3.0,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Joint pIC50/inhibition evaluation with all evidence held out by scaffold."""

    compound_required = {"compound_id", "scaffold", *feature_columns}
    potency_required = {"compound_id", "pic50_lower", "pic50_upper"}
    inhibition_required = {"compound_id", "test_concentration_um", "inhibition_percent"}
    if missing := sorted(compound_required - set(compounds.columns)):
        raise ValueError(f"Joint hERG compounds are missing columns: {missing}")
    if missing := sorted(potency_required - set(potency.columns)):
        raise ValueError(f"Joint hERG potency is missing columns: {missing}")
    if missing := sorted(inhibition_required - set(inhibition.columns)):
        raise ValueError(f"Joint hERG inhibition is missing columns: {missing}")
    feature_frame = compounds.drop_duplicates("compound_id").copy()
    feature_frame["compound_id"] = feature_frame["compound_id"].astype(str)
    feature_frame = feature_frame.set_index("compound_id")
    evaluable_ids = [
        value for value in potency["compound_id"].astype(str).unique() if value in feature_frame.index
    ]
    evaluation = feature_frame.loc[evaluable_ids].reset_index()
    groups = evaluation["scaffold"].astype(str).to_numpy()
    splits = _group_splits(groups, folds)
    rows: list[dict[str, Any]] = []
    inhibition_rows: list[dict[str, Any]] = []
    for fold_index, (train, test) in enumerate(splits):
        train_ids = set(evaluation.iloc[train]["compound_id"].astype(str))
        test_ids = set(evaluation.iloc[test]["compound_id"].astype(str))
        potency_train = potency[potency["compound_id"].astype(str).isin(train_ids)].copy()
        inhibition_train = inhibition[inhibition["compound_id"].astype(str).isin(train_ids)].copy()
        # Fit imputation/scaling only on distinct training compounds.
        train_features = feature_frame.loc[list(train_ids), feature_columns]
        test_features = feature_frame.loc[list(test_ids), feature_columns]
        imputer = SimpleImputer()
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(imputer.fit_transform(train_features))
        test_scaled = scaler.transform(imputer.transform(test_features))
        train_lookup = dict(zip(train_features.index.astype(str), train_scaled, strict=True))
        test_lookup = dict(zip(test_features.index.astype(str), test_scaled, strict=True))
        potency_X = np.vstack([train_lookup[str(value)] for value in potency_train["compound_id"]])
        inhibition_X = (
            np.vstack([train_lookup[str(value)] for value in inhibition_train["compound_id"]])
            if len(inhibition_train)
            else np.empty((0, len(feature_columns)))
        )
        model = JointCensoredHergRidge(alpha=alpha).fit_joint(
            potency_X,
            potency_train["pic50_lower"].to_numpy(dtype=float),
            potency_train["pic50_upper"].to_numpy(dtype=float),
            inhibition_X,
            inhibition_train["test_concentration_um"].to_numpy(dtype=float),
            inhibition_train["inhibition_percent"].to_numpy(dtype=float),
        )
        test_id_order = list(test_features.index.astype(str))
        prediction = model.predict(np.vstack([test_lookup[value] for value in test_id_order]))
        probability = model.blocker_probability(np.vstack([test_lookup[value] for value in test_id_order]))
        predicted_lookup = dict(zip(test_id_order, prediction, strict=True))
        probability_lookup = dict(zip(test_id_order, probability, strict=True))
        potency_test = potency[potency["compound_id"].astype(str).isin(test_ids)]
        for evidence in potency_test.itertuples(index=False):
            lo, hi = float(evidence.pic50_lower), float(evidence.pic50_upper)
            exact = np.isfinite(lo) and np.isfinite(hi) and np.isclose(lo, hi)
            decisive = (
                1
                if np.isfinite(lo) and lo >= 5.0
                else 0
                if np.isfinite(hi) and hi <= 6.0 - np.log10(30.0)
                else np.nan
            )
            compound_id = str(evidence.compound_id)
            rows.append(
                {
                    "compound_id": compound_id,
                    "fold": fold_index,
                    "group": str(feature_frame.loc[compound_id, "scaffold"]),
                    "pic50_lower": lo,
                    "pic50_upper": hi,
                    "is_exact": exact,
                    "observed_pic50": lo if exact else np.nan,
                    "predicted_pic50": float(predicted_lookup[compound_id]),
                    "predictive_sigma": model.sigma_,
                    "inhibition_sigma_percent": model.inhibition_sigma_,
                    "fitted_hill_slope": model.hill_slope_,
                    "blocker_probability": float(probability_lookup[compound_id]),
                    "decisive_label": decisive,
                    "training_inhibition_rows": int(len(inhibition_train)),
                    "fit_converged": model.converged_,
                    "optimization_iterations": model.optimization_iterations_,
                    "evidence_type": "potency",
                    "test_concentration_um": np.nan,
                    "observed_inhibition_percent": np.nan,
                    "predicted_inhibition_percent": np.nan,
                }
            )
        inhibition_test = inhibition[inhibition["compound_id"].astype(str).isin(test_ids)]
        for evidence in inhibition_test.itertuples(index=False):
            compound_id = str(evidence.compound_id)
            predicted_pic50 = float(predicted_lookup[compound_id])
            concentration = float(evidence.test_concentration_um)
            log_concentration = np.log(max(concentration, 1e-12))
            log_ic50_um = np.log(10.0) * (6.0 - predicted_pic50)
            predicted_inhibition = 100.0 * expit(model.hill_slope_ * (log_concentration - log_ic50_um))
            inhibition_rows.append(
                {
                    "compound_id": compound_id,
                    "fold": fold_index,
                    "group": str(feature_frame.loc[compound_id, "scaffold"]),
                    "pic50_lower": np.nan,
                    "pic50_upper": np.nan,
                    "is_exact": False,
                    "observed_pic50": np.nan,
                    "predicted_pic50": predicted_pic50,
                    "predictive_sigma": model.sigma_,
                    "inhibition_sigma_percent": model.inhibition_sigma_,
                    "fitted_hill_slope": model.hill_slope_,
                    "blocker_probability": float(probability_lookup[compound_id]),
                    "decisive_label": np.nan,
                    "training_inhibition_rows": int(len(inhibition_train)),
                    "fit_converged": model.converged_,
                    "optimization_iterations": model.optimization_iterations_,
                    "evidence_type": "concentration_inhibition",
                    "test_concentration_um": concentration,
                    "observed_inhibition_percent": float(evidence.inhibition_percent),
                    "predicted_inhibition_percent": predicted_inhibition,
                }
            )
    potency_predictions = pd.DataFrame(rows)
    inhibition_predictions = pd.DataFrame(inhibition_rows)
    predictions = pd.concat([potency_predictions, inhibition_predictions], ignore_index=True, sort=False)
    exact = potency_predictions["is_exact"].astype(bool)
    metrics: dict[str, Any] = {
        "model": "joint_censored_pic50_plus_concentration_inhibition",
        "evaluation_unit": "measurement_row",
        "n_unique_potency_compounds": int(potency_predictions["compound_id"].nunique()),
        "n_unique_inhibition_compounds": int(inhibition_predictions["compound_id"].nunique()),
        "censored_negative_log_likelihood": _heldout_censored_nll(potency_predictions),
        "fit_converged_fraction": float(potency_predictions["fit_converged"].mean()),
        **continuous_herg_metrics(
            potency_predictions.loc[exact, "observed_pic50"],
            potency_predictions.loc[exact, "predicted_pic50"],
        ),
        "median_fitted_hill_slope": float(potency_predictions["fitted_hill_slope"].median()),
        "median_inhibition_sigma_percent": float(potency_predictions["inhibition_sigma_percent"].median()),
    }
    if not inhibition_predictions.empty:
        inhibition_error = (
            inhibition_predictions["predicted_inhibition_percent"]
            - inhibition_predictions["observed_inhibition_percent"]
        ).to_numpy(dtype=float)
        sigma = np.maximum(inhibition_predictions["inhibition_sigma_percent"].to_numpy(dtype=float), 1e-6)
        metrics.update(
            {
                "heldout_inhibition_n": int(len(inhibition_predictions)),
                "heldout_inhibition_mae_percent": float(np.mean(np.abs(inhibition_error))),
                "heldout_inhibition_rmse_percent": float(np.sqrt(np.mean(inhibition_error**2))),
                "heldout_inhibition_negative_log_likelihood": float(
                    np.mean(np.log(sigma) + 0.5 * (inhibition_error / sigma) ** 2 + 0.5 * np.log(2 * np.pi))
                ),
            }
        )
    decisive = potency_predictions["decisive_label"].notna()
    if decisive.any():
        metrics.update(
            {
                f"classification_{key}": value
                for key, value in herg_classification_metrics(
                    potency_predictions.loc[decisive, "decisive_label"].astype(int),
                    potency_predictions.loc[decisive, "blocker_probability"],
                ).items()
            }
        )
    return metrics, predictions


def grouped_exact_herg_benchmark(
    data: pd.DataFrame,
    *,
    feature_columns: list[str],
    target_column: str = "observed_pic50",
    group_column: str = "scaffold",
    folds: int = 5,
    interval_level: float = 0.90,
    random_state: int = 20260721,
    include_optional_boosters: bool = False,
    model_names: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Conventional exact-pIC50 comparator on the same group-held-out folds."""

    required = {"compound_id", target_column, group_column, *feature_columns}
    if missing := sorted(required - set(data.columns)):
        raise ValueError(f"Exact hERG benchmark is missing columns: {missing}")
    frame = data.dropna(subset=[target_column, group_column]).reset_index(drop=True)
    y = frame[target_column].to_numpy(dtype=float)
    X = frame[feature_columns].replace([np.inf, -np.inf], np.nan)
    groups = frame[group_column].astype(str).to_numpy()
    splits = _group_splits(groups, folds)
    families = _model_families(random_state, include_optional_boosters=include_optional_boosters)
    if model_names is not None:
        missing_models = sorted(set(model_names) - set(families))
        if missing_models:
            raise ValueError(f"Requested exact hERG models are unavailable: {missing_models}")
        families = {name: model for name, model in families.items() if name in set(model_names)}
    rows: list[dict[str, Any]] = []
    for model_name, prototype in families.items():
        for fold_index, (train, test) in enumerate(splits):
            outer_groups = groups[train]
            if len(np.unique(outer_groups)) >= 3:
                fit_rel, calibration_rel = next(
                    GroupShuffleSplit(
                        n_splits=1, test_size=0.25, random_state=random_state + fold_index
                    ).split(train, groups=outer_groups)
                )
                fit_indices, calibration_indices = train[fit_rel], train[calibration_rel]
            else:
                fit_indices = calibration_indices = train
            model = clone(prototype)
            model.fit(X.iloc[fit_indices], y[fit_indices])
            calibration_prediction = np.asarray(model.predict(X.iloc[calibration_indices]), dtype=float)
            residual = np.abs(y[calibration_indices] - calibration_prediction)
            quantile_level = min(1.0, math.ceil((len(residual) + 1) * interval_level) / max(len(residual), 1))
            radius = float(np.quantile(residual, quantile_level, method="higher"))
            sigma = (
                max(float(np.std(y[calibration_indices] - calibration_prediction, ddof=1)), 1e-6)
                if len(residual) > 1
                else max(radius / 1.645, 1e-6)
            )
            predicted = np.asarray(model.predict(X.iloc[test]), dtype=float)
            probability = norm.sf((5.0 - predicted) / sigma)
            if "standardized_smiles" in frame:
                train_similarity, _, _ = nearest_neighbor_tanimoto(
                    frame.iloc[fit_indices]["standardized_smiles"],
                    frame.iloc[fit_indices]["standardized_smiles"],
                    exclude_identical_positions=True,
                )
                domain_threshold = float(np.quantile(train_similarity[np.isfinite(train_similarity)], 0.05))
                test_similarity, _, _ = nearest_neighbor_tanimoto(
                    frame.iloc[test]["standardized_smiles"],
                    frame.iloc[fit_indices]["standardized_smiles"],
                )
            else:
                domain_threshold = float("nan")
                test_similarity = np.full(len(test), np.nan)
            for position, index in enumerate(test):
                observed = y[index]
                decisive = 1 if observed >= 5.0 else 0 if observed <= 6.0 - np.log10(30.0) else np.nan
                rows.append(
                    {
                        "compound_id": frame.loc[index, "compound_id"],
                        "model": model_name,
                        "fold": fold_index,
                        "group": groups[index],
                        "observed_pic50": observed,
                        "predicted_pic50": predicted[position],
                        "interval_lower_pic50": predicted[position] - radius,
                        "interval_upper_pic50": predicted[position] + radius,
                        "predictive_sigma": sigma,
                        "fit_rows": int(len(fit_indices)),
                        "fit_groups": int(len(np.unique(groups[fit_indices]))),
                        "calibration_rows": int(len(calibration_indices)),
                        "calibration_groups": int(len(np.unique(groups[calibration_indices]))),
                        "interval_radius_pic50": radius,
                        "blocker_probability": probability[position],
                        "decisive_label": decisive,
                        "max_train_tanimoto": test_similarity[position],
                        "domain_threshold": domain_threshold,
                        "inside_applicability_domain": bool(test_similarity[position] >= domain_threshold)
                        if np.isfinite(domain_threshold)
                        else False,
                    }
                )
    predictions = pd.DataFrame(rows)
    metric_rows: list[dict[str, Any]] = []
    for model_name, group in predictions.groupby("model", sort=True):
        metrics = continuous_herg_metrics(group["observed_pic50"], group["predicted_pic50"])
        decisive = group["decisive_label"].notna()
        if decisive.any():
            metrics.update(
                {
                    f"classification_{key}": value
                    for key, value in herg_classification_metrics(
                        group.loc[decisive, "decisive_label"].astype(int),
                        group.loc[decisive, "blocker_probability"],
                    ).items()
                }
            )
        metrics["prediction_interval_coverage"] = float(
            np.mean(
                (group["observed_pic50"] >= group["interval_lower_pic50"])
                & (group["observed_pic50"] <= group["interval_upper_pic50"])
            )
        )
        metrics["inside_domain_fraction"] = float(group["inside_applicability_domain"].mean())
        metric_rows.append(
            {
                "model": model_name,
                "evaluation_unit": "measurement_row",
                "n_unique_compounds": int(group["compound_id"].nunique()),
                **metrics,
            }
        )
    return pd.DataFrame(metric_rows).sort_values(["pic50_mae", "model"]).reset_index(drop=True), predictions


def _heldout_censored_nll(predictions: pd.DataFrame) -> float:
    mu = predictions["predicted_pic50"].to_numpy(dtype=float)
    sigma = np.maximum(predictions["predictive_sigma"].to_numpy(dtype=float), 1e-6)
    lower = predictions["pic50_lower"].to_numpy(dtype=float)
    upper = predictions["pic50_upper"].to_numpy(dtype=float)
    exact = np.isfinite(lower) & np.isfinite(upper) & np.isclose(lower, upper)
    logp = np.zeros(len(mu), dtype=float)
    logp[exact] = norm.logpdf(lower[exact], loc=mu[exact], scale=sigma[exact])
    lower_only = np.isfinite(lower) & ~np.isfinite(upper)
    logp[lower_only] = norm.logsf((lower[lower_only] - mu[lower_only]) / sigma[lower_only])
    upper_only = ~np.isfinite(lower) & np.isfinite(upper)
    logp[upper_only] = norm.logcdf((upper[upper_only] - mu[upper_only]) / sigma[upper_only])
    interval = np.isfinite(lower) & np.isfinite(upper) & ~exact
    if np.any(interval):
        probability = norm.cdf((upper[interval] - mu[interval]) / sigma[interval]) - norm.cdf(
            (lower[interval] - mu[interval]) / sigma[interval]
        )
        logp[interval] = np.log(np.maximum(probability, EPSILON))
    return float(-np.mean(logp))


def matched_pair_direction_accuracy(
    predictions: pd.DataFrame,
    pairs: pd.DataFrame,
    *,
    observed_column: str,
    predicted_column: str,
) -> dict[str, float]:
    """Score whether a model recovers signed changes for predeclared matched pairs."""

    lookup = predictions.set_index("compound_id")[[observed_column, predicted_column]]
    outcomes: list[bool] = []
    for row in pairs.itertuples(index=False):
        left, right = str(row.compound_id_a), str(row.compound_id_b)
        if left not in lookup.index or right not in lookup.index:
            continue
        observed_delta = float(lookup.loc[right, observed_column] - lookup.loc[left, observed_column])
        predicted_delta = float(lookup.loc[right, predicted_column] - lookup.loc[left, predicted_column])
        if observed_delta == 0:
            continue
        outcomes.append(bool(np.sign(observed_delta) == np.sign(predicted_delta)))
    return {
        "n_pairs": float(len(outcomes)),
        "direction_accuracy": float(np.mean(outcomes)) if outcomes else float("nan"),
    }


def promotion_decision(
    baseline_metrics: dict[str, Any],
    candidate_metrics: dict[str, Any],
    *,
    primary_metric: str,
    lower_is_better: bool = True,
    calibrated: bool,
    converged: bool,
) -> dict[str, Any]:
    """Apply the locked two-track rule without hiding a mechanistically useful loss."""

    baseline = float(baseline_metrics[primary_metric])
    candidate = float(candidate_metrics[primary_metric])
    noninferior = candidate <= baseline if lower_is_better else candidate >= baseline
    decision_ready = bool(noninferior and calibrated and converged)
    return {
        "promotion_status": "decision-track" if decision_ready else "discovery-track",
        "primary_metric": primary_metric,
        "baseline_value": baseline,
        "candidate_value": candidate,
        "noninferior_to_baseline": bool(noninferior),
        "calibrated": bool(calibrated),
        "physics_converged": bool(converged),
        "reason": (
            "Meets non-inferiority, calibration, and convergence gates."
            if decision_ready
            else "Retained for mechanism discovery; at least one decision-track gate is not met."
        ),
    }


def model_ladder_registry() -> pd.DataFrame:
    """Explicitly distinguish implemented fits from scientifically premature architectures."""

    return pd.DataFrame(
        [
            {"domain": "pk", "rung": 1, "model": "RDKit/Morgan RF-ET-SVR-XGB-LGBM", "status": "implemented"},
            {
                "domain": "pk",
                "rung": 2,
                "model": "grouped hierarchical endpoint models",
                "status": "implemented_compound_balanced_partial_pooling_discovery_evaluation_only",
            },
            {
                "domain": "pk",
                "rung": 3,
                "model": "state/conformer ensemble ablation",
                "status": "implemented",
            },
            {
                "domain": "pk",
                "rung": 4,
                "model": "MD/membrane observables",
                "status": "architected_waiting_for_hpc_observables",
            },
            {
                "domain": "pk",
                "rung": 5,
                "model": "gray-box rat IV/PO",
                "status": "implemented_nonidentifiability_contract",
            },
            {
                "domain": "pk",
                "rung": 6,
                "model": "PBPK sensitivity",
                "status": "hpc_bundle_only_not_calibrated",
            },
            {
                "domain": "pk",
                "rung": 7,
                "model": "neural ODE profiles",
                "status": "blocked_until_raw_profiles",
            },
            {
                "domain": "herg",
                "rung": 1,
                "model": "preserved same-series/scaffold baselines",
                "status": "preserved",
            },
            {
                "domain": "herg",
                "rung": 2,
                "model": "Sun SVC/SVR reproduction",
                "status": "partial_missing_atom_typing",
            },
            {"domain": "herg", "rung": 3, "model": "continuous censored pIC50", "status": "implemented"},
            {"domain": "herg", "rung": 4, "model": "RF/SVR/boosting", "status": "implemented_exact_subset"},
            {
                "domain": "herg",
                "rung": 4,
                "model": "D-MPNN",
                "status": "architected_not_decision_fit_small_domain",
            },
            {
                "domain": "herg",
                "rung": 4,
                "model": "state-aware multiple instance",
                "status": "implemented_ensemble_aggregation",
            },
            {
                "domain": "herg",
                "rung": 5,
                "model": "receptor/membrane physics residual",
                "status": "architected_waiting_for_hpc_observables",
            },
            {
                "domain": "herg",
                "rung": 6,
                "model": "state-dependent Markov trapping",
                "status": "architected_not_fit_missing_kinetics",
            },
        ]
    )
