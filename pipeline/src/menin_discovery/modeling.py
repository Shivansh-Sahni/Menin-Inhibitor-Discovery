"""Reproducible Menin activity and hERG liability modeling.

The module intentionally favors compact, defensible scikit-learn baselines over
large hyperparameter searches.  Candidate selection occurs within the training
partition; the untouched holdout is used once for model comparison and the
reported uncertainty/calibration diagnostics.
"""

from __future__ import annotations

import hashlib
import json
import platform
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from .features import SmilesFeatureTransformer, nearest_neighbor_tanimoto, scaffold_key
from .settings import ROOT
from .splitting import SplitResult, extract_year, make_cv_folds, make_split

MODEL_SCHEMA_VERSION = "2.0"


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _data_sha256(data: pd.DataFrame, columns: list[str]) -> str:
    available = [column for column in columns if column in data.columns]
    stable = data[available].fillna("").astype(str).sort_values(available, kind="mergesort")
    return hashlib.sha256(stable.to_csv(index=False, lineterminator="\n").encode("utf-8")).hexdigest()


def _environment_metadata() -> dict[str, str]:
    metadata = {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
    }
    try:
        import rdkit

        metadata["rdkit"] = rdkit.__version__
    except ImportError:
        metadata["rdkit"] = "not_installed"
    try:
        import skops

        metadata["skops"] = skops.__version__
    except ImportError:
        metadata["skops"] = "not_installed"
    return metadata


def _serialize_model(
    estimator: BaseEstimator,
    models_dir: Path,
    stem: str,
    *,
    artifact_build_root: Path | None = None,
    artifact_release_root: Path | None = None,
    artifact_project_root: Path | None = None,
) -> dict[str, Any]:
    """Prefer skops; fall back to joblib with an explicit trust warning."""

    models_dir.mkdir(parents=True, exist_ok=True)
    skops_error: str | None = None
    try:
        import skops.io as sio

        path = models_dir / f"{stem}.skops"
        sio.dump(estimator, path)
        serialization = "skops"
        trust = "Load only after reviewing skops.io.get_untrusted_types for this project-defined transformer."
    except (ImportError, OSError, TypeError, ValueError) as exc:
        skops_error = f"{type(exc).__name__}: {exc}"
        path = models_dir / f"{stem}.joblib"
        joblib.dump(estimator, path, compress=3)
        serialization = "joblib"
        trust = "Unsafe for untrusted files; load only artifacts produced by this controlled pipeline."
    display_path = path.resolve()
    if artifact_build_root is not None and artifact_release_root is not None:
        relative_artifact = path.resolve().relative_to(artifact_build_root.resolve())
        display_path = (artifact_release_root / relative_artifact).resolve()
    project_root = (artifact_project_root or ROOT).resolve()
    try:
        portable_path = display_path.relative_to(project_root).as_posix()
        is_repository_relative = True
    except ValueError:
        portable_path = path.name
        is_repository_relative = False
    return {
        "path": portable_path,
        "path_is_repository_relative": is_repository_relative,
        "filename": path.name,
        "format": serialization,
        "sha256": _file_sha256(path),
        "trust_boundary": trust,
        "skops_fallback_reason": skops_error,
    }


def _safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return np.nan
    return float(np.corrcoef(left, right)[0, 1])


def _regression_metrics(y_true: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    errors = y_true - prediction
    ranked_true = pd.Series(y_true).rank(method="average").to_numpy()
    ranked_prediction = pd.Series(prediction).rank(method="average").to_numpy()
    prediction_variance = float(np.var(prediction))
    if prediction_variance > 1e-12 and len(prediction) >= 2:
        calibration_slope = float(
            np.mean((prediction - np.mean(prediction)) * (y_true - np.mean(y_true))) / prediction_variance
        )
        calibration_intercept = float(np.mean(y_true) - calibration_slope * np.mean(prediction))
    else:
        calibration_slope, calibration_intercept = np.nan, np.nan
    return {
        "mae": float(mean_absolute_error(y_true, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, prediction))),
        "median_absolute_error": float(median_absolute_error(y_true, prediction)),
        "r2": float(r2_score(y_true, prediction)) if len(y_true) >= 2 else np.nan,
        "pearson_r": _safe_correlation(y_true, prediction),
        "spearman_r": _safe_correlation(ranked_true, ranked_prediction),
        "mean_signed_error": float(np.mean(errors)),
        "fraction_within_0p5_log_unit": float(np.mean(np.abs(errors) <= 0.5)),
        "fraction_within_1_log_unit": float(np.mean(np.abs(errors) <= 1.0)),
        "calibration_slope": float(calibration_slope),
        "calibration_intercept": float(calibration_intercept),
    }


def _expected_calibration_error(y_true: np.ndarray, probability: np.ndarray, n_bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.clip(np.digitize(probability, edges[1:-1], right=True), 0, n_bins - 1)
    result = 0.0
    for bin_id in range(n_bins):
        mask = bin_ids == bin_id
        if np.any(mask):
            result += float(np.mean(mask)) * abs(
                float(np.mean(y_true[mask])) - float(np.mean(probability[mask]))
            )
    return float(result)


def _classification_metrics(
    y_true: np.ndarray,
    probability: np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=int)
    probability = np.clip(np.asarray(probability, dtype=float), 1e-7, 1 - 1e-7)
    prediction = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    both_classes = len(np.unique(y_true)) == 2
    return {
        "roc_auc": float(roc_auc_score(y_true, probability)) if both_classes else np.nan,
        "pr_auc": float(average_precision_score(y_true, probability)) if both_classes else np.nan,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "accuracy": float(accuracy_score(y_true, prediction)),
        "sensitivity": float(recall_score(y_true, prediction, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else np.nan,
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "matthews_correlation": float(matthews_corrcoef(y_true, prediction)),
        "brier_score": float(brier_score_loss(y_true, probability)),
        "log_loss": float(log_loss(y_true, probability, labels=[0, 1])),
        "expected_calibration_error_10bin": _expected_calibration_error(y_true, probability),
        "threshold": float(threshold),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def _bootstrap_confidence_intervals(
    y_true: np.ndarray,
    prediction: np.ndarray,
    *,
    task_type: str,
    iterations: int,
    random_state: int,
    groups: np.ndarray | None = None,
) -> dict[str, Any]:
    """Paired row- or chemical-group bootstrap intervals for holdout metrics."""

    if iterations <= 0 or len(y_true) < 3:
        return {}
    rng = np.random.default_rng(random_state)
    metric_names = (
        ("mae", "rmse", "r2", "spearman_r")
        if task_type == "regression"
        else ("roc_auc", "pr_auc", "balanced_accuracy", "brier_score", "f1")
    )
    samples: dict[str, list[float]] = {name: [] for name in metric_names}
    group_values = np.asarray(groups, dtype=object) if groups is not None else None
    unique_groups = np.unique(group_values) if group_values is not None else np.array([])
    for _ in range(iterations):
        if group_values is not None and len(unique_groups) >= 2:
            sampled_groups = rng.choice(
                unique_groups,
                size=len(unique_groups),
                replace=True,
            )
            indices = np.concatenate([np.flatnonzero(group_values == group) for group in sampled_groups])
        else:
            indices = rng.integers(0, len(y_true), size=len(y_true))
        sample_y = np.asarray(y_true)[indices]
        sample_prediction = np.asarray(prediction)[indices]
        if task_type == "classification" and len(np.unique(sample_y)) < 2:
            continue
        metric_values = (
            _regression_metrics(sample_y, sample_prediction)
            if task_type == "regression"
            else _classification_metrics(sample_y, sample_prediction)
        )
        for name in metric_names:
            value = metric_values.get(name, np.nan)
            if value is not None and np.isfinite(value):
                samples[name].append(float(value))
    intervals: dict[str, Any] = {}
    for name, sample_values in samples.items():
        if sample_values:
            intervals[name] = {
                "lower_95": float(np.quantile(sample_values, 0.025)),
                "upper_95": float(np.quantile(sample_values, 0.975)),
                "n_successful_resamples": int(len(sample_values)),
            }
    intervals["resampling_unit"] = (
        "chemical_scaffold_group" if group_values is not None and len(unique_groups) >= 2 else "holdout_row"
    )
    intervals["n_resampling_groups"] = int(len(unique_groups)) if groups is not None else int(len(y_true))
    return intervals


def _scaffold_bootstrap_groups(data: pd.DataFrame) -> np.ndarray | None:
    if "smiles" not in data.columns or len(data) < 3:
        return None
    return data["smiles"].map(lambda value: scaffold_key(value)[0]).to_numpy(dtype=object)


def _feature_pipeline(
    estimator: BaseEstimator,
    *,
    feature_backend: str,
    feature_n_bits: int,
    feature_radius: int,
) -> Pipeline:
    return Pipeline(
        [
            (
                "features",
                SmilesFeatureTransformer(
                    n_features=feature_n_bits,
                    backend=feature_backend,
                    radius=feature_radius,
                ),
            ),
            ("model", estimator),
        ]
    )


def _regression_candidates(
    *,
    random_state: int,
    feature_backend: str,
    feature_n_bits: int,
    feature_radius: int,
    tree_estimators: int,
) -> dict[str, Pipeline]:
    return {
        "dummy_median": _feature_pipeline(
            DummyRegressor(strategy="median"),
            feature_backend=feature_backend,
            feature_n_bits=feature_n_bits,
            feature_radius=feature_radius,
        ),
        "ridge_alpha_1": _feature_pipeline(
            Ridge(alpha=1.0),
            feature_backend=feature_backend,
            feature_n_bits=feature_n_bits,
            feature_radius=feature_radius,
        ),
        "ridge_alpha_10": _feature_pipeline(
            Ridge(alpha=10.0),
            feature_backend=feature_backend,
            feature_n_bits=feature_n_bits,
            feature_radius=feature_radius,
        ),
        "extra_trees": _feature_pipeline(
            ExtraTreesRegressor(
                n_estimators=tree_estimators,
                min_samples_leaf=2,
                max_features="sqrt",
                n_jobs=1,
                random_state=random_state,
            ),
            feature_backend=feature_backend,
            feature_n_bits=feature_n_bits,
            feature_radius=feature_radius,
        ),
    }


def _classification_candidates(
    *,
    random_state: int,
    feature_backend: str,
    feature_n_bits: int,
    feature_radius: int,
    tree_estimators: int,
) -> dict[str, Pipeline]:
    return {
        "dummy_prior": _feature_pipeline(
            DummyClassifier(strategy="prior"),
            feature_backend=feature_backend,
            feature_n_bits=feature_n_bits,
            feature_radius=feature_radius,
        ),
        "logistic_c0p3": _feature_pipeline(
            LogisticRegression(
                C=0.3,
                max_iter=3000,
                class_weight="balanced",
                solver="liblinear",
                random_state=random_state,
            ),
            feature_backend=feature_backend,
            feature_n_bits=feature_n_bits,
            feature_radius=feature_radius,
        ),
        "logistic_c1": _feature_pipeline(
            LogisticRegression(
                C=1.0,
                max_iter=3000,
                class_weight="balanced",
                solver="liblinear",
                random_state=random_state,
            ),
            feature_backend=feature_backend,
            feature_n_bits=feature_n_bits,
            feature_radius=feature_radius,
        ),
        "extra_trees": _feature_pipeline(
            ExtraTreesClassifier(
                n_estimators=tree_estimators,
                min_samples_leaf=2,
                max_features="sqrt",
                class_weight="balanced",
                n_jobs=1,
                random_state=random_state,
            ),
            feature_backend=feature_backend,
            feature_n_bits=feature_n_bits,
            feature_radius=feature_radius,
        ),
    }


def _select_and_compare_candidates(
    candidates: dict[str, Pipeline],
    X_train: pd.Series,
    y_train: np.ndarray,
    X_test: pd.Series,
    y_test: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    *,
    task_type: str,
) -> tuple[str, Pipeline, pd.DataFrame]:
    """Select exclusively by training CV, then evaluate only that choice on holdout."""

    scoring = "neg_mean_absolute_error" if task_type == "regression" else "average_precision"
    rows: list[dict[str, Any]] = []
    selected_name = ""
    selected_score = -np.inf
    for name, template in candidates.items():
        scores = cross_val_score(
            template,
            X_train,
            y_train,
            cv=folds,
            scoring=scoring,
            n_jobs=1,
            error_score=np.nan,
        )
        finite_scores = scores[np.isfinite(scores)]
        mean_score = float(np.mean(finite_scores)) if len(finite_scores) else -np.inf
        row: dict[str, Any] = {
            "candidate": name,
            "selection_metric": "cv_mae" if task_type == "regression" else "cv_pr_auc",
            "cv_mean": float(-mean_score) if task_type == "regression" else mean_score,
            "cv_std": float(np.std(finite_scores)) if len(finite_scores) else np.nan,
            "n_successful_folds": int(len(finite_scores)),
        }
        rows.append(row)
        if mean_score > selected_score:
            selected_score = mean_score
            selected_name = name
    if not selected_name:
        raise RuntimeError("All candidate models failed cross-validation")
    fitted_selected = clone(candidates[selected_name]).fit(X_train, y_train)
    if task_type == "regression":
        holdout_prediction = fitted_selected.predict(X_test)
        holdout = _regression_metrics(y_test, holdout_prediction)
    else:
        holdout_prediction = fitted_selected.predict_proba(X_test)[:, 1]
        holdout = _classification_metrics(y_test, holdout_prediction)
    comparison = pd.DataFrame(rows)
    comparison["selected_from_training_cv"] = comparison["candidate"] == selected_name
    for key, value in holdout.items():
        if not isinstance(value, dict):
            comparison.loc[comparison["selected_from_training_cv"], f"holdout_{key}"] = value
    return selected_name, fitted_selected, comparison


def _oof_predictions(
    template: Pipeline,
    X: pd.Series,
    y: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    *,
    probability: bool,
) -> np.ndarray:
    result = np.full(len(y), np.nan, dtype=float)
    for train, validation in folds:
        fitted = clone(template).fit(X.iloc[train], y[train])
        result[validation] = (
            fitted.predict_proba(X.iloc[validation])[:, 1]
            if probability
            else fitted.predict(X.iloc[validation])
        )
    return result


def _conformal_radius(y_true: np.ndarray, oof_prediction: np.ndarray, coverage: float) -> tuple[float, int]:
    mask = np.isfinite(oof_prediction)
    residuals = np.abs(np.asarray(y_true)[mask] - np.asarray(oof_prediction)[mask])
    if not len(residuals):
        return np.nan, 0
    finite_sample_quantile = min(1.0, np.ceil((len(residuals) + 1) * coverage) / len(residuals))
    return float(np.quantile(residuals, finite_sample_quantile, method="higher")), int(len(residuals))


def _applicability_domain(
    train_smiles: pd.Series,
    query_smiles: pd.Series,
    *,
    backend: str,
    n_bits: int,
    radius: int,
    threshold_quantile: float,
    random_state: int,
) -> dict[str, Any]:
    train_values = train_smiles.fillna("").astype(str).tolist()
    query_values = query_smiles.fillna("").astype(str).tolist()
    rng = np.random.default_rng(random_state)
    if len(train_values) > 1000:
        reference_indices = np.sort(rng.choice(len(train_values), size=1000, replace=False))
        domain_reference = [train_values[index] for index in reference_indices]
    else:
        domain_reference = train_values
    if len(domain_reference) >= 2:
        train_neighbor_similarity, _, resolved = nearest_neighbor_tanimoto(
            domain_reference,
            domain_reference,
            backend=backend,
            n_bits=n_bits,
            radius=radius,
            exclude_identical_positions=True,
        )
        positive = train_neighbor_similarity[
            np.isfinite(train_neighbor_similarity) & (train_neighbor_similarity > 0)
        ]
        threshold = float(np.quantile(positive, threshold_quantile)) if len(positive) else 0.0
    else:
        resolved = backend
        threshold = 0.0
    similarities, neighbor_indices, resolved = nearest_neighbor_tanimoto(
        query_values,
        train_values,
        backend=resolved,
        n_bits=n_bits,
        radius=radius,
    )
    nearest_smiles = [train_values[index] if index >= 0 else "" for index in neighbor_indices]
    return {
        "similarity": similarities,
        "neighbor_index": neighbor_indices,
        "nearest_smiles": nearest_smiles,
        "threshold": threshold,
        "inside_domain": similarities >= threshold,
        "fingerprint_backend": resolved,
        "threshold_quantile": float(threshold_quantile),
        "definition": (
            "Inside domain when maximum training-set Tanimoto is at least the configured "
            f"{threshold_quantile:.3f} quantile of sampled training nearest-neighbor similarities."
        ),
    }


def _artifact_prefix(base: str, endpoint: str | None, assay_family: str | None = None) -> str:
    if endpoint is None and assay_family is None:
        return base
    label = "_".join(value for value in (endpoint, assay_family) if value)
    suffix = re.sub(r"[^a-z0-9]+", "_", label.strip().lower()).strip("_") or "task"
    return f"{base}_{suffix}"


def _join_unique(values: pd.Series) -> str:
    return ";".join(sorted({str(value) for value in values.dropna() if str(value).strip()}))


def prepare_menin_task(
    compounds_or_measurements: pd.DataFrame,
    *,
    endpoint: str | None = None,
    assay_family: str | None = None,
    collapse_cross_source_mirrors: bool = True,
    exclude_heterogeneous_labels: bool = False,
    heterogeneity_log_spread_threshold: float = 2.0,
) -> pd.DataFrame:
    """Build a compound-level regression task from measurements or curated rows.

    Measurement-level input (``p_value`` present) is preferred because an
    endpoint filter can be applied before aggregation.  Endpoint filtering of an
    already aggregated table can select relevant compounds but cannot unmix its
    median; that limitation is recorded in ``DataFrame.attrs['task_metadata']``.
    """

    if compounds_or_measurements.empty:
        result = pd.DataFrame()
        result.attrs["task_metadata"] = {
            "endpoint": endpoint,
            "assay_family": assay_family,
            "status": "empty",
        }
        return result
    data = compounds_or_measurements.copy()
    metadata: dict[str, Any] = {"endpoint": endpoint, "assay_family": assay_family}
    if "p_value" in data.columns:
        metadata["input_level"] = "measurement"
        metadata["input_rows"] = int(len(data))
        if "is_modeling_eligible" in data.columns:
            eligible = (
                data["is_modeling_eligible"].astype(str).str.strip().str.lower().isin({"true", "1", "yes"})
            )
            metadata["ineligible_rows_excluded"] = int((~eligible).sum())
            data = data[eligible].copy()
        exact = data.get("is_exact", pd.Series(True, index=data.index))
        exact_mask = exact.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})
        data = data[exact_mask].copy()
        if "is_core_endpoint" in data.columns:
            core = data["is_core_endpoint"].astype(str).str.strip().str.lower().isin({"true", "1", "yes"})
            data = data[core].copy()
        if endpoint is not None and "endpoint" in data.columns:
            data = data[data["endpoint"].astype(str).str.casefold() == endpoint.casefold()].copy()
        if assay_family is not None:
            if "assay_family" not in data.columns:
                raise KeyError("assay_family was requested but is absent from the measurement table")
            data = data[data["assay_family"].astype(str).str.casefold() == assay_family.casefold()].copy()
        data["p_value"] = pd.to_numeric(data["p_value"], errors="coerce")
        data = data.dropna(subset=["smiles", "p_value"])
        data = data[data["smiles"].astype(str).str.strip().ne("")]
        data["_year"] = data.get("document_year", pd.Series(np.nan, index=data.index)).map(extract_year)
        identity = next(
            (
                column
                for column in (
                    "structure_id",
                    "standard_inchi_key",
                    "standardized_smiles",
                    "canonical_smiles",
                    "smiles",
                )
                if column in data.columns
            ),
            "smiles",
        )
        provenance_aggregations: dict[str, tuple[str, Any]] = {
            "n_source_rows": ("p_value", "size"),
        }
        if "source" in data.columns:
            provenance_aggregations.update(
                {
                    "sources": ("source", _join_unique),
                    "n_sources": ("source", "nunique"),
                }
            )
        provenance = data.groupby(identity, as_index=False).agg(**provenance_aggregations)
        redundant = (
            data["is_cross_source_mirror_redundant"]
            .astype(str)
            .str.strip()
            .str.casefold()
            .isin({"true", "1", "yes"})
            if "is_cross_source_mirror_redundant" in data.columns
            else pd.Series(False, index=data.index)
        )
        if not collapse_cross_source_mirrors:
            redundant = pd.Series(False, index=data.index)
        metadata["cross_source_mirror_rows_collapsed"] = int(redundant.sum())
        metadata["collapse_cross_source_mirrors"] = bool(collapse_cross_source_mirrors)
        data = data.loc[~redundant].copy()
        label_spread = data.groupby(identity, dropna=False)["p_value"].transform(
            lambda values: values.max() - values.min()
        )
        heterogeneous = label_spread.gt(heterogeneity_log_spread_threshold)
        metadata["heterogeneous_source_rows"] = int(heterogeneous.sum())
        metadata["heterogeneous_structures"] = int(data.loc[heterogeneous, identity].nunique())
        metadata["heterogeneity_log_spread_threshold"] = float(heterogeneity_log_spread_threshold)
        metadata["exclude_heterogeneous_labels"] = bool(exclude_heterogeneous_labels)
        if exclude_heterogeneous_labels:
            data = data.loc[~heterogeneous].copy()
        metadata["replicate_weighting_policy"] = (
            "same-source replicates retained; lower-priority exact normalized cross-source "
            "mirror rows excluded from the label but retained in measurement provenance"
        )
        aggregations: dict[str, tuple[str, Any]] = {
            "p_activity_median": ("p_value", "median"),
            "n_measurements": ("p_value", "size"),
            "document_year": ("_year", "min"),
        }
        if identity != "smiles":
            aggregations["smiles"] = ("smiles", "first")
        if "endpoint" in data.columns:
            aggregations["endpoints"] = ("endpoint", _join_unique)
        if "assay_family" in data.columns:
            aggregations["assay_families"] = ("assay_family", _join_unique)
        if "date_provenance" in data.columns:
            aggregations["date_provenance"] = ("date_provenance", _join_unique)
        result = data.groupby(identity, as_index=False).agg(**aggregations)
        result = result.merge(provenance, on=identity, how="left", validate="one_to_one")
        metadata["identity_column"] = identity
        metadata["label_policy"] = (
            "median exact eligible normalized p-value after endpoint/assay filtering, "
            "retaining same-source replicates and excluding linked lower-priority "
            "cross-source mirrors before standardized-structure aggregation"
        )
    else:
        metadata["input_level"] = "compound"
        result = data.copy()
        if endpoint is None:
            metadata["label_policy"] = "existing compound-level p_activity_median"
        if endpoint is not None:
            endpoint_column = (
                "endpoint"
                if "endpoint" in result.columns
                else "endpoints"
                if "endpoints" in result.columns
                else None
            )
            if endpoint_column:
                mask = (
                    result[endpoint_column]
                    .fillna("")
                    .astype(str)
                    .str.split(";")
                    .map(
                        lambda values: any(
                            value.strip().casefold() == endpoint.casefold() for value in values
                        )
                    )
                )
                result = result[mask].copy()
            metadata["label_policy"] = "filtered aggregated table; median may contain other endpoints"
            metadata["aggregated_endpoint_mixing_possible"] = True
        if assay_family is not None:
            family_column = (
                "assay_family"
                if "assay_family" in result.columns
                else "assay_families"
                if "assay_families" in result.columns
                else None
            )
            if family_column:
                mask = (
                    result[family_column]
                    .fillna("")
                    .astype(str)
                    .str.split(";")
                    .map(
                        lambda values: any(
                            value.strip().casefold() == assay_family.casefold() for value in values
                        )
                    )
                )
                result = result[mask].copy()
            metadata["aggregated_assay_mixing_possible"] = True
    if "p_activity_median" in result.columns:
        result["p_activity_median"] = pd.to_numeric(result["p_activity_median"], errors="coerce")
        result = result.dropna(subset=["smiles", "p_activity_median"])
        result = result[result["smiles"].astype(str).str.strip().ne("")].reset_index(drop=True)
    metadata["n_compounds"] = int(len(result))
    result.attrs["task_metadata"] = metadata
    return result


def prepare_herg_task(
    compounds: pd.DataFrame,
    *,
    endpoint: str | None = None,
    assay_family: str | None = None,
) -> pd.DataFrame:
    """Collapse endpoint-stratified hERG rows to one unambiguous label per structure.

    A structure measured in multiple assay families is one statistical unit. If
    its threshold-derived binary labels disagree, the structure is excluded and
    counted in task metadata rather than duplicated across model folds.
    """

    if compounds.empty:
        result = pd.DataFrame()
        result.attrs["task_metadata"] = {"status": "empty"}
        return result
    data = compounds.copy()
    input_rows = int(len(data))
    if endpoint is not None:
        endpoint_column = "endpoint" if "endpoint" in data else "endpoints"
        data = data[data[endpoint_column].fillna("").astype(str).str.casefold() == endpoint.casefold()].copy()
    if assay_family is not None:
        family_column = "assay_family" if "assay_family" in data else "assay_families"
        data = data[
            data[family_column].fillna("").astype(str).str.casefold() == assay_family.casefold()
        ].copy()
    data["herg_blocker_label"] = pd.to_numeric(data.get("herg_blocker_label"), errors="coerce")
    data = data.dropna(subset=["smiles", "herg_blocker_label"])
    data = data[data["smiles"].astype(str).str.strip().ne("")]
    data = data[data["herg_blocker_label"].isin([0, 1])].copy()
    label_policy = _join_unique(data["herg_label_policy"]) if "herg_label_policy" in data.columns else ""
    identity = next(
        (
            column
            for column in (
                "structure_id",
                "standard_inchi_key",
                "standardized_smiles",
                "canonical_smiles",
                "smiles",
            )
            if column in data.columns
        ),
        "smiles",
    )
    label_counts = data.groupby(identity, dropna=False)["herg_blocker_label"].nunique()
    conflicting = set(label_counts[label_counts > 1].index)
    resolved = data[~data[identity].isin(conflicting)].copy()
    if resolved.empty:
        result = pd.DataFrame()
    else:
        aggregations: dict[str, tuple[str, Any]] = {
            "herg_blocker_label": ("herg_blocker_label", "first"),
            "n_task_rows": ("herg_blocker_label", "size"),
        }
        if identity != "smiles":
            aggregations["smiles"] = ("smiles", "first")
        for source, output in (
            ("endpoint", "endpoints"),
            ("endpoints", "endpoints"),
            ("assay_family", "assay_families"),
            ("assay_families", "assay_families"),
            ("document_years", "document_years"),
        ):
            if source in resolved.columns and output not in aggregations:
                aggregations[output] = (source, _join_unique)
        if "document_year" in resolved.columns:
            resolved["_document_year"] = resolved["document_year"].map(extract_year)
            aggregations["document_year"] = ("_document_year", "min")
        elif "document_years" in resolved.columns:
            resolved["_document_year"] = resolved["document_years"].map(extract_year)
            aggregations["document_year"] = ("_document_year", "min")
        for column in (
            "standard_inchi_key",
            "standardized_smiles",
            "canonical_smiles",
        ):
            if column in resolved.columns and column != identity:
                aggregations[column] = (column, "first")
        result = resolved.groupby(identity, dropna=False).agg(**aggregations).reset_index()
    result.attrs["task_metadata"] = {
        "input_rows": input_rows,
        "labeled_rows_after_endpoint_assay_filter": int(len(data)),
        "endpoint": endpoint,
        "assay_family": assay_family,
        "n_structures": int(label_counts.size),
        "n_conflicting_structures_excluded": int(len(conflicting)),
        "n_resolved_structures": int(len(result)),
        "identity_column": identity,
        "conflict_policy": "exclude structures with both blocker and non-blocker labels",
        "label_policy": label_policy,
    }
    return result


def _save_split_assignments(
    data: pd.DataFrame,
    split: SplitResult,
    reports_dir: Path,
    prefix: str,
    target_column: str,
) -> None:
    columns = [
        column
        for column in (
            "structure_id",
            "standard_inchi_key",
            "smiles",
            target_column,
            "endpoint",
            "endpoints",
            "assay_family",
            "assay_families",
            "sources",
            "document_year",
            "document_years",
        )
        if column in data.columns
    ]
    output = data[columns].copy().reset_index(drop=True)
    output.insert(0, "modeling_row", np.arange(len(output)))
    structure_group = (
        data["structure_id"].fillna("").astype(str)
        if "structure_id" in data.columns
        else data["smiles"].fillna("").astype(str)
    )
    output["structure_group_key"] = structure_group.to_numpy()
    scaffold = data["smiles"].map(scaffold_key)
    output["bemis_murcko_group"] = scaffold.map(lambda item: item[0]).to_numpy()
    output["scaffold_grouping_method"] = scaffold.map(lambda item: item[1]).to_numpy()
    output["requested_split_strategy"] = split.metadata.get("requested_strategy")
    output["actual_split_strategy"] = split.metadata.get("strategy")
    output["split_sha256"] = split.metadata.get("split_sha256")
    output["dataset_sha256"] = _data_sha256(data, columns)
    output["split"] = split.assignments(len(data))
    output.to_csv(reports_dir / f"{prefix}_split_assignments.csv", index=False)


def _prediction_context(data: pd.DataFrame) -> pd.DataFrame:
    columns = [
        column
        for column in (
            "structure_id",
            "standard_inchi_key",
            "smiles",
            "sources",
            "endpoint",
            "endpoints",
            "assay_family",
            "assay_families",
            "document_year",
            "document_years",
        )
        if column in data.columns
    ]
    return data[columns].reset_index(drop=True).copy()


def _model_manifest(
    *,
    artifact: dict[str, Any],
    estimator: BaseEstimator,
    data: pd.DataFrame,
    data_columns: list[str],
    split: SplitResult,
    cv_metadata: dict[str, Any],
    feature_metadata: dict[str, Any],
    selection_metric: str,
    task_metadata: dict[str, Any],
    provenance_context: dict[str, Any] | None,
) -> dict[str, Any]:
    hashed_columns = [column for column in data_columns if column in data.columns]
    return {
        "schema_version": MODEL_SCHEMA_VERSION,
        "artifact": artifact,
        "estimator_class": f"{type(estimator).__module__}.{type(estimator).__name__}",
        "dataset_sha256": _data_sha256(data, hashed_columns),
        "dataset_columns_hashed": hashed_columns,
        "dataset_columns_requested": data_columns,
        "split": split.metadata,
        "cross_validation": cv_metadata,
        "features": feature_metadata,
        "task": task_metadata,
        "provenance": provenance_context or {},
        "selection_metric": selection_metric,
        "environment": _environment_metadata(),
        "reproduction_requirements": [
            "immutable input snapshot matching dataset_sha256",
            "split and CV hashes in this manifest",
            "the recorded Python/package environment",
            "the exact source revision that generated this artifact",
        ],
    }


def train_menin_activity_model(
    compounds: pd.DataFrame,
    models_dir: Path,
    reports_dir: Path,
    *,
    random_state: int = 13,
    split_strategy: str = "scaffold",
    test_size: float = 0.2,
    endpoint: str | None = None,
    assay_family: str | None = None,
    time_column: str | None = None,
    feature_backend: str = "auto",
    feature_n_bits: int = 2048,
    feature_radius: int = 2,
    applicability_domain_quantile: float = 0.05,
    cv_folds: int = 3,
    bootstrap_iterations: int = 500,
    prediction_interval_coverage: float = 0.90,
    tree_estimators: int = 200,
    min_samples: int = 40,
    provenance_context: dict[str, Any] | None = None,
    artifact_build_root: Path | None = None,
    artifact_release_root: Path | None = None,
    artifact_project_root: Path | None = None,
    collapse_cross_source_mirrors: bool = True,
    exclude_heterogeneous_labels: bool = False,
    heterogeneity_log_spread_threshold: float = 2.0,
) -> dict[str, Any]:
    """Train endpoint-aware activity baselines with chemical holdout evaluation."""

    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    prefix = _artifact_prefix("menin_activity", endpoint, assay_family)
    data = prepare_menin_task(
        compounds,
        endpoint=endpoint,
        assay_family=assay_family,
        collapse_cross_source_mirrors=collapse_cross_source_mirrors,
        exclude_heterogeneous_labels=exclude_heterogeneous_labels,
        heterogeneity_log_spread_threshold=heterogeneity_log_spread_threshold,
    )
    task_metadata = dict(data.attrs.get("task_metadata", {}))

    if len(data) < min_samples:
        insufficient_metrics = {
            "schema_version": MODEL_SCHEMA_VERSION,
            "status": "insufficient_data",
            "n_compounds": int(len(data)),
            "minimum_required": int(min_samples),
            "task": task_metadata,
            "provenance": provenance_context or {},
        }
        _write_json(reports_dir / f"{prefix}_model_metrics.json", insufficient_metrics)
        return insufficient_metrics

    split = make_split(
        data,
        strategy=split_strategy,
        test_size=test_size,
        random_state=random_state,
        target_column="p_activity_median",
        task_type="regression",
        time_column=time_column,
    )
    train = data.iloc[split.train_indices].reset_index(drop=True)
    test = data.iloc[split.test_indices].reset_index(drop=True)
    X_train = train["smiles"].astype(str)
    X_test = test["smiles"].astype(str)
    y_train = train["p_activity_median"].to_numpy(dtype=float)
    y_test = test["p_activity_median"].to_numpy(dtype=float)
    folds, cv_metadata = make_cv_folds(
        train,
        strategy=split.metadata["strategy"],
        n_splits=cv_folds,
        random_state=random_state,
        target_column="p_activity_median",
        task_type="regression",
        time_column=time_column,
    )
    candidates = _regression_candidates(
        random_state=random_state,
        feature_backend=feature_backend,
        feature_n_bits=feature_n_bits,
        feature_radius=feature_radius,
        tree_estimators=tree_estimators,
    )
    selected_name, selected, comparison = _select_and_compare_candidates(
        candidates, X_train, y_train, X_test, y_test, folds, task_type="regression"
    )
    comparison.to_csv(reports_dir / f"{prefix}_model_comparison.csv", index=False)

    prediction = selected.predict(X_test)
    test_metrics = _regression_metrics(y_test, prediction)
    confidence_intervals = _bootstrap_confidence_intervals(
        y_test,
        prediction,
        task_type="regression",
        iterations=bootstrap_iterations,
        random_state=random_state + 101,
        groups=_scaffold_bootstrap_groups(test),
    )
    selected_template = candidates[selected_name]
    oof = _oof_predictions(selected_template, X_train, y_train, folds, probability=False)
    conformal_radius, n_oof = _conformal_radius(y_train, oof, prediction_interval_coverage)
    interval_lower = prediction - conformal_radius
    interval_upper = prediction + conformal_radius
    interval_coverage = (
        float(np.mean((y_test >= interval_lower) & (y_test <= interval_upper)))
        if np.isfinite(conformal_radius)
        else np.nan
    )

    resolved_backend = selected.named_steps["features"].backend_
    domain = _applicability_domain(
        X_train,
        X_test,
        backend=resolved_backend,
        n_bits=feature_n_bits,
        radius=feature_radius,
        threshold_quantile=applicability_domain_quantile,
        random_state=random_state,
    )
    prediction_table = pd.concat(
        [
            _prediction_context(test),
            pd.DataFrame(
                {
                    "observed_p_activity_median": y_test,
                    "predicted_p_activity_median": prediction,
                    "residual_observed_minus_predicted": y_test - prediction,
                    "absolute_error": np.abs(y_test - prediction),
                    f"prediction_interval_lower_{prediction_interval_coverage:.2f}": interval_lower,
                    f"prediction_interval_upper_{prediction_interval_coverage:.2f}": interval_upper,
                    "max_training_tanimoto": domain["similarity"],
                    "nearest_training_smiles": domain["nearest_smiles"],
                    "inside_applicability_domain": domain["inside_domain"],
                }
            ),
        ],
        axis=1,
    ).sort_values("absolute_error", ascending=False)
    prediction_table.to_csv(reports_dir / f"{prefix}_model_test_predictions.csv", index=False)
    _save_split_assignments(data, split, reports_dir, prefix, "p_activity_median")

    artifact = _serialize_model(
        selected,
        models_dir,
        f"{prefix}_{selected_name}",
        artifact_build_root=artifact_build_root,
        artifact_release_root=artifact_release_root,
        artifact_project_root=artifact_project_root,
    )
    feature_metadata = dict(selected.named_steps["features"].feature_metadata_)
    manifest = _model_manifest(
        artifact=artifact,
        estimator=selected,
        data=data,
        data_columns=[
            "smiles",
            "p_activity_median",
            "endpoint",
            "endpoints",
            "assay_family",
            "assay_families",
            "document_year",
            "document_years",
        ],
        split=split,
        cv_metadata=cv_metadata,
        feature_metadata=feature_metadata,
        selection_metric="training CV mean absolute error",
        task_metadata=task_metadata,
        provenance_context=provenance_context,
    )
    _write_json(models_dir / f"{prefix}_manifest.json", manifest)

    fitted_model = selected.named_steps["model"]
    metrics: dict[str, Any] = {
        "schema_version": MODEL_SCHEMA_VERSION,
        "status": "trained",
        "task": task_metadata,
        "model": selected_name,
        "n_compounds": int(len(data)),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "split": split.metadata,
        "cross_validation": cv_metadata,
        "features": feature_metadata,
        "test_metrics": test_metrics,
        "test_metric_bootstrap_95_ci": confidence_intervals,
        "uncertainty": {
            "method": "cross-validated absolute-residual conformal interval",
            "nominal_coverage": float(prediction_interval_coverage),
            "empirical_holdout_coverage": interval_coverage,
            "interval_half_width_p_activity": conformal_radius,
            "n_oof_residuals": n_oof,
        },
        "applicability_domain": {
            "fingerprint_backend": domain["fingerprint_backend"],
            "similarity_threshold": domain["threshold"],
            "holdout_fraction_inside_domain": float(np.mean(domain["inside_domain"])),
            "definition": domain["definition"],
        },
        "artifact": artifact,
        "provenance": provenance_context or {},
        # Backward-compatible flat keys.
        "test_mae_pchembl": test_metrics["mae"],
        "test_rmse_pchembl": test_metrics["rmse"],
        "test_r2": test_metrics["r2"],
        "interpretation": "Use with scaffold/temporal validation, uncertainty intervals, and applicability-domain flags; prospective experimental validation remains required.",
    }
    if isinstance(fitted_model, Ridge):
        metrics["best_alpha"] = float(fitted_model.alpha)
    _write_json(reports_dir / f"{prefix}_model_metrics.json", metrics)
    return metrics


def _calibrate_classifier(
    template: Pipeline,
    X_train: pd.Series,
    y_train: np.ndarray,
    *,
    folds: list[tuple[np.ndarray, np.ndarray]],
    cv_metadata: dict[str, Any],
) -> tuple[BaseEstimator, dict[str, Any]]:
    if len(folds) < 2:
        fitted = clone(template).fit(X_train, y_train)
        return fitted, {"method": "none", "reason": "insufficient audited CV folds"}
    try:
        calibrated = CalibratedClassifierCV(
            estimator=clone(template),
            method="sigmoid",
            cv=folds,
            ensemble=False,
        )
        calibrated.fit(X_train, y_train)
        return calibrated, {
            "method": "Platt sigmoid",
            "cv": "same audited group-aware folds used for model selection",
            "n_splits": len(folds),
            "strategy": cv_metadata.get("strategy"),
            "cv_sha256": cv_metadata.get("cv_sha256"),
            "maximum_structure_overlap": cv_metadata.get("maximum_structure_overlap"),
        }
    except (TypeError, ValueError) as exc:
        fitted = clone(template).fit(X_train, y_train)
        return fitted, {"method": "none", "reason": f"{type(exc).__name__}: {exc}"}


def _binary_entropy(probability: np.ndarray) -> np.ndarray:
    probability = np.clip(np.asarray(probability, dtype=float), 1e-12, 1 - 1e-12)
    return -(probability * np.log2(probability) + (1 - probability) * np.log2(1 - probability))


def train_herg_classifier_and_predict(
    herg_compounds: pd.DataFrame,
    menin_compounds: pd.DataFrame,
    models_dir: Path,
    reports_dir: Path,
    *,
    random_state: int = 13,
    split_strategy: str = "scaffold",
    test_size: float = 0.2,
    endpoint: str | None = None,
    assay_family: str | None = None,
    menin_endpoint: str | None = None,
    menin_assay_family: str | None = None,
    time_column: str | None = None,
    feature_backend: str = "auto",
    feature_n_bits: int = 2048,
    feature_radius: int = 2,
    applicability_domain_quantile: float = 0.05,
    cv_folds: int = 3,
    bootstrap_iterations: int = 500,
    tree_estimators: int = 200,
    min_samples: int = 80,
    provenance_context: dict[str, Any] | None = None,
    artifact_build_root: Path | None = None,
    artifact_release_root: Path | None = None,
    artifact_project_root: Path | None = None,
) -> dict[str, Any]:
    """Train calibrated hERG baselines and score Menin compounds with AD flags."""

    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    prefix = "herg_classifier"
    data = prepare_herg_task(herg_compounds, endpoint=endpoint, assay_family=assay_family)
    task_metadata = dict(data.attrs.get("task_metadata", {}))
    if not data.empty:
        data["herg_blocker_label"] = data["herg_blocker_label"].astype(int)
        data = data.reset_index(drop=True)

    if len(data) < min_samples or data["herg_blocker_label"].nunique() < 2:
        insufficient_metrics = {
            "schema_version": MODEL_SCHEMA_VERSION,
            "status": "insufficient_data",
            "n_compounds": int(len(data)),
            "n_classes": int(data["herg_blocker_label"].nunique()) if len(data) else 0,
            "minimum_required": int(min_samples),
            "task": task_metadata,
            "provenance": provenance_context or {},
        }
        _write_json(reports_dir / "herg_classifier_metrics.json", insufficient_metrics)
        return insufficient_metrics

    split = make_split(
        data,
        strategy=split_strategy,
        test_size=test_size,
        random_state=random_state,
        target_column="herg_blocker_label",
        task_type="classification",
        time_column=time_column,
    )
    train = data.iloc[split.train_indices].reset_index(drop=True)
    test = data.iloc[split.test_indices].reset_index(drop=True)
    X_train = train["smiles"].astype(str)
    X_test = test["smiles"].astype(str)
    y_train = train["herg_blocker_label"].to_numpy(dtype=int)
    y_test = test["herg_blocker_label"].to_numpy(dtype=int)
    folds, cv_metadata = make_cv_folds(
        train,
        strategy=split.metadata["strategy"],
        n_splits=cv_folds,
        random_state=random_state,
        target_column="herg_blocker_label",
        task_type="classification",
        time_column=time_column,
    )
    candidates = _classification_candidates(
        random_state=random_state,
        feature_backend=feature_backend,
        feature_n_bits=feature_n_bits,
        feature_radius=feature_radius,
        tree_estimators=tree_estimators,
    )
    selected_name, selected_raw, comparison = _select_and_compare_candidates(
        candidates, X_train, y_train, X_test, y_test, folds, task_type="classification"
    )
    comparison.to_csv(reports_dir / "herg_classifier_model_comparison.csv", index=False)

    raw_probability = selected_raw.predict_proba(X_test)[:, 1]
    predictor, calibration_metadata = _calibrate_classifier(
        candidates[selected_name],
        X_train,
        y_train,
        folds=folds,
        cv_metadata=cv_metadata,
    )
    probability = predictor.predict_proba(X_test)[:, 1]
    prediction = (probability >= 0.5).astype(int)
    test_metrics = _classification_metrics(y_test, probability)
    raw_test_metrics = _classification_metrics(y_test, raw_probability)
    confidence_intervals = _bootstrap_confidence_intervals(
        y_test,
        probability,
        task_type="classification",
        iterations=bootstrap_iterations,
        random_state=random_state + 211,
        groups=_scaffold_bootstrap_groups(test),
    )

    resolved_backend = selected_raw.named_steps["features"].backend_
    domain = _applicability_domain(
        X_train,
        X_test,
        backend=resolved_backend,
        n_bits=feature_n_bits,
        radius=feature_radius,
        threshold_quantile=applicability_domain_quantile,
        random_state=random_state,
    )
    test_table = pd.concat(
        [
            _prediction_context(test),
            pd.DataFrame(
                {
                    "observed_herg_blocker_label": y_test,
                    "predicted_herg_blocker_probability": probability,
                    "predicted_label_0p5": prediction,
                    "probability_entropy_bits": _binary_entropy(probability),
                    "max_training_tanimoto": domain["similarity"],
                    "nearest_training_smiles": domain["nearest_smiles"],
                    "inside_applicability_domain": domain["inside_domain"],
                }
            ),
        ],
        axis=1,
    )
    test_table.to_csv(reports_dir / "herg_classifier_test_predictions.csv", index=False)
    _save_split_assignments(data, split, reports_dir, prefix, "herg_blocker_label")

    fraction_positive, mean_predicted = calibration_curve(
        y_test,
        probability,
        n_bins=min(10, max(3, len(y_test) // 5)),
        strategy="quantile",
    )
    pd.DataFrame(
        {"mean_predicted_probability": mean_predicted, "observed_positive_fraction": fraction_positive}
    ).to_csv(reports_dir / "herg_classifier_calibration_curve.csv", index=False)

    menin = menin_compounds.copy()
    menin["_prediction_input_order"] = np.arange(len(menin))
    prediction_scope = {
        "input_task_rows": int(len(menin)),
        "menin_endpoint": menin_endpoint,
        "menin_assay_family": menin_assay_family,
    }
    if menin_endpoint is not None and "endpoint" in menin.columns:
        menin = menin[
            menin["endpoint"].fillna("").astype(str).str.casefold() == menin_endpoint.casefold()
        ].copy()
    if menin_assay_family is not None and "assay_family" in menin.columns:
        menin = menin[
            menin["assay_family"].fillna("").astype(str).str.casefold() == menin_assay_family.casefold()
        ].copy()
    menin_identity = next(
        (column for column in ("structure_id", "standard_inchi_key", "smiles") if column in menin.columns),
        "smiles",
    )
    if not menin.empty:
        menin = menin.sort_values(
            [menin_identity, "p_activity_median"]
            if "p_activity_median" in menin.columns
            else [menin_identity],
            ascending=[True, False] if "p_activity_median" in menin.columns else True,
            kind="stable",
        ).drop_duplicates(subset=[menin_identity], keep="first")
        menin = menin.sort_values("_prediction_input_order", kind="stable").drop(
            columns=["_prediction_input_order"]
        )
    prediction_scope["unique_structures_scored"] = int(len(menin))
    observed_herg = herg_compounds.copy()
    if endpoint is not None and "endpoint" in observed_herg.columns:
        observed_herg = observed_herg[
            observed_herg["endpoint"].fillna("").astype(str).str.casefold() == endpoint.casefold()
        ]
    if assay_family is not None and "assay_family" in observed_herg.columns:
        observed_herg = observed_herg[
            observed_herg["assay_family"].fillna("").astype(str).str.casefold() == assay_family.casefold()
        ]
    observed_identity = menin_identity if menin_identity in observed_herg.columns else "smiles"
    observed_keys = set(observed_herg.get(observed_identity, pd.Series(dtype=str)).dropna().astype(str))
    if not menin.empty and "smiles" in menin.columns:
        menin["scored_menin_endpoint"] = menin_endpoint or ""
        menin["scored_menin_assay_family"] = menin_assay_family or ""
        menin["has_observed_primary_herg_record"] = (
            menin[menin_identity].fillna("").astype(str).isin(observed_keys)
        )
        menin["predicted_herg_blocker_probability"] = np.nan
        menin["predicted_herg_probability_entropy_bits"] = np.nan
        menin["herg_max_training_tanimoto"] = np.nan
        menin["herg_nearest_training_smiles"] = ""
        menin["herg_inside_applicability_domain"] = False
        valid = menin["smiles"].notna() & menin["smiles"].astype(str).str.strip().ne("")
        if valid.any():
            valid_smiles = menin.loc[valid, "smiles"].astype(str)
            menin_probability = predictor.predict_proba(valid_smiles)[:, 1]
            application_domain = _applicability_domain(
                X_train,
                valid_smiles,
                backend=resolved_backend,
                n_bits=feature_n_bits,
                radius=feature_radius,
                threshold_quantile=applicability_domain_quantile,
                random_state=random_state,
            )
            menin.loc[valid, "predicted_herg_blocker_probability"] = menin_probability
            menin.loc[valid, "predicted_herg_probability_entropy_bits"] = _binary_entropy(menin_probability)
            menin.loc[valid, "herg_max_training_tanimoto"] = application_domain["similarity"]
            menin.loc[valid, "herg_nearest_training_smiles"] = application_domain["nearest_smiles"]
            menin.loc[valid, "herg_inside_applicability_domain"] = application_domain["inside_domain"]
        menin["predicted_herg_risk"] = np.select(
            [
                menin["predicted_herg_blocker_probability"] >= 0.70,
                menin["predicted_herg_blocker_probability"] <= 0.30,
            ],
            ["high", "low"],
            default="medium",
        )
        menin.loc[~valid, "predicted_herg_risk"] = "unscored"
        menin.to_csv(reports_dir / "menin_with_predicted_herg_risk.csv", index=False)

    artifact = _serialize_model(
        predictor,
        models_dir,
        f"herg_liability_{selected_name}_calibrated",
        artifact_build_root=artifact_build_root,
        artifact_release_root=artifact_release_root,
        artifact_project_root=artifact_project_root,
    )
    feature_metadata = dict(selected_raw.named_steps["features"].feature_metadata_)
    manifest = _model_manifest(
        artifact=artifact,
        estimator=predictor,
        data=data,
        data_columns=[
            "structure_id",
            "smiles",
            "herg_blocker_label",
            "endpoint",
            "endpoints",
            "assay_family",
            "assay_families",
            "document_year",
            "document_years",
        ],
        split=split,
        cv_metadata=cv_metadata,
        feature_metadata=feature_metadata,
        selection_metric="training CV precision-recall AUC",
        task_metadata=task_metadata,
        provenance_context=provenance_context,
    )
    manifest["calibration"] = calibration_metadata
    _write_json(models_dir / "herg_classifier_manifest.json", manifest)

    fitted_model = selected_raw.named_steps["model"]
    metrics: dict[str, Any] = {
        "schema_version": MODEL_SCHEMA_VERSION,
        "status": "trained",
        "model": selected_name,
        "n_compounds": int(len(data)),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "class_counts": {
            str(key): int(value)
            for key, value in data["herg_blocker_label"].value_counts().sort_index().items()
        },
        "label_policy": task_metadata.get("label_policy"),
        "positive_label": "blocker class under the recorded label policy",
        "negative_label": "non-blocker class under the recorded label policy",
        "task": task_metadata,
        "split": split.metadata,
        "cross_validation": cv_metadata,
        "features": feature_metadata,
        "calibration": calibration_metadata,
        "uncalibrated_selected_model_test_metrics": raw_test_metrics,
        "test_metrics": test_metrics,
        "test_metric_bootstrap_95_ci": confidence_intervals,
        "applicability_domain": {
            "fingerprint_backend": domain["fingerprint_backend"],
            "similarity_threshold": domain["threshold"],
            "holdout_fraction_inside_domain": float(np.mean(domain["inside_domain"])),
            "definition": domain["definition"],
        },
        "artifact": artifact,
        "provenance": provenance_context or {},
        "menin_prediction_scope": prediction_scope,
        "risk_band_note": "0.30/0.70 risk bands are communication bands, not experimentally optimized decision thresholds.",
        # Backward-compatible flat keys.
        "test_roc_auc": test_metrics["roc_auc"],
        "test_balanced_accuracy": test_metrics["balanced_accuracy"],
        "interpretation": "Calibrated liability screen with explicit domain diagnostics; external and prospective hERG validation is required for decision use.",
    }
    if isinstance(fitted_model, LogisticRegression):
        metrics["best_C"] = float(fitted_model.C)
    _write_json(reports_dir / "herg_classifier_metrics.json", metrics)
    return metrics


def run_models(
    processed_dir: Path,
    models_dir: Path,
    reports_dir: Path,
    *,
    random_state: int = 13,
    split_strategy: str = "scaffold",
    menin_endpoint: str | None = None,
    menin_assay_family: str | None = None,
    time_column: str | None = None,
    feature_backend: str = "auto",
) -> dict[str, dict[str, Any]]:
    """Compatibility entry point requiring an explicit non-pooled Menin task."""

    if not menin_endpoint or not menin_assay_family:
        raise ValueError(
            "run_models requires menin_endpoint and menin_assay_family; use the CLI for "
            "the full endpoint matrix and scoped hERG analysis"
        )

    menin_path = processed_dir / "menin_compounds_curated.csv"
    measurement_path = processed_dir / "menin_activity_measurements.csv"
    herg_path = processed_dir / "herg_compounds_curated.csv"
    menin = pd.read_csv(menin_path) if menin_path.exists() else pd.DataFrame()
    herg = pd.read_csv(herg_path) if herg_path.exists() else pd.DataFrame()

    menin_training = menin
    if menin_endpoint is not None and measurement_path.exists():
        menin_training = pd.read_csv(measurement_path)

    return {
        "menin_activity": train_menin_activity_model(
            menin_training,
            models_dir,
            reports_dir,
            random_state=random_state,
            split_strategy=split_strategy,
            endpoint=menin_endpoint,
            assay_family=menin_assay_family,
            time_column=time_column,
            feature_backend=feature_backend,
        ),
        "herg_liability": train_herg_classifier_and_predict(
            herg,
            menin,
            models_dir,
            reports_dir,
            random_state=random_state,
            split_strategy=split_strategy,
            endpoint="IC50",
            assay_family="electrophysiology_functional",
            menin_endpoint=menin_endpoint,
            menin_assay_family=menin_assay_family,
            time_column=time_column,
            feature_backend=feature_backend,
        ),
    }
