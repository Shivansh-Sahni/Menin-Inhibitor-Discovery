"""Lightweight diagnostic baselines, sanity controls, and error schemas.

These models validate data flow and split difficulty.  Their outputs are
explicitly marked diagnostic and must not be reported as prospective or
external scientific evidence.  No hyperparameter sweep is implemented here.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .platform_features import stable_json_digest
from .platform_metrics import binary_classification_metrics, regression_metrics

BASELINE_SCHEMA_VERSION = "1.0.0"
DIAGNOSTIC_INTERPRETATION = "lightweight_diagnostic_not_scientific_or_prospective_evidence"
TaskType = Literal["regression", "classification"]
OutcomeKind = Literal["experimental_raw", "experimental_summary", "curated_assertion", "derived"]


@dataclass(frozen=True)
class BaselineConfig:
    """Restrained, fixed baseline settings; no search space is implied."""

    task_type: TaskType
    seed: int = 20260804
    class_imbalance: Literal["none", "balanced_class_weight"] = "balanced_class_weight"
    classification_threshold: float = 0.5
    ridge_alpha: float = 1.0
    logistic_c: float = 1.0
    tree_estimators: int = 100
    tree_max_depth: int = 12
    tree_min_samples_leaf: int = 2
    include_tree_baseline: bool = True

    def validate(self) -> None:
        if self.task_type not in {"regression", "classification"}:
            raise ValueError("task_type must be regression or classification")
        if not 0 <= self.classification_threshold <= 1:
            raise ValueError("classification_threshold must be in [0, 1]")
        if self.ridge_alpha <= 0 or self.logistic_c <= 0:
            raise ValueError("Regularization parameters must be positive")
        if self.tree_estimators < 1 or self.tree_max_depth < 1 or self.tree_min_samples_leaf < 1:
            raise ValueError("Tree settings must be positive")


@dataclass
class BaselineResult:
    metrics: pd.DataFrame
    predictions: pd.DataFrame
    fitted_models: dict[str, BaseEstimator]
    metadata: dict[str, Any]


def numeric_preprocessing_pipeline(
    estimator: BaseEstimator,
    *,
    imputation: Literal["median", "constant"] = "median",
    add_missing_indicator: bool = True,
    scale: bool = True,
) -> Pipeline:
    """Fit imputation/scaling only through a training-fitted sklearn pipeline."""

    if imputation not in {"median", "constant"}:
        raise ValueError("imputation must be median or constant")
    steps: list[tuple[str, Any]] = [
        (
            "imputer",
            SimpleImputer(
                strategy=imputation,
                fill_value=0.0 if imputation == "constant" else None,
                add_indicator=add_missing_indicator,
            ),
        )
    ]
    if scale:
        steps.append(("scaler", StandardScaler(with_mean=False)))
    steps.append(("model", estimator))
    return Pipeline(steps)


def class_imbalance_options(labels: Iterable[object]) -> dict[str, Any]:
    """Quantify support and prepare defensible options without selecting one."""

    y = np.asarray(list(labels))
    if y.ndim != 1 or not len(y):
        raise ValueError("A non-empty one-dimensional label array is required")
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        raise ValueError("Class imbalance analysis requires at least two classes")
    n = len(y)
    weights = {
        str(label): float(n / (len(classes) * count)) for label, count in zip(classes, counts, strict=True)
    }
    sample_weights = np.asarray([weights[str(label)] for label in y], dtype=float)
    return {
        "n": n,
        "class_counts": {str(label): int(count) for label, count in zip(classes, counts, strict=True)},
        "class_fractions": {
            str(label): float(count / n) for label, count in zip(classes, counts, strict=True)
        },
        "balanced_class_weights": weights,
        "effective_sample_size_balanced_weights": float(
            sample_weights.sum() ** 2 / np.sum(sample_weights**2)
        ),
        "prepared_options": [
            {
                "name": "unweighted",
                "use": "calibrated natural-prevalence reference",
                "risk": "minority recall may be poor",
            },
            {
                "name": "balanced_class_weight",
                "use": "loss reweighting fit only on training counts",
                "risk": "probabilities require calibration on natural-prevalence validation data",
            },
            {
                "name": "balanced_sampler",
                "use": "training-only sampler; never validation/test",
                "risk": "duplicate exposure and distorted effective prevalence",
            },
            {
                "name": "focal_loss",
                "parameters": {"gamma_candidates": [1.0, 2.0], "alpha": "fit_from_training_only"},
                "use": "future neural classification sensitivity analysis",
                "risk": "requires post-hoc probability calibration",
            },
            {
                "name": "threshold_adjustment",
                "use": "choose on validation for a declared operating constraint",
                "risk": "threshold is population/prevalence dependent",
            },
        ],
        "synthetic_structured_oversampling_default": "prohibited",
    }


def balanced_sample_weights(labels: Iterable[object]) -> np.ndarray:
    """Training-only inverse-frequency sample weights."""

    y = np.asarray(list(labels))
    options = class_imbalance_options(y)
    weights = options["balanced_class_weights"]
    return np.asarray([weights[str(label)] for label in y], dtype=float)


def _validate_matrix_pair(X_train: Any, X_eval: Any, y_train: np.ndarray, y_eval: np.ndarray) -> None:
    if getattr(X_train, "ndim", None) != 2 or getattr(X_eval, "ndim", None) != 2:
        raise ValueError("Feature matrices must be two-dimensional")
    if X_train.shape[0] != len(y_train) or X_eval.shape[0] != len(y_eval):
        raise ValueError("Feature rows must match target rows")
    if X_train.shape[1] != X_eval.shape[1]:
        raise ValueError("Training and evaluation matrices must have equal feature width")
    if X_train.shape[1] < 1:
        raise ValueError("At least one feature is required")


def _baseline_models(config: BaselineConfig) -> dict[str, BaseEstimator]:
    if config.task_type == "regression":
        models: dict[str, BaseEstimator] = {
            "dummy_median": DummyRegressor(strategy="median"),
            "ridge_fixed": Ridge(alpha=config.ridge_alpha),
        }
        if config.include_tree_baseline:
            models["extra_trees_restrained"] = ExtraTreesRegressor(
                n_estimators=config.tree_estimators,
                max_depth=config.tree_max_depth,
                min_samples_leaf=config.tree_min_samples_leaf,
                random_state=config.seed,
                n_jobs=1,
            )
        return models
    class_weight = "balanced" if config.class_imbalance == "balanced_class_weight" else None
    models = {
        "dummy_prior": DummyClassifier(strategy="prior"),
        "logistic_fixed": LogisticRegression(
            C=config.logistic_c,
            class_weight=class_weight,
            max_iter=1000,
            random_state=config.seed,
            solver="liblinear",
        ),
    }
    if config.include_tree_baseline:
        models["extra_trees_restrained"] = ExtraTreesClassifier(
            n_estimators=config.tree_estimators,
            max_depth=config.tree_max_depth,
            min_samples_leaf=config.tree_min_samples_leaf,
            class_weight=class_weight,
            random_state=config.seed,
            n_jobs=1,
        )
    return models


def _require_sha256(value: str, name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value.lower()):
        raise ValueError(f"{name} must be a 64-character SHA-256 hexadecimal digest")


def _outcome_kind_array(
    values: Sequence[object],
    *,
    n_rows: int,
    lineage_digests: Sequence[object] | None,
) -> tuple[np.ndarray, np.ndarray]:
    aliases = {
        "observed": "experimental_raw",
        "curated": "curated_assertion",
        "experimental_observation": "experimental_raw",
        "curated_label": "curated_assertion",
    }
    raw = list(values)
    if len(raw) != n_rows:
        raise ValueError("Row-level outcome kinds must match target rows")
    kinds = np.asarray(
        [aliases.get(str(value).strip().lower(), str(value).strip().lower()) for value in raw], dtype=object
    )
    allowed = {"experimental_raw", "experimental_summary", "curated_assertion", "derived"}
    invalid = sorted(set(kinds) - allowed)
    if invalid or any(not str(value) for value in kinds):
        raise ValueError(f"Prohibited or unknown outcome_kind values: {invalid}")
    digests = np.asarray(
        [""] * n_rows if lineage_digests is None else [str(value or "").strip() for value in lineage_digests],
        dtype=object,
    )
    if len(digests) != n_rows:
        raise ValueError("Row-level lineage digests must match target rows")
    if np.any((kinds == "derived") & (digests == "")):
        raise ValueError("Every derived label requires a row-level lineage digest")
    for digest in digests[kinds == "derived"]:
        _require_sha256(str(digest), "derived label lineage digest")
    return kinds, digests


def _classification_probability(model: BaseEstimator, X: Any) -> np.ndarray:
    if not hasattr(model, "predict_proba"):
        raise TypeError(f"Classification baseline {type(model).__name__} lacks predict_proba")
    probability = np.asarray(model.predict_proba(X), dtype=float)
    classes = np.asarray(model.classes_)
    positive = np.flatnonzero(classes == 1)
    if len(positive) != 1:
        raise ValueError("Classification baselines require training labels encoded as 0 and 1")
    return probability[:, int(positive[0])]


def _suspicion_flags(metrics: Mapping[str, Any], task_type: TaskType) -> list[str]:
    """Heuristic review flags; they are not proof of leakage."""

    flags: list[str] = []
    if task_type == "classification":
        if metrics.get("roc_auc") is not None and float(metrics["roc_auc"]) >= 0.98:
            flags.append("near_perfect_roc_auc_requires_duplicate_and_feature_leakage_review")
        if metrics.get("balanced_accuracy") is not None and float(metrics["balanced_accuracy"]) >= 0.95:
            flags.append("near_perfect_balanced_accuracy_requires_leakage_review")
    else:
        if metrics.get("r2") is not None and float(metrics["r2"]) >= 0.90:
            flags.append("very_high_r2_requires_target_proxy_and_near_duplicate_review")
        if metrics.get("mae") is not None and float(metrics["mae"]) <= 1e-8:
            flags.append("near_zero_error_requires_target_copy_review")
    return flags


def run_diagnostic_baselines(
    X_train: Any,
    y_train: Iterable[object],
    X_eval: Any,
    y_eval: Iterable[object],
    *,
    eval_record_ids: Sequence[object],
    config: BaselineConfig,
    task_id: str,
    split_name: str,
    feature_set_name: str,
    train_outcome_kinds: Sequence[object],
    eval_outcome_kinds: Sequence[object],
    train_label_lineage_digests: Sequence[object] | None = None,
    eval_label_lineage_digests: Sequence[object] | None = None,
    dataset_sha256: str,
    split_manifest_sha256: str,
    feature_artifact_sha256: str,
    eval_partition: Literal["validation", "test", "external"] = "test",
) -> BaselineResult:
    """Fit a fixed small model panel and return schema-valid predictions."""

    config.validate()
    _require_sha256(dataset_sha256, "dataset_sha256")
    _require_sha256(split_manifest_sha256, "split_manifest_sha256")
    _require_sha256(feature_artifact_sha256, "feature_artifact_sha256")
    train_target = np.asarray(list(y_train), dtype=float)
    eval_target = np.asarray(list(y_eval), dtype=float)
    if train_target.ndim != 1 or eval_target.ndim != 1:
        raise ValueError("Targets must be one-dimensional")
    if not np.all(np.isfinite(train_target)) or not np.all(np.isfinite(eval_target)):
        raise ValueError("Baselines require finite targets selected by an explicit task view")
    train_kinds, train_lineage = _outcome_kind_array(
        train_outcome_kinds,
        n_rows=len(train_target),
        lineage_digests=train_label_lineage_digests,
    )
    eval_kinds, eval_lineage = _outcome_kind_array(
        eval_outcome_kinds,
        n_rows=len(eval_target),
        lineage_digests=eval_label_lineage_digests,
    )
    if len(eval_record_ids) != len(eval_target):
        raise ValueError("eval_record_ids must match y_eval length")
    if len(set(map(str, eval_record_ids))) != len(eval_record_ids):
        raise ValueError("Evaluation record IDs must be unique")
    _validate_matrix_pair(X_train, X_eval, train_target, eval_target)
    if config.task_type == "classification":
        if set(np.unique(train_target)) != {0.0, 1.0}:
            raise ValueError("Classification training labels must contain both 0 and 1")
        if not set(np.unique(eval_target)).issubset({0.0, 1.0}):
            raise ValueError("Classification evaluation labels must be 0 or 1")

    metrics_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    fitted_models: dict[str, BaseEstimator] = {}
    for name, model in _baseline_models(config).items():
        scale = name in {"ridge_fixed", "logistic_fixed"}
        fitted_pipeline = numeric_preprocessing_pipeline(
            model,
            imputation="median",
            add_missing_indicator=True,
            scale=scale,
        )
        fitted_pipeline.fit(X_train, train_target)
        if config.task_type == "classification":
            prediction = _classification_probability(fitted_pipeline, X_eval)
            metrics = binary_classification_metrics(
                eval_target, prediction, threshold=config.classification_threshold
            )
            predicted_label = (prediction >= config.classification_threshold).astype(int)
            residual = prediction - eval_target
        else:
            prediction = np.asarray(fitted_pipeline.predict(X_eval), dtype=float)
            metrics = regression_metrics(eval_target, prediction)
            predicted_label = np.full(len(prediction), np.nan)
            residual = prediction - eval_target
        flags = _suspicion_flags(metrics, config.task_type)
        metrics_rows.append(
            {
                "schema_version": BASELINE_SCHEMA_VERSION,
                "task_id": task_id,
                "task_type": config.task_type,
                "split_name": split_name,
                "evaluation_partition": eval_partition,
                "feature_set": feature_set_name,
                "model_name": name,
                "seed": config.seed,
                "outcome_kind_counts_train": json.dumps(
                    dict(sorted(Counter(train_kinds).items())), sort_keys=True
                ),
                "outcome_kind_counts_evaluation": json.dumps(
                    dict(sorted(Counter(eval_kinds).items())), sort_keys=True
                ),
                "prediction_origin": "computational_prediction",
                "interpretation": DIAGNOSTIC_INTERPRETATION,
                "suspicion_flags": ";".join(flags),
                **{key: value for key, value in metrics.items() if key != "confusion_matrix"},
            }
        )
        prediction_frames.append(
            pd.DataFrame(
                {
                    "schema_version": BASELINE_SCHEMA_VERSION,
                    "record_id": [str(item) for item in eval_record_ids],
                    "task_id": task_id,
                    "task_type": config.task_type,
                    "split_name": split_name,
                    "partition": eval_partition,
                    "feature_set": feature_set_name,
                    "model_name": name,
                    "observed_label": eval_target,
                    "outcome_kind": eval_kinds,
                    "derived_label_lineage_digest": eval_lineage,
                    "predicted_value": prediction,
                    "predicted_label": predicted_label,
                    "prediction_origin": "computational_prediction",
                    "residual_prediction_minus_observation": residual,
                    "classification_threshold": (
                        config.classification_threshold if config.task_type == "classification" else np.nan
                    ),
                    "interpretation": DIAGNOSTIC_INTERPRETATION,
                }
            )
        )
        fitted_models[name] = fitted_pipeline
    predictions = pd.concat(prediction_frames, ignore_index=True)
    validate_prediction_schema(predictions)
    metadata = {
        "schema_version": BASELINE_SCHEMA_VERSION,
        "config": asdict(config),
        "task_id": task_id,
        "split_name": split_name,
        "feature_set": feature_set_name,
        "n_train": int(len(train_target)),
        "n_evaluation": int(len(eval_target)),
        "n_features": int(X_train.shape[1]),
        "sparse_input": bool(sparse.issparse(X_train)),
        "model_selection_performed": False,
        "hyperparameter_sweep_performed": False,
        "dataset_sha256": dataset_sha256,
        "split_manifest_sha256": split_manifest_sha256,
        "feature_artifact_sha256": feature_artifact_sha256,
        "preprocessing": {
            "imputation": "training_fitted_median_with_missing_indicators",
            "linear_scaling": "training_fitted_standard_scaler_without_centering",
            "tree_scaling": "none",
        },
        "preprocessing_sha256": stable_json_digest(
            {
                "imputation": "training_fitted_median_with_missing_indicators",
                "linear_scaling": "training_fitted_standard_scaler_without_centering",
                "tree_scaling": "none",
            }
        ),
        "training_outcome_kind_counts": dict(sorted(Counter(train_kinds).items())),
        "evaluation_outcome_kind_counts": dict(sorted(Counter(eval_kinds).items())),
        "interpretation": DIAGNOSTIC_INTERPRETATION,
    }
    return BaselineResult(
        metrics=pd.DataFrame(metrics_rows),
        predictions=predictions,
        fitted_models=fitted_models,
        metadata=metadata,
    )


def validate_prediction_schema(predictions: pd.DataFrame) -> None:
    """Fail if observations and predictions are conflated or IDs are ambiguous."""

    required = {
        "record_id",
        "task_id",
        "task_type",
        "split_name",
        "partition",
        "model_name",
        "observed_label",
        "outcome_kind",
        "predicted_value",
        "prediction_origin",
        "interpretation",
    }
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"Prediction frame is missing columns: {missing}")
    if predictions[list(required)].isna().any().any():
        nullable = {"predicted_label"}
        checked = list(required - nullable)
        if predictions[checked].isna().any().any():
            raise ValueError("Required prediction schema fields may not be missing")
    normalization = {
        "observed": "experimental_raw",
        "curated": "curated_assertion",
        "experimental_observation": "experimental_raw",
        "curated_label": "curated_assertion",
    }
    kinds = predictions["outcome_kind"].astype(str).map(lambda value: normalization.get(value, value))
    allowed = {"experimental_raw", "experimental_summary", "curated_assertion", "derived"}
    if not set(kinds).issubset(allowed):
        raise ValueError(
            "Outcome kinds must be experimental, curated assertions, or lineage-backed derived labels"
        )
    derived = kinds.eq("derived")
    if derived.any():
        if "derived_label_lineage_digest" not in predictions.columns:
            raise ValueError("Derived labels require derived_label_lineage_digest")
        present = (
            predictions.loc[derived, "derived_label_lineage_digest"].fillna("").astype(str).str.strip().ne("")
        )
        if not present.all():
            raise ValueError("Derived labels require a non-empty lineage digest")
    if set(predictions["prediction_origin"].astype(str)) != {"computational_prediction"}:
        raise ValueError("Prediction rows must be explicitly computational predictions")
    duplicate_key = ["record_id", "task_id", "split_name", "partition", "model_name"]
    if predictions.duplicated(duplicate_key).any():
        raise ValueError("Prediction frame contains duplicate record/task/split/model rows")


def build_error_analysis(predictions: pd.DataFrame) -> pd.DataFrame:
    """Rank errors and assign task-appropriate review categories."""

    validate_prediction_schema(predictions)
    output = predictions.copy()
    output["absolute_error"] = np.abs(
        pd.to_numeric(output["predicted_value"]) - pd.to_numeric(output["observed_label"])
    )
    classification = output["task_type"].eq("classification")
    output["error_type"] = "regression_error"
    if "predicted_label" in output.columns:
        observed = pd.to_numeric(output["observed_label"], errors="coerce")
        predicted = pd.to_numeric(output["predicted_label"], errors="coerce")
        output.loc[classification & (observed == 0) & (predicted == 1), "error_type"] = "false_positive"
        output.loc[classification & (observed == 1) & (predicted == 0), "error_type"] = "false_negative"
        output.loc[classification & (observed == predicted), "error_type"] = "correct_classification"
    output["distance_to_binary_threshold"] = np.where(
        classification,
        np.abs(
            pd.to_numeric(output["predicted_value"])
            - pd.to_numeric(output["classification_threshold"], errors="coerce")
        ),
        np.nan,
    )
    output["error_rank_within_model"] = (
        output.groupby(["task_id", "split_name", "model_name"])["absolute_error"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    return output.sort_values(
        ["task_id", "split_name", "model_name", "error_rank_within_model"], kind="mergesort"
    ).reset_index(drop=True)


def numeric_target_leakage_scan(
    features: pd.DataFrame,
    target: Iterable[object],
    *,
    exact_tolerance: float = 1e-12,
    correlation_threshold: float = 0.995,
) -> pd.DataFrame:
    """Flag direct copies and near-deterministic numeric target proxies."""

    y = np.asarray(list(target), dtype=float)
    if len(features) != len(y) or not np.all(np.isfinite(y)):
        raise ValueError("Features and finite target must have equal row counts")
    rows: list[dict[str, Any]] = []
    for column in features.columns:
        values = pd.to_numeric(features[column], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(values)
        if np.sum(finite) < 2:
            continue
        exact = bool(np.all(np.abs(values[finite] - y[finite]) <= exact_tolerance))
        correlation = (
            float(np.corrcoef(values[finite], y[finite])[0, 1])
            if np.std(values[finite]) > 0 and np.std(y[finite]) > 0
            else None
        )
        rows.append(
            {
                "feature": str(column),
                "n_finite": int(np.sum(finite)),
                "fraction_finite": float(np.mean(finite)),
                "exact_target_copy": exact,
                "pearson_correlation": correlation,
                "high_absolute_correlation": bool(
                    correlation is not None and abs(correlation) >= correlation_threshold
                ),
                "requires_review": exact
                or bool(correlation is not None and abs(correlation) >= correlation_threshold),
            }
        )
    columns = [
        "feature",
        "n_finite",
        "fraction_finite",
        "exact_target_copy",
        "pearson_correlation",
        "high_absolute_correlation",
        "requires_review",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    return (
        pd.DataFrame(rows, columns=columns)
        .sort_values(["requires_review", "exact_target_copy", "feature"], ascending=[False, False, True])
        .reset_index(drop=True)
    )


def run_label_permutation_control(
    X_train: Any,
    y_train: Iterable[object],
    X_eval: Any,
    y_eval: Iterable[object],
    *,
    config: BaselineConfig,
) -> dict[str, Any]:
    """One deterministic label-permutation control, not a significance test."""

    train = np.asarray(list(y_train), dtype=float)
    evaluation = np.asarray(list(y_eval), dtype=float)
    if not np.all(np.isfinite(train)) or not np.all(np.isfinite(evaluation)):
        raise ValueError("Permutation control targets must be finite")
    _validate_matrix_pair(X_train, X_eval, train, evaluation)
    rng = np.random.default_rng(config.seed + 104729)
    permuted = train[rng.permutation(len(train))]
    if config.task_type == "classification":
        model: BaseEstimator = LogisticRegression(
            C=config.logistic_c,
            class_weight="balanced" if config.class_imbalance == "balanced_class_weight" else None,
            max_iter=1000,
            random_state=config.seed,
            solver="liblinear",
        )
        model.fit(X_train, permuted)
        prediction = _classification_probability(model, X_eval)
        metrics = binary_classification_metrics(evaluation, prediction)
    else:
        model = Ridge(alpha=config.ridge_alpha)
        model.fit(X_train, permuted)
        prediction = np.asarray(model.predict(X_eval), dtype=float)
        metrics = regression_metrics(evaluation, prediction)
    return {
        "control": "single_training_label_permutation",
        "seed": config.seed + 104729,
        "metrics": metrics,
        "interpretation": (
            "sanity_control_only; unexpected performance triggers leakage review but one draw is not a p-value"
        ),
    }


def run_identifier_hash_control(
    train_identifiers: Sequence[object],
    y_train: Iterable[object],
    eval_identifiers: Sequence[object],
    y_eval: Iterable[object],
    *,
    config: BaselineConfig,
    n_features: int = 256,
) -> dict[str, Any]:
    """Test whether textual identifiers alone encode suspicious outcome signal."""

    if n_features < 1:
        raise ValueError("n_features must be positive")
    vectorizer = HashingVectorizer(
        analyzer="char",
        ngram_range=(2, 5),
        n_features=n_features,
        alternate_sign=False,
        norm="l2",
        lowercase=False,
    )
    train_text = [str(item) for item in train_identifiers]
    eval_text = [str(item) for item in eval_identifiers]
    X_train = vectorizer.transform(train_text)
    X_eval = vectorizer.transform(eval_text)
    train = np.asarray(list(y_train), dtype=float)
    evaluation = np.asarray(list(y_eval), dtype=float)
    if not np.all(np.isfinite(train)) or not np.all(np.isfinite(evaluation)):
        raise ValueError("Identifier control targets must be finite")
    _validate_matrix_pair(X_train, X_eval, train, evaluation)
    if config.task_type == "classification":
        model: BaseEstimator = LogisticRegression(
            C=config.logistic_c,
            class_weight="balanced" if config.class_imbalance == "balanced_class_weight" else None,
            max_iter=1000,
            random_state=config.seed,
            solver="liblinear",
        )
        model.fit(X_train, train)
        prediction = _classification_probability(model, X_eval)
        metrics = binary_classification_metrics(evaluation, prediction)
    else:
        model = Ridge(alpha=config.ridge_alpha)
        model.fit(X_train, train)
        prediction = np.asarray(model.predict(X_eval), dtype=float)
        metrics = regression_metrics(evaluation, prediction)
    return {
        "control": "identifier_text_hash",
        "identifier_digest_train": hashlib.sha256("\n".join(train_text).encode()).hexdigest(),
        "identifier_digest_eval": hashlib.sha256("\n".join(eval_text).encode()).hexdigest(),
        "n_hash_features": n_features,
        "metrics": metrics,
        "interpretation": "identifier_only_negative_control; strong performance requires provenance/leakage review",
    }


def robustness_configuration_matrix() -> pd.DataFrame:
    """Preregistered sensitivity axes to run after model training authorization."""

    rows = [
        ("structure_policy", "parent_standardized", "default"),
        ("structure_policy", "exact_submitted_state", "sensitivity"),
        ("censoring", "censored_likelihood", "default"),
        ("censoring", "exact_only", "sensitivity"),
        ("label_confidence", "all_model_eligible", "default"),
        ("label_confidence", "gold_protocol_only", "sensitivity"),
        ("source", "all_public_sources", "default"),
        ("source", "one_source_removed", "sensitivity"),
        ("features", "molecule_only", "baseline"),
        ("features", "molecule_plus_protein", "primary_multimodal"),
        ("features", "structure_conditioned", "incremental_value"),
        ("similarity_threshold", "tanimoto_0.70", "sensitivity"),
        ("similarity_threshold", "tanimoto_0.80", "default"),
        ("similarity_threshold", "tanimoto_0.90", "sensitivity"),
        ("random_seed", "fixed_primary_seed", "default"),
        ("random_seed", "predeclared_additional_seeds", "algorithmic_stability"),
        ("imbalance", "natural_prevalence", "calibration_reference"),
        ("imbalance", "balanced_class_weight", "sensitivity"),
        ("outliers", "retain_valid_extremes", "default"),
        ("outliers", "winsorization_fit_on_train", "sensitivity_only"),
    ]
    return pd.DataFrame(rows, columns=["axis", "variant", "role"]).assign(
        execution_status="configured_not_run",
        selection_rule="compare_on_frozen_validation; report all prespecified locked-test sensitivities",
    )
