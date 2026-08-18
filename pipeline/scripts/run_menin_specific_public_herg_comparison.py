#!/usr/bin/env python3
"""Evaluate Menin-specific public hERG augmentation without replacing prior work.

This analysis adds two new training regimes to the retained hERG mix-and-match
study:

* internal exact measurements plus Menin-specific public evidence; and
* Menin-specific public evidence only (an external-only transfer stress test).

The primary augmentation uses a Gaussian censored-ridge likelihood so that
one-sided public IC50 limits remain limits.  Exact-only RF/SVR/ridge/ExtraTrees
experiments are retained as sensitivity analyses.  All evaluation compounds
and folds are inherited from the existing mix-and-match outputs, which keeps
the comparison paired and prevents a favorable re-split.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import log_ndtr
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "pipeline/scripts"))
sys.path.insert(0, str(ROOT / "pipeline/src"))

import run_herg_pk_mix_match as base  # noqa: E402
from menin_discovery.features import canonicalize_smiles, scaffold_key  # noqa: E402
from menin_discovery.research_common import (  # noqa: E402
    atomic_write_csv,
    atomic_write_json,
    atomic_write_parquet,
    atomic_write_text,
)

SEED = 20260803
RIDGE_ALPHA = 10.0
PERMUTATION_DRAWS = 500
FEATURE_LAYERS = ("compact_proxies", "morgan_latent", "hybrid")
EXACT_MODELS = ("ridge", "svr", "random_forest", "extra_trees")
PRIMARY_ROLES = {
    "core_within_series",
    "core_censored",
    "core_censored_with_protocol_flag",
}
CONDITIONAL_ROLES = {"conditional_2d_exact", "conditional_2d_censored"}

DEFAULT_SOURCE = (
    ROOT
    / "research/data/pk_herg/canonical/menin_specific_public_herg"
    / "menin_specific_public_herg_dataset.xlsx"
)
DEFAULT_PRIOR = ROOT / "research/reports/pk_herg/mix_match/herg_mix_match_predictions.parquet"
DEFAULT_OUTPUT = ROOT / "research/reports/pk_herg/menin_specific_public_comparison"


def _safe_spearman(observed: np.ndarray, predicted: np.ndarray) -> float:
    if len(observed) < 3 or np.unique(observed).size < 2 or np.unique(predicted).size < 2:
        return float("nan")
    return float(spearmanr(observed, predicted).statistic)


def _read_source(path: Path) -> dict[str, pd.DataFrame]:
    sheets = {
        name: pd.read_excel(path, sheet_name=name, header=3)
        for name in ("Compound Registry", "Measurements Raw", "Model View", "Source Catalog")
    }
    registry = sheets["Compound Registry"].copy()
    registry = registry[registry["compound_id"].notna()].reset_index(drop=True)
    registry["canonicalized_smiles"] = registry["canonical_smiles"].map(
        lambda value: canonicalize_smiles(value) if pd.notna(value) else ""
    )
    registry["scaffold"] = registry["canonicalized_smiles"].map(
        lambda value: scaffold_key(value)[0] if value else ""
    )
    sheets["Compound Registry"] = registry
    for name in ("Measurements Raw", "Model View", "Source Catalog"):
        id_column = {
            "Measurements Raw": "measurement_id",
            "Model View": "model_record_id",
            "Source Catalog": "source_id",
        }[name]
        sheets[name] = sheets[name][sheets[name][id_column].notna()].reset_index(drop=True)
    return sheets


def _interval_bounds(row: pd.Series) -> tuple[float, float]:
    value = float(row["pIC50"])
    relation = str(row["pIC50_relation"]).strip()
    if relation == "EQ":
        return value, value
    if relation in {">", ">="}:
        return value, math.inf
    if relation in {"<", "<="}:
        return -math.inf, value
    raise ValueError(f"Unsupported pIC50 relation: {relation}")


def _public_frames(
    sheets: dict[str, pd.DataFrame],
    *,
    include_conditional: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    registry = sheets["Compound Registry"]
    model = sheets["Model View"].merge(
        registry[
            [
                "compound_id",
                "canonicalized_smiles",
                "scaffold",
                "model_structure_eligible",
            ]
        ],
        on="compound_id",
        how="left",
        validate="many_to_one",
    )
    allowed_roles = set(PRIMARY_ROLES)
    if include_conditional:
        allowed_roles |= CONDITIONAL_ROLES
    eligible = model[
        model["endpoint"].eq("IC50")
        & model["model_role"].isin(allowed_roles)
        & model["canonicalized_smiles"].astype(str).ne("")
        & model["pIC50"].notna()
    ].copy()
    if not include_conditional:
        eligible = eligible[eligible["model_structure_eligible"].eq("Yes")].copy()
    duplicate = eligible["compound_id"].duplicated(keep=False)
    if duplicate.any():
        names = sorted(eligible.loc[duplicate, "compound_id"].astype(str).unique())
        raise ValueError(f"Primary public selection has duplicate compound rows: {names}")
    bounds = eligible.apply(_interval_bounds, axis=1, result_type="expand")
    eligible["pic50_lower"] = bounds[0].astype(float)
    eligible["pic50_upper"] = bounds[1].astype(float)
    eligible["record_id"] = "menin_public:" + eligible["compound_id"].astype(str)
    eligible["standardized_smiles"] = eligible["canonicalized_smiles"]
    eligible["source_group"] = "menin_specific_public"
    eligible["dataset_role"] = np.where(
        eligible["model_role"].isin(CONDITIONAL_ROLES),
        "conditional_stereo_unresolved_sensitivity",
        "primary_menin_specific_public",
    )
    eligible["target_pic50"] = np.where(
        np.isclose(eligible["pic50_lower"], eligible["pic50_upper"]),
        eligible["pic50_lower"],
        np.nan,
    )
    keep = [
        "record_id",
        "compound_id",
        "preferred_name",
        "series_id",
        "standardized_smiles",
        "scaffold",
        "target_pic50",
        "pic50_lower",
        "pic50_upper",
        "pIC50_relation",
        "model_role",
        "source_id",
        "source_group",
        "dataset_role",
        "structure_quality",
    ]
    intervals = base._attach_features(eligible[keep].reset_index(drop=True))
    exact = intervals[intervals["target_pic50"].notna()].reset_index(drop=True)
    audit = model.copy()
    audit["included_primary_interval"] = audit["model_record_id"].isin(
        eligible.loc[~eligible["model_role"].isin(CONDITIONAL_ROLES), "model_record_id"]
    )
    audit["included_conditional_sensitivity"] = audit["model_record_id"].isin(
        eligible.loc[eligible["model_role"].isin(CONDITIONAL_ROLES), "model_record_id"]
    )
    audit["exclusion_reason"] = ""
    audit.loc[audit["endpoint"].ne("IC50"), "exclusion_reason"] = (
        "concentration-specific inhibition retained for future joint model; not an IC50 label"
    )
    audit.loc[
        audit["endpoint"].eq("IC50") & ~audit["model_role"].isin(allowed_roles),
        "exclusion_reason",
    ] = "duplicate/protocol-sensitivity/comparator role; excluded from independent training"
    audit.loc[audit["canonicalized_smiles"].astype(str).eq(""), "exclusion_reason"] = "structure unresolved"
    if not include_conditional:
        audit.loc[audit["model_structure_eligible"].eq("Conditional"), "exclusion_reason"] = (
            "stereochemistry-unresolved parent; conditional sensitivity only"
        )
    return exact, intervals, audit


@dataclass
class CensoredFit:
    prediction: np.ndarray
    sigma_pic50: float
    converged: bool
    iterations: int
    objective: float


def _fit_censored_ridge(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    feature_layer: str,
    alpha: float = RIDGE_ALPHA,
) -> CensoredFit:
    transformer = base.FeatureTransformer(feature_layer)
    train_x = transformer.fit_transform(train)
    test_x = transformer.transform(test)
    lower = train["pic50_lower"].to_numpy(dtype=float)
    upper = train["pic50_upper"].to_numpy(dtype=float)
    exact = np.isfinite(lower) & np.isfinite(upper) & np.isclose(lower, upper)
    if exact.sum() < 3:
        raise ValueError("Censored ridge requires at least three exact anchors")
    center = float(np.mean(lower[exact]))
    scale = float(np.std(lower[exact], ddof=1))
    scale = max(scale, 0.25)
    lower_z = (lower - center) / scale
    upper_z = (upper - center) / scale
    pseudo = np.where(
        exact,
        lower_z,
        np.where(np.isfinite(lower_z), lower_z + 0.35, upper_z - 0.35),
    )
    initializer = Ridge(alpha=alpha).fit(train_x, pseudo)
    initial_prediction = initializer.predict(train_x)
    initial_sigma = max(float(np.std(pseudo - initial_prediction)), 0.35)
    theta0 = np.r_[initializer.intercept_, initializer.coef_, math.log(initial_sigma)]

    def objective(theta: np.ndarray) -> float:
        intercept = theta[0]
        coefficients = theta[1:-1]
        sigma = math.exp(float(theta[-1]))
        mu = intercept + train_x @ coefficients
        log_likelihood = np.zeros(len(train), dtype=float)
        if exact.any():
            residual = (lower_z[exact] - mu[exact]) / sigma
            log_likelihood[exact] = (
                -0.5 * residual * residual - math.log(sigma) - 0.5 * math.log(2.0 * math.pi)
            )
        lower_only = np.isfinite(lower_z) & ~np.isfinite(upper_z)
        if lower_only.any():
            log_likelihood[lower_only] = log_ndtr((mu[lower_only] - lower_z[lower_only]) / sigma)
        upper_only = ~np.isfinite(lower_z) & np.isfinite(upper_z)
        if upper_only.any():
            log_likelihood[upper_only] = log_ndtr((upper_z[upper_only] - mu[upper_only]) / sigma)
        interval = np.isfinite(lower_z) & np.isfinite(upper_z) & ~exact
        if interval.any():
            hi = log_ndtr((upper_z[interval] - mu[interval]) / sigma)
            lo = log_ndtr((lower_z[interval] - mu[interval]) / sigma)
            difference = np.maximum(np.exp(hi) - np.exp(lo), 1e-300)
            log_likelihood[interval] = np.log(difference)
        penalty = 0.5 * alpha * float(np.dot(coefficients, coefficients))
        return float(-np.sum(log_likelihood) + penalty)

    result = minimize(
        objective,
        theta0,
        method="L-BFGS-B",
        bounds=[(None, None)] * (len(theta0) - 1) + [(math.log(0.05), math.log(3.0))],
        options={"maxiter": 2500, "ftol": 1e-11, "gtol": 1e-7},
    )
    theta = result.x
    prediction_z = theta[0] + test_x @ theta[1:-1]
    return CensoredFit(
        prediction=prediction_z * scale + center,
        sigma_pic50=math.exp(float(theta[-1])) * scale,
        converged=bool(result.success),
        iterations=int(result.nit),
        objective=float(result.fun),
    )


def _fold_definitions(
    internal: pd.DataFrame,
    extension: pd.DataFrame,
    prior: pd.DataFrame,
) -> list[dict[str, Any]]:
    definitions: list[dict[str, Any]] = []
    anchor = prior[
        prior["data_regime"].eq("internal_only")
        & prior["feature_layer"].eq("hybrid")
        & prior["model"].eq("ridge")
    ]
    for (evaluation, fold), group in anchor.groupby(["evaluation", "fold"], sort=True):
        if evaluation not in {"internal_scaffold_cv", "angelo_fixed_nonoverlap"}:
            continue
        test_ids = set(group["record_id"].astype(str))
        if evaluation == "internal_scaffold_cv":
            test = internal[internal["record_id"].isin(test_ids)].copy()
            train = internal[~internal["record_id"].isin(test_ids)].copy()
        else:
            test = extension[extension["record_id"].isin(test_ids)].copy()
            train = internal.copy()
        if len(test) != len(test_ids):
            missing = sorted(test_ids - set(test["record_id"].astype(str)))
            raise ValueError(f"Could not reconstruct retained test fold {evaluation}/{fold}: {missing}")
        definitions.append(
            {
                "evaluation": evaluation,
                "fold": int(fold),
                "train": train.reset_index(drop=True),
                "test": test.reset_index(drop=True),
            }
        )
    return definitions


def _prediction_rows(
    definition: dict[str, Any],
    prediction: np.ndarray,
    train: pd.DataFrame,
    *,
    regime: str,
    feature_layer: str,
    model: str,
    sigma: float = math.nan,
    converged: bool = True,
    iterations: int = 0,
) -> list[dict[str, Any]]:
    test = definition["test"]
    max_similarity = base._max_tanimoto(test, train.drop_duplicates("record_id"))
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(test.itertuples(index=False)):
        rows.append(
            {
                "evaluation": definition["evaluation"],
                "fold": definition["fold"],
                "record_id": record.record_id,
                "compound_id": record.compound_id,
                "scaffold": record.scaffold,
                "observed_pic50": float(record.target_pic50),
                "predicted_pic50": float(prediction[index]),
                "residual": float(prediction[index] - record.target_pic50),
                "absolute_error": float(abs(prediction[index] - record.target_pic50)),
                "data_regime": regime,
                "feature_layer": feature_layer,
                "model": model,
                "training_structures": int(train["record_id"].nunique()),
                "training_internal_structures": int(train["source_group"].eq("internal").sum()),
                "training_menin_public_structures": int(
                    train["source_group"].eq("menin_specific_public").sum()
                ),
                "max_any_train_tanimoto": float(max_similarity[index]),
                "fitted_sigma_pic50": sigma,
                "optimizer_converged": converged,
                "optimizer_iterations": iterations,
            }
        )
    return rows


def _prepare_interval_train(exact_train: pd.DataFrame) -> pd.DataFrame:
    result = exact_train.copy()
    result["pic50_lower"] = result["target_pic50"].astype(float)
    result["pic50_upper"] = result["target_pic50"].astype(float)
    return result


def _run_new_matrix(
    definitions: list[dict[str, Any]],
    public_exact: pd.DataFrame,
    public_intervals: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for definition in definitions:
        internal_train = definition["train"]
        test = definition["test"]
        exact_pools = {
            "internal_only_reproduced": internal_train,
            "internal_plus_menin_public_exact": pd.concat([internal_train, public_exact], ignore_index=True),
            "menin_public_only_exact": public_exact.copy(),
        }
        for regime, train in exact_pools.items():
            for layer in FEATURE_LAYERS:
                for model_name in EXACT_MODELS:
                    try:
                        prediction = base._fit_predict(
                            train.assign(sample_weight=1.0),
                            test,
                            feature_layer=layer,
                            model_name=model_name,
                        )
                        rows.extend(
                            _prediction_rows(
                                definition,
                                prediction,
                                train,
                                regime=regime,
                                feature_layer=layer,
                                model=model_name,
                            )
                        )
                    except Exception as exc:
                        failures.append(
                            {
                                "evaluation": definition["evaluation"],
                                "fold": definition["fold"],
                                "data_regime": regime,
                                "feature_layer": layer,
                                "model": model_name,
                                "failure": f"{type(exc).__name__}: {exc}",
                            }
                        )
            try:
                prediction = base._tanimoto_predict(train, test, neighbors=3)
                rows.extend(
                    _prediction_rows(
                        definition,
                        prediction,
                        train,
                        regime=regime,
                        feature_layer="morgan_tanimoto",
                        model="tanimoto_3nn",
                    )
                )
            except Exception as exc:
                failures.append(
                    {
                        "evaluation": definition["evaluation"],
                        "fold": definition["fold"],
                        "data_regime": regime,
                        "feature_layer": "morgan_tanimoto",
                        "model": "tanimoto_3nn",
                        "failure": f"{type(exc).__name__}: {exc}",
                    }
                )
        interval_pools = {
            "internal_only_censored": _prepare_interval_train(internal_train),
            "internal_plus_menin_public_censored": pd.concat(
                [_prepare_interval_train(internal_train), public_intervals],
                ignore_index=True,
            ),
            "menin_public_only_censored": public_intervals.copy(),
        }
        for regime, train in interval_pools.items():
            for layer in FEATURE_LAYERS:
                try:
                    fit = _fit_censored_ridge(
                        train,
                        test,
                        feature_layer=layer,
                        alpha=RIDGE_ALPHA,
                    )
                    rows.extend(
                        _prediction_rows(
                            definition,
                            fit.prediction,
                            train,
                            regime=regime,
                            feature_layer=layer,
                            model="censored_ridge",
                            sigma=fit.sigma_pic50,
                            converged=fit.converged,
                            iterations=fit.iterations,
                        )
                    )
                except Exception as exc:
                    failures.append(
                        {
                            "evaluation": definition["evaluation"],
                            "fold": definition["fold"],
                            "data_regime": regime,
                            "feature_layer": layer,
                            "model": "censored_ridge",
                            "failure": f"{type(exc).__name__}: {exc}",
                        }
                    )
    return pd.DataFrame(rows), pd.DataFrame(failures)


def _metric_row(frame: pd.DataFrame) -> dict[str, float]:
    observed = frame["observed_pic50"].to_numpy(dtype=float)
    predicted = frame["predicted_pic50"].to_numpy(dtype=float)
    error = predicted - observed
    return {
        "n": int(len(frame)),
        "n_scaffolds": int(frame["scaffold"].nunique()),
        "mae": float(mean_absolute_error(observed, predicted)),
        "rmse": float(math.sqrt(mean_squared_error(observed, predicted))),
        "r2": float(r2_score(observed, predicted)),
        "spearman": _safe_spearman(observed, predicted),
        "mean_signed_error": float(np.mean(error)),
        "fraction_within_0p5_log": float(np.mean(np.abs(error) <= 0.5)),
        "fraction_within_1p0_log": float(np.mean(np.abs(error) <= 1.0)),
        "median_max_train_tanimoto": float(frame["max_any_train_tanimoto"].median()),
        "maximum_absolute_error": float(np.max(np.abs(error))),
        "all_optimizers_converged": bool(frame["optimizer_converged"].all()),
    }


def _scaffold_bootstrap_delta(
    candidate: pd.DataFrame,
    baseline: pd.DataFrame,
    *,
    draws: int = 5000,
) -> dict[str, float]:
    paired = candidate[["record_id", "scaffold", "absolute_error"]].merge(
        baseline[["record_id", "absolute_error"]],
        on="record_id",
        validate="one_to_one",
        suffixes=("_candidate", "_baseline"),
    )
    paired["delta"] = paired["absolute_error_candidate"] - paired["absolute_error_baseline"]
    groups = paired.groupby("scaffold", sort=True)["delta"].agg(["sum", "count"])
    rng = np.random.default_rng(SEED)
    sampled = rng.integers(0, len(groups), size=(draws, len(groups)))
    delta = groups["sum"].to_numpy()[sampled].sum(axis=1) / groups["count"].to_numpy()[sampled].sum(axis=1)
    return {
        "mae_delta_vs_internal": float(paired["delta"].mean()),
        "mae_delta_lower_95": float(np.quantile(delta, 0.025)),
        "mae_delta_upper_95": float(np.quantile(delta, 0.975)),
        "bootstrap_probability_improved": float(np.mean(delta < 0.0)),
    }


def _summarize(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = ["evaluation", "data_regime", "feature_layer", "model"]
    for key, frame in predictions.groupby(keys, sort=True):
        rows.append({**dict(zip(keys, key, strict=True)), **_metric_row(frame)})
    metrics = pd.DataFrame(rows)
    delta_rows: list[dict[str, Any]] = []
    for key, frame in predictions.groupby(keys, sort=True):
        evaluation, regime, layer, model = key
        baseline_regime = (
            "internal_only_censored" if model == "censored_ridge" else "internal_only_reproduced"
        )
        baseline = predictions[
            predictions["evaluation"].eq(evaluation)
            & predictions["data_regime"].eq(baseline_regime)
            & predictions["feature_layer"].eq(layer)
            & predictions["model"].eq(model)
        ]
        if baseline.empty:
            continue
        delta_rows.append(
            {
                **dict(zip(keys, key, strict=True)),
                "comparison_baseline_regime": baseline_regime,
                **_scaffold_bootstrap_delta(frame, baseline),
            }
        )
    return (
        metrics.merge(pd.DataFrame(delta_rows), on=keys, how="left")
        .sort_values(["evaluation", "mae", "data_regime", "feature_layer", "model"])
        .reset_index(drop=True)
    )


def _prior_comparators(prior: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = prior[
        prior["evaluation"].isin({"internal_scaffold_cv", "angelo_fixed_nonoverlap"})
        & prior["feature_layer"].eq("hybrid")
        & prior["model"].eq("ridge")
        & prior["data_regime"].isin({"internal_only", "internal_plus_public_balanced"})
    ].copy()
    selected["comparator_label"] = selected["data_regime"].map(
        {
            "internal_only": "retained_internal_only",
            "internal_plus_public_balanced": "retained_internal_plus_broad_public",
        }
    )
    metrics: list[dict[str, Any]] = []
    for (evaluation, label), frame in selected.groupby(["evaluation", "comparator_label"]):
        adapted = frame.rename(columns={"max_any_train_tanimoto": "max_any_train_tanimoto"})
        adapted["optimizer_converged"] = True
        metrics.append(
            {
                "evaluation": evaluation,
                "comparison_regime": label,
                "feature_layer": "hybrid",
                "model": "ridge",
                **_metric_row(adapted),
            }
        )
    return selected, pd.DataFrame(metrics)


def _exact_label_permutation_test(
    definitions: list[dict[str, Any]],
    public_exact: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    draws: int = PERMUTATION_DRAWS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Test whether public structure-label pairing beats shuffled public labels.

    The hybrid ridge model is the locked diagnostic.  Hybrid SVR is retained as
    explicitly post-hoc because it showed the most consistent improvement in
    the full matrix; its permutation p-value does not erase that selection.
    """

    rng = np.random.default_rng(SEED + 91)
    configurations = [
        ("hybrid", "ridge", "locked_primary"),
        ("hybrid", "svr", "post_hoc_model_sensitivity"),
    ]
    draw_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    labels = public_exact["target_pic50"].to_numpy(dtype=float)
    for layer, model_name, selection_status in configurations:
        observed_by_evaluation: dict[str, float] = {}
        baseline_error_by_evaluation: dict[str, pd.DataFrame] = {}
        for evaluation in sorted(predictions["evaluation"].unique()):
            candidate = predictions[
                predictions["evaluation"].eq(evaluation)
                & predictions["data_regime"].eq("internal_plus_menin_public_exact")
                & predictions["feature_layer"].eq(layer)
                & predictions["model"].eq(model_name)
            ]
            baseline = predictions[
                predictions["evaluation"].eq(evaluation)
                & predictions["data_regime"].eq("internal_only_reproduced")
                & predictions["feature_layer"].eq(layer)
                & predictions["model"].eq(model_name)
            ][["record_id", "absolute_error"]].rename(columns={"absolute_error": "baseline_absolute_error"})
            paired = candidate[["record_id", "absolute_error"]].merge(
                baseline, on="record_id", validate="one_to_one"
            )
            observed_by_evaluation[evaluation] = float(
                (paired["absolute_error"] - paired["baseline_absolute_error"]).mean()
            )
            baseline_error_by_evaluation[evaluation] = baseline
        null_by_evaluation: dict[str, list[float]] = {evaluation: [] for evaluation in observed_by_evaluation}
        for draw in range(draws):
            permuted = public_exact.copy()
            permuted["target_pic50"] = rng.permutation(labels)
            evaluation_rows: dict[str, list[pd.DataFrame]] = {
                evaluation: [] for evaluation in observed_by_evaluation
            }
            for definition in definitions:
                evaluation = definition["evaluation"]
                train = pd.concat([definition["train"], permuted], ignore_index=True)
                predicted = base._fit_predict(
                    train.assign(sample_weight=1.0),
                    definition["test"],
                    feature_layer=layer,
                    model_name=model_name,
                )
                evaluation_rows[evaluation].append(
                    pd.DataFrame(
                        {
                            "record_id": definition["test"]["record_id"].astype(str),
                            "absolute_error": np.abs(
                                predicted - definition["test"]["target_pic50"].to_numpy(dtype=float)
                            ),
                        }
                    )
                )
            for evaluation, pieces in evaluation_rows.items():
                candidate = pd.concat(pieces, ignore_index=True)
                paired = candidate.merge(
                    baseline_error_by_evaluation[evaluation],
                    on="record_id",
                    validate="one_to_one",
                )
                delta = float((paired["absolute_error"] - paired["baseline_absolute_error"]).mean())
                null_by_evaluation[evaluation].append(delta)
                draw_rows.append(
                    {
                        "feature_layer": layer,
                        "model": model_name,
                        "selection_status": selection_status,
                        "evaluation": evaluation,
                        "draw": draw,
                        "null_mae_delta_vs_matched_internal": delta,
                    }
                )
        for evaluation, values in null_by_evaluation.items():
            null = np.asarray(values, dtype=float)
            observed = observed_by_evaluation[evaluation]
            summary_rows.append(
                {
                    "feature_layer": layer,
                    "model": model_name,
                    "selection_status": selection_status,
                    "evaluation": evaluation,
                    "draws": draws,
                    "observed_mae_delta_vs_matched_internal": observed,
                    "null_mean_delta": float(np.mean(null)),
                    "null_lower_95": float(np.quantile(null, 0.025)),
                    "null_upper_95": float(np.quantile(null, 0.975)),
                    "one_sided_permutation_p_correct_pairing_better": float(
                        (1 + np.sum(null <= observed)) / (draws + 1)
                    ),
                }
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(draw_rows)


def _dataset_summary(
    registry: pd.DataFrame,
    model: pd.DataFrame,
    exact: pd.DataFrame,
    intervals: pd.DataFrame,
    internal: pd.DataFrame,
    extension_test: pd.DataFrame,
) -> pd.DataFrame:
    rows = [
        ("Menin-public structured compounds", int(registry["canonicalized_smiles"].ne("").sum())),
        (
            "Menin-public structured series",
            int(registry.loc[registry["canonicalized_smiles"].ne(""), "series_id"].nunique()),
        ),
        ("Menin-public source-specific model-view rows", int(len(model))),
        ("Primary resolved exact IC50 training compounds", int(len(exact))),
        ("Primary resolved exact + censored IC50 training compounds", int(len(intervals))),
        ("Primary training series", int(intervals["series_id"].nunique())),
        ("Primary exact values from Pfizer series", int(exact["series_id"].eq("SER-PFZ-SPIRO").sum())),
        ("Primary one-sided censored values", int(intervals["target_pic50"].isna().sum())),
        ("Internal exact compounds", int(len(internal))),
        ("Internal scaffolds", int(internal["scaffold"].nunique())),
        ("Fixed Angelo exact test compounds", int(len(extension_test))),
        (
            "Exact structure overlap: public vs internal",
            len(set(exact["standardized_smiles"]) & set(internal["standardized_smiles"])),
        ),
        (
            "Exact structure overlap: public vs Angelo",
            len(set(exact["standardized_smiles"]) & set(extension_test["standardized_smiles"])),
        ),
    ]
    return pd.DataFrame(rows, columns=["quantity", "value"])


def _write_report(
    output: Path,
    metrics: pd.DataFrame,
    comparator_metrics: pd.DataFrame,
    dataset_summary: pd.DataFrame,
    public_exact: pd.DataFrame,
    public_intervals: pd.DataFrame,
    failures: pd.DataFrame,
    reproduction: dict[str, float],
    permutation_summary: pd.DataFrame,
) -> None:
    def row(evaluation: str, regime: str, layer: str, model: str) -> pd.Series:
        match = metrics[
            metrics["evaluation"].eq(evaluation)
            & metrics["data_regime"].eq(regime)
            & metrics["feature_layer"].eq(layer)
            & metrics["model"].eq(model)
        ]
        if match.empty:
            raise ValueError(f"Missing result: {evaluation}/{regime}/{layer}/{model}")
        return match.iloc[0]

    lines = [
        "# Menin-specific public hERG integration",
        "",
        "## Bottom line",
        "",
        "This is an **additional comparison**, not a replacement for the retained internal or broad-public analyses. "
        "The scientifically primary new regime combines internal exact pIC50 observations with all independently usable, "
        "structure-resolved Menin-public IC50 evidence using a one-sided-censoring likelihood.",
        "",
    ]
    for evaluation, title in (
        ("internal_scaffold_cv", "held-out internal scaffold CV"),
        ("angelo_fixed_nonoverlap", "fixed retrospective Angelo non-overlap set"),
    ):
        baseline = row(evaluation, "internal_only_reproduced", "hybrid", "ridge")
        censored_baseline = row(evaluation, "internal_only_censored", "hybrid", "censored_ridge")
        combined_exact = row(evaluation, "internal_plus_menin_public_exact", "hybrid", "ridge")
        combined_censored = row(evaluation, "internal_plus_menin_public_censored", "hybrid", "censored_ridge")
        external_only = row(evaluation, "menin_public_only_censored", "hybrid", "censored_ridge")
        lines.extend(
            [
                f"### {title}",
                "",
                f"- Retained/reproduced internal-only hybrid ridge: MAE **{baseline.mae:.3f}**, "
                f"RMSE {baseline.rmse:.3f}, Spearman {baseline.spearman:.3f}.",
                f"- Matched internal-only censored hybrid ridge control: MAE **{censored_baseline.mae:.3f}**, "
                f"RMSE {censored_baseline.rmse:.3f}, Spearman {censored_baseline.spearman:.3f}.",
                f"- Internal + Menin-public exact-only hybrid ridge: MAE **{combined_exact.mae:.3f}** "
                f"(paired delta {combined_exact.mae_delta_vs_internal:+.3f}; 95% scaffold-bootstrap "
                f"{combined_exact.mae_delta_lower_95:+.3f} to {combined_exact.mae_delta_upper_95:+.3f}).",
                f"- Internal + Menin-public censored hybrid ridge: MAE **{combined_censored.mae:.3f}** "
                f"(paired delta versus the matched censored control {combined_censored.mae_delta_vs_internal:+.3f}; 95% scaffold-bootstrap "
                f"{combined_censored.mae_delta_lower_95:+.3f} to {combined_censored.mae_delta_upper_95:+.3f}).",
                f"- Menin-public-only censored transfer: MAE **{external_only.mae:.3f}**, "
                f"Spearman {external_only.spearman:.3f}. This is a stress test, not a deployable model.",
                "",
            ]
        )
    lines.extend(
        [
            "## What was integrated",
            "",
            f"- {len(public_exact)} primary exact IC50 compounds were usable for ordinary regression.",
            f"- {len(public_intervals)} primary compounds were usable when one-sided censoring was modeled; "
            f"{int(public_intervals['target_pic50'].isna().sum())} of these contribute limits rather than invented point labels.",
            "- Duplicate Revumenib protocol values, concentration-specific inhibition, unresolved structures, and stereo-unresolved "
            "Acerand parents were retained in the audit but excluded from the primary independent training pool.",
            "- The public exact subset is strongly imbalanced: nearly all exact values are from one Pfizer diazaspiro series. "
            "Therefore it cannot validate broad Menin-scaffold generalization on its own.",
            "",
            "## Interpretation rules",
            "",
            "- A lower MAE is encouraging only if the paired scaffold-bootstrap interval supports it; otherwise the result is "
            "compatible with no improvement.",
            "- Public-only performance measures transfer into the internal/Angelo chemistry. It does not establish a universal "
            "hERG predictor because the public training pool is small, series-imbalanced, and protocol heterogeneous.",
            "- The common feature space is deliberately limited to nine interpretable 2D physicochemical proxies and/or an "
            "8-component Morgan latent representation. Uncomputed microstate, conformational, membrane, and receptor-state "
            "physics are not fabricated for these public structures.",
            "- A series-held-out validation within the Menin-public workbook is underidentified: withholding the Pfizer series "
            "removes all but a few censored anchors. That test is rejected rather than reported as a favorable random split.",
            "",
            "## Public-label permutation control",
            "",
        ]
    )
    for record in permutation_summary.itertuples(index=False):
        lines.append(
            f"- {record.evaluation}, {record.feature_layer}/{record.model} ({record.selection_status}): "
            f"observed paired MAE delta {record.observed_mae_delta_vs_matched_internal:+.3f}; "
            f"label-permutation p={record.one_sided_permutation_p_correct_pairing_better:.3f}."
        )
    lines.extend(
        [
            "- This asks whether the correct public structure–hERG pairing helps more than adding the same structures with shuffled "
            "labels. The SVR result remains post-hoc even if its pairing test is favorable.",
            "",
            "## Reproducibility and leakage checks",
            "",
            f"- Maximum absolute difference between the rerun internal-only predictions and retained predictions: "
            f"{reproduction['maximum_absolute_prediction_difference']:.3g} pIC50.",
            f"- Failed model fits: {len(failures)}.",
            "- Exact structures and test scaffolds were excluded from overlap; the source workbook and prior outputs remain unchanged.",
            "",
            "## Decision",
            "",
            "Promotion should be based on the locked hybrid-ridge comparison, not the best row selected after inspection. "
            "The full model matrix is retained to reveal model sensitivity. Any benefit that is small relative to the paired "
            "bootstrap interval is discovery-track evidence only. The Menin-public-only model remains a negative-control/transfer "
            "stress test until more independent Menin series with exact, protocol-compatible hERG curves arrive.",
            "",
        ]
    )
    atomic_write_text(output / "evidence_review.md", "\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--prior", type=Path, default=DEFAULT_PRIOR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    sheets = _read_source(args.source)
    public_exact, public_intervals, audit = _public_frames(sheets)
    conditional_exact, conditional_intervals, _ = _public_frames(sheets, include_conditional=True)
    conditional_ids = set(
        sheets["Model View"]
        .loc[
            sheets["Model View"]["model_role"].isin(CONDITIONAL_ROLES),
            "model_record_id",
        ]
        .astype(str)
    )
    audit["included_conditional_sensitivity"] = audit["model_record_id"].astype(str).isin(conditional_ids)
    internal, internal_intervals, _ = base._internal_herg_frames()
    extension, _ = base._extension_frames(set(internal_intervals["compound_id"].astype(str)))
    prior = pd.read_parquet(args.prior)
    definitions = _fold_definitions(internal, extension, prior)
    fixed_extension_ids = {
        record_id
        for definition in definitions
        if definition["evaluation"] == "angelo_fixed_nonoverlap"
        for record_id in definition["test"]["record_id"].astype(str)
    }
    fixed_extension = extension[extension["record_id"].isin(fixed_extension_ids)].copy()

    internal_canonical = set(internal["standardized_smiles"].map(canonicalize_smiles))
    extension_canonical = set(fixed_extension["standardized_smiles"].map(canonicalize_smiles))
    overlap_internal = public_intervals["standardized_smiles"].isin(internal_canonical)
    overlap_extension = public_intervals["standardized_smiles"].isin(extension_canonical)
    if overlap_internal.any() or overlap_extension.any():
        raise ValueError("Menin-public primary rows overlap an evaluation structure")
    test_scaffolds = set(internal["scaffold"].astype(str)) | set(fixed_extension["scaffold"].astype(str))
    public_scaffold_overlap = public_intervals["scaffold"].astype(str).isin(test_scaffolds)
    if public_scaffold_overlap.any():
        # Excluding these globally is conservative and makes both evaluation
        # comparisons use the same external training pool.
        excluded_ids = set(public_intervals.loc[public_scaffold_overlap, "record_id"].astype(str))
        public_intervals = public_intervals[~public_intervals["record_id"].isin(excluded_ids)].reset_index(
            drop=True
        )
        public_exact = public_exact[~public_exact["record_id"].isin(excluded_ids)].reset_index(drop=True)

    predictions, failures = _run_new_matrix(
        definitions,
        public_exact,
        public_intervals,
    )
    if predictions.empty:
        raise RuntimeError("No comparison predictions were produced")
    metrics = _summarize(predictions)
    prior_predictions, prior_metrics = _prior_comparators(prior)
    permutation_summary, permutation_draws = _exact_label_permutation_test(
        definitions,
        public_exact,
        predictions,
    )

    retained_internal = prior_predictions[prior_predictions["data_regime"].eq("internal_only")][
        ["evaluation", "fold", "record_id", "predicted_pic50"]
    ].rename(columns={"predicted_pic50": "retained_prediction"})
    reproduced = predictions[
        predictions["data_regime"].eq("internal_only_reproduced")
        & predictions["feature_layer"].eq("hybrid")
        & predictions["model"].eq("ridge")
    ][["evaluation", "fold", "record_id", "predicted_pic50"]].rename(
        columns={"predicted_pic50": "reproduced_prediction"}
    )
    reproduction_frame = retained_internal.merge(
        reproduced,
        on=["evaluation", "fold", "record_id"],
        validate="one_to_one",
    )
    reproduction_frame["absolute_difference"] = (
        reproduction_frame["retained_prediction"] - reproduction_frame["reproduced_prediction"]
    ).abs()
    reproduction = {
        "n": int(len(reproduction_frame)),
        "maximum_absolute_prediction_difference": float(reproduction_frame["absolute_difference"].max()),
        "mean_absolute_prediction_difference": float(reproduction_frame["absolute_difference"].mean()),
    }

    dataset_summary = _dataset_summary(
        sheets["Compound Registry"],
        sheets["Model View"],
        public_exact,
        public_intervals,
        internal,
        fixed_extension,
    )
    similarities = pd.DataFrame(
        {
            "record_id": public_intervals["record_id"],
            "compound_id": public_intervals["compound_id"],
            "preferred_name": public_intervals["preferred_name"],
            "series_id": public_intervals["series_id"],
            "pIC50_relation": public_intervals["pIC50_relation"],
            "pic50_lower": public_intervals["pic50_lower"],
            "pic50_upper": public_intervals["pic50_upper"],
            "max_internal_tanimoto": base._max_tanimoto(public_intervals, internal),
            "max_fixed_angelo_tanimoto": base._max_tanimoto(public_intervals, fixed_extension),
        }
    )

    normalized_root = ROOT / "research/data/pk_herg/canonical/menin_specific_public_herg"
    normalized_root.mkdir(parents=True, exist_ok=True)
    atomic_write_parquet(normalized_root / "compound_registry.parquet", sheets["Compound Registry"])
    atomic_write_parquet(normalized_root / "measurements_raw.parquet", sheets["Measurements Raw"])
    atomic_write_parquet(normalized_root / "model_view.parquet", sheets["Model View"])
    atomic_write_parquet(normalized_root / "primary_exact_training.parquet", public_exact)
    atomic_write_parquet(normalized_root / "primary_interval_training.parquet", public_intervals)
    atomic_write_csv(normalized_root / "model_inclusion_audit.csv", audit)

    atomic_write_parquet(args.output / "predictions.parquet", predictions)
    atomic_write_csv(args.output / "predictions_review.csv", predictions.drop(columns=[]))
    atomic_write_csv(args.output / "model_metrics.csv", metrics)
    atomic_write_csv(args.output / "retained_comparator_metrics.csv", prior_metrics)
    atomic_write_csv(args.output / "dataset_summary.csv", dataset_summary)
    atomic_write_csv(args.output / "public_similarity_domain.csv", similarities)
    atomic_write_csv(args.output / "exact_label_permutation_summary.csv", permutation_summary)
    atomic_write_parquet(args.output / "exact_label_permutation_draws.parquet", permutation_draws)
    atomic_write_csv(args.output / "model_failures.csv", failures)
    atomic_write_csv(args.output / "reproduction_check.csv", reproduction_frame)
    atomic_write_json(args.output / "reproduction_check.json", reproduction)
    atomic_write_json(
        args.output / "analysis_manifest.json",
        {
            "source_workbook": str(args.source.relative_to(ROOT)),
            "prior_predictions": str(args.prior.relative_to(ROOT)),
            "primary_exact_public_n": int(len(public_exact)),
            "primary_interval_public_n": int(len(public_intervals)),
            "primary_plus_conditional_exact_public_n": int(len(conditional_exact)),
            "primary_plus_conditional_interval_public_n": int(len(conditional_intervals)),
            "ridge_alpha_locked": RIDGE_ALPHA,
            "exact_label_permutation_draws": PERMUTATION_DRAWS,
            "feature_layers": list(FEATURE_LAYERS),
            "exact_models": list(EXACT_MODELS),
            "fixed_evaluations": sorted(predictions["evaluation"].unique()),
            "excluded_scaffold_overlap_n": int(public_scaffold_overlap.sum()),
            "primary_roles": sorted(PRIMARY_ROLES),
            "conditional_roles": sorted(CONDITIONAL_ROLES),
            "interpretation": (
                "Additional comparison only; no retained model or prior output was overwritten."
            ),
        },
    )
    locked_new = metrics[
        metrics["feature_layer"].eq("hybrid")
        & (
            (
                metrics["model"].eq("ridge")
                & metrics["data_regime"].isin(
                    {
                        "internal_only_reproduced",
                        "internal_plus_menin_public_exact",
                        "menin_public_only_exact",
                    }
                )
            )
            | (
                metrics["model"].eq("censored_ridge")
                & metrics["data_regime"].isin(
                    {
                        "internal_only_censored",
                        "internal_plus_menin_public_censored",
                        "menin_public_only_censored",
                    }
                )
            )
        )
    ].copy()
    locked_new["comparison_regime"] = locked_new["data_regime"]
    prior_display = prior_metrics.copy()
    for evaluation in prior_display["evaluation"].unique():
        baseline_mae = float(
            prior_display.loc[
                prior_display["evaluation"].eq(evaluation)
                & prior_display["comparison_regime"].eq("retained_internal_only"),
                "mae",
            ].iloc[0]
        )
        prior_display.loc[prior_display["evaluation"].eq(evaluation), "mae_delta_vs_internal"] = (
            prior_display.loc[prior_display["evaluation"].eq(evaluation), "mae"] - baseline_mae
        )
    workbook_payload = {
        "dataset_summary": json.loads(dataset_summary.to_json(orient="records")),
        "locked_new_results": json.loads(locked_new.to_json(orient="records")),
        "retained_comparators": json.loads(prior_display.to_json(orient="records")),
        "full_model_matrix": json.loads(metrics.to_json(orient="records")),
        "public_evidence": json.loads(similarities.to_json(orient="records")),
        "permutation_summary": json.loads(permutation_summary.to_json(orient="records")),
        "reproduction": [reproduction],
        "inclusion_audit": json.loads(audit.to_json(orient="records")),
        "interpretation": [
            {
                "finding": "Primary locked exact integration",
                "interpretation": "No reliable gain: hybrid-ridge MAE is essentially unchanged on both fixed evaluations.",
                "status": "Do not promote",
            },
            {
                "finding": "Censored Menin-public integration",
                "interpretation": "Small internal-CV improvement versus the matched censored estimator, but its interval crosses zero; no improvement on the fixed Angelo set.",
                "status": "Discovery track",
            },
            {
                "finding": "Post-hoc SVR improvement",
                "interpretation": "The same apparent gain occurs after public labels are shuffled, so it is representation/regularization rather than learned Menin-specific hERG biology.",
                "status": "Negative control; not mechanistic evidence",
            },
            {
                "finding": "Menin-public-only transfer",
                "interpretation": "Fails strongly because nine exact labels come from one series, public/internal similarity is low, and the public label distribution is shifted.",
                "status": "Not deployable",
            },
        ],
    }
    atomic_write_json(args.output / "workbook_payload.json", workbook_payload)
    _write_report(
        args.output,
        metrics,
        prior_metrics,
        dataset_summary,
        public_exact,
        public_intervals,
        failures,
        reproduction,
        permutation_summary,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "predictions": len(predictions),
                "metrics": len(metrics),
                "failures": len(failures),
                "primary_exact_public": len(public_exact),
                "primary_interval_public": len(public_intervals),
                "reproduction_max_abs_diff": reproduction["maximum_absolute_prediction_difference"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
