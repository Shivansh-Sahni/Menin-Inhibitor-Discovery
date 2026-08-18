#!/usr/bin/env python3
"""Leakage-safe hERG evaluation using old internal plus Angelo data.

This analysis preserves the fixed Angelo extension benchmark while also fitting
and evaluating a combined same-series model. Repeated slide/deck mirrors are
collapsed at the standardized-structure level; incompatible source labels are
quarantined rather than averaged into a fictitious measurement.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

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
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
CANONICAL = ROOT / "research/data/pk_herg/canonical"
SOURCE = CANONICAL / "ascentage_herg_2026_07_28/normalized_records.parquet"
RECOVERY = ROOT / "research/reports/pk_herg/ascentage_herg_extension/predictions.parquet"
EXISTING = ROOT / "research/reports/pk_herg/ascentage_herg_extension/complete_feature_model"
OUTPUT = ROOT / "research/reports/pk_herg/combined_internal_angelo"
SEED = 20260803
ALPHA = 3.0
COMPONENTS = 8
FOLDS = 5
DISCORDANT_EXACT_SPREAD = 0.50
BOOTSTRAPS = 5000


def _fingerprints(smiles: pd.Series) -> np.ndarray:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    rows: list[np.ndarray] = []
    for value in smiles.astype(str):
        molecule = Chem.MolFromSmiles(value)
        if molecule is None:
            raise ValueError(f"Invalid standardized SMILES: {value}")
        rows.append(generator.GetFingerprintAsNumPy(molecule).astype(float))
    return np.asarray(rows)


def _relation(lower: float, upper: float) -> str:
    if np.isfinite(lower) and np.isfinite(upper) and np.isclose(lower, upper):
        return "="
    if not np.isfinite(lower) and np.isfinite(upper):
        return "<"
    if np.isfinite(lower) and not np.isfinite(upper):
        return ">"
    return "interval"


def _collapse_internal(
    potency: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    retained: list[pd.Series] = []
    audit_rows: list[dict[str, Any]] = []
    ambiguous_rows: list[dict[str, Any]] = []
    for compound_id, group in potency.groupby("compound_id", sort=True):
        intervals = (
            group[["pic50_lower", "pic50_upper"]]
            .assign(
                lower_key=lambda frame: frame["pic50_lower"].round(6),
                upper_key=lambda frame: frame["pic50_upper"].round(6),
            )
            .drop_duplicates(["lower_key", "upper_key"])
            .drop(columns=["lower_key", "upper_key"])
            .reset_index(drop=True)
        )
        exact = intervals[
            np.isfinite(intervals["pic50_lower"])
            & np.isfinite(intervals["pic50_upper"])
            & np.isclose(intervals["pic50_lower"], intervals["pic50_upper"])
        ]
        censored = intervals.drop(index=exact.index)
        reason = ""
        chosen_lower = float("nan")
        chosen_upper = float("nan")
        exact_spread = float("nan")
        if not exact.empty:
            exact_values = exact["pic50_lower"].to_numpy(dtype=float)
            exact_spread = float(np.max(exact_values) - np.min(exact_values))
            chosen = float(np.median(exact_values))
            incompatible = any(
                (np.isfinite(row.pic50_lower) and chosen < row.pic50_lower - 1e-8)
                or (np.isfinite(row.pic50_upper) and chosen > row.pic50_upper + 1e-8)
                for row in censored.itertuples()
            )
            if incompatible:
                reason = "exact measurement conflicts with a censored source record"
            elif exact_spread > DISCORDANT_EXACT_SPREAD:
                reason = (
                    f"unique exact measurements span {exact_spread:.3f} log unit, "
                    f"above the locked {DISCORDANT_EXACT_SPREAD:.2f} threshold"
                )
            else:
                chosen_lower = chosen
                chosen_upper = chosen
        else:
            finite_lower = intervals.loc[np.isfinite(intervals["pic50_lower"]), "pic50_lower"]
            finite_upper = intervals.loc[np.isfinite(intervals["pic50_upper"]), "pic50_upper"]
            chosen_lower = float(finite_lower.max()) if not finite_lower.empty else -np.inf
            chosen_upper = float(finite_upper.min()) if not finite_upper.empty else np.inf
            if chosen_lower > chosen_upper:
                reason = "censored source intervals have an empty intersection"

        base = group.iloc[0].copy()
        source_values = sorted(
            {f"{row.relation}{float(row.value):g} uM @ {row.source_locator}" for row in group.itertuples()}
        )
        audit = {
            "compound_id": compound_id,
            "display_name": str(base.get("display_name", compound_id)),
            "scaffold": str(base["scaffold"]),
            "raw_measurement_rows": int(len(group)),
            "unique_interval_records": int(len(intervals)),
            "unique_exact_records": int(len(exact)),
            "unique_censored_records": int(len(censored)),
            "unique_exact_spread_pic50": exact_spread,
            "source_values": " | ".join(source_values),
            "resolution": "quarantined" if reason else "retained_once_per_structure",
            "resolution_reason": reason or "compatible source evidence collapsed without duplicate weighting",
        }
        audit_rows.append(audit)
        if reason:
            ambiguous_rows.append(audit)
            continue
        base["pic50_lower"] = chosen_lower
        base["pic50_upper"] = chosen_upper
        base["observed_pic50"] = (
            chosen_lower if np.isfinite(chosen_lower) and np.isclose(chosen_lower, chosen_upper) else np.nan
        )
        base["source_group"] = "old_internal"
        base["source_record_count"] = int(len(group))
        base["unique_interval_count"] = int(len(intervals))
        base["label_resolution"] = "structure_level_consensus"
        retained.append(base)

    clean = pd.DataFrame(retained).reset_index(drop=True)
    if clean["compound_id"].duplicated().any():
        raise ValueError("Collapsed internal data still contain duplicate compound IDs")
    return clean, pd.DataFrame(audit_rows), pd.DataFrame(ambiguous_rows)


def _prepare_data() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    list[str],
]:
    tables = load_canonical_tables(CANONICAL)
    compounds = compound_model_frame(tables["compounds"], tables.get("compound_aliases"))
    base_features, layers = merge_feature_layers(compounds)
    _, potency, _ = prepare_herg_evidence(compounds, tables["measurements"], base_features)
    controls = [
        column for column in layers["structure_2d"] if column in potency and potency[column].notna().any()
    ]
    if len(controls) != 9:
        raise ValueError(f"Expected nine audited controls; found {controls}")
    internal, internal_audit, ambiguous = _collapse_internal(potency)

    source = load_ascentage_source(SOURCE, recovery_artifact=RECOVERY)
    old_structure_ids = set(compounds["structure_id"].astype(str))
    overlap = source[source["structure_id"].isin(old_structure_ids)].copy()
    novel = source[~source["structure_id"].isin(old_structure_ids)].copy()
    measured = novel[
        novel["herg_ic50_censoring"].ne("missing") & novel["synthesis_status"].eq("synthesized_by_cro")
    ].copy()
    extension_features = structure_feature_frame(
        pd.DataFrame(
            {
                "compound_id": measured["structure_id"],
                "standardized_smiles": measured["standardized_smiles"],
            }
        )
    )
    measured = measured.merge(
        extension_features[["compound_id", *controls]],
        left_on="structure_id",
        right_on="compound_id",
        validate="one_to_one",
    )
    measured["compound_id"] = measured["structure_id"]
    measured["pic50_lower"] = measured["herg_pic50_lower_bound"]
    measured["pic50_upper"] = measured["herg_pic50_upper_bound"]
    measured["observed_pic50"] = measured["herg_pic50_value"].where(measured["herg_pic50_relation"].eq("="))
    measured["source_group"] = "angelo_new"
    measured["source_record_count"] = 1
    measured["unique_interval_count"] = 1
    measured["label_resolution"] = "single_source_record"
    combined = pd.concat([internal, measured], ignore_index=True, sort=False)
    if combined["compound_id"].duplicated().any():
        raise ValueError("Combined clean data contain duplicate structure-level IDs")
    return compounds, internal, measured, combined, overlap, internal_audit, ambiguous, controls


def _fold_project(
    train_bits: np.ndarray,
    train_controls: np.ndarray,
    test_bits: np.ndarray,
    test_controls: np.ndarray,
    *,
    model: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    medians = np.nanmedian(train_controls, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    train_controls = np.where(np.isfinite(train_controls), train_controls, medians)
    test_controls = np.where(np.isfinite(test_controls), test_controls, medians)
    if model == "compact_global_controls":
        scaler = StandardScaler()
        return scaler.fit_transform(train_controls), scaler.transform(test_controls), 0.0
    if model != "hybrid_ecfp8":
        raise ValueError(f"Unknown model: {model}")
    reducer = TruncatedSVD(n_components=COMPONENTS, random_state=seed)
    train_latent = reducer.fit_transform(train_bits)
    test_latent = reducer.transform(test_bits)
    scaler = StandardScaler()
    train = scaler.fit_transform(np.column_stack([train_latent, train_controls]))
    test = scaler.transform(np.column_stack([test_latent, test_controls]))
    return train, test, float(reducer.explained_variance_ratio_.sum())


def _max_similarity(test_bits: np.ndarray, train_bits: np.ndarray) -> np.ndarray:
    train_unique = np.unique(train_bits, axis=0)
    intersection = test_bits @ train_unique.T
    denominator = (
        test_bits.sum(axis=1, keepdims=True) + train_unique.sum(axis=1, keepdims=True).T - intersection
    )
    similarity = np.divide(
        intersection,
        denominator,
        out=np.zeros_like(intersection, dtype=float),
        where=denominator > 0,
    )
    return similarity.max(axis=1)


def _nll(lower: np.ndarray, upper: np.ndarray, prediction: np.ndarray, sigma: np.ndarray) -> float:
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


def _metrics(frame: pd.DataFrame) -> dict[str, Any]:
    lower = frame["pic50_lower"].to_numpy(dtype=float)
    upper = frame["pic50_upper"].to_numpy(dtype=float)
    prediction = frame["predicted_pic50"].to_numpy(dtype=float)
    sigma = frame["predictive_sigma"].to_numpy(dtype=float)
    exact = np.isfinite(lower) & np.isfinite(upper) & np.isclose(lower, upper)
    observed = lower[exact]
    predicted = prediction[exact]
    error = predicted - observed
    decisive = (lower >= 5.0) | (upper < 5.0)
    observed_class = (lower[decisive] >= 5.0).astype(int)
    probability = norm.sf((5.0 - prediction[decisive]) / np.maximum(sigma[decisive], 1e-6))
    predicted_class = (prediction[decisive] >= 5.0).astype(int)
    classification: dict[str, Any] = {
        "n_decisive": int(decisive.sum()),
        "n_blockers": int(observed_class.sum()),
        "n_nonblockers": int((1 - observed_class).sum()),
    }
    if len(np.unique(observed_class)) == 2:
        classification.update(
            {
                "roc_auc": float(roc_auc_score(observed_class, probability)),
                "pr_auc": float(average_precision_score(observed_class, probability)),
                "balanced_accuracy": float(balanced_accuracy_score(observed_class, predicted_class)),
                "mcc": float(matthews_corrcoef(observed_class, predicted_class)),
                "brier": float(brier_score_loss(observed_class, probability)),
            }
        )
    return {
        "n_rows": int(len(frame)),
        "n_structures": int(frame["compound_id"].nunique()),
        "n_scaffolds": int(frame["scaffold"].nunique()),
        "n_exact": int(exact.sum()),
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
        "classification_at_pic50_5": classification,
    }


def _oof(
    frame: pd.DataFrame,
    controls: list[str],
    *,
    model: str,
    evaluation: str,
    splitter: Iterable[tuple[np.ndarray, np.ndarray]],
) -> tuple[dict[str, Any], pd.DataFrame]:
    bits = _fingerprints(frame["standardized_smiles"])
    raw_controls = frame[controls].to_numpy(dtype=float)
    lower = frame["pic50_lower"].to_numpy(dtype=float)
    upper = frame["pic50_upper"].to_numpy(dtype=float)
    prediction = np.full(len(frame), np.nan)
    sigma = np.full(len(frame), np.nan)
    fold = np.full(len(frame), -1, dtype=int)
    convergence = np.zeros(len(frame), dtype=bool)
    explained = np.zeros(len(frame))
    similarity = np.zeros(len(frame))
    for fold_id, (train_index, test_index) in enumerate(splitter):
        train, test, variance = _fold_project(
            bits[train_index],
            raw_controls[train_index],
            bits[test_index],
            raw_controls[test_index],
            model=model,
            seed=SEED + fold_id,
        )
        fitted = CensoredGaussianRidge(alpha=ALPHA, maxiter=5000).fit(
            train,
            lower[train_index],
            upper[train_index],
        )
        prediction[test_index] = fitted.predict(test)
        sigma[test_index] = fitted.sigma_
        fold[test_index] = fold_id
        convergence[test_index] = fitted.converged_
        explained[test_index] = variance
        similarity[test_index] = _max_similarity(bits[test_index], bits[train_index])
    if np.isnan(prediction).any() or (fold < 0).any():
        raise RuntimeError(f"{evaluation}/{model} did not produce exactly one OOF prediction per row")
    result = frame[
        [
            "compound_id",
            "standardized_smiles",
            "scaffold",
            "source_group",
            "pic50_lower",
            "pic50_upper",
        ]
    ].copy()
    result["evaluation"] = evaluation
    result["model"] = model
    result["fold"] = fold
    result["predicted_pic50"] = prediction
    result["predictive_sigma"] = sigma
    result["fit_converged"] = convergence
    result["max_train_tanimoto"] = similarity
    result["explained_fingerprint_variance"] = explained
    metrics = {
        "evaluation": evaluation,
        "model": model,
        "all_folds_converged": bool(convergence.all()),
        "mean_max_train_tanimoto": float(np.mean(similarity)),
        "median_max_train_tanimoto": float(np.median(similarity)),
        "mean_fold_explained_fingerprint_variance": float(np.mean(explained)),
        **_metrics(result),
    }
    return metrics, result


def _fixed_score(
    training: pd.DataFrame,
    scoring: pd.DataFrame,
    controls: list[str],
    training_oof: pd.DataFrame,
    *,
    model: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    train_bits = _fingerprints(training["standardized_smiles"])
    score_bits = _fingerprints(scoring["standardized_smiles"])
    train_controls = training[controls].to_numpy(dtype=float)
    score_controls = scoring[controls].to_numpy(dtype=float)
    train, score, variance = _fold_project(
        train_bits,
        train_controls,
        score_bits,
        score_controls,
        model=model,
        seed=SEED,
    )
    fitted = CensoredGaussianRidge(alpha=ALPHA, maxiter=5000).fit(
        train,
        training["pic50_lower"].to_numpy(dtype=float),
        training["pic50_upper"].to_numpy(dtype=float),
    )
    exact_oof = np.isfinite(training_oof["pic50_lower"]) & np.isclose(
        training_oof["pic50_lower"], training_oof["pic50_upper"]
    )
    residual = training_oof.loc[exact_oof, "predicted_pic50"].to_numpy(dtype=float) - training_oof.loc[
        exact_oof, "pic50_lower"
    ].to_numpy(dtype=float)
    predictive_sigma = max(float(np.std(residual, ddof=1)), 1e-6)
    result = scoring[
        [
            "compound_id",
            "standardized_smiles",
            "scaffold",
            "source_group",
            "pic50_lower",
            "pic50_upper",
        ]
    ].copy()
    result["evaluation"] = "old_to_fixed_angelo"
    result["model"] = model
    result["fold"] = -1
    result["predicted_pic50"] = fitted.predict(score)
    result["predictive_sigma"] = predictive_sigma
    result["fit_converged"] = fitted.converged_
    result["max_train_tanimoto"] = _max_similarity(score_bits, train_bits)
    result["explained_fingerprint_variance"] = variance
    metrics = {
        "evaluation": "old_to_fixed_angelo",
        "model": model,
        "all_folds_converged": bool(fitted.converged_),
        "mean_max_train_tanimoto": float(result["max_train_tanimoto"].mean()),
        "median_max_train_tanimoto": float(result["max_train_tanimoto"].median()),
        "mean_fold_explained_fingerprint_variance": variance,
        **_metrics(result),
    }
    return metrics, result


def _bootstrap_model_difference(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    evaluation: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    keys = ["compound_id", "scaffold", "pic50_lower", "pic50_upper"]
    merged = left[keys + ["predicted_pic50"]].merge(
        right[keys + ["predicted_pic50"]],
        on=keys,
        suffixes=("_hybrid", "_controls"),
        validate="one_to_one",
    )
    exact = merged[
        np.isfinite(merged["pic50_lower"]) & np.isclose(merged["pic50_lower"], merged["pic50_upper"])
    ].copy()
    scaffolds = exact["scaffold"].drop_duplicates().to_numpy()
    rng = np.random.default_rng(SEED + 991)
    rows: list[dict[str, float]] = []
    for replicate in range(BOOTSTRAPS):
        sampled = rng.choice(scaffolds, size=len(scaffolds), replace=True)
        pieces = [exact[exact["scaffold"].eq(scaffold)] for scaffold in sampled]
        boot = pd.concat(pieces, ignore_index=True)
        observed = boot["pic50_lower"].to_numpy(dtype=float)
        hybrid = np.abs(boot["predicted_pic50_hybrid"].to_numpy(dtype=float) - observed).mean()
        controls = np.abs(boot["predicted_pic50_controls"].to_numpy(dtype=float) - observed).mean()
        rows.append(
            {
                "replicate": replicate,
                "hybrid_mae": float(hybrid),
                "controls_mae": float(controls),
                "hybrid_minus_controls_mae": float(hybrid - controls),
            }
        )
    frame = pd.DataFrame(rows)
    low, high = np.quantile(frame["hybrid_minus_controls_mae"], [0.025, 0.975])
    summary = {
        "evaluation": evaluation,
        "bootstrap_unit": "Bemis-Murcko scaffold",
        "replicates": BOOTSTRAPS,
        "hybrid_minus_controls_mae_mean": float(frame["hybrid_minus_controls_mae"].mean()),
        "hybrid_minus_controls_mae_95_lower": float(low),
        "hybrid_minus_controls_mae_95_upper": float(high),
        "hybrid_significantly_better": bool(high < 0.0),
    }
    frame["evaluation"] = evaluation
    return frame, summary


def _bootstrap_absolute_mae(predictions: pd.DataFrame) -> pd.DataFrame:
    """Scaffold-bootstrap uncertainty for each reported exact-outcome MAE."""
    rng = np.random.default_rng(SEED + 1777)
    rows: list[dict[str, Any]] = []
    for (evaluation, model), frame in predictions.groupby(["evaluation", "model"], sort=True):
        exact = frame[
            np.isfinite(frame["pic50_lower"]) & np.isclose(frame["pic50_lower"], frame["pic50_upper"])
        ].copy()
        exact["absolute_error"] = np.abs(exact["predicted_pic50"] - exact["pic50_lower"])
        scaffold_errors = {
            scaffold: group["absolute_error"].to_numpy(dtype=float)
            for scaffold, group in exact.groupby("scaffold", sort=False)
        }
        scaffolds = np.asarray(list(scaffold_errors))
        estimates = np.zeros(BOOTSTRAPS)
        for replicate in range(BOOTSTRAPS):
            sampled = rng.choice(scaffolds, size=len(scaffolds), replace=True)
            estimates[replicate] = np.concatenate([scaffold_errors[scaffold] for scaffold in sampled]).mean()
        low, high = np.quantile(estimates, [0.025, 0.975])
        rows.append(
            {
                "evaluation": evaluation,
                "model": model,
                "pic50_mae_scaffold_bootstrap_95_lower": float(low),
                "pic50_mae_scaffold_bootstrap_95_upper": float(high),
                "pic50_mae_scaffold_bootstrap_replicates": BOOTSTRAPS,
            }
        )
    return pd.DataFrame(rows)


def _source_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (evaluation, model, source), frame in predictions.groupby(
        ["evaluation", "model", "source_group"], sort=True
    ):
        if evaluation not in {"combined_scaffold_cv", "combined_leave_one_scaffold_out"}:
            continue
        rows.append(
            {
                "evaluation": evaluation,
                "model": model,
                "source_group": source,
                **_metrics(frame),
            }
        )
    return pd.json_normalize(rows, sep=".")


def _equal_weight_ensembles(
    predictions: pd.DataFrame,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    """Combine the two locked models without tuning against evaluation outcomes."""
    rows: list[pd.DataFrame] = []
    metrics: list[dict[str, Any]] = []
    join_keys = [
        "compound_id",
        "standardized_smiles",
        "scaffold",
        "source_group",
        "pic50_lower",
        "pic50_upper",
        "evaluation",
        "fold",
    ]
    for evaluation in predictions["evaluation"].drop_duplicates():
        controls = predictions[
            predictions["evaluation"].eq(evaluation) & predictions["model"].eq("compact_global_controls")
        ]
        hybrid = predictions[
            predictions["evaluation"].eq(evaluation) & predictions["model"].eq("hybrid_ecfp8")
        ]
        merged = controls.merge(
            hybrid,
            on=join_keys,
            suffixes=("_controls", "_hybrid"),
            validate="one_to_one",
        )
        ensemble = merged[join_keys].copy()
        ensemble["model"] = "equal_weight_ensemble"
        ensemble["predicted_pic50"] = 0.5 * (
            merged["predicted_pic50_controls"] + merged["predicted_pic50_hybrid"]
        )
        # Moment-matched variance of an equal mixture: within-model variance plus
        # model-form disagreement. This is fixed in advance and uses no test labels.
        mean = ensemble["predicted_pic50"].to_numpy(dtype=float)
        control_mean = merged["predicted_pic50_controls"].to_numpy(dtype=float)
        hybrid_mean = merged["predicted_pic50_hybrid"].to_numpy(dtype=float)
        variance = 0.5 * (
            merged["predictive_sigma_controls"].to_numpy(dtype=float) ** 2 + (control_mean - mean) ** 2
        ) + 0.5 * (merged["predictive_sigma_hybrid"].to_numpy(dtype=float) ** 2 + (hybrid_mean - mean) ** 2)
        ensemble["predictive_sigma"] = np.sqrt(np.maximum(variance, 1e-12))
        ensemble["fit_converged"] = merged["fit_converged_controls"] & merged["fit_converged_hybrid"]
        ensemble["max_train_tanimoto"] = merged["max_train_tanimoto_controls"]
        ensemble["explained_fingerprint_variance"] = merged["explained_fingerprint_variance_hybrid"]
        rows.append(ensemble)
        metrics.append(
            {
                "evaluation": evaluation,
                "model": "equal_weight_ensemble",
                "all_folds_converged": bool(ensemble["fit_converged"].all()),
                "mean_max_train_tanimoto": float(ensemble["max_train_tanimoto"].mean()),
                "median_max_train_tanimoto": float(ensemble["max_train_tanimoto"].median()),
                "mean_fold_explained_fingerprint_variance": float(
                    ensemble["explained_fingerprint_variance"].mean()
                ),
                **_metrics(ensemble),
            }
        )
    return metrics, pd.concat(rows, ignore_index=True)


def _overlap_audit(
    overlap: pd.DataFrame,
    compounds: pd.DataFrame,
    internal_audit: pd.DataFrame,
) -> pd.DataFrame:
    mapping = compounds[["compound_id", "structure_id"]].drop_duplicates("structure_id")
    audit = overlap.merge(mapping, on="structure_id", how="left", validate="one_to_one")
    audit = audit.merge(
        internal_audit[["compound_id", "source_values", "resolution", "resolution_reason"]],
        on="compound_id",
        how="left",
        validate="one_to_one",
    )
    audit["overlap_role"] = "duplicate_source_mirror_excluded_from_combined_training"
    return audit[
        [
            "internal_id",
            "structure_id",
            "compound_id",
            "herg_ic50_relation",
            "herg_ic50_value_um",
            "herg_pic50_value",
            "source_values",
            "resolution",
            "resolution_reason",
            "overlap_role",
        ]
    ]


def _report(
    metrics: pd.DataFrame,
    source_metrics: pd.DataFrame,
    bootstraps: pd.DataFrame,
    counts: dict[str, Any],
) -> str:
    def row(evaluation: str, model: str) -> pd.Series:
        return metrics[(metrics["evaluation"] == evaluation) & (metrics["model"] == model)].iloc[0]

    old_fixed = row("old_to_fixed_angelo", "hybrid_ecfp8")
    combined = row("combined_scaffold_cv", "hybrid_ecfp8")
    combined_loso = row("combined_leave_one_scaffold_out", "hybrid_ecfp8")
    ensemble_fixed = row("old_to_fixed_angelo", "equal_weight_ensemble")
    ensemble_old = row("old_internal_scaffold_cv", "equal_weight_ensemble")
    ensemble_combined = row("combined_scaffold_cv", "equal_weight_ensemble")
    ensemble_loso = row("combined_leave_one_scaffold_out", "equal_weight_ensemble")
    delta = bootstraps[bootstraps["evaluation"].eq("combined_scaffold_cv")].iloc[0]
    return f"""# Combined old-internal + Angelo hERG model

## Answer

Yes. The candidate combined lead is an untuned equal-weight ensemble of two censored
Gaussian ridge models trained from every nonduplicative, compatible hERG record in the old
internal workbook and new Angelo set. One uses nine compact global physicochemical controls;
the other adds eight fold-fitted ECFP4 latent structure components. These are interpretable
or associative pre-HPC features, not claimed as fundamental molecular physics.

## Data boundary

- Old workbook: 111 submitted compound rows representing
  {counts["old_workbook_unique_structures"]} unique structures; {counts["old_workbook_structures"]}
  structures have hERG IC50 evidence before QC.
- Angelo document: {counts["angelo_total_structures"]} structures, including
  {counts["angelo_overlap_structures"]} exact structure overlaps already represented in the
  old workbook, {counts["angelo_novel_measured_structures"]} genuinely new measured structures,
  and {counts["angelo_novel_missing_structures"]} unmeasured structures.
- The 22 overlaps reproduce already integrated values and are not double-weighted.
- Five old structures have incompatible protocol/source labels and remain visible in the
  quarantine audit rather than being converted into artificial averages.
- Primary combined training/evaluation therefore contains {counts["combined_clean_structures"]}
  unique structures: {counts["old_clean_structures"]} old internal plus
  {counts["angelo_novel_measured_structures"]} new Angelo measurements.

## Results

| Question | Validation | Exact n | MAE (95% scaffold bootstrap) | RMSE | Spearman | Within 0.5 log |
|---|---|---:|---:|---:|---:|---:|
| How stable is the old internal model across its own scaffolds? | Old-internal five-fold scaffold CV | {int(ensemble_old["n_exact"])} | {ensemble_old["pic50_mae"]:.3f} ({ensemble_old["pic50_mae_scaffold_bootstrap_95_lower"]:.3f}–{ensemble_old["pic50_mae_scaffold_bootstrap_95_upper"]:.3f}) | {ensemble_old["pic50_rmse"]:.3f} | {ensemble_old["spearman"]:.3f} | {ensemble_old["fraction_within_0p5_log"]:.1%} |
| Can old data predict the fixed new Angelo set? | Old-only train; Angelo never trained | {int(ensemble_fixed["n_exact"])} | {ensemble_fixed["pic50_mae"]:.3f} ({ensemble_fixed["pic50_mae_scaffold_bootstrap_95_lower"]:.3f}–{ensemble_fixed["pic50_mae_scaffold_bootstrap_95_upper"]:.3f}) | {ensemble_fixed["pic50_rmse"]:.3f} | {ensemble_fixed["spearman"]:.3f} | {ensemble_fixed["fraction_within_0p5_log"]:.1%} |
| How does one model trained from both sources generalize across held-out scaffolds? | Five-fold scaffold CV on combined set | {int(ensemble_combined["n_exact"])} | {ensemble_combined["pic50_mae"]:.3f} ({ensemble_combined["pic50_mae_scaffold_bootstrap_95_lower"]:.3f}–{ensemble_combined["pic50_mae_scaffold_bootstrap_95_upper"]:.3f}) | {ensemble_combined["pic50_rmse"]:.3f} | {ensemble_combined["spearman"]:.3f} | {ensemble_combined["fraction_within_0p5_log"]:.1%} |
| Does the result survive the stricter one-scaffold-at-a-time test? | Leave-one-scaffold-out | {int(ensemble_loso["n_exact"])} | {ensemble_loso["pic50_mae"]:.3f} ({ensemble_loso["pic50_mae_scaffold_bootstrap_95_lower"]:.3f}–{ensemble_loso["pic50_mae_scaffold_bootstrap_95_upper"]:.3f}) | {ensemble_loso["pic50_rmse"]:.3f} | {ensemble_loso["spearman"]:.3f} | {ensemble_loso["fraction_within_0p5_log"]:.1%} |

The table reports the untuned equal-weight ensemble. Its uncertainty includes both fitted
residual variance and disagreement between the two model forms. The standalone hybrid gives
MAE {old_fixed["pic50_mae"]:.3f}, {combined["pic50_mae"]:.3f}, and
{combined_loso["pic50_mae"]:.3f}, respectively, while providing the strongest continuous
ranking signal in the combined five-fold analysis (Spearman {combined["spearman"]:.3f}).

For combined five-fold scaffold CV, the hybrid-minus-control MAE scaffold-bootstrap interval is
[{delta["hybrid_minus_controls_mae_95_lower"]:.3f}, {delta["hybrid_minus_controls_mae_95_upper"]:.3f}].
This is the appropriate test of whether local structural identity adds reproducible value beyond
the compact global controls.

## Interpretation

The fixed Angelo result remains the cleanest retrospective transfer test because its outcomes
never enter training. The combined cross-validation result answers a different question: how
well a final same-series model using all available unique measurements may interpolate when an
entire scaffold is withheld. It is not an independent external validation, because Angelo data
participate in the other folds. A future frozen batch or a genuinely different Menin series is
still required before promotion to the decision track.

The final model can be refit on all {counts["combined_clean_structures"]} clean unique structures
for a new similar Menin inhibitor. It should output continuous pIC50, equivalent hERG IC50,
blocker probability at the 10 uM threshold, OOF-derived residual/model-form uncertainty, nearest-training
similarity, scaffold support, and an explicit discovery-only/domain flag.
"""


def run() -> dict[str, Any]:
    (
        compounds,
        internal,
        extension,
        combined,
        overlap,
        internal_audit,
        ambiguous,
        controls,
    ) = _prepare_data()
    metric_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    old_oof_by_model: dict[str, pd.DataFrame] = {}

    for model in ("compact_global_controls", "hybrid_ecfp8"):
        old_splits = list(GroupKFold(FOLDS).split(internal, groups=internal["scaffold"].astype(str)))
        old_metrics, old_oof = _oof(
            internal,
            controls,
            model=model,
            evaluation="old_internal_scaffold_cv",
            splitter=old_splits,
        )
        metric_rows.append(old_metrics)
        prediction_frames.append(old_oof)
        old_oof_by_model[model] = old_oof

        fixed_metrics, fixed = _fixed_score(
            internal,
            extension,
            controls,
            old_oof,
            model=model,
        )
        metric_rows.append(fixed_metrics)
        prediction_frames.append(fixed)

        combined_splits = list(GroupKFold(FOLDS).split(combined, groups=combined["scaffold"].astype(str)))
        combined_metrics, combined_oof = _oof(
            combined,
            controls,
            model=model,
            evaluation="combined_scaffold_cv",
            splitter=combined_splits,
        )
        metric_rows.append(combined_metrics)
        prediction_frames.append(combined_oof)

        loso_splits = list(LeaveOneGroupOut().split(combined, groups=combined["scaffold"].astype(str)))
        loso_metrics, loso = _oof(
            combined,
            controls,
            model=model,
            evaluation="combined_leave_one_scaffold_out",
            splitter=loso_splits,
        )
        metric_rows.append(loso_metrics)
        prediction_frames.append(loso)

    metrics = pd.json_normalize(metric_rows, sep=".")
    predictions = pd.concat(prediction_frames, ignore_index=True)
    ensemble_metrics, ensemble_predictions = _equal_weight_ensembles(predictions)
    metric_rows.extend(ensemble_metrics)
    predictions = pd.concat([predictions, ensemble_predictions], ignore_index=True)
    metrics = pd.json_normalize(metric_rows, sep=".")
    absolute_mae_bootstrap = _bootstrap_absolute_mae(predictions)
    metrics = metrics.merge(
        absolute_mae_bootstrap,
        on=["evaluation", "model"],
        how="left",
        validate="one_to_one",
    )
    source_metrics = _source_metrics(predictions)
    bootstrap_frames: list[pd.DataFrame] = []
    bootstrap_summaries: list[dict[str, Any]] = []
    for evaluation in (
        "old_internal_scaffold_cv",
        "old_to_fixed_angelo",
        "combined_scaffold_cv",
        "combined_leave_one_scaffold_out",
    ):
        hybrid = predictions[
            predictions["evaluation"].eq(evaluation) & predictions["model"].eq("hybrid_ecfp8")
        ]
        controls_frame = predictions[
            predictions["evaluation"].eq(evaluation) & predictions["model"].eq("compact_global_controls")
        ]
        replicates, summary = _bootstrap_model_difference(
            hybrid,
            controls_frame,
            evaluation=evaluation,
        )
        bootstrap_frames.append(replicates)
        bootstrap_summaries.append(summary)
    bootstrap = pd.concat(bootstrap_frames, ignore_index=True)
    bootstrap_summary = pd.DataFrame(bootstrap_summaries)

    source = load_ascentage_source(SOURCE, recovery_artifact=RECOVERY)
    novel = source[~source["structure_id"].isin(set(compounds["structure_id"]))]
    counts = {
        "old_workbook_structures": int(internal_audit["compound_id"].nunique()),
        "old_workbook_unique_structures": int(compounds["structure_id"].nunique()),
        "old_clean_structures": int(internal["compound_id"].nunique()),
        "old_quarantined_structures": int(len(ambiguous)),
        "angelo_total_structures": int(len(source)),
        "angelo_overlap_structures": int(len(overlap)),
        "angelo_novel_structures": int(len(novel)),
        "angelo_novel_measured_structures": int(len(extension)),
        "angelo_novel_missing_structures": int(novel["herg_ic50_censoring"].eq("missing").sum()),
        "combined_clean_structures": int(combined["compound_id"].nunique()),
        "combined_exact_structures": int(
            (
                np.isfinite(combined["pic50_lower"])
                & np.isclose(combined["pic50_lower"], combined["pic50_upper"])
            ).sum()
        ),
        "combined_censored_structures": int(
            (
                ~(
                    np.isfinite(combined["pic50_lower"])
                    & np.isclose(combined["pic50_lower"], combined["pic50_upper"])
                )
            ).sum()
        ),
    }
    overlap_audit = _overlap_audit(overlap, compounds, internal_audit)
    existing_augmented = json.loads((EXISTING / "augmented_grouped_metrics.json").read_text())
    comparison_context = {
        "existing_all_provenance_rows_sensitivity": existing_augmented,
        "primary_structure_level_counts": counts,
        "locked_feature_contract": {
            "compact_global_controls": controls,
            "hybrid_addition": "ECFP4 radius 2, 2048 bits compressed to 8 components inside each fold",
            "censoring": "interval-censored Gaussian ridge",
            "ridge_alpha": ALPHA,
            "component_count": COMPONENTS,
            "component_selection_boundary": "locked from old-internal grouped analysis",
        },
        "claim_boundary": (
            "same-series retrospective evidence; combined CV is not independent external validation"
        ),
    }
    validation = {
        "status": "pass",
        "checks": {
            "source_has_110_unique_old_workbook_structures": len(compounds) == 110,
            "angelo_has_76_unique_structures": len(source) == source["structure_id"].nunique() == 76,
            "angelo_overlap_count_is_22": len(overlap) == 22,
            "angelo_novel_measured_count_is_46": len(extension) == 46,
            "ambiguous_internal_labels_quarantined": len(ambiguous) == 5,
            "combined_rows_are_unique_structures": len(combined) == combined["compound_id"].nunique(),
            "combined_has_no_old_angelo_structure_overlap": not bool(
                set(internal["standardized_smiles"]) & set(extension["standardized_smiles"])
            ),
            "every_model_fit_converged": bool(metrics["all_folds_converged"].all()),
            "every_oof_row_scored_once": not predictions["predicted_pic50"].isna().any(),
            "bootstrap_replicates_complete": len(bootstrap) == 4 * BOOTSTRAPS,
        },
    }
    if not all(validation["checks"].values()):
        failed = [key for key, passed in validation["checks"].items() if not passed]
        raise RuntimeError(f"Combined internal/Angelo validation failed: {failed}")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(OUTPUT / "model_comparison.csv", metrics)
    atomic_write_json(OUTPUT / "model_comparison.json", metrics.to_dict(orient="records"))
    atomic_write_csv(OUTPUT / "mae_scaffold_bootstrap_summary.csv", absolute_mae_bootstrap)
    atomic_write_csv(OUTPUT / "source_stratified_metrics.csv", source_metrics)
    atomic_write_csv(OUTPUT / "internal_structure_label_audit.csv", internal_audit)
    atomic_write_csv(OUTPUT / "ambiguous_internal_measurements.csv", ambiguous)
    atomic_write_csv(OUTPUT / "angelo_overlap_duplicate_audit.csv", overlap_audit)
    atomic_write_parquet(OUTPUT / "predictions.parquet", predictions)
    atomic_write_csv(OUTPUT / "predictions.csv", predictions)
    atomic_write_parquet(OUTPUT / "scaffold_bootstrap_replicates.parquet", bootstrap)
    atomic_write_csv(OUTPUT / "scaffold_bootstrap_summary.csv", bootstrap_summary)
    atomic_write_json(OUTPUT / "combined_model_contract.json", comparison_context)
    atomic_write_json(OUTPUT / "validation_report.json", validation)
    atomic_write_text(
        OUTPUT / "combined_model_report.md",
        _report(metrics, source_metrics, bootstrap_summary, counts),
    )
    return {
        "status": "pass",
        "output": str(OUTPUT),
        "counts": counts,
        "metrics": metric_rows,
        "bootstrap": bootstrap_summaries,
    }


def main() -> None:
    print(json.dumps(run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
