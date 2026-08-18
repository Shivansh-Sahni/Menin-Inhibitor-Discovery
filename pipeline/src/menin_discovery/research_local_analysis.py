"""High-fidelity CPU analyses for the PK/hERG program.

This module deliberately analyzes existing canonical measurements and
group-held-out predictions.  It does not generate molecular structures,
simulate dynamics, or promote retrospective results to the decision track.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from menin_discovery.research_common import (
    atomic_write_csv,
    atomic_write_json,
    atomic_write_parquet,
    atomic_write_text,
    require_columns,
)
from menin_discovery.research_modeling import (
    grouped_regression_benchmark,
    merge_feature_layers,
)
from menin_discovery.research_workflows import (
    compound_model_frame,
    load_canonical_tables,
    prepare_pk_tasks,
)

PK_ENDPOINTS = (
    "iv_auc_dose_normalized",
    "po_auc_dose_normalized",
    "vdss",
    "po_cmax_dose_normalized",
    "po_tmax",
)
PROTOCOL_FIELDS = (
    "species",
    "strain",
    "matrix",
    "route",
    "cell_system",
    "method",
    "temperature_c",
    "p_h",
    "duration_value",
    "test_concentration_value",
)
PK_STUDY_FIELDS = (
    "species",
    "strain",
    "sex",
    "route",
    "dose_value",
    "dose_unit",
    "formulation",
    "vehicle",
    "matrix",
)


def _present(series: pd.Series) -> pd.Series:
    return series.notna() & series.astype(str).str.strip().ne("")


def _safe_spearman(x: pd.Series | np.ndarray, y: pd.Series | np.ndarray) -> float:
    x_array = np.asarray(x, dtype=float)
    y_array = np.asarray(y, dtype=float)
    finite = np.isfinite(x_array) & np.isfinite(y_array)
    if finite.sum() < 3 or np.unique(x_array[finite]).size < 2 or np.unique(y_array[finite]).size < 2:
        return float("nan")
    return float(spearmanr(x_array[finite], y_array[finite]).statistic)


def _r2(observed: np.ndarray, predicted: np.ndarray) -> float:
    denominator = float(np.sum((observed - observed.mean()) ** 2))
    if denominator <= 0:
        return float("nan")
    return 1.0 - float(np.sum((observed - predicted) ** 2)) / denominator


def _performance(frame: pd.DataFrame) -> dict[str, float]:
    observed = frame["observed"].to_numpy(dtype=float)
    predicted = frame["predicted"].to_numpy(dtype=float)
    residual = observed - predicted
    result = {
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "r2": _r2(observed, predicted),
        "spearman": _safe_spearman(observed, predicted),
    }
    if {"interval_lower", "interval_upper"}.issubset(frame.columns):
        result["interval_coverage"] = float(
            ((observed >= frame["interval_lower"]) & (observed <= frame["interval_upper"])).mean()
        )
    return result


def grouped_bootstrap_performance(
    frame: pd.DataFrame,
    *,
    group_column: str = "scaffold",
    bootstrap_replicates: int = 2000,
    random_state: int = 20260724,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Bootstrap fixed OOF performance by scaffold, without refitting models."""

    require_columns(
        frame,
        {"observed", "predicted", group_column},
        label="compound-level prediction frame",
    )
    work = frame.dropna(subset=["observed", "predicted", group_column]).copy()
    if work.empty:
        raise ValueError("No complete rows are available for grouped performance")
    observed = _performance(work)
    groups = work[group_column].astype(str).unique()
    group_frames = {group: work.loc[work[group_column].astype(str).eq(group)] for group in groups}
    rng = np.random.default_rng(random_state)
    rows: list[dict[str, float | int]] = []
    for replicate in range(bootstrap_replicates):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        sample = pd.concat([group_frames[group] for group in sampled], ignore_index=True)
        rows.append({"replicate": replicate, **_performance(sample)})
    bootstrap = pd.DataFrame(rows)
    for metric in ("mae", "rmse", "r2", "spearman", "interval_coverage"):
        if metric not in bootstrap:
            continue
        finite = pd.to_numeric(bootstrap[metric], errors="coerce").dropna()
        observed[f"{metric}_lower_95"] = float(finite.quantile(0.025)) if len(finite) else float("nan")
        observed[f"{metric}_upper_95"] = float(finite.quantile(0.975)) if len(finite) else float("nan")
    return observed, bootstrap


def grouped_bootstrap_spearman(
    frame: pd.DataFrame,
    *,
    x: str,
    y: str,
    group_column: str = "scaffold",
    bootstrap_replicates: int = 2000,
    random_state: int = 20260724,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Estimate scaffold robustness and a within-scaffold permutation test."""

    require_columns(frame, {x, y, group_column}, label="correlation frame")
    work = frame[[x, y, group_column]].dropna().copy()
    groups = work[group_column].astype(str).unique()
    result: dict[str, Any] = {
        "x": x,
        "y": y,
        "n": int(len(work)),
        "n_scaffolds": int(len(groups)),
        "spearman": _safe_spearman(work[x], work[y]),
    }
    group_size = work.groupby(group_column).size()
    informative_groups = group_size.loc[group_size.ge(2)].index
    within = work.loc[work[group_column].isin(informative_groups)].copy()
    if len(within):
        within["_x_centered"] = within[x] - within.groupby(group_column)[x].transform("mean")
        within["_y_centered"] = within[y] - within.groupby(group_column)[y].transform("mean")
    between = work.groupby(group_column, as_index=False)[[x, y]].mean()
    leave_one_out = [
        _safe_spearman(
            work.loc[work[group_column].ne(group), x],
            work.loc[work[group_column].ne(group), y],
        )
        for group in groups
    ]
    finite_leave_one_out = np.asarray(
        [value for value in leave_one_out if np.isfinite(value)],
        dtype=float,
    )
    result.update(
        {
            "within_scaffold_n": int(len(within)),
            "within_scaffold_n_groups": int(len(informative_groups)),
            "within_scaffold_spearman": (
                _safe_spearman(within["_x_centered"], within["_y_centered"]) if len(within) else float("nan")
            ),
            "between_scaffold_spearman": _safe_spearman(between[x], between[y]),
            "leave_one_scaffold_min": (
                float(finite_leave_one_out.min()) if len(finite_leave_one_out) else float("nan")
            ),
            "leave_one_scaffold_max": (
                float(finite_leave_one_out.max()) if len(finite_leave_one_out) else float("nan")
            ),
        }
    )
    within_observed = float(result["within_scaffold_spearman"])
    if len(within) >= 10 and len(informative_groups) >= 3 and np.isfinite(within_observed):
        permutation_rng = np.random.default_rng(random_state + 10_000)
        permutation_values = np.empty(bootstrap_replicates, dtype=float)
        centered_x = within["_x_centered"].to_numpy(dtype=float)
        centered_y = within["_y_centered"].to_numpy(dtype=float)
        index_lookup = {index: position for position, index in enumerate(within.index)}
        position_groups = [
            np.asarray([index_lookup[index] for index in group.index], dtype=int)
            for _, group in within.groupby(group_column, sort=False)
        ]
        for replicate in range(bootstrap_replicates):
            permuted_y = centered_y.copy()
            for positions in position_groups:
                permuted_y[positions] = permutation_rng.permutation(permuted_y[positions])
            permutation_values[replicate] = _safe_spearman(centered_x, permuted_y)
        finite_permutations = permutation_values[np.isfinite(permutation_values)]
        result["within_scaffold_permutation_replicates"] = int(len(finite_permutations))
        result["within_scaffold_permutation_p"] = float(
            (1 + np.sum(np.abs(finite_permutations) >= abs(within_observed))) / (len(finite_permutations) + 1)
        )
    else:
        result["within_scaffold_permutation_replicates"] = 0
        result["within_scaffold_permutation_p"] = float("nan")
    if len(work) < 10 or len(groups) < 3:
        result.update(
            {
                "spearman_lower_95": float("nan"),
                "spearman_upper_95": float("nan"),
                "interval_excludes_zero": False,
                "status": "insufficient_overlap",
            }
        )
        return result, pd.DataFrame(columns=["replicate", "spearman"])

    group_frames = {group: work.loc[work[group_column].astype(str).eq(group)] for group in groups}
    rng = np.random.default_rng(random_state)
    rows: list[dict[str, float | int]] = []
    for replicate in range(bootstrap_replicates):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        sample = pd.concat([group_frames[group] for group in sampled], ignore_index=True)
        rows.append({"replicate": replicate, "spearman": _safe_spearman(sample[x], sample[y])})
    bootstrap = pd.DataFrame(rows)
    finite = bootstrap["spearman"].dropna()
    lower = float(finite.quantile(0.025))
    upper = float(finite.quantile(0.975))
    result.update(
        {
            "spearman_lower_95": lower,
            "spearman_upper_95": upper,
            "interval_excludes_zero": bool(lower > 0 or upper < 0),
            "status": "descriptive_scaffold_bootstrap",
        }
    )
    return result, bootstrap


def _benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    """Return monotone Benjamini-Hochberg q-values while preserving missingness."""

    numeric = pd.to_numeric(p_values, errors="coerce")
    finite = numeric.dropna().clip(lower=0.0, upper=1.0)
    result = pd.Series(np.nan, index=p_values.index, dtype=float)
    if finite.empty:
        return result
    ordered = finite.sort_values()
    ranks = np.arange(1, len(ordered) + 1, dtype=float)
    adjusted = ordered.to_numpy(dtype=float) * len(ordered) / ranks
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result.loc[ordered.index] = np.minimum(adjusted, 1.0)
    return result


def compound_level_predictions(
    predictions: pd.DataFrame,
    *,
    model: str | None,
    observed_column: str,
    predicted_column: str,
    interval_lower_column: str | None = None,
    interval_upper_column: str | None = None,
) -> pd.DataFrame:
    """Balance repeated evidence rows so every compound contributes once."""

    work = predictions.copy()
    if model is not None:
        require_columns(work, {"model"}, label="prediction frame")
        work = work.loc[work["model"].astype(str).eq(model)].copy()
    required = {"compound_id", "group", observed_column, predicted_column}
    require_columns(work, required, label="prediction frame")
    aggregations: dict[str, tuple[str, str]] = {
        "observed": (observed_column, "mean"),
        "predicted": (predicted_column, "mean"),
        "scaffold": ("group", "first"),
    }
    if interval_lower_column and interval_lower_column in work:
        aggregations["interval_lower"] = (interval_lower_column, "mean")
    if interval_upper_column and interval_upper_column in work:
        aggregations["interval_upper"] = (interval_upper_column, "mean")
    if "inside_applicability_domain" in work:
        aggregations["inside_applicability_domain"] = ("inside_applicability_domain", "first")
    balanced = work.groupby("compound_id", as_index=False).agg(**aggregations)
    balanced["residual"] = balanced["observed"] - balanced["predicted"]
    return balanced


def _scaffold_and_domain_diagnostics(
    endpoint_frames: list[tuple[str, pd.DataFrame]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Quantify independent-series support, scaffold influence, and AD behavior."""

    support_rows: list[dict[str, Any]] = []
    scaffold_rows: list[dict[str, Any]] = []
    domain_rows: list[dict[str, Any]] = []
    for endpoint, frame in endpoint_frames:
        counts = frame.groupby("scaffold").size().sort_values(ascending=False)
        fractions = counts / counts.sum()
        support_rows.append(
            {
                "endpoint": endpoint,
                "n_compounds": int(len(frame)),
                "n_scaffolds": int(len(counts)),
                "largest_scaffold_n": int(counts.iloc[0]),
                "largest_scaffold_fraction": float(fractions.iloc[0]),
                "singleton_scaffold_count": int(counts.eq(1).sum()),
                "effective_scaffold_count_simpson": float(1.0 / np.sum(fractions**2)),
                "interpretation": (
                    "Effective count is 1/sum(p_scaffold^2); it exposes support concentrated in a few series."
                ),
            }
        )
        for scaffold, group in frame.groupby("scaffold", sort=False):
            scaffold_rows.append(
                {
                    "endpoint": endpoint,
                    "scaffold": scaffold,
                    "n_compounds": int(len(group)),
                    "mean_residual_bias": float(group["residual"].mean()),
                    "inside_domain_fraction": (
                        float(group["inside_applicability_domain"].mean())
                        if "inside_applicability_domain" in group
                        else float("nan")
                    ),
                    **_performance(group),
                    "rank_metric_status": (
                        "estimable" if len(group) >= 3 else "insufficient_within_scaffold_n"
                    ),
                }
            )
        if "inside_applicability_domain" in frame:
            for inside, group in frame.groupby("inside_applicability_domain", dropna=False):
                domain_rows.append(
                    {
                        "endpoint": endpoint,
                        "domain_stratum": "inside" if bool(inside) else "outside",
                        "n_compounds": int(len(group)),
                        "n_scaffolds": int(group["scaffold"].nunique()),
                        **_performance(group),
                        "interpretation": (
                            "Retrospective OOF diagnostic; the AD threshold was not prospectively calibrated."
                        ),
                    }
                )
    return (
        pd.DataFrame(support_rows),
        pd.DataFrame(scaffold_rows),
        pd.DataFrame(domain_rows),
    )


def _best_conventional_pk_model(project_root: Path, endpoint: str) -> str:
    path = (
        project_root
        / "research/models/pk_herg/pk"
        / endpoint
        / "hierarchical_vs_conventional_compound_balanced.csv"
    )
    metrics = pd.read_csv(path)
    candidates = metrics.loc[metrics["comparison_role"].eq("retained_conventional_candidate")].copy()
    if candidates.empty:
        raise ValueError(f"No retained conventional candidate exists for {endpoint}")
    return str(candidates.sort_values(["log_mae", "model"]).iloc[0]["model"])


def _load_balanced_predictions(project_root: Path) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    endpoint_frames: dict[str, pd.DataFrame] = {}
    for endpoint in PK_ENDPOINTS:
        model = _best_conventional_pk_model(project_root, endpoint)
        predictions = pd.read_parquet(
            project_root / "research/models/pk_herg/pk" / endpoint / "structure_2d_predictions.parquet"
        )
        balanced = compound_level_predictions(
            predictions,
            model=model,
            observed_column="observed_log10",
            predicted_column="predicted_log10",
            interval_lower_column="interval_lower_log10",
            interval_upper_column="interval_upper_log10",
        )
        balanced["endpoint"] = endpoint
        balanced["model"] = model
        endpoint_frames[endpoint] = balanced

    herg_metrics = pd.read_csv(
        project_root / "research/models/pk_herg/herg/conventional_exact_pic50_metrics.csv"
    )
    herg_model = str(herg_metrics.sort_values(["pic50_mae", "model"]).iloc[0]["model"])
    herg_predictions = pd.read_parquet(
        project_root / "research/models/pk_herg/herg/conventional_exact_pic50_predictions.parquet"
    )
    herg = compound_level_predictions(
        herg_predictions,
        model=herg_model,
        observed_column="observed_pic50",
        predicted_column="predicted_pic50",
        interval_lower_column="interval_lower_pic50",
        interval_upper_column="interval_upper_pic50",
    )
    herg["endpoint"] = "herg_pic50"
    herg["model"] = herg_model
    return endpoint_frames, herg


def _compound_balanced_training_sensitivity(
    project_root: Path,
    *,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Retrain fixed core models on one equally weighted row per compound."""

    canonical_root = project_root / "research/data/pk_herg/canonical"
    tables = load_canonical_tables(canonical_root)
    compounds = compound_model_frame(tables["compounds"], tables.get("compound_aliases"))
    features, layers = merge_feature_layers(compounds, None)
    tasks = prepare_pk_tasks(
        compounds,
        tables["measurements"],
        tables["pk_studies"],
        features,
    )
    model_names = ["ridge", "random_forest", "extra_trees", "svr"]
    metric_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    comparison_frames: list[pd.DataFrame] = []
    for endpoint, task in tasks.items():
        usable = [
            column for column in layers["structure_2d"] if column in task and task[column].notna().any()
        ]
        first_columns = [
            "standardized_smiles",
            "scaffold",
            *usable,
        ]
        aggregations: dict[str, tuple[str, str]] = {
            "target_value": ("target_value", "mean"),
            **{column: (column, "first") for column in first_columns},
        }
        balanced = task.groupby("compound_id", as_index=False).agg(**aggregations)
        metrics, predictions = grouped_regression_benchmark(
            balanced,
            feature_columns=usable,
            target_column="target_value",
            folds=5,
            random_state=random_state,
            interval_level=0.90,
            model_names=model_names,
        )
        metrics.insert(0, "endpoint", endpoint)
        metrics["training_unit"] = "one_equally_weighted_mean_per_compound"
        predictions.insert(0, "endpoint", endpoint)
        predictions["training_unit"] = "one_equally_weighted_mean_per_compound"
        metric_frames.append(metrics)
        prediction_frames.append(predictions)

        existing = pd.read_csv(
            project_root
            / "research/models/pk_herg/pk"
            / endpoint
            / "hierarchical_vs_conventional_compound_balanced.csv"
        )
        existing = existing.loc[existing["model"].isin(model_names)].copy()
        existing = existing[
            ["model", "log_mae", "log_rmse", "spearman", "prediction_interval_coverage"]
        ].rename(
            columns={
                "log_mae": "evidence_row_trained_log_mae",
                "log_rmse": "evidence_row_trained_log_rmse",
                "spearman": "evidence_row_trained_spearman",
                "prediction_interval_coverage": "evidence_row_trained_interval_coverage",
            }
        )
        comparison = metrics.merge(existing, on="model", how="inner", validate="one_to_one")
        comparison["delta_log_mae_compound_minus_evidence_training"] = (
            comparison["log_mae"] - comparison["evidence_row_trained_log_mae"]
        )
        comparison["delta_spearman_compound_minus_evidence_training"] = (
            comparison["spearman"] - comparison["evidence_row_trained_spearman"]
        )
        comparison["interpretation"] = (
            "Training-unit sensitivity only; neither fit is prospectively calibrated."
        )
        comparison_frames.append(comparison)
    return (
        pd.concat(metric_frames, ignore_index=True),
        pd.concat(prediction_frames, ignore_index=True),
        pd.concat(comparison_frames, ignore_index=True),
    )


def _residual_model_family_sensitivity(
    project_root: Path,
    *,
    herg: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Test whether residual associations persist across fixed model families."""

    model_names = ("ridge", "random_forest", "extra_trees", "svr", "xgboost", "lightgbm")
    frames: dict[tuple[str, str], pd.DataFrame] = {}
    for endpoint in PK_ENDPOINTS:
        predictions = pd.read_parquet(
            project_root / "research/models/pk_herg/pk" / endpoint / "structure_2d_predictions.parquet"
        )
        for model in model_names:
            if model not in set(predictions["model"]):
                continue
            frames[(endpoint, model)] = compound_level_predictions(
                predictions,
                model=model,
                observed_column="observed_log10",
                predicted_column="predicted_log10",
            )
    if herg:
        predictions = pd.read_parquet(
            project_root / "research/models/pk_herg/herg/conventional_exact_pic50_predictions.parquet"
        )
        for model in model_names:
            if model not in set(predictions["model"]):
                continue
            frames[("herg_pic50", model)] = compound_level_predictions(
                predictions,
                model=model,
                observed_column="observed_pic50",
                predicted_column="predicted_pic50",
            )

    endpoints = (*PK_ENDPOINTS, "herg_pic50") if herg else PK_ENDPOINTS
    rows: list[dict[str, Any]] = []
    for left_index, left in enumerate(endpoints):
        for right in endpoints[left_index + 1 :]:
            for model in model_names:
                left_frame = frames.get((left, model))
                right_frame = frames.get((right, model))
                if left_frame is None or right_frame is None:
                    continue
                merged = left_frame[["compound_id", "residual"]].merge(
                    right_frame[["compound_id", "residual"]],
                    on="compound_id",
                    suffixes=("_left", "_right"),
                    validate="one_to_one",
                )
                rows.append(
                    {
                        "left_endpoint": left,
                        "right_endpoint": right,
                        "model": model,
                        "n": int(len(merged)),
                        "spearman": _safe_spearman(
                            merged["residual_left"],
                            merged["residual_right"],
                        ),
                    }
                )
    detail = pd.DataFrame(rows)
    summaries: list[dict[str, Any]] = []
    for (left, right), group in detail.groupby(["left_endpoint", "right_endpoint"]):
        values = group["spearman"].dropna().to_numpy(dtype=float)
        observed_sign = np.sign(np.median(values))
        summaries.append(
            {
                "left_endpoint": left,
                "right_endpoint": right,
                "n_models": int(len(values)),
                "median_spearman": float(np.median(values)),
                "minimum_spearman": float(np.min(values)),
                "maximum_spearman": float(np.max(values)),
                "sign_consistency_fraction": float(np.mean(np.sign(values) == observed_sign)),
                "all_model_families_same_sign": bool(np.all(np.sign(values) == observed_sign)),
                "interpretation": (
                    "Residual-pattern sensitivity across fixed model families; "
                    "not an independent biological replication."
                ),
            }
        )
    summary = pd.DataFrame(summaries).sort_values(
        ["all_model_families_same_sign", "sign_consistency_fraction", "median_spearman"],
        ascending=[False, False, False],
    )
    return detail, summary


def _coverage_table(measurements: pd.DataFrame, pk_samples: pd.DataFrame) -> pd.DataFrame:
    def compounds_for(endpoints: tuple[str, ...]) -> int:
        return int(measurements.loc[measurements["endpoint"].isin(endpoints), "compound_id"].nunique())

    rows = [
        {
            "process": "chemical_state_speciation",
            "support_level": "approximate_proxy",
            "internal_compounds": compounds_for(("acidic_pka_below_13", "basic_pka")),
            "available_evidence": "reported/basic pKa fields",
            "critical_gap": "microscopic pKas, tautomer free energies, exchange rates",
        },
        {
            "process": "dissolution_and_free_monomer",
            "support_level": "none",
            "internal_compounds": compounds_for(("thermodynamic_solubility", "kinetic_solubility")),
            "available_evidence": "none",
            "critical_gap": "pH solubility, dissolution, precipitation, aggregation",
        },
        {
            "process": "environment_conditioned_conformation",
            "support_level": "none",
            "internal_compounds": 0,
            "available_evidence": "structure hypotheses only",
            "critical_gap": "converged solvent-conditioned populations and transition rates",
        },
        {
            "process": "membrane_permeation",
            "support_level": "none",
            "internal_compounds": compounds_for(("pampa_permeability", "caco2_permeability")),
            "available_evidence": "none",
            "critical_gap": "passive flux, recovery, pH and membrane-condition dependence",
        },
        {
            "process": "enterocyte_fate",
            "support_level": "none",
            "internal_compounds": compounds_for(("efflux_ratio",)),
            "available_evidence": "none",
            "critical_gap": "AB/BA flux, recovery, inhibitors, concentration series, gut stability",
        },
        {
            "process": "hepatic_extraction_and_metabolism",
            "support_level": "sparse_partial",
            "internal_compounds": compounds_for(
                ("hepatic_extraction_ratio", "microsomal_stability_half_life")
            ),
            "available_evidence": "hepatic extraction and microsomal stability",
            "critical_gap": "hepatocyte CLint, uptake, fu-inc, biliary transport",
        },
        {
            "process": "systemic_distribution_and_binding",
            "support_level": "sparse_partial",
            "internal_compounds": compounds_for(
                ("plasma_protein_bound_percent", "plasma_protein_unbound_percent")
            ),
            "available_evidence": "plasma protein binding",
            "critical_gap": "blood:plasma ratio, tissue exchange, subcellular sequestration",
        },
        {
            "process": "renal_and_other_elimination",
            "support_level": "none",
            "internal_compounds": compounds_for(("renal_clearance",)),
            "available_evidence": "none",
            "critical_gap": "filtration, secretion, reabsorption and urinary recovery",
        },
        {
            "process": "raw_pk_time_course",
            "support_level": "none",
            "internal_compounds": int(pk_samples["compound_id"].nunique()) if len(pk_samples) else 0,
            "available_evidence": f"{len(pk_samples)} concentration-time rows",
            "critical_gap": "per-animal IV/PO profiles, LLOQ, formulation and sampling metadata",
        },
        {
            "process": "static_herg_potency",
            "support_level": "moderate_endpoint_evidence",
            "internal_compounds": compounds_for(("herg_ic50", "herg_percent_inhibition")),
            "available_evidence": "IC50 limits/exacts and concentration-specific inhibition",
            "critical_gap": "free concentration and harmonized protocol metadata",
        },
        {
            "process": "dynamic_herg_binding_and_trapping",
            "support_level": "none",
            "internal_compounds": compounds_for(("herg_onset", "herg_recovery", "herg_trapping")),
            "available_evidence": "none",
            "critical_gap": "onset, washout, recovery and trapping under declared voltage protocols",
        },
    ]
    return pd.DataFrame(rows)


def _protocol_completeness(
    protocols: pd.DataFrame,
    pk_studies: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for family, group in protocols.groupby("assay_family", dropna=False):
        for field in PROTOCOL_FIELDS:
            rows.append(
                {
                    "record_type": "assay_protocol",
                    "stratum": str(family),
                    "field": field,
                    "n_records": int(len(group)),
                    "n_present": int(_present(group[field]).sum()),
                    "fraction_present": float(_present(group[field]).mean()),
                }
            )
    for field in PK_STUDY_FIELDS:
        rows.append(
            {
                "record_type": "pk_study",
                "stratum": "all_internal_pk_studies",
                "field": field,
                "n_records": int(len(pk_studies)),
                "n_present": int(_present(pk_studies[field]).sum()),
                "fraction_present": float(_present(pk_studies[field]).mean()),
            }
        )
    return pd.DataFrame(rows)


def _source_echo_audit(
    measurements: pd.DataFrame,
    pk_studies: pd.DataFrame,
) -> pd.DataFrame:
    study_context = pk_studies[
        ["pk_study_id", "dose_value", "dose_unit", "formulation", "vehicle"]
    ].drop_duplicates("pk_study_id")
    work = measurements.merge(study_context, on="pk_study_id", how="left")
    signature_columns = [
        "compound_id",
        "endpoint",
        "submitted_value",
        "value",
        "unit",
        "relation",
        "species",
        "route",
        "test_concentration_value",
        "test_concentration_unit",
        "dose_value",
        "dose_unit",
    ]
    signature = work[signature_columns].copy()
    for column in signature:
        signature[column] = signature[column].astype(str).fillna("<missing>")
    work["_signature"] = pd.util.hash_pandas_object(signature, index=False).astype(str)
    rows: list[dict[str, Any]] = []
    for endpoint, group in work.groupby("endpoint"):
        multiplicity = group.groupby("_signature").size()
        value_counts = (
            group.assign(_numeric=pd.to_numeric(group["value"], errors="coerce"))
            .groupby("compound_id")["_numeric"]
            .nunique(dropna=True)
        )
        rows.append(
            {
                "endpoint": endpoint,
                "record_rows": int(len(group)),
                "unique_compounds": int(group["compound_id"].nunique()),
                "unique_measurement_signatures": int(len(multiplicity)),
                "rows_in_repeated_signatures": int(multiplicity.loc[multiplicity.gt(1)].sum()),
                "repeated_signature_fraction": float(multiplicity.loc[multiplicity.gt(1)].sum() / len(group)),
                "maximum_signature_multiplicity": int(multiplicity.max()),
                "compounds_with_multiple_distinct_numeric_values": int(value_counts.gt(1).sum()),
                "interpretation": (
                    "Repeated signatures are potential source echoes, not biological replicates; "
                    "replicate identity is unavailable."
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["repeated_signature_fraction", "endpoint"],
        ascending=[False, True],
    )


def _markdown_table(frame: pd.DataFrame, columns: list[str], *, maximum_rows: int = 20) -> str:
    display = frame.loc[:, columns].head(maximum_rows).copy()
    for column in display:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(lambda value: "" if pd.isna(value) else f"{value:.3f}")
    header = "| " + " | ".join(columns) + " |"
    separator = "|" + "|".join("---" for _ in columns) + "|"
    rows = [
        "| " + " | ".join(str(value).replace("|", "/") for value in row) + " |"
        for row in display.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def _write_publication_figures(
    output: Path,
    endpoint_frames: list[tuple[str, pd.DataFrame]],
    correlations: pd.DataFrame,
    associations: pd.DataFrame,
) -> None:
    """Write compact figures from the same canonical result tables."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    endpoint_labels = {
        "iv_auc_dose_normalized": "IV dose-normalized AUC",
        "po_auc_dose_normalized": "PO dose-normalized AUC",
        "vdss": "Vdss",
        "po_cmax_dose_normalized": "PO dose-normalized Cmax",
        "po_tmax": "PO Tmax",
        "herg_pic50": "hERG pIC50",
    }

    figure, axes = plt.subplots(2, 3, figsize=(12, 7.6), constrained_layout=True)
    for axis, (endpoint, frame) in zip(axes.flat, endpoint_frames, strict=True):
        inside = frame.get(
            "inside_applicability_domain",
            pd.Series(False, index=frame.index),
        ).fillna(False)
        axis.scatter(
            frame.loc[~inside, "observed"],
            frame.loc[~inside, "predicted"],
            s=28,
            alpha=0.70,
            color="#9aa0a6",
            edgecolor="none",
            label="outside AD",
        )
        axis.scatter(
            frame.loc[inside, "observed"],
            frame.loc[inside, "predicted"],
            s=32,
            alpha=0.85,
            color="#1f77b4",
            edgecolor="white",
            linewidth=0.35,
            label="inside AD",
        )
        bounds = np.asarray(
            [
                frame["observed"].min(),
                frame["observed"].max(),
                frame["predicted"].min(),
                frame["predicted"].max(),
            ],
            dtype=float,
        )
        padding = max(0.05, float(np.ptp(bounds)) * 0.06)
        lower, upper = float(bounds.min() - padding), float(bounds.max() + padding)
        axis.plot([lower, upper], [lower, upper], color="#202124", linewidth=0.8)
        axis.set_xlim(lower, upper)
        axis.set_ylim(lower, upper)
        metrics = _performance(frame)
        axis.set_title(
            f"{endpoint_labels[endpoint]}\n"
            f"n={len(frame)}; MAE={metrics['mae']:.3f}; ρ={metrics['spearman']:.2f}",
            fontsize=10,
        )
        axis.set_xlabel("Observed")
        axis.set_ylabel("OOF predicted")
        axis.grid(alpha=0.18, linewidth=0.5)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=2, frameon=False)
    for suffix in ("png", "pdf"):
        path = figures / f"oof_endpoint_performance.{suffix}"
        figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)

    residual_plot = correlations.loc[correlations["analysis"].eq("oof_residual_covariance")].copy()
    residual_plot["pair"] = residual_plot.apply(
        lambda row: (
            endpoint_labels.get(
                str(row["x"]).replace("__residual", ""),
                str(row["x"]).replace("__residual", ""),
            )
            + " / "
            + endpoint_labels.get(
                str(row["y"]).replace("__residual", ""),
                str(row["y"]).replace("__residual", ""),
            )
        ),
        axis=1,
    )
    residual_plot = residual_plot.sort_values(
        "spearman",
        key=lambda series: series.abs(),
        ascending=False,
    ).head(10)
    joint_plot = associations.copy()
    endpoint_labels["oral_minus_iv"] = "PO minus IV AUC"
    joint_plot["pair"] = joint_plot["x"].map(
        lambda value: (
            endpoint_labels.get(
                str(value).replace("__observed", ""),
                str(value).replace("__observed", ""),
            )
            + " / hERG"
        )
    )
    figure, axes = plt.subplots(1, 2, figsize=(13, 6.8), constrained_layout=True)
    for axis, frame, title in (
        (axes[0], residual_plot, "OOF residual associations"),
        (axes[1], joint_plot, "Observed PK–hERG associations"),
    ):
        positions = np.arange(len(frame))
        axis.axvline(0.0, color="#202124", linewidth=0.8)
        for column, offset, color, label in (
            ("spearman", -0.16, "#6a3d9a", "overall"),
            ("within_scaffold_spearman", 0.0, "#1f78b4", "within scaffold"),
            ("between_scaffold_spearman", 0.16, "#e31a1c", "between scaffold"),
        ):
            axis.scatter(
                frame[column],
                positions + offset,
                s=38,
                color=color,
                label=label,
                zorder=3,
            )
        significant = frame["within_scaffold_fdr_005"].fillna(False).to_numpy()
        for position, passed in zip(positions, significant, strict=True):
            if passed:
                axis.text(1.01, position, "FDR", va="center", fontsize=8)
        axis.set_yticks(positions, frame["pair"])
        axis.invert_yaxis()
        axis.set_xlim(-1.05, 1.13)
        axis.set_xlabel("Spearman ρ")
        axis.set_title(title)
        axis.grid(axis="x", alpha=0.20, linewidth=0.5)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    for suffix in ("png", "pdf"):
        path = figures / f"association_scaffold_decomposition.{suffix}"
        figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def run_local_analysis(
    project_root: Path,
    *,
    bootstrap_replicates: int = 2000,
    random_state: int = 20260724,
) -> dict[str, Any]:
    """Run deterministic, non-HPC analyses and write validated canonical outputs."""

    canonical = project_root / "research/data/pk_herg/canonical/internal"
    output = project_root / "research/reports/pk_herg/local_m3"
    measurements = pd.read_parquet(canonical / "measurements.parquet")
    protocols = pd.read_parquet(canonical / "assay_protocols.parquet")
    pk_studies = pd.read_parquet(canonical / "pk_studies.parquet")
    pk_samples = pd.read_parquet(canonical / "pk_samples.parquet")
    compounds = pd.read_parquet(canonical / "compounds.parquet")

    coverage = _coverage_table(measurements, pk_samples)
    protocol = _protocol_completeness(protocols, pk_studies)
    echoes = _source_echo_audit(measurements, pk_studies)
    atomic_write_csv(output / "process_coverage.csv", coverage)
    atomic_write_csv(output / "protocol_completeness.csv", protocol)
    atomic_write_csv(output / "source_echo_audit.csv", echoes)

    balanced_training_metrics, balanced_training_predictions, training_unit_comparison = (
        _compound_balanced_training_sensitivity(
            project_root,
            random_state=random_state,
        )
    )
    atomic_write_csv(
        output / "compound_balanced_training_metrics.csv",
        balanced_training_metrics,
    )
    atomic_write_parquet(
        output / "compound_balanced_training_predictions.parquet",
        balanced_training_predictions,
    )
    atomic_write_csv(
        output / "training_unit_sensitivity.csv",
        training_unit_comparison,
    )

    pk_frames, herg = _load_balanced_predictions(project_root)
    endpoint_frame_list = [*pk_frames.items(), ("herg_pic50", herg)]
    scaffold_support, scaffold_performance, domain_performance = _scaffold_and_domain_diagnostics(
        endpoint_frame_list
    )
    atomic_write_csv(output / "scaffold_support.csv", scaffold_support)
    atomic_write_csv(output / "scaffold_performance.csv", scaffold_performance)
    atomic_write_csv(output / "applicability_domain_performance.csv", domain_performance)
    model_sensitivity_detail, model_sensitivity_summary = _residual_model_family_sensitivity(project_root)
    atomic_write_csv(
        output / "residual_model_family_sensitivity_detail.csv",
        model_sensitivity_detail,
    )
    atomic_write_csv(
        output / "residual_model_family_sensitivity_summary.csv",
        model_sensitivity_summary,
    )
    performance_rows: list[dict[str, Any]] = []
    performance_bootstrap: list[pd.DataFrame] = []
    for index, (endpoint, frame) in enumerate(endpoint_frame_list):
        summary, bootstrap = grouped_bootstrap_performance(
            frame,
            bootstrap_replicates=bootstrap_replicates,
            random_state=random_state + index,
        )
        inside = frame.loc[frame.get("inside_applicability_domain", False).eq(True)]
        outside = frame.loc[frame.get("inside_applicability_domain", False).eq(False)]
        performance_rows.append(
            {
                "endpoint": endpoint,
                "model": str(frame["model"].iloc[0]),
                "n_compounds": int(len(frame)),
                "n_scaffolds": int(frame["scaffold"].nunique()),
                "inside_domain_n": int(len(inside)),
                "outside_domain_n": int(len(outside)),
                "inside_domain_mae": _performance(inside)["mae"] if len(inside) else float("nan"),
                "outside_domain_mae": _performance(outside)["mae"] if len(outside) else float("nan"),
                **summary,
                "uncertainty_scope": (
                    "scaffold bootstrap of fixed group-held-out predictions; models were not refit"
                ),
                "promotion": "retrospective_discovery_evidence_only",
            }
        )
        bootstrap.insert(0, "endpoint", endpoint)
        performance_bootstrap.append(bootstrap)
    performance = pd.DataFrame(performance_rows)
    atomic_write_csv(output / "retrospective_oof_robustness.csv", performance)
    atomic_write_parquet(
        output / "retrospective_oof_scaffold_bootstrap.parquet",
        pd.concat(performance_bootstrap, ignore_index=True),
    )

    wide = compounds[["compound_id", "scaffold"]].copy()
    for endpoint, frame in [*pk_frames.items(), ("herg_pic50", herg)]:
        values = frame[["compound_id", "observed", "predicted", "residual"]].rename(
            columns={
                "observed": f"{endpoint}__observed",
                "predicted": f"{endpoint}__predicted",
                "residual": f"{endpoint}__residual",
            }
        )
        wide = wide.merge(values, on="compound_id", how="left")
    wide["oral_specific__residual"] = (
        wide["po_auc_dose_normalized__residual"] - wide["iv_auc_dose_normalized__residual"]
    )
    wide["cmax_shape_specific__residual"] = (
        wide["po_cmax_dose_normalized__residual"] - wide["po_auc_dose_normalized__residual"]
    )
    wide["oral_minus_iv__observed"] = (
        wide["po_auc_dose_normalized__observed"] - wide["iv_auc_dose_normalized__observed"]
    )
    atomic_write_parquet(output / "compound_level_oof_residuals.parquet", wide)

    base_residuals = [f"{endpoint}__residual" for endpoint in (*PK_ENDPOINTS, "herg_pic50")]
    correlation_rows: list[dict[str, Any]] = []
    correlation_bootstrap: list[pd.DataFrame] = []
    pair_index = 0
    for left_index, left in enumerate(base_residuals):
        for right in base_residuals[left_index + 1 :]:
            result, bootstrap = grouped_bootstrap_spearman(
                wide,
                x=left,
                y=right,
                bootstrap_replicates=bootstrap_replicates,
                random_state=random_state + 100 + pair_index,
            )
            result["analysis"] = "oof_residual_covariance"
            result["causal_status"] = "hypothesis_localization_not_causal_identification"
            correlation_rows.append(result)
            if len(bootstrap):
                bootstrap.insert(0, "x", left)
                bootstrap.insert(1, "y", right)
                correlation_bootstrap.append(bootstrap)
            pair_index += 1

    derived_pairs = (
        ("oral_specific__residual", "po_tmax__residual"),
        ("oral_specific__residual", "herg_pic50__residual"),
        ("cmax_shape_specific__residual", "po_tmax__residual"),
    )
    for left, right in derived_pairs:
        result, bootstrap = grouped_bootstrap_spearman(
            wide,
            x=left,
            y=right,
            bootstrap_replicates=bootstrap_replicates,
            random_state=random_state + 200 + pair_index,
        )
        result["analysis"] = "derived_process_residual"
        result["causal_status"] = "algebraically_derived_residual_hypothesis_only"
        correlation_rows.append(result)
        if len(bootstrap):
            bootstrap.insert(0, "x", left)
            bootstrap.insert(1, "y", right)
            correlation_bootstrap.append(bootstrap)
        pair_index += 1

    correlations = pd.DataFrame(correlation_rows)
    correlations["within_scaffold_fdr_q"] = correlations.groupby("analysis", group_keys=False)[
        "within_scaffold_permutation_p"
    ].transform(_benjamini_hochberg)
    correlations["within_scaffold_fdr_005"] = correlations["within_scaffold_fdr_q"].le(0.05).fillna(False)
    correlations = correlations.sort_values(
        ["within_scaffold_fdr_005", "interval_excludes_zero", "n", "spearman"],
        ascending=[False, False, False, False],
    )
    atomic_write_csv(output / "oof_residual_correlations.csv", correlations)
    atomic_write_parquet(
        output / "oof_residual_correlation_bootstrap.parquet",
        pd.concat(correlation_bootstrap, ignore_index=True),
    )

    association_rows: list[dict[str, Any]] = []
    association_bootstrap: list[pd.DataFrame] = []
    for index, endpoint in enumerate((*PK_ENDPOINTS, "oral_minus_iv")):
        x = "oral_minus_iv__observed" if endpoint == "oral_minus_iv" else f"{endpoint}__observed"
        result, bootstrap = grouped_bootstrap_spearman(
            wide,
            x=x,
            y="herg_pic50__observed",
            bootstrap_replicates=bootstrap_replicates,
            random_state=random_state + 300 + index,
        )
        result["analysis"] = "joint_observed_pk_herg_association"
        result["causal_status"] = "descriptive_only_series_and_protocol_confounded"
        association_rows.append(result)
        if len(bootstrap):
            bootstrap.insert(0, "x", x)
            bootstrap.insert(1, "y", "herg_pic50__observed")
            association_bootstrap.append(bootstrap)
    associations = pd.DataFrame(association_rows)
    associations["within_scaffold_fdr_q"] = _benjamini_hochberg(associations["within_scaffold_permutation_p"])
    associations["within_scaffold_fdr_005"] = associations["within_scaffold_fdr_q"].le(0.05).fillna(False)
    associations = associations.sort_values(
        ["within_scaffold_fdr_005", "spearman"],
        ascending=[False, False],
    )
    atomic_write_csv(output / "joint_pk_herg_associations.csv", associations)
    atomic_write_parquet(
        output / "joint_pk_herg_association_bootstrap.parquet",
        pd.concat(association_bootstrap, ignore_index=True),
    )
    _write_publication_figures(output, endpoint_frame_list, correlations, associations)

    robust_residual = correlations.loc[correlations["within_scaffold_fdr_005"]].copy()
    robust_joint = associations.loc[associations["within_scaffold_fdr_005"]].copy()
    zero_support = coverage.loc[coverage["support_level"].eq("none"), "process"].tolist()
    hERG_protocols = protocol.loc[protocol["stratum"].eq("herg_inhibition")].set_index("field")
    missing_herg_fields = [
        field
        for field in ("cell_system", "method", "temperature_c", "p_h", "duration_value")
        if field in hERG_protocols.index and float(hERG_protocols.loc[field, "fraction_present"]) == 0.0
    ]

    run_summary: dict[str, Any] = {
        "analysis_scope": "high-fidelity CPU-only retrospective analyses",
        "bootstrap_replicates": bootstrap_replicates,
        "random_state": random_state,
        "n_internal_compounds": int(len(compounds)),
        "n_pk_compounds_in_oof_analysis": int(wide["iv_auc_dose_normalized__observed"].notna().sum()),
        "n_herg_compounds_in_oof_analysis": int(wide["herg_pic50__observed"].notna().sum()),
        "n_joint_pk_herg_compounds": int(
            wide[["iv_auc_dose_normalized__observed", "herg_pic50__observed"]].dropna().shape[0]
        ),
        "processes_with_no_direct_internal_support": zero_support,
        "missing_herg_protocol_fields": missing_herg_fields,
        "within_scaffold_fdr_residual_association_count": int(len(robust_residual)),
        "within_scaffold_fdr_joint_association_count": int(len(robust_joint)),
        "compound_balanced_training_endpoint_count": int(balanced_training_metrics["endpoint"].nunique()),
        "residual_pairs_stable_in_all_model_families": int(
            model_sensitivity_summary["all_model_families_same_sign"].sum()
        ),
        "physics_executed": False,
        "decision_track_promotion": False,
        "truth_boundary": (
            "Results quantify existing evidence and fixed OOF behavior. They do not identify "
            "latent physical rates, validate causal mechanisms, or replace HPC/experimental gates."
        ),
    }
    atomic_write_json(output / "local_analysis_summary.json", run_summary)

    performance_display = performance[
        [
            "endpoint",
            "model",
            "n_compounds",
            "mae",
            "mae_lower_95",
            "mae_upper_95",
            "spearman",
            "spearman_lower_95",
            "spearman_upper_95",
            "inside_domain_n",
        ]
    ]
    residual_display = correlations[
        [
            "x",
            "y",
            "n",
            "n_scaffolds",
            "spearman",
            "spearman_lower_95",
            "spearman_upper_95",
            "within_scaffold_spearman",
            "between_scaffold_spearman",
            "leave_one_scaffold_min",
            "leave_one_scaffold_max",
            "within_scaffold_permutation_p",
            "within_scaffold_fdr_q",
            "within_scaffold_fdr_005",
            "interval_excludes_zero",
        ]
    ]
    joint_display = associations[
        [
            "x",
            "n",
            "n_scaffolds",
            "spearman",
            "spearman_lower_95",
            "spearman_upper_95",
            "within_scaffold_spearman",
            "between_scaffold_spearman",
            "leave_one_scaffold_min",
            "leave_one_scaffold_max",
            "within_scaffold_permutation_p",
            "within_scaffold_fdr_q",
            "within_scaffold_fdr_005",
            "interval_excludes_zero",
        ]
    ]
    training_display = training_unit_comparison.sort_values(
        ["endpoint", "delta_log_mae_compound_minus_evidence_training", "model"]
    )[
        [
            "endpoint",
            "model",
            "evidence_row_trained_log_mae",
            "log_mae",
            "delta_log_mae_compound_minus_evidence_training",
            "evidence_row_trained_spearman",
            "spearman",
        ]
    ]
    model_sensitivity_display = model_sensitivity_summary[
        [
            "left_endpoint",
            "right_endpoint",
            "n_models",
            "median_spearman",
            "minimum_spearman",
            "maximum_spearman",
            "sign_consistency_fraction",
            "all_model_families_same_sign",
        ]
    ]
    scaffold_support_display = scaffold_support[
        [
            "endpoint",
            "n_compounds",
            "n_scaffolds",
            "largest_scaffold_fraction",
            "singleton_scaffold_count",
            "effective_scaffold_count_simpson",
        ]
    ]
    domain_display = domain_performance[
        [
            "endpoint",
            "domain_stratum",
            "n_compounds",
            "n_scaffolds",
            "mae",
            "spearman",
            "interval_coverage",
        ]
    ]
    report = f"""# M3 local analysis: rigorous results available before HPC

## Scope and truth boundary

This run uses canonical internal records and already group-held-out predictions. It performs
compound balancing and {bootstrap_replicates:,} scaffold-bootstrap resamples. It does not
generate conformers, run abbreviated MD, infer missing physical rates, refit against a final
holdout, or promote a model to the decision track.

Raw PO Cmax is not modeled because dose (3 or 5 mg/kg) is compound-confounded
and no compound was measured at both doses. The reported
`po_cmax_dose_normalized` endpoint is a linear-PK sensitivity analysis only; it
remains discovery-only until within-compound dose proportionality is measured.

## Direct process coverage

{_markdown_table(coverage, ["process", "support_level", "internal_compounds", "critical_gap"])}

The current structures support hypotheses, but dissolution, permeability, enterocyte fate,
renal elimination, raw PK profiles, and dynamic hERG kinetics have no direct internal
measurement support. Hepatic and protein-binding evidence covers only four compounds.

## Retrospective group-held-out model robustness

{_markdown_table(performance_display, list(performance_display.columns))}

Intervals quantify scaffold-sampling uncertainty of the existing OOF predictions; models
were not refit inside the bootstrap. They are therefore more honest than a single metric,
but they are not prospective calibration evidence. Interval coverage must be read with
interval width; the companion audit shows that the selected hERG interval is wider on
average than the entire observed pIC50 range.

### Independent-series support and applicability domain

{_markdown_table(scaffold_support_display, list(scaffold_support_display.columns))}

{_markdown_table(domain_display, list(domain_display.columns))}

The nominal scaffold count overstates independent support when many scaffolds are
singletons or one series dominates. The Simpson effective-scaffold count reports the
concentration explicitly. Inside/outside-domain comparisons are diagnostic only because
the domain threshold has not been prospectively calibrated.

## Training-unit sensitivity

{_markdown_table(training_display, list(training_display.columns), maximum_rows=20)}

This analysis refits the four fixed core model families after collapsing each endpoint to
one equally weighted mean per compound. It tests whether repeated evidence rows influence
training. A change does not prove that one representation is biologically correct because
replicate identity and study covariates are missing; it establishes the sensitivity that a
publication must report.

## Residual covariance and hidden-process hypotheses

{_markdown_table(residual_display, list(residual_display.columns), maximum_rows=12)}

Positive IV/PO AUC residual covariance is consistent with a shared unmodeled systemic or
study-context component. Positive PO AUC/Cmax residual covariance is expected because both
belong to the oral-exposure family. These patterns localize where a missing process may
exist; they do not identify that process. Algebraically constructed residual differences
are labeled separately and cannot be counted as independent evidence. The permutation
test shuffles outcomes only within informative scaffolds; Benjamini-Hochberg correction
is applied separately to primary and algebraically derived residual families. A global
bootstrap interval that excludes zero is not called robust unless the within-scaffold
FDR gate also passes.

Because the historical Cmax endpoint is dose-confounded, no Cmax residual association is
an independent mechanistic result. Its persistence after dose normalization is a
falsification-priority signal only; formulation, dose proportionality, and source context
remain viable explanations.

### Sensitivity to model family

{_markdown_table(model_sensitivity_display, list(model_sensitivity_display.columns), maximum_rows=15)}

Associations that retain their direction across ridge, random forest, extra trees, SVR,
XGBoost, and LightGBM are less likely to be artifacts of one estimator. They remain
retrospective correlations among model errors, not biological replication.

## Joint PK–hERG evidence

{_markdown_table(joint_display, list(joint_display.columns))}

Only {summary["n_joint_pk_herg_compounds"]} compounds have overlapping exact hERG and IV
exposure evidence in this analysis. Any apparent association is series- and
protocol-confounded and is not evidence of an unavoidable PK–hERG trade-off.
The within-scaffold-centered column is the more relevant test of an analogue-series
relationship; the between-scaffold column diagnoses series separation. Leave-one-scaffold
ranges expose associations that reverse when one series is removed. Joint associations
are corrected as one six-test family; no cross-series association is promoted when its
within-scaffold permutation result fails the FDR gate.

## Measurement and protocol limits

- The canonical table contains repeated measurement signatures that can be source echoes;
  replicate identities are absent, so they were not used to estimate a biological noise
  ceiling.
- Internal PK studies have dose and route, but strain, sex, formulation, vehicle, and
  matrix are absent.
- hERG protocols lack: {", ".join(missing_herg_fields) if missing_herg_fields else "no audited fields"}.
- Static endpoint accuracy therefore cannot validate the proposed dissolution, transport,
  metabolism, distribution, membrane, or kinetic mechanisms.

## High-quality decisions from this local run

1. Preserve the current conventional models as retrospective discovery baselines.
2. Do not add another descriptor/model family before direct process evidence exists.
3. Use the 16-compound panel only for mechanism-rich assay design. It is outcome-informed
   and therefore cannot be the unbiased prospective model test. Use eight raw rat profiles
   plus six dynamic hERG experiments for rate identifiability.
4. Keep all conformer, membrane, receptor, and free-energy production behind the declared
   HPC convergence gates.
5. Treat the residual covariance table as a preregistered hypothesis map for selecting
   experiments, not as causal proof.
"""
    atomic_write_text(output / "local_analysis_report.md", report)
    return run_summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--random-state", type=int, default=20260724)
    args = parser.parse_args()
    summary = run_local_analysis(
        args.project_root.resolve(),
        bootstrap_replicates=args.bootstrap_replicates,
        random_state=args.random_state,
    )
    print(
        "Local analysis complete: "
        f"{summary['n_pk_compounds_in_oof_analysis']} PK, "
        f"{summary['n_herg_compounds_in_oof_analysis']} hERG, "
        f"{summary['n_joint_pk_herg_compounds']} joint compounds."
    )


if __name__ == "__main__":
    main()
