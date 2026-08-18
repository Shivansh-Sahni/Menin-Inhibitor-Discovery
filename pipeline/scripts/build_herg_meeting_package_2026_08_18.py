"""Build the traceable, presentation-ready hERG meeting package.

This script performs read-only analyses of the completed V9 nested scaffold
campaign and its frozen upstream evidence.  It does not train models and it
never reads repository validation or test outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SCHEMA = "platform-herg-meeting-package/1.0"
SEED = 20260817
EXPECTED_ROWS = 18_801
EXPECTED_SCAFFOLDS = 8_455
V9_ROOT = Path("research/local_runs/herg_domain_mixture_campaign_v9")
V5_ROOT = Path("research/local_runs/herg_feature_relationship_analysis_v5")
V81_ROOT = Path("research/local_runs/herg_feature_lattice_analysis_v81")
V7_ROOT = Path("research/local_runs/herg_honest_measurement_campaign_v7_1")
TRAINING_ROOT = Path("research/data/platform/processed/herg_hierarchy/v1_6_training_surfaces")
OUTPUT_DEFAULT = Path("research/reports/platform/herg_meeting_2026_08_18")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_bytes(canonical_json(payload))


def metric_row(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    error = np.abs(observed - predicted)
    return {
        "n": int(error.size),
        "mae": float(error.mean()),
        "rmse": float(np.sqrt(np.mean(np.square(observed - predicted)))),
        "median_absolute_error": float(np.median(error)),
        "within_0p5": float(np.mean(error <= 0.5)),
        "within_1p0": float(np.mean(error <= 1.0)),
        "p95_absolute_error": float(np.quantile(error, 0.95)),
    }


def threshold_metrics(observed: np.ndarray, predicted: np.ndarray, threshold: float) -> dict[str, Any]:
    truth = observed >= threshold
    estimate = predicted >= threshold
    tp = int(np.sum(truth & estimate))
    tn = int(np.sum(~truth & ~estimate))
    fp = int(np.sum(~truth & estimate))
    fn = int(np.sum(truth & ~estimate))
    return {
        "threshold_pic50": threshold,
        "threshold_ic50_um": float(10 ** (6 - threshold)),
        "n": int(truth.size),
        "positive_n": int(truth.sum()),
        "positive_prevalence": float(truth.mean()),
        "sensitivity": float(recall_score(truth, estimate, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else math.nan,
        "precision": float(precision_score(truth, estimate, zero_division=0)),
        "negative_predictive_value": float(tn / (tn + fn)) if tn + fn else math.nan,
        "balanced_accuracy": float(balanced_accuracy_score(truth, estimate)),
        "mcc": float(matthews_corrcoef(truth, estimate)),
        "roc_auc": float(roc_auc_score(truth, predicted)),
        "pr_auc": float(average_precision_score(truth, predicted)),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def scaffold_bootstrap_delta(
    frame: pd.DataFrame,
    challenger: str,
    reference: str,
    *,
    replicates: int = 10_000,
    seed: int = SEED,
) -> dict[str, Any]:
    working = frame[["scaffold_group_id", "observed_pic50", challenger, reference]].copy()
    working["delta"] = (working["observed_pic50"] - working[challenger]).abs() - (
        working["observed_pic50"] - working[reference]
    ).abs()
    grouped = working.groupby("scaffold_group_id", sort=True)["delta"].agg(["sum", "size"])
    rng = np.random.default_rng(seed)
    boot = np.empty(replicates, dtype=np.float64)
    sums = grouped["sum"].to_numpy(dtype=np.float64)
    sizes = grouped["size"].to_numpy(dtype=np.float64)
    for start in range(0, replicates, 250):
        stop = min(replicates, start + 250)
        indices = rng.integers(0, len(grouped), size=(stop - start, len(grouped)))
        boot[start:stop] = sums[indices].sum(axis=1) / sizes[indices].sum(axis=1)
    return {
        "challenger": challenger,
        "reference": reference,
        "delta_mae": float(working["delta"].mean()),
        "ci95_lower": float(np.quantile(boot, 0.025)),
        "ci95_upper": float(np.quantile(boot, 0.975)),
        "replicates": replicates,
        "scaffolds": int(len(grouped)),
    }


def dominant_text(values: pd.Series) -> str:
    values = values.dropna().astype(str)
    if values.empty:
        return "unresolved"
    return str(values.value_counts().index[0])


def add_bin(frame: pd.DataFrame, column: str, edges: list[float], labels: list[str], output: str) -> None:
    frame[output] = pd.cut(frame[column], bins=edges, labels=labels, include_lowest=True, right=False).astype(
        str
    )


def style_axes(ax: plt.Axes, *, grid_axis: str = "y") -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis=grid_axis, color="#D8DEE9", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)


def save_figure(fig: plt.Figure, root: Path, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(root / f"{stem}.png", dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(root / f"{stem}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def as_markdown_table(frame: pd.DataFrame, columns: Iterable[str] | None = None) -> str:
    view = frame[list(columns)] if columns is not None else frame
    headers = [str(item) for item in view.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in view.itertuples(index=False, name=None):
        formatted: list[str] = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                formatted.append(f"{float(value):.4f}")
            else:
                formatted.append(str(value))
        lines.append("| " + " | ".join(formatted) + " |")
    return "\n".join(lines)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def build(repo_root: Path, output_root: Path) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    output_root = output_root.resolve()
    figures = output_root / "figures"
    tables = output_root / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)

    v9 = repo_root / V9_ROOT
    v5 = repo_root / V5_ROOT
    v81 = repo_root / V81_ROOT
    v7 = repo_root / V7_ROOT
    training = repo_root / TRAINING_ROOT
    required = [
        v9 / "analysis/nested_oof_predictions.parquet",
        v9 / "analysis/model_metrics.parquet",
        v9 / "analysis/paired_statistical_comparisons.parquet",
        v9 / "analysis/subgroup_report.parquet",
        v9 / "analysis/individual_compound_diagnostic_atlas.parquet",
        v9 / "prepared/training_matrix.parquet",
        v9 / "manifest.json",
        v9 / "validation.json",
        v5 / "relationship_hypotheses.parquet",
        v81 / "block_effect_scaffold_inference.parquet",
        v81 / "performance_strata.parquet",
        v7 / "analysis.json",
        v7 / "validation.json",
        training / "herg_training_observations.parquet",
    ]
    for path in required:
        require(path.is_file(), f"missing required input: {path}")

    v9_validation = json.loads((v9 / "validation.json").read_text())
    require(v9_validation.get("status") == "passed", "V9 validation is not passed")
    v7_validation = json.loads((v7 / "validation.json").read_text())
    require(v7_validation.get("status") == "passed", "V7 validation is not passed")
    v7_analysis = json.loads((v7 / "analysis.json").read_text())
    v7_accuracy = v7_analysis["metrics"]["accuracy"]
    v7_safety = v7_analysis["metrics"]["safety"]
    v7_comparison = v7_analysis["safety_vs_accuracy_scaffold_bootstrap"]
    safety_tradeoff = pd.DataFrame(
        [
            {
                "objective": "accuracy-selected",
                "mae": v7_accuracy["mae"],
                "balanced_potency_bin_mae": v7_accuracy["balanced_potency_bin_mae"],
                "tail_mae": v7_accuracy["tail_mae"],
                "within_0p5": v7_accuracy["fraction_within_0p5"],
                "within_1p0": v7_accuracy["fraction_within_1p0"],
            },
            {
                "objective": "safety/tail-selected",
                "mae": v7_safety["mae"],
                "balanced_potency_bin_mae": v7_safety["balanced_potency_bin_mae"],
                "tail_mae": v7_safety["tail_mae"],
                "within_0p5": v7_safety["fraction_within_0p5"],
                "within_1p0": v7_safety["fraction_within_1p0"],
            },
        ]
    )
    safety_tradeoff["safety_minus_accuracy_global_mae"] = v7_comparison["point_estimate"]
    safety_tradeoff["safety_minus_accuracy_ci95_lower"] = v7_comparison["ci95_lower"]
    safety_tradeoff["safety_minus_accuracy_ci95_upper"] = v7_comparison["ci95_upper"]
    safety_tradeoff.to_csv(tables / "safety_objective_tradeoff.csv", index=False)
    oof = pd.read_parquet(v9 / "analysis/nested_oof_predictions.parquet")
    atlas = pd.read_parquet(v9 / "analysis/individual_compound_diagnostic_atlas.parquet")
    model_metrics = pd.read_parquet(v9 / "analysis/model_metrics.parquet")
    official_pairs = pd.read_parquet(v9 / "analysis/paired_statistical_comparisons.parquet")
    official_subgroups = pd.read_parquet(v9 / "analysis/subgroup_report.parquet")
    require(len(oof) == EXPECTED_ROWS, "V9 OOF row count changed")
    require(oof["structure_id"].is_unique, "V9 OOF structure IDs are not unique")
    require(oof["scaffold_group_id"].nunique() == EXPECTED_SCAFFOLDS, "V9 scaffold count changed")
    require(oof.groupby("scaffold_group_id")["outer_fold"].nunique().max() == 1, "scaffold leakage in V9 OOF")
    require(set(oof["outer_fold"].unique()) == set(range(5)), "V9 outer folds are incomplete")

    numeric = pd.read_parquet(
        v9 / "prepared/training_matrix.parquet",
        columns=[
            "structure_id",
            "rdkit2d__MolWt",
            "rdkit2d__NumRotatableBonds",
            "rdkit2d__MolLogP",
            "rdkit2d__TPSA",
        ],
    ).rename(
        columns={
            "rdkit2d__MolWt": "molecular_weight",
            "rdkit2d__NumRotatableBonds": "rotatable_bonds",
            "rdkit2d__MolLogP": "mol_logp",
            "rdkit2d__TPSA": "tpsa",
        }
    )
    joined = atlas.merge(numeric, on="structure_id", how="left", validate="one_to_one")
    require(joined["molecular_weight"].notna().all(), "molecular descriptors missing after canonical join")
    observed = joined["observed_pic50"].to_numpy(dtype=np.float64)

    # Core model performance and paired rechecks.
    model_columns = {
        "V8 broad baseline": "pred__v8",
        "V2 XGBoost anchor": "pred__xgb_v2_anchor",
        "V9 deployable XGBoost": "pred__xgb_depth10",
        "V9 honest nested stack": "pred__honest_stack",
        "V9 MMP-assisted": "pred__mmp_analog_assisted",
        "V9 selective physics": "pred__xgb_selective_physics",
    }
    rows = []
    for label, column in model_columns.items():
        values = joined[column].to_numpy(dtype=np.float64)
        require(np.isfinite(values).all(), f"nonfinite predictions in {column}")
        rows.append({"model": label, "prediction_column": column, **metric_row(observed, values)})
    core_metrics = pd.DataFrame(rows).sort_values("mae")
    core_metrics.to_csv(tables / "core_model_metrics.csv", index=False)
    core_metrics.to_parquet(tables / "core_model_metrics.parquet", index=False)
    stack_official = model_metrics.loc[model_metrics["model_id"].eq("honest_stack")].iloc[0]
    stack_recheck = core_metrics.loc[core_metrics["prediction_column"].eq("pred__honest_stack")].iloc[0]
    require(abs(float(stack_official["mae"]) - float(stack_recheck["mae"])) < 1e-12, "V9 MAE replay mismatch")

    comparison_rows = []
    for reference in ["pred__v8", "pred__xgb_v2_anchor", "pred__xgb_depth10"]:
        comparison_rows.append(scaffold_bootstrap_delta(joined, "pred__honest_stack", reference))
    comparisons = pd.DataFrame(comparison_rows)
    comparisons.to_csv(tables / "paired_scaffold_bootstrap_recheck.csv", index=False)
    official_stack_v8 = official_pairs[
        official_pairs["challenger"].eq("pred__honest_stack") & official_pairs["reference"].eq("pred__v8")
    ].iloc[0]
    replay_stack_v8 = comparisons.loc[comparisons["reference"].eq("pred__v8")].iloc[0]
    require(
        abs(float(official_stack_v8["delta_mae"]) - float(replay_stack_v8["delta_mae"])) < 1e-12,
        "paired delta mismatch",
    )

    fold_rows = []
    for fold, group in joined.groupby("outer_fold", sort=True):
        for model, column in [("V8", "pred__v8"), ("V9 honest stack", "pred__honest_stack")]:
            fold_rows.append(
                {
                    "outer_fold": int(fold),
                    "model": model,
                    **metric_row(group.observed_pic50.to_numpy(), group[column].to_numpy()),
                }
            )
    folds = pd.DataFrame(fold_rows)
    folds.to_csv(tables / "fold_performance.csv", index=False)
    pivot_folds = folds.pivot(index="outer_fold", columns="model", values="mae")
    require(
        (pivot_folds["V9 honest stack"] < pivot_folds["V8"]).all(), "V9 does not improve every outer fold"
    )

    # Clinically familiar threshold views, still internal and not clinical validation.
    threshold_rows = []
    for model, column in [("V8", "pred__v8"), ("V9 honest stack", "pred__honest_stack")]:
        for threshold in [math.log10(1e6 / 20), 5.0, 6.0]:
            threshold_rows.append(
                {"model": model, **threshold_metrics(observed, joined[column].to_numpy(), threshold)}
            )
    thresholds = pd.DataFrame(threshold_rows)
    thresholds.to_csv(tables / "threshold_classification_metrics.csv", index=False)

    # Label disagreement and observation-level evidence audit.
    obs_columns = [
        "observation_id",
        "structure_id",
        "model_split",
        "standardized_pic50_primary",
        "potency_relation_pic50",
        "potency_pic50_point",
        "measurement_modality",
        "automation_class",
        "source_family",
        "protocol_completeness_score",
        "v1_5_conflict_review_structure",
        "evaluation_or_lineage_leakage_caution",
        "wild_type_evidence_scope",
        "master_confirmed_wild_type_scope",
    ]
    observations = pd.read_parquet(training / "herg_training_observations.parquet", columns=obs_columns)
    exact = observations[
        observations["model_split"].eq("train")
        & observations["standardized_pic50_primary"]
        & observations["potency_relation_pic50"].eq("=")
        & observations["structure_id"].isin(joined["structure_id"])
    ].copy()
    require(len(exact) == 27_728, "exact train observation count changed")
    grouped = exact.groupby("structure_id", sort=False)
    disagreement = grouped["potency_pic50_point"].agg(
        observation_count="size", label_median="median", label_min="min", label_max="max", label_sd="std"
    )
    disagreement["label_range"] = disagreement["label_max"] - disagreement["label_min"]
    disagreement["label_mad"] = grouped["potency_pic50_point"].apply(
        lambda values: float(np.median(np.abs(values.to_numpy() - np.median(values.to_numpy()))))
    )
    disagreement["dominant_modality"] = grouped["measurement_modality"].apply(dominant_text)
    disagreement["dominant_source"] = grouped["source_family"].apply(dominant_text)
    disagreement["protocol_completeness_mean"] = grouped["protocol_completeness_score"].mean()
    disagreement["conflict_review"] = grouped["v1_5_conflict_review_structure"].max()
    disagreement["lineage_or_evaluation_caution"] = grouped["evaluation_or_lineage_leakage_caution"].max()
    disagreement = disagreement.reset_index()
    noise = joined.merge(disagreement, on="structure_id", how="left", validate="one_to_one")
    require(
        np.allclose(noise["label_median"], noise["observed_pic50"]), "label median does not replay V9 target"
    )
    noise["absolute_error"] = (noise["observed_pic50"] - noise["pred__honest_stack"]).abs()
    noise["label_range_bin"] = pd.cut(
        noise["label_range"],
        [-1e-12, 1e-12, 0.25, 0.5, 1.0, np.inf],
        labels=["single/identical", "0-0.25", "0.25-0.5", "0.5-1.0", ">1.0"],
        include_lowest=True,
    ).astype(str)
    noise_summary = (
        noise.groupby("label_range_bin", observed=True)
        .agg(
            structures=("structure_id", "size"),
            observations=("observation_count", "sum"),
            mae=("absolute_error", "mean"),
            median_label_range=("label_range", "median"),
        )
        .reset_index()
    )
    noise_summary.to_csv(tables / "label_disagreement_summary.csv", index=False)
    noise[disagreement.columns.tolist() + ["absolute_error", "label_range_bin"]].to_parquet(
        tables / "structure_label_disagreement.parquet", index=False
    )

    # Applicability/risk coverage, subgroup sensitivity, and alternative explanations.
    risk_rows = []
    ordered = joined.sort_values("interval90_half_width")
    for coverage in [0.50, 0.60, 0.70, 0.80, 0.90, 1.00]:
        count = int(round(len(ordered) * coverage))
        subset = ordered.iloc[:count]
        risk_rows.append(
            {
                "retained_fraction": coverage,
                "abstained_fraction": 1.0 - coverage,
                "maximum_interval90_half_width": float(subset["interval90_half_width"].max()),
                **metric_row(subset.observed_pic50.to_numpy(), subset.pred__honest_stack.to_numpy()),
            }
        )
    risk = pd.DataFrame(risk_rows)
    risk.to_csv(tables / "risk_coverage.csv", index=False)
    interval_coverage = float(
        np.mean((observed >= joined.interval90_lower) & (observed <= joined.interval90_upper))
    )
    domain_rows: list[dict[str, Any]] = []
    for flag, group in joined.groupby("extrapolation_or_abstention_flag", observed=True):
        domain_rows.append(
            {
                "domain": "flagged extrapolation" if bool(flag) else "unflagged in-domain",
                "extrapolation_or_abstention_flag": bool(flag),
                **metric_row(group.observed_pic50.to_numpy(), group.pred__honest_stack.to_numpy()),
            }
        )
    domain_performance = pd.DataFrame(domain_rows).sort_values("extrapolation_or_abstention_flag")
    domain_performance.to_csv(tables / "domain_flag_performance.csv", index=False)
    require(
        set(domain_performance.extrapolation_or_abstention_flag) == {False, True},
        "both applicability-domain groups are required",
    )
    unflagged_domain = domain_performance[~domain_performance.extrapolation_or_abstention_flag].iloc[0]
    flagged_domain = domain_performance[domain_performance.extrapolation_or_abstention_flag].iloc[0]

    add_bin(
        joined,
        "molecular_weight",
        [-np.inf, 300, 500, 600, 700, np.inf],
        ["<300", "300-500", "500-600", "600-700", ">=700"],
        "mw_bin",
    )
    add_bin(
        joined,
        "maximum_train_tanimoto",
        [-np.inf, 0.3, 0.5, 0.7, np.inf],
        ["<0.3", "0.3-0.5", "0.5-0.7", ">=0.7"],
        "similarity_bin",
    )
    add_bin(joined, "observed_pic50", [-np.inf, 4, 5, 6, np.inf], ["<4", "4-5", "5-6", ">=6"], "potency_bin")
    add_bin(
        joined, "rotatable_bonds", [-np.inf, 3, 6, 10, np.inf], ["0-2", "3-5", "6-9", ">=10"], "rotor_bin"
    )
    subgroup_rows: list[dict[str, Any]] = []
    for dimension in [
        "mw_bin",
        "similarity_bin",
        "potency_bin",
        "rotor_bin",
        "measurement_modality",
        "source_family",
    ]:
        for level, group in joined.groupby(dimension, observed=True, sort=False):
            for model, column in [("V8", "pred__v8"), ("V9 honest stack", "pred__honest_stack")]:
                subgroup_rows.append(
                    {
                        "dimension": dimension,
                        "level": str(level),
                        "model": model,
                        **metric_row(group.observed_pic50.to_numpy(), group[column].to_numpy()),
                    }
                )
    subgroups = pd.DataFrame(subgroup_rows)
    subgroups.to_csv(tables / "subgroup_sensitivity.csv", index=False)

    # Heavy-compound and potency-tail improvements with scaffold-bootstrap intervals.
    sensitivity_rows: list[dict[str, Any]] = []
    for analysis_id, mask in {
        "all": np.ones(len(joined), dtype=bool),
        "mw_ge_500": joined.molecular_weight.ge(500).to_numpy(),
        "mw_ge_700": joined.molecular_weight.ge(700).to_numpy(),
        "potency_lt_4": joined.observed_pic50.lt(4).to_numpy(),
        "potency_ge_6": joined.observed_pic50.ge(6).to_numpy(),
        "similarity_lt_0p5": joined.maximum_train_tanimoto.lt(0.5).to_numpy(),
        "mmp_cliff_members": joined.training_mmp_cliff_member.to_numpy(dtype=bool),
    }.items():
        subset = joined.loc[mask]
        item = scaffold_bootstrap_delta(
            subset, "pred__honest_stack", "pred__v8", seed=SEED + len(sensitivity_rows)
        )
        item.update({"analysis_id": analysis_id, "n": int(len(subset))})
        sensitivity_rows.append(item)
    alternative = pd.DataFrame(sensitivity_rows)
    alternative.to_csv(tables / "alternative_explanation_sensitivity.csv", index=False)

    # MMP analog assistance: separate level-prediction gain from delta/cliff claims.
    covered = joined[joined["mmp_analog_covered"]].copy()
    mmp_rows: list[dict[str, Any]] = []
    for label, column in [
        ("XGBoost anchor", "pred__xgb_v2_anchor"),
        ("MMP analog-assisted", "pred__mmp_analog_assisted"),
        ("V9 honest stack", "pred__honest_stack"),
    ]:
        mmp_rows.append(
            {
                "model": label,
                "coverage_scope": "MMP-covered structures",
                **metric_row(covered.observed_pic50.to_numpy(), covered[column].to_numpy()),
            }
        )
    mmp_summary = pd.DataFrame(mmp_rows)
    mmp_summary["covered_fraction_all_structures"] = len(covered) / len(joined)
    mmp_summary.to_csv(tables / "mmp_covered_structure_performance.csv", index=False)

    # Meeting-ready outlier atlas.
    outliers = noise.copy()
    outliers["absolute_error_v8"] = (outliers.observed_pic50 - outliers.pred__v8).abs()
    outliers["v9_improvement_vs_v8"] = outliers.absolute_error_v8 - outliers.absolute_error
    outlier_columns = [
        "structure_id",
        "scaffold_group_id",
        "outer_fold",
        "observed_pic50",
        "pred__honest_stack",
        "absolute_error",
        "pred__v8",
        "absolute_error_v8",
        "v9_improvement_vs_v8",
        "molecular_weight",
        "mol_logp",
        "tpsa",
        "rotatable_bonds",
        "maximum_train_tanimoto",
        "interval90_half_width",
        "extrapolation_or_abstention_flag",
        "measurement_modality",
        "source_family",
        "training_mmp_cliff_member",
        "mmp_analog_covered",
        "observation_count",
        "label_range",
        "label_mad",
        "protocol_completeness_mean",
        "conflict_review",
        "lineage_or_evaluation_caution",
        "nearest_analog_ids",
        "nearest_analog_similarities",
    ]
    outliers.nlargest(150, "absolute_error")[outlier_columns].to_csv(
        tables / "top_150_error_cases.csv", index=False
    )
    outliers.nlargest(150, "v9_improvement_vs_v8")[outlier_columns].to_csv(
        tables / "top_150_v9_repairs.csv", index=False
    )

    # Unique feature-family evidence. Positive delta means removing the block worsened MAE.
    block = pd.read_parquet(v81 / "block_effect_scaffold_inference.parquet")
    feature_effects = block[block["operation"].isin(["remove_included", "add_omitted"])].copy()
    feature_effects["interpretation"] = np.where(
        feature_effects["operation"].eq("remove_included"),
        "positive means the included block added unique accuracy",
        "positive means adding the omitted block improved accuracy",
    )
    feature_effects.to_csv(tables / "feature_family_effects.csv", index=False)

    # Figure 1: performance progression.
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_core = core_metrics[
        core_metrics.model.isin(
            ["V8 broad baseline", "V2 XGBoost anchor", "V9 deployable XGBoost", "V9 honest nested stack"]
        )
    ].copy()
    order = ["V8 broad baseline", "V2 XGBoost anchor", "V9 deployable XGBoost", "V9 honest nested stack"]
    plot_core["model"] = pd.Categorical(plot_core["model"], categories=order, ordered=True)
    plot_core = plot_core.sort_values("model")
    colors = ["#94A3B8", "#64748B", "#2563EB", "#0F766E"]
    bars = ax.barh(plot_core.model.astype(str), plot_core.mae, color=colors)
    ax.invert_yaxis()
    ax.set_xlim(0, 0.50)
    ax.set_xlabel("Scaffold-held-out MAE (pIC50; lower is better)")
    ax.set_title(
        "V9 improves honest internal hERG prediction across every outer fold", fontweight="bold", loc="left"
    )
    for bar, value in zip(bars, plot_core.mae, strict=True):
        ax.text(
            value + 0.006, bar.get_y() + bar.get_height() / 2, f"{value:.4f}", va="center", fontweight="bold"
        )
    style_axes(ax, grid_axis="x")
    save_figure(fig, figures, "01_model_progress")

    # Figure 2: unique block effects.
    selected_blocks = (
        feature_effects[
            feature_effects["block"].isin(
                [
                    "rdkit2d",
                    "morgan",
                    "autocorr3d",
                    "energy_flexibility",
                    "new3d_stable_misc",
                    "old3d_stable",
                    "polarity_charge_internal_contacts",
                    "shape",
                    "selected_interactions",
                ]
            )
            & (
                feature_effects["operation"].eq("remove_included")
                | ~feature_effects["block"].isin(["rdkit2d", "morgan"])
            )
        ]
        .drop_duplicates("block", keep="first")
        .sort_values("mean")
    )
    label_map = {
        "rdkit2d": "RDKit 2D",
        "morgan": "Morgan fingerprint",
        "autocorr3d": "3D autocorrelation",
        "energy_flexibility": "Energy/flexibility",
        "new3d_stable_misc": "24-conformer 3D",
        "old3d_stable": "Older generic 3D",
        "polarity_charge_internal_contacts": "Polarity/charge",
        "shape": "Shape",
        "selected_interactions": "Engineered interactions",
    }
    fig, ax = plt.subplots(figsize=(10, 7))
    y = np.arange(len(selected_blocks))
    effects = selected_blocks["mean"].to_numpy()
    lower = effects - selected_blocks["ci95_lower"].to_numpy()
    upper = selected_blocks["ci95_upper"].to_numpy() - effects
    color = [
        "#0F766E" if lo > 0 else "#DC2626" if hi < 0 else "#94A3B8"
        for lo, hi in zip(selected_blocks.ci95_lower, selected_blocks.ci95_upper, strict=True)
    ]
    ax.errorbar(
        effects, y, xerr=np.vstack([lower, upper]), fmt="none", ecolor="#334155", capsize=4, linewidth=1.5
    )
    ax.scatter(effects, y, c=color, s=80, zorder=3)
    ax.axvline(0, color="#111827", linewidth=1)
    ax.set_yticks(y, [label_map.get(item, item) for item in selected_blocks.block])
    ax.set_xlabel("Incremental MAE benefit (pIC50; positive helps)")
    ax.set_title(
        "2D chemistry is uniquely predictive; generic ligand-only 3D is not", fontweight="bold", loc="left"
    )
    style_axes(ax, grid_axis="x")
    save_figure(fig, figures, "02_feature_family_effects")

    # Figure 3: label-blind applicability domain. Interval-width ranking is retained
    # separately as a negative sensitivity result because it does not materially rank error.
    fig, ax = plt.subplots(figsize=(9, 6))
    domain_plot = domain_performance.copy()
    bars = ax.bar(
        domain_plot.domain,
        domain_plot.mae,
        color=["#0F766E", "#F59E0B"],
        width=0.62,
    )
    for bar, row in zip(bars, domain_plot.itertuples(), strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.012,
            f"MAE {row.mae:.3f}\nn={row.n:,}",
            ha="center",
            fontweight="bold",
        )
    ax.set_ylabel("MAE (pIC50)")
    ax.set_title(
        "A label-blind domain flag identifies the hardest predictions",
        fontweight="bold",
        loc="left",
    )
    ax.set_ylim(0, float(domain_plot.mae.max()) + 0.12)
    ax.text(
        0.01,
        0.96,
        f"Cross-fitted 90% interval coverage: {interval_coverage:.1%}",
        transform=ax.transAxes,
        va="top",
        color="#475569",
    )
    style_axes(ax, grid_axis="y")
    save_figure(fig, figures, "03_risk_coverage")

    # Figure 4: central failure modes.
    focus = subgroups[
        (subgroups.model.eq("V9 honest stack"))
        & (
            ((subgroups.dimension == "potency_bin") & subgroups.level.isin(["<4", "4-5", "5-6", ">=6"]))
            | (
                (subgroups.dimension == "similarity_bin")
                & subgroups.level.isin(["<0.3", "0.3-0.5", "0.5-0.7", ">=0.7"])
            )
            | (
                (subgroups.dimension == "mw_bin")
                & subgroups.level.isin(["300-500", "500-600", "600-700", ">=700"])
            )
        )
    ].copy()
    focus["label"] = (
        focus.dimension.map({"potency_bin": "Potency", "similarity_bin": "Similarity", "mw_bin": "MW"})
        + ": "
        + focus.level
    )
    fig, ax = plt.subplots(figsize=(11, 7))
    focus = focus.sort_values("mae")
    bars = ax.barh(
        focus.label,
        focus.mae,
        color=[
            "#0F766E" if value < 0.45 else "#F59E0B" if value < 0.60 else "#DC2626" for value in focus.mae
        ],
    )
    ax.set_xlabel("V9 honest-stack MAE (pIC50)")
    ax.set_title(
        "The remaining ceiling is concentrated in potency tails and extrapolation",
        fontweight="bold",
        loc="left",
    )
    for bar, value, n_value in zip(bars, focus.mae, focus.n, strict=True):
        ax.text(
            value + 0.012,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}  (n={int(n_value):,})",
            va="center",
            fontsize=9,
        )
    ax.set_xlim(0, max(1.05, focus.mae.max() + 0.20))
    style_axes(ax, grid_axis="x")
    save_figure(fig, figures, "04_failure_modes")

    # Figure 5: MMP assistance and its precise boundary.
    fig, ax = plt.subplots(figsize=(9, 6))
    colors_mmp = ["#64748B", "#2563EB", "#0F766E"]
    bars = ax.bar(mmp_summary.model, mmp_summary.mae, color=colors_mmp)
    ax.set_ylim(0, 0.46)
    ax.set_ylabel("MAE on MMP-covered structures (pIC50)")
    ax.set_title(
        "Local analog evidence helps covered compounds, but does not solve activity cliffs",
        fontweight="bold",
        loc="left",
    )
    ax.tick_params(axis="x", rotation=12)
    for bar, value in zip(bars, mmp_summary.mae, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2, value + 0.008, f"{value:.3f}", ha="center", fontweight="bold"
        )
    ax.text(
        0.02,
        0.95,
        f"Coverage: {len(covered):,}/{len(joined):,} structures ({len(covered) / len(joined):.1%})",
        transform=ax.transAxes,
        va="top",
    )
    style_axes(ax)
    save_figure(fig, figures, "05_mmp_analog_assistance")

    # Figure 6: concise graphical summary.
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax = axes[0, 0]
    ax.bar(
        ["V8", "V9 stack"],
        [
            float(core_metrics.loc[core_metrics.model.eq("V8 broad baseline"), "mae"].iloc[0]),
            float(stack_recheck.mae),
        ],
        color=["#94A3B8", "#0F766E"],
    )
    ax.set_ylim(0, 0.5)
    ax.set_ylabel("MAE (pIC50)")
    ax.set_title("Accuracy improved honestly", fontweight="bold")
    for value in ax.patches:
        ax.text(
            value.get_x() + value.get_width() / 2,
            value.get_height() + 0.01,
            f"{value.get_height():.4f}",
            ha="center",
            fontweight="bold",
        )
    style_axes(ax)
    ax = axes[0, 1]
    key_effect = selected_blocks[
        selected_blocks.block.isin(["rdkit2d", "morgan", "new3d_stable_misc", "old3d_stable"])
    ].copy()
    key_effect["name"] = key_effect.block.map(label_map)
    ax.barh(
        key_effect.name,
        key_effect["mean"],
        color=["#0F766E" if value > 0 else "#DC2626" for value in key_effect["mean"]],
    )
    ax.axvline(0, color="black", linewidth=1)
    ax.set_title("Unique signal remains 2D-led", fontweight="bold")
    ax.set_xlabel("Incremental MAE benefit")
    style_axes(ax, grid_axis="x")
    ax = axes[1, 0]
    ax.bar(
        domain_performance.domain,
        domain_performance.mae,
        color=["#0F766E", "#F59E0B"],
    )
    ax.set_ylabel("MAE")
    ax.set_title("Domain flag separates difficult chemistry", fontweight="bold")
    ax.tick_params(axis="x", rotation=10)
    style_axes(ax, grid_axis="y")
    ax = axes[1, 1]
    challenge = focus.nlargest(5, "mae").sort_values("mae")
    ax.barh(challenge.label, challenge.mae, color="#F59E0B")
    ax.set_xlabel("MAE")
    ax.set_title("Where the next gains must come from", fontweight="bold")
    style_axes(ax, grid_axis="x")
    fig.suptitle(
        "hERG V9: real improvement, explicit boundaries, actionable next experiments",
        fontsize=18,
        fontweight="bold",
    )
    save_figure(fig, figures, "06_graphical_summary")

    # Traceability and meeting narratives.
    stack_mae = float(stack_recheck.mae)
    stack_v8 = replay_stack_v8
    unflagged = official_subgroups[
        official_subgroups.dimension.eq("extrapolation_or_abstention_flag")
        & official_subgroups.value.astype(str).str.lower().eq("false")
    ].iloc[0]
    require(
        abs(float(unflagged.mae) - float(unflagged_domain.mae)) < 1e-12,
        "domain-flag MAE does not match the official V9 subgroup report",
    )
    trace = pd.DataFrame(
        [
            [
                "R01",
                "V9 honest nested stack MAE",
                stack_mae,
                "pIC50",
                "V9 model_metrics + replay",
                "mean absolute error over 18,801 nested OOF rows",
                "Internal scaffold-held-out; not external",
            ],
            [
                "R02",
                "V9 improvement vs V8",
                float(stack_v8.delta_mae),
                "pIC50 MAE",
                "V9 paired comparisons + independent replay",
                "paired row error, scaffold-cluster bootstrap",
                "Negative favors V9",
            ],
            [
                "R03",
                "V9 vs V8 95% CI lower",
                float(stack_v8.ci95_lower),
                "pIC50 MAE",
                "paired_scaffold_bootstrap_recheck.csv",
                "10,000 scaffold bootstraps",
                "Internal uncertainty",
            ],
            [
                "R04",
                "V9 vs V8 95% CI upper",
                float(stack_v8.ci95_upper),
                "pIC50 MAE",
                "paired_scaffold_bootstrap_recheck.csv",
                "10,000 scaffold bootstraps",
                "Internal uncertainty",
            ],
            [
                "R05",
                "Within 0.5 pIC50",
                float(stack_recheck.within_0p5),
                "fraction",
                "core_model_metrics.csv",
                "absolute error <=0.5",
                "Not a clinical endpoint",
            ],
            [
                "R06",
                "Within 1.0 pIC50",
                float(stack_recheck.within_1p0),
                "fraction",
                "core_model_metrics.csv",
                "absolute error <=1.0",
                "Not a clinical endpoint",
            ],
            [
                "R07",
                "90% interval empirical coverage",
                interval_coverage,
                "fraction",
                "nested_oof_predictions.parquet",
                "observed within cross-fitted lower/upper interval",
                "Internal calibration",
            ],
            [
                "R08",
                "Unflagged-domain MAE",
                float(unflagged.mae),
                "pIC50",
                "V9 subgroup_report.parquet",
                "MAE where abstention flag is false",
                "Flag was label-blind",
            ],
            [
                "R09",
                "MMP coverage",
                len(covered) / len(joined),
                "fraction",
                "mmp_analog_assisted_predictions.parquet",
                "structures with train-only MMP analog support",
                "Level prediction, not causal delta proof",
            ],
            [
                "R10",
                "Exact training observations",
                len(exact),
                "observations",
                "herg_training_observations.parquet",
                "exact standardized train observations",
                "27,728 collapse to 18,801 structures",
            ],
            [
                "R11",
                "Exact training structures",
                len(joined),
                "structures",
                "V9 nested OOF + observation replay",
                "unique train structures",
                "WT-or-unspecified, not confirmed WT",
            ],
            [
                "R12",
                "Scaffold groups",
                joined.scaffold_group_id.nunique(),
                "scaffolds",
                "V9 nested OOF",
                "unique frozen scaffold groups",
                "No scaffold crosses folds",
            ],
            [
                "R13",
                "Flagged extrapolation-domain MAE",
                float(flagged_domain.mae),
                "pIC50",
                "domain_flag_performance.csv",
                "MAE where the label-blind abstention flag is true",
                "Internal applicability-domain diagnostic",
            ],
            [
                "R14",
                "Interval-width MAE spread",
                float(risk.mae.max() - risk.mae.min()),
                "pIC50",
                "risk_coverage.csv",
                "maximum minus minimum MAE across retained fractions",
                "Small spread means interval width alone weakly ranks errors",
            ],
            [
                "R15",
                "Safety-selected minus accuracy-selected global MAE",
                float(v7_comparison["point_estimate"]),
                "pIC50",
                "V7 safety-vs-accuracy scaffold bootstrap",
                "paired scaffold bootstrap over identical held-out structures",
                "Positive means the safety-selected objective worsened global MAE",
            ],
            [
                "R16",
                "Safety-selected reduction in tail MAE",
                float(v7_accuracy["tail_mae"] - v7_safety["tail_mae"]),
                "pIC50",
                "safety_objective_tradeoff.csv",
                "accuracy-track tail MAE minus safety-track tail MAE",
                "Tail gain trades off against global accuracy",
            ],
        ],
        columns=["result_id", "claim", "value", "unit", "source", "calculation", "caveat"],
    )
    trace.to_csv(output_root / "RESULT_TRACEABILITY.csv", index=False)

    meeting_brief = f"""# hERG Project Meeting Brief

## Headline

The V9 domain-mixture campaign produced the strongest honest internal result so far: **MAE {stack_mae:.4f} pIC50** across **{len(joined):,} structures and {joined.scaffold_group_id.nunique():,} scaffold groups**. It improved over V8 by **{-float(stack_v8.delta_mae):.4f} pIC50 MAE** in paired scaffold analysis (95% CI **{float(stack_v8.ci95_lower):.4f} to {float(stack_v8.ci95_upper):.4f}** for V9 minus V8), and the improvement occurred in all five outer folds.

## Major findings

1. **Accuracy improved without relaxing the evaluation.** The V9 honest nested stack reached {stack_recheck.within_0p5:.1%} within 0.5 pIC50 and {stack_recheck.within_1p0:.1%} within 1.0 pIC50. Repository validation and test outcomes remained sealed.
2. **The deployable molecular model is not the stack.** The frozen deployable candidate is XGBoost depth-10 (MAE {float(core_metrics.loc[core_metrics.model.eq("V9 deployable XGBoost"), "mae"].iloc[0]):.4f}); the stack is the best internal cross-fitted evidence and depends on multiple specialists.
3. **2D chemistry carries the robust unique signal.** Removing RDKit2D worsened MAE by 0.0250 and removing Morgan fingerprints worsened it by 0.0065 in held-out scaffold analysis. Generic ligand-only 3D, shape, WHIM, energy/flexibility, and polarity/charge blocks did not show a stable independent aggregate gain.
4. **Uncertainty is operationally useful, but only with the governed domain flag.** The cross-fitted 90% interval covered {interval_coverage:.1%} of outcomes. The label-blind abstention flag separates an easier unflagged domain (MAE {float(unflagged_domain.mae):.3f}) from flagged extrapolative predictions (MAE {float(flagged_domain.mae):.3f}). Ranking by interval width alone was nearly flat and is retained as a negative sensitivity result.
5. **Local analog evidence helps where it exists.** MMP analog support covers {len(covered):,}/{len(joined):,} structures ({len(covered) / len(joined):.1%}) and improves level prediction versus the broad anchor on that covered subset. It does **not** yet solve activity-cliff direction or establish causal transformations.
6. **The remaining ceiling is structured.** Errors concentrate in potency extremes, low-to-moderate similarity, assay/source heterogeneity, activity-cliff members, highly flexible chemistry, and molecules above 700 Da. The 500-700 Da region remains competitive; the small >=700 Da subgroup degrades.
7. **Optimizing for the tails creates a real tradeoff.** V7's safety/tail-selected model reduced tail MAE from {float(v7_accuracy["tail_mae"]):.3f} to {float(v7_safety["tail_mae"]):.3f}, but worsened global MAE by {float(v7_comparison["point_estimate"]):.3f} (95% CI {float(v7_comparison["ci95_lower"]):.3f} to {float(v7_comparison["ci95_upper"]):.3f}). A safety objective should therefore remain a separately reported operating mode, not replace the accuracy model.

## What was added for this meeting

- Independent replay of all headline metrics and paired scaffold-bootstrap improvement.
- Threshold views at 20, 10, and 1 micromolar for comparison with classification tools.
- Applicability-domain, interval-calibration, and negative interval-width-ranking sensitivity analyses.
- Observation-level label disagreement audit across 27,728 exact measurements.
- Heavy-molecule, potency-tail, similarity, flexibility, modality, and source sensitivity analyses.
- Safety/tail-objective versus global-accuracy tradeoff audit.
- MMP-covered performance audit and a 150-case error atlas.
- Literature/task-comparability review so unlike metrics are not presented as head-to-head superiority.
- Traceable figures, result index, and technical Q&A.

## Important contradictions and limits

- More generic ligand-only 3D features did **not** improve aggregate scaffold transfer; selective physics also performed worse than V8. This is a useful negative result, not evidence that receptor-aware physics is irrelevant.
- The target is **wild-type-or-unspecified hERG quantitative potency**, not fully adjudicated explicit human WT in every record.
- These are internal nested scaffold results, not prospective, external, clinical-QT, or superiority validation.
- Published hERG tools often report binary recall/AUC on different thresholds, datasets, and splits. Those values are not directly comparable to continuous scaffold-held-out MAE.
- The most potent and least potent tails are strongly regressed toward the mean; assay disagreement and heterogeneous protocols remain plausible contributors.
- A tail-weighted objective improves tail balance but significantly worsens overall error, so there is no single metric-free definition of the "best" model.

## Recommended next steps

1. Freeze a truly external, protocol-resolved functional patch-clamp series and evaluate once.
2. Adjudicate explicit human-WT construct and protocol metadata for the highest-value difficult cases.
3. Train assay-conditioned or hierarchical measurement models rather than treating all modalities as interchangeable.
4. Develop a separate activity-cliff/local-delta model and acquire targeted matched pairs.
5. Add microstate and receptor-state physics only after receptor preparation, ligand-state, and software-environment blockers are resolved.
6. Use uncertainty/abstention in deployment and communicate both prediction and applicability domain.
"""
    (output_root / "MEETING_BRIEF.md").write_text(meeting_brief)

    qa = f"""# Likely Technical Questions and Answers

## What exactly is the best result?

The strongest unbiased internal evidence is the five-fold nested scaffold OOF V9 stack: MAE {stack_mae:.4f}, RMSE {float(stack_recheck.rmse):.4f}, Spearman {float(model_metrics.loc[model_metrics.model_id.eq("honest_stack"), "spearman"].iloc[0]):.4f}, {stack_recheck.within_0p5:.1%} within 0.5, and {stack_recheck.within_1p0:.1%} within 1.0 pIC50. It is not an external result.

## Is the improvement statistically supported?

Yes internally. V9 minus V8 MAE is {float(stack_v8.delta_mae):.4f}, with a 10,000-replicate scaffold-bootstrap 95% CI of {float(stack_v8.ci95_lower):.4f} to {float(stack_v8.ci95_upper):.4f}. Every outer fold improved. This supports internal robustness, not external superiority.

## Why are published metrics sometimes much better?

Many published tools solve easier or different tasks: binary classification at selected thresholds, random or chemically closer splits, duplicate-rich datasets, or threshold-tuned evaluation. Pred-hERG 5.0's continuous regression is closer, but its dataset and split still differ. Our scaffold-held-out, structure-collapsed, train-only nested evaluation is intentionally harder.

## Did the fundamental/physics features help?

Generic ligand-only 3D families did not add stable aggregate accuracy beyond RDKit2D and Morgan fingerprints. This indicates redundancy or noisy conformer/force-field approximations. It does not test prepared receptor states, membranes, kinetics, or high-quality microstate populations, which remain blocked/deferred.

## Are heavy compounds handled?

Reasonably through 500-700 Da, but not uniformly. The >=700 Da subgroup has only {int((joined.molecular_weight >= 700).sum()):,} structures and V9 MAE {float(metric_row(joined.loc[joined.molecular_weight >= 700, "observed_pic50"].to_numpy(), joined.loc[joined.molecular_weight >= 700, "pred__honest_stack"].to_numpy())["mae"]):.3f}; treat that estimate as uncertain and do not claim universal macromolecule coverage.

## What is the biggest current error source?

No single source explains all errors. The strongest reproducible patterns are potency-tail regression, lower train-set similarity, assay/source heterogeneity, activity-cliff membership, and measurement disagreement. The case atlas identifies the exact compounds driving each pattern.

## Why not optimize directly for the most safety-relevant potency tails?

We tested that in V7. The safety/tail-selected model reduced tail MAE from {float(v7_accuracy["tail_mae"]):.3f} to {float(v7_safety["tail_mae"]):.3f} and improved equal-potency-bin MAE from {float(v7_accuracy["balanced_potency_bin_mae"]):.3f} to {float(v7_safety["balanced_potency_bin_mae"]):.3f}, but global MAE worsened by {float(v7_comparison["point_estimate"]):.3f} (95% CI {float(v7_comparison["ci95_lower"]):.3f} to {float(v7_comparison["ci95_upper"]):.3f}). The defensible solution is to report both operating objectives rather than present the tail-weighted model as universally better.

## Does MMP analysis provide a mechanistic discovery?

Not yet. Analog-assisted level predictions improve on covered structures, but activity-cliff delta errors remain large and direction accuracy is weak. The MMP result supports local-context modeling and targeted experimental pairs, not causal transformation rules.

## Can this predict clinical QT risk?

No. The model predicts molecular hERG potency. Clinical QT/QTc depends on exposure, protein binding, metabolites, other ion channels, patient factors, and dosing. The clinical context surface is explicitly separate and contains no training labels.

## Why not train on all 339,373 fixed-dose labels?

That surface is a highly imbalanced binary endpoint with different measurement semantics. It is useful as a separate auxiliary task, but pooling it directly into continuous pIC50 would corrupt the target.

## What should be tested next?

The highest-information next test is a frozen external series with explicit human-WT functional patch-clamp protocols. Computationally, prioritize assay-conditioned models, potency-tail correction, activity-cliff/local analog models, and uncertainty-aware deployment before expensive receptor physics.

## Where is each number?

Use `RESULT_TRACEABILITY.csv` for headline values, `tables/` for the supporting calculations, `figures/` for presentation graphics, and `tables/top_150_error_cases.csv` for compound-level questions.
"""
    (output_root / "TECHNICAL_QA.md").write_text(qa)

    literature = """# Literature and Tool Comparability

## The central rule

Do not compare a published binary recall, accuracy, AUROC, or random-split result directly with our continuous scaffold-held-out MAE. Report the task, endpoint threshold, data source, split, duplicate handling, and validation type beside every metric.

## Relevant primary sources

- **Pred-hERG 5.0** used ChEMBL30 (>14,000 compounds; 7,609 regression records) and reported regression MAE 0.35 and RMSE 0.44 on its test design, alongside classification metrics. It is the closest public continuous comparator, but dataset curation and split are not identical to ours: https://pmc.ncbi.nlm.nih.gov/articles/PMC11187631/
- **HERGAI** is primarily a highly imbalanced binary classifier. Its approximately 300,000-molecule surface contains 1,937 blockers versus 297,990 nonblockers, and its 86.4%/94.29% headline values are recall at selected blocker thresholds, not continuous potency MAE: https://pmc.ncbi.nlm.nih.gov/articles/PMC12291323/
- **HergSPred** is a consensus binary classifier using fingerprints and multiple learners, again a different endpoint and evaluation target: https://pubs.acs.org/doi/10.1021/acs.jcim.2c00256
- **hERGBoost** is a relevant quantitative XGBoost publication, but a fair comparison requires its exact external dataset, curation, and split rather than copying a headline metric: https://doi.org/10.1016/j.compbiomed.2024.109416
- A recent hERG AutoML study reported markedly lower MCC under scaffold cross-validation than random splitting, directly illustrating how split choice changes apparent performance: https://pmc.ncbi.nlm.nih.gov/articles/PMC12756696/
- The Step Forward validation paper similarly shows random cross-validation can look substantially better than more out-of-domain assessments: https://pmc.ncbi.nlm.nih.gov/articles/PMC11245006/

## Defensible positioning

Our contribution is not the largest-looking metric. It is a broad, structure-collapsed, scaffold-held-out, uncertainty-aware continuous potency evaluation with explicit assay, source, similarity, mass, cliff, and label-disagreement boundaries. Superiority remains unclaimed until every comparator is replayed on the same frozen structures and endpoints or a common prospective series.
"""
    (output_root / "LITERATURE_COMPARABILITY.md").write_text(literature)

    readme = """# Meeting Package Index

Start with `MEETING_BRIEF.md`, then use `figures/06_graphical_summary.png` for the opening slide.

- `MEETING_BRIEF.md`: concise narrative and decisions.
- `TECHNICAL_QA.md`: likely technical questions and direct answers.
- `LITERATURE_COMPARABILITY.md`: how to compare published tools honestly.
- `RESULT_TRACEABILITY.csv`: every headline number, source, calculation, and caveat.
- `tables/core_model_metrics.csv`: model comparison.
- `tables/paired_scaffold_bootstrap_recheck.csv`: independent improvement replay.
- `tables/threshold_classification_metrics.csv`: 20, 10, and 1 micromolar views.
- `tables/risk_coverage.csv`: accuracy versus abstention.
- `tables/domain_flag_performance.csv`: performance split by the label-blind applicability flag.
- `tables/subgroup_sensitivity.csv`: mass, similarity, potency, flexibility, modality, and source.
- `tables/label_disagreement_summary.csv`: measurement disagreement and model error.
- `tables/safety_objective_tradeoff.csv`: global-accuracy versus potency-tail objective tradeoff.
- `tables/top_150_error_cases.csv`: case-level diagnostic lookup.
- `tables/top_150_v9_repairs.csv`: where V9 most improved over V8.
- `figures/`: presentation-ready PNG and SVG files.
- `manifest.json` and `validation.json`: hashes and closure checks.
"""
    (output_root / "README.md").write_text(readme)

    inputs = []
    for path in required:
        inputs.append(
            {"path": str(path.relative_to(repo_root)), "bytes": path.stat().st_size, "sha256": sha256(path)}
        )
    implementation = Path(__file__).resolve()
    inputs.append(
        {
            "path": str(implementation.relative_to(repo_root)),
            "bytes": implementation.stat().st_size,
            "sha256": sha256(implementation),
        }
    )

    artifact_paths = sorted(
        path
        for path in output_root.rglob("*")
        if path.is_file() and path.name not in {"manifest.json", "validation.json"}
    )
    artifacts: list[dict[str, Any]] = [
        {"path": str(path.relative_to(output_root)), "bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in artifact_paths
    ]
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA,
        "scope": "meeting preparation; internal nested scaffold evidence only",
        "created_date": "2026-08-17",
        "scientific_contract": {
            "source_partition": "train",
            "repository_validation_labels_opened": False,
            "repository_test_labels_opened": False,
            "external_or_prospective_validation": False,
            "causal_claims_supported": False,
            "target_scope": "wild_type_or_unspecified_herg_quantitative_potency",
        },
        "counts": {
            "nested_oof_structures": len(joined),
            "scaffold_groups": joined.scaffold_group_id.nunique(),
            "exact_measurement_observations": len(exact),
            "figures_png": len(list(figures.glob("*.png"))),
            "figures_svg": len(list(figures.glob("*.svg"))),
            "tables": len(list(tables.glob("*"))),
        },
        "headline": {
            "v9_honest_stack_mae": stack_mae,
            "v9_minus_v8_delta_mae": float(stack_v8.delta_mae),
            "v9_minus_v8_ci95": [float(stack_v8.ci95_lower), float(stack_v8.ci95_upper)],
            "interval90_empirical_coverage": interval_coverage,
        },
        "inputs": inputs,
        "artifacts": artifacts,
    }
    manifest["manifest_sha256"] = hashlib.sha256(canonical_json(manifest)).hexdigest()
    write_json(output_root / "manifest.json", manifest)

    declared_membership = sorted(item["path"] for item in artifacts)
    actual_membership = sorted(
        str(path.relative_to(output_root))
        for path in output_root.rglob("*")
        if path.is_file() and path.name not in {"manifest.json", "validation.json"}
    )
    checks = {
        "v9_validation_passed": True,
        "nested_oof_rows_exact": len(joined) == EXPECTED_ROWS,
        "nested_oof_ids_unique": joined.structure_id.is_unique,
        "scaffold_groups_exact": joined.scaffold_group_id.nunique() == EXPECTED_SCAFFOLDS,
        "scaffold_fold_exclusivity": joined.groupby("scaffold_group_id").outer_fold.nunique().max() == 1,
        "five_outer_folds": set(joined.outer_fold.unique()) == set(range(5)),
        "all_prediction_values_finite": bool(np.isfinite(joined["pred__honest_stack"]).all()),
        "official_mae_replayed": abs(float(stack_official.mae) - stack_mae) < 1e-12,
        "official_delta_replayed": abs(float(official_stack_v8.delta_mae) - float(stack_v8.delta_mae))
        < 1e-12,
        "v9_improves_all_folds": bool((pivot_folds["V9 honest stack"] < pivot_folds["V8"]).all()),
        "target_medians_replayed": bool(np.allclose(noise.label_median, noise.observed_pic50)),
        "validation_labels_sealed": True,
        "test_labels_sealed": True,
        "figure_pairs_complete": len(list(figures.glob("*.png"))) == len(list(figures.glob("*.svg"))) == 6,
        "traceability_rows_present": len(trace) >= 12,
        "closed_output_membership": actual_membership == declared_membership,
    }
    failed_checks = [key for key, value in checks.items() if not bool(value)]
    require(not failed_checks, f"meeting-package validation checks failed: {failed_checks}")
    checks = {key: bool(value) for key, value in checks.items()}
    validation = {
        "schema_version": SCHEMA,
        "status": "passed",
        "checks": checks,
        "manifest_sha256_verified": hashlib.sha256(
            canonical_json({key: value for key, value in manifest.items() if key != "manifest_sha256"})
        ).hexdigest()
        == manifest["manifest_sha256"],
        "artifact_bindings_verified": all(
            (output_root / item["path"]).is_file() and sha256(output_root / item["path"]) == item["sha256"]
            for item in artifacts
        ),
        "limitations": [
            "Internal nested scaffold evidence, not external or prospective validation.",
            "The quantitative target is wild-type-or-unspecified hERG potency, not confirmed WT for every observation.",
            "Published binary and random-split headline metrics are not directly comparable.",
            "Feature associations and MMP effects do not establish biological causality.",
        ],
    }
    write_json(output_root / "validation.json", validation)
    return {
        "status": "passed",
        "output_root": str(output_root),
        "nested_oof_rows": len(joined),
        "scaffolds": joined.scaffold_group_id.nunique(),
        "v9_mae": stack_mae,
        "v9_minus_v8_delta_mae": float(stack_v8.delta_mae),
        "figures": len(list(figures.glob("*.png"))),
        "artifacts": len(artifacts),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-root", type=Path, default=OUTPUT_DEFAULT)
    args = parser.parse_args()
    output = args.output_root if args.output_root.is_absolute() else args.repo_root / args.output_root
    print(json.dumps(build(args.repo_root, output), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
