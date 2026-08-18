#!/usr/bin/env python3
"""Consolidate hERG/PK mix-and-match results into a professor-ready package."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "pipeline/src"))

from menin_discovery.research_common import (  # noqa: E402
    atomic_write_csv,
    atomic_write_json,
    atomic_write_text,
)

DEFAULT_OUTPUT = ROOT / "research/reports/pk_herg/mix_match"
MODEL_FAMILIES = {
    "ridge",
    "svr",
    "random_forest",
    "extra_trees",
    "logistic",
    "svc_rbf",
}


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _workbook_records(
    frame: pd.DataFrame,
    columns: list[str] | None = None,
) -> list[dict[str, Any]]:
    selected = frame[columns].copy() if columns is not None else frame.copy()
    return selected.to_dict("records")


def _best(
    frame: pd.DataFrame,
    *,
    evaluation: str,
    regime: str,
    metric: str,
) -> pd.Series:
    subset = frame[
        frame["evaluation"].eq(evaluation)
        & frame["data_regime"].eq(regime)
        & frame["model"].isin(MODEL_FAMILIES)
    ]
    if subset.empty:
        raise ValueError(f"No candidate rows for {evaluation}/{regime}/{metric}")
    return subset.nsmallest(1, metric).iloc[0]


def _public_challenger(
    frame: pd.DataFrame,
    *,
    metric: str,
) -> tuple[pd.Series, pd.DataFrame]:
    evaluations = {"internal_scaffold_cv", "angelo_fixed_nonoverlap"}
    subset = frame[
        frame["evaluation"].isin(evaluations)
        & frame["data_regime"].str.contains("public")
        & ~frame["data_regime"].str.contains("extension")
        & frame["model"].isin(MODEL_FAMILIES)
    ].copy()
    subset["evaluation_rank_fraction"] = subset.groupby("evaluation")[metric].rank(
        method="average",
        pct=True,
    )
    keys = ["data_regime", "feature_layer", "model"]
    cross = (
        subset.groupby(keys, as_index=False)
        .agg(
            evaluations=("evaluation", "nunique"),
            mean_rank_fraction=("evaluation_rank_fraction", "mean"),
            worst_rank_fraction=("evaluation_rank_fraction", "max"),
        )
        .query("evaluations == 2")
        .sort_values(["mean_rank_fraction", "worst_rank_fraction"])
    )
    if cross.empty:
        raise ValueError("No public challenger is represented in both primary evaluations")
    selected = cross.iloc[0]
    match = subset[
        subset["data_regime"].eq(selected["data_regime"])
        & subset["feature_layer"].eq(selected["feature_layer"])
        & subset["model"].eq(selected["model"])
    ].copy()
    return selected, match


def _interval(value: Any, lower: Any, upper: Any, digits: int = 3) -> str:
    if not all(pd.notna(item) and np.isfinite(float(item)) for item in (value, lower, upper)):
        return f"{float(value):.{digits}f}" if pd.notna(value) else "NA"
    return f"{float(value):.{digits}f} [{float(lower):.{digits}f}, {float(upper):.{digits}f}]"


def _gain_text(
    gains: pd.DataFrame,
    row: pd.Series,
    *,
    continuous: bool,
) -> str:
    match = gains[
        gains["evaluation"].eq(row["evaluation"])
        & gains["data_regime"].eq(row["data_regime"])
        & gains["feature_layer"].eq(row["feature_layer"])
        & gains["model"].eq(row["model"])
    ]
    if match.empty:
        return "No paired internal-only comparator with the same representation/model."
    gain = match.iloc[0]
    if continuous:
        return (
            f"paired ΔMAE {gain['external_minus_internal_only_mae']:+.3f} "
            f"[{gain['delta_lower_95']:+.3f}, {gain['delta_upper_95']:+.3f}]"
        )
    return (
        f"paired ΔBrier {gain['external_minus_internal_brier']:+.3f} "
        f"[{gain['brier_delta_lower_95']:+.3f}, "
        f"{gain['brier_delta_upper_95']:+.3f}]"
    )


def _key_results(
    continuous: pd.DataFrame,
    binary: pd.DataFrame,
    pk: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    continuous_anchor = _best(
        continuous,
        evaluation="internal_scaffold_cv",
        regime="internal_only",
        metric="mae",
    )
    continuous_same_series = _best(
        continuous,
        evaluation="angelo_augmented_scaffold_cv",
        regime="internal_plus_extension",
        metric="mae",
    )
    continuous_public_selection, continuous_public_rows = _public_challenger(
        continuous,
        metric="mae",
    )
    binary_anchor = _best(
        binary,
        evaluation="internal_scaffold_cv",
        regime="internal_only",
        metric="brier",
    )
    binary_same_series = _best(
        binary,
        evaluation="angelo_augmented_scaffold_cv",
        regime="internal_plus_extension",
        metric="brier",
    )
    binary_public_selection, binary_public_rows = _public_challenger(
        binary,
        metric="brier",
    )

    rows: list[dict[str, Any]] = []

    def add_continuous(role: str, row: pd.Series, limitation: str) -> None:
        rows.append(
            {
                "domain": "hERG continuous",
                "model_role": role,
                "evaluation": row["evaluation"],
                "data_regime": row["data_regime"],
                "feature_layer": row["feature_layer"],
                "model": row["model"],
                "n": int(row["n"]),
                "primary_metric": "pIC50 MAE",
                "primary_value": float(row["mae"]),
                "lower_95": float(row["mae_lower_95"]),
                "upper_95": float(row["mae_upper_95"]),
                "secondary_metric": "Spearman",
                "secondary_value": float(row["spearman"]),
                "evidence_limit": limitation,
            }
        )

    def add_binary(role: str, row: pd.Series, limitation: str) -> None:
        rows.append(
            {
                "domain": "hERG decisive class",
                "model_role": role,
                "evaluation": row["evaluation"],
                "data_regime": row["data_regime"],
                "feature_layer": row["feature_layer"],
                "model": row["model"],
                "n": int(row["n"]),
                "primary_metric": "Brier",
                "primary_value": float(row["brier"]),
                "lower_95": float(row["brier_lower_95"]),
                "upper_95": float(row["brier_upper_95"]),
                "secondary_metric": "balanced accuracy",
                "secondary_value": float(row["balanced_accuracy"]),
                "evidence_limit": limitation,
            }
        )

    add_continuous(
        "internal anchor",
        continuous_anchor,
        "retrospective scaffold CV within the original internal collection",
    )
    add_continuous(
        "same-series augmented discovery",
        continuous_same_series,
        "retrospective CV after outcomes from the same series were available",
    )
    for record in continuous_public_rows.itertuples(index=False):
        add_continuous(
            "public-data challenger selected by mean rank across two retrospective evaluations",
            pd.Series(record._asdict()),
            "exploratory selection; public assays are heterogeneous and not protocol matched",
        )
    add_binary(
        "internal interval-aware anchor",
        binary_anchor,
        "55 decisive internal structures; nonblocker count is limited",
    )
    add_binary(
        "same-series interval-aware augmented discovery",
        binary_same_series,
        "retrospective same-series CV; only four decisive Angelo nonblockers",
    )
    for record in binary_public_rows.itertuples(index=False):
        add_binary(
            "public-data challenger selected by mean rank across two retrospective evaluations",
            pd.Series(record._asdict()),
            "exploratory selection; public labels and internal intervals are not protocol matched",
        )
    for endpoint in pk["endpoint"].unique():
        row = pk[pk["endpoint"].eq(endpoint)].nsmallest(1, "log_mae").iloc[0]
        rows.append(
            {
                "domain": "rat PK",
                "model_role": endpoint,
                "evaluation": "internal_scaffold_cv",
                "data_regime": "internal_only",
                "feature_layer": row["feature_layer"],
                "model": row["model"],
                "n": int(row["n"]),
                "primary_metric": "log10 MAE",
                "primary_value": float(row["log_mae"]),
                "lower_95": float(row["log_mae_lower_95"]),
                "upper_95": float(row["log_mae_upper_95"]),
                "secondary_metric": "median fold error",
                "secondary_value": float(row["median_fold_error"]),
                "evidence_limit": ("summary-parameter data only; no concentration-time-profile calibration"),
            }
        )
    selections = {
        "continuous_anchor": continuous_anchor.to_dict(),
        "continuous_same_series": continuous_same_series.to_dict(),
        "continuous_public_selection": continuous_public_selection.to_dict(),
        "continuous_public_rows": continuous_public_rows.to_dict("records"),
        "binary_anchor": binary_anchor.to_dict(),
        "binary_same_series": binary_same_series.to_dict(),
        "binary_public_selection": binary_public_selection.to_dict(),
        "binary_public_rows": binary_public_rows.to_dict("records"),
    }
    return pd.DataFrame(rows), selections


def _registry(
    key_results: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    pk_records = key_results[key_results["domain"].eq("rat PK")].copy()
    for _index, record in key_results.iterrows():
        if record["domain"] == "rat PK":
            continue
        elif record["model_role"] == "internal anchor":
            identifier = "HERG_CONTINUOUS_INTERNAL_ANCHOR"
            status = "retained_internal_comparator"
            prediction_output = "continuous pIC50"
        elif record["model_role"] == "internal interval-aware anchor":
            identifier = "HERG_BINARY_INTERNAL_ANCHOR"
            status = "discovery_only_unstable_specificity"
            prediction_output = "blocker probability at 10 uM decision boundary"
        elif "same-series" in str(record["model_role"]):
            identifier = (
                "HERG_CONTINUOUS_SAME_SERIES_AUGMENTED"
                if record["domain"] == "hERG continuous"
                else "HERG_BINARY_SAME_SERIES_AUGMENTED"
            )
            status = "discovery_track_retrospective"
            prediction_output = (
                "continuous pIC50" if record["domain"] == "hERG continuous" else "blocker probability"
            )
        else:
            identifier = (
                "HERG_CONTINUOUS_PUBLIC_CHALLENGER"
                if record["domain"] == "hERG continuous"
                else "HERG_BINARY_PUBLIC_CHALLENGER"
            )
            status = "challenger_not_promoted"
            prediction_output = (
                "continuous pIC50" if record["domain"] == "hERG continuous" else "blocker probability"
            )
        rows.append(
            {
                "model_id": identifier,
                "target": record["domain"],
                "training_data": record["data_regime"],
                "representation": record["feature_layer"],
                "estimator": record["model"],
                "primary_evidence": record["evaluation"],
                "primary_metric": record["primary_metric"],
                "primary_value": record["primary_value"],
                "prediction_output": prediction_output,
                "status": status,
                "allowed_use": (
                    "same-series assay prioritization only"
                    if str(record["domain"]).startswith("hERG")
                    else "retrospective internal PK hypothesis generation"
                ),
                "required_next_gate": (
                    "untouched protocol-matched Menin series"
                    if str(record["domain"]).startswith("hERG")
                    else "additional Menin series plus rat concentration-time profiles"
                ),
            }
        )
    if not pk_records.empty:
        rows.append(
            {
                "model_id": "PK_INTERNAL_ENDPOINT_LADDER",
                "target": "rat IV/PO PK summary endpoints",
                "training_data": "internal_only",
                "representation": (
                    "endpoint-specific compact, Morgan, hybrid, and pKa-sensitivity comparators"
                ),
                "estimator": "endpoint-specific retained comparator",
                "primary_evidence": "internal_scaffold_cv",
                "primary_metric": "log10 MAE by endpoint",
                "primary_value": float(pk_records["primary_value"].mean()),
                "prediction_output": ("IV/PO dose-normalized AUC, Vdss, PO Cmax/dose, and Tmax hypotheses"),
                "status": "discovery_track_internal_only",
                "allowed_use": "retrospective internal PK hypothesis generation",
                "required_next_gate": ("additional Menin series plus rat concentration-time profiles"),
            }
        )
    rows.append(
        {
            "model_id": "HERG_RELEASED_CONSERVATIVE_MODEL_PAIR",
            "target": "hERG continuous and blocker probability",
            "training_data": "internal plus measured nonoverlapping same-series extension",
            "representation": "nine compact proxies and complete associative feature model",
            "estimator": "retained two-model envelope",
            "primary_evidence": "existing grouped same-series residual audit",
            "primary_metric": "model disagreement plus conservative interval envelope",
            "primary_value": np.nan,
            "prediction_output": "two pIC50 predictions, intervals, blocker probabilities, AD",
            "status": "operational_discovery_only_interface",
            "allowed_use": "new same-series Menin analog assay prioritization",
            "required_next_gate": "blind frozen-batch evaluation before outcomes are inspected",
        }
    )
    registry = pd.DataFrame(rows).drop_duplicates("model_id", keep="first")
    return registry.sort_values("model_id").reset_index(drop=True)


def _experiment_inventory(
    continuous: pd.DataFrame,
    binary: pd.DataFrame,
    pk: pd.DataFrame,
) -> pd.DataFrame:
    continuous_inventory = continuous[
        ["evaluation", "data_regime", "feature_layer", "model", "n", "mae", "spearman"]
    ].copy()
    continuous_inventory.insert(0, "analysis", "hERG_continuous")
    continuous_inventory["primary_metric"] = "mae"
    continuous_inventory["primary_value"] = continuous_inventory["mae"]
    binary_inventory = binary[
        [
            "evaluation",
            "data_regime",
            "feature_layer",
            "model",
            "n",
            "brier",
            "balanced_accuracy",
        ]
    ].copy()
    binary_inventory.insert(0, "analysis", "hERG_interval_decisive_class")
    binary_inventory["primary_metric"] = "brier"
    binary_inventory["primary_value"] = binary_inventory["brier"]
    pk_inventory = pk[["endpoint", "feature_layer", "model", "n", "log_mae", "spearman"]].copy()
    pk_inventory.insert(0, "analysis", "rat_PK")
    pk_inventory["evaluation"] = "internal_scaffold_cv"
    pk_inventory["data_regime"] = "internal_only"
    pk_inventory["primary_metric"] = "log_mae"
    pk_inventory["primary_value"] = pk_inventory["log_mae"]
    return pd.concat(
        [continuous_inventory, binary_inventory, pk_inventory],
        ignore_index=True,
        sort=False,
    )


def _report(
    continuous: pd.DataFrame,
    continuous_gain: pd.DataFrame,
    binary: pd.DataFrame,
    binary_gain: pd.DataFrame,
    pk: pd.DataFrame,
    pk_compatibility: pd.DataFrame,
    key_results: pd.DataFrame,
    selections: dict[str, Any],
    fit_failure_count: int,
    target_audit_count: int,
    permutation_summary: pd.DataFrame,
    repeated_summary: pd.DataFrame,
    similarity: pd.DataFrame,
    residual_associations: pd.DataFrame,
    disagreements: pd.DataFrame,
    cliffs: pd.DataFrame,
) -> str:
    ca = pd.Series(selections["continuous_anchor"])
    cs = pd.Series(selections["continuous_same_series"])
    ba = pd.Series(selections["binary_anchor"])
    bs = pd.Series(selections["binary_same_series"])
    cp_rows = pd.DataFrame(selections["continuous_public_rows"])
    bp_rows = pd.DataFrame(selections["binary_public_rows"])
    cp_internal = cp_rows[cp_rows["evaluation"].eq("internal_scaffold_cv")].iloc[0]
    cp_angelo = cp_rows[cp_rows["evaluation"].eq("angelo_fixed_nonoverlap")].iloc[0]
    bp_internal = bp_rows[bp_rows["evaluation"].eq("internal_scaffold_cv")].iloc[0]
    bp_angelo = bp_rows[bp_rows["evaluation"].eq("angelo_fixed_nonoverlap")].iloc[0]
    rejected_pk = int(pk_compatibility["integration_decision"].str.startswith("rejected").sum())
    continuous_permutation = permutation_summary[permutation_summary["analysis"].eq("continuous")].iloc[0]
    binary_permutation = permutation_summary[permutation_summary["analysis"].eq("binary")].iloc[0]
    continuous_repeated = repeated_summary[
        repeated_summary["analysis"].eq("continuous") & repeated_summary["metric"].eq("mae")
    ].iloc[0]
    binary_brier_repeated = repeated_summary[
        repeated_summary["analysis"].eq("binary") & repeated_summary["metric"].eq("brier")
    ].iloc[0]
    binary_balanced_repeated = repeated_summary[
        repeated_summary["analysis"].eq("binary") & repeated_summary["metric"].eq("balanced_accuracy")
    ].iloc[0]
    binary_specificity_repeated = repeated_summary[
        repeated_summary["analysis"].eq("binary") & repeated_summary["metric"].eq("specificity")
    ].iloc[0]
    significant_residual = residual_associations[
        residual_associations["p_value_bh_within_all_tests"].lt(0.10)
    ]
    comparable_cliffs = cliffs[cliffs["prediction_delta_comparable"].eq(True)]
    missed_cliffs = comparable_cliffs[comparable_cliffs["model_under_resolves_cliff"].eq(True)]
    material_continuous_disagreements = disagreements[
        disagreements["analysis"].eq("continuous_pIC50")
        & disagreements["absolute_model_disagreement"].ge(0.5)
    ]
    best_pk = {
        endpoint: pk[pk["endpoint"].eq(endpoint)].nsmallest(1, "log_mae").iloc[0]
        for endpoint in pk["endpoint"].unique()
    }
    lines = [
        "# Menin-inhibitor hERG and rat-PK mix-and-match results",
        "",
        "## Executive conclusion",
        "",
        "The prediction target is **hERG liability for Menin inhibitors**, not Menin "
        "potency. Internal evidence is included in every candidate training regime. The "
        "completed matrix separates continuous pIC50 regression from interval-aware "
        "classification so that `>30 µM` measurements contribute as nonblockers without "
        "being converted into invented exact IC50 values.",
        "",
        f"The internal continuous anchor is **{ca['model']} + {ca['feature_layer']}**: "
        f"held-scaffold pIC50 MAE {_interval(ca['mae'], ca['mae_lower_95'], ca['mae_upper_95'])}, "
        f"Spearman {ca['spearman']:.3f} (n={int(ca['n'])}). The interval-aware internal "
        f"classifier is **{ba['model']} + {ba['feature_layer']}**: Brier "
        f"{_interval(ba['brier'], ba['brier_lower_95'], ba['brier_upper_95'])}, "
        f"balanced accuracy {ba['balanced_accuracy']:.3f}, ROC-AUC {ba['roc_auc']:.3f} "
        f"(n={int(ba['n'])}; {int(ba['n_nonblockers'])} nonblockers).",
        "",
        "These are retrospective internal comparators, not proof of new-scaffold "
        "generalization. The operational new-compound scorer remains a conservative "
        "same-series two-model envelope with applicability and disagreement flags.",
        "",
        f"Both anchors outperform shuffled outcomes under the same grouped folds "
        f"(continuous MAE p={continuous_permutation['mae_permutation_p']:.4f}; "
        f"binary Brier p={binary_permutation['brier_permutation_p']:.4f}). However, "
        f"binary repeated-split median balanced accuracy is only "
        f"{binary_balanced_repeated['median']:.3f} and median specificity is "
        f"{binary_specificity_repeated['median']:.3f}; the binary model therefore "
        "remains discovery-only.",
        "",
        "## What was compared",
        "",
        "- Training evidence: internal only; internal plus same-series Angelo extension; "
        "naive public pooling; source/class-balanced public pooling; outcome-blind nearest "
        "public chemistry; MW>=650 public chemistry; same-series plus balanced public; and, "
        "for continuous pIC50, public prior plus internal residual correction.",
        "- Representations: nine compact interpretable physical-property proxies; ECFP4 "
        "compressed inside each training fold; and their hybrid. The proxies are controls "
        "for size, neutral-parent lipophilic drive, polar exposure capacity, H-bonding "
        "capacity, flexibility, ring topology, aromaticity, and saturation—not free "
        "energies, rates, membrane fluxes, or receptor kinetics.",
        "- Estimators: linear ridge/logistic models, nonlinear SVR/SVC, random forest, "
        "ExtraTrees, train-mean/prior controls, and Tanimoto 3-nearest-neighbor controls.",
        "- Evaluation: structure/scaffold-grouped internal CV, a fixed nonoverlapping "
        "same-series Angelo set, CV after same-series augmentation, and a held-out public "
        "source diagnostic. No random-compound CV is used as primary evidence.",
        "",
        "## Continuous hERG findings",
        "",
        f"- Internal anchor: {ca['model']} + {ca['feature_layer']}, MAE "
        f"{ca['mae']:.3f}, Spearman {ca['spearman']:.3f}; {ca['fraction_within_0p5_log']:.1%} "
        "of predictions are within 0.5 pIC50 log unit.",
        f"- Same-series augmentation: {cs['model']} + {cs['feature_layer']} in Angelo "
        f"grouped CV, MAE {cs['mae']:.3f}, Spearman {cs['spearman']:.3f}. This is useful "
        "development evidence but post-outcome and same-series.",
        f"- Cross-evaluation public challenger: {cp_internal['data_regime']} with "
        f"{cp_internal['model']} + {cp_internal['feature_layer']}. Internal CV MAE "
        f"{cp_internal['mae']:.3f} ({_gain_text(continuous_gain, cp_internal, continuous=True)}); "
        f"fixed Angelo MAE {cp_angelo['mae']:.3f} "
        f"({_gain_text(continuous_gain, cp_angelo, continuous=True)}).",
        "- Even the cross-evaluation public challenger does not improve the fixed internal "
        "hybrid-ridge comparator on either primary evaluation; broad public evidence is "
        "therefore not promoted into the released continuous model.",
        "- Public data are representation-dependent: they can rescue a weak fingerprint "
        "fit while degrading another estimator. Therefore the public pool is retained as "
        "a challenger and negative-control matrix, not silently merged into the release.",
        "",
        "## Interval-aware hERG findings",
        "",
        "Binary labels were made only when the entire measurement interval was decisive: "
        "blocker when pIC50 lower bound >=5 (IC50 <=10 µM), nonblocker when pIC50 upper "
        "bound <=4.522879 (IC50 >=30 µM), otherwise excluded as intermediate.",
        f"- Internal interval-aware anchor: {ba['model']} + {ba['feature_layer']}, Brier "
        f"{ba['brier']:.3f}, balanced accuracy {ba['balanced_accuracy']:.3f}, sensitivity "
        f"{ba['sensitivity']:.3f}, specificity {ba['specificity']:.3f}.",
        f"- Same-series augmentation: {bs['model']} + {bs['feature_layer']}, Brier "
        f"{bs['brier']:.3f}, balanced accuracy {bs['balanced_accuracy']:.3f}.",
        f"- Cross-evaluation public challenger: {bp_internal['data_regime']} with "
        f"{bp_internal['model']} + {bp_internal['feature_layer']}. Internal Brier "
        f"{bp_internal['brier']:.3f} ({_gain_text(binary_gain, bp_internal, continuous=False)}); "
        f"fixed Angelo Brier {bp_angelo['brier']:.3f} "
        f"({_gain_text(binary_gain, bp_angelo, continuous=False)}).",
        "- The public binary challenger improves its matched internal logistic comparator "
        "on internal Brier but does not beat the compact-proxy ExtraTrees internal anchor. "
        "This is representation rescue, not evidence that public pooling is globally better.",
        "- The fixed Angelo classification set has only four decisive nonblockers. Its "
        "specificity and balanced accuracy are therefore informative but high variance; "
        "continuous pIC50 and individual predictions must be shown alongside class metrics.",
        "",
        "## Robustness and applicability controls",
        "",
        f"- Continuous anchor: fixed grouped OOF MAE "
        f"{continuous_permutation['observed_mae']:.3f} versus shuffled median "
        f"{continuous_permutation['null_mae_median']:.3f}; repeated scaffold-split MAE "
        f"median {continuous_repeated['median']:.3f} "
        f"[{continuous_repeated['lower_2p5']:.3f}, "
        f"{continuous_repeated['upper_97p5']:.3f}].",
        f"- Binary anchor: fixed grouped OOF Brier "
        f"{binary_permutation['observed_brier']:.3f} versus shuffled median "
        f"{binary_permutation['null_brier_median']:.3f}; repeated scaffold-split Brier "
        f"median {binary_brier_repeated['median']:.3f} "
        f"[{binary_brier_repeated['lower_2p5']:.3f}, "
        f"{binary_brier_repeated['upper_97p5']:.3f}].",
        "- Leave-one-scaffold-out error is not monotonic with nearest-neighbor similarity. "
        "The high-similarity binary stratum contains very few nonblockers and still fails "
        "its nonblocker(s), showing that a Tanimoto cutoff alone is not a reliable hERG "
        "applicability guarantee.",
        "- These controls were run after model selection on the same collection. They show "
        "non-random internal signal and instability; they are not an independent validation.",
        "",
        "## Failure strata and analogue cliffs",
        "",
        f"- {len(significant_residual)}/{len(residual_associations)} residual-stratification "
        "tests have BH q<0.10. Absolute continuous error increases with aromatic-ring "
        "count and TPSA and decreases with fraction sp3. These variables are already "
        "model inputs, so this indicates heteroscedastic subdomains or missing interactions/"
        "states—not a reason to add duplicate descriptors.",
        f"- {len(cliffs)} same-series high-similarity analogue pairs differ by at least "
        f"1.0 pIC50. Of {len(comparable_cliffs)} pairs scored by a directly comparable "
        f"anchor fit, {len(missed_cliffs)} are under-resolved (<0.5 predicted pIC50 "
        "separation). These are the strongest mechanistic falsification candidates.",
        f"- Internal and public-prior continuous models disagree by >=0.5 pIC50 for only "
        f"{len(material_continuous_disagreements)} record(s). Agreement between them is "
        "not independent confirmation because they share internal residual calibration.",
        "- Candidate cliffs require protocol-matched replication and exact chemical-edit "
        "review before invoking ionization, environment-dependent conformation, membrane "
        "access, or receptor-state explanations.",
        "",
        "## Rat PK findings",
        "",
    ]
    for endpoint, row in best_pk.items():
        lines.append(
            f"- **{endpoint}:** {row['model']} + {row['feature_layer']}; log10-MAE "
            f"{row['log_mae']:.3f} [{row['log_mae_lower_95']:.3f}, "
            f"{row['log_mae_upper_95']:.3f}], median fold error "
            f"{row['median_fold_error']:.2f}, Spearman {row['spearman']:.3f}."
        )
    lines.extend(
        [
            "",
            f"External PK pooling was rejected for {rejected_pk}/{len(pk_compatibility)} "
            "endpoint definitions because the currently curated external rows do not "
            "provide enough protocol-compatible rat structures sharing endpoint, route, "
            "species, and usable units. The negative result is preserved rather than "
            "filling the matrix with biologically incompatible observations.",
            "",
            "## What is genuinely going well",
            "",
            "1. The project now distinguishes potency regression, censored screening "
            "classification, calibration, chemical-domain support, and model disagreement.",
            "2. Simple interpretable models remain competitive. That is scientifically "
            "useful: the limited data do not justify claiming that extra complexity is "
            "automatically superior.",
            "3. External data have been tested in multiple controlled roles instead of "
            "being treated as uniformly transferable. Both successful and harmful pooling "
            "results remain visible.",
            "4. The same-series extension provides a real additional chemical/assay test, "
            "while the public source holdout reveals assay/domain shift.",
            "5. A new Menin inhibitor can already be scored through the retained "
            "same-series interface with pIC50, blocker probability, uncertainty, nearest "
            "analogs, domain status, and model disagreement.",
            "6. Outcome-permutation controls show that the selected internal anchors contain "
            "real signal beyond chance, while repeated splits reveal exactly where that "
            "signal is not yet dependable.",
            "7. The failure analysis identifies concrete high-similarity analogue cliffs "
            "that current models under-resolve, providing focused mechanistic test cases.",
            "",
            "## What did not work or remains unsupported",
            "",
            f"- Model-fit failures: {fit_failure_count}. The {target_audit_count} retained "
            "PK target-audit rows are validation records, not failed model fits.",
            "- Broad public hERG evidence is heterogeneous and cannot substitute for an "
            "untouched protocol-matched Menin series.",
            "- The nine compact descriptors are not fundamental physical observables. "
            "Microstate free energies, membrane partition/PMF/diffusivity, environment-"
            "dependent conformational populations, and channel-state binding kinetics "
            "remain outside model inputs until HPC/experimental admission gates pass.",
            "- PK uses summary parameters, so absorption, gut loss, first pass, distribution, "
            "and clearance components are not separately identifiable.",
            "- Menin potency is unavailable for the Angelo compounds, so no efficacy-hERG "
            "multiobjective score is calculated.",
            "- Binary hERG specificity is not stable under repeated scaffold splits. Sparse "
            "nonblockers are now the main statistical bottleneck.",
            "",
            "## Presentation-ready claims",
            "",
            "- “We built a leakage-aware hERG platform specifically for Menin inhibitors, "
            "with internal evidence in every candidate model and explicit tests of how "
            "same-series and public data help or hurt.”",
            "- “We preserve `>30 µM` as censoring; we do not invent exact IC50 values.”",
            "- “The continuous internal anchor is roughly 0.36 pIC50 MAE under scaffold "
            "holdout and beats shuffled outcomes, while external strategies are "
            "model-dependent and remain challengers.”",
            "- “The binary model detects internal signal but is not yet decision-grade "
            "because nonblocker specificity is unstable.”",
            "- “The result is already callable for new same-series structures, but promotion "
            "requires a frozen prospective batch from a new Menin series.”",
            "",
            "## Highest-value next evidence",
            "",
            "1. Freeze predictions for the next protocol-matched Menin series before "
            "revealing hERG outcomes; this is the decisive generalization experiment.",
            "2. Collect full concentration-response values, free concentrations, and assay "
            "conditions, not only thresholds, so calibration and protocol effects can be "
            "separated.",
            "3. Add the new multi-series internal data with series labels, dates, and "
            "protocol identifiers; rerun leave-series-out rather than random CV.",
            "4. On HPC, test environment-dependent conformer populations and membrane/"
            "receptor-state quantities first on replicated, under-resolved matched-pair/"
            "analogue-cliff candidates selected to falsify specific mechanisms.",
            "5. For PK, obtain rat IV/PO concentration-time profiles and compatible external "
            "Menin-series data before attempting calibrated gray-box/PBPK decomposition.",
            "",
            "## Files and use",
            "",
            "- Full continuous, binary, PK, external-gain, feature-contract, dataset-audit, "
            "prediction-level, and failure tables are retained in this directory.",
            "- Run `pipeline/scripts/predict_herg_new_compounds.py` for a new blinded "
            "same-series batch, then evaluate it later with "
            "`pipeline/scripts/evaluate_blind_herg_predictions.py`.",
            "",
        ]
    )
    return "\n".join(lines)


def _talking_points(key_results: pd.DataFrame) -> str:
    continuous = key_results[
        (key_results["domain"].eq("hERG continuous")) & (key_results["model_role"].eq("internal anchor"))
    ].iloc[0]
    binary = key_results[
        (key_results["domain"].eq("hERG decisive class"))
        & (key_results["model_role"].eq("internal interval-aware anchor"))
    ].iloc[0]
    return "\n".join(
        [
            "# Short presentation script",
            "",
            "- Target: hERG liability for Menin inhibitors—not Menin potency.",
            "- Built internal-only, same-series-augmented, and several public-data "
            "mix-and-match models using compact physical-property proxies, fingerprints, "
            "and hybrids.",
            f"- Continuous internal anchor: {continuous['model']} + "
            f"{continuous['feature_layer']}, scaffold-holdout MAE "
            f"{continuous['primary_value']:.3f} pIC50.",
            f"- Interval-aware internal classifier: {binary['model']} + "
            f"{binary['feature_layer']}, Brier {binary['primary_value']:.3f}, "
            f"balanced accuracy {binary['secondary_value']:.3f}.",
            "- Both anchors beat shuffled outcomes (empirical p≈0.008), but binary "
            "specificity is unstable across scaffold splits, so it is not decision-grade.",
            "- Preserved >30 µM as censored nonblocker evidence and excluded intermediate "
            "intervals instead of fabricating exact values.",
            "- External data sometimes help and sometimes hurt depending on representation; "
            "public models remain challengers, while internal data remain mandatory.",
            "- New same-series compounds can already be scored with pIC50, blocker "
            "probability, uncertainty, similarity/domain, and model-disagreement flags.",
            "- Immediate ask: a frozen protocol-matched new Menin series with outcomes "
            "released only after predictions are recorded.",
        ]
    )


def run(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    continuous = _read_csv(output / "herg_mix_match_summary.csv")
    continuous_gain = _read_csv(output / "external_gain_vs_internal.csv")
    binary = _read_csv(output / "herg_binary_summary.csv")
    binary_gain = _read_csv(output / "herg_binary_external_gain.csv")
    pk = _read_csv(output / "pk_feature_model_summary.csv")
    pk_compatibility = _read_csv(output / "external_pk_compatibility.csv")
    permutation_summary = _read_csv(output / "permutation_test_summary.csv")
    repeated_summary = _read_csv(output / "repeated_group_split_summary.csv")
    similarity = _read_csv(output / "similarity_strata.csv")
    residual_associations = _read_csv(output / "herg_residual_proxy_associations.csv")
    disagreements = _read_csv(output / "herg_model_disagreements.csv")
    cliffs = _read_csv(output / "same_series_analogue_cliff_candidates.csv")
    dataset_audit = _read_csv(output / "dataset_audit.csv")
    label_audit = _read_csv(output / "herg_binary_label_audit.csv")
    feature_contract = _read_csv(output / "feature_contract.csv")
    if any(
        frame.empty
        for frame in (
            continuous,
            continuous_gain,
            binary,
            binary_gain,
            pk,
            pk_compatibility,
            permutation_summary,
            repeated_summary,
            similarity,
            residual_associations,
            disagreements,
            cliffs,
        )
    ):
        raise ValueError("Mix-and-match source outputs are incomplete")

    key_results, selections = _key_results(continuous, binary, pk)
    registry = _registry(key_results)
    inventory = _experiment_inventory(continuous, binary, pk)
    failure_frames = [
        _read_csv(output / "failure_ledger.csv"),
        _read_csv(output / "herg_binary_failure_ledger.csv"),
    ]
    failure_frames = [frame for frame in failure_frames if not frame.empty]
    all_failures = (
        pd.concat(failure_frames, ignore_index=True, sort=False) if failure_frames else pd.DataFrame()
    )
    fit_failure_count = int(
        all_failures.get("domain", pd.Series(dtype=str)).isin({"herg", "herg_binary", "pk"}).sum()
    )
    target_audit_count = int(all_failures.get("domain", pd.Series(dtype=str)).eq("pk_target_audit").sum())
    report = _report(
        continuous,
        continuous_gain,
        binary,
        binary_gain,
        pk,
        pk_compatibility,
        key_results,
        selections,
        fit_failure_count,
        target_audit_count,
        permutation_summary,
        repeated_summary,
        similarity,
        residual_associations,
        disagreements,
        cliffs,
    )
    talking_points = _talking_points(key_results)
    label_summary = (
        label_audit.groupby(
            ["dataset_role", "binary_label_basis"],
            dropna=False,
            as_index=False,
        )
        .agg(
            rows=("record_id", "size"),
            blockers=("target_class", lambda values: int(values.eq(1).sum())),
            nonblockers=("target_class", lambda values: int(values.eq(0).sum())),
        )
        .sort_values(["dataset_role", "binary_label_basis"])
    )
    disagreement_top = (
        disagreements.groupby(["analysis", "evaluation"], group_keys=False).head(15).reset_index(drop=True)
    )
    workbook_payload = {
        "generated_for": "Menin project review",
        "scientific_target": "hERG liability for Menin inhibitors; rat PK secondary",
        "status": "retrospective discovery package; not prospective validation",
        "key_results": _workbook_records(key_results),
        "continuous_summary": _workbook_records(
            continuous,
            [
                "evaluation",
                "data_regime",
                "feature_layer",
                "model",
                "n",
                "n_scaffolds",
                "mae",
                "mae_lower_95",
                "mae_upper_95",
                "rmse",
                "r2",
                "spearman",
                "fraction_within_0p5_log",
                "fraction_within_1p0_log",
                "median_max_internal_like_tanimoto",
                "training_structures_median",
                "training_public_structures_median",
                "evidence_status",
            ],
        ),
        "binary_summary": _workbook_records(
            binary,
            [
                "evaluation",
                "data_regime",
                "feature_layer",
                "model",
                "n",
                "n_scaffolds",
                "n_blockers",
                "n_nonblockers",
                "roc_auc",
                "pr_auc",
                "balanced_accuracy",
                "balanced_accuracy_lower_95",
                "balanced_accuracy_upper_95",
                "sensitivity",
                "specificity",
                "mcc",
                "brier",
                "brier_lower_95",
                "brier_upper_95",
                "log_loss",
                "ece_10bin",
                "median_max_internal_like_tanimoto",
                "training_structures_median",
                "training_public_structures_median",
                "evidence_status",
            ],
        ),
        "continuous_external_gain": _workbook_records(continuous_gain),
        "binary_external_gain": _workbook_records(binary_gain),
        "pk_summary": _workbook_records(pk),
        "permutation_summary": _workbook_records(permutation_summary),
        "repeated_summary": _workbook_records(repeated_summary),
        "similarity_strata": _workbook_records(similarity),
        "residual_associations": _workbook_records(residual_associations),
        "analogue_cliffs": _workbook_records(cliffs),
        "top_model_disagreements": _workbook_records(disagreement_top),
        "dataset_audit": _workbook_records(dataset_audit),
        "binary_label_summary": _workbook_records(label_summary),
        "feature_contract": _workbook_records(feature_contract),
        "model_registry": _workbook_records(registry),
        "external_pk_compatibility": _workbook_records(pk_compatibility),
        "target_audit": _workbook_records(all_failures),
        "presentation_points": [
            "Target is hERG liability for Menin inhibitors, not Menin potency.",
            "Continuous internal anchor: hybrid ridge, held-scaffold MAE 0.356 pIC50.",
            "Both selected anchors beat shuffled outcomes (empirical p≈0.008).",
            "Binary specificity is unstable; the classifier remains discovery-only.",
            "Public hERG data help selectively but do not beat the internal anchor consistently.",
            "Twenty-two high-similarity analogue cliffs expose mechanisms the models miss.",
            "A new same-series compound can be scored now with uncertainty and domain flags.",
            "Promotion requires a frozen protocol-matched new Menin series.",
        ],
    }

    atomic_write_csv(output / "key_results.csv", key_results)
    atomic_write_csv(output / "model_registry.csv", registry)
    atomic_write_csv(output / "experiment_inventory.csv", inventory)
    atomic_write_csv(output / "all_failure_ledger.csv", all_failures)
    atomic_write_text(output / "professor_briefing.md", report)
    atomic_write_text(output / "presentation_talking_points.md", talking_points)
    atomic_write_json(output / "professor_workbook_payload.json", workbook_payload)
    payload = {
        "status": "completed",
        "continuous_experiments": int(len(continuous)),
        "binary_experiments": int(len(binary)),
        "pk_experiments": int(len(pk)),
        "experiment_inventory_rows": int(len(inventory)),
        "registered_models": int(len(registry)),
        "failure_or_target_audit_rows": int(len(all_failures)),
        "model_fit_failures": fit_failure_count,
        "target_audit_rows": target_audit_count,
        "selections": selections,
        "claim_boundary": (
            "retrospective same-series discovery; public strategies are challengers; "
            "prospective protocol-matched Menin series required for promotion"
        ),
    }
    atomic_write_json(output / "professor_package_summary.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2))


if __name__ == "__main__":
    main()
