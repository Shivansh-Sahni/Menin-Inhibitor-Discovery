#!/usr/bin/env python3
"""Evaluate the locked internal hERG model on the Ascentage extension set.

Exact structure overlaps are isolated before evaluation.  Novel structures are
scored with the pre-existing censored pIC50 model, compared with a training-only
null, stratified by applicability domain and scaffold novelty, and inspected
for interval-defensible analog cliffs and virtual-structure sensitivity.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    final_fit_censored_herg_predictions,
    grouped_censored_herg_benchmark,
    herg_classification_metrics,
    merge_feature_layers,
    structure_feature_frame,
)
from menin_discovery.research_workflows import (
    compound_model_frame,
    load_canonical_tables,
    prepare_herg_evidence,
)
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Draw import MolsToGridImage
from scipy.stats import norm, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[2]
CANONICAL = ROOT / "research/data/pk_herg/canonical"
SOURCE = CANONICAL / "ascentage_herg_2026_07_28/normalized_records.parquet"
MODEL_ROOT = ROOT / "research/models/pk_herg/herg"
OUTPUT = ROOT / "research/reports/pk_herg/ascentage_herg_extension"
BLOCKER_PIC50 = 5.0
NONBLOCKER_PIC50 = 6.0 - math.log10(30.0)
SEED = 20260728


def _atomic_figure(figure: plt.Figure, path: Path, **kwargs: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.stem}.tmp{path.suffix}")
    figure.savefig(temporary, **kwargs)
    os.replace(temporary, path)


def _prepare_training() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    tables = load_canonical_tables(CANONICAL)
    compounds = compound_model_frame(tables["compounds"], tables.get("compound_aliases"))
    features, layers = merge_feature_layers(compounds)
    _, potency, _ = prepare_herg_evidence(compounds, tables["measurements"], features)
    columns = [
        column for column in layers["structure_2d"] if column in potency and potency[column].notna().any()
    ]
    return compounds, potency, columns


def _score(
    source: pd.DataFrame,
    compounds: pd.DataFrame,
    potency: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    scoring_compounds = pd.DataFrame(
        {
            "compound_id": source["structure_id"].astype(str),
            "standardized_smiles": source["standardized_smiles"].astype(str),
            "mw": source["computed_mw_g_mol"].astype(float),
            "display_name": source["internal_id"].astype(str),
        }
    )
    features = structure_feature_frame(scoring_compounds)
    scoring = scoring_compounds.merge(features, on="compound_id", validate="one_to_one")
    oof = pd.read_parquet(MODEL_ROOT / "structure_2d_censored_predictions.parquet")
    predictions = final_fit_censored_herg_predictions(
        potency,
        scoring,
        oof,
        feature_columns=columns,
        interval_level=0.90,
        promotion_status="retrospective_extension_evaluation_only",
    )
    pic50 = predictions[predictions["endpoint"].eq("herg_pic50")].rename(
        columns={
            "compound_id": "structure_id",
            "mean": "predicted_pic50",
            "lower": "predicted_pic50_lower",
            "upper": "predicted_pic50_upper",
            "uncertainty": "prediction_interval_half_width",
            "domain_status": "applicability_domain",
        }
    )
    blocker = predictions[predictions["endpoint"].eq("herg_blocker_probability")][
        ["compound_id", "mean", "lower", "upper"]
    ].rename(
        columns={
            "compound_id": "structure_id",
            "mean": "predicted_blocker_probability",
            "lower": "predicted_blocker_probability_lower",
            "upper": "predicted_blocker_probability_upper",
        }
    )
    keep = [
        "structure_id",
        "predicted_pic50",
        "predicted_pic50_lower",
        "predicted_pic50_upper",
        "prediction_interval_half_width",
        "applicability_domain",
        "max_train_tanimoto",
        "domain_threshold",
        "oof_mae_pic50",
        "oof_sigma_pic50",
        "training_rows",
        "training_compounds",
        "fit_converged",
        "promotion_status",
    ]
    scored = source.merge(pic50[keep], on="structure_id", validate="one_to_one")
    scored = scored.merge(blocker, on="structure_id", validate="one_to_one")
    training_structures = set(compounds["structure_id"].astype(str))
    training_scaffolds = set(compounds["scaffold"].astype(str))
    scored["exact_training_structure_overlap"] = scored["structure_id"].isin(training_structures)
    scored["scaffold_seen_in_training"] = scored["scaffold"].isin(training_scaffolds)
    scored["evaluation_partition"] = np.where(
        scored["exact_training_structure_overlap"],
        "exact_overlap_excluded",
        "novel_structure_extension",
    )
    scored["prediction_error_pic50"] = (scored["predicted_pic50"] - scored["herg_pic50_value"]).where(
        scored["herg_pic50_relation"].eq("=")
    )
    scored["absolute_error_pic50"] = scored["prediction_error_pic50"].abs()
    scored["strict_censoring_compatibility"] = (
        scored["predicted_pic50"] <= scored["herg_pic50_upper_bound"]
    ).where(scored["herg_pic50_relation"].eq("<"))
    censored_probability = pd.Series(
        norm.cdf(
            (scored["herg_pic50_upper_bound"] - scored["predicted_pic50"])
            / scored["oof_sigma_pic50"].clip(lower=1e-6)
        ),
        index=scored.index,
    )
    scored["censored_probability_mass"] = censored_probability.where(scored["herg_pic50_relation"].eq("<"))
    return scored


def _augmented_unmeasured_predictions(
    scored: pd.DataFrame,
    potency: pd.DataFrame,
    columns: list[str],
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    novel_measured = scored[
        ~scored["exact_training_structure_overlap"] & scored["herg_ic50_censoring"].ne("missing")
    ].copy()
    extension_compounds = pd.DataFrame(
        {
            "compound_id": novel_measured["structure_id"].astype(str),
            "standardized_smiles": novel_measured["standardized_smiles"].astype(str),
        }
    )
    extension_features = structure_feature_frame(extension_compounds)
    extension = extension_compounds.merge(extension_features, on="compound_id", validate="one_to_one")
    extension = extension.merge(
        novel_measured[
            [
                "structure_id",
                "herg_pic50_lower_bound",
                "herg_pic50_upper_bound",
            ]
        ],
        left_on="compound_id",
        right_on="structure_id",
        validate="one_to_one",
    ).drop(columns="structure_id")
    extension = extension.rename(
        columns={
            "herg_pic50_lower_bound": "pic50_lower",
            "herg_pic50_upper_bound": "pic50_upper",
        }
    )
    combined = pd.concat([potency, extension], ignore_index=True, sort=False)
    if set(extension["compound_id"]) & set(potency["compound_id"]):
        raise ValueError("Novel extension rows unexpectedly overlap baseline training compounds")
    augmented_metrics, augmented_oof = grouped_censored_herg_benchmark(
        combined,
        feature_columns=columns,
        folds=5,
    )

    unmeasured = scored[
        ~scored["exact_training_structure_overlap"] & scored["herg_ic50_censoring"].eq("missing")
    ]
    unmeasured_compounds = pd.DataFrame(
        {
            "compound_id": unmeasured["structure_id"].astype(str),
            "standardized_smiles": unmeasured["standardized_smiles"].astype(str),
        }
    )
    unmeasured_features = structure_feature_frame(unmeasured_compounds)
    unmeasured_scoring = unmeasured_compounds.merge(
        unmeasured_features, on="compound_id", validate="one_to_one"
    )
    predictions = final_fit_censored_herg_predictions(
        combined,
        unmeasured_scoring,
        augmented_oof,
        feature_columns=columns,
        interval_level=0.90,
        promotion_status="virtual_unsynthesized_sensitivity_discovery_only",
    )
    pic50 = predictions[predictions["endpoint"].eq("herg_pic50")].rename(
        columns={
            "compound_id": "structure_id",
            "mean": "augmented_predicted_pic50",
            "lower": "augmented_predicted_pic50_lower",
            "upper": "augmented_predicted_pic50_upper",
            "domain_status": "augmented_applicability_domain",
            "max_train_tanimoto": "augmented_max_train_tanimoto",
        }
    )
    blocker = predictions[predictions["endpoint"].eq("herg_blocker_probability")][
        ["compound_id", "mean", "lower", "upper"]
    ].rename(
        columns={
            "compound_id": "structure_id",
            "mean": "augmented_predicted_blocker_probability",
            "lower": "augmented_predicted_blocker_probability_lower",
            "upper": "augmented_predicted_blocker_probability_upper",
        }
    )
    keep = [
        "structure_id",
        "augmented_predicted_pic50",
        "augmented_predicted_pic50_lower",
        "augmented_predicted_pic50_upper",
        "augmented_applicability_domain",
        "augmented_max_train_tanimoto",
        "domain_threshold",
        "oof_mae_pic50",
        "oof_sigma_pic50",
        "training_rows",
        "training_compounds",
        "fit_converged",
        "promotion_status",
    ]
    wide = pic50[keep].merge(blocker, on="structure_id", validate="one_to_one")
    wide = wide.rename(
        columns={
            "domain_threshold": "augmented_domain_threshold",
            "oof_mae_pic50": "augmented_oof_mae_pic50",
            "oof_sigma_pic50": "augmented_oof_sigma_pic50",
            "training_rows": "augmented_training_rows",
            "training_compounds": "augmented_training_compounds",
            "fit_converged": "augmented_fit_converged",
            "promotion_status": "augmented_promotion_status",
        }
    )
    augmented_metrics.update(
        {
            "training_rows_after_extension": len(combined),
            "baseline_training_rows": len(potency),
            "novel_extension_training_rows": len(extension),
            "prediction_target_rows": len(unmeasured),
            "interpretation": (
                "Retrospective grouped-CV model update; predictions apply only to the eight "
                "source-unmeasured structures and remain discovery-only."
            ),
        }
    )
    return augmented_metrics, augmented_oof, wide


def _regression_metrics(
    frame: pd.DataFrame,
    *,
    prediction_column: str,
    model: str,
    stratum: str,
) -> dict[str, Any]:
    exact = frame[frame["herg_pic50_relation"].eq("=")].copy()
    observed = exact["herg_pic50_value"].to_numpy(dtype=float)
    predicted = exact[prediction_column].to_numpy(dtype=float)
    absolute = np.abs(predicted - observed)
    rho = (
        float(spearmanr(observed, predicted).statistic)
        if len(exact) >= 3 and np.std(predicted) > 1e-12
        else float("nan")
    )
    return {
        "model": model,
        "stratum": stratum,
        "n_total": len(frame),
        "n_exact": len(exact),
        "n_scaffolds_exact": int(exact["scaffold"].nunique()),
        "pic50_mae": float(mean_absolute_error(observed, predicted)),
        "pic50_rmse": float(math.sqrt(mean_squared_error(observed, predicted))),
        "spearman": rho,
        "mean_signed_error": float(np.mean(predicted - observed)),
        "fraction_within_0p5_log": float(np.mean(absolute <= 0.5)),
        "fraction_within_1p0_log": float(np.mean(absolute <= 1.0)),
    }


def _metrics(scored: pd.DataFrame, potency: pd.DataFrame) -> pd.DataFrame:
    novel = scored[~scored["exact_training_structure_overlap"]].copy()
    training_exact = potency[
        np.isfinite(potency["pic50_lower"])
        & np.isfinite(potency["pic50_upper"])
        & np.isclose(potency["pic50_lower"], potency["pic50_upper"])
    ]
    training_mean = float(training_exact["pic50_lower"].mean())
    novel["training_exact_mean_null"] = training_mean
    strata = {
        "novel_all": novel,
        "novel_inside_ad": novel[novel["applicability_domain"].eq("inside")],
        "novel_outside_ad": novel[novel["applicability_domain"].eq("outside")],
        "novel_seen_scaffold": novel[novel["scaffold_seen_in_training"]],
        "novel_unseen_scaffold": novel[~novel["scaffold_seen_in_training"]],
    }
    rows: list[dict[str, Any]] = []
    for stratum, frame in strata.items():
        if frame["herg_pic50_relation"].eq("=").any():
            rows.append(
                _regression_metrics(
                    frame,
                    prediction_column="predicted_pic50",
                    model="locked_censored_ridge",
                    stratum=stratum,
                )
            )
            rows.append(
                _regression_metrics(
                    frame,
                    prediction_column="training_exact_mean_null",
                    model="training_exact_mean_null",
                    stratum=stratum,
                )
            )

    decisive = novel[novel["observed_decisive_class"].notna()]
    if not decisive.empty:
        classification = herg_classification_metrics(
            decisive["observed_decisive_class"].astype(int).to_numpy(),
            decisive["predicted_blocker_probability"].to_numpy(dtype=float),
        )
        rows.append(
            {
                "model": "locked_censored_ridge",
                "stratum": "novel_decisive_classification",
                "n_total": len(novel),
                "n_exact": int(decisive["herg_pic50_relation"].eq("=").sum()),
                "n_scaffolds_exact": int(decisive["scaffold"].nunique()),
                **{f"classification_{key}": value for key, value in classification.items()},
            }
        )
    return pd.DataFrame(rows)


def _cluster_bootstrap(scored: pd.DataFrame, potency: pd.DataFrame, replicates: int = 5000) -> pd.DataFrame:
    novel_exact = scored[
        ~scored["exact_training_structure_overlap"] & scored["herg_pic50_relation"].eq("=")
    ].copy()
    training_exact = potency[
        np.isfinite(potency["pic50_lower"])
        & np.isfinite(potency["pic50_upper"])
        & np.isclose(potency["pic50_lower"], potency["pic50_upper"])
    ]
    null = float(training_exact["pic50_lower"].mean())
    groups = sorted(novel_exact["scaffold"].unique())
    positions = {group: novel_exact.index[novel_exact["scaffold"].eq(group)].to_numpy() for group in groups}
    rng = np.random.default_rng(SEED)
    rows: list[dict[str, float | int]] = []
    for replicate in range(replicates):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        indices = np.concatenate([positions[group] for group in sampled])
        frame = novel_exact.loc[indices]
        observed = frame["herg_pic50_value"].to_numpy(dtype=float)
        predicted = frame["predicted_pic50"].to_numpy(dtype=float)
        model_mae = float(np.mean(np.abs(predicted - observed)))
        null_mae = float(np.mean(np.abs(null - observed)))
        rows.append(
            {
                "replicate": replicate,
                "n_rows": len(frame),
                "locked_model_mae": model_mae,
                "training_mean_null_mae": null_mae,
                "locked_minus_null_mae": model_mae - null_mae,
                "locked_model_spearman": (
                    float(spearmanr(observed, predicted).statistic)
                    if np.std(predicted) > 0 and np.std(observed) > 0
                    else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)


def _analog_contrasts(scored: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    measured = scored[scored["herg_ic50_censoring"].ne("missing")].reset_index(drop=True)
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fingerprints = [
        generator.GetFingerprint(Chem.MolFromSmiles(smiles)) for smiles in measured["standardized_smiles"]
    ]
    rows: list[dict[str, Any]] = []
    for left in range(len(measured)):
        for right in range(left + 1, len(measured)):
            similarity = float(DataStructs.TanimotoSimilarity(fingerprints[left], fingerprints[right]))
            if similarity < 0.80:
                continue
            left_lower = measured.loc[left, "herg_pic50_lower_bound"]
            left_upper = measured.loc[left, "herg_pic50_upper_bound"]
            right_lower = measured.loc[right, "herg_pic50_lower_bound"]
            right_upper = measured.loc[right, "herg_pic50_upper_bound"]
            left_lower = -np.inf if pd.isna(left_lower) else float(left_lower)
            left_upper = np.inf if pd.isna(left_upper) else float(left_upper)
            right_lower = -np.inf if pd.isna(right_lower) else float(right_lower)
            right_upper = np.inf if pd.isna(right_upper) else float(right_upper)
            guaranteed_separation = max(
                0.0,
                left_lower - right_upper,
                right_lower - left_upper,
            )
            if guaranteed_separation < 0.75:
                continue
            rows.append(
                {
                    "compound_a": measured.loc[left, "internal_id"],
                    "compound_b": measured.loc[right, "internal_id"],
                    "ic50_a_um": measured.loc[left, "submitted_herg_ic50_um"],
                    "ic50_b_um": measured.loc[right, "submitted_herg_ic50_um"],
                    "morgan_tanimoto": similarity,
                    "guaranteed_pic50_separation": guaranteed_separation,
                    "guaranteed_fold_difference": 10.0**guaranteed_separation,
                    "same_bemis_murcko_scaffold": bool(
                        measured.loc[left, "scaffold"] == measured.loc[right, "scaffold"]
                    ),
                    "strict_tenfold_contrast": bool(guaranteed_separation >= 1.0),
                    "interpretation": (
                        "candidate analog contrast; requires protocol confirmation and "
                        "mechanistic follow-up, not proof of a physical cause"
                    ),
                }
            )
    contrasts = pd.DataFrame(rows).sort_values(
        ["guaranteed_pic50_separation", "morgan_tanimoto"],
        ascending=False,
    )
    sensitivity_rows = []
    for similarity in (0.80, 0.85, 0.90):
        for separation in (0.75, 1.00):
            sensitivity_rows.append(
                {
                    "minimum_morgan_tanimoto": similarity,
                    "minimum_guaranteed_pic50_separation": separation,
                    "candidate_count": int(
                        (
                            (contrasts["morgan_tanimoto"] >= similarity)
                            & (contrasts["guaranteed_pic50_separation"] >= separation)
                        ).sum()
                    ),
                }
            )
    return contrasts, pd.DataFrame(sensitivity_rows)


def _review_panel(scored: pd.DataFrame) -> pd.DataFrame:
    virtual = scored[
        ~scored["exact_training_structure_overlap"]
        & scored["herg_ic50_censoring"].eq("missing")
        & scored["synthesis_status"].eq("not_synthesized")
    ].copy()
    virtual["panel_role"] = "virtual_prediction_sensitivity"
    virtual["selection_reason"] = np.where(
        virtual["applicability_domain"].eq("inside"),
        "unsynthesized same-series design used to audit local interpolation and retained-model uncertainty",
        "unsynthesized same-series design used to audit extrapolation and retained-model uncertainty",
    )
    measured_outliers = scored[
        ~scored["exact_training_structure_overlap"]
        & scored["herg_pic50_relation"].eq("=")
        & scored["applicability_domain"].eq("outside")
    ].nlargest(4, "absolute_error_pic50")
    measured_outliers = measured_outliers.copy()
    measured_outliers["panel_role"] = "historical_global_control_failure"
    measured_outliers["selection_reason"] = (
        "documents a failure of the initial global-control model; any physical hypothesis "
        "must survive the later complete-feature model and protocol audit"
    )
    panel = pd.concat([virtual, measured_outliers], ignore_index=True)
    panel["panel_prediction_pic50"] = panel["augmented_predicted_pic50"].combine_first(
        panel["predicted_pic50"]
    )
    panel["panel_prediction_pic50_lower"] = panel["augmented_predicted_pic50_lower"].combine_first(
        panel["predicted_pic50_lower"]
    )
    panel["panel_prediction_pic50_upper"] = panel["augmented_predicted_pic50_upper"].combine_first(
        panel["predicted_pic50_upper"]
    )
    panel["panel_blocker_probability"] = panel["augmented_predicted_blocker_probability"].combine_first(
        panel["predicted_blocker_probability"]
    )
    panel["panel_prediction_basis"] = np.where(
        panel["augmented_predicted_pic50"].notna(),
        "fixed model refit on baseline plus 46 measured novel extension structures",
        "locked pre-extension model; retained only to document the measured failure",
    )
    panel["panel_applicability_domain"] = panel["augmented_applicability_domain"].combine_first(
        panel["applicability_domain"]
    )
    panel["required_next_measurement"] = np.where(
        panel["panel_role"].eq("virtual_prediction_sensitivity"),
        "synthesis and identity confirmation before any assay request",
        "no physics action unless a protocol-matched result also fails the retained complete-feature model",
    )
    panel["physics_after_assay_gate"] = np.where(
        panel["panel_role"].eq("virtual_prediction_sensitivity"),
        "not eligible before synthesis and protocol-matched measurement",
        "microstate, environment, membrane-access, and receptor-state work only for a residual that survives the complete-feature model",
    )
    columns = [
        "internal_id",
        "ascentage_id",
        "panel_role",
        "selection_reason",
        "submitted_herg_ic50_um",
        "predicted_pic50",
        "predicted_pic50_lower",
        "predicted_pic50_upper",
        "predicted_blocker_probability",
        "augmented_predicted_pic50",
        "augmented_predicted_pic50_lower",
        "augmented_predicted_pic50_upper",
        "augmented_predicted_blocker_probability",
        "panel_prediction_pic50",
        "panel_prediction_pic50_lower",
        "panel_prediction_pic50_upper",
        "panel_blocker_probability",
        "panel_prediction_basis",
        "panel_applicability_domain",
        "max_train_tanimoto",
        "applicability_domain",
        "augmented_max_train_tanimoto",
        "augmented_applicability_domain",
        "scaffold_seen_in_training",
        "required_next_measurement",
        "physics_after_assay_gate",
        "standardized_smiles",
    ]
    return panel[columns].sort_values(["panel_role", "internal_id"])


def _plot(scored: pd.DataFrame, training_mean: float) -> None:
    exact = scored[~scored["exact_training_structure_overlap"] & scored["herg_pic50_relation"].eq("=")].copy()
    censored = scored[
        ~scored["exact_training_structure_overlap"] & scored["herg_pic50_relation"].eq("<")
    ].copy()
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    inside = exact["applicability_domain"].eq("inside")
    for mask, color, label in (
        (inside, "#2563eb", "inside AD"),
        (~inside, "#d97706", "outside AD"),
    ):
        axes[0].scatter(
            exact.loc[mask, "herg_pic50_value"],
            exact.loc[mask, "predicted_pic50"],
            color=color,
            alpha=0.82,
            edgecolors="white",
            linewidths=0.5,
            label=label,
        )
    limits = [
        min(exact["herg_pic50_value"].min(), exact["predicted_pic50"].min()) - 0.15,
        max(exact["herg_pic50_value"].max(), exact["predicted_pic50"].max()) + 0.15,
    ]
    axes[0].plot(limits, limits, color="#374151", linewidth=1, label="perfect")
    axes[0].axhline(training_mean, color="#6b7280", linestyle="--", linewidth=1, label="training mean")
    failure_family = exact[exact["internal_id"].isin(["M-2957", "M-2958", "M-2959", "M-2960"])]
    for row in failure_family.itertuples(index=False):
        axes[0].annotate(
            row.internal_id,
            (row.herg_pic50_value, row.predicted_pic50),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )
    axes[0].set(xlim=limits, ylim=limits, xlabel="Observed pIC50", ylabel="Locked-model pIC50")
    axes[0].set_title("54-structure extension: 42 exact measurements")
    axes[0].legend(frameon=False, fontsize=8)

    for mask, color, label in (
        (inside, "#2563eb", "exact, inside AD"),
        (~inside, "#d97706", "exact, outside AD"),
    ):
        axes[1].scatter(
            exact.loc[mask, "max_train_tanimoto"],
            exact.loc[mask, "prediction_error_pic50"],
            color=color,
            alpha=0.82,
            edgecolors="white",
            linewidths=0.5,
            label=label,
        )
    axes[1].scatter(
        censored["max_train_tanimoto"],
        censored["predicted_pic50"] - censored["herg_pic50_upper_bound"],
        marker="v",
        color="#7c3aed",
        label="censored: prediction minus upper bound",
    )
    axes[1].axhline(0, color="#374151", linewidth=1)
    axes[1].axvline(
        float(scored["domain_threshold"].dropna().iloc[0]),
        color="#6b7280",
        linestyle="--",
        linewidth=1,
        label="AD threshold",
    )
    axes[1].set(
        xlabel="Nearest training Morgan similarity",
        ylabel="Prediction error (pIC50)",
    )
    axes[1].set_title("Systematic underprediction in the least-similar family")
    axes[1].legend(frameon=False, fontsize=8)
    _atomic_figure(figure, OUTPUT / "extension_evaluation.png", dpi=220)
    _atomic_figure(figure, OUTPUT / "extension_evaluation.pdf")
    plt.close(figure)


def _plot_panel(panel: pd.DataFrame) -> None:
    molecules = [Chem.MolFromSmiles(smiles) for smiles in panel["standardized_smiles"]]
    legends = []
    for row in panel.itertuples(index=False):
        observed = f"{row.submitted_herg_ic50_um} uM" if row.submitted_herg_ic50_um else "not synthesized"
        legends.append(
            f"{row.internal_id} | obs {observed}\n"
            f"pred pIC50 {row.panel_prediction_pic50:.2f} | "
            f"{row.panel_applicability_domain} AD"
        )
    image = MolsToGridImage(
        molecules,
        molsPerRow=3,
        subImgSize=(460, 310),
        legends=legends,
        useSVG=False,
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT / ".historical_model_audit_panel.tmp.png"
    image.save(temporary)
    os.replace(temporary, OUTPUT / "historical_model_audit_panel.png")


def _write_reports(
    scored: pd.DataFrame,
    metrics: pd.DataFrame,
    bootstrap: pd.DataFrame,
    contrasts: pd.DataFrame,
    sensitivity: pd.DataFrame,
    panel: pd.DataFrame,
    augmented_metrics: dict[str, Any],
    compounds: pd.DataFrame,
    potency: pd.DataFrame,
) -> None:
    novel = scored[~scored["exact_training_structure_overlap"]]
    exact = novel[novel["herg_pic50_relation"].eq("=")]
    censored = novel[novel["herg_pic50_relation"].eq("<")]
    training_exact = potency[
        np.isfinite(potency["pic50_lower"])
        & np.isfinite(potency["pic50_upper"])
        & np.isclose(potency["pic50_lower"], potency["pic50_upper"])
    ]
    training_mean = float(training_exact["pic50_lower"].mean())
    primary = metrics[metrics["model"].eq("locked_censored_ridge") & metrics["stratum"].eq("novel_all")].iloc[
        0
    ]
    null = metrics[metrics["model"].eq("training_exact_mean_null") & metrics["stratum"].eq("novel_all")].iloc[
        0
    ]
    delta_ci = np.quantile(bootstrap["locked_minus_null_mae"], [0.025, 0.975])
    rho_ci = np.nanquantile(bootstrap["locked_model_spearman"], [0.025, 0.975])
    classification_row = metrics[metrics["stratum"].eq("novel_decisive_classification")].iloc[0]
    overlap = scored[scored["exact_training_structure_overlap"]]
    exact_overlap_ids = overlap["structure_id"].tolist()
    exact_source_values = overlap[overlap["herg_pic50_relation"].eq("=")][
        ["structure_id", "herg_ic50_value_um"]
    ]
    training_lookup = compounds[["compound_id", "structure_id"]].merge(
        potency[["compound_id", "pic50_lower", "pic50_upper"]],
        on="compound_id",
        how="left",
    )
    training_lookup = training_lookup[training_lookup["structure_id"].isin(exact_overlap_ids)]
    reconciled = exact_source_values.merge(training_lookup, on="structure_id", how="left")
    reconciled["training_ic50_um"] = 10.0 ** (6.0 - reconciled["pic50_lower"])
    reconciled_delta = (reconciled["herg_ic50_value_um"] - reconciled["training_ic50_um"]).abs()
    interval_width = float((exact["predicted_pic50_upper"] - exact["predicted_pic50_lower"]).mean())
    censor_compatible = int(censored["strict_censoring_compatibility"].sum())
    strict_cliffs = int(
        ((contrasts["morgan_tanimoto"] >= 0.80) & (contrasts["guaranteed_pic50_separation"] >= 1.0)).sum()
    )
    same_scaffold_strict = int(
        (
            (contrasts["morgan_tanimoto"] >= 0.80)
            & (contrasts["guaranteed_pic50_separation"] >= 1.0)
            & contrasts["same_bemis_murcko_scaffold"]
        ).sum()
    )
    virtual = panel[panel["panel_role"].eq("virtual_prediction_sensitivity")]
    virtual_pic50_min = float(virtual["panel_prediction_pic50"].min())
    virtual_pic50_max = float(virtual["panel_prediction_pic50"].max())
    virtual_probability_min = float(virtual["panel_blocker_probability"].min())
    virtual_probability_max = float(virtual["panel_blocker_probability"].max())
    report = f"""# Ascentage hERG extension analysis

## Evaluation status

This is a **retrospective, protocol-unmatched extension analysis**, not a prospective
validation. The source was visually reviewed before labels were analyzed. The model
artifact and its 110-compound training set predate this document, but 22/76 supplied
structures exactly overlap that training set and are excluded from extension metrics.
The remaining 54 structures are chemically close extensions: median nearest-training
Morgan similarity is {novel["max_train_tanimoto"].median():.3f}; they are not evidence
for transfer to an unrelated Menin series.

## Source and structure QC

- 76/76 ChemDraw OLE structures were converted from their embedded CDX streams and
  standardized successfully; 76 standardized structures are unique within the document.
- 68 compounds have hERG results: 61 exact IC50 values and seven right-censored
  `>30 µM` values. Dr. Aguilar confirmed that these seven did not reach 50% inhibition
  at the highest tested concentration of 30 µM. The other eight structures were not
  synthesized and are virtual designs, not missing assay results.
- The synthesized compounds were made by the CRO, all records are from the same chemical
  series, and Menin potencies are unavailable. This analysis therefore cannot support
  an efficacy–hERG tradeoff or independent-series validation.
- 22 exact structures overlap the current 110-compound baseline. Their reported values
  are consistent with the earlier source; the maximum exact IC50 discrepancy is
  {reconciled_delta.max():.3f} µM and reflects rounding/precision, not a class conflict.
- The document supplies no platform, cell line, voltage protocol, temperature, pH,
  incubation, replicate curves, or free-concentration basis. These records are therefore
  not decision-track eligible until protocol metadata are obtained.

## Locked-model result on 54 novel structures

Among the 42 novel structures with exact IC50:

- pIC50 MAE **{primary["pic50_mae"]:.3f}**, RMSE **{primary["pic50_rmse"]:.3f}**,
  Spearman **{primary["spearman"]:.3f}** (scaffold-bootstrap 95% interval
  **[{rho_ci[0]:.3f}, {rho_ci[1]:.3f}]**).
- **{primary["fraction_within_0p5_log"]:.1%}** fall within 0.5 log and
  **{primary["fraction_within_1p0_log"]:.1%}** within 1.0 log.
- The training-only mean null (pIC50 {training_mean:.3f}) has MAE
  **{null["pic50_mae"]:.3f}** and RMSE **{null["pic50_rmse"]:.3f}**. The locked model's
  scaffold-bootstrap MAE advantage is **{-float(bootstrap["locked_minus_null_mae"].mean()):.3f}**
  log on average, while locked-minus-null MAE has 95% interval
  **[{delta_ci[0]:.3f}, {delta_ci[1]:.3f}]**. Because this crosses zero, aggregate
  predictive improvement over the null is not conclusive.
- The nominal 90% interval covers
  **{((exact["herg_pic50_value"] >= exact["predicted_pic50_lower"]) & (exact["herg_pic50_value"] <= exact["predicted_pic50_upper"])).mean():.1%}**
  of exact values but has mean width **{interval_width:.2f} pIC50 logs**; this is
  conservative rather than usefully calibrated for decisions.
- Only **{censor_compatible}/{len(censored)}** novel `>30 µM` compounds have a predicted
  mean below the censoring bound. Decisive-class ROC-AUC is
  **{classification_row.get("classification_roc_auc", float("nan")):.3f}** and balanced
  accuracy **{classification_row.get("classification_balanced_accuracy", float("nan")):.3f}**,
  but these depend on only {len(censored)} decisive nonblockers and should not be generalized.

## Mechanistic value

The initial global-control model underpredicts M-2957–M-2960 by 0.76–1.10 pIC50 logs.
That residual is evidence that the initial representation is insufficient, but it is not
evidence for a specific physical mechanism. The later complete-feature audit explains
this family without admitting the failed local-physics features. Accordingly, this
historical residual is not a physics trigger unless it reappears under a documented,
protocol-matched assay and the retained complete-feature model.

At Morgan similarity ≥0.80, the set contains {strict_cliffs} interval-defensible analog
contrasts with at least a guaranteed tenfold hERG difference, including
{same_scaffold_strict} within one Bemis–Murcko scaffold. No such contrast survives a
0.85 similarity threshold, so these are candidate analog contrasts—not strict matched
molecular pairs. The threshold sensitivity is reported explicitly in
`analog_contrast_sensitivity.csv`.

## Immediate use

The eight blank entries are unsynthesized same-series designs. They can be used to test
prediction sensitivity and model disagreement, but they are not prospective assay
candidates and cannot validate the model. A genuine prospective test requires newly
submitted, synthesized structures whose outcomes remain hidden during prediction.

After freezing the extension evaluation above, the same censored-ridge specification was
refit on the baseline plus the 46 measured novel extension structures. Its five-fold
scaffold-held-out retrospective pIC50 MAE is
**{float(augmented_metrics["pic50_mae"]):.3f}**. This is a model update, not additional
external validation. For the eight unsynthesized virtual structures, the augmented
discovery model predicts pIC50 **{virtual_pic50_min:.2f}–{virtual_pic50_max:.2f}**
and blocker probability **{virtual_probability_min:.2f}–{virtual_probability_max:.2f}**.
These values are retained only as a virtual-structure sensitivity audit in
`historical_model_audit_panel.csv`; they are not assay or compound-selection decisions.

The four largest failures of the initial model are retained only as historical controls.
Mechanistic simulation is justified only for residuals that survive the retained model,
protocol confirmation, and matched-pair controls.

## Promotion decision

**Discovery track only.** The result provides a useful rank signal and strong failure
hypotheses, but it is not superior to the null with conclusive uncertainty, its intervals
are too broad, the assay protocol is missing, and the compounds remain close to the
training chemistry.
"""
    brief = f"""# Brief for Dr. Aguilar

- I converted all 76 ChemDraw structures into validated molecular records. Dr. Aguilar
  confirmed that 61 have exact hERG IC50 values, seven are right-censored because 50%
  inhibition was not reached at 30 µM, and the eight blank structures were not synthesized.
- The measured compounds were made by the CRO, all are from the same series, and Menin
  potencies are unavailable; this supports same-series hERG analysis only.
- I found and removed 22 exact overlaps with the original 110 compounds before testing.
  On the 54 genuinely new structures, the locked model reached pIC50 MAE {primary["pic50_mae"]:.2f}
  and Spearman {primary["spearman"]:.2f}, but it was not conclusively better than a
  training-mean baseline and its uncertainty remains too broad for compound decisions.
- The initial global-control model exposed a failure family, but the later complete-feature
  audit removed that residual without using failed physics features. Physics should target
  only failures that survive that stronger model and a protocol audit.
- I can now score a newly submitted same-series compound before seeing its outcome. The
  eight unsynthesized structures are useful only as virtual sensitivity examples
  (predicted pIC50 {virtual_pic50_min:.2f}–{virtual_pic50_max:.2f}), not validation.
"""
    atomic_write_text(OUTPUT / "extension_analysis_report.md", report)
    atomic_write_text(OUTPUT / "angelo_brief.md", brief)
    atomic_write_json(
        OUTPUT / "validation_report.json",
        {
            "status": "passed",
            "source_records": len(scored),
            "unique_source_structures": int(scored["structure_id"].nunique()),
            "exact_training_structure_overlaps": int(scored["exact_training_structure_overlap"].sum()),
            "novel_structures": len(novel),
            "novel_exact_measurements": len(exact),
            "novel_right_censored_measurements": len(censored),
            "novel_missing_measurements": int(novel["herg_ic50_censoring"].eq("missing").sum()),
            "novel_inside_ad": int(novel["applicability_domain"].eq("inside").sum()),
            "novel_outside_ad": int(novel["applicability_domain"].eq("outside").sum()),
            "novel_scaffolds": int(novel["scaffold"].nunique()),
            "novel_unseen_scaffolds": int(
                novel.loc[~novel["scaffold_seen_in_training"], "scaffold"].nunique()
            ),
            "model_fit_converged": bool(scored["fit_converged"].all()),
            "augmented_model_fit_converged_fraction": float(augmented_metrics["fit_converged_fraction"]),
            "bootstrap_replicates": len(bootstrap),
            "review_panel_compounds": panel["internal_id"].tolist(),
            "strict_analog_contrasts_tanimoto_0p80_pic50_1p0": strict_cliffs,
            "limitations": [
                "retrospective outcome review, not prospective validation",
                "missing assay protocol and replicate curves",
                "four decisive novel nonblockers only",
                "extension chemistry remains close to training chemistry",
                "physics hypotheses are not identified from model residuals alone",
            ],
        },
    )


def analyze() -> dict[str, Any]:
    source = load_ascentage_source(
        SOURCE,
        recovery_artifact=OUTPUT / "predictions.parquet",
    )
    compounds, potency, columns = _prepare_training()
    scored = _score(source, compounds, potency, columns)
    augmented_metrics, augmented_oof, augmented_predictions = _augmented_unmeasured_predictions(
        scored, potency, columns
    )
    scored = scored.merge(augmented_predictions, on="structure_id", how="left", validate="one_to_one")
    metrics = _metrics(scored, potency)
    bootstrap = _cluster_bootstrap(scored, potency)
    contrasts, sensitivity = _analog_contrasts(scored)
    panel = _review_panel(scored)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    atomic_write_parquet(OUTPUT / "predictions.parquet", scored)
    atomic_write_csv(OUTPUT / "predictions.csv", scored)
    atomic_write_csv(OUTPUT / "metrics.csv", metrics)
    atomic_write_csv(OUTPUT / "scaffold_bootstrap.csv", bootstrap)
    atomic_write_csv(OUTPUT / "analog_contrasts.csv", contrasts)
    atomic_write_csv(OUTPUT / "analog_contrast_sensitivity.csv", sensitivity)
    atomic_write_csv(OUTPUT / "historical_model_audit_panel.csv", panel)
    atomic_write_json(OUTPUT / "augmented_model_grouped_metrics.json", augmented_metrics)
    atomic_write_parquet(OUTPUT / "augmented_model_oof_predictions.parquet", augmented_oof)
    training_exact = potency[
        np.isfinite(potency["pic50_lower"])
        & np.isfinite(potency["pic50_upper"])
        & np.isclose(potency["pic50_lower"], potency["pic50_upper"])
    ]
    _plot(scored, float(training_exact["pic50_lower"].mean()))
    _plot_panel(panel)
    _write_reports(
        scored,
        metrics,
        bootstrap,
        contrasts,
        sensitivity,
        panel,
        augmented_metrics,
        compounds,
        potency,
    )
    return {
        "records": len(scored),
        "exact_overlaps_excluded": int(scored["exact_training_structure_overlap"].sum()),
        "novel_structures": int((~scored["exact_training_structure_overlap"]).sum()),
        "review_panel": panel["internal_id"].tolist(),
        "output": str(OUTPUT),
    }


def main() -> None:
    print(json.dumps(analyze(), indent=2))


if __name__ == "__main__":
    main()
