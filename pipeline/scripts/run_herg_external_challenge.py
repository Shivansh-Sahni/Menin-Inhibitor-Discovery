#!/usr/bin/env python3
"""Build a fail-closed external hERG challenge panel from existing evidence.

The internal model is fitted without public structures.  Public labels are then
used for a complete-set stress test and an explicitly outcome-informed
mechanistic review panel.  Panel performance is never reported as prospective
validation.
"""

from __future__ import annotations

import argparse
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
from menin_discovery.features import RDKIT_DESCRIPTOR_COLUMNS
from menin_discovery.research_common import (
    atomic_write_csv,
    atomic_write_json,
    atomic_write_parquet,
    atomic_write_text,
)
from menin_discovery.research_modeling import (
    final_fit_censored_herg_predictions,
    merge_feature_layers,
    structure_feature_frame,
)
from menin_discovery.research_public_herg import prepare_sun_source_holdout
from menin_discovery.research_workflows import (
    compound_model_frame,
    load_canonical_tables,
    prepare_herg_evidence,
)
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, Lipinski, rdFingerprintGenerator, rdMolDescriptors
from rdkit.Chem.Draw import MolsToGridImage, rdMolDraw2D
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    balanced_accuracy_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[2]
CANONICAL = ROOT / "research/data/pk_herg/canonical"
MODEL_ROOT = ROOT / "research/models/pk_herg/herg"
OUTPUT = ROOT / "research/reports/pk_herg/herg_external_challenge"
BLOCKER_PIC50 = 5.0
NONBLOCKER_PIC50 = 6.0 - math.log10(30.0)
SEED = 20260728


def _atomic_figure(figure: plt.Figure, path: Path, **kwargs: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.stem}.tmp{path.suffix}")
    figure.savefig(temporary, **kwargs)
    os.replace(temporary, path)


def _mechanistic_diagnostics(smiles: str) -> dict[str, float | int | str]:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Invalid challenge SMILES: {smiles}")
    rings = molecule.GetRingInfo().AtomRings()
    basic_pattern = Chem.MolFromSmarts("[N;H0,H1,H2;+0;!$(N[C,S,P]=O);!$(N[a])]")
    return {
        "achiral_smiles": Chem.MolToSmiles(molecule, isomericSmiles=False),
        "selection_logp": float(Descriptors.MolLogP(molecule)),
        "selection_tpsa_a2": float(rdMolDescriptors.CalcTPSA(molecule)),
        "selection_rotatable_bonds": int(Lipinski.NumRotatableBonds(molecule)),
        "selection_fraction_csp3": float(rdMolDescriptors.CalcFractionCSP3(molecule)),
        "selection_aromatic_rings": int(rdMolDescriptors.CalcNumAromaticRings(molecule)),
        "selection_macrocycles": int(sum(len(ring) >= 12 for ring in rings)),
        "selection_largest_ring_atoms": max((len(ring) for ring in rings), default=0),
        "selection_basic_nitrogen_matches": int(len(molecule.GetSubstructMatches(basic_pattern))),
    }


def _prepare_internal() -> tuple[pd.DataFrame, list[str]]:
    tables = load_canonical_tables(CANONICAL)
    aliases = tables.get("compound_aliases")
    compounds = compound_model_frame(tables["compounds"], aliases)
    features, layers = merge_feature_layers(compounds)
    _, potency, _ = prepare_herg_evidence(
        compounds,
        tables["measurements"],
        features,
    )
    baseline_columns = [
        column for column in layers["structure_2d"] if column in potency and potency[column].notna().any()
    ]
    return potency, baseline_columns


def _prepare_public_exact() -> pd.DataFrame:
    prepared = prepare_sun_source_holdout(CANONICAL / "public_herg")
    train = prepared["regression_train"]
    validation = prepared["regression_validation"]
    assert isinstance(train, pd.DataFrame)
    assert isinstance(validation, pd.DataFrame)
    exact = pd.concat(
        [
            train.assign(public_source_role="regression_train"),
            validation.assign(public_source_role="validation_source_held_out"),
        ],
        ignore_index=True,
    )
    if exact.duplicated("structure_id").any():
        raise ValueError("Curated public exact table contains duplicate structure IDs")
    diagnostics = pd.DataFrame([_mechanistic_diagnostics(smiles) for smiles in exact["raw_smiles"]])
    return pd.concat([exact.reset_index(drop=True), diagnostics], axis=1)


def _score_public(
    potency: pd.DataFrame,
    baseline_columns: list[str],
    public: pd.DataFrame,
) -> pd.DataFrame:
    scoring_compounds = pd.DataFrame(
        {
            "compound_id": public["structure_id"].astype(str),
            "standardized_smiles": public["raw_smiles"].astype(str),
            "mw": public["computed_mw_g_mol"].astype(float),
            "display_name": public["structure_id"].astype(str),
        }
    )
    scoring_features = structure_feature_frame(scoring_compounds)
    scoring = scoring_compounds.merge(
        scoring_features,
        on="compound_id",
        how="inner",
        validate="one_to_one",
    )
    oof = pd.read_parquet(MODEL_ROOT / "structure_2d_censored_predictions.parquet")
    predictions = final_fit_censored_herg_predictions(
        potency,
        scoring,
        oof,
        feature_columns=baseline_columns,
        interval_level=0.90,
        promotion_status="external_stress_test_only",
    )
    pic50 = predictions[predictions["endpoint"] == "herg_pic50"].copy()
    pic50 = pic50.rename(
        columns={
            "compound_id": "structure_id",
            "mean": "internal_model_pic50",
            "lower": "internal_model_pic50_lower",
            "upper": "internal_model_pic50_upper",
            "uncertainty": "internal_model_interval_half_width",
            "domain_status": "internal_model_domain_status",
        }
    )
    probability = predictions[predictions["endpoint"] == "herg_blocker_probability"][
        ["compound_id", "mean", "lower", "upper"]
    ].rename(
        columns={
            "compound_id": "structure_id",
            "mean": "internal_model_blocker_probability",
            "lower": "internal_model_blocker_probability_lower",
            "upper": "internal_model_blocker_probability_upper",
        }
    )
    keep = [
        "structure_id",
        "internal_model_pic50",
        "internal_model_pic50_lower",
        "internal_model_pic50_upper",
        "internal_model_interval_half_width",
        "internal_model_domain_status",
        "max_train_tanimoto",
        "domain_threshold",
        "oof_mae_pic50",
        "oof_sigma_pic50",
        "training_rows",
        "training_compounds",
        "fit_converged",
    ]
    scored = public.merge(pic50[keep], on="structure_id", validate="one_to_one")
    scored = scored.merge(probability, on="structure_id", validate="one_to_one")
    scored["observed_decisive_class"] = np.select(
        [
            scored["pic50_value"] >= BLOCKER_PIC50,
            scored["pic50_value"] <= NONBLOCKER_PIC50,
        ],
        [1.0, 0.0],
        default=np.nan,
    )
    scored["internal_model_error_pic50"] = scored["internal_model_pic50"] - scored["pic50_value"]
    scored["internal_model_absolute_error_pic50"] = scored["internal_model_error_pic50"].abs()
    scored["evaluation_role"] = "external_protocol_unmatched_stress_test"
    scored["model_admission"] = "prohibited_all_public_structures_out_of_domain"
    return scored


def _stress_metrics(
    scored: pd.DataFrame,
    quarantined_structure_ids: set[str],
) -> pd.DataFrame:
    large = scored[scored["computed_mw_g_mol"] >= 650]
    strata = {
        "all_public_exact": scored,
        "mw_below_600": scored[scored["computed_mw_g_mol"] < 600],
        "mw_600_to_649": scored[(scored["computed_mw_g_mol"] >= 600) & (scored["computed_mw_g_mol"] < 650)],
        "large_molecule_mw_650_plus_all": large,
        "large_molecule_mw_650_plus_ambiguity_quarantined": large[
            ~large["structure_id"].isin(quarantined_structure_ids)
        ],
    }
    rows: list[dict[str, Any]] = []
    for stratum, frame in strata.items():
        if frame.empty:
            continue
        observed = frame["pic50_value"].to_numpy(dtype=float)
        predicted = frame["internal_model_pic50"].to_numpy(dtype=float)
        decisive = frame["observed_decisive_class"].notna()
        metrics: dict[str, Any] = {
            "stratum": stratum,
            "n": len(frame),
            "n_scaffolds": int(frame["scaffold"].nunique()),
            "pic50_mae": float(mean_absolute_error(observed, predicted)),
            "pic50_rmse": float(math.sqrt(mean_squared_error(observed, predicted))),
            "spearman": float(spearmanr(observed, predicted).statistic),
            "fraction_within_0p5_log": float(np.mean(np.abs(observed - predicted) <= 0.5)),
            "fraction_within_1p0_log": float(np.mean(np.abs(observed - predicted) <= 1.0)),
            "interval_coverage": float(
                np.mean(
                    (observed >= frame["internal_model_pic50_lower"])
                    & (observed <= frame["internal_model_pic50_upper"])
                )
            ),
            "inside_domain_fraction": float(frame["internal_model_domain_status"].eq("inside").mean()),
            "median_max_internal_tanimoto": float(frame["max_train_tanimoto"].median()),
            "maximum_max_internal_tanimoto": float(frame["max_train_tanimoto"].max()),
            "decisive_n": int(decisive.sum()),
        }
        if decisive.any():
            labels = frame.loc[decisive, "observed_decisive_class"].astype(int)
            probability = frame.loc[decisive, "internal_model_blocker_probability"].to_numpy(dtype=float)
            predicted_class = (probability >= 0.5).astype(int)
            metrics.update(
                {
                    "roc_auc": float(roc_auc_score(labels, probability)),
                    "balanced_accuracy": float(balanced_accuracy_score(labels, predicted_class)),
                    "mcc": float(matthews_corrcoef(labels, predicted_class)),
                }
            )
        rows.append(metrics)
    return pd.DataFrame(rows)


def _public_large_bridge(
    scored: pd.DataFrame,
    quarantined_structure_ids: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit a public-domain comparator with every large test scaffold excluded.

    This is a retrospective, same-source MW/scaffold extrapolation test. It is
    intentionally separate from the internal model and cannot validate the
    internal program.
    """

    large = scored[scored["computed_mw_g_mol"] >= 650].copy()
    held_out_scaffolds = set(large["scaffold"].astype(str))
    train = scored[
        (scored["public_source_role"] == "regression_train")
        & (scored["computed_mw_g_mol"] < 650)
        & (~scored["scaffold"].astype(str).isin(held_out_scaffolds))
    ].copy()
    feature_columns = [column for column in RDKIT_DESCRIPTOR_COLUMNS if column != "invalid_structure"]
    descriptor_model = Pipeline(
        [
            ("impute", SimpleImputer()),
            (
                "model",
                ExtraTreesRegressor(
                    n_estimators=500,
                    min_samples_leaf=2,
                    max_features="sqrt",
                    n_jobs=-1,
                    random_state=SEED,
                ),
            ),
        ]
    )
    descriptor_model.fit(
        train[feature_columns].replace([np.inf, -np.inf], np.nan),
        train["pic50_value"].astype(float),
    )
    descriptor_prediction = descriptor_model.predict(
        large[feature_columns].replace([np.inf, -np.inf], np.nan)
    )

    fingerprint_generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

    def fingerprint_matrix(frame: pd.DataFrame) -> np.ndarray:
        return np.asarray(
            [
                fingerprint_generator.GetFingerprintAsNumPy(Chem.MolFromSmiles(smiles))
                for smiles in frame["raw_smiles"]
            ],
            dtype=np.uint8,
        )

    fingerprint_model = ExtraTreesRegressor(
        n_estimators=500,
        min_samples_leaf=2,
        max_features="sqrt",
        n_jobs=-1,
        random_state=SEED,
    )
    fingerprint_model.fit(
        fingerprint_matrix(train),
        train["pic50_value"].astype(float),
    )
    fingerprint_prediction = fingerprint_model.predict(fingerprint_matrix(large))
    ensemble_prediction = 0.5 * (descriptor_prediction + fingerprint_prediction)
    large["public_bridge_descriptor_pic50"] = descriptor_prediction
    large["public_bridge_morgan_pic50"] = fingerprint_prediction
    large["public_bridge_pic50"] = ensemble_prediction
    large["public_bridge_error_pic50"] = ensemble_prediction - large["pic50_value"]
    large["public_bridge_absolute_error_pic50"] = large["public_bridge_error_pic50"].abs()
    large["public_bridge_role"] = "same_source_retrospective_mw_and_scaffold_extrapolation_comparator"
    large["public_bridge_eligible_for_model_promotion"] = False
    large["source_identity_quarantined"] = large["structure_id"].isin(quarantined_structure_ids)

    rows: list[dict[str, Any]] = []
    evaluation_sets = {
        "all_large_records": large,
        "ambiguity_quarantined_large_records": large[~large["source_identity_quarantined"]],
    }
    training_mean = float(train["pic50_value"].mean())
    training_median = float(train["pic50_value"].median())
    large["public_training_mean_pic50"] = training_mean
    large["public_training_median_pic50"] = training_median
    for evaluation_stratum, frame in evaluation_sets.items():
        observed = frame["pic50_value"].to_numpy(dtype=float)
        predictions = {
            "training_mean_null": np.full(len(frame), training_mean),
            "training_median_null": np.full(len(frame), training_median),
            "extra_trees_rdkit_descriptors": frame["public_bridge_descriptor_pic50"].to_numpy(dtype=float),
            "extra_trees_morgan": frame["public_bridge_morgan_pic50"].to_numpy(dtype=float),
            "equal_weight_descriptor_morgan_ensemble": frame["public_bridge_pic50"].to_numpy(dtype=float),
        }
        for model_name, frame_prediction in predictions.items():
            decisive = frame["observed_decisive_class"].notna()
            predicted_class = (frame_prediction[decisive.to_numpy()] >= BLOCKER_PIC50).astype(int)
            labels = frame.loc[decisive, "observed_decisive_class"].astype(int).to_numpy()
            is_null = model_name.endswith("_null")
            rows.append(
                {
                    "model": model_name,
                    "evaluation_stratum": evaluation_stratum,
                    "feature_role": (
                        "constant_null" if is_null else "conventional_2d_comparator_not_mechanistic_features"
                    ),
                    "training_rule": (
                        "public regression-train; MW<650; exclude every scaffold "
                        "present in the MW>=650 challenge"
                    ),
                    "test_rule": (
                        "all curated public exact structures with MW>=650"
                        if evaluation_stratum == "all_large_records"
                        else "MW>=650 after quarantining unresolved same-connectivity records"
                    ),
                    "n_train": len(train),
                    "n_test": len(frame),
                    "n_test_scaffolds": int(frame["scaffold"].nunique()),
                    "train_maximum_mw": float(train["computed_mw_g_mol"].max()),
                    "test_minimum_mw": float(frame["computed_mw_g_mol"].min()),
                    "train_test_scaffold_overlap": int(
                        len(set(train["scaffold"].astype(str)) & held_out_scaffolds)
                    ),
                    "pic50_mae": float(mean_absolute_error(observed, frame_prediction)),
                    "pic50_rmse": float(math.sqrt(mean_squared_error(observed, frame_prediction))),
                    "spearman": (
                        np.nan if is_null else float(spearmanr(observed, frame_prediction).statistic)
                    ),
                    "fraction_within_0p5_log": float(np.mean(np.abs(observed - frame_prediction) <= 0.5)),
                    "fraction_within_1p0_log": float(np.mean(np.abs(observed - frame_prediction) <= 1.0)),
                    "decisive_n": int(decisive.sum()),
                    "balanced_accuracy": float(balanced_accuracy_score(labels, predicted_class)),
                    "mcc": float(matthews_corrcoef(labels, predicted_class)),
                    "claim_boundary": (
                        "same-source retrospective comparator; not prospective, "
                        "protocol-matched, or eligible to validate the internal model"
                    ),
                }
            )
    metrics = pd.DataFrame(rows)
    return large, metrics


def _large_scaffold_bootstrap(
    bridge_predictions: pd.DataFrame,
    *,
    iterations: int = 5000,
) -> pd.DataFrame:
    """Quantify scaffold-sampling uncertainty conditional on fixed predictions."""

    clean = bridge_predictions[~bridge_predictions["source_identity_quarantined"]].copy()
    model_columns = {
        "internal_series_censored_ridge": "internal_model_pic50",
        "training_median_null": "public_training_median_pic50",
        "extra_trees_rdkit_descriptors": "public_bridge_descriptor_pic50",
        "extra_trees_morgan": "public_bridge_morgan_pic50",
        "equal_weight_descriptor_morgan_ensemble": "public_bridge_pic50",
    }
    scaffold_groups = [group.index.to_numpy() for _, group in clean.groupby("scaffold", sort=True)]
    if len(scaffold_groups) < 2:
        raise ValueError("Scaffold bootstrap requires at least two scaffold groups")
    counts = clean["scaffold"].value_counts(normalize=True)
    effective_scaffolds = float(1.0 / np.square(counts).sum())
    rng = np.random.default_rng(SEED)
    bootstrap_values: dict[tuple[str, str], list[float]] = {
        (model, metric): []
        for model in model_columns
        for metric in ("pic50_mae", "spearman", "delta_mae_vs_training_median_null")
    }
    for _ in range(iterations):
        sampled_groups = rng.integers(0, len(scaffold_groups), size=len(scaffold_groups))
        sampled_index = np.concatenate([scaffold_groups[index] for index in sampled_groups])
        sample = clean.loc[sampled_index]
        observed = sample["pic50_value"].to_numpy(dtype=float)
        null_prediction = sample["public_training_median_pic50"].to_numpy(dtype=float)
        null_mae = mean_absolute_error(observed, null_prediction)
        for model, column in model_columns.items():
            prediction = sample[column].to_numpy(dtype=float)
            mae = float(mean_absolute_error(observed, prediction))
            bootstrap_values[(model, "pic50_mae")].append(mae)
            bootstrap_values[(model, "delta_mae_vs_training_median_null")].append(mae - null_mae)
            if np.unique(prediction).size > 1 and np.unique(observed).size > 1:
                correlation = float(spearmanr(observed, prediction).statistic)
                bootstrap_values[(model, "spearman")].append(correlation)

    rows: list[dict[str, Any]] = []
    observed = clean["pic50_value"].to_numpy(dtype=float)
    null_prediction = clean["public_training_median_pic50"].to_numpy(dtype=float)
    null_mae = mean_absolute_error(observed, null_prediction)
    for model, column in model_columns.items():
        prediction = clean[column].to_numpy(dtype=float)
        estimates = {
            "pic50_mae": float(mean_absolute_error(observed, prediction)),
            "spearman": (
                np.nan
                if np.unique(prediction).size == 1
                else float(spearmanr(observed, prediction).statistic)
            ),
            "delta_mae_vs_training_median_null": float(mean_absolute_error(observed, prediction) - null_mae),
        }
        for metric, estimate in estimates.items():
            values = np.asarray(bootstrap_values[(model, metric)], dtype=float)
            rows.append(
                {
                    "model": model,
                    "metric": metric,
                    "estimate": estimate,
                    "scaffold_bootstrap_95ci_lower": (
                        float(np.quantile(values, 0.025)) if values.size else np.nan
                    ),
                    "scaffold_bootstrap_95ci_upper": (
                        float(np.quantile(values, 0.975)) if values.size else np.nan
                    ),
                    "n_structures": len(clean),
                    "n_scaffolds": len(scaffold_groups),
                    "effective_scaffold_count": effective_scaffolds,
                    "bootstrap_iterations": iterations,
                    "uncertainty_scope": (
                        "cluster resampling of fixed predictions; excludes model "
                        "selection, retraining, assay, and source uncertainty"
                    ),
                }
            )
    return pd.DataFrame(rows)


def _connectivity_ambiguities(large: pd.DataFrame) -> pd.DataFrame:
    summaries: list[dict[str, Any]] = []
    for achiral_smiles, group in large.groupby("achiral_smiles", sort=True):
        if len(group) < 2:
            continue
        spread = float(group["pic50_value"].max() - group["pic50_value"].min())
        if spread < 1.0:
            continue
        summaries.append(
            {
                "ambiguity_id": f"ACHIRAL-{len(summaries) + 1:02d}",
                "achiral_smiles": achiral_smiles,
                "structure_ids": ";".join(sorted(group["structure_id"].astype(str))),
                "source_smiles_count": int(group["raw_smiles"].nunique()),
                "minimum_pic50": float(group["pic50_value"].min()),
                "maximum_pic50": float(group["pic50_value"].max()),
                "pic50_spread": spread,
                "interpretation": (
                    "same achiral connectivity but stereochemical/representation "
                    "and potency disagreement; source identity and assay provenance "
                    "must be resolved before calling an activity cliff"
                ),
            }
        )
    return pd.DataFrame(summaries)


def _activity_cliffs(large: pd.DataFrame) -> pd.DataFrame:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    molecules = [Chem.MolFromSmiles(smiles) for smiles in large["raw_smiles"]]
    fingerprints = [generator.GetFingerprint(molecule) for molecule in molecules]
    rows: list[dict[str, Any]] = []
    for left in range(len(large)):
        for right in range(left + 1, len(large)):
            if large.iloc[left]["achiral_smiles"] == large.iloc[right]["achiral_smiles"]:
                continue
            similarity = DataStructs.TanimotoSimilarity(fingerprints[left], fingerprints[right])
            delta = abs(float(large.iloc[left]["pic50_value"]) - float(large.iloc[right]["pic50_value"]))
            if similarity < 0.80 or delta < 1.0:
                continue
            rows.append(
                {
                    "pair_id": "",
                    "structure_id_a": large.iloc[left]["structure_id"],
                    "structure_id_b": large.iloc[right]["structure_id"],
                    "morgan_tanimoto": similarity,
                    "pic50_a": float(large.iloc[left]["pic50_value"]),
                    "pic50_b": float(large.iloc[right]["pic50_value"]),
                    "absolute_delta_pic50": delta,
                    "model_delta_pic50": abs(
                        float(large.iloc[left]["internal_model_pic50"])
                        - float(large.iloc[right]["internal_model_pic50"])
                    ),
                    "pair_role": "outcome_informed_mechanistic_falsification_pair",
                    "claim_boundary": (
                        "public protocol-unmatched activity-cliff candidate; "
                        "requires source identity and assay replication"
                    ),
                }
            )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result = result.sort_values(
        ["absolute_delta_pic50", "morgan_tanimoto"],
        ascending=False,
    ).reset_index(drop=True)
    result["pair_id"] = [f"CLIFF-{index + 1:02d}" for index in range(len(result))]
    return result


def _greedy_pair_selection(cliffs: pd.DataFrame, maximum_pairs: int = 3) -> list[str]:
    chosen: list[str] = []
    used: set[str] = set()
    for row in cliffs.itertuples(index=False):
        if row.structure_id_a in used or row.structure_id_b in used:
            continue
        chosen.extend([row.structure_id_a, row.structure_id_b])
        used.update([row.structure_id_a, row.structure_id_b])
        if len(chosen) >= 2 * maximum_pairs:
            break
    return chosen


def _select_panel(
    scored: pd.DataFrame,
    bridge_predictions: pd.DataFrame,
    cliffs: pd.DataFrame,
    quarantined_structure_ids: set[str],
) -> pd.DataFrame:
    large = scored[
        (scored["computed_mw_g_mol"] >= 650) & (~scored["structure_id"].isin(quarantined_structure_ids))
    ].copy()
    selected_ids = _greedy_pair_selection(cliffs, maximum_pairs=3)
    selection_reason: dict[str, str] = {
        structure_id: "member_of_high_similarity_large_molecule_activity_cliff"
        for structure_id in selected_ids
    }
    remaining = large[~large["structure_id"].isin(selected_ids)].copy()
    false_safe = remaining[
        (remaining["pic50_value"] >= 6.0) & (remaining["internal_model_pic50"] < remaining["pic50_value"])
    ].sort_values("internal_model_absolute_error_pic50", ascending=False)
    false_liability = remaining[
        (remaining["pic50_value"] <= NONBLOCKER_PIC50)
        & (remaining["internal_model_pic50"] > remaining["pic50_value"])
    ].sort_values("internal_model_absolute_error_pic50", ascending=False)
    polar_blocker = remaining[
        (remaining["pic50_value"] >= 6.0)
        & ((remaining["selection_macrocycles"] > 0) | (remaining["selection_tpsa_a2"] >= 100))
    ].sort_values(["pic50_value", "selection_tpsa_a2"], ascending=False)
    candidate_groups = [
        (false_safe, "large_out_of_domain_false_safe_stress_case"),
        (false_liability, "large_out_of_domain_false_liability_stress_case"),
        (polar_blocker, "large_polar_or_macrocyclic_blocker_mechanism_case"),
    ]
    for frame, reason in candidate_groups:
        for structure_id in frame["structure_id"]:
            if structure_id in selection_reason:
                continue
            selection_reason[str(structure_id)] = reason
            selected_ids.append(str(structure_id))
            break
    if len(selected_ids) < 10:
        for structure_id in remaining.sort_values("internal_model_absolute_error_pic50", ascending=False)[
            "structure_id"
        ]:
            if structure_id in selection_reason:
                continue
            selection_reason[str(structure_id)] = "large_out_of_domain_error_extreme"
            selected_ids.append(str(structure_id))
            if len(selected_ids) == 10:
                break
    panel = large.set_index("structure_id").loc[selected_ids].reset_index()
    panel = panel.merge(
        bridge_predictions[
            [
                "structure_id",
                "public_bridge_pic50",
                "public_bridge_descriptor_pic50",
                "public_bridge_morgan_pic50",
                "public_bridge_error_pic50",
                "public_bridge_absolute_error_pic50",
                "public_bridge_role",
                "public_bridge_eligible_for_model_promotion",
            ]
        ],
        on="structure_id",
        how="left",
        validate="one_to_one",
    )
    panel["selection_reason"] = panel["structure_id"].map(selection_reason)
    panel["challenge_id"] = [f"EXT-{index + 1:02d}" for index in range(len(panel))]
    panel["primary_mechanistic_question"] = panel.apply(
        _mechanistic_question,
        axis=1,
    )
    panel["minimum_falsifying_test"] = panel.apply(_falsifying_test, axis=1)
    mechanistic_plans = panel.apply(_mechanistic_plan, axis=1, result_type="expand")
    mechanistic_plans.columns = [
        "primary_missing_process",
        "candidate_physics_observable",
        "physics_admission_gate",
    ]
    panel = pd.concat([panel, mechanistic_plans], axis=1)
    panel["selection_is_outcome_informed"] = True
    panel["eligible_for_performance_claim"] = False
    return panel


def _mechanistic_question(row: pd.Series) -> str:
    reason = str(row["selection_reason"])
    if "activity_cliff" in reason:
        return (
            "Does the local edit change microscopic protonation, cation exposure, "
            "membrane/cavity access, or state-specific hERG stabilization?"
        )
    if row["selection_macrocycles"] > 0 or row["selection_largest_ring_atoms"] >= 12:
        return (
            "Does an environment-dependent folded or rare extended state control "
            "free concentration and hERG access?"
        )
    if row["selection_tpsa_a2"] >= 100:
        return (
            "Can folding or intramolecular compensation hide the nominal polar "
            "burden, or is the reported block an assay/free-concentration artifact?"
        )
    if "false_safe" in reason:
        return (
            "Which access or receptor-state mechanism produces strong block that "
            "the internal-series structure baseline misses?"
        )
    if "false_liability" in reason:
        return (
            "Which desolvation, conformation, or access barrier prevents the "
            "nominally hERG-like structure from blocking?"
        )
    return (
        "Is the error driven by chemical-state, conformation, membrane access, "
        "receptor state, or incompatible assay protocol?"
    )


def _falsifying_test(row: pd.Series) -> str:
    if row["selection_macrocycles"] > 0 or row["selection_largest_ring_atoms"] >= 12:
        return (
            "repeat free-concentration hERG at matched pH plus solution/membrane "
            "conformational measurement before state-resolved sampling"
        )
    return (
        "repeat free-concentration concentration-response hERG under one voltage, "
        "temperature, pH, and incubation protocol; add pKa for charged cases"
    )


def _mechanistic_plan(row: pd.Series) -> tuple[str, str, str]:
    reason = str(row["selection_reason"])
    if "activity_cliff" in reason:
        return (
            "microstate_gated_electrostatic_presentation_and_receptor_stabilization",
            (
                "microscopic-state free-energy differences; exposed-cation "
                "orientation distribution; state-specific receptor interaction "
                "and unbinding free-energy differences"
            ),
            (
                "source identity and matched-protocol hERG replicated; microscopic "
                "site assignment calibrated; state and receptor replicas converged"
            ),
        )
    if row["selection_macrocycles"] > 0 or row["selection_largest_ring_atoms"] >= 12:
        return (
            "environment_conditioned_folding_and_rare_state_access",
            (
                "folded-to-extended free-energy difference; transition MFPT/reactive "
                "flux; microstate-weighted membrane entry and core-crossing barriers"
            ),
            (
                "adaptive conformer/state convergence; replicated transition "
                "statistics; PMF/diffusivity and hysteresis gates pass"
            ),
        )
    if "false_safe" in reason:
        return (
            "rare_access_state_or_unmodeled_receptor_state",
            (
                "productive-state population times transition flux; state-specific "
                "receptor stabilization and residence-time proxy"
            ),
            (
                "free concentration and pKa resolved; rare-state probability and "
                "transition-rate intervals stable across independent replicas"
            ),
        )
    if "false_liability" in reason:
        return (
            "desolvation_or_membrane_access_barrier",
            (
                "hydration/desolvation free energy; membrane PMF and diffusivity; "
                "cavity-access probability before receptor binding"
            ),
            (
                "matched-protocol nonblock replicated; charge-state weighting and "
                "membrane-coordinate/hysteresis checks pass"
            ),
        )
    return (
        "chemical_state_conformation_access_or_assay_incompatibility",
        (
            "state-population uncertainty; folded/extended transition kinetics; "
            "membrane access; receptor-state interaction dispersion"
        ),
        (
            "source identity and assay provenance resolved, then each upstream "
            "state/sampling/convergence gate passed sequentially"
        ),
    )


def _draw_molecule_svg(smiles: str, width: int = 520, height: int = 300) -> str:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Cannot draw invalid SMILES: {smiles}")
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    options = drawer.drawOptions()
    options.clearBackground = False
    options.padding = 0.08
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, molecule)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def _write_structure_svgs(panel: pd.DataFrame) -> None:
    destination = OUTPUT / "structures"
    destination.mkdir(parents=True, exist_ok=True)
    for row in panel.itertuples(index=False):
        path = destination / f"{row.challenge_id}.svg"
        temporary = path.with_suffix(".tmp.svg")
        temporary.write_text(_draw_molecule_svg(row.raw_smiles), encoding="utf-8")
        os.replace(temporary, path)


def _write_structure_panel(panel: pd.DataFrame) -> None:
    molecules = [Chem.MolFromSmiles(smiles) for smiles in panel["raw_smiles"]]
    if any(molecule is None for molecule in molecules):
        raise ValueError("Challenge structure panel contains an invalid SMILES")
    legends = [
        (
            f"{row.challenge_id}  observed {row.pic50_value:.2f}\n"
            f"internal {row.internal_model_pic50:.2f} | bridge {row.public_bridge_pic50:.2f}"
        )
        for row in panel.itertuples(index=False)
    ]
    image = MolsToGridImage(
        molecules,
        molsPerRow=2,
        subImgSize=(560, 330),
        legends=legends,
        useSVG=False,
    )
    path = OUTPUT / "challenge_structure_panel.png"
    temporary = path.with_name(f"{path.stem}.tmp{path.suffix}")
    image.save(temporary)
    os.replace(temporary, path)


def _make_overview_figure(scored: pd.DataFrame, panel: pd.DataFrame) -> None:
    large = scored[scored["computed_mw_g_mol"] >= 650]
    figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.7))
    axes[0].scatter(
        scored["pic50_value"],
        scored["internal_model_pic50"],
        s=10,
        alpha=0.20,
        color="#4C78A8",
        linewidths=0,
        label="Public exact",
    )
    axes[0].scatter(
        large["pic50_value"],
        large["internal_model_pic50"],
        s=30,
        alpha=0.75,
        facecolors="none",
        edgecolors="#F58518",
        linewidths=0.9,
        label="MW ≥650",
    )
    axes[0].scatter(
        panel["pic50_value"],
        panel["internal_model_pic50"],
        s=48,
        color="#E45756",
        edgecolors="white",
        linewidths=0.6,
        label="Challenge panel",
        zorder=4,
    )
    bounds = [
        min(scored["pic50_value"].min(), scored["internal_model_pic50"].min()),
        max(scored["pic50_value"].max(), scored["internal_model_pic50"].max()),
    ]
    axes[0].plot(bounds, bounds, color="#666666", linewidth=1, linestyle="--")
    axes[0].axvline(BLOCKER_PIC50, color="#999999", linewidth=0.8)
    axes[0].axhline(BLOCKER_PIC50, color="#999999", linewidth=0.8)
    axes[0].set(
        xlabel="Observed public pIC50",
        ylabel="Internal-series model pIC50",
        title="External stress test—not model validation",
    )
    axes[0].legend(frameon=False, fontsize=8)

    order = panel.sort_values("pic50_value").reset_index(drop=True)
    positions = np.arange(len(order))
    axes[1].hlines(
        positions,
        order["pic50_value"],
        order["internal_model_pic50"],
        color="#BBBBBB",
        linewidth=1.2,
    )
    axes[1].scatter(
        order["pic50_value"],
        positions,
        color="#F58518",
        s=38,
        label="Observed",
        zorder=3,
    )
    axes[1].scatter(
        order["internal_model_pic50"],
        positions,
        color="#4C78A8",
        s=38,
        label="Internal model",
        zorder=3,
    )
    axes[1].scatter(
        order["public_bridge_pic50"],
        positions,
        facecolors="none",
        edgecolors="#54A24B",
        linewidths=1.2,
        s=44,
        label="Public bridge",
        zorder=3,
    )
    axes[1].axvline(BLOCKER_PIC50, color="#999999", linewidth=0.8, linestyle="--")
    axes[1].set_yticks(positions, order["challenge_id"])
    axes[1].set(
        xlabel="pIC50",
        title="Selected mechanistic challenge cases",
    )
    axes[1].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="both", color="#DDDDDD", linewidth=0.5, alpha=0.6)
    figure.tight_layout()
    _atomic_figure(figure, OUTPUT / "external_challenge_overview.png", dpi=220)
    _atomic_figure(figure, OUTPUT / "external_challenge_overview.pdf")
    plt.close(figure)


def _report(
    scored: pd.DataFrame,
    metrics: pd.DataFrame,
    bridge_metrics: pd.DataFrame,
    bootstrap: pd.DataFrame,
    panel: pd.DataFrame,
    cliffs: pd.DataFrame,
    ambiguities: pd.DataFrame,
) -> str:
    large_metric = metrics[metrics["stratum"] == "large_molecule_mw_650_plus_ambiguity_quarantined"].iloc[0]
    all_metric = metrics[metrics["stratum"] == "all_public_exact"].iloc[0]
    clean_bridge_metrics = bridge_metrics[
        bridge_metrics["evaluation_stratum"] == "ambiguity_quarantined_large_records"
    ]
    bridge_metric = clean_bridge_metrics[
        clean_bridge_metrics["model"] == "equal_weight_descriptor_morgan_ensemble"
    ].iloc[0]
    fingerprint_metric = clean_bridge_metrics[clean_bridge_metrics["model"] == "extra_trees_morgan"].iloc[0]
    median_null_metric = clean_bridge_metrics[clean_bridge_metrics["model"] == "training_median_null"].iloc[0]
    internal_mae_ci = bootstrap[
        (bootstrap["model"] == "internal_series_censored_ridge") & (bootstrap["metric"] == "pic50_mae")
    ].iloc[0]
    ensemble_rank_ci = bootstrap[
        (bootstrap["model"] == "equal_weight_descriptor_morgan_ensemble")
        & (bootstrap["metric"] == "spearman")
    ].iloc[0]
    morgan_delta_ci = bootstrap[
        (bootstrap["model"] == "extra_trees_morgan")
        & (bootstrap["metric"] == "delta_mae_vs_training_median_null")
    ].iloc[0]
    return f"""# External hERG challenge: immediate pre-HPC stress test

## Decision boundary

This analysis asks what the current internal-series model does on structurally
different public molecules with unusual hERG behavior. The internal model was
fit without public structures. However, the public assays are not
protocol-matched and every public structure is outside the internal
applicability domain. This is therefore a falsification/stress test and
candidate-selection exercise, not external validation or evidence that the
model generalizes.

The challenge-panel selection is explicitly outcome-informed. Panel accuracy is
not reported. Only complete-set metrics are summarized.

## Complete-set result

- Public exact-pIC50 structures: **{len(scored):,}**.
- Large public structures (MW ≥650 Da): **{int(large_metric["n"]) + 2}** total;
  **{int(large_metric["n"])}** remain after quarantining two records from one
  unresolved same-connectivity group, across
  **{int(large_metric["n_scaffolds"])}** deposited Bemis–Murcko scaffolds.
- Maximum public-to-internal Morgan similarity: **{all_metric["maximum_max_internal_tanimoto"]:.3f}**;
  inside-domain fraction: **{all_metric["inside_domain_fraction"]:.3f}**.
- Complete public-set pIC50 MAE: **{all_metric["pic50_mae"]:.3f}**; large-molecule
  pIC50 MAE: **{large_metric["pic50_mae"]:.3f}** (scaffold-bootstrap 95% interval
  **{internal_mae_ci["scaffold_bootstrap_95ci_lower"]:.3f}–{internal_mae_ci["scaffold_bootstrap_95ci_upper"]:.3f}**).
- Large-molecule rank correlation: **{large_metric["spearman"]:.3f}**; fraction
  within 1 log unit: **{large_metric["fraction_within_1p0_log"]:.3f}**.

These numbers quantify failure under domain shift. They must not be used to
select a release model because assay and chemical domains differ from the
internal program.

## Broad-public bridge comparator

As a bounded secondary analysis, descriptor and Morgan ExtraTrees comparators
were trained on
**{int(bridge_metric["n_train"]):,}** public regression-train structures below
650 Da after excluding every scaffold found in the 58-structure large-molecule
test. On the 56-record ambiguity-quarantined test, the exploratory unweighted
ensemble reached pIC50 MAE
**{bridge_metric["pic50_mae"]:.3f}**, rank correlation
**{bridge_metric["spearman"]:.3f}** (scaffold-bootstrap 95% interval
**{ensemble_rank_ci["scaffold_bootstrap_95ci_lower"]:.3f}–{ensemble_rank_ci["scaffold_bootstrap_95ci_upper"]:.3f}**), and
**{bridge_metric["fraction_within_1p0_log"]:.3f}** within 1 log unit.
The Morgan model alone had MAE **{fingerprint_metric["pic50_mae"]:.3f}** and
rank correlation **{fingerprint_metric["spearman"]:.3f}**. A constant
training-median null had MAE **{median_null_metric["pic50_mae"]:.3f}**.
The Morgan-minus-null MAE difference was **{morgan_delta_ci["estimate"]:.3f}**
with scaffold-bootstrap 95% interval
**{morgan_delta_ci["scaffold_bootstrap_95ci_lower"]:.3f}–{morgan_delta_ci["scaffold_bootstrap_95ci_upper"]:.3f}**.

The ensemble recovers moderate rank signal, but it does not beat the
training-median null on MAE. The 5,000-replicate bootstrap resamples 31
scaffolds (effective scaffold count
**{ensemble_rank_ci["effective_scaffold_count"]:.2f}**) conditional on fixed
predictions; it excludes model-selection, retraining, assay, and source
uncertainty. This rejects any claim that adding broad public 2D data has solved
large-molecule hERG. Because training and test come from the same public
compilation and the test was defined retrospectively, these remain conventional
bridge comparators—not independent validation, mechanistic models, or models
eligible for optimization.

## What is immediately useful

The outcome-informed panel contains **{len(panel)}** large, structurally distant
cases. It includes high-similarity activity-cliff candidates, false-safe and
false-liability stress cases, and polar/macrocyclic blockers that test whether
environment-dependent folding, rare states, membrane access, or protocol/free
concentration explain behavior missed by the internal baseline.

The large-molecule subset produced **{len(cliffs)}** candidate pairs with Morgan
similarity ≥0.80 and absolute ΔpIC50 ≥1.0 after excluding identical achiral
connectivity. These are hypotheses, not confirmed chemical cliffs.

The source also contains **{len(ambiguities)}** achiral-connectivity groups with
at least a 1-log pIC50 spread. They are quarantined as identity/stereochemistry
or assay-provenance ambiguities and must not be presented as medicinal-chemistry
effects.

## What to show Dr. Aguilar

1. The model can now be challenged on visibly different structures, but the
   applicability-domain flag correctly rejects every one of them.
2. The selected pairs give concrete candidates for asking why small edits cause
   large hERG changes.
3. Dr. Aguilar's multi-series, protocol-resolved data are needed to distinguish
   genuine new-series mechanisms from public-source artifacts.
4. The first follow-up should be a blind, protocol-matched challenge set.
   Mechanistic calculations should start only after identity, microscopic
   protonation, and assay provenance are resolved.

## Parameter integrity

No unvalidated physics descriptor entered any model. The fitted models are the
previously defined internal continuous censored-pIC50 2D baseline and explicit
public-domain conventional controls (coarse RDKit descriptors and Morgan
fingerprints). Molecular weight, lipophilicity, polar surface, ring size,
flexibility, and basic-nitrogen counts are used only to stratify and interpret
the challenge panel. Proposed mechanistic parameters—microstate population,
environment-conditioned folded/extended populations, cation exposure, membrane
barriers, and state-specific receptor interaction—remain excluded until their
experimental or HPC gates pass.
"""


def run() -> dict[str, Any]:
    potency, baseline_columns = _prepare_internal()
    public = _prepare_public_exact()
    scored = _score_public(
        potency,
        baseline_columns,
        public,
    )
    large = scored[scored["computed_mw_g_mol"] >= 650].copy()
    ambiguities = _connectivity_ambiguities(large)
    quarantined_structure_ids = (
        set(";".join(ambiguities["structure_ids"]).split(";")) if not ambiguities.empty else set()
    )
    clean_large = large[~large["structure_id"].isin(quarantined_structure_ids)]
    metrics = _stress_metrics(scored, quarantined_structure_ids)
    bridge_predictions, bridge_metrics = _public_large_bridge(
        scored,
        quarantined_structure_ids,
    )
    bootstrap = _large_scaffold_bootstrap(bridge_predictions)
    cliffs = _activity_cliffs(clean_large)
    panel = _select_panel(
        scored,
        bridge_predictions,
        cliffs,
        quarantined_structure_ids,
    )

    if not scored["fit_converged"].all():
        raise RuntimeError("Internal censored model failed to converge")
    if not scored["internal_model_domain_status"].eq("outside").all():
        raise RuntimeError("Expected every public challenge structure to remain out of domain")
    if panel["structure_id"].duplicated().any() or len(panel) != 10:
        raise RuntimeError("Challenge panel must contain ten unique structures")
    if not panel["selection_is_outcome_informed"].all():
        raise RuntimeError("Challenge panel selection boundary was lost")
    if int(bridge_metrics.iloc[0]["train_test_scaffold_overlap"]) != 0:
        raise RuntimeError("Public bridge train/test scaffold isolation failed")
    if float(bridge_metrics.iloc[0]["train_maximum_mw"]) >= 650:
        raise RuntimeError("Public bridge training set contains a large molecule")
    if not panel["eligible_for_performance_claim"].eq(False).all():  # noqa: E712
        raise RuntimeError("Outcome-informed panel was incorrectly made performance-eligible")
    if panel["structure_id"].isin(quarantined_structure_ids).any():
        raise RuntimeError("A quarantined source identity entered the panel")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    atomic_write_parquet(OUTPUT / "external_predictions.parquet", scored)
    atomic_write_csv(OUTPUT / "stress_test_metrics.csv", metrics)
    atomic_write_parquet(
        OUTPUT / "public_bridge_predictions.parquet",
        bridge_predictions,
    )
    atomic_write_csv(OUTPUT / "public_bridge_metrics.csv", bridge_metrics)
    atomic_write_csv(
        OUTPUT / "large_molecule_scaffold_bootstrap.csv",
        bootstrap,
    )
    atomic_write_csv(OUTPUT / "large_molecule_activity_cliff_candidates.csv", cliffs)
    atomic_write_csv(OUTPUT / "achiral_connectivity_ambiguities.csv", ambiguities)
    atomic_write_csv(OUTPUT / "challenge_panel.csv", panel)
    atomic_write_text(
        OUTPUT / "external_challenge_report.md",
        _report(
            scored,
            metrics,
            bridge_metrics,
            bootstrap,
            panel,
            cliffs,
            ambiguities,
        ),
    )
    _write_structure_svgs(panel)
    _write_structure_panel(panel)
    _make_overview_figure(scored, panel)
    validation = {
        "all_checks_passed": True,
        "checks": [
            {
                "check": "internal_external_structure_isolation",
                "passed": bool(
                    scored["internal_model_domain_status"].eq("outside").all()
                    and scored["max_train_tanimoto"].max() < scored["domain_threshold"].min()
                ),
                "evidence": (
                    f"maximum similarity={scored['max_train_tanimoto'].max():.3f}; "
                    f"minimum admission threshold={scored['domain_threshold'].min():.3f}"
                ),
            },
            {
                "check": "public_bridge_mw_and_scaffold_isolation",
                "passed": bool(
                    bridge_metrics.iloc[0]["train_maximum_mw"] < 650
                    and bridge_metrics.iloc[0]["test_minimum_mw"] >= 650
                    and bridge_metrics.iloc[0]["train_test_scaffold_overlap"] == 0
                ),
                "evidence": (
                    f"train max MW={bridge_metrics.iloc[0]['train_maximum_mw']:.3f}; "
                    f"test min MW={bridge_metrics.iloc[0]['test_minimum_mw']:.3f}; "
                    "scaffold overlap=0"
                ),
            },
            {
                "check": "outcome_informed_panel_claim_control",
                "passed": bool(
                    len(panel) == 10
                    and panel["structure_id"].nunique() == 10
                    and panel["selection_is_outcome_informed"].all()
                    and panel["eligible_for_performance_claim"].eq(False).all()  # noqa: E712
                ),
                "evidence": "10 unique cases; outcome-informed; performance claims prohibited",
            },
            {
                "check": "source_ambiguity_quarantine",
                "passed": bool(
                    ambiguities.empty
                    or (
                        ambiguities["interpretation"].str.contains("must be resolved").all()
                        and not panel["structure_id"].isin(quarantined_structure_ids).any()
                    )
                ),
                "evidence": (
                    f"{len(ambiguities)} achiral-connectivity ambiguity group(s); "
                    f"{len(quarantined_structure_ids)} records excluded from clean "
                    "metrics, cliff detection, and panel selection"
                ),
            },
            {
                "check": "required_presentation_outputs",
                "passed": all(
                    path.exists()
                    for path in [
                        OUTPUT / "external_challenge_overview.png",
                        OUTPUT / "external_challenge_overview.pdf",
                        OUTPUT / "challenge_structure_panel.png",
                        OUTPUT / "external_challenge_report.md",
                        OUTPUT / "large_molecule_scaffold_bootstrap.csv",
                    ]
                ),
                "evidence": ("overview PNG/PDF, structure panel, report, and scaffold bootstrap table exist"),
            },
        ],
    }
    validation["all_checks_passed"] = all(bool(check["passed"]) for check in validation["checks"])
    if not validation["all_checks_passed"]:
        raise RuntimeError("External challenge validation failed")
    atomic_write_json(OUTPUT / "validation_report.json", validation)
    clean_bridge = bridge_metrics[
        bridge_metrics["evaluation_stratum"] == "ambiguity_quarantined_large_records"
    ].set_index("model")
    bridge_summary_metric = clean_bridge.loc["equal_weight_descriptor_morgan_ensemble"]
    ensemble_rank_bootstrap = bootstrap[
        (bootstrap["model"] == "equal_weight_descriptor_morgan_ensemble")
        & (bootstrap["metric"] == "spearman")
    ].iloc[0]
    morgan_delta_bootstrap = bootstrap[
        (bootstrap["model"] == "extra_trees_morgan")
        & (bootstrap["metric"] == "delta_mae_vs_training_median_null")
    ].iloc[0]
    summary = {
        "public_exact_structures": len(scored),
        "large_molecule_structures": len(large),
        "challenge_panel_size": len(panel),
        "activity_cliff_candidate_count": len(cliffs),
        "achiral_connectivity_ambiguity_count": len(ambiguities),
        "large_molecule_ambiguity_quarantined_structures": len(clean_large),
        "public_bridge_training_structures": int(bridge_metrics.iloc[0]["n_train"]),
        "public_bridge_large_molecule_mae": float(bridge_summary_metric["pic50_mae"]),
        "public_bridge_large_molecule_spearman": float(bridge_summary_metric["spearman"]),
        "public_bridge_large_molecule_fraction_within_1_log": float(
            bridge_summary_metric["fraction_within_1p0_log"]
        ),
        "public_bridge_morgan_large_molecule_mae": float(clean_bridge.loc["extra_trees_morgan", "pic50_mae"]),
        "public_bridge_training_median_null_mae": float(
            clean_bridge.loc["training_median_null", "pic50_mae"]
        ),
        "public_bridge_ensemble_spearman_scaffold_bootstrap_95ci": [
            float(ensemble_rank_bootstrap["scaffold_bootstrap_95ci_lower"]),
            float(ensemble_rank_bootstrap["scaffold_bootstrap_95ci_upper"]),
        ],
        "public_bridge_morgan_delta_mae_vs_median_null_scaffold_bootstrap_95ci": [
            float(morgan_delta_bootstrap["scaffold_bootstrap_95ci_lower"]),
            float(morgan_delta_bootstrap["scaffold_bootstrap_95ci_upper"]),
        ],
        "all_public_outside_internal_domain": bool(
            scored["internal_model_domain_status"].eq("outside").all()
        ),
        "maximum_public_internal_tanimoto": float(scored["max_train_tanimoto"].max()),
        "model_admission": "prohibited_external_stress_test_only",
        "panel_selection": "outcome_informed_mechanistic_falsification",
        "truth_boundary": (
            "Public protocol-unmatched labels and universally out-of-domain structures "
            "can falsify transfer assumptions and prioritize experiments; they cannot "
            "validate the internal model or enter optimizer decisions."
        ),
    }
    atomic_write_json(OUTPUT / "challenge_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    print(json.dumps(run(), indent=2))


if __name__ == "__main__":
    main()
