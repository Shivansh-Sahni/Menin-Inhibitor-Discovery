#!/usr/bin/env python3
"""Deep, train-only post-hoc analysis of the completed hERG V8 lattice.

This script never fits or selects a model and never opens repository validation or
test labels.  It converts the already frozen outer-scaffold-held-out V8 artifacts
into inferential, subgroup, applicability-domain, activity-cliff, and molecule-
level feature evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.stats import spearmanr

SCHEMA = "platform-local-herg-feature-lattice-analysis-v81/1.0"
PHYSICS_BLOCKS = {
    "polarity_charge_internal_contacts",
    "energy_flexibility",
    "shape",
    "autocorr3d",
    "whim",
    "old3d_stable",
    "new3d_stable_misc",
    "selected_interactions",
}


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), path, compression="zstd")


def _bh(p_values: np.ndarray) -> np.ndarray:
    n = len(p_values)
    order = np.argsort(p_values)
    ranked = p_values[order]
    adjusted = np.minimum.accumulate((ranked * n / np.arange(1, n + 1))[::-1])[::-1]
    out = np.empty(n, dtype=float)
    out[order] = np.minimum(adjusted, 1.0)
    return out


def _scaffold_inference(
    values: np.ndarray,
    scaffolds: np.ndarray,
    *,
    seed: int,
    replicates: int,
) -> dict[str, float]:
    frame = pd.DataFrame({"value": values, "scaffold": scaffolds})
    grouped = frame.groupby("scaffold", sort=True).value.agg(["sum", "count"])
    sums = grouped["sum"].to_numpy(dtype=float)
    counts = grouped["count"].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    boot = np.empty(replicates, dtype=float)
    null = np.empty(replicates, dtype=float)
    total_count = counts.sum()
    size = len(sums)
    for start in range(0, replicates, 256):
        stop = min(start + 256, replicates)
        width = stop - start
        indices = rng.integers(0, size, size=(width, size))
        boot[start:stop] = sums[indices].sum(axis=1) / counts[indices].sum(axis=1)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(width, size))
        null[start:stop] = (signs * sums).sum(axis=1) / total_count
    point = float(values.mean())
    return {
        "mean": point,
        "ci95_lower": float(np.quantile(boot, 0.025)),
        "ci95_upper": float(np.quantile(boot, 0.975)),
        "probability_positive": float(np.mean(boot > 0)),
        "sign_flip_p_two_sided": float((1 + np.sum(np.abs(null) >= abs(point))) / (replicates + 1)),
        "scaffolds": float(size),
    }


def _metrics(frame: pd.DataFrame) -> dict[str, float]:
    observed = frame.observed_pic50.to_numpy(float)
    predicted = frame.predicted_pic50.to_numpy(float)
    error = predicted - observed
    absolute = np.abs(error)
    return {
        "n": float(len(frame)),
        "scaffolds": float(frame.scaffold_group_id.nunique()),
        "mae": float(absolute.mean()),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "bias_prediction_minus_observed": float(error.mean()),
        "median_absolute_error": float(np.median(absolute)),
        "within_0p5": float(np.mean(absolute <= 0.5)),
        "within_1p0": float(np.mean(absolute <= 1.0)),
        "p95_absolute_error": float(np.quantile(absolute, 0.95)),
    }


def _add_bins(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["molecular_weight_regime"] = pd.cut(
        out["molecular_weight"],
        [-np.inf, 300, 500, 700, 1000, np.inf],
        labels=["mw_le_300", "mw_300_500", "mw_500_700", "mw_700_1000", "mw_gt_1000"],
    ).astype(str)
    out["heavy_molecule"] = out.molecular_weight > 500
    out["flexibility_regime"] = pd.cut(
        out.rotatable_bonds,
        [-np.inf, 2, 5, 9, np.inf],
        labels=["rotors_0_2", "rotors_3_5", "rotors_6_9", "rotors_ge_10"],
    ).astype(str)
    out["lipophilicity_regime"] = pd.cut(
        out.mol_logp,
        [-np.inf, 2, 4, 6, np.inf],
        labels=["logp_le_2", "logp_2_4", "logp_4_6", "logp_gt_6"],
    ).astype(str)
    out["polarity_regime"] = pd.cut(
        out.tpsa,
        [-np.inf, 40, 80, 120, np.inf],
        labels=["tpsa_le_40", "tpsa_40_80", "tpsa_80_120", "tpsa_gt_120"],
    ).astype(str)
    out["potency_regime"] = pd.cut(
        out.observed_pic50,
        [-np.inf, 4, 5, 6, np.inf],
        labels=["pic50_lt_4", "pic50_4_5", "pic50_5_6", "pic50_ge_6"],
    ).astype(str)
    if "maximum_train_tanimoto" in out:
        out["domain_regime"] = pd.cut(
            out.maximum_train_tanimoto,
            [-np.inf, 0.4, 0.6, 0.8, np.inf],
            labels=["tanimoto_lt_0p4", "tanimoto_0p4_0p6", "tanimoto_0p6_0p8", "tanimoto_ge_0p8"],
        ).astype(str)
    return out


def _performance_strata(frame: pd.DataFrame) -> pd.DataFrame:
    dimensions = [
        "molecular_weight_regime",
        "heavy_molecule",
        "flexibility_regime",
        "lipophilicity_regime",
        "polarity_regime",
        "potency_regime",
        "domain_regime",
        "measurement_modality",
        "automation_class",
        "assay_family",
        "source_family",
        "outer_fold",
    ]
    rows: list[dict[str, Any]] = [{"dimension": "overall", "level": "all", **_metrics(frame)}]
    for dimension in dimensions:
        if dimension not in frame:
            continue
        for level, group in frame.groupby(dimension, observed=True):
            if len(group) < 20:
                continue
            rows.append({"dimension": dimension, "level": str(level), **_metrics(group)})
    return pd.DataFrame(rows)


def analyze(v8_root: Path, output_root: Path, *, replicates: int, seed: int) -> dict[str, Any]:
    required = [
        "analysis.json",
        "validation.json",
        "nested_oof_predictions.parquet",
        "global_block_shapley.parquet",
        "pairwise_block_synergy.parquet",
        "individual_block_shapley.parquet",
        "nested_block_contributions.parquet",
        "nested_paired_block_effects.parquet",
        "prepared/training_matrix.parquet",
    ]
    for relative in required:
        if not (v8_root / relative).is_file():
            raise FileNotFoundError(v8_root / relative)
    validation = json.loads((v8_root / "validation.json").read_text())
    if validation.get("status") != "passed":
        raise ValueError("V8 validation is not passed")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing nonempty output root: {output_root}")
    staging = output_root.with_name(output_root.name + ".staging")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    oof = pq.read_table(v8_root / "nested_oof_predictions.parquet").to_pandas()
    if len(oof) != 18_801 or oof.structure_id.nunique() != 18_801:
        raise ValueError("nested OOF must cover exactly 18,801 structures")
    if set(oof.outer_fold.unique()) != set(range(5)):
        raise ValueError("nested OOF must contain outer folds 0..4")
    if oof.groupby("scaffold_group_id").outer_fold.nunique().max() != 1:
        raise ValueError("scaffold leakage in nested OOF")

    descriptor_columns = [
        "structure_id",
        "scaffold_group_id",
        "target_pic50",
        "rdkit2d__MolWt",
        "rdkit2d__HeavyAtomCount",
        "rdkit2d__NumRotatableBonds",
        "rdkit2d__MolLogP",
        "rdkit2d__TPSA",
    ]
    matrix = pq.read_table(
        v8_root / "prepared/training_matrix.parquet", columns=descriptor_columns
    ).to_pandas()
    matrix = matrix.rename(
        columns={
            "target_pic50": "matrix_target",
            "rdkit2d__MolWt": "molecular_weight",
            "rdkit2d__HeavyAtomCount": "heavy_atom_count",
            "rdkit2d__NumRotatableBonds": "rotatable_bonds",
            "rdkit2d__MolLogP": "mol_logp",
            "rdkit2d__TPSA": "tpsa",
        }
    )
    cohort = oof.merge(matrix, on=["structure_id", "scaffold_group_id"], validate="one_to_one")
    if not np.allclose(cohort.observed_pic50, cohort.matrix_target, rtol=0, atol=1e-7):
        raise ValueError("OOF target mismatch")

    # Reuse the label-blind, outer-fold train-neighbor similarity already computed in V1.
    ad_path = Path("research/local_runs/herg_discovery_campaign_v1/analysis/outer_oof_predictions.parquet")
    if ad_path.is_file():
        ad = pq.read_table(
            ad_path,
            columns=["model_id", "structure_id", "outer_fold", "maximum_train_tanimoto"],
        ).to_pandas()
        ad = ad[ad.model_id == "similarity_tanimoto_knn"].drop(columns="model_id")
        cohort = cohort.merge(ad, on=["structure_id", "outer_fold"], how="left", validate="one_to_one")
    cohort = _add_bins(cohort)
    cohort["absolute_error"] = np.abs(cohort.predicted_pic50 - cohort.observed_pic50)
    performance = _performance_strata(cohort)
    _write_parquet(performance, staging / "performance_strata.parquet")

    # V8 versus the completed V7 accuracy track, overall and by molecular regime.
    v7_path = Path(
        "research/local_runs/herg_honest_measurement_campaign_v7_1/nested_accuracy_oof_predictions.parquet"
    )
    comparison_rows: list[dict[str, Any]] = []
    if v7_path.is_file():
        v7 = (
            pq.read_table(v7_path)
            .to_pandas()[["structure_id", "predicted_pic50"]]
            .rename(columns={"predicted_pic50": "v7_prediction"})
        )
        compared = cohort.merge(v7, on="structure_id", validate="one_to_one")
        compared["delta_v8_minus_v7_abs_error"] = compared.absolute_error - np.abs(
            compared.v7_prediction - compared.observed_pic50
        )
        for dimension in ["overall", "molecular_weight_regime", "heavy_molecule", "domain_regime"]:
            groups = (
                [("all", compared)] if dimension == "overall" else compared.groupby(dimension, observed=True)
            )
            for level, group in groups:
                inference = _scaffold_inference(
                    group.delta_v8_minus_v7_abs_error.to_numpy(float),
                    group.scaffold_group_id.to_numpy(),
                    seed=seed + len(comparison_rows),
                    replicates=replicates,
                )
                comparison_rows.append(
                    {"dimension": dimension, "level": str(level), "n": len(group), **inference}
                )
    comparison = pd.DataFrame(comparison_rows)
    _write_parquet(comparison, staging / "v8_vs_v7_scaffold_inference.parquet")

    global_shapley = pq.read_table(v8_root / "global_block_shapley.parquet").to_pandas()
    global_summary = (
        global_shapley.groupby("block")
        .shapley_mae_improvement.agg(["mean", "std", "min", "max", "count"])
        .reset_index()
    )
    signs = (
        global_shapley.assign(positive=global_shapley.shapley_mae_improvement > 0)
        .groupby("block")
        .positive.sum()
    )
    global_summary["positive_outer_folds"] = global_summary.block.map(signs).astype(int)
    _write_parquet(global_summary, staging / "global_shapley_stability.parquet")

    synergy = pq.read_table(v8_root / "pairwise_block_synergy.parquet").to_pandas()
    synergy_summary = (
        synergy.groupby(["block_a", "block_b"])
        .banzhaf_synergy.agg(["mean", "std", "min", "max", "count"])
        .reset_index()
    )
    synergy_summary["same_sign_outer_folds"] = (
        synergy.groupby(["block_a", "block_b"])
        .banzhaf_synergy.apply(lambda x: int(max((x > 0).sum(), (x < 0).sum())))
        .to_numpy()
    )
    _write_parquet(synergy_summary, staging / "pairwise_synergy_stability.parquet")

    effects = pq.read_table(v8_root / "nested_paired_block_effects.parquet").to_pandas()
    collapsed = effects.groupby(
        ["structure_id", "scaffold_group_id", "outer_fold", "operation", "block"], as_index=False
    ).agg(baseline_abs_error=("baseline_abs_error", "first"), changed_abs_error=("changed_abs_error", "mean"))
    collapsed["delta_changed_minus_baseline"] = collapsed.changed_abs_error - collapsed.baseline_abs_error
    collapsed["benefit_of_block"] = np.where(
        collapsed.operation == "add_omitted",
        -collapsed.delta_changed_minus_baseline,
        collapsed.delta_changed_minus_baseline,
    )
    effect_rows: list[dict[str, Any]] = []
    for index, ((operation, block), group) in enumerate(collapsed.groupby(["operation", "block"])):
        inference = _scaffold_inference(
            group.benefit_of_block.to_numpy(float),
            group.scaffold_group_id.to_numpy(),
            seed=seed + 100 + index,
            replicates=replicates,
        )
        folds = group.groupby("outer_fold").benefit_of_block.mean()
        effect_rows.append(
            {
                "operation": operation,
                "block": block,
                "n": len(group),
                "positive_outer_folds": int((folds > 0).sum()),
                **inference,
            }
        )
    effect_inference = pd.DataFrame(effect_rows)
    effect_inference["bh_q"] = _bh(effect_inference.sign_flip_p_two_sided.to_numpy(float))
    effect_inference["discovery_grade_incremental"] = (
        (effect_inference.ci95_lower > 0)
        & (effect_inference.bh_q < 0.05)
        & (effect_inference.positive_outer_folds >= 4)
    )
    _write_parquet(effect_inference, staging / "block_effect_scaffold_inference.parquet")

    # Heavy-molecule analysis for every held-out feature intervention.
    heavy_effects = collapsed.merge(
        cohort[["structure_id", "molecular_weight", "molecular_weight_regime", "heavy_molecule"]],
        on="structure_id",
        validate="many_to_one",
    )
    heavy_rows: list[dict[str, Any]] = []
    for (operation, block, regime), group in heavy_effects.groupby(
        ["operation", "block", "molecular_weight_regime"], observed=True
    ):
        if len(group) < 50:
            continue
        heavy_rows.append(
            {
                "operation": operation,
                "block": block,
                "molecular_weight_regime": regime,
                "n": len(group),
                "scaffolds": group.scaffold_group_id.nunique(),
                "mean_block_benefit": group.benefit_of_block.mean(),
                "median_block_benefit": group.benefit_of_block.median(),
                "positive_outer_folds": int((group.groupby("outer_fold").benefit_of_block.mean() > 0).sum()),
            }
        )
    _write_parquet(pd.DataFrame(heavy_rows), staging / "heavy_molecule_block_effects.parquet")

    local = pq.read_table(v8_root / "individual_block_shapley.parquet").to_pandas()
    local = local.groupby(["structure_id", "scaffold_group_id", "block"], as_index=False).agg(
        local_shapley_mean=("local_shapley_abs_error_improvement", "mean"),
        local_shapley_sd=("local_shapley_abs_error_improvement", "std"),
        contexts=("outer_context", "nunique"),
    )
    local = local.merge(
        cohort[
            [
                "structure_id",
                "molecular_weight",
                "molecular_weight_regime",
                "rotatable_bonds",
                "mol_logp",
                "tpsa",
                "absolute_error",
            ]
        ],
        on="structure_id",
        how="left",
        validate="many_to_one",
    )
    local["physics_block"] = local.block.isin(PHYSICS_BLOCKS)
    _write_parquet(local, staging / "molecule_block_shapley_atlas.parquet")
    local_summary = (
        local.groupby(["block", "molecular_weight_regime"], observed=True)
        .agg(
            structures=("structure_id", "nunique"),
            mean_local_improvement=("local_shapley_mean", "mean"),
            median_local_improvement=("local_shapley_mean", "median"),
            positive_fraction=("local_shapley_mean", lambda x: float(np.mean(x > 0))),
        )
        .reset_index()
    )
    _write_parquet(local_summary, staging / "local_shapley_regime_summary.parquet")

    # Cross-fold conformal intervals: calibration residuals always come from other outer folds.
    uncertainty_rows: list[dict[str, Any]] = []
    for coverage in [0.5, 0.8, 0.9, 0.95]:
        for fold in range(5):
            calibration = cohort.loc[cohort.outer_fold != fold, "absolute_error"].to_numpy(float)
            radius = float(np.quantile(calibration, coverage, method="higher"))
            heldout = cohort[cohort.outer_fold == fold]
            uncertainty_rows.append(
                {
                    "nominal_coverage": coverage,
                    "outer_fold": fold,
                    "level": "all",
                    "n": len(heldout),
                    "radius_pic50": radius,
                    "empirical_coverage": float(np.mean(heldout.absolute_error <= radius)),
                }
            )
            for level, group in heldout.groupby("molecular_weight_regime", observed=True):
                uncertainty_rows.append(
                    {
                        "nominal_coverage": coverage,
                        "outer_fold": fold,
                        "level": str(level),
                        "n": len(group),
                        "radius_pic50": radius,
                        "empirical_coverage": float(np.mean(group.absolute_error <= radius)),
                    }
                )
    _write_parquet(pd.DataFrame(uncertainty_rows), staging / "cross_fold_conformal_coverage.parquet")

    # Training-only matched molecular pair performance and activity-cliff behavior.
    mmp_path = Path(
        "research/data/platform/processed/herg_hierarchy/v1_5_mmp_analysis/training_mmp_effects.parquet"
    )
    mmp_summary: dict[str, Any] = {"available": False}
    if mmp_path.is_file():
        mmp = pq.read_table(mmp_path).to_pandas()
        pred = cohort[
            ["structure_id", "predicted_pic50", "molecular_weight", "outer_fold", "scaffold_group_id"]
        ]
        mmp = mmp.merge(
            pred.add_suffix("_a"), left_on="structure_id_a", right_on="structure_id_a", how="inner"
        )
        mmp = mmp.merge(
            pred.add_suffix("_b"), left_on="structure_id_b", right_on="structure_id_b", how="inner"
        )
        all_joined_pairs = len(mmp)
        # A delta between predictions from different outer-fold models is not a
        # controlled matched-pair comparison. Retain same-model pairs only.
        mmp = mmp[mmp.outer_fold_a == mmp.outer_fold_b].copy()
        mmp["predicted_delta_b_minus_a"] = mmp.predicted_pic50_b - mmp.predicted_pic50_a
        mmp["delta_error"] = mmp.predicted_delta_b_minus_a - mmp.delta_pic50_b_minus_a
        mmp["either_heavy"] = (mmp.molecular_weight_a > 500) | (mmp.molecular_weight_b > 500)
        mmp["predicted_direction_correct"] = np.sign(mmp.predicted_delta_b_minus_a) == np.sign(
            mmp.delta_pic50_b_minus_a
        )
        _write_parquet(mmp, staging / "training_mmp_v8_predictions.parquet")
        groups = {
            "all": mmp,
            "activity_cliffs": mmp[mmp.activity_cliff_ge_1_pic50],
            "either_heavy": mmp[mmp.either_heavy],
        }
        mmp_summary = {
            "available": True,
            "all_joined_pairs": all_joined_pairs,
            "same_outer_model_pairs": len(mmp),
            "cross_outer_model_pairs_excluded": all_joined_pairs - len(mmp),
            "same_scaffold_pairs": int((mmp.scaffold_group_id_a == mmp.scaffold_group_id_b).sum()),
        }
        for name, group in groups.items():
            rho = spearmanr(group.delta_pic50_b_minus_a, group.predicted_delta_b_minus_a).statistic
            mmp_summary[name] = {
                "pairs": len(group),
                "delta_mae": float(np.mean(np.abs(group.delta_error))),
                "delta_spearman": float(rho),
                "direction_accuracy": float(group.predicted_direction_correct.mean()),
            }

    best_effects = effect_inference.sort_values("mean", ascending=False).head(10)
    heavy_perf = performance[
        (performance.dimension == "molecular_weight_regime") | (performance.dimension == "overall")
    ]
    summary: dict[str, Any] = {
        "schema_version": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "scientific_scope": {
            "partition": "train",
            "nested_outer_scaffold_heldout": True,
            "repository_validation_labels_opened": False,
            "repository_test_labels_opened": False,
            "explicit_wild_type_claim_allowed": False,
            "causal_claims_allowed": False,
        },
        "v8_metrics": _metrics(cohort),
        "v8_vs_v7_overall": comparison_rows[0] if comparison_rows else None,
        "mmp": mmp_summary,
        "counts": {
            "structures": len(cohort),
            "scaffolds": cohort.scaffold_group_id.nunique(),
            "heavy_structures_gt_500_mw": int(cohort.heavy_molecule.sum()),
            "effect_hypotheses": len(effect_inference),
            "discovery_grade_incremental_effects": int(effect_inference.discovery_grade_incremental.sum()),
            "local_shapley_structures": local.structure_id.nunique(),
        },
        "top_incremental_block_effects": best_effects.to_dict(orient="records"),
        "heavy_performance": heavy_perf.to_dict(orient="records"),
        "interpretation_boundary": [
            "Global and local Shapley values describe predictive allocation, not biological causality.",
            "Paired block interventions are outer-heldout model effects but remain associative.",
            "Heavy-molecule and activity-cliff analyses are internal, training-only hypotheses.",
            "V8 targets wild-type-or-unspecified hERG, not explicitly adjudicated human WT only.",
        ],
    }
    _write_json(staging / "analysis.json", summary)

    report_lines = [
        "# hERG V8.1 Deep Post-hoc Analysis",
        "",
        "This release analyzes only frozen train-partition, outer-scaffold-held-out V8 predictions.",
        "Repository validation and test labels remained sealed.",
        "",
        f"V8 nested MAE is {summary['v8_metrics']['mae']:.4f} across 18,801 structures.",
        f"There are {summary['counts']['heavy_structures_gt_500_mw']:,} structures above 500 Da.",
        f"{summary['counts']['discovery_grade_incremental_effects']} of {summary['counts']['effect_hypotheses']} block-operation hypotheses pass the prespecified internal CI, fold-direction, and BH gates.",
        "",
        "The output includes heavy-molecule regimes, applicability-domain strata, cross-fold conformal coverage, molecule-level block Shapley values, pairwise block synergy stability, and matched-molecular-pair/activity-cliff diagnostics.",
        "These are hypothesis-generating predictive relationships, not causal or receptor-mechanistic proof.",
    ]
    (staging / "ANALYSIS.md").write_text("\n".join(report_lines) + "\n")

    outputs = sorted(p for p in staging.iterdir() if p.is_file())
    manifest = {
        "schema_version": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "inputs": [
            {
                "path": str((v8_root / rel).resolve()),
                "bytes": (v8_root / rel).stat().st_size,
                "sha256": _sha(v8_root / rel),
            }
            for rel in required
        ],
        "artifacts": [{"path": p.name, "bytes": p.stat().st_size, "sha256": _sha(p)} for p in outputs],
    }
    _write_json(staging / "manifest.json", manifest)
    validation_out = {
        "schema_version": SCHEMA,
        "status": "passed",
        "structures_verified": len(cohort),
        "scaffold_exclusivity_verified": True,
        "targets_verified": True,
        "repository_validation_labels_opened": False,
        "repository_test_labels_opened": False,
        "artifact_bindings_verified": len(outputs),
    }
    _write_json(staging / "validation.json", validation_out)
    staging.rename(output_root)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v8-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260817)
    args = parser.parse_args()
    result = analyze(
        args.v8_root.resolve(),
        args.output_root.resolve(),
        replicates=args.bootstrap_replicates,
        seed=args.seed,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
