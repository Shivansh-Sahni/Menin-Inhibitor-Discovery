#!/usr/bin/env python3
"""Explain selected hERG model residuals, disagreements, and analogue cliffs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "pipeline/scripts"
SRC = ROOT / "pipeline/src"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SRC))

import run_herg_pk_mix_match as base  # noqa: E402
from menin_discovery.research_common import (  # noqa: E402
    atomic_write_csv,
    atomic_write_json,
    atomic_write_text,
)

DEFAULT_OUTPUT = base.DEFAULT_OUTPUT


def _bh_adjust(p_values: list[float]) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 1.0
    for reverse_rank, index in enumerate(order[::-1], start=1):
        rank = len(values) - reverse_rank + 1
        running = min(running, values[index] * len(values) / rank)
        adjusted[index] = running
    return np.clip(adjusted, 0.0, 1.0)


def _residual_correlations(
    predictions: pd.DataFrame,
    features: pd.DataFrame,
) -> pd.DataFrame:
    anchor = predictions[
        predictions["evaluation"].eq("internal_scaffold_cv")
        & predictions["data_regime"].eq("internal_only")
        & predictions["feature_layer"].eq("hybrid")
        & predictions["model"].eq("ridge")
    ].copy()
    work = anchor.merge(
        features[["record_id", *base.PROXY_COLUMNS]],
        on="record_id",
        how="inner",
        validate="one_to_one",
    )
    work["signed_residual"] = work["predicted_pic50"] - work["observed_pic50"]
    work["absolute_residual"] = work["signed_residual"].abs()
    rows: list[dict[str, Any]] = []
    for outcome in ("signed_residual", "absolute_residual"):
        for feature in (*base.PROXY_COLUMNS, "max_internal_like_train_tanimoto"):
            statistic, p_value = spearmanr(
                work[feature].to_numpy(dtype=float),
                work[outcome].to_numpy(dtype=float),
            )
            rows.append(
                {
                    "evaluation": "internal_scaffold_cv",
                    "anchor": "internal_only_hybrid_ridge",
                    "residual_quantity": outcome,
                    "error_stratification_variable": feature,
                    "n": len(work),
                    "spearman": float(statistic),
                    "p_value_unadjusted": float(p_value),
                    "interpretation": (
                        "exploratory residual association with an existing input/domain "
                        "variable; not causal and not a new admitted model feature"
                    ),
                }
            )
    result = pd.DataFrame(rows)
    result["p_value_bh_within_all_tests"] = _bh_adjust(result["p_value_unadjusted"].tolist())
    return result.sort_values(["residual_quantity", "p_value_bh_within_all_tests"]).reset_index(drop=True)


def _prediction_disagreements(
    continuous_predictions: pd.DataFrame,
    binary_predictions: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for evaluation in ("internal_scaffold_cv", "angelo_fixed_nonoverlap"):
        continuous_anchor = continuous_predictions[
            continuous_predictions["evaluation"].eq(evaluation)
            & continuous_predictions["data_regime"].eq("internal_only")
            & continuous_predictions["feature_layer"].eq("hybrid")
            & continuous_predictions["model"].eq("ridge")
        ][
            [
                "record_id",
                "compound_id",
                "scaffold",
                "observed_pic50",
                "predicted_pic50",
                "max_internal_like_train_tanimoto",
            ]
        ].rename(columns={"predicted_pic50": "anchor_prediction"})
        continuous_challenger = continuous_predictions[
            continuous_predictions["evaluation"].eq(evaluation)
            & continuous_predictions["data_regime"].eq("public_prior_internal_residual")
            & continuous_predictions["feature_layer"].eq("hybrid")
            & continuous_predictions["model"].eq("ridge")
        ][["record_id", "predicted_pic50"]].rename(columns={"predicted_pic50": "challenger_prediction"})
        continuous_pair = continuous_anchor.merge(
            continuous_challenger,
            on="record_id",
            validate="one_to_one",
        )
        continuous_pair["analysis"] = "continuous_pIC50"
        continuous_pair["evaluation"] = evaluation
        continuous_pair["absolute_model_disagreement"] = (
            continuous_pair["anchor_prediction"] - continuous_pair["challenger_prediction"]
        ).abs()
        continuous_pair["priority_reason"] = "ranked model-form disagreement; >=0.5 pIC50 is material"
        rows.append(continuous_pair)

        binary_anchor = binary_predictions[
            binary_predictions["evaluation"].eq(evaluation)
            & binary_predictions["data_regime"].eq("internal_only")
            & binary_predictions["feature_layer"].eq("compact_proxies")
            & binary_predictions["model"].eq("extra_trees")
        ][
            [
                "record_id",
                "compound_id",
                "scaffold",
                "observed_class",
                "predicted_blocker_probability",
                "max_internal_like_train_tanimoto",
            ]
        ].rename(
            columns={
                "predicted_blocker_probability": "anchor_prediction",
            }
        )
        binary_challenger = binary_predictions[
            binary_predictions["evaluation"].eq(evaluation)
            & binary_predictions["data_regime"].eq("internal_plus_public_nearest")
            & binary_predictions["feature_layer"].eq("hybrid")
            & binary_predictions["model"].eq("logistic")
        ][["record_id", "predicted_blocker_probability"]].rename(
            columns={
                "predicted_blocker_probability": "challenger_prediction",
            }
        )
        binary_pair = binary_anchor.merge(
            binary_challenger,
            on="record_id",
            validate="one_to_one",
        )
        binary_pair["analysis"] = "interval_decisive_blocker_probability"
        binary_pair["evaluation"] = evaluation
        binary_pair["absolute_model_disagreement"] = (
            binary_pair["anchor_prediction"] - binary_pair["challenger_prediction"]
        ).abs()
        binary_pair["priority_reason"] = "ranked model-form disagreement; probability gap is descriptive"
        rows.append(binary_pair)
    combined = pd.concat(rows, ignore_index=True, sort=False)
    return combined.sort_values(
        ["analysis", "evaluation", "absolute_model_disagreement"],
        ascending=[True, True, False],
    ).reset_index(drop=True)


def _cliff_candidates(
    internal: pd.DataFrame,
    extension: pd.DataFrame,
    continuous_predictions: pd.DataFrame,
) -> pd.DataFrame:
    structures = pd.concat([internal, extension], ignore_index=True, sort=False)
    structures = structures.drop_duplicates("record_id").reset_index(drop=True)
    bits = np.vstack(structures["fingerprint"].to_numpy()).astype(bool)
    standard_deviation = structures[base.PROXY_COLUMNS].std(ddof=0).replace(0, np.nan)
    internal_prediction = continuous_predictions[
        continuous_predictions["evaluation"].eq("internal_scaffold_cv")
        & continuous_predictions["data_regime"].eq("internal_only")
        & continuous_predictions["feature_layer"].eq("hybrid")
        & continuous_predictions["model"].eq("ridge")
    ][["record_id", "predicted_pic50", "fold"]].rename(columns={"predicted_pic50": "anchor_prediction"})
    extension_prediction = continuous_predictions[
        continuous_predictions["evaluation"].eq("angelo_fixed_nonoverlap")
        & continuous_predictions["data_regime"].eq("internal_only")
        & continuous_predictions["feature_layer"].eq("hybrid")
        & continuous_predictions["model"].eq("ridge")
    ][["record_id", "predicted_pic50", "fold"]].rename(columns={"predicted_pic50": "anchor_prediction"})
    anchor = pd.concat(
        [internal_prediction, extension_prediction],
        ignore_index=True,
    ).drop_duplicates("record_id")
    structures = structures.merge(anchor, on="record_id", how="left")
    rows: list[dict[str, Any]] = []
    for left in range(len(structures)):
        intersections = np.logical_and(bits[left + 1 :], bits[left]).sum(axis=1)
        unions = np.logical_or(bits[left + 1 :], bits[left]).sum(axis=1)
        similarities = np.divide(
            intersections,
            unions,
            out=np.zeros_like(intersections, dtype=float),
            where=unions > 0,
        )
        for offset in np.flatnonzero(similarities >= 0.80):
            right = left + 1 + int(offset)
            first = structures.iloc[left]
            second = structures.iloc[right]
            observed_delta = abs(float(first["target_pic50"]) - float(second["target_pic50"]))
            if observed_delta < 1.0:
                continue
            standardized_difference = (
                (first[base.PROXY_COLUMNS].astype(float) - second[base.PROXY_COLUMNS].astype(float)).abs()
                / standard_deviation
            ).sort_values(ascending=False)
            comparable = bool(str(first["source_group"]) == str(second["source_group"]))
            if first["source_group"] == "internal" and second["source_group"] == "internal":
                comparable = bool(first["fold"] == second["fold"])
                comparison_note = (
                    "same OOF fold model"
                    if comparable
                    else "different OOF fold fits; predicted delta not directly comparable"
                )
            elif (
                first["source_group"] == "angelo_same_series_extension"
                and second["source_group"] == "angelo_same_series_extension"
            ):
                comparable = True
                comparison_note = "same full internal anchor fit"
            else:
                comparable = False
                comparison_note = "internal OOF versus fixed-extension fit"
            predicted_delta = (
                abs(float(first["anchor_prediction"]) - float(second["anchor_prediction"]))
                if pd.notna(first["anchor_prediction"]) and pd.notna(second["anchor_prediction"])
                else np.nan
            )
            rows.append(
                {
                    "record_id_a": first["record_id"],
                    "record_id_b": second["record_id"],
                    "source_a": first["source_group"],
                    "source_b": second["source_group"],
                    "tanimoto_ecfp4": float(similarities[offset]),
                    "observed_pic50_a": float(first["target_pic50"]),
                    "observed_pic50_b": float(second["target_pic50"]),
                    "observed_absolute_delta": observed_delta,
                    "anchor_predicted_absolute_delta": predicted_delta,
                    "prediction_delta_comparable": comparable,
                    "model_under_resolves_cliff": (bool(predicted_delta < 0.5) if comparable else np.nan),
                    "prediction_comparison_note": comparison_note,
                    "largest_standardized_proxy_change": standardized_difference.index[0],
                    "second_standardized_proxy_change": standardized_difference.index[1],
                    "third_standardized_proxy_change": standardized_difference.index[2],
                    "scientific_role": (
                        "outcome-informed analogue-cliff candidate; requires protocol-"
                        "matched replication and chemistry review before mechanistic use"
                    ),
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=[
                "record_id_a",
                "record_id_b",
                "tanimoto_ecfp4",
                "observed_absolute_delta",
            ]
        )
    return (
        pd.DataFrame(rows)
        .sort_values(
            ["observed_absolute_delta", "tanimoto_ecfp4"],
            ascending=False,
        )
        .reset_index(drop=True)
    )


def _report(
    correlations: pd.DataFrame,
    disagreements: pd.DataFrame,
    cliffs: pd.DataFrame,
) -> str:
    significant = correlations[correlations["p_value_bh_within_all_tests"].lt(0.10)]
    material_continuous = disagreements[
        disagreements["analysis"].eq("continuous_pIC50")
        & disagreements["absolute_model_disagreement"].ge(0.5)
    ]
    comparable_cliffs = cliffs[cliffs["prediction_delta_comparable"].eq(True)]
    missed_cliffs = comparable_cliffs[comparable_cliffs["model_under_resolves_cliff"].eq(True)]
    lines = [
        "# hERG residual, disagreement, and analogue-cliff analysis",
        "",
        f"- Residual-proxy tests: {len(correlations)}; BH q<0.10: "
        f"{len(significant)}. These are exploratory hidden-variable screens, not causal "
        "features.",
        f"- Ranked model disagreements: {len(disagreements)} records; continuous gaps "
        f">=0.5 pIC50: {len(material_continuous)}.",
        f"- High-similarity (ECFP4 Tanimoto >=0.80), >=1.0-pIC50 analogue-cliff "
        f"candidates: {len(cliffs)}; prediction-comparable pairs: "
        f"{len(comparable_cliffs)}; under-resolved by the anchor: {len(missed_cliffs)}.",
        "",
        "A high similarity value is not itself a mechanistic explanation. Cliff candidates "
        "are prioritized because they can falsify the idea that the retained global proxies "
        "and local motifs capture all relevant hERG changes. Protocol replication, exact "
        "chemical-difference review, ionization evidence, and eventually environment/"
        "receptor-state physics are required before assigning a mechanism.",
        "",
    ]
    if not significant.empty:
        lines.append("Exploratory residual associations surviving BH q<0.10:")
        for row in significant.itertuples(index=False):
            lines.append(
                f"- {row.residual_quantity} vs "
                f"{row.error_stratification_variable}: Spearman {row.spearman:.3f}, "
                f"q={row.p_value_bh_within_all_tests:.3f}."
            )
    else:
        lines.append(
            "No compact proxy or nearest-neighbor similarity association survives BH "
            "q<0.10. This argues against adding another 2D descriptor solely because it "
            "correlates with the current residuals."
        )
    if not significant.empty:
        lines.extend(
            [
                "",
                "Because these variables are already inputs, the associations indicate "
                "heteroscedastic failure strata or missing interactions/state variables; "
                "they do not justify duplicating the descriptors. High aromaticity/polar "
                "capacity and low sp3 character should instead guide matched-pair review "
                "and future mechanistic measurements.",
            ]
        )
    return "\n".join(lines)


def run(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    continuous_predictions = pd.read_parquet(output / "herg_mix_match_predictions.parquet")
    binary_predictions = pd.read_parquet(output / "herg_binary_predictions.parquet")
    internal, _, tables = base._internal_herg_frames()
    internal_structure_ids = set(tables["compounds"]["structure_id"].astype(str))
    extension, _ = base._extension_frames(internal_structure_ids)

    correlations = _residual_correlations(continuous_predictions, internal)
    disagreements = _prediction_disagreements(
        continuous_predictions,
        binary_predictions,
    )
    cliffs = _cliff_candidates(
        internal,
        extension,
        continuous_predictions,
    )
    report = _report(correlations, disagreements, cliffs)

    atomic_write_csv(output / "herg_residual_proxy_associations.csv", correlations)
    atomic_write_csv(output / "herg_model_disagreements.csv", disagreements)
    atomic_write_csv(output / "same_series_analogue_cliff_candidates.csv", cliffs)
    atomic_write_text(output / "herg_failure_analysis.md", report)
    payload = {
        "status": "completed",
        "residual_proxy_tests": len(correlations),
        "residual_associations_bh_q_lt_0p10": int(correlations["p_value_bh_within_all_tests"].lt(0.10).sum()),
        "ranked_model_disagreement_rows": len(disagreements),
        "continuous_disagreements_ge_0p5": int(
            (
                disagreements["analysis"].eq("continuous_pIC50")
                & disagreements["absolute_model_disagreement"].ge(0.5)
            ).sum()
        ),
        "analogue_cliff_candidates": len(cliffs),
        "prediction_comparable_cliffs": int(
            cliffs.get(
                "prediction_delta_comparable",
                pd.Series(dtype=bool),
            )
            .eq(True)
            .sum()
        ),
        "claim_boundary": (
            "outcome-informed failure analysis; candidates require protocol replication "
            "and mechanistic falsification"
        ),
    }
    atomic_write_json(output / "failure_analysis_summary.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2))


if __name__ == "__main__":
    main()
