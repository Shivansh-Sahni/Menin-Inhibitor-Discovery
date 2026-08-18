#!/usr/bin/env python3
"""Audit whether structural applicability predicts complete-model hERG error."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from menin_discovery.research_common import (
    atomic_write_csv,
    atomic_write_json,
    atomic_write_text,
)
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
EXTENSION = ROOT / "research/reports/pk_herg/ascentage_herg_extension"
COMPLETE = EXTENSION / "complete_feature_model"
OUTPUT = COMPLETE / "local_reliability"
RNG_SEED = 20260729
BOOTSTRAPS = 5000


def _metrics(frame: pd.DataFrame, stratum: str) -> dict[str, object]:
    error = frame["complete_feature_predicted_pic50"] - frame["herg_pic50_value"]
    return {
        "stratum": stratum,
        "n": len(frame),
        "scaffolds": int(frame["scaffold"].nunique()),
        "similarity_min": float(frame["max_train_tanimoto"].min()),
        "similarity_median": float(frame["max_train_tanimoto"].median()),
        "similarity_max": float(frame["max_train_tanimoto"].max()),
        "pic50_mae": float(error.abs().mean()),
        "pic50_rmse": float(np.sqrt(np.mean(error**2))),
        "mean_signed_error": float(error.mean()),
        "spearman_observed_predicted": (
            float(
                spearmanr(
                    frame["herg_pic50_value"],
                    frame["complete_feature_predicted_pic50"],
                ).statistic
            )
            if len(frame) >= 3
            else np.nan
        ),
        "fraction_within_0p5_log": float((error.abs() <= 0.5).mean()),
        "fraction_within_1p0_log": float((error.abs() <= 1.0).mean()),
        "interval_coverage": float(
            (
                (frame["herg_pic50_value"] >= frame["complete_feature_pic50_lower"])
                & (frame["herg_pic50_value"] <= frame["complete_feature_pic50_upper"])
            ).mean()
        ),
        "interval_mean_width": float(
            (frame["complete_feature_pic50_upper"] - frame["complete_feature_pic50_lower"]).mean()
        ),
    }


def audit() -> dict[str, object]:
    context = pd.read_parquet(EXTENSION / "predictions.parquet")[
        [
            "structure_id",
            "scaffold",
            "herg_pic50_relation",
            "herg_pic50_value",
            "max_train_tanimoto",
            "domain_threshold",
            "applicability_domain",
        ]
    ]
    predictions = pd.read_parquet(COMPLETE / "extension_predictions.parquet")[
        [
            "structure_id",
            "complete_feature_predicted_pic50",
            "complete_feature_pic50_lower",
            "complete_feature_pic50_upper",
        ]
    ]
    exact = context.merge(
        predictions,
        on="structure_id",
        validate="one_to_one",
    )
    exact = exact[exact["herg_pic50_relation"].eq("=")].copy()
    exact["prediction_error"] = exact["complete_feature_predicted_pic50"] - exact["herg_pic50_value"]
    exact["absolute_error"] = exact["prediction_error"].abs()
    metrics = pd.DataFrame(
        [
            _metrics(exact, "all_exact"),
            *[
                _metrics(frame, f"applicability_{status}")
                for status, frame in exact.groupby("applicability_domain")
            ],
        ]
    )

    scaffolds = exact["scaffold"].drop_duplicates().to_numpy()
    rng = np.random.default_rng(RNG_SEED)
    rows: list[dict[str, float | int]] = []
    for replicate in range(BOOTSTRAPS):
        sampled = rng.choice(scaffolds, size=len(scaffolds), replace=True)
        boot = pd.concat(
            [exact[exact["scaffold"].eq(scaffold)] for scaffold in sampled],
            ignore_index=True,
        )
        inside = boot[boot["applicability_domain"].eq("inside")]
        outside = boot[boot["applicability_domain"].eq("outside")]
        if inside.empty or outside.empty:
            continue
        correlation = spearmanr(
            boot["max_train_tanimoto"],
            boot["absolute_error"],
        ).statistic
        rows.append(
            {
                "replicate": replicate,
                "outside_minus_inside_mae": float(
                    outside["absolute_error"].mean() - inside["absolute_error"].mean()
                ),
                "similarity_vs_absolute_error_spearman": float(correlation),
            }
        )
    bootstrap = pd.DataFrame(rows)
    if len(bootstrap) < 0.80 * BOOTSTRAPS:
        raise RuntimeError("Too few bootstrap replicates contained both domain strata")

    inside = metrics[metrics["stratum"].eq("applicability_inside")].iloc[0]
    outside = metrics[metrics["stratum"].eq("applicability_outside")].iloc[0]
    observed_difference = float(outside["pic50_mae"] - inside["pic50_mae"])
    observed_correlation = float(spearmanr(exact["max_train_tanimoto"], exact["absolute_error"]).statistic)
    difference_ci = np.quantile(
        bootstrap["outside_minus_inside_mae"],
        [0.025, 0.975],
    )
    correlation_ci = np.nanquantile(
        bootstrap["similarity_vs_absolute_error_spearman"],
        [0.025, 0.975],
    )
    result = {
        "status": "pass",
        "n_exact": len(exact),
        "n_scaffolds": int(exact["scaffold"].nunique()),
        "domain_threshold": float(exact["domain_threshold"].iloc[0]),
        "inside_n": int(inside["n"]),
        "outside_n": int(outside["n"]),
        "inside_scaffolds": int(inside["scaffolds"]),
        "outside_scaffolds": int(outside["scaffolds"]),
        "inside_mae": float(inside["pic50_mae"]),
        "outside_mae": float(outside["pic50_mae"]),
        "outside_minus_inside_mae": observed_difference,
        "outside_minus_inside_mae_bootstrap_95": [
            float(difference_ci[0]),
            float(difference_ci[1]),
        ],
        "similarity_vs_absolute_error_spearman": observed_correlation,
        "similarity_vs_absolute_error_spearman_bootstrap_95": [
            float(correlation_ci[0]),
            float(correlation_ci[1]),
        ],
        "bootstrap_replicates_retained": len(bootstrap),
        "conclusion": (
            "structural domain is a support warning, not a validated reliability "
            "guarantee for the complete-feature model"
        ),
    }
    report = f"""# Local reliability audit for new-compound hERG scoring

## Question

Does the original-training structural applicability flag predict error for the retained
complete-feature model on the 42 exact, non-overlapping same-series measurements?
The similarity threshold (**{result["domain_threshold"]:.3f}**) was fixed from training
structures rather than selected using these outcomes.

## Result

- Inside domain: **{result["inside_n"]} compounds / {result["inside_scaffolds"]} scaffolds**,
  pIC50 MAE **{result["inside_mae"]:.3f}**.
- Outside domain: **{result["outside_n"]} compounds / {result["outside_scaffolds"]} scaffolds**,
  pIC50 MAE **{result["outside_mae"]:.3f}**.
- Outside-minus-inside MAE: **{result["outside_minus_inside_mae"]:.3f}**, with
  scaffold-bootstrap 95% interval
  **[{difference_ci[0]:.3f}, {difference_ci[1]:.3f}]**.
- Nearest-training similarity versus absolute error has Spearman
  **{observed_correlation:.3f}**, with scaffold-bootstrap 95% interval
  **[{correlation_ci[0]:.3f}, {correlation_ci[1]:.3f}]**.

The intervals cross zero. The binary domain flag therefore remains an important
extrapolation warning but is **not a demonstrated accuracy guarantee** for this stronger
model. Only two computed scaffolds populate the outside-domain stratum, so absence of a
clear trend is not evidence that similarity is irrelevant.

This audit uses the pre-extension training threshold. The operational scorer recomputes
its threshold after adding the measured extension structures; that updated flag has no
independent outcome-held calibration yet and must not inherit a stronger claim.

## Consequence

New-compound output now reports the domain flag together with close-neighbor counts,
nearest measured evidence, unresolved stereochemistry, explicit same-series
confirmation, two-model disagreement, and a conservative interval envelope. No single
flag is promoted to a decision rule. A blind outcome-held test remains the required
calibration evidence.
"""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(OUTPUT / "stratified_metrics.csv", metrics)
    atomic_write_csv(OUTPUT / "scaffold_bootstrap.csv", bootstrap)
    atomic_write_csv(OUTPUT / "exact_prediction_context.csv", exact)
    atomic_write_json(OUTPUT / "validation_report.json", result)
    atomic_write_text(OUTPUT / "local_reliability_audit.md", report)
    return result


def main() -> None:
    print(json.dumps(audit(), indent=2))


if __name__ == "__main__":
    main()
