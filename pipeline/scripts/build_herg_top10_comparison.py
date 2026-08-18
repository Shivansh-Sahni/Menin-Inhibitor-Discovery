#!/usr/bin/env python3
"""Compare top-ten public/private hERG models and their OOF ensembles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    matthews_corrcoef,
    roc_auc_score,
)

REGIMES = ("equal_importance", "confidential_prioritized")


def _selection_score(roc_auc: float, balanced_accuracy: float, mcc: float, brier: float) -> float:
    return float(
        0.40 * roc_auc + 0.30 * balanced_accuracy + 0.15 * ((mcc + 1.0) / 2.0) + 0.15 * (1.0 - brier)
    )


def _ensemble_metrics(probabilities: pd.DataFrame) -> dict[str, float | int]:
    observed = probabilities.index.get_level_values("observed_label").to_numpy(dtype=int)
    probability = probabilities.mean(axis=1).to_numpy(dtype=float)
    prediction = (probability >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(observed, prediction, labels=[0, 1]).ravel()
    roc_auc = float(roc_auc_score(observed, probability))
    balanced_accuracy = float(balanced_accuracy_score(observed, prediction))
    mcc = float(matthews_corrcoef(observed, prediction))
    brier = float(brier_score_loss(observed, probability))
    return {
        "n_models": int(probabilities.shape[1]),
        "n_oof_predictions": int(len(observed)),
        "accuracy": float((tn + tp) / len(observed)),
        "roc_auc": roc_auc,
        "pr_auc": float(average_precision_score(observed, probability)),
        "balanced_accuracy": balanced_accuracy,
        "mcc": mcc,
        "brier": brier,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "selection_score": _selection_score(roc_auc, balanced_accuracy, mcc, brier),
    }


def build_comparison(input_dir: Path) -> None:
    cv_results = pd.read_csv(input_dir / "cv_results.csv")
    oof = pd.read_csv(input_dir / "oof_private_predictions.csv")
    manifest = json.loads((input_dir / "run_manifest.json").read_text())

    ranking_frames: list[pd.DataFrame] = []
    ensemble_rows: list[dict[str, float | int | str]] = []
    for regime in REGIMES:
        ranked = (
            cv_results[cv_results["regime"].eq(regime)]
            .sort_values("selection_score", ascending=False)
            .head(10)
            .copy()
        )
        ranked.insert(1, "rank", np.arange(1, len(ranked) + 1, dtype=int))
        ranked["accuracy"] = (ranked["tn"] + ranked["tp"]) / (
            ranked["tn"] + ranked["fp"] + ranked["fn"] + ranked["tp"]
        )
        ranking_frames.append(ranked)

        selected = set(zip(ranked["model_key"], ranked["feature_set"], strict=True))
        regime_oof = oof[oof["regime"].eq(regime)].copy()
        keep = [
            (model_key, feature_set) in selected
            for model_key, feature_set in zip(regime_oof["model_key"], regime_oof["feature_set"], strict=True)
        ]
        regime_oof = regime_oof[keep].copy()
        regime_oof["configuration"] = regime_oof["model_key"] + "__" + regime_oof["feature_set"]
        probability_matrix = regime_oof.pivot(
            index=["fold", "private_labeled_position", "observed_label"],
            columns="configuration",
            values="probability",
        )
        ensemble_rows.append({"regime": regime, **_ensemble_metrics(probability_matrix)})

    rankings = pd.concat(ranking_frames, ignore_index=True)
    ensembles = pd.DataFrame(ensemble_rows)
    rankings.to_csv(input_dir / "top10_model_rankings.csv", index=False)
    ensembles.to_csv(input_dir / "top10_ensemble_summary.csv", index=False)

    private_audit = manifest["private_audit"]
    public_audit = manifest["public_audit"]
    lines = [
        "# Top-ten public/private hERG comparison",
        "",
        "Both experiments use the same public and private structures. Validation is always performed on held-out private Menin compounds.",
        "",
        f"- Public training pool: {public_audit['n_primary_labeled_structures']} electrophysiology IC50 structures.",
        f"- Private labeled pool: {private_audit['n_labeled_unique_structures']} structures "
        f"({private_audit['n_blockers']} blockers, {private_audit['n_nonblockers']} nonblockers).",
        "- Equal-importance regime: every public and private training example has source weight 1.",
        f"- Private-prioritized regime: private training examples have source weight {manifest['confidential_priority_weight']:.0f}; public examples retain weight 1.",
        "- Evaluation: five-fold repeated stratified structure CV, three repeats; 183 held-out predictions per configuration.",
        "",
        "## Ten-model ensemble results",
        "",
        "| Regime | Accuracy | ROC AUC | PR AUC | Balanced accuracy | MCC | Brier |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in ensembles.itertuples(index=False):
        lines.append(
            f"| {row.regime} | {row.accuracy:.3f} | {row.roc_auc:.3f} | {row.pr_auc:.3f} | "
            f"{row.balanced_accuracy:.3f} | {row.mcc:.3f} | {row.brier:.3f} |"
        )
    for regime in REGIMES:
        lines.extend(
            [
                "",
                f"## {regime.replace('_', ' ').title()}: top ten",
                "",
                "| Rank | Family | Complexity | Features | Accuracy | ROC AUC | Balanced accuracy | MCC | Brier |",
                "|---:|---|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in rankings[rankings["regime"].eq(regime)].itertuples(index=False):
            lines.append(
                f"| {row.rank} | {row.family} | {row.complexity} | {row.feature_set} | "
                f"{row.accuracy:.3f} | {row.roc_auc:.3f} | {row.balanced_accuracy:.3f} | "
                f"{row.mcc:.3f} | {row.brier:.3f} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The ten-model average is a fixed, equal-weight ensemble evaluated entirely out of fold. It is not selected or tuned on the held-out probabilities. The best single model should remain preferred when its balanced accuracy and calibration exceed the ensemble.",
            "",
        ]
    )
    (input_dir / "top10_comparison.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("research/benchmarks/herg/strong_ml_indomain"),
    )
    args = parser.parse_args()
    build_comparison(args.input_dir.resolve())


if __name__ == "__main__":
    main()
