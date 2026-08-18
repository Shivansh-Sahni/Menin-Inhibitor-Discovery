#!/usr/bin/env python3
"""Apply an explicit applicability gate to the strong in-series hERG model."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator


def _fingerprints(smiles: list[str]):
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    return [generator.GetFingerprint(Chem.MolFromSmiles(text)) for text in smiles]


def build_release(
    results_dir: Path,
    *,
    similarity_threshold: float,
    disagreement_threshold: float,
) -> None:
    predictions = pd.read_csv(results_dir / "private_compound_predictions.csv")
    best = pd.read_csv(results_dir / "best_models.csv")
    primary = predictions[predictions["regime"] == "confidential_only"].copy()
    reference = (
        primary[primary["herg_blocker_label"].notna()]
        .drop_duplicates("standard_inchi_key")
        .reset_index(drop=True)
    )
    query_fingerprints = _fingerprints(primary["smiles"].astype(str).tolist())
    reference_fingerprints = _fingerprints(reference["smiles"].astype(str).tolist())
    reference_keys = reference["standard_inchi_key"].astype(str).tolist()

    similarities: list[float] = []
    for query_key, query_fingerprint in zip(
        primary["standard_inchi_key"].astype(str), query_fingerprints, strict=True
    ):
        scores = list(DataStructs.BulkTanimotoSimilarity(query_fingerprint, reference_fingerprints))
        for index, reference_key in enumerate(reference_keys):
            if query_key == reference_key:
                scores[index] = -1.0
        similarities.append(max(scores) if scores else 0.0)

    primary["nearest_labeled_tanimoto"] = similarities
    primary["within_similarity_domain"] = primary["nearest_labeled_tanimoto"] >= similarity_threshold
    primary["low_model_disagreement"] = primary["ensemble_std"] <= disagreement_threshold
    primary["release_status"] = np.where(
        primary["within_similarity_domain"] & primary["low_model_disagreement"],
        "prediction_released",
        "assay_required",
    )
    primary["released_risk_band"] = np.where(
        primary["release_status"] == "prediction_released",
        primary["risk_band"],
        "unresolved",
    )
    primary.to_csv(results_dir / "herg_indomain_predictions.csv", index=False)

    winner = best[best["regime"] == "confidential_only"].sort_values("ensemble_rank").iloc[0]
    accuracy = (winner["tn"] + winner["tp"]) / (winner["tn"] + winner["fp"] + winner["fn"] + winner["tp"])
    unlabeled = primary[primary["herg_blocker_label"].isna()]
    released = unlabeled[unlabeled["release_status"] == "prediction_released"]
    lines = [
        "# Strong in-domain hERG model card",
        "",
        "## Approved use",
        "",
        "Analog prioritization within the current Menin chemical series. Predictions outside the similarity/disagreement gate are withheld and require an experimental hERG assay.",
        "",
        "## Winning straightforward model",
        "",
        f"- Model: {winner['family']} ({winner['complexity']}).",
        f"- Features: {winner['feature_set']}.",
        f"- Parameters: `{winner['parameters_json']}`.",
        "- Validation: 5-fold repeated stratified structure CV, 3 repeats (15 held-out folds).",
        f"- ROC AUC: {winner['roc_auc']:.3f}.",
        f"- Accuracy: {accuracy:.3f}.",
        f"- Balanced accuracy: {winner['balanced_accuracy']:.3f}.",
        f"- MCC: {winner['mcc']:.3f}.",
        f"- Sensitivity/specificity: {winner['sensitivity']:.3f}/{winner['specificity']:.3f}.",
        f"- Brier score: {winner['brier']:.3f}.",
        "",
        "## Release gate",
        "",
        f"- Nearest labeled Morgan-Tanimoto similarity must be at least {similarity_threshold:.2f}.",
        f"- Ensemble standard deviation must be at most {disagreement_threshold:.2f}.",
        f"- Unlabeled/intermediate compounds released: {len(released)}/{len(unlabeled)}.",
        f"- Unlabeled/intermediate compounds routed to assay: {len(unlabeled) - len(released)}/{len(unlabeled)}.",
        "",
        "## Non-approved use",
        "",
        "Novel-scaffold extrapolation. Locked scaffold validation did not reach the 0.90 performance gate and must not be represented by the in-domain score.",
        "",
    ]
    (results_dir / "indomain_model_card.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--similarity-threshold", type=float, default=0.80)
    parser.add_argument("--disagreement-threshold", type=float, default=0.20)
    args = parser.parse_args()
    build_release(
        args.results_dir.resolve(),
        similarity_threshold=args.similarity_threshold,
        disagreement_threshold=args.disagreement_threshold,
    )


if __name__ == "__main__":
    main()
