#!/usr/bin/env python3
"""Evaluate a frozen hERG prediction batch after blinded outcomes are released.

The command never refits a model. IC50 relations are converted into pIC50
intervals so exact and censored outcomes retain their correct direction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from menin_discovery.research_common import (
    atomic_write_csv,
    atomic_write_json,
    atomic_write_parquet,
    atomic_write_text,
)
from scipy.stats import spearmanr
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)

REQUIRED_OUTCOME_COLUMNS = {
    "compound_id",
    "herg_ic50_relation",
    "herg_ic50_value_um",
}
MODEL_COLUMNS = {
    "global_control": {
        "mean": "global_control_predicted_pic50",
        "lower": "global_control_pic50_lower",
        "upper": "global_control_pic50_upper",
        "probability": "global_control_blocker_probability",
    },
    "complete_feature": {
        "mean": "complete_feature_predicted_pic50",
        "lower": "complete_feature_pic50_lower",
        "upper": "complete_feature_pic50_upper",
        "probability": "complete_feature_blocker_probability",
    },
}


def _normalize_relation(value: object) -> str:
    relation = str(value).strip()
    aliases = {"==": "=", ">=": ">", "<=": "<"}
    relation = aliases.get(relation, relation)
    if relation not in {"=", ">", "<"}:
        raise ValueError(f"Unsupported IC50 relation: {value!r}")
    return relation


def _normalize_outcomes(path: Path) -> pd.DataFrame:
    outcomes = pd.read_csv(path)
    missing = sorted(REQUIRED_OUTCOME_COLUMNS - set(outcomes.columns))
    if missing:
        raise ValueError(f"Outcome release is missing columns: {missing}")
    if outcomes.empty:
        raise ValueError("Outcome release has no rows")
    if outcomes["compound_id"].isna().any():
        raise ValueError("Every outcome must have a compound_id")
    outcomes["compound_id"] = outcomes["compound_id"].astype(str).str.strip()
    if outcomes["compound_id"].eq("").any() or outcomes["compound_id"].duplicated().any():
        raise ValueError("Outcome compound_id values must be nonblank and unique")
    outcomes["herg_ic50_relation"] = outcomes["herg_ic50_relation"].map(_normalize_relation)
    outcomes["herg_ic50_value_um"] = pd.to_numeric(outcomes["herg_ic50_value_um"], errors="raise")
    if (~np.isfinite(outcomes["herg_ic50_value_um"]) | outcomes["herg_ic50_value_um"].le(0)).any():
        raise ValueError("Every IC50 value or censoring limit must be finite and positive")

    boundary = 6.0 - np.log10(outcomes["herg_ic50_value_um"].to_numpy(dtype=float))
    relation = outcomes["herg_ic50_relation"]
    outcomes["herg_pic50_relation"] = relation.map({"=": "=", ">": "<", "<": ">"})
    outcomes["herg_pic50_lower"] = np.where(
        relation.eq(">"),
        -np.inf,
        boundary,
    )
    outcomes["herg_pic50_upper"] = np.where(
        relation.eq("<"),
        np.inf,
        boundary,
    )
    outcomes["herg_pic50_value"] = np.where(relation.eq("="), boundary, np.nan)
    outcomes["definitive_blocker_at_pic50_5"] = np.select(
        [
            outcomes["herg_pic50_lower"].ge(5.0),
            outcomes["herg_pic50_upper"].lt(5.0),
        ],
        [1.0, 0.0],
        default=np.nan,
    )
    return outcomes


def _ece(labels: np.ndarray, probabilities: np.ndarray, bins: int = 5) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    assignments = np.minimum(np.digitize(probabilities, edges[1:-1]), bins - 1)
    total = len(labels)
    return float(
        sum(
            (mask.sum() / total) * abs(float(labels[mask].mean()) - float(probabilities[mask].mean()))
            for index in range(bins)
            if (mask := assignments == index).any()
        )
    )


def _model_metrics(frame: pd.DataFrame, name: str, columns: dict[str, str]) -> dict[str, object]:
    exact = frame[frame["herg_pic50_relation"].eq("=")]
    censored = frame[~frame["herg_pic50_relation"].eq("=")]
    definitive = frame[frame["definitive_blocker_at_pic50_5"].notna()]
    row: dict[str, object] = {
        "model": name,
        "n_total": len(frame),
        "n_exact": len(exact),
        "n_censored": len(censored),
        "n_definitive_class": len(definitive),
    }
    if not exact.empty:
        observed = exact["herg_pic50_value"].to_numpy(dtype=float)
        predicted = exact[columns["mean"]].to_numpy(dtype=float)
        row.update(
            {
                "pic50_mae": float(mean_absolute_error(observed, predicted)),
                "pic50_rmse": float(np.sqrt(mean_squared_error(observed, predicted))),
                "spearman": (float(spearmanr(observed, predicted).statistic) if len(exact) >= 3 else np.nan),
                "mean_signed_error": float(np.mean(predicted - observed)),
                "fraction_within_0p5_log": float(np.mean(np.abs(predicted - observed) <= 0.5)),
                "fraction_within_1p0_log": float(np.mean(np.abs(predicted - observed) <= 1.0)),
                "interval_coverage": float(
                    np.mean(
                        (observed >= exact[columns["lower"]].to_numpy(dtype=float))
                        & (observed <= exact[columns["upper"]].to_numpy(dtype=float))
                    )
                ),
                "interval_mean_width": float(
                    np.mean(
                        exact[columns["upper"]].to_numpy(dtype=float)
                        - exact[columns["lower"]].to_numpy(dtype=float)
                    )
                ),
            }
        )
    if not censored.empty:
        point = censored[columns["mean"]].to_numpy(dtype=float)
        lower = censored["herg_pic50_lower"].to_numpy(dtype=float)
        upper = censored["herg_pic50_upper"].to_numpy(dtype=float)
        row["censored_point_compatibility"] = float(np.mean((point >= lower) & (point <= upper)))
        predicted_lower = censored[columns["lower"]].to_numpy(dtype=float)
        predicted_upper = censored[columns["upper"]].to_numpy(dtype=float)
        row["censored_interval_compatibility"] = float(
            np.mean((predicted_upper >= lower) & (predicted_lower <= upper))
        )
    if not definitive.empty:
        labels = definitive["definitive_blocker_at_pic50_5"].to_numpy(dtype=int)
        probability = np.clip(
            definitive[columns["probability"]].to_numpy(dtype=float),
            1e-8,
            1.0 - 1e-8,
        )
        calls = probability >= 0.5
        row.update(
            {
                "brier": float(brier_score_loss(labels, probability)),
                "ece_5_bin": _ece(labels, probability),
                "log_loss": float(log_loss(labels, probability, labels=[0, 1])),
                "balanced_accuracy": (
                    float(balanced_accuracy_score(labels, calls)) if len(np.unique(labels)) == 2 else np.nan
                ),
                "mcc": (float(matthews_corrcoef(labels, calls)) if len(np.unique(labels)) == 2 else np.nan),
                "roc_auc": (
                    float(roc_auc_score(labels, probability)) if len(np.unique(labels)) == 2 else np.nan
                ),
            }
        )
    return row


def evaluate(
    predictions_path: Path,
    outcomes_path: Path,
    output: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    predictions = pd.read_csv(predictions_path)
    if predictions["compound_id"].astype(str).duplicated().any():
        raise ValueError("Frozen predictions contain duplicate compound_id values")
    required_prediction_columns = {
        "compound_id",
        "prediction_eligibility",
        *{column for specification in MODEL_COLUMNS.values() for column in specification.values()},
    }
    missing_predictions = sorted(required_prediction_columns - set(predictions.columns))
    if missing_predictions:
        raise ValueError(f"Frozen prediction file is missing columns: {missing_predictions}")
    outcomes = _normalize_outcomes(outcomes_path)
    unknown = sorted(set(outcomes["compound_id"]) - set(predictions["compound_id"].astype(str)))
    if unknown:
        raise ValueError(f"Outcomes contain compounds absent from frozen predictions: {unknown}")
    joined = predictions.merge(
        outcomes,
        on="compound_id",
        how="inner",
        validate="one_to_one",
        suffixes=("", "_observed"),
    )
    eligible = joined[joined["prediction_eligibility"].eq("eligible_same_series_discovery_hypothesis")].copy()
    metrics = pd.DataFrame(
        [
            {
                "evaluation_stratum": stratum,
                **_model_metrics(frame, model, columns),
            }
            for stratum, frame in (("all_released", joined), ("eligible_only", eligible))
            if not frame.empty
            for model, columns in MODEL_COLUMNS.items()
        ]
    )
    released_ids = set(outcomes["compound_id"])
    predicted_ids = set(predictions["compound_id"].astype(str))
    protocol_columns = [
        "assay_protocol_id",
        "platform",
        "cell_line",
        "temperature_c",
        "ph",
        "voltage_protocol",
        "replicate_count",
        "concentration_basis",
    ]
    missing_protocol_columns = [column for column in protocol_columns if column not in outcomes.columns]
    incomplete_protocol_columns = [
        column
        for column in protocol_columns
        if column in outcomes.columns
        and (outcomes[column].isna().any() or outcomes[column].astype(str).str.strip().eq("").any())
    ]
    validation = {
        "status": "pass",
        "models_refit": False,
        "prediction_rows": len(predictions),
        "released_outcome_rows": len(outcomes),
        "eligible_released_rows": len(eligible),
        "unreleased_prediction_rows": len(predicted_ids - released_ids),
        "all_predictions_have_outcomes": predicted_ids == released_ids,
        "exact_outcomes": int(outcomes["herg_pic50_relation"].eq("=").sum()),
        "censored_outcomes": int(outcomes["herg_pic50_relation"].ne("=").sum()),
        "missing_protocol_columns": missing_protocol_columns,
        "incomplete_protocol_columns": incomplete_protocol_columns,
        "protocol_complete": not missing_protocol_columns and not incomplete_protocol_columns,
        "claim_limit": (
            "prospective same-series evaluation only if structures and predictions were "
            "frozen before outcome release; independent-series transfer requires a "
            "separately declared series holdout"
        ),
    }
    report = f"""# Blinded hERG prediction evaluation

The evaluator joined **{len(outcomes)}** released outcomes to a previously generated
prediction file and refit **no models**. There are **{len(eligible)}** records that met
the scorer's same-series, stereochemistry, and applicability requirements.

- Exact outcomes: **{validation["exact_outcomes"]}**
- Censored outcomes: **{validation["censored_outcomes"]}**
- Unreleased prediction rows: **{validation["unreleased_prediction_rows"]}**
- Protocol metadata complete: **{validation["protocol_complete"]}**

IC50 `>x` records are evaluated as pIC50 `<6-log10(x)` intervals; they are never replaced
by `x` or treated as exact values. Primary continuous metrics use exact observations.
Censored point/interval compatibility and definitive blocker classes are reported
separately. The blocker threshold is pIC50 5 (IC50 10 µM).

These results are prospective only if the prediction file was generated before outcomes
were disclosed and was not replaced afterward. No file hash is required by the project,
so dated custody and an outcome-release note must establish that ordering. Same-series
performance does not establish unrelated-series transfer.
"""
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(output / "joined_prediction_outcomes.csv", joined)
    atomic_write_parquet(output / "joined_prediction_outcomes.parquet", joined)
    atomic_write_csv(output / "metrics.csv", metrics)
    atomic_write_json(output / "validation_report.json", validation)
    atomic_write_text(output / "evaluation_report.md", report)
    return joined, metrics, validation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--outcomes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    joined, _, validation = evaluate(args.predictions, args.outcomes, args.output)
    print(
        json.dumps(
            {
                "evaluated": len(joined),
                "eligible": validation["eligible_released_rows"],
                "output": str(args.output.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
