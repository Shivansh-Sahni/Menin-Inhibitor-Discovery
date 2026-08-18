#!/usr/bin/env python3
"""Robustness controls for the selected internal hERG anchors.

The controls are deliberately narrow: they test the two selected internal
anchors rather than opening another model-selection loop.

* continuous pIC50: hybrid representation plus ridge regression;
* interval-decisive class: compact proxies plus ExtraTrees.

Outputs include leave-one-scaffold-out predictions, repeated scaffold-group
splits, similarity-stratified errors, and outcome-permutation nulls.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/menin-robustness-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/menin-robustness-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "pipeline/scripts"
SRC = ROOT / "pipeline/src"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SRC))

import run_herg_binary_mix_match as binary  # noqa: E402
import run_herg_pk_mix_match as continuous  # noqa: E402
from menin_discovery.research_common import (  # noqa: E402
    atomic_write_csv,
    atomic_write_json,
    atomic_write_parquet,
    atomic_write_text,
)

SEED = continuous.SEED + 20
PERMUTATIONS = 250
REPEATED_SPLITS = 250
DEFAULT_OUTPUT = continuous.DEFAULT_OUTPUT


def _continuous_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> np.ndarray:
    work = train.copy()
    work["sample_weight"] = 1.0
    return continuous._fit_predict(
        work,
        test,
        feature_layer="hybrid",
        model_name="ridge",
    )


def _binary_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> np.ndarray:
    work = train.copy()
    work["sample_weight"] = binary._class_balanced_weights(
        work,
        source_balance=False,
    )
    return binary._fit_predict_probability(
        work,
        test,
        feature_layer="compact_proxies",
        model_name="extra_trees",
    )


def _regression_metric_row(
    observed: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, float]:
    error = predicted - observed
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "spearman": continuous._safe_spearman(observed, predicted),
        "fraction_within_0p5": float(np.mean(np.abs(error) <= 0.5)),
    }


def _classification_metric_row(
    observed: np.ndarray,
    probability: np.ndarray,
) -> dict[str, float]:
    frame = pd.DataFrame(
        {
            "observed_class": observed.astype(int),
            "predicted_blocker_probability": probability,
            "predicted_class": (probability >= 0.5).astype(int),
            "scaffold": np.arange(len(observed)).astype(str),
            "max_internal_like_train_tanimoto": 1.0,
        }
    )
    metrics = binary._metrics(frame)
    return {
        "n_blockers": float(observed.sum()),
        "n_nonblockers": float((observed == 0).sum()),
        "brier": metrics["brier"],
        "roc_auc": metrics["roc_auc"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "sensitivity": metrics["sensitivity"],
        "specificity": metrics["specificity"],
    }


def _loso_predictions(
    frame: pd.DataFrame,
    *,
    target: str,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for fold, scaffold in enumerate(sorted(frame["scaffold"].astype(str).unique())):
        test = frame[frame["scaffold"].astype(str).eq(scaffold)].copy()
        train = frame[~frame["scaffold"].astype(str).eq(scaffold)].copy()
        if target == "continuous":
            predicted = _continuous_predict(train, test)
            result = test[["record_id", "compound_id", "scaffold", "target_pic50"]].copy()
            result = result.rename(columns={"target_pic50": "observed"})
        else:
            if train["target_class"].nunique() < 2:
                continue
            predicted = _binary_predict(train, test)
            result = test[["record_id", "compound_id", "scaffold", "target_class"]].copy()
            result = result.rename(columns={"target_class": "observed"})
        result["predicted"] = predicted
        result["fold"] = fold
        result["analysis"] = target
        result["max_train_tanimoto"] = continuous._max_tanimoto(test, train)
        rows.append(result)
    return pd.concat(rows, ignore_index=True)


def _repeated_group_splits(
    frame: pd.DataFrame,
    *,
    target: str,
) -> pd.DataFrame:
    splitter = GroupShuffleSplit(
        n_splits=REPEATED_SPLITS,
        test_size=0.25,
        random_state=SEED,
    )
    rows: list[dict[str, Any]] = []
    for split, (train_index, test_index) in enumerate(
        splitter.split(frame, groups=frame["scaffold"].astype(str))
    ):
        train = frame.iloc[train_index].copy()
        test = frame.iloc[test_index].copy()
        if target == "continuous":
            observed = test["target_pic50"].to_numpy(dtype=float)
            predicted = _continuous_predict(train, test)
            metrics = _regression_metric_row(observed, predicted)
        else:
            if train["target_class"].nunique() < 2 or test["target_class"].nunique() < 2:
                rows.append(
                    {
                        "analysis": target,
                        "split": split,
                        "n_train": len(train),
                        "n_test": len(test),
                        "train_scaffolds": train["scaffold"].nunique(),
                        "test_scaffolds": test["scaffold"].nunique(),
                        "status": "excluded_single_class_train_or_test",
                    }
                )
                continue
            observed = test["target_class"].to_numpy(dtype=int)
            predicted = _binary_predict(train, test)
            metrics = _classification_metric_row(observed, predicted)
        rows.append(
            {
                "analysis": target,
                "split": split,
                "n_train": len(train),
                "n_test": len(test),
                "train_scaffolds": train["scaffold"].nunique(),
                "test_scaffolds": test["scaffold"].nunique(),
                "status": "completed",
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def _fixed_group_oof(
    frame: pd.DataFrame,
    *,
    target: str,
    permuted: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    splitter = GroupKFold(n_splits=5)
    observed_parts: list[np.ndarray] = []
    prediction_parts: list[np.ndarray] = []
    for train_index, test_index in splitter.split(
        frame,
        groups=frame["scaffold"].astype(str),
    ):
        train = frame.iloc[train_index].copy()
        test = frame.iloc[test_index].copy()
        if target == "continuous":
            if permuted is not None:
                train["target_pic50"] = permuted[train_index]
                observed = permuted[test_index]
            else:
                observed = test["target_pic50"].to_numpy(dtype=float)
            predicted = _continuous_predict(train, test)
        else:
            if permuted is not None:
                train["target_class"] = permuted[train_index].astype(int)
                observed = permuted[test_index].astype(int)
            else:
                observed = test["target_class"].to_numpy(dtype=int)
            if train["target_class"].nunique() < 2:
                raise ValueError("Permuted training fold lost a class")
            predicted = _binary_predict(train, test)
        observed_parts.append(np.asarray(observed))
        prediction_parts.append(np.asarray(predicted))
    return np.concatenate(observed_parts), np.concatenate(prediction_parts)


def _permutation_null(
    frame: pd.DataFrame,
    *,
    target: str,
) -> tuple[pd.DataFrame, dict[str, float]]:
    observed_y, observed_prediction = _fixed_group_oof(frame, target=target)
    if target == "continuous":
        observed_metrics = _regression_metric_row(observed_y, observed_prediction)
        source_y = frame["target_pic50"].to_numpy(dtype=float)
        primary = "mae"
        secondary = "spearman"
    else:
        observed_metrics = _classification_metric_row(observed_y, observed_prediction)
        source_y = frame["target_class"].to_numpy(dtype=int)
        primary = "brier"
        secondary = "roc_auc"
    rng = np.random.default_rng(SEED + (0 if target == "continuous" else 1000))
    rows: list[dict[str, Any]] = []
    for permutation in range(PERMUTATIONS):
        permuted = rng.permutation(source_y)
        null_y, null_prediction = _fixed_group_oof(
            frame,
            target=target,
            permuted=permuted,
        )
        metrics = (
            _regression_metric_row(null_y, null_prediction)
            if target == "continuous"
            else _classification_metric_row(null_y, null_prediction)
        )
        rows.append(
            {
                "analysis": target,
                "permutation": permutation,
                **metrics,
            }
        )
    null = pd.DataFrame(rows)
    primary_p = float(
        (1 + np.sum(null[primary].to_numpy(dtype=float) <= observed_metrics[primary])) / (PERMUTATIONS + 1)
    )
    secondary_values = null[secondary].to_numpy(dtype=float)
    secondary_values = secondary_values[np.isfinite(secondary_values)]
    secondary_p = float(
        (1 + np.sum(secondary_values >= observed_metrics[secondary])) / (len(secondary_values) + 1)
    )
    summary = {
        **{f"observed_{key}": float(value) for key, value in observed_metrics.items()},
        f"{primary}_permutation_p": primary_p,
        f"{secondary}_permutation_p": secondary_p,
        f"null_{primary}_median": float(null[primary].median()),
        f"null_{secondary}_median": float(null[secondary].median()),
        "permutations": PERMUTATIONS,
    }
    return null, summary


def _similarity_strata(loso: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for analysis, frame in loso.groupby("analysis", sort=True):
        work = frame.copy()
        work["similarity_stratum"] = pd.qcut(
            work["max_train_tanimoto"],
            q=3,
            labels=["low", "middle", "high"],
            duplicates="drop",
        )
        for stratum, group in work.groupby("similarity_stratum", observed=True):
            observed = group["observed"].to_numpy()
            predicted = group["predicted"].to_numpy(dtype=float)
            metrics = (
                _regression_metric_row(observed.astype(float), predicted)
                if analysis == "continuous"
                else _classification_metric_row(observed.astype(int), predicted)
            )
            rows.append(
                {
                    "analysis": analysis,
                    "similarity_stratum": str(stratum),
                    "n": len(group),
                    "similarity_min": float(group["max_train_tanimoto"].min()),
                    "similarity_median": float(group["max_train_tanimoto"].median()),
                    "similarity_max": float(group["max_train_tanimoto"].max()),
                    "n_blockers": (
                        int(group["observed"].astype(int).sum()) if analysis == "binary" else np.nan
                    ),
                    "n_nonblockers": (
                        int(group["observed"].astype(int).eq(0).sum()) if analysis == "binary" else np.nan
                    ),
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def _summarize_repeated(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for analysis, group in frame[frame["status"].eq("completed")].groupby("analysis"):
        metrics = (
            ["mae", "rmse", "spearman", "fraction_within_0p5"]
            if analysis == "continuous"
            else [
                "brier",
                "roc_auc",
                "balanced_accuracy",
                "sensitivity",
                "specificity",
            ]
        )
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            rows.append(
                {
                    "analysis": analysis,
                    "metric": metric,
                    "completed_splits": len(group),
                    "median": float(values.median()),
                    "lower_2p5": float(values.quantile(0.025)),
                    "upper_97p5": float(values.quantile(0.975)),
                }
            )
    return pd.DataFrame(rows)


def _report(
    permutation_summary: pd.DataFrame,
    repeated_summary: pd.DataFrame,
    similarity: pd.DataFrame,
) -> str:
    c = permutation_summary[permutation_summary["analysis"].eq("continuous")].iloc[0]
    b = permutation_summary[permutation_summary["analysis"].eq("binary")].iloc[0]
    continuous_repeated = repeated_summary[
        (repeated_summary["analysis"].eq("continuous")) & (repeated_summary["metric"].eq("mae"))
    ].iloc[0]
    binary_repeated = repeated_summary[
        (repeated_summary["analysis"].eq("binary")) & (repeated_summary["metric"].eq("brier"))
    ].iloc[0]
    lines = [
        "# Selected-anchor robustness controls",
        "",
        "These controls do not select new models. They stress-test the already selected "
        "internal continuous hybrid-ridge anchor and compact-proxy ExtraTrees classifier.",
        "",
        f"- Continuous fixed 5-fold scaffold OOF MAE: {c['observed_mae']:.3f}; "
        f"median shuffled-target MAE {c['null_mae_median']:.3f}; empirical "
        f"p={c['mae_permutation_p']:.4f} across {int(c['permutations'])} permutations.",
        f"- Continuous repeated scaffold split MAE median "
        f"{continuous_repeated['median']:.3f} "
        f"[{continuous_repeated['lower_2p5']:.3f}, "
        f"{continuous_repeated['upper_97p5']:.3f}].",
        f"- Binary fixed 5-fold scaffold OOF Brier: {b['observed_brier']:.3f}; "
        f"median shuffled-target Brier {b['null_brier_median']:.3f}; empirical "
        f"p={b['brier_permutation_p']:.4f} across {int(b['permutations'])} permutations.",
        f"- Binary repeated scaffold split Brier median "
        f"{binary_repeated['median']:.3f} "
        f"[{binary_repeated['lower_2p5']:.3f}, "
        f"{binary_repeated['upper_97p5']:.3f}].",
        "",
        "Permutation evidence tests whether the selected models capture more structure-"
        "outcome signal than chance under the same grouped folds. It does not establish "
        "prospective transfer, because model selection and robustness testing use the same "
        "internal collection.",
        "",
        "Similarity-stratified results are in `similarity_strata.csv`; the low-similarity "
        "stratum is the most relevant applicability stress test. Small stratum counts and "
        "class imbalance must be retained when interpreting apparent gradients.",
        "",
    ]
    for record in similarity.itertuples(index=False):
        metric = (
            f"MAE {record.mae:.3f}"
            if record.analysis == "continuous"
            else (f"Brier {record.brier:.3f}; {int(record.n_nonblockers)} nonblockers")
        )
        lines.append(
            f"- {record.analysis}, {record.similarity_stratum} similarity "
            f"(median {record.similarity_median:.3f}, n={record.n}): {metric}."
        )
    return "\n".join(lines)


def _figures(
    permutation: pd.DataFrame,
    permutation_summary: pd.DataFrame,
    output: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    definitions = [
        ("continuous", "mae", "Continuous pIC50 MAE", "#2A6F97"),
        ("binary", "brier", "Decisive-class Brier", "#386641"),
    ]
    for axis, (analysis, metric, title, color) in zip(
        axes,
        definitions,
        strict=True,
    ):
        values = permutation[permutation["analysis"].eq(analysis)][metric]
        observed = float(
            permutation_summary[permutation_summary["analysis"].eq(analysis)][f"observed_{metric}"].iloc[0]
        )
        axis.hist(values, bins=25, color=color, alpha=0.8)
        axis.axvline(observed, color="#9B2226", linewidth=2, label="observed")
        axis.set_title(title)
        axis.set_xlabel(f"Shuffled-target {metric}")
        axis.set_ylabel("Count")
        axis.legend()
        axis.grid(alpha=0.2)
    figure.suptitle("Grouped outcome-permutation controls")
    figure.tight_layout()
    continuous._atomic_figure(
        figure,
        output / "selected_anchor_permutation_controls.png",
        dpi=220,
    )
    continuous._atomic_figure(
        figure,
        output / "selected_anchor_permutation_controls.pdf",
    )
    plt.close(figure)


def run(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    internal_continuous, internal_intervals, tables = continuous._internal_herg_frames()
    internal_binary, _ = binary._decisive_interval_frame(
        internal_intervals,
        dataset_role="internal_interval_decisive",
    )

    continuous_loso = _loso_predictions(
        internal_continuous,
        target="continuous",
    )
    binary_loso = _loso_predictions(internal_binary, target="binary")
    loso = pd.concat([continuous_loso, binary_loso], ignore_index=True)
    repeated = pd.concat(
        [
            _repeated_group_splits(internal_continuous, target="continuous"),
            _repeated_group_splits(internal_binary, target="binary"),
        ],
        ignore_index=True,
        sort=False,
    )
    repeated_summary = _summarize_repeated(repeated)
    continuous_null, continuous_summary = _permutation_null(
        internal_continuous,
        target="continuous",
    )
    binary_null, binary_summary = _permutation_null(
        internal_binary,
        target="binary",
    )
    permutation = pd.concat(
        [continuous_null, binary_null],
        ignore_index=True,
        sort=False,
    )
    permutation_summary = pd.DataFrame(
        [
            {"analysis": "continuous", **continuous_summary},
            {"analysis": "binary", **binary_summary},
        ]
    )
    similarity = _similarity_strata(loso)
    report = _report(permutation_summary, repeated_summary, similarity)

    atomic_write_parquet(output / "selected_anchor_loso_predictions.parquet", loso)
    atomic_write_csv(output / "repeated_group_split_metrics.csv", repeated)
    atomic_write_csv(output / "repeated_group_split_summary.csv", repeated_summary)
    atomic_write_csv(output / "permutation_null_metrics.csv", permutation)
    atomic_write_csv(output / "permutation_test_summary.csv", permutation_summary)
    atomic_write_csv(output / "similarity_strata.csv", similarity)
    atomic_write_text(output / "selected_anchor_robustness.md", report)
    _figures(permutation, permutation_summary, output)

    payload = {
        "status": "completed",
        "continuous_structures": len(internal_continuous),
        "binary_structures": len(internal_binary),
        "scaffolds": int(internal_continuous["scaffold"].nunique()),
        "permutations_per_anchor": PERMUTATIONS,
        "repeated_group_splits_requested": REPEATED_SPLITS,
        "binary_repeated_splits_completed": int(
            (repeated["analysis"].eq("binary") & repeated["status"].eq("completed")).sum()
        ),
        "binary_repeated_splits_excluded_single_class": int(
            (repeated["analysis"].eq("binary") & ~repeated["status"].eq("completed")).sum()
        ),
        "continuous_mae_permutation_p": continuous_summary["mae_permutation_p"],
        "binary_brier_permutation_p": binary_summary["brier_permutation_p"],
        "claim_boundary": (
            "internal robustness control after model selection; not independent prospective evidence"
        ),
    }
    atomic_write_json(output / "robustness_run_summary.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2))


if __name__ == "__main__":
    main()
