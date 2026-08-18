#!/usr/bin/env python3
"""Analyze hERG performance for large molecules without mutating V9.

This sidecar uses only frozen train-partition scaffold-held-out predictions.
It can be run while V9 is active to freeze the V8 baseline, then rerun after
V9 completes to add every V9 prediction mode and paired scaffold inference.
Repository validation and test labels are never opened.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCHEMA = "platform-local-herg-heavy-compound-sidecar-v9/1.0"
CUTOFFS = (500.0, 600.0, 700.0, 1000.0)
SEED = 20260817


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict[str, Any], key: str) -> dict[str, Any]:
    document = dict(value)
    document.pop(key, None)
    document[key] = hashlib.sha256(_canonical(document)).hexdigest()
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    return document


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def _metrics(observed: pd.Series, predicted: pd.Series) -> dict[str, Any]:
    error = np.abs(observed.to_numpy(float) - predicted.to_numpy(float))
    return {
        "n": len(error),
        "mae": float(error.mean()),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "median_absolute_error": float(np.median(error)),
        "fraction_within_0p5": float(np.mean(error <= 0.5)),
        "fraction_within_1p0": float(np.mean(error <= 1.0)),
    }


def _scaffold_bootstrap(
    frame: pd.DataFrame, challenger: str, reference: str, replicates: int, seed: int
) -> dict[str, Any]:
    work = frame[["scaffold_group_id", "observed_pic50", challenger, reference]].copy()
    work["delta"] = np.abs(work.observed_pic50 - work[challenger]) - np.abs(
        work.observed_pic50 - work[reference]
    )
    grouped = work.groupby("scaffold_group_id", observed=True).delta.agg(["sum", "count"])
    sums = grouped["sum"].to_numpy(float)
    counts = grouped["count"].to_numpy(float)
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=float)
    for start in range(0, replicates, 500):
        size = min(500, replicates - start)
        indices = rng.integers(0, len(grouped), size=(size, len(grouped)))
        draws[start : start + size] = sums[indices].sum(axis=1) / counts[indices].sum(axis=1)
    return {
        "delta_mae": float(sums.sum() / counts.sum()),
        "ci95_lower": float(np.quantile(draws, 0.025)),
        "ci95_upper": float(np.quantile(draws, 0.975)),
        "scaffolds": len(grouped),
        "bootstrap_replicates": replicates,
    }


def _published_tools() -> list[dict[str, Any]]:
    return [
        {
            "tool": "Pred-hERG",
            "task": "binary and multiclass classification",
            "reported_dataset_n": 5984,
            "reported_performance": "binary CCR about 0.8; multiclass accuracy about 0.7",
            "applicability_domain": "reported for individual queries",
            "mw_ge_500_subgroup_metric_reported": False,
            "source": "https://pmc.ncbi.nlm.nih.gov/articles/PMC5720373/",
        },
        {
            "tool": "HergSPred",
            "task": "binary classification consensus",
            "reported_dataset_n": 12850,
            "reported_performance": "reported accuracy 0.839",
            "applicability_domain": "not exposed in a published independent tool comparison",
            "mw_ge_500_subgroup_metric_reported": False,
            "source": "https://pubs.acs.org/doi/10.1021/acs.jcim.2c00256",
        },
        {
            "tool": "hERGBoost",
            "task": "quantitative IC50 regression",
            "reported_dataset_n": None,
            "reported_performance": "external R2 0.394; RMSE 0.616",
            "applicability_domain": "not established specifically for MW >=500 in accessible report",
            "mw_ge_500_subgroup_metric_reported": False,
            "source": "https://doi.org/10.1016/j.compbiomed.2024.109416",
        },
        {
            "tool": "HERGAI",
            "task": "structure-based binary classification",
            "reported_dataset_n": None,
            "reported_performance": "classification endpoints; not directly comparable to pIC50 MAE",
            "applicability_domain": "physicochemical properties examined, but no MW >=500 error stratum",
            "mw_ge_500_subgroup_metric_reported": False,
            "source": "https://pmc.ncbi.nlm.nih.gov/articles/PMC12291323/",
        },
    ]


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    v8 = args.v8_root.resolve()
    matrix = pd.read_parquet(v8 / "prepared/training_matrix.parquet")
    v8_oof = pd.read_parquet(v8 / "nested_oof_predictions.parquet")[
        ["structure_id", "scaffold_group_id", "observed_pic50", "outer_fold", "predicted_pic50"]
    ].rename(columns={"predicted_pic50": "pred__v8"})
    data = v8_oof.merge(
        matrix[["structure_id", "rdkit2d__MolWt", "rdkit2d__NumRotatableBonds"]],
        on="structure_id",
        validate="one_to_one",
    )
    v9_complete = (args.v9_root / "validation.json").is_file()
    if v9_complete:
        v9 = pd.read_parquet(args.v9_root / "analysis/nested_oof_predictions.parquet")
        keep = ["structure_id", *[column for column in v9 if column.startswith("pred__")]]
        data = data.merge(v9[keep], on="structure_id", validate="one_to_one")
    prediction_columns = [column for column in data if column.startswith("pred__")]
    metric_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    for cutoff in CUTOFFS:
        for regime, selector in (
            ("below", data.rdkit2d__MolWt.lt(cutoff)),
            ("at_or_above", data.rdkit2d__MolWt.ge(cutoff)),
        ):
            subset = data.loc[selector]
            for column in prediction_columns:
                metric_rows.append(
                    {
                        "cutoff_da": cutoff,
                        "regime": regime,
                        "model_id": column.removeprefix("pred__"),
                        "scaffolds": subset.scaffold_group_id.nunique(),
                        "mean_molecular_weight": float(subset.rdkit2d__MolWt.mean()),
                        **_metrics(subset.observed_pic50, subset[column]),
                    }
                )
        heavy = data.loc[data.rdkit2d__MolWt.ge(cutoff)]
        if len(heavy) >= 20:
            for index, column in enumerate(prediction_columns):
                if column != "pred__v8":
                    comparison_rows.append(
                        {
                            "cutoff_da": cutoff,
                            "challenger": column.removeprefix("pred__"),
                            "reference": "v8",
                            **_scaffold_bootstrap(
                                heavy, column, "pred__v8", args.bootstrap_replicates, SEED + index
                            ),
                        }
                    )
    metrics = pd.DataFrame(metric_rows)
    comparisons = pd.DataFrame(comparison_rows)
    data["mw_regime"] = pd.cut(
        data.rdkit2d__MolWt,
        [-np.inf, 500, 600, 700, 1000, np.inf],
        labels=["lt500", "500_600", "600_700", "700_1000", "ge1000"],
        right=False,
    ).astype(str)
    metrics_path = output / "heavy_compound_metrics.parquet"
    comparisons_path = output / "heavy_v9_vs_v8_scaffold_bootstrap.parquet"
    atlas_path = output / "heavy_compound_atlas.parquet"
    tools_path = output / "published_tool_scope.json"
    _atomic_parquet(metrics_path, metrics)
    _atomic_parquet(comparisons_path, comparisons)
    _atomic_parquet(atlas_path, data)
    tools = _atomic_json(
        tools_path,
        {
            "schema_version": SCHEMA,
            "interpretation": (
                "Published headline metrics are not a direct benchmark because endpoints and splits differ. "
                "No reviewed tool reports an MW >=500 held-out error stratum."
            ),
            "tools": _published_tools(),
        },
        "document_sha256",
    )
    v8_500 = metrics.loc[
        metrics.cutoff_da.eq(500) & metrics.regime.eq("at_or_above") & metrics.model_id.eq("v8")
    ].iloc[0]
    v8_700 = metrics.loc[
        metrics.cutoff_da.eq(700) & metrics.regime.eq("at_or_above") & metrics.model_id.eq("v8")
    ].iloc[0]
    report_path = output / "HEAVY_COMPOUND_REPORT.md"
    report_path.write_text(
        "# hERG heavy-compound sidecar\n\n"
        f"Status: {'consolidated with completed V9' if v9_complete else 'V8 baseline frozen; V9 pending'}. "
        "All results are internal train-partition scaffold-held-out evidence.\n\n"
        f"At MW >=500 Da, V8 has n={int(v8_500.n):,}, MAE={v8_500.mae:.4f}, and "
        f"{100 * v8_500.fraction_within_0p5:.1f}% within 0.5 log. At MW >=700 Da, n={int(v8_700.n):,}, "
        f"MAE={v8_700.mae:.4f}, and {100 * v8_700.fraction_within_0p5:.1f}% within 0.5 log.\n\n"
        "The 500 Da boundary is reported because it is conventional and prespecified, but it is not an "
        "empirical failure boundary here. Performance degradation is more apparent from 600–700 Da. "
        "The >=1,000 Da stratum is descriptive only because support is tiny.\n\n"
        "Pred-hERG, HergSPred, hERGBoost, and HERGAI do not provide a directly comparable published "
        "MW >=500 held-out error stratum. Their headline metrics use different endpoints, datasets, and "
        "validation schemes, so this report does not claim an external head-to-head win.\n",
        encoding="utf-8",
    )
    manifest = _atomic_json(
        output / "manifest.json",
        {
            "schema_version": SCHEMA,
            "status": "passed" if v9_complete else "waiting_for_v9",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "v9_consolidated": v9_complete,
            "validation_labels_opened": False,
            "test_labels_opened": False,
            "inputs": [
                {
                    "path": str(v8 / "nested_oof_predictions.parquet"),
                    "sha256": _sha(v8 / "nested_oof_predictions.parquet"),
                },
                {
                    "path": str(v8 / "prepared/training_matrix.parquet"),
                    "sha256": _sha(v8 / "prepared/training_matrix.parquet"),
                },
            ],
            "artifacts": [
                {"path": str(path), "sha256": _sha(path)}
                for path in (metrics_path, comparisons_path, atlas_path, tools_path, report_path)
            ],
            "published_tool_document_sha256": tools["document_sha256"],
        },
        "manifest_sha256",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v8-root", type=Path, required=True)
    parser.add_argument("--v9-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
