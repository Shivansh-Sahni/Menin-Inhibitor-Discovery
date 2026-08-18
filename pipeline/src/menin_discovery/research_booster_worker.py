"""Isolated XGBoost/LightGBM worker for macOS OpenMP runtime safety."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .research_common import atomic_write_csv, atomic_write_parquet
from .research_modeling import grouped_exact_herg_benchmark, grouped_regression_benchmark


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--mode", choices=["pk", "herg"], default="pk")
    parser.add_argument("--features", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--group", default="scaffold")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--models", default="xgboost,lightgbm")
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    args = parser.parse_args(argv)
    frame = pd.read_parquet(args.input)
    kwargs = {
        "feature_columns": [value for value in args.features.split(",") if value],
        "target_column": args.target,
        "group_column": args.group,
        "folds": args.folds,
        "include_optional_boosters": True,
        "model_names": [value for value in args.models.split(",") if value],
    }
    if args.mode == "pk":
        metrics, predictions = grouped_regression_benchmark(frame, **kwargs)
    else:
        metrics, predictions = grouped_exact_herg_benchmark(frame, **kwargs)
    atomic_write_csv(args.metrics, metrics)
    atomic_write_parquet(args.predictions, predictions)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
