#!/usr/bin/env python3
"""Run the configurable private/public hERG liability benchmark."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# LightGBM/XGBoost/PyTorch wheels can bundle different OpenMP runtimes on macOS.
# One OpenMP worker avoids a native barrier deadlock; estimator-level process
# parallelism remains available to the classical models.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from menin_discovery.herg_benchmark import run_benchmark


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workbook", type=Path, required=True, help="Lab workbook containing the SMILES sheet."
    )
    parser.add_argument(
        "--public-data",
        type=Path,
        default=Path("research/data/processed/herg_compounds_curated.csv"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("pipeline/config/herg_benchmark.yaml"),
    )
    parser.add_argument(
        "--profile",
        choices=("smoke", "quick", "strong_ml", "full"),
        default="quick",
    )
    parser.add_argument("--output", type=Path, default=Path("research/benchmarks/herg"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_benchmark(
        workbook_path=args.workbook.resolve(),
        public_path=args.public_data.resolve(),
        config_path=args.config.resolve(),
        output_dir=args.output.resolve(),
        profile=args.profile,
    )


if __name__ == "__main__":
    main()
