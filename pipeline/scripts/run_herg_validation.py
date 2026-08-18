#!/usr/bin/env python3
"""Run nested confidential-scaffold validation for hERG liability models."""

from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", ".matplotlib_cache")
os.environ.setdefault("XDG_CACHE_HOME", ".cache")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "8")

from menin_discovery.herg_validation import run_nested_validation

warnings.filterwarnings(
    "ignore",
    message="The `probability` parameter was deprecated",
    category=FutureWarning,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", type=Path, required=True)
    parser.add_argument(
        "--public-data",
        type=Path,
        default=Path("research/data/processed/herg_compounds_curated.csv"),
    )
    parser.add_argument(
        "--benchmark-config",
        type=Path,
        default=Path("pipeline/config/herg_benchmark.yaml"),
    )
    parser.add_argument(
        "--validation-config",
        type=Path,
        default=Path("pipeline/config/herg_validation.yaml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("research/benchmarks/herg/nested_scaffold"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_nested_validation(
        workbook_path=args.workbook.resolve(),
        public_path=args.public_data.resolve(),
        benchmark_config_path=args.benchmark_config.resolve(),
        validation_config_path=args.validation_config.resolve(),
        output_dir=args.output.resolve(),
    )


if __name__ == "__main__":
    main()
