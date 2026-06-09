#!/usr/bin/env python3
"""Run public data collection, curation, modeling, and reporting."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib_cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from menin_discovery.bindingdb import download_bindingdb_sources, load_bindingdb_tsvs
from menin_discovery.chembl import (
    fetch_molecule_activities,
    fetch_target_activities,
    fetch_target_search,
)
from menin_discovery.config import HERG_TARGET, MENIN_TARGET
from menin_discovery.curation import write_processed_tables
from menin_discovery.modeling import run_models
from menin_discovery.pubchem import collect_pubchem_assays, load_pubchem_assay_csvs


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False) if path.exists() else pd.DataFrame()


def collect_data(args: argparse.Namespace) -> None:
    raw = ROOT / "data" / "raw"
    chembl_raw = raw / "chembl"
    bindingdb_raw = raw / "bindingdb"
    pubchem_raw = raw / "pubchem"

    if not args.skip_network:
        fetch_target_search("menin", chembl_raw / "chembl_target_search_menin.json")
        fetch_target_search("KCNH2 hERG", chembl_raw / "chembl_target_search_herg.json")
        fetch_target_activities(
            MENIN_TARGET["chembl_id"],
            chembl_raw / "chembl_menin_activities.csv",
            max_records=args.max_menin_records,
        )
        fetch_target_activities(
            HERG_TARGET["chembl_id"],
            chembl_raw / "chembl_herg_activities.csv",
            max_records=args.max_herg_records,
        )
        download_bindingdb_sources(bindingdb_raw)
        collect_pubchem_assays(
            pubchem_raw,
            max_aids=args.max_pubchem_aids,
            retmax_per_term=args.pubchem_retmax_per_term,
        )

        if not args.skip_pk:
            menin = read_csv_if_exists(chembl_raw / "chembl_menin_activities.csv")
            molecule_ids = (
                menin.get("molecule_chembl_id", pd.Series(dtype=str))
                .dropna()
                .astype(str)
                .drop_duplicates()
                .tolist()
            )
            fetch_molecule_activities(
                molecule_ids,
                chembl_raw / "chembl_menin_molecule_all_activities.csv",
                max_chunks=args.max_pk_chunks,
            )

    pubchem_catalog = read_csv_if_exists(pubchem_raw / "pubchem_assay_catalog.csv")
    tables = write_processed_tables(
        chembl_menin_raw=read_csv_if_exists(chembl_raw / "chembl_menin_activities.csv"),
        bindingdb_raw=load_bindingdb_tsvs(bindingdb_raw),
        pubchem_raw=load_pubchem_assay_csvs(pubchem_raw),
        pubchem_catalog=pubchem_catalog,
        herg_raw=read_csv_if_exists(chembl_raw / "chembl_herg_activities.csv"),
        pk_raw=read_csv_if_exists(chembl_raw / "chembl_menin_molecule_all_activities.csv"),
        processed_dir=ROOT / "data" / "processed",
    )
    for name, table in tables.items():
        print(f"{name}: {len(table):,} rows")


def run_model_stage() -> None:
    metrics = run_models(ROOT / "data" / "processed", ROOT / "models", ROOT / "reports")
    print(metrics)


def run_report_stage() -> None:
    from menin_discovery.reporting import write_summary_report

    path = write_summary_report(ROOT / "data" / "processed", ROOT / "reports", ROOT / "models")
    print(f"Wrote {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=["all", "data", "models", "report"],
        default="all",
        help="Pipeline stage to run.",
    )
    parser.add_argument("--skip-network", action="store_true", help="Reuse existing raw files only.")
    parser.add_argument("--skip-pk", action="store_true", help="Skip molecule-level ChEMBL PK/ADMET collection.")
    parser.add_argument("--max-menin-records", type=int, default=None)
    parser.add_argument("--max-herg-records", type=int, default=None)
    parser.add_argument("--max-pubchem-aids", type=int, default=250)
    parser.add_argument("--pubchem-retmax-per-term", type=int, default=250)
    parser.add_argument("--max-pk-chunks", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stage in {"all", "data"}:
        collect_data(args)
    if args.stage in {"all", "models"}:
        run_model_stage()
    if args.stage in {"all", "report"}:
        run_report_stage()


if __name__ == "__main__":
    main()
