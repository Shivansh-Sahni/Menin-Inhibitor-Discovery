"""BindingDB download and normalization helpers."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import requests

from .config import BINDINGDB_SOURCES, BIOACTIVITY_ENDPOINTS


def _download_text(url: str, timeout: int = 60) -> str:
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    text = response.text

    if text.lstrip().lower().startswith("<!doctype") or "<html" in text[:500].lower():
        match = re.search(r'href="([^"]+\.tsv)"', text)
        if not match:
            raise ValueError(f"BindingDB returned HTML and no TSV link was found for {url}")
        next_url = match.group(1)
        if next_url.startswith("/"):
            next_url = "https://www.bindingdb.org" + next_url
        response = requests.get(next_url, timeout=timeout)
        response.raise_for_status()
        text = response.text

    return text


def download_bindingdb_sources(raw_dir: Path) -> list[Path]:
    """Download configured Menin BindingDB TSVs."""

    raw_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for source in BINDINGDB_SOURCES:
        text = _download_text(source.url)
        path = raw_dir / f"{source.name}.tsv"
        path.write_text(text, encoding="utf-8")
        paths.append(path)
    return paths


def load_bindingdb_tsvs(raw_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for source in BINDINGDB_SOURCES:
        path = raw_dir / f"{source.name}.tsv"
        if not path.exists():
            continue
        df = pd.read_csv(path, sep="\t", low_memory=False)
        df["bindingdb_source_file"] = path.name
        df["bindingdb_target_hint"] = source.target_hint
        frames.append(df)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def bindingdb_to_long(df: pd.DataFrame) -> pd.DataFrame:
    """Convert BindingDB wide Ki/IC50/Kd/EC50 columns to the common long schema."""

    if df.empty:
        return pd.DataFrame()

    endpoint_columns = {
        endpoint: f"{endpoint} (nM)"
        for endpoint in BIOACTIVITY_ENDPOINTS
        if f"{endpoint} (nM)" in df.columns
    }
    rows: list[pd.DataFrame] = []
    for endpoint, column in endpoint_columns.items():
        part = df.copy()
        part["endpoint"] = endpoint
        part["value_raw"] = part[column]
        rows.append(part)

    if not rows:
        return pd.DataFrame()

    long = pd.concat(rows, ignore_index=True, sort=False)
    long = long[long["value_raw"].notna() & (long["value_raw"].astype(str).str.strip() != "")]

    return pd.DataFrame(
        {
            "source": "BindingDB",
            "source_record_id": long.get("BindingDB Reactant_set_id"),
            "compound_id": long.get("BindingDB MonomerID"),
            "compound_name": long.get("BindingDB Ligand Name"),
            "smiles": long.get("Ligand SMILES"),
            "inchi_key": long.get("Ligand InChI Key"),
            "target_name": long.get("Target Name"),
            "target_id": long.get("UniProt (SwissProt) Primary ID of Target Chain"),
            "endpoint": long["endpoint"],
            "relation": long["value_raw"].astype(str).str.extract(r"^\s*([<>=~]+)", expand=False).fillna("="),
            "value_raw": long["value_raw"],
            "standard_units": "nM",
            "assay_description": "",
            "assay_type": "",
            "document_id": long.get("Article DOI").fillna(long.get("Patent Number")),
            "document_year": long.get("Date of publication"),
            "reference": long.get("Link to Ligand-Target Pair in BindingDB"),
            "source_detail": long.get("bindingdb_source_file"),
        }
    )
