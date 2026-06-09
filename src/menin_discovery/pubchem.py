"""PubChem BioAssay collection helpers."""

from __future__ import annotations

import json
import time
from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from .config import PUBCHEM_EUTILS_BASE_URL, PUBCHEM_PUG_BASE_URL, PUBCHEM_SEARCH_TERMS

ENDPOINT_COLUMNS = ("IC50", "Ki", "Kd", "EC50")


def _get_json(url: str, params: dict[str, Any], timeout: int = 60) -> dict[str, Any]:
    response = requests.get(url, params=params, timeout=timeout)
    response.raise_for_status()
    return response.json()


def search_pcassay_ids(term: str, retmax: int = 250) -> list[int]:
    data = _get_json(
        f"{PUBCHEM_EUTILS_BASE_URL}/esearch.fcgi",
        {"db": "pcassay", "term": term, "retmode": "json", "retmax": retmax},
    )
    return [int(aid) for aid in data.get("esearchresult", {}).get("idlist", [])]


def summarize_pcassays(aids: list[int], chunk_size: int = 100) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for i in range(0, len(aids), chunk_size):
        chunk = aids[i : i + chunk_size]
        data = _get_json(
            f"{PUBCHEM_EUTILS_BASE_URL}/esummary.fcgi",
            {"db": "pcassay", "id": ",".join(map(str, chunk)), "retmode": "json"},
        )
        result = data.get("result", {})
        for aid in result.get("uids", []):
            item = result.get(str(aid), {})
            records.append(
                {
                    "aid": int(aid),
                    "assay_name": item.get("assayname", ""),
                    "assay_source_id": item.get("assaysourceid", ""),
                    "current_source_name": item.get("currentsourcename", ""),
                    "source_names": ";".join(item.get("sourcenamelist", []) or []),
                    "activity_outcome_method": item.get("activityoutcomemethod", ""),
                    "active_sid_count": item.get("activesidcount", ""),
                    "total_sid_count": item.get("totalsidcount", ""),
                    "target_count": item.get("targetcount", ""),
                    "deposit_date": item.get("depositdate", ""),
                    "modify_date": item.get("modifydate", ""),
                    "assay_description": item.get("assaydescription", ""),
                }
            )
    return pd.DataFrame(records).drop_duplicates(subset=["aid"])


def download_assay_description(aid: int, output_path: Path) -> bool:
    url = f"{PUBCHEM_PUG_BASE_URL}/assay/aid/{aid}/description/JSON"
    response = requests.get(url, timeout=60)
    if response.status_code != 200:
        return False
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(response.json(), indent=2), encoding="utf-8")
    return True


def download_assay_csv(aid: int, output_path: Path) -> bool:
    url = f"{PUBCHEM_PUG_BASE_URL}/assay/aid/{aid}/CSV"
    response = requests.get(url, timeout=90)
    if response.status_code != 200 or not response.text.startswith("PUBCHEM_"):
        return False
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(response.text, encoding="utf-8")
    return True


def collect_pubchem_assays(
    raw_dir: Path,
    *,
    search_terms: tuple[str, ...] = PUBCHEM_SEARCH_TERMS,
    retmax_per_term: int = 250,
    max_aids: int | None = None,
    sleep_seconds: float = 0.12,
) -> pd.DataFrame:
    """Search, summarize, and download available PubChem MEN1/menin assay exports."""

    raw_dir.mkdir(parents=True, exist_ok=True)
    all_aids: list[int] = []
    search_rows: list[dict[str, Any]] = []

    for term in search_terms:
        aids = search_pcassay_ids(term, retmax=retmax_per_term)
        search_rows.append({"term": term, "n_aids": len(aids), "aids": ";".join(map(str, aids))})
        all_aids.extend(aids)
        time.sleep(sleep_seconds)

    deduped = sorted(set(all_aids), reverse=True)
    if max_aids is not None:
        deduped = deduped[:max_aids]

    catalog = summarize_pcassays(deduped)
    search_df = pd.DataFrame(search_rows)
    search_df.to_csv(raw_dir / "pubchem_search_terms.csv", index=False)
    catalog.to_csv(raw_dir / "pubchem_assay_catalog.csv", index=False)

    desc_dir = raw_dir / "assay_descriptions"
    data_dir = raw_dir / "assay_data"
    download_status: list[dict[str, Any]] = []
    for aid in deduped:
        has_description = download_assay_description(aid, desc_dir / f"AID_{aid}.json")
        time.sleep(sleep_seconds)
        has_data = download_assay_csv(aid, data_dir / f"AID_{aid}.csv")
        download_status.append(
            {"aid": aid, "description_downloaded": has_description, "csv_downloaded": has_data}
        )
        time.sleep(sleep_seconds)

    status_df = pd.DataFrame(download_status)
    status_df.to_csv(raw_dir / "pubchem_download_status.csv", index=False)
    return catalog.merge(status_df, on="aid", how="left")


def load_pubchem_assay_csvs(raw_dir: Path) -> pd.DataFrame:
    data_dir = raw_dir / "assay_data"
    frames: list[pd.DataFrame] = []
    for path in sorted(data_dir.glob("AID_*.csv")):
        aid = int(path.stem.replace("AID_", ""))
        text = path.read_text(encoding="utf-8")
        df = pd.read_csv(StringIO(text), low_memory=False)
        first_col = df.columns[0]
        unit_rows = df[df[first_col].astype(str).eq("RESULT_UNIT")]
        unit_row = unit_rows.iloc[0] if not unit_rows.empty else pd.Series(dtype=object)
        df = df[~df[first_col].astype(str).str.startswith("RESULT_")].copy()
        for endpoint in ENDPOINT_COLUMNS:
            if endpoint in df.columns:
                df[f"{endpoint} Units"] = unit_row.get(endpoint, "")
        if "PubChem Standard Value" in df.columns:
            df["PubChem Standard Value Units"] = unit_row.get("PubChem Standard Value", "MICROMOLAR")
        df["aid"] = aid
        frames.append(df)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)
