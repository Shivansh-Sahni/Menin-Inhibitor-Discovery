"""ChEMBL REST API helpers."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from .config import CHEMBL_BASE_URL


def _get_json(url: str, params: dict[str, Any] | None = None, timeout: int = 60) -> dict[str, Any]:
    response = requests.get(url, params=params, timeout=timeout)
    response.raise_for_status()
    return response.json()


def fetch_target_search(query: str, output_path: Path | None = None) -> list[dict[str, Any]]:
    """Search ChEMBL targets by free text."""

    url = f"{CHEMBL_BASE_URL}/target/search.json"
    data = _get_json(url, {"q": query})
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data.get("targets", [])


def fetch_paged_records(
    endpoint: str,
    root_key: str,
    params: dict[str, Any],
    *,
    page_size: int = 1000,
    sleep_seconds: float = 0.05,
    max_records: int | None = None,
) -> list[dict[str, Any]]:
    """Fetch paged ChEMBL records from an endpoint such as ``activity``."""

    url = f"{CHEMBL_BASE_URL}/{endpoint}.json"
    records: list[dict[str, Any]] = []
    offset = 0
    total_count: int | None = None

    while True:
        page_params = dict(params)
        page_params.update({"limit": page_size, "offset": offset})
        data = _get_json(url, page_params)
        page_records = data.get(root_key, [])
        records.extend(page_records)
        page_meta = data.get("page_meta", {})
        total_count = page_meta.get("total_count", total_count)

        if max_records is not None and len(records) >= max_records:
            return records[:max_records]
        if not page_records:
            break
        offset += page_size
        if total_count is not None and offset >= int(total_count):
            break
        time.sleep(sleep_seconds)

    return records


def fetch_target_activities(
    target_chembl_id: str,
    output_csv: Path,
    *,
    page_size: int = 1000,
    max_records: int | None = None,
) -> pd.DataFrame:
    """Fetch all ChEMBL activity rows for one target and save them as CSV."""

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if output_csv.exists():
        output_csv.unlink()

    url = f"{CHEMBL_BASE_URL}/activity.json"
    offset = 0
    total_count: int | None = None
    written = 0
    first_page = True
    while True:
        limit = page_size
        if max_records is not None:
            remaining = max_records - written
            if remaining <= 0:
                break
            limit = min(limit, remaining)

        data = _get_json(
            url,
            {"target_chembl_id": target_chembl_id, "limit": limit, "offset": offset},
        )
        records = data.get("activities", [])
        page_meta = data.get("page_meta", {})
        total_count = page_meta.get("total_count", total_count)
        if not records:
            break

        page = pd.DataFrame(records)
        page.to_csv(output_csv, index=False, mode="w" if first_page else "a", header=first_page)
        first_page = False
        written += len(page)
        offset += limit
        planned_total = min(total_count or written, max_records or total_count or written)
        print(f"ChEMBL {target_chembl_id}: wrote {written:,}/{planned_total:,} activity rows", flush=True)

        if total_count is not None and offset >= int(total_count):
            break
        time.sleep(0.05)

    return pd.read_csv(output_csv, low_memory=False) if output_csv.exists() else pd.DataFrame()


def fetch_molecule_activities(
    molecule_chembl_ids: list[str],
    output_csv: Path,
    *,
    chunk_size: int = 50,
    page_size: int = 1000,
    sleep_seconds: float = 0.08,
    max_chunks: int | None = None,
) -> pd.DataFrame:
    """Fetch ChEMBL activity rows for a list of molecules using ``__in`` chunks."""

    chunks = [
        molecule_chembl_ids[i : i + chunk_size]
        for i in range(0, len(molecule_chembl_ids), chunk_size)
    ]
    if max_chunks is not None:
        chunks = chunks[:max_chunks]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if output_csv.exists():
        output_csv.unlink()

    first_page = True
    written = 0
    for chunk_index, chunk in enumerate(chunks, start=1):
        records = fetch_paged_records(
            "activity",
            "activities",
            {"molecule_chembl_id__in": ",".join(chunk)},
            page_size=page_size,
            sleep_seconds=sleep_seconds,
        )
        if records:
            page = pd.DataFrame(records)
            page.to_csv(output_csv, index=False, mode="w" if first_page else "a", header=first_page)
            first_page = False
            written += len(page)
        print(
            f"ChEMBL molecule activities: chunk {chunk_index:,}/{len(chunks):,}, wrote {written:,} rows",
            flush=True,
        )
        time.sleep(sleep_seconds)

    return pd.read_csv(output_csv, low_memory=False) if output_csv.exists() else pd.DataFrame()
