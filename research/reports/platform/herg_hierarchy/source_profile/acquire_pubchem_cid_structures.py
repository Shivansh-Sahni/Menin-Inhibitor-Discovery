#!/usr/bin/env python3
"""Acquire PubChem structures for every CID in AID 720551.

The downloader is deliberately resumable and fail-closed. Each deterministic
CID batch has an immutable raw CSV response and a JSON receipt binding the
request and response by SHA-256. The final merged table is written only when
the returned CID set exactly equals the requested source CID set.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import random
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

SCRIPT = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT.parents[5]
DEFAULT_SOURCE = PROJECT_ROOT / "research/data/platform/raw/external_public/herg_expansion/avicenna/assays/AID_720551_concise.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "research/data/platform/raw/external_public/herg_hierarchy/source_profile/pubchem_aid720551_structures"
ENDPOINT = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/property/IsomericSMILES,CanonicalSMILES,InChIKey/CSV"
USER_AGENT = "Menin-hERG-research/1.0 (official PubChem structure acquisition)"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    except BaseException:
        try:
            os.unlink(name)
        except FileNotFoundError:
            pass
        raise


def json_bytes(obj: object) -> bytes:
    return (json.dumps(obj, indent=2, sort_keys=True) + "\n").encode("utf-8")


def source_cids(path: Path) -> tuple[list[int], int]:
    seen: set[int] = set()
    rows = 0
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if "CID" not in (reader.fieldnames or []):
            raise RuntimeError(f"CID column absent from {path}")
        for row in reader:
            rows += 1
            value = (row.get("CID") or "").strip()
            if not value or not value.isdigit() or int(value) <= 0:
                raise RuntimeError(f"invalid CID at data row {rows}: {value!r}")
            seen.add(int(value))
    return sorted(seen), rows


def parse_response(data: bytes) -> dict[int, dict[str, str]]:
    text = data.decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(text))
    names = reader.fieldnames or []
    if "CID" not in names or "InChIKey" not in names:
        raise RuntimeError(f"unexpected PubChem fields: {names}")

    # PubChem's current names are SMILES (isomeric) and ConnectivitySMILES
    # (canonical connectivity). Accept historical names without weakening the
    # normalized output contract.
    smiles_key = next((x for x in ("SMILES", "IsomericSMILES") if x in names), None)
    connectivity_key = next((x for x in ("ConnectivitySMILES", "CanonicalSMILES") if x in names), None)
    if not smiles_key or not connectivity_key:
        raise RuntimeError(f"SMILES property fields absent: {names}")

    result: dict[int, dict[str, str]] = {}
    for row_number, row in enumerate(reader, start=2):
        raw_cid = (row.get("CID") or "").strip()
        if not raw_cid.isdigit():
            raise RuntimeError(f"invalid returned CID at CSV row {row_number}: {raw_cid!r}")
        cid = int(raw_cid)
        if cid in result:
            raise RuntimeError(f"duplicate returned CID: {cid}")
        normalized = {
            "CID": str(cid),
            "SMILES": (row.get(smiles_key) or "").strip(),
            "ConnectivitySMILES": (row.get(connectivity_key) or "").strip(),
            "InChIKey": (row.get("InChIKey") or "").strip(),
        }
        if not all(normalized[k] for k in ("SMILES", "ConnectivitySMILES", "InChIKey")):
            raise RuntimeError(f"blank required structure property for CID {cid}")
        result[cid] = normalized
    return result


def valid_existing(batch_path: Path, receipt_path: Path, expected: list[int]) -> bool:
    if not batch_path.is_file() or not receipt_path.is_file():
        return False
    try:
        receipt = json.loads(receipt_path.read_text())
        raw = batch_path.read_bytes()
        parsed = parse_response(raw)
    except Exception:
        return False
    return (
        receipt.get("status") == "complete"
        and receipt.get("requested_cids") == expected
        and receipt.get("requested_count") == len(expected)
        and receipt.get("response_sha256") == sha256_bytes(raw)
        and receipt.get("returned_cids") == sorted(parsed)
        and sorted(parsed) == expected
    )


def fetch_batch(cids: list[int], attempts: int, timeout: float) -> tuple[bytes, int, str]:
    headers = {
        "Accept": "text/csv",
        "Content-Type": "application/x-www-form-urlencoded",
        "User-Agent": USER_AGENT,
    }
    last_error = ""
    for attempt in range(1, attempts + 1):
        response: requests.Response | None = None
        try:
            response = requests.post(
                ENDPOINT,
                data={"cid": ",".join(map(str, cids))},
                headers=headers,
                timeout=timeout,
            )
            response.raise_for_status()
            data = response.content
            parsed = parse_response(data)
            if sorted(parsed) != cids:
                missing = sorted(set(cids) - set(parsed))
                extra = sorted(set(parsed) - set(cids))
                raise RuntimeError(f"coverage mismatch missing={missing[:10]} extra={extra[:10]}")
            return data, attempt, ""
        except (requests.RequestException, TimeoutError, RuntimeError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt == attempts:
                break
            retry_after = response.headers.get("Retry-After") if response is not None else None
            if retry_after and retry_after.isdigit():
                delay = float(retry_after)
            else:
                delay = min(60.0, 1.5 * (2 ** (attempt - 1))) + random.Random(cids[0] + attempt).uniform(0, 0.5)
            time.sleep(delay)
    raise RuntimeError(f"batch beginning CID {cids[0]} failed after {attempts} attempts: {last_error}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--attempts", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--min-interval", type=float, default=0.22)
    args = parser.parse_args()
    if args.batch_size < 1 or args.batch_size > 1000:
        parser.error("--batch-size must be between 1 and 1000")

    source = args.source.resolve()
    output = args.output_dir.resolve()
    batches_dir = output / "batches"
    receipts_dir = output / "receipts"
    batches_dir.mkdir(parents=True, exist_ok=True)
    receipts_dir.mkdir(parents=True, exist_ok=True)

    cids, source_rows = source_cids(source)
    source_hash = sha256_file(source)
    chunks = [cids[i : i + args.batch_size] for i in range(0, len(cids), args.batch_size)]
    resumed = 0
    fetched = 0
    last_request = 0.0

    for index, chunk in enumerate(chunks):
        stem = f"batch_{index:06d}"
        batch_path = batches_dir / f"{stem}.csv"
        receipt_path = receipts_dir / f"{stem}.json"
        if valid_existing(batch_path, receipt_path, chunk):
            resumed += 1
            continue

        elapsed = time.monotonic() - last_request
        if elapsed < args.min_interval:
            time.sleep(args.min_interval - elapsed)
        started = utc_now()
        data, attempt_count, _ = fetch_batch(chunk, args.attempts, args.timeout)
        last_request = time.monotonic()
        parsed = parse_response(data)
        atomic_write(batch_path, data)
        receipt = {
            "schema_version": "pubchem-cid-structure-batch/1.0",
            "status": "complete",
            "endpoint": ENDPOINT,
            "http_method": "POST",
            "request_content_type": "application/x-www-form-urlencoded",
            "batch_index": index,
            "requested_count": len(chunk),
            "requested_cids": chunk,
            "returned_count": len(parsed),
            "returned_cids": sorted(parsed),
            "response_bytes": len(data),
            "response_sha256": sha256_bytes(data),
            "attempt_count": attempt_count,
            "started_utc": started,
            "completed_utc": utc_now(),
        }
        atomic_write(receipt_path, json_bytes(receipt))
        fetched += 1
        if fetched % 25 == 0 or index + 1 == len(chunks):
            print(f"completed {index + 1}/{len(chunks)} batches; fetched={fetched} resumed={resumed}", flush=True)

    merged: dict[int, dict[str, str]] = {}
    batch_bindings: list[dict[str, object]] = []
    for index, chunk in enumerate(chunks):
        stem = f"batch_{index:06d}"
        batch_path = batches_dir / f"{stem}.csv"
        receipt_path = receipts_dir / f"{stem}.json"
        if not valid_existing(batch_path, receipt_path, chunk):
            raise RuntimeError(f"invalid batch during final verification: {stem}")
        raw = batch_path.read_bytes()
        rows = parse_response(raw)
        overlap = set(merged).intersection(rows)
        if overlap:
            raise RuntimeError(f"CID duplicated across batches: {sorted(overlap)[:10]}")
        merged.update(rows)
        batch_bindings.append({
            "batch_index": index,
            "path": str(batch_path.relative_to(output)),
            "receipt_path": str(receipt_path.relative_to(output)),
            "requested_count": len(chunk),
            "response_bytes": len(raw),
            "response_sha256": sha256_bytes(raw),
        })

    returned = sorted(merged)
    if returned != cids:
        missing = sorted(set(cids) - set(returned))
        extra = sorted(set(returned) - set(cids))
        raise RuntimeError(f"FINAL CID COVERAGE FAILURE missing={missing[:25]} extra={extra[:25]}")

    out = io.StringIO(newline="")
    writer = csv.DictWriter(out, fieldnames=["CID", "SMILES", "ConnectivitySMILES", "InChIKey"], lineterminator="\n")
    writer.writeheader()
    for cid in cids:
        writer.writerow(merged[cid])
    merged_bytes = out.getvalue().encode("utf-8")
    merged_path = output / "aid720551_pubchem_structures.csv"
    atomic_write(merged_path, merged_bytes)

    manifest = {
        "schema_version": "pubchem-aid720551-structure-acquisition/1.0",
        "status": "complete_exact_coverage",
        "generated_utc": utc_now(),
        "source": {
            "path": str(source),
            "bytes": source.stat().st_size,
            "sha256": source_hash,
            "data_rows": source_rows,
            "unique_cids": len(cids),
        },
        "request": {
            "endpoint": ENDPOINT,
            "http_method": "POST",
            "properties": ["IsomericSMILES", "CanonicalSMILES", "InChIKey"],
            "batch_size": args.batch_size,
            "batch_count": len(chunks),
            "minimum_request_interval_seconds": args.min_interval,
            "attempt_limit": args.attempts,
        },
        "coverage": {
            "requested_unique_cids": len(cids),
            "returned_unique_cids": len(returned),
            "missing_cids": 0,
            "extra_cids": 0,
            "all_required_properties_nonblank": True,
        },
        "execution": {"fetched_batches": fetched, "resumed_batches": resumed},
        "merged_artifact": {
            "path": merged_path.name,
            "rows": len(returned),
            "bytes": len(merged_bytes),
            "sha256": sha256_bytes(merged_bytes),
            "columns": ["CID", "SMILES", "ConnectivitySMILES", "InChIKey"],
        },
        "batches": batch_bindings,
    }
    manifest_path = output / "manifest.json"
    atomic_write(manifest_path, json_bytes(manifest))
    print(json.dumps({
        "status": manifest["status"],
        "unique_cids": len(cids),
        "batches": len(chunks),
        "fetched_batches": fetched,
        "resumed_batches": resumed,
        "merged_sha256": manifest["merged_artifact"]["sha256"],
        "manifest": str(manifest_path),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
