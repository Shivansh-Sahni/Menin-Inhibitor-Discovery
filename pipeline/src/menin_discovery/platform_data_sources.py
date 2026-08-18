"""Public-source snapshotting and bounded acquisition for the platform build.

The module copies already-present *public* raw artifacts byte-for-byte into an
isolated platform namespace and can collect a bounded, similarity-agnostic
multi-target ChEMBL panel.  It never scans or copies ``research/data/internal``
or the internal PK/hERG canonical tree.

The bounded panel is a heterogeneous integration/QC corpus used to exercise
the source contract.  Its deterministic first/middle/last page sampling is
explicitly nonrepresentative and makes no claim of global coverage.  The full
official ChEMBL release is handled separately by :mod:`platform_data_bulk`.
"""

from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from .http import build_session, get_response
from .platform_data_schema import SCHEMA_VERSION, canonical_json, clean_text, stable_id

CHEMBL_API = "https://www.ebi.ac.uk/chembl/api/data"


@dataclass(frozen=True)
class ChEMBLPanelTarget:
    target_chembl_id: str
    target_name: str
    target_family: str
    expected_uniprot: str


# A small deliberately heterogeneous integration panel: kinases, GPCRs,
# nuclear receptor, ion channel, metabolic enzyme, and the historical target.
DEFAULT_CHEMBL_PANEL = (
    ChEMBLPanelTarget("CHEMBL203", "EGFR", "receptor_tyrosine_kinase", "P00533"),
    ChEMBLPanelTarget("CHEMBL1862", "ABL1", "nonreceptor_tyrosine_kinase", "P00519"),
    ChEMBLPanelTarget("CHEMBL217", "Dopamine D2 receptor", "class_a_gpcr", "P14416"),
    ChEMBLPanelTarget("CHEMBL210", "Beta-2 adrenergic receptor", "class_a_gpcr", "P07550"),
    ChEMBLPanelTarget("CHEMBL206", "Estrogen receptor alpha", "nuclear_receptor", "P03372"),
    ChEMBLPanelTarget("CHEMBL340", "Cytochrome P450 3A4", "metabolic_enzyme", "P08684"),
    ChEMBLPanelTarget("CHEMBL240", "KCNH2 / hERG", "voltage_gated_ion_channel", "Q12809"),
    ChEMBLPanelTarget("CHEMBL1615381", "Menin", "scaffold_protein", "O00255"),
)
DEFAULT_PANEL_ENDPOINTS = ("Kd", "Ki", "IC50", "EC50")

_LOCAL_PUBLIC_FILES = (("research/data/raw/chembl", "local_legacy/chembl"),)
_CONDITIONAL_LOCAL_SOURCES = (
    ("BindingDB targeted exports", Path("research/data/raw/bindingdb"), "mixed origin/release unresolved"),
    ("PubChem BioAssay", Path("research/data/raw/pubchem"), "contributor-specific rights unresolved"),
)
_SUN_WORKBOOK = Path(
    "research/literature/herg/predictive_modeling/2026_sun_wang_shen_jcim_6c00163/supplementary_dataset.xlsx"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: str | os.PathLike[str], *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_json(path: Path, document: Any) -> None:
    _atomic_write_bytes(
        path,
        (json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8"),
    )


def _copy_immutable(source: Path, destination: Path) -> tuple[str, bool]:
    """Copy bytes once and reject a later collision with different content."""

    source_hash = sha256_file(source)
    if destination.exists():
        destination_hash = sha256_file(destination)
        if destination_hash != source_hash:
            raise RuntimeError(
                f"Immutable raw snapshot collision: {destination} has {destination_hash}, "
                f"source has {source_hash}"
            )
        return source_hash, False
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        shutil.copyfile(source, temporary)
        if sha256_file(temporary) != source_hash:
            raise RuntimeError(f"Raw snapshot hash changed during copy: {source}")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return source_hash, True


def snapshot_local_public_sources(
    project_root: str | os.PathLike[str], raw_root: str | os.PathLike[str]
) -> dict[str, Any]:
    """Create byte-identical snapshots of all clearly public local raw data.

    The allow-list is explicit so that a future internal file cannot be copied
    merely because it appears somewhere under ``research/data``.
    """

    project = Path(project_root).resolve()
    destination_root = Path(raw_root).resolve()
    copied: list[dict[str, Any]] = []
    missing: list[str] = []
    for public_origin, destination_relative in _LOCAL_PUBLIC_FILES:
        source_directory = project / public_origin
        if not source_directory.exists():
            missing.append(public_origin)
            continue
        for source in sorted(path for path in source_directory.rglob("*") if path.is_file()):
            relative = source.relative_to(source_directory)
            destination = destination_root / destination_relative / relative
            digest, was_copied = _copy_immutable(source, destination)
            copied.append(
                {
                    "origin_path": source.relative_to(project).as_posix(),
                    "snapshot_path": destination.relative_to(destination_root).as_posix(),
                    "sha256": digest,
                    "size_bytes": source.stat().st_size,
                    "copied_now": was_copied,
                }
            )

    conditional: list[dict[str, Any]] = []
    for source_name, conditional_origin, reason in _CONDITIONAL_LOCAL_SOURCES:
        source_path = project / conditional_origin
        if not source_path.exists():
            continue
        candidates = (
            [source_path]
            if source_path.is_file()
            else sorted(path for path in source_path.rglob("*") if path.is_file())
        )
        for source in candidates:
            conditional.append(
                {
                    "source_name": source_name,
                    "origin_path": source.relative_to(project).as_posix(),
                    "sha256": sha256_file(source),
                    "size_bytes": source.stat().st_size,
                    "decision": "excluded_from_public_raw_and_canonical",
                    "reason": reason,
                }
            )
    sun_source = project / _SUN_WORKBOOK
    if sun_source.exists():
        conditional.append(
            {
                "source_name": "Sun/Wang/Shen hERG supplementary dataset",
                "origin_path": sun_source.relative_to(project).as_posix(),
                "sha256": sha256_file(sun_source),
                "size_bytes": sun_source.stat().st_size,
                "decision": "excluded_from_public_raw_and_canonical",
                "reason": "publisher supplementary-information redistribution rights unresolved",
            }
        )

    existing_manifest = destination_root / "local_snapshot_manifest.json"
    if existing_manifest.exists():
        previous = json.loads(existing_manifest.read_text(encoding="utf-8"))
        captured_at = str(previous.get("captured_at_utc", "")) or utc_now()
    else:
        captured_at = utc_now()
    document: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "snapshot_policy": "explicit-public-allow-list-byte-identical-v1",
        "captured_at_utc": captured_at,
        "confidentiality_guard": (
            "only rights-verified ChEMBL bytes are copied; internal, BindingDB, PubChem, and Sun bytes are not copied"
        ),
        "file_count": len(copied),
        "total_size_bytes": int(sum(row["size_bytes"] for row in copied)),
        "missing_allow_list_paths": sorted(missing),
        "files": sorted(copied, key=lambda row: row["snapshot_path"]),
    }
    # copied_now is execution metadata and would make the manifest unstable.
    for row in document["files"]:
        row.pop("copied_now", None)
    _atomic_write_json(existing_manifest, document)
    _atomic_write_json(
        destination_root / "conditional_source_inventory.json",
        {
            "schema_version": SCHEMA_VERSION,
            "policy": "manifest-only-fail-closed-v1",
            "file_count": len(conditional),
            "total_size_bytes": int(sum(row["size_bytes"] for row in conditional)),
            "files": sorted(conditional, key=lambda row: (row["source_name"], row["origin_path"])),
        },
    )
    return document


def _request_bytes(
    session: requests.Session,
    url: str,
    *,
    params: dict[str, Any] | None = None,
    timeout: int = 90,
) -> tuple[bytes, str]:
    response = get_response(url, params=params, timeout=(10, timeout), session=session)
    return response.content, str(response.url)


def _fetch_raw_json(
    session: requests.Session,
    url: str,
    destination: Path,
    *,
    params: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], str]:
    if destination.exists():
        return json.loads(destination.read_text(encoding="utf-8")), url
    content, resolved_url = _request_bytes(session, url, params=params)
    document = json.loads(content.decode("utf-8"))
    _atomic_write_bytes(destination, content)
    return document, resolved_url


def _sampling_offsets(total_count: int, page_size: int, max_records: int) -> list[int]:
    if total_count <= 0:
        return [0]
    if total_count <= max_records:
        return list(range(0, total_count, page_size))
    page_count = max(1, max_records // page_size)
    if page_count == 1:
        return [0]
    last = max(0, total_count - page_size)
    offsets = {int(round(index * last / (page_count - 1))) for index in range(page_count)}
    return sorted(offsets)


def collect_chembl_panel(
    raw_root: str | os.PathLike[str],
    *,
    targets: tuple[ChEMBLPanelTarget, ...] = DEFAULT_CHEMBL_PANEL,
    endpoints: tuple[str, ...] = DEFAULT_PANEL_ENDPOINTS,
    page_size: int = 40,
    max_records_per_target_endpoint: int = 120,
    sleep_seconds: float = 0.05,
    allow_partial: bool = True,
) -> dict[str, Any]:
    """Collect a bounded, auditable ChEMBL_37 multi-target snapshot.

    Raw API response bytes are preserved per page.  For large target/endpoint
    result sets, deterministic first/middle/last offset pages are sampled.
    This is an integration and representation panel, not a random population
    sample and not a substitute for the full bulk release.
    """

    if page_size <= 0 or max_records_per_target_endpoint < page_size:
        raise ValueError("max_records_per_target_endpoint must be at least one positive page")
    destination_root = Path(raw_root).resolve() / "chembl_37_panel"
    acquisition_path = destination_root / "acquisition.json"
    if acquisition_path.exists():
        existing = json.loads(acquisition_path.read_text(encoding="utf-8"))
        expected_plan = {
            "targets": [asdict(target) for target in targets],
            "endpoints": list(endpoints),
            "page_size": page_size,
            "max_records_per_target_endpoint": max_records_per_target_endpoint,
        }
        if existing.get("query_plan") != expected_plan:
            raise RuntimeError(
                "An immutable ChEMBL panel already exists with a different query plan; "
                "use a new snapshot directory/version rather than overwriting it."
            )
        return existing

    destination_root.mkdir(parents=True, exist_ok=True)
    session = build_session(backoff_factor=0.8)
    errors: list[dict[str, str]] = []
    query_rows: list[dict[str, Any]] = []
    retrieved_at = utc_now()
    status: dict[str, Any] = {}
    try:
        status, _ = _fetch_raw_json(session, f"{CHEMBL_API}/status.json", destination_root / "status.json")
    except (requests.RequestException, ValueError, json.JSONDecodeError) as exc:
        errors.append({"scope": "status", "error": f"{type(exc).__name__}: {exc}"})
        if not allow_partial:
            raise

    target_directory = destination_root / "targets"
    for target in targets:
        try:
            _fetch_raw_json(
                session,
                f"{CHEMBL_API}/target/{target.target_chembl_id}.json",
                target_directory / f"{target.target_chembl_id}.json",
            )
        except (requests.RequestException, ValueError, json.JSONDecodeError) as exc:
            errors.append(
                {
                    "scope": f"target:{target.target_chembl_id}",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            if not allow_partial:
                raise
        time.sleep(sleep_seconds)

    page_directory = destination_root / "activity_pages"
    for target in targets:
        for endpoint in endpoints:
            base_params: dict[str, Any] = {
                "target_chembl_id": target.target_chembl_id,
                "standard_type": endpoint,
                "order_by": "activity_id",
                "limit": page_size,
            }
            first_path = page_directory / f"{target.target_chembl_id}__{endpoint}__offset_0.json"
            try:
                first, resolved_url = _fetch_raw_json(
                    session,
                    f"{CHEMBL_API}/activity.json",
                    first_path,
                    params={**base_params, "offset": 0},
                )
                total_count = int(
                    first.get("page_meta", {}).get("total_count", len(first.get("activities", [])))
                )
                offsets = _sampling_offsets(total_count, page_size, max_records_per_target_endpoint)
                page_rows: list[dict[str, Any]] = []
                for offset in offsets:
                    page_path = page_directory / (
                        f"{target.target_chembl_id}__{endpoint}__offset_{offset}.json"
                    )
                    if offset == 0:
                        page = first
                    else:
                        page, resolved_url = _fetch_raw_json(
                            session,
                            f"{CHEMBL_API}/activity.json",
                            page_path,
                            params={**base_params, "offset": offset},
                        )
                        time.sleep(sleep_seconds)
                    page_rows.append(
                        {
                            "offset": offset,
                            "file": page_path.relative_to(destination_root).as_posix(),
                            "sha256": sha256_file(page_path),
                            "returned_records": len(page.get("activities", [])),
                        }
                    )
                query_rows.append(
                    {
                        "target_chembl_id": target.target_chembl_id,
                        "target_name": target.target_name,
                        "target_family": target.target_family,
                        "endpoint": endpoint,
                        "total_matching_records": total_count,
                        "sampling_design": "deterministic_even_offset_pages",
                        "sampled_offsets": offsets,
                        "sampled_records_before_id_dedup": int(
                            sum(row["returned_records"] for row in page_rows)
                        ),
                        "resolved_url": resolved_url,
                        "pages": page_rows,
                    }
                )
            except (requests.RequestException, ValueError, json.JSONDecodeError, OSError) as exc:
                errors.append(
                    {
                        "scope": f"activity:{target.target_chembl_id}:{endpoint}",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                if not allow_partial:
                    raise

    query_plan = {
        "targets": [asdict(target) for target in targets],
        "endpoints": list(endpoints),
        "page_size": page_size,
        "max_records_per_target_endpoint": max_records_per_target_endpoint,
    }
    release = str(status.get("chembl_db_version", "unresolved"))
    document = {
        "schema_version": SCHEMA_VERSION,
        "connector_version": "chembl-bounded-panel-1.0",
        "retrieved_at_utc": retrieved_at,
        "source_url": CHEMBL_API,
        "source_version": release,
        "source_release_date": status.get("chembl_release_date", ""),
        "query_plan": query_plan,
        "retrieval_status": "complete" if not errors else "partial",
        "errors": errors,
        "queries": sorted(query_rows, key=lambda row: (row["target_chembl_id"], row["endpoint"])),
        "coverage_boundary": (
            "bounded deterministic offset sample for connector/schema/QC readiness; "
            "not random, exhaustive, or suitable for population-frequency claims"
        ),
    }
    _atomic_write_json(acquisition_path, document)
    return document


def load_chembl_panel_activities(raw_root: str | os.PathLike[str]) -> pd.DataFrame:
    """Load and source-tag the preserved ChEMBL panel response pages."""

    root = Path(raw_root).resolve() / "chembl_37_panel"
    acquisition_path = root / "acquisition.json"
    if not acquisition_path.exists():
        return pd.DataFrame()
    acquisition = json.loads(acquisition_path.read_text(encoding="utf-8"))
    frames: list[pd.DataFrame] = []
    for query in acquisition.get("queries", []):
        for page in query.get("pages", []):
            path = root / str(page["file"])
            document = json.loads(path.read_text(encoding="utf-8"))
            frame = pd.DataFrame(document.get("activities", []))
            if frame.empty:
                continue
            frame["platform_raw_file"] = path.relative_to(Path(raw_root).resolve()).as_posix()
            frame["platform_snapshot_role"] = "bounded_multitarget_panel"
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True, sort=False)
    if "activity_id" in combined.columns:
        combined = combined.sort_values("activity_id", kind="stable").drop_duplicates(
            "activity_id", keep="first"
        )
    return combined.reset_index(drop=True)


def _source_ids() -> dict[str, str]:
    return {
        "chembl": stable_id("SRC", "ChEMBL", "ChEMBL_37"),
        "bindingdb": stable_id("SRC", "BindingDB", "unresolved_target_export_2026-07-14"),
        "pubchem": stable_id("SRC", "PubChem BioAssay", "live_archive_2026-07-14"),
        "sun": stable_id("SRC", "Sun Wang Shen hERG supplement", "2026-jcim-6c00163"),
    }


def source_registry(raw_root: str | os.PathLike[str]) -> pd.DataFrame:
    """Return verified/unresolved source metadata as independent axes."""

    root = Path(raw_root).resolve()
    identifiers = _source_ids()
    local_manifest_path = root / "local_snapshot_manifest.json"
    captured_at = ""
    if local_manifest_path.exists():
        captured_at = str(
            json.loads(local_manifest_path.read_text(encoding="utf-8")).get("captured_at_utc", "")
        )
    panel_path = root / "chembl_37_panel" / "acquisition.json"
    panel: dict[str, Any] = {}
    if panel_path.exists():
        panel = json.loads(panel_path.read_text(encoding="utf-8"))

    records = [
        {
            "source_id": identifiers["chembl"],
            "snapshot_id": stable_id("SNP", "ChEMBL", "local-targeted", "2026-07-14"),
            "source_name": "ChEMBL",
            "source_version": "ChEMBL_37",
            "retrieval_date_utc": captured_at or "2026-07-14T19:19:14Z",
            "source_url": "https://www.ebi.ac.uk/chembl/",
            "query_json": canonical_json(
                {
                    "scope": ["CHEMBL240", "CHEMBL1615381", "activities for Menin-associated molecules"],
                    "artifact": "legacy targeted REST exports",
                }
            ),
            "citation": (
                "Mendez et al. ChEMBL: towards direct deposition of bioassay data. "
                "Nucleic Acids Research 2019; DOI 10.1093/nar/gky1075."
            ),
            "license_name": "CC BY-SA 3.0",
            "license_url": "https://creativecommons.org/licenses/by-sa/3.0/",
            "license_status": "verified_official_source_page_2026-08-04",
            "access_class": "public_redistributable",
            "redistribution_status": "share_alike_required",
            "source_record_scope": "targeted Menin/hERG and Menin-molecule neighborhood; not full ChEMBL",
            "retrieval_status": "local_snapshot",
            "limitations": "Target/molecule-query selection bias; use the separately manifested bulk snapshot for global coverage.",
        },
        {
            "source_id": identifiers["bindingdb"],
            "snapshot_id": stable_id("SNP", "BindingDB", "menin-target-exports", "2026-07-14"),
            "source_name": "BindingDB",
            "source_version": "unresolved_target_export_2026-07-14",
            "retrieval_date_utc": captured_at or "2026-07-14T19:15:00Z",
            "source_url": "https://www.bindingdb.org/",
            "query_json": canonical_json(
                {"targets": ["Menin", "Histone-lysine N-methyltransferase 2A/Menin"]}
            ),
            "citation": "Gilson et al. BindingDB in 2024: a FAIR knowledgebase of protein-small molecule binding data. NAR 2024.",
            "license_name": "CC BY 4.0 for BindingDB-curated; CC BY-SA 3.0 for ChEMBL-derived rows",
            "license_url": "https://creativecommons.org/licenses/by/4.0/",
            "license_status": "mixed_row_level_curation_source_review_required",
            "access_class": "public_access_restricted",
            "redistribution_status": "conditional_on_curation_datasource",
            "source_record_scope": "two targeted live TSV exports; not full BindingDB",
            "retrieval_status": "excluded_rights_pending",
            "limitations": "Not copied or admitted to public canonical outputs: release and row-level origin review required.",
        },
        {
            "source_id": identifiers["pubchem"],
            "snapshot_id": stable_id("SNP", "PubChem BioAssay", "MEN1-search", "2026-07-14"),
            "source_name": "PubChem BioAssay",
            "source_version": "live_archive_snapshot_2026-07-14",
            "retrieval_date_utc": captured_at or "2026-07-14T19:19:14Z",
            "source_url": "https://pubchem.ncbi.nlm.nih.gov/",
            "query_json": canonical_json(
                {"scope": "MEN1/menin search terms and gene-linked assays", "assay_count_reported": 323}
            ),
            "citation": "Kim et al. PubChem 2025 update. Nucleic Acids Research.",
            "license_name": "contributor-specific",
            "license_url": "https://pubchem.ncbi.nlm.nih.gov/docs/data-sources",
            "license_status": "source_specific_review_required",
            "access_class": "public_access_restricted",
            "redistribution_status": "conditional_not_blanket_approved",
            "source_record_scope": "MEN1/menin search-derived assay snapshot; not all PubChem BioAssay",
            "retrieval_status": "excluded_rights_pending",
            "limitations": "Assay contributor licenses and endpoint metadata vary; unresolved rows remain review-only.",
        },
        {
            "source_id": identifiers["sun"],
            "snapshot_id": stable_id("SNP", "Sun Wang Shen", "supplementary workbook", "2026"),
            "source_name": "Sun/Wang/Shen hERG supplementary dataset",
            "source_version": "JCIM_66_2026_7515-7523",
            "retrieval_date_utc": captured_at or "2026-07-21T15:37:00Z",
            "source_url": "https://doi.org/10.1021/acs.jcim.6c00163",
            "query_json": canonical_json({"sheets": ["Classification", "Regression", "Validation"]}),
            "citation": (
                "Sun, Wang, and Shen. Modeling hERG Channel Liability: From Structural Insight to "
                "Highly Accurate Qualitative and Quantitative Models. JCIM 66 (2026) 7515-7523."
            ),
            "license_name": "publisher supplementary-information terms unresolved",
            "license_url": "https://pubs.acs.org/doi/10.1021/acs.jcim.6c00163",
            "license_status": "license_review_required",
            "access_class": "public_access_restricted",
            "redistribution_status": "derived_use_pending_review",
            "source_record_scope": "article supplementary classification/regression/validation workbook",
            "retrieval_status": "excluded_rights_pending",
            "limitations": "Compiled assay protocols are largely absent; article-specific model atom typing is unavailable.",
        },
    ]
    if panel:
        records.append(
            {
                "source_id": identifiers["chembl"],
                "snapshot_id": stable_id(
                    "SNP",
                    "ChEMBL",
                    panel.get("source_version", "unresolved"),
                    canonical_json(panel.get("query_plan", {})),
                ),
                "source_name": "ChEMBL",
                "source_version": panel.get("source_version", "unresolved"),
                "retrieval_date_utc": panel.get("retrieved_at_utc", ""),
                "source_url": CHEMBL_API,
                "query_json": canonical_json(panel.get("query_plan", {})),
                "citation": records[0]["citation"],
                "license_name": "CC BY-SA 3.0",
                "license_url": "https://creativecommons.org/licenses/by-sa/3.0/",
                "license_status": "verified_official_source_page_2026-08-04",
                "access_class": "public_redistributable",
                "redistribution_status": "share_alike_required",
                "source_record_scope": "bounded heterogeneous target/endpoint offset panel; not full ChEMBL",
                "retrieval_status": panel.get("retrieval_status", "partial"),
                "limitations": panel.get("coverage_boundary", "bounded non-exhaustive panel"),
            }
        )
    release_manifest_path = root / "chembl_37_bulk" / "release_metadata" / "release_metadata_manifest.json"
    if release_manifest_path.exists():
        release = json.loads(release_manifest_path.read_text(encoding="utf-8"))
        archive_manifest_path = root / "chembl_37_bulk" / "archive_manifest.json"
        archive = (
            json.loads(archive_manifest_path.read_text(encoding="utf-8"))
            if archive_manifest_path.exists()
            else {}
        )
        records.append(
            {
                "source_id": identifiers["chembl"],
                "snapshot_id": stable_id(
                    "SNP",
                    "ChEMBL",
                    "ChEMBL_37",
                    str(archive.get("archive_sha256", "release-metadata-only")),
                    "full-official-release",
                ),
                "source_name": "ChEMBL",
                "source_version": "ChEMBL_37",
                "retrieval_date_utc": release.get("captured_at_utc", ""),
                "source_url": release.get("release_root", ""),
                "query_json": canonical_json({"scope": "full official SQLite release", "selection": "none"}),
                "citation": records[0]["citation"],
                "license_name": "CC BY-SA 3.0",
                "license_url": "https://creativecommons.org/licenses/by-sa/3.0/",
                "license_status": "official_release_license_and_attribution_snapshotted",
                "access_class": "public_redistributable",
                "redistribution_status": "share_alike_and_required_attribution",
                "source_record_scope": "full official ChEMBL_37 release; canonical eligibility is QC-gated",
                "retrieval_status": "archive_verified" if archive else "release_metadata_snapshot",
                "limitations": "Source assertions include heterogeneous activity types and are not uniformly task eligible.",
            }
        )
    return (
        pd.DataFrame(records)
        .sort_values(["source_name", "snapshot_id"], kind="stable")
        .reset_index(drop=True)
    )


def source_file_inventory(raw_root: str | os.PathLike[str], sources: pd.DataFrame) -> pd.DataFrame:
    """Hash every preserved raw file and map it to a source/snapshot."""

    root = Path(raw_root).resolve()
    bulk_allowed: set[str] = set()
    bulk_declared: dict[str, tuple[str, int]] = {}
    bulk_root = root / "chembl_37_bulk"
    for manifest_name in ("archive_manifest.json", "archive_assembly_manifest.json"):
        manifest_path = bulk_root / manifest_name
        if manifest_path.exists():
            bulk_allowed.add(manifest_path.relative_to(root).as_posix())
    archive_manifest_path = bulk_root / "archive_manifest.json"
    if archive_manifest_path.exists():
        archive_manifest = json.loads(archive_manifest_path.read_text(encoding="utf-8"))
        archive_file = clean_text(archive_manifest.get("archive_file"))
        if archive_file:
            relative = (Path("chembl_37_bulk") / archive_file).as_posix()
            bulk_allowed.add(relative)
            bulk_declared[relative] = (
                clean_text(archive_manifest.get("archive_sha256")),
                int(archive_manifest.get("archive_size_bytes", -1)),
            )
    release_manifest_path = bulk_root / "release_metadata" / "release_metadata_manifest.json"
    if release_manifest_path.exists():
        release_manifest = json.loads(release_manifest_path.read_text(encoding="utf-8"))
        bulk_allowed.add(release_manifest_path.relative_to(root).as_posix())
        for record in release_manifest.get("files", []):
            filename = clean_text(record.get("filename"))
            if filename:
                relative = (Path("chembl_37_bulk/release_metadata") / filename).as_posix()
                bulk_allowed.add(relative)
                bulk_declared[relative] = (
                    clean_text(record.get("sha256")),
                    int(record.get("size_bytes", -1)),
                )
    extraction_manifest_path = bulk_root / "extracted" / "extraction_manifest.json"
    if extraction_manifest_path.exists():
        extraction_manifest = json.loads(extraction_manifest_path.read_text(encoding="utf-8"))
        bulk_allowed.add(extraction_manifest_path.relative_to(root).as_posix())
        for record in extraction_manifest.get("files", []):
            filename = clean_text(record.get("output_file"))
            if filename:
                relative = (Path("chembl_37_bulk/extracted") / filename).as_posix()
                bulk_allowed.add(relative)
                bulk_declared[relative] = (
                    clean_text(record.get("sha256")),
                    int(record.get("size_bytes", -1)),
                )
    origin_lookup: dict[str, str] = {}
    local_manifest_path = root / "local_snapshot_manifest.json"
    if local_manifest_path.exists():
        manifest = json.loads(local_manifest_path.read_text(encoding="utf-8"))
        origin_lookup = {
            str(row["snapshot_path"]): str(row["origin_path"]) for row in manifest.get("files", [])
        }

    chembl_rows = sources[sources["source_name"] == "ChEMBL"]
    local_rows = chembl_rows[
        chembl_rows["source_record_scope"].str.startswith("targeted Menin/hERG", na=False)
    ]
    panel_rows = chembl_rows[
        chembl_rows["source_record_scope"].str.startswith("bounded heterogeneous", na=False)
    ]
    bulk_rows = chembl_rows[chembl_rows["source_record_scope"].str.startswith("full official", na=False)]
    local_chembl_source = local_rows.iloc[0] if not local_rows.empty else None
    panel_source = panel_rows.iloc[0] if not panel_rows.empty else None
    bulk_source = bulk_rows.iloc[0] if not bulk_rows.empty else None

    def source_for(relative: str) -> pd.Series | None:
        if relative.startswith("local_legacy/chembl/"):
            return local_chembl_source
        if relative.startswith("chembl_37_panel/"):
            return panel_source
        if relative in bulk_allowed:
            return bulk_source
        return None

    rows: list[dict[str, Any]] = []
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative in {"local_snapshot_manifest.json"}:
            continue
        source = source_for(relative)
        if source is None:
            continue
        if relative in bulk_declared:
            digest, declared_size = bulk_declared[relative]
            if not digest or path.stat().st_size != declared_size:
                raise RuntimeError(f"Manifested ChEMBL bulk file failed size/hash metadata gate: {relative}")
        else:
            digest = sha256_file(path)
        row_count: int | None = None
        if path.suffix.casefold() in {".csv", ".tsv"}:
            try:
                with path.open("rb") as handle:
                    row_count = max(0, sum(1 for _ in handle) - 1)
            except OSError:
                row_count = None
        rows.append(
            {
                "source_file_id": stable_id("FILE", digest),
                "source_id": source["source_id"],
                "snapshot_id": source["snapshot_id"],
                "relative_path": relative,
                "origin_path": origin_lookup.get(relative, ""),
                "sha256": digest,
                "size_bytes": int(path.stat().st_size),
                "row_count": row_count,
                "media_type": mimetypes.guess_type(path.name, strict=False)[0] or "application/octet-stream",
                "immutability_status": (
                    "manifest_hash_verified_at_acquisition"
                    if relative in bulk_declared
                    else "hash_verified_current_inventory"
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("relative_path", kind="stable").reset_index(drop=True)


def acquisition_readiness_plan(
    raw_root: str | os.PathLike[str] = "research/data/platform/raw",
    interim_root: str | os.PathLike[str] = "research/data/platform/interim",
) -> dict[str, Any]:
    """Declare what is and is not acquired before large-model training."""

    raw = Path(raw_root).resolve()
    interim = Path(interim_root).resolve()
    release_manifest = raw / "chembl_37_bulk" / "release_metadata" / "release_metadata_manifest.json"
    archive_manifest = raw / "chembl_37_bulk" / "archive_manifest.json"
    export_manifest = interim / "chembl_37_bulk" / "activity_export_manifest.json"
    completed = [
        "byte-identical rights-verified ChEMBL snapshots",
        "hash-only fail-closed inventory for conditional BindingDB/PubChem/Sun artifacts",
        "bounded heterogeneous ChEMBL_37 protein/endpoint integration panel when network retrieval succeeds",
        "raw-page query manifests with version, license, citation, and hashes",
    ]
    if release_manifest.exists():
        completed.append("official ChEMBL_37 release license/attribution/checksum/schema metadata snapshot")
    if archive_manifest.exists():
        completed.append("official ChEMBL_37 SQLite archive checksum verification")
    if export_manifest.exists():
        exported = json.loads(export_manifest.read_text(encoding="utf-8"))
        completed.append(
            f"full ChEMBL_37 source-activity assertion export ({int(exported.get('rows_written', 0)):,} rows)"
        )
    if export_manifest.exists():
        chembl_status = {
            "source": "ChEMBL full release",
            "status": "activity_export_complete",
            "reason": "verified export manifest is present; canonical QC/task materialization remains separately gated",
        }
    elif archive_manifest.exists():
        chembl_status = {
            "source": "ChEMBL full release",
            "status": "archive_verified_export_pending",
            "reason": "the official archive is staged and verified, but the complete activity export is not yet manifest-complete",
        }
    else:
        chembl_status = {
            "source": "ChEMBL full release",
            "status": "not_acquired",
            "reason": (
                "24,527,044 activities require the official multi-GB release, persistent storage, "
                "share-alike handling, checksum verification, and an incremental build budget."
            ),
            "connector_plan": (
                "download official ChEMBL_37 SQLite archive plus checksums; stage immutable archive; "
                "export in activity_id keyset chunks; canonicalize without endpoint pooling"
            ),
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "global_corpus_ready": False,
        "chembl_bulk": chembl_status,
        "completed": completed,
        "blocked_or_deferred": [
            {
                "source": "BindingDB full release",
                "status": "not_acquired",
                "reason": (
                    "current local artifacts are two target exports; bulk ingestion requires current release/checksum "
                    "metadata and row-level license separation for BindingDB- versus ChEMBL-curated records"
                ),
                "connector_plan": (
                    "retrieve official all-binding TSV/SDF release into a versioned snapshot; partition by "
                    "Curation/DataSource; verify sequence/complex fields; then canonicalize in bounded chunks"
                ),
            },
            {
                "source": "Human PK and concentration-time repositories",
                "status": "not_acquired",
                "reason": "No license-cleared, structure-linked, protocol-complete human PK snapshot is local.",
                "connector_plan": "evaluate PK-DB/API and regulatory review documents with study/analyte/dose/route schema",
            },
            {
                "source": "Clinical QT/QTc and regulatory outcomes",
                "status": "not_acquired",
                "reason": (
                    "trial registries do not by themselves provide molecule-resolved concentration-QT observations; "
                    "registry presence is not a reported clinical result"
                ),
                "connector_plan": (
                    "snapshot ClinicalTrials.gov/AACT registry metadata separately; curate reported concentration-QT "
                    "tables from results/regulatory documents with human review and explicit compound identity"
                ),
            },
        ],
        "prohibited_inferences": [
            "no-trial match is not explicit preclinical status",
            "registry-only is not results-reported",
            "hERG potency is not clinical cardiotoxicity or TdP probability",
        ],
    }


__all__ = [
    "CHEMBL_API",
    "DEFAULT_CHEMBL_PANEL",
    "DEFAULT_PANEL_ENDPOINTS",
    "ChEMBLPanelTarget",
    "acquisition_readiness_plan",
    "collect_chembl_panel",
    "load_chembl_panel_activities",
    "sha256_file",
    "snapshot_local_public_sources",
    "source_file_inventory",
    "source_registry",
]
