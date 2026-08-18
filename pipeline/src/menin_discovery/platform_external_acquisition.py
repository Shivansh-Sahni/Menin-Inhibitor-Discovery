"""Immutable acquisition of external public platform evidence.

This module deliberately stops at byte- and parse-level acquisition.  It does
not construct canonical observations, resolve scientific conflicts, or admit
any downloaded row to a modeling task.  The source-specific manifests record
the semantic and rights boundaries that a later reviewed admission stage must
enforce.

Large downloads are written to ``*.part`` files, resumed only when the server
honours an exact byte range, verified, and atomically promoted.  ZIP archives
are never extracted by this module: their exact physical membership is
streamed into content-hashed JSONL inventories while reading every member to
verify CRC-32 and uncompressed byte counts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import tempfile
import zipfile
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import IO, Any, TextIO
from urllib.parse import urlencode

import pandas as pd
import requests

from .http import build_session

ACQUISITION_SCHEMA_VERSION = "platform-external-acquisition/1.0"
PARSER_VERSION = "platform_external_acquisition/1.0"
USER_AGENT = "menin-discovery-platform-external/1.0"
DEFAULT_CHUNK_BYTES = 8 * 1024 * 1024
SAFE_RESPONSE_HEADERS = frozenset(
    {
        "accept-ranges",
        "age",
        "cache-control",
        "content-disposition",
        "content-encoding",
        "content-length",
        "content-md5",
        "content-range",
        "content-type",
        "date",
        "etag",
        "expires",
        "last-modified",
        "link",
        "location",
        "server",
        "transfer-encoding",
        "vary",
        "x-api-version",
        "x-total-results",
        "x-uniprot-release",
        "x-uniprot-release-date",
    }
)

BINDINGDB_RELEASE_ID = "202608"
BINDINGDB_RELEASE_NOTE_DATE = "2026-07-26"
BINDINGDB_LANDING_URL = "https://www.bindingdb.org/rwd/bind/chemsearch/marvin/Download.jsp?all_download=yes"
BINDINGDB_FILES = (
    "BindingDB_BindingDB_Articles_202608_tsv.zip",
    "BindingDB_Assays_202608_tsv.zip",
    "BindingDB_rsid_eaids_202608_tsv.zip",
    "BindingDBTargetSequences.fasta",
    "BindingDB_UniProt.txt",
)
BINDINGDB_DOWNLOAD_ROOT = "https://www.bindingdb.org/rwd/bind"
BINDINGDB_CITATION = (
    "Gilson et al. BindingDB in 2024: a FAIR knowledgebase of protein-small molecule "
    "binding data. Nucleic Acids Research; DOI 10.1093/nar/gkae1075."
)
BINDINGDB_PRIMARY_PAPER_URL = "https://pmc.ncbi.nlm.nih.gov/articles/PMC11701568/"
BINDINGDB_PRIMARY_PAPER_XML_URL = (
    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id=11701568&rettype=full&retmode=xml"
)
BINDINGDB_PUBLISHED_MD5_FILES = {
    "BindingDB_BindingDB_Articles_202608_tsv.zip": "BindingDB_BindingDB_Articles_202608_tsv.md5",
    "BindingDB_Assays_202608_tsv.zip": "BindingDB_Assays_202608_tsv.md5",
    "BindingDB_rsid_eaids_202608_tsv.zip": "BindingDB_rsid_eaids_202608_tsv.md5",
}

UNIPROT_RELEASE_ID = "2026_02"
UNIPROT_RELEASE_DATE = "2026-06-10"
UNIPROT_RELEASE_HEADER_DATE = "10-June-2026"
UNIPROT_RELEASE_URL = "https://www.uniprot.org/release-notes/2026-06-10-release"
UNIPROT_LICENSE_URL = "https://www.uniprot.org/help/license"
UNIPROT_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"
UNIPROT_QUERY_BATCH_SIZE = 100
UNIPROT_ACCESSION_RE = re.compile(
    r"^(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})"
    r"(?:-[0-9]+)?$"
)

CLINICALTRIALS_VERSION_URL = "https://clinicaltrials.gov/api/v2/version"
CLINICALTRIALS_OPENAPI_URL = "https://clinicaltrials.gov/api/oas/v2/ctg-oas-v2.yaml"
CLINICALTRIALS_API_DOCS_URL = "https://clinicaltrials.gov/data-api/api"
CLINICALTRIALS_STUDY_METADATA_URL = "https://clinicaltrials.gov/api/v2/studies/metadata"
CLINICALTRIALS_TERMS_URL = "https://clinicaltrials.gov/about-site/terms-conditions"
CLINICALTRIALS_STUDIES_URL = "https://clinicaltrials.gov/api/v2/studies"
CLINICALTRIALS_DRUG_QUERY = "AREA[InterventionType]DRUG"
CLINICALTRIALS_DRUG_SORT = "LastUpdatePostDate:desc"
CLINICALTRIALS_PAGE_SIZE = 1_000
CLINICALTRIALS_DRUG_FIELDS = (
    "NCTId",
    "StudyType",
    "Phase",
    "OverallStatus",
    "StartDate",
    "StartDateType",
    "PrimaryCompletionDate",
    "PrimaryCompletionDateType",
    "CompletionDate",
    "CompletionDateType",
    "StudyFirstSubmitDate",
    "StudyFirstSubmitQCDate",
    "StudyFirstPostDate",
    "StudyFirstPostDateType",
    "ResultsFirstSubmitDate",
    "ResultsFirstSubmitQCDate",
    "ResultsFirstPostDate",
    "ResultsFirstPostDateType",
    "LastUpdateSubmitDate",
    "LastUpdatePostDate",
    "LastUpdatePostDateType",
    "LeadSponsorName",
    "LeadSponsorClass",
    "Condition",
    "InterventionType",
    "InterventionName",
    "InterventionDescription",
    "HasResults",
)
CLINICALTRIALS_CARDIAC_SAFETY_QUERY = (
    'AREA[InterventionType]DRUG AND (QT OR QTc OR hERG OR "torsades de pointes" '
    'OR cardiotoxicity OR "cardiac repolarization")'
)
CLINICALTRIALS_CARDIAC_SAFETY_FIELDS = (
    *CLINICALTRIALS_DRUG_FIELDS,
    "BriefTitle",
    "OfficialTitle",
    "BriefSummary",
    "DetailedDescription",
    "Keyword",
    "OutcomeMeasuresModule",
    "AdverseEventsModule",
)
CLINICALTRIALS_CARDIAC_TERM_PATTERNS: dict[str, re.Pattern[str]] = {
    "QT_or_QTc": re.compile(r"(?i)(?<![A-Za-z0-9])qtc?(?![A-Za-z0-9])"),
    "hERG": re.compile(r"(?i)(?<![A-Za-z0-9])hERG(?![A-Za-z0-9])"),
    "torsades": re.compile(r"(?i)torsad(?:e|es)"),
    "cardiotoxicity": re.compile(r"(?i)cardiotoxic"),
    "cardiac_repolarization": re.compile(r"(?i)cardiac\s+repolari[sz]ation"),
}

DRUGSFDA_LANDING_URL = "https://www.fda.gov/drugs/drug-approvals-and-databases/drugsfda-data-files"
DRUGSFDA_ARCHIVE_URL = "https://www.fda.gov/media/89850/download?attachment="
DRUGSFDA_ERD_URL = "https://www.fda.gov/media/102072/download"
DRUGSFDA_EXPECTED_FILENAME = "datdaf20260804.zip"
DRUGSFDA_EXPECTED_TABLES = 12
DRUGSFDA_PRIMARY_KEYS: dict[str, tuple[str, ...]] = {
    "ActionTypes_Lookup.txt": ("ActionTypes_LookupID",),
    "ApplicationDocs.txt": ("ApplicationDocsID",),
    "Applications.txt": ("ApplNo",),
    "ApplicationsDocsType_Lookup.txt": ("ApplicationDocsType_Lookup_ID",),
    "Join_Submission_ActionTypes_Lookup.txt": ("j_submissionActionTypeID",),
    "MarketingStatus.txt": ("ApplNo", "ProductNo"),
    "MarketingStatus_Lookup.txt": ("MarketingStatusID",),
    "Products.txt": ("ApplNo", "ProductNo"),
    "SubmissionClass_Lookup.txt": ("SubmissionClassCodeID",),
    "SubmissionPropertyType.txt": (
        "ApplNo",
        "SubmissionType",
        "SubmissionNo",
        "SubmissionPropertyTypeCode",
        "SubmissionPropertyTypeID",
    ),
    "Submissions.txt": ("ApplNo", "SubmissionType", "SubmissionNo"),
    "TE.txt": ("ApplNo", "ProductNo", "MarketingStatusID", "TECode"),
}
DRUGSFDA_FOREIGN_KEYS: tuple[tuple[str, tuple[str, ...], str, tuple[str, ...]], ...] = (
    (
        "ApplicationDocs.txt",
        ("ApplicationDocsTypeID",),
        "ApplicationsDocsType_Lookup.txt",
        ("ApplicationDocsType_Lookup_ID",),
    ),
    ("ApplicationDocs.txt", ("ApplNo",), "Applications.txt", ("ApplNo",)),
    (
        "ApplicationDocs.txt",
        ("ApplNo", "SubmissionType", "SubmissionNo"),
        "Submissions.txt",
        ("ApplNo", "SubmissionType", "SubmissionNo"),
    ),
    ("Products.txt", ("ApplNo",), "Applications.txt", ("ApplNo",)),
    ("MarketingStatus.txt", ("ApplNo", "ProductNo"), "Products.txt", ("ApplNo", "ProductNo")),
    ("MarketingStatus.txt", ("MarketingStatusID",), "MarketingStatus_Lookup.txt", ("MarketingStatusID",)),
    ("Submissions.txt", ("ApplNo",), "Applications.txt", ("ApplNo",)),
    ("Submissions.txt", ("SubmissionClassCodeID",), "SubmissionClass_Lookup.txt", ("SubmissionClassCodeID",)),
    (
        "Join_Submission_ActionTypes_Lookup.txt",
        ("ApplNo", "SubmissionType", "SubmissionNo"),
        "Submissions.txt",
        ("ApplNo", "SubmissionType", "SubmissionNo"),
    ),
    (
        "Join_Submission_ActionTypes_Lookup.txt",
        ("ActionTypes_LookupID",),
        "ActionTypes_Lookup.txt",
        ("ActionTypes_LookupID",),
    ),
    (
        "SubmissionPropertyType.txt",
        ("ApplNo", "SubmissionType", "SubmissionNo"),
        "Submissions.txt",
        ("ApplNo", "SubmissionType", "SubmissionNo"),
    ),
    ("TE.txt", ("ApplNo", "ProductNo"), "Products.txt", ("ApplNo", "ProductNo")),
    ("TE.txt", ("MarketingStatusID",), "MarketingStatus_Lookup.txt", ("MarketingStatusID",)),
)

DAILYMED_LANDING_URL = "https://dailymed.nlm.nih.gov/dailymed/spl-resources-all-drug-labels.cfm"
DAILYMED_DOWNLOAD_ROOT = "https://dailymed-data.nlm.nih.gov/public-release-files"
DAILYMED_PARTS = (
    ("dm_spl_release_human_rx_part1.zip", "6c604e98645bd9d328ef673efcff192e", 14_989, 3_221_045_354),
    ("dm_spl_release_human_rx_part2.zip", "01cab1e33066edb4b4fddc6f857b8d13", 9_605, 3_221_063_745),
    ("dm_spl_release_human_rx_part3.zip", "7ee94cb9778e237661d6d2bf925b7e52", 9_132, 3_220_663_350),
    ("dm_spl_release_human_rx_part4.zip", "9fca06d732755e90a79a702fab59bab1", 8_369, 3_220_858_219),
    ("dm_spl_release_human_rx_part5.zip", "4d12e445eb19b70798643a575a5422fb", 8_716, 3_220_978_159),
    ("dm_spl_release_human_rx_part6.zip", "0ad83aff1595e5f2a5d09898a5503ebc", 3_861, 1_662_589_295),
)
DAILYMED_RELEASE_PAGE_DATE = "2026-08-03"
DAILYMED_RELEASE_HTTP_LAST_MODIFIED_DATE = "2026-08-04"


SOURCE_BOUNDARIES: dict[str, dict[str, Any]] = {
    "bindingdb_curated_202608": {
        "license_review_state": "institutional_review_required_before_redistribution_or_admission",
        "license_boundary": (
            "Preserve BindingDB release terms and attribution. Only origin-isolated "
            "BindingDB-curated-articles rows are candidates for later CC BY 4.0 review; "
            "imported ChEMBL, PubChem, patent, PDSP, CSAR, and other origins are excluded."
        ),
        "citation": BINDINGDB_CITATION,
        "evidence_admission": (
            "Acquired bytes may enter a multisource evidence view only after origin, identity, "
            "endpoint, unit, censoring, assay, mirror, conflict, and rights review."
        ),
        "default_label_admission": (
            "prohibited_at_acquisition; never pool Kd, Ki, IC50, or EC50 and never count "
            "imported ChEMBL mirrors as independent evidence"
        ),
    },
    "uniprotkb_targeted_2026_02": {
        "license_review_state": "CC_BY_4_0_attribution_and_third_party_disclaimer_review_required",
        "license_boundary": (
            "UniProt identity and sequence content is acquired under the release-specific "
            "copyright, CC BY 4.0, patent, and third-party disclaimer boundary."
        ),
        "citation": "UniProt Consortium release 2026_02 (2026-06-10).",
        "evidence_admission": (
            "Identity/sequence evidence only after accession-state, construct, component, "
            "isoform, taxonomy, and sequence-checksum review."
        ),
        "default_label_admission": "never_a_label",
    },
    "clinicaltrials_gov_v2": {
        "license_review_state": "ClinicalTrials.gov_terms_and_disclaimer_review_required",
        "license_boundary": "Public registry records only; source attribution and disclaimer required.",
        "citation": "ClinicalTrials.gov API v2; record NCT ID, dataTimestamp, and retrieval date.",
        "evidence_admission": (
            "Registration, study-state, and posted-results are separate evidence states; "
            "molecule mappings require reviewed aliases and confidence."
        ),
        "default_label_admission": (
            "prohibited_at_acquisition; no record or no posted results is absence, never a "
            "negative molecular efficacy, safety, cardiotoxicity, or activity outcome"
        ),
    },
    "drugs_at_fda_bulk": {
        "license_review_state": "US_federal_public_data_site_policy_and_institutional_review_required",
        "license_boundary": "Preserve FDA source, site policy, disclaimer, and product context.",
        "citation": "Drugs@FDA data files page, update date, application/product IDs, retrieval date.",
        "evidence_admission": (
            "Regulatory application/product/action evidence only after relational and "
            "molecule/product/formulation mapping QC."
        ),
        "default_label_admission": (
            "never_a_molecular_activity_efficacy_safety_or_PK_label; approval and marketing "
            "status are product-, formulation-, indication-, time-, and jurisdiction-specific"
        ),
    },
    "dailymed_spl_v2_human_rx": {
        "license_review_state": "NLM_DailyMed_terms_copyright_and_API_review_required",
        "license_boundary": (
            "Human-prescription SPL scope only; preserve FDA/NLM attribution, label author, "
            "SETID/version, and XML provenance."
        ),
        "citation": "DailyMed SETID/version URL, label effective date, SPL author, retrieval date.",
        "evidence_admission": (
            "Versioned label-section evidence only after temporal, section-identity, product, and mapping QC."
        ),
        "default_label_admission": (
            "prohibited_at_acquisition; prose or section presence is not a normalized "
            "molecule-level safety, PK, efficacy, or cardiotoxicity label"
        ),
    },
}


@dataclass(frozen=True)
class DownloadSpec:
    """Contract for one immutable HTTP artifact."""

    canonical_url: str
    destination: Path
    expected_sha256: str | None = None
    expected_md5: str | None = None
    expected_bytes: int | None = None
    expected_filename: str | None = None


def utc_now() -> str:
    """Return second-resolution UTC in a portable manifest representation."""

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: str | os.PathLike[str], *, chunk_bytes: int = DEFAULT_CHUNK_BYTES) -> str:
    """Hash a file without materializing it in memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def md5_file(path: str | os.PathLike[str], *, chunk_bytes: int = DEFAULT_CHUNK_BYTES) -> str:
    """Compute a published-source MD5 integrity value (not a security identity)."""

    digest = hashlib.md5(usedforsecurity=False)
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize JSON deterministically for manifest identities."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def document_with_sha256(document: Mapping[str, Any]) -> dict[str, Any]:
    """Attach a non-circular SHA-256 over the manifest body."""

    body = {str(key): value for key, value in document.items() if key != "manifest_sha256"}
    body["manifest_sha256"] = hashlib.sha256(canonical_json_bytes(body)).hexdigest()
    return body


def verify_document_sha256(document: Mapping[str, Any]) -> bool:
    """Return whether a manifest's self-declared non-circular identity is exact."""

    expected = str(document.get("manifest_sha256", ""))
    if not expected:
        return False
    body = {str(key): value for key, value in document.items() if key != "manifest_sha256"}
    return hashlib.sha256(canonical_json_bytes(body)).hexdigest() == expected


def atomic_write_json(path: str | os.PathLike[str], value: Mapping[str, Any]) -> dict[str, Any]:
    """Write an fsync'd, self-identifying JSON manifest atomically."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    document = document_with_sha256(value)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(document, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return document


def _safe_headers(headers: Mapping[str, str]) -> dict[str, str]:
    return {
        str(name).casefold(): str(value)
        for name, value in headers.items()
        if str(name).casefold() in SAFE_RESPONSE_HEADERS
    }


def annotate_acquisition_record(
    record: Mapping[str, Any],
    *,
    artifact_role: str,
    local_path: str | None = None,
) -> dict[str, Any]:
    """Add source-manifest context without invalidating an acquisition sidecar identity."""

    annotated = dict(record)
    sidecar_digest = annotated.pop("manifest_sha256", None)
    if sidecar_digest:
        annotated["acquisition_sidecar_manifest_sha256"] = sidecar_digest
    annotated["artifact_role"] = artifact_role
    if local_path is not None:
        annotated["local_path"] = local_path
    return annotated


def _parse_content_range_total(value: str) -> int | None:
    match = re.fullmatch(r"bytes\s+(\d+)-(\d+)/(\d+|\*)", value.strip(), flags=re.IGNORECASE)
    if not match or match.group(3) == "*":
        return None
    return int(match.group(3))


def _parse_content_range_start(value: str) -> int | None:
    match = re.fullmatch(r"bytes\s+(\d+)-(\d+)/(\d+|\*)", value.strip(), flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def _response_total_bytes(response: requests.Response, *, resume_offset: int) -> int | None:
    content_range = response.headers.get("Content-Range", "")
    total = _parse_content_range_total(content_range)
    if total is not None:
        return total
    length = response.headers.get("Content-Length")
    if length and length.isdigit():
        return int(length) + (resume_offset if response.status_code == 206 else 0)
    return None


def _validate_existing_download(spec: DownloadSpec) -> dict[str, Any]:
    path = spec.destination
    size = path.stat().st_size
    sha256 = sha256_file(path)
    md5 = md5_file(path) if spec.expected_md5 else None
    if spec.expected_bytes is not None and size != spec.expected_bytes:
        raise ValueError(f"Existing artifact has {size} bytes; expected {spec.expected_bytes}: {path}")
    if spec.expected_sha256 and sha256.casefold() != spec.expected_sha256.casefold():
        raise ValueError(f"Existing artifact SHA-256 mismatch: {path}")
    if spec.expected_md5 and md5 != spec.expected_md5.casefold():
        raise ValueError(f"Existing artifact published MD5 mismatch: {path}")
    return {
        "acquisition_status": "already_present_verified",
        "acquired_bytes": size,
        "acquired_sha256": sha256,
        "acquired_md5": md5,
    }


def _partial_download_contract(spec: DownloadSpec) -> dict[str, Any]:
    """Return the immutable contract that a resumable prefix must remain bound to."""

    return {
        "canonical_url": spec.canonical_url,
        "upstream_filename": spec.expected_filename or spec.destination.name,
        "expected_bytes": spec.expected_bytes,
        "expected_sha256": spec.expected_sha256,
        "upstream_md5_when_published": spec.expected_md5,
    }


def _is_strong_etag(value: str | None) -> bool:
    """Return whether an HTTP ETag is present and not weak (``W/``)."""

    normalized = str(value or "").strip()
    if (
        len(normalized) < 2
        or normalized.casefold().startswith("w/")
        or not normalized.startswith('"')
        or not normalized.endswith('"')
    ):
        return False
    inner = normalized[1:-1]
    return all(character != '"' and ord(character) >= 0x21 and ord(character) != 0x7F for character in inner)


def download_immutable(
    spec: DownloadSpec,
    *,
    session: requests.Session | None = None,
    timeout: tuple[float, float] = (15.0, 180.0),
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
) -> dict[str, Any]:
    """Download, resume, verify, and atomically promote one immutable artifact.

    An existing destination is never overwritten.  It must satisfy the caller's
    declared hashes/size and is then re-manifested as already verified.  A
    partial file is appended only after an exact 206 response beginning at the
    current byte offset; a server that ignores the range causes a safe restart
    of the unpromoted partial, never an append of duplicate bytes.
    """

    destination = Path(spec.destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    acquisition_manifest = destination.with_name(f"{destination.name}.acquisition.json")
    if destination.exists():
        if not destination.is_file():
            raise ValueError(f"Download destination is not a regular file: {destination}")
        verified = _validate_existing_download(spec)
        if acquisition_manifest.exists():
            record = json.loads(acquisition_manifest.read_text(encoding="utf-8"))
            if not verify_document_sha256(record):
                raise ValueError(f"Acquisition sidecar identity failed: {acquisition_manifest}")
            if record.get("canonical_url") != spec.canonical_url:
                raise ValueError(f"Acquisition sidecar URL differs: {acquisition_manifest}")
            if int(record.get("acquired_bytes", -1)) != int(verified["acquired_bytes"]):
                raise ValueError(f"Acquisition sidecar byte count differs: {acquisition_manifest}")
            if record.get("acquired_sha256") != verified["acquired_sha256"]:
                raise ValueError(f"Acquisition sidecar SHA-256 differs: {acquisition_manifest}")
            if spec.expected_md5 and record.get("acquired_md5") != verified["acquired_md5"]:
                raise ValueError(f"Acquisition sidecar MD5 differs: {acquisition_manifest}")
            return record
        record = {
            "schema_version": ACQUISITION_SCHEMA_VERSION,
            "canonical_url": spec.canonical_url,
            "resolved_url": spec.canonical_url,
            "upstream_filename": spec.expected_filename or destination.name,
            "local_path": destination.name,
            "retrieval_utc": None,
            "first_manifested_utc": utc_now(),
            "response_status": None,
            "response_headers": {},
            "expected_bytes": spec.expected_bytes,
            "expected_sha256": spec.expected_sha256,
            "upstream_md5_when_published": spec.expected_md5,
            "resumed_from_bytes": 0,
            **verified,
        }
        record["acquisition_status"] = "preexisting_bytes_first_manifested_verified"
        return atomic_write_json(acquisition_manifest, record)

    partial = destination.with_name(f".{destination.name}.part")
    partial_metadata = destination.with_name(f".{destination.name}.part.json")
    offset = partial.stat().st_size if partial.exists() else 0
    restart_reason: str | None = None
    previous: dict[str, Any] | None = None
    validated_partial_sha256: str | None = None
    contract = _partial_download_contract(spec)
    contract_sha256 = hashlib.sha256(canonical_json_bytes(contract)).hexdigest()
    if offset:
        try:
            if not partial_metadata.is_file():
                raise ValueError("missing_partial_sidecar")
            loaded = json.loads(partial_metadata.read_text(encoding="utf-8"))
            if not isinstance(loaded, dict) or not verify_document_sha256(loaded):
                raise ValueError("partial_sidecar_identity_failed")
            if loaded.get("canonical_url") != spec.canonical_url:
                raise ValueError("partial_canonical_url_changed")
            if loaded.get("download_contract") != contract:
                raise ValueError("partial_expected_contract_changed")
            if loaded.get("download_contract_sha256") != contract_sha256:
                raise ValueError("partial_contract_digest_failed")
            if loaded.get("resume_forbidden_reason"):
                raise ValueError(str(loaded["resume_forbidden_reason"]))
            if int(loaded.get("partial_bytes", -1)) != offset:
                raise ValueError("partial_persisted_offset_differs")
            validated_partial_sha256 = sha256_file(partial)
            if loaded.get("partial_sha256") != validated_partial_sha256:
                raise ValueError("partial_prefix_sha256_changed")
            previous = loaded
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            restart_reason = str(exc)
            offset = 0
    client = session or build_session(retries=8, backoff_factor=1.0)
    client.headers.update({"User-Agent": USER_AGENT, "Accept-Encoding": "identity"})
    request_headers: dict[str, str] = {"Accept-Encoding": "identity"}
    if offset:
        strong_published_digest = bool(spec.expected_sha256 or spec.expected_md5)
        previous_etag = str(previous.get("etag") or "") if previous else ""
        if not _is_strong_etag(previous_etag) and not strong_published_digest:
            restart_reason = "partial_has_no_strong_validator_or_published_digest"
            offset = 0
        else:
            request_headers["Range"] = f"bytes={offset}-"
        if offset and _is_strong_etag(previous_etag):
            request_headers["If-Range"] = previous_etag
    retrieved_at = utc_now()
    with client.get(
        spec.canonical_url,
        headers=request_headers,
        stream=True,
        allow_redirects=True,
        timeout=timeout,
    ) as response:
        response.raise_for_status()
        if offset and (
            response.status_code != 206
            or _parse_content_range_start(response.headers.get("Content-Range", "")) != offset
        ):
            restart_reason = "server_did_not_honor_exact_range"
            offset = 0
        if offset and previous and not (spec.expected_sha256 or spec.expected_md5):
            previous_etag = str(previous.get("etag") or "")
            response_etag = str(response.headers.get("ETag") or "")
            if not _is_strong_etag(response_etag) or previous_etag != response_etag:
                forbidden = dict(previous)
                forbidden["resume_forbidden_reason"] = "resume_response_missing_or_changed_strong_etag"
                atomic_write_json(partial_metadata, forbidden)
                return download_immutable(
                    spec,
                    session=client,
                    timeout=timeout,
                    chunk_bytes=chunk_bytes,
                )
        mode = "ab" if offset else "wb"
        expected_total = _response_total_bytes(response, resume_offset=offset)
        if (
            expected_total is not None
            and spec.expected_bytes is not None
            and expected_total != spec.expected_bytes
        ):
            raise ValueError(
                f"HTTP total differs from published byte contract for {destination.name}: "
                f"{expected_total} != {spec.expected_bytes}"
            )
        partial_state = {
            "schema_version": ACQUISITION_SCHEMA_VERSION,
            "canonical_url": spec.canonical_url,
            "resolved_url": str(response.url),
            "retrieval_started_utc": retrieved_at,
            "etag": response.headers.get("ETag"),
            "last_modified": response.headers.get("Last-Modified"),
            "expected_total_bytes_from_response": expected_total,
            "resume_offset": offset,
            "partial_bytes": offset,
            "partial_sha256": (validated_partial_sha256 if offset else hashlib.sha256(b"").hexdigest()),
            "download_contract": contract,
            "download_contract_sha256": contract_sha256,
            "restart_reason": restart_reason,
        }
        atomic_write_json(partial_metadata, partial_state)
        checkpoint_sha256: str | None = None
        try:
            with partial.open(mode) as output:
                for chunk in response.iter_content(chunk_size=chunk_bytes):
                    if chunk:
                        output.write(chunk)
                output.flush()
                os.fsync(output.fileno())
        finally:
            if partial.exists():
                checkpoint_sha256 = sha256_file(partial)
                partial_state.update(
                    {
                        "partial_bytes": partial.stat().st_size,
                        "partial_sha256": checkpoint_sha256,
                        "checkpoint_utc": utc_now(),
                    }
                )
                atomic_write_json(partial_metadata, partial_state)
        response_record = {
            "resolved_url": str(response.url),
            "response_status": response.status_code,
            "response_headers": _safe_headers(response.headers),
        }

    actual_bytes = partial.stat().st_size
    if expected_total is not None and actual_bytes != expected_total:
        raise ValueError(
            f"HTTP byte-count mismatch for {destination.name}: got {actual_bytes}, expected {expected_total}"
        )
    if spec.expected_bytes is not None and actual_bytes != spec.expected_bytes:
        raise ValueError(
            f"Published byte-count mismatch for {destination.name}: got {actual_bytes}, "
            f"expected {spec.expected_bytes}"
        )
    actual_sha256 = checkpoint_sha256 or sha256_file(partial)
    if spec.expected_sha256 and actual_sha256.casefold() != spec.expected_sha256.casefold():
        raise ValueError(f"Published SHA-256 mismatch for {destination.name}")
    actual_md5 = md5_file(partial) if spec.expected_md5 else None
    if spec.expected_md5 and actual_md5 != spec.expected_md5.casefold():
        raise ValueError(
            f"Published MD5 mismatch for {destination.name}: got {actual_md5}, expected {spec.expected_md5}"
        )
    os.replace(partial, destination)
    partial_metadata.unlink(missing_ok=True)
    record = {
        "schema_version": ACQUISITION_SCHEMA_VERSION,
        "canonical_url": spec.canonical_url,
        **response_record,
        "upstream_filename": spec.expected_filename or destination.name,
        "local_path": destination.name,
        "retrieval_utc": retrieved_at,
        "acquisition_status": "downloaded_verified",
        "acquired_bytes": actual_bytes,
        "acquired_sha256": actual_sha256,
        "acquired_md5": actual_md5,
        "expected_bytes": spec.expected_bytes,
        "expected_sha256": spec.expected_sha256,
        "upstream_md5_when_published": spec.expected_md5,
        "resumed_from_bytes": offset,
        "partial_restart_reason": restart_reason,
    }
    return atomic_write_json(acquisition_manifest, record)


def snapshot_http_bytes(
    url: str,
    destination: str | os.PathLike[str],
    *,
    session: requests.Session | None = None,
    timeout: tuple[float, float] = (15.0, 180.0),
) -> dict[str, Any]:
    """Snapshot a small official HTTP resource through the immutable downloader."""

    return download_immutable(
        DownloadSpec(canonical_url=url, destination=Path(destination)),
        session=session,
        timeout=timeout,
    )


def _unsafe_archive_member(name: str) -> bool:
    normalized = name.replace("\\", "/")
    path = PurePosixPath(normalized)
    return path.is_absolute() or ".." in path.parts or "\x00" in name


def iter_binary_chunks(handle: IO[bytes], *, chunk_bytes: int = DEFAULT_CHUNK_BYTES) -> Iterator[bytes]:
    """Yield bounded binary chunks from a stream."""

    while True:
        chunk = handle.read(chunk_bytes)
        if not chunk:
            return
        yield chunk


def verify_zip_archive(
    archive_path: str | os.PathLike[str],
    inventory_path: str | os.PathLike[str],
    *,
    expected_member_count: int | None = None,
) -> dict[str, Any]:
    """Read every ZIP member, verify CRC/length, and write exact JSONL membership.

    The inventory itself is atomically promoted and content hashed.  Duplicate
    member names and unsafe paths fail closed because name-only extraction or
    lookup could otherwise be ambiguous, even though this module never extracts.
    """

    archive = Path(archive_path)
    inventory = Path(inventory_path)
    inventory.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{inventory.name}.", dir=inventory.parent)
    temporary = Path(temporary_name)
    seen: set[str] = set()
    member_count = 0
    file_count = 0
    directory_count = 0
    total_uncompressed = 0
    total_compressed = 0
    try:
        with (
            zipfile.ZipFile(archive, "r") as zipped,
            os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as output,
        ):
            for index, info in enumerate(zipped.infolist()):
                if info.filename in seen:
                    raise ValueError(f"Duplicate ZIP member name: {info.filename!r}")
                if _unsafe_archive_member(info.filename):
                    raise ValueError(f"Unsafe ZIP member name: {info.filename!r}")
                seen.add(info.filename)
                member_count += 1
                directory = info.is_dir()
                directory_count += int(directory)
                file_count += int(not directory)
                total_uncompressed += info.file_size
                total_compressed += info.compress_size
                digest = hashlib.sha256()
                observed_bytes = 0
                if not directory:
                    with zipped.open(info, "r") as member:
                        for chunk in iter_binary_chunks(member):
                            digest.update(chunk)
                            observed_bytes += len(chunk)
                if observed_bytes != info.file_size:
                    raise ValueError(
                        f"ZIP member byte mismatch for {info.filename!r}: "
                        f"{observed_bytes} != {info.file_size}"
                    )
                record = {
                    "member_index": index,
                    "archive_member_path": info.filename,
                    "is_directory": directory,
                    "uncompressed_bytes": info.file_size,
                    "compressed_bytes": info.compress_size,
                    "crc32_hex": f"{info.CRC:08x}",
                    "member_sha256": digest.hexdigest(),
                    "compression_method": info.compress_type,
                    "flag_bits": info.flag_bits,
                    "external_attributes": info.external_attr,
                    "zip_datetime": list(info.date_time),
                }
                output.write(canonical_json_bytes(record).decode("utf-8") + "\n")
            output.flush()
            os.fsync(output.fileno())
        if expected_member_count is not None and file_count != expected_member_count:
            raise ValueError(
                f"ZIP file-member count mismatch: got {file_count}, expected {expected_member_count}"
            )
        os.replace(temporary, inventory)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "archive_sha256": sha256_file(archive),
        "archive_bytes": archive.stat().st_size,
        "inventory_path": inventory.name,
        "inventory_sha256": sha256_file(inventory),
        "inventory_bytes": inventory.stat().st_size,
        "member_count": member_count,
        "file_member_count": file_count,
        "directory_member_count": directory_count,
        "total_member_uncompressed_bytes": total_uncompressed,
        "total_member_compressed_bytes": total_compressed,
        "zip_crc_and_stream_integrity": "passed",
        "duplicate_member_names": 0,
        "unsafe_member_paths": 0,
    }


def _text_wrapper(handle: IO[bytes], *, encoding: str) -> TextIO:
    return io.TextIOWrapper(handle, encoding=encoding, errors="strict", newline="")


def inspect_delimited_zip(
    archive_path: str | os.PathLike[str],
    *,
    delimiter: str = "\t",
    encoding_candidates: Sequence[str] = ("utf-8-sig", "cp1252"),
) -> dict[str, Any]:
    """Parse every non-directory ZIP member as delimited text without extraction."""

    archive = Path(archive_path)
    records: list[dict[str, Any]] = []
    with zipfile.ZipFile(archive, "r") as zipped:
        for info in zipped.infolist():
            if info.is_dir() or info.filename.startswith("__MACOSX/"):
                continue
            selected_encoding: str | None = None
            last_error: UnicodeDecodeError | None = None
            row_count = 0
            malformed_width_rows = 0
            header: list[str] = []
            malformed_width_examples: list[dict[str, Any]] = []
            for encoding in encoding_candidates:
                attempted_row_count = 0
                attempted_malformed_width_rows = 0
                attempted_header: list[str] = []
                attempted_malformed_examples: list[dict[str, Any]] = []
                try:
                    with zipped.open(info, "r") as binary, _text_wrapper(binary, encoding=encoding) as text:
                        reader = csv.reader(text, delimiter=delimiter)
                        attempted_header = next(reader, [])
                        expected_width = len(attempted_header)
                        for physical_row_number, row in enumerate(reader, start=2):
                            if not row or (len(row) == 1 and row[0] == ""):
                                continue
                            attempted_row_count += 1
                            if len(row) != expected_width:
                                attempted_malformed_width_rows += 1
                                if len(attempted_malformed_examples) < 10:
                                    attempted_malformed_examples.append(
                                        {
                                            "logical_record_number_one_based": physical_row_number,
                                            "expected_width": expected_width,
                                            "observed_width": len(row),
                                            "first_field": row[0] if row else None,
                                            "row_sha256": hashlib.sha256(
                                                canonical_json_bytes(row)
                                            ).hexdigest(),
                                        }
                                    )
                    header = attempted_header
                    row_count = attempted_row_count
                    malformed_width_rows = attempted_malformed_width_rows
                    malformed_width_examples = attempted_malformed_examples
                    selected_encoding = encoding
                    break
                except UnicodeDecodeError as exc:
                    last_error = exc
            if selected_encoding is None:
                raise ValueError(f"No declared encoding parsed {info.filename!r}") from last_error
            records.append(
                {
                    "archive_member_path": info.filename,
                    "encoding": selected_encoding,
                    "delimiter": delimiter,
                    "column_count": len(header),
                    "columns": header,
                    "data_row_count": row_count,
                    "malformed_width_rows": malformed_width_rows,
                    "malformed_width_examples": malformed_width_examples,
                    "parse_integrity": "passed" if header and malformed_width_rows == 0 else "failed",
                }
            )
    if not records:
        raise ValueError(f"No delimited file members found in {archive}")
    return {
        "parser_version": PARSER_VERSION,
        "members": records,
        "member_count": len(records),
        "total_data_rows": sum(int(record["data_row_count"]) for record in records),
        "total_malformed_width_rows": sum(int(record["malformed_width_rows"]) for record in records),
        "parse_integrity": (
            "passed" if all(record["parse_integrity"] == "passed" for record in records) else "failed"
        ),
    }


def inspect_fasta(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Validate FASTA framing and report exact record/sequence counts."""

    records = 0
    sequence_characters = 0
    blank_sequences = 0
    current_length: int | None = None
    with Path(path).open("r", encoding="utf-8", errors="strict", newline="") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith(">"):
                if current_length == 0:
                    blank_sequences += 1
                records += 1
                current_length = 0
                continue
            if current_length is None:
                raise ValueError(f"FASTA sequence before header at line {line_number}")
            if re.search(r"[^A-Za-z*.-]", stripped):
                raise ValueError(f"Invalid FASTA sequence characters at line {line_number}")
            current_length += len(stripped)
            sequence_characters += len(stripped)
    if current_length == 0:
        blank_sequences += 1
    if not records:
        raise ValueError("FASTA contains no records")
    return {
        "parser_version": PARSER_VERSION,
        "record_count": records,
        "sequence_character_count": sequence_characters,
        "blank_sequence_count": blank_sequences,
        "parse_integrity": "passed" if blank_sequences == 0 else "failed",
    }


def inspect_delimited_text(
    path: str | os.PathLike[str],
    *,
    delimiter: str = "\t",
    encoding: str = "utf-8-sig",
) -> dict[str, Any]:
    """Parse an unpacked text mapping with exact physical row accounting."""

    with Path(path).open("r", encoding=encoding, errors="strict", newline="") as handle:
        reader = csv.reader(handle, delimiter=delimiter)
        header = next(reader, [])
        row_count = 0
        malformed = 0
        for row in reader:
            if not row or (len(row) == 1 and row[0] == ""):
                continue
            row_count += 1
            malformed += int(len(row) != len(header))
    if not header:
        raise ValueError(f"Missing header: {path}")
    return {
        "parser_version": PARSER_VERSION,
        "encoding": encoding,
        "delimiter": delimiter,
        "columns": header,
        "column_count": len(header),
        "data_row_count": row_count,
        "malformed_width_rows": malformed,
        "parse_integrity": "passed" if malformed == 0 else "failed",
    }


def audit_bindingdb_articles_origin(archive_path: str | os.PathLike[str]) -> dict[str, Any]:
    """Inventory every Articles-row origin and endpoint without scientific admission."""

    endpoint_columns = (
        "Ki (nM)",
        "IC50 (nM)",
        "Kd (nM)",
        "EC50 (nM)",
        "kon (M-1-s-1)",
        "koff (s-1)",
    )
    origin_counts: dict[str, int] = {}
    endpoint_nonblank_counts = {column: 0 for column in endpoint_columns}
    reactant_ids: set[str] = set()
    duplicate_reactant_set_id_rows = 0
    missing_reactant_set_id_rows = 0
    rows_with_doi = 0
    rows_with_pmid = 0
    rows_with_patent = 0
    physical_rows = 0
    with zipfile.ZipFile(archive_path, "r") as zipped:
        members = [info for info in zipped.infolist() if not info.is_dir()]
        if len(members) != 1:
            raise ValueError("BindingDB Articles archive must contain exactly one TSV member")
        with zipped.open(members[0], "r") as binary, _text_wrapper(binary, encoding="utf-8-sig") as text:
            reader = csv.DictReader(text, delimiter="\t")
            required = {
                "BindingDB Reactant_set_id",
                "Curation/DataSource",
                "Article DOI",
                "PMID",
                "Patent Number",
                *endpoint_columns,
            }
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"BindingDB Articles missing origin columns: {sorted(missing)}")
            for row in reader:
                physical_rows += 1
                origin = str(row["Curation/DataSource"] or "").strip() or "<blank>"
                origin_counts[origin] = origin_counts.get(origin, 0) + 1
                reactant_id = str(row["BindingDB Reactant_set_id"] or "").strip()
                if not reactant_id:
                    missing_reactant_set_id_rows += 1
                elif reactant_id in reactant_ids:
                    duplicate_reactant_set_id_rows += 1
                else:
                    reactant_ids.add(reactant_id)
                for column in endpoint_columns:
                    endpoint_nonblank_counts[column] += int(bool(str(row[column] or "").strip()))
                rows_with_doi += int(bool(str(row["Article DOI"] or "").strip()))
                rows_with_pmid += int(bool(str(row["PMID"] or "").strip()))
                rows_with_patent += int(bool(str(row["Patent Number"] or "").strip()))
    disposition = {
        "Curated from the literature by BindingDB": "candidate_after_rights_and_scientific_review",
        "ChEMBL": "exclude_as_cross_source_mirror_not_independent_evidence",
        "Taylor Research Group, UCSD": "quarantine_pending_origin_and_rights_review",
    }
    unknown_origins = sorted(set(origin_counts) - set(disposition))
    return {
        "parser_version": PARSER_VERSION,
        "physical_measurement_rows": physical_rows,
        "curation_data_source_counts": dict(sorted(origin_counts.items())),
        "curation_data_source_disposition": disposition,
        "unmapped_origin_values": unknown_origins,
        "origin_allowlist_exhaustive": not unknown_origins,
        "later_candidate_bindingdb_curated_rows": origin_counts.get(
            "Curated from the literature by BindingDB", 0
        ),
        "excluded_chembl_mirror_rows": origin_counts.get("ChEMBL", 0),
        "quarantined_other_origin_rows": sum(
            count
            for origin, count in origin_counts.items()
            if origin != "Curated from the literature by BindingDB" and origin != "ChEMBL"
        ),
        "unique_reactant_set_ids": len(reactant_ids),
        "duplicate_reactant_set_id_rows": duplicate_reactant_set_id_rows,
        "missing_reactant_set_id_rows": missing_reactant_set_id_rows,
        "endpoint_nonblank_row_counts": endpoint_nonblank_counts,
        "rows_with_article_doi": rows_with_doi,
        "rows_with_pmid": rows_with_pmid,
        "rows_with_patent_number": rows_with_patent,
        "canonical_rows_admitted": 0,
        "model_labels_admitted": 0,
    }


def _source_manifest(
    source_id: str,
    *,
    release_id: str,
    release_date: str | None,
    files: Sequence[Mapping[str, Any]],
    status: str,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    document: dict[str, Any] = {
        "schema_version": ACQUISITION_SCHEMA_VERSION,
        "source_id": source_id,
        "release_id": release_id,
        "release_date": release_date,
        "parser_version": PARSER_VERSION,
        "snapshot_status": status,
        "raw_bytes_immutable": True,
        "canonical_rows_admitted": 0,
        "model_labels_admitted": 0,
        "semantic_and_rights_boundaries": SOURCE_BOUNDARIES[source_id],
        "files": [dict(record) for record in files],
        "exact_source_artifact_count": len(files),
        "exact_source_artifact_bytes": sum(int(record.get("acquired_bytes", 0)) for record in files),
        "exact_physical_file_count": len(files),
        "exact_physical_bytes": sum(int(record.get("acquired_bytes", 0)) for record in files),
    }
    if extra:
        document.update(extra)
    return document


def _bundle_artifact_role(relative_path: str, source_paths: set[str]) -> str:
    if relative_path in source_paths:
        return "source_HTTP_artifact"
    if relative_path.endswith(".acquisition.json"):
        return "source_HTTP_acquisition_sidecar"
    if relative_path.endswith(".members.jsonl"):
        return "archive_member_content_inventory"
    if relative_path.endswith("page_token_chain.jsonl"):
        return "pagination_chain_inventory"
    if relative_path.endswith("nct_membership.jsonl"):
        return "registry_cohort_membership_inventory"
    if relative_path.endswith("accession_source_membership.jsonl"):
        return "accession_source_membership_inventory"
    if relative_path.endswith("accession_resolution.jsonl"):
        return "accession_resolution_inventory"
    if relative_path.endswith(".part_manifest.json"):
        return "completed_release_part_manifest"
    if re.search(r"pages/page_[0-9]{6}\.manifest\.json$", relative_path):
        return "API_page_acquisition_metadata"
    return "derived_or_operational_acquisition_artifact"


def _source_bundle_inventory(
    source_root: Path,
    manifest_path: Path,
    source_files: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Hash every regular bundle artifact except the self-referential source manifest."""

    root = source_root.resolve()
    target = manifest_path.resolve()
    if root != target.parent and root not in target.parents:
        raise ValueError("Source manifest path escapes its declared source root")
    source_paths = {
        PurePosixPath(str(item.get("local_path", item.get("path", "")))).as_posix() for item in source_files
    }
    entries: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Symlink is prohibited in immutable source bundle: {path}")
        if not path.is_file() or path.resolve() == target:
            continue
        relative = path.relative_to(root).as_posix()
        if path.name.endswith(".part") or path.name.endswith(".part.json"):
            raise ValueError(f"Unpromoted partial is prohibited in completed source bundle: {relative}")
        entry: dict[str, Any] = {
            "path": relative,
            "artifact_role": _bundle_artifact_role(relative, source_paths),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        if path.suffix == ".jsonl":
            with path.open("rb") as handle:
                entry["line_count"] = sum(1 for _line in handle)
        entries.append(entry)
    return {
        "inventory_schema_version": ACQUISITION_SCHEMA_VERSION,
        "root": ".",
        "included_artifacts": "every regular file recursively below source root",
        "excluded_paths": [target.relative_to(root).as_posix()],
        "exclusion_reason": "source manifest is self-referential and bound by manifest_sha256",
        "symlink_policy": "rejected",
        "unpromoted_partial_policy": "rejected",
        "entries": entries,
        "entry_count": len(entries),
        "total_bytes": sum(int(entry["bytes"]) for entry in entries),
        "entries_sha256": hashlib.sha256(canonical_json_bytes(entries)).hexdigest(),
    }


def atomic_write_source_manifest(
    source_root: str | os.PathLike[str],
    manifest_path: str | os.PathLike[str],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind a completed source manifest to exact recursive bundle membership."""

    root = Path(source_root)
    destination = Path(manifest_path)
    files = manifest.get("files")
    if not isinstance(files, list):
        raise ValueError("Source manifest files must be a list before bundle finalization")
    bundle = _source_bundle_inventory(root, destination, files)
    finalized = dict(manifest)
    finalized["bundle_inventory"] = bundle
    finalized["exact_physical_file_count"] = bundle["entry_count"]
    finalized["exact_physical_bytes"] = bundle["total_bytes"]
    return atomic_write_json(destination, finalized)


def acquire_bindingdb(
    raw_root: str | os.PathLike[str],
    *,
    session: requests.Session | None = None,
) -> dict[str, Any]:
    """Acquire the five minimum BindingDB 202608 curated-articles inputs."""

    root = Path(raw_root) / "bindingdb_curated_202608"
    root.mkdir(parents=True, exist_ok=True)
    client = session or build_session(retries=8, backoff_factor=1.0)
    metadata_dir = root / "metadata"
    metadata_dir.mkdir(exist_ok=True)
    landing = snapshot_http_bytes(
        BINDINGDB_LANDING_URL,
        metadata_dir / "bindingdb_download_page.html",
        session=client,
    )
    landing = annotate_acquisition_record(
        landing,
        artifact_role="official_release_and_terms_page",
        local_path="metadata/bindingdb_download_page.html",
    )
    paper = snapshot_http_bytes(
        BINDINGDB_PRIMARY_PAPER_URL,
        metadata_dir / "bindingdb_primary_paper_pmc.html",
        session=client,
    )
    paper = annotate_acquisition_record(
        paper,
        artifact_role=(
            "primary_citation_landing_access_challenge_evidence"
            if "Checking your browser"
            in (metadata_dir / "bindingdb_primary_paper_pmc.html").read_text(encoding="utf-8")
            else "primary_citation_landing_snapshot"
        ),
        local_path="metadata/bindingdb_primary_paper_pmc.html",
    )
    paper_xml = snapshot_http_bytes(
        BINDINGDB_PRIMARY_PAPER_XML_URL,
        metadata_dir / "bindingdb_primary_paper_pmc.xml",
        session=client,
    )
    paper_xml = annotate_acquisition_record(
        paper_xml,
        artifact_role="primary_citation_full_text_XML_snapshot",
        local_path="metadata/bindingdb_primary_paper_pmc.xml",
    )
    paper_xml_text = (metadata_dir / "bindingdb_primary_paper_pmc.xml").read_text(encoding="utf-8")
    if "PMC11701568" not in paper_xml_text or "gkae1075" not in paper_xml_text:
        raise ValueError("BindingDB primary-paper XML failed article identity validation")

    md5_records: list[dict[str, Any]] = []
    published_md5: dict[str, str] = {}
    for payload_filename, md5_filename in BINDINGDB_PUBLISHED_MD5_FILES.items():
        md5_path = metadata_dir / md5_filename
        md5_record = snapshot_http_bytes(
            f"{BINDINGDB_DOWNLOAD_ROOT}/downloads/{md5_filename}",
            md5_path,
            session=client,
        )
        md5_record = annotate_acquisition_record(
            md5_record,
            artifact_role="published_upstream_MD5",
            local_path=f"metadata/{md5_filename}",
        )
        matches = set(re.findall(r"(?i)(?<![0-9a-f])[0-9a-f]{32}(?![0-9a-f])", md5_path.read_text()))
        if len(matches) != 1:
            raise ValueError(f"BindingDB published MD5 file is ambiguous: {md5_filename}")
        published_md5[payload_filename] = next(iter(matches)).casefold()
        md5_records.append(md5_record)
    records: list[dict[str, Any]] = []
    parse_records: list[dict[str, Any]] = []
    for filename in BINDINGDB_FILES:
        subpath = f"downloads/{filename}" if filename.endswith(".zip") else filename
        url = f"{BINDINGDB_DOWNLOAD_ROOT}/{subpath}"
        path = root / filename
        record = download_immutable(
            DownloadSpec(canonical_url=url, destination=path, expected_filename=filename),
            session=client,
        )
        record = annotate_acquisition_record(record, artifact_role="required_release_payload")
        if filename.endswith(".zip"):
            expected_md5 = published_md5[filename]
            actual_md5 = md5_file(path)
            if actual_md5 != expected_md5:
                raise ValueError(
                    f"BindingDB published MD5 mismatch for {filename}: {actual_md5} != {expected_md5}"
                )
            record["upstream_md5_when_published"] = expected_md5
            record["acquired_md5"] = actual_md5
            record["published_md5_verification"] = "passed"
            inventory = root / f"{filename}.members.jsonl"
            zip_record = verify_zip_archive(path, inventory)
            parse = inspect_delimited_zip(path)
            record["archive_integrity"] = zip_record
            parse_records.append({"file": filename, **parse})
        elif filename.endswith(".fasta"):
            parse_records.append({"file": filename, **inspect_fasta(path)})
        else:
            parse_records.append({"file": filename, **inspect_delimited_text(path)})
        records.append(record)
    origin_audit = audit_bindingdb_articles_origin(root / "BindingDB_BindingDB_Articles_202608_tsv.zip")
    article_parse_rows = next(
        int(item["total_data_rows"])
        for item in parse_records
        if item["file"] == "BindingDB_BindingDB_Articles_202608_tsv.zip"
    )
    if int(origin_audit["physical_measurement_rows"]) != article_parse_rows:
        raise ValueError("BindingDB Articles origin audit does not reconcile to parsed rows")
    status = (
        "complete_raw_origin_mixture_detected_no_rows_admitted"
        if all(item["parse_integrity"] == "passed" for item in parse_records)
        and origin_audit["origin_allowlist_exhaustive"]
        else "failed"
    )
    manifest = _source_manifest(
        "bindingdb_curated_202608",
        release_id=BINDINGDB_RELEASE_ID,
        release_date=BINDINGDB_RELEASE_NOTE_DATE,
        files=[landing, paper, paper_xml, *md5_records, *records],
        status=status,
        extra={
            "official_landing_url": BINDINGDB_LANDING_URL,
            "landing_page_snapshot": landing,
            "primary_paper_XML_snapshot": paper_xml,
            "primary_paper_landing_snapshot_status": paper["artifact_role"],
            "minimum_snapshot_scope": (
                "Complete official BindingDB Articles payload plus assay, reaction-set mapping, "
                "target FASTA, and polymer-to-UniProt mapping bytes. The Articles payload is "
                "origin-mixed; no row was canonically admitted."
            ),
            "source_origin_allowlist_for_later_review": ["Curated from the literature by BindingDB"],
            "source_origin_exclusions_for_later_review": [
                "ChEMBL (cross-source mirror; not independent evidence)",
                "Taylor Research Group, UCSD (quarantine pending rights/origin review)",
            ],
            "parse_inventory": parse_records,
            "articles_origin_and_endpoint_audit": origin_audit,
            "query_or_selection_digest": hashlib.sha256(
                canonical_json_bytes({"release": BINDINGDB_RELEASE_ID, "files": BINDINGDB_FILES})
            ).hexdigest(),
        },
    )
    return atomic_write_source_manifest(
        root,
        root / "bindingdb_curated_202608_manifest.json",
        manifest,
    )


def _acquire_clinicaltrials_legacy_metadata(
    raw_root: str | os.PathLike[str],
    *,
    session: requests.Session | None = None,
) -> dict[str, Any]:
    """Freeze API version, OpenAPI schema, and terms before any intervention query."""

    root = Path(raw_root) / "clinicaltrials_gov_v2"
    root.mkdir(parents=True, exist_ok=True)
    client = session or build_session(retries=8, backoff_factor=1.0)
    records: list[dict[str, Any]] = []
    for url, path, role in (
        (CLINICALTRIALS_VERSION_URL, root / "version.json", "api_version_response"),
        (CLINICALTRIALS_API_DOCS_URL, root / "api_documentation.html", "official_api_documentation"),
        (
            CLINICALTRIALS_STUDY_METADATA_URL,
            root / "studies_metadata.json",
            "authoritative_live_study_schema_metadata",
        ),
        (CLINICALTRIALS_TERMS_URL, root / "terms-conditions.html", "official_terms_and_conditions"),
    ):
        record = snapshot_http_bytes(url, path, session=client)
        record = annotate_acquisition_record(record, artifact_role=role)
        records.append(record)
    openapi_available = True
    openapi_error: dict[str, Any] | None = None
    error_path = root / "ctg-oas-v2.http-error"
    error_manifest_path = root / "ctg-oas-v2.http-error.acquisition.json"
    if error_path.exists() and error_manifest_path.exists():
        openapi_available = False
        openapi_error = json.loads(error_manifest_path.read_text(encoding="utf-8"))
        if not verify_document_sha256(openapi_error):
            raise ValueError("ClinicalTrials.gov OpenAPI failure manifest identity failed")
        if int(openapi_error.get("acquired_bytes", -1)) != error_path.stat().st_size:
            raise ValueError("ClinicalTrials.gov OpenAPI failure byte count changed")
        if openapi_error.get("acquired_sha256") != sha256_file(error_path):
            raise ValueError("ClinicalTrials.gov OpenAPI failure bytes changed")
        records.append(openapi_error)
    else:
        try:
            openapi_record = snapshot_http_bytes(
                CLINICALTRIALS_OPENAPI_URL,
                root / "ctg-oas-v2.yaml",
                session=client,
            )
            openapi_record = annotate_acquisition_record(
                openapi_record,
                artifact_role="frozen_openapi_schema",
            )
            records.append(openapi_record)
        except requests.HTTPError:
            openapi_available = False
            response = client.get(CLINICALTRIALS_OPENAPI_URL, timeout=(15, 180))
            raw = response.content
            _atomic_write_bytes(error_path, raw)
            openapi_error = {
                "schema_version": ACQUISITION_SCHEMA_VERSION,
                "canonical_url": CLINICALTRIALS_OPENAPI_URL,
                "resolved_url": str(response.url),
                "response_status": response.status_code,
                "response_headers": _safe_headers(response.headers),
                "retrieval_utc": utc_now(),
                "acquisition_status": "official_documented_endpoint_unavailable",
                "artifact_role": "openapi_endpoint_failure_evidence",
                "local_path": error_path.name,
                "acquired_bytes": len(raw),
                "acquired_sha256": hashlib.sha256(raw).hexdigest(),
            }
            openapi_error = atomic_write_json(error_manifest_path, openapi_error)
            records.append(openapi_error)
    version = json.loads((root / "version.json").read_text(encoding="utf-8"))
    api_version = str(version.get("apiVersion", ""))
    data_timestamp = str(version.get("dataTimestamp", ""))
    if not api_version.startswith("2.") or not data_timestamp:
        raise ValueError("ClinicalTrials.gov version response lacks API v2/dataTimestamp")
    study_metadata = json.loads((root / "studies_metadata.json").read_text(encoding="utf-8"))
    if not isinstance(study_metadata, (dict, list)) or not study_metadata:
        raise ValueError("ClinicalTrials.gov study metadata snapshot failed structural validation")
    if openapi_available:
        schema_text = (root / "ctg-oas-v2.yaml").read_text(encoding="utf-8")
        if "openapi:" not in schema_text or "/studies" not in schema_text:
            raise ValueError("ClinicalTrials.gov OpenAPI snapshot failed structural validation")
    manifest = _source_manifest(
        "clinicaltrials_gov_v2",
        release_id=f"api-{api_version}",
        release_date=data_timestamp,
        files=records,
        status=(
            "metadata_frozen_openapi_endpoint_unavailable_query_blocked"
            if not openapi_available
            else "metadata_frozen_query_blocked"
        ),
        extra={
            "api_major": "v2",
            "api_version": api_version,
            "dataTimestamp": data_timestamp,
            "openapi_snapshot_status": "available" if openapi_available else "documented_endpoint_http_error",
            "openapi_sha256": sha256_file(root / "ctg-oas-v2.yaml") if openapi_available else None,
            "openapi_failure_evidence": openapi_error,
            "study_metadata_sha256": sha256_file(root / "studies_metadata.json"),
            "version_response_sha256": sha256_file(root / "version.json"),
            "terms_snapshot_content_status": (
                "SPA_shell_only_identical_to_api_documentation_HTML; terms text requires "
                "human browser or separate official static source review"
                if sha256_file(root / "terms-conditions.html") == sha256_file(root / "api_documentation.html")
                else "distinct_HTTP_bytes_frozen"
            ),
            "targeted_study_query_status": "not_run",
            "targeted_study_query_blocker": (
                "This metadata-only entrypoint intentionally stops before study retrieval. "
                "The separate broad DRUG-intervention cohort entrypoint performs no molecule "
                "alias or identity mapping."
            ),
            "absence_semantics": "no_record_and_no_posted_results_are_not_negative_outcomes",
        },
    )
    return atomic_write_json(root / "clinicaltrials_gov_v2_manifest.json", manifest)


def _clinicaltrials_metadata_piece_names(metadata: Any) -> set[str]:
    """Return every API field ``piece`` declared by the frozen study metadata."""

    pieces: set[str] = set()
    if isinstance(metadata, dict):
        piece = metadata.get("piece")
        if isinstance(piece, str) and piece:
            pieces.add(piece)
        for value in metadata.values():
            pieces.update(_clinicaltrials_metadata_piece_names(value))
    elif isinstance(metadata, list):
        for value in metadata:
            pieces.update(_clinicaltrials_metadata_piece_names(value))
    return pieces


def _write_jsonl_records(
    path: Path,
    records: Iterable[Mapping[str, Any]],
    *,
    relative_to: Path | None = None,
) -> dict[str, Any]:
    """Atomically write canonical JSONL and return its exact physical inventory."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    rows = 0
    try:
        with os.fdopen(descriptor, "wb") as output:
            for record in records:
                output.write(canonical_json_bytes(record))
                output.write(b"\n")
                rows += 1
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "path": path.relative_to(relative_to).as_posix() if relative_to else path.name,
        "rows": rows,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _clinicaltrials_study_audit(
    study: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one projected registry study without interpreting it as an outcome."""

    protocol = study.get("protocolSection")
    if not isinstance(protocol, dict):
        raise ValueError("ClinicalTrials.gov study lacks protocolSection")
    identification = protocol.get("identificationModule")
    if not isinstance(identification, dict):
        raise ValueError("ClinicalTrials.gov study lacks identificationModule")
    nct_id = str(identification.get("nctId", ""))
    if not re.fullmatch(r"NCT[0-9]{8}", nct_id):
        raise ValueError(f"ClinicalTrials.gov study has invalid NCT ID: {nct_id!r}")
    arms = protocol.get("armsInterventionsModule")
    interventions = arms.get("interventions", []) if isinstance(arms, dict) else []
    if not isinstance(interventions, list):
        raise ValueError(f"ClinicalTrials.gov interventions are not a list: {nct_id}")
    intervention_types: list[str] = []
    for intervention in interventions:
        if not isinstance(intervention, dict):
            raise ValueError(f"ClinicalTrials.gov intervention is not an object: {nct_id}")
        intervention_types.append(str(intervention.get("type", "")))
    if "DRUG" not in intervention_types:
        raise ValueError(
            f"ClinicalTrials.gov DRUG query returned a study without a DRUG intervention: {nct_id}"
        )

    status_module = protocol.get("statusModule")
    status_module = status_module if isinstance(status_module, dict) else {}
    design_module = protocol.get("designModule")
    design_module = design_module if isinstance(design_module, dict) else {}
    phases = design_module.get("phases", [])
    if not isinstance(phases, list):
        phases = []
    has_results = study.get("hasResults")
    if has_results not in (True, False, None):
        raise ValueError(f"ClinicalTrials.gov HasResults is not boolean/null: {nct_id}")
    return {
        "nct_id": nct_id,
        "study_sha256": hashlib.sha256(canonical_json_bytes(study)).hexdigest(),
        "intervention_count": len(interventions),
        "drug_intervention_count": intervention_types.count("DRUG"),
        "has_results": has_results,
        "overall_status": status_module.get("overallStatus"),
        "study_type": design_module.get("studyType"),
        "phases": [str(value) for value in phases],
        "last_update_post_date": (
            status_module.get("lastUpdatePostDateStruct", {}).get("date")
            if isinstance(status_module.get("lastUpdatePostDateStruct"), dict)
            else None
        ),
    }


def _clinicaltrials_version_identity(path: Path) -> dict[str, str]:
    """Parse the exact API version and data timestamp from one frozen probe."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"ClinicalTrials.gov version response is not an object: {path}")
    identity = {
        "apiVersion": str(payload.get("apiVersion", "")),
        "dataTimestamp": str(payload.get("dataTimestamp", "")),
    }
    if not identity["apiVersion"].startswith("2.") or not identity["dataTimestamp"]:
        raise ValueError(f"ClinicalTrials.gov version identity is incomplete: {path}")
    return identity


def _clinicaltrials_snapshot_key(identity: Mapping[str, str]) -> str:
    """Create a filesystem-safe cohort key from the exact version identity."""

    version = re.sub(r"[^A-Za-z0-9._-]+", "-", identity["apiVersion"]).strip("-")
    timestamp = re.sub(r"[^A-Za-z0-9._-]+", "-", identity["dataTimestamp"]).strip("-")
    return f"api-{version}__data-{timestamp}"


def _next_numbered_snapshot_path(directory: Path, prefix: str) -> Path:
    """Return a never-before-used numbered snapshot path in ``directory``."""

    pattern = re.compile(rf"{re.escape(prefix)}_([0-9]{{6}})\.json")
    indices = [
        int(match.group(1))
        for path in directory.glob(f"{prefix}_*.json")
        if (match := pattern.fullmatch(path.name)) is not None
    ]
    return directory / f"{prefix}_{max(indices, default=-1) + 1:06d}.json"


def _load_clinicaltrials_version_records(
    root: Path,
    cohort_root: Path,
    prefix: str,
    artifact_role: str,
    client: requests.Session,
) -> list[dict[str, Any]]:
    """Reverify every numbered immutable version probe and its sidecar."""

    records: list[dict[str, Any]] = []
    pattern = re.compile(rf"{re.escape(prefix)}_[0-9]{{6}}\.json")
    for path in sorted(item for item in cohort_root.iterdir() if pattern.fullmatch(item.name)):
        record = download_immutable(
            DownloadSpec(canonical_url=CLINICALTRIALS_VERSION_URL, destination=path),
            session=client,
        )
        records.append(
            annotate_acquisition_record(
                record,
                artifact_role=artifact_role,
                local_path=path.relative_to(root).as_posix(),
            )
        )
    return records


def acquire_clinicaltrials_metadata(
    raw_root: str | os.PathLike[str],
    *,
    session: requests.Session | None = None,
    update_current_pointer: bool = True,
) -> dict[str, Any]:
    """Freeze a fresh, version-keyed API metadata snapshot without deleting history."""

    root = Path(raw_root) / "clinicaltrials_gov_v2"
    root.mkdir(parents=True, exist_ok=True)
    client = session or build_session(retries=8, backoff_factor=1.0)

    discovery_root = root / "version_discovery_probes"
    discovery_root.mkdir(exist_ok=True)
    discovery_path = _next_numbered_snapshot_path(discovery_root, "version_probe")
    discovery = snapshot_http_bytes(
        CLINICALTRIALS_VERSION_URL,
        discovery_path,
        session=client,
    )
    discovery = annotate_acquisition_record(
        discovery,
        artifact_role="fresh_API_version_discovery_probe",
        local_path=discovery_path.relative_to(root).as_posix(),
    )
    identity = _clinicaltrials_version_identity(discovery_path)
    snapshot_key = _clinicaltrials_snapshot_key(identity)
    snapshot_root = root / "metadata_snapshots" / snapshot_key
    snapshot_root.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    for url, filename, role in (
        (CLINICALTRIALS_VERSION_URL, "version.json", "version_keyed_api_version_response"),
        (
            CLINICALTRIALS_API_DOCS_URL,
            "api_documentation.html",
            "version_keyed_official_api_documentation",
        ),
        (
            CLINICALTRIALS_STUDY_METADATA_URL,
            "studies_metadata.json",
            "version_keyed_authoritative_live_study_schema_metadata",
        ),
        (
            CLINICALTRIALS_TERMS_URL,
            "terms-conditions.html",
            "version_keyed_official_terms_and_conditions",
        ),
    ):
        path = snapshot_root / filename
        record = snapshot_http_bytes(url, path, session=client)
        records.append(
            annotate_acquisition_record(
                record,
                artifact_role=role,
                local_path=path.relative_to(root).as_posix(),
            )
        )
    version_path = snapshot_root / "version.json"
    if _clinicaltrials_version_identity(version_path) != identity:
        raise ValueError("ClinicalTrials.gov discovery and version-keyed metadata identities differ")

    openapi_available = True
    openapi_error: dict[str, Any] | None = None
    openapi_path = snapshot_root / "ctg-oas-v2.yaml"
    error_path = snapshot_root / "ctg-oas-v2.http-error"
    error_manifest_path = snapshot_root / "ctg-oas-v2.http-error.acquisition.json"
    if error_path.exists() and error_manifest_path.exists():
        openapi_available = False
        openapi_error = json.loads(error_manifest_path.read_text(encoding="utf-8"))
        if not verify_document_sha256(openapi_error):
            raise ValueError("ClinicalTrials.gov versioned OpenAPI failure manifest failed")
        if int(openapi_error.get("acquired_bytes", -1)) != error_path.stat().st_size:
            raise ValueError("ClinicalTrials.gov versioned OpenAPI failure bytes changed")
        if openapi_error.get("acquired_sha256") != sha256_file(error_path):
            raise ValueError("ClinicalTrials.gov versioned OpenAPI failure hash changed")
        records.append(openapi_error)
    else:
        try:
            openapi = snapshot_http_bytes(
                CLINICALTRIALS_OPENAPI_URL,
                openapi_path,
                session=client,
            )
            records.append(
                annotate_acquisition_record(
                    openapi,
                    artifact_role="version_keyed_frozen_openapi_schema",
                    local_path=openapi_path.relative_to(root).as_posix(),
                )
            )
        except requests.HTTPError:
            openapi_available = False
            response = client.get(CLINICALTRIALS_OPENAPI_URL, timeout=(15, 180))
            raw = response.content
            _atomic_write_bytes(error_path, raw)
            openapi_error = atomic_write_json(
                error_manifest_path,
                {
                    "schema_version": ACQUISITION_SCHEMA_VERSION,
                    "canonical_url": CLINICALTRIALS_OPENAPI_URL,
                    "resolved_url": str(response.url),
                    "response_status": response.status_code,
                    "response_headers": _safe_headers(response.headers),
                    "retrieval_utc": utc_now(),
                    "acquisition_status": "official_documented_endpoint_unavailable",
                    "artifact_role": "version_keyed_openapi_endpoint_failure_evidence",
                    "local_path": error_path.relative_to(root).as_posix(),
                    "acquired_bytes": len(raw),
                    "acquired_sha256": hashlib.sha256(raw).hexdigest(),
                },
            )
            records.append(openapi_error)

    metadata_path = snapshot_root / "studies_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, (dict, list)) or not metadata:
        raise ValueError("ClinicalTrials.gov version-keyed metadata failed structural validation")
    if openapi_available:
        schema_text = openapi_path.read_text(encoding="utf-8")
        if "openapi:" not in schema_text or "/studies" not in schema_text:
            raise ValueError("ClinicalTrials.gov version-keyed OpenAPI schema failed validation")
    api_docs_path = snapshot_root / "api_documentation.html"
    terms_path = snapshot_root / "terms-conditions.html"
    manifest = _source_manifest(
        "clinicaltrials_gov_v2",
        release_id=f"api-{identity['apiVersion']}",
        release_date=identity["dataTimestamp"],
        files=[discovery, *records],
        status="version_keyed_metadata_frozen_query_not_run",
        extra={
            "api_major": "v2",
            "api_version": identity["apiVersion"],
            "dataTimestamp": identity["dataTimestamp"],
            "metadata_snapshot_key": snapshot_key,
            "metadata_snapshot_path": snapshot_root.relative_to(root).as_posix(),
            "study_metadata_local_path": metadata_path.relative_to(root).as_posix(),
            "openapi_snapshot_status": (
                "available" if openapi_available else "documented_endpoint_http_error"
            ),
            "openapi_failure_evidence": openapi_error,
            "study_metadata_sha256": sha256_file(metadata_path),
            "version_response_sha256": sha256_file(version_path),
            "fresh_version_discovery_sha256": discovery["acquired_sha256"],
            "terms_snapshot_content_status": (
                "SPA_shell_only_identical_to_api_documentation_HTML; terms text requires "
                "human browser or separate official static source review"
                if sha256_file(terms_path) == sha256_file(api_docs_path)
                else "distinct_HTTP_bytes_frozen"
            ),
            "targeted_study_query_status": "not_run",
            "targeted_study_query_blocker": (
                "Metadata-only acquisition is complete; cohort retrieval is a separate stage."
            ),
            "absence_semantics": "no_record_and_no_posted_results_are_not_negative_outcomes",
        },
    )
    core_manifest_path = snapshot_root / "metadata_manifest.json"
    core_manifest = atomic_write_json(core_manifest_path, manifest)
    history_root = root / "metadata_history"
    history_root.mkdir(exist_ok=True)
    history_path = history_root / f"{discovery_path.stem}.manifest.json"
    history_manifest = atomic_write_json(
        history_path,
        {
            **core_manifest,
            "versioned_core_manifest_path": core_manifest_path.relative_to(root).as_posix(),
            "versioned_core_manifest_sha256": core_manifest["manifest_sha256"],
            "history_manifest_path": history_path.relative_to(root).as_posix(),
        },
    )
    if update_current_pointer:
        return atomic_write_json(root / "clinicaltrials_gov_v2_manifest.json", history_manifest)
    return history_manifest


def acquire_clinicaltrials_drug_cohort(
    raw_root: str | os.PathLike[str],
    *,
    session: requests.Session | None = None,
    page_size: int = CLINICALTRIALS_PAGE_SIZE,
) -> dict[str, Any]:
    """Freeze the complete unmapped API-v2 cohort containing a DRUG intervention.

    The query intentionally operates on the registry's InterventionType enum,
    not guessed molecule aliases.  It projects registry status, dates, sponsor,
    condition, and intervention text only; no outcome measure or adverse-event
    result is fetched or admitted as a molecular label.
    """

    if page_size < 1 or page_size > CLINICALTRIALS_PAGE_SIZE:
        raise ValueError(f"ClinicalTrials.gov page_size must be in 1..{CLINICALTRIALS_PAGE_SIZE}")
    client = session or build_session(retries=10, backoff_factor=1.0)
    metadata_manifest = acquire_clinicaltrials_metadata(
        raw_root,
        session=client,
        update_current_pointer=False,
    )
    root = Path(raw_root) / "clinicaltrials_gov_v2"
    metadata_identity = {
        "apiVersion": str(metadata_manifest["api_version"]),
        "dataTimestamp": str(metadata_manifest["dataTimestamp"]),
    }
    cohort_snapshot_key = _clinicaltrials_snapshot_key(metadata_identity)
    cohort_root = root / "drug_intervention_cohorts" / cohort_snapshot_key
    pages_root = cohort_root / "pages"
    pages_root.mkdir(parents=True, exist_ok=True)

    metadata_path = root / str(metadata_manifest["study_metadata_local_path"])
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    available_pieces = _clinicaltrials_metadata_piece_names(metadata)
    missing_pieces = sorted(set(CLINICALTRIALS_DRUG_FIELDS) - available_pieces)
    if missing_pieces:
        raise ValueError(
            "ClinicalTrials.gov requested fields are absent from frozen metadata: "
            + ", ".join(missing_pieces)
        )

    before_path = cohort_root / "version_before.json"
    before_already_frozen = before_path.exists()
    before_record = snapshot_http_bytes(CLINICALTRIALS_VERSION_URL, before_path, session=client)
    before_record = annotate_acquisition_record(
        before_record,
        artifact_role="drug_cohort_api_version_before",
        local_path=before_path.relative_to(root).as_posix(),
    )
    before_identity = _clinicaltrials_version_identity(before_path)
    if before_identity != metadata_identity:
        raise ValueError("ClinicalTrials.gov metadata and cohort pre-version snapshots differ")

    if before_already_frozen:
        resume_probe_path = _next_numbered_snapshot_path(cohort_root, "version_resume_probe")
        snapshot_http_bytes(
            CLINICALTRIALS_VERSION_URL,
            resume_probe_path,
            session=client,
        )
        live_resume_identity = _clinicaltrials_version_identity(resume_probe_path)
        if live_resume_identity != before_identity:
            live_key = _clinicaltrials_snapshot_key(live_resume_identity)
            raise ValueError(
                "ClinicalTrials.gov live resume version differs from the frozen cohort; "
                f"refusing to append pages to {cohort_snapshot_key}. Start a new "
                f"version-keyed cohort {live_key}."
            )
    resume_probe_records = _load_clinicaltrials_version_records(
        root,
        cohort_root,
        "version_resume_probe",
        "drug_cohort_live_version_before_resume",
        client,
    )
    for probe_record in resume_probe_records:
        probe_path = root / str(probe_record["local_path"])
        if _clinicaltrials_version_identity(probe_path) != before_identity:
            raise ValueError("ClinicalTrials.gov persisted resume probe differs from cohort version")

    query_contract = {
        "endpoint": CLINICALTRIALS_STUDIES_URL,
        "query.term": CLINICALTRIALS_DRUG_QUERY,
        "format": "json",
        "pageSize": page_size,
        "countTotal": True,
        "sort": CLINICALTRIALS_DRUG_SORT,
        "fields": list(CLINICALTRIALS_DRUG_FIELDS),
        "cohort_membership_rule": (
            "API query membership plus independent validation that every returned study has at "
            "least one protocolSection.armsInterventionsModule.interventions.type == DRUG"
        ),
    }
    query_digest = hashlib.sha256(canonical_json_bytes(query_contract)).hexdigest()
    page_records: list[dict[str, Any]] = []
    chain_records: list[dict[str, Any]] = []
    nct_membership: list[dict[str, Any]] = []
    seen_nct_ids: set[str] = set()
    seen_page_tokens: set[str] = set()
    total_count: int | None = None
    next_page_token: str | None = None
    status_counts: Counter[str] = Counter()
    study_type_counts: Counter[str] = Counter()
    phase_counts: Counter[str] = Counter()
    has_results_counts: Counter[str] = Counter()
    total_interventions = 0
    total_drug_interventions = 0
    concatenated_page_digest = hashlib.sha256()

    for page_index in range(100_000):
        params: dict[str, str] = {
            "query.term": CLINICALTRIALS_DRUG_QUERY,
            "format": "json",
            "pageSize": str(page_size),
            "countTotal": "true",
            "sort": CLINICALTRIALS_DRUG_SORT,
            "fields": ",".join(CLINICALTRIALS_DRUG_FIELDS),
        }
        input_page_token = next_page_token
        if input_page_token is not None:
            if input_page_token in seen_page_tokens:
                raise ValueError("ClinicalTrials.gov page-token cycle detected")
            seen_page_tokens.add(input_page_token)
            params["pageToken"] = input_page_token
        request_url = f"{CLINICALTRIALS_STUDIES_URL}?{urlencode(params)}"
        page_path = pages_root / f"page_{page_index:06d}.json"
        acquisition = download_immutable(
            DownloadSpec(canonical_url=request_url, destination=page_path),
            session=client,
            timeout=(20.0, 300.0),
        )
        page_record = annotate_acquisition_record(
            acquisition,
            artifact_role="complete_drug_intervention_cohort_page",
            local_path=page_path.relative_to(root).as_posix(),
        )
        raw = page_path.read_bytes()
        concatenated_page_digest.update(raw)
        try:
            payload = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid ClinicalTrials.gov JSON page: {page_path}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"ClinicalTrials.gov page is not a JSON object: {page_path}")
        studies = payload.get("studies")
        if not isinstance(studies, list):
            raise ValueError(f"ClinicalTrials.gov page lacks studies list: {page_path}")
        page_total = payload.get("totalCount")
        if total_count is None:
            if not isinstance(page_total, int) or page_total < 0:
                raise ValueError(f"ClinicalTrials.gov first page lacks nonnegative totalCount: {page_path}")
            total_count = page_total
        elif page_total is not None and page_total != total_count:
            raise ValueError("ClinicalTrials.gov totalCount changed across cohort pages")
        if len(studies) > page_size:
            raise ValueError(f"ClinicalTrials.gov page exceeds declared pageSize: {page_path}")
        if total_count and not studies:
            raise ValueError(f"ClinicalTrials.gov returned an empty page before completion: {page_path}")

        first_nct: str | None = None
        last_nct: str | None = None
        for study_index, study in enumerate(studies):
            if not isinstance(study, dict):
                raise ValueError(f"ClinicalTrials.gov study is not an object: {page_path}")
            audit = _clinicaltrials_study_audit(study)
            nct_id = str(audit["nct_id"])
            if nct_id in seen_nct_ids:
                raise ValueError(f"Duplicate ClinicalTrials.gov cohort NCT ID: {nct_id}")
            seen_nct_ids.add(nct_id)
            first_nct = first_nct or nct_id
            last_nct = nct_id
            total_interventions += int(audit["intervention_count"])
            total_drug_interventions += int(audit["drug_intervention_count"])
            status_counts[str(audit["overall_status"] or "MISSING")] += 1
            study_type_counts[str(audit["study_type"] or "MISSING")] += 1
            for phase in audit["phases"]:
                phase_counts[str(phase)] += 1
            has_results_counts[
                "true"
                if audit["has_results"] is True
                else "false"
                if audit["has_results"] is False
                else "missing"
            ] += 1
            nct_membership.append(
                {
                    "nct_id": nct_id,
                    "page_index": page_index,
                    "study_index_within_page": study_index,
                    "study_sha256": audit["study_sha256"],
                }
            )

        output_page_token_value = payload.get("nextPageToken")
        if output_page_token_value is not None and not isinstance(output_page_token_value, str):
            raise ValueError(f"ClinicalTrials.gov nextPageToken is not text: {page_path}")
        output_page_token = output_page_token_value or None
        page_record.update(
            {
                "page_index": page_index,
                "query_contract_sha256": query_digest,
                "input_page_token": input_page_token,
                "output_page_token": output_page_token,
                "study_count": len(studies),
                "reported_total_count": total_count,
                "first_nct_id": first_nct,
                "last_nct_id": last_nct,
            }
        )
        page_records.append(page_record)
        chain_records.append(
            {
                "page_index": page_index,
                "request_url_sha256": hashlib.sha256(request_url.encode("utf-8")).hexdigest(),
                "input_page_token": input_page_token,
                "output_page_token": output_page_token,
                "page_path": page_path.relative_to(root).as_posix(),
                "page_bytes": len(raw),
                "page_sha256": hashlib.sha256(raw).hexdigest(),
                "study_count": len(studies),
                "first_nct_id": first_nct,
                "last_nct_id": last_nct,
            }
        )
        next_page_token = output_page_token
        if next_page_token is None:
            break
        if total_count is not None and page_index + 1 > (total_count // page_size) + 2:
            raise ValueError("ClinicalTrials.gov pagination exceeded totalCount-derived bound")
    else:  # pragma: no cover - safety guard
        raise ValueError("ClinicalTrials.gov pagination exceeded hard page limit")

    if total_count is None or len(seen_nct_ids) != total_count:
        raise ValueError(
            "ClinicalTrials.gov unique cohort membership does not reconcile to totalCount: "
            f"{len(seen_nct_ids)} != {total_count}"
        )
    expected_page_files = {pages_root / f"page_{index:06d}.json" for index in range(len(page_records))}
    observed_page_files = {
        path for path in pages_root.iterdir() if re.fullmatch(r"page_[0-9]{6}\.json", path.name)
    }
    if observed_page_files != expected_page_files:
        raise ValueError("ClinicalTrials.gov page directory contains an unexpected page set")

    after_path = _next_numbered_snapshot_path(cohort_root, "version_after_probe")
    after_record = snapshot_http_bytes(
        CLINICALTRIALS_VERSION_URL,
        after_path,
        session=client,
    )
    after_record = annotate_acquisition_record(
        after_record,
        artifact_role="drug_cohort_api_version_after",
        local_path=after_path.relative_to(root).as_posix(),
    )
    after_identity = _clinicaltrials_version_identity(after_path)
    if after_identity != before_identity:
        raise ValueError("ClinicalTrials.gov API version/dataTimestamp changed during cohort pagination")
    after_probe_records = _load_clinicaltrials_version_records(
        root,
        cohort_root,
        "version_after_probe",
        "drug_cohort_api_version_after",
        client,
    )
    for probe_record in after_probe_records:
        probe_path = root / str(probe_record["local_path"])
        if _clinicaltrials_version_identity(probe_path) != before_identity:
            raise ValueError("ClinicalTrials.gov persisted post-page probe differs from cohort version")

    chain_inventory = _write_jsonl_records(
        cohort_root / "page_token_chain.jsonl",
        chain_records,
        relative_to=root,
    )
    membership_inventory = _write_jsonl_records(
        cohort_root / "nct_membership.jsonl",
        nct_membership,
        relative_to=root,
    )
    manifest = _source_manifest(
        "clinicaltrials_gov_v2",
        release_id=f"api-{before_identity['apiVersion']}",
        release_date=before_identity["dataTimestamp"],
        files=[
            *metadata_manifest["files"],
            before_record,
            *resume_probe_records,
            *page_records,
            *after_probe_records,
        ],
        status="complete_unmapped_drug_intervention_cohort",
        extra={
            "api_major": "v2",
            "api_version": before_identity["apiVersion"],
            "dataTimestamp": before_identity["dataTimestamp"],
            "metadata_manifest_precohort_sha256": metadata_manifest["manifest_sha256"],
            "openapi_snapshot_status": metadata_manifest["openapi_snapshot_status"],
            "openapi_failure_evidence": metadata_manifest["openapi_failure_evidence"],
            "study_metadata_sha256": metadata_manifest["study_metadata_sha256"],
            "version_response_sha256": metadata_manifest["version_response_sha256"],
            "terms_snapshot_content_status": metadata_manifest["terms_snapshot_content_status"],
            "targeted_study_query_status": "complete_unmapped_drug_intervention_cohort",
            "query_contract": query_contract,
            "query_contract_sha256": query_digest,
            "pre_and_post_version_identity_match": True,
            "cohort_snapshot_key": cohort_snapshot_key,
            "fresh_resume_version_probe_count": len(resume_probe_records),
            "fresh_post_pagination_version_probe_count": len(after_probe_records),
            "page_count": len(page_records),
            "page_token_chain_inventory": chain_inventory,
            "nct_membership_inventory": membership_inventory,
            "reported_total_count": total_count,
            "unique_nct_count": len(seen_nct_ids),
            "total_intervention_records": total_interventions,
            "drug_intervention_records": total_drug_interventions,
            "registry_has_results_counts": dict(sorted(has_results_counts.items())),
            "registry_overall_status_counts": dict(sorted(status_counts.items())),
            "registry_study_type_counts": dict(sorted(study_type_counts.items())),
            "registry_phase_counts": dict(sorted(phase_counts.items())),
            "concatenated_raw_page_bytes_sha256": concatenated_page_digest.hexdigest(),
            "molecule_alias_and_identity_mapping_status": "not_attempted",
            "outcome_and_adverse_event_result_fields_retrieved": False,
            "absence_semantics": "no_record_and_no_posted_results_are_not_negative_outcomes",
        },
    )
    return atomic_write_source_manifest(
        root,
        root / "clinicaltrials_gov_v2_manifest.json",
        manifest,
    )


def _merge_source_file_records(
    *groups: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Merge source records by path while rejecting physical-identity conflicts."""

    merged: dict[str, dict[str, Any]] = {}
    for group in groups:
        for raw_record in group:
            record = dict(raw_record)
            path = PurePosixPath(str(record.get("local_path", record.get("path", "")))).as_posix()
            if not path:
                raise ValueError("Cannot merge a source record without a local path")
            existing = merged.get(path)
            if existing is not None:
                if existing.get("acquired_sha256") != record.get("acquired_sha256") or existing.get(
                    "acquired_bytes"
                ) != record.get("acquired_bytes"):
                    raise ValueError(f"Conflicting source records for {path}")
                continue
            merged[path] = record
    return [merged[path] for path in sorted(merged)]


def acquire_clinicaltrials_cardiac_safety_cohort(
    raw_root: str | os.PathLike[str],
    *,
    session: requests.Session | None = None,
    page_size: int = CLINICALTRIALS_PAGE_SIZE,
) -> dict[str, Any]:
    """Freeze a separate heuristic cardiac-safety registry/results cohort.

    Search membership is deliberately high-recall and text-heuristic.  Raw posted
    outcome-measure and adverse-event modules are preserved when present, but no
    registry value is interpreted as a molecule-level efficacy or safety label.
    """

    if page_size < 1 or page_size > CLINICALTRIALS_PAGE_SIZE:
        raise ValueError(f"ClinicalTrials.gov page_size must be in 1..{CLINICALTRIALS_PAGE_SIZE}")
    root = Path(raw_root) / "clinicaltrials_gov_v2"
    top_manifest_path = root / "clinicaltrials_gov_v2_manifest.json"
    prior_manifest: dict[str, Any] | None = None
    if top_manifest_path.exists():
        candidate = json.loads(top_manifest_path.read_text(encoding="utf-8"))
        if verify_document_sha256(candidate):
            prior_manifest = candidate
    client = session or build_session(retries=10, backoff_factor=1.0)
    metadata_manifest = acquire_clinicaltrials_metadata(
        raw_root,
        session=client,
        update_current_pointer=False,
    )
    identity = {
        "apiVersion": str(metadata_manifest["api_version"]),
        "dataTimestamp": str(metadata_manifest["dataTimestamp"]),
    }
    if (
        prior_manifest
        and prior_manifest.get("targeted_study_query_status") == "complete_unmapped_drug_intervention_cohort"
        and prior_manifest.get("dataTimestamp") != identity["dataTimestamp"]
    ):
        raise ValueError(
            "ClinicalTrials.gov dataTimestamp changed between broad and cardiac cohorts; "
            "the accepted broad top manifest is preserved and the composite is incomplete"
        )
    snapshot_key = _clinicaltrials_snapshot_key(identity)
    cohort_root = root / "cardiac_safety_heuristic_cohorts" / snapshot_key
    pages_root = cohort_root / "pages"
    pages_root.mkdir(parents=True, exist_ok=True)

    metadata_path = root / str(metadata_manifest["study_metadata_local_path"])
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    available_pieces = _clinicaltrials_metadata_piece_names(metadata)
    missing_pieces = sorted(set(CLINICALTRIALS_CARDIAC_SAFETY_FIELDS) - available_pieces)
    if missing_pieces:
        raise ValueError(
            "ClinicalTrials.gov cardiac-safety fields are absent from frozen metadata: "
            + ", ".join(missing_pieces)
        )

    before_path = cohort_root / "version_before.json"
    before_already_frozen = before_path.exists()
    before = snapshot_http_bytes(CLINICALTRIALS_VERSION_URL, before_path, session=client)
    before = annotate_acquisition_record(
        before,
        artifact_role="cardiac_safety_cohort_api_version_before",
        local_path=before_path.relative_to(root).as_posix(),
    )
    before_identity = _clinicaltrials_version_identity(before_path)
    if before_identity != identity:
        raise ValueError("ClinicalTrials.gov cardiac cohort metadata/pre-version mismatch")
    if before_already_frozen:
        resume_path = _next_numbered_snapshot_path(cohort_root, "version_resume_probe")
        snapshot_http_bytes(CLINICALTRIALS_VERSION_URL, resume_path, session=client)
        live_identity = _clinicaltrials_version_identity(resume_path)
        if live_identity != before_identity:
            raise ValueError(
                "ClinicalTrials.gov live version changed before cardiac cohort resume; "
                "use the new version-keyed cohort"
            )
    resume_records = _load_clinicaltrials_version_records(
        root,
        cohort_root,
        "version_resume_probe",
        "cardiac_safety_cohort_live_version_before_resume",
        client,
    )
    for record in resume_records:
        if _clinicaltrials_version_identity(root / str(record["local_path"])) != before_identity:
            raise ValueError("Persisted cardiac-safety resume probe has version drift")

    query_contract = {
        "endpoint": CLINICALTRIALS_STUDIES_URL,
        "query.term": CLINICALTRIALS_CARDIAC_SAFETY_QUERY,
        "format": "json",
        "pageSize": page_size,
        "countTotal": True,
        "sort": CLINICALTRIALS_DRUG_SORT,
        "fields": list(CLINICALTRIALS_CARDIAC_SAFETY_FIELDS),
        "membership_semantics": (
            "Heuristic text-search cohort constrained to InterventionType=DRUG; false positives, "
            "ambiguous cardiac context, co-interventions, and unmapped drug names are retained."
        ),
    }
    query_digest = hashlib.sha256(canonical_json_bytes(query_contract)).hexdigest()
    page_records: list[dict[str, Any]] = []
    chain_records: list[dict[str, Any]] = []
    memberships: list[dict[str, Any]] = []
    seen_nct_ids: set[str] = set()
    seen_tokens: set[str] = set()
    next_token: str | None = None
    total_count: int | None = None
    term_counts: Counter[str] = Counter()
    has_results_counts: Counter[str] = Counter()
    outcome_module_count = 0
    adverse_event_module_count = 0
    no_projected_term_match_count = 0
    concatenated_digest = hashlib.sha256()

    for page_index in range(100_000):
        params: dict[str, str] = {
            "query.term": CLINICALTRIALS_CARDIAC_SAFETY_QUERY,
            "format": "json",
            "pageSize": str(page_size),
            "countTotal": "true",
            "sort": CLINICALTRIALS_DRUG_SORT,
            "fields": ",".join(CLINICALTRIALS_CARDIAC_SAFETY_FIELDS),
        }
        input_token = next_token
        if input_token is not None:
            if input_token in seen_tokens:
                raise ValueError("ClinicalTrials.gov cardiac cohort page-token cycle")
            seen_tokens.add(input_token)
            params["pageToken"] = input_token
        request_url = f"{CLINICALTRIALS_STUDIES_URL}?{urlencode(params)}"
        page_path = pages_root / f"page_{page_index:06d}.json"
        acquired = download_immutable(
            DownloadSpec(canonical_url=request_url, destination=page_path),
            session=client,
            timeout=(20.0, 300.0),
        )
        raw = page_path.read_bytes()
        concatenated_digest.update(raw)
        payload = json.loads(raw)
        if not isinstance(payload, dict) or not isinstance(payload.get("studies"), list):
            raise ValueError(f"Invalid ClinicalTrials.gov cardiac cohort page: {page_path}")
        studies = payload["studies"]
        reported_total = payload.get("totalCount")
        if total_count is None:
            if not isinstance(reported_total, int) or reported_total < 0:
                raise ValueError(f"Cardiac cohort first page lacks nonnegative totalCount: {page_path}")
            total_count = reported_total
        elif reported_total is not None and total_count != reported_total:
            raise ValueError("Cardiac cohort totalCount changed across pages")
        if len(studies) > page_size or (total_count and not studies):
            raise ValueError(f"Cardiac cohort page-size/emptiness invariant failed: {page_path}")

        first_nct: str | None = None
        last_nct: str | None = None
        for study_index, study in enumerate(studies):
            if not isinstance(study, dict):
                raise ValueError(f"Cardiac cohort study is not an object: {page_path}")
            study_audit = _clinicaltrials_study_audit(study)
            nct_id = str(study_audit["nct_id"])
            if nct_id in seen_nct_ids:
                raise ValueError(f"Duplicate cardiac cohort NCT ID: {nct_id}")
            seen_nct_ids.add(nct_id)
            first_nct = first_nct or nct_id
            last_nct = nct_id
            study_text = canonical_json_bytes(study).decode("utf-8")
            matched_terms = sorted(
                name
                for name, pattern in CLINICALTRIALS_CARDIAC_TERM_PATTERNS.items()
                if pattern.search(study_text)
            )
            if not matched_terms:
                no_projected_term_match_count += 1
            term_counts.update(matched_terms)
            results_section = study.get("resultsSection")
            results_section = results_section if isinstance(results_section, dict) else {}
            has_outcomes = isinstance(results_section.get("outcomeMeasuresModule"), dict)
            has_adverse_events = isinstance(results_section.get("adverseEventsModule"), dict)
            outcome_module_count += int(has_outcomes)
            adverse_event_module_count += int(has_adverse_events)
            has_results = study_audit["has_results"]
            has_results_counts[
                "true" if has_results is True else "false" if has_results is False else "missing"
            ] += 1
            memberships.append(
                {
                    "nct_id": nct_id,
                    "page_index": page_index,
                    "study_index_within_page": study_index,
                    "study_sha256": study_audit["study_sha256"],
                    "projected_heuristic_term_matches": matched_terms,
                    "has_posted_outcome_measures_module": has_outcomes,
                    "has_posted_adverse_events_module": has_adverse_events,
                }
            )

        output_value = payload.get("nextPageToken")
        if output_value is not None and not isinstance(output_value, str):
            raise ValueError(f"Cardiac cohort nextPageToken is invalid: {page_path}")
        output_token = output_value or None
        page_record = annotate_acquisition_record(
            acquired,
            artifact_role="heuristic_cardiac_safety_registry_results_page",
            local_path=page_path.relative_to(root).as_posix(),
        )
        page_record.update(
            {
                "page_index": page_index,
                "query_contract_sha256": query_digest,
                "input_page_token": input_token,
                "output_page_token": output_token,
                "study_count": len(studies),
                "reported_total_count": total_count,
                "first_nct_id": first_nct,
                "last_nct_id": last_nct,
            }
        )
        page_records.append(page_record)
        chain_records.append(
            {
                "page_index": page_index,
                "input_page_token": input_token,
                "output_page_token": output_token,
                "page_path": page_path.relative_to(root).as_posix(),
                "page_bytes": len(raw),
                "page_sha256": hashlib.sha256(raw).hexdigest(),
                "study_count": len(studies),
            }
        )
        next_token = output_token
        if next_token is None:
            break
        if total_count is not None and page_index + 1 > (total_count // page_size) + 2:
            raise ValueError("Cardiac cohort pagination exceeded totalCount-derived bound")
    else:  # pragma: no cover - safety guard
        raise ValueError("Cardiac cohort pagination exceeded hard page limit")

    if total_count is None or len(seen_nct_ids) != total_count:
        raise ValueError(
            f"Cardiac cohort unique NCT membership does not reconcile: {len(seen_nct_ids)} != {total_count}"
        )
    expected_pages = {pages_root / f"page_{index:06d}.json" for index in range(len(page_records))}
    observed_pages = {
        path for path in pages_root.iterdir() if re.fullmatch(r"page_[0-9]{6}\.json", path.name)
    }
    if observed_pages != expected_pages:
        raise ValueError("Cardiac cohort page directory contains unexpected pages")

    after_path = _next_numbered_snapshot_path(cohort_root, "version_after_probe")
    snapshot_http_bytes(CLINICALTRIALS_VERSION_URL, after_path, session=client)
    if _clinicaltrials_version_identity(after_path) != before_identity:
        raise ValueError("ClinicalTrials.gov version changed during cardiac cohort pagination")
    after_records = _load_clinicaltrials_version_records(
        root,
        cohort_root,
        "version_after_probe",
        "cardiac_safety_cohort_api_version_after",
        client,
    )
    for record in after_records:
        if _clinicaltrials_version_identity(root / str(record["local_path"])) != before_identity:
            raise ValueError("Persisted cardiac-safety post-page probe has version drift")

    chain_inventory = _write_jsonl_records(
        cohort_root / "page_token_chain.jsonl",
        chain_records,
        relative_to=root,
    )
    membership_inventory = _write_jsonl_records(
        cohort_root / "nct_membership.jsonl",
        memberships,
        relative_to=root,
    )
    cardiac_summary = {
        "cohort_snapshot_key": snapshot_key,
        "query_contract": query_contract,
        "query_contract_sha256": query_digest,
        "page_count": len(page_records),
        "reported_total_count": total_count,
        "unique_nct_count": len(seen_nct_ids),
        "page_token_chain_inventory": chain_inventory,
        "nct_membership_inventory": membership_inventory,
        "projected_heuristic_term_match_counts": dict(sorted(term_counts.items())),
        "query_matches_without_term_in_projected_fields": no_projected_term_match_count,
        "registry_has_results_counts": dict(sorted(has_results_counts.items())),
        "studies_with_posted_outcome_measures_module": outcome_module_count,
        "studies_with_posted_adverse_events_module": adverse_event_module_count,
        "concatenated_raw_page_bytes_sha256": concatenated_digest.hexdigest(),
        "false_positive_and_context_ambiguity_status": (
            "retained_unreviewed_heuristic_membership; no molecular inference"
        ),
        "molecule_alias_and_identity_mapping_status": "not_attempted",
        "raw_posted_results_interpretation_status": "not_attempted",
        "canonical_rows_admitted": 0,
        "model_labels_admitted": 0,
    }
    prior_is_matching_broad = bool(
        prior_manifest
        and prior_manifest.get("dataTimestamp") == identity["dataTimestamp"]
        and prior_manifest.get("targeted_study_query_status") == "complete_unmapped_drug_intervention_cohort"
    )
    file_groups: list[Sequence[Mapping[str, Any]]] = [metadata_manifest["files"]]
    if prior_is_matching_broad and prior_manifest is not None:
        file_groups.insert(0, prior_manifest["files"])
    file_groups.append([before, *resume_records, *page_records, *after_records])
    merged_files = _merge_source_file_records(*file_groups)
    broad_summary = None
    if prior_is_matching_broad and prior_manifest is not None:
        broad_summary = {
            key: prior_manifest[key]
            for key in (
                "cohort_snapshot_key",
                "query_contract",
                "query_contract_sha256",
                "page_count",
                "reported_total_count",
                "unique_nct_count",
                "page_token_chain_inventory",
                "nct_membership_inventory",
                "registry_has_results_counts",
                "registry_overall_status_counts",
                "registry_study_type_counts",
                "registry_phase_counts",
                "concatenated_raw_page_bytes_sha256",
            )
        }
    manifest = _source_manifest(
        "clinicaltrials_gov_v2",
        release_id=f"api-{identity['apiVersion']}",
        release_date=identity["dataTimestamp"],
        files=merged_files,
        status=(
            "complete_broad_drug_and_heuristic_cardiac_safety_cohorts"
            if prior_is_matching_broad
            else "complete_heuristic_cardiac_safety_cohort_only"
        ),
        extra={
            "api_major": "v2",
            "api_version": identity["apiVersion"],
            "dataTimestamp": identity["dataTimestamp"],
            "metadata_snapshot_key": metadata_manifest["metadata_snapshot_key"],
            "openapi_snapshot_status": metadata_manifest["openapi_snapshot_status"],
            "terms_snapshot_content_status": metadata_manifest["terms_snapshot_content_status"],
            "targeted_study_query_status": (
                "complete_unmapped_drug_intervention_cohort"
                if prior_is_matching_broad
                else "broad_drug_cohort_not_present_for_same_timestamp"
            ),
            "alias_independent_all_drug_cohort": broad_summary,
            "cardiac_safety_heuristic_cohort": cardiac_summary,
            "outcome_and_adverse_event_result_fields_retrieved": True,
            "outcome_and_adverse_event_result_fields_admitted_as_labels": False,
            "absence_semantics": "no_record_and_no_posted_results_are_not_negative_outcomes",
        },
    )
    return atomic_write_source_manifest(root, top_manifest_path, manifest)


def acquire_clinicaltrials_complete(
    raw_root: str | os.PathLike[str],
    *,
    session: requests.Session | None = None,
) -> dict[str, Any]:
    """Acquire both the all-DRUG registry cohort and separate cardiac-results cohort."""

    client = session or build_session(retries=10, backoff_factor=1.0)
    acquire_clinicaltrials_drug_cohort(raw_root, session=client)
    result = acquire_clinicaltrials_cardiac_safety_cohort(raw_root, session=client)
    if result.get("snapshot_status") != ("complete_broad_drug_and_heuristic_cardiac_safety_cohorts"):
        raise ValueError(
            "ClinicalTrials.gov composite acquisition did not complete both same-version cohorts"
        )
    return result


def inspect_drugsfda_archive(archive_path: str | os.PathLike[str]) -> dict[str, Any]:
    """Parse the Drugs@FDA archive and preserve exact source-width anomalies."""

    parsed = inspect_delimited_zip(archive_path, delimiter="\t", encoding_candidates=("utf-8-sig", "cp1252"))
    tables = [
        record for record in parsed["members"] if str(record["archive_member_path"]).lower().endswith(".txt")
    ]
    if len(tables) != DRUGSFDA_EXPECTED_TABLES:
        raise ValueError(f"Drugs@FDA archive has {len(tables)} TXT tables; expected 12")
    normalized_names = {Path(str(record["archive_member_path"])).stem.casefold() for record in tables}
    if len(normalized_names) != len(tables):
        raise ValueError("Drugs@FDA table names are not unique")
    anomaly_count = int(parsed["total_malformed_width_rows"])
    return {
        **parsed,
        "txt_table_count": len(tables),
        "normalized_table_names": sorted(normalized_names),
        "source_width_anomaly_rows": anomaly_count,
        "table_tokenization_status": "passed_all_rows_tokenized",
        "row_shape_status": "passed" if anomaly_count == 0 else "source_anomalies_retained_for_quarantine",
    }


def audit_drugsfda_relations(archive_path: str | os.PathLike[str]) -> dict[str, Any]:
    """Audit declared primary keys and foreign-key attrition from raw ZIP bytes.

    The contracts follow the official definitions frozen on the Drugs@FDA
    landing page.  Malformed-width source rows and blank keys are reported and
    excluded from set membership; they are never repaired or coerced here.
    """

    archive = Path(archive_path)
    key_sets: dict[str, set[tuple[str, ...]]] = {}
    primary_reports: list[dict[str, Any]] = []
    with zipfile.ZipFile(archive, "r") as zipped:
        for member, columns in DRUGSFDA_PRIMARY_KEYS.items():
            seen: set[tuple[str, ...]] = set()
            source_rows = 0
            malformed_rows = 0
            blank_key_rows = 0
            duplicate_key_rows = 0
            with zipped.open(member, "r") as binary, _text_wrapper(binary, encoding="cp1252") as text:
                reader = csv.DictReader(text, delimiter="\t")
                missing_columns = set(columns) - set(reader.fieldnames or [])
                if missing_columns:
                    raise ValueError(f"{member} lacks declared key columns {sorted(missing_columns)}")
                for row in reader:
                    source_rows += 1
                    malformed = None in row or any(value is None for value in row.values())
                    if malformed:
                        malformed_rows += 1
                        continue
                    key = tuple(str(row[column]).strip() for column in columns)
                    if any(not value for value in key):
                        blank_key_rows += 1
                        continue
                    if key in seen:
                        duplicate_key_rows += 1
                    else:
                        seen.add(key)
            key_sets[member] = seen
            primary_reports.append(
                {
                    "table": member,
                    "declared_key_columns": list(columns),
                    "source_rows": source_rows,
                    "malformed_width_rows_excluded": malformed_rows,
                    "blank_key_rows": blank_key_rows,
                    "unique_nonblank_keys": len(seen),
                    "duplicate_key_rows": duplicate_key_rows,
                    "key_integrity": (
                        "passed"
                        if not malformed_rows and not blank_key_rows and not duplicate_key_rows
                        else "source_anomalies_retained"
                    ),
                }
            )

        foreign_reports: list[dict[str, Any]] = []
        for source_table, source_columns, target_table, target_columns in DRUGSFDA_FOREIGN_KEYS:
            target_keys = key_sets[target_table]
            source_rows = 0
            malformed_rows = 0
            blank_reference_rows = 0
            checked_reference_rows = 0
            missing_reference_rows = 0
            missing_examples: list[list[str]] = []
            with zipped.open(source_table, "r") as binary, _text_wrapper(binary, encoding="cp1252") as text:
                reader = csv.DictReader(text, delimiter="\t")
                missing_columns = set(source_columns) - set(reader.fieldnames or [])
                if missing_columns:
                    raise ValueError(
                        f"{source_table} lacks declared relation columns {sorted(missing_columns)}"
                    )
                for row in reader:
                    source_rows += 1
                    malformed = None in row or any(value is None for value in row.values())
                    if malformed:
                        malformed_rows += 1
                        continue
                    key = tuple(str(row[column]).strip() for column in source_columns)
                    if any(not value for value in key):
                        blank_reference_rows += 1
                        continue
                    checked_reference_rows += 1
                    if key not in target_keys:
                        missing_reference_rows += 1
                        if len(missing_examples) < 10:
                            missing_examples.append(list(key))
            foreign_reports.append(
                {
                    "source_table": source_table,
                    "source_columns": list(source_columns),
                    "target_table": target_table,
                    "target_columns": list(target_columns),
                    "source_rows": source_rows,
                    "malformed_width_rows_excluded": malformed_rows,
                    "blank_reference_rows": blank_reference_rows,
                    "checked_nonblank_reference_rows": checked_reference_rows,
                    "missing_reference_rows": missing_reference_rows,
                    "missing_reference_examples": missing_examples,
                    "join_integrity": ("passed" if not missing_reference_rows else "source_orphans_retained"),
                }
            )

    return {
        "parser_version": PARSER_VERSION,
        "official_contract_source": "frozen Drugs@FDA landing-page definitions as of 2025-01-10",
        "primary_key_reports": primary_reports,
        "foreign_key_reports": foreign_reports,
        "total_primary_duplicate_rows": sum(int(report["duplicate_key_rows"]) for report in primary_reports),
        "total_blank_primary_key_rows": sum(int(report["blank_key_rows"]) for report in primary_reports),
        "total_malformed_width_rows_excluded_from_key_audit": sum(
            int(report["malformed_width_rows_excluded"]) for report in primary_reports
        ),
        "total_missing_foreign_key_rows_across_relations": sum(
            int(report["missing_reference_rows"]) for report in foreign_reports
        ),
        "relational_integrity_status": (
            "passed"
            if all(report["key_integrity"] == "passed" for report in primary_reports)
            and all(report["join_integrity"] == "passed" for report in foreign_reports)
            else "source_anomalies_or_orphans_retained"
        ),
    }


def acquire_drugsfda(
    raw_root: str | os.PathLike[str],
    *,
    session: requests.Session | None = None,
) -> dict[str, Any]:
    """Acquire the complete official dated Drugs@FDA relational archive."""

    root = Path(raw_root) / "drugs_at_fda_bulk"
    root.mkdir(parents=True, exist_ok=True)
    client = session or build_session(retries=8, backoff_factor=1.0)
    landing = snapshot_http_bytes(DRUGSFDA_LANDING_URL, root / "drugsfda_data_files.html", session=client)
    landing = annotate_acquisition_record(landing, artifact_role="official_release_landing_page")
    landing_text = (root / "drugsfda_data_files.html").read_text(encoding="utf-8")
    if "Data Last Updated: August 4th, 2026" not in landing_text or "12 text tables" not in landing_text:
        raise ValueError("Drugs@FDA landing page does not match the dated 12-table release contract")
    erd = snapshot_http_bytes(
        DRUGSFDA_ERD_URL,
        root / "drugsfda_erd_20250110.bin",
        session=client,
    )
    erd = annotate_acquisition_record(
        erd,
        artifact_role="official_entity_relationship_diagram_as_of_2025-01-10",
    )
    path = root / DRUGSFDA_EXPECTED_FILENAME
    record = download_immutable(
        DownloadSpec(
            canonical_url=DRUGSFDA_ARCHIVE_URL,
            destination=path,
            expected_filename=DRUGSFDA_EXPECTED_FILENAME,
        ),
        session=client,
    )
    record = annotate_acquisition_record(record, artifact_role="complete_relational_archive")
    content_disposition = str(record.get("response_headers", {}).get("content-disposition", ""))
    if content_disposition and DRUGSFDA_EXPECTED_FILENAME.casefold() not in content_disposition.casefold():
        raise ValueError(
            f"Drugs@FDA response filename is not {DRUGSFDA_EXPECTED_FILENAME}: {content_disposition}"
        )
    archive_integrity = verify_zip_archive(path, root / f"{path.name}.members.jsonl")
    parse = inspect_drugsfda_archive(path)
    relational = audit_drugsfda_relations(path)
    record["archive_integrity"] = archive_integrity
    has_source_anomalies = bool(
        int(parse["source_width_anomaly_rows"]) or relational["relational_integrity_status"] != "passed"
    )
    manifest = _source_manifest(
        "drugs_at_fda_bulk",
        release_id=Path(DRUGSFDA_EXPECTED_FILENAME).stem,
        release_date="2026-08-04",
        files=[landing, erd, record],
        status=(
            "acquired_parse_and_relational_audited_with_source_anomalies"
            if has_source_anomalies
            else "complete_parse_and_relational_integrity_verified"
        ),
        extra={
            "official_landing_url": DRUGSFDA_LANDING_URL,
            "landing_page_snapshot": landing,
            "source_data_last_updated": "2026-08-04",
            "data_definitions_version": "2025-01-10",
            "erd_digest": erd["acquired_sha256"],
            "archive_member_table": parse,
            "relational_key_and_join_audit": relational,
            "canonical_mapping_status": "not_attempted",
        },
    )
    return atomic_write_source_manifest(
        root,
        root / "drugs_at_fda_bulk_manifest.json",
        manifest,
    )


def acquire_dailymed_part(
    raw_root: str | os.PathLike[str],
    part: tuple[str, str, int, int],
    *,
    session: requests.Session | None = None,
) -> dict[str, Any]:
    """Acquire and fully verify one independently resumable DailyMed release part."""

    filename, expected_md5, expected_files, expected_bytes = part
    root = Path(raw_root) / "dailymed_spl_v2_human_rx"
    root.mkdir(parents=True, exist_ok=True)
    client = session or build_session(retries=10, backoff_factor=1.0)
    path = root / filename
    record = download_immutable(
        DownloadSpec(
            canonical_url=f"{DAILYMED_DOWNLOAD_ROOT}/{filename}",
            destination=path,
            expected_md5=expected_md5,
            expected_bytes=expected_bytes,
            expected_filename=filename,
        ),
        session=client,
        timeout=(20.0, 300.0),
    )
    record = annotate_acquisition_record(record, artifact_role="human_prescription_release_part")
    if filename in {item[0] for item in DAILYMED_PARTS}:
        observed_last_modified = str(record.get("response_headers", {}).get("last-modified", ""))
        if "04 Aug 2026" not in observed_last_modified:
            raise ValueError(
                f"DailyMed release drift for {filename}: Last-Modified={observed_last_modified!r}"
            )
    record["release_page_last_modified_date"] = DAILYMED_RELEASE_PAGE_DATE
    record["expected_http_last_modified_date"] = DAILYMED_RELEASE_HTTP_LAST_MODIFIED_DATE
    record["expected_file_member_count"] = expected_files
    record["archive_integrity"] = verify_zip_archive(
        path,
        root / f"{filename}.members.jsonl",
        expected_member_count=expected_files,
    )
    part_manifest = _source_manifest(
        "dailymed_spl_v2_human_rx",
        release_id="human-rx-full-2026-08-03",
        release_date=DAILYMED_RELEASE_PAGE_DATE,
        files=[record],
        status="part_complete_raw_archive_crc_and_membership_verified",
        extra={
            "partial_scope": filename,
            "expected_part_count_in_complete_release": len(DAILYMED_PARTS),
        },
    )
    atomic_write_json(root / f"{filename}.part_manifest.json", part_manifest)
    return record


def acquire_dailymed(
    raw_root: str | os.PathLike[str],
    *,
    session: requests.Session | None = None,
    parts: Sequence[tuple[str, str, int, int]] = DAILYMED_PARTS,
) -> dict[str, Any]:
    """Acquire all declared human-prescription SPL release parts without extraction."""

    root = Path(raw_root) / "dailymed_spl_v2_human_rx"
    root.mkdir(parents=True, exist_ok=True)
    client = session or build_session(retries=10, backoff_factor=1.0)
    landing = snapshot_http_bytes(DAILYMED_LANDING_URL, root / "all_drug_labels.html", session=client)
    landing = annotate_acquisition_record(
        landing,
        artifact_role="official_release_landing_page_with_published_md5_membership_and_date",
    )
    records: list[dict[str, Any]] = []
    for part in parts:
        records.append(acquire_dailymed_part(raw_root, part, session=client))
    expected_total = sum(item[2] for item in parts)
    actual_total = sum(int(item["archive_integrity"]["file_member_count"]) for item in records)
    if actual_total != expected_total:
        raise ValueError(f"DailyMed total file membership mismatch: {actual_total} != {expected_total}")
    manifest = _source_manifest(
        "dailymed_spl_v2_human_rx",
        release_id="human-rx-full-2026-08-03",
        release_date=DAILYMED_RELEASE_PAGE_DATE,
        files=[landing, *records],
        status="complete_raw_archives_crc_and_membership_verified",
        extra={
            "official_landing_url": DAILYMED_LANDING_URL,
            "landing_page_snapshot": landing,
            "scope": "all_six_current_human_prescription_release_parts",
            "release_part_count": len(records),
            "expected_total_transfer_bytes": sum(item[3] for item in parts),
            "expected_and_verified_file_member_count": actual_total,
            "setid_version_history_status": (
                "not_retrieved; raw XML/ZIP membership is frozen but per-SETID history requires "
                "a later parsed SETID inventory and API acquisition"
            ),
            "section_extraction_status": "not_attempted",
            "canonical_mapping_status": "not_attempted",
        },
    )
    return atomic_write_source_manifest(
        root,
        root / "dailymed_spl_v2_human_rx_manifest.json",
        manifest,
    )


def collect_chembl_accessions(target_components_path: str | os.PathLike[str]) -> set[str]:
    """Collect syntactically valid nonblank accessions from the immutable ChEMBL component view."""

    frame = pd.read_parquet(target_components_path, columns=["accession"])
    values: set[str] = set()
    for raw in frame["accession"].dropna().astype(str):
        for token in re.split(r"[;,|\s]+", raw.strip()):
            if token and UNIPROT_ACCESSION_RE.fullmatch(token):
                values.add(token)
    return values


def collect_chembl_source_identifiers(target_components_path: str | os.PathLike[str]) -> set[str]:
    """Collect every nonblank source identifier, including non-UniProt quarantine values."""

    frame = pd.read_parquet(target_components_path, columns=["accession"])
    values: set[str] = set()
    for raw in frame["accession"].dropna().astype(str):
        values.update(token for token in re.split(r"[;,|\s]+", raw.strip()) if token)
    return values


def collect_bindingdb_accessions(mapping_path: str | os.PathLike[str]) -> set[str]:
    """Conservatively collect UniProt-looking tokens from the preserved BindingDB mapping."""

    values: set[str] = set()
    with Path(mapping_path).open("r", encoding="utf-8-sig", errors="strict", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        field = next((name for name in (reader.fieldnames or []) if name.casefold() == "uniprot"), None)
        if field is None:
            raise ValueError("BindingDB mapping lacks the declared UniProt column")
        for row in reader:
            for token in re.split(r"[^A-Za-z0-9-]+", str(row.get(field, "")).strip()):
                if UNIPROT_ACCESSION_RE.fullmatch(token):
                    values.add(token)
    return values


def write_uniprot_source_membership(
    target_components_path: str | os.PathLike[str],
    bindingdb_mapping_path: str | os.PathLike[str],
    destination: str | os.PathLike[str],
) -> dict[str, Any]:
    """Write exact source-row-to-request-identifier lineage without canonical admission."""

    output_path = Path(destination)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output_path.name}.", dir=output_path.parent)
    temporary = Path(temporary_name)
    source_rows = 0
    identifier_references = 0
    valid_references = 0
    invalid_references = 0
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as output:
            frame = pd.read_parquet(
                target_components_path,
                columns=["target_chembl_id", "component_id", "accession"],
            )
            for row_index, row in frame.iterrows():
                raw = "" if pd.isna(row["accession"]) else str(row["accession"]).strip()
                if not raw:
                    continue
                source_rows += 1
                tokens = [token for token in re.split(r"[;,|\s]+", raw) if token]
                for token in tokens:
                    valid = bool(UNIPROT_ACCESSION_RE.fullmatch(token))
                    identifier_references += 1
                    valid_references += int(valid)
                    invalid_references += int(not valid)
                    quarantine_record: dict[str, Any] = {
                        "source_id": "chembl_37_target_components",
                        "source_file": Path(target_components_path).name,
                        "source_row_index_zero_based": int(row_index),
                        "source_target_id": str(row["target_chembl_id"]),
                        "source_component_id": (
                            None if pd.isna(row["component_id"]) else int(row["component_id"])
                        ),
                        "source_mapping_row_number": None,
                        "source_accession_value": raw,
                        "normalized_identifier": token,
                        "uniprot_accession_syntax_valid": valid,
                        "admission_state": "request_candidate" if valid else "identifier_syntax_quarantine",
                    }
                    output.write(canonical_json_bytes(quarantine_record).decode("utf-8") + "\n")

            with Path(bindingdb_mapping_path).open(
                "r", encoding="utf-8-sig", errors="strict", newline=""
            ) as handle:
                reader = csv.DictReader(handle, delimiter="\t")
                names = reader.fieldnames or []
                uniprot_field = next(
                    (name for name in names if name.casefold() == "uniprot"),
                    None,
                )
                polymer_field = next(
                    (name for name in names if name.casefold() == "polymerid"),
                    None,
                )
                if uniprot_field is None or polymer_field is None:
                    raise ValueError("BindingDB mapping lacks polymerid/UniProt columns")
                for row_number, row in enumerate(reader, start=2):
                    raw = str(row.get(uniprot_field, "")).strip()
                    if not raw:
                        continue
                    source_rows += 1
                    tokens = [token for token in re.split(r"[^A-Za-z0-9-]+", raw) if token]
                    for token in tokens:
                        valid = bool(UNIPROT_ACCESSION_RE.fullmatch(token))
                        identifier_references += 1
                        valid_references += int(valid)
                        invalid_references += int(not valid)
                        record = {
                            "source_id": "bindingdb_curated_202608_mapping",
                            "source_file": Path(bindingdb_mapping_path).name,
                            "source_row_index_zero_based": None,
                            "source_target_id": None,
                            "source_component_id": None,
                            "source_mapping_row_number": row_number,
                            "source_polymer_id": row.get(polymer_field),
                            "source_accession_value": raw,
                            "normalized_identifier": token,
                            "uniprot_accession_syntax_valid": valid,
                            "admission_state": (
                                "request_candidate" if valid else "identifier_syntax_quarantine"
                            ),
                        }
                        output.write(canonical_json_bytes(record).decode("utf-8") + "\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "path": output_path.name,
        "bytes": output_path.stat().st_size,
        "sha256": sha256_file(output_path),
        "source_rows_with_identifiers": source_rows,
        "identifier_reference_rows": identifier_references,
        "valid_uniprot_accession_reference_rows": valid_references,
        "identifier_syntax_quarantine_reference_rows": invalid_references,
    }


def normalized_accession_digest(accessions: Iterable[str]) -> str:
    """Hash sorted unique accession membership with an unambiguous newline framing."""

    normalized = sorted({value.strip() for value in accessions if value.strip()})
    return hashlib.sha256(("\n".join(normalized) + "\n").encode("ascii")).hexdigest()


def _uniprot_query(accessions: Sequence[str]) -> str:
    return " OR ".join(f"accession:{accession}" for accession in accessions)


def _uniprot_primary_protein_name(entry: Mapping[str, Any]) -> str | None:
    """Return the source-declared recommended/submitted full protein name, if present."""

    description = entry.get("proteinDescription")
    if not isinstance(description, dict):
        return None
    recommended = description.get("recommendedName")
    if isinstance(recommended, dict):
        full_name = recommended.get("fullName")
        if isinstance(full_name, dict) and full_name.get("value"):
            return str(full_name["value"])
    submitted = description.get("submissionNames")
    if not isinstance(submitted, list):
        submitted = description.get("submittedNames")
    if isinstance(submitted, list):
        for name in submitted:
            if not isinstance(name, dict):
                continue
            full_name = name.get("fullName")
            if isinstance(full_name, dict) and full_name.get("value"):
                return str(full_name["value"])
    return None


def _response_json_bytes(response: requests.Response) -> bytes:
    raw = response.content
    try:
        json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid JSON response from {response.url}") from exc
    return raw


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def acquire_uniprot_targeted(
    raw_root: str | os.PathLike[str],
    *,
    chembl_target_components: str | os.PathLike[str],
    bindingdb_mapping: str | os.PathLike[str],
    session: requests.Session | None = None,
    batch_size: int = UNIPROT_QUERY_BATCH_SIZE,
) -> dict[str, Any]:
    """Freeze targeted UniProtKB 2026_02 JSON responses for the source-accession union."""

    if batch_size < 1 or batch_size > 200:
        raise ValueError("UniProt batch_size must be between 1 and 200")
    root = Path(raw_root) / "uniprotkb_targeted_2026_02"
    pages_dir = root / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    chembl_source_identifiers = collect_chembl_source_identifiers(chembl_target_components)
    chembl_accessions = {
        value for value in chembl_source_identifiers if UNIPROT_ACCESSION_RE.fullmatch(value)
    }
    chembl_identifier_quarantine = chembl_source_identifiers - chembl_accessions
    bindingdb_accessions = collect_bindingdb_accessions(bindingdb_mapping)
    requested = sorted(chembl_accessions | bindingdb_accessions)
    if not requested:
        raise ValueError("No UniProt accessions were found in the declared source union")
    request_digest = normalized_accession_digest(requested)
    accession_list_path = root / "requested_accessions.txt"
    if accession_list_path.exists():
        existing = accession_list_path.read_text(encoding="ascii").splitlines()
        if existing != requested:
            raise ValueError("Existing UniProt accession request membership differs; use a new snapshot root")
    else:
        _atomic_write_bytes(accession_list_path, ("\n".join(requested) + "\n").encode("ascii"))
    source_membership_path = root / "accession_source_membership.jsonl"
    source_membership = write_uniprot_source_membership(
        chembl_target_components,
        bindingdb_mapping,
        source_membership_path,
    )

    client = session or build_session(retries=8, backoff_factor=1.0)
    client.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})
    metadata_dir = root / "metadata"
    metadata_dir.mkdir(exist_ok=True)
    release_snapshot = snapshot_http_bytes(
        UNIPROT_RELEASE_URL,
        metadata_dir / "uniprot_release_2026_02.html",
        session=client,
    )
    release_snapshot = annotate_acquisition_record(
        release_snapshot,
        artifact_role="official_release_note",
        local_path="metadata/uniprot_release_2026_02.html",
    )
    license_snapshot = snapshot_http_bytes(
        UNIPROT_LICENSE_URL,
        metadata_dir / "uniprot_license.html",
        session=client,
    )
    license_snapshot = annotate_acquisition_record(
        license_snapshot,
        artifact_role="official_license_and_disclaimer",
        local_path="metadata/uniprot_license.html",
    )
    pages: list[dict[str, Any]] = []
    requested_to_primary: dict[str, set[str]] = {accession: set() for accession in requested}
    primary_records: dict[str, dict[str, Any]] = {}
    observed_release: str | None = None
    observed_release_date: str | None = None
    for page_index, start in enumerate(range(0, len(requested), batch_size)):
        batch = requested[start : start + batch_size]
        query = _uniprot_query(batch)
        params = {"query": f"({query})", "format": "json", "size": "500"}
        request_url = f"{UNIPROT_SEARCH_URL}?{urlencode(params)}"
        page_path = pages_dir / f"page_{page_index:06d}.json"
        metadata_path = pages_dir / f"page_{page_index:06d}.manifest.json"
        retrieval_utc: str | None = None
        headers: dict[str, str] = {}
        resolved_url = request_url
        if page_path.exists() and metadata_path.exists():
            page_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if not verify_document_sha256(page_metadata):
                raise ValueError(f"UniProt page metadata digest failed: {metadata_path}")
            if (
                str(page_metadata.get("request_query_digest"))
                != hashlib.sha256(query.encode("ascii")).hexdigest()
            ):
                raise ValueError(f"UniProt page query changed: {page_path}")
            if sha256_file(page_path) != page_metadata.get("acquired_sha256"):
                raise ValueError(f"UniProt page bytes changed: {page_path}")
            raw = page_path.read_bytes()
            headers = dict(page_metadata.get("response_headers", {}))
            resolved_url = str(page_metadata.get("resolved_url", request_url))
            retrieval_utc = str(page_metadata.get("retrieval_utc") or "") or None
        else:
            retrieval_utc = utc_now()
            response = client.get(UNIPROT_SEARCH_URL, params=params, timeout=(15, 180))
            response.raise_for_status()
            raw = _response_json_bytes(response)
            headers = _safe_headers(response.headers)
            resolved_url = str(response.url)
            _atomic_write_bytes(page_path, raw)
            atomic_write_json(
                metadata_path,
                {
                    "schema_version": ACQUISITION_SCHEMA_VERSION,
                    "page_index": page_index,
                    "request_url": request_url,
                    "resolved_url": resolved_url,
                    "request_query_digest": hashlib.sha256(query.encode("ascii")).hexdigest(),
                    "requested_accessions": batch,
                    "response_headers": headers,
                    "retrieval_utc": retrieval_utc,
                    "acquired_bytes": len(raw),
                    "acquired_sha256": hashlib.sha256(raw).hexdigest(),
                },
            )
        release = str(headers.get("x-uniprot-release", ""))
        release_date = str(headers.get("x-uniprot-release-date", ""))
        if release != UNIPROT_RELEASE_ID:
            raise ValueError(f"UniProt release drift: expected {UNIPROT_RELEASE_ID}, got {release!r}")
        if release_date != UNIPROT_RELEASE_HEADER_DATE:
            raise ValueError(
                f"UniProt release-date drift: expected {UNIPROT_RELEASE_HEADER_DATE!r}, got {release_date!r}"
            )
        if observed_release not in (None, release) or observed_release_date not in (None, release_date):
            raise ValueError("UniProt release headers changed across pages")
        observed_release = release
        observed_release_date = release_date
        payload = json.loads(raw)
        results = payload.get("results", [])
        if not isinstance(results, list):
            raise ValueError(f"UniProt response has no results list: {page_path}")
        total_results = str(headers.get("x-total-results", ""))
        if not total_results.isdigit() or int(total_results) != len(results):
            raise ValueError(
                f"UniProt X-Total-Results does not reconcile on {page_path}: "
                f"{total_results!r} != {len(results)}"
            )
        if headers.get("link"):
            raise ValueError(f"UniProt response unexpectedly paginated within accession batch: {page_path}")
        for result in results:
            if not isinstance(result, dict):
                raise ValueError(f"Non-object UniProt result: {page_path}")
            primary = str(result.get("primaryAccession", "")).strip()
            if not primary:
                raise ValueError(f"UniProt result lacks primaryAccession: {page_path}")
            if primary in primary_records and primary_records[primary] != result:
                raise ValueError(f"Conflicting duplicate UniProt primary entry: {primary}")
            primary_records[primary] = result
            sequence = result.get("sequence", {})
            if isinstance(sequence, dict) and sequence.get("value") and sequence.get("md5"):
                observed_md5 = hashlib.md5(
                    str(sequence["value"]).encode("ascii"),
                    usedforsecurity=False,
                ).hexdigest()
                if observed_md5.casefold() != str(sequence["md5"]).casefold():
                    raise ValueError(f"UniProt sequence MD5 mismatch for {primary}")
            aliases = {primary}
            secondary = result.get("secondaryAccessions", [])
            if isinstance(secondary, list):
                aliases.update(str(item).strip() for item in secondary)
            for accession in set(batch) & aliases:
                requested_to_primary[accession].add(primary)
        pages.append(
            {
                "page_index": page_index,
                "path": page_path.relative_to(root).as_posix(),
                "local_path": page_path.relative_to(root).as_posix(),
                "artifact_role": "targeted_uniprotkb_json_page",
                "metadata_path": metadata_path.relative_to(root).as_posix(),
                "request_url": request_url,
                "resolved_url": resolved_url,
                "request_query_digest": hashlib.sha256(query.encode("ascii")).hexdigest(),
                "requested_count": len(batch),
                "returned_primary_count": len(results),
                "retrieval_utc": retrieval_utc,
                "response_headers": headers,
                "acquired_bytes": len(raw),
                "acquired_sha256": hashlib.sha256(raw).hexdigest(),
            }
        )

    mapping_path = root / "accession_resolution.jsonl"
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{mapping_path.name}.", dir=root)
    temporary = Path(temporary_name)
    resolved_count = 0
    secondary_count = 0
    missing_count = 0
    multi_mapped_count = 0
    replaced_count = 0
    ambiguous_accessions: list[dict[str, Any]] = []
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as output:
            resolution_universe = sorted(set(requested) | chembl_identifier_quarantine)
            for accession in resolution_universe:
                if accession in chembl_identifier_quarantine:
                    quarantine_record: dict[str, Any] = {
                        "requested_accession": accession,
                        "resolution_state": "non_uniprot_identifier_syntax_quarantine",
                        "returned_primary_accessions": [],
                        "replacement_state": "not_applicable_non_uniprot_identifier",
                    }
                    output.write(canonical_json_bytes(quarantine_record).decode("utf-8") + "\n")
                    continue
                primaries = sorted(requested_to_primary[accession])
                if not primaries:
                    state = "missing_or_retired_unresolved"
                    missing_count += 1
                elif len(primaries) > 1:
                    state = "ambiguous_multi_mapped_quarantine"
                    multi_mapped_count += 1
                    ambiguous_accessions.append(
                        {
                            "requested_accession": accession,
                            "returned_primary_accessions": primaries,
                        }
                    )
                elif primaries[0] == accession:
                    state = "resolved_primary"
                    resolved_count += 1
                else:
                    state = "resolved_secondary_to_primary"
                    secondary_count += 1
                    resolved_count += 1
                record: dict[str, Any] = {
                    "requested_accession": accession,
                    "resolution_state": state,
                    "returned_primary_accessions": primaries,
                    "replacement_state": "not_reported_by_search_endpoint",
                }
                if len(primaries) == 1:
                    entry = primary_records[primaries[0]]
                    sequence = entry.get("sequence", {})
                    audit = entry.get("entryAudit", {})
                    organism = entry.get("organism", {})
                    record.update(
                        {
                            "returned_primary_accession": primaries[0],
                            "entry_name": entry.get("uniProtkbId"),
                            "protein_name": _uniprot_primary_protein_name(entry),
                            "entry_type": entry.get("entryType"),
                            "secondary_accessions": entry.get("secondaryAccessions", []),
                            "organism_scientific_name": organism.get("scientificName"),
                            "taxonomy_id": organism.get("taxonId"),
                            "entry_version": audit.get("entryVersion"),
                            "sequence_version": audit.get("sequenceVersion"),
                            "last_modified": audit.get("lastSequenceUpdateDate"),
                            "last_annotation_update_date": audit.get("lastAnnotationUpdateDate"),
                            "first_public_date": audit.get("firstPublicDate"),
                            "sequence_length": sequence.get("length"),
                            "sequence_md5": sequence.get("md5"),
                            "sequence_crc64": sequence.get("crc64"),
                            "sequence_sha256": (
                                hashlib.sha256(str(sequence.get("value", "")).encode("ascii")).hexdigest()
                                if sequence.get("value")
                                else None
                            ),
                            "canonical_or_isoform_role": (
                                "isoform" if "-" in accession else "canonical_or_secondary_canonical"
                            ),
                        }
                    )
                output.write(canonical_json_bytes(record).decode("utf-8") + "\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, mapping_path)
    finally:
        temporary.unlink(missing_ok=True)

    concatenated_dataset_digest = hashlib.sha256()
    for page in pages:
        with (root / str(page["path"])).open("rb") as page_handle:
            for chunk in iter_binary_chunks(page_handle):
                concatenated_dataset_digest.update(chunk)
    entry_type_counts: Counter[str] = Counter()
    taxonomy_ids: set[str] = set()
    sequence_sha256_counts: Counter[str] = Counter()
    entries_with_sequence = 0
    entries_missing_sequence = 0
    inactive_entries_missing_sequence = 0
    sequence_md5_verified_entries = 0
    entries_missing_primary_protein_name = 0
    for primary, entry in sorted(primary_records.items()):
        entry_type_counts[str(entry.get("entryType") or "MISSING")] += 1
        organism = entry.get("organism")
        if isinstance(organism, dict) and organism.get("taxonId") is not None:
            taxonomy_ids.add(str(organism["taxonId"]))
        if _uniprot_primary_protein_name(entry) is None:
            entries_missing_primary_protein_name += 1
        sequence = entry.get("sequence")
        sequence = sequence if isinstance(sequence, dict) else {}
        sequence_value = sequence.get("value")
        sequence_md5 = sequence.get("md5")
        if not sequence_value:
            entries_missing_sequence += 1
            if str(entry.get("entryType") or "") == "Inactive":
                inactive_entries_missing_sequence += 1
            continue
        entries_with_sequence += 1
        encoded_sequence = str(sequence_value).encode("ascii")
        if sequence.get("length") != len(encoded_sequence):
            raise ValueError(f"UniProt sequence length mismatch for {primary}")
        observed_md5 = hashlib.md5(encoded_sequence, usedforsecurity=False).hexdigest()
        if not sequence_md5 or observed_md5.casefold() != str(sequence_md5).casefold():
            raise ValueError(f"UniProt sequence MD5 mismatch for {primary}")
        sequence_md5_verified_entries += 1
        sequence_sha256_counts[hashlib.sha256(encoded_sequence).hexdigest()] += 1
    duplicate_sequence_groups = {
        digest: count for digest, count in sequence_sha256_counts.items() if count > 1
    }
    manifest = _source_manifest(
        "uniprotkb_targeted_2026_02",
        release_id=observed_release or UNIPROT_RELEASE_ID,
        release_date=UNIPROT_RELEASE_DATE,
        files=[release_snapshot, license_snapshot, *pages],
        status=("complete_with_explicit_quarantine" if missing_count or multi_mapped_count else "complete"),
        extra={
            "official_release_url": UNIPROT_RELEASE_URL,
            "observed_release_header_date": observed_release_date,
            "release_and_license_HTTP_snapshot_status": (
                "identical_SPA_shell_bytes; release identity is bound by exact API page headers; "
                "license text still requires human/static-source review"
                if release_snapshot["acquired_sha256"] == license_snapshot["acquired_sha256"]
                else "distinct_HTTP_bytes_frozen"
            ),
            "request_source_counts": {
                "chembl_unique_nonblank_source_identifiers": len(chembl_source_identifiers),
                "chembl_unique_accessions": len(chembl_accessions),
                "chembl_non_uniprot_identifier_syntax_quarantine": len(chembl_identifier_quarantine),
                "bindingdb_unique_accessions": len(bindingdb_accessions),
                "union_requested_accessions": len(requested),
            },
            "accession_source_membership": source_membership,
            "normalized_sorted_accession_query_digest": request_digest,
            "requested_accession_list": {
                "path": accession_list_path.name,
                "bytes": accession_list_path.stat().st_size,
                "sha256": sha256_file(accession_list_path),
            },
            "requested_fields": "full UniProtKB JSON entry including accession, audit, organism, names, sequence",
            "page_count": len(pages),
            "pages": pages,
            "resolution_counts": {
                "requested": len(requested),
                "resolved": resolved_count,
                "resolved_secondary_to_primary": secondary_count,
                "missing_or_retired_unresolved": missing_count,
                "replacement_identified": replaced_count,
                "replacement_resolution_unavailable_for_missing": missing_count,
                "ambiguous_multi_mapped": multi_mapped_count,
                "non_uniprot_identifier_syntax_quarantine": len(chembl_identifier_quarantine),
                "unique_returned_primary": len(primary_records),
            },
            "ambiguous_accession_quarantine": ambiguous_accessions,
            "protein_entry_inventory": {
                "unique_returned_primary_entries": len(primary_records),
                "entries_with_sequence": entries_with_sequence,
                "entries_missing_sequence": entries_missing_sequence,
                "sequence_ready_entries": entries_with_sequence,
                "inactive_accession_returned_sequence_unavailable": (inactive_entries_missing_sequence),
                "other_entries_sequence_unavailable": (
                    entries_missing_sequence - inactive_entries_missing_sequence
                ),
                "sequence_unavailable_downstream_disposition": (
                    "quarantine_for_protein_identity_or_sequence_features; returned accession "
                    "status is retained but is not sequence-ready"
                ),
                "sequence_md5_verified_unique_entries": sequence_md5_verified_entries,
                "unique_sequence_sha256": len(sequence_sha256_counts),
                "duplicate_sequence_sha256_groups": len(duplicate_sequence_groups),
                "entries_in_duplicate_sequence_groups": sum(duplicate_sequence_groups.values()),
                "entry_type_counts": dict(sorted(entry_type_counts.items())),
                "distinct_taxonomy_ids": len(taxonomy_ids),
                "requested_isoform_accessions": sum("-" in accession for accession in requested),
                "entries_missing_primary_protein_name": entries_missing_primary_protein_name,
            },
            "accession_resolution_inventory": {
                "path": mapping_path.name,
                "bytes": mapping_path.stat().st_size,
                "sha256": sha256_file(mapping_path),
                "rows": len(requested) + len(chembl_identifier_quarantine),
            },
            "concatenated_raw_page_bytes_sha256": concatenated_dataset_digest.hexdigest(),
            "replacement_resolution_limit": (
                "The search endpoint resolves current primary and secondary accessions. Missing or "
                "deleted accessions remain explicit quarantine; no replacement was inferred."
            ),
            "release_drift_check": "passed_all_page_headers_exact",
        },
    )
    return atomic_write_source_manifest(
        root,
        root / "uniprotkb_targeted_2026_02_manifest.json",
        manifest,
    )


def write_acquisition_summary(
    report_root: str | os.PathLike[str],
    manifests: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Write a non-admission summary over exact source manifest identities."""

    root = Path(report_root)
    root.mkdir(parents=True, exist_ok=True)
    sources = []
    for manifest in manifests:
        if not verify_document_sha256(manifest):
            raise ValueError(f"Source manifest failed identity: {manifest.get('source_id')}")
        sources.append(
            {
                "source_id": manifest["source_id"],
                "release_id": manifest["release_id"],
                "snapshot_status": manifest["snapshot_status"],
                "manifest_sha256": manifest["manifest_sha256"],
                "exact_physical_file_count": manifest["exact_physical_file_count"],
                "exact_physical_bytes": manifest["exact_physical_bytes"],
                "canonical_rows_admitted": manifest["canonical_rows_admitted"],
                "model_labels_admitted": manifest["model_labels_admitted"],
            }
        )
    summary = {
        "schema_version": ACQUISITION_SCHEMA_VERSION,
        "report_type": "external_public_acquisition_summary",
        "sources": sorted(sources, key=lambda item: str(item["source_id"])),
        "source_count": len(sources),
        "canonical_rows_admitted": 0,
        "model_labels_admitted": 0,
        "substantive_training_started": False,
    }
    return atomic_write_json(root / "external_public_acquisition_summary.json", summary)


def verify_source_acquisition_manifest(
    source_root: str | os.PathLike[str],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Independently reconcile a source manifest to every declared physical byte."""

    root = Path(source_root).resolve()
    if not verify_document_sha256(manifest):
        raise ValueError(f"Source manifest identity failed: {manifest.get('source_id')}")
    files = manifest.get("files")
    if not isinstance(files, list):
        raise ValueError("Source manifest files must be a list")
    expected_source_count = int(
        manifest.get("exact_source_artifact_count", manifest.get("exact_physical_file_count", -1))
    )
    if expected_source_count != len(files):
        raise ValueError("Source manifest source-artifact count does not reconcile")
    verified_bytes = 0
    archive_inventories = 0
    seen_source_paths: set[str] = set()
    for item in files:
        if not isinstance(item, dict):
            raise ValueError("Source manifest file record is not an object")
        relative = PurePosixPath(str(item.get("local_path", item.get("path", ""))))
        if not str(relative) or relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe or missing source artifact path: {relative}")
        if relative.as_posix() in seen_source_paths:
            raise ValueError(f"Duplicate source artifact path: {relative}")
        seen_source_paths.add(relative.as_posix())
        candidate_path = root / Path(*relative.parts)
        if candidate_path.is_symlink():
            raise ValueError(f"Source artifact path is a symlink: {relative}")
        path = candidate_path.resolve()
        if root not in path.parents:
            raise ValueError(f"Source artifact escapes root: {relative}")
        if not path.is_file():
            raise ValueError(f"Source artifact is missing: {path}")
        expected_bytes = int(item.get("acquired_bytes", -1))
        if path.stat().st_size != expected_bytes:
            raise ValueError(f"Source artifact byte count changed: {relative}")
        if sha256_file(path) != item.get("acquired_sha256"):
            raise ValueError(f"Source artifact SHA-256 changed: {relative}")
        if item.get("acquired_md5") and md5_file(path) != item.get("acquired_md5"):
            raise ValueError(f"Source artifact MD5 changed: {relative}")
        verified_bytes += expected_bytes

        sidecar_digest = item.get("acquisition_sidecar_manifest_sha256")
        if sidecar_digest:
            sidecar_path = path.with_name(f"{path.name}.acquisition.json")
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
            if not verify_document_sha256(sidecar) or sidecar.get("manifest_sha256") != sidecar_digest:
                raise ValueError(f"Acquisition sidecar identity failed: {relative}")
            if sidecar.get("acquired_sha256") != item.get("acquired_sha256"):
                raise ValueError(f"Acquisition sidecar bytes differ: {relative}")
        elif item.get("manifest_sha256"):
            if not verify_document_sha256(item):
                raise ValueError(f"Embedded acquisition record identity failed: {relative}")

        page_metadata_relative = item.get("metadata_path")
        if page_metadata_relative:
            metadata_relative = PurePosixPath(str(page_metadata_relative))
            if metadata_relative.is_absolute() or ".." in metadata_relative.parts:
                raise ValueError(f"Unsafe page metadata path: {metadata_relative}")
            metadata_path = (root / Path(*metadata_relative.parts)).resolve()
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if not verify_document_sha256(metadata):
                raise ValueError(f"Page metadata identity failed: {metadata_relative}")
            if metadata.get("acquired_sha256") != item.get("acquired_sha256"):
                raise ValueError(f"Page metadata hash differs: {relative}")

        archive = item.get("archive_integrity")
        if isinstance(archive, dict):
            if archive.get("archive_sha256") != item.get("acquired_sha256"):
                raise ValueError(f"Archive integrity hash differs: {relative}")
            inventory_relative = PurePosixPath(str(archive.get("inventory_path", "")))
            if inventory_relative.is_absolute() or ".." in inventory_relative.parts:
                raise ValueError(f"Unsafe archive inventory path: {inventory_relative}")
            inventory_path = (root / Path(*inventory_relative.parts)).resolve()
            if inventory_path.stat().st_size != int(archive.get("inventory_bytes", -1)):
                raise ValueError(f"Archive inventory byte count changed: {inventory_relative}")
            if sha256_file(inventory_path) != archive.get("inventory_sha256"):
                raise ValueError(f"Archive inventory hash changed: {inventory_relative}")
            line_count = sum(1 for _line in inventory_path.open("r", encoding="utf-8"))
            if line_count != int(archive.get("member_count", -1)):
                raise ValueError(f"Archive inventory membership changed: {inventory_relative}")
            archive_inventories += 1

    expected_source_bytes = int(
        manifest.get("exact_source_artifact_bytes", manifest.get("exact_physical_bytes", -1))
    )
    if verified_bytes != expected_source_bytes:
        raise ValueError("Source manifest source-artifact byte total does not reconcile")

    bundle = manifest.get("bundle_inventory")
    verified_bundle_count = len(files)
    verified_bundle_bytes = verified_bytes
    if bundle is not None:
        if not isinstance(bundle, dict):
            raise ValueError("Source bundle inventory is not an object")
        for candidate in root.rglob("*"):
            if candidate.is_symlink():
                raise ValueError(f"Symlink is prohibited in source bundle: {candidate}")
        entries = bundle.get("entries")
        if not isinstance(entries, list):
            raise ValueError("Source bundle inventory entries must be a list")
        if hashlib.sha256(canonical_json_bytes(entries)).hexdigest() != bundle.get("entries_sha256"):
            raise ValueError("Source bundle inventory entry digest failed")
        if int(bundle.get("entry_count", -1)) != len(entries):
            raise ValueError("Source bundle inventory entry count failed")
        exclusions = bundle.get("excluded_paths")
        if not isinstance(exclusions, list) or len(exclusions) != 1:
            raise ValueError("Source bundle must exclude exactly its self-referential manifest")
        excluded_paths: set[str] = set()
        for raw_relative in exclusions:
            relative = PurePosixPath(str(raw_relative))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Unsafe source bundle exclusion: {relative}")
            excluded_paths.add(relative.as_posix())
        manifest_on_disk = root / Path(*PurePosixPath(next(iter(excluded_paths))).parts)
        if not manifest_on_disk.is_file():
            raise ValueError("Excluded source manifest is missing from bundle root")
        disk_document = json.loads(manifest_on_disk.read_text(encoding="utf-8"))
        if disk_document != manifest:
            raise ValueError("Source manifest argument differs from excluded on-disk manifest")

        declared_paths: set[str] = set()
        bundle_bytes = 0
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError("Source bundle entry is not an object")
            relative = PurePosixPath(str(entry.get("path", "")))
            if (
                not str(relative)
                or relative.is_absolute()
                or ".." in relative.parts
                or relative.as_posix() in declared_paths
            ):
                raise ValueError(f"Unsafe or duplicate source bundle path: {relative}")
            declared_paths.add(relative.as_posix())
            path = root / Path(*relative.parts)
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"Source bundle artifact is missing or a symlink: {relative}")
            expected_bytes = int(entry.get("bytes", -1))
            if path.stat().st_size != expected_bytes:
                raise ValueError(f"Source bundle artifact byte count changed: {relative}")
            if sha256_file(path) != entry.get("sha256"):
                raise ValueError(f"Source bundle artifact SHA-256 changed: {relative}")
            if "line_count" in entry:
                with path.open("rb") as handle:
                    line_count = sum(1 for _line in handle)
                if line_count != int(entry["line_count"]):
                    raise ValueError(f"Source bundle JSONL line count changed: {relative}")
            bundle_bytes += expected_bytes
        observed_paths = {
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file() and path.relative_to(root).as_posix() not in excluded_paths
        }
        if observed_paths != declared_paths:
            missing = sorted(declared_paths - observed_paths)
            unexpected = sorted(observed_paths - declared_paths)
            raise ValueError(f"Source bundle membership changed; missing={missing}, unexpected={unexpected}")
        raw_paths = {
            PurePosixPath(str(item.get("local_path", item.get("path", "")))).as_posix() for item in files
        }
        if not raw_paths.issubset(declared_paths):
            raise ValueError("Source HTTP artifacts are absent from exact bundle inventory")
        if bundle_bytes != int(bundle.get("total_bytes", -1)):
            raise ValueError("Source bundle byte total does not reconcile")
        if len(entries) != int(manifest.get("exact_physical_file_count", -1)):
            raise ValueError("Source manifest exact physical file count differs from bundle")
        if bundle_bytes != int(manifest.get("exact_physical_bytes", -1)):
            raise ValueError("Source manifest exact physical bytes differ from bundle")
        verified_bundle_count = len(entries)
        verified_bundle_bytes = bundle_bytes
    if int(manifest.get("canonical_rows_admitted", -1)) != 0:
        raise ValueError("External acquisition manifest unexpectedly admits canonical rows")
    if int(manifest.get("model_labels_admitted", -1)) != 0:
        raise ValueError("External acquisition manifest unexpectedly admits model labels")
    return {
        "source_id": manifest["source_id"],
        "release_id": manifest["release_id"],
        "manifest_sha256": manifest["manifest_sha256"],
        "verified_source_artifact_count": len(files),
        "verified_source_bytes": verified_bytes,
        "verified_bundle_artifact_count": verified_bundle_count,
        "verified_bundle_bytes": verified_bundle_bytes,
        "verified_archive_inventory_count": archive_inventories,
        "verification_status": "passed",
        "canonical_rows_admitted": 0,
        "model_labels_admitted": 0,
    }


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the module-local acquisition parser (central CLI integration is lead-owned)."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source",
        choices=(
            "bindingdb",
            "clinicaltrials",
            "drugsfda",
            "dailymed",
            "dailymed-part",
            "uniprot",
        ),
    )
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--part-number", type=int)
    parser.add_argument("--chembl-target-components", type=Path)
    parser.add_argument("--bindingdb-mapping", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one explicitly selected external acquisition stage."""

    args = build_argument_parser().parse_args(argv)
    if args.source == "bindingdb":
        result = acquire_bindingdb(args.raw_root)
    elif args.source == "clinicaltrials":
        result = acquire_clinicaltrials_complete(args.raw_root)
    elif args.source == "drugsfda":
        result = acquire_drugsfda(args.raw_root)
    elif args.source == "dailymed":
        result = acquire_dailymed(args.raw_root)
    elif args.source == "dailymed-part":
        if args.part_number is None or not 1 <= args.part_number <= len(DAILYMED_PARTS):
            raise SystemExit(f"dailymed-part requires --part-number in 1..{len(DAILYMED_PARTS)}")
        result = acquire_dailymed_part(args.raw_root, DAILYMED_PARTS[args.part_number - 1])
    else:
        if args.chembl_target_components is None or args.bindingdb_mapping is None:
            raise SystemExit("uniprot requires --chembl-target-components and --bindingdb-mapping")
        result = acquire_uniprot_targeted(
            args.raw_root,
            chembl_target_components=args.chembl_target_components,
            bindingdb_mapping=args.bindingdb_mapping,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
