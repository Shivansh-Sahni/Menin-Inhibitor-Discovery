"""Bounded, fail-closed PK-DB public metadata candidate audit.

The module deliberately separates HTTP acquisition from deterministic
normalization.  It inventories official PK-DB endpoint definitions and a tiny
public-access probe, but it never emits a PK observation, compound link, or
training label.  If output records or their usage rights are not reproducible
without credentials, canonical admission is zero by construction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
import tempfile
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlparse

import requests

SCHEMA_VERSION = "platform-pkdb-public-candidates/1.0"
ACQUISITION_VERSION = "platform_pkdb_candidates/acquire-1.0"
NORMALIZER_VERSION = "platform_pkdb_candidates/normalize-1.1"
SOURCE_ID = "pkdb_public_api_v1"
OFFICIAL_ORIGIN = "https://pk-db.com"
API_ROOT = f"{OFFICIAL_ORIGIN}/api/v1"
MAX_RESPONSE_BYTES = 1_048_576
USER_AGENT = "Menin-platform-PKDB-metadata-audit/1.0"

TARGET_INFO_NODES = (
    "auc-end",
    "auc-inf",
    "bioavailability",
    "clearance",
    "cmax",
    "fraction-unbound",
    "thalf",
    "tmax",
    "vd",
    "vd-ss",
)

RAW_RETAINED_REQUESTS = (
    ("statistics", f"{API_ROOT}/statistics/", "statistics.json", MAX_RESPONSE_BYTES),
    (
        "openapi",
        f"{API_ROOT}/swagger/?format=openapi",
        "openapi.json",
        MAX_RESPONSE_BYTES,
    ),
    (
        "outputs_anonymous_probe",
        f"{API_ROOT}/outputs/?page_size=1",
        "outputs_anonymous_probe.json",
        262_144,
    ),
)

SENSITIVE_TEXT_KEYS = {
    "abstract",
    "authors",
    "comment",
    "description",
    "reference",
    "short_name",
    "title",
}


class PKDBCandidateError(RuntimeError):
    """Raised when the audit cannot prove its immutable, fail-closed state."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def document_with_sha256(document: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(document)
    result.pop("document_sha256", None)
    result["document_sha256"] = sha256_bytes(canonical_json_bytes(result))
    return result


def verify_document_sha256(document: Mapping[str, Any]) -> bool:
    expected = document.get("document_sha256")
    if not isinstance(expected, str):
        return False
    body = dict(document)
    body.pop("document_sha256", None)
    return expected == sha256_bytes(canonical_json_bytes(body))


def _write_json(path: Path, value: Any, *, self_hash: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = document_with_sha256(value) if self_hash else value
    data = json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True).encode("utf-8") + b"\n"
    _atomic_write(path, data)


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _safe_relative_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise PKDBCandidateError(f"unsafe relative path: {value!r}")
    return path


def _safe_existing_directory(path: Path, *, context: str) -> Path:
    if path.is_symlink() or not path.is_dir():
        raise PKDBCandidateError(f"missing or symlinked {context}: {path}")
    return path.resolve()


def _closed_regular_files(root: Path, *, context: str) -> set[str]:
    observed: set[str] = set()
    for directory, directory_names, file_names in os.walk(root, followlinks=False):
        current = Path(directory)
        for name in directory_names:
            candidate = current / name
            if candidate.is_symlink():
                raise PKDBCandidateError(f"symlinked directory in {context}: {candidate}")
        for name in file_names:
            candidate = current / name
            if not stat.S_ISREG(candidate.lstat().st_mode):
                raise PKDBCandidateError(f"non-regular file in {context}: {candidate}")
            observed.add(candidate.relative_to(root).as_posix())
    return observed


def _official_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != "pk-db.com" or parsed.port is not None:
        raise PKDBCandidateError(f"request escaped official PK-DB HTTPS origin: {url}")


def _get_bounded(session: requests.Session, url: str, *, max_bytes: int) -> tuple[bytes, dict[str, Any]]:
    """GET one official URL without redirects and enforce a hard byte ceiling."""

    _official_url(url)
    with session.get(url, timeout=(10, 30), allow_redirects=False, stream=True) as response:
        if response.is_redirect:
            raise PKDBCandidateError(f"redirect not allowed for immutable acquisition: {url}")
        response.raise_for_status()
        content_length = response.headers.get("Content-Length")
        if content_length and int(content_length) > max_bytes:
            raise PKDBCandidateError(f"declared response too large for {url}: {content_length}")
        body = bytearray()
        for chunk in response.iter_content(chunk_size=65_536):
            body.extend(chunk)
            if len(body) > max_bytes:
                raise PKDBCandidateError(f"response exceeded byte ceiling for {url}")
        final_url = str(response.url)
        _official_url(final_url)
        receipt = {
            "request_method": "GET",
            "request_url": url,
            "final_url": final_url,
            "request_user_agent": USER_AGENT,
            "request_accept": session.headers.get("Accept", "*/*"),
            "http_status": response.status_code,
            "response_headers": {
                key.lower(): value
                for key, value in response.headers.items()
                if key.lower()
                in {
                    "allow",
                    "content-length",
                    "content-type",
                    "date",
                    "etag",
                    "last-modified",
                }
            },
            "body_bytes": len(body),
            "body_sha256": sha256_bytes(bytes(body)),
            "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        return bytes(body), receipt


def _json_object(body: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PKDBCandidateError(f"{label} was not valid JSON") from exc
    if not isinstance(value, dict):
        raise PKDBCandidateError(f"{label} JSON was not an object")
    return value


def _page_records(payload: Mapping[str, Any]) -> tuple[int, list[dict[str, Any]]]:
    """Support the public API's nested pagination envelope, rejecting surprises."""

    data = payload.get("data")
    if not isinstance(data, Mapping):
        raise PKDBCandidateError("API page omitted data envelope")
    count = data.get("count")
    records = data.get("data")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise PKDBCandidateError("API page count was not a non-negative integer")
    if not isinstance(records, list) or any(not isinstance(item, dict) for item in records):
        raise PKDBCandidateError("API page records were malformed")
    return count, records


def sanitize_study_probe(payload: Mapping[str, Any]) -> dict[str, Any]:
    count, records = _page_records(payload)
    safe_records: list[dict[str, Any]] = []
    for record in records:
        safe_records.append(
            {
                "sid": record.get("sid"),
                "access": record.get("access"),
                "licence": record.get("licence"),
                "group_count": record.get("group_count"),
                "individual_count": record.get("individual_count"),
                "intervention_count": record.get("intervention_count"),
                "output_count": record.get("output_count"),
                "timecourse_count": record.get("timecourse_count"),
                "substance_sids": sorted(
                    str(item.get("sid"))
                    for item in record.get("substances", [])
                    if isinstance(item, Mapping) and item.get("sid")
                ),
            }
        )
    return {
        "reported_count": count,
        "returned_record_count": len(records),
        "records": safe_records,
        "text_minimization": "abstract/title/description/authors/reference fields intentionally discarded",
    }


def sanitize_intervention_probe(payload: Mapping[str, Any]) -> dict[str, Any]:
    count, records = _page_records(payload)
    safe_records: list[dict[str, Any]] = []
    for record in records:
        forbidden = SENSITIVE_TEXT_KEYS.intersection(record)
        safe_records.append(
            {
                "pk": record.get("pk"),
                "study_sid": _sid(record.get("study_sid") or record.get("study")),
                "measurement_type_sid": _sid(record.get("measurement_type_sid")),
                "route_sid": _sid(record.get("route")),
                "form_sid": _sid(record.get("form")),
                "application_sid": _sid(record.get("application")),
                "time_present": record.get("time") is not None,
                "time_unit": record.get("time_unit"),
                "dose_value_present": record.get("value") is not None,
                "dose_unit": record.get("unit"),
                "substance_sid": _sid(record.get("substance_sid") or record.get("substance")),
                "discarded_sensitive_text_field_count": len(forbidden),
            }
        )
    return {
        "reported_count": count,
        "returned_record_count": len(records),
        "records": safe_records,
        "value_policy": "numeric dose values intentionally discarded; presence and reported unit retained",
    }


def _sid(value: Any) -> str | None:
    if isinstance(value, Mapping):
        value = value.get("sid")
    return str(value) if value not in (None, "") else None


def _artifact(path: Path, root: Path, role: str) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "role": role,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def acquire(raw_root: Path) -> dict[str, Any]:
    """Acquire the small official metadata snapshot and privacy-minimized probes."""

    if raw_root.exists() or raw_root.is_symlink():
        raise PKDBCandidateError(f"raw output root already exists: {raw_root}")
    raw_root.mkdir(parents=True, exist_ok=False)
    session = requests.Session()
    # The documented Swagger endpoint serves application/openapi+json and
    # responds 406 to an application/json-only Accept header.
    session.headers.update({"Accept": "*/*", "User-Agent": USER_AGENT})
    receipts: list[dict[str, Any]] = []

    for request_id, url, relative, max_bytes in RAW_RETAINED_REQUESTS:
        body, receipt = _get_bounded(session, url, max_bytes=max_bytes)
        _json_object(body, request_id)
        path = raw_root / relative
        _atomic_write(path, body)
        receipt.update(
            {
                "request_id": request_id,
                "retention": "exact_raw_body_retained",
                "retained_path": relative,
            }
        )
        receipts.append(receipt)

    info_dir = raw_root / "info_nodes"
    for sid in TARGET_INFO_NODES:
        body, receipt = _get_bounded(
            session,
            f"{API_ROOT}/info_nodes/{sid}/",
            max_bytes=131_072,
        )
        _json_object(body, f"info node {sid}")
        relative = f"info_nodes/{sid}.json"
        _atomic_write(info_dir / f"{sid}.json", body)
        receipt.update(
            {
                "request_id": f"info_node:{sid}",
                "retention": "exact_raw_body_retained",
                "retained_path": relative,
            }
        )
        receipts.append(receipt)

    minimized = (
        (
            "studies_access_probe",
            f"{API_ROOT}/studies/?page_size=1",
            "studies_access_probe_sanitized.json",
            sanitize_study_probe,
        ),
        (
            "interventions_context_probe",
            f"{API_ROOT}/interventions/?page_size=1",
            "interventions_context_probe_sanitized.json",
            sanitize_intervention_probe,
        ),
    )
    closed_licence_seen = False
    for request_id, url, relative, sanitizer in minimized:
        body, receipt = _get_bounded(session, url, max_bytes=524_288)
        sanitized = sanitizer(_json_object(body, request_id))
        if request_id == "studies_access_probe":
            closed_licence_seen = any(
                isinstance(record, Mapping) and record.get("licence") == "closed"
                for record in sanitized.get("records", [])
            )
        _write_json(raw_root / relative, sanitized, self_hash=True)
        receipt.update(
            {
                "request_id": request_id,
                "retention": "raw_body_discarded_after_in_memory_minimization",
                "retained_path": relative,
                "retained_body_is_sanitized": True,
            }
        )
        receipts.append(receipt)

    _write_json(
        raw_root / "http_receipts.json",
        {
            "schema_version": SCHEMA_VERSION,
            "source_id": SOURCE_ID,
            "acquisition_version": ACQUISITION_VERSION,
            "receipts": receipts,
        },
        self_hash=True,
    )
    _write_json(
        raw_root / "rights_snapshot.json",
        {
            "schema_version": SCHEMA_VERSION,
            "source_id": SOURCE_ID,
            "scope": "API metadata snapshot only",
            "terms_pointer_source": "openapi.json#/info/termsOfService",
            "api_license_pointer_source": "openapi.json#/info/license",
            "data_record_rights_status": "unresolved_requires_record_level_and_terms_review",
            "closed_licence_seen_in_public_access_probe": closed_licence_seen,
            "canonical_admission_allowed": False,
        },
        self_hash=True,
    )

    artifacts = [
        _artifact(path, raw_root, "frozen_official_or_sanitized_source_artifact")
        for path in sorted(raw_root.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    ]
    manifest = document_with_sha256(
        {
            "schema_version": SCHEMA_VERSION,
            "source_id": SOURCE_ID,
            "acquisition_version": ACQUISITION_VERSION,
            "official_origin": OFFICIAL_ORIGIN,
            "artifact_count": len(artifacts),
            "artifacts": artifacts,
            "canonical_observation_count": 0,
            "training_label_count": 0,
        }
    )
    _write_json(raw_root / "manifest.json", manifest)
    return manifest


def _load_bound_raw(raw_root: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    raw_root = _safe_existing_directory(raw_root, context="raw PK-DB root")
    manifest_path = raw_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not verify_document_sha256(manifest):
        raise PKDBCandidateError("raw manifest self-hash failed")
    if manifest.get("source_id") != SOURCE_ID:
        raise PKDBCandidateError("raw manifest source mismatch")
    if manifest.get("canonical_observation_count") != 0 or manifest.get("training_label_count") != 0:
        raise PKDBCandidateError("raw manifest violated zero-admission invariant")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != manifest.get("artifact_count"):
        raise PKDBCandidateError("raw artifact inventory malformed")
    expected = {"manifest.json"}
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            raise PKDBCandidateError("raw artifact entry malformed")
        relative = _safe_relative_path(str(artifact.get("path", "")))
        expected.add(relative.as_posix())
        path = raw_root / Path(*relative.parts)
        if path.is_symlink() or not path.is_file():
            raise PKDBCandidateError(f"missing or symlinked raw artifact: {relative}")
        if path.stat().st_size != artifact.get("bytes") or sha256_file(path) != artifact.get("sha256"):
            raise PKDBCandidateError(f"raw artifact binding failed: {relative}")
    if _closed_regular_files(raw_root, context="raw PK-DB root") != expected:
        raise PKDBCandidateError("raw artifact membership drift")

    names = [
        "statistics.json",
        "openapi.json",
        "outputs_anonymous_probe.json",
        "studies_access_probe_sanitized.json",
        "interventions_context_probe_sanitized.json",
        "rights_snapshot.json",
    ]
    names += [f"info_nodes/{sid}.json" for sid in TARGET_INFO_NODES]
    payloads: dict[str, dict[str, Any]] = {}
    for payload_name in names:
        value = json.loads((raw_root / payload_name).read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise PKDBCandidateError(f"raw JSON was not an object: {payload_name}")
        if payload_name.endswith("_sanitized.json") or payload_name == "rights_snapshot.json":
            if not verify_document_sha256(value):
                raise PKDBCandidateError(f"sanitized/rights document self-hash failed: {payload_name}")
        payloads[payload_name] = value
    return payloads, manifest


def _info_node_inventory(sid: str, node: Mapping[str, Any]) -> dict[str, Any]:
    measurement_type = node.get("measurement_type")
    nested_units = measurement_type.get("units") if isinstance(measurement_type, Mapping) else None
    allowed = {
        "sid": sid,
        "name": node.get("name"),
        "label": node.get("label"),
        "type": node.get("type") or node.get("ntype"),
        "dtype": node.get("dtype"),
        "deprecated": node.get("deprecated"),
        "unit": node.get("unit"),
        "allowed_units": node.get("allowed_units") or node.get("units") or nested_units,
        "parent_sids": sorted(
            str(parent.get("sid"))
            for parent in node.get("parents", [])
            if isinstance(parent, Mapping) and parent.get("sid")
        ),
        "raw_pointer": f"info_nodes/{sid}.json",
    }
    return allowed


def normalize(raw_root: Path, interim_root: Path, report_root: Path) -> dict[str, Any]:
    """Create deterministic inventories and a fail-closed admission decision."""

    payloads, raw_manifest = _load_bound_raw(raw_root)
    if interim_root.exists() or interim_root.is_symlink():
        raise PKDBCandidateError(f"interim output root already exists: {interim_root}")
    if report_root.exists() or report_root.is_symlink():
        raise PKDBCandidateError(f"report output root already exists: {report_root}")
    statistics = payloads["statistics.json"]
    openapi = payloads["openapi.json"]
    outputs = payloads["outputs_anonymous_probe.json"]
    studies_probe = payloads["studies_access_probe_sanitized.json"]
    interventions_probe = payloads["interventions_context_probe_sanitized.json"]
    output_count, output_records = _page_records(outputs)
    reported_outputs = statistics.get("output_count")
    if isinstance(reported_outputs, bool) or not isinstance(reported_outputs, int):
        raise PKDBCandidateError("statistics output_count malformed")

    info = openapi.get("info")
    security = openapi.get("security")
    if not isinstance(info, Mapping):
        raise PKDBCandidateError("OpenAPI info object missing")
    auth_declared = bool(security)
    access_reproducible = not (reported_outputs > 0 and output_count == 0 and not output_records)
    rights_resolved = False
    blockers = []
    if not access_reproducible:
        blockers.append(
            "official statistics report positive output_count but anonymous outputs endpoint returned zero records"
        )
    if auth_declared:
        blockers.append("OpenAPI declares API-wide authentication security")
    blockers.append(
        "record-level data reuse rights remain unresolved; public access is not equivalent to open licence"
    )

    endpoints = [_info_node_inventory(sid, payloads[f"info_nodes/{sid}.json"]) for sid in TARGET_INFO_NODES]
    endpoint_inventory = document_with_sha256(
        {
            "schema_version": SCHEMA_VERSION,
            "source_id": SOURCE_ID,
            "normalizer_version": NORMALIZER_VERSION,
            "raw_manifest_sha256": sha256_file(raw_root / "manifest.json"),
            "semantics_policy": "each ontology SID remains a separate endpoint; no AUC, volume, or half-life pooling",
            "endpoints": endpoints,
            "endpoint_count": len(endpoints),
        }
    )
    decision = document_with_sha256(
        {
            "schema_version": SCHEMA_VERSION,
            "source_id": SOURCE_ID,
            "normalizer_version": NORMALIZER_VERSION,
            "pkdb_reported_output_count": reported_outputs,
            "anonymous_outputs_probe_count": output_count,
            "anonymous_outputs_returned_records": len(output_records),
            "openapi_version": info.get("version"),
            "server_software_version": statistics.get("version"),
            "openapi_global_security_declared": auth_declared,
            "anonymous_output_access_reproducible": access_reproducible,
            "record_level_reuse_rights_resolved": rights_resolved,
            "canonical_admission": False,
            "candidate_observation_rows": 0,
            "canonical_observation_rows": 0,
            "training_label_rows": 0,
            "compound_identity_links": 0,
            "blockers": blockers,
            "next_gate": "obtain documented public output access and legal/record-level licence review, then reacquire",
        }
    )
    context = document_with_sha256(
        {
            "schema_version": SCHEMA_VERSION,
            "source_id": SOURCE_ID,
            "fields_required_before_future_observation_admission": [
                "study_sid",
                "study_access",
                "study_licence",
                "species",
                "route",
                "matrix_or_tissue",
                "dose_value_and_unit",
                "time_value_and_unit",
                "endpoint_sid",
                "endpoint_value_and_unit",
                "raw_record_pointer",
            ],
            "coverage_measurement_status": "not_measurable_from_anonymous_zero-record output response",
            "public_count_consistency_audit": {
                "statistics_study_count": statistics.get("study_count"),
                "studies_query_reported_count": studies_probe.get("reported_count"),
                "study_counts_match": statistics.get("study_count") == studies_probe.get("reported_count"),
                "statistics_intervention_count": statistics.get("intervention_count"),
                "interventions_query_reported_count": interventions_probe.get("reported_count"),
                "intervention_counts_match": statistics.get("intervention_count")
                == interventions_probe.get("reported_count"),
                "interpretation": "query/statistics definitions or access filters differ; counts are not interchangeable",
            },
            "unknown_is_not_absent": True,
            "numeric_values_emitted": 0,
        }
    )

    interim_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)
    _write_json(interim_root / "endpoint_inventory.json", endpoint_inventory)
    _write_json(interim_root / "context_availability.json", context)
    _write_json(interim_root / "admission_decision.json", decision)

    report = document_with_sha256(
        {
            "schema_version": SCHEMA_VERSION,
            "source_id": SOURCE_ID,
            "status": "fail_closed_zero_admission",
            "raw_manifest_document_sha256": raw_manifest["document_sha256"],
            "input_manifest_file_sha256": sha256_file(raw_root / "manifest.json"),
            "code_sha256": sha256_file(Path(__file__)),
            "endpoint_inventory_document_sha256": endpoint_inventory["document_sha256"],
            "context_availability_document_sha256": context["document_sha256"],
            "admission_decision_document_sha256": decision["document_sha256"],
            "official_statistics": {
                "study_count": statistics.get("study_count"),
                "output_count": reported_outputs,
                "version": statistics.get("version"),
            },
            "anonymous_output_probe": {"count": output_count, "records": len(output_records)},
            "endpoint_semantics_inventory_count": len(endpoints),
            "public_count_consistency_audit": context["public_count_consistency_audit"],
            "canonical_observation_rows": 0,
            "training_label_rows": 0,
            "finding": "PK-DB is a relevant PK source, but its output records and reusable rights were not anonymously reproducible in this snapshot.",
            "blockers": blockers,
        }
    )
    _write_json(report_root / "report.json", report)
    summary = f"""# PK-DB public candidate audit

## Outcome

**Fail closed: 0 candidate observations, 0 canonical observations, and 0 training labels.**

The official statistics endpoint reported **{reported_outputs:,} outputs**, while the anonymous
outputs probe returned **{output_count}** records. The official OpenAPI document also declares
API-wide authentication, and the retained public study probe demonstrates that `access=public`
can coexist with a closed licence. Therefore no numeric PK result, compound link, or model label
was admitted.

The public count surfaces are also not interchangeable: statistics reported **{statistics.get("study_count")}**
studies and **{statistics.get("intervention_count")}** interventions, versus **{studies_probe.get("reported_count")}**
and **{interventions_probe.get("reported_count")}** from the corresponding anonymous list queries. This may reflect
different definitions or access filters and must be resolved before any coverage claim.

## What was retained

- Exact official statistics, OpenAPI, anonymous output probe, and ten PK ontology nodes.
- HTTP status/date/version/body-size/SHA-256 receipts and a complete artifact manifest.
- Privacy-minimized study/intervention probes; publication text and numeric dose values were discarded.
- Separate semantics for AUC-end, AUC-infinity, clearance, half-life, distribution volume,
  steady-state volume, Cmax, Tmax, bioavailability, and fraction unbound.

## Gate for future use

Obtain documented reproducible output access, complete a record-level/terms licence review, and
then measure study/species/route/matrix/dose/time/unit context completeness. Unknown fields must
not be interpreted as absent, and endpoint families must not be pooled without an explicit model.
"""
    _atomic_write(report_root / "summary.md", summary.encode("utf-8"))

    output_artifacts = []
    for root, prefix in ((interim_root, "interim"), (report_root, "report")):
        for path in sorted(root.iterdir()):
            if path.is_file() and path.name != "manifest.json":
                output_artifacts.append(
                    {
                        "path": f"{prefix}/{path.name}",
                        "bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                )
    output_manifest = document_with_sha256(
        {
            "schema_version": SCHEMA_VERSION,
            "source_id": SOURCE_ID,
            "normalizer_version": NORMALIZER_VERSION,
            "input_manifest_file_sha256": sha256_file(raw_root / "manifest.json"),
            "code_sha256": sha256_file(Path(__file__)),
            "artifacts": output_artifacts,
            "artifact_count": len(output_artifacts),
            "canonical_observation_rows": 0,
            "training_label_rows": 0,
        }
    )
    _write_json(interim_root / "manifest.json", output_manifest)
    _write_json(report_root / "manifest.json", output_manifest)
    return report


def verify(raw_root: Path, interim_root: Path, report_root: Path) -> dict[str, Any]:
    """Verify all manifests, hashes, invariants, and deterministic normalized replay."""

    _load_bound_raw(raw_root)
    interim_root = _safe_existing_directory(interim_root, context="PK-DB interim root")
    report_root = _safe_existing_directory(report_root, context="PK-DB report root")
    manifests: list[dict[str, Any]] = []
    for root in (interim_root, report_root):
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        if not isinstance(manifest, dict) or not verify_document_sha256(manifest):
            raise PKDBCandidateError(f"output manifest self-hash failed: {root}")
        if manifest.get("canonical_observation_rows") != 0 or manifest.get("training_label_rows") != 0:
            raise PKDBCandidateError("output manifest violated zero-admission invariant")
        if manifest.get("code_sha256") != sha256_file(Path(__file__)):
            raise PKDBCandidateError("normalizer code/output binding failed")
        artifacts = manifest.get("artifacts")
        if not isinstance(artifacts, list) or len(artifacts) != manifest.get("artifact_count"):
            raise PKDBCandidateError("output artifact inventory malformed")
        manifests.append(manifest)
        for artifact in artifacts:
            relative = _safe_relative_path(str(artifact.get("path", "")))
            prefix, *rest = relative.parts
            artifact_root = (
                interim_root if prefix == "interim" else report_root if prefix == "report" else None
            )
            if artifact_root is None or not rest:
                raise PKDBCandidateError("output manifest path escaped expected roots")
            path = artifact_root.joinpath(*rest)
            if not path.is_file() or path.is_symlink():
                raise PKDBCandidateError(f"missing output artifact: {relative}")
            if path.stat().st_size != artifact.get("bytes") or sha256_file(path) != artifact.get("sha256"):
                raise PKDBCandidateError(f"output artifact binding failed: {relative}")

    if manifests[0] != manifests[1]:
        raise PKDBCandidateError("interim/report manifest copies differ")
    artifacts = manifests[0]["artifacts"]
    for prefix, root in (("interim", interim_root), ("report", report_root)):
        expected_members = {"manifest.json"}
        expected_members.update(
            str(artifact["path"]).split("/", 1)[1]
            for artifact in artifacts
            if str(artifact.get("path", "")).startswith(f"{prefix}/")
        )
        if _closed_regular_files(root, context=f"PK-DB {prefix} root") != expected_members:
            raise PKDBCandidateError(f"{prefix} output membership drift")

    decision = json.loads((interim_root / "admission_decision.json").read_text(encoding="utf-8"))
    if not verify_document_sha256(decision):
        raise PKDBCandidateError("admission decision self-hash failed")
    if any(
        decision.get(key) != 0
        for key in (
            "candidate_observation_rows",
            "canonical_observation_rows",
            "training_label_rows",
            "compound_identity_links",
        )
    ):
        raise PKDBCandidateError("nonzero admitted record count")
    if decision.get("canonical_admission") is not False:
        raise PKDBCandidateError("canonical admission was not false")

    with tempfile.TemporaryDirectory(prefix="pkdb-replay-") as temporary:
        replay_root = Path(temporary)
        replay_interim = replay_root / "interim"
        replay_report = replay_root / "report"
        normalize(raw_root, replay_interim, replay_report)
        comparisons: list[dict[str, Any]] = []
        for replay_name in (
            "endpoint_inventory.json",
            "context_availability.json",
            "admission_decision.json",
            "manifest.json",
        ):
            expected_path = interim_root / replay_name
            actual = replay_interim / replay_name
            same = expected_path.read_bytes() == actual.read_bytes()
            comparisons.append({"path": f"interim/{replay_name}", "byte_identical": same})
            if not same:
                raise PKDBCandidateError(f"normalized replay mismatch: interim/{replay_name}")
        for replay_name in ("report.json", "summary.md", "manifest.json"):
            expected_path = report_root / replay_name
            actual = replay_report / replay_name
            same = expected_path.read_bytes() == actual.read_bytes()
            comparisons.append({"path": f"report/{replay_name}", "byte_identical": same})
            if not same:
                raise PKDBCandidateError(f"normalized replay mismatch: report/{replay_name}")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "verified_fail_closed",
        "byte_identical_replay": True,
        "comparisons": comparisons,
        "canonical_observation_rows": 0,
        "training_label_rows": 0,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("acquire", "normalize", "verify", "all"):
        command = subparsers.add_parser(name)
        command.add_argument("--raw-root", type=Path, required=True)
        if name != "acquire":
            command.add_argument("--interim-root", type=Path, required=True)
            command.add_argument("--report-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command in {"acquire", "all"}:
            acquire(args.raw_root)
        if args.command in {"normalize", "all"}:
            normalize(args.raw_root, args.interim_root, args.report_root)
        if args.command in {"verify", "all"}:
            result = verify(args.raw_root, args.interim_root, args.report_root)
            print(json.dumps(result, sort_keys=True))
        return 0
    except (OSError, ValueError, requests.RequestException, PKDBCandidateError) as exc:
        print(f"PK-DB candidate audit failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
