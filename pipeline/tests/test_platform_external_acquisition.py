from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import pandas as pd
import pytest
from menin_discovery import platform_external_acquisition as ext


class FakeResponse:
    def __init__(
        self,
        content: bytes,
        *,
        url: str = "https://resolved.example/artifact",
        status_code: int = 200,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.content = content
        self.url = url
        self.status_code = status_code
        self.headers = headers or {
            "Content-Length": str(len(content)),
            "Content-Type": "application/octet-stream",
        }

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def iter_content(self, chunk_size: int) -> Any:
        for start in range(0, len(self.content), chunk_size):
            yield self.content[start : start + chunk_size]


class FakeSession:
    def __init__(
        self,
        responses: dict[str, FakeResponse | list[FakeResponse]] | None = None,
    ) -> None:
        self.headers: dict[str, str] = {}
        self.responses = responses or {}
        self.calls: list[dict[str, Any]] = []

    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append({"url": url, **kwargs})
        key = url
        if key not in self.responses:
            raise AssertionError(f"Unexpected URL {url}")
        response = self.responses[key]
        if isinstance(response, list):
            if not response:
                raise AssertionError(f"No responses remain for {url}")
            return response.pop(0)
        return response


def make_zip(path: Path, members: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zipped:
        for name, content in members.items():
            zipped.writestr(name, content)


def test_manifest_identity_is_non_circular_and_detects_change(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    document = ext.atomic_write_json(path, {"source": "fixture", "count": 2})
    assert ext.verify_document_sha256(document)
    loaded = json.loads(path.read_text())
    loaded["count"] = 3
    assert not ext.verify_document_sha256(loaded)


def test_download_new_bytes_persists_immutable_sidecar(tmp_path: Path) -> None:
    content = b"abcdef" * 100
    url = "https://example.test/data.bin"
    response = FakeResponse(
        content,
        url="https://cdn.example.test/data.bin",
        headers={
            "Content-Length": str(len(content)),
            "ETag": '"fixture"',
            "Set-Cookie": "must-not-be-recorded",
        },
    )
    session = FakeSession({url: response})
    destination = tmp_path / "data.bin"
    record = ext.download_immutable(
        ext.DownloadSpec(
            canonical_url=url,
            destination=destination,
            expected_bytes=len(content),
            expected_sha256=hashlib.sha256(content).hexdigest(),
        ),
        session=session,  # type: ignore[arg-type]
        chunk_bytes=17,
    )
    assert destination.read_bytes() == content
    assert record["acquired_sha256"] == hashlib.sha256(content).hexdigest()
    assert record["resolved_url"] == "https://cdn.example.test/data.bin"
    assert "set-cookie" not in record["response_headers"]
    assert ext.verify_document_sha256(record)
    sidecar = destination.with_name("data.bin.acquisition.json")
    assert json.loads(sidecar.read_text()) == record

    second = ext.download_immutable(
        ext.DownloadSpec(canonical_url=url, destination=destination),
        session=FakeSession(),  # type: ignore[arg-type]
    )
    assert second == record


def test_download_resumes_only_exact_range(tmp_path: Path) -> None:
    url = "https://example.test/resume.bin"
    destination = tmp_path / "resume.bin"
    partial = tmp_path / ".resume.bin.part"
    partial.write_bytes(b"abc")
    spec = ext.DownloadSpec(canonical_url=url, destination=destination, expected_bytes=6)
    contract = ext._partial_download_contract(spec)
    ext.atomic_write_json(
        tmp_path / ".resume.bin.part.json",
        {
            "canonical_url": url,
            "etag": '"v1"',
            "last_modified": None,
            "partial_bytes": 3,
            "partial_sha256": hashlib.sha256(b"abc").hexdigest(),
            "download_contract": contract,
            "download_contract_sha256": hashlib.sha256(ext.canonical_json_bytes(contract)).hexdigest(),
        },
    )
    response = FakeResponse(
        b"def",
        status_code=206,
        headers={"Content-Length": "3", "Content-Range": "bytes 3-5/6", "ETag": '"v1"'},
    )
    session = FakeSession({url: response})
    record = ext.download_immutable(
        spec,
        session=session,  # type: ignore[arg-type]
        chunk_bytes=2,
    )
    assert destination.read_bytes() == b"abcdef"
    assert record["resumed_from_bytes"] == 3
    assert session.calls[0]["headers"]["Range"] == "bytes=3-"
    assert session.calls[0]["headers"]["If-Range"] == '"v1"'


def test_server_ignoring_range_restarts_partial(tmp_path: Path) -> None:
    url = "https://example.test/restart.bin"
    destination = tmp_path / "restart.bin"
    partial = tmp_path / ".restart.bin.part"
    partial.write_bytes(b"stale-prefix")
    spec = ext.DownloadSpec(canonical_url=url, destination=destination, expected_bytes=5)
    contract = ext._partial_download_contract(spec)
    ext.atomic_write_json(
        tmp_path / ".restart.bin.part.json",
        {
            "canonical_url": url,
            "etag": '"v1"',
            "partial_bytes": len(b"stale-prefix"),
            "partial_sha256": hashlib.sha256(b"stale-prefix").hexdigest(),
            "download_contract": contract,
            "download_contract_sha256": hashlib.sha256(ext.canonical_json_bytes(contract)).hexdigest(),
        },
    )
    response = FakeResponse(b"fresh", status_code=200)
    record = ext.download_immutable(
        spec,
        session=FakeSession({url: response}),  # type: ignore[arg-type]
    )
    assert destination.read_bytes() == b"fresh"
    assert record["resumed_from_bytes"] == 0
    assert record["partial_restart_reason"] == "server_did_not_honor_exact_range"


def test_changed_partial_contract_is_never_sent_as_range(tmp_path: Path) -> None:
    url = "https://example.test/changed-contract.bin"
    destination = tmp_path / "changed-contract.bin"
    partial = tmp_path / ".changed-contract.bin.part"
    partial.write_bytes(b"old")
    old_spec = ext.DownloadSpec(canonical_url=url, destination=destination, expected_bytes=6)
    old_contract = ext._partial_download_contract(old_spec)
    ext.atomic_write_json(
        tmp_path / ".changed-contract.bin.part.json",
        {
            "canonical_url": url,
            "partial_bytes": 3,
            "partial_sha256": hashlib.sha256(b"old").hexdigest(),
            "download_contract": old_contract,
            "download_contract_sha256": hashlib.sha256(ext.canonical_json_bytes(old_contract)).hexdigest(),
        },
    )
    new_spec = ext.DownloadSpec(canonical_url=url, destination=destination, expected_bytes=5)
    session = FakeSession({url: FakeResponse(b"fresh")})
    record = ext.download_immutable(
        new_spec,
        session=session,  # type: ignore[arg-type]
    )
    assert destination.read_bytes() == b"fresh"
    assert "Range" not in session.calls[0]["headers"]
    assert record["partial_restart_reason"] == "partial_expected_contract_changed"


def test_partial_without_strong_validator_or_digest_restarts(tmp_path: Path) -> None:
    url = "https://example.test/no-validator.bin"
    destination = tmp_path / "no-validator.bin"
    partial = tmp_path / ".no-validator.bin.part"
    partial.write_bytes(b"old")
    spec = ext.DownloadSpec(canonical_url=url, destination=destination, expected_bytes=5)
    contract = ext._partial_download_contract(spec)
    ext.atomic_write_json(
        tmp_path / ".no-validator.bin.part.json",
        {
            "canonical_url": url,
            "etag": 'W/"weak"',
            "last_modified": "yesterday",
            "partial_bytes": 3,
            "partial_sha256": hashlib.sha256(b"old").hexdigest(),
            "download_contract": contract,
            "download_contract_sha256": hashlib.sha256(ext.canonical_json_bytes(contract)).hexdigest(),
        },
    )
    session = FakeSession({url: FakeResponse(b"fresh")})
    record = ext.download_immutable(spec, session=session)  # type: ignore[arg-type]
    assert destination.read_bytes() == b"fresh"
    assert "Range" not in session.calls[0]["headers"]
    assert record["partial_restart_reason"] == ("partial_has_no_strong_validator_or_published_digest")


def test_unquoted_etag_is_not_a_strong_resume_validator(tmp_path: Path) -> None:
    url = "https://example.test/unquoted-etag.bin"
    destination = tmp_path / "unquoted-etag.bin"
    partial = tmp_path / ".unquoted-etag.bin.part"
    partial.write_bytes(b"old")
    spec = ext.DownloadSpec(canonical_url=url, destination=destination, expected_bytes=5)
    contract = ext._partial_download_contract(spec)
    ext.atomic_write_json(
        tmp_path / ".unquoted-etag.bin.part.json",
        {
            "canonical_url": url,
            "etag": "not-quoted",
            "partial_bytes": 3,
            "partial_sha256": hashlib.sha256(b"old").hexdigest(),
            "download_contract": contract,
            "download_contract_sha256": hashlib.sha256(ext.canonical_json_bytes(contract)).hexdigest(),
        },
    )
    session = FakeSession({url: FakeResponse(b"fresh")})
    record = ext.download_immutable(spec, session=session)  # type: ignore[arg-type]
    assert destination.read_bytes() == b"fresh"
    assert "Range" not in session.calls[0]["headers"]
    assert record["partial_restart_reason"] == ("partial_has_no_strong_validator_or_published_digest")


def test_partial_with_published_digest_may_resume_without_validator(tmp_path: Path) -> None:
    url = "https://example.test/digest-resume.bin"
    destination = tmp_path / "digest-resume.bin"
    partial = tmp_path / ".digest-resume.bin.part"
    partial.write_bytes(b"abc")
    spec = ext.DownloadSpec(
        canonical_url=url,
        destination=destination,
        expected_bytes=6,
        expected_md5=hashlib.md5(b"abcdef", usedforsecurity=False).hexdigest(),
    )
    contract = ext._partial_download_contract(spec)
    ext.atomic_write_json(
        tmp_path / ".digest-resume.bin.part.json",
        {
            "canonical_url": url,
            "partial_bytes": 3,
            "partial_sha256": hashlib.sha256(b"abc").hexdigest(),
            "download_contract": contract,
            "download_contract_sha256": hashlib.sha256(ext.canonical_json_bytes(contract)).hexdigest(),
        },
    )
    session = FakeSession(
        {
            url: FakeResponse(
                b"def",
                status_code=206,
                headers={"Content-Length": "3", "Content-Range": "bytes 3-5/6"},
            )
        }
    )
    record = ext.download_immutable(spec, session=session)  # type: ignore[arg-type]
    assert destination.read_bytes() == b"abcdef"
    assert session.calls[0]["headers"]["Range"] == "bytes=3-"
    assert "If-Range" not in session.calls[0]["headers"]
    assert record["acquired_md5"] == spec.expected_md5


def test_missing_response_etag_restarts_with_fresh_full_request(tmp_path: Path) -> None:
    url = "https://example.test/missing-response-etag.bin"
    destination = tmp_path / "missing-response-etag.bin"
    partial = tmp_path / ".missing-response-etag.bin.part"
    partial.write_bytes(b"abc")
    spec = ext.DownloadSpec(canonical_url=url, destination=destination, expected_bytes=6)
    contract = ext._partial_download_contract(spec)
    ext.atomic_write_json(
        tmp_path / ".missing-response-etag.bin.part.json",
        {
            "canonical_url": url,
            "etag": '"strong-v1"',
            "partial_bytes": 3,
            "partial_sha256": hashlib.sha256(b"abc").hexdigest(),
            "download_contract": contract,
            "download_contract_sha256": hashlib.sha256(ext.canonical_json_bytes(contract)).hexdigest(),
        },
    )
    session = FakeSession(
        {
            url: [
                FakeResponse(
                    b"def",
                    status_code=206,
                    headers={"Content-Length": "3", "Content-Range": "bytes 3-5/6"},
                ),
                FakeResponse(b"abcdef"),
            ]
        }
    )
    record = ext.download_immutable(spec, session=session)  # type: ignore[arg-type]
    assert destination.read_bytes() == b"abcdef"
    assert session.calls[0]["headers"]["If-Range"] == '"strong-v1"'
    assert "Range" not in session.calls[1]["headers"]
    assert record["partial_restart_reason"] == ("resume_response_missing_or_changed_strong_etag")


def test_failed_hash_never_promotes_raw_bytes(tmp_path: Path) -> None:
    url = "https://example.test/bad.bin"
    destination = tmp_path / "bad.bin"
    with pytest.raises(ValueError, match="SHA-256"):
        ext.download_immutable(
            ext.DownloadSpec(canonical_url=url, destination=destination, expected_sha256="0" * 64),
            session=FakeSession({url: FakeResponse(b"content")}),  # type: ignore[arg-type]
        )
    assert not destination.exists()
    assert (tmp_path / ".bad.bin.part").read_bytes() == b"content"


def test_preexisting_bytes_are_verified_and_first_manifested(tmp_path: Path) -> None:
    path = tmp_path / "existing.bin"
    path.write_bytes(b"immutable")
    record = ext.download_immutable(
        ext.DownloadSpec(
            canonical_url="https://example.test/existing",
            destination=path,
            expected_sha256=hashlib.sha256(b"immutable").hexdigest(),
        ),
        session=FakeSession(),  # type: ignore[arg-type]
    )
    assert record["retrieval_utc"] is None
    assert record["acquisition_status"] == "preexisting_bytes_first_manifested_verified"
    assert ext.verify_document_sha256(record)


def test_existing_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "existing.bin"
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        ext.download_immutable(
            ext.DownloadSpec(
                canonical_url="https://example.test/existing",
                destination=path,
                expected_sha256="0" * 64,
            )
        )


def test_zip_inventory_hashes_every_member_and_crc(tmp_path: Path) -> None:
    archive = tmp_path / "fixture.zip"
    make_zip(archive, {"a/file.txt": b"alpha", "b.bin": b"\x00\x01"})
    inventory = tmp_path / "members.jsonl"
    report = ext.verify_zip_archive(archive, inventory, expected_member_count=2)
    rows = [json.loads(line) for line in inventory.read_text().splitlines()]
    assert report["zip_crc_and_stream_integrity"] == "passed"
    assert report["file_member_count"] == 2
    assert [row["archive_member_path"] for row in rows] == ["a/file.txt", "b.bin"]
    assert rows[0]["member_sha256"] == hashlib.sha256(b"alpha").hexdigest()
    assert report["inventory_sha256"] == ext.sha256_file(inventory)


def test_zip_inventory_rejects_unsafe_and_duplicate_names(tmp_path: Path) -> None:
    unsafe = tmp_path / "unsafe.zip"
    make_zip(unsafe, {"../escape": b"x"})
    with pytest.raises(ValueError, match="Unsafe"):
        ext.verify_zip_archive(unsafe, tmp_path / "unsafe.jsonl")

    duplicate = tmp_path / "duplicate.zip"
    with zipfile.ZipFile(duplicate, "w") as zipped:
        zipped.writestr("same", b"one")
        with pytest.warns(UserWarning):
            zipped.writestr("same", b"two")
    with pytest.raises(ValueError, match="Duplicate"):
        ext.verify_zip_archive(duplicate, tmp_path / "duplicate.jsonl")


def test_delimited_zip_counts_logical_quoted_rows(tmp_path: Path) -> None:
    archive = tmp_path / "table.zip"
    make_zip(archive, {"table.tsv": b'id\ttext\n1\t"two\nlines"\n2\tok\n'})
    report = ext.inspect_delimited_zip(archive)
    assert report["total_data_rows"] == 2
    assert report["total_malformed_width_rows"] == 0
    assert report["members"][0]["columns"] == ["id", "text"]


def test_delimited_zip_reports_width_failure(tmp_path: Path) -> None:
    archive = tmp_path / "bad-table.zip"
    make_zip(archive, {"table.tsv": b"a\tb\n1\t2\t3\n"})
    report = ext.inspect_delimited_zip(archive)
    assert report["parse_integrity"] == "failed"
    assert report["total_malformed_width_rows"] == 1


def test_delimited_zip_encoding_fallback_does_not_double_count_partial_attempt(tmp_path: Path) -> None:
    archive = tmp_path / "cp1252.zip"
    make_zip(archive, {"table.tsv": b"id\ttext\n1\tok\n2\tcaf\xe9\n"})
    report = ext.inspect_delimited_zip(archive)
    assert report["members"][0]["encoding"] == "cp1252"
    assert report["total_data_rows"] == 2
    assert report["total_malformed_width_rows"] == 0


def test_fasta_and_text_mapping_integrity(tmp_path: Path) -> None:
    fasta = tmp_path / "targets.fasta"
    fasta.write_text(">one\nACDE\n>two\nFG*\n")
    assert ext.inspect_fasta(fasta) == {
        "parser_version": ext.PARSER_VERSION,
        "record_count": 2,
        "sequence_character_count": 7,
        "blank_sequence_count": 0,
        "parse_integrity": "passed",
    }
    mapping = tmp_path / "map.tsv"
    mapping.write_text("id\taccession\n1\tP00533\n")
    assert ext.inspect_delimited_text(mapping)["data_row_count"] == 1


def test_fasta_rejects_sequence_before_header(tmp_path: Path) -> None:
    path = tmp_path / "bad.fasta"
    path.write_text("ACDE\n")
    with pytest.raises(ValueError, match="before header"):
        ext.inspect_fasta(path)


def test_collect_accessions_is_syntactic_and_deterministic(tmp_path: Path) -> None:
    parquet = tmp_path / "components.parquet"
    pd.DataFrame({"accession": ["P00533", "Q9Y243;P38398", None, "not-an-accession"]}).to_parquet(parquet)
    assert ext.collect_chembl_accessions(parquet) == {"P00533", "Q9Y243", "P38398"}
    mapping = tmp_path / "BindingDB_UniProt.txt"
    mapping.write_text("polymerid\tUniProt\tBindingDB Name\n10\tP00533|Q9Y243\tone\n11\tinvalid\ttwo\n")
    assert ext.collect_bindingdb_accessions(mapping) == {"P00533", "Q9Y243"}
    digest = ext.normalized_accession_digest(["Q9Y243", "P00533", "P00533"])
    assert digest == hashlib.sha256(b"P00533\nQ9Y243\n").hexdigest()


def uniprot_payload() -> bytes:
    return json.dumps(
        {
            "results": [
                {
                    "primaryAccession": "P00533",
                    "secondaryAccessions": ["Q9Y243"],
                    "uniProtkbId": "EGFR_HUMAN",
                    "entryType": "UniProtKB reviewed (Swiss-Prot)",
                    "entryAudit": {
                        "entryVersion": 210,
                        "sequenceVersion": 2,
                        "lastSequenceUpdateDate": "2012-09-19",
                    },
                    "organism": {"scientificName": "Homo sapiens", "taxonId": 9606},
                    "sequence": {
                        "value": "ACDE",
                        "length": 4,
                        "md5": hashlib.md5(b"ACDE", usedforsecurity=False).hexdigest(),
                        "crc64": "y",
                    },
                }
            ]
        }
    ).encode()


def test_uniprot_targeted_reconciles_primary_secondary_and_missing(tmp_path: Path) -> None:
    components = tmp_path / "components.parquet"
    pd.DataFrame(
        {
            "target_chembl_id": ["CHEMBL1", "CHEMBL2", "CHEMBL3"],
            "component_id": [1, 2, 3],
            "accession": ["P00533", "Q9Y243", "Q8N158"],
        }
    ).to_parquet(components)
    mapping = tmp_path / "mapping.tsv"
    mapping.write_text("polymerid\tUniProt\tBindingDB Name\n1\tP00533\tfixture\n")
    metadata_html = b"official"
    response = FakeResponse(
        uniprot_payload(),
        headers={
            "Content-Length": str(len(uniprot_payload())),
            "X-UniProt-Release": ext.UNIPROT_RELEASE_ID,
            "X-UniProt-Release-Date": ext.UNIPROT_RELEASE_HEADER_DATE,
            "X-Total-Results": "1",
            "Content-Type": "application/json",
        },
    )
    session = FakeSession(
        {
            ext.UNIPROT_RELEASE_URL: FakeResponse(metadata_html),
            ext.UNIPROT_LICENSE_URL: FakeResponse(metadata_html),
            ext.UNIPROT_SEARCH_URL: response,
        }
    )
    manifest = ext.acquire_uniprot_targeted(
        tmp_path / "raw",
        chembl_target_components=components,
        bindingdb_mapping=mapping,
        session=session,  # type: ignore[arg-type]
        batch_size=100,
    )
    counts = manifest["resolution_counts"]
    assert counts == {
        "requested": 3,
        "resolved": 2,
        "resolved_secondary_to_primary": 1,
        "missing_or_retired_unresolved": 1,
        "replacement_identified": 0,
        "replacement_resolution_unavailable_for_missing": 1,
        "ambiguous_multi_mapped": 0,
        "non_uniprot_identifier_syntax_quarantine": 0,
        "unique_returned_primary": 1,
    }
    assert manifest["canonical_rows_admitted"] == 0
    assert manifest["model_labels_admitted"] == 0
    assert manifest["snapshot_status"] == "complete_with_explicit_quarantine"
    assert ext.verify_document_sha256(manifest)
    resolution = Path(tmp_path / "raw/uniprotkb_targeted_2026_02/accession_resolution.jsonl")
    assert len(resolution.read_text().splitlines()) == 3


def test_uniprot_release_drift_fails_closed(tmp_path: Path) -> None:
    components = tmp_path / "components.parquet"
    pd.DataFrame({"target_chembl_id": ["CHEMBL1"], "component_id": [1], "accession": ["P00533"]}).to_parquet(
        components
    )
    mapping = tmp_path / "mapping.tsv"
    mapping.write_text("polymerid\tUniProt\tBindingDB Name\n")
    session = FakeSession(
        {
            ext.UNIPROT_RELEASE_URL: FakeResponse(b"release"),
            ext.UNIPROT_LICENSE_URL: FakeResponse(b"license"),
            ext.UNIPROT_SEARCH_URL: FakeResponse(
                uniprot_payload(),
                headers={
                    "X-UniProt-Release": "2026_03",
                    "X-UniProt-Release-Date": ext.UNIPROT_RELEASE_HEADER_DATE,
                    "X-Total-Results": "1",
                    "Content-Length": str(len(uniprot_payload())),
                },
            ),
        }
    )
    with pytest.raises(ValueError, match="release drift"):
        ext.acquire_uniprot_targeted(
            tmp_path / "raw",
            chembl_target_components=components,
            bindingdb_mapping=mapping,
            session=session,  # type: ignore[arg-type]
        )


def test_clinicaltrials_metadata_freezes_schema_without_negative_labels(tmp_path: Path) -> None:
    version = json.dumps({"apiVersion": "2.0.5", "dataTimestamp": "2026-08-04T09:00:05"}).encode()
    schema = b"openapi: 3.0.3\npaths:\n  /studies: {}\n"
    session = FakeSession(
        {
            ext.CLINICALTRIALS_VERSION_URL: FakeResponse(version),
            ext.CLINICALTRIALS_API_DOCS_URL: FakeResponse(b"official API documentation"),
            ext.CLINICALTRIALS_STUDY_METADATA_URL: FakeResponse(b'{"types": {}}'),
            ext.CLINICALTRIALS_OPENAPI_URL: FakeResponse(schema),
            ext.CLINICALTRIALS_TERMS_URL: FakeResponse(b"terms"),
        }
    )
    manifest = ext.acquire_clinicaltrials_metadata(tmp_path, session=session)  # type: ignore[arg-type]
    assert manifest["dataTimestamp"] == "2026-08-04T09:00:05"
    assert manifest["targeted_study_query_status"] == "not_run"
    assert manifest["absence_semantics"] == "no_record_and_no_posted_results_are_not_negative_outcomes"
    assert manifest["model_labels_admitted"] == 0


def test_clinicaltrials_complete_drug_cohort_reconciles_token_chain(tmp_path: Path) -> None:
    version = json.dumps({"apiVersion": "2.0.5", "dataTimestamp": "2026-08-04T09:00:05"}).encode()
    metadata = json.dumps([{"piece": field} for field in ext.CLINICALTRIALS_DRUG_FIELDS]).encode()
    studies = [
        {
            "protocolSection": {
                "identificationModule": {"nctId": "NCT00000001"},
                "statusModule": {
                    "overallStatus": "COMPLETED",
                    "lastUpdatePostDateStruct": {"date": "2026-07-02"},
                },
                "designModule": {"studyType": "INTERVENTIONAL", "phases": ["PHASE2"]},
                "armsInterventionsModule": {"interventions": [{"type": "DRUG", "name": "Fixture drug"}]},
            },
            "hasResults": True,
        },
        {
            "protocolSection": {
                "identificationModule": {"nctId": "NCT00000002"},
                "statusModule": {
                    "overallStatus": "RECRUITING",
                    "lastUpdatePostDateStruct": {"date": "2026-06-01"},
                },
                "designModule": {"studyType": "OBSERVATIONAL"},
                "armsInterventionsModule": {
                    "interventions": [
                        {"type": "PROCEDURE", "name": "Fixture procedure"},
                        {"type": "DRUG", "name": "Second fixture drug"},
                    ]
                },
            },
            "hasResults": False,
        },
    ]
    cohort_payload = json.dumps({"totalCount": 2, "studies": studies}).encode()
    params = {
        "query.term": ext.CLINICALTRIALS_DRUG_QUERY,
        "format": "json",
        "pageSize": "1000",
        "countTotal": "true",
        "sort": ext.CLINICALTRIALS_DRUG_SORT,
        "fields": ",".join(ext.CLINICALTRIALS_DRUG_FIELDS),
    }
    page_url = f"{ext.CLINICALTRIALS_STUDIES_URL}?{urlencode(params)}"
    session = FakeSession(
        {
            ext.CLINICALTRIALS_VERSION_URL: FakeResponse(version),
            ext.CLINICALTRIALS_API_DOCS_URL: FakeResponse(b"official API documentation"),
            ext.CLINICALTRIALS_STUDY_METADATA_URL: FakeResponse(metadata),
            ext.CLINICALTRIALS_OPENAPI_URL: FakeResponse(b"openapi: 3.0.3\npaths:\n  /studies: {}\n"),
            ext.CLINICALTRIALS_TERMS_URL: FakeResponse(b"terms"),
            page_url: FakeResponse(cohort_payload, url=page_url),
        }
    )
    manifest = ext.acquire_clinicaltrials_drug_cohort(
        tmp_path,
        session=session,  # type: ignore[arg-type]
    )
    assert manifest["snapshot_status"] == "complete_unmapped_drug_intervention_cohort"
    assert manifest["reported_total_count"] == 2
    assert manifest["unique_nct_count"] == 2
    assert manifest["page_count"] == 1
    assert manifest["registry_has_results_counts"] == {"false": 1, "true": 1}
    assert manifest["outcome_and_adverse_event_result_fields_retrieved"] is False
    assert manifest["canonical_rows_admitted"] == 0
    assert manifest["model_labels_admitted"] == 0
    assert (
        ext.verify_source_acquisition_manifest(tmp_path / "clinicaltrials_gov_v2", manifest)[
            "verification_status"
        ]
        == "passed"
    )


def test_clinicaltrials_drug_cohort_rejects_filter_leakage() -> None:
    study = {
        "protocolSection": {
            "identificationModule": {"nctId": "NCT00000001"},
            "armsInterventionsModule": {"interventions": [{"type": "DEVICE", "name": "Not a drug"}]},
        },
        "hasResults": False,
    }
    with pytest.raises(ValueError, match="without a DRUG intervention"):
        ext._clinicaltrials_study_audit(study)


def test_clinicaltrials_next_timestamp_starts_new_key_without_appending_old_cohort(
    tmp_path: Path,
) -> None:
    first_version = json.dumps({"apiVersion": "2.0.5", "dataTimestamp": "2026-08-04T09:00:05"}).encode()
    metadata = json.dumps([{"piece": field} for field in ext.CLINICALTRIALS_DRUG_FIELDS]).encode()
    first_study = {
        "protocolSection": {
            "identificationModule": {"nctId": "NCT00000001"},
            "armsInterventionsModule": {"interventions": [{"type": "DRUG", "name": "Fixture drug"}]},
        },
        "hasResults": False,
    }
    params = {
        "query.term": ext.CLINICALTRIALS_DRUG_QUERY,
        "format": "json",
        "pageSize": "1",
        "countTotal": "true",
        "sort": ext.CLINICALTRIALS_DRUG_SORT,
        "fields": ",".join(ext.CLINICALTRIALS_DRUG_FIELDS),
    }
    first_page_url = f"{ext.CLINICALTRIALS_STUDIES_URL}?{urlencode(params)}"
    interrupted_session = FakeSession(
        {
            ext.CLINICALTRIALS_VERSION_URL: FakeResponse(first_version),
            ext.CLINICALTRIALS_API_DOCS_URL: FakeResponse(b"official API documentation"),
            ext.CLINICALTRIALS_STUDY_METADATA_URL: FakeResponse(metadata),
            ext.CLINICALTRIALS_OPENAPI_URL: FakeResponse(b"openapi: 3.0.3\npaths:\n  /studies: {}\n"),
            ext.CLINICALTRIALS_TERMS_URL: FakeResponse(b"terms"),
            first_page_url: FakeResponse(
                json.dumps(
                    {
                        "totalCount": 2,
                        "studies": [first_study],
                        "nextPageToken": "next-token",
                    }
                ).encode(),
                url=first_page_url,
            ),
        }
    )
    with pytest.raises(AssertionError, match="Unexpected URL"):
        ext.acquire_clinicaltrials_drug_cohort(
            tmp_path,
            session=interrupted_session,  # type: ignore[arg-type]
            page_size=1,
        )

    changed_version = json.dumps({"apiVersion": "2.0.5", "dataTimestamp": "2026-08-05T09:00:05"}).encode()
    second_study = {
        "protocolSection": {
            "identificationModule": {"nctId": "NCT00000002"},
            "armsInterventionsModule": {"interventions": [{"type": "DRUG", "name": "New-day fixture drug"}]},
        },
        "hasResults": False,
    }
    resume_session = FakeSession(
        {
            ext.CLINICALTRIALS_VERSION_URL: FakeResponse(changed_version),
            ext.CLINICALTRIALS_API_DOCS_URL: FakeResponse(b"new official API documentation"),
            ext.CLINICALTRIALS_STUDY_METADATA_URL: FakeResponse(metadata),
            ext.CLINICALTRIALS_OPENAPI_URL: FakeResponse(b"openapi: 3.0.3\npaths:\n  /studies: {}\n"),
            ext.CLINICALTRIALS_TERMS_URL: FakeResponse(b"new terms"),
            first_page_url: FakeResponse(
                json.dumps({"totalCount": 1, "studies": [second_study]}).encode(),
                url=first_page_url,
            ),
        }
    )
    manifest = ext.acquire_clinicaltrials_drug_cohort(
        tmp_path,
        session=resume_session,  # type: ignore[arg-type]
        page_size=1,
    )
    old_key = ext._clinicaltrials_snapshot_key(
        {"apiVersion": "2.0.5", "dataTimestamp": "2026-08-04T09:00:05"}
    )
    new_key = ext._clinicaltrials_snapshot_key(
        {"apiVersion": "2.0.5", "dataTimestamp": "2026-08-05T09:00:05"}
    )
    root = tmp_path / "clinicaltrials_gov_v2/drug_intervention_cohorts"
    assert (root / old_key / "pages/page_000000.json").is_file()
    assert (root / new_key / "pages/page_000000.json").is_file()
    assert manifest["cohort_snapshot_key"] == new_key
    assert manifest["unique_nct_count"] == 1
    assert not any("pageToken=next-token" in call["url"] for call in resume_session.calls)


def test_clinicaltrials_cardiac_safety_cohort_is_separate_and_unlabeled(
    tmp_path: Path,
) -> None:
    version = json.dumps({"apiVersion": "2.0.5", "dataTimestamp": "2026-08-04T09:00:05"}).encode()
    metadata = json.dumps([{"piece": field} for field in ext.CLINICALTRIALS_CARDIAC_SAFETY_FIELDS]).encode()
    broad_study = {
        "protocolSection": {
            "identificationModule": {"nctId": "NCT00000001"},
            "armsInterventionsModule": {"interventions": [{"type": "DRUG", "name": "Broad fixture drug"}]},
        },
        "hasResults": False,
    }
    cardiac_study = {
        "protocolSection": {
            "identificationModule": {
                "nctId": "NCT00000002",
                "briefTitle": "hERG and QTc cardiac repolarization fixture",
            },
            "armsInterventionsModule": {"interventions": [{"type": "DRUG", "name": "Cardiac fixture drug"}]},
        },
        "resultsSection": {
            "outcomeMeasuresModule": {"outcomeMeasures": [{"title": "QTc interval", "type": "PRIMARY"}]},
            "adverseEventsModule": {"seriousEvents": [{"term": "Torsades de pointes"}]},
        },
        "hasResults": True,
    }
    broad_params = {
        "query.term": ext.CLINICALTRIALS_DRUG_QUERY,
        "format": "json",
        "pageSize": "1000",
        "countTotal": "true",
        "sort": ext.CLINICALTRIALS_DRUG_SORT,
        "fields": ",".join(ext.CLINICALTRIALS_DRUG_FIELDS),
    }
    cardiac_params = {
        "query.term": ext.CLINICALTRIALS_CARDIAC_SAFETY_QUERY,
        "format": "json",
        "pageSize": "1000",
        "countTotal": "true",
        "sort": ext.CLINICALTRIALS_DRUG_SORT,
        "fields": ",".join(ext.CLINICALTRIALS_CARDIAC_SAFETY_FIELDS),
    }
    broad_url = f"{ext.CLINICALTRIALS_STUDIES_URL}?{urlencode(broad_params)}"
    cardiac_url = f"{ext.CLINICALTRIALS_STUDIES_URL}?{urlencode(cardiac_params)}"
    session = FakeSession(
        {
            ext.CLINICALTRIALS_VERSION_URL: FakeResponse(version),
            ext.CLINICALTRIALS_API_DOCS_URL: FakeResponse(b"official API documentation"),
            ext.CLINICALTRIALS_STUDY_METADATA_URL: FakeResponse(metadata),
            ext.CLINICALTRIALS_OPENAPI_URL: FakeResponse(b"openapi: 3.0.3\npaths:\n  /studies: {}\n"),
            ext.CLINICALTRIALS_TERMS_URL: FakeResponse(b"terms"),
            broad_url: FakeResponse(
                json.dumps({"totalCount": 1, "studies": [broad_study]}).encode(),
                url=broad_url,
            ),
            cardiac_url: FakeResponse(
                json.dumps({"totalCount": 1, "studies": [cardiac_study]}).encode(),
                url=cardiac_url,
            ),
        }
    )
    ext.acquire_clinicaltrials_drug_cohort(
        tmp_path,
        session=session,  # type: ignore[arg-type]
    )
    manifest = ext.acquire_clinicaltrials_cardiac_safety_cohort(
        tmp_path,
        session=session,  # type: ignore[arg-type]
    )
    cardiac = manifest["cardiac_safety_heuristic_cohort"]
    assert manifest["snapshot_status"] == ("complete_broad_drug_and_heuristic_cardiac_safety_cohorts")
    assert manifest["alias_independent_all_drug_cohort"]["unique_nct_count"] == 1
    assert cardiac["unique_nct_count"] == 1
    assert cardiac["studies_with_posted_outcome_measures_module"] == 1
    assert cardiac["studies_with_posted_adverse_events_module"] == 1
    assert cardiac["model_labels_admitted"] == 0
    assert manifest["outcome_and_adverse_event_result_fields_admitted_as_labels"] is False
    assert (
        ext.verify_source_acquisition_manifest(tmp_path / "clinicaltrials_gov_v2", manifest)[
            "verification_status"
        ]
        == "passed"
    )


def test_clinicaltrials_injected_cardiac_failure_preserves_broad_top_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    version = json.dumps({"apiVersion": "2.0.5", "dataTimestamp": "2026-08-04T09:00:05"}).encode()
    metadata = json.dumps([{"piece": field} for field in ext.CLINICALTRIALS_DRUG_FIELDS]).encode()
    study = {
        "protocolSection": {
            "identificationModule": {"nctId": "NCT00000001"},
            "armsInterventionsModule": {"interventions": [{"type": "DRUG", "name": "Fixture drug"}]},
        },
        "hasResults": False,
    }
    params = {
        "query.term": ext.CLINICALTRIALS_DRUG_QUERY,
        "format": "json",
        "pageSize": "1000",
        "countTotal": "true",
        "sort": ext.CLINICALTRIALS_DRUG_SORT,
        "fields": ",".join(ext.CLINICALTRIALS_DRUG_FIELDS),
    }
    page_url = f"{ext.CLINICALTRIALS_STUDIES_URL}?{urlencode(params)}"
    session = FakeSession(
        {
            ext.CLINICALTRIALS_VERSION_URL: FakeResponse(version),
            ext.CLINICALTRIALS_API_DOCS_URL: FakeResponse(b"official API documentation"),
            ext.CLINICALTRIALS_STUDY_METADATA_URL: FakeResponse(metadata),
            ext.CLINICALTRIALS_OPENAPI_URL: FakeResponse(b"openapi: 3.0.3\npaths:\n  /studies: {}\n"),
            ext.CLINICALTRIALS_TERMS_URL: FakeResponse(b"terms"),
            page_url: FakeResponse(
                json.dumps({"totalCount": 1, "studies": [study]}).encode(),
                url=page_url,
            ),
        }
    )

    def injected_failure(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("injected cardiac failure")

    monkeypatch.setattr(
        ext,
        "acquire_clinicaltrials_cardiac_safety_cohort",
        injected_failure,
    )
    with pytest.raises(RuntimeError, match="injected cardiac failure"):
        ext.acquire_clinicaltrials_complete(
            tmp_path,
            session=session,  # type: ignore[arg-type]
        )
    top_path = tmp_path / "clinicaltrials_gov_v2/clinicaltrials_gov_v2_manifest.json"
    top = json.loads(top_path.read_text())
    assert top["snapshot_status"] == "complete_unmapped_drug_intervention_cohort"
    assert top["targeted_study_query_status"] == "complete_unmapped_drug_intervention_cohort"
    assert (
        ext.verify_source_acquisition_manifest(tmp_path / "clinicaltrials_gov_v2", top)["verification_status"]
        == "passed"
    )


def test_drugsfda_requires_exactly_twelve_parseable_tables(tmp_path: Path) -> None:
    archive = tmp_path / "fda.zip"
    make_zip(archive, {f"table_{index}.txt": b"id\tvalue\n1\tok\n" for index in range(12)})
    report = ext.inspect_drugsfda_archive(archive)
    assert report["txt_table_count"] == 12
    assert report["total_data_rows"] == 12
    bad = tmp_path / "bad.zip"
    make_zip(bad, {"only.txt": b"id\n1\n"})
    with pytest.raises(ValueError, match="expected 12"):
        ext.inspect_drugsfda_archive(bad)


def test_dailymed_custom_fixture_checks_md5_bytes_and_member_count(tmp_path: Path) -> None:
    archive_buffer = io.BytesIO()
    with zipfile.ZipFile(archive_buffer, "w", compression=zipfile.ZIP_DEFLATED) as zipped:
        zipped.writestr("setid/version.xml", b"<document/>")
    payload = archive_buffer.getvalue()
    filename = "fixture-human-rx.zip"
    url = f"{ext.DAILYMED_DOWNLOAD_ROOT}/{filename}"
    session = FakeSession(
        {
            ext.DAILYMED_LANDING_URL: FakeResponse(b"landing"),
            url: FakeResponse(payload),
        }
    )
    manifest = ext.acquire_dailymed(
        tmp_path,
        session=session,  # type: ignore[arg-type]
        parts=[
            (
                filename,
                hashlib.md5(payload, usedforsecurity=False).hexdigest(),
                1,
                len(payload),
            )
        ],
    )
    assert manifest["expected_and_verified_file_member_count"] == 1
    assert manifest["snapshot_status"] == "complete_raw_archives_crc_and_membership_verified"
    assert manifest["canonical_rows_admitted"] == 0


def test_acquisition_summary_binds_manifests_and_prohibits_training(tmp_path: Path) -> None:
    first = ext.document_with_sha256(
        {
            "source_id": "a",
            "release_id": "1",
            "snapshot_status": "complete",
            "exact_physical_file_count": 1,
            "exact_physical_bytes": 2,
            "canonical_rows_admitted": 0,
            "model_labels_admitted": 0,
        }
    )
    summary = ext.write_acquisition_summary(tmp_path, [first])
    assert summary["substantive_training_started"] is False
    assert summary["sources"][0]["manifest_sha256"] == first["manifest_sha256"]
    first["release_id"] = "changed"
    with pytest.raises(ValueError, match="identity"):
        ext.write_acquisition_summary(tmp_path, [first])


def test_source_manifest_verifier_reconciles_sidecar_and_detects_byte_change(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    payload = b"raw"
    url = "https://example.test/raw"
    acquired = ext.download_immutable(
        ext.DownloadSpec(canonical_url=url, destination=source_root / "raw.bin"),
        session=FakeSession({url: FakeResponse(payload)}),  # type: ignore[arg-type]
    )
    annotated = ext.annotate_acquisition_record(acquired, artifact_role="raw_payload")
    manifest = ext.document_with_sha256(
        {
            "source_id": "fixture",
            "release_id": "1",
            "files": [annotated],
            "exact_physical_file_count": 1,
            "exact_physical_bytes": len(payload),
            "canonical_rows_admitted": 0,
            "model_labels_admitted": 0,
        }
    )
    report = ext.verify_source_acquisition_manifest(source_root, manifest)
    assert report["verification_status"] == "passed"
    (source_root / "raw.bin").write_bytes(b"changed")
    with pytest.raises(ValueError, match="byte count"):
        ext.verify_source_acquisition_manifest(source_root, manifest)


def test_final_source_bundle_verifier_binds_sidecars_and_derived_inventories(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    url = "https://example.test/raw"
    acquired = ext.download_immutable(
        ext.DownloadSpec(canonical_url=url, destination=source_root / "raw.bin"),
        session=FakeSession({url: FakeResponse(b"raw")}),  # type: ignore[arg-type]
    )
    annotated = ext.annotate_acquisition_record(acquired, artifact_role="raw_payload")
    assert "manifest_sha256" not in annotated
    assert "acquisition_sidecar_manifest_sha256" in annotated
    inventory_path = source_root / "derived.jsonl"
    ext._write_jsonl_records(inventory_path, [{"row": 1}, {"row": 2}])
    manifest_body = ext._source_manifest(
        "bindingdb_curated_202608",
        release_id="fixture",
        release_date=None,
        files=[annotated],
        status="fixture_complete",
    )
    manifest_path = source_root / "source_manifest.json"
    manifest = ext.atomic_write_source_manifest(source_root, manifest_path, manifest_body)
    report = ext.verify_source_acquisition_manifest(source_root, manifest)
    assert report["verified_source_artifact_count"] == 1
    assert report["verified_bundle_artifact_count"] == 3

    inventory_path.write_bytes(inventory_path.read_bytes().replace(b'"row":1', b'"row":9'))
    with pytest.raises(ValueError, match="bundle artifact SHA-256"):
        ext.verify_source_acquisition_manifest(source_root, manifest)


def test_final_source_bundle_verifier_rejects_undeclared_extra_and_stale_annotation(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    url = "https://example.test/raw"
    acquired = ext.download_immutable(
        ext.DownloadSpec(canonical_url=url, destination=source_root / "raw.bin"),
        session=FakeSession({url: FakeResponse(b"raw")}),  # type: ignore[arg-type]
    )
    annotated = ext.annotate_acquisition_record(acquired, artifact_role="raw_payload")
    manifest_body = ext._source_manifest(
        "bindingdb_curated_202608",
        release_id="fixture",
        release_date=None,
        files=[annotated],
        status="fixture_complete",
    )
    manifest = ext.atomic_write_source_manifest(
        source_root,
        source_root / "source_manifest.json",
        manifest_body,
    )
    (source_root / "undeclared.txt").write_text("extra")
    with pytest.raises(ValueError, match="membership changed"):
        ext.verify_source_acquisition_manifest(source_root, manifest)

    (source_root / "undeclared.txt").unlink()
    stale = dict(acquired)
    stale["artifact_role"] = "mutated_after_self_digest"
    stale_manifest = ext.document_with_sha256(
        {
            "source_id": "fixture",
            "release_id": "1",
            "files": [stale],
            "exact_physical_file_count": 1,
            "exact_physical_bytes": 3,
            "canonical_rows_admitted": 0,
            "model_labels_admitted": 0,
        }
    )
    with pytest.raises(ValueError, match="Embedded acquisition record identity failed"):
        ext.verify_source_acquisition_manifest(source_root, stale_manifest)


def test_source_verifier_rejects_duplicate_file_records(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    url = "https://example.test/raw"
    acquired = ext.download_immutable(
        ext.DownloadSpec(canonical_url=url, destination=source_root / "raw.bin"),
        session=FakeSession({url: FakeResponse(b"raw")}),  # type: ignore[arg-type]
    )
    annotated = ext.annotate_acquisition_record(acquired, artifact_role="raw_payload")
    manifest = ext.document_with_sha256(
        {
            "source_id": "fixture",
            "release_id": "1",
            "files": [annotated, dict(annotated)],
            "exact_physical_file_count": 2,
            "exact_physical_bytes": 6,
            "canonical_rows_admitted": 0,
            "model_labels_admitted": 0,
        }
    )
    with pytest.raises(ValueError, match="Duplicate source artifact path"):
        ext.verify_source_acquisition_manifest(source_root, manifest)


def test_source_bundle_verifier_rejects_symlink_directory(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    url = "https://example.test/raw"
    acquired = ext.download_immutable(
        ext.DownloadSpec(canonical_url=url, destination=source_root / "raw.bin"),
        session=FakeSession({url: FakeResponse(b"raw")}),  # type: ignore[arg-type]
    )
    annotated = ext.annotate_acquisition_record(acquired, artifact_role="raw_payload")
    manifest_body = ext._source_manifest(
        "bindingdb_curated_202608",
        release_id="fixture",
        release_date=None,
        files=[annotated],
        status="fixture_complete",
    )
    manifest = ext.atomic_write_source_manifest(
        source_root,
        source_root / "source_manifest.json",
        manifest_body,
    )
    target = tmp_path / "target"
    target.mkdir()
    (source_root / "linked-directory").symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="Symlink is prohibited"):
        ext.verify_source_acquisition_manifest(source_root, manifest)


def test_all_source_boundaries_prohibit_acquisition_labels() -> None:
    assert set(ext.SOURCE_BOUNDARIES) == {
        "bindingdb_curated_202608",
        "uniprotkb_targeted_2026_02",
        "clinicaltrials_gov_v2",
        "drugs_at_fda_bulk",
        "dailymed_spl_v2_human_rx",
    }
    for boundary in ext.SOURCE_BOUNDARIES.values():
        text = str(boundary["default_label_admission"])
        assert any(token in text for token in ("prohibited", "never"))


def test_cli_requires_uniprot_inputs(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="requires"):
        ext.main(["uniprot", "--raw-root", str(tmp_path)])
