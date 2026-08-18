from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from menin_discovery import platform_pkdb_candidates as pkdb


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _raw_fixture(tmp_path: Path, *, output_count: int = 12, probe_count: int = 0) -> Path:
    raw = tmp_path / "raw"
    _write(
        raw / "statistics.json",
        {
            "version": "fixture",
            "study_count": 2,
            "intervention_count": 4,
            "output_count": output_count,
        },
    )
    _write(
        raw / "openapi.json",
        {
            "info": {
                "version": "v1",
                "termsOfService": "https://example.invalid/terms",
                "license": {"name": "fixture software licence"},
            },
            "security": [{"Basic": []}],
        },
    )
    _write(
        raw / "outputs_anonymous_probe.json",
        {
            "current_page": 1,
            "last_page": 1,
            "data": {"count": probe_count, "data": []},
        },
    )
    for sid in pkdb.TARGET_INFO_NODES:
        _write(
            raw / "info_nodes" / f"{sid}.json",
            {
                "sid": sid,
                "name": f"fixture {sid}",
                "label": sid.upper(),
                "type": "measurement",
                "dtype": "float",
                "deprecated": False,
                "allowed_units": ["fixture-unit"],
            },
        )
    _write(
        raw / "studies_access_probe_sanitized.json",
        pkdb.document_with_sha256(
            {
                "reported_count": 2,
                "records": [{"sid": "S1", "access": "public", "licence": "closed"}],
            }
        ),
    )
    _write(
        raw / "rights_snapshot.json",
        pkdb.document_with_sha256(
            {"data_record_rights_status": "unresolved", "canonical_admission_allowed": False}
        ),
    )
    _write(
        raw / "interventions_context_probe_sanitized.json",
        pkdb.document_with_sha256({"reported_count": 8, "records": [{"pk": 1, "dose_value_present": True}]}),
    )
    artifacts = [pkdb._artifact(path, raw, "fixture") for path in sorted(raw.rglob("*")) if path.is_file()]
    _write(
        raw / "manifest.json",
        pkdb.document_with_sha256(
            {
                "schema_version": pkdb.SCHEMA_VERSION,
                "source_id": pkdb.SOURCE_ID,
                "artifact_count": len(artifacts),
                "artifacts": artifacts,
                "canonical_observation_count": 0,
                "training_label_count": 0,
            }
        ),
    )
    return raw


def test_sanitizers_remove_publication_text_and_numeric_dose_values() -> None:
    study = pkdb.sanitize_study_probe(
        {
            "data": {
                "count": 1,
                "data": [
                    {
                        "sid": "S1",
                        "access": "public",
                        "licence": "closed",
                        "title": "must disappear",
                        "abstract": "must disappear",
                        "output_count": 3,
                        "substances": [{"sid": "warfarin", "name": "not retained"}],
                    }
                ],
            }
        }
    )
    encoded_study = json.dumps(study)
    assert "must disappear" not in encoded_study
    assert study["records"][0]["licence"] == "closed"
    assert study["records"][0]["substance_sids"] == ["warfarin"]

    intervention = pkdb.sanitize_intervention_probe(
        {
            "data": {
                "count": 1,
                "data": [
                    {
                        "pk": 7,
                        "value": 123.45,
                        "unit": "mg",
                        "description": "must disappear",
                        "route": "oral",
                        "study": {"sid": "PKDB00001", "name": "must disappear"},
                        "substance": {"sid": "caf", "label": "must disappear"},
                    }
                ],
            }
        }
    )
    encoded_intervention = json.dumps(intervention)
    assert "123.45" not in encoded_intervention
    assert "must disappear" not in encoded_intervention
    assert intervention["records"][0]["dose_value_present"] is True
    assert intervention["records"][0]["dose_unit"] == "mg"
    assert intervention["records"][0]["study_sid"] == "PKDB00001"
    assert intervention["records"][0]["substance_sid"] == "caf"


def test_normalize_fails_closed_and_keeps_endpoint_semantics_separate(tmp_path: Path) -> None:
    raw = _raw_fixture(tmp_path)
    interim = tmp_path / "interim"
    report = tmp_path / "report"
    result = pkdb.normalize(raw, interim, report)

    assert result["status"] == "fail_closed_zero_admission"
    assert result["canonical_observation_rows"] == 0
    decision = json.loads((interim / "admission_decision.json").read_text())
    assert decision["pkdb_reported_output_count"] == 12
    assert decision["anonymous_outputs_probe_count"] == 0
    assert decision["anonymous_output_access_reproducible"] is False
    assert decision["record_level_reuse_rights_resolved"] is False
    assert decision["canonical_admission"] is False
    assert decision["training_label_rows"] == 0

    inventory = json.loads((interim / "endpoint_inventory.json").read_text())
    sids = [item["sid"] for item in inventory["endpoints"]]
    assert sids == list(pkdb.TARGET_INFO_NODES)
    assert "auc-end" in sids and "auc-inf" in sids
    assert "vd" in sids and "vd-ss" in sids
    assert "no AUC, volume, or half-life pooling" in inventory["semantics_policy"]
    context = json.loads((interim / "context_availability.json").read_text())
    assert context["public_count_consistency_audit"]["study_counts_match"] is True
    assert context["public_count_consistency_audit"]["intervention_counts_match"] is False


def test_verify_performs_byte_identical_normalized_replay(tmp_path: Path) -> None:
    raw = _raw_fixture(tmp_path)
    interim = tmp_path / "interim"
    report = tmp_path / "report"
    pkdb.normalize(raw, interim, report)
    verified = pkdb.verify(raw, interim, report)
    assert verified["status"] == "verified_fail_closed"
    assert verified["byte_identical_replay"] is True
    assert all(item["byte_identical"] for item in verified["comparisons"])
    assert verified["training_label_rows"] == 0


def test_raw_tamper_is_detected_before_normalization(tmp_path: Path) -> None:
    raw = _raw_fixture(tmp_path)
    (raw / "statistics.json").write_text('{"output_count": 0}', encoding="utf-8")
    with pytest.raises(pkdb.PKDBCandidateError, match="binding failed"):
        pkdb.normalize(raw, tmp_path / "interim", tmp_path / "report")


def test_output_tamper_is_detected_by_verifier(tmp_path: Path) -> None:
    raw = _raw_fixture(tmp_path)
    interim = tmp_path / "interim"
    report = tmp_path / "report"
    pkdb.normalize(raw, interim, report)
    (interim / "admission_decision.json").write_text("{}", encoding="utf-8")
    with pytest.raises(pkdb.PKDBCandidateError, match="binding failed"):
        pkdb.verify(raw, interim, report)


def test_unsafe_manifest_path_is_rejected(tmp_path: Path) -> None:
    raw = _raw_fixture(tmp_path)
    manifest_path = raw / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifacts"][0]["path"] = "../escape.json"
    manifest = pkdb.document_with_sha256(manifest)
    _write(manifest_path, manifest)
    with pytest.raises(pkdb.PKDBCandidateError, match="unsafe relative path"):
        pkdb.normalize(raw, tmp_path / "interim", tmp_path / "report")


class _FakeResponse:
    def __init__(
        self,
        body: bytes,
        *,
        status: int = 200,
        url: str = "https://pk-db.com/api/v1/statistics/",
        headers: dict[str, str] | None = None,
        redirect: bool = False,
    ) -> None:
        self.body = body
        self.status_code = status
        self.url = url
        self.headers = headers or {}
        self.is_redirect = redirect

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError("fixture HTTP error")

    def iter_content(self, chunk_size: int) -> list[bytes]:
        del chunk_size
        return [self.body]


class _FakeSession:
    def __init__(self, response: _FakeResponse) -> None:
        self.response = response

    def get(self, *args: object, **kwargs: object) -> _FakeResponse:
        del args, kwargs
        return self.response


def test_bounded_get_rejects_non_official_origin_redirect_and_oversize() -> None:
    with pytest.raises(pkdb.PKDBCandidateError, match="official PK-DB"):
        pkdb._get_bounded(
            _FakeSession(_FakeResponse(b"{}")),  # type: ignore[arg-type]
            "https://example.com/api",
            max_bytes=10,
        )
    with pytest.raises(pkdb.PKDBCandidateError, match="redirect not allowed"):
        pkdb._get_bounded(
            _FakeSession(_FakeResponse(b"{}", redirect=True)),  # type: ignore[arg-type]
            "https://pk-db.com/api/v1/statistics/",
            max_bytes=10,
        )
    with pytest.raises(pkdb.PKDBCandidateError, match="exceeded byte ceiling"):
        pkdb._get_bounded(
            _FakeSession(_FakeResponse(b"01234567890")),  # type: ignore[arg-type]
            "https://pk-db.com/api/v1/statistics/",
            max_bytes=10,
        )


def test_pagination_envelope_rejects_ambiguous_or_negative_counts() -> None:
    with pytest.raises(pkdb.PKDBCandidateError, match="count"):
        pkdb._page_records({"data": {"count": -1, "data": []}})
    with pytest.raises(pkdb.PKDBCandidateError, match="records"):
        pkdb._page_records({"data": {"count": 1, "data": ["not-an-object"]}})


def test_rights_blocker_is_independent_of_anonymous_record_visibility(tmp_path: Path) -> None:
    raw = _raw_fixture(tmp_path, output_count=0, probe_count=0)
    interim = tmp_path / "interim"
    report = tmp_path / "report"
    pkdb.normalize(raw, interim, report)
    decision = json.loads((interim / "admission_decision.json").read_text())
    assert decision["anonymous_output_access_reproducible"] is True
    assert decision["record_level_reuse_rights_resolved"] is False
    assert decision["canonical_admission"] is False
    assert decision["training_label_rows"] == 0


def test_unexpected_output_and_symlink_fail_closed(tmp_path: Path) -> None:
    raw = _raw_fixture(tmp_path)
    interim = tmp_path / "interim"
    report = tmp_path / "report"
    pkdb.normalize(raw, interim, report)
    (interim / "unexpected.json").write_text("{}", encoding="utf-8")
    with pytest.raises(pkdb.PKDBCandidateError, match="membership drift"):
        pkdb.verify(raw, interim, report)
    (interim / "unexpected.json").unlink()
    (report / "unexpected-link").symlink_to(report / "summary.md")
    with pytest.raises(pkdb.PKDBCandidateError, match="non-regular"):
        pkdb.verify(raw, interim, report)
