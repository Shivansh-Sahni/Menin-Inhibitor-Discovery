from __future__ import annotations

import json
from pathlib import Path

import pytest
from menin_discovery.platform_data_sources import sha256_file
from menin_discovery.platform_determinism import (
    compare_canonical_builds,
    write_determinism_report,
)


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _fixture(parent: Path, timestamp: str) -> tuple[Path, Path]:
    build = parent / "full_chembl37"
    reports = parent / "reports"
    build.mkdir(parents=True)
    reports.mkdir(parents=True)
    component = build / "observations.parquet"
    component.write_bytes(b"deterministic-canonical-bytes")
    manifest = {
        "schema_version": "test",
        "built_at_utc": timestamp,
        "component_inventory": [
            {
                "path": component.name,
                "sha256": sha256_file(component),
                "size_bytes": component.stat().st_size,
            }
        ],
    }
    _json(build / "build_manifest.json", manifest)
    table = reports / "qc.csv"
    table.write_text("field,count\nx,1\n", encoding="utf-8")
    qc = {
        "generated_at_utc": timestamp,
        "build_manifest_sha256": sha256_file(build / "build_manifest.json"),
        "qc_passed": True,
        "artifacts": {
            "table": {
                "path": table.name,
                "sha256": sha256_file(table),
                "size_bytes": table.stat().st_size,
            }
        },
        "figures": [],
    }
    _json(reports / "qc_report.json", qc)
    _json(reports / "eda_summary.json", {"rows": 1})
    _json(reports / "data_bulk_canonical_manifest.json", manifest)
    return build, reports


def test_two_build_comparison_ignores_only_declared_fields(tmp_path: Path) -> None:
    build_a, reports_a = _fixture(tmp_path / "a", "2026-08-04T01:00:00Z")
    build_b, reports_b = _fixture(tmp_path / "b", "2026-08-04T02:00:00Z")

    result = compare_canonical_builds(build_a, reports_a, build_b, reports_b)
    assert result["content_equivalent"] is True
    assert result["canonical_component_count"] == 1
    assert result["substantive_training_started"] is False
    output = tmp_path / "verification.json"
    write_determinism_report(output, result)
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == ("passed_content_equivalent")


def test_component_tamper_and_unbound_membership_fail_closed(tmp_path: Path) -> None:
    build_a, reports_a = _fixture(tmp_path / "a", "2026-08-04T01:00:00Z")
    build_b, reports_b = _fixture(tmp_path / "b", "2026-08-04T02:00:00Z")
    (build_b / "observations.parquet").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="size mismatch"):
        compare_canonical_builds(build_a, reports_a, build_b, reports_b)

    build_b, reports_b = _fixture(tmp_path / "c", "2026-08-04T03:00:00Z")
    (build_b / "unbound.txt").write_text("unbound", encoding="utf-8")
    with pytest.raises(ValueError, match="membership"):
        compare_canonical_builds(build_a, reports_a, build_b, reports_b)


def test_matched_stale_report_manifests_and_report_symlinks_are_rejected(
    tmp_path: Path,
) -> None:
    build_a, reports_a = _fixture(tmp_path / "a", "2026-08-04T01:00:00Z")
    build_b, reports_b = _fixture(tmp_path / "b", "2026-08-04T02:00:00Z")
    stale_a = {"built_at_utc": "2026-08-04T01:00:00Z", "stale": True}
    stale_b = {"built_at_utc": "2026-08-04T02:00:00Z", "stale": True}
    _json(reports_a / "data_bulk_canonical_manifest.json", stale_a)
    _json(reports_b / "data_bulk_canonical_manifest.json", stale_b)
    with pytest.raises(ValueError, match="physical build manifest"):
        compare_canonical_builds(build_a, reports_a, build_b, reports_b)

    build_a, reports_a = _fixture(tmp_path / "c", "2026-08-04T03:00:00Z")
    build_b, reports_b = _fixture(tmp_path / "d", "2026-08-04T04:00:00Z")
    (reports_b / "eda_summary.json").unlink()
    (reports_b / "eda_summary.json").symlink_to(reports_a / "eda_summary.json")
    with pytest.raises(ValueError, match="symlink"):
        compare_canonical_builds(build_a, reports_a, build_b, reports_b)


def test_undeclared_manifest_qc_and_report_mutations_are_rejected(tmp_path: Path) -> None:
    build_a, reports_a = _fixture(tmp_path / "a", "2026-08-04T01:00:00Z")
    build_b, reports_b = _fixture(tmp_path / "b", "2026-08-04T02:00:00Z")
    manifest_b = json.loads((build_b / "build_manifest.json").read_text(encoding="utf-8"))
    manifest_b["undeclared_difference"] = True
    _json(build_b / "build_manifest.json", manifest_b)
    qc_b = json.loads((reports_b / "qc_report.json").read_text(encoding="utf-8"))
    qc_b["build_manifest_sha256"] = sha256_file(build_b / "build_manifest.json")
    _json(reports_b / "qc_report.json", qc_b)
    report_b = json.loads((reports_b / "data_bulk_canonical_manifest.json").read_text(encoding="utf-8"))
    report_b["undeclared_difference"] = True
    _json(reports_b / "data_bulk_canonical_manifest.json", report_b)
    with pytest.raises(ValueError, match="beyond built_at_utc"):
        compare_canonical_builds(build_a, reports_a, build_b, reports_b)

    build_b, reports_b = _fixture(tmp_path / "c", "2026-08-04T03:00:00Z")
    qc_b = json.loads((reports_b / "qc_report.json").read_text(encoding="utf-8"))
    qc_b["undeclared_qc_difference"] = True
    _json(reports_b / "qc_report.json", qc_b)
    with pytest.raises(ValueError, match="declared timestamp"):
        compare_canonical_builds(build_a, reports_a, build_b, reports_b)

    build_b, reports_b = _fixture(tmp_path / "d", "2026-08-04T04:00:00Z")
    (reports_b / "extra.txt").write_text("unbound", encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected membership"):
        compare_canonical_builds(build_a, reports_a, build_b, reports_b)
