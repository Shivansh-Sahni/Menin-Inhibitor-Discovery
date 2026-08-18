import copy
import hashlib
import json
import sys
import types

import pytest
from menin_discovery.cli import (
    ANALYSIS_MANIFEST_EXCLUDES,
    MODEL_MANIFEST_EXCLUDES,
    REPORT_MANIFEST_EXCLUDES,
    _analysis_manifest_candidate,
    _expected_analysis_metadata,
    _expected_report_metadata,
    _model_manifest_candidate,
    _report_manifest_candidate,
    _validate_analysis_lineage,
    run_analysis_stage,
    run_manifest_stage,
    run_report_stage,
    run_verify_stage,
)
from menin_discovery.provenance import (
    create_data_manifests,
    create_manifest,
    load_manifest,
    sha256_file,
    verify_data_manifests,
    verify_manifest,
    write_manifest,
)
from menin_discovery.settings import settings_snapshot

CREATED_AT = "2026-01-02T03:04:05Z"


def _analysis_release_fixture(tmp_path):
    for directory in (
        "research/data/raw",
        "research/data/processed",
        "research/models",
        "research/analysis",
        "research/reports",
        "pipeline/config",
    ):
        (tmp_path / directory).mkdir(parents=True)
    (tmp_path / "research/data/raw/source.csv").write_text("id,value\n1,10\n", encoding="utf-8")
    (tmp_path / "research/data/processed/curated.csv").write_text("id,value\n1,10\n", encoding="utf-8")
    for filename in (
        "menin_activity_measurements.csv",
        "herg_compounds_curated.csv",
        "pk_admet_observations.csv",
    ):
        (tmp_path / "research/data/processed" / filename).write_text(
            "structure_id,value\nA,1\n", encoding="utf-8"
        )
    (tmp_path / "research/reports/menin_with_predicted_herg_risk.csv").write_text(
        "structure_id,predicted_herg_blocker_probability\nA,0.1\n",
        encoding="utf-8",
    )
    (tmp_path / "pipeline/config/pipeline.yaml").write_text("project: {}\n", encoding="utf-8")
    settings = {
        "project": {"root": str(tmp_path)},
        "paths": {
            "raw": "research/data/raw",
            "processed": "research/data/processed",
            "models": "research/models",
            "analysis": "research/analysis",
            "reports": "research/reports",
        },
        "analysis": {"enabled": True},
    }

    manifests = run_manifest_stage(settings, include_analysis_artifacts=False)
    processed = manifests["processed"]
    software = manifests["software"]
    processed_manifest_path = tmp_path / "research/reports/manifests/processed_manifest.json"
    software_manifest_path = tmp_path / "research/reports/manifests/software_manifest.json"
    artifact_path = tmp_path / "research/models/model.skops"
    artifact_path.write_bytes(b"model")
    model_document = {
        "artifact": {
            "filename": artifact_path.name,
            "path": "research/models/model.skops",
            "path_is_repository_relative": True,
            "sha256": sha256_file(artifact_path),
        },
        "provenance": {
            "processed_build_id": processed["build_id"],
            "processed_dataset_sha256": processed["dataset_sha256"],
            "processed_manifest_sha256": sha256_file(processed_manifest_path),
            "software_dataset_sha256": software["dataset_sha256"],
            "software_manifest_sha256": sha256_file(software_manifest_path),
        },
    }
    (tmp_path / "research/models/model_manifest.json").write_text(
        json.dumps(model_document), encoding="utf-8"
    )
    models = _model_manifest_candidate(
        {
            "models": tmp_path / "research/models",
        },
        processed,
        software,
    )
    return settings, processed, software, models


def _write_analysis_summary(analysis_dir, processed_dir, reports_dir):
    input_paths = {
        "scored_menin_herg": reports_dir / "menin_with_predicted_herg_risk.csv",
        "menin_measurements": processed_dir / "menin_activity_measurements.csv",
        "observed_herg": processed_dir / "herg_compounds_curated.csv",
        "pk_admet": processed_dir / "pk_admet_observations.csv",
    }
    (analysis_dir / "analysis_summary.json").write_text(
        json.dumps({"input_sha256": {name: sha256_file(path) for name, path in sorted(input_paths.items())}}),
        encoding="utf-8",
    )


def test_manifest_is_deterministic_and_records_table_metadata(tmp_path):
    (tmp_path / "z.txt").write_text("publication notes\n", encoding="utf-8")
    (tmp_path / "a.csv").write_text("id,value\n1,10\n2,20\n", encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "records.json").write_text(
        '[{"compound":"A","active":true},{"compound":"B","active":false}]\n',
        encoding="utf-8",
    )

    first = create_manifest(tmp_path, stage="raw", created_at=CREATED_AT)
    second = create_manifest(tmp_path, stage="raw", created_at=CREATED_AT)

    assert first == second
    assert first["created_at"] == CREATED_AT
    assert first["build_id"].startswith("build-")
    assert len(first["dataset_sha256"]) == 64
    assert len(first["manifest_sha256"]) == 64
    assert [entry["path"] for entry in first["files"]] == [
        "a.csv",
        "nested/records.json",
        "z.txt",
    ]
    csv_entry = first["files"][0]
    assert csv_entry["row_count"] == 2
    assert csv_entry["column_count"] == 2
    assert [column["name"] for column in csv_entry["schema"]] == ["id", "value"]
    json_entry = first["files"][1]
    assert json_entry["row_count"] == 2


def test_manifest_round_trip_and_verification(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    (root / "table.csv").write_text("id,value\n1,10\n2,20\n", encoding="utf-8")
    manifest = create_manifest(root, stage="processed", created_at=CREATED_AT)
    manifest_path = write_manifest(manifest, tmp_path / "processed_manifest.json")

    assert load_manifest(manifest_path) == manifest
    result = verify_manifest(manifest_path, root=root, verified_at=CREATED_AT)
    assert result.valid
    assert result.checked_files == 1
    assert result.expected_files == 1

    (root / "table.csv").write_text("id,value\n1,999\n", encoding="utf-8")
    changed = verify_manifest(manifest, root=root, verified_at=CREATED_AT)
    codes = {issue.code for issue in changed.issues}
    assert {"sha256_mismatch", "size_mismatch", "row_count_mismatch"}.issubset(codes)
    assert not changed.valid


def test_manifest_detects_document_tampering_and_extra_files(tmp_path):
    (tmp_path / "one.txt").write_text("one", encoding="utf-8")
    manifest = create_manifest(tmp_path, stage="raw", created_at=CREATED_AT)
    tampered = copy.deepcopy(manifest)
    tampered["stage"] = "processed"

    verification = verify_manifest(tampered, root=tmp_path)
    codes = {issue.code for issue in verification.issues}
    assert "manifest_digest_mismatch" in codes
    assert "dataset_digest_mismatch" in codes

    (tmp_path / "extra.txt").write_text("extra", encoding="utf-8")
    strict = verify_manifest(manifest, root=tmp_path, allow_extra=False)
    assert any(issue.code == "unexpected_file" and issue.path == "extra.txt" for issue in strict.issues)


def test_linked_raw_and_processed_manifests(tmp_path):
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    output = tmp_path / "manifests"
    raw.mkdir()
    processed.mkdir()
    (raw / "source.csv").write_text("id,value\n1,10\n", encoding="utf-8")
    (processed / "curated.csv").write_text("compound_id,value_nm\nA,10\n", encoding="utf-8")

    manifests = create_data_manifests(
        raw,
        processed,
        created_at=CREATED_AT,
        output_directory=output,
    )
    assert manifests["raw"]["build_id"] == manifests["processed"]["build_id"]
    assert manifests["processed"]["upstream"] == [
        {
            "stage": "raw",
            "dataset_sha256": manifests["raw"]["dataset_sha256"],
        }
    ]
    assert (output / "raw_manifest.json").exists()
    assert (output / "processed_manifest.json").exists()

    results = verify_data_manifests(
        manifests,
        raw_root=raw,
        processed_root=processed,
        verified_at=CREATED_AT,
    )
    assert results["raw"].valid
    assert results["processed"].valid


def test_link_verification_rejects_wrong_upstream_digest_and_build_id(tmp_path):
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    raw.mkdir()
    processed.mkdir()
    (raw / "source.csv").write_text("id,value\n1,10\n", encoding="utf-8")
    (processed / "curated.csv").write_text("id,value\n1,10\n", encoding="utf-8")

    raw_manifest = create_manifest(
        raw,
        stage="raw",
        build_id="build-raw",
        created_at=CREATED_AT,
    )
    processed_manifest = create_manifest(
        processed,
        stage="processed",
        build_id="build-processed",
        created_at=CREATED_AT,
        upstream=({"stage": "raw", "dataset_sha256": "0" * 64},),
    )
    results = verify_data_manifests(
        {"raw": raw_manifest, "processed": processed_manifest},
        raw_root=raw,
        processed_root=processed,
        verified_at=CREATED_AT,
    )
    codes = {issue.code for issue in results["processed"].issues}
    assert {"upstream_manifest_mismatch", "build_id_mismatch"}.issubset(codes)
    assert not results["processed"].valid


def test_release_manifests_link_all_stages_and_verify(tmp_path):
    for directory in (
        "research/data/raw",
        "research/data/processed",
        "research/models",
        "research/reports",
        "pipeline/config",
    ):
        (tmp_path / directory).mkdir(parents=True)
    (tmp_path / "research/data/raw/source.csv").write_text("id,value\n1,10\n", encoding="utf-8")
    (tmp_path / "research/data/processed/curated.csv").write_text("id,value\n1,10\n", encoding="utf-8")
    (tmp_path / "research/reports/report.txt").write_text("report\n", encoding="utf-8")
    (tmp_path / "pipeline/config/pipeline.yaml").write_text("project: {}\n", encoding="utf-8")
    settings = {
        "project": {"root": str(tmp_path)},
        "paths": {
            "raw": "research/data/raw",
            "processed": "research/data/processed",
            "models": "research/models",
            "reports": "research/reports",
        },
    }

    data_manifests = run_manifest_stage(settings, include_analysis_artifacts=False)
    processed = data_manifests["processed"]
    software = data_manifests["software"]
    processed_manifest_path = tmp_path / "research/reports/manifests/processed_manifest.json"
    software_manifest_path = tmp_path / "research/reports/manifests/software_manifest.json"
    artifact_path = tmp_path / "research/models/model.skops"
    artifact_path.write_bytes(b"model")
    model_document = {
        "artifact": {
            "filename": artifact_path.name,
            "path": "research/models/model.skops",
            "path_is_repository_relative": True,
            "sha256": sha256_file(artifact_path),
        },
        "provenance": {
            "processed_build_id": processed["build_id"],
            "processed_dataset_sha256": processed["dataset_sha256"],
            "processed_manifest_sha256": sha256_file(processed_manifest_path),
            "software_dataset_sha256": software["dataset_sha256"],
            "software_manifest_sha256": sha256_file(software_manifest_path),
        },
    }
    (tmp_path / "research/models/model_manifest.json").write_text(
        json.dumps(model_document), encoding="utf-8"
    )
    model_manifest = create_manifest(
        tmp_path / "research/models",
        stage="models",
        build_id=processed["build_id"],
        exclude=MODEL_MANIFEST_EXCLUDES,
        upstream=(
            {"stage": "processed", "dataset_sha256": processed["dataset_sha256"]},
            {"stage": "software", "dataset_sha256": software["dataset_sha256"]},
        ),
    )
    report_manifest = create_manifest(
        tmp_path / "research/reports",
        stage="reports",
        build_id=processed["build_id"],
        exclude=REPORT_MANIFEST_EXCLUDES,
        upstream=(
            {"stage": "processed", "dataset_sha256": processed["dataset_sha256"]},
            {"stage": "models", "dataset_sha256": model_manifest["dataset_sha256"]},
            {"stage": "software", "dataset_sha256": software["dataset_sha256"]},
        ),
    )
    (tmp_path / "research/reports/report_build_metadata.json").write_text(
        json.dumps(
            {
                "build_id": processed["build_id"],
                "processed_dataset_sha256": processed["dataset_sha256"],
                "software_dataset_sha256": software["dataset_sha256"],
                "models_dataset_sha256": model_manifest["dataset_sha256"],
                "reports_dataset_sha256": report_manifest["dataset_sha256"],
                "resolved_settings_sha256": hashlib.sha256(
                    settings_snapshot(settings).encode("utf-8")
                ).hexdigest(),
            }
        ),
        encoding="utf-8",
    )

    manifests = run_manifest_stage(settings)
    build_ids = {manifest["build_id"] for manifest in manifests.values()}
    assert len(build_ids) == 1
    assert {item["stage"] for item in manifests["models"]["upstream"]} == {
        "processed",
        "software",
    }
    assert {item["stage"] for item in manifests["reports"]["upstream"]} == {
        "processed",
        "models",
        "software",
    }
    assert run_verify_stage(settings) == {
        "raw": True,
        "processed": True,
        "software": True,
        "models": True,
        "reports": True,
    }
    (tmp_path / "research/reports/report.txt").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="reports"):
        run_verify_stage(settings)
    failed_report = json.loads(
        (tmp_path / "research/reports/verification/reports_verification.json").read_text(encoding="utf-8")
    )
    assert not failed_report["valid"]
    assert "manifest_scope_changed" in {issue["code"] for issue in failed_report["issues"]}


def test_release_manifest_rejects_unprovenanced_model_bytes(tmp_path):
    for directory in (
        "research/data/raw",
        "research/data/processed",
        "research/models",
        "research/reports",
        "pipeline/config",
    ):
        (tmp_path / directory).mkdir(parents=True)
    (tmp_path / "research/data/raw/source.csv").write_text("id\n1\n", encoding="utf-8")
    (tmp_path / "research/data/processed/curated.csv").write_text("id\n1\n", encoding="utf-8")
    (tmp_path / "pipeline/config/pipeline.yaml").write_text("project: {}\n", encoding="utf-8")
    (tmp_path / "research/models/stale.bin").write_bytes(b"unprovenanced")
    settings = {
        "project": {"root": str(tmp_path)},
        "paths": {
            "raw": "research/data/raw",
            "processed": "research/data/processed",
            "models": "research/models",
            "reports": "research/reports",
        },
    }

    with pytest.raises(RuntimeError, match="No release model manifests"):
        run_manifest_stage(settings)


def test_release_verification_requires_all_five_manifests(tmp_path):
    for directory in ("research/data/raw", "research/data/processed", "research/reports/manifests"):
        (tmp_path / directory).mkdir(parents=True)
    (tmp_path / "research/data/raw/source.csv").write_text("id\n1\n", encoding="utf-8")
    (tmp_path / "research/data/processed/curated.csv").write_text("id\n1\n", encoding="utf-8")
    data = create_data_manifests(
        tmp_path / "research/data/raw",
        tmp_path / "research/data/processed",
        output_directory=tmp_path / "research/reports/manifests",
    )
    assert set(data) == {"raw", "processed"}
    settings = {
        "project": {"root": str(tmp_path)},
        "paths": {
            "raw": "research/data/raw",
            "processed": "research/data/processed",
            "models": "research/models",
            "reports": "research/reports",
        },
    }

    with pytest.raises(RuntimeError, match="software, models, reports"):
        run_verify_stage(settings)
    missing = json.loads(
        (tmp_path / "research/reports/verification/software_verification.json").read_text(encoding="utf-8")
    )
    assert missing["issues"][0]["code"] == "missing_manifest"


def test_enabled_analysis_extends_release_dag_and_detects_tampering(tmp_path):
    settings, processed, software, models = _analysis_release_fixture(tmp_path)
    assert "analysis_build_metadata.json" in ANALYSIS_MANIFEST_EXCLUDES

    analysis_dir = tmp_path / "research/analysis"
    (analysis_dir / "chemical_intelligence.csv").write_text("structure_id,priority\nA,1\n", encoding="utf-8")
    _write_analysis_summary(
        analysis_dir,
        tmp_path / "research/data/processed",
        tmp_path / "research/reports",
    )
    analysis = _analysis_manifest_candidate(
        analysis_dir,
        processed,
        software,
        models,
    )
    (analysis_dir / "analysis_build_metadata.json").write_text(
        json.dumps(
            _expected_analysis_metadata(
                settings,
                processed,
                software,
                models,
                analysis,
            )
        ),
        encoding="utf-8",
    )

    (tmp_path / "research/reports/report.txt").write_text("report\n", encoding="utf-8")
    reports = _report_manifest_candidate(
        tmp_path / "research/reports",
        processed,
        software,
        models,
        analysis,
    )
    (tmp_path / "research/reports/report_build_metadata.json").write_text(
        json.dumps(
            _expected_report_metadata(
                settings,
                processed,
                software,
                models,
                reports,
                analysis,
            )
        ),
        encoding="utf-8",
    )

    manifests = run_manifest_stage(settings)
    assert tuple(manifests) == (
        "raw",
        "processed",
        "software",
        "models",
        "analysis",
        "reports",
    )
    assert {item["stage"] for item in manifests["analysis"]["upstream"]} == {
        "processed",
        "models",
        "software",
    }
    assert {item["stage"] for item in manifests["reports"]["upstream"]} == {
        "processed",
        "models",
        "analysis",
        "software",
    }
    assert run_verify_stage(settings) == {
        "raw": True,
        "processed": True,
        "software": True,
        "models": True,
        "analysis": True,
        "reports": True,
    }

    (analysis_dir / "chemical_intelligence.csv").write_text(
        "structure_id,priority\nA,999\n", encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="analysis"):
        run_verify_stage(settings)
    failed = json.loads(
        (tmp_path / "research/reports/verification/analysis_verification.json").read_text(encoding="utf-8")
    )
    assert not failed["valid"]
    assert "manifest_scope_changed" in {issue["code"] for issue in failed["issues"]}


def test_analysis_directory_promotion_rolls_back_on_build_failure(tmp_path, monkeypatch):
    settings, processed, software, models = _analysis_release_fixture(tmp_path)
    analysis_dir = tmp_path / "research/analysis"
    (analysis_dir / "previous.csv").write_text("id\nold\n", encoding="utf-8")

    analysis_module = types.ModuleType("menin_discovery.analysis")

    def fail_analysis(processed_dir, reports_dir, output_dir, *, settings):
        del processed_dir, reports_dir, settings
        (output_dir / "partial.csv").write_text("id\npartial\n", encoding="utf-8")
        raise RuntimeError("analysis failed")

    analysis_module.write_chemical_intelligence = fail_analysis
    monkeypatch.setitem(sys.modules, "menin_discovery.analysis", analysis_module)
    with pytest.raises(RuntimeError, match="analysis failed"):
        run_analysis_stage(settings)
    assert (analysis_dir / "previous.csv").read_text(encoding="utf-8") == "id\nold\n"
    assert not (analysis_dir / "partial.csv").exists()

    def write_analysis(processed_dir, reports_dir, output_dir, *, settings):
        del settings
        (output_dir / "complete.csv").write_text("id\nnew\n", encoding="utf-8")
        _write_analysis_summary(output_dir, processed_dir, reports_dir)
        return {"rows": 1}

    analysis_module.write_chemical_intelligence = write_analysis
    assert run_analysis_stage(settings) == analysis_dir
    assert not (analysis_dir / "previous.csv").exists()
    assert (analysis_dir / "complete.csv").read_text(encoding="utf-8") == "id\nnew\n"
    analysis = _analysis_manifest_candidate(
        analysis_dir,
        processed,
        software,
        models,
    )
    metadata = json.loads((analysis_dir / "analysis_build_metadata.json").read_text(encoding="utf-8"))
    assert metadata == _expected_analysis_metadata(
        settings,
        processed,
        software,
        models,
        analysis,
    )


def test_report_stage_validates_and_receives_analysis_directory(tmp_path, monkeypatch):
    settings, processed, software, models = _analysis_release_fixture(tmp_path)
    analysis_dir = tmp_path / "research/analysis"
    (analysis_dir / "chemical_intelligence.csv").write_text("structure_id,priority\nA,1\n", encoding="utf-8")
    _write_analysis_summary(
        analysis_dir,
        tmp_path / "research/data/processed",
        tmp_path / "research/reports",
    )
    analysis = _analysis_manifest_candidate(
        analysis_dir,
        processed,
        software,
        models,
    )
    (analysis_dir / "analysis_build_metadata.json").write_text(
        json.dumps(
            _expected_analysis_metadata(
                settings,
                processed,
                software,
                models,
                analysis,
            )
        ),
        encoding="utf-8",
    )

    import menin_discovery.reporting as reporting

    received = {}

    def write_report(processed_dir, reports_dir, models_dir, *, settings, analysis_dir):
        received.update(
            {
                "processed_dir": processed_dir,
                "models_dir": models_dir,
                "settings": settings,
                "analysis_dir": analysis_dir,
            }
        )
        (reports_dir / "tables").mkdir(parents=True)
        (reports_dir / "publication_summary.md").write_text("publication\n", encoding="utf-8")
        (reports_dir / "summary.md").write_text("summary\n", encoding="utf-8")
        return reports_dir / "publication_summary.md"

    monkeypatch.setattr(reporting, "write_summary_report", write_report)
    assert run_report_stage(settings) == tmp_path / "research/reports/publication_summary.md"
    assert received == {
        "processed_dir": tmp_path / "research/data/processed",
        "models_dir": tmp_path / "research/models",
        "settings": settings,
        "analysis_dir": analysis_dir,
    }
    report_metadata = json.loads(
        (tmp_path / "research/reports/report_build_metadata.json").read_text(encoding="utf-8")
    )
    assert report_metadata["analysis_dataset_sha256"] == analysis["dataset_sha256"]

    (tmp_path / "research/reports/menin_with_predicted_herg_risk.csv").write_text(
        "structure_id,predicted_herg_blocker_probability\nA,0.9\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="input_sha256"):
        _validate_analysis_lineage(settings, processed, software, models)
