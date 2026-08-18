from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd
import pytest
from menin_discovery.platform_data_schema import canonical_json
from menin_discovery.platform_data_sources import sha256_file
from menin_discovery.platform_statistical_analysis import (
    benjamini_hochberg,
    main,
    run_statistical_analysis,
    verify_statistical_analysis,
)


def _write_parquet(root: Path, relative: str, frame: pd.DataFrame) -> dict[str, object]:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return {
        "path": relative,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "rows": len(frame),
    }


def _observation(
    index: int,
    *,
    domain: str = "binding_affinity",
    endpoint: str = "Kd",
    assay: str = "A2",
    relation: str = "=",
    value: float | None = 10.0,
    lower: float | None = 10.0,
    upper: float | None = 10.0,
    status: str = "included",
    reason: str = "",
    kind: str = "experimental_summary",
) -> dict[str, object]:
    return {
        "observation_id": f"O{index:04d}",
        "source_id": "chembl",
        "snapshot_id": "chembl37-test",
        "source_record_id": f"activity:{index}",
        "molecule_id": f"M{index % 7}",
        "protein_id": "P1",
        "assay_id": assay,
        "evidence_domain": domain,
        "endpoint": endpoint,
        "relation": relation,
        "value_raw": "" if value is None else str(value),
        "value_numeric": value,
        "original_unit": "nM",
        "canonical_value": value,
        "canonical_unit": "nM" if endpoint != "standard_binding_free_energy" else "kcal/mol",
        "lower_bound": lower,
        "upper_bound": upper,
        "observation_kind": kind,
        "evidence_stage": "in_vitro" if index % 2 == 0 else "in_vivo",
        "development_stage": "unknown",
        "result_status": "reported",
        "quality_grade": "curated",
        "access_class": "public_redistributable",
        "inclusion_status": status,
        "exclusion_reason": reason,
        "dedup_group_id": "DUP-A" if index in {10, 11} else f"DUP-{index}",
        "conflict_group_id": "CONFLICT-A" if index in {10, 11} else "",
        "document_year": 2001 if index % 4 < 2 else 2011,
    }


def _task(row: dict[str, object], *, task_id: str = "TASK-KD") -> dict[str, object]:
    exact = row["relation"] == "="
    return {
        **row,
        "task_id": task_id,
        "task_type": "default__binding_affinity__kd__binding__nm__continuous_exact",
        "label_kind": "continuous_exact" if exact else "continuous_censored",
        "label_value": row["canonical_value"] if exact else None,
        "label_text": "",
        "label_relation": row["relation"],
        "label_lower_bound": row["lower_bound"],
        "label_upper_bound": row["upper_bound"],
        "label_unit": row["canonical_unit"],
        "default_task_eligible": True,
        "sensitivity_task_eligible": False,
        "required_modalities": "small_molecule_structure;protein_sequence",
        "assay_family": "binding",
        "canonical_target_id": "T1",
    }


def _synthetic_build(parent: Path) -> tuple[Path, Path]:
    root = parent / "full_chembl37"
    root.mkdir(parents=True)
    records: list[dict[str, object]] = []
    records.append(
        _write_parquet(
            root,
            "assays/part-00000.parquet",
            pd.DataFrame(
                [
                    {"assay_id": "A1", "assay_family": "herg_functional"},
                    {"assay_id": "A2", "assay_family": "binding"},
                ]
            ),
        )
    )
    records.append(
        _write_parquet(
            root,
            "proteins/part-00000.parquet",
            pd.DataFrame([{"protein_id": "P1", "canonical_target_id": "T1", "sequence": "ACDE"}]),
        )
    )
    records.append(
        _write_parquet(
            root,
            "molecules/part-00000.parquet",
            pd.DataFrame(
                [
                    {
                        "molecule_id": f"M{index}",
                        "standardized_smiles": "CC",
                        "standard_inchi_key": f"IK{index}",
                    }
                    for index in range(7)
                ]
            ),
        )
    )
    source_rows = [_observation(index) for index in range(120)]
    source_rows[15] = _observation(15, status="quarantined", reason="invalid_unit")
    herg_rows = [
        _observation(
            200, domain="herg", endpoint="IC50", assay="A1", value=5_000.0, lower=5_000.0, upper=5_000.0
        ),
        _observation(
            201, domain="herg", endpoint="IC50", assay="A1", value=40_000.0, lower=40_000.0, upper=40_000.0
        ),
        _observation(
            202, domain="herg", endpoint="IC50", assay="A1", value=20_000.0, lower=20_000.0, upper=20_000.0
        ),
        _observation(
            203,
            domain="herg",
            endpoint="IC50",
            assay="A1",
            relation=">",
            value=None,
            lower=5_000.0,
            upper=None,
        ),
    ]
    derived = _observation(
        300,
        endpoint="standard_binding_free_energy",
        value=-10.0,
        lower=-10.0,
        upper=-10.0,
        kind="derived",
    )
    observations = pd.DataFrame([*source_rows, *herg_rows, derived])
    records.append(_write_parquet(root, "observations/part-00000.parquet", observations))

    default_tasks = [_task(row) for row in source_rows if row["inclusion_status"] == "included"]
    herg_continuous: list[dict[str, object]] = []
    for row in herg_rows:
        task = _task(row, task_id="TASK-HERG-CONTINUOUS")
        task["task_type"] = "default__herg__ic50__herg_functional__nm__continuous"
        task["assay_family"] = "herg_functional"
        herg_continuous.append(task)
    binary_tasks: list[dict[str, object]] = []
    for row, label, value in ((herg_rows[0], "blocker", 1.0), (herg_rows[1], "nonblocker", 0.0)):
        task = _task(row, task_id="TASK-HERG-BINARY")
        task.update(
            {
                "task_type": "default__herg__ic50__herg_functional__binary_exact__10_30um",
                "assay_family": "herg_functional",
                "label_kind": "categorical",
                "label_value": value,
                "label_text": label,
                "label_relation": "=",
                "label_lower_bound": None,
                "label_upper_bound": None,
                "label_unit": "class",
            }
        )
        binary_tasks.append(task)
    default = pd.DataFrame([*default_tasks, *herg_continuous, *binary_tasks])
    records.append(_write_parquet(root, "tasks/default/default/part-00000.parquet", default))

    sensitivity = _task(derived, task_id="TASK-DG")
    sensitivity.update(
        {
            "task_type": "sensitivity__binding_affinity__standard_binding_free_energy",
            "label_unit": "kcal/mol",
            "default_task_eligible": False,
            "sensitivity_task_eligible": True,
        }
    )
    records.append(
        _write_parquet(
            root,
            "tasks/derived_sensitivity/dg/part-00000.parquet",
            pd.DataFrame([sensitivity]),
        )
    )
    exclusion = default.iloc[[0]].copy()
    exclusion["model_readiness_exclusion_reason"] = "missing_protein_sequence"
    records.append(
        _write_parquet(
            root,
            "task_exclusions/default/part-00000.parquet",
            exclusion,
        )
    )
    records.append(
        _write_parquet(
            root,
            "molecule_development_annotations/part-00000.parquet",
            pd.DataFrame(
                [
                    {
                        "development_metadata_id": "DEV1",
                        "molecule_id": "M1",
                        "max_phase": 2.0,
                        "first_approval": None,
                        "withdrawn_flag": 0,
                        "black_box_warning": 0,
                        "therapeutic_flag": 1,
                        "molecule_type": "Small molecule",
                        "semantic_role": "development_metadata_not_outcome_or_model_label",
                    }
                ]
            ),
        )
    )
    records.append(
        _write_parquet(
            root,
            "views/binding_free_energy_standard/part-00000.parquet",
            pd.DataFrame(
                [
                    {
                        "observation_id": "O0300",
                        "delta_g_kcal_mol": -10.0,
                        "temperature_source": "reference_temperature_approximation",
                        "temperature_k": 298.15,
                        "formula": "delta_g=RTlnKd",
                        "standard_state": "1 mol/L",
                        "source_kd_value": 10.0,
                        "source_kd_unit": "nM",
                    }
                ]
            ),
        )
    )
    task_datasets: dict[str, object] = {}
    task_shards: list[dict[str, object]] = []
    for record in [row for row in records if str(row["path"]).startswith("tasks/")]:
        relative = str(record["path"])
        row_count = record["rows"]
        size_bytes = record["size_bytes"]
        assert isinstance(row_count, int) and isinstance(size_bytes, int)
        scope = Path(relative).parts[1]
        task_type = (
            "synthetic_default_mixed_tasks"
            if scope == "default"
            else "synthetic_binding_free_energy_sensitivity"
        )
        arrow_digest = hashlib.sha256(f"schema:{task_type}".encode()).hexdigest()
        part = {
            "path": relative,
            "rows": row_count,
            "sha256": str(record["sha256"]),
            "arrow_schema_sha256": arrow_digest,
        }
        task_datasets[f"{scope}::{task_type}"] = {
            "task_scope": scope,
            "task_type": task_type,
            "row_count": row_count,
            "part_count": 1,
            "parts": [part],
            "arrow_schema": {"sha256": arrow_digest},
            "dataset_sha256": hashlib.sha256(canonical_json([part]).encode()).hexdigest(),
        }
        task_shards.append(
            {
                "relative_path": relative,
                "rows": row_count,
                "sha256": str(record["sha256"]),
                "size_bytes": size_bytes,
            }
        )
    task_datasets_path = root / "task_datasets.json"
    task_datasets_path.write_text(
        json.dumps(task_datasets, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    records.append(
        {
            "path": "task_datasets.json",
            "sha256": sha256_file(task_datasets_path),
            "size_bytes": task_datasets_path.stat().st_size,
        }
    )
    source_status = pd.Series([row["inclusion_status"] for row in [*source_rows, *herg_rows]]).value_counts()
    manifest = {
        "schema_version": "test-v1",
        "source_id": "chembl",
        "snapshot_id": "chembl37-test",
        "unique_activity_rows": len(source_rows) + len(herg_rows),
        "derived_binding_free_energy_rows": 1,
        "molecule_development_annotations": {"rows": 1},
        "canonical_attrition": {
            "inclusion_status_counts": {str(key): int(value) for key, value in source_status.items()},
            "exclusion_reason_counts": {"invalid_unit": 1},
        },
        "model_readiness_policy": {
            "stage_counts": {
                "default": {"candidate": len(default) + 1, "eligible": len(default), "excluded": 1},
                "derived_sensitivity": {"candidate": 1, "eligible": 1, "excluded": 0},
            }
        },
        "shard_artifacts": task_shards,
        "task_datasets_manifest_sha256": hashlib.sha256(canonical_json(task_datasets).encode()).hexdigest(),
        "component_inventory": sorted(records, key=lambda record: str(record["path"])),
    }
    manifest_path = root / "build_manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
    qc = parent / "qc_report.json"
    qc.write_text(
        json.dumps(
            {
                "qc_passed": True,
                "build_manifest_sha256": sha256_file(manifest_path),
                "counts": {"observations": len(observations)},
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return root, qc


def test_benjamini_hochberg_is_monotone_and_order_preserving() -> None:
    adjusted = benjamini_hochberg([0.04, 0.001, 0.03, 0.2])
    assert adjusted == pytest.approx([0.05333333333333334, 0.004, 0.05333333333333334, 0.2])
    with pytest.raises(ValueError, match="p-values"):
        benjamini_hochberg([1.1])


def test_statistical_analysis_is_manifest_bound_exact_and_zero_training(tmp_path: Path) -> None:
    canonical, qc = _synthetic_build(tmp_path / "input")
    output = tmp_path / "reports" / "statistical_analysis"
    manifest = run_statistical_analysis(canonical, qc, output, batch_size=17, sample_cap=5)
    assert manifest["zero_training"] is True
    assert manifest["training_actions"] == []
    assert manifest["reconciliation"]["source_observation_rows"] == 124
    assert manifest["reconciliation"]["derived_observation_rows"] == 1
    assert not (output / ".analysis_state.sqlite").exists()

    herg = pd.read_csv(output / "herg_10_30uM_support.csv")
    counts = dict(zip(herg["category"], herg["rows"], strict=True))
    assert counts["classifier_candidate__blocker_le_10uM"] == 1
    assert counts["classifier_candidate__nonblocker_ge_30uM"] == 1
    assert counts["not_class__exact_intermediate_10_to_30uM"] == 1
    assert counts["not_class__censored_or_interval"] == 1
    assert counts["emitted_class__blocker"] == 1
    assert counts["emitted_class__nonblocker"] == 1

    strata = pd.read_csv(output / "compatible_stratum_summaries.csv")
    assert set(strata["measure_role"]) >= {"exact_value", "lower_censor_bound"}
    assert not ((strata["endpoint"] == "Kd") & (strata["unit"] == "kcal/mol")).any()
    sensitivity = strata[strata["dataset"] == "kd_free_energy_sensitivity"]
    assert set(sensitivity["endpoint"]) == {"standard_binding_free_energy"}
    assert set(sensitivity["unit"]) == {"kcal/mol"}

    declared = manifest["exact_recursive_membership"]["paths"]
    actual = sorted(path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file())
    assert declared == actual
    for artifact in manifest["artifacts"]:
        assert sha256_file(output / artifact["path"]) == artifact["sha256"]


def test_statistical_analysis_rejects_unbound_qc_and_tampered_component(tmp_path: Path) -> None:
    canonical, qc = _synthetic_build(tmp_path / "input")
    qc_payload = json.loads(qc.read_text(encoding="utf-8"))
    qc_payload["build_manifest_sha256"] = "0" * 64
    qc.write_text(json.dumps(qc_payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="not bound"):
        run_statistical_analysis(canonical, qc, tmp_path / "bad-qc")

    canonical, qc = _synthetic_build(tmp_path / "second")
    observations = canonical / "observations" / "part-00000.parquet"
    observations.write_bytes(observations.read_bytes() + b"tamper")
    with pytest.raises(RuntimeError, match="size mismatch"):
        run_statistical_analysis(canonical, qc, tmp_path / "tampered")


def test_statistical_analysis_is_byte_deterministic(tmp_path: Path) -> None:
    canonical_one, qc_one = _synthetic_build(tmp_path / "one")
    canonical_two, qc_two = _synthetic_build(tmp_path / "two")
    output_one = tmp_path / "output-one"
    output_two = tmp_path / "output-two"
    first = run_statistical_analysis(canonical_one, qc_one, output_one, batch_size=19, sample_cap=7)
    second = run_statistical_analysis(canonical_two, qc_two, output_two, batch_size=23, sample_cap=7)
    first_hashes = {record["path"]: record["sha256"] for record in first["artifacts"]}
    second_hashes = {record["path"]: record["sha256"] for record in second["artifacts"]}
    assert first_hashes == second_hashes
    assert (output_one / "analysis_manifest.json").read_bytes() == (
        output_two / "analysis_manifest.json"
    ).read_bytes()


def _rewrite_analysis_manifest(output: Path, payload: dict[str, object]) -> None:
    (output / "analysis_manifest.json").write_text(canonical_json(payload) + "\n", encoding="utf-8")


def _refresh_artifact_record(output: Path, relative: str, manifest: dict[str, object]) -> None:
    artifacts = manifest["artifacts"]
    assert isinstance(artifacts, list)
    record = next(row for row in artifacts if isinstance(row, dict) and row["path"] == relative)
    path = output / relative
    record["sha256"] = sha256_file(path)
    record["size_bytes"] = path.stat().st_size


def test_verify_statistical_analysis_and_module_command_rebind_source(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    canonical, qc = _synthetic_build(tmp_path / "input")
    output = tmp_path / "analysis"
    manifest = run_statistical_analysis(canonical, qc, output, batch_size=17, sample_cap=5)

    local = verify_statistical_analysis(output)
    rebound = verify_statistical_analysis(output, canonical_build_root=canonical, qc_report_path=qc)
    assert local["status"] == "verified"
    assert local["source_reverified"] is False
    assert rebound["source_reverified"] is True
    assert rebound["artifact_count"] == len(manifest["artifacts"])
    assert rebound["zero_training"] is True
    assert (
        main(
            [
                "verify-existing",
                "--output-root",
                str(output),
                "--canonical-build-root",
                str(canonical),
                "--qc-report",
                str(qc),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["status"] == "verified"


def test_verifier_rejects_hash_row_schema_and_membership_drift(tmp_path: Path) -> None:
    canonical, qc = _synthetic_build(tmp_path / "input")
    pristine = tmp_path / "pristine"
    run_statistical_analysis(canonical, qc, pristine, batch_size=17, sample_cap=5)

    hash_drift = tmp_path / "hash-drift"
    shutil.copytree(pristine, hash_drift)
    with (hash_drift / "composition.csv").open("a", encoding="utf-8") as handle:
        handle.write("rogue,rogue,rogue,1\n")
    with pytest.raises(ValueError, match="size drift|SHA-256 drift"):
        verify_statistical_analysis(hash_drift)

    row_drift = tmp_path / "row-drift"
    shutil.copytree(pristine, row_drift)
    row_path = row_drift / "composition.csv"
    rows = row_path.read_text(encoding="utf-8").splitlines()
    row_path.write_text("\n".join(rows[:-1]) + "\n", encoding="utf-8")
    row_manifest = json.loads((row_drift / "analysis_manifest.json").read_text(encoding="utf-8"))
    _refresh_artifact_record(row_drift, "composition.csv", row_manifest)
    _rewrite_analysis_manifest(row_drift, row_manifest)
    with pytest.raises(ValueError, match="row-count drift"):
        verify_statistical_analysis(row_drift)

    schema_drift = tmp_path / "schema-drift"
    shutil.copytree(pristine, schema_drift)
    schema_path = schema_drift / "composition.csv"
    schema_rows = schema_path.read_text(encoding="utf-8").splitlines()
    schema_rows[0] = schema_rows[0].replace("dataset", "dataset_drift", 1)
    schema_path.write_text("\n".join(schema_rows) + "\n", encoding="utf-8")
    schema_manifest = json.loads((schema_drift / "analysis_manifest.json").read_text(encoding="utf-8"))
    _refresh_artifact_record(schema_drift, "composition.csv", schema_manifest)
    _rewrite_analysis_manifest(schema_drift, schema_manifest)
    with pytest.raises(ValueError, match="header drift"):
        verify_statistical_analysis(schema_drift)

    rogue = tmp_path / "rogue-membership"
    shutil.copytree(pristine, rogue)
    (rogue / "unbound-empty-directory").mkdir()
    with pytest.raises(ValueError, match="recursive membership"):
        verify_statistical_analysis(rogue)


def test_verifier_rejects_root_and_nested_symlinks(tmp_path: Path) -> None:
    canonical, qc = _synthetic_build(tmp_path / "input")
    output = tmp_path / "analysis"
    run_statistical_analysis(canonical, qc, output, batch_size=17, sample_cap=5)

    alias = tmp_path / "analysis-alias"
    alias.symlink_to(output, target_is_directory=True)
    with pytest.raises(ValueError, match="path chain contains a symlink"):
        verify_statistical_analysis(alias)
    with pytest.raises(ValueError, match="parent traversal"):
        verify_statistical_analysis(alias / ".." / output.name)

    nested = output / "plots" / "unbound-link.svg"
    nested.symlink_to(output / "plots" / "top_task_types.svg")
    with pytest.raises(ValueError, match="contains a symlink"):
        verify_statistical_analysis(output)


@pytest.mark.parametrize("drift", ["zero_training", "scientific_boundary", "absolute_path"])
def test_verifier_rejects_zero_training_scientific_and_path_drift(tmp_path: Path, drift: str) -> None:
    canonical, qc = _synthetic_build(tmp_path / "input")
    output = tmp_path / "analysis"
    run_statistical_analysis(canonical, qc, output, batch_size=17, sample_cap=5)
    manifest = json.loads((output / "analysis_manifest.json").read_text(encoding="utf-8"))
    if drift == "zero_training":
        manifest["zero_training"] = False
        manifest["training_actions"] = ["fit"]
    elif drift == "scientific_boundary":
        manifest["scientific_boundaries"][1] = "hERG is clinical cardiotoxicity."
    else:
        manifest["input_binding"]["canonical_build_root_name"] = "/tmp/escape"
    _rewrite_analysis_manifest(output, manifest)
    with pytest.raises(ValueError):
        verify_statistical_analysis(output)


def test_verifier_rebinds_canonical_task_counts_not_only_hashes(tmp_path: Path) -> None:
    canonical, qc = _synthetic_build(tmp_path / "input")
    output = tmp_path / "analysis"
    run_statistical_analysis(canonical, qc, output, batch_size=17, sample_cap=5)

    build_manifest_path = canonical / "build_manifest.json"
    build_manifest = json.loads(build_manifest_path.read_text(encoding="utf-8"))
    build_manifest["model_readiness_policy"]["stage_counts"]["default"]["candidate"] += 1
    build_manifest["model_readiness_policy"]["stage_counts"]["default"]["eligible"] += 1
    build_manifest_path.write_text(json.dumps(build_manifest, sort_keys=True) + "\n", encoding="utf-8")
    qc_payload = json.loads(qc.read_text(encoding="utf-8"))
    qc_payload["build_manifest_sha256"] = sha256_file(build_manifest_path)
    qc.write_text(json.dumps(qc_payload, sort_keys=True) + "\n", encoding="utf-8")

    analysis_manifest = json.loads((output / "analysis_manifest.json").read_text(encoding="utf-8"))
    analysis_manifest["input_binding"]["canonical_build_manifest_sha256"] = sha256_file(build_manifest_path)
    analysis_manifest["input_binding"]["canonical_qc_report_sha256"] = sha256_file(qc)
    _rewrite_analysis_manifest(output, analysis_manifest)
    with pytest.raises(ValueError, match="model-readiness counts"):
        verify_statistical_analysis(output, canonical_build_root=canonical, qc_report_path=qc)
