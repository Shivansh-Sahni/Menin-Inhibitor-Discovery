from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest
from menin_discovery.platform_data_bulk_canonical import (
    _component_inventory,
    _normalize_partition_records,
    _partition_dataset_manifest,
    _write_normative_singleton,
)
from menin_discovery.platform_data_pipeline import binding_free_energy_view
from menin_discovery.platform_data_qc import (
    _audit_binding_free_energy,
    _create_qc_state,
    _parquet_row_count,
    _verify_component_inventory,
    _verify_full_build_artifacts,
    _verify_full_build_count_conservation,
    _verify_model_readiness_accounting,
    _verify_partitioned_dataset,
    run_platform_qc,
)
from menin_discovery.platform_data_schema import arrow_schema_contract, canonical_json
from menin_discovery.platform_data_sources import sha256_file


def test_qc_rejects_cross_part_arrow_schema_drift_from_all_null_column(
    tmp_path: Path,
) -> None:
    canonical = tmp_path / "canonical"
    directory = canonical / "molecules"
    directory.mkdir(parents=True)
    paths = [directory / "part-00000.parquet", directory / "part-00001.parquet"]
    pd.DataFrame({"id": ["A"], "value": pd.Series([None], dtype="float64")}).to_parquet(paths[0], index=False)
    pd.DataFrame({"id": ["B"], "value": [2.0]}).to_parquet(paths[1], index=False)
    schema = pq.ParquetFile(paths[0]).schema_arrow.remove_metadata()
    contract = arrow_schema_contract(schema)
    parts = [
        {
            "relative_path": path.relative_to(canonical).as_posix(),
            "rows": 1,
            "sha256": sha256_file(path),
            "arrow_schema_sha256": contract["sha256"],
        }
        for path in paths
    ]

    def dataset_digest() -> str:
        return hashlib.sha256(
            canonical_json(
                [
                    {
                        "path": part["relative_path"],
                        "rows": part["rows"],
                        "sha256": part["sha256"],
                        "arrow_schema_sha256": part["arrow_schema_sha256"],
                    }
                    for part in parts
                ]
            ).encode("utf-8")
        ).hexdigest()

    record = {
        "rows": 2,
        "part_count": 2,
        "parts": parts,
        "arrow_schema": contract,
        "dataset_sha256": dataset_digest(),
    }
    issues: list[dict[str, object]] = []
    _verify_partitioned_dataset(canonical, "molecules", record, issues)
    assert issues == []

    pd.DataFrame({"id": ["A"], "value": [None]}).to_parquet(paths[0], index=False)
    parts[0]["sha256"] = sha256_file(paths[0])
    record["dataset_sha256"] = dataset_digest()
    _verify_partitioned_dataset(canonical, "molecules", record, issues)
    assert any(issue["code"] == "arrow_schema_contract_mismatch" for issue in issues)


def test_qc_recomputes_root_singleton_schema_and_registry_only_fields(
    tmp_path: Path,
) -> None:
    canonical = tmp_path / "full_chembl37"
    canonical.mkdir()
    frames = {
        "sources": pd.DataFrame({"source_id": ["SRC1"]}),
        "source_files": pd.DataFrame({"source_file_id": ["FILE1"], "size_bytes": [10]}),
        "task_registry": pd.DataFrame(
            {
                "task_id": ["TASK1"],
                "row_count": [1],
                "relation_counts_json": ['{"=":1}'],
            }
        ),
    }
    artifacts = {
        name: _write_normative_singleton(
            frame,
            canonical / f"{name}.parquet",
            canonical,
        )
        for name, frame in frames.items()
    }
    manifest = {
        "build_type": "public_chembl37_full_specialized_canonical",
        "shard_artifacts": [],
        "shard_dataset_schemas": {},
        "entity_artifacts": artifacts,
    }
    issues: list[dict[str, object]] = []
    _verify_full_build_artifacts(canonical, manifest, issues)
    assert issues == []

    task_path = canonical / "task_registry.parquet"
    pd.DataFrame({"task_id": ["TASK1"], "row_count": [1], "relation_counts_json": [1]}).to_parquet(
        task_path, index=False
    )
    artifacts["task_registry"].update(
        {
            "rows": 1,
            "sha256": sha256_file(task_path),
            "size_bytes": task_path.stat().st_size,
        }
    )
    _verify_full_build_artifacts(canonical, manifest, issues)
    assert any(issue["code"] == "arrow_schema_contract_mismatch" for issue in issues)


def test_qc_reconciles_sequence_exclusion_and_proves_no_task_leakage(
    tmp_path: Path,
) -> None:
    canonical = tmp_path / "full_chembl37"
    path = canonical / "task_exclusions" / "default" / "part-00000.parquet"
    path.parent.mkdir(parents=True)
    exclusion = pd.DataFrame(
        {
            "observation_id": ["OBS1"],
            "task_id": ["TASK1"],
            "task_type": ["default__other__ec50"],
            "task_scope": ["default"],
            "source_id": ["SRC1"],
            "snapshot_id": ["SNP1"],
            "source_record_id": ["REC1"],
            "molecule_id": ["M1"],
            "protein_id": ["P1"],
            "assay_id": ["A1"],
            "canonical_target_id": ["CHEMBL612545"],
            "required_modalities": ["small_molecule_structure;protein_sequence"],
            "model_readiness_exclusion_reason": ["missing_protein_sequence"],
            "missing_standardized_smiles": [False],
            "missing_standard_inchi_key": [False],
            "missing_protein_sequence": [True],
        }
    )
    exclusion.to_parquet(path, index=False)
    part = {
        "path": path.name,
        "relative_path": path.relative_to(canonical).as_posix(),
        "rows": 1,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }
    contract = _normalize_partition_records(canonical, [part])
    dataset = _partition_dataset_manifest([part], contract)
    stage_counts = {
        "default": {"candidate": 1, "eligible": 0, "excluded": 1},
        "derived_sensitivity": {"candidate": 0, "eligible": 0, "excluded": 0},
    }
    empty_dimensions = {
        dimension: {}
        for dimension in (
            "task_type",
            "source_id",
            "protein_id",
            "canonical_target_id",
        )
    }
    default_dimensions = {
        "task_type": {"default__other__ec50": {"candidate": 1, "eligible": 0, "excluded": 1}},
        "source_id": {"SRC1": {"candidate": 1, "eligible": 0, "excluded": 1}},
        "protein_id": {"P1": {"candidate": 1, "eligible": 0, "excluded": 1}},
        "canonical_target_id": {"CHEMBL612545": {"candidate": 1, "eligible": 0, "excluded": 1}},
    }
    manifest = {
        "build_type": "public_chembl37_full_specialized_canonical",
        "model_readiness_exclusion_datasets": {"default": dataset},
        "model_readiness_policy": {
            "policy_version": "platform-model-readiness-v1",
            "allowed_modality_declarations": [
                "small_molecule_structure",
                "small_molecule_structure;protein_sequence",
            ],
            "reason_order": [
                "missing_standardized_smiles",
                "missing_standard_inchi_key",
                "missing_protein_sequence",
            ],
            "stage_counts": stage_counts,
            "reason_counts": {
                "default": {"missing_protein_sequence": 1},
                "derived_sensitivity": {},
            },
            "reason_combination_counts": {
                "default": {"missing_protein_sequence": 1},
                "derived_sensitivity": {},
            },
            "dimension_counts": {
                "default": default_dimensions,
                "derived_sensitivity": empty_dimensions,
            },
            "exclusion_artifact_root": "task_exclusions",
            "evidence_layer_policy": (
                "source observations and lineage remain unchanged; only model-task admission is gated"
            ),
        },
    }
    connection = sqlite3.connect(":memory:")
    _create_qc_state(connection)
    connection.execute("INSERT INTO molecules VALUES ('M1',1,1,1)")
    connection.execute("INSERT INTO proteins VALUES ('P1',0,'CHEMBL612545')")
    connection.execute(
        "INSERT INTO observations VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            "OBS1",
            "SRC1",
            "SNP1",
            "REC1",
            "M1",
            "P1",
            "A1",
            "other",
            "EC50",
            "=",
            "nM",
            1.0,
            "included",
            "",
            "experimental_summary",
            "public_redistributable",
            "",
            "",
            "",
            "",
            "",
        ),
    )
    connection.execute("INSERT INTO lineage VALUES ('L1','OBS1','SRC1','SNP1','FILE1','primary')")
    issues: list[dict[str, object]] = []
    _verify_model_readiness_accounting(
        canonical,
        manifest,
        issues,
        connection,
        task_counts={"default": 0, "derived_sensitivity": 0},
        task_dimension_counts={
            scope: {
                dimension: Counter()
                for dimension in (
                    "task_type",
                    "source_id",
                    "protein_id",
                    "canonical_target_id",
                )
            }
            for scope in ("default", "derived_sensitivity")
        },
    )
    assert issues == []
    assert connection.execute("SELECT COUNT(*) FROM task_input_exclusions").fetchone() == (1,)
    connection.close()


def _write_qc_fixture(root: Path, *, rows: int, mismatch_last_primary: bool) -> None:
    root.mkdir(parents=True)
    source = pd.DataFrame(
        [
            {
                "source_id": "SRC1",
                "snapshot_id": "SNP1",
                "source_name": "ChEMBL",
                "source_version": "37",
                "retrieval_date_utc": "2026-08-01T00:00:00Z",
                "source_url": "https://www.ebi.ac.uk/chembl/",
                "license_name": "CC BY-SA 3.0",
                "license_status": "verified",
                "access_class": "public_redistributable",
            }
        ]
    )
    source.to_parquet(root / "sources.parquet", index=False)
    pd.DataFrame(
        [
            {
                "source_file_id": "FILE1",
                "source_id": "SRC1",
                "snapshot_id": "SNP1",
                "relative_path": "raw/file.parquet",
                "sha256": "a" * 64,
                "size_bytes": 1,
                "immutability_status": "content_hashed",
            }
        ]
    ).to_parquet(root / "source_files.parquet", index=False)
    pd.DataFrame(
        [
            {
                "molecule_id": "M1",
                "structure_id": "STR1",
                "submitted_smiles": "CC",
                "canonical_smiles": "CC",
                "standardized_smiles": "CC",
                "standard_inchi_key": "OTMSDBZUPAUEDD-UHFFFAOYSA-N",
                "standardization_version": "test",
                "identity_resolution_status": "resolved",
            }
        ]
    ).to_parquet(root / "molecules.parquet", index=False)
    pd.DataFrame(
        [
            {
                "molecule_alias_id": "MA1",
                "molecule_id": "M1",
                "source_id": "SRC1",
                "snapshot_id": "SNP1",
                "source_compound_id": "CHEMBL1",
                "source_record_id": "ChEMBL:molecule:CHEMBL1",
            }
        ]
    ).to_parquet(root / "molecule_aliases.parquet", index=False)
    pd.DataFrame(
        [
            {
                "protein_id": "P1",
                "entity_type": "single_protein",
                "canonical_target_id": "P12345",
                "target_name": "Target",
                "uniprot_accession": "P12345",
                "sequence": "AAAA",
                "species": "Homo sapiens",
                "identity_resolution_status": "resolved",
            }
        ]
    ).to_parquet(root / "proteins.parquet", index=False)
    pd.DataFrame(
        columns=[
            "construct_id",
            "protein_id",
            "sequence",
            "sequence_sha256",
            "source_id",
            "source_record_id",
        ]
    ).to_parquet(root / "protein_constructs.parquet", index=False)
    pd.DataFrame(
        [
            {
                "assay_id": "A1",
                "source_id": "SRC1",
                "snapshot_id": "SNP1",
                "source_assay_id": "CHEMBL-A1",
                "protein_id": "P1",
                "assay_type": "B",
                "assay_family": "binding",
                "description": "purified protein binding",
                "protocol_completeness": 1.0,
            }
        ]
    ).to_parquet(root / "assays.parquet", index=False)

    observation_ids = [f"OBS-{index:06d}" for index in range(rows)]
    observations = pd.DataFrame(
        {
            "observation_id": observation_ids,
            "source_id": "SRC1",
            "snapshot_id": "SNP1",
            "source_record_id": [f"R-{index}" for index in range(rows)],
            "molecule_id": "M1",
            "protein_id": "P1",
            "assay_id": "A1",
            "evidence_domain": "binding",
            "endpoint": "Kd",
            "endpoint_family": "equilibrium_affinity",
            "relation": "=",
            "value_raw": "10",
            "value_numeric": 10.0,
            "original_unit": "nM",
            "canonical_value": 10.0,
            "canonical_unit": "nM",
            "lower_bound": 10.0,
            "upper_bound": 10.0,
            "observation_kind": "experimental_summary",
            "evidence_stage": "preclinical_in_vitro",
            "development_stage": "unknown",
            "result_status": "reported",
            "quality_grade": "protocol_sufficient",
            "access_class": "public_redistributable",
            "inclusion_status": "included",
            "exclusion_reason": "",
            "dedup_group_id": "D1",
            "conflict_group_id": "",
            "document_id": "DOC1",
            "document_year": 2020,
            "activity_origin_name": "ChEMBL",
        }
    )
    observations.to_parquet(root / "observations.parquet", index=False)
    lineage_snapshots = ["SNP1"] * rows
    if mismatch_last_primary:
        lineage_snapshots[-1] = "WRONG-SNAPSHOT"
    pd.DataFrame(
        {
            "lineage_id": [f"LIN-{index:06d}" for index in range(rows)],
            "observation_id": observation_ids,
            "source_id": "SRC1",
            "snapshot_id": lineage_snapshots,
            "source_file_id": "FILE1",
            "lineage_role": "primary",
        }
    ).to_parquet(root / "observation_lineage.parquet", index=False)

    task = pd.DataFrame(
        [
            {
                "task_id": "TASK1",
                "task_type": "default__binding__kd__binding__nm__continuous_exact",
                "observation_id": observation_ids[0],
                "molecule_id": "M1",
                "protein_id": "P1",
                "assay_id": "A1",
                "source_id": "SRC1",
                "snapshot_id": "SNP1",
                "source_record_id": "R-0",
                "label_kind": "continuous_exact",
                "label_value": 10.0,
                "label_text": "",
                "label_relation": "=",
                "label_lower_bound": 10.0,
                "label_upper_bound": 10.0,
                "label_unit": "nM",
                "observation_kind": "experimental_summary",
                "access_class": "public_redistributable",
                "default_task_eligible": True,
                "sensitivity_task_eligible": False,
                "evidence_domain": "binding",
                "endpoint": "Kd",
                "assay_family": "binding",
                "canonical_target_id": "P12345",
                "inclusion_status": "included",
                "standardized_smiles": "CC",
                "standard_inchi_key": "OTMSDBZUPAUEDD-UHFFFAOYSA-N",
                "sequence": "AAAA",
                "required_modalities": ("small_molecule_structure;protein_sequence"),
            }
        ]
    )
    (root / "tasks").mkdir()
    task.to_parquet(root / "tasks" / "public_model_tasks.parquet", index=False)
    pd.DataFrame(
        [
            {
                "task_id": "TASK1",
                "task_type": task.loc[0, "task_type"],
                "evidence_domain": "binding",
                "endpoint": "Kd",
                "assay_family": "binding",
                "label_kind": "continuous_exact",
                "label_unit": "nM",
                "observation_kind": "experimental_summary",
                "default_task_eligible": True,
                "sensitivity_task_eligible": False,
                "required_modalities": ("small_molecule_structure;protein_sequence"),
                "policy_version": "platform-task-contract-v1",
                "row_count": 1,
                "relation_counts_json": json.dumps({"=": 1}, separators=(",", ":")),
            }
        ]
    ).to_parquet(root / "task_registry.parquet", index=False)


def test_qc_cli_core_contract_passes_small_exact_fixture(tmp_path: Path) -> None:
    canonical = tmp_path / "canonical"
    reports = tmp_path / "reports"
    _write_qc_fixture(canonical, rows=2, mismatch_last_primary=False)
    manifest_path = canonical / "build_manifest.json"
    manifest_path.write_text(
        json.dumps({"build_type": "unit_test_canonical"}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = run_platform_qc(canonical, reports)
    assert report["qc_passed"] is True
    assert report["build_manifest_sha256"] == sha256_file(manifest_path)
    missingness = pd.read_csv(reports / "qc_missingness.csv")
    assert {"missing_rows", "denominator_rows", "missing_rate"}.issubset(missingness.columns)
    assert {
        "sources",
        "source_files",
        "molecules",
        "molecule_aliases",
        "proteins",
        "protein_constructs",
        "assays",
        "observations",
        "observation_lineage",
        "tasks",
        "task_registry",
    }.issubset(set(missingness["table"]))


def test_qc_detects_primary_tuple_mismatch_above_100k_without_scale_skip(
    tmp_path: Path,
) -> None:
    canonical = tmp_path / "canonical"
    reports = tmp_path / "reports"
    _write_qc_fixture(canonical, rows=100_001, mismatch_last_primary=True)
    with pytest.raises(RuntimeError, match="Platform QC failed"):
        run_platform_qc(canonical, reports)
    issues = pd.read_csv(reports / "qc_issues.csv")
    assert "primary_source_snapshot_mismatch" in set(issues["code"])
    assert "orphan_lineage_file_source_snapshot_tuple" in set(issues["code"])


def test_qc_recomputes_every_binding_free_energy_digest_and_roundtrip(tmp_path: Path) -> None:
    canonical = tmp_path / "canonical"
    (canonical / "views").mkdir(parents=True)
    source = pd.DataFrame(
        [
            {
                "observation_id": "OBS1",
                "source_id": "SRC1",
                "snapshot_id": "SNP1",
                "source_record_id": "R1",
                "molecule_id": "M1",
                "protein_id": "P1",
                "assay_id": "A1",
                "endpoint": "Kd",
                "relation": "=",
                "canonical_unit": "nM",
                "canonical_value": 12.0,
                "inclusion_status": "included",
                "observation_kind": "experimental_summary",
            }
        ]
    )
    derivations, derived = binding_free_energy_view(
        source,
        pd.DataFrame([{"assay_id": "A1", "temperature_c": float("nan")}]),
        pd.DataFrame([{"protein_id": "P1", "entity_type": "single_protein"}]),
    )
    pd.concat([source, derived], ignore_index=True, sort=False).to_parquet(
        canonical / "observations.parquet", index=False
    )
    derivations.to_parquet(
        canonical / "views" / "binding_free_energy_standard.parquet",
        index=False,
    )
    issues: list[dict[str, object]] = []
    audit = _audit_binding_free_energy(canonical, issues)  # type: ignore[arg-type]
    assert issues == []
    assert audit["derivation_rows"] == 1

    tampered = derivations.copy()
    tampered.loc[0, "label_lineage_digest"] = "0" * 64
    tampered.to_parquet(
        canonical / "views" / "binding_free_energy_standard.parquet",
        index=False,
    )
    issues = []
    _audit_binding_free_energy(canonical, issues)  # type: ignore[arg-type]
    assert any(issue["code"] == "binding_free_energy_integrity_failure" for issue in issues)


def test_parquet_footer_row_count_is_exact_above_100k(tmp_path: Path) -> None:
    path = tmp_path / "rows.parquet"
    pd.DataFrame({"row_id": range(200_001)}).to_parquet(path, index=False)
    assert _parquet_row_count(path) == 200_001


def test_qc_rejects_task_observation_provenance_drift(tmp_path: Path) -> None:
    canonical = tmp_path / "canonical"
    reports = tmp_path / "reports"
    _write_qc_fixture(canonical, rows=2, mismatch_last_primary=False)
    task_path = canonical / "tasks" / "public_model_tasks.parquet"
    task = pd.read_parquet(task_path)
    task.loc[0, "source_record_id"] = "WRONG"
    task.to_parquet(task_path, index=False)
    with pytest.raises(RuntimeError, match="Platform QC failed"):
        run_platform_qc(canonical, reports)
    issues = pd.read_csv(reports / "qc_issues.csv")
    assert "task_observation_identity_or_provenance_mismatch" in set(issues["code"])


@pytest.mark.parametrize(
    ("updates", "expected_code"),
    [
        (
            {"label_value": float("nan")},
            "continuous_exact_cross_column_violation",
        ),
        (
            {
                "label_kind": "continuous_censored",
                "label_relation": "<=",
                "label_value": float("nan"),
                "label_lower_bound": 10.0,
                "label_upper_bound": 10.0,
            },
            "continuous_censored_cross_column_violation",
        ),
        (
            {"label_kind": "categorical", "label_unit": "class"},
            "categorical_herg_threshold_provenance_violation",
        ),
    ],
)
def test_qc_rejects_cross_column_task_contract_violations(
    tmp_path: Path,
    updates: dict[str, object],
    expected_code: str,
) -> None:
    canonical = tmp_path / "canonical"
    reports = tmp_path / "reports"
    _write_qc_fixture(canonical, rows=2, mismatch_last_primary=False)
    task_path = canonical / "tasks" / "public_model_tasks.parquet"
    task = pd.read_parquet(task_path)
    for column, value in updates.items():
        task.loc[0, column] = value
    task.to_parquet(task_path, index=False)
    with pytest.raises(RuntimeError, match="Platform QC failed"):
        run_platform_qc(canonical, reports)
    issues = pd.read_csv(reports / "qc_issues.csv")
    assert expected_code in set(issues["code"])


def test_component_inventory_detects_tamper_and_unmanifested_files(tmp_path: Path) -> None:
    canonical = tmp_path / "canonical"
    canonical.mkdir()
    pd.DataFrame({"value": [1]}).to_csv(canonical / "data.csv", index=False)
    manifest = {
        "build_type": "public_chembl37_full_specialized_canonical",
        "component_inventory": _component_inventory(canonical),
        "qc_acceptance": {
            "required_before_promotion": True,
            "report_path": "qc_report.json",
            "binding": "qc_report.build_manifest_sha256 == SHA256(build_manifest.json)",
        },
    }
    issues: list[dict[str, object]] = []
    _verify_component_inventory(canonical, manifest, issues)  # type: ignore[arg-type]
    assert issues == []

    pd.DataFrame({"value": [2]}).to_csv(canonical / "data.csv", index=False)
    issues = []
    _verify_component_inventory(canonical, manifest, issues)  # type: ignore[arg-type]
    assert any(issue["code"] == "component_inventory_record_mismatch" for issue in issues)

    manifest["component_inventory"] = _component_inventory(canonical)
    (canonical / "unmanifested.txt").write_text("unexpected\n", encoding="utf-8")
    issues = []
    _verify_component_inventory(canonical, manifest, issues)  # type: ignore[arg-type]
    assert any(issue["code"] == "component_inventory_membership_mismatch" for issue in issues)


def test_full_build_stage_counts_are_conserved() -> None:
    manifest = {
        "build_type": "public_chembl37_full_specialized_canonical",
        "unique_activity_rows": 2,
        "derived_binding_free_energy_rows": 1,
        "input_summary": {"view_row_counts": {"single_protein_kd_ki": 2}},
        "inventory_membership_counts_before_cross_view_dedup": {"single_protein_kd_ki": 2},
        "entity_counts": {
            "molecules": 1,
            "molecule_aliases": 1,
            "proteins": 1,
            "protein_constructs": 0,
            "assays": 1,
            "tasks": 1,
            "sensitivity_tasks": 1,
        },
        "task_registry_rows": 2,
    }
    issues: list[dict[str, object]] = []
    _verify_full_build_count_conservation(
        manifest,
        issues,  # type: ignore[arg-type]
        observation_rows=3,
        lineage_rows=3,
        derived_observation_rows=1,
        derivation_rows=1,
        entity_counts={
            "molecules": 1,
            "molecule_aliases": 1,
            "proteins": 1,
            "protein_constructs": 0,
            "assays": 1,
        },
        default_task_rows=1,
        sensitivity_task_rows=1,
        task_registry_rows=2,
    )
    assert issues == []
    manifest["unique_activity_rows"] = 3
    _verify_full_build_count_conservation(
        manifest,
        issues,  # type: ignore[arg-type]
        observation_rows=3,
        lineage_rows=3,
        derived_observation_rows=1,
        derivation_rows=1,
        entity_counts={
            "molecules": 1,
            "molecule_aliases": 1,
            "proteins": 1,
            "protein_constructs": 0,
            "assays": 1,
        },
        default_task_rows=1,
        sensitivity_task_rows=1,
        task_registry_rows=2,
    )
    assert any(issue["code"] == "observation_stage_count_nonconservation" for issue in issues)


def test_qc_temp_state_is_removed_when_audit_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    temporary_root = tmp_path / "temporary"
    temporary_root.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(temporary_root))
    with pytest.raises(FileNotFoundError):
        run_platform_qc(tmp_path / "missing-canonical", tmp_path / "reports")
    assert list(temporary_root.iterdir()) == []


def test_qc_import_uses_writable_project_cache_defaults(tmp_path: Path) -> None:
    environment = os.environ.copy()
    environment.pop("MPLCONFIGDIR", None)
    environment.pop("XDG_CACHE_HOME", None)
    environment["HOME"] = str(tmp_path / "missing-home")
    source_root = Path(__file__).resolve().parents[1] / "src"
    environment["PYTHONPATH"] = os.pathsep.join([str(source_root), environment.get("PYTHONPATH", "")]).rstrip(
        os.pathsep
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import os; import menin_discovery.platform_data_qc; "
                "print(os.environ['MPLCONFIGDIR']); print(os.environ['XDG_CACHE_HOME'])"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    output_paths = result.stdout.strip().splitlines()
    assert output_paths[0].endswith("/.matplotlib_cache")
    assert output_paths[1].endswith("/.cache")
    assert "temporary cache" not in result.stderr.casefold()
    assert "not writable" not in result.stderr.casefold()
