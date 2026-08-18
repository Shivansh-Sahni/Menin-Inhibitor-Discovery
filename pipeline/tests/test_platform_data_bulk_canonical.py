from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from menin_discovery.platform_data_bulk import (
    ACTIVITY_ARROW_SCHEMA,
    DEVELOPMENT_ARROW_SCHEMA,
    TARGET_COMPONENT_ARROW_SCHEMA,
    _write_frame_with_schema,
)
from menin_discovery.platform_data_bulk_canonical import (
    _bulk_protein_entities,
    _component_inventory,
    _component_metadata,
    _DiskRegistry,
    _load_input_parts,
    _normalize_partition_records,
    _partition_model_ready_tasks,
    _promote_qc_accepted_build,
    _select_bulk_source_files,
    _TaskRegistryAccumulator,
    _validate_derivation_bundle,
    _write_normative_singleton,
    materialize_chembl37_specialized_canonical,
)
from menin_discovery.platform_data_pipeline import (
    CHEMBL_SOURCE_ID,
    _join_model_inputs,
    binding_free_energy_view,
)
from menin_discovery.platform_data_schema import SCHEMA_VERSION, arrow_schema_contract
from menin_discovery.platform_data_sources import sha256_file

_SPECIALIZED_VIEWS = (
    "cardiac_qt_apd_inventory",
    "pk_adme_candidates",
    "herg_all_endpoints",
    "single_protein_kd_ki",
    "single_protein_ic50_ec50_candidates",
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_specialized_contract(tmp_path: Path) -> tuple[Path, Path]:
    interim = tmp_path / "interim"
    root = interim / "chembl_37_bulk" / "specialized_views"
    database_sha256 = "d" * 64
    activity_contract = arrow_schema_contract(ACTIVITY_ARROW_SCHEMA)
    development_contract = arrow_schema_contract(DEVELOPMENT_ARROW_SCHEMA)
    target_contract = arrow_schema_contract(TARGET_COMPONENT_ARROW_SCHEMA)
    full_part = root.parent / "activity_facts" / "part-00000.parquet"
    full_part.parent.mkdir(parents=True, exist_ok=True)
    full_frame = pd.DataFrame({field.name: [None] for field in ACTIVITY_ARROW_SCHEMA})
    full_frame.loc[0, "activity_id"] = 99
    _write_frame_with_schema(full_frame, full_part, ACTIVITY_ARROW_SCHEMA)
    _write_json(
        root.parent / "activity_export_manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "source_version": "ChEMBL_37",
            "database_sha256": database_sha256,
            "rows_written": 1,
            "part_count": 1,
            "arrow_schema": activity_contract,
            "parts": [
                {
                    "path": full_part.relative_to(root.parent).as_posix(),
                    "rows": 1,
                    "sha256": sha256_file(full_part),
                    "size_bytes": full_part.stat().st_size,
                    "arrow_schema_sha256": activity_contract["sha256"],
                }
            ],
        },
    )
    views: dict[str, object] = {}
    for index, view_name in enumerate(_SPECIALIZED_VIEWS, start=1):
        part = root / view_name / "part-00000.parquet"
        part.parent.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame({field.name: [None] for field in ACTIVITY_ARROW_SCHEMA})
        frame.loc[0, "activity_id"] = index
        _write_frame_with_schema(frame, part, ACTIVITY_ARROW_SCHEMA)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "view_name": view_name,
            "source_version": "ChEMBL_37",
            "database_sha256": database_sha256,
            "row_count": 1,
            "part_count": 1,
            "arrow_schema": activity_contract,
            "parts": [
                {
                    "path": part.relative_to(root).as_posix(),
                    "rows": 1,
                    "sha256": sha256_file(part),
                    "size_bytes": part.stat().st_size,
                    "arrow_schema_sha256": activity_contract["sha256"],
                }
            ],
        }
        views[view_name] = manifest
        _write_json(root / f"{view_name}_manifest.json", manifest)

    development_part = root / "molecule_development_annotations" / "part-00000.parquet"
    development_part.parent.mkdir(parents=True, exist_ok=True)
    development_frame = pd.DataFrame({field.name: [None] for field in DEVELOPMENT_ARROW_SCHEMA})
    development_frame.loc[0, "molecule_chembl_id"] = "CHEMBL1"
    _write_frame_with_schema(
        development_frame,
        development_part,
        DEVELOPMENT_ARROW_SCHEMA,
    )
    development = {
        "schema_version": SCHEMA_VERSION,
        "view_name": "molecule_development_annotations",
        "source_version": "ChEMBL_37",
        "database_sha256": database_sha256,
        "semantic_role": "metadata_only_not_outcome",
        "row_count": 1,
        "part_count": 1,
        "arrow_schema": development_contract,
        "parts": [
            {
                "path": development_part.relative_to(root).as_posix(),
                "rows": 1,
                "sha256": sha256_file(development_part),
                "size_bytes": development_part.stat().st_size,
                "arrow_schema_sha256": development_contract["sha256"],
            }
        ],
    }
    views["molecule_development_annotations"] = development
    _write_json(root / "molecule_development_annotations_manifest.json", development)

    target_path = root / "target_components.parquet"
    target_frame = pd.DataFrame({field.name: [None] for field in TARGET_COMPONENT_ARROW_SCHEMA})
    target_frame.loc[0, "target_chembl_id"] = "CHEMBL240"
    _write_frame_with_schema(
        target_frame,
        target_path,
        TARGET_COMPONENT_ARROW_SCHEMA,
    )
    target = {
        "schema_version": SCHEMA_VERSION,
        "view_name": "target_components",
        "source_version": "ChEMBL_37",
        "database_sha256": database_sha256,
        "path": target_path.name,
        "row_count": 1,
        "sha256": sha256_file(target_path),
        "size_bytes": target_path.stat().st_size,
        "query_sha256": "e" * 64,
        "arrow_schema": target_contract,
        "arrow_schema_sha256": target_contract["sha256"],
    }
    views["target_components"] = target
    _write_json(root / "target_components_manifest.json", target)
    _write_json(
        root / "specialized_views_manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "source_version": "ChEMBL_37",
            "database_sha256": database_sha256,
            "views": views,
        },
    )
    return interim, root


def test_disk_registry_deduplicates_activities_and_rejects_entity_collisions(
    tmp_path: Path,
) -> None:
    registry = _DiskRegistry(tmp_path / "registry.sqlite")
    assert registry.claim_activity_ids(pd.Series([1, 2]), "first").tolist() == [True, True]
    assert registry.claim_activity_ids(pd.Series([2, 3]), "second").tolist() == [False, True]
    registry.add_frame("molecules", "molecule_id", pd.DataFrame([{"molecule_id": "M1", "x": 1}]))
    registry.add_frame("molecules", "molecule_id", pd.DataFrame([{"molecule_id": "M1", "x": 2}]))
    with pytest.raises(RuntimeError, match="Conflicting duplicate"):
        registry.assert_consistent()
    registry.close()


def test_disk_registry_normalizes_integral_float_dtype_drift(tmp_path: Path) -> None:
    registry = _DiskRegistry(tmp_path / "registry.sqlite")
    registry.add_frame(
        "molecules",
        "molecule_id",
        pd.DataFrame([{"molecule_id": "M1", "fragment_count": 1}]),
    )
    registry.add_frame(
        "molecules",
        "molecule_id",
        pd.DataFrame([{"molecule_id": "M1", "fragment_count": 1.0}]),
    )
    registry.assert_consistent()
    registry.add_frame(
        "molecules",
        "molecule_id",
        pd.DataFrame([{"molecule_id": "M1", "fragment_count": 1.5}]),
    )
    with pytest.raises(RuntimeError, match="Conflicting duplicate"):
        registry.assert_consistent()
    registry.close()


def test_bulk_components_are_stable_and_complex_sequences_are_not_concatenated() -> None:
    components = pd.DataFrame(
        [
            {
                "target_chembl_id": "T-COMPLEX",
                "target_name": "A/B complex",
                "target_type": "PROTEIN COMPLEX",
                "target_organism": "Homo sapiens",
                "component_id": 2,
                "accession": "P2",
                "sequence": "BBBB",
                "component_type": "PROTEIN",
                "component_organism": "Homo sapiens",
            },
            {
                "target_chembl_id": "T-SINGLE",
                "target_name": "Single",
                "target_type": "SINGLE PROTEIN",
                "target_organism": "Homo sapiens",
                "component_id": 3,
                "accession": "P3",
                "sequence": "CCCC",
                "component_type": "PROTEIN",
                "component_organism": "Homo sapiens",
            },
            {
                "target_chembl_id": "T-COMPLEX",
                "target_name": "A/B complex",
                "target_type": "PROTEIN COMPLEX",
                "target_organism": "Homo sapiens",
                "component_id": 1,
                "accession": "P1",
                "sequence": "AAAA",
                "component_type": "PROTEIN",
                "component_organism": "Homo sapiens",
            },
        ]
    )
    metadata = _component_metadata(components.sample(frac=1.0, random_state=7))
    rows = pd.DataFrame(
        [
            {
                "_source_target_key": "T-COMPLEX",
                "target_type": "PROTEIN COMPLEX",
                "target_pref_name": "A/B complex",
                "target_organism": "Homo sapiens",
            },
            {
                "_source_target_key": "T-SINGLE",
                "target_type": "SINGLE PROTEIN",
                "target_pref_name": "Single",
                "target_organism": "Homo sapiens",
            },
        ]
    )
    proteins, target_map, constructs = _bulk_protein_entities(rows, metadata)
    single = proteins.set_index("protein_id").loc[target_map["T-SINGLE"]]
    complex_parent = proteins.set_index("protein_id").loc[target_map["T-COMPLEX"]]
    assert single["sequence"] == "CCCC"
    assert single["uniprot_accession"] == "P3"
    assert complex_parent["sequence"] == ""
    assert len(complex_parent["component_protein_ids"].split(";")) == 2
    complex_constructs = constructs[constructs["parent_target_id"].eq("T-COMPLEX")]
    assert complex_constructs.sort_values("component_order")["component_accession"].tolist() == [
        "P1",
        "P2",
    ]

    joined = _join_model_inputs(
        pd.DataFrame([{"molecule_id": "M1", "protein_id": target_map["T-SINGLE"], "assay_id": "A1"}]),
        pd.DataFrame(
            [
                {
                    "molecule_id": "M1",
                    "standardized_smiles": "CC",
                    "canonical_smiles": "CC",
                    "standard_inchi_key": "KEY",
                    "structure_id": "S1",
                }
            ]
        ),
        proteins,
        pd.DataFrame([{"assay_id": "A1", "description": "", "matrix": "", "route": ""}]),
        pd.DataFrame(),
    )
    assert joined.loc[0, "sequence"] == "CCCC"


def test_snapshot_selection_is_exact_and_derivation_digest_is_recomputed(tmp_path: Path) -> None:
    files = pd.DataFrame(
        [
            {
                "source_file_id": "DB",
                "source_id": CHEMBL_SOURCE_ID,
                "snapshot_id": "BULK",
                "relative_path": "chembl/extracted/chembl_37.db",
            },
            {
                "source_file_id": "PANEL",
                "source_id": CHEMBL_SOURCE_ID,
                "snapshot_id": "PANEL-SNAPSHOT",
                "relative_path": "panel/file.json",
            },
        ]
    )
    selected, database_id = _select_bulk_source_files(files, "BULK")
    assert database_id == "DB"
    assert set(selected["snapshot_id"]) == {"BULK"}

    source = pd.DataFrame(
        [
            {
                "observation_id": "OBS1",
                "source_id": CHEMBL_SOURCE_ID,
                "snapshot_id": "BULK",
                "source_record_id": "ChEMBL:activity:1",
                "molecule_id": "M1",
                "protein_id": "P1",
                "assay_id": "A1",
                "endpoint": "Kd",
                "relation": "=",
                "canonical_unit": "nM",
                "canonical_value": 10.0,
                "inclusion_status": "included",
                "observation_kind": "experimental_summary",
            }
        ]
    )
    assays = pd.DataFrame([{"assay_id": "A1", "temperature_c": float("nan")}])
    proteins = pd.DataFrame([{"protein_id": "P1", "entity_type": "single_protein"}])
    derivations, derived = binding_free_energy_view(source, assays, proteins)
    _validate_derivation_bundle(source, derivations, derived)
    tampered = derivations.copy()
    tampered.loc[0, "label_lineage_digest"] = "0" * 64
    with pytest.raises(RuntimeError, match="digest mismatch"):
        _validate_derivation_bundle(source, tampered, derived)


def test_task_registry_accumulator_reconciles_cross_shard_counts() -> None:
    base = {
        "task_id": "TASK1",
        "task_type": "default__binding__kd__binding__nm__continuous_exact",
        "evidence_domain": "binding",
        "endpoint": "Kd",
        "assay_family": "binding",
        "label_kind": "continuous_exact",
        "label_unit": "nM",
        "observation_kind": "experimental_summary",
        "default_task_eligible": True,
        "sensitivity_task_eligible": False,
        "required_modalities": "small_molecule_structure;protein_sequence",
        "label_relation": "=",
    }
    accumulator = _TaskRegistryAccumulator()
    accumulator.add(pd.DataFrame([{**base, "observation_id": "O1"}]))
    accumulator.add(pd.DataFrame([{**base, "observation_id": "O2"}]))
    registry = accumulator.frame()
    assert int(registry.loc[0, "row_count"]) == 2
    assert json_load(registry.loc[0, "relation_counts_json"]) == {"=": 2}


def test_partition_normalization_stabilizes_null_only_and_numeric_shards(
    tmp_path: Path,
) -> None:
    building = tmp_path / ".full_chembl37.building"
    directory = building / "observations"
    directory.mkdir(parents=True)
    first = directory / "part-00000.parquet"
    second = directory / "part-00001.parquet"
    pd.DataFrame({"observation_id": ["A"], "canonical_value": [None]}).to_parquet(first, index=False)
    pd.DataFrame({"observation_id": ["B"], "canonical_value": [2.5]}).to_parquet(second, index=False)
    records = [
        {
            "relative_path": path.relative_to(building).as_posix(),
            "path": path.name,
            "rows": 1,
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in (first, second)
    ]
    contract = _normalize_partition_records(building, records)
    schemas = [pq.ParquetFile(path).schema_arrow.remove_metadata() for path in (first, second)]
    assert schemas[0].equals(schemas[1], check_metadata=True)
    assert str(schemas[0].field("observation_id").type) == "large_string"
    assert str(schemas[0].field("canonical_value").type) == "double"
    assert arrow_schema_contract(schemas[0]) == contract
    assert all(record["arrow_schema_sha256"] == contract["sha256"] for record in records)
    assert pq.read_table(first)["canonical_value"].null_count == 1
    assert pq.read_table(second)["canonical_value"].to_pylist() == [2.5]


def test_partition_normalization_stabilizes_null_double_and_string_shards(
    tmp_path: Path,
) -> None:
    building = tmp_path / ".full_chembl37.building"
    directory = building / "derived_observations"
    directory.mkdir(parents=True)
    first = directory / "part-00006.parquet"
    second = directory / "part-00007.parquet"
    first_ids = [f"D{index:02d}" for index in range(45)]
    second_ids = ["E00", "E01", "E02"]
    pq.write_table(
        pa.table(
            {
                "observation_id": pa.array(first_ids, type=pa.string()),
                "evidence_stage": pa.array([None] * 45, type=pa.float64()),
            }
        ),
        first,
    )
    pq.write_table(
        pa.table(
            {
                "observation_id": pa.array(second_ids, type=pa.large_string()),
                "evidence_stage": pa.array(
                    ["preclinical_in_vitro", "clinical", "reported"],
                    type=pa.large_string(),
                ),
                "endpoint": pa.array(["Kd", "Ki", "IC50"], type=pa.string()),
            }
        ),
        second,
    )
    records = [
        {
            "relative_path": path.relative_to(building).as_posix(),
            "path": path.name,
            "rows": pq.ParquetFile(path).metadata.num_rows,
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in (first, second)
    ]

    contract = _normalize_partition_records(building, records)

    schemas = [pq.ParquetFile(path).schema_arrow.remove_metadata() for path in (first, second)]
    assert schemas[0].equals(schemas[1], check_metadata=True)
    assert str(schemas[0].field("evidence_stage").type) == "large_string"
    assert arrow_schema_contract(schemas[0]) == contract
    first_table = pq.read_table(first)
    second_table = pq.read_table(second)
    assert first_table["observation_id"].to_pylist() == first_ids
    assert second_table["observation_id"].to_pylist() == second_ids
    assert first_table["evidence_stage"].to_pylist() == [None] * 45
    assert first_table["endpoint"].to_pylist() == [None] * 45
    assert second_table["evidence_stage"].to_pylist() == [
        "preclinical_in_vitro",
        "clinical",
        "reported",
    ]
    for record, path, expected_rows in zip(records, (first, second), (45, 3), strict=True):
        assert record["rows"] == expected_rows
        assert record["size_bytes"] == path.stat().st_size
        assert record["sha256"] == sha256_file(path)
        assert record["arrow_schema_sha256"] == contract["sha256"]


def test_partition_normalization_rejects_non_null_numeric_string_field(
    tmp_path: Path,
) -> None:
    building = tmp_path / ".full_chembl37.building"
    path = building / "observations" / "part-00000.parquet"
    path.parent.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "observation_id": pa.array(["A"], type=pa.string()),
                "evidence_stage": pa.array([1.5], type=pa.float64()),
            }
        ),
        path,
    )
    records = [
        {
            "relative_path": path.relative_to(building).as_posix(),
            "path": path.name,
            "rows": 1,
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    ]

    with pytest.raises(RuntimeError, match="normative Arrow type"):
        _normalize_partition_records(building, records)


def test_partition_normalization_does_not_rewrite_before_all_parts_validate(
    tmp_path: Path,
) -> None:
    building = tmp_path / ".full_chembl37.building"
    directory = building / "observations"
    directory.mkdir(parents=True)
    first = directory / "part-00000.parquet"
    second = directory / "part-00001.parquet"
    pq.write_table(
        pa.table(
            {
                "observation_id": pa.array(["A"], type=pa.string()),
                "evidence_stage": pa.array([None], type=pa.float64()),
            }
        ),
        first,
    )
    pq.write_table(
        pa.table(
            {
                "observation_id": pa.array(["B"], type=pa.string()),
                "evidence_stage": pa.array([1.5], type=pa.float64()),
            }
        ),
        second,
    )
    first_before = first.read_bytes()
    records = [
        {
            "relative_path": path.relative_to(building).as_posix(),
            "path": path.name,
            "rows": 1,
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in (first, second)
    ]

    with pytest.raises(RuntimeError, match="normative Arrow type"):
        _normalize_partition_records(building, records)

    assert first.read_bytes() == first_before
    assert not list(building.rglob("*.schema.part"))


def test_partition_normalization_rejects_symlinks_and_hardlinks(
    tmp_path: Path,
) -> None:
    building = tmp_path / ".full_chembl37.building"
    directory = building / "observations"
    directory.mkdir(parents=True)
    target = directory / "target.parquet"
    pd.DataFrame({"observation_id": ["A"]}).to_parquet(target, index=False)
    target_before = target.read_bytes()
    link = directory / "part-00000.parquet"
    link.symlink_to(target.name)
    record = {
        "relative_path": link.relative_to(building).as_posix(),
        "path": link.name,
        "rows": 1,
        "sha256": sha256_file(link),
        "size_bytes": link.stat().st_size,
    }

    with pytest.raises(RuntimeError, match="symlink"):
        _normalize_partition_records(building, [record])

    assert link.is_symlink()
    assert target.read_bytes() == target_before
    link.unlink()
    os.link(target, link)
    with pytest.raises(RuntimeError, match="hard-linked"):
        _normalize_partition_records(building, [record])


@pytest.mark.parametrize(
    "relative_path",
    (
        "../outside.parquet",
        "observations/../outside.parquet",
        "/outside.parquet",
        " observations/part-00000.parquet ",
        "observations/part-00000.parquet\n",
        "observations\\part-00000.parquet",
    ),
)
def test_partition_normalization_rejects_noncanonical_paths(
    tmp_path: Path,
    relative_path: str,
) -> None:
    building = tmp_path / ".full_chembl37.building"
    building.mkdir()
    outside = tmp_path / "outside.parquet"
    pd.DataFrame({"observation_id": ["A"]}).to_parquet(outside, index=False)
    record = {
        "relative_path": relative_path,
        "path": "outside.parquet",
        "rows": 1,
        "sha256": sha256_file(outside),
        "size_bytes": outside.stat().st_size,
    }

    with pytest.raises(RuntimeError, match="Non-canonical partition path"):
        _normalize_partition_records(building, [record])


def test_partition_normalization_rejects_dangling_staging_symlink(
    tmp_path: Path,
) -> None:
    building = tmp_path / ".full_chembl37.building"
    path = building / "observations" / "part-00000.parquet"
    path.parent.mkdir(parents=True)
    pd.DataFrame({"observation_id": ["A"]}).to_parquet(path, index=False)
    before = path.read_bytes()
    outside = tmp_path / "outside.parquet"
    temporary = path.with_suffix(".parquet.schema.part")
    temporary.symlink_to(outside)
    record = {
        "relative_path": path.relative_to(building).as_posix(),
        "path": path.name,
        "rows": 1,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }

    with pytest.raises(RuntimeError, match="existing schema-normalization staging"):
        _normalize_partition_records(building, [record])

    assert temporary.is_symlink()
    assert not outside.exists()
    assert path.read_bytes() == before


def test_interrupted_partition_commit_remains_private_and_unreusable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical_root = tmp_path / "canonical"
    building = canonical_root / ".full_chembl37.building"
    destination = canonical_root / "full_chembl37"
    reports = tmp_path / "reports"
    directory = building / "observations"
    directory.mkdir(parents=True)
    paths = [directory / f"part-{index:05d}.parquet" for index in range(2)]
    for index, path in enumerate(paths):
        pq.write_table(
            pa.table(
                {
                    "observation_id": pa.array([f"O{index}"], type=pa.string()),
                    "evidence_stage": pa.array([None], type=pa.float64()),
                }
            ),
            path,
        )
    records = [
        {
            "relative_path": path.relative_to(building).as_posix(),
            "path": path.name,
            "rows": 1,
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in paths
    ]
    real_replace = os.replace
    commit_calls = 0

    def fail_second_commit(source: str | os.PathLike[str], target: str | os.PathLike[str]) -> None:
        nonlocal commit_calls
        if str(source).endswith(".parquet.schema.part"):
            commit_calls += 1
            if commit_calls == 2:
                raise OSError("injected second partition commit failure")
        real_replace(source, target)

    with monkeypatch.context() as patch:
        patch.setattr(os, "replace", fail_second_commit)
        with pytest.raises(OSError, match="injected second partition commit failure"):
            _normalize_partition_records(building, records)

    assert commit_calls == 2
    assert building.is_dir()
    assert not destination.exists()
    assert not (building / "build_manifest.json").exists()
    assert not reports.exists()
    with pytest.raises(RuntimeError, match="Incomplete bulk canonical build exists"):
        materialize_chembl37_specialized_canonical(
            tmp_path / "raw",
            tmp_path / "interim",
            canonical_root,
            reports,
        )


def test_partition_normalization_rejects_semantically_wrong_integer_values(
    tmp_path: Path,
) -> None:
    building = tmp_path / ".full_chembl37.building"
    path = building / "observations" / "part-00000.parquet"
    path.parent.mkdir(parents=True)
    pd.DataFrame({"observation_id": ["A"], "document_year": [2024.5]}).to_parquet(path, index=False)
    records = [
        {
            "relative_path": path.relative_to(building).as_posix(),
            "path": path.name,
            "rows": 1,
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    ]
    with pytest.raises(RuntimeError, match="normative Arrow type"):
        _normalize_partition_records(building, records)


def test_root_singleton_binds_normative_task_registry_schema(tmp_path: Path) -> None:
    building = tmp_path / ".full_chembl37.building"
    building.mkdir()
    frame = pd.DataFrame(
        {
            "task_id": ["TASK1"],
            "row_count": [2],
            "relation_counts_json": ['{"=":2}'],
            "policy_version": ["platform-task-contract-v1"],
            "intended_use": ["modeling"],
            "prohibited_claim": ["clinical outcome"],
        }
    )
    path = building / "task_registry.parquet"
    record = _write_normative_singleton(frame, path, building)
    physical = pq.ParquetFile(path).schema_arrow.remove_metadata()
    assert record["arrow_schema"] == arrow_schema_contract(physical)
    assert record["arrow_schema_sha256"] == record["arrow_schema"]["sha256"]
    assert record["relative_path"] == "task_registry.parquet"
    assert str(physical.field("relation_counts_json").type) == "large_string"
    assert str(physical.field("row_count").type) == "int64"


def test_model_readiness_partition_is_reasoned_and_modality_aware() -> None:
    base = {
        "task_id": "TASK1",
        "task_type": "default__binding__ic50",
        "source_id": "SRC1",
        "snapshot_id": "SNP1",
        "source_record_id": "REC1",
        "molecule_id": "M1",
        "protein_id": "P1",
        "assay_id": "A1",
        "canonical_target_id": "CHEMBL1",
        "required_modalities": "small_molecule_structure;protein_sequence",
        "standardized_smiles": "CC",
        "standard_inchi_key": "OTMSDBZUPAUEDD-UHFFFAOYSA-N",
        "sequence": "AAAA",
    }
    candidates = pd.DataFrame(
        [
            {**base, "observation_id": "READY"},
            {**base, "observation_id": "NO_SMILES", "standardized_smiles": ""},
            {**base, "observation_id": "NO_INCHI", "standard_inchi_key": ""},
            {**base, "observation_id": "NO_SEQUENCE", "sequence": ""},
            {
                **base,
                "observation_id": "LIGAND_ONLY",
                "required_modalities": "small_molecule_structure",
                "sequence": "",
            },
        ]
    )
    eligible, exclusions = _partition_model_ready_tasks(
        candidates,
        task_scope="default",
    )
    assert set(eligible["observation_id"]) == {"READY", "LIGAND_ONLY"}
    reasons = exclusions.set_index("observation_id")["model_readiness_exclusion_reason"].to_dict()
    assert reasons == {
        "NO_SMILES": "missing_standardized_smiles",
        "NO_INCHI": "missing_standard_inchi_key",
        "NO_SEQUENCE": "missing_protein_sequence",
    }
    assert exclusions["task_scope"].eq("default").all()


def json_load(value: object) -> dict[str, int]:
    return json.loads(str(value))


@pytest.mark.parametrize("drift_location", ["standalone", "summary"])
def test_specialized_loader_rejects_summary_child_manifest_drift(
    tmp_path: Path,
    drift_location: str,
) -> None:
    interim, root = _write_specialized_contract(tmp_path)
    records, summary, target_path, development = _load_input_parts(interim)
    assert len(records) == len(_SPECIALIZED_VIEWS)
    assert summary["view_row_counts"] == {name: 1 for name in _SPECIALIZED_VIEWS}
    assert target_path.name == "target_components.parquet"
    assert development["row_count"] == 1

    if drift_location == "standalone":
        path = root / "single_protein_kd_ki_manifest.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["scientific_boundary"] = "tampered"
    else:
        path = root / "specialized_views_manifest.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["views"]["single_protein_kd_ki"]["scientific_boundary"] = "tampered"
    _write_json(path, payload)
    with pytest.raises(RuntimeError, match="summary/standalone child manifest drift"):
        _load_input_parts(interim)


def test_failed_qc_never_promotes_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    building = tmp_path / ".full_chembl37.building"
    destination = tmp_path / "full_chembl37"
    reports = tmp_path / "reports"
    building.mkdir()
    pd.DataFrame({"value": [1]}).to_csv(building / "data_dictionary.csv", index=False)
    manifest = {
        "build_type": "public_chembl37_full_specialized_canonical",
        "component_inventory": _component_inventory(building),
    }
    _write_json(building / "build_manifest.json", manifest)

    from menin_discovery import platform_data_qc

    def fail_qc(_canonical: Path, _reports: Path) -> dict[str, object]:
        raise RuntimeError("injected QC failure")

    monkeypatch.setattr(platform_data_qc, "run_platform_qc", fail_qc)
    with pytest.raises(RuntimeError, match="injected QC failure"):
        _promote_qc_accepted_build(building, destination, reports)
    assert building.is_dir()
    assert not destination.exists()


def test_existing_destination_requires_fresh_manifest_bound_qc(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical_root = tmp_path / "canonical"
    destination = canonical_root / "full_chembl37"
    destination.mkdir(parents=True)
    pd.DataFrame({"value": [1]}).to_csv(destination / "data_dictionary.csv", index=False)
    manifest = {
        "build_type": "public_chembl37_full_specialized_canonical",
        "component_inventory": _component_inventory(destination),
    }
    _write_json(destination / "build_manifest.json", manifest)

    from menin_discovery import platform_data_qc

    calls: list[Path] = []

    def accept_qc(canonical: Path, _reports: Path) -> dict[str, object]:
        calls.append(Path(canonical))
        return {
            "qc_passed": True,
            "build_manifest_sha256": sha256_file(Path(canonical) / "build_manifest.json"),
        }

    monkeypatch.setattr(platform_data_qc, "run_platform_qc", accept_qc)
    returned = materialize_chembl37_specialized_canonical(
        tmp_path / "raw",
        tmp_path / "interim",
        canonical_root,
        tmp_path / "reports",
    )
    assert returned == manifest
    assert calls == [destination.resolve()]
