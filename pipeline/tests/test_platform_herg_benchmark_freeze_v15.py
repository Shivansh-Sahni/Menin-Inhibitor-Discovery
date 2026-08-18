from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from menin_discovery.platform_herg_benchmark_freeze_v15 import (
    BLOCKER_EVIDENCE_NAME,
    CANONICAL_CONTEXT_COLUMNS,
    EXACT_TASK_COLUMNS,
    FORBIDDEN_READ_COLUMNS,
    MEMBERSHIP_COLUMNS,
    MEMBERSHIP_NAME,
    OBSERVATION_COLUMNS,
    REGISTRY_NAME,
    STRUCTURE_COLUMNS,
    TASK_COLUMNS,
    FreezeV15Config,
    HergBenchmarkFreezeV15Error,
    _guard_projection,
    _purge_cross_partition_identities,
    _temporal_boundaries,
    build_benchmark_freeze_v15,
    validate_benchmark_freeze_v15,
)


def _write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), path)


def _synthetic_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    master_root = tmp_path / "master"
    canonical_root = tmp_path / "canonical"
    exact_task_path = canonical_root / "tasks" / "exact_herg" / "part-00000.parquet"
    master_root.mkdir(parents=True)
    canonical_root.mkdir(parents=True)
    (master_root / "herg_master_manifest.json").write_text('{"release":"synthetic"}\n')
    (canonical_root / "build_manifest.json").write_text('{"release":"synthetic"}\n')

    smiles = [
        "CC",
        "CCC",
        "CCCC",
        "CCN",
        "CCO",
        "CCCl",
        "CCBr",
        "c1ccccc1",
        "c1ccncc1",
        "C1CCCCC1",
        "CC(=O)O",
        "CC#N",
        "COc1ccccc1",
        "CCc1ccccc1",
        "O=C(O)c1ccccc1",
        "CN(C)C",
        "CC(C)C",
        "NCCO",
    ]
    q2_rows: list[dict[str, object]] = []
    observation_rows: list[dict[str, object]] = []
    structure_rows: list[dict[str, object]] = []
    context_rows: list[dict[str, object]] = []
    for index, structure_smiles in enumerate(smiles):
        if index < 10:
            split = "train"
        elif index < 14:
            split = "validation"
        else:
            split = "test"
        observation_id = f"HOBS-{index:04d}"
        structure_id = f"HSTR-{index:04d}"
        activity_id = 10_000 + index
        q2_rows.append(
            {
                "membership_id": f"MEM-Q2-{index:04d}",
                "task_id": "Q2_FUNCTIONAL_ASSAY_AWARE",
                "quality_level": "Q2",
                "source_artifact": "synthetic_q2.parquet",
                "record_id": f"REC-Q2-{index:04d}",
                "observation_id": observation_id,
                "structure_id": structure_id,
                "target_scope": "wild_type_or_unspecified",
                "source_family": "chembl_herg_specialized_view",
                "measurement_technology": (
                    "automated_patch_clamp" if split == "test" else "functional_technology_unspecified"
                ),
                "model_split": split,
                "scaffold_group_id": f"HSCF-{index:04d}",
                "eligible": True,
                "use_as_training_label": True,
                "clinical_context_only": False,
            }
        )
        observation_rows.append(
            {
                "observation_id": observation_id,
                "source_family": "chembl_herg_specialized_view",
                "source_record_id": f"ACTIVITY:{activity_id}",
            }
        )
        structure_rows.append(
            {
                "structure_id": structure_id,
                "standardized_smiles": structure_smiles,
                "model_split": split,
                "scaffold_group_id": f"HSCF-{index:04d}",
            }
        )
        context_rows.append(
            {
                "source_record_id": f"ChEMBL:activity:{activity_id}",
                "observation_id": f"OBS-{index:04d}",
                "assay_id": f"ASSAY-{index % 6}",
                "document_id": f"DOC-{index % 6}",
                "document_year": 2000 + index // 5,
            }
        )

    q1_rows: list[dict[str, object]] = []
    for index in range(6):
        q1_rows.append(
            {
                "membership_id": f"MEM-Q1-{index:04d}",
                "task_id": "Q1_QUANTITATIVE_PIC50",
                "quality_level": "Q1",
                "source_artifact": "synthetic_q1.parquet",
                "record_id": f"REC-Q1-{index:04d}",
                "observation_id": f"Q1OBS-{index:04d}",
                "structure_id": f"Q1STR-{index:04d}",
                "target_scope": "wild_type_or_unspecified",
                "source_family": (
                    "quantitative_pic50_release" if index < 4 else "chembl_herg_specialized_view"
                ),
                "measurement_technology": "functional_technology_unspecified",
                "model_split": "train" if index < 3 else ("validation" if index == 3 else "test"),
                "scaffold_group_id": f"Q1SCF-{index:04d}",
                "eligible": True,
                "use_as_training_label": True,
                "clinical_context_only": False,
            }
        )

    tasks = pd.DataFrame(q2_rows + q1_rows, columns=TASK_COLUMNS)
    observations = pd.DataFrame(observation_rows, columns=OBSERVATION_COLUMNS)
    structures = pd.DataFrame(structure_rows, columns=STRUCTURE_COLUMNS)
    context = pd.DataFrame(context_rows, columns=CANONICAL_CONTEXT_COLUMNS)
    exact = pd.DataFrame(
        {"source_record_id": [row["source_record_id"] for row in context_rows]},
        columns=EXACT_TASK_COLUMNS,
    )
    _write_parquet(master_root / "task_membership.parquet", tasks)
    _write_parquet(master_root / "observation_master.parquet", observations)
    _write_parquet(master_root / "structure_master.parquet", structures)
    _write_parquet(canonical_root / "observations" / "part-00000.parquet", context)
    _write_parquet(exact_task_path, exact)
    return master_root, canonical_root, exact_task_path


def test_projection_contract_excludes_every_forbidden_outcome_column() -> None:
    projections = (
        TASK_COLUMNS,
        OBSERVATION_COLUMNS,
        STRUCTURE_COLUMNS,
        CANONICAL_CONTEXT_COLUMNS,
        EXACT_TASK_COLUMNS,
    )
    assert all(not (set(columns) & FORBIDDEN_READ_COLUMNS) for columns in projections)
    with pytest.raises(HergBenchmarkFreezeV15Error, match="forbidden outcome"):
        _guard_projection(("structure_id", "target_class"), "synthetic")


def test_purge_removes_full_structure_and_scaffold_crossings() -> None:
    source = pd.DataFrame(
        {
            "challenge_split": ["train", "test", "train", "validation", "test"],
            "structure_id": ["s1", "s1", "s2", "s3", "s4"],
            "scaffold_group_id": ["g1", "g1", "g2", "g2", "g4"],
        }
    )
    kept, evidence = _purge_cross_partition_identities(source)
    assert kept[["structure_id", "scaffold_group_id"]].to_dict("records") == [
        {"structure_id": "s4", "scaffold_group_id": "g4"}
    ]
    assert evidence["cross_partition_structures"] == 1
    assert evidence["cross_partition_scaffolds"] == 2


def test_temporal_boundaries_reserve_a_whole_year_for_each_partition() -> None:
    counts = pd.Series([60, 15, 10, 15], index=[2000, 2001, 2002, 2003])
    train_end, validation_end = _temporal_boundaries(counts)
    assert train_end < validation_end < 2003


def test_build_and_full_source_replay_are_label_blind(tmp_path: Path) -> None:
    master_root, canonical_root, exact_task_path = _synthetic_inputs(tmp_path)
    output_root = tmp_path / "v1_5"
    manifest = build_benchmark_freeze_v15(
        master_root=master_root,
        canonical_root=canonical_root,
        exact_task_path=exact_task_path,
        output_root=output_root,
        config=FreezeV15Config(minimum_holdout_structures=3),
    )
    replayed = validate_benchmark_freeze_v15(output_root)
    assert replayed["manifest_sha256"] == manifest["manifest_sha256"]
    assert manifest["scientific_contract"]["target_values_read"] is False
    assert manifest["scientific_contract"]["target_classes_read"] is False
    assert manifest["scientific_contract"]["test_labels_opened"] is False

    memberships = pd.read_parquet(output_root / MEMBERSHIP_NAME)
    assert tuple(memberships.columns) == MEMBERSHIP_COLUMNS
    assert not set(memberships.columns) & FORBIDDEN_READ_COLUMNS
    registry = pd.read_parquet(output_root / REGISTRY_NAME)
    assert registry["status"].eq("materialized_label_blind_membership").sum() == 7
    blockers = pd.read_parquet(output_root / BLOCKER_EVIDENCE_NAME)
    assert blockers["challenge_id"].nunique() == 6
    assert json.loads((output_root / "benchmark_freeze_v1_5_manifest.json").read_text())[
        "projection_contract"
    ]["forbidden_read_columns"]
