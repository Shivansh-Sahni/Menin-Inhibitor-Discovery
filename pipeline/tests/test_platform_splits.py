from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest
from menin_discovery.platform_splits import (
    SplitConfig,
    audit_split_leakage,
    default_split_suite,
    make_split_manifest,
    resolve_manifest_bound_parquet_dataset,
    stream_hash_group_split_manifest,
)


def _task_frame(n: int = 24) -> pd.DataFrame:
    smiles_bank = [
        "CCO",
        "CCN",
        "CCC",
        "CCCl",
        "CCBr",
        "CC(=O)O",
        "c1ccccc1",
        "c1ccncc1",
        "C1CCCCC1",
        "CCOC",
        "CCNC",
        "COC",
        "CNC",
        "CCS",
        "CCF",
        "O=C=O",
        "N#N",
        "CC#N",
        "CC=C",
        "C=CCO",
        "CC(C)O",
        "CC(C)N",
        "C1CCNCC1",
        "c1ccoc1",
    ]
    return pd.DataFrame(
        {
            "observation_id": [f"obs-{index:03d}" for index in range(n)],
            "molecule_id": [f"mol-{index:03d}" for index in range(n)],
            "standardized_smiles": smiles_bank[:n],
            "protein_id": [f"protein-{index % 6}" for index in range(n)],
            "canonical_target_id": [f"target-{index % 6}" for index in range(n)],
            "source_id": [f"source-{index % 4}" for index in range(n)],
            "document_year": [2000 + index % 12 for index in range(n)],
            "task_id": ["herg_binary"] * n,
            "task_type": ["classification"] * n,
            "label_value": [index % 2 for index in range(n)],
            "label_relation": ["="] * n,
            "label_unit": ["binary"] * n,
            "observation_kind": ["curated_assertion"] * n,
            "sequence": ["ACDEFGHIKLMNPQRSTVWY" + chr(65 + index % 3) for index in range(n)],
        }
    )


def _streaming_task_frame(n: int = 240) -> pd.DataFrame:
    base = _task_frame(24)
    frame = pd.concat([base] * ((n + len(base) - 1) // len(base)), ignore_index=True).iloc[:n].copy()
    frame["observation_id"] = [f"stream-{index:06d}" for index in range(n)]
    frame["snapshot_id"] = "chembl-37"
    frame["assay_id"] = [f"assay-{index % 8}" for index in range(n)]
    frame["source_record_id"] = [f"chembl-record-{index:06d}" for index in range(n)]
    frame["label_kind"] = "categorical"
    frame["label_text"] = ""
    frame["label_lower_bound"] = None
    frame["label_upper_bound"] = None
    frame["label_unit"] = "binary"
    frame["access_class"] = "public_redistributable"
    frame["inclusion_status"] = "included"
    frame["default_task_eligible"] = True
    frame["evidence_domain"] = "herg"
    frame["endpoint"] = "herg_blocker_class"
    frame["assay_family"] = "herg_functional"
    return frame


def _manifest_bound_task_directory(
    root: Path,
    frame: pd.DataFrame,
    *,
    split_at: int | None = None,
) -> Path:
    build_root = root / "full_chembl37"
    task_directory = build_root / "tasks" / "default" / "classification"
    task_directory.mkdir(parents=True)
    split_at = split_at or len(frame) // 2
    parts = [frame.iloc[:split_at].copy(), frame.iloc[split_at:].copy()]
    artifacts: list[dict[str, object]] = []
    for index, part_frame in enumerate(parts):
        path = task_directory / f"part-{index:05d}.parquet"
        part_frame.to_parquet(path, index=False)
        artifacts.append(
            {
                "path": path.name,
                "relative_path": path.relative_to(build_root).as_posix(),
                "rows": len(part_frame),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "size_bytes": path.stat().st_size,
            }
        )
    (build_root / "build_manifest.json").write_text(
        json.dumps({"schema_version": "test", "shard_artifacts": list(reversed(artifacts))}),
        encoding="utf-8",
    )
    return task_directory


def test_molecule_split_is_deterministic_under_row_reordering_and_exact_safe() -> None:
    frame = _task_frame()
    config = SplitConfig(
        name="molecule_test",
        strategy="molecule_grouped",
        intended_use="unseen molecule",
        task_type="classification",
    )
    first, first_meta = make_split_manifest(frame, config)
    shuffled, second_meta = make_split_manifest(frame.sample(frac=1, random_state=9), config)
    pd.testing.assert_frame_equal(first, shuffled)
    assert first_meta["manifest_sha256"] == second_meta["manifest_sha256"]

    audit, _ = audit_split_leakage(
        frame,
        first,
        config,
        run_chemical_near_duplicate_audit=False,
        run_protein_near_duplicate_audit=False,
    )
    assert audit["exact_leakage_gate_passed"]
    assert audit["exact_overlap"]["molecule"]["n_cross_partition"] == 0


def test_prediction_observation_kind_fails_and_kind_changes_dataset_hash() -> None:
    frame = _task_frame()
    config = SplitConfig(
        name="origin_test",
        strategy="molecule_grouped",
        intended_use="unseen molecule",
        task_type="classification",
    )
    manifest, metadata = make_split_manifest(frame, config)
    alternate_valid_kind = frame.copy()
    alternate_valid_kind.loc[0, "observation_kind"] = "experimental_summary"
    _, alternate_metadata = make_split_manifest(alternate_valid_kind, config)
    assert metadata["dataset_sha256"] != alternate_metadata["dataset_sha256"]

    contaminated = frame.copy()
    contaminated.loc[0, "observation_kind"] = "prediction"
    with pytest.raises(ValueError, match="Prediction rows"):
        make_split_manifest(contaminated, config)

    audit, _ = audit_split_leakage(
        contaminated,
        manifest,
        config,
        run_chemical_near_duplicate_audit=False,
        run_protein_near_duplicate_audit=False,
    )
    assert not audit["exact_leakage_gate_passed"]
    assert "prediction" in audit["prohibited_label_origins"]


def test_missing_observation_kind_fails_closed() -> None:
    frame = _task_frame().drop(columns="observation_kind")
    config = SplitConfig(name="missing_origin", strategy="molecule_grouped", intended_use="test")
    with pytest.raises(ValueError, match="observation_kind"):
        make_split_manifest(frame, config)


def test_derived_labels_require_explicit_policy_and_lineage() -> None:
    frame = _task_frame()
    frame.loc[0, "observation_kind"] = "derived"
    config = SplitConfig(name="derived", strategy="molecule_grouped", intended_use="test")
    with pytest.raises(ValueError, match="Derived labels"):
        make_split_manifest(frame, config)

    frame["label_lineage_digest"] = "a" * 64
    allowed = SplitConfig(
        name="derived_allowed",
        strategy="molecule_grouped",
        intended_use="lineage-backed derived sensitivity task",
        allow_derived_labels=True,
    )
    allowed_manifest, _ = make_split_manifest(frame, allowed)
    allowed_audit, _ = audit_split_leakage(
        frame,
        allowed_manifest,
        allowed,
        run_chemical_near_duplicate_audit=False,
        run_protein_near_duplicate_audit=False,
    )
    assert allowed_audit["exact_leakage_gate_passed"]


def test_blank_and_unknown_observation_kinds_fail_closed() -> None:
    for value in ("", "mystery"):
        frame = _task_frame()
        frame.loc[0, "observation_kind"] = value
        config = SplitConfig(name="invalid_kind", strategy="molecule_grouped", intended_use="test")
        with pytest.raises(ValueError, match="canonical nonblank"):
            make_split_manifest(frame, config)


def test_temporal_split_excludes_unknown_dates() -> None:
    frame = _task_frame()
    frame.loc[[2, 7], "document_year"] = None
    config = SplitConfig(name="temporal", strategy="temporal", intended_use="future molecules")
    manifest, metadata = make_split_manifest(frame, config)
    unknown_ids = set(frame.loc[[2, 7], "observation_id"])
    observed_unknown = manifest.set_index("record_id").loc[list(unknown_ids), "split"]
    assert set(observed_unknown) == {"excluded_unknown_date"}
    assert metadata["strategy_metadata"]["unknown_date_rows_excluded"] == 2


@pytest.mark.parametrize(
    ("strategy", "exclusive_family"),
    [("source_holdout", "source"), ("protein_holdout", "protein")],
)
def test_entity_holdout_strategies_are_exclusive(strategy: str, exclusive_family: str) -> None:
    frame = _task_frame()
    config = SplitConfig(name=strategy, strategy=strategy, intended_use="entity holdout")
    manifest, _ = make_split_manifest(frame, config)
    audit, _ = audit_split_leakage(
        frame,
        manifest,
        config,
        run_chemical_near_duplicate_audit=False,
        run_protein_near_duplicate_audit=False,
    )
    assert audit["exact_overlap"][exclusive_family]["n_cross_partition"] == 0
    assert audit["exact_leakage_gate_passed"]


def test_small_chemical_near_duplicate_audit_is_complete() -> None:
    frame = _task_frame(12)
    config = SplitConfig(
        name="near",
        strategy="molecule_grouped",
        intended_use="unseen molecule",
        near_duplicate_tanimoto=0.5,
    )
    manifest, _ = make_split_manifest(frame, config)
    audit, examples = audit_split_leakage(
        frame,
        manifest,
        config,
        run_chemical_near_duplicate_audit=True,
        run_protein_near_duplicate_audit=False,
    )
    assert audit["near_duplicate"]["chemical"]["pair_audits"]
    assert set(examples.columns) >= {"modality", "similarity", "threshold"}


def test_near_duplicate_work_guard_and_unavailable_modality_are_incomplete() -> None:
    frame = _task_frame(12).drop(columns="sequence")
    config = SplitConfig(
        name="guarded",
        strategy="molecule_grouped",
        intended_use="unseen molecule",
        near_duplicate_max_pair_comparisons=1,
    )
    manifest, _ = make_split_manifest(frame, config)
    audit, _ = audit_split_leakage(frame, manifest, config)
    assert not audit["near_duplicate_audit_complete"]
    assert not audit["near_duplicate_modality_completeness"]["chemical"]["complete"]
    assert not audit["near_duplicate_modality_completeness"]["protein"]["complete"]


def test_near_duplicate_audit_collapses_repeated_measurements_before_work_guard() -> None:
    base = _task_frame(6)
    frame = pd.concat([base] * 10, ignore_index=True)
    frame["observation_id"] = [f"repeat-{index:03d}" for index in range(len(frame))]
    config = SplitConfig(
        name="repeated",
        strategy="molecule_grouped",
        intended_use="unseen molecule",
        near_duplicate_max_pair_comparisons=16,
        near_duplicate_max_protein_pair_comparisons=16,
    )
    manifest, _ = make_split_manifest(frame, config)
    audit, _ = audit_split_leakage(frame, manifest, config)
    assert audit["near_duplicate_audit_complete"]
    pair = audit["near_duplicate"]["chemical"]["pair_audits"]["train_vs_test"]
    assert pair["n_query_records"] > pair["n_query_unique_valid_structures"]


def test_default_suite_matches_every_configured_generalization_strategy() -> None:
    suite = default_split_suite(task_type="regression")
    assert len(suite) == 8
    assert {config.strategy for config in suite} == {
        "molecule_grouped",
        "scaffold",
        "chemical_cluster",
        "temporal",
        "source_holdout",
        "protein_holdout",
        "target_holdout",
        "double_cold",
    }


def test_direct_scaffold_split_rejects_bad_stereo_exact_proxy() -> None:
    frame = _task_frame()
    frame.loc[0, "standardized_smiles"] = r"N/C(=N\N=C\c1ccc(O)c(O)c1)c1nonc1N"
    config = SplitConfig(
        name="bad_stereo_direct",
        strategy="scaffold",
        intended_use="true Bemis-Murcko scaffold",
        task_type="classification",
    )

    with pytest.raises(
        ValueError,
        match="true scaffold split cannot admit exact-SMILES proxy groups",
    ):
        make_split_manifest(frame, config)


def test_streaming_scaffold_split_rejects_bad_stereo_without_publication(tmp_path: Path) -> None:
    frame = _streaming_task_frame(48)
    frame.loc[0, "standardized_smiles"] = r"N/C(=N\N=C\c1ccc(O)c(O)c1)c1nonc1N"
    source = tmp_path / "bad-stereo.parquet"
    output = tmp_path / "bad-stereo-split.parquet"
    frame.to_parquet(source, index=False)
    config = SplitConfig(
        name="bad_stereo_streaming",
        strategy="scaffold",
        intended_use="true Bemis-Murcko scaffold",
        task_type="classification",
    )

    with pytest.raises(
        ValueError,
        match="true streaming scaffold split cannot admit exact-SMILES proxy groups",
    ):
        stream_hash_group_split_manifest(source, output, config, batch_size=7)

    assert not output.exists()
    assert not output.with_suffix(output.suffix + ".manifest.json").exists()
    assert not list(tmp_path.glob(f".{output.name}.*"))


def test_stream_hash_split_is_bounded_deterministic_and_disk_audited(tmp_path: Path) -> None:
    frame = _streaming_task_frame()
    source = tmp_path / "task.parquet"
    shuffled_source = tmp_path / "task-shuffled.parquet"
    frame.to_parquet(source, index=False)
    frame.sample(frac=1, random_state=12).to_parquet(shuffled_source, index=False)
    config = SplitConfig(
        name="molecule_stream_v1",
        strategy="molecule_grouped",
        intended_use="new molecule at public scale",
        task_type="classification",
    )
    first_path = tmp_path / "first.parquet"
    second_path = tmp_path / "second.parquet"
    first = stream_hash_group_split_manifest(source, first_path, config, batch_size=7)
    stream_hash_group_split_manifest(shuffled_source, second_path, config, batch_size=11)
    first_frame = pd.read_parquet(first_path).sort_values("record_id").reset_index(drop=True)
    second_frame = pd.read_parquet(second_path).sort_values("record_id").reset_index(drop=True)
    pd.testing.assert_series_equal(first_frame["split"], second_frame["split"])
    pd.testing.assert_series_equal(first_frame["group_id"], second_frame["group_id"])
    assert first["record_count"] == len(frame)
    assert first["record_id_uniqueness"].startswith("complete_disk_backed")
    assert first["bounded_memory"]["maximum_observed_batch_rows"] <= 7
    assert first["near_duplicate_audit"].startswith("separate_required")
    assert (tmp_path / "first.parquet.manifest.json").is_file()


def test_manifest_bound_partitioned_task_split_preserves_global_order_and_binding(
    tmp_path: Path,
) -> None:
    frame = _streaming_task_frame()
    task_directory = _manifest_bound_task_directory(tmp_path, frame, split_at=113)
    resolved = resolve_manifest_bound_parquet_dataset(task_directory)
    assert resolved.input_kind == "manifest_bound_directory"
    assert resolved.total_rows == len(frame)
    assert [part.path.name for part in resolved.parts] == [
        "part-00000.parquet",
        "part-00001.parquet",
    ]

    config = SplitConfig(
        name="partitioned_molecule_stream_v1",
        strategy="molecule_grouped",
        intended_use="new molecule from manifest-bound task shards",
        task_type="classification",
    )
    output = tmp_path / "partitioned-split.parquet"
    metadata = stream_hash_group_split_manifest(
        task_directory,
        output,
        config,
        batch_size=17,
    )
    manifest = pd.read_parquet(output)
    assert manifest["record_id"].tolist() == frame["observation_id"].tolist()
    assert manifest["source_row_index"].tolist() == list(range(len(frame)))
    assert metadata["source_dataset_sha256"] == resolved.dataset_sha256
    assert metadata["source_dataset"]["manifest_sha256"] == resolved.manifest_sha256
    assert metadata["source_dataset"]["total_rows"] == len(frame)
    assert metadata["bounded_memory"]["maximum_observed_batch_rows"] <= 17


def test_manifest_bound_partitioned_task_fails_on_global_duplicate_or_heterogeneity(
    tmp_path: Path,
) -> None:
    config = SplitConfig(
        name="partitioned_failure_v1",
        strategy="molecule_grouped",
        intended_use="manifest-bound integrity test",
        task_type="classification",
    )
    duplicate = _streaming_task_frame(80)
    duplicate.loc[40, "observation_id"] = duplicate.loc[0, "observation_id"]
    duplicate_directory = _manifest_bound_task_directory(
        tmp_path / "duplicate",
        duplicate,
        split_at=40,
    )
    with pytest.raises(ValueError, match="Duplicate record ID"):
        stream_hash_group_split_manifest(
            duplicate_directory,
            tmp_path / "duplicate-split.parquet",
            config,
            batch_size=13,
        )

    heterogeneous = _streaming_task_frame(80)
    heterogeneous.loc[40:, "task_id"] = "different-task"
    heterogeneous_directory = _manifest_bound_task_directory(
        tmp_path / "heterogeneous",
        heterogeneous,
        split_at=40,
    )
    with pytest.raises(ValueError, match="Task signature changed across Parquet batches"):
        stream_hash_group_split_manifest(
            heterogeneous_directory,
            tmp_path / "heterogeneous-split.parquet",
            config,
            batch_size=13,
        )


def test_manifest_bound_partitioned_task_rejects_unmanifested_or_hash_changed_part(
    tmp_path: Path,
) -> None:
    frame = _streaming_task_frame(80)
    unmanifested_directory = _manifest_bound_task_directory(
        tmp_path / "unmanifested",
        frame,
    )
    frame.iloc[:4].to_parquet(unmanifested_directory / "part-99999.parquet", index=False)
    with pytest.raises(ValueError, match="do not exactly match"):
        resolve_manifest_bound_parquet_dataset(unmanifested_directory)

    changed_directory = _manifest_bound_task_directory(tmp_path / "changed", frame)
    changed_part = changed_directory / "part-00001.parquet"
    changed_part.write_bytes(changed_part.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        resolve_manifest_bound_parquet_dataset(changed_directory)


def test_stream_hash_split_rejects_cross_batch_duplicates_and_unsupported_strategy(
    tmp_path: Path,
) -> None:
    frame = _streaming_task_frame(48)
    frame.loc[30, "observation_id"] = frame.loc[0, "observation_id"]
    source = tmp_path / "duplicates.parquet"
    frame.to_parquet(source, index=False)
    config = SplitConfig(
        name="duplicate_stream",
        strategy="molecule_grouped",
        intended_use="new molecule",
        task_type="classification",
    )
    with pytest.raises(ValueError, match="Duplicate record ID"):
        stream_hash_group_split_manifest(source, tmp_path / "manifest.parquet", config, batch_size=8)

    clean_source = tmp_path / "clean.parquet"
    _streaming_task_frame(48).to_parquet(clean_source, index=False)
    unsupported = SplitConfig(
        name="cluster_stream",
        strategy="chemical_cluster",
        intended_use="new chemical cluster",
        task_type="classification",
    )
    with pytest.raises(ValueError, match="not implemented"):
        stream_hash_group_split_manifest(
            clean_source,
            tmp_path / "unsupported.parquet",
            unsupported,
        )


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("inclusion_status", "review", "inclusion_status=included"),
        ("inclusion_status", "quarantined", "inclusion_status=included"),
        ("observation_kind", "mystery", "unknown observation kinds"),
    ],
)
def test_stream_hash_split_fails_closed_on_ineligible_or_unknown_rows(
    tmp_path: Path,
    column: str,
    value: str,
    message: str,
) -> None:
    frame = _streaming_task_frame(48)
    frame.loc[20, column] = value
    source = tmp_path / f"invalid-{column}-{value}.parquet"
    frame.to_parquet(source, index=False)
    config = SplitConfig(
        name="invalid_stream",
        strategy="molecule_grouped",
        intended_use="new molecule",
        task_type="classification",
    )
    with pytest.raises(ValueError, match=message):
        stream_hash_group_split_manifest(
            source,
            tmp_path / f"invalid-{column}-{value}-manifest.parquet",
            config,
            batch_size=8,
        )
