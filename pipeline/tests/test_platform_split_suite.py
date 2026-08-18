from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from menin_discovery import platform_split_suite as split_suite_module
from menin_discovery.platform_data_schema import canonical_json
from menin_discovery.platform_data_sources import sha256_file
from menin_discovery.platform_split_suite import (
    SplitSuiteConfig,
    materialize_split_suite,
    verify_split_suite,
)

LABEL_COLUMNS = {
    "label_value",
    "label_text",
    "label_lower_bound",
    "label_upper_bound",
    "label_relation",
    "label_unit",
    "label_kind",
}


def _task_frame(rows: int = 90) -> pd.DataFrame:
    rings = ["C1" + "C" * (size - 1) + "1" for size in range(3, 33)]
    return pd.DataFrame(
        {
            "observation_id": [f"obs-{index:05d}" for index in range(rows)],
            "molecule_id": [f"molecule-{index:05d}" for index in range(rows)],
            "protein_id": ["protein-only"] * rows,
            "canonical_target_id": ["target-only"] * rows,
            "source_id": ["chembl-only"] * rows,
            "standardized_smiles": [rings[index % len(rings)] for index in range(rows)],
            "sequence": ["ACDEFGHIKLMNPQRSTVWY"] * rows,
            # These fields are deliberately distinctive.  The guarded Parquet
            # accessor below fails if production code requests any of them.
            "label_kind": ["continuous_exact"] * rows,
            "label_value": [float(10_000 + index) for index in range(rows)],
            "label_text": [f"POISON-LABEL-{index:05d}" for index in range(rows)],
            "label_relation": ["="] * rows,
            "label_lower_bound": [float(10_000 + index) for index in range(rows)],
            "label_upper_bound": [float(10_000 + index) for index in range(rows)],
            "label_unit": ["nM"] * rows,
        }
    )


def _write_task_dataset(
    root: Path,
    frame: pd.DataFrame,
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    directory = root / "tasks" / "default" / "binding_kd"
    directory.mkdir(parents=True)
    boundaries = (0, len(frame) // 2, len(frame))
    parts: list[dict[str, object]] = []
    components: list[dict[str, object]] = []
    shards: list[dict[str, object]] = []
    arrow_digest = hashlib.sha256(b"split-suite-test-schema").hexdigest()
    for index, (start, stop) in enumerate(zip(boundaries[:-1], boundaries[1:], strict=True)):
        path = directory / f"part-{index:05d}.parquet"
        frame.iloc[start:stop].to_parquet(path, index=False)
        relative = path.relative_to(root).as_posix()
        digest = sha256_file(path)
        part = {
            "path": relative,
            "rows": stop - start,
            "sha256": digest,
            "arrow_schema_sha256": arrow_digest,
        }
        parts.append(part)
        components.append(
            {
                "path": relative,
                "rows": stop - start,
                "sha256": digest,
                "size_bytes": path.stat().st_size,
            }
        )
        shards.append(
            {
                "relative_path": relative,
                "rows": stop - start,
                "sha256": digest,
                "size_bytes": path.stat().st_size,
            }
        )
    payload: dict[str, object] = {
        "task_scope": "default",
        "task_type": "binding_kd",
        "row_count": len(frame),
        "part_count": len(parts),
        "parts": parts,
        "arrow_schema": {"sha256": arrow_digest},
        "dataset_sha256": hashlib.sha256(canonical_json(parts).encode()).hexdigest(),
    }
    return payload, components, shards


def _canonical_fixture(
    parent: Path,
    frame: pd.DataFrame | None = None,
) -> tuple[Path, Path, set[Path]]:
    root = parent / "full_chembl37"
    root.mkdir(parents=True)
    payload, components, shards = _write_task_dataset(root, _task_frame() if frame is None else frame)
    task_datasets = {"default::binding_kd": payload}
    task_path = root / "task_datasets.json"
    task_path.write_text(json.dumps(task_datasets, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    components.append(
        {
            "path": "task_datasets.json",
            "sha256": sha256_file(task_path),
            "size_bytes": task_path.stat().st_size,
        }
    )
    manifest = {
        "schema_version": "test-v1",
        "source_id": "chembl",
        "snapshot_id": "chembl37-split-suite-test",
        "shard_artifacts": shards,
        "task_datasets_manifest_sha256": hashlib.sha256(canonical_json(task_datasets).encode()).hexdigest(),
        "component_inventory": sorted(components, key=lambda row: str(row["path"])),
    }
    manifest_path = root / "build_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    qc_path = parent / "qc_report.json"
    qc_path.write_text(
        json.dumps(
            {"qc_passed": True, "build_manifest_sha256": sha256_file(manifest_path)},
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    canonical_parts = {path.resolve() for path in root.glob("tasks/default/*/*.parquet")}
    return root, qc_path, canonical_parts


def _config() -> SplitSuiteConfig:
    return SplitSuiteConfig(
        batch_size=11,
        near_sample_cap_per_partition=3,
        chemical_fingerprint_bits=128,
    )


def _assert_no_absolute_path(value: Any) -> None:
    if isinstance(value, dict):
        for item in value.values():
            _assert_no_absolute_path(item)
    elif isinstance(value, list):
        for item in value:
            _assert_no_absolute_path(item)
    elif isinstance(value, str):
        assert not value.startswith("/")
        assert not (len(value) > 2 and value[1] == ":" and value[2] in "\\/")


def _write_canonical_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _recompute_accounting(acceptance: dict[str, Any]) -> None:
    accounting: Counter[str] = Counter()
    for task in acceptance["tasks"]:
        accounting["tasks_enumerated"] += 1
        for item in task["strategies"]:
            status = item["status"]
            if status == "materialized":
                accounting["strategies_materialized"] += 1
            else:
                accounting["strategies_skipped"] += 1
            accounting[f"strategy_{item['strategy']}_{status}"] += 1
    acceptance["accounting"] = dict(sorted(accounting.items()))


def _write_bound_task_tree(output: Path, acceptance: dict[str, Any]) -> None:
    for task in acceptance["tasks"]:
        task_root = output / "tasks" / task["task_slug"]
        for item in task["strategies"]:
            _write_canonical_json(
                task_root / "strategies" / item["strategy"] / "status.json",
                item,
            )
        _write_canonical_json(task_root / "task_status.json", task)


def _refresh_inventory(output: Path, acceptance: dict[str, Any]) -> None:
    inventory = split_suite_module._component_inventory(output)
    acceptance["component_inventory"] = inventory
    acceptance["component_inventory_sha256"] = split_suite_module.stable_json_digest(inventory)
    _write_canonical_json(output / "acceptance.json", acceptance)


def _guard_canonical_label_reads(
    monkeypatch: pytest.MonkeyPatch,
    canonical_parts: set[Path],
) -> list[tuple[Path, tuple[str, ...]]]:
    original = pq.ParquetFile
    reads: list[tuple[Path, tuple[str, ...]]] = []

    class GuardedParquetFile:
        def __init__(self, path: str | Path, *args: object, **kwargs: object) -> None:
            self.path = Path(path).resolve()
            self.delegate = original(path, *args, **kwargs)

        def __getattr__(self, name: str) -> Any:
            return getattr(self.delegate, name)

        def iter_batches(
            self,
            *args: object,
            columns: list[str] | None = None,
            **kwargs: object,
        ) -> Any:
            if self.path in canonical_parts:
                if columns is None:
                    raise AssertionError("canonical task was read without a feature-only projection")
                forbidden = LABEL_COLUMNS & set(columns)
                if forbidden:
                    raise AssertionError(f"canonical labels were requested: {sorted(forbidden)}")
                reads.append((self.path, tuple(columns)))
            return self.delegate.iter_batches(*args, columns=columns, **kwargs)

    monkeypatch.setattr(split_suite_module.pq, "ParquetFile", GuardedParquetFile)
    return reads


def test_split_suite_is_label_blind_deterministic_and_explicit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical, qc, canonical_parts = _canonical_fixture(tmp_path / "input")
    reads = _guard_canonical_label_reads(monkeypatch, canonical_parts)
    first_root = tmp_path / "split-suite-one"
    first = materialize_split_suite(canonical, qc, first_root, _config())

    assert reads
    assert all(not (set(columns) & LABEL_COLUMNS) for _, columns in reads)
    assert first["label_access_contract"] == {
        "canonical_label_columns_requested": [],
        "label_values_read": False,
        "test_labels_disclosed": False,
        "published_parquet_contains_label_columns": False,
    }
    assert first["substantive_training_started"] is False
    assert first["task_order"] == ["default::binding_kd"]
    strategies = {item["strategy"]: item for item in first["tasks"][0]["strategies"]}
    assert strategies["molecule_grouped"]["status"] == "materialized"
    assert strategies["scaffold"]["status"] == "materialized"
    for strategy in ("source_holdout", "protein_holdout", "target_holdout", "double_cold"):
        assert strategies[strategy]["status"] == "skipped_inapplicable"
        assert strategies[strategy]["reasons"]
    assert strategies["source_holdout"]["reasons"] == ["fewer_than_three_unique_source_entities"]

    for path in first_root.rglob("*.parquet"):
        assert not any(column.startswith("label") for column in pq.ParquetFile(path).schema_arrow.names)
    for path in first_root.rglob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["substantive_training_started"] is False
        _assert_no_absolute_path(payload)

    for strategy in ("molecule_grouped", "scaffold"):
        leakage_path = Path(strategies[strategy]["leakage_diagnostics_path"])
        leakage = json.loads((first_root / leakage_path).read_text(encoding="utf-8"))
        assert leakage["exact_overlap_audit"]["completeness"] == "exhaustive"
        assert (
            leakage["exact_overlap_audit"]["entities"]["group"][
                "entities_present_in_multiple_official_partitions"
            ]
            == 0
        )
        assert leakage["chemical_near_similarity"]["completeness"] == "sampled_non_exhaustive"
        assert leakage["protein_near_similarity"]["completeness"] == "sampled_non_exhaustive"
        assert max(leakage["chemical_near_similarity"]["sampled_unique_entities"].values()) <= 3
        assert "not_claim_ready" in leakage["claim_readiness"]

    verified = verify_split_suite(
        first_root,
        canonical_build_root=canonical,
        qc_report_path=qc,
    )
    assert verified["status"] == "verified"
    assert verified["source_reverified"] is True
    assert verified["label_values_read"] is False

    second_root = tmp_path / "split-suite-two"
    second = materialize_split_suite(canonical, qc, second_root, _config())
    assert first["component_inventory_sha256"] == second["component_inventory_sha256"]
    assert first["accounting"] == second["accounting"]


def test_split_suite_bad_stereo_skips_scaffold_but_materializes_molecule_grouped(
    tmp_path: Path,
) -> None:
    frame = _task_frame()
    frame.loc[0, "standardized_smiles"] = r"N/C(=N\N=C\c1ccc(O)c(O)c1)c1nonc1N"
    canonical, qc, _ = _canonical_fixture(tmp_path / "input", frame)
    output = tmp_path / "split-suite"

    result = materialize_split_suite(canonical, qc, output, _config())
    task = result["tasks"][0]
    strategies = {item["strategy"]: item for item in task["strategies"]}

    assert task["feature_projection"]["scaffold_method_row_counts"] == {
        "bemis_murcko": len(frame) - 1,
        "exact_smiles_proxy_rdkit_exception": 1,
    }
    assert strategies["molecule_grouped"]["status"] == "materialized"
    assert strategies["molecule_grouped"]["split_materialized"] is True
    scaffold = strategies["scaffold"]
    assert scaffold["status"] == "skipped_inapplicable_mandatory_candidate"
    assert scaffold["applicability"] == {
        "evaluated": True,
        "supported": False,
        "reasons": ["rdkit_scaffold_exception_requires_exact_smiles_proxy"],
    }
    assert scaffold["reasons"] == ["rdkit_scaffold_exception_requires_exact_smiles_proxy"]
    assert scaffold["split_materialized"] is False
    assert not (output / "tasks" / task["task_slug"] / "strategies" / "scaffold" / "split.parquet").exists()
    assert (
        verify_split_suite(
            output,
            canonical_build_root=canonical,
            qc_report_path=qc,
        )["status"]
        == "verified"
    )


def test_split_suite_verifier_detects_tampering(tmp_path: Path) -> None:
    canonical, qc, _ = _canonical_fixture(tmp_path / "input")
    output = tmp_path / "split-suite"
    result = materialize_split_suite(canonical, qc, output, _config())
    materialized = next(item for item in result["tasks"][0]["strategies"] if item["status"] == "materialized")
    leakage = output / materialized["leakage_diagnostics_path"]
    leakage.write_text(leakage.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="size mismatch"):
        verify_split_suite(output)


def test_split_suite_verifier_rejects_semantic_acceptance_exploits(tmp_path: Path) -> None:
    canonical, qc, _ = _canonical_fixture(tmp_path / "input")
    output = tmp_path / "split-suite"
    materialize_split_suite(canonical, qc, output, _config())
    acceptance_path = output / "acceptance.json"
    original = json.loads(acceptance_path.read_text(encoding="utf-8"))

    exploit = json.loads(json.dumps(original))
    exploit["label_access_contract"]["label_values_read"] = True
    _write_canonical_json(acceptance_path, exploit)
    with pytest.raises(ValueError, match="fixed top-level scientific policy"):
        verify_split_suite(output, canonical_build_root=canonical, qc_report_path=qc)

    exploit = json.loads(json.dumps(original))
    exploit["tasks"][0]["canonical_task_dataset_sha256"] = "0" * 64
    _write_canonical_json(acceptance_path, exploit)
    with pytest.raises(ValueError, match="differs from bound task status"):
        verify_split_suite(output, canonical_build_root=canonical, qc_report_path=qc)

    exploit = json.loads(json.dumps(original))
    materialized = next(
        item for item in exploit["tasks"][0]["strategies"] if item["status"] == "materialized"
    )
    materialized["exact_overlap_status"] = "tampered"
    _write_canonical_json(acceptance_path, exploit)
    with pytest.raises(ValueError, match="differs from bound task status"):
        verify_split_suite(output, canonical_build_root=canonical, qc_report_path=qc)

    exploit = json.loads(json.dumps(original))
    exploit["source_binding"]["source_id"] = "fabricated-source"
    exploit["source_binding"]["snapshot_id"] = "fabricated-snapshot"
    _write_canonical_json(acceptance_path, exploit)
    with pytest.raises(ValueError, match="source binding no longer matches"):
        verify_split_suite(output, canonical_build_root=canonical, qc_report_path=qc)

    _write_canonical_json(acceptance_path, original)
    feature_path = output / original["tasks"][0]["feature_projection_path"]
    outside = tmp_path / "identical-features.parquet"
    outside.write_bytes(feature_path.read_bytes())
    feature_path.unlink()
    feature_path.symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        verify_split_suite(output, canonical_build_root=canonical, qc_report_path=qc)


def test_split_suite_source_rebind_validates_each_task_semantic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical, qc, _ = _canonical_fixture(tmp_path / "input")
    output = tmp_path / "split-suite"
    materialize_split_suite(canonical, qc, output, _config())
    binding = split_suite_module._verify_canonical_corpus(canonical, qc)
    changed_tasks = json.loads(json.dumps(binding.task_datasets))
    changed_tasks["default::binding_kd"]["dataset_sha256"] = "f" * 64
    changed_binding = split_suite_module.CanonicalCorpusBinding(
        root=binding.root,
        build_manifest=binding.build_manifest,
        build_manifest_sha256=binding.build_manifest_sha256,
        component_inventory_sha256=binding.component_inventory_sha256,
        qc_report_sha256=binding.qc_report_sha256,
        task_datasets_path=binding.task_datasets_path,
        task_datasets_sha256=binding.task_datasets_sha256,
        task_datasets=changed_tasks,
    )
    monkeypatch.setattr(
        split_suite_module,
        "_verify_canonical_corpus",
        lambda *_args, **_kwargs: changed_binding,
    )
    with pytest.raises(ValueError, match="task semantics no longer match"):
        verify_split_suite(output, canonical_build_root=canonical, qc_report_path=qc)


def test_split_suite_source_rebind_rejects_coordinated_feature_rewrite(
    tmp_path: Path,
) -> None:
    canonical, qc, _ = _canonical_fixture(tmp_path / "input")
    output = tmp_path / "split-suite"
    materialize_split_suite(canonical, qc, output, _config())
    acceptance_path = output / "acceptance.json"
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    task = acceptance["tasks"][0]
    feature_path = output / task["feature_projection_path"]
    features = pq.read_table(feature_path)
    smiles = features.column("smiles").to_pylist()
    smiles[0] = "C1" + "C" * 40 + "1"
    features = features.set_column(
        features.schema.get_field_index("smiles"),
        pa.field("smiles", pa.string()),
        pa.array(smiles, type=pa.string()),
    )
    pq.write_table(features, feature_path, compression="zstd")
    task["feature_projection"]["feature_file_sha256"] = sha256_file(feature_path)
    task["feature_projection"]["feature_file_bytes"] = feature_path.stat().st_size
    for item in task["strategies"]:
        if item["status"] != "materialized":
            continue
        sidecar_path = output / item["sidecar_path"]
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        sidecar["feature_projection_sha256"] = sha256_file(feature_path)
        _write_canonical_json(sidecar_path, sidecar)
        item["sidecar_sha256"] = sha256_file(sidecar_path)
    _write_bound_task_tree(output, acceptance)
    _refresh_inventory(output, acceptance)

    with pytest.raises(ValueError, match="canonical feature-only regeneration"):
        verify_split_suite(output, canonical_build_root=canonical, qc_report_path=qc)


def test_split_suite_source_rebind_rejects_fixed_seed_assignment_drift(
    tmp_path: Path,
) -> None:
    canonical, qc, _ = _canonical_fixture(tmp_path / "input")
    output = tmp_path / "split-suite"
    materialize_split_suite(canonical, qc, output, _config())
    acceptance = json.loads((output / "acceptance.json").read_text(encoding="utf-8"))
    task = acceptance["tasks"][0]
    item = next(candidate for candidate in task["strategies"] if candidate["strategy"] == "molecule_grouped")
    split_path = output / item["split_path"]
    split = pq.read_table(split_path)
    values = split.column("split").to_pylist()
    train_index = values.index("train")
    test_index = values.index("test")
    values[train_index], values[test_index] = values[test_index], values[train_index]
    split = split.set_column(
        split.schema.get_field_index("split"),
        pa.field("split", pa.string()),
        pa.array(values, type=pa.string()),
    )
    pq.write_table(split, split_path, compression="zstd")
    item["split_sha256"] = sha256_file(split_path)
    sidecar_path = output / item["sidecar_path"]
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["manifest_sha256"] = sha256_file(split_path)
    _write_canonical_json(sidecar_path, sidecar)
    item["sidecar_sha256"] = sha256_file(sidecar_path)
    _write_bound_task_tree(output, acceptance)
    _refresh_inventory(output, acceptance)

    with pytest.raises(ValueError, match="deterministic fixed-seed regeneration"):
        verify_split_suite(output, canonical_build_root=canonical, qc_report_path=qc)


def test_split_suite_rejects_fabricated_fixed_seed_skip_after_artifact_removal(
    tmp_path: Path,
) -> None:
    canonical, qc, _ = _canonical_fixture(tmp_path / "input")
    output = tmp_path / "split-suite"
    materialize_split_suite(canonical, qc, output, _config())
    acceptance = json.loads((output / "acceptance.json").read_text(encoding="utf-8"))
    task = acceptance["tasks"][0]
    item = next(candidate for candidate in task["strategies"] if candidate["strategy"] == "molecule_grouped")
    for field in ("split_path", "sidecar_path", "leakage_diagnostics_path"):
        (output / item[field]).unlink()
    for field in (
        "split_path",
        "split_sha256",
        "split_rows",
        "partition_counts",
        "sidecar_path",
        "sidecar_sha256",
        "leakage_diagnostics_path",
        "leakage_diagnostics_sha256",
        "exact_overlap_status",
        "near_similarity_status",
    ):
        item.pop(field, None)
    item.update(
        {
            "status": "skipped_fixed_seed_mandatory_candidate",
            "reasons": ["fixed_seed_partition_support_empty_no_seed_retry"],
            "split_materialized": False,
            "failure_class": "expected_fixed_seed_support_limitation",
        }
    )
    task["mandatory_candidates_materialized"] = False
    _recompute_accounting(acceptance)
    _write_bound_task_tree(output, acceptance)
    _refresh_inventory(output, acceptance)

    with pytest.raises(ValueError, match="materializable without seed search"):
        verify_split_suite(output, canonical_build_root=canonical, qc_report_path=qc)


def test_split_suite_rejects_arbitrary_topology_config_and_special_entries(
    tmp_path: Path,
) -> None:
    canonical, qc, _ = _canonical_fixture(tmp_path / "input")
    output = tmp_path / "split-suite"
    materialize_split_suite(canonical, qc, output, _config())
    acceptance_path = output / "acceptance.json"
    original = json.loads(acceptance_path.read_text(encoding="utf-8"))

    arbitrary = output / "inventoried-but-unexpected.txt"
    arbitrary.write_text("unexpected\n", encoding="utf-8")
    exploit = json.loads(json.dumps(original))
    _refresh_inventory(output, exploit)
    with pytest.raises(ValueError, match="exact output topology mismatch"):
        verify_split_suite(output)
    arbitrary.unlink()

    exploit = json.loads(json.dumps(original))
    exploit["configuration"]["unexpected_key"] = True
    exploit["configuration_sha256"] = split_suite_module.stable_json_digest(exploit["configuration"])
    _write_canonical_json(acceptance_path, exploit)
    with pytest.raises(ValueError, match="configuration keys differ"):
        verify_split_suite(output)

    exploit = json.loads(json.dumps(original))
    exploit["unreviewed_claim"] = "fabricated"
    _write_canonical_json(acceptance_path, exploit)
    with pytest.raises(ValueError, match="top acceptance keys differ"):
        verify_split_suite(output)

    _write_canonical_json(acceptance_path, original)
    empty = output / "unexpected-empty-directory"
    empty.mkdir()
    with pytest.raises(ValueError, match="exact output topology mismatch"):
        verify_split_suite(output)
    empty.rmdir()

    fifo = output / "unexpected-fifo"
    os.mkfifo(fifo)
    try:
        with pytest.raises(ValueError, match="special filesystem entry"):
            verify_split_suite(output)
    finally:
        fifo.unlink()

    exploit = json.loads(json.dumps(original))
    relative = next(iter(exploit["component_inventory"]))
    exploit["component_inventory"][relative]["relative_path"] = "fabricated/path"
    exploit["component_inventory"][relative]["claim_ready"] = True
    exploit["component_inventory_sha256"] = split_suite_module.stable_json_digest(
        exploit["component_inventory"]
    )
    _write_canonical_json(acceptance_path, exploit)
    with pytest.raises(ValueError, match="component inventory entry .* keys differ"):
        verify_split_suite(output, canonical_build_root=canonical, qc_report_path=qc)

    exploit["component_inventory"][relative].pop("claim_ready")
    exploit["component_inventory_sha256"] = split_suite_module.stable_json_digest(
        exploit["component_inventory"]
    )
    _write_canonical_json(acceptance_path, exploit)
    with pytest.raises(ValueError, match="relative_path differs from its key"):
        verify_split_suite(output, canonical_build_root=canonical, qc_report_path=qc)


def test_split_suite_rejects_large_training_flag_and_duplicate_json_key(
    tmp_path: Path,
) -> None:
    canonical, qc, _ = _canonical_fixture(tmp_path / "input")
    output = tmp_path / "split-suite"
    materialize_split_suite(canonical, qc, output, _config())
    acceptance_path = output / "acceptance.json"
    original_text = acceptance_path.read_text(encoding="utf-8")
    original = json.loads(original_text)
    original["large_model_training_started"] = True
    _write_canonical_json(acceptance_path, original)
    with pytest.raises(ValueError, match="large-model no-training"):
        verify_split_suite(output)

    acceptance_path.write_text(
        original_text.replace(
            '  "schema_version": "platform_split_suite_v1",',
            '  "schema_version": "platform_split_suite_v1",\n  "schema_version": "platform_split_suite_v1",',
            1,
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Duplicate JSON object key"):
        verify_split_suite(output)


def test_unexpected_split_failure_rolls_back_outer_transaction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical, qc, _ = _canonical_fixture(tmp_path / "input")
    output = tmp_path / "split-suite"
    original = split_suite_module.stream_hash_group_split_manifest
    calls = 0

    def fail_second_call(*args: object, **kwargs: object) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected split-suite failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(
        split_suite_module,
        "stream_hash_group_split_manifest",
        fail_second_call,
    )
    with pytest.raises(RuntimeError, match="injected split-suite failure"):
        materialize_split_suite(canonical, qc, output, _config())
    assert not output.exists()
    assert not list(tmp_path.glob(".split-suite.building-*"))


def test_split_suite_config_rejects_unbounded_sampling() -> None:
    with pytest.raises(ValueError, match="between 1 and 2048"):
        SplitSuiteConfig(near_sample_cap_per_partition=2_049).validate()
    with pytest.raises(ValueError, match="not a boolean"):
        SplitSuiteConfig(seed=True).validate()
