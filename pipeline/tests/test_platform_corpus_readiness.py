from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest
from menin_discovery import platform_corpus_readiness as corpus_module
from menin_discovery.platform_corpus_readiness import (
    CorpusReadinessConfig,
    materialize_corpus_readiness_bundle,
    verify_corpus_readiness_bundle,
)
from menin_discovery.platform_data_schema import canonical_json
from menin_discovery.platform_data_sources import sha256_file


def _task_frame(
    task_id: str,
    task_type: str,
    n_rows: int,
    *,
    observation_kind: str = "experimental_summary",
) -> pd.DataFrame:
    smiles = ("CCO", "CCN", "CCC", "CCCl", "CCBr", "COC")
    derived = observation_kind == "derived"
    return pd.DataFrame(
        {
            "observation_id": [f"{task_id}-obs-{index:05d}" for index in range(n_rows)],
            "molecule_id": [f"{task_id}-mol-{index:05d}" for index in range(n_rows)],
            "protein_id": [f"protein-{index % 5}" for index in range(n_rows)],
            "canonical_target_id": [f"target-{index % 3}" for index in range(n_rows)],
            "assay_id": [f"assay-{index % 7}" for index in range(n_rows)],
            "source_id": ["chembl"] * n_rows,
            "snapshot_id": ["chembl37-test"] * n_rows,
            "source_record_id": [f"source-{task_id}-{index}" for index in range(n_rows)],
            "access_class": ["public_redistributable"] * n_rows,
            "default_task_eligible": [not derived] * n_rows,
            "sensitivity_task_eligible": [derived] * n_rows,
            "inclusion_status": ["included"] * n_rows,
            "standardized_smiles": [smiles[index % len(smiles)] for index in range(n_rows)],
            "canonical_smiles": [smiles[index % len(smiles)] for index in range(n_rows)],
            "standard_inchi_key": [f"IK-{task_id}-{index}" for index in range(n_rows)],
            "structure_id": [f"structure-{task_id}-{index}" for index in range(n_rows)],
            "sequence": ["ACDEFGHIKLMNPQRSTVWY"] * n_rows,
            "target_name": ["target"] * n_rows,
            "species": ["Homo sapiens"] * n_rows,
            "description": ["test assay"] * n_rows,
            "matrix": [""] * n_rows,
            "route": [""] * n_rows,
            "task_id": [task_id] * n_rows,
            "task_type": [task_type] * n_rows,
            "label_kind": ["continuous_exact"] * n_rows,
            "label_value": [float(index % 31) for index in range(n_rows)],
            "label_text": [""] * n_rows,
            "label_relation": ["="] * n_rows,
            "label_lower_bound": [float(index % 31) for index in range(n_rows)],
            "label_upper_bound": [float(index % 31) for index in range(n_rows)],
            "label_unit": ["nM"] * n_rows,
            "observation_kind": [observation_kind] * n_rows,
            "label_lineage_digest": ["d" * 64 if derived else ""] * n_rows,
            "evidence_domain": ["binding"] * n_rows,
            "endpoint": ["Kd"] * n_rows,
            "assay_family": ["binding"] * n_rows,
            "document_year": [2010 + index % 10 for index in range(n_rows)],
        }
    )


def _write_task_dataset(
    root: Path,
    scope: str,
    task_type: str,
    frame: pd.DataFrame,
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    slug = task_type.replace("/", "_")
    directory = root / "tasks" / scope / slug
    directory.mkdir(parents=True)
    boundaries = [0, len(frame) // 2, len(frame)]
    parts: list[dict[str, object]] = []
    components: list[dict[str, object]] = []
    shards: list[dict[str, object]] = []
    arrow_digest = hashlib.sha256(f"schema:{task_type}".encode()).hexdigest()
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
    payload = {
        "task_scope": scope,
        "task_type": task_type,
        "row_count": len(frame),
        "part_count": len(parts),
        "parts": parts,
        "arrow_schema": {"sha256": arrow_digest},
        "dataset_sha256": hashlib.sha256(canonical_json(parts).encode()).hexdigest(),
    }
    return payload, components, shards


def _canonical_fixture(
    parent: Path,
    *,
    supported_rows: int = 180,
    unique_continuous_labels: bool = False,
) -> tuple[Path, Path]:
    root = parent / "full_chembl37"
    root.mkdir(parents=True)
    task_datasets: dict[str, object] = {}
    components: list[dict[str, object]] = []
    shards: list[dict[str, object]] = []
    supported = _task_frame("TASK-SUPPORTED", "default_binding_kd_exact_supported", supported_rows)
    if unique_continuous_labels:
        values = [float(index) for index in range(supported_rows)]
        supported["label_value"] = values
        supported["label_lower_bound"] = values
        supported["label_upper_bound"] = values
    datasets = (
        (
            "default",
            "default_binding_kd_exact_supported",
            supported,
        ),
        (
            "default",
            "default_binding_kd_exact_tiny",
            _task_frame("TASK-TINY", "default_binding_kd_exact_tiny", 2),
        ),
        (
            "derived_sensitivity",
            "sensitivity_binding_free_energy",
            _task_frame(
                "TASK-DERIVED",
                "sensitivity_binding_free_energy",
                20,
                observation_kind="derived",
            ),
        ),
    )
    for scope, task_type, frame in datasets:
        payload, task_components, task_shards = _write_task_dataset(root, scope, task_type, frame)
        task_datasets[f"{scope}::{task_type}"] = payload
        components.extend(task_components)
        shards.extend(task_shards)
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
        "snapshot_id": "chembl37-test",
        "shard_artifacts": shards,
        "task_datasets_manifest_sha256": hashlib.sha256(canonical_json(task_datasets).encode()).hexdigest(),
        "component_inventory": sorted(components, key=lambda row: str(row["path"])),
    }
    manifest_path = root / "build_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    qc_path = parent / "qc_report.json"
    qc_path.write_text(
        json.dumps(
            {
                "qc_passed": True,
                "build_manifest_sha256": sha256_file(manifest_path),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return root, qc_path


def _config() -> CorpusReadinessConfig:
    return CorpusReadinessConfig(
        preflight_batch_size=23,
        split_batch_size=29,
        serialization_batch_size=31,
        loader_batch_size=4,
        loader_maximum_batches=2,
        diagnostic_maximum_train_examples=32,
        diagnostic_maximum_validation_examples=16,
        diagnostic_fingerprint_bits=64,
    )


def test_corpus_orchestration_integrates_every_supported_default_and_records_skips(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical, qc = _canonical_fixture(tmp_path / "input")
    output = tmp_path / "corpus-readiness"
    original_sha256 = corpus_module.sha256_file
    test_hash_calls: list[Path] = []

    def guarded_sha256(path: Path) -> str:
        if Path(path).name == "test.jsonl":
            test_hash_calls.append(Path(path))
            raise AssertionError("top orchestrator reopened a routed test lockbox")
        return original_sha256(path)

    monkeypatch.setattr(corpus_module, "sha256_file", guarded_sha256)
    result = materialize_corpus_readiness_bundle(
        canonical,
        qc,
        output,
        _config(),
    )
    assert test_hash_calls == []
    assert result["task_counts"]["default_tasks_enumerated"] == 2
    assert result["task_counts"]["default_tasks_integrated"] == 1
    assert result["task_counts"]["default_tasks_preflight_skipped"] == 1
    assert result["task_counts"]["derived_sensitivity_tasks_enumerated"] == 1
    assert result["task_counts"]["derived_sensitivity_tasks_integrated"] == 1
    assert result["task_order"] == sorted(result["task_order"])
    by_key = {record["dataset_key"]: record for record in result["tasks"]}
    supported = by_key["default::default_binding_kd_exact_supported"]
    assert supported["status"] == "integrated_and_diagnostic_completed"
    assert supported["diagnostic_status"] == "completed_diagnostic_only"
    assert supported["test_partition_opened_by_diagnostics"] is False
    tiny = by_key["default::default_binding_kd_exact_tiny"]
    assert tiny["status"] == "skipped_insufficient_fixed_split_support"
    assert tiny["skip_reasons"]
    derived = by_key["derived_sensitivity::sensitivity_binding_free_energy"]
    assert derived["status"] == "integrated_with_diagnostic_skipped"
    assert derived["diagnostic_status"] == "skipped"
    assert derived["diagnostic_reason"] == "skipped_derived_sensitivity_not_primary_diagnostic"

    locked = [
        entry
        for entry in result["component_inventory"].values()
        if entry["verification_source"] == "transitively_bound_during_physical_routing_no_reopen"
    ]
    assert len(locked) == 2
    verified = verify_corpus_readiness_bundle(
        output,
        canonical_build_root=canonical,
        qc_report_path=qc,
    )
    assert verified["status"] == "verified"
    assert verified["source_reverified"] is True
    assert verified["test_lockboxes_opened_or_hashed"] is False
    assert test_hash_calls == []


def test_corpus_verifier_detects_nonlockbox_tampering(tmp_path: Path) -> None:
    canonical, qc = _canonical_fixture(tmp_path / "input")
    output = tmp_path / "corpus-readiness"
    materialize_corpus_readiness_bundle(canonical, qc, output, _config())
    task_status = next(output.glob("tasks/*/task_status.json"))
    task_status.write_text(task_status.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="size mismatch"):
        verify_corpus_readiness_bundle(output)


def test_unexpected_task_error_aborts_outer_transaction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical, qc = _canonical_fixture(tmp_path / "input")
    output = tmp_path / "corpus-readiness"

    def fail_integration(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise RuntimeError("injected unexpected integration failure")

    monkeypatch.setattr(corpus_module, "materialize_task_integration_bundle", fail_integration)
    with pytest.raises(RuntimeError, match="injected unexpected"):
        materialize_corpus_readiness_bundle(canonical, qc, output, _config())
    assert not output.exists()
    assert not list(tmp_path.glob(".corpus-readiness.building-*"))


def test_canonical_tamper_is_rejected_before_output(tmp_path: Path) -> None:
    canonical, qc = _canonical_fixture(tmp_path / "input")
    task_part = next(canonical.glob("tasks/default/*/*.parquet"))
    task_part.write_bytes(task_part.read_bytes() + b"tamper")
    output = tmp_path / "corpus-readiness"
    with pytest.raises(ValueError, match="size mismatch"):
        materialize_corpus_readiness_bundle(canonical, qc, output, _config())
    assert not output.exists()


def test_structural_preflight_never_requests_or_emits_any_partition_labels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical, qc = _canonical_fixture(tmp_path / "input")
    binding = corpus_module._verify_canonical_corpus(canonical, qc)
    key = "default::default_binding_kd_exact_supported"
    entry = binding.task_datasets[key]
    original_parquet_file = corpus_module.pq.ParquetFile
    requested_columns: list[tuple[str, ...]] = []

    class GuardedParquetFile:
        def __init__(self, path: Path) -> None:
            self._delegate = original_parquet_file(path)

        @property
        def schema_arrow(self) -> object:
            return self._delegate.schema_arrow

        def iter_batches(self, **kwargs: object) -> object:
            raw_columns = kwargs.get("columns")
            assert isinstance(raw_columns, list)
            columns = tuple(str(value) for value in raw_columns)
            requested_columns.append(columns)
            assert "label_value" not in columns
            assert "label_text" not in columns
            return self._delegate.iter_batches(**kwargs)

    monkeypatch.setattr(corpus_module.pq, "ParquetFile", GuardedParquetFile)
    preflight = corpus_module._preflight_task(binding, key, entry, _config())
    assert requested_columns
    assert preflight["label_access"] == {
        "label_columns_read": [],
        "training_labels_read": False,
        "validation_labels_read": False,
        "test_labels_read": False,
        "policy": "structural preflight never requests label_value or label_text columns",
    }
    serialized = canonical_json(preflight)
    assert "partition_label_counts" not in serialized
    assert "capped_diagnostic_label_counts" not in serialized


def test_categorical_support_reads_only_routed_train_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    integration = tmp_path / "integration"
    partitions = integration / "model_ready" / "partitions"
    partitions.mkdir(parents=True)
    train = partitions / "train.jsonl"
    train.write_text(
        "".join(
            json.dumps(
                {
                    "record_id": f"train-{index}",
                    "label": {"value": float(index % 2), "text": ""},
                },
                sort_keys=True,
            )
            + "\n"
            for index in range(8)
        ),
        encoding="utf-8",
    )
    test = partitions / "test.jsonl"
    test.write_text(
        json.dumps(
            {
                "record_id": "test-secret",
                "label": {"value": 999.0, "text": "NEVER_INSPECT_TEST_SECRET"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    partition_manifest = {
        "partitions": {
            "train": {
                "path": "train.jsonl",
                "sha256": sha256_file(train),
                "record_count": 8,
            },
            "test": {
                "path": "test.jsonl",
                "sha256": sha256_file(test),
                "record_count": 1,
            },
        }
    }
    manifest_path = partitions / "partition_manifest.json"
    manifest_path.write_text(json.dumps(partition_manifest, sort_keys=True) + "\n", encoding="utf-8")
    acceptance = {
        "task_semantics": {"label_kind": "categorical"},
        "physical_partition_manifest_file_sha256_external": sha256_file(manifest_path),
    }
    opened: list[str] = []

    def guarded_dataset(path: Path, **kwargs: object) -> object:
        opened.append(Path(path).name)
        assert Path(path).name == "train.jsonl"
        assert kwargs["expected_sha256"] == sha256_file(train)
        return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]

    monkeypatch.setattr(corpus_module, "JsonlIterableDataset", guarded_dataset)
    support, mapping = corpus_module._training_categorical_support(
        integration,
        acceptance,
        _config(),
    )
    assert opened == ["train.jsonl"]
    assert support["status"] == "supported"
    assert support["training_label_counts"] == {"0": 4, "1": 4}
    assert support["validation_labels_read"] is False
    assert support["test_labels_read"] is False
    assert support["test_partition_opened_or_hashed"] is False
    assert mapping == ()
    assert "NEVER_INSPECT_TEST_SECRET" not in canonical_json(support)


def test_high_cardinality_continuous_preflight_has_bounded_label_free_output(
    tmp_path: Path,
) -> None:
    canonical, qc = _canonical_fixture(
        tmp_path / "input",
        supported_rows=5_000,
        unique_continuous_labels=True,
    )
    binding = corpus_module._verify_canonical_corpus(canonical, qc)
    key = "default::default_binding_kd_exact_supported"
    preflight = corpus_module._preflight_task(
        binding,
        key,
        binding.task_datasets[key],
        _config(),
    )
    encoded = canonical_json(preflight)
    assert preflight["rows_scanned"] == 5_000
    assert preflight["task_semantics"]["label_kind"] == "continuous_exact"
    assert preflight["label_access"]["label_columns_read"] == []
    assert "partition_label_counts" not in preflight
    assert "training_label_counts" not in preflight
    assert len(encoded) < 5_000


def test_verifier_rejects_top_only_structural_label_access_tampering(
    tmp_path: Path,
) -> None:
    canonical, qc = _canonical_fixture(tmp_path / "input")
    output = tmp_path / "corpus-readiness"
    materialize_corpus_readiness_bundle(canonical, qc, output, _config())
    acceptance_path = output / "acceptance.json"
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    integrated = next(record for record in acceptance["tasks"] if record["integration_materialized"] is True)
    integrated["preflight"]["label_access"]["training_labels_read"] = True
    integrated["preflight"]["label_access"]["validation_labels_read"] = True
    acceptance_path.write_text(json.dumps(acceptance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(
        ValueError,
        match="diverges from its inventory-bound|inspected or persisted",
    ):
        verify_corpus_readiness_bundle(output)


def test_categorical_cardinality_cap_fails_closed_with_suppressed_counts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    integration = tmp_path / "integration"
    partitions = integration / "model_ready" / "partitions"
    partitions.mkdir(parents=True)
    train = partitions / "train.jsonl"
    examples = [
        {
            "record_id": f"train-{index}",
            "label": {"value": float(index), "text": ""},
        }
        for index in range(33)
    ]
    train.write_text(
        "".join(json.dumps(example, sort_keys=True) + "\n" for example in examples),
        encoding="utf-8",
    )
    manifest = {
        "partitions": {
            "train": {
                "path": "train.jsonl",
                "sha256": sha256_file(train),
                "record_count": len(examples),
            }
        }
    }
    manifest_path = partitions / "partition_manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
    acceptance = {
        "task_semantics": {"label_kind": "categorical"},
        "physical_partition_manifest_file_sha256_external": sha256_file(manifest_path),
    }
    opened: list[str] = []

    def guarded_dataset(path: Path, **kwargs: object) -> object:
        opened.append(Path(path).name)
        assert kwargs["expected_sha256"] == sha256_file(train)
        return examples

    monkeypatch.setattr(corpus_module, "JsonlIterableDataset", guarded_dataset)
    support, mapping = corpus_module._training_categorical_support(
        integration,
        acceptance,
        _config(),
    )
    assert opened == ["train.jsonl"]
    assert support["status"] == "insufficient_training_label_support"
    assert support["skip_reasons"] == ["training_categorical_cardinality_exceeds_declared_cap"]
    assert support["training_rows_scanned"] == 33
    assert support["training_scan_complete"] is False
    assert support["training_label_counts"] == {}
    assert support["capped_training_label_counts"] == {}
    assert support["validation_labels_read"] is False
    assert support["test_labels_read"] is False
    assert mapping == ()
    assert len(canonical_json(support)) < 2_000
