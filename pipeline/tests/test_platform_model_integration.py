from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import pytest
from menin_discovery import platform_model_integration as integration_module
from menin_discovery.platform_model_integration import (
    DiagnosticConfig,
    TaskIntegrationConfig,
    materialize_capped_diagnostic_bundle,
    materialize_task_integration_bundle,
)
from menin_discovery.platform_pretraining import file_sha256


def _task_frame(
    *,
    label_kind: str = "continuous_exact",
    observation_kind: str = "experimental_raw",
    n_rows: int = 240,
) -> pd.DataFrame:
    molecules = [
        "CCO",
        "CCN",
        "CCC",
        "CCCl",
        "CCBr",
        "CC(=O)O",
        "CC(C)O",
        "CC(C)N",
        "c1ccccc1",
        "c1ccncc1",
        "C1CCCCC1",
        "COC",
    ]
    if label_kind == "categorical":
        values: list[float | None] = [float(index % 2) for index in range(n_rows)]
        relations = ["="] * n_rows
        lower_bounds: list[float | None] = [None] * n_rows
        label_unit = "binary"
        task_type = "binary_classification"
        task_id = "herg_blocker_binary_test"
        evidence_domain = "cardiotoxicity"
        endpoint = "herg_blocker_class"
        assay_family = "herg_functional"
    elif label_kind == "continuous_censored":
        values = [None] * n_rows
        relations = [">"] * n_rows
        lower_bounds = [float(10 + index % 17) for index in range(n_rows)]
        label_unit = "nM"
        task_type = "quantitative_regression"
        task_id = "binding_kd_censored_test"
        evidence_domain = "binding_affinity"
        endpoint = "Kd"
        assay_family = "biochemical_binding"
    else:
        values = [float(4 + (index % 31) / 10) for index in range(n_rows)]
        relations = ["="] * n_rows
        lower_bounds = [None] * n_rows
        label_unit = "pKd"
        task_type = "quantitative_regression"
        task_id = "binding_pkd_exact_test"
        evidence_domain = "binding_affinity"
        endpoint = "pKd"
        assay_family = "biochemical_binding"
    derived = observation_kind == "derived"
    return pd.DataFrame(
        {
            "observation_id": [f"obs-{index:05d}" for index in range(n_rows)],
            "molecule_id": [f"mol-{index:05d}" for index in range(n_rows)],
            "protein_id": [f"protein-{index % 7}" for index in range(n_rows)],
            "canonical_target_id": [f"target-{index % 5}" for index in range(n_rows)],
            "assay_id": [f"assay-{index % 11}" for index in range(n_rows)],
            "source_id": [f"source-{index % 3}" for index in range(n_rows)],
            "snapshot_id": ["snapshot-test"] * n_rows,
            "source_record_id": [f"source-record-{index:05d}" for index in range(n_rows)],
            "access_class": ["public_redistributable"] * n_rows,
            "default_task_eligible": [not derived] * n_rows,
            "sensitivity_task_eligible": [derived] * n_rows,
            "inclusion_status": ["included"] * n_rows,
            "standardized_smiles": [molecules[index % len(molecules)] for index in range(n_rows)],
            "sequence": ["ACDEFGHIKLMNPQRSTVWY"] * n_rows,
            "task_id": [task_id] * n_rows,
            "task_type": [task_type] * n_rows,
            "label_kind": [label_kind] * n_rows,
            "label_value": values,
            "label_text": [""] * n_rows,
            "label_relation": relations,
            "label_lower_bound": lower_bounds,
            "label_upper_bound": [None] * n_rows,
            "label_unit": [label_unit] * n_rows,
            "observation_kind": [observation_kind] * n_rows,
            "label_lineage_digest": ["d" * 64 if derived else ""] * n_rows,
            "evidence_domain": [evidence_domain] * n_rows,
            "endpoint": [endpoint] * n_rows,
            "assay_family": [assay_family] * n_rows,
            "document_year": [2010 + index % 15 for index in range(n_rows)],
        }
    )


def _manifest_bound_task_directory(root: Path, frame: pd.DataFrame) -> Path:
    build_root = root / "full_chembl37"
    task_directory = build_root / "tasks" / "default" / str(frame["task_id"].iat[0])
    task_directory.mkdir(parents=True)
    boundaries = (0, len(frame) // 3, 2 * len(frame) // 3, len(frame))
    artifacts: list[dict[str, object]] = []
    for index, (start, stop) in enumerate(zip(boundaries[:-1], boundaries[1:], strict=True)):
        path = task_directory / f"part-{index:05d}.parquet"
        frame.iloc[start:stop].to_parquet(path, index=False)
        artifacts.append(
            {
                "path": path.name,
                "relative_path": path.relative_to(build_root).as_posix(),
                "rows": stop - start,
                "sha256": file_sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    (build_root / "build_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "test_manifest_v1",
                "shard_artifacts": list(reversed(artifacts)),
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return task_directory


def _integration_config(
    *,
    task_eligibility_mode: Literal["default", "derived_sensitivity"] = "default",
) -> TaskIntegrationConfig:
    return TaskIntegrationConfig(
        split_batch_size=37,
        serialization_batch_size=29,
        task_eligibility_mode=task_eligibility_mode,
        loader_batch_size=4,
        loader_maximum_batches=2,
    )


def _assert_all_json_declares_no_training(root: Path) -> None:
    paths = list(root.rglob("*.json"))
    assert paths
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["substantive_training_started"] is False, path


def _assert_complete_component_inventory(root: Path, acceptance_name: str) -> None:
    acceptance_path = root / acceptance_name
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    inventory = acceptance["component_inventory"]
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != acceptance_path
    }
    assert set(inventory) == actual
    for relative, entry in inventory.items():
        path = root / relative
        assert entry["relative_path"] == relative
        assert entry["size_bytes"] == path.stat().st_size
        assert entry["sha256"] == file_sha256(path)


def test_task_integration_bundle_is_transactional_bound_and_immutable(tmp_path: Path) -> None:
    task_directory = _manifest_bound_task_directory(tmp_path, _task_frame())
    output = tmp_path / "regression_integration"

    result = materialize_task_integration_bundle(
        task_directory,
        output,
        _integration_config(),
    )

    assert result["task_semantics"]["resolved_task_family"] == "regression"
    assert result["source_record_count"] == 240
    assert sum(result["physical_partition_counts"].values()) == 240
    assert result["loader_smoke_status"] == "passed"
    assert result["loader_smoke_examples"] <= 32
    assert result["claim_readiness"].startswith("not_claim_ready")
    assert result["model_ready_combined_corpus_retained"] is False
    assert result["bundle_layout"]["combined_model_ready_corpus"] is None
    assert file_sha256(output / "acceptance.json") == result["acceptance_file_sha256_external"]
    assert not (output / "model_ready" / "molecule_hash_stream_v1.jsonl").exists()
    assert not (output / "model_ready" / "molecule_hash_stream_v1.jsonl.manifest.json").exists()
    assert (output / "model_ready" / "combined_serialization_receipt.json").is_file()
    assert {
        path.relative_to(output / "model_ready" / "partitions").as_posix()
        for path in (output / "model_ready" / "partitions").glob("*.jsonl")
    } == {"train.jsonl", "validation.jsonl", "test.jsonl"}
    partition_manifest = json.loads(
        (output / "model_ready" / "partitions" / "partition_manifest.json").read_text(encoding="utf-8")
    )
    assert "source_path" not in partition_manifest
    assert partition_manifest["transient_combined_source"]["retained_in_final_bundle"] is False
    assert not list(tmp_path.glob(".regression_integration.building-*"))
    _assert_all_json_declares_no_training(output)
    _assert_complete_component_inventory(output, "acceptance.json")
    assert "model_ready/combined_serialization_receipt.json" in result["component_inventory"]
    assert not any(
        relative
        in {
            "model_ready/molecule_hash_stream_v1.jsonl",
            "model_ready/molecule_hash_stream_v1.jsonl.manifest.json",
        }
        for relative in result["component_inventory"]
    )
    lengths = json.loads((output / "readiness" / "length_inventory.json").read_text(encoding="utf-8"))
    assert lengths["partitions"]["test"]["status"] == "not_inspected_locked_test"
    assert lengths["test_partition_opened_after_routing"] is False

    with pytest.raises(FileExistsError, match="Immutable output already exists"):
        materialize_task_integration_bundle(
            task_directory,
            output,
            _integration_config(),
        )


def test_task_integration_auto_classification_and_explicit_derived_mode(
    tmp_path: Path,
) -> None:
    classification_directory = _manifest_bound_task_directory(
        tmp_path / "classification",
        _task_frame(label_kind="categorical"),
    )
    classification_output = tmp_path / "classification_integration"
    classification = materialize_task_integration_bundle(
        classification_directory,
        classification_output,
        _integration_config(),
    )
    assert classification["task_semantics"]["resolved_task_family"] == "classification"

    derived_directory = _manifest_bound_task_directory(
        tmp_path / "derived",
        _task_frame(observation_kind="derived"),
    )
    refused_output = tmp_path / "derived_refused"
    with pytest.raises(ValueError, match="Default integration refuses derived labels"):
        materialize_task_integration_bundle(
            derived_directory,
            refused_output,
            _integration_config(),
        )
    assert not refused_output.exists()

    derived_output = tmp_path / "derived_sensitivity_integration"
    derived = materialize_task_integration_bundle(
        derived_directory,
        derived_output,
        _integration_config(task_eligibility_mode="derived_sensitivity"),
    )
    assert derived["task_eligibility_mode"] == "derived_sensitivity"
    assert derived["task_semantics"]["observation_kind"] == "derived"
    _assert_all_json_declares_no_training(derived_output)

    diagnostic_output = tmp_path / "derived_diagnostics"
    diagnostic = materialize_capped_diagnostic_bundle(
        derived_output,
        diagnostic_output,
        integration_acceptance_sha256=derived["acceptance_file_sha256_external"],
    )
    assert diagnostic["status"] == "skipped"
    assert diagnostic["reason"] == "skipped_derived_sensitivity_not_primary_diagnostic"
    _assert_complete_component_inventory(diagnostic_output, "diagnostic_status.json")


def test_capped_regression_diagnostics_do_not_resolve_or_open_test(
    tmp_path: Path,
) -> None:
    task_directory = _manifest_bound_task_directory(tmp_path, _task_frame())
    integration_output = tmp_path / "integration"
    integration = materialize_task_integration_bundle(
        task_directory,
        integration_output,
        _integration_config(),
    )
    # If diagnostics resolve, hash, or open this file, the run must fail.  The
    # immutable manifest still binds its original digest, which is intentionally
    # irrelevant to train/validation-only diagnostics.
    test_partition = integration_output / "model_ready" / "partitions" / "test.jsonl"
    test_partition.write_text("this is deliberately invalid and digest-mismatched\n", encoding="utf-8")

    diagnostic_output = tmp_path / "regression_diagnostics"
    diagnostic = materialize_capped_diagnostic_bundle(
        integration_output,
        diagnostic_output,
        integration_acceptance_sha256=integration["acceptance_file_sha256_external"],
        config=DiagnosticConfig(
            maximum_train_examples=48,
            maximum_validation_examples=24,
            fingerprint_bits=128,
        ),
    )

    assert diagnostic["status"] == "completed_diagnostic_only"
    assert diagnostic["opened_partitions"] == ["train", "validation"]
    assert diagnostic["test_partition_opened"] is False
    selected = pd.read_parquet(diagnostic_output / "features" / "selected_rows.parquet")
    assert set(selected["partition"]) == {"train", "validation"}
    assert len(selected.query("partition == 'train'")) <= 48
    assert len(selected.query("partition == 'validation'")) <= 24
    assert (diagnostic_output / "baselines" / "dummy_median.joblib").is_file()
    assert (diagnostic_output / "baselines" / "ridge_fixed.joblib").is_file()
    metadata = json.loads(
        (diagnostic_output / "baselines" / "baseline_metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["config"]["class_imbalance"] == "none"
    assert diagnostic["selection"]["rank_bits"] == 256
    assert {
        "features/selected_rows.parquet",
        "features/descriptors.parquet",
        "features/fingerprints_rdkit_morgan_r2_128.npz",
        "features/feature_config.json",
        "features/feature_failure_summary.csv",
        "features/feature_manifest.json",
        "features/numeric_target_leakage_scan.csv",
        "baselines/metrics.csv",
        "baselines/validation_predictions.parquet",
        "baselines/validation_error_analysis.parquet",
        "baselines/label_permutation_control.json",
        "baselines/identifier_hash_control.json",
        "baselines/dummy_median.joblib",
        "baselines/ridge_fixed.joblib",
        "baselines/baseline_metadata.json",
    }.issubset(diagnostic["component_inventory"])
    _assert_all_json_declares_no_training(diagnostic_output)
    _assert_complete_component_inventory(diagnostic_output, "acceptance.json")


def test_capped_binary_diagnostics_are_fixed_and_validation_only(tmp_path: Path) -> None:
    task_directory = _manifest_bound_task_directory(
        tmp_path,
        _task_frame(label_kind="categorical"),
    )
    integration_output = tmp_path / "integration"
    integration = materialize_task_integration_bundle(
        task_directory,
        integration_output,
        _integration_config(),
    )
    diagnostic_output = tmp_path / "classification_diagnostics"
    diagnostic = materialize_capped_diagnostic_bundle(
        integration_output,
        diagnostic_output,
        integration_acceptance_sha256=integration["acceptance_file_sha256_external"],
        config=DiagnosticConfig(
            maximum_train_examples=80,
            maximum_validation_examples=40,
            fingerprint_bits=128,
        ),
    )

    assert diagnostic["target"]["task_family"] == "classification"
    assert diagnostic["model_selection_performed"] is False
    assert diagnostic["hyperparameter_sweep_performed"] is False
    metrics = pd.read_csv(diagnostic_output / "baselines" / "metrics.csv")
    assert set(metrics["model_name"]) == {"dummy_prior", "logistic_fixed"}
    assert set(metrics["evaluation_partition"]) == {"validation"}
    imbalance = json.loads(
        (diagnostic_output / "baselines" / "class_imbalance_and_prevalence.json").read_text(encoding="utf-8")
    )
    assert imbalance["threshold_policy"] == "fixed_0.5_diagnostic_not_selected"
    assert "baselines/class_imbalance_and_prevalence.json" in diagnostic["component_inventory"]
    _assert_all_json_declares_no_training(diagnostic_output)
    _assert_complete_component_inventory(diagnostic_output, "acceptance.json")


def test_censored_diagnostics_skip_without_imputation_or_test_access(tmp_path: Path) -> None:
    task_directory = _manifest_bound_task_directory(
        tmp_path,
        _task_frame(label_kind="continuous_censored"),
    )
    integration_output = tmp_path / "integration"
    integration = materialize_task_integration_bundle(
        task_directory,
        integration_output,
        _integration_config(),
    )
    diagnostic_output = tmp_path / "censored_diagnostics"
    diagnostic = materialize_capped_diagnostic_bundle(
        integration_output,
        diagnostic_output,
        integration_acceptance_sha256=integration["acceptance_file_sha256_external"],
    )

    assert diagnostic["status"] == "skipped"
    assert diagnostic["reason"] == "skipped_requires_censored_model_no_midpoint_imputation"
    assert diagnostic["test_partition_opened"] is False
    assert not (diagnostic_output / "features").exists()
    assert not (diagnostic_output / "baselines").exists()
    _assert_all_json_declares_no_training(diagnostic_output)
    _assert_complete_component_inventory(diagnostic_output, "diagnostic_status.json")


def test_integration_never_accesses_test_lockbox_after_routing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_directory = _manifest_bound_task_directory(tmp_path, _task_frame())
    output = tmp_path / "guarded_integration"
    routing_complete = False
    original_partition = integration_module._partition_model_ready_jsonl
    original_file_sha256 = integration_module.file_sha256
    original_dataset = integration_module.JsonlIterableDataset

    def guarded_partition(*args: Any, **kwargs: Any) -> dict[str, Any]:
        nonlocal routing_complete
        result = original_partition(*args, **kwargs)
        routing_complete = True
        return result

    def guarded_file_sha256(path: Path) -> str:
        if routing_complete and Path(path).name == "test.jsonl":
            raise AssertionError("test lockbox was hashed after physical routing")
        return original_file_sha256(path)

    def guarded_dataset(*args: Any, **kwargs: Any) -> Any:
        path = Path(args[0] if args else kwargs["path"])
        if routing_complete and path.name == "test.jsonl":
            raise AssertionError("test lockbox was opened or iterated after physical routing")
        return original_dataset(*args, **kwargs)

    monkeypatch.setattr(integration_module, "_partition_model_ready_jsonl", guarded_partition)
    monkeypatch.setattr(integration_module, "file_sha256", guarded_file_sha256)
    monkeypatch.setattr(integration_module, "JsonlIterableDataset", guarded_dataset)

    result = materialize_task_integration_bundle(
        task_directory,
        output,
        _integration_config(),
    )
    assert result["test_lockbox_after_physical_routing"]["open_calls"] == 0
    assert result["test_lockbox_after_physical_routing"]["hash_calls"] == 0


def test_failed_diagnostic_transaction_leaves_no_partial_bundle(tmp_path: Path) -> None:
    frame = _task_frame(label_kind="categorical")
    frame["label_value"] = 1.0
    task_directory = _manifest_bound_task_directory(tmp_path, frame)
    integration_output = tmp_path / "integration"
    integration = materialize_task_integration_bundle(
        task_directory,
        integration_output,
        _integration_config(),
    )
    diagnostic_output = tmp_path / "failed_diagnostics"

    with pytest.raises(ValueError, match="must contain both encoded classes"):
        materialize_capped_diagnostic_bundle(
            integration_output,
            diagnostic_output,
            integration_acceptance_sha256=integration["acceptance_file_sha256_external"],
            config=DiagnosticConfig(
                maximum_train_examples=60,
                maximum_validation_examples=30,
                fingerprint_bits=128,
            ),
        )

    assert not diagnostic_output.exists()
    assert not list(tmp_path.glob(".failed_diagnostics.building-*"))


def test_manifest_binding_detects_part_mutation_before_output(tmp_path: Path) -> None:
    task_directory = _manifest_bound_task_directory(tmp_path, _task_frame())
    part = task_directory / "part-00001.parquet"
    original = part.read_bytes()
    part.write_bytes(original + b"tamper")
    output = tmp_path / "rejected_integration"

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        materialize_task_integration_bundle(
            task_directory,
            output,
            _integration_config(),
        )
    assert not output.exists()


def test_integration_refuses_final_data_while_sibling_build_is_provisional(
    tmp_path: Path,
) -> None:
    task_directory = _manifest_bound_task_directory(tmp_path, _task_frame())
    provisional = tmp_path / ".full_chembl37.building"
    provisional.mkdir()
    output = tmp_path / "rejected_while_building"

    with pytest.raises(ValueError, match="provisional sibling build"):
        materialize_task_integration_bundle(
            task_directory,
            output,
            _integration_config(),
        )
    assert not output.exists()


def test_acceptance_digest_must_bind_diagnostic_input(tmp_path: Path) -> None:
    task_directory = _manifest_bound_task_directory(tmp_path, _task_frame())
    integration_output = tmp_path / "integration"
    materialize_task_integration_bundle(
        task_directory,
        integration_output,
        _integration_config(),
    )
    output = tmp_path / "rejected_diagnostics"

    with pytest.raises(ValueError, match="digest does not match"):
        materialize_capped_diagnostic_bundle(
            integration_output,
            output,
            integration_acceptance_sha256=hashlib.sha256(b"wrong").hexdigest(),
        )
    assert not output.exists()
