from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest
from menin_discovery.platform_pretraining import (
    CollatorConfig,
    FineTuningConfig,
    JsonlDataset,
    JsonlIterableDataset,
    MultimodalCollator,
    attach_fixed_split,
    build_training_vocabulary,
    candidate_model_registry,
    checkpoint_contract,
    estimate_model_ready_materialization_memory,
    estimate_runtime_scenario,
    estimate_training_resources,
    file_sha256,
    loader_smoke_test,
    materialize_static_readiness_registries,
    model_ready_examples_from_task_view,
    observed_contrastive_pairs,
    serialize_model_ready_jsonl,
    serialize_model_ready_jsonl_streaming,
    streaming_loader_smoke_test,
    training_readiness_check,
    validate_model_ready_example,
    validate_resume_checkpoint,
)
from menin_discovery.platform_splits import SplitConfig, stream_hash_group_split_manifest


def _canonical_task_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "observation_id": ["o1", "o2", "o3", "o4"],
            "molecule_id": ["m1", "m2", "m3", "m4"],
            "protein_id": ["p1"] * 4,
            "assay_id": ["a1"] * 4,
            "source_id": ["s1"] * 4,
            "snapshot_id": ["snap"] * 4,
            "source_record_id": ["source-o1", "source-o2", "source-o3", "source-o4"],
            "access_class": ["public_redistributable"] * 4,
            "default_task_eligible": [True] * 4,
            "inclusion_status": ["included"] * 4,
            "standardized_smiles": ["CCO", "CCN", "CCC", "CCCl"],
            "sequence": ["ACDEFG"] * 4,
            "task_id": ["binary"] * 4,
            "task_type": ["binary_classification"] * 4,
            "label_kind": ["categorical"] * 4,
            "label_value": [1, 0, 1, 0],
            "label_text": [""] * 4,
            "label_relation": ["="] * 4,
            "label_lower_bound": [None] * 4,
            "label_upper_bound": [None] * 4,
            "label_unit": ["binary"] * 4,
            "observation_kind": ["curated_assertion"] * 4,
            "evidence_domain": ["herg"] * 4,
            "endpoint": ["herg_blocker_class"] * 4,
            "assay_family": ["herg_functional"] * 4,
            "document_year": [2020, 2021, 2022, 2023],
        }
    )


def _examples() -> list[dict]:
    return model_ready_examples_from_task_view(_canonical_task_frame())


def _streaming_task_frame(n: int = 40) -> pd.DataFrame:
    base = _canonical_task_frame()
    frame = pd.concat([base] * ((n + len(base) - 1) // len(base)), ignore_index=True).iloc[:n].copy()
    frame["observation_id"] = [f"stream-o{index:04d}" for index in range(n)]
    frame["source_record_id"] = [f"stream-source-{index:04d}" for index in range(n)]
    return frame


def _manifest_bound_task_directory(root: Path, frame: pd.DataFrame) -> Path:
    build_root = root / "full_chembl37"
    task_directory = build_root / "tasks" / "default" / "binary_classification"
    task_directory.mkdir(parents=True)
    artifacts: list[dict[str, object]] = []
    for index, part_frame in enumerate((frame.iloc[:17], frame.iloc[17:])):
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


def _collator(examples: list[dict], *, truncation_policy: str = "error") -> MultimodalCollator:
    smiles = build_training_vocabulary(
        [item["inputs"]["smiles"] for item in examples], modality="smiles", fitted_partition="train"
    )
    proteins = build_training_vocabulary(
        [item["inputs"]["protein_sequence"] for item in examples],
        modality="protein",
        fitted_partition="train",
    )
    text = build_training_vocabulary([""], modality="text", fitted_partition="train")
    return MultimodalCollator(
        smiles_vocabulary=smiles,
        protein_vocabulary=proteins,
        text_vocabulary=text,
        config=CollatorConfig(
            max_smiles_tokens=16,
            max_protein_tokens=16,
            max_text_tokens=8,
            truncation_policy=truncation_policy,
            pad_to_multiple_of=4,
        ),
    )


def test_model_ready_adapter_rejects_prediction_and_private_rows() -> None:
    frame = _canonical_task_frame()
    examples = model_ready_examples_from_task_view(frame)
    assert len(examples) == 4
    assert examples[0]["label"]["outcome_kind"] == "curated_assertion"
    validate_model_ready_example(examples[0])

    prediction = frame.copy()
    prediction.loc[0, "observation_kind"] = "prediction"
    with pytest.raises(ValueError, match="Prohibited"):
        model_ready_examples_from_task_view(prediction)

    private = frame.copy()
    private.loc[0, "access_class"] = "private"
    with pytest.raises(ValueError, match="Non-public"):
        model_ready_examples_from_task_view(private)
    legacy_alias = frame.copy()
    legacy_alias.loc[0, "access_class"] = "public"
    with pytest.raises(ValueError, match="Non-public"):
        model_ready_examples_from_task_view(legacy_alias)


def test_model_ready_adapter_requires_homogeneous_explicitly_eligible_task_view() -> None:
    heterogeneous = _canonical_task_frame()
    heterogeneous.loc[0, "task_id"] = "observation-specific-task"
    with pytest.raises(ValueError, match="homogeneous"):
        model_ready_examples_from_task_view(heterogeneous)

    missing_gate = _canonical_task_frame().drop(columns=["default_task_eligible", "inclusion_status"])
    with pytest.raises(ValueError, match="missing columns"):
        model_ready_examples_from_task_view(missing_gate)

    ineligible = _canonical_task_frame()
    ineligible.loc[0, "default_task_eligible"] = False
    with pytest.raises(ValueError, match="Default-ineligible"):
        model_ready_examples_from_task_view(ineligible)

    for status in ("review", "quarantined", ""):
        not_included = _canonical_task_frame()
        not_included.loc[0, "inclusion_status"] = status
        with pytest.raises(ValueError, match="inclusion_status=included"):
            model_ready_examples_from_task_view(not_included)


@pytest.mark.parametrize(
    "column",
    [
        "observation_id",
        "molecule_id",
        "protein_id",
        "assay_id",
        "source_id",
        "snapshot_id",
        "source_record_id",
        "task_id",
        "task_type",
        "label_kind",
        "label_value",
        "label_text",
        "label_relation",
        "label_lower_bound",
        "label_upper_bound",
        "label_unit",
        "observation_kind",
        "access_class",
        "inclusion_status",
        "evidence_domain",
        "endpoint",
        "assay_family",
    ],
)
def test_model_ready_adapter_fails_closed_when_contract_column_is_missing(column: str) -> None:
    with pytest.raises(ValueError, match="missing columns"):
        model_ready_examples_from_task_view(_canonical_task_frame().drop(columns=column))


@pytest.mark.parametrize(
    "column",
    [
        "molecule_id",
        "protein_id",
        "assay_id",
        "source_id",
        "snapshot_id",
        "source_record_id",
        "task_id",
        "task_type",
        "label_kind",
        "label_relation",
        "label_unit",
        "observation_kind",
        "access_class",
        "evidence_domain",
        "endpoint",
        "assay_family",
    ],
)
def test_model_ready_adapter_fails_closed_when_identity_or_semantics_are_blank(column: str) -> None:
    frame = _canonical_task_frame()
    frame.loc[0, column] = ""
    with pytest.raises(ValueError, match="blank|required|homogeneous"):
        model_ready_examples_from_task_view(frame)


def test_model_ready_adapter_blocks_derived_labels_without_explicit_opt_in() -> None:
    frame = _canonical_task_frame()
    frame["observation_kind"] = "derived"
    frame["label_lineage_digest"] = "d" * 64
    with pytest.raises(ValueError, match="disabled by default"):
        model_ready_examples_from_task_view(frame)
    with pytest.raises(ValueError, match="Generic derived-label opt-in"):
        model_ready_examples_from_task_view(frame, allow_derived_labels=True)

    frame["default_task_eligible"] = False
    frame["sensitivity_task_eligible"] = True
    examples = model_ready_examples_from_task_view(
        frame,
        allow_derived_labels=True,
        task_eligibility_mode="derived_sensitivity",
    )
    assert {item["label"]["outcome_kind"] for item in examples} == {"derived"}


def test_derived_sensitivity_mode_requires_every_independent_gate() -> None:
    valid = _canonical_task_frame()
    valid["observation_kind"] = "derived"
    valid["label_lineage_digest"] = "d" * 64
    valid["default_task_eligible"] = False
    valid["sensitivity_task_eligible"] = True

    missing_flag = valid.drop(columns="sensitivity_task_eligible")
    with pytest.raises(ValueError, match="sensitivity_task_eligible"):
        model_ready_examples_from_task_view(
            missing_flag,
            allow_derived_labels=True,
            task_eligibility_mode="derived_sensitivity",
        )

    default_eligible = valid.copy()
    default_eligible["default_task_eligible"] = True
    with pytest.raises(ValueError, match="default task path"):
        model_ready_examples_from_task_view(
            default_eligible,
            allow_derived_labels=True,
            task_eligibility_mode="derived_sensitivity",
        )

    mixed = valid.copy()
    mixed.loc[0, "observation_kind"] = "experimental_raw"
    with pytest.raises(ValueError, match="derived labels only"):
        model_ready_examples_from_task_view(
            mixed,
            allow_derived_labels=True,
            task_eligibility_mode="derived_sensitivity",
        )

    invalid_lineage = valid.copy()
    invalid_lineage["label_lineage_digest"] = "not-a-digest"
    with pytest.raises(ValueError, match="SHA-256"):
        model_ready_examples_from_task_view(
            invalid_lineage,
            allow_derived_labels=True,
            task_eligibility_mode="derived_sensitivity",
        )


def test_categorical_text_and_interval_labels_follow_kind_specific_contract() -> None:
    text_frame = _canonical_task_frame().iloc[[0]].copy()
    text_frame["label_value"] = None
    text_frame["label_text"] = "blocker"
    text_example = model_ready_examples_from_task_view(text_frame)[0]
    assert text_example["label"]["text"] == "blocker"

    interval = _canonical_task_frame().iloc[[0]].copy()
    interval["task_type"] = "regression"
    interval["label_kind"] = "continuous_censored"
    interval["label_value"] = None
    interval["label_relation"] = "interval"
    interval["label_lower_bound"] = 1.0
    interval["label_upper_bound"] = 2.0
    interval["label_unit"] = "nM"
    example = model_ready_examples_from_task_view(interval)[0]
    assert example["label"]["relation"] == "interval"
    assert example["label"]["lower_bound"] == 1.0
    assert example["label"]["upper_bound"] == 2.0


def test_fixed_split_attachment_and_deterministic_jsonl_integrity(tmp_path: Path) -> None:
    examples = _examples()
    manifest = pd.DataFrame(
        {
            "record_id": ["o1", "o2", "o3", "o4"],
            "split": ["train", "train", "validation", "test"],
            "split_name": ["fixed"] * 4,
            "strategy": ["molecule_grouped"] * 4,
            "group_id": ["m1", "m2", "m3", "m4"],
        }
    )
    attached = attach_fixed_split(examples, manifest)
    output = tmp_path / "examples.jsonl"
    metadata = serialize_model_ready_jsonl(
        attached,
        output,
        source_dataset_sha256="a" * 64,
        split_manifest_sha256="b" * 64,
        build_config={"public_only": True},
    )
    dataset = JsonlDataset(output, expected_sha256=metadata["file_sha256"])
    assert len(dataset) == 4
    assert dataset[0]["record_id"] == "o1"
    assert dataset[-1]["record_id"] == "o4"
    assert len(JsonlDataset(output, partition="train")) == 2
    manifest_payload = json.loads(output.with_suffix(".jsonl.manifest.json").read_text())
    assert manifest_payload["outcome_kind_counts"] == {"curated_assertion": 4}

    output.write_text(output.read_text() + "{}\n")
    with pytest.raises(ValueError, match="digest"):
        JsonlDataset(output, expected_sha256=metadata["file_sha256"])


def test_streaming_split_serialization_is_bounded_and_batch_size_deterministic(
    tmp_path: Path,
) -> None:
    task = _streaming_task_frame()
    task_path = tmp_path / "task.parquet"
    task.to_parquet(task_path, index=False)
    split_path = tmp_path / "split.parquet"
    split_config = SplitConfig(
        name="stream_molecule_v1",
        strategy="molecule_grouped",
        intended_use="new molecule",
        task_type="classification",
    )
    split_metadata = stream_hash_group_split_manifest(
        task_path,
        split_path,
        split_config,
        batch_size=7,
    )
    source_digest = file_sha256(task_path)
    split_digest = file_sha256(split_path)
    split_sidecar_digest = file_sha256(split_path.with_suffix(".parquet.manifest.json"))
    first_path = tmp_path / "first.jsonl"
    second_path = tmp_path / "second.jsonl"
    first = serialize_model_ready_jsonl_streaming(
        task_path,
        split_path,
        first_path,
        source_dataset_sha256=source_digest,
        split_manifest_sha256=split_digest,
        split_sidecar_sha256=split_sidecar_digest,
        build_config={"split": split_config.name, "mode": "default"},
        batch_size=5,
    )
    second = serialize_model_ready_jsonl_streaming(
        task_path,
        split_path,
        second_path,
        source_dataset_sha256=source_digest,
        split_manifest_sha256=split_digest,
        split_sidecar_sha256=split_sidecar_digest,
        build_config={"split": split_config.name, "mode": "default"},
        batch_size=11,
    )
    assert first["file_sha256"] == second["file_sha256"]
    assert first["source_record_count"] == len(task)
    assert first["record_count"] == len(task)
    assert first["bounded_memory"]["maximum_observed_batch_rows"] <= 5
    assert split_metadata["manifest_sha256"] == split_digest
    streamed = JsonlIterableDataset(first_path, expected_sha256=first["file_sha256"])
    assert sum(1 for _ in streamed) == len(task)
    smoke = streaming_loader_smoke_test(
        JsonlIterableDataset(first_path, partition="train"),
        _collator(_examples()),
        batch_size=2,
        maximum_batches=2,
    )
    assert smoke["status"] == "passed"
    assert smoke["performance_evidence"] is False

    split_sidecar_path = split_path.with_suffix(".parquet.manifest.json")
    split_sidecar_path.write_text(
        split_sidecar_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="sidecar digest"):
        serialize_model_ready_jsonl_streaming(
            task_path,
            split_path,
            tmp_path / "tampered-sidecar.jsonl",
            source_dataset_sha256=source_digest,
            split_manifest_sha256=split_digest,
            split_sidecar_sha256=split_sidecar_digest,
            build_config={"split": split_config.name, "mode": "default"},
            batch_size=5,
        )


def test_streaming_serialization_accepts_manifest_bound_partitioned_task(
    tmp_path: Path,
) -> None:
    task = _streaming_task_frame()
    task_directory = _manifest_bound_task_directory(tmp_path, task)
    split_path = tmp_path / "partitioned-split.parquet"
    split_config = SplitConfig(
        name="partitioned_stream_molecule_v1",
        strategy="molecule_grouped",
        intended_use="new molecule across canonical task shards",
        task_type="classification",
    )
    split_metadata = stream_hash_group_split_manifest(
        task_directory,
        split_path,
        split_config,
        batch_size=7,
    )
    output = tmp_path / "partitioned.jsonl"
    metadata = serialize_model_ready_jsonl_streaming(
        task_directory,
        split_path,
        output,
        source_dataset_sha256=split_metadata["source_dataset_sha256"],
        split_manifest_sha256=split_metadata["manifest_sha256"],
        split_sidecar_sha256=split_metadata["sidecar_sha256"],
        build_config={"split": split_config.name, "mode": "default"},
        batch_size=5,
    )
    assert metadata["source_record_count"] == len(task)
    assert metadata["record_count"] == len(task)
    assert metadata["source_dataset"]["input_kind"] == "manifest_bound_directory"
    assert len(metadata["source_dataset"]["parts"]) == 2
    assert [item["record_id"] for item in JsonlIterableDataset(output)] == task["observation_id"].tolist()

    build_manifest = task_directory.parents[2] / "build_manifest.json"
    manifest_payload = json.loads(build_manifest.read_text(encoding="utf-8"))
    manifest_payload["tampered_after_split"] = True
    build_manifest.write_text(json.dumps(manifest_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="Source task digest"):
        serialize_model_ready_jsonl_streaming(
            task_directory,
            split_path,
            tmp_path / "partitioned-tampered.jsonl",
            source_dataset_sha256=split_metadata["source_dataset_sha256"],
            split_manifest_sha256=split_metadata["manifest_sha256"],
            split_sidecar_sha256=split_metadata["sidecar_sha256"],
            build_config={"split": split_config.name, "mode": "default"},
            batch_size=5,
        )


def test_materialization_memory_estimate_labels_full_list_as_prohibited() -> None:
    estimate = estimate_model_ready_materialization_memory(
        _streaming_task_frame(),
        target_row_count=3_100_000,
        sample_rows=8,
    )
    assert estimate["extrapolated_combined_gib"] > 0
    assert estimate["risk"].endswith("prohibited_for_platform_scale")
    assert estimate["required_path"] == "serialize_model_ready_jsonl_streaming"


def test_serializer_rejects_unassigned_excluded_and_invalid_digests(tmp_path: Path) -> None:
    examples = _examples()
    with pytest.raises(ValueError, match="attached fixed split"):
        serialize_model_ready_jsonl(
            examples,
            tmp_path / "unassigned.jsonl",
            source_dataset_sha256="a" * 64,
            split_manifest_sha256="b" * 64,
            build_config={},
        )
    with pytest.raises(ValueError, match="empty"):
        serialize_model_ready_jsonl(
            [],
            tmp_path / "empty.jsonl",
            source_dataset_sha256="a" * 64,
            split_manifest_sha256="b" * 64,
            build_config={},
        )
    excluded = [dict(examples[0])]
    excluded[0]["split"] = {
        "partition": "excluded_unknown_date",
        "split_name": "temporal",
        "strategy": "temporal",
    }
    with pytest.raises(ValueError, match="rejects unassigned/excluded"):
        serialize_model_ready_jsonl(
            excluded,
            tmp_path / "excluded.jsonl",
            source_dataset_sha256="a" * 64,
            split_manifest_sha256="b" * 64,
            build_config={},
        )
    attached = [dict(examples[0])]
    attached[0]["split"] = {
        "partition": "train",
        "split_name": "fixed",
        "strategy": "molecule_grouped",
    }
    with pytest.raises(ValueError, match="SHA-256"):
        serialize_model_ready_jsonl(
            attached,
            tmp_path / "bad_digest.jsonl",
            source_dataset_sha256="not-a-digest",
            split_manifest_sha256="b" * 64,
            build_config={},
        )


def test_vocabulary_fits_train_only_and_collator_reports_shapes() -> None:
    examples = _examples()
    with pytest.raises(ValueError, match="training partition only"):
        build_training_vocabulary(["CCO"], modality="smiles", fitted_partition="validation")
    reserved = build_training_vocabulary(["<PAD> ordinary"], modality="text", fitted_partition="train")
    assert reserved.token_to_id["<PAD>"] == 0
    collator = _collator(examples)
    batch = collator(examples[:2])
    assert batch["smiles_input_ids"].shape[0] == 2
    assert batch["protein_input_ids"].shape[0] == 2
    assert batch["labels"].tolist() == [1.0, 0.0]
    assert batch["truncation_counts"] == {"smiles": 0, "protein": 0, "text": 0}


def test_collator_preserves_boundary_tokens_and_marks_missing_modalities() -> None:
    examples = _examples()
    examples[0]["inputs"]["protein_sequence"] = ""
    examples[0]["inputs"]["protein_sequence_status"] = "missing"
    examples[0]["inputs"]["protein_sequence_failure_reason"] = "sequence_unavailable"
    smiles_vocab = build_training_vocabulary(
        [item["inputs"]["smiles"] for item in examples], modality="smiles", fitted_partition="train"
    )
    protein_vocab = build_training_vocabulary(
        [item["inputs"]["protein_sequence"] for item in examples],
        modality="protein",
        fitted_partition="train",
    )
    text_vocab = build_training_vocabulary([""], modality="text", fitted_partition="train")
    collator = MultimodalCollator(
        smiles_vocabulary=smiles_vocab,
        protein_vocabulary=protein_vocab,
        text_vocabulary=text_vocab,
        config=CollatorConfig(
            max_smiles_tokens=4,
            max_protein_tokens=8,
            max_text_tokens=4,
            truncation_policy="right",
            pad_to_multiple_of=1,
        ),
    )
    batch = collator(examples[:2])
    assert not bool(batch["protein_present_mask"][0])
    assert batch["protein_attention_mask"][0].sum() == 0
    first_smiles_length = int(batch["smiles_attention_mask"][0].sum())
    assert batch["smiles_input_ids"][0, 0] == smiles_vocab.token_to_id["<BOS>"]
    assert batch["smiles_input_ids"][0, first_smiles_length - 1] == smiles_vocab.token_to_id["<EOS>"]


def test_graph_collation_preserves_missing_reason() -> None:
    frame = _canonical_task_frame().iloc[:2].copy()
    frame.loc[0, "standardized_smiles"] = ""
    examples = model_ready_examples_from_task_view(frame, include_graph=True)
    collator = _collator(examples)
    collator.config = CollatorConfig(
        max_smiles_tokens=16,
        max_protein_tokens=16,
        max_text_tokens=8,
        truncation_policy="error",
        pad_to_multiple_of=4,
        include_graph=True,
    )
    batch = collator(examples)
    assert not bool(batch["graph"]["graph_present_mask"][0])
    assert batch["graph"]["graph_failure_reasons"][0] == "smiles_unavailable"
    assert int(batch["graph"]["n_graphs"][0]) == 1
    assert int(batch["graph"]["batch_size"][0]) == 2


def test_observed_contrastive_pairs_never_invent_negative_labels() -> None:
    examples = _examples()
    for example in examples:
        example["split"] = {
            "partition": "train",
            "split_name": "fixed",
            "strategy": "molecule_grouped",
        }
    build = observed_contrastive_pairs(examples, negatives_per_positive=1)
    assert len(build.pairs) == 2
    assert {item["sampling_semantics"] for item in build.pairs} == {
        "both_classes_are_observed_or_curated_assertions"
    }
    derived = _examples()[0]
    derived["split"] = {
        "partition": "train",
        "split_name": "fixed",
        "strategy": "molecule_grouped",
    }
    derived["label"]["outcome_kind"] = "derived"
    derived["label"]["lineage_digest"] = "a" * 64
    filtered = observed_contrastive_pairs([*examples, derived])
    assert filtered.metadata["exclusion_counts"]["non_observed_or_curated_assertion"] == 1
    validation_example = _examples()[0]
    validation_example["split"] = {
        "partition": "validation",
        "split_name": "fixed",
        "strategy": "molecule_grouped",
    }
    with pytest.raises(ValueError, match="train partition"):
        observed_contrastive_pairs([validation_example])


def test_checkpoint_resume_contract_and_training_guard() -> None:
    candidate = next(item for item in candidate_model_registry() if item.key == "chemprop_dmpnn")
    config = FineTuningConfig(
        experiment_name="future_graph_run",
        candidate_key=candidate.key,
        task_ids=("binary",),
        dataset_sha256="a" * 64,
        split_manifest_sha256="b" * 64,
        tokenizer_sha256="c" * 64,
        loss_name="binary_cross_entropy",
        censoring_policy="not_applicable",
        class_weight_policy="natural_prevalence",
        model_selection_metric="average_precision_pr_auc",
        model_selection_direction="maximize",
        precision="fp32",
        parameter_strategy="full",
        lora=None,
    )
    config.validate(candidate)
    contract = checkpoint_contract(config, base_model_revision="none", code_commit="abc123")
    actual = {**contract, "saved_state": contract["required_state"]}
    validate_resume_checkpoint(contract, actual)
    with pytest.raises(ValueError, match="mismatches"):
        validate_resume_checkpoint(contract, {**actual, "dataset_sha256": "d" * 64})

    prohibited = replace(config, substantive_training_authorized=True)
    with pytest.raises(PermissionError, match="prohibits"):
        prohibited.validate(candidate)


def test_resource_and_runtime_estimates_disclose_scenario_status() -> None:
    resources = estimate_training_resources(
        parameter_count=100_000_000,
        trainable_parameter_count=1_000_000,
        precision="bf16",
        batch_size_per_device=4,
        sequence_tokens_per_example=512,
        hidden_size=768,
        layer_count=12,
        gradient_checkpointing=True,
    )
    assert resources["estimated_peak_gib_per_device"] > 0
    assert resources["estimate_type"].startswith("analytic_scenario")
    runtime = estimate_runtime_scenario(
        n_training_examples=1000,
        maximum_epochs=2,
        effective_batch_size=32,
        assumed_examples_per_second_low=2,
        assumed_examples_per_second_high=10,
        checkpoint_count=3,
        checkpoint_size_gib=1.5,
    )
    assert runtime["runtime_hours_range"][0] < runtime["runtime_hours_range"][1]
    assert runtime["estimate_type"].startswith("uncalibrated_scenario")


def test_loader_smoke_is_capped_and_model_readiness_remains_blocked() -> None:
    examples = _examples()
    result = loader_smoke_test(examples, _collator(examples), batch_size=2, maximum_batches=2)
    assert result["status"] == "passed"
    assert result["performance_evidence"] is False
    with pytest.raises(ValueError, match="capped"):
        loader_smoke_test(examples, _collator(examples), batch_size=16, maximum_batches=3)

    candidate = next(item for item in candidate_model_registry() if item.key == "molecule_transformer")
    config = FineTuningConfig(
        experiment_name="future_transformer",
        candidate_key=candidate.key,
        task_ids=("binary",),
        dataset_sha256="a" * 64,
        split_manifest_sha256="b" * 64,
        tokenizer_sha256="c" * 64,
        loss_name="binary_cross_entropy",
        censoring_policy="not_applicable",
        class_weight_policy="natural_prevalence",
        model_selection_metric="average_precision_pr_auc",
        model_selection_direction="maximize",
    )
    readiness = training_readiness_check(
        candidate,
        config,
        available_modalities={"smiles"},
        available_hardware={"device_count": 0, "bf16_supported": False},
    )
    assert readiness["ready_for_substantive_training"] is False
    assert "exact_pretrained_identifier_unresolved" in readiness["blockers"]


def test_readiness_understands_modality_alternatives_and_optional_context() -> None:
    candidate = next(item for item in candidate_model_registry() if item.key == "chai_1_external_evaluation")
    config = FineTuningConfig(
        experiment_name="future_chai_eval",
        candidate_key=candidate.key,
        task_ids=("pose",),
        dataset_sha256="a" * 64,
        split_manifest_sha256="b" * 64,
        tokenizer_sha256="c" * 64,
        loss_name="pose_confidence_regression",
        censoring_policy="not_applicable",
        class_weight_policy="not_applicable",
        model_selection_metric="pose_rmsd",
        model_selection_direction="minimize",
        precision="bf16",
        parameter_strategy="frozen",
        lora=None,
    )
    readiness = training_readiness_check(
        candidate,
        config,
        available_modalities={"molecular_graph", "protein_sequence"},
        available_hardware={"device_count": 1, "bf16_supported": True},
        completed_preflight=candidate.required_preflight,
    )
    assert readiness["configuration_complete"]
    assert not any(str(item).startswith("missing_modality") for item in readiness["blockers"])
    assert readiness["ready_for_substantive_training"] is False

    nesso = next(item for item in candidate_model_registry() if item.key == "nesso_1_external_evaluation")
    assert "mixed potency/affinity-score" in nesso.scientific_role
    assert "never reinterpret" in nesso.scientific_role


def test_static_readiness_registries_are_deterministic_and_never_authorize_training(
    tmp_path: Path,
) -> None:
    feature_directory = tmp_path / "features"
    model_directory = tmp_path / "models"
    first = materialize_static_readiness_registries(
        feature_directory=feature_directory,
        model_directory=model_directory,
        evidence_checked_date="2026-08-04",
    )
    second = materialize_static_readiness_registries(
        feature_directory=feature_directory,
        model_directory=model_directory,
        evidence_checked_date="2026-08-04",
    )
    assert first["manifest"]["sha256"] == second["manifest"]["sha256"]
    assert first["substantive_training_started"] is False
    assert (feature_directory / "feature_registry.csv").is_file()
    for artifact in first["artifacts"].values():
        assert (model_directory / artifact["path"]).resolve().is_file()
    candidate_payload = json.loads(
        (model_directory / "model_candidate_registry.json").read_text(encoding="utf-8")
    )
    assert candidate_payload["training_authorized"] is False
    assert {item["key"] for item in candidate_payload["candidates"]} >= {
        "chai_1_external_evaluation",
        "boltz_2_external_evaluation",
        "nesso_1_external_evaluation",
    }

    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        materialize_static_readiness_registries(
            feature_directory=feature_directory,
            model_directory=model_directory,
            evidence_checked_date="August 4",
        )
