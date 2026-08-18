from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from menin_discovery.platform_features import (
    FEATURE_REGISTRY_VERSION,
    FeatureRegistry,
    categorical_imbalance_summary,
    deterministic_descriptor_frame,
    length_and_truncation_analysis,
    prepare_molecular_graph,
    prepare_protein_sequence,
    prepare_text,
    scan_text_for_label_leakage,
    tokenize_smiles,
)


def test_feature_registry_fails_closed_and_requires_conditional_lineage() -> None:
    registry = FeatureRegistry()
    registry.assert_model_inputs(["canonical_smiles", "protein_sequence"])

    with pytest.raises(ValueError, match="forbidden"):
        registry.assert_model_inputs(["label_value"])
    with pytest.raises(KeyError, match="not registered"):
        registry.assert_model_inputs(["mystery_feature"])
    with pytest.raises(ValueError, match="missing_conditional_lineage"):
        registry.assert_model_inputs(["cross_endpoint_observation"])

    registry.assert_model_inputs(
        ["cross_endpoint_observation"],
        conditional_lineage={
            "cross_endpoint_observation": {
                "admission_approved": True,
                "lineage_digest": "a" * 64,
            }
        },
    )
    assert len(registry.digest()) == 64
    assert {"name", "leakage_role", "required_lineage"}.issubset(registry.frame().columns)


def test_smiles_tokenization_and_graph_are_deterministic() -> None:
    smiles = "C[C@H](O)Cl"
    assert "".join(tokenize_smiles(smiles)) == smiles
    first = prepare_molecular_graph(smiles)
    second = prepare_molecular_graph(smiles)
    assert first == second
    assert first["valid"] is True
    assert len(first["edges"]) % 2 == 0
    assert len(first["structure_sha256"]) == 64

    invalid = prepare_molecular_graph("not-a-smiles")
    assert invalid["valid"] is False
    assert invalid["nodes"] == []


def test_sequence_chunking_reports_invalid_and_never_silently_truncates() -> None:
    prepared = prepare_protein_sequence("ACDEFGHIKLMN", max_length=5, policy="chunk", overlap=1)
    assert prepared.valid
    assert prepared.original_length == 12
    assert len(prepared.chunks) == 3
    assert prepared.truncated is False
    assert "".join(prepared.chunks).startswith("ACDEF")

    right = prepare_protein_sequence("ACDEFGH", max_length=4, policy="right", overlap=0)
    assert right.chunks == ("ACDE",)
    assert right.truncated

    with pytest.raises(ValueError, match="exceeds"):
        prepare_protein_sequence("ACDEFGH", max_length=4, policy="error", overlap=0)

    invalid = prepare_protein_sequence("ACD?EF", max_length=10, policy="chunk", overlap=1)
    assert not invalid.valid
    assert invalid.invalid_characters == ("?",)
    assert invalid.chunks == ()


def test_text_preparation_flags_label_like_content_and_chunks() -> None:
    text = "Measured hERG IC50 = 12 nM after incubation. " + "word " * 20
    findings = scan_text_for_label_leakage(text)
    assert {item["kind"] for item in findings} >= {"endpoint_numeric_value", "numeric_bioactivity_unit"}
    prepared = prepare_text(text, max_tokens=8, overlap=2, chunk_long_text=True)
    assert len(prepared.chunks) > 1
    assert not prepared.label_like_scan_clean
    assert not prepared.default_model_input_admitted
    assert not prepared.truncated


def test_text_missing_and_overlap_contracts_fail_cleanly() -> None:
    assert prepare_text(pd.NA).normalized_text == ""
    with pytest.raises(ValueError, match="non-negative"):
        prepare_text("safe text", max_tokens=8, overlap=-1)
    clean = prepare_text("safe descriptive text")
    assert clean.label_like_scan_clean
    assert not clean.default_model_input_admitted


def test_length_analysis_quantifies_hard_truncation() -> None:
    result = length_and_truncation_analysis(
        ["CC", "CCCC", "CCCCCC"],
        kind="characters",
        candidate_max_lengths=[3, 5],
    )
    at_three = result.set_index("candidate_max_length").loc[3]
    assert at_three["n_affected"] == 2
    assert at_three["total_units_lost_if_hard_truncated"] == 4
    assert np.isclose(at_three["fraction_affected"], 2 / 3)


def test_descriptor_artifact_has_lineage_and_invalid_flag() -> None:
    records = pd.DataFrame(
        {
            "molecule_id": ["m1", "m2"],
            "standardized_smiles": ["CCO", "not-a-smiles"],
        }
    )
    result = deterministic_descriptor_frame(records)
    assert result["molecule_id"].tolist() == ["m1", "m2"]
    assert result["invalid_structure"].tolist() == [0.0, 1.0]
    assert set(result["feature_registry_version"]) == {FEATURE_REGISTRY_VERSION}
    assert result["input_structure_sha256"].str.len().eq(64).all()


def test_categorical_imbalance_summary_preserves_missing() -> None:
    summary = categorical_imbalance_summary(["a", "a", "b", None])
    assert summary["n"] == 4
    assert summary["counts"] == {"__MISSING__": 1, "a": 2, "b": 1}
    assert summary["majority_fraction"] == 0.5
