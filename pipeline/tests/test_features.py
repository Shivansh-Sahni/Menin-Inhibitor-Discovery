import hashlib

import menin_discovery.features as features
import numpy as np
import pytest
from menin_discovery.features import (
    SmilesFeatureTransformer,
    fingerprint_matrix,
    nearest_neighbor_tanimoto,
    scaffold_key,
    smiles_descriptors,
)


def test_hashed_feature_fallback_is_deterministic_and_finite():
    smiles = ["CCO", "CCN", "c1ccccc1", ""]
    transformer = SmilesFeatureTransformer(n_features=64, backend="hashed")
    first = transformer.fit_transform(smiles)
    second = transformer.transform(smiles)
    assert transformer.backend_ == "hashed_smiles"
    assert first.shape == second.shape == (4, 64 + len(smiles_descriptors(smiles).columns))
    assert np.allclose(first.toarray(), second.toarray())
    assert np.isfinite(first.toarray()).all()
    assert transformer.feature_metadata_["fallback_used"]


def test_binary_fingerprints_and_tanimoto_detect_identical_structure():
    matrix, backend = fingerprint_matrix(["CCO", "CCN"], backend="hashed", n_bits=128)
    assert backend == "hashed_smiles"
    assert set(np.unique(matrix.data)) <= {1.0}
    similarity, neighbor, resolved = nearest_neighbor_tanimoto(
        ["CCO"], ["CCN", "CCO"], backend="hashed", n_bits=128
    )
    assert resolved == "hashed_smiles"
    assert similarity[0] == 1.0
    assert neighbor[0] == 1


def test_scaffold_key_is_stable_and_groups_duplicates():
    first, first_method = scaffold_key("c1ccccc1CCO")
    second, second_method = scaffold_key("c1ccccc1CCO")
    assert first == second
    assert first_method == second_method
    if features.RDKIT_AVAILABLE:
        assert first == "c1ccccc1"
        assert first_method == "bemis_murcko"


def test_scaffold_key_real_bad_bond_stereo_is_deterministic_across_rdkit_versions():
    smiles = r"N/C(=N\N=C\c1ccc(O)c(O)c1)c1nonc1N"

    first = scaffold_key(smiles)
    second = scaffold_key(smiles)

    expected_proxy = f"exact:{hashlib.sha256(smiles.encode()).hexdigest()[:20]}"
    assert first == second
    if first[1] == "exact_smiles_proxy_rdkit_exception":
        assert first[0] == expected_proxy
    else:
        # Newer RDKit releases can sanitize this formerly failing stereo form.
        assert first[1] == "bemis_murcko"
        assert first[0]


@pytest.mark.parametrize("failure_stage", ["parse", "scaffold", "canonicalization"])
def test_scaffold_key_rdkit_exceptions_fail_closed_to_exact_proxy(monkeypatch, failure_stage):
    if failure_stage != "parse" and not features.RDKIT_AVAILABLE:
        pytest.skip("RDKit-specific scaffold exception test requires RDKit")

    smiles = "CCN"

    def fail(*_args, **_kwargs):
        raise RuntimeError(f"synthetic {failure_stage} failure")

    if failure_stage == "parse":
        monkeypatch.setattr(features, "_mol_from_smiles", fail)
    elif failure_stage == "scaffold":
        monkeypatch.setattr(features.MurckoScaffold, "MurckoScaffoldSmiles", fail)
    else:
        monkeypatch.setattr(
            features.MurckoScaffold,
            "MurckoScaffoldSmiles",
            lambda **_kwargs: "",
        )
        monkeypatch.setattr(features.Chem, "MolToSmiles", fail)

    expected = f"exact:{hashlib.sha256(smiles.encode()).hexdigest()[:20]}"
    assert scaffold_key(smiles) == (expected, "exact_smiles_proxy_rdkit_exception")


def test_scaffold_key_does_not_mask_unexpected_programming_errors(monkeypatch):
    def fail(_smiles):
        raise TypeError("synthetic programming error")

    monkeypatch.setattr(features, "_mol_from_smiles", fail)
    with pytest.raises(TypeError, match="synthetic programming error"):
        scaffold_key("CCN")


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        ("exact_smiles_proxy", True),
        ("exact_smiles_proxy_rdkit_exception", True),
        ("exact_smiles_proxy_future_named_failure", True),
        ("exact_smiles_proxying", False),
        ("bemis_murcko", False),
        ("bemis_murcko_with_exact_acyclic", False),
    ],
)
def test_exact_smiles_proxy_method_predicate_is_suffix_aware(method, expected):
    assert features.is_exact_smiles_proxy_method(method) is expected
