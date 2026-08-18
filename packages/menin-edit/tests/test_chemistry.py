import pytest

from menin_edit.chemistry import (
    canonicalize_smiles,
    fragment_single_cuts,
    join_single_attachment_fragments,
    molecule_id,
    normalize_attachment_fragment,
    validate_product,
)


def test_every_returned_single_cut_reconstructs_the_parent():
    parent = "CCOc1ccccc1"

    contexts = fragment_single_cuts(
        parent,
        min_core_heavy_atoms=6,
        max_variable_heavy_atoms=4,
    )

    assert contexts
    assert all(
        join_single_attachment_fragments(context.core_smiles, context.variable_smiles)
        == canonicalize_smiles(parent)
        for context in contexts
    )
    assert len({(item.core_smiles, item.variable_smiles) for item in contexts}) == len(contexts)


def test_attachment_normalization_is_stable_and_requires_one_dummy():
    normalized = normalize_attachment_fragment("[13*]CC")

    assert normalized == normalize_attachment_fragment(normalized)
    assert "*:1" in normalized
    with pytest.raises(ValueError, match="exactly one attachment"):
        normalize_attachment_fragment("[*:1]CC[*:2]")
    with pytest.raises(ValueError, match="Invalid attachment"):
        normalize_attachment_fragment("not-smiles")


def test_join_replaces_fragment_and_validation_enforces_edit_bounds():
    product = join_single_attachment_fragments("c1ccc(O[*:1])cc1", "C[*:1]")

    assert product == canonicalize_smiles("COc1ccccc1")
    accepted = validate_product(
        "CCOc1ccccc1",
        product,
        max_changed_heavy_atoms=2,
        min_parent_similarity=0.0,
    )
    assert accepted.valid
    assert accepted.heavy_atom_delta == -1
    assert accepted.reason == "ok"

    identity = validate_product(
        product,
        product,
        max_changed_heavy_atoms=2,
        min_parent_similarity=0.0,
    )
    assert not identity.valid
    assert identity.reason == "identity_edit"

    too_large = validate_product(
        "COc1ccccc1",
        "CCCCOc1ccccc1",
        max_changed_heavy_atoms=1,
        min_parent_similarity=0.0,
    )
    assert not too_large.valid
    assert too_large.reason == "heavy_atom_delta_exceeded"


def test_canonical_identity_is_invariant_to_smiles_order():
    assert canonicalize_smiles("OCC") == canonicalize_smiles("CCO")
    assert molecule_id("OCC") == molecule_id("CCO")
