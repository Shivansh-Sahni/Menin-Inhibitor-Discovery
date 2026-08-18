from __future__ import annotations

import pandas as pd
import pytest
from menin_discovery.platform_herg_mmp_analysis import (
    HergMmpAnalysisError,
    _assert_mmp_split_exclusivity,
    _pair_registry,
)


def test_pair_registry_is_split_contained_and_label_free() -> None:
    membership = pd.DataFrame(
        {
            "structure_id": ["A", "B", "C", "D"],
            "model_split": ["train", "train", "test", "test"],
        }
    )
    structures = pd.DataFrame(
        {
            "structure_id": ["A", "B", "C", "D"],
            "standardized_smiles": [
                "CCOc1ccccc1",
                "CCNc1ccccc1",
                "CCOc1ccncc1",
                "CCNc1ccncc1",
            ],
        }
    )
    rows, _, failures = _pair_registry(membership, structures)
    assert failures == 0
    for row in rows:
        assert row["pair_definition_uses_labels"] is False
        assert "pic50" not in row
        split_ids = set(membership.loc[membership["model_split"].eq(row["model_split"]), "structure_id"])
        assert {row["structure_id_a"], row["structure_id_b"]} <= split_ids


def test_scaffold_crossing_is_rejected() -> None:
    registry = pd.DataFrame(
        {
            "structure_id_a": ["A", "C"],
            "structure_id_b": ["B", "D"],
            "model_split": ["train", "test"],
        }
    )
    structures = pd.DataFrame(
        {
            "structure_id": ["A", "B", "C", "D"],
            "scaffold_group_id": ["G-X", "G-1", "G-X", "G-2"],
        }
    )
    with pytest.raises(HergMmpAnalysisError, match="scaffold"):
        _assert_mmp_split_exclusivity(registry, structures)
