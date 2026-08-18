from pathlib import Path

import pandas as pd
import pytest

from menin_edit.chemistry import canonicalize_smiles, normalize_attachment_fragment
from menin_edit.edits import EditLibrary
from menin_edit.evidence import EvidenceIndex


def _write_pairs(path: Path) -> None:
    pd.DataFrame(
        [
            {
                "structure_id_a": "A-TRAIN",
                "structure_id_b": "B-TRAIN",
                "core_smiles": "c1ccc(O[*:1])cc1",
                "variable_fragment_a": "CC[*:1]",
                "variable_fragment_b": "C[*:1]",
                "p_activity_a": 6.0,
                "p_activity_b": 7.0,
                "split_role": "train",
                "source_scope": "public",
                "evidence_context_grade": "same_assay",
            },
            {
                "structure_id_a": "A-DEV",
                "structure_id_b": "B-DEV",
                "core_smiles": "c1ccc(O[*:1])cc1",
                "variable_fragment_a": "[*:7]CC",
                "variable_fragment_b": "[*:9]C",
                "p_activity_a": 6.5,
                "p_activity_b": 8.0,
                "split_role": "development",
                "source_scope": "public",
                "evidence_context_grade": "same_series",
            },
            {
                "structure_id_a": "A-TEST",
                "structure_id_b": "B-TEST",
                "core_smiles": "c1ccc(O[*:1])cc1",
                "variable_fragment_a": "CC[*:1]",
                "variable_fragment_b": "C[*:1]",
                "p_activity_a": 1.0,
                "p_activity_b": 10.0,
                "split_role": "test",
                "source_scope": "public",
                "evidence_context_grade": "same_assay",
            },
        ]
    ).to_csv(path, index=False)


def test_public_pairs_create_bidirectional_leakage_filtered_rules(tmp_path):
    pair_path = tmp_path / "pairs.csv"
    _write_pairs(pair_path)

    library = EditLibrary.from_public_pairs(pair_path, minimum_support=2)
    ethyl = normalize_attachment_fragment("CC[*:1]")
    methyl = normalize_attachment_fragment("C[*:1]")
    forward = library.rules_for(ethyl)
    reverse = library.rules_for(methyl)

    assert len(forward) == len(reverse) == 1
    assert forward[0].target_fragment == methyl
    assert reverse[0].target_fragment == ethyl
    assert forward[0].support_count == reverse[0].support_count == 2
    assert forward[0].endpoint_mean_deltas["menin_biochemical_pIC50"] == pytest.approx(1.25)
    assert reverse[0].endpoint_mean_deltas["menin_biochemical_pIC50"] == pytest.approx(-1.25)
    assert forward[0].endpoint_std_deltas["menin_biochemical_pIC50"] == pytest.approx(0.25)
    assert {row.split_role for row in library.evidence} == {"train", "development"}
    assert library.manifest() == {
        "rule_count": 2,
        "evidence_count": 4,
        "endpoint_count": 1,
        "endpoints": ["menin_biochemical_pIC50"],
        "bidirectional": True,
        "edit_scope": "single_cut_observed_transformations",
    }


def test_supported_rule_generates_a_valid_product_and_honors_visited(tmp_path):
    pair_path = tmp_path / "pairs.csv"
    _write_pairs(pair_path)
    library = EditLibrary.from_public_pairs(pair_path, minimum_support=2)
    parent = "CCOc1ccccc1"
    expected = canonicalize_smiles("COc1ccccc1")

    generated = library.enumerate(
        parent,
        min_core_heavy_atoms=7,
        max_changed_heavy_atoms=3,
        min_parent_similarity=0.0,
        candidates_per_node=10,
    )

    shortened = next(edit for edit in generated if edit.product_smiles == expected)
    assert shortened.rule.source_fragment == normalize_attachment_fragment("CC[*:1]")
    assert shortened.rule.target_fragment == normalize_attachment_fragment("C[*:1]")
    assert shortened.parent_smiles == parent
    assert shortened.heavy_atom_delta == -1
    assert 0 <= shortened.parent_similarity < 1

    without_visited = library.enumerate(
        parent,
        min_core_heavy_atoms=7,
        max_changed_heavy_atoms=3,
        min_parent_similarity=0.0,
        candidates_per_node=10,
        visited_smiles={expected},
    )
    assert expected not in {edit.product_smiles for edit in without_visited}


def test_evidence_lookup_prioritizes_exact_context_and_preserves_direction(tmp_path):
    pair_path = tmp_path / "pairs.csv"
    _write_pairs(pair_path)
    library = EditLibrary.from_public_pairs(pair_path, minimum_support=2)
    rule = library.rules_for("CC[*:1]")[0]

    summary = EvidenceIndex(library).lookup(
        rule.rule_id,
        "menin_biochemical_pIC50",
        query_core_smiles="c1ccc(O[*:1])cc1",
    )

    assert summary.evidence_grade == "exact_context"
    assert summary.support_count == 2
    assert summary.mean_observed_delta == pytest.approx(1.25)
    assert {row.structure_id_a for row in summary.records} == {"A-TRAIN", "A-DEV"}
    assert all(row.observed_delta > 0 for row in summary.records)


def test_minimum_support_can_remove_the_entire_library(tmp_path):
    pair_path = tmp_path / "pairs.csv"
    _write_pairs(pair_path)

    library = EditLibrary.from_public_pairs(pair_path, minimum_support=3)

    assert library.rules == ()
    assert library.evidence == ()


def test_public_private_merge_pools_endpoints_without_counting_one_pair_twice(tmp_path):
    pair_path = tmp_path / "pairs.csv"
    _write_pairs(pair_path)
    public = EditLibrary.from_public_pairs(pair_path, minimum_support=2)
    private_rows = pd.DataFrame(
        [
            {
                "structure_id_a": "PRIVATE-A",
                "structure_id_b": "PRIVATE-B",
                "core_smiles": "c1ccc(O[*:1])cc1",
                "source_fragment": "CC[*:1]",
                "target_fragment": "C[*:1]",
                "endpoint": endpoint,
                "delta": delta,
                "split_role": "development",
                "source_scope": "private",
                "evidence_grade": "same_series",
            }
            for endpoint, delta in (("herg_pIC50", -0.4), ("mv411_cellular_pIC50", 0.7))
        ]
    )
    private = EditLibrary.from_long_pairs(private_rows)
    private_rule = private.rules_for("CC[*:1]")[0]
    assert private_rule.support_count == 1
    assert private_rule.endpoint_support == {
        "herg_pIC50": 1.0,
        "mv411_cellular_pIC50": 1.0,
    }

    merged = EditLibrary.merge(public, private)
    rule = merged.rules_for("CC[*:1]")[0]
    assert rule.support_count == 3
    assert set(rule.endpoint_mean_deltas) == {
        "menin_biochemical_pIC50",
        "herg_pIC50",
        "mv411_cellular_pIC50",
    }
    assert {row.source_scope for row in merged.evidence_for(rule.rule_id)} == {
        "public",
        "private",
    }
