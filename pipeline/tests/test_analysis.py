import copy
import math

import numpy as np
import pandas as pd
import pytest
from menin_discovery.analysis import (
    _as_bool,
    assign_similarity_clusters,
    build_prospective_selection_plan,
    identify_activity_cliffs,
    medicinal_chemistry_profiles,
    pareto_ranks,
    prioritize_candidates,
    reference_compound_coverage,
)
from menin_discovery.chemistry import standardize_smiles
from menin_discovery.features import RDKIT_AVAILABLE
from menin_discovery.settings import load_settings, validate_settings


def _analysis_settings() -> dict:
    return {
        "herg": {
            "primary_endpoint": "IC50",
            "primary_assay_family": "electrophysiology_functional",
        },
        "modeling": {"fingerprint_bits": 2048, "fingerprint_radius": 2},
        "analysis": {
            "fingerprint_bits": 2048,
            "fingerprint_radius": 2,
            "medicinal_chemistry": {
                "alert_catalogs": ["PAINS", "BRENK"],
                "property_windows": {
                    "mol_wt": {"min": 0, "max": 1_000},
                    "logp": {"min": -10, "max": 10},
                    "tpsa": {"min": 0, "max": 500},
                },
            },
            "prioritization": {
                "strong_pactivity": 7.0,
                "potency_desirability_lower": 6.0,
                "potency_desirability_upper": 9.0,
                "maximum_activity_range_log10": 1.0,
                "maximum_property_violations": 1,
                "require_no_pains": True,
                "lower_herg_probability": 0.30,
                "high_herg_probability": 0.70,
            },
        },
    }


def _population(structures: list[tuple[str, str, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "structure_id": [value[0] for value in structures],
            "standardized_smiles": [value[1] for value in structures],
            "standard_inchi_key": [""] * len(structures),
            "p_activity_median": [value[2] for value in structures],
        }
    )


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="chemical intelligence requires RDKit")
def test_known_rdkit_descriptors_and_pains_brenk_alerts():
    population = _population(
        [
            ("ETOH", "CCO", 7.0),
            ("RHOD", "O=C1NC(=S)SC1", 7.0),
            ("QUIN", "O=C1C=CC(=O)C=C1", 7.0),
            ("CAT", "Oc1ccccc1O", 7.0),
        ]
    )
    profiles, _molecules, _achiral, _chiral = medicinal_chemistry_profiles(
        population, settings=_analysis_settings()
    )
    profiles = profiles.set_index("structure_id")

    ethanol = profiles.loc["ETOH"]
    assert ethanol["mol_wt"] == pytest.approx(46.069, abs=1e-3)
    assert ethanol["exact_mol_wt"] == pytest.approx(46.041865, abs=1e-6)
    assert ethanol["logp"] == pytest.approx(-0.0014, abs=1e-4)
    assert ethanol["tpsa"] == pytest.approx(20.23, abs=1e-2)
    assert ethanol["h_bond_donors"] == 1
    assert ethanol["h_bond_acceptors"] == 1
    assert ethanol["heavy_atom_count"] == 3
    assert ethanol["pains_alert_count"] == 0
    assert ethanol["brenk_alert_count"] == 0

    assert profiles.loc["RHOD", "pains_alerts"] == "rhod_sat_A(33)"
    assert profiles.loc["RHOD", "brenk_alerts"] == "Thiocarbonyl_group"
    assert profiles.loc["QUIN", "pains_alerts"] == "quinone_A(370)"
    assert profiles.loc["QUIN", "brenk_alerts"] == "chinone_1"
    assert profiles.loc["CAT", "pains_alerts"] == "catechol_A(92)"
    assert profiles.loc["CAT", "brenk_alerts"] == "catechol"


def test_boolean_parsing_is_strict_for_false_strings_and_missing_values():
    values = pd.Series([True, False, "TRUE", "False", 1, 0, "yes", "no", None, np.nan])
    assert _as_bool(values).tolist() == [
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        False,
        False,
    ]


def test_out_of_domain_herg_prediction_receives_no_safety_credit():
    profiles = pd.DataFrame(
        {
            "structure_id": ["IN", "OOD"],
            "herg_inside_applicability_domain": ["true", "False"],
            "predicted_herg_blocker_probability": [0.20, 0.01],
            "p_activity_median": [8.0, 8.0],
            "p_activity_min": [8.0, 8.0],
            "qed": [0.5, 0.5],
            "property_desirability": [1.0, 1.0],
            "n_exact_measurements": [2, 2],
            "n_sources": [2, 2],
            "activity_range_log10": [0.0, 0.0],
            "invalid_structure": [0, 0],
            "is_activity_heterogeneous": [False, False],
            "property_window_violation_count": [0, 0],
            "pains_alert_count": [0, 0],
        }
    )
    # A nonmatching observation is enough to exercise the ordinary merge path
    # while leaving both candidates without observed hERG evidence.
    pk = pd.DataFrame(
        {
            "structure_id": ["UNRELATED"],
            "admet_endpoint": ["solubility"],
            "species": ["human"],
            "matrix": ["buffer"],
        }
    )
    priorities, _trace, gaps = prioritize_candidates(
        profiles,
        observed_herg=pd.DataFrame(),
        pk=pk,
        settings=_analysis_settings(),
    )
    priorities = priorities.set_index("structure_id")
    gaps = gaps.set_index("structure_id")

    assert priorities.loc["IN", "herg_evidence_status"] == "predicted_lower_concern"
    assert priorities.loc["IN", "herg_desirability"] == pytest.approx(0.8)
    assert bool(priorities.loc["IN", "herg_prediction_used"])
    assert pd.notna(priorities.loc["IN", "complete_evidence_rank"])

    outside = priorities.loc["OOD"]
    assert outside["herg_evidence_status"] == "unknown_outside_applicability_domain"
    assert pd.isna(outside["herg_probability_used_for_decision"])
    assert not bool(outside["herg_prediction_used"])
    assert pd.isna(outside["herg_desirability"])
    assert pd.isna(outside["complete_evidence_score"])
    assert pd.isna(outside["complete_evidence_pareto_rank"])
    assert pd.isna(outside["complete_evidence_rank"])
    assert priorities.loc["IN", "experimental_followup_tier"].startswith("priority_1")
    assert outside["experimental_followup_tier"].startswith("priority_2")
    assert bool(gaps.loc["OOD", "requires_herg_measurement"])


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="chemical intelligence requires RDKit")
def test_scaffold_series_and_butina_cluster_ids_are_row_order_invariant():
    structures = [
        ("N1", "COc1ccc2ccccc2c1", 9.0),
        ("N2", "Oc1ccc2ccccc2c1", 8.0),
        ("N3", "Clc1ccc2ccccc2c1", 8.0),
        ("N4", "Cc1ccc2ccccc2c1", 8.9),
        ("N5", "CCOc1ccc2ccccc2c1", 6.0),
        ("N6", "c1ccc2ccccc2c1", 7.0),
        ("L1", "CCO", 5.0),
        ("L2", "CCCO", 5.0),
        ("L3", "CCCCO", 5.0),
    ]

    def analyze(population: pd.DataFrame):
        profiles, _molecules, achiral, _chiral = medicinal_chemistry_profiles(
            population, settings=_analysis_settings()
        )
        members, summary = assign_similarity_clusters(profiles, achiral, similarity_threshold=0.35)
        profile_map = profiles.set_index("structure_id")[["series_id", "series_size"]].sort_index()
        member_map = members.set_index("structure_id")[
            [
                "similarity_cluster_id",
                "similarity_cluster_size",
                "similarity_cluster_representative",
            ]
        ].sort_index()
        return profile_map, member_map, summary

    population = _population(structures)
    first_profiles, first_members, first_summary = analyze(population)
    second_profiles, second_members, second_summary = analyze(
        population.sample(frac=1.0, random_state=41).reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(first_profiles, second_profiles)
    pd.testing.assert_frame_equal(first_members, second_members)
    pd.testing.assert_frame_equal(
        first_summary.sort_values("similarity_cluster_id").reset_index(drop=True),
        second_summary.sort_values("similarity_cluster_id").reset_index(drop=True),
    )

    assert first_profiles.loc[[f"N{index}" for index in range(1, 7)], "series_id"].nunique() == 1
    assert set(first_profiles.loc[["L1", "L2", "L3"], "series_size"]) == {1}
    cluster_memberships = {frozenset(value.split(";")) for value in first_summary["member_structure_ids"]}
    assert cluster_memberships == {
        frozenset({"N1", "N2", "N3", "N4", "N5", "N6"}),
        frozenset({"L1", "L2", "L3"}),
    }


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="chemical intelligence requires RDKit")
def test_butina_converts_similarity_threshold_to_distance_threshold():
    # These two radius-2 Morgan fingerprints have Tanimoto similarity exactly
    # 0.5. They therefore separate at 0.55 and cluster at 0.49.
    population = _population(
        [
            ("METHOXY", "COc1ccc2ccccc2c1", 8.0),
            ("METHYL", "Cc1ccc2ccccc2c1", 8.0),
        ]
    )
    profiles, _molecules, achiral, _chiral = medicinal_chemistry_profiles(
        population, settings=_analysis_settings()
    )
    _members, strict_summary = assign_similarity_clusters(profiles, achiral, similarity_threshold=0.55)
    _members, permissive_summary = assign_similarity_clusters(profiles, achiral, similarity_threshold=0.49)
    assert len(strict_summary) == 2
    assert len(permissive_summary) == 1
    assert permissive_summary.loc[0, "cluster_size"] == 2


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="chemical intelligence requires RDKit")
def test_activity_cliff_finds_the_expected_pair_once():
    population = _population(
        [
            ("N1", "COc1ccc2ccccc2c1", 9.0),
            ("N5", "CCOc1ccc2ccccc2c1", 6.0),
            ("N4", "Cc1ccc2ccccc2c1", 8.9),
            ("L1", "CCO", 2.0),
        ]
    )
    profiles, _molecules, achiral, chiral = medicinal_chemistry_profiles(
        population, settings=_analysis_settings()
    )
    cliffs = identify_activity_cliffs(
        profiles,
        achiral,
        chiral,
        similarity_threshold=0.55,
        minimum_delta_pactivity=2.0,
        contexts={},
    )
    assert len(cliffs) == 1
    cliff = cliffs.iloc[0]
    assert (cliff["structure_id_a"], cliff["structure_id_b"]) == ("N1", "N5")
    assert cliff["higher_potency_structure_id"] == "N1"
    assert cliff["absolute_delta_pactivity"] == pytest.approx(3.0)
    assert cliff["achiral_morgan_tanimoto"] == pytest.approx(0.5769230769230769)
    assert cliff["evidence_context_grade"] == "cross_context"


def test_pareto_dominance_ties_missing_values_and_eligibility():
    # Both columns are already oriented so larger is better.
    values = np.asarray(
        [
            [9.0, 0.2],  # A
            [8.0, 0.8],  # B
            [7.0, 0.7],  # C
            [9.0, 0.8],  # D
            [8.5, 0.9],  # E
            [10.0, 0.99],  # F, explicitly ineligible
            [np.nan, 0.5],  # G, incomplete
            [9.0, 0.8],  # H, exact tie with D
        ]
    )
    eligible = np.asarray([True, True, True, True, True, False, True, True])
    ranks = pareto_ranks(values, eligible)
    assert ranks[[3, 4, 7]].tolist() == [1.0, 1.0, 1.0]
    assert ranks[[0, 1]].tolist() == [2.0, 2.0]
    assert ranks[2] == 3.0
    assert math.isnan(ranks[5])
    assert math.isnan(ranks[6])


def test_prospective_selection_is_deterministic_diverse_and_keeps_cliff_pairs():
    structure_ids = ["A", "B", "C", "D", "E", "F"]
    candidates = pd.DataFrame(
        {
            "structure_id": structure_ids,
            "standardized_smiles": ["CCO", "CCN", "CCC", "CCCl", "CCF", "CCBr"],
            "p_activity_median": [9.0, 5.0, 8.5, 6.0, 8.0, 5.0],
            "series_id": ["S1", "S1", "S2", "S2", "S1", "S1"],
            "series_size": [4, 4, 2, 2, 4, 4],
            "qed": [0.5] * 6,
            "property_window_violation_count": [0] * 6,
            "herg_evidence_status": ["unknown_missing_prediction"] * 6,
            "pk_observation_count": [0] * 6,
            "pk_endpoint_count": [0] * 6,
            "local_novelty_achiral": [0.2] * 6,
            "experimental_followup_tier": ["priority_5_context_only"] * 6,
            "discovery_rank_without_safety": list(range(1, 7)),
            "complete_evidence_rank": list(range(1, 7)),
            "invalid_structure": [0] * 6,
        }
    )
    cliffs = pd.DataFrame(
        {
            "structure_id_a": ["A", "E", "C"],
            "structure_id_b": ["B", "F", "D"],
            "shared_assay_id": [True, True, True],
            "absolute_delta_pactivity": [4.0, 3.0, 2.5],
            "achiral_morgan_tanimoto": [0.90, 0.88, 0.85],
        }
    )
    settings = _analysis_settings()
    settings["analysis"]["prospective_selection"] = {
        "enabled": True,
        "maximum_per_scaffold_series": 2,
        # An odd quota also verifies that the final available slot is not
        # filled by an unpaired cliff member.
        "quotas": {
            "potent_safety_gap": 0,
            "liability_characterization": 0,
            "novel_scaffold_exploration": 0,
            "activity_cliff_confirmation": 5,
            "negative_control": 0,
            "pk_bridge": 0,
        },
    }

    first, first_summary = build_prospective_selection_plan(candidates, cliffs, settings=settings)
    second, second_summary = build_prospective_selection_plan(
        candidates.sample(frac=1.0, random_state=17).reset_index(drop=True),
        cliffs.sample(frac=1.0, random_state=23).reset_index(drop=True),
        settings=settings,
    )
    pd.testing.assert_frame_equal(first, second)
    pd.testing.assert_frame_equal(first_summary, second_summary)

    assert first["structure_id"].is_unique
    assert set(first["selection_category"]) == {"activity_cliff_confirmation"}
    assert set(first["structure_id"]) == {"A", "B", "C", "D"}
    assert first.groupby("series_id")["structure_id"].size().max() <= 2
    assert set(first.groupby("paired_cliff_id")["structure_id"].size()) == {2}
    selected_pairs = {
        frozenset(group["structure_id"]) for _pair_id, group in first.groupby("paired_cliff_id")
    }
    assert selected_pairs == {frozenset({"A", "B"}), frozenset({"C", "D"})}
    cliff_summary = first_summary.set_index("selection_category").loc["activity_cliff_confirmation"]
    assert cliff_summary["requested_quota"] == 5
    assert cliff_summary["selected_structures"] == 4
    assert cliff_summary["shortfall"] == 1


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="chemical intelligence requires RDKit")
def test_reference_coverage_distinguishes_primary_and_any_measurement_stably():
    ethanol = standardize_smiles("CCO", require_rdkit=True)
    propanol = standardize_smiles("CCCO", require_rdkit=True)
    settings = _analysis_settings()
    settings["curation"] = {"strip_salts": True, "canonicalize_tautomers": False}

    def reference_record(name: str, cid: int, structure) -> dict:
        return {
            "name": name,
            "pubchem_cid": cid,
            "pubchem_inchi_key": structure.standard_inchi_key,
            "pubchem_isomeric_smiles": structure.standardized_smiles,
            "regulatory_status": "synthetic test reference",
            "source_checked_at": "2026-07-14",
            "approval_context": "unit test only",
            "pubchem_url": f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}",
            "regulatory_url": "https://example.test/reference",
        }

    settings["analysis"]["reference_compounds"] = [
        reference_record("exact-primary", 1, ethanol),
        reference_record("any-measurement-only", 2, propanol),
    ]
    # P-A and P-B are deliberately equivalent fingerprints. The nearest-neighbor
    # tie must resolve by structure ID, not input order.
    primary_population = pd.DataFrame(
        {
            "structure_id": ["P-B", "P-A"],
            "standardized_smiles": ["OCC", "CCO"],
            "standard_inchi_key": [ethanol.standard_inchi_key] * 2,
            "p_activity_median": [6.0, 8.0],
        }
    )
    all_measurements = pd.DataFrame(
        {
            "standard_inchi_key": [ethanol.standard_inchi_key, propanol.standard_inchi_key],
            "endpoint": ["IC50", "Ki"],
            "assay_family": ["biochemical_binding", "cellular_functional"],
        }
    )

    def coverage(population: pd.DataFrame) -> pd.DataFrame:
        profiles, _molecules, achiral, chiral = medicinal_chemistry_profiles(population, settings=settings)
        return reference_compound_coverage(
            profiles,
            all_measurements,
            achiral,
            chiral,
            settings=settings,
        )

    first = coverage(primary_population)
    second = coverage(primary_population.iloc[::-1].reset_index(drop=True))
    pd.testing.assert_frame_equal(first, second)
    result = first.set_index("name")

    exact = result.loc["exact-primary"]
    assert bool(exact["has_exact_primary_task_structure"])
    assert bool(exact["has_any_public_menin_measurement"])
    assert exact["nearest_primary_structure_id"] == "P-A"
    assert exact["maximum_primary_achiral_tanimoto"] == pytest.approx(1.0)

    any_only = result.loc["any-measurement-only"]
    assert not bool(any_only["has_exact_primary_task_structure"])
    assert bool(any_only["has_any_public_menin_measurement"])
    assert any_only["public_menin_endpoints"] == "Ki"
    assert any_only["public_menin_assay_families"] == "cellular_functional"
    assert any_only["nearest_primary_structure_id"] == "P-A"
    assert any_only["nearest_primary_p_activity"] == pytest.approx(8.0)


def test_duplicate_and_malformed_reference_compounds_fail_validation():
    duplicate = load_settings()
    duplicate["analysis"]["reference_compounds"].append(
        copy.deepcopy(duplicate["analysis"]["reference_compounds"][0])
    )
    with pytest.raises(ValueError, match="names and PubChem CIDs must be unique"):
        validate_settings(duplicate)

    malformed = load_settings()
    malformed["analysis"]["reference_compounds"] = [{"name": "incomplete"}]
    with pytest.raises(ValueError, match=r"reference_compounds\[0\]\.pubchem_cid"):
        validate_settings(malformed)


@pytest.mark.parametrize(
    "override_text, message",
    [
        ("analysis:\n  enabled: maybe\n", "analysis.enabled"),
        (
            "analysis:\n  clustering:\n    similarity_threshold: 0\n",
            "clustering.similarity_threshold",
        ),
        (
            "analysis:\n  clustering:\n    similarity_threshold: .nan\n",
            "must be finite",
        ),
        (
            "analysis:\n  activity_cliffs:\n    minimum_delta_pactivity: 0\n",
            "minimum_delta_pactivity",
        ),
        (
            "analysis:\n  medicinal_chemistry:\n    alert_catalogs: [PAINS, PAINS]\n",
            "non-empty and unique",
        ),
        (
            "analysis:\n  medicinal_chemistry:\n    property_windows:\n      mol_wt: {min: 500, max: 100}\n",
            "requires min < max",
        ),
        (
            "analysis:\n  prioritization:\n    lower_herg_probability: 0.8\n    high_herg_probability: 0.7\n",
            "hERG probability thresholds",
        ),
        (
            "analysis:\n  prioritization:\n    weights:\n      potency: -0.1\n",
            "weights must be non-negative",
        ),
        (
            "analysis:\n  prioritization:\n    weights:\n      mystery: 0.1\n",
            "Unknown analysis prioritization weights",
        ),
        (
            "analysis:\n  prospective_selection:\n    enabled: maybe\n",
            "prospective_selection.enabled",
        ),
        (
            "analysis:\n  prospective_selection:\n    maximum_per_scaffold_series: 0\n",
            "maximum_per_scaffold_series",
        ),
        (
            "analysis:\n  prospective_selection:\n    quotas:\n      unknown_arm: 1\n",
            "Unknown prospective selection quotas",
        ),
        (
            "analysis:\n  prospective_selection:\n    quotas:\n      activity_cliff_confirmation: -1\n",
            "activity_cliff_confirmation must be non-negative",
        ),
    ],
)
def test_invalid_analysis_configuration_fails_early(tmp_path, override_text: str, message: str):
    override = tmp_path / "invalid-analysis.yaml"
    override.write_text(override_text, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_settings(override)


def test_valid_boundary_analysis_configuration_and_empty_result_schemas(tmp_path):
    override = tmp_path / "valid-analysis.yaml"
    override.write_text(
        "analysis:\n"
        "  series_minimum_size: 1\n"
        "  clustering:\n"
        "    similarity_threshold: 1\n"
        "  activity_cliffs:\n"
        "    similarity_threshold: 1\n"
        "    minimum_delta_pactivity: 0.001\n"
        "    connectivity_minimum_delta_pactivity: 0.001\n",
        encoding="utf-8",
    )
    settings = load_settings(override)
    assert settings["analysis"]["clustering"]["similarity_threshold"] == 1

    empty_profiles = pd.DataFrame(
        columns=[
            "structure_id",
            "series_id",
            "p_activity_median",
            "standardized_smiles",
            "connectivity_key",
        ]
    )
    members, clusters = assign_similarity_clusters(empty_profiles, [], similarity_threshold=0.65)
    cliffs = identify_activity_cliffs(
        empty_profiles,
        [],
        [],
        similarity_threshold=0.8,
        minimum_delta_pactivity=2.0,
        contexts={},
    )
    ranks = pareto_ranks(np.empty((0, 2)))

    assert members.empty
    assert {
        "structure_id",
        "series_id",
        "similarity_cluster_id",
        "similarity_cluster_size",
        "similarity_cluster_representative",
    } == set(members.columns)
    assert clusters.empty
    assert {
        "similarity_cluster_id",
        "cluster_size",
        "representative_structure_id",
    } == set(clusters.columns)
    assert cliffs.empty
    assert {
        "structure_id_a",
        "structure_id_b",
        "absolute_delta_pactivity",
        "achiral_morgan_tanimoto",
    }.issubset(cliffs.columns)
    assert ranks.shape == (0,)
