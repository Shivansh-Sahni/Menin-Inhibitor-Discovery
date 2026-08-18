from __future__ import annotations

import math

import numpy as np
import pytest
from menin_discovery.research_local_physics import (
    acid_deprotonated_fraction,
    add_hydrogens_to_heavy_coordinates,
    apply_transform,
    basin_selection_sensitivity,
    generate_diverse_basin_conformers,
    geometry_observables,
    kabsch_transform,
    polybase_macrostate_distribution,
    protonate_at_atom,
    protonation_site_audit,
    rare_state_flux_threshold,
    site_specific_charge_observables,
)
from rdkit import Chem
from rdkit.Chem import AllChem


def test_polybase_macrostate_distribution_matches_stepwise_equilibria() -> None:
    result = polybase_macrostate_distribution([9.0, 7.0], ph=8.0)

    # Relative weights B:BH+:BH2+ are 1:10:1.
    assert result.probabilities == pytest.approx((1 / 12, 10 / 12, 1 / 12))
    assert sum(result.probabilities) == pytest.approx(1.0)
    assert result.neutral_fraction == pytest.approx(1 / 12)
    assert result.mean_positive_charge == pytest.approx(1.0)


def test_common_pka_offset_changes_neutral_fraction_monotonically() -> None:
    lower = polybase_macrostate_distribution([9.7, 6.5], ph=7.4, pka_offset=-1.0)
    nominal = polybase_macrostate_distribution([9.7, 6.5], ph=7.4)
    upper = polybase_macrostate_distribution([9.7, 6.5], ph=7.4, pka_offset=1.0)

    assert lower.neutral_fraction > nominal.neutral_fraction > upper.neutral_fraction


def test_equal_reported_pka_steps_are_not_collapsed() -> None:
    with_repeated_step = polybase_macrostate_distribution(
        [9.73, 6.51, 6.51, 6.22],
        ph=7.4,
    )
    incorrectly_deduplicated = polybase_macrostate_distribution(
        [9.73, 6.51, 6.22],
        ph=7.4,
    )

    assert len(with_repeated_step.probabilities) == 5
    assert len(incorrectly_deduplicated.probabilities) == 4
    assert with_repeated_step.mean_positive_charge > incorrectly_deduplicated.mean_positive_charge


def test_acid_fraction_and_rare_state_threshold_have_expected_limits() -> None:
    assert acid_deprotonated_fraction(7.4, ph=7.4) == pytest.approx(0.5)
    threshold = rare_state_flux_threshold(0.001, target_flux_fraction=0.5)
    assert threshold["required_state_specific_permeability_ratio"] == pytest.approx(999.0)
    assert threshold["required_free_energy_advantage_kcal_mol"] == pytest.approx(
        0.00198720425864083 * 310.0 * math.log(999.0)
    )


def test_kabsch_transform_recovers_rigid_body_alignment() -> None:
    reference = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    angle = math.pi / 3
    rotation_true = np.asarray(
        [
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    mobile = reference @ rotation_true + np.asarray([4.0, -2.0, 3.0])
    rotation, translation, rmsd = kabsch_transform(mobile, reference)

    assert rmsd < 1e-10
    assert apply_transform(mobile, rotation, translation) == pytest.approx(reference)


def test_diverse_basin_selection_is_bounded_and_auditable() -> None:
    basins = generate_diverse_basin_conformers(
        "CCOC(=O)NCCOCC",
        seeds=(101, 202),
        pool_size=8,
        maximum_per_seed=2,
        energy_window_kcal_mol=20.0,
        minimum_heavy_atom_rmsd_angstrom=0.25,
    )

    assert 2 <= len(basins) <= 4
    assert {metadata["seed"] for _mol, metadata in basins} == {101, 202}
    assert all(metadata["mmff94s_delta_from_seed_minimum_kcal_mol"] <= 20.0 for _mol, metadata in basins)
    for mol, metadata in basins:
        coordinates = np.asarray(mol.GetConformer().GetPositions())
        observables = geometry_observables(mol, coordinates)
        assert observables["radius_of_gyration_angstrom"] > 0.0
        assert observables["polar_heavy_atom_sasa_angstrom2"] > 0.0
        assert metadata["selection_semantics"].endswith(
            "not a sampled population or conformer-count estimate"
        )


def test_basin_selection_sensitivity_reuses_declared_pool_grid() -> None:
    rows = basin_selection_sensitivity(
        "CCOC(=O)NCCOCC",
        seeds=(303,),
        maximum_pool_size=8,
        pool_size_prefixes=(4, 8),
        energy_windows_kcal_mol=(5.0, 10.0),
        rmsd_gates_angstrom=(0.2, 0.4),
        maximum_per_seed=2,
    )

    assert {row["pool_size_prefix"] for row in rows} == {4, 8}
    assert {row["energy_window_kcal_mol"] for row in rows} == {5.0, 10.0}
    assert {row["rmsd_gate_angstrom"] for row in rows} == {0.2, 0.4}
    assert all(row["selected_basin_index"] in {1, 2} for row in rows)
    assert all(row["selection_semantics"].startswith("computational gate sensitivity") for row in rows)


def test_protonation_site_audit_excludes_resonance_deactivated_nitrogen() -> None:
    audit = protonation_site_audit("CC(=O)NCCN1CCCCC1")
    by_index = {row["atom_index_zero_based"]: row for row in audit}

    assert by_index[3]["candidate_for_site_ranking"] is False
    assert by_index[3]["exclusion_reason"] == "amide_or_carbamate_resonance_deactivated"
    assert any(row["candidate_for_site_ranking"] and row["site_class"] == "aliphatic_amine" for row in audit)


def test_site_specific_charged_observables_are_finite_and_origin_invariant() -> None:
    smiles = "CN1CCC(CC1)c1cn(C)cn1"
    audit = protonation_site_audit(smiles)
    site = next(
        int(row["atom_index_zero_based"])
        for row in audit
        if row["site_class"] == "aliphatic_amine" and row["candidate_for_site_ranking"]
    )
    protonated = protonate_at_atom(smiles, site)
    embedded = Chem.AddHs(protonated)
    assert AllChem.EmbedMolecule(embedded, randomSeed=444) == 0
    heavy = Chem.RemoveHs(embedded)
    heavy_coordinates = np.asarray(heavy.GetConformer().GetPositions())
    template = add_hydrogens_to_heavy_coordinates(protonated, heavy_coordinates)
    coordinates = np.asarray(template.GetConformer().GetPositions())
    charges = np.zeros(template.GetNumAtoms())
    charges[site] = 0.7
    attached_hydrogen = next(
        atom.GetIdx() for atom in template.GetAtomWithIdx(site).GetNeighbors() if atom.GetAtomicNum() == 1
    )
    charges[attached_hydrogen] = 0.5
    ring_nitrogen = next(
        atom.GetIdx() for atom in template.GetAtoms() if atom.GetAtomicNum() == 7 and atom.GetIsAromatic()
    )
    charges[ring_nitrogen] = -0.2

    original = site_specific_charge_observables(
        template,
        coordinates,
        charges,
        protonated_atom_index_zero_based=site,
    )
    translated = site_specific_charge_observables(
        template,
        coordinates + np.asarray([10.0, -3.0, 7.0]),
        charges,
        protonated_atom_index_zero_based=site,
    )

    assert original["total_atomic_charge"] == pytest.approx(1.0)
    assert original["cation_fragment_sasa_angstrom2"] > 0.0
    assert original["cation_to_edited_ring_centroid_distance_angstrom"] == pytest.approx(
        translated["cation_to_edited_ring_centroid_distance_angstrom"]
    )
    assert original["positive_negative_charge_centroid_separation_angstrom"] == pytest.approx(
        translated["positive_negative_charge_centroid_separation_angstrom"]
    )
