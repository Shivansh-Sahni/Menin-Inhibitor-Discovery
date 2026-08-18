#!/usr/bin/env python3
"""Bounded site-specific +1 and receptor-preparation extension for the M3 pilot.

The script operates only on the preregistered LOCAL-MP-02 hERG cliff and the
four core receptor coordinates.  It never assigns equilibrium populations,
binding affinity, or transition rates.  Every xTB job is restartable and
wall-clock bounded.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from Bio.PDB import MMCIFParser
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from menin_discovery.research_local_physics import (
    R_KCAL_MOL_K,
    add_hydrogens_to_heavy_coordinates,
    basin_selection_sensitivity,
    generate_diverse_basin_conformers,
    geometry_observables,
    mol_to_xyz_text,
    protonate_at_atom,
    protonation_site_audit,
    read_xyz_ensemble,
    site_specific_charge_observables,
)
from rdkit import Chem
from run_local_m3_physics import (
    HARTREE_TO_KCAL_MOL,
    RECEPTOR_PATHS,
    ROOT,
    START_SEEDS,
    XTB_SOLVENT_ARGUMENT,
    _heavy_atom_aligned_rmsd,
    _run_one_xtb_job,
    _xtb_json_properties,
)

PILOT_ROOT = ROOT / "research/simulations/pk_herg/local_m3_pilot"
OUTPUT = PILOT_ROOT / "site_specific_extension"
PAIR_ID = "LOCAL-MP-02"
DOMINANT_SEEDS = (2026072501, 2026072502)
SITE_RANK_GAP_KCAL_MOL = R_KCAL_MOL_K * 310.0 * math.log(99.0)
CORE_RECEPTORS = ("8ZYN", "8ZYP", "9CHP", "9CHQ")
KEY_RESIDUE_ATOMS = {
    620: ("N", "CA", "C", "O", "CB", "OG"),
    623: ("N", "CA", "C", "O", "CB", "OG1", "CG2"),
    624: ("N", "CA", "C", "O", "CB", "OG"),
    625: ("N", "CA", "C", "O", "CB", "CG1", "CG2"),
    626: ("N", "CA", "C", "O"),
    627: ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"),
    628: ("N", "CA", "C", "O"),
    629: ("N", "CA", "C", "O", "CB", "CG", "OD1", "ND2"),
    630: ("N", "CA", "C", "O", "CB", "CG1", "CG2"),
    631: ("N", "CA", "C", "O", "CB", "OG"),
    652: ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"),
    656: ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"),
}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _pair_selection() -> pd.DataFrame:
    selected = pd.read_csv(PILOT_ROOT / "selected_compounds.csv")
    result = selected[selected["pair_id"].eq(PAIR_ID)].copy()
    if len(result) != 2:
        raise ValueError("LOCAL-MP-02 must contain exactly two compounds")
    return result


def _write_single_sdf(path: Path, mol: Chem.Mol) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = Chem.SDWriter(str(path))
    writer.write(mol)
    writer.close()


def prepare_site_ranking() -> pd.DataFrame:
    """Prepare equal-formula +1 protomer comparisons from two neutral basins."""

    selection = _pair_selection()
    neutral_jobs = pd.read_csv(PILOT_ROOT / "xtb_basin_jobs.csv")
    audit_rows: list[dict[str, Any]] = []
    job_rows: list[dict[str, Any]] = []
    input_root = OUTPUT / "site_ranking_inputs"
    run_root = OUTPUT / "site_ranking_runs"
    for row in selection.itertuples(index=False):
        audit = protonation_site_audit(row.standardized_smiles)
        for site in audit:
            audit_rows.append({"compound_id": row.compound_id, **site})
        candidates = [site for site in audit if bool(site["candidate_for_site_ranking"])]
        for seed in START_SEEDS:
            source = neutral_jobs[
                neutral_jobs["compound_id"].eq(row.compound_id)
                & neutral_jobs["solvent"].eq("water")
                & neutral_jobs["seed"].eq(seed)
                & neutral_jobs["basin_index_within_seed"].eq(1)
            ]
            if len(source) != 1:
                raise ValueError(f"Could not resolve one neutral source for {row.compound_id}/{seed}")
            source_row = source.iloc[0]
            symbols, frames, _comments = read_xyz_ensemble(
                ROOT / str(source_row["job_directory"]) / "xtbopt.xyz"
            )
            parent = Chem.MolFromSmiles(row.standardized_smiles)
            if parent is None or symbols[: parent.GetNumAtoms()] != [
                atom.GetSymbol() for atom in parent.GetAtoms()
            ]:
                raise ValueError("Neutral optimized coordinates do not preserve heavy-atom order")
            heavy_coordinates = frames[-1][: parent.GetNumAtoms()]
            for site in candidates:
                site_index = int(site["atom_index_zero_based"])
                protonated = protonate_at_atom(row.standardized_smiles, site_index)
                template = add_hydrogens_to_heavy_coordinates(protonated, heavy_coordinates)
                input_dir = input_root / row.compound_id
                stem = f"seed{seed}__siteN{site_index}"
                xyz_path = input_dir / f"{stem}.xyz"
                sdf_path = input_dir / f"{stem}.sdf"
                xyz_path.parent.mkdir(parents=True, exist_ok=True)
                xyz_path.write_text(
                    mol_to_xyz_text(
                        template,
                        comment=(
                            f"{row.compound_id} +1 site-N{site_index} from neutral seed {seed}; "
                            "site-ranking hypothesis, not a micro-pKa or population"
                        ),
                    ),
                    encoding="utf-8",
                )
                _write_single_sdf(sdf_path, template)
                job_id = f"{row.compound_id}__plus1_siteN{site_index}__water__seed{seed}"
                job_rows.append(
                    {
                        "job_id": job_id,
                        "compound_id": row.compound_id,
                        "seed": int(seed),
                        "protonated_atom_index_zero_based": site_index,
                        "site_class": site["site_class"],
                        "rdkit_positive_ionizable_feature": bool(site["rdkit_positive_ionizable_feature"]),
                        "candidate_smiles": Chem.MolToSmiles(protonated, isomericSmiles=True),
                        "input_xyz": str(xyz_path.relative_to(ROOT)),
                        "input_sdf": str(sdf_path.relative_to(ROOT)),
                        "job_directory": str((run_root / job_id).relative_to(ROOT)),
                        "xtb_solvent_argument": "water",
                        "formal_charge": 1,
                        "threads": 4,
                        "truth_boundary": (
                            "same-formula +1 local protomer comparison from one fixed heavy-atom "
                            "start; no microscopic pKa, population, or equilibrium ensemble"
                        ),
                    }
                )
    audit_frame = pd.DataFrame(audit_rows)
    jobs = pd.DataFrame(job_rows)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    audit_frame.to_csv(OUTPUT / "protonation_site_audit.csv", index=False)
    jobs.to_csv(OUTPUT / "site_ranking_jobs.csv", index=False)
    return jobs


def _run_jobs(
    jobs_path: Path,
    *,
    xtb_executable: Path,
    per_job_timeout_minutes: float,
    total_wallclock_minutes: float,
    force: bool,
) -> list[dict[str, Any]]:
    jobs = pd.read_csv(jobs_path)
    deadline = time.monotonic() + float(total_wallclock_minutes) * 60.0
    records: list[dict[str, Any]] = []
    for _, row in jobs.iterrows():
        remaining = deadline - time.monotonic()
        if remaining < 30.0:
            break
        record = _run_one_xtb_job(
            row,
            xtb_executable=xtb_executable,
            timeout_seconds=min(float(per_job_timeout_minutes) * 60.0, remaining),
            force=force,
        )
        records.append(record)
    return records


def analyze_site_ranking() -> pd.DataFrame:
    jobs = pd.read_csv(OUTPUT / "site_ranking_jobs.csv")
    rows: list[dict[str, Any]] = []
    electrostatic_rows: list[dict[str, Any]] = []
    for row in jobs.itertuples(index=False):
        job_dir = ROOT / row.job_directory
        record_path = job_dir / "run_record.json"
        if not record_path.exists():
            continue
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record.get("status") != "completed":
            continue
        properties = _xtb_json_properties(job_dir / "xtbout.json")
        supplier = Chem.SDMolSupplier(str(ROOT / row.input_sdf), removeHs=False)
        template = supplier[0]
        if template is None:
            raise ValueError(f"Could not read site-ranking template for {row.job_id}")
        symbols, frames, _comments = read_xyz_ensemble(job_dir / "xtbopt.xyz")
        if symbols != [atom.GetSymbol() for atom in template.GetAtoms()]:
            raise ValueError(f"xTB changed atom ordering for {row.job_id}")
        optimized = frames[-1]
        charges = np.loadtxt(job_dir / "charges", dtype=float)
        charge_observables = site_specific_charge_observables(
            template,
            optimized,
            charges,
            protonated_atom_index_zero_based=int(row.protonated_atom_index_zero_based),
        )
        if abs(charge_observables["total_atomic_charge"] - 1.0) > 0.02:
            raise ValueError(f"xTB charge closure failed for {row.job_id}")
        electrostatic_rows.append(
            {
                "compound_id": row.compound_id,
                "seed": int(row.seed),
                "protonated_atom_index_zero_based": int(row.protonated_atom_index_zero_based),
                "site_class": row.site_class,
                **geometry_observables(template, optimized),
                **charge_observables,
                "truth_boundary": (
                    "one water-optimized +1 site hypothesis; charge partition is "
                    "Hamiltonian dependent and no site population is assigned"
                ),
            }
        )
        rows.append(
            {
                "compound_id": row.compound_id,
                "seed": int(row.seed),
                "protonated_atom_index_zero_based": int(row.protonated_atom_index_zero_based),
                "site_class": row.site_class,
                "rdkit_positive_ionizable_feature": bool(row.rdkit_positive_ionizable_feature),
                "total_energy_hartree": properties["total_energy_hartree"],
                "elapsed_seconds": float(record["elapsed_seconds"]),
                "truth_boundary": row.truth_boundary,
            }
        )
    ranking = pd.DataFrame(rows)
    if len(ranking) != len(jobs):
        raise ValueError(f"Only {len(ranking)}/{len(jobs)} site-ranking jobs completed")
    ranking["relative_energy_within_compound_seed_kcal_mol"] = ranking.groupby(["compound_id", "seed"])[
        "total_energy_hartree"
    ].transform(lambda values: (values - values.min()) * HARTREE_TO_KCAL_MOL)
    ranking["energy_rank_within_compound_seed"] = ranking.groupby(["compound_id", "seed"])[
        "total_energy_hartree"
    ].rank(method="min")
    ranking.to_csv(OUTPUT / "site_ranking_results.csv", index=False)
    electrostatics = pd.DataFrame(electrostatic_rows).merge(
        ranking[
            [
                "compound_id",
                "seed",
                "protonated_atom_index_zero_based",
                "relative_energy_within_compound_seed_kcal_mol",
                "energy_rank_within_compound_seed",
            ]
        ],
        on=["compound_id", "seed", "protonated_atom_index_zero_based"],
        validate="one_to_one",
    )
    electrostatics.to_csv(
        OUTPUT / "site_ranking_electrostatic_observables.csv",
        index=False,
    )

    presentation_observables = (
        "cation_fragment_sasa_angstrom2",
        "cation_fragment_xtb_charge",
        "edited_ring_xtb_charge",
        "edited_ring_nitrogen_xtb_charge",
        "cation_to_edited_ring_centroid_distance_angstrom",
        "positive_negative_charge_centroid_separation_angstrom",
        "radius_of_gyration_angstrom",
        "polar_heavy_atom_sasa_angstrom2",
    )
    uncertainty_rows: list[dict[str, Any]] = []
    for (compound_id, seed), group in electrostatics.groupby(
        ["compound_id", "seed"],
        sort=True,
    ):
        for observable in presentation_observables:
            values = group[observable].astype(float)
            uncertainty_rows.append(
                {
                    "compound_id": compound_id,
                    "seed": int(seed),
                    "observable": observable,
                    "protonation_site_hypothesis_count": len(values),
                    "minimum_across_sites": float(values.min()),
                    "maximum_across_sites": float(values.max()),
                    "range_across_sites": float(values.max() - values.min()),
                    "standard_deviation_across_sites": float(values.std(ddof=1)),
                    "truth_boundary": (
                        "spread across admitted +1 site hypotheses at one local minimum "
                        "each; not a thermodynamic uncertainty interval"
                    ),
                }
            )
    uncertainty = pd.DataFrame(uncertainty_rows)
    uncertainty.to_csv(
        OUTPUT / "protonation_site_presentation_uncertainty.csv",
        index=False,
    )

    pair_signal_rows: list[dict[str, Any]] = []
    compound_ids = sorted(electrostatics["compound_id"].unique())
    n28 = electrostatics[electrostatics["protonated_atom_index_zero_based"].eq(28)]
    for seed, group in n28.groupby("seed", sort=True):
        by_compound = group.set_index("compound_id")
        if not set(compound_ids).issubset(by_compound.index):
            continue
        for observable in presentation_observables:
            left = float(by_compound.loc[compound_ids[0], observable])
            right = float(by_compound.loc[compound_ids[1], observable])
            pair_signal_rows.append(
                {
                    "seed": int(seed),
                    "observable": observable,
                    "compound_a": compound_ids[0],
                    "compound_b": compound_ids[1],
                    "compound_a_n28_value": left,
                    "compound_b_n28_value": right,
                    "absolute_n28_pair_difference": abs(left - right),
                    "truth_boundary": (
                        "plausible N28 sensitivity contrast from corresponding generation "
                        "seeds; not a population-weighted matched-pair effect"
                    ),
                }
            )
    pair_signal = pd.DataFrame(pair_signal_rows)
    pair_signal.to_csv(OUTPUT / "n28_pair_presentation_sensitivity.csv", index=False)

    context_rows: list[dict[str, Any]] = []
    for observable in presentation_observables:
        site_ranges = uncertainty[uncertainty["observable"].eq(observable)]["range_across_sites"].astype(
            float
        )
        pair_differences = pair_signal[pair_signal["observable"].eq(observable)][
            "absolute_n28_pair_difference"
        ].astype(float)
        median_site_range = float(site_ranges.median())
        median_pair_difference = float(pair_differences.median())
        context_rows.append(
            {
                "observable": observable,
                "median_within_compound_site_range": median_site_range,
                "median_absolute_n28_pair_difference": median_pair_difference,
                "site_range_to_pair_difference_ratio": (
                    median_site_range / median_pair_difference if median_pair_difference > 0 else float("inf")
                ),
                "site_identity_spread_exceeds_n28_pair_difference": bool(
                    median_site_range > median_pair_difference
                ),
                "decision": ("all_candidate_screening_span_only_no_admission_decision"),
                "truth_boundary": (
                    "descriptive scale comparison across selected local hypotheses; "
                    "not variance decomposition or statistical inference"
                ),
            }
        )
    pd.DataFrame(context_rows).to_csv(
        OUTPUT / "site_uncertainty_vs_pair_signal.csv",
        index=False,
    )

    selection = _pair_selection().set_index("compound_id")
    gate_rows: list[dict[str, Any]] = []
    for compound_id, group in ranking.groupby("compound_id", sort=True):
        top = group.loc[group.groupby("seed")["total_energy_hartree"].idxmin()].sort_values("seed")
        expected = group[
            group["site_class"].eq("aliphatic_amine") & group["rdkit_positive_ionizable_feature"]
        ]["protonated_atom_index_zero_based"].unique()
        if len(expected) != 1:
            raise ValueError(f"Expected one dominant aliphatic basic center for {compound_id}")
        expected_site = int(expected[0])
        gaps: list[float] = []
        for _seed, seed_group in group.groupby("seed"):
            energies = sorted(seed_group["total_energy_hartree"].astype(float))
            gaps.append((energies[1] - energies[0]) * HARTREE_TO_KCAL_MOL)
        top_sites = [int(value) for value in top["protonated_atom_index_zero_based"]]
        highest_macro_pka = max(
            float(value) for value in str(selection.loc[compound_id, "basic_macro_pkas"]).split(";")
        )
        gate_passed = (
            len(top_sites) == len(START_SEEDS)
            and set(top_sites) == {expected_site}
            and min(gaps) >= SITE_RANK_GAP_KCAL_MOL
            and highest_macro_pka >= 8.5
        )
        gate_rows.append(
            {
                "compound_id": compound_id,
                "expected_dominant_plus1_atom_index_zero_based": expected_site,
                "top_site_by_seed": ";".join(
                    f"{int(seed)}:{int(site)}" for seed, site in zip(top["seed"], top_sites, strict=True)
                ),
                "runner_up_gap_kcal_mol_by_seed": ";".join(
                    f"{int(seed)}:{gap:.6g}" for seed, gap in zip(top["seed"], gaps, strict=True)
                ),
                "minimum_runner_up_gap_kcal_mol": min(gaps),
                "declared_99_to_1_thermal_gap_kcal_mol_310K": SITE_RANK_GAP_KCAL_MOL,
                "highest_measured_basic_macro_pka": highest_macro_pka,
                "local_site_assignment_gate_passed": gate_passed,
                "interpretation": (
                    "conformation-replicated local-energy support for the dominant +1 "
                    "site hypothesis; not a microscopic pKa or population"
                ),
            }
        )
    gate = pd.DataFrame(gate_rows)
    gate.to_csv(OUTPUT / "dominant_plus1_site_gate.csv", index=False)
    return gate


def prepare_gfn1_site_sensitivity() -> pd.DataFrame:
    """Prepare a second xTB Hamiltonian as a method-sensitivity falsifier."""

    primary = pd.read_csv(OUTPUT / "site_ranking_jobs.csv")
    rows: list[dict[str, Any]] = []
    run_root = OUTPUT / "site_ranking_gfn1_runs"
    for row in primary.itertuples(index=False):
        primary_job_dir = ROOT / row.job_directory
        optimized = primary_job_dir / "xtbopt.xyz"
        record_path = primary_job_dir / "run_record.json"
        if not optimized.exists() or not record_path.exists():
            raise ValueError(f"Primary GFN2 site job is incomplete: {row.job_id}")
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record.get("status") != "completed":
            raise ValueError(f"Primary GFN2 site job did not complete: {row.job_id}")
        job_id = f"{row.job_id}__gfn1_sensitivity"
        rows.append(
            {
                "job_id": job_id,
                "primary_gfn2_job_id": row.job_id,
                "compound_id": row.compound_id,
                "seed": int(row.seed),
                "protonated_atom_index_zero_based": int(row.protonated_atom_index_zero_based),
                "input_xyz": str(optimized.relative_to(ROOT)),
                "job_directory": str((run_root / job_id).relative_to(ROOT)),
                "xtb_solvent_argument": "water",
                "formal_charge": 1,
                "threads": 4,
                "gfn_level": 1,
                "truth_boundary": (
                    "GFN1-xTB/ALPB reoptimization from the GFN2 minimum is a shared-family "
                    "Hamiltonian sensitivity control, not high-level validation"
                ),
            }
        )
    jobs = pd.DataFrame(rows)
    jobs.to_csv(OUTPUT / "site_ranking_gfn1_jobs.csv", index=False)
    return jobs


def analyze_gfn1_site_sensitivity() -> pd.DataFrame:
    jobs = pd.read_csv(OUTPUT / "site_ranking_gfn1_jobs.csv")
    rows: list[dict[str, Any]] = []
    for row in jobs.itertuples(index=False):
        job_dir = ROOT / row.job_directory
        record_path = job_dir / "run_record.json"
        if not record_path.exists():
            continue
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record.get("status") != "completed":
            continue
        properties = _xtb_json_properties(job_dir / "xtbout.json")
        rows.append(
            {
                "compound_id": row.compound_id,
                "seed": int(row.seed),
                "protonated_atom_index_zero_based": int(row.protonated_atom_index_zero_based),
                "gfn1_total_energy_hartree": properties["total_energy_hartree"],
                "elapsed_seconds": float(record["elapsed_seconds"]),
                "truth_boundary": row.truth_boundary,
            }
        )
    gfn1 = pd.DataFrame(rows)
    if len(gfn1) != len(jobs):
        raise ValueError(f"Only {len(gfn1)}/{len(jobs)} GFN1 sensitivity jobs completed")
    gfn1["gfn1_relative_energy_within_compound_seed_kcal_mol"] = gfn1.groupby(["compound_id", "seed"])[
        "gfn1_total_energy_hartree"
    ].transform(lambda values: (values - values.min()) * HARTREE_TO_KCAL_MOL)
    gfn1["gfn1_energy_rank_within_compound_seed"] = gfn1.groupby(["compound_id", "seed"])[
        "gfn1_total_energy_hartree"
    ].rank(method="min")
    primary = pd.read_csv(OUTPUT / "site_ranking_results.csv")
    merged = primary.merge(
        gfn1,
        on=["compound_id", "seed", "protonated_atom_index_zero_based"],
        validate="one_to_one",
    )
    merged.to_csv(OUTPUT / "site_ranking_hamiltonian_sensitivity.csv", index=False)

    electrostatics = pd.read_csv(OUTPUT / "site_ranking_electrostatic_observables.csv")
    joined = electrostatics.merge(
        gfn1[
            [
                "compound_id",
                "seed",
                "protonated_atom_index_zero_based",
                "gfn1_relative_energy_within_compound_seed_kcal_mol",
                "gfn1_energy_rank_within_compound_seed",
            ]
        ],
        on=["compound_id", "seed", "protonated_atom_index_zero_based"],
        validate="one_to_one",
    )
    joined["within_99_to_1_window_under_either_xtb_hamiltonian"] = joined[
        "relative_energy_within_compound_seed_kcal_mol"
    ].le(SITE_RANK_GAP_KCAL_MOL) | joined["gfn1_relative_energy_within_compound_seed_kcal_mol"].le(
        SITE_RANK_GAP_KCAL_MOL
    )
    joined.to_csv(
        OUTPUT / "site_ranking_joint_hamiltonian_admission.csv",
        index=False,
    )
    plausible = joined[joined["within_99_to_1_window_under_either_xtb_hamiltonian"]].copy()
    presentation_observables = (
        "cation_fragment_sasa_angstrom2",
        "cation_fragment_xtb_charge",
        "edited_ring_xtb_charge",
        "edited_ring_nitrogen_xtb_charge",
        "cation_to_edited_ring_centroid_distance_angstrom",
        "positive_negative_charge_centroid_separation_angstrom",
        "radius_of_gyration_angstrom",
        "polar_heavy_atom_sasa_angstrom2",
    )
    near_rows: list[dict[str, Any]] = []
    for (compound_id, seed), group in plausible.groupby(
        ["compound_id", "seed"],
        sort=True,
    ):
        site_ids = sorted(group["protonated_atom_index_zero_based"].astype(int).unique())
        for observable in presentation_observables:
            values = group[observable].astype(float)
            near_rows.append(
                {
                    "compound_id": compound_id,
                    "seed": int(seed),
                    "observable": observable,
                    "admitted_site_ids": ";".join(str(site) for site in site_ids),
                    "admitted_site_count": len(site_ids),
                    "minimum_across_admitted_sites": float(values.min()),
                    "maximum_across_admitted_sites": float(values.max()),
                    "range_across_admitted_sites": float(values.max() - values.min()),
                    "admission_energy_window_kcal_mol": SITE_RANK_GAP_KCAL_MOL,
                    "admission_rule": (
                        "within the 99:1 thermal gap of the local minimum under GFN1 "
                        "or GFN2; union protects against false exclusion by one Hamiltonian"
                    ),
                    "truth_boundary": (
                        "local near-degenerate-site sensitivity; no microscopic pKa, "
                        "population, or calibrated Hamiltonian uncertainty"
                    ),
                }
            )
    near = pd.DataFrame(near_rows)
    near.to_csv(
        OUTPUT / "near_degenerate_site_presentation_uncertainty.csv",
        index=False,
    )
    pair_signal = pd.read_csv(OUTPUT / "n28_pair_presentation_sensitivity.csv")
    context_rows: list[dict[str, Any]] = []
    for observable in presentation_observables:
        site_ranges = near[near["observable"].eq(observable)]["range_across_admitted_sites"].astype(float)
        pair_differences = pair_signal[pair_signal["observable"].eq(observable)][
            "absolute_n28_pair_difference"
        ].astype(float)
        median_site_range = float(site_ranges.median())
        median_pair_difference = float(pair_differences.median())
        context_rows.append(
            {
                "observable": observable,
                "median_near_degenerate_site_range": median_site_range,
                "median_absolute_n28_pair_difference": median_pair_difference,
                "site_range_to_pair_difference_ratio": (
                    median_site_range / median_pair_difference if median_pair_difference > 0 else float("inf")
                ),
                "near_degenerate_site_spread_exceeds_n28_pair_difference": bool(
                    median_site_range > median_pair_difference
                ),
                "single_site_feature_decision": ("rejected_site_assignment_not_hamiltonian_robust"),
                "truth_boundary": (
                    "descriptive comparison after energy-window admission under either "
                    "xTB Hamiltonian; not a variance decomposition"
                ),
            }
        )
    pd.DataFrame(context_rows).to_csv(
        OUTPUT / "near_degenerate_site_uncertainty_vs_pair_signal.csv",
        index=False,
    )

    summary_rows: list[dict[str, Any]] = []
    for (compound_id, seed), group in merged.groupby(
        ["compound_id", "seed"],
        sort=True,
    ):
        gfn2_top = int(group.loc[group["total_energy_hartree"].idxmin()]["protonated_atom_index_zero_based"])
        gfn1_top = int(
            group.loc[group["gfn1_total_energy_hartree"].idxmin()]["protonated_atom_index_zero_based"]
        )
        gfn1_energies = sorted(group["gfn1_total_energy_hartree"].astype(float))
        gfn1_gap = (gfn1_energies[1] - gfn1_energies[0]) * HARTREE_TO_KCAL_MOL
        rank_correlation = float(
            group[
                [
                    "energy_rank_within_compound_seed",
                    "gfn1_energy_rank_within_compound_seed",
                ]
            ]
            .corr(method="spearman")
            .iloc[0, 1]
        )
        summary_rows.append(
            {
                "compound_id": compound_id,
                "seed": int(seed),
                "gfn2_top_site": gfn2_top,
                "gfn1_top_site": gfn1_top,
                "top_site_stable_across_xtb_hamiltonians": gfn2_top == gfn1_top,
                "gfn1_runner_up_gap_kcal_mol": gfn1_gap,
                "rank_spearman_gfn2_vs_gfn1": rank_correlation,
                "shared_family_method_limit": (
                    "GFN1/GFN2 agreement is supportive only; disagreement falsifies "
                    "Hamiltonian-robust site assignment"
                ),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUTPUT / "site_ranking_hamiltonian_summary.csv", index=False)
    return summary


def prepare_dominant_plus1() -> pd.DataFrame:
    gate = pd.read_csv(OUTPUT / "dominant_plus1_site_gate.csv")
    if not bool(gate["local_site_assignment_gate_passed"].all()):
        raise ValueError("Dominant +1 site gate did not pass for both compounds")
    selection = _pair_selection().set_index("compound_id")
    input_root = OUTPUT / "dominant_plus1_inputs"
    run_root = OUTPUT / "dominant_plus1_runs"
    audit_rows: list[dict[str, Any]] = []
    job_rows: list[dict[str, Any]] = []
    sensitivity_rows: list[dict[str, Any]] = []
    for gate_row in gate.itertuples(index=False):
        smiles = str(selection.loc[gate_row.compound_id, "standardized_smiles"])
        protonated = protonate_at_atom(
            smiles,
            int(gate_row.expected_dominant_plus1_atom_index_zero_based),
        )
        protonated_smiles = Chem.MolToSmiles(
            protonated,
            canonical=False,
            isomericSmiles=True,
        )
        basins = generate_diverse_basin_conformers(
            protonated_smiles,
            seeds=DOMINANT_SEEDS,
            pool_size=64,
            maximum_per_seed=2,
            energy_window_kcal_mol=12.0,
            minimum_heavy_atom_rmsd_angstrom=1.25,
        )
        sensitivity = basin_selection_sensitivity(
            protonated_smiles,
            seeds=DOMINANT_SEEDS,
            maximum_pool_size=64,
            pool_size_prefixes=(32, 64),
            energy_windows_kcal_mol=(8.0, 12.0, 16.0),
            rmsd_gates_angstrom=(1.0, 1.25, 1.5),
            maximum_per_seed=2,
        )
        sensitivity_rows.extend({"compound_id": gate_row.compound_id, **item} for item in sensitivity)
        basin_counts: Counter[int] = Counter()
        for mol, metadata in basins:
            seed = int(metadata["seed"])
            basin_counts[seed] += 1
            basin_index = basin_counts[seed]
            charged_sites = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetFormalCharge() == 1]
            if len(charged_sites) != 1:
                raise ValueError("Dominant +1 conformer lost its unique formal cation center")
            charged_site = int(charged_sites[0])
            input_dir = input_root / gate_row.compound_id
            stem = f"seed{seed}__basin{basin_index}"
            xyz_path = input_dir / f"{stem}.xyz"
            sdf_path = input_dir / f"{stem}.sdf"
            xyz_path.parent.mkdir(parents=True, exist_ok=True)
            xyz_path.write_text(
                mol_to_xyz_text(
                    mol,
                    comment=(
                        f"{gate_row.compound_id} dominant +1 site-N{charged_site}; "
                        f"seed {seed}, basin {basin_index}; no equilibrium weight"
                    ),
                ),
                encoding="utf-8",
            )
            _write_single_sdf(sdf_path, mol)
            audit_rows.append(
                {
                    "compound_id": gate_row.compound_id,
                    "seed": seed,
                    "basin_index_within_seed": basin_index,
                    "protonated_atom_index_zero_based": charged_site,
                    "input_xyz": str(xyz_path.relative_to(ROOT)),
                    "input_sdf": str(sdf_path.relative_to(ROOT)),
                    **metadata,
                }
            )
            for solvent, xtb_solvent in XTB_SOLVENT_ARGUMENT.items():
                job_id = f"{gate_row.compound_id}__dominant_plus1__{solvent}__seed{seed}__basin{basin_index}"
                job_rows.append(
                    {
                        "job_id": job_id,
                        "compound_id": gate_row.compound_id,
                        "seed": seed,
                        "basin_index_within_seed": basin_index,
                        "protonated_atom_index_zero_based": charged_site,
                        "solvent": solvent,
                        "input_xyz": str(xyz_path.relative_to(ROOT)),
                        "input_sdf": str(sdf_path.relative_to(ROOT)),
                        "job_directory": str((run_root / job_id).relative_to(ROOT)),
                        "xtb_solvent_argument": xtb_solvent,
                        "formal_charge": 1,
                        "threads": 4,
                        "truth_boundary": (
                            "dominant +1 environment-conditioned local minimum; no "
                            "equilibrium weight, binding affinity, or transition rate"
                        ),
                    }
                )
    audit = pd.DataFrame(audit_rows)
    jobs = pd.DataFrame(job_rows)
    sensitivity_frame = pd.DataFrame(sensitivity_rows)
    audit.to_csv(OUTPUT / "dominant_plus1_basin_audit.csv", index=False)
    jobs.to_csv(OUTPUT / "dominant_plus1_jobs.csv", index=False)
    sensitivity_frame.to_csv(
        OUTPUT / "dominant_plus1_basin_selection_sensitivity.csv",
        index=False,
    )
    return jobs


def analyze_dominant_plus1() -> dict[str, int]:
    jobs = pd.read_csv(OUTPUT / "dominant_plus1_jobs.csv")
    rows: list[dict[str, Any]] = []
    coordinates: dict[str, np.ndarray] = {}
    templates: dict[str, Chem.Mol] = {}
    for row in jobs.itertuples(index=False):
        job_dir = ROOT / row.job_directory
        record_path = job_dir / "run_record.json"
        if not record_path.exists():
            continue
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record.get("status") != "completed":
            continue
        supplier = Chem.SDMolSupplier(str(ROOT / row.input_sdf), removeHs=False)
        template = supplier[0]
        if template is None:
            raise ValueError(f"Could not read template for {row.job_id}")
        symbols, frames, _comments = read_xyz_ensemble(job_dir / "xtbopt.xyz")
        if symbols != [atom.GetSymbol() for atom in template.GetAtoms()]:
            raise ValueError(f"xTB changed atom ordering for {row.job_id}")
        optimized = frames[-1]
        charges = np.loadtxt(job_dir / "charges", dtype=float)
        charge_observables = site_specific_charge_observables(
            template,
            optimized,
            charges,
            protonated_atom_index_zero_based=int(row.protonated_atom_index_zero_based),
        )
        if abs(charge_observables["total_atomic_charge"] - 1.0) > 0.02:
            raise ValueError(f"xTB charge closure failed for {row.job_id}")
        properties = _xtb_json_properties(job_dir / "xtbout.json")
        rows.append(
            {
                "job_id": row.job_id,
                "compound_id": row.compound_id,
                "seed": int(row.seed),
                "basin_index_within_seed": int(row.basin_index_within_seed),
                "solvent": row.solvent,
                "elapsed_seconds": float(record["elapsed_seconds"]),
                "total_energy_hartree": properties["total_energy_hartree"],
                **geometry_observables(template, optimized),
                **charge_observables,
                "truth_boundary": row.truth_boundary,
            }
        )
        coordinates[row.job_id] = optimized
        templates[row.compound_id] = template
    results = pd.DataFrame(rows)
    if len(results) != len(jobs):
        raise ValueError(f"Only {len(results)}/{len(jobs)} dominant +1 jobs completed")
    results["relative_energy_within_compound_solvent_kcal_mol"] = results.groupby(["compound_id", "solvent"])[
        "total_energy_hartree"
    ].transform(lambda values: (values - values.min()) * HARTREE_TO_KCAL_MOL)
    results.to_csv(OUTPUT / "dominant_plus1_local_minima.csv", index=False)

    observables = (
        "cation_fragment_sasa_angstrom2",
        "cation_fragment_xtb_charge",
        "edited_ring_xtb_charge",
        "edited_ring_nitrogen_xtb_charge",
        "cation_to_edited_ring_centroid_distance_angstrom",
        "positive_negative_charge_centroid_separation_angstrom",
        "radius_of_gyration_angstrom",
        "polar_heavy_atom_sasa_angstrom2",
    )
    paired_rows: list[dict[str, Any]] = []
    for keys, group in results.groupby(
        ["compound_id", "seed", "basin_index_within_seed"],
        sort=True,
    ):
        by_solvent = group.set_index("solvent")
        for observable in observables:
            water = float(by_solvent.loc["water", observable])
            chloroform = float(by_solvent.loc["chloroform", observable])
            paired_rows.append(
                {
                    "compound_id": keys[0],
                    "seed": int(keys[1]),
                    "basin_index_within_seed": int(keys[2]),
                    "observable": observable,
                    "water_value": water,
                    "chloroform_value": chloroform,
                    "water_minus_chloroform": water - chloroform,
                    "truth_boundary": ("paired charged-state local-minimum response; no ensemble shift"),
                }
            )
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(OUTPUT / "dominant_plus1_paired_solvent_responses.csv", index=False)

    summary_rows: list[dict[str, Any]] = []
    for (compound_id, observable), group in paired.groupby(
        ["compound_id", "observable"],
        sort=True,
    ):
        values = group["water_minus_chloroform"].astype(float)
        median = float(values.median())
        summary_rows.append(
            {
                "compound_id": compound_id,
                "observable": observable,
                "local_basin_count": len(values),
                "median_water_minus_chloroform": median,
                "minimum_water_minus_chloroform": float(values.min()),
                "maximum_water_minus_chloroform": float(values.max()),
                "same_nonzero_direction_in_all_basins": bool(values.gt(0).all() or values.lt(0).all()),
                "sign_agreement_fraction": float((np.sign(values) == np.sign(median)).mean())
                if median
                else float(values.eq(0).mean()),
                "promotion_status": "mechanistic_hypothesis_only",
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUTPUT / "dominant_plus1_solvent_response_summary.csv", index=False)

    comparison_rows: list[dict[str, Any]] = []
    compound_ids = sorted(results["compound_id"].unique())
    for solvent in ("water", "chloroform"):
        for observable in observables:
            left = results[results["compound_id"].eq(compound_ids[0]) & results["solvent"].eq(solvent)][
                observable
            ].astype(float)
            right = results[results["compound_id"].eq(compound_ids[1]) & results["solvent"].eq(solvent)][
                observable
            ].astype(float)
            overlap = max(0.0, min(left.max(), right.max()) - max(left.min(), right.min()))
            comparison_rows.append(
                {
                    "solvent": solvent,
                    "observable": observable,
                    "compound_a": compound_ids[0],
                    "compound_b": compound_ids[1],
                    "compound_a_median": float(left.median()),
                    "compound_a_range": f"{left.min():.8g};{left.max():.8g}",
                    "compound_b_median": float(right.median()),
                    "compound_b_range": f"{right.min():.8g};{right.max():.8g}",
                    "selected_basin_interval_overlap": float(overlap),
                    "selected_basin_intervals_nonoverlapping": bool(overlap == 0.0),
                    "truth_boundary": (
                        "four selected local basins per compound; not a confidence interval "
                        "or equilibrium distribution"
                    ),
                }
            )
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(OUTPUT / "dominant_plus1_pair_comparison.csv", index=False)

    convergence_rows: list[dict[str, Any]] = []
    for (compound_id, solvent), group in results.groupby(
        ["compound_id", "solvent"],
        sort=True,
    ):
        seed_minima = group.loc[group.groupby("seed")["total_energy_hartree"].idxmin()]
        if len(seed_minima) != 2:
            continue
        left, right = list(seed_minima.itertuples(index=False))
        convergence_rows.append(
            {
                "compound_id": compound_id,
                "solvent": solvent,
                "absolute_energy_gap_kcal_mol": abs(
                    float(left.total_energy_hartree) - float(right.total_energy_hartree)
                )
                * HARTREE_TO_KCAL_MOL,
                "heavy_atom_rmsd_angstrom": _heavy_atom_aligned_rmsd(
                    templates[compound_id],
                    coordinates[left.job_id],
                    coordinates[right.job_id],
                ),
                "truth_boundary": (
                    "cross-seed lowest discovered local-minimum agreement; no equilibrium-convergence claim"
                ),
            }
        )
    pd.DataFrame(convergence_rows).to_csv(
        OUTPUT / "dominant_plus1_cross_seed_convergence.csv",
        index=False,
    )
    return {
        "completed_dominant_plus1_local_minima": len(results),
        "paired_solvent_responses": len(paired),
        "pair_comparisons": len(comparison),
        "cross_seed_comparisons": len(convergence_rows),
    }


def _cif_list(dictionary: dict[str, Any], key: str) -> list[str]:
    value = dictionary.get(key, [])
    return [str(item) for item in (value if isinstance(value, list) else [value])]


def receptor_preparation_audit() -> dict[str, Any]:
    """Audit preparation-critical differences without constructing receptors."""

    summary_rows: list[dict[str, Any]] = []
    residue_rows: list[dict[str, Any]] = []
    parser = MMCIFParser(QUIET=True)
    for pdb_id in CORE_RECEPTORS:
        path = RECEPTOR_PATHS[pdb_id]
        dictionary = MMCIF2Dict(str(path))
        structure = parser.get_structure(pdb_id, str(path))
        model = next(structure.get_models())
        sequence_text = "".join(_cif_list(dictionary, "_entity_poly.pdbx_seq_one_letter_code_can"))
        sequence = "".join(sequence_text.split())
        protein_chains = [chain for chain in model if chain.id in {"A", "B", "C", "D"}]
        resolved_counts = [sum(residue.id[0] == " " for residue in chain) for chain in protein_chains]
        resolved_numbers = [
            residue.id[1] for chain in protein_chains for residue in chain if residue.id[0] == " "
        ]
        hetero = Counter(
            residue.resname for chain in model for residue in chain if residue.id[0] not in {" ", "W"}
        )
        hydrogens = sum(atom.element == "H" for atom in model.get_atoms())
        altloc_atoms = sum(
            atom.is_disordered() or (atom.get_altloc().strip() not in {"", "A"}) for atom in model.get_atoms()
        )
        declared_mutations = sorted(
            {value for value in _cif_list(dictionary, "_entity.pdbx_mutation") if value not in {"", ".", "?"}}
        )
        difference_details = _cif_list(dictionary, "_struct_ref_seq_dif.details")
        difference_classes = Counter(difference_details)
        coordinate_monomers = _cif_list(dictionary, "_struct_ref_seq_dif.mon_id")
        reference_monomers = _cif_list(dictionary, "_struct_ref_seq_dif.db_mon_id")
        substitution_records = sum(
            coordinate not in {"", ".", "?"} and reference not in {"", ".", "?"} and coordinate != reference
            for coordinate, reference in zip(
                coordinate_monomers,
                reference_monomers,
                strict=False,
            )
        )
        resolution_values = [
            float(value)
            for value in _cif_list(dictionary, "_em_3d_reconstruction.resolution")
            if value not in {"", ".", "?"}
        ]
        for chain in protein_chains:
            for residue_number, expected_atoms in KEY_RESIDUE_ATOMS.items():
                residue_id = (" ", residue_number, " ")
                present = {atom.name for atom in chain[residue_id]} if residue_id in chain else set()
                missing = sorted(set(expected_atoms) - present)
                residue_rows.append(
                    {
                        "pdb_id": pdb_id,
                        "chain_id": chain.id,
                        "residue_number": residue_number,
                        "expected_heavy_atoms": ";".join(expected_atoms),
                        "missing_heavy_atoms": ";".join(missing),
                        "preparation_gate_passed": not missing,
                    }
                )
        summary_rows.append(
            {
                "pdb_id": pdb_id,
                "canonical_construct_sequence_length": len(sequence),
                "source_construct_segments": ";".join(
                    f"{beg}-{end}"
                    for beg, end in zip(
                        _cif_list(dictionary, "_entity_src_gen.pdbx_beg_seq_num"),
                        _cif_list(dictionary, "_entity_src_gen.pdbx_end_seq_num"),
                        strict=False,
                    )
                ),
                "uniprot_accession": ";".join(
                    sorted(set(_cif_list(dictionary, "_struct_ref_seq.pdbx_db_accession")))
                ),
                "uniprot_author_alignment_ranges": ";".join(
                    sorted(
                        set(
                            f"{beg}-{end}"
                            for beg, end in zip(
                                _cif_list(
                                    dictionary,
                                    "_struct_ref_seq.pdbx_auth_seq_align_beg",
                                ),
                                _cif_list(
                                    dictionary,
                                    "_struct_ref_seq.pdbx_auth_seq_align_end",
                                ),
                                strict=False,
                            )
                        )
                    )
                ),
                "assembly_asym_ids": ";".join(
                    _cif_list(dictionary, "_pdbx_struct_assembly_gen.asym_id_list")
                ),
                "resolved_protein_chain_count": len(protein_chains),
                "resolved_residues_min_per_chain": min(resolved_counts),
                "resolved_residues_max_per_chain": max(resolved_counts),
                "resolved_author_residue_min": min(resolved_numbers),
                "resolved_author_residue_max": max(resolved_numbers),
                "explicit_hydrogen_atom_count": hydrogens,
                "water_count": sum(residue.id[0] == "W" for chain in model for residue in chain),
                "nonwater_hetero_components": ";".join(
                    f"{key}:{value}" for key, value in sorted(hetero.items())
                ),
                "alternate_location_atom_count": altloc_atoms,
                "reported_global_resolution_angstrom": (
                    resolution_values[0] if resolution_values else np.nan
                ),
                "depositor_declared_mutations": ";".join(declared_mutations),
                "sequence_difference_records_across_all_chains": len(difference_details),
                "sequence_difference_classes_across_all_chains": ";".join(
                    f"{key}:{value}" for key, value in sorted(difference_classes.items())
                ),
                "nonreference_substitution_records_across_all_chains": (substitution_records),
                "construct_family": (
                    "miyashita_211_1024_plus_expression_tag_family"
                    if pdb_id.startswith("8ZY")
                    else "lau_241_1024_family"
                ),
                "truth_boundary": (
                    "deposited-coordinate preparation audit; no repaired, protonated, "
                    "membrane-embedded, or simulation-ready receptor"
                ),
            }
        )
    summary = pd.DataFrame(summary_rows)
    residues = pd.DataFrame(residue_rows)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT / "core_receptor_preparation_audit.csv", index=False)
    residues.to_csv(OUTPUT / "core_receptor_key_residue_completeness.csv", index=False)
    construct_family_count = int(summary["construct_family"].nunique())
    hydrogen_regimes = int(summary["explicit_hydrogen_atom_count"].gt(0).nunique())
    all_key_atoms_present = bool(residues["preparation_gate_passed"].all())
    payload = {
        "core_receptor_count": len(summary),
        "all_key_pore_and_cavity_heavy_atoms_present": all_key_atoms_present,
        "construct_family_count": construct_family_count,
        "inconsistent_explicit_hydrogen_regimes": hydrogen_regimes > 1,
        "all_deposits_lack_resolved_waters": bool(summary["water_count"].eq(0).all()),
        "no_depositor_declared_mutations": bool(summary["depositor_declared_mutations"].eq("").all()),
        "no_nonreference_substitution_records": bool(
            summary["nonreference_substitution_records_across_all_chains"].eq(0).all()
        ),
        "simulation_preparation_gate": "blocked_requires_common_construct_and_protonation_policy",
        "required_actions": [
            "choose a purpose-specific construct family; do not interpret 8ZY/9CH differences as pure state effects",
            "treat absence of deposited substitution records as distinct from a native full-length construct; preserve truncation/tag provenance",
            "strip deposited hydrogens and reprotonate all systems under one reviewed protocol",
            "review missing segments, termini, ions, ligand retention, and membrane placement",
            "retain intact deposited filter/cavity families; do not create coordinate chimeras",
            "perform finite-energy, membrane-equilibration, restart, and replica-stability smoke gates",
        ],
        "truth_boundary": (
            "key cavity/filter heavy atoms are complete, but construct and preparation "
            "heterogeneity prohibit direct docking or MD"
        ),
    }
    _write_json(OUTPUT / "core_receptor_preparation_gate.json", payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        required=True,
        choices=(
            "prepare-site-ranking",
            "run-site-ranking",
            "analyze-site-ranking",
            "prepare-gfn1-site-sensitivity",
            "run-gfn1-site-sensitivity",
            "analyze-gfn1-site-sensitivity",
            "prepare-dominant-plus1",
            "run-dominant-plus1",
            "analyze-dominant-plus1",
            "receptor-audit",
        ),
    )
    parser.add_argument(
        "--xtb-executable",
        default="/private/tmp/menin-crest212-env/bin/xtb",
    )
    parser.add_argument("--per-job-timeout-minutes", type=float, default=5.0)
    parser.add_argument("--total-wallclock-minutes", type=float, default=25.0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {"stage": args.stage}
    if args.stage == "prepare-site-ranking":
        result["prepared_jobs"] = len(prepare_site_ranking())
    elif args.stage == "run-site-ranking":
        result["records"] = _run_jobs(
            OUTPUT / "site_ranking_jobs.csv",
            xtb_executable=Path(args.xtb_executable),
            per_job_timeout_minutes=args.per_job_timeout_minutes,
            total_wallclock_minutes=args.total_wallclock_minutes,
            force=args.force,
        )
    elif args.stage == "analyze-site-ranking":
        gate = analyze_site_ranking()
        result["site_gates"] = gate.to_dict("records")
    elif args.stage == "prepare-gfn1-site-sensitivity":
        result["prepared_jobs"] = len(prepare_gfn1_site_sensitivity())
    elif args.stage == "run-gfn1-site-sensitivity":
        result["records"] = _run_jobs(
            OUTPUT / "site_ranking_gfn1_jobs.csv",
            xtb_executable=Path(args.xtb_executable),
            per_job_timeout_minutes=args.per_job_timeout_minutes,
            total_wallclock_minutes=args.total_wallclock_minutes,
            force=args.force,
        )
    elif args.stage == "analyze-gfn1-site-sensitivity":
        summary = analyze_gfn1_site_sensitivity()
        result["hamiltonian_sensitivity"] = summary.to_dict("records")
    elif args.stage == "prepare-dominant-plus1":
        result["prepared_jobs"] = len(prepare_dominant_plus1())
    elif args.stage == "run-dominant-plus1":
        result["records"] = _run_jobs(
            OUTPUT / "dominant_plus1_jobs.csv",
            xtb_executable=Path(args.xtb_executable),
            per_job_timeout_minutes=args.per_job_timeout_minutes,
            total_wallclock_minutes=args.total_wallclock_minutes,
            force=args.force,
        )
    elif args.stage == "analyze-dominant-plus1":
        result["analysis"] = analyze_dominant_plus1()
    elif args.stage == "receptor-audit":
        result["receptor_gate"] = receptor_preparation_audit()
    _write_json(OUTPUT / f"stage_{args.stage}_summary.json", result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
