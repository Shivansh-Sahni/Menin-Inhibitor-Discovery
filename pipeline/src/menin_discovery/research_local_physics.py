"""Bounded, publication-oriented physics analyses for the local M3 pilot.

The functions in this module deliberately avoid turning inexpensive
calculations into stronger evidence than they are:

* macroscopic pKa values define charge-*macrostate* sensitivity, not
  site-resolved microstate populations;
* ETKDG/MMFF conformers are independent starting hypotheses for CREST/xTB,
  not solvent ensembles;
* deposited hERG coordinates support structural-state comparison, not docking
  affinity or molecular-dynamics readiness.

The expensive pieces are executed by a resumable driver in
``pipeline/scripts/run_local_m3_physics.py``.  This module contains the
deterministic scientific calculations so that they can be unit tested.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem, RDConfig
from rdkit.Chem import AllChem, ChemicalFeatures, rdFreeSASA

R_KCAL_MOL_K = 0.00198720425864083
LOCAL_PHYSICS_VERSION = "local-m3-mechanistic-pilot-v2"


@dataclass(frozen=True)
class MacrostateResult:
    """Charge-macrostate distribution from ordered macroscopic base pKas."""

    ph: float
    pka_offset: float
    probabilities: tuple[float, ...]

    @property
    def neutral_fraction(self) -> float:
        return self.probabilities[0]

    @property
    def mean_positive_charge(self) -> float:
        return float(sum(index * value for index, value in enumerate(self.probabilities)))


def polybase_macrostate_distribution(
    basic_pkas: Sequence[float],
    *,
    ph: float,
    pka_offset: float = 0.0,
) -> MacrostateResult:
    """Return probabilities for B, BH+, BH2+, ... from stepwise macro-pKas.

    The supplied values are sorted from the strongest to weakest basic
    protonation step.  If ``K_i`` is the equilibrium constant for adding the
    ``i``th proton, the unnormalised weight of the state carrying ``k``
    protons is

    ``10 ** sum_{i=1..k}(pKa_i - pH)``.

    This is exact for declared stepwise *macroscopic* constants.  It does not
    assign the proton to an atom and therefore must not be called a microstate
    distribution.
    """

    if not math.isfinite(ph):
        raise ValueError("pH must be finite")
    pkas = np.asarray(sorted((float(value) for value in basic_pkas), reverse=True), dtype=float)
    if pkas.size == 0 or not np.isfinite(pkas).all():
        raise ValueError("At least one finite basic pKa is required")
    log10_weights = np.concatenate(
        [
            np.asarray([0.0]),
            np.cumsum(pkas + float(pka_offset) - float(ph)),
        ]
    )
    shifted = log10_weights - float(np.max(log10_weights))
    weights = np.power(10.0, np.clip(shifted, -300.0, 0.0))
    probabilities = weights / weights.sum()
    return MacrostateResult(
        ph=float(ph),
        pka_offset=float(pka_offset),
        probabilities=tuple(float(value) for value in probabilities),
    )


def acid_deprotonated_fraction(
    acidic_pka: float,
    *,
    ph: float,
    pka_offset: float = 0.0,
) -> float:
    """Henderson-Hasselbalch deprotonated fraction for one acidic macrostep."""

    exponent = float(acidic_pka) + float(pka_offset) - float(ph)
    if exponent >= 300.0:
        return 0.0
    if exponent <= -300.0:
        return 1.0
    return float(1.0 / (1.0 + 10.0**exponent))


def rare_state_flux_threshold(
    state_fraction: float,
    *,
    target_flux_fraction: float,
    temperature_kelvin: float = 310.0,
) -> dict[str, float]:
    """Return the state-specific transport advantage needed for target flux.

    For a rare state with population ``f`` and all other states represented by
    one effective permeability ``P_other``,

    ``flux_rare = f R / (f R + 1 - f)``, where ``R=P_rare/P_other``.

    The returned free-energy advantage is ``RT ln(R)``.  It is a threshold,
    not a calculated membrane free energy.
    """

    fraction = float(state_fraction)
    target = float(target_flux_fraction)
    if not 0.0 < fraction < 1.0:
        raise ValueError("state_fraction must lie strictly between zero and one")
    if not 0.0 < target < 1.0:
        raise ValueError("target_flux_fraction must lie strictly between zero and one")
    ratio = target * (1.0 - fraction) / ((1.0 - target) * fraction)
    return {
        "required_state_specific_permeability_ratio": float(ratio),
        "required_free_energy_advantage_kcal_mol": float(
            R_KCAL_MOL_K * float(temperature_kelvin) * math.log(ratio)
        ),
    }


def _mmff_energies(mol: Chem.Mol, conformer_ids: Iterable[int]) -> dict[int, float]:
    properties = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94s")
    if properties is None:
        raise ValueError("MMFF94s parameters are unavailable; no UFF fallback is permitted")
    energies: dict[int, float] = {}
    for conformer_id in conformer_ids:
        force_field = AllChem.MMFFGetMoleculeForceField(
            mol,
            properties,
            confId=int(conformer_id),
            nonBondedThresh=100.0,
        )
        if force_field is None:
            raise ValueError(f"MMFF94s force field construction failed for conformer {conformer_id}")
        status = int(force_field.Minimize(maxIts=2000, energyTol=1e-6, forceTol=1e-4))
        if status != 0:
            raise ValueError(f"MMFF94s minimization did not converge for conformer {conformer_id}")
        energy = float(force_field.CalcEnergy())
        if not math.isfinite(energy):
            raise ValueError(f"MMFF94s returned non-finite energy for conformer {conformer_id}")
        energies[int(conformer_id)] = energy
    return energies


def _heavy_atom_best_rms(left: Chem.Mol, right: Chem.Mol) -> float:
    left_heavy = Chem.RemoveHs(Chem.Mol(left))
    right_heavy = Chem.RemoveHs(Chem.Mol(right))
    return float(AllChem.GetBestRMS(left_heavy, right_heavy, prbId=0, refId=0))


def generate_independent_starting_conformers(
    smiles: str,
    *,
    seeds: Sequence[int],
    pool_size: int = 64,
    minimum_heavy_atom_rmsd_angstrom: float = 1.5,
) -> list[tuple[Chem.Mol, dict[str, Any]]]:
    """Generate low-energy but geometrically independent CREST starts.

    Each seed creates a separate ETKDGv3 pool.  The first seed contributes its
    minimum-energy conformer.  Later seeds contribute the lowest-energy
    conformer whose heavy-atom best-RMSD from every already selected start is
    at least the declared threshold.  If no such conformer exists, the
    minimum-energy conformer is retained and the independence gate is marked
    failed rather than hidden.
    """

    parent = Chem.MolFromSmiles(smiles)
    if parent is None:
        raise ValueError("Invalid SMILES")
    parent = Chem.AddHs(parent)
    heavy_atom_count = int(parent.GetNumHeavyAtoms())
    outputs: list[tuple[Chem.Mol, dict[str, Any]]] = []
    for seed_index, seed in enumerate(seeds):
        mol = Chem.Mol(parent)
        params = AllChem.ETKDGv3()
        params.randomSeed = int(seed)
        params.enforceChirality = True
        params.useRandomCoords = True
        params.pruneRmsThresh = -1.0
        params.numThreads = 1
        conformer_ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=int(pool_size), params=params))
        if len(conformer_ids) != int(pool_size):
            raise ValueError(f"ETKDG embedded {len(conformer_ids)}/{pool_size} requested conformers")
        energies = _mmff_energies(mol, conformer_ids)
        ranked = sorted(conformer_ids, key=lambda conf_id: (energies[int(conf_id)], int(conf_id)))
        selected_id = int(ranked[0])
        minimum_rmsd = float("nan")
        independence_passed = seed_index == 0
        if outputs:
            for candidate_id in ranked:
                candidate = Chem.Mol(mol)
                keep = Chem.Mol(candidate)
                keep.RemoveAllConformers()
                keep.AddConformer(
                    Chem.Conformer(candidate.GetConformer(int(candidate_id))),
                    assignId=True,
                )
                rmsds = [_heavy_atom_best_rms(previous, keep) for previous, _metadata in outputs]
                if min(rmsds) >= float(minimum_heavy_atom_rmsd_angstrom):
                    selected_id = int(candidate_id)
                    minimum_rmsd = float(min(rmsds))
                    independence_passed = True
                    break
            if not independence_passed:
                candidate = Chem.Mol(mol)
                keep = Chem.Mol(candidate)
                keep.RemoveAllConformers()
                keep.AddConformer(
                    Chem.Conformer(candidate.GetConformer(selected_id)),
                    assignId=True,
                )
                minimum_rmsd = min(_heavy_atom_best_rms(previous, keep) for previous, _metadata in outputs)

        selected = Chem.Mol(mol)
        selected.RemoveAllConformers()
        selected.AddConformer(Chem.Conformer(mol.GetConformer(selected_id)), assignId=True)
        outputs.append(
            (
                selected,
                {
                    "seed": int(seed),
                    "pool_size": int(pool_size),
                    "selected_pool_conformer_id": selected_id,
                    "selected_mmff94s_energy_kcal_mol": float(energies[selected_id]),
                    "heavy_atom_count": heavy_atom_count,
                    "minimum_rmsd_to_previous_start_angstrom": minimum_rmsd,
                    "independent_start_gate_passed": bool(independence_passed),
                    "minimum_requested_rmsd_angstrom": float(minimum_heavy_atom_rmsd_angstrom),
                },
            )
        )
    return outputs


def _single_conformer_copy(mol: Chem.Mol, conformer_id: int) -> Chem.Mol:
    copied = Chem.Mol(mol)
    copied.RemoveAllConformers()
    copied.AddConformer(
        Chem.Conformer(mol.GetConformer(int(conformer_id))),
        assignId=True,
    )
    return copied


def generate_diverse_basin_conformers(
    smiles: str,
    *,
    seeds: Sequence[int],
    pool_size: int = 64,
    maximum_per_seed: int = 2,
    energy_window_kcal_mol: float = 12.0,
    minimum_heavy_atom_rmsd_angstrom: float = 1.25,
) -> list[tuple[Chem.Mol, dict[str, Any]]]:
    """Select a small, auditable set of diverse MMFF basins for xTB refinement.

    Each random seed defines an independent ETKDGv3/MMFF94s pool.  Selection is
    performed *within* each pool so that the two seeds remain genuine
    replication attempts:

    1. retain the minimum-energy conformer;
    2. among conformers within the declared energy window, greedily retain the
       conformer with the largest minimum heavy-atom best-RMSD to the already
       retained basins;
    3. stop rather than fill the quota when no candidate clears the RMSD gate.

    The returned structures are local-minimum starting hypotheses.  Their
    counts and MMFF energies must not be converted to populations.
    """

    if int(maximum_per_seed) < 1:
        raise ValueError("maximum_per_seed must be at least one")
    if float(energy_window_kcal_mol) <= 0.0:
        raise ValueError("energy_window_kcal_mol must be positive")
    if float(minimum_heavy_atom_rmsd_angstrom) <= 0.0:
        raise ValueError("minimum_heavy_atom_rmsd_angstrom must be positive")

    parent = Chem.MolFromSmiles(smiles)
    if parent is None:
        raise ValueError("Invalid SMILES")
    parent = Chem.AddHs(parent)
    outputs: list[tuple[Chem.Mol, dict[str, Any]]] = []
    for seed in seeds:
        mol = Chem.Mol(parent)
        params = AllChem.ETKDGv3()
        params.randomSeed = int(seed)
        params.enforceChirality = True
        params.useRandomCoords = True
        params.pruneRmsThresh = -1.0
        params.numThreads = 1
        conformer_ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=int(pool_size), params=params))
        if len(conformer_ids) != int(pool_size):
            raise ValueError(f"ETKDG embedded {len(conformer_ids)}/{pool_size} requested conformers")
        energies = _mmff_energies(mol, conformer_ids)
        ranked = sorted(conformer_ids, key=lambda conf_id: (energies[int(conf_id)], int(conf_id)))
        minimum_energy = float(energies[int(ranked[0])])
        eligible = [
            int(conformer_id)
            for conformer_id in ranked
            if float(energies[int(conformer_id)]) - minimum_energy <= float(energy_window_kcal_mol)
        ]
        selected_ids = [eligible[0]]
        selected_minimum_rmsd = [float("nan")]
        while len(selected_ids) < int(maximum_per_seed):
            candidates: list[tuple[float, float, int]] = []
            selected_molecules = [_single_conformer_copy(mol, conformer_id) for conformer_id in selected_ids]
            for candidate_id in eligible:
                if candidate_id in selected_ids:
                    continue
                candidate = _single_conformer_copy(mol, candidate_id)
                minimum_rmsd = min(
                    _heavy_atom_best_rms(previous, candidate) for previous in selected_molecules
                )
                candidates.append(
                    (
                        float(minimum_rmsd),
                        -float(energies[candidate_id]),
                        int(candidate_id),
                    )
                )
            if not candidates:
                break
            best_rmsd, _negative_energy, best_id = max(candidates)
            if best_rmsd < float(minimum_heavy_atom_rmsd_angstrom):
                break
            selected_ids.append(int(best_id))
            selected_minimum_rmsd.append(float(best_rmsd))

        for basin_index, (conformer_id, minimum_rmsd) in enumerate(
            zip(selected_ids, selected_minimum_rmsd, strict=True),
            start=1,
        ):
            outputs.append(
                (
                    _single_conformer_copy(mol, conformer_id),
                    {
                        "seed": int(seed),
                        "basin_index_within_seed": int(basin_index),
                        "pool_size": int(pool_size),
                        "eligible_pool_conformer_count": len(eligible),
                        "selected_pool_conformer_id": int(conformer_id),
                        "selected_mmff94s_energy_kcal_mol": float(energies[int(conformer_id)]),
                        "mmff94s_delta_from_seed_minimum_kcal_mol": float(
                            energies[int(conformer_id)] - minimum_energy
                        ),
                        "minimum_rmsd_to_selected_seed_basin_angstrom": float(minimum_rmsd),
                        "energy_window_kcal_mol": float(energy_window_kcal_mol),
                        "minimum_requested_rmsd_angstrom": float(minimum_heavy_atom_rmsd_angstrom),
                        "maximum_per_seed": int(maximum_per_seed),
                        "selection_semantics": (
                            "diverse local-minimum hypotheses; not a sampled "
                            "population or conformer-count estimate"
                        ),
                    },
                )
            )
    return outputs


def basin_selection_sensitivity(
    smiles: str,
    *,
    seeds: Sequence[int],
    maximum_pool_size: int = 64,
    pool_size_prefixes: Sequence[int] = (32, 64),
    energy_windows_kcal_mol: Sequence[float] = (8.0, 12.0, 16.0),
    rmsd_gates_angstrom: Sequence[float] = (1.0, 1.25, 1.5),
    maximum_per_seed: int = 2,
) -> list[dict[str, Any]]:
    """Audit whether basin selection depends on computational gate settings.

    One maximum-size pool is embedded and minimized per seed.  Prefixes of that
    exact pool are then reselected across the declared energy-window and RMSD
    grid.  Reusing the pool isolates selection-rule sensitivity from a change in
    random coordinates.  It does not test whether the maximum pool is complete.
    """

    prefixes = tuple(int(value) for value in pool_size_prefixes)
    if not prefixes or min(prefixes) < 1 or max(prefixes) > int(maximum_pool_size):
        raise ValueError("pool_size_prefixes must lie within maximum_pool_size")
    parent = Chem.MolFromSmiles(smiles)
    if parent is None:
        raise ValueError("Invalid SMILES")
    parent = Chem.AddHs(parent)
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        mol = Chem.Mol(parent)
        params = AllChem.ETKDGv3()
        params.randomSeed = int(seed)
        params.enforceChirality = True
        params.useRandomCoords = True
        params.pruneRmsThresh = -1.0
        params.numThreads = 1
        conformer_ids = list(
            AllChem.EmbedMultipleConfs(
                mol,
                numConfs=int(maximum_pool_size),
                params=params,
            )
        )
        if len(conformer_ids) != int(maximum_pool_size):
            raise ValueError(f"ETKDG embedded {len(conformer_ids)}/{maximum_pool_size} conformers")
        energies = _mmff_energies(mol, conformer_ids)
        for prefix in prefixes:
            prefix_ids = conformer_ids[:prefix]
            ranked = sorted(
                prefix_ids,
                key=lambda conf_id: (energies[int(conf_id)], int(conf_id)),
            )
            minimum_energy = float(energies[int(ranked[0])])
            for energy_window in energy_windows_kcal_mol:
                eligible = [
                    int(conformer_id)
                    for conformer_id in ranked
                    if float(energies[int(conformer_id)]) - minimum_energy <= float(energy_window)
                ]
                for rmsd_gate in rmsd_gates_angstrom:
                    selected_ids = [eligible[0]]
                    selected_rmsds = [float("nan")]
                    while len(selected_ids) < int(maximum_per_seed):
                        selected_molecules = [
                            _single_conformer_copy(mol, conformer_id) for conformer_id in selected_ids
                        ]
                        candidates: list[tuple[float, float, int]] = []
                        for candidate_id in eligible:
                            if candidate_id in selected_ids:
                                continue
                            candidate = _single_conformer_copy(mol, candidate_id)
                            minimum_rmsd = min(
                                _heavy_atom_best_rms(previous, candidate) for previous in selected_molecules
                            )
                            candidates.append(
                                (
                                    float(minimum_rmsd),
                                    -float(energies[candidate_id]),
                                    int(candidate_id),
                                )
                            )
                        if not candidates:
                            break
                        best_rmsd, _negative_energy, best_id = max(candidates)
                        if best_rmsd < float(rmsd_gate):
                            break
                        selected_ids.append(int(best_id))
                        selected_rmsds.append(float(best_rmsd))
                    setting_id = f"pool{prefix}__window{float(energy_window):g}__rmsd{float(rmsd_gate):g}"
                    for selected_index, (conformer_id, minimum_rmsd) in enumerate(
                        zip(selected_ids, selected_rmsds, strict=True),
                        start=1,
                    ):
                        rows.append(
                            {
                                "seed": int(seed),
                                "setting_id": setting_id,
                                "pool_size_prefix": int(prefix),
                                "energy_window_kcal_mol": float(energy_window),
                                "rmsd_gate_angstrom": float(rmsd_gate),
                                "selected_basin_index": int(selected_index),
                                "selected_pool_conformer_id": int(conformer_id),
                                "selected_mmff94s_energy_kcal_mol": float(energies[int(conformer_id)]),
                                "delta_from_prefix_minimum_kcal_mol": float(
                                    energies[int(conformer_id)] - minimum_energy
                                ),
                                "minimum_rmsd_to_prior_selected_angstrom": float(minimum_rmsd),
                                "eligible_conformer_count": len(eligible),
                                "selection_semantics": (
                                    "computational gate sensitivity; not a "
                                    "thermodynamic state or population analysis"
                                ),
                            }
                        )
    return rows


def mol_to_xyz_text(mol: Chem.Mol, *, comment: str) -> str:
    """Serialize the sole conformer of a hydrogen-explicit molecule to XYZ."""

    if mol.GetNumConformers() != 1:
        raise ValueError("XYZ serialization requires exactly one conformer")
    coordinates = np.asarray(mol.GetConformer(0).GetPositions(), dtype=float)
    if coordinates.shape != (mol.GetNumAtoms(), 3) or not np.isfinite(coordinates).all():
        raise ValueError("XYZ coordinates are incomplete or non-finite")
    lines = [str(mol.GetNumAtoms()), comment]
    for atom, (x, y, z) in zip(mol.GetAtoms(), coordinates, strict=True):
        lines.append(f"{atom.GetSymbol():<2s} {x: .10f} {y: .10f} {z: .10f}")
    return "\n".join(lines) + "\n"


def read_xyz_ensemble(path: str | Path) -> tuple[list[str], list[np.ndarray], list[str]]:
    """Read a single- or multi-structure XYZ file with fixed atom ordering."""

    lines = Path(path).read_text(encoding="utf-8").splitlines()
    position = 0
    symbols_reference: list[str] | None = None
    frames: list[np.ndarray] = []
    comments: list[str] = []
    while position < len(lines):
        if not lines[position].strip():
            position += 1
            continue
        atom_count = int(lines[position].strip())
        if position + atom_count + 1 >= len(lines):
            raise ValueError("XYZ ensemble ends with a partial structure")
        comment = lines[position + 1]
        symbols: list[str] = []
        coordinates: list[list[float]] = []
        for line in lines[position + 2 : position + 2 + atom_count]:
            tokens = line.split()
            if len(tokens) < 4:
                raise ValueError("XYZ atom row contains fewer than four fields")
            symbols.append(tokens[0])
            coordinates.append([float(tokens[1]), float(tokens[2]), float(tokens[3])])
        if symbols_reference is None:
            symbols_reference = symbols
        elif symbols != symbols_reference:
            raise ValueError("XYZ ensemble changes atom identity or ordering between frames")
        array = np.asarray(coordinates, dtype=float)
        if not np.isfinite(array).all():
            raise ValueError("XYZ ensemble contains non-finite coordinates")
        frames.append(array)
        comments.append(comment)
        position += atom_count + 2
    if symbols_reference is None or not frames:
        raise ValueError("XYZ ensemble contains no structures")
    return symbols_reference, frames, comments


def _feature_factory() -> Any:
    feature_file = Path(RDConfig.RDDataDir) / "BaseFeatures.fdef"
    return ChemicalFeatures.BuildFeatureFactory(str(feature_file))


def _donor_acceptor_atoms(mol: Chem.Mol) -> tuple[set[int], set[int]]:
    donors: set[int] = set()
    acceptors: set[int] = set()
    for feature in _feature_factory().GetFeaturesForMol(mol):
        if feature.GetFamily() == "Donor":
            donors.update(int(index) for index in feature.GetAtomIds())
        elif feature.GetFamily() == "Acceptor":
            acceptors.update(int(index) for index in feature.GetAtomIds())
    return donors, acceptors


def geometry_observables(
    hydrogenated_template: Chem.Mol,
    coordinates: np.ndarray,
) -> dict[str, float]:
    """Calculate a compact set of interpretable geometry diagnostics."""

    mol = Chem.Mol(hydrogenated_template)
    mol.RemoveAllConformers()
    conformer = Chem.Conformer(mol.GetNumAtoms())
    for index, (x, y, z) in enumerate(np.asarray(coordinates, dtype=float)):
        conformer.SetAtomPosition(index, (float(x), float(y), float(z)))
    mol.AddConformer(conformer, assignId=True)

    masses = np.asarray([atom.GetMass() for atom in mol.GetAtoms()], dtype=float)
    center = np.average(coordinates, axis=0, weights=masses)
    rg = math.sqrt(float(np.average(np.sum(np.square(coordinates - center), axis=1), weights=masses)))

    radii = rdFreeSASA.classifyAtoms(mol)
    total_sasa = float(rdFreeSASA.CalcSASA(mol, radii, confIdx=0))
    polar_heavy_sasa = 0.0
    carbon_halogen_heavy_sasa = 0.0
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1 or not atom.HasProp("SASA"):
            continue
        atom_sasa = float(atom.GetProp("SASA"))
        if atom.GetAtomicNum() in {7, 8, 15, 16}:
            polar_heavy_sasa += atom_sasa
        elif atom.GetAtomicNum() in {6, 9, 17, 35, 53}:
            carbon_halogen_heavy_sasa += atom_sasa
    donors, acceptors = _donor_acceptor_atoms(mol)
    topology = Chem.GetDistanceMatrix(mol)
    imhb_pairs = 0
    for donor in donors:
        for acceptor in acceptors:
            if donor == acceptor or topology[donor, acceptor] < 4:
                continue
            if float(np.linalg.norm(coordinates[donor] - coordinates[acceptor])) <= 3.5:
                imhb_pairs += 1
    return {
        "radius_of_gyration_angstrom": float(rg),
        "total_sasa_angstrom2": total_sasa,
        "polar_heavy_atom_sasa_angstrom2": float(polar_heavy_sasa),
        "carbon_halogen_heavy_atom_sasa_angstrom2": float(carbon_halogen_heavy_sasa),
        "imhb_heavy_atom_distance_pair_count": float(imhb_pairs),
    }


def protonation_site_audit(smiles: str) -> list[dict[str, Any]]:
    """Classify nitrogen atoms for an auditable, site-specific +1 pilot.

    This is deliberately a chemical-candidate audit rather than a pKa
    predictor.  Resonance-deactivated amide/carbamate and sulfonamide
    nitrogens, already protonated atoms, and non-accepting pyrrolic aromatic
    nitrogens are excluded.  Neutral aliphatic, aniline-like, and
    heteroaromatic acceptor nitrogens are retained as protonation hypotheses.
    """

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES")
    factory = _feature_factory()
    acceptors: set[int] = set()
    positive_ionizable: set[int] = set()
    for feature in factory.GetFeaturesForMol(mol):
        if feature.GetFamily() == "Acceptor":
            acceptors.update(int(index) for index in feature.GetAtomIds())
        elif feature.GetFamily() == "PosIonizable":
            positive_ionizable.update(int(index) for index in feature.GetAtomIds())

    rows: list[dict[str, Any]] = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 7:
            continue
        index = int(atom.GetIdx())
        bonded_to_carbonyl = False
        bonded_to_sulfonyl = False
        for neighbor in atom.GetNeighbors():
            double_oxygen_count = sum(
                bond.GetBondType() == Chem.BondType.DOUBLE and bond.GetOtherAtom(neighbor).GetAtomicNum() == 8
                for bond in neighbor.GetBonds()
            )
            if neighbor.GetAtomicNum() == 6 and double_oxygen_count >= 1:
                bonded_to_carbonyl = True
            if neighbor.GetAtomicNum() == 16 and double_oxygen_count >= 2:
                bonded_to_sulfonyl = True

        exclusion_reason = ""
        if atom.GetFormalCharge() != 0:
            exclusion_reason = "already_formally_charged"
        elif bonded_to_carbonyl:
            exclusion_reason = "amide_or_carbamate_resonance_deactivated"
        elif bonded_to_sulfonyl:
            exclusion_reason = "sulfonamide_resonance_deactivated"
        elif atom.GetIsAromatic() and index not in acceptors:
            exclusion_reason = "pyrrolic_or_substituted_nonaccepting_aromatic_nitrogen"

        if atom.GetIsAromatic():
            site_class = "heteroaromatic_acceptor"
        elif atom.GetHybridization() == Chem.HybridizationType.SP3:
            site_class = "aliphatic_amine"
        else:
            site_class = "conjugated_or_aniline_like_amine"
        rows.append(
            {
                "atom_index_zero_based": index,
                "element": atom.GetSymbol(),
                "site_class": site_class,
                "aromatic": bool(atom.GetIsAromatic()),
                "heavy_atom_degree": int(atom.GetDegree()),
                "rdkit_acceptor": index in acceptors,
                "rdkit_positive_ionizable_feature": index in positive_ionizable,
                "candidate_for_site_ranking": not exclusion_reason,
                "exclusion_reason": exclusion_reason,
                "truth_boundary": (
                    "rule-based protonation candidate audit; neither a microscopic "
                    "pKa assignment nor a population estimate"
                ),
            }
        )
    return rows


def protonate_at_atom(smiles: str, atom_index_zero_based: int) -> Chem.Mol:
    """Return a sanitized heavy-atom molecule protonated at one audited site."""

    audit = {int(row["atom_index_zero_based"]): row for row in protonation_site_audit(smiles)}
    index = int(atom_index_zero_based)
    if index not in audit or not bool(audit[index]["candidate_for_site_ranking"]):
        raise ValueError(f"Atom {index} is not an admitted protonation-site candidate")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES")
    editable = Chem.RWMol(mol)
    atom = editable.GetAtomWithIdx(index)
    atom.SetFormalCharge(atom.GetFormalCharge() + 1)
    atom.SetNumExplicitHs(atom.GetNumExplicitHs() + 1)
    atom.SetNoImplicit(True)
    protonated = editable.GetMol()
    Chem.SanitizeMol(protonated)
    if Chem.GetFormalCharge(protonated) != 1:
        raise ValueError("Site-specific protonation did not produce a +1 molecule")
    return protonated


def add_hydrogens_to_heavy_coordinates(
    heavy_mol: Chem.Mol,
    heavy_coordinates: np.ndarray,
) -> Chem.Mol:
    """Attach hydrogens while preserving an audited heavy-atom geometry."""

    coordinates = np.asarray(heavy_coordinates, dtype=float)
    if coordinates.shape != (heavy_mol.GetNumAtoms(), 3) or not np.isfinite(coordinates).all():
        raise ValueError("Heavy-atom coordinates do not match the molecule")
    mol = Chem.Mol(heavy_mol)
    mol.RemoveAllConformers()
    conformer = Chem.Conformer(mol.GetNumAtoms())
    for index, (x, y, z) in enumerate(coordinates):
        conformer.SetAtomPosition(index, (float(x), float(y), float(z)))
    mol.AddConformer(conformer, assignId=True)
    hydrogenated = Chem.AddHs(mol, addCoords=True)
    if hydrogenated.GetNumConformers() != 1:
        raise ValueError("Hydrogen placement did not preserve one conformer")
    return hydrogenated


def site_specific_charge_observables(
    hydrogenated_template: Chem.Mol,
    coordinates: np.ndarray,
    atomic_charges: Sequence[float],
    *,
    protonated_atom_index_zero_based: int,
) -> dict[str, float]:
    """Measure distinct charged-state presentation diagnostics.

    The charge-centroid separation is origin invariant, unlike the dipole of a
    net-charged molecule.  All charge-partition quantities remain
    Hamiltonian-dependent diagnostics and are not binding free energies.
    """

    coords = np.asarray(coordinates, dtype=float)
    charges = np.asarray(tuple(float(value) for value in atomic_charges), dtype=float)
    if coords.shape != (hydrogenated_template.GetNumAtoms(), 3):
        raise ValueError("Coordinate count does not match template")
    if charges.shape != (hydrogenated_template.GetNumAtoms(),) or not np.isfinite(charges).all():
        raise ValueError("Atomic charges do not match template")
    cation_index = int(protonated_atom_index_zero_based)
    cation_atom = hydrogenated_template.GetAtomWithIdx(cation_index)
    if cation_atom.GetAtomicNum() != 7 or cation_atom.GetFormalCharge() != 1:
        raise ValueError("Declared cation center is not a formally protonated nitrogen")

    mol = Chem.Mol(hydrogenated_template)
    mol.RemoveAllConformers()
    conformer = Chem.Conformer(mol.GetNumAtoms())
    for index, (x, y, z) in enumerate(coords):
        conformer.SetAtomPosition(index, (float(x), float(y), float(z)))
    mol.AddConformer(conformer, assignId=True)
    radii = rdFreeSASA.classifyAtoms(mol)
    rdFreeSASA.CalcSASA(mol, radii, confIdx=0)
    cation_fragment = {cation_index}
    cation_fragment.update(int(atom.GetIdx()) for atom in cation_atom.GetNeighbors())
    cation_sasa = sum(
        float(mol.GetAtomWithIdx(index).GetProp("SASA"))
        for index in cation_fragment
        if mol.GetAtomWithIdx(index).HasProp("SASA")
    )

    edited_ring_candidates: list[tuple[int, tuple[int, ...]]] = []
    for ring in mol.GetRingInfo().AtomRings():
        if len(ring) != 5:
            continue
        ring_tuple = tuple(int(index) for index in ring)
        nitrogen_count = sum(mol.GetAtomWithIdx(index).GetAtomicNum() == 7 for index in ring_tuple)
        if nitrogen_count >= 2 and all(mol.GetAtomWithIdx(index).GetIsAromatic() for index in ring_tuple):
            edited_ring_candidates.append((nitrogen_count, ring_tuple))
    if len(edited_ring_candidates) != 1:
        raise ValueError("Expected exactly one edited five-membered heteroaromatic ring")
    _nitrogen_count, edited_ring = edited_ring_candidates[0]
    ring_centroid = np.mean(coords[list(edited_ring)], axis=0)
    cation_to_ring = float(np.linalg.norm(coords[cation_index] - ring_centroid))
    ring_nitrogens = [index for index in edited_ring if mol.GetAtomWithIdx(index).GetAtomicNum() == 7]

    positive_weights = np.clip(charges, 0.0, None)
    negative_weights = np.clip(-charges, 0.0, None)
    if positive_weights.sum() <= 0.0 or negative_weights.sum() <= 0.0:
        raise ValueError("Charge-centroid separation requires positive and negative atomic charges")
    positive_centroid = np.average(coords, axis=0, weights=positive_weights)
    negative_centroid = np.average(coords, axis=0, weights=negative_weights)
    charge_centroid_separation = float(np.linalg.norm(positive_centroid - negative_centroid))

    return {
        "total_atomic_charge": float(charges.sum()),
        "cation_fragment_sasa_angstrom2": float(cation_sasa),
        "cation_fragment_xtb_charge": float(charges[list(cation_fragment)].sum()),
        "edited_ring_xtb_charge": float(charges[list(edited_ring)].sum()),
        "edited_ring_nitrogen_xtb_charge": float(charges[ring_nitrogens].sum()),
        "cation_to_edited_ring_centroid_distance_angstrom": cation_to_ring,
        "positive_negative_charge_centroid_separation_angstrom": charge_centroid_separation,
    }


def kabsch_transform(
    mobile: np.ndarray,
    reference: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return rotation, translation, and fitted RMSD for row-vector coordinates."""

    mobile_array = np.asarray(mobile, dtype=float)
    reference_array = np.asarray(reference, dtype=float)
    if mobile_array.shape != reference_array.shape or mobile_array.ndim != 2:
        raise ValueError("Kabsch inputs must have equal (n, d) shape")
    if len(mobile_array) < 3:
        raise ValueError("Kabsch alignment requires at least three points")
    mobile_center = mobile_array.mean(axis=0)
    reference_center = reference_array.mean(axis=0)
    covariance = (mobile_array - mobile_center).T @ (reference_array - reference_center)
    left, _singular, right_t = np.linalg.svd(covariance)
    rotation = left @ right_t
    if np.linalg.det(rotation) < 0:
        left[:, -1] *= -1
        rotation = left @ right_t
    translation = reference_center - mobile_center @ rotation
    fitted = mobile_array @ rotation + translation
    rmsd = float(np.sqrt(np.mean(np.sum(np.square(fitted - reference_array), axis=1))))
    return rotation, translation, rmsd


def apply_transform(
    coordinates: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    return np.asarray(coordinates, dtype=float) @ rotation + translation


__all__ = [
    "LOCAL_PHYSICS_VERSION",
    "MacrostateResult",
    "acid_deprotonated_fraction",
    "apply_transform",
    "basin_selection_sensitivity",
    "generate_diverse_basin_conformers",
    "generate_independent_starting_conformers",
    "geometry_observables",
    "kabsch_transform",
    "mol_to_xyz_text",
    "polybase_macrostate_distribution",
    "rare_state_flux_threshold",
    "read_xyz_ensemble",
]
