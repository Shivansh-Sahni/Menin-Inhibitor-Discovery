"""Deterministic, state-aware fast physics for large-molecule research.

This module intentionally separates *approximations* from observations.  RDKit
is used to generate feasible tautomer/protomer hypotheses and conformers, but
the resulting populations are not microscopic-pKa predictions.  They are
Henderson-Hasselbalch sensitivity weights tied to caller-supplied evidence (or
an explicitly labelled heuristic when evidence is absent).

The fast layer is designed to triage compounds and choose expensive physics
pilots.  It must not be described as a replacement for constant-pH simulation,
free-energy calculations, or measured permeability.
"""

from __future__ import annotations

import json
import math
import multiprocessing
import os
import shutil
import tempfile
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field, fields, replace
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from .chemistry import standardize_smiles

try:  # pragma: no cover - the project dependency supplies RDKit in production.
    from rdkit import Chem, RDConfig, rdBase
    from rdkit.Chem import AllChem, ChemicalFeatures, Lipinski, rdFreeSASA, rdMolDescriptors
    from rdkit.Chem.MolStandardize import rdMolStandardize
    from rdkit.ML.Cluster import Butina

    RDKIT_AVAILABLE = True
    _RDKIT_IMPORT_ERROR = ""
except ImportError as exc:  # pragma: no cover
    Chem = ChemicalFeatures = RDConfig = rdBase = None  # type: ignore[assignment]
    AllChem = Lipinski = rdFreeSASA = rdMolDescriptors = None  # type: ignore[assignment]
    rdMolStandardize = Butina = None  # type: ignore[assignment]
    RDKIT_AVAILABLE = False
    _RDKIT_IMPORT_ERROR = str(exc)


FAST_PHYSICS_VERSION = "fast-physics-state-ensemble-v2-sampling-audit"
DEFAULT_PH_GRID = (2.0, 5.0, 6.5, 7.4)
PKA_SCENARIOS = (("pka_minus_1", -1.0), ("nominal", 0.0), ("pka_plus_1", 1.0))
STATE_POPULATION_AUDIT_THRESHOLDS = (0.01, 0.001, 0.0001)
STATE_SENSITIVITY_POPULATION_FLOOR = 0.0001
LOCAL_GENERATED_CONFORMER_DEPTH = 25
PILOT_ESCALATION_CONFORMER_DEPTHS = (50, 100, 250)
DEFERRED_VALIDATION_CONFORMER_CEILING = 250
PILOT_COMPARATOR_CONFORMER_DEPTH = 500
DEFAULT_STRUCTURE_WORKERS = 1
_CACHE_REQUIRED_FILES = (
    "states.parquet",
    "state_populations.parquet",
    "conformers.parquet",
    "ensemble_summary.parquet",
    "composites.parquet",
    "quality_gates.parquet",
    "conformers.sdf",
    "metadata.json",
)


@dataclass(frozen=True)
class PKaEvidence:
    """Approximate pKa evidence attached to a candidate ionization site.

    ``atom_index`` refers to the atom index in the standardized structure.
    ``atom_smarts`` is an alternative for callers that do not control atom
    ordering.  If neither is supplied, the evidence is used for the first
    compatible site only when it is the sole evidence item of that kind.
    """

    label: str
    pka: float
    kind: Literal["acid", "base"]
    source: str
    atom_index: int | None = None
    atom_smarts: str = ""

    def __post_init__(self) -> None:
        if not self.label.strip():
            raise ValueError("pKa evidence requires a non-empty label")
        if not self.source.strip():
            raise ValueError("pKa evidence requires a source or rationale")
        if not math.isfinite(self.pka):
            raise ValueError("pKa evidence must be finite")
        if self.kind not in {"acid", "base"}:
            raise ValueError("pKa evidence kind must be 'acid' or 'base'")


@dataclass(frozen=True)
class FastPhysicsConfig:
    """Configuration for deterministic state and conformer enumeration."""

    ph_grid: tuple[float, ...] = DEFAULT_PH_GRID
    herg_ph: float = 7.4
    min_state_population: float = 0.001
    max_tautomers: int = 16
    max_states: int = 24
    max_conformers_per_state: int = LOCAL_GENERATED_CONFORMER_DEPTH
    max_retained_conformers: int = LOCAL_GENERATED_CONFORMER_DEPTH
    cluster_rmsd_angstrom: float = 1.5
    temperature_kelvin: float = 298.15
    random_seed: int = 20260721
    default_acid_pka: float = 4.5
    default_base_pka: float = 7.0
    default_phenol_pka: float = 10.0
    exposed_atom_sasa_ang2: float = 5.0
    smoke_mode: bool = False
    smoke_conformers_per_state: int = 8
    smoke_retained_conformers: int = 4

    def __post_init__(self) -> None:
        ph_values = (*self.ph_grid, self.herg_ph)
        if any(not math.isfinite(value) or not 0.0 <= value <= 14.0 for value in ph_values):
            raise ValueError("All pH values must be finite and in [0, 14]")
        if not 0.0 < self.min_state_population < 1.0:
            raise ValueError("min_state_population must be between 0 and 1")
        if self.min_state_population < STATE_SENSITIVITY_POPULATION_FLOOR:
            raise ValueError("min_state_population cannot be below the declared 0.01% sensitivity floor")
        if self.max_tautomers < 1 or self.max_states < 1:
            raise ValueError("max_tautomers and max_states must be positive")
        if self.max_conformers_per_state < 1 or self.max_retained_conformers < 1:
            raise ValueError("conformer limits must be positive")
        if self.max_retained_conformers > self.max_conformers_per_state:
            raise ValueError("retained conformer limit cannot exceed generated limit")
        if self.smoke_conformers_per_state < 1 or self.smoke_retained_conformers < 1:
            raise ValueError("smoke conformer limits must be positive")
        if self.smoke_retained_conformers > self.smoke_conformers_per_state:
            raise ValueError("smoke retained conformer limit cannot exceed its generated limit")
        if self.cluster_rmsd_angstrom <= 0 or self.temperature_kelvin <= 0:
            raise ValueError("cluster RMSD and temperature must be positive")

    @property
    def all_ph_values(self) -> tuple[float, ...]:
        """Return the base grid followed by hERG pH, without duplicates."""

        return tuple(dict.fromkeys(float(value) for value in (*self.ph_grid, self.herg_ph)))

    @property
    def generated_conformer_limit(self) -> int:
        return self.smoke_conformers_per_state if self.smoke_mode else self.max_conformers_per_state

    @property
    def retained_conformer_limit(self) -> int:
        return self.smoke_retained_conformers if self.smoke_mode else self.max_retained_conformers


@dataclass
class FastPhysicsResult:
    """In-memory result plus conformer molecules needed for SDF export."""

    structure_id: str
    standardized_smiles: str
    states: pd.DataFrame
    populations: pd.DataFrame
    conformers: pd.DataFrame
    ensemble_summary: pd.DataFrame
    composites: pd.DataFrame
    quality: pd.DataFrame
    metadata: dict[str, Any]
    state_molecules: dict[str, Any] = field(default_factory=dict, repr=False)


@dataclass
class _Microstate:
    mol: Any
    smiles: str
    state_id: str
    tautomer_index: int
    transformation: str
    site_atom_index: int | None
    pka_kind: str
    pka_value: float | None
    pka_label: str
    pka_source: str
    pka_basis: str
    formal_charge: int
    hbd: int
    hba: int


def _require_rdkit() -> None:
    if not RDKIT_AVAILABLE:
        raise RuntimeError(f"RDKit is required for fast physics: {_RDKIT_IMPORT_ERROR}")


def _digest(*parts: object, length: int = 20) -> str:
    material = "\0".join(str(part) for part in parts)
    return sha256(material.encode()).hexdigest()[:length].upper()


def _seed_for(*parts: object, base_seed: int) -> int:
    value = int(_digest(base_seed, *parts, length=8), 16)
    return int(value % 2_000_000_000) + 1


def _canonical_smiles(mol: Any) -> str:
    return str(Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True))


def _tautomer_molecules(mol: Any, limit: int) -> list[Any]:
    enumerator = rdMolStandardize.TautomerEnumerator()
    # Specified stereochemistry is evidence.  RDKit's default tautomer policy
    # removes tetrahedral/bond stereo around transformed atoms, which is useful
    # for canonicalization but inappropriate for state-resolved physics.
    if hasattr(enumerator, "SetRemoveSp3Stereo"):
        enumerator.SetRemoveSp3Stereo(False)
    if hasattr(enumerator, "SetRemoveBondStereo"):
        enumerator.SetRemoveBondStereo(False)
    if hasattr(enumerator, "SetReassignStereo"):
        enumerator.SetReassignStereo(True)
    if hasattr(enumerator, "SetMaxTautomers"):
        enumerator.SetMaxTautomers(int(limit))
    enumerated = list(enumerator.Enumerate(mol))
    unique: dict[str, Any] = {}
    for tautomer in enumerated:
        try:
            Chem.SanitizeMol(tautomer)
        except Exception:
            continue
        unique.setdefault(_canonical_smiles(tautomer), Chem.Mol(tautomer))
    if not unique:
        unique[_canonical_smiles(mol)] = Chem.Mol(mol)
    return [unique[key] for key in sorted(unique)[:limit]]


def _is_carbonyl_like(atom: Any) -> bool:
    for bond in atom.GetBonds():
        other = bond.GetOtherAtom(atom)
        if bond.GetBondTypeAsDouble() >= 1.9 and other.GetAtomicNum() in {7, 8, 15, 16}:
            return True
    return False


def _is_amide_like_nitrogen(atom: Any) -> bool:
    if atom.GetAtomicNum() != 7:
        return False
    return any(
        neighbor.GetAtomicNum() in {6, 15, 16} and _is_carbonyl_like(neighbor)
        for neighbor in atom.GetNeighbors()
    )


def _candidate_sites(mol: Any, config: FastPhysicsConfig) -> list[dict[str, Any]]:
    sites: list[dict[str, Any]] = []
    for atom in mol.GetAtoms():
        idx = int(atom.GetIdx())
        atomic_number = atom.GetAtomicNum()
        charge = atom.GetFormalCharge()
        total_h = atom.GetTotalNumHs(includeNeighbors=True)

        if atomic_number == 7 and charge == 0 and not _is_amide_like_nitrogen(atom):
            if atom.GetTotalValence() <= 3 or atom.GetIsAromatic():
                sites.append(
                    {
                        "kind": "base",
                        "atom_index": idx,
                        "transformation": "protonated_base",
                        "heuristic_pka": config.default_base_pka,
                    }
                )

        if atomic_number in {8, 16} and charge == 0 and total_h > 0:
            neighbor = next(iter(atom.GetNeighbors()), None)
            if neighbor is not None and _is_carbonyl_like(neighbor):
                heuristic_pka = config.default_acid_pka
            elif neighbor is not None and neighbor.GetIsAromatic():
                heuristic_pka = config.default_phenol_pka
            else:
                continue
            sites.append(
                {
                    "kind": "acid",
                    "atom_index": idx,
                    "transformation": "deprotonated_acid",
                    "heuristic_pka": heuristic_pka,
                }
            )
    return sites


def _matches_evidence(mol: Any, atom_index: int, evidence: PKaEvidence) -> bool:
    if evidence.atom_index is not None:
        return evidence.atom_index == atom_index
    if evidence.atom_smarts:
        pattern = Chem.MolFromSmarts(evidence.atom_smarts)
        if pattern is None:
            return False
        return any(atom_index in match for match in mol.GetSubstructMatches(pattern))
    return False


def _site_evidence(
    mol: Any,
    site: dict[str, Any],
    evidence: tuple[PKaEvidence, ...],
) -> tuple[float, str, str, str]:
    compatible = [item for item in evidence if item.kind == site["kind"]]
    direct = [item for item in compatible if _matches_evidence(mol, site["atom_index"], item)]
    if direct:
        chosen = sorted(direct, key=lambda item: (item.label, item.source))[0]
        return chosen.pka, chosen.label, chosen.source, "documented_site_evidence"
    unlocated = [item for item in compatible if item.atom_index is None and not item.atom_smarts]
    if len(compatible) == 1 and len(unlocated) == 1:
        chosen = unlocated[0]
        return chosen.pka, chosen.label, chosen.source, "documented_unlocated_evidence"
    return (
        float(site["heuristic_pka"]),
        f"heuristic_{site['kind']}_site_{site['atom_index']}",
        "FastPhysicsConfig default; replace with measured or cited site evidence",
        "heuristic_not_micro_pka",
    )


def _apply_site_transformation(mol: Any, site: dict[str, Any]) -> Any | None:
    editable = Chem.RWMol(mol)
    atom = editable.GetAtomWithIdx(int(site["atom_index"]))
    try:
        total_h = int(atom.GetTotalNumHs(includeNeighbors=True))
        if site["kind"] == "base":
            atom.SetFormalCharge(atom.GetFormalCharge() + 1)
            atom.SetNumExplicitHs(total_h + 1)
            atom.SetNoImplicit(True)
        else:
            if total_h < 1:
                return None
            atom.SetFormalCharge(atom.GetFormalCharge() - 1)
            atom.SetNumExplicitHs(max(0, total_h - 1))
            atom.SetNoImplicit(True)
        transformed = editable.GetMol()
        Chem.SanitizeMol(transformed)
        return transformed
    except Exception:
        return None


def _microstate_id(structure_id: str, smiles: str) -> str:
    return f"MST-{_digest(FAST_PHYSICS_VERSION, structure_id, smiles)}"


def _make_microstate(
    mol: Any,
    *,
    structure_id: str,
    tautomer_index: int,
    transformation: str,
    site_atom_index: int | None = None,
    pka_kind: str = "",
    pka_value: float | None = None,
    pka_label: str = "",
    pka_source: str = "",
    pka_basis: str = "reference_tautomer",
) -> _Microstate:
    smiles = _canonical_smiles(mol)
    return _Microstate(
        mol=Chem.Mol(mol),
        smiles=smiles,
        state_id=_microstate_id(structure_id, smiles),
        tautomer_index=tautomer_index,
        transformation=transformation,
        site_atom_index=site_atom_index,
        pka_kind=pka_kind,
        pka_value=pka_value,
        pka_label=pka_label,
        pka_source=pka_source,
        pka_basis=pka_basis,
        formal_charge=int(Chem.GetFormalCharge(mol)),
        hbd=int(Lipinski.NumHDonors(mol)),
        hba=int(Lipinski.NumHAcceptors(mol)),
    )


def _enumerate_raw_states(
    mol: Any,
    *,
    structure_id: str,
    evidence: tuple[PKaEvidence, ...],
    config: FastPhysicsConfig,
) -> list[_Microstate]:
    unique: dict[str, _Microstate] = {}
    for tautomer_index, tautomer in enumerate(_tautomer_molecules(mol, config.max_tautomers)):
        reference = _make_microstate(
            tautomer,
            structure_id=structure_id,
            tautomer_index=tautomer_index,
            transformation="reference_tautomer",
        )
        unique.setdefault(reference.smiles, reference)
        for site in _candidate_sites(tautomer, config):
            transformed = _apply_site_transformation(tautomer, site)
            if transformed is None:
                continue
            pka, label, source, basis = _site_evidence(tautomer, site, evidence)
            state = _make_microstate(
                transformed,
                structure_id=structure_id,
                tautomer_index=tautomer_index,
                transformation=str(site["transformation"]),
                site_atom_index=int(site["atom_index"]),
                pka_kind=str(site["kind"]),
                pka_value=float(pka),
                pka_label=label,
                pka_source=source,
                pka_basis=basis,
            )
            existing = unique.get(state.smiles)
            if existing is None or ("documented" in basis and "documented" not in existing.pka_basis):
                unique[state.smiles] = state
    return [unique[key] for key in sorted(unique)]


def _relative_state_score(state: _Microstate, ph: float, pka_offset: float) -> float:
    if state.pka_value is None or not state.pka_kind:
        return 1.0
    pka = state.pka_value + pka_offset
    exponent = pka - ph if state.pka_kind == "base" else ph - pka
    return float(10.0 ** np.clip(exponent, -12.0, 12.0))


def _population_table(states: list[_Microstate], config: FastPhysicsConfig) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for ph in config.all_ph_values:
        for scenario, offset in PKA_SCENARIOS:
            raw = np.asarray([_relative_state_score(state, ph, offset) for state in states], dtype=float)
            raw /= raw.sum()
            for state, value in zip(states, raw, strict=True):
                rows.append(
                    {
                        "state_id": state.state_id,
                        "ph": float(ph),
                        "pka_scenario": scenario,
                        "pka_offset": float(offset),
                        "raw_probability_pre_retention": float(value),
                    }
                )
    return pd.DataFrame(rows)


def _is_zwitterionic_state(state: _Microstate) -> bool:
    atom_charges = [int(atom.GetFormalCharge()) for atom in state.mol.GetAtoms()]
    return any(charge > 0 for charge in atom_charges) and any(charge < 0 for charge in atom_charges)


def _state_threshold_audit(
    populations: pd.DataFrame,
    *,
    retained_ids: set[str],
) -> list[dict[str, Any]]:
    """Quantify threshold effects without treating a threshold as physical truth."""

    rows: list[dict[str, Any]] = []
    for (ph, scenario), group in populations.groupby(["ph", "pka_scenario"], sort=True):
        retained = group["state_id"].astype(str).isin(retained_ids)
        retained_mass = float(group.loc[retained, "raw_probability_pre_retention"].sum())
        for threshold in STATE_POPULATION_AUDIT_THRESHOLDS:
            qualifies = group["raw_probability_pre_retention"] >= threshold
            omitted_qualifying = qualifies & ~retained
            qualifying_mass = float(group.loc[qualifies, "raw_probability_pre_retention"].sum())
            rows.append(
                {
                    "ph": float(ph),
                    "pka_scenario": str(scenario),
                    "threshold_fraction": float(threshold),
                    "threshold_percent": float(100.0 * threshold),
                    "raw_state_count": int(len(group)),
                    "states_at_or_above_threshold": int(qualifies.sum()),
                    "probability_mass_at_or_above_threshold": qualifying_mass,
                    "probability_mass_below_threshold": float(max(0.0, 1.0 - qualifying_mass)),
                    "retained_state_count": int(retained.sum()),
                    "retained_probability_mass": retained_mass,
                    "omitted_probability_mass_after_retention": float(max(0.0, 1.0 - retained_mass)),
                    "qualifying_states_omitted_by_cap": int(omitted_qualifying.sum()),
                    "qualifying_probability_mass_omitted_by_cap": float(
                        group.loc[omitted_qualifying, "raw_probability_pre_retention"].sum()
                    ),
                    "interpretation": (
                        "Sensitivity accounting only; approximate Henderson-Hasselbalch weights are "
                        "not measured microstate populations."
                    ),
                }
            )
    return rows


def _retain_states(
    states: list[_Microstate], populations: pd.DataFrame, config: FastPhysicsConfig
) -> tuple[list[_Microstate], pd.DataFrame, dict[str, Any]]:
    nominal = populations.loc[populations["pka_scenario"] == "nominal"]
    max_population = nominal.groupby("state_id")["raw_probability_pre_retention"].max().to_dict()
    reference_signatures = {
        (state.tautomer_index, state.formal_charge, state.hbd, state.hba)
        for state in states
        if state.transformation == "reference_tautomer"
    }
    reasons: dict[str, set[str]] = {state.state_id: set() for state in states}
    for state in states:
        if state.transformation == "reference_tautomer":
            reasons[state.state_id].add("reference_tautomer")
        if max_population.get(state.state_id, 0.0) >= config.min_state_population:
            reasons[state.state_id].add("nominal_population_at_or_above_0.1_percent")
        elif max_population.get(state.state_id, 0.0) >= STATE_SENSITIVITY_POPULATION_FLOOR:
            reasons[state.state_id].add("population_sensitivity_candidate_at_or_above_0.01_percent")
        signature = (state.tautomer_index, state.formal_charge, state.hbd, state.hba)
        if signature not in reference_signatures:
            reasons[state.state_id].add("charge_or_exposure_signature_change")

    for charge in sorted({state.formal_charge for state in states}):
        options = [state for state in states if state.formal_charge == charge]
        chosen = max(options, key=lambda state: (max_population.get(state.state_id, 0.0), state.state_id))
        reasons[chosen.state_id].add("charge_state_representative")

    neutral = [state for state in states if state.formal_charge == 0]
    if neutral:
        chosen = max(neutral, key=lambda state: (max_population.get(state.state_id, 0.0), state.state_id))
        reasons[chosen.state_id].add("rare_neutral_state_representative")
    zwitterionic = [state for state in states if _is_zwitterionic_state(state)]
    if zwitterionic:
        chosen = max(
            zwitterionic,
            key=lambda state: (max_population.get(state.state_id, 0.0), state.state_id),
        )
        reasons[chosen.state_id].add("rare_zwitterionic_state_representative")

    exposure_groups: dict[tuple[int, int, int], list[_Microstate]] = {}
    for state in states:
        exposure_groups.setdefault((state.formal_charge, state.hbd, state.hba), []).append(state)
    for options in exposure_groups.values():
        chosen = max(options, key=lambda state: (max_population.get(state.state_id, 0.0), state.state_id))
        reasons[chosen.state_id].add("unique_charge_hbond_capacity_representative")

    ranked = sorted(
        states,
        key=lambda state: (
            not bool(reasons[state.state_id]),
            -max_population.get(state.state_id, 0.0),
            state.state_id,
        ),
    )
    qualifying = [state for state in ranked if reasons[state.state_id]]
    retained = qualifying[: config.max_states]
    retained_ids = {state.state_id for state in retained}
    retained_pop = populations.loc[populations["state_id"].isin(retained_ids)].copy()
    group_cols = ["ph", "pka_scenario"]
    retained_pop["retained_probability_mass"] = retained_pop.groupby(group_cols)[
        "raw_probability_pre_retention"
    ].transform("sum")
    retained_pop["state_weight"] = (
        retained_pop["raw_probability_pre_retention"] / retained_pop["retained_probability_mass"]
    )
    retained_pop["retention_reason"] = retained_pop["state_id"].map(
        lambda state_id: ";".join(sorted(reasons[state_id]))
    )
    retained_pop = retained_pop.sort_values(["ph", "pka_scenario", "state_id"]).reset_index(drop=True)
    discarded = 1.0 - retained_pop.groupby(group_cols)["retained_probability_mass"].first()
    omitted_qualifying_ids = {state.state_id for state in qualifying[config.max_states :]}
    threshold_audit = _state_threshold_audit(populations, retained_ids=retained_ids)
    metadata = {
        "raw_state_count": len(states),
        "retained_state_count": len(retained),
        "nominal_population_threshold_fraction": float(config.min_state_population),
        "sensitivity_population_floor_fraction": STATE_SENSITIVITY_POPULATION_FLOOR,
        "thresholds_audited_fraction": list(STATE_POPULATION_AUDIT_THRESHOLDS),
        "state_cap_applied": bool(omitted_qualifying_ids),
        "qualifying_states_omitted_by_cap": len(omitted_qualifying_ids),
        "qualifying_state_ids_omitted_by_cap": sorted(omitted_qualifying_ids),
        "maximum_discarded_probability_mass": float(discarded.max()) if not discarded.empty else 0.0,
        "threshold_audit": threshold_audit,
        "retention_interpretation": (
            "0.1% is the nominal inclusion threshold; states down to 0.01% are retained for "
            "sensitivity when the state budget permits. Neutral, zwitterionic, charge, and "
            "H-bond-capacity representatives are explicit exceptions. max_states is a compute "
            "cap, not a physical target; omission of any qualifying state is a hard admissibility failure."
        ),
    }
    return retained, retained_pop, metadata


@lru_cache(maxsize=1)
def _feature_factory() -> Any:
    return ChemicalFeatures.BuildFeatureFactory(str(Path(RDConfig.RDDataDir) / "BaseFeatures.fdef"))


def _donor_acceptor_atoms(mol: Any) -> tuple[set[int], set[int]]:
    donors: set[int] = set()
    acceptors: set[int] = set()
    for feature in _feature_factory().GetFeaturesForMol(mol):
        family = feature.GetFamily()
        if family == "Donor":
            donors.update(int(index) for index in feature.GetAtomIds())
        elif family == "Acceptor":
            acceptors.update(int(index) for index in feature.GetAtomIds())
    return donors, acceptors


def _minimize_conformers(mol: Any) -> tuple[dict[int, float], dict[int, int], str]:
    energies: dict[int, float] = {}
    status: dict[int, int] = {}
    method = "MMFF94s"
    properties = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94s")
    for conformer in mol.GetConformers():
        conf_id = int(conformer.GetId())
        force_field = None
        if properties is not None:
            force_field = AllChem.MMFFGetMoleculeForceField(mol, properties, confId=conf_id)
        if force_field is None:
            method = "UFF_fallback"
            try:
                force_field = AllChem.UFFGetMoleculeForceField(mol, confId=conf_id)
            except Exception:
                force_field = None
        if force_field is None:
            energies[conf_id] = float("nan")
            status[conf_id] = -1
            continue
        try:
            status[conf_id] = int(force_field.Minimize(maxIts=500))
            energies[conf_id] = float(force_field.CalcEnergy())
        except Exception:
            energies[conf_id] = float("nan")
            status[conf_id] = -1
    return energies, status, method


def _cluster_and_select(
    mol: Any,
    energies: dict[int, float],
    *,
    cutoff: float,
    limit: int,
) -> tuple[list[int], dict[int, int]]:
    conf_ids = [int(conf.GetId()) for conf in mol.GetConformers()]
    if len(conf_ids) <= 1:
        return conf_ids, {conf_id: 0 for conf_id in conf_ids}
    distances = list(AllChem.GetConformerRMSMatrix(mol, prealigned=False))
    clusters = Butina.ClusterData(distances, len(conf_ids), cutoff, isDistData=True, reordering=True)
    cluster_by_conf: dict[int, int] = {}
    representatives: list[int] = []
    for cluster_index, cluster in enumerate(clusters):
        member_ids = [conf_ids[int(position)] for position in cluster]
        for conf_id in member_ids:
            cluster_by_conf[conf_id] = cluster_index
        representatives.append(
            min(
                member_ids,
                key=lambda conf_id: (
                    math.inf if not math.isfinite(energies[conf_id]) else energies[conf_id],
                    conf_id,
                ),
            )
        )
    selected = sorted(
        representatives,
        key=lambda conf_id: (
            math.inf if not math.isfinite(energies[conf_id]) else energies[conf_id],
            conf_id,
        ),
    )[:limit]
    if len(selected) < min(limit, len(conf_ids)):
        remainder = sorted(
            (conf_id for conf_id in conf_ids if conf_id not in selected),
            key=lambda conf_id: (
                math.inf if not math.isfinite(energies[conf_id]) else energies[conf_id],
                conf_id,
            ),
        )
        selected.extend(remainder[: min(limit, len(conf_ids)) - len(selected)])
    return selected, cluster_by_conf


def _boltzmann_weights(energies: list[float], temperature: float) -> np.ndarray:
    values = np.asarray(energies, dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        return np.full(len(values), 1.0 / len(values), dtype=float)
    replacement = float(np.nanmax(values[finite]) + 10.0)
    values = np.where(finite, values, replacement)
    relative = values - values.min()
    exponent = np.clip(-relative / (0.00198720425864083 * temperature), -700.0, 0.0)
    weights = np.exp(exponent)
    return weights / weights.sum()


def _gasteiger_charges(mol: Any) -> np.ndarray:
    try:
        AllChem.ComputeGasteigerCharges(mol, nIter=12, throwOnParamFailure=False)
    except Exception:
        return np.zeros(mol.GetNumAtoms(), dtype=float)
    charges = []
    for atom in mol.GetAtoms():
        try:
            value = float(atom.GetProp("_GasteigerCharge"))
        except Exception:
            value = 0.0
        charges.append(value if math.isfinite(value) else 0.0)
    return np.asarray(charges, dtype=float)


def _conformer_descriptors(
    mol: Any,
    conf_id: int,
    *,
    exposed_sasa_threshold: float,
    charges: np.ndarray,
) -> dict[str, Any]:
    radii = rdFreeSASA.classifyAtoms(mol)
    total_sasa = float(rdFreeSASA.CalcSASA(mol, radii, confIdx=conf_id))
    atomic_sasa = np.asarray(
        [float(atom.GetProp("SASA")) if atom.HasProp("SASA") else 0.0 for atom in mol.GetAtoms()]
    )
    polar_indices: set[int] = set()
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() in {7, 8, 15, 16} or atom.GetFormalCharge() != 0:
            polar_indices.add(int(atom.GetIdx()))
            polar_indices.update(
                int(neighbor.GetIdx()) for neighbor in atom.GetNeighbors() if neighbor.GetAtomicNum() == 1
            )
    polar_sasa = float(atomic_sasa[list(sorted(polar_indices))].sum()) if polar_indices else 0.0
    nonpolar_sasa = max(0.0, total_sasa - polar_sasa)

    donors, acceptors = _donor_acceptor_atoms(mol)
    donor_sasa = float(sum(atomic_sasa[index] for index in donors))
    acceptor_sasa = float(sum(atomic_sasa[index] for index in acceptors))
    exposed_donors = sum(atomic_sasa[index] >= exposed_sasa_threshold for index in donors)
    exposed_acceptors = sum(atomic_sasa[index] >= exposed_sasa_threshold for index in acceptors)

    conformer = mol.GetConformer(conf_id)
    coordinates = np.asarray(conformer.GetPositions(), dtype=float)
    masses = np.asarray([atom.GetMass() for atom in mol.GetAtoms()], dtype=float)
    center_of_mass = np.average(coordinates, axis=0, weights=masses)
    dipole_e_angstrom = np.sum(charges[:, None] * (coordinates - center_of_mass), axis=0)
    positive = np.clip(charges, 0.0, None)
    negative = np.clip(-charges, 0.0, None)
    if positive.sum() > 1e-12 and negative.sum() > 1e-12:
        positive_center = np.average(coordinates, axis=0, weights=positive)
        negative_center = np.average(coordinates, axis=0, weights=negative)
        charge_centroid_separation = float(np.linalg.norm(positive_center - negative_center))
    else:
        charge_centroid_separation = 0.0

    topology = Chem.GetDistanceMatrix(mol)
    imhb_edges: list[str] = []
    for donor in sorted(donors):
        for acceptor in sorted(acceptors):
            if donor == acceptor or topology[donor, acceptor] < 4:
                continue
            distance = float(np.linalg.norm(coordinates[donor] - coordinates[acceptor]))
            if distance <= 3.5:
                imhb_edges.append(f"{donor}-{acceptor}@{distance:.2f}")

    pmi1 = float(rdMolDescriptors.CalcPMI1(mol, confId=conf_id))
    pmi2 = float(rdMolDescriptors.CalcPMI2(mol, confId=conf_id))
    pmi3 = float(rdMolDescriptors.CalcPMI3(mol, confId=conf_id))
    return {
        "sa_3d_psa_ang2": polar_sasa,
        "polar_sasa_ang2": polar_sasa,
        "nonpolar_sasa_ang2": nonpolar_sasa,
        "total_sasa_ang2": total_sasa,
        "exposed_hbd_count_proxy": int(exposed_donors),
        "exposed_hba_count_proxy": int(exposed_acceptors),
        "exposed_hbd_sasa_ang2": donor_sasa,
        "exposed_hba_sasa_ang2": acceptor_sasa,
        "radius_of_gyration_angstrom": float(rdMolDescriptors.CalcRadiusOfGyration(mol, confId=conf_id)),
        "pmi1": pmi1,
        "pmi2": pmi2,
        "pmi3": pmi3,
        "npr1": 0.0 if pmi3 <= 0 else pmi1 / pmi3,
        "npr2": 0.0 if pmi3 <= 0 else pmi2 / pmi3,
        "gasteiger_dipole_proxy_debye": float(np.linalg.norm(dipole_e_angstrom) * 4.8032047),
        "charge_centroid_separation_angstrom": charge_centroid_separation,
        "imhb_count_proxy": len(imhb_edges),
        "imhb_network_proxy": ";".join(imhb_edges),
    }


def _generate_state_conformers(
    state: _Microstate,
    *,
    structure_id: str,
    config: FastPhysicsConfig,
) -> tuple[Any, list[dict[str, Any]], dict[str, Any]]:
    mol = Chem.AddHs(Chem.Mol(state.mol), addCoords=False)
    params = AllChem.ETKDGv3()
    params.randomSeed = _seed_for(structure_id, state.state_id, base_seed=config.random_seed)
    params.enforceChirality = True
    params.useRandomCoords = False
    params.numThreads = 1
    params.pruneRmsThresh = -1.0
    embedded = list(
        AllChem.EmbedMultipleConfs(
            mol,
            numConfs=int(config.generated_conformer_limit),
            params=params,
        )
    )
    if not embedded:
        return mol, [], {"embedding_status": "failed", "n_embedded": 0, "n_retained": 0}

    energies, minimization_status, method = _minimize_conformers(mol)
    selected, cluster_by_conf = _cluster_and_select(
        mol,
        energies,
        cutoff=config.cluster_rmsd_angstrom,
        limit=config.retained_conformer_limit,
    )
    selected = sorted(
        selected,
        key=lambda conf_id: (
            math.inf if not math.isfinite(energies[conf_id]) else energies[conf_id],
            conf_id,
        ),
    )
    weights = _boltzmann_weights([energies[conf_id] for conf_id in selected], config.temperature_kelvin)

    retained_mol = Chem.Mol(mol)
    retained_mol.RemoveAllConformers()
    old_to_new: dict[int, int] = {}
    for conf_id in selected:
        new_id = int(retained_mol.AddConformer(Chem.Conformer(mol.GetConformer(conf_id)), assignId=True))
        old_to_new[conf_id] = new_id
    charges = _gasteiger_charges(retained_mol)
    rows: list[dict[str, Any]] = []
    minimum_energy = min(
        (energies[conf_id] for conf_id in selected if math.isfinite(energies[conf_id])), default=float("nan")
    )
    for rank, (old_conf_id, weight) in enumerate(zip(selected, weights, strict=True), start=1):
        new_conf_id = old_to_new[old_conf_id]
        conformer_id = f"CNF-{_digest(state.state_id, rank, length=20)}"
        energy = energies[old_conf_id]
        descriptors = _conformer_descriptors(
            retained_mol,
            new_conf_id,
            exposed_sasa_threshold=config.exposed_atom_sasa_ang2,
            charges=charges,
        )
        rows.append(
            {
                "structure_id": structure_id,
                "state_id": state.state_id,
                "conformer_id": conformer_id,
                "conformer_rank": rank,
                "rdkit_conformer_id": new_conf_id,
                "cluster_id": int(cluster_by_conf[old_conf_id]),
                "minimization_method": method,
                "minimization_status": int(minimization_status[old_conf_id]),
                "energy_kcal_mol": float(energy),
                "relative_energy_kcal_mol": (
                    float(energy - minimum_energy)
                    if math.isfinite(energy) and math.isfinite(minimum_energy)
                    else float("nan")
                ),
                "conformer_weight": float(weight),
                **descriptors,
            }
        )
    effective_count = float(1.0 / np.square(weights).sum())
    quality = {
        "embedding_status": "embedded",
        "n_embedded": len(embedded),
        "n_retained": len(selected),
        "cluster_count": len(set(cluster_by_conf.values())),
        "minimization_method": method,
        "minimization_failure_count": sum(status != 0 for status in minimization_status.values()),
        "effective_conformer_count": effective_count,
    }
    return retained_mol, rows, quality


_DISTRIBUTION_FEATURES = (
    "sa_3d_psa_ang2",
    "polar_sasa_ang2",
    "nonpolar_sasa_ang2",
    "total_sasa_ang2",
    "exposed_hbd_count_proxy",
    "exposed_hba_count_proxy",
    "exposed_hbd_sasa_ang2",
    "exposed_hba_sasa_ang2",
    "radius_of_gyration_angstrom",
    "npr1",
    "npr2",
    "gasteiger_dipole_proxy_debye",
    "charge_centroid_separation_angstrom",
    "imhb_count_proxy",
    "formal_charge",
    "absolute_formal_charge",
)

_ENVIRONMENT_SENSITIVITY_STRENGTH = 0.5
_RARE_STATE_POPULATION_MAX = 0.05
_COMPOSITE_VALUE_PREFIX = "composite__"
_COMPOSITE_PKA_SPAN_PREFIX = "composite_pka_sensitivity_span__"


def _standardized_sensitivity_coordinate(values: np.ndarray) -> np.ndarray:
    """Return a deterministic, unitless coordinate for sensitivity analysis.

    This is deliberately not a solvent free-energy transformation.  Scaling
    over the retained conformers only makes heterogeneous descriptors
    commensurate for a within-molecule perturbation.  A constant descriptor
    contributes zero rather than an invented environmental preference.
    """

    values = np.asarray(values, dtype=float)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Environmental sensitivity coordinates require finite descriptors")
    center = float(np.median(values))
    scale = float(np.sqrt(np.mean(np.square(values - center))))
    if scale <= 1e-12:
        return np.zeros_like(values)
    return np.clip((values - center) / scale, -4.0, 4.0)


def _normalized_perturbation_weights(base_weights: np.ndarray, log_factor: np.ndarray) -> np.ndarray:
    """Reweight a retained ensemble without interpreting the factor as energy."""

    base_weights = np.asarray(base_weights, dtype=float)
    log_factor = np.asarray(log_factor, dtype=float)
    if base_weights.shape != log_factor.shape or base_weights.size == 0:
        raise ValueError("Perturbation weights require aligned, non-empty arrays")
    if not np.isfinite(base_weights).all() or not np.isfinite(log_factor).all():
        raise ValueError("Perturbation weights require finite inputs")
    if np.any(base_weights < 0.0) or float(base_weights.sum()) <= 0.0:
        raise ValueError("Base ensemble weights must be non-negative with positive mass")
    shifted = np.clip(log_factor - float(np.max(log_factor)), -50.0, 0.0)
    perturbed = base_weights * np.exp(shifted)
    total = float(perturbed.sum())
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("Environmental sensitivity perturbation produced zero mass")
    return perturbed / total


def _environment_sensitivity_table(states: pd.DataFrame, conformers: pd.DataFrame) -> pd.DataFrame:
    """Attach a declared descriptor-only polar/low-dielectric sensitivity axis.

    Positive axis values denote conformers that expose more polarity/charge
    and fewer intramolecular hydrogen-bond proxies.  Symmetric reweighting
    along this axis represents *water-favouring* and *low-dielectric-favouring*
    hypotheses.  It is not a calculation of solvation thermodynamics, solvent
    populations, permeability, or transfer free energy.
    """

    required_states = {"state_id", "formal_charge", "hbd_count", "hba_count"}
    required_conformers = {
        "state_id",
        "conformer_id",
        "polar_sasa_ang2",
        "total_sasa_ang2",
        "exposed_hbd_sasa_ang2",
        "exposed_hba_sasa_ang2",
        "radius_of_gyration_angstrom",
        "imhb_count_proxy",
    }
    if missing := sorted(required_states - set(states.columns)):
        raise ValueError(f"State table lacks environmental sensitivity fields: {missing}")
    if missing := sorted(required_conformers - set(conformers.columns)):
        raise ValueError(f"Conformer table lacks environmental sensitivity fields: {missing}")
    state_fields = states[list(sorted(required_states))].drop_duplicates("state_id")
    table = conformers.merge(state_fields, on="state_id", how="left", validate="many_to_one")
    if table[["formal_charge", "hbd_count", "hba_count"]].isna().any().any():
        raise ValueError("Every conformer must resolve to a state formal charge and H-bond counts")

    total_sasa = np.maximum(table["total_sasa_ang2"].to_numpy(dtype=float), 1e-9)
    table["polar_exposure_fraction"] = table["polar_sasa_ang2"].to_numpy(dtype=float) / total_sasa
    table["exposed_hbond_fraction"] = (
        table["exposed_hbd_sasa_ang2"].to_numpy(dtype=float)
        + table["exposed_hba_sasa_ang2"].to_numpy(dtype=float)
    ) / total_sasa
    table["absolute_formal_charge"] = np.abs(table["formal_charge"].to_numpy(dtype=float))
    hbond_capacity = 1.0 + table["hbd_count"].to_numpy(dtype=float) + table["hba_count"].to_numpy(dtype=float)
    table["imhb_capacity_fraction"] = table["imhb_count_proxy"].to_numpy(dtype=float) / hbond_capacity

    # Coefficients declare a perturbation direction, not calibrated solvent
    # energetics.  Keeping them fixed makes the feature deterministic and
    # prevents outcome labels from entering the transformation.
    table["environment_sensitivity_axis"] = (
        _standardized_sensitivity_coordinate(table["polar_exposure_fraction"].to_numpy(dtype=float))
        + 0.5 * _standardized_sensitivity_coordinate(table["exposed_hbond_fraction"].to_numpy(dtype=float))
        + 1.0 * _standardized_sensitivity_coordinate(table["absolute_formal_charge"].to_numpy(dtype=float))
        - 0.75 * _standardized_sensitivity_coordinate(table["imhb_capacity_fraction"].to_numpy(dtype=float))
        + 0.25
        * _standardized_sensitivity_coordinate(table["radius_of_gyration_angstrom"].to_numpy(dtype=float))
    )
    if not np.isfinite(table["environment_sensitivity_axis"].to_numpy(dtype=float)).all():
        raise ValueError("Environmental sensitivity axis contains non-finite values")
    return table


def _add_composite_uncertainty(composites: pd.DataFrame) -> pd.DataFrame:
    """Attach pKa-scenario sensitivity without claiming statistical error."""

    result = composites.copy()
    group = result.groupby(["structure_id", "ph", "composite_name"], sort=False)["value"]
    result["pka_sensitivity_min"] = group.transform("min")
    result["pka_sensitivity_max"] = group.transform("max")
    result["pka_sensitivity_span"] = result["pka_sensitivity_max"] - result["pka_sensitivity_min"]
    result["uncertainty_semantics"] = (
        "The fixed-pH min/max/span quantify only the declared pKa -1/nominal/+1 "
        "sensitivity. They are not confidence intervals and exclude conformer-sampling, "
        "force-field, explicit-solvent, membrane, and assay uncertainty."
    )
    result["environment_model"] = "descriptor_reweighting_sensitivity_surrogate_no_explicit_solvent"
    result["surrogate_assumptions"] = (
        "axis = z(polar-SASA/total-SASA) + 0.5*z(exposed-H-bond-SASA/total-SASA) "
        "+ z(abs(formal-charge)) - 0.75*z(IMHB-count/(1+HBD+HBA)) + 0.25*z(Rg); "
        "water-favouring and low-dielectric-favouring weights are normalized "
        "w_base*exp(+/-0.5*axis). z is a retained-conformer within-molecule sensitivity scale."
    )
    result["permissible_model_role"] = "mechanistic_discovery_only_until_externally_validated"
    return result


def _pivot_composites_into_summary(summary: pd.DataFrame, composites: pd.DataFrame) -> pd.DataFrame:
    """Expose one finite composite value per condition to downstream model layers."""

    keys = ["structure_id", "ph", "pka_scenario"]
    duplicate = composites.duplicated([*keys, "composite_name"], keep=False)
    if duplicate.any():
        examples = composites.loc[duplicate, [*keys, "composite_name"]].head().to_dict("records")
        raise ValueError(f"Composite names must be unique within an ensemble condition: {examples}")
    values = composites.pivot(index=keys, columns="composite_name", values="value")
    values.columns = [f"{_COMPOSITE_VALUE_PREFIX}{name}" for name in values.columns]
    spans = composites.pivot(index=keys, columns="composite_name", values="pka_sensitivity_span")
    spans.columns = [f"{_COMPOSITE_PKA_SPAN_PREFIX}{name}" for name in spans.columns]
    wide = pd.concat([values, spans], axis=1).reset_index()
    return summary.merge(wide, on=keys, how="left", validate="one_to_one")


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights)
    threshold = float(quantile) * float(sorted_weights.sum())
    return float(
        sorted_values[min(int(np.searchsorted(cumulative, threshold, side="left")), len(values) - 1)]
    )


def _ensemble_tables(
    *,
    structure_id: str,
    states: pd.DataFrame,
    populations: pd.DataFrame,
    conformers: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sensitivity = _environment_sensitivity_table(states, conformers)
    joined = populations.merge(sensitivity, on="state_id", how="inner", validate="many_to_many")
    joined["joint_weight"] = joined["state_weight"] * joined["conformer_weight"]
    summary_rows: list[dict[str, Any]] = []
    composite_rows: list[dict[str, Any]] = []
    for (ph, scenario), group in joined.groupby(["ph", "pka_scenario"], sort=True):
        weights = group["joint_weight"].to_numpy(dtype=float).copy()
        weights /= weights.sum()
        row: dict[str, Any] = {
            "structure_id": structure_id,
            "ph": float(ph),
            "pka_scenario": str(scenario),
            "joint_weight_sum": float(weights.sum()),
            "effective_joint_state_conformer_count": float(1.0 / np.square(weights).sum()),
        }
        for feature in _DISTRIBUTION_FEATURES:
            values = group[feature].to_numpy(dtype=float)
            mean = float(np.average(values, weights=weights))
            row[f"{feature}__mean"] = mean
            row[f"{feature}__sd"] = float(np.sqrt(np.average(np.square(values - mean), weights=weights)))
            for quantile in (0.05, 0.50, 0.95):
                row[f"{feature}__q{int(quantile * 100):02d}"] = _weighted_quantile(values, weights, quantile)
        entropy = float(-np.sum(weights * np.log(np.clip(weights, 1e-300, None))))
        row["joint_conformational_entropy_nats"] = entropy
        row["joint_conformational_entropy_normalized"] = (
            0.0 if len(weights) <= 1 else entropy / math.log(len(weights))
        )
        summary_rows.append(row)

        rg = group["radius_of_gyration_angstrom"].to_numpy(dtype=float)
        polar = group["sa_3d_psa_ang2"].to_numpy(dtype=float)
        rg_median = _weighted_quantile(rg, weights, 0.5)
        polar_median = _weighted_quantile(polar, weights, 0.5)
        reproduced_fraction = float(weights[(rg <= rg_median) & (polar <= polar_median)].sum())
        exposure_burden = (
            group["exposed_hbd_sasa_ang2"].to_numpy(dtype=float)
            + group["exposed_hba_sasa_ang2"].to_numpy(dtype=float)
        ) / np.maximum(group["total_sasa_ang2"].to_numpy(dtype=float), 1e-9)
        shielding = group["imhb_count_proxy"].to_numpy(dtype=float) / np.maximum(polar, 1.0)
        charge_separation = group["charge_centroid_separation_angstrom"].to_numpy(dtype=float) / np.maximum(
            rg, 1e-9
        )
        environment_axis = group["environment_sensitivity_axis"].to_numpy(dtype=float)
        water_favouring_weights = _normalized_perturbation_weights(
            weights,
            _ENVIRONMENT_SENSITIVITY_STRENGTH * environment_axis,
        )
        low_dielectric_favouring_weights = _normalized_perturbation_weights(
            weights,
            -_ENVIRONMENT_SENSITIVITY_STRENGTH * environment_axis,
        )
        polar_fraction = group["polar_exposure_fraction"].to_numpy(dtype=float)
        water_polarity = float(np.average(polar_fraction, weights=water_favouring_weights))
        low_dielectric_polarity = float(np.average(polar_fraction, weights=low_dielectric_favouring_weights))
        water_rg = float(np.average(rg, weights=water_favouring_weights))
        low_dielectric_rg = float(np.average(rg, weights=low_dielectric_favouring_weights))
        base_rg = max(float(np.average(rg, weights=weights)), 1e-9)
        folded_mask = (rg <= rg_median) & (polar <= polar_median)
        water_folded_fraction = float(water_favouring_weights[folded_mask].sum())
        low_dielectric_folded_fraction = float(low_dielectric_favouring_weights[folded_mask].sum())
        hydration_shedding_with_compensation = (
            (
                group["polar_exposure_fraction"].to_numpy(dtype=float)
                + group["exposed_hbond_fraction"].to_numpy(dtype=float)
            )
            * (1.0 + group["absolute_formal_charge"].to_numpy(dtype=float))
            / (1.0 + group["imhb_capacity_fraction"].to_numpy(dtype=float))
        )

        # P_s is a relative descriptor propensity, not permeability.  State
        # contribution follows f_s * P_s / sum(f_s * P_s), preserving each
        # pH/pKa population scenario rather than collapsing microstates.
        low_dielectric_factor = np.exp(
            np.clip(-_ENVIRONMENT_SENSITIVITY_STRENGTH * environment_axis, -50.0, 50.0)
        )
        state_transport_rows: list[tuple[float, float]] = []
        for _state_id, state_group in group.assign(low_dielectric_factor=low_dielectric_factor).groupby(
            "state_id", sort=True
        ):
            conformer_weights = state_group["conformer_weight"].to_numpy(dtype=float).copy()
            conformer_weights /= conformer_weights.sum()
            propensity = float(
                np.average(
                    state_group["low_dielectric_factor"].to_numpy(dtype=float),
                    weights=conformer_weights,
                )
            )
            state_transport_rows.append((float(state_group["state_weight"].iloc[0]), propensity))
        state_transport = np.asarray(state_transport_rows, dtype=float)
        transport_mass = state_transport[:, 0] * state_transport[:, 1]
        transport_contribution = transport_mass / transport_mass.sum()
        rare_state_transport_dominance = float(
            transport_contribution[state_transport[:, 0] <= _RARE_STATE_POPULATION_MAX].sum()
        )
        definitions = (
            (
                "folded_low_polarity_fraction",
                "reproduced",
                reproduced_fraction,
                "Weight below the ensemble medians of both radius of gyration and solvent-accessible 3D polar surface; reproduces the literature descriptor combination, not its exact thresholds.",
            ),
            (
                "exposure_adjusted_hbond_burden",
                "extended",
                float(np.average(exposure_burden, weights=weights)),
                "Weighted exposed donor-plus-acceptor SASA divided by total SASA.",
            ),
            (
                "intramolecular_shielding_candidate",
                "candidate",
                float(np.average(shielding, weights=weights)),
                "Weighted IMHB distance-network count divided by solvent-accessible 3D polar surface.",
            ),
            (
                "charge_separation_per_gyration_candidate",
                "candidate",
                float(np.average(charge_separation, weights=weights)),
                "Weighted Gasteiger positive-negative charge-centroid separation divided by radius of gyration.",
            ),
            (
                "environment_conditioned_polarity_response_surrogate",
                "candidate",
                low_dielectric_polarity - water_polarity,
                "Difference in mean polar-SASA fraction after symmetric descriptor-only low-dielectric-favouring versus water-favouring sensitivity reweighting (strength 0.5); not solvent populations or transfer thermodynamics.",
            ),
            (
                "environment_conditioned_shape_response_surrogate",
                "candidate",
                (low_dielectric_rg - water_rg) / base_rg,
                "Difference in reweighted mean radius of gyration (low-dielectric-favouring minus water-favouring), divided by the base-ensemble mean; descriptor sensitivity only, not explicit-solvent folding.",
            ),
            (
                "water_to_low_dielectric_folded_fraction_shift_surrogate",
                "candidate",
                low_dielectric_folded_fraction - water_folded_fraction,
                "Change in compact/low-polarity mask weight under low-dielectric-favouring versus water-favouring descriptor reweighting; thresholds are base-ensemble medians and no solvent free energy is implied.",
            ),
            (
                "hydration_shedding_imhb_compensation_surrogate",
                "candidate",
                float(np.average(hydration_shedding_with_compensation, weights=weights)),
                "Base-ensemble mean of (polar-SASA fraction + exposed-H-bond-SASA fraction) times (1 + absolute formal charge), divided by (1 + IMHB-count/H-bond-capacity); a screening burden, not hydration free energy.",
            ),
            (
                "rare_state_transport_dominance_surrogate",
                "candidate",
                rare_state_transport_dominance,
                "Sum of f_s*P_s/sum(f_s*P_s) for retained states with f_s <= 0.05, where P_s is the conformer-weighted low-dielectric descriptor propensity; P_s is not measured or simulated permeability.",
            ),
        )
        for name, evidence_class, value, definition in definitions:
            composite_rows.append(
                {
                    "structure_id": structure_id,
                    "ph": float(ph),
                    "pka_scenario": str(scenario),
                    "composite_name": name,
                    "evidence_class": evidence_class,
                    "value": value,
                    "definition": definition,
                }
            )
    summary = pd.DataFrame(summary_rows)
    composites = _add_composite_uncertainty(pd.DataFrame(composite_rows))
    finite_columns = [
        column
        for column in composites.columns
        if column in {"value", "pka_sensitivity_min", "pka_sensitivity_max", "pka_sensitivity_span"}
    ]
    if not np.isfinite(composites[finite_columns].to_numpy(dtype=float)).all():
        raise ValueError("Composite sensitivity features must be finite")
    return _pivot_composites_into_summary(summary, composites), composites


def _quality_table(
    states: pd.DataFrame,
    populations: pd.DataFrame,
    conformers: pd.DataFrame,
    state_quality: list[dict[str, Any]],
    state_metadata: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rows.append(
        {
            "scope": "state_enumeration",
            "entity_id": "all_states",
            "gate": "discarded_probability_mass_le_0.05",
            "value": float(state_metadata["maximum_discarded_probability_mass"]),
            "threshold": 0.05,
            "passed": bool(state_metadata["maximum_discarded_probability_mass"] <= 0.05),
            "severity": "warning",
            "note": "Population is an evidence-weighted approximation with explicit ±1 pKa sensitivity.",
        }
    )
    rows.append(
        {
            "scope": "state_enumeration",
            "entity_id": "all_states",
            "gate": "no_qualifying_state_omitted_by_compute_cap",
            "value": float(state_metadata.get("qualifying_states_omitted_by_cap", 0)),
            "threshold": 0.0,
            "passed": bool(state_metadata.get("qualifying_states_omitted_by_cap", 0) == 0),
            "severity": "error",
            "note": (
                "max_states is a compute cap, not a physical state-count target. Any omitted "
                "0.01%-sensitivity or structural-exception state makes the features inadmissible."
            ),
        }
    )
    for item in state_quality:
        rows.extend(
            [
                {
                    "scope": "conformer_generation",
                    "entity_id": item["state_id"],
                    "gate": "at_least_one_retained_conformer",
                    "value": float(item["n_retained"]),
                    "threshold": 1.0,
                    "passed": bool(item["n_retained"] >= 1),
                    "severity": "error",
                    "note": item["embedding_status"],
                },
                {
                    "scope": "conformer_generation",
                    "entity_id": item["state_id"],
                    "gate": "effective_conformer_count_ge_1_5_or_single",
                    "value": float(item.get("effective_conformer_count", 0.0)),
                    "threshold": 1.5,
                    "passed": bool(
                        item["n_retained"] == 1 or item.get("effective_conformer_count", 0.0) >= 1.5
                    ),
                    "severity": "warning",
                    "note": "Low ESS indicates one minimized conformer dominates Boltzmann weights.",
                },
            ]
        )
    for state_id, group in conformers.groupby("state_id", sort=True):
        if len(group) < 4:
            delta = float("nan")
            passed = True
        else:
            ordered = group.sort_values("conformer_rank")
            first = ordered.iloc[::2]
            second = ordered.iloc[1::2]
            deltas = []
            for feature in ("sa_3d_psa_ang2", "radius_of_gyration_angstrom", "imhb_count_proxy"):
                first_mean = float(first[feature].mean())
                second_mean = float(second[feature].mean())
                deltas.append(abs(first_mean - second_mean) / max(abs(first_mean), abs(second_mean), 1.0))
            delta = max(deltas)
            passed = delta <= 0.20
        rows.append(
            {
                "scope": "conformer_convergence",
                "entity_id": state_id,
                "gate": "deterministic_split_relative_delta_le_0.20",
                "value": delta,
                "threshold": 0.20,
                "passed": bool(passed),
                "severity": "warning",
                "note": "A screening diagnostic, not proof of equilibrium convergence.",
            }
        )
    weights = populations.groupby(["ph", "pka_scenario"])["state_weight"].sum()
    rows.append(
        {
            "scope": "normalization",
            "entity_id": "all_conditions",
            "gate": "state_weights_sum_to_one",
            "value": float(np.max(np.abs(weights.to_numpy(dtype=float) - 1.0))),
            "threshold": 1e-10,
            "passed": bool(np.allclose(weights.to_numpy(dtype=float), 1.0, atol=1e-10)),
            "severity": "error",
            "note": "Weights are renormalized over retained states.",
        }
    )
    conformer_weights = conformers.groupby("state_id")["conformer_weight"].sum()
    rows.append(
        {
            "scope": "normalization",
            "entity_id": "all_states",
            "gate": "conformer_weights_sum_to_one",
            "value": float(np.max(np.abs(conformer_weights.to_numpy(dtype=float) - 1.0))),
            "threshold": 1e-10,
            "passed": bool(np.allclose(conformer_weights.to_numpy(dtype=float), 1.0, atol=1e-10)),
            "severity": "error",
            "note": "Conformer weights are conditional within each retained chemical state.",
        }
    )
    return pd.DataFrame(rows)


def _relative_delta(left: float, right: float) -> float:
    return abs(left - right) / max(abs(left), abs(right), 1.0)


def _conformer_sampling_escalation_queue(
    states: pd.DataFrame,
    conformers: pd.DataFrame,
    config: FastPhysicsConfig,
) -> pd.DataFrame:
    """Audit deterministic nested retained subsets without claiming convergence.

    The subsets are prefixes of one generated ETKDG pool after deterministic
    energy/cluster ordering. They test internal stabilization only; they are
    neither independent replicas nor proof that the local conformer pool is exhaustive.
    """

    rows: list[dict[str, Any]] = []
    state_lookup = states.set_index("state_id", drop=False)
    criteria = (
        "last-two nested retained subsets: relative delta <=0.10 for weighted 3D polar SASA "
        "and radius of gyration; absolute IMHB-count delta <=0.50; minimum-energy delta "
        "<=1.0 kcal/mol; largest-subset effective count >=1.5 unless only one conformer"
    )
    for state_id, raw_group in conformers.groupby("state_id", sort=True):
        group = raw_group.sort_values("conformer_rank").reset_index(drop=True)
        retained_count = len(group)
        candidate_depths = [depth for depth in (4, 8, 16, 25, 50, 100, 250, 500) if depth <= retained_count]
        if retained_count not in candidate_depths:
            candidate_depths.append(retained_count)
        candidate_depths = sorted(set(candidate_depths))
        audits: list[dict[str, float]] = []
        for depth in candidate_depths:
            subset = group.iloc[:depth]
            weights = subset["conformer_weight"].to_numpy(dtype=float).copy()
            weights /= weights.sum()
            audits.append(
                {
                    "depth": float(depth),
                    "polar_sasa": float(
                        np.average(subset["sa_3d_psa_ang2"].to_numpy(dtype=float), weights=weights)
                    ),
                    "rg": float(
                        np.average(
                            subset["radius_of_gyration_angstrom"].to_numpy(dtype=float),
                            weights=weights,
                        )
                    ),
                    "imhb": float(
                        np.average(subset["imhb_count_proxy"].to_numpy(dtype=float), weights=weights)
                    ),
                    "minimum_energy": float(np.nanmin(subset["energy_kcal_mol"].to_numpy(dtype=float))),
                    "effective_count": float(1.0 / np.square(weights).sum()),
                }
            )
        latest = audits[-1]
        previous = audits[-2] if len(audits) > 1 else None
        polar_delta = (
            math.inf if previous is None else _relative_delta(latest["polar_sasa"], previous["polar_sasa"])
        )
        rg_delta = math.inf if previous is None else _relative_delta(latest["rg"], previous["rg"])
        imhb_delta = math.inf if previous is None else abs(latest["imhb"] - previous["imhb"])
        energy_delta = (
            math.inf if previous is None else abs(latest["minimum_energy"] - previous["minimum_energy"])
        )
        nested_criteria_passed = bool(
            polar_delta <= 0.10
            and rg_delta <= 0.10
            and imhb_delta <= 0.50
            and energy_delta <= 1.0
            and (retained_count == 1 or latest["effective_count"] >= 1.5)
        )
        generated_depth = int(state_lookup.loc[state_id, "n_embedded"])
        audit_covers_generated_depth = retained_count >= generated_depth
        sampling_audit_abstained = not audit_covers_generated_depth
        internally_stable = bool(nested_criteria_passed and audit_covers_generated_depth)
        if generated_depth < LOCAL_GENERATED_CONFORMER_DEPTH:
            queue_status = "incomplete_local_screen_retry_25"
            next_depth = LOCAL_GENERATED_CONFORMER_DEPTH
            escalation_required = True
        elif sampling_audit_abstained:
            queue_status = "abstain_retained_depth_does_not_cover_generated_pool"
            next_depth = generated_depth
            escalation_required = True
        elif generated_depth < PILOT_ESCALATION_CONFORMER_DEPTHS[0]:
            queue_status = (
                "escalate_unstable_25_to_50"
                if not internally_stable
                else "25_nested_screen_stable_defer_50_until_claim_critical"
            )
            next_depth = PILOT_ESCALATION_CONFORMER_DEPTHS[0]
            escalation_required = not internally_stable
        elif generated_depth < PILOT_ESCALATION_CONFORMER_DEPTHS[1]:
            queue_status = (
                "escalate_unstable_50_to_100"
                if not internally_stable
                else "50_nested_screen_stable_defer_100_until_claim_critical"
            )
            next_depth = PILOT_ESCALATION_CONFORMER_DEPTHS[1]
            escalation_required = not internally_stable
        elif generated_depth < DEFERRED_VALIDATION_CONFORMER_CEILING:
            queue_status = (
                "escalate_unstable_100_to_250"
                if not internally_stable
                else "100_nested_screen_stable_defer_250_until_claim_critical"
            )
            next_depth = DEFERRED_VALIDATION_CONFORMER_CEILING
            escalation_required = not internally_stable
        elif generated_depth < PILOT_COMPARATOR_CONFORMER_DEPTH:
            queue_status = (
                "escalate_unstable_250_to_500_pilot_comparator"
                if not internally_stable
                else "250_nested_screen_stable_not_exhaustive"
            )
            next_depth = PILOT_COMPARATOR_CONFORMER_DEPTH if not internally_stable else 0
            escalation_required = not internally_stable
        else:
            queue_status = (
                "method_or_enhanced_sampling_review"
                if not internally_stable
                else "500_pilot_stable_not_validated"
            )
            next_depth = 0
            escalation_required = not internally_stable
        rows.append(
            {
                "structure_id": str(group["structure_id"].iloc[0]),
                "state_id": str(state_id),
                "generated_conformer_depth": generated_depth,
                "retained_conformer_count": retained_count,
                "nested_retained_depths": ";".join(str(int(item["depth"])) for item in audits),
                "nested_polar_sasa_relative_delta": polar_delta,
                "nested_rg_relative_delta": rg_delta,
                "nested_imhb_mean_absolute_delta": imhb_delta,
                "nested_minimum_energy_absolute_delta_kcal_mol": energy_delta,
                "largest_subset_effective_conformer_count": latest["effective_count"],
                "nested_criteria_passed": nested_criteria_passed,
                "audit_covers_generated_depth": audit_covers_generated_depth,
                "sampling_audit_abstained": sampling_audit_abstained,
                "sampling_audit_abstention_reason": (
                    "retained conformer depth does not cover the generated pool"
                    if sampling_audit_abstained
                    else "none"
                ),
                "nested_internal_stability_passed": internally_stable,
                "escalation_required": escalation_required,
                "queue_status": queue_status,
                "recommended_next_generated_depth": next_depth,
                "deferred_validation_ceiling": DEFERRED_VALIDATION_CONFORMER_CEILING,
                "pilot_comparator_depth": PILOT_COMPARATOR_CONFORMER_DEPTH,
                "criteria": criteria,
                "interpretation": (
                    "25 is the time-bounded all-series screen, 250 is a deferred validation ceiling, "
                    "and 500 is a pilot comparator. None is a validated universal constant."
                ),
            }
        )
    return pd.DataFrame(rows)


def _physics_admissibility(
    *,
    structure_id: str,
    quality: pd.DataFrame,
    summary: pd.DataFrame,
    composites: pd.DataFrame,
    states: pd.DataFrame,
    populations: pd.DataFrame | None,
    conformers: pd.DataFrame | None,
    smoke_mode: bool,
) -> dict[str, Any]:
    """Classify feature validity separately from finite-sampling limitations."""

    substantive_gates = {
        "discarded_probability_mass_le_0.05",
        "at_least_one_retained_conformer",
        "state_weights_sum_to_one",
        "conformer_weights_sum_to_one",
        "no_qualifying_state_omitted_by_compute_cap",
    }
    sampling_gates = {
        "effective_conformer_count_ge_1_5_or_single",
        "deterministic_split_relative_delta_le_0.20",
    }
    passed = quality["passed"].fillna(False).astype(bool)
    failed_gates = quality.loc[~passed, "gate"].astype(str).tolist()
    substantive = [gate for gate in failed_gates if gate in substantive_gates]
    sampling = [gate for gate in failed_gates if gate in sampling_gates]
    uncategorized = [
        gate for gate in failed_gates if gate not in substantive_gates and gate not in sampling_gates
    ]
    substantive.extend(f"uncategorized_failed_gate:{gate}" for gate in uncategorized)

    if populations is not None and not populations.empty:
        population_sums = populations.groupby(["ph", "pka_scenario"])["state_weight"].sum()
        if not np.allclose(population_sums.to_numpy(dtype=float), 1.0, atol=1e-10):
            substantive.append("state_weights_sum_to_one")
    else:
        substantive.append("missing_state_populations")
    if conformers is not None and not conformers.empty:
        conformer_sums = conformers.groupby("state_id")["conformer_weight"].sum()
        if not np.allclose(conformer_sums.to_numpy(dtype=float), 1.0, atol=1e-10):
            substantive.append("conformer_weights_sum_to_one")
    else:
        substantive.append("missing_conformers")

    numeric_summary = summary.select_dtypes(include=[np.number])
    numeric_composites = composites.select_dtypes(include=[np.number])
    state_charge = (
        pd.to_numeric(states["formal_charge"], errors="coerce")
        if "formal_charge" in states.columns
        else pd.Series(dtype=float)
    )
    if (
        numeric_summary.empty
        or not np.isfinite(numeric_summary.to_numpy(dtype=float)).all()
        or numeric_composites.empty
        or not np.isfinite(numeric_composites.to_numpy(dtype=float)).all()
        or state_charge.empty
        or not np.isfinite(state_charge.to_numpy(dtype=float)).all()
    ):
        substantive.append("nonfinite_or_missing_aggregate_or_formal_charge")

    substantive_failure_count = len(substantive)
    sampling_failure_count = len(sampling)
    substantive_types = sorted(set(substantive))
    sampling_types = sorted(set(sampling))
    feature_admissible = not substantive_types
    smoke_sampling = sampling_types if smoke_mode else []
    production_sampling = [] if smoke_mode else sampling_types
    reason_flags = [f"substantive:{reason}" for reason in substantive_types]
    reason_flags.extend(f"smoke_sampling:{reason}" for reason in smoke_sampling)
    reason_flags.extend(f"sampling:{reason}" for reason in production_sampling)
    if substantive_types:
        status = "ineligible_substantive_failure"
    elif smoke_sampling:
        status = "eligible_discovery_with_smoke_sampling_limit"
    elif production_sampling:
        status = "eligible_discovery_with_sampling_warning"
    elif smoke_mode:
        status = "eligible_discovery_smoke_screen"
    else:
        status = "eligible_discovery_screen"
    return {
        "structure_id": structure_id,
        "physics_model_eligible": bool(feature_admissible),
        "physics_decision_track_eligible": False,
        "physics_convergence_claimed": False,
        "physics_smoke_mode": bool(smoke_mode),
        "physics_quality_status": status,
        "physics_substantive_failure_count": substantive_failure_count,
        "physics_smoke_sampling_failure_count": sampling_failure_count if smoke_mode else 0,
        "physics_production_sampling_warning_count": 0 if smoke_mode else sampling_failure_count,
        "physics_quality_reason_flags": ";".join(reason_flags) if reason_flags else "none",
        "physics_quality_interpretation": (
            "Eligibility means only that retained screening descriptors are finite, normalized, and "
            "do not violate the declared state-mass gate. It is not a convergence claim or "
            "decision-track promotion. Smoke sampling failures are reported separately."
        ),
    }


def _attach_physics_admissibility(
    *,
    structure_id: str,
    quality: pd.DataFrame,
    summary: pd.DataFrame,
    composites: pd.DataFrame,
    states: pd.DataFrame,
    populations: pd.DataFrame | None,
    conformers: pd.DataFrame | None,
    smoke_mode: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    admissibility = pd.DataFrame(
        [
            _physics_admissibility(
                structure_id=structure_id,
                quality=quality,
                summary=summary,
                composites=composites,
                states=states,
                populations=populations,
                conformers=conformers,
                smoke_mode=smoke_mode,
            )
        ]
    )
    annotated = summary.merge(admissibility, on="structure_id", how="left", validate="many_to_one")
    return annotated, admissibility


def run_structure_fast_physics(
    smiles: object,
    *,
    pka_evidence: tuple[PKaEvidence, ...] | list[PKaEvidence] = (),
    config: FastPhysicsConfig | None = None,
) -> FastPhysicsResult:
    """Enumerate approximate states, conformers, and state-aware descriptors.

    The input is standardized once with tautomer canonicalization disabled so
    specified stereochemistry and tautomer hypotheses remain visible.
    """

    _require_rdkit()
    config = config or FastPhysicsConfig()
    evidence = tuple(pka_evidence)
    standardized = standardize_smiles(
        smiles,
        strip_salts=True,
        canonicalize_tautomer=False,
        require_rdkit=True,
    )
    if not standardized.structure_valid or not standardized.standardized_smiles:
        raise ValueError(f"A valid structure is required: {standardized.structure_error}")
    parent = Chem.MolFromSmiles(standardized.standardized_smiles)
    if parent is None:
        raise ValueError("RDKit could not parse the standardized structure")

    raw_states = _enumerate_raw_states(
        parent,
        structure_id=standardized.structure_id,
        evidence=evidence,
        config=config,
    )
    raw_populations = _population_table(raw_states, config)
    states, populations, state_metadata = _retain_states(raw_states, raw_populations, config)

    state_rows: list[dict[str, Any]] = []
    conformer_rows: list[dict[str, Any]] = []
    state_quality: list[dict[str, Any]] = []
    state_molecules: dict[str, Any] = {}
    for state in states:
        retained_mol, rows, quality = _generate_state_conformers(
            state,
            structure_id=standardized.structure_id,
            config=config,
        )
        quality["state_id"] = state.state_id
        state_quality.append(quality)
        if rows:
            state_molecules[state.state_id] = retained_mol
            conformer_rows.extend(rows)
        state_rows.append(
            {
                "structure_id": standardized.structure_id,
                "state_id": state.state_id,
                "state_smiles": state.smiles,
                "tautomer_index": state.tautomer_index,
                "transformation": state.transformation,
                "site_atom_index": state.site_atom_index,
                "formal_charge": state.formal_charge,
                "hbd_count": state.hbd,
                "hba_count": state.hba,
                "pka_kind": state.pka_kind,
                "pka_value_approximate": state.pka_value,
                "pka_label": state.pka_label,
                "pka_source": state.pka_source,
                "pka_basis": state.pka_basis,
                **quality,
            }
        )

    states_table = pd.DataFrame(state_rows).sort_values("state_id").reset_index(drop=True)
    conformers_table = pd.DataFrame(conformer_rows)
    if conformers_table.empty:
        raise RuntimeError("No conformers could be generated for any retained state")
    conformers_table = conformers_table.sort_values(["state_id", "conformer_rank"]).reset_index(drop=True)
    successful_ids = set(conformers_table["state_id"])
    states_table = states_table.loc[states_table["state_id"].isin(successful_ids)].reset_index(drop=True)
    populations = populations.loc[populations["state_id"].isin(successful_ids)].copy()
    populations["state_weight"] = populations["state_weight"] / populations.groupby(["ph", "pka_scenario"])[
        "state_weight"
    ].transform("sum")
    populations = populations.sort_values(["ph", "pka_scenario", "state_id"]).reset_index(drop=True)
    ensemble_summary, composites = _ensemble_tables(
        structure_id=standardized.structure_id,
        states=states_table,
        populations=populations,
        conformers=conformers_table,
    )
    quality = _quality_table(states_table, populations, conformers_table, state_quality, state_metadata)
    ensemble_summary, _admissibility = _attach_physics_admissibility(
        structure_id=standardized.structure_id,
        quality=quality,
        summary=ensemble_summary,
        composites=composites,
        states=states_table,
        populations=populations,
        conformers=conformers_table,
        smoke_mode=config.smoke_mode,
    )
    metadata = {
        "fast_physics_version": FAST_PHYSICS_VERSION,
        "structure_id": standardized.structure_id,
        "standardized_smiles": standardized.standardized_smiles,
        "standardization_version": standardized.structure_standardization_version,
        "rdkit_version": str(rdBase.rdkitVersion),
        "config": asdict(config),
        "pka_evidence": [asdict(item) for item in evidence],
        "pka_interpretation": (
            "Approximate Henderson-Hasselbalch evidence weighting only. Values are not exact "
            "microscopic pKas; every result includes pKa -1, nominal, and pKa +1 sensitivity."
        ),
        "state_retention": state_metadata,
        "composite_evidence_classes": {
            "reproduced": "A literature descriptor combination implemented with declared local thresholds.",
            "extended": "A direct extension of literature descriptors.",
            "candidate": "An unvalidated hypothesis; precedent and novelty have not been established.",
        },
        "environment_sensitivity_interpretation": (
            "Symmetric descriptor reweighting is a falsifiable sensitivity surrogate only. It is not "
            "explicit-solvent sampling, a solvent population estimate, permeability, or transfer free energy."
        ),
        "conformer_sampling_interpretation": (
            f"The configured generated depth ({config.generated_conformer_limit}) is a deterministic "
            f"screening budget, not a validated constant. Local depth {LOCAL_GENERATED_CONFORMER_DEPTH} "
            "is audited with nested retained "
            "subsets; 250 is a deferred validation ceiling and 500 is a selected-pilot comparator. "
            "Neither nested prefixes nor one ETKDG seed establish ensemble convergence."
        ),
        "physics_admissibility": _admissibility.iloc[0].to_dict(),
        "smoke_mode": bool(config.smoke_mode),
    }
    return FastPhysicsResult(
        structure_id=standardized.structure_id,
        standardized_smiles=standardized.standardized_smiles,
        states=states_table,
        populations=populations,
        conformers=conformers_table,
        ensemble_summary=ensemble_summary,
        composites=composites,
        quality=quality,
        metadata=metadata,
        state_molecules=state_molecules,
    )


def _config_from_mapping(
    value: Mapping[str, Any] | FastPhysicsConfig | None,
    *,
    smoke: bool,
) -> FastPhysicsConfig:
    if isinstance(value, FastPhysicsConfig):
        return replace(value, smoke_mode=smoke or value.smoke_mode)
    raw: Mapping[str, Any] = value or {}
    if "fast_physics" in raw and isinstance(raw["fast_physics"], Mapping):
        raw = raw["fast_physics"]
    allowed = {item.name for item in fields(FastPhysicsConfig)}
    kwargs = {key: raw[key] for key in raw if key in allowed}
    for tuple_key in ("ph_grid",):
        if tuple_key in kwargs:
            kwargs[tuple_key] = tuple(float(item) for item in kwargs[tuple_key])
    kwargs["smoke_mode"] = bool(smoke or kwargs.get("smoke_mode", False))
    return FastPhysicsConfig(**kwargs)


def _evidence_from_table(
    chemical_states: pd.DataFrame | None,
    *,
    compound_ids: set[str],
) -> tuple[PKaEvidence, ...]:
    if chemical_states is None or chemical_states.empty or "compound_id" not in chemical_states.columns:
        return ()
    subset = chemical_states.loc[chemical_states["compound_id"].astype(str).isin(compound_ids)]
    evidence: list[PKaEvidence] = []
    for row_index, row in subset.iterrows():
        pka_value = row.get("pka_value", row.get("pka", np.nan))
        kind = str(row.get("pka_kind", row.get("kind", ""))).strip().lower()
        if pd.isna(pka_value) or kind not in {"acid", "base"}:
            continue
        atom_index = row.get("atom_index")
        evidence.append(
            PKaEvidence(
                label=str(row.get("pka_label", row.get("label", f"chemical_states_row_{row_index}"))),
                pka=float(pka_value),
                kind=kind,  # type: ignore[arg-type]
                source=str(row.get("pka_source", row.get("source", "chemical_states table evidence"))),
                atom_index=None if pd.isna(atom_index) else int(atom_index),
                atom_smarts=str(row.get("atom_smarts", "") or ""),
            )
        )
    deduplicated = {
        (item.label, item.pka, item.kind, item.source, item.atom_index, item.atom_smarts): item
        for item in evidence
    }
    return tuple(deduplicated[key] for key in sorted(deduplicated, key=str))


@dataclass(frozen=True)
class _StructureJob:
    structure_id: str
    standardized_smiles: str
    compound_ids: tuple[str, ...]
    evidence: tuple[PKaEvidence, ...]
    config: FastPhysicsConfig
    cache_root: str


def _cache_identity(job: _StructureJob) -> dict[str, Any]:
    """Return the scientific identity of a cached structure calculation."""

    payload = {
        "implementation": FAST_PHYSICS_VERSION,
        "structure_id": job.structure_id,
        "standardized_smiles": job.standardized_smiles,
        "config": asdict(job.config),
        "pka_evidence": [asdict(item) for item in job.evidence],
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return {**payload, "cache_key": f"FPC-{_digest(serialized, length=32)}"}


def _cache_path(job: _StructureJob, identity: Mapping[str, Any]) -> Path:
    return Path(job.cache_root) / job.structure_id / str(identity["cache_key"])


def _normalized_json_value(value: Any) -> Any:
    """Normalize tuples/numpy-free dataclass payloads as JSON represents them."""

    return json.loads(json.dumps(value, sort_keys=True, separators=(",", ":")))


def _sdf_record_count(path: Path) -> int:
    count = 0
    with path.open("rb") as handle:
        for line in handle:
            if line.strip() == b"$$$$":
                count += 1
    return count


def _load_cached_structure(path: Path) -> dict[str, pd.DataFrame]:
    return {
        "states": pd.read_parquet(path / "states.parquet"),
        "populations": pd.read_parquet(path / "state_populations.parquet"),
        "conformers": pd.read_parquet(path / "conformers.parquet"),
        "ensemble_summary": pd.read_parquet(path / "ensemble_summary.parquet"),
        "composites": pd.read_parquet(path / "composites.parquet"),
        "quality": pd.read_parquet(path / "quality_gates.parquet"),
    }


def _cache_is_complete(path: Path, job: _StructureJob, identity: Mapping[str, Any]) -> bool:
    """Validate identity, table relationships, row counts, and SDF completeness."""

    try:
        if not path.is_dir():
            return False
        if any(
            not (path / name).is_file() or (path / name).stat().st_size == 0 for name in _CACHE_REQUIRED_FILES
        ):
            return False
        metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        if metadata.get("cache_identity") != _normalized_json_value(dict(identity)):
            return False
        tables = _load_cached_structure(path)
        required_columns = {
            "states": {"structure_id", "state_id"},
            "populations": {"state_id", "ph", "pka_scenario", "state_weight"},
            "conformers": {"structure_id", "state_id", "conformer_id", "conformer_weight"},
            "ensemble_summary": {"structure_id", "ph", "pka_scenario", "joint_weight_sum"},
            "composites": {"structure_id", "composite_name", "value"},
            "quality": {"scope", "entity_id", "gate", "passed"},
        }
        if any(
            frame.empty or not required_columns[name] <= set(frame.columns) for name, frame in tables.items()
        ):
            return False
        expected_counts = metadata.get("output_row_counts", {})
        if any(int(expected_counts.get(name, -1)) != len(frame) for name, frame in tables.items()):
            return False
        if set(tables["states"]["structure_id"].astype(str)) != {job.structure_id}:
            return False
        if set(tables["conformers"]["structure_id"].astype(str)) != {job.structure_id}:
            return False
        if set(tables["ensemble_summary"]["structure_id"].astype(str)) != {job.structure_id}:
            return False
        state_ids = set(tables["states"]["state_id"].astype(str))
        if not set(tables["populations"]["state_id"].astype(str)) <= state_ids:
            return False
        if not set(tables["conformers"]["state_id"].astype(str)) <= state_ids:
            return False
        state_weight_sums = tables["populations"].groupby(["ph", "pka_scenario"])["state_weight"].sum()
        conformer_weight_sums = tables["conformers"].groupby("state_id")["conformer_weight"].sum()
        if not np.allclose(state_weight_sums.to_numpy(dtype=float), 1.0, atol=1e-10):
            return False
        if not np.allclose(conformer_weight_sums.to_numpy(dtype=float), 1.0, atol=1e-10):
            return False
        if not np.allclose(tables["ensemble_summary"]["joint_weight_sum"], 1.0, atol=1e-10):
            return False
        if _sdf_record_count(path / "conformers.sdf") != len(tables["conformers"]):
            return False
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False
    return True


def _write_cache_atomically(result: FastPhysicsResult, job: _StructureJob, identity: dict[str, Any]) -> Path:
    target = _cache_path(job, identity)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{identity['cache_key']}.partial-", dir=target.parent))
    try:
        result.metadata["cache_identity"] = _normalized_json_value(identity)
        result.metadata["output_row_counts"] = {
            "states": len(result.states),
            "populations": len(result.populations),
            "conformers": len(result.conformers),
            "ensemble_summary": len(result.ensemble_summary),
            "composites": len(result.composites),
            "quality": len(result.quality),
        }
        write_fast_physics_outputs(result, temporary)
        if not _cache_is_complete(temporary, job, identity):
            raise RuntimeError(f"Fast-physics cache validation failed for {job.structure_id}")
        if target.exists():
            if _cache_is_complete(target, job, identity):
                return target
            shutil.rmtree(target)
        os.replace(temporary, target)
        return target
    finally:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)


def _execute_structure_job(job: _StructureJob) -> dict[str, Any]:
    """Load one valid cache entry or calculate and atomically promote it."""

    identity = _cache_identity(job)
    target = _cache_path(job, identity)
    if _cache_is_complete(target, job, identity):
        return {
            "structure_id": job.structure_id,
            "compound_ids": job.compound_ids,
            "cache_path": str(target),
            "cache_hit": True,
            "error": "",
        }
    try:
        result = run_structure_fast_physics(
            job.standardized_smiles,
            pka_evidence=job.evidence,
            config=job.config,
        )
        if result.structure_id != job.structure_id:
            raise ValueError(
                f"Structure identity changed from {job.structure_id} to {result.structure_id} during calculation"
            )
        target = _write_cache_atomically(result, job, identity)
        return {
            "structure_id": job.structure_id,
            "compound_ids": job.compound_ids,
            "cache_path": str(target),
            "cache_hit": False,
            "error": "",
        }
    except Exception as exc:
        return {
            "structure_id": job.structure_id,
            "compound_ids": job.compound_ids,
            "cache_path": "",
            "cache_hit": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _execution_options(
    value: Mapping[str, Any] | FastPhysicsConfig | None,
    *,
    destination: Path,
) -> tuple[int, Path]:
    raw: Mapping[str, Any] = value if isinstance(value, Mapping) else {}
    physics_raw = raw.get("fast_physics", raw)
    if not isinstance(physics_raw, Mapping):
        physics_raw = {}
    requested_workers = int(physics_raw.get("workers", DEFAULT_STRUCTURE_WORKERS))
    if requested_workers < 1:
        raise ValueError("fast_physics.workers must be at least one")
    workers = min(requested_workers, max(1, os.cpu_count() or 1))

    cache_value = physics_raw.get("cache_dir")
    paths = raw.get("paths", {})
    if cache_value is None and isinstance(paths, Mapping):
        cache_value = paths.get("physics_cache")
    cache_root = Path(cache_value) if cache_value is not None else destination.parent / "fast_physics_cache"
    if not cache_root.is_absolute():
        cache_root = destination.parent / cache_root
    return workers, cache_root.resolve()


def _materialize_cached_structure(cache_path: Path, destination: Path, job: _StructureJob) -> None:
    identity = _cache_identity(job)
    if not _cache_is_complete(cache_path, job, identity):
        raise RuntimeError(f"Refusing to materialize incomplete cache for {job.structure_id}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{job.structure_id}.materialize-", dir=destination.parent))
    shutil.rmtree(temporary)
    try:
        shutil.copytree(cache_path, temporary)
        if not _cache_is_complete(temporary, job, identity):
            raise RuntimeError(f"Materialized cache validation failed for {job.structure_id}")
        if destination.exists():
            shutil.rmtree(destination)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)


def run_fast_physics(
    compounds: pd.DataFrame,
    chemical_states: pd.DataFrame | None = None,
    output_dir: str | Path | None = None,
    config: Mapping[str, Any] | FastPhysicsConfig | None = None,
    smoke: bool = False,
) -> dict[str, Path | int]:
    """Run fast physics once per unique structure and materialize canonical outputs.

    Required compound columns are ``compound_id`` and ``standardized_smiles``;
    ``mw`` is carried into the canonical summary when present.  Duplicate
    structures are calculated once, then mapped back to every compound ID.
    """

    required = {"compound_id", "standardized_smiles"}
    missing = sorted(required - set(compounds.columns))
    if missing:
        raise ValueError(f"compounds is missing required columns: {missing}")
    if output_dir is None:
        raise ValueError("output_dir is required")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    structure_root = destination / "structures"
    structure_root.mkdir(parents=True, exist_ok=True)
    resolved_config = _config_from_mapping(config, smoke=smoke)
    requested_workers, cache_root = _execution_options(config, destination=destination)
    cache_root.mkdir(parents=True, exist_ok=True)

    registry_rows: list[dict[str, Any]] = []
    for row in compounds.itertuples(index=False):
        standardized = standardize_smiles(
            row.standardized_smiles,
            strip_salts=True,
            canonicalize_tautomer=False,
            require_rdkit=True,
        )
        registry_rows.append(
            {
                "compound_id": str(row.compound_id),
                "input_standardized_smiles": str(row.standardized_smiles),
                "structure_id": standardized.structure_id,
                "standardized_smiles": standardized.standardized_smiles,
                "mw": getattr(row, "mw", np.nan),
                "structure_valid": standardized.structure_valid,
                "structure_error": standardized.structure_error,
            }
        )
    registry = pd.DataFrame(registry_rows)
    valid = registry.loc[(registry["structure_valid"] == True) & (registry["structure_id"] != "")].copy()  # noqa: E712
    failures: list[dict[str, Any]] = registry.loc[~registry.index.isin(valid.index)].to_dict("records")
    jobs: list[_StructureJob] = []
    for structure_id, group in valid.groupby("structure_id", sort=True):
        smiles_value = str(group["standardized_smiles"].iloc[0])
        compound_ids = set(group["compound_id"].astype(str))
        evidence = _evidence_from_table(chemical_states, compound_ids=compound_ids)
        jobs.append(
            _StructureJob(
                structure_id=str(structure_id),
                standardized_smiles=smiles_value,
                compound_ids=tuple(sorted(compound_ids)),
                evidence=evidence,
                config=resolved_config,
                cache_root=str(cache_root),
            )
        )

    worker_count = min(requested_workers, max(1, len(jobs)))
    requested_worker_count = worker_count
    if worker_count == 1:
        job_results = [_execute_structure_job(job) for job in jobs]
    else:
        context = multiprocessing.get_context("spawn")
        try:
            with ProcessPoolExecutor(max_workers=worker_count, mp_context=context) as executor:
                job_results = list(executor.map(_execute_structure_job, jobs, chunksize=1))
        except (NotImplementedError, OSError):
            # Restricted runtimes can deny the semaphore syscalls used by
            # ProcessPoolExecutor.  Scientific correctness is more important
            # than parallel speed, and any cache entries already promoted by
            # workers make this deterministic fallback resumable.
            worker_count = 1
            job_results = [_execute_structure_job(job) for job in jobs]

    job_by_structure = {job.structure_id: job for job in jobs}
    successful_cache_paths: dict[str, Path] = {}
    cache_hit_count = 0
    for job_result in job_results:
        structure_id = str(job_result["structure_id"])
        if job_result["error"]:
            for compound_id in job_result["compound_ids"]:
                failures.append(
                    {
                        "compound_id": compound_id,
                        "structure_id": structure_id,
                        "standardized_smiles": job_by_structure[structure_id].standardized_smiles,
                        "structure_valid": True,
                        "structure_error": str(job_result["error"]),
                    }
                )
            continue
        cache_path = Path(str(job_result["cache_path"]))
        _materialize_cached_structure(
            cache_path,
            structure_root / structure_id,
            job_by_structure[structure_id],
        )
        successful_cache_paths[structure_id] = cache_path
        cache_hit_count += int(bool(job_result["cache_hit"]))

    if not successful_cache_paths:
        raise RuntimeError("Fast physics failed for every valid unique structure")

    compound_map = valid[["compound_id", "structure_id", "mw"]].copy()
    frames: dict[str, list[pd.DataFrame]] = {
        "summary": [],
        "admissibility": [],
        "states": [],
        "populations": [],
        "conformers": [],
        "composites": [],
        "quality": [],
        "state_threshold_audit": [],
        "sampling_escalation_queue": [],
    }
    for structure_id in sorted(successful_cache_paths):
        materialized_path = structure_root / structure_id
        tables = _load_cached_structure(materialized_path)
        cache_metadata = json.loads((materialized_path / "metadata.json").read_text(encoding="utf-8"))
        mapping = compound_map.loc[compound_map["structure_id"] == structure_id]
        # Aggregate from the cached scientific primitives, not cached derived
        # summaries.  This preserves cache identity while allowing corrected or
        # extended feature definitions to be applied to existing conformers.
        refreshed_summary, refreshed_composites = _ensemble_tables(
            structure_id=structure_id,
            states=tables["states"],
            populations=tables["populations"],
            conformers=tables["conformers"],
        )
        refreshed_summary, admissibility = _attach_physics_admissibility(
            structure_id=structure_id,
            quality=tables["quality"],
            summary=refreshed_summary,
            composites=refreshed_composites,
            states=tables["states"],
            populations=tables["populations"],
            conformers=tables["conformers"],
            smoke_mode=resolved_config.smoke_mode,
        )
        frames["summary"].append(mapping.merge(refreshed_summary, on="structure_id", how="inner"))
        frames["admissibility"].append(admissibility)
        frames["states"].append(tables["states"])
        frames["populations"].append(tables["populations"].assign(structure_id=structure_id))
        frames["conformers"].append(tables["conformers"])
        frames["composites"].append(refreshed_composites)
        frames["quality"].append(tables["quality"].assign(structure_id=structure_id))
        threshold_rows = cache_metadata.get("state_retention", {}).get("threshold_audit", [])
        if threshold_rows:
            frames["state_threshold_audit"].append(
                pd.DataFrame(threshold_rows).assign(structure_id=structure_id)
            )
        frames["sampling_escalation_queue"].append(
            _conformer_sampling_escalation_queue(tables["states"], tables["conformers"], resolved_config)
        )

    canonical_paths: dict[str, Path] = {
        "summary_path": destination / "fast_physics_summary.parquet",
        "admissibility_path": destination / "fast_physics_admissibility.parquet",
        "states_path": destination / "fast_physics_states.parquet",
        "populations_path": destination / "fast_physics_state_populations.parquet",
        "conformers_path": destination / "fast_physics_conformers.parquet",
        "composites_path": destination / "fast_physics_composites.parquet",
        "quality_path": destination / "fast_physics_quality_gates.parquet",
        "registry_path": destination / "fast_physics_structure_registry.parquet",
        "failures_path": destination / "fast_physics_failures.parquet",
        "state_threshold_audit_path": destination / "fast_physics_state_threshold_audit.parquet",
        "sampling_escalation_queue_path": destination / "fast_physics_sampling_escalation_queue.parquet",
    }
    for key, frame_key in (
        ("summary_path", "summary"),
        ("admissibility_path", "admissibility"),
        ("states_path", "states"),
        ("populations_path", "populations"),
        ("conformers_path", "conformers"),
        ("composites_path", "composites"),
        ("quality_path", "quality"),
        ("state_threshold_audit_path", "state_threshold_audit"),
        ("sampling_escalation_queue_path", "sampling_escalation_queue"),
    ):
        pd.concat(frames[frame_key], ignore_index=True).to_parquet(canonical_paths[key], index=False)
    registry.to_parquet(canonical_paths["registry_path"], index=False)
    pd.DataFrame(failures).to_parquet(canonical_paths["failures_path"], index=False)
    admissibility_table = pd.concat(frames["admissibility"], ignore_index=True)
    escalation_table = pd.concat(frames["sampling_escalation_queue"], ignore_index=True)

    return {
        **canonical_paths,
        "ensemble_sdf_dir": structure_root,
        "cache_dir": cache_root,
        "compound_count": int(len(compounds)),
        "unique_structure_count": int(valid["structure_id"].nunique()),
        "successful_structure_count": int(len(successful_cache_paths)),
        "failed_compound_count": int(len(failures)),
        "cache_hit_structure_count": int(cache_hit_count),
        "computed_structure_count": int(len(successful_cache_paths) - cache_hit_count),
        "physics_model_eligible_structure_count": int(
            admissibility_table["physics_model_eligible"].fillna(False).astype(bool).sum()
        ),
        "physics_substantively_ineligible_structure_count": int(
            (~admissibility_table["physics_model_eligible"].fillna(False).astype(bool)).sum()
        ),
        "physics_smoke_sampling_limited_structure_count": int(
            (admissibility_table["physics_smoke_sampling_failure_count"] > 0).sum()
        ),
        "worker_count": int(worker_count),
        "requested_worker_count": int(requested_worker_count),
        "sampling_audited_state_count": int(len(escalation_table)),
        "sampling_escalation_required_state_count": int(
            escalation_table["escalation_required"].fillna(False).astype(bool).sum()
        ),
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def write_fast_physics_outputs(result: FastPhysicsResult, output_dir: str | Path) -> dict[str, Path]:
    """Write Parquet summaries, a multi-record conformer SDF, and metadata."""

    _require_rdkit()
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    paths = {
        "states": destination / "states.parquet",
        "populations": destination / "state_populations.parquet",
        "conformers": destination / "conformers.parquet",
        "ensemble_summary": destination / "ensemble_summary.parquet",
        "composites": destination / "composites.parquet",
        "quality": destination / "quality_gates.parquet",
        "sdf": destination / "conformers.sdf",
        "metadata": destination / "metadata.json",
    }
    result.states.to_parquet(paths["states"], index=False)
    result.populations.to_parquet(paths["populations"], index=False)
    result.conformers.to_parquet(paths["conformers"], index=False)
    result.ensemble_summary.to_parquet(paths["ensemble_summary"], index=False)
    result.composites.to_parquet(paths["composites"], index=False)
    result.quality.to_parquet(paths["quality"], index=False)

    writer = Chem.SDWriter(str(paths["sdf"]))
    try:
        for state_id, group in result.conformers.groupby("state_id", sort=True):
            mol = result.state_molecules[state_id]
            for row in group.sort_values("conformer_rank").itertuples(index=False):
                mol.SetProp("_Name", str(row.conformer_id))
                mol.SetProp("structure_id", result.structure_id)
                mol.SetProp("state_id", str(state_id))
                mol.SetProp("conformer_id", str(row.conformer_id))
                mol.SetProp("conformer_weight", f"{float(row.conformer_weight):.12g}")
                mol.SetProp("energy_kcal_mol", f"{float(row.energy_kcal_mol):.12g}")
                writer.write(mol, confId=int(row.rdkit_conformer_id))
    finally:
        writer.close()
    paths["metadata"].write_text(
        json.dumps(result.metadata, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    return paths


__all__ = [
    "DEFAULT_PH_GRID",
    "FAST_PHYSICS_VERSION",
    "FastPhysicsConfig",
    "FastPhysicsResult",
    "PKaEvidence",
    "run_fast_physics",
    "run_structure_fast_physics",
    "write_fast_physics_outputs",
]
