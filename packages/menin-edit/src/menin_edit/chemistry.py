"""Conservative RDKit primitives for supported single-fragment edits."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdFingerprintGenerator, rdMMPA, rdMolDescriptors


@dataclass(frozen=True)
class FragmentContext:
    """One single-cut decomposition with a retained core and replaceable fragment."""

    core_smiles: str
    variable_smiles: str
    core_heavy_atoms: int
    variable_heavy_atoms: int


@dataclass(frozen=True)
class ProductValidation:
    valid: bool
    canonical_smiles: str
    reason: str
    parent_similarity: float
    heavy_atom_delta: int


def canonicalize_smiles(smiles: str) -> str:
    """Return canonical isomeric SMILES or raise a clear validation error."""

    molecule = Chem.MolFromSmiles(str(smiles).strip())
    if molecule is None:
        raise ValueError("RDKit could not parse the supplied SMILES")
    return str(Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True))


def molecule_id(smiles: str) -> str:
    canonical = canonicalize_smiles(smiles)
    return "MOL-" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:20].upper()


def _heavy_atoms(fragment_smiles: str) -> int:
    molecule = Chem.MolFromSmiles(fragment_smiles)
    if molecule is None:
        return 0
    return int(sum(atom.GetAtomicNum() > 1 for atom in molecule.GetAtoms()))


def normalize_attachment_fragment(fragment_smiles: str) -> str:
    """Canonicalize a one-attachment MMP fragment into one stable namespace."""

    molecule = Chem.MolFromSmiles(str(fragment_smiles))
    if molecule is None:
        raise ValueError(f"Invalid attachment fragment: {fragment_smiles!r}")
    dummies = [atom for atom in molecule.GetAtoms() if atom.GetAtomicNum() == 0]
    if len(dummies) != 1 or dummies[0].GetDegree() != 1:
        raise ValueError("Menin-Edit v0.1 supports exactly one attachment atom per fragment")
    dummy = dummies[0]
    dummy.SetIsotope(0)
    dummy.SetAtomMapNum(1)
    return str(Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True))


@lru_cache(maxsize=100_000)
def fragment_single_cuts(
    smiles: str,
    min_core_heavy_atoms: int = 10,
    max_variable_heavy_atoms: int = 12,
) -> tuple[FragmentContext, ...]:
    """Enumerate conservative one-cut contexts compatible with the public MMP table."""

    molecule = Chem.MolFromSmiles(canonicalize_smiles(smiles))
    if molecule is None:  # pragma: no cover - canonicalization already guards this.
        return ()
    seen: set[tuple[str, str]] = set()
    contexts: list[FragmentContext] = []
    fragments = rdMMPA.FragmentMol(
        molecule,
        minCuts=1,
        maxCuts=1,
        maxCutBonds=20,
        resultsAsMols=False,
    )
    for _, fragment_smiles in fragments:
        parts = str(fragment_smiles).split(".")
        if len(parts) != 2:
            continue
        parsed = sorted(
            ((_heavy_atoms(part), part) for part in parts),
            key=lambda item: (-item[0], item[1]),
        )
        core_heavy, core_raw = parsed[0]
        variable_heavy, variable_raw = parsed[1]
        if core_heavy < min_core_heavy_atoms or variable_heavy > max_variable_heavy_atoms:
            continue
        try:
            core = normalize_attachment_fragment(core_raw)
            variable = normalize_attachment_fragment(variable_raw)
        except ValueError:
            continue
        key = (core, variable)
        if key in seen:
            continue
        seen.add(key)
        contexts.append(
            FragmentContext(
                core_smiles=core,
                variable_smiles=variable,
                core_heavy_atoms=core_heavy,
                variable_heavy_atoms=variable_heavy,
            )
        )
    return tuple(sorted(contexts, key=lambda item: (item.variable_smiles, item.core_smiles)))


def join_single_attachment_fragments(core_smiles: str, variable_smiles: str) -> str:
    """Join two one-dummy fragments and return a sanitized canonical product."""

    core = Chem.MolFromSmiles(normalize_attachment_fragment(core_smiles))
    variable = Chem.MolFromSmiles(normalize_attachment_fragment(variable_smiles))
    if core is None or variable is None:  # pragma: no cover - normalizer already guards this.
        raise ValueError("Could not parse edit fragments")
    core_dummy = next(atom for atom in core.GetAtoms() if atom.GetAtomicNum() == 0)
    variable_dummy = next(atom for atom in variable.GetAtoms() if atom.GetAtomicNum() == 0)
    core_neighbor = core_dummy.GetNeighbors()[0].GetIdx()
    variable_neighbor = variable_dummy.GetNeighbors()[0].GetIdx() + core.GetNumAtoms()
    bond_type = core.GetBondBetweenAtoms(core_dummy.GetIdx(), core_neighbor).GetBondType()
    combined = Chem.CombineMols(core, variable)
    editable = Chem.RWMol(combined)
    editable.AddBond(core_neighbor, variable_neighbor, bond_type)
    dummy_indices = sorted([core_dummy.GetIdx(), variable_dummy.GetIdx() + core.GetNumAtoms()], reverse=True)
    for index in dummy_indices:
        editable.RemoveAtom(index)
    product = editable.GetMol()
    Chem.SanitizeMol(product)
    if any(atom.GetAtomicNum() == 0 for atom in product.GetAtoms()):
        raise ValueError("Product retains an unresolved attachment atom")
    return str(Chem.MolToSmiles(product, canonical=True, isomericSmiles=True))


@lru_cache(maxsize=100_000)
def _fingerprint(smiles: str):
    molecule = Chem.MolFromSmiles(canonicalize_smiles(smiles))
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    return generator.GetFingerprint(molecule)


def tanimoto_similarity(left_smiles: str, right_smiles: str) -> float:
    return float(DataStructs.TanimotoSimilarity(_fingerprint(left_smiles), _fingerprint(right_smiles)))


def validate_product(
    parent_smiles: str,
    product_smiles: str,
    *,
    max_changed_heavy_atoms: int,
    min_parent_similarity: float,
) -> ProductValidation:
    try:
        parent = Chem.MolFromSmiles(canonicalize_smiles(parent_smiles))
        canonical = canonicalize_smiles(product_smiles)
        product = Chem.MolFromSmiles(canonical)
    except (ValueError, RuntimeError) as exc:
        return ProductValidation(False, "", f"invalid_product:{exc}", 0.0, 0)
    if parent is None or product is None:
        return ProductValidation(False, "", "invalid_product", 0.0, 0)
    if canonical == canonicalize_smiles(parent_smiles):
        return ProductValidation(False, canonical, "identity_edit", 1.0, 0)
    heavy_delta = int(product.GetNumHeavyAtoms() - parent.GetNumHeavyAtoms())
    if abs(heavy_delta) > max_changed_heavy_atoms:
        return ProductValidation(False, canonical, "heavy_atom_delta_exceeded", 0.0, heavy_delta)
    similarity = tanimoto_similarity(parent_smiles, canonical)
    if similarity < min_parent_similarity:
        return ProductValidation(False, canonical, "parent_similarity_below_limit", similarity, heavy_delta)
    return ProductValidation(True, canonical, "ok", similarity, heavy_delta)


def molecular_descriptors(smiles: str) -> dict[str, float]:
    molecule = Chem.MolFromSmiles(canonicalize_smiles(smiles))
    if molecule is None:  # pragma: no cover
        raise ValueError("Invalid molecule")
    return {
        "mol_wt": float(Descriptors.MolWt(molecule)),
        "logp": float(Crippen.MolLogP(molecule)),
        "tpsa": float(rdMolDescriptors.CalcTPSA(molecule)),
        "h_bond_donors": float(Lipinski.NumHDonors(molecule)),
        "h_bond_acceptors": float(Lipinski.NumHAcceptors(molecule)),
        "rotatable_bonds": float(Lipinski.NumRotatableBonds(molecule)),
        "heavy_atom_count": float(molecule.GetNumHeavyAtoms()),
        "formal_charge": float(Chem.GetFormalCharge(molecule)),
    }


def descriptor_deltas(parent_smiles: str, product_smiles: str) -> dict[str, float]:
    parent = molecular_descriptors(parent_smiles)
    product = molecular_descriptors(product_smiles)
    return {key: float(product[key] - parent[key]) for key in parent}


def finite_mapping(values: dict[str, Any]) -> dict[str, float]:
    """Return only finite numeric values for stable JSON/report output."""

    result: dict[str, float] = {}
    for key, value in values.items():
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number):
            result[str(key)] = number
    return result
