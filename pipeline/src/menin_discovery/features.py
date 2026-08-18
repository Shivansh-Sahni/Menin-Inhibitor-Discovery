"""Chemical structure featurization with an explicit dependency-light fallback.

RDKit Morgan fingerprints and physicochemical descriptors are used when RDKit
is installed.  The hashed-SMILES backend is intentionally retained so data
validation, tests, and baseline modeling remain runnable in minimal
environments.  Model metadata records which backend was actually used; the
fallback should not be described as chemically equivalent to RDKit features.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Sequence
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

try:  # Optional chemistry dependency.
    from rdkit import Chem
    from rdkit.Chem import Crippen, Descriptors, Lipinski, rdFingerprintGenerator, rdMolDescriptors
    from rdkit.Chem.Scaffolds import MurckoScaffold

    RDKIT_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised in the dependency-light test environment.
    Chem = Crippen = Descriptors = Lipinski = None
    rdFingerprintGenerator = rdMolDescriptors = MurckoScaffold = None
    RDKIT_AVAILABLE = False


RDKIT_DESCRIPTOR_COLUMNS = (
    "mol_wt",
    "exact_mol_wt",
    "logp",
    "tpsa",
    "h_bond_donors",
    "h_bond_acceptors",
    "rotatable_bonds",
    "ring_count",
    "aromatic_ring_count",
    "fraction_csp3",
    "heavy_atom_count",
    "formal_charge",
    "invalid_structure",
)
EXACT_SMILES_PROXY_METHOD = "exact_smiles_proxy"


def _as_smiles_list(smiles_values: Iterable[object]) -> list[str]:
    """Materialize an arbitrary iterable as normalized strings exactly once."""

    return ["" if value is None or pd.isna(value) else str(value).strip() for value in smiles_values]


def smiles_descriptors(smiles_values: Iterable[object]) -> pd.DataFrame:
    """Compute deterministic string descriptors for the fallback backend."""

    rows: list[dict[str, float]] = []
    for text in _as_smiles_list(smiles_values):
        atom_tokens = re.findall(r"Cl|Br|\[[^\]]+\]|[BCNOFPSIbcno]", text)
        rows.append(
            {
                "smiles_len": float(len(text)),
                "n_atoms_rough": float(len(atom_tokens)),
                "n_c": float(text.count("C") + text.count("c")),
                "n_n": float(text.count("N") + text.count("n")),
                "n_o": float(text.count("O") + text.count("o")),
                "n_s": float(text.count("S") + text.count("s")),
                "n_f": float(text.count("F")),
                "n_cl": float(text.count("Cl")),
                "n_br": float(text.count("Br")),
                "n_ring_digits": float(sum(ch.isdigit() for ch in text)),
                "n_branches": float(text.count("(") + text.count(")")),
                "n_aromatic": float(sum(ch in "bcnops" for ch in text)),
                "n_charged": float(text.count("+") + text.count("-")),
                "n_stereo": float(text.count("@") + text.count("/") + text.count("\\")),
                "n_double_bonds": float(text.count("=")),
                "n_triple_bonds": float(text.count("#")),
                "invalid_structure": float(not bool(text)),
            }
        )
    return pd.DataFrame(rows, dtype=float).fillna(0.0)


@lru_cache(maxsize=100_000)
def _mol_from_smiles(smiles: str):
    if not RDKIT_AVAILABLE or not smiles:
        return None
    try:
        return Chem.MolFromSmiles(smiles)
    except (ValueError, RuntimeError):
        return None


@lru_cache(maxsize=100_000)
def _rdkit_descriptor_values(text: str) -> tuple[float, ...]:
    mol = _mol_from_smiles(text)
    if mol is None:
        values = {column: 0.0 for column in RDKIT_DESCRIPTOR_COLUMNS}
        values["invalid_structure"] = 1.0
    else:
        values = {
            "mol_wt": float(Descriptors.MolWt(mol)),
            "exact_mol_wt": float(Descriptors.ExactMolWt(mol)),
            "logp": float(Crippen.MolLogP(mol)),
            "tpsa": float(rdMolDescriptors.CalcTPSA(mol)),
            "h_bond_donors": float(Lipinski.NumHDonors(mol)),
            "h_bond_acceptors": float(Lipinski.NumHAcceptors(mol)),
            "rotatable_bonds": float(Lipinski.NumRotatableBonds(mol)),
            "ring_count": float(Lipinski.RingCount(mol)),
            "aromatic_ring_count": float(rdMolDescriptors.CalcNumAromaticRings(mol)),
            "fraction_csp3": float(rdMolDescriptors.CalcFractionCSP3(mol)),
            "heavy_atom_count": float(mol.GetNumHeavyAtoms()),
            "formal_charge": float(Chem.GetFormalCharge(mol)),
            "invalid_structure": 0.0,
        }
    return tuple(values[column] for column in RDKIT_DESCRIPTOR_COLUMNS)


def rdkit_descriptors(smiles_values: Iterable[object]) -> pd.DataFrame:
    """Calculate a compact, interpretable physicochemical descriptor panel."""

    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit is required for rdkit_descriptors()")

    rows = [_rdkit_descriptor_values(text) for text in _as_smiles_list(smiles_values)]
    return pd.DataFrame(rows, columns=RDKIT_DESCRIPTOR_COLUMNS, dtype=float).fillna(0.0)


@lru_cache(maxsize=100_000)
def _morgan_on_bits(smiles: str, n_bits: int, radius: int) -> tuple[int, ...]:
    mol = _mol_from_smiles(smiles)
    if mol is None:
        return ()
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    return tuple(int(index) for index in generator.GetFingerprint(mol).GetOnBits())


def _morgan_matrix(smiles: Sequence[str], *, n_bits: int, radius: int) -> sparse.csr_matrix:
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit is required for Morgan fingerprints")
    indices: list[int] = []
    indptr = [0]
    for text in smiles:
        indices.extend(_morgan_on_bits(text, n_bits, radius))
        indptr.append(len(indices))
    data = np.ones(len(indices), dtype=np.float32)
    return sparse.csr_matrix(
        (data, np.asarray(indices, dtype=np.int32), np.asarray(indptr, dtype=np.int64)),
        shape=(len(smiles), n_bits),
        dtype=np.float32,
    )


def _hashed_fingerprint_matrix(
    smiles: Sequence[str],
    *,
    n_bits: int,
    ngram_min: int = 2,
    ngram_max: int = 4,
) -> sparse.csr_matrix:
    vectorizer = HashingVectorizer(
        analyzer="char",
        ngram_range=(ngram_min, ngram_max),
        n_features=n_bits,
        alternate_sign=False,
        binary=True,
        norm=None,
        lowercase=False,
    )
    matrix = vectorizer.transform(smiles).tocsr().astype(np.float32)
    matrix.data[:] = 1.0
    return matrix


def fingerprint_matrix(
    smiles_values: Iterable[object],
    *,
    backend: str = "auto",
    n_bits: int = 2048,
    radius: int = 2,
) -> tuple[sparse.csr_matrix, str]:
    """Return binary fingerprints and the resolved backend name.

    ``auto`` resolves to ``rdkit_morgan`` when available, otherwise to the
    transparent ``hashed_smiles`` fallback.  Requesting ``rdkit`` explicitly
    raises if the dependency is absent so publication runs cannot silently
    change representations.
    """

    smiles = _as_smiles_list(smiles_values)
    requested = backend.strip().lower()
    if requested not in {"auto", "rdkit", "rdkit_morgan", "hashed", "hashed_smiles"}:
        raise ValueError(f"Unsupported fingerprint backend: {backend!r}")
    if requested in {"rdkit", "rdkit_morgan"} and not RDKIT_AVAILABLE:
        raise ImportError("RDKit backend requested but RDKit is not installed")
    if requested in {"rdkit", "rdkit_morgan"} or (requested == "auto" and RDKIT_AVAILABLE):
        return _morgan_matrix(smiles, n_bits=n_bits, radius=radius), "rdkit_morgan"
    return _hashed_fingerprint_matrix(smiles, n_bits=n_bits), "hashed_smiles"


def canonicalize_smiles(smiles: object, *, isomeric: bool = True) -> str:
    """Canonicalize with RDKit, retaining the original text as a clear fallback."""

    text = _as_smiles_list([smiles])[0]
    mol = _mol_from_smiles(text)
    if mol is None:
        return text
    return str(Chem.MolToSmiles(mol, canonical=True, isomericSmiles=isomeric))


def _exact_smiles_proxy(text: str, *, method: str) -> tuple[str, str]:
    normalized = re.sub(r"\s+", "", text)
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:20]
    return f"exact:{digest}", method


def is_exact_smiles_proxy_method(method: object) -> bool:
    """Return whether a scaffold method is an exact-SMILES leakage proxy.

    The suffix-aware predicate deliberately covers both the base invalid-
    structure fallback and named future failure modes without treating an
    unrelated method that merely begins with the same letters as equivalent.
    """

    value = str(method).strip()
    return value == EXACT_SMILES_PROXY_METHOD or value.startswith(f"{EXACT_SMILES_PROXY_METHOD}_")


def scaffold_key(smiles: object) -> tuple[str, str]:
    """Return a Bemis-Murcko scaffold key and the method used.

    Acyclic structures receive an exact canonical-structure key rather than a
    shared empty scaffold.  Without RDKit, exact normalized SMILES groups are
    used.  That fallback prevents duplicate leakage but is not a scaffold split,
    and callers expose the method in their metadata.
    """

    text = _as_smiles_list([smiles])[0]
    try:
        mol = _mol_from_smiles(text)
        if mol is not None:
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
            if scaffold:
                return scaffold, "bemis_murcko"
            canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
            digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:20]
            return f"acyclic:{digest}", "bemis_murcko_with_exact_acyclic"
    except (RuntimeError, ValueError):
        # RDKit can parse a structure successfully yet fail while extracting or
        # canonicalizing its Murcko scaffold (for example, a malformed retained
        # double-bond stereo flag). Preserve exact grouping deterministically
        # instead of aborting a feature-only split transaction.
        return _exact_smiles_proxy(
            text,
            method=f"{EXACT_SMILES_PROXY_METHOD}_rdkit_exception",
        )
    return _exact_smiles_proxy(text, method=EXACT_SMILES_PROXY_METHOD)


def nearest_neighbor_tanimoto(
    query_smiles: Iterable[object],
    reference_smiles: Iterable[object],
    *,
    backend: str = "auto",
    n_bits: int = 2048,
    radius: int = 2,
    chunk_size: int = 256,
    exclude_identical_positions: bool = False,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Compute maximum binary-fingerprint Tanimoto similarity in bounded memory."""

    query = _as_smiles_list(query_smiles)
    reference = _as_smiles_list(reference_smiles)
    if not reference:
        return np.zeros(len(query), dtype=float), np.full(len(query), -1, dtype=int), "none"

    ref_matrix, resolved = fingerprint_matrix(reference, backend=backend, n_bits=n_bits, radius=radius)
    query_matrix, _ = fingerprint_matrix(query, backend=resolved, n_bits=n_bits, radius=radius)
    ref_counts = np.asarray(ref_matrix.sum(axis=1)).ravel()
    query_counts = np.asarray(query_matrix.sum(axis=1)).ravel()
    maxima = np.zeros(len(query), dtype=float)
    neighbors = np.full(len(query), -1, dtype=int)

    for start in range(0, len(query), max(1, chunk_size)):
        stop = min(start + max(1, chunk_size), len(query))
        intersections = (query_matrix[start:stop] @ ref_matrix.T).toarray()
        denominators = query_counts[start:stop, None] + ref_counts[None, :] - intersections
        similarities = np.divide(
            intersections,
            denominators,
            out=np.zeros_like(intersections, dtype=float),
            where=denominators > 0,
        )
        if exclude_identical_positions and len(query) == len(reference):
            local = np.arange(start, stop)
            similarities[np.arange(stop - start), local] = -1.0
        best = np.argmax(similarities, axis=1)
        best_values = similarities[np.arange(stop - start), best]
        invalid = query_counts[start:stop] <= 0
        best_values[invalid] = 0.0
        best[invalid] = -1
        maxima[start:stop] = np.maximum(best_values, 0.0)
        neighbors[start:stop] = best
    return maxima, neighbors, resolved


class SmilesFeatureTransformer(BaseEstimator, TransformerMixin):
    """Morgan fingerprints plus descriptors, with hashed-SMILES fallback.

    The historical class name is retained so existing pipelines and imports
    continue to work.  ``backend='auto'`` is convenient for exploration;
    publication runs should set ``backend='rdkit'`` and record the environment.
    """

    def __init__(
        self,
        n_features: int = 2048,
        ngram_min: int = 2,
        ngram_max: int = 4,
        *,
        backend: str = "auto",
        radius: int = 2,
        include_descriptors: bool = True,
        scale_descriptors: bool = True,
    ):
        self.n_features = n_features
        self.ngram_min = ngram_min
        self.ngram_max = ngram_max
        self.backend = backend
        self.radius = radius
        self.include_descriptors = include_descriptors
        self.scale_descriptors = scale_descriptors

    def fit(self, X: Iterable[object], y: object = None) -> SmilesFeatureTransformer:
        smiles = _as_smiles_list(X)
        _, self.backend_ = fingerprint_matrix(
            smiles[:1], backend=self.backend, n_bits=self.n_features, radius=self.radius
        )
        if self.backend_ == "hashed_smiles":
            self.vectorizer_ = HashingVectorizer(
                analyzer="char",
                ngram_range=(self.ngram_min, self.ngram_max),
                n_features=self.n_features,
                alternate_sign=False,
                norm="l2",
                lowercase=False,
            )

        self.descriptor_columns_: list[str] = []
        self.scaler_ = None
        if self.include_descriptors:
            descriptors = (
                rdkit_descriptors(smiles) if self.backend_ == "rdkit_morgan" else smiles_descriptors(smiles)
            )
            self.descriptor_columns_ = list(descriptors.columns)
            if self.scale_descriptors:
                self.scaler_ = StandardScaler()
                self.scaler_.fit(descriptors.to_numpy(dtype=float))
        self.feature_metadata_ = {
            "backend": self.backend_,
            "fingerprint_bits": int(self.n_features),
            "morgan_radius": int(self.radius) if self.backend_ == "rdkit_morgan" else None,
            "descriptor_columns": list(self.descriptor_columns_),
            "fallback_used": self.backend_ != "rdkit_morgan",
        }
        return self

    def transform(self, X: Iterable[object]):
        check_is_fitted(self, "backend_")
        smiles = _as_smiles_list(X)
        if self.backend_ == "rdkit_morgan":
            fingerprints = _morgan_matrix(smiles, n_bits=self.n_features, radius=self.radius)
        else:
            fingerprints = self.vectorizer_.transform(smiles).tocsr().astype(np.float32)

        if not self.include_descriptors:
            return fingerprints
        descriptors = (
            rdkit_descriptors(smiles) if self.backend_ == "rdkit_morgan" else smiles_descriptors(smiles)
        )
        descriptors = descriptors.reindex(columns=self.descriptor_columns_, fill_value=0.0)
        values = descriptors.to_numpy(dtype=float)
        if self.scaler_ is not None:
            values = self.scaler_.transform(values)
        return sparse.hstack([fingerprints, sparse.csr_matrix(values)], format="csr")

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        check_is_fitted(self, "backend_")
        prefix = "morgan" if self.backend_ == "rdkit_morgan" else "smiles_hash"
        fingerprint_names = [f"{prefix}_{index}" for index in range(self.n_features)]
        return np.asarray(fingerprint_names + list(self.descriptor_columns_), dtype=object)
