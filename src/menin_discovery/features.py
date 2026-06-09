"""Lightweight SMILES featurization that works without RDKit."""

from __future__ import annotations

import re
from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import StandardScaler


def smiles_descriptors(smiles_values: Iterable[str]) -> pd.DataFrame:
    """Compute simple string-derived descriptors as a dependency-light baseline."""

    rows: list[dict[str, float]] = []
    for smiles in smiles_values:
        text = "" if smiles is None else str(smiles)
        atom_tokens = re.findall(r"Cl|Br|[BCNOFPSIbcno]", text)
        rows.append(
            {
                "smiles_len": len(text),
                "n_atoms_rough": len(atom_tokens),
                "n_c": text.count("C") + text.count("c"),
                "n_n": text.count("N") + text.count("n"),
                "n_o": text.count("O") + text.count("o"),
                "n_s": text.count("S") + text.count("s"),
                "n_f": text.count("F"),
                "n_cl": text.count("Cl"),
                "n_br": text.count("Br"),
                "n_ring_digits": sum(ch.isdigit() for ch in text),
                "n_branches": text.count("(") + text.count(")"),
                "n_aromatic": sum(ch in "bcnops" for ch in text),
                "n_charged": text.count("+") + text.count("-"),
                "n_stereo": text.count("@"),
                "n_double_bonds": text.count("="),
                "n_triple_bonds": text.count("#"),
            }
        )
    return pd.DataFrame(rows).fillna(0.0)


class SmilesFeatureTransformer(BaseEstimator, TransformerMixin):
    """Hash character n-grams plus simple string descriptors.

    This is intentionally RDKit-free so the baseline can run in minimal
    environments. RDKit Morgan fingerprints should replace or augment it once
    the lab environment is available.
    """

    def __init__(self, n_features: int = 2048, ngram_min: int = 2, ngram_max: int = 4):
        self.n_features = n_features
        self.ngram_min = ngram_min
        self.ngram_max = ngram_max

    def fit(self, X: Iterable[str], y: object = None) -> "SmilesFeatureTransformer":
        self.vectorizer_ = HashingVectorizer(
            analyzer="char",
            ngram_range=(self.ngram_min, self.ngram_max),
            n_features=self.n_features,
            alternate_sign=False,
            norm="l2",
        )
        desc = smiles_descriptors(X)
        self.scaler_ = StandardScaler()
        self.scaler_.fit(desc.values)
        self.descriptor_columns_ = list(desc.columns)
        return self

    def transform(self, X: Iterable[str]):
        smiles = ["" if x is None else str(x) for x in X]
        hashed = self.vectorizer_.transform(smiles)
        desc = smiles_descriptors(smiles)
        desc = desc.reindex(columns=self.descriptor_columns_, fill_value=0.0)
        desc_scaled = self.scaler_.transform(desc.values)
        return sparse.hstack([hashed, sparse.csr_matrix(desc_scaled)], format="csr")
