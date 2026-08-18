"""Small-data directed message-passing comparator for continuous hERG potency.

This is a transparent research comparator, not a release model.  It uses the
same scaffold-held-out folds and censored Gaussian objective as the tabular
models so representation—not split optimism—is what changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem
from sklearn.model_selection import GroupKFold

try:
    import torch
    from torch import nn

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    torch = nn = None
    TORCH_AVAILABLE = False

from .research_feature_ontology import MODEL_CONFORMER_FEATURES
from .research_modeling import _heldout_censored_nll, continuous_herg_metrics, herg_classification_metrics

ATOM_DIM = 12
BOND_DIM = 7


@dataclass(frozen=True)
class MolGraph:
    atom_features: Any
    edge_features: Any
    edge_sources: Any
    edge_targets: Any
    reverse_edges: Any


def _atom_features(atom: Chem.Atom) -> list[float]:
    hybridization = atom.GetHybridization()
    return [
        atom.GetAtomicNum() / 100.0,
        atom.GetTotalDegree() / 6.0,
        atom.GetFormalCharge() / 4.0,
        atom.GetTotalNumHs() / 4.0,
        atom.GetMass() / 200.0,
        float(atom.GetIsAromatic()),
        float(atom.IsInRing()),
        float(hybridization == Chem.HybridizationType.SP),
        float(hybridization == Chem.HybridizationType.SP2),
        float(hybridization == Chem.HybridizationType.SP3),
        float(atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED),
        float(atom.GetNoImplicit()),
    ]


def _bond_features(bond: Chem.Bond) -> list[float]:
    bond_type = bond.GetBondType()
    return [
        float(bond_type == Chem.BondType.SINGLE),
        float(bond_type == Chem.BondType.DOUBLE),
        float(bond_type == Chem.BondType.TRIPLE),
        float(bond_type == Chem.BondType.AROMATIC),
        float(bond.GetIsConjugated()),
        float(bond.IsInRing()),
        float(bond.GetStereo() != Chem.BondStereo.STEREONONE),
    ]


def molecule_graph(smiles: str) -> MolGraph:
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for the D-MPNN comparator")
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError("Invalid SMILES for D-MPNN")
    atoms = torch.tensor([_atom_features(atom) for atom in mol.GetAtoms()], dtype=torch.float32)
    edge_sources: list[int] = []
    edge_targets: list[int] = []
    edge_features: list[list[float]] = []
    reverse: list[int] = []
    for bond in mol.GetBonds():
        begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feature = _bond_features(bond)
        first = len(edge_sources)
        edge_sources.extend([begin, end])
        edge_targets.extend([end, begin])
        edge_features.extend([feature, feature])
        reverse.extend([first + 1, first])
    if not edge_sources:
        edges = torch.empty((0, BOND_DIM), dtype=torch.float32)
        sources = targets = reverse_tensor = torch.empty(0, dtype=torch.long)
    else:
        edges = torch.tensor(edge_features, dtype=torch.float32)
        sources = torch.tensor(edge_sources, dtype=torch.long)
        targets = torch.tensor(edge_targets, dtype=torch.long)
        reverse_tensor = torch.tensor(reverse, dtype=torch.long)
    return MolGraph(atoms, edges, sources, targets, reverse_tensor)


class DirectedMPNN(nn.Module):
    def __init__(self, hidden_size: int = 64, depth: int = 3, dropout: float = 0.15):
        super().__init__()
        self.hidden_size = hidden_size
        self.depth = depth
        self.edge_input = nn.Linear(ATOM_DIM + BOND_DIM, hidden_size)
        self.message = nn.Linear(hidden_size, hidden_size, bias=False)
        self.atom_output = nn.Linear(ATOM_DIM + hidden_size, hidden_size)
        self.head = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )
        self.log_sigma = nn.Parameter(torch.tensor(-0.3))

    def forward_graph(self, graph: MolGraph):
        atoms = graph.atom_features
        if len(graph.edge_sources) == 0:
            incoming = torch.zeros((len(atoms), self.hidden_size), dtype=atoms.dtype, device=atoms.device)
        else:
            initial = torch.relu(
                self.edge_input(torch.cat([atoms[graph.edge_sources], graph.edge_features], dim=1))
            )
            hidden = initial
            for _ in range(self.depth - 1):
                atom_incoming = torch.zeros(
                    (len(atoms), self.hidden_size), dtype=hidden.dtype, device=hidden.device
                )
                atom_incoming.index_add_(0, graph.edge_targets, hidden)
                message = atom_incoming[graph.edge_sources] - hidden[graph.reverse_edges]
                hidden = torch.relu(initial + self.message(message))
            incoming = torch.zeros((len(atoms), self.hidden_size), dtype=hidden.dtype, device=hidden.device)
            incoming.index_add_(0, graph.edge_targets, hidden)
        atom_hidden = torch.relu(self.atom_output(torch.cat([atoms, incoming], dim=1)))
        pooled = atom_hidden.mean(dim=0)
        return self.head(pooled).squeeze(-1)

    def forward(self, graphs: list[MolGraph]):
        return torch.stack([self.forward_graph(graph) for graph in graphs])


class ConformerAttentionRegressor(nn.Module):
    """Multiple-instance model over weighted state/conformer observations."""

    def __init__(self, feature_size: int, hidden_size: int = 48, dropout: float = 0.15):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(feature_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.attention = nn.Linear(hidden_size, 1)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2), nn.ReLU(), nn.Linear(hidden_size // 2, 1)
        )
        self.log_sigma = nn.Parameter(torch.tensor(-0.3))

    def forward_bag(self, features, prior_weights):
        encoded = self.encoder(features)
        log_prior = torch.log(torch.clamp(prior_weights, min=1e-8)).reshape(-1, 1)
        attention = torch.softmax(self.attention(encoded) + log_prior, dim=0)
        pooled = torch.sum(attention * encoded, dim=0)
        return self.head(pooled).squeeze(-1)

    def forward(self, bags: list[tuple[Any, Any]]):
        return torch.stack([self.forward_bag(features, weights) for features, weights in bags])


def _censored_nll_torch(mu, sigma, lower, upper):
    finite_lower = torch.isfinite(lower)
    finite_upper = torch.isfinite(upper)
    exact = finite_lower & finite_upper & (torch.abs(lower - upper) < 1e-8)
    logp = torch.zeros_like(mu)
    if exact.any():
        z = (lower[exact] - mu[exact]) / sigma
        logp[exact] = -torch.log(sigma) - 0.5 * z**2 - 0.5 * np.log(2 * np.pi)
    lower_only = finite_lower & ~finite_upper
    if lower_only.any():
        z = (mu[lower_only] - lower[lower_only]) / sigma
        logp[lower_only] = torch.special.log_ndtr(z)
    upper_only = ~finite_lower & finite_upper
    if upper_only.any():
        z = (upper[upper_only] - mu[upper_only]) / sigma
        logp[upper_only] = torch.special.log_ndtr(z)
    interval = finite_lower & finite_upper & ~exact
    if interval.any():
        upper_cdf = torch.special.ndtr((upper[interval] - mu[interval]) / sigma)
        lower_cdf = torch.special.ndtr((lower[interval] - mu[interval]) / sigma)
        logp[interval] = torch.log(torch.clamp(upper_cdf - lower_cdf, min=1e-12))
    return -logp.mean()


def grouped_dmpnn_herg_benchmark(
    data: pd.DataFrame,
    *,
    folds: int = 5,
    epochs: int = 80,
    hidden_size: int = 64,
    depth: int = 3,
    learning_rate: float = 2e-3,
    weight_decay: float = 1e-4,
    random_state: int = 20260721,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Fit one fixed D-MPNN configuration on scaffold-held-out censored data."""

    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for the D-MPNN comparator")
    required = {"compound_id", "standardized_smiles", "scaffold", "pic50_lower", "pic50_upper"}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"D-MPNN data are missing columns: {missing}")
    # Keep every measurement row. Conflicting or repeated evidence for one
    # structure must remain visible to the censored likelihood; scaffold
    # grouping still guarantees that all rows for that chemistry stay in the
    # same held-out fold.
    frame = data.copy().reset_index(drop=True)
    valid_rows: list[int] = []
    graphs: list[MolGraph] = []
    for index, smiles in enumerate(frame["standardized_smiles"]):
        try:
            graphs.append(molecule_graph(str(smiles)))
            valid_rows.append(index)
        except ValueError:
            continue
    frame = frame.iloc[valid_rows].reset_index(drop=True)
    groups = frame["scaffold"].astype(str).to_numpy()
    n_splits = min(folds, len(np.unique(groups)))
    splitter = GroupKFold(n_splits=n_splits)
    rows: list[dict[str, Any]] = []
    lower_all = frame["pic50_lower"].to_numpy(dtype=float)
    upper_all = frame["pic50_upper"].to_numpy(dtype=float)
    for fold_index, (train, test) in enumerate(splitter.split(frame, groups=groups)):
        torch.manual_seed(random_state + fold_index)
        np.random.seed(random_state + fold_index)
        model = DirectedMPNN(hidden_size=hidden_size, depth=depth)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        lower = torch.tensor(lower_all[train], dtype=torch.float32)
        upper = torch.tensor(upper_all[train], dtype=torch.float32)
        train_graphs = [graphs[index] for index in train]
        model.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            mu = model(train_graphs)
            sigma = torch.nn.functional.softplus(model.log_sigma) + 1e-3
            loss = _censored_nll_torch(mu, sigma, lower, upper)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            predicted = model([graphs[index] for index in test]).numpy()
            sigma = float(torch.nn.functional.softplus(model.log_sigma).item() + 1e-3)
            probability = 1.0 - torch.special.ndtr(torch.tensor((5.0 - predicted) / sigma)).numpy()
        for position, index in enumerate(test):
            lo, hi = lower_all[index], upper_all[index]
            is_exact = np.isfinite(lo) and np.isfinite(hi) and np.isclose(lo, hi)
            decisive = (
                1
                if np.isfinite(lo) and lo >= 5.0
                else 0
                if np.isfinite(hi) and hi <= 6.0 - np.log10(30.0)
                else np.nan
            )
            rows.append(
                {
                    "compound_id": frame.loc[index, "compound_id"],
                    "fold": fold_index,
                    "group": groups[index],
                    "pic50_lower": lo,
                    "pic50_upper": hi,
                    "is_exact": is_exact,
                    "observed_pic50": lo if is_exact else np.nan,
                    "predicted_pic50": float(predicted[position]),
                    "predictive_sigma": sigma,
                    "blocker_probability": float(probability[position]),
                    "decisive_label": decisive,
                }
            )
    predictions = pd.DataFrame(rows)
    exact = predictions["is_exact"].astype(bool)
    metrics: dict[str, Any] = {
        "model": "directed_message_passing_neural_network",
        "promotion_status": "discovery-track",
        "censored_negative_log_likelihood": _heldout_censored_nll(predictions),
        **continuous_herg_metrics(
            predictions.loc[exact, "observed_pic50"].to_numpy(),
            predictions.loc[exact, "predicted_pic50"].to_numpy(),
        ),
    }
    decisive = predictions["decisive_label"].notna()
    if decisive.any():
        metrics.update(
            {
                f"classification_{key}": value
                for key, value in herg_classification_metrics(
                    predictions.loc[decisive, "decisive_label"].astype(int).to_numpy(),
                    predictions.loc[decisive, "blocker_probability"].to_numpy(),
                ).items()
            }
        )
    return metrics, predictions


def grouped_conformer_mil_benchmark(
    compounds: pd.DataFrame,
    conformers: pd.DataFrame,
    *,
    feature_columns: list[str],
    weight_column: str = "ensemble_weight",
    folds: int = 5,
    epochs: int = 100,
    hidden_size: int = 48,
    random_state: int = 20260721,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Evaluate conformers as weighted bags without flattening them into pseudo-replicates."""

    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for the conformer MIL comparator")
    unapproved = sorted(set(feature_columns) - set(MODEL_CONFORMER_FEATURES))
    if unapproved:
        raise ValueError(f"MIL conformer features are not ontology-approved: {unapproved}")
    if len(feature_columns) != len(set(feature_columns)):
        raise ValueError("MIL conformer features must be unique")
    compound_required = {"compound_id", "scaffold", "pic50_lower", "pic50_upper"}
    conformer_required = {"compound_id", weight_column, *feature_columns}
    if missing := sorted(compound_required - set(compounds.columns)):
        raise ValueError(f"MIL compound data are missing columns: {missing}")
    if missing := sorted(conformer_required - set(conformers.columns)):
        raise ValueError(f"MIL conformer data are missing columns: {missing}")
    # One conformer bag may legitimately supervise multiple source-specific
    # interval measurements. Preserve those evidence rows rather than silently
    # choosing the first label for a compound.
    frame = compounds.copy()
    available = set(conformers["compound_id"].astype(str))
    frame = frame[frame["compound_id"].astype(str).isin(available)].reset_index(drop=True)
    groups = frame["scaffold"].astype(str).to_numpy()
    bags: list[tuple[Any, Any]] = []
    for compound_id in frame["compound_id"].astype(str):
        bag = conformers[conformers["compound_id"].astype(str) == compound_id]
        values = (
            bag[feature_columns]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(bag[feature_columns].median())
            .fillna(0.0)
        )
        weights = (
            pd.to_numeric(bag[weight_column], errors="coerce")
            .fillna(0.0)
            .clip(lower=0.0)
            .to_numpy(dtype=float)
        )
        if weights.sum() <= 0:
            weights = np.full(len(bag), 1.0 / len(bag))
        else:
            weights = weights / weights.sum()
        bags.append(
            (torch.tensor(values.to_numpy(dtype=np.float32)), torch.tensor(weights, dtype=torch.float32))
        )
    lower_all = frame["pic50_lower"].to_numpy(dtype=float)
    upper_all = frame["pic50_upper"].to_numpy(dtype=float)
    splitter = GroupKFold(n_splits=min(folds, len(np.unique(groups))))
    rows: list[dict[str, Any]] = []
    for fold_index, (train, test) in enumerate(splitter.split(frame, groups=groups)):
        torch.manual_seed(random_state + fold_index)
        model = ConformerAttentionRegressor(len(feature_columns), hidden_size=hidden_size)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
        lower = torch.tensor(lower_all[train], dtype=torch.float32)
        upper = torch.tensor(upper_all[train], dtype=torch.float32)
        model.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            mu = model([bags[index] for index in train])
            sigma = torch.nn.functional.softplus(model.log_sigma) + 1e-3
            loss = _censored_nll_torch(mu, sigma, lower, upper)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            prediction = model([bags[index] for index in test]).numpy()
            sigma = float(torch.nn.functional.softplus(model.log_sigma).item() + 1e-3)
            probability = 1.0 - torch.special.ndtr(torch.tensor((5.0 - prediction) / sigma)).numpy()
        for position, index in enumerate(test):
            lo, hi = lower_all[index], upper_all[index]
            exact = np.isfinite(lo) and np.isfinite(hi) and np.isclose(lo, hi)
            decisive = (
                1
                if np.isfinite(lo) and lo >= 5.0
                else 0
                if np.isfinite(hi) and hi <= 6.0 - np.log10(30.0)
                else np.nan
            )
            rows.append(
                {
                    "compound_id": frame.loc[index, "compound_id"],
                    "fold": fold_index,
                    "group": groups[index],
                    "pic50_lower": lo,
                    "pic50_upper": hi,
                    "is_exact": exact,
                    "observed_pic50": lo if exact else np.nan,
                    "predicted_pic50": float(prediction[position]),
                    "predictive_sigma": sigma,
                    "blocker_probability": float(probability[position]),
                    "decisive_label": decisive,
                }
            )
    predictions = pd.DataFrame(rows)
    exact = predictions["is_exact"].astype(bool)
    metrics: dict[str, Any] = {
        "model": "state_conformer_attention_multiple_instance",
        "promotion_status": "discovery-track",
        "censored_negative_log_likelihood": _heldout_censored_nll(predictions),
        **continuous_herg_metrics(
            predictions.loc[exact, "observed_pic50"].to_numpy(),
            predictions.loc[exact, "predicted_pic50"].to_numpy(),
        ),
    }
    decisive = predictions["decisive_label"].notna()
    if decisive.any():
        metrics.update(
            {
                f"classification_{key}": value
                for key, value in herg_classification_metrics(
                    predictions.loc[decisive, "decisive_label"].astype(int).to_numpy(),
                    predictions.loc[decisive, "blocker_probability"].to_numpy(),
                ).items()
            }
        )
    return metrics, predictions
