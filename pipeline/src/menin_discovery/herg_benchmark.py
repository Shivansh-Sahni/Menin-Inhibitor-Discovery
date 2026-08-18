"""Configurable hERG benchmarking for private Menin chemistry plus public data.

The benchmark deliberately evaluates every regime on held-out private compounds,
because the intended deployment domain is the lab's Menin series.  After model
selection, production ensembles are refit with every eligible label and score all
private compounds, including unlabeled and intermediate-potency structures.
"""

from __future__ import annotations

import hashlib
import json
import platform
import re
import time
import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import joblib
import numpy as np
import pandas as pd
import sklearn
import yaml
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import Descriptors, MACCSkeys, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy import sparse
from sklearn.base import BaseEstimator
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import ParameterGrid, RepeatedStratifiedKFold, StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler, StandardScaler
from sklearn.svm import SVC

PRIVATE_SMILES_COLUMN = "Kekule Canonical SMILES"
PRIVATE_ID_COLUMN = "Compound"
PRIVATE_HERG_IC50_COLUMN = "hERG IC50 (µM)"
PRIVATE_HERG_PERCENT_COLUMN = "hERG % inhibition"
PRIMARY_PUBLIC_ENDPOINT = "IC50"
PRIMARY_PUBLIC_ASSAY_FAMILY = "electrophysiology_functional"
REGIMES = ("confidential_only", "confidential_prioritized", "equal_importance")


@dataclass(frozen=True)
class ModelSpec:
    family: str
    complexity: str
    parameters: dict[str, Any]
    feature_sets: tuple[str, ...]

    @property
    def key(self) -> str:
        payload = json.dumps(self.parameters, sort_keys=True, default=str)
        digest = hashlib.sha1(payload.encode("utf-8"), usedforsecurity=False).hexdigest()[:8]
        return f"{self.family}__{self.complexity}__{digest}"


@dataclass
class FittedModel:
    family: str
    estimator: Any
    parameters: dict[str, Any]


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(dict(payload)), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_structure(smiles: object) -> tuple[str, str, str]:
    text = "" if pd.isna(smiles) else str(smiles).strip()
    if not text:
        return "", "", ""
    try:
        mol = Chem.MolFromSmiles(text)
    except (ValueError, RuntimeError):
        mol = None
    if mol is None:
        return "", "", ""
    canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    inchi_key = Chem.MolToInchiKey(mol)
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    return canonical, inchi_key, scaffold or canonical


def parse_ic50_um(value: object, *, blocker_max_um: float, nonblocker_min_um: float) -> dict[str, Any]:
    """Parse point or one-sided-censored IC50 and derive an interval-certain class."""

    if pd.isna(value) or not str(value).strip():
        return {"relation": None, "value_um": np.nan, "label": np.nan, "status": "missing"}
    text = str(value).strip().replace("μ", "u").replace("µ", "u")
    match = re.match(r"^\s*(<=|>=|<|>|~)?\s*([0-9]+(?:\.[0-9]+)?)", text)
    if not match:
        return {"relation": None, "value_um": np.nan, "label": np.nan, "status": "unparsed"}
    relation = match.group(1) or "="
    numeric = float(match.group(2))
    label: float = np.nan
    status = "intermediate"
    if relation in {"<", "<="} and numeric <= blocker_max_um:
        label, status = 1.0, "blocker_interval_certain"
    elif relation in {">", ">="} and numeric >= nonblocker_min_um:
        label, status = 0.0, "nonblocker_interval_certain"
    elif relation == "=" and numeric <= blocker_max_um:
        label, status = 1.0, "blocker"
    elif relation == "=" and numeric >= nonblocker_min_um:
        label, status = 0.0, "nonblocker"
    return {"relation": relation, "value_um": numeric, "label": label, "status": status}


def parse_percent_inhibition(value: object, concentration_um: int) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).replace("μ", "u").replace("µ", "u").replace(" ", "")
    match = re.search(rf"(-?[0-9]+(?:\.[0-9]+)?)%?@{concentration_um}uM", text, re.I)
    return float(match.group(1)) if match else np.nan


def load_private_workbook(
    path: Path,
    *,
    blocker_max_um: float = 10.0,
    nonblocker_min_um: float = 30.0,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Read the lab workbook and return row-level and unique-structure tables."""

    raw = pd.read_excel(path, sheet_name="SMILES")
    required = {
        PRIVATE_ID_COLUMN,
        PRIVATE_SMILES_COLUMN,
        PRIVATE_HERG_IC50_COLUMN,
        PRIVATE_HERG_PERCENT_COLUMN,
    }
    missing = sorted(required - set(raw.columns))
    if missing:
        raise KeyError(f"Private workbook is missing required columns: {missing}")
    rows = raw.copy()
    rows.insert(0, "private_row_number", np.arange(2, len(rows) + 2, dtype=int))
    structures = rows[PRIVATE_SMILES_COLUMN].map(_canonical_structure)
    rows["smiles"] = structures.map(lambda item: item[0])
    rows["standard_inchi_key"] = structures.map(lambda item: item[1])
    rows["scaffold"] = structures.map(lambda item: item[2])
    parsed = rows[PRIVATE_HERG_IC50_COLUMN].map(
        lambda value: parse_ic50_um(
            value,
            blocker_max_um=blocker_max_um,
            nonblocker_min_um=nonblocker_min_um,
        )
    )
    rows["herg_relation"] = parsed.map(lambda item: item["relation"])
    rows["herg_ic50_um"] = parsed.map(lambda item: item["value_um"])
    rows["herg_blocker_label"] = parsed.map(lambda item: item["label"])
    rows["herg_label_status"] = parsed.map(lambda item: item["status"])
    rows["herg_inhibition_10um_pct"] = rows[PRIVATE_HERG_PERCENT_COLUMN].map(
        lambda value: parse_percent_inhibition(value, 10)
    )
    rows["herg_inhibition_30um_pct"] = rows[PRIVATE_HERG_PERCENT_COLUMN].map(
        lambda value: parse_percent_inhibition(value, 30)
    )
    rows["source"] = "confidential"
    invalid = rows["smiles"].eq("")

    unique_records: list[dict[str, Any]] = []
    for structure_key, group in rows[~invalid].groupby("standard_inchi_key", sort=False):
        labels = sorted(group["herg_blocker_label"].dropna().astype(int).unique().tolist())
        label = float(labels[0]) if len(labels) == 1 else np.nan
        unique_records.append(
            {
                "standard_inchi_key": structure_key,
                "smiles": group["smiles"].iloc[0],
                "scaffold": group["scaffold"].iloc[0],
                "compound_id": " | ".join(sorted(group[PRIVATE_ID_COLUMN].dropna().astype(str).unique())),
                "private_row_numbers": " | ".join(group["private_row_number"].astype(str)),
                "n_private_rows": int(len(group)),
                "herg_blocker_label": label,
                "label_conflict": len(labels) > 1,
                "herg_ic50_um": float(group["herg_ic50_um"].median())
                if group["herg_ic50_um"].notna().any()
                else np.nan,
                "source": "confidential",
            }
        )
    unique = pd.DataFrame(unique_records)
    labeled = unique["herg_blocker_label"].dropna().astype(int)
    audit = {
        "workbook": str(path.resolve()),
        "workbook_sha256": _sha256_file(path),
        "n_rows": int(len(rows)),
        "n_valid_structures": int((~invalid).sum()),
        "n_unique_structures": int(len(unique)),
        "n_duplicate_structure_rows": int(rows["standard_inchi_key"].duplicated(keep=False).sum()),
        "n_labeled_unique_structures": int(len(labeled)),
        "n_blockers": int((labeled == 1).sum()),
        "n_nonblockers": int((labeled == 0).sum()),
        "n_intermediate_rows": int((rows["herg_label_status"] == "intermediate").sum()),
        "n_missing_ic50_rows": int((rows["herg_label_status"] == "missing").sum()),
        "n_percent_inhibition_10um": int(rows["herg_inhibition_10um_pct"].notna().sum()),
        "n_percent_inhibition_30um": int(rows["herg_inhibition_30um_pct"].notna().sum()),
        "n_label_conflicts": int(unique["label_conflict"].sum()),
        "blocker_max_um": float(blocker_max_um),
        "nonblocker_min_um": float(nonblocker_min_um),
    }
    return rows, unique, audit


def load_public_herg(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    public = pd.read_csv(path, low_memory=False)
    endpoint = public["endpoint"].fillna("").astype(str).str.casefold()
    family = public["assay_family"].fillna("").astype(str).str.casefold()
    selected = public[
        endpoint.eq(PRIMARY_PUBLIC_ENDPOINT.casefold()) & family.eq(PRIMARY_PUBLIC_ASSAY_FAMILY.casefold())
    ].copy()
    selected["herg_blocker_label"] = pd.to_numeric(selected["herg_blocker_label"], errors="coerce")
    selected = selected[selected["herg_blocker_label"].isin([0, 1])].copy()
    selected["smiles"] = selected["smiles"].fillna("").astype(str)
    structures = selected["smiles"].map(_canonical_structure)
    selected["smiles"] = structures.map(lambda item: item[0])
    selected["computed_inchi_key"] = structures.map(lambda item: item[1])
    selected["scaffold"] = structures.map(lambda item: item[2])
    selected["standard_inchi_key"] = selected.get("standard_inchi_key", selected["computed_inchi_key"])
    selected["standard_inchi_key"] = selected["standard_inchi_key"].fillna(selected["computed_inchi_key"])
    selected = (
        selected[selected["smiles"].ne("")].drop_duplicates("standard_inchi_key").reset_index(drop=True)
    )
    selected["source"] = "public"
    audit = {
        "path": str(path.resolve()),
        "sha256": _sha256_file(path),
        "n_primary_labeled_structures": int(len(selected)),
        "n_blockers": int((selected["herg_blocker_label"] == 1).sum()),
        "n_nonblockers": int((selected["herg_blocker_label"] == 0).sum()),
        "endpoint": PRIMARY_PUBLIC_ENDPOINT,
        "assay_family": PRIMARY_PUBLIC_ASSAY_FAMILY,
    }
    return selected, audit


def _descriptor_frame(smiles_values: Iterable[str]) -> pd.DataFrame:
    names = sorted(Descriptors.CalcMolDescriptors(Chem.MolFromSmiles("CC"), silent=True))
    rows: list[dict[str, float]] = []
    for text in smiles_values:
        mol = Chem.MolFromSmiles(str(text))
        values = Descriptors.CalcMolDescriptors(mol, missingVal=np.nan, silent=True) if mol else {}
        rows.append({name: float(values.get(name, np.nan)) for name in names})
    # Missing descriptor values are replaced by a fixed value rather than a
    # median calculated from the complete modeling universe.  The latter is an
    # easy-to-miss source of outer-test leakage in nested validation.
    return pd.DataFrame(rows, columns=names, dtype=float).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _model_descriptor_matrix(descriptors: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply a deterministic, sample-independent numerical stabilization.

    ``Ipc`` is RDKit's information-content descriptor and can exceed float32
    range for large public structures.  A fixed signed-log transform is used
    for that descriptor only.  No quantile, median, or other statistic is
    calculated from validation/test structures, so a molecule's feature vector
    is invariant to which other molecules happen to be present.
    """

    values = descriptors.to_numpy(dtype=np.float64, copy=True)
    transformed_columns = [column for column in ("Ipc",) if column in descriptors.columns]
    for column in transformed_columns:
        column_index = descriptors.columns.get_loc(column)
        vector = values[:, column_index]
        values[:, column_index] = np.sign(vector) * np.log1p(np.abs(vector))
    values = np.nan_to_num(values, nan=0.0, posinf=1e12, neginf=-1e12)
    values = np.clip(values, -1e12, 1e12).astype(np.float32)
    if not np.isfinite(values).all():
        raise ValueError("Descriptor stabilization failed to produce a finite matrix")
    return values, {
        "method": "sample_independent_fixed_transform",
        "missing_value_fill": 0.0,
        "signed_log_columns": transformed_columns,
        "n_signed_log_columns": len(transformed_columns),
        "absolute_clip": 1e12,
    }


def _morgan_matrix(smiles_values: Sequence[str], *, n_bits: int, radius: int = 2) -> sparse.csr_matrix:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    rows: list[np.ndarray] = []
    for text in smiles_values:
        vector = np.zeros((n_bits,), dtype=np.float32)
        mol = Chem.MolFromSmiles(str(text))
        if mol is not None:
            DataStructs.ConvertToNumpyArray(generator.GetFingerprint(mol), vector)
        rows.append(vector)
    return sparse.csr_matrix(np.vstack(rows), dtype=np.float32)


def _maccs_matrix(smiles_values: Sequence[str]) -> sparse.csr_matrix:
    rows: list[np.ndarray] = []
    for text in smiles_values:
        vector = np.zeros((167,), dtype=np.float32)
        mol = Chem.MolFromSmiles(str(text))
        if mol is not None:
            DataStructs.ConvertToNumpyArray(MACCSkeys.GenMACCSKeys(mol), vector)
        rows.append(vector)
    return sparse.csr_matrix(np.vstack(rows), dtype=np.float32)


def calculate_feature_registry(
    smiles_values: Sequence[str],
) -> tuple[dict[str, sparse.csr_matrix | np.ndarray], pd.DataFrame, dict[str, Any]]:
    """Calculate every reusable structure representation exactly once."""

    descriptors = _descriptor_frame(smiles_values)
    descriptor_matrix, descriptor_transform = _model_descriptor_matrix(descriptors)
    morgan_1024 = _morgan_matrix(smiles_values, n_bits=1024, radius=2)
    morgan_2048 = _morgan_matrix(smiles_values, n_bits=2048, radius=2)
    maccs = _maccs_matrix(smiles_values)
    matrices: dict[str, sparse.csr_matrix | np.ndarray] = {
        "rdkit_2d_descriptors": descriptor_matrix,
        "maccs_167": maccs,
        "morgan_1024_r2": morgan_1024,
        "morgan_2048_r2": morgan_2048,
        "morgan_1024_plus_rdkit": sparse.hstack(
            [morgan_1024, sparse.csr_matrix(descriptor_matrix)], format="csr"
        ),
        "morgan_2048_plus_rdkit": sparse.hstack(
            [morgan_2048, sparse.csr_matrix(descriptor_matrix)], format="csr"
        ),
    }
    metadata: dict[str, Any] = {
        name: {
            "n_rows": int(matrix.shape[0]),
            "n_features": int(matrix.shape[1]),
            "sparse": bool(sparse.issparse(matrix)),
        }
        for name, matrix in matrices.items()
    }
    metadata["rdkit_2d_descriptors"]["feature_names"] = list(descriptors.columns)
    metadata["rdkit_2d_descriptors"]["modeling_transform"] = descriptor_transform
    return matrices, descriptors, metadata


def _complexity_label(index: int, total: int) -> str:
    if total <= 1:
        return "single"
    if index == 0:
        return "simple"
    if index == total - 1:
        return "complex"
    return "moderate"


def build_model_specs(config: Mapping[str, Any], profile: str) -> list[ModelSpec]:
    profile_config = config["profiles"][profile]
    default_features = tuple(profile_config["feature_sets"])
    specs: list[ModelSpec] = []
    for family, family_config in config["models"].items():
        if family not in profile_config["model_families"]:
            continue
        grid_definition = family_config.get(f"{profile}_grid", family_config.get("grid", {}))
        combinations = list(ParameterGrid(grid_definition)) if grid_definition else [{}]
        feature_sets = tuple(family_config.get("feature_sets", default_features))
        if family == "rnn":
            feature_sets = ("smiles_tokens",)
        for index, parameters in enumerate(combinations):
            specs.append(
                ModelSpec(
                    family=family,
                    complexity=_complexity_label(index, len(combinations)),
                    parameters=dict(parameters),
                    feature_sets=feature_sets,
                )
            )
    return specs


def _balanced_sample_weights(y: np.ndarray, source_weights: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=int)
    counts = np.bincount(y, minlength=2).astype(float)
    class_weights = np.divide(len(y), 2.0 * counts, out=np.ones(2), where=counts > 0)
    weights = class_weights[y] * np.asarray(source_weights, dtype=float)
    return weights / max(float(np.mean(weights)), 1e-12)


def _make_estimator(spec: ModelSpec, *, random_state: int, n_features: int) -> tuple[BaseEstimator, str]:
    params = dict(spec.parameters)
    family = spec.family
    if family == "dummy":
        return DummyClassifier(strategy=params.get("strategy", "prior")), "model"
    if family == "logistic":
        model = LogisticRegression(
            C=float(params.get("C", 1.0)),
            max_iter=int(params.get("max_iter", 4000)),
            solver=str(params.get("solver", "liblinear")),
            random_state=random_state,
        )
        return Pipeline([("scale", MaxAbsScaler()), ("model", model)]), "model"
    if family == "random_forest":
        model = RandomForestClassifier(
            n_estimators=int(params.get("n_estimators", 300)),
            max_depth=params.get("max_depth"),
            min_samples_leaf=int(params.get("min_samples_leaf", 1)),
            max_features=params.get("max_features", "sqrt"),
            n_jobs=-1,
            random_state=random_state,
        )
        return model, "model"
    if family == "extra_trees":
        model = ExtraTreesClassifier(
            n_estimators=int(params.get("n_estimators", 600)),
            max_depth=params.get("max_depth"),
            min_samples_leaf=int(params.get("min_samples_leaf", 1)),
            max_features=params.get("max_features", "sqrt"),
            bootstrap=bool(params.get("bootstrap", False)),
            n_jobs=-1,
            random_state=random_state,
        )
        return model, "model"
    if family == "svm":
        model = SVC(
            C=float(params.get("C", 1.0)),
            kernel=str(params.get("kernel", "rbf")),
            gamma=params.get("gamma", "scale"),
            probability=True,
            cache_size=2048,
            random_state=random_state,
        )
        return Pipeline([("scale", MaxAbsScaler()), ("model", model)]), "model"
    if family == "knn":
        model = KNeighborsClassifier(
            n_neighbors=int(params.get("n_neighbors", 7)),
            weights=str(params.get("weights", "distance")),
            metric=str(params.get("metric", "cosine")),
            algorithm="brute",
            n_jobs=-1,
        )
        return Pipeline([("scale", MaxAbsScaler()), ("model", model)]), "none"
    if family == "xgboost":
        from xgboost import XGBClassifier

        model = XGBClassifier(
            n_estimators=int(params.get("n_estimators", 250)),
            max_depth=int(params.get("max_depth", 4)),
            learning_rate=float(params.get("learning_rate", 0.08)),
            min_child_weight=float(params.get("min_child_weight", 1.0)),
            subsample=float(params.get("subsample", 0.85)),
            colsample_bytree=float(params.get("colsample_bytree", 0.8)),
            reg_lambda=float(params.get("reg_lambda", 1.0)),
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=-1,
            random_state=random_state,
        )
        return model, "model"
    if family == "lightgbm":
        from lightgbm import LGBMClassifier

        model = LGBMClassifier(
            n_estimators=int(params.get("n_estimators", 250)),
            max_depth=int(params.get("max_depth", -1)),
            num_leaves=int(params.get("num_leaves", 31)),
            learning_rate=float(params.get("learning_rate", 0.05)),
            min_child_samples=int(params.get("min_child_samples", 20)),
            subsample=float(params.get("subsample", 0.85)),
            colsample_bytree=float(params.get("colsample_bytree", 0.8)),
            reg_lambda=float(params.get("reg_lambda", 1.0)),
            verbosity=-1,
            n_jobs=-1,
            random_state=random_state,
        )
        return model, "model"
    if family == "clustering":
        components = min(int(params.get("svd_components", 32)), max(2, n_features - 1))
        clusters = int(params.get("n_clusters", 8))
        model = Pipeline(
            [
                ("svd", TruncatedSVD(n_components=components, random_state=random_state)),
                (
                    "clusters",
                    MiniBatchKMeans(
                        n_clusters=clusters,
                        batch_size=256,
                        n_init=10,
                        random_state=random_state,
                    ),
                ),
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=float(params.get("C", 1.0)),
                        max_iter=4000,
                        solver="liblinear",
                        random_state=random_state,
                    ),
                ),
            ]
        )
        return model, "model"
    raise ValueError(f"Unsupported model family: {family}")


def _torch_modules():
    try:
        import torch
        from torch import nn

        return torch, nn
    except ImportError as exc:  # pragma: no cover - optional dependency path
        raise RuntimeError("PyTorch is required for the RNN model family") from exc


class SmilesRNNBundle:
    """Small serializable holder for a trained character-level GRU/LSTM."""

    def __init__(self, *, state: dict[str, Any], vocabulary: dict[str, int], parameters: dict[str, Any]):
        self.state = state
        self.vocabulary = vocabulary
        self.parameters = parameters


def _rnn_network(vocabulary_size: int, parameters: Mapping[str, Any]):
    torch, nn = _torch_modules()
    cell = str(parameters.get("cell", "gru")).lower()

    class Network(nn.Module):  # type: ignore[name-defined]
        def __init__(self):
            super().__init__()
            embedding_dim = int(parameters.get("embedding_dim", 24))
            hidden_dim = int(parameters.get("hidden_dim", 48))
            layers = int(parameters.get("layers", 1))
            dropout = float(parameters.get("dropout", 0.0)) if layers > 1 else 0.0
            self.embedding = nn.Embedding(vocabulary_size, embedding_dim, padding_idx=0)
            recurrent = nn.LSTM if cell == "lstm" else nn.GRU
            self.rnn = recurrent(
                embedding_dim,
                hidden_dim,
                num_layers=layers,
                batch_first=True,
                dropout=dropout,
            )
            self.output = nn.Linear(hidden_dim, 1)

        def forward(self, tokens, lengths):
            embedded = self.embedding(tokens)
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded,
                lengths.cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            _, hidden = self.rnn(packed)
            if isinstance(hidden, tuple):
                hidden = hidden[0]
            return self.output(hidden[-1]).squeeze(1)

    return Network(), torch


def _encode_smiles(
    smiles_values: Sequence[str],
    vocabulary: Mapping[str, int],
    *,
    max_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    encoded = np.zeros((len(smiles_values), max_length), dtype=np.int64)
    lengths = np.ones(len(smiles_values), dtype=np.int64)
    unknown = int(vocabulary.get("<UNK>", 1))
    for row_index, text in enumerate(smiles_values):
        token_ids = [int(vocabulary.get(char, unknown)) for char in str(text)[:max_length]]
        lengths[row_index] = max(1, len(token_ids))
        if token_ids:
            encoded[row_index, : len(token_ids)] = token_ids
    return encoded, lengths


def _fit_rnn(
    smiles_values: Sequence[str],
    y: np.ndarray,
    sample_weight: np.ndarray,
    parameters: Mapping[str, Any],
    *,
    random_state: int,
) -> FittedModel:
    torch, nn = _torch_modules()
    torch.set_num_threads(1)
    torch.manual_seed(random_state)
    np.random.seed(random_state)
    characters = sorted(set("".join(map(str, smiles_values))))
    vocabulary = {"<PAD>": 0, "<UNK>": 1, **{char: i + 2 for i, char in enumerate(characters)}}
    max_length = int(parameters.get("max_length", 180))
    tokens, lengths = _encode_smiles(smiles_values, vocabulary, max_length=max_length)
    network, _ = _rnn_network(len(vocabulary), parameters)
    optimizer = torch.optim.AdamW(
        network.parameters(),
        lr=float(parameters.get("learning_rate", 0.002)),
        weight_decay=float(parameters.get("weight_decay", 1e-4)),
    )
    loss_function = nn.BCEWithLogitsLoss(reduction="none")
    batch_size = int(parameters.get("batch_size", 64))
    epochs = int(parameters.get("epochs", 10))
    generator = torch.Generator().manual_seed(random_state)
    dataset = torch.utils.data.TensorDataset(
        torch.as_tensor(tokens),
        torch.as_tensor(lengths),
        torch.as_tensor(np.asarray(y, dtype=np.float32)),
        torch.as_tensor(np.asarray(sample_weight, dtype=np.float32)),
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    network.train()
    for _ in range(epochs):
        for batch_tokens, batch_lengths, batch_y, batch_weight in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = network(batch_tokens, batch_lengths)
            loss = (loss_function(logits, batch_y) * batch_weight).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(network.parameters(), max_norm=5.0)
            optimizer.step()
    fitted_parameters = dict(parameters)
    fitted_parameters["max_length"] = max_length
    bundle = SmilesRNNBundle(
        state={key: value.detach().cpu() for key, value in network.state_dict().items()},
        vocabulary=vocabulary,
        parameters=fitted_parameters,
    )
    return FittedModel(family="rnn", estimator=bundle, parameters=fitted_parameters)


def _fit_model(
    spec: ModelSpec,
    X: sparse.csr_matrix | np.ndarray | Sequence[str],
    y: np.ndarray,
    source_weights: np.ndarray,
    *,
    random_state: int,
) -> FittedModel:
    weights = _balanced_sample_weights(y, source_weights)
    if spec.family == "rnn":
        return _fit_rnn(list(X), y, weights, spec.parameters, random_state=random_state)
    matrix = cast(sparse.csr_matrix | np.ndarray, X)
    estimator, weight_step = _make_estimator(
        spec,
        random_state=random_state,
        n_features=int(matrix.shape[1]),
    )
    if weight_step == "none":
        repeat_counts = np.clip(np.rint(weights / max(float(np.min(weights)), 1e-6)), 1, 12).astype(int)
        repeated = np.repeat(np.arange(len(y)), repeat_counts)
        estimator.fit(X[repeated], y[repeated])
    elif isinstance(estimator, Pipeline):
        estimator.fit(X, y, **{f"{weight_step}__sample_weight": weights})
    else:
        estimator.fit(X, y, sample_weight=weights)
    return FittedModel(family=spec.family, estimator=estimator, parameters=dict(spec.parameters))


def _predict_model(
    fitted: FittedModel,
    X: sparse.csr_matrix | np.ndarray | Sequence[str],
) -> np.ndarray:
    if fitted.family != "rnn":
        return np.asarray(fitted.estimator.predict_proba(X)[:, 1], dtype=float)
    torch, _ = _torch_modules()
    bundle: SmilesRNNBundle = fitted.estimator
    tokens, lengths = _encode_smiles(
        list(X),
        bundle.vocabulary,
        max_length=int(bundle.parameters["max_length"]),
    )
    network, _ = _rnn_network(len(bundle.vocabulary), bundle.parameters)
    network.load_state_dict(bundle.state)
    network.eval()
    probabilities: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(tokens), 256):
            logits = network(
                torch.as_tensor(tokens[start : start + 256]),
                torch.as_tensor(lengths[start : start + 256]),
            )
            probabilities.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probabilities).astype(float)


def _classification_metrics(
    y_true: np.ndarray, probability: np.ndarray, threshold: float = 0.5
) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=int)
    probability = np.clip(np.asarray(probability, dtype=float), 1e-7, 1 - 1e-7)
    prediction = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    both_classes = len(np.unique(y_true)) == 2
    return {
        "roc_auc": float(roc_auc_score(y_true, probability)) if both_classes else np.nan,
        "pr_auc": float(average_precision_score(y_true, probability)) if both_classes else np.nan,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "mcc": float(matthews_corrcoef(y_true, prediction)),
        "sensitivity": float(recall_score(y_true, prediction, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else np.nan,
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "brier": float(brier_score_loss(y_true, probability)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def _selection_score(metrics: Mapping[str, Any]) -> float:
    roc = float(metrics.get("roc_auc", np.nan))
    balanced = float(metrics.get("balanced_accuracy", np.nan))
    mcc = float(metrics.get("mcc", np.nan))
    brier = float(metrics.get("brier", np.nan))
    if not all(np.isfinite([roc, balanced, mcc, brier])):
        return -np.inf
    return float(0.40 * roc + 0.30 * balanced + 0.15 * ((mcc + 1) / 2) + 0.15 * (1 - brier))


def _private_folds(
    private_labeled: pd.DataFrame,
    *,
    n_splits: int,
    n_repeats: int,
    random_state: int,
    strategy: str,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], str]:
    y = private_labeled["herg_blocker_label"].to_numpy(dtype=int)
    if strategy == "scaffold":
        all_folds: list[tuple[np.ndarray, np.ndarray]] = []
        viable = True
        for repeat in range(n_repeats):
            splitter = StratifiedGroupKFold(
                n_splits=n_splits, shuffle=True, random_state=random_state + repeat
            )
            repeat_folds = list(splitter.split(np.zeros(len(y)), y, groups=private_labeled["scaffold"]))
            if any(
                len(np.unique(y[test])) < 2 or len(np.unique(y[train])) < 2 for train, test in repeat_folds
            ):
                viable = False
                break
            all_folds.extend(repeat_folds)
        if viable:
            return all_folds, "scaffold"
        warnings.warn(
            "Scaffold folds lacked both classes; falling back to repeated stratified folds.",
            stacklevel=2,
        )
    splitter = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state,
    )
    return list(splitter.split(np.zeros(len(y)), y)), "stratified_structure"


def _regime_training_indices(
    regime: str,
    *,
    public_indices: np.ndarray,
    private_train_indices: np.ndarray,
    private_weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    if regime == "confidential_only":
        return private_train_indices, np.ones(len(private_train_indices), dtype=float)
    indices = np.concatenate([public_indices, private_train_indices])
    private_source_weight = private_weight if regime == "confidential_prioritized" else 1.0
    source_weights = np.concatenate(
        [
            np.ones(len(public_indices), dtype=float),
            np.full(len(private_train_indices), private_source_weight),
        ]
    )
    return indices, source_weights


def _feature_slice(
    feature_set: str,
    matrices: Mapping[str, sparse.csr_matrix | np.ndarray],
    smiles_values: np.ndarray,
    indices: np.ndarray,
):
    if feature_set == "smiles_tokens":
        return smiles_values[indices].tolist()
    return matrices[feature_set][indices]


def _save_model(path: Path, fitted: FittedModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fitted.family != "rnn":
        joblib.dump(fitted, path.with_suffix(".joblib"), compress=3)
        return
    torch, _ = _torch_modules()
    bundle: SmilesRNNBundle = fitted.estimator
    torch.save(
        {
            "family": "rnn",
            "state_dict": bundle.state,
            "vocabulary": bundle.vocabulary,
            "parameters": bundle.parameters,
        },
        path.with_suffix(".pt"),
    )


def _plot_results(cv_results: pd.DataFrame, predictions: pd.DataFrame, output_dir: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    top = (
        cv_results.sort_values("selection_score", ascending=False)
        .groupby(["regime", "family"], as_index=False)
        .head(1)
    )
    pivot = top.pivot(index="family", columns="regime", values="selection_score")
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis", ax=ax)
    ax.set_title("Best hERG model score by family and training regime")
    fig.tight_layout()
    path = figure_dir / "model_family_regime_heatmap.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path))

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(
        data=predictions,
        x="ensemble_probability",
        hue="regime",
        bins=15,
        element="step",
        stat="count",
        common_norm=False,
        ax=ax,
    )
    ax.axvline(0.3, color="grey", linestyle="--", linewidth=1)
    ax.axvline(0.7, color="grey", linestyle="--", linewidth=1)
    ax.set_title("Predicted hERG blocker probabilities for all lab compounds")
    fig.tight_layout()
    path = figure_dir / "private_prediction_distributions.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path))
    return paths


def _write_summary(
    path: Path,
    *,
    audit: Mapping[str, Any],
    public_audit: Mapping[str, Any],
    overlap: int,
    cv_results: pd.DataFrame,
    predictions: pd.DataFrame,
    profile: str,
) -> None:
    winners = (
        cv_results.sort_values("selection_score", ascending=False).groupby("regime", as_index=False).head(1)
    )
    lines = [
        "# hERG liability benchmark results",
        "",
        f"Profile: **{profile}**",
        "",
        "## Data audit",
        "",
        f"- Confidential workbook: {audit['n_rows']} rows / {audit['n_unique_structures']} unique valid structures.",
        f"- Decisive confidential labels: {audit['n_labeled_unique_structures']} "
        f"({audit['n_blockers']} blockers, {audit['n_nonblockers']} nonblockers).",
        f"- Intermediate IC50 rows excluded from binary training: {audit['n_intermediate_rows']}.",
        f"- Public primary set: {public_audit['n_primary_labeled_structures']} labeled structures.",
        f"- Exact private/public structural overlap removed: {overlap}.",
        "",
        "## Best cross-validated private-domain models",
        "",
        "| Regime | Family | Complexity | Features | ROC AUC | Balanced accuracy | MCC | Brier | Score |",
        "|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in winners.iterrows():
        lines.append(
            f"| {row['regime']} | {row['family']} | {row['complexity']} | {row['feature_set']} | "
            f"{row['roc_auc']:.3f} | {row['balanced_accuracy']:.3f} | {row['mcc']:.3f} | "
            f"{row['brier']:.3f} | {row['selection_score']:.3f} |"
        )
    lines.extend(["", "## Production scoring", ""])
    for regime, group in predictions.groupby("regime"):
        counts = group["risk_band"].value_counts()
        lines.append(
            f"- {regime}: {int(counts.get('high', 0))} high, {int(counts.get('medium', 0))} medium, "
            f"{int(counts.get('low', 0))} low predicted-risk rows."
        )
    lines.extend(
        [
            "",
            "All performance values above come from held-out confidential compounds. The final production ensembles were then refit on all eligible labels. The intermediate and missing-IC50 compounds were scored but never converted into training labels.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_benchmark(
    *,
    workbook_path: Path,
    public_path: Path,
    config_path: Path,
    output_dir: Path,
    profile: str = "quick",
) -> dict[str, Any]:
    started = time.time()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if profile not in config["profiles"]:
        raise KeyError(f"Unknown benchmark profile: {profile}")
    run_config = config["run"]
    seed = int(run_config.get("random_state", 13))
    np.random.seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    private_rows, private_unique, private_audit = load_private_workbook(
        workbook_path,
        blocker_max_um=float(run_config.get("blocker_max_um", 10.0)),
        nonblocker_min_um=float(run_config.get("nonblocker_min_um", 30.0)),
    )
    public, public_audit = load_public_herg(public_path)
    private_keys = set(private_unique["standard_inchi_key"])
    overlap = int(public["standard_inchi_key"].isin(private_keys).sum())
    public = public[~public["standard_inchi_key"].isin(private_keys)].reset_index(drop=True)

    private_unique = private_unique.reset_index(drop=True)
    private_labeled_local = np.flatnonzero(private_unique["herg_blocker_label"].notna().to_numpy())
    universe = pd.concat(
        [
            public[["standard_inchi_key", "smiles", "scaffold", "herg_blocker_label", "source"]],
            private_unique[["standard_inchi_key", "smiles", "scaffold", "herg_blocker_label", "source"]],
        ],
        ignore_index=True,
    )
    n_public = len(public)
    public_indices = np.arange(n_public, dtype=int)
    private_indices = n_public + np.arange(len(private_unique), dtype=int)
    private_labeled_indices = private_indices[private_labeled_local]
    smiles_values = universe["smiles"].to_numpy(dtype=object)
    labels = pd.to_numeric(universe["herg_blocker_label"], errors="coerce").to_numpy(dtype=float)

    print(f"[features] calculating descriptors and fingerprints for {len(universe):,} structures", flush=True)
    matrices, descriptors, feature_metadata = calculate_feature_registry(smiles_values.tolist())
    private_descriptor_rows = descriptors.iloc[private_indices].reset_index(drop=True)
    calculated = private_rows.merge(
        private_unique[["standard_inchi_key"]]
        .reset_index()
        .rename(columns={"index": "private_structure_index"}),
        on="standard_inchi_key",
        how="left",
    )
    calculated = pd.concat(
        [
            calculated.reset_index(drop=True),
            private_descriptor_rows.iloc[
                calculated["private_structure_index"].fillna(0).astype(int).to_numpy()
            ]
            .reset_index(drop=True)
            .add_prefix("rdkit_"),
        ],
        axis=1,
    )
    calculated.to_csv(output_dir / "calculated_molecular_parameters.csv", index=False)
    _write_json(output_dir / "feature_registry.json", feature_metadata)

    specs = build_model_specs(config, profile)
    grid_rows: list[dict[str, Any]] = []
    for spec in specs:
        for feature_set in spec.feature_sets:
            grid_rows.append(
                {
                    "model_key": spec.key,
                    "family": spec.family,
                    "complexity": spec.complexity,
                    "feature_set": feature_set,
                    "parameters_json": json.dumps(spec.parameters, sort_keys=True),
                }
            )
    parameter_grid = pd.DataFrame(grid_rows)
    parameter_grid.to_csv(output_dir / "model_parameter_grid.csv", index=False)

    private_labeled = private_unique.iloc[private_labeled_local].reset_index(drop=True)
    folds, resolved_split = _private_folds(
        private_labeled,
        n_splits=int(config["profiles"][profile]["cv_folds"]),
        n_repeats=int(config["profiles"][profile].get("cv_repeats", 1)),
        random_state=seed,
        strategy=str(config["profiles"][profile].get("split_strategy", "stratified")),
    )
    print(
        f"[search] {len(parameter_grid)} parameter/feature combinations x {len(REGIMES)} regimes x "
        f"{len(folds)} folds ({resolved_split})",
        flush=True,
    )
    fold_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    total_jobs = len(parameter_grid) * len(REGIMES)
    completed_jobs = 0
    private_weight = float(run_config.get("confidential_priority_weight", 5.0))
    for regime in REGIMES:
        for spec in specs:
            for feature_set in spec.feature_sets:
                for fold_index, (private_train_local, private_test_local) in enumerate(folds):
                    private_train_global = private_labeled_indices[private_train_local]
                    private_test_global = private_labeled_indices[private_test_local]
                    train_indices, source_weights = _regime_training_indices(
                        regime,
                        public_indices=public_indices,
                        private_train_indices=private_train_global,
                        private_weight=private_weight,
                    )
                    y_train = labels[train_indices].astype(int)
                    y_test = labels[private_test_global].astype(int)
                    X_train = _feature_slice(feature_set, matrices, smiles_values, train_indices)
                    X_test = _feature_slice(feature_set, matrices, smiles_values, private_test_global)
                    row_base = {
                        "regime": regime,
                        "model_key": spec.key,
                        "family": spec.family,
                        "complexity": spec.complexity,
                        "feature_set": feature_set,
                        "fold": fold_index,
                        "n_train": len(train_indices),
                        "n_test_private": len(private_test_global),
                        "parameters_json": json.dumps(spec.parameters, sort_keys=True),
                    }
                    try:
                        fitted = _fit_model(
                            spec,
                            X_train,
                            y_train,
                            source_weights,
                            random_state=seed + fold_index,
                        )
                        probability = _predict_model(fitted, X_test)
                        metrics = _classification_metrics(y_test, probability)
                        fold_rows.append({**row_base, "status": "ok", "error": "", **metrics})
                        for local_position, global_position, observed, score in zip(
                            private_test_local,
                            private_test_global,
                            y_test,
                            probability,
                            strict=True,
                        ):
                            prediction_rows.append(
                                {
                                    **{
                                        key: row_base[key]
                                        for key in (
                                            "regime",
                                            "model_key",
                                            "family",
                                            "complexity",
                                            "feature_set",
                                            "fold",
                                        )
                                    },
                                    "private_labeled_position": int(local_position),
                                    "universe_position": int(global_position),
                                    "standard_inchi_key": universe.iloc[global_position][
                                        "standard_inchi_key"
                                    ],
                                    "observed_label": int(observed),
                                    "probability": float(score),
                                }
                            )
                    except Exception as exc:  # retain failed combinations in the audit table
                        fold_rows.append(
                            {
                                **row_base,
                                "status": "failed",
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                        )
                completed_jobs += 1
                if completed_jobs % 10 == 0 or completed_jobs == total_jobs:
                    pd.DataFrame(fold_rows).to_csv(output_dir / "fold_results.checkpoint.csv", index=False)
                    pd.DataFrame(prediction_rows).to_csv(
                        output_dir / "oof_private_predictions.checkpoint.csv", index=False
                    )
                    print(f"[search] completed {completed_jobs}/{total_jobs} combinations", flush=True)

    fold_results = pd.DataFrame(fold_rows)
    fold_results.to_csv(output_dir / "fold_results.csv", index=False)
    oof_predictions = pd.DataFrame(prediction_rows)
    oof_predictions.to_csv(output_dir / "oof_private_predictions.csv", index=False)
    (output_dir / "fold_results.checkpoint.csv").unlink(missing_ok=True)
    (output_dir / "oof_private_predictions.checkpoint.csv").unlink(missing_ok=True)
    result_rows: list[dict[str, Any]] = []
    for keys, group in oof_predictions.groupby(
        ["regime", "model_key", "family", "complexity", "feature_set"], sort=False
    ):
        metrics = _classification_metrics(
            group["observed_label"].to_numpy(dtype=int),
            group["probability"].to_numpy(dtype=float),
        )
        fold_group = fold_results[
            (fold_results["regime"] == keys[0])
            & (fold_results["model_key"] == keys[1])
            & (fold_results["feature_set"] == keys[4])
        ]
        result_rows.append(
            {
                "regime": keys[0],
                "model_key": keys[1],
                "family": keys[2],
                "complexity": keys[3],
                "feature_set": keys[4],
                "parameters_json": fold_group["parameters_json"].iloc[0],
                "n_oof_predictions": int(len(group)),
                "n_successful_folds": int((fold_group["status"] == "ok").sum()),
                **metrics,
                "selection_score": _selection_score(metrics),
            }
        )
    cv_results = pd.DataFrame(result_rows).sort_values(["regime", "selection_score"], ascending=[True, False])
    cv_results.to_csv(output_dir / "cv_results.csv", index=False)
    failed = fold_results[fold_results["status"] != "ok"]
    failed.to_csv(output_dir / "failed_fits.csv", index=False)

    print("[refit] fitting top diverse ensembles and scoring every confidential compound", flush=True)
    private_mapping = private_unique.reset_index().set_index("standard_inchi_key")["index"]
    production_rows: list[pd.DataFrame] = []
    best_rows: list[dict[str, Any]] = []
    ensemble_size = int(config["profiles"][profile].get("production_ensemble_size", 3))
    for regime in REGIMES:
        ranked = cv_results[cv_results["regime"] == regime].sort_values("selection_score", ascending=False)
        selected_rows = ranked.drop_duplicates("family").head(ensemble_size)
        probability_columns: list[str] = []
        regime_output = private_rows[
            [
                "private_row_number",
                PRIVATE_ID_COLUMN,
                "smiles",
                "standard_inchi_key",
                PRIVATE_HERG_IC50_COLUMN,
                PRIVATE_HERG_PERCENT_COLUMN,
                "herg_blocker_label",
                "herg_label_status",
            ]
        ].copy()
        for rank, (_, selected) in enumerate(selected_rows.iterrows(), start=1):
            matching_spec = next(spec for spec in specs if spec.key == selected["model_key"])
            feature_set = str(selected["feature_set"])
            train_indices, source_weights = _regime_training_indices(
                regime,
                public_indices=public_indices,
                private_train_indices=private_labeled_indices,
                private_weight=private_weight,
            )
            fitted = _fit_model(
                matching_spec,
                _feature_slice(feature_set, matrices, smiles_values, train_indices),
                labels[train_indices].astype(int),
                source_weights,
                random_state=seed + 1000 + rank,
            )
            private_all_probability = _predict_model(
                fitted,
                _feature_slice(feature_set, matrices, smiles_values, private_indices),
            )
            row_positions = private_rows["standard_inchi_key"].map(private_mapping).to_numpy(dtype=float)
            valid_positions = np.isfinite(row_positions)
            row_probability = np.full(len(private_rows), np.nan, dtype=float)
            row_probability[valid_positions] = private_all_probability[
                row_positions[valid_positions].astype(int)
            ]
            column = f"model_{rank}_{selected['family']}_probability"
            regime_output[column] = row_probability
            probability_columns.append(column)
            best_rows.append(
                {
                    "regime": regime,
                    "ensemble_rank": rank,
                    **selected.to_dict(),
                }
            )
            _save_model(output_dir / "models" / f"{regime}__rank{rank}__{selected['model_key']}", fitted)
        regime_output["ensemble_probability"] = regime_output[probability_columns].mean(axis=1)
        regime_output["ensemble_std"] = regime_output[probability_columns].std(axis=1, ddof=0)
        regime_output["risk_band"] = pd.cut(
            regime_output["ensemble_probability"],
            bins=[-np.inf, 0.30, 0.70, np.inf],
            labels=["low", "medium", "high"],
            right=False,
        ).astype(str)
        regime_output.insert(0, "regime", regime)
        production_rows.append(regime_output)
    production_predictions = pd.concat(production_rows, ignore_index=True)
    production_predictions.to_csv(output_dir / "private_compound_predictions.csv", index=False)
    best_models = pd.DataFrame(best_rows)
    best_models.to_csv(output_dir / "best_models.csv", index=False)
    figures = _plot_results(cv_results, production_predictions, output_dir)
    _write_summary(
        output_dir / "results_summary.md",
        audit=private_audit,
        public_audit=public_audit,
        overlap=overlap,
        cv_results=cv_results,
        predictions=production_predictions,
        profile=profile,
    )

    manifest: dict[str, Any] = {
        "status": "complete",
        "profile": profile,
        "runtime_seconds": time.time() - started,
        "random_state": seed,
        "resolved_split_strategy": resolved_split,
        "regimes": list(REGIMES),
        "confidential_priority_weight": private_weight,
        "private_audit": private_audit,
        "public_audit": public_audit,
        "private_public_exact_overlap_removed": overlap,
        "n_parameter_feature_combinations": int(len(parameter_grid)),
        "n_regime_combinations": int(len(parameter_grid) * len(REGIMES)),
        "n_folds": int(len(folds)),
        "n_successful_fits": int((fold_results["status"] == "ok").sum()),
        "n_failed_fits": int((fold_results["status"] != "ok").sum()),
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "rdkit": rdBase.rdkitVersion,
        },
        "feature_registry": feature_metadata,
        "figures": figures,
        "output_files": sorted(path.name for path in output_dir.iterdir() if path.is_file()),
    }
    for package in ("xgboost", "lightgbm", "torch", "openpyxl"):
        try:
            module = __import__(package)
            manifest["software"][package] = getattr(module, "__version__", "installed")
        except ImportError:
            manifest["software"][package] = "not_installed"
    _write_json(output_dir / "run_manifest.json", manifest)
    print(f"[complete] results written to {output_dir.resolve()}", flush=True)
    return manifest
