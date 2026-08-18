#!/usr/bin/env python3
"""Extended, resumable, train-only hERG discovery worker.

This worker extends the completed v1 campaign without changing it.  Repository
validation and test labels are forbidden inputs.  ``outer`` and ``inner`` folds
are scaffold folds made solely inside the repository training partition.

The broad fixed-dose endpoint is an auxiliary binary task.  It is never pooled
with exact or censored pIC50.  Feature-response and MMP results are exploratory
hypotheses, not causal or mechanistic proof.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
import time
from collections import Counter
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SCHEMA = "platform-local-herg-discovery-worker-v3/1.0"
SEED = 20260812
MAX_ABS = 1.0e30
EXACT_ROWS = 18_801
BASE_REQUIRED = (
    "prepared/exact_train_cache.parquet",
    "prepared/nested_scaffold_splits.parquet",
    "prepared/feature_registry.json",
    "analysis/validation.json",
)
OPERATIONS = {
    "repeated_seed_tree",
    "expanded_hpo_tree",
    "nested_robustness",
    "feature_ablation",
    "interaction_stability",
    "assay_quality_strata",
    "uncertainty_calibration",
    "censored_sensitivity",
    "mmp_cliff_residual",
    "chemprop_ensemble",
    "broad_wt_auxiliary",
    "finalist_refit_artifact",
}
REGRESSION_OPERATIONS = {
    "repeated_seed_tree",
    "expanded_hpo_tree",
    "nested_robustness",
    "feature_ablation",
    "interaction_stability",
}
DEFAULT_BROAD = Path(
    "research/data/platform/processed/herg_hierarchy/v1_6_training_surfaces/"
    "confirmed_wt_fixed_dose_structure_labels.parquet"
)
DEFAULT_OBSERVATIONS = Path(
    "research/data/platform/processed/herg_hierarchy/v1_6_training_surfaces/"
    "herg_training_observations.parquet"
)
DEFAULT_MMP = Path(
    "research/data/platform/processed/herg_hierarchy/v1_5_mmp_analysis/training_mmp_effects.parquet"
)
DEFAULT_GLOBAL_2D = Path("research/local_runs/local_multicpu_2d_features_v1")


class WorkerError(RuntimeError):
    """Scientific or file-contract failure."""


class Unavailable(WorkerError):
    """Optional operation cannot run with the available local inputs."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _spec_hash(spec: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical(spec)).hexdigest()


def _atomic(path: Path, writer: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        writer(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json(path: Path, value: dict[str, Any], hash_key: str) -> dict[str, Any]:
    document = dict(value)
    document.pop(hash_key, None)
    document[hash_key] = hashlib.sha256(_canonical(document)).hexdigest()
    _atomic(path, lambda temporary: temporary.write_bytes(_canonical(document)))
    return document


def _read_self_json(path: Path, hash_key: str) -> dict[str, Any]:
    document = json.loads(path.read_text("utf-8"))
    expected = document.pop(hash_key, None)
    actual = hashlib.sha256(_canonical(document)).hexdigest()
    document[hash_key] = expected
    if not isinstance(expected, str) or expected != actual:
        raise WorkerError(f"self-hash mismatch: {path}")
    return document


def _write_parquet(path: Path, frame: pd.DataFrame) -> None:
    _atomic(path, lambda temporary: frame.to_parquet(temporary, index=False, compression="zstd"))


def _binding(path: Path, role: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "role": role,
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _sha(path),
    }
    if path.suffix == ".parquet":
        metadata = pq.read_metadata(path)
        result.update(rows=metadata.num_rows, columns=metadata.num_columns)
    return result


def _verify_binding(binding: dict[str, Any]) -> None:
    path = Path(str(binding["path"]))
    if not path.is_file() or path.stat().st_size != int(binding["bytes"]):
        raise WorkerError(f"bound artifact missing or changed size: {path}")
    if _sha(path) != str(binding["sha256"]):
        raise WorkerError(f"bound artifact changed hash: {path}")
    if path.suffix == ".parquet" and pq.read_metadata(path).num_rows != int(binding["rows"]):
        raise WorkerError(f"bound artifact changed rows: {path}")


def _scope(endpoint: str) -> dict[str, Any]:
    target_scope = (
        "confirmed wild-type hERG" if "confirmed-WT" in endpoint else "wild-type-or-unspecified hERG"
    )
    return {
        "source_partition": "train",
        "repository_validation_labels_opened": False,
        "repository_test_labels_opened": False,
        "target_scope": target_scope,
        "endpoint": endpoint,
        "broad_fixed_dose_pooled_into_pic50": False,
        "causal_interpretation_allowed": False,
    }


def _safe_id(value: str) -> str:
    if not value or any(
        character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_."
        for character in value
    ):
        raise WorkerError("unsafe or empty unit id")
    return value


def _unit_dir(results_root: Path, unit_id: str) -> Path:
    return results_root / "units" / _safe_id(unit_id)


def _existing_unit(path: Path, spec: dict[str, Any]) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        document = _read_self_json(path, "unit_json_sha256")
        if document.get("status") != "passed" or document.get("unit_spec_sha256") != _spec_hash(spec):
            return None
        for artifact in document.get("artifacts", []):
            _verify_binding(artifact)
        return document
    except Exception:
        return None


def _write_unit(
    directory: Path,
    *,
    unit_id: str,
    operation: str,
    spec: dict[str, Any],
    metrics: dict[str, Any],
    artifacts: list[dict[str, Any]],
    status: str = "passed",
    limitations: list[str] | None = None,
) -> dict[str, Any]:
    candidate = spec.get("candidate", {})
    models = [item for item in artifacts if item["role"].startswith("model")]
    executed_spec = {
        key: spec[key]
        for key in sorted(spec)
        if key
        not in {
            "repository_validation_labels_opened",
            "repository_test_labels_opened",
            "broad_fixed_dose_pooled_into_pic50",
            "source_partition",
            "task",
        }
    }
    document = {
        "schema_version": SCHEMA,
        "created_at": _now(),
        "status": status,
        "operation": operation,
        "unit_id": unit_id,
        "unit_spec": spec,
        "unit_spec_sha256": _spec_hash(spec),
        "executed_spec": executed_spec,
        "executed_spec_sha256": _spec_hash(executed_spec),
        "candidate_id": candidate.get("candidate_id", spec.get("candidate_id", unit_id)),
        "metrics": metrics,
        "model_artifacts": models,
        "artifacts": artifacts,
        "scientific_scope": _scope(
            "confirmed-WT fixed-dose auxiliary classification"
            if operation == "broad_wt_auxiliary"
            else "exact quantitative pIC50"
            if operation != "censored_sensitivity"
            else "censored quantitative sensitivity"
        ),
        "limitations": limitations or [],
    }
    return _write_json(directory / "unit.json", document, "unit_json_sha256")


def _candidate(spec: dict[str, Any]) -> dict[str, Any]:
    candidate = dict(spec.get("resolved_candidate") or spec.get("candidate") or {})
    candidate.setdefault("candidate_id", spec.get("candidate_id", "xgb_fundamental"))
    candidate.setdefault("engine", spec.get("engine", "xgboost"))
    candidate.setdefault("feature_set", spec.get("feature_set", "fundamental_interactions"))
    candidate.setdefault("params", dict(spec.get("params") or {}))
    return candidate


def _base_paths(base: Path) -> tuple[Path, Path, Path]:
    return (
        base / "prepared" / "exact_train_cache.parquet",
        base / "prepared" / "nested_scaffold_splits.parquet",
        base / "prepared" / "feature_registry.json",
    )


def _expanded_interactions(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    aliases = {
        "logp": "rdkit2d__MolLogP",
        "tpsa": "rdkit2d__TPSA",
        "mw": "rdkit2d__MolWt",
        "aromatic": "rdkit2d__NumAromaticRings",
        "rotors": "rdkit2d__NumRotatableBonds",
        "hba": "rdkit2d__NumHAcceptors",
        "hbd": "rdkit2d__NumHDonors",
        "basic": "f3d__basic_site_proxy_count",
        "charge": "f3d__formal_charge",
        "dipole": "f3d__ensemble_gasteiger_dipole_proxy_eA__mean",
        "polar_exposure": "f3d__ensemble_polar_radial_exposure__mean",
        "polar_contacts": "f3d__ensemble_internal_polar_contact_count__mean",
        "energy_range": "f3d__energy_range_kcal_mol",
        "effective_conformers": "f3d__effective_conformer_count",
    }
    pair_definitions = {
        "logp_x_tpsa": ("logp", "tpsa", "lipophilic-polar balance"),
        "logp_x_aromatic": ("logp", "aromatic", "aromatic hydrophobic burden"),
        "logp_x_basic": ("logp", "basic", "cationic-lipophilic hERG liability hypothesis"),
        "basic_x_aromatic": ("basic", "aromatic", "basic aromatic pharmacophore hypothesis"),
        "rotors_x_logp": ("rotors", "logp", "flexible lipophilic burden"),
        "rotors_x_aromatic": ("rotors", "aromatic", "flexible aromatic burden"),
        "hba_x_logp": ("hba", "logp", "acceptor-lipophilic balance"),
        "hbd_x_logp": ("hbd", "logp", "donor-lipophilic balance"),
        "abs_charge_x_logp": ("charge", "logp", "formal-charge lipophilicity"),
        "dipole_x_logp": ("dipole", "logp", "spatial charge separation and lipophilicity"),
        "polar_exposure_x_logp": ("polar_exposure", "logp", "3D polar exposure and lipophilicity"),
        "polar_contacts_x_logp": ("polar_contacts", "logp", "intramolecular polarity masking"),
        "energy_range_x_rotors": ("energy_range", "rotors", "conformational energetic flexibility"),
        "energy_range_x_effective_conformers": (
            "energy_range",
            "effective_conformers",
            "conformer population diversity",
        ),
    }
    out = pd.DataFrame({"structure_id": frame["structure_id"].astype(str)})
    descriptions: dict[str, str] = {}
    numeric: dict[str, pd.Series] = {}
    for alias, column in aliases.items():
        if column in frame:
            numeric[alias] = pd.to_numeric(frame[column], errors="coerce")
    for name, (left, right, description) in pair_definitions.items():
        if left not in numeric or right not in numeric:
            continue
        left_values = numeric[left].abs() if left == "charge" else numeric[left]
        right_values = numeric[right].abs() if right == "charge" else numeric[right]
        column = f"v3interaction__{name}"
        out[column] = left_values * right_values
        descriptions[column] = description
    for name in ("logp", "tpsa", "mw", "aromatic", "rotors", "basic", "dipole"):
        if name in numeric:
            column = f"v3nonlinear__{name}_squared"
            out[column] = numeric[name] ** 2
            descriptions[column] = f"nonlinear response in {name}"
    return out, descriptions


def _validate_base(base: Path) -> None:
    for relative in BASE_REQUIRED:
        if not (base / relative).is_file():
            raise WorkerError(f"completed base campaign missing {relative}")
    exact, splits, _ = _base_paths(base)
    if pq.read_metadata(exact).num_rows != EXACT_ROWS:
        raise WorkerError("base exact quantitative cache row count changed")
    if pq.read_metadata(splits).num_rows != EXACT_ROWS * 5:
        raise WorkerError("base nested scaffold split registry is incomplete")
    validation = json.loads((base / "analysis" / "validation.json").read_text())
    if validation.get("status") != "passed":
        raise WorkerError("base campaign analysis did not pass")


def _prepare(args: argparse.Namespace) -> dict[str, Any]:
    repo = Path(args.repo_root).resolve()
    base = Path(args.base_campaign_root).resolve()
    output = Path(args.output_root).resolve()
    validation_path = output / "validation.json"
    if validation_path.is_file():
        return _read_self_json(validation_path, "validation_sha256")
    _validate_base(base)
    # A passed validation file is written last.  Therefore, a nonempty directory
    # without it is an interrupted v3-only preparation and is safe to rebuild.
    if output.exists() and any(output.iterdir()):
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    exact, splits, registry = _base_paths(base)
    exact_frame = pq.read_table(exact).to_pandas()
    if len(exact_frame) != EXACT_ROWS or exact_frame["structure_id"].duplicated().any():
        raise WorkerError("exact train cache identity closure failed")
    interactions, descriptions = _expanded_interactions(exact_frame)
    interactions_path = output / "expanded_interactions.parquet"
    _write_parquet(interactions_path, interactions)

    broad_path = (repo / DEFAULT_BROAD).resolve()
    observations_path = (repo / DEFAULT_OBSERVATIONS).resolve()
    mmp_path = (repo / DEFAULT_MMP).resolve()
    global_root = (repo / DEFAULT_GLOBAL_2D).resolve()
    required_inputs = [broad_path, observations_path, mmp_path, global_root / "feature_index.parquet"]
    for path in required_inputs:
        if not path.is_file():
            raise WorkerError(f"v3 required input missing: {path}")

    broad_columns = [
        "structure_id",
        "standardized_smiles",
        "standard_inchi_key",
        "model_split",
        "scaffold_group_id",
        "target_class",
        "direct_herg_label",
        "use_as_training_label",
        "target_scope",
        "endpoint_semantics",
        "measurement_modality",
    ]
    broad = pq.read_table(
        broad_path, columns=broad_columns, filters=[("model_split", "=", "train")]
    ).to_pandas()
    if set(broad["model_split"].astype(str)) != {"train"}:
        raise WorkerError("nontraining broad fixed-dose records entered preparation")
    broad = broad.loc[
        broad["use_as_training_label"].fillna(False) & broad["direct_herg_label"].fillna(False)
    ].copy()
    if set(broad["target_scope"].astype(str)) != {"confirmed_wild_type"}:
        raise WorkerError("broad surface target scope is not confirmed wild-type")
    if set(broad["endpoint_semantics"].astype(str)) != {"AID720551_fixed_dose_activity_consensus_not_IC50"}:
        raise WorkerError("broad surface endpoint semantic contract changed")
    if set(broad["measurement_modality"].astype(str)) != {"high_throughput_thallium_flux"}:
        raise WorkerError("broad surface measurement modality contract changed")
    index = pq.read_table(
        global_root / "feature_index.parquet", columns=["feature_order", "standard_inchi_key"]
    ).to_pandas()
    index = index.drop_duplicates("standard_inchi_key", keep="first")
    broad = broad.merge(index, on="standard_inchi_key", how="inner", validate="many_to_one")
    if len(broad) != 265_625 or int(pd.to_numeric(broad["target_class"]).sum()) != 987:
        raise WorkerError("broad fixed-dose train partition count contract changed")
    if broad.empty or broad["target_class"].nunique() != 2:
        raise WorkerError("broad fixed-dose train routing lacks binary labels")
    broad_routing_path = output / "broad_wt_train_routing.parquet"
    _write_parquet(broad_routing_path, broad)

    observation_columns = [
        "structure_id",
        "standard_inchi_key",
        "scaffold_group_id",
        "model_split",
        "target_variant",
        "structure_model_eligible",
        "sensitivity_training_eligible",
        "potency_relation_pic50",
        "potency_pic50_point",
        "potency_pic50_lower_bound",
        "potency_pic50_upper_bound",
        "potency_censoring",
        "measurement_modality",
        "automation_class",
        "assay_family",
        "source_family",
    ]
    censored = pq.read_table(
        observations_path,
        columns=observation_columns,
        filters=[("model_split", "=", "train"), ("target_variant", "=", "wild_type_or_unspecified")],
    ).to_pandas()
    censored = censored.loc[
        censored["structure_model_eligible"].fillna(False)
        & censored["sensitivity_training_eligible"].fillna(False)
        & censored["potency_relation_pic50"].isin(["=", "<", "<=", ">", ">="])
    ].copy()
    censored = censored.merge(index, on="standard_inchi_key", how="inner", validate="many_to_one")
    if censored.empty or set(censored["model_split"].astype(str)) != {"train"}:
        raise WorkerError("censored sensitivity routing closure failed")
    censored_routing_path = output / "censored_train_routing.parquet"
    _write_parquet(censored_routing_path, censored)

    feature_registry = {
        "schema_version": SCHEMA,
        "base_registry": _binding(registry, "base_feature_registry"),
        "expanded_interactions": descriptions,
        "feature_sets": {
            "rdkit2d": "all rdkit2d__ columns",
            "fundamental_core": "physicochemical, polarity, topology, charge, and conformer scalars",
            "fundamental_interactions": "fundamental_core plus prespecified v3 interactions/nonlinearities",
            "fingerprint_plus_fundamental": "Morgan bits plus fundamental_interactions",
            "all_scalable_v3": "Morgan, RDKit 2D, selected 3D, and v3 interactions",
            "broad_fundamental_2d": "compact label-blind global RDKit physicochemical descriptors",
        },
        "label_blind_feature_construction": True,
    }
    registry_path = output / "feature_registry_v3.json"
    _write_json(registry_path, feature_registry, "registry_sha256")

    global_feature_shards = sorted((global_root / "features").glob("part-*.parquet"))
    if not global_feature_shards:
        raise WorkerError("global 2D feature shards are missing")
    source_bindings = [
        _binding(exact, "base_exact_train_cache"),
        _binding(splits, "base_nested_scaffold_splits"),
        _binding(interactions_path, "expanded_interactions"),
        _binding(broad_routing_path, "broad_train_routing"),
        _binding(censored_routing_path, "censored_train_routing"),
        _binding(registry_path, "feature_registry_v3"),
        _binding(broad_path, "broad_fixed_dose_surface"),
        _binding(observations_path, "training_observations"),
        _binding(mmp_path, "training_mmp_effects"),
        _binding(global_root / "feature_index.parquet", "global_2d_index"),
        _binding(global_root / "feature_cache_manifest.json", "global_2d_manifest"),
        _binding(global_root / "feature_schema.json", "global_2d_schema"),
        *[_binding(path, "global_2d_feature_shard") for path in global_feature_shards],
    ]
    manifest = {
        "schema_version": SCHEMA,
        "created_at": _now(),
        "status": "passed",
        "repo_root": str(repo),
        "base_campaign_root": str(base),
        "source_bindings": source_bindings,
        "counts": {
            "exact_quantitative_structures": len(exact_frame),
            "broad_wt_auxiliary_structures": len(broad),
            "broad_wt_auxiliary_positive": int(pd.to_numeric(broad["target_class"]).sum()),
            "censored_sensitivity_observations": len(censored),
            "censored_sensitivity_structures": int(censored["structure_id"].nunique()),
        },
        "scientific_scope": _scope("multiple separate train-only endpoints"),
    }
    manifest_path = output / "manifest.json"
    _write_json(manifest_path, manifest, "manifest_sha256")
    validation = {
        "schema_version": SCHEMA,
        "status": "passed",
        "created_at": _now(),
        "exact_rows": len(exact_frame),
        "broad_train_rows": len(broad),
        "censored_rows": len(censored),
        "supported_operations": sorted(OPERATIONS),
        "repository_validation_labels_opened": False,
        "repository_test_labels_opened": False,
        "base_campaign_unchanged": True,
        "manifest": _binding(manifest_path, "prepared_manifest"),
    }
    return _write_json(validation_path, validation, "validation_sha256")


def _validate_prepared(prepared: Path, *, full: bool = False) -> dict[str, Any]:
    validation = _read_self_json(prepared / "validation.json", "validation_sha256")
    if validation.get("status") != "passed":
        raise WorkerError("v3 prepared surface did not pass")
    _verify_binding(validation["manifest"])
    manifest = _read_self_json(prepared / "manifest.json", "manifest_sha256")
    for binding in manifest["source_bindings"]:
        path = Path(str(binding["path"]))
        if not path.is_file() or path.stat().st_size != int(binding["bytes"]):
            raise WorkerError(f"prepared source missing or changed size: {path}")
        if full or binding["role"] in {
            "expanded_interactions",
            "broad_train_routing",
            "censored_train_routing",
            "feature_registry_v3",
        }:
            _verify_binding(binding)
    if manifest["scientific_scope"]["repository_validation_labels_opened"]:
        raise WorkerError("prepared surface reports validation-label access")
    return manifest


def _finite_numeric(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    result = frame.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce")
    values = result.to_numpy(dtype=np.float64, copy=True)
    values[~np.isfinite(values) | (np.abs(values) > MAX_ABS)] = np.nan
    return pd.DataFrame(values.astype(np.float32), columns=list(columns), index=frame.index)


def _fundamental_columns(columns: Sequence[str]) -> list[str]:
    exact_names = {
        "rdkit2d__MolLogP",
        "rdkit2d__TPSA",
        "rdkit2d__MolWt",
        "rdkit2d__ExactMolWt",
        "rdkit2d__NumHAcceptors",
        "rdkit2d__NumHDonors",
        "rdkit2d__NumRotatableBonds",
        "rdkit2d__NumAromaticRings",
        "rdkit2d__FractionCSP3",
        "rdkit2d__HeavyAtomCount",
        "rdkit2d__RingCount",
        "rdkit2d__MaxPartialCharge",
        "rdkit2d__MinPartialCharge",
        "rdkit2d__LabuteASA",
        "rdkit2d__qed",
        "rdkit2d__BertzCT",
        "rdkit2d__BalabanJ",
        "rdkit2d__NHOHCount",
        "rdkit2d__NOCount",
    }
    terms = (
        "formal_charge",
        "basic_site",
        "acidic_site",
        "rotatable_bond_count",
        "gasteiger_dipole",
        "absolute_charge_radius",
        "polar_radial_exposure",
        "internal_polar_contact",
        "energy_range",
        "effective_conformer_count",
        "dominant_conformer_weight",
        "radius_of_gyration",
        "asphericity",
        "spherocity",
    )
    return sorted(
        column
        for column in columns
        if column in exact_names or (column.startswith("f3d__") and any(term in column for term in terms))
    )


def _load_exact(base: Path, prepared: Path, feature_set: str) -> tuple[pd.DataFrame, list[str]]:
    exact, _, _ = _base_paths(base)
    frame = pq.read_table(exact).to_pandas()
    interactions = pq.read_table(prepared / "expanded_interactions.parquet").to_pandas()
    frame = frame.merge(interactions, on="structure_id", how="left", validate="one_to_one")
    numeric = [
        column
        for column in frame.columns
        if pd.api.types.is_numeric_dtype(frame[column])
        and column not in {"target_pic50", "exact_observation_count", "feature_order"}
    ]
    morgan = sorted(column for column in numeric if column.startswith("morgan__"))
    fundamental = _fundamental_columns(numeric)
    interactions_columns = sorted(
        column
        for column in numeric
        if column.startswith(("interaction__", "v3interaction__", "v3nonlinear__"))
    )
    base_registry = json.loads((base / "prepared" / "feature_registry.json").read_text("utf-8"))
    groups = base_registry.get("feature_groups", {})
    resolved_base_sets: dict[str, list[str]] = {}
    for name, group_names in base_registry.get("feature_sets", {}).items():
        resolved_base_sets[name] = sorted(
            set(column for group_name in group_names for column in groups.get(group_name, []))
        )
    feature_sets = {
        **resolved_base_sets,
        "fundamental_core": fundamental,
        "fundamental_interactions": sorted(set(fundamental + interactions_columns)),
        "fingerprint_plus_fundamental": sorted(set(morgan + fundamental + interactions_columns)),
        "all_scalable_v3": sorted(set(resolved_base_sets.get("all_scalable", []) + interactions_columns)),
    }
    if feature_set not in feature_sets:
        raise WorkerError(f"unknown feature set {feature_set!r}; available={sorted(feature_sets)}")
    features = feature_sets[feature_set]
    if not features:
        raise WorkerError(f"feature set {feature_set} resolved empty")
    frame[features] = _finite_numeric(frame, features)
    return frame, features


def _folds(value: Any, default: int = 5) -> list[int]:
    if value is None:
        return list(range(default))
    if isinstance(value, int):
        return list(range(value))
    return [int(item) for item in value]


def _model(candidate: dict[str, Any], workers: int, seed: int, classification: bool = False) -> Any:
    engine = str(candidate.get("engine", "xgboost")).lower()
    params = dict(candidate.get("params") or {})
    if engine in {"xgb", "xgboost"}:
        try:
            from xgboost import XGBClassifier, XGBRegressor
        except ImportError as error:
            raise Unavailable("xgboost is not installed") from error
        defaults: dict[str, Any] = {
            "n_estimators": 500,
            "max_depth": 7,
            "learning_rate": 0.035,
            "subsample": 0.85,
            "colsample_bytree": 0.72,
            "min_child_weight": 4,
            "reg_lambda": 2.0,
            "n_jobs": workers,
            "random_state": seed,
            "tree_method": "hist",
        }
        defaults.update(params)
        if classification:
            defaults.setdefault("objective", "binary:logistic")
            defaults.setdefault("eval_metric", "logloss")
            return XGBClassifier(**defaults)
        defaults.setdefault("objective", "reg:squarederror")
        return XGBRegressor(**defaults)
    if engine in {"extra_trees", "extratrees"}:
        defaults = {
            "n_estimators": 600,
            "min_samples_leaf": 2,
            "max_features": 0.7,
            "random_state": seed,
            "n_jobs": workers,
            "class_weight": "balanced",
        }
        defaults.update(params)
        if classification:
            return ExtraTreesClassifier(**defaults)
        defaults.pop("class_weight", None)
        return ExtraTreesRegressor(**defaults)
    if engine in {"lightgbm", "lgbm"}:
        try:
            from lightgbm import LGBMClassifier, LGBMRegressor
        except ImportError as error:
            raise Unavailable("lightgbm is not installed") from error
        defaults = {
            "n_estimators": 700,
            "learning_rate": 0.035,
            "num_leaves": 31,
            "n_jobs": workers,
            "random_state": seed,
            "verbosity": -1,
        }
        defaults.update(params)
        if classification:
            return LGBMClassifier(**defaults)
        return LGBMRegressor(**defaults)
    if engine in {"random_forest", "rf"} and not classification:
        defaults = {
            "n_estimators": 500,
            "min_samples_leaf": 2,
            "max_features": 0.7,
            "random_state": seed,
            "n_jobs": workers,
        }
        defaults.update(params)
        return RandomForestRegressor(**defaults)
    if engine == "ridge" and not classification:
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(**params))
    raise Unavailable(f"unsupported local engine: {engine}")


def _regression_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    finite = np.isfinite(observed) & np.isfinite(predicted)
    observed, predicted = observed[finite], predicted[finite]
    if len(observed) < 2:
        raise WorkerError("too few finite predictions")
    residual = observed - predicted
    pearson = pearsonr(observed, predicted).statistic if np.std(predicted) > 0 else 0.0
    spearman = spearmanr(observed, predicted).statistic if np.std(predicted) > 0 else 0.0
    return {
        "n": len(observed),
        "mae": float(mean_absolute_error(observed, predicted)),
        "rmse": float(mean_squared_error(observed, predicted) ** 0.5),
        "r2": float(r2_score(observed, predicted)),
        "pearson": float(np.nan_to_num(pearson)),
        "spearman": float(np.nan_to_num(spearman)),
        "fraction_within_0p5": float(np.mean(np.abs(residual) <= 0.5)),
        "fraction_within_1p0": float(np.mean(np.abs(residual) <= 1.0)),
    }


def _fit_regression_model(model: Any, fit_x: pd.DataFrame, fit_y: np.ndarray) -> Any:
    if hasattr(model, "named_steps"):
        model.fit(fit_x, fit_y)
        return model
    pipeline = make_pipeline(SimpleImputer(strategy="median"), model)
    pipeline.fit(_finite_numeric(fit_x, fit_x.columns), fit_y)
    return pipeline


def _predict_model(model: Any, frame: pd.DataFrame) -> np.ndarray:
    return np.asarray(model.predict(_finite_numeric(frame, frame.columns)), dtype=np.float64)


def _importance(model: Any, features: list[str], fold: int, candidate_id: str) -> pd.DataFrame:
    resolved = model
    if hasattr(resolved, "named_steps"):
        resolved = list(resolved.named_steps.values())[-1]
    if hasattr(resolved, "feature_importances_"):
        values = np.asarray(resolved.feature_importances_, dtype=np.float64)
    elif hasattr(resolved, "coef_"):
        values = np.abs(np.ravel(resolved.coef_))
    else:
        values = np.zeros(len(features), dtype=np.float64)
    return pd.DataFrame(
        {
            "candidate_id": candidate_id,
            "fold": fold,
            "feature": features,
            "importance": values,
            "biological_process_hypothesis": [_process(item) for item in features],
            "causal_interpretation_allowed": False,
        }
    )


def _process(feature: str) -> str:
    lower = feature.lower()
    if feature.startswith(("v3interaction__", "interaction__")):
        return "prespecified_parameter_interaction"
    if any(term in lower for term in ("charge", "dipole", "autocorr")):
        return "spatial_electrostatics"
    if any(term in lower for term in ("logp", "slogp")):
        return "lipophilic_partitioning"
    if any(term in lower for term in ("tpsa", "polar", "accept", "donor")):
        return "polarity_desolvation_proxy"
    if any(term in lower for term in ("arom", "ring")):
        return "aromatic_hydrophobic_architecture"
    if any(term in lower for term in ("energy", "conformer", "rotatable")):
        return "conformational_flexibility"
    if any(term in lower for term in ("shape", "whim", "pmi", "spher")):
        return "ligand_shape"
    if feature.startswith(("morgan__", "maccs__")):
        return "substructure_identity"
    return "other_molecular_descriptor"


def _regression_unit(
    base: Path,
    prepared: Path,
    directory: Path,
    unit_id: str,
    operation: str,
    spec: dict[str, Any],
    workers: int,
) -> dict[str, Any]:
    candidate = _candidate(spec)
    if operation == "feature_ablation" and spec.get("base_feature_set"):
        candidate["feature_set"] = str(spec["base_feature_set"])
    if operation == "interaction_stability":
        candidate["feature_set"] = "fundamental_interactions"
        if "terms" in spec or "seeds" in spec or "heldout_permutation_repeats" in spec:
            raise WorkerError(
                "interaction_stability accepts the complete prespecified interaction surface only; "
                "named-term/repeated-permutation fields are unsupported"
            )
    if operation == "feature_ablation" and any(key in spec for key in ("remove_groups", "add_groups")):
        raise WorkerError("feature_ablation must specify a materially distinct base_feature_set only")
    if operation == "expanded_hpo_tree" and spec.get("evaluation_stage", "inner") != "inner":
        raise WorkerError("expanded_hpo_tree must use evaluation_stage=inner")
    if operation == "nested_robustness" and spec.get("evaluation_stage", "outer") != "outer":
        raise WorkerError("nested_robustness must use evaluation_stage=outer")
    seed = int(spec.get("seed", SEED))
    frame, features = _load_exact(base, prepared, str(candidate["feature_set"]))
    _, split_path, _ = _base_paths(base)
    splits = pq.read_table(split_path).to_pandas()
    stage = str(spec.get("evaluation_stage", "inner" if operation == "expanded_hpo_tree" else "outer"))
    records: list[pd.DataFrame] = []
    importances: list[pd.DataFrame] = []
    models: list[Path] = []
    started = time.monotonic()
    if stage == "inner":
        outer_fold = int(spec.get("outer_fold", 0))
        assignments = splits.loc[splits["outer_fold"].eq(outer_fold)].copy()
        assignments = assignments.loc[assignments["outer_role"].eq("fit")]
        evaluation_folds = _folds(spec.get("inner_folds"), 3)
        fold_specs = [
            (
                inner,
                set(assignments.loc[assignments["inner_fold"].ne(inner), "structure_id"].astype(str)),
                set(assignments.loc[assignments["inner_fold"].eq(inner), "structure_id"].astype(str)),
            )
            for inner in evaluation_folds
        ]
    elif stage == "outer":
        fold_specs = []
        for outer in _folds(spec.get("outer_folds"), 5):
            assignments = splits.loc[splits["outer_fold"].eq(outer)].copy()
            fold_specs.append(
                (
                    outer,
                    set(assignments.loc[assignments["outer_role"].eq("fit"), "structure_id"].astype(str)),
                    set(assignments.loc[assignments["outer_role"].eq("heldout"), "structure_id"].astype(str)),
                )
            )
    else:
        raise WorkerError(f"unsupported evaluation stage {stage}")
    for fold, fit_ids, eval_ids in fold_specs:
        if fit_ids & eval_ids:
            raise WorkerError("scaffold evaluation overlap")
        fit = frame.loc[frame["structure_id"].astype(str).isin(fit_ids)].copy()
        evaluation = frame.loc[frame["structure_id"].astype(str).isin(eval_ids)].copy()
        fit_groups = set(fit["scaffold_group_id"].astype(str))
        eval_groups = set(evaluation["scaffold_group_id"].astype(str))
        if fit_groups & eval_groups:
            raise WorkerError("scaffold leakage in regression unit")
        model = _model(candidate, workers, seed + int(fold))
        model = _fit_regression_model(model, fit[features], fit["target_pic50"].to_numpy(dtype=np.float64))
        predicted = _predict_model(model, evaluation[features])
        records.append(
            pd.DataFrame(
                {
                    "structure_id": evaluation["structure_id"].astype(str),
                    "scaffold_group_id": evaluation["scaffold_group_id"].astype(str),
                    "candidate_id": candidate["candidate_id"],
                    "stage": stage,
                    "fold": int(fold),
                    "observed_pic50": evaluation["target_pic50"].to_numpy(dtype=np.float64),
                    "predicted_pic50": predicted,
                }
            )
        )
        importances.append(_importance(model, features, int(fold), str(candidate["candidate_id"])))
        if bool(spec.get("retain_model_artifact", False)):
            model_path = directory / "models" / f"fold_{int(fold)}.joblib"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, model_path)
            models.append(model_path)
    predictions = pd.concat(records, ignore_index=True)
    predictions["residual_observed_minus_predicted"] = (
        predictions["observed_pic50"] - predictions["predicted_pic50"]
    )
    metrics = _regression_metrics(
        predictions["observed_pic50"].to_numpy(), predictions["predicted_pic50"].to_numpy()
    )
    metrics.update(
        selection_score=metrics["mae"],
        evaluation_stage=stage,
        folds=len(fold_specs),
        feature_count=len(features),
        fit_elapsed_seconds=time.monotonic() - started,
    )
    prediction_path = directory / "oof_predictions.parquet"
    importance_path = directory / "feature_importance.parquet"
    schema_path = directory / "feature_schema.json"
    _write_parquet(prediction_path, predictions)
    _write_parquet(importance_path, pd.concat(importances, ignore_index=True))
    _write_json(
        schema_path,
        {
            "schema_version": SCHEMA,
            "feature_set": candidate["feature_set"],
            "features": features,
            "feature_construction_used_labels": False,
        },
        "feature_schema_sha256",
    )
    artifacts = [
        _binding(prediction_path, "oof_predictions"),
        _binding(importance_path, "feature_importance"),
        _binding(schema_path, "feature_schema"),
    ] + [_binding(path, "model_fold") for path in models]
    if operation == "interaction_stability":
        artifacts.extend(
            _relationship_artifacts(directory, frame, features, predictions, candidate, workers, seed)
        )
    return _write_unit(
        directory,
        unit_id=unit_id,
        operation=operation,
        spec=spec,
        metrics=metrics,
        artifacts=artifacts,
        limitations=[
            "All evaluation is nested within repository train; it is not external validation.",
            "Feature responses are noncausal molecular hypotheses.",
        ],
    )


def _relationship_artifacts(
    directory: Path,
    frame: pd.DataFrame,
    features: list[str],
    predictions: pd.DataFrame,
    candidate: dict[str, Any],
    workers: int,
    seed: int,
) -> list[dict[str, Any]]:
    joined = frame.merge(
        predictions[["structure_id", "observed_pic50", "predicted_pic50"]], on="structure_id", how="inner"
    )
    fundamental = [feature for feature in features if not feature.startswith(("morgan__", "maccs__"))]
    rows: list[dict[str, Any]] = []
    for feature in fundamental:
        values = pd.to_numeric(joined[feature], errors="coerce")
        finite = values.notna() & np.isfinite(values)
        if int(finite.sum()) < 30 or values[finite].nunique() < 3:
            continue
        target_rho = spearmanr(values[finite], joined.loc[finite, "observed_pic50"]).statistic
        residual_rho = spearmanr(
            values[finite], joined.loc[finite, "observed_pic50"] - joined.loc[finite, "predicted_pic50"]
        ).statistic
        rows.append(
            {
                "feature": feature,
                "biological_process_hypothesis": _process(feature),
                "signed_spearman_target": float(np.nan_to_num(target_rho)),
                "signed_spearman_residual": float(np.nan_to_num(residual_rho)),
                "structures": int(finite.sum()),
                "causal_interpretation_allowed": False,
            }
        )
    association_path = directory / "directional_associations.parquet"
    _write_parquet(association_path, pd.DataFrame(rows))
    return [_binding(association_path, "directional_associations")]


GLOBAL_FEATURES = [
    "rdkit2d__MolLogP",
    "rdkit2d__TPSA",
    "rdkit2d__MolWt",
    "rdkit2d__ExactMolWt",
    "rdkit2d__NumHAcceptors",
    "rdkit2d__NumHDonors",
    "rdkit2d__NumRotatableBonds",
    "rdkit2d__NumAromaticRings",
    "rdkit2d__FractionCSP3",
    "rdkit2d__HeavyAtomCount",
    "rdkit2d__RingCount",
    "rdkit2d__MaxPartialCharge",
    "rdkit2d__MinPartialCharge",
    "rdkit2d__LabuteASA",
    "rdkit2d__qed",
    "rdkit2d__BertzCT",
    "rdkit2d__BalabanJ",
    "rdkit2d__NHOHCount",
    "rdkit2d__NOCount",
]


def _load_global_features(
    repo: Path, routing: pd.DataFrame, *, all_rdkit_descriptors: bool = False
) -> pd.DataFrame:
    wanted = set(pd.to_numeric(routing["feature_order"], errors="raise").astype(int))
    frames: list[pd.DataFrame] = []
    features_root = repo / DEFAULT_GLOBAL_2D / "features"
    shards = sorted(features_root.glob("part-*.parquet"))
    if not shards:
        raise WorkerError("global 2D feature shards are missing")
    available = set(pq.read_schema(shards[0]).names)
    selected = (
        sorted(column for column in available if column.startswith("rdkit2d__"))
        if all_rdkit_descriptors
        else [column for column in GLOBAL_FEATURES if column in available]
    )
    columns = ["feature_order", *selected]
    for shard in shards:
        orders = pq.read_table(shard, columns=["feature_order"]).column(0).to_numpy()
        local = set(int(item) for item in orders) & wanted
        if not local:
            continue
        block = pq.read_table(shard, columns=columns).to_pandas()
        block = block.loc[block["feature_order"].isin(local)]
        frames.append(block)
    if not frames:
        raise WorkerError("no requested identities found in global 2D cache")
    features = pd.concat(frames, ignore_index=True)
    if features["feature_order"].duplicated().any():
        raise WorkerError("global feature cache contains duplicate feature orders")
    return routing.merge(features, on="feature_order", how="inner", validate="many_to_one")


def _classification_metrics(
    observed: np.ndarray, probability: np.ndarray, fold_values: np.ndarray, *, fixed_fpr: float = 0.01
) -> tuple[dict[str, Any], float]:
    observed = observed.astype(int)
    probability = np.clip(probability.astype(float), 0.0, 1.0)
    decisions = np.zeros(len(observed), dtype=bool)
    thresholds: list[float] = []
    for fold in sorted(set(int(item) for item in fold_values)):
        evaluation = fold_values == fold
        calibration = ~evaluation
        negative = probability[calibration & (observed == 0)]
        if len(negative) < 2:
            raise WorkerError("too few cross-fold negatives for fixed-FPR threshold")
        threshold = float(np.quantile(negative, 1.0 - fixed_fpr))
        thresholds.append(threshold)
        decisions[evaluation] = probability[evaluation] >= threshold
    threshold = float(np.median(thresholds))
    positive = observed == 1
    ece = 0.0
    bins = pd.qcut(pd.Series(probability), q=min(20, len(np.unique(probability))), duplicates="drop")
    for _, indices in pd.Series(np.arange(len(observed))).groupby(bins, observed=False):
        selected = indices.to_numpy(dtype=int)
        ece += (
            len(selected)
            / len(observed)
            * abs(float(probability[selected].mean() - observed[selected].mean()))
        )
    metrics: dict[str, Any] = {
        "n": len(observed),
        "positives": int(observed.sum()),
        "prevalence": float(observed.mean()),
        "roc_auc": float(roc_auc_score(observed, probability)),
        "pr_auc": float(average_precision_score(observed, probability)),
        "mcc": float(matthews_corrcoef(observed, decisions)),
        "brier": float(brier_score_loss(observed, probability)),
        "mcc_threshold": threshold,
        "threshold_rule": f"cross-fold threshold at {fixed_fpr:.3%} false-positive rate",
        "recall_at_fixed_fpr": float(decisions[positive].mean()),
        "precision_at_fixed_fpr": float(observed[decisions].mean()) if decisions.any() else 0.0,
        "expected_calibration_error": float(ece),
    }
    metrics["selection_score"] = -metrics["pr_auc"]
    return metrics, threshold


def _broad_unit(
    repo: Path,
    prepared: Path,
    directory: Path,
    unit_id: str,
    spec: dict[str, Any],
    workers: int,
) -> dict[str, Any]:
    routing = pq.read_table(prepared / "broad_wt_train_routing.parquet").to_pandas()
    candidate = _candidate(spec)
    broad_feature_set = str(candidate.get("feature_set", "broad_all_rdkit"))
    if broad_feature_set not in {"broad_compact_19", "broad_all_rdkit"}:
        raise WorkerError("broad feature_set must be broad_compact_19 or broad_all_rdkit")
    frame = _load_global_features(repo, routing, all_rdkit_descriptors=broad_feature_set == "broad_all_rdkit")
    features = (
        [column for column in GLOBAL_FEATURES if column in frame]
        if broad_feature_set == "broad_compact_19"
        else sorted(column for column in frame if column.startswith("rdkit2d__"))
    )
    x = _finite_numeric(frame, features)
    y = pd.to_numeric(frame["target_class"], errors="raise").to_numpy(dtype=int)
    groups = frame["scaffold_group_id"].fillna(frame["structure_id"]).astype(str).to_numpy()
    params = dict(candidate.get("params") or {})
    if candidate["engine"].lower() in {"xgb", "xgboost"}:
        params.setdefault("scale_pos_weight", float((len(y) - y.sum()) / max(int(y.sum()), 1)))
    candidate["params"] = params
    folds = int(spec.get("outer_folds", 3))
    predictions = np.full(len(frame), np.nan)
    raw_predictions = np.full(len(frame), np.nan)
    fold_values = np.full(len(frame), -1, dtype=int)
    models: list[Path] = []
    for fold, (fit_index, eval_index) in enumerate(
        StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=int(spec.get("seed", SEED))).split(
            x, y, groups
        )
    ):
        if set(groups[fit_index]) & set(groups[eval_index]):
            raise WorkerError("scaffold leakage in broad auxiliary classifier")
        inner_train, calibration = next(
            StratifiedGroupKFold(
                n_splits=5, shuffle=True, random_state=int(spec.get("seed", SEED)) + fold
            ).split(x.iloc[fit_index], y[fit_index], groups[fit_index])
        )
        model_fit_index = fit_index[inner_train]
        calibration_index = fit_index[calibration]
        fold_candidate = dict(candidate)
        fold_params = dict(candidate.get("params") or {})
        if candidate["engine"].lower() in {"xgb", "xgboost"}:
            fold_y = y[model_fit_index]
            fold_params["scale_pos_weight"] = float((len(fold_y) - fold_y.sum()) / max(int(fold_y.sum()), 1))
        fold_candidate["params"] = fold_params
        model = _model(fold_candidate, workers, int(spec.get("seed", SEED)) + fold, classification=True)
        pipeline = make_pipeline(SimpleImputer(strategy="median"), model)
        pipeline.fit(x.iloc[model_fit_index], y[model_fit_index])
        calibration_raw = pipeline.predict_proba(x.iloc[calibration_index])[:, 1]
        calibrator = IsotonicRegression(out_of_bounds="clip").fit(calibration_raw, y[calibration_index])
        outer_raw = pipeline.predict_proba(x.iloc[eval_index])[:, 1]
        raw_predictions[eval_index] = outer_raw
        predictions[eval_index] = calibrator.predict(outer_raw)
        fold_values[eval_index] = fold
        if bool(spec.get("retain_model_artifact", False)) or bool(spec.get("final_refit", False)):
            path = directory / "models" / f"fold_{fold}.joblib"
            path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump({"classifier": pipeline, "calibrator": calibrator, "features": features}, path)
            models.append(path)
    metrics, threshold = _classification_metrics(
        y, predictions, fold_values, fixed_fpr=float(spec.get("fixed_fpr", 0.01))
    )
    metrics.update(
        evaluation_stage="train_only_grouped_scaffold_oof",
        folds=folds,
        feature_count=len(features),
        endpoint_separate_from_pic50=True,
        full_surface_structures=339_373,
        train_partition_structures=len(frame),
    )
    oof = pd.DataFrame(
        {
            "structure_id": frame["structure_id"].astype(str),
            "scaffold_group_id": groups,
            "observed_class": y,
            "predicted_probability": predictions,
            "raw_predicted_probability": raw_predictions,
            "fold": fold_values,
        }
    )
    prediction_path = directory / "oof_predictions.parquet"
    _write_parquet(prediction_path, oof)
    calibration_rows: list[dict[str, Any]] = []
    bins = pd.qcut(pd.Series(predictions), q=min(20, len(np.unique(predictions))), duplicates="drop")
    for band, index in pd.Series(np.arange(len(y))).groupby(bins, observed=False):
        selected = index.to_numpy(dtype=int)
        calibration_rows.append(
            {
                "probability_band": str(band),
                "structures": len(selected),
                "mean_probability": float(predictions[selected].mean()),
                "observed_rate": float(y[selected].mean()),
            }
        )
    calibration_path = directory / "calibration.parquet"
    _write_parquet(calibration_path, pd.DataFrame(calibration_rows))
    enrichment: dict[str, Any] = {"prevalence": float(y.mean()), "top_fraction": {}}
    ranked = np.argsort(-np.asarray(predictions, dtype=np.float64))
    for fraction in (0.001, 0.005, 0.01, 0.05, 0.10):
        n = max(1, int(math.ceil(len(y) * fraction)))
        rate = float(y[ranked[:n]].mean())
        enrichment["top_fraction"][str(fraction)] = {
            "structures": n,
            "positive_rate": rate,
            "enrichment_over_prevalence": rate / max(float(y.mean()), 1e-12),
        }
    enrichment_path = directory / "enrichment.json"
    _write_json(enrichment_path, enrichment, "enrichment_sha256")
    schema_path = directory / "feature_schema.json"
    _write_json(
        schema_path,
        {
            "schema_version": SCHEMA,
            "features": features,
            "endpoint": "confirmed-WT fixed-dose binary auxiliary",
            "label_blind_features": True,
            "not_pic50": True,
        },
        "feature_schema_sha256",
    )
    artifacts = [
        _binding(prediction_path, "oof_predictions"),
        _binding(calibration_path, "calibration"),
        _binding(enrichment_path, "enrichment"),
        _binding(schema_path, "feature_schema"),
    ] + [_binding(path, "model_fold") for path in models]
    if bool(spec.get("final_refit", False)):
        model_path = directory / "model.joblib"
        # Preserve the calibrated grouped-scaffold ensemble whose probability
        # scales were each calibrated on disjoint inner data.  A new all-data
        # classifier plus an OOF calibrator would be scale-incompatible.
        fold_bundles = [joblib.load(path) for path in models]
        if not fold_bundles:
            raise WorkerError("broad final refit requires retained calibrated fold models")
        joblib.dump(
            {
                "model_kind": "calibrated_grouped_scaffold_fold_ensemble",
                "members": fold_bundles,
                "features": features,
                "fixed_fpr_threshold": threshold,
                "full_train_single_model_refit": False,
            },
            model_path,
        )
        smoke_member_probabilities = []
        for bundle in fold_bundles:
            raw = bundle["classifier"].predict_proba(x.iloc[: min(16, len(x))])[:, 1]
            smoke_member_probabilities.append(bundle["calibrator"].predict(raw))
        smoke_probability = np.mean(np.asarray(smoke_member_probabilities), axis=0)
        smoke_path = directory / "inference_smoke.json"
        _write_json(
            smoke_path,
            {
                "status": "passed",
                "rows": len(smoke_probability),
                "all_finite": bool(np.isfinite(smoke_probability).all()),
                "probabilities_in_unit_interval": bool(
                    ((smoke_probability >= 0) & (smoke_probability <= 1)).all()
                ),
                "decision_threshold": threshold,
            },
            "inference_smoke_sha256",
        )
        artifacts.extend([_binding(model_path, "model"), _binding(smoke_path, "inference_smoke")])
    return _write_unit(
        directory,
        unit_id=unit_id,
        operation="broad_wt_auxiliary",
        spec=spec,
        metrics=metrics,
        artifacts=artifacts,
        limitations=[
            "The 339,373-structure surface contributes only its 265,625-row train partition here.",
            "Extreme class imbalance makes PR-AUC, calibration, and enrichment primary diagnostics.",
            "This fixed-dose binary endpoint is never treated as exact pIC50 or pooled with it.",
        ],
    )


def _source_prediction(results_root: Path, spec: dict[str, Any], base: Path) -> pd.DataFrame:
    source = spec.get("source_unit_id")
    if source:
        document = _read_self_json(_unit_dir(results_root, str(source)) / "unit.json", "unit_json_sha256")
        artifact = next((item for item in document["artifacts"] if item["role"] == "oof_predictions"), None)
        if artifact is None:
            raise Unavailable(f"source unit {source} has no OOF predictions")
        _verify_binding(artifact)
        return pd.read_parquet(artifact["path"])
    model_id = spec.get("source_model_id")
    path = base / "analysis" / "outer_oof_predictions.parquet"
    if not path.is_file():
        raise Unavailable("base outer OOF predictions are unavailable")
    frame = pd.read_parquet(path)
    if model_id is not None and "model_id" in frame:
        frame = frame.loc[frame["model_id"].eq(model_id)]
    if frame.empty:
        raise Unavailable("source prediction selection is empty")
    return frame


def _assay_unit(
    base: Path,
    prepared: Path,
    results_root: Path,
    directory: Path,
    unit_id: str,
    spec: dict[str, Any],
) -> dict[str, Any]:
    predictions = _source_prediction(results_root, spec, base)
    exact, _, _ = _base_paths(base)
    metadata_columns = [
        "structure_id",
        "measurement_modality",
        "automation_class",
        "assay_family",
        "source_family",
        "protocol_completeness_mean",
        "wild_type_evidence_scope",
        "master_confirmed_wild_type_scope",
    ]
    metadata = pq.read_table(exact, columns=metadata_columns).to_pandas()
    frame = predictions.merge(metadata, on="structure_id", how="inner", validate="many_to_one")
    observed_column = "observed_pic50"
    predicted_column = "predicted_pic50"
    if observed_column not in frame or predicted_column not in frame:
        raise Unavailable("assay stratification requires quantitative OOF predictions")
    rows: list[dict[str, Any]] = []
    requested_stratum = spec.get("stratum")
    aliases = {"assay_type": "assay_family", "quality_tier": "protocol_completeness_mean"}
    requested_stratum = aliases.get(str(requested_stratum), requested_stratum)
    dimensions = metadata_columns[1:] if requested_stratum in (None, "all") else [str(requested_stratum)]
    if any(dimension not in metadata_columns[1:] for dimension in dimensions):
        raise WorkerError(f"unsupported assay stratum: {requested_stratum}")
    for dimension in dimensions:
        values = frame[dimension]
        if dimension == "protocol_completeness_mean":
            values = pd.cut(pd.to_numeric(values), [-np.inf, 0, 2, 4, np.inf]).astype(str)
        for level, group in frame.assign(_level=values).groupby("_level", dropna=False):
            if len(group) < int(spec.get("minimum_stratum_size", spec.get("minimum_subgroup_size", 50))):
                continue
            metrics = _regression_metrics(
                group[observed_column].to_numpy(), group[predicted_column].to_numpy()
            )
            rows.append({"dimension": dimension, "level": str(level), **metrics, "exploratory_only": True})
    output_path = directory / "assay_quality_strata.parquet"
    _write_parquet(output_path, pd.DataFrame(rows))
    metrics = {
        "strata": len(rows),
        "structures": int(frame["structure_id"].nunique()),
        "selection_score": None,
        "chemistry_source_confounding_acknowledged": True,
    }
    return _write_unit(
        directory,
        unit_id=unit_id,
        operation="assay_quality_strata",
        spec=spec,
        metrics=metrics,
        artifacts=[_binding(output_path, "assay_quality_strata")],
        limitations=["Assay strata are observational and confounded by chemistry, source, and protocol."],
    )


def _uncertainty_unit(
    base: Path,
    results_root: Path,
    directory: Path,
    unit_id: str,
    spec: dict[str, Any],
) -> dict[str, Any]:
    method = str(spec.get("method", "ensemble_variance_conformal"))
    if method not in {"ensemble_variance_conformal", "quantile_residual_conformal"}:
        raise WorkerError(f"unsupported uncertainty method: {method}")
    member_ids = [str(item) for item in spec.get("member_unit_ids", [])]
    frames: list[pd.DataFrame] = []
    if member_ids:
        for member in member_ids:
            member_spec = dict(spec)
            member_spec["source_unit_id"] = member
            frame = _source_prediction(results_root, member_spec, base)
            if (
                len(frame) != EXACT_ROWS
                or frame["structure_id"].nunique() != EXACT_ROWS
                or frame["structure_id"].duplicated().any()
                or set(frame.get("stage", pd.Series(dtype=str)).astype(str)) != {"outer"}
            ):
                raise WorkerError(f"uncertainty member {member} lacks full 18,801-row outer OOF coverage")
            selected = ["structure_id", "scaffold_group_id", "observed_pic50", "predicted_pic50"]
            if not frames and "fold" in frame:
                selected.append("fold")
            frames.append(frame[selected].rename(columns={"predicted_pic50": f"prediction__{member}"}))
    else:
        frames.append(_source_prediction(results_root, spec, base))
    combined = frames[0]
    for frame in frames[1:]:
        check = combined[["structure_id", "scaffold_group_id", "observed_pic50"]].merge(
            frame[["structure_id", "scaffold_group_id", "observed_pic50"]],
            on="structure_id",
            suffixes=("_left", "_right"),
            validate="one_to_one",
        )
        if len(check) != EXACT_ROWS or not (
            check["scaffold_group_id_left"].astype(str).eq(check["scaffold_group_id_right"].astype(str)).all()
            and np.isclose(check["observed_pic50_left"], check["observed_pic50_right"]).all()
        ):
            raise WorkerError("uncertainty member identity/target/scaffold alignment failed")
        if "fold" in frame and "fold" in combined:
            fold_check = combined[["structure_id", "fold"]].merge(
                frame[["structure_id", "fold"]], on="structure_id", suffixes=("_left", "_right")
            )
            if not fold_check["fold_left"].eq(fold_check["fold_right"]).all():
                raise WorkerError("uncertainty members use inconsistent outer-fold assignments")
        combined = combined.merge(
            frame.drop(columns=["scaffold_group_id", "observed_pic50", "fold"], errors="ignore"),
            on="structure_id",
            validate="one_to_one",
        )
    prediction_columns = [column for column in combined if column.startswith("prediction__")]
    if not prediction_columns and "predicted_pic50" in combined:
        prediction_columns = ["predicted_pic50"]
    combined["ensemble_prediction"] = combined[prediction_columns].mean(axis=1)
    combined["ensemble_sd"] = combined[prediction_columns].std(axis=1, ddof=0)
    absolute_error = np.abs(combined["observed_pic50"] - combined["ensemble_prediction"])
    if "fold" not in combined or combined["fold"].nunique() < 2:
        raise Unavailable("uncertainty calibration requires at least two sealed outer OOF folds")
    rows: list[dict[str, Any]] = []
    levels = [float(item) for item in spec.get("coverage_levels", [0.5, 0.8, 0.9, 0.95])]
    if any(level <= 0 or level >= 1 for level in levels):
        raise WorkerError("coverage levels must lie strictly between zero and one")
    for level in levels:
        lower_values = np.full(len(combined), np.nan)
        upper_values = np.full(len(combined), np.nan)
        for fold in sorted(combined["fold"].unique()):
            evaluation = combined["fold"].eq(fold).to_numpy()
            calibration = ~evaluation
            quantile = float(np.quantile(absolute_error[calibration], level))
            lower_values[evaluation] = combined.loc[evaluation, "ensemble_prediction"] - quantile
            upper_values[evaluation] = combined.loc[evaluation, "ensemble_prediction"] + quantile
        covered = (combined["observed_pic50"].to_numpy() >= lower_values) & (
            combined["observed_pic50"].to_numpy() <= upper_values
        )
        combined[f"lower_{int(level * 100)}"] = lower_values
        combined[f"upper_{int(level * 100)}"] = upper_values
        rows.append(
            {
                "nominal_coverage": level,
                "empirical_cross_conformal_coverage": float(np.mean(covered)),
                "mean_interval_width": float(np.mean(upper_values - lower_values)),
                "calibration_excludes_evaluation_outer_fold": True,
            }
        )
    calibration_path = directory / "uncertainty_calibration.parquet"
    prediction_path = directory / "ensemble_oof_predictions.parquet"
    _write_parquet(calibration_path, pd.DataFrame(rows))
    _write_parquet(prediction_path, combined)
    metrics = _regression_metrics(
        combined["observed_pic50"].to_numpy(), combined["ensemble_prediction"].to_numpy()
    )
    metrics.update(
        selection_score=metrics["mae"],
        members=len(prediction_columns),
        calibration_scope="outer-fold cross-conformal train-only",
    )
    return _write_unit(
        directory,
        unit_id=unit_id,
        operation="uncertainty_calibration",
        spec=spec,
        metrics=metrics,
        artifacts=[_binding(prediction_path, "oof_predictions"), _binding(calibration_path, "calibration")],
        limitations=[
            "Each train-only outer fold is evaluated with residual quantiles calibrated on other outer folds; this is not external calibration."
        ],
    )


def _censored_unit(
    repo: Path,
    prepared: Path,
    directory: Path,
    unit_id: str,
    spec: dict[str, Any],
    workers: int,
) -> dict[str, Any]:
    treatment = str(spec.get("treatment", "interval_likelihood"))
    if treatment != "interval_likelihood":
        raise WorkerError("only treatment=interval_likelihood is scientifically implemented")
    try:
        from menin_discovery.research_modeling import CensoredGaussianRidge
    except ImportError as error:
        raise Unavailable("CensoredGaussianRidge package import is unavailable") from error
    routing = pq.read_table(prepared / "censored_train_routing.parquet").to_pandas()
    frame = _load_global_features(repo, routing)
    features = [column for column in GLOBAL_FEATURES if column in frame]
    x = _finite_numeric(frame, features)
    groups = frame["scaffold_group_id"].fillna(frame["structure_id"]).astype(str).to_numpy()
    # Arrow-backed pandas columns and pandas copy-on-write can expose read-only
    # NumPy views.  The exact-row bound normalization below is intentionally
    # mutating, so own every censoring array explicitly before assignment.
    relation = frame["potency_relation_pic50"].astype(str).to_numpy(copy=True)
    point = pd.to_numeric(frame["potency_pic50_point"], errors="coerce").to_numpy(dtype=float, copy=True)
    lower = pd.to_numeric(frame["potency_pic50_lower_bound"], errors="coerce").to_numpy(
        dtype=float, copy=True
    )
    upper = pd.to_numeric(frame["potency_pic50_upper_bound"], errors="coerce").to_numpy(
        dtype=float, copy=True
    )
    exact = relation == "="
    lower[exact] = point[exact]
    upper[exact] = point[exact]
    predictions = np.full(len(frame), np.nan)
    sigmas = np.full(len(frame), np.nan)
    folds = int(spec.get("outer_folds", 5))
    for _fold, (fit_index, eval_index) in enumerate(GroupKFold(n_splits=folds).split(x, groups=groups)):
        if set(groups[fit_index]) & set(groups[eval_index]):
            raise WorkerError("scaffold leakage in censored sensitivity")
        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        fit_x = scaler.fit_transform(imputer.fit_transform(x.iloc[fit_index]))
        eval_x = scaler.transform(imputer.transform(x.iloc[eval_index]))
        model = CensoredGaussianRidge(
            alpha=float(spec.get("alpha", 10.0)), maxiter=int(spec.get("max_iter", 400))
        )
        model.fit(fit_x, lower[fit_index], upper[fit_index])
        predictions[eval_index] = model.predict(eval_x)
        sigmas[eval_index] = float(getattr(model, "sigma_", 1.0))
    exact_eval = exact & np.isfinite(point) & np.isfinite(predictions)
    if exact_eval.sum() < 2:
        raise WorkerError("censored sensitivity produced no exact-point diagnostic")
    metrics = _regression_metrics(point[exact_eval], predictions[exact_eval])
    violations = np.zeros(len(frame), dtype=float)
    finite_lower = np.isfinite(lower)
    finite_upper = np.isfinite(upper)
    violations[finite_lower] += np.maximum(lower[finite_lower] - predictions[finite_lower], 0)
    violations[finite_upper] += np.maximum(predictions[finite_upper] - upper[finite_upper], 0)
    metrics.update(
        interval_violation_mae=float(np.nanmean(violations)),
        censored_observations=int((~exact).sum()),
        exact_observations=int(exact.sum()),
        selection_score=float(np.nanmean(violations) + metrics["mae"]),
        endpoint="censored sensitivity; not pooled into exact primary model",
    )
    output = frame[["structure_id", "scaffold_group_id", "potency_relation_pic50"]].copy()
    output["lower_bound_pic50"] = lower
    output["upper_bound_pic50"] = upper
    output["predicted_pic50"] = predictions
    output["estimated_sigma"] = sigmas
    prediction_path = directory / "censored_oof_predictions.parquet"
    _write_parquet(prediction_path, output)
    return _write_unit(
        directory,
        unit_id=unit_id,
        operation="censored_sensitivity",
        spec=spec,
        metrics=metrics,
        artifacts=[_binding(prediction_path, "censored_oof_predictions")],
        limitations=["This is a sensitivity track; censored and exact records remain distinguishable."],
    )


def _mmp_unit(
    repo: Path,
    base: Path,
    results_root: Path,
    directory: Path,
    unit_id: str,
    spec: dict[str, Any],
) -> dict[str, Any]:
    analysis_id = str(spec.get("analysis_id", "matched_pairs_and_activity_cliffs"))
    if analysis_id not in {"matched_pairs", "activity_cliff_residuals", "matched_pairs_and_activity_cliffs"}:
        raise WorkerError(f"unsupported MMP analysis_id: {analysis_id}")
    predictions = _source_prediction(results_root, spec, base)
    if not {"observed_pic50", "predicted_pic50"}.issubset(predictions):
        raise Unavailable("MMP analysis requires quantitative OOF predictions")
    prediction_map = predictions.drop_duplicates("structure_id").set_index("structure_id")["predicted_pic50"]
    effects = pq.read_table(repo / DEFAULT_MMP).to_pandas()
    effects = effects.loc[
        effects["structure_id_a"].isin(prediction_map.index)
        & effects["structure_id_b"].isin(prediction_map.index)
    ].copy()
    if len(effects) < int(spec.get("minimum_pair_support", 50)):
        raise Unavailable("too few matched-pair OOF structures")
    effects["predicted_delta_pic50_b_minus_a"] = effects["structure_id_b"].map(prediction_map) - effects[
        "structure_id_a"
    ].map(prediction_map)
    observed = pd.to_numeric(effects["delta_pic50_b_minus_a"], errors="coerce").to_numpy(dtype=float)
    predicted = effects["predicted_delta_pic50_b_minus_a"].to_numpy(dtype=float)
    metrics = {
        "pairs": len(effects),
        "delta_mae": float(np.mean(np.abs(observed - predicted))),
        "delta_spearman": float(np.nan_to_num(spearmanr(observed, predicted).statistic)),
        "direction_accuracy": float(np.mean(np.sign(observed) == np.sign(predicted))),
        "observed_activity_cliffs": int(
            effects.get("activity_cliff_ge_1_pic50", pd.Series(False, index=effects.index)).sum()
        ),
        "selection_score": None,
        "analysis_id": analysis_id,
    }
    output_path = directory / "mmp_cliff_residuals.parquet"
    _write_parquet(output_path, effects)
    return _write_unit(
        directory,
        unit_id=unit_id,
        operation="mmp_cliff_residual",
        spec=spec,
        metrics=metrics,
        artifacts=[_binding(output_path, "mmp_cliff_residuals")],
        limitations=["Matched molecular pairs are training-only associations, not causal evidence."],
    )


def _chemprop_ensemble_unit(
    base: Path,
    results_root: Path,
    directory: Path,
    unit_id: str,
    spec: dict[str, Any],
) -> dict[str, Any]:
    tree = _source_prediction(results_root, spec, base)
    chemprop_path = base / "analysis" / "outer_oof_predictions.parquet"
    if not chemprop_path.is_file():
        raise Unavailable("base Chemprop OOF predictions unavailable")
    chemprop = pd.read_parquet(chemprop_path)
    if "model_id" in chemprop:
        chemprop = chemprop.loc[chemprop["model_id"].astype(str).str.contains("chemprop", case=False)]
    if chemprop.empty:
        raise Unavailable("five-fold complete Chemprop OOF predictions unavailable")
    required = ["structure_id", "observed_pic50", "predicted_pic50"]
    left = tree[required + (["scaffold_group_id"] if "scaffold_group_id" in tree else [])].drop_duplicates(
        "structure_id"
    )
    right = chemprop[required].drop_duplicates("structure_id")
    combined = left.merge(
        right, on=["structure_id", "observed_pic50"], suffixes=("_tree", "_chemprop"), validate="one_to_one"
    )
    if len(combined) != EXACT_ROWS:
        raise Unavailable("Chemprop/tree OOF alignment is not five-fold complete")
    weight = float(spec.get("tree_weight", 0.5))
    combined["predicted_pic50"] = (
        weight * combined["predicted_pic50_tree"] + (1.0 - weight) * combined["predicted_pic50_chemprop"]
    )
    combined["ensemble_sd"] = combined[["predicted_pic50_tree", "predicted_pic50_chemprop"]].std(
        axis=1, ddof=0
    )
    metrics = _regression_metrics(
        combined["observed_pic50"].to_numpy(), combined["predicted_pic50"].to_numpy()
    )
    metrics.update(selection_score=metrics["mae"], tree_weight=weight, members=2)
    output_path = directory / "oof_predictions.parquet"
    _write_parquet(output_path, combined)
    return _write_unit(
        directory,
        unit_id=unit_id,
        operation="chemprop_ensemble",
        spec=spec,
        metrics=metrics,
        artifacts=[_binding(output_path, "oof_predictions")],
        limitations=["Chemprop ensemble is accepted only with complete train-only outer OOF coverage."],
    )


def _finalist_unit(
    base: Path,
    prepared: Path,
    results_root: Path,
    directory: Path,
    unit_id: str,
    spec: dict[str, Any],
    workers: int,
) -> dict[str, Any]:
    candidate = _candidate(spec)
    frame, features = _load_exact(base, prepared, str(candidate["feature_set"]))
    model = _model(candidate, workers, int(spec.get("seed", SEED)))
    model = _fit_regression_model(model, frame[features], frame["target_pic50"].to_numpy(dtype=float))
    model_path = directory / "model.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    schema_path = directory / "feature_schema.json"
    _write_json(
        schema_path,
        {
            "schema_version": SCHEMA,
            "features": features,
            "feature_set": candidate["feature_set"],
            "target": "exact quantitative pIC50",
            "training_structures": len(frame),
            "label_blind_features": True,
        },
        "feature_schema_sha256",
    )
    predictions = _predict_model(model, frame.iloc[: min(16, len(frame))][features])
    smoke_path = directory / "inference_smoke.json"
    _write_json(
        smoke_path,
        {
            "status": "passed",
            "rows": len(predictions),
            "all_finite": bool(np.isfinite(predictions).all()),
            "model_loadable": bool(joblib.load(model_path) is not None),
        },
        "inference_smoke_sha256",
    )
    source_metrics: dict[str, Any] = {}
    source_artifacts: list[dict[str, Any]] = []
    source_id = spec.get("source_unit_id")
    if not source_id:
        raise WorkerError("finalist refit requires a bound source_unit_id with full outer OOF evidence")
    source_path = _unit_dir(results_root, str(source_id)) / "unit.json"
    source = _read_self_json(source_path, "unit_json_sha256")
    if source.get("status") != "passed" or source.get("operation") not in REGRESSION_OPERATIONS:
        raise WorkerError("finalist source is not a passed exact quantitative regression unit")
    source_candidate = _candidate(source.get("unit_spec", {}))
    for key in ("candidate_id", "engine", "feature_set", "params"):
        if source_candidate.get(key) != candidate.get(key):
            raise WorkerError(f"finalist candidate mismatch with source for {key}")
    source_oof = next(
        (artifact for artifact in source.get("artifacts", []) if artifact["role"] == "oof_predictions"),
        None,
    )
    if source_oof is None:
        raise WorkerError("finalist source has no OOF prediction artifact")
    _verify_binding(source_oof)
    source_predictions = pd.read_parquet(source_oof["path"])
    if (
        len(source_predictions) != EXACT_ROWS
        or source_predictions["structure_id"].nunique() != EXACT_ROWS
        or source_predictions["structure_id"].duplicated().any()
        or set(source_predictions["stage"].astype(str)) != {"outer"}
    ):
        raise WorkerError("finalist source does not provide full 18,801-row outer OOF coverage exactly once")
    source_metrics = dict(source.get("metrics") or {})
    source_artifacts = [
        _binding(source_path, "selection_source_unit"),
        dict(source_oof, role="selection_source_oof"),
    ]
    metrics = {
        "training_structures": len(frame),
        "feature_count": len(features),
        "selection_score": source_metrics.get("selection_score"),
        "source_oof_mae": source_metrics.get("mae"),
        "source_oof_rmse": source_metrics.get("rmse"),
        "source_oof_spearman": source_metrics.get("spearman"),
        "refit_uses_all_exact_train": True,
    }
    return _write_unit(
        directory,
        unit_id=unit_id,
        operation="finalist_refit_artifact",
        spec=spec,
        metrics=metrics,
        artifacts=[
            _binding(model_path, "model"),
            _binding(schema_path, "feature_schema"),
            _binding(smoke_path, "inference_smoke"),
            *source_artifacts,
        ],
        limitations=[
            "The persisted finalist is refit on all train data; its performance comes only from bound OOF source metrics."
        ],
    )


def _run_unit(args: argparse.Namespace) -> dict[str, Any]:
    repo = Path(args.repo_root).resolve()
    base = Path(args.base_campaign_root).resolve()
    prepared = Path(args.prepared_root).resolve()
    results_root = Path(args.output_root).resolve()
    _validate_prepared(prepared, full=False)
    spec = json.loads(args.unit_spec)
    if not isinstance(spec, dict):
        raise WorkerError("unit spec must decode to a JSON object")
    operation = str(spec.get("operation", spec.get("kind", "")))
    if operation not in OPERATIONS:
        raise WorkerError(f"unsupported operation {operation!r}; supported={sorted(OPERATIONS)}")
    if spec.get("source_partition", "train") != "train":
        raise WorkerError("only source_partition=train is allowed")
    if spec.get("validation_labels_opened", False) or spec.get("test_labels_opened", False):
        raise WorkerError("repository validation/test label access is forbidden")
    directory = _unit_dir(results_root, args.unit_id)
    directory.mkdir(parents=True, exist_ok=True)
    existing = _existing_unit(directory / "unit.json", spec)
    if existing is not None:
        return existing
    if any(directory.iterdir()):
        shutil.rmtree(directory)
        directory.mkdir(parents=True)
    if operation in REGRESSION_OPERATIONS:
        return _regression_unit(base, prepared, directory, args.unit_id, operation, spec, args.workers)
    if operation == "broad_wt_auxiliary":
        return _broad_unit(repo, prepared, directory, args.unit_id, spec, args.workers)
    if operation == "assay_quality_strata":
        return _assay_unit(base, prepared, results_root, directory, args.unit_id, spec)
    if operation == "uncertainty_calibration":
        return _uncertainty_unit(base, results_root, directory, args.unit_id, spec)
    if operation == "censored_sensitivity":
        return _censored_unit(repo, prepared, directory, args.unit_id, spec, args.workers)
    if operation == "mmp_cliff_residual":
        return _mmp_unit(repo, base, results_root, directory, args.unit_id, spec)
    if operation == "chemprop_ensemble":
        return _chemprop_ensemble_unit(base, results_root, directory, args.unit_id, spec)
    if operation == "finalist_refit_artifact":
        return _finalist_unit(base, prepared, results_root, directory, args.unit_id, spec, args.workers)
    raise WorkerError(f"operation dispatch missing: {operation}")


def _validated_nested_composite(
    frames: Sequence[pd.DataFrame],
) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    if not frames:
        return pd.DataFrame(), None
    composite = pd.concat(frames, ignore_index=True)
    if (
        len(composite) != EXACT_ROWS
        or composite["structure_id"].nunique() != EXACT_ROWS
        or composite["structure_id"].duplicated().any()
        or set(composite["stage"].astype(str)) != {"outer"}
        or composite["fold"].nunique() != 5
    ):
        raise WorkerError(
            "nested model-selection composite must be five disjoint outer folds covering exactly 18,801 structures"
        )
    metrics = _regression_metrics(
        composite["observed_pic50"].to_numpy(), composite["predicted_pic50"].to_numpy()
    )
    metrics.update(
        estimate_scope="unbiased_internal_nested_model_selection_estimate",
        candidate_may_differ_by_outer_fold=True,
        outer_folds=5,
    )
    return composite, metrics


def _analyze(args: argparse.Namespace) -> dict[str, Any]:
    prepared = Path(args.prepared_root).resolve()
    results_root = Path(args.results_root).resolve()
    output = Path(args.output_root).resolve()
    _validate_prepared(prepared, full=True)
    validation_path = output / "validation.json"
    if validation_path.is_file():
        return _read_self_json(validation_path, "validation_sha256")
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise WorkerError("analysis output must be empty or already validated")
    units: list[dict[str, Any]] = []
    rejected: list[dict[str, str]] = []
    for path in sorted((results_root / "units").glob("*/unit.json")):
        try:
            document = _read_self_json(path, "unit_json_sha256")
            if document.get("status") != "passed":
                rejected.append({"path": str(path), "reason": f"status={document.get('status')}"})
                continue
            scope = document.get("scientific_scope", {})
            if scope.get("source_partition") != "train":
                raise WorkerError("unit is not train-only")
            if scope.get("repository_validation_labels_opened") or scope.get("repository_test_labels_opened"):
                raise WorkerError("sealed outcome flag failed")
            if scope.get("broad_fixed_dose_pooled_into_pic50"):
                raise WorkerError("broad fixed-dose endpoint was pooled")
            for artifact in document.get("artifacts", []):
                _verify_binding(artifact)
            units.append(document)
        except Exception as error:
            rejected.append({"path": str(path), "reason": str(error)})
    if not units:
        raise WorkerError("no valid passed v3 units found")

    ranking_rows: list[dict[str, Any]] = []
    model_cards: list[dict[str, Any]] = []
    relationship_rows: list[dict[str, Any]] = []
    finalists: list[dict[str, Any]] = []
    operation_counts: Counter[str] = Counter()
    nested_frames: list[pd.DataFrame] = []
    for unit in units:
        operation_counts[unit["operation"]] += 1
        metrics = unit.get("metrics", {})
        ranking_rows.append(
            {
                "unit_id": unit["unit_id"],
                "candidate_id": unit.get("candidate_id"),
                "operation": unit["operation"],
                "selection_score": metrics.get("selection_score"),
                "mae": metrics.get("mae"),
                "rmse": metrics.get("rmse"),
                "spearman": metrics.get("spearman"),
                "fraction_within_0p5": metrics.get("fraction_within_0p5"),
                "roc_auc": metrics.get("roc_auc"),
                "pr_auc": metrics.get("pr_auc"),
                "mcc": metrics.get("mcc"),
                "brier": metrics.get("brier"),
                "evaluation_stage": metrics.get("evaluation_stage"),
                "evaluation_structures": metrics.get("n"),
                "full_outer_oof_eligible": bool(
                    metrics.get("evaluation_stage") == "outer" and metrics.get("n") == EXACT_ROWS
                ),
            }
        )
        model_cards.append(
            {
                "unit_id": unit["unit_id"],
                "candidate_id": unit.get("candidate_id"),
                "operation": unit["operation"],
                "metrics": metrics,
                "scientific_scope": unit["scientific_scope"],
                "limitations": unit.get("limitations", []),
                "intended_use": "train-only model selection and biological-hypothesis generation",
                "external_or_prospective_validation_completed": False,
            }
        )
        if unit["operation"] == "nested_robustness" and metrics.get("evaluation_stage") == "outer":
            oof_artifact = next(
                (item for item in unit.get("artifacts", []) if item["role"] == "oof_predictions"),
                None,
            )
            if oof_artifact is not None:
                nested = pd.read_parquet(oof_artifact["path"])
                nested["selected_candidate_id"] = unit.get("candidate_id")
                nested["source_unit_id"] = unit["unit_id"]
                nested_frames.append(nested)
        relationship_eligible = bool(
            unit["operation"] in {"interaction_stability", "feature_ablation"}
            and metrics.get("evaluation_stage") == "outer"
            and metrics.get("n") == EXACT_ROWS
        )
        for artifact in unit.get("artifacts", []):
            if artifact["role"] in {"feature_importance", "directional_associations"}:
                if not relationship_eligible:
                    continue
                frame = pd.read_parquet(artifact["path"])
                if artifact["role"] == "feature_importance":
                    summary = frame.groupby(["feature", "biological_process_hypothesis"], as_index=False).agg(
                        mean_importance=("importance", "mean"),
                        sd_importance=("importance", "std"),
                        fold_selection_frequency=("fold", "nunique"),
                    )
                    for row in summary.to_dict("records"):
                        relationship_rows.append(
                            {
                                "unit_id": unit["unit_id"],
                                "evidence_type": "outer_fold_model_importance_variation",
                                **row,
                                "causal_interpretation_allowed": False,
                            }
                        )
                else:
                    for row in frame.to_dict("records"):
                        relationship_rows.append(
                            {
                                "unit_id": unit["unit_id"],
                                "evidence_type": "directional_association",
                                **row,
                            }
                        )
        if unit["operation"] == "finalist_refit_artifact" or (
            unit["operation"] == "broad_wt_auxiliary"
            and any(item["role"] == "model" for item in unit["artifacts"])
        ):
            required_roles = {"model", "feature_schema", "inference_smoke"}
            roles = {item["role"] for item in unit["artifacts"]}
            if unit["operation"] == "broad_wt_auxiliary":
                required_roles |= {"calibration", "enrichment"}
            if not required_roles.issubset(roles):
                rejected.append(
                    {
                        "path": unit["unit_id"],
                        "reason": f"finalist missing roles {sorted(required_roles - roles)}",
                    }
                )
            else:
                finalists.append(
                    {
                        "unit_id": unit["unit_id"],
                        "operation": unit["operation"],
                        "candidate_id": unit.get("candidate_id"),
                        "metrics": metrics,
                        "artifacts": [item for item in unit["artifacts"] if item["role"] in required_roles],
                    }
                )
    rankings = pd.DataFrame(ranking_rows)
    regression = rankings.loc[
        rankings["operation"].isin(
            {"repeated_seed_tree", "feature_ablation", "interaction_stability", "chemprop_ensemble"}
        )
    ].copy()
    regression = regression.loc[regression["full_outer_oof_eligible"].fillna(False)]
    regression = regression.loc[pd.to_numeric(regression["selection_score"], errors="coerce").notna()]
    regression = regression.sort_values(["selection_score", "unit_id"], kind="stable")
    broad = rankings.loc[rankings["operation"].eq("broad_wt_auxiliary")].copy()
    broad = broad.sort_values(["selection_score", "unit_id"], kind="stable")
    nested_composite, nested_composite_metrics = _validated_nested_composite(nested_frames)
    nested_path = output / "nested_model_selection_oof.parquet"
    if nested_composite_metrics is not None:
        _write_parquet(nested_path, nested_composite)
    decision_ledger = {
        "schema_version": SCHEMA,
        "created_at": _now(),
        "primary_endpoint": "exact quantitative pIC50",
        "best_train_only_exact_unit": regression.iloc[0]["unit_id"] if len(regression) else None,
        "nested_model_selection_composite_metrics": nested_composite_metrics,
        "best_train_only_broad_auxiliary_unit": broad.iloc[0]["unit_id"] if len(broad) else None,
        "broad_surface": {
            "full_structures": 339_373,
            "train_partition_structures": 265_625,
            "pooled_into_exact_pic50": False,
            "role": "separate auxiliary classification/pretraining",
        },
        "selection_rule": (
            "The five-fold nested composite is the primary unbiased internal model-selection estimate. "
            "Repeated full OOF units measure finalist recipe/seed stability only; broad score is negative PR-AUC."
        ),
        "locked_repository_validation_or_test_used": False,
        "predictive_superiority_established": False,
        "next_gate": "freeze finalist and prespecify external or repository locked evaluation",
    }
    model_cards_path = output / "model_cards.json"
    ledger_path = output / "decision_ledger.json"
    relationships_path = output / "feature_relationships.json"
    finalists_path = output / "final_models_manifest.json"
    _write_json(
        model_cards_path,
        {
            "schema_version": SCHEMA,
            "created_at": _now(),
            "models": model_cards,
        },
        "model_cards_sha256",
    )
    _write_json(ledger_path, decision_ledger, "decision_ledger_sha256")
    _write_json(
        relationships_path,
        {
            "schema_version": SCHEMA,
            "created_at": _now(),
            "relationships": relationship_rows,
            "interpretation": (
                "Outer-fold variation and directional associations are discovery-grade hypotheses only. "
                "They are not permutation/SHAP evidence, do not control multiplicity, and do not prove "
                "causality or direct channel binding."
            ),
        },
        "feature_relationships_sha256",
    )
    _write_json(
        finalists_path,
        {
            "schema_version": SCHEMA,
            "created_at": _now(),
            "finalists": finalists,
            "all_models_have_inference_smoke": all(
                any(item["role"] == "inference_smoke" for item in finalist["artifacts"])
                for finalist in finalists
            ),
        },
        "final_models_manifest_sha256",
    )
    rankings_path = output / "unit_rankings.parquet"
    _write_parquet(rankings_path, rankings)
    lines = [
        "# Extended train-only hERG discovery campaign",
        "",
        f"- Valid passed units: {len(units)}.",
        f"- Rejected/incomplete units: {len(rejected)}.",
        "- Primary endpoint: exact quantitative pIC50 on 18,801 train structures.",
        "- Broad auxiliary endpoint: 339,373 structures total, but only the 265,625 train structures are read.",
        "- Broad fixed-dose binary labels are never pooled with exact pIC50.",
        "- Repository validation and test labels remain sealed.",
        "- Feature relationships are hypotheses and do not establish causality or direct binding mechanisms.",
    ]
    if len(regression):
        best = regression.iloc[0]
        lines.extend(
            [
                "",
                "## Current train-only quantitative leader",
                "",
                f"- Unit: {best['unit_id']}.",
                f"- OOF MAE: {best['mae']}.",
                f"- OOF RMSE: {best['rmse']}.",
                f"- OOF Spearman: {best['spearman']}.",
            ]
        )
    if nested_composite_metrics is not None:
        lines.extend(
            [
                "",
                "## Primary unbiased internal nested model-selection estimate",
                "",
                f"- MAE: {nested_composite_metrics['mae']}.",
                f"- RMSE: {nested_composite_metrics['rmse']}.",
                f"- Spearman: {nested_composite_metrics['spearman']}.",
                "- Each outer fold selected its recipe using only that outer fold's inner training data.",
                "- Repeated full OOF recipes below are stability evidence, not nested selection estimates.",
            ]
        )
    if len(broad):
        best = broad.iloc[0]
        lines.extend(
            [
                "",
                "## Current broad-WT auxiliary leader",
                "",
                f"- Unit: {best['unit_id']}.",
                f"- PR-AUC: {best['pr_auc']}.",
                f"- ROC-AUC: {best['roc_auc']}.",
            ]
        )
    analysis_path = output / "analysis.md"
    _atomic(analysis_path, lambda temporary: temporary.write_text("\n".join(lines) + "\n", "utf-8"))
    validation = {
        "schema_version": SCHEMA,
        "status": "passed",
        "created_at": _now(),
        "valid_units": len(units),
        "rejected_units": rejected,
        "operation_counts": dict(operation_counts),
        "repository_validation_labels_opened": False,
        "repository_test_labels_opened": False,
        "broad_fixed_dose_pooled_into_pic50": False,
        "finalist_models": len(finalists),
    }
    _write_json(validation_path, validation, "validation_sha256")
    manifest_artifacts = [
        _binding(validation_path, "validation"),
        _binding(analysis_path, "analysis"),
        _binding(model_cards_path, "model_cards"),
        _binding(ledger_path, "decision_ledger"),
        _binding(relationships_path, "feature_relationships"),
        _binding(finalists_path, "final_models_manifest"),
        _binding(rankings_path, "unit_rankings"),
    ]
    if nested_composite_metrics is not None:
        manifest_artifacts.append(_binding(nested_path, "nested_model_selection_oof"))
    _write_json(
        output / "manifest.json",
        {
            "schema_version": SCHEMA,
            "created_at": _now(),
            "status": "passed",
            "artifacts": manifest_artifacts,
        },
        "manifest_sha256",
    )
    return validation


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="bind completed v1 train-only data and create v3 routing")
    prepare.add_argument("--repo-root", required=True)
    prepare.add_argument("--base-campaign-root", required=True)
    prepare.add_argument("--output-root", required=True, help="v3 prepared output directory")
    prepare.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))

    run = commands.add_parser("run-unit", help="run or resume one immutable v3 scientific unit")
    run.add_argument("--repo-root", required=True)
    run.add_argument("--base-campaign-root", required=True)
    run.add_argument("--prepared-root", required=True)
    run.add_argument("--output-root", required=True, help="v3 results root containing units/")
    run.add_argument("--unit-id", required=True)
    run.add_argument("--unit-spec", required=True, help="JSON object containing operation and candidate")
    run.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))

    analyze = commands.add_parser("analyze", help="aggregate validated v3 units into final deliverables")
    analyze.add_argument("--repo-root", required=True)
    analyze.add_argument("--base-campaign-root", required=True)
    analyze.add_argument("--prepared-root", required=True)
    analyze.add_argument("--results-root", required=True, help="v3 root containing units/")
    analyze.add_argument("--output-root", required=True, help="v3 analysis output directory")
    analyze.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.workers < 1:
            raise WorkerError("workers must be positive")
        if args.command == "prepare":
            result = _prepare(args)
        elif args.command == "run-unit":
            result = _run_unit(args)
        elif args.command == "analyze":
            result = _analyze(args)
        else:
            parser.error("unknown command")
            return 2
        print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
        return 0
    except Unavailable as error:
        print(json.dumps({"status": "unavailable", "reason": str(error)}, sort_keys=True))
        return 3
    except Exception as error:
        print(
            json.dumps(
                {"status": "failed", "error_type": type(error).__name__, "reason": str(error)}, sort_keys=True
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
