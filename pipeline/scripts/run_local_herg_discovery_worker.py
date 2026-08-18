#!/usr/bin/env python3
"""Resumable, train-only worker for the local hERG discovery campaign.

The worker deliberately separates preparation, one-model fit units, similarity
units, Chemprop export, and aggregation.  Every outcome read from the platform
observation store is predicate-pushed to the repository ``train`` partition.
The words "outer" and "inner" below therefore refer to nested folds made inside
that training partition; they never refer to the repository validation or test
partitions.

This is a scientific screening worker, not a superiority-claiming benchmark.
In particular, the quantitative target currently has
``wild_type_or_unspecified`` scope and substantial unresolved assay metadata.
Those limitations are carried into every manifest and final interpretation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

SCHEMA_VERSION = "platform-local-herg-discovery-worker/1.0"
DEFAULT_MATRIX_ROOT = Path("research/local_runs/herg_quantitative_feature_matrix_v1")
DEFAULT_OBSERVATIONS = Path(
    "research/data/platform/processed/herg_hierarchy/v1_6_training_surfaces/"
    "herg_training_observations.parquet"
)
DEFAULT_MMP_ROOT = Path("research/data/platform/processed/herg_hierarchy/v1_5_mmp_analysis")
MAX_ABSOLUTE_NUMERIC_FEATURE = 1.0e30
MORGAN_BITS = 2048
MACCS_BITS = 167
SEED = 20260811
CHEMPROP_MODEL_ID = "chemprop_dmpnn"


class DiscoveryWorkerError(RuntimeError):
    """Raised when a worker action violates its scientific or file contract."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _atomic_replace(path: Path, writer: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        writer(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_replace(path, lambda temporary: temporary.write_text(_canonical_json(value), "utf-8"))


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    _atomic_replace(
        path,
        lambda temporary: frame.to_parquet(temporary, index=False, compression="zstd", engine="pyarrow"),
    )


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    _atomic_replace(path, lambda temporary: frame.to_csv(temporary, index=False))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _binding(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }
    if path.suffix == ".parquet":
        metadata = pq.read_metadata(path)
        result["rows"] = metadata.num_rows
        result["columns"] = metadata.num_columns
    return result


def _validate_binding(binding: dict[str, Any]) -> None:
    path = Path(str(binding["path"]))
    if not path.is_file() or path.stat().st_size != int(binding["bytes"]):
        raise DiscoveryWorkerError(f"bound file missing or changed size: {path}")
    if _sha256(path) != binding["sha256"]:
        raise DiscoveryWorkerError(f"bound file hash changed: {path}")
    if path.suffix == ".parquet" and pq.read_metadata(path).num_rows != int(binding["rows"]):
        raise DiscoveryWorkerError(f"bound parquet row count changed: {path}")


def _mode(values: pd.Series) -> str:
    cleaned = [str(value) for value in values.dropna()]
    if not cleaned:
        return "unresolved"
    counts = Counter(cleaned)
    maximum = max(counts.values())
    return sorted(value for value, count in counts.items() if count == maximum)[0]


def _unpack_binary_fingerprint(value: bytes, n_bits: int) -> np.ndarray:
    """Unpack RDKit BinaryText using the explicit little-endian bit contract."""
    expected_bytes = (n_bits + 7) // 8
    if not isinstance(value, bytes) or len(value) != expected_bytes:
        raise DiscoveryWorkerError(
            f"fingerprint byte length mismatch: expected {expected_bytes}, got "
            f"{len(value) if isinstance(value, bytes) else type(value).__name__}"
        )
    return np.unpackbits(np.frombuffer(value, dtype=np.uint8), bitorder="little")[:n_bits]


def _unpack_fingerprint_column(values: Sequence[bytes], n_bits: int) -> np.ndarray:
    result = np.empty((len(values), n_bits), dtype=np.uint8)
    for index, value in enumerate(values):
        result[index] = _unpack_binary_fingerprint(value, n_bits)
    return result


def _is_shape(column: str) -> bool:
    return any(
        term in column
        for term in (
            "pmi",
            "npr",
            "asphericity",
            "eccentricity",
            "inertial_shape",
            "radius_of_gyration",
            "spherocity",
            "pbf",
            "heavy_pair_distance",
            "heavy_contact_density",
            "retained_pairwise_rmsd",
        )
    )


def _is_polarity_charge(column: str) -> bool:
    return any(
        term in column
        for term in (
            "formal_charge",
            "basic_site",
            "acidic_site",
            "tautomer",
            "polar_radial_exposure",
            "internal_polar_contact",
            "gasteiger_dipole",
            "absolute_charge_radius",
            "sasa",
        )
    )


def _is_energy_flexibility(column: str) -> bool:
    return any(
        term in column
        for term in (
            "rotatable_bond",
            "embedded_conformer_count",
            "retained_conformer_count",
            "unconverged_retained_count",
            "energy_range_kcal_mol",
            "effective_conformer_count",
            "dominant_conformer_weight",
            "energy_polar_exposure_correlation",
        )
    )


def _descriptor_subgroups(columns: Sequence[str]) -> dict[str, list[str]]:
    functional = [column for column in columns if "__fr_" in column]
    polarity_terms = (
        "TPSA",
        "PartialCharge",
        "PEOE_VSA",
        "SlogP_VSA",
        "SMR_VSA",
        "HAcceptors",
        "HDonors",
        "NHOHCount",
        "NOCount",
        "MolLogP",
    )
    polarity = [column for column in columns if any(term in column for term in polarity_terms)]
    topology_terms = (
        "Chi",
        "Kappa",
        "Balaban",
        "Bertz",
        "Ring",
        "FractionCSP3",
        "HeavyAtom",
        "Rotatable",
        "SPS",
    )
    topology = [column for column in columns if any(term in column for term in topology_terms)]
    return {
        "rdkit2d_functional_groups": sorted(functional),
        "rdkit2d_polarity_lipophilicity": sorted(polarity),
        "rdkit2d_topology_aromaticity": sorted(topology),
    }


def _targeted_interactions(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    definitions = {
        "interaction__logp_x_tpsa": (
            "rdkit2d__MolLogP",
            "rdkit2d__TPSA",
            "lipophilic partitioning multiplied by polar surface burden",
        ),
        "interaction__logp_x_aromatic_rings": (
            "rdkit2d__MolLogP",
            "rdkit2d__NumAromaticRings",
            "lipophilicity multiplied by aromatic scaffold burden",
        ),
        "interaction__abs_charge_x_logp": (
            "f3d__formal_charge",
            "rdkit2d__MolLogP",
            "absolute formal charge multiplied by lipophilicity",
        ),
        "interaction__abs_charge_x_tpsa": (
            "f3d__formal_charge",
            "rdkit2d__TPSA",
            "absolute formal charge multiplied by polar surface burden",
        ),
        "interaction__basic_sites_x_aromatic_rings": (
            "f3d__basic_site_proxy_count",
            "rdkit2d__NumAromaticRings",
            "basic-center abundance multiplied by aromatic burden",
        ),
        "interaction__rotors_x_aromatic_rings": (
            "f3d__rotatable_bond_count",
            "rdkit2d__NumAromaticRings",
            "conformational flexibility multiplied by aromatic burden",
        ),
        "interaction__dipole_x_logp": (
            "f3d__ensemble_gasteiger_dipole_proxy_eA__mean",
            "rdkit2d__MolLogP",
            "charge-separation proxy multiplied by lipophilicity",
        ),
        "interaction__charge_radius_x_abs_charge": (
            "f3d__ensemble_absolute_charge_radius_A__mean",
            "f3d__formal_charge",
            "charge radius multiplied by absolute formal charge",
        ),
        "interaction__polar_exposure_x_logp": (
            "f3d__ensemble_polar_radial_exposure__mean",
            "rdkit2d__MolLogP",
            "three-dimensional polar exposure multiplied by lipophilicity",
        ),
        "interaction__polar_contacts_x_logp": (
            "f3d__ensemble_internal_polar_contact_count__mean",
            "rdkit2d__MolLogP",
            "intramolecular polar contacts multiplied by lipophilicity",
        ),
        "interaction__energy_range_x_effective_conformers": (
            "f3d__energy_range_kcal_mol",
            "f3d__effective_conformer_count",
            "energy-span multiplied by conformer-ensemble diversity",
        ),
    }
    columns: dict[str, np.ndarray] = {}
    descriptions: dict[str, str] = {}
    for output, (left, right, description) in definitions.items():
        if left not in frame or right not in frame:
            continue
        left_values = pd.to_numeric(frame[left], errors="coerce").to_numpy(dtype=np.float64)
        right_values = pd.to_numeric(frame[right], errors="coerce").to_numpy(dtype=np.float64)
        if "abs_charge" in output:
            if left == "f3d__formal_charge":
                left_values = np.abs(left_values)
            if right == "f3d__formal_charge":
                right_values = np.abs(right_values)
        columns[output] = left_values * right_values
        descriptions[output] = description
    return pd.DataFrame(columns, index=frame.index), descriptions


def _load_exact_train_targets(observations_path: Path) -> pd.DataFrame:
    columns = [
        "structure_id",
        "standardized_smiles",
        "standard_inchi_key",
        "model_split",
        "scaffold_group_id",
        "target_variant",
        "wild_type_evidence_scope",
        "master_confirmed_wild_type_scope",
        "measurement_modality",
        "automation_class",
        "assay_family",
        "source_family",
        "protocol_completeness_score",
        "potency_relation_pic50",
        "potency_pic50_point",
        "standardized_pic50_primary",
    ]
    filters = [
        ("model_split", "=", "train"),
        ("standardized_pic50_primary", "=", True),
        ("potency_relation_pic50", "=", "="),
        ("target_variant", "=", "wild_type_or_unspecified"),
    ]
    observations = pq.read_table(observations_path, columns=columns, filters=filters).to_pandas()
    if observations.empty:
        raise DiscoveryWorkerError("no exact train-only pIC50 observations were found")
    if set(observations["model_split"].astype(str)) != {"train"}:
        raise DiscoveryWorkerError("repository validation/test outcomes entered preparation")
    if set(observations["target_variant"].astype(str)) != {"wild_type_or_unspecified"}:
        raise DiscoveryWorkerError("non-WT-or-unspecified target variants entered preparation")

    # A small number of evidence rows retain alternate tautomeric SMILES for the
    # same stable structure/InChIKey.  Keep a deterministic modal SMILES for
    # Chemprop export, but require the true identity and split grouping to agree.
    identity_columns = ["standard_inchi_key", "scaffold_group_id"]
    for column in identity_columns:
        conflicts = observations.groupby("structure_id")[column].nunique(dropna=False)
        if int(conflicts.max()) != 1:
            raise DiscoveryWorkerError(f"structure has conflicting {column}: {conflicts.idxmax()}")
    return (
        observations.groupby("structure_id", as_index=False)
        .agg(
            standardized_smiles=("standardized_smiles", _mode),
            standard_inchi_key=("standard_inchi_key", "first"),
            scaffold_group_id=("scaffold_group_id", "first"),
            target_pic50=("potency_pic50_point", "median"),
            exact_observation_count=("potency_pic50_point", "size"),
            measurement_modality=("measurement_modality", _mode),
            automation_class=("automation_class", _mode),
            assay_family=("assay_family", _mode),
            source_family=("source_family", _mode),
            protocol_completeness_mean=("protocol_completeness_score", "mean"),
            wild_type_evidence_scope=("wild_type_evidence_scope", _mode),
            master_confirmed_wild_type_scope=("master_confirmed_wild_type_scope", "max"),
        )
        .sort_values("structure_id", kind="stable")
        .reset_index(drop=True)
    )


def _make_nested_splits(
    frame: pd.DataFrame,
    *,
    outer_folds: int,
    inner_folds: int,
    seed: int = SEED,
) -> pd.DataFrame:
    del seed  # GroupKFold is deterministic for stable input order.
    groups = frame["scaffold_group_id"].astype(str).to_numpy()
    if len(set(groups)) < outer_folds:
        raise DiscoveryWorkerError("insufficient scaffold groups for outer folds")
    indices = np.arange(len(frame))
    rows: list[pd.DataFrame] = []
    for outer_fold, (outer_fit, outer_heldout) in enumerate(
        GroupKFold(n_splits=outer_folds).split(indices, groups=groups)
    ):
        inner_groups = groups[outer_fit]
        if len(set(inner_groups)) < inner_folds:
            raise DiscoveryWorkerError(f"outer fold {outer_fold} has too few inner scaffold groups")
        inner_assignment = np.full(len(frame), -1, dtype=np.int16)
        for inner_fold, (_, inner_heldout_relative) in enumerate(
            GroupKFold(n_splits=inner_folds).split(outer_fit, groups=inner_groups)
        ):
            inner_assignment[outer_fit[inner_heldout_relative]] = inner_fold
        outer_role = np.full(len(frame), "fit", dtype=object)
        outer_role[outer_heldout] = "heldout"
        rows.append(
            pd.DataFrame(
                {
                    "structure_id": frame["structure_id"].astype(str),
                    "scaffold_group_id": groups,
                    "source_partition": "train",
                    "outer_fold": outer_fold,
                    "outer_role": outer_role,
                    "inner_fold": inner_assignment,
                }
            )
        )
    result = pd.concat(rows, ignore_index=True)
    expected = len(frame) * outer_folds
    if len(result) != expected or set(result["source_partition"]) != {"train"}:
        raise DiscoveryWorkerError("nested split registry is incomplete")
    for outer_fold, outer in result.groupby("outer_fold"):
        fit_groups = set(outer.loc[outer["outer_role"].eq("fit"), "scaffold_group_id"])
        heldout_groups = set(outer.loc[outer["outer_role"].eq("heldout"), "scaffold_group_id"])
        if fit_groups & heldout_groups:
            raise DiscoveryWorkerError(f"scaffold leakage in outer fold {outer_fold}")
        if set(outer.loc[outer["outer_role"].eq("fit"), "inner_fold"]) != set(range(inner_folds)):
            raise DiscoveryWorkerError(f"inner fold closure failure in outer fold {outer_fold}")
    return result


def _prepare(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(args.repo_root).resolve()
    matrix_root = Path(args.matrix_root).resolve()
    observations_path = Path(args.observations).resolve()
    output = Path(args.output_root).resolve()
    if (output / "validation.json").is_file():
        return _validate_prepared(output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise DiscoveryWorkerError("prepare output must be empty or already validated")

    matrix_validation_path = matrix_root / "validation.json"
    matrix_path = matrix_root / "combined_feature_matrix.parquet"
    if json.loads(matrix_validation_path.read_text()).get("status") != "passed":
        raise DiscoveryWorkerError("quantitative feature matrix has not passed validation")
    # Predicate pushdown is a core leakage barrier: no repository validation/test rows are loaded.
    matrix = pq.read_table(matrix_path, filters=[("model_split", "=", "train")]).to_pandas()
    if set(matrix["model_split"].astype(str)) != {"train"}:
        raise DiscoveryWorkerError("nontraining feature rows entered preparation")
    targets = _load_exact_train_targets(observations_path)
    routing_drop = [
        column for column in ("standard_inchi_key", "model_split", "scaffold_group_id") if column in matrix
    ]
    frame = targets.merge(
        matrix.drop(columns=routing_drop), on="structure_id", validate="one_to_one", how="inner"
    )
    if len(frame) != len(targets):
        raise DiscoveryWorkerError("exact train-only target/feature join is incomplete")

    morgan_raw = frame["morgan_r2_2048"].tolist()
    maccs_raw = frame["maccs_167"].tolist()
    morgan = _unpack_fingerprint_column(morgan_raw, MORGAN_BITS)
    maccs = _unpack_fingerprint_column(maccs_raw, MACCS_BITS)
    morgan_columns = [f"morgan__{index:04d}" for index in range(MORGAN_BITS)]
    maccs_columns = [f"maccs__{index:03d}" for index in range(MACCS_BITS)]
    expanded = pd.DataFrame(
        np.concatenate([morgan, maccs], axis=1),
        columns=[*morgan_columns, *maccs_columns],
        dtype=np.uint8,
    )
    interactions, interaction_descriptions = _targeted_interactions(frame)
    frame = frame.drop(columns=["morgan_r2_2048", "maccs_167"])
    frame = frame.rename(
        columns={
            "morgan_r2_2048": "morgan_r2_2048_raw",
            "maccs_167": "maccs_167_raw",
        }
    )
    # Keep the packed representation for exact, memory-efficient similarity calculations.
    frame.insert(len(frame.columns), "morgan_r2_2048_raw", morgan_raw)
    frame.insert(len(frame.columns), "maccs_167_raw", maccs_raw)
    frame = pd.concat([frame.reset_index(drop=True), expanded, interactions.reset_index(drop=True)], axis=1)

    rdkit2d = sorted(column for column in frame if column.startswith("rdkit2d__"))
    numeric_3d = sorted(
        column
        for column in frame
        if column.startswith("f3d__")
        and column not in {"f3d__feature_order", "f3d__energy_min_kcal_mol"}
        and pd.api.types.is_numeric_dtype(frame[column])
    )
    groups: dict[str, list[str]] = {
        "rdkit2d": rdkit2d,
        **_descriptor_subgroups(rdkit2d),
        "morgan": morgan_columns,
        "maccs": maccs_columns,
        "shape_scalars": [column for column in numeric_3d if _is_shape(column)],
        "polarity_charge_scalars": [column for column in numeric_3d if _is_polarity_charge(column)],
        "energy_flexibility": [column for column in numeric_3d if _is_energy_flexibility(column)],
        "autocorr3d": [column for column in numeric_3d if "dominant_autocorr3d" in column],
        "whim3d": [column for column in numeric_3d if "dominant_whim" in column],
        "all_conformer3d": numeric_3d,
        "targeted_interactions": sorted(interaction_descriptions),
    }
    if any(not values for values in groups.values()):
        empty = sorted(name for name, values in groups.items() if not values)
        raise DiscoveryWorkerError(f"one or more feature groups are empty: {empty}")
    groups["safe_classical"] = sorted(
        set(
            groups["rdkit2d"]
            + groups["polarity_charge_scalars"]
            + groups["energy_flexibility"]
            + groups["shape_scalars"]
            + groups["targeted_interactions"]
        )
    )
    feature_sets = {
        "rdkit2d": ["rdkit2d"],
        "morgan": ["morgan"],
        "morgan_rdkit2d": ["morgan", "rdkit2d"],
        "morgan_rdkit2d_maccs": ["morgan", "rdkit2d", "maccs"],
        "physics_selected": ["autocorr3d", "polarity_charge_scalars"],
        "candidate_primary": [
            "morgan",
            "rdkit2d",
            "autocorr3d",
            "polarity_charge_scalars",
            "targeted_interactions",
        ],
        "all_scalable": [
            "morgan",
            "maccs",
            "rdkit2d",
            "all_conformer3d",
            "targeted_interactions",
        ],
        "safe_classical": ["safe_classical"],
    }
    split_registry = _make_nested_splits(frame, outer_folds=args.outer_folds, inner_folds=args.inner_folds)
    cache_path = output / "exact_train_cache.parquet"
    split_path = output / "nested_scaffold_splits.parquet"
    registry_path = output / "feature_registry.json"
    summary_path = output / "source_summary.json"
    _atomic_parquet(cache_path, frame.sort_values("structure_id", kind="stable"))
    _atomic_parquet(split_path, split_registry)
    _atomic_json(
        registry_path,
        {
            "schema_version": SCHEMA_VERSION,
            "fingerprint_encoding": {
                "source": "RDKit BitVectToBinaryText",
                "bit_order": "little",
                "morgan_bits": MORGAN_BITS,
                "maccs_bits": MACCS_BITS,
            },
            "feature_groups": groups,
            "feature_sets": feature_sets,
            "targeted_interaction_interpretation": interaction_descriptions,
        },
    )
    _atomic_json(
        summary_path,
        {
            "structures": len(frame),
            "exact_observations": int(frame["exact_observation_count"].sum()),
            "target_scope": "wild_type_or_unspecified_hERG_exact_pIC50",
            "repository_source_partition": "train_only",
            "measurement_modality_counts": frame["measurement_modality"].value_counts().to_dict(),
            "automation_class_counts": frame["automation_class"].value_counts().to_dict(),
            "assay_family_counts": frame["assay_family"].value_counts().to_dict(),
            "source_family_counts": frame["source_family"].value_counts().to_dict(),
            "limitations": [
                "wild_type_or_unspecified is not equivalent to experimentally confirmed WT",
                "assay modality, automation, protocol, temperature, and cell context are often unresolved",
                "the broad confirmed-WT fixed-dose surface is classification evidence and must not be "
                "numerically pooled into exact pIC50 regression",
                "ligand-only descriptors do not represent receptor state, membrane access, or binding free energy",
            ],
        },
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "command": "prepare",
        "repo_root": str(repo_root),
        "scientific_scope": {
            "repository_partition_loaded": "train_only",
            "validation_outcomes_loaded": False,
            "test_outcomes_loaded": False,
            "target": "exact_standardized_pIC50",
            "target_variant": "wild_type_or_unspecified",
            "predictive_superiority_established": False,
        },
        "parameters": {
            "outer_folds": args.outer_folds,
            "inner_folds": args.inner_folds,
            "seed": SEED,
        },
        "inputs": [
            _binding(matrix_path),
            _binding(matrix_validation_path),
            _binding(observations_path),
            _binding(Path(__file__).resolve()),
        ],
        "artifacts": [
            _binding(cache_path),
            _binding(split_path),
            _binding(registry_path),
            _binding(summary_path),
        ],
    }
    _atomic_json(output / "manifest.json", manifest)
    validation = _validate_prepared(output)
    _atomic_json(output / "validation.json", validation)
    return validation


def _validate_prepared(output: Path) -> dict[str, Any]:
    manifest = json.loads((output / "manifest.json").read_text())
    for binding in manifest["inputs"] + manifest["artifacts"]:
        _validate_binding(binding)
    cache = pq.read_table(
        output / "exact_train_cache.parquet",
        columns=["structure_id", "scaffold_group_id", "target_pic50"],
    ).to_pandas()
    splits = pq.read_table(output / "nested_scaffold_splits.parquet").to_pandas()
    parameters = manifest["parameters"]
    if cache["structure_id"].duplicated().any() or cache["target_pic50"].isna().any():
        raise DiscoveryWorkerError("prepared cache identity/target contract failed")
    if set(splits["source_partition"]) != {"train"}:
        raise DiscoveryWorkerError("prepared split registry contains nontraining rows")
    if len(splits) != len(cache) * int(parameters["outer_folds"]):
        raise DiscoveryWorkerError("prepared split registry row count mismatch")
    return {
        "status": "passed",
        "structures": len(cache),
        "outer_folds": int(parameters["outer_folds"]),
        "inner_folds": int(parameters["inner_folds"]),
        "repository_validation_outcomes_loaded": False,
        "repository_test_outcomes_loaded": False,
        "source_partition": "train_only",
    }


def _validate_prepared_fast(prepared: Path) -> None:
    validation = json.loads((prepared / "validation.json").read_text())
    if validation.get("status") != "passed" or validation.get("source_partition") != "train_only":
        raise DiscoveryWorkerError("prepared cache is not a validated train-only cache")


def _parse_parameters(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    possible_path = Path(value)
    text = possible_path.read_text() if possible_path.is_file() else value
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise DiscoveryWorkerError("model parameters must be a JSON object")
    return parsed


def _safe_unit_id(value: str) -> str:
    if not value or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,159}", value):
        raise DiscoveryWorkerError("unit-id must contain only letters, numbers, dot, underscore, dash")
    return value


def _resolve_features(prepared: Path, groups_argument: str) -> tuple[list[str], list[str]]:
    registry = json.loads((prepared / "feature_registry.json").read_text())
    requested = [item.strip() for item in groups_argument.split(",") if item.strip()]
    if not requested:
        raise DiscoveryWorkerError("at least one feature group or feature set is required")
    expanded_groups: list[str] = []
    for name in requested:
        if name in registry["feature_sets"]:
            expanded_groups.extend(registry["feature_sets"][name])
        elif name in registry["feature_groups"]:
            expanded_groups.append(name)
        else:
            raise DiscoveryWorkerError(f"unknown feature group/set: {name}")
    columns: list[str] = []
    seen: set[str] = set()
    for group in expanded_groups:
        for column in registry["feature_groups"][group]:
            if column not in seen:
                seen.add(column)
                columns.append(column)
    return columns, list(dict.fromkeys(expanded_groups))


def _unit_membership(
    prepared: Path,
    *,
    stage: str,
    outer_fold: int,
    inner_fold: int | None,
) -> pd.DataFrame:
    splits = pq.read_table(
        prepared / "nested_scaffold_splits.parquet",
        filters=[("outer_fold", "=", outer_fold)],
    ).to_pandas()
    if splits.empty:
        raise DiscoveryWorkerError(f"outer fold does not exist: {outer_fold}")
    if stage == "outer":
        splits["unit_role"] = np.where(splits["outer_role"].eq("heldout"), "evaluate", "fit")
    else:
        if inner_fold is None:
            raise DiscoveryWorkerError("inner stage requires --inner-fold")
        if inner_fold not in set(splits.loc[splits["outer_role"].eq("fit"), "inner_fold"]):
            raise DiscoveryWorkerError(f"inner fold does not exist: {inner_fold}")
        splits["unit_role"] = "excluded_outer_heldout"
        outer_fit = splits["outer_role"].eq("fit")
        splits.loc[outer_fit, "unit_role"] = np.where(
            splits.loc[outer_fit, "inner_fold"].eq(inner_fold), "evaluate", "fit"
        )
    return splits


def _load_unit_frame(
    prepared: Path,
    membership: pd.DataFrame,
    feature_columns: Sequence[str],
) -> pd.DataFrame:
    routing = [
        "structure_id",
        "scaffold_group_id",
        "target_pic50",
        "measurement_modality",
        "automation_class",
        "assay_family",
        "source_family",
        "protocol_completeness_mean",
    ]
    cache = pq.read_table(
        prepared / "exact_train_cache.parquet", columns=[*routing, *feature_columns]
    ).to_pandas()
    frame = membership.merge(cache, on=["structure_id", "scaffold_group_id"], validate="one_to_one")
    if set(frame["unit_role"]) != {"fit", "evaluate", "excluded_outer_heldout"} and set(
        frame["unit_role"]
    ) != {"fit", "evaluate"}:
        raise DiscoveryWorkerError("unit role closure failed")
    return frame


def _sanitize(
    frame: pd.DataFrame,
    columns: Sequence[str],
    fit_mask: np.ndarray,
) -> tuple[np.ndarray, list[str], int]:
    raw = frame[list(columns)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64, copy=True)
    invalid = ~np.isfinite(raw) | (np.abs(raw) > MAX_ABSOLUTE_NUMERIC_FEATURE)
    invalid_count = int(invalid.sum())
    raw[invalid] = np.nan
    fit = raw[fit_mask]
    finite_count = np.isfinite(fit).sum(axis=0)
    with np.errstate(all="ignore"):
        minimum = np.nanmin(fit, axis=0)
        maximum = np.nanmax(fit, axis=0)
    retained_mask = (finite_count > 0) & ((maximum - minimum) > 1.0e-12)
    retained = [column for column, keep in zip(columns, retained_mask, strict=True) if keep]
    if not retained:
        raise DiscoveryWorkerError("no nonconstant finite features remain in unit fit partition")
    return np.ascontiguousarray(raw[:, retained_mask], dtype=np.float32), retained, invalid_count


def _metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    residual = observed - predicted
    return {
        "n": int(len(observed)),
        "mae": float(mean_absolute_error(observed, predicted)),
        "median_absolute_error": float(np.median(np.abs(residual))),
        "rmse": float(math.sqrt(mean_squared_error(observed, predicted))),
        "r2": float(r2_score(observed, predicted)),
        "pearson": float(pearsonr(observed, predicted).statistic),
        "spearman": float(spearmanr(observed, predicted).statistic),
        "fraction_within_0p5": float(np.mean(np.abs(residual) <= 0.5)),
        "fraction_within_1p0": float(np.mean(np.abs(residual) <= 1.0)),
        "mean_signed_error_observed_minus_predicted": float(np.mean(residual)),
    }


def _importance_frame(
    model: Any,
    model_id: str,
    unit_id: str,
    columns: Sequence[str],
    X: np.ndarray,
    y: np.ndarray,
    fit_mask: np.ndarray,
) -> pd.DataFrame:
    if hasattr(model, "feature_importances_"):
        importance = np.asarray(model.feature_importances_, dtype=np.float64)
    elif hasattr(model, "coef_"):
        importance = np.abs(np.asarray(model.coef_, dtype=np.float64).ravel())
    else:
        return pd.DataFrame(
            columns=[
                "model_id",
                "unit_id",
                "feature_name",
                "importance",
                "signed_spearman_fit",
            ]
        )
    order = np.argsort(-importance, kind="stable")
    rows: list[dict[str, Any]] = []
    for index in order[: min(250, len(order))]:
        values = X[fit_mask, index].astype(np.float64)
        finite = np.isfinite(values) & np.isfinite(y[fit_mask])
        correlation = math.nan
        if finite.sum() >= 10 and np.unique(values[finite]).size > 1:
            correlation = float(spearmanr(values[finite], y[fit_mask][finite]).statistic)
        rows.append(
            {
                "model_id": model_id,
                "unit_id": unit_id,
                "feature_name": columns[index],
                "importance": float(importance[index]),
                "signed_spearman_fit": correlation if math.isfinite(correlation) else None,
            }
        )
    return pd.DataFrame(rows)


def _unit_paths(output_root: Path, unit_id: str) -> tuple[Path, Path, Path]:
    unit = output_root / "units" / _safe_unit_id(unit_id)
    return unit / "metrics.json", unit / "predictions.parquet", unit / "feature_importance.parquet"


def _existing_unit(metrics_path: Path, predictions_path: Path) -> dict[str, Any] | None:
    if not metrics_path.is_file():
        return None
    metrics = json.loads(metrics_path.read_text())
    if metrics.get("status") != "passed":
        return None
    binding = metrics.get("prediction_artifact")
    if not isinstance(binding, dict):
        return None
    _validate_binding(binding)
    if Path(binding["path"]).resolve() != predictions_path.resolve():
        raise DiscoveryWorkerError("existing unit prediction path does not match unit-id")
    return metrics


def _write_unit(
    *,
    metrics_path: Path,
    predictions_path: Path,
    importance_path: Path,
    metrics: dict[str, Any],
    predictions: pd.DataFrame,
    importance: pd.DataFrame,
) -> dict[str, Any]:
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_parquet(predictions_path, predictions)
    _atomic_parquet(importance_path, importance)
    metrics["prediction_artifact"] = _binding(predictions_path)
    metrics["importance_artifact"] = _binding(importance_path)
    metrics["status"] = "passed"
    _atomic_json(metrics_path, metrics)
    return metrics


def _base_unit_metadata(
    args: argparse.Namespace,
    *,
    unit_kind: str,
    expanded_groups: Sequence[str],
    parameters: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "unit_kind": unit_kind,
        "unit_id": args.unit_id,
        "model_id": args.model_id,
        "stage": args.stage,
        "outer_fold": args.outer_fold,
        "inner_fold": args.inner_fold if args.stage == "inner" else None,
        "expanded_feature_groups": list(expanded_groups),
        "parameters": parameters,
        "scientific_scope": {
            "source_partition": "train_only",
            "outer_and_inner_are_nested_training_folds": True,
            "repository_validation_outcomes_loaded": False,
            "repository_test_outcomes_loaded": False,
            "target_variant": "wild_type_or_unspecified",
            "causal_or_mechanistic_claim_allowed": False,
            "predictive_superiority_established": False,
        },
    }


def _tree_unit(args: argparse.Namespace) -> dict[str, Any]:
    prepared = Path(args.prepared_root).resolve()
    output_root = Path(args.output_root).resolve()
    _validate_prepared_fast(prepared)
    metrics_path, predictions_path, importance_path = _unit_paths(output_root, args.unit_id)
    existing = _existing_unit(metrics_path, predictions_path)
    if existing is not None:
        return existing
    columns, groups = _resolve_features(prepared, args.groups)
    membership = _unit_membership(
        prepared,
        stage=args.stage,
        outer_fold=args.outer_fold,
        inner_fold=args.inner_fold,
    )
    frame = _load_unit_frame(prepared, membership, columns)
    fit_mask = frame["unit_role"].eq("fit").to_numpy()
    evaluate_mask = frame["unit_role"].eq("evaluate").to_numpy()
    X, retained, invalid_count = _sanitize(frame, columns, fit_mask)
    y = np.ascontiguousarray(frame["target_pic50"].to_numpy(dtype=np.float32))
    parameters = _parse_parameters(args.params_json)
    engine = args.engine.lower()
    if engine == "xgboost":
        from xgboost import XGBRegressor

        defaults: dict[str, Any] = {
            "objective": "reg:squarederror",
            "n_estimators": 700,
            "max_depth": 6,
            "learning_rate": 0.03,
            "min_child_weight": 3.0,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.05,
            "reg_lambda": 1.0,
            "tree_method": "hist",
        }
        defaults.update(parameters)
        defaults.update(n_jobs=args.workers, random_state=args.seed, verbosity=0)
        model: Any = XGBRegressor(**defaults)
        parameters = defaults
    elif engine == "lightgbm":
        from lightgbm import LGBMRegressor

        defaults = {
            "objective": "regression_l1",
            "n_estimators": 700,
            "learning_rate": 0.03,
            "num_leaves": 31,
            "max_depth": -1,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.05,
            "reg_lambda": 1.0,
        }
        defaults.update(parameters)
        defaults.update(n_jobs=args.workers, random_state=args.seed, verbosity=-1)
        model = LGBMRegressor(**defaults)
        parameters = defaults
    else:
        raise DiscoveryWorkerError(f"unsupported tree engine: {engine}")
    started = time.monotonic()
    model.fit(X[fit_mask], y[fit_mask])
    elapsed = time.monotonic() - started
    predicted = np.asarray(model.predict(X[evaluate_mask]), dtype=np.float64)
    observed = y[evaluate_mask].astype(np.float64)
    predictions = frame.loc[
        evaluate_mask,
        ["structure_id", "scaffold_group_id", "measurement_modality", "automation_class"],
    ].copy()
    predictions.insert(0, "unit_id", args.unit_id)
    predictions.insert(0, "model_id", args.model_id)
    predictions["source_partition"] = "train"
    predictions["observed_pic50"] = observed
    predictions["predicted_pic50"] = predicted
    predictions["residual_observed_minus_predicted"] = observed - predicted
    importance = _importance_frame(model, args.model_id, args.unit_id, retained, X, y, fit_mask)
    metrics = _base_unit_metadata(
        args, unit_kind=f"tree_{engine}", expanded_groups=groups, parameters=parameters
    )
    metrics.update(
        feature_count=len(retained),
        requested_feature_count=len(columns),
        fit_structures=int(fit_mask.sum()),
        evaluation_structures=int(evaluate_mask.sum()),
        invalid_or_overflow_cells_treated_as_missing=invalid_count,
        fit_elapsed_seconds=elapsed,
        metrics=_metrics(observed, predicted),
    )
    return _write_unit(
        metrics_path=metrics_path,
        predictions_path=predictions_path,
        importance_path=importance_path,
        metrics=metrics,
        predictions=predictions,
        importance=importance,
    )


def _classical_unit(args: argparse.Namespace) -> dict[str, Any]:
    prepared = Path(args.prepared_root).resolve()
    output_root = Path(args.output_root).resolve()
    _validate_prepared_fast(prepared)
    metrics_path, predictions_path, importance_path = _unit_paths(output_root, args.unit_id)
    existing = _existing_unit(metrics_path, predictions_path)
    if existing is not None:
        return existing
    columns, groups = _resolve_features(prepared, args.groups)
    if any(column.startswith(("morgan__", "maccs__")) for column in columns):
        raise DiscoveryWorkerError("classical-unit accepts reduced numeric groups, not bit fingerprints")
    if len(columns) > args.maximum_features:
        raise DiscoveryWorkerError(
            f"classical-unit requested {len(columns)} features; maximum is {args.maximum_features}"
        )
    membership = _unit_membership(
        prepared,
        stage=args.stage,
        outer_fold=args.outer_fold,
        inner_fold=args.inner_fold,
    )
    frame = _load_unit_frame(prepared, membership, columns)
    fit_mask = frame["unit_role"].eq("fit").to_numpy()
    evaluate_mask = frame["unit_role"].eq("evaluate").to_numpy()
    X, retained, invalid_count = _sanitize(frame, columns, fit_mask)
    y = np.ascontiguousarray(frame["target_pic50"].to_numpy(dtype=np.float64))
    parameters = _parse_parameters(args.params_json)
    from sklearn.impute import SimpleImputer

    imputer = SimpleImputer(strategy="median")
    X_fit = imputer.fit_transform(X[fit_mask])
    X_evaluate = imputer.transform(X[evaluate_mask])
    model_name = args.model.lower()
    if model_name == "ridge":
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X_fit = scaler.fit_transform(X_fit)
        X_evaluate = scaler.transform(X_evaluate)
        defaults: dict[str, Any] = {"alpha": 10.0, "solver": "lsqr"}
        defaults.update(parameters)
        model: Any = Ridge(**defaults)
        parameters = defaults
    elif model_name in {"extratrees", "randomforest"}:
        if model_name == "extratrees":
            from sklearn.ensemble import ExtraTreesRegressor as Forest
        else:
            from sklearn.ensemble import RandomForestRegressor as Forest
        defaults = {
            "n_estimators": 500,
            "max_features": 0.7,
            "min_samples_leaf": 2,
            "n_jobs": args.workers,
            "random_state": args.seed,
        }
        defaults.update(parameters)
        defaults.update(n_jobs=args.workers, random_state=args.seed)
        model = Forest(**defaults)
        parameters = defaults
    else:
        raise DiscoveryWorkerError(f"unsupported classical model: {model_name}")
    started = time.monotonic()
    model.fit(X_fit, y[fit_mask])
    elapsed = time.monotonic() - started
    predicted = np.asarray(model.predict(X_evaluate), dtype=np.float64)
    observed = y[evaluate_mask]
    predictions = frame.loc[
        evaluate_mask,
        ["structure_id", "scaffold_group_id", "measurement_modality", "automation_class"],
    ].copy()
    predictions.insert(0, "unit_id", args.unit_id)
    predictions.insert(0, "model_id", args.model_id)
    predictions["source_partition"] = "train"
    predictions["observed_pic50"] = observed
    predictions["predicted_pic50"] = predicted
    predictions["residual_observed_minus_predicted"] = observed - predicted
    # Use the original sanitized array for model-independent feature direction.
    importance = _importance_frame(model, args.model_id, args.unit_id, retained, X, y, fit_mask)
    metrics = _base_unit_metadata(
        args, unit_kind=f"classical_{model_name}", expanded_groups=groups, parameters=parameters
    )
    metrics.update(
        feature_count=len(retained),
        requested_feature_count=len(columns),
        fit_structures=int(fit_mask.sum()),
        evaluation_structures=int(evaluate_mask.sum()),
        invalid_or_overflow_cells_treated_as_missing=invalid_count,
        fit_elapsed_seconds=elapsed,
        metrics=_metrics(observed, predicted),
    )
    return _write_unit(
        metrics_path=metrics_path,
        predictions_path=predictions_path,
        importance_path=importance_path,
        metrics=metrics,
        predictions=predictions,
        importance=importance,
    )


def _similarity_unit(args: argparse.Namespace) -> dict[str, Any]:
    prepared = Path(args.prepared_root).resolve()
    output_root = Path(args.output_root).resolve()
    _validate_prepared_fast(prepared)
    metrics_path, predictions_path, importance_path = _unit_paths(output_root, args.unit_id)
    existing = _existing_unit(metrics_path, predictions_path)
    if existing is not None:
        return existing
    membership = _unit_membership(
        prepared,
        stage=args.stage,
        outer_fold=args.outer_fold,
        inner_fold=args.inner_fold,
    )
    cache = pq.read_table(
        prepared / "exact_train_cache.parquet",
        columns=["structure_id", "scaffold_group_id", "target_pic50", "morgan_r2_2048_raw"],
    ).to_pandas()
    frame = membership.merge(cache, on=["structure_id", "scaffold_group_id"], validate="one_to_one")
    fit_mask = frame["unit_role"].eq("fit").to_numpy()
    evaluate_mask = frame["unit_role"].eq("evaluate").to_numpy()
    from rdkit import DataStructs

    fit_fingerprints = [
        DataStructs.CreateFromBinaryText(value) for value in frame.loc[fit_mask, "morgan_r2_2048_raw"]
    ]
    fit_targets = frame.loc[fit_mask, "target_pic50"].to_numpy(dtype=np.float64)
    global_mean = float(np.mean(fit_targets))
    query_values = frame.loc[evaluate_mask, "morgan_r2_2048_raw"].tolist()
    predicted: list[float] = []
    maximum_similarity: list[float] = []
    mean_similarity: list[float] = []
    effective_neighbors: list[int] = []
    started = time.monotonic()
    for chunk_start in range(0, len(query_values), args.chunk_size):
        chunk = query_values[chunk_start : chunk_start + args.chunk_size]
        for value in chunk:
            query = DataStructs.CreateFromBinaryText(value)
            similarities = np.asarray(
                DataStructs.BulkTanimotoSimilarity(query, fit_fingerprints), dtype=np.float64
            )
            candidates = np.flatnonzero(similarities >= args.similarity_floor)
            if candidates.size:
                count = min(args.neighbors, candidates.size)
                local = candidates[np.argpartition(similarities[candidates], -count)[-count:]]
                local = local[np.argsort(-similarities[local], kind="stable")]
                weights = np.power(similarities[local], args.weight_power)
                prediction = (
                    float(np.average(fit_targets[local], weights=weights))
                    if float(weights.sum()) > 0
                    else float(np.mean(fit_targets[local]))
                )
                selected_similarities = similarities[local]
            else:
                prediction = global_mean
                selected_similarities = np.empty(0, dtype=np.float64)
            predicted.append(prediction)
            maximum_similarity.append(
                float(selected_similarities.max()) if len(selected_similarities) else 0.0
            )
            mean_similarity.append(float(selected_similarities.mean()) if len(selected_similarities) else 0.0)
            effective_neighbors.append(len(selected_similarities))
        print(
            f"unit={args.unit_id} queries={min(chunk_start + len(chunk), len(query_values))}/"
            f"{len(query_values)}",
            flush=True,
        )
    elapsed = time.monotonic() - started
    observed = frame.loc[evaluate_mask, "target_pic50"].to_numpy(dtype=np.float64)
    predicted_array = np.asarray(predicted, dtype=np.float64)
    predictions = frame.loc[evaluate_mask, ["structure_id", "scaffold_group_id"]].copy()
    predictions.insert(0, "unit_id", args.unit_id)
    predictions.insert(0, "model_id", args.model_id)
    predictions["source_partition"] = "train"
    predictions["observed_pic50"] = observed
    predictions["predicted_pic50"] = predicted_array
    predictions["residual_observed_minus_predicted"] = observed - predicted_array
    predictions["maximum_train_tanimoto"] = maximum_similarity
    predictions["mean_neighbor_tanimoto"] = mean_similarity
    predictions["effective_neighbors"] = effective_neighbors
    importance = pd.DataFrame(
        columns=[
            "model_id",
            "unit_id",
            "feature_name",
            "importance",
            "signed_spearman_fit",
        ]
    )
    parameters = {
        "neighbors": args.neighbors,
        "similarity_floor": args.similarity_floor,
        "weight_power": args.weight_power,
        "chunk_size": args.chunk_size,
    }
    metrics = _base_unit_metadata(
        args, unit_kind="similarity_morgan_tanimoto_knn", expanded_groups=["morgan"], parameters=parameters
    )
    metrics.update(
        feature_count=MORGAN_BITS,
        fit_structures=int(fit_mask.sum()),
        evaluation_structures=int(evaluate_mask.sum()),
        fit_elapsed_seconds=elapsed,
        applicability_domain={
            "median_maximum_train_tanimoto": float(np.median(maximum_similarity)),
            "fraction_below_0p3_maximum_train_tanimoto": float(np.mean(np.asarray(maximum_similarity) < 0.3)),
        },
        metrics=_metrics(observed, predicted_array),
    )
    return _write_unit(
        metrics_path=metrics_path,
        predictions_path=predictions_path,
        importance_path=importance_path,
        metrics=metrics,
        predictions=predictions,
        importance=importance,
    )


def _chemprop_prepare(args: argparse.Namespace) -> dict[str, Any]:
    prepared = Path(args.prepared_root).resolve()
    output = Path(args.output_root).resolve()
    _validate_prepared_fast(prepared)
    if (output / "validation.json").is_file():
        return json.loads((output / "validation.json").read_text())
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise DiscoveryWorkerError("chemprop output must be empty or already validated")
    cache = pq.read_table(
        prepared / "exact_train_cache.parquet",
        columns=["structure_id", "standardized_smiles", "target_pic50", "scaffold_group_id"],
    ).to_pandas()
    if cache["standardized_smiles"].isna().any():
        raise DiscoveryWorkerError("Chemprop export contains missing SMILES")
    _atomic_csv(output / "all_train_only.csv", cache)
    split_registry = pq.read_table(prepared / "nested_scaffold_splits.parquet").to_pandas()
    outer_folds = (
        [args.outer_fold] if args.outer_fold is not None else sorted(split_registry["outer_fold"].unique())
    )
    artifacts = [output / "all_train_only.csv"]
    split_summary: list[dict[str, Any]] = []
    for outer_fold in outer_folds:
        membership = split_registry[split_registry["outer_fold"].eq(outer_fold)]
        if membership.empty:
            raise DiscoveryWorkerError(f"unknown outer fold for Chemprop: {outer_fold}")
        outer_root = output / f"outer_{outer_fold}"
        outer_root.mkdir()
        joined = membership.merge(cache, on=["structure_id", "scaffold_group_id"], validate="one_to_one")
        fit = joined[joined["outer_role"].eq("fit")]
        heldout = joined[joined["outer_role"].eq("heldout")]
        fit_path = outer_root / "outer_fit_train_only.csv"
        heldout_path = outer_root / "outer_heldout_train_only.csv"
        _atomic_csv(fit_path, fit[["standardized_smiles", "target_pic50", "structure_id"]])
        _atomic_csv(heldout_path, heldout[["standardized_smiles", "target_pic50", "structure_id"]])
        artifacts.extend([fit_path, heldout_path])
        for inner_fold in sorted(fit["inner_fold"].unique()):
            inner_fit = fit[~fit["inner_fold"].eq(inner_fold)]
            inner_validation = fit[fit["inner_fold"].eq(inner_fold)]
            inner_fit_path = outer_root / f"inner_{inner_fold}_fit.csv"
            inner_validation_path = outer_root / f"inner_{inner_fold}_validation.csv"
            _atomic_csv(
                inner_fit_path,
                inner_fit[["standardized_smiles", "target_pic50", "structure_id"]],
            )
            _atomic_csv(
                inner_validation_path,
                inner_validation[["standardized_smiles", "target_pic50", "structure_id"]],
            )
            artifacts.extend([inner_fit_path, inner_validation_path])
        split_summary.append(
            {
                "outer_fold": int(outer_fold),
                "outer_fit_structures": len(fit),
                "outer_heldout_structures": len(heldout),
                "inner_folds": int(fit["inner_fold"].nunique()),
            }
        )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "command": "chemprop-prepare",
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "scientific_scope": {
            "source_partition": "train_only",
            "repository_validation_rows_exported": 0,
            "repository_test_rows_exported": 0,
            "outer_heldout_files_are_training-partition_nested_folds": True,
        },
        "splits": split_summary,
        "artifacts": [_binding(path) for path in artifacts],
    }
    _atomic_json(output / "manifest.json", manifest)
    validation = {
        "status": "passed",
        "structures": len(cache),
        "outer_folds_exported": len(outer_folds),
        "repository_validation_rows_exported": 0,
        "repository_test_rows_exported": 0,
    }
    _atomic_json(output / "validation.json", validation)
    return validation


def _flatten_unit_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    result = {
        "unit_id": metrics["unit_id"],
        "model_id": metrics["model_id"],
        "unit_kind": metrics["unit_kind"],
        "stage": metrics["stage"],
        "outer_fold": metrics["outer_fold"],
        "inner_fold": metrics.get("inner_fold"),
        "feature_count": metrics.get("feature_count"),
        "fit_structures": metrics.get("fit_structures"),
        "evaluation_structures": metrics.get("evaluation_structures"),
        "fit_elapsed_seconds": metrics.get("fit_elapsed_seconds"),
        "expanded_feature_groups_json": json.dumps(metrics.get("expanded_feature_groups", [])),
        "parameters_json": json.dumps(metrics.get("parameters", {}), sort_keys=True),
    }
    result.update(metrics["metrics"])
    return result


def _verify_chemprop_unit_document(document: dict[str, Any]) -> None:
    claimed = document.get("unit_json_sha256")
    payload = dict(document)
    payload.pop("unit_json_sha256", None)
    actual = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    if not isinstance(claimed, str) or claimed != actual:
        raise DiscoveryWorkerError("Chemprop unit self-hash mismatch")
    resolved_spec = document.get("resolved_spec")
    if not isinstance(resolved_spec, dict):
        raise DiscoveryWorkerError("Chemprop unit has no resolved specification")
    spec_hash = hashlib.sha256(
        json.dumps(resolved_spec, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    if document.get("resolved_spec_sha256") != spec_hash:
        raise DiscoveryWorkerError("Chemprop resolved-spec hash mismatch")
    for binding in [*document.get("inputs", []), *document.get("artifacts", [])]:
        if not isinstance(binding, dict):
            raise DiscoveryWorkerError("Chemprop unit contains a malformed binding")
        _validate_binding(binding)


def _complete_chemprop_outer_oof(
    prepared: Path, results_root: Path
) -> tuple[list[dict[str, Any]], list[pd.DataFrame], dict[str, Any]]:
    """Admit Chemprop only as a complete, validated five-fold outer OOF model."""
    unit_paths = sorted((results_root / "chemprop_units").glob("*/unit.json"))
    integration: dict[str, Any] = {
        "model_id": CHEMPROP_MODEL_ID,
        "required_outer_folds": 5,
        "discovered_unit_documents": len(unit_paths),
        "valid_outer_folds": [],
        "excluded_units": [],
        "status": "absent" if not unit_paths else "incomplete",
    }
    if not unit_paths:
        return [], [], integration

    cache = pq.read_table(
        prepared / "exact_train_cache.parquet",
        columns=["structure_id", "scaffold_group_id", "target_pic50"],
    ).to_pandas()
    cache["structure_id"] = cache["structure_id"].astype(str)
    cache["scaffold_group_id"] = cache["scaffold_group_id"].astype(str)
    cache_by_id = cache.set_index("structure_id")
    if not cache_by_id.index.is_unique:
        raise DiscoveryWorkerError("prepared cache has duplicate structure identifiers")
    splits = pq.read_table(prepared / "nested_scaffold_splits.parquet").to_pandas()
    valid: dict[int, tuple[dict[str, Any], pd.DataFrame]] = {}
    duplicate_folds: set[int] = set()

    for unit_path in unit_paths:
        try:
            document = json.loads(unit_path.read_text(encoding="utf-8"))
            if not isinstance(document, dict) or document.get("status") != "passed":
                raise DiscoveryWorkerError("Chemprop unit is not passed")
            _verify_chemprop_unit_document(document)
            contract = document.get("scientific_contract", {})
            if (
                contract.get("repository_validation_labels_opened") is not False
                or contract.get("repository_test_labels_opened") is not False
                or contract.get("data_scope") != "exact_pic50_repository_train_partition_only"
            ):
                raise DiscoveryWorkerError("Chemprop scientific contract is not train-only")
            outer_fold = int(document["resolved_spec"]["outer_fold"])
            if outer_fold not in range(5):
                raise DiscoveryWorkerError(f"invalid Chemprop outer fold: {outer_fold}")
            if outer_fold in valid:
                duplicate_folds.add(outer_fold)
                raise DiscoveryWorkerError(f"duplicate passed Chemprop outer fold: {outer_fold}")

            prediction_path = Path(str(document.get("oof_predictions_path", ""))).resolve()
            expected_path = (unit_path.parent / "oof_predictions.parquet").resolve()
            if prediction_path != expected_path:
                raise DiscoveryWorkerError("Chemprop OOF path escapes its governed unit directory")
            artifact_paths = {
                Path(str(item["path"])).resolve()
                for item in document.get("artifacts", [])
                if isinstance(item, dict) and "path" in item
            }
            if prediction_path not in artifact_paths:
                raise DiscoveryWorkerError("Chemprop OOF file is not artifact-bound")
            predictions = pd.read_parquet(prediction_path)
            required = {
                "structure_id",
                "scaffold_group_id",
                "source_partition",
                "inner_role",
                "outer_fold",
                "observed_pic50",
                "predicted_pic50",
            }
            if not required <= set(predictions):
                raise DiscoveryWorkerError("Chemprop OOF schema is incomplete")
            predictions["structure_id"] = predictions["structure_id"].astype(str)
            predictions["scaffold_group_id"] = predictions["scaffold_group_id"].astype(str)
            if predictions["structure_id"].duplicated().any():
                raise DiscoveryWorkerError("Chemprop fold contains duplicate structures")
            if set(predictions["source_partition"].astype(str)) != {"train"}:
                raise DiscoveryWorkerError("Chemprop OOF contains a nontraining partition")
            if set(predictions["inner_role"].astype(str)) != {"holdout"}:
                raise DiscoveryWorkerError("Chemprop OOF is not an outer holdout")
            if set(pd.to_numeric(predictions["outer_fold"], errors="raise").astype(int)) != {outer_fold}:
                raise DiscoveryWorkerError("Chemprop OOF fold disagrees with its unit specification")

            expected_membership = splits[
                splits["outer_fold"].eq(outer_fold) & splits["outer_role"].eq("heldout")
            ][["structure_id", "scaffold_group_id"]].copy()
            expected_membership["structure_id"] = expected_membership["structure_id"].astype(str)
            expected_membership["scaffold_group_id"] = expected_membership["scaffold_group_id"].astype(str)
            actual_membership = predictions[["structure_id", "scaffold_group_id"]]
            checked = expected_membership.merge(
                actual_membership,
                on=["structure_id", "scaffold_group_id"],
                how="outer",
                indicator=True,
                validate="one_to_one",
            )
            if set(checked["_merge"].astype(str)) != {"both"}:
                raise DiscoveryWorkerError("Chemprop OOF does not match the governed outer scaffold fold")
            expected_targets = predictions["structure_id"].map(cache_by_id["target_pic50"])
            observed = pd.to_numeric(predictions["observed_pic50"], errors="coerce")
            predicted = pd.to_numeric(predictions["predicted_pic50"], errors="coerce")
            if (
                expected_targets.isna().any()
                or not np.isfinite(observed.to_numpy(dtype=np.float64)).all()
                or not np.isfinite(predicted.to_numpy(dtype=np.float64)).all()
                or not np.allclose(
                    observed.to_numpy(dtype=np.float64),
                    expected_targets.to_numpy(dtype=np.float64),
                    rtol=0.0,
                    atol=1.0e-7,
                )
            ):
                raise DiscoveryWorkerError("Chemprop OOF target values do not match the prepared cache")

            normalized = predictions[
                ["structure_id", "scaffold_group_id", "source_partition", "observed_pic50", "predicted_pic50"]
            ].copy()
            normalized.insert(0, "unit_id", str(document["unit_id"]))
            normalized.insert(0, "model_id", CHEMPROP_MODEL_ID)
            normalized["outer_fold"] = outer_fold
            normalized["residual_observed_minus_predicted"] = (
                normalized["observed_pic50"] - normalized["predicted_pic50"]
            )
            valid[outer_fold] = (document, normalized)
        except (
            DiscoveryWorkerError,
            OSError,
            ValueError,
            TypeError,
            KeyError,
            json.JSONDecodeError,
        ) as error:
            integration["excluded_units"].append({"path": str(unit_path), "reason": str(error)})

    integration["valid_outer_folds"] = sorted(valid)
    if duplicate_folds or set(valid) != set(range(5)):
        if duplicate_folds:
            integration["duplicate_outer_folds"] = sorted(duplicate_folds)
        return [], [], integration

    prediction_frames = [valid[fold][1] for fold in range(5)]
    combined = pd.concat(prediction_frames, ignore_index=True)
    if combined["structure_id"].duplicated().any() or set(combined["structure_id"]) != set(
        cache["structure_id"]
    ):
        integration["status"] = "incomplete"
        integration["exclusion_reason"] = "five Chemprop folds do not cover every prepared structure once"
        return [], [], integration

    metric_rows: list[dict[str, Any]] = []
    for outer_fold in range(5):
        document, predictions = valid[outer_fold]
        observed = predictions["observed_pic50"].to_numpy(dtype=np.float64)
        predicted = predictions["predicted_pic50"].to_numpy(dtype=np.float64)
        preparation = document.get("preparation", {})
        role_counts = preparation.get("role_counts", {})
        metric_rows.append(
            {
                "unit_id": str(document["unit_id"]),
                "model_id": CHEMPROP_MODEL_ID,
                "unit_kind": "chemprop_dmpnn",
                "stage": "outer",
                "outer_fold": outer_fold,
                "inner_fold": None,
                "feature_count": None,
                "fit_structures": int(role_counts.get("train", 0)) + int(role_counts.get("validation", 0)),
                "evaluation_structures": len(predictions),
                "fit_elapsed_seconds": float(document.get("runtime", {}).get("elapsed_seconds", 0.0)),
                "expanded_feature_groups_json": "[]",
                "parameters_json": json.dumps(document["resolved_spec"], sort_keys=True),
                **_metrics(observed, predicted),
            }
        )
    integration.update(
        status="integrated",
        integrated_outer_folds=5,
        integrated_prediction_rows=len(combined),
        complete_structure_coverage=True,
    )
    return metric_rows, prediction_frames, integration


def _grouped_model_bootstrap(
    candidate: pd.DataFrame,
    baseline: pd.DataFrame,
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    paired = candidate[["structure_id", "scaffold_group_id", "observed_pic50", "predicted_pic50"]].merge(
        baseline[["structure_id", "predicted_pic50"]],
        on="structure_id",
        suffixes=("_candidate", "_baseline"),
        validate="one_to_one",
    )
    paired["candidate_loss"] = np.abs(paired["observed_pic50"] - paired["predicted_pic50_candidate"])
    paired["baseline_loss"] = np.abs(paired["observed_pic50"] - paired["predicted_pic50_baseline"])
    paired["loss_delta"] = paired["candidate_loss"] - paired["baseline_loss"]
    groups = [group for _, group in paired.groupby("scaffold_group_id", sort=True)]
    if len(groups) < 2:
        raise DiscoveryWorkerError("too few shared scaffold groups for paired bootstrap")
    rng = np.random.default_rng(seed)
    deltas = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        selected = rng.integers(0, len(groups), size=len(groups))
        sampled = np.concatenate([groups[index]["loss_delta"].to_numpy() for index in selected])
        deltas[replicate] = float(np.mean(sampled))
    group_deltas = np.asarray([group["loss_delta"].mean() for group in groups], dtype=np.float64)
    permutation_replicates = min(replicates, 20_000)
    permuted = np.empty(permutation_replicates, dtype=np.float64)
    for replicate in range(permutation_replicates):
        signs = rng.choice(np.array([-1.0, 1.0]), size=len(group_deltas))
        permuted[replicate] = float(np.mean(group_deltas * signs))
    point = float(paired["loss_delta"].mean())
    return {
        "paired_structures": len(paired),
        "paired_scaffolds": len(groups),
        "mae_delta_candidate_minus_baseline": point,
        "bootstrap_lower_95": float(np.quantile(deltas, 0.025)),
        "bootstrap_upper_95": float(np.quantile(deltas, 0.975)),
        "probability_candidate_better": float(np.mean(deltas < 0)),
        "group_sign_permutation_two_sided_p": float(
            (1 + np.sum(np.abs(permuted) >= abs(float(group_deltas.mean())))) / (1 + permutation_replicates)
        ),
    }


def _biological_process(feature: str) -> str:
    if feature.startswith("interaction__"):
        return "targeted_parameter_interaction"
    if "autocorr3d" in feature or "dipole" in feature or "charge" in feature:
        return "spatial_electrostatics"
    if "MolLogP" in feature or "SlogP" in feature:
        return "lipophilic_partitioning"
    if "TPSA" in feature or "polar" in feature or "HAccept" in feature or "HDon" in feature:
        return "polarity_desolvation_proxy"
    if "arom" in feature.lower() or "benzene" in feature.lower() or "Ring" in feature:
        return "aromatic_hydrophobic_architecture"
    if "energy" in feature or "conformer" in feature or "rotatable" in feature:
        return "conformational_ensemble_flexibility"
    if "shape" in feature or "whim" in feature or "pmi" in feature:
        return "ligand_shape"
    if feature.startswith("morgan__") or feature.startswith("maccs__"):
        return "substructure_identity_not_directly_mechanistic"
    return "other_molecular_descriptor"


def _analyze(args: argparse.Namespace) -> dict[str, Any]:
    prepared = Path(args.prepared_root).resolve()
    results_root = Path(args.results_root).resolve()
    output = Path(args.output_root).resolve()
    _validate_prepared_fast(prepared)
    if (output / "validation.json").is_file():
        return json.loads((output / "validation.json").read_text())
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise DiscoveryWorkerError("analysis output must be empty or already validated")
    metric_paths = sorted((results_root / "units").glob("*/metrics.json"))
    if not metric_paths:
        raise DiscoveryWorkerError("no completed worker units were found")
    metrics_documents: list[dict[str, Any]] = []
    for path in metric_paths:
        document = json.loads(path.read_text())
        if document.get("status") != "passed":
            continue
        _validate_binding(document["prediction_artifact"])
        _validate_binding(document["importance_artifact"])
        metrics_documents.append(document)
    if not metrics_documents:
        raise DiscoveryWorkerError("no valid completed worker units were found")
    chemprop_metric_rows, chemprop_prediction_frames, chemprop_integration = _complete_chemprop_outer_oof(
        prepared, results_root
    )
    unit_metric_rows = [_flatten_unit_metrics(item) for item in metrics_documents]
    unit_metric_rows.extend(chemprop_metric_rows)
    unit_metrics = pd.DataFrame(unit_metric_rows)
    model_rankings = (
        unit_metrics.groupby(["stage", "model_id", "unit_kind"], as_index=False)
        .agg(
            completed_units=("unit_id", "size"),
            mean_mae=("mae", "mean"),
            sd_mae=("mae", "std"),
            mean_rmse=("rmse", "mean"),
            mean_r2=("r2", "mean"),
            mean_spearman=("spearman", "mean"),
            mean_fraction_within_0p5=("fraction_within_0p5", "mean"),
            total_fit_seconds=("fit_elapsed_seconds", "sum"),
        )
        .sort_values(["stage", "mean_mae", "model_id"], kind="stable")
    )
    model_rankings["rank_within_stage"] = model_rankings.groupby("stage").cumcount() + 1

    prediction_frames: list[pd.DataFrame] = []
    importance_frames: list[pd.DataFrame] = []
    document_by_unit = {document["unit_id"]: document for document in metrics_documents}
    for document in document_by_unit.values():
        if document["stage"] == "outer":
            prediction = pd.read_parquet(document["prediction_artifact"]["path"])
            prediction["outer_fold"] = document["outer_fold"]
            prediction_frames.append(prediction)
            importance = pd.read_parquet(document["importance_artifact"]["path"])
            if not importance.empty:
                importance["outer_fold"] = document["outer_fold"]
                importance_frames.append(importance)
    prediction_frames.extend(chemprop_prediction_frames)
    outer_predictions = (
        pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    )
    outer_model_summary_rows: list[dict[str, Any]] = []
    if not outer_predictions.empty:
        for model_id, model_frame in outer_predictions.groupby("model_id", sort=True):
            if model_frame["structure_id"].duplicated().any():
                raise DiscoveryWorkerError(f"outer OOF structures are duplicated for model {model_id}")
            outer_model_summary_rows.append(
                {
                    "model_id": model_id,
                    **_metrics(
                        model_frame["observed_pic50"].to_numpy(dtype=np.float64),
                        model_frame["predicted_pic50"].to_numpy(dtype=np.float64),
                    ),
                }
            )
    outer_model_summary = pd.DataFrame(outer_model_summary_rows)
    if not outer_model_summary.empty:
        outer_model_summary = outer_model_summary.sort_values("mae", kind="stable")
        outer_model_summary["rank_by_mae"] = np.arange(1, len(outer_model_summary) + 1)

    comparison_rows: list[dict[str, Any]] = []
    baseline_model = args.baseline_model
    if not outer_predictions.empty and baseline_model is None:
        candidate_names = sorted(set(outer_predictions["model_id"]))
        baseline_model = next(
            (name for name in candidate_names if "baseline" in name.lower()),
            candidate_names[0],
        )
    if baseline_model is not None and baseline_model in set(outer_predictions.get("model_id", [])):
        baseline = outer_predictions[outer_predictions["model_id"].eq(baseline_model)]
        for model_id, candidate in outer_predictions.groupby("model_id", sort=True):
            if model_id == baseline_model:
                continue
            row = _grouped_model_bootstrap(
                candidate,
                baseline,
                replicates=args.bootstrap_replicates,
                seed=args.seed + len(comparison_rows),
            )
            row.update(candidate_model=model_id, baseline_model=baseline_model)
            comparison_rows.append(row)
    comparisons = pd.DataFrame(comparison_rows)

    feature_stability = pd.DataFrame()
    if importance_frames:
        importance = pd.concat(importance_frames, ignore_index=True)
        importance["direction_sign"] = np.sign(
            pd.to_numeric(importance["signed_spearman_fit"], errors="coerce")
        )
        feature_stability = (
            importance.groupby(["model_id", "feature_name"], as_index=False)
            .agg(
                outer_units_selected=("unit_id", "nunique"),
                mean_importance=("importance", "mean"),
                sd_importance=("importance", "std"),
                mean_signed_spearman_fit=("signed_spearman_fit", "mean"),
                direction_sign_mean=("direction_sign", "mean"),
            )
            .sort_values(["model_id", "mean_importance"], ascending=[True, False], kind="stable")
        )
        unit_counts = importance.groupby("model_id")["unit_id"].nunique().to_dict()
        feature_stability["selection_frequency"] = feature_stability.apply(
            lambda row: row["outer_units_selected"] / unit_counts[row["model_id"]], axis=1
        )
        feature_stability["direction_consistency"] = feature_stability["direction_sign_mean"].abs()
        feature_stability["biological_process_hypothesis"] = feature_stability["feature_name"].map(
            _biological_process
        )
        feature_stability["causal_interpretation_allowed"] = False

    assay_residual_rows: list[dict[str, Any]] = []
    if not outer_predictions.empty:
        metadata = pq.read_table(
            prepared / "exact_train_cache.parquet",
            columns=[
                "structure_id",
                "measurement_modality",
                "automation_class",
                "assay_family",
                "source_family",
                "protocol_completeness_mean",
            ],
        ).to_pandas()
        predictions_with_metadata = outer_predictions.drop(
            columns=["measurement_modality", "automation_class"], errors="ignore"
        ).merge(metadata, on="structure_id", validate="many_to_one")
        predictions_with_metadata["absolute_error"] = np.abs(
            predictions_with_metadata["residual_observed_minus_predicted"]
        )
        predictions_with_metadata["protocol_completeness_band"] = pd.cut(
            predictions_with_metadata["protocol_completeness_mean"],
            bins=[-np.inf, 0, 2, 4, np.inf],
            labels=["zero", "one_to_two", "three_to_four", "five_plus"],
        ).astype(str)
        for model_id, model_frame in predictions_with_metadata.groupby("model_id"):
            for dimension in (
                "measurement_modality",
                "automation_class",
                "assay_family",
                "source_family",
                "protocol_completeness_band",
            ):
                for level, level_frame in model_frame.groupby(dimension, dropna=False):
                    if len(level_frame) < args.minimum_subgroup_size:
                        continue
                    assay_residual_rows.append(
                        {
                            "model_id": model_id,
                            "dimension": dimension,
                            "level": str(level),
                            "structures": len(level_frame),
                            "mae": float(level_frame["absolute_error"].mean()),
                            "mean_signed_error_observed_minus_predicted": float(
                                level_frame["residual_observed_minus_predicted"].mean()
                            ),
                            "exploratory_only": True,
                            "confounded_by_chemistry_and_source": True,
                        }
                    )
    assay_residuals = pd.DataFrame(assay_residual_rows)

    mmp_rows: list[dict[str, Any]] = []
    mmp_root = Path(args.mmp_root).resolve()
    effects_path = mmp_root / "training_mmp_effects.parquet"
    if not outer_predictions.empty and effects_path.is_file():
        effects = pq.read_table(
            effects_path,
            columns=[
                "pair_id",
                "structure_id_a",
                "structure_id_b",
                "delta_pic50_b_minus_a",
                "activity_cliff_ge_1_pic50",
                "exploratory_training_only",
            ],
            filters=[("exploratory_training_only", "=", True)],
        ).to_pandas()
        for model_id, model_frame in outer_predictions.groupby("model_id"):
            prediction_map = model_frame.set_index("structure_id")["predicted_pic50"]
            selected = effects[
                effects["structure_id_a"].isin(prediction_map.index)
                & effects["structure_id_b"].isin(prediction_map.index)
            ].copy()
            if len(selected) < args.minimum_subgroup_size:
                continue
            selected["predicted_delta_pic50_b_minus_a"] = selected["structure_id_b"].map(
                prediction_map
            ) - selected["structure_id_a"].map(prediction_map)
            observed_delta = selected["delta_pic50_b_minus_a"].to_numpy(dtype=np.float64)
            predicted_delta = selected["predicted_delta_pic50_b_minus_a"].to_numpy(dtype=np.float64)
            mmp_rows.append(
                {
                    "model_id": model_id,
                    "training_only_mmp_pairs": len(selected),
                    "delta_spearman": float(spearmanr(observed_delta, predicted_delta).statistic),
                    "direction_accuracy": float(np.mean(np.sign(observed_delta) == np.sign(predicted_delta))),
                    "delta_mae": float(np.mean(np.abs(observed_delta - predicted_delta))),
                    "observed_activity_cliffs": int(selected["activity_cliff_ge_1_pic50"].sum()),
                    "exploratory_only": True,
                    "causal_or_mechanistic_claim_allowed": False,
                }
            )
    mmp_relationships = pd.DataFrame(mmp_rows)

    _atomic_parquet(output / "unit_metrics.parquet", unit_metrics)
    _atomic_parquet(output / "model_rankings.parquet", model_rankings)
    _atomic_parquet(output / "outer_oof_predictions.parquet", outer_predictions)
    _atomic_parquet(output / "outer_model_summary.parquet", outer_model_summary)
    _atomic_parquet(output / "paired_model_uncertainty.parquet", comparisons)
    _atomic_parquet(output / "feature_direction_stability.parquet", feature_stability)
    _atomic_parquet(output / "assay_quality_residuals.parquet", assay_residuals)
    _atomic_parquet(output / "training_mmp_relationships.parquet", mmp_relationships)
    _atomic_json(output / "chemprop_integration.json", chemprop_integration)

    lines = [
        "# Train-only hERG discovery campaign analysis",
        "",
        "- All model selection results are nested scaffold evaluations inside the repository training partition.",
        "- Repository validation and test outcomes were never loaded by this worker.",
        "- The quantitative target is WT-or-unspecified hERG, not uniformly experimentally confirmed WT.",
        "- Most records lack complete assay, automation, temperature, protocol, and cell-system context.",
        "- Feature importance and signed association generate biological hypotheses; neither proves causality.",
        "- The broad fixed-dose surface is not pooled into pIC50 because a threshold response is not a "
        "continuous potency measurement.",
    ]
    if not outer_model_summary.empty:
        best = outer_model_summary.iloc[0]
        lines.extend(
            [
                "",
                "## Current outer-fold leader",
                "",
                f"- Model: {best['model_id']}.",
                f"- MAE: {best['mae']:.4f} log units.",
                f"- RMSE: {best['rmse']:.4f} log units.",
                f"- Fraction within 0.5 log: {best['fraction_within_0p5']:.3f}.",
                "- Treat this as train-only hypothesis prioritization until a prespecified candidate is frozen.",
            ]
        )
    if not feature_stability.empty:
        stable = feature_stability.sort_values(
            ["selection_frequency", "mean_importance"], ascending=[False, False]
        ).head(15)
        lines.extend(["", "## Stable feature hypotheses", ""])
        for row in stable.itertuples():
            direction = (
                "higher feature values associate with stronger measured potency"
                if (row.mean_signed_spearman_fit or 0) > 0
                else "higher feature values associate with weaker measured potency"
            )
            lines.append(
                f"- {row.feature_name}: {row.biological_process_hypothesis}; {direction}; "
                f"selection frequency {row.selection_frequency:.2f}."
            )
    lines.extend(
        [
            "",
            "## Interpretation boundaries",
            "",
            "- Select feature families and parameters with inner folds, then estimate the selected workflow "
            "with outer folds.",
            "- Do not select a final model by repeatedly reading repository validation performance.",
            "- Freeze one candidate and analysis plan before any locked or external evaluation.",
            "- Validate promising electrostatic, polarity, or interaction signals using matched pairs and "
            "receptor-aware calculations before describing them as biological mechanisms.",
        ]
    )
    _atomic_replace(output / "analysis.md", lambda path: path.write_text("\n".join(lines) + "\n", "utf-8"))
    artifact_names = [
        "unit_metrics.parquet",
        "model_rankings.parquet",
        "outer_oof_predictions.parquet",
        "outer_model_summary.parquet",
        "paired_model_uncertainty.parquet",
        "feature_direction_stability.parquet",
        "assay_quality_residuals.parquet",
        "training_mmp_relationships.parquet",
        "chemprop_integration.json",
        "analysis.md",
    ]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "command": "analyze",
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "scientific_scope": {
            "source_partition": "train_only",
            "repository_validation_outcomes_loaded": False,
            "repository_test_outcomes_loaded": False,
            "WT_scope": "wild_type_or_unspecified",
            "assay_modality_limitations_preserved": True,
            "causal_or_mechanistic_claim_allowed": False,
        },
        "counts": {
            "completed_units": len(unit_metrics),
            "outer_prediction_rows": len(outer_predictions),
            "ranked_models": len(model_rankings),
            "chemprop_outer_folds_integrated": int(chemprop_integration.get("integrated_outer_folds", 0)),
            "chemprop_prediction_rows_integrated": int(
                chemprop_integration.get("integrated_prediction_rows", 0)
            ),
        },
        "artifacts": [_binding(output / name) for name in artifact_names],
    }
    _atomic_json(output / "manifest.json", manifest)
    validation = {
        "status": "passed",
        **manifest["counts"],
        "repository_validation_outcomes_loaded": False,
        "repository_test_outcomes_loaded": False,
        "predictive_superiority_established": False,
    }
    _atomic_json(output / "validation.json", validation)
    return validation


def _add_unit_split_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--prepared-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--unit-id", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--stage", choices=["inner", "outer"], required=True)
    parser.add_argument("--outer-fold", type=int, required=True)
    parser.add_argument("--inner-fold", type=int)
    parser.add_argument("--seed", type=int, default=SEED)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--repo-root", default=".")
    prepare.add_argument("--matrix-root", default=str(DEFAULT_MATRIX_ROOT))
    prepare.add_argument("--observations", default=str(DEFAULT_OBSERVATIONS))
    prepare.add_argument("--output-root", required=True)
    prepare.add_argument("--outer-folds", type=int, default=5)
    prepare.add_argument("--inner-folds", type=int, default=3)
    prepare.set_defaults(function=_prepare)

    tree = subparsers.add_parser("tree-unit")
    _add_unit_split_arguments(tree)
    tree.add_argument("--engine", choices=["xgboost", "lightgbm"], required=True)
    tree.add_argument("--groups", required=True)
    tree.add_argument("--params-json")
    tree.add_argument("--workers", type=int, default=1)
    tree.set_defaults(function=_tree_unit)

    classical = subparsers.add_parser("classical-unit")
    _add_unit_split_arguments(classical)
    classical.add_argument("--model", choices=["ridge", "extratrees", "randomforest"], required=True)
    classical.add_argument("--groups", required=True)
    classical.add_argument("--params-json")
    classical.add_argument("--workers", type=int, default=1)
    classical.add_argument("--maximum-features", type=int, default=768)
    classical.set_defaults(function=_classical_unit)

    similarity = subparsers.add_parser("similarity-unit")
    _add_unit_split_arguments(similarity)
    similarity.add_argument("--neighbors", type=int, default=10)
    similarity.add_argument("--similarity-floor", type=float, default=0.0)
    similarity.add_argument("--weight-power", type=float, default=2.0)
    similarity.add_argument("--chunk-size", type=int, default=128)
    similarity.set_defaults(function=_similarity_unit)

    chemprop = subparsers.add_parser("chemprop-prepare")
    chemprop.add_argument("--prepared-root", required=True)
    chemprop.add_argument("--output-root", required=True)
    chemprop.add_argument("--outer-fold", type=int)
    chemprop.set_defaults(function=_chemprop_prepare)

    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("--prepared-root", required=True)
    analyze.add_argument("--results-root", required=True)
    analyze.add_argument("--output-root", required=True)
    analyze.add_argument("--baseline-model")
    analyze.add_argument("--mmp-root", default=str(DEFAULT_MMP_ROOT))
    analyze.add_argument("--bootstrap-replicates", type=int, default=5000)
    analyze.add_argument("--minimum-subgroup-size", type=int, default=30)
    analyze.add_argument("--seed", type=int, default=SEED)
    analyze.set_defaults(function=_analyze)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = args.function(args)
    print(_canonical_json(result), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
