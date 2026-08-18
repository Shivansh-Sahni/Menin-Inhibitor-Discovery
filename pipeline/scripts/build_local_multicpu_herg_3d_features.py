#!/usr/bin/env python3
"""Build a bounded, resumable, hERG-focused conformer feature surface.

This is a local CPU feature-generation pilot for the quantitative hERG surface.
It deliberately avoids labels and outcome values.  It generates several
conformers per parent structure, optimizes them with MMFF94s (UFF fallback), and
stores ensemble geometry, charge/polar-exposure, internal-polar-contact,
flexibility, microstate-complexity, and dominant-conformer 3D descriptor blocks.

The features are physics-informed approximations, not docking, membrane PMFs,
free energies, kinetic rates, pKa predictions, or receptor-bound simulations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import shutil
import sys
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import Chem, RDLogger, rdBase
from rdkit.Chem import (
    AllChem,
    Descriptors3D,
    Lipinski,
    MolStandardize,
    rdFreeSASA,
    rdMolAlign,
    rdMolDescriptors,
)

SCHEMA_VERSION = "platform-local-herg-conformer-features/1.0"
INPUT_SURFACE = Path(
    "research/data/platform/processed/herg_hierarchy/v1_6_training_surfaces/"
    "herg_training_observations.parquet"
)
INPUT_MANIFEST = Path(
    "research/data/platform/processed/herg_hierarchy/v1_6_training_surfaces/"
    "herg_training_surfaces_manifest.json"
)
STRUCTURE_MASTER = Path(
    "research/data/platform/processed/herg_hierarchy/v1_3_master/structure_master.parquet"
)
F2D_MAPPING = Path("research/local_runs/local_multicpu_2d_features_v1/source_to_feature_mapping.parquet")
F2D_MANIFEST = Path("research/local_runs/local_multicpu_2d_features_v1/feature_cache_manifest.json")
ROUTING_COLUMN = "standardized_pic50_primary"
R_KCAL_MOL_K = 0.00198720425864083
TEMPERATURE_K = 298.15
VECTOR_BLOCKS = {
    "autocorr3d": 80,
    "whim": 114,
}
CONFORMER_SCALARS = (
    "pmi1",
    "pmi2",
    "pmi3",
    "npr1",
    "npr2",
    "asphericity",
    "eccentricity",
    "inertial_shape_factor",
    "radius_of_gyration",
    "spherocity",
    "pbf",
    "sasa",
    "heavy_pair_distance_mean",
    "heavy_pair_distance_max",
    "heavy_contact_density_4p5A",
    "polar_radial_exposure",
    "internal_polar_contact_count",
    "gasteiger_dipole_proxy_eA",
    "absolute_charge_radius_A",
)

_BASIC_SMARTS = tuple(
    Chem.MolFromSmarts(value)
    for value in (
        "[N;H0,H1,H2;!$(N-[C,S,P]=[O,S,N]);!$(N[a]);!$([N+])]",
        "[nH0;+0]",
        "[N;H1,H2]-[C;X3](=[N;H0,H1,H2])",
        "[N;H0,H1,H2]-[C;X3](=[N;H0,H1,H2])-[N;H0,H1,H2]",
    )
)
_ACIDIC_SMARTS = tuple(
    Chem.MolFromSmarts(value)
    for value in (
        "[C,S,P](=[O,S])-[O;H1,-1]",
        "[nH]",
        "[N;H1]-[S](=O)=O",
        "[O;H1]-[c,n]",
    )
)


class FeatureBuildError(RuntimeError):
    """Raised when the feature surface cannot be built or validated."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _schema_sha(path: Path) -> str:
    return hashlib.sha256(pq.read_schema(path).serialize().to_pybytes()).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _initialize_worker() -> None:
    RDLogger.DisableLog("rdApp.warning")
    RDLogger.DisableLog("rdApp.info")


def _seed(structure_id: str) -> int:
    return int(hashlib.sha256(structure_id.encode()).hexdigest()[:7], 16)


def _count_matches(molecule: Chem.Mol, patterns: Iterable[Chem.Mol | None]) -> int:
    atoms: set[int] = set()
    for pattern in patterns:
        if pattern is None:
            continue
        for match in molecule.GetSubstructMatches(pattern):
            atoms.update(match)
    return len(atoms)


def _tautomer_count(molecule: Chem.Mol) -> tuple[int, bool]:
    enumerator = MolStandardize.rdMolStandardize.TautomerEnumerator()
    enumerator.SetMaxTautomers(16)
    enumerator.SetMaxTransforms(100)
    try:
        enumerated = enumerator.Enumerate(molecule)
        count = len(enumerated)
        return count, count >= 16
    except Exception:
        return 0, False


def _embed_and_optimize(
    molecule: Chem.Mol,
    *,
    structure_id: str,
    requested_conformers: int,
    retained_conformers: int,
    max_iterations: int,
) -> tuple[Chem.Mol, list[int], np.ndarray, str, int]:
    hydrogenated = Chem.AddHs(molecule)
    params = AllChem.ETKDGv3()
    params.randomSeed = _seed(structure_id)
    params.numThreads = 1
    params.pruneRmsThresh = 0.5
    params.useSmallRingTorsions = True
    params.useMacrocycleTorsions = True
    params.enforceChirality = True
    conf_ids = list(
        AllChem.EmbedMultipleConfs(
            hydrogenated,
            numConfs=requested_conformers,
            params=params,
        )
    )
    if not conf_ids:
        raise FeatureBuildError("conformer_embedding_failed")

    method = "MMFF94s"
    try:
        if AllChem.MMFFHasAllMoleculeParams(hydrogenated):
            optimized = AllChem.MMFFOptimizeMoleculeConfs(
                hydrogenated,
                numThreads=1,
                maxIters=max_iterations,
                mmffVariant="MMFF94s",
            )
        else:
            method = "UFF"
            optimized = AllChem.UFFOptimizeMoleculeConfs(
                hydrogenated,
                numThreads=1,
                maxIters=max_iterations,
            )
    except Exception as error:
        raise FeatureBuildError(f"force_field_failed:{type(error).__name__}") from error
    if len(optimized) != len(conf_ids):
        raise FeatureBuildError("force_field_result_count_mismatch")
    energies = np.asarray([float(value[1]) for value in optimized], dtype=np.float64)
    if not np.isfinite(energies).all():
        raise FeatureBuildError("nonfinite_conformer_energy")
    order = np.argsort(energies, kind="stable")[:retained_conformers]
    selected_ids = [int(conf_ids[int(index)]) for index in order]
    selected_energies = energies[order]
    unconverged = sum(int(optimized[int(index)][0] != 0) for index in order)
    return hydrogenated, selected_ids, selected_energies, method, unconverged


def _weights(energies: np.ndarray) -> np.ndarray:
    delta = energies - float(np.min(energies))
    raw = np.exp(-np.minimum(delta / (R_KCAL_MOL_K * TEMPERATURE_K), 700.0))
    return raw / float(raw.sum())


def _weighted(values: np.ndarray, weights: np.ndarray, prefix: str) -> dict[str, float]:
    mean = float(np.sum(values * weights))
    variance = float(np.sum(weights * np.square(values - mean)))
    return {
        f"{prefix}__mean": mean,
        f"{prefix}__sd": math.sqrt(max(variance, 0.0)),
        f"{prefix}__min": float(np.min(values)),
        f"{prefix}__max": float(np.max(values)),
    }


def _coordinates(molecule: Chem.Mol, conf_id: int, atom_indices: list[int]) -> np.ndarray:
    conformer = molecule.GetConformer(conf_id)
    return np.asarray(
        [
            [
                conformer.GetAtomPosition(index).x,
                conformer.GetAtomPosition(index).y,
                conformer.GetAtomPosition(index).z,
            ]
            for index in atom_indices
        ],
        dtype=np.float64,
    )


def _conformer_scalars(
    molecule: Chem.Mol,
    conf_id: int,
    charges: np.ndarray,
) -> dict[str, float]:
    heavy = [atom.GetIdx() for atom in molecule.GetAtoms() if atom.GetAtomicNum() > 1]
    polar = [atom.GetIdx() for atom in molecule.GetAtoms() if atom.GetAtomicNum() in {7, 8, 15, 16}]
    xyz = _coordinates(molecule, conf_id, heavy)
    centroid = xyz.mean(axis=0)
    centered = xyz - centroid
    radii = np.linalg.norm(centered, axis=1)
    pairwise = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=2)
    upper = pairwise[np.triu_indices(len(heavy), 1)]
    contact_density = 0.0 if upper.size == 0 else float(np.mean(upper < 4.5))
    polar_exposure = 0.0
    if polar:
        polar_xyz = _coordinates(molecule, conf_id, polar)
        polar_exposure = float(
            np.mean(np.linalg.norm(polar_xyz - centroid, axis=1)) / max(float(np.max(radii)), 1e-6)
        )

    internal_polar_contacts = 0
    if len(polar) > 1:
        polar_xyz = _coordinates(molecule, conf_id, polar)
        polar_distances = np.linalg.norm(polar_xyz[:, None, :] - polar_xyz[None, :, :], axis=2)
        topology = Chem.GetDistanceMatrix(molecule)
        for left in range(len(polar)):
            for right in range(left + 1, len(polar)):
                if polar_distances[left, right] <= 3.5 and topology[polar[left], polar[right]] >= 4:
                    internal_polar_contacts += 1

    all_xyz = _coordinates(molecule, conf_id, list(range(molecule.GetNumAtoms())))
    charge_centered = all_xyz - centroid
    dipole_proxy = float(np.linalg.norm(np.sum(charges[:, None] * charge_centered, axis=0)))
    charge_radius = float(
        np.sum(np.abs(charges) * np.linalg.norm(charge_centered, axis=1))
        / max(float(np.sum(np.abs(charges))), 1e-8)
    )
    try:
        radii_sasa = rdFreeSASA.classifyAtoms(molecule)
        sasa = float(rdFreeSASA.CalcSASA(molecule, radii_sasa, confIdx=conf_id))
    except Exception:
        sasa = math.nan
    return {
        "pmi1": float(Descriptors3D.PMI1(molecule, confId=conf_id)),
        "pmi2": float(Descriptors3D.PMI2(molecule, confId=conf_id)),
        "pmi3": float(Descriptors3D.PMI3(molecule, confId=conf_id)),
        "npr1": float(Descriptors3D.NPR1(molecule, confId=conf_id)),
        "npr2": float(Descriptors3D.NPR2(molecule, confId=conf_id)),
        "asphericity": float(Descriptors3D.Asphericity(molecule, confId=conf_id)),
        "eccentricity": float(Descriptors3D.Eccentricity(molecule, confId=conf_id)),
        "inertial_shape_factor": float(Descriptors3D.InertialShapeFactor(molecule, confId=conf_id)),
        "radius_of_gyration": float(Descriptors3D.RadiusOfGyration(molecule, confId=conf_id)),
        "spherocity": float(Descriptors3D.SpherocityIndex(molecule, confId=conf_id)),
        "pbf": float(Descriptors3D.PBF(molecule, confId=conf_id)),
        "sasa": sasa,
        "heavy_pair_distance_mean": float(np.mean(upper)) if upper.size else 0.0,
        "heavy_pair_distance_max": float(np.max(upper)) if upper.size else 0.0,
        "heavy_contact_density_4p5A": contact_density,
        "polar_radial_exposure": polar_exposure,
        "internal_polar_contact_count": float(internal_polar_contacts),
        "gasteiger_dipole_proxy_eA": dipole_proxy,
        "absolute_charge_radius_A": charge_radius,
    }


def _dominant_vectors(molecule: Chem.Mol, conf_id: int) -> dict[str, float | None]:
    functions = {
        "autocorr3d": rdMolDescriptors.CalcAUTOCORR3D,
        "whim": rdMolDescriptors.CalcWHIM,
    }
    result: dict[str, float | None] = {}
    for block, expected in VECTOR_BLOCKS.items():
        try:
            values = list(functions[block](molecule, confId=conf_id))
        except Exception:
            values = []
        if len(values) != expected:
            values = [math.nan] * expected
        for index, value in enumerate(values):
            numeric = float(value)
            result[f"dominant_{block}__{index:03d}"] = numeric if math.isfinite(numeric) else None
    return result


def _output_schema() -> pa.Schema:
    fields = [
        pa.field("feature_order", pa.int64(), nullable=False),
        pa.field("structure_id", pa.large_string(), nullable=False),
        pa.field("f2d_feature_id", pa.large_string(), nullable=False),
        pa.field("feature_status", pa.large_string(), nullable=False),
        pa.field("error_class", pa.large_string(), nullable=True),
        pa.field("formal_charge", pa.int16(), nullable=True),
        pa.field("basic_site_proxy_count", pa.int16(), nullable=True),
        pa.field("acidic_site_proxy_count", pa.int16(), nullable=True),
        pa.field("tautomer_count_capped16", pa.int16(), nullable=True),
        pa.field("tautomer_enumeration_capped", pa.bool_(), nullable=True),
        pa.field("rotatable_bond_count", pa.int16(), nullable=True),
        pa.field("force_field", pa.large_string(), nullable=True),
        pa.field("embedded_conformer_count", pa.int16(), nullable=True),
        pa.field("retained_conformer_count", pa.int16(), nullable=True),
        pa.field("unconverged_retained_count", pa.int16(), nullable=True),
        pa.field("energy_min_kcal_mol", pa.float32(), nullable=True),
        pa.field("energy_range_kcal_mol", pa.float32(), nullable=True),
        pa.field("effective_conformer_count", pa.float32(), nullable=True),
        pa.field("dominant_conformer_weight", pa.float32(), nullable=True),
    ]
    for name in CONFORMER_SCALARS:
        for suffix in ("mean", "sd", "min", "max"):
            fields.append(pa.field(f"ensemble_{name}__{suffix}", pa.float32(), nullable=True))
    fields.extend(
        [
            pa.field("retained_pairwise_rmsd_mean_A", pa.float32(), nullable=True),
            pa.field("retained_pairwise_rmsd_max_A", pa.float32(), nullable=True),
            pa.field("energy_polar_exposure_correlation", pa.float32(), nullable=True),
        ]
    )
    for block, count in VECTOR_BLOCKS.items():
        for index in range(count):
            fields.append(pa.field(f"dominant_{block}__{index:03d}", pa.float32(), nullable=True))
    return pa.schema(fields)


def _pairwise_rmsd(molecule: Chem.Mol, conf_ids: list[int]) -> tuple[float, float]:
    values: list[float] = []
    for left in range(len(conf_ids)):
        for right in range(left + 1, len(conf_ids)):
            try:
                values.append(
                    float(rdMolAlign.GetBestRMS(molecule, molecule, conf_ids[left], conf_ids[right]))
                )
            except Exception:
                continue
    if not values:
        return 0.0, 0.0
    return float(np.mean(values)), float(np.max(values))


def _compute_one(item: tuple[Any, ...]) -> dict[str, Any]:
    (
        feature_order,
        structure_id,
        smiles,
        f2d_feature_id,
        requested_conformers,
        retained_conformers,
        max_iterations,
    ) = item
    base: dict[str, Any] = {
        "feature_order": int(feature_order),
        "structure_id": str(structure_id),
        "f2d_feature_id": str(f2d_feature_id),
        "feature_status": "ok",
        "error_class": None,
    }
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return {**base, "feature_status": "input_invalid", "error_class": "rdkit_parse_failed"}
    try:
        tautomer_count, tautomer_capped = _tautomer_count(molecule)
        base.update(
            {
                "formal_charge": int(Chem.GetFormalCharge(molecule)),
                "basic_site_proxy_count": _count_matches(molecule, _BASIC_SMARTS),
                "acidic_site_proxy_count": _count_matches(molecule, _ACIDIC_SMARTS),
                "tautomer_count_capped16": int(tautomer_count),
                "tautomer_enumeration_capped": bool(tautomer_capped),
                "rotatable_bond_count": int(Lipinski.NumRotatableBonds(molecule)),
            }
        )
        embedded, conf_ids, energies, force_field, unconverged = _embed_and_optimize(
            molecule,
            structure_id=str(structure_id),
            requested_conformers=int(requested_conformers),
            retained_conformers=int(retained_conformers),
            max_iterations=int(max_iterations),
        )
        try:
            AllChem.ComputeGasteigerCharges(embedded)
            charges = np.asarray(
                [float(atom.GetProp("_GasteigerCharge")) for atom in embedded.GetAtoms()],
                dtype=np.float64,
            )
            if not np.isfinite(charges).all():
                charges = np.zeros(embedded.GetNumAtoms(), dtype=np.float64)
        except Exception:
            charges = np.zeros(embedded.GetNumAtoms(), dtype=np.float64)
        weights = _weights(energies)
        scalars = [_conformer_scalars(embedded, conf_id, charges) for conf_id in conf_ids]
        base.update(
            {
                "force_field": force_field,
                "embedded_conformer_count": embedded.GetNumConformers(),
                "retained_conformer_count": len(conf_ids),
                "unconverged_retained_count": int(unconverged),
                "energy_min_kcal_mol": float(np.min(energies)),
                "energy_range_kcal_mol": float(np.max(energies) - np.min(energies)),
                "effective_conformer_count": float(1.0 / np.sum(np.square(weights))),
                "dominant_conformer_weight": float(weights[0]),
            }
        )
        for name in scalars[0]:
            values = np.asarray([row[name] for row in scalars], dtype=np.float64)
            finite = np.isfinite(values)
            if finite.all():
                base.update(_weighted(values, weights, f"ensemble_{name}"))
            else:
                for suffix in ("mean", "sd", "min", "max"):
                    base[f"ensemble_{name}__{suffix}"] = None
        rmsd_mean, rmsd_max = _pairwise_rmsd(embedded, conf_ids)
        base["retained_pairwise_rmsd_mean_A"] = rmsd_mean
        base["retained_pairwise_rmsd_max_A"] = rmsd_max
        polar_values = np.asarray([row["polar_radial_exposure"] for row in scalars], dtype=np.float64)
        if len(conf_ids) > 1 and float(np.std(energies)) > 1e-12 and float(np.std(polar_values)) > 1e-12:
            base["energy_polar_exposure_correlation"] = float(np.corrcoef(energies, polar_values)[0, 1])
        else:
            base["energy_polar_exposure_correlation"] = 0.0
        base.update(_dominant_vectors(embedded, conf_ids[0]))
        return base
    except Exception as error:
        return {
            **base,
            "feature_status": "conformer_failed",
            "error_class": f"{type(error).__name__}:{str(error)[:160]}",
        }


def _load_index(root: Path, limit: int | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = ["structure_id", ROUTING_COLUMN]
    source = pq.read_table(root / INPUT_SURFACE, columns=columns).to_pandas()
    source = source[source[ROUTING_COLUMN].eq(True)].copy()  # noqa: E712
    membership = source[["structure_id"]].drop_duplicates("structure_id")
    master = pq.read_table(
        root / STRUCTURE_MASTER,
        columns=[
            "structure_id",
            "standardized_smiles",
            "standard_inchi_key",
            "model_split",
            "scaffold_group_id",
        ],
    ).to_pandas()
    routing = (
        membership.merge(master, on="structure_id", how="left", validate="one_to_one")
        .sort_values("structure_id", kind="stable")
        .reset_index(drop=True)
    )
    if routing.isna().any().any():
        raise FeatureBuildError("quantitative membership is not closed against structure master")
    if limit is not None:
        routing = routing.head(limit).copy()
    mapping = pq.read_table(root / F2D_MAPPING).to_pandas()
    mapping = mapping[mapping["source_family"].eq("herg")][["source_structure_id", "feature_id"]].rename(
        columns={"source_structure_id": "structure_id", "feature_id": "f2d_feature_id"}
    )
    routing = routing.merge(mapping, on="structure_id", how="left", validate="one_to_one")
    if routing["f2d_feature_id"].isna().any():
        raise FeatureBuildError("quantitative hERG structures are missing from completed 2D cache")
    routing.insert(0, "feature_order", np.arange(len(routing), dtype=np.int64))
    index = routing[["feature_order", "structure_id", "standardized_smiles", "f2d_feature_id"]].copy()
    return routing, index


def _write_shards(
    output: Path,
    index: pd.DataFrame,
    *,
    workers: int,
    shard_size: int,
    requested_conformers: int,
    retained_conformers: int,
    max_iterations: int,
) -> None:
    feature_root = output / "features"
    feature_root.mkdir(parents=True, exist_ok=True)
    total = len(index)
    started = time.monotonic()
    context = mp.get_context("spawn")
    schema = _output_schema()
    with context.Pool(processes=workers, initializer=_initialize_worker) as pool:
        for shard_index, start in enumerate(range(0, total, shard_size)):
            target = feature_root / f"part-{shard_index:05d}.parquet"
            expected = min(shard_size, total - start)
            if target.exists():
                if pq.read_metadata(target).num_rows != expected:
                    raise FeatureBuildError(f"incomplete existing shard: {target}")
                continue
            block = index.iloc[start : start + shard_size]
            items = [
                (
                    row.feature_order,
                    row.structure_id,
                    row.standardized_smiles,
                    row.f2d_feature_id,
                    requested_conformers,
                    retained_conformers,
                    max_iterations,
                )
                for row in block.itertuples(index=False)
            ]
            rows = list(pool.imap(_compute_one, items, chunksize=4))
            frame = pd.DataFrame(rows).sort_values("feature_order", kind="stable")
            temporary = target.with_suffix(".parquet.tmp")
            table = pa.Table.from_pandas(
                frame,
                schema=schema,
                preserve_index=False,
                safe=False,
            )
            pq.write_table(table, temporary, compression="zstd")
            temporary.replace(target)
            completed = start + expected
            elapsed = max(time.monotonic() - started, 1e-9)
            print(
                f"structures={completed:,}/{total:,} rate={completed / elapsed:.2f}/s "
                f"shards={shard_index + 1}",
                flush=True,
            )


def _bindings(paths: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        record: dict[str, Any] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        if path.suffix == ".parquet":
            record["rows"] = pq.read_metadata(path).num_rows
            record["arrow_schema_sha256"] = _schema_sha(path)
        records.append(record)
    return records


def _validate(output: Path, *, expected_rows: int | None = None) -> dict[str, Any]:
    manifest_path = output / "manifest.json"
    if not manifest_path.is_file():
        raise FeatureBuildError("manifest is missing")
    manifest = json.loads(manifest_path.read_text())
    shards = sorted((output / "features").glob("part-*.parquet"))
    if [path.name for path in shards] != manifest["feature_shards"]:
        raise FeatureBuildError("feature shard membership mismatch")
    rows = sum(pq.read_metadata(path).num_rows for path in shards)
    if rows != int(manifest["counts"]["structure_count"]):
        raise FeatureBuildError("feature row count mismatch")
    if expected_rows is not None and rows != expected_rows:
        raise FeatureBuildError("unexpected structure count")
    status: dict[str, int] = {}
    orders: list[np.ndarray] = []
    for path in shards:
        table = pq.read_table(path, columns=["feature_order", "feature_status"])
        orders.append(table["feature_order"].to_numpy())
        values = table["feature_status"].to_pylist()
        for value in values:
            status[str(value)] = status.get(str(value), 0) + 1
    combined = np.concatenate(orders) if orders else np.asarray([], dtype=np.int64)
    if not np.array_equal(combined, np.arange(rows, dtype=np.int64)):
        raise FeatureBuildError("feature order is incomplete or duplicated")
    for binding in manifest["inputs"] + manifest["artifacts"]:
        path = Path(binding["path"])
        if not path.is_absolute():
            path = Path(manifest["repo_root"]) / path
        if not path.is_file() or path.stat().st_size != int(binding["bytes"]):
            raise FeatureBuildError(f"bound path changed: {path}")
        if _sha256(path) != binding["sha256"]:
            raise FeatureBuildError(f"bound hash changed: {path}")
        if path.suffix == ".parquet":
            if pq.read_metadata(path).num_rows != int(binding["rows"]):
                raise FeatureBuildError(f"bound rows changed: {path}")
            if _schema_sha(path) != binding["arrow_schema_sha256"]:
                raise FeatureBuildError(f"bound schema changed: {path}")
    return {"status": "passed", "rows": rows, "feature_status_counts": status}


def _build(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.repo_root).resolve()
    output = Path(args.output_root).resolve()
    if output.exists() and args.overwrite:
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    routing, index = _load_index(root, args.limit)
    routing.drop(columns=["standardized_smiles"]).to_parquet(output / "routing.parquet", index=False)
    index.to_parquet(output / "feature_index.parquet", index=False)
    print(
        f"quantitative_herg_structures={len(index):,} workers={args.workers} "
        f"requested_conformers={args.conformers} retained={args.retain}",
        flush=True,
    )
    _write_shards(
        output,
        index,
        workers=args.workers,
        shard_size=args.shard_size,
        requested_conformers=args.conformers,
        retained_conformers=args.retain,
        max_iterations=args.max_iterations,
    )
    shards = sorted((output / "features").glob("part-*.parquet"))
    artifacts = [output / "routing.parquet", output / "feature_index.parquet", *shards]
    inputs = [
        root / INPUT_SURFACE,
        root / INPUT_MANIFEST,
        root / STRUCTURE_MASTER,
        root / F2D_MAPPING,
        root / F2D_MANIFEST,
        Path(__file__).resolve(),
    ]
    feature_columns = pq.read_schema(shards[0]).names if shards else []
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "repo_root": str(root),
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "scientific_scope": {
            "target": "human wild-type-or-unspecified hERG quantitative exact/censored pIC50 surface",
            "routing_only": True,
            "labels_or_outcome_values_opened": False,
            "feature_semantics": "bounded parent-state conformer and physics-informed proxy features",
            "not_claimed": [
                "pKa prediction",
                "equilibrium microstate population",
                "docking or binding affinity",
                "membrane PMF or permeability",
                "molecular dynamics or kinetics",
                "predictive superiority",
            ],
        },
        "parameters": {
            "workers": args.workers,
            "requested_conformers": args.conformers,
            "retained_conformers": args.retain,
            "maximum_force_field_iterations": args.max_iterations,
            "temperature_K": TEMPERATURE_K,
            "shard_size": args.shard_size,
            "limit": args.limit,
        },
        "software": {
            "python": sys.version,
            "rdkit": rdBase.rdkitVersion,
            "numpy": np.__version__,
            "pyarrow": pa.__version__,
        },
        "counts": {
            "structure_count": len(index),
            "feature_column_count_including_metadata": len(feature_columns),
            "dominant_vector_feature_count": sum(VECTOR_BLOCKS.values()),
        },
        "feature_shards": [path.name for path in shards],
        "inputs": _bindings(inputs),
        "artifacts": _bindings(artifacts),
    }
    manifest["manifest_payload_sha256"] = hashlib.sha256(_canonical_json(manifest).encode()).hexdigest()
    (output / "manifest.json").write_text(_canonical_json(manifest), encoding="utf-8")
    validation = _validate(output, expected_rows=len(index))
    (output / "validation.json").write_text(_canonical_json(validation), encoding="utf-8")
    return validation


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--conformers", type=int, default=8)
    parser.add_argument("--retain", type=int, default=4)
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument("--shard-size", type=int, default=1_000)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.conformers < 1 or args.retain < 1 or args.retain > args.conformers:
        parser.error("require 1 <= retain <= conformers")
    if args.validate_only:
        result = _validate(Path(args.output_root).resolve())
    else:
        result = _build(args)
    print(_canonical_json(result), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
