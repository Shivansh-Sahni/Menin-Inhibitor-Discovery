#!/usr/bin/env python3
"""Build a resumable, label-blind, multicore candidate 2D feature cache.

This local workload resolves the three CPU-available parent-molecule families
in future_feature_contract.json: Morgan radius-2/2048, MACCS-167, and the full
RDKit 2D descriptor registry. It reads structure registries only. It never
opens activity, PK, hERG, affinity, clinical, split-outcome, or test labels.

The result is deliberately candidate-only: it is useful for schema, coverage,
runtime, missingness, and future model-input work, but is not admitted to the
production feature store until the Python/container production gate is frozen.
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
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import Descriptors, MACCSkeys, rdFingerprintGenerator

SCHEMA_VERSION = "platform-local-multicpu-2d-feature-cache/1.0"
MORGAN_BITS = 2048
MACCS_BITS = 167
DEFAULT_SHARD_SIZE = 25_000

AFFINITY_ROOT = Path(
    "research/data/platform/processed/affinity_training/v1_0_chembl37_bindingdb202608/ligands"
)
PK_REGISTRY = Path(
    "research/data/platform/processed/pk_adme/v1_0_trainable_surfaces/molecule_registry.parquet"
)
HERG_REGISTRY = Path("research/data/platform/processed/herg_hierarchy/v1_3_master/structure_master.parquet")
FEATURE_CONTRACT = Path("research/reports/platform/herg_paper/pre_hpc_contracts/future_feature_contract.json")

_MORGAN: Any = None
_DESCRIPTORS: list[tuple[str, Any]] = []


class FeatureCacheError(RuntimeError):
    """Raised when feature-cache construction or validation fails."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _feature_id(smiles: str) -> str:
    return "F2D-" + hashlib.sha256(smiles.encode("utf-8")).hexdigest()


def _descriptor_registry() -> list[tuple[str, Any]]:
    names: set[str] = set()
    registry: list[tuple[str, Any]] = []
    for name, function in Descriptors._descList:  # noqa: SLF001 - frozen RDKit public registry
        if name in names:
            raise FeatureCacheError(f"duplicate RDKit descriptor name: {name}")
        names.add(name)
        registry.append((str(name), function))
    if not registry:
        raise FeatureCacheError("RDKit descriptor registry is empty")
    return registry


def _initialize_worker() -> None:
    global _MORGAN, _DESCRIPTORS
    _MORGAN = rdFingerprintGenerator.GetMorganGenerator(
        radius=2,
        fpSize=MORGAN_BITS,
        includeChirality=True,
    )
    _DESCRIPTORS = _descriptor_registry()


def _largest_ring_size(molecule: Chem.Mol) -> int:
    atom_rings = molecule.GetRingInfo().AtomRings()
    return max((len(ring) for ring in atom_rings), default=0)


def _compute_one(item: tuple[int, str, str]) -> tuple[Any, ...]:
    order, feature_id, smiles = item
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return (
            order,
            feature_id,
            None,
            None,
            len(_DESCRIPTORS),
            "rdkit_parse_failed",
            *([None] * len(_DESCRIPTORS)),
        )

    try:
        morgan = DataStructs.BitVectToBinaryText(_MORGAN.GetFingerprint(molecule))
        maccs = DataStructs.BitVectToBinaryText(MACCSkeys.GenMACCSKeys(molecule))
    except Exception as error:  # pragma: no cover - guarded by production registries
        return (
            order,
            feature_id,
            None,
            None,
            len(_DESCRIPTORS),
            f"fingerprint_failed:{type(error).__name__}",
            *([None] * len(_DESCRIPTORS)),
        )

    values: list[float | None] = []
    missing = 0
    for _, function in _DESCRIPTORS:
        descriptor_value: float | None
        try:
            descriptor_value = float(function(molecule))
            if not math.isfinite(descriptor_value):
                descriptor_value = None
        except Exception:
            descriptor_value = None
        if descriptor_value is None:
            missing += 1
        values.append(descriptor_value)
    return order, feature_id, morgan, maccs, missing, None, *values


def _input_paths(root: Path) -> list[Path]:
    paths = sorted((root / AFFINITY_ROOT).glob("*.parquet"))
    paths.extend([root / PK_REGISTRY, root / HERG_REGISTRY, root / FEATURE_CONTRACT])
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FeatureCacheError(f"missing inputs: {missing}")
    return paths


def _source_table(
    *,
    family: str,
    table: pa.Table,
    id_column: str,
    limit: int | None,
) -> pd.DataFrame:
    if limit is not None:
        table = table.slice(0, limit)
    frame = table.select([id_column, "standardized_smiles", "standard_inchi_key"]).to_pandas()
    frame = frame.rename(columns={id_column: "source_structure_id"})
    frame.insert(0, "source_family", family)
    if frame.isna().any().any():
        raise FeatureCacheError(f"{family} registry has null molecule identities")
    return frame


def _load_source_mapping(root: Path, limit: int | None) -> pd.DataFrame:
    affinity = ds.dataset(root / AFFINITY_ROOT, format="parquet").to_table(
        columns=["structure_id", "standardized_smiles", "standard_inchi_key"]
    )
    pk = pq.read_table(
        root / PK_REGISTRY,
        columns=["molecule_id", "standardized_smiles", "standard_inchi_key"],
    )
    herg = pq.read_table(
        root / HERG_REGISTRY,
        columns=["structure_id", "standardized_smiles", "standard_inchi_key"],
    )
    mapping = pd.concat(
        [
            _source_table(
                family="affinity",
                table=affinity,
                id_column="structure_id",
                limit=limit,
            ),
            _source_table(
                family="pk_adme",
                table=pk,
                id_column="molecule_id",
                limit=limit,
            ),
            _source_table(
                family="herg",
                table=herg,
                id_column="structure_id",
                limit=limit,
            ),
        ],
        ignore_index=True,
    )
    if mapping.duplicated(["source_family", "source_structure_id"]).any():
        raise FeatureCacheError("source structure identity is not unique within family")
    mapping["feature_id"] = mapping["standardized_smiles"].map(_feature_id)
    return mapping


def _feature_index(mapping: pd.DataFrame) -> pd.DataFrame:
    conflicts = mapping.groupby("feature_id", sort=False)["standardized_smiles"].nunique()
    if int(conflicts.max()) != 1:
        raise FeatureCacheError("feature-id collision detected")
    index = (
        mapping[["feature_id", "standardized_smiles", "standard_inchi_key"]]
        .drop_duplicates("feature_id")
        .sort_values("feature_id", kind="stable")
        .reset_index(drop=True)
    )
    index.insert(0, "feature_order", np.arange(len(index), dtype=np.int64))
    return index


def _write_metadata(root: Path, mapping: pd.DataFrame, index: pd.DataFrame) -> None:
    mapping_output = mapping[
        ["source_family", "source_structure_id", "feature_id", "standard_inchi_key"]
    ].sort_values(["source_family", "source_structure_id"], kind="stable")
    mapping_output.to_parquet(root / "source_to_feature_mapping.parquet", index=False)
    index.to_parquet(root / "feature_index.parquet", index=False)


def _output_schema(descriptor_names: Sequence[str]) -> pa.Schema:
    fields = [
        pa.field("feature_order", pa.int64(), nullable=False),
        pa.field("feature_id", pa.large_string(), nullable=False),
        pa.field("morgan_r2_2048", pa.binary(MORGAN_BITS // 8), nullable=True),
        pa.field("maccs_167", pa.binary((MACCS_BITS + 7) // 8), nullable=True),
        pa.field("descriptor_missing_count", pa.int16(), nullable=False),
        pa.field("feature_error", pa.large_string(), nullable=True),
    ]
    fields.extend(pa.field(f"rdkit2d__{name}", pa.float32(), nullable=True) for name in descriptor_names)
    return pa.schema(fields)


def _rows_to_table(rows: list[tuple[Any, ...]], schema: pa.Schema) -> pa.Table:
    columns = list(zip(*rows, strict=True))
    arrays = [pa.array(column, type=field.type) for column, field in zip(columns, schema, strict=True)]
    return pa.Table.from_arrays(arrays, schema=schema)


def _existing_feature_rows(features_root: Path, schema: pa.Schema) -> int:
    files = sorted(features_root.glob("part-*.parquet"))
    total = 0
    for expected_index, path in enumerate(files):
        if path.name != f"part-{expected_index:05d}.parquet":
            raise FeatureCacheError("feature shard sequence is not contiguous")
        parquet = pq.ParquetFile(path)
        if not parquet.schema_arrow.equals(schema):
            raise FeatureCacheError(f"feature shard schema mismatch: {path}")
        total += parquet.metadata.num_rows
    return total


def _items(index: pd.DataFrame, offset: int) -> Iterator[tuple[int, str, str]]:
    selected = index.iloc[offset:]
    for row in selected.itertuples(index=False):
        yield int(row.feature_order), str(row.feature_id), str(row.standardized_smiles)


def _write_feature_shards(
    *,
    output_root: Path,
    index: pd.DataFrame,
    workers: int,
    shard_size: int,
) -> dict[str, Any]:
    descriptor_names = [name for name, _ in _descriptor_registry()]
    schema = _output_schema(descriptor_names)
    features_root = output_root / "features"
    features_root.mkdir(parents=True, exist_ok=True)
    completed = _existing_feature_rows(features_root, schema)
    if completed > len(index):
        raise FeatureCacheError("feature shards contain more rows than the feature index")
    shard_index = len(list(features_root.glob("part-*.parquet")))
    started = time.monotonic()
    buffer: list[tuple[Any, ...]] = []

    context = mp.get_context("spawn")
    with context.Pool(processes=workers, initializer=_initialize_worker) as pool:
        results = pool.imap(
            _compute_one,
            _items(index, completed),
            chunksize=128,
        )
        for row in results:
            buffer.append(row)
            if len(buffer) >= shard_size:
                path = features_root / f"part-{shard_index:05d}.parquet"
                pq.write_table(
                    _rows_to_table(buffer, schema),
                    path,
                    compression="zstd",
                    compression_level=3,
                    use_dictionary=False,
                )
                completed += len(buffer)
                shard_index += 1
                buffer.clear()
                elapsed = max(time.monotonic() - started, 1e-9)
                print(
                    f"features={completed:,}/{len(index):,} "
                    f"new_rate={(completed / elapsed):,.1f}/s shards={shard_index}",
                    flush=True,
                )
        if buffer:
            path = features_root / f"part-{shard_index:05d}.parquet"
            pq.write_table(
                _rows_to_table(buffer, schema),
                path,
                compression="zstd",
                compression_level=3,
                use_dictionary=False,
            )
            completed += len(buffer)
            shard_index += 1
            buffer.clear()

    return {
        "descriptor_names": descriptor_names,
        "feature_rows": completed,
        "feature_shards": shard_index,
        "feature_schema_serialized_sha256": hashlib.sha256(schema.serialize().to_pybytes()).hexdigest(),
    }


def _artifact_binding(path: Path, root: Path) -> dict[str, Any]:
    binding: dict[str, Any] = {
        "path": str(path.relative_to(root)),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }
    if path.suffix == ".parquet":
        parquet = pq.ParquetFile(path)
        binding["rows"] = parquet.metadata.num_rows
        binding["arrow_schema_sha256"] = hashlib.sha256(
            parquet.schema_arrow.serialize().to_pybytes()
        ).hexdigest()
    return binding


def _seal_release(
    *,
    repo_root: Path,
    output_root: Path,
    index: pd.DataFrame,
    mapping: pd.DataFrame,
    feature_summary: dict[str, Any],
    workers: int,
    limit: int | None,
    started: float,
) -> dict[str, Any]:
    feature_files = sorted((output_root / "features").glob("part-*.parquet"))
    feature_rows = sum(pq.ParquetFile(path).metadata.num_rows for path in feature_files)
    if feature_rows != len(index):
        raise FeatureCacheError("feature shards do not exactly cover the feature index")

    descriptor_schema = {
        "schema_version": SCHEMA_VERSION,
        "rdkit_version": rdBase.rdkitVersion,
        "families": {
            "morgan_r2_2048": {
                "radius": 2,
                "bits": MORGAN_BITS,
                "include_chirality": True,
                "storage": "bit-packed fixed-size binary",
            },
            "maccs_167": {
                "bits": MACCS_BITS,
                "storage": "bit-packed fixed-size binary",
            },
            "rdkit_2d": {
                "dtype": "float32",
                "descriptor_count": len(feature_summary["descriptor_names"]),
                "descriptor_names": feature_summary["descriptor_names"],
                "missing_values": "null with descriptor_missing_count",
            },
        },
    }
    (output_root / "feature_schema.json").write_text(_canonical_json(descriptor_schema), encoding="utf-8")

    implementation = Path(__file__).resolve()
    input_bindings = [_artifact_binding(path, repo_root) for path in _input_paths(repo_root)]
    input_bindings.append(_artifact_binding(implementation, repo_root))
    artifacts = [
        output_root / "feature_index.parquet",
        output_root / "source_to_feature_mapping.parquet",
        output_root / "feature_schema.json",
        *feature_files,
    ]
    artifact_bindings = [_artifact_binding(path, output_root) for path in artifacts]
    source_counts = {
        str(key): int(value) for key, value in mapping["source_family"].value_counts().sort_index().items()
    }
    missing_total = 0
    error_total = 0
    for path in feature_files:
        table = pq.read_table(path, columns=["descriptor_missing_count", "feature_error"])
        missing_total += int(pa.compute.sum(table["descriptor_missing_count"]).as_py() or 0)
        error_total += int(pa.compute.count(table["feature_error"]).as_py() or 0)

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete_candidate_local_feature_cache",
        "repo_root": str(repo_root),
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "workers": workers,
        "limit_per_source_for_smoke_test": limit,
        "rdkit_version": rdBase.rdkitVersion,
        "python_version": sys.version.split()[0],
        "counts": {
            "source_identity_rows": len(mapping),
            "unique_feature_rows": len(index),
            "feature_shards": len(feature_files),
            "descriptor_count": len(feature_summary["descriptor_names"]),
            "descriptor_missing_cells": missing_total,
            "feature_error_rows": error_total,
            "source_identity_rows_by_family": source_counts,
        },
        "input_bindings": input_bindings,
        "artifact_bindings": artifact_bindings,
        "scientific_contract": {
            "outcome_or_label_columns_opened": False,
            "test_labels_opened": False,
            "clinical_or_qt_labels_generated": False,
            "models_trained": False,
            "hpc_executed": False,
            "production_admission": False,
            "candidate_only_reason": (
                "RDKit is pinned, but the future-feature contract still records "
                "the production Python/container lock as unresolved; this cache "
                "is for coverage, runtime, schema, missingness, and later admission."
            ),
        },
    }
    payload = dict(manifest)
    manifest["manifest_payload_sha256"] = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
    (output_root / "feature_cache_manifest.json").write_text(_canonical_json(manifest), encoding="utf-8")
    incomplete = output_root / "INCOMPLETE.json"
    if incomplete.exists():
        incomplete.unlink()
    return manifest


def validate_release(output_root: Path) -> dict[str, Any]:
    output_root = output_root.resolve()
    manifest_path = output_root / "feature_cache_manifest.json"
    if not manifest_path.is_file():
        raise FeatureCacheError("complete manifest is absent")
    if (output_root / "INCOMPLETE.json").exists():
        raise FeatureCacheError("release remains marked incomplete")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload = dict(manifest)
    declared = payload.pop("manifest_payload_sha256")
    actual = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
    if actual != declared:
        raise FeatureCacheError("manifest payload self-hash mismatch")
    repo_root = Path(manifest["repo_root"]).resolve()
    for binding in manifest["input_bindings"]:
        path = repo_root / binding["path"]
        if not path.is_file():
            raise FeatureCacheError(f"missing bound input: {path}")
        if path.stat().st_size != int(binding["bytes"]) or _sha256_file(path) != binding["sha256"]:
            raise FeatureCacheError(f"input binding mismatch: {path}")
        if path.suffix == ".parquet":
            parquet = pq.ParquetFile(path)
            schema_hash = hashlib.sha256(parquet.schema_arrow.serialize().to_pybytes()).hexdigest()
            if parquet.metadata.num_rows != int(binding["rows"]):
                raise FeatureCacheError(f"input row mismatch: {path}")
            if schema_hash != binding["arrow_schema_sha256"]:
                raise FeatureCacheError(f"input schema mismatch: {path}")
    for binding in manifest["artifact_bindings"]:
        path = output_root / binding["path"]
        if not path.is_file():
            raise FeatureCacheError(f"missing artifact: {path}")
        if path.stat().st_size != int(binding["bytes"]) or _sha256_file(path) != binding["sha256"]:
            raise FeatureCacheError(f"artifact binding mismatch: {path}")
        if path.suffix == ".parquet":
            parquet = pq.ParquetFile(path)
            schema_hash = hashlib.sha256(parquet.schema_arrow.serialize().to_pybytes()).hexdigest()
            if parquet.metadata.num_rows != int(binding["rows"]):
                raise FeatureCacheError(f"artifact row mismatch: {path}")
            if schema_hash != binding["arrow_schema_sha256"]:
                raise FeatureCacheError(f"artifact schema mismatch: {path}")
    declared_paths = {binding["path"] for binding in manifest["artifact_bindings"]} | {
        "feature_cache_manifest.json"
    }
    physical_paths = {str(path.relative_to(output_root)) for path in output_root.rglob("*") if path.is_file()}
    if physical_paths != declared_paths:
        raise FeatureCacheError("output directory membership is not closed")
    feature_rows = sum(
        int(binding.get("rows", 0))
        for binding in manifest["artifact_bindings"]
        if str(binding["path"]).startswith("features/")
    )
    if feature_rows != int(manifest["counts"]["unique_feature_rows"]):
        raise FeatureCacheError("feature shards do not cover the declared index")
    if manifest["scientific_contract"] != {
        "outcome_or_label_columns_opened": False,
        "test_labels_opened": False,
        "clinical_or_qt_labels_generated": False,
        "models_trained": False,
        "hpc_executed": False,
        "production_admission": False,
        "candidate_only_reason": manifest["scientific_contract"]["candidate_only_reason"],
    }:
        raise FeatureCacheError("scientific boundary was weakened")
    print(_canonical_json({"status": "passed", "counts": manifest["counts"]}), end="")
    return manifest


def build(
    *,
    repo_root: Path,
    output_root: Path,
    workers: int,
    shard_size: int,
    limit: int | None,
) -> dict[str, Any]:
    started = time.monotonic()
    repo_root = repo_root.resolve()
    output_root = output_root.resolve()
    if workers < 1:
        raise FeatureCacheError("workers must be positive")
    if shard_size < 1:
        raise FeatureCacheError("shard size must be positive")
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "feature_cache_manifest.json"
    if manifest_path.exists():
        return validate_release(output_root)
    (output_root / "INCOMPLETE.json").write_text(
        _canonical_json(
            {
                "status": "incomplete_resumable",
                "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
        ),
        encoding="utf-8",
    )

    mapping_path = output_root / "source_to_feature_mapping.parquet"
    index_path = output_root / "feature_index.parquet"
    if mapping_path.exists() and index_path.exists():
        mapping = pd.read_parquet(mapping_path)
        index = pd.read_parquet(index_path)
    else:
        mapping = _load_source_mapping(repo_root, limit)
        index = _feature_index(mapping)
        _write_metadata(output_root, mapping, index)
    print(
        f"source_identities={len(mapping):,} unique_features={len(index):,} workers={workers}",
        flush=True,
    )
    feature_summary = _write_feature_shards(
        output_root=output_root,
        index=index,
        workers=workers,
        shard_size=shard_size,
    )
    manifest = _seal_release(
        repo_root=repo_root,
        output_root=output_root,
        index=index,
        mapping=mapping,
        feature_summary=feature_summary,
        workers=workers,
        limit=limit,
        started=started,
    )
    validate_release(output_root)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, os.cpu_count() or 1),
        help="worker processes; defaults to every logical CPU",
    )
    parser.add_argument("--shard-size", type=int, default=DEFAULT_SHARD_SIZE)
    parser.add_argument(
        "--limit-per-source",
        type=int,
        help="smoke-test only: cap each source registry before union",
    )
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output_root = args.output_root.resolve()
    if args.overwrite and output_root.exists():
        shutil.rmtree(output_root)
    if args.validate_only:
        validate_release(output_root)
    else:
        build(
            repo_root=args.repo_root,
            output_root=output_root,
            workers=args.workers,
            shard_size=args.shard_size,
            limit=args.limit_per_source,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
