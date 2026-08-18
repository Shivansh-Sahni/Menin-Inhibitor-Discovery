#!/usr/bin/env python3
"""Assemble and audit the local quantitative-hERG 2D+3D feature matrix.

This label-blind step joins the completed candidate feature caches by immutable
structure identity.  It creates a directly loadable matrix and unsupervised QC
artifacts without opening potency values, classes, or held-out outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from rdkit import rdBase

SCHEMA_VERSION = "platform-local-herg-feature-matrix/1.0"
F2D_ROOT = Path("research/local_runs/local_multicpu_2d_features_v1")
F3D_ROOT = Path("research/local_runs/herg_quantitative_mechanistic_3d_v1")
F3D_NONNUMERIC_OR_ROUTING_COLUMNS = {
    "f3d__feature_order",
    "f3d__feature_status",
    "f3d__error_class",
    "f3d__force_field",
}


class MatrixBuildError(RuntimeError):
    """Raised when the feature matrix cannot be assembled safely."""


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


def _binding(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }
    if path.suffix == ".parquet":
        result["rows"] = pq.read_metadata(path).num_rows
        result["arrow_schema_sha256"] = _schema_sha(path)
    return result


def _validate_cache(root: Path, name: str) -> dict[str, Any]:
    validation_path = root / "validation.json"
    if validation_path.is_file():
        validation = json.loads(validation_path.read_text())
        if validation.get("status") != "passed":
            raise MatrixBuildError(f"{name} cache did not pass validation")
        return validation
    manifest_path = root / "feature_cache_manifest.json"
    if not manifest_path.is_file():
        raise MatrixBuildError(f"{name} validation record is missing")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "complete_candidate_local_feature_cache":
        raise MatrixBuildError(f"{name} cache is not complete")
    return {"status": "passed_from_complete_manifest"}


def _load_3d(root: Path) -> pd.DataFrame:
    feature_root = root / "features"
    shards = sorted(feature_root.glob("part-*.parquet"))
    if not shards:
        raise MatrixBuildError("3D feature shards are missing")
    schema = pq.read_schema(shards[0])
    for shard in shards[1:]:
        if pq.read_schema(shard) != schema:
            raise MatrixBuildError(f"3D shard schema mismatch: {shard}")
    frame = ds.dataset(feature_root, format="parquet").to_table().to_pandas()
    if frame["structure_id"].duplicated().any():
        raise MatrixBuildError("3D structure IDs are duplicated")
    return frame.sort_values("feature_order", kind="stable").reset_index(drop=True)


def _load_selected_2d(root: Path, feature_ids: set[str]) -> pd.DataFrame:
    feature_root = root / "features"
    frames: list[pd.DataFrame] = []
    value_set = pa.array(sorted(feature_ids), type=pa.large_string())
    for shard in sorted(feature_root.glob("part-*.parquet")):
        table = pq.read_table(shard)
        selected = table.filter(pc.is_in(table["feature_id"], value_set=value_set))
        if selected.num_rows:
            frames.append(selected.to_pandas())
    if not frames:
        raise MatrixBuildError("no requested 2D features were recovered")
    result = pd.concat(frames, ignore_index=True)
    if result["feature_id"].duplicated().any():
        raise MatrixBuildError("selected 2D feature IDs are duplicated")
    observed = set(result["feature_id"].astype(str))
    if observed != feature_ids:
        raise MatrixBuildError(f"2D feature closure failure: missing {len(feature_ids - observed)}")
    return result


def _feature_qc(matrix: pd.DataFrame) -> pd.DataFrame:
    feature_columns = [
        column
        for column in matrix.columns
        if column.startswith("rdkit2d__")
        or column.startswith("f3d__")
        and column not in F3D_NONNUMERIC_OR_ROUTING_COLUMNS
    ]
    splits = matrix["model_split"].astype(str).to_numpy()
    train_mask = splits == "train"
    rows: list[dict[str, Any]] = []
    for column in feature_columns:
        values = pd.to_numeric(matrix[column], errors="coerce").to_numpy(dtype=np.float64)
        finite = np.isfinite(values)
        train = values[train_mask & finite]
        train_mean = float(np.mean(train)) if train.size else math.nan
        train_sd = float(np.std(train)) if train.size else math.nan
        record: dict[str, Any] = {
            "feature_name": column,
            "feature_family": "rdkit2d" if column.startswith("rdkit2d__") else "conformer3d",
            "total_missing_count": int((~finite).sum()),
            "train_valid_count": int(train.size),
            "train_mean": train_mean if math.isfinite(train_mean) else None,
            "train_sd": train_sd if math.isfinite(train_sd) else None,
            "constant_or_all_missing_in_train": bool(train.size == 0 or train_sd <= 1e-12),
        }
        for split in ("validation", "test"):
            selected = values[(splits == split) & finite]
            selected_mean = float(np.mean(selected)) if selected.size else math.nan
            if math.isfinite(train_sd) and train_sd > 1e-12 and math.isfinite(selected_mean):
                smd = (selected_mean - train_mean) / train_sd
            else:
                smd = math.nan
            record[f"{split}_valid_count"] = int(selected.size)
            record[f"{split}_standardized_mean_shift"] = float(smd) if math.isfinite(smd) else None
        rows.append(record)
    return pd.DataFrame(rows).sort_values("feature_name", kind="stable")


def _redundancy(matrix: pd.DataFrame, qc: pd.DataFrame) -> pd.DataFrame:
    usable = qc.loc[~qc["constant_or_all_missing_in_train"], "feature_name"].tolist()
    train = matrix[matrix["model_split"].eq("train")][usable].copy()
    if len(train) > 5_000:
        train = train.sample(n=5_000, random_state=20260811)
    train = train.apply(pd.to_numeric, errors="coerce")
    train = train.fillna(train.median(numeric_only=True))
    correlation = train.corr(method="pearson").to_numpy(dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for left in range(len(usable)):
        for right in range(left + 1, len(usable)):
            value = correlation[left, right]
            if math.isfinite(value) and abs(value) >= 0.995:
                rows.append(
                    {
                        "feature_a": usable[left],
                        "feature_b": usable[right],
                        "pearson_correlation_train_sample": float(value),
                        "absolute_correlation": float(abs(value)),
                        "sample_size_maximum": len(train),
                        "action": "review_one_for_removal_using_training_data_only",
                    }
                )
    columns = [
        "feature_a",
        "feature_b",
        "pearson_correlation_train_sample",
        "absolute_correlation",
        "sample_size_maximum",
        "action",
    ]
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["absolute_correlation", "feature_a", "feature_b"],
        ascending=[False, True, True],
        kind="stable",
    )


def _validate_output(output: Path) -> dict[str, Any]:
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    expected = {
        "combined_feature_matrix.parquet",
        "feature_qc.parquet",
        "feature_redundancy.parquet",
        "manifest.json",
        "validation.json",
    }
    observed = {path.name for path in output.iterdir() if path.is_file()}
    if observed != expected:
        raise MatrixBuildError(f"output membership mismatch: {observed ^ expected}")
    for binding in manifest["inputs"] + manifest["artifacts"]:
        path = Path(binding["path"])
        if not path.is_file() or path.stat().st_size != int(binding["bytes"]):
            raise MatrixBuildError(f"bound file changed: {path}")
        if _sha256(path) != binding["sha256"]:
            raise MatrixBuildError(f"bound hash changed: {path}")
        if path.suffix == ".parquet":
            if pq.read_metadata(path).num_rows != int(binding["rows"]):
                raise MatrixBuildError(f"bound row count changed: {path}")
            if _schema_sha(path) != binding["arrow_schema_sha256"]:
                raise MatrixBuildError(f"bound schema changed: {path}")
    matrix = pq.read_table(output / "combined_feature_matrix.parquet")
    if matrix.num_rows != int(manifest["counts"]["structures"]):
        raise MatrixBuildError("matrix row count mismatch")
    if len(set(matrix["structure_id"].to_pylist())) != matrix.num_rows:
        raise MatrixBuildError("matrix structure IDs are not unique")
    return {
        "status": "passed",
        "structures": matrix.num_rows,
        "columns": matrix.num_columns,
        "label_or_outcome_columns": 0,
    }


def _build(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.repo_root).resolve()
    output = Path(args.output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise MatrixBuildError("output directory must be new and empty")
    f2d_root = root / F2D_ROOT
    f3d_root = root / F3D_ROOT
    _validate_cache(f2d_root, "2D")
    _validate_cache(f3d_root, "3D")

    three_d = _load_3d(f3d_root)
    routing = pq.read_table(f3d_root / "routing.parquet").to_pandas()
    if routing["structure_id"].duplicated().any():
        raise MatrixBuildError("routing structure IDs are duplicated")
    expected_ids = set(three_d["f2d_feature_id"].astype(str))
    two_d = _load_selected_2d(f2d_root, expected_ids)
    two_d = two_d.rename(columns={"feature_id": "f2d_feature_id"})

    two_d_keep = [
        "f2d_feature_id",
        "morgan_r2_2048",
        "maccs_167",
        "descriptor_missing_count",
        "feature_error",
        *[column for column in two_d.columns if column.startswith("rdkit2d__")],
    ]
    three_d = three_d.rename(
        columns={
            column: f"f3d__{column}"
            for column in three_d.columns
            if column not in {"structure_id", "f2d_feature_id"}
        }
    )
    matrix = routing.merge(two_d[two_d_keep], on="f2d_feature_id", how="left", validate="one_to_one").merge(
        three_d, on=["structure_id", "f2d_feature_id"], how="left", validate="one_to_one"
    )
    if len(matrix) != len(routing):
        raise MatrixBuildError("join changed matrix grain")
    if matrix["morgan_r2_2048"].isna().any() or matrix["f3d__feature_status"].isna().any():
        raise MatrixBuildError("feature join is incomplete")
    matrix = matrix.sort_values("structure_id", kind="stable").reset_index(drop=True)
    matrix.to_parquet(output / "combined_feature_matrix.parquet", index=False, compression="zstd")

    qc = _feature_qc(matrix)
    qc.to_parquet(output / "feature_qc.parquet", index=False, compression="zstd")
    redundancy = _redundancy(matrix, qc)
    redundancy.to_parquet(output / "feature_redundancy.parquet", index=False, compression="zstd")

    input_paths = [
        f2d_root / "feature_cache_manifest.json",
        f2d_root / "feature_index.parquet",
        f2d_root / "source_to_feature_mapping.parquet",
        *sorted((f2d_root / "features").glob("part-*.parquet")),
        f3d_root / "manifest.json",
        f3d_root / "validation.json",
        f3d_root / "routing.parquet",
        *sorted((f3d_root / "features").glob("part-*.parquet")),
        Path(__file__).resolve(),
    ]
    artifact_paths = [
        output / "combined_feature_matrix.parquet",
        output / "feature_qc.parquet",
        output / "feature_redundancy.parquet",
    ]
    shift_columns = ["validation_standardized_mean_shift", "test_standardized_mean_shift"]
    max_shift = max(
        float(qc[column].abs().max(skipna=True)) if qc[column].notna().any() else 0.0
        for column in shift_columns
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "scientific_scope": {
            "target_family": "quantitative hERG",
            "label_or_outcome_values_opened": False,
            "test_labels_opened": False,
            "split_and_scaffold_routing_preserved": True,
            "feature_selection_performed": False,
            "training_performed": False,
            "predictive_superiority_claimed": False,
        },
        "counts": {
            "structures": len(matrix),
            "matrix_columns": len(matrix.columns),
            "continuous_qc_features": len(qc),
            "constant_or_all_missing_train_features": int(qc["constant_or_all_missing_in_train"].sum()),
            "highly_correlated_training_sample_pairs": len(redundancy),
            "successful_3d_structures": int(matrix["f3d__feature_status"].eq("ok").sum()),
            "explicit_3d_failures": int((~matrix["f3d__feature_status"].eq("ok")).sum()),
        },
        "maximum_absolute_unsupervised_split_mean_shift": max_shift,
        "software": {
            "python": sys.version,
            "rdkit": rdBase.rdkitVersion,
            "numpy": np.__version__,
            "pyarrow": pa.__version__,
        },
        "inputs": [_binding(path) for path in input_paths],
        "artifacts": [_binding(path) for path in artifact_paths],
    }
    manifest["manifest_payload_sha256"] = hashlib.sha256(_canonical_json(manifest).encode()).hexdigest()
    (output / "manifest.json").write_text(_canonical_json(manifest), encoding="utf-8")
    placeholder = {
        "status": "pending_final_validation",
        "structures": len(matrix),
        "columns": len(matrix.columns),
        "label_or_outcome_columns": 0,
    }
    (output / "validation.json").write_text(_canonical_json(placeholder), encoding="utf-8")
    validation = _validate_output(output)
    (output / "validation.json").write_text(_canonical_json(validation), encoding="utf-8")
    return {"validation": validation, "counts": manifest["counts"]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_root).resolve()
    result = _validate_output(output) if args.validate_only else _build(args)
    print(_canonical_json(result), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
