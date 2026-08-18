#!/usr/bin/env python3
"""Analyze paired validation errors from the local hERG feature ablation.

The analysis uses only already-generated validation predictions.  It performs
paired scaffold-cluster bootstrap inference and descriptive, explicitly post
hoc molecular subgroup analyses.  It never reads locked test outcomes and does
not train or select a model.
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
import pyarrow.parquet as pq

SCHEMA_VERSION = "platform-local-herg-ablation-analysis/1.0"
BASELINE_ID = "xgb_2d_descriptors"
THREED_ID = "xgb_2d_plus_conformer3d"


class AnalysisError(RuntimeError):
    """Raised when paired analysis contracts are violated."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _schema_sha(path: Path) -> str:
    return hashlib.sha256(pq.read_schema(path).serialize().to_pybytes()).hexdigest()


def _json(value: Any) -> str:
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


def _paired_predictions(ablation_root: Path, matrix_root: Path) -> pd.DataFrame:
    predictions = pd.read_parquet(ablation_root / "validation_predictions.parquet")
    if set(predictions["model_split"].astype(str)) != {"validation"}:
        raise AnalysisError("non-validation predictions entered paired analysis")
    expected_models = {BASELINE_ID, THREED_ID}
    if set(predictions["model_id"].astype(str)) != expected_models:
        raise AnalysisError("unexpected model set")
    observed_counts = predictions.groupby("structure_id")["observed_pic50"].nunique()
    if int(observed_counts.max()) != 1:
        raise AnalysisError("paired observations disagree")
    wide = predictions.pivot(index="structure_id", columns="model_id", values="predicted_pic50").reset_index()
    observed = predictions.drop_duplicates("structure_id")[["structure_id", "observed_pic50"]]
    matrix_columns = [
        "structure_id",
        "model_split",
        "scaffold_group_id",
        "f3d__feature_status",
        "f3d__formal_charge",
        "f3d__rotatable_bond_count",
        "f3d__energy_range_kcal_mol",
        "f3d__effective_conformer_count",
        "f3d__ensemble_polar_radial_exposure__mean",
        "f3d__ensemble_gasteiger_dipole_proxy_eA__mean",
    ]
    matrix = pd.read_parquet(matrix_root / "combined_feature_matrix.parquet", columns=matrix_columns)
    matrix = matrix[matrix["model_split"].eq("validation")].drop(columns="model_split")
    paired = observed.merge(wide, on="structure_id", validate="one_to_one").merge(
        matrix, on="structure_id", validate="one_to_one"
    )
    if len(paired) != len(observed):
        raise AnalysisError("validation feature join is incomplete")
    paired["absolute_error_2d"] = np.abs(paired["observed_pic50"] - paired[BASELINE_ID])
    paired["absolute_error_2d_plus_3d"] = np.abs(paired["observed_pic50"] - paired[THREED_ID])
    paired["squared_error_2d"] = np.square(paired["observed_pic50"] - paired[BASELINE_ID])
    paired["squared_error_2d_plus_3d"] = np.square(paired["observed_pic50"] - paired[THREED_ID])
    paired["delta_absolute_error_3d_minus_2d"] = (
        paired["absolute_error_2d_plus_3d"] - paired["absolute_error_2d"]
    )
    paired["three_d_better_absolute_error"] = paired["delta_absolute_error_3d_minus_2d"] < 0
    return paired


def _bootstrap(paired: pd.DataFrame, replicates: int) -> dict[str, Any]:
    grouped = (
        paired.groupby("scaffold_group_id", as_index=False)
        .agg(
            count=("structure_id", "size"),
            absolute_2d=("absolute_error_2d", "sum"),
            absolute_3d=("absolute_error_2d_plus_3d", "sum"),
            squared_2d=("squared_error_2d", "sum"),
            squared_3d=("squared_error_2d_plus_3d", "sum"),
        )
        .sort_values("scaffold_group_id", kind="stable")
    )
    arrays = {
        column: grouped[column].to_numpy(dtype=np.float64)
        for column in ("count", "absolute_2d", "absolute_3d", "squared_2d", "squared_3d")
    }
    rng = np.random.default_rng(20260811)
    mae_delta = np.empty(replicates, dtype=np.float64)
    rmse_delta = np.empty(replicates, dtype=np.float64)
    group_count = len(grouped)
    for index in range(replicates):
        sampled = rng.integers(0, group_count, size=group_count)
        count = float(arrays["count"][sampled].sum())
        mae_2d = float(arrays["absolute_2d"][sampled].sum() / count)
        mae_3d = float(arrays["absolute_3d"][sampled].sum() / count)
        rmse_2d = math.sqrt(float(arrays["squared_2d"][sampled].sum() / count))
        rmse_3d = math.sqrt(float(arrays["squared_3d"][sampled].sum() / count))
        mae_delta[index] = mae_3d - mae_2d
        rmse_delta[index] = rmse_3d - rmse_2d

    def summarize(values: np.ndarray) -> dict[str, float]:
        return {
            "point_estimate": float(np.mean(values)),
            "bootstrap_median": float(np.median(values)),
            "ci95_lower": float(np.quantile(values, 0.025)),
            "ci95_upper": float(np.quantile(values, 0.975)),
            "probability_3d_better": float(np.mean(values < 0)),
        }

    point_mae_delta = float(paired["absolute_error_2d_plus_3d"].mean() - paired["absolute_error_2d"].mean())
    point_rmse_delta = float(
        math.sqrt(paired["squared_error_2d_plus_3d"].mean()) - math.sqrt(paired["squared_error_2d"].mean())
    )
    mae_summary = summarize(mae_delta)
    rmse_summary = summarize(rmse_delta)
    mae_summary["point_estimate"] = point_mae_delta
    rmse_summary["point_estimate"] = point_rmse_delta
    return {
        "replicates": replicates,
        "resampling_unit": "scaffold_group_id",
        "scaffold_groups": group_count,
        "structures": len(paired),
        "delta_definition": "2d_plus_3d_minus_2d; negative favors 3d",
        "mae_delta": mae_summary,
        "rmse_delta": rmse_summary,
    }


def _subgroups(paired: pd.DataFrame) -> pd.DataFrame:
    frame = paired.copy()
    charge = pd.to_numeric(frame["f3d__formal_charge"], errors="coerce")
    frame["charge_group"] = np.select(
        [charge < 0, charge == 0, charge > 0],
        ["negative", "neutral", "positive"],
        default="missing",
    )
    rotatable = pd.to_numeric(frame["f3d__rotatable_bond_count"], errors="coerce")
    frame["flexibility_group"] = (
        pd.cut(
            rotatable,
            bins=[-np.inf, 2, 5, np.inf],
            labels=["0_to_2", "3_to_5", "6_plus"],
        )
        .astype("string")
        .fillna("missing")
    )
    energy_range = pd.to_numeric(frame["f3d__energy_range_kcal_mol"], errors="coerce")
    frame["energy_range_group"] = (
        pd.cut(
            energy_range,
            bins=[-np.inf, 2, 5, 10, np.inf],
            labels=["up_to_2", "2_to_5", "5_to_10", "above_10"],
        )
        .astype("string")
        .fillna("missing")
    )
    definitions = {
        "feature_status": "f3d__feature_status",
        "formal_charge": "charge_group",
        "rotatable_bonds": "flexibility_group",
        "conformer_energy_range": "energy_range_group",
    }
    rows: list[dict[str, Any]] = []
    for axis, column in definitions.items():
        for value, selected in frame.groupby(column, dropna=False):
            rows.append(
                {
                    "subgroup_axis": axis,
                    "subgroup_value": str(value),
                    "structures": len(selected),
                    "mae_2d": float(selected["absolute_error_2d"].mean()),
                    "mae_2d_plus_3d": float(selected["absolute_error_2d_plus_3d"].mean()),
                    "delta_mae_3d_minus_2d": float(
                        selected["absolute_error_2d_plus_3d"].mean() - selected["absolute_error_2d"].mean()
                    ),
                    "fraction_structures_3d_better": float(selected["three_d_better_absolute_error"].mean()),
                    "analysis_role": "post_hoc_descriptive_hypothesis_generation_only",
                }
            )
    return pd.DataFrame(rows).sort_values(["subgroup_axis", "subgroup_value"], kind="stable")


def _validate(output: Path) -> dict[str, Any]:
    manifest = json.loads((output / "manifest.json").read_text())
    for binding in manifest["inputs"] + manifest["artifacts"]:
        path = Path(binding["path"])
        if not path.is_file() or path.stat().st_size != int(binding["bytes"]):
            raise AnalysisError(f"bound file changed: {path}")
        if _sha256(path) != binding["sha256"]:
            raise AnalysisError(f"bound hash changed: {path}")
        if path.suffix == ".parquet":
            if pq.read_metadata(path).num_rows != int(binding["rows"]):
                raise AnalysisError(f"bound rows changed: {path}")
            if _schema_sha(path) != binding["arrow_schema_sha256"]:
                raise AnalysisError(f"bound schema changed: {path}")
    paired = pd.read_parquet(output / "paired_validation_errors.parquet")
    if len(paired) != int(manifest["counts"]["validation_structures"]):
        raise AnalysisError("paired row count mismatch")
    return {
        "status": "passed",
        "validation_structures": len(paired),
        "test_labels_opened": False,
        "training_or_model_selection_performed": False,
    }


def _build(args: argparse.Namespace) -> dict[str, Any]:
    ablation_root = Path(args.ablation_root).resolve()
    matrix_root = Path(args.matrix_root).resolve()
    output = Path(args.output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise AnalysisError("output directory must be new and empty")
    validation = json.loads((ablation_root / "validation.json").read_text())
    if validation.get("status") != "passed" or int(validation.get("test_prediction_rows", -1)) != 0:
        raise AnalysisError("upstream ablation is not an accepted validation-only release")
    paired = _paired_predictions(ablation_root, matrix_root)
    bootstrap = _bootstrap(paired, args.bootstrap_replicates)
    subgroups = _subgroups(paired)
    paired.to_parquet(output / "paired_validation_errors.parquet", index=False)
    subgroups.to_parquet(output / "subgroup_error_summary.parquet", index=False)
    (output / "scaffold_bootstrap.json").write_text(_json(bootstrap), encoding="utf-8")
    input_paths = [
        ablation_root / "manifest.json",
        ablation_root / "validation.json",
        ablation_root / "validation_predictions.parquet",
        matrix_root / "manifest.json",
        matrix_root / "validation.json",
        matrix_root / "combined_feature_matrix.parquet",
        Path(__file__).resolve(),
    ]
    artifact_paths = [
        output / "paired_validation_errors.parquet",
        output / "subgroup_error_summary.parquet",
        output / "scaffold_bootstrap.json",
    ]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "scientific_scope": {
            "partition": "validation_only",
            "paired_structure_analysis": True,
            "scaffold_cluster_bootstrap": True,
            "subgroups": "post_hoc_descriptive_hypothesis_generation_only",
            "test_labels_opened": False,
            "training_or_model_selection_performed": False,
            "superiority_established": False,
        },
        "parameters": {"bootstrap_replicates": args.bootstrap_replicates, "seed": 20260811},
        "counts": {
            "validation_structures": len(paired),
            "scaffold_groups": paired["scaffold_group_id"].nunique(),
            "subgroup_rows": len(subgroups),
        },
        "inputs": [_binding(path) for path in input_paths],
        "artifacts": [_binding(path) for path in artifact_paths],
        "software": {
            "python": sys.version,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pyarrow": pa.__version__,
        },
    }
    manifest["manifest_payload_sha256"] = hashlib.sha256(_json(manifest).encode()).hexdigest()
    (output / "manifest.json").write_text(_json(manifest), encoding="utf-8")
    result = _validate(output)
    (output / "validation.json").write_text(_json(result), encoding="utf-8")
    return {"validation": result, "bootstrap": bootstrap}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation-root", required=True)
    parser.add_argument("--matrix-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_root).resolve()
    result = _validate(output) if args.validate_only else _build(args)
    print(_json(result), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
