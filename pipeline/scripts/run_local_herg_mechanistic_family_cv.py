#!/usr/bin/env python3
"""Run training-only scaffold CV for hERG mechanistic feature families.

The experiment compares fixed XGBoost regressors using the same 2D baseline
plus preregistered conformer-feature families.  Only exact-pIC50 rows returned
from the training partition are used.  Validation and test outcomes are not
returned, scored, or used for feature-family selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import scipy
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor

SCHEMA_VERSION = "platform-local-herg-mechanistic-family-cv/1.0"
OBSERVATIONS = Path(
    "research/data/platform/processed/herg_hierarchy/v1_6_training_surfaces/"
    "herg_training_observations.parquet"
)
MAX_ABSOLUTE_NUMERIC_FEATURE = 1.0e30


class FamilyCVError(RuntimeError):
    """Raised when a train-only cross-validation contract is violated."""


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


def _targets(path: Path) -> pd.DataFrame:
    table = pq.read_table(
        path,
        columns=[
            "structure_id",
            "model_split",
            "potency_relation_pic50",
            "potency_pic50_point",
            "standardized_pic50_primary",
        ],
        filters=[
            ("standardized_pic50_primary", "=", True),
            ("potency_relation_pic50", "=", "="),
            ("model_split", "=", "train"),
        ],
    ).to_pandas()
    if set(table["model_split"].astype(str)) != {"train"}:
        raise FamilyCVError("nontraining outcomes entered the analysis frame")
    return (
        table.groupby("structure_id", as_index=False)
        .agg(
            target_pic50=("potency_pic50_point", "median"),
            exact_observation_count=("potency_pic50_point", "size"),
        )
        .sort_values("structure_id", kind="stable")
    )


def _is_scalar_shape(column: str) -> bool:
    terms = (
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
    return any(term in column for term in terms)


def _is_polarity_charge(column: str) -> bool:
    terms = (
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
    return any(term in column for term in terms)


def _is_ensemble_energy_flexibility(column: str) -> bool:
    terms = (
        "rotatable_bond",
        "embedded_conformer_count",
        "retained_conformer_count",
        "unconverged_retained_count",
        "energy_range_kcal_mol",
        "effective_conformer_count",
        "dominant_conformer_weight",
        "energy_polar_exposure_correlation",
    )
    return any(term in column for term in terms)


def _families(matrix: pd.DataFrame) -> dict[str, list[str]]:
    two_d = sorted(column for column in matrix if column.startswith("rdkit2d__"))
    numeric_3d = sorted(
        column
        for column in matrix
        if column.startswith("f3d__")
        and column not in {"f3d__feature_order", "f3d__energy_min_kcal_mol"}
        and pd.api.types.is_numeric_dtype(matrix[column])
    )
    groups = {
        "shape_scalars": [column for column in numeric_3d if _is_scalar_shape(column)],
        "polarity_charge_scalars": [column for column in numeric_3d if _is_polarity_charge(column)],
        "ensemble_energy_flexibility": [
            column for column in numeric_3d if _is_ensemble_energy_flexibility(column)
        ],
        "autocorr3d": [column for column in numeric_3d if "dominant_autocorr3d" in column],
        "whim3d": [column for column in numeric_3d if "dominant_whim" in column],
        "all_conformer3d": numeric_3d,
    }
    if not two_d or any(not values for values in groups.values()):
        raise FamilyCVError("one or more preregistered feature families are empty")
    return {"2d_baseline": two_d, **{name: [*two_d, *values] for name, values in groups.items()}}


def _sanitize(frame: pd.DataFrame, columns: list[str]) -> tuple[np.ndarray, int]:
    raw = frame[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64, copy=True)
    invalid = ~np.isfinite(raw) | (np.abs(raw) > MAX_ABSOLUTE_NUMERIC_FEATURE)
    count = int(invalid.sum())
    raw[invalid] = np.nan
    return np.ascontiguousarray(raw, dtype=np.float32), count


def _model(workers: int, estimators: int) -> XGBRegressor:
    return XGBRegressor(
        objective="reg:squarederror",
        n_estimators=estimators,
        max_depth=6,
        learning_rate=0.05,
        min_child_weight=3.0,
        subsample=0.80,
        colsample_bytree=0.80,
        reg_alpha=0.05,
        reg_lambda=1.0,
        tree_method="hist",
        n_jobs=workers,
        random_state=20260811,
        verbosity=0,
    )


def _metrics(observed: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    return {
        "n": len(observed),
        "mae": float(mean_absolute_error(observed, prediction)),
        "rmse": float(math.sqrt(mean_squared_error(observed, prediction))),
        "r2": float(r2_score(observed, prediction)),
        "pearson": float(pearsonr(observed, prediction).statistic),
        "spearman": float(spearmanr(observed, prediction).statistic),
    }


def _summarize(folds: pd.DataFrame) -> pd.DataFrame:
    summary = (
        folds.groupby("feature_family", as_index=False)
        .agg(
            folds=("fold", "size"),
            feature_count=("feature_count", "first"),
            mean_mae=("mae", "mean"),
            sd_mae=("mae", "std"),
            mean_rmse=("rmse", "mean"),
            sd_rmse=("rmse", "std"),
            mean_r2=("r2", "mean"),
            mean_pearson=("pearson", "mean"),
            mean_spearman=("spearman", "mean"),
            total_fit_seconds=("fit_elapsed_seconds", "sum"),
        )
        .sort_values(["mean_mae", "feature_family"], kind="stable")
    )
    baseline = float(summary.loc[summary["feature_family"].eq("2d_baseline"), "mean_mae"].iloc[0])
    summary["delta_mean_mae_vs_2d"] = summary["mean_mae"] - baseline
    summary["rank_by_mean_mae"] = np.arange(1, len(summary) + 1)
    return summary


def _validate(output: Path) -> dict[str, Any]:
    manifest = json.loads((output / "manifest.json").read_text())
    for binding in manifest["inputs"] + manifest["artifacts"]:
        path = Path(binding["path"])
        if not path.is_file() or path.stat().st_size != int(binding["bytes"]):
            raise FamilyCVError(f"bound file changed: {path}")
        if _sha256(path) != binding["sha256"]:
            raise FamilyCVError(f"bound hash changed: {path}")
        if path.suffix == ".parquet":
            if pq.read_metadata(path).num_rows != int(binding["rows"]):
                raise FamilyCVError(f"bound rows changed: {path}")
            if _schema_sha(path) != binding["arrow_schema_sha256"]:
                raise FamilyCVError(f"bound schema changed: {path}")
    folds = pd.read_parquet(output / "fold_metrics.parquet")
    predictions = pd.read_parquet(output / "training_oof_predictions.parquet")
    if len(folds) != 35 or set(predictions["source_partition"]) != {"train"}:
        raise FamilyCVError("cross-validation output contract mismatch")
    return {
        "status": "passed",
        "feature_families": 7,
        "fold_fits": len(folds),
        "validation_labels_opened": False,
        "test_labels_opened": False,
        "training_only_oof_prediction_rows": len(predictions),
    }


def _build(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.repo_root).resolve()
    matrix_root = Path(args.matrix_root).resolve()
    output = Path(args.output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FamilyCVError("output directory must be new and empty")
    matrix_validation = json.loads((matrix_root / "validation.json").read_text())
    if matrix_validation.get("status") != "passed":
        raise FamilyCVError("feature matrix is not validated")
    matrix = pd.read_parquet(matrix_root / "combined_feature_matrix.parquet")
    matrix = matrix[matrix["model_split"].eq("train")].copy()
    targets = _targets(root / OBSERVATIONS)
    frame = targets.merge(matrix, on="structure_id", validate="one_to_one")
    if len(frame) != len(targets) or set(frame["model_split"]) != {"train"}:
        raise FamilyCVError("training feature join is incomplete")
    families = _families(frame)
    groups = frame["scaffold_group_id"].astype(str).to_numpy()
    y = np.ascontiguousarray(frame["target_pic50"].to_numpy(dtype=np.float32))
    splits = list(GroupKFold(n_splits=5).split(frame, y, groups=groups))
    fold_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    for family_id, columns in families.items():
        X, invalid_count = _sanitize(frame, columns)
        print(f"family={family_id} features={len(columns)}", flush=True)
        for fold_index, (fit_index, heldout_index) in enumerate(splits):
            model = _model(args.workers, args.estimators)
            started = time.monotonic()
            model.fit(X[fit_index], y[fit_index], verbose=False)
            elapsed = time.monotonic() - started
            prediction = model.predict(X[heldout_index])
            row = {
                "feature_family": family_id,
                "fold": fold_index,
                "feature_count": len(columns),
                "fit_structures": len(fit_index),
                "heldout_structures": len(heldout_index),
                "fit_elapsed_seconds": elapsed,
                "invalid_or_overflow_cells_in_full_training_matrix": invalid_count,
                **_metrics(y[heldout_index], prediction),
            }
            fold_rows.append(row)
            prediction_frames.append(
                pd.DataFrame(
                    {
                        "feature_family": family_id,
                        "fold": fold_index,
                        "source_partition": "train",
                        "structure_id": frame.iloc[heldout_index]["structure_id"].to_numpy(),
                        "scaffold_group_id": groups[heldout_index],
                        "observed_pic50": y[heldout_index],
                        "predicted_pic50": prediction,
                    }
                )
            )
            print(_json(row), end="", flush=True)
    fold_frame = pd.DataFrame(fold_rows)
    summary = _summarize(fold_frame)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    fold_frame.to_parquet(output / "fold_metrics.parquet", index=False)
    summary.to_parquet(output / "family_summary.parquet", index=False)
    predictions.to_parquet(output / "training_oof_predictions.parquet", index=False)
    print("\nTRAINING-ONLY FAMILY SUMMARY\n", summary.to_string(index=False), flush=True)
    input_paths = [
        root / OBSERVATIONS,
        matrix_root / "combined_feature_matrix.parquet",
        matrix_root / "manifest.json",
        matrix_root / "validation.json",
        Path(__file__).resolve(),
    ]
    artifact_paths = [
        output / "fold_metrics.parquet",
        output / "family_summary.parquet",
        output / "training_oof_predictions.parquet",
    ]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "scientific_scope": {
            "partition": "train_only",
            "cross_validation_unit": "scaffold_group_id",
            "validation_labels_opened": False,
            "test_labels_opened": False,
            "hyperparameter_search_performed": False,
            "feature_family_comparison": "training_only_hypothesis_prioritization",
            "predictive_superiority_established": False,
        },
        "parameters": {
            "folds": 5,
            "estimators": args.estimators,
            "workers": args.workers,
            "seed": 20260811,
            "overflow_to_missing_threshold": MAX_ABSOLUTE_NUMERIC_FEATURE,
        },
        "counts": {
            "training_structures": len(frame),
            "feature_families": len(families),
            "fold_fits": len(fold_frame),
            "oof_prediction_rows": len(predictions),
        },
        "inputs": [_binding(path) for path in input_paths],
        "artifacts": [_binding(path) for path in artifact_paths],
        "software": {
            "python": sys.version,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pyarrow": pa.__version__,
            "scipy": scipy.__version__,
        },
    }
    manifest["manifest_payload_sha256"] = hashlib.sha256(_json(manifest).encode()).hexdigest()
    (output / "manifest.json").write_text(_json(manifest), encoding="utf-8")
    validation = _validate(output)
    (output / "validation.json").write_text(_json(validation), encoding="utf-8")
    return {"validation": validation, "summary": summary.to_dict("records")}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--matrix-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--estimators", type=int, default=250)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_root).resolve()
    result = _validate(output) if args.validate_only else _build(args)
    print(_json(result), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
