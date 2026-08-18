#!/usr/bin/env python3
"""Run a fixed validation-only hERG 2D versus 2D+3D ablation.

Only exact quantitative training and validation labels are returned to the
analysis frame.  The locked test partition is not scored, used for fitting,
used for early stopping, or used for model selection.  Hyperparameters are
fixed in this script before validation metrics are calculated.
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

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import scipy
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

SCHEMA_VERSION = "platform-local-herg-feature-ablation/1.0"
MAX_ABSOLUTE_NUMERIC_FEATURE = 1.0e30
OBSERVATIONS = Path(
    "research/data/platform/processed/herg_hierarchy/v1_6_training_surfaces/"
    "herg_training_observations.parquet"
)
EXCLUDED_CONTINUOUS_COLUMNS = {
    "f3d__feature_order",
    "f3d__energy_min_kcal_mol",
}


class AblationError(RuntimeError):
    """Raised when the diagnostic ablation cannot be executed safely."""


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


def _load_targets(path: Path, limit_train: int | None) -> pd.DataFrame:
    columns = [
        "structure_id",
        "model_split",
        "potency_relation_pic50",
        "potency_pic50_point",
        "standardized_pic50_primary",
    ]
    table = pq.read_table(
        path,
        columns=columns,
        filters=[
            ("standardized_pic50_primary", "=", True),
            ("potency_relation_pic50", "=", "="),
            ("model_split", "in", ["train", "validation"]),
        ],
    )
    frame = table.to_pandas()
    if set(frame["model_split"].astype(str)) - {"train", "validation"}:
        raise AblationError("locked test values entered the analysis frame")
    if frame["potency_pic50_point"].isna().any():
        raise AblationError("exact target rows contain missing points")
    targets = (
        frame.groupby(["structure_id", "model_split"], as_index=False)
        .agg(
            target_pic50=("potency_pic50_point", "median"),
            exact_observation_count=("potency_pic50_point", "size"),
        )
        .sort_values(["model_split", "structure_id"], kind="stable")
    )
    if targets["structure_id"].duplicated().any():
        raise AblationError("target structures cross partitions")
    if limit_train is not None:
        train = targets[targets["model_split"].eq("train")].head(limit_train)
        validation = targets[targets["model_split"].eq("validation")]
        targets = pd.concat([train, validation], ignore_index=True)
    return targets


def _feature_columns(matrix: pd.DataFrame) -> tuple[list[str], list[str]]:
    two_d = sorted(column for column in matrix if column.startswith("rdkit2d__"))
    three_d = sorted(
        column
        for column in matrix
        if column.startswith("f3d__")
        and column not in EXCLUDED_CONTINUOUS_COLUMNS
        and pd.api.types.is_numeric_dtype(matrix[column])
    )
    if not two_d or not three_d:
        raise AblationError("required 2D or 3D feature block is absent")
    return two_d, three_d


def _remove_constant_train(frame: pd.DataFrame, columns: list[str], train_mask: np.ndarray) -> list[str]:
    retained: list[str] = []
    for column in columns:
        values = pd.to_numeric(frame.loc[train_mask, column], errors="coerce").to_numpy(dtype=np.float64)
        finite = values[np.isfinite(values)]
        if len(finite) and float(finite.max() - finite.min()) > 1e-12:
            retained.append(column)
    return retained


def _metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    return {
        "n": len(observed),
        "mae": float(mean_absolute_error(observed, predicted)),
        "rmse": float(math.sqrt(mean_squared_error(observed, predicted))),
        "r2": float(r2_score(observed, predicted)),
        "pearson": float(pearsonr(observed, predicted).statistic),
        "spearman": float(spearmanr(observed, predicted).statistic),
    }


def _model(seed: int, workers: int, estimators: int) -> XGBRegressor:
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
        random_state=seed,
        verbosity=1,
    )


def _fit_one(
    frame: pd.DataFrame,
    columns: list[str],
    *,
    model_id: str,
    workers: int,
    estimators: int,
    output: Path,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    train_mask = frame["model_split"].eq("train").to_numpy()
    validation_mask = frame["model_split"].eq("validation").to_numpy()
    retained = _remove_constant_train(frame, columns, train_mask)
    raw = frame[retained].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64, copy=True)
    invalid = ~np.isfinite(raw) | (np.abs(raw) > MAX_ABSOLUTE_NUMERIC_FEATURE)
    invalid_numeric_cells = int(invalid.sum())
    raw[invalid] = np.nan
    X = np.ascontiguousarray(raw, dtype=np.float32)
    y = np.ascontiguousarray(frame["target_pic50"].to_numpy(dtype=np.float32))
    model = _model(20260811, workers, estimators)
    started = time.monotonic()
    model.fit(X[train_mask], y[train_mask], verbose=False)
    elapsed = time.monotonic() - started
    prediction = model.predict(X[validation_mask])
    observed = y[validation_mask]
    metrics = {
        "model_id": model_id,
        "partition": "validation",
        "feature_count": len(retained),
        "training_structures": int(train_mask.sum()),
        "invalid_or_overflow_feature_cells_treated_as_missing": invalid_numeric_cells,
        "fit_elapsed_seconds": elapsed,
        **_metrics(observed, prediction),
    }
    predictions = pd.DataFrame(
        {
            "model_id": model_id,
            "structure_id": frame.loc[validation_mask, "structure_id"].to_numpy(),
            "model_split": "validation",
            "observed_pic50": observed,
            "predicted_pic50": prediction,
            "residual": observed - prediction,
        }
    )
    importance = pd.DataFrame(
        {
            "model_id": model_id,
            "feature_name": retained,
            "feature_importance_gain_proxy": model.feature_importances_.astype(float),
        }
    ).sort_values(
        ["feature_importance_gain_proxy", "feature_name"],
        ascending=[False, True],
        kind="stable",
    )
    joblib.dump(model, output / f"{model_id}.joblib", compress=3)
    return metrics, predictions, importance


def _validate(output: Path) -> dict[str, Any]:
    manifest = json.loads((output / "manifest.json").read_text())
    for binding in manifest["inputs"] + manifest["artifacts"]:
        path = Path(binding["path"])
        if not path.is_file() or path.stat().st_size != int(binding["bytes"]):
            raise AblationError(f"bound file changed: {path}")
        if _sha256(path) != binding["sha256"]:
            raise AblationError(f"bound hash changed: {path}")
        if path.suffix == ".parquet":
            if pq.read_metadata(path).num_rows != int(binding["rows"]):
                raise AblationError(f"bound row count changed: {path}")
            if _schema_sha(path) != binding["arrow_schema_sha256"]:
                raise AblationError(f"bound schema changed: {path}")
    metrics = pd.read_parquet(output / "validation_metrics.parquet")
    predictions = pd.read_parquet(output / "validation_predictions.parquet")
    if set(metrics["partition"]) != {"validation"} or set(predictions["model_split"]) != {"validation"}:
        raise AblationError("a non-validation partition entered evaluation artifacts")
    return {
        "status": "passed",
        "models": len(metrics),
        "validation_prediction_rows": len(predictions),
        "test_prediction_rows": 0,
        "test_labels_returned_to_analysis_frame": False,
    }


def _build(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.repo_root).resolve()
    matrix_root = Path(args.matrix_root).resolve()
    output = Path(args.output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise AblationError("output directory must be new and empty")
    matrix_path = matrix_root / "combined_feature_matrix.parquet"
    matrix_validation = json.loads((matrix_root / "validation.json").read_text())
    if matrix_validation.get("status") != "passed":
        raise AblationError("feature matrix did not pass validation")
    matrix = pd.read_parquet(matrix_path)
    targets = _load_targets(root / OBSERVATIONS, args.limit_train)
    frame = targets.merge(matrix, on=["structure_id", "model_split"], validate="one_to_one")
    if len(frame) != len(targets):
        raise AblationError("exact target feature join is incomplete")
    two_d, three_d = _feature_columns(frame)
    model_specs = {
        "xgb_2d_descriptors": two_d,
        "xgb_2d_plus_conformer3d": [*two_d, *three_d],
    }
    metric_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    importance_frames: list[pd.DataFrame] = []
    for model_id, columns in model_specs.items():
        print(f"fitting {model_id} with {len(columns)} candidate features", flush=True)
        metrics, predictions, importance = _fit_one(
            frame,
            columns,
            model_id=model_id,
            workers=args.workers,
            estimators=args.estimators,
            output=output,
        )
        metric_rows.append(metrics)
        prediction_frames.append(predictions)
        importance_frames.append(importance)
        print(_canonical_json(metrics), end="", flush=True)
    metrics_frame = pd.DataFrame(metric_rows)
    predictions_frame = pd.concat(prediction_frames, ignore_index=True)
    importance_frame = pd.concat(importance_frames, ignore_index=True)
    targets.to_parquet(output / "exact_structure_targets_train_validation.parquet", index=False)
    metrics_frame.to_parquet(output / "validation_metrics.parquet", index=False)
    predictions_frame.to_parquet(output / "validation_predictions.parquet", index=False)
    importance_frame.to_parquet(output / "feature_importance.parquet", index=False)
    artifact_paths = [
        output / "exact_structure_targets_train_validation.parquet",
        output / "validation_metrics.parquet",
        output / "validation_predictions.parquet",
        output / "feature_importance.parquet",
        output / "xgb_2d_descriptors.joblib",
        output / "xgb_2d_plus_conformer3d.joblib",
    ]
    input_paths = [
        root / OBSERVATIONS,
        matrix_path,
        matrix_root / "manifest.json",
        matrix_root / "validation.json",
        Path(__file__).resolve(),
    ]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "scientific_scope": {
            "status": "cpu_diagnostic_internal_validation_ablation",
            "target": "structure-median exact pIC50",
            "censored_rows_used": False,
            "test_labels_returned_to_analysis_frame": False,
            "test_scoring_performed": False,
            "validation_used_for_early_stopping": False,
            "hyperparameter_search_performed": False,
            "predictive_superiority_established": False,
        },
        "parameters": {
            "workers": args.workers,
            "estimators": args.estimators,
            "limit_train": args.limit_train,
            "seed": 20260811,
            "absolute_numeric_overflow_to_missing_threshold": MAX_ABSOLUTE_NUMERIC_FEATURE,
        },
        "counts": {
            "training_structures": int(frame["model_split"].eq("train").sum()),
            "validation_structures": int(frame["model_split"].eq("validation").sum()),
            "models": 2,
        },
        "software": {
            "python": sys.version,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pyarrow": pa.__version__,
            "scipy": scipy.__version__,
        },
        "inputs": [_binding(path) for path in input_paths],
        "artifacts": [_binding(path) for path in artifact_paths],
    }
    manifest["manifest_payload_sha256"] = hashlib.sha256(_canonical_json(manifest).encode()).hexdigest()
    (output / "manifest.json").write_text(_canonical_json(manifest), encoding="utf-8")
    validation = _validate(output)
    (output / "validation.json").write_text(_canonical_json(validation), encoding="utf-8")
    return {"validation": validation, "metrics": metric_rows}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--matrix-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--estimators", type=int, default=400)
    parser.add_argument("--limit-train", type=int)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_root).resolve()
    result = _validate(output) if args.validate_only else _build(args)
    print(_canonical_json(result), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
