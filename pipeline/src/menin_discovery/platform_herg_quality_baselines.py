"""Run fixed CPU-only quantitative baselines for the hERG quality hierarchy.

Q1 is evaluated as structure-level pIC50 regression with a Morgan ridge model.
Q2 is evaluated separately with fundamental two-dimensional descriptors using
both an exact-only ridge and a penalized censored-Gaussian (Tobit) ridge.  These
are diagnostic anchors, not final models or evidence of predictive superiority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import Chem, rdBase
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdFingerprintGenerator
from scipy import sparse
from scipy.optimize import minimize
from scipy.special import log_ndtr
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

SCHEMA_VERSION = "platform-herg-quality-baselines/1.0"
SEED = 20260807
MORGAN_BITS = 1024
MORGAN_RADIUS = 2
RIDGE_ALPHA = 1.0
TOBIT_ALPHA = 1.0
MANIFEST_NAME = "quality_baseline_manifest.json"
Q1_TARGETS = "q1_structure_targets.parquet"
Q1_METRICS = "q1_metrics.parquet"
Q1_PREDICTIONS = "q1_validation_test_predictions.parquet"
Q1_MODEL = "q1_morgan_ridge.joblib"
Q2_TARGETS = "q2_ic50_structure_targets.parquet"
Q2_METRICS = "q2_metrics.parquet"
Q2_PREDICTIONS = "q2_validation_test_predictions.parquet"
Q2_MODEL = "q2_descriptor_models.joblib"
REPORT_NAME = "QUALITY_BASELINE_RESULTS.md"

DESCRIPTOR_NAMES = (
    "molecular_weight",
    "mol_logp",
    "tpsa",
    "hbond_donors",
    "hbond_acceptors",
    "rotatable_bonds",
    "ring_count",
    "aromatic_ring_count",
    "fraction_csp3",
    "formal_charge",
    "heavy_atom_count",
)

_TARGET_SCHEMA = pa.schema(
    [
        pa.field("task_id", pa.large_string(), nullable=False),
        pa.field("structure_id", pa.large_string(), nullable=False),
        pa.field("standardized_smiles", pa.large_string(), nullable=False),
        pa.field("model_split", pa.large_string(), nullable=False),
        pa.field("scaffold_group_id", pa.large_string(), nullable=False),
        pa.field("target_relation", pa.large_string(), nullable=False),
        pa.field("target_pic50", pa.float64()),
        pa.field("lower_pic50", pa.float64()),
        pa.field("upper_pic50", pa.float64()),
        pa.field("observation_count", pa.int64(), nullable=False),
        pa.field("exact_observation_count", pa.int64(), nullable=False),
        pa.field("censored_observation_count", pa.int64(), nullable=False),
        pa.field("measurement_technologies_json", pa.large_string(), nullable=False),
    ]
)

_METRIC_SCHEMA = pa.schema(
    [
        pa.field("task_id", pa.large_string(), nullable=False),
        pa.field("model", pa.large_string(), nullable=False),
        pa.field("partition", pa.large_string(), nullable=False),
        pa.field("n_exact_structures", pa.int64(), nullable=False),
        pa.field("mae", pa.float64(), nullable=False),
        pa.field("median_absolute_error", pa.float64(), nullable=False),
        pa.field("rmse", pa.float64(), nullable=False),
        pa.field("r2", pa.float64(), nullable=False),
        pa.field("pearson", pa.float64(), nullable=False),
        pa.field("spearman", pa.float64(), nullable=False),
    ]
)

_PREDICTION_SCHEMA = pa.schema(
    [
        pa.field("task_id", pa.large_string(), nullable=False),
        pa.field("structure_id", pa.large_string(), nullable=False),
        pa.field("model_split", pa.large_string(), nullable=False),
        pa.field("observed_pic50", pa.float64(), nullable=False),
        pa.field("mean_prediction", pa.float64(), nullable=False),
        pa.field("ridge_prediction", pa.float64(), nullable=False),
        pa.field("censored_gaussian_ridge_prediction", pa.float64()),
    ]
)


class HergQualityBaselineError(RuntimeError):
    """Raised when quality-baseline inputs or artifacts fail validation."""


def _json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_digest(body: Mapping[str, Any]) -> dict[str, Any]:
    output = dict(body)
    output["manifest_sha256"] = hashlib.sha256(_json(output).encode()).hexdigest()
    return output


def _checked_task(path: Path) -> Path:
    required = {
        "structure_id",
        "standardized_smiles",
        "model_split",
        "scaffold_group_id",
        "target_relation",
        "target_pic50",
        "measurement_technology",
        "eligible",
        "use_as_training_label",
    }
    if path.is_symlink() or not path.is_file() or path.suffix.casefold() != ".parquet":
        raise HergQualityBaselineError(f"missing or unsafe task artifact: {path}")
    missing = sorted(required - set(pq.ParquetFile(path).schema_arrow.names))
    if missing:
        raise HergQualityBaselineError(f"task artifact missing columns: {missing}")
    return path.resolve()


def _representative_smiles(rows: Sequence[Mapping[str, Any]]) -> str:
    counts = Counter(str(row["standardized_smiles"]) for row in rows)
    return min(counts, key=lambda value: (-counts[value], value))


def _aggregate_targets(path: Path, task_id: str) -> list[dict[str, Any]]:
    rows = pq.read_table(path).to_pylist()
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["eligible"] and row["use_as_training_label"] and row["target_pic50"] is not None:
            grouped[str(row["structure_id"])].append(row)
    output: list[dict[str, Any]] = []
    for structure_id, members in sorted(grouped.items()):
        splits = {str(row["model_split"]) for row in members}
        groups = {str(row["scaffold_group_id"]) for row in members}
        if len(splits) != 1 or len(groups) != 1 or next(iter(splits)) not in {"train", "validation", "test"}:
            raise HergQualityBaselineError(f"entity/scaffold split conflict for {structure_id}")
        # Approximate (~) values remain available upstream but are not upgraded
        # to exact points in this diagnostic regression release.
        exact = [float(row["target_pic50"]) for row in members if row["target_relation"] == "="]
        upper = [float(row["target_pic50"]) for row in members if row["target_relation"] in {"<", "<="}]
        lower = [float(row["target_pic50"]) for row in members if row["target_relation"] in {">", ">="}]
        if exact:
            relation, target = "=", float(np.median(exact))
            low, high = None, None
        elif upper and lower:
            low, high = max(lower), min(upper)
            if not low < high:
                raise HergQualityBaselineError(f"incompatible censoring bounds for {structure_id}")
            relation, target = "interval", None
        elif upper:
            relation, target, low, high = "<", None, None, min(upper)
        elif lower:
            relation, target, low, high = ">", None, max(lower), None
        else:
            continue
        output.append(
            {
                "task_id": task_id,
                "structure_id": structure_id,
                "standardized_smiles": _representative_smiles(members),
                "model_split": next(iter(splits)),
                "scaffold_group_id": next(iter(groups)),
                "target_relation": relation,
                "target_pic50": target,
                "lower_pic50": low,
                "upper_pic50": high,
                "observation_count": len(members),
                "exact_observation_count": len(exact),
                "censored_observation_count": len(upper) + len(lower),
                "measurement_technologies_json": _json(
                    sorted({str(row["measurement_technology"]) for row in members})
                ),
            }
        )
    if not output:
        raise HergQualityBaselineError(f"no quantitative structures found for {task_id}")
    return output


def _morgan(smiles: Sequence[str]) -> sparse.csr_matrix:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=MORGAN_RADIUS, fpSize=MORGAN_BITS)
    indices: list[int] = []
    indptr = [0]
    with rdBase.BlockLogs():
        for index, value in enumerate(smiles):
            molecule = Chem.MolFromSmiles(value)
            if molecule is None:
                raise HergQualityBaselineError(f"invalid SMILES at row {index}")
            indices.extend(int(bit) for bit in generator.GetFingerprint(molecule).GetOnBits())
            indptr.append(len(indices))
    return sparse.csr_matrix(
        (
            np.ones(len(indices), dtype=np.float64),
            np.asarray(indices, dtype=np.int32),
            np.asarray(indptr, dtype=np.int64),
        ),
        shape=(len(smiles), MORGAN_BITS),
    )


def _descriptors(smiles: Sequence[str]) -> np.ndarray:
    values: list[list[float]] = []
    with rdBase.BlockLogs():
        for index, value in enumerate(smiles):
            molecule = Chem.MolFromSmiles(value)
            if molecule is None:
                raise HergQualityBaselineError(f"invalid SMILES at row {index}")
            values.append(
                [
                    Descriptors.MolWt(molecule),
                    Crippen.MolLogP(molecule),
                    Descriptors.TPSA(molecule),
                    float(Lipinski.NumHDonors(molecule)),
                    float(Lipinski.NumHAcceptors(molecule)),
                    float(Lipinski.NumRotatableBonds(molecule)),
                    float(Lipinski.RingCount(molecule)),
                    float(Lipinski.NumAromaticRings(molecule)),
                    Descriptors.FractionCSP3(molecule),
                    float(Chem.GetFormalCharge(molecule)),
                    float(molecule.GetNumHeavyAtoms()),
                ]
            )
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.shape[1] != len(DESCRIPTOR_NAMES) or not np.isfinite(matrix).all():
        raise HergQualityBaselineError("descriptor calculation produced an invalid matrix")
    return matrix


def _regression_metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    if len(y) < 2:
        raise HergQualityBaselineError("at least two exact structures are required for evaluation")
    variable = float(np.ptp(prediction)) > 1e-12 and float(np.ptp(y)) > 1e-12
    pearson = float(pearsonr(y, prediction).statistic) if variable else 0.0
    spearman = float(spearmanr(y, prediction).statistic) if variable else 0.0
    return {
        "n_exact_structures": len(y),
        "mae": float(mean_absolute_error(y, prediction)),
        "median_absolute_error": float(median_absolute_error(y, prediction)),
        "rmse": float(math.sqrt(mean_squared_error(y, prediction))),
        "r2": float(r2_score(y, prediction)),
        "pearson": pearson,
        "spearman": spearman,
    }


def _fit_tobit(
    X: np.ndarray,
    relation: Sequence[str],
    target: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    alpha: float = TOBIT_ALPHA,
) -> dict[str, Any]:
    """Fit penalized Gaussian interval/censoring likelihood by L-BFGS."""

    n_features = X.shape[1]
    exact_values = target[np.asarray(relation) == "="]
    if len(exact_values) < 3:
        raise HergQualityBaselineError("Tobit fit requires at least three exact training structures")
    initial = np.zeros(n_features + 2, dtype=float)
    initial[0] = float(np.mean(exact_values))
    initial[-1] = math.log(max(float(np.std(exact_values)), 0.25))

    def objective(parameters: np.ndarray) -> float:
        intercept, beta, log_sigma = parameters[0], parameters[1:-1], parameters[-1]
        sigma = math.exp(float(np.clip(log_sigma, -8.0, 8.0)))
        mu = intercept + X @ beta
        total = 0.0
        for index, kind in enumerate(relation):
            if kind == "=":
                z = (target[index] - mu[index]) / sigma
                total += 0.5 * z * z + math.log(sigma) + 0.5 * math.log(2.0 * math.pi)
            elif kind == "<":
                total -= float(log_ndtr((upper[index] - mu[index]) / sigma))
            elif kind == ">":
                total -= float(log_ndtr((mu[index] - lower[index]) / sigma))
            elif kind == "interval":
                z_high = (upper[index] - mu[index]) / sigma
                z_low = (lower[index] - mu[index]) / sigma
                high = math.exp(float(log_ndtr(z_high)))
                low = math.exp(float(log_ndtr(z_low)))
                total -= math.log(max(high - low, 1e-300))
            else:
                raise HergQualityBaselineError(f"unsupported relation: {kind}")
        return float(total + 0.5 * alpha * np.dot(beta, beta))

    result = minimize(objective, initial, method="L-BFGS-B", options={"maxiter": 1000, "ftol": 1e-10})
    if not result.success or not np.isfinite(result.fun):
        raise HergQualityBaselineError(f"censored Gaussian ridge failed: {result.message}")
    return {
        "intercept": float(result.x[0]),
        "coef": np.asarray(result.x[1:-1], dtype=float),
        "sigma": math.exp(float(result.x[-1])),
        "objective": float(result.fun),
        "iterations": int(result.nit),
        "converged": True,
        "alpha": alpha,
    }


def _write_table(path: Path, rows: Sequence[Mapping[str, Any]], schema: pa.Schema) -> dict[str, Any]:
    table = pa.Table.from_pylist(list(rows), schema=schema)
    pq.write_table(table, path, compression="zstd", use_dictionary=False, version="2.6")
    return {"path": path.name, "rows": table.num_rows, "bytes": path.stat().st_size, "sha256": _sha256(path)}


def _write_report(path: Path, manifest: Mapping[str, Any]) -> None:
    q1_test = [
        row for row in manifest["metrics"] if row["task_id"] == "Q1" and row["partition"] == "locked_test"
    ]
    q2_test = [
        row for row in manifest["metrics"] if row["task_id"] == "Q2" and row["partition"] == "locked_test"
    ]
    lines = [
        "# hERG quality-specific CPU baseline results",
        "",
        "These are fixed diagnostic baselines on the entity-exclusive scaffold split. They do not establish predictive superiority, prospective validity, or clinical utility.",
        "",
        "## Data contract",
        "",
        f"- Q1: {manifest['counts']['q1_structures']:,} structure-level exact pIC50 targets.",
        f"- Q2: {manifest['counts']['q2_structures']:,} IC50/pIC50 structures; {manifest['counts']['q2_censored_structures']:,} are censoring-only constraints.",
        "- Test partitions were evaluated once and never used for fitting or model selection.",
        "- Q2 is deliberately separate from AC50, fixed-dose inhibition, QT, and other endpoint semantics.",
        "",
        "## Locked-test metrics",
        "",
        "| Task | Model | n | MAE | RMSE | R2 | Spearman |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in [*q1_test, *q2_test]:
        lines.append(
            f"| {row['task_id']} | {row['model']} | {row['n_exact_structures']:,} | {row['mae']:.4f} | {row['rmse']:.4f} | {row['r2']:.4f} | {row['spearman']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "The project already has an established data-engineering advantage in WT scope, scale, assay semantics, evidence levels, entity-exclusive splitting, and provenance. Model-performance superiority remains unestablished until like-for-like external and prospective comparisons are run.",
            "",
            "The censored Gaussian model is a penalized Tobit/interval likelihood, not a Cox survival model. Its present Q2 result is low-power and primarily verifies that bounded IC50 values can be retained instead of discarded or falsely treated as exact.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_quality_baselines(*, q1_path: Path, q2_path: Path, output_root: Path) -> dict[str, Any]:
    q1_path, q2_path = _checked_task(q1_path), _checked_task(q2_path)
    output_root = output_root.resolve()
    if output_root.exists():
        raise HergQualityBaselineError("output_root already exists")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.staging.", dir=output_root.parent))
    try:
        q1 = _aggregate_targets(q1_path, "Q1")
        q2 = _aggregate_targets(q2_path, "Q2")
        artifacts: dict[str, Any] = {}
        artifacts[Q1_TARGETS] = _write_table(staging / Q1_TARGETS, q1, _TARGET_SCHEMA)
        artifacts[Q2_TARGETS] = _write_table(staging / Q2_TARGETS, q2, _TARGET_SCHEMA)

        metrics: list[dict[str, Any]] = []
        predictions: dict[str, list[dict[str, Any]]] = {"Q1": [], "Q2": []}

        q1_smiles = [str(row["standardized_smiles"]) for row in q1]
        q1_X = _morgan(q1_smiles)
        q1_y = np.asarray([float(row["target_pic50"]) for row in q1])
        q1_split = np.asarray([str(row["model_split"]) for row in q1])
        q1_train = q1_split == "train"
        q1_mean = float(np.mean(q1_y[q1_train]))
        q1_model = Ridge(alpha=RIDGE_ALPHA, solver="lsqr", fit_intercept=True)
        q1_model.fit(q1_X[q1_train], q1_y[q1_train])
        q1_ridge = q1_model.predict(q1_X)
        for partition, name in (("validation", "validation"), ("test", "locked_test")):
            mask = q1_split == partition
            for model_name, prediction in (
                ("train_mean", np.full(mask.sum(), q1_mean)),
                ("morgan_ridge", q1_ridge[mask]),
            ):
                metrics.append(
                    {
                        "task_id": "Q1",
                        "model": model_name,
                        "partition": name,
                        **_regression_metrics(q1_y[mask], prediction),
                    }
                )
            for index in np.flatnonzero(mask):
                predictions["Q1"].append(
                    {
                        "task_id": "Q1",
                        "structure_id": q1[index]["structure_id"],
                        "model_split": str(q1_split[index]),
                        "observed_pic50": float(q1_y[index]),
                        "mean_prediction": q1_mean,
                        "ridge_prediction": float(q1_ridge[index]),
                        "censored_gaussian_ridge_prediction": None,
                    }
                )
        joblib.dump(q1_model, staging / Q1_MODEL, compress=3)
        artifacts[Q1_MODEL] = {
            "path": Q1_MODEL,
            "bytes": (staging / Q1_MODEL).stat().st_size,
            "sha256": _sha256(staging / Q1_MODEL),
        }

        q2_smiles = [str(row["standardized_smiles"]) for row in q2]
        q2_X_raw = _descriptors(q2_smiles)
        q2_split = np.asarray([str(row["model_split"]) for row in q2])
        q2_relation = [str(row["target_relation"]) for row in q2]
        q2_target = np.asarray(
            [np.nan if row["target_pic50"] is None else float(row["target_pic50"]) for row in q2]
        )
        q2_lower = np.asarray(
            [np.nan if row["lower_pic50"] is None else float(row["lower_pic50"]) for row in q2]
        )
        q2_upper = np.asarray(
            [np.nan if row["upper_pic50"] is None else float(row["upper_pic50"]) for row in q2]
        )
        q2_train = q2_split == "train"
        q2_train_exact = q2_train & np.asarray(q2_relation).__eq__("=")
        scaler = StandardScaler().fit(q2_X_raw[q2_train])
        q2_X = scaler.transform(q2_X_raw)
        q2_mean = float(np.mean(q2_target[q2_train_exact]))
        q2_ridge_model = Ridge(alpha=RIDGE_ALPHA, solver="lsqr", fit_intercept=True)
        q2_ridge_model.fit(q2_X[q2_train_exact], q2_target[q2_train_exact])
        q2_ridge = q2_ridge_model.predict(q2_X)
        q2_tobit = _fit_tobit(
            q2_X[q2_train],
            [q2_relation[index] for index in np.flatnonzero(q2_train)],
            q2_target[q2_train],
            q2_lower[q2_train],
            q2_upper[q2_train],
        )
        q2_tobit_prediction = q2_tobit["intercept"] + q2_X @ q2_tobit["coef"]
        for partition, name in (("validation", "validation"), ("test", "locked_test")):
            mask = (q2_split == partition) & np.asarray(q2_relation).__eq__("=")
            for model_name, prediction in (
                ("train_mean", np.full(mask.sum(), q2_mean)),
                ("descriptor_ridge_exact_only", q2_ridge[mask]),
                ("descriptor_censored_gaussian_ridge", q2_tobit_prediction[mask]),
            ):
                metrics.append(
                    {
                        "task_id": "Q2",
                        "model": model_name,
                        "partition": name,
                        **_regression_metrics(q2_target[mask], prediction),
                    }
                )
            for index in np.flatnonzero(mask):
                predictions["Q2"].append(
                    {
                        "task_id": "Q2",
                        "structure_id": q2[index]["structure_id"],
                        "model_split": str(q2_split[index]),
                        "observed_pic50": float(q2_target[index]),
                        "mean_prediction": q2_mean,
                        "ridge_prediction": float(q2_ridge[index]),
                        "censored_gaussian_ridge_prediction": float(q2_tobit_prediction[index]),
                    }
                )
        q2_bundle = {
            "descriptor_names": DESCRIPTOR_NAMES,
            "scaler": scaler,
            "exact_ridge": q2_ridge_model,
            "censored_gaussian_ridge": q2_tobit,
        }
        joblib.dump(q2_bundle, staging / Q2_MODEL, compress=3)
        artifacts[Q2_MODEL] = {
            "path": Q2_MODEL,
            "bytes": (staging / Q2_MODEL).stat().st_size,
            "sha256": _sha256(staging / Q2_MODEL),
        }

        artifacts[Q1_METRICS] = _write_table(
            staging / Q1_METRICS, [row for row in metrics if row["task_id"] == "Q1"], _METRIC_SCHEMA
        )
        artifacts[Q2_METRICS] = _write_table(
            staging / Q2_METRICS, [row for row in metrics if row["task_id"] == "Q2"], _METRIC_SCHEMA
        )
        artifacts[Q1_PREDICTIONS] = _write_table(
            staging / Q1_PREDICTIONS, predictions["Q1"], _PREDICTION_SCHEMA
        )
        artifacts[Q2_PREDICTIONS] = _write_table(
            staging / Q2_PREDICTIONS, predictions["Q2"], _PREDICTION_SCHEMA
        )

        manifest = _manifest_digest(
            {
                "schema_version": SCHEMA_VERSION,
                "inputs": {
                    "q1": {
                        "path": str(q1_path),
                        "sha256": _sha256(q1_path),
                        "rows": pq.ParquetFile(q1_path).metadata.num_rows,
                    },
                    "q2": {
                        "path": str(q2_path),
                        "sha256": _sha256(q2_path),
                        "rows": pq.ParquetFile(q2_path).metadata.num_rows,
                    },
                },
                "scientific_contract": {
                    "status": "cpu_diagnostic_not_external_or_prospective_validation",
                    "target": "wild-type-or-unspecified hERG quantitative potency; Q1 and Q2 never pooled",
                    "split": "fixed entity-exclusive whole-scaffold split",
                    "test_used_for_fitting_or_selection": False,
                    "predictive_superiority_established": False,
                    "established_project_advantages": [
                        "wild-type scope governance",
                        "larger standardized public evidence layer",
                        "assay and quality stratification",
                        "entity-exclusive scaffold evaluation",
                        "record-level provenance",
                    ],
                },
                "configuration": {
                    "seed": SEED,
                    "q1": {
                        "features": f"Morgan radius {MORGAN_RADIUS}, {MORGAN_BITS} bits",
                        "ridge_alpha": RIDGE_ALPHA,
                        "aggregation": "structure median",
                    },
                    "q2": {
                        "features": list(DESCRIPTOR_NAMES),
                        "ridge_alpha": RIDGE_ALPHA,
                        "tobit_alpha": TOBIT_ALPHA,
                        "censoring_likelihood": "Gaussian exact/left/right/interval",
                    },
                },
                "counts": {
                    "q1_structures": len(q1),
                    "q2_structures": len(q2),
                    "q2_exact_structures": sum(row["target_relation"] == "=" for row in q2),
                    "q2_censored_structures": sum(row["target_relation"] != "=" for row in q2),
                },
                "metrics": metrics,
                "q2_optimizer": {key: value for key, value in q2_tobit.items() if key != "coef"},
                "artifacts": artifacts,
                "substantive_large_model_training_started": False,
            }
        )
        _write_report(staging / REPORT_NAME, manifest)
        artifacts[REPORT_NAME] = {
            "path": REPORT_NAME,
            "bytes": (staging / REPORT_NAME).stat().st_size,
            "sha256": _sha256(staging / REPORT_NAME),
        }
        manifest["artifacts"] = artifacts
        manifest.pop("manifest_sha256")
        manifest = _manifest_digest(manifest)
        (staging / MANIFEST_NAME).write_text(_json(manifest) + "\n", encoding="utf-8")
        os.replace(staging, output_root)
        validate_quality_baselines(output_root)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def validate_quality_baselines(output_root: Path) -> dict[str, Any]:
    root = output_root.resolve()
    manifest_path = root / MANIFEST_NAME
    if root.is_symlink() or not root.is_dir() or not manifest_path.is_file():
        raise HergQualityBaselineError("missing quality-baseline release")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    supplied = manifest.pop("manifest_sha256", None)
    expected = hashlib.sha256(_json(manifest).encode()).hexdigest()
    manifest["manifest_sha256"] = supplied
    if supplied != expected:
        raise HergQualityBaselineError("manifest digest mismatch")
    for name, metadata in manifest["artifacts"].items():
        path = root / name
        if path.is_symlink() or not path.is_file() or _sha256(path) != metadata["sha256"]:
            raise HergQualityBaselineError(f"artifact digest mismatch: {name}")
    for name, schema in (
        (Q1_TARGETS, _TARGET_SCHEMA),
        (Q2_TARGETS, _TARGET_SCHEMA),
        (Q1_METRICS, _METRIC_SCHEMA),
        (Q2_METRICS, _METRIC_SCHEMA),
        (Q1_PREDICTIONS, _PREDICTION_SCHEMA),
        (Q2_PREDICTIONS, _PREDICTION_SCHEMA),
    ):
        if pq.ParquetFile(root / name).schema_arrow != schema:
            raise HergQualityBaselineError(f"artifact schema mismatch: {name}")
    if manifest["scientific_contract"]["predictive_superiority_established"] is not False:
        raise HergQualityBaselineError("diagnostic run cannot assert predictive superiority")
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--q1", type=Path, required=True)
    build.add_argument("--q2", type=Path, required=True)
    build.add_argument("--output-root", type=Path, required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    result = (
        run_quality_baselines(q1_path=args.q1, q2_path=args.q2, output_root=args.output_root)
        if args.command == "build"
        else validate_quality_baselines(args.output_root)
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
