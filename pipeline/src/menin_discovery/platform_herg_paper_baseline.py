"""Run a fixed, CPU-only molecular baseline on the frozen hERG scaffold split.

This is a diagnostic anchor, not a superiority claim.  Model settings and the
decision-threshold rule are fixed; the threshold is selected on validation and
applied once to the locked test partition.  The test partition is never used
for fitting or threshold selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import Chem, rdBase
from rdkit.Chem import rdFingerprintGenerator
from scipy import sparse
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    matthews_corrcoef,
    roc_auc_score,
)

SCHEMA_VERSION = "platform-herg-paper-baseline/1.0"
SEED = 20260807
N_BITS = 1024
RADIUS = 2
MAX_ITER = 100
MANIFEST_NAME = "baseline_manifest.json"
METRICS_NAME = "baseline_metrics.parquet"
PREDICTIONS_NAME = "locked_evaluation_predictions.parquet"
MODEL_NAME = "morgan_sgd_logistic.joblib"

_REQUIRED = (
    "structure_id",
    "standardized_smiles",
    "herg_blocker_label",
    "split",
    "scaffold_group_id",
)

_METRICS_SCHEMA = pa.schema(
    [
        pa.field("model", pa.large_string(), nullable=False),
        pa.field("partition", pa.large_string(), nullable=False),
        pa.field("threshold", pa.float64(), nullable=False),
        pa.field("n", pa.int64(), nullable=False),
        pa.field("positives", pa.int64(), nullable=False),
        pa.field("roc_auc", pa.float64(), nullable=False),
        pa.field("average_precision", pa.float64(), nullable=False),
        pa.field("balanced_accuracy", pa.float64(), nullable=False),
        pa.field("mcc", pa.float64(), nullable=False),
        pa.field("brier", pa.float64(), nullable=False),
        pa.field("tn", pa.int64(), nullable=False),
        pa.field("fp", pa.int64(), nullable=False),
        pa.field("fn", pa.int64(), nullable=False),
        pa.field("tp", pa.int64(), nullable=False),
    ]
)

_PREDICTION_SCHEMA = pa.schema(
    [
        pa.field("structure_id", pa.large_string(), nullable=False),
        pa.field("split", pa.large_string(), nullable=False),
        pa.field("herg_blocker_label", pa.int8(), nullable=False),
        pa.field("dummy_prior_probability", pa.float64(), nullable=False),
        pa.field("morgan_sgd_probability", pa.float64(), nullable=False),
        pa.field("morgan_sgd_prediction", pa.int8(), nullable=False),
    ]
)


class HergPaperBaselineError(RuntimeError):
    """Raised when benchmark inputs or generated artifacts fail closed."""


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_with_digest(body: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(body)
    result["manifest_sha256"] = hashlib.sha256(_canonical_json_bytes(result)).hexdigest()
    return result


def _checked_split(path: Path) -> Path:
    if path.is_symlink() or not path.is_file() or path.suffix.casefold() != ".parquet":
        raise HergPaperBaselineError(f"Missing, unsafe, or non-Parquet split input: {path}")
    path = path.resolve()
    names = set(pq.ParquetFile(path).schema_arrow.names)
    missing = sorted(set(_REQUIRED) - names)
    if missing:
        raise HergPaperBaselineError(f"Split input is missing required columns: {missing}")
    return path


def _load_split(path: Path) -> dict[str, list[Any]]:
    rows = pq.read_table(path, columns=list(_REQUIRED)).to_pylist()
    if not rows:
        raise HergPaperBaselineError("Split input is empty")
    ids: set[str] = set()
    group_split: dict[str, str] = {}
    output: dict[str, list[Any]] = {column: [] for column in _REQUIRED}
    for row in rows:
        structure_id = str(row["structure_id"] or "").strip()
        smiles = str(row["standardized_smiles"] or "").strip()
        split = str(row["split"] or "").strip()
        group = str(row["scaffold_group_id"] or "").strip()
        if not structure_id or not smiles or not group or split not in {"train", "validation", "test"}:
            raise HergPaperBaselineError(
                "Split contains blank identifiers, structures, groups, or invalid partitions"
            )
        if structure_id in ids:
            raise HergPaperBaselineError(f"Duplicate structure_id: {structure_id}")
        ids.add(structure_id)
        if group in group_split and group_split[group] != split:
            raise HergPaperBaselineError(f"Scaffold group leaked across partitions: {group}")
        group_split[group] = split
        try:
            label = int(row["herg_blocker_label"])
        except (TypeError, ValueError) as error:
            raise HergPaperBaselineError("Labels must be integers in {0, 1}") from error
        if label not in {0, 1}:
            raise HergPaperBaselineError("Labels must be integers in {0, 1}")
        output["structure_id"].append(structure_id)
        output["standardized_smiles"].append(smiles)
        output["herg_blocker_label"].append(label)
        output["split"].append(split)
        output["scaffold_group_id"].append(group)
    if set(output["split"]) != {"train", "validation", "test"}:
        raise HergPaperBaselineError("All train/validation/test partitions must be populated")
    for split in ("train", "validation", "test"):
        labels = {
            output["herg_blocker_label"][i] for i, value in enumerate(output["split"]) if value == split
        }
        if labels != {0, 1}:
            raise HergPaperBaselineError(f"Partition {split!r} must contain both classes")
    return output


def _morgan_matrix(smiles_values: Sequence[str]) -> sparse.csr_matrix:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    indices: list[int] = []
    indptr = [0]
    with rdBase.BlockLogs():
        for row_number, smiles in enumerate(smiles_values):
            molecule = Chem.MolFromSmiles(smiles)
            if molecule is None:
                raise HergPaperBaselineError(f"Invalid standardized SMILES at row {row_number}")
            indices.extend(int(bit) for bit in generator.GetFingerprint(molecule).GetOnBits())
            indptr.append(len(indices))
    data = np.ones(len(indices), dtype=np.float32)
    return sparse.csr_matrix(
        (data, np.asarray(indices, dtype=np.int32), np.asarray(indptr, dtype=np.int64)),
        shape=(len(smiles_values), N_BITS),
        dtype=np.float32,
    )


def _metrics(y: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, Any]:
    probability = np.clip(np.asarray(probability, dtype=float), 1e-7, 1 - 1e-7)
    prediction = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "n": int(len(y)),
        "positives": int(np.sum(y)),
        "roc_auc": float(roc_auc_score(y, probability)),
        "average_precision": float(average_precision_score(y, probability)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "mcc": float(matthews_corrcoef(y, prediction)),
        "brier": float(brier_score_loss(y, probability)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def _validation_threshold(y: np.ndarray, probability: np.ndarray) -> float:
    # Exact O(n log n) sweep over distinct score thresholds.  Recomputing a
    # confusion matrix once per observation is quadratic on the 33k-row
    # validation partition and is both unnecessary and prohibitively slow.
    order = np.argsort(-probability, kind="stable")
    scores = probability[order]
    labels = y[order].astype(np.int64)
    cumulative_tp = np.cumsum(labels)
    cumulative_fp = np.cumsum(1 - labels)
    group_ends = np.flatnonzero(np.r_[scores[:-1] != scores[1:], True])
    tp = cumulative_tp[group_ends].astype(float)
    fp = cumulative_fp[group_ends].astype(float)
    positives = float(np.sum(labels))
    negatives = float(len(labels) - positives)
    fn = positives - tp
    tn = negatives - fp
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = np.divide(tp * tn - fp * fn, denominator, out=np.zeros_like(tp), where=denominator > 0)
    best_score = np.max(mcc)
    best_thresholds = scores[group_ends[np.flatnonzero(mcc == best_score)]]
    return float(np.min(best_thresholds))


def run_herg_paper_baseline(*, split_path: Path, output_root: Path) -> dict[str, Any]:
    """Fit the fixed diagnostic baseline and evaluate the locked scaffold split."""

    split_path = _checked_split(split_path)
    data = _load_split(split_path)
    output_root = output_root.resolve()
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent))
    try:
        X = _morgan_matrix(data["standardized_smiles"])
        y = np.asarray(data["herg_blocker_label"], dtype=np.int8)
        partitions = np.asarray(data["split"], dtype=str)
        train = partitions == "train"
        validation = partitions == "validation"
        test = partitions == "test"
        dummy = DummyClassifier(strategy="prior")
        dummy.fit(X[train], y[train])
        model = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=1e-5,
            class_weight="balanced",
            max_iter=MAX_ITER,
            tol=1e-4,
            average=True,
            random_state=SEED,
        )
        model.fit(X[train], y[train])
        dummy_probability = dummy.predict_proba(X)[:, 1]
        model_probability = model.predict_proba(X)[:, 1]
        threshold = _validation_threshold(y[validation], model_probability[validation])
        metric_rows: list[dict[str, Any]] = []
        for name, probability, model_threshold in (
            ("dummy_prior", dummy_probability, 0.5),
            ("morgan_r2_1024_sgd_logistic", model_probability, threshold),
        ):
            for partition, mask in (("validation", validation), ("locked_test", test)):
                metric_rows.append(
                    {
                        "model": name,
                        "partition": partition,
                        **_metrics(y[mask], probability[mask], model_threshold),
                    }
                )
        pq.write_table(pa.Table.from_pylist(metric_rows, schema=_METRICS_SCHEMA), temporary / METRICS_NAME)
        eval_mask = validation | test
        prediction_rows = [
            {
                "structure_id": data["structure_id"][index],
                "split": data["split"][index],
                "herg_blocker_label": int(y[index]),
                "dummy_prior_probability": float(dummy_probability[index]),
                "morgan_sgd_probability": float(model_probability[index]),
                "morgan_sgd_prediction": int(model_probability[index] >= threshold),
            }
            for index in np.flatnonzero(eval_mask)
        ]
        pq.write_table(
            pa.Table.from_pylist(prediction_rows, schema=_PREDICTION_SCHEMA),
            temporary / PREDICTIONS_NAME,
            compression="zstd",
        )
        joblib.dump(model, temporary / MODEL_NAME, compress=3)
        artifacts = {
            name: {"sha256": _sha256_file(temporary / name), "bytes": (temporary / name).stat().st_size}
            for name in (METRICS_NAME, PREDICTIONS_NAME, MODEL_NAME)
        }
        manifest = _manifest_with_digest(
            {
                "schema_version": SCHEMA_VERSION,
                "input": {"path": str(split_path), "sha256": _sha256_file(split_path)},
                "scientific_contract": {
                    "status": "cpu_diagnostic_baseline_not_external_or_prospective_validation",
                    "target_scope": "confirmed wild-type AID720551 backbone",
                    "feature_scope": "molecule_only_morgan_fingerprint",
                    "selection_partition": "validation_only",
                    "locked_test_used_for_selection": False,
                    "superiority_established": False,
                    "imbalance_note": "balanced fitting distorts natural-prevalence calibration",
                },
                "configuration": {
                    "seed": SEED,
                    "fingerprint": {"type": "Morgan", "radius": RADIUS, "bits": N_BITS},
                    "model": "SGDClassifier log-loss, fixed settings",
                    "optimization": {
                        "iterations_run": int(model.n_iter_),
                        "maximum_iterations": MAX_ITER,
                        "converged_before_cap": bool(model.n_iter_ < MAX_ITER),
                    },
                    "threshold_rule": "maximize MCC on validation; smallest threshold breaks ties",
                    "selected_threshold": threshold,
                },
                "counts": {
                    split: {
                        "rows": int(np.sum(partitions == split)),
                        "positives": int(np.sum(y[partitions == split])),
                    }
                    for split in ("train", "validation", "test")
                },
                "metrics": metric_rows,
                "artifacts": artifacts,
            }
        )
        (temporary / MANIFEST_NAME).write_bytes(_canonical_json_bytes(manifest) + b"\n")
        if output_root.exists():
            shutil.rmtree(output_root)
        os.replace(temporary, output_root)
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def validate_herg_paper_baseline(output_root: Path) -> dict[str, Any]:
    """Validate output hashes, schemas, accounting, and scientific contract."""

    root = output_root.resolve()
    manifest = json.loads((root / MANIFEST_NAME).read_text(encoding="utf-8"))
    expected = manifest.pop("manifest_sha256", None)
    actual = hashlib.sha256(_canonical_json_bytes(manifest)).hexdigest()
    if expected != actual:
        raise HergPaperBaselineError("Manifest digest mismatch")
    manifest["manifest_sha256"] = expected
    for name in (METRICS_NAME, PREDICTIONS_NAME, MODEL_NAME):
        if _sha256_file(root / name) != manifest["artifacts"][name]["sha256"]:
            raise HergPaperBaselineError(f"Artifact digest mismatch: {name}")
    if pq.read_table(root / METRICS_NAME).schema != _METRICS_SCHEMA:
        raise HergPaperBaselineError("Metrics schema mismatch")
    predictions = pq.read_table(root / PREDICTIONS_NAME)
    if predictions.schema != _PREDICTION_SCHEMA:
        raise HergPaperBaselineError("Prediction schema mismatch")
    expected_rows = manifest["counts"]["validation"]["rows"] + manifest["counts"]["test"]["rows"]
    if predictions.num_rows != expected_rows:
        raise HergPaperBaselineError("Evaluation prediction row count mismatch")
    if manifest["scientific_contract"]["superiority_established"] is not False:
        raise HergPaperBaselineError("Diagnostic baseline must not assert superiority")
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    manifest = (
        validate_herg_paper_baseline(args.output_root)
        if args.validate_only
        else run_herg_paper_baseline(split_path=args.split, output_root=args.output_root)
    )
    print(json.dumps(manifest, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
