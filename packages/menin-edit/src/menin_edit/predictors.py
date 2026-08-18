"""Lazy, evidence-aware property predictors used by Menin-Edit.

This module adapts the validated models already present in the parent Menin
repository to one deliberately small interface.  It does not retrain models,
silently reinterpret endpoints, or present RDKit structural alerts as measured
toxicity.  Every returned estimate carries its artifact lineage, applicability
status, and the limitations needed by the optimization engine.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import threading
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import fields
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

# LightGBM, XGBoost, and NumPy wheels can bundle different OpenMP runtimes on
# macOS.  These guards must be set before importing numerical libraries or
# unpickling the existing mixed-family private ensemble.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import joblib
import numpy as np
import pandas as pd
from menin_discovery.chemistry import standardize_smiles
from menin_discovery.features import nearest_neighbor_tanimoto
from menin_discovery.herg_benchmark import _predict_model, calculate_feature_registry
from rdkit import Chem, rdBase
from rdkit.Chem import FilterCatalog

from .schemas import PropertyEstimate

_REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
_NATIVE_ISOLATED_FAMILIES = frozenset({"lightgbm", "xgboost"})


_NATIVE_PREDICTION_SCRIPT = r"""
import json
import joblib
import sys

from menin_discovery.herg_benchmark import calculate_feature_registry, _predict_model

payload = json.load(sys.stdin)
smiles = payload["smiles"]
feature_set = payload["feature_set"]
model = joblib.load(payload["model_path"])
matrices, _descriptors, _metadata = calculate_feature_registry(smiles)
features = smiles if feature_set == "smiles_tokens" else matrices[feature_set]
probability = _predict_model(model, features).tolist()
print("MENIN_EDIT_RESULT=" + json.dumps(probability, separators=(",", ":")))
"""


class ArtifactVerificationError(RuntimeError):
    """Raised when an artifact does not match its recorded provenance."""


@runtime_checkable
class Predictor(Protocol):
    """Common interface for all absolute-property predictors."""

    endpoint: str

    @property
    def model_version(self) -> str:
        """Stable version derived from the loaded artifact or method."""

    def predict(self, smiles: str) -> PropertyEstimate:
        """Predict one standardized molecule."""

    def predict_many(self, smiles: Sequence[str]) -> list[PropertyEstimate]:
        """Predict a sequence while preserving input order."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(parts: Sequence[str]) -> str:
    return hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()


def _load_local_joblib(path: Path) -> Any:
    """Load a controlled local artifact without NumPy legacy-shape noise."""

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Setting the shape on a NumPy array has been deprecated.*",
            category=DeprecationWarning,
            module=r"joblib\.numpy_pickle",
        )
        return joblib.load(path)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ArtifactVerificationError(f"Expected a JSON object in {path}")
    return payload


def _resolve(path: str | Path, root: Path) -> Path:
    candidate = Path(path).expanduser()
    return candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()


def _canonical_smiles(smiles: str) -> str:
    standardized = standardize_smiles(
        smiles,
        strip_salts=True,
        canonicalize_tautomer=False,
        require_rdkit=True,
    )
    if not standardized.structure_valid or not standardized.standardized_smiles:
        reason = standardized.structure_error or standardized.structure_standardization_status
        raise ValueError(f"Invalid molecular structure: {reason}")
    return str(standardized.standardized_smiles)


def _isolated_native_prediction(
    *, model_path: str, feature_set: str, smiles: Sequence[str], repository_root: Path
) -> np.ndarray:
    """Score native boosting artifacts in a clean OpenMP process.

    LightGBM/XGBoost and the already-loaded scikit-learn/RDKit stack can load
    incompatible OpenMP runtimes in one macOS process.  A fresh interpreter is
    a reliability boundary, not model parallelism; it receives only an artifact
    path, representation key, and canonical SMILES and returns probabilities.
    """

    environment = os.environ.copy()
    environment.update(
        {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        }
    )
    package_src = str(Path(__file__).resolve().parents[1])
    discovery_src = str((repository_root / "src").resolve())
    existing_pythonpath = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = os.pathsep.join(
        value for value in (package_src, discovery_src, existing_pythonpath) if value
    )
    payload = json.dumps(
        {
            "model_path": str(model_path),
            "feature_set": str(feature_set),
            "smiles": list(smiles),
        }
    )
    completed = subprocess.run(
        [sys.executable, "-c", _NATIVE_PREDICTION_SCRIPT],
        input=payload,
        text=True,
        capture_output=True,
        env=environment,
        timeout=180,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip().splitlines()[-1:] or ["no stderr"]
        raise RuntimeError(
            f"Isolated native hERG model prediction failed (exit {completed.returncode}): {detail[0]}"
        )
    marker = "MENIN_EDIT_RESULT="
    result_line = next(
        (line for line in reversed(completed.stdout.splitlines()) if line.startswith(marker)),
        None,
    )
    if result_line is None:
        raise RuntimeError("Isolated native hERG model returned no parseable result")
    values = json.loads(result_line[len(marker) :])
    return np.asarray(values, dtype=float)


def _estimate(
    *,
    endpoint: str,
    mean: float,
    lower: float,
    upper: float,
    inside_domain: bool,
    model_version: str,
    evidence_status: str,
    metadata: Mapping[str, Any],
) -> PropertyEstimate:
    """Construct an estimate across the schema transition that added metadata."""

    values: dict[str, Any] = {
        "endpoint": endpoint,
        "mean": float(mean),
        "lower": float(lower),
        "upper": float(upper),
        "inside_domain": bool(inside_domain),
        "model_version": str(model_version),
        "evidence_status": str(evidence_status),
    }
    # PropertyEstimate.metadata is part of the Menin-Edit contract.  This
    # conditional only keeps this module importable while parallel scaffolding
    # creates that field; it takes effect automatically as soon as it is present.
    if "metadata" in {field.name for field in fields(PropertyEstimate)}:
        values["metadata"] = dict(metadata)
    return PropertyEstimate(**values)


def _estimate_metadata(estimate: PropertyEstimate) -> dict[str, Any]:
    if hasattr(estimate, "to_dict"):
        payload = estimate.to_dict()
        if isinstance(payload, dict):
            return payload
    return {
        "endpoint": estimate.endpoint,
        "mean": estimate.mean,
        "lower": estimate.lower,
        "upper": estimate.upper,
        "inside_domain": estimate.inside_domain,
        "model_version": estimate.model_version,
        "evidence_status": estimate.evidence_status,
    }


class _SkopsAdapter:
    """Shared verified loading and applicability-domain behavior."""

    endpoint = ""
    default_artifact = Path()
    default_manifest = Path()
    default_metrics: Path | None = None
    default_domain_reference = Path()

    def __init__(
        self,
        *,
        artifact: str | Path | None = None,
        manifest: str | Path | None = None,
        metrics: str | Path | None = None,
        domain_reference: str | Path | None = None,
        domain_threshold: float | None = None,
        repository_root: str | Path | None = None,
        verify_hash: bool = True,
    ) -> None:
        self.repository_root = Path(repository_root or _REPOSITORY_ROOT).resolve()
        self.manifest_path = _resolve(manifest or self.default_manifest, self.repository_root)
        metrics_value = metrics if metrics is not None else self.default_metrics
        self.metrics_path = (
            _resolve(metrics_value, self.repository_root) if metrics_value is not None else None
        )
        self.domain_reference_path = _resolve(
            domain_reference or self.default_domain_reference, self.repository_root
        )
        self._explicit_artifact = artifact
        self._configured_domain_threshold = domain_threshold
        self.verify_hash = bool(verify_hash)
        self._model: Any | None = None
        self._manifest: dict[str, Any] | None = None
        self._metrics: dict[str, Any] = {}
        self._artifact_path: Path | None = None
        self._artifact_sha256 = ""
        self._untrusted_types_reviewed: tuple[str, ...] = ()
        self._reference_smiles: tuple[str, ...] = ()
        self._domain_threshold = float("nan")
        self._load_lock = threading.RLock()

    @property
    def model_version(self) -> str:
        self._load()
        return f"sha256:{self._artifact_sha256[:16]}"

    def _load(self) -> None:
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            if not self.manifest_path.exists():
                raise FileNotFoundError(f"Model manifest not found: {self.manifest_path}")
            manifest = _read_json(self.manifest_path)
            artifact_record = manifest.get("artifact")
            if not isinstance(artifact_record, Mapping):
                raise ArtifactVerificationError("Model manifest has no artifact record")
            artifact_value = self._explicit_artifact or artifact_record.get("path")
            if not artifact_value:
                raise ArtifactVerificationError("Model manifest has no artifact path")
            artifact_path = _resolve(artifact_value, self.repository_root)
            if not artifact_path.is_file():
                raise FileNotFoundError(f"Model artifact not found: {artifact_path}")
            actual_sha256 = _sha256_file(artifact_path)
            expected_sha256 = str(artifact_record.get("sha256", "")).strip().lower()
            if self.verify_hash:
                if not expected_sha256:
                    raise ArtifactVerificationError("Model manifest has no artifact SHA-256")
                if actual_sha256.lower() != expected_sha256:
                    raise ArtifactVerificationError(
                        f"Artifact SHA-256 mismatch for {artifact_path.name}: "
                        f"expected {expected_sha256}, observed {actual_sha256}"
                    )
            if str(artifact_record.get("format", "skops")).casefold() != "skops":
                raise ArtifactVerificationError("Public model adapter requires a skops artifact")

            import skops.io as sio

            reviewed_types = tuple(sorted(sio.get_untrusted_types(file=artifact_path)))
            # The artifact hash is verified before its project-defined transformer
            # is trusted.  Arbitrary external skops files must not be passed here.
            model = sio.load(artifact_path, trusted=list(reviewed_types))

            metrics_payload = (
                _read_json(self.metrics_path)
                if self.metrics_path is not None and self.metrics_path.is_file()
                else {}
            )
            reference = self._load_domain_reference(manifest)
            threshold = self._configured_domain_threshold
            if threshold is None:
                threshold = (
                    metrics_payload.get("applicability_domain", {}).get("similarity_threshold")
                    if isinstance(metrics_payload.get("applicability_domain"), Mapping)
                    else None
                )
            if threshold is None:
                raise ArtifactVerificationError(
                    "No applicability-domain threshold was configured or recorded"
                )
            threshold = float(threshold)
            if not 0 <= threshold <= 1:
                raise ArtifactVerificationError("Applicability-domain threshold must be in [0, 1]")

            self._manifest = manifest
            self._metrics = metrics_payload
            self._artifact_path = artifact_path
            self._artifact_sha256 = actual_sha256
            self._untrusted_types_reviewed = reviewed_types
            self._reference_smiles = reference
            self._domain_threshold = threshold
            self._model = model

    def _loaded_state(self) -> tuple[Any, dict[str, Any]]:
        """Return the verified lazy-load state with an explicit type contract."""

        self._load()
        if self._model is None or self._manifest is None:
            raise ArtifactVerificationError("Verified model loading did not initialize model state")
        return self._model, self._manifest

    def _load_domain_reference(self, manifest: Mapping[str, Any]) -> tuple[str, ...]:
        path = self.domain_reference_path
        if not path.is_file():
            raise FileNotFoundError(f"Applicability-domain reference not found: {path}")
        frame = pd.read_csv(path, low_memory=False)
        if "split" in frame.columns:
            frame = frame[frame["split"].fillna("").astype(str).str.casefold().eq("train")]
        column = next(
            (name for name in ("smiles", "standardized_smiles", "canonical_smiles") if name in frame.columns),
            None,
        )
        if column is None:
            raise ArtifactVerificationError(f"Applicability-domain reference {path} has no SMILES column")
        values = tuple(
            dict.fromkeys(frame[column].dropna().astype(str).str.strip().loc[lambda value: value.ne("")])
        )
        if not values:
            raise ArtifactVerificationError("Applicability-domain reference contains no structures")

        # Split-assignment tables carry the exact model dataset/split lineage.
        # The model manifest and split table intentionally hash different
        # column projections, so their dataset digests are not directly
        # comparable.  The shared split digest below has identical semantics.
        expected_split = str(manifest.get("split", {}).get("split_sha256", ""))
        if expected_split and "split_sha256" in frame.columns:
            observed = set(frame["split_sha256"].dropna().astype(str))
            if observed and observed != {expected_split}:
                raise ArtifactVerificationError("Domain reference split hash disagrees with manifest")
        return values

    def _domain(self, smiles: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        _model, manifest = self._loaded_state()
        similarities, _indices, _backend = nearest_neighbor_tanimoto(
            smiles,
            self._reference_smiles,
            backend="rdkit_morgan",
            n_bits=int(manifest.get("features", {}).get("fingerprint_bits", 2048)),
            radius=int(manifest.get("features", {}).get("morgan_radius", 2)),
        )
        return similarities, similarities >= self._domain_threshold

    def _base_metadata(self, similarity: float) -> dict[str, Any]:
        return {
            "artifact_path": str(self._artifact_path),
            "artifact_sha256": self._artifact_sha256,
            "manifest_path": str(self.manifest_path),
            "artifact_hash_verified": bool(self.verify_hash),
            "skops_types_reviewed": list(self._untrusted_types_reviewed),
            "nearest_reference_tanimoto": float(similarity),
            "domain_threshold": float(self._domain_threshold),
            "domain_reference_path": str(self.domain_reference_path),
        }


class PublicMeninSkopsPredictor(_SkopsAdapter):
    """Adapter for the public Menin biochemical pIC50 regression model."""

    endpoint = "menin_biochemical_pIC50"
    default_artifact = Path("research/models/menin_activity_ic50_biochemical_binding_extra_trees.skops")
    default_manifest = Path("research/models/menin_activity_ic50_biochemical_binding_manifest.json")
    default_metrics = Path("research/reports/menin_activity_ic50_biochemical_binding_model_metrics.json")
    default_domain_reference = Path(
        "research/reports/menin_activity_ic50_biochemical_binding_split_assignments.csv"
    )

    def __init__(self, *, interval_half_width: float | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._configured_interval_half_width = interval_half_width

    def predict(self, smiles: str) -> PropertyEstimate:
        return self.predict_many([smiles])[0]

    def predict_many(self, smiles: Sequence[str]) -> list[PropertyEstimate]:
        canonical = [_canonical_smiles(value) for value in smiles]
        if not canonical:
            return []
        model, _manifest = self._loaded_state()
        predictions = np.asarray(model.predict(pd.Series(canonical)), dtype=float)
        similarities, inside = self._domain(canonical)
        radius = self._configured_interval_half_width
        if radius is None:
            radius = self._metrics.get("uncertainty", {}).get("interval_half_width_p_activity")
        if radius is None:
            raise ArtifactVerificationError("Menin model has no prediction-interval half-width")
        radius = float(radius)
        if not math.isfinite(radius) or radius < 0:
            raise ArtifactVerificationError("Menin interval half-width must be finite and non-negative")
        estimates: list[PropertyEstimate] = []
        for prediction, similarity, is_inside in zip(predictions, similarities, inside, strict=True):
            metadata = self._base_metadata(float(similarity))
            metadata.update(
                {
                    "target": "Menin/MEN1",
                    "endpoint_semantics": "predicted biochemical-binding pIC50",
                    "assay_family": "biochemical_binding",
                    "interval_method": "training OOF absolute-residual conformal half-width",
                    "interval_half_width_pIC50": radius,
                    "prediction_is_not_measurement": True,
                }
            )
            estimates.append(
                _estimate(
                    endpoint=self.endpoint,
                    mean=float(prediction),
                    lower=float(prediction - radius),
                    upper=float(prediction + radius),
                    inside_domain=bool(is_inside),
                    model_version=self.model_version,
                    evidence_status=(
                        "model_prediction_in_domain" if is_inside else "model_prediction_outside_domain"
                    ),
                    metadata=metadata,
                )
            )
        return estimates


class PublicHergSkopsPredictor(_SkopsAdapter):
    """Adapter for the calibrated public primary hERG blocker classifier."""

    endpoint = "herg_public_blocker_probability"
    default_artifact = Path("research/models/herg_liability_extra_trees_calibrated.skops")
    default_manifest = Path("research/models/herg_classifier_manifest.json")
    default_metrics = Path("research/reports/herg_classifier_metrics.json")
    default_domain_reference = Path("research/reports/herg_classifier_split_assignments.csv")

    def predict(self, smiles: str) -> PropertyEstimate:
        return self.predict_many([smiles])[0]

    def predict_many(self, smiles: Sequence[str]) -> list[PropertyEstimate]:
        canonical = [_canonical_smiles(value) for value in smiles]
        if not canonical:
            return []
        model, manifest = self._loaded_state()
        probabilities = np.asarray(model.predict_proba(pd.Series(canonical))[:, 1], dtype=float)
        similarities, inside = self._domain(canonical)
        estimates: list[PropertyEstimate] = []
        for probability, similarity, is_inside in zip(probabilities, similarities, inside, strict=True):
            probability = float(np.clip(probability, 0.0, 1.0))
            metadata = self._base_metadata(float(similarity))
            metadata.update(
                {
                    "target": "hERG/KCNH2",
                    "endpoint_semantics": "probability of project-defined hERG blocker class",
                    "label_policy": manifest.get("task", {}).get("label_policy"),
                    "calibration": manifest.get("calibration", {}),
                    "interval_method": "none; lower and upper equal the calibrated point probability",
                    "not_clinical_cardiotoxicity_probability": True,
                    "prediction_is_not_measurement": True,
                }
            )
            estimates.append(
                _estimate(
                    endpoint=self.endpoint,
                    mean=probability,
                    lower=probability,
                    upper=probability,
                    inside_domain=bool(is_inside),
                    model_version=self.model_version,
                    evidence_status=(
                        "calibrated_model_prediction_in_domain"
                        if is_inside
                        else "calibrated_model_prediction_outside_domain"
                    ),
                    metadata=metadata,
                )
            )
        return estimates


class GovernedSkopsClassifierPredictor(_SkopsAdapter):
    """Generic hash-verified adapter for calibrated binary endpoint models.

    This is the plug-in boundary for governed DILI, Ames, or other explicit
    toxicity classifiers.  It does not supply a model or reinterpret a proxy
    endpoint as toxicity; enabling it requires an artifact, manifest,
    applicability reference, and threshold.
    """

    def __init__(
        self,
        *,
        endpoint: str,
        positive_class_index: int = 1,
        endpoint_semantics: str = "project-defined positive-class probability",
        **kwargs: Any,
    ) -> None:
        key = str(endpoint).strip()
        if not key:
            raise ValueError("Governed classifier endpoint must not be empty")
        self.endpoint = key
        self.positive_class_index = int(positive_class_index)
        if self.positive_class_index < 0:
            raise ValueError("positive_class_index must be non-negative")
        self.endpoint_semantics = str(endpoint_semantics).strip()
        super().__init__(**kwargs)

    def predict(self, smiles: str) -> PropertyEstimate:
        return self.predict_many([smiles])[0]

    def predict_many(self, smiles: Sequence[str]) -> list[PropertyEstimate]:
        canonical = [_canonical_smiles(value) for value in smiles]
        if not canonical:
            return []
        model, manifest = self._loaded_state()
        if not hasattr(model, "predict_proba"):
            raise ArtifactVerificationError("Governed classifier does not expose predict_proba")
        matrix = np.asarray(model.predict_proba(pd.Series(canonical)), dtype=float)
        if matrix.ndim != 2 or self.positive_class_index >= matrix.shape[1]:
            raise ArtifactVerificationError("Configured positive class is absent from model output")
        probabilities = np.clip(matrix[:, self.positive_class_index], 0.0, 1.0)
        similarities, inside = self._domain(canonical)
        estimates: list[PropertyEstimate] = []
        for probability, similarity, is_inside in zip(probabilities, similarities, inside, strict=True):
            metadata = self._base_metadata(float(similarity))
            metadata.update(
                {
                    "endpoint_semantics": self.endpoint_semantics,
                    "positive_class_index": self.positive_class_index,
                    "calibration": manifest.get("calibration", {}),
                    "interval_method": "none; point probability only",
                    "prediction_is_not_measurement": True,
                    "requires_endpoint_specific_validation": True,
                }
            )
            probability = float(probability)
            estimates.append(
                _estimate(
                    endpoint=self.endpoint,
                    mean=probability,
                    lower=probability,
                    upper=probability,
                    inside_domain=bool(is_inside),
                    model_version=self.model_version,
                    evidence_status=(
                        "governed_classifier_in_domain" if is_inside else "governed_classifier_outside_domain"
                    ),
                    metadata=metadata,
                )
            )
        return estimates


class PrivateQuickHergEnsemblePredictor:
    """Adapter for one private/public quick-benchmark production ensemble.

    The quick benchmark is a development interpolation screen.  Its ensemble
    spread is exposed as model disagreement, not a statistical confidence
    interval, and its private-chemistry applicability rule is recalculated from
    the saved private descriptor table.
    """

    endpoint = "herg_private_ensemble_probability"

    def __init__(
        self,
        *,
        benchmark_root: str | Path = "research/benchmarks/herg/quick",
        regime: str = "equal_importance",
        domain_reference: str | Path | None = None,
        domain_threshold: float | None = None,
        domain_quantile: float = 0.05,
        repository_root: str | Path | None = None,
    ) -> None:
        self.repository_root = Path(repository_root or _REPOSITORY_ROOT).resolve()
        self.benchmark_root = _resolve(benchmark_root, self.repository_root)
        self.regime = str(regime)
        self.domain_reference_path = _resolve(
            domain_reference or (self.benchmark_root / "calculated_molecular_parameters.csv"),
            self.repository_root,
        )
        self._configured_domain_threshold = domain_threshold
        self.domain_quantile = float(domain_quantile)
        if not 0 <= self.domain_quantile <= 1:
            raise ValueError("domain_quantile must lie in [0, 1]")
        self._members: list[tuple[dict[str, Any], Any]] | None = None
        self._manifest: dict[str, Any] = {}
        self._model_hashes: tuple[str, ...] = ()
        self._version = ""
        self._reference_smiles: tuple[str, ...] = ()
        self._domain_threshold = float("nan")
        self._load_lock = threading.RLock()

    @property
    def model_version(self) -> str:
        self._load()
        return self._version

    def _load(self) -> None:
        if self._members is not None:
            return
        with self._load_lock:
            if self._members is not None:
                return
            manifest_path = self.benchmark_root / "run_manifest.json"
            best_path = self.benchmark_root / "best_models.csv"
            if not manifest_path.is_file() or not best_path.is_file():
                raise FileNotFoundError(
                    f"Private quick hERG benchmark is incomplete under {self.benchmark_root}"
                )
            manifest = _read_json(manifest_path)
            if manifest.get("status") != "complete":
                raise ArtifactVerificationError("Private quick hERG benchmark is not complete")
            declared_outputs = set(manifest.get("output_files", []))
            if declared_outputs and "best_models.csv" not in declared_outputs:
                raise ArtifactVerificationError("Benchmark manifest does not declare best_models.csv")
            best = pd.read_csv(best_path)
            required = {"regime", "ensemble_rank", "model_key", "family", "feature_set"}
            missing = sorted(required - set(best.columns))
            if missing:
                raise ArtifactVerificationError(f"best_models.csv is missing columns: {missing}")
            selected = best[best["regime"].astype(str).eq(self.regime)].sort_values("ensemble_rank")
            if selected.empty:
                raise ArtifactVerificationError(
                    f"No private hERG ensemble members found for regime {self.regime!r}"
                )
            members: list[tuple[dict[str, Any], Any]] = []
            model_hashes: list[str] = []
            for row in selected.to_dict(orient="records"):
                rank = int(row["ensemble_rank"])
                model_path = (
                    self.benchmark_root / "models" / f"{self.regime}__rank{rank}__{row['model_key']}.joblib"
                )
                if not model_path.is_file():
                    raise FileNotFoundError(f"Private hERG ensemble artifact missing: {model_path}")
                digest = _sha256_file(model_path)
                # joblib is executable/trust-sensitive.  Only locally generated,
                # access-controlled benchmark artifacts are loaded here. Native
                # boosting models are loaded later in an isolated OpenMP process.
                family = str(row["family"]).casefold()
                fitted = None if family in _NATIVE_ISOLATED_FAMILIES else _load_local_joblib(model_path)
                row["artifact_path"] = str(model_path)
                row["artifact_sha256"] = digest
                members.append((row, fitted))
                model_hashes.append(digest)

            reference_frame = pd.read_csv(self.domain_reference_path, low_memory=False)
            reference_column = next(
                (
                    column
                    for column in ("smiles", "standardized_smiles", "Kekule Canonical SMILES")
                    if column in reference_frame.columns
                ),
                None,
            )
            if reference_column is None:
                raise ArtifactVerificationError("Private domain reference has no SMILES column")
            reference = tuple(
                dict.fromkeys(
                    reference_frame[reference_column]
                    .dropna()
                    .astype(str)
                    .str.strip()
                    .loc[lambda value: value.ne("")]
                )
            )
            if not reference:
                raise ArtifactVerificationError("Private domain reference has no valid structures")
            threshold = self._configured_domain_threshold
            if threshold is None:
                if len(reference) < 2:
                    raise ArtifactVerificationError(
                        "At least two private reference structures are needed to derive a domain"
                    )
                self_similarity, _indices, _backend = nearest_neighbor_tanimoto(
                    reference,
                    reference,
                    backend="rdkit_morgan",
                    n_bits=2048,
                    radius=2,
                    exclude_identical_positions=True,
                )
                threshold = float(np.quantile(self_similarity, self.domain_quantile))
            threshold = float(threshold)
            if not 0 <= threshold <= 1:
                raise ArtifactVerificationError("Private domain threshold must be in [0, 1]")

            version_digest = _sha256_text(
                [
                    _sha256_file(manifest_path),
                    _sha256_file(best_path),
                    self.regime,
                    *model_hashes,
                ]
            )
            self._manifest = manifest
            self._model_hashes = tuple(model_hashes)
            self._reference_smiles = reference
            self._domain_threshold = threshold
            self._version = f"quick-{self.regime}-sha256:{version_digest[:16]}"
            self._members = members

    def predict(self, smiles: str) -> PropertyEstimate:
        return self.predict_many([smiles])[0]

    def predict_many(self, smiles: Sequence[str]) -> list[PropertyEstimate]:
        canonical = [_canonical_smiles(value) for value in smiles]
        if not canonical:
            return []
        self._load()
        matrices, _descriptors, _feature_metadata = calculate_feature_registry(canonical)
        member_probabilities: list[np.ndarray] = []
        member_records: list[dict[str, Any]] = []
        for row, fitted in self._members or []:
            feature_set = str(row["feature_set"])
            if str(row["family"]).casefold() in _NATIVE_ISOLATED_FAMILIES:
                probability = _isolated_native_prediction(
                    model_path=str(row["artifact_path"]),
                    feature_set=feature_set,
                    smiles=canonical,
                    repository_root=self.repository_root,
                )
            else:
                features: Any = canonical if feature_set == "smiles_tokens" else matrices[feature_set]
                probability = np.asarray(_predict_model(fitted, features), dtype=float)
            member_probabilities.append(probability)
            member_records.append(
                {
                    "rank": int(row["ensemble_rank"]),
                    "family": str(row["family"]),
                    "feature_set": feature_set,
                    "model_key": str(row["model_key"]),
                    "artifact_sha256": str(row["artifact_sha256"]),
                }
            )
        matrix = np.vstack(member_probabilities)
        means = np.clip(matrix.mean(axis=0), 0.0, 1.0)
        stds = matrix.std(axis=0, ddof=0)
        similarities, _indices, _backend = nearest_neighbor_tanimoto(
            canonical,
            self._reference_smiles,
            backend="rdkit_morgan",
            n_bits=2048,
            radius=2,
        )
        inside = similarities >= self._domain_threshold
        estimates: list[PropertyEstimate] = []
        for index, (mean, std, similarity, is_inside) in enumerate(
            zip(means, stds, similarities, inside, strict=True)
        ):
            per_model = [float(value) for value in matrix[:, index]]
            metadata = {
                "benchmark_root": str(self.benchmark_root),
                "regime": self.regime,
                "members": member_records,
                "member_probabilities": per_model,
                "ensemble_std": float(std),
                "interval_method": "ensemble mean plus/minus one member standard deviation",
                "interval_is_not_calibrated_confidence_interval": True,
                "nearest_private_reference_tanimoto": float(similarity),
                "domain_threshold": float(self._domain_threshold),
                "domain_policy": "private-reference leave-one-out nearest-neighbor quantile",
                "domain_quantile": self.domain_quantile,
                "domain_reference_path": str(self.domain_reference_path),
                "benchmark_status_verified": True,
                "joblib_trust_boundary": (
                    "load only locally generated, access-controlled benchmark artifacts"
                ),
                "quick_benchmark_is_development_evidence": True,
                "not_clinical_cardiotoxicity_probability": True,
                "prediction_is_not_measurement": True,
            }
            estimates.append(
                _estimate(
                    endpoint=self.endpoint,
                    mean=float(mean),
                    lower=float(max(0.0, mean - std)),
                    upper=float(min(1.0, mean + std)),
                    inside_domain=bool(is_inside),
                    model_version=self.model_version,
                    evidence_status=(
                        "private_quick_ensemble_in_domain"
                        if is_inside
                        else "private_quick_ensemble_outside_domain"
                    ),
                    metadata=metadata,
                )
            )
        return estimates


class ConservativeHergConsensusPredictor:
    """Worst-case consensus across public and private hERG predictors."""

    endpoint = "herg_consensus_probability"

    def __init__(self, predictors: Sequence[Predictor]) -> None:
        self.predictors = tuple(predictors)
        if not self.predictors:
            raise ValueError("At least one hERG predictor is required")

    @property
    def model_version(self) -> str:
        versions = [f"{predictor.endpoint}:{predictor.model_version}" for predictor in self.predictors]
        return f"conservative-max-sha256:{_sha256_text(versions)[:16]}"

    def predict(self, smiles: str) -> PropertyEstimate:
        return self.predict_many([smiles])[0]

    def predict_many(self, smiles: Sequence[str]) -> list[PropertyEstimate]:
        if not smiles:
            return []
        component_batches = [predictor.predict_many(smiles) for predictor in self.predictors]
        if any(len(batch) != len(smiles) for batch in component_batches):
            raise RuntimeError("A consensus member returned the wrong number of estimates")
        output: list[PropertyEstimate] = []
        for index in range(len(smiles)):
            components = [batch[index] for batch in component_batches]
            means = [float(component.mean) for component in components]
            mean = max(means)
            lower = min(float(component.lower) for component in components)
            upper = max(float(component.upper) for component in components)
            all_inside = all(bool(component.inside_domain) for component in components)
            metadata = {
                "consensus_rule": "maximum blocker probability across members",
                "interval_rule": "envelope across component estimates",
                "component_estimates": [_estimate_metadata(component) for component in components],
                "component_probability_range": float(max(means) - min(means)),
                "all_components_inside_domain": all_inside,
                "conservative_not_calibrated_joint_probability": True,
                "not_clinical_cardiotoxicity_probability": True,
            }
            output.append(
                _estimate(
                    endpoint=self.endpoint,
                    mean=mean,
                    lower=min(lower, mean),
                    upper=max(upper, mean),
                    inside_domain=all_inside,
                    model_version=self.model_version,
                    evidence_status=(
                        "conservative_consensus_all_in_domain"
                        if all_inside
                        else "conservative_consensus_contains_out_of_domain_evidence"
                    ),
                    metadata=metadata,
                )
            )
        return output


class StructuralAlertProxyPredictor:
    """Count RDKit alert matches as a review proxy, never as toxicity."""

    endpoint = "structural_alert_count"

    def __init__(self, *, catalogs: Sequence[str] = ("PAINS", "BRENK", "NIH")) -> None:
        normalized = tuple(str(value).strip().upper() for value in catalogs)
        if not normalized or any(not value for value in normalized):
            raise ValueError("At least one non-empty structural-alert catalog is required")
        self.catalog_names = normalized
        self._catalogs: dict[str, Any] | None = None
        self._load_lock = threading.RLock()

    @property
    def model_version(self) -> str:
        return f"rdkit-{rdBase.rdkitVersion}-catalogs:{'-'.join(self.catalog_names)}"

    def _load(self) -> None:
        if self._catalogs is not None:
            return
        with self._load_lock:
            if self._catalogs is not None:
                return
            catalogs: dict[str, Any] = {}
            for name in self.catalog_names:
                enum = getattr(FilterCatalog.FilterCatalogParams.FilterCatalogs, name, None)
                if enum is None:
                    raise ValueError(f"Unsupported RDKit structural-alert catalog: {name}")
                parameters = FilterCatalog.FilterCatalogParams()
                parameters.AddCatalog(enum)
                catalogs[name] = FilterCatalog.FilterCatalog(parameters)
            self._catalogs = catalogs

    def predict(self, smiles: str) -> PropertyEstimate:
        return self.predict_many([smiles])[0]

    def predict_many(self, smiles: Sequence[str]) -> list[PropertyEstimate]:
        canonical = [_canonical_smiles(value) for value in smiles]
        if not canonical:
            return []
        self._load()
        output: list[PropertyEstimate] = []
        for value in canonical:
            molecule = Chem.MolFromSmiles(value)
            if molecule is None:  # pragma: no cover - canonicalization already validates.
                raise ValueError("RDKit could not parse standardized SMILES")
            matches_by_catalog: dict[str, list[str]] = {}
            total_count = 0
            for name, catalog in (self._catalogs or {}).items():
                descriptions = sorted({str(match.GetDescription()) for match in catalog.GetMatches(molecule)})
                matches_by_catalog[name] = descriptions
                total_count += len(descriptions)
            count = float(total_count)
            metadata = {
                "canonical_smiles": value,
                "catalogs": list(self.catalog_names),
                "matches_by_catalog": matches_by_catalog,
                "proxy_only": True,
                "not_a_toxicity_prediction": True,
                "not_an_assay_interference_determination": True,
                "interpretation": (
                    "RDKit catalog matches are review flags. A match is not evidence of toxicity, "
                    "and no matches are not evidence of safety."
                ),
            }
            output.append(
                _estimate(
                    endpoint=self.endpoint,
                    mean=count,
                    lower=count,
                    upper=count,
                    # This deterministic rule engine is applicable to every
                    # valid RDKit molecule; "in domain" does not promote it
                    # from a review proxy to a toxicity model.
                    inside_domain=True,
                    model_version=self.model_version,
                    evidence_status="heuristic_structural_alert_proxy_not_toxicity",
                    metadata=metadata,
                )
            )
        return output


# Short aliases keep configuration/factory code readable without creating a
# second implementation surface.
PublicMeninPredictor = PublicMeninSkopsPredictor
PublicHergPredictor = PublicHergSkopsPredictor
PrivateQuickHergPredictor = PrivateQuickHergEnsemblePredictor
ConservativeHergPredictor = ConservativeHergConsensusPredictor
RDKitStructuralAlertPredictor = StructuralAlertProxyPredictor


__all__ = [
    "ArtifactVerificationError",
    "ConservativeHergConsensusPredictor",
    "ConservativeHergPredictor",
    "GovernedSkopsClassifierPredictor",
    "Predictor",
    "PrivateQuickHergEnsemblePredictor",
    "PrivateQuickHergPredictor",
    "PublicHergPredictor",
    "PublicHergSkopsPredictor",
    "PublicMeninPredictor",
    "PublicMeninSkopsPredictor",
    "RDKitStructuralAlertPredictor",
    "StructuralAlertProxyPredictor",
]
