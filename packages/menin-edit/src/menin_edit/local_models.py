"""Confidential historical-lab regression models for Menin-Edit.

This module turns the governed long-form tables from :mod:`menin_edit.data`
into small, endpoint-specific development models.  It is intentionally strict:
only exact, finite, non-conflicted ``train``/``development`` observations are
eligible; scaffold groups never cross cross-validation folds; and an endpoint
with too little usable evidence does not produce an artifact.

Artifacts are controlled local ``joblib`` files.  They contain the fitted
pipeline and canonical reference structures required for applicability-domain
checks, but never source compound IDs, pseudonymous IDs, observation IDs, or
raw workbook rows.  A hash-verified JSON manifest records validation,
calibration, and data-lineage summaries without exposing structures.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, TypedDict

import joblib
import numpy as np
import pandas as pd
import sklearn
from menin_discovery.chemistry import standardize_smiles
from menin_discovery.features import (
    SmilesFeatureTransformer,
    nearest_neighbor_tanimoto,
    scaffold_key,
)
from rdkit import rdBase
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

from .schemas import PropertyEstimate

_ALLOWED_SPLIT_ROLES = frozenset({"train", "development"})
_ARTIFACT_SCHEMA_VERSION = "menin-edit-local-regression-v1"


class LocalModelError(RuntimeError):
    """Base error for governed local-model training and loading."""


class InsufficientTrainingDataError(LocalModelError):
    """Raised when an endpoint cannot support honest model development."""


class UnsafeArtifactPathError(LocalModelError):
    """Raised when confidential artifacts would be written to an exposed path."""


class LocalArtifactVerificationError(LocalModelError):
    """Raised when a local model and its manifest do not agree."""


@dataclass(frozen=True)
class LocalRegressionConfig:
    """Small, explicit training policy for one continuous endpoint."""

    min_samples: int = 20
    min_unique_scaffolds: int = 3
    min_calibration_residuals: int = 12
    cv_folds: int = 5
    coverage: float = 0.90
    domain_quantile: float = 0.05
    min_domain_similarity: float = 0.20
    duplicate_conflict_tolerance: float = 0.50
    fingerprint_bits: int = 2048
    morgan_radius: int = 2
    n_estimators: int = 300
    min_samples_leaf: int = 2
    max_features: str | float = "sqrt"
    random_state: int = 13
    min_baseline_mae_improvement_fraction: float = 0.05
    min_oof_r2: float = 0.05

    def __post_init__(self) -> None:
        if self.min_samples < 8:
            raise ValueError("min_samples must be at least 8")
        if self.min_unique_scaffolds < 3:
            raise ValueError("min_unique_scaffolds must be at least 3")
        if self.min_calibration_residuals < 5:
            raise ValueError("min_calibration_residuals must be at least 5")
        if self.cv_folds < 2:
            raise ValueError("cv_folds must be at least 2")
        if not 0 < self.coverage < 1:
            raise ValueError("coverage must lie strictly between zero and one")
        if not 0 <= self.domain_quantile < 0.5:
            raise ValueError("domain_quantile must lie in [0, 0.5)")
        if not 0 <= self.min_domain_similarity <= 1:
            raise ValueError("min_domain_similarity must lie in [0, 1]")
        if self.duplicate_conflict_tolerance < 0:
            raise ValueError("duplicate_conflict_tolerance must be non-negative")
        if self.fingerprint_bits < 64:
            raise ValueError("fingerprint_bits must be at least 64")
        if self.morgan_radius <= 0:
            raise ValueError("morgan_radius must be positive")
        if self.n_estimators < 10:
            raise ValueError("n_estimators must be at least 10")
        if self.min_samples_leaf <= 0:
            raise ValueError("min_samples_leaf must be positive")
        if not 0 <= self.min_baseline_mae_improvement_fraction < 1:
            raise ValueError("min_baseline_mae_improvement_fraction must lie in [0, 1)")
        if not -1 <= self.min_oof_r2 <= 1:
            raise ValueError("min_oof_r2 must lie in [-1, 1]")


@dataclass(frozen=True)
class LocalRegressionArtifact:
    """Paths and public-safe metadata returned after successful training."""

    artifact_path: Path
    manifest_path: Path
    endpoint: str
    model_version: str
    manifest: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        object.__setattr__(self, "artifact_path", Path(self.artifact_path).resolve())
        object.__setattr__(self, "manifest_path", Path(self.manifest_path).resolve())
        object.__setattr__(self, "manifest", MappingProxyType(dict(self.manifest)))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_controlled_joblib(path: Path) -> Any:
    """Load a hash-verified local artifact without NumPy legacy-shape noise."""

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Setting the shape on a NumPy array has been deprecated.*",
            category=DeprecationWarning,
            module=r"joblib\.numpy_pickle",
        )
        return joblib.load(path)


def _canonical_smiles(smiles: object) -> str:
    standardized = standardize_smiles(
        "" if smiles is None else str(smiles),
        strip_salts=True,
        canonicalize_tautomer=False,
        require_rdkit=True,
    )
    if not standardized.structure_valid or not standardized.standardized_smiles:
        reason = standardized.structure_error or standardized.structure_standardization_status
        raise ValueError(f"Invalid molecular structure: {reason}")
    return str(standardized.standardized_smiles)


def _safe_endpoint_slug(endpoint: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "_", endpoint.strip()).strip("._")
    if not slug:
        raise ValueError("endpoint must not be empty")
    return slug


def _is_private_named_path(path: Path) -> bool:
    lowered = [part.casefold() for part in path.parts]
    # Limit the name-based allowance to the destination's nearby ancestors.
    # On macOS every temporary path begins with the system mount ``/private``;
    # that root is not evidence that a caller selected a protected directory.
    nearby = lowered[-4:]
    if any(part == "private" or part.startswith("private-") for part in nearby):
        return True
    return any(
        lowered[index] == "artifacts" and lowered[index + 1] in {"local", "models"}
        for index in range(len(lowered) - 1)
    )


def _is_git_ignored(path: Path) -> bool:
    """Best-effort Git ignore check; failure is treated as not ignored."""

    for parent in (path, *path.parents):
        if not (parent / ".git").exists():
            continue
        try:
            result = subprocess.run(
                ["git", "check-ignore", "--quiet", "--", str(path)],
                cwd=parent,
                capture_output=True,
                check=False,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            return False
        return result.returncode == 0
    return False


def _assert_private_output_path(path: Path) -> None:
    if _is_private_named_path(path) or _is_git_ignored(path):
        return
    raise UnsafeArtifactPathError(
        "Refusing to write a confidential local model outside a path named "
        "'private' or covered by Git ignore rules"
    )


def _required_columns(frame: pd.DataFrame, required: set[str], *, label: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"{label} table is missing required columns: {missing}")


def _prepare_endpoint_table(
    compounds: pd.DataFrame,
    observations: pd.DataFrame,
    *,
    endpoint: str,
    config: LocalRegressionConfig,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Return one anonymous, exact label per standardized structure."""

    _required_columns(compounds, {"structure_id", "smiles"}, label="Compound")
    _required_columns(
        observations,
        {
            "structure_id",
            "endpoint",
            "model_value",
            "is_exact",
            "label_conflict",
            "split_role",
        },
        label="Observation",
    )
    endpoint = endpoint.strip()
    if not endpoint:
        raise ValueError("endpoint must not be empty")

    endpoint_rows = observations[observations["endpoint"].astype(str).eq(endpoint)].copy()
    audit = {
        "endpoint_observations": int(len(endpoint_rows)),
        "excluded_non_development_role": 0,
        "excluded_non_exact_or_censored": 0,
        "excluded_label_conflict": 0,
        "excluded_nonfinite": 0,
        "excluded_missing_structure": 0,
        "excluded_split_conflict": 0,
        "excluded_duplicate_label_conflict": 0,
    }
    if endpoint_rows.empty:
        raise InsufficientTrainingDataError(f"No observations are available for endpoint {endpoint!r}")

    roles = endpoint_rows["split_role"].fillna("").astype(str).str.casefold()
    role_mask = roles.isin(_ALLOWED_SPLIT_ROLES)
    audit["excluded_non_development_role"] = int((~role_mask).sum())
    endpoint_rows = endpoint_rows.loc[role_mask].copy()
    endpoint_rows["split_role"] = roles.loc[role_mask]

    exact_mask = endpoint_rows["is_exact"].fillna(False).astype(bool)
    if "is_censored" in endpoint_rows:
        exact_mask &= ~endpoint_rows["is_censored"].fillna(False).astype(bool)
    audit["excluded_non_exact_or_censored"] = int((~exact_mask).sum())
    endpoint_rows = endpoint_rows.loc[exact_mask].copy()

    conflict_mask = endpoint_rows["label_conflict"].fillna(True).astype(bool)
    if "provenance_conflict" in endpoint_rows:
        conflict_mask |= endpoint_rows["provenance_conflict"].fillna(False).astype(bool)
    audit["excluded_label_conflict"] = int(conflict_mask.sum())
    endpoint_rows = endpoint_rows.loc[~conflict_mask].copy()

    endpoint_rows["target"] = pd.to_numeric(endpoint_rows["model_value"], errors="coerce")
    finite_mask = np.isfinite(endpoint_rows["target"].to_numpy(dtype=float))
    audit["excluded_nonfinite"] = int((~finite_mask).sum())
    endpoint_rows = endpoint_rows.loc[finite_mask].copy()

    structure_map = compounds[["structure_id", "smiles"]].dropna(subset=["structure_id", "smiles"]).copy()
    structure_map["structure_id"] = structure_map["structure_id"].astype(str)
    ambiguous = structure_map.groupby("structure_id")["smiles"].nunique().loc[lambda value: value > 1]
    if not ambiguous.empty:
        raise LocalModelError("Compound table maps one structure_id to multiple SMILES")
    structure_map = structure_map.drop_duplicates("structure_id", keep="first")
    endpoint_rows["structure_id"] = endpoint_rows["structure_id"].astype(str)
    merged = endpoint_rows.merge(structure_map, on="structure_id", how="left", validate="many_to_one")
    missing_structure = merged["smiles"].isna() | merged["smiles"].astype(str).str.strip().eq("")
    audit["excluded_missing_structure"] = int(missing_structure.sum())
    merged = merged.loc[~missing_structure, ["smiles", "target", "split_role"]].copy()
    merged["smiles"] = merged["smiles"].map(_canonical_smiles)

    records: list[dict[str, Any]] = []
    for smiles, group in merged.groupby("smiles", sort=True):
        roles_for_structure = set(group["split_role"].astype(str))
        if len(roles_for_structure) != 1:
            audit["excluded_split_conflict"] += 1
            continue
        values = group["target"].to_numpy(dtype=float)
        if float(np.max(values) - np.min(values)) > config.duplicate_conflict_tolerance:
            audit["excluded_duplicate_label_conflict"] += 1
            continue
        records.append(
            {
                "smiles": smiles,
                "target": float(np.median(values)),
                "split_role": next(iter(roles_for_structure)),
            }
        )

    prepared = pd.DataFrame(records, columns=["smiles", "target", "split_role"])
    if not prepared.empty:
        scaffold_values = prepared["smiles"].map(scaffold_key)
        prepared["scaffold"] = scaffold_values.map(lambda value: value[0])
        prepared["scaffold_method"] = scaffold_values.map(lambda value: value[1])
        prepared = prepared.sort_values("smiles", kind="stable", ignore_index=True)
    audit["eligible_unique_structures"] = int(len(prepared))
    return prepared, audit


def _pipeline(config: LocalRegressionConfig) -> Pipeline:
    return Pipeline(
        [
            (
                "features",
                SmilesFeatureTransformer(
                    backend="rdkit",
                    n_features=config.fingerprint_bits,
                    radius=config.morgan_radius,
                    include_descriptors=True,
                    scale_descriptors=True,
                ),
            ),
            (
                "regressor",
                ExtraTreesRegressor(
                    n_estimators=config.n_estimators,
                    min_samples_leaf=config.min_samples_leaf,
                    max_features=config.max_features,
                    n_jobs=1,
                    random_state=config.random_state,
                ),
            ),
        ]
    )


def _finite_sample_conformal_radius(residuals: np.ndarray, coverage: float) -> float:
    residuals = np.asarray(residuals, dtype=float)
    residuals = residuals[np.isfinite(residuals)]
    if not len(residuals):
        raise InsufficientTrainingDataError("No finite out-of-fold residuals are available")
    quantile = min(1.0, math.ceil((len(residuals) + 1) * coverage) / len(residuals))
    return float(np.quantile(residuals, quantile, method="higher"))


class _RegressionMetrics(TypedDict):
    mae: float
    rmse: float
    r2: float | None


def _regression_metrics(y_true: np.ndarray, prediction: np.ndarray) -> _RegressionMetrics:
    r2 = (
        float(r2_score(y_true, prediction))
        if len(y_true) >= 2 and float(np.ptp(np.asarray(y_true, dtype=float))) > 1e-12
        else None
    )
    return {
        "mae": float(mean_absolute_error(y_true, prediction)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, prediction))),
        # JSON has no portable NaN representation.  A constant held-out target
        # makes R² undefined, which is represented honestly as null.
        "r2": r2 if r2 is None or math.isfinite(r2) else None,
    }


def _dataset_digest(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    for row in frame.sort_values("smiles", kind="stable").itertuples(index=False):
        digest.update(f"{row.smiles}\t{float(row.target):.12g}\t{row.split_role}\n".encode())
    return digest.hexdigest()


def _atomic_joblib_dump(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        joblib.dump(payload, temporary, compress=3)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json_dump(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def train_local_regression(
    compounds: pd.DataFrame,
    observations: pd.DataFrame,
    *,
    endpoint: str,
    output_dir: str | Path,
    config: LocalRegressionConfig | None = None,
    overwrite: bool = False,
) -> LocalRegressionArtifact:
    """Train and persist one private endpoint model with audited uncertainty.

    The final model uses all eligible ``train`` and ``development`` structures.
    Its reported performance and conformal-style interval come only from
    scaffold-grouped out-of-fold predictions.  If explicit ``train`` and
    ``development`` roles both exist, an additional fixed development-holdout
    result is recorded before the final refit; it is not called external test
    evidence because those rows subsequently enter the final model.
    """

    policy = config or LocalRegressionConfig()
    endpoint = endpoint.strip()
    slug = _safe_endpoint_slug(endpoint)
    destination = Path(output_dir).expanduser().resolve()
    _assert_private_output_path(destination)
    artifact_path = destination / f"{slug}.joblib"
    manifest_path = destination / f"{slug}.manifest.json"
    if not overwrite and (artifact_path.exists() or manifest_path.exists()):
        raise FileExistsError(f"Local model output already exists for {endpoint!r}")

    data, exclusions = _prepare_endpoint_table(
        compounds,
        observations,
        endpoint=endpoint,
        config=policy,
    )
    n_samples = len(data)
    if n_samples < policy.min_samples:
        raise InsufficientTrainingDataError(
            f"Endpoint {endpoint!r} has {n_samples} eligible unique structures; "
            f"at least {policy.min_samples} are required"
        )
    n_scaffolds = int(data["scaffold"].nunique())
    if n_scaffolds < policy.min_unique_scaffolds:
        raise InsufficientTrainingDataError(
            f"Endpoint {endpoint!r} has {n_scaffolds} unique scaffold groups; "
            f"at least {policy.min_unique_scaffolds} are required"
        )
    target = data["target"].to_numpy(dtype=float)
    if not math.isfinite(float(np.std(target))) or float(np.std(target)) <= 1e-8:
        raise InsufficientTrainingDataError("Endpoint target has no usable variance")

    fold_count = min(policy.cv_folds, n_scaffolds)
    splitter = GroupKFold(n_splits=fold_count)
    folds = list(splitter.split(data["smiles"], target, groups=data["scaffold"]))
    template = _pipeline(policy)
    oof_prediction = np.full(n_samples, np.nan, dtype=float)
    baseline_prediction = np.full(n_samples, np.nan, dtype=float)
    fold_records: list[dict[str, int]] = []
    for fold_number, (train_indices, validation_indices) in enumerate(folds, start=1):
        if not len(train_indices) or not len(validation_indices):
            raise InsufficientTrainingDataError("Scaffold cross-validation produced an empty fold")
        fitted = clone(template).fit(data.iloc[train_indices]["smiles"], target[train_indices])
        oof_prediction[validation_indices] = fitted.predict(data.iloc[validation_indices]["smiles"])
        baseline_prediction[validation_indices] = float(np.median(target[train_indices]))
        fold_records.append(
            {
                "fold": fold_number,
                "train_structures": int(len(train_indices)),
                "validation_structures": int(len(validation_indices)),
                "train_scaffolds": int(data.iloc[train_indices]["scaffold"].nunique()),
                "validation_scaffolds": int(data.iloc[validation_indices]["scaffold"].nunique()),
            }
        )
    if not np.isfinite(oof_prediction).all():
        raise InsufficientTrainingDataError("Scaffold cross-validation did not score every structure")

    residuals = np.abs(target - oof_prediction)
    if len(residuals) < policy.min_calibration_residuals:
        raise InsufficientTrainingDataError(
            f"Only {len(residuals)} out-of-fold residuals are available; "
            f"at least {policy.min_calibration_residuals} are required"
        )
    interval_radius = _finite_sample_conformal_radius(residuals, policy.coverage)
    oof_metrics = _regression_metrics(target, oof_prediction)
    baseline_metrics = _regression_metrics(target, baseline_prediction)
    baseline_mae = float(baseline_metrics["mae"])
    improvement_fraction = (
        float((baseline_mae - float(oof_metrics["mae"])) / baseline_mae) if baseline_mae > 0 else 0.0
    )
    oof_r2 = oof_metrics["r2"]
    beats_baseline = bool(
        improvement_fraction >= policy.min_baseline_mae_improvement_fraction
        and oof_r2 is not None
        and float(oof_r2) >= policy.min_oof_r2
    )

    holdout: dict[str, Any] | None = None
    train_mask = data["split_role"].eq("train").to_numpy()
    development_mask = data["split_role"].eq("development").to_numpy()
    if int(train_mask.sum()) >= 8 and int(development_mask.sum()) >= 3:
        holdout_model = clone(template).fit(data.loc[train_mask, "smiles"], target[train_mask])
        holdout_prediction = holdout_model.predict(data.loc[development_mask, "smiles"])
        train_scaffolds = set(data.loc[train_mask, "scaffold"].astype(str))
        development_scaffolds = set(data.loc[development_mask, "scaffold"].astype(str))
        holdout = {
            "status": "development_role_holdout_before_final_refit",
            "train_structures": int(train_mask.sum()),
            "development_structures": int(development_mask.sum()),
            "development_scaffolds": int(len(development_scaffolds)),
            "development_scaffold_overlap_count": int(
                len(train_scaffolds.intersection(development_scaffolds))
            ),
            "metrics": _regression_metrics(target[development_mask], holdout_prediction),
            "not_locked_external_validation": True,
        }

    similarities, _neighbors, fingerprint_backend = nearest_neighbor_tanimoto(
        data["smiles"],
        data["smiles"],
        backend="rdkit_morgan",
        n_bits=policy.fingerprint_bits,
        radius=policy.morgan_radius,
        exclude_identical_positions=True,
    )
    if not np.isfinite(similarities).all():
        raise InsufficientTrainingDataError("Applicability-domain similarities are not finite")
    empirical_threshold = float(np.quantile(similarities, policy.domain_quantile, method="linear"))
    domain_threshold = float(max(policy.min_domain_similarity, empirical_threshold))

    final_model = clone(template).fit(data["smiles"], target)
    feature_metadata = dict(final_model.named_steps["features"].feature_metadata_)
    payload = {
        "schema_version": _ARTIFACT_SCHEMA_VERSION,
        "endpoint": endpoint,
        "model": final_model,
        "reference_smiles": tuple(data["smiles"].astype(str)),
        "interval_half_width": interval_radius,
        "coverage": policy.coverage,
        "domain_threshold": domain_threshold,
        "fingerprint_bits": policy.fingerprint_bits,
        "morgan_radius": policy.morgan_radius,
        "recommended_for_optimization": beats_baseline,
        "training_structure_count": n_samples,
    }
    _atomic_joblib_dump(payload, artifact_path)
    artifact_sha256 = _sha256_file(artifact_path)
    model_version = f"local-lab:{endpoint}:sha256:{artifact_sha256[:16]}"
    split_counts = {
        str(role): int(count) for role, count in data["split_role"].value_counts().sort_index().items()
    }
    manifest: dict[str, Any] = {
        "schema_version": _ARTIFACT_SCHEMA_VERSION,
        "endpoint": endpoint,
        "artifact": {
            "filename": artifact_path.name,
            "format": "controlled-local-joblib",
            "sha256": artifact_sha256,
            "model_version": model_version,
            "contains_raw_or_pseudonymous_ids": False,
            "contains_reference_structures_for_domain_checks": True,
            "trusted_local_artifact_only": True,
        },
        "data": {
            "source_scope": "private_historical_lab",
            "eligible_split_roles": sorted(_ALLOWED_SPLIT_ROLES),
            "unique_structures": n_samples,
            "unique_scaffolds": n_scaffolds,
            "split_role_counts": split_counts,
            "target_min": float(np.min(target)),
            "target_max": float(np.max(target)),
            "target_median": float(np.median(target)),
            "anonymous_dataset_sha256": _dataset_digest(data),
            "exclusions": exclusions,
        },
        "model": {
            "family": "ExtraTreesRegressor",
            "feature_metadata": feature_metadata,
            "training_policy": asdict(policy),
            "random_state": policy.random_state,
        },
        "validation": {
            "primary_method": "scaffold_grouped_out_of_fold",
            "fold_count": fold_count,
            "folds": fold_records,
            "oof_metrics": oof_metrics,
            "fold_specific_median_baseline_metrics": baseline_metrics,
            "baseline_mae_improvement_fraction": improvement_fraction,
            "passes_optimization_gate": beats_baseline,
            "development_holdout": holdout,
            "no_locked_external_or_prospective_claim": True,
        },
        "uncertainty": {
            "method": "absolute scaffold-OOF residual finite-sample conformal-style interval",
            "coverage": policy.coverage,
            "calibration_residual_count": int(len(residuals)),
            "interval_half_width": interval_radius,
        },
        "applicability_domain": {
            "method": "nearest training-structure Morgan Tanimoto",
            "fingerprint_backend": fingerprint_backend,
            "threshold": domain_threshold,
            "empirical_leave-one-out_quantile": empirical_threshold,
            "quantile": policy.domain_quantile,
            "configured_floor": policy.min_domain_similarity,
        },
        "status": {
            "development_model": True,
            "recommended_for_optimization": beats_baseline,
            "reason": (
                "scaffold-OOF metrics clear the configured baseline-improvement gate"
                if beats_baseline
                else "scaffold-OOF metrics do not clear the configured baseline-improvement gate"
            ),
        },
        "software": {
            "scikit_learn": sklearn.__version__,
            "rdkit": rdBase.rdkitVersion,
        },
    }
    _atomic_json_dump(manifest, manifest_path)
    return LocalRegressionArtifact(
        artifact_path=artifact_path,
        manifest_path=manifest_path,
        endpoint=endpoint,
        model_version=model_version,
        manifest=manifest,
    )


class LocalLabRegressionPredictor:
    """Hash-verified Predictor adapter for one confidential local model."""

    def __init__(
        self,
        artifact: str | Path,
        *,
        manifest: str | Path | None = None,
        verify_hash: bool = True,
    ) -> None:
        self.artifact_path = Path(artifact).expanduser().resolve()
        self.manifest_path = (
            Path(manifest).expanduser().resolve()
            if manifest is not None
            else self.artifact_path.with_name(
                f"{self.artifact_path.name.removesuffix('.joblib')}.manifest.json"
            )
        )
        self.verify_hash = bool(verify_hash)
        self._payload: dict[str, Any] | None = None
        self._manifest: dict[str, Any] | None = None
        self.endpoint = ""
        self._model_version = "unloaded-local-lab-model"
        self._load_metadata()

    def _load_metadata(self) -> None:
        if not self.artifact_path.is_file():
            raise FileNotFoundError(self.artifact_path)
        if not self.manifest_path.is_file():
            raise FileNotFoundError(self.manifest_path)
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise LocalArtifactVerificationError("Local model manifest must be a JSON object")
        artifact_record = manifest.get("artifact", {})
        expected_hash = str(artifact_record.get("sha256", ""))
        if not expected_hash:
            raise LocalArtifactVerificationError("Local model manifest has no artifact SHA-256")
        if self.verify_hash:
            observed_hash = _sha256_file(self.artifact_path)
            if observed_hash != expected_hash:
                raise LocalArtifactVerificationError("Local model artifact SHA-256 mismatch")
        endpoint = str(manifest.get("endpoint", "")).strip()
        if not endpoint:
            raise LocalArtifactVerificationError("Local model manifest has no endpoint")
        self.endpoint = endpoint
        self._model_version = str(
            artifact_record.get("model_version") or f"local-lab:{endpoint}:sha256:{expected_hash[:16]}"
        )
        self._manifest = manifest

    @property
    def model_version(self) -> str:
        return self._model_version

    @property
    def recommended_for_optimization(self) -> bool:
        assert self._manifest is not None
        return bool(self._manifest.get("status", {}).get("recommended_for_optimization", False))

    def _load(self) -> None:
        if self._payload is not None:
            return
        payload = _load_controlled_joblib(self.artifact_path)
        if not isinstance(payload, dict):
            raise LocalArtifactVerificationError("Local model artifact payload must be a mapping")
        required = {
            "schema_version",
            "endpoint",
            "model",
            "reference_smiles",
            "interval_half_width",
            "domain_threshold",
            "fingerprint_bits",
            "morgan_radius",
        }
        missing = sorted(required - set(payload))
        if missing:
            raise LocalArtifactVerificationError(f"Local model artifact is missing fields: {missing}")
        if payload["schema_version"] != _ARTIFACT_SCHEMA_VERSION:
            raise LocalArtifactVerificationError("Unsupported local model artifact schema")
        if str(payload["endpoint"]) != self.endpoint:
            raise LocalArtifactVerificationError("Artifact endpoint disagrees with manifest")
        reference = tuple(str(value) for value in payload["reference_smiles"])
        if not reference:
            raise LocalArtifactVerificationError("Local model has no domain reference structures")
        payload["reference_smiles"] = reference
        self._payload = payload

    def predict(self, smiles: str) -> PropertyEstimate:
        return self.predict_many([smiles])[0]

    def predict_many(self, smiles: Sequence[str]) -> list[PropertyEstimate]:
        canonical = [_canonical_smiles(value) for value in smiles]
        if not canonical:
            return []
        self._load()
        assert self._payload is not None  # narrowed by _load
        assert self._manifest is not None
        prediction = np.asarray(self._payload["model"].predict(pd.Series(canonical)), dtype=float)
        similarities, _neighbors, backend = nearest_neighbor_tanimoto(
            canonical,
            self._payload["reference_smiles"],
            backend="rdkit_morgan",
            n_bits=int(self._payload["fingerprint_bits"]),
            radius=int(self._payload["morgan_radius"]),
        )
        interval = float(self._payload["interval_half_width"])
        threshold = float(self._payload["domain_threshold"])
        recommended = bool(self._payload.get("recommended_for_optimization", False))
        estimates: list[PropertyEstimate] = []
        for mean, similarity in zip(prediction, similarities, strict=True):
            inside = bool(float(similarity) >= threshold)
            status_parts = ["local_historical_lab_development_prediction"]
            status_parts.append("in_domain" if inside else "outside_domain")
            if not recommended:
                status_parts.append("no_scaffold_oof_baseline_improvement")
            estimates.append(
                PropertyEstimate(
                    endpoint=self.endpoint,
                    mean=float(mean),
                    lower=float(mean - interval),
                    upper=float(mean + interval),
                    inside_domain=inside,
                    model_version=self.model_version,
                    evidence_status="_".join(status_parts),
                    metadata={
                        "artifact_path": str(self.artifact_path),
                        "manifest_path": str(self.manifest_path),
                        "artifact_hash_verified": self.verify_hash,
                        "source_scope": "private_historical_lab",
                        "prediction_is_not_measurement": True,
                        "development_model": True,
                        "recommended_for_optimization": recommended,
                        "nearest_reference_tanimoto": float(similarity),
                        "domain_threshold": threshold,
                        "fingerprint_backend": backend,
                        "interval_method": self._manifest["uncertainty"]["method"],
                        "interval_coverage": float(self._manifest["uncertainty"]["coverage"]),
                        "training_structure_count": int(self._manifest["data"]["unique_structures"]),
                        "no_locked_external_or_prospective_validation": True,
                    },
                )
            )
        return estimates
