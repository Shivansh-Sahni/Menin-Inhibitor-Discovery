from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from menin_discovery.chemistry import standardize_smiles

from menin_edit.local_models import (
    InsufficientTrainingDataError,
    LocalArtifactVerificationError,
    LocalLabRegressionPredictor,
    LocalRegressionConfig,
    UnsafeArtifactPathError,
    train_local_regression,
)
from menin_edit.predictors import Predictor


def _synthetic_tables(count: int = 24) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Acyclic compounds receive exact-structure scaffold keys, giving this fast
    # synthetic fixture enough independent groups to exercise GroupKFold.
    smiles_values = ["C" * length for length in range(2, count + 2)]
    compounds = []
    observations = []
    for index, smiles in enumerate(smiles_values):
        standardized = standardize_smiles(smiles, require_rdkit=True)
        assert standardized.structure_valid
        compounds.append(
            {
                "structure_id": standardized.structure_id,
                "smiles": standardized.standardized_smiles,
                "compound_id": f"SECRET-COMPOUND-{index}",
            }
        )
        observations.append(
            {
                "structure_id": standardized.structure_id,
                "observation_id": f"SECRET-OBSERVATION-{index}",
                "endpoint": "menin_biochemical_pIC50",
                "model_value": 5.0 + 0.09 * index,
                "is_exact": True,
                "is_censored": False,
                "label_conflict": False,
                "provenance_conflict": False,
                "split_role": "train" if index < count - 5 else "development",
            }
        )

    # Rows that must be excluded before training.
    template = dict(observations[0])
    observations.extend(
        [
            {**template, "observation_id": "SECRET-LOCKED", "split_role": "locked_external"},
            {**template, "observation_id": "SECRET-CENSORED", "is_exact": False, "is_censored": True},
            {**template, "observation_id": "SECRET-CONFLICT", "label_conflict": True},
        ]
    )
    return pd.DataFrame(compounds), pd.DataFrame(observations)


def _fast_config(**overrides) -> LocalRegressionConfig:
    values = {
        "min_samples": 12,
        "min_unique_scaffolds": 3,
        "min_calibration_residuals": 8,
        "cv_folds": 3,
        "fingerprint_bits": 128,
        "n_estimators": 16,
        "min_domain_similarity": 0.05,
    }
    values.update(overrides)
    return LocalRegressionConfig(**values)


def test_train_save_load_predict_without_identifier_leakage(tmp_path: Path):
    compounds, observations = _synthetic_tables()
    result = train_local_regression(
        compounds,
        observations,
        endpoint="menin_biochemical_pIC50",
        output_dir=tmp_path / "private" / "models",
        config=_fast_config(),
    )

    assert result.artifact_path.is_file()
    assert result.manifest_path.is_file()
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["data"]["unique_structures"] == 24
    assert manifest["data"]["exclusions"]["excluded_non_development_role"] == 1
    assert manifest["data"]["exclusions"]["excluded_non_exact_or_censored"] == 1
    assert manifest["data"]["exclusions"]["excluded_label_conflict"] == 1
    assert manifest["validation"]["primary_method"] == "scaffold_grouped_out_of_fold"
    assert manifest["uncertainty"]["calibration_residual_count"] == 24
    assert manifest["artifact"]["contains_raw_or_pseudonymous_ids"] is False
    assert b"SECRET-COMPOUND" not in result.artifact_path.read_bytes()
    assert b"SECRET-OBSERVATION" not in result.artifact_path.read_bytes()

    predictor = LocalLabRegressionPredictor(result.artifact_path)
    assert isinstance(predictor, Predictor)
    estimate = predictor.predict(compounds.iloc[3]["smiles"])
    assert estimate.endpoint == "menin_biochemical_pIC50"
    assert estimate.lower <= estimate.mean <= estimate.upper
    assert estimate.inside_domain is True
    assert estimate.model_version == result.model_version
    assert estimate.metadata["artifact_hash_verified"] is True
    assert estimate.metadata["source_scope"] == "private_historical_lab"


def test_tiny_or_constant_endpoint_fails_without_writing(tmp_path: Path):
    compounds, observations = _synthetic_tables(count=8)
    destination = tmp_path / "private" / "models"
    with pytest.raises(InsufficientTrainingDataError, match="eligible unique structures"):
        train_local_regression(
            compounds,
            observations,
            endpoint="menin_biochemical_pIC50",
            output_dir=destination,
            config=_fast_config(min_samples=12),
        )
    assert not destination.exists()

    compounds, observations = _synthetic_tables(count=14)
    observations.loc[observations["split_role"].isin(["train", "development"]), "model_value"] = 7.0
    with pytest.raises(InsufficientTrainingDataError, match="no usable variance"):
        train_local_regression(
            compounds,
            observations,
            endpoint="menin_biochemical_pIC50",
            output_dir=destination,
            config=_fast_config(),
        )
    assert not destination.exists()


def test_training_refuses_unprotected_output_path(tmp_path: Path):
    compounds, observations = _synthetic_tables()
    with pytest.raises(UnsafeArtifactPathError, match="confidential local model"):
        train_local_regression(
            compounds,
            observations,
            endpoint="menin_biochemical_pIC50",
            output_dir=tmp_path / "published-models",
            config=_fast_config(),
        )


def test_predictor_rejects_hash_mismatch(tmp_path: Path):
    compounds, observations = _synthetic_tables()
    result = train_local_regression(
        compounds,
        observations,
        endpoint="menin_biochemical_pIC50",
        output_dir=tmp_path / "private" / "models",
        config=_fast_config(),
    )
    with result.artifact_path.open("ab") as handle:
        handle.write(b"tampered")

    with pytest.raises(LocalArtifactVerificationError, match="SHA-256 mismatch"):
        LocalLabRegressionPredictor(result.artifact_path)
