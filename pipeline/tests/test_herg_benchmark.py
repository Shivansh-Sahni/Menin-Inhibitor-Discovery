from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml
from menin_discovery.herg_benchmark import (
    REGIMES,
    ModelSpec,
    _fit_model,
    _predict_model,
    _regime_training_indices,
    build_model_specs,
    calculate_feature_registry,
    parse_ic50_um,
    parse_percent_inhibition,
)
from scipy import sparse


def test_ic50_parser_handles_exact_censored_intermediate_and_missing_values():
    kwargs = {"blocker_max_um": 10.0, "nonblocker_min_um": 30.0}
    assert parse_ic50_um("0.4", **kwargs)["label"] == 1
    assert parse_ic50_um("<0.37", **kwargs)["label"] == 1
    assert parse_ic50_um(">30", **kwargs)["label"] == 0
    assert np.isnan(parse_ic50_um("18.2", **kwargs)["label"])
    assert parse_ic50_um(None, **kwargs)["status"] == "missing"
    assert parse_percent_inhibition("41%@10uM and 75%@30uM", 10) == 41
    assert parse_percent_inhibition("13@10uM and 26%@30uM", 10) == 13


def test_feature_registry_is_finite_and_has_expected_representations():
    smiles = ["CCN", "c1ccccc1", "CC(=O)Nc1ccc(O)cc1", "C[N+](C)(C)CCO"]
    matrices, descriptors, metadata = calculate_feature_registry(smiles)
    assert descriptors.shape[0] == len(smiles)
    assert metadata["rdkit_2d_descriptors"]["n_features"] >= 200
    assert metadata["morgan_1024_r2"]["n_features"] == 1024
    assert metadata["maccs_167"]["n_features"] == 167
    for matrix in matrices.values():
        values = matrix.data if sparse.issparse(matrix) else matrix
        assert np.isfinite(values).all()


def test_quick_grid_covers_requested_model_families_and_complexities():
    config_path = Path(__file__).parents[1] / "config" / "herg_benchmark.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    specs = build_model_specs(config, "quick")
    families = {spec.family for spec in specs}
    assert {
        "xgboost",
        "lightgbm",
        "random_forest",
        "svm",
        "clustering",
        "knn",
        "rnn",
    }.issubset(families)
    for family in families - {"dummy"}:
        complexities = {spec.complexity for spec in specs if spec.family == family}
        assert {"simple", "complex"}.issubset(complexities)


def test_strong_ml_grid_is_focused_classical_search_with_three_complexities():
    config_path = Path(__file__).parents[1] / "config" / "herg_benchmark.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    specs = build_model_specs(config, "strong_ml")
    assert {spec.family for spec in specs} == {
        "logistic",
        "random_forest",
        "extra_trees",
        "svm",
        "xgboost",
        "lightgbm",
    }
    assert len(specs) == 18
    for family in {spec.family for spec in specs}:
        assert {spec.complexity for spec in specs if spec.family == family} == {
            "simple",
            "moderate",
            "complex",
        }


def test_three_regimes_apply_expected_public_and_private_weights():
    public = np.array([0, 1, 2])
    private = np.array([10, 11])
    for regime in REGIMES:
        indices, weights = _regime_training_indices(
            regime,
            public_indices=public,
            private_train_indices=private,
            private_weight=5.0,
        )
        if regime == "confidential_only":
            assert indices.tolist() == private.tolist()
            assert weights.tolist() == [1.0, 1.0]
        elif regime == "confidential_prioritized":
            assert indices.tolist() == [0, 1, 2, 10, 11]
            assert weights.tolist() == [1.0, 1.0, 1.0, 5.0, 5.0]
        else:
            assert indices.tolist() == [0, 1, 2, 10, 11]
            assert weights.tolist() == [1.0] * 5


def test_character_rnn_fits_and_returns_probabilities():
    smiles = ["CC", "CCC", "CCCC", "c1ccccc1", "NCCN", "O=C=O", "CCO", "CCN"] * 2
    y = np.array([0, 0, 1, 1, 1, 0, 0, 1] * 2, dtype=int)
    spec = ModelSpec(
        family="rnn",
        complexity="simple",
        parameters={
            "cell": "gru",
            "embedding_dim": 4,
            "hidden_dim": 8,
            "layers": 1,
            "epochs": 1,
            "batch_size": 16,
        },
        feature_sets=("smiles_tokens",),
    )
    fitted = _fit_model(spec, smiles, y, np.ones(len(y)), random_state=7)
    probability = _predict_model(fitted, smiles)
    assert probability.shape == (len(smiles),)
    assert np.isfinite(probability).all()
    assert ((probability >= 0) & (probability <= 1)).all()
