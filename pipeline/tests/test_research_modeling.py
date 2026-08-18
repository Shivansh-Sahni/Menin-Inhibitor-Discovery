from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from menin_discovery.research_feature_ontology import CONVENTIONAL_DESCRIPTOR_COLUMNS
from menin_discovery.research_modeling import (
    CensoredGaussianRidge,
    final_fit_censored_herg_predictions,
    final_fit_regression_predictions,
    grouped_censored_herg_benchmark,
    grouped_joint_herg_benchmark,
    grouped_regression_benchmark,
    merge_feature_layers,
    promotion_decision,
    regression_metrics,
)


def test_ineligible_physics_values_are_masked_before_model_layers() -> None:
    compounds = pd.DataFrame(
        {
            "compound_id": ["C1", "C2"],
            "standardized_smiles": ["CCN", "CCO"],
        }
    )
    physics = pd.DataFrame(
        {
            "compound_id": ["C1", "C2"],
            "polar_sasa_ang2__q05": [75.0, 999.0],
            "composite__unreviewed_mechanism": [0.25, 999.0],
            "physics_model_eligible": [True, False],
            "physics_substantive_failure_count": [0, 1],
        }
    )
    features, layers = merge_feature_layers(compounds, physics)

    assert "polar_sasa_ang2__q05" in layers["state_conformer_physics"]
    assert set(layers) >= {
        "structure_2d",
        "physics_conformation_and_exposure",
        "state_conformer_physics",
    }
    assert "composite__unreviewed_mechanism" not in layers["state_conformer_physics"]
    assert features.loc[features["compound_id"] == "C1", "polar_sasa_ang2__q05"].iloc[0] == 75.0
    assert pd.isna(features.loc[features["compound_id"] == "C2", "polar_sasa_ang2__q05"].iloc[0])
    assert "physics_substantive_failure_count" not in layers["state_conformer_physics"]
    assert layers["structure_2d"] == list(CONVENTIONAL_DESCRIPTOR_COLUMNS)
    assert "exact_mol_wt" not in layers["structure_2d"]
    assert "heavy_atom_count" not in layers["structure_2d"]
    assert "formal_charge" not in layers["structure_2d"]


def test_regression_metrics_are_multiplicative() -> None:
    metrics = regression_metrics(np.array([1.0, 2.0]), np.array([1.30103, 1.69897]))
    assert metrics["median_fold_error"] == pytest.approx(2.0, rel=1e-4)
    assert metrics["fraction_within_2fold"] == 1.0


def test_censored_gaussian_ridge_respects_direction() -> None:
    X = np.arange(12, dtype=float).reshape(-1, 1)
    exact = 4.0 + 0.2 * X.ravel()
    lower = exact.copy()
    upper = exact.copy()
    lower[-2:] = exact[-2:] - 0.1
    upper[-2:] = np.inf
    model = CensoredGaussianRidge(alpha=0.01).fit(X, lower, upper)
    assert model.predict(np.array([[10.0]]))[0] > model.predict(np.array([[1.0]]))[0]


def test_censored_gaussian_ridge_converges_with_mixed_censoring_and_many_features() -> None:
    rng = np.random.default_rng(20260729)
    X = rng.normal(size=(48, 30))
    signal = 5.2 + X[:, :4] @ np.array([0.25, -0.2, 0.15, 0.1])
    observed = signal + rng.normal(0.0, 0.12, len(X))
    lower = observed.copy()
    upper = observed.copy()
    lower[::8] = observed[::8] - 0.1
    upper[::8] = np.inf
    lower[1::8] = -np.inf
    upper[1::8] = observed[1::8] + 0.1
    lower[2::8] = observed[2::8] - 0.15
    upper[2::8] = observed[2::8] + 0.15

    model = CensoredGaussianRidge(alpha=3.0, maxiter=3000).fit(X, lower, upper)

    assert model.converged_
    assert np.isfinite(model.predict(X)).all()
    assert model.sigma_ > 0


def test_grouped_benchmark_holds_out_groups() -> None:
    rng = np.random.default_rng(4)
    frame = pd.DataFrame(
        {
            "compound_id": [f"C{i}" for i in range(24)],
            "scaffold": np.repeat([f"S{i}" for i in range(6)], 4),
            "feature": rng.normal(size=24),
        }
    )
    frame["target"] = np.exp(frame["feature"] + rng.normal(0, 0.05, 24))
    metrics, predictions = grouped_regression_benchmark(
        frame,
        feature_columns=["feature"],
        target_column="target",
        folds=3,
    )
    assert set(predictions["compound_id"]) == set(frame["compound_id"])
    assert {"log_mae", "prediction_interval_coverage"}.issubset(metrics.columns)


def test_promotion_requires_all_gates() -> None:
    result = promotion_decision(
        {"mae": 0.5},
        {"mae": 0.4},
        primary_metric="mae",
        calibrated=False,
        converged=True,
    )
    assert result["promotion_status"] == "discovery-track"


def test_final_pk_fit_scores_full_library_from_oof_uncertainty() -> None:
    values = np.linspace(-1.0, 1.0, 18)
    training = pd.DataFrame(
        {
            "compound_id": [f"C{i}" for i in range(18)],
            "standardized_smiles": ["C" * (i + 2) for i in range(18)],
            "scaffold": np.repeat([f"S{i}" for i in range(6)], 3),
            "feature": values,
            "target": np.power(10.0, 2.0 + 0.25 * values),
        }
    )
    _, oof = grouped_regression_benchmark(
        training,
        feature_columns=["feature"],
        target_column="target",
        folds=3,
        model_names=["ridge"],
    )
    scoring = pd.DataFrame(
        {
            "compound_id": [*training["compound_id"], "NEW"],
            "standardized_smiles": [*training["standardized_smiles"], "COC"],
            "feature": [*training["feature"], 0.25],
        }
    )
    final = final_fit_regression_predictions(
        training,
        scoring,
        oof,
        feature_columns=["feature"],
        target_column="target",
        selected_model="ridge",
        endpoint="rat_po_auc_dose_normalized",
        unit="ng*h/mL per mg/kg",
    )
    assert len(final) == len(scoring)
    assert final["compound_id"].nunique() == len(scoring)
    assert (final["lower"] < final["mean"]).all()
    assert (final["mean"] < final["upper"]).all()
    assert final["uncertainty_method"].str.contains("OOF").all()
    assert set(final["promotion_status"]) == {"provisional-discovery"}


def test_final_censored_herg_fit_derives_probability_for_full_library() -> None:
    values = np.linspace(-1.0, 1.0, 18)
    observed = 5.0 + 0.4 * values
    training = pd.DataFrame(
        {
            "compound_id": [f"H{i}" for i in range(18)],
            "standardized_smiles": ["N" + "C" * (i + 1) for i in range(18)],
            "scaffold": np.repeat([f"S{i}" for i in range(6)], 3),
            "feature": values,
            "pic50_lower": observed,
            "pic50_upper": observed,
        }
    )
    _, oof = grouped_censored_herg_benchmark(
        training,
        feature_columns=["feature"],
        folds=3,
        alpha=0.1,
    )
    scoring = pd.DataFrame(
        {
            "compound_id": [*training["compound_id"], "NEW"],
            "standardized_smiles": [*training["standardized_smiles"], "NCO"],
            "feature": [*training["feature"], 0.0],
        }
    )
    final = final_fit_censored_herg_predictions(
        training,
        scoring,
        oof,
        feature_columns=["feature"],
        alpha=0.1,
    )
    assert len(final) == 2 * len(scoring)
    assert set(final["endpoint"]) == {"herg_pic50", "herg_blocker_probability"}
    probability = final[final["endpoint"] == "herg_blocker_probability"]
    assert probability["mean"].between(0.0, 1.0).all()
    assert probability["lower"].between(0.0, 1.0).all()
    assert probability["upper"].between(0.0, 1.0).all()


def test_joint_herg_benchmark_scores_held_out_inhibition_evidence() -> None:
    feature = np.linspace(-1.0, 1.0, 18)
    compounds = pd.DataFrame(
        {
            "compound_id": [f"J{i}" for i in range(18)],
            "scaffold": np.repeat([f"S{i}" for i in range(6)], 3),
            "feature": feature,
        }
    )
    pic50 = 5.0 + 0.35 * feature
    potency = pd.DataFrame(
        {
            "compound_id": compounds["compound_id"],
            "pic50_lower": pic50,
            "pic50_upper": pic50,
        }
    )
    concentration_um = np.repeat([1.0, 3.0, 10.0], 6)
    ic50_um = np.power(10.0, 6.0 - pic50)
    inhibition = pd.DataFrame(
        {
            "compound_id": compounds["compound_id"],
            "test_concentration_um": concentration_um,
            "inhibition_percent": 100.0 * concentration_um / (concentration_um + ic50_um),
        }
    )

    metrics, predictions = grouped_joint_herg_benchmark(
        compounds,
        potency,
        inhibition,
        feature_columns=["feature"],
        folds=3,
        alpha=0.1,
    )

    inhibition_predictions = predictions[predictions["evidence_type"] == "concentration_inhibition"]
    assert len(inhibition_predictions) == len(inhibition)
    assert inhibition_predictions["predicted_inhibition_percent"].between(0.0, 100.0).all()
    assert metrics["heldout_inhibition_n"] == len(inhibition)
    assert np.isfinite(metrics["heldout_inhibition_negative_log_likelihood"])
