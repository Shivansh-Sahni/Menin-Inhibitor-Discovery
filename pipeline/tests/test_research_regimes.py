from __future__ import annotations

import numpy as np
import pandas as pd
from menin_discovery.research_regimes import apply_cross_outcome_cutoff_gate, bootstrap_mw_change_point


def test_change_point_rejects_unsupported_small_groups() -> None:
    frame = pd.DataFrame(
        {
            "mw": np.linspace(650, 780, 20),
            "outcome": np.linspace(0, 1, 20),
            "scaffold": [f"S{i // 2}" for i in range(20)],
        }
    )
    summary, _ = bootstrap_mw_change_point(
        frame,
        outcome="outcome",
        minimum_per_side=15,
        bootstrap_replicates=10,
    )
    assert summary["bootstrap_breakpoint_selection_frequency"] == 0.0


def test_cross_outcome_gate_requires_two() -> None:
    rows = [
        {
            "outcome": "A",
            "bootstrap_breakpoint_selection_frequency": 0.9,
            "location_interval_width_da": 20.0,
            "effect_direction_stability": 0.9,
            "location_interval_95_low_da": 690.0,
            "location_interval_95_high_da": 710.0,
        }
    ]
    _, result = apply_cross_outcome_cutoff_gate(rows)
    assert result["supported_cutoff"] is False


def test_change_point_models_mw_interactions_across_available_physical_families() -> None:
    rng = np.random.default_rng(4)
    n = 120
    mw = np.linspace(650.0, 780.0, n)
    flexibility = 3.0 + 0.02 * (mw - 650.0) + rng.normal(0.0, 0.5, n)
    charge = rng.integers(-1, 3, n).astype(float)
    exposed_polarity = 100.0 + rng.normal(0.0, 10.0, n)
    folded_fraction = np.clip(0.6 - 0.002 * (mw - 650.0) + rng.normal(0.0, 0.05, n), 0.0, 1.0)
    outcome = (
        0.004 * (mw - 700.0)
        + 0.15 * flexibility
        - 0.08 * charge
        + 0.003 * exposed_polarity
        + 0.4 * folded_fraction
        + 0.05 * np.maximum(mw - 710.0, 0.0)
        + rng.normal(0.0, 0.05, n)
    )
    frame = pd.DataFrame(
        {
            "mw": mw,
            "outcome": outcome,
            "scaffold": [f"S{index // 4}" for index in range(n)],
            "rotatable_bonds": flexibility,
            "formal_charge": charge,
            "tpsa": exposed_polarity,
            "joint_conformational_entropy_normalized": flexibility / 10.0,
            "charge_centroid_separation_angstrom__mean": charge + 2.0,
            "polar_sasa_ang2__mean": exposed_polarity * 1.5,
            "folded_low_polarity_fraction": folded_fraction,
        }
    )

    summary, bootstrap = bootstrap_mw_change_point(
        frame,
        outcome="outcome",
        covariates=["rotatable_bonds", "formal_charge", "tpsa"],
        bootstrap_replicates=30,
        random_state=3,
    )

    assert summary["covariate_families"] == {
        "joint_conformational_entropy_normalized": "flexibility",
        "charge_centroid_separation_angstrom__mean": "charge_state_behavior",
        "polar_sasa_ang2__mean": "exposed_polarity",
        "folded_low_polarity_fraction": "folding",
    }
    assert summary["mw_interaction_terms"] == [
        "mw_x_joint_conformational_entropy_normalized",
        "mw_x_charge_centroid_separation_angstrom__mean",
        "mw_x_polar_sasa_ang2__mean",
        "mw_x_folded_low_polarity_fraction",
    ]
    assert summary["observed_threshold_da"] == 710.0
    assert summary["observed_n_left"] >= 15
    assert summary["observed_n_right"] >= 15
    assert summary["observed_breakpoint_model_probability"] > 0.99
    assert bootstrap["selected"].mean() >= 0.70


def _passing_summary(outcome: str, low: float = 690.0, high: float = 710.0) -> dict[str, object]:
    return {
        "outcome": outcome,
        "observed_selected": True,
        "observed_n_left": 25,
        "observed_n_right": 25,
        "minimum_per_side": 15,
        "bootstrap_breakpoint_selection_frequency": 0.9,
        "location_interval_width_da": high - low,
        "effect_direction_stability": 0.9,
        "location_interval_95_low_da": low,
        "location_interval_95_high_da": high,
    }


def test_cross_outcome_gate_requires_mechanistic_adjacency() -> None:
    summaries = [
        _passing_summary("herg_pic50"),
        _passing_summary("log10_iv_auc_dose_normalized"),
    ]

    _, result = apply_cross_outcome_cutoff_gate(summaries)

    assert result["supported_cutoff"] is False
    assert result["supported_cutoff_da"] is None
    assert "No defensible single MW cutoff" in result["reason"]


def test_cross_outcome_gate_promotes_only_overlapping_adjacent_pk_outcomes() -> None:
    summaries = [
        _passing_summary("log10_iv_auc_dose_normalized", 690.0, 710.0),
        _passing_summary("log10_po_auc_dose_normalized", 700.0, 720.0),
    ]

    frame, result = apply_cross_outcome_cutoff_gate(summaries)

    assert frame["passes_within_outcome_gate"].all()
    assert result["supported_cutoff"] is True
    assert result["supported_interval_da"] == [700.0, 710.0]
    assert result["supported_cutoff_da"] == 705.0


def test_cross_outcome_gate_enforces_observation_count_and_width() -> None:
    too_small = _passing_summary("log10_iv_auc_dose_normalized")
    too_small["observed_n_left"] = 14
    too_wide = _passing_summary("log10_po_auc_dose_normalized", 650.0, 720.0)

    frame, result = apply_cross_outcome_cutoff_gate([too_small, too_wide])

    assert not frame["passes_within_outcome_gate"].any()
    assert "fewer_than_15_observations_on_one_side" in frame.loc[0, "gate_failures"]
    assert "location_interval_too_wide_or_unidentified" in frame.loc[1, "gate_failures"]
    assert result["supported_cutoff"] is False
