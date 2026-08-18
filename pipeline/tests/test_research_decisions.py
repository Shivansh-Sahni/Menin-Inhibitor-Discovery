from __future__ import annotations

import pandas as pd
import pytest
from menin_discovery.research_decisions import build_optimizer_contract, select_assay_panel


def test_optimizer_contract_has_no_rank_or_scalar() -> None:
    compounds = pd.DataFrame({"compound_id": ["A"], "standardized_smiles": ["CC"]})
    contract = build_optimizer_contract(compounds)
    assert contract.loc[0, "scalar_objective"] == "NOT_DEFINED"
    assert contract.loc[0, "molecule_rank"] == "NOT_COMPUTED"
    assert not bool(contract.loc[0, "generation_allowed"])
    assert contract.loc[0, "contract_status"] == "BLOCKED_FOR_OPTIMIZATION_PENDING_PROSPECTIVE_VALIDATION"
    assert bool(contract.loc[0, "required_data__rat_po_cmax_dose_normalized"])


def test_optimizer_contract_pivots_normalized_predictions_without_ranking() -> None:
    compounds = pd.DataFrame({"compound_id": ["A"], "standardized_smiles": ["CC"]})
    predictions = pd.DataFrame(
        {
            "compound_id": ["A", "A"],
            "endpoint": ["herg_pic50", "herg_blocker_probability"],
            "mean": [5.4, 0.71],
            "lower": [4.8, 0.30],
            "upper": [6.0, 0.93],
            "uncertainty": [0.6, 0.315],
            "domain_status": ["inside", "inside"],
            "promotion_status": ["provisional-discovery", "provisional-discovery"],
        }
    )
    contract = build_optimizer_contract(compounds, predictions)
    assert contract.loc[0, "mean__herg_pic50"] == 5.4
    assert contract.loc[0, "mean__herg_blocker_probability"] == 0.71
    assert not bool(contract.loc[0, "required_data__herg_pic50"])
    assert contract.loc[0, "molecule_rank"] == "NOT_COMPUTED"


def test_optimizer_contract_adds_required_data_flags_for_all_modeled_and_closure_endpoints() -> None:
    compounds = pd.DataFrame({"compound_id": ["A"], "standardized_smiles": ["CC"]})
    predictions = pd.DataFrame(
        {
            "compound_id": ["A", "A"],
            "endpoint": ["rat_iv_auc_dose_normalized", "custom_mechanistic_endpoint"],
            "mean": [125.0, 0.42],
        }
    )

    contract = build_optimizer_contract(compounds, predictions)

    assert not bool(contract.loc[0, "required_data__rat_iv_auc_dose_normalized"])
    assert not bool(contract.loc[0, "required_data__custom_mechanistic_endpoint"])
    assert bool(contract.loc[0, "required_data__rat_bioavailability_closure_percent"])


def test_optimizer_contract_rejects_duplicate_endpoint_predictions() -> None:
    compounds = pd.DataFrame({"compound_id": ["A"], "standardized_smiles": ["CC"]})
    duplicate = pd.DataFrame(
        {
            "compound_id": ["A", "A"],
            "endpoint": ["herg_pic50", "herg_pic50"],
            "mean": [5.0, 5.1],
        }
    )
    with pytest.raises(ValueError, match="one row per compound and endpoint"):
        build_optimizer_contract(compounds, duplicate)


def test_panel_applies_mw_quotas() -> None:
    smiles = ["C" * (index + 2) for index in range(18)]
    frame = pd.DataFrame(
        {
            "compound_id": [f"C{index}" for index in range(18)],
            "standardized_smiles": smiles,
            "mw": [660] * 5 + [720] * 8 + [760] * 5,
            "model_uncertainty": [index / 18 for index in range(18)],
        }
    )
    panel, pk, herg, _ = select_assay_panel(frame, minimum_matched_pairs=0)
    assert len(panel) == 16
    assert (panel["mw_bin"] == "650-699").sum() >= 3
    assert (panel["mw_bin"] == "700-749").sum() >= 6
    assert (panel["mw_bin"] == "750+").sum() >= 3
    assert len(pk) == 8
    assert len(herg) == 6
    assert "acquisition_priority_score" in panel
    assert "information_gain_score" not in panel
    assert set(panel["evaluation_role"]) == {"mechanistic_assay_design_not_unbiased_prospective_model_test"}
