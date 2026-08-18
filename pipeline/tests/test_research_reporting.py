from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from menin_discovery.research_reporting import (
    compact_model_evidence,
    model_failure_findings,
    optimizer_endpoint_summary,
    validate_explanation_contract,
    write_current_status_report,
)


def test_explanation_contract_rejects_incomplete_run() -> None:
    with pytest.raises(ValueError, match="incomplete"):
        validate_explanation_contract({"promotion_status": "decision-track"})


def test_explanation_contract_accepts_all_fields() -> None:
    payload = {
        "dataset_definition": "d",
        "split_definition": "s",
        "metrics_with_uncertainty": {},
        "calibration": "c",
        "applicability_domain": "a",
        "residual_clusters": [],
        "feature_layer_ablations": [],
        "matched_pair_examples": [],
        "proposed_physical_explanation": "p",
        "competing_explanations_and_confounders": "x",
        "falsifying_simulation_or_assay": "f",
        "promotion_status": "discovery-track",
    }
    validate_explanation_contract(payload)


def test_compact_model_evidence_normalizes_herg_schema_and_rejection() -> None:
    summary = pd.DataFrame(
        [
            {
                "model": "censored_gaussian_ridge",
                "feature_layer": "state_conformer_physics",
                "status": "rejected-nonconverged",
                "pic50_mae": 1.045,
                "classification_brier": 0.378,
                "classification_ece_8bin": 0.392,
                "fit_converged_fraction": 0.0,
                "n_unique_compounds": 75,
            }
        ]
    )

    result = compact_model_evidence(summary)

    assert result.loc[0, "domain"] == "hERG"
    assert result.loc[0, "endpoint"] == "continuous_hERG"
    assert result.loc[0, "promotion"] == "rejected"
    assert result.loc[0, "primary_evidence"] == "pIC50 MAE=1.045"
    assert "fit convergence 0.000" in result.loc[0, "gate_or_failure"]


def test_failure_findings_state_joint_herg_failure_as_falsification() -> None:
    summary = pd.DataFrame(
        [
            {
                "domain": "herg",
                "feature_layer": "state_conformer_physics",
                "pic50_mae": 1.045,
                "fit_converged_fraction": 0.0,
            },
            {
                "domain": "herg",
                "feature_layer": "structure_2d_joint_observations",
                "heldout_inhibition_mae_percent": 31.48,
                "heldout_inhibition_rmse_percent": 39.34,
            },
        ]
    )

    findings = "\n".join(model_failure_findings(summary))

    assert "must not be interpreted" in findings
    assert "falsifies a single global static Hill mapping" in findings


def test_optimizer_summary_distinguishes_predictions_domain_and_required_data() -> None:
    contract = pd.DataFrame(
        {
            "mean__herg_pic50": [5.2, 4.8],
            "domain_status__herg_pic50": ["inside", "outside"],
            "promotion_status__herg_pic50": ["provisional-discovery"] * 2,
            "lineage_role__herg_pic50": ["modeled_endpoint"] * 2,
            "required_data__herg_pic50": [False, False],
            "mean__free_exposure_margin": [float("nan"), float("nan")],
            "domain_status__free_exposure_margin": ["required_data"] * 2,
            "promotion_status__free_exposure_margin": ["required_data"] * 2,
            "required_data__free_exposure_margin": [True, True],
        }
    )

    result = optimizer_endpoint_summary(contract).set_index("endpoint")

    assert result.loc["herg_pic50", "predictions"] == 2
    assert result.loc["herg_pic50", "inside_domain"] == 1
    assert result.loc["herg_pic50", "outside_domain"] == 1
    assert result.loc["free_exposure_margin", "required_data"] == 2


def test_status_report_exposes_source_qc_failures_and_optimizer_readiness(tmp_path: Path) -> None:
    model_summary = pd.DataFrame(
        [
            {
                "domain": "herg",
                "endpoint": "continuous_hERG",
                "model": "joint",
                "feature_layer": "structure_2d_joint_observations",
                "status": "evaluated",
                "pic50_mae": 0.58,
                "heldout_inhibition_mae_percent": 31.48,
                "heldout_inhibition_rmse_percent": 39.34,
            }
        ]
    )
    optimizer = pd.DataFrame(
        [
            {
                "endpoint": "free_exposure_margin",
                "predictions": 0,
                "inside_domain": 0,
                "outside_domain": 0,
                "promotion": "required_data",
                "lineage_role": "required_data",
                "required_data": 110,
            }
        ]
    )
    output = tmp_path / "status.md"

    write_current_status_report(
        output,
        inventory={"n_unique_compounds": 110, "mw_min": 665, "mw_max": 813},
        stage_status=pd.DataFrame([{"stage": "report", "status": "complete"}]),
        model_summary=model_summary,
        regime_result={"reason": "No defensible cutoff."},
        assay_summary={"panel_size": 16, "matched_pairs": 4, "rat_profiles": 8, "herg_kinetics": 6},
        heavy_physics_status={"status": "not launched"},
        source_qc={
            "public_unique_structures": 9406,
            "public_measurements": 25573,
            "public_quarantine": 21,
            "public_train_validation_overlap": 1,
            "public_mw_domain_contradictions": 326,
        },
        optimizer_summary=optimizer,
        run_context={"smoke_mode": True},
    )

    text = output.read_text(encoding="utf-8")
    assert "source class `0` means blocker" in text
    assert "**21 rows** are quarantined" in text
    assert "single global static Hill mapping" in text
    assert "free_exposure_margin" in text
    assert "smoke/diagnostic" in text
    assert "| hERG" in text
