from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from menin_discovery.research_common import atomic_write_json
from menin_discovery.research_workflows import (
    _censored_fit_status,
    _merge_conformer_state_weights,
    prepare_herg_evidence,
    prepare_pk_tasks,
)


def _compound_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    compounds = pd.DataFrame(
        {
            "compound_id": ["C1"],
            "standardized_smiles": ["CCO"],
            "mw": [700.0],
            "display_name": ["C1"],
        }
    )
    features = pd.DataFrame({"compound_id": ["C1"], "scaffold": ["acyclic"]})
    return compounds, features


def test_prepare_herg_evidence_uses_normalized_inhibition_endpoint() -> None:
    compounds, features = _compound_tables()
    measurements = pd.DataFrame(
        {
            "compound_id": ["C1", "C1"],
            "endpoint": ["herg_ic50", "herg_percent_inhibition"],
            "value": [12.0, 47.0],
            "relation": ["=", "="],
            "test_concentration_value": [pd.NA, 10.0],
        }
    )
    _, potency, inhibition = prepare_herg_evidence(compounds, measurements, features)
    assert len(potency) == 1
    assert len(inhibition) == 1
    assert inhibition.iloc[0]["inhibition_percent"] == 47.0


def test_prepare_pk_tasks_excludes_unresolved_studies() -> None:
    compounds, features = _compound_tables()
    measurements = pd.DataFrame(
        {
            "compound_id": ["C1", "C1"],
            "measurement_id": ["M-resolved", "M-unresolved"],
            "pk_study_id": ["S-resolved", "S-unresolved"],
            "endpoint": ["vdss", "vdss"],
            "value": [2.0, 9.0],
            "relation": ["=", "="],
            "route": ["IV", "IV"],
            "species": ["rat", "rat"],
            "pairing_status": ["resolved", "unresolved"],
            "model_eligible": [True, True],
        }
    )
    studies = pd.DataFrame(
        {
            "pk_study_id": ["S-resolved", "S-unresolved"],
            "dose_value": [2.0, 2.0],
            "dose_unit": ["mg/kg", "mg/kg"],
            "route": ["IV", "IV"],
            "species": ["rat", "rat"],
            "pairing_status": ["resolved", "unresolved"],
        }
    )
    tasks = prepare_pk_tasks(compounds, measurements, studies, features)
    assert tasks["vdss"]["measurement_id"].tolist() == ["M-resolved"]


def test_prepare_pk_tasks_uses_dose_normalized_cmax_only() -> None:
    compounds, features = _compound_tables()
    measurements = pd.DataFrame(
        {
            "compound_id": ["C1"],
            "measurement_id": ["M1"],
            "pk_study_id": ["S1"],
            "endpoint": ["cmax"],
            "value": [300.0],
            "unit": ["ng/mL"],
            "relation": ["="],
            "route": ["PO"],
            "species": ["rat"],
            "pairing_status": ["resolved"],
            "model_eligible": [True],
        }
    )
    studies = pd.DataFrame(
        {
            "pk_study_id": ["S1"],
            "dose_value": [3.0],
            "dose_unit": ["mg/kg"],
            "route": ["PO"],
            "species": ["rat"],
            "pairing_status": ["resolved"],
        }
    )
    tasks = prepare_pk_tasks(compounds, measurements, studies, features)
    assert "po_cmax" not in tasks
    assert tasks["po_cmax_dose_normalized"].iloc[0]["target_value"] == pytest.approx(100.0)
    assert "dose proportionality" in tasks["po_cmax_dose_normalized"].iloc[0]["target_definition"]


def test_prepare_pk_tasks_rejects_route_mismatched_dose_link() -> None:
    compounds, features = _compound_tables()
    measurements = pd.DataFrame(
        {
            "compound_id": ["C1"],
            "measurement_id": ["M1"],
            "pk_study_id": ["S1"],
            "endpoint": ["auc_0_inf"],
            "value": [1000.0],
            "unit": ["ng*h/mL"],
            "relation": ["="],
            "route": ["IV"],
            "species": ["rat"],
            "pairing_status": ["resolved"],
            "model_eligible": [True],
        }
    )
    studies = pd.DataFrame(
        {
            "pk_study_id": ["S1"],
            "dose_value": [2.0],
            "dose_unit": ["mg/kg"],
            "route": ["PO"],
            "species": ["rat"],
            "pairing_status": ["resolved"],
        }
    )
    with pytest.raises(ValueError, match="route or species mismatch"):
        prepare_pk_tasks(compounds, measurements, studies, features)


def test_prepare_herg_evidence_preserves_one_sided_pic50_bounds() -> None:
    compounds, features = _compound_tables()
    measurements = pd.DataFrame(
        {
            "compound_id": ["C1", "C1"],
            "endpoint": ["herg_ic50", "herg_ic50"],
            "value": [10.0, 30.0],
            "unit": ["uM", "uM"],
            "relation": ["<", ">"],
            "test_concentration_value": [pd.NA, pd.NA],
        }
    )
    _, potency, _ = prepare_herg_evidence(compounds, measurements, features)
    blocker_bound, nonblocker_bound = potency.iloc[0], potency.iloc[1]
    assert blocker_bound["pic50_lower"] == pytest.approx(5.0)
    assert np.isposinf(blocker_bound["pic50_upper"])
    assert np.isneginf(nonblocker_bound["pic50_lower"])
    assert nonblocker_bound["pic50_upper"] == pytest.approx(6.0 - np.log10(30.0))


def test_prepare_herg_evidence_excludes_quarantined_measurements() -> None:
    compounds, features = _compound_tables()
    measurements = pd.DataFrame(
        {
            "compound_id": ["C1", "C1"],
            "endpoint": ["herg_ic50", "herg_ic50"],
            "value": [5.0, 0.01],
            "unit": ["uM", "uM"],
            "relation": ["=", "="],
            "test_concentration_value": [pd.NA, pd.NA],
            "model_eligible": [True, False],
        }
    )
    _, potency, _ = prepare_herg_evidence(compounds, measurements, features)
    assert potency["value"].tolist() == [5.0]


def test_atomic_json_replaces_nonfinite_values_with_null(tmp_path) -> None:
    output = tmp_path / "payload.json"
    atomic_write_json(output, {"missing": float("nan"), "infinite": float("inf"), "value": 2.0})
    text = output.read_text(encoding="utf-8")
    assert "NaN" not in text
    assert "Infinity" not in text
    assert json.loads(text) == {"infinite": None, "missing": None, "value": 2.0}


def test_conformer_population_join_preserves_structure_identity() -> None:
    conformers = pd.DataFrame(
        {
            "state_id": ["STATE-1", "STATE-1"],
            "structure_id": ["STRUCT-1", "STRUCT-1"],
            "conformer_id": ["CONF-1", "CONF-2"],
            "conformer_weight": [0.4, 0.6],
        }
    )
    populations = pd.DataFrame(
        {
            "state_id": ["STATE-1"],
            "structure_id": ["STRUCT-1"],
            "state_weight": [0.25],
        }
    )
    registry = pd.DataFrame({"compound_id": ["CMP-1"], "structure_id": ["STRUCT-1"]})

    merged = _merge_conformer_state_weights(conformers, populations, registry)

    assert merged["structure_id"].tolist() == ["STRUCT-1", "STRUCT-1"]
    assert merged["compound_id"].tolist() == ["CMP-1", "CMP-1"]
    assert merged["ensemble_weight"].tolist() == pytest.approx([0.1, 0.15])


def test_censored_fit_status_rejects_any_nonconverged_fold() -> None:
    assert _censored_fit_status({"fit_converged_fraction": 1.0}) == "evaluated"
    assert _censored_fit_status({"fit_converged_fraction": 0.8}) == "rejected-nonconverged"
    assert _censored_fit_status({}) == "rejected-nonconverged"
