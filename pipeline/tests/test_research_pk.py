from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from menin_discovery.research_pk import (
    derive_pk_prediction_views,
    one_compartment_iv_profile,
    recompute_pk_closure,
)


def test_pk_closure_equations() -> None:
    frame = pd.DataFrame(
        {
            "iv_dose_mg_kg": [1.0],
            "po_dose_mg_kg": [3.0],
            "iv_auc0_inf_ng_h_ml": [1000.0],
            "po_auc0_inf_ng_h_ml": [300.0],
            "reported_cl_ml_kg_min": [16.6667],
            "reported_f_percent": [10.0],
        }
    )
    result = recompute_pk_closure(frame)
    assert result.loc[0, "cl_recomputed_ml_kg_min"] == pytest.approx(16.6667, rel=1e-4)
    assert result.loc[0, "f_recomputed_percent"] == pytest.approx(10.0)


def test_iv_profile_decreases() -> None:
    values = one_compartment_iv_profile(
        np.array([0.0, 1.0, 2.0]), dose_mg_kg=1.0, clearance_ml_kg_min=10.0, vdss_l_kg=2.0
    )
    assert np.all(np.diff(values) < 0)


def test_prediction_closure_views_are_derived_from_auc_only() -> None:
    common = {
        "compound_id": "C1",
        "domain_status": "inside",
        "model_name": "ridge",
        "uncertainty": 0.0,
        "unit": "ng*h/mL per mg/kg",
    }
    predictions = pd.DataFrame(
        [
            {
                **common,
                "endpoint": "rat_iv_auc_dose_normalized",
                "mean": 1000.0,
                "lower": 800.0,
                "upper": 1250.0,
            },
            {
                **common,
                "endpoint": "rat_po_auc_dose_normalized",
                "mean": 200.0,
                "lower": 150.0,
                "upper": 300.0,
            },
        ]
    )
    result = derive_pk_prediction_views(predictions)
    clearance = result[result["endpoint"] == "rat_iv_clearance_ml_kg_min"].iloc[0]
    bioavailability = result[result["endpoint"] == "rat_bioavailability_closure_percent"].iloc[0]
    assert clearance["mean"] == pytest.approx(16.6667, rel=1e-4)
    assert bioavailability["mean"] == pytest.approx(20.0)
    assert bioavailability["lower"] == pytest.approx(12.0)
    assert bioavailability["upper"] == pytest.approx(37.5)
    assert "not an independent" in bioavailability["estimate_semantics"]
