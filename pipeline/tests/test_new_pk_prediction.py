from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from predict_pk_new_compounds import _standardize_pk_input, predict  # noqa: E402


def test_pk_prediction_input_rejects_outcomes(tmp_path: Path) -> None:
    source = tmp_path / "not_blind.csv"
    pd.DataFrame(
        {
            "compound_id": ["NEW-1"],
            "smiles": ["CCO"],
            "vdss": [2.0],
        }
    ).to_csv(source, index=False)

    with pytest.raises(ValueError, match="not blind"):
        _standardize_pk_input(source)


def test_end_to_end_pk_prediction_preserves_scientific_gates(
    tmp_path: Path,
) -> None:
    source = Path("research/templates/new_herg_pk_prediction_example.csv")
    if not source.exists():
        source = Path(__file__).resolve().parents[2] / source
    output = tmp_path / "pk_prediction"

    summary, derived = predict(source, output)

    assert set(summary["endpoint"]) == {
        "iv_auc_dose_normalized",
        "po_auc_dose_normalized",
        "po_cmax_dose_normalized",
        "po_tmax",
        "vdss",
    }
    assert summary["max_train_tanimoto"].between(0, 1).all()
    assert (
        summary.loc[summary["endpoint"].eq("po_tmax"), "prediction_status"].iloc[0]
        == "withheld_model_has_no_reliable_rank_signal"
    )
    assert (
        summary.loc[~summary["endpoint"].eq("po_tmax"), "prediction_status"]
        .eq("eligible_same_series_discovery_hypothesis")
        .all()
    )
    assert (
        derived.loc[
            derived["derived_endpoint"].eq("systemic_clearance_from_iv_auc"),
            "status",
        ].iloc[0]
        == "derived_closure_not_independent_model"
    )
    assert {"iv_auc_at_planned_dose", "po_auc_at_planned_dose"}.issubset(set(derived["derived_endpoint"]))
    assert (output / "pk_predictions_long.parquet").exists()
    assert (output / "pk_endpoint_summary.csv").exists()
    assert (output / "pk_derived_closure.csv").exists()
    assert (output / "pk_prediction_report.md").exists()


def test_out_of_domain_structure_is_withheld(tmp_path: Path) -> None:
    source = tmp_path / "unrelated.csv"
    pd.DataFrame(
        {
            "compound_id": ["UNRELATED-1"],
            "smiles": ["CCO"],
            "same_series_confirmed": [True],
        }
    ).to_csv(source, index=False)

    summary, _ = predict(source, tmp_path / "prediction")

    assert summary["applicability_domain"].eq("outside").all()
    assert summary["prediction_status"].str.startswith("withheld_").all()
