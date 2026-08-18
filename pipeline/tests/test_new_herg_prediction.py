from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from evaluate_blind_herg_predictions import evaluate  # noqa: E402
from predict_herg_new_compounds import _standardize_input, predict  # noqa: E402


def test_prediction_input_rejects_outcomes_to_preserve_blinding(tmp_path: Path) -> None:
    source = tmp_path / "not_blind.csv"
    pd.DataFrame(
        {
            "compound_id": ["NEW-1"],
            "smiles": ["CCO"],
            "herg_ic50_um": [3.0],
        }
    ).to_csv(source, index=False)

    with pytest.raises(ValueError, match="not blind"):
        _standardize_input(source)


def test_prediction_input_audits_series_and_stereochemistry(tmp_path: Path) -> None:
    source = tmp_path / "structures.csv"
    pd.DataFrame(
        {
            "compound_id": ["NEW-1", "NEW-2"],
            "smiles": ["C/C=C/C", "CC=CC"],
            "same_series_confirmed": ["yes", ""],
            "series_id": ["SERIES-A", ""],
        }
    ).to_csv(source, index=False)

    result = _standardize_input(source)

    assert result.loc[0, "same_series_confirmed"] == "true"
    assert result.loc[0, "unresolved_stereoelement_count"] == 0
    assert result.loc[1, "same_series_confirmed"] == "not_provided"
    assert result.loc[1, "unresolved_stereoelement_count"] == 1
    assert result.loc[1, "stereochemistry_status"] == "unresolved_potential_stereochemistry"


def test_end_to_end_new_compound_prediction_emits_safeguards(tmp_path: Path) -> None:
    source = Path("research/templates/new_herg_prediction_example.csv")
    if not source.exists():
        source = Path(__file__).resolve().parents[2] / source
    output = tmp_path / "prediction"

    result = predict(source, output)
    row = result.iloc[0]

    assert row["same_series_confirmed"] == "true"
    assert row["unresolved_stereoelement_count"] == 0
    assert row["prediction_eligibility"] == "eligible_same_series_discovery_hypothesis"
    assert row["decision_status"].endswith("requires_measurement")
    assert pd.notna(row["ensemble_predicted_pic50"])
    assert row["ensemble_predictive_sigma"] > 0
    assert row["ensemble_predicted_herg_ic50_um"] == pytest.approx(
        10 ** (6 - row["ensemble_predicted_pic50"])
    )
    assert row["neighbor_structures_ge_0p80"] > 0
    assert (
        row["conservative_model_envelope_pic50_lower"]
        <= row["retained_model_pic50_min"]
        <= row["retained_model_pic50_max"]
        <= row["conservative_model_envelope_pic50_upper"]
    )
    assert row["threshold_interval_status"] == "envelope_crosses_threshold"
    assert not row["exact_augmented_training_overlap"]
    assert (output / "predictions.csv").exists()
    assert (output / "predictions.parquet").exists()
    assert (output / "prediction_report.md").exists()
    assert (output / "prediction_summary.json").exists()


def test_blind_evaluator_preserves_greater_than_30_as_censoring(
    tmp_path: Path,
) -> None:
    predictions_path = tmp_path / "frozen_predictions.csv"
    outcomes_path = tmp_path / "released_outcomes.csv"
    output = tmp_path / "evaluation"
    eligibility = "eligible_same_series_discovery_hypothesis"
    pd.DataFrame(
        {
            "compound_id": ["A", "B", "C", "D"],
            "prediction_eligibility": [eligibility] * 4,
            "global_control_predicted_pic50": [5.8, 4.7, 4.2, 5.4],
            "global_control_pic50_lower": [5.2, 4.1, 3.6, 4.8],
            "global_control_pic50_upper": [6.4, 5.3, 4.8, 6.0],
            "global_control_blocker_probability": [0.9, 0.3, 0.1, 0.7],
            "complete_feature_predicted_pic50": [5.9, 4.6, 4.3, 5.3],
            "complete_feature_pic50_lower": [5.3, 4.0, 3.7, 4.7],
            "complete_feature_pic50_upper": [6.5, 5.2, 4.9, 5.9],
            "complete_feature_blocker_probability": [0.92, 0.25, 0.12, 0.65],
        }
    ).to_csv(predictions_path, index=False)
    pd.DataFrame(
        {
            "compound_id": ["A", "B", "C", "D"],
            "herg_ic50_relation": ["=", "=", ">", "="],
            "herg_ic50_value_um": [1.0, 20.0, 30.0, 3.0],
        }
    ).to_csv(outcomes_path, index=False)

    joined, metrics, validation = evaluate(
        predictions_path,
        outcomes_path,
        output,
    )

    censored = joined.loc[joined["compound_id"].eq("C")].iloc[0]
    assert censored["herg_pic50_relation"] == "<"
    assert censored["herg_pic50_upper"] == pytest.approx(4.522879, rel=1e-6)
    assert censored["definitive_blocker_at_pic50_5"] == 0
    assert validation["models_refit"] is False
    assert validation["exact_outcomes"] == 3
    assert validation["censored_outcomes"] == 1
    assert set(metrics["model"]) == {"global_control", "complete_feature"}
    assert (output / "evaluation_report.md").exists()
