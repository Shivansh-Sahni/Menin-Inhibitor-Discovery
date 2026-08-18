from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from menin_discovery.research_ascentage import (
    FIRST_DERIVED_PREDICTION_COLUMN,
    load_ascentage_source,
)

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from run_program_health_audit import (  # noqa: E402
    _interval_score,
    _threshold_metrics_from_predictions,
    _wilson,
)


def test_wilson_interval_contains_observed_fraction() -> None:
    lower, upper = _wilson(6, 9)
    assert lower < 6 / 9 < upper
    assert np.isclose(lower, 0.35420213558039613)
    assert np.isclose(upper, 0.8794161816331391)


def test_interval_score_penalizes_misses_beyond_width() -> None:
    score = _interval_score(
        np.asarray([0.5, 2.0]),
        np.asarray([0.0, 0.0]),
        np.asarray([1.0, 1.0]),
        alpha=0.10,
    )
    assert np.isclose(score[0], 1.0)
    assert np.isclose(score[1], 21.0)


def test_strict_greater_than_30_um_is_definite_nonblocker_at_30_um() -> None:
    threshold = 6.0 - np.log10(30.0)
    predictions = pd.DataFrame(
        {
            "herg_pic50_relation": ["=", "<"],
            "herg_pic50_value": [5.0, threshold],
            "herg_pic50_upper_bound": [5.0, threshold],
            "complete_feature_oof_sigma": [0.5, 0.5],
            "complete_feature_predicted_pic50": [5.0, 4.0],
        }
    )
    metrics = _threshold_metrics_from_predictions(predictions, threshold)
    assert metrics["n"] == 2
    assert metrics["n_blockers"] == 1
    assert metrics["n_nonblockers"] == 1


def test_ascentage_recovery_removes_every_prediction_column(tmp_path: Path) -> None:
    recovery = Path("research/reports/pk_herg/ascentage_herg_extension/predictions.parquet")
    if not recovery.exists():
        recovery = Path(__file__).resolve().parents[2] / recovery
    recovered = load_ascentage_source(
        tmp_path / "absent.parquet",
        recovery_artifact=recovery,
    )
    assert len(recovered) == 76
    assert FIRST_DERIVED_PREDICTION_COLUMN not in recovered
    assert not any(column.startswith("augmented_") for column in recovered)
    assert recovered["herg_ic50_relation"].eq(">").sum() == 7
