from __future__ import annotations

import numpy as np
import pandas as pd
from menin_discovery.research_local_analysis import (
    _benjamini_hochberg,
    compound_level_predictions,
    grouped_bootstrap_performance,
    grouped_bootstrap_spearman,
)


def test_compound_level_predictions_balances_repeated_evidence_rows() -> None:
    predictions = pd.DataFrame(
        {
            "compound_id": ["A", "A", "B"],
            "model": ["m", "m", "m"],
            "group": ["S1", "S1", "S2"],
            "observed": [1.0, 3.0, 4.0],
            "predicted": [2.0, 2.0, 3.5],
            "lower": [0.0, 0.0, 3.0],
            "upper": [4.0, 4.0, 4.0],
        }
    )

    balanced = compound_level_predictions(
        predictions,
        model="m",
        observed_column="observed",
        predicted_column="predicted",
        interval_lower_column="lower",
        interval_upper_column="upper",
    )

    assert len(balanced) == 2
    first = balanced.set_index("compound_id").loc["A"]
    assert first["observed"] == 2.0
    assert first["predicted"] == 2.0
    assert first["residual"] == 0.0


def test_grouped_bootstrap_performance_reports_fixed_oof_uncertainty() -> None:
    frame = pd.DataFrame(
        {
            "observed": np.arange(12, dtype=float),
            "predicted": np.arange(12, dtype=float) + 0.2,
            "scaffold": np.repeat(["S1", "S2", "S3", "S4"], 3),
        }
    )

    summary, bootstrap = grouped_bootstrap_performance(
        frame,
        bootstrap_replicates=20,
        random_state=4,
    )

    assert len(bootstrap) == 20
    assert np.isclose(summary["mae"], 0.2)
    assert summary["spearman"] == 1.0
    assert summary["mae_lower_95"] <= summary["mae"] <= summary["mae_upper_95"]


def test_grouped_bootstrap_spearman_preserves_perfect_direction() -> None:
    frame = pd.DataFrame(
        {
            "x": np.arange(15, dtype=float),
            "y": np.arange(15, dtype=float) * 2.0,
            "scaffold": np.repeat(["S1", "S2", "S3", "S4", "S5"], 3),
        }
    )

    summary, bootstrap = grouped_bootstrap_spearman(
        frame,
        x="x",
        y="y",
        bootstrap_replicates=20,
        random_state=5,
    )

    assert len(bootstrap) == 20
    assert summary["spearman"] == 1.0
    assert summary["interval_excludes_zero"] is True
    assert summary["within_scaffold_permutation_p"] < 0.05


def test_benjamini_hochberg_is_monotone_and_preserves_missingness() -> None:
    values = pd.Series([0.01, 0.04, np.nan, 0.03, 0.20])

    adjusted = _benjamini_hochberg(values)

    assert np.isnan(adjusted.iloc[2])
    assert adjusted.iloc[0] <= adjusted.iloc[3] <= adjusted.iloc[1]
    assert adjusted.iloc[1] <= adjusted.iloc[4]
