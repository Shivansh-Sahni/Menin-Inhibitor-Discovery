from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from menin_discovery.platform_metrics import (
    benjamini_hochberg,
    binary_classification_metrics,
    bootstrap_metric_intervals,
    calibration_bins,
    censored_regression_metrics,
    enrichment_metrics,
    metric_reporting_registry,
    multiclass_classification_metrics,
    paired_bootstrap_difference,
    prediction_interval_metrics,
    regression_metrics,
    subgroup_metrics,
)


def test_regression_metrics_match_hand_calculation() -> None:
    result = regression_metrics([1, 2, 3], [1, 2, 4])
    assert result["mae"] == pytest.approx(1 / 3)
    assert result["median_absolute_error"] == 0
    assert result["mse"] == pytest.approx(1 / 3)
    assert result["rmse"] == pytest.approx(math.sqrt(1 / 3))
    assert result["r2"] == pytest.approx(0.5)
    assert result["mean_signed_error_prediction_minus_observation"] == pytest.approx(1 / 3)


def test_binary_metrics_match_confusion_matrix_and_degenerate_is_explicit() -> None:
    result = binary_classification_metrics([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8])
    assert result["confusion_matrix"] == {"tn": 2, "fp": 0, "fn": 1, "tp": 1}
    assert result["accuracy"] == 0.75
    assert result["recall_sensitivity"] == 0.5
    assert result["specificity"] == 1.0
    assert result["precision"] == 1.0
    assert result["f1"] == pytest.approx(2 / 3)
    assert result["n_events"] == 2
    assert result["n_non_events"] == 2

    degenerate = binary_classification_metrics([0, 0, 0], [0.1, 0.2, 0.3])
    assert degenerate["estimability_status"] == "one_class_probability_metrics_only"
    assert degenerate["roc_auc"] is None
    assert degenerate["balanced_accuracy"] is None
    assert degenerate["cohen_kappa"] is None
    assert degenerate["matthews_correlation_coefficient"] is None


def test_calibration_bins_weighted_gap_is_hand_checkable() -> None:
    bins = calibration_bins([0, 1], [0.25, 0.75], n_bins=2)
    assert bins["n"].sum() == 2
    assert bins["weighted_absolute_gap"].sum() == pytest.approx(0.25)


def test_censoring_and_prediction_interval_metrics() -> None:
    censored = censored_regression_metrics(
        [1.0, -np.inf, 3.0],
        [1.0, 2.0, np.inf],
        [1.5, 1.5, 2.0],
    )
    assert censored["fraction_predictions_consistent_with_bounds"] == pytest.approx(1 / 3)
    assert censored["mean_interval_violation"] == pytest.approx(0.5)
    assert censored["exact_subset_mae"] == 0.5

    intervals = prediction_interval_metrics(
        [1.0, 2.0, 3.0], [0.5, 1.5, 3.5], [1.5, 2.5, 4.0], nominal_coverage=0.8
    )
    assert intervals["empirical_coverage"] == pytest.approx(2 / 3)
    assert intervals["mean_interval_width"] == pytest.approx(5 / 6)


def test_enrichment_and_bh_adjustment() -> None:
    enrichment = enrichment_metrics([1, 0, 1, 0], [0.9, 0.8, 0.7, 0.1], fractions=[0.5])
    assert enrichment["top_0p5_hit_rate"] == 0.5
    assert enrichment["enrichment_factor_0p5"] == 1.0
    adjusted = benjamini_hochberg([0.01, 0.04, 0.03, np.nan])
    assert adjusted[:3] == pytest.approx([0.03, 0.04, 0.04])
    assert np.isnan(adjusted[3])


def test_group_bootstrap_is_deterministic() -> None:
    first = bootstrap_metric_intervals(
        [1, 2, 3, 4],
        [1, 2.5, 2.5, 4],
        regression_metrics,
        groups=["a", "a", "b", "b"],
        iterations=30,
        seed=7,
    )
    second = bootstrap_metric_intervals(
        [1, 2, 3, 4],
        [1, 2.5, 2.5, 4],
        regression_metrics,
        groups=["a", "a", "b", "b"],
        iterations=30,
        seed=7,
    )
    assert first == second
    assert first["resampling_unit"] == "group"


def test_subgroups_require_independent_groups_and_events() -> None:
    frame = pd.DataFrame(
        {
            "source": ["a"] * 8 + ["b"] * 8,
            "group": [f"g{index}" for index in range(16)],
            "target": [0] * 8 + [0, 1] * 4,
            "prediction": np.linspace(0.1, 0.9, 16),
        }
    )
    result = subgroup_metrics(
        frame,
        subgroup_columns=["source"],
        target_column="target",
        prediction_column="prediction",
        task_type="classification",
        minimum_size=4,
        group_column="group",
        minimum_independent_groups=4,
        minimum_events_per_class=2,
    )
    source_a = result[result["subgroup_value"] == "a"].iloc[0]
    assert source_a["status"] == "suppressed_insufficient_event_or_nonevent_support"
    assert source_a["n_events"] == 0
    assert source_a["n_non_events"] == 8


def test_brier_comparison_validates_binary_probability_contract() -> None:
    with pytest.raises(ValueError, match="binary"):
        paired_bootstrap_difference([0, 2], [0.1, 0.8], [0.2, 0.7], metric="brier_score", iterations=5)
    with pytest.raises(ValueError, match="probabilities"):
        paired_bootstrap_difference([0, 1], [0.1, 1.2], [0.2, 0.7], metric="brier_score", iterations=5)


def test_metric_registry_declares_roles() -> None:
    registry = metric_reporting_registry()
    assert {"primary", "secondary", "diagnostic"}.issubset(set(registry["role"]))
    assert (
        (registry["task_family"] == "binary_classification")
        & (registry["metric"] == "average_precision_pr_auc")
        & (registry["role"] == "primary")
    ).any()
    binary_primary = registry[
        (registry["task_family"] == "binary_classification") & (registry["role"] == "primary")
    ]
    assert binary_primary["metric"].tolist() == ["average_precision_pr_auc"]


def test_empty_degenerate_and_domain_edge_contracts() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        multiclass_classification_metrics([], np.empty((0, 2)), class_labels=["a", "b"])
    with pytest.raises(ValueError, match="at least one"):
        censored_regression_metrics([], [], [])
    with pytest.raises(ValueError, match="at least one"):
        prediction_interval_metrics([], [], [], nominal_coverage=0.9)
    with pytest.raises(ValueError, match="iterations"):
        paired_bootstrap_difference([0, 1], [0.1, 0.9], [0.2, 0.8], metric="brier_score", iterations=0)
    with pytest.raises(ValueError, match="missing or blank"):
        paired_bootstrap_difference(
            [0, 1], [0.1, 0.9], [0.2, 0.8], metric="brier_score", groups=["", "g"], iterations=2
        )

    from menin_discovery.platform_metrics import applicability_domain_metrics

    with pytest.raises(ValueError, match="outside"):
        applicability_domain_metrics([0, 1], [0.1, 0.9], [-0.1, 1.1], task_type="classification")
