from __future__ import annotations

import numpy as np
import pandas as pd
from menin_discovery.research_reviewer_audit import (
    _component_labels,
    _effective_count,
    _herg_observation_consistency,
    _metrics,
    _validation_issue_audit,
)


def test_similarity_components_are_outcome_blind_connected_components() -> None:
    similarity = np.asarray(
        [
            [1.0, 0.90, 0.10, 0.10],
            [0.90, 1.0, 0.86, 0.10],
            [0.10, 0.86, 1.0, 0.20],
            [0.10, 0.10, 0.20, 1.0],
        ]
    )

    labels = _component_labels(similarity, 0.85)

    assert labels[0] == labels[1] == labels[2]
    assert labels[3] != labels[0]


def test_effective_group_count_penalizes_imbalanced_groups() -> None:
    balanced = _effective_count(pd.Series(["A", "A", "B", "B"]))
    imbalanced = _effective_count(pd.Series(["A", "A", "A", "B"]))

    assert np.isclose(balanced, 2.0)
    assert imbalanced < balanced


def test_metrics_reward_exact_predictions() -> None:
    observed = np.asarray([1.0, 2.0, 3.0])
    result = _metrics(observed, observed.copy())

    assert result["mae"] == 0.0
    assert result["r2"] == 1.0
    assert result["spearman"] == 1.0


def test_validation_issue_audit_does_not_count_exact_duplicates_as_independent() -> None:
    issues = pd.DataFrame(
        {
            "source": ["internal", "internal"],
            "record_type": ["issue", "issue"],
            "severity": ["error", "error"],
            "code": ["unresolved", "unresolved"],
            "context_json": ['{"compound":"A"}', '{"compound":"A"}'],
        }
    )

    result = _validation_issue_audit(issues).iloc[0]

    assert result["raw_issue_rows"] == 2
    assert result["unique_issue_rows"] == 1
    assert result["exact_duplicate_issue_rows"] == 1
    assert result["affected_compounds"] == 1


def test_herg_consistency_hill_one_identity() -> None:
    measurements = pd.DataFrame(
        {
            "compound_id": ["A", "A"],
            "endpoint": ["herg_ic50", "herg_percent_inhibition"],
            "relation": ["=", "="],
            "value": [10.0, 50.0],
            "test_concentration_value": [np.nan, 10.0],
            "test_concentration_unit": [None, "uM"],
            "source_locator": ["source", "source"],
        }
    )

    detail, summary = _herg_observation_consistency(measurements)

    assert np.isclose(detail.iloc[0]["hill1_expected_inhibition_percent"], 50.0)
    assert np.isclose(summary["mean_absolute_hill1_discrepancy_percent"], 0.0)
