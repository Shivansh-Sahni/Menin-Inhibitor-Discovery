from __future__ import annotations

import numpy as np
import pandas as pd
from menin_discovery.herg_benchmark import ModelSpec
from menin_discovery.herg_validation import (
    Candidate,
    _bootstrap_intervals,
    _fit_sigmoid_calibrator,
    _nearest_tanimoto,
    _optimal_threshold,
    _retained_candidates,
    _scaffold_folds,
    validation_metrics,
)
from scipy import sparse


def test_scaffold_folds_keep_scaffolds_disjoint_and_both_classes_present():
    data = pd.DataFrame(
        {
            "scaffold": np.repeat([f"s{i}" for i in range(10)], 2),
            "herg_blocker_label": [0, 1] * 10,
        }
    )
    folds, metadata = _scaffold_folds(data, n_splits=5, random_state=11)
    assert metadata["resolved_splits"] == 5
    for train, test in folds:
        assert set(data.iloc[train]["scaffold"]).isdisjoint(set(data.iloc[test]["scaffold"]))
        assert set(data.iloc[test]["herg_blocker_label"]) == {0, 1}


def test_sigmoid_calibration_and_threshold_are_finite():
    y = np.array([0, 0, 0, 1, 1, 1])
    raw = np.array([0.15, 0.35, 0.45, 0.55, 0.70, 0.95])
    calibrator = _fit_sigmoid_calibrator(y, raw)
    probability = calibrator.transform(raw)
    threshold, score = _optimal_threshold(y, probability)
    assert np.isfinite(probability).all()
    assert ((probability > 0) & (probability < 1)).all()
    assert 0.05 <= threshold <= 0.95
    assert 0.5 <= score <= 1.0


def test_nearest_tanimoto_supports_external_and_leave_one_out_queries():
    reference = sparse.csr_matrix([[1, 1, 0], [0, 1, 1]], dtype=np.float32)
    query = sparse.csr_matrix([[1, 1, 0], [1, 0, 1]], dtype=np.float32)
    external = _nearest_tanimoto(query, reference)
    leave_one_out = _nearest_tanimoto(reference, reference, exclude_diagonal=True)
    assert np.allclose(external, [1.0, 1 / 3])
    assert np.allclose(leave_one_out, [1 / 3, 1 / 3])


def test_successive_retention_preserves_best_member_of_each_requested_family():
    candidates = []
    scores = {}
    for family_index, family in enumerate(("svm", "random_forest", "rnn")):
        for complexity_index, complexity in enumerate(("simple", "complex")):
            spec = ModelSpec(
                family=family,
                complexity=complexity,
                parameters={"level": complexity_index},
                feature_sets=("morgan_1024_r2",),
            )
            candidate = Candidate(spec=spec, feature_set="morgan_1024_r2")
            candidates.append(candidate)
            scores[candidate.key] = family_index + complexity_index / 10
    retained = _retained_candidates(
        candidates,
        scores,
        keep_fraction=0.2,
        minimum_candidates=1,
    )
    assert {candidate.spec.family for candidate in retained} == {"svm", "random_forest", "rnn"}
    for family in ("svm", "random_forest", "rnn"):
        assert any(
            candidate.spec.family == family and candidate.spec.complexity == "complex"
            for candidate in retained
        )


def test_scaffold_bootstrap_emits_intervals_for_all_core_metrics():
    predictions = pd.DataFrame(
        {
            "regime": ["confidential_only"] * 12,
            "strategy": ["selected"] * 12,
            "scaffold": np.repeat(["a", "b", "c", "d"], 3),
            "observed_label": [0, 1, 1] * 4,
            "probability": [0.2, 0.7, 0.8, 0.1, 0.6, 0.9, 0.3, 0.65, 0.75, 0.25, 0.7, 0.85],
            "predicted_label": [0, 1, 1] * 4,
        }
    )
    intervals = _bootstrap_intervals(predictions, iterations=100, random_state=3)
    assert {"roc_auc", "balanced_accuracy", "brier"}.issubset(set(intervals["metric"]))
    assert (intervals["successful_resamples"] > 0).all()
    metrics = validation_metrics(
        predictions["observed_label"].to_numpy(),
        predictions["probability"].to_numpy(),
        predicted_label=predictions["predicted_label"].to_numpy(),
    )
    assert metrics["balanced_accuracy"] == 1.0
