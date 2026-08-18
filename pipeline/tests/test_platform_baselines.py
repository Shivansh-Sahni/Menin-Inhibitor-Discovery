from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from menin_discovery.platform_baselines import (
    BaselineConfig,
    build_error_analysis,
    class_imbalance_options,
    numeric_target_leakage_scan,
    run_diagnostic_baselines,
    run_label_permutation_control,
    validate_prediction_schema,
)

DIGEST_ARGUMENTS = {
    "dataset_sha256": "a" * 64,
    "split_manifest_sha256": "b" * 64,
    "feature_artifact_sha256": "c" * 64,
}


def _classification_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_train = np.asarray([[index, index % 3, index % 2] for index in range(30)], dtype=float)
    y_train = (X_train[:, 0] > 14).astype(int)
    X_test = np.asarray([[index, index % 3, index % 2] for index in range(30, 40)], dtype=float)
    y_test = np.ones(10, dtype=int)
    return X_train, y_train, X_test, y_test


def test_classification_baselines_emit_distinct_observation_and_prediction_types() -> None:
    X_train, y_train, X_test, y_test = _classification_data()
    config = BaselineConfig(task_type="classification", tree_estimators=8)
    result = run_diagnostic_baselines(
        X_train,
        y_train,
        X_test,
        y_test,
        eval_record_ids=[f"test-{index}" for index in range(10)],
        config=config,
        task_id="herg_binary",
        split_name="molecule_grouped_v1",
        feature_set_name="tiny_numeric",
        train_outcome_kinds=["curated_assertion"] * len(y_train),
        eval_outcome_kinds=["curated_assertion"] * len(y_test),
        **DIGEST_ARGUMENTS,
    )
    assert set(result.metrics["model_name"]) == {"dummy_prior", "logistic_fixed", "extra_trees_restrained"}
    assert set(result.predictions["outcome_kind"]) == {"curated_assertion"}
    assert set(result.predictions["prediction_origin"]) == {"computational_prediction"}
    assert result.metadata["hyperparameter_sweep_performed"] is False
    validate_prediction_schema(result.predictions)


def test_invalid_or_unlineaged_outcome_kind_fails_before_fit() -> None:
    X_train, y_train, X_test, y_test = _classification_data()
    config = BaselineConfig(task_type="classification", include_tree_baseline=False)
    with pytest.raises(ValueError, match="Prohibited"):
        run_diagnostic_baselines(
            X_train,
            y_train,
            X_test,
            y_test,
            eval_record_ids=[f"test-{index}" for index in range(10)],
            config=config,
            task_id="task",
            split_name="split",
            feature_set_name="features",
            train_outcome_kinds=["prediction"] * len(y_train),
            eval_outcome_kinds=["prediction"] * len(y_test),
            **DIGEST_ARGUMENTS,
        )
    with pytest.raises(ValueError, match="lineage"):
        run_diagnostic_baselines(
            X_train,
            y_train,
            X_test,
            y_test,
            eval_record_ids=[f"test-{index}" for index in range(10)],
            config=config,
            task_id="task",
            split_name="split",
            feature_set_name="features",
            train_outcome_kinds=["derived"] * len(y_train),
            eval_outcome_kinds=["derived"] * len(y_test),
            **DIGEST_ARGUMENTS,
        )


def test_regression_baselines_and_error_ranking() -> None:
    X_train = np.arange(60, dtype=float).reshape(20, 3)
    y_train = X_train[:, 0] / 10
    X_test = np.arange(60, 90, dtype=float).reshape(10, 3)
    y_test = X_test[:, 0] / 10
    result = run_diagnostic_baselines(
        X_train,
        y_train,
        X_test,
        y_test,
        eval_record_ids=[f"r{index}" for index in range(10)],
        config=BaselineConfig(task_type="regression", include_tree_baseline=False),
        task_id="binding_log",
        split_name="scaffold_v1",
        feature_set_name="numeric",
        train_outcome_kinds=["experimental_summary"] * len(y_train),
        eval_outcome_kinds=["experimental_summary"] * len(y_test),
        **DIGEST_ARGUMENTS,
    )
    errors = build_error_analysis(result.predictions)
    assert set(errors["error_type"]) == {"regression_error"}
    assert errors.groupby("model_name")["error_rank_within_model"].max().eq(10).all()


def test_target_leakage_scan_and_imbalance_options() -> None:
    target = np.asarray([0.0, 1.0, 2.0, 3.0])
    features = pd.DataFrame({"safe": [1.0, 0.0, 1.0, 0.0], "copy": target, "near": target * 2})
    scan = numeric_target_leakage_scan(features, target)
    copy = scan.set_index("feature").loc["copy"]
    assert bool(copy["exact_target_copy"])
    assert bool(copy["requires_review"])
    assert bool(scan.set_index("feature").loc["near", "high_absolute_correlation"])

    options = class_imbalance_options([0, 0, 0, 1])
    assert options["class_counts"] == {"0": 3, "1": 1}
    assert options["synthetic_structured_oversampling_default"] == "prohibited"
    empty = numeric_target_leakage_scan(pd.DataFrame({"text": ["a", "b", "c", "d"]}), target)
    assert empty.empty
    assert "requires_review" in empty.columns


def test_permutation_control_is_deterministic_and_labeled_sanity_only() -> None:
    X_train, y_train, X_test, y_test = _classification_data()
    config = BaselineConfig(task_type="classification", include_tree_baseline=False)
    first = run_label_permutation_control(X_train, y_train, X_test, y_test, config=config)
    second = run_label_permutation_control(X_train, y_train, X_test, y_test, config=config)
    assert first == second
    assert first["control"] == "single_training_label_permutation"
    assert "not a p-value" in first["interpretation"]
