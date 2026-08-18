"""Focused contract tests for the extended local hERG worker."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "run_local_herg_discovery_worker_v3.py"
SPEC = importlib.util.spec_from_file_location("herg_worker_v3", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
worker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(worker)


def test_cli_contract() -> None:
    parser = worker._parser()
    prepare = parser.parse_args(
        ["prepare", "--repo-root", "/repo", "--base-campaign-root", "/base", "--output-root", "/out"]
    )
    assert prepare.command == "prepare"
    run = parser.parse_args(
        [
            "run-unit",
            "--repo-root",
            "/repo",
            "--base-campaign-root",
            "/base",
            "--prepared-root",
            "/prepared",
            "--output-root",
            "/out",
            "--unit-id",
            "u1",
            "--unit-spec",
            '{"operation":"expanded_hpo_tree"}',
        ]
    )
    assert run.unit_id == "u1"


def test_candidate_prefers_adaptively_resolved_recipe() -> None:
    candidate = worker._candidate(
        {
            "candidate": {"candidate_id": "coarse", "engine": "xgboost", "feature_set": "rdkit2d"},
            "resolved_candidate": {
                "candidate_id": "promoted",
                "engine": "lightgbm",
                "feature_set": "fundamental_interactions",
                "params": {"num_leaves": 31},
            },
        }
    )
    assert candidate["candidate_id"] == "promoted"
    assert candidate["engine"] == "lightgbm"
    assert candidate["feature_set"] == "fundamental_interactions"
    assert candidate["params"] == {"num_leaves": 31}


def test_outer_fold_specs_keep_each_folds_own_ids(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "structure_id": ["a", "b", "c", "d"],
            "scaffold_group_id": ["a", "b", "c", "d"],
            "target_pic50": [1.0, 2.0, 3.0, 4.0],
            "rdkit2d__MolLogP": [1.0, 2.0, 3.0, 4.0],
        }
    )
    splits = pd.DataFrame(
        [
            {
                "structure_id": sid,
                "scaffold_group_id": sid,
                "source_partition": "train",
                "outer_fold": fold,
                "outer_role": "heldout" if index == fold else "fit",
                "inner_fold": index % 2,
            }
            for fold in range(2)
            for index, sid in enumerate(frame["structure_id"])
        ]
    )
    split_path = tmp_path / "splits.parquet"
    splits.to_parquet(split_path, index=False)
    monkeypatch.setattr(worker, "_load_exact", lambda *_: (frame.copy(), ["rdkit2d__MolLogP"]))
    monkeypatch.setattr(
        worker, "_base_paths", lambda *_: (tmp_path / "exact", split_path, tmp_path / "registry")
    )
    monkeypatch.setattr(worker, "_model", lambda *_args, **_kwargs: worker.Ridge())
    directory = tmp_path / "unit"
    directory.mkdir()
    result = worker._regression_unit(
        tmp_path,
        tmp_path,
        directory,
        "outer",
        "repeated_seed_tree",
        {
            "operation": "repeated_seed_tree",
            "candidate": {"engine": "ridge", "feature_set": "rdkit2d"},
            "evaluation_stage": "outer",
            "outer_folds": [0, 1],
        },
        1,
    )
    assert result["status"] == "passed"
    predictions = pd.read_parquet(directory / "oof_predictions.parquet")
    assert set(predictions["structure_id"]) == {"a", "b"}
    assert predictions.groupby("fold")["structure_id"].nunique().to_dict() == {0: 1, 1: 1}


def test_classification_threshold_is_cross_fold() -> None:
    observed = np.asarray([0, 0, 1, 1, 0, 0, 1, 1], dtype=int)
    probability = np.asarray([0.01, 0.1, 0.8, 0.9, 0.02, 0.2, 0.7, 0.95])
    folds = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])
    metrics, threshold = worker._classification_metrics(observed, probability, folds, fixed_fpr=0.1)
    assert metrics["threshold_rule"].startswith("cross-fold")
    assert 0 <= threshold <= 1
    assert metrics["pr_auc"] > metrics["prevalence"]


def test_scope_keeps_broad_endpoint_separate() -> None:
    scope = worker._scope("confirmed-WT fixed-dose auxiliary classification")
    assert scope["target_scope"] == "confirmed wild-type hERG"
    assert scope["broad_fixed_dose_pooled_into_pic50"] is False
    assert scope["repository_validation_labels_opened"] is False
    assert scope["repository_test_labels_opened"] is False


def test_nested_composite_requires_disjoint_complete_outer_coverage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(worker, "EXACT_ROWS", 5)
    frames = []
    for fold in range(5):
        frames.append(
            pd.DataFrame(
                {
                    "structure_id": [f"s{fold}"],
                    "scaffold_group_id": [f"g{fold}"],
                    "stage": ["outer"],
                    "fold": [fold],
                    "observed_pic50": [float(fold)],
                    "predicted_pic50": [float(fold) + 0.1],
                }
            )
        )
    composite, metrics = worker._validated_nested_composite(frames)
    assert len(composite) == 5
    assert metrics is not None
    assert metrics["estimate_scope"] == "unbiased_internal_nested_model_selection_estimate"
    duplicated = [frame.copy() for frame in frames]
    duplicated[-1]["structure_id"] = "s0"
    with pytest.raises(worker.WorkerError, match="five disjoint outer folds"):
        worker._validated_nested_composite(duplicated)


def test_censored_unit_owns_arrow_backed_arrays_before_mutation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    rows = 20
    routing = pd.DataFrame(
        {
            "structure_id": [f"s{index}" for index in range(rows)],
            "scaffold_group_id": [f"g{index}" for index in range(rows)],
            "feature_order": np.arange(rows),
            "potency_relation_pic50": ["=" if index % 2 == 0 else ">" for index in range(rows)],
            "potency_pic50_point": [5.0 + index / 100 if index % 2 == 0 else np.nan for index in range(rows)],
            "potency_pic50_lower_bound": [np.nan if index % 2 == 0 else 4.5 for index in range(rows)],
            "potency_pic50_upper_bound": [np.nan] * rows,
        }
    )
    routing_path = tmp_path / "censored_train_routing.parquet"
    routing.to_parquet(routing_path, index=False)
    feature_frame = routing.copy()
    feature_frame["rdkit2d__MolLogP"] = np.linspace(0.5, 4.0, rows)
    monkeypatch.setattr(worker, "GLOBAL_FEATURES", ["rdkit2d__MolLogP"])
    monkeypatch.setattr(worker, "_load_global_features", lambda *_args, **_kwargs: feature_frame.copy())

    class DummyCensoredRidge:
        sigma_ = 0.5

        def __init__(self, **_kwargs: object) -> None:
            self.mean = 5.0

        def fit(self, _x: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> DummyCensoredRidge:
            exact = np.isfinite(lower) & np.isfinite(upper) & np.isclose(lower, upper)
            if exact.any():
                self.mean = float(lower[exact].mean())
            return self

        def predict(self, x: np.ndarray) -> np.ndarray:
            return np.full(len(x), self.mean)

    import menin_discovery.research_modeling as research_modeling

    monkeypatch.setattr(research_modeling, "CensoredGaussianRidge", DummyCensoredRidge)
    output = tmp_path / "unit"
    output.mkdir()
    result = worker._censored_unit(
        tmp_path,
        tmp_path,
        output,
        "censored-smoke",
        {"operation": "censored_sensitivity", "treatment": "interval_likelihood", "outer_folds": 5},
        1,
    )
    assert result["status"] == "passed"
    predictions = pd.read_parquet(output / "censored_oof_predictions.parquet")
    exact = predictions["potency_relation_pic50"].eq("=")
    assert predictions.loc[exact, "lower_bound_pic50"].notna().all()
    assert predictions.loc[exact, "upper_bound_pic50"].notna().all()
    assert np.isfinite(predictions["predicted_pic50"]).all()
