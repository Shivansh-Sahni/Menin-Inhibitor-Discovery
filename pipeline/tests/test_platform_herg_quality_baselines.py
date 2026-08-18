from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from menin_discovery.platform_herg_quality_baselines import (
    HergQualityBaselineError,
    _aggregate_targets,
    _fit_tobit,
    _regression_metrics,
)


def test_aggregate_targets_enforces_entity_exclusive_split(tmp_path) -> None:
    schema = pa.schema(
        [
            pa.field("structure_id", pa.string()),
            pa.field("standardized_smiles", pa.string()),
            pa.field("model_split", pa.string()),
            pa.field("scaffold_group_id", pa.string()),
            pa.field("target_relation", pa.string()),
            pa.field("target_pic50", pa.float64()),
            pa.field("measurement_technology", pa.string()),
            pa.field("eligible", pa.bool_()),
            pa.field("use_as_training_label", pa.bool_()),
        ]
    )
    path = tmp_path / "task.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "structure_id": "S1",
                    "standardized_smiles": "CCO",
                    "model_split": "train",
                    "scaffold_group_id": "G1",
                    "target_relation": "=",
                    "target_pic50": 5.0,
                    "measurement_technology": "patch",
                    "eligible": True,
                    "use_as_training_label": True,
                },
                {
                    "structure_id": "S1",
                    "standardized_smiles": "CCO",
                    "model_split": "test",
                    "scaffold_group_id": "G1",
                    "target_relation": "=",
                    "target_pic50": 5.2,
                    "measurement_technology": "patch",
                    "eligible": True,
                    "use_as_training_label": True,
                },
            ],
            schema=schema,
        ),
        path,
    )
    with pytest.raises(HergQualityBaselineError, match="split conflict"):
        _aggregate_targets(path, "Q1")


def test_aggregate_targets_does_not_upgrade_approximate_to_exact(tmp_path) -> None:
    path = tmp_path / "approximate.parquet"
    pq.write_table(
        pa.table(
            {
                "structure_id": ["S1", "S2"],
                "standardized_smiles": ["CCO", "CCN"],
                "model_split": ["train", "train"],
                "scaffold_group_id": ["G1", "G2"],
                "target_relation": ["~", "="],
                "target_pic50": [5.1, 6.0],
                "measurement_technology": ["patch", "patch"],
                "eligible": [True, True],
                "use_as_training_label": [True, True],
            }
        ),
        path,
    )
    rows = _aggregate_targets(path, "Q2")
    assert [row["structure_id"] for row in rows] == ["S2"]


def test_censored_gaussian_ridge_fits_exact_and_upper_bound() -> None:
    X = np.asarray([[-1.0], [0.0], [1.0], [2.0], [3.0]])
    relation = ["=", "=", "=", "=", "<"]
    target = np.asarray([4.0, 5.0, 6.0, 7.0, np.nan])
    lower = np.full(5, np.nan)
    upper = np.asarray([np.nan, np.nan, np.nan, np.nan, 8.5])
    fitted = _fit_tobit(X, relation, target, lower, upper, alpha=0.1)
    assert fitted["converged"] is True
    assert fitted["sigma"] > 0
    assert np.isfinite(fitted["coef"]).all()


def test_regression_metrics_are_hand_checkable() -> None:
    metrics = _regression_metrics(np.asarray([1.0, 2.0, 3.0]), np.asarray([1.0, 2.0, 3.0]))
    assert metrics["mae"] == 0.0
    assert metrics["rmse"] == 0.0
    assert metrics["r2"] == 1.0
    assert metrics["spearman"] == pytest.approx(1.0)
    constant = _regression_metrics(np.asarray([1.0, 2.0, 3.0]), np.asarray([2.0, 2.0, 2.0]))
    assert constant["pearson"] == 0.0
    assert constant["spearman"] == 0.0
