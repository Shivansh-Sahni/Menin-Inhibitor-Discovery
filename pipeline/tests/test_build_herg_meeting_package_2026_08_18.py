from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT = Path(__file__).parents[1] / "scripts" / "build_herg_meeting_package_2026_08_18.py"
SPEC = importlib.util.spec_from_file_location("meeting_package", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_metric_row() -> None:
    result = MODULE.metric_row(np.array([4.0, 5.0, 6.0]), np.array([4.0, 5.5, 5.0]))
    assert result["n"] == 3
    assert result["mae"] == 0.5
    assert result["within_0p5"] == 2 / 3


def test_threshold_metrics_are_complete() -> None:
    result = MODULE.threshold_metrics(np.array([4.0, 5.0, 6.0, 7.0]), np.array([4.2, 5.4, 6.2, 6.7]), 6.0)
    assert result["tp"] == 2
    assert result["tn"] == 2
    assert result["mcc"] == 1.0


def test_scaffold_bootstrap_delta_preserves_row_weighted_estimand() -> None:
    frame = pd.DataFrame(
        {
            "scaffold_group_id": ["a", "a", "b"],
            "observed_pic50": [5.0, 6.0, 4.0],
            "challenger": [5.0, 6.0, 4.0],
            "reference": [6.0, 7.0, 5.0],
        }
    )
    result = MODULE.scaffold_bootstrap_delta(frame, "challenger", "reference", replicates=100, seed=1)
    assert result["delta_mae"] == -1.0
    assert result["scaffolds"] == 2
