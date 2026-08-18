from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "analyze_local_herg_feature_lattice_v81.py"
SPEC = importlib.util.spec_from_file_location("lattice_v81", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_bh_is_monotone_at_sorted_p_values() -> None:
    result = MODULE._bh(np.array([0.04, 0.01, 0.03]))
    assert result == pytest.approx([0.04, 0.03, 0.04])


def test_scaffold_inference_is_deterministic_and_row_weighted() -> None:
    values = np.array([0.2, 0.2, 0.2, 0.8])
    scaffolds = np.array(["a", "a", "a", "b"])
    first = MODULE._scaffold_inference(values, scaffolds, seed=11, replicates=2_000)
    second = MODULE._scaffold_inference(values, scaffolds, seed=11, replicates=2_000)
    assert first == second
    assert first["mean"] == pytest.approx(0.35)
    assert first["ci95_lower"] > 0


def test_heavy_bins_are_prespecified() -> None:
    frame = pd.DataFrame(
        {
            "molecular_weight": [299.0, 400.0, 600.0, 800.0, 1200.0],
            "rotatable_bonds": [1, 4, 7, 11, 2],
            "mol_logp": [1, 3, 5, 7, 0],
            "tpsa": [20, 60, 100, 150, 10],
            "observed_pic50": [3, 4.5, 5.5, 6.5, 5],
        }
    )
    result = MODULE._add_bins(frame)
    assert result.molecular_weight_regime.tolist() == [
        "mw_le_300",
        "mw_300_500",
        "mw_500_700",
        "mw_700_1000",
        "mw_gt_1000",
    ]
    assert result.heavy_molecule.tolist() == [False, False, True, True, True]
