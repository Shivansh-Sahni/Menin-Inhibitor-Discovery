from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT = Path(__file__).parents[1] / "scripts/run_local_herg_feature_lattice_campaign_v8.py"
SPEC = importlib.util.spec_from_file_location("herg_v8", SCRIPT)
assert SPEC and SPEC.loader
V8 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = V8
SPEC.loader.exec_module(V8)


def test_complete_lattice_and_unique_profiles() -> None:
    assert len(V8.BLOCKS) == 11
    assert len(V8._all_masks()) == 2048
    assert V8._mask_blocks(0) == ()
    assert set(V8._mask_blocks(2047)) == set(V8.BLOCKS)
    payloads = [profile.payload() for profile in V8._profiles()]
    assert len(payloads) == 10
    assert len({V8._digest(payload) for payload in payloads}) == 10
    assert {profile.engine for profile in V8._profiles()} == {"xgboost", "lightgbm"}


def test_exact_shapley_recovers_additive_game() -> None:
    weights = np.array([1.25, -0.5, 2.0])
    values = np.array([sum(weights[i] for i in range(3) if mask & (1 << i)) for mask in range(8)])
    np.testing.assert_allclose(V8._shapley_values(values, 3), weights)
    np.testing.assert_allclose(V8._pairwise_banzhaf(values, 3), 0.0)


def test_pairwise_banzhaf_recovers_synergy() -> None:
    values = np.zeros(8)
    for mask in range(8):
        values[mask] = float(bool(mask & 1)) + 3.0 * float(bool(mask & 1) and bool(mask & 2))
    result = V8._pairwise_banzhaf(values, 3)
    assert math.isclose(result[0, 1], 3.0)
    assert math.isclose(result[0, 2], 0.0)


def test_promotion_is_deterministic_and_diverse() -> None:
    rows = []
    for mask in range(2048):
        rows.append({"coalition_mask": mask, "coalition_size": mask.bit_count(), "mae": 0.4 + mask / 1e6})
    promoted = V8._promote(pd.DataFrame(rows))
    assert promoted == V8._promote(pd.DataFrame(rows))
    assert len(promoted) == V8.PROMOTED_COALITIONS
    assert 0 in promoted
    assert (1 << len(V8.BLOCKS)) - 1 in promoted
    for index in range(len(V8.BLOCKS)):
        assert any(mask & (1 << index) for mask in promoted)


def test_final_recipe_requires_all_outer_contexts() -> None:
    profiles = V8._profiles()
    units = []
    for outer in range(5):
        for mask, profile, score in (
            (3, profiles[0], 0.44),
            (7, profiles[1], 0.45),
        ):
            units.append(
                {
                    "unit_spec": {
                        "outer_fold": outer,
                        "coalition_mask": mask,
                        "profile": profile.payload(),
                    },
                    "metrics": {"selection_score": score},
                }
            )
    units.append(
        {
            "unit_spec": {
                "outer_fold": 0,
                "coalition_mask": 15,
                "profile": profiles[2].payload(),
            },
            "metrics": {"selection_score": 0.01},
        }
    )
    mask, profile, ranking = V8._select_final_recipe(units)
    assert mask == 3
    assert profile.profile_id == profiles[0].profile_id
    assert ranking.outer_contexts.eq(5).all()


def test_v7_prediction_column_compatibility(tmp_path: Path) -> None:
    path = tmp_path / "v7.parquet"
    pd.DataFrame({"structure_id": ["a", "b"], "predicted_pic50": [5.0, 6.0]}).to_parquet(path)
    result = V8._load_v7_accuracy_predictions(path)
    assert result.columns.tolist() == ["structure_id", "accuracy_prediction"]
    assert result.accuracy_prediction.tolist() == [5.0, 6.0]
