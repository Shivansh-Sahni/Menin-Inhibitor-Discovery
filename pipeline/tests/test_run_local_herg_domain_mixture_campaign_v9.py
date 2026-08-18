from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPT = Path(__file__).parents[1] / "scripts" / "run_local_herg_domain_mixture_campaign_v9.py"
SPEC = importlib.util.spec_from_file_location("herg_v9", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
v9: Any = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = v9
SPEC.loader.exec_module(v9)


def test_candidate_plan_is_material_and_contains_required_specialists() -> None:
    plan = v9._candidate_plan(False)
    payloads = [v9._digest(v9.asdict(candidate)) for candidate in plan]
    assert len(plan) >= 20
    assert len(payloads) == len(set(payloads))
    identifiers = {candidate.candidate_id for candidate in plan}
    assert {
        "xgb_v2_anchor",
        "xgb_selective_physics",
        "xgb_heavy_flexible",
        "xgb_potency_tail",
        "xgb_cliff_risk",
        "xgb_reliability_weighted",
        "xgb_hierarchical_target",
        "lgb_huber",
        "extratrees_rdkit2d",
    } <= identifiers


def test_connected_components_join_pair_graph() -> None:
    frame = pd.DataFrame(
        {
            "structure_id_a": ["a", "b", "x"],
            "structure_id_b": ["b", "c", "y"],
        }
    )
    components = v9._components(frame)
    assert components["a"] == components["b"] == components["c"]
    assert components["x"] == components["y"]
    assert components["a"] != components["x"]


def test_scaffold_bootstrap_uses_row_weighted_delta() -> None:
    frame = pd.DataFrame(
        {
            "scaffold_group_id": ["a", "a", "b"],
            "observed_pic50": [5.0, 5.0, 5.0],
            "challenger": [5.0, 5.0, 6.0],
            "reference": [6.0, 6.0, 5.0],
        }
    )
    result = v9._bootstrap_delta(frame, "challenger", "reference", 1000, 7)
    assert np.isclose(result["delta_mae"], -1 / 3)
    assert result["replicates"] == 1000


def test_complete_prediction_columns_excludes_fold_specific_missing_values() -> None:
    frame = pd.DataFrame(
        {
            "pred__complete": [1.0, 2.0, 3.0],
            "pred__fold_specific": [1.0, np.nan, np.nan],
            "pred__nonfinite": [1.0, 2.0, np.inf],
            "observed_pic50": [1.0, 2.0, 3.0],
        }
    )
    assert v9._complete_prediction_columns(frame) == ["pred__complete"]


def test_fit_predict_rejects_scaffold_crossing() -> None:
    matrix = pd.DataFrame(
        {
            "structure_id": ["a", "b"],
            "scaffold_group_id": ["same", "same"],
            "target_pic50": [5.0, 6.0],
            "rdkit2d__MolWt": [200.0, 300.0],
        }
    )
    observations = pd.DataFrame(
        {
            "structure_id": ["a", "b"],
            "potency_pic50_point": [5.0, 6.0],
            "measurement_modality": ["x", "x"],
            "assay_family": ["x", "x"],
            "source_family": ["x", "x"],
            "automation_class": ["x", "x"],
            "v1_5_conflict_review_structure": [False, False],
            "evaluation_or_lineage_leakage_caution": [False, False],
        }
    )
    mmp = pd.DataFrame(columns=["structure_id_a", "structure_id_b", "activity_cliff_ge_1_pic50"])
    candidate = v9.Candidate("tiny", "extratrees", "rdkit2d", {"n_estimators": 2, "max_features": 1.0})
    try:
        v9._fit_predict(matrix, observations, mmp, ["rdkit2d__MolWt"], {"a"}, {"b"}, candidate, 1, 1)
    except v9.CampaignError as error:
        assert "scaffold leakage" in str(error)
    else:
        raise AssertionError("scaffold crossing was accepted")


def test_self_hash_rejects_tampering(tmp_path: Path) -> None:
    path = tmp_path / "value.json"
    v9._atomic_json(path, {"status": "passed", "count": 1}, "document_sha256")
    value = v9._read_json(path, "document_sha256")
    value["count"] = 2
    path.write_text(v9.json.dumps(value), encoding="utf-8")
    try:
        v9._read_json(path, "document_sha256")
    except v9.CampaignError as error:
        assert "self-hash mismatch" in str(error)
    else:
        raise AssertionError("tampered JSON was accepted")
