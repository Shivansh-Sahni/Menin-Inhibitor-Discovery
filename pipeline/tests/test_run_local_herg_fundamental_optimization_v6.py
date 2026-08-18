from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_local_herg_fundamental_optimization_v6.py"
SPEC = importlib.util.spec_from_file_location("herg_v6_tested", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
V6 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(V6)


def test_candidate_plan_is_material_and_spans_fundamental_surfaces() -> None:
    candidates = V6._candidate_plan()
    V6._validate_candidate_plan(candidates)
    assert len(candidates) >= 35
    assert len({V6.V5._digest(candidate.payload()) for candidate in candidates}) == len(candidates)
    assert {candidate.engine for candidate in candidates} == {"xgboost", "lightgbm"}
    surfaces = {candidate.surface for candidate in candidates}
    assert {"electrostatic", "pharmacophore", "conformer", "qc_physics"} <= surfaces
    assert len([candidate for candidate in candidates if candidate.surface == "full"]) == 2


def test_v2_anchor_is_explicitly_included() -> None:
    anchors = [
        candidate for candidate in V6._candidate_plan() if candidate.candidate_id == "xgb_2d_morgan_v2_anchor"
    ]
    assert len(anchors) == 1
    assert anchors[0].params["max_depth"] == 8
    assert anchors[0].params["colsample_bytree"] == 0.60
    assert anchors[0].params["reg_lambda"] == 5.0


def test_surfaces_preserve_foundation_and_separate_physics() -> None:
    families = {
        "rdkit2d": ["r"],
        "morgan": ["m"],
        "polarity_charge_internal_contacts": ["e"],
        "energy_flexibility": ["f"],
        "shape": ["s"],
        "new3d_stable_misc": ["n"],
        "autocorr3d": ["a"],
        "whim": ["w"],
        "selected_interactions": ["i"],
        "old3d_stable": ["o"],
    }
    surfaces = V6._surfaces(families)
    assert surfaces["2d_morgan"] == ["m", "r"]
    assert set(surfaces["electrostatic"]) == {"e", "m", "r"}
    assert "m" not in surfaces["physics_only"]
    assert "r" not in surfaces["physics_only"]


def test_parser_defaults_to_six_workers_and_new_root() -> None:
    args = V6._parser().parse_args(["run"])
    assert args.workers == 6
    assert "fundamental_optimization_v6" in args.output_root
