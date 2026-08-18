from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_local_herg_honest_measurement_campaign_v7.py"
SPEC = importlib.util.spec_from_file_location("herg_v7_test_module", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
V7 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = V7
SPEC.loader.exec_module(V7)


def _observations() -> pd.DataFrame:
    rows = []
    for structure, center in (("A", 5.0), ("B", 6.0), ("C", 4.0)):
        for index, (source, delta) in enumerate((("release", -0.2), ("specialized", 0.2))):
            rows.append(
                {
                    "observation_id": f"{structure}-{index}",
                    "structure_id": structure,
                    "scaffold_group_id": f"S-{structure}",
                    "potency_pic50_point": center + delta,
                    "source_family": source,
                    "assay_family": "functional" if index else "mixed",
                    "measurement_modality": "patch" if index else "unresolved",
                    "automation_class": "manual" if index else "unresolved",
                    "modality_confidence": "high" if index else "unresolved",
                    "endpoint_class": "potency_ic50",
                    "protocol_completeness_score": 5 if index else 0,
                    "v1_5_conflict_review_structure": False,
                    "evaluation_or_lineage_leakage_caution": False,
                }
            )
    return pd.DataFrame(rows)


def test_candidate_plan_is_material_and_unique() -> None:
    candidates = V7._candidate_plan()
    assert len(candidates) == 10
    assert len({V7._digest(candidate.payload()) for candidate in candidates}) == 10
    assert {candidate.engine for candidate in candidates} == {"xgboost", "lightgbm"}
    assert {candidate.surface for candidate in candidates} == {"2d_morgan", "qc_physics"}
    assert any(candidate.measurement_correction for candidate in candidates)
    assert any(candidate.mixture for candidate in candidates)


def test_measurement_offsets_use_only_fit_identities() -> None:
    observations = _observations()
    tables, adjusted = V7._fit_measurement_offsets(observations, {"A", "B"})
    assert set(adjusted.structure_id) == {"A", "B"}
    assert "C" not in set(adjusted.structure_id)
    assert tables["source_family"]["release"] < 0
    assert tables["source_family"]["specialized"] > 0
    assert np.isfinite(adjusted.measurement_corrected_pic50).all()


def test_metrics_report_global_tails_and_thresholds() -> None:
    observed = np.array([3.5, 4.5, 5.2, 5.5, 5.8, 6.5, 7.5])
    predicted = np.array([4.0, 4.7, 5.2, 5.4, 5.8, 6.2, 7.0])
    metrics = V7._metrics(observed, predicted)
    assert metrics["n"] == 7
    assert metrics["mae"] > 0
    assert metrics["balanced_potency_bin_mae"] > 0
    assert metrics["tail_mae"] == 0.5
    assert set(metrics["thresholds"]) == {"20uM", "10uM", "1uM"}
    assert metrics["safety_selection_score"] != metrics["accuracy_selection_score"]


def test_structure_targets_downweight_conflicts_and_preserve_baseline() -> None:
    observations = _observations()
    observations.loc[observations.structure_id.eq("B"), "v1_5_conflict_review_structure"] = True
    matrix = pd.DataFrame(
        {
            "structure_id": ["A", "B", "C"],
            "target_pic50": [5.0, 6.0, 4.0],
        }
    )
    candidate = V7._candidate_plan()[2]
    targets = V7._structure_training_targets(matrix, observations, {"A", "B"}, candidate)
    assert set(targets.structure_id) == {"A", "B"}
    weight_a = float(targets.loc[targets.structure_id.eq("A"), "sample_weight"].iloc[0])
    weight_b = float(targets.loc[targets.structure_id.eq("B"), "sample_weight"].iloc[0])
    assert weight_b < weight_a
    baseline = V7._structure_training_targets(matrix, observations, {"A", "B"}, V7._candidate_plan()[0])
    assert np.allclose(baseline.training_target, baseline.canonical_target)
    quality = V7._structure_training_targets(
        matrix, observations, {"A", "B", "C"}, V7._candidate_plan()[2]
    )
    tail = V7._structure_training_targets(
        matrix, observations, {"A", "B", "C"}, V7._candidate_plan()[3]
    )
    assert np.isfinite(tail.sample_weight).all()
    assert tail.loc[tail.structure_id == "C", "sample_weight"].item() > quality.loc[
        quality.structure_id == "C", "sample_weight"
    ].item()
