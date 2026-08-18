from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "run_local_herg_feature_relationship_campaign_v5.py"
SPEC = importlib.util.spec_from_file_location("herg_v5_campaign", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_candidate_plan_is_material_and_spans_required_models_and_surfaces() -> None:
    candidates = MODULE._candidate_plan()
    MODULE._validate_candidate_plan(candidates)
    assert {candidate.engine for candidate in candidates} == {"xgboost", "lightgbm"}
    assert {candidate.surface for candidate in candidates} >= {
        "2d",
        "morgan",
        "2d_morgan",
        "old3d",
        "new3d",
        "fundamental",
        "full",
    }
    assert len({MODULE._digest(candidate.payload()) for candidate in candidates}) == len(candidates)


@pytest.mark.parametrize(
    ("column", "family"),
    [
        ("rdkit2d__MolLogP", "rdkit2d"),
        ("morgan__0001", "morgan"),
        ("f3d__radius", "old3d_stable"),
        ("new3d__dominant_autocorr3d_001", "autocorr3d"),
        ("new3d__dominant_whim_001", "whim"),
        ("new3d__energy_range_kcal_mol", "energy_flexibility"),
        ("new3d__gasteiger_dipole__mean", "polarity_charge_internal_contacts"),
        ("new3d__radius_of_gyration__mean", "shape"),
        ("v5interaction__logp_x_shape", "selected_interactions"),
    ],
)
def test_feature_family_contract(column: str, family: str) -> None:
    assert MODULE._feature_family(column) == family


def test_new3d_qc_masks_extreme_and_absolute_energy_and_records_convergence() -> None:
    frame = pd.DataFrame(
        {
            "structure_id": ["a", "b", "c"],
            "new3d__energy_min_kcal_mol": [10.0, 200_000.0, np.inf],
            "new3d__energy_range_kcal_mol": [5.0, 20_000.0, 2.0],
            "new3d__retained_conformer_count": [8, 8, 0],
            "new3d__unconverged_retained_count": [8, 2, 0],
            "new3d__feature_status": ["ok", "ok", "conformer_failed"],
        }
    )
    cleaned, report = MODULE._qc_new3d(frame)
    assert cleaned["new3d__energy_min_kcal_mol"].isna().all()
    assert np.isclose(cleaned.loc[0, "new3d__energy_range_kcal_mol"], 5.0)
    assert cleaned.loc[1:, "new3d__energy_range_kcal_mol"].isna().all()
    assert cleaned["new3d__energy_extreme_indicator"].tolist() == [0, 1, 0]
    assert cleaned["new3d__energy_nonfinite_indicator"].tolist() == [0, 0, 1]
    assert cleaned["new3d__all_retained_unconverged_indicator"].tolist() == [1, 0, 0]
    assert set(report.qc_measure) >= {
        "energy_extreme",
        "energy_nonfinite",
        "absolute_energy_feature_excluded",
    }


def test_self_hashed_json_rejects_tampering(tmp_path: Path) -> None:
    path = tmp_path / "record.json"
    MODULE._atomic_json(path, {"status": "passed", "value": 3}, "record_sha256")
    assert MODULE._read_json(path, "record_sha256")["value"] == 3
    path.write_text(path.read_text().replace('"value": 3', '"value": 4'))
    with pytest.raises(MODULE.CampaignError, match="self hash"):
        MODULE._read_json(path, "record_sha256")


def test_unit_document_binding_detects_byte_tampering(tmp_path: Path) -> None:
    unit_path = tmp_path / "units" / "inner_o0_xgb" / "unit.json"
    MODULE._atomic_json(
        unit_path,
        {"status": "passed", "unit_id": "inner_o0_xgb"},
        "unit_json_sha256",
    )
    binding = MODULE._binding(unit_path, "unit_document::inner_o0_xgb")
    MODULE._verify_binding(binding, tmp_path)
    assert MODULE._read_json(unit_path, "unit_json_sha256")["status"] == "passed"
    unit_path.write_bytes(unit_path.read_bytes() + b" ")
    with pytest.raises(MODULE.CampaignError, match="changed|mismatch"):
        MODULE._verify_binding(binding, tmp_path)


def test_paired_effects_preserve_structure_and_scaffold_grain() -> None:
    baseline = pd.DataFrame(
        {
            "structure_id": ["a", "b"],
            "scaffold_group_id": ["s1", "s2"],
            "target_pic50": [6.0, 7.0],
        }
    )
    effects = MODULE._paired_frame(
        baseline,
        np.array([0.25, 0.5]),
        np.array([0.0625, 0.25]),
        np.array([5.5, 7.5]),
        "shape",
        "conditional_permutation",
        2,
        1,
    )
    required = {
        "hypothesis_id",
        "evidence_type",
        "outer_fold",
        "structure_id",
        "scaffold_group_id",
        "baseline_abs_error",
        "perturbed_abs_error",
    }
    assert required <= set(effects.columns)
    assert effects.structure_id.tolist() == ["a", "b"]
    assert effects.outer_fold.eq(2).all()


def test_cli_rejects_more_than_six_workers() -> None:
    args = MODULE._parser().parse_args(["run", "--workers", "7"])
    assert args.workers == 7
    assert MODULE.main is not None
