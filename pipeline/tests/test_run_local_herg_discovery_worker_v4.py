from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
WORKER_PATH = ROOT / "pipeline/scripts/run_local_herg_discovery_worker_v4.py"
SPEC = importlib.util.spec_from_file_location("herg_worker_v4", WORKER_PATH)
assert SPEC is not None and SPEC.loader is not None
worker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(worker)


def governance(operation: str, **values: Any) -> dict[str, Any]:
    return {
        "operation": operation,
        "source_partition": "train",
        "repository_validation_labels_opened": False,
        "repository_test_labels_opened": False,
        "broad_fixed_dose_pooled_into_pic50": False,
        **values,
    }


def test_capabilities_are_compute_only_and_strict() -> None:
    capabilities = worker._capabilities()
    assert capabilities["schema_version"].endswith("capabilities/1.0")
    assert set(capabilities["operations"]) == {
        "prepare",
        "baseline_reproduction",
        "classical_hpo",
        "chemprop_hpo",
        "nested_outer_evaluation",
        "aggregate_nested_oof",
        "microstate_conformer",
    }
    assert capabilities["operations"]["chemprop_hpo"]["adaptive_selection_eligible"] is False
    with pytest.raises(worker.WorkerError, match="unknown spec keys"):
        worker._validate_spec(governance("baseline_reproduction", anchor_source="v2_nested", ignored=True))


def test_chemprop_argv_executes_every_declared_training_knob(tmp_path: Path) -> None:
    spec = governance(
        "chemprop_hpo",
        evaluation_stage="inner",
        outer_fold=0,
        inner_folds=[0, 1, 2],
        epochs=40,
        batch_size=96,
        depth=4,
        hidden_dim=384,
        dropout=0.15,
        ffn_num_layers=3,
        patience=8,
        loss="mae",
        warmup_epochs=3.0,
        init_lr=1e-4,
        max_lr=8e-4,
        final_lr=1e-5,
    )
    worker._validate_spec(spec)
    argv = worker._chemprop_argv(Path("chemprop"), tmp_path / "data.csv", tmp_path / "out", spec, 7)
    joined = " ".join(argv)
    for flag, value in {
        "--epochs": "40",
        "--batch-size": "96",
        "--depth": "4",
        "--message-hidden-dim": "384",
        "--ffn-num-layers": "3",
        "--dropout": "0.15",
        "--loss-function": "mae",
        "--warmup-epochs": "3.0",
        "--init-lr": "0.0001",
        "--max-lr": "0.0008",
        "--final-lr": "1e-05",
    }.items():
        assert f"{flag} {value}" in joined


def test_conformer_specs_enforce_real_distinct_surfaces() -> None:
    whole = governance(
        "microstate_conformer",
        conformer_feature_set="conformer_ensemble_full",
        requested_conformers=24,
        retained_conformers=8,
        max_iterations=100,
        panel_selection="all_exact_train",
        panel_size=18_801,
        levels=[6, 8],
        aggregation="boltzmann",
        shard_size=250,
    )
    assert worker._validate_spec(whole) == whole
    convergence = governance(
        "microstate_conformer",
        conformer_feature_set="conformer_converged_subset",
        requested_conformers=50,
        retained_conformers=50,
        max_iterations=100,
        panel_selection="nested_residual_stratified",
        panel_size=500,
        levels=[6, 20, 50],
        aggregation="boltzmann",
        shard_size=100,
        source_unit_id="nested",
    )
    assert worker._validate_spec(convergence) == convergence
    bad = dict(convergence, retained_conformers=20)
    with pytest.raises(worker.WorkerError, match="largest convergence level"):
        worker._validate_spec(bad)


def test_one_molecule_fresh_conformer_smoke_is_finite() -> None:
    row = SimpleNamespace(structure_id="ethanol", standardized_smiles="CCO")
    spec = {
        "requested_conformers": 6,
        "retained_conformers": 6,
        "max_iterations": 25,
        "levels": [6],
        "seed": 17,
    }
    rows, error = worker._conformer_record(row, spec)
    assert error is None
    assert len(rows) == 1
    assert rows[0]["retained_conformers"] == 6
    for key in (
        "radius_gyration_boltzmann",
        "gasteiger_dipole_proxy_eA_boltzmann",
        "absolute_charge_radius_A_boltzmann",
        "polar_radial_exposure_A_boltzmann",
        "internal_polar_contact_count_boltzmann",
    ):
        assert worker.np.isfinite(rows[0][key])


def test_prepare_and_baseline_real_data_smoke(tmp_path: Path) -> None:
    v2 = ROOT / "research/local_runs/herg_discovery_campaign_v1"
    v3 = ROOT / "research/local_runs/herg_discovery_campaign_v3"
    if (
        not (v2 / "prepared/exact_train_cache.parquet").is_file()
        or not (v3 / "analysis/nested_model_selection_oof.parquet").is_file()
    ):
        pytest.skip("canonical local campaign artifacts unavailable")
    prepared = tmp_path / "prepared"
    validation = worker._prepare(
        SimpleNamespace(repo_root=ROOT, base_v2_root=v2, base_v3_root=v3, output_root=prepared)
    )
    assert validation["status"] == "passed"
    assert validation["exact_rows"] == 18_801
    results = tmp_path / "results"
    spec = governance("baseline_reproduction", anchor_source="v3_nested")
    unit = worker._run_unit(
        SimpleNamespace(
            repo_root=ROOT,
            prepared_root=prepared,
            results_root=results,
            unit_id="v3_anchor",
            unit_spec_json=json.dumps(spec),
            workers=1,
        )
    )
    assert unit["status"] == "passed"
    assert unit["metrics"]["n"] == 18_801
    assert unit["unit_spec_sha256"] == worker._digest(spec)
