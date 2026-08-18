from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from run_local_herg_discovery_campaign import (  # noqa: E402
    CampaignError,
    _candidates,
    _command,
    _initial,
    _plan,
    _resolve_candidate,
    _resume,
    _run_unit_with_retry,
    _self_hashed,
    _verify_self_hash,
)


def test_plan_covers_features_models_hpo_nested_confirmation_and_analysis() -> None:
    plan = _plan()
    stages = {unit["campaign_stage"] for unit in plan}
    kinds = {unit["kind"] for unit in plan}
    assert {
        "prepare",
        "coarse_halving",
        "promoted_halving",
        "nested_confirmation",
        "feature_relationships",
        "fixed_alternatives",
        "optional_chemprop",
        "finalize",
    } <= stages
    assert {"prepare", "tree", "classical", "similarity", "chemprop", "analyze"} <= kinds
    groups = {candidate["groups"] for candidate in _candidates()}
    assert {
        "rdkit2d",
        "morgan",
        "morgan_rdkit2d",
        "morgan_rdkit2d_maccs",
        "physics_selected",
        "candidate_primary",
        "all_scalable",
    } <= groups
    assert any(candidate["engine"] == "lightgbm" for candidate in _candidates())
    assert len([unit for unit in plan if unit["campaign_stage"] == "nested_confirmation"]) == 5
    expected_minutes = sum(unit["expected_minutes"] for unit in plan)
    assert expected_minutes == 1495
    chemprop = [unit for unit in plan if unit["kind"] == "chemprop"]
    assert all(unit["expected_minutes"] == 150 for unit in chemprop)
    assert all(unit["spec"]["epochs"] == 40 and unit["spec"]["patience"] == 8 for unit in chemprop)


def test_self_hash_detects_tampering() -> None:
    value = _self_hashed({"value": 1}, "self_hash")
    _verify_self_hash(value, "self_hash")
    value["value"] = 2
    with pytest.raises(CampaignError, match="mismatch"):
        _verify_self_hash(value, "self_hash")


def _checkpoint(tmp_path: Path) -> dict[str, object]:
    return {
        "python_path": sys.executable,
        "worker_path": str(SCRIPTS / "run_local_herg_discovery_worker.py"),
        "chemprop_runner_path": str(SCRIPTS / "run_local_herg_chemprop_unit.py"),
        "repo_root": str(tmp_path),
        "output_root": str(tmp_path / "campaign"),
        "matrix_root": str(tmp_path / "matrix"),
        "observations": str(tmp_path / "observations.parquet"),
        "workers": 6,
        "units": _plan(),
    }


def test_real_worker_commands_use_concrete_cli_not_abstract_unit_spec(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path)
    prepare = next(unit for unit in checkpoint["units"] if unit["kind"] == "prepare")
    prepare_command = _command(checkpoint, prepare)
    assert prepare_command[2] == "prepare"
    assert "--matrix-root" in prepare_command
    coarse = next(unit for unit in checkpoint["units"] if unit["campaign_stage"] == "coarse_halving")
    tree_command = _command(checkpoint, coarse)
    assert tree_command[2] == "tree-unit"
    assert "--prepared-root" in tree_command
    assert "--engine" in tree_command
    assert "--groups" in tree_command
    assert "--params-json" in tree_command
    assert "--unit-spec" not in tree_command
    assert tree_command[tree_command.index("--workers") + 1] == "6"


def test_initial_checkpoint_preserves_virtualenv_python_symlink(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    matrix = root / "matrix"
    matrix.mkdir()
    for name in ("combined_feature_matrix.parquet", "manifest.json", "validation.json"):
        (matrix / name).write_bytes(b"bound")
    observations = root / "observations.parquet"
    observations.write_bytes(b"bound")
    worker = root / "worker.py"
    worker.write_bytes(b"bound")
    chemprop = root / "chemprop.py"
    chemprop.write_bytes(b"bound")
    framework_python = root / "framework-python"
    framework_python.write_bytes(b"python")
    venv_python = root / "venv-python"
    venv_python.symlink_to(framework_python)
    args = type(
        "Args",
        (),
        {
            "repo_root": str(root),
            "output_root": str(root / "campaign"),
            "worker": str(worker),
            "chemprop_runner": str(chemprop),
            "python": str(venv_python),
            "matrix_root": str(matrix),
            "observations": str(observations),
            "workers": 6,
            "target_hours": 24.0,
            "hard_hours": 30.0,
            "finalization_reserve_minutes": 60.0,
        },
    )()
    checkpoint = _initial(args)
    assert checkpoint["python_path"] == str(venv_python.absolute())
    assert checkpoint["python_path"] != str(venv_python.resolve())


def test_halving_promotes_by_observed_inner_mae(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path)
    output = Path(checkpoint["output_root"])
    coarse = [
        unit
        for unit in checkpoint["units"]
        if unit["campaign_stage"] == "coarse_halving" and unit["spec"]["outer_fold"] == 0
    ]
    for index, unit in enumerate(coarse):
        unit["status"] = "passed"
        root = output / "units" / unit["unit_id"]
        root.mkdir(parents=True)
        (root / "metrics.json").write_text(
            json.dumps({"status": "passed", "metrics": {"mae": float(index + 1)}}),
            encoding="utf-8",
        )
    promoted = next(
        unit
        for unit in checkpoint["units"]
        if unit["campaign_stage"] == "promoted_halving"
        and unit["spec"]["outer_fold"] == 0
        and unit["spec"]["rank"] == 0
    )
    selected = _resolve_candidate(checkpoint, promoted)
    assert selected["candidate_key"] == coarse[0]["spec"]["candidate"]["candidate_key"]


def test_failed_unit_is_retried_once_with_separate_logs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0

    def fake_run(command: list[str], log: Path, environment: dict[str, str], deadline: float):
        nonlocal calls
        del command, environment, deadline
        calls += 1
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(f"attempt {calls}\n", encoding="utf-8")
        return (1 if calls == 1 else 0), False, 1.0

    monkeypatch.setattr("run_local_herg_discovery_campaign._run_process", fake_run)
    unit = {"unit_id": "retry_me", "attempts": 0, "status": "running"}
    checkpoint = {
        "updated_utc": "old",
        "active_elapsed_seconds": 0.0,
        "invocations": [{"active_elapsed_seconds": 0.0}],
    }
    returncode, _, elapsed, _ = _run_unit_with_retry(["worker"], unit, tmp_path, checkpoint, {}, float("inf"))
    assert returncode == 0
    assert elapsed == 2.0
    assert unit["attempts"] == 2
    assert len(unit["attempt_logs"]) == 2
    assert checkpoint["active_elapsed_seconds"] == 2.0


def test_resume_reconsiders_dependency_wait_and_preserves_active_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "campaign"
    output.mkdir()
    unit = {
        "unit_id": "downstream",
        "status": "skipped_dependency",
        "attempts": 0,
    }
    checkpoint = _self_hashed(
        {
            "bindings": [],
            "repo_root": str(tmp_path.resolve()),
            "output_root": str(output.resolve()),
            "worker_path": str((tmp_path / "worker.py").resolve()),
            "chemprop_runner_path": str((tmp_path / "chemprop.py").resolve()),
            "workers": 6,
            "active_elapsed_seconds": 123.0,
            "units": [unit],
        },
        "checkpoint_sha256",
    )
    (output / "checkpoint.json").write_text(json.dumps(checkpoint), encoding="utf-8")
    args = type(
        "Args",
        (),
        {
            "repo_root": str(tmp_path),
            "worker": str(tmp_path / "worker.py"),
            "chemprop_runner": str(tmp_path / "chemprop.py"),
            "workers": 6,
        },
    )()
    monkeypatch.setattr("run_local_herg_discovery_campaign._verify_bindings", lambda bindings: None)
    resumed = _resume(args, output)
    assert resumed["units"][0]["status"] == "pending"
    assert resumed["active_elapsed_seconds"] == 123.0
    assert "hard_deadline_epoch" not in resumed
