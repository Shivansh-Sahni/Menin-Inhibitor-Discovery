from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from run_local_herg_discovery_campaign_v3 import (  # noqa: E402
    ANALYSIS_FILES,
    RECOVERY_CENSORED_UNIT_ID,
    RECOVERY_MIGRATION_ID,
    RECOVERY_OLD_IMPLEMENTATION_HASHES,
    CampaignError,
    _atomic_json,
    _command,
    _dependencies_ready,
    _hpo_candidates,
    _plan,
    _recovery_migration,
    _resume,
    _run_unit_with_retry,
    _self_hashed,
    _validate_scientific_scope,
    _verify_self_hash,
    _worker_hash,
)


def _write_worker_json(path: Path, value: dict[str, object]) -> None:
    document = copy.deepcopy(value)
    document["unit_json_sha256"] = _worker_hash(document)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _production_recovery_fixture(tmp_path: Path) -> tuple[Path, Path, float]:
    output = tmp_path / "v3"
    worker = tmp_path / "worker.py"
    worker.write_text("# repaired worker\n", encoding="utf-8")
    units: list[dict[str, object]] = []
    for outer in range(5):
        for index in range(24):
            unit_id = f"hpo_coarse_o{outer}__candidate_{index}"
            spec = {
                "operation": "expanded_hpo_tree",
                "source_partition": "train",
                "repository_validation_labels_opened": False,
                "repository_test_labels_opened": False,
                "broad_fixed_dose_pooled_into_pic50": False,
                "candidate": {"candidate_id": f"candidate_{index}"},
                "outer_fold": outer,
            }
            units.append(
                {
                    "unit_id": unit_id,
                    "stage": f"hpo_coarse_o{outer}",
                    "operation": "expanded_hpo_tree",
                    "status": "failed_noncritical",
                    "attempts": 2,
                    "critical": False,
                    "spec": spec,
                    "attempt_results": [
                        {"attempt": attempt, "artifact_validation_error": "unit_json_sha256 mismatch"}
                        for attempt in (1, 2)
                    ],
                }
            )
    for index in range(4):
        unit_id = f"broad_hpo__candidate_{index}"
        spec = {
            "operation": "broad_wt_auxiliary",
            "source_partition": "train",
            "repository_validation_labels_opened": False,
            "repository_test_labels_opened": False,
            "broad_fixed_dose_pooled_into_pic50": False,
            "candidate": {"candidate_id": f"broad_{index}"},
        }
        units.append(
            {
                "unit_id": unit_id,
                "stage": "broad_wt_hpo",
                "operation": "broad_wt_auxiliary",
                "status": "failed_noncritical",
                "attempts": 2,
                "critical": False,
                "spec": spec,
                "attempt_results": [
                    {"attempt": attempt, "artifact_validation_error": "unit_json_sha256 mismatch"}
                    for attempt in (1, 2)
                ],
            }
        )
    for unit in units:
        broad = unit["operation"] == "broad_wt_auxiliary"
        artifact_count = 4 if broad else 3
        bindings = []
        for artifact_index in range(artifact_count):
            artifact = output / "units" / str(unit["unit_id"]) / f"artifact_{artifact_index}.bin"
            artifact.parent.mkdir(parents=True, exist_ok=True)
            artifact.write_bytes(f"{unit['unit_id']}:{artifact_index}".encode())
            bindings.append(
                {
                    "role": f"role_{artifact_index}",
                    "path": str(artifact.resolve()),
                    "bytes": artifact.stat().st_size,
                    "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                }
            )
        spec = unit["spec"]
        assert isinstance(spec, dict)
        executed_spec = {"operation": unit["operation"]}
        _write_worker_json(
            output / "units" / str(unit["unit_id"]) / "unit.json",
            {
                "unit_id": unit["unit_id"],
                "operation": unit["operation"],
                "status": "passed",
                "unit_spec": spec,
                "unit_spec_sha256": _worker_hash(spec),
                "executed_spec": executed_spec,
                "executed_spec_sha256": _worker_hash(executed_spec),
                "scientific_scope": {
                    "source_partition": "train",
                    "repository_validation_labels_opened": False,
                    "repository_test_labels_opened": False,
                    "broad_fixed_dose_pooled_into_pic50": False,
                },
                "metrics": {"selection_score": -0.02 if broad else 0.48},
                "artifacts": bindings,
            },
        )
    censored_logs = []
    for attempt in (1, 2):
        log = output / "logs" / f"{RECOVERY_CENSORED_UNIT_ID}.attempt_{attempt}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("ValueError: assignment destination is read-only\n", encoding="utf-8")
        censored_logs.append(log)
    units.append(
        {
            "unit_id": RECOVERY_CENSORED_UNIT_ID,
            "stage": "censored_sensitivity",
            "operation": "censored_sensitivity",
            "status": "failed_noncritical",
            "attempts": 2,
            "critical": False,
            "spec": {},
            "attempt_results": [
                {"attempt": attempt, "log_path": str(censored_logs[attempt - 1])} for attempt in (1, 2)
            ],
        }
    )
    units.append(
        {
            "unit_id": "prepare_v3",
            "stage": "prepare",
            "operation": "prepare",
            "status": "passed",
            "attempts": 1,
        }
    )
    units.extend(
        {
            "unit_id": f"downstream_{index}",
            "stage": "downstream",
            "operation": "analysis",
            "status": "pending",
            "attempts": 0,
        }
        for index in range(45)
    )
    active_seconds = 1608.4409897
    checkpoint = {
        "schema_version": "platform-local-herg-discovery-campaign-v3/1.0",
        "status": "incomplete",
        "output_root": str(output.resolve()),
        "worker_path": str(worker.resolve()),
        "active_elapsed_seconds": active_seconds,
        "invocations": [{"active_elapsed_seconds": active_seconds}],
        "incomplete_reasons": ["stale"],
        "bindings": [
            {
                "role": "v3_orchestrator",
                "path": str((SCRIPTS / "run_local_herg_discovery_campaign_v3.py").resolve()),
                "bytes": 1,
                "sha256": RECOVERY_OLD_IMPLEMENTATION_HASHES["v3_orchestrator"],
            },
            {
                "role": "v3_worker",
                "path": str(worker.resolve()),
                "bytes": 1,
                "sha256": RECOVERY_OLD_IMPLEMENTATION_HASHES["v3_worker"],
            },
        ],
        "units": units,
    }
    _atomic_json(output / "checkpoint.json", checkpoint, "checkpoint_sha256")
    return output, worker, active_seconds


def test_plan_has_distinct_tasks_real_hpo_and_required_deliverables() -> None:
    plan = _plan()
    operations = {unit["operation"] for unit in plan}
    assert {
        "prepare",
        "expanded_hpo_tree",
        "nested_robustness",
        "repeated_seed_tree",
        "feature_ablation",
        "interaction_stability",
        "assay_quality_strata",
        "uncertainty_calibration",
        "censored_sensitivity",
        "mmp_cliff_residual",
        "broad_wt_auxiliary",
        "chemprop_ensemble",
        "finalist_refit_artifact",
        "analyze",
    } <= operations
    candidates = _hpo_candidates()
    assert len(candidates) == 24
    assert {item["engine"] for item in candidates} == {"xgboost", "lightgbm"}
    assert len({item["candidate_id"] for item in candidates}) == len(candidates)
    assert sum(unit["status"] == "pending" for unit in plan) == len(plan)
    assert sum(unit["stage"] == "nested_confirmation" for unit in plan) == 5
    assert sum(unit["stage"] == "repeated_seed_confirmation" for unit in plan) == 4
    assert sum(unit["stage"] == "broad_wt_hpo" for unit in plan) == 4
    assert sum(unit["stage"] == "chemprop_ensemble" for unit in plan) == 3
    required = {unit["unit_id"] for unit in plan if unit["required_for_completion"]}
    assert required == {
        "prepare_v3",
        "broad_wt_final_refit",
        "exact_pic50_final_refit",
        "analyze_v3",
    }
    expected = sum(unit["expected_minutes"] for unit in plan)
    assert 0 < expected <= 29 * 60


def test_every_unit_declares_train_only_and_label_blind_scope() -> None:
    for unit in _plan():
        spec = unit["spec"]
        assert spec["source_partition"] == "train"
        assert spec["repository_validation_labels_opened"] is False
        assert spec["repository_test_labels_opened"] is False
        assert spec["broad_fixed_dose_pooled_into_pic50"] is False
    broad = [unit for unit in _plan() if unit["operation"] == "broad_wt_auxiliary"]
    assert broad
    assert all(unit["spec"]["task"] == "confirmed_wt_fixed_dose_binary" for unit in broad)
    assert all(unit["spec"]["fit_partition"] == "train" for unit in broad)
    assert all(unit["spec"]["full_surface_rows"] == 339373 for unit in broad)


def test_worker_commands_match_v3_contract(tmp_path: Path) -> None:
    checkpoint = {
        "python_path": sys.executable,
        "worker_path": str(SCRIPTS / "run_local_herg_discovery_worker_v3.py"),
        "repo_root": str(tmp_path),
        "base_campaign_root": str(tmp_path / "v2"),
        "output_root": str(tmp_path / "v3"),
        "workers": 6,
    }
    prepare = next(unit for unit in _plan() if unit["operation"] == "prepare")
    command = _command(checkpoint, prepare)
    assert command[2] == "prepare"
    assert "--base-campaign-root" in command
    run_unit = next(unit for unit in _plan() if unit["operation"] == "expanded_hpo_tree")
    command = _command(checkpoint, run_unit)
    assert command[2] == "run-unit"
    assert "--unit-spec" in command
    spec = json.loads(command[command.index("--unit-spec") + 1])
    assert spec["operation"] == "expanded_hpo_tree"
    analyze = next(unit for unit in _plan() if unit["operation"] == "analyze")
    command = _command(checkpoint, analyze)
    assert command[2] == "analyze"
    assert "--results-root" in command
    assert command[command.index("--workers") + 1] == "6"


def test_dependencies_wait_without_becoming_terminal() -> None:
    plan = _plan()
    checkpoint = {"units": plan}
    promoted = next(unit for unit in plan if unit["stage"] == "hpo_promoted_o0")
    assert not _dependencies_ready(checkpoint, promoted)
    coarse = [unit for unit in plan if unit["stage"] == "hpo_coarse_o0"]
    for unit in coarse:
        unit["status"] = "passed"
    assert _dependencies_ready(checkpoint, promoted)


def test_hash_and_scientific_scope_reject_tampering() -> None:
    value = _self_hashed({"value": 1}, "sha")
    _verify_self_hash(value, "sha")
    value["value"] = 2
    with pytest.raises(CampaignError, match="mismatch"):
        _verify_self_hash(value, "sha")
    with pytest.raises(CampaignError, match="validation-label"):
        _validate_scientific_scope(
            {
                "source_partition": "train",
                "repository_validation_labels_opened": True,
                "repository_test_labels_opened": False,
                "broad_fixed_dose_pooled_into_pic50": False,
            },
            "broad_wt_auxiliary",
        )


def test_retry_is_in_place_and_code_three_is_not_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0

    def fake_run(command: list[str], log: Path, environment: dict[str, str], deadline: float):
        nonlocal calls
        del command, environment, deadline
        calls += 1
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(str(calls), encoding="utf-8")
        return (2 if calls == 1 else 3), False, 1.0

    monkeypatch.setattr("run_local_herg_discovery_campaign_v3._run_process", fake_run)
    monkeypatch.setattr("run_local_herg_discovery_campaign_v3._checkpoint", lambda out, value: value)
    unit = {"unit_id": "optional", "attempts": 0, "operation": "chemprop_ensemble"}
    checkpoint = {"active_elapsed_seconds": 0.0, "invocations": [{"active_elapsed_seconds": 0.0}]}
    result, _, elapsed, _ = _run_unit_with_retry(["worker"], unit, tmp_path, checkpoint, {}, float("inf"))
    assert result == 3
    assert unit["attempts"] == 2
    assert calls == 2
    assert elapsed == 2.0
    assert checkpoint["active_elapsed_seconds"] == 2.0


def test_resume_preserves_active_time_and_terminalizes_exhausted_interruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "v3"
    output.mkdir()
    checkpoint = _self_hashed(
        {
            "bindings": [],
            "repo_root": str(tmp_path.resolve()),
            "output_root": str(output.resolve()),
            "base_campaign_root": str((tmp_path / "v2").resolve()),
            "broad_surface": str((tmp_path / "broad.parquet").resolve()),
            "worker_path": str((tmp_path / "worker.py").resolve()),
            "workers": 6,
            "status": "running",
            "active_elapsed_seconds": 777.0,
            "units": [
                {"unit_id": "retryable", "status": "running", "attempts": 1, "critical": False},
                {"unit_id": "exhausted", "status": "running", "attempts": 2, "critical": True},
            ],
        },
        "checkpoint_sha256",
    )
    (output / "checkpoint.json").write_text(json.dumps(checkpoint), encoding="utf-8")
    args = type(
        "Args",
        (),
        {
            "repo_root": str(tmp_path),
            "output_root": str(output),
            "base_campaign_root": str(tmp_path / "v2"),
            "broad_surface": str(tmp_path / "broad.parquet"),
            "worker": str(tmp_path / "worker.py"),
            "workers": 6,
        },
    )()
    monkeypatch.setattr("run_local_herg_discovery_campaign_v3._verify_bindings", lambda values: None)
    resumed = _resume(args, output)
    assert resumed["active_elapsed_seconds"] == 777.0
    assert resumed["units"][0]["status"] == "pending"
    assert resumed["units"][1]["status"] == "failed_critical"
    assert "hard_deadline_epoch" not in resumed


def test_resume_adopts_atomic_completed_artifact_before_exhausting_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "v3"
    output.mkdir()
    checkpoint = _self_hashed(
        {
            "bindings": [],
            "repo_root": str(tmp_path.resolve()),
            "output_root": str(output.resolve()),
            "base_campaign_root": str((tmp_path / "v2").resolve()),
            "broad_surface": str((tmp_path / "broad.parquet").resolve()),
            "worker_path": str((tmp_path / "worker.py").resolve()),
            "workers": 6,
            "status": "running",
            "active_elapsed_seconds": 1.0,
            "units": [
                {
                    "unit_id": "atomic_done",
                    "status": "running",
                    "attempts": 2,
                    "critical": True,
                }
            ],
        },
        "checkpoint_sha256",
    )
    (output / "checkpoint.json").write_text(json.dumps(checkpoint), encoding="utf-8")
    args = type(
        "Args",
        (),
        {
            "repo_root": str(tmp_path),
            "output_root": str(output),
            "base_campaign_root": str(tmp_path / "v2"),
            "broad_surface": str(tmp_path / "broad.parquet"),
            "worker": str(tmp_path / "worker.py"),
            "workers": 6,
        },
    )()
    monkeypatch.setattr("run_local_herg_discovery_campaign_v3._verify_bindings", lambda values: None)
    monkeypatch.setattr(
        "run_local_herg_discovery_campaign_v3._validate_completed_unit", lambda out, unit: None
    )
    resumed = _resume(args, output)
    assert resumed["units"][0]["status"] == "passed"
    assert resumed["units"][0]["recovered_completed_artifact_after_interruption"] is True


def test_terminal_analysis_contract_is_comprehensive() -> None:
    assert set(ANALYSIS_FILES) == {
        "validation.json",
        "analysis.md",
        "manifest.json",
        "model_cards.json",
        "decision_ledger.json",
        "feature_relationships.json",
        "final_models_manifest.json",
    }


def test_production_recovery_adopts_artifacts_preserves_history_and_is_idempotent(
    tmp_path: Path,
) -> None:
    output, worker, active_seconds = _production_recovery_fixture(tmp_path)
    original_checkpoint_bytes = (output / "checkpoint.json").read_bytes()
    result = _recovery_migration(output, worker, RECOVERY_MIGRATION_ID)
    assert result["status"] == "migration_applied"
    assert result["adopted_units"] == 124
    assert result["verified_artifact_bindings"] == 376
    checkpoint = json.loads((output / "checkpoint.json").read_text(encoding="utf-8"))
    assert checkpoint["active_elapsed_seconds"] == active_seconds
    assert checkpoint["invocations"] == [{"active_elapsed_seconds": active_seconds}]
    assert "incomplete_reasons" not in checkpoint
    assert checkpoint["status"] == "recovered_ready_to_resume"
    adopted = [unit for unit in checkpoint["units"] if "recovered_worker_artifact" in unit]
    assert len(adopted) == 124
    assert all(unit["status"] == "passed" and unit["attempts"] == 2 for unit in adopted)
    assert all(len(unit["attempt_results"]) == 2 for unit in adopted)
    censored = next(unit for unit in checkpoint["units"] if unit["unit_id"] == RECOVERY_CENSORED_UNIT_ID)
    assert censored["status"] == "pending"
    assert censored["attempts"] == 0
    assert len(censored["attempt_results"]) == 2
    assert censored["recovery_attempt_cycle"] == 1
    assert len(checkpoint["migrations"]) == 1
    migration = checkpoint["migrations"][0]
    snapshot_binding = migration["pre_checkpoint_snapshot_binding"]
    snapshot = Path(snapshot_binding["path"])
    assert snapshot.read_bytes() == original_checkpoint_bytes
    assert hashlib.sha256(original_checkpoint_bytes).hexdigest() == snapshot_binding["sha256"]
    assert snapshot.name == f"checkpoint.pre_{migration['pre_checkpoint_sha256']}.json"
    again = _recovery_migration(output, worker, RECOVERY_MIGRATION_ID)
    assert again["status"] == "already_applied"
    checkpoint_again = json.loads((output / "checkpoint.json").read_text(encoding="utf-8"))
    assert len(checkpoint_again["migrations"]) == 1


def test_recovery_refuses_tampered_worker_artifact(tmp_path: Path) -> None:
    output, worker, _ = _production_recovery_fixture(tmp_path)
    path = next((output / "units").glob("*/unit.json"))
    document = json.loads(path.read_text(encoding="utf-8"))
    document["metrics"]["selection_score"] = 999.0
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(CampaignError, match="unit_json_sha256 mismatch"):
        _recovery_migration(output, worker, RECOVERY_MIGRATION_ID)


def test_recovery_requires_exact_confirmation(tmp_path: Path) -> None:
    output, worker, _ = _production_recovery_fixture(tmp_path)
    with pytest.raises(CampaignError, match="confirmation"):
        _recovery_migration(output, worker, "yes")
