from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest
import yaml

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import run_local_herg_discovery_campaign_v4 as campaign_v4  # noqa: E402
from run_local_herg_discovery_campaign_v4 import (  # noqa: E402
    CAPABILITIES_SCHEMA_VERSION,
    CONFIG_SCHEMA_VERSION,
    CampaignError,
    _atomic_json,
    _command,
    _dependencies_ready,
    _hash_value,
    _resume,
    _run_with_retry,
    _self_hashed,
    _validate_config,
    _validate_scientific_plan_shape,
    _validate_scope,
    _validate_unit,
    _verify_self_hash,
)


@pytest.fixture(autouse=True)
def _small_config_fixture(monkeypatch: pytest.MonkeyPatch) -> None:
    """Most unit tests use a three-unit fixture; shape has a dedicated test."""
    monkeypatch.setattr(campaign_v4, "_validate_scientific_plan_shape", lambda _units: None)


COMMON = {
    "operation",
    "seed",
    "source_partition",
    "repository_validation_labels_opened",
    "repository_test_labels_opened",
    "broad_fixed_dose_pooled_into_pic50",
}


def _capabilities() -> dict[str, object]:
    def op(extra: set[str], required: set[str], *, score: bool = False) -> dict[str, object]:
        return {
            "allowed_spec_keys": sorted(COMMON | extra),
            "required_spec_keys": sorted({"operation"} | required),
            "optional": False,
            "score_required": score,
            "engines": ["xgboost", "lightgbm", "extratrees"],
            "feature_sets": ["morgan_rdkit2d", "all_scalable"],
        }

    return {
        "schema_version": CAPABILITIES_SCHEMA_VERSION,
        "unit_document_hash_contract": "compact_sorted_json_plus_newline",
        "operations": {
            "prepare": op(set(), set()),
            "classical_hpo": op(
                {"candidate", "evaluation_stage", "outer_fold", "inner_folds", "budget_fraction"},
                {"candidate", "evaluation_stage", "outer_fold", "inner_folds", "budget_fraction"},
                score=True,
            ),
            "aggregate_nested_oof": op({"source_unit_ids"}, {"source_unit_ids"}, score=True),
            "chemprop_hpo": op(
                {
                    "batch_size",
                    "depth",
                    "dropout",
                    "epochs",
                    "evaluation_stage",
                    "ffn_num_layers",
                    "final_lr",
                    "hidden_dim",
                    "init_lr",
                    "inner_folds",
                    "loss",
                    "max_lr",
                    "outer_fold",
                    "patience",
                    "warmup_epochs",
                },
                {
                    "batch_size",
                    "depth",
                    "dropout",
                    "epochs",
                    "evaluation_stage",
                    "ffn_num_layers",
                    "final_lr",
                    "hidden_dim",
                    "init_lr",
                    "inner_folds",
                    "loss",
                    "max_lr",
                    "outer_fold",
                    "patience",
                    "warmup_epochs",
                },
                score=True,
            ),
            "analyze": {
                **op(set(), set()),
                "required_artifacts": ["validation.json", "analysis.md"],
            },
        },
    }


def _spec(operation: str, **extra: object) -> dict[str, object]:
    return {
        "operation": operation,
        "source_partition": "train",
        "repository_validation_labels_opened": False,
        "repository_test_labels_opened": False,
        "broad_fixed_dose_pooled_into_pic50": False,
        **extra,
    }


def _unit(
    unit_id: str,
    stage: str,
    operation: str,
    expected: float,
    spec: dict[str, object],
    dependencies: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "unit_id": unit_id,
        "stage": stage,
        "operation": operation,
        "expected_minutes": expected,
        "critical": operation in {"prepare", "analyze"},
        "required_for_completion": operation in {"prepare", "analyze"},
        "priority_tier": "core",
        "dependencies": dependencies,
        "spec": spec,
    }


def _config(tmp_path: Path, *, duplicate: bool = False) -> tuple[Path, dict[str, object]]:
    source = tmp_path / "source.bin"
    source.write_bytes(b"bound")
    prepare = _unit("prepare_v4", "prepare", "prepare", 20, _spec("prepare"), [])
    candidate = {
        "candidate_id": "xgb_anchor",
        "engine": "xgboost",
        "feature_set": "morgan_rdkit2d",
        "params": {"n_estimators": 800},
    }
    hpo_spec = _spec(
        "classical_hpo",
        candidate=candidate,
        evaluation_stage="inner",
        outer_fold=0,
        inner_folds=[0, 1, 2],
        budget_fraction=1.0,
    )
    hpo = _unit(
        "hpo_0",
        "hpo",
        "classical_hpo",
        700,
        hpo_spec,
        [{"stage": "prepare", "minimum_passed": 1, "require_terminal": True}],
    )
    analyze = _unit(
        "analyze_v4",
        "analyze",
        "analyze",
        60,
        _spec("analyze"),
        [{"stage": "hpo", "minimum_passed": 1, "require_terminal": True}],
    )
    units = [prepare, hpo, analyze]
    if duplicate:
        copied = copy.deepcopy(hpo)
        copied["unit_id"] = "hpo_duplicate"
        units.insert(2, copied)
    nominal = sum(float(unit["expected_minutes"]) for unit in units)
    config: dict[str, object] = {
        "schema_version": CONFIG_SCHEMA_VERSION,
        "campaign": {
            "workers": 6,
            "target_active_hours": 13.5,
            "hard_active_hours": 15,
            "finalization_reserve_minutes": 60,
            "minimum_free_disk_gib": 25,
            "minimum_available_memory_gib": 4,
            "maximum_output_gib": 12,
        },
        "scientific_contract": {
            "source_partition": "train",
            "repository_validation_labels_opened": False,
            "repository_test_labels_opened": False,
            "broad_fixed_dose_pooled_into_pic50": False,
            "exact_train_structures": 18801,
            "broad_full_structures": 339373,
            "broad_train_structures": 265625,
            "fixed_outer_folds": 5,
            "fixed_inner_folds": 3,
        },
        "inputs": {"bound_source": str(source)},
        "plan": {"units": units},
        "completion": {
            "stage_minimum_passed": {"prepare": 1, "hpo": 1, "analyze": 1},
            "required_artifact_roles": {
                "classical_hpo": ["oof_predictions", "feature_importance", "feature_schema"]
            },
            "exact_nested_rows": 18801,
        },
        "empirical_basis": {
            "v2_active_hours": 4.1436,
            "v3_active_hours": 0.996683,
            "chemprop_five_fold_40_epoch_minutes": 32.8,
            "nominal_useful_minutes": nominal,
            "estimate_method": "empirical_pilots_and_conservative_stage_bounds",
        },
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path, config


def _write_worker_json(path: Path, value: dict[str, object]) -> None:
    document = copy.deepcopy(value)
    document["unit_json_sha256"] = _hash_value(document, worker=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n",
        encoding="utf-8",
    )


def test_config_is_explicit_material_unique_and_within_fifteen_hour_cap(tmp_path: Path) -> None:
    path, _ = _config(tmp_path)
    validated = _validate_config(path, tmp_path, _capabilities())
    assert validated["nominal_expected_minutes"] == 780
    assert validated["material_specs"] == len(validated["units"])
    assert validated["stage_counts"] == {"prepare": 1, "hpo": 1, "analyze": 1}


def test_config_rejects_pseudovariation_unknown_keys_and_short_plan(tmp_path: Path) -> None:
    path, _ = _config(tmp_path, duplicate=True)
    with pytest.raises(CampaignError, match="pseudo-varied"):
        _validate_config(path, tmp_path, _capabilities())


def test_scientific_shape_rejects_runtime_padding_without_deep_work(tmp_path: Path) -> None:
    path, _ = _config(tmp_path)
    units = _validate_config(path, tmp_path, _capabilities())["units"]
    with pytest.raises(CampaignError, match="scientific plan lacks"):
        _validate_scientific_plan_shape(units)
    path, config = _config(tmp_path)
    units = config["plan"]["units"]
    assert isinstance(units, list)
    units[1]["spec"]["ignored_knob"] = 1
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(CampaignError, match="unsupported spec keys"):
        _validate_config(path, tmp_path, _capabilities())
    units[1]["spec"].pop("ignored_knob")
    units[1]["expected_minutes"] = 900
    config["empirical_basis"]["nominal_useful_minutes"] = 980
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(CampaignError, match="exceeds active work budget"):
        _validate_config(path, tmp_path, _capabilities())


def test_chemprop_cannot_enter_adaptive_classical_selection() -> None:
    chemprop = {
        "unit_id": "chemprop_0",
        "operation": "chemprop_hpo",
        "stage": "chemprop",
        "spec": {
            "operation": "chemprop_hpo",
            "evaluation_stage": "inner",
            "outer_fold": 0,
            "inner_folds": [0, 1, 2],
        },
    }
    nested = {
        "unit_id": "nested_0",
        "operation": "nested_outer_evaluation",
        "stage": "nested",
        "spec": {"outer_fold": 0, "selection_source_unit_ids": ["chemprop_0"]},
    }
    with pytest.raises(CampaignError, match="only classical HPO"):
        campaign_v4._validate_dependency_sources([chemprop, nested])


def test_scope_and_hash_tampering_are_rejected() -> None:
    value = _self_hashed({"value": 1}, "sha")
    _verify_self_hash(value, "sha")
    value["value"] = 2
    with pytest.raises(CampaignError, match="mismatch"):
        _verify_self_hash(value, "sha")
    with pytest.raises(CampaignError, match="test-label"):
        _validate_scope(
            {
                "source_partition": "train",
                "repository_validation_labels_opened": False,
                "repository_test_labels_opened": True,
                "broad_fixed_dose_pooled_into_pic50": False,
            }
        )


def test_commands_match_versioned_worker_contract(tmp_path: Path) -> None:
    checkpoint = {
        "python_path": sys.executable,
        "worker_path": str(tmp_path / "worker.py"),
        "repo_root": str(tmp_path),
        "base_v2_root": str(tmp_path / "v2"),
        "base_v3_root": str(tmp_path / "v3"),
        "output_root": str(tmp_path / "v4"),
        "workers": 6,
    }
    prepare = {"operation": "prepare", "unit_id": "prepare", "spec": _spec("prepare")}
    assert _command(checkpoint, prepare)[2] == "prepare"
    assert "--base-v2-root" in _command(checkpoint, prepare)
    unit = {
        "operation": "classical_hpo",
        "unit_id": "hpo",
        "spec": _spec("classical_hpo"),
    }
    command = _command(checkpoint, unit)
    assert command[2] == "run-unit"
    assert "--unit-spec-json" in command and "--results-root" in command
    analyze = {"operation": "analyze", "unit_id": "analyze", "spec": _spec("analyze")}
    assert _command(checkpoint, analyze)[2] == "analyze"


def test_dependencies_wait_without_terminal_skip(tmp_path: Path) -> None:
    path, _ = _config(tmp_path)
    units = _validate_config(path, tmp_path, _capabilities())["units"]
    checkpoint = {"units": units}
    hpo = units[1]
    assert not _dependencies_ready(checkpoint, hpo)
    units[0]["status"] = "passed"
    assert _dependencies_ready(checkpoint, hpo)


def test_nested_selection_sources_must_be_inner_upstream_same_outer(tmp_path: Path) -> None:
    path, config = _config(tmp_path)
    units = config["plan"]["units"]
    assert isinstance(units, list)
    nested = _unit(
        "nested_o0",
        "nested",
        "classical_hpo",
        10,
        _spec(
            "classical_hpo",
            candidate=units[1]["spec"]["candidate"],
            evaluation_stage="outer",
            outer_fold=0,
            inner_folds=[0, 1, 2],
            budget_fraction=1.0,
        ),
        [{"stage": "hpo", "minimum_passed": 1, "require_terminal": True}],
    )
    # This fixture operation does not declare selection IDs, so it remains a
    # normal outer diagnostic. Unknown explicit source identifiers still fail.
    nested["spec"]["source_unit_id"] = "missing"
    units.insert(2, nested)
    config["empirical_basis"]["nominal_useful_minutes"] += 10
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(CampaignError, match="unsupported spec keys|unknown source"):
        _validate_config(path, tmp_path, _capabilities())


def test_worker_artifact_validation_binds_executed_material_and_files(tmp_path: Path) -> None:
    path, _ = _config(tmp_path)
    unit = _validate_config(path, tmp_path, _capabilities())["units"][1]
    output = tmp_path / "v4"
    artifact = output / "units/hpo_0/predictions.parquet"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"predictions")
    executed = unit["spec"]
    governance = {
        "source_partition",
        "repository_validation_labels_opened",
        "repository_test_labels_opened",
        "broad_fixed_dose_pooled_into_pic50",
    }
    material = {key: value for key, value in executed.items() if key not in governance}
    _write_worker_json(
        artifact.parent / "unit.json",
        {
            "status": "passed",
            "unit_id": "hpo_0",
            "operation": "classical_hpo",
            "unit_spec": unit["spec"],
            "unit_spec_sha256": _hash_value(unit["spec"], worker=True),
            "executed_spec": executed,
            "executed_spec_sha256": _hash_value(executed, worker=True),
            "material_spec_sha256": _hash_value(material, worker=True),
            "metrics": {"selection_score": 0.44},
            "scientific_scope": _spec("scope"),
            "artifacts": [
                {
                    "role": "oof_predictions",
                    "path": str(artifact),
                    "bytes": artifact.stat().st_size,
                    "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                }
            ],
        },
    )
    _validate_unit(output, unit, _capabilities())
    document = json.loads((artifact.parent / "unit.json").read_text())
    document.pop("unit_json_sha256")
    document["material_spec_sha256"] = "0" * 64
    _write_worker_json(artifact.parent / "unit.json", document)
    with pytest.raises(CampaignError, match="material spec"):
        _validate_unit(output, unit, _capabilities())


def test_retry_is_in_place_and_unavailable_is_not_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0

    def fake_run(*_args: object, **_kwargs: object) -> tuple[int, bool, float]:
        nonlocal calls
        calls += 1
        return 3, False, 0.1

    monkeypatch.setattr("run_local_herg_discovery_campaign_v4._run_process", fake_run)
    monkeypatch.setattr("run_local_herg_discovery_campaign_v4._output_bytes", lambda _path: 0)
    unit = {
        "unit_id": "optional",
        "attempts": 0,
        "operation": "prepare",
        "spec": _spec("prepare"),
    }
    checkpoint = {
        "active_elapsed_seconds": 0.0,
        "invocations": [{"active_elapsed_seconds": 0.0}],
        "campaign_governance": {"maximum_output_gib": 1},
        "units": [unit],
    }
    _atomic_json(tmp_path / "checkpoint.json", checkpoint, "checkpoint_sha256")
    result = _run_with_retry(["worker"], unit, tmp_path, checkpoint, _capabilities(), {}, 10**12)
    assert result[0] == 3 and calls == 1 and unit["attempts"] == 1


def test_resume_adopts_valid_artifact_before_attempt_terminalization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "v4"
    output.mkdir()
    args = type(
        "Args",
        (),
        {
            "repo_root": str(tmp_path),
            "output_root": str(output),
            "config": str(tmp_path / "config.yaml"),
            "protocol": str(tmp_path / "protocol.md"),
            "base_v2_root": str(tmp_path / "v2"),
            "base_v3_root": str(tmp_path / "v3"),
            "worker": str(tmp_path / "worker.py"),
            "python": sys.executable,
        },
    )()
    unit = {
        "unit_id": "done",
        "operation": "classical_hpo",
        "status": "running",
        "attempts": 2,
        "critical": False,
    }
    checkpoint = {
        "repo_root": str(tmp_path.resolve()),
        "output_root": str(output.resolve()),
        "config_path": str(Path(args.config).resolve()),
        "protocol_path": str(Path(args.protocol).resolve()),
        "base_v2_root": str(Path(args.base_v2_root).resolve()),
        "base_v3_root": str(Path(args.base_v3_root).resolve()),
        "worker_path": str(Path(args.worker).resolve()),
        "python_path": str(Path(args.python).absolute()),
        "bindings": [],
        "capabilities": _capabilities(),
        "artifact_adoptions": [],
        "units": [unit],
    }
    _atomic_json(output / "checkpoint.json", checkpoint, "checkpoint_sha256")
    artifact = output / "units/done/unit.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("{}", encoding="utf-8")
    monkeypatch.setattr("run_local_herg_discovery_campaign_v4._validate_unit", lambda *_args: None)
    resumed = _resume(args, output)
    assert resumed["units"][0]["status"] == "passed"
    assert resumed["artifact_adoptions"][0]["attempts_preserved"] == 2
