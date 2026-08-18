from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from menin_edit import cli
from menin_edit.config import load_config
from menin_edit.registry import PredictorRegistry


def _private_args(output_dir: Path) -> argparse.Namespace:
    return argparse.Namespace(
        output_dir=output_dir,
        workbook=output_dir.parent / "must-not-be-opened.xlsx",
        key_env="MISSING_TEST_SECRET",
        cohort_role="development",
        endpoints=None,
        overwrite=False,
    )


def test_default_configuration_is_public_only_and_registry_is_constructible() -> None:
    config = load_config(cli.DEFAULT_CONFIG)

    private = config.model_configs["herg_private_ensemble_probability"]
    consensus = config.model_configs["herg_consensus_probability"]
    assert private["enabled"] is False
    assert consensus["enabled"] is True
    assert consensus["members"] == ["herg_public_blocker_probability"]

    registry = PredictorRegistry.from_config(
        config.model_configs,
        repository_root=config.repository_root,
    )
    assert "herg_private_ensemble_probability" not in registry.predictors
    consensus_predictor = registry.predictors["herg_consensus_probability"]
    assert [member.endpoint for member in consensus_predictor.predictors] == [
        "herg_public_blocker_probability"
    ]


def test_private_output_guard_rejects_repository_and_symlink_targets(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    inside = repository / "private-output"

    with pytest.raises(ValueError, match="outside the Git repository"):
        cli._validated_private_output_directory(inside, repository_root=repository)

    approved_link = tmp_path / "apparently-approved"
    approved_link.symlink_to(repository, target_is_directory=True)
    with pytest.raises(ValueError, match="outside the Git repository"):
        cli._validated_private_output_directory(
            approved_link / "models",
            repository_root=repository,
        )

    approved = tmp_path / "approved" / "models"
    assert (
        cli._validated_private_output_directory(
            approved,
            repository_root=repository,
        )
        == approved.resolve()
    )


def test_prepare_lab_rejects_repository_before_private_data_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    private_access_attempted = False

    def forbidden_private_access(_args: argparse.Namespace):
        nonlocal private_access_attempted
        private_access_attempted = True
        raise AssertionError("private data must not be accessed")

    monkeypatch.setattr(cli, "_lab_tables", forbidden_private_access)
    with pytest.raises(ValueError, match="outside the Git repository"):
        cli._prepare_lab_outputs(
            _private_args(repository / "derived"),
            repository_root=repository,
        )
    assert private_access_attempted is False


def test_train_lab_rejects_repository_before_workbook_read_or_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    workbook_read = False
    training_started = False

    def forbidden_workbook_read(*_args, **_kwargs):
        nonlocal workbook_read
        workbook_read = True
        raise AssertionError("workbook must not be opened")

    def forbidden_training(*_args, **_kwargs):
        nonlocal training_started
        training_started = True
        raise AssertionError("training must not start")

    monkeypatch.setattr(cli, "load_historical_lab_workbook", forbidden_workbook_read)
    monkeypatch.setattr(cli, "train_local_regression", forbidden_training)
    with pytest.raises(ValueError, match="outside the Git repository"):
        cli._train_lab_models(
            _private_args(repository / "models"),
            repository_root=repository,
        )
    assert workbook_read is False
    assert training_started is False


def test_train_lab_cli_wires_repository_guard_before_workbook_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workbook_read = False

    def forbidden_workbook_read(*_args, **_kwargs):
        nonlocal workbook_read
        workbook_read = True
        raise AssertionError("workbook must not be opened")

    monkeypatch.setattr(cli, "load_historical_lab_workbook", forbidden_workbook_read)
    with pytest.raises(ValueError, match="outside the Git repository"):
        cli.main(
            [
                "--config",
                str(cli.DEFAULT_CONFIG),
                "train-lab",
                "--workbook",
                "must-not-be-opened.xlsx",
                "--output-dir",
                str(cli.PACKAGE_ROOT / "artifacts" / "private-models"),
            ]
        )
    assert workbook_read is False
