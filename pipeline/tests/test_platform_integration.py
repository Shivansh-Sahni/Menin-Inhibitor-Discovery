"""Lead-owned cross-workstream contracts for the platform expansion."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml
from menin_discovery.platform_cli import build_argument_parser
from menin_discovery.platform_cli import main as platform_main
from menin_discovery.platform_data_pipeline import PUBLIC_ACCESS_CLASS, _assay_family
from menin_discovery.platform_data_schema import ACCESS_CLASSES

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PLATFORM_CONFIG = PROJECT_ROOT / "pipeline" / "config" / "platform.yaml"
NON_HPC_CONFIG = PROJECT_ROOT / "pipeline" / "config" / "non_hpc_readiness.yaml"
STATIC_MANIFEST = PROJECT_ROOT / "research" / "models" / "platform" / "pretraining_static_manifest.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_shared_config_prohibits_substantive_training_and_private_admission() -> None:
    config = yaml.safe_load(PLATFORM_CONFIG.read_text(encoding="utf-8"))

    assert config["project"]["substantive_large_model_training_authorized"] is False
    assert config["pretraining_interface"]["substantive_training_authorized"] is False
    assert config["release"]["public_only"] is True
    assert config["release"]["allowed_access_classes"] == [PUBLIC_ACCESS_CLASS]
    assert PUBLIC_ACCESS_CLASS in ACCESS_CLASSES
    assert config["tasks"]["model_input_contract"] == {
        "current_default_task_modality": "molecule_plus_protein",
        "required_fields": [
            "standardized_smiles",
            "standard_inchi_key",
            "protein_sequence",
        ],
        "missing_input_policy": (
            "retain_source_observation_and_lineage_but_materialize_reasoned_task_exclusion"
        ),
        "ligand_only_default_tasks": [],
    }


def test_shared_herg_policy_uses_the_canonical_data_ontology() -> None:
    config = yaml.safe_load(PLATFORM_CONFIG.read_text(encoding="utf-8"))
    herg = config["tasks"]["herg"]
    continuous = herg["continuous_activity_scope"]
    classifier = herg["classification_task_scope"]

    assert continuous == {
        "endpoint": "IC50",
        "assay_family": _assay_family("IC50", "CHEMBL240", "F", ""),
        "admitted_relations": ["=", "<", "<=", ">", ">=", "interval"],
        "exact_label_kind": "continuous_exact",
        "censored_label_kind": "continuous_censored",
    }
    assert classifier == {
        "endpoint": "IC50",
        "assay_family": _assay_family("IC50", "CHEMBL240", "F", ""),
        "admitted_relations": ["="],
        "label_kind": "categorical",
        "exact_values_only": True,
    }
    assert herg["blocker_max_nm"] == 10_000.0
    assert herg["nonblocker_min_nm"] == 30_000.0
    assert continuous["assay_family"] == classifier["assay_family"] == "herg_functional"
    assert "retained_in_continuous_task" in herg["intermediate_exact_value_policy"]
    assert "retained_in_censored_task" in herg["threshold_crossing_or_censored_interval_policy"]


def test_non_hpc_decision_config_records_bounded_pilot_without_authorizing_training() -> None:
    config = yaml.safe_load(NON_HPC_CONFIG.read_text(encoding="utf-8"))

    assert config["training"]["substantive_large_model_training_authorized"] is False
    assert config["training"]["substantive_large_model_training_started"] is False
    assert config["task_decision"]["selected_task_id"] == (
        "default::default__herg__ic50__herg_functional__nm__continuous_exact"
    )
    assert config["task_decision"]["primary_evaluation_strategy"] == "scaffold"
    assert config["task_decision"]["role"].startswith("bounded_cpu_pilot_only")
    assert config["leakage"]["deep_audit"]["model_ready_test_lockbox_opened_or_hashed"] is False
    assert config["model_selection"]["checkpoint_download_authorized"] is False


def test_static_pretraining_manifest_is_portable_and_training_remains_disabled() -> None:
    manifest = json.loads(STATIC_MANIFEST.read_text(encoding="utf-8"))

    assert manifest["substantive_training_started"] is False
    assert manifest["schema_version"] == "static_pretraining_readiness_manifest_v1"
    assert len(manifest["artifacts"]) >= 4
    for record in manifest["artifacts"].values():
        relative = Path(record["path"])
        assert not relative.is_absolute()
        artifact = (STATIC_MANIFEST.parent / relative).resolve()
        assert artifact.is_file()
        assert _sha256(artifact) == record["sha256"]


def test_platform_status_is_read_only_and_preserves_the_no_training_boundary(
    tmp_path: Path,
    capsys,
) -> None:
    before = list(tmp_path.iterdir())
    assert platform_main(["--project-root", str(tmp_path), "status"]) == 0
    status = json.loads(capsys.readouterr().out)

    assert status["schema_version"] == "protein_molecule_platform_status_v1"
    assert status["substantive_training_started"] is False
    assert status["large_model_training_started"] is False
    assert status["interpretation"].startswith("inventory_only")
    assert list(tmp_path.iterdir()) == before


def test_platform_contract_surface_exposes_no_training_command(capsys) -> None:
    assert platform_main(["contracts"]) == 0
    contract = json.loads(capsys.readouterr().out)

    assert contract["training_command_exposed"] is False
    assert contract["substantive_training_started"] is False
    assert contract["large_model_training_started"] is False
    assert {
        "chemprop_dmpnn",
        "chai_1_external_evaluation",
        "boltz_2_external_evaluation",
        "nesso_1_external_evaluation",
    }.issubset(contract["frozen_model_candidates"])


def test_platform_command_surface_covers_every_pretraining_readiness_stage() -> None:
    parser = build_argument_parser()
    subparser_action = next(action for action in parser._actions if getattr(action, "choices", None))
    commands = set(subparser_action.choices)

    assert {
        "normalize-chembl-exports",
        "canonicalize-chembl",
        "verify-canonical-determinism",
        "acquire-external",
        "verify-external",
        "normalize-external",
        "verify-external-normalized",
        "analyze-external-admission",
        "verify-external-admission",
        "analyze-deep-leakage",
        "verify-deep-leakage",
        "prepare-structure-metadata",
        "verify-structure-metadata",
        "prepare-context-splits",
        "verify-context-splits",
        "prepare-clinical-results",
        "verify-clinical-results",
        "prepare-regulatory-records",
        "verify-regulatory-records",
        "acquire-pkdb-candidates",
        "prepare-pkdb-candidates",
        "verify-pkdb-candidates",
        "analyze-canonical",
        "verify-statistical-analysis",
        "prepare-split-suite",
        "verify-split-suite",
        "prepare-static",
        "integrate-task",
        "diagnose-task",
        "prepare-corpus-readiness",
        "verify-corpus-readiness",
        "verify-final-artifacts",
        "audit-non-hpc-governance",
        "verify-non-hpc-completion",
    }.issubset(commands)
    assert not any("train" in command for command in commands)

    corpus = parser.parse_args(["prepare-corpus-readiness"])
    assert corpus.seed == 20260804
    assert corpus.output_directory == Path("research/models/platform/corpus_readiness/full_chembl37")

    split_suite = parser.parse_args(["prepare-split-suite"])
    assert split_suite.seed == 20260804
    assert split_suite.output_directory == Path("research/data/platform/splits/full_chembl37")
    assert split_suite.train_fraction + split_suite.validation_fraction + split_suite.test_fraction == 1.0
