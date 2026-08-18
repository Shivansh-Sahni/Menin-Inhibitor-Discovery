from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq
import yaml

ROOT = Path(__file__).resolve().parents[2]


def _json(path: Path) -> dict:
    return json.loads(path.read_text())


def _self_hash(payload: dict) -> str:
    unsigned = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    return hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def test_first_paper_config_binds_final_pre_hpc_releases_and_contracts() -> None:
    config = yaml.safe_load((ROOT / "pipeline/config/herg_first_paper.yaml").read_text())
    assert config["schema_version"] == "herg-first-paper-plan/1.1"
    manifest_keys = [
        "pre_hpc_assets_manifest",
        "candidate_adjudication_manifest",
        "benchmark_v1_5_manifest",
        "wt_reference_manifest",
        "mmp_analysis_manifest",
        "qt_exposure_manifest",
        "hpc_preflight_manifest",
    ]
    for key in manifest_keys:
        path = ROOT / config["inputs"][key]
        assert path.is_file(), key
        payload = _json(path)
        assert payload["manifest_sha256"] == _self_hash(payload), key

    contract = _json(ROOT / config["inputs"]["feature_contract"])
    smoke = _json(ROOT / config["inputs"]["smoke_test_contract"])
    assert contract["contract_version"] == config["pre_hpc_release_contract"]["feature_contract_version"]
    assert smoke["spec_version"] == config["pre_hpc_release_contract"]["smoke_spec_version"]
    assert config["feature_families"]["morgan_fingerprints"]["bit_sizes"] == [2048]
    assert "assay_metadata" not in config["feature_families"]
    assert config["observation_assay_covariates"]["stored_outside_structure_feature_store"]


def test_config_challenges_and_zero_promotion_states_replay() -> None:
    config = yaml.safe_load((ROOT / "pipeline/config/herg_first_paper.yaml").read_text())
    benchmark_manifest = _json(ROOT / config["inputs"]["benchmark_v1_5_manifest"])
    registry_path = (ROOT / config["inputs"]["benchmark_v1_5_manifest"]).parent / benchmark_manifest[
        "artifacts"
    ]["benchmark_challenge_registry_v1_5.parquet"]["path"]
    registry = pq.read_table(registry_path).to_pandas()
    materialized = set(
        registry.loc[registry["status"].eq("materialized_label_blind_membership"), "challenge_id"]
    )
    blocked = set(registry.loc[~registry["status"].eq("materialized_label_blind_membership"), "challenge_id"])
    assert materialized == set(config["split_contract"]["materialized_pre_hpc_challenges"])
    assert blocked == set(config["split_contract"]["blocked_challenges"])
    release = config["pre_hpc_release_contract"]
    assert not release["candidate_evidence_is_gold_standard"]
    assert not release["candidate_human_adjudication_completed"]
    assert not release["qt_exposure_margin_computed"]
    assert not release["qt_or_clinical_context_used_as_herg_label"]
    assert not release["production_model_input_feature_store_generated"]
    assert (
        release["hpc_preflight_blocking_gates_passed"],
        release["hpc_preflight_blocking_gates_total"],
    ) == (2, 7)
