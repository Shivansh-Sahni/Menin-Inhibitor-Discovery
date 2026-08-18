from __future__ import annotations

import copy

import pytest
from menin_discovery.platform_training_surface_registry import (
    PlatformRegistryError,
    _manifest_hash,
    _validate_science,
)


def _registry_fixture() -> dict:
    return {
        "families": {
            "herg": {
                "broad_trainable_observations": 395_575,
                "confirmed_wt_fixed_dose_structure_labels": 339_373,
                "mutants_admitted": 0,
            },
            "pk_adme": {"modeling_rows": 642_065},
            "affinity_potency": {
                "primary_kd_ki_ic50_observations": 2_000_000,
                "endpoint_observations": {"Kd": 100_000, "Ki": 200_000, "IC50": 1_700_000},
            },
            "multimodal_pretraining": {"prism_verified_finite_viability_values": 8_372_603},
        },
        "execution_state": {"production_feature_store_generated": False},
    }


def test_science_gate_accepts_large_separated_surfaces() -> None:
    _validate_science(_registry_fixture())


def test_science_gate_rejects_mutants_small_endpoints_and_feature_overclaim() -> None:
    for path, value in [
        (("families", "herg", "mutants_admitted"), 1),
        (("families", "affinity_potency", "endpoint_observations", "Kd"), 99_999),
        (("execution_state", "production_feature_store_generated"), True),
    ]:
        registry = copy.deepcopy(_registry_fixture())
        target = registry
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        with pytest.raises(PlatformRegistryError):
            _validate_science(registry)


def test_manifest_hash_excludes_only_its_self_hash() -> None:
    manifest = {"schema_version": "fixture", "inputs": [{"path": "a"}]}
    digest = _manifest_hash(manifest)
    manifest["manifest_sha256"] = digest
    assert _manifest_hash(manifest) == digest
    manifest["inputs"][0]["path"] = "b"
    assert _manifest_hash(manifest) != digest
