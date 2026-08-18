#!/usr/bin/env python3
"""Structural validator for pre-HPC contracts; performs no feature/model work."""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LANDSCAPE = ROOT.parent / "model_landscape_v2"


def j(name: str) -> dict:
    with (ROOT / name).open(encoding="utf-8") as handle:
        return json.load(handle)


def c(name: str) -> list[dict[str, str]]:
    with (ROOT / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    feature = j("future_feature_contract.json")
    registry = j("competitor_reproduction_readiness.json")
    registry_csv = c("competitor_reproduction_readiness.csv")
    hpc = j("hpc_execution_plan.json")
    smoke = j("smoke_test_spec.json")
    priority_map = c("priority_reproduction_map.csv")

    assert feature["execution_status"] == "specification_only_no_features_computed"
    assert feature["contract_version"] == "1.1.0" and feature["as_of"] == "2026-08-09"
    assert feature["production_gate"].startswith("all_required_exact_versions_non_NR")
    assert feature["canonical_parent_join_key"] == "structure_id"
    levels = set(feature["feature_level_enum"])
    assert levels == {
        "parent",
        "protomer_tautomer",
        "conformer",
        "protein_construct",
        "protein_conformation",
        "pose",
        "complex",
    }
    assert set(feature["hierarchy"]) == levels
    assert feature["row_key_contract"]["structure_id_is_not_globally_unique"] is True
    assert {"feature_family", "feature_version", "feature_level", "structure_id"} <= set(
        feature["row_key_contract"]["composite_fields"]
    )
    scopes = feature["identity_fields_by_scope"]
    assert "structure_id" in scopes["ligand_only"]["required"]
    assert "protein_construct_id" in scopes["protein_only"]["required"]
    assert scopes["protein_only"]["duplication_per_structure_forbidden"] is True
    nullable = feature["nullable_hierarchy_fields_by_level"]
    assert nullable["parent"]["protomer_id"] is True
    assert nullable["conformer"]["conformer_id"] is False
    assert nullable["pose"]["pose_id"] is False
    assert set(nullable) == levels
    assert "zero_vector_on_failure" in feature["missingness"]
    assert feature["missingness"]["zero_vector_on_failure"] is False
    assert len(feature["feature_status_enum"]) >= 8
    forbidden_feature_words = {"label", "activity", "clinical_stage", "test_outcome"}
    family_ids = {row["id"] for row in feature["families"]}
    assert len(family_ids) == 12
    assert not (forbidden_feature_words & family_ids)
    assert feature["post_fit_outputs"]["is_input_feature_family"] is False
    assert all(set(row["allowed_levels"]) <= levels and row["allowed_levels"] for row in feature["families"])
    assert not any("level" in row for row in feature["families"])
    environment = feature["software_environment"]
    assert environment["python"]["constraint"] == "3.11.*"
    assert environment["rdkit"]["exact_version"] == "2026.03.3"
    assert environment["pytorch"]["constraint"] == "2.7.*"
    assert environment["container_digest"] == "NR"
    assert environment["lockfile_sha256"] == "NR"
    assert "uncertainty" in feature["post_fit_outputs"]["fields"]
    rdkit_versions = {row["version"] for row in feature["families"] if row.get("software") == "RDKit"}
    assert rdkit_versions == {"2026.03.3"}
    protein = next(row for row in feature["families"] if row["id"] == "herg_protein_embedding")
    assert protein["wt_sequence_accession"] == "Q12809"
    assert protein["wt_sequence_length"] == 1159
    assert protein["wt_sequence_sha256"] == "287332153da38b59cc1be9554cc3a29f14d3b9e2a33150b4d54137773b22d1f7"
    assert len(protein["wt_reference_manifest_sha256"]) == 64
    conditions = feature["enumeration_conditions"]
    assert conditions["pH"] == 7.4 and conditions["temperature_K"] == 298.15
    assert conditions["minimum_predicted_population"] == 0.01
    aggregation = feature["aggregation"]
    assert aggregation["cross_state_energy_comparison"] is False
    assert aggregation["within_state_energy_zero"] == "minimum_valid_conformer_energy_for_that_state"
    assert aggregation["state_weights"] == "pH_specific_predicted_population"

    csv_ids = {row["model_id"] for row in registry_csv}
    json_ids = {row["model_id"] for row in registry["systems"]}
    assert len(csv_ids) == len(json_ids) == 15
    assert csv_ids == json_ids
    assert registry["execution_status"] == "registry_only_no_reproductions_run"
    json_by_id = {row["model_id"]: row for row in registry["systems"]}
    parity_fields = [
        "priority_link",
        "code_url",
        "audited_commit",
        "checkpoint",
        "license",
        "required_input",
        "preprocessing",
        "overlap_audit",
        "adapter_needed",
        "status",
        "blocking_or_next_action",
    ]
    for row in registry_csv:
        assert row["code_url"]
        assert row["checkpoint"]
        assert row["license"]
        assert row["required_input"] and row["preprocessing"]
        assert row["overlap_audit"] and row["adapter_needed"] and row["status"]
        machine_row = json_by_id[row["model_id"]]
        for field in parity_fields:
            assert row[field] == machine_row[field], f"registry parity mismatch: {row['model_id']} {field}"

    if (LANDSCAPE / "model_comparison_matrix.json").exists():
        landscape_ids = {x["model_id"] for x in j_from(LANDSCAPE / "model_comparison_matrix.json")["models"]}
        assert landscape_ids == json_ids, "readiness registry does not match 15-system landscape"
    assert len(priority_map) == 14
    assert len({row["work_item"] for row in priority_map}) == 14
    if (LANDSCAPE / "benchmark_adapter_priority_matrix.json").exists():
        landscape_items = {
            x["work_item"] for x in j_from(LANDSCAPE / "benchmark_adapter_priority_matrix.json")["items"]
        }
        assert landscape_items == {row["work_item"] for row in priority_map}

    assert hpc["execution_status"] == "plan_only_no_hpc_jobs_submitted"
    assert hpc["all_resource_values_are_estimates"] is True
    assert {x["id"] for x in hpc["stages"]} == {"S0", "S1", "S2", "S3", "S4", "S5", "S6"}
    assert hpc["storage_zones"] == [
        "raw_immutable",
        "standardized_versioned",
        "feature_objects_content_addressed",
        "runs",
    ]
    assert (
        next(x for x in hpc["stages"] if x["id"] == "S6")["gate"]
        == "P0_baselines_frozen_and_ablation_preregistered"
    )
    assert smoke["execution_status"] == "specified_not_run"
    assert smoke["spec_version"] == "1.1.0" and smoke["as_of"] == "2026-08-09"
    assert smoke["valid_training_fixture"]["allowed_molecule_count"] == {
        "minimum": 10,
        "default": 32,
        "maximum": 100,
    }
    assert "training partition only" in smoke["valid_training_fixture"]["selection"]
    assert smoke["negative_control_fixture"]["separate_from_valid_count"] is True
    assert smoke["negative_control_fixture"]["never_training_or_evaluation_data"] is True
    assert smoke["runs"] == 2 and len(smoke["assertions"]) >= 10
    assert "no_duplicate_composite_feature_row_keys" in smoke["assertions"]

    print(
        f"PASS: {len(feature['families'])} input-feature families plus separate post-fit outputs; no features computed"
    )
    print("PASS: parent join/composite row keys, complete hierarchy, RDKit pin, and within-state aggregation")
    print("PASS: 15-system CSV/JSON row parity and model-landscape match")
    print("PASS: 14-priority readiness crosswalk matches priority landscape")
    print("PASS: 7 estimated HPC stages and unexecuted 10-100 molecule smoke specification")


def j_from(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


if __name__ == "__main__":
    main()
