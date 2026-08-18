from __future__ import annotations

import ast
import csv
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml
from menin_discovery.research_hpc import (
    RECEPTOR_RAW_COORDINATE_PATHS,
    HPCBundleConfig,
    PilotSelectionConfig,
    _herg_cross_class_pair_map,
    _select_herg_pilots,
    create_hpc_bundle,
    generate_hpc_bundles,
    select_pilot_compounds,
)


def _write_raw_receptor_coordinates(project_root: Path) -> None:
    for pdb_id, relative_path in RECEPTOR_RAW_COORDINATE_PATHS.items():
        path = project_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            f"""data_{pdb_id}
_entry.id {pdb_id}
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.auth_asym_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.pdbx_PDB_model_num
ATOM 1 A 1.0 2.0 3.0 1
#
""",
            encoding="utf-8",
        )


def test_herg_pilot_selection_enforces_class_quota_and_complete_cross_class_pairs():
    candidates = pd.DataFrame(
        {
            "compound_id": ["B1", "B2", "B3", "N1", "N2", "N3"],
            "standardized_smiles": [
                "N[C@@H](C)C(=O)Oc1ccccc1",
                "N[C@@H](CC)C(=O)Oc1ccncc1",
                "N[C@@H](CO)C(=O)Oc1ccc(F)cc1",
                "N[C@H](C)C(=O)Oc1ccccc1",
                "N[C@H](CC)C(=O)Oc1ccncc1",
                "N[C@H](CO)C(=O)Oc1ccc(F)cc1",
            ],
            "herg_class": ["blocker", "blocker", "blocker", "nonblocker", "nonblocker", "nonblocker"],
            "mw": [122.0, 123.0, 140.0, 121.0, 122.0, 139.0],
            "polar": [10.0, 20.0, 12.0, 18.0, 28.0, 20.0],
            "series_id": ["S1", "S2", "S3", "S1", "S2", "S3"],
        }
    )

    pair_map = _herg_cross_class_pair_map(candidates)
    selected, selected_pairs = _select_herg_pilots(
        candidates,
        feature_columns=["mw", "polar"],
        selection_config=PilotSelectionConfig(),
    )

    assert len(pair_map) >= 1
    assert selected["herg_class"].value_counts().to_dict() == {"blocker": 3, "nonblocker": 3}
    complete = selected.dropna(subset=["herg_matched_pair_id"]).groupby("herg_matched_pair_id")
    assert any(set(group["herg_class"]) == {"blocker", "nonblocker"} for _, group in complete)
    assert selected_pairs.equals(pair_map)


def _candidates():
    return pd.DataFrame(
        {
            "compound_id": ["A", "B", "C", "D", "E", "F"],
            "mw": [650, 655, 700, 730, 760, 810],
            "polar": [100, 102, 130, 160, 120, 180],
            "matched_pair_id": ["PAIR-1", "PAIR-1", None, None, "PAIR-2", "PAIR-2"],
            "series_id": ["S1", "S1", "S2", "S3", "S4", "S4"],
            "herg_class": ["low", "high", "low", "high", "mid", "high"],
            "pk_data_present": [True, True, False, True, False, True],
            "endpoint": [1.0, 9.0, 3.0, 7.0, 2.0, 8.0],
        }
    )


def test_pilot_selection_is_deterministic_and_preserves_matched_pair_archetype():
    config = PilotSelectionConfig(
        n_select=5,
        matched_pair_slots=2,
        endpoint_column="endpoint",
        strata_columns=("herg_class",),
    )
    first = select_pilot_compounds(_candidates(), feature_columns=["mw", "polar"], config=config)
    second = select_pilot_compounds(_candidates(), feature_columns=["mw", "polar"], config=config)

    assert first["compound_id"].tolist() == second["compound_id"].tolist()
    assert len(first) == 5
    paired = first.loc[first["selection_reason"].str.contains("matched_pair")]
    assert len(paired) == 2
    assert paired["matched_pair_id"].nunique() == 1
    assert first["pilot_rank"].tolist() == list(range(1, 6))
    assert first["selection_feature_columns"].str.contains("mw").all()


def test_bundle_contains_exact_smoke_guard_protocols_and_acceptance_gates(tmp_path):
    pilots = _candidates().iloc[:3].copy()
    outputs = create_hpc_bundle(tmp_path, pilots=pilots, config=HPCBundleConfig())

    manifest = json.loads(outputs["manifest_path"].read_text())
    assert manifest["production_launch_enabled"] is False
    assert manifest["production_submission_scripts_present"] is False
    assert manifest["prepared_systems_present"] is False
    assert manifest["completed_simulations_present"] is False
    assert manifest["readiness_status"] == "blocked_input_preparation"
    assert manifest["local_smoke_duration_ns"] == 2.0
    assert manifest["local_smoke_total_steps"] == 1_000_000
    assert outputs["receptor_count"] == 6
    readiness = json.loads((tmp_path / "readiness_initial.json").read_text())
    receptor_blocker = next(
        item for item in readiness["blockers"] if item["code"] == "unprepared_receptor_ensemble"
    )
    assert receptor_blocker["count"] == 6
    assert "six selected deposited coordinates" in receptor_blocker["remediation"]

    membrane = yaml.safe_load(outputs["membrane_pmf_protocol"].read_text())
    assert membrane["patch_size_sensitivity_popc"] == [64, 128, 256]
    assert membrane["primary_patch_popc"] == 128
    assert membrane["replicates"] == 3
    assert membrane["acceptance_gates"]["last_half_drift_kcal_mol_max"] == 0.5
    assert membrane["acceptance_gates"]["local_global_z_r2_min"] == 0.95
    assert membrane["production_launch_enabled"] is False

    for script in tmp_path.rglob("*.py"):
        ast.parse(script.read_text())
    smoke_runner = (tmp_path / "environment_md" / "run_openmm_smoke.py").read_text()
    assert "Only the exact local 2 ns smoke protocol is permitted" in smoke_runner
    assert "TOTAL_STEPS = 1_000_000" in smoke_runner
    assert "NVT_STEPS = 100_000" in smoke_runner
    assert "--checkpoint-in" in smoke_runner
    assert "--pause-at-step" in smoke_runner
    assert "smoke_complete.chk" in smoke_runner
    assert not list(tmp_path.rglob("*production*.slurm"))
    assert "<LIGAND_HEAVY_ATOMS>" in (tmp_path / "membrane_pmf" / "plumed_umbrella.dat").read_text()

    ligand_states = pd.read_csv(outputs["ligand_states_path"])
    assert ligand_states["ligand_state_id"].str.startswith("UNRESOLVED::").all()
    environment_runs = pd.read_csv(tmp_path / "environment_md" / "run_matrix.csv")
    membrane_runs = pd.read_csv(tmp_path / "membrane_pmf" / "run_matrix.csv")
    herg_runs = pd.read_csv(tmp_path / "herg_ensemble" / "run_matrix.csv")
    assert len(environment_runs) == 45
    assert (environment_runs["run_requirement"] == "required").sum() == 18
    assert len(membrane_runs) == 90
    assert (membrane_runs["run_requirement"] == "required").sum() == 36
    assert len(herg_runs) == 36
    assert herg_runs["receptor_pdb_id"].isna().all()
    assert all(frame["run_id"].is_unique for frame in (environment_runs, membrane_runs, herg_runs))
    assert all(frame["random_seed"].is_unique for frame in (environment_runs, membrane_runs, herg_runs))
    assert (environment_runs["random_seed"] > 0).all()
    commands = pd.read_csv(tmp_path / "environment_md" / "smoke_commands.csv")
    assert commands["restart_smoke_command"].str.contains("--checkpoint-in").all()
    assert commands["restart_test_part1_command"].str.contains("--pause-at-step 500000").all()
    assert commands["restart_test_part2_command"].str.contains("--checkpoint-in").all()
    assert commands["fresh_smoke_command"].str.contains("--seed").all()
    assert not commands["executable_now"].any()
    assert (environment_runs["force_field_role"] == "predeclared_sensitivity_subset").sum() == 18

    receptors = pd.read_csv(outputs["receptors_path"])
    assert set(receptors["pdb_id"]) == {"8ZYN", "8ZYO", "8ZYP", "8ZYQ", "9CHP", "9CHQ"}
    assert receptors["production_role"].eq("canonical_raw_coordinate").all()
    sensitivity = pd.read_csv(outputs["sensitivity_subset_path"])
    assert sensitivity["compound_id"].tolist()[:2] == ["A", "B"]


def test_bundle_preflight_is_a_hard_block_until_inputs_are_prepared(tmp_path):
    create_hpc_bundle(tmp_path, pilots=_candidates().iloc[:2], config=HPCBundleConfig())
    command = [
        sys.executable,
        str(tmp_path / "preflight.py"),
        "--bundle-dir",
        str(tmp_path),
        "--output",
        "preflight_report.json",
    ]
    blocked = subprocess.run(command, check=False, capture_output=True, text=True)
    assert blocked.returncode == 2
    report = json.loads((tmp_path / "preflight_report.json").read_text())
    assert report["status"] == "blocked_input_preparation"
    codes = {item["code"] for item in report["blockers"]}
    assert "missing_or_nonportable_prepared_system_xml" in codes
    assert "unresolved_ligand_state" in codes
    assert "software_environment_not_recorded" in codes
    conditional_codes = {item["code"] for item in report["conditional_blockers"]}
    assert "rbfe_closed_cycle_not_ready" in conditional_codes

    reported = subprocess.run([*command, "--report-only"], check=False)
    assert reported.returncode == 0


def test_preflight_can_release_required_smokes_while_conditional_physics_stays_disabled(tmp_path):
    base = pd.DataFrame(
        {
            "compound_id": [f"C{index:02d}" for index in range(12)],
            "pilot_rank": range(1, 13),
            "matched_pair_id": ["P1", "P1", "P2", "P2", *([None] * 8)],
            "ph": [7.4] * 12,
        }
    )
    herg = base.iloc[:6].copy()
    herg["herg_class"] = ["blocker"] * 3 + ["nonblocker"] * 3
    herg["herg_matched_pair_id"] = ["HMP-01", None, None, "HMP-01", None, None]
    create_hpc_bundle(
        tmp_path,
        pilots=base,
        config=HPCBundleConfig(),
        workflow_pilots={
            "environment_md": base,
            "membrane_pmf": base.iloc[:4],
            "herg_ensemble": herg,
            "relative_free_energy": base.iloc[:4],
        },
    )
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    for filename in ("system.xml", "coordinates.pdb", "ligand.prm", "pose.pdb", "receptor.pdb"):
        (prepared / filename).write_text(f"test {filename}\n")
    for workflow in ("environment_md", "membrane_pmf", "herg_ensemble"):
        path = tmp_path / workflow / "run_matrix.csv"
        runs = pd.read_csv(path)
        required = runs["run_requirement"] == "required"
        text_columns = [
            "ligand_state_id",
            "ligand_parameter_file",
            "parameter_review_status",
            "prepared_system_xml",
            "coordinates",
            "readiness_status",
        ]
        if workflow == "herg_ensemble":
            text_columns.extend(["receptor_pdb_id", "receptor_state", "pose_coordinates"])
        runs[text_columns] = runs[text_columns].astype(object)
        runs.loc[required, "ligand_state_id"] = "STATE-READY"
        runs.loc[required, "formal_charge"] = 1
        runs.loc[required, "ligand_parameter_file"] = "prepared/ligand.prm"
        runs.loc[required, "parameter_review_status"] = "approved"
        runs.loc[required, "prepared_system_xml"] = "prepared/system.xml"
        runs.loc[required, "coordinates"] = "prepared/coordinates.pdb"
        runs.loc[required, "readiness_status"] = "ready"
        if workflow == "herg_ensemble":
            runs.loc[required, "receptor_pdb_id"] = "9CHP"
            runs.loc[required, "receptor_state"] = "high_K_C4"
            runs.loc[required, "pose_coordinates"] = "prepared/pose.pdb"
        runs.to_csv(path, index=False)
    receptors_path = tmp_path / "herg_ensemble" / "receptors.csv"
    receptors = pd.read_csv(receptors_path)
    receptor_text_columns = [
        "source_structure_path",
        "prepared_receptor_path",
        "ion_occupancy_assignment",
        "missing_residue_review",
        "protonation_review",
        "readiness_status",
    ]
    receptors[receptor_text_columns] = receptors[receptor_text_columns].astype(object)
    receptor = receptors["pdb_id"] == "9CHP"
    receptors.loc[receptor, "source_structure_path"] = "prepared/receptor.pdb"
    receptors.loc[receptor, "prepared_receptor_path"] = "prepared/receptor.pdb"
    receptors.loc[receptor, "ion_occupancy_assignment"] = "reviewed_high_K_C4"
    receptors.loc[receptor, "missing_residue_review"] = "approved"
    receptors.loc[receptor, "protonation_review"] = "approved"
    receptors.loc[receptor, "readiness_status"] = "ready"
    receptors.to_csv(receptors_path, index=False)
    (tmp_path / "software_environment.json").write_text(
        json.dumps(
            {
                "openmm_version": "reviewed-test",
                "pymbar_version": "reviewed-test",
                "plumed_version": "reviewed-test",
                "parameterization_tool_versions": {"primary": "reviewed-test"},
                "plumed_openmm_integration_verified": True,
                "review_status": "approved_for_local_smoke",
            }
        )
    )
    command = [
        sys.executable,
        str(tmp_path / "preflight.py"),
        "--bundle-dir",
        str(tmp_path),
        "--output",
        "preflight_report.json",
    ]
    assert subprocess.run(command, check=False).returncode == 0
    report = json.loads((tmp_path / "preflight_report.json").read_text())
    assert report["blocker_count"] == 0
    assert report["status"] == "ready_for_required_local_smoke_with_conditional_workflows_disabled"
    conditional_codes = {item["code"] for item in report["conditional_blockers"]}
    assert conditional_codes == {
        "force_field_sensitivity_runs_not_ready",
        "rbfe_closed_cycle_not_ready",
    }


def test_rbfe_validator_requires_closed_charge_preserving_local_cycle(tmp_path):
    create_hpc_bundle(tmp_path, pilots=_candidates(), config=HPCBundleConfig())
    rbfe = tmp_path / "relative_free_energy"
    edges_path = rbfe / "rbfe_edges.csv"
    cycles_path = rbfe / "rbfe_cycles.csv"
    edge_fields = [
        "edge_id",
        "ligand_a",
        "ligand_b",
        "formal_charge_a",
        "formal_charge_b",
        "ligand_state_a",
        "ligand_state_b",
        "protonation_family_a",
        "protonation_family_b",
        "receptor_state",
        "pose_family",
        "local_perturbation_approved",
    ]
    with edges_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=edge_fields)
        writer.writeheader()
        for edge_id, ligand_a, ligand_b in (("AB", "A", "B"), ("BC", "B", "C"), ("CA", "C", "A")):
            writer.writerow(
                {
                    "edge_id": edge_id,
                    "ligand_a": ligand_a,
                    "ligand_b": ligand_b,
                    "formal_charge_a": "1",
                    "formal_charge_b": "1",
                    "ligand_state_a": f"{ligand_a}-cation",
                    "ligand_state_b": f"{ligand_b}-cation",
                    "protonation_family_a": "cationic",
                    "protonation_family_b": "cationic",
                    "receptor_state": "9CHP",
                    "pose_family": "pose-1",
                    "local_perturbation_approved": "true",
                }
            )
    pd.DataFrame(
        [
            {
                "cycle_id": "cycle-1",
                "ordered_ligands": "A>B>C",
                "receptor_state": "9CHP",
                "pose_family": "pose-1",
            }
        ]
    ).to_csv(cycles_path, index=False)
    output = rbfe / "validation.json"
    command = [
        sys.executable,
        str(rbfe / "validate_rbfe_cycles.py"),
        "--edges",
        str(edges_path),
        "--cycles",
        str(cycles_path),
        "--output",
        str(output),
    ]
    passed = subprocess.run(command, check=False)
    assert passed.returncode == 0
    assert json.loads(output.read_text())["eligible_for_rbfe"] is True

    edges = pd.read_csv(edges_path)
    edges.loc[0, "formal_charge_b"] = 0
    edges.to_csv(edges_path, index=False)
    failed = subprocess.run(command, check=False)
    assert failed.returncode == 2
    failures = json.loads(output.read_text())["failures"]
    assert any(item["code"] == "charge_change" for item in failures)

    edges.loc[0, "formal_charge_b"] = 1
    edges.loc[1, "ligand_state_a"] = "B-alternate-cation"
    edges.to_csv(edges_path, index=False)
    assert subprocess.run(command, check=False).returncode == 2
    failures = json.loads(output.read_text())["failures"]
    assert any(item["code"] == "inconsistent_ligand_state_across_cycle" for item in failures)


def test_convergence_evaluator_requires_every_declared_metric(tmp_path):
    outputs = create_hpc_bundle(tmp_path, pilots=_candidates().iloc[:2], config=HPCBundleConfig())
    workflow = tmp_path / "environment_md"
    protocol = yaml.safe_load(outputs["environment_md_protocol"].read_text())
    metrics_path = workflow / "metrics.json"
    output_path = workflow / "convergence.json"
    metrics_path.write_text("{}\n")
    command = [
        sys.executable,
        str(workflow / "evaluate_convergence.py"),
        "--protocol",
        str(workflow / "protocol.yaml"),
        "--metrics",
        str(metrics_path),
        "--output",
        str(output_path),
    ]
    assert subprocess.run(command, check=False).returncode == 2
    assert json.loads(output_path.read_text())["eligible_for_model_ingestion"] is False

    passing = {}
    for gate, threshold in protocol["acceptance_gates"].items():
        passing[gate] = threshold if ("minimum" in gate or gate.endswith("_min")) else 0.0
    metrics_path.write_text(json.dumps(passing))
    assert subprocess.run(command, check=False).returncode == 0
    assert json.loads(output_path.read_text())["eligible_for_model_ingestion"] is True


def test_smoke_validator_requires_exact_endpoint_finite_stability_and_exercised_restart(tmp_path):
    outputs = create_hpc_bundle(tmp_path, pilots=_candidates().iloc[:1], config=HPCBundleConfig())
    workflow = tmp_path / "environment_md"
    smoke_dir = workflow / "runs" / "synthetic"
    smoke_dir.mkdir(parents=True)
    for name in ("smoke.dcd", "smoke_complete.chk"):
        (smoke_dir / name).write_bytes(b"test-placeholder")
    rows = []
    for step in range(5_000, 1_000_001, 5_000):
        rows.append(
            {
                '#"Step"': step,
                "Time (ps)": step * 0.002,
                "Potential Energy (kJ/mole)": -1000.0 + step / 1_000_000,
                "Temperature (K)": 310.0,
                "Density (g/mL)": 1.0,
            }
        )
    pd.DataFrame(rows).to_csv(smoke_dir / "smoke.csv", index=False)
    (smoke_dir / "smoke_run_record.json").write_text(
        json.dumps(
            {
                "completed_steps": 1_000_000,
                "restart_validation_completed": True,
                "status": "completed_local_smoke_not_production",
            }
        )
    )
    output = smoke_dir / "validation.json"
    command = [
        sys.executable,
        str(workflow / "validate_smoke.py"),
        "--protocol",
        str(outputs["environment_md_protocol"]),
        "--smoke-dir",
        str(smoke_dir),
        "--output",
        str(output),
    ]
    assert subprocess.run(command, check=False).returncode == 0
    assert json.loads(output.read_text())["eligible_for_production_preparation_review"] is True

    record = json.loads((smoke_dir / "smoke_run_record.json").read_text())
    record["restart_validation_completed"] = False
    (smoke_dir / "smoke_run_record.json").write_text(json.dumps(record))
    assert subprocess.run(command, check=False).returncode == 2
    assert "restart_path_not_exercised" in json.loads(output.read_text())["failures"]


def test_bundle_manifests_and_run_matrices_are_deterministic(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    create_hpc_bundle(first, pilots=_candidates(), config=HPCBundleConfig())
    create_hpc_bundle(second, pilots=_candidates(), config=HPCBundleConfig())
    assert json.loads((first / "manifest.json").read_text()) == json.loads(
        (second / "manifest.json").read_text()
    )
    for relative in (
        "ligand_state_matrix.csv",
        "environment_md/run_matrix.csv",
        "membrane_pmf/run_matrix.csv",
        "herg_ensemble/run_matrix.csv",
    ):
        assert (first / relative).read_bytes() == (second / relative).read_bytes()


def test_only_exact_two_ns_smoke_is_allowed():
    with pytest.raises(ValueError, match="exactly 2 ns"):
        HPCBundleConfig(smoke_duration_ns=1.0)


def test_generate_hpc_bundles_integration_api(tmp_path):
    compounds = _candidates()
    compounds["standardized_smiles"] = ["CCN", "CCO", "CCC", "CCCl", "CCBr", "CCF"]
    physics_summary = compounds[["compound_id"]].copy()
    physics_summary["structure_id"] = [f"STR-{index}" for index in range(len(compounds))]
    physics_summary["ph"] = 7.4
    physics_summary["pka_scenario"] = "nominal"
    physics_summary["sa_3d_psa_ang2__mean"] = compounds["polar"]
    physics_summary["radius_of_gyration_angstrom__mean"] = compounds["mw"] / 100

    outputs = generate_hpc_bundles(
        compounds,
        physics_summary,
        tmp_path,
        {"pilot_selection": {"n_select": 4, "matched_pair_slots": 2}},
    )
    assert outputs["candidate_count"] == 6
    assert outputs["pilot_count"] == 4
    assert outputs["selection_path"].exists()
    assert outputs["manifest_path"].exists()


def test_generate_hpc_bundle_wires_validated_raw_coordinates_without_claiming_preparation(
    tmp_path: Path,
) -> None:
    _write_raw_receptor_coordinates(tmp_path)
    compounds = _candidates()
    compounds["standardized_smiles"] = ["CCN", "CCO", "CCC", "CCCl", "CCBr", "CCF"]
    physics_summary = compounds[["compound_id"]].copy()
    physics_summary["structure_id"] = [f"STR-{index}" for index in range(len(compounds))]
    physics_summary["ph"] = 7.4
    physics_summary["pka_scenario"] = "nominal"
    physics_summary["sa_3d_psa_ang2__mean"] = compounds["polar"]

    output = tmp_path / "bundle"
    generate_hpc_bundles(
        compounds,
        physics_summary,
        output,
        {
            "project_root": tmp_path,
            "pilot_selection": {"n_select": 4, "matched_pair_slots": 2},
        },
    )

    receptors = pd.read_csv(output / "herg_ensemble" / "receptors.csv")
    manifest = json.loads((output / "manifest.json").read_text())
    assert set(receptors["canonical_raw_coordinate_path"]) == set(RECEPTOR_RAW_COORDINATE_PATHS.values())
    assert receptors["raw_coordinate_validation_status"].eq("validated_entry_and_atom_site").all()
    assert receptors["raw_atom_count"].eq(1).all()
    assert receptors["source_structure_path"].isna().all()
    assert receptors["prepared_receptor_path"].isna().all()
    assert receptors["readiness_status"].eq("blocked_raw_coordinate_only_requires_preparation").all()
    assert manifest["validated_raw_receptor_coordinate_count"] == 6
    assert manifest["prepared_systems_present"] is False
