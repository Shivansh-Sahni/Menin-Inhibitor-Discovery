from __future__ import annotations

from pathlib import Path

import pandas as pd
from menin_discovery.research_contracts import validate_contract_frame
from menin_discovery.research_feature_ontology import MODEL_PHYSICS_FEATURES
from menin_discovery.research_physics_contracts import project_fast_physics_contracts


def _write_detailed_fast_tables(root: Path) -> None:
    registry = pd.DataFrame(
        [
            {"compound_id": "CMP-A", "structure_id": "STR-1"},
            {"compound_id": "CMP-B", "structure_id": "STR-1"},
        ]
    )
    states = pd.DataFrame(
        [
            {
                "structure_id": "STR-1",
                "state_id": "MST-1",
                "state_smiles": "C[NH3+]",
                "transformation": "protonated_base",
                "formal_charge": 1,
                "pka_basis": "reported evidence",
                "pka_source": "test",
            },
            {
                "structure_id": "STR-1",
                "state_id": "MST-2",
                "state_smiles": "CN",
                "transformation": "reference_tautomer",
                "formal_charge": 0,
                "pka_basis": "reported evidence",
                "pka_source": "test",
            },
        ]
    )
    population_rows = []
    for scenario, first_weight in (("pka_minus_1", 0.4), ("nominal", 0.6), ("pka_plus_1", 0.8)):
        population_rows.extend(
            [
                {
                    "structure_id": "STR-1",
                    "state_id": "MST-1",
                    "ph": 7.4,
                    "pka_scenario": scenario,
                    "state_weight": first_weight,
                },
                {
                    "structure_id": "STR-1",
                    "state_id": "MST-2",
                    "ph": 7.4,
                    "pka_scenario": scenario,
                    "state_weight": 1.0 - first_weight,
                },
            ]
        )
    conformers = pd.DataFrame(
        [
            {
                "structure_id": "STR-1",
                "state_id": state_id,
                "conformer_id": f"CNF-{state_id[-1]}",
                "conformer_rank": 1,
                "relative_energy_kcal_mol": 0.0,
                "conformer_weight": 1.0,
                "minimization_method": "MMFF94s",
                "cluster_id": 0,
            }
            for state_id in ("MST-1", "MST-2")
        ]
    )
    summary = pd.DataFrame(
        [
            {
                "compound_id": compound_id,
                "structure_id": "STR-1",
                "mw": 700.0,
                "ph": 7.4,
                "pka_scenario": scenario,
                "joint_weight_sum": 1.0,
                "sa_3d_psa_ang2__mean": mean,
                "sa_3d_psa_ang2__sd": 2.0,
                "polar_sasa_ang2__mean": mean,
                "polar_sasa_ang2__q05": mean - 10.0,
            }
            for compound_id in ("CMP-A", "CMP-B")
            for scenario, mean in (("pka_minus_1", 95.0), ("nominal", 100.0), ("pka_plus_1", 105.0))
        ]
    )
    composites = pd.DataFrame(
        [
            {
                "structure_id": "STR-1",
                "ph": 7.4,
                "pka_scenario": scenario,
                "composite_name": "folded_low_polarity_fraction",
                "evidence_class": "reproduced",
                "value": value,
                "definition": "joint weight below declared shape and polarity thresholds",
            }
            for scenario, value in (("pka_minus_1", 0.2), ("nominal", 0.3), ("pka_plus_1", 0.4))
        ]
    )
    for name, frame in {
        "fast_physics_structure_registry": registry,
        "fast_physics_states": states,
        "fast_physics_state_populations": pd.DataFrame(population_rows),
        "fast_physics_conformers": conformers,
        "fast_physics_summary": summary,
        "fast_physics_composites": composites,
    }.items():
        frame.to_parquet(root / f"{name}.parquet", index=False)


def test_fast_physics_contract_projection_is_typed_and_preserves_sensitivity(tmp_path: Path) -> None:
    _write_detailed_fast_tables(tmp_path)
    result = project_fast_physics_contracts(tmp_path, random_seed=17)

    assert result["chemical_states_rows"] == 4
    assert result["conformers_rows"] == 4
    assert result["physics_runs_rows"] == 6
    assert result["feature_lineage_rows"] > 0

    for contract in (
        "chemical_states",
        "conformers",
        "physics_runs",
        "physics_observables",
        "feature_lineage",
    ):
        frame = pd.read_parquet(tmp_path / f"{contract}.parquet")
        assert not frame.empty
        validate_contract_frame(contract, frame)

    states = pd.read_parquet(tmp_path / "chemical_states.parquet")
    conformers = pd.read_parquet(tmp_path / "conformers.parquet")
    runs = pd.read_parquet(tmp_path / "physics_runs.parquet")
    observables = pd.read_parquet(tmp_path / "physics_observables.parquet")
    charged = states[states["smiles"] == "C[NH3+]"]
    assert set(charged["fractional_population"]) == {0.6}
    assert set(charged["uncertainty"].round(8)) == {0.2}
    assert set(conformers["chemical_state_id"]) <= set(states["chemical_state_id"])
    assert set(conformers["source_run_id"]) <= set(runs["physics_run_id"])
    assert set(observables["physics_run_id"]) <= set(runs["physics_run_id"])

    lineage = pd.read_parquet(tmp_path / "feature_lineage.parquet")
    nominal = lineage[lineage["feature_lineage_id"].str.contains(":nominal:")]
    sensitivity = lineage[~lineage["feature_lineage_id"].str.contains(":nominal:")]
    selected = nominal[nominal["feature_name"] == "polar_sasa_ang2__q05"]
    assert selected["model_eligible"].all()
    assert not nominal[nominal["feature_name"] != "polar_sasa_ang2__q05"]["model_eligible"].any()
    assert not sensitivity["model_eligible"].any()
    assert lineage["source_entity_ids"].map(len).min() == 1
    assert set(lineage["source_entity_ids"].explode()) <= set(observables["physics_observable_id"])
    ontology = pd.read_csv(tmp_path / "fast_physics_feature_ontology.csv")
    assert result["feature_ontology_rows"] == len(ontology)
    selected_count = ontology["selected_model_columns"].fillna("").str.len().gt(0).sum()
    # Selection is fail-closed and evidence-governed; its size is not a
    # scientific constant and may shrink after a causal audit.
    assert selected_count == len(MODEL_PHYSICS_FEATURES)
