from __future__ import annotations

import json

import menin_discovery.research_physics as research_physics
import numpy as np
import pandas as pd
import pytest
from menin_discovery.chemistry import rdkit_available
from menin_discovery.research_physics import (
    DEFERRED_VALIDATION_CONFORMER_CEILING,
    LOCAL_GENERATED_CONFORMER_DEPTH,
    PILOT_COMPARATOR_CONFORMER_DEPTH,
    FastPhysicsConfig,
    PKaEvidence,
    _conformer_sampling_escalation_queue,
    _normalized_perturbation_weights,
    _physics_admissibility,
    _state_threshold_audit,
    run_fast_physics,
    run_structure_fast_physics,
    write_fast_physics_outputs,
)

pytestmark = pytest.mark.skipif(not rdkit_available(), reason="fast physics requires RDKit")


def _smoke_config(**overrides):
    values = {
        "smoke_mode": True,
        "smoke_conformers_per_state": 3,
        "smoke_retained_conformers": 3,
        "max_tautomers": 3,
        "max_states": 6,
    }
    values.update(overrides)
    return FastPhysicsConfig(**values)


def _dataframe_config(cache_dir, *, workers=1):
    return {
        "paths": {"physics_cache": cache_dir},
        "fast_physics": {
            "workers": workers,
            "max_tautomers": 1,
            "max_states": 3,
            "smoke_conformers_per_state": 2,
            "smoke_retained_conformers": 2,
        },
    }


def _synthetic_sampling_tables(*, generated: int, retained: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    states = pd.DataFrame({"state_id": ["STATE-1"], "n_embedded": [generated]})
    conformers = pd.DataFrame(
        {
            "structure_id": ["STR-1"] * retained,
            "state_id": ["STATE-1"] * retained,
            "conformer_rank": np.arange(1, retained + 1),
            "conformer_weight": np.full(retained, 1.0 / retained),
            "sa_3d_psa_ang2": np.full(retained, 100.0),
            "radius_of_gyration_angstrom": np.full(retained, 8.0),
            "imhb_count_proxy": np.full(retained, 2.0),
            "energy_kcal_mol": np.full(retained, -10.0),
        }
    )
    return states, conformers


def test_future_local_defaults_generate_and_retain_25() -> None:
    config = FastPhysicsConfig()
    assert config.max_conformers_per_state == LOCAL_GENERATED_CONFORMER_DEPTH == 25
    assert config.max_retained_conformers == LOCAL_GENERATED_CONFORMER_DEPTH


def test_sampling_queue_marks_incomplete_local_pool_for_retry() -> None:
    states, conformers = _synthetic_sampling_tables(generated=24, retained=24)
    queue = _conformer_sampling_escalation_queue(states, conformers, FastPhysicsConfig())

    assert queue.loc[0, "queue_status"] == "incomplete_local_screen_retry_25"
    assert queue.loc[0, "recommended_next_generated_depth"] == 25
    assert bool(queue.loc[0, "escalation_required"])


def test_sampling_queue_abstains_when_retained_depth_does_not_cover_generated_pool() -> None:
    states, conformers = _synthetic_sampling_tables(generated=50, retained=25)
    config = FastPhysicsConfig(max_conformers_per_state=50, max_retained_conformers=25)
    queue = _conformer_sampling_escalation_queue(states, conformers, config)

    assert queue.loc[0, "queue_status"] == "abstain_retained_depth_does_not_cover_generated_pool"
    assert not bool(queue.loc[0, "audit_covers_generated_depth"])
    assert bool(queue.loc[0, "sampling_audit_abstained"])
    assert not bool(queue.loc[0, "nested_internal_stability_passed"])
    assert queue.loc[0, "recommended_next_generated_depth"] == 50


def test_state_and_conformer_generation_is_deterministic_and_state_aware():
    evidence = (
        PKaEvidence("amine analog", 9.5, "base", "measured close analog", atom_index=2),
        PKaEvidence("carboxyl", 2.3, "acid", "measured close analog", atom_smarts="C(=O)[OH]"),
    )
    config = _smoke_config(herg_ph=7.2)
    first = run_structure_fast_physics("C[C@H](N)C(=O)O", pka_evidence=evidence, config=config)
    second = run_structure_fast_physics("C[C@H](N)C(=O)O", pka_evidence=evidence, config=config)

    assert first.structure_id == second.structure_id
    assert first.states["state_id"].tolist() == second.states["state_id"].tolist()
    assert first.conformers["conformer_id"].tolist() == second.conformers["conformer_id"].tolist()
    numeric = ["energy_kcal_mol", "sa_3d_psa_ang2", "radius_of_gyration_angstrom"]
    np.testing.assert_allclose(first.conformers[numeric], second.conformers[numeric], atol=1e-10)

    reference_smiles = first.states.loc[
        first.states["transformation"] == "reference_tautomer", "state_smiles"
    ].tolist()
    assert any("@" in smiles for smiles in reference_smiles)
    assert {"pka_minus_1", "nominal", "pka_plus_1"} == set(first.populations["pka_scenario"])
    assert set(config.all_ph_values) == set(first.populations["ph"])
    assert np.allclose(
        first.populations.groupby(["ph", "pka_scenario"])["state_weight"].sum(),
        1.0,
    )
    assert "not exact microscopic pKas" in first.metadata["pka_interpretation"]
    assert set(first.composites["evidence_class"]) == {"reproduced", "extended", "candidate"}
    assert {
        "sa_3d_psa_ang2",
        "polar_sasa_ang2",
        "nonpolar_sasa_ang2",
        "exposed_hbd_count_proxy",
        "exposed_hba_count_proxy",
        "gasteiger_dipole_proxy_debye",
        "imhb_network_proxy",
    } <= set(first.conformers.columns)


def test_descriptor_perturbation_weights_are_deterministic_finite_and_normalized():
    base = np.array([0.6, 0.3, 0.1])
    coordinate = np.array([-1.5, 0.25, 2.0])
    first = _normalized_perturbation_weights(base, coordinate)
    second = _normalized_perturbation_weights(base, coordinate)

    np.testing.assert_array_equal(first, second)
    assert np.isfinite(first).all()
    assert first.sum() == pytest.approx(1.0, abs=1e-15)
    assert (first >= 0.0).all()


def test_state_threshold_audit_reports_all_declared_thresholds_and_omitted_mass():
    populations = pd.DataFrame(
        {
            "state_id": ["S1", "S2", "S3"],
            "ph": [7.4, 7.4, 7.4],
            "pka_scenario": ["nominal", "nominal", "nominal"],
            "raw_probability_pre_retention": [0.989, 0.0105, 0.0005],
        }
    )
    audit = pd.DataFrame(_state_threshold_audit(populations, retained_ids={"S1", "S3"}))

    assert set(audit["threshold_percent"]) == {1.0, 0.1, 0.01}
    one_percent = audit[audit["threshold_percent"] == 1.0].iloc[0]
    assert one_percent["states_at_or_above_threshold"] == 2
    assert one_percent["probability_mass_below_threshold"] == pytest.approx(0.0005)
    assert one_percent["qualifying_states_omitted_by_cap"] == 1
    assert one_percent["qualifying_probability_mass_omitted_by_cap"] == pytest.approx(0.0105)


def test_state_compute_cap_is_a_hard_admissibility_failure_not_a_target():
    result = run_structure_fast_physics(
        "NCC(=O)O",
        config=_smoke_config(max_tautomers=3, max_states=1),
    )
    retention = result.metadata["state_retention"]
    cap_gate = result.quality[result.quality["gate"] == "no_qualifying_state_omitted_by_compute_cap"]

    assert retention["state_cap_applied"] is True
    assert retention["qualifying_states_omitted_by_cap"] > 0
    assert "not a physical target" in retention["retention_interpretation"]
    assert cap_gate["passed"].tolist() == [False]
    assert result.metadata["physics_admissibility"]["physics_model_eligible"] is False


def test_composite_surrogates_are_pivoted_and_preserve_pka_scenarios():
    result = run_structure_fast_physics(
        "CCN",
        pka_evidence=[PKaEvidence("amine", 7.4, "base", "sensitivity test", atom_index=2)],
        config=_smoke_config(max_tautomers=1, max_states=3),
    )
    expected = {
        "environment_conditioned_polarity_response_surrogate",
        "environment_conditioned_shape_response_surrogate",
        "water_to_low_dielectric_folded_fraction_shift_surrogate",
        "hydration_shedding_imhb_compensation_surrogate",
        "rare_state_transport_dominance_surrogate",
    }
    assert expected <= set(result.composites["composite_name"])
    assert np.isfinite(
        result.composites[
            ["value", "pka_sensitivity_min", "pka_sensitivity_max", "pka_sensitivity_span"]
        ].to_numpy(dtype=float)
    ).all()
    assert result.composites["uncertainty_semantics"].str.contains("not confidence intervals").all()
    assert result.composites["environment_model"].str.contains("no_explicit_solvent").all()

    target_ph = result.ensemble_summary[np.isclose(result.ensemble_summary["ph"], 7.4)]
    assert set(target_ph["pka_scenario"]) == {"pka_minus_1", "nominal", "pka_plus_1"}
    assert target_ph["formal_charge__mean"].nunique() > 1
    for composite_name in expected:
        summary_column = f"composite__{composite_name}"
        span_column = f"composite_pka_sensitivity_span__{composite_name}"
        assert summary_column in result.ensemble_summary.columns
        assert span_column in result.ensemble_summary.columns
        source = result.composites[result.composites["composite_name"] == composite_name]
        merged = target_ph[["ph", "pka_scenario", summary_column, span_column]].merge(
            source[["ph", "pka_scenario", "value", "pka_sensitivity_span"]],
            on=["ph", "pka_scenario"],
            validate="one_to_one",
        )
        np.testing.assert_allclose(merged[summary_column], merged["value"])
        np.testing.assert_allclose(merged[span_column], merged["pka_sensitivity_span"])


def test_smoke_admissibility_separates_sampling_from_substantive_failure():
    quality = pd.DataFrame(
        {
            "gate": [
                "effective_conformer_count_ge_1_5_or_single",
                "discarded_probability_mass_le_0.05",
            ],
            "passed": [False, False],
        }
    )
    summary = pd.DataFrame({"structure_id": ["STR-X"], "ph": [7.4], "value": [1.0]})
    composites = pd.DataFrame({"value": [0.2], "pka_sensitivity_span": [0.1]})
    states = pd.DataFrame({"formal_charge": [0]})
    populations = pd.DataFrame({"ph": [7.4], "pka_scenario": ["nominal"], "state_weight": [1.0]})
    conformers = pd.DataFrame({"state_id": ["S1"], "conformer_weight": [1.0]})
    admissibility = _physics_admissibility(
        structure_id="STR-X",
        quality=quality,
        summary=summary,
        composites=composites,
        states=states,
        populations=populations,
        conformers=conformers,
        smoke_mode=True,
    )

    assert admissibility["physics_model_eligible"] is False
    assert admissibility["physics_convergence_claimed"] is False
    assert admissibility["physics_substantive_failure_count"] == 1
    assert admissibility["physics_smoke_sampling_failure_count"] == 1
    assert "substantive:discarded_probability_mass" in admissibility["physics_quality_reason_flags"]
    assert "smoke_sampling:effective_conformer" in admissibility["physics_quality_reason_flags"]


def test_fast_physics_writes_parquet_sdf_and_metadata(tmp_path):
    result = run_structure_fast_physics("CCN", config=_smoke_config(max_tautomers=1, max_states=3))
    paths = write_fast_physics_outputs(result, tmp_path)

    assert all(path.exists() for path in paths.values())
    assert len(pd.read_parquet(paths["conformers"])) == len(result.conformers)
    metadata = json.loads(paths["metadata"].read_text())
    assert metadata["smoke_mode"] is True
    assert metadata["fast_physics_version"]

    from rdkit import Chem

    records = [mol for mol in Chem.SDMolSupplier(str(paths["sdf"]), removeHs=False) if mol is not None]
    assert len(records) == len(result.conformers)
    assert all(mol.HasProp("state_id") and mol.HasProp("conformer_id") for mol in records)


def test_dataframe_api_computes_duplicate_structure_once_and_maps_to_compounds(tmp_path):
    compounds = pd.DataFrame(
        {
            "compound_id": ["CMP-1", "CMP-2"],
            "standardized_smiles": ["CCN", "CCN"],
            "mw": [45.08, 45.08],
        }
    )
    chemical_states = pd.DataFrame(
        {
            "compound_id": ["CMP-1"],
            "pka_kind": ["base"],
            "pka_value": [9.8],
            "pka_label": ["amine"],
            "pka_source": ["experimental analog"],
            "atom_index": [2],
        }
    )
    outputs = run_fast_physics(
        compounds,
        chemical_states,
        tmp_path,
        {
            "max_tautomers": 1,
            "max_states": 3,
            "smoke_conformers_per_state": 2,
            "smoke_retained_conformers": 2,
        },
        smoke=True,
    )

    assert outputs["compound_count"] == 2
    assert outputs["unique_structure_count"] == 1
    assert outputs["successful_structure_count"] == 1
    assert outputs["sampling_audited_state_count"] >= 1
    summary = pd.read_parquet(outputs["summary_path"])
    assert set(summary["compound_id"]) == {"CMP-1", "CMP-2"}
    assert pd.read_parquet(outputs["states_path"])["structure_id"].nunique() == 1
    assert (outputs["ensemble_sdf_dir"] / summary["structure_id"].iloc[0] / "conformers.sdf").exists()
    threshold_audit = pd.read_parquet(outputs["state_threshold_audit_path"])
    assert set(threshold_audit["threshold_percent"]) == {1.0, 0.1, 0.01}
    queue = pd.read_parquet(outputs["sampling_escalation_queue_path"])
    assert queue["recommended_next_generated_depth"].eq(LOCAL_GENERATED_CONFORMER_DEPTH).all()
    assert queue["deferred_validation_ceiling"].eq(DEFERRED_VALIDATION_CONFORMER_CEILING).all()
    assert queue["pilot_comparator_depth"].eq(PILOT_COMPARATOR_CONFORMER_DEPTH).all()
    assert queue["interpretation"].str.contains("validated universal constant").all()


def test_valid_structure_cache_resumes_without_recalculation(tmp_path, monkeypatch):
    compounds = pd.DataFrame({"compound_id": ["CMP-1"], "standardized_smiles": ["CCN"], "mw": [45.08]})
    config = _dataframe_config(tmp_path / "persistent_cache")
    first = run_fast_physics(compounds, None, tmp_path / "first", config, smoke=True)
    assert first["computed_structure_count"] == 1
    assert first["cache_hit_structure_count"] == 0

    def unexpected_recalculation(*args, **kwargs):
        raise AssertionError("a complete cache entry must be reused")

    monkeypatch.setattr(research_physics, "run_structure_fast_physics", unexpected_recalculation)
    second = run_fast_physics(compounds, None, tmp_path / "second", config, smoke=True)
    assert second["computed_structure_count"] == 0
    assert second["cache_hit_structure_count"] == 1
    pd.testing.assert_frame_equal(
        pd.read_parquet(first["summary_path"]),
        pd.read_parquet(second["summary_path"]),
    )


def test_cached_primitives_are_reaggregated_without_changing_cache_identity(tmp_path, monkeypatch):
    compounds = pd.DataFrame({"compound_id": ["CMP-1"], "standardized_smiles": ["CCN"], "mw": [45.08]})
    cache_dir = tmp_path / "persistent_cache"
    config = _dataframe_config(cache_dir)
    first = run_fast_physics(compounds, None, tmp_path / "first", config, smoke=True)
    cache_path = next(cache_dir.glob("*/*/metadata.json")).parent
    metadata_path = cache_path / "metadata.json"
    metadata = json.loads(metadata_path.read_text())

    # Emulate a scientifically valid cache created before the extended
    # composites existed.  Raw states/populations/conformers remain untouched.
    cached_summary = pd.read_parquet(cache_path / "ensemble_summary.parquet")
    cached_summary = cached_summary[
        [column for column in cached_summary if not column.startswith("composite")]
    ]
    cached_composites = pd.read_parquet(cache_path / "composites.parquet")
    cached_composites = cached_composites[
        cached_composites["composite_name"].isin(
            {
                "folded_low_polarity_fraction",
                "exposure_adjusted_hbond_burden",
                "intramolecular_shielding_candidate",
                "charge_separation_per_gyration_candidate",
            }
        )
    ].reset_index(drop=True)
    cached_summary.to_parquet(cache_path / "ensemble_summary.parquet", index=False)
    cached_composites.to_parquet(cache_path / "composites.parquet", index=False)
    metadata["output_row_counts"]["composites"] = len(cached_composites)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    def unexpected_recalculation(*args, **kwargs):
        raise AssertionError("compatible raw cache primitives must be reused")

    monkeypatch.setattr(research_physics, "run_structure_fast_physics", unexpected_recalculation)
    second = run_fast_physics(compounds, None, tmp_path / "second", config, smoke=True)

    assert first["cache_dir"] == second["cache_dir"]
    assert second["cache_hit_structure_count"] == 1
    refreshed = pd.read_parquet(second["summary_path"])
    assert "composite__rare_state_transport_dominance_surrogate" in refreshed.columns
    assert (tmp_path / "second" / "fast_physics_admissibility.parquet").exists()


def test_incomplete_cache_is_recomputed_and_repaired(tmp_path, monkeypatch):
    compounds = pd.DataFrame({"compound_id": ["CMP-1"], "standardized_smiles": ["CCN"], "mw": [45.08]})
    cache_dir = tmp_path / "persistent_cache"
    config = _dataframe_config(cache_dir)
    run_fast_physics(compounds, None, tmp_path / "first", config, smoke=True)
    cached_sdf = next(cache_dir.glob("*/*/conformers.sdf"))
    cached_sdf.unlink()

    calls = 0
    original = research_physics.run_structure_fast_physics

    def counted_recalculation(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(research_physics, "run_structure_fast_physics", counted_recalculation)
    repaired = run_fast_physics(compounds, None, tmp_path / "second", config, smoke=True)
    assert calls == 1
    assert repaired["computed_structure_count"] == 1
    assert repaired["cache_hit_structure_count"] == 0
    assert cached_sdf.exists()


def test_cache_identity_includes_pka_evidence(tmp_path):
    compounds = pd.DataFrame({"compound_id": ["CMP-1"], "standardized_smiles": ["CCN"], "mw": [45.08]})
    cache_dir = tmp_path / "persistent_cache"
    config = _dataframe_config(cache_dir)

    def evidence(pka_value):
        return pd.DataFrame(
            {
                "compound_id": ["CMP-1"],
                "pka_kind": ["base"],
                "pka_value": [pka_value],
                "pka_label": ["amine"],
                "pka_source": ["experimental analog"],
                "atom_index": [2],
            }
        )

    first = run_fast_physics(
        compounds,
        evidence(9.8),
        tmp_path / "evidence_first",
        config,
        smoke=True,
    )
    second = run_fast_physics(
        compounds,
        evidence(8.8),
        tmp_path / "evidence_second",
        config,
        smoke=True,
    )
    assert first["computed_structure_count"] == 1
    assert second["computed_structure_count"] == 1
    assert len(list(cache_dir.glob("*/*/metadata.json"))) == 2


def test_cache_survives_interruption_after_structure_completion(tmp_path, monkeypatch):
    compounds = pd.DataFrame({"compound_id": ["CMP-1"], "standardized_smiles": ["CCN"], "mw": [45.08]})
    cache_dir = tmp_path / "persistent_cache"
    config = _dataframe_config(cache_dir)

    with monkeypatch.context() as context:
        context.setattr(
            research_physics,
            "_materialize_cached_structure",
            lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("simulated interruption")),
        )
        with pytest.raises(RuntimeError, match="simulated interruption"):
            run_fast_physics(compounds, None, tmp_path / "interrupted", config, smoke=True)

    assert len(list(cache_dir.glob("*/*/metadata.json"))) == 1
    resumed = run_fast_physics(compounds, None, tmp_path / "resumed", config, smoke=True)
    assert resumed["cache_hit_structure_count"] == 1
    assert resumed["computed_structure_count"] == 0


def test_parallel_structure_workers_preserve_deterministic_outputs(tmp_path):
    compounds = pd.DataFrame(
        {
            "compound_id": ["CMP-1", "CMP-2"],
            "standardized_smiles": ["CCN", "CCO"],
            "mw": [45.08, 46.07],
        }
    )
    serial = run_fast_physics(
        compounds,
        None,
        tmp_path / "serial",
        _dataframe_config(tmp_path / "serial_cache", workers=1),
        smoke=True,
    )
    parallel = run_fast_physics(
        compounds,
        None,
        tmp_path / "parallel",
        _dataframe_config(tmp_path / "parallel_cache", workers=2),
        smoke=True,
    )
    assert serial["worker_count"] == 1
    assert parallel["requested_worker_count"] == 2
    assert parallel["worker_count"] in {1, 2}
    for name in (
        "summary_path",
        "admissibility_path",
        "states_path",
        "populations_path",
        "conformers_path",
        "composites_path",
        "quality_path",
        "registry_path",
    ):
        pd.testing.assert_frame_equal(
            pd.read_parquet(serial[name]),
            pd.read_parquet(parallel[name]),
        )
