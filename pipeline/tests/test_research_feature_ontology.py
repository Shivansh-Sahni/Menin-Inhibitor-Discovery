from __future__ import annotations

from menin_discovery.research_feature_ontology import (
    CONVENTIONAL_DESCRIPTOR_COLUMNS,
    MODEL_CONFORMER_FEATURES,
    MODEL_PHYSICS_FEATURE_BLOCKS,
    MODEL_PHYSICS_FEATURES,
    classify_physics_feature,
    conventional_feature_ontology_frame,
    feature_ontology_frame,
    selected_model_conformer_features,
    selected_model_physics_features,
)
from menin_discovery.research_physics import _DISTRIBUTION_FEATURES


def test_every_generated_distribution_family_has_one_causal_concept() -> None:
    for primitive in _DISTRIBUTION_FEATURES:
        for aggregation in ("mean", "sd", "q05", "q50", "q95"):
            concept = classify_physics_feature(f"{primitive}__{aggregation}")
            assert concept is not None, primitive
            assert concept.physical_phenomenon
            assert concept.biological_event
            assert concept.hidden_variables_and_confounders
            assert concept.permissible_model_roles
            assert concept.falsification_test


def test_model_selection_is_small_predeclared_and_fail_closed() -> None:
    columns = ["unreviewed_numeric_feature", *reversed(MODEL_PHYSICS_FEATURES)]
    selected = selected_model_physics_features(columns)

    assert selected == list(MODEL_PHYSICS_FEATURES)
    # The count is deliberately not a scientific constant.  It can shrink
    # after an evidence/causal audit without weakening fail-closed selection.
    assert 0 < len(selected) <= 8
    assert len(selected) == len(set(selected))
    assert "unreviewed_numeric_feature" not in selected
    assert classify_physics_feature("unreviewed_numeric_feature") is None
    assert all(
        classify_physics_feature(feature).status == "provisional_discovery_proxy"  # type: ignore[union-attr]
        for feature in selected
    )

    assert "absolute_formal_charge__mean" not in selected
    assert "joint_conformational_entropy_normalized" not in selected
    assert "npr1__mean" not in selected
    assert "npr2__mean" not in selected


def test_conformer_mil_selection_excludes_algorithmic_and_redundant_columns() -> None:
    columns = [
        "cluster_id",
        "energy_kcal_mol",
        "total_sasa_ang2",
        "sa_3d_psa_ang2",
        "unreviewed_numeric_feature",
        *reversed(MODEL_CONFORMER_FEATURES),
    ]

    assert selected_model_conformer_features(columns) == list(MODEL_CONFORMER_FEATURES)


def test_physics_ablation_blocks_partition_selected_features_once() -> None:
    flattened = [feature for block in MODEL_PHYSICS_FEATURE_BLOCKS.values() for feature in block]

    assert set(flattened) == set(MODEL_PHYSICS_FEATURES)
    assert len(flattened) == len(set(flattened))


def test_ontology_declares_redundancy_and_nonpredictive_roles() -> None:
    ontology = feature_ontology_frame()
    assert ontology["feature_family"].is_unique
    assert ontology["physical_phenomenon"].str.len().gt(0).all()
    assert ontology["causal_location"].str.len().gt(0).all()
    assert ontology["redundancy_group"].str.len().gt(0).all()
    assert ontology["falsification_test"].str.len().gt(0).all()

    alias = classify_physics_feature("sa_3d_psa_ang2__mean")
    sensitivity = classify_physics_feature(
        "composite_pka_sensitivity_span__rare_state_transport_dominance_surrogate"
    )
    assert alias is not None and alias.status == "remove_duplicate"
    assert sensitivity is not None and sensitivity.status == "uncertainty_only"
    assert sensitivity.status != "core_discovery_predictor"


def test_conventional_inputs_are_controls_not_fundamental_parameters() -> None:
    ontology = conventional_feature_ontology_frame().set_index("feature")

    assert set(CONVENTIONAL_DESCRIPTOR_COLUMNS) == set(
        ontology.index[ontology["selected_internal_descriptor"]]
    )
    assert ontology.loc["exact_mol_wt", "production_action"] == ("remove_from_internal_model_matrix")
    assert ontology.loc["heavy_atom_count", "production_action"] == ("remove_from_internal_model_matrix")
    assert ontology.loc["formal_charge", "mechanistic_status"] == "not_an_assay_state"
    assert ontology.loc["invalid_structure", "production_action"] == "exclude_from_all_models"
    assert not ontology["mechanistic_status"].str.contains("fundamental").any()
