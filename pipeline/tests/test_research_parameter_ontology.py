from __future__ import annotations

from menin_discovery.research_parameter_ontology import (
    parameter_ontology_frame,
    validate_parameter_ontology,
)


def test_parameter_ontology_separates_fundamental_and_derived_quantities() -> None:
    ontology = parameter_ontology_frame()
    validate_parameter_ontology(ontology)

    assert ontology["parameter_id"].is_unique
    derived = ontology["parameter_class"].str.startswith("derived")
    assert ontology.loc[derived, "parent_parameter_ids"].str.len().gt(0).all()
    assert not ontology["permissible_role_now"].str.contains("decision", case=False).any()


def test_effective_permeability_and_herg_occupancy_declare_their_parents() -> None:
    ontology = parameter_ontology_frame().set_index("parameter_id")

    permeability_parents = set(ontology.loc["pk_effective_permeability", "parent_parameter_ids"].split(";"))
    occupancy_parents = set(
        ontology.loc["herg_protocol_integrated_occupancy", "parent_parameter_ids"].split(";")
    )

    assert {
        "pk_microstate_free_energy",
        "pk_membrane_pmf",
        "pk_membrane_diffusivity",
    } <= permeability_parents
    assert {
        "herg_drug_free_gating_generator",
        "herg_association_rate",
        "herg_dissociation_rate",
        "herg_trapping_escape_rate",
    } <= occupancy_parents
