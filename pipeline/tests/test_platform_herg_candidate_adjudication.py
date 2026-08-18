from __future__ import annotations

from menin_discovery.platform_herg_candidate_adjudication import (
    _build_lineage_evidence,
    _relation_unit_audit,
    _target_evidence,
    _value_clusters,
)


def _observation(observation_id: str, value: float, **updates: object) -> dict[str, object]:
    row: dict[str, object] = {
        "observation_id": observation_id,
        "structure_id": "S1",
        "source_family": "chembl_herg_specialized_view",
        "source_record_id": f"R-{observation_id}",
        "assay_id": "A1",
        "native_endpoint": "IC50",
        "native_relation": "=",
        "native_value": 10.0,
        "native_unit": "nM",
        "potency_relation_pic50": "=",
        "potency_pic50_point": value,
        "potency_pic50_lower_bound": value,
        "potency_pic50_upper_bound": value,
        "potency_censoring": "exact",
        "model_split": "test",
    }
    row.update(updates)
    return row


def _source(**updates: object) -> dict[str, object]:
    row: dict[str, object] = {
        "document_id": "D1",
        "reported_source": "LITERATURE",
        "assay_description": "Inhibition of human hERG",
        "target_chembl_id": "CHEMBL240",
        "component_accessions": "Q12809",
        "target_type": "SINGLE PROTEIN",
        "target_organism": "Homo sapiens",
        "target_relationship_type": "D",
        "target_variant_id": None,
    }
    row.update(updates)
    return row


def test_relation_unit_audit_preserves_relation_inversion_and_numeric_value() -> None:
    exact = _observation("O1", 8.0)
    exact_audit = _relation_unit_audit(exact)
    assert exact_audit["status"] == "consistent_exact_relation_and_unit_conversion"
    assert exact_audit["expected_pic50"] == 8.0

    censored = _observation(
        "O2",
        8.0,
        native_relation=">",
        potency_relation_pic50="<",
        potency_pic50_point=None,
        potency_pic50_lower_bound=None,
        potency_pic50_upper_bound=8.0,
        potency_censoring="pic50_upper_bounded",
    )
    assert _relation_unit_audit(censored)["status"] == "consistent_censored_relation_inversion_and_bound"

    inconsistent = dict(censored, potency_relation_pic50=">")
    assert _relation_unit_audit(inconsistent)["status"] == "inconsistent_relation_direction"


def test_target_metadata_never_upgrades_null_variant_to_confirmed_wild_type() -> None:
    row = {"source_family": "chembl_herg_specialized_view"}
    direct, explicit_wt, _, limitations = _target_evidence(row, _source())
    assert direct == "direct_human_kcnh2_single_protein_no_variant_annotation"
    assert explicit_wt is False
    assert "absence_of_variant_annotation_is_not_explicit_wild_type_confirmation" in limitations

    homologue, _, _, _ = _target_evidence(row, _source(target_relationship_type="H"))
    assert homologue == "homologue_relationship_to_human_kcnh2_no_variant_annotation"

    compilation, _, _, _ = _target_evidence({"source_family": "quantitative_pic50_release"}, _source())
    assert compilation == "compilation_target_assertion_without_assay_level_target_status"


def test_value_clustering_uses_bounded_pic50_tolerance() -> None:
    clusters = _value_clusters(
        [
            _observation("O1", 7.0),
            _observation("O2", 7.0 + 5e-7),
            _observation("O3", 7.0 + 2e-6),
        ]
    )
    assert [[row["observation_id"] for row in cluster] for cluster in clusters] == [
        ["O1", "O2"],
        ["O3"],
    ]


def test_cross_source_exact_value_is_mirror_candidate_not_adjudicated_duplicate() -> None:
    chembl = _observation("O1", 7.0)
    compilation = _observation(
        "O2",
        7.0 + 5e-8,
        source_family="quantitative_pic50_release",
        source_record_id="CHEMBL1",
        assay_id=None,
        native_endpoint="pIC50",
        native_value=7.0 + 5e-8,
        native_unit="pIC50",
    )
    evidence = {
        "O1": _source(document_id="D1", reported_source="LITERATURE"),
        "O2": _source(document_id=None, reported_source="ChEMBL"),
    }
    groups, by_observation, clusters = _build_lineage_evidence([chembl, compilation], evidence, {"O1", "O2"})
    equal_value = [row for row in groups if row["lineage_group_kind"] == "standardized_equal_value_cluster"]
    assert len(equal_value) == 1
    assert (
        equal_value[0]["automated_lineage_class"]
        == "cross_source_exact_value_mirror_candidate_reported_chembl"
    )
    assert equal_value[0]["human_adjudication_status"] == "pending_human_adjudication"
    assert len(by_observation["O1"]) == 1
    assert clusters["O2"]["size"] == 2
