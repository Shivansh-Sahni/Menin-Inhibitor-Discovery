import json

import pytest

from menin_edit.objectives import (
    apply_constraint_summary,
    build_edit_effect,
    evaluate_constraint,
    evaluate_constraints,
    normalize_priorities,
    utility_delta,
)
from menin_edit.schemas import (
    CandidateNode,
    ConstraintSpec,
    EndpointSpec,
    MolecularEdit,
    ObjectiveSpec,
    PropertyEstimate,
    SearchSpec,
)


def estimate(
    endpoint: str,
    mean: float,
    lower: float,
    upper: float,
    *,
    inside: bool = True,
) -> PropertyEstimate:
    return PropertyEstimate(
        endpoint=endpoint,
        mean=mean,
        lower=lower,
        upper=upper,
        inside_domain=inside,
        model_version="test-v1",
        metadata={"nearest_neighbor": "CMP-1"},
    )


def root(node_id: str, predictions: dict[str, PropertyEstimate]) -> CandidateNode:
    return CandidateNode(
        node_id=node_id,
        smiles="CCN",
        parent_id=None,
        depth=0,
        predictions=predictions,
    )


def child(
    node_id: str,
    parent_id: str,
    predictions: dict[str, PropertyEstimate],
    *,
    effects=None,
) -> CandidateNode:
    return CandidateNode(
        node_id=node_id,
        smiles="CCO",
        parent_id=parent_id,
        depth=1,
        edit=MolecularEdit(
            rule_id="EDIT-1",
            transformation="[*:1]N>>[*:1]O",
            changed_atoms=(2,),
            edit_type="substitution",
            support_count=4,
            context_similarity=0.8,
        ),
        predictions=predictions,
        effects=effects or {},
    )


def test_endpoint_direction_makes_desirable_changes_positive():
    potency = EndpointSpec("menin_pIC50", "regression", "maximize")
    herg = EndpointSpec("herg_risk", "classification", "minimize")
    parent_potency = estimate("menin_pIC50", 7.5, 7.3, 7.7)
    product_potency = estimate("menin_pIC50", 8.0, 7.8, 8.2)
    parent_herg = estimate("herg_risk", 0.60, 0.50, 0.70)
    product_herg = estimate("herg_risk", 0.25, 0.15, 0.35)

    assert utility_delta(potency, parent_potency, product_potency) == pytest.approx(0.5)
    assert utility_delta(herg, parent_herg, product_herg) == pytest.approx(0.35)
    effect = build_edit_effect(herg, parent_herg, product_herg)
    assert effect.raw_delta == pytest.approx(-0.35)
    assert effect.utility_delta == pytest.approx(0.35)
    assert effect.lower == pytest.approx(-0.55)
    assert effect.upper == pytest.approx(-0.15)


def test_absolute_constraint_uses_least_favorable_interval_edge():
    candidate = root("C", {"herg_risk": estimate("herg_risk", 0.25, 0.10, 0.36)})
    constraint = ConstraintSpec("herg_risk", "<=", 0.30)

    evaluation = evaluate_constraint(constraint, candidate)

    assert not evaluation.passed
    assert evaluation.mean == pytest.approx(0.25)
    assert evaluation.conservative_value == pytest.approx(0.36)
    assert evaluation.margin == pytest.approx(-0.06)


def test_start_relative_constraint_uses_conservative_difference_interval():
    baseline = root("START", {"potency": estimate("potency", 8.0, 7.8, 8.2)})
    candidate = child(
        "NEXT",
        "START",
        {"potency": estimate("potency", 7.8, 7.7, 7.9)},
    )
    constraint = ConstraintSpec("potency", ">=", -0.30, relative_to="start")

    evaluation = evaluate_constraint(constraint, candidate, start=baseline)

    assert evaluation.mean == pytest.approx(-0.2)
    assert evaluation.conservative_value == pytest.approx(-0.5)
    assert not evaluation.passed


def test_previous_relative_constraint_prefers_paired_edit_interval():
    endpoint = EndpointSpec("potency", "regression", "maximize")
    previous_estimate = estimate("potency", 8.0, 7.8, 8.2)
    candidate_estimate = estimate("potency", 7.9, 7.7, 8.1)
    paired_effect = build_edit_effect(endpoint, previous_estimate, candidate_estimate)
    # A paired delta model can provide a narrower calibrated interval.
    paired_effect = paired_effect.__class__(
        endpoint="potency",
        raw_delta=-0.1,
        utility_delta=-0.1,
        lower=-0.2,
        upper=0.0,
        direct_delta=-0.1,
        absolute_model_delta=-0.1,
        disagreement=0.0,
    )
    previous = root("PREVIOUS", {"potency": previous_estimate})
    candidate = child(
        "NEXT",
        "PREVIOUS",
        {"potency": candidate_estimate},
        effects={"potency": paired_effect},
    )
    constraint = ConstraintSpec("potency", ">=", -0.25, relative_to="previous")

    evaluation = evaluate_constraint(constraint, candidate, previous=previous)

    assert evaluation.passed
    assert evaluation.conservative_value == pytest.approx(-0.2)


def test_missing_and_out_of_domain_policies_are_explicit():
    missing = root("MISSING", {})
    strict_missing = evaluate_constraint(ConstraintSpec("tox", "<=", 0.2), missing)
    warning_missing = evaluate_constraint(
        ConstraintSpec("tox", "<=", 0.2, missing_policy="warn"),
        missing,
    )
    assert not strict_missing.passed
    assert warning_missing.passed and warning_missing.status == "warning"

    outside = root("OOD", {"tox": estimate("tox", 0.1, 0.05, 0.15, inside=False)})
    strict_ood = evaluate_constraint(ConstraintSpec("tox", "<=", 0.2), outside)
    warning_ood = evaluate_constraint(
        ConstraintSpec("tox", "<=", 0.2, out_of_domain_policy="warn"),
        outside,
    )
    assert not strict_ood.passed
    assert warning_ood.passed and warning_ood.status == "warning"


def test_constraint_summary_and_final_only_scope():
    candidate = root("C", {"tox": estimate("tox", 0.1, 0.05, 0.15)})
    constraints = (
        ConstraintSpec("tox", "<=", 0.2),
        ConstraintSpec("potency", ">=", 8.0, apply_to="final"),
    )
    intermediate = evaluate_constraints(constraints, candidate, is_final=False)
    final = evaluate_constraints(constraints, candidate, is_final=True)
    assert intermediate.feasible
    assert len(intermediate.evaluations) == 1
    assert not final.feasible
    constrained = apply_constraint_summary(candidate, final)
    assert not constrained.feasible
    assert constrained.violations
    assert constrained.constraint_evaluations == final.evaluations
    assert constrained.warnings == final.warnings


def test_schemas_are_immutable_and_json_serializable():
    prediction = estimate("potency", 8.0, 7.7, 8.3)
    candidate = root("ROOT", {"potency": prediction})
    with pytest.raises(TypeError):
        candidate.predictions["other"] = prediction
    with pytest.raises(TypeError):
        prediction.metadata["new"] = "value"
    payload = json.loads(candidate.to_json())
    assert payload["predictions"]["potency"]["metadata"]["nearest_neighbor"] == "CMP-1"


def test_priorities_and_search_parameters_validate():
    weights = normalize_priorities([ObjectiveSpec("potency", 3.0), ObjectiveSpec("herg", 1.0)])
    assert weights == {"potency": pytest.approx(0.75), "herg": pytest.approx(0.25)}
    search = SearchSpec(
        min_parent_similarity=0.5,
        uncertainty_penalty=0.2,
        path_complexity_penalty=0.04,
    )
    assert search.min_parent_similarity == 0.5
    with pytest.raises(ValueError):
        SearchSpec(min_parent_similarity=1.1)
    with pytest.raises(ValueError):
        SearchSpec(uncertainty_penalty=-0.1)
