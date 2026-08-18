import numpy as np

from menin_edit.pareto import pareto_ranks, rank_candidates, robust_objective_matrix
from menin_edit.schemas import CandidateNode, EndpointSpec, ObjectiveSpec, PropertyEstimate


def estimate(endpoint, mean, lower=None, upper=None, *, inside=True):
    return PropertyEstimate(
        endpoint=endpoint,
        mean=mean,
        lower=mean if lower is None else lower,
        upper=mean if upper is None else upper,
        inside_domain=inside,
        model_version="test-v1",
    )


def node(node_id, potency, herg, *, feasible=True, path_cost=0.0, inside=True):
    predictions = {}
    if potency is not None:
        predictions["potency"] = estimate("potency", potency, inside=inside)
    if herg is not None:
        predictions["herg"] = estimate("herg", herg, inside=inside)
    return CandidateNode(
        node_id=node_id,
        smiles=f"C{node_id}",
        parent_id=None,
        depth=0,
        predictions=predictions,
        feasible=feasible,
        path_cost=path_cost,
    )


SPECS = {
    "potency": EndpointSpec("potency", "regression", "maximize"),
    "herg": EndpointSpec("herg", "classification", "minimize"),
}


def test_non_dominated_sorting_assigns_complete_fronts():
    values = np.asarray(
        [
            [9.0, -0.45],
            [8.3, -0.10],
            [8.0, -0.50],
            [7.0, -0.60],
        ]
    )
    ranks = pareto_ranks(values)
    assert ranks.tolist() == [1.0, 1.0, 2.0, 3.0]


def test_priorities_reorder_only_within_the_same_pareto_front():
    candidates = [
        node("A", potency=9.0, herg=0.45),
        node("B", potency=8.3, herg=0.10),
        node("C", potency=8.0, herg=0.50),
    ]
    potency_first = rank_candidates(
        candidates,
        SPECS,
        [ObjectiveSpec("potency", 0.9), ObjectiveSpec("herg", 0.1)],
    )
    safety_first = rank_candidates(
        candidates,
        SPECS,
        [ObjectiveSpec("potency", 0.1), ObjectiveSpec("herg", 0.9)],
    )

    assert [row.node_id for row in potency_first] == ["A", "B", "C"]
    assert [row.node_id for row in safety_first] == ["B", "A", "C"]
    assert {row.node_id: row.pareto_rank for row in potency_first} == {
        "A": 1,
        "B": 1,
        "C": 2,
    }
    assert potency_first[-1].pareto_rank == 2
    assert safety_first[-1].pareto_rank == 2


def test_failed_hard_constraint_is_never_rescued_by_priority():
    candidates = [
        node("INFEASIBLE", potency=10.0, herg=0.01, feasible=False),
        node("FEASIBLE", potency=7.0, herg=0.40, feasible=True),
    ]
    ranked = rank_candidates(
        candidates,
        SPECS,
        [ObjectiveSpec("potency", 1.0), ObjectiveSpec("herg", 0.0)],
    )
    assert [row.node_id for row in ranked] == ["FEASIBLE", "INFEASIBLE"]
    assert ranked[0].eligible
    assert not ranked[1].eligible
    assert ranked[1].pareto_rank is None


def test_robust_ranking_uses_lower_bound_for_maximize_and_upper_for_minimize():
    candidate = CandidateNode(
        node_id="A",
        smiles="CC",
        parent_id=None,
        depth=0,
        predictions={
            "potency": estimate("potency", 9.0, lower=8.2, upper=9.4),
            "herg": estimate("herg", 0.2, lower=0.1, upper=0.4),
        },
    )
    matrix, eligible, _ = robust_objective_matrix(
        [candidate],
        SPECS,
        [ObjectiveSpec("potency"), ObjectiveSpec("herg")],
    )
    assert eligible.tolist() == [True]
    assert matrix.tolist() == [[8.2, -0.4]]


def test_missing_and_out_of_domain_objectives_never_receive_favorable_credit():
    strict_candidates = [
        node("VALID", potency=8.0, herg=0.2),
        node("MISSING", potency=9.0, herg=None),
        node("OOD", potency=9.5, herg=0.01, inside=False),
    ]
    ranked = rank_candidates(
        strict_candidates,
        SPECS,
        [ObjectiveSpec("potency"), ObjectiveSpec("herg")],
    )
    assert [row.node_id for row in ranked] == ["VALID", "MISSING", "OOD"]
    assert ranked[0].eligible
    assert all(not row.eligible for row in ranked[1:])

    warning_specs = {
        "potency": SPECS["potency"],
        "herg": EndpointSpec(
            "herg",
            "classification",
            "minimize",
            missing_policy="warn",
            out_of_domain_policy="warn",
        ),
    }
    matrix, eligible, reasons = robust_objective_matrix(
        strict_candidates,
        warning_specs,
        [ObjectiveSpec("potency"), ObjectiveSpec("herg")],
    )
    assert eligible.tolist() == [True, True, False]
    # OOD also has an OOD potency value under the strict potency specification.
    assert matrix[1, 1] < matrix[0, 1]
    assert "worst-case credit" in reasons[1]


def test_path_cost_breaks_exact_ties_deterministically():
    candidates = [
        node("LONG", potency=8.0, herg=0.2, path_cost=2.0),
        node("SHORT", potency=8.0, herg=0.2, path_cost=1.0),
    ]
    ranked = rank_candidates(
        candidates,
        SPECS,
        [ObjectiveSpec("potency"), ObjectiveSpec("herg")],
    )
    assert [row.node_id for row in ranked] == ["SHORT", "LONG"]
