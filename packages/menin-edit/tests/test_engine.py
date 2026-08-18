from __future__ import annotations

from dataclasses import dataclass

import pytest

from menin_edit.chemistry import FragmentContext, canonicalize_smiles
from menin_edit.edits import EditRule, GeneratedEdit
from menin_edit.engine import MeninEditEngine
from menin_edit.registry import PredictorRegistry
from menin_edit.schemas import (
    ConstraintSpec,
    EndpointSpec,
    ObjectiveSpec,
    OptimizationRequest,
    PropertyEstimate,
    SearchSpec,
)


@dataclass
class _TablePredictor:
    endpoint: str
    values: dict[str, tuple[float, float, float]]
    predict_count: int = 0
    batch_count: int = 0

    @property
    def model_version(self) -> str:
        return f"{self.endpoint}-table-v1"

    def _estimate(self, smiles: str) -> PropertyEstimate:
        mean, lower, upper = self.values[canonicalize_smiles(smiles)]
        return PropertyEstimate(
            endpoint=self.endpoint,
            mean=mean,
            lower=lower,
            upper=upper,
            inside_domain=True,
            model_version=self.model_version,
        )

    def predict(self, smiles: str) -> PropertyEstimate:
        self.predict_count += 1
        return self._estimate(smiles)

    def predict_many(self, smiles):
        self.batch_count += 1
        return [self._estimate(value) for value in smiles]


class _DeterministicLibrary:
    def __init__(self) -> None:
        self.transitions = {
            "CC": (("CCC", "R-ADD-C", 2.0, 8), ("CCO", "R-ADD-O", 1.5, 5)),
            "CCC": (("CCCC", "R-EXTEND", 1.0, 4),),
            "CCO": (("CCN", "R-O-TO-N", 0.7, 6),),
        }

    def enumerate(
        self,
        parent_smiles,
        *,
        candidates_per_node,
        visited_smiles=None,
        **_kwargs,
    ):
        parent = canonicalize_smiles(parent_smiles)
        visited = visited_smiles or set()
        generated = []
        for product, rule_id, direct_delta, support in self.transitions.get(parent, ()):
            canonical_product = canonicalize_smiles(product)
            if canonical_product in visited:
                continue
            rule = EditRule(
                rule_id=rule_id,
                source_fragment="C[*:1]",
                target_fragment="N[*:1]",
                support_count=support,
                endpoint_mean_deltas={"potency": direct_delta},
                changed_heavy_atoms=1,
            )
            generated.append(
                GeneratedEdit(
                    rule=rule,
                    parent_smiles=parent,
                    product_smiles=canonical_product,
                    context=FragmentContext("C[*:1]", "C[*:1]", 1, 1),
                    parent_similarity=0.7,
                    heavy_atom_delta=1,
                )
            )
        return generated[:candidates_per_node]

    def evidence_for(self, _rule_id):
        return ()


ENDPOINTS = {
    "potency": EndpointSpec("potency", "regression", "maximize"),
    "herg": EndpointSpec("herg", "classification", "minimize"),
}


def _engine():
    potency = _TablePredictor(
        "potency",
        {
            "CC": (5.0, 4.9, 5.1),
            "CCC": (7.0, 6.9, 7.1),
            "CCO": (6.5, 6.4, 6.6),
            "CCCC": (8.0, 7.9, 8.1),
            "CCN": (7.2, 7.1, 7.3),
        },
    )
    herg = _TablePredictor(
        "herg",
        {
            "CC": (0.40, 0.35, 0.45),
            "CCC": (0.65, 0.60, 0.70),
            "CCO": (0.15, 0.10, 0.20),
            "CCCC": (0.85, 0.80, 0.90),
            "CCN": (0.20, 0.15, 0.25),
        },
    )
    registry = PredictorRegistry({"potency": potency, "herg": herg})
    return (
        MeninEditEngine(
            endpoints=ENDPOINTS,
            predictors=registry,
            edit_library=_DeterministicLibrary(),
        ),
        potency,
        herg,
    )


def _request(*, constraints=()):
    return OptimizationRequest(
        starting_smiles="CC",
        objectives=(ObjectiveSpec("potency", 0.6), ObjectiveSpec("herg", 0.4)),
        constraints=constraints,
        search=SearchSpec(
            max_steps=2,
            beam_width=2,
            candidates_per_node=4,
            top_paths=4,
            min_core_heavy_atoms=1,
            max_changed_heavy_atoms=2,
            min_parent_similarity=0.0,
            uncertainty_penalty=0.1,
            path_complexity_penalty=0.02,
        ),
    )


def _by_smiles(result):
    return {candidate.smiles: candidate for candidate in result.candidates}


def test_engine_builds_stepwise_paths_constraints_effects_and_pareto_fronts():
    engine, _potency, _herg = _engine()
    request = _request(constraints=(ConstraintSpec("herg", "<=", 0.75),))

    result = engine.optimize(request)
    candidates = _by_smiles(result)

    assert set(candidates) == {"CCC", "CCO", "CCCC", "CCN"}
    assert candidates["CCCC"].feasible is False
    assert "herg" in candidates["CCCC"].violations[0]
    assert candidates["CCN"].feasible is True
    assert candidates["CCN"].parent_id == candidates["CCO"].node_id
    assert candidates["CCN"].effects["potency"].raw_delta == pytest.approx(0.7)
    assert candidates["CCN"].effects["potency"].direct_delta == pytest.approx(0.7)
    assert candidates["CCN"].path_cost > candidates["CCO"].path_cost > 0

    path = engine.get_path(result.session_id, candidates["CCN"].node_id)
    assert [node.smiles for node in path] == ["CC", "CCO", "CCN"]
    rankings = {row.node_id: row for row in result.rankings}
    assert not rankings[candidates["CCCC"].node_id].eligible
    assert rankings[candidates["CCO"].node_id].pareto_rank == 1
    assert rankings[candidates["CCN"].node_id].pareto_rank == 1
    assert rankings[candidates["CCC"].node_id].pareto_rank == 2


def test_final_bounds_apply_to_every_stopping_point_but_do_not_block_recovery():
    engine, _potency, _herg = _engine()
    request = _request(
        constraints=(
            ConstraintSpec("herg", "<=", 0.75, apply_to="each_step"),
            ConstraintSpec("potency", ">=", 7.0, apply_to="final"),
        )
    )

    result = engine.optimize(request)
    candidates = _by_smiles(result)

    assert not candidates["CCO"].feasible
    assert any("potency" in violation for violation in candidates["CCO"].violations)
    assert "CCN" in candidates, "a recoverable final-only failure must remain expandable"
    assert candidates["CCN"].feasible
    assert not candidates["CCCC"].feasible


def test_rescore_uses_cached_predictions_propagates_path_failure_and_continue_works():
    engine, potency, herg = _engine()
    result = engine.optimize(_request())
    calls_after_search = (
        potency.predict_count,
        potency.batch_count,
        herg.predict_count,
        herg.batch_count,
        engine.predictors.cache_size,
    )
    candidates = _by_smiles(result)

    rescored = engine.rescore(
        result.session_id,
        objectives=(ObjectiveSpec("potency", 0.2), ObjectiveSpec("herg", 0.8)),
        constraints=(ConstraintSpec("potency", ">=", 6.8, apply_to="each_step"),),
    )
    rescored_candidates = _by_smiles(rescored)

    assert not rescored_candidates["CCO"].feasible
    assert not rescored_candidates["CCN"].feasible
    assert "candidate path violates an each-step hard bound" in rescored_candidates["CCN"].violations
    assert (
        potency.predict_count,
        potency.batch_count,
        herg.predict_count,
        herg.batch_count,
        engine.predictors.cache_size,
    ) == calls_after_search

    follow_up = OptimizationRequest(
        starting_smiles="ignored-by-continue",
        objectives=(ObjectiveSpec("potency"), ObjectiveSpec("herg")),
        search=SearchSpec(
            max_steps=1,
            beam_width=2,
            candidates_per_node=2,
            top_paths=2,
            min_core_heavy_atoms=1,
            max_changed_heavy_atoms=2,
            min_parent_similarity=0.0,
        ),
    )
    continued = engine.continue_from(
        result.session_id,
        candidates["CCO"].node_id,
        request=follow_up,
    )

    assert continued.baseline.smiles == "CCO"
    assert [candidate.smiles for candidate in continued.candidates] == ["CCN"]
    assert [
        node.smiles for node in engine.get_path(continued.session_id, continued.candidates[0].node_id)
    ] == ["CCO", "CCN"]
    assert engine.get_result(continued.session_id) is continued
