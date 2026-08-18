"""Endpoint direction, interval-safe hard bounds, and objective utilities."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Literal

from .schemas import (
    BoundEvaluation,
    CandidateNode,
    ConstraintSpec,
    ConstraintSummary,
    EditEffect,
    EndpointSpec,
    ObjectiveSpec,
    PropertyEstimate,
)


def normalize_priorities(objectives: tuple[ObjectiveSpec, ...] | list[ObjectiveSpec]) -> dict[str, float]:
    """Return normalized, endpoint-keyed priorities without changing objectives."""

    if not objectives:
        raise ValueError("at least one objective is required")
    total = sum(float(objective.priority) for objective in objectives)
    if total <= 0:
        raise ValueError("at least one objective priority must be positive")
    return {objective.endpoint: float(objective.priority) / total for objective in objectives}


def directional_value(value: float, endpoint: EndpointSpec) -> float:
    """Orient a raw endpoint value so that larger always means more desirable."""

    return float(value) if endpoint.direction == "maximize" else -float(value)


def utility_delta(
    endpoint: EndpointSpec,
    parent: PropertyEstimate,
    product: PropertyEstimate,
) -> float:
    """Return positive values for desirable changes regardless of endpoint direction."""

    _require_matching_endpoint(endpoint, parent, product)
    raw_delta = float(product.mean - parent.mean)
    return raw_delta if endpoint.direction == "maximize" else -raw_delta


def build_edit_effect(
    endpoint: EndpointSpec,
    parent: PropertyEstimate,
    product: PropertyEstimate,
    *,
    direct_delta: float | None = None,
    disagreement: float | None = None,
) -> EditEffect:
    """Build an interval-valued edit effect from two absolute predictions.

    The independent-interval difference is intentionally conservative.  A
    paired delta predictor can later replace these limits while preserving the
    same public schema.
    """

    _require_matching_endpoint(endpoint, parent, product)
    raw_delta = float(product.mean - parent.mean)
    return EditEffect(
        endpoint=endpoint.key,
        raw_delta=raw_delta,
        utility_delta=raw_delta if endpoint.direction == "maximize" else -raw_delta,
        lower=float(product.lower - parent.upper),
        upper=float(product.upper - parent.lower),
        direct_delta=direct_delta,
        absolute_model_delta=raw_delta,
        disagreement=disagreement,
    )


def _require_matching_endpoint(
    endpoint: EndpointSpec,
    *estimates: PropertyEstimate,
) -> None:
    for estimate in estimates:
        if estimate.endpoint != endpoint.key:
            raise ValueError(f"estimate endpoint {estimate.endpoint!r} does not match {endpoint.key!r}")


def _missing_evaluation(constraint: ConstraintSpec, reason: str) -> BoundEvaluation:
    warning = constraint.missing_policy == "warn"
    return BoundEvaluation(
        endpoint=constraint.endpoint,
        passed=warning,
        status="warning" if warning else "fail",
        operator=constraint.operator,
        bound=float(constraint.value),
        mean=None,
        conservative_value=None,
        margin=None,
        relative_to=constraint.relative_to,
        reason=reason,
    )


def _relative_interval(
    candidate: PropertyEstimate,
    reference: PropertyEstimate,
    *,
    effect: EditEffect | None,
) -> tuple[float, float, float]:
    if effect is not None:
        if effect.endpoint != candidate.endpoint:
            raise ValueError("edit effect endpoint does not match candidate estimate")
        return float(effect.raw_delta), float(effect.lower), float(effect.upper)
    return (
        float(candidate.mean - reference.mean),
        float(candidate.lower - reference.upper),
        float(candidate.upper - reference.lower),
    )


def evaluate_constraint(
    constraint: ConstraintSpec,
    candidate: CandidateNode,
    *,
    start: CandidateNode | None = None,
    previous: CandidateNode | None = None,
) -> BoundEvaluation:
    """Evaluate one bound using the interval edge least favorable to passing."""

    estimate = candidate.predictions.get(constraint.endpoint)
    if estimate is None:
        return _missing_evaluation(constraint, "candidate prediction is missing")

    reference: CandidateNode | None = None
    effect: EditEffect | None = None
    if constraint.relative_to == "start":
        reference = start
    elif constraint.relative_to == "previous":
        reference = previous
        effect = candidate.effects.get(constraint.endpoint)

    reference_estimate: PropertyEstimate | None = None
    if constraint.relative_to != "absolute":
        if reference is None:
            return _missing_evaluation(
                constraint,
                f"{constraint.relative_to} reference candidate is missing",
            )
        reference_estimate = reference.predictions.get(constraint.endpoint)
        if reference_estimate is None:
            return _missing_evaluation(
                constraint,
                f"{constraint.relative_to} reference prediction is missing",
            )

    out_of_domain = not estimate.inside_domain or (
        reference_estimate is not None and not reference_estimate.inside_domain
    )
    if out_of_domain and constraint.out_of_domain_policy == "reject":
        return BoundEvaluation(
            endpoint=constraint.endpoint,
            passed=False,
            status="fail",
            operator=constraint.operator,
            bound=float(constraint.value),
            mean=None,
            conservative_value=None,
            margin=None,
            relative_to=constraint.relative_to,
            reason="prediction is outside the applicability domain",
        )

    if constraint.relative_to == "absolute":
        mean, lower, upper = float(estimate.mean), float(estimate.lower), float(estimate.upper)
    else:
        assert reference_estimate is not None  # narrowed above
        mean, lower, upper = _relative_interval(
            estimate,
            reference_estimate,
            effect=effect if constraint.relative_to == "previous" else None,
        )

    if constraint.operator == ">=":
        conservative = lower
        margin = conservative - float(constraint.value)
    else:
        conservative = upper
        margin = float(constraint.value) - conservative
    passed = margin >= 0
    warning = passed and out_of_domain and constraint.out_of_domain_policy == "warn"
    status: Literal["pass", "fail", "warning"]
    if not passed:
        reason = "conservative confidence bound violates the hard limit"
        status = "fail"
    elif warning:
        reason = "numeric bound passes, but prediction is outside the applicability domain"
        status = "warning"
    else:
        reason = "conservative confidence bound passes"
        status = "pass"
    return BoundEvaluation(
        endpoint=constraint.endpoint,
        passed=passed,
        status=status,
        operator=constraint.operator,
        bound=float(constraint.value),
        mean=mean,
        conservative_value=conservative,
        margin=margin,
        relative_to=constraint.relative_to,
        reason=reason,
    )


def evaluate_constraints(
    constraints: tuple[ConstraintSpec, ...] | list[ConstraintSpec],
    candidate: CandidateNode,
    *,
    start: CandidateNode | None = None,
    previous: CandidateNode | None = None,
    is_final: bool = False,
) -> ConstraintSummary:
    """Evaluate applicable bounds and preserve separate failures and warnings."""

    applicable = [constraint for constraint in constraints if constraint.apply_to == "each_step" or is_final]
    evaluations = tuple(
        evaluate_constraint(constraint, candidate, start=start, previous=previous)
        for constraint in applicable
    )
    violations = tuple(
        f"{evaluation.endpoint}: {evaluation.reason}" for evaluation in evaluations if not evaluation.passed
    )
    warnings = tuple(
        f"{evaluation.endpoint}: {evaluation.reason}"
        for evaluation in evaluations
        if evaluation.status == "warning"
    )
    return ConstraintSummary(
        feasible=not violations,
        evaluations=evaluations,
        violations=violations,
        warnings=warnings,
    )


def apply_constraint_summary(candidate: CandidateNode, summary: ConstraintSummary) -> CandidateNode:
    """Return an immutable candidate copy carrying hard-bound feasibility."""

    return replace(
        candidate,
        feasible=summary.feasible,
        constraint_evaluations=summary.evaluations,
        violations=summary.violations,
        warnings=summary.warnings,
    )


def endpoint_specs_by_key(specs: Mapping[str, EndpointSpec] | list[EndpointSpec]) -> dict[str, EndpointSpec]:
    if isinstance(specs, Mapping):
        result = dict(specs)
    else:
        result = {spec.key: spec for spec in specs}
        if len(result) != len(specs):
            raise ValueError("endpoint keys must be unique")
    for key, spec in result.items():
        if key != spec.key:
            raise ValueError(f"endpoint registry key {key!r} does not match spec.key")
    return result
