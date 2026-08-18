"""Deterministic, uncertainty-aware Pareto ranking for Menin-Edit."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from .objectives import endpoint_specs_by_key, normalize_priorities
from .schemas import CandidateNode, CandidateRanking, EndpointSpec, ObjectiveSpec, PropertyEstimate


def pareto_ranks(values: np.ndarray, eligible: np.ndarray | None = None) -> np.ndarray:
    """Return one-based non-dominated sorting ranks; ineligible rows receive NaN.

    Every column is assumed to be oriented so that larger is better.
    Deterministic iteration makes ties and repeated runs stable.
    """

    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("Pareto values must be a two-dimensional matrix")
    valid = np.all(np.isfinite(matrix), axis=1)
    if eligible is not None:
        supplied = np.asarray(eligible, dtype=bool)
        if supplied.shape != (len(matrix),):
            raise ValueError("eligible mask length must match the number of rows")
        valid &= supplied
    ranks = np.full(len(matrix), np.nan, dtype=float)
    remaining = set(np.flatnonzero(valid).tolist())
    rank = 1
    while remaining:
        front: list[int] = []
        for candidate in sorted(remaining):
            dominated = any(
                other != candidate
                and np.all(matrix[other] >= matrix[candidate])
                and np.any(matrix[other] > matrix[candidate])
                for other in remaining
            )
            if not dominated:
                front.append(candidate)
        if not front:  # pragma: no cover - strict dominance guarantees a front
            raise RuntimeError("could not resolve Pareto front")
        ranks[front] = rank
        remaining.difference_update(front)
        rank += 1
    return ranks


def robust_objective_value(
    estimate: PropertyEstimate,
    endpoint: EndpointSpec,
    objective: ObjectiveSpec | None = None,
) -> float:
    """Return the confidence-bound value in a common higher-is-better orientation."""

    if estimate.endpoint != endpoint.key:
        raise ValueError("property estimate and endpoint specification do not match")
    value = float(estimate.lower) if endpoint.direction == "maximize" else -float(estimate.upper)
    if objective is not None and objective.target is not None:
        target = float(objective.target) if endpoint.direction == "maximize" else -float(objective.target)
        value = min(value, target)
    return value


def _worst_fill(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return 0.0
    span = float(np.max(finite) - np.min(finite))
    return float(np.min(finite) - max(1e-9, 0.01 * max(span, 1.0)))


def robust_objective_matrix(
    candidates: Sequence[CandidateNode],
    endpoint_specs: Mapping[str, EndpointSpec] | list[EndpointSpec],
    objectives: Sequence[ObjectiveSpec],
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """Create the robust utility matrix and eligibility mask.

    Missing or out-of-domain values never receive favorable credit.  A
    ``reject`` policy removes the candidate; ``warn`` uses a worst-case fill;
    ``ignore`` uses an available OOD value but still worst-fills missing data.
    """

    specs = endpoint_specs_by_key(endpoint_specs)
    if not candidates:
        return np.empty((0, len(objectives))), np.empty(0, dtype=bool), ()
    if not objectives:
        raise ValueError("at least one objective is required")
    unknown = [objective.endpoint for objective in objectives if objective.endpoint not in specs]
    if unknown:
        raise KeyError(f"objective endpoint specifications are missing: {sorted(set(unknown))}")

    matrix = np.full((len(candidates), len(objectives)), np.nan, dtype=float)
    eligible = np.asarray([candidate.feasible for candidate in candidates], dtype=bool)
    reasons: list[list[str]] = [
        ([] if candidate.feasible else ["candidate failed hard constraints"]) for candidate in candidates
    ]
    needs_worst_fill = np.zeros_like(matrix, dtype=bool)

    for column, objective in enumerate(objectives):
        spec = specs[objective.endpoint]
        for row, candidate in enumerate(candidates):
            estimate = candidate.predictions.get(objective.endpoint)
            if estimate is None:
                if spec.missing_policy == "reject":
                    eligible[row] = False
                    reasons[row].append(f"missing required objective {objective.endpoint}")
                else:
                    needs_worst_fill[row, column] = True
                    if spec.missing_policy == "warn":
                        reasons[row].append(f"missing objective {objective.endpoint}; worst-case credit")
                continue
            if not estimate.inside_domain:
                if spec.out_of_domain_policy == "reject":
                    eligible[row] = False
                    reasons[row].append(f"objective {objective.endpoint} is outside domain")
                    continue
                if spec.out_of_domain_policy == "warn":
                    needs_worst_fill[row, column] = True
                    reasons[row].append(
                        f"objective {objective.endpoint} is outside domain; worst-case credit"
                    )
                    continue
            matrix[row, column] = robust_objective_value(estimate, spec, objective)

    for column in range(matrix.shape[1]):
        fill = _worst_fill(matrix[:, column])
        matrix[needs_worst_fill[:, column], column] = fill
    reason_text = tuple("; ".join(dict.fromkeys(items)) for items in reasons)
    return matrix, eligible, reason_text


def _priority_scores(
    values: np.ndarray,
    eligible: np.ndarray,
    objectives: Sequence[ObjectiveSpec],
) -> np.ndarray:
    """Min-max normalize endpoint scales, then combine user priorities."""

    result = np.full(len(values), np.nan, dtype=float)
    indices = np.flatnonzero(eligible & np.all(np.isfinite(values), axis=1))
    if not len(indices):
        return result
    subset = values[indices]
    normalized = np.zeros_like(subset, dtype=float)
    for column in range(subset.shape[1]):
        low = float(np.min(subset[:, column]))
        high = float(np.max(subset[:, column]))
        normalized[:, column] = (subset[:, column] - low) / (high - low) if high > low else 0.5
    weights_by_endpoint = normalize_priorities(list(objectives))
    weights = np.asarray([weights_by_endpoint[objective.endpoint] for objective in objectives])
    result[indices] = normalized @ weights
    return result


def rank_candidates(
    candidates: Sequence[CandidateNode],
    endpoint_specs: Mapping[str, EndpointSpec] | list[EndpointSpec],
    objectives: Sequence[ObjectiveSpec],
) -> tuple[CandidateRanking, ...]:
    """Rank feasible candidates by Pareto front, then by user priorities.

    Priority scores are deliberately a within-front tie-breaker.  No weight can
    move a dominated candidate ahead of a candidate on a better Pareto front.
    """

    node_ids = [candidate.node_id for candidate in candidates]
    if len(node_ids) != len(set(node_ids)):
        raise ValueError("candidate node_ids must be unique")
    matrix, eligible, reasons = robust_objective_matrix(candidates, endpoint_specs, objectives)
    ranks = pareto_ranks(matrix, eligible)
    priority = _priority_scores(matrix, eligible, objectives)
    ordered_indices = sorted(
        range(len(candidates)),
        key=lambda index: (
            not bool(eligible[index]),
            int(ranks[index]) if np.isfinite(ranks[index]) else 10**9,
            -float(priority[index]) if np.isfinite(priority[index]) else 0.0,
            float(candidates[index].path_cost),
            candidates[index].node_id,
        ),
    )
    order_by_index = {index: order for order, index in enumerate(ordered_indices, start=1)}
    rows = [
        CandidateRanking(
            node_id=candidate.node_id,
            eligible=bool(eligible[index]),
            pareto_rank=int(ranks[index]) if np.isfinite(ranks[index]) else None,
            priority_score=float(priority[index]) if np.isfinite(priority[index]) else None,
            order=order_by_index[index],
            reason=reasons[index],
        )
        for index, candidate in enumerate(candidates)
    ]
    return tuple(sorted(rows, key=lambda row: row.order))
