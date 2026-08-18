"""Stepwise, bounded, Pareto-ranked Menin molecular-edit engine."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pandas as pd

from .chemistry import canonicalize_smiles, tanimoto_similarity
from .config import MeninEditConfig, load_config
from .edits import EditLibrary, GeneratedEdit
from .evidence import EvidenceIndex
from .objectives import apply_constraint_summary, build_edit_effect, evaluate_constraints
from .pareto import rank_candidates
from .registry import PredictorRegistry
from .schemas import (
    CandidateNode,
    CandidateRanking,
    MolecularEdit,
    OptimizationRequest,
    OptimizationResult,
)


@dataclass
class _Session:
    request: OptimizationRequest
    baseline: CandidateNode
    nodes: dict[str, CandidateNode]
    result: OptimizationResult


class MeninEditEngine:
    """Generate supported edits one step at a time and retain the full path graph."""

    def __init__(
        self,
        *,
        endpoints: Mapping[str, Any],
        predictors: PredictorRegistry,
        edit_library: EditLibrary,
        config: MeninEditConfig | None = None,
    ) -> None:
        self.endpoints = dict(endpoints)
        self.predictors = predictors
        self.edit_library = edit_library
        self.evidence = EvidenceIndex(edit_library)
        self.config = config
        self._sessions: dict[str, _Session] = {}

    @classmethod
    def from_config(cls, path: str | Path) -> MeninEditEngine:
        config = load_config(path)
        predictors = PredictorRegistry.from_config(
            config.model_configs,
            repository_root=config.repository_root,
        )
        pair_path = config.edit_library.get("public_pairs")
        if not pair_path:
            raise ValueError("edit_library.public_pairs must be configured")
        library = EditLibrary.from_public_pairs(
            pair_path,
            minimum_support=int(config.edit_library.get("minimum_support", 1)),
            allowed_split_roles=tuple(
                config.edit_library.get("allowed_split_roles", ("train", "development"))
            ),
        )
        private_pair_path = config.edit_library.get("private_pairs")
        if private_pair_path:
            private_library = EditLibrary.from_long_pairs(
                pd.read_csv(private_pair_path),
                minimum_support=int(config.edit_library.get("private_minimum_support", 1)),
                allowed_split_roles=tuple(
                    config.edit_library.get("allowed_split_roles", ("train", "development"))
                ),
            )
            library = EditLibrary.merge(
                library,
                private_library,
                minimum_support=int(config.edit_library.get("minimum_support", 1)),
            )
        return cls(
            endpoints=config.endpoints,
            predictors=predictors,
            edit_library=library,
            config=config,
        )

    def default_request(self, starting_smiles: str) -> OptimizationRequest:
        if self.config is None:
            raise RuntimeError("No configuration is attached to this engine")
        return OptimizationRequest(
            starting_smiles=starting_smiles,
            objectives=self.config.objectives,
            constraints=self.config.constraints,
            search=self.config.search,
        )

    def score(self, smiles: str) -> CandidateNode:
        canonical = canonicalize_smiles(smiles)
        predictions = self.predictors.predict_all(canonical)
        return CandidateNode(
            node_id=_root_node_id(canonical),
            smiles=canonical,
            parent_id=None,
            depth=0,
            predictions=predictions,
        )

    def optimize(self, request: OptimizationRequest) -> OptimizationResult:
        canonical_request = replace(
            request,
            starting_smiles=canonicalize_smiles(request.starting_smiles),
        )
        _validate_request_endpoints(canonical_request, self.endpoints, self.predictors)
        baseline = self.score(canonical_request.starting_smiles)
        baseline_summary = evaluate_constraints(
            canonical_request.constraints,
            baseline,
            start=baseline,
            previous=None,
            is_final=canonical_request.search.max_steps == 0,
        )
        baseline = apply_constraint_summary(baseline, baseline_summary)
        nodes: dict[str, CandidateNode] = {baseline.node_id: baseline}
        beam = [baseline]

        for depth in range(1, canonical_request.search.max_steps + 1):
            proposals: list[tuple[CandidateNode, GeneratedEdit]] = []
            for parent in beam:
                visited = {node.smiles for node in self._path_nodes(parent, nodes)}
                generated = self.edit_library.enumerate(
                    parent.smiles,
                    min_core_heavy_atoms=canonical_request.search.min_core_heavy_atoms,
                    max_changed_heavy_atoms=canonical_request.search.max_changed_heavy_atoms,
                    min_parent_similarity=canonical_request.search.min_parent_similarity,
                    candidates_per_node=canonical_request.search.candidates_per_node,
                    visited_smiles=visited,
                )
                proposals.extend((parent, edit) for edit in generated)
            if not proposals:
                break
            prediction_batches = self.predictors.predict_all_many(
                [edit.product_smiles for _parent, edit in proposals]
            )
            children: list[CandidateNode] = []
            for (parent, edit), predictions in zip(proposals, prediction_batches, strict=True):
                child = self._build_child(
                    parent,
                    edit,
                    predictions=predictions,
                    depth=depth,
                    request=canonical_request,
                    baseline=baseline,
                    is_final=depth == canonical_request.search.max_steps,
                )
                existing = nodes.get(child.node_id)
                if existing is None or child.path_cost < existing.path_cost:
                    nodes[child.node_id] = child
                    children.append(child)
            if not children:
                break
            depth_rankings = rank_candidates(children, self.endpoints, canonical_request.objectives)
            beam = self._select_diverse_beam(
                children,
                depth_rankings,
                width=canonical_request.search.beam_width,
            )
            if not beam:
                break

        # Every retained node is a user-selectable stopping point, including an
        # intermediate step.  Final-only bounds must therefore be evaluated on
        # every returned candidate, while remaining non-pruning during beam
        # expansion so a later edit can still rescue an intermediate miss.
        for node in sorted(nodes.values(), key=lambda value: (value.depth, value.node_id)):
            if node.depth == 0:
                continue
            previous = nodes.get(node.parent_id or "")
            stopping_summary = evaluate_constraints(
                canonical_request.constraints,
                node,
                start=baseline,
                previous=previous,
                is_final=True,
            )
            nodes[node.node_id] = apply_constraint_summary(node, stopping_summary)
        candidates = tuple(
            sorted(
                (node for node in nodes.values() if node.depth > 0),
                key=lambda node: (node.depth, node.node_id),
            )
        )
        rankings = rank_candidates(candidates, self.endpoints, canonical_request.objectives)
        session_id = _session_id(canonical_request, candidates)
        result = OptimizationResult(
            session_id=session_id,
            request=canonical_request,
            baseline=baseline,
            candidates=candidates,
            rankings=rankings,
            model_versions=self.predictors.model_versions(),
        )
        self._sessions[session_id] = _Session(
            request=canonical_request,
            baseline=baseline,
            nodes=nodes,
            result=result,
        )
        return result

    def rescore(
        self,
        session_id: str,
        *,
        objectives: tuple[Any, ...],
        constraints: tuple[Any, ...],
    ) -> OptimizationResult:
        """Apply new priorities/bounds to cached predictions without rerunning models."""

        session = self._sessions[session_id]
        request = replace(session.request, objectives=tuple(objectives), constraints=tuple(constraints))
        _validate_request_endpoints(request, self.endpoints, self.predictors)
        rescored: dict[str, CandidateNode] = {session.baseline.node_id: session.baseline}
        traversal_feasible: dict[str, bool] = {session.baseline.node_id: True}
        for original in sorted(session.nodes.values(), key=lambda node: (node.depth, node.node_id)):
            if original.depth == 0:
                continue
            previous = rescored.get(original.parent_id or "")
            traversal_summary = evaluate_constraints(
                request.constraints,
                original,
                start=session.baseline,
                previous=previous,
                is_final=False,
            )
            path_is_feasible = traversal_summary.feasible and traversal_feasible.get(
                original.parent_id or "", True
            )
            stopping_summary = evaluate_constraints(
                request.constraints,
                original,
                start=session.baseline,
                previous=previous,
                is_final=True,
            )
            candidate = apply_constraint_summary(original, stopping_summary)
            if not path_is_feasible:
                candidate = replace(
                    candidate,
                    feasible=False,
                    violations=tuple(candidate.violations)
                    + ("candidate path violates an each-step hard bound",),
                )
            traversal_feasible[candidate.node_id] = path_is_feasible
            rescored[candidate.node_id] = candidate
        candidates = tuple(
            sorted(
                (node for node in rescored.values() if node.depth > 0),
                key=lambda node: (node.depth, node.node_id),
            )
        )
        rankings = rank_candidates(candidates, self.endpoints, request.objectives)
        result = OptimizationResult(
            session_id=session_id,
            request=request,
            baseline=session.baseline,
            candidates=candidates,
            rankings=rankings,
            model_versions=session.result.model_versions,
        )
        session.request = request
        session.nodes = rescored
        session.result = result
        return result

    def continue_from(
        self,
        session_id: str,
        node_id: str,
        *,
        request: OptimizationRequest | None = None,
    ) -> OptimizationResult:
        """Start another auditable search from any retained intermediate structure."""

        session = self._sessions[session_id]
        node = session.nodes[node_id]
        next_request = request or replace(session.request, starting_smiles=node.smiles)
        next_request = replace(next_request, starting_smiles=node.smiles)
        return self.optimize(next_request)

    def get_result(self, session_id: str) -> OptimizationResult:
        return self._sessions[session_id].result

    def get_path(self, session_id: str, node_id: str) -> tuple[CandidateNode, ...]:
        session = self._sessions[session_id]
        return tuple(self._path_nodes(session.nodes[node_id], session.nodes))

    def _build_child(
        self,
        parent: CandidateNode,
        generated: GeneratedEdit,
        *,
        predictions: Mapping[str, Any],
        depth: int,
        request: OptimizationRequest,
        baseline: CandidateNode,
        is_final: bool,
    ) -> CandidateNode:
        effects = {}
        for endpoint, spec in self.endpoints.items():
            if endpoint not in parent.predictions or endpoint not in predictions:
                continue
            absolute_delta = predictions[endpoint].mean - parent.predictions[endpoint].mean
            direct = generated.rule.endpoint_mean_deltas.get(endpoint)
            disagreement = abs(float(direct) - absolute_delta) if direct is not None else None
            effects[endpoint] = build_edit_effect(
                spec,
                parent.predictions[endpoint],
                predictions[endpoint],
                direct_delta=None if direct is None else float(direct),
                disagreement=disagreement,
            )
        edit = MolecularEdit(
            rule_id=generated.rule.rule_id,
            transformation=generated.rule.transformation,
            changed_atoms=(),
            edit_type="supported_single_fragment_replacement",
            support_count=generated.rule.support_count,
            context_similarity=generated.parent_similarity,
        )
        active_endpoints = {
            *(objective.endpoint for objective in request.objectives),
            *(constraint.endpoint for constraint in request.constraints),
        }
        active_estimates = [
            estimate for endpoint, estimate in predictions.items() if endpoint in active_endpoints
        ]
        mean_interval_width = (
            sum(value.upper - value.lower for value in active_estimates) / len(active_estimates)
            if active_estimates
            else 0.0
        )
        path_cost = (
            parent.path_cost
            + request.search.path_complexity_penalty
            + request.search.uncertainty_penalty * mean_interval_width
            + 0.001 * generated.rule.changed_heavy_atoms
        )
        node = CandidateNode(
            node_id=_child_node_id(parent.node_id, generated.product_smiles, generated.rule.rule_id),
            smiles=generated.product_smiles,
            parent_id=parent.node_id,
            depth=depth,
            edit=edit,
            predictions=predictions,
            effects=effects,
            path_cost=path_cost,
        )
        summary = evaluate_constraints(
            request.constraints,
            node,
            start=baseline,
            previous=parent,
            is_final=is_final,
        )
        return apply_constraint_summary(node, summary)

    @staticmethod
    def _path_nodes(node: CandidateNode, nodes: Mapping[str, CandidateNode]) -> list[CandidateNode]:
        path = [node]
        current = node
        while current.parent_id is not None:
            current = nodes[current.parent_id]
            path.append(current)
        return list(reversed(path))

    @staticmethod
    def _select_diverse_beam(
        candidates: list[CandidateNode],
        rankings: tuple[CandidateRanking, ...],
        *,
        width: int,
    ) -> list[CandidateNode]:
        by_id = {candidate.node_id: candidate for candidate in candidates}
        selected: list[CandidateNode] = []
        for ranking in rankings:
            if not ranking.eligible:
                continue
            candidate = by_id[ranking.node_id]
            if any(tanimoto_similarity(candidate.smiles, kept.smiles) >= 0.98 for kept in selected):
                continue
            selected.append(candidate)
            if len(selected) >= width:
                break
        return selected


def _validate_request_endpoints(
    request: OptimizationRequest,
    endpoint_specs: Mapping[str, Any],
    predictors: PredictorRegistry,
) -> None:
    requested = {objective.endpoint for objective in request.objectives}
    requested.update(constraint.endpoint for constraint in request.constraints)
    missing_specs = sorted(requested - set(endpoint_specs))
    if missing_specs:
        raise KeyError(f"Request references unregistered endpoints: {missing_specs}")
    missing_models = sorted(requested - set(predictors.predictors))
    if missing_models:
        raise KeyError("Request requires endpoints without active predictors: " + ", ".join(missing_models))


def _root_node_id(smiles: str) -> str:
    return "NODE-" + hashlib.sha256(f"root\0{smiles}".encode()).hexdigest()[:20].upper()


def _child_node_id(parent_id: str, smiles: str, rule_id: str) -> str:
    return "NODE-" + hashlib.sha256(f"{parent_id}\0{smiles}\0{rule_id}".encode()).hexdigest()[:20].upper()


def _session_id(request: OptimizationRequest, candidates: tuple[CandidateNode, ...]) -> str:
    payload = request.to_json(indent=None) + "\0" + "\0".join(node.node_id for node in candidates)
    return "SESSION-" + hashlib.sha256(payload.encode()).hexdigest()[:20].upper()
