"""Immutable public schemas for Menin-Edit optimization.

The core engine exchanges plain, versioned dataclasses rather than estimator-
specific objects.  This keeps optimization sessions reproducible and makes the
same result payload usable from Python, a CLI, or a future HTTP service.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from types import MappingProxyType
from typing import Any, Literal

Direction = Literal["maximize", "minimize"]
TaskType = Literal["regression", "classification", "structural_score"]
EvidencePolicy = Literal["reject", "warn", "ignore"]
ConstraintOperator = Literal[">=", "<="]
ConstraintReference = Literal["absolute", "start", "previous"]
ConstraintScope = Literal["each_step", "final"]


def _require_finite(value: float, *, label: str) -> None:
    if not math.isfinite(float(value)):
        raise ValueError(f"{label} must be finite")


def _immutable_mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return MappingProxyType(dict(value or {}))


def _jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _jsonable(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


class JsonSerializable:
    """Stable JSON serialization shared by all public schemas."""

    def to_dict(self) -> dict[str, Any]:
        payload = _jsonable(self)
        if not isinstance(payload, dict):  # pragma: no cover - defensive contract
            raise TypeError("schema did not serialize to an object")
        return payload

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, allow_nan=False)


@dataclass(frozen=True)
class EndpointSpec(JsonSerializable):
    key: str
    task: TaskType
    direction: Direction
    display_unit: str | None = None
    model_version: str = "unversioned"
    missing_policy: EvidencePolicy = "reject"
    out_of_domain_policy: EvidencePolicy = "reject"

    def __post_init__(self) -> None:
        if not self.key.strip():
            raise ValueError("endpoint key must not be empty")
        if self.task not in {"regression", "classification", "structural_score"}:
            raise ValueError(f"unsupported endpoint task: {self.task!r}")
        if self.direction not in {"maximize", "minimize"}:
            raise ValueError(f"unsupported endpoint direction: {self.direction!r}")
        for name, policy in (
            ("missing_policy", self.missing_policy),
            ("out_of_domain_policy", self.out_of_domain_policy),
        ):
            if policy not in {"reject", "warn", "ignore"}:
                raise ValueError(f"unsupported {name}: {policy!r}")


@dataclass(frozen=True)
class ObjectiveSpec(JsonSerializable):
    endpoint: str
    priority: float = 1.0
    target: float | None = None
    minimum_meaningful_gain: float = 0.0

    def __post_init__(self) -> None:
        if not self.endpoint.strip():
            raise ValueError("objective endpoint must not be empty")
        _require_finite(self.priority, label="objective priority")
        _require_finite(self.minimum_meaningful_gain, label="minimum_meaningful_gain")
        if self.priority < 0:
            raise ValueError("objective priority must be non-negative")
        if self.minimum_meaningful_gain < 0:
            raise ValueError("minimum_meaningful_gain must be non-negative")
        if self.target is not None:
            _require_finite(self.target, label="objective target")


@dataclass(frozen=True)
class ConstraintSpec(JsonSerializable):
    endpoint: str
    operator: ConstraintOperator
    value: float
    confidence: float = 0.90
    relative_to: ConstraintReference = "absolute"
    apply_to: ConstraintScope = "each_step"
    missing_policy: Literal["reject", "warn"] = "reject"
    out_of_domain_policy: Literal["reject", "warn"] = "reject"

    def __post_init__(self) -> None:
        if not self.endpoint.strip():
            raise ValueError("constraint endpoint must not be empty")
        if self.operator not in {">=", "<="}:
            raise ValueError(f"unsupported constraint operator: {self.operator!r}")
        _require_finite(self.value, label="constraint value")
        _require_finite(self.confidence, label="constraint confidence")
        if not 0 < self.confidence < 1:
            raise ValueError("constraint confidence must lie strictly between zero and one")
        if self.relative_to not in {"absolute", "start", "previous"}:
            raise ValueError(f"unsupported constraint reference: {self.relative_to!r}")
        if self.apply_to not in {"each_step", "final"}:
            raise ValueError(f"unsupported constraint scope: {self.apply_to!r}")
        if self.missing_policy not in {"reject", "warn"}:
            raise ValueError(f"unsupported missing policy: {self.missing_policy!r}")
        if self.out_of_domain_policy not in {"reject", "warn"}:
            raise ValueError(f"unsupported out-of-domain policy: {self.out_of_domain_policy!r}")


@dataclass(frozen=True)
class SearchSpec(JsonSerializable):
    max_steps: int = 3
    beam_width: int = 30
    candidates_per_node: int = 100
    top_paths: int = 20
    max_changed_heavy_atoms: int = 8
    min_core_heavy_atoms: int = 10
    min_parent_similarity: float = 0.45
    uncertainty_penalty: float = 0.15
    path_complexity_penalty: float = 0.03
    require_supported_edit: bool = True
    random_seed: int = 13

    def __post_init__(self) -> None:
        positive = {
            "max_steps": self.max_steps,
            "beam_width": self.beam_width,
            "candidates_per_node": self.candidates_per_node,
            "top_paths": self.top_paths,
            "max_changed_heavy_atoms": self.max_changed_heavy_atoms,
            "min_core_heavy_atoms": self.min_core_heavy_atoms,
        }
        for name, positive_value in positive.items():
            if int(positive_value) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0 <= self.min_parent_similarity <= 1:
            raise ValueError("min_parent_similarity must lie between zero and one")
        for name, penalty_value in (
            ("uncertainty_penalty", self.uncertainty_penalty),
            ("path_complexity_penalty", self.path_complexity_penalty),
        ):
            _require_finite(penalty_value, label=name)
            if penalty_value < 0:
                raise ValueError(f"{name} must be non-negative")


@dataclass(frozen=True)
class PropertyEstimate(JsonSerializable):
    endpoint: str
    mean: float
    lower: float
    upper: float
    inside_domain: bool
    model_version: str
    evidence_status: str = "model_prediction"
    metadata: Mapping[str, Any] = MappingProxyType({})

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _immutable_mapping(self.metadata))
        if not self.endpoint.strip():
            raise ValueError("property estimate endpoint must not be empty")
        for name, estimate_value in (
            ("mean", self.mean),
            ("lower", self.lower),
            ("upper", self.upper),
        ):
            _require_finite(estimate_value, label=f"property estimate {name}")
        if self.lower > self.mean or self.mean > self.upper:
            raise ValueError("property estimate must satisfy lower <= mean <= upper")
        if not self.model_version.strip():
            raise ValueError("property estimate model_version must not be empty")


@dataclass(frozen=True)
class EditEffect(JsonSerializable):
    endpoint: str
    raw_delta: float
    utility_delta: float
    lower: float
    upper: float
    direct_delta: float | None = None
    absolute_model_delta: float | None = None
    disagreement: float | None = None

    def __post_init__(self) -> None:
        if not self.endpoint.strip():
            raise ValueError("edit-effect endpoint must not be empty")
        for name, value in (
            ("raw_delta", self.raw_delta),
            ("utility_delta", self.utility_delta),
            ("lower", self.lower),
            ("upper", self.upper),
        ):
            _require_finite(value, label=f"edit effect {name}")
        if self.lower > self.raw_delta or self.raw_delta > self.upper:
            raise ValueError("edit effect must satisfy lower <= raw_delta <= upper")
        for name, optional_value in (
            ("direct_delta", self.direct_delta),
            ("absolute_model_delta", self.absolute_model_delta),
            ("disagreement", self.disagreement),
        ):
            if optional_value is not None:
                _require_finite(optional_value, label=f"edit effect {name}")
        if self.disagreement is not None and self.disagreement < 0:
            raise ValueError("edit-effect disagreement must be non-negative")


@dataclass(frozen=True)
class MolecularEdit(JsonSerializable):
    rule_id: str
    transformation: str
    changed_atoms: tuple[int, ...]
    edit_type: str
    support_count: int = 0
    context_similarity: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "changed_atoms", tuple(int(value) for value in self.changed_atoms))
        if not self.rule_id.strip():
            raise ValueError("edit rule_id must not be empty")
        if not self.transformation.strip():
            raise ValueError("edit transformation must not be empty")
        if self.support_count < 0:
            raise ValueError("edit support_count must be non-negative")
        if not 0 <= self.context_similarity <= 1:
            raise ValueError("edit context_similarity must lie between zero and one")
        if any(value < 0 for value in self.changed_atoms):
            raise ValueError("changed atom indices must be non-negative")


@dataclass(frozen=True)
class CandidateNode(JsonSerializable):
    node_id: str
    smiles: str
    parent_id: str | None
    depth: int
    edit: MolecularEdit | None = None
    predictions: Mapping[str, PropertyEstimate] = MappingProxyType({})
    effects: Mapping[str, EditEffect] = MappingProxyType({})
    feasible: bool = True
    constraint_evaluations: tuple[BoundEvaluation, ...] = ()
    violations: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    path_cost: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "predictions", _immutable_mapping(self.predictions))
        object.__setattr__(self, "effects", _immutable_mapping(self.effects))
        object.__setattr__(self, "constraint_evaluations", tuple(self.constraint_evaluations))
        object.__setattr__(self, "violations", tuple(str(value) for value in self.violations))
        object.__setattr__(self, "warnings", tuple(str(value) for value in self.warnings))
        if not self.node_id.strip():
            raise ValueError("candidate node_id must not be empty")
        if not self.smiles.strip():
            raise ValueError("candidate smiles must not be empty")
        if self.depth < 0:
            raise ValueError("candidate depth must be non-negative")
        if self.depth == 0 and self.parent_id is not None:
            raise ValueError("depth-zero candidate cannot have a parent")
        if self.depth > 0 and (self.parent_id is None or self.edit is None):
            raise ValueError("non-root candidate requires both parent_id and edit")
        _require_finite(self.path_cost, label="candidate path_cost")
        if self.path_cost < 0:
            raise ValueError("candidate path_cost must be non-negative")
        for key, prediction in self.predictions.items():
            if key != prediction.endpoint:
                raise ValueError(f"prediction key {key!r} does not match endpoint")
        for key, effect in self.effects.items():
            if key != effect.endpoint:
                raise ValueError(f"effect key {key!r} does not match endpoint")


@dataclass(frozen=True)
class BoundEvaluation(JsonSerializable):
    endpoint: str
    passed: bool
    status: Literal["pass", "fail", "warning"]
    operator: ConstraintOperator
    bound: float
    mean: float | None
    conservative_value: float | None
    margin: float | None
    relative_to: ConstraintReference
    reason: str


@dataclass(frozen=True)
class ConstraintSummary(JsonSerializable):
    feasible: bool
    evaluations: tuple[BoundEvaluation, ...]
    violations: tuple[str, ...]
    warnings: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "evaluations", tuple(self.evaluations))
        object.__setattr__(self, "violations", tuple(str(value) for value in self.violations))
        object.__setattr__(self, "warnings", tuple(str(value) for value in self.warnings))


@dataclass(frozen=True)
class CandidateRanking(JsonSerializable):
    node_id: str
    eligible: bool
    pareto_rank: int | None
    priority_score: float | None
    order: int
    reason: str = ""


@dataclass(frozen=True)
class OptimizationRequest(JsonSerializable):
    starting_smiles: str
    objectives: tuple[ObjectiveSpec, ...]
    constraints: tuple[ConstraintSpec, ...] = ()
    search: SearchSpec = SearchSpec()

    def __post_init__(self) -> None:
        object.__setattr__(self, "objectives", tuple(self.objectives))
        object.__setattr__(self, "constraints", tuple(self.constraints))
        if not self.starting_smiles.strip():
            raise ValueError("starting_smiles must not be empty")
        if not self.objectives:
            raise ValueError("at least one objective is required")
        endpoints = [objective.endpoint for objective in self.objectives]
        if len(endpoints) != len(set(endpoints)):
            raise ValueError("objective endpoints must be unique")
        if sum(objective.priority for objective in self.objectives) <= 0:
            raise ValueError("at least one objective priority must be positive")


@dataclass(frozen=True)
class OptimizationResult(JsonSerializable):
    session_id: str
    request: OptimizationRequest
    baseline: CandidateNode
    candidates: tuple[CandidateNode, ...]
    rankings: tuple[CandidateRanking, ...]
    model_versions: Mapping[str, str]
    schema_version: str = "menin-edit-v1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidates", tuple(self.candidates))
        object.__setattr__(self, "rankings", tuple(self.rankings))
        object.__setattr__(self, "model_versions", _immutable_mapping(self.model_versions))
        if not self.session_id.strip():
            raise ValueError("optimization session_id must not be empty")


def as_jsonable(value: JsonSerializable | Sequence[JsonSerializable]) -> Any:
    """Return a JSON-safe representation without exposing mutable internals."""

    return _jsonable(value)
