"""Validated YAML configuration for the Menin-Edit engine."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, cast

import yaml

from .schemas import (
    ConstraintOperator,
    ConstraintReference,
    ConstraintScope,
    ConstraintSpec,
    Direction,
    EndpointSpec,
    EvidencePolicy,
    ObjectiveSpec,
    SearchSpec,
    TaskType,
)


@dataclass(frozen=True)
class MeninEditConfig:
    source_path: Path
    repository_root: Path
    model_configs: Mapping[str, Mapping[str, Any]]
    endpoints: Mapping[str, EndpointSpec]
    objectives: tuple[ObjectiveSpec, ...]
    constraints: tuple[ConstraintSpec, ...]
    search: SearchSpec
    edit_library: Mapping[str, Any]
    raw: Mapping[str, Any]


def _resolve_path(value: str | Path, *, base: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def load_config(path: str | Path) -> MeninEditConfig:
    source = Path(path).expanduser().resolve()
    payload = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise TypeError("Menin-Edit configuration must be a YAML mapping")
    config_root = source.parent
    repository_root = _resolve_path(payload.get("repository_root", "../.."), base=config_root)

    endpoints: dict[str, EndpointSpec] = {}
    for key, definition in dict(payload.get("endpoints", {})).items():
        definition = dict(definition or {})
        endpoints[str(key)] = EndpointSpec(
            key=str(key),
            task=cast(TaskType, str(definition.get("task", "regression"))),
            direction=cast(Direction, str(definition.get("direction", "maximize"))),
            display_unit=definition.get("display_unit"),
            model_version=str(definition.get("model_version", "configured")),
            missing_policy=cast(EvidencePolicy, str(definition.get("missing_policy", "reject"))),
            out_of_domain_policy=cast(EvidencePolicy, str(definition.get("out_of_domain_policy", "reject"))),
        )
    if not endpoints:
        raise ValueError("At least one endpoint must be configured")

    objectives = tuple(
        ObjectiveSpec(
            endpoint=str(item["endpoint"]),
            priority=float(item.get("priority", 1.0)),
            target=None if item.get("target") is None else float(item["target"]),
            minimum_meaningful_gain=float(item.get("minimum_meaningful_gain", 0.0)),
        )
        for item in payload.get("objectives", [])
    )
    if not objectives:
        raise ValueError("At least one objective must be configured")
    _require_registered([item.endpoint for item in objectives], endpoints, "objective")

    constraints = tuple(
        ConstraintSpec(
            endpoint=str(item["endpoint"]),
            operator=cast(ConstraintOperator, str(item["operator"])),
            value=float(item["value"]),
            confidence=float(item.get("confidence", 0.90)),
            relative_to=cast(ConstraintReference, str(item.get("relative_to", "absolute"))),
            apply_to=cast(ConstraintScope, str(item.get("apply_to", "each_step"))),
            missing_policy=cast(Literal["reject", "warn"], str(item.get("missing_policy", "reject"))),
            out_of_domain_policy=cast(
                Literal["reject", "warn"], str(item.get("out_of_domain_policy", "reject"))
            ),
        )
        for item in payload.get("constraints", [])
    )
    _require_registered([item.endpoint for item in constraints], endpoints, "constraint")

    search_values = dict(payload.get("search", {}))
    search = SearchSpec(**search_values)

    model_configs: dict[str, Mapping[str, Any]] = {}
    for key, definition in dict(payload.get("models", {})).items():
        values = dict(definition or {})
        for path_key in (
            "artifact",
            "manifest",
            "metrics",
            "domain_reference",
            "benchmark_root",
        ):
            if values.get(path_key):
                values[path_key] = str(_resolve_path(values[path_key], base=config_root))
        model_configs[str(key)] = MappingProxyType(values)

    edit_library = dict(payload.get("edit_library", {}))
    for path_key in ("public_pairs", "public_profiles", "private_pairs"):
        if edit_library.get(path_key):
            edit_library[path_key] = str(_resolve_path(edit_library[path_key], base=config_root))

    return MeninEditConfig(
        source_path=source,
        repository_root=repository_root,
        model_configs=MappingProxyType(model_configs),
        endpoints=MappingProxyType(endpoints),
        objectives=objectives,
        constraints=constraints,
        search=search,
        edit_library=MappingProxyType(edit_library),
        raw=MappingProxyType(payload),
    )


def _require_registered(keys: list[str], endpoints: Mapping[str, EndpointSpec], label: str) -> None:
    missing = sorted(set(keys) - set(endpoints))
    if missing:
        raise KeyError(f"Unknown {label} endpoints: {missing}")
