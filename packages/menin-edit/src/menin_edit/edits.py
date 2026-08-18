"""Observed transformation library and conservative edit application."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd

from .chemistry import (
    FragmentContext,
    fragment_single_cuts,
    join_single_attachment_fragments,
    normalize_attachment_fragment,
    validate_product,
)


def _freeze_float_mapping(values: Mapping[str, Any] | None) -> Mapping[str, float]:
    result: dict[str, float] = {}
    for key, value in dict(values or {}).items():
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number):
            result[str(key)] = number
    return MappingProxyType(result)


@dataclass(frozen=True)
class EditRule:
    rule_id: str
    source_fragment: str
    target_fragment: str
    support_count: int
    endpoint_mean_deltas: Mapping[str, float] = field(default_factory=dict)
    endpoint_std_deltas: Mapping[str, float] = field(default_factory=dict)
    endpoint_support: Mapping[str, float] = field(default_factory=dict)
    changed_heavy_atoms: int = 0
    source_scopes: tuple[str, ...] = ("public",)

    def __post_init__(self) -> None:
        object.__setattr__(self, "endpoint_mean_deltas", _freeze_float_mapping(self.endpoint_mean_deltas))
        object.__setattr__(self, "endpoint_std_deltas", _freeze_float_mapping(self.endpoint_std_deltas))
        object.__setattr__(self, "endpoint_support", _freeze_float_mapping(self.endpoint_support))
        object.__setattr__(self, "source_scopes", tuple(sorted(set(self.source_scopes))))
        if self.support_count < 1:
            raise ValueError("EditRule support_count must be positive")

    @property
    def transformation(self) -> str:
        return f"{self.source_fragment}>>{self.target_fragment}"


@dataclass(frozen=True)
class GeneratedEdit:
    rule: EditRule
    parent_smiles: str
    product_smiles: str
    context: FragmentContext
    parent_similarity: float
    heavy_atom_delta: int


@dataclass(frozen=True)
class RuleEvidence:
    rule_id: str
    endpoint: str
    observed_delta: float
    core_smiles: str
    structure_id_a: str
    structure_id_b: str
    source_scope: str
    split_role: str
    evidence_grade: str


class EditLibrary:
    """Indexed, bidirectional set of experimentally observed one-cut edits."""

    def __init__(self, rules: Iterable[EditRule], evidence: Iterable[RuleEvidence] = ()) -> None:
        self.rules = tuple(sorted(rules, key=lambda rule: rule.rule_id))
        by_source: dict[str, list[EditRule]] = defaultdict(list)
        for rule in self.rules:
            by_source[rule.source_fragment].append(rule)
        self._by_source = {
            key: tuple(sorted(values, key=lambda rule: (-rule.support_count, rule.rule_id)))
            for key, values in by_source.items()
        }
        self.evidence = tuple(evidence)
        self._evidence_by_rule: dict[str, tuple[RuleEvidence, ...]] = {}
        for rule_id, rows in _group_evidence(self.evidence).items():
            self._evidence_by_rule[rule_id] = tuple(rows)

    @classmethod
    def from_public_pairs(
        cls,
        path: str | Path,
        *,
        minimum_support: int = 1,
        allowed_split_roles: tuple[str, ...] = ("train", "development"),
    ) -> EditLibrary:
        frame = pd.read_csv(path)
        required = {
            "structure_id_a",
            "structure_id_b",
            "core_smiles",
            "variable_fragment_a",
            "variable_fragment_b",
            "p_activity_a",
            "p_activity_b",
        }
        missing = sorted(required - set(frame.columns))
        if missing:
            raise KeyError(f"Matched-pair table is missing required columns: {missing}")
        if "split_role" in frame:
            frame = frame[frame["split_role"].fillna("").astype(str).isin(allowed_split_roles)].copy()

        raw: list[dict[str, Any]] = []
        evidence_rows: list[RuleEvidence] = []
        for row in frame.to_dict(orient="records"):
            try:
                fragment_a = normalize_attachment_fragment(str(row["variable_fragment_a"]))
                fragment_b = normalize_attachment_fragment(str(row["variable_fragment_b"]))
                delta = float(row["p_activity_b"]) - float(row["p_activity_a"])
            except (TypeError, ValueError):
                continue
            if not np.isfinite(delta) or fragment_a == fragment_b:
                continue
            split_role = str(row.get("split_role", "train") or "train")
            source_scope = str(row.get("source_scope", "public") or "public")
            grade = str(row.get("evidence_context_grade", "cross_context") or "cross_context")
            for source, target, signed_delta, left_id, right_id in (
                (fragment_a, fragment_b, delta, row["structure_id_a"], row["structure_id_b"]),
                (fragment_b, fragment_a, -delta, row["structure_id_b"], row["structure_id_a"]),
            ):
                rule_id = _rule_id(source, target)
                raw.append(
                    {
                        "rule_id": rule_id,
                        "source_fragment": source,
                        "target_fragment": target,
                        "endpoint": "menin_biochemical_pIC50",
                        "delta": signed_delta,
                        "source_scope": source_scope,
                        "pair_key": f"{left_id}\0{right_id}",
                    }
                )
                evidence_rows.append(
                    RuleEvidence(
                        rule_id=rule_id,
                        endpoint="menin_biochemical_pIC50",
                        observed_delta=float(signed_delta),
                        core_smiles=str(row["core_smiles"]),
                        structure_id_a=str(left_id),
                        structure_id_b=str(right_id),
                        source_scope=source_scope,
                        split_role=split_role,
                        evidence_grade=grade,
                    )
                )
        return cls._from_long_records(raw, evidence_rows, minimum_support=minimum_support)

    @classmethod
    def from_long_pairs(
        cls,
        frame: pd.DataFrame,
        *,
        minimum_support: int = 1,
        allowed_split_roles: tuple[str, ...] = ("train", "development"),
    ) -> EditLibrary:
        """Build a vector-valued library from an already governed long pair table."""

        required = {
            "source_fragment",
            "target_fragment",
            "endpoint",
            "delta",
            "structure_id_a",
            "structure_id_b",
            "core_smiles",
        }
        missing = sorted(required - set(frame.columns))
        if missing:
            raise KeyError(f"Long pair table is missing required columns: {missing}")
        data = frame.copy()
        if "split_role" in data:
            data = data[data["split_role"].fillna("").astype(str).isin(allowed_split_roles)]
        raw: list[dict[str, Any]] = []
        evidence: list[RuleEvidence] = []
        for row in data.to_dict(orient="records"):
            try:
                source = normalize_attachment_fragment(str(row["source_fragment"]))
                target = normalize_attachment_fragment(str(row["target_fragment"]))
                delta = float(row["delta"])
            except (TypeError, ValueError):
                continue
            if not np.isfinite(delta) or source == target:
                continue
            rule_id = _rule_id(source, target)
            raw.append(
                {
                    "rule_id": rule_id,
                    "source_fragment": source,
                    "target_fragment": target,
                    "endpoint": str(row["endpoint"]),
                    "delta": delta,
                    "source_scope": str(row.get("source_scope", "private")),
                    "pair_key": (f"{row['structure_id_a']}\0{row['structure_id_b']}"),
                }
            )
            evidence.append(
                RuleEvidence(
                    rule_id=rule_id,
                    endpoint=str(row["endpoint"]),
                    observed_delta=delta,
                    core_smiles=str(row["core_smiles"]),
                    structure_id_a=str(row["structure_id_a"]),
                    structure_id_b=str(row["structure_id_b"]),
                    source_scope=str(row.get("source_scope", "private")),
                    split_role=str(row.get("split_role", "development")),
                    evidence_grade=str(row.get("evidence_grade", "same_series")),
                )
            )
        return cls._from_long_records(raw, evidence, minimum_support=minimum_support)

    @classmethod
    def merge(
        cls,
        *libraries: EditLibrary,
        minimum_support: int = 1,
    ) -> EditLibrary:
        """Pool public and governed private evidence in one rule namespace.

        The merge is evidence-based rather than a blind overwrite: endpoint
        means, dispersions, and support are recomputed from the underlying
        observed deltas.  This lets the historical lab series supply local SAR
        while retaining every public provenance record.
        """

        evidence: list[RuleEvidence] = []
        raw: list[dict[str, Any]] = []
        for library in libraries:
            rules_by_id = {rule.rule_id: rule for rule in library.rules}
            for row in library.evidence:
                rule = rules_by_id.get(row.rule_id)
                if rule is None:
                    continue
                evidence.append(row)
                raw.append(
                    {
                        "rule_id": row.rule_id,
                        "source_fragment": rule.source_fragment,
                        "target_fragment": rule.target_fragment,
                        "endpoint": row.endpoint,
                        "delta": row.observed_delta,
                        "source_scope": row.source_scope,
                        "pair_key": f"{row.structure_id_a}\0{row.structure_id_b}",
                    }
                )
        return cls._from_long_records(raw, evidence, minimum_support=minimum_support)

    @classmethod
    def _from_long_records(
        cls,
        raw: list[dict[str, Any]],
        evidence: list[RuleEvidence],
        *,
        minimum_support: int,
    ) -> EditLibrary:
        if not raw:
            return cls(())
        table = pd.DataFrame(raw)
        rules: list[EditRule] = []
        for (rule_id, source, target), group in table.groupby(
            ["rule_id", "source_fragment", "target_fragment"], sort=True
        ):
            pair_support = int(
                group["pair_key"].nunique() if "pair_key" in group else group.groupby("endpoint").size().max()
            )
            if pair_support < minimum_support:
                continue
            means = group.groupby("endpoint")["delta"].mean().to_dict()
            stds = group.groupby("endpoint")["delta"].std(ddof=0).fillna(0.0).to_dict()
            support = group.groupby("endpoint")["delta"].size().astype(float).to_dict()
            source_heavy = _fragment_heavy_atoms(source)
            target_heavy = _fragment_heavy_atoms(target)
            rules.append(
                EditRule(
                    rule_id=str(rule_id),
                    source_fragment=str(source),
                    target_fragment=str(target),
                    support_count=pair_support,
                    endpoint_mean_deltas=means,
                    endpoint_std_deltas=stds,
                    endpoint_support=support,
                    changed_heavy_atoms=max(source_heavy, target_heavy),
                    source_scopes=tuple(group["source_scope"].astype(str).unique()),
                )
            )
        retained = {rule.rule_id for rule in rules}
        return cls(rules, (row for row in evidence if row.rule_id in retained))

    def rules_for(self, source_fragment: str) -> tuple[EditRule, ...]:
        try:
            normalized = normalize_attachment_fragment(source_fragment)
        except ValueError:
            return ()
        return self._by_source.get(normalized, ())

    def evidence_for(self, rule_id: str) -> tuple[RuleEvidence, ...]:
        return self._evidence_by_rule.get(rule_id, ())

    def enumerate(
        self,
        parent_smiles: str,
        *,
        min_core_heavy_atoms: int,
        max_changed_heavy_atoms: int,
        min_parent_similarity: float,
        candidates_per_node: int,
        visited_smiles: set[str] | None = None,
    ) -> list[GeneratedEdit]:
        visited = visited_smiles or set()
        generated: dict[str, GeneratedEdit] = {}
        for context in fragment_single_cuts(
            parent_smiles,
            min_core_heavy_atoms=min_core_heavy_atoms,
            max_variable_heavy_atoms=max_changed_heavy_atoms,
        ):
            for rule in self.rules_for(context.variable_smiles):
                if rule.changed_heavy_atoms > max_changed_heavy_atoms:
                    continue
                try:
                    product = join_single_attachment_fragments(context.core_smiles, rule.target_fragment)
                except (ValueError, RuntimeError):
                    continue
                validation = validate_product(
                    parent_smiles,
                    product,
                    max_changed_heavy_atoms=max_changed_heavy_atoms,
                    min_parent_similarity=min_parent_similarity,
                )
                if not validation.valid or validation.canonical_smiles in visited:
                    continue
                candidate = GeneratedEdit(
                    rule=rule,
                    parent_smiles=parent_smiles,
                    product_smiles=validation.canonical_smiles,
                    context=context,
                    parent_similarity=validation.parent_similarity,
                    heavy_atom_delta=validation.heavy_atom_delta,
                )
                existing = generated.get(validation.canonical_smiles)
                if existing is None or rule.support_count > existing.rule.support_count:
                    generated[validation.canonical_smiles] = candidate
        ordered = sorted(
            generated.values(),
            key=lambda item: (-item.rule.support_count, -item.parent_similarity, item.product_smiles),
        )
        return ordered[:candidates_per_node]

    def manifest(self) -> dict[str, Any]:
        endpoints = sorted({key for rule in self.rules for key in rule.endpoint_mean_deltas})
        return {
            "rule_count": len(self.rules),
            "evidence_count": len(self.evidence),
            "endpoint_count": len(endpoints),
            "endpoints": endpoints,
            "bidirectional": True,
            "edit_scope": "single_cut_observed_transformations",
        }


def _fragment_heavy_atoms(fragment: str) -> int:
    from rdkit import Chem

    molecule = Chem.MolFromSmiles(fragment)
    if molecule is None:
        return 0
    return int(sum(atom.GetAtomicNum() > 1 for atom in molecule.GetAtoms()))


def _rule_id(source: str, target: str) -> str:
    digest = hashlib.sha256(f"{source}\0{target}".encode()).hexdigest()[:20]
    return f"EDIT-{digest.upper()}"


def _group_evidence(
    evidence: Iterable[RuleEvidence],
) -> dict[str, list[RuleEvidence]]:
    grouped: dict[str, list[RuleEvidence]] = defaultdict(list)
    for row in evidence:
        grouped[row.rule_id].append(row)
    grade_order = {"same_assay": 0, "same_series": 1, "same_document": 2, "cross_context": 3}
    for rows in grouped.values():
        rows.sort(
            key=lambda row: (
                grade_order.get(row.evidence_grade, 9),
                -abs(row.observed_delta),
                row.structure_id_a,
                row.structure_id_b,
            )
        )
    return grouped
