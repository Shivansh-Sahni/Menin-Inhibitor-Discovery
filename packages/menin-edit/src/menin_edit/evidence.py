"""Evidence retrieval for explaining edit recommendations."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from .edits import EditLibrary, RuleEvidence


@dataclass(frozen=True)
class EvidenceSummary:
    rule_id: str
    endpoint: str
    support_count: int
    mean_observed_delta: float | None
    evidence_grade: str
    records: tuple[RuleEvidence, ...]


class EvidenceIndex:
    def __init__(self, library: EditLibrary) -> None:
        self.library = library

    def lookup(
        self,
        rule_id: str,
        endpoint: str,
        *,
        query_core_smiles: str | None = None,
        top_k: int = 5,
        allowed_split_roles: Iterable[str] = ("train", "development"),
    ) -> EvidenceSummary:
        allowed = set(allowed_split_roles)
        rows = [
            row
            for row in self.library.evidence_for(rule_id)
            if row.endpoint == endpoint and row.split_role in allowed
        ]
        rows.sort(
            key=lambda row: (
                0 if query_core_smiles and row.core_smiles == query_core_smiles else 1,
                0 if row.evidence_grade in {"same_assay", "same_series"} else 1,
                -abs(row.observed_delta),
                row.structure_id_a,
            )
        )
        selected = tuple(rows[:top_k])
        mean = sum(row.observed_delta for row in rows) / len(rows) if rows else None
        if selected and query_core_smiles and selected[0].core_smiles == query_core_smiles:
            grade = "exact_context"
        elif selected:
            grade = "exact_transformation"
        else:
            grade = "model_only"
        return EvidenceSummary(
            rule_id=rule_id,
            endpoint=endpoint,
            support_count=len(rows),
            mean_observed_delta=mean,
            evidence_grade=grade,
            records=selected,
        )
