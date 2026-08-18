"""Human-readable, evidence-grounded explanations for stepwise edit paths."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .evidence import EvidenceIndex
from .schemas import CandidateNode, CandidateRanking, EndpointSpec, OptimizationResult


def explain_path(
    path: Sequence[CandidateNode],
    *,
    endpoints: Mapping[str, EndpointSpec],
    evidence: EvidenceIndex,
) -> dict[str, Any]:
    if not path:
        raise ValueError("Cannot explain an empty path")
    steps: list[dict[str, Any]] = []
    for parent, child in zip(path[:-1], path[1:], strict=True):
        endpoint_rows: list[dict[str, Any]] = []
        for endpoint, effect in child.effects.items():
            spec = endpoints.get(endpoint)
            if spec is None:
                continue
            support = evidence.lookup(
                child.edit.rule_id if child.edit else "",
                endpoint,
                top_k=5,
            )
            endpoint_rows.append(
                {
                    "endpoint": endpoint,
                    "direction": spec.direction,
                    "before": parent.predictions[endpoint].mean,
                    "after": child.predictions[endpoint].mean,
                    "predicted_conditional_delta": effect.raw_delta,
                    "desirability_delta": effect.utility_delta,
                    "delta_interval": [effect.lower, effect.upper],
                    "observed_transform_delta": effect.direct_delta,
                    "model_evidence_disagreement": effect.disagreement,
                    "inside_domain": child.predictions[endpoint].inside_domain,
                    "evidence_grade": support.evidence_grade,
                    "evidence_support": support.support_count,
                    "mean_observed_delta": support.mean_observed_delta,
                    "evidence_records": [
                        {
                            "source_scope": row.source_scope,
                            "evidence_grade": row.evidence_grade,
                            "observed_delta": row.observed_delta,
                            "structure_id_a": row.structure_id_a,
                            "structure_id_b": row.structure_id_b,
                        }
                        for row in support.records
                    ],
                }
            )
        steps.append(
            {
                "step": child.depth,
                "node_id": child.node_id,
                "parent_node_id": parent.node_id,
                "parent_smiles": parent.smiles,
                "product_smiles": child.smiles,
                "rule_id": child.edit.rule_id if child.edit else None,
                "transformation": child.edit.transformation if child.edit else None,
                "edit_type": child.edit.edit_type if child.edit else None,
                "support_count": child.edit.support_count if child.edit else 0,
                "context_similarity": child.edit.context_similarity if child.edit else None,
                "feasible_stopping_point": child.feasible,
                "constraint_evaluations": [
                    evaluation.to_dict() for evaluation in child.constraint_evaluations
                ],
                "violations": list(child.violations),
                "warnings": list(child.warnings),
                "endpoint_effects": endpoint_rows,
            }
        )
    total_changes: dict[str, float] = {}
    for endpoint in endpoints:
        if endpoint in path[0].predictions and endpoint in path[-1].predictions:
            total_changes[endpoint] = path[-1].predictions[endpoint].mean - path[0].predictions[endpoint].mean
    telescoping_checks = {
        endpoint: {
            "final_minus_start": delta,
            "sum_of_step_deltas": sum(
                step["predicted_conditional_delta"]
                for record in steps
                for step in record["endpoint_effects"]
                if step["endpoint"] == endpoint
            ),
        }
        for endpoint, delta in total_changes.items()
    }
    return {
        "starting_node_id": path[0].node_id,
        "final_node_id": path[-1].node_id,
        "number_of_edits": len(path) - 1,
        "can_stop_after_any_step": True,
        "effect_language": "predicted conditional effect, not experimentally established causality",
        "steps": steps,
        "total_changes": total_changes,
        "telescoping_checks": telescoping_checks,
    }


def render_markdown_report(
    result: OptimizationResult,
    *,
    paths: Mapping[str, Sequence[CandidateNode]],
    endpoints: Mapping[str, EndpointSpec],
    evidence: EvidenceIndex,
    top_k: int = 10,
) -> str:
    selected = [ranking for ranking in result.rankings if ranking.eligible][:top_k]
    lines = [
        "# Menin-Edit optimization report",
        "",
        f"- Session: `{result.session_id}`",
        f"- Starting molecule: `{result.baseline.smiles}`",
        f"- Generated candidates: {len(result.candidates)}",
        f"- Feasible candidates: {sum(ranking.eligible for ranking in result.rankings)}",
        "- Interpretation: computational design hypotheses; not experimental potency or safety claims.",
        "",
        "## Ranked stopping points",
        "",
        "| Order | Node | Step | Pareto rank | Priority score | Feasible |",
        "|---:|---|---:|---:|---:|---|",
    ]
    candidates = {candidate.node_id: candidate for candidate in result.candidates}
    for ranking in selected:
        candidate = candidates[ranking.node_id]
        score = "" if ranking.priority_score is None else f"{ranking.priority_score:.3f}"
        lines.append(
            f"| {ranking.order} | `{ranking.node_id}` | {candidate.depth} | "
            f"{ranking.pareto_rank or ''} | {score} | yes |"
        )
    if not selected:
        lines.append("| — | — | — | — | — | no feasible candidates |")

    for ranking in selected[: min(5, len(selected))]:
        candidate = candidates[ranking.node_id]
        explanation = explain_path(paths[candidate.node_id], endpoints=endpoints, evidence=evidence)
        lines.extend(["", f"## Path to {candidate.node_id}", ""])
        for step in explanation["steps"]:
            lines.append(f"### Step {step['step']}: `{step['transformation']}`")
            lines.append("")
            lines.append(
                f"Supported by {step['support_count']} training pair(s); "
                f"context similarity {step['context_similarity']:.3f}."
            )
            lines.append("")
            lines.append(
                "| Endpoint | Before | After | Model delta | Desirability delta | "
                "Observed MMP delta | Disagreement | Evidence |"
            )
            lines.append("|---|---:|---:|---:|---:|---:|---:|---|")
            for effect in step["endpoint_effects"]:
                observed = effect["observed_transform_delta"]
                disagreement = effect["model_evidence_disagreement"]
                observed_text = "—" if observed is None else f"{observed:+.3f}"
                disagreement_text = "—" if disagreement is None else f"{disagreement:.3f}"
                lines.append(
                    f"| {effect['endpoint']} | {effect['before']:.3f} | {effect['after']:.3f} | "
                    f"{effect['predicted_conditional_delta']:+.3f} | "
                    f"{effect['desirability_delta']:+.3f} | "
                    f"{observed_text} | {disagreement_text} | "
                    f"{effect['evidence_grade']} (n={effect['evidence_support']}) |"
                )
            if step["violations"]:
                lines.append("")
                lines.append("Hard-bound violations: " + "; ".join(step["violations"]))
            if step["warnings"]:
                lines.append("")
                lines.append("Hard-bound warnings: " + "; ".join(step["warnings"]))
            if step["constraint_evaluations"]:
                lines.append("")
                lines.append("| Bound endpoint | Test | Conservative value | Bound | Margin | Status |")
                lines.append("|---|---|---:|---:|---:|---|")
                for evaluation in step["constraint_evaluations"]:
                    conservative = evaluation["conservative_value"]
                    margin = evaluation["margin"]
                    conservative_text = "—" if conservative is None else f"{conservative:.3f}"
                    margin_text = "—" if margin is None else f"{margin:+.3f}"
                    lines.append(
                        f"| {evaluation['endpoint']} | {evaluation['operator']} | "
                        f"{conservative_text} | {evaluation['bound']:.3f} | "
                        f"{margin_text} | {evaluation['status']} |"
                    )
    lines.extend(
        [
            "",
            "## Model and evidence boundary",
            "",
            "Every complex proposal is decomposed into single edits. Reported contributions are model-predicted conditional differences and telescope from the starting structure to the selected stopping point. They are not causal or experimentally validated effects.",
        ]
    )
    return "\n".join(lines) + "\n"


def ranking_lookup(rankings: Sequence[CandidateRanking]) -> dict[str, CandidateRanking]:
    return {ranking.node_id: ranking for ranking in rankings}
