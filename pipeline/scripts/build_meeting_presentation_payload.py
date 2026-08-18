"""Assemble a compact, source-linked payload for the PK/hERG meeting workbook."""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MIX = ROOT / "research/reports/pk_herg/mix_match"
EXAMPLE = ROOT / "research/reports/pk_herg/new_compound_predictions/integrated_example"
OUTPUT = MIX / "meeting_presentation_payload.json"


def read_csv(path: Path) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def numeric(records: list[dict[str, object]]) -> list[dict[str, object]]:
    """Convert numeric-looking CSV fields while preserving identifiers and text."""

    converted: list[dict[str, object]] = []
    for record in records:
        row: dict[str, object] = {}
        for key, value in record.items():
            if value is None or value == "":
                row[key] = None
                continue
            if value in {"True", "False"}:
                row[key] = value == "True"
                continue
            try:
                row[key] = float(value)
            except (TypeError, ValueError):
                row[key] = value
        converted.append(row)
    return converted


with (MIX / "professor_workbook_payload.json").open(encoding="utf-8") as handle:
    source = json.load(handle)

pk_all = numeric(read_csv(MIX / "pk_feature_model_summary.csv"))
pk_selection_keys = {
    ("iv_auc_dose_normalized", "morgan_tanimoto", "tanimoto_3nn"),
    ("po_auc_dose_normalized", "morgan_tanimoto", "tanimoto_3nn"),
    ("po_cmax_dose_normalized", "morgan_tanimoto", "tanimoto_3nn"),
    ("po_tmax", "compact_proxies", "random_forest"),
    ("vdss", "hybrid_pka", "extra_trees"),
}
pk_selected = [
    row for row in pk_all if (row["endpoint"], row["feature_layer"], row["model"]) in pk_selection_keys
]

cliffs = numeric(read_csv(MIX / "same_series_analogue_cliff_candidates.csv"))
cliffs.sort(key=lambda row: float(row["observed_absolute_delta"]), reverse=True)
cliffs_selected = cliffs[:8]

permutation = numeric(read_csv(MIX / "permutation_test_summary.csv"))
example_integrated = numeric(read_csv(EXAMPLE / "integrated_summary.csv"))
example_herg = numeric(read_csv(EXAMPLE / "herg/predictions.csv"))
example_pk = numeric(read_csv(EXAMPLE / "pk/pk_endpoint_summary.csv"))
example_derived = numeric(read_csv(EXAMPLE / "pk/pk_derived_closure.csv"))

dataset_counts = {row["dataset"]: row for row in source["dataset_audit"]}

summary_kpis = [
    {
        "metric": "Internal continuous hERG anchor",
        "value": 0.3561648066884147,
        "unit": "pIC50 MAE",
        "interpretation": "Reproducible held-scaffold signal; about 2.27-fold IC50 error on the concentration scale.",
    },
    {
        "metric": "Internal decisive-class hERG anchor",
        "value": 0.147125247137572,
        "unit": "Brier score",
        "interpretation": "Better than shuffled outcomes, but nonblocker specificity is unstable; discovery use only.",
    },
    {
        "metric": "Rat PK best endpoint",
        "value": 1.2656719556171088,
        "unit": "median fold error (Vdss)",
        "interpretation": "Useful same-series hypothesis generation from summary PK; not calibrated PBPK.",
    },
    {
        "metric": "Anchor permutation evidence",
        "value": 0.00796812749003984,
        "unit": "empirical p",
        "interpretation": "Both selected anchors beat 250 shuffled-label controls.",
    },
]

evidence_counts = [
    {
        "evidence": "Internal exact hERG",
        "structures": dataset_counts["internal_hERG_exact_compound_collapsed"]["structures"],
        "scaffolds": dataset_counts["internal_hERG_exact_compound_collapsed"]["scaffolds"],
        "meeting_use": "Required continuous anchor and scaffold-grouped evaluation",
    },
    {
        "evidence": "Internal interval hERG",
        "structures": dataset_counts["internal_hERG_interval_compound_collapsed"]["structures"],
        "scaffolds": dataset_counts["internal_hERG_interval_compound_collapsed"]["scaffolds"],
        "meeting_use": "Censoring, >30 uM handling, and conflict-aware classification",
    },
    {
        "evidence": "Angelo same-series exact extension",
        "structures": dataset_counts["Angelo_Ascentage_nonoverlap_exact"]["structures"],
        "scaffolds": dataset_counts["Angelo_Ascentage_nonoverlap_exact"]["scaffolds"],
        "meeting_use": "Retrospective new-candidate stress test; protocol metadata still needed",
    },
    {
        "evidence": "Angelo same-series all measured",
        "structures": dataset_counts["Angelo_Ascentage_nonoverlap_all_measured"]["structures"],
        "scaffolds": dataset_counts["Angelo_Ascentage_nonoverlap_all_measured"]["scaffolds"],
        "meeting_use": "Includes four >30 uM nonblocker limits and five ambiguous/intermediate rows",
    },
    {
        "evidence": "Public hERG training source",
        "structures": dataset_counts["Sun_public_regression_train"]["structures"],
        "scaffolds": dataset_counts["Sun_public_regression_train"]["scaffolds"],
        "meeting_use": "External challenger/pretraining only; heterogeneous assays",
    },
    {
        "evidence": "Internal rat PK",
        "structures": 46,
        "scaffolds": 15,
        "meeting_use": "Summary-parameter IV/PO models; no concentration-time profiles",
    },
]

hypothesis_controls = [
    {
        "question": "Is continuous hERG signal stronger than chance?",
        "evidence": "Observed MAE 0.356 versus shuffled median 0.446; empirical p=0.008.",
        "status": "Supported retrospectively",
        "falsifier": "Failure on a frozen, protocol-matched incoming Menin series.",
    },
    {
        "question": "Is decisive-class hERG reliable for both classes?",
        "evidence": "Brier 0.147 beats shuffled median 0.214, but repeated splits give median specificity 0.",
        "status": "Partially supported; discovery only",
        "falsifier": "Prospective sensitivity and specificity outside uncertainty targets.",
    },
    {
        "question": "Does broad public hERG data improve Menin predictions?",
        "evidence": "Selected public challenger is slightly worse than the internal anchor on internal scaffold CV (MAE +0.009; Brier +0.010).",
        "status": "No consistent gain",
        "falsifier": "Protocol-aware transfer that improves a frozen external Menin test series.",
    },
    {
        "question": "Do current representations explain activity cliffs?",
        "evidence": "22 high-similarity cliff candidates; all 7 directly comparable cases are under-resolved.",
        "status": "No; hidden state/path variables are plausible",
        "falsifier": "Protocol replication shows cliffs are assay noise or a controlled edit model resolves them.",
    },
]

next_actions = [
    {
        "priority": 1,
        "action": "Freeze the incoming multi-series Menin hERG set before revealing outcomes.",
        "evidence_produced": "Prospective same-series and leave-series-out generalization with honest calibration.",
        "dependency": "Dr. Aguilar data plus assay protocol metadata",
        "promotion_gate": "Calibrated pIC50/interval predictions and usable sensitivity/specificity across series.",
    },
    {
        "priority": 2,
        "action": "Run protocol-matched concentration-response hERG on selected blockers, nonblockers, intermediates, and cliff pairs.",
        "evidence_produced": "Resolves censoring, source conflicts, free-concentration effects, and activity-cliff reality.",
        "dependency": "Experimental assay access",
        "promotion_gate": "Replicate direction and uncertainty-compatible effect sizes.",
    },
    {
        "priority": 3,
        "action": "Use HPC on a small matched-pair panel for environment-dependent folding, membrane access, and 3-6 prepared hERG receptor states.",
        "evidence_produced": "Tests whether state populations and transport/binding pathways explain under-resolved cliffs.",
        "dependency": "HPC access and converged microstate/conformer preparation",
        "promotion_gate": "Replica convergence, falsifiable pairwise direction, and added value beyond current proxies.",
    },
    {
        "priority": 4,
        "action": "Obtain complete rat IV/PO concentration-time profiles and study metadata for a designed compound panel.",
        "evidence_produced": "Separates absorption, first-pass loss, distribution, and clearance instead of fitting algebraically linked summaries.",
        "dependency": "New/internal raw PK data",
        "promotion_gate": "Profile-level predictive checks and uncertainty calibration.",
    },
    {
        "priority": 5,
        "action": "Connect promoted endpoints to Wang's AI workflow as uncertainty-aware filters before any structure generation or ranking.",
        "evidence_produced": "A defensible structure-to-property loop with domain flags and explicit required-data flags.",
        "dependency": "Prospective hERG promotion; PK remains secondary",
        "promotion_gate": "No optimization outside validated domain and no scalar score hiding uncertainty.",
    },
]

meeting_decisions = [
    {
        "decision": "Primary near-term endpoint",
        "recommended_answer": "hERG first; PK remains a parallel hypothesis model until profile data arrive.",
        "why_now": "The current hERG evidence base is larger and the incoming data directly tests generalization.",
    },
    {
        "decision": "AI workflow scope",
        "recommended_answer": "Start by scoring/refining candidate structures; add generative design only after prospective model promotion.",
        "why_now": "Generation can amplify model error if the property models are not calibrated outside the current series.",
    },
    {
        "decision": "Immediate HPC experiment",
        "recommended_answer": "Matched-pair folding/membrane-access study, followed by receptor-state hERG interaction tests.",
        "why_now": "The analogue cliffs identify where 2D structure and conventional proxies fail and where physics has the highest information value.",
    },
]

payload = {
    "generated_for": "Dr. Shaomeng Wang and Dr. Angelo Aguilar meeting",
    "scope": "Menin-inhibitor hERG liability first; rat IV/PO PK second",
    "summary_kpis": summary_kpis,
    "evidence_counts": evidence_counts,
    "dataset_audit": source["dataset_audit"],
    "binary_label_summary": source["binary_label_summary"],
    "feature_contract": source["feature_contract"],
    "key_results": source["key_results"],
    "permutation_summary": permutation,
    "hypothesis_controls": hypothesis_controls,
    "cliffs_selected": cliffs_selected,
    "pk_selected": pk_selected,
    "example_integrated": example_integrated,
    "example_herg": example_herg,
    "example_pk": example_pk,
    "example_derived": example_derived,
    "next_actions": next_actions,
    "meeting_decisions": meeting_decisions,
    "model_summary_count": (
        len(source["continuous_summary"]) + len(source["binary_summary"]) + len(source["pk_summary"])
    ),
    "analogue_cliff_count": len(source["analogue_cliffs"]),
}

with OUTPUT.open("w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")

print(OUTPUT)
