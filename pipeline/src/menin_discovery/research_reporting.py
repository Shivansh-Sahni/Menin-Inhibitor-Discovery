"""Causal explanation contracts and status reporting for PK/hERG research."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from .research_common import atomic_write_json, atomic_write_text
from .research_feature_ontology import (
    CONVENTIONAL_DESCRIPTOR_COLUMNS,
    MODEL_CONFORMER_FEATURES,
    MODEL_PHYSICS_FEATURES,
)

EXPLANATION_FIELDS = (
    "dataset_definition",
    "split_definition",
    "metrics_with_uncertainty",
    "calibration",
    "applicability_domain",
    "residual_clusters",
    "feature_layer_ablations",
    "matched_pair_examples",
    "proposed_physical_explanation",
    "competing_explanations_and_confounders",
    "falsifying_simulation_or_assay",
    "promotion_status",
)


def residual_process_clusters(
    predictions: pd.DataFrame,
    mechanism_features: pd.DataFrame,
    *,
    observed_column: str,
    predicted_column: str,
    maximum_clusters: int = 4,
    random_state: int = 20260721,
) -> pd.DataFrame:
    """Find reproducible failure strata; clusters are hypotheses, not causal proof."""

    frame = predictions.merge(mechanism_features, on="compound_id", how="left")
    frame["residual"] = frame[observed_column] - frame[predicted_column]
    numeric = [
        column
        for column in mechanism_features.select_dtypes(include=[np.number]).columns
        if column != "compound_id"
    ]
    if len(frame) < 4 or not numeric:
        frame["residual_cluster"] = 0
        return frame[["compound_id", "residual", "residual_cluster"]]
    values = frame[numeric].replace([np.inf, -np.inf], np.nan)
    values = values.fillna(values.median()).fillna(0.0)
    matrix = StandardScaler().fit_transform(values)
    clusters = min(maximum_clusters, max(2, len(frame) // 8), len(frame))
    frame["residual_cluster"] = KMeans(n_clusters=clusters, n_init=20, random_state=random_state).fit_predict(
        matrix
    )
    columns = ["compound_id", "residual", "residual_cluster", *numeric]
    return frame[columns]


def validate_explanation_contract(payload: dict[str, Any]) -> None:
    missing = [field for field in EXPLANATION_FIELDS if field not in payload]
    if missing:
        raise ValueError(f"Explanation contract is incomplete: {missing}")
    status = payload["promotion_status"]
    if status not in {"decision-track", "discovery-track", "rejected"}:
        raise ValueError(f"Invalid promotion status: {status!r}")


def write_explanation_contract(path: Path, payload: dict[str, Any]) -> Path:
    validate_explanation_contract(payload)
    return atomic_write_json(path, payload)


def _markdown_table(frame: pd.DataFrame, columns: list[str] | None = None) -> str:
    if frame.empty:
        return "_No rows available._"
    selected = frame[columns] if columns else frame
    return selected.to_markdown(index=False)


def _present(value: Any) -> bool:
    if value is None:
        return False
    try:
        return bool(pd.notna(value))
    except (TypeError, ValueError):
        return True


def _format_value(value: Any) -> str:
    if not _present(value):
        return "—"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.3f}"
    return str(value)


def _promotion_status(row: pd.Series) -> str:
    explicit = row.get("promotion_status")
    if _present(explicit) and str(explicit) in {"decision-track", "discovery-track", "rejected"}:
        return str(explicit)
    status = str(row.get("status", "")).casefold()
    if "rejected" in status or "nonconverged" in status:
        return "rejected"
    # An evaluated model is not decision-ready merely because training completed.
    return "discovery-track"


def compact_model_evidence(model_summary: pd.DataFrame) -> pd.DataFrame:
    """Normalize heterogeneous PK/hERG ladders into a readable scientific table."""

    columns = [
        "domain",
        "endpoint",
        "feature_layer",
        "model",
        "n",
        "primary_evidence",
        "calibration_or_domain",
        "promotion",
        "gate_or_failure",
    ]
    if model_summary.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    for _, row in model_summary.iterrows():
        domain = str(row.get("domain", "")).strip().casefold()
        if not domain or domain == "nan":
            domain = "herg"
        endpoint = row.get("endpoint")
        if not _present(endpoint):
            endpoint = "continuous_hERG"
        model = row.get("best_model")
        if not _present(model):
            model = row.get("model", "not available")
        metric = row.get("primary_metric")
        value = row.get("primary_value")
        if not _present(metric) or not _present(value):
            if _present(row.get("pic50_mae")):
                metric, value = "pIC50 MAE", row.get("pic50_mae")
            elif _present(row.get("roc_auc")):
                metric, value = "ROC-AUC", row.get("roc_auc")
            elif _present(row.get("classification_roc_auc")):
                metric, value = "ROC-AUC", row.get("classification_roc_auc")
            else:
                metric, value = "not available", None
        n_value = row.get("n_unique_compounds")
        if not _present(n_value):
            n_value = row.get("n")
        if not _present(n_value):
            n_value = row.get("classification_n")
        if not _present(n_value):
            n_value = row.get("n_exact")

        calibration_parts: list[str] = []
        if _present(row.get("prediction_interval_coverage")):
            calibration_parts.append(f"PI coverage {_format_value(row.get('prediction_interval_coverage'))}")
        elif _present(row.get("interval_coverage")):
            calibration_parts.append(f"PI coverage {_format_value(row.get('interval_coverage'))}")
        if _present(row.get("classification_brier")):
            calibration_parts.append(f"Brier {_format_value(row.get('classification_brier'))}")
        if _present(row.get("classification_ece_8bin")):
            calibration_parts.append(f"ECE {_format_value(row.get('classification_ece_8bin'))}")
        if _present(row.get("inside_domain_fraction")):
            calibration_parts.append(
                f"inside-domain fraction {_format_value(row.get('inside_domain_fraction'))}"
            )
        if not calibration_parts:
            calibration_parts.append("not demonstrated")

        failure = row.get("promotion_reason")
        if not _present(failure):
            failure = row.get("status", "not reported")
        if str(failure) == "evaluated":
            failure = "evaluation/reference row; prospective calibration gate not passed"
        if _present(row.get("fit_converged_fraction")) and float(row["fit_converged_fraction"]) < 1.0:
            failure = f"fit convergence {_format_value(row['fit_converged_fraction'])}; {failure}"
        if _present(row.get("heldout_inhibition_mae_percent")):
            failure = (
                f"held-out inhibition MAE {_format_value(row['heldout_inhibition_mae_percent'])} %-points; "
                f"{failure}"
            )
        rows.append(
            {
                "domain": domain.upper() if domain == "pk" else "hERG",
                "endpoint": str(endpoint),
                "feature_layer": str(row.get("feature_layer", "not available")),
                "model": str(model),
                "n": _format_value(n_value),
                "primary_evidence": f"{metric}={_format_value(value)}",
                "calibration_or_domain": "; ".join(calibration_parts),
                "promotion": _promotion_status(row),
                "gate_or_failure": str(failure),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def model_failure_findings(model_summary: pd.DataFrame) -> list[str]:
    """Extract consequential ablation failures rather than narrating every row."""

    if model_summary.empty:
        return ["No model ladder was available; no predictive or mechanistic conclusion is supportable."]
    findings: list[str] = []
    pk = model_summary[model_summary.get("domain", pd.Series("", index=model_summary.index)).eq("pk")]
    for _, row in pk.iterrows():
        if str(row.get("feature_layer")) != "state_conformer_physics":
            continue
        baseline = row.get("baseline_value")
        candidate = row.get("candidate_value", row.get("primary_value"))
        if _present(baseline) and _present(candidate):
            delta = float(candidate) - float(baseline)
            direction = "improved" if delta < 0 else "worsened"
            findings.append(
                f"PK {row.get('endpoint')}: the fast-physics layer {direction} log-MAE by "
                f"{abs(delta):.3f} versus the structure-only comparator "
                f"({_format_value(baseline)} to {_format_value(candidate)}); it remains discovery-track "
                "because calibration and production-physics convergence are absent."
            )
    herg = model_summary[model_summary.get("domain", pd.Series("", index=model_summary.index)).eq("herg")]
    physics = herg[herg.get("feature_layer", pd.Series("", index=herg.index)).eq("state_conformer_physics")]
    if not physics.empty:
        row = physics.iloc[0]
        findings.append(
            "hERG fast-physics fit was rejected: "
            f"pIC50 MAE={_format_value(row.get('pic50_mae'))}, "
            f"fit-converged fraction={_format_value(row.get('fit_converged_fraction'))}. "
            "Its apparent mechanistic coefficients must not be interpreted."
        )
    joint = herg[
        herg.get("feature_layer", pd.Series("", index=herg.index)).eq("structure_2d_joint_observations")
    ]
    if not joint.empty and _present(joint.iloc[0].get("heldout_inhibition_mae_percent")):
        row = joint.iloc[0]
        findings.append(
            "The joint pIC50/concentration-response model did not transfer to held-out inhibition rows "
            f"(MAE={_format_value(row.get('heldout_inhibition_mae_percent'))} and "
            f"RMSE={_format_value(row.get('heldout_inhibition_rmse_percent'))} percentage points). "
            "This falsifies a single global static Hill mapping for the present mixed protocols."
        )
    for layer, label in (("molecular_graph", "D-MPNN"), ("state_conformer_bags", "conformer MIL")):
        subset = herg[herg.get("feature_layer", pd.Series("", index=herg.index)).eq(layer)]
        if not subset.empty and _present(subset.iloc[0].get("pic50_mae")):
            row = subset.iloc[0]
            if layer == "molecular_graph":
                findings.append(
                    f"D-MPNN reached pIC50 MAE={_format_value(row.get('pic50_mae'))} but only "
                    f"ROC-AUC={_format_value(row.get('classification_roc_auc'))}; it improved on the "
                    "censored ridge's continuous error yet did not beat the exact-row ridge or establish "
                    "calibration. It remains an architectural discovery comparator."
                )
            else:
                findings.append(
                    f"{label} produced pIC50 MAE={_format_value(row.get('pic50_mae'))}, but remains "
                    "an architectural experiment until its input ensemble is converged and calibrated."
                )
    public = herg[
        herg.get("feature_layer", pd.Series("", index=herg.index)).astype(str).str.startswith("sun_public")
    ]
    if not public.empty:
        findings.append(
            "The Sun/Wang/Shen benchmark is an independent RDKit reconstruction only: paper-specific "
            "atom typing and correction-factor mapping are unavailable, and its low structural overlap "
            "precludes using public performance as internal-series validation."
        )
    return findings


def optimizer_endpoint_summary(contract: pd.DataFrame) -> pd.DataFrame:
    """Summarize optimizer readiness without creating a scalar score or rank."""

    columns = [
        "endpoint",
        "predictions",
        "inside_domain",
        "outside_domain",
        "promotion",
        "lineage_role",
        "required_data",
    ]
    if contract.empty:
        return pd.DataFrame(columns=columns)
    endpoints = sorted(column.removeprefix("mean__") for column in contract if column.startswith("mean__"))
    rows: list[dict[str, Any]] = []
    for endpoint in endpoints:
        mean = pd.to_numeric(contract[f"mean__{endpoint}"], errors="coerce")
        domain = contract.get(f"domain_status__{endpoint}", pd.Series("required_data", index=contract.index))
        promotion = contract.get(
            f"promotion_status__{endpoint}", pd.Series("required_data", index=contract.index)
        )
        role = contract.get(f"lineage_role__{endpoint}", pd.Series("required_data", index=contract.index))
        required = contract.get(f"required_data__{endpoint}", mean.isna()).fillna(True).astype(bool)
        rows.append(
            {
                "endpoint": endpoint,
                "predictions": int(mean.notna().sum()),
                "inside_domain": int(domain.astype(str).eq("inside").sum()),
                "outside_domain": int(domain.astype(str).eq("outside").sum()),
                "promotion": " | ".join(sorted(set(promotion.fillna("required_data").astype(str)))),
                "lineage_role": " | ".join(sorted(set(role.fillna("required_data").astype(str)))),
                "required_data": int(required.sum()),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def write_model_card(
    path: Path,
    *,
    title: str,
    contract: dict[str, Any],
    metrics: pd.DataFrame,
    limitations: list[str],
) -> Path:
    validate_explanation_contract(contract)
    metric_text = _markdown_table(compact_model_evidence(metrics))
    limitation_text = "\n".join(f"- {item}" for item in limitations)
    contract_metrics = contract["metrics_with_uncertainty"]
    if isinstance(contract_metrics, dict):
        contract_metric_text = "\n".join(
            f"- {key}: {_format_value(value)}" for key, value in contract_metrics.items()
        )
    else:
        contract_metric_text = str(contract_metrics)
    text = f"""# {title}

## Promotion status

**{contract["promotion_status"]}**

## Scientific contract

- Dataset: {contract["dataset_definition"]}
- Split: {contract["split_definition"]}
- Calibration: {contract["calibration"]}
- Applicability domain: {contract["applicability_domain"]}

## Metrics

{metric_text}

### Held-out metric contract

{contract_metric_text}

## Mechanistic interpretation

{contract["proposed_physical_explanation"]}

Competing explanations and confounders: {contract["competing_explanations_and_confounders"]}

Falsification test: {contract["falsifying_simulation_or_assay"]}

## Residual and ablation evidence

- Residual clusters: {contract["residual_clusters"]}
- Feature-layer ablations: {contract["feature_layer_ablations"]}
- Matched-pair examples: {contract["matched_pair_examples"]}

## Limitations

{limitation_text}
"""
    return atomic_write_text(path, text)


def write_current_status_report(
    path: Path,
    *,
    inventory: dict[str, Any],
    stage_status: pd.DataFrame,
    model_summary: pd.DataFrame,
    regime_result: dict[str, Any],
    assay_summary: dict[str, Any],
    heavy_physics_status: dict[str, Any],
    source_qc: dict[str, Any] | None = None,
    optimizer_summary: pd.DataFrame | None = None,
    failure_findings: list[str] | None = None,
    run_context: dict[str, Any] | None = None,
) -> Path:
    """Write the principal handoff report without overstating unfinished physics."""

    status_table = _markdown_table(stage_status)
    compact_models = compact_model_evidence(model_summary)
    model_table = _markdown_table(compact_models)
    failures = failure_findings if failure_findings is not None else model_failure_findings(model_summary)
    failure_text = "\n".join(f"- {finding}" for finding in failures)
    cutoff_statement = regime_result.get(
        "reason", "MW regime analysis has not produced a supported boundary."
    )
    source_qc = source_qc or {}
    optimizer_summary = optimizer_summary if optimizer_summary is not None else pd.DataFrame()
    optimizer_table = _markdown_table(optimizer_summary)
    run_context = run_context or {}
    run_mode = "smoke/diagnostic" if bool(run_context.get("smoke_mode")) else "release/local"
    sampling_tier = str(run_context.get("physics_sampling_tier", "unknown"))
    generated_per_state = run_context.get("generated_conformers_per_state")
    retained_per_state = run_context.get("retained_conformers_per_state")
    reference_per_state = run_context.get("reference_conformers_per_state")
    physics_deferred = str(run_context.get("physics_execution_status", "")) == "deferred_to_hpc"
    provisional_feature_count = len(MODEL_PHYSICS_FEATURES)
    conformer_primitive_count = len(MODEL_CONFORMER_FEATURES)
    conventional_control_count = len(CONVENTIONAL_DESCRIPTOR_COLUMNS)
    if physics_deferred:
        run_mode += "; local chemical-state/conformer physics deliberately deferred to HPC"
    if not bool(run_context.get("smoke_mode")) and sampling_tier == "local_time_bounded_discovery":
        run_mode += (
            f"; time-bounded discovery physics ({generated_per_state} candidates/state locally, "
            f"up to {retained_per_state} retained; {reference_per_state} is a deferred validation "
            "ceiling, not a fixed target)"
        )
    if physics_deferred:
        sampling_admission_text = f"""- No local chemical-state or conformer ensemble was calculated, and no structure-derived physics feature entered a PK or hERG model in this release.
- There is no fixed retained-state count target and no HPC 24-state cap. Candidate states are retained by the 0.1% working gate, 1%/0.1%/0.01% sensitivity, and explicit neutral, zwitterionic, net-charge, and H-bond-topology exceptions.
- Conformer work starts at 25 only as a diagnostic rung and escalates adaptively through 50, 100, 250, and 500 with independent seeds. A state stops only after its declared physical observables and new-cluster mass stabilize; 250 and 500 are ceilings/comparators, not constants.
- The historical 11-column count was retired after the causal/evidence audit. The current fail-closed discovery allowlist has {provisional_feature_count} provisional proxy columns and conformer MIL has {conformer_primitive_count} raw primitives; these counts may change only through the documented admission/retirement rules. Nothing is admitted until state-threshold and conformer-sampling convergence gates pass."""
    else:
        sampling_admission_text = f"""- The nominal candidate-state gate is **0.1%**, with explicit audits at **1%, 0.1%, and 0.01%**. Neutral, zwitterionic, charge-state, and H-bond-capacity representatives are preserved as mechanism exceptions. These are approximate chemical-state hypotheses, not measured equilibrium microstate populations.
- Any structure for which the local compute cap omits qualifying chemistry is model-inadmissible rather than silently renormalized. Current cap-affected/inadmissible structure count: **{run_context.get("physics_substantively_ineligible_structure_count", "NA")}**.
- Local conformer sampling generated **{generated_per_state}** candidates and retained at most **{retained_per_state}** per state. The audit contains **{run_context.get("sampling_audited_state_count", "NA")} states**, of which **{run_context.get("sampling_escalation_required_state_count", "NA")}** require escalation under the local nested-subset diagnostic.
- `250` is a selected-pilot validation ceiling and `500` is its comparator. Neither is accepted as a universal count. Adequacy requires independent seeds, stable physical means/distributions/folded and IMHB fractions, low new-cluster mass, and no systematic failure in flexible or charge-sensitive strata.
- Model inputs are fail-closed: the current ensemble model admits {provisional_feature_count} explicitly provisional proxy columns, while conformer MIL admits {conformer_primitive_count} declared raw physical primitives. These are evidence-governed counts, not scientific constants. Numeric aliases, algebraic closures, identifiers, minimization energy/rank, QC values, uncertainty spans, and unreviewed future columns cannot enter merely because they were calculated."""
    decision_count = (
        int(compact_models["promotion"].eq("decision-track").sum()) if not compact_models.empty else 0
    )
    discovery_count = (
        int(compact_models["promotion"].eq("discovery-track").sum()) if not compact_models.empty else 0
    )
    rejected_count = int(compact_models["promotion"].eq("rejected").sum()) if not compact_models.empty else 0
    text = f"""# Current status and next steps: mechanistic PK + hERG

> **Claim-control notice:** This is the operational pipeline handoff. The
> authoritative end-of-local-phase scientific assessment is
> `../reviewer_audit/critical_peer_review_and_redesigned_plan.md`, together
> with its claim and methodological-gap registers. No model below is
> decision-track, no Bemis–Murcko group is assumed to be an independent
> medicinal series, and no optimizer output is qualified for use.

## Scope and decision boundary

This program is separate from the frozen Menin/Menin-Edit baseline. It targets the {inventory.get("n_unique_compounds", "current")} unique internal large-molecule structures, rat IV/PO exposure, and hERG liability. It does not rank or generate molecules. Production MD and free-energy calculations were not launched locally.

**Run context:** {run_mode}. Smoke/diagnostic results, when present, establish executable interfaces and failure modes; they are not release evidence.

Two tracks are reported independently:

- **Decision track:** calibrated, group-held-out models that pass applicability and non-inferiority gates.
- **Mechanistic discovery track:** physics or causal decompositions retained because they create testable explanations, even when prediction metrics fall.

## Data reconstructed

- Internal structures: **{inventory.get("n_unique_compounds", "not available")}** unique; MW range **{inventory.get("mw_min", "NA")}–{inventory.get("mw_max", "NA")} Da**.
- Rat PK compounds with any summary endpoint: **{inventory.get("n_rat_pk", "NA")}**.
- hERG compounds with IC50 or inhibition evidence: **{inventory.get("n_herg", "NA")}**.
- Raw rat concentration-time samples: **{inventory.get("n_pk_samples", 0)}**; PBPK therefore remains sensitivity-only.
- Raw PO Cmax is retired because 3- and 5-mg/kg records are compound-confounded. The modeled dose-normalized Cmax replacement is discovery-only pending within-compound proportionality.
- The reported CL and F values are marked derived and are excluded as independent parents/labels when their source dose/AUC values are used.
- One-sided hERG limits and concentration-specific inhibition observations remain censored measurements rather than being converted to artificial exact values.

## Source QC and external-domain limits

- The Sun/Wang/Shen source polarity is normalized explicitly: source class `0` means blocker, while canonical `is_blocker=1` means blocker.
- Its regression field is interpreted as `log10(IC50 in nM)` and converted as `pIC50 = 9 - stored_value`; it is never read as an nM concentration.
- Public hERG normalization retained **{source_qc.get("public_unique_structures", "NA")} unique structures** and **{source_qc.get("public_measurements", "NA")} measurement rows**; **{source_qc.get("public_quarantine", "NA")} rows** are quarantined rather than averaged away.
- There is **{source_qc.get("public_train_validation_overlap", "NA")} standardized-structure train/validation overlap** and **{source_qc.get("public_mw_domain_contradictions", "NA")} supplied structures** contradict the paper's stated 200–600 Da domain.
- The public hERG set has no exact internal overlap and maximum nearest Morgan similarity only **{source_qc.get("maximum_internal_public_similarity", 0.309)}**. Its source-held-out score is external context, not internal calibration evidence.
- The PROTAC-1 efflux values `309 ± 72` (PDF) and `165 ± 72.3` (CSV) remain separate, conflicting source records; neither is selected or averaged.
- The evidence-row PK algebraic-closure audit at the 15% tolerance was **CL: {source_qc.get("cl_closure_pass", "NA")} pass / {source_qc.get("cl_closure_fail", "NA")} fail** and **F: {source_qc.get("f_closure_pass", "NA")} pass / {source_qc.get("f_closure_fail", "NA")} fail**. Median relative discrepancies were **{_format_value(source_qc.get("cl_closure_median_relative_error"))} for CL** and **{_format_value(source_qc.get("f_closure_median_relative_error"))} for F**; failures remain closure diagnostics, never independent labels. The reviewer audit separately collapses source signatures and is authoritative for unique-compound failure counts.
- **{source_qc.get("unresolved_study_pairing_issues", "NA")} unresolved-pairing** and **{source_qc.get("missing_explicit_dose_pair_issues", "NA")} missing-dose-pair** raw validation-issue rows remain explicit. Exact duplicate issue rows are counted separately in the reviewer audit. Values are not averaged across unresolved studies.

## Stage status

{status_table}

## Model evidence

{model_table}

Decision-track gate result: **{decision_count} decision-track, {discovery_count} discovery-track, and {rejected_count} rejected model rows**. An `evaluated` fit is not automatically decision-ready; current optimizer predictions remain provisional until prospective calibration and non-inferiority gates pass.

Aggregate accuracy is not treated as mechanism. Each result has a separate explanation contract containing split provenance, observed held-out metrics, calibration evidence, applicability, residual clusters, matched pairs, competing explanations, and a falsification experiment.

## Findings and failure analysis

{failure_text}

## Molecular-weight regime conclusion

{cutoff_statement}

MW≥650 Da remains an inclusion description, not a discovered biological threshold. A 700- or 750-Da boundary is not adopted unless bootstrap location, scaffold stability, and cross-outcome gates all pass.

## What is mechanistically identifiable now

1. **IV exposure family:** dose-normalized IV AUC and CL are one algebraic family. Only one can be an independent label.
2. **Oral exposure family:** observed F constrains only the product Fa×Fg×Fh. The three factors are not separately identified by summary PK.
3. **hERG endpoint:** continuous potency with censoring is supportable; state-dependent binding/trapping kinetics are not identifiable without voltage-protocol onset/recovery data.
4. **Fast molecular physics:** microstate/conformer distributions can define hypotheses and pilot selection. They do not become calibrated transport or affinity predictions merely by being three-dimensional.
5. **Heavy physics:** membrane PMF and receptor-ensemble workflows are packaged with convergence gates; their observables remain unavailable until GPU/HPC production completes.

## State, conformer, and feature admission

{sampling_admission_text}

The internal structure-only matrix contains **{conventional_control_count} conventional
controls**. They are an empirical null, not {conventional_control_count} fundamental
mechanisms. Redundant exact-mass/heavy-atom axes and the constant
submitted-parent formal charge are excluded. Fundamental PK/hERG parameters
are restricted to declared free energies, rates, transport coefficients, and
boundary conditions; populations, permeability, occupancy, and endpoint
summaries remain derived functionals or observations.

## Mechanisms retained for falsification

- **Transition gating:** productive-state reactive flux and MFPT, not a global
  compactness or q05-polarity score.
- **Interfacial trapping:** membrane entry, core crossing, and release as
  separable bottlenecks, with PMF, diffusivity, coordinate, and hysteresis
  controls.
- **Charge-gated hERG access/binding:** local state switching couples
  membrane/cavity access to state-conditioned stabilization, unbinding, and
  trapping.

These are hypotheses, not established causes. No universal chameleonicity,
compactness, ETR, IMHB, or receptor-contact score is promoted. Their fastest
falsifiers combine free-monomer/flux/efflux measurements, matched-pair raw PK
profiles, dynamic hERG protocols, and only then converged path-resolved
simulation.

## Assay and simulation actions

- In-vitro panel: **{assay_summary.get("panel_size", "NA")} compounds**, including **{assay_summary.get("matched_pairs", "NA")} matched pairs**.
- Full rat IV/PO profiles: **{assay_summary.get("rat_profiles", "NA")} compounds**.
- State-dependent hERG protocols: **{assay_summary.get("herg_kinetics", "NA")} compounds**.
- Heavy-physics status: {heavy_physics_status.get("status", "HPC bundle generated; production not launched")}.

## Optimizer readiness

{optimizer_table}

`predictions` counts numerical outputs, not qualified recommendations. `inside_domain` is a similarity/domain flag, not proof of calibration. Algebraic CL and F closure remain derived lineage roles, and free-exposure margin remains required-data until unbound Cmax is supportable. No scalar objective, rank, or generation instruction is defined.

## Immediate next steps

1. Obtain chemist-validated series and chronology metadata; lock an outcome-blind new-series Panel V and the intended-use/statistical protocol before outcomes are released.
2. Resolve or quarantine the 130 unique canonical errors and eight stereochemical ambiguities; never repair conflicts by averaging.
3. Separate Panel V from the outcome-informed 16-compound Panel M and confirm at least four atom-mapped mechanistic pairs.
4. Measure free monomer, flux/recovery/efflux, disposition constraints, per-animal IV/PO profiles, and protocol-complete dynamic hERG for Panel M.
5. Use those direct results to select decisive compounds for adaptive state/conformer, transition-network, membrane, and bounded receptor HPC work; do not simulate all 110 by default.
6. Fit the integrated State–Path–Flux observation model, then evaluate Panel V once without retuning.

## Still unidentified

- Experimental micro-pKas and solution/membrane state populations.
- Formulation, dissolution, intestinal extraction, transporter contributions, and blood partition for most compounds.
- Raw concentration-time curves, animal metadata, sampling, LLOQ, and individual variability.
- hERG free assay concentrations, temperature/voltage harmonization, binding kinetics, and trapping.
- Converged membrane and receptor-state observables for the internal series.

The optimizer contract therefore exposes separate continuous endpoints, uncertainty, domain status, and required-data flags. It contains no scalar objective, compound rank, or generation instruction.
"""
    return atomic_write_text(path, text)


def write_run_summary(path: Path, payload: dict[str, Any]) -> Path:
    """Persist stage facts only; repository hashes and git state are intentionally absent."""

    forbidden = {"git", "commit", "sha", "confidentiality"}
    if any(token in key.casefold() for key in payload for token in forbidden):
        raise ValueError("Research run summaries must not contain git/SHA/confidentiality state")
    return atomic_write_json(path, payload)
