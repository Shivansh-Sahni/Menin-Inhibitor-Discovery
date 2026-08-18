# Platform scientific-audit package

Audit date: 2026-08-04 (America/Detroit)

This directory defines the scientific contract for the protein-agnostic
platform. The frozen ChEMBL-only canonical/statistical/split/corpus artifact
stack passed mechanical verification on 2026-08-04. This does not assert that a
rights-cleared multisource clinical corpus, final model, public release, or
product is complete. The post-gate independent reconciliation is under
`research/reports/platform/audit/`.

## Status vocabulary

Every material capability or artifact must use exactly one of these statuses:

| Status | Meaning |
|---|---|
| `completed` | Implemented, tested, documented, and accepted by the integration lead. |
| `locally_evidenced` | Directly observed in the current workspace, but not necessarily release-ready or independently reproduced. |
| `planned` | Designed or specified; no claim of working implementation. |
| `unavailable` | Required evidence or input was not accessible during this audit. |
| `license_blocked` | Use or redistribution awaits rights-holder or institutional approval. |
| `hardware_blocked` | The design exists, but the required compute was not available or authorized. |
| `requires_human_review` | A scientific, legal, confidentiality, or product decision cannot be made safely by automation. |
| `failed_current_check` | A check was run or a check artifact was inspected and it did not pass at the stated time. |
| `not_applicable` | The requirement does not apply, with a recorded reason. |

`locally_evidenced` is intentionally weaker than `completed`. In particular, the presence of a file, model artifact, or passing unit test is not evidence that its scientific claims, rights, or external validity are accepted.

## Audit documents

- `evidence_and_endpoint_ontology.md`: required entity, evidence-stage, endpoint, provenance, and claim semantics.
- `source_and_model_review.md`: dated source, license, version, citation, and external-model assessment.
- `claim_boundaries.md`: allowed and prohibited statements for affinity, PK, hERG/QT, clinical evidence, and predictions.
- `research/reports/platform/audit/repository_inventory.md`: repository state, observed artifacts, and implementation/documentation discrepancies.
- `research/reports/platform/audit/bias_missingness_selection_plan.md`: bias, missingness, selection, and subgroup analysis contract.
- `research/reports/platform/audit/statistical_analysis_plan.md`: estimands, uncertainty, comparison, and multiplicity rules.
- `research/reports/platform/audit/sensitivity_ablation_plan.md`: mandatory robustness and ablation matrix.
- `research/reports/platform/audit/publication_reproducibility_checklist.md`: release and paper gates.
- `research/reports/platform/audit/risk_register.csv`: machine-readable unresolved risks and owners.
- `research/reports/platform/audit/required_source_acquisition_acceptance.csv`: machine-readable phase-two acquisition contracts for the mandatory BindingDB, UniProt, ClinicalTrials.gov, Drugs@FDA, and DailyMed sources.
- `research/reports/platform/audit/readiness_rubric.md`: independent pretraining-readiness verdict and scoring rubric.
- `research/reports/platform/audit/final_cross_workstream_verification.md`: tiered physical and scientific acceptance protocol, adversarial findings, 25-section crosswalk, residual human blockers, and training boundary.
- `research/reports/platform/audit/independent_validation_results.json`: machine-readable independent check results, frozen artifact identities, tier decisions, and blocking gates used for the final audit verdict.
- `research/reports/platform/audit/final_25_section_status_matrix.md`: actual
  final status of every governing workflow section.
- `research/reports/platform/audit/final_reproduction_and_next_steps.md`:
  exact rebuild/verification commands, accepted hashes, training blockers, and
  authorized next-step boundary.

## Decision rule

The platform is not eligible for substantive large-model training until every
blocking item in the readiness rubric is passed. A human waiver can acknowledge
operational risk but cannot create missing rights, data, external validity, or
scientific evidence. Bounded diagnostics and loader smoke are not substantive
training or final model performance.
