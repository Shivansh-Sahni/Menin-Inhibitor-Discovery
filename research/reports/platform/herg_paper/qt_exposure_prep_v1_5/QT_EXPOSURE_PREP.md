# QT/QTc Exposure Preparation v1.5

## Outcome

This additive release creates collection and review assets for **95 structures**, **143 structure-trial contexts**, and **221 reported QT/QTc endpoints**. It computes no real safety margins and admits no QT, clinical, DailyMed, Drugs@FDA, or PK candidate as a direct hERG, clinical-risk, or training label.

All adjudicated dose, route, Cmax, fraction-unbound, metabolite, hERG IC50, and margin fields remain null. Candidate evidence is retained only to direct human review.

## Evidence actually present

| Evidence domain | Structures or contexts with candidate evidence | Interpretation |
|---|---:|---|
| Explicit QT correction method | 65 structures | Source-reported method context; not a hERG label |
| ClinicalTrials dose/strength text | 77 structure-trial contexts | Regex-bounded verbatim candidate; arm attribution still requires review |
| ClinicalTrials route text | 76 structure-trial contexts | Reported intervention text candidate |
| Drugs@FDA exact-name-linked application | 36 structures | Product metadata; formulation strength is not administered trial dose |
| Exact-application-linked DailyMed PK document | 33 structures | Candidate label document, not asserted current/preferred label |
| DailyMed Cmax candidate section | 31 structures | Machine-detected section only; no numeric value promoted |
| DailyMed bounded protein-binding term | 21 structures | Presence in stored evidence spans; non-detection is unknown |
| DailyMed bounded metabolite term | 26 structures | Presence in stored evidence spans; identity/activity unresolved |
| ChEMBL human Cmax candidate | 13 structures | Raw candidate awaiting dose/matrix/analyte adjudication |
| ChEMBL human FU/PPB candidate | 49 structures | Raw candidate; tissue/cell/plasma semantics remain source-native |
| Both human Cmax and FU/PPB candidates | 9 structures | Highest-priority reconciliation set; still not margin-ready |
| Adjudicated human Cmax plus fraction unbound | 0 structures | Required before real margin computation |

The processed PK/ADME observation table overlaps 0 cohort structures. The broader ChEMBL 37 inventory overlaps 93 structures but remains candidate-level. The local PK-DB admission decision reports 0 candidate observation rows and canonical admission = `false`.

## Review queues

The gap queue contains 143 structure-trial rows: P0_candidate_pair_reconciliation=14, P1_single_numeric_domain_plus_context=72, P2_source_enrichment_and_adjudication=47, P3_external_exposure_collection=10. P0 identifies contexts whose structure has both human Cmax and human FU/PPB candidates, not completed margins.

The source queue contains 1,072 review records: chembl37_pk_candidate=134, clinicaltrials_gov_intervention=181, dailymed_pk_candidate_section=192, drugs_at_fda_product=565. DailyMed review candidates are deterministically capped at 10 sections per structure; complete candidate-pool counts remain in the structure template.

## Margin contract

The machine-readable contract defines `IC50 / unbound Cmax` in micromolar units, analyte-specific molecular-weight conversion, fraction-unbound normalization, pIC50 bound reversal, and conservative interval arithmetic. It blocks calculation for missing or unadjudicated identity, unit, analyte, matrix, dose, route, population, schedule, time-basis, or source compatibility. No imputation is allowed.

## Boundaries

- ClinicalTrials intervention descriptions are trial-level candidates; endpoint-group-to-arm-to-dose attribution is not assumed.
- Drugs@FDA route/form and strength fields describe products, not necessarily the regimen used in a QT trial.
- DailyMed links use exact application-number overlap and preserve document/version/member hashes. They are candidate review links, not regulator-preferred-label assertions.
- DailyMed protein-binding, unbound, and metabolite flags are searched only within stored bounded PK evidence spans. Absence means not detected in those spans, never evidence of biological absence.
- ChEMBL PK inventory records preserve raw endpoint, relation, value, unit, organism, assay text, and document identifiers. They are not promoted into curated exposure inputs.
- Parent compound and active metabolites remain separate analytes.
- QT/QTc endpoints remain human phenotype context and are never converted to molecular hERG labels.
- Aggregate trial outcomes are not patient-level risk, causal attribution, a clinical safety classification, or medical guidance; this release makes no clinical-risk inference.

## Artifacts

- `qt_exposure_structure_template.parquet`: one row per structure with source coverage and hard blockers.
- `qt_exposure_structure_trial_template.parquet`: one row per structure-trial with source text candidates and intentionally blank adjudication fields.
- `qt_exposure_gap_priority_queue.parquet`: prioritized missing-input and reconciliation queue.
- `qt_exposure_source_adjudication_queue.parquet`: source-linked adjudication queue with provenance and limitations.
- `ic50_unbound_cmax_margin_contract.json`: units, identity, censoring, and interval-arithmetic contract.
- `qt_exposure_prep_manifest.json`: input/output hashes, schemas, counts, and zero-label checks.
