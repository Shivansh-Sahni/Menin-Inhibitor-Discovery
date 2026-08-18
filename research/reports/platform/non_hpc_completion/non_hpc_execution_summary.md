# Bounded non-HPC expansion: integrated execution summary

**Evidence date:** 2026-08-05  
**Disposition:** all presently feasible bounded CPU/data-engineering work is complete; scientific,
clinical, rights, checkpoint, HPC, validation, and public-release approvals remain open.

## What is now implemented

- A target-agnostic canonical ChEMBL 37 corpus already containing 3,938,372 accepted
  observations with source lineage, identity registries, QC, fixed splits, serialized examples,
  and lightweight diagnostics.
- Deterministic external-source acquisition/normalization boundaries for BindingDB, UniProt,
  ClinicalTrials.gov, Drugs@FDA, and DailyMed, with zero external rows silently promoted into
  the canonical corpus.
- A 49-source primary/official literature review and 17-candidate model decision matrix. The
  remembered “~100,000 structures” model is most plausibly RoseTTAFold All-Atom (121,800
  protein–small-molecule structures), but this remains a moderate-high-confidence inference that
  the professor should confirm.
- An external-admission audit over 93,712 BindingDB rows: 33,075 unique exact molecule/single-
  target links, 34,265 linked measurements, 20,306 exact cross-source mirror candidates, and
  1,548 disagreement groups requiring adjudication. Admission remains zero.
- A deeper leakage audit covering 28 tasks and 85 materialized strategies. Chemical work includes
  18 exhaustive audits over 64,983,791 pairs and 13 explicit CPU-budget not-runs; protein work
  includes 52 exhaustive and two deterministic sampled audits. Evidence scopes are never pooled.
- A provisional, bounded first CPU pilot: exact hERG functional IC50 (137 rows, 104 molecules,
  one protein, one source), evaluated primarily by scaffold split with molecule grouping as a
  sensitivity. Exact binding Kd is the larger second-stage candidate after added controls. This is
  a planning decision, not authorization to fit or claim a model.
- Six supplemental label-blind context split candidates. Strict-temporal hERG routes 71/17/43
  rows with six unknown-year exclusions; exact binding Kd routes 43,491/8,045/10,381 with 417
  exclusions. Context-group overlap is zero, while molecule/protein/source overlap is reported.
- Frozen SIFTS structure metadata: 1,009,429 mapping segments and exact accession candidates for
  5,428/9,411 canonical proteins and 7,176/14,983 external UniProt entries. No coordinates or
  predicted models were downloaded; 9,071 construct records remain unreconciled.
- ClinicalTrials.gov result candidates: six inventories spanning 3,879 studies, including 10,144
  conservative QT/QTc/PK candidates (9,492 genuine metric/event candidates). Missing results are
  unknown, never negative; intervention and group linkage is not inferred.
- Drugs@FDA candidates: 630,928 normalized application, product, submission, action, document,
  ingredient, and anomaly rows. All 7,220 anomalies are retained; 60,157 ingredient-name
  components remain unresolved text, not molecules. DailyMed remains archive-inventory only.
- PK-DB audit: 17 official/sanitized artifacts and ten distinct PK endpoint ontologies. Official
  statistics report 138,411 outputs, but anonymous output retrieval returned zero, global Basic
  security is declared, and the sampled public study is closed-licensed. Consequently, zero PK
  observations, identity links, or labels were admitted.
- A fail-closed configuration, release/governance packet, compute/operations plan, external and
  prospective validation protocol, decision register, shared task ledger, integrated CLI/Make
  targets, and a lead-owned cross-workstream completion verifier.

## Verification standard

Every new materialized data workstream has source/configuration/code binding, closed topology,
tamper tests, and a real byte-identical replay. The lead reviewed each delegated implementation,
repaired cross-workstream weaknesses where found, regenerated superseded outputs rather than
overwriting evidence, and retained explicit zero-label/zero-training assertions. The final gate
also reruns the original whole-platform mechanical verifier and independently binds every new
critical artifact.

## What this does not honestly establish

- No substantive model was trained, no large checkpoint was downloaded, and no test lockbox was
  opened for this expansion.
- No hERG, QT/QTc, PK, affinity, binding-free-energy, or clinical prediction has been validated.
- No external clinical/regulatory/PK candidate is yet a canonical label.
- Dataset/source/model redistribution rights and a top-level repository license are not approved.
- The provisional task, estimand, claim, metrics, leakage thresholds, exact checkpoint, HPC plan,
  and independent external/prospective validation still require named human owners and approval.

The defensible achievement is therefore a rigorously organized, reproducible evidence and
model-readiness platform with unusually explicit failure boundaries—not a completed predictive
drug-discovery product.
