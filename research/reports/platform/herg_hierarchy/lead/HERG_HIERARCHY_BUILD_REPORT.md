# hERG hierarchy v1: lead integration and training-readiness report

Date: 2026-08-07  
Scope: wild-type/unspecified hERG only; no EPA or strict-task input; no model training.

## Result

The project now has a compact, structure-resolved hERG weak-label backbone and
separate assay-native quantitative, provenance, development, and clinical-QT
surfaces. The primary binary table is homogeneous: only source-reported
PubChem AID 720551 calls can determine its label. ChEMBL IC50, curated pIC50,
and clinical metadata cannot modify that label.

The immediately usable binary backbone contains **339,373 standardized
structures** with exactly four fields:

| Field | Meaning |
| --- | --- |
| `structure_id` | Deterministic identifier derived from the standardized InChIKey |
| `standardized_smiles` | RDKit cleanup, fragment-parent, uncharged, canonical isomeric SMILES |
| `standard_inchi_key` | Structure identifier after the declared standardization policy |
| `herg_blocker_label` | AID 720551 source call: `1` Active/blocker-like, `0` Inactive/nonblocker-like |

This is a weak fixed-dose functional-screen classification target. It is not
patch-clamp IC50, QT prolongation, torsades risk, or a clinical safety label.

## Count reconciliation

### Primary PubChem backbone

| Stage | Rows/structures | Active | Inactive | Excluded/other |
| --- | ---: | ---: | ---: | ---: |
| Source AID rows | 343,909 | 1,267 | 341,912 | 730 inconclusive |
| Unique PubChem CIDs with structures | 343,666 | 1,263 conflict-free | 341,669 conflict-free | 729 inconclusive-only; 5 CID conflicts |
| Standardized parent structures | 340,107 | — | — | 3,559 CID-to-parent merges |
| Final one-label backbone | **339,373** | **1,238** | **338,135** | 710 inconclusive-only; 24 structure conflicts |

Standardization merged 3,559 additional CIDs into already represented parent
structures and exposed 24 structure-level label conflicts. Those conflicts are
retained in the observation ledger and excluded from the one-label table.
The final active prevalence is **0.3648%**.

### All reported hERG evidence (T0)

| Evidence | Count |
| --- | ---: |
| Native reported observations | **407,956** |
| Valid-structure observations | 407,854 |
| Standardized structures with any reported evidence | **369,546** |
| AID 720551 observations | 343,909 |
| Zenodo 8359714 reported pIC50 observations | 22,969 |
| ChEMBL 37 CHEMBL240 observations | 41,078 |
| Quantitative pIC50 view | 23,186 observations / 22,081 structures |
| Quantitative derived blocker/nonblocker/gray calls | 13,947 / 5,139 / 4,100 |

The quantitative total includes the 22,969 reported pIC50 values and only 217
ChEMBL values converted from exact, positive, wild-type/unspecified functional
IC50 in nM or uM. Censored IC50, binding-assay IC50, Ki, inhibition percentages,
AC50, binary calls, and kinetics are not converted to pIC50.

## Reporting/evidence hierarchy

The tiers describe strength and reporting context. They are not interchangeable
biomedical labels.

| Tier | Operational meaning | Current result |
| --- | --- | --- |
| T0 reported | Traceable public hERG assay observation with preserved native endpoint and structure | 407,956 observations / 369,546 structures |
| T1 preclinical validation | Independent, concordant functional lineages with comparable assay semantics | 27 review candidates: 11 concordant, 16 discordant; **0 formally assigned** |
| T2 clinical validation | Structure-resolved human cardiac evidence with interpretable exposure/context; development phase alone is insufficient | 221 exact-linked cardiac-result candidates; **0 formally assigned** |
| T3 clinical-trial reported | T2-compatible evidence tied to a posted trial result, arm/outcome/time/unit/value | 221 posted-result candidates; **0 formally assigned** |

The T2 and T3 candidate inventories are presently identical because the only
human-cardiac input used here is the posted-results ClinicalTrials.gov extract.
This is an extraction boundary, not proof that T2 and T3 are scientifically the
same tier.

### Less-conservative operational v1.1

For modeling and coverage analysis, a second grain-explicit hierarchy avoids
shrinking every stage to only formally adjudicated unique compounds:

| Operational stage | Headline count and grain | Important disclosed coverage |
| --- | ---: | --- |
| O0 public reported hERG | **407,956 observations** | 369,546 structures |
| O1 curated/quantitative preclinical | **64,047 observations** | 30,660 structures; 23,186 normalized pIC50 values |
| O2 clinical development/regulatory | **3,056 structures** | 2,943 phase >=1; 17,824 FDA ingredient/product links |
| O3 clinical-trial reported | **1,694 exact intervention links** | 1,148 trials; 1,588 trial-structure pairs |
| O3-QT posted results | **3,828 result-value records** | 3,819 numeric values; 221 endpoints |

Every headline is above 1,000 at its declared physical grain. The manifest and
validator prevent a link, trial, endpoint, or result cell from being mislabeled
as a unique compound or direct hERG measurement.

Only 27 structures have both a decisive AID call and quality-gated functional
ChEMBL IC50-derived evidence. Eleven agree and sixteen disagree. AID versus the
larger pIC50 compilation also shows substantial disagreement. This is expected
in part because the fixed-dose AID assay and an IC50 threshold answer different
questions; it is evidence against naive cross-assay pooling.

## Clinical and development linkage

| Linkage result | Count |
| --- | ---: |
| Structures checked | 369,546 |
| Exact ChEMBL structure matches | 311,878 |
| Development/regulatory annotations | 3,056 |
| ChEMBL maximum phase 4 annotations | 1,554 |
| Exact Drugs@FDA application annotations | 707 |
| Exact name links in audit | 19,518 records / 940 structures |
| Posted QT/QTc result candidates | 221 endpoints / 95 molecules / 143 NCTs |
| Clinical hERG labels admitted | **0** |
| Model labels admitted from clinical metadata | **0** |

Absence is unknown, not negative. Approval, phase, trial registration, a QT
endpoint name, or a numeric trial result does not automatically validate a
molecular hERG label. All 221 candidates require human scientific review of
drug identity, exposure, comparator, direction, units, outcome semantics, and
whether the result is informative for the intended hERG claim.

## Files to use

Primary production root: `research/data/platform/processed/herg_hierarchy/v1/`

- `structure_consensus_binary.parquet`: simplest large binary training table.
- `observation_ledger.parquet`: authoritative assay-native evidence and lineage.
- `quantitative_pic50.parquet`: continuous potency view with origin and split.
- `hierarchy_annotations.parquet`: one row per evidence-bearing structure.
- `manifest.json`: bound input hashes, transformation policy, counts, artifact hashes.

Clinical candidate root:
`research/data/platform/processed/herg_hierarchy/v1_clinical_links/`

- `structure_development_annotations.parquet`: phase/FDA annotations, never labels.
- `t2_clinical_cardiac_evidence_candidates.parquet`: human-review candidates.
- `t3_posted_qt_trial_result_candidates.parquet`: posted-result candidates.
- `exact_name_structure_link_audit.parquet`: every accepted/rejected name link.
- `herg_clinical_links_manifest.json`: input and output bindings.

Model-ready split root:
`research/data/platform/processed/herg_hierarchy/v1_model_ready/`

- `structure_consensus_binary_scaffold_split.parquet`: the four primary fields
  plus `split` and `scaffold_group_id`.
- `manifest.json`: fixed assignment policy, source/artifact hashes, class counts,
  group counts, and zero-overlap QC.

Unified tier root:
`research/data/platform/processed/herg_hierarchy/v1_evidence_tiers/`

- `structure_evidence_tiers.parquet`: one row per structure across T0 and all
  higher-tier annotations/candidates.
- `cross_lineage_t1_candidates.parquet`: the 27 cross-lineage review records.
- `herg_evidence_tiers_manifest.json`: formal zero-promotion contract and hashes.

Operational v1.1 root:
`research/data/platform/processed/herg_hierarchy/v1_1_operational_tiers/`

- `operational_stage_records.parquet`: compact O0-O3 source-record references.
- `operational_qt_record_index.parquet`: result/denominator pointers without
  duplicating native trial payloads.
- `operational_stage_summary.parquet`: five grain-explicit headline rows.

Expanded trial-molecule root:
`research/data/platform/processed/herg_hierarchy/v1_1_trial_uplift/`

- `trial_structure_link_candidates.parquet`: parent-standardized ChEMBL trial
  drug structures with rule tier, ambiguity, molecule sets, and local-hERG flag.
- `trial_structure_link_audit.parquet`: all 9,831 intervention decisions.
- `herg_trial_uplift_manifest.json`: source/artifact bindings and cumulative
  exact, punctuation, cleaned-name, and component-set statistics.

The practical cleaned-name tier resolves **3,995 intervention records across
2,277 trials and 1,177 unique parent drug structures**. Of those, 571 already
intersect the measured local hERG hierarchy; the rest are an immediate clinical
drug universe for hERG prediction. The broader component-set tier reaches
4,191 records, 2,361 trials, and 1,216 structures while preserving combinations
as molecule sets rather than assigning one arbitrary component.

The source acquisition uses 103 MB, the core processed corpus 52 MB, and the
clinical candidate package 15 MB. No strict/EPA data were copied into these
products, and HERGAI was intentionally omitted rather than duplicating the
large AID lineage.

## What to train first

1. **First task: AID-only binary molecular classifier.** Use the 339,373-row
   table as weak-label pretraining. It is the largest homogeneous target and has
   a clear structure-to-call contract.
2. Use the completed scaffold-grouped split, not a random row split. It contains
   95,136 whole groups with zero exact-structure or scaffold-group overlap:
   train 265,625 (987 active), validation 32,850 (103 active), and test 40,898
   (148 active). The row fractions are 78.27/9.68/12.05 rather than exactly
   80/10/10 because scaffold groups are indivisible. Acyclic structures use 613
   declared exact-structure proxy groups rather than one empty-scaffold group.
3. Handle the 0.365% active prevalence with balanced batches or explicit class
   weighting. Preserve the natural-prevalence validation/test sets.
4. Report PR-AUC, active recall, precision at declared review budgets,
   enrichment, calibration, and confusion matrices. Accuracy alone is invalid.
5. **Second task: quantitative pIC50 regression or ordinal modeling.** Begin
   with the source-provided Zenodo development/evaluation boundaries; reconcile
   ChEMBL overlap before treating the 217 converted values as independent.
6. Later use separate assay-aware heads for fixed-dose flux calls, quantitative
   functional IC50/pIC50, binding/proxy endpoints, and clinical cardiac evidence.
   Do not collapse them into a fabricated universal potency.

## Remaining gates before a publishable model claim

- Freeze intended use: high-sensitivity triage, ranking, or calibrated potency.
- Complete the bounded near-neighbor audit and add time/source holdouts; the
  exact and Murcko-scaffold isolation gate already passes.
- Resolve record-level rights and redistribution terms for every released row.
- Manually adjudicate the 27 preclinical and 221 clinical candidates.
- Verify assay modality and upstream independence before promoting any T1 row.
- Preserve and evaluate the AID-to-pIC50 disagreement rather than deleting it.
- Define an external patch-clamp benchmark with overlap removed.
- Train only after split manifests, metrics, seeds, and model cards are frozen.

## Other modalities preserved, not rebuilt in this hERG pass

- BindingDB 2026-08 remains checksum-verified at 3,234,499 measurement rows and
  2,481,305 ligand-target keys. The local ChEMBL census separately retains
  858,991 positive single-protein Kd/Ki rows and 2,164,194 positive
  single-protein IC50/EC50 rows; overlaps and endpoint meanings prevent summing
  those as unique training labels.
- PK remains candidate-stage: 2,437 published systemic-PK benchmark rows lack
  required context, alongside 274,168 upstream ADME/safety rows and 330,261
  ChEMBL candidates. Context-complete systemic-PK admissions remain zero.
- Tox21 hERG AID 1671200 (9,667 rows / 7,671 CIDs), AID 588834 (5,381 rows /
  4,743 CIDs), and the verified complete ToxCast archive remain preserved as
  future assay-aware toxicology/functional auxiliaries. None were duplicated
  into the AID 720551 backbone.

## Verification performed

- Official PubChem structures returned for 343,666/343,666 requested CIDs, with
  zero missing/extra CIDs and zero blank structure fields.
- The 344-batch acquisition replay resumed all 344 batches without a new request;
  merged SHA-256 remained
  `595efcd06b0a1a8662220559e8de08afd10a0e680120af884d3be594b5b3fd7f`.
- Core build validator rechecked exact output membership, schemas, row counts,
  artifact hashes, unique observation IDs, input bytes/hashes, source contracts,
  and all manifest counts.
- Clinical validator rechecked closed output membership, schemas, counts,
  artifact hashes, every input binding, and the zero-label contract.
- Focused hierarchy and clinical tests passed before the full project gate.
  The final gate passed 695 pipeline tests and 53 Menin-Edit tests, Ruff over
  both trees and the acquisition script, mypy over 80 platform and 15
  Menin-Edit source modules, compilation, and `git diff --check`. The only test
  warning was the known sandbox physical-core detection fallback.
- The model-ready split validator recomputed every scaffold, hash assignment,
  class/group count, source row, and zero-overlap invariant.
- The unified tier validator rechecked input bindings, artifact hashes, counts,
  concordance states, and zero formal T1/T2/T3/model-label assignment.

## Honest boundary

This work makes hERG data organized, auditable, compact, split, and ready for
model-interface preparation. It does not demonstrate predictive performance, clinical
validity, prospective safety, causal relationships, or readiness to deploy.
