# External evidence canonical-admission readiness

## Result

This CPU-only audit is mechanically complete, but **no external source is ready for
canonical admission or model training**. It created zero canonical observations and
zero model labels. It did not train a model.

## What was measured

- Reverified the exact frozen normalized bundle, including its internal manifest,
  file inventory, schemas, row counts, hashes, zero-label flags, and source semantics.
- Streamed every normalized Parquet field and counted non-null, null, and blank-string
  values under fixed schema contracts.
- Matched BindingDB ligand InChIKeys and single target accessions exactly to the frozen
  ChEMBL molecule/protein identities. 33,075 article
  rows and 34,265 affinity
  measurements have unique exact dual-link candidates. These are review candidates,
  not admissions.
- Compared linked BindingDB measurements with canonical ChEMBL observations. There are
  20,306 external measurement
  candidates with at least one exact identity/endpoint/relation/value match. This is a
  possible mirror signal, not proof of shared experimental provenance.
- Audited ClinicalTrials.gov inventory coverage: 210,644 unique
  studies, including 1,343 heuristic-cohort
  studies marked as having both posted results modules. The normalized data contain no
  outcome rows, arms, denominators, intervention-to-molecule links, QT/QTc values, PK
  values, or adverse-event counts.
- Audited Drugs@FDA and DailyMed as archive/table inventories only. They contain no
  normalized record-level product/molecule/outcome evidence.

## Conservative decision rules

1. A prediction is never an observed label.
2. Missing or unposted evidence is unknown, never a negative label.
3. One exact molecule and one exact single-protein accession match are only identity
   candidates; multicomponent, missing, and multi-match cases remain review/quarantine.
4. Exact numeric agreement is only a duplicate/mirror candidate; disagreement is only a
   conflict candidate until assay, construct, condition, document, and provenance review.
5. Every source is rights-blocked because the frozen manifests contain no machine-readable
   source-specific clearance for intended use, training, and redistribution.
6. No endpoint pooling, automated deduplication, identity replacement, negative-label
   inference, canonical admission, or model-label creation is performed.

## What must happen before admission

- Obtain and record versioned terms/license, intended-use, redistribution, attribution,
  and human data-steward/legal approval per source.
- For BindingDB, perform molecule standardization, target/construct/assay review, document
  provenance reconciliation, unit/relation review, and explicit mirror/conflict decisions.
- Re-extract version-bound ClinicalTrials.gov results modules and normalize arms,
  interventions, time frames, units, denominators, and adverse-event groups; then link
  interventions to molecules with human review.
- Repair or quarantine FDA relational anomalies, normalize applications/products/actions
  and active ingredients, parse versioned DailyMed SPL sections, and create reviewed
  molecule links.
- Freeze an admission policy and independently reproduce it before rebuilding canonical
  data. Until then, external canonical rows and labels must remain zero.

## Limits

This is a readiness and candidate-accounting analysis, not scientific validation. Exact
identity matching does not resolve salts, mixtures, stereochemistry policies, constructs,
assay context, clinical causality, or regulatory interpretation. Clinical cohort selection
is heuristic and retains false positives. Source-local and cross-source value differences
can be legitimate experimental heterogeneity. The analysis used no HPC and no network data.
