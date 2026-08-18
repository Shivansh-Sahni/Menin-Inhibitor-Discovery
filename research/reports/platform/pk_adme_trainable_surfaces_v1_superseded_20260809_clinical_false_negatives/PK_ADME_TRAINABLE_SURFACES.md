# PK/ADME trainable surfaces v1.0

## Outcome

The release contains 675,744 source-bound endpoint observations and 642,065 rows in endpoint-specific modeling tasks. It covers 189,912 standardized molecules, 175,965 connectivity leakage groups, and 93 tasks meeting the minimum size contract.

This is a noncanonical modeling-preparation release. It creates no clinical-outcome, QT, hERG, safety, efficacy, or approval labels, and it does not promote any row into the canonical PK store. Source-native human clinical PK measurements may remain as PK targets and are explicitly flagged; QT or ECG wording is context only and is never a target.

## Honest source counts

- chembl_37: 469,913 standardized endpoint observations; 396,039 exact and 40,509 censored-only modeling rows.
- dailymed: 0 standardized endpoint observations; 0 exact and 0 censored-only modeling rows.
- drugs_at_fda: 0 standardized endpoint observations; 0 exact and 0 censored-only modeling rows.
- ncats_adme: 7,697 standardized endpoint observations; 4,713 exact and 2,745 censored-only modeling rows.
- openadmet: 85,293 standardized endpoint observations; 84,596 exact and 697 censored-only modeling rows.
- pkdb: 0 standardized endpoint observations; 0 exact and 0 censored-only modeling rows.
- tdc_adme: 112,841 standardized endpoint observations; 112,824 exact and 0 censored-only modeling rows.

Counts are endpoint observations, not document hits or physical file rows. A paired structure/label row is counted once per non-null endpoint. ExpansionRx raw/train/test overlap is represented only by the full raw file. Octant and NCATS replicate/well fields are not multiplied into endpoint observations.

## Largest endpoint-specific tasks

- chembl37__half_life__source_context__h: 46,412 exact rows, 3,337 censored-only rows, and 28,989 leakage groups.
- chembl37__auc__source_window__ng_h_ml: 39,284 exact rows, 123 censored-only rows, and 19,096 leakage groups.
- chembl37__clearance__systemic_or_total__ml_min_kg: 31,623 exact rows, 180 censored-only rows, and 20,624 leakage groups.
- chembl37__logd__source_ph__unitless: 29,685 exact rows, 841 censored-only rows, and 26,600 leakage groups.
- chembl37__bioavailability__reported_or_absolute__percent: 29,166 exact rows, 1,191 censored-only rows, and 21,190 leakage groups.
- chembl37__cmax__molar__nm: 26,239 exact rows, 315 censored-only rows, and 15,007 leakage groups.
- chembl37__solubility__molar_source_context__nm: 24,040 exact rows, 7,475 censored-only rows, and 26,858 leakage groups.
- openadmet__chemeleon_permeability_logd_ppb__logd: 22,806 exact rows, 0 censored-only rows, and 22,109 leakage groups.
- chembl37__half_life__microsomal__h: 22,420 exact rows, 7,324 censored-only rows, and 16,988 leakage groups.
- chembl37__intrinsic_clearance__microsome_or_tissue__ml_min_g: 21,935 exact rows, 5,153 censored-only rows, and 17,314 leakage groups.
- chembl37__logp__source_method__unitless: 20,008 exact rows, 224 censored-only rows, and 17,141 leakage groups.
- chembl37__solubility__mass_source_context__ug_ml: 19,988 exact rows, 6,589 censored-only rows, and 19,739 leakage groups.

Task identifiers are scientific boundaries. Training code must filter one task at a time or use an explicitly reviewed multitask objective; it must not concatenate normalized_value across tasks as a common target.

## Leakage controls

- The split group hashes the standardized InChIKey connectivity block, conservatively co-locating stereoisomers and protonation variants.
- A scaffold group derived from the exact standardized representation is supplied for holdout evaluation; the connectivity leakage group remains mandatory so tautomeric or protonation representations cannot cross splits.
- ChEMBL-derived OpenADMET CheMeleon tables are flagged as lineage-overlap risks and are not eligible for the default cross-source union.
- Duplicate-group identifiers expose same-task, same-structure, same-value repeats without asserting that separate source records are the same experiment.

## Context and censoring

Species, matrix, route, dose, time, assay, and document context are populated only from physically present source fields or narrowly defined text parsers. Missing context remains null. Absolute exposure tasks such as AUC and Cmax carry a context-completeness flag and must not treat missing dose or route as a reference regimen.
The release flags 1,759 observations with explicit human clinical or patient PK context and 3 observations whose source context mentions QT or ECG. These flags stratify provenance; they are not clinical-risk, QT, or hERG labels.

Exact, less-than, less-than-or-equal, greater-than, and greater-than-or-equal relations remain distinct. Censored observations expose only the applicable bound. Approximate or unknown relations remain in the measurement ledger but are excluded from the default modeling surface.

## Blockers and limits

- TDC benchmark files generally lack unit, route, dose, and matrix columns; source-native-scale tasks remain separate and the underlying dataset rights require source-by-source review.
- NCATS CYP qHTS replicate tables are excluded because broad local structure resolution and a reviewed replicate-aggregation contract are absent.
- DailyMed sections and tables are machine-detected candidates, not normalized measurements, so they contribute zero labels.
- Drugs@FDA application, product, and action rows are regulatory metadata, not molecule-level PK measurements, so they contribute zero labels.
- PK-DB reports a large official corpus, but the local anonymous output acquisition contains zero records and its rights/access boundary remains unresolved.
- Rights fields are carried per observation; modeling readiness does not resolve redistribution obligations.

## Validation

All 78 consumed input files and 9 release artifacts are SHA-256 bound. Parquet schemas, row counts, unique observation IDs, task contracts, canonical-admission=false, and the absence of hERG/QT label fields are validated in validation.json.
