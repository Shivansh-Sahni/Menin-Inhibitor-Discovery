# hERG training surfaces v1.6

## Release outcome

This downstream release exposes 395,575 primary-eligible source-faithful observations spanning 366,281 structures. It also provides 339,373 one-row-per-structure confirmed-wild-type fixed-dose consensus labels backed by 343,114 decisive observations.

No model was trained. No row was promoted to gold or formal T1. No upstream release was modified.

## Scientific boundary

The broad primary surface admits only two label forms: decisive AID720551 fixed-dose binary observations whose structure has a unique Q0 consensus, and finite public native numeric measurements whose endpoint, nonapproximate relation, and unit are present. Native numeric endpoints are not pooled. Censored relations and standardized pIC50 bounds remain intact. Approximate relations are isolated in a sensitivity-only surface.

Clinical QT/QTc and development records are context only. They never supply a molecular hERG training label. Explicit mutants remain excluded. Wild-type-or-unspecified records are never upgraded to confirmed wild type.

## Nested surfaces

The preclinical native numeric surface contains 52,461 observations across 28,074 structures. The how-measured functional subset contains 12,636 observations across 9,434 structures; 7,610 of those are exact or censored standardized IC50/pIC50 observations. Functional membership follows the parsed measurement modality, not the source assay-family label, because locally available descriptions often identify patch clamp even when the source family field says binding or other.

The formal validated T1 surface is intentionally empty: 0 formal labels. The 27 cross-lineage structures and 115 curated functional observations remain review candidates only.

Clinical context is separate: 3,277 context rows spanning 3,056 structures, with zero direct hERG labels.

## Leakage and use

Every molecular row retains the frozen structure and whole-scaffold split. Candidate, conflict-queue, and automated lineage flags from v1.5 are carried as cautions, not automatic exclusions or proof of duplication. Evaluation panels must be adjudicated and then refrozen across structure, scaffold, assay, document, and measurement lineage before use.

Use the structure artifact for the large fixed-dose binary task. Use the observation artifact for assay-aware numeric or multitask objectives, conditioning or separating by endpoint, unit, relation, and measurement modality. The reporting inventory itself is not a pooled target.

## Validation

The build validated 407,698 admitted observations, 369,546 master structures, 258 replayed explicit-mutant exclusions, 3,604 deterministic measurement strata, all physical input and output hashes, Arrow schemas, row counts, relation semantics, clinical nonpromotion, formal-tier nonpromotion, and structure/scaffold split exclusivity.

## Limits

Clean means structurally and semantically eligible under the stated source-specific rules; it does not mean experimentally independent or human-adjudicated. Protocol completeness is inherited from local metadata and may remain unresolved. The quantitative compilation does not resolve original assay modality. Cross-source equal values and reuse signatures remain automated evidence only. No primary papers were newly adjudicated in this build.
