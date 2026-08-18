# hERG pre-HPC benchmark freeze v1.5

This supplemental release is label-blind. It reads only routing metadata, structures, assay/document context, and policy booleans. It embeds no target relation, value, bound, class, native value, or native label; opens no test label; and trains no model.

## Materialized challenges

| Challenge | Rows | Train structures | Validation structures | Test structures | Purged rows |
|---|---:|---:|---:|---:|---:|
| Q2_ASSAY_GROUP_HOLDOUT | 1,610 | 1,205 | 57 | 268 | 1,938 |
| Q2_AUTOMATED_PATCH_MODALITY_HOLDOUT | 2,753 | 1,560 | 221 | 370 | 305 |
| Q2_CHEMBL37_EXACT_TEMPORAL_COMPLETE_CASE | 106 | 56 | 10 | 19 | 25 |
| Q2_DOCUMENT_GROUP_HOLDOUT | 2,380 | 1,296 | 268 | 342 | 1,168 |
| Q2_DOCUMENT_YEAR_TEMPORAL_COMPLETE_CASE | 1,604 | 83 | 5 | 1,489 | 473 |
| Q2_LOW_SIMILARITY_060_SCAFFOLD | 3,441 | 1,979 | 244 | 466 | 0 |
| Q2_NO_NEAR_DUPLICATE_080_SCAFFOLD | 3,541 | 1,979 | 283 | 507 | 0 |

Assay/document partitions use deterministic group-size balancing with a fixed seed and no seed search, followed by complete removal of every structure or master scaffold that crossed a proposed partition. Temporal challenges use whole document years and exclude undated rows before the same leakage purge.

The two chemical-distance challenges use RDKit Morgan fingerprints (radius 2, 2048 bits; RDKit 2026.03.3). Validation structures are compared with training; test structures are compared with training plus the threshold-qualified validation structures. The thresholds are strict `<0.60` and `<0.80`.

## Explicit blockers

| Challenge | Status | Blocker |
|---|---|---|
| LOW_SIMILARITY_EXTERNAL_HOLDOUT | blocked_no_external_panel | The v1.5 low-similarity challenges are internal; no external panel is frozen. |
| PROSPECTIVE_MANUAL_PATCH_GOLD | blocked_no_prospective_adjudication | No independently adjudicated prospective manual-patch panel exists locally. |
| Q1_CROSS_SOURCE_HOLDOUT | blocked_underpowered_after_leakage_purge | The independent ChEMBL test surface is too small after structure/scaffold isolation. |
| Q2_MANUAL_VS_AUTOMATED_MODALITY_HOLDOUT | blocked_underpowered_after_leakage_purge | Manual-patch structures largely overlap automated-patch structures. |
| Q2_SOURCE_FAMILY_HOLDOUT | blocked_single_source | All eligible Q2 memberships come from one source family. |
| STRICT_ASSAY_DATE_TEMPORAL_HOLDOUT | blocked_no_assay_date | Document year is available only as a mixed source/publication clock. |

## Scientific limits

- Every materialized challenge remains internal to the assembled public corpus.
- Document year is not assay date, synthesis date, or first disclosure date.
- Similarity depends on standardization, Morgan radius, bit width, and threshold.
- Context-group balancing uses row counts only; it never inspects outcome prevalence.
- These memberships are sensitivity benchmarks, not a gold standard and not evidence of predictive superiority or clinical safety.

## Reproducibility

The manifest binds 29 source files by full SHA-256, size, Parquet row count/schema where applicable, the implementation hash, every output hash/schema, and a full deterministic source replay. v1.4 is unchanged.
