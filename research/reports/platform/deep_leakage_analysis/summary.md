# Deep leakage audit and first-task decision

## Decision

The first defensible **CPU pilot**, not a production or clinical claim, is
`default::default__herg__ic50__herg_functional__nm__continuous_exact`. It was selected because it is a narrow hERG
functional IC50 task with exact continuous measurements and both molecule- and
scaffold-based split candidates. It contains only 137 rows from
1 source and 1 protein, so it cannot
support cross-source, cross-protein, clinical, or prospective claims.

Use the **scaffold split as the primary evaluation**: its supplemental audit
found 0
validation/test structures at or above the frozen Morgan threshold and
cross-partition scaffold overlap was
`false`.
Keep the molecule-grouped split as sensitivity analysis only: it had
4
high-similarity queries and maximum nearest-train similarity
1.0.

The recommended second-stage scale benchmark is the exact binding Kd task. Kd
is closer to a direct equilibrium-affinity quantity than IC50, but its assay
heterogeneity and unavailable validated protein-family hierarchy still require
careful stratification. Derived binding-free-energy views are transformations
of the same measurements, not independent validation data.

## What was actually checked

- 28 tasks and 85 materialized strategies were enumerated.
- Exact molecule, protein, target/source identity overlaps were recomputed over every published split row.
- Assay, document, and document-year context were read from canonical Parquet using an explicit label-free projection and joined exhaustively by observation ID.
- 18 chemical audits evaluated every declared validation/test-versus-train Morgan-fingerprint pair; 13 larger audits were not run because they exceeded the frozen CPU pair budget.
- Protein sequence identity was exhaustive. K-mer Jaccard evidence is exhaustive only below the pair budget and otherwise is a deterministic sample.
- The complete local model-candidate registry was reviewed for pretrained-overlap audit readiness. No actual pretrained-corpus overlap could be measured because exact checkpoint hashes and training corpora are not frozen.
- No canonical label column, validation label, test label, or model-ready test lockbox was opened. No model was trained and no official split was modified.

## Hard limitations and next decisions

1. A random molecule/scaffold partition is not temporal or prospective. Use a separately frozen temporal split where year completeness permits it.
2. A single ChEMBL source makes source holdout impossible. Admit independently governed external evidence before cross-source claims.
3. Assay and document overlap is reported, not wished away. Add assay/document-grouped sensitivity splits before publication.
4. Fingerprint similarity is representation- and threshold-dependent. The report binds RDKit, radius, bit length, threshold, population, and pair count.
5. A target ID is not a protein-family annotation. A reviewed family hierarchy or sequence-cluster contract is still missing.
6. The task score is a transparent engineering prioritization heuristic, not a learned scientific result.

## Bottom line

This closes the strongest feasible label-blind CPU audit for the accepted split
suite. It improves evidence about leakage and selects a bounded first pilot,
but it does **not** establish performance, clinical safety, scientific claim
readiness, or authorization for substantive training.
