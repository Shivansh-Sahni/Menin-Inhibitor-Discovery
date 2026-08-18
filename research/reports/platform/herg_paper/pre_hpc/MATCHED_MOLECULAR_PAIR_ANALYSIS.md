# Q1 matched-molecular-pair analysis

## Result

The Q1 quantitative hERG set now has 48,988 deterministic, single-cut matched-molecular-pair definitions across 22,081 structures. Pair construction used structure and split information only: 43,824 train pairs, 2,900 validation definitions, and 2,264 test definitions. Validation/test targets are not retained in any pair artifact. Two oversized common cores were explicitly skipped under the preregistered 64-variable-fragment cap; there were zero fragmentation failures.

Training-only effect estimation identified 6,161 pairs with an absolute median potency difference of at least 1 pIC50. These are activity-cliff candidates, not automatically independent biological replications: structures and transformations recur, assay/source confounding remains, and no nominal p-value should be treated as confirmatory.

## Fundamental-property signals

Across the 43,824 exploratory training-pair contrasts, the largest absolute signed Spearman associations with potency change were molecular logP (`rho = 0.294`), topological polar surface area (`rho = -0.245`), hydrogen-bond donors (`rho = -0.156`), and acceptors (`rho = -0.148`). The previously highlighted standardized logP × TPSA interaction was much weaker within matched pairs (`rho = 0.027`). This is useful negative evidence: the broad interaction signal does not reproduce strongly as a within-pair monotonic effect and should not be advertised as a mechanism.

All results are exploratory, training-only, noncausal, and unadjusted for repeated structures, repeated transformations, protocol, or source. Their immediate use is to preregister transformation-stratified evaluation, prioritize assay-aware review, and compare whether future models recover signed analog-series changes—not to claim that changing one descriptor causes hERG blockade.

## Contract

The builder honestly records that the source task file's label column was requested through a predicate-pushed training-only projection. No nontraining label value was returned to the analysis frame or retained, but physical Parquet page-level nonaccess cannot be proven because source row groups mix partitions. The label-free registry contains no pIC50, class, or target column. Pair definitions are structure- and scaffold-exclusive across splits and use a conservative one-cut MMP rule: core at least 10 heavy atoms, variable fragment at most 12 heavy atoms and 35% of the molecule, one representative per variable fragment, and largest-core deduplication. MMP cores and descriptor deltas are exploratory analysis fields, not a production model-input feature store.

Machine artifacts and the self-hashed manifest are under `research/data/platform/processed/herg_hierarchy/v1_5_mmp_analysis/`. Manifest: `9298e68e6ae2841be5ed466a86f693bed64e18ff84304899f4e494d4d1d2ace7`.
