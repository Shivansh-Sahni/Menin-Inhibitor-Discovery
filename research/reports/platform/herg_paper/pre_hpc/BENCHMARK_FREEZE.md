# Pre-HPC label-blind benchmark freeze

The release materializes benchmark **membership only**. It embeds no target
values, performs no training, does not create an adjudicated gold standard, and
does not authorize a superiority claim.

## Materialized challenges

| Challenge | Rows | Structures | Test rows | Interpretation |
|---|---:|---:|---:|---|
| Q0 official scaffold | 339,373 | 339,373 | 40,898 | Existing large weak fixed-dose split |
| Q1 official scaffold | 23,186 | 22,081 | 2,400 | Quantitative pIC50 split |
| Q2 official scaffold | 3,548 | 2,776 | 602 | Eligible functional assay-aware split |
| Q2 patch-clamp transport | 910 | 458 | 119 | Patch-clamp subset; not an independent external panel |
| Q2 automated-patch stress | 854 | 434 | 112 | Automated subset |
| Q2 manual-patch stress | 34 | 34 | 4 | Too small for a standalone performance claim |
| Q2 ChEMBL-source stress | 3,548 | 2,776 | 602 | Equals current eligible Q2; a true source holdout is not yet possible |
| QT translation context | 221 | 95 | 24 | Clinical context only; no hERG labels embedded |

The confirmed-WT sensitivity membership currently equals Q0 because the large
qHTS backbone is the only task whose source explicitly establishes WT at that
scale. It is therefore a scope check, not an independent evaluation.

## Explicit blockers retained

- Strict temporal holdout: document year is incomplete on the master task
  surface.
- Low-similarity external holdout: similarity threshold and the external panel
  must be frozen before feature computation.
- Prospective manual-patch gold set: requires independent adjudication and a
  sealed experimental panel.

The machine-readable registry, label-blind memberships, input hash, artifact
hashes, schemas, and scientific contract are under
`research/data/platform/processed/herg_hierarchy/v1_4_benchmark_freeze/`.
Manifest self-hash:
`277675187ad7b475ebeb59c875291f7b69783a1e40ed853b0cb221efc0bb9b1e`.
