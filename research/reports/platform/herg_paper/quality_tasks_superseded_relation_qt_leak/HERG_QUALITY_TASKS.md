# Wild-type hERG quality-specific task bundle

This build preserves the existing hERG corpus and compiles separate modeling problems instead of pooling incompatible labels.

## Task ladder

- **Q0_WEAK_FIXED_DOSE_BINARY:** 339,373 eligible rows, 339,373 structures.
- **Q1_QUANTITATIVE_PIC50:** 23,186 eligible rows, 22,081 structures.
- **Q2_FUNCTIONAL_ASSAY_AWARE:** 3,557 eligible rows, 2,785 structures.
- **C0_CLINICAL_DEVELOPMENT_CONTEXT:** 3,056 eligible rows, 3,056 structures.
- **C1_QT_CONTEXT_EVALUATION:** 5,605 eligible rows, 95 structures.
  This record-level headline resolves to 221 exact linked QT/QTc endpoint candidates; the grains are not interchangeable.

## Interpretation

- Q0 is a large, weak but homogeneous automated FluxOR screen. It is the scale-first baseline, not an IC50 dataset.
- Q1 supports quantitative pIC50 regression and a blocker/gray/nonblocker ordinal analysis. The unresolved compilation method is retained explicitly.
- Q2 preserves exact and censored functional IC50 separately from AC50, inhibition, potency, and other native endpoints. Measurement technology is a covariate and error-analysis stratum.
- C0 and C1 are clinical-development and QT/QTc context. They may be used for stratification, external evaluation, and later exposure-aware modeling, but never as direct hERG training labels.

## Target scope and leakage

- Explicit mutant/variant observations excluded: 258.
- Confirmed wild type and wild-type-or-unspecified are reported separately; unspecified is not upgraded.
- One fixed whole-scaffold split is shared across tasks. Source-declared splits are provenance only.
- Clinical/QT rows are label-disabled, and only test-partition QT rows are marked held-out evaluation eligible.

## Recommended paper analyses

1. Compare Q0-only, Q1-only, Q2-only, sequential Q0→Q1/Q2, and assay-aware multitask models on the same scaffold holdouts.
2. Report error and calibration by measurement technology, assay family, endpoint, target-scope certainty, and evidence level.
3. Treat QT/QTc as downstream human repolarization context influenced by exposure, metabolism, ion-channel polypharmacology, physiology, and study design—not as a synonym for hERG block.
4. Use the clinical cohorts to test transport and concordance; do not leak them into molecular training labels.

No substantive model was trained by this build.
