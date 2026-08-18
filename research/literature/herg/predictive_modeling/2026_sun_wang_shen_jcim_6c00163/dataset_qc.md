# Dataset QC: Sun, Wang, and Shen hERG workbook

## Scope

This audit concerns `supplementary_dataset.xlsx`. The copied workbook is preserved unchanged. All counts below describe a read-only inspection and should be regenerated in the eventual ingestion pipeline rather than manually embedded in model code.

## Critical semantic rules

1. **Binary polarity:** class `0` means blocker (reported IC50 below 10 micromolar); class `1` means nonblocker/inactive. This is opposite the positive-class convention used by many classifiers. Internally, define a new explicit field such as `is_blocker` and test the mapping.
2. **Regression units:** the field labeled `IC50(nM)` contains `log10(IC50 in nM)`, not an IC50 in nM and not conventional pIC50. Use `pIC50 = 9 - stored_value` if a molar negative-log target is required.
3. **Qualitative values:** 2,354 classification rows lack a numeric IC50 and are primarily class-1 qualitative records. They support interval or binary supervision, not point regression.
4. **Prediction-looking column:** the regression workbook column named `hERG` appears to be a model output. Quarantine it from feature and label construction until the authors provide its exact generation and split provenance.

## Row and duplicate audit

| Sheet/use | Rows | Canonically unique | Duplicate/conflict observations |
|---|---:|---:|---|
| Classification | 8,334 | 8,310 | 24 duplicate groups; 12 have differing numeric values; 2 have opposite classes |
| Regression | 7,772 | 7,772 | no canonical duplicates detected in the inspected valid rows |
| External validation | 1,133 | 1,097 | 29 duplicate groups; 36 extra rows; 25 groups have differing numeric values |

One standardized structure overlaps between the regression and validation sheets. Any recreated validation must resolve that overlap before fitting.

Duplicate handling must be provenance-first:

- never average values before determining whether records share protocol, species, construct, temperature, and endpoint semantics;
- keep same-assay replicates as a distribution with count and dispersion;
- keep incompatible assays as separate observations connected to one compound identity;
- mark class-boundary conflicts explicitly; and
- form train/test groups at the standardized parent-structure level so duplicates cannot cross a split.

## Label composition

The classification sheet contains 3,914 blockers and 4,420 nonblockers/qualitative inactives. Only 5,980 rows contain numeric IC50 values. The coexistence of exact numeric values, inequality statements, and qualitative labels requires a typed observation schema:

- `exact`: numeric concentration and unit;
- `left_censored` or `right_censored`: explicit inequality and bound;
- `interval`: bounded range;
- `qualitative_active` or `qualitative_inactive`;
- `derived_binary`: threshold and source observation recorded.

Do not manufacture a numeric value at the 10 micromolar boundary for qualitative inactives. For regression, use exact and compatible censored-likelihood observations; for classification, run sensitivity analyses that exclude purely qualitative labels.

## Molecular-weight and intended-domain audit

Recomputed molecular weights do not exactly match the stated 200-600 Da curation range:

- classification: 173 rows outside 200-600 Da;
- regression: 190 rows outside 200-600 Da;
- classification at least 650 Da: 58 rows, 28 blockers and 30 nonblockers;
- classification at least 700 Da: 28 rows, 9 blockers and 19 nonblockers;
- classification at least 750 Da: 13 rows, 3 blockers and 10 nonblockers.

These counts are insufficient for an independent cutoff search. They also show that a hard threshold selected after looking at performance would be unstable. Molecular weight should initially be treated as one axis of a multidimensional applicability surface. Candidate strata (650/700/750 Da) can be reported descriptively, but a scientific boundary should emerge only if there is a reproducible change in conformational complexity, exposed-polarity behavior, assay missingness, error, or mechanism.

## Domain overlap with the current internal set

For 111 internal rows (110 unique standardized structures), radius-2, 2,048-bit Morgan nearest-neighbor similarity to this workbook is low:

- maximum similarity over the complete published workbook: 0.309;
- maximum similarity to published structures at least 650 Da: 0.252;
- count with similarity at least 0.5: zero.

This is a severe extrapolation warning. Randomly mixing the two datasets could produce a misleading aggregate score dominated by the public set while saying little about internal large molecules. Evaluation must preserve a fully held-out internal-domain test and report errors against both structural and mechanistic distance.

## Reproducibility gaps

The attachments do not provide:

- row-level primary source and assay-protocol metadata;
- published train/test assignments or random seeds;
- executable atom-typing and correction-factor code;
- fitted model artifacts;
- a complete map resolving 40 main-text correction factors versus M1-M30 in the supporting document; or
- clear provenance for every externally corrected value.

Consequently, a future implementation should be labeled an independent reconstruction, not a reproduction, unless these assets are obtained.

## Required ingestion gates

1. Parse structures and retain original strings.
2. Normalize salts, charges, tautomers, and stereochemistry under a versioned policy while retaining reversible links to originals.
3. Validate binary polarity and log-unit conversion with fixed unit tests.
4. Type exact, censored, interval, and qualitative observations.
5. Resolve duplicate groups at the compound-protocol level.
6. Exclude prediction-looking columns from both features and targets.
7. Split by standardized parent structure, scaffold, time/source, and large-molecule domain before any model selection.
8. Fit all preprocessing, calibration, and uncertainty models inside training folds.
9. Publish attrition, conflict, and coverage tables with every benchmark.

## Minimum benchmark reporting

Classification requires ROC-AUC and PR-AUC with confidence intervals, balanced accuracy, sensitivity, specificity, MCC, calibration slope/intercept, Brier score, reliability curves, and decision-curve or threshold-utility results. Regression requires MAE, median absolute error, RMSE, R-squared, Spearman correlation, signed bias, fraction within 0.5 and 1 log unit, and prediction-interval coverage. Every metric should be stratified by source/protocol, molecular-weight band, formal-charge state, scaffold novelty, nearest-neighbor similarity, and conformational-complexity band.

The primary success criterion is not recovery of the random-split headline result. It is honest performance, calibration, and failure localization when chemistry and mechanism move toward the intended large-molecule domain.
