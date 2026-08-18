# Evidence review: Sun, Wang, and Shen (2026)

## Citation and role in this project

Sun, Wang, and Shen, "Modeling hERG Channel Liability: From Structural Insight to Highly Accurate Qualitative and Quantitative Models," *Journal of Chemical Information and Modeling* 66 (2026), 7515-7523. DOI: `10.1021/acs.jcim.6c00163`.

This study and its workbook are useful as:

1. a broad public-data comparator for conventional hERG prediction;
2. a source of structures and labels that can be recurred independently after curation;
3. a concrete example of atom-type and correction-factor feature engineering; and
4. a warning about applicability-domain, provenance, protocol, and split limitations.

It is not evidence that a model trained mainly from 200-600 Da molecules transfers to the intended large-molecule domain. It also does not decompose measured hERG inhibition into access, channel-state selection, binding thermodynamics, binding kinetics, or assay-protocol effects.

## What the paper reports

### Data and targets

The article reports a curated, nonredundant classification collection of 8,333 molecules with molecular weight from 200 to 600 Da and a regression collection of 7,772 molecules. A post-2021 external validation collection contains 1,133 compounds under 600 Da. The classification boundary is 10 micromolar: blockers below the threshold and nonblockers at or above it. The workbook's binary coding is easy to invert accidentally: `0` denotes blocker and `1` denotes nonblocker/inactive.

The quantitative target is represented in the workbook as `log10(IC50 in nM)`. It is not conventional pIC50 even though it is a logarithmic potency target. Conversion to conventional molar pIC50 is:

`pIC50 = 9 - log10(IC50 in nM)`.

That transformation must be explicit in every downstream use.

### Representation and learners

The paper describes 221 atom types combined with physicochemical correction factors. It trains an RBF-kernel nu-SVC for classification and an RBF-kernel epsilon-SVR for regression. Hyperparameters are selected with five-fold cross-validation. Random train/test partitions from 90/10 through 50/50 are repeated ten times.

There is a documentation discrepancy that must remain open: the main article describes 40 correction factors, while the supplied supporting-information list visibly enumerates M1-M30. Without the executable feature generator or a complete mapping table, the published representation cannot be assumed to be exactly reproducible.

### Reported performance

For the primary classification analysis, the article reports ROC-AUC 0.88, accuracy 0.80, sensitivity 0.78, and specificity 0.83. Even at a 50/50 random split, reported ROC-AUC remains 0.85.

For regression, it reports test-set R-squared 0.63, average absolute error 0.383 log units, and RMSEP 0.548 log units. It reports 74.3% of predictions within 0.5 log unit and 93.2% within 1 log unit. The raw post-2021 external-set average absolute error is 0.59 log unit. After correcting ten ChEMBL entries described as 1,000-fold unit errors, the reported value is 0.50 log unit.

These are useful benchmark numbers, but the random-split design mostly measures interpolation across a broad public chemical collection. It is not a prospective large-molecule, scaffold-novel, protocol-controlled validation.

## Direct audit of the attached workbook

The following findings were calculated from the attached workbook and are not claims copied from the paper:

- The classification sheet contains 8,334 rows, of which 8,310 are unique after canonical structure normalization.
- It contains 3,914 class-0 blockers and 4,420 class-1 nonblockers or qualitative inactives.
- Only 5,980 classification rows have numeric IC50 values. The other 2,354 are class-1 qualitative/missing-value records and should not be silently converted to exact concentrations.
- The regression sheet contains 7,772 valid, canonically unique structures.
- The external validation sheet has 1,133 rows but 1,097 canonically unique structures. There are 29 duplicate groups representing 36 extra rows, and 25 groups contain differing numeric values.
- The classification sheet has 24 duplicate groups; 12 contain differing values and two cross the binary class boundary.
- One standardized structure overlaps between the regression and external-validation sheets.
- The workbook column named `hERG` appears to contain predictions. It must not be treated as an out-of-fold training label unless row-level split and prediction provenance are supplied.

The article describes a 200-600 Da range, but molecular weights recalculated from the workbook structures find 173 classification and 190 regression rows outside that interval. In the classification sheet, 58 structures are at least 650 Da, 28 are at least 700 Da, and 13 are at least 750 Da. The corresponding blocker/nonblocker counts are 28/30, 9/19, and 3/10. These tails are too small and too potentially idiosyncratic to validate a large-molecule model.

The current internal hERG set is chemically remote from this workbook. Using radius-2, 2,048-bit Morgan fingerprints after the same standardization, the maximum nearest-neighbor similarity across the internal set is 0.309 overall and 0.252 when the comparator is restricted to workbook molecules of at least 650 Da. None reaches 0.5. The published data can therefore serve as pretraining or contextual evidence only with explicit domain-shift controls; it cannot be treated as matched support for the internal chemistry.

## Mechanistic interpretation

The atom types and correction factors can encode useful correlates of cationic centers, aromatic surface, hydrophobicity, and molecular size. Those correlates are downstream shadows of several different physical processes:

1. solution microstate populations determine which species are available;
2. membrane partition and intracellular access set the free concentration near the channel;
3. channel gating and selectivity-filter state determine receptor availability;
4. ligand and receptor conformational ensembles determine encounter and bound-state populations;
5. binding free energies and kinetic barriers determine occupancy, residence time, and trapping;
6. voltage-clamp protocol, temperature, expression system, and exposure time map occupancy to an apparent IC50.

A single structure-to-IC50 kernel model collapses this causal chain. High predictive accuracy within a familiar domain does not show which process caused a prediction, and an atom-type contribution should not be described as a binding mechanism unless independently connected to structural, kinetic, or perturbational evidence.

## How to use this precedent without copying its weaknesses

- Recurate the structures under row-level provenance and assay-context rules; preserve qualitative and censored labels.
- Treat the published model family as a conventional comparator, alongside fingerprints, physicochemical descriptors, graph neural networks, and conformer-aware baselines.
- Evaluate random, scaffold, temporal, source-held-out, and molecular-weight/shape extrapolation splits. Primary claims should rely on the latter four.
- Report ROC-AUC, PR-AUC, balanced accuracy, sensitivity and specificity at declared thresholds, Brier score, calibration intercept/slope, expected calibration error, regression MAE/RMSE/R-squared/Spearman, interval coverage, and applicability-stratified errors.
- Build separate observation models for protocol-defined IC50 values rather than pooling incompatible assays without context.
- Use the mechanistic layer to explain residuals and cross-domain failures: microstate distributions, membrane access, receptor-state ensemble interactions, and kinetic features should be compared against the conventional model under ablation.
- Inflate uncertainty or abstain when chemical, conformational, mechanistic, or protocol distance exceeds the supported domain.

## Bottom line

The paper establishes a strong conventional interpolation benchmark and supplies a valuable large public workbook. The attached evidence does not establish reproducible end-to-end implementation, independent large-molecule validation, or a causal hERG mechanism. Its highest value here is as a carefully audited comparator and data source against which a process-resolved model must demonstrate improved transportability, calibration, and mechanistic consistency rather than merely a higher random-split score.
