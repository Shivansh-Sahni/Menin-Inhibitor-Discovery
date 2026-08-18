# Wild-type hERG first-paper implementation report

**Lead review date:** 2026-08-07  
**Status:** non-HPC data, task, protocol, analysis, benchmark, and CPU-diagnostic
infrastructure implemented; no predictive-superiority claim and no clinical
model claim.

## Executive result

The existing large hERG build was preserved. New versioned layers now make the
project explicitly wild-type-only, separate incompatible evidence into five
model contracts, classify measurement technology, automation, dose design, and
reported protocol evidence, preserve QT/QTc as a separate clinical phenotype,
and define a fair comparison against 15 recent model/method systems. Every
paper-facing derivative is physically bound to its exact upstream files.

The admitted paper scope contains **407,698 observations across 369,546 structures**:

- **343,909 confirmed wild-type** observations from PubChem AID 720551;
- **63,789 wild-type-or-unspecified** observations retained for scale but never called confirmed wild type; and
- **258 explicit mutant/variant** ChEMBL observations physically excluded to a separate audit artifact.

An independent metadata audit found mutation text in the same 258 records and no extra mutation-text hits among the variant-unspecified records. This supports the implemented permissive rule: exclude explicit mutants, retain unknown variant status with a visible uncertainty label.

## Implemented model-task ladder

| Contract | Eligible scale | What it predicts | Proper role |
|---|---:|---|---|
| Q0 weak fixed-dose binary | 339,373 structures | AID720551 automated FluxOR activity call | Scale-first molecular baseline/pretraining; not IC50 |
| Q1 quantitative pIC50 | 23,186 observations / 22,081 structures | pIC50 plus blocker/gray/nonblocker ordinal zone | Quantitative molecular modeling |
| Q2 functional assay-aware | 3,597 retained observations; 3,548 eligible / 2,776 structures | Native functional endpoints, exact/approximate/censored IC50 separately | Higher-quality assay-aware fine-tuning and evaluation |
| C0 clinical-development context | 3,056 structures | Development/regulatory context only | Stratification or unlabeled domain analysis; never a hERG label |
| C1 QT/QTc context | 221 endpoints / 5,605 result-or-denominator records | Human QT/QTc context | Held-out translation analysis; never a hERG label |

All molecular contracts use one fixed whole-scaffold partition rule. Explicit mutants fail closed. Source-declared splits remain provenance only. The C0/C1 schemas and validators technically prohibit direct hERG-label or training-label use.

Q0 retains 734 non-training structures in its file so exclusions remain
auditable: 710 inconclusive-only structures and 24 structures with conflicting
fixed-dose outcomes. Q2 retains 49 excluded rows: seven clinical-QT phenotype
rows, six IC50 rows with missing native relation, 13 rows without a numeric
native endpoint, and 23 rows without a usable structure. Its eligible rows
comprise 3,321 auxiliary functional observations, 218 exact-or-approximate
regression observations, and nine genuinely censored observations. The
combined exclusion ledger contains 1,041 rows, including all 258 mutants.

## Measurement method and automation

The method index covers every admitted observation and keeps modality, automation, and dose design as separate axes.

| Measurement modality | Observations |
|---|---:|
| High-throughput thallium flux | 344,029 |
| Patch-clamp electrophysiology | 16,200 |
| Radioligand binding | 9,787 |
| Binding, method unresolved | 10,094 |
| Functional electrophysiology | 1,667 |
| Functional ion flux | 444 |
| Functional, method unresolved | 816 |
| Curated clinical-QT/QTc phenotype assay | 7 |
| Technology unresolved | 24,654 |

Automation evidence is **354,057 automated**, **579 explicitly
manual/conventional**, **53,055 unresolved**, and **7 not applicable**. These
totals must not be interpreted as a causal automated-versus-manual comparison:
automated records are dominated by the single 343,909-row qHTS source. The
paper should prioritize within-source, matched-structure, source-adjusted, and
leave-one-modality-out comparisons. A lead audit corrected an early
keyword-only overcall: 30 in-vitro hERG assays merely mentioned QT in
background text and are no longer classified as clinical phenotypes.

Within eligible Q2, the model-relevant technology strata are 854 automated
patch-clamp, 34 explicit manual patch-clamp, 22 patch-clamp records with
unresolved automation, and 2,638 functional-technology-unresolved records. All
six optical/flux rows in the retained file are ineligible under the final task
contract. Q1 contains 22,969 compiled records whose original assay technology
is unresolved. This missingness is a result, not a reason to invent method
labels.

## QT/QTc route

Dr. Wang's QT prolongation comment is implemented as a downstream human translation axis, not as another name for hERG block. The exact structure-linked inventory contains **221 endpoints, 95 structures, 143 trials, 3,828 result records (3,819 numeric), and 1,777 denominator records**.

There are 73 interval endpoints and 148 event/threshold endpoints. Correction-method evidence includes 92 QTcF-only, 10 QTcB-only, 44 containing both, one QTcI, and 74 unresolved. Native values, units, time frames, arms/result JSON, denominators, NCT IDs, source paths, and JSON pointers are preserved. Nothing converts QT to IC50/pIC50, and missing QT reporting is never interpreted as safety.

The scientifically defensible analysis is: train molecular hERG models only on assay evidence; generate blinded predictions for the linked drugs; later combine hERG potency with unbound exposure/PK and major-metabolite evidence; analyze QT/QTc concordance and audited counterexamples. The current 95-structure overlap is valuable for external context but too small and confounded to justify a standalone deep clinical model.

## Existing-model comparison and their main weaknesses

The updated literature audit compares 15 model/method systems, including
2025–2026 transformer, pretrained, ensemble, large-HTS, multitask, and
censor-aware directions, across source data, label definitions, assay
handling, features, architecture, splits, metrics, availability, licensing,
and reproducibility. Fourteen benchmark/adaptation priorities are separately
ranked for local implementation.

Important findings:

1. Many apparent competitors reuse a few dataset families: Ogura/Ylipää/XGB-ISE, DeepHIT/CardioTox, and BayeshERG/AttenhERG are not fully independent evidence universes.
2. Random or inherited splits can place analogs, source signatures, and mirrored measurements on both sides of evaluation.
3. Several large classifiers harmonize IC50, EC50, Ki, Kd, and percent inhibition into one label, losing assay semantics.
4. High-throughput negatives and confirmatory actives often have asymmetric evidence quality.
5. Tiny 41–157-compound external panels are repeatedly reused and cannot alone establish superiority.
6. Extreme imbalance makes accuracy and AUROC insufficient; PR-AUC, precision/recall at a declared review budget, enrichment, calibration, coverage, and natural-prevalence tests are required.
7. Some published implementations rely on commercial descriptors/docking or have incomplete/unclear artifact licenses.
8. Attention/SHAP/fingerprint maps explain a model's behavior, not biological causality.

The project has six frozen comparison challenges: WT-Flux, quantitative pIC50, quality-level/learning-curve, assay-modality transfer, source/time/domain shift, and QT translation. Published headline metrics will not be compared directly against local metrics unless a model is reproduced on identical structures, labels, preprocessing, and splits.

## CPU diagnostic baseline: honest result

A fixed Morgan-radius-2/1,024-bit SGD logistic baseline was trained on the Q0 scaffold split. It fitted 265,625 training structures, selected its threshold only on 32,850 validation structures, and was evaluated once on 40,898 locked test structures. The test set contains only 148 positives.

Locked-test results were:

- ROC-AUC **0.662**;
- average precision **0.00806**, versus natural positive prevalence **0.00362**;
- balanced accuracy **0.644**;
- MCC **0.0405**;
- 78 true positives, 70 false negatives, 9,733 false positives, and 31,017 true negatives.

This is better than a prior-only baseline as a ranking diagnostic, but it is not a useful production classifier: precision is extremely low, class-weighted fitting produced poor natural-prevalence calibration, and SGD reached its fixed 100-iteration cap without declared convergence. The result exposes the difficulty of Q0 and is exactly why later claims must emphasize PR-AUC, calibration, assay quality, and usable operating points rather than dataset size or accuracy. It does **not** establish superiority.

Two further CPU diagnostics exercise the quantitative paths. On the locked Q1
test partition (2,275 exact structures), Morgan ridge gives MAE **0.582**,
RMSE **0.774**, R² **0.184**, and Spearman **0.432**. On the deliberately small
locked Q2 test partition (11 exact structures), censored-Gaussian descriptor
ridge gives MAE **1.142**, RMSE **1.616**, R² **0.094**, and Spearman **0.255**.
The Q2 target builder contains 123 IC50 structures: 114 exact `=` structures
and nine genuinely censored `<` structures. Approximate `~` measurements are
retained but never silently upgraded to exact. These results validate the
plumbing and provide honest floors; Q2 is far too small for a performance
claim.

## Analyses possible now

The content-bound descriptive analysis quantifies several paper-relevant
effects without making causal claims. Source and endpoint are almost perfectly
entangled (bias-corrected Cramér's V **0.998**, n=407,698), showing why naive
pooling can reward source recognition. Among 591 exact structures shared by the
large thallium-flux source and the quantitative release, binary consensus agrees
only **34.5%**; among 97 structures shared by ChEMBL and the quantitative
release, agreement is **97.9%**. Automation comparisons are now explicit: 614
structures support automated-versus-unresolved binary comparison, but only 28
support automated-versus-manual comparison, so broad causal claims would be
misleading. There are 843 structures with replicated exact pIC50 values; the
largest observed within-structure range is 5.0 log units and is an audit target,
not an automatic outlier. In Q0, logP is the strongest single prespecified
descriptor association (AUC 0.630; standardized mean difference 0.468), while
the largest train-to-validation/test marginal shift is 0.389 SD. These results
motivate assay-aware heads, matched-structure analyses, source balancing,
fundamental-feature ablations, and protocol-focused follow-up.

## Paper experiment sequence

1. Freeze all task files and audit competitor overlap at parent InChIKey, scaffold, measurement lineage, and preregistered near-neighbor thresholds.
2. Reproduce open ECFP baselines on identical splits: regularized logistic regression, RF, XGBoost/LightGBM, and SVM. Report fit coverage and failures.
3. Train D-MPNN and AttentiveFP baselines with the same tuning budget and five fixed seeds.
4. Compare Q0-only, Q1-only, Q2-only, Q0→Q1/Q2 transfer, and shared-encoder quality-specific heads. Use size-matched learning curves so quality is not just a proxy for sample size.
5. Run molecule-only versus molecule-plus-method models, modality-specific heads, source-balanced fitting, leave-one-modality-out transport, and matched-compound method disagreement.
6. Evaluate time/source/near-neighbor exclusions and frozen competitor artifacts separately from retrained competitor architectures.
7. Keep natural prevalence in locked tests and use paired scaffold/source-cluster bootstrap confidence intervals.
8. Add exposure-aware QT translation only after defensible PK/Cmax/free-fraction linkage is available.

Superiority may be claimed only for a named frozen challenge. For WT-Flux, the preregistered rule requires a paired PR-AUC confidence interval excluding zero, noninferior calibration, no material recall loss, consistent direction across seeds, and complete coverage accounting. A win on one challenge must never be generalized to “state-of-the-art hERG prediction.”

## Lead verification and artifact map

Key implementations:

- `pipeline/src/menin_discovery/platform_herg_wildtype_scope.py`
- `pipeline/src/menin_discovery/platform_herg_quality_tasks.py`
- `pipeline/src/menin_discovery/platform_herg_modality_qt.py`
- `pipeline/src/menin_discovery/platform_herg_paper_baseline.py`
- `pipeline/src/menin_discovery/platform_herg_quality_baselines.py`
- `pipeline/src/menin_discovery/platform_herg_master_dataset.py`
- `pipeline/src/menin_discovery/platform_herg_current_analysis.py`

Key generated artifacts:

- `research/data/platform/processed/herg_paper/wildtype_scope_v1/`
- `research/data/platform/processed/herg_hierarchy/v1_2_quality_tasks/`
- `research/data/platform/processed/herg_hierarchy/v1_2_modality_qt/`
- `research/data/platform/processed/herg_paper/cpu_baseline_v1/`
- `research/data/platform/processed/herg_paper/quality_baselines_v1/`
- `research/data/platform/processed/herg_hierarchy/v1_3_master/`
- `research/data/platform/processed/herg_hierarchy/v1_3_current_analysis/`
- `research/reports/platform/herg_paper/model_landscape_v2/`

The build is additive and preserves the v1/v1.1 hierarchy, split,
clinical-link, and operational-tier artifacts. Every generated data layer has
deterministic schemas, content hashes, declared input bindings, count
reconciliation, and a validator. The separately labeled AID720551 curated
protocol contract is hashed inside the master artifact and corroborated by the
source-profile report; that narrative report is not itself listed as a master
manifest input. This phase performed no large neural-model training and made
no clinical or predictive-superiority claim.

Final lead/subagent reconciliation, manifest hashes, corrected counts, completed
analyses, test evidence, storage disposition, and remaining limitations are
recorded in `HERG_FINAL_RECONCILIATION_2026_08_07.md` in this directory.
