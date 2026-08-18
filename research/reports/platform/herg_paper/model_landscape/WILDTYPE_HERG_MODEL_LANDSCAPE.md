# Wild-type hERG model landscape and fair superiority protocol

**Status:** literature/model audit and preregistration design; no model has been trained or compared in this workstream.  
**Scope:** human wild-type hERG/KCNH2 small-molecule inhibition, assay effects, and downstream QT translation. Explicit mutants are out of scope.  
**Retrieved:** 2026-08-07. Sources are primary papers, official repositories, official assay records, and FDA/ICH guidance.

## Executive decision

The first paper should not be framed as “we used more rows, therefore our model is superior.” It should test a sharper claim: **preserving wild-type target identity, native assay semantics, data lineage, evidence quality, and hard out-of-domain splits produces predictions that transport better than models trained on pooled labels.** This is a plausible superiority hypothesis, not an achieved result.

The local foundation is unusually large but difficult: **407,956 observations**, including **343,909 confirmed wild-type PubChem observations**, **63,789 wild-type-or-unspecified observations**, and **258 explicit mutant ChEMBL observations**. The mutants must be excluded from every paper model. The current bare binary table has **339,373 structures but only 1,238 positives (0.365%)**; its scale is an advantage for chemical coverage, while its class imbalance and fixed-dose FluxOR semantics are major limitations. The quantitative table has **23,186 observations**. These are different tasks and should remain different targets.

Recommended paper models:

1. **WT-Flux model:** large weak-label wild-type model for the AID720551 fixed-dose FluxOR call.
2. **WT-pIC50 model:** exact functional IC50/pIC50 regression, with censoring and replicate metadata preserved.
3. **Assay-aware model:** manual patch clamp, automated patch clamp, flux, binding, and unresolved modalities represented by separate heads or explicit assay tokens.
4. **Quality-level models:** identical architecture trained separately at each evidence level, plus a pretrain/fine-tune model. This is an analysis, not the main deployment model.
5. **QT translation analysis:** predicted hERG potency plus exposure/PK is evaluated against clinical QT/QTc; QT is not turned into a hERG label.

## What the field actually contains

The apparent model diversity hides substantial dataset reuse.

- The 2018 integrated database combined ChEMBL, PubChem/NCGC, commercial GOSTAR, and hERGCentral and explicitly investigated protocol-driven disagreement ([Sato et al.](https://doi.org/10.1371/journal.pone.0199348)). Its 291,219-compound classifier view became the basis of the [Ogura SVM](https://doi.org/10.1038/s41598-019-47536-3), the large GNN/XGBoost comparison by [Ylipää et al.](https://doi.org/10.1016/j.crtox.2023.100121), D-MPNN comparisons, and the 2025 XGBoost/ISE model. These are not independent experimental universes.
- DeepHIT assembled 14,440 structures from BindingDB, ChEMBL, literature, and in-house data, then tested on only 44 dissimilar patch-clamp compounds ([DeepHIT](https://doi.org/10.1093/bioinformatics/btaa075)). CardioTox reused the DeepHIT compilation without the in-house component ([CardioTox net](https://doi.org/10.1186/s13321-021-00541-z)).
- BayeshERG's fine-tuning set contains 14,322 compounds and uses 304,045 10-uM inhibition measurements for pretraining ([BayeshERG](https://doi.org/10.1093/bib/bbac211)). AttenhERG uses the same 14,322 total and the same small external-set family ([AttenhERG](https://doi.org/10.1186/s13321-024-00940-y)). Consequently, a paper's “external” set may already have influenced an earlier model, threshold, or architecture.
- HERGAI is genuinely large—299,927 molecules—but has only 1,937 blockers. It uses a 75/25 Bemis–Murcko group split and PLEC fingerprints from selected docking poses ([HERGAI](https://doi.org/10.1186/s13321-025-01063-8)). Its released train/test sets and workflow are useful comparators, but most of its scale is the same public high-throughput negative universe represented locally.

The complete machine-readable comparison, including exact train/test sizes, definitions, features, licenses, and known defects, is in `model_comparison_matrix.csv` and `model_comparison_matrix.json`.

## Existing feature and model families

### Classical ligand-based models

ECFP/Morgan fingerprints, molecular descriptors, RF, SVM, LightGBM, and XGBoost remain serious baselines. Pred-hERG 5.0 reports binary test BACC 0.86/MCC 0.54/AUC 0.81 and pIC50 test R2 0.61/RMSE 0.44, although it uses a stratified random 80/20 split and pools SP, HEK293, and CHO IC50 ([Pred-hERG 5.0](https://doi.org/10.1021/acs.chemrestox.3c00400)). The 2020 prospective study found XGBoost plus RDKit descriptors strongest among its individual approaches on an approximately 840-compound same-assay prospective set ([Siramshetty et al.](https://doi.org/10.1021/acs.jcim.0c00884)). A deep model that does not beat tuned XGBoost/ECFP under identical splits has not established an advance.

### Graph and sequence models

DeepHIT ensembles descriptor DNN, fingerprint DNN, and graph GCN predictions to raise sensitivity, at the expense of specificity. BayeshERG combines a D-MPNN, point-inhibition transfer learning, Monte Carlo dropout, and global multihead attention. AttenhERG adds atom- and molecule-level attention and uncertainty rejection. A D-MPNN plus 206 commercial MOE descriptors reported strong performance, including AUC 0.922 on a smaller scaffold split and AUC 0.968 on the inherited Ogura split ([Shan et al.](https://doi.org/10.1039/d1ra07956e)). These results justify graph baselines, but not the assumption that graph networks are automatically superior.

### Structure-based models

A docking-score/interaction-fingerprint SVM reached maximum AUC 0.86 across repeated balanced subsets of a curated 8,337-entry ChEMBL set ([Creanza et al.](https://doi.org/10.1021/acs.jcim.1c00744)). HERGAI uses docking-derived PLEC fingerprints and a stacking ensemble. Structure features can aid mechanism and enrichment, but wild-type hERG is constant across molecules: the protein input does not confer cross-protein generalization, and results can depend on receptor conformation, docking, and pose selection. Our fair comparison should therefore include ligand-only and structure-based models on the same frozen molecules.

### Quantitative models

Quantitative prediction is more useful than a single arbitrary cutoff. hERGBoost reports external R2 0.394 and RMSE 0.616 after aggregating ChEMBL, BindingDB, and six publications ([hERGBoost](https://doi.org/10.1016/j.compbiomed.2024.109416)). Pred-hERG reports stronger internal/random-split regression. These numbers cannot be directly ranked because their compounds, assays, curation, and split hardness differ. Our quantitative comparator must use one frozen pIC50 benchmark and the same parent/scaffold/source partitions for every model.

## Flaws our paper can test rather than merely assert

1. **Analog and source leakage.** Random splits allow near-identical chemistry and records from the same assay/document into train and test. Even a scaffold split must be audited for exact parent, same scaffold, and high-Tanimoto neighbors.
2. **Database-wrapper leakage.** ChEMBL, BindingDB, PubChem mirrors, TDC wrappers, EPA reprocessing, and literature compilations can contain the same experiment. “Different database” is not automatically independent evidence.
3. **Assay pooling.** The Ogura/Ylipää lineage harmonized IC50, EC50, ED50, Ki, Kd, and percent inhibition into a classifier target. Ylipää explicitly identifies this as a data-quality limitation. Pred-hERG narrows to IC50 but still pools cell/assay contexts.
4. **Asymmetric evidence.** Large screens often demand confirmatory curves for actives but call many negatives from primary or fixed-dose screens. A model can learn selection policy and source chemistry rather than channel pharmacology.
5. **Imbalance and misleading metrics.** Accuracy of 99.6% is achievable locally by predicting every structure inactive. AUROC can also appear strong while precision is unusable. PR-AUC, enrichment, recall, calibration, and precision at a fixed review budget are mandatory.
6. **Tiny, reused external sets.** The 44-compound DeepHIT set and related 41/157-compound sets are repeatedly reused. They are useful compatibility panels, not decisive evidence of superiority.
7. **Unresolved target variant.** Many literature tables say only “hERG.” Lack of a documented mutant filter is not proof that mutants are included, but it prevents a strict wild-type claim. Our 258 explicit mutants can be removed deterministically; wild-type-or-unspecified rows remain a disclosed sensitivity stratum.
8. **Interpretability overclaim.** Attention, SHAP, and fingerprint contribution maps explain model behavior, not biological causality. Validate proposed liability motifs with matched pairs or prospective measurements.
9. **Availability and licensing.** The audited official repositories show uneven reproducibility: AttenhERG (MIT) and BayeshERG (MIT code, restricted weights/derived artifacts) are runnable; CardioTox has no detected top-level license; the current Pred-hERG repository contains notebooks but placeholder data/model directories; HERGAI is MIT but inherited measurement terms still apply. Repository commits are recorded in the matrix.
10. **hERG/QT conflation.** A molecular hERG block prediction is not a clinical QT or torsade prediction. Exposure, metabolites, protein binding, heart rate, and other ion channels intervene.

## Measurement technology analysis

Create a controlled `measurement_modality` field with: `manual_patch_clamp`, `automated_patch_clamp`, `flux_fluorescence`, `radioligand_binding`, `other_functional`, and `unresolved`. Also retain cell line, platform, voltage protocol, temperature, incubation time, nominal/measured concentration, endpoint, relation/censoring, and source assay ID.

This is scientifically substantive. Cross-site automated patch-clamp IC50 variation has been reported even under recommended protocols, including a 10.4-fold interquartile span for dofetilide in one study ([Li et al.](https://pubmed.ncbi.nlm.nih.gov/32221320/)). A 2025 standardized manual-patch multi-laboratory study still observed systematic potency differences ([Alvarez Baron et al.](https://doi.org/10.1038/s41598-025-15761-8)). The conclusion is not that automated data are “bad”; it is that platform/protocol effects should be modeled and reported.

Run three analyses:

- paired-compound modality disagreement and Bland–Altman analysis where the same parent structure has comparable quantitative results;
- leave-one-modality-out transport tests;
- identical molecular models with versus without assay metadata, plus separate-head multitask learning.

Do not infer a manual/automated label from the words “cell-based” or “functional.” Unresolved rows stay unresolved.

## QT prolongation route

Interpret Dr. Wang's QT comment as a **translation layer**. FDA's final E14/S7B Q&A calls for integrated nonclinical and clinical QT/QTc assessment, including hERG safety margin and concentration-response ECG evidence ([FDA/ICH guidance, August 2022](https://www.fda.gov/regulatory-information/search-fda-guidance-documents/e14-and-s7b-clinical-and-nonclinical-evaluation-qtqtc-interval-prolongation-and-proarrhythmic)). Therefore:

- keep hERG IC50/pIC50, QT/QTc, torsade, and general cardiotoxicity as separate endpoints;
- standardize clinical QT to the reported correction (prefer QTcF when present), baseline/placebo adjustment, dose, time, arm, denominator, and concentration;
- test whether predicted hERG potency or the hERG safety margin (`IC50 / unbound Cmax`) adds held-out-drug value beyond exposure alone;
- later add major metabolites and other cardiac currents; do not interpret missing QT reporting as a negative result.

This can yield a meaningful paper section even if the clinical set is too small to train a standalone deep model: association, calibration, concordance, and carefully audited counterexamples are publishable.

## Fair benchmark and claim language

The frozen protocol is in `superiority_evaluation_matrix.csv`. Nonnegotiable rules are:

- same structures, labels, preprocessing, splits, tuning budget, seeds, and metric code for all trainable models;
- compare both retrained open baselines and frozen original competitor artifacts; never mix those two claims;
- remove competitor-training overlap from any supposedly external test at parent-InChIKey, scaffold, and preregistered near-neighbor levels;
- keep natural prevalence in final tests;
- report five seeds and paired scaffold/source-cluster bootstrap confidence intervals;
- report failures and coverage: an unavailable prediction is not silently discarded;
- stratify by source, modality, evidence quality, chemical similarity, and potency band;
- publish all thresholds before examining final-test performance.

Minimum open baselines are logistic regression, RF, and XGBoost on ECFP4; an SVM on ECFP4; D-MPNN; and AttentiveFP. Add descriptor models and a calibrated assay-aware multitask model. HERGAI, BayeshERG, AttenhERG, CardioTox, DeepHIT, and Pred-hERG should be run as frozen tools only where licensing, installation, input coverage, and versioning permit.

Use “outperformed on the frozen WT-Flux benchmark” only if the paired PR-AUC confidence interval excludes zero, calibration is noninferior, recall does not materially decline, and the direction holds across seeds. Use separate language for pIC50, modality transfer, and QT translation. Never generalize one win to “state of the art hERG prediction.”

## Immediate implementation order

1. Freeze the wild-type exclusion rule and regenerate every training view without the 258 explicit mutants.
2. Build the assay dictionary and resolve the highest-volume ChEMBL assays first; leave unknowns unknown.
3. Freeze WT-Flux, WT-pIC50, source-holdout, modality-holdout, time-holdout, and QT-translation panels before training.
4. Audit every competitor and every external panel for parent/scaffold/near-neighbor and measurement-lineage overlap.
5. Reproduce open classical baselines first; then graph models; then quality-specific and assay-aware models.
6. Run learning curves and size-matched quality experiments so “quality” is not merely a proxy for sample count.
7. Add QT only as a separate, exposure-aware external analysis.
8. Claim superiority only after the preregistered evaluation succeeds.

## Repository audit record

Official repositories were inspected at these commits on 2026-08-07: BayeshERG `25e9466499905a952f9d41cc6bc6886c3f247acb`; CardioTox `6096ef004016f82a64df99e5df8c1133d7092550`; Pred-hERG `7e11ce78014f00f6fc822a1b00b2e118fd03cde9`; AttenhERG `4b6ce4dffce336d55b88769a96eac5fd2259f1e5`; HERGAI `21f1b0ef34ab8c818015f1ac6bdfd6c8e1bff351`. This records observed availability, not an endorsement or relicensing of bundled data.

