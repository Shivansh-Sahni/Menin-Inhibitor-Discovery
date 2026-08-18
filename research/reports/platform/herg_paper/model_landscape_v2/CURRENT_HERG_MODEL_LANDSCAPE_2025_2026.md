# Current hERG model landscape, 2025-2026

## Bottom line

The defensible project advantage is not yet a higher AUC. It is a substantially richer and more auditable wild-type hERG data substrate: 407,698 admitted observations mapped to 369,546 unique structures, including 343,909 confirmed-wild-type observations and 63,789 wild-type-or-unspecified observations, explicit exclusion of 258 mutant observations, quality tiers, source lineage, endpoint semantics, assay modality, automation status, and clinical/QT linkage. Recent models rarely preserve all of those dimensions. That is an established superiority of the data asset and experimental design.

Predictive superiority remains a hypothesis. It must be demonstrated by running all models on the same frozen parent-structure/scaffold/source/temporal partitions, with the same endpoint definitions and no test-set selection. Paper-reported AUC values from random splits, balanced 100-compound panels, or selective-coverage subsets cannot establish that hypothesis.

## What changed in 2025-2026

Recent work moved in five useful directions:

1. **Feature and representation fusion.** MaxQSARing searches fingerprints, descriptors, and pretrained representations; Transformer_Morgan and hERGAT combine SMILES/attention with Morgan fingerprints; hERG-MFFGNN and TDMFLSGAT fuse graphs, fingerprints, and sequence encoders.
2. **Extreme-imbalance handling.** XGB+ISE uses 29 balanced XGBoost learners and an isometric stratified ensemble map; HERGAI uses structure-derived PLEC features and stacked models on a roughly 0.65% active public-screening universe.
3. **Quantitative prediction.** hERGBoost predicts continuous IC50, while hERG-MFFGNN remains binary. Censored-likelihood work outside hERG now supplies a practical Tobit/interval-loss adapter.
4. **Multi-ion-channel modeling.** MultiCTox, CToxPred2, Mixture-of-Experts cardiotoxicity, and the 2026 CardioSafe preprint model hERG with Nav1.5/Cav1.2 or broader adverse-outcome endpoints.
5. **Leakage and applicability-domain awareness.** Scaffold splits, low-similarity external sets, ISE coverage, and CardioSafe's post-publication split audit are progress, though not uniformly applied.

None of those directions makes the local project obsolete. CardioSafe is the strongest current warning against novelty overstatement: its preprint covers 334,444 compounds and 331,127 hERG labels, performs explicit similarity/deleakage audits, and includes multi-channel and censored quantitative heads. Our honest distinction is observation-level WT/variant and assay-quality semantics—not merely scale, multimodality, censorship, or the phrase “multi-ion-channel.”

## Comparator findings

| Model | Exact audited cohort/split | Headline result, as reported | What the number does not prove |
|---|---:|---|---|
| MaxQSARing (2025) | 12,620 development; external I 740 (30/710), external II 116 (53/63) | scaffold-CV MCC 0.608; external-I MCC 0.232 | A ten-feature decision tree reached MCC 0.248 on external I; pretrained representations were not consistently superior and external II is tiny. |
| Transformer_Morgan (2025) | paper says 22,247 standardized; split arithmetic says 17,797 + 4,461 = 22,258; external 100, balanced 50/50 | random-test AUC 0.9347; external AUC 0.93 | Random splitting and a deliberately balanced 100-compound panel do not estimate deployment performance. The encoder is trained for this task, not a pretrained molecular language model. |
| XGB+ISE (2025) | official SI: 290,731 total; train 183,082, internal 20,343, external 87,306 | full external MCC about 0.41; chosen ISE strata MCC 0.72 at 64% coverage | The best headline is selective prediction on 55,885/87,306 compounds. The internal set also informed ISE bins and descriptor selection. |
| HERGAI (2025) | 299,927 total: 1,937 blockers, 297,990 nonblockers; test 74,982 | DNN stack AUC 0.863, balanced accuracy 0.758 | Docking/pose generation is required; actives are only 0.646%; pooled screening labels lack our WT and assay semantics. |
| hERGBoost (2025 volume) | curated quantitative study set 10,798; external 706 | external R2 0.394, RMSE 0.616, MAE 0.438 | Error deteriorates for extreme potency: RMSE 1.154 for 113 compounds with logIC50 below 0 versus 0.445 for the other 593. |
| hERGAT (2025) | 23,381 unique: 14,183 blocker, 9,198 nonblocker; random 80/10/10 | paper test AUC 0.907; official repository currently reports 0.94946 | Random, high-prevalence splitting is easier than chemical/source holdout. Repository and paper result versions differ. |
| hERG-MFFGNN (2025) | official repo benchmark 14,322 (8,488/5,834); externals 44, 157, 41, 740 | paper AUC 0.909, accuracy 0.854 | Released code uses one fixed random test fold, evaluates it every epoch, and saves a best-test checkpoint. That invalidates using repository test selection as a clean benchmark. |
| TDMFLSGAT (2026) | official workbook 10,355; archived test sheet empty; script also contains an 8,284/2,071 fixed split | fivefold-CV AUC 0.901, AP 0.915, MCC 0.641 | The archive is incomplete/nonportable: hard-coded absolute paths, missing generated feature files, and a mismatch between paper CV and fixed-split code paths. |
| MultiCTox (2025) | official repo: hERG 22,246 development (17,796/4,450), externals 250 and 473; Nav1.5 2,069; Cav1.2 802 | paper reports improvement over prior methods | hERG scale is modest and secondary-channel cohorts are tiny; split is not a scaffold/source stress test; no explicit license was found in the audited snapshot. |
| CardioSafe v1.1 (2026 preprint) | 334,444 total; hERG 331,127 (11,881 blockers); tan70 241,790/46,328/46,326 | v1.1 hERG AUC 0.9085, MCC 0.4385; quantitative test Pearson r 0.5422 | It is a preprint. v1.0 contained 12/15 train-to-validation similarity violations that v1.1 fixed, showing why exact artifact versioning matters. End-to-end training data loaders/caches remain incomplete. |

Detailed fields and primary links are in `model_comparison_matrix.csv` and `.json`.

## Recurring flaws in existing models

### 1. Endpoint collapse

Most classifiers reduce heterogeneous hERG evidence to one binary label at or around 10 micromolar. Binding, flux, automated patch clamp, manual patch clamp, fixed-dose inhibition, IC50, and literature calls are not interchangeable measurements. Pooling them can enlarge a dataset while degrading the scientific meaning of the target.

**Our established advantage:** the local observation table preserves endpoint, relation, units, modality, automation, source, and WT/variant disposition. The model should exploit those fields as strata or auxiliary tasks, not discard them.

### 2. Split optimism

Random splits dominate Transformer_Morgan, hERGAT, large 2023 HTS benchmarking, and repository implementations. Close analogs and source families can cross partitions. This can yield an AUC around 0.9 while saying little about new-scaffold or new-source generalization.

**Required fix:** freeze parent-structure deduplication before splitting, then publish scaffold, source, temporal, low-similarity, and assay-modality holdouts. Never select epochs or thresholds on test performance.

### 3. Prevalence distortion

External panels are often balanced or enriched. Transformer_Morgan's 100-compound panel is 50/50, while real public screening can be below 1% active. ROC-AUC remains useful but hides screening burden. Accuracy can be meaningless.

**Required fix:** report average precision with its prevalence baseline, MCC, sensitivity at fixed false-positive rates, enrichment, calibration, and prospective workload. For selective models, report both performance and coverage.

### 4. Assay and wild-type ambiguity

The audited papers generally do not build explicit wild-type-only cohorts or quantify automatic versus manual measurement effects. Pooled databases usually leave protocol, temperature, cell system, expression construct, and assay technology unresolved.

**Our established advantage:** explicit mutant exclusion and assay metadata allow the Dr. Wang analyses—quality-tier-specific models and automation/modality effects—to be tested directly. Missingness must remain visible; unknown must not be relabeled as wild type or manual.

### 5. Incomplete reproducibility

Code availability does not guarantee a frozen, end-to-end reproduction. Archived repositories contain version drift, hard-coded paths, absent generated features, missing licenses, external downloads, and test-aware checkpointing. A web server alone is not a benchmark artifact.

**Required fix:** vendor only license-compatible adapters; record commit hashes and checksums; freeze conda/container environments; add one-command inference; retain raw predictions and audit logs.

### 6. Interpretation without mechanism

SHAP on a surrogate XGBoost model does not explain a Transformer. Attention weights are not causal explanations. Docking-derived contacts are more physically grounded but depend strongly on pose selection and one channel conformation.

**Required fix:** compare learned attributions with stable physicochemical descriptors, matched molecular-pair effects, uncertainty, applicability domain, and—where affordable—docking/interaction features. Physics should be evaluated as incremental evidence, not asserted as mechanistic truth.

## What can be claimed now

### Established and safe to emphasize

- **Wild-type scope:** 343,909 confirmed-WT observations plus 63,789 WT-or-unspecified observations, with 258 mutant observations explicitly excluded in the audited local snapshot.
- **Observation and structure scale:** 407,698 admitted observations and 369,546 unique structures. The structure count is nominally larger than CardioSafe's 331,127 hERG-labeled compounds, but the endpoint/admission definitions differ, so this is a scale comparison—not a matched-quality or predictive-superiority result.
- **Evidence stratification:** Q0 339,373 structures, Q1 23,186 quantitative observations, and Q2 3,548 functional-training-eligible observations (3,597 retained) in the audited snapshot.
- **Assay visibility:** 344,029 flux, 16,200 patch-clamp, and 9,787 binding records; automation status includes 354,057 automated, 579 manual, 53,055 unresolved, and seven not-applicable clinical-phenotype records.
- **Research design:** quality-tier models, automation/modality analysis, clinical/QT evidence linkage, and provenance-preserving disagreement analysis are planned directly from available fields.

### Not established and must not be claimed yet

- Higher AUC, AP, MCC, calibration, or prospective hit rate than any recent model.
- Better new-scaffold generalization.
- A causal explanation of hERG blockade.
- Clinical QT or torsades prediction from hERG structure alone.
- Uniqueness of large-scale, multimodal, multi-ion-channel, censored, or leakage-audited modeling.

The paper should say that the project is **designed to test** whether richer WT- and assay-aware evidence improves robust generalization; it should not say that it already has.

## Benchmark that can establish predictive superiority

### Frozen tasks

1. **Broad WT screening classification:** one label per parent structure with prespecified conflict rules; Q0 is retained as a broad/noisy training tier, never represented as equivalent to Q2.
2. **Functional hERG classification:** patch-clamp/functional endpoints at a preregistered threshold, with binding and flux evaluated separately.
3. **Quantitative pIC50:** exact values only, relation-aware; do not convert `>`, `<`, or intervals to false point values.
4. **Assay-aware multitask learning:** endpoint/modality/source heads or covariates; assess transfer without leaking structure families.
5. **Quality-tier experiments:** train Q0, Q1, Q2, and nested/multitask combinations; hold evaluation cohorts fixed.
6. **Automation effect:** automated, manual, and unresolved strata with matched chemical-space analyses. The tiny known-manual cohort requires uncertainty intervals and may be descriptive rather than a full standalone model.
7. **QT extension:** keep molecular hERG blockade separate from clinical QTc/TdP. Use QT evidence for triangulation, discordance analyses, and a later exposure-aware clinical model—not as a synonym for hERG.

### Frozen partitions

- Parent-InChIKey deduplication before splitting.
- Bemis-Murcko scaffold partition.
- Source-family holdout.
- Temporal holdout where dates are defensible.
- Low-similarity challenge set at prespecified Tanimoto thresholds.
- Assay-modality and automation transfer matrices.
- One untouched final test set whose labels are unavailable to model selection code.

### Required metrics

- Classification: ROC-AUC, average precision and prevalence, MCC, balanced accuracy, sensitivity, specificity, calibration/Brier/ECE, enrichment, and sensitivity at fixed false-positive rates.
- Regression: MAE, RMSE, Spearman, R2, within-threefold/tenfold IC50, calibration, interval coverage, and censored likelihood where applicable.
- Selective prediction: metric-versus-coverage curves and risk-coverage area, never a selected-subset metric alone.
- Every metric: paired bootstrap confidence interval at the parent-structure level and per-source/per-modality/per-quality subgroup results.

### Comparator ladder

Run cheap, strong baselines first: prevalence/median, logistic or ridge on Morgan, random forest, and XGBoost. Then add D-MPNN/GNN, hERGAT/Transformer_Morgan-style fusion, imbalance ensembles, docking/PLEC, quantitative/censored heads, and multi-channel/pretraining adapters. Deep models earn inclusion only through fixed-split incremental value.

## Highest-value analyses available before HPC

1. Produce endpoint/relation/unit/source/modality/automation/quality cross-tabs with denominators and missingness.
2. Compute overlap and label-disagreement matrices after parent-structure standardization.
3. Build source/scaffold/temporal split manifests and leakage reports.
4. Quantify active prevalence and chemical-space coverage per quality tier.
5. Run Morgan logistic/ridge, random-forest, and XGBoost baselines on CPU-sized stratified samples, then score the untouched full test manifests.
6. Compare automated versus manual/unknown measurements only within matched structures or matched chemical neighborhoods.
7. Create a hERG-QT evidence bridge with explicit entity, dose/exposure, study, endpoint, and confidence fields; do not collapse QTc into a hERG label.
8. Reproduce open comparators at pinned commits before adapting them. Repository-specific audit risks are in the source ledger.

## Recommended paper positioning

> Existing hERG predictors show strong performance on their own evaluation settings, but those settings frequently collapse assay semantics, use random chemical splits, distort deployment prevalence, or omit explicit wild-type and protocol provenance. We therefore built a substantially larger observation-level wild-type hERG resource with tiered evidence, assay modality and automation fields, clinical/QT linkage, and leakage-audited evaluation. Predictive superiority will be assessed—not presumed—through matched frozen benchmarks against strong classical, graph, sequence, ensemble, structure-based, quantitative, and multi-channel comparators.

That wording emphasizes the project's real current superiority without overstating results.
