# Comprehensive hERG liability platform — final results

## Strong classical-ML release update

For analog prioritization **inside the current Menin chemical series**, the completed focused classical search identified regularized logistic regression (`C=3.0`) on a 1,024-bit Morgan fingerprint plus 217 RDKit descriptors as the best model. Across 5-fold repeated stratified structure validation with three repeats (15 held-out folds), it achieved **ROC AUC 0.945, accuracy 0.940, balanced accuracy 0.923, MCC 0.839, sensitivity 0.957, specificity 0.889, and Brier score 0.058**.

The search covered logistic regression, SVM, Random Forest, Extra Trees, XGBoost, and LightGBM at three complexity levels across three molecular representations: **54 candidates, 162 model–regime combinations, 2,430 held-out validation fits, and zero failed fits**.

This model is released only behind a domain gate: nearest labeled Morgan-Tanimoto similarity ≥0.80 and ensemble standard deviation ≤0.20. The gate releases predictions for 34 of 50 unlabeled/intermediate compounds and routes 16 directly to experimental assay. Novel-scaffold extrapolation remains unapproved because locked scaffold validation does not meet the 0.90 requirement.

## Executive result

The project now has a configurable, provenance-tracked hERG modeling platform spanning confidential-only learning, confidential-prioritized public transfer, and equal-weight public transfer. It calculates multiple molecular representations once, searches nine requested classifier families across simple and complex parameterizations, performs inner-only model/feature/threshold/ensemble selection, and evaluates final choices on confidential scaffolds never seen during training.

The strongest defensible signal is the **equal-importance diverse ensemble**. Its raw held-out discrimination was **ROC AUC 0.575**. Using inner-calibrated probabilities and inner-selected thresholds, it achieved **balanced accuracy 0.649**, **MCC 0.257**, **sensitivity 0.565**, **specificity 0.733**, and **Brier score 0.203**. Confidence intervals remain wide, so this is a promising development result rather than a publication-grade efficacy claim.

## Data foundation

| Component | Result |
|---|---:|
| Confidential workbook rows | 111 |
| Unique valid confidential structures | 110 |
| Decisively labeled confidential structures | 61 |
| Confidential blockers / nonblockers | 46 / 15 |
| Intermediate IC50 structures reserved from binary labels | 14 |
| Missing IC50 rows | 36 |
| Exact numeric IC50 measurements | 59 |
| One-sided censored IC50 measurements | 16 |
| Paired inhibition measurements at 10 and 30 µM | 31 |
| Public decisive hERG structures | 2,777 |
| Public blockers / nonblockers | 2,412 / 365 |
| Exact public/private structure overlap | 0 |

Binary labels are interval-safe: IC50 ≤10 µM is a blocker, IC50 ≥30 µM is a nonblocker, and 10–30 µM is excluded from binary training. The `<0.37` and `>30` measurements are used only where their censoring interval determines the class unambiguously.

The paired inhibition measurements are internally strong: 10 and 30 µM inhibition have Spearman correlation 0.985, both track IC50 in the expected inverse direction (about −0.96), and all 31 pairs are dose-monotonic.

## Molecular parameter space

The reusable feature registry contains:

- 217 RDKit 2D molecular descriptors;
- 167-bit MACCS structural keys;
- Morgan circular fingerprints at 1,024 and 2,048 bits, radius 2;
- descriptor–fingerprint hybrid representations;
- raw SMILES token sequences for recurrent neural networks.

Numerical stabilization is deterministic and sample-independent, so a molecule's descriptor vector no longer changes when validation/test molecules are added or removed.

## Model and optimization coverage

The completed optimization covers:

- logistic regression;
- random forest;
- support vector machine;
- k-nearest neighbors;
- XGBoost;
- LightGBM;
- clustering/SVD/logistic hybrids;
- character-level GRU recurrent networks;
- dummy-prior controls.

The rapid search evaluated **135 model–feature–regime combinations** and completed **405 cross-validation fits with zero failures**. The publication-style nested run began with **45 candidates**, completed **1,169 inner fits with zero failures**, refit **45 outer models**, and generated **366 held-out predictions** across the three regimes and two strategies.

Successive halving reduced weaker combinations while preserving the best member of every requested model family. All model, feature, complexity, calibration, threshold, and ensemble choices were made using only the corresponding outer-training partition.

## Validation results

### Rapid stratified development screen

| Regime | Best development model | ROC AUC | Balanced accuracy | MCC |
|---|---|---:|---:|---:|
| Confidential only | Logistic + RDKit descriptors | 0.981 | 0.946 | 0.818 |
| Confidential prioritized | SVM + Morgan/RDKit hybrid | 0.972 | 0.945 | 0.871 |
| Equal importance | SVM + Morgan/RDKit hybrid | 0.970 | 0.945 | 0.871 |

These scores measure interpolation among closely related compounds and are optimization-screen results, not final publication estimates.

### Nested unseen-scaffold evaluation

| Regime | Strategy | Raw ROC AUC | Calibrated ROC AUC | Balanced accuracy | MCC | Sensitivity | Specificity | Brier |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Confidential only | Selected | 0.507 | 0.411 | 0.492 | −0.015 | 0.717 | 0.267 | 0.235 |
| Confidential only | Diverse ensemble | 0.481 | 0.397 | 0.538 | 0.066 | 0.609 | 0.467 | 0.224 |
| Confidential prioritized | Selected | 0.451 | 0.390 | 0.463 | −0.067 | 0.326 | 0.600 | 0.261 |
| Confidential prioritized | Diverse ensemble | 0.454 | 0.425 | 0.494 | −0.010 | 0.522 | 0.467 | 0.220 |
| Equal importance | Selected | 0.448 | 0.403 | 0.462 | −0.066 | 0.457 | 0.467 | 0.250 |
| **Equal importance** | **Diverse ensemble** | **0.575** | **0.512** | **0.649** | **0.257** | **0.565** | **0.733** | **0.203** |

For the leading equal-weight ensemble, the scaffold-bootstrap interval was wide: calibrated ROC AUC 0.512 [0.226, 0.745], balanced accuracy 0.649 [0.383, 0.802], and MCC 0.257 [−0.179, 0.509]. The result is therefore best interpreted as evidence that public information plus model diversity helps, while the available confidential sample is still too small for a stable unseen-series claim.

## Applicability and stability findings

- The labeled confidential set contains 18 Bemis–Murcko scaffolds; one scaffold contains 17 compounds, so scaffold-level uncertainty is substantial.
- Thirty-eight of 61 held-out observations fell inside the fold-specific private-chemistry applicability domain; 23 were out of domain.
- Nearest public similarity was only about 0.27 on average, confirming that the public set is useful background chemistry but is not a close analogue set for this confidential series.
- Selected families and feature sets changed across outer folds. This instability is recorded rather than hidden and is a primary reason not to nominate a single universal classifier yet.
- Calibration never reversed model rankings; the weaker calibrated pooled AUC arises from fold-specific probability scaling and genuine scaffold instability. Raw discrimination and calibrated probability metrics are now reported separately.

## Reproducibility and publication readiness

The platform emits candidate registries, molecular parameters, fold assignments, inner-fit audits, successive-halving histories, outer-fold selections, raw and calibrated predictions, bootstrap intervals, applicability-domain analyses, calibration curves, serialized fold models/calibrators, figures, configuration snapshots, and SHA-256 provenance hashes.

Verification status:

- latest relevant unit tests: 10/10 passing;
- final end-to-end nested smoke run: 32/32 fits successful, zero failures;
- completed full nested run: 1,169/1,169 inner fits successful, zero failures;
- confidential workbook hash: `88150f02e9d01aa3c4ba599e3e71aa08d3ad26c6ace1ad27fc16ebcd6fd6642b`;
- public hERG dataset hash: `f89779d1bf21932c5510ba90b2c3834b9adde9490b3288667132a800334dd832`.

## Decision

Use the equal-importance diverse ensemble as the current research baseline and uncertainty/ranking aid. Do not treat any individual probability as a validated safety result. The highest-value next experiment is a locked prospective hERG panel chosen to cover underrepresented scaffolds and model-disagreement regions; those results should be evaluated once without retuning before making a publication-level performance claim.
