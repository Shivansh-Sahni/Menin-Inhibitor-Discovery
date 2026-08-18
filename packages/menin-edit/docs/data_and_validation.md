# Data and validation

## Data roles in Menin-Edit

Menin-Edit has two evidence layers with different purposes:

1. **Public background evidence** supplies broad Menin and hERG coverage, the current production artifacts, and 3,899 public Menin matched molecular pairs.
2. **Historical lab evidence** supplies the chemically local series needed to learn how edits behave around the intended Menin inhibitors. It must stay in approved storage and be separated into development and evaluation roles before fitting or rule discovery.

The public model should not dominate simply because it has more rows. For the historical series, chemical relevance and honest validation matter more than raw dataset size.

## Public data inventory

The current processed public build contains:

| Dataset layer | Count |
|---|---:|
| Menin activity measurement rows | 8,176 |
| Menin standardized compounds across tasks | 2,104 |
| Primary Menin biochemical-binding IC50 structures | 849 |
| hERG activity measurement rows | 41,078 |
| hERG standardized compounds across tasks | 11,549 |
| Decisively labeled primary hERG IC50 structures | 2,777 |
| Public Menin matched molecular pairs | 3,899 |
| Directed Menin edit rules in the current Menin-Edit audit | 5,824 |
| Directed public evidence records | 7,798 |

The primary hERG binary policy labels IC50 `<=10 µM` as blocker and `>=30 µM` as non-blocker; the intermediate region is excluded. The 2,777 decisive structures contain 2,412 blockers and 365 non-blockers, so balanced metrics and calibration are more informative than accuracy alone.

## Historical workbook inventory

The workbook currently has 111 source compound rows. All structure strings parse under the registered RDKit policy, producing 110 unique standardized structures. There is no exact public-structure overlap.

Raw populated source-row counts are:

| Field | Populated source rows | Interpretation |
|---|---:|---|
| Menin binding IC50 | 87 | Primary local biochemical potency |
| Binding slope | 86 | Assay-shape/context information |
| MV4;11 IC50 | 86 | Menin-dependent cellular potency |
| MOLM13 IC50 | 86 | Menin-dependent cellular potency |
| HL60 IC50 | 87 | Selectivity-control evidence; every value is `>1000 nM` |
| hERG IC50 | 75 | 59 exact and 16 one-sided censored values |
| hERG percent-inhibition panels | 31 | Mostly measurements at 10 and 30 µM |
| Rat PO Cmax / AUC0-t | 50 / 50 | Exposure endpoints with dose context |
| Rat PO half-life / AUC0-inf | 48 / 48 | Exposure/PK endpoints |
| Rat IV AUC, clearance, Vdss | 50 each | Exposure/disposition endpoints |
| Rat bioavailability | 50 | `%F` |
| pKa descriptors | 111 | Predicted descriptors, not an optimization endpoint by default |
| Mouse plasma / tumor at 24 h | 11 / 11 | Sparse local exposure observations |
| Mouse PK and most stability/PPB fields | 3–4 | Too sparse for a primary model today |
| WT or mutant Menin IC50 fields | 0 | No current mutation-resistance labels |
| CYP panel / Caco-2 | 0 / 0 | No current measurements |

Fifty-two source compounds have Menin binding, both Menin-dependent cellular assays, and hERG IC50. Forty-two also have rat PO AUC0-t. This overlap supports multi-endpoint retrospective modeling, but the effective sample remains small after structure grouping and conflict filtering.

The governed loader collapses the duplicate structure and assay context, resulting in 110 compound rows and 1,165 observations across the currently registered endpoints. Its current aggregated counts include 86 Menin binding, 85 MV4;11, 85 MOLM13, 86 HL60, 75 hERG IC50, 61 concentration-specific hERG inhibition observations, and the registered rat PK endpoints. It currently does not normalize every sparse mouse, stability, PPB, tumor, or pKa-detail field into observation rows; pKa1/pKa2 remain compound descriptors.

Under the `development` role and exact-only MMP settings, the current builder produces 14,778 directed endpoint-evidence rows across 1,610 directed transformations. The engine can merge this governed CSV with the public library when an approved external `edit_library.private_pairs` path is configured. The checked-in default remains public-only.

## Data-quality findings

- One source-identifier pair has the same standardized structure but conflicting potency/cellular values. It must remain one structure group and be adjudicated or excluded; treating the records as independent train and test molecules would be leakage. The raw identifiers stay in approved storage.
- The loader currently flags one duplicate-structure group, 13 structure groups with an automated descriptor or label conflict flag, and 61 aggregated observations with provenance conflicts. These are review flags, not automatic proof that every value is wrong.
- Qualified concentration values are not converted to exact labels. For example, `>30 µM` hERG provides a bound and may define a non-blocker class under the project policy, while `>1000 nM` HL60 remains censored selectivity evidence.
- Workbook identifiers are pseudonymized with a runtime HMAC key by default. Raw identifiers are excluded unless explicitly requested in Python.
- Derived private structures, observations, matched pairs, features, models, logs, and reports remain confidential even after identifiers are removed.

## Applicability-domain result

The public Menin model's recorded domain threshold is Morgan Tanimoto 0.625. Against its training reference, the 110 unique historical structures have nearest similarities from 0.344 to 0.491, with a mean of 0.426. Therefore **0 of 110 historical structures are inside the public Menin domain**.

The public hERG threshold is approximately 0.2535. Fifty-five of 110 unique historical structures are inside that domain; counting the duplicate source row gives 56 of 111. The historical-series hERG ensemble supplies a second, private-chemistry domain perspective, but both domain flags and model disagreement must remain visible.

Consequences:

- public Menin predictions are background priors, not trustworthy local absolute answers;
- out-of-domain estimates must never receive favorable safety/potency credit by accident;
- historical local SAR and direct edit deltas are essential;
- validation must report performance by domain, series, and scaffold, not only one pooled score.

## Existing validation evidence

### Public Menin biochemical IC50 model

The primary scaffold split uses 680 training and 169 test structures with zero structure/scaffold overlap.

| Metric | Scaffold test | Temporal test |
|---|---:|---:|
| MAE, pIC50 log units | 0.759 | 1.302 |
| RMSE | 0.935 | 1.538 |
| R² | 0.541 | -2.737 |
| Spearman | 0.733 | 0.159 |
| Fraction within 1 log unit | 0.716 | 0.405 |

The scaffold interval achieved 0.917 empirical coverage for nominal 0.90 in the original artifact report. The temporal result shows severe distribution shift and is the reason Menin-Edit must not present a precise public-model pIC50 for the historical series as fact. The current default adapter uses the interval half-width configured in `default.yaml`; manifests should record this runtime value separately from the original artifact's interval.

### Public hERG classifier

The primary scaffold split uses 2,223 training and 554 test structures.

| Metric | Scaffold test | Temporal test |
|---|---:|---:|
| ROC AUC | 0.869 | 0.641 |
| Balanced accuracy | 0.731 | 0.551 |
| Sensitivity | 0.963 | 0.769 |
| Specificity | 0.500 | 0.333 |
| Brier score | 0.080 | 0.206 |

The temporal decline and low specificity matter for optimization: simply lowering a probability is not evidence of a safe cardiac exposure window.

### Historical/public hERG work

The historical hERG dataset has 61 decisive unique structures: 46 blockers and 15 non-blockers, with 14 intermediate structures excluded from binary labels and 36 source rows missing IC50. The existing rapid development screen is strong for interpolation, but nested unseen-scaffold evaluation is the defensible reference.

For the leading equal-importance diverse ensemble, nested unseen-scaffold results were raw ROC AUC 0.575, calibrated ROC AUC 0.512, balanced accuracy 0.649, MCC 0.257, sensitivity 0.565, specificity 0.733, and Brier score 0.203. The balanced-accuracy 95% scaffold-bootstrap interval was 0.383–0.802 and the MCC interval crossed zero. Menin-Edit therefore uses this work as a conservative ranking/disagreement signal, not as a publication-grade safety claim.

## Current endpoint semantics

| Endpoint | What it means | What it does not mean |
|---|---|---|
| `menin_biochemical_pIC50` | Predicted biochemical IC50 potency on `-log10(M)` scale | Mutant coverage, cellular efficacy, residence time, or clinical response |
| `herg_consensus_probability` | Conservative probability of the project's hERG blocker class | Clinical torsade/cardiotoxicity probability or exposure-adjusted safety margin |
| `structural_alert_count` | Number of RDKit PAINS/BRENK/NIH catalog matches | Toxicity probability; zero alerts is not proof of safety |
| `mv411_cellular_pIC50` / `molm13_cellular_pIC50` | Historical cell-line potency endpoints prepared for local modeling | General anticancer efficacy |
| `hl60_cellular_pIC50` | Censored cellular selectivity-control endpoint | General toxicity |
| rat PK endpoints | Historical species/route-specific exposure and disposition | Human PK or a clinical dose prediction |

DILI and Ames contracts are disabled until explicit models pass their own validation. A separate structure-based “binding” endpoint is also absent; the current biochemical IC50 is already a binding-potency assay endpoint.

## Validation plan for local models

### Partition before modeling

1. Group by standardized structure so duplicates never cross partitions.
2. Group closely related molecules by Bemis–Murcko scaffold or medicinal-chemistry series.
3. Assign development and locked roles before feature/model selection.
4. If chronological source information is defensible, reserve the newest series as a time-forward challenge.
5. Never use locked labels for edit discovery, model selection, calibration, thresholds, objective weights, or bound selection.

The implemented local regression trainer restricts fitting to exact, finite, non-conflicted `train`/`development` observations; performs scaffold-grouped cross-validation; calibrates an absolute-residual interval from out-of-fold predictions; compares against a fold-specific median baseline; derives a local similarity domain; and writes hash-verified private artifacts. The registry fails closed unless the manifest records at least 5% MAE improvement and OOF R² of at least 0.05.

Its first temporary private audit produced:

| Endpoint | Structures / scaffolds | OOF MAE | Median-baseline MAE | OOF R² | Recommendation |
|---|---:|---:|---:|---:|---|
| Menin biochemical pIC50 | 82 / 29 | 0.1361 | 0.1320 | -0.036 | Disabled |
| MV4;11 pIC50 | 78 / 29 | 0.3880 | 0.4366 | 0.136 | Passes development gate |
| MOLM13 pIC50 | 77 / 29 | 0.3257 | 0.3849 | 0.259 | Passes development gate |
| Numeric hERG pIC50 | 56 / 17 | 0.4023 | 0.4265 | 0.010 | Disabled by R² gate |

No local artifact is checked in or enabled by default. The cellular results justify preserving governed artifacts for further locked evaluation. The Menin result says the current descriptor/fingerprint regressor adds no value over a simple scaffold-fold median and must not replace the public prior or direct SAR evidence. The numeric hERG model likewise remains off. The historical hERG ensemble is implemented but disabled in the checked-in public configuration; it becomes an active historical signal only through an approved local configuration.

### Model evaluation

For regression endpoints, report MAE, RMSE, R², rank correlation, interval coverage/width, pairwise edit-direction accuracy, and performance by scaffold/domain. Always compare with simple baselines: training median, nearest-neighbor prediction, direct matched-pair mean, and the existing public model.

For hERG or future toxicity classification, report ROC AUC, PR AUC, balanced accuracy, sensitivity, specificity, MCC, Brier score, calibration, enrichment of liabilities in the top-ranked concern set, and performance by scaffold/domain. Thresholds must be chosen only within development data.

For the optimization system, evaluate more than endpoint prediction:

- fraction of held-out observed edits whose direction is predicted correctly;
- regret versus the best held-out feasible analogue;
- hard-bound violation rate under conservative intervals;
- Pareto-front precision against observed multi-endpoint outcomes;
- path stability under bootstrap/model perturbation;
- improvement over single-endpoint, weighted-sum-only, and random-supported-edit baselines.

### Ablations

To establish which parts add value, compare:

1. public models only;
2. local models only;
3. direct matched-pair evidence only;
4. absolute models plus direct delta evidence;
5. checked-in hERG public-only mode versus governed historical-ensemble-only and two-component conservative-consensus configurations;
6. Pareto plus bounds versus weighted sum alone;
7. one-step ranking versus the full stop-anywhere path search.

No retrospective result should be described as prospective drug discovery or wet-lab validation. It is evidence that the system prioritizes known outcomes under an honest holdout.
