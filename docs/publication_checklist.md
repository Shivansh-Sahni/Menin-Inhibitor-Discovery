# Publication readiness checklist

Use this as a release gate, not a retrospective formality. A checkbox should link to evidence in the frozen release bundle. Items marked “required” block publication claims supported by this pipeline.

## 1. Governance, authorship, and scope

- [ ] **Required:** The manuscript question, prediction endpoint, intended use, and excluded claims are written before final evaluation.
- [ ] **Required:** All authors, contributors, affiliations, acknowledgments, and funding/conflict disclosures are approved by the relevant people and institutions.
- [ ] **Required:** The Wang lab data owner has approved the use and disclosure of any internal data, aggregates, figures, and models.
- [ ] **Required:** Patent, sponsor, NDA, embargo, export, animal/human, and institutional obligations have been reviewed where applicable.
- [ ] **Required:** A repository software license has been selected by the rights holder or the repository is clearly distributed without a reuse license.
- [ ] Third-party database attribution and redistribution terms have been reviewed for every released artifact.
- [ ] `CITATION.cff`, repository metadata, and manuscript software/data availability statements agree.

## 2. Frozen computational specification

- [ ] **Required:** Code is tagged at the analyzed commit and the working tree is clean.
- [ ] **Required:** The resolved configuration and command line are archived.
- [ ] **Required:** The exact Python/package environment or container digest is archived.
- [ ] **Required:** Publication runs explicitly used the RDKit backend; no silent feature fallback occurred.
- [ ] Random seeds, thread settings, and platform/BLAS details are recorded.
- [ ] Tests, lint, and an offline full-pipeline smoke test pass in a clean environment.
- [ ] Generated reports can be reproduced from the frozen processed snapshot.

## 3. Data provenance and release

- [ ] **Required:** Raw, processed, software, models, analysis, and reports manifests verify with no issues when analysis is enabled, share the expected build ID, and have valid upstream digest/direct-input links.
- [ ] **Required:** Source release/status, access date, search terms, target IDs, and record limits are reported.
- [ ] **Required:** Every released table has a data dictionary and source/license notice.
- [ ] **Required:** Raw values, units, relations, structures, source record IDs, assay IDs, and document provenance are retained internally.
- [ ] Cross-source and within-source duplicate handling is documented; potential mirror links have been source-reviewed where they materially affect labels.
- [ ] Public/private overlap was checked by standardized structure and provenance.
- [ ] Any unavailable input has a clear access procedure, steward, and independent verification path.
- [ ] Release files have passed confidentiality and small-cell/structure-disclosure review.

## 4. Curation validation

- [ ] **Required:** Unit mappings and pActivity conversion are unit-tested and spot-checked against source records.
- [ ] **Required:** Unknown/missing units and unresolved PubChem endpoints/targets are not assigned defaults.
- [ ] **Required:** Exact, approximate, left-censored, and right-censored records are distinguished.
- [ ] **Required:** ChEMBL `standard_flag`, validity comments, potential duplicates, pChEMBL agreement, and variants are audited.
- [ ] **Required:** Structure standardization policy/version and parent/tautomer/stereochemistry limitations are reported.
- [ ] **Required:** Modeling tables are stratified by compatible endpoint and assay family.
- [ ] Quarantine counts and every exclusion reason are reported by source.
- [ ] A domain expert has reviewed the highest-impact excluded rows and all manual PubChem registry decisions.
- [ ] High-disagreement compounds and multi-source conflicts have source-document review notes.
- [ ] Sensitivity analyses cover plausible curation policies, including the cross-source-mirror no-collapse run, clean-label/heterogeneity policy, validity warnings, censoring, and structure-parent choices.

## 5. Descriptive analysis

- [ ] Measurement, eligible-row, structure, endpoint, assay-family, source, year, and label counts are reported.
- [ ] Potency and hERG-label distributions are shown before and after curation.
- [ ] Chemical-space/scaffold coverage and train/test differences are characterized.
- [ ] Missingness and assay-context completeness are reported.
- [ ] Within-compound variability and source disagreement are quantified.
- [ ] PK/ADMET is presented by endpoint, unit, species, matrix, route, and protocol context—not as a pooled score.

## 6. Chemical intelligence and experiment design

- [ ] **Required:** The analysis is explicitly scoped to the pre-specified primary Menin endpoint/assay family; it is not silently pooled across tasks.
- [ ] Descriptor, QED, property-window, apparent ligand-efficiency/LLE, fingerprint, scaffold, Butina, cliff, MMP, connectivity, alert-catalog, and RDKit version/settings are archived.
- [ ] Apparent efficiencies are labeled as IC50-derived heuristics rather than thermodynamic binding efficiencies.
- [ ] PAINS/Brenk/NIH matches are reviewed as alerts and are not described as proof of assay interference, toxicity, or invalidity.
- [ ] Bemis–Murcko series, Butina clusters, and achiral/chiral nearest-neighbor novelty are labeled as representation- and dataset-dependent; local novelty is not presented as patent novelty or freedom to operate.
- [ ] Fingerprint cliffs, SALI, single-cut matched pairs, and connectivity-variant groups have source/assay review; no pair or transform is interpreted causally without experimental evidence.
- [ ] **Required:** Observed primary hERG evidence overrides predictions; missing and out-of-domain predictions are `unknown`, receive no safety credit, and are not described as low risk.
- [ ] The chemistry gate, five follow-up tiers, objective weights, missing policy, Pareto definitions, and base/leave-one-out/emphasized-objective sensitivity results are reported.
- [ ] A zero-count `priority_1_balanced_public_evidence` tier is retained and interpreted as an evidence result, not treated as a failed run or repaired by post hoc threshold changes.
- [ ] PK/ADMET contributes coverage/data-gap context only and is not represented as a pooled desirability score.
- [ ] The revumenib/ziftomenib reference structures, PubChem records, FDA status/indications, and check date are archived; approved-reference coverage is described only as a public-dataset coverage benchmark, not an efficacy comparison.
- [ ] The configured prospective-selection quotas, scaffold-series cap, category rationales, shortfalls, and frozen plan are archived before experiments begin.
- [ ] `prospective_selection_plan.csv` is described as a public-data experiment-design template—not prospective validation, a lead recommendation, or evidence that any selected compound is safe or efficacious.

## 7. Model design and leakage control

- [ ] **Required:** All rows for a registered structure remain in one partition and one CV fold at a time.
- [ ] **Required:** Public/private and source-duplicate structures cannot cross partitions.
- [ ] **Required:** Model/hyperparameter selection uses training-only cross-validation.
- [ ] **Required:** Final holdout labels were not used for candidate, threshold, feature, or curation-policy selection.
- [ ] **Required:** Dummy and simple linear baselines are included.
- [ ] Primary scaffold split and its grouping method are reported.
- [ ] Compound-grouped random split is labeled as an optimistic comparator.
- [ ] Temporal validation is used only with adequate date coverage and a defensible cutoff.
- [ ] Public temporal results disclose that ChEMBL/BindingDB document years and PubChem assay-deposit years are mixed, report `date_provenance`/missing dates, and are not described as assay chronology.
- [ ] Any split fallback is disclosed and the requested-strategy claim is withdrawn.
- [ ] Split assignments and their digest are archived.

## 8. Evaluation and statistics

- [ ] **Required:** Holdout sample sizes, class counts/prevalence, and endpoint composition are reported.
- [ ] Menin results include MAE, RMSE, median absolute error, R², rank/linear correlations, bias, and error-within-threshold rates.
- [ ] hERG results include ROC-AUC, PR-AUC, balanced accuracy, sensitivity, specificity, precision, F1, MCC, Brier, log loss, calibration, and confusion counts.
- [ ] Confidence intervals report scaffold-group resampling (or the disclosed row fallback), resampling-group count, and successful resample count.
- [ ] Split/seed sensitivity is reported without tuning to the final holdout.
- [ ] Regression residuals are examined by potency, source, endpoint, assay family, scaffold, similarity, and date.
- [ ] Classification errors and calibration are examined by source, endpoint/assay family, scaffold, similarity, and class.
- [ ] Performance is compared with the dummy baseline and justified in terms of experimental utility, not only statistical significance.
- [ ] Multiple comparisons and exploratory analyses are labeled and controlled where inferential claims are made.

## 9. Uncertainty, domain, and validation

- [ ] **Required:** Prediction intervals/probabilities and their empirical holdout behavior are reported.
- [ ] **Required:** Applicability-domain definition, threshold, and holdout coverage are reported.
- [ ] Results are stratified inside versus outside the domain.
- [ ] Activity cliffs and nearest-neighbor failure cases are discussed.
- [ ] An independent external set has been de-duplicated against training data and evaluated with compatible labels, or its absence is explicit.
- [ ] A prospective, time-forward experiment tests utility on newly selected compounds.
- [ ] Experimentalists evaluating prospective compounds are blinded where feasible.
- [ ] Decision thresholds are justified from costs/exposure margins and validated, not adopted from communication bands.

## 10. Figures, tables, and manuscript claims

- [ ] **Required:** Every figure/table is generated from the frozen build and maps to a script or report artifact.
- [ ] Axes, units, n values, endpoint/assay scope, split, model, confidence intervals, and exclusions are stated.
- [ ] No chart conflates observations, aggregates, and predictions.
- [ ] Predictions outside the applicability domain are visually identified.
- [ ] Chemical structures and internal identifiers have release approval.
- [ ] Main-text claims match the primary pre-specified analysis; random-split or exploratory maxima are not promoted.
- [ ] Main-text Menin claims use the pre-specified `IC50 × biochemical_binding` task; other endpoint/assay tasks are clearly labeled exploratory or separately pre-specified.
- [ ] Main-text hERG claims use the `IC50 × electrophysiology_functional` task; pooled hERG results are labeled sensitivity-only and never described as a clinical safety estimate.
- [ ] Negative results, failed split strategies, insufficient-data endpoints, and model limitations are retained.
- [ ] Chemical-intelligence outputs distinguish measurements, predictions, review alerts, unknown evidence, priority policy, and proposed experiments; no tier/score/plan is called a validated candidate list.
- [ ] The manuscript includes the limitations in [`limitations.md`](limitations.md) that apply to the final build.

## 11. Archive and independent review

- [ ] **Required:** A read-only release bundle contains code, configuration, environment, manifests, verification, tables, split assignments, model evidence, complete analysis evidence, figures, and documentation allowed for release.
- [ ] **Required:** Another person has reproduced the public-only build from a clean environment.
- [ ] A cheminformatics reviewer has inspected structure/identity/split handling.
- [ ] A Menin assay expert has inspected endpoint/assay comparability.
- [ ] A safety pharmacology expert has reviewed hERG interpretation.
- [ ] A statistician or ML reviewer has inspected validation, intervals, calibration, and claims.
- [ ] Approved artifacts receive a persistent repository/archive identifier and access date.
- [ ] Post-publication correction/versioning and data-retention responsibilities are assigned.

For any internal validation, additionally require:

- [ ] **Required:** Every row has a reviewed `development`, `locked_external`, or `prospective_blind` role.
- [ ] **Required:** Locked-external and prospective-blind rows were physically/logically excluded from training, calibration, feature/threshold selection, curation-policy tuning, and stopping decisions.
- [ ] **Required:** Prospective-blind labels remained sealed until code, configuration, split, model, thresholds, and the analysis plan were frozen; the lock and unblind events are recorded.
- [ ] **Required:** Development, locked-external, and prospective-blind cohorts were run in separate approved private roots/builds/manifests; the public CLI was not used as a confidential-data intake path.

## Recommended manuscript claim language

Prefer:

> On the frozen, curated public-data snapshot and pre-specified scaffold holdout, the selected baseline achieved [metrics with intervals]. Performance decreased/increased under [temporal/random] sensitivity analyses. Predictions were considered in-domain only under the disclosed similarity rule and require prospective validation.

Avoid:

> The model accurately predicts Menin potency and hERG safety.

The latter omits the dataset, endpoint, assay, validation design, uncertainty, domain, and the fact that hERG activity is not clinical safety.
