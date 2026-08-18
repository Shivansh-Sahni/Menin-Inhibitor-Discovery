# Statistical analysis plan

Status: `partially_executed`. The zero-training real-corpus descriptive census
and adequately supported association panel were frozen, source-reverified, and
reproduced byte-identically. Model comparison, calibration, subgroup
performance, final-lockbox inference, and prospective validation remain
planned. The final test lockbox was not opened.

## 1. Analysis hierarchy

1. Freeze source versions, ontology, curation policy, analysis populations, feature versions, split manifests, endpoints, primary metrics, and multiplicity families.
2. Develop and tune on training/development folds only.
3. Lock code, environment, model, threshold/calibration, and analysis notebook/report template.
4. Evaluate once on the final lockbox. Corrections after seeing lockbox labels require a new version and new untouched lockbox.
5. Conduct prospective validation only after the retrospective model and plan are locked.

The unit of inference is generally an independent chemical/protein/source grouping, not a row. Replicates and mirrored records do not increase the independent sample count.

## 2. Prespecified estimands

Every task registers:

- population, endpoint, assay/study context, evidence stage, label relation policy, and observation period;
- intervention/prediction rule and comparator;
- prediction horizon if temporal;
- aggregation and cluster unit;
- primary metric and direction;
- applicability-domain/abstention policy;
- minimum practically important difference;
- missingness/selection assumptions.

Examples:

| Task | Primary estimand | Required companion estimands |
|---|---|---|
| Continuous endpoint (`pKd`, endpoint-specific `pIC50`) | Out-of-cluster MAE on the locked population | RMSE, median AE, Spearman, R-squared, calibration slope/intercept, coverage |
| Binary experimental activity/liability | Out-of-cluster average precision or prespecified cost-weighted utility | ROC AUC, balanced accuracy, sensitivity, specificity, PPV/NPV at stated prevalence, Brier, calibration |
| Screening/ranking | Recall/enrichment at a prespecified budget and target-macro average | BEDROC/NDCG or EF with confidence intervals, hit prevalence and random baseline |
| Structure/pose | Success probability under prespecified RMSD/interface criteria | distributional metrics, confidence calibration, failure/abstention rate |
| PK endpoint | Context-specific error for log endpoint in held-out chemical/study groups | bias, coverage, species/route strata; never pool incompatible endpoints |
| Uncertainty/abstention | Empirical interval coverage or risk at retained coverage | interval width, selective-risk curve, calibration error under shift |

Report both micro and macro estimates where targets/assays differ strongly. Macro averaging prevents high-volume targets from defining platform performance; micro estimates remain useful for operational throughput.

## 3. Label and measurement models

- Model `Kd`, `Ki`, `IC50`, and `EC50` separately. Any multi-task sharing retains separate heads/context and reports endpoint-specific performance.
- Transform compatible molar concentrations as `pX = -log10(X mol/L)` after unit validation.
- Use interval/right/left-censored likelihoods for inequality labels when implemented. The exact-only analysis remains a named sensitivity population.
- Preserve within-assay replicates. When aggregation is needed, report robust center, range/dispersion, replicate/source count, and conflict flag.
- Where support exists, use hierarchical measurement models with random effects for assay/source/lab/study and cluster-aware residuals. Do not estimate more levels than the data identify.
- For PK, keep subject/study grouping and concentration-time data; derived NCA statistics are secondary observations with method and covariance when available.

## 4. Splits, selection, and tuning

The default evaluation suite includes ligand scaffold/chemical cluster, temporal, target sequence/family/pocket, assay/source, and double-cold ligand+target holdouts. Exact identities and plausible mirror groups must never cross folds. A split that cannot satisfy nonempty/bounded constraints fails closed; it does not silently fall back to random.

Model/feature/hyperparameter selection uses nested group-aware cross-validation inside the development data. The lockbox is not used for choosing architecture, seeds, thresholds, calibration method, applicability-domain threshold, or early stopping. Report the number of attempted configurations and total compute.

Pretrained external models require an additional overlap stratum: known overlap, similarity overlap, post-cutoff, and overlap unknown. “Zero shot” is not claimed when training-source overlap is unknown.

## 5. Uncertainty

- Use paired cluster bootstrap for metric differences, resampling the highest relevant independent grouping (for example target then ligand cluster, or study) and preserving paired predictions.
- Use at least 2,000 bootstrap replicates for final intervals when computationally feasible; report percentile or BCa method and failure count. Development reports may use fewer and must say so.
- Report 95% confidence intervals and raw independent-group counts. For very small group counts, use exact/permutation methods or label estimates descriptive.
- Repeated seeds quantify training/split stochasticity on development data; report median/range and a variance decomposition where feasible. Do not average multiple peeks at the lockbox.
- Prediction intervals and class probabilities require held-out calibration. Evaluate coverage/calibration within source, time, target-family, and applicability-domain strata.
- Confidence intervals conditional on one curated dataset do not include source, curation, endpoint-definition, model-selection, or future-distribution uncertainty. State this explicitly.

## 6. Model comparisons

The primary comparison is the selected model versus the strongest prespecified low-complexity baseline on identical observations. Compute paired metric differences and cluster-bootstrap intervals. A complex model is superior only if:

1. the direction and minimum effect were prespecified;
2. the confidence interval excludes no improvement (and, for a confirmatory claim, exceeds the practical margin);
3. calibration/coverage and failure rate are not materially worse;
4. improvement persists on the primary difficult split and required sensitivities;
5. compute and latency are reported;
6. the comparison is not invalidated by known training overlap.

For target-level correlation, do not average undefined correlations or tiny assays without a declared rule. Report target-wise scatter/distribution and weight choice. Molecular-weight, nearest-neighbor, and target-prior baselines are mandatory for affinity benchmarks.

## 7. Multiplicity

Register hypothesis families before evaluation:

- **Confirmatory family**: at most one primary metric for each of no more than two primary tasks. Use Holm correction across these claims, two-sided family-wise alpha 0.05 unless a justified directional plan is frozen.
- **Key secondary family**: calibration, external/double-cold performance, and prespecified subgroup interactions. Control false discovery rate with Benjamini-Hochberg at q=0.05 within the named family.
- **Exploratory family**: all additional endpoints, models, thresholds, subgroups, features, and mechanistic correlations. Report raw effect/interval and BH q-values by family; label exploratory regardless of q-value.

No correction is applied across purely descriptive inventory counts, but uncertainty and denominator remain required. No claim may be promoted from exploratory to confirmatory after seeing results. Repeated seeds, metrics, thresholds, endpoints, and subgroup slices are multiplicity, not free replications.

## 8. Missingness and selection

Primary performance is measured on the locked eligible population and accompanied by source-to-model attrition. Do not impute outcomes. Feature imputation is trained within folds. If inverse-probability weights estimate a resolvable-source-population estimand, report weight model, trimming, balance, positivity violations, effective sample size, and unweighted results.

Conduct exact-only versus censoring-aware, complete-case versus supported-imputation, and selection delta-adjustment analyses. Divergent results narrow the claim; they do not justify choosing the most favorable analysis.

## 9. Error and subgroup analysis

Error taxonomies are frozen before lockbox review: entity/standardization, endpoint/context, source conflict, representation failure, applicability-domain, activity cliff, protein/ligand novelty, censoring/extreme value, and apparent model error with verified label. High-impact errors require source-document review by a scientist blinded to model identity where practical.

Subgroup interactions follow the support/multiplicity rules in `bias_missingness_selection_plan.md`. Report “insufficient support” rather than a point estimate that invites overinterpretation.

## 10. Prospective analysis

A prospective experiment requires preregistered candidate selection, assay/protocol, controls, sample-size rationale, plate/randomization/blinding plan, replicate handling, failure rules, primary endpoint, analysis script, and disclosure of all attempted compounds. Model builders should remain blinded to outcomes until data lock. Prospective success is defined against a prespecified baseline and decision threshold, not by retrospective cherry-picking.

## 11. Reporting requirements

For every result provide dataset build, split ID, model/feature hashes, sample and independent-group counts, metric definition, estimate, interval, multiplicity family, adjusted value where relevant, applicability coverage, failures/abstentions, source/target macro result, and prohibited extrapolation.

P-values are optional and never substitute for effect sizes, uncertainty, calibration, practical relevance, or external validation.
