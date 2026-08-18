# hERG v4 Local Discovery Campaign: Preregistered Protocol

## Purpose

This campaign is a local, CPU-based discovery experiment for the first hERG-focused paper and the wider molecular-safety platform. It is designed to answer two distinct questions:

1. Can a rigorously selected model improve scaffold-transfer prediction of quantitative hERG potency beyond the strongest existing internal result?
2. Which molecular, conformational, and assay-context variables provide reproducible, nonredundant information about hERG measurements and activity cliffs?

The campaign cannot guarantee novelty. It may establish a **candidate novel relationship** only when the prespecified replication gates below pass. It cannot establish predictive superiority without an external or prospective blinded challenge, and it cannot establish a causal molecular mechanism from observational data alone.

## Frozen starting evidence

- The primary quantitative endpoint is exact pIC50 for 18,801 training structures, grouped into fixed, nonoverlapping Bemis-Murcko scaffold folds.
- These quantitative records are human hERG and mutant-excluded, but they are `wild-type-or-unspecified`; they are not all experimentally confirmed wild type.
- Repository validation and test outcomes are sealed. They are not read during feature generation, model selection, stacking, calibration, interpretation, or final refitting.
- The confirmed-WT fixed-dose surface contains 339,373 structures, of which 265,625 training structures and 987 positives are authorized for the broad auxiliary classifier. It remains a separate endpoint and is never pooled with pIC50.
- V2 is the frozen internal performance anchor: scaffold-held-out MAE 0.445856 and RMSE 0.647011 on 18,801 structures.
- V3 is a completed negative/control result: primary nested MAE 0.451971 and RMSE 0.654598. V3 was approximately 0.0061 pIC50 MAE worse than V2 by paired scaffold bootstrap.
- V3 generic ligand 3D interactions did not improve global prediction. Fundamental-core MAE was 0.525495 and adding prespecified interactions worsened it to 0.526574.
- V3 assay strata showed materially different errors, including lower error for functional electrophysiology and higher error for binding/radioligand strata. These are confounded observations, not causal measurement-method effects.
- V3 activity-cliff prediction was poor. This is a priority failure mode rather than evidence that the problem has been solved.

Every starting artifact, split, implementation, configuration, model, and output must be byte- and schema-bound in the v4 manifest.

## Primary success criteria

### Predictive criterion

The primary estimate is one prediction per structure from a five-fold, whole-scaffold, nested outer evaluation. All candidate choice, hyperparameter selection, early stopping, preprocessing, stack weights, and calibration occur inside the corresponding outer-training partition.

An internal predictive improvement is declared only if all conditions hold:

- the v4-v2 paired MAE difference is negative;
- the 95% scaffold-cluster bootstrap confidence interval is entirely below zero;
- v4 MAE is below 0.445856;
- the improvement is present in at least four of five outer folds;
- RMSE, error above 1 log unit, and scaffold-balanced MAE do not materially worsen;
- no repository validation or test labels were opened.

If these conditions fail, the simplest statistically tied model is preferred and the result is reported as no demonstrated improvement.

### Candidate relationship criterion

A feature or interaction becomes a candidate discovery only if all conditions hold:

- it was included in the frozen hypothesis registry before the analysis ran;
- its conditional held-out importance or drop-column effect has a 95% scaffold-bootstrap interval excluding zero;
- its signed effect is directionally consistent in at least four of five outer folds;
- its false-discovery-rate-adjusted q-value is below 0.05 within its hypothesis family;
- it replicates in at least two chemistry-matched assay or source strata with sufficient support;
- it is not explained solely by label disagreement, low similarity, one source, or conformer failure;
- its direction is concordant with a matched-pair or activity-cliff analysis when an appropriate pair set exists.

Passing these gates supports a reproducible observational relationship, not causality. Causal language requires prospective chemical or electrophysiological perturbation.

## Prespecified molecular hypotheses

The interpretable hypothesis set is deliberately small. It includes:

- hydrophobic exposure, represented by logP and nonpolar surface, increases hERG potency in a bounded, nonlinear manner;
- a protonatable/basic center combined with aromatic surface is more informative than either property alone;
- polar surface or exposed polarity moderates hydrophobic/aromatic liability;
- intramolecular polar contacts can mask polarity and increase effective hydrophobic exposure;
- flexibility changes the probability of presenting a favorable charge-aromatic geometry;
- conformer energy spread and conformer population concentration identify compounds whose single-conformer representation is unreliable;
- the benefit of ligand 3D features is concentrated in activity cliffs, low-similarity structures, or specific assay strata rather than in the global mean;
- assay modality and source create systematic offsets and heteroscedastic noise after chemistry matching.

No more than 30 interpretable variables and 10 prespecified interactions may be promoted into the primary relationship analysis. Fingerprint bits may improve prediction but are not individually presented as biological mechanisms.

## Campaign stages

### 1. Immutable preparation and baseline reproduction

Validate all input hashes, the exact 18,801-row training closure, the 339,373/265,625 broad closure, fixed scaffold folds, feature identities, censoring relations, and the absence of repository validation/test outcomes. Reproduce the V2 and V3 anchors from their frozen prediction artifacts before fitting v4 models.

### 2. Material classical-model optimization

Run XGBoost, LightGBM, and bounded-memory ExtraTrees candidates. The search must vary parameters independently, include the V2 winning region, and include robust-loss and regularized alternatives. It must not encode several hyperparameters into a single opaque “complexity” setting and then claim individual parameter insight.

The selection objective combines molecule-level MAE, scaffold-balanced MAE, RMSE, tail error, and stability. One-fold screens may eliminate clearly poor candidates, but finalists must run across all inner folds before an outer prediction is produced.

### 3. Genuine Chemprop optimization

Train new Chemprop D-MPNN models from the frozen training SMILES. V3 reused older predictions; v4 must not call reuse “training.” Candidate architectures vary depth, hidden width, dropout, feed-forward depth, loss, learning-rate schedule, and seed. Each outer fold has an inner-only screen, an inner-fold promotion stage, and exactly one selected outer evaluation.

Chemprop runs are isolated CPU subprocesses to avoid OpenMP conflicts. Their checkpoints, exact commands, environment, early-stopping epochs, and predictions are retained for selected candidates.

### 4. Nested heterogeneous stacking

Combine genuinely diverse inner-OOF predictions: the selected tree model, selected D-MPNN, similarity baseline, and a compact classical model. Fit nonnegative or ridge stack weights only on inner OOF predictions. Freeze weights before applying them to the outer fold. Compare stacking against the strongest single model with paired scaffold bootstraps.

### 5. Broad confirmed-WT auxiliary learning

Train a separate class-imbalanced classifier on 265,625 confirmed-WT fixed-dose training structures. Use sparse Morgan plus descriptors, prevalence-preserving grouped folds, probability calibration, and metrics appropriate to 0.37% prevalence: PR-AUC, MCC, Brier score, expected calibration error, and recall/precision at fixed false-positive and review-budget thresholds.

Test one strictly cross-fitted transfer feature in the quantitative model. A broad probability for a quantitative outer structure must come from a broad model that did not train on that structure or its leakage group. The transfer feature is accepted only through the same nested scaffold gate as any other feature.

### 6. Censored quantitative sensitivity

Preserve exact, left-censored, and right-censored potency bounds. Fit an interval-aware sensitivity model without turning thresholds into fabricated point labels. The exact-regression task remains primary. Report interval violations, exact-subset error, and whether censored evidence changes potency-tail behavior.

### 7. Expanded ligand conformer physics

Generate a genuinely new, label-blind 24-conformer ensemble for every authorized exact-training structure, retaining up to eight optimized conformers per structure. Use deterministic ETKDG, MMFF94s with recorded UFF fallback, 298.15 K within-parent Boltzmann summaries, convergence flags, charge/polar exposure, internal polar contacts, shape, flexibility, and conformer uncertainty.

Run a 50-conformer convergence audit on a frozen information-rich panel selected only from nested OOF residuals, activity-cliff membership, and applicability-domain strata. Compare feature stability at 6, 20, and 50 requested conformers. This panel is explanatory; it cannot be used to estimate unbiased global model performance unless selection is repeated inside every outer fold.

The local software does not provide validated pH-specific protomer populations. Tautomer counts and protonation-site counts are hypothesis descriptors only. They must never be called microstate populations, pKa predictions, or free energies.

### 8. Receptor-state gate

The six local cryo-EM coordinate sets motivate a receptor ensemble, but raw mmCIF files are not docking receptors. Receptor-derived features are blocked unless a separate manifest proves construct/assembly selection, missing-residue treatment, protonation, ions, pocket definition, ligand preparation, software versions, and redocking controls for the three bound ligands.

The v4 campaign records this gate. If it is not satisfied, no docking score or pose feature enters a model. Compute is reallocated to trained graph models, conformer convergence, assay heterogeneity, and activity-cliff analysis. A blocked receptor branch is an honest deliverable, not a failed campaign.

### 9. Assay and data-quality model

Use nested OOF residuals and observation-level provenance to estimate modality-, automation-, assay-family-, source-, and protocol-completeness offsets and variances. Chemistry-matched comparisons are primary; raw stratum means are descriptive. Manual patch-clamp effects are not generalized when sample support is small.

This stage must distinguish measurement heterogeneity from molecular prediction. It may justify assay-conditioned heads or uncertainty, but it cannot silently adjust the label used for the deployable molecule-only model.

### 9a. External assay-replication track

The campaign should freeze a public multi-site patch-clamp package when a redistributable, record-level source can be obtained and independently hashed. This package is used first to estimate assay reproducibility, laboratory/platform offsets, and an empirical measurement-noise floor. It is not automatically an external predictive challenge: every compound is checked against all training structures, parents, connectivity keys, and scaffolds. Compounds found in training are marked overlap and cannot support an external generalization claim.

If an overlap-free subset with adequate size remains, its outcomes stay sealed until the v4 recipe, preprocessing, and decision thresholds are frozen. Otherwise the package remains an external assay-replication analysis only. Failure to obtain a legally reusable record-level package is recorded as a blocker and never replaced with values copied from figures.

### 10. Activity-cliff and matched-pair analysis

Evaluate the 43,824 existing training-only matched pairs and the threshold-defined cliff subset using nested OOF predictions. Measure pair-delta MAE, direction accuracy, coverage within 0.5 log unit, and residual repair by each feature block. Test whether conformer/charge/polarity features repair particular transformations consistently across scaffolds.

### 11. Uncertainty and applicability domain

Fit cross-conformal intervals and an applicability score combining ensemble spread and nearest-training-structure Tanimoto similarity. Calibration is evaluated out of fold. Report coverage, interval width, error by similarity bin, and abstention tradeoffs.

### 12. Final models and dossier

After the nested recipe is frozen, refit deployable exact and broad models on all authorized training data. Each bundle includes the serialized model, preprocessing, feature order, versions, input hashes, predictor interface, model card, and round-trip smoke predictions.

The final scientific dossier must include:

- an executive outcome and claim ledger;
- a complete methods report;
- the unbiased nested model-selection result and comparison with V2/V3;
- parameter-search and model-family results at a common evaluation grain;
- the feature hypothesis registry with pass/fail evidence;
- conditional importance, signed-effect, multiplicity, and replication results;
- assay/source offset and variance findings;
- activity-cliff and matched-pair findings;
- conformer convergence and failure analysis;
- broad confirmed-WT classifier results;
- censored-data sensitivity;
- uncertainty/applicability results;
- receptor readiness or blocked-gate report;
- model cards and inference instructions;
- a failure/exception ledger and hash-bound manifest;
- a manuscript-ready results narrative that separates findings, hypotheses, negative results, and limitations.

The report must not reduce this work to a one-page summary. Machine-readable Parquet/JSON evidence accompanies every substantive conclusion.

## Statistical analysis contract

All primary model comparisons are paired at the structure level and resampled by whole scaffold. The primary uncertainty interval uses at least 10,000 scaffold-cluster bootstrap replicates. Candidate discovery testing is performed at the prespecified feature or interaction grain, not once per fold and then pooled as if fold observations were independent.

Conditional feature value is estimated on held-out outer-fold structures. The default test is repeated conditional permutation within prespecified chemistry neighborhoods, with the change in absolute error paired within structures and clustered by scaffold. A relationship must retain its direction across at least four outer folds. False-discovery-rate correction is applied once across the frozen hypothesis family. Feature importance based solely on tree gain, split count, or impurity is descriptive and cannot pass the relationship gate.

Signed response curves use a genuine accumulated-local-effect calculation or an explicitly named alternative whose construction is verified on held-out data. Quantile-binned mean predictions are not called ALE. Chemistry-matched replication uses propensity or nearest-neighbor matching within common support and reports balance diagnostics before comparing assay/source strata.

Stack weights, probability calibrators, conformal scores, broad-transfer transforms, and abstention thresholds are learned using inner or cross-fitted predictions only. No quantity chosen using an outer-fold outcome is evaluated on that same outer-fold outcome as if it were confirmatory.

## Runtime calibration and compute allocation

Before the full schedule begins, one representative candidate from each expensive family is timed with peak resident memory, average CPU utilization, output growth, and thermal state recorded. The scheduler updates only expected durations and safe concurrency from this pilot; it does not alter scientific hypotheses or evaluation folds.

The intended active-work allocation is:

- approximately 0.5 hour for immutable preparation, empirical timing, and baseline reproduction;
- approximately 3-5 hours for genuinely multidimensional nested XGBoost, LightGBM, and ExtraTrees optimization anchored at the v2 winner;
- approximately 8-12 hours for newly trained nested Chemprop candidates and selected outer refits;
- approximately 7-10 hours for new 24-conformer exact-training ensembles and a 6/20/50-conformer convergence panel;
- approximately 2-4 hours for broad-WT modeling, cross-fitted transfer, censored sensitivity, stacking, uncertainty, assay matching, activity-cliff analysis, and final dossier generation.

These are evidence-based bounds, not promises that the laptop remains busy for a fixed wall time. The campaign does not sleep, repeat identical fits, or preserve scientifically inferior checkpoints merely to consume time. If empirical timings show that all prespecified useful work will finish early, it finishes early and reports the measured compute ledger. If required work cannot fit within the 30-active-hour ceiling, lower-priority sensitivity units are deferred before any primary nested evaluation or final analysis is sacrificed.

## Scenario-dependent decisions

- If the tree model remains best, deploy it and report that learned graph representations did not add scaffold-transfer value.
- If Chemprop improves only after stacking, report complementary representation value rather than D-MPNN superiority.
- If conformer features improve activity cliffs but not mean MAE, keep them as a targeted correction/interpretation layer rather than burdening all inference.
- If assay-conditioned analysis explains a large fraction of residual error, prioritize protocol-resolved data acquisition and measurement-specific heads.
- If the broad transfer feature helps, present confirmed-WT fixed-dose evidence as auxiliary representation; never relabel it quantitative potency.
- If no v4 model beats V2 with confidence, retain V2 as the internal champion and report an apparent assay/noise/representation ceiling.
- If no feature relationship passes every gate, report no confirmed novel molecular relationship and list the best preregistered hypotheses for prospective testing.

## Runtime and resource contract

The target is approximately 22-27 hours of scientifically useful active work on the current M3 laptop, with a hard 30-active-hour ceiling and a 90-minute finalization reserve. The script does not sleep or repeat identical work to fill time. It may finish early if all prespecified useful work finishes early.

At most six compute threads are used. Broad sparse models run alone; smaller model trials may run concurrently only after an empirical memory pilot. The campaign pauses before starting new work below 25 GiB free disk, below 4 GiB available memory, or above its 12 GiB output budget. Completed units are atomic, self-hashed, resumable, and retained only when scientifically necessary.

The laptop must remain on AC power on a hard, ventilated surface. Thermal throttling can slow the work and is not a reason to bypass resource safeguards.

## Claim boundary

The strongest possible outcome of this local campaign is a robust internal performance improvement plus one or more replicated candidate relationships. “Groundbreaking,” “state of the art,” “clinically validated,” “mechanistic,” and “superior” remain forbidden without the relevant external, prospective, or perturbational evidence.
