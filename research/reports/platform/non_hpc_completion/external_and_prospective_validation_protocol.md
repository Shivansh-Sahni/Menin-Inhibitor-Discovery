# External and prospective validation protocol

Status: prespecified protocol, not executed external validation. Snapshot date:
2026-08-05. The current ChEMBL-only platform has no accepted independent or
prospective validation result.

## Governing rule

Validation must match one frozen task, endpoint, assay/study context, intended
use, and time horizon. A dataset is not external merely because it was
downloaded from another website: imported ChEMBL rows, shared publications,
identical assays, molecular series, and pretrained-model overlap must be found
and reported.

## Validation tiers

1. **Internal difficult split:** molecule/scaffold plus protein/target or
   double-cold split. This measures controlled retrospective generalization but
   is not external validation.
2. **Post-cutoff source update:** a later ChEMBL release frozen after model and
   preprocessing lock. Remove records available before the cutoff and cluster
   linked series/publications across the boundary. This is temporal evidence,
   not source independence.
3. **Independent-source lockbox:** rights-cleared BindingDB-curated or another
   primary experimental source after exact/near mirror, paper, assay, molecule,
   protein, and model-pretraining-overlap removal. Conflicting measurements are
   retained as context-aware evidence rather than forced to one truth.
4. **Prospective experiment:** preregistered compound selection and assay,
   blinded outcomes, all attempted compounds disclosed, prespecified decision
   threshold and analysis, then one locked evaluation.

No tier substitutes for the next tier. External-model scores are comparators,
not labels or independent experiments.

## General lock procedure

- Freeze canonical, task, split, feature, checkpoint/tokenizer, environment,
  metric, calibration, threshold, and overlap-audit hashes before any lockbox
  label is opened.
- Use stable record identifiers and bind every evaluated observation to source
  version, raw record, molecule/material, protein/construct, assay context, and
  disclosure date.
- Fit preprocessing, imputation, calibration, thresholds, applicability-domain
  cutoffs, and early stopping only on training/development data.
- Evaluate the lockbox once. A correction after label inspection creates a new
  model/version and requires a new untouched lockbox.
- Report attrition, abstention, failures, and missingness; absence is never an
  inactive, safe, or negative label.

## Binding/potency validation

- Keep `Kd`, `Ki`, `IC50`, and `EC50` separate.
- The preferred first confirmatory regression endpoint is exact compatible
  `Kd` or another task selected after the deep-leakage audit—not a pooled
  “affinity” score.
- Primary candidate metric: cluster-aware MAE on the chosen log-concentration
  scale. Required companions: median absolute error, RMSE, Spearman correlation,
  calibration slope/intercept, coverage/abstention, target-macro performance,
  and bootstrap intervals over the highest independent grouping.
- Molecular-weight, target-prior, nearest-neighbor, fingerprint-linear, and
  fingerprint-tree baselines use the identical evaluated observations.

## hERG, QT/QTc, and cardiac-safety boundary

hERG channel block is one component of an integrated proarrhythmic-risk
assessment. It is not a QT/QTc, torsades, or patient-safety label. The
[FDA final E14/S7B Q&A guidance](https://www.fda.gov/regulatory-information/search-fda-guidance-documents/e14-and-s7b-clinical-and-nonclinical-evaluation-qtqtc-interval-prolongation-and-proarrhythmic)
and [ICH training material](https://database.ich.org/sites/default/files/ICH_E14-S7B_TrainingMaterial_2022_0407.pdf)
support separate nonclinical hERG/in-vivo evidence and clinical QT evidence in
an integrated assessment.

For an experimental hERG validation set:

- record cell system, temperature, voltage protocol, exposure duration,
  concentration series, controls, replicate/cell counts, fit method, and
  censoring;
- verify actual exposure concentration where required, because nominal and
  cell-exposed concentrations can differ through binding, instability, or
  solubility ([FDA discussion](https://www.fda.gov/science-research/fda-science-forum/determination-drugs-poor-solubility-herg-external-solution-lc-msms-support-herg-potency-assessment));
- validate continuous potency separately from any thresholded classifier and
  report threshold sensitivity;
- report assay/laboratory heterogeneity, applicability domain, calibration, and
  performance at the clinically relevant prevalence; and
- prohibit a “cardiotoxicity” conclusion from hERG performance alone.

Clinical QT/QTc validation requires molecule/intervention, arm, dose, exposure,
time, population, ECG endpoint/correction, comparator, denominator, and study
version. Registry status, a posted-results flag, regulatory approval, or label
text presence is not a clinical outcome.

## Prospective study minimums

Before outcomes exist, register:

- target population and candidate-selection rule;
- comparator/baseline and practical success margin;
- assay protocol, controls, randomization, plate layout, blinding, replicates,
  failure/retest rules, and quality exclusions;
- sample-size rationale based on independent compound/series units;
- primary endpoint/metric, interval method, multiplicity family, and subgroup
  plan;
- all attempted compounds and any abstentions; and
- data lock, code lock, reviewer roles, and publication policy.

Model builders should not receive outcomes until the experimental and analysis
data are locked. A prospective hit is still an experiment under its recorded
conditions, not universal efficacy or safety.

## Decision status

The protocol is ready to use, but the gate remains `blocked_external`: no exact
checkpoint/task is approved, no rights-cleared independent lockbox has been
frozen, and no prospective assay has been authorized or run.
