# Platform trainable-data surfaces v1.0

## What is ready

The platform now has physically bound, validated training surfaces at useful scale without generating the expensive HPC feature store or fitting substantive models.

- hERG has 395,575 broad clean observations and 339,373 confirmed-WT fixed-dose structure labels. The reporting-quality hierarchy remains nested and smaller where the underlying evidence is genuinely rarer.
- PK/ADME has 642,065 endpoint-specific modeling rows across 93 trainable tasks and 175,965 connectivity leakage groups.
- Protein-conditioned binding and potency has 3,304,480 primary Kd/Ki/IC50 observations. The separate endpoint counts are Kd 193,925, Ki 661,274, and IC50 2,449,281; EC50 contributes 229,146 auxiliary observations and is never relabelled as affinity.
- PRISM contributes 8,372,603 verified finite viability values. LINCS contributes 976,325 structure-linked instances; its 954,845,850 profile-gene positions are explicitly metadata-derived and not claimed as scanned finite labels.

## hERG quality hierarchy

The first paper remains wild-type human KCNH2/hERG focused. Mutants are quarantined. Broad clean training, confirmed-WT fixed-dose consensus, preclinical native numeric, standardized exact-or-censored pIC50, functional method-resolved measurements, automation/modality, and clinical QT context remain separate surfaces. Clinical QT context is never a direct hERG label.

## Recommended training sequence

- hERG confirmed-WT fixed-dose binary baseline
- hERG preclinical native-numeric and censor-aware pIC50
- PK/ADME endpoint-specific tasks with conditioning
- Kd endpoint-semantics pilot
- Ki target-conditioned affinity
- IC50 target-conditioned potency
- PRISM viability representation pretraining
- LINCS expression only after GCTX finite-value staging
- shared molecule-protein encoder and quality-specific heads
- physics and structure feature increments on HPC

## Tool direction

The eventual researcher-facing tool accepts a molecule, a protein sequence or structure when the task requires it, the requested endpoint, and optional assay or exposure context. It should return an endpoint-specific prediction, calibrated uncertainty, applicability-domain evidence, nearest training analogs, provenance and reporting quality, and eventually constrained optimization suggestions. General affinity and IC50 models must condition on both molecule and protein; PK/ADME and hERG use task-specific context rather than pretending all labels are interchangeable.

## Boundaries

- Do not pool Kd, Ki, IC50, EC50, PK/ADME tasks, or hERG reporting levels as one scalar target.
- Do not admit mutant hERG into the wild-type paper scope.
- Do not convert QT, ECG, trial, approval, or candidate context into hERG labels.
- Do not count known mirrors or metadata-addressable matrix positions as independent measurements.
- Fit preprocessing only on training partitions and preserve structure, scaffold, target, and source grouping.

## What is not done

No production HPC feature store, large representation model, physics calculation, docking campaign, or substantive final model was created in this release. Predictive superiority is not established until competitors and internal models are evaluated on identical frozen challenges with calibration, coverage, uncertainty, and applicability-domain reporting. The established advantage today is the integrated scale, provenance, quality hierarchy, censoring semantics, mirror control, and evaluation design.
