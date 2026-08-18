# Sensitivity and ablation plan

Status: `planned`. The matrix is mandatory for claims, but only scientifically identifiable cells should be run. All variants use frozen seeds/splits and must be reported, not selected post hoc.

## Core sensitivity matrix

| Dimension | Primary policy | Required alternatives | Failure signal |
|---|---|---|---|
| Molecule standardization | Versioned parent/stereo-aware policy | submitted material; salt-retaining; neutralization off; tautomer policy variants; stereo-unspecified view | Large metric/ranking shifts or entity collisions |
| Protein mapping | Versioned accession+sequence+construct policy | canonical sequence only; isoform/mutant exclusion; low-confidence mappings removed | Performance depends on ambiguous mapping or canonical leakage |
| Endpoint | Endpoint/context-specific | exact endpoint-only; closely related assay-family strata; no cross-endpoint pooling | Claimed gain disappears when endpoints are separated |
| Relations/censoring | Censoring-aware where available | exact-only; interval-bound stress; excluded-censored population characterization | Exact-only selection changes value/source/chemistry materially |
| Units | Strict supported conversion | source-native validation; exclude inferred/ambiguous conversions | Results driven by one unit/source transform |
| Replicates | Robust context-specific aggregation | raw hierarchical model; median/mean; one-record-per-source; conflict exclusion | Aggregation choice reverses conclusions |
| Mirrors/duplicates | Provenance-linked grouping | no collapse; strict exact-only collapse; expanded approximate linkage | Performance inflated by plausible mirrors or collapse changes labels |
| Source | All admitted sources | leave-one-source-out; each major source alone; curation-origin strata | Model fails on a source not represented in training |
| Time | Conservative disclosure cutoff | alternative clock definitions; temporal gaps/embargoes; post-cutoff only | Performance collapses with a true later-information cutoff |
| Ligand novelty | Chemical/scaffold primary | random diagnostic; multiple cluster thresholds/fingerprints; activity-cliff subset | Gain exists only under random/easy similarity |
| Target novelty | Sequence/family/pocket holdout | identity thresholds; family leave-one-out; construct/mutant strata | Claimed protein generalization disappears under related-target removal |
| Double-cold | Ligand+target non-overlap | independent ligand-cold and target-cold; stricter similarity thresholds | Cross-modal model relies on one familiar side |
| Assay context | Context-aware model/task | assay/source/lab holdout; protocol-rich subset; unknown-context exclusion | Apparent structure signal is assay/source recognition |
| Missing covariates | Fold-fit imputation/indicators | complete case; supported alternative imputation; modality-specific missing token | Imputation dominates or coverage collapses |
| Selection | Eligible population | inverse-probability weighted exploratory; trimming grid; delta/pattern mixtures | Conclusions require unstable weights or unverifiable assumptions |
| Class definitions | Prespecified thresholds | threshold grid; continuous/ordinal alternative; include ambiguous as separate class | Performance depends on extreme-label exclusion |
| Negatives/decoys | Explicit experimental negatives | no synthetic decoys; harder property-matched decoys; source-matched negatives | Classifier distinguishes decoy generator/properties rather than binding |
| Applicability domain | Training-only threshold | fingerprint/sequence/pocket alternatives; quantile grid; no-AD descriptive | “Confidence” fails to order error or coverage is too small |
| Calibration | Held-out prespecified method | uncalibrated; Platt/isotonic/temperature as development comparisons | Calibration fails under temporal/source/target shift |
| Randomness | Prespecified seeds | at least five training seeds on development; repeated group splits | Effect smaller than seed/split variance |
| External overlap | Remove known/similar overlap | known-overlap, post-cutoff, and overlap-unknown strata | Claimed advantage confined to likely training overlap |

## Representation ablations

Run only with identical label/split populations:

1. constant/target/compound priors;
2. scalar physicochemical descriptors;
3. fingerprint only;
4. molecular graph only;
5. protein sequence only;
6. ligand + protein without cross-interaction module;
7. ligand + protein with interaction module;
8. structure/pocket features only where structures are available;
9. full multimodal system;
10. full system with each modality removed in turn;
11. full system with provenance/assay context removed;
12. missing-modality and modality-dropout variants.

The purpose is to identify incremental information, not to declare causal mechanisms. A feature is useful only if it improves the difficult prespecified split without unacceptable coverage, calibration, or compute cost.

## Leakage and sanity tests

- y-scramble globally and within assay/source/target strata;
- permute protein targets while preserving ligand/source marginals;
- permute ligands within target/assay;
- train on identifiers/provenance fields alone as a leakage detector;
- compare with molecular weight, target prevalence, and nearest-neighbor baselines;
- deliberately insert an exact duplicate in a test fixture and require the audit to fail;
- assert no preprocessing/normalization vocabulary is fitted on validation/test;
- remove every train observation within the external model’s known cutoff/similarity envelope where possible;
- test that unavailable labels cannot enter features, sampling, early stopping, or loss masks.

Chance-like scrambled performance is necessary but does not prove absence of leakage. Identifier/provenance performance above a trivial level requires investigation.

## hERG/QT-specific sensitivities

- exact functional `IC50` only versus broader hERG families;
- temperature, cell system, protocol, incubation and dynamic/static strata;
- continuous potency versus blocker/ambiguous/nonblocker thresholds;
- conflicting-label compounds retained as conflict class versus removed;
- source and publication holdout;
- multichannel or clinical QT evidence evaluated separately, never merged as the same label;
- exposure-margin analyses only with compatible unbound clinical exposure and uncertainty.

## PK-specific sensitivities

- species, route, dose, formulation, matrix, analyte, fed/fasted, and single/repeat-dose strata;
- raw concentration-time model versus source-reported NCA statistics;
- dose-normalized views only after testing dose proportionality;
- parent versus metabolite, total versus unbound, and blood versus plasma separation;
- study/subject holdout and leave-study-source-out;
- closure/consistency checks (dose, clearance, AUC, bioavailability) with suspect records excluded;
- complete protocol subset versus broad sparse inventory.

## Structure/physics-specific sensitivities

- experimental holo, apo, predicted, and template-free inputs;
- alternative protonation/tautomer/stereo states and cofactors/waters/metals;
- pocket definition and construct boundaries;
- pose ensemble versus top pose; confidence-selection bias;
- docking/ML/physics baselines under matched preparation;
- repeated seeds and independent replicas;
- compute budget and convergence criteria;
- external model version/weights and training-cutoff overlap.

Production HPC results remain `hardware_blocked` until jobs, inputs, environments, logs, convergence/QC, and artifact checksums are returned and reviewed. A local smoke output cannot satisfy this cell.

## Decision rules

- Designate one primary analysis and all alternatives before lockbox evaluation.
- Summarize variant effects as paired differences with cluster-aware intervals, not a table of only favorable metrics.
- A conclusion is robust only if its sign and practical interpretation survive all critical plausible variants.
- If a reasonable variant reverses the conclusion, narrow the claim and explain why; do not select the preferred pipeline retrospectively.
- Ablation results identify dependence, not causal importance or mechanism.
- All unrun required cells receive `planned`, `unavailable`, `license_blocked`, or `hardware_blocked` with reason—never an empty cell that looks like success.
