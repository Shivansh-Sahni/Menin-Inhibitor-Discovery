# Implementation plan

## Outcome

The near-term objective is a credible Menin design platform that accepts an inhibitor, explores a small number of evidence-supported edits, predicts the effect of each step on several defined endpoints, respects user limits, and returns ranked, inspectable stopping points. It should be useful with existing data before any new compound testing and should remain honest where data are sparse.

## Completed vertical slice

The following is implemented in `Menin-Edit` now:

| Capability | Current implementation |
|---|---|
| Package boundary | Standalone `pyproject.toml`, configuration, CLI, schemas, and tests |
| Public edit evidence | 3,899 public Menin pairs converted to 5,824 bidirectional directed rules |
| Candidate generation | Observed one-cut replacements with chemical, size, similarity, duplicate, and cycle guards |
| Menin scoring | Existing public biochemical pIC50 artifact with hash verification, interval, and domain flag |
| hERG scoring | Checked-in calibrated public model plus optional governed historical/public quick ensemble and conservative consensus |
| Review alerts | PAINS/BRENK/NIH match counts, explicitly not labeled as toxicity |
| Multi-objective decision | Conservative hard bounds, Pareto fronts, priorities within fronts, uncertainty/path penalties |
| Explanations | One edit per step, before/after values, conditional deltas, evidence, disagreement, stop points, telescoping |
| Scenario exploration | Cached re-scoring with new priorities/bounds and continuation from an intermediate in Python |
| Historical data intake | Read-only workbook audit, censoring-aware normalization, pseudonymization, conflicts, cohort roles |
| Historical edit evidence | Multi-endpoint MMP construction for eligible private development data |
| Private integration | Optional external `private_pairs` merge plus `local_lab_regression` predictor registration |
| Local model training | Scaffold-grouped OOF regression, conformal-style interval, local domain, hash manifest, baseline/R² gate |
| Reproducibility | Deterministic identifiers, model/artifact hashes, request/result hashes, manifests |

This is enough to exercise the complete public-data path from starting structure to ranked explanation. It is not yet enough to claim reliable local Menin potency, broad toxicity, or binding-pose prediction for the historical chemical series.

## Fast path to a useful internal platform

### Stage 1 — integrate the historical series locally

**Goal:** use the most relevant SAR without exposing private rows or contaminating validation.

1. Freeze structure standardization and adjudicate the current conflict report. The exact duplicate-structure pair with conflicting assay values must remain one structure group, never two split observations; source identifiers stay in approved storage.
2. Assign `development`, `locked_external`, and, if available, `prospective_blind` roles before any modeling. Keep all variants of one standardized structure and closely related scaffold groups in one partition.
3. Run `lab-audit` and `prepare-lab` in approved storage. Record workbook and output hashes.
4. Build the private multi-endpoint edit library using only eligible development rows. Preserve exact versus censored delta bounds and source-level provenance.
5. Set `edit_library.private_pairs` to the approved external evidence CSV. The implemented engine merges it with public evidence in memory. Never commit merged evidence because it is derived from confidential structures and labels.
6. Add a source/context policy: local same-series evidence should be displayed separately from public cross-context evidence, not blindly averaged into one opaque number.

**Acceptance gate:** an audit shows zero locked/prospective rows in rules, zero structure leakage across partitions, conflict dispositions, stable hashes, and a path explanation that identifies public versus private support.

### Stage 2 — fit local absolute and delta models

**Goal:** stop treating the out-of-domain public Menin model as the main estimate for historical-series analogues.

The implemented Extra Trees trainer has already completed a first governed audit run, with artifacts retained only in temporary private storage:

| Endpoint | Eligible structures / scaffolds | Scaffold-OOF MAE | Median-baseline MAE | OOF R² | Default gate |
|---|---:|---:|---:|---:|---|
| Menin biochemical pIC50 | 82 / 29 | 0.1361 | 0.1320 | -0.036 | Fail; keep disabled |
| MV4;11 pIC50 | 78 / 29 | 0.3880 | 0.4366 | 0.136 | Pass |
| MOLM13 pIC50 | 77 / 29 | 0.3257 | 0.3849 | 0.259 | Pass |
| Numeric hERG pIC50 | 56 / 17 | 0.4023 | 0.4265 | 0.010 | Fail combined gate; keep existing ensemble |

The default recommendation gate requires at least 5% MAE improvement over the fold-specific median and OOF R² of at least 0.05. The result demonstrates fail-closed behavior: available data does not automatically mean an endpoint model is enabled. Next, freeze final development/locked roles and preserve the passing cellular models in approved storage before any configuration change.

Continue with a small, strong model panel rather than a large architecture search:

- absolute models for Menin biochemical pIC50, MV4;11 pIC50, and MOLM13 pIC50;
- a direct edit-delta model for Menin pIC50, followed by cellular deltas where support permits;
- a censored-regression or interval-loss strategy where qualified values materially contribute;
- public model output and molecular fingerprints/descriptors as optional priors/features, never as ground truth.

Evaluate structure-grouped and scaffold-grouped splits, plus a locked chronological or series holdout if source order supports one. Compare Extra Trees, random forest, regularized linear fingerprints, and one graph model only if it improves a locked metric. Calibrate uncertainty from out-of-fold residuals and fit applicability domains against the actual development chemistry.

Use direct delta and absolute predictions together:

- agreement increases confidence in an edit's direction;
- disagreement is surfaced as a warning and a useful ranking feature;
- no model should erase the raw matched-pair evidence.

**Acceptance gate:** local models beat the public-only baseline on untouched structure/scaffold groups, interval coverage is measured rather than assumed, and performance is reported separately in and out of domain. If they do not beat the baseline, retain the evidence-only local method instead of promoting a weak model.

### Stage 3 — productionize the hERG integration

**Goal:** keep all existing hERG work useful while avoiding overconfidence.

1. Package one frozen research ensemble and its complete feature/calibration/domain manifest.
2. Separate the rapid-development selection metrics from nested unseen-scaffold estimates in every report.
3. In the governed private configuration, retain the conservative maximum-probability consensus and expose public/private component probabilities, member spread, domain flags, and disagreement. Keep the checked-in default public-only.
4. Add an observed-value override for historical compounds with decisive hERG data; observed assay evidence should not be hidden behind a prediction.
5. Add edit-level hERG delta evidence from the historical MMP table. This directly answers which substitutions improved or damaged hERG within the series.
6. Validate ranking metrics on the locked historical set: pairwise order accuracy, high-liability enrichment, calibration, and performance by scaffold/domain.

**Acceptance gate:** no development metric is presented as external validation, every probability states its blocker definition, and the platform demonstrably preserves or improves hERG ranking on held-out structures compared with the public model alone.

### Stage 4 — add real toxicity endpoints

**Goal:** replace the ambiguous word "toxicity" with defined, validated endpoints.

Begin with two public-data classifiers that have reasonably reproducible labels, such as Ames mutagenicity and DILI concern. Each needs a dataset license/provenance manifest, structure deduplication, scaffold split, temporal or source holdout where possible, calibration, domain checks, and a clear label definition. Register them only after locked validation.

Keep structural alerts as a separate review endpoint. HL60 is a cellular selectivity control—all 87 populated source rows are censored at `>1000 nM`—and must not be relabeled as general toxicity. hERG is a cardiac ion-channel liability endpoint, not a substitute for systemic toxicity.

**Acceptance gate:** the UI and output never display a generic toxicity probability. It displays named risks with dataset, model version, uncertainty, domain, and validation results.

### Stage 5 — define and add a separate binding endpoint

**Goal:** decide what “binding” adds beyond the existing Menin biochemical IC50.

The current potency model already predicts a biochemical binding assay endpoint. A separate binding component should therefore be one of the following, chosen explicitly:

- pose plausibility and interaction retention against an approved Menin structure;
- docking/physics score used only as an orthogonal filter;
- mutation-specific potency or binding retention once mutant labels exist;
- predicted residence-time or kinetic proxy if suitable data become available.

Do not create another nominal “binding score” that is mathematically redundant with pIC50. A structure-based route requires protein preparation, binding-site definition, ligand-state enumeration, pose controls, known-ligand redocking, score directionality, and pose-confidence reporting.

**Acceptance gate:** the endpoint has a nonredundant biological meaning and improves held-out ranking or explanation beyond ligand-only potency. The current workbook contains zero populated WT/mutant Menin IC50 fields, so mutation-resistance claims cannot be trained from it today.

### Stage 6 — user-facing platform

**Goal:** let a medicinal-chemistry user explore trade-offs without editing YAML.

Build a thin API and interface around the existing schemas:

- structure input and rendering;
- endpoint importance sliders;
- absolute/start-relative/step-relative bound controls;
- path view with one molecular difference highlighted per step;
- endpoint trajectories and uncertainty/domain badges;
- “stop here,” “continue from here,” and “change priorities without rerunning” actions;
- compare two paths and export the full manifest/report.

The backend should be a queueable service with persistent sessions, authenticated access, private-data separation, resource limits, and deterministic replay. The UI must not bypass the same schemas and validation logic used by the CLI.

**Acceptance gate:** a user can reproduce a CLI session from the exported request, and a priority-only change reuses cached predictions while a structure/search change creates a new auditable run.

## Engineering work packages

### Tests

The current suite already covers deterministic multi-step paths, final and each-step bounds, Pareto order, cached rescoring, continuation, public artifact scoring, isolated private hERG scoring, fragment reconstruction, supported bidirectional edits, registry batching, censoring direction, pseudonymous lab intake, and split-role leakage rejection.

Before expanding model count, add the remaining release-level checks:

- a configured public/private library-merge smoke run;
- a larger stereochemistry and attachment-bond regression corpus;
- output replay and hash stability across fresh processes;
- explicit CLI no-private-write and repository-output-rejection tests;
- locked-role model-training rejection and accepted-registry-snippet replay.

### Performance

Profile only after correctness. The existing prediction cache and endpoint batching should be retained. Next optimizations are edit-library indexing by normalized fragment, cross-parent product deduplication before scoring, parallel endpoint batches, and persisted prediction caches keyed by structure plus model hash.

### Reproducibility and governance

Every release should freeze:

- source dataset hashes and license/provenance records;
- standardization version;
- split assignments and structure/scaffold overlap audits;
- model, feature, calibration, threshold, and domain manifests;
- edit-library role/source counts;
- request and result hashes;
- software environment and random seed.

## Definition of platform-ready

Menin-Edit is ready for serious retrospective design use when it can meet all of the following without new wet-lab testing:

1. The historical development series is integrated locally with zero locked-set leakage.
2. Menin, cellular, and hERG estimates are evaluated on untouched structure/scaffold groups and beat clearly defined baselines or fall back to evidence-only rules.
3. Hard bounds, Pareto fronts, priorities, and stop points are reproducible under unit and end-to-end tests.
4. Every endpoint has precise semantics; alerts are not called toxicity and docking is not called binding affinity.
5. Every path exposes uncertainty, domain, direct evidence, and model/evidence disagreement.
6. A user can rescore or stop at an intermediate without losing the audit trail.

Experimental validation remains necessary before making claims about actual potency, safety, or developability, but it is not required to complete the retrospective computational platform described here.
