# Novelty and positioning

## Short answer

Menin-Edit should not be described as completely unprecedented. QSAR, matched molecular pairs, multi-parameter optimization, Pareto ranking, molecular generation, applicability domains, and local explanations all have prior art. The strong and defensible novelty is their integration into a **Menin-specific, evidence-supported, stop-anywhere editing framework** that answers how each small structural change affects several endpoints under explicit limits.

The most compelling claim is not “we invented multi-objective drug design.” It is:

> Menin-Edit represents optimization as a sequence of experimentally precedented molecular edits, estimates each edit's conditional effect on potency and liabilities, compares model predictions with observed transformation evidence, enforces user-defined uncertainty-aware bounds at every selectable stopping point, and ranks only non-dominated feasible paths.

That is specific, testable, useful, and broader than a single QSAR model without being inflated.

## Core research question

**Can an evidence-grounded, stepwise molecular-edit framework prioritize Menin inhibitor changes that preserve or improve biochemical/cellular potency while reducing hERG liability and respecting user-defined endpoint limits, more reliably and transparently than single-endpoint QSAR or one-shot weighted optimization?**

This question can be answered retrospectively with held-out public and historical data. It does not require claiming prospective compound validation.

## What is distinct

### The edit—not only the molecule—is the prediction object

Most endpoint models score complete molecules. Menin-Edit preserves those absolute scores but also makes each edge in a design path explicit. It reports the conditional parent-to-product delta, its interval, the corresponding observed matched-pair delta when available, and their disagreement. This directly supports medicinal-chemistry decisions such as “replace this group first, then decide whether the second change is still worth making.”

### Complex designs become selectable sequences

A three-part proposal is not explained with one global attribution map. It is decomposed into three chemically valid single edits, and the user can stop after step one or two if the endpoint balance is already acceptable. The telescoping accounting guarantees that the sum of reported step deltas matches the model's final-minus-start change.

### Bounds and preferences have different authority

Hard limits are checked before ranking and can apply at each step. Pareto fronts preserve genuine trade-offs. User priorities act only within a front. This avoids a common failure of weighted sums: a sufficiently large potency weight cannot silently compensate for crossing a hERG limit.

### Public and local evidence have visible roles

The framework does not pretend that a large public model is locally reliable. All historical structures are outside the current public Menin domain, so public predictions are treated as priors and local matched-pair/model evidence is elevated. Public versus private, same-series versus cross-context, observed versus predicted, and in-domain versus out-of-domain evidence can remain separately visible.

### Existing hERG work becomes part of the decision, not a side report

The public calibrated model and historical/public ensemble are combined conservatively, with component values, model spread, domain status, and disagreement retained. Historical hERG edit deltas can be attached to the same transformations used for potency. This gives the earlier hERG program a direct role in selecting structural changes.

### The same decision grammar can generalize

Menin-Edit is target-specific at the evidence and activity-model layers, while constraints, Pareto sorting, stepwise paths, caching, provenance, and explanations are reusable. The longer-term platform can swap in another target pack without changing the decision grammar.

## Position relative to common approaches

| Related approach | Typical output | Menin-Edit difference | Evidence needed before claiming superiority |
|---|---|---|---|
| Single-endpoint QSAR | Potency score for each molecule | Joint potency/liability path with explicit limits and stop points | Better held-out feasible analogue ranking |
| Multi-task QSAR dashboard | Several property predictions | Converts predictions into supported edits, paths, constraints, and decisions | User-study/actionability and ranking benchmarks |
| Matched-pair analysis | Average effect of one transformation | Applies transformations prospectively in context and compares observed delta with absolute-model delta | Held-out edit-direction and delta-error improvement |
| Weighted multi-parameter score | One composite rank | Pareto first, priorities second; hard bounds cannot be traded away | Lower observed constraint-violation/regret |
| De novo/generative optimization | Many novel molecules | Restricts the initial search to observed, auditable one-cut replacements | Comparable or better hit/feasible rate at lower review burden |
| Attribution/saliency explanation | Atom or fragment importance map | Full before/after molecules and conditional effects for each actionable step | Explanation faithfulness and chemist preference |
| Generic hERG filter | One predicted probability | Public-only default with an optional governed public/private consensus, local domain, disagreement, and edit-level evidence | Locked-series ranking/calibration improvement |
| Docking-led design | Pose and score | Ligand/SAR decision engine now; structure-based evidence can be an orthogonal endpoint later | Incremental held-out value beyond ligand-only models |

The current system is plausibly superior in **auditability, controllability, and actionability** by design. It is not yet proven superior in predictive accuracy or molecule quality. Those claims require the benchmark and ablation plan below.

## Most defensible novelty layers

### Platform contribution

A versioned framework in which molecular transformations, multi-endpoint estimates, uncertainty, applicability, evidence, constraints, Pareto ranks, preferences, and stopping decisions are all first-class objects in one replayable session.

### Method contribution

The implemented decision rule combines conservative absolute endpoint estimates, path complexity and uncertainty, hard-bound feasibility, and non-dominated sorting followed by preference-based tie-breaking. In parallel, every step records observed transformation deltas and model-versus-evidence disagreement so the scientist can audit whether the absolute model agrees with local SAR.

A future benchmarked extension can incorporate calibrated direct-delta evidence into ranking itself. That extension should not be claimed as implemented until its fusion rule and ablations are frozen. The current candidate method novelty is the precise ordering of supported edit generation, conservative feasibility, Pareto ranking, preferences, and separately visible evidence disagreement—not any single ingredient.

### Menin scientific contribution

A systematic map of how local substituent changes trade Menin biochemical/cellular potency against hERG liability and available PK/selectivity endpoints within the historical series, while documenting where public Menin models fail to cover that chemistry.

### Human-decision contribution

A stop-anywhere interface that lets a scientist specify endpoint importance and non-negotiable limits, inspect which edit caused which predicted effect, and continue from the intermediate they judge most useful.

## How to prove value

Use a frozen retrospective benchmark with structure/scaffold-separated public and historical pairs. Compare Menin-Edit with:

1. public absolute models only;
2. local absolute models only;
3. nearest-neighbor and average matched-pair rules;
4. a conventional weighted sum;
5. Pareto ranking without stepwise paths;
6. random evidence-supported edits;
7. an unconstrained generator only if one is later introduced.

Primary system metrics should be:

- held-out edit-direction accuracy and delta MAE;
- recovery of observed non-dominated analogues;
- regret relative to the best observed feasible analogue;
- hard-bound violation rate;
- top-k enrichment for potency-preserving hERG improvements;
- calibration and interval coverage by applicability domain;
- path/explanation stability under bootstrap perturbation;
- time and number of structures a chemist must review to find a feasible option.

Ablate private evidence, direct deltas, conservative intervals, hard bounds, Pareto ordering, priorities, and the ability to stop early. This shows which parts contribute rather than attributing all gains to the full package.

## Claims that are ready now

- A working system decomposes supported Menin inhibitor proposals into one-step paths.
- It predicts and records per-step Menin, hERG, and alert changes under applicability and uncertainty metadata.
- It enforces absolute or relative bounds and preserves intermediate stopping points.
- It ranks feasible candidates with Pareto dominance before user priorities.
- It uses the public hERG model by default and can reuse the historical/public ensemble in a governed conservative consensus.
- It can ingest historical data under explicit cohort roles and build censored-aware multi-endpoint edit evidence.
- It implements a scaffold-CV local regression trainer and hash-verified predictor for governed historical endpoints, although no workbook-derived local artifact is enabled by default.

## Claims that are not ready

- The platform has discovered a more potent or safer Menin inhibitor.
- A recommended edit causally changes any endpoint.
- A hERG probability is a clinical cardiotoxicity or exposure-window estimate.
- Structural alerts are a toxicity model.
- The platform predicts mutant resistance or a high evolutionary barrier; the current workbook has no mutant labels.
- It predicts binding poses, affinity, free energy, or kinetics beyond the biochemical pIC50 model.
- It is more accurate than all related methods or completely novel.
- Retrospective cross-validation is prospective or wet-lab validation.

## Suggested title and abstract-level positioning

**Title:** *Menin-Edit: Evidence-Grounded, Constraint-Aware Stepwise Optimization of Menin Inhibitors Across Potency and hERG Liability*

**Positioning:** Menin-Edit converts multi-parameter molecular optimization from a one-shot molecule-ranking problem into a sequence of supported, auditable decisions. It couples absolute endpoint models with local transformation evidence, exposes uncertainty and applicability, enforces non-negotiable limits, and lets users stop at any intermediate Pareto-feasible structure. Menin is the first application because the project combines a substantial public activity set, thousands of matched pairs, a chemically local historical series, and an existing hERG modeling program.

## Path to a broader platform

After the Menin benchmark, the core can be positioned as a target-agnostic **EditPath** engine with Menin-Edit as its first target pack. Generalization is itself an empirical claim: demonstrate it on a second target with different chemistry and endpoint availability before presenting the framework as universal.
