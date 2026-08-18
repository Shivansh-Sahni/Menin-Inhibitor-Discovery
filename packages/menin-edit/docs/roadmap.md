# Roadmap

## Priority order

The platform should improve local Menin decision quality before adding more endpoint names or more elaborate neural models. The highest-value sequence is:

1. integrate and validate the historical series;
2. strengthen Menin and cellular local predictions;
3. productionize the existing hERG ensemble and edit evidence;
4. deliver the stop-anywhere decision interface;
5. add defined toxicity endpoints;
6. add a nonredundant binding/structure endpoint only if it improves decisions.

This roadmap assumes no new compound testing. It focuses on retrospective modeling, held-out validation, software, and productization using the data already available. Experimental confirmation remains a later scientific requirement, not a prerequisite for building the platform.

## Immediate: freeze and package the verified system

**Target: one to two focused days**

- Freeze the package environment and archive the now-passing complete unit/smoke suite.
- Archive the completed `audit`, model-loading, molecule-score, and 40-candidate tutorial manifests and runtimes.
- Retain the implemented end-to-end synthetic engine test with known edits, bounds, Pareto fronts, and stop points as a release gate.
- Retain the verified public-only mode when no private-pair path is configured.
- Confirm that an approved external `private_pairs` CSV is merged when configured and that evaluation-only roles are excluded.
- Audit the historical workbook in memory and export governed tables only to approved external storage.
- Adjudicate the duplicate structure and review the 13 automatically conflicted structure groups.
- Freeze a private split-role file before retaining or registering any local artifact. A first temporary scaffold-CV audit is complete, but it is development evidence rather than a locked result.

**Deliverable:** versioned public-only demo plus a hashed, private prepared-data package and zero-leakage audit.

## Next: local Menin, cellular, and edit models

**Target: first useful internal model within one week**

- Repeat the implemented scaffold-CV training under the frozen roles and retain approved artifacts for MV4;11 and MOLM13, which passed the first default gate (OOF MAE 0.3880 versus 0.4366 and 0.3257 versus 0.3849; OOF R² 0.136 and 0.259).
- Keep the first Menin absolute model disabled (OOF MAE 0.1361 versus median 0.1320; R² -0.036) and focus on direct local edit deltas, series-aware baselines, or better feature/data formulations.
- Keep the numeric local hERG regressor disabled (OOF MAE 0.4023 versus 0.4265; R² 0.010) and retain the existing hERG ensemble.
- Register only artifacts that improve baseline MAE by at least 5%, have OOF R² of at least 0.05, and show usable interval coverage.
- Compare public-only, local-only, nearest-neighbor, and matched-pair baselines on untouched scaffold groups.
- Add direct edit-delta learners for Menin first, then the two cell endpoints if support is adequate.
- Combine absolute and delta estimates without hiding disagreement.
- Use HL60 only as a lower-bound selectivity constraint; do not train a generic toxicity model from its uniformly censored values.
- Expose source context in explanations: same-series private, cross-series private, same-assay public, or cross-context public.

**Deliverable:** a private configuration that uses accepted cellular artifacts and merged multi-endpoint edit evidence, while Menin remains evidence-only/public-prior until a local artifact passes, with a validation card for every endpoint.

**Go/no-go:** if a local model fails its baseline, leave it disabled and use the direct evidence/domain warning instead. Small data does not become credible by adding model complexity.

## Then: retain and tighten the hERG work

**Target: one to two weeks in parallel with local Menin work**

- Freeze the selected equal-importance ensemble and manifest its exact feature/model/calibration lineage.
- Add observed-hERG override behavior for compounds that already have decisive historical measurements.
- Add historical edit-level hERG deltas to each transformation explanation.
- Re-evaluate public model, private ensemble, and conservative consensus on locked scaffold groups.
- Report component probabilities, ensemble spread, public/private domain status, and disagreement together.
- Add exposure-aware *reporting* using rat Cmax/AUC as separate observed/predicted fields; do not call it a human cardiac safety margin.
- Add an abstention mode when every relevant model is out of domain or disagreement exceeds a configured maximum.

**Deliverable:** one defensible hERG research endpoint that uses all existing work and clearly separates measured evidence, model probability, model disagreement, and domain.

## Product layer: interactive stop-anywhere optimization

**Target: two to four weeks after stable endpoint contracts**

Build a thin interface around the existing request/result schemas:

- draw or paste the starting molecule;
- select endpoint priorities;
- set absolute, start-relative, or step-relative limits;
- run a bounded edit search;
- inspect a path as highlighted one-step transformations;
- view endpoint trajectories, intervals, applicability, evidence grade, and violations;
- stop at any feasible intermediate;
- re-rank cached results without rerunning models;
- continue searching from a chosen intermediate;
- compare alternatives on a Pareto plot and export JSON/Markdown.

The first UI should be a single-user internal application. Persistent sessions, authentication, access logs, private/public storage separation, and request quotas come before multi-user deployment.

**Deliverable:** an internal Menin-Edit workbench that reproduces CLI results exactly.

## Expand endpoints carefully

**Target: one to two months, after the Menin/hERG path is validated**

### Toxicity

- Select named public endpoints with usable data and licensing, initially Ames and DILI.
- Build source/scaffold/temporal splits, calibration, uncertainty, and domains.
- Register only models that meet endpoint-specific gates.
- Keep structural alerts as a separate heuristic review layer.
- Register the implemented governed `skops` classifier adapter only when a named endpoint's artifact and validation can be pinned.

### Binding

- Decide whether this means pose retention, interaction fingerprints, docking score, or mutation-specific potency.
- Prefer a nonredundant pose/interaction-confidence endpoint over another ligand-only potency proxy.
- Validate with known Menin ligands, redocking controls, enrichment, and ablations against ligand-only models.
- Defer mutant-resistance optimization until labels exist; the current workbook has no populated WT or mutant-Menin IC50 values.

### PK and exposure

- Fit rat PK models only for endpoints with roughly 48–50 usable observations and only if grouped validation beats simple baselines.
- Keep species and route explicit.
- Treat sparse mouse/stability/PPB data as context, not primary learned endpoints.

**Deliverable:** additional endpoint cards, each with a precise meaning, data contract, validation report, and enable/disable decision.

## Generalize beyond Menin

The reusable product should separate a target pack from the core engine:

- target-specific activity and structure-based predictors;
- project-specific edit evidence;
- shared hERG/toxicity/PK endpoints;
- shared constraints, Pareto ranking, path explanations, caching, and UI.

Menin-Edit remains the reference implementation because it has a coherent public dataset, a historical local series, and substantial hERG work. A second target should be attempted only after the Menin benchmark shows that stepwise, evidence-supported optimization improves held-out decisions.

## Decision gates

| Gate | Evidence required | Action if it fails |
|---|---|---|
| Historical-data readiness | Frozen roles, conflict review, no structure/scaffold leakage | Stop model fitting; repair governance |
| Local endpoint value | Scaffold-OOF improvement over median/nearest/public baselines | Disable model; retain evidence-only endpoint |
| hERG value | Locked ranking/calibration improvement or complementary disagreement value | Keep as warning/abstention signal, not objective |
| Optimization value | Better held-out feasible-edit ranking than weighted sum and random supported edits | Simplify search/ranking and re-evaluate |
| Toxicity readiness | Named endpoint, governed dataset, calibrated held-out performance, domain | Leave endpoint disabled |
| Binding readiness | Nonredundant signal and validated pose/ranking benefit | Do not add a decorative score |
| Platform release | Replayable sessions, private isolation, tests, endpoint cards | Keep CLI/internal research status |

## What not to do

- Do not call structural alerts toxicity.
- Do not treat HL60 or hERG as a universal toxicity label.
- Do not pool locked/prospective data into training for a larger sample count.
- Do not report random-split performance as evidence of new-series generalization.
- Do not let user priorities override hard limits or Pareto dominance.
- Do not hide out-of-domain estimates in a single composite score.
- Do not add an unconstrained generative model until it beats the supported-edit baseline on held-out decisions.
- Do not claim a mutation barrier, exposure window, or drug-like safety improvement without labels and appropriate validation.

## Platform success measures

The computational platform is succeeding when:

- users can identify the best next supported edit and understand its trade-offs;
- a multi-edit idea is decomposed into meaningful, selectable stopping points;
- changing priorities changes the ordering without changing the underlying evidence;
- hard limits are never silently traded away;
- local historical evidence improves or appropriately overrides public priors;
- hERG work contributes measurable ranking or uncertainty value;
- held-out known outcomes are prioritized better than simple baselines;
- every result can be replayed from its request, data/model versions, and hashes.
