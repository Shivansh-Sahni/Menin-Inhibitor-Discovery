# Protein--ligand affinity and potency training surfaces

## Outcome

This release creates a local, endpoint-separated protein--molecule training surface from the canonical ChEMBL 37 tasks and the independent portion of BindingDB 2026-08. It keeps real censoring, standardized parent structures, accession/sequence target identity, assay and document provenance, and leakage-safe split memberships.

## Scale

- Retained observations: 3,533,626.
- Primary Kd/Ki/IC50 observations: 3,304,480.
- Auxiliary EC50 observations: 229,146.
- Unique standardized ligand parents: 1,320,199.
- Unique leakage-grouped targets: 9,128.
- Unique ligand--target pairs: 2,214,530.
- Target-by-endpoint tasks: 15,061.
- Exact-structure scaffold fallbacks after a rare RDKit Murcko failure: 84.

### Kd

- Observations: 193,925.
- Unique ligand--target pairs: 145,281.
- Unique structures: 36,627.
- Target-specific tasks: 3,747.
- Endpoint role: primary protein--molecule task.

### Ki

- Observations: 661,274.
- Unique ligand--target pairs: 466,805.
- Unique structures: 243,486.
- Target-specific tasks: 3,317.
- Endpoint role: primary protein--molecule task.

### IC50

- Observations: 2,449,281.
- Unique ligand--target pairs: 1,516,789.
- Unique structures: 995,604.
- Target-specific tasks: 6,393.
- Endpoint role: primary protein--molecule task.

### EC50

- Observations: 229,146.
- Unique ligand--target pairs: 153,600.
- Unique structures: 117,421.
- Target-specific tasks: 1,604.
- Endpoint role: auxiliary potency only; it is not affinity.

## What was deliberately removed

- Explicit BindingDB rows sourced from ChEMBL: 1,658,155 endpoint records.
- Same-document exact ChEMBL mirrors found in non-ChEMBL BindingDB rows: 22,665.
- Same-document exact internal BindingDB mirrors: 42,460.
- Rights-pending source records: 274.
- Multi-chain targets, missing target identity, invalid parent structures, nonpositive labels, and unsupported label syntax were excluded and counted in the manifest.

## Scientific boundaries

- Kd, Ki, IC50, and EC50 remain separate. No endpoint substitution or silent pooling occurred.
- Less-than, greater-than, inclusive inequalities, and ChEMBL intervals remain bounds. No censored threshold was converted to an exact point.
- Binding free energy was not manufactured from IC50, Ki, or EC50. Kd-to-Delta-G also remains deferred where temperature and standard-state evidence are absent.
- Exact structure and scaffold groups cannot cross ligand-cold splits. Exact accession/sequence connected components cannot cross target-cold splits.
- Double-cold membership is available only when the independently assigned ligand and target splits agree; mixed assignments are explicitly ineligible rather than forced.
- Exact target identity leakage is controlled. Homology-level clustering is not claimed and remains a later sequence preprocessing step.
- The release is trainable data preparation. It contains no generated production features, fitted models, HPC execution, or evidence of predictive superiority.

## Recommended first training order

- Begin with IC50 because it is the largest primary surface, while retaining assay context and evaluating target-cold and double-cold generalization.
- Train Ki as the cleaner equilibrium-style primary complement, then Kd as the most direct affinity endpoint despite its smaller scale.
- Use EC50 only as an auxiliary potency task with its own head and metrics.
- Compare exact-only baselines against censor-aware objectives. Never use a censored threshold as an exact regression target.
- Report random-like performance only as a diagnostic; scaffold-cold, target-cold, and double-cold results are the meaningful generalization tests.
