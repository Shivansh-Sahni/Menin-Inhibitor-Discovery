# Architecture

## Design goal

Menin-Edit is a decision engine around molecular changes, not another endpoint dashboard. Its central object is an auditable path:

`starting inhibitor -> supported edit -> intermediate stopping point -> supported edit -> candidate`

Each edge records what changed and each node records the predicted endpoint state, uncertainty, applicability, hard-bound result, and ranking. This representation supports the practical question: *Which small change should we make next, what is it expected to improve or damage, and where should we stop?*

## System flow

```mermaid
flowchart LR
    A["Starting SMILES + objectives + bounds"] --> B["Canonicalize and score baseline"]
    B --> C["Single-cut fragmentation"]
    C --> D["Apply evidence-supported replacement rules"]
    D --> E["Chemical and similarity guards"]
    E --> F["Endpoint predictor registry"]
    F --> G["Per-step effects and evidence disagreement"]
    G --> H["Conservative hard-bound evaluation"]
    H --> I["Pareto fronts"]
    I --> J["Priorities within front + diverse beam"]
    J --> C
    J --> K["Ranked stopping points and path reports"]
```

The engine does not generate arbitrary decorations. By default, a candidate must be produced by a single-cut variable-fragment replacement observed in the edit library. This makes the proposal space smaller, more interpretable, and easier to audit.

## Components

### Configuration and schemas

`config/default.yaml` declares models, endpoints, objective priorities, hard bounds, search guards, and edit-library paths. `schemas.py` defines immutable request/result objects for endpoints, estimates, edits, effects, candidates, constraint evaluations, rankings, and complete sessions. Stable JSON serialization lets the same contract support the current CLI and a future API or user interface.

Important request concepts are separate:

- An **objective** says which direction is preferable and how important the endpoint is after Pareto sorting.
- A **target** prevents extra credit beyond a sufficient endpoint level.
- A **constraint** is a hard acceptability limit, not a soft weight.
- A **search specification** limits path length, branching, edit size, retained diversity, and structural similarity.

### Governed data boundary

`data.py` reads the historical wide workbook without writing it. It standardizes structures, generates keyed pseudonyms, preserves `<`, `>`, `<=`, `>=`, `~`, and exact relations, converts concentration values to directionally correct pIC50 bounds, retains assay/species/route/dose context, and flags duplicate or conflicting records.

The loader assigns every row a role:

- `train` and `development` may contribute to edit discovery;
- `locked_external` and `prospective_blind` are evaluation-only;
- attempts to construct edit evidence from evaluation-only roles are rejected.

`prepare-lab` and `train-lab` resolve the configured repository and requested output before accessing the workbook. They reject the repository itself, every descendant, and external-looking symlinks that resolve back into it. The workbook, pseudonymization key, derived tables, and private models remain in approved storage.

### Edit library

`edits.py` converts matched molecular pairs into bidirectional rules. For each directed fragment replacement it retains support, endpoint mean and dispersion, source scope, split role, and individual evidence records. The public library currently contains only observed Menin pIC50 deltas. The historical data builder emits vector-valued evidence for Menin binding, cellular assays, hERG, and rat PK. The checked-in configuration is public-only; setting `edit_library.private_pairs` to the approved external evidence CSV makes the engine load and merge the public and private libraries.

At search time, the engine:

1. identifies valid single-bond cuts with a sufficiently large retained core;
2. looks up rules whose source fragment matches the parent's variable fragment;
3. joins the retained core to the observed target fragment;
4. rejects invalid products, excessive heavy-atom changes, insufficient parent similarity, visited structures, and duplicate products;
5. keeps the most strongly supported rule when several cuts yield the same product.

### Predictor registry

All predictors return one common `PropertyEstimate`: endpoint key, mean, lower and upper estimate, applicability flag, model version, evidence status, and metadata. Predictions are cached by endpoint and canonical SMILES and are batched over new structures.

The checked-in public registry contains:

- `menin_biochemical_pIC50`: existing public Extra Trees regression artifact, a configured conformal-style interval, and a Morgan-similarity domain check;
- `herg_public_blocker_probability`: existing calibrated public Extra Trees classifier and public domain check;
- `herg_consensus_probability`: a stable one-member wrapper around `herg_public_blocker_probability`; this preserves the endpoint contract but is explicitly not independent model corroboration;
- `structural_alert_count`: deterministic PAINS/BRENK/NIH matches, explicitly labeled as review flags rather than toxicity.

The implemented `herg_private_ensemble_probability` adapter remains disabled by default because its benchmark artifacts are governed and absent from a clean public checkout. An approved local configuration may enable it and append it to the consensus members. In that two-component mode, the consensus uses the maximum blocker probability, the envelope of component bounds, and in-domain status only when every member is in domain. The local configuration and private artifacts must remain outside Git.

`local_models.py` implements an additional governed route for historical continuous endpoints. It excludes censored, conflicted, non-development, and non-finite rows; groups cross-validation by scaffold; calibrates an out-of-fold residual interval; compares with a fold-specific median baseline; derives a local similarity domain; and writes a hash-verified private artifact. The registry supports `kind: local_lab_regression` and fails closed unless the artifact's manifest passes the configured recommendation gate. Local artifacts are not enabled in the checked-in public configuration.

Public `skops` artifacts are SHA-256 checked before loading. Private `joblib` artifacts are treated as local trusted executable objects, and native boosting members are isolated to reduce runtime-library conflicts. Model hashes and versions are propagated to the session manifest.

### Stepwise search

`engine.py` scores the baseline, then performs bounded beam search. Each child has one parent, one molecular edit, a prediction vector, endpoint effects, feasibility state, violations, and path cost. The default maximum depth is three, beam width is 24, and each parent contributes at most 80 generated products.

The path cost adds configured penalties for another editing step, endpoint-interval width, and edit size. A diversity filter prevents nearly identical structures from occupying the whole beam. Node and session identifiers are deterministic hashes of the relevant structures, rules, and request.

### Per-step attribution

For endpoint `e`, the conditional change at a step is:

`delta_e(step) = prediction_e(product) - prediction_e(parent)`

The desirability delta reverses sign for minimized endpoints, so positive always means helpful. The interval for a delta is conservatively formed from the two absolute prediction intervals. When a matched-pair observed delta exists, the report shows both it and the absolute-model delta, plus their disagreement.

This is local accounting, not causal attribution. The step deltas telescope exactly to final prediction minus baseline prediction, which makes a multi-edit proposal understandable without pretending the steps were independently tested.

### Hard bounds

A constraint can be:

- absolute, such as Menin pIC50 `>= 6.5`;
- relative to the start, such as hERG probability no more than `+0.05` versus the original inhibitor;
- relative to the previous step, such as never losing more than `0.2` pIC50 in one edit;
- checked after every edit or only at a final stopping point.

For a maximize endpoint, the lower estimate must clear a `>=` limit. For a minimize endpoint, the upper estimate must clear a `<=` limit. Missing and out-of-domain behavior is configured independently as `reject`, `warn`, or, for objective scoring, `ignore`.

An ancestor that fails an each-step bound makes its descendants unacceptable; a final-only miss at an intermediate may still be rescued by a later edit. The current predictors expose fixed or model-specific intervals. Although the request stores a confidence level, arbitrary confidence-specific interval recalculation is a planned extension.

### Pareto, then priorities

Only feasible candidates with usable objective evidence enter non-dominated sorting. Every endpoint is transformed to a higher-is-better conservative value: lower bound for a maximized endpoint and negative upper bound for a minimized endpoint. Warned out-of-domain or missing objectives receive worst-case credit; rejected ones are ineligible.

Candidates are ordered by:

1. eligibility;
2. Pareto front;
3. normalized user-priority score within that front;
4. lower path cost;
5. deterministic node identifier.

This order preserves trade-offs. Priority sliders express preference among non-dominated alternatives; they cannot hide that one candidate is worse on every active objective.

### Explanations and sessions

The CLI writes full machine-readable results and compact Markdown. Each path states the transformation, parent and product, support count, context similarity, every endpoint's before/after value and conditional delta, applicability, evidence grade, constraint status, and whether the user can stop there.

The Python engine keeps sessions in memory. It can re-rank cached candidates under new objectives and constraints, or start a new search from any intermediate node. Persistent server-side session storage, authentication, and multi-user access are not implemented yet.

## Extension points

New endpoints should implement the common predictor protocol and declare an `EndpointSpec`. A future direct edit-delta model, DILI/Ames classifier, or structure-based binding score can therefore participate in the same constraints, Pareto logic, cache, and explanations without rewriting the search algorithm. Historical absolute regression already uses this extension route.

The key condition is semantic clarity. Menin biochemical pIC50, cell-killing potency, hERG block, DILI risk, and docking confidence are different endpoints and must not be collapsed into an ambiguous "activity" or "toxicity" score.
