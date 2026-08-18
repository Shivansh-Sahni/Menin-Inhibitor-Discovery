# Menin-Edit

Menin-Edit is an explainable molecular-editing engine for Menin inhibitor optimization. Given a starting molecule, it proposes small, experimentally precedented changes, predicts how each change affects Menin biochemical potency and hERG liability, enforces user-set limits, and returns complete edit paths that can be stopped after any acceptable step.

The current release is a working computational vertical slice, not a claim that a drug candidate has been discovered. It is designed to turn the substantial public and historical work already in this repository into auditable design hypotheses.

## What works now

- An installable Python package and CLI, layered on the repository's `menin-discovery` runtime, with versioned, JSON-serializable request and result schemas.
- A bidirectional edit library built from 3,899 public Menin matched molecular pairs: 5,824 directed rules and 7,798 directed evidence records in the current audit.
- One-to-three-step, beam-guided molecular editing with core-size, changed-atom, parent-similarity, duplicate, and cycle guards.
- Artifact-hash-verified prediction adapters for the existing public Menin biochemical pIC50 model and calibrated public hERG classifier.
- A public-only default hERG path backed by the calibrated public classifier. The private equal-importance quick ensemble and public/private maximum-probability consensus remain available only through an explicit governed local configuration.
- Per-step before/after predictions, conditional deltas, prediction intervals, applicability-domain status, matched-pair support, model-versus-observed-delta disagreement, and a telescoping check from the starting molecule to the final proposal.
- Hard bounds that can be absolute, relative to the starting molecule, or relative to the preceding step; bounds can apply at every step or only to the final structure.
- Uncertainty-aware Pareto sorting first, followed by user priorities only within a Pareto front. A large weight therefore cannot promote a dominated molecule above a better front.
- Re-scoring cached candidates with different priorities and bounds without rerunning molecular models, plus continuation from any retained intermediate node through the Python API.
- A governed historical-workbook loader that preserves censoring and assay context, pseudonymizes identifiers, flags conflicts, and can create multi-endpoint matched-pair evidence without copying the workbook into the repository.
- A scaffold-cross-validated local regression trainer, hash-verified local predictor, and fail-closed registry gate: a private model is rejected from optimization unless it improves median-baseline MAE by at least 5% and has out-of-fold R² of at least 0.05.

The default active optimization endpoints are Menin biochemical pIC50, public hERG blocker probability, and structural-alert count. The stable `herg_consensus_probability` endpoint is a one-member wrapper around the public classifier in the checked-in configuration; it is not evidence of independent model agreement. The alert count is a review proxy, not a toxicity prediction. DILI and Ames endpoint contracts are present but intentionally disabled until governed models are trained and validated. No docking, free-energy, pose-confidence, mutant-Menin, or general-toxicity model is claimed as complete.

## How a decision is made

For each retained parent, Menin-Edit fragments one bond at a time and applies only replacement rules observed in the eligible evidence library. Chemically valid products are scored once and cached. Hard limits remove unacceptable stopping points. Remaining candidates are compared using the conservative side of each prediction interval, assigned to Pareto fronts, and ordered within each front using the user's priorities, uncertainty penalty, path complexity, and deterministic tie-breakers.

Every intermediate node remains a visible stopping point. A user can therefore choose a one-step molecule that already satisfies the limits instead of being forced to accept a more complex final design. The explanation reports a change as a *predicted conditional effect*, not as experimental causality.

## Quick start

From the repository root, install the shared runtime and Menin-Edit together:

```bash
python -m pip install -e '.[dev]' -e './packages/menin-edit[lab,dev]'
cd packages/menin-edit
menin-edit audit
menin-edit audit --load-models
pytest -q
```

The checked-in configuration never loads ignored private benchmark artifacts. For approved local work, copy `config/default.yaml` to governed storage, update its repository/public-artifact paths, point `herg_private_ensemble_probability.benchmark_root` at approved external storage, set that predictor's `enabled` flag to `true`, and append it to the consensus `members`. Keep that local configuration and every referenced private artifact outside Git. Only then does `herg_consensus_probability` take the maximum component probability and envelope their intervals.

Score one structure:

```bash
menin-edit score \
  --smiles 'CCN(C(C)C)C(=O)C1=C(C=CC(=C1)F)OC2=CN=CN=C2N3CC4(C3)CCN(CC4)CC5CCC(CC5)NS(=O)(=O)CC'
```

Run the example bounded search:

```bash
menin-edit optimize \
  --request examples/request.yaml \
  --output-dir artifacts/sessions/example
```

The tutorial request is deliberately permissive so it produces a small software demonstration: Menin lower bound 5.0, hERG upper bound 0.98, and warning-only OOD handling. It is not the production policy. The checked-in default uses the stricter 6.5 Menin and 0.70 hERG limits; production use should reject out-of-domain decisions unless a documented endpoint-specific exception is intended.

An optimization writes:

- `result.json`: the baseline, every retained candidate, endpoint estimates, effects, feasibility, and ranking;
- `paths.json`: step-by-step explanations and evidence records;
- `report.md`: a compact human-readable ranking and the leading edit paths;
- `manifest.json`: request/result hashes, model versions, edit-library summary, and interpretation boundary.

A run with zero proposed candidates is a valid result: it means no public evidence-supported edit passed the configured structural guards for that starting structure. It is not permission to silently switch to unsupported generation.

## Priorities and hard limits

Edit `examples/request.yaml` or supply another request file. Objectives express what to improve; constraints express what must not be crossed. The default is 55% Menin potency, 35% hERG, and 10% alert minimization, with production-default hard bounds of Menin pIC50 at least 6.5, hERG blocker probability at most 0.70, and at most two alert matches.

Priorities do not replace Pareto optimization. They resolve choices within the same non-dominated front. Limits are evaluated conservatively: a maximize endpoint uses its lower estimate, while a minimize endpoint uses its upper estimate. Current interval levels are predictor-specific; the request's `confidence` field is retained for the public contract but does not yet recalculate arbitrary confidence levels.

The `target` objective field caps utility once a desired level is reached. `minimum_meaningful_gain` is currently schema/configuration metadata and is not yet used to prune or rank candidates.

## Historical lab data

The attached workbook is deliberately not copied into this folder. Audit it in memory with a runtime-only pseudonymization secret:

```bash
export MENIN_EDIT_HMAC_KEY='replace-with-an-approved-secret-at-least-16-bytes'
menin-edit lab-audit \
  --workbook /approved/private/storage/Combined_menin_full_profile.xlsx \
  --cohort-role development
```

To prepare governed derivative tables, the CLI requires an output directory outside the Git repository:

```bash
menin-edit prepare-lab \
  --workbook /approved/private/storage/Combined_menin_full_profile.xlsx \
  --cohort-role development \
  --output-dir /approved/private/storage/menin-edit-prepared
```

Train the four default historical regressors under scaffold-grouped validation:

```bash
menin-edit train-lab \
  --workbook /approved/private/storage/Combined_menin_full_profile.xlsx \
  --cohort-role development \
  --output-dir /approved/private/storage/menin-edit-models
```

Training writes a private validation summary and `accepted_registry_snippet.yaml` containing only models that clear the default scaffold-OOF gate. Both `prepare-lab` and `train-lab` resolve and reject any output inside the repository before opening the workbook or fitting a model. Training never silently replaces the active Menin or hERG artifact.

The current workbook contains 111 source rows and 110 unique standardized structures. The loader produces 1,165 assay-context observations and, under the development role and current exact-only settings, 14,778 directed multi-endpoint evidence rows across 1,610 directed transformations. The checked-in default remains public-only, but an approved `edit_library.private_pairs` path activates an in-memory public/private merge now.

The first local-model audit has also been run without retaining artifacts in the repository. MV4;11 and MOLM13 passed the default scaffold-OOF gate. Menin binding did not beat the median MAE baseline, and numeric hERG did not meet the combined improvement/R² gate, so both remain disabled. The historical hERG ensemble is likewise disabled in the checked-in public configuration and requires explicit governed opt-in; this fail-closed result is more useful than forcing every available endpoint into production.

Use `train` or `development` only for discovery. `locked_external` and `prospective_blind` rows are rejected from edit-rule construction and remain evaluation-only.

## Scientific boundary

The public Menin model is useful background but is a weak basis for absolute predictions in the historical series: none of the 110 unique lab structures falls inside its recorded applicability domain. Its temporal test MAE is 1.302 log units with negative R². The system therefore exposes domain status, applies worst-case credit to warned out-of-domain objectives, and treats the historical series as essential local evidence rather than as a decorative external dataset.

The hERG work is retained rather than discarded. The public default uses the calibrated public classifier. An approved local configuration can additionally expose the historical/public ensemble, explicit model disagreement, private-chemistry domain status, and a conservative maximum-probability consensus. The one-member public wrapper is not independent corroboration, and the nested unseen-scaffold estimates for the optional ensemble are uncertain. Every hERG output is a ranking aid and project-defined blocker probability, not a clinical cardiotoxicity probability.

See [Architecture](docs/architecture.md), [Implementation plan](docs/implementation_plan.md), [Data and validation](docs/data_and_validation.md), [Roadmap](docs/roadmap.md), and [Novelty and positioning](docs/novelty_and_positioning.md) for the complete plan and evidence boundary.
