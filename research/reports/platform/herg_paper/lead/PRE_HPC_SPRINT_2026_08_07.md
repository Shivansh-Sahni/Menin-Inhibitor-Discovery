# Pre-HPC sprint reconciliation — 2026-08-07

## Completed without training or feature generation

- **Evaluation-candidate queue:** 226 quantitative functional observations
  across 123 structures: 217 exact and nine correctly one-sided-censored.
  None explicitly confirms WT in its source metadata, so every row remains
  candidate-only pending target-status/lineage adjudication and preferably
  independent retesting.
- **Conflict-review queue:** 1,340 structures with broader exact standardized
  IC50/pIC50 ranges greater than `1e-6` pIC50. The tolerance excludes 6,699
  float-conversion-noise cases from the superseded first build. There are 112
  critical ranges ≥2 and 121 high ranges ≥1.
- **Protocol-enrichment queue:** 4,779 incomplete assays covering 404,890
  mutually exclusive observation assignments; missing host, platform, voltage,
  temperature, time, and recording fields remain unresolved rather than
  inferred.
- **Benchmark freeze:** nine label-blind materialized challenges containing
  711,047 challenge-membership rows and three explicit blockers. The builder
  projects routing columns only and never reads or embeds target relation,
  values, bounds, or classes.
- **Future feature contract:** 12 input-feature families plus separate post-fit
  applicability/uncertainty outputs, canonical `structure_id` joins, exact
  RDKit 2026.03.3 compatibility, per-level nullable IDs, and explicit
  provenance/failure/leakage rules.
- **State aggregation contract:** pH 7.4, 298.15 K, 0.15 mol/L ionic strength,
  within-state energy zeros, population cutoff/renormalization, and prohibition
  on comparing conformer energies across protonation states.
- **Competitor readiness:** machine/human parity for all 15 systems and all 14
  implementation priorities across code, checkpoint, license, inputs,
  preprocessing, overlap, adapter, blocker, and next action.
- **HPC execution contract:** seven estimated stages, content-addressed storage,
  atomic publication, checkpoint/retry/determinism rules, and an unexecuted
  10–100 valid-molecule smoke specification with quarantined/synthetic negative
  controls kept outside coverage and training.

## Accepted releases

- Pre-HPC review assets manifest:
  `cca60c80eb4b817806e770d88d403854173129fb9256104da83827bb691a4dab`.
- Label-blind benchmark-freeze manifest:
  `277675187ad7b475ebeb59c875291f7b69783a1e40ed853b0cb221efc0bb9b1e`.
- Contract validator: PASS for feature, identity, aggregation, competitor,
  priority, HPC-stage, and smoke-test consistency.

## Verification and boundaries

Seven focused tests, Ruff, Ruff formatting, mypy, both production validators,
source/master/artifact/schema rebinding, exact challenge/registry replay,
structure/scaffold exclusivity, and `git diff --check` pass. No model, feature,
competitor reproduction, smoke test, or HPC job was run. No candidate was
promoted to an adjudicated gold standard. No purge was performed.
