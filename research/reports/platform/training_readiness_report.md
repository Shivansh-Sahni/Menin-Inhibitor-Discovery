# Training readiness report

Status as of 2026-08-04: preprocessing and fine-tuning interfaces are implemented, the accepted ChEMBL 37 canonical corpus has been materialized through real task-level integration, and the final mechanical artifact/source-rebinding gate passes. Scientific task claims and substantive large-model training remain not ready and not authorized; substantive training was not initiated.

## Completed interfaces

- Canonical public-only task-to-JSONL adaptation with exact access-class checks.
- Explicit observed/curated/derived label typing; prediction labels are prohibited. Derived labels cannot enter the default path: the separate sensitivity mode requires explicit sensitivity eligibility, retained default ineligibility, explicit caller opt-in, derived-only rows, and SHA-256 lineage.
- Exact, categorical, ordinal, one-sided censored, and interval-censored label validation.
- Frozen split attachment and deterministic JSONL plus SHA-256 sidecars.
- Bounded-memory Parquet split construction for molecule, scaffold, source, protein, target, and double-cold grouping. It accepts either one Parquet file or a canonical partitioned task directory whose deterministically sorted parts, row counts, and SHA-256 digests are verified against the nearest build manifest. It then uses deterministic SHA-256 group assignment, a disk-backed global record-ID uniqueness audit across all parts, and exact global source-row binding.
- Lockstep partitioned-Parquet-to-JSONL streaming bound to the source dataset contract, canonical build manifest, every source-part digest, split, and split-sidecar SHA-256 digests; no full example, offset, or line-digest list is retained.
- A transactional task-integration API now composes the manifest-bound split, streaming serialization, physical routing, train-only vocabularies, train/validation length inventory, and capped loader smoke into one immutable bundle. It refuses overwrite and removes partial sibling transactions on failure.
- The combined all-partition JSONL is a transient router input only. After count reconciliation it and its original sidecar are removed; the final bundle retains a non-dangling hash/count receipt and exactly the train, validation, and sealed-test JSONL payloads.
- After physical routing, the integration workflow does not open, hash, or iterate the test JSONL. Test length status is explicitly `not_inspected_locked_test`; vocabulary fitting and the at-most-32-example loader smoke are train-only.
- Integration and capped-diagnostic acceptances contain exact non-circular component inventories (relative path, SHA-256, and byte size). The sealed test uses its routing-time binding so inventory verification does not reopen it.
- A second transactional API produces deterministic, capped, train/validation-only RDKit descriptors and Morgan fingerprints; fixed dummy plus ridge/logistic diagnostics; validation predictions/errors; and label-permutation and identifier controls. Censored, ordinal, and derived-sensitivity tasks commit explicit skip records rather than being coerced.
- Training-only vocabularies for SMILES, protein sequence, and text, fit with a one-pass token counter and incremental corpus digest.
- A validating iterable JSONL reader and streaming loader smoke test for platform-scale use, alongside an explicit sample-measured memory-risk estimator for prohibited full-list materialization.
- BOS/EOS-preserving truncation, attention masks, modality-presence masks, graph batching, and missing/invalid modality reasons.
- Observed-only contrastive pairs and explicitly unlabeled nonmatching candidates, both restricted to the fixed training partition.
- Fine-tuning configuration validation covering loss, censoring, imbalance policy, optimizer, schedule, evaluation, selection metric, checkpointing, resume, determinism, precision, and PEFT/LoRA.
- Analytic memory/storage estimates and explicitly uncalibrated runtime scenarios.
- Loader smoke tests capped at 32 examples and a tiny Torch wiring smoke capped at two steps and fewer than 100,000 parameters. Smoke outputs are never performance evidence.
- Exact checkpoint resume contracts binding code, data, split, tokenizer, model revision, optimizer/scheduler, and RNG state.

## Accepted real-corpus execution

`research/models/platform/corpus_readiness/full_chembl37/acceptance.json`
(physical SHA-256
`386396b134d94ac1e60ff791f481f85db008f0490c877158bd0745f622615fdb`)
binds 524 components across all 28 canonical task datasets. Nineteen of 23
default tasks and three of five derived-sensitivity tasks passed fixed-seed
structural preflight and were integrated. Four default tasks and two
derived-sensitivity tasks were explicitly skipped for insufficient support;
the orchestrator did not search for a favorable seed.

Eleven eligible default tasks completed capped train/validation diagnostic
baselines. Eight censored tasks and three sensitivity tasks committed explicit
diagnostic skips instead of imputing censoring bounds or presenting derived
labels as primary evidence. The published bundle contains 525 physical files
and 5,369,568,807 bytes. Lead and independent source-rebound verifiers both
reported `source_reverified: true`, `test_lockboxes_opened_or_hashed: false`,
and false large-model/substantive-training flags.

## Accepted multi-strategy split execution

Both independently built official split suites directly regenerate from the
accepted canonical corpus and QC bindings. Each acceptance binds 479
components and enumerates 28 tasks × six strategies: 85 strategies were
materialized and 83 were explicitly skipped for fixed-seed support or
predeclared inapplicability. The exact A/B comparison passes across 480 files,
225 directories, and 564,604,068 bytes with inventory SHA-256
`0307c23c8635b4cad86aa03e0673ffb81a115dc9e4dfb81914fd931c2820296a`.

Exact group/entity overlap checks are exhaustive. Chemical/protein
near-similarity checks remain capped and non-exhaustive, so they cannot support
broad scaffold-, ligand-, or protein-generalization claims. Twelve task views
contain 214 rows for which RDKit Murcko processing fails closed to an explicit
exact-SMILES proxy; those views are never described as true scaffold splits.

## Current blockers

- Stable-hash split fractions are approximate rather than greedily balanced or stratified. These scalable manifests are deliberately not claim-ready until a separate cross-partition near-duplicate audit is complete and accepted.
- Chemical-cluster and temporal splits do not yet have bounded-memory implementations and fail closed rather than silently falling back. Million-scale fingerprint near-neighbor auditing and protein homology/alignment auditing remain separate required work for the corresponding scientific claims.
- Exact large-model checkpoints, immutable weight hashes, model-specific maximum lengths, license snapshots/review, and training-overlap audits are not yet frozen.
- The repository has no approved top-level `LICENSE`, `COPYING`, or `NOTICE`; code, data, model-weight, and redistribution rights require named human approval before release or substantive training.
- The audited 53-pin macOS/CPython 3.13 core provenance lock has zero known vulnerabilities. The broader current developer environment is not the frozen release environment and contains optional `aiohttp==3.14.1` with three reported PYSEC findings; rebuild and audit the authorized clean training environment rather than reusing this expanded environment.
- The local evidence snapshot reports no CUDA or MPS accelerator. This is not a claim about later HPC availability.
- Throughput has not been measured on the intended hardware; runtime ranges must remain scenario estimates until a capped dry run on the exact model/hardware.
- Human authorization is required in a later run before substantive training.

## Static evidence

`research/models/platform/pretraining_static_manifest.json` binds the feature registry, candidate registry, package versions, and the explicit `substantive_training_started: false` state. Passing unit tests or smoke tests demonstrates interface behavior only; it does not establish scientific validity, model performance, clinical utility, or prospective generalization.

The same manifest binds `model_metric_registry.csv`, which distinguishes primary, secondary, and diagnostic metrics, and `baseline_robustness_matrix.csv`, which predeclares split, feature, label-confidence, source, and seed sensitivity analyses without claiming that the data-dependent runs are complete.

The lead-owned final report is
`research/reports/final_verification/platform_final_artifact_verification.json`
(physical SHA-256
`f27ceb4d46edeb9c0dfb2610ef5d2b02075aace1ee1f1051351ec4876bd945d5`,
20,573 bytes). It reports `mechanical_artifact_verification: passed`, direct
source rebinding for both split builds and the corpus, byte-identical A/B
statistical and split trees, 2,117 regular files with zero inode aliases, and
a 1,090-document no-training scan. Its final boundary remains
`scientific_task_claim_ready: false`,
`substantive_large_model_training_ready: false`, and
`substantive_large_model_training_authorized: false`.

## Bounded-memory scale smoke

The frozen interfaces were exercised end to end on a synthetic, homogeneous 100,003-row Parquet task with irregular 13,337-row row groups. The split step completed in 1.49 seconds with a configured and observed maximum batch of 7,777 rows and 3,639,768 measured Pandas deep bytes. Streaming model-ready serialization completed in 9.58 seconds with a configured and observed maximum batch of 4,093 rows, 2,962,133 measured input-plus-manifest Pandas deep bytes, and 10,242,985 recursively measured example-object bytes. Its train/validation/test counts (69,801/15,130/15,072) reconciled exactly with the split manifest.

All three training-only vocabularies were then fit by streaming iteration, and the bounded loader smoke passed its hard cap of 32 examples (four batches of eight). A 1,000-row sample measured 2,972.423 combined bytes per row for the prohibited full-list path, extrapolating to approximately 8.58 GiB at 3.1 million rows. Process peak RSS was 444,022,784 bytes (423.45 MiB), including construction of the synthetic source DataFrame.

These timings and memory observations are local synthetic engineering evidence only. They are not model-performance, scientific-validity, real-corpus throughput, HPC-capacity, or prospective-generalization evidence. The structured record is `research/models/platform/streaming_scale_smoke_evidence.json`, SHA-256 `79fe75b949324d778f148bcc9091bc235faf149f1945331cb3d833a48d3fd733`.
