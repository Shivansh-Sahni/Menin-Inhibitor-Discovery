# HPC execution, storage, and recovery plan

## Status and estimate policy

This is a future plan; no HPC jobs were submitted. Every time, memory, GPU, and storage figure below is an **engineering estimate**, not a measured result. The 10–100-molecule smoke test must pass before any full-scale job.

## Stage gates

| Stage | Work | Suggested resources—estimate | Runtime—estimate | Output—estimate | Gate |
|---|---|---:|---:|---:|---|
| S0 | 10–100 molecule smoke test | 4 CPU, 16 GB RAM; optional 1 GPU only for model load | 10–60 minutes/family | under 5 GB | all schema/determinism/leakage/failure tests pass |
| S1 | 2D fingerprints/descriptors for ~370k parents | 32 CPU, 128 GB RAM, array shards | 2–12 hours | 20–100 GB | row/ID/checksum conservation and rerun equality |
| S2 | conformer/protomer enumeration | 64 CPU, 256 GB RAM, array shards | 1–5 days | 0.5–3 TB | bounded ensemble, explicit failures, aggregation audit |
| S3 | classical baselines | 32–64 CPU, 128–256 GB RAM | 2–24 hours/model/split | 10–100 GB | frozen predictions/metrics; no test-aware selection |
| S4 | GNN/SMILES/deep comparators | 1–4 GPUs, 16–80 GB VRAM/GPU, 16–64 CPU | 6–72 hours/run | 50–500 GB | deterministic seed subset and checkpoint-resume test |
| S5 | docking/PLEC on selected functional cohort | 64–256 CPU or licensed GPU nodes | days to weeks | 1–10 TB | receptor/pose versions pinned; coverage/cost justified |
| S6 | multitask/pretraining/censored ablations | 1–8 GPUs depending model | 1–7 days/run | 0.1–2 TB | P0 baselines already frozen; ablation benefit preregistered |

Do not reserve S2/S5/S6 resources until sampled throughput converts estimates into measured forecasts.

## Job topology

- One immutable run manifest produces deterministic shard manifests.
- Array jobs operate on explicit parent-ID lists; no glob-discovered membership.
- Each shard writes to a unique staging directory, validates, then atomically publishes a manifest and immutable data object.
- Merge jobs verify schema version, row counts, unique IDs, checksums, status distribution, and split manifest before catalog registration.
- Training jobs mount feature objects read-only and write run-specific checkpoints/predictions.

## Storage

Use four logical zones: `raw_immutable`, `standardized_versioned`, `feature_objects_content_addressed`, and `runs`. Raw inputs and split manifests are immutable. Feature objects are content-addressed by input/config/software digest. Run artifacts contain configs, logs, checkpoints, raw predictions, metrics, environment lock, and provenance—not duplicate source datasets.

Retention: keep final/decision checkpoints, best validation checkpoint, last resumable checkpoint, raw test predictions, manifests, and failures. Ephemeral per-epoch checkpoints expire only after final checksum/metric reproduction. Never delete the sole copy of raw/source data through a job cleanup.

## Checkpoints and retries

- Checkpoint after a bounded number of samples or at most every 30 minutes—whichever comes first—and on scheduler preemption signal.
- Checkpoints include model/optimizer/scheduler/scaler states, epoch/step, RNG states for Python/NumPy/framework/CUDA, sampler position, manifest/config hashes, and code/container digests.
- Retry `preempted`, transient filesystem/network, and single-shard timeout at most twice with exponential backoff.
- Retry OOM once only with a preregistered smaller batch while preserving effective batch size; record configuration change.
- Do not retry invalid chemistry, deterministic software exceptions, checksum mismatch, schema mismatch, NaN divergence, or leakage-gate failure automatically.
- Failed shards remain in the manifest; a later repair produces a new attempt ID, never an overwritten result.

## Determinism

Pin container digest, driver/CUDA compatibility, code commit, lockfile, dataset/split/config checksums, and all seeds. Enable deterministic algorithms where supported; record nondeterministic operations and hardware. Run the smoke fixture twice and require exact IDs/statuses/checksums for deterministic integer features and tolerance-bounded equality for floating-point/3D outputs. Publish seed-level results rather than only a favorable seed.

## Scheduling order

1. Validate contracts and freeze the split manifest.
2. Run S0 twice.
3. Measure throughput and revise estimates.
4. Run S1 and cheap S3 baselines.
5. Approve S4 only if baselines and data gates pass.
6. Run S2/S5 on a bounded high-value cohort before considering scale-up.
7. Run S6 only after comparator and license readiness gates pass.
