# Compute and operations plan before substantive training

Snapshot date: 2026-08-05. No HPC allocation or substantive training is
authorized. This plan records what can be measured locally and what must be
approved later.

## Measured local baseline

| Item | Observed value |
|---|---:|
| Host | macOS 15.7.5, arm64 |
| Python | 3.13.7 |
| Logical CPUs | 8 |
| Local filesystem capacity | 494,384,795,648 bytes |
| Platform data | 3,076 files / 67,347,162,115 bytes |
| Platform model/interface artifacts | 531 files / 5,369,595,870 bytes |
| Platform reports at governance scan | 89 files / 18,446,885 bytes |

The approximately 72.7 GB current platform footprint excludes `.tmp` and does
not include a selected large-model checkpoint, optimizer states, repeated
training checkpoints, or future structure corpora. A safe rebuild needs space
for preserved inputs, a private transactional build, the promoted output, and
failure/quarantine evidence. Therefore current observed bytes are a lower
bound, not a training storage estimate.

## Work permitted on this host

- Schema, provenance, manifest, rights, linkage, task-support, and leakage
  analyses that are explicitly memory/disk bounded.
- Deterministic classical baselines capped by the existing diagnostic policy.
- Tokenizer/loader/collator tests and at most two smoke steps with no more than
  100,000 parameters.
- Checkpoint metadata and license review without downloading large weights.
- Test-free structural inspection of routed partitions; no post-routing test
  label access.

## Work prohibited until approval

- Large pretrained checkpoint downloads.
- Substantive pretraining, fine-tuning, or large hyperparameter searches.
- Any job whose cost, monitoring, data access, retention, or failure policy is
  not frozen.
- Any run that opens the sealed test set for selection, calibration, debugging,
  threshold choice, or early stopping.

## HPC approval sheet

The compute owner must fill this with measurements from a capped dry run on the
exact selected checkpoint and hardware.

| Field | Required value |
|---|---|
| Hardware | accelerator type/count, CPU/RAM, interconnect, storage tier |
| Exact model | repository/package, revision, weight/tokenizer/config SHA-256 |
| Data | canonical/split/corpus manifest hashes and access class |
| Precision/distribution | fp32/bf16/fp16 policy, DDP/FSDP/other exact settings |
| Batch plan | per-device batch, accumulation, global batch, sequence/graph limits |
| Throughput | measured examples/tokens per second and peak memory |
| Storage | checkpoint size, count/retention, optimizer/RNG state, logs, temporary space |
| Budget | wall time, accelerator-hours, monetary estimate, and budget owner |
| Energy | available facility/accounting estimate and reporting method |
| Monitoring | loss/gradient/nonfinite/data-order/resource alerts and responsible-use checks |
| Recovery | injected interruption plus exact resume equivalence test |
| Security | identities, secrets, access control, encryption, logs, incident owner |
| Authorization | named approver, scope, date, expiration, stop conditions |

## Required capped dry run

After the scientific task, checkpoint, licenses, and compute allocation are
approved, the first accelerator job must remain a dry run. It must verify input
and label masks, deterministic data order, finite loss/gradients, optimizer and
scheduler state, checkpoint atomicity, interruption/restart equivalence,
measured peak memory/throughput/storage, and sealed-test inaccessibility. Its
metrics are engineering evidence, not model performance.

Only after that evidence is bound into a new versioned readiness report may a
human authorize the prespecified substantive job.
