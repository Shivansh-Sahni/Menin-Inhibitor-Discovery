# DATA-to-MODEL integration runbook

Status: implementation-ready on 2026-08-04; not executed against provisional DATA output.

This runbook uses the transactional APIs in
`menin_discovery.platform_model_integration`. It prepares fixed splits, physically
separated model inputs, train-only vocabularies, bounded loader evidence, and optional
lightweight diagnostics. It never starts substantive pretrained-model training.

## Preconditions

Use one canonical task directory under:

```text
research/data/platform/canonical/full_chembl37/tasks/default/<task_slug>/
```

The API fails closed unless all of the following hold:

- the task is a directory of canonical `part-*.parquet` files;
- the nearest ancestor `build_manifest.json` lists exactly those parts;
- every listed path, row count, and SHA-256 verifies;
- schemas are uniform, record IDs are globally unique, and task semantics are homogeneous;
- no `.full_chembl37.building` sibling or other provisional `.building` path exists;
- the destination does not already exist.

Do not point this workflow at a live or partial DATA build. Do not weaken
`require_manifest_bound_directory` for a publication or platform run.

## 1. Materialize one task-readiness bundle

Recommended final path:

```text
research/models/platform/integration/full_chembl37/<task_slug>/molecule_hash_stream_v1/
```

Run from the repository root after replacing `<task_slug>`:

```bash
PLATFORM_TASK_SLUG="<task_slug>" PYTHONPATH=pipeline/src .venv/bin/python - <<'PY'
from __future__ import annotations

import json
import os
from pathlib import Path

from menin_discovery.platform_model_integration import (
    TaskIntegrationConfig,
    materialize_task_integration_bundle,
)

task_slug = os.environ["PLATFORM_TASK_SLUG"].strip()
if not task_slug or Path(task_slug).name != task_slug:
    raise ValueError("PLATFORM_TASK_SLUG must be one directory name")

task_directory = (
    Path("research/data/platform/canonical/full_chembl37/tasks/default") / task_slug
)
output_directory = (
    Path("research/models/platform/integration/full_chembl37")
    / task_slug
    / "molecule_hash_stream_v1"
)
acceptance = materialize_task_integration_bundle(
    task_directory,
    output_directory,
    TaskIntegrationConfig(
        split_name="molecule_hash_stream_v1",
        split_strategy="molecule_grouped",
        intended_use=(
            "new molecule within the observed public task domain at platform scale"
        ),
        seed=20260804,
        split_batch_size=50_000,
        serialization_batch_size=8_192,
        loader_batch_size=8,
        loader_maximum_batches=4,
        task_eligibility_mode="default",
        require_manifest_bound_directory=True,
    ),
)
print(json.dumps(acceptance, indent=2, sort_keys=True))
PY
```

Task family is resolved from the canonical label contract, not guessed from the task
name:

| Canonical `label_kind` | Split/model family | Diagnostic disposition |
|---|---|---|
| `continuous_exact` | regression | fixed dummy-median and ridge allowed |
| `continuous_censored` | regression | committed skip; no midpoint imputation |
| `categorical` | classification | fixed binary diagnostics only after explicit encoding |
| `ordinal` | classification | committed skip; no ordinal-to-binary coercion |

The scalable molecule-group hash split guarantees exact molecule-ID exclusion. Its
fractions are approximate and unstratified. It remains `not_claim_ready` until the
separate chemical near-neighbor and protein-homology/alignment audits pass.

### Final bundle layout

```text
acceptance.json
split/
  molecule_hash_stream_v1.parquet
  molecule_hash_stream_v1.parquet.manifest.json
model_ready/
  combined_serialization_receipt.json
  partitions/
    partition_manifest.json
    train.jsonl
    validation.jsonl
    test.jsonl
readiness/
  task_semantics.json
  length_inventory.json
  vocabulary_smiles.json
  vocabulary_protein.json
  vocabulary_text.json
  streaming_loader_smoke.json
```

There is deliberately no combined all-partition JSONL in the committed bundle. The
combined stream exists only inside the private output transaction long enough to route
and reconcile train, validation, and test. It and its original sidecar are then removed.
`combined_serialization_receipt.json` preserves their SHA-256, count, ordered-line
digest, source/split bindings, and measured memory metadata without retaining a dangling
path or any record payload.

### Sealed-test rule

The test payload is written and hashed while the physical router creates it. After the
router returns, the integration workflow does not open, hash, or iterate `test.jsonl`:

- length inventory computes train and validation statistics only; test is reported as
  `not_inspected_locked_test` with its routing count;
- all vocabularies are fit from `train.jsonl` only;
- the loader smoke reads at most 32 training examples;
- collator smoke limits use training maxima only;
- the complete component inventory reuses the test digest and byte count bound during
  routing rather than reopening the lockbox.

The final `acceptance.json` inventories every other committed component by relative
path, SHA-256, and byte size. It inventories `test.jsonl` with the explicit verification
source `bound_during_physical_routing_no_reopen`. Acceptance excludes itself to avoid a
circular digest; its external SHA-256 is returned by the API and must be retained by the
caller.

## 2. Explicit derived-label sensitivity bundle

Derived labels never enter the default path. A derived-only canonical sensitivity task
must independently have `sensitivity_task_eligible=true`,
`default_task_eligible=false`, and a valid SHA-256 lineage digest on every row. Use:

```python
TaskIntegrationConfig(task_eligibility_mode="derived_sensitivity")
```

The API rejects mixed observed/derived tasks and rejects derived rows in `default` mode.
The diagnostic API commits a machine-readable
`skipped_derived_sensitivity_not_primary_diagnostic` record for this bundle; it does not
quietly present a derived-label sensitivity analysis as primary experimental evidence.

## 3. Optional capped diagnostic bundle

Recommended final path:

```text
research/models/platform/diagnostics/full_chembl37/<task_slug>/molecule_hash_stream_v1/
```

The exact external SHA-256 returned by phase 1 is mandatory input:

```bash
PLATFORM_TASK_SLUG="<task_slug>" \
PLATFORM_INTEGRATION_ACCEPTANCE_SHA256="<64-hex-digest>" \
PYTHONPATH=pipeline/src .venv/bin/python - <<'PY'
from __future__ import annotations

import json
import os
from pathlib import Path

from menin_discovery.platform_model_integration import (
    DiagnosticConfig,
    materialize_capped_diagnostic_bundle,
)

task_slug = os.environ["PLATFORM_TASK_SLUG"].strip()
integration = (
    Path("research/models/platform/integration/full_chembl37")
    / task_slug
    / "molecule_hash_stream_v1"
)
output = (
    Path("research/models/platform/diagnostics/full_chembl37")
    / task_slug
    / "molecule_hash_stream_v1"
)
acceptance = materialize_capped_diagnostic_bundle(
    integration,
    output,
    integration_acceptance_sha256=os.environ[
        "PLATFORM_INTEGRATION_ACCEPTANCE_SHA256"
    ],
    config=DiagnosticConfig(
        seed=20260804,
        maximum_train_examples=50_000,
        maximum_validation_examples=10_000,
        fingerprint_bits=2_048,
        fingerprint_radius=2,
        # Required for nonnumeric binary labels, for example:
        # binary_label_mapping=(("inactive", 0), ("active", 1)),
    ),
)
print(json.dumps(acceptance, indent=2, sort_keys=True))
PY
```

Diagnostic caps cannot be raised above 50,000 train and 10,000 validation examples.
Selection retains the lowest full 256-bit `SHA256("<seed>|<record_id>")` ranks with a
fixed-size heap. No test path is resolved, hashed, opened, or iterated.

Completed exact-regression or explicitly binary-classification bundles contain:

```text
acceptance.json
features/
  selected_rows.parquet
  descriptors.parquet
  feature_failure_summary.csv
  fingerprints_rdkit_morgan_r<radius>_<bits>.npz
  feature_config.json
  feature_manifest.json
  numeric_target_leakage_scan.csv
baselines/
  metrics.csv
  validation_predictions.parquet
  validation_error_analysis.parquet
  label_permutation_control.json
  identifier_hash_control.json
  dummy_median.joblib OR dummy_prior.joblib
  ridge_fixed.joblib OR logistic_fixed.joblib
  class_imbalance_and_prevalence.json  # binary classification only
  baseline_metadata.json
```

Fingerprints require the explicit RDKit Morgan backend; there is no silent hashed-text
fallback. Baselines are fixed dummy plus ridge/logistic models with no tree, sweep,
threshold selection, or model selection. Regression records `class_imbalance="none"`;
binary classification records balanced training weights and natural train/validation
prevalence separately. Evaluation is validation-only. Label permutation and identifier
hash controls are diagnostics, not p-values or scientific results.

The diagnostic acceptance inventories every generated feature, failure report, leakage
scan, metric, prediction, error analysis, negative control, class report when applicable,
model object, configuration, and manifest by relative path, SHA-256, and size. It binds
the exact phase-1 acceptance digest. Censored, ordinal, and explicitly derived tasks
commit a small `diagnostic_status.json` skip bundle instead of fabricating a target.

## 4. Atomicity and non-overwrite behavior

Both APIs build in a uniquely named sibling transaction directory, verify all JSON
no-training declarations and exact component membership, and commit with one directory
rename. Existing destinations are immutable and cause `FileExistsError`. Any exception
removes only the newly created transaction directory; it cannot replace a prior bundle.

Every generated JSON states `substantive_training_started: false`. Acceptance and
diagnostic metadata also state `large_model_training_started: false` where applicable.

## 5. Acceptance checklist

- Final DATA manifest exists, no sibling provisional build exists, and every task part
  verifies by path, count, hash, schema, ID uniqueness, row order, and task signature.
- Split counts reconcile to source counts, including explicit exclusions; the sidecar
  binds source parts and the exact split artifact.
- Claim readiness remains blocked pending cross-partition near-neighbor and protein
  homology/alignment review.
- The final bundle contains exactly three JSONL payloads: train, validation, and test.
  No combined all-partition payload or dangling combined path survives.
- The post-routing test access counters are zero; test length status is locked.
- Vocabularies and loader smoke are train-only; loader smoke processes no more than 32
  examples and performs no optimizer step.
- Component inventory membership exactly equals every committed non-acceptance file;
  each relative path, SHA-256, and size verifies under its stated verification source.
- Diagnostics, if supported, use only deterministic capped train/validation samples,
  fixed features/models, validation evaluation, and both negative controls.
- No locked-test metric, hyperparameter search, substantive embedding run, checkpoint
  download, pretrained-model fit, or HPC throughput claim is produced.

## 6. Remaining scientific blockers

This workflow establishes preprocessing and engineering readiness, not scientific claim
readiness. Still required are the million-scale chemical-neighbor audit, protein
homology/alignment audit, model/checkpoint revision and license approval, pretrained-data
overlap analysis, final model-specific length policy, hardware/resource preflight,
external validation design, and a separately authorized locked-test evaluation.
