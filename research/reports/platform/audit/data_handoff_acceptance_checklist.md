# DATA handoff acceptance checklist

Audit snapshot: 2026-08-04. This is the independent DATA-to-MODEL gate for the
platform workstream. It distinguishes a verified source archive, an exported
source-assertion corpus, a canonical evidence corpus, and default model tasks;
passing an earlier layer never implies that a later layer is ready.

## Current independent evidence

| Layer | Evidence checked | Snapshot result |
|---|---|---|
| Official ChEMBL archive | 5,764,252,857 bytes; SHA-256 `33c203740555f96067710cdfc1c3c55d890660e5908ec5cbf5817492c290d281`; 24 contiguous assembly ranges | **Pass: archive verified** |
| SQLite extraction | 30,480,314,368-byte `chembl_37.db`; extracted-database SHA-256 `4be13df3b68e25dcd0bff44bf094033b5aebe98f415acdc8c1cdf380e0c15142` recorded inside the extraction manifest; extraction code publishes only after SQLite magic and `PRAGMA quick_check` pass | **Pass: extracted source database locally evidenced** |
| Release counts | 24,527,044 activities, 2,921,148 molecule records, and 18,552 target records queried from the extracted database | **Pass: core count identity** |
| Activity-source dictionary | ChEMBL 37 uses table `source`, not `source_dictionary`; all 24,527,044 activities have a non-null `src_id`, all join to `source`, and 50 activity sources are represented | **Pass after code correction and actual-schema regression coverage** |
| PK source identity | `src_id=39` is one `DRUG_PK` record described as curated drug PK manually extracted from DailyMed labels; it contributes 1,163 activities | **Pass as source inventory, not automatic PK-task eligibility** |
| Query compatibility | One-row full and specialized queries compiled against the actual SQLite schema; source names survived (`LITERATURE`, `DRUG_PK`), and `pchembl_value` was absent. The accepted export rechecked this contract over all source assertions. | **Pass at export layer** |
| Full activity export | The superseding explicit-schema export contains 24,527,044 rows in exactly 123 Parquet parts. Independent fresh reads verified recursive membership, hashes, bytes, footer row counts, strict global `activity_id` order/uniqueness, source attribution, and one declared/physical Arrow schema. Manifest-file SHA-256: `835ba4e62ed671a707c204b9d0d45e686ec0696de788de8515afaaec6367f796`. | **Pass: source-assertion export-ready** |
| Specialized exports | The repaired generation contains 23 specialized-activity parts, 15 development-metadata parts, and one target-component part. Together with the 123 full parts, the durable transaction receipt accounts for exactly 162 Parquets; no marker, staging file, symlink, `.part`, or checkpoint survived. Specialized-summary file SHA-256: `f9b98aece97af7babf420b2500363a3012d23f713620073a4ac21c2363734245`; normalization-receipt file SHA-256: `f15fdded578000962073676796a879c77c395c4f92a099b20b072a0990a653df`. | **Pass: specialized inventories export-ready** |
| Cross-view overlap | Fresh intersections over the repaired physical identifiers reproduce Kd/Ki∩hERG 2,783; IC50/EC50∩hERG 17,338; PK/ADME∩hERG 30; QT/APD∩hERG 124. The specialized views contain 6,852,381 deliberately overlapping activity memberships plus 22,622 target-component rows. | **Pass as overlap accounting; counts must never be added as disjoint coverage** |
| Canonical/QC/default-task corpus | A/B manifests `1ace39…`/`85bcbe…`, QC `d481c8…`/`acc14b…`, 260 components/1,154,513,167 bytes, 3,938,372 observations and lineage, and 28 task datasets were independently rebound | **Pass: artifact-ready for frozen ChEMBL-only corpus** |
| Statistical census | A/B source-reverified; 19 byte-identical files and 1,374,306 bytes | **Pass: descriptive/statistical artifact-ready** |
| Fixed split suite | A/B directly source-reverified; 480 byte-identical files, 225 dirs, 564,604,068 bytes; 85 materialized/83 reasoned skips | **Pass mechanically; near-similarity remains capped and task claim not ready** |
| Corpus readiness | 524 components; 22 tasks integrated, six support-skipped; 11 capped diagnostics completed and 11 skipped; test lockboxes unopened/unhashed | **Pass: bounded MODEL handoff/interface-ready** |
| Final physical gate | Report `f27ceb…`, 20,573 bytes; artifact/source integrity true, task/training readiness and authorization false | **Pass mechanically; substantive training NOT READY** |

### Final handoff boundary

The DATA-to-MODEL artifact handoff is closed for the frozen ChEMBL-only
snapshot. All 63 model-readiness exclusions are missing-protein-sequence cases:
54 default-task candidates and nine sensitivity-task candidates. The accepted
split feature projections contain no label columns. Corpus structural preflight
also requests no label columns; categorical support is computed only from the
physical training partition. Validation/test labels are not used, and test
lockboxes are neither opened nor hashed after routing.

This does not make any task claim-ready. External normalized candidates still
admit zero canonical observations and labels; source/redistribution rights,
clinical/PK/QT outcome extraction, intended-use and near-leakage thresholds,
checkpoint overlap, compute, clean release, and external validation remain
blocking gates.

## Hard fail conditions

The DATA handoff fails if any condition below is true.

1. The exporter queries a guessed table/column, silently replaces an available
   attribution/context field with null, or fails to preserve `src_id`, source
   short name, and source description. The exact ChEMBL 37 `source` table is
   authoritative. Compatibility fallbacks must be tested and release-scoped.
2. Any exported activity with a non-null source `src_id` lacks a matching source
   record, or `src_id=39` is selected without recording its exact source row and
   its candidate-inventory-only role.
3. `compound_properties`, `pchembl_value`, or another calculated/commercial
   descriptor is admitted as an experimental label or default input feature.
   Such fields may remain untouched inside the immutable raw archive but must be
   absent from activity export and canonical/model views unless separately
   classified with complete lineage and an approved use.
4. This declared ChEMBL-only canonical build opens or depends on BindingDB,
   PubChem, Sun supplement, internal, restricted, or mixed-access bytes. The
   separately frozen external-source bundles are acquisition evidence only;
   their later admission requires a new manifest, explicit rights/identity/
   endpoint gates, and cross-source conflict handling rather than silent
   inclusion in this build.
5. A transient, partial, dot-prefixed, or unmanifested bulk file enters source
   inventory. Archive, release, extraction, and export files must come from
   their completed manifests, and canonicalization must refuse an active write.
   Every part in one logical Parquet dataset—source export, canonical entity,
   observation, lineage, registry, or task—must also have the exact declared
   Arrow schema. Chunk-dependent inference (including all-null versus typed and
   integer versus floating-point columns) is a hard failure even when row counts,
   identifiers, and file hashes reconcile. The writer must bind schema
   fingerprints, and pre-promotion/reuse QC must recompute and reject drift.
6. Any row loses source release, snapshot, record locator, raw-file identifier,
   citation/license scope, access class, observation kind, or transformation
   lineage before the model-ready boundary.
7. An unsupported qualifier or ambiguous range becomes exact. Intervals require
   two finite ordered bounds; one-sided censoring retains the correct bound; no
   midpoint, cap substitution, or endpoint substitution is allowed.
8. `Kd`, `Ki`, `IC50`, `EC50`, kinetic, PK, hERG, QT/APD, clinical, and derived
   free-energy observations are pooled as interchangeable targets. Standard
   binding free energy may derive only from included, exact, positive `Kd` in a
   compatible single-protein context, with temperature/standard state/formula
   and a lineage digest. Reference-temperature rows remain approximations.
9. Missing molecule, target, assay, endpoint, structure, result, unit, or
   required context is repaired with a convenient inference. Row-stable fallback
   identities may retain a quarantined assertion, but must not make it eligible.
10. Generic assay wording or absence of clinical evidence is used to invent an
    explicit preclinical/development stage. Stage may be unknown.
11. A default task contains a prediction, a lineage-free derivation, a
    review/quarantined row, a non-redistributable row, explicit QT/APD evidence,
    context-incomplete PK, a missing endpoint/result, or a label marked
    `default_task_eligible=false`.
12. A `task_id` is observation-specific or spans heterogeneous task semantics.
    It must be shared by rows with one declared task signature and must bind at
    least evidence domain, endpoint, assay family/context policy, label kind,
    unit, and task-policy version. hERG binding, functional, and unresolved
    assay families remain separate.
13. A default task mixes exact and censored label contracts without an explicit
    supported likelihood/policy, or a derived sensitivity task is presented as
    a default observed-label task.
14. DATA schema validation passes an artifact that the MODEL adapter cannot
    consume. The joined task view must carry nonblank snapshot/source provenance,
    `access_class=public_redistributable`, explicit eligibility, stable molecular
    input, observation kind, and any required derived-lineage digest.
15. Physical-input, unique-source-record, canonical-observation, inclusion,
    task, and exclusion counts do not reconcile exactly, or duplicate/conflict
    handling discards a contradictory row without an audit trace.
16. The checked-in bounded REST panel, targeted legacy snapshot, or a specialized
    candidate inventory is described as a representative/global training corpus.
    Full-corpus readiness requires completed bulk export, canonical QC, coverage,
    missingness, leakage, rights, and reproducibility evidence.
17. A task candidate missing a modality declared in `required_modalities`
    reaches a default/sensitivity task, or disappears from accounting. Source
    observations and lineage must remain; each candidate must be exactly one of
    eligible or reasoned `task_exclusions/<scope>`, with no observation-ID leak
    from an exclusion shard into the corresponding model-task scope.

## Required acceptance evidence

### A. Source and export

- Verify the archive size/hash, extracted database size/hash, SQLite integrity,
  release metadata, license, required attribution, and the three core table
  counts from fresh reads.
- Compile and execute bounded full/specialized queries against the real ChEMBL
  37 schema. Assert the `source` join and available context columns, not merely
  the shape of a synthetic fixture.
- Reconcile export row count to `activities` exactly. Verify monotonic unique
  `activity_id`, immutable part hashes, relative paths, query hash, database
  hash, and absence of unmanifested parts.
- Compare the complete Arrow schema of every physical part in each logical
  dataset against one explicit release-scoped schema and its manifest
  fingerprint. Tests must force an all-null chunk followed by a typed chunk and
  integer-valued followed by floating-point-valued chunks; both must serialize
  with the one declared field type and load as one dataset without coercion.
- Treat repair of the existing 162 physical Parquet artifacts as one explicit
  recovery transaction: 123 full-activity, 23 specialized-activity, 15
  development, and one target-component file. Updated manifests must be staged
  before commit; an injected mid-swap interruption must leave every reader
  fail-closed, and an idempotent resume or whole-directory rollback/swap must
  restore one fully bound generation. The final receipt's membership, hashes,
  and schemas must match all 162 files exactly.
- Treat Kd/Ki, IC50/EC50, hERG, PK/ADME, and QT/APD specialized outputs as
  deliberately overlapping inventories; report overlap counts and never add
  their counts as if disjoint.
- Inventory any available ChEMBL field intentionally omitted from export with a
  scientific and legal rationale. In particular retain available unit ontology,
  assay modality/system/tissue/strain/variant, source, and document-release
  context needed for later eligibility decisions.

### B. Canonical evidence and QC

- Verify each canonical table and partitioned task against an explicit field
  schema and manifest fingerprint. Independently recompute physical schemas for
  every shard; a writer-side cast without a QC-side cross-part check is not
  acceptance evidence.
- Exercise exact, approximate, left/right-censored, valid interval, reversed
  interval, unsupported qualifier, missing unit, incompatible unit, nonpositive
  concentration, dimensionless, and categorical-result fixtures.
- Demonstrate stable molecule/protein/assay/observation/source/task identifiers,
  including missing-source-ID quarantine paths, multiple aliases, parent/full
  structure collisions, stereochemistry ambiguity, constructs/complexes, and
  target component order.
- Produce machine-readable rule results and counts for identity failures, unit
  conversion, impossible ranges, missing results/context, duplicates, mirrors,
  conflicts, source-record disagreements, stage/status, quality, access, and
  inclusion.
- Show field/source/domain/task missingness, attrition from source assertion to
  default task, target/ligand/assay/document/source concentration, endpoint/unit/
  relation distributions, and high-impact human-review queues.
- Prove every derived free-energy row round-trips to its source `Kd` within the
  declared tolerance and carries its source observation and lineage digest.

### C. Default task and MODEL contract

- Publish a task registry with one row per shared task ID and its complete
  scientific signature, counts, relation policy, eligibility policy, intended
  use, and prohibited claim.
- Assert, for every default task artifact: one task ID/type/domain/endpoint/
  assay-family/label-kind/unit signature; only included and explicitly eligible
  rows; only allowed observed/summary/curated kinds; public-redistributable
  access; nonblank source/snapshot/record provenance; no label/input leakage.
- Keep derived free-energy sensitivity artifacts physically and semantically
  separate. MODEL must require explicit derived-label opt-in plus lineage.
- Bind each task's `required_modalities`; publish missing-input candidates in
  explicit exclusion shards with reason flags and exact candidate = eligible +
  excluded reconciliation by scope/task/source/protein/target. Independently
  prove excluded observation IDs do not occur in the corresponding task scope.
- Run the DATA frame through the MODEL adapter, split guard, fixed-manifest join,
  serializer, loader, and collator. Rejections are acceptance evidence for
  private, prediction, default-ineligible, heterogeneous-task, missing-lineage,
  unsupported-relation, and excluded-split fixtures.

### D. Reproducibility

- Build twice from the same frozen inputs in clean temporary roots and compare
  all content hashes after excluding declared timestamps. No absolute personal
  path may appear in artifacts or manifests.
- Run focused DATA and cross-contract tests, Ruff, mypy, the full regression
  suite, staged disclosure/large-file/secret/path scans, and final offline
  manifest verification. Record exact commands and exit statuses.
- Keep substantive large-model training disabled. Interface and tiny-loader
  passes establish artifact plumbing only, not task validity or scientific
  performance.

## Acceptance labels

- **Archive-ready:** A verified official source archive and extraction exist.
- **Export-ready:** Complete source assertions and specialized inventories have
  verified manifests and source attribution.
- **Artifact-ready:** Canonical/QC/task artifacts build deterministically and
  pass DATA-to-MODEL contract tests.
- **Task-ready:** A specified endpoint task additionally has adequate support,
  context, QC, leakage-safe splits, diagnostics, and an intended-use claim.
- **Substantive-training-ready:** All rights, large-corpus coverage, external
  validation, model/checkpoint, overlap, compute, governance, and reproducible
  release gates pass. Earlier labels never imply this label.
