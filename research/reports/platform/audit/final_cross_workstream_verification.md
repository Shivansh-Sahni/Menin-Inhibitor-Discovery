# Independent cross-workstream verification record

Audit date: 2026-08-04. This record is the scientific-audit handoff to the
integration lead. It applies the governing 25-section execution specification
to the frozen DATA, external-acquisition, MODEL, software, documentation, and
artifact surfaces. A workstream's own success statement is not sufficient:
acceptance requires an independent physical or executable check and lead review.

## Decision architecture

Readiness is tiered. Passing one tier does not imply the next.

| Tier | Meaning | Required minimum evidence |
|---|---|---|
| Archive-ready | An authentic, immutable upstream snapshot is locally available | Owner route, release/retrieval identity, license/citation record, byte size/hash, integrity/parse evidence |
| Export-ready | Complete source assertions are exported without silent schema or provenance loss | Exact count conservation, uniform physical schema, manifest membership/hashes, source identity, stable record IDs |
| Artifact-ready | Canonical evidence, QC, and task artifacts are physically complete and reproducible | Explicit schemas, full component inventory, bound passing QC, semantic/count reconciliation, deterministic second build |
| Interface-ready | Fixed-split, model-ready, loader/collator, metric, baseline, and fine-tuning interfaces are executable | Frozen source binding, lockbox isolation, train-only fitted state, focused tests, bounded smoke evidence |
| Task-ready | One declared scientific prediction task can support its intended claim | Adequate support/context, rights, missingness/bias analysis, exact/near leakage audit, difficult split, accepted diagnostics |
| Substantive-training-ready | Large-model training can start responsibly | Every preceding gate plus checkpoint/license/overlap, compute, monitoring, failure-recovery, governance, and human authorization |

The governing decision rule is conjunctive: one blocking rights, leakage,
external-validity, or governance failure is enough for a **NOT READY** verdict,
irrespective of a numeric rubric score.

## Adversarial findings that changed acceptance

Independent review must record defects found as well as final passes. The
following issues materially changed the implementation or acceptance boundary:

1. ChEMBL 37 exposes activity attribution through table `source`, not the
   initially assumed `source_dictionary`; fail-closed source preservation was
   required.
2. `pchembl_value` was not available in the actual ChEMBL 37 query surface and
   calculated/commercial descriptor fields were excluded from experimental
   labels and default features.
3. The first full 123-part activity export had 53 Arrow schema groups because
   pandas inferred all-null, integer-valued, and floating-point-valued chunks
   differently. Multipart specialized/development exports showed the same
   class of defect. All affected manifest digests were withdrawn from
   acceptance pending explicit-schema regeneration.
4. Canonical writers originally had the same inference risk and QC did not
   compare physical schemas across shards. Acceptance therefore requires
   explicit canonical schemas, manifest fingerprints, and QC-side recomputation
   for every entity, observation, lineage, registry, and task dataset.
5. A model-ready combined train/validation/test JSONL originally survived
   routing, undermining the physical lockbox. The frozen design removes the
   combined bytes and sidecar after verified routing and retains only a
   non-dangling receipt plus physically separated partitions.
6. Dependency declarations initially had inconsistent upper bounds and a
   missing direct `urllib3` declaration. The reconciled environment must be
   checked semantically across project, requirements, environment, and lock
   files rather than compared as raw strings.
7. External cohort/mapping inventories were initially bound in top-level
   metadata but omitted from the physical verifier's `files` walk, and nested
   ClinicalTrials inventory paths lost their source-root-relative location.
   Exact committed-artifact membership and tamper checks are required.
8. Resuming a partial external download initially did not authenticate the
   partial sidecar, URL, and persisted offset before a 206 append. Resume must
   fail closed or restart when provenance is missing or a source/release
   contract changes.
9. A syntactically valid `206` response was still insufficient when the partial
   object lacked an authenticated strong validator. The hardened contract
   permits append only when a quoted strong ETag is persisted and returned
   exactly, or when the completed object can be checked against a published
   digest; weak, unquoted, missing, or changed validators are adversarially
   rejected.
10. The first ClinicalTrials.gov paging design could have resumed old pages
    after the live database advanced. Cohort directories are now keyed by exact
    API version plus `dataTimestamp`, with version probes before acquisition,
    before every resume request, and after the last page. A timestamp change
    ends the run without appending to the older cohort.
11. ClinicalTrials.gov metadata setup for the cardiac-results cohort initially
    could overwrite an already accepted broad all-DRUG top manifest before the
    composite finished. Metadata writes are now versioned without moving that
    pointer; injected cardiac failure and timestamp rollover preserve the broad
    verifier-passing manifest, while the composite entry point rejects a
    cardiac-only result.
12. An early UniProt top manifest retained stale self-identity and later counted
    returned `Inactive` records too generously for sequence use. The superseding
    manifest recursively binds every artifact and explicitly separates 13,976
    sequence-ready entries from 1,007 inactive records, three ambiguous
    requested accessions, and 46 syntax-invalid source identifiers.
13. BindingDB could not be treated as an independent second source merely
    because its bytes came from a different server. The exact 202608 origin
    audit found 93,023 BindingDB-curated candidate rows, 429 imported ChEMBL
    mirror rows, and 260 Taylor Research rows; the latter two groups remain
    excluded/quarantined rather than inflating multisource coverage.
14. The schema-normalization repair itself initially lacked a sufficiently
    durable, replay-safe transaction. The accepted design journals the exact
    old/new generation, owner, and operation identity, syncs state before
    swaps, rejects readers during interruption, and makes recovery idempotent;
    exact recursive membership prevents an unlisted shard from surviving.
15. Single-file canonical registries were initially outside the same normative
    Arrow-schema contract as partitioned datasets. Source, source-file, and task
    registries now require declared physical schemas and QC-side recomputation;
    singleton status is not an exemption from type/nullability validation.
16. Candidate task rows originally lacked an explicit, modality-aware admission
    boundary. The accepted contract preserves every source observation and
    lineage edge, declares each task's required modalities, and writes missing-
    structure/sequence candidates to reasoned `task_exclusions/<scope>` shards.
    Counts must reconcile by scope, task, source, protein, and canonical target,
    while QC proves that no excluded observation reaches a model task.
17. The first corpus-wide preflight counted and persisted labels from the
    already determined test partition and kept an unbounded counter of exact
    continuous values. A first repair removed that leakage, but its verifier
    still accepted a top-level claim that structural preflight had read training
    and validation labels. The accepted design requests no label columns during
    structural preflight, derives categorical support only from the physically
    routed training partition under a fail-closed cardinality cap, materializes
    no continuous/censored/ordinal label histogram, binds every top task record
    exactly to its inventoried `task_status.json`, and rejects the reproduced
    acceptance-tampering exploit. Test-label bytes remain transitively bound but
    unopened after routing.
18. The first two-build comparator compared the two report manifests to each
    other but did not bind either one back to its corresponding canonical build;
    it also accepted a symlinked report artifact. The accepted comparator
    requires each report-manifest file to have the exact physical SHA-256 of its
    build manifest, rejects symlinks throughout both input path chains and
    trees, requires distinct build and report roots, checks exact B-report-root
    membership, and rejects undeclared component, manifest, QC, and report
    mutations. Only `built_at_utc` and the QC generation-time/build-digest pair
    may differ.
19. The second real canonical attempt reached its exact entity registry and
    failed closed on 777 molecule payload conflicts. Independent inspection
    proved a bijection between 1,304,011 source-compound IDs and molecule IDs,
    zero alias-map conflicts, and one invariant compound/SMILES/InChIKey tuple
    for every affected identity. The apparent disagreement was JSON `1` versus
    `1.0` for `fragment_count`, caused by pandas promoting an integer column in
    shards containing a null. Registry serialization now treats only finite
    integral floats as their integer-equivalent value; a nonintegral `1.5`
    regression remains a true fail-closed conflict. The unpromoted attempt is
    retained in quarantine rather than erased.
20. A repaired split-suite verifier rejected four earlier top-record and
    symlink exploits but still accepted a coordinated source-projection
    substitution. The audit changed one published SMILES, synchronized every
    self-authored feature/sidecar/status/inventory hash, left the split and
    leakage diagnostics stale, and received `verified` with
    `source_reverified=true`. Canonical task hashes and row counts do not prove
    that the projected feature values and order came from those tasks. Final
    acceptance therefore requires an exact, label-blind canonical-to-published
    feature-content rebind and a regression reproducing this exploit.
21. The same split-suite verifier accepted an applicable, already materialized
    mandatory molecule-group split relabeled as arbitrary `skipped_fabricated`
    with a fabricated reason, synchronized internal accounting, and retained
    split artifacts. Final acceptance therefore also requires recomputation of
    feature support and strategy applicability, exact terminal status/reason
    rules, and topology checks that reject materialized-only artifacts under a
    skipped status. No real split suite may run before both split-verifier
    findings are closed and independently replayed.
22. A third split-suite candidate closed feature substitution and fabricated
    skip handling but regenerated only skipped strategies during source rebind.
    The audit swapped one training and one test assignment in a materialized
    molecule-group split, preserving partition counts, synchronized the split,
    sidecar, status, task, and inventory hashes, and left leakage diagnostics
    stale. Verification again returned `source_reverified=true`. Final source
    rebind must independently rerun every materialized fixed-seed strategy from
    the regenerated feature routing and compare exact assignments and portable
    sidecar semantics, then regenerate and compare the corresponding exact and
    sampled leakage diagnostics.
23. A fifth split-suite candidate closed materialized assignment drift and
    canonical source-identity substitution, but its component inventory still
    treated each mapping key as authoritative while ignoring the entry's own
    `relative_path` and any unknown fields. The audit changed an inventoried
    path to `fabricated/path`, added `claim_ready=true`, recomputed only the
    inventory digest, and again received `verified` with
    `source_reverified=true`. Final acceptance therefore requires an exact
    inventory-entry schema, equality between each mapping key and its declared
    relative path, type/range checks for size and row counts, and rejection of
    claim-bearing extra fields.
24. The first final cross-workstream gate allowed its caller-selected output
    report to alias a verified input. Setting the output path to
    `pipeline/config/platform.yaml` produced a passing report and then replaced
    the verified YAML with that JSON, leaving the report's critical input hash
    stale. The final writer must be confined to its report namespace, reject
    every verified file and immutable artifact-tree alias, and preserve a
    differing pre-existing report rather than overwrite it.
25. The same final gate's no-training review did not recognize a generic
    `training_authorized` field in the hash-bound model-candidate registry and
    did not recursively review YAML. A coordinated edit setting
    `training_authorized=true`, or setting
    `tasks.clinical.enabled_for_training=true` while retaining the two checked
    top-level false flags, still produced a mechanical pass. Static readiness
    also accepted any nonempty artifact set although its generator has an exact
    six-artifact contract. Final acceptance requires duplicate-safe YAML,
    recursive training-policy checks, the exact static manifest/artifact
    topology, and semantic or deterministic-regeneration verification of those
    registries.
26. The first final gate treated absence of an exception as child-workstream
    success and never checked returned terminal fields. An injected external
    result with `status=failed` was embedded in a report whose own mechanical
    result was `passed`. The integration gate must require each verifier's exact
    passing status, source-reverified flag, no-training boundary, and relevant
    lockbox/label boundary before constructing its top-level pass.
27. A later final-gate candidate required an external verifier pass but still
    accepted the verifier's explicit nonstandard-fixture branch. An independently
    constructed one-source bundle with one nonstandard Parquet artifact passed
    raw rebinding while returning
    `semantic_verification=not_applicable_nonstandard_fixture`. Final production
    acceptance must require the exact five frozen external source identities and
    the exact nine standard normalized artifacts, evidenced by a semantic result
    whose admission prohibitions were physically recomputed.
28. No-training JSON scanning initially covered canonical data directories but
    omitted their accepted report roots. Adding `training_authorized=true` to
    both QC reports preserved two-build equivalence and was not seen by the
    scanner. Every accepted QC/report JSON must be included. The external child
    result also rendered its output root relative to the process working
    directory, so identical bytes could produce different final report hashes;
    the integration record must normalize every reported path against the
    declared project root.
29. Byte-identical A/B tree comparison did not establish independent physical
    builds when corresponding files were hard links to the same inode. An
    identical pre-existing final report could likewise be a hard link mutable
    through an untracked alias after acceptance. Independent trees and final
    immutable evidence must reject shared-link/inode aliases in addition to
    symlinks and special files.
30. The nearly frozen final gate applied canonical portable-path checks to its
    output but initially allowed a literal backslash in an input path on POSIX,
    where it is a valid filename character rather than a separator. That could
    produce a machine-specific path in otherwise portable evidence. The
    accepted input resolver now rejects backslashes as well as absolute paths,
    parent traversal, control characters, and noncanonical path spellings.
31. The third real canonical attempt completed all 23 input parts and wrote 203
    Parquet artifacts, including 3,938,372 source observations and matching
    lineage rows, but failed before manifest construction, bound QC, or
    promotion. `derived_observations/part-00006.parquet` held 45/45 null
    `evidence_stage` values and therefore acquired physical type `double`, while
    sibling shards used the normative string type. The failed build was moved
    intact to
    `research/data/platform/quarantine/full_chembl37_failed_attempt_3_arrow_schema_unification/`;
    its rows are evidence of a failed transaction, not an accepted corpus. The
    first normative-schema repair candidate correctly recognized the null-only
    exception but was independently rejected before rerun authorization: an
    injected second `os.replace` failure partially rewrote a logical group,
    in-root symlinks could rewrite their targets, and its regression did not
    reproduce the live 45-row shard or establish the enclosing disposable
    `.building` directory as the tested atomic-publication boundary. A second
    candidate documented and tested that outer boundary and the live offender,
    but the audit then reproduced two path escapes: dirty whitespace/control
    paths were cleaned and accepted, and a dangling staging symlink could create
    an outside Parquet and become a canonical shard. The frozen repair at module
    SHA-256 `240b52193bf8d275c2401e13632d708810daab764112d2037c0c74376d2b980d`
    and test SHA-256
    `e3a601b175ae6edf1c7d9e460b2ed7770fba0b8a0bb5b2ea1bea40fb317a6c29`
    rejects the original, staging, dirty-path, symlink, and hardlink exploits.
    The primary independent gate passed 44 combined tests, Ruff, format, and
    mypy; a second audit replay passed 25 focused cases plus Ruff and mypy. This
    is implementation-level authorization for a fresh attempt, not evidence of
    a promoted real corpus.
32. The first real split A/B executions failed transactionally when RDKit raised
    `bad bond stereo` during Bemis--Murcko extraction. The initial repair
    deterministically returned method
    `exact_smiles_proxy_rdkit_exception`, but the split-suite applicability
    gate recognized only the base `exact_smiles_proxy` name. Independent replay
    produced `(True, [])` for a purported scaffold candidate containing the
    exception proxy: an exact-SMILES grouping could therefore have been
    published as a true scaffold split. Both invalid reruns were interrupted
    before publication and their private staging trees were removed without
    changing raw, canonical, statistical, or corpus artifacts. The frozen
    repair centralizes a suffix-aware exact-proxy predicate in `features.py`,
    makes split-suite scaffold applicability fail closed, and makes direct and
    streaming scaffold generation reject the proxy without publishing output;
    unexpected programming exceptions still propagate. Independent real-SMILES
    replay, 48 focused tests, Ruff, format, and mypy passed under final module
    hashes `628b6627864c3b795c81f68ceca0d67df85b974a02b914213d339f542493be8f`
    (`features.py`),
    `2524a5ec9b39204a764aff9e594af6d8be8b80e217871f41038a9e40246031b4`
    (`platform_split_suite.py`), and
    `712db5b17c1cb9368f3770ddff75cd53a904bf7cbc2dd5d51dad2ff45f0a141c`
    (`platform_splits.py`). The accepted real A/B trees then exposed 214
    exception-proxy rows across 12 overlapping task views. Their scaffold
    candidates were reasoned skips, not materialized pseudo-scaffolds. Across
    all 28 tasks, scaffold outcomes were nine materialized, 17 inapplicable
    mandatory candidates (12 proxy-triggered plus low-support views), and two
    fixed-seed support skips.

These are not cosmetic corrections: each could otherwise permit a corpus to
pass file/count checks while remaining scientifically or operationally unsafe.

## Frozen verifier candidates before real execution

The following implementation bytes were accepted for mechanical real-artifact
execution after the adversarial findings above were repaired. This is code-
level acceptance only; it is not evidence that a pending real build passed and
does not authorize training.

| Gate | Module SHA-256 | Focused test SHA-256 | Independent evidence |
|---|---|---|---|
| Canonical materialization/QC/schema repair | `240b52193bf8d275c2401e13632d708810daab764112d2037c0c74376d2b980d` | `e3a601b175ae6edf1c7d9e460b2ed7770fba0b8a0bb5b2ea1bea40fb317a6c29` | Integral finite-float identity equivalence, the real 45-row null-only schema offender, private-build interruption, canonical paths, and source/staging symlink/hardlink exploits were independently replayed. Primary: 44 combined tests plus Ruff/format/mypy. Secondary: 25 focused tests plus Ruff/mypy. |
| Canonical A/B comparison | `7511a2ae62a611eb0387aea3537a3af927ab3145a20f16d8288307ba2ce967fe` | `61d66a55f9d30a0d35ed3d70b043c4f9fceb00bf8b2e70b8f093d595e6872f6e` | Report-to-build rebinding, exact tree membership, symlink and distinct-root checks were independently reviewed. |
| Statistical census | `304b0d3077269285016f8292475298094c4c305f03a03259ff65a88069893e16` | `211668f2cf30dcc1b88bf4ece8647c779d1c4460393037cefe25195d3d6b50c4` | Four focused tests, Ruff, and mypy passed before the later real A/B source-reverified execution. |
| Corpus readiness | `ca5150f371f80602d97632d5d8fe538ac7be3ce9f6967fb0911470daae7691cd` | `6378c0505f51522a43ada8e46669c2e8cda9d8a1a4162543c5c697330093cdab` | Test-label/preflight and acceptance-tampering exploits were closed before the later 524-component real bundle passed source rebind. |
| Official split suite | `2524a5ec9b39204a764aff9e594af6d8be8b80e217871f41038a9e40246031b4` | `61dba4f479544d45e2445800322b1aef0a6e0ba82a6e61902cbac6c0ae66c2a9` | The final RDKit-exception repair passed 48 focused tests, independent real bad-stereo replay, Ruff, format, and mypy. Shared split engine/test hashes are `712db5…` and `b08797…`. |
| Final cross-workstream verifier | `b387a218c77fd7ed921c479278084e868251a10336f02e195012445434a46786` | `311eef6286053d6f2429b335dcdded3de5007dabfb5615640c71ef8022bca0a9` | 28 focused and 65 combined FINAL/integration/SPLIT/STAT/CORPUS tests passed with Ruff and mypy; output/input aliasing, recursive no-training policy, child contracts, exact external topology, hardlinks, report roots, and portable paths were covered. |

The implementation table is superseded by the physical closure below.

## Final physical closure

`research/reports/final_verification/platform_final_artifact_verification.json`
exists at SHA-256
`f27ceb4d46edeb9c0dfb2610ef5d2b02075aace1ee1f1051351ec4876bd945d5`
and 20,573 bytes. It passed mechanical artifact verification and reports
`artifact_integrity_and_source_rebinding_verified=true`. The independent audit
reopened its critical inputs and preserved these exact outcomes:

- canonical A/B content equivalence: 260 components and 1,154,513,167 bytes;
- statistical A/B: both source-reverified; 19 byte-identical files and
  1,374,306 bytes;
- split A/B: both directly source-reverified; acceptance SHA `45393617…`;
  479 components per tree; 480 byte-identical files, 225 directories, and
  564,604,068 bytes; 85 strategies materialized and 83 skipped;
- corpus readiness: source-reverified, 524 components, 22 tasks integrated,
  six support-skipped, and test lockboxes neither opened nor hashed;
- five raw external bundles and nine normalized artifacts reverified, while
  preserving zero external canonical observations and labels;
- 2,117 generated regular files checked with zero hardlink/inode aliases; and
- 1,090 JSON documents scanned with both training flags false, no training
  actions, and no routed test payload opened.

Software reconciliation also passed: 579 pipeline tests, 53 Menin-Edit tests,
Ruff, a 177-file format check, no-incremental mypy over 80 source files, `pip
check`, and the 53-pin core dependency audit with zero known vulnerabilities.
The expanded local environment separately contains unused `aiohttp==3.14.1`
with three 2026 advisories; this remains a local-environment hygiene issue and
is not misreported as a core-lock finding.

Mechanical closure does not change the conjunctive scientific decision. The
same final report sets task claim readiness, substantive-training readiness,
and training authorization to false and lists eight human/external blockers.

## Governing-specification coverage matrix

The `Acceptance evidence` column defines what counts. “Implemented interface”
never substitutes for an executed real-corpus analysis.

| Section | Required closure | Acceptance evidence |
|---:|---|---|
| 1. Full project audit | Dataset/code/output/migration/confidentiality inventory and implementation-versus-documentation review | `repository_inventory.md`, discrepancy CSV, risk register, final git/staged inventory |
| 2. Collection/provenance | Preserved raw bytes, dated source/license/citation identity, failed retrievals, overlap policy | ChEMBL archive/extraction manifests plus frozen external-source manifests and acquisition report |
| 3. Schema/dictionary | Canonical fields, types, units, roles, provenance, missing conventions, observation semantics | Machine schema, data dictionary, ontology crosswalk, exact Arrow schema fingerprints |
| 4. Integrity/QC | Machine-readable complete QC, corrections, unresolved queues, count conservation | Bound `qc_report.json`, issue/missingness/distribution tables, adversarial tests |
| 5. Dedup/entity resolution | Stable molecule/protein/assay/observation identities without collapsing meaningful repeats | Entity registries, lineage, conflicts, cross-source mirror report, identity-policy sensitivities |
| 6. Normalization | Original and normalized values, deterministic rules, unit/relation validation, effect counts | Transformation fields/versions, conversion fixtures, quarantine and derivation audits |
| 7. Missing data | Source-universe denominators, missingness mechanisms, attrition, supported handling alternatives | Missingness/attrition artifacts and the frozen analysis plan |
| 8. EDA | Target/source/context distributions, concentration, time/batch effects, coverage, useful figures/tables | Executed real-corpus QC/EDA outputs with denominators and uncertainty where meaningful |
| 9. Target/label validation | Endpoint-specific task signatures, censoring semantics, thresholds, lineage, leakage exclusion | Task registry, task-observation equality audit, hERG/free-energy tests |
| 10. Preprocessing | Raw-to-canonical-to-task deterministic path, atomic publication, no silent loss | Immutable manifests, pre-promotion QC, reusable-final QC, two-build equivalence |
| 11. Features | Versioned admissible families, failures/masks, target-leakage policy, train-fit boundary | Feature registry and data-dependent feature coverage/failure artifacts |
| 12. Representations | Serialization, vocabulary, truncation, masks, graph/text/protein handling, bounded loading | MODEL interfaces, synthetic scale evidence, real-corpus integration bundle |
| 13. Splits/leakage | Intended-use splits, exact and near ligand/protein/source/time/assay overlap, immutable lockbox | Split manifests, physical partitions, near-neighbor and homology audits |
| 14. Imbalance/sampling | Global/subgroup support and prespecified alternatives without synthetic shortcut | Task prevalence/support outputs and robustness matrix |
| 15. Statistics | Frozen estimands, uncertainty, model comparison, multiplicity, missingness sensitivity | Statistical analysis plan plus later executed locked analyses |
| 16. Baselines/sanity | Fixed modest baselines and negative controls on identical development splits | Diagnostic bundle, predictions/errors, permutation/identifier controls; no test use |
| 17. Fine-tuning preparation | Candidate/checkpoint/input/loss/optimizer/resume/PEFT/resource contracts | Candidate registry, config validation, tiny smoke only, exact checkpoint preflight still required |
| 18. Metrics | Hand-checked primary/secondary/diagnostic metrics, calibration and subgroup framework | Metric registry and focused tests; real-task results only after task gate |
| 19. Robustness | Prespecified preprocessing/identity/source/split/threshold/representation alternatives | Sensitivity and baseline-robustness matrices with explicit unrun statuses |
| 20. Bias/subgroups | Selection mechanisms, representation, missingness, support thresholds, no unsupported fairness claim | Bias/missingness/subgroup plan plus executed corpus artifacts |
| 21. Error analysis | Stable IDs, source links, ranked errors, taxonomy, AD/confidence, blinded review queue | Diagnostic error artifact contract and later locked-model outputs |
| 22. Reproducibility/software | Coherent dependencies, tests/lint/types, no personal paths/secrets, clean rebuild | Exact command log, clean-clone/staged checks, manifests, deterministic builds |
| 23. Documentation | Actual/planned separation, complete methods/limitations/reproduction handoff | Platform docs, reports, source/model review, claim scan |
| 24. Output management | Raw/interim/canonical/QC/features/splits/models/reports isolated and immutable | Directory policy, component inventories, atomic/non-overwrite tests, disclosure scan |
| 25. Final readiness | Every feasible pretraining stage reconciled; residual blockers and next steps explicit | This record, readiness rubric, lead sign-off, no-training attestation |

## Scientific non-equivalence checks

The final lead review must preserve all of these boundaries:

- ChEMBL is a large single release/aggregator corpus, not proof of independent
  multisource, clinical, or regulatory coverage.
- Specialized Kd/Ki, IC50/EC50, hERG, PK/ADME, and QT/APD views overlap and are
  candidate inventories; their counts are not additive and their rows are not
  automatically model eligible.
- `Kd`, `Ki`, `IC50`, `EC50`, kinetic, PK, hERG, QT, clinical, regulatory, and
  derived free-energy evidence are separate endpoint/task families.
- Standard binding free energy may derive only from exact positive compatible
  `Kd`, with standard state, temperature, formula, approximation status, source
  observation, and integrity digest. It is never a measured label or a default
  observed-label task.
- hERG binding/function is not QT prolongation, torsades, clinical
  cardiotoxicity, or patient safety.
- Trial registration/development phase, posted results, regulatory approval,
  label text, and postmarketing evidence are distinct; absence is never a
  negative outcome.
- ChEMBL molecule-development annotations are metadata, not clinical outcomes.
- External model outputs are predictions under their documented endpoint and
  training domain, never ground truth or interchangeable affinity/free-energy
  labels.

## Required physical DATA gate

The accepted frozen export must pass all of the following from independent
reads, not only its generating process:

1. archive size/hash, 24-piece assembly coverage, database size/hash, SQLite
   quick check, release identity, and core table counts;
2. exact manifest-to-directory membership, byte sizes, SHA-256 values, Parquet
   footer counts, global activity-ID ordering/uniqueness, query/database/source
   bindings, and source-attribution completeness;
3. exact Arrow field order, types, nullability, and declared metadata across all
   123 full-activity parts and every specialized/development part, with the
   recomputed fingerprint equal to its manifest and no surviving schema-
   normalization marker, `.part`, checkpoint, or mixed-generation shard;
4. the schema repair transaction accounts for exactly 162 Parquet artifacts
   (123 full activity, 23 specialized activity, 15 development, and one target
   component); updated manifests are staged before commit; an injected mid-swap
   interruption blocks every reader; idempotent resume or whole-directory
   rollback/swap restores a fully bound generation; receipt membership, hashes,
   and schemas match all 162 files;
5. child/summary manifest identity and digest agreement, exact candidate-view
   counts, target-component inventory, and independently recomputed overlaps;
6. canonical component membership/hashes/counts/schemas, no unmanifested
   component, task/entity/lineage count conservation, provenance equality,
   derived-label round trips, and `qc_passed=true` bound to the exact build
   manifest digest;
7. failed-QC non-promotion, reused-final fresh bound QC, tamper/unlisted-file
   rejection, all-null/type-drift rejection, and no active `.building` input;
8. a second independent build from the same frozen inputs with equivalent
   content after only explicitly declared nondeterministic timestamps are
   normalized.

## Required MODEL gate

The accepted interface and any real-corpus bundle must prove:

- source directory membership, part hashes, row counts, schema, record-ID
  uniqueness, and task signature are bound to the final DATA manifest;
- split rows reconcile exactly and partitions are physically distinct;
- the final bundle contains exactly train, validation, and test JSONL payloads,
  with no combined corpus or dangling combined path;
- the test payload is bound during routing and is not opened, hashed, iterated,
  or used after routing; vocabulary, length policy, fitting, diagnostics, and
  loader smoke are train/validation only as declared;
- derived labels require explicit sensitivity opt-in, censored/ordinal tasks are
  skipped rather than coerced by unsupported diagnostics, and predictions can
  never enter labels;
- every committed non-acceptance component has a non-circular path/hash/size
  inventory; failures clean only their private transaction and do not overwrite
  a prior accepted bundle;
- tiny/synthetic smoke outputs state `substantive_training_started=false` and
  are not reported as scientific performance.

## Required external-evidence gate

Acquisition success is distinct from canonical admission. Each source must first
pass physical manifest verification and then its domain-specific semantic gate:

- BindingDB 202608: admit only rows whose declared curation origin is approved;
  exclude imported ChEMBL rows as independent multisource evidence, quarantine
  Taylor Research rows pending rights review, retain assay/reaction-set/article/
  protein mappings, and report exact/near duplicates and conflicts against
  ChEMBL without erasing either source assertion.
- UniProtKB 2026_02: reconcile requested, returned, missing, obsolete/replaced,
  isoform, sequence-conflict, and multi-mapped accessions exactly; bind sequence
  checksums and release identity; quarantine construct mappings that canonical
  sequence cannot resolve.
- Drugs@FDA: parse all declared bulk tables, conserve application/product/
  submission keys and source-update identity, and expose approval evidence only
  at its indication/formulation/date scope.
- DailyMed human prescription SPL: verify every declared release part and
  upstream checksum, preserve SETID/version/effective-date and XML member
  location, inventory label sections, and prohibit prose/section presence from
  becoming a normalized molecule-level outcome.
- ClinicalTrials.gov v2: collect the alias-independent cohort of all studies
  whose intervention type is `DRUG`; bind the exact page-token chain, API
  version, and pre/post `dataTimestamp`; reconcile advertised, physical, and
  unique-NCT counts; retain registry status separately from results-posting
  status; and make molecule linkage a later exact-match/ambiguity-quarantine
  layer. Prove that no absent study/result is encoded as a negative label.

Downloaded bytes or an exact-name candidate match alone cannot satisfy these
admission rules. Human adjudication queues remain explicit evidence, not silent
mapping failures.

## Human and external blockers that code cannot waive

The immutable final verifier records exactly eight blockers:

1. resolve source-specific rights and canonical admission, ambiguity and
   quarantine, cross-source duplicate/conflict reconciliation, molecule
   linkage, and genuinely reported clinical, PK, QT/QTc, and outcome evidence;
2. complete public-release staged disclosure, secrets/PII/private-
   correspondence review, and large-artifact storage/redistribution decisions;
3. approve the repository license and all conditional source/model
   redistribution terms;
4. freeze the intended scientific task and accept ligand, protein, source,
   assay, and temporal leakage thresholds;
5. freeze exact model checkpoint revisions/hashes, weight terms, training
   cutoff, and corpus-overlap audit;
6. approve accelerator allocation, measured budget, monitoring,
   checkpoint/resume, and responsible-compute controls;
7. obtain appropriately designed independent external/prospective evidence
   before broad translational, safety, or product claims; and
8. identify and review the meeting-referenced model reportedly trained on
   approximately 100,000 structures.

## Training boundary

No substantive large-model pretraining or fine-tuning is authorized by this
record. Static registries, deterministic preprocessing, lightweight diagnostic
baselines, bounded synthetic scale runs, and tiny loader/forward-backward smoke
tests are engineering evidence only. The exact next steps after all blocking
gates pass are: obtain human authorization; freeze the task, split, model,
checkpoint, license, overlap and compute manifests; run a capped dry run on the
approved hardware; verify loss/masks/checkpoint-resume and measured resources;
then launch the prespecified training job under monitoring without accessing the
sealed test set.
