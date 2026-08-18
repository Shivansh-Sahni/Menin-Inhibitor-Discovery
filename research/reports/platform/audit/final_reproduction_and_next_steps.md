# Reproduction contract and next steps

Snapshot date: 2026-08-04. Run every command from the repository root. The
commands and manifests use project-relative paths and contain no workstation-
specific home directory. Do not substitute a provisional `.building`
directory for a promoted artifact.

This human-readable file is post-gate reconciliation. The immutable final
machine report remains the authoritative record of its own execution.

## Reproduction boundary

Two different operations must not be conflated:

1. **Verify/rebuild the frozen snapshot.** This begins from the preserved raw
   bytes and can reproduce the accepted derived artifacts.
2. **Acquire a new live-source snapshot.** Public sources can change or remove
   records. A later download is a new dataset version and cannot be expected to
   reproduce the 2026-08-04 bytes.

The platform command surface intentionally has no substantive large-model
training command. The commands below end at deterministic artifacts, sealed
splits, capped diagnostic baselines, loader smoke, and final verification.

## Frozen environment contract

The audited environment is macOS 15.7.5 arm64, Python 3.13.7. The host's bare
`python3` was Python 3.14.3, so it must not be used as an implicit substitute.
The lock freezes package names and versions but does not contain distribution
hashes, so it is exact provenance for this workstation build rather than a
bit-for-bit cross-platform wheel guarantee.

```bash
cd "$(git rev-parse --show-toplevel)"
python3.13 --version  # must report Python 3.13.7 for this exact environment
python3.13 -m venv .venv
.venv/bin/python -m pip install -r pipeline/environments/requirements.lock
.venv/bin/python -m pip install -e . --no-deps
.venv/bin/python -m pip install -e 'packages/menin-edit[lab,dev]' --no-deps
.venv/bin/python -m pip install pip-audit==2.10.1
.venv/bin/python -m pip check
.venv/bin/python --version
.venv/bin/python -m pip freeze --all
```

`pip-audit` is a separately pinned audit tool, not a runtime dependency in the
53-entry application lock. Installing it adds a separate transitive tool layer
to this environment; the `pip freeze --all` output must be archived with a
release. The vulnerability input remains the exact application lock shown
below. Until that complete freeze is archived, the lock proves application-pin
provenance but not the entire audit-tool environment.

`packages/menin-edit/` is currently an untracked project surface. A final
checkout/reconstruction bundle must include it through an approved commit or a
separate exact file manifest before the install and package-test commands can
be called clean-clone reproducible.

Frozen control digests:

| Artifact | SHA-256 |
|---|---|
| `pipeline/environments/requirements.lock` | `04da36d2320a625bbfebe4da5135b9396930a778218acd441d65a8fe959e16e5` |
| `pipeline/config/platform.yaml` | `b00971d4d5d4aaf9d765793cc03c23dd73171a2097a3a5dc1b0f2b832fbd1058` |
| `research/reports/platform/audit/dependency_vulnerability_audit.json` | `c223e9769bae512294ab5e1760ba6668084e1e8f6126bf390bf62847a85bb7cf` |

The vulnerability record is a point-in-time advisory query, not a guarantee
against future disclosures. Refresh it before a later release, but retain the
frozen record used here.

## Verify the five preserved external sources

Run from the repository root. Each command must exit zero and recursively
reconcile manifest membership, bytes, hashes, archive inventories, and the
zero-admission boundary.

```bash
.venv/bin/python -m menin_discovery.platform_cli verify-external --source-root research/data/platform/raw/external_public/bindingdb_curated_202608 --manifest research/data/platform/raw/external_public/bindingdb_curated_202608/bindingdb_curated_202608_manifest.json
.venv/bin/python -m menin_discovery.platform_cli verify-external --source-root research/data/platform/raw/external_public/clinicaltrials_gov_v2 --manifest research/data/platform/raw/external_public/clinicaltrials_gov_v2/clinicaltrials_gov_v2_manifest.json
.venv/bin/python -m menin_discovery.platform_cli verify-external --source-root research/data/platform/raw/external_public/dailymed_spl_v2_human_rx --manifest research/data/platform/raw/external_public/dailymed_spl_v2_human_rx/dailymed_spl_v2_human_rx_manifest.json
.venv/bin/python -m menin_discovery.platform_cli verify-external --source-root research/data/platform/raw/external_public/drugs_at_fda_bulk --manifest research/data/platform/raw/external_public/drugs_at_fda_bulk/drugs_at_fda_bulk_manifest.json
.venv/bin/python -m menin_discovery.platform_cli verify-external --source-root research/data/platform/raw/external_public/uniprotkb_targeted_2026_02 --manifest research/data/platform/raw/external_public/uniprotkb_targeted_2026_02/uniprotkb_targeted_2026_02_manifest.json
```

The frozen source counts and digests are recorded in
`research/reports/platform/external/external_public_acquisition_report.md`.
Reacquisition uses `acquire-external SOURCE` one source at a time and must write
a new versioned snapshot; it is not part of byte reproduction.

## Full deterministic platform build

These commands are the exact default-parameter sequence. Run them in a fresh
reconstruction root that includes the preserved raw snapshot, or after
confirming that every generated destination is absent. Transactional data,
split, corpus, and analysis generators reject unsafe/nonidentical publication;
static readiness deterministically replaces only its six declared registries.

```bash
# 1. Reconstruct raw-to-interim ChEMBL 37 exports, then reconcile all 162 schemas.
.venv/bin/python -m menin_discovery.platform_data_bulk all --archive research/data/platform/raw/chembl_37_bulk/chembl_37_sqlite.tar.gz --raw-root research/data/platform/raw --interim-root research/data/platform/interim --chunk-size 200000
.venv/bin/python -m menin_discovery.platform_cli normalize-chembl-exports

# 2. Build and QC-promote canonical A and an independent canonical B.
.venv/bin/python -m menin_discovery.platform_cli canonicalize-chembl
.venv/bin/python -m menin_discovery.platform_cli canonicalize-chembl --canonical-root research/data/platform/determinism_build_b --reports-root research/reports/platform/determinism_build_b
.venv/bin/python -m menin_discovery.platform_cli verify-canonical-determinism

# 3. Normalize the frozen external bundles without admitting observations or labels.
.venv/bin/python -m menin_discovery.platform_cli normalize-external
.venv/bin/python -m menin_discovery.platform_cli verify-external-normalized

# 4. Deterministically regenerate the six static readiness registries.
.venv/bin/python -m menin_discovery.platform_cli prepare-static --evidence-checked-date 2026-08-04

# 5. Run the real zero-training statistical census twice from canonical A.
.venv/bin/python -m menin_discovery.platform_cli analyze-canonical
.venv/bin/python -m menin_discovery.platform_cli analyze-canonical --output-root research/reports/platform/statistical_analysis_determinism_b
.venv/bin/python -m menin_discovery.platform_cli verify-statistical-analysis
.venv/bin/python -m menin_discovery.platform_cli verify-statistical-analysis --output-root research/reports/platform/statistical_analysis_determinism_b

# 6. Build and independently regenerate the fixed-seed split suite from canonical A.
.venv/bin/python -m menin_discovery.platform_cli prepare-split-suite
.venv/bin/python -m menin_discovery.platform_cli prepare-split-suite --output-directory research/data/platform/splits/determinism_build_b/full_chembl37
.venv/bin/python -m menin_discovery.platform_cli verify-split-suite
.venv/bin/python -m menin_discovery.platform_cli verify-split-suite --output-directory research/data/platform/splits/determinism_build_b/full_chembl37

# 7. Preflight every accepted task and run only capped train/validation diagnostics.
.venv/bin/python -m menin_discovery.platform_cli prepare-corpus-readiness
.venv/bin/python -m menin_discovery.platform_cli verify-corpus-readiness

# 8. Replay every verifier and bind the final physical bytes.
.venv/bin/python -m menin_discovery.platform_cli verify-final-artifacts
.venv/bin/python -m menin_discovery.platform_cli status
```

The split defaults made explicit by the command contract are seed `20260804`,
fractions `0.70/0.15/0.15`, batch size `50,000`, near-sample cap `256`, Morgan
radius `2` with `2,048` bits and Tanimoto threshold `0.80`, and protein 3-mer
Jaccard threshold `0.80`. Corpus readiness caps diagnostics at 10,000 training
and 2,500 validation examples per supported task; its loader smoke reads at
most four batches of eight. It does not inspect routed test labels.

## Accepted physical outputs for this snapshot

The sequence above produced and the final verifier rebound these exact
artifacts:

| Output | Accepted identity |
|---|---|
| Canonical A | manifest `1ace39ef8bfc3cb41aa8cc9f54abce734b0d3e45118cffabab402eb97ee937eb`; QC `d481c8afa622db4b4e3396512fa6f4620788d059ce08c0d83e88e68c621bfd90` |
| Canonical B | manifest `85bcbe09fc4c7874a169074e1f97310d1f35a9dd13ed082983d62ccb2a2d7b9d`; QC `acc14bdc3936dc3fbbcd5bf83e61f3d440149acd8625b9bdad86baeddd4562e3` |
| Canonical equivalence | 260 components, 1,154,513,167 bytes; report `ab5fc416642d4be5939020cd890e5aa9b862e9c0ef511bb1e3c225d981ff2db8` |
| Statistical A/B | manifest `5ee4456cb195456bee3a548faa99591522adc7f003e479e07749aec322677045`; 19 byte-identical files, 1,374,306 bytes |
| Split A/B | acceptance `453936171d0ea64c6b627e258fafba8e05aeac97e50f5eda4eebc466a21d5b2a`; 480 files, 225 directories, 564,604,068 bytes; exact-tree inventory `0307c23c8635b4cad86aa03e0673ffb81a115dc9e4dfb81914fd931c2820296a` |
| Corpus readiness | acceptance `386396b134d94ac1e60ff791f481f85db008f0490c877158bd0745f622615fdb`; 524 components; 22 integrated and six support-skipped tasks |
| Static readiness | six artifacts; manifest `97c69cf0590d37e6f2e78a1852e511b68baf32d61a7fa89fcbff07d2f5d5201d` |
| Final verifier | `research/reports/final_verification/platform_final_artifact_verification.json`; 20,573 bytes; `f27ceb4d46edeb9c0dfb2610ef5d2b02075aace1ee1f1051351ec4876bd945d5` |

Both split trees were directly regenerated from canonical feature projections
by the final verifier, not merely compared to each other. The report also
reverified both statistical roots and corpus readiness, checked 2,117 generated
regular files with zero inode aliases, and scanned 1,090 JSON documents without
opening routed test payloads.

## Integrated software checks

Use the frozen interpreter explicitly and suppress test caches/bytecode when a
clean evidence tree is required:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider -c packages/menin-edit/pyproject.toml packages/menin-edit/tests
.venv/bin/python -m ruff check pipeline/src pipeline/scripts pipeline/tests packages/menin-edit/src packages/menin-edit/tests
.venv/bin/python -m ruff format --check pipeline/src pipeline/scripts pipeline/tests packages/menin-edit/src packages/menin-edit/tests
.venv/bin/python -m mypy --no-incremental pipeline/src/menin_discovery packages/menin-edit/src/menin_edit
.venv/bin/python -m pip check
.venv/bin/python -m pip_audit --strict --no-deps --disable-pip --progress-spinner off --cache-dir .cache/pip-audit --format json --requirement pipeline/environments/requirements.lock
```

Final observed results were: pipeline `579 passed` with 64% coverage and one
harmless joblib physical-core warning; Menin-Edit `53 passed`; Ruff passed;
format passed for 177 files; no-incremental mypy passed for 80 source files;
and `pip check` passed. CI installs Menin-Edit with `[lab,dev]`, types both
packages, audits the exact 53-pin macOS/CPython-3.13 provenance lock, and
freezes/audits each resolved matrix environment while excluding only the two
local editable source distributions.

The independent software audit also parsed the CI YAML and exercised its 13
shell snippets, dry-ran 22 Make targets, checked TOML/diff hygiene, and verified
the CLI/status contracts. The command surface contains no substantive training
entry point.

The exact 53-pin core lock returned zero known vulnerabilities at this point in
time. A separate expanded-local-environment audit covered 117 dependencies and
found unused optional `aiohttp==3.14.1` with `PYSEC-2026-3545`,
`PYSEC-2026-3546`, and `PYSEC-2026-3547` (fixed in `aiohttp>=3.14.3`). This is
outside the core lock and declared project dependencies, but it should be
removed or upgraded before the current developer environment is called
release-clean.

For Make targets, force the same interpreter, for example:

```bash
make PYTHON=.venv/bin/python platform-status
make PYTHON=.venv/bin/python platform-verify-final-artifacts
```

The final command/result counts and artifact SHA-256 values are recorded in
`independent_validation_results.json`. A later rebuild must replace them with
its observed results rather than treating this snapshot as evergreen.

## Exact next steps before large-model training

There is no responsible training launch command yet. The next actions are
conjunctive gates, in this order:

1. Resolve source-specific rights and canonical-admission policy for external
   BindingDB, UniProt, ClinicalTrials.gov, Drugs@FDA, and DailyMed evidence.
   Complete molecule/product/intervention linkage, duplicate/conflict review,
   ambiguity quarantine, and genuine clinical/PK/QT/QTc/outcome extraction.
2. Select the repository/content/model licenses and complete the staged
   disclosure, secrets/PII/private-correspondence, redistribution, and large-
   artifact storage reviews.
3. Freeze one intended scientific task and claim. Approve its label policy,
   minimum support, and exact ligand, protein, assay/source, and temporal
   leakage thresholds. Do not use hERG evidence as a QT, torsades, or patient-
   safety label.
4. Rebuild a new versioned canonical corpus after any admission change; rerun
   canonical A/B, real statistical A/B, split A/B, corpus readiness, and the
   final artifact verifier. No earlier final report survives changed bytes.
5. Select and freeze the exact model checkpoint/revision and weight hash,
   tokenizer/config hashes, license terms, upstream training cutoff, and corpus-
   overlap audit. Identify and separately review the meeting-referenced model
   reportedly trained on about 100,000 structures.
6. Obtain approved accelerator/HPC allocation. Freeze measured storage,
   throughput, cost/carbon, monitoring, checkpoint/resume, failure-recovery,
   retention, and responsible-use controls.
7. Implement a training entry point only after steps 1--6 are signed. First run
   a capped dry run on the exact chosen model/hardware, confirm masks, loss,
   gradients, deterministic data order, checkpoint/resume, and resource
   measurements, and keep the sealed test set inaccessible.
8. Re-run `verify-final-artifacts` against the newly frozen pretraining bundle.
   Only an explicit human authorization record may then permit the prespecified
   substantive job. Independent external/prospective validation is still
   required before broad translational, safety, or product claims.

Because the model, checkpoint, intended task, licensing, and compute allocation
are unresolved, inventing a shell command for step 7 would fabricate decisions
that the evidence does not support. The repository correctly omits such a
command today.

## Training attestation

Substantive large-model pretraining or fine-tuning was not initiated. Static
registries, deterministic preprocessing, lightweight capped baselines,
synthetic scale tests, and tiny loader/forward-backward smoke tests are
engineering evidence only and must not be reported as final model performance.
