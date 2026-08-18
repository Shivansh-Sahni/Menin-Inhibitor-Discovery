# Independent repository and scientific inventory

Audit snapshot: 2026-08-04. Base commit:
`01210746809b7acec64c86a0e7f4b97a6fb5574f` (`Initial menin inhibitor
discovery workflow`, 2026-06-09). Starting file counts below remain historical
inventory facts; the final platform-artifact closure is recorded separately
from the still-dirty release migration.

## Executive finding

The repository contains a substantial Menin/hERG legacy workflow, a separate
mixed-access local research workspace, and a newly completed public-platform
artifact stack. It did **not** begin this audit as a protein-agnostic,
clinical-evidence-aware corpus. It now has a large deterministic ChEMBL-only
canonical corpus, real statistics, fixed splits, corpus integration, and
bounded diagnostics. It is not a rights-cleared multisource clinical corpus and
does not validate a production foundation model.

Independent status at this snapshot:

- legacy public-data evidence and baseline workflow: `locally_evidenced`;
- frozen platform artifact/source integrity: `passed` by the final verifier;
- clean public release: `not_passed` because the migration is dirty, no
  repository license exists, and staged/clean-clone review remains;
- new platform canonical/split/corpus interface: `executed_real` for the
  ChEMBL-only snapshot;
- production HPC physics and large-model work: `hardware_blocked` or `planned`;
- source/model/data redistribution rights: partly `requires_human_review` and partly `license_blocked`;
- substantive large-model training: `not_applicable` to this readiness phase and not authorized.

The current independent verdict is **not ready for substantive large-model training**. Passing interface smoke tests is necessary but not sufficient to change that verdict.

### Final platform checkpoint

The completed integration materially advanced beyond the starting inventory
while preserving the substantive-training verdict:

- the official ChEMBL 37 archive/database and the complete 24,527,044-row
  source-assertion export are verified; a durable explicit-schema transaction
  binds all 162 full/specialized/development/target Parquets;
- five required public source acquisitions now pass exact recursive byte and
  semantic verification: 404 source artifacts (18,913,867,042 bytes) and 853
  bundle artifacts (18,994,847,039 bytes), with ten archive inventories;
- a deterministic external-normalization bundle now binds nine artifacts,
  60,377,234 bytes, and 464,123 Parquet rows while retaining zero canonical
  observations, zero labels, and no training. It separates BindingDB origins,
  UniProt sequence/quarantine states, registry cohort membership, and regulatory
  archive inventories without inventing cross-source identity or outcomes;
- canonical A/B are content-equivalent across 260 components and
  1,154,513,167 bytes; the promoted build contains 3,938,372 observations and
  matching lineage, 1,304,011 molecules and aliases, 430,776 assays, 9,411
  proteins, and 9,071 constructs;
- statistical A/B are source-reverified and byte-identical: 19 files and
  1,374,306 bytes;
- split A/B are independently generated, directly source-reverified, and
  byte-identical: 480 files, 225 directories, 564,604,068 bytes, with 85
  materialized and 83 reasoned-skip decisions;
- corpus readiness binds 524 components, integrates 22 tasks, support-skips
  six, completes 11 capped diagnostics, skips 11, and does not open or hash
  test lockboxes;
- the final physical verifier is 20,573 bytes at SHA `f27ceb…`, checks 2,117
  generated regular files with zero aliases, and reports artifact/source
  integrity true while scientific-task readiness, training readiness, and
  authorization remain false;
- final software evidence is 579 pipeline tests, 53 Menin-Edit tests, Ruff,
  177 formatted files, no-incremental mypy over 80 source files, `pip check`,
  and 53 core dependency pins with zero known vulnerabilities.

These additions are large and mechanically verified, but they do not make the
corpus multisource-admitted, clinically adjudicated, exhaustively near-leakage
accepted, externally validated, rights-approved, clean-release-ready, or ready
for substantive training.

## 1. Repository inventory

At the observed snapshot:

| Surface | Observed state | Status |
|---|---|---|
| Git history | One commit on `main`; large in-progress migration from root `data/src/scripts/tests/reports/models` into `research/pipeline` | `requires_human_review` before release |
| Tracked files | 475 | `locally_evidenced` |
| Git short status | 466 tracked deletions, 9 tracked modifications, 20 untracked top-level/path entries; 1,867 non-ignored untracked files | `failed_current_check` for clean-release gate |
| `research/` | About 1.0 GB and 3,736 files including public artifacts, literature, ignored mixed-access research, models, simulations, and reports | `locally_evidenced`; access is mixed |
| `pipeline/` | About 7.1 MB and 249 files at snapshot, including package code, scripts, configs, and tests | `locally_evidenced` |
| `packages/` | About 740 KB and 50 files | `locally_evidenced`; role needs release documentation |
| `docs/` | About 276 KB and 23 files before platform audit expansion | `locally_evidenced` |
| CI | Python 3.10/3.12/3.13 jobs install Menin-Edit `[lab,dev]`, type both packages, test, lint/format, audit the 53-pin provenance lock, and audit each resolved matrix environment | `locally_evidenced`; hosted clean-branch run remains release evidence |
| Repository license | No `LICENSE`; `docs/licensing.md` records that no code license has been selected | `requires_human_review`; public-release blocker |

File counts include ignored/local-only material and are inventory facts, not a proposed commit set. The public release inventory must be derived from `git ls-files` after staging, not from the workspace tree.

## 2. Migration integrity

An automated old-to-new path comparison found replacements for 461 of the deleted legacy files: 435 were byte-identical at the comparison point and 26 had modified replacements. The modified replacements include processed tables, PubChem catalogs, figures, model/report artifacts, and code/tests. The old pickle models, legacy metrics/predictions, root requirements file, and runner do not all have byte-identical replacements because the reorganized repository uses newer model artifacts/reports and environment files.

This supports a deliberate migration, but not yet a safe commit. The integration lead must review the final staged rename/add/delete set, verify that intended large files are tracked or excluded, and regenerate every manifest whose scope changed.

## 3. Access and confidentiality surfaces

The repository contains ignored raw internal inputs and multiple local research trees whose outputs can inherit confidential information. During integration, the lead classified these as local-only and added scoped ignore rules for:

- `research/data/internal/`;
- `research/data/pk_herg/`;
- `research/models/pk_herg/`;
- `research/reports/pk_herg/`;
- `research/simulations/`;
- `research/outputs/`;
- top-level `outputs/`;
- `.tmp/`.

This is a necessary containment control, not a disclosure review. Already generated model weights, plots, aggregates, logs, cached features, inspected workbook exports, and manifests can leak proprietary structures or assay outcomes even when the raw workbook is ignored. Before any public add/commit, run a staged-file access-class scan and obtain data-owner approval. No internal row values or compound identities are reproduced in this audit.

The public platform artifacts must remain under separate `research/data/platform`, `research/models/platform`, and `research/reports/platform` roots and must be buildable without any ignored input. A public build that silently reads an internal workbook is a hard failure.

## 4. Legacy public dataset

The locally frozen collection metadata reports:

| Source/task | Locally observed snapshot |
|---|---:|
| ChEMBL release | ChEMBL 37, 2026-05-01 |
| ChEMBL hERG source rows | 41,078 |
| Target-filtered Menin rows | 1,776 |
| Menin-associated molecule-activity rows | 8,202 |
| PubChem assays retrieved | 323 |
| Processed Menin measurements | 8,176 |
| Processed hERG measurements | 41,078 |
| Processed PK/ADME observations | 204 |

The processed build reports 2,104 Menin compound-task rows and 11,549 hERG compound-task rows. These counts describe a target-specific legacy snapshot, not platform coverage. The two legacy target-filtered BindingDB exports still lack a recoverable monthly release identity and remain unsuitable as platform evidence; they are superseded, rather than retroactively relabeled, by the independently verified origin-separated BindingDB 202608 acquisition. The PubChem snapshot is content-frozen locally, while live depositor records may later change or be revoked.

Eligibility filters are consequential. The source audit reports thousands of exclusions for unsupported endpoints, missing/non-numeric values, units, structures, nonstandard records, and target or validity issues. The eligible-table gate can pass while an inventory-wide QC report intentionally retains errors for excluded rows; documentation must keep those two populations distinct.

## 5. Legacy models and generalization evidence

The locally reported primary tasks are:

- Menin `IC50 × biochemical_binding`: 849 compounds;
- hERG `IC50 × electrophysiology_functional`: 2,777 compounds, with a strongly imbalanced blocker population and a separately excluded ambiguous band.

The primary models are fingerprint/tree baselines with scaffold-aware evaluation and artifact manifests. Exact structure and reported split-group overlap checks are locally evidenced. These checks do not establish new-target generalization, assay transfer, or prospective utility.

The public temporal stress tests materially underperform the easier scaffold/random views: the Menin temporal result has negative R-squared and weak rank correlation, while hERG temporal discrimination, balanced accuracy, and calibration are modest. These results should anchor expectations under drift. They must not be hidden by headline random/scaffold metrics.

Only one legacy primary Menin structure has an observed matching primary hERG record, and fewer than half are within the legacy hERG similarity domain. Consequently, hERG predictions for that set are liability flags and missing-evidence indicators, not experimentally validated safety rankings.

The PK output is an observation inventory with extensive missing matrix/route context and only a small target-associated chemical set. It is not a general PK model. The configuration explicitly disables production simulations and defers execution to HPC, so local structure/physics artifacts are exploratory rather than production evidence.

## 6. Existing scientific strengths

The legacy documentation already draws several important boundaries correctly:

- `IC50`, `Ki`, `Kd`, and `EC50` are not exchangeable;
- unknown units and censored values are quarantined rather than guessed;
- hERG is not clinical QT or arrhythmia risk;
- random split results are sensitivity evidence rather than the headline estimate;
- source absence is not inactivity or safety;
- applicability-domain and calibration estimates are conditional and not clinical probabilities;
- model-based priorities are hypotheses, not validated candidates;
- internal/private cohorts require separate build roots and access controls.

These safeguards should be generalized, not discarded, during platform expansion.

## 7. Scientific weaknesses requiring closure

### Coverage and ontology

- The legacy namespace, CLI, configs, README, and most reports are Menin-centric.
- No legacy canonical entity graph fully represents protein accession/version, isoform/construct/mutation, molecule material/form, clinical study, regulatory record, evidence stage, and source license as orthogonal axes.
- Binding, functional potency, PK, hERG, QT, clinical outcome, and regulatory evidence are not yet available as a comprehensive multi-target corpus.
- Clinical-development tiers cannot be derived reliably from a compound name match alone; intervention identity and record version require confidence and review.

### Bias and missingness

- Public medicinal-chemistry evidence is publication-, patent-, target-, series-, assay-, and quantification-selected.
- Exact-only training removes censored-only evidence and can shift potency distributions.
- Requiring valid structures, supported units, and sufficient metadata preferentially retains better-documented compounds and sources.
- Public hERG labels and the explicit ambiguous-band exclusion create a selected, imbalanced population.
- PK coverage is context-dependent and not missing at random.
- Cross-source mirror heuristics can both miss approximate duplicates and collapse independent equal-valued measurements.

### Validation

- No harmonized external multi-target test corpus or prospective blinded validation is locally evidenced.
- One scaffold split is not a variance estimate; repeated development splits must not consume the final lockbox.
- Molecular identity leakage is only one form of leakage. Protein sequence/family/pocket similarity, publication/patent series, assay/source, temporal, pretrained-model cutoff, and benchmark overlap all require audits.
- External foundation-model scores may mix incompatible endpoints and may share training data with public benchmarks.

### Release and software

- An earlier 2026-08-04 manifest verification artifact failed while integration
  was in progress. It is superseded for the platform artifact scope by the
  passing final report at SHA `f27ceb…`; it remains relevant history for why
  stale artifacts were not accepted.
- `environment.yml` pins Python 3.12, the frozen lock records a macOS CPython 3.13 build, the package advertises Python 3.10+, and CI spans three versions. This can be valid only if supported and lock-generation environments are documented separately and tested.
- The final dependency lock, editable install, CLI entry points, generated artifacts, and offline build must be verified from a clean clone.
- The repository has no selected software/content/model license.

## 8. Implemented-versus-documented discrepancies

The detailed machine-readable matrix is `implemented_documented_discrepancies.csv`. Highest-impact discrepancies at the audit snapshot are:

1. comprehensive protein-agnostic product vision versus legacy Menin-specific production implementation;
2. mechanically verified platform manifests versus a still-dirty, unlicensed,
   not-yet-clean-clone-reproduced release migration;
3. public-release narrative versus mixed-access derived outputs that required newly added ignore containment;
4. “binding affinity/free energy” product language versus heterogeneous endpoint semantics that prohibit pooling;
5. hERG/cardiotoxicity/QT language versus a hERG assay classifier with no human QT corpus;
6. PK modeling ambition versus a small heterogeneous inventory with no audited endpoint-specific PK models;
7. evidence-tier ambition versus no comprehensive clinical/results/regulatory ingestion in the legacy pipeline;
8. broad structure/model ambition versus exploratory/local or external-model candidates whose training overlap, terms, and hardware remain unresolved.

## 9. Audit commands and evidence

Read-only inventory and comparison commands used include:

```text
git status --short
git ls-files
git ls-files --others --exclude-standard
git diff --stat
git log -1 --format=...
rg --files ...
rg -n ...
find ... -type f
du -sh ...
git check-ignore -v ...
jq ... collection_metadata.json build_summary.json run_metadata.json model manifests
cmp ... for old/new migration replacements
```

Exact final commands, paths, results, and hashes are recorded in
`final_reproduction_and_next_steps.md` and
`independent_validation_results.json`. Network source facts and citations are
recorded in `docs/platform/source_and_model_review.md`.

## 10. Final lead-check disposition

1. Platform writers stopped and the final artifact roots were physically
   rebound: **passed**. The Git staging inventory remains a release review.
2. Workstreams and discrepancy/status documents reconciled: **passed for the
   post-gate audit handoff**.
3. Public platform artifact dependency and access-class boundary: **passed
   mechanically**; human staged disclosure remains.
4. Full tests/lint/format/types: **passed** (579 pipeline, 53 Menin-Edit, Ruff,
   177 format, 80-source mypy).
5. Canonical, statistical, and split independent A/B determinism: **passed**.
6. Exact split overlap: **passed**; capped near-similarity remains not
   claim-ready and requires task-owner threshold acceptance or expansion.
7. Final offline platform verification: **passed**, report SHA `f27ceb…`.
8. Staged internal-data/personal-path/secret/large-file review: **human release
   blocker**.
9. Code/content/model/source license approval: **human blocker**.
10. Tiered decision: **interface/artifact-ready for frozen ChEMBL-only bytes;
    not public-release-ready, broad-corpus-ready, paper/product-claim-ready, or
    substantive-training-ready**.

No umbrella “complete,” “publication-ready,” “production-ready,” or
“pretraining-ready” claim is supported beyond the explicitly mechanical
artifact boundary.
