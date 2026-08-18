# Reproducibility

## Reproducibility levels

The project distinguishes three levels:

1. **Computational repeatability**: the same files, code revision, configuration, environment, and seed reproduce materially equivalent outputs on one platform.
2. **Data reconstruction**: a recorded public-source snapshot and linked manifests allow another environment to verify exactly which bytes and table schemas were analyzed.
3. **Scientific reproducibility**: an independent group obtains compatible results with independently acquired data or prospective experiments.

The pipeline directly supports the first two. The third requires external and prospective validation.

## Environment setup

Pip development setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Conda setup:

```bash
conda env create -f pipeline/environments/environment.yml
conda activate menin-discovery
python -m pip install -e . --no-deps
```

The checked-in `pipeline/environments/requirements.lock` records the complete Python 3.13 environment used for the current macOS build, including development tools. Reconstruct that environment with `python -m pip install -r pipeline/environments/requirements.lock` followed by `python -m pip install -e . --no-deps`. Because wheel availability and compiled numerical libraries are platform-specific, the lock is exact provenance for this build rather than a promise that every pinned wheel exists on every operating system. Archive the platform, Python version, package inventory, code commit, and lock file together. The broader `pipeline/environments/requirements.txt` remains the portable installation specification.

Verify the environment before a release run:

```bash
python --version
python -m pip check
python -m pip freeze
python -c "import rdkit, sklearn; print(rdkit.__version__, sklearn.__version__)"
pytest
ruff check .
```

Publication modeling should force the RDKit backend. `auto` is useful for smoke tests but can resolve to a non-chemical hashed-SMILES fallback if RDKit is unavailable.

## Configuration

The versioned default policy is [`pipeline/config/pipeline.yaml`](../pipeline/config/pipeline.yaml). Record the complete resolved settings snapshot, not only the values changed on the command line.

| Section | Examples of controlled decisions |
| --- | --- |
| `project` | Project name and random seed. |
| `paths` | Raw, processed, model, analysis, and report roots. |
| `curation` | Core endpoints, RDKit/parent/tautomer policy, exact-value policy, PubChem target relevance, ChEMBL validity, duplicate/variant policy, cross-source mirror policy, heterogeneity threshold, and endpoint/assay stratification. |
| `herg` | Blocker/non-blocker thresholds, primary endpoint/assay family, and pooled-sensitivity control. |
| `modeling` | Primary endpoint/assay family, exhaustive eligible-task analysis, primary and comparison splits, holdout sizes, minimum support, fingerprint bit/radius parameters, bootstrap count, applicability-domain quantile, and uncertainty coverage. |
| `analysis` | Enable/disable flag; primary Menin/hERG task; Morgan radius/bits; dated approved-reference panel; series/Butina/cliff/MMP thresholds; alert catalogs/property windows; hERG evidence bands; scoring weights; chemistry gate; prioritization sensitivity policy; and prospective-selection categories, quotas, and per-series cap. |

Changing any curation, label, split, feature, chemical-intelligence, reference-panel, or prospective-selection setting creates a new analysis specification and should produce a new build ID/output directory or release tag. Do not overwrite the only copy of a reported build.

## Recommended run sequence

### Existing immutable raw snapshot

```bash
menin-pipeline --config pipeline/config/pipeline.yaml --stage all --skip-network
```

The `all` stage runs curation, fail-closed quality gating, initial data/software manifests, transactional model generation, the transactional `analyze` stage when enabled, reporting, manifest creation, and verification in dependency order. With analysis enabled this is the six-manifest DAG; with analysis disabled it is the five-manifest chain. It then refreshes the semantic readiness report and re-manifests/re-verifies the final report bundle. Model provenance is linked to the current processed/software manifests, analysis to processed/models/software, and reports to processed/models/software/analysis when present.

When debugging individual stages, preserve the same dependency order. In particular, a standalone `analyze` requires a compatible completed model build and its Menin/hERG scoring table; it refreshes data/software manifests and validates model lineage before replacing `research/analysis/`. Rebuild the report after changing analysis and rerun `manifest` plus `verify` after the final report. A standalone `manifest` stage creates all available release manifests, so an interim pre-model invocation is not the final release evidence.

```bash
menin-pipeline --config pipeline/config/pipeline.yaml --stage analyze --skip-network
menin-pipeline --config pipeline/config/pipeline.yaml --stage report --skip-network
menin-pipeline --config pipeline/config/pipeline.yaml --stage manifest --skip-network
menin-pipeline --config pipeline/config/pipeline.yaml --stage verify --skip-network
```

### Public-source refresh

```bash
menin-pipeline --config pipeline/config/pipeline.yaml --stage collect
menin-pipeline --config pipeline/config/pipeline.yaml --stage all --skip-network
menin-pipeline --config pipeline/config/pipeline.yaml --stage verify
```

Separating collection from analysis makes network failures and source drift easier to audit. Collection first copies the existing raw root into an all-source staging area, refreshes the staged sources, records limits and promotion policy, and promotes that snapshot only after success. Preserve source status, search-term, catalog, download-status, and collection metadata with the raw snapshot.

### Smoke test

```bash
menin-pipeline --stage all --skip-network --fast
```

Fast mode automatically appends `smoke/` to the configured model, analysis, and report roots, producing `research/models/smoke/`, `research/analysis/smoke/`, and `research/reports/smoke/`. Reduced artifacts therefore cannot overwrite release models, analysis, or reports. It does **not** create separate raw or processed roots; it reads the same configured snapshot. Pass `--fast` again when verifying the smoke namespace.

`--fast` is for CI and interface checks. Do not quote its reduced-compute metrics as final results.

## Endpoint and split analyses

Run Menin endpoint-specific tasks rather than pooling endpoints when there is adequate support:

```bash
menin-pipeline --stage models --skip-network --menin-endpoint IC50 \
  --menin-assay-family biochemical_binding --split-strategy scaffold
```

By default, a full model run also analyzes every eligible endpoint × assay-family pair with at least the configured minimum compound count. The pre-specified Menin primary is `IC50 × biochemical_binding`; the pre-specified hERG primary is `IC50 × electrophysiology_functional`, and pooled hERG is labeled only as sensitivity evidence. At minimum, compare the configured scaffold, temporal, and compound-grouped random evaluations for the Menin primary task. Report scaffold as the primary generalization estimate, temporal only when years are adequate and meaningful, and random as an optimistic comparator. PubChem contributes assay-deposit year while ChEMBL/BindingDB commonly contribute document years, so disclose this mixed temporal clock and the recorded `date_provenance`. Record any automatic split fallback. Do not choose a task definition retrospectively because its holdout is favorable.

The default full run also includes two Menin curation sensitivities: one retains potential cross-source mirror rows that the central label collapses, and one excludes structures above the configured within-structure spread threshold. Compare their populations and metrics with the central analysis; do not promote the most favorable result.

For release-quality comparisons, save each policy in a separate output location or immutable build; do not let sequential exploratory runs overwrite reportable artifacts.

## Chemical-intelligence reconstruction

The analysis root is a derived, content-addressed build. Preserve `research/analysis/analysis_summary.json`, `research/analysis/analysis_build_metadata.json`, every table, the resolved `analysis` configuration, and the exact RDKit version. The summary records the SHA-256 of the four direct input files and the algorithm contract for fingerprinting, scaffolds, clustering, cliffs, alerts, and unknown hERG evidence.

The configured approved-reference structures and regulatory/source URLs are part of the resolved specification. For each frozen release, record when PubChem structures and FDA status/indications were rechecked. `approved_reference_coverage.csv` is a coverage benchmark, not a clinical-efficacy comparison.

If the experiment template will guide testing, freeze `prospective_selection_plan.csv`, its quota/shortfall summary, the complete resolved analysis configuration, availability/identity review, assay protocol, endpoints, blinding, success criteria, and statistical plan before generating outcomes. A generated plan is not itself prospective evidence.

Deterministic sorting/stable identifiers make the same inputs and environment repeatable, but RDKit algorithm/version changes can alter standardization, descriptors, fingerprints, scaffolds, alerts, fragments, clusters, and pairs. Treat such a change as a new analysis build even when configuration text is unchanged.

## Release manifests

With `analysis.enabled: true`, the manifest stage creates linked `raw`, `processed`, `software`, `models`, `analysis`, and `reports` manifests. Verification checks:

- the manifest's own digest;
- dataset digest, file count, and total size;
- path safety and duplicate entries;
- file presence, byte size, and SHA-256;
- tabular format, row count, column count, and schema when recorded;
- matching build IDs;
- raw → processed, processed/software → models, processed/models/software → analysis, and processed/models/software/analysis → reports upstream digests;
- that every serialized release model is covered by a matching per-model hash and the exact processed/software manifest lineage recorded at training time;
- that the analysis build and its recorded direct-input SHA-256 values match the current processed/model/report inputs;
- that the report-build digest matches the processed, software, model, analysis, settings, and report bytes captured before manifest creation; and
- that all six stage manifests exist. Smoke artifacts and verification/run-metadata files are isolated from release content digests.

When analysis is explicitly disabled, the analysis manifest/upstream link is omitted and verification requires the remaining five manifests. Do not mix the two DAG shapes within one claimed release.

Run verification before using a cached snapshot and immediately before archiving a release:

```bash
menin-pipeline --stage verify
```

When the underlying data bytes are unchanged, the CLI preserves the prior raw/processed manifest timestamp so the exact manifest hash recorded by model provenance remains stable. Manifest timestamps can also be made byte-reproducible through the provenance API's `created_at` parameter or `SOURCE_DATE_EPOCH`. Content-derived dataset and build digests do not depend on timestamps or absolute paths.

## Model reproducibility record

Each release model should have all of the following:

- serialized estimator and SHA-256;
- input dataset digest and the actual input columns covered by that digest;
- split assignment CSV and split digest;
- cross-validation fold digest;
- requested and resolved split strategy, including fallback reason;
- feature backend, fingerprint size/radius, descriptor list, and fallback status;
- training-CV candidate comparison and selected candidate;
- untouched holdout predictions and metrics;
- scaffold-group bootstrap interval settings, resampling unit/group count, and successful resample counts;
- uncertainty and applicability-domain definition;
- Python, NumPy, pandas, scikit-learn, RDKit, and serialization-library versions; and
- exact code revision, dirty-state flag, resolved configuration snapshot/hash, and processed-build digest linkage.

Per-model manifests capture this evidence automatically. The release-level software/models/analysis/reports manifests protect the complete artifact collection; preserve both layers.

## Determinism limits

Even with fixed `random_state: 13`, exact floating-point results can vary with operating system, BLAS implementation, package build, thread scheduling, and estimator/library changes. Remote sources can add, correct, or remove records. ChEMBL, BindingDB, and PubChem should therefore be cited with the access date or release/status metadata recorded for the build.

For strict comparisons:

- reuse the same verified raw snapshot;
- use the same environment lock and platform/container;
- set thread counts consistently;
- preserve the same split assignment rather than merely reusing a seed;
- compare file/table hashes and toleranced numeric metrics; and
- treat any curation-policy change as a new dataset.

## Release bundle

A publication or internal decision package should include:

- code revision/tag and `CITATION.cff`;
- resolved configuration;
- environment lock or container digest and `pip freeze`;
- raw, processed, software, models, analysis, and reports manifests plus every verification report when analysis is enabled;
- curation summaries and quarantine reason counts;
- final measurement/modeling tables subject to source and confidentiality permissions;
- split assignments, candidate comparisons, metrics, confidence intervals, and calibration/domain diagnostics;
- the complete analysis root, including decision traces, data gaps, sensitivity scenarios, series/clusters, novelty, cliffs, MMPs, connectivity audit, approved-reference coverage, and direct-input hashes;
- final figures/tables and their generation command; and
- a limitations statement and source-specific citations.

Before calling a bundle publication-ready, the working tree must be clean and committed; all six enabled-stage manifests and the eligible-data quality gate must pass; the intended release license, authorship, and third-party redistribution decisions must be approved; and independent external plus prospective validation must be complete or explicitly identified as absent. Generated readiness rows are evidence checks, not governance or scientific sign-off.

If raw or proprietary data cannot be released, publish only approved aggregate evidence and a synthetic or public-only reproduction path. Do not claim full computational reproducibility when essential inputs are unavailable; state the access restriction and independent verification process explicitly.
