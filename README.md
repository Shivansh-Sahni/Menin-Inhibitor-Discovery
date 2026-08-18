# Protein–molecule evidence and pretraining-readiness platform

This repository is expanding from a Menin-specific workflow into a reproducible, publication-oriented platform for protein–molecule evidence, binding affinity, hERG/cardiac safety, PK/ADME, development metadata, and leakage-controlled model readiness. The new `platform` namespace is target-agnostic and preserves source assertions, provenance, uncertainty, censoring, quality state, access state, and intended-use boundaries. The earlier Menin/MEN1 workflow remains available as a legacy/reference surface; it is not silently mixed into the new public platform corpus.

The project is an analytical research system, not a validated clinical, toxicology, or compound-selection product. Model outputs are hypotheses that require expert review and experimental validation.

## What is stored in GitHub

This private repository intentionally stays compact. GitHub contains the code,
configuration, tests, documentation, schemas, compact manifests/checksums,
concise reports, and selected meeting or publication materials. Raw downloads,
processed datasets, feature matrices, local campaign runs, model binaries,
checkpoints, caches, and private research inputs are stored separately and are
not uploaded to GitHub. Their absence from a clone is intentional: manifests and
SHA-256 bindings identify the governed external artifacts needed to reproduce or
verify a result. See the [repository and artifact storage
policy](docs/artifact_storage.md) before adding files or sharing access.

## Integrated trainable-data release

The current non-HPC integration provides large, endpoint-specific training surfaces without copying incompatible labels into one target. hERG contains 395,575 broad clean observations and 339,373 confirmed-WT fixed-dose structure labels, with smaller nested preclinical, quantitative pIC50, measurement-method, and clinical-context tiers preserved at their real evidence sizes. PK/ADME contains 642,065 modeling rows across 93 trainable task contracts. Protein-conditioned binding and potency contains 3,533,626 observations: 193,925 Kd, 661,274 Ki, 2,449,281 IC50, and 229,146 auxiliary EC50. PRISM adds 8,372,603 verified finite viability values; LINCS metadata identifies 976,325 structure-linked instances, while its profile-gene positions remain explicitly unscanned rather than being overstated as numeric labels.

The machine registry and prose handoff are in [`v1_0_platform_registry`](research/data/platform/processed/training_surfaces/v1_0_platform_registry). The frozen configuration is [`platform_training_surfaces.yaml`](pipeline/config/platform_training_surfaces.yaml). These releases prepare trainable data only: the production HPC feature store, protein/molecule representation models, physics features, docking, final model fitting, and predictive-superiority evaluation remain pending.

## Expanded platform workflow

The platform command surface intentionally ends at immutable model-ready bundles and capped diagnostic baselines. It does not expose substantive large-model training.

```bash
# Read-only inventory and machine-readable contracts
protein-molecule-platform status
protein-molecule-platform contracts

# Transactional ChEMBL source-schema normalization and canonical materialization
protein-molecule-platform normalize-chembl-exports
protein-molecule-platform canonicalize-chembl
protein-molecule-platform canonicalize-chembl \
  --canonical-root research/data/platform/determinism_build_b \
  --reports-root research/reports/platform/determinism_build_b
protein-molecule-platform verify-canonical-determinism

# Origin-preserving external normalization; zero canonical labels are admitted
protein-molecule-platform normalize-external
protein-molecule-platform verify-external-normalized

# Non-HPC evidence expansion and leakage controls; all outputs remain candidate-only
protein-molecule-platform verify-external-admission
protein-molecule-platform verify-deep-leakage
protein-molecule-platform verify-structure-metadata
protein-molecule-platform verify-context-splits
protein-molecule-platform verify-clinical-results
protein-molecule-platform verify-regulatory-records
protein-molecule-platform verify-pkdb-candidates

# Accepted-corpus census, then exhaustive fixed-split/capped-diagnostic preparation
protein-molecule-platform analyze-canonical
protein-molecule-platform verify-statistical-analysis
protein-molecule-platform prepare-split-suite
protein-molecule-platform verify-split-suite
protein-molecule-platform prepare-split-suite \
  --output-directory research/data/platform/splits/determinism_build_b/full_chembl37
protein-molecule-platform verify-split-suite \
  --output-directory research/data/platform/splits/determinism_build_b/full_chembl37
protein-molecule-platform prepare-corpus-readiness
protein-molecule-platform verify-corpus-readiness

# Final cross-workstream byte/source gate; reports blockers and never authorizes training
protein-molecule-platform verify-final-artifacts

# Local release/resource audit; records human blockers and never grants approval
protein-molecule-platform audit-non-hpc-governance
protein-molecule-platform verify-non-hpc-completion

# Static feature/model registries only; no checkpoint download or fitting
protein-molecule-platform prepare-static --evidence-checked-date 2026-08-04
```

External sources are acquired one explicitly named snapshot at a time with `acquire-external`, then rehashed with `verify-external`. `normalize-external` preserves source rows and dispositions but intentionally admits no canonical observation or model label until the scientific and rights gates are resolved. `prepare-corpus-readiness` exhaustively enumerates the accepted canonical task manifest, predicts one fixed molecule-grouped split without seed searching, records explicit insufficiency skips, creates immutable integration bundles, and invokes only capped train/validation diagnostics while keeping routed test lockboxes closed. The narrower `integrate-task` and `diagnose-task` commands remain available for one-task inspection. Exact accepted source versions, counts, checksums, quarantines, and unresolved blockers are maintained in [`docs/project/pretraining_readiness_ledger.md`](docs/project/pretraining_readiness_ledger.md).

The supplemental non-HPC phase adds deterministic external-admission accounting, an exhaustive/sampled/not-run leakage audit, exact SIFTS UniProt-to-PDB metadata coverage without coordinate downloads, label-blind assay/document/strict-temporal split candidates, 10,144 ClinicalTrials.gov QT/QTc/PK review candidates, 630,928 normalized Drugs@FDA record/anomaly candidates, and a bounded PK-DB audit. PK-DB admits zero observations because its statistics report 138,411 outputs while anonymous output retrieval returned none and record-level reuse rights remain unresolved. The current first large hERG task is the homogeneous PubChem AID 720551 weak-label backbone: 339,373 standardized structures (1,238 Active and 338,135 Inactive) with a fixed, zero-scaffold-overlap 265,625/32,850/40,898 train/validation/test split. A separate assay-native ledger retains 407,956 AID, quantitative pIC50, and ChEMBL observations across 369,546 structures. The paper-facing wild-type scope now admits 407,698 observations—343,909 confirmed WT and 63,789 WT-or-unspecified—while physically quarantining all 258 explicit mutants. Five separate task contracts cover the large weak screen, 23,186 quantitative pIC50 observations, 3,597 retained functional assay-aware observations (3,548 eligible across 2,776 structures), clinical-development context, and 221 QT/QTc endpoints. Measurement modality is indexed across the admitted corpus, including 16,200 patch-clamp and 344,029 thallium-flux observations; only seven curated source rows are clinical QT/QTc phenotypes, which remain separate from hERG labels. The less-conservative operational hierarchy contains 407,956 public observations, 64,047 curated/quantitative preclinical observations, 3,056 clinical-development structures, 1,694 exact trial intervention links, and 3,828 posted-QT result cells, each at an explicit grain. Expanded parent/name resolution provides 1,177 unique clinical-trial drug structures at the practical cleaned-name tier for hERG prediction. See the [hERG first-paper implementation report](research/reports/platform/herg_paper/lead/HERG_FIRST_PAPER_IMPLEMENTATION_REPORT.md), [final reconciliation and verification](research/reports/platform/herg_paper/lead/HERG_FINAL_RECONCILIATION_2026_08_07.md), [established data/design advantages](docs/platform/herg_established_advantages.md), and [original hierarchy report](research/reports/platform/herg_hierarchy/lead/HERG_HIERARCHY_BUILD_REPORT.md). The CPU results are diagnostic floors, not production, clinical, or predictive-superiority claims.

The frozen [final mechanical verification report](research/reports/final_verification/platform_final_artifact_verification.json) passes artifact integrity and direct source rebinding, while explicitly recording `scientific_task_claim_ready: false` and substantive large-model training as neither ready nor authorized. Mechanical reproducibility is not scientific, clinical, rights, compute, or release approval.

The expanded evidence model keeps several axes orthogonal: experimental versus derived evidence, preclinical versus clinical context, public versus restricted access, and accepted versus quarantined data quality. In particular, ChEMBL molecule-development metadata is not treated as a clinical outcome; an absent hERG/QT report is not a negative safety label; and derived binding free energy is opt-in sensitivity evidence rather than a source-reported measurement.

## What the preserved Menin pipeline does

- Collects target-anchored activity data from ChEMBL, BindingDB, and PubChem BioAssay.
- Retains raw source fields and assigns stable measurement and structure identifiers.
- Uses RDKit to clean structures, select the parent fragment, neutralize charge, canonicalize SMILES, and generate InChIKeys without silently canonicalizing tautomers.
- Converts supported concentration units to nM; unknown or missing units remain unresolved and are quarantined rather than assumed to be nM.
- Preserves exact, approximate, and censored relations and their corresponding bounds.
- Enforces target, endpoint, ChEMBL validity, duplicate, assay-variant, structure, and unit quality gates.
- Links conservative exact cross-source mirror candidates without removing measurement rows or same-source replicates, and runs a no-collapse sensitivity for the primary Menin task.
- Keeps Menin and hERG summaries stratified by endpoint and assay family.
- Trains an endpoint/assay-specific Menin regression matrix and a primary hERG `IC50 × electrophysiology_functional` classifier, with a separately labeled pooled hERG sensitivity analysis, under compound-grouped random, Bemis–Murcko scaffold, chemical-cluster, or temporal splits.
- Reports scaffold-group bootstrap confidence intervals, regression conformal intervals, probability calibration, and nearest-neighbor applicability-domain diagnostics.
- Runs a primary-task-only chemical-intelligence stage with descriptors, QED, apparent ligand efficiency/LLE, medicinal-chemistry alerts, Bemis–Murcko series, deterministic Butina clusters, achiral/chiral Morgan novelty, fingerprint activity cliffs, conservative single-cut matched molecular pairs, connectivity-variant review, applicability-aware hERG evidence, transparent tiers/Pareto/sensitivity outputs, approved-reference coverage, and a configurable diversity-capped prospective experiment template.
- Writes row-level quality findings, exclusion summaries, split assignments, model manifests, and a six-manifest SHA-256 DAG for raw data, processed data, software, models, analysis, and reports when chemical intelligence is enabled.
- Provides a separate offline CSV/TSV/SDF intake library that validates private assay/endpoint registries and cohort roles, pseudonymizes source identifiers with a runtime-only HMAC key, quarantines conflicts, and writes deterministic private outputs.

The fixed target anchors are Menin `CHEMBL1615381` / UniProt `O00255` and hERG `CHEMBL240` / UniProt `Q12809`. Text search supplements these identifiers; it does not replace target verification.

## Confidential and proprietary data

Do **not** place Wang lab or other non-public data in this repository or send it to an external service by default. `menin_discovery.internal_data` is an implemented offline validation/pseudonymization boundary, but it is not wired into the public pipeline and does not itself grant authorization or secure the surrounding environment. The public CLI, including `analyze`, consumes only the configured public processed/model/report artifacts; it is not a confidential-data upload or intake path. Keep private inputs and outputs in access-controlled storage outside the Git working tree, confirm the governing agreement and release policy, and follow [the proprietary-data intake protocol](docs/proprietary_data_intake.md). The checked-in [`pipeline/config/internal_data_template.yaml`](pipeline/config/internal_data_template.yaml) contains schema placeholders only; copy and customize it in approved private storage. Every accepted internal row must be assigned a governed cohort role: `development`, `locked_external`, or `prospective_blind`. Run these as separate approved private builds: development data may support fitting and exploratory chemical intelligence; locked-external data are evaluation-only; and prospective-blind labels stay sealed until the model, thresholds, and analysis plan are frozen. Neither locked nor prospective data may influence training, calibration, feature selection, threshold selection, curation policy, or prioritization policy. Structures and values remain sensitive even after source identifiers are pseudonymized; models, analysis tables, plots, manifests, and logs derived from internal data require the same review.

## Quick start

Python 3.10 or newer is required. For an editable development environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Run the tests, then rebuild from the existing raw snapshot without network access:

```bash
pytest
menin-pipeline --stage all --skip-network
menin-pipeline --stage verify
```

To refresh public sources and run all stages:

```bash
menin-pipeline --stage all
```

To rebuild chemical intelligence after a compatible model build, then refresh its report surface:

```bash
menin-pipeline --stage analyze --skip-network
menin-pipeline --stage report --skip-network
menin-pipeline --stage manifest --skip-network
menin-pipeline --stage verify --skip-network
```

Use a custom policy file or override the primary evaluation split:

```bash
menin-pipeline --config pipeline/config/pipeline.yaml --stage all --skip-network \
  --split-strategy scaffold --menin-endpoint IC50 \
  --menin-assay-family biochemical_binding
```

The default Menin task is `IC50 × biochemical_binding`, the best-supported stratum with direct assay context and public source-year coverage in the current snapshot; additional endpoint/assay-family tasks run only when their eligible compound support meets the configured minimum. Public temporal evidence mixes ChEMBL/BindingDB document years with PubChem assay-deposit years and is therefore a sensitivity analysis, not assay chronology. The headline hERG task is restricted to `IC50 × electrophysiology_functional`, while the broader pooled endpoint/assay result is emitted separately as a sensitivity analysis rather than presented as the primary safety estimate.

`--fast` reduces computational work for smoke testing and writes model, analysis, and report artifacts under isolated `research/models/smoke/`, `research/analysis/smoke/`, and `research/reports/smoke/` roots; its metrics are not release results and are excluded from release manifests. It still reads the configured raw and processed roots. Public collection builds a complete all-source snapshot in staging before promotion, curation promotes the completed processed directory with rollback, and model/analysis/report builds replace their release-owned artifacts only after staged execution succeeds. External databases still change over time, so preserve the generated manifests and source status metadata for every reported build.

## Adjustable stages

| Stage | Purpose |
| --- | --- |
| `collect` | Refresh raw public source files and collection metadata. |
| `curate` | Normalize measurements, standardize structures, aggregate eligible tasks, and write quarantines. |
| `quality` | Audit schemas, units, ranges, identifiers, targets, assays, duplicates, and conflicts; `all` fails closed on eligible-table errors. |
| `models` | Train and evaluate Menin and hERG models under the configured split policy. |
| `analyze` | Build configured primary-task chemical-intelligence tables/JSON from curated data and verified model evidence; atomically promote the `research/analysis/` root. The report stage renders the corresponding figures. |
| `report` | Create the analytical summary, publication tables, and figures. |
| `manifest` | Create linked content-addressed manifests for raw data, processed data, software, models, analysis, and reports when analysis is enabled. |
| `verify` | Require and verify all six stage manifests, file hashes, counts, schemas, build IDs, declared release scope, model/analysis/report lineage, and upstream digest links when analysis is enabled. |
| `all` | Run the applicable stages in dependency order. |

The main policy surface is [`pipeline/config/pipeline.yaml`](pipeline/config/pipeline.yaml). See [reproducibility](docs/reproducibility.md) before changing defaults or comparing runs.

## Evidence layers and outputs

| Layer | Representative outputs |
| --- | --- |
| Raw observations | `research/data/raw/chembl/`, `research/data/raw/bindingdb/`, `research/data/raw/pubchem/` |
| Normalized measurements | `research/data/processed/menin_activity_measurements.csv`, `herg_activity_measurements.csv` |
| Excluded records | `research/data/processed/menin_activity_quarantine.csv`, `herg_activity_quarantine.csv` |
| Modeling tables | `research/data/processed/menin_compounds_curated.csv`, `herg_compounds_curated.csv` |
| Large hERG v1 | Assay-native ledger, 339,373-structure AID-only binary backbone, quantitative pIC50, and unified evidence tiers under `research/data/platform/processed/herg_hierarchy/` |
| Large hERG model-ready split | Fixed scaffold-grouped table and QC manifest under `research/data/platform/processed/herg_hierarchy/v1_model_ready/` |
| hERG first-paper v1.3 | Joined observation/structure/assay/protocol master tables, quality tasks, modality/QT ontology, and descriptive analyses under `research/data/platform/processed/herg_hierarchy/{v1_2_*,v1_3_*}/` |
| hERG pre-HPC v1.4 | Evaluation-candidate, replicate-conflict and protocol-priority queues plus the original label-blind benchmark under `research/data/platform/processed/herg_hierarchy/v1_4_*`; candidates are not adjudicated gold labels |
| hERG pre-HPC v1.5 | Automated candidate/lineage evidence, seven harder label-blind challenges, WT Q12809 sequence reference, split-contained matched-pair analysis, QT/exposure collection templates, and an executable HPC-readiness preflight under `research/data/platform/processed/herg_hierarchy/v1_5_*`; zero gold promotions, margins, training labels, model training, production feature generation, or HPC jobs |
| Observed ADMET | Analysis-ready rows in `research/data/processed/pk_admet_observations.csv`; complete classified inventory and quarantine in the adjacent `*_all.csv` and `*_quarantine.csv` files |
| Quality evidence | `research/data/processed/data_quality_summary.csv`, `research/reports/quality/` |
| Model evidence | Metrics, candidate comparisons, split assignments, and test predictions under `research/reports/` and its evaluation subdirectories |
| Predictions | `research/reports/menin_with_predicted_herg_risk.csv` |
| Chemical intelligence | Primary-task profiles, series/clusters, novelty, cliffs, matched pairs, connectivity variants, evidence-aware tiers/Pareto frontiers, data gaps, sensitivity traces, approved-reference coverage, and a pre-experiment selection template under `research/analysis/`; selected tables/figures are mirrored into `research/reports/` |
| Reproducibility | Per-model manifests plus `research/reports/manifests/{raw,processed,software,models,analysis,reports}_manifest.json` and `research/reports/verification/` when analysis is enabled |
| Human-readable synthesis | `research/reports/publication_summary.md`, `research/reports/summary.md`, and generated figures/tables |

Generated row counts and metrics belong in the versioned report and manifests, not in this README: rerunning collection or changing curation policy changes the analytical population.

Chemical priorities, the approved-reference coverage table, and the prospective-selection plan are research evidence surfaces, not validated leads. Missing or out-of-domain hERG evidence is unknown and receives no safety credit; PK/ADMET contributes coverage only. A run with no balanced priority-1 candidates is a valid outcome, not a pipeline error. The prospective plan is a public-data experiment-design template to freeze before testing, not prospective validation or a claim of efficacy, safety, availability, or freedom to operate.

## Repository guide

- [`docs/platform/README.md`](docs/platform/README.md): scientific-audit index for the expanded platform.
- [`docs/project/pretraining_readiness_ledger.md`](docs/project/pretraining_readiness_ledger.md): live workstream ledger, accepted evidence, checksums, blockers, and readiness boundary.
- [`docs/platform/evidence_and_endpoint_ontology.md`](docs/platform/evidence_and_endpoint_ontology.md): orthogonal evidence, endpoint, quality, access, and result-status semantics.
- [`docs/platform/claim_boundaries.md`](docs/platform/claim_boundaries.md): allowed interpretations and prohibited clinical/model claims.
- [`docs/repository_structure.md`](docs/repository_structure.md): canonical file hierarchy, naming, and cleanup rules.
- [`docs/architecture.md`](docs/architecture.md): stages, module boundaries, and extension points.
- [`docs/methodology.md`](docs/methodology.md): curation, modeling, uncertainty, and validation methods.
- [`research/reports/platform/herg_hierarchy/lead/HERG_HIERARCHY_BUILD_REPORT.md`](research/reports/platform/herg_hierarchy/lead/HERG_HIERARCHY_BUILD_REPORT.md): current large hERG counts, tier semantics, model-ready split, verification, and training sequence.
- [`docs/platform/herg_established_advantages.md`](docs/platform/herg_established_advantages.md): the project's established data/design superiorities, unestablished predictive claims, and the evidence required for a future matched-comparator superiority claim.
- [`research/reports/platform/herg_paper/lead/HERG_FIRST_PAPER_IMPLEMENTATION_REPORT.md`](research/reports/platform/herg_paper/lead/HERG_FIRST_PAPER_IMPLEMENTATION_REPORT.md): wild-type scope, quality tasks, measurement-method/QT layers, model comparison protocol, CPU baseline, and paper experiment sequence.
- [`docs/data_dictionary.md`](docs/data_dictionary.md): processed and report schemas.
- [`docs/reproducibility.md`](docs/reproducibility.md): environments, configuration, manifests, and release reconstruction.
- [`docs/limitations.md`](docs/limitations.md): current claim boundaries and unresolved risks.
- [`docs/external_sources.md`](docs/external_sources.md): databases, software, and methodological references.
- [`docs/publication_checklist.md`](docs/publication_checklist.md): analysis and release gates.
- [`docs/proprietary_data_intake.md`](docs/proprietary_data_intake.md): controlled intake of non-public lab data.
- [`docs/herg_benchmark.md`](docs/herg_benchmark.md): configurable private/public hERG model search and scoring.
- [`docs/pk_herg_research_program.md`](docs/pk_herg_research_program.md): separate large-molecule rat PK/hERG causal-modeling, fast-physics, HPC-bundle, assay-panel, and optimizer-contract workflow. Run it with `menin-research --config pipeline/config/pk_herg_research.yaml --stage all-local`; it does not execute or rewrite the Menin/Menin-Edit baseline.
- [`docs/project/contributing.md`](docs/project/contributing.md): development and data-change expectations.

## Citation, third-party data, and license status

Use [`CITATION.cff`](CITATION.cff) to cite the software repository and cite each upstream database used in a particular build. [`docs/data_source_notice.md`](docs/data_source_notice.md) explains third-party data responsibilities.

No repository software license has been selected. This is intentional: a license must be approved by the rights holder, and third-party source data retain their own terms. See [`docs/licensing.md`](docs/licensing.md) before redistribution or reuse.
