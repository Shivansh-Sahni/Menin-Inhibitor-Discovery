# Audit evidence manifest and handoff

Audit agent: `scientific_audit`. Review date: 2026-08-04. Write scope was restricted to `docs/platform/` and `research/reports/platform/audit/`; no code, configuration, root documentation, data-foundation artifact, model-readiness artifact, or shared ledger was modified by this workstream.

## Files created

1. `docs/platform/README.md`
2. `docs/platform/evidence_and_endpoint_ontology.md`
3. `docs/platform/claim_boundaries.md`
4. `docs/platform/source_and_model_review.md`
5. `research/reports/platform/audit/repository_inventory.md`
6. `research/reports/platform/audit/implemented_documented_discrepancies.csv`
7. `research/reports/platform/audit/bias_missingness_selection_plan.md`
8. `research/reports/platform/audit/statistical_analysis_plan.md`
9. `research/reports/platform/audit/sensitivity_ablation_plan.md`
10. `research/reports/platform/audit/publication_reproducibility_checklist.md`
11. `research/reports/platform/audit/risk_register.csv`
12. `research/reports/platform/audit/readiness_rubric.md`
13. `research/reports/platform/audit/audit_evidence_manifest.md`
14. `research/reports/platform/audit/data_handoff_acceptance_checklist.md`
15. `research/reports/platform/audit/required_source_acquisition_acceptance.csv`
16. `research/reports/platform/audit/final_cross_workstream_verification.md`
17. `research/reports/platform/audit/independent_validation_results.json`
18. `research/reports/platform/audit/final_25_section_status_matrix.md`
19. `research/reports/platform/audit/final_reproduction_and_next_steps.md`

## Local evidence inspected

- governing attached 25-section project specification and shared task ledger;
- git history, diff/status, tracked/untracked/ignored inventories, file counts/sizes, and migration path/byte comparisons;
- `.gitignore`, `.gitattributes`, CI, pre-commit, package/environment/lock files, Makefile, README, architecture, reproducibility, methodology, limitations, licensing, source notice, publication checklist, and repository-structure documents;
- raw collection metadata and raw/processed/software/model/analysis/report manifest surfaces;
- processed build/source/eligibility/QC/missingness summaries;
- model manifests, metrics, predictions, split/overlap/applicability/calibration artifacts, and public summary reports;
- PK/hERG program configuration and non-sensitive reviewer conclusions without reproducing internal rows/identities;
- simulation/HPC plans and execution-policy settings;
- current manifest verification run metadata.

## File movement, quarantine, and deletion accounting

This audit workstream created/updated only its owned documentation and audit
records. It moved, archived, or deleted no source, data, model, code,
configuration, ledger, or report-bound artifact.

The integrated DATA workstream retained failed/interrupted canonical attempts
as recoverable evidence under:

- `research/data/platform/quarantine/full_chembl37_failed_attempt_1/`;
- `research/data/platform/quarantine/full_chembl37_failed_attempt_2_numeric_dtype_collision/`;
- `research/data/platform/quarantine/full_chembl37_failed_attempt_3_arrow_schema_unification/`;
- `research/data/platform/quarantine/full_chembl37_interrupted_attempt_4_determinism_b_audit_stop/`; and
- `research/data/platform/quarantine/full_chembl37_interrupted_attempt_4_primary_audit_stop/`.

Invalid split reruns were stopped before publication; only their private
staging trees were removed. Final inspection found no split `.building` tree.
Raw sources, promoted canonical A/B, statistical A/B, split A/B, corpus
readiness, dependency audit, and immutable final report were not overwritten by
the post-gate reconciliation.

## Authoritative external sources reviewed

- [ChEMBL](https://www.ebi.ac.uk/chembl/)
- [BindingDB downloads](https://www.bindingdb.org/rwd/bind/chemsearch/marvin/Download.jsp?all_download=yes) and [2024 primary paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC11701568/)
- [PubChem citation guidance](https://pubchem.ncbi.nlm.nih.gov/docs/citation-guidelines) and [assay update/revocation guidance](https://pubchem.ncbi.nlm.nih.gov/docs/update-or-revoke-bioassays)
- [UniProt release 2026_02](https://www.uniprot.org/release-notes/2026-06-10-release) and [license](https://www.uniprot.org/help/license)
- [RCSB PDB policies](https://www.rcsb.org/pages/policies)
- [PLINDER documentation](https://plinder-org.github.io/plinder/tutorial/dataset.html) and [repository](https://github.com/plinder-org/plinder)
- [PK-DB](https://pk-db.com/), [paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC7779054/), and [API](https://pk-db.com/api/v1/swagger/)
- [EPA CompTox release notes](https://www.epa.gov/comptox-tools/comptox-chemicals-dashboard-release-notes), [ToxCast downloads](https://www.epa.gov/comptox-tools/exploring-toxcast-data), and [data/API terms](https://www.epa.gov/comptox-tools/comptox-data-and-apis)
- [ClinicalTrials.gov API](https://clinicaltrials.gov/data-api/api)
- [Drugs@FDA data files](https://www.fda.gov/drugs/drug-approvals-and-databases/drugsfda-data-files), [openFDA](https://open.fda.gov/apis/drug/drugsfda/), and [DailyMed API](https://dailymed.nlm.nih.gov/dailymed/app-support-web-services.cfm)
- [Open Targets 26.03 announcement](https://blog.opentargets.org/open-targets-platform-26-03-has-been-released/) and [data access](https://platform-docs.opentargets.org/data-access)
- [ICH E14/S7B Q&As](https://database.ich.org/sites/default/files/E14-S7B_QAs_Step4_2022_0221.pdf)
- [Sun, Wang & Shen hERG paper](https://pubs.acs.org/doi/10.1021/acs.jcim.6c00163)
- [AlphaFold 3 repository](https://github.com/google-deepmind/alphafold3) and [weight terms](https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md)
- [Chai-1 repository](https://github.com/chaidiscovery/chai-lab)
- [Boltz repository](https://github.com/jwohlwend/boltz)
- [Nesso-1 model card](https://huggingface.co/recursionpharma/nesso), [repository](https://github.com/recursionpharma/nesso), and [technical report](https://www.valencelabs.com/wp-content/uploads/2026/07/nesso1.pdf)
- [DrugBank academic access](https://go.drugbank.com/academic_research) and [terms](https://trust.drugbank.com/drugbank-trust-center/terms-of-use)

## Independent executable and physical checks

The audit did not rely only on workstream reports. Frozen checks completed to
date include:

- all six frozen MODEL workstream test files: 110 tests passed; the full MODEL
  source surface compiled and passed Ruff and mypy;
- eight explicit adversarial DATA cases passed: failed QC does not promote,
  reused final data requires fresh manifest-bound QC, specialized summary/child
  drift is rejected, component tamper and unlisted files are rejected, stage
  counts reconcile, a source/snapshot mismatch beyond 100,000 rows is detected,
  and task/observation provenance drift is rejected;
- the DATA schema/pipeline/bulk/canonical/QC surface passed 53 focused tests,
  Ruff, and mypy after the numeric-identity repair; the later null-only Arrow
  schema and path-integrity repair passed 44 combined canonical/QC/determinism
  tests plus Ruff/format/mypy, while a second auditor independently replayed 25
  focused cases including dirty/control paths and dangling staging symlinks.
  Exact accepted hashes are recorded in `independent_validation_results.json`;
- all artifacts listed by `pretraining_static_manifest.json` matched their
  SHA-256 digests, and its substantive-training flag was false;
- the synthetic 100,003-row bounded-memory evidence reconciled exactly to
  69,801 train, 15,130 validation, and 15,072 test rows; its 32-example loader
  cap and non-performance/no-training declarations were present;
- dependency name/specifier sets matched across the project, requirements and
  environment declarations; every frozen lock version satisfied those bounds,
  and `pip check` reported no broken requirements;
- an independent Parquet-footer review invalidated the first full and
  specialized ChEMBL export manifests despite their correct hashes/counts: the
  123 full parts formed 53 Arrow schema groups, and multipart specialized/
  development families also drifted under null/integer/floating inference. All
  earlier export-summary digests were withdrawn from acceptance;
- the superseding schema transaction was checked over exactly 162 Parquets
  (123 full, 23 specialized activity, 15 development, one target component).
  Exact recursive membership, every hash/byte/footer/schema, source attribution,
  order, counts, child/summary/receipt bindings, and physical overlaps pass;
- all five external raw bundles were recursively reverified: 404 declared
  source artifacts totaling 18,913,867,042 bytes and 853 bundle artifacts
  totaling 18,994,847,039 bytes, with ten archive inventories and zero
  canonical rows or labels;
- the promoted external normalization was independently streamed and checked:
  nine artifacts, 60,377,234 bytes, and 464,123 Parquet rows. Every one of
  93,712 BindingDB 640-field source maps was re-parsed/rehashed, all 95,506
  candidate endpoint cells remained endpoint-separated, all 13,976 UniProt
  sequences were rehashed, cohort and quarantine counts reconciled, release-
  path privacy passed, and zero canonical-observation/label/training flags held;
- canonical A/B were independently reopened and accepted as content-equivalent:
  260 components and 1,154,513,167 bytes; A/B manifests `1ace39…`/`85bcbe…`,
  QC `d481c8…`/`acc14b…`, and determinism report `ab5fc4…`;
- statistical A/B are source-reverified and byte-identical: 19 files,
  1,374,306 bytes, 18 declared artifacts per build, and manifest `5ee445…`;
- the first real split executions exposed an RDKit bad-bond-stereo failure. An
  initial repair was independently rejected because its exception proxy still
  appeared scaffold-applicable. The final centralized fail-closed repair passed
  real-SMILES direct/streaming replay, 48 focused tests, Ruff, format, and mypy;
- accepted split A/B are independently generated, directly source-reverified,
  and byte-identical: 480 files, 225 directories, 564,604,068 bytes, acceptance
  `453936…`, and 85 materialized/83 skipped strategies. The audit rehashed every
  inventory component and all 113 Parquet footers/schemas in each tree;
- corpus readiness acceptance `386396…` binds 524 components, integrates 22 of
  28 tasks, support-skips six without seed search, completes 11 capped
  train/validation diagnostics, explicitly skips 11, and neither opens nor
  hashes test lockboxes;
- the final physical report
  `research/reports/final_verification/platform_final_artifact_verification.json`
  is 20,573 bytes with SHA-256 `f27ceb…`; mechanical verification and source
  rebinding pass while scientific-task readiness, substantive-training
  readiness, and authorization remain false;
- the final gate checked 2,117 generated regular files with zero inode aliases
  and scanned 1,090 JSON documents without opening routed test payloads;
- final software evidence is 579 pipeline tests, 53 Menin-Edit tests, Ruff,
  177-file format check, no-incremental mypy over 80 source files, `pip check`,
  and a 53-pin core audit with zero known vulnerabilities. The expanded local
  environment separately retains unused `aiohttp==3.14.1` with three 2026
  advisories and is not called release-clean.

The machine final report predates the authorized post-gate update of
`independent_validation_results.json`; that JSON explicitly records the timing
boundary. No report-bound artifact, dependency audit, code, configuration,
data, split, model, ledger, or final report was modified by this audit
reconciliation.

## Assumptions

- The product remains research decision support and is not intended for clinical use.
- “Complete” means complete evidence accounting and honest statuses, not that unavailable/license/hardware/human-blocked work can be fabricated.
- The DATA workstream uses the real frozen ChEMBL 37 source/export for its
  canonical artifact. The later 22-task corpus-readiness bundle is real but its
  diagnostics remain bounded engineering evidence, not final performance.
- Existing mixed-access PK/hERG trees are local-only; the public platform build is separate.
- No external model is accepted as ground truth; all are evaluation candidates subject to endpoint, overlap, terms, and hardware review.
- The meeting’s “approximately 100,000 structures” model remains unidentified until a human supplies the exact paper/model.

## Final blockers

The final report lists eight conjunctive blockers:

1. external evidence canonical admission and genuine clinical/PK/QT/QTc/outcome
   extraction, linkage, rights, ambiguity, duplicate, and conflict resolution;
2. public-release hygiene, staged disclosure, and artifact storage;
3. repository and conditional source/model redistribution licenses;
4. a frozen task, intended use, support, and accepted leakage thresholds;
5. exact large-model checkpoint, terms, cutoff, and corpus-overlap audit;
6. authorized compute, monitoring, budget, resume, failure, and responsible-use
   operations;
7. independent external/prospective validation; and
8. identification of the meeting-referenced approximately 100,000-structure
   model.

The dirty migration, missing `LICENSE`, absence of an independent clean-clone
run, and the expanded-local-environment `aiohttp` advisory also prevent a clean
public-release claim.

## Independently reconciled claims and residual reviews

1. All normative vocabulary values in the data schema exactly match `docs/platform/evidence_and_endpoint_ontology.md`; aliases are explicit and unknowns fail rather than silently coerce.
2. Predictions are rejected from observation/label builds and cannot enter labels through aliases or serialized fixtures.
3. Public platform artifacts have no dependency on ignored internal/mixed-access paths.
4. Source license, release, retrieval, query, citation, and row lineage fields survive raw-to-model-ready transformations.
5. Endpoint pooling and free-energy guards reject incompatible values.
6. Split artifacts are mutually exclusive and exact/near overlap audits use fitted-on-train thresholds and correct ligand/protein/source/time groups.
7. Feature transforms, vocabularies, normalizers, imputers, and samplers fit on training data only.
8. Baselines are diagnostic, use observed/curated labels only, and do not claim platform performance from tiny fixtures.
9. JSONL/collator smoke tests exercise masks, task IDs, missing modalities, deterministic ordering, and boundary lengths without substantive training.
10. Hardware/runtime/storage outputs are estimates unless measured and identify assumptions/hardware.
11. Final full tests/lint/types and focused platform tests pass on the reconciled tree. **Verified.**
12. Canonical, statistical, and split A/B match under their declared determinism contracts. **Verified locally; a clean committed clone remains outstanding.**
13. All bound platform manifests and the offline physical final verifier pass. **Verified for the frozen workspace.**
14. Final staged paths contain no ignored data, local outputs, caches, secrets, usernames, absolute personal paths, or unauthorized binaries. **Human staged review remains outstanding.**
15. The final rubric preserves the large-model-training verdict at NOT READY. **Verified.**

## Reproduction handoff

`final_reproduction_and_next_steps.md` now records the Python 3.13.7
environment boundary, editable installs, raw-to-interim ChEMBL rebuild,
external verification/normalization, canonical A/B, statistical A/B, split A/B,
corpus readiness, static artifacts, final verifier, full tests, Ruff, format,
mypy, `pip check`, dependency audit, exact accepted hashes, and next gates.
These commands reproduce the frozen snapshot from preserved raw bytes; a live
reacquisition is a new dataset version.

Substantive large-model training was not initiated by this audit workstream.
