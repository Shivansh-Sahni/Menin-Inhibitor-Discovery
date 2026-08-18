# Publication and reproducibility readiness checklist

Final snapshot: 2026-08-04. `passed` means the frozen local artifact was
physically or executably verified. `partial` means useful evidence exists but a
required scientific or release component is absent. `blocked` requires a human,
new source, model decision, hardware, or external study.

This checklist is a post-gate reconciliation; it does not rewrite the immutable
final verification report.

## A. Governance, rights, and release

| ID | Requirement | Final status | Evidence or remaining action |
|---|---|---|---|
| A01 | Named data owner and release approver | `blocked_human` | Record owner, scope, date, and release approval. |
| A02 | Repository code license | `blocked_human` | No `LICENSE`; select and install an approved license. |
| A03 | Documentation/figure/content license | `blocked_human` | Select content license and notices. |
| A04 | Per-source and model-weight terms | `partial` | Source identities/terms are recorded; approve conditional redistribution and exact future checkpoint terms. |
| A05 | Public build separated from internal/mixed-access inputs | `passed_mechanical` | Platform roots, dependency binding, and generated-artifact integrity pass; ignored local research remains outside the public build. |
| A06 | Staged secrets/PII/private-correspondence disclosure review | `blocked_human` | Review the complete staged content, not only ignore rules. |
| A07 | Large-artifact storage and redistribution policy | `blocked_human` | Select content-addressed storage, retention, and redistributable scope. |
| A08 | Clean committed release candidate | `not_passed` | Dirty migration and untracked surfaces remain; review the exact staged add/move/delete set. |

## B. Sources, provenance, and canonical admission

| ID | Requirement | Final status | Evidence or remaining action |
|---|---|---|---|
| B01 | ChEMBL 37 archive/export identity and integrity | `passed` | Complete archive/database, 162 explicit-schema Parquets, manifests, and transaction receipt verified. |
| B02 | Five external raw source bundles | `passed_acquisition` | 404 declared source artifacts and 853 recursive bundle artifacts verified across BindingDB, UniProt, ClinicalTrials.gov, Drugs@FDA, and DailyMed. |
| B03 | External normalized evidence | `passed_normalization` | Nine artifacts, 60,377,234 bytes, 464,123 Parquet rows, and five inputs semantically rebound. |
| B04 | External canonical observation/label admission | `blocked_rights_science` | Exactly zero observations and zero labels admitted; resolve rights, linkage, duplicates/conflicts, ambiguity, and outcome semantics. |
| B05 | Stable source-row lineage | `passed_for_admitted_chembl` | 3,938,372 observations and matching lineage rows; source and task registries bound. |
| B06 | Cross-source mirror/origin handling | `passed_candidate_disposition` | BindingDB separates 93,023 curated candidates, 429 ChEMBL mirrors, and 260 Taylor-rights quarantines. No multisource claim is inferred. |
| B07 | Clinical/regulatory record preservation | `passed_inventory_only` | Registry/regulatory archives are preserved and normalized without creating molecule-level outcomes or negatives. |
| B08 | Genuine clinical/PK/QT/QTc/outcome corpus | `blocked_external` | Requires identity linkage, reported results extraction, clinical semantics, and human review. |

## C. Schema, identity, labels, and QC

| ID | Requirement | Final status | Evidence or remaining action |
|---|---|---|---|
| C01 | Canonical molecule/protein/construct/assay/observation schema | `passed` | Machine schema, dictionary, ontology, Arrow fingerprints, and allowed values bind the accepted corpus. |
| C02 | Original and normalized values/relations/units preserved | `passed` | Source values, standardization, relation/censoring, and derivation lineage remain separate. |
| C03 | Endpoint families not silently pooled | `passed` | Kd, Ki, IC50, EC50, hERG function/binding, PK/ADME, QT, and derived free energy remain distinct task signatures. |
| C04 | Predictions prohibited as labels | `passed` | Schema/build/QC and adversarial tests fail closed. |
| C05 | Canonical QC bound to exact build | `passed` | A/B manifests and QC reports are content-equivalent across 260 components and 1,154,513,167 bytes. |
| C06 | Duplicate/repeat/conflict accounting | `passed_descriptive` | Repeated groups and conflicts are reported without collapsing meaningful measurements. Human high-impact adjudication remains later task work. |
| C07 | Missingness and attrition | `passed_descriptive` | Real missingness, attrition, exclusions, compatible strata, and denominators generated twice identically; no target imputation. |
| C08 | Task rows and exclusions reconcile | `passed` | 28 task datasets; default 2,548,252 candidates→2,548,198 rows, sensitivity 67,848→67,839; exclusions are reasoned missing-sequence cases. |

## D. Statistics, bias, splits, and leakage

| ID | Requirement | Final status | Evidence or remaining action |
|---|---|---|---|
| D01 | Real descriptive census | `passed` | Statistical A/B source-reverified and byte-identical: 19 files and 1,374,306 bytes. |
| D02 | Multiplicity and inferential boundaries | `passed_for_executed_panel` | BH policy applied; only adequately supported QT decade×evidence-stage panel tested; explicitly descriptive/noncausal. |
| D03 | Exact split leakage audit | `passed` | Exhaustive disk-backed exact overlap for every materialized split. |
| D04 | Split determinism and source rebind | `passed` | A/B both directly source-reverified; 480 identical files, 225 dirs, 564,604,068 bytes, distinct inodes. |
| D05 | Strategy accounting | `passed` | 28×6 decisions: 85 materialized, 83 reasoned skips; fixed seed and no seed search. |
| D06 | RDKit scaffold failures | `passed_fail_closed` | 214 exception-proxy rows across 12 task views; affected scaffold candidates skipped, never represented as true scaffolds. |
| D07 | Near ligand/protein similarity and intended-use thresholds | `partial` | Deterministic capped sample exists; exhaustive/indexed audit or scientific acceptance of thresholds remains required. |
| D08 | Bias/fairness claims | `partial_descriptive_only` | Concentration, attrition, missingness, temporal/stage and subgroup support exist; no model fairness or clinical-equity result. |

## E. Model-ready interfaces and bounded diagnostics

| ID | Requirement | Final status | Evidence or remaining action |
|---|---|---|---|
| E01 | Label-blind feature routing | `passed` | All canonical label columns excluded from split feature projections. |
| E02 | Corpus integration | `passed_bounded` | 22 tasks integrated; six support-skipped; 524 components source-reverified. |
| E03 | Test lockbox isolation | `passed` | Test lockboxes were neither opened nor hashed after routing; final scanner opened no routed test payload. |
| E04 | Censoring/derived-label handling | `passed_interface` | Unsupported censored diagnostics are skipped, not midpoint-imputed; derived tasks require sensitivity scope. |
| E05 | Capped development diagnostics | `passed_bounded` | Eleven completed, 11 explicit diagnostic skips; caps 10,000 train/2,500 validation. Not final performance. |
| E06 | Metrics and error artifacts | `passed_interface` | Registry/tests and completed diagnostic prediction/error contracts pass; no headline sealed-test result. |
| E07 | Expensive embeddings/foundation-model execution | `not_run_by_design` | Candidate interfaces exist; no expensive embedding corpus or substantive training. |
| E08 | External/prospective model validation | `blocked_external` | Required before translational, safety, or product claims. |

## F. Software and deterministic reconstruction

| ID | Requirement | Final status | Evidence or remaining action |
|---|---|---|---|
| F01 | Final physical artifact verifier | `passed` | Report SHA `f27ceb…`, 20,573 bytes; artifact/source integrity true. |
| F02 | Pipeline tests | `passed` | 579 passed, 64% coverage; one harmless joblib physical-core warning. |
| F03 | Menin-Edit tests | `passed` | 53 passed; CI installs `[lab,dev]`. |
| F04 | Lint/format/types | `passed` | Ruff pass; 177 files formatted; mypy no-incremental pass over 80 source files. |
| F05 | Dependency consistency | `passed_core` | `pip check` passes; exact 53-pin core lock has zero known vulnerabilities. |
| F06 | Expanded local environment hygiene | `not_passed` | Unused `aiohttp==3.14.1` has three 2026 advisories; remove or upgrade to >=3.14.3. |
| F07 | Storage/path/link integrity | `passed` | 2,117 generated regular files; zero hardlink/inode aliases; portable paths and special-file guards pass. |
| F08 | Independent clean-clone reproduction | `blocked_release` | Exact commands are frozen, but a second clean committed checkout has not been run. |

## G. Final training and claim gates

| ID | Requirement | Final status | Evidence or remaining action |
|---|---|---|---|
| G01 | Exact intended task and claim | `blocked_human` | Freeze population, endpoint, label policy, support, and leakage thresholds. |
| G02 | Exact checkpoint and tokenizer/config/weight hashes | `blocked_human_model` | Select revision, license, cutoff, and corpus-overlap audit. |
| G03 | Meeting-referenced ~100k-structure model | `blocked_external` | Identify exact paper/model and review its endpoint, corpus, overlap, terms, and evidence. |
| G04 | Approved compute and responsible operations | `blocked_human_hardware` | Accelerator allocation, measured budget, monitoring, resume/failure, retention, and responsible-use approval. |
| G05 | Capped approved-hardware dry run | `not_run` | May occur only after G01–G04; must preserve sealed test. |
| G06 | Substantive training authorization | `false` | Final verifier explicitly reports not ready and not authorized. |
| G07 | Substantive training actions | `none` | Final report records `training_actions=[]`; both training flags false. |
| G08 | Independent scientist/owner/license sign-off | `blocked_human` | Required before training and before any paper/product headline claim. |

## Release rule

The frozen artifacts are mechanically verified, but any `blocked`,
`not_passed`, or claim-critical `partial` item prevents an umbrella
publication-ready, product-ready, or substantive-pretraining-ready claim. A
human waiver can acknowledge residual operational risk; it cannot manufacture
missing rights, data, external validity, or scientific evidence.
