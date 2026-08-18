# Source and external-model review

Review date: 2026-08-04. Status is source-specific. Web facts below are dated snapshots and must be refreshed at each release. This is a scientific/technical review, not legal advice.

## Decision labels

- **Admit**: suitable for a versioned initial public build if all row-level QC and attribution checks pass.
- **Conditional**: useful, but redistribution, identity, endpoint, or access review is required.
- **Evaluation only**: may be benchmarked without making it the source of truth.
- **Blocked**: do not ingest or distribute until the stated issue is resolved.
- **Future**: valuable expansion, not present in the audited local corpus.

## Locally observed source state

The legacy public snapshot was collected on 2026-07-14. Its metadata identifies ChEMBL 37 (release date 2026-05-01), 41,078 ChEMBL hERG rows, 1,776 target-filtered Menin rows, 8,202 Menin-associated molecule-activity rows, 323 PubChem assays, and two targeted BindingDB exports. These are locally evidenced counts, not platform-wide coverage claims.

The legacy BindingDB target exports still do not encode a recoverable monthly
release and remain unsuitable as reproducible platform evidence. They have now
been superseded for the platform acquisition layer by an immutable BindingDB
202608 package. Its independently verified manifest is
`5a7425f680e6820ea665212e051bd11a2ece2e03d916534456640bdd67297406`:
11 declared HTTP artifacts (44,266,829 bytes) and an exact recursive bundle of
25 non-manifest artifacts (44,280,360 bytes). The Articles file has 93,712
physical rows: 93,023 BindingDB-curated candidates, 429 ChEMBL mirrors excluded
from independent-source evidence, and 260 Taylor Research Group rows
quarantined pending origin/rights review. Acquisition does not constitute
canonical admission.

A targeted UniProtKB 2026_02 snapshot has also passed byte-level acquisition
verification. Manifest
`d4000c69e3c51ff1fbd7b68b0c32e78274fdbbdac40b6b1938f462fef816017a`
binds 152 declared source artifacts (661,506,168 bytes) and 307 recursive
bundle artifacts (686,337,253 bytes). Of 14,983 requested accessions, 14,980
resolved to one primary accession and three are ambiguity quarantines; 46
non-UniProt-syntax source identifiers are separately quarantined. Only 13,976
returned entries carry sequences. The 1,007 `Inactive` entries without
sequence/name/taxonomy/version remain prohibited from protein-input readiness.

The local PK/ADME table is a Menin-associated subset, not a general PK corpus.
The local public hERG task is a project-specific retrospective classifier, not
a QT or clinical-cardiotoxicity dataset. The final ClinicalTrials.gov v2
acquisition (`5b33bcc2…`) freezes 210,644 unique all-DRUG registry records and a
separate 3,879-record heuristic cardiac-safety cohort at API 2.0.5 / data
timestamp 2026-08-04T09:00:05. The latter contains 1,343 posted outcome modules
and 1,343 adverse-event modules, but its membership is text-heuristic,
unmapped, and unreviewed; no record or module was interpreted as a molecular
label. The official OpenAPI route returned an archived HTTP error and the terms
snapshot was an SPA shell, so human terms review remains open.

Drugs@FDA acquisition (`d995f330…`) freezes all 12 official tables: 959,447
parsed rows with one retained width anomaly, 16 blank declared primary-key
rows, and 6,906 source foreign-key orphan occurrences. It is regulatory/product
evidence only and has zero canonical admission. DailyMed acquisition
(`d29a97bb…`) freezes all six 2026-08-03 human-prescription archives: exactly
17,767,198,122 archive bytes and 54,672 members passed published MD5, SHA-256,
membership, member-hash, and CRC checks. SETID histories, XML section
extraction, product/molecule mapping, and label-fact adjudication remain
unperformed; raw archive completeness is not label-evidence admission.

The separately promoted external-normalization bundle is also independently
verified. Manifest
`4a9060a154049da0d02721bb66b7072b7cd73599c2502b17f36fa40752f3bc6c`
binds nine artifacts totaling 60,377,234 bytes and 464,123 Parquet rows. It
preserves the 93,712 BindingDB source rows and 95,506 endpoint-specific
candidate cells, the UniProt entry/resolution/source-membership inventories,
both ClinicalTrials.gov cohort memberships, and regulatory archive inventories.
It performs no cross-source molecular admission: the frozen flags remain zero
canonical observations, zero model labels, and no substantive training. A
clean-room rebuild was byte-identical; the final report's internal identity is
`fc326388709657e4ade9dd831c565716984f1e304dec6d6bdfdde22882d0d2e9`.

## Recommended data-source matrix

| Source | Current verified state | Scientific role | Rights/citation boundary | Decision |
|---|---|---|---|---|
| ChEMBL | ChEMBL 37, released 2026-05-01; official site reports 24,527,044 activities, 2,921,148 compounds, and 18,552 targets | Core activity, assay, molecule, target, and development metadata | CC BY-SA 3.0; preserve release DOI, attribution, source record, and share-alike review for distributed derivatives | **Admit**, row/version aware ([official ChEMBL](https://www.ebi.ac.uk/chembl/)) |
| BindingDB | Dated 202608 package is frozen; the normalized layer preserves 93,712 source rows and 95,506 endpoint-separated candidate cells with an exhaustive three-origin audit, but no row is canonically admitted | Core binding observations and assay text | BindingDB-curated rows CC BY 4.0; ChEMBL-curated rows remain CC BY-SA 3.0; Taylor rows require separate review. Retain curation-source field | **Acquisition/normalization accepted; admission conditional** on identity, rights, endpoint, mirror/conflict, and duplicate QC ([downloads](https://www.bindingdb.org/rwd/bind/chemsearch/marvin/Download.jsp?all_download=yes), [2024 paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC11701568/)) |
| PubChem BioAssay | Live depositor-integrating service; records can be updated or revoked | Assay discovery, source assertions, cross-identifiers, and selected contributor datasets | PubChem aggregation does not make every contributor record globally redistributable. Preserve depositor, AID/SID/CID, update/revocation state, retrieval time, and primary citation | **Conditional** by contributor ([citation guidance](https://pubchem.ncbi.nlm.nih.gov/docs/citation-guidelines), [update/revoke behavior](https://pubchem.ncbi.nlm.nih.gov/docs/update-or-revoke-bioassays)) |
| UniProtKB | Targeted release 2026_02 is frozen and normalized for 14,983 returned primaries; 13,976 sequence-bearing entries are candidates, while inactive, ambiguous, and syntax-invalid references remain quarantine | Canonical protein sequence/accession/version, isoforms, taxonomy, evidence annotations | Database copyrightable content CC BY 4.0; retain accession and entry/sequence versions. The saved license/release pages are identical SPA shells, so human terms review remains open | **Acquisition/normalization accepted; sequence admission conditional** on source reconciliation and quarantine disposition ([release notes](https://www.uniprot.org/release-notes/2026-06-10-release), [license](https://www.uniprot.org/help/license)) |
| wwPDB/RCSB PDB | Continuously updated structural archive | Experimental macromolecular structures, ligand/construct/experimental metadata | PDB archive data are CC0; cite each entry DOI and primary publication; integrated third-party annotations may have different terms | **Admit** with entry version/date ([RCSB policies](https://www.rcsb.org/pages/policies)) |
| PLINDER | Versioned protein-ligand systems, similarity matrices, clusters, and splits; full data are hundreds of GB | Structure curation, redundancy analysis, and leakage-resistant evaluation | PLINDER code/curated package is Apache-2.0, but underlying PDB and linked-source provenance/citations remain required | **Conditional** after storage budget and exact release/iteration freeze ([dataset tutorial](https://plinder-org.github.io/plinder/tutorial/dataset.html), [repository](https://github.com/plinder-org/plinder)) |
| PDBbind | Registered/download-restricted dataset with high benchmark reuse | Historical affinity benchmark only | The current BindingDB paper states PDBbind forbids redistribution without explicit permission; benchmark leakage is also extensive | **License blocked** unless explicit permission and leakage audit exist ([BindingDB review](https://academic.oup.com/nar/article/53/D1/D1633/7906836)) |
| Binding MOAD | The historic domain no longer presented an auditable scientific-download surface during this review | Potential curated structure-affinity evidence | Provenance, maintenance status, and current terms could not be established | **Unavailable**; do not ingest from mirrors until owner and rights are authenticated |
| PK-DB | Study-, subject-, intervention-, and concentration-time-oriented pharmacokinetic database with API | Rich public human PK candidate | Platform code is LGPL-3.0, while data access/redistribution terms require separate review; preserve study and subject grouping | **Conditional** after terms and ethics/privacy review ([PK-DB](https://pk-db.com/), [primary paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC7779054/), [API](https://pk-db.com/api/v1/swagger/)) |
| EPA CompTox/ToxCast | CompTox Dashboard 2.8 released 2026-03-17; latest ToxCast invitrodb is v4.3 (August 2025) | Additional public toxicology assays and chemical identities; not automatically clinical cardiac evidence | EPA states its computational toxicology data are open and free of copyright restrictions; retain assay/protocol/QC and release metadata | **Future**, endpoint-reviewed ([release notes](https://www.epa.gov/comptox-tools/comptox-chemicals-dashboard-release-notes), [ToxCast downloads](https://www.epa.gov/comptox-tools/exploring-toxcast-data), [data terms](https://www.epa.gov/comptox-tools/comptox-data-and-apis)) |
| Sun, Wang & Shen hERG supplement | JCIM article published 2026-06-18, DOI 10.1021/acs.jcim.6c00163; a workbook is present locally | External hERG data/model benchmark candidate | Article access does not by itself prove the workbook may be redistributed in a derived public training corpus. Assay/protocol harmonization and overlap audit are required | **License review required** ([article](https://pubs.acs.org/doi/10.1021/acs.jcim.6c00163)) |
| ClinicalTrials.gov | API 2.0.5/data timestamp 2026-08-04T09:00:05 is frozen and normalized as 210,644 all-DRUG memberships plus a separate 3,879-record cardiac-text heuristic; zero canonical observations or labels admitted | Trial registration and posted-results evidence | Preserve NCT ID, exact page/query/version lineage, intervention-mapping confidence, results-posted state, and source citations; archived OpenAPI failure and SPA-only terms snapshot require human review | **Acquisition/inventory normalization accepted; evidence admission conditional** and never efficacy/safety labels by default ([API](https://clinicaltrials.gov/data-api/api), [download guidance](https://clinicaltrials.gov/data-api/how-download-study-records)) |
| AACT | Relational ClinicalTrials.gov snapshots | Reproducible bulk trial analyses | Preserve AACT snapshot and upstream ClinicalTrials.gov timestamps/fields | **Future/conditional**, depending snapshot terms and infrastructure |
| Drugs@FDA | `datdaf20260804` archive, landing page, and ERD are frozen; all 12 tables were parsed and relationally audited with retained upstream anomalies; zero canonical rows or labels admitted | US application, approval, action-date, and product evidence | US government source; preserve application/product relationships and update date; approval is indication/formulation specific | **Acquisition accepted; regulatory-view admission conditional** on mapping and anomaly policy ([data files](https://www.fda.gov/drugs/drug-approvals-and-databases/drugsfda-data-files), [openFDA API](https://open.fda.gov/apis/drug/drugsfda/)) |
| DailyMed | All six human-prescription release parts dated 2026-08-03 are frozen and member-verified: 54,672 members; no SETID history, section extraction, or mapping yet | Label sections, current labeling, and versioned safety statements | Preserve SETID/SPL version, effective date, section identity, and retrieval date; label text is not a normalized event table without review | **Raw acquisition accepted; label-view admission conditional** on parsing, history, mapping, rights, and adjudication ([API](https://dailymed.nlm.nih.gov/dailymed/app-support-web-services.cfm)) |
| Open Targets | Current verified release 26.03, released 2026-03-23; quarterly | Target-disease/drug evidence discovery and cross-checking | Platform downloads are CC0, but underlying evidence-source provenance and citations must remain visible | **Conditional** as an evidence index, not a replacement for primary records ([release announcement](https://blog.opentargets.org/open-targets-platform-26-03-has-been-released/), [data access](https://platform-docs.opentargets.org/data-access)) |
| DrugBank | Current academic release page reports 5.1.22 and temporarily paused downloads | Drug identity, target, PK, trial, and regulatory enrichment | Requires an accepted license; academic use is noncommercial and redistribution/product-development restrictions apply | **License blocked** for the public tool absent negotiated rights ([academic access](https://go.drugbank.com/academic_research), [terms](https://trust.drugbank.com/drugbank-trust-center/terms-of-use)) |

## Readiness-handoff decision for recommended sources

This decision records both acquisition and the separate admission gate.
Category A means that a bounded, versioned acquisition is scientifically
necessary before any **comprehensive**, **multisource**, **clinical-layered**,
or substantive-training-ready claim; a completed A acquisition can still have
zero admitted rows. Category B means that deferral is responsible because a
concrete rights, governance, storage, endpoint, or modality design is
unresolved. A Category B deferral still leaves the associated modality and
claim **not ready**.

| Source | Category | Independent decision and minimum acceptable next action | Readiness consequence while absent |
|---|---|---|---|
| BindingDB 202608 | **A — acquisition/normalization completed; admission necessary** | The exact `5a7425…` raw bundle and `4a9060…` normalized bundle preserve all 93,712 640-field source maps. The origin audit identifies 93,023 BindingDB-curated candidates and 95,506 candidate endpoint cells, excludes 429 ChEMBL mirrors from independent-source counting, and quarantines 260 Taylor rows; 190 repeated reactant-set-ID excess rows remain for semantic review. Next: reconcile structures, proteins, assays, conditions, exact/near mirrors, and conflicts against ChEMBL without collapsing source assertions. ([official downloads](https://www.bindingdb.org/rwd/bind/chemsearch/marvin/Download.jsp?all_download=yes)) | Multisource candidate evidence now exists, but canonical multisource coverage, cross-source conflict analysis, rights approval, and model-label eligibility still fail until admission QC is executed. |
| Contributor-filtered PubChem BioAssay | **B — responsibly deferred** | PubChem records are depositor assertions that can be replaced or revoked. A public-training allowlist requires depositor-specific provenance/terms, version/tombstone handling, endpoint mapping, and a cross-source mirror policy. Preserve the existing local bytes only as conditional evidence; do not admit them until that policy is approved. ([citation guidance](https://pubchem.ncbi.nlm.nih.gov/docs/citation-guidelines), [record update/revocation behavior](https://pubchem.ncbi.nlm.nih.gov/docs/update-or-revoke-bioassays)) | PubChem assay breadth and contributor-level evidence are absent; absence is not inactivity and cannot be called comprehensive public assay coverage. |
| UniProtKB 2026_02 | **A — acquisition/normalization completed; reconciliation necessary** | The release-pinned `d4000c69…` package and `4a9060…` normalized entry/resolution/membership tables are frozen. Sequence MD5/length/SHA-256 checks passed for 13,976 records; three requested accessions are ambiguity quarantine, 46 resolution identifiers are syntax quarantine, and 1,007 inactive returned records lack the metadata/sequence needed for model input. No identity was silently replaced. Next: reconcile every ChEMBL/BindingDB component and construct, then run sequence/homology leakage analysis. ([release](https://www.uniprot.org/release-notes/2026-06-10-release), [license](https://www.uniprot.org/help/license), [API](https://www.uniprot.org/help/api_queries)) | Release-aligned identity candidates now exist, but unresolved/inactive/construct states and the missing homology audit keep protein-agnostic task readiness closed. |
| wwPDB/RCSB PDB and PLINDER | **B — responsibly deferred for this sequence-plus-SMILES handoff** | PDB archive/API data are CC0, but no structure-conditioned default task, coordinate/assembly/construct admission policy, or pose-quality contract is frozen. PLINDER requires release+iteration pinning and its full production dataset is hundreds of GB. Defer the full structure corpus until that modality and storage budget are approved; a later targeted PDB metadata adapter should precede any structure-model claim. ([RCSB usage policy](https://www.rcsb.org/pages/usage-policy), [PLINDER dataset tutorial](https://plinder-org.github.io/plinder/tutorial/dataset.html)) | Sequence-plus-SMILES artifact plumbing can be tested; structure-conditioned training, pocket generalization, pose benchmarking, and structure coverage remain not ready. |
| PK-DB | **B — responsibly deferred** | The API exposes study, group, individual, intervention, output, and concentration-time entities, but the site states that data use is governed by separate terms. Obtain and archive those terms, approve subject-level governance/small-cell policy, and freeze a PK endpoint/species/route/matrix/analyte/statistic ontology before ingestion. The LGPL software label is not a data redistribution license. ([PK-DB](https://pk-db.com/), [API](https://pk-db.com/api/v1/swagger/)) | The broad ChEMBL PK/ADME candidate inventory is not a context-complete PK corpus; PK task and product claims fail. |
| ClinicalTrials.gov / AACT | **A — authoritative JSON acquisition and membership normalization completed; mapping/admission necessary** | The exact same-version all-DRUG and heuristic cardiac cohorts are recursively bound in raw manifest `5b33bcc2…` and normalized manifest `4a9060…`: 210,644 and 3,879 unique NCT memberships respectively. Registry status, `HasResults`, outcome-module presence, and adverse-event-module presence remain separate inventory facts; zero outcomes were interpreted. Next: resolve interventions to molecule/material entities with confidence and review, define compatible arm/denominator/outcome semantics, and complete terms review. AACT remains optional. ([official API](https://clinicaltrials.gov/data-api/api), [download guidance](https://clinicaltrials.gov/data-api/how-download-study-records), [AACT snapshots](https://aact.ctti-clinicaltrials.org/archive/download)) | Clinical-development and posted-module inventories now exist, but no molecule-linked clinical outcome or safety task exists. ChEMBL `max_phase`, no record, and no posted result remain non-outcomes. |
| Drugs@FDA | **A — acquisition completed; mapping/admission necessary** | The exact `d995f330…` manifest binds the landing page, ERD, full 12-table archive, parse audit, and relational-key audit. Retain the one width anomaly, 16 blank key rows, and 6,906 foreign-key orphan occurrences rather than repairing silently. Next: map application/product/form/strength/ingredient identities with confidence and temporal scope. ([data files](https://www.fda.gov/drugs/drug-approvals-and-databases/drugsfda-data-files)) | Regulatory bytes now exist, but no application/product row is a universal molecular efficacy, safety, PK, or activity label. |
| DailyMed | **A — complete raw release acquired; parsing/history/admission necessary** | The exact `d29a97bb…` manifest binds the landing page and every byte/member of all six human-prescription archives. Next: parse SPL XML, inventory SETID/document/version/effective-time identity, acquire histories for the admitted scope, map products/ingredients, and adjudicate quantitative facts with raw XML paths. ([DailyMed API](https://dailymed.nlm.nih.gov/dailymed/app-support-web-services.cfm)) | Complete raw label archives now exist, but versioned label-section evidence and quantitative PK/safety facts remain absent until parsing and review; the 1,163 ChEMBL `DRUG_PK` assertions are not a substitute. |
| EPA ToxCast / CompTox | **B — responsibly deferred** | The current ToxCast release is invitrodb v4.3 (August 2025) and EPA describes the data as open, but it is heterogeneous screening toxicology rather than automatic hERG, QT, or clinical cardiotoxicity evidence. Freeze the precise chemical/assay/curve-processing/QC admission design and cross-map DTXSID/material identities before ingestion. ([ToxCast downloads](https://www.epa.gov/comptox-tools/exploring-toxcast-data), [open-data terms](https://www.epa.gov/comptox-tools/downloadable-computational-toxicology-data)) | Broad toxicology coverage is absent. The source cannot be counted as a clinical cardiac endpoint, and hERG task readiness does not imply ToxCast readiness. |

This classification deliberately does not use local runtime pressure as a
scientific excuse. Category A omissions are remaining feasible work and hard
evidence gaps. Category B omissions are documented deferrals, not completed
coverage. Neither category permits ChEMBL scale to be relabeled as multisource,
clinical, regulatory, structural, PK, or broad-toxicity coverage.

## Citation and provenance acceptance test

A source may enter a release build only if all answers are yes:

1. Is the upstream owner authentic and the access route documented?
2. Is an immutable release/version, record version, or content checksum stored?
3. Are retrieval timestamp and exact query/export parameters stored?
4. Are the source record and primary publication identifiers retained per row?
5. Is contributor/curation origin retained where an aggregator mixes sources?
6. Is the license/terms snapshot stored with the scope to which it applies?
7. Are redistribution, model-training, commercial-use, and derived-artifact rights separately classified?
8. Is the endpoint/protocol information sufficient for the intended task?
9. Have exact and near duplicates against all other sources and benchmarks been audited?
10. Is there an exclusion path that preserves the raw record and reason without turning absence into a negative label?

Failure of questions 1–7 is a release blocker. Failure of 8–10 may allow evidence-browser display but not model-label admission.

## External model assessment

External models are candidates for locked evaluation adapters, not automatic platform dependencies or ground truth.

| Model | Verified capability and terms | Principal scientific risk | Decision |
|---|---|---|---|
| AlphaFold 3 | Current inference source code is Apache-2.0; weights and outputs are separately restricted to specified noncommercial uses and AF3 output may not train similar structure models | Product-use and training restrictions; prediction confidence is not affinity; substantial GPU/database requirements | **License blocked** for general product/training use unless institutional counsel approves an exact workflow ([repository](https://github.com/google-deepmind/alphafold3), [weight terms](https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md)) |
| Chai-1 | Apache-2.0 repository; current README recommends pinned `chai_lab==0.6.1`, Linux and a CUDA/bfloat16 GPU | Structure prediction only; version/MSA/template choices and training-cutoff similarity must be audited; no affinity truth | **Evaluation only**, hardware-blocked locally ([repository](https://github.com/chaidiscovery/chai-lab)) |
| Boltz-2 | MIT code and weights; cofolds structures and emits binder probability plus an affinity value documented as `log10(IC50)` from micromolar IC50 | Official repository still marks updated Boltz-2 training and evaluation code “coming soon”; published benchmark overlap and chemistry similarity limit broad claims; output is not thermodynamic free energy | **Evaluation only**, pending reproducibility and overlap review ([repository](https://github.com/jwohlwend/boltz)) |
| Nesso-1 | Apache-2.0 model card, code, and weights; released July 2026; sequence+ligand input with score and binder head | Extremely new and not independently validated. The report explicitly treats `Ki`, `Kd`, `IC50`, and `EC50` interchangeably and trains on BindingDB, ChEMBL v34, PubChem 1.8.1, CeMM, and MIDAS. Its output must therefore be labeled a mixed potency/affinity score, never `Kd` or binding free energy. The report itself finds public benchmarks largely in-distribution and only modest improvement over molecular-weight ranking on OpenBind | **Evaluation only** with endpoint warning and full overlap audit ([model card](https://huggingface.co/recursionpharma/nesso), [technical report](https://www.valencelabs.com/wp-content/uploads/2026/07/nesso1.pdf), [repository](https://github.com/recursionpharma/nesso)) |

The meeting note referring to a model trained on approximately 100,000 structures could not be mapped confidently to a specific publication or release. None of the reviewed primary materials establishes that identification. Status: `requires_human_review`; the model/paper must be named before it enters a comparison table.

## Required in-house baselines before foundation-model claims

For each endpoint/task, run a hierarchy whose complexity is justified by the data:

1. training-set mean/median or prevalence prior;
2. molecule and target marginal priors where permitted by the split;
3. nearest-neighbor or matched-series baseline with train-only fitting;
4. fingerprint plus regularized linear/logistic model;
5. fingerprint plus tree ensemble;
6. ligand graph model and sequence-ligand baseline when sample size supports them;
7. structure-aware or external foundation model only after all preceding baselines and overlap audits.

All models must use the identical locked test population, endpoint definition, censoring policy, and admissible information. Hyperparameter selection, calibration, thresholds, and early stopping are training/validation operations only. External-model inference caches must store upstream model commit/release, weight hash, input hash, settings, hardware, runtime, and license classification.

## Local legacy model assessment

- The Menin primary model is a single-target, molecular-fingerprint Extra Trees regression baseline. Its scaffold split and exact-structure isolation are locally evidenced; target generalization is not tested.
- The hERG primary model is a molecular-fingerprint calibrated Extra Trees classifier using project thresholds and an ambiguous exclusion interval. It estimates a curated hERG task, not QT or clinical cardiac risk.
- The locally reported temporal stress tests are weak (negative Menin temporal R-squared and modest hERG discrimination/calibration). These are useful warnings against broad generalization, not reasons to hide the split.
- The PK corpus is an inventory/analysis surface with high missingness and heterogeneous contexts, not an audited PK prediction model.
- Existing structure/physics artifacts are exploratory local work. The configuration records production simulation as disabled and execution deferred to HPC; no production-HPC completion claim is supported.

No legacy model should be presented as the new protein-agnostic platform model. Its appropriate role is a diagnostic baseline and migration regression test.
