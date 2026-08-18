# Lead reconciliation: scale expansion beyond hERG, PK, and affinity

Date: 2026-08-07  
Scope: public, non-HPC acquisition and source audit. Nothing in this report is a canonical training label or a performance claim.

## Executive result

The project now has a defensible path to **millions of measurements**, but not by pretending that millions of independent hERG or systemic-PK observations exist publicly.

- hERG now has **773,908 gross wild-type/standard candidate rows** across the named local source files, plus 342,311 mutant-channel rows: **1,116,219 gross hERG-related physical rows**. The increase adds 1,360 NVS/ERF EPA result rows but does not add the duplicate-lineage EPA Tox21 result set. This gross figure deliberately includes mirrors, repeated structures, assay repeats, controls, binary wrappers, and the quarantined compact BindingDB snapshot; it is not a count of independent measurements. A second exhaustive PubChem KCNH2 audit added Tox21 AID 1671200 (9,667 SIDs; 7,671 structures; 6,515 structures absent local ChEMBL) and demonstrated that 4,005 of 4,067 KCNH2 AIDs are ChEMBL/BindingDB mirrors. The current planning estimate remains roughly 300,000–350,000 unique structures after cross-source standardization; it is not a measured canonical count.
- Genuine systemic PK remains scarce. Only 2,437 public benchmark rows were acquired, and none contains the complete structured endpoint/unit/species/route/matrix/dose/time context required for a high-confidence systemic-PK label. Another 274,168 physical rows are upstream ADME/safety observations, while 330,261 ChEMBL type hits are reclassification candidates rather than admitted PK labels. A completed scan of all 54,672 frozen DailyMed SPL packages found 43,162 quantitative PK candidate sections and 29,664 machine-detected Tier-A sections, but still admitted zero measurements because value-level context and ingredient resolution remain unverified.
- Binding affinity is genuinely multi-million-scale. The checksum-verified BindingDB 2026-08 archive contains 3,234,499 rows and 2,481,305 distinct monomer–target keys. Local ChEMBL 37 contains 2,117,384 Kd/Ki/IC50/EC50 task rows, but 1,658,927 BindingDB rows are ChEMBL-tagged. Exact standardized matching recovered 1,229,978 of 1,658,153 eligible tagged measurements and bounded that mirror-domain union at 2,864,273–3,266,356 distinct measurement keys, so the sources must not be added arithmetically.
- Two locally acquired, important adjacent modalities add scale and mechanistic diversity: LINCS L1000 has 1,665,114 perturbation instances, and PRISM has 8,506,400 observed drug–cell-line viability values across its two collapsed matrices. The complete 17,651,807,605-byte EPA ToxCast invitrodb v4.3 archive is now local and passes publisher MD5, SHA-256, and gzip integrity checks; EPA reports that its relational database contains more than half a billion records.
- JUMP Cell Painting is the strongest image-based extension: 116,000+ compounds and 15,000+ genetic perturbations, over 8 million images, and over 1.5 billion segmented cells. Its official metadata and CC0 terms were pinned locally; the 358.4 TB payload was deliberately remote-indexed rather than downloaded without HPC/object-storage planning.

These sources address complementary questions. They remain separate endpoint tiers and are never pooled merely to make a larger headline.

## 1. hERG saturation audit

The second-round audit enumerated every PubChem KCNH2 target record accessible through official target-centric queries:

| Evidence | Verified scale | Lead disposition |
| --- | ---: | --- |
| PubChem KCNH2 AID universe | 4,067 AIDs | 3,878 ChEMBL mirrors, 127 BindingDB mirrors, 62 other; do not count mirrors as new assays |
| Tox21 AID 1671200 | 9,667 SIDs; 7,671 structures | Source-grade fluorescence qHTS; keep separate from patch-clamp and binding |
| BindingDB Q12809 | 18,759 measurements; 14,832 structures | Useful delta candidates; quarantine compact-API rows until curator, assay, and inherited-license lineage are available |
| Zenodo ToxTree 5807719 | 8,879 rows | Checksum verified, but mostly redundant with later release and ChEMBL |
| First-round public inventory | >700,000 wild-type/standard rows; 342,311 mutant rows | Raw scale only; cross-source structure and assay deduplication still required |

The lead reconstructed the prior gross named-source total as 772,548 wild-type/standard candidates: ChEMBL 41,078; HERGAI 299,927; PubChem AIDs 720551/588834/1671200 at 343,909/5,381/9,667; quantitative three-channel 22,969; Zenodo union/ToxTree 7,879/8,879; TDC hERG/hERG Karim 655/13,445; and quarantined BindingDB Q12809 18,759. Adding 1,360 physical NVS/ERF EPA results and the separately labeled 342,311 mutant rows produces 1,116,219 physical rows. This accounting is useful for storage/provenance only and must never be advertised as an independent or deduplicated label count.

This is close to public-source saturation for dedicated hERG datasets, not proof that every proprietary or unpublished experiment has been obtained. hERGCentral lacks a current authoritative licensed snapshot, the 2018 integrated release contains commercial GOSTAR material, and several public wrappers repackage the same upstream records. EPA v4.3 contains exactly three human KCNH2 endpoints: NVS 686 and ERF 3184 are independent radioligand-binding lineages, whereas Tox21 3210 is the same experimental lineage as PubChem AID 1671200. The completed full-archive extraction recovered 1,219 NVS and 141 ERF result rows, with 1,077 and 131 resolved DTXSIDs respectively. Across all three endpoints the union is 7,959 DTXSIDs, but only 88 (13 NVS-only and 75 ERF-only) fall outside EPA Tox21. Tox21's 9,667 mapped rows exactly match the PubChem study, so they cannot be counted as an independent experiment. Cross-source novelty of the 88 remains fail-closed pending a structure-resolved DTXSID crosswalk against ChEMBL and BindingDB.

## 2. PK and ADME reality check

Systemic PK labels require endpoint, unit, molecule/formulation, species, route, matrix, dose, sampling/time basis, and study provenance. Applying that definition prevents permeability, microsomal stability, CYP activity, or a label called “half-life” from being silently presented as clinical/in-vivo PK.

| Tier | Verified local evidence | Scientific meaning |
| --- | ---: | --- |
| Published systemic-PK benchmarks | 2,437 rows | Bioavailability 640; half-life 667; Vdss 1,130; all context-incomplete |
| Upstream ADME/safety files | 274,168 rows | Non-deduplicated TDC, NCATS, and OpenADMET physical rows |
| ChEMBL exact PK-type candidates | 330,261 rows / 85,441 molecules | Requires document, unit, route, dose, and semantic adjudication |
| PK-DB advertised outputs | 138,411 | Zero anonymously retrieved; Basic-authentication and rights blockers preserved |
| Context-complete admitted systemic PK | **0** | Correct fail-closed result, not missing bookkeeping |

The correct route to a large PK program is structured extraction from study-level sources plus authorized PK-DB access, not counting ADME proxies as PK. The completed DailyMed scan parsed 54,672/54,672 SPL packages with zero errors and identified 41,960 candidate document versions, 43,162 candidate sections, 26,610 tables, and 313,014 bounded evidence spans. Of the sections, 29,664 contain all core context term classes in one section, but applicability to individual values remains unverified. Exact SPL approval identifiers join 19,652 candidate versions to Drugs@FDA, while the strict active-ingredient XML path resolves only 10,976 candidate versions. These artifacts materially improve the curation queue but still contribute zero structured PK quantities.

## 3. Binding-affinity scale

The full BindingDB archive is the primary multi-million source. It contains 130,168 Kd, 627,186 Ki, 2,209,905 IC50, and 270,125 EC50 populated endpoint cells. Relations and ranges are preserved; 648,109 populated cells are censored/range-like rather than exact values. The archive also exposes extensive lineage overlap: 1,658,927 rows tagged ChEMBL, 1,339,238 US-patent rows, 102,722 PubChem rows, and 93,085 BindingDB-curated literature rows.

Guide to PHARMACOLOGY 2026.2 adds 24,599 curated interactions (22,316 quantitative; 20,258 official unique pairs). BioLiP2 adds 68,890 structure/affinity links, not 68,890 independent affinities. Papyrus was retained as release metadata rather than downloaded wholesale because its dominant ChEMBL content is older than the locally pinned ChEMBL 37.

For modeling, Kd, Ki, IC50, and EC50 remain different tasks. The exact ChEMBL-tagged overlap audit found 1,229,978 combined-identity matches (74.18% of 1,658,153 eligible measurements), including 759,801 IC50, 304,312 Ki, 67,214 Kd, and 98,651 EC50 rows. It also exposed 455,907 explicit-ID rows with different exact InChIKeys, which require parent/salt/stereo/release-drift review rather than automatic deletion. Experimental Kd may support a stated standard-free-energy transformation under explicit thermodynamic assumptions; IC50 and EC50 must never simply be renamed binding free energy.

## 4. Additional high-value, million-scale modalities

### 4.1 LINCS L1000 — perturbational transcriptomics

Official NCBI GEO metadata were acquired for Phase I GSE92742 and Phase II GSE70138. Physical metadata contain:

- Phase I: 1,319,138 perturbation instances and 473,647 aggregate signatures.
- Phase II: 345,976 perturbation instances and 118,050 aggregate signatures.
- Combined: **1,665,114 instances and 591,697 signatures**, with 978 directly measured landmark genes. The complete selected Level-2 layer is present locally: Phase I epsilon (1,269,922 profiles) plus delta (49,216), and Phase II (345,976). Its profile-by-gene matrices contain **1,628,481,492 numeric matrix positions** (1,665,114 × 978), a more meaningful scale statement than counting each profile as one datum.

This modality can connect a molecule to pathway/cell-state response, supporting mechanism-of-action, toxicity, repurposing, and out-of-domain validation. Replicate instances and aggregate signatures are different units and are not summed.

### 4.2 PRISM Repurposing — drug–cell-line viability

The CC BY 4.0 Figshare v4 files were downloaded and all nine local files match the publisher MD5 values.

| Matrix | Shape | Possible cells | Observed numeric values | Missing |
| --- | ---: | ---: | ---: | ---: |
| Primary replicate-collapsed | 578 × 4,686 | 2,708,508 | 2,632,171 | 76,337 |
| Secondary replicate-collapsed | 489 × 13,008 | 6,360,912 | 5,874,229 | 486,683 |
| **Combined observed** | — | — | **8,506,400** | — |

The secondary curve-parameter table contains 701,004 data rows. Matrix columns represent treatment samples/conditions, not necessarily unique parent compounds; chemical identity, dose, replicate, and screen phase must remain explicit.

### 4.3 EPA ToxCast/Tox21 — mechanistic toxicity

The official current invitrodb v4.3 release (August 2025, CC0 at the Figshare release level) was selected because it standardizes concentration-response data across diverse toxicological pathways and approximately 10,000 substances. The clean replacement transfer completed at exactly **17,651,807,605 bytes**. Its MD5 `5d05c0f4075408a104bb50003c5170a1` and SHA-256 `ee159e1cdd28996f85db13e742700d8d76ef9d5baf31e3b5e00d249899529c7b` match the publisher, and `gzip -t` passes. The earlier 5,518,020,608-byte invalid `.partial` was deleted after verification to reclaim space.

EPA describes the database as exceeding half a billion records. This is a relational evidence source: raw wells, normalized responses, curve fits, hit calls, chemicals, and assay metadata are different record types, so “half a billion” is not half a billion independent labels. The full dump was not expanded into a local MySQL instance because the expanded database exceeds the remaining safe local storage. A fail-closed streaming parser nevertheless consumed the complete archive and exactly extracted the three human KCNH2 endpoints: 9,787 MC and 1,243 single-concentration rows, 7,959 resolved DTXSIDs, and 11 rows belonging to eight named controls without `sample` mappings. An independent validator matched every CSV cell to the compact SQLite audit database, checked all hashes and arithmetic, and passed. Canonical and training admission remain zero pending structure-resolved cross-source reconciliation and modality-specific admission.

### 4.4 JUMP Cell Painting — morphological phenomics

The official Cell Painting Gallery repository was pinned at commit `8224008f9682a1e9d3ec0dbf371232f5e625bf98` and the official JUMP datasets/metadata repository at `016e865fa0691244e0860943e41c7d6a88ed2580`. The gallery is CC0; the metadata repository carries a BSD 3-Clause license. Local gzip-validated metadata contain 115,796 compound rows, 7,977 CRISPR rows, 15,132 ORF rows, 2,525 plates, and 1,151,808 well rows. Publisher statistics sum to 8,126,570 sites, 58,267,491 channel TIFFs, and approximately 1,539,797,273 cells. The gallery lists 358.4 TB total for cpg0016-jump. This is important because morphology can capture cellular mechanisms missed by target-only affinity and single-endpoint ADME assays. Images and full numerical profiles remain remote; their ingestion requires object-storage and HPC design.

## 5. Integration and leakage rules

1. Count source rows, standardized structures, biological pairs, assay observations, replicate wells, signatures, and matrix values separately.
2. Preserve source record IDs, source version, license, citation, original units/relations, assay modality, species, cell line, dose/concentration, time, and construct.
3. Resolve exact evidence duplicates before train/validation/test assignment. Use structure, target, assay/publication, cell line, and time-aware groups appropriate to each task.
4. Keep experimental measurements separate from derived transformations and predictions.
5. Never use a source’s derivative mirror as independent external validation of its upstream source.
6. Admit no PK row until its context passes the systemic-PK contract; admit no compact BindingDB hERG delta until source/assay/rights lineage is restored.

## 6. Source and literature basis

- EPA ToxCast current release and pipeline: https://www.epa.gov/comptox-tools/exploring-toxcast-data
- EPA invitrodb v4.3 record: https://epa.figshare.com/articles/dataset/ToxCast_Database_invitroDB_/6062623
- LINCS Phase I: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE92742
- LINCS Phase II: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE70138
- PRISM data and study: https://depmap.org/repurposing/ and DOI 10.1038/s43018-019-0018-6
- Cell Painting Gallery: https://registry.opendata.aws/cellpainting-gallery/
- JUMP Cell Painting: DOI 10.1101/2023.03.23.534023
- BindingDB current download/update: https://www.bindingdb.org/rwd/bind/chemsearch/marvin/Download.jsp and DOI 10.1093/nar/gkae1075
- Guide to PHARMACOLOGY: https://www.guidetopharmacology.org/download.jsp
- BioLiP2: https://zhanggroup.org/BioLiP/ and DOI 10.1093/nar/gkad630
- Papyrus: DOI 10.1186/s13321-022-00672-x
- TDC: https://tdcommons.ai/single_pred_tasks/overview/
- OpenADMET: https://openadmet.org/datasetsmodels/
- PK-DB: https://pk-db.com/

## Honest completion boundary

The acquisition and audit establish multi-million-scale public evidence and a much stronger scientific program. They do **not** establish millions of unique compounds, millions of independent hERG labels, millions of context-complete PK measurements, clinical validity, or a finished training corpus. Canonical identity reconciliation, assay-aware admission, rights approval, frozen leakage-resistant splits, modeling, and external validation remain separate gates.
