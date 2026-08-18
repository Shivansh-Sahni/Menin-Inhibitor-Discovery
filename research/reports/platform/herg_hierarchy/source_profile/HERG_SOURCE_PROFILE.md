# hERG source profile and admission recommendation

Date: 2026-08-07  
Scope: local, already-acquired hERG evidence only. This report profiles sources; it does not alter the raw data, canonical corpus, labels, or model splits.

## Lead decision

The fastest defensible way to obtain a **hundreds-of-thousands-scale hERG training table** is to use **PubChem AID 720551 as the primary weak-label backbone**. It now has exact local structure coverage for all **343,666 unique CIDs**. After collapsing repeated CIDs, excluding five outcome-conflict CIDs, and excluding inconclusive calls, it provides **342,932 unique-CID binary candidates: 1,263 Active and 341,669 Inactive**.

This is large and structurally usable, but it is **not a 342,932-row IC50 dataset**. AID 720551 is a KCNH2 wild-type thallium-flux screen with activity reported at 0.369 and 1.840 µM and categorical calls derived from maximum normalized activity. Its clean binary set is only **0.3683% Active**, so a random split or unweighted accuracy would be misleading.

The recommended compact model table is therefore:

```text
canonical_smiles, herg_call
```

The minimum provenance sidecar must retain:

```text
pubchem_cid, pubchem_sid, aid, raw_outcome, evidence_type,
assay_modality, tested_concentrations_uM, source_structure_smiles,
standardization_version, standard_inchi_key, exclusion_reason
```

The sidecar is essential. It prevents a categorical flux result from later being silently relabeled as patch-clamp IC50 or as a clinically validated result.

## Source-level recommendation

| Source | Verified local size | Endpoint meaning | Recommended role | Admission decision |
| --- | ---: | --- | --- | --- |
| **PubChem AID 720551** | 343,909 SID rows; 343,666 CIDs/structures; 342,932 conflict-free binary CIDs | Two-point wild-type KCNH2 thallium-flux outcome, not IC50 | Primary large weak-label pretraining backbone | **Admit after deterministic structure standardization; exclude conflicts and inconclusive calls** |
| **Zenodo 8359714** | 22,969 pIC50 rows/unique InChIKeys | Curated quantitative pIC50 from mixed upstream sources | Quantitative fine-tuning and calibration | **Admit as a separate quantitative task after provenance/overlap reconciliation** |
| **ChEMBL 37 CHEMBL240** | 41,078 observations; 26,254 structures; 4,829 assays; 3,494 documents | Mixed IC50, inhibition, Ki, kinetics, AC50, and other endpoints | Provenance-rich assay and quantitative backbone | **Admit endpoint-by-endpoint; never pool all rows into one label** |
| **PubChem AID 588834** | 5,381 SIDs; 4,743 CIDs | Full concentration-response hERG FluxOR qHTS with AC50/curve evidence | Functional-flux auxiliary and validation source | **Admit separately from patch clamp and from AID 720551** |
| **PubChem/Tox21 AID 1671200** | 9,667 SIDs; 7,671 CIDs | Replicated FluxOR II qHTS with curve/QC fields | Auxiliary functional-flux task | **Retain as direct PubChem source; do not duplicate its EPA mirror** |
| **HERGAI** | 299,927 rows; 298,402 exact SMILES; 1,937 Active | Secondary binary compilation; potency numeric for only 2,340 rows | Crosswalk, sensitivity analysis, or secondary weak-label benchmark | **Do not concatenate with AID 720551** |
| **BindingDB Q12809 API** | 18,759 affinity rows; 14,832 RDKit-valid structures | Mostly IC50 plus Ki/EC50/Kd; mixed binding/functional context | Candidate delta and provenance audit | **Quarantine pending assay-rich curator/source-license export** |

The first five named inventories in the earlier expansion report sum to **713,264 raw rows** (343,909 + 299,927 + 41,078 + 22,969 + 5,381). That number is an inventory count, **not a unique-compound count**. The 342,932-row AID 720551 table is the honest immediate scale claim; other sources should add richer labels and independent evidence, not be advertised as additive molecules.

## 1. Primary scale backbone: PubChem AID 720551

### Verified counts

- Raw concise rows and unique SIDs: **343,909**.
- Unique PubChem CIDs: **343,666**.
- Raw outcomes: **1,267 Active + 341,912 Inactive + 730 Inconclusive = 343,909**.
- Repeated-CID groups: **237**.
- Outcome-conflict CIDs: **5**: four Active/Inactive and one Inactive/Inconclusive.
- Nonconflicting CID outcomes: 1,263 Active, 341,669 Inactive, and 729 Inconclusive.
- Conflict-free binary CIDs: **1,263 + 341,669 = 342,932**.
- Active fraction in that binary set: **1,263 / 342,932 = 0.3683%**.

### Structure readiness

The local PubChem property acquisition returned **343,666/343,666 requested unique CIDs**, with **zero missing CIDs, zero extra CIDs, and nonblank isomeric SMILES, connectivity SMILES, and InChIKey for every returned CID**. The merged structure CSV has SHA-256 `595efcd06b0a1a8662220559e8de08afd10a0e680120af884d3be594b5b3fd7f`.

This proves source-identifier coverage. It does **not** yet prove that all 343,666 records remain unique after parent-fragment removal, charge normalization, tautomer policy, and stereochemistry policy. The final training count may decrease slightly after RDKit standardization and standardized-structure conflict adjudication.

### Endpoint semantics

The official retained description identifies a confirmatory KCNH2 wild-type U2OS assay using a FluxOR thallium-flux surrogate. It records activity at **0.369 and 1.840 µM**. Outcome rules are based on maximum normalized activity:

- Active: maximum activity below -70%.
- Inconclusive: between -70% and -50%.
- Inactive: above -50%.

The concise export's `Activity Value [uM]` field is empty, so it cannot supply per-molecule IC50/AC50 regression labels. A binary `herg_call` is defensible if it is explicitly named a weak flux-screen call. An exact potency is not.

### Provenance and rights

The raw assay, summary, and description came from official PubChem/NCBI endpoints and retain NCGC depositor identity. NCBI does not impose an additional restriction on molecular data but also does not transfer contributor rights. Preserve PubChem, NCGC/NIH, assay ID, retrieval, and source attribution; do not represent PubChem access as a blanket relicensing of depositor content.

### Admission role

Use this source for chemical-space representation learning and weak-label hERG classification. Use class weighting or balanced batches, scaffold/structure-group splits, PR-AUC, active recall, precision at a declared review budget, and calibration. Do not use raw accuracy as a headline metric.

## 2. Quantitative calibration: Zenodo 8359714

This CC BY 4.0 release supplies **22,969 nonoverlapping, source-provided hERG pIC50 rows**, all unique by supplied InChIKey and exact SMILES:

| Supplied split | Rows | pIC50 >= 5 | pIC50 <= 4.522879 | Gray zone |
| --- | ---: | ---: | ---: | ---: |
| Development | 22,246 | 13,428 | 4,869 | 3,949 |
| Evaluation <=0.60 similarity | 250 | 125 | 89 | 36 |
| Evaluation <=0.70 similarity | 473 | 264 | 113 | 96 |
| **Total** | **22,969** | **13,817** | **5,071** | **4,081** |

The development source mix is ChEMBL 15,624; Didziapetris et al. 2,372; PubChem 2,081; US Patent 1,238; Konda et al. 463; Doddareddy et al. 417; Munwar et al. 36; and BindingDB 15. This is a curated integration, not 22,969 independent assay lineages. In particular, its ChEMBL rows must be reconciled with the local ChEMBL 37 extract before novelty or validation claims.

Use supplied pIC50 for a separate regression/calibration head. The thresholds above are profiling cutoffs, not justification for discarding the 4,081 intermediate values. Preserve pIC50 continuously.

## 3. Provenance-rich backbone: ChEMBL 37

The specialized CHEMBL240 view contains:

- **41,078 rows**, **26,310 molecule IDs**, **26,254 unique nonnull InChIKeys/SMILES**, **4,829 assays**, and **3,494 documents**.
- IC50 19,130; inhibition 7,808; Ki 3,689; `kon` 2,946; `k_off` 2,946; AC50 1,823; plus smaller heterogeneous endpoint groups.
- Exact, positive IC50 in nM or µM with a nonnull SMILES: **12,008 rows / 9,802 structures**.
- Assay types: binding 32,640; toxicity 4,390; functional 3,597; ADME 451. These database categories still require description-level modality review.

ChEMBL is CC BY-SA 3.0. Preserve attribution and share-alike obligations in any redistributed adaptation. ChEMBL's source tags are provenance, not independent datasets: the view includes literature, BindingDB, DrugMatrix, PubChem, and other imports.

Only exact positive IC50 values with verified units may be converted using `pIC50 = 9 - log10(IC50_nM)`. Inequalities must remain censored; inhibition percentages, Ki, AC50, kinetics, and categorical calls must not be passed through that formula.

## 4. Useful auxiliary functional assays

### PubChem AID 588834

- **5,381 data rows/SIDs**, **4,743 mapped CIDs**, 664 Active, 4,107 Inactive, and 610 Inconclusive.
- **5,363 rows have depositor SMILES**, representing 4,743 unique exact SMILES.
- This is a human hERG FluxOR concentration-response assay with AC50, efficacy, Hill-fit, curve-class, response-at-concentration, and QC fields.

It is substantially smaller than AID 720551 but scientifically richer. Keep its AC50/curve evidence intact and use it as an assay-specific auxiliary or validation task.

### PubChem/Tox21 AID 1671200

- **9,667 data rows/SIDs**, **7,671 CIDs**, 704 Active, 8,297 Inactive, and 666 Inconclusive.
- **9,524 rows have depositor SMILES**, representing 7,671 unique exact SMILES.
- The full local export retains up to 51 replicate series plus concentration responses, fit statistics, and QC.
- It shares 2,992 CIDs with AID 588834 and 1,510 CIDs with AID 720551, demonstrating that these rows are not all novel compounds.

Use the direct PubChem/Tox21 deposit as the traceable source. The EPA invitrodb representation of the same experiment is a processing mirror and should not become a second training lineage.

## 5. Secondary scale source: HERGAI

HERGAI contributes **299,927 rows/SIDs**, **298,402 unique exact SMILES**, 1,937 Active and 297,990 Inactive. Numeric potency exists for only **2,340 rows**; therefore this is primarily a binary compilation, not a potency set.

Its overlap with AID 720551 is decisive:

- **293,718 shared SIDs**, mapping to 293,718 shared CIDs.
- Every shared row is Inactive in both sources.
- Those CIDs cover **293,912 AID 720551 rows** because some CIDs repeat.

HERGAI must not be appended to AID 720551 as if it contributed another 299,927 measurements. Its practical roles are crosswalk, label-consistency audit, and possibly a secondary weak-label benchmark. AID 720551's official PubChem lineage should be preferred for the primary table.

The pinned repository contains an MIT license, while its README says the data and source code are subject to copyright, and per-record upstream rights are not itemized. Treat redistribution/admission as conditional until that tension and inherited record lineage are reviewed.

## 6. Conditional or excluded sources

### BindingDB Q12809: conditional delta only

The compact API snapshot has **18,759 rows**, **14,864 monomer IDs**, and **14,832 RDKit-valid structures**: IC50 15,753; Ki 2,847; EC50 110; Kd 49. It includes **5,437 censored values**. Standardized overlap analysis found 13,363 structures already in local ChEMBL 37 and 1,469 absent.

The API lacks sufficient row-level assay conditions, curator provenance, and inherited source-license fields for safe canonical admission. Obtain the assay-rich export and distinguish BindingDB-curated CC BY 4.0 rows from ChEMBL-derived CC BY-SA rows before use.

### Explicit exclusions from the primary wild-type training table

- **EPA invitrodb hERG:** audit/validation only. Its 11,030 result rows are mostly a mirror of the Tox21 experiment; only 1,360 physical NVS/ERF rows are outside that experiment, and they are binding/proxy modalities rather than equivalent functional labels. The user explicitly declined EPA hERG for the primary model.
- **Strict 137-row hERG task:** engineering smoke-test view, not the requested scale product. It remains useful for pipeline tests but is excluded from the large corpus.
- **PubChem AID 720553:** 342,311 mutant-channel rows. Different construct/biology; never pool into the wild-type call without a variant-aware task.
- **TDC hERG (655) and hERG Karim (13,445):** derivative benchmark wrappers with source overlap, duplicate structures, and label contradictions. Freeze for external audit only.
- **Zenodo 5807719:** 8,879 rows / 8,877 RDKit structures; 8,617 structures overlap the later Zenodo 8359714 release. Superseded benchmark, not additive evidence.
- **Zenodo 6334912:** 7,879 structures with balanced 0/1 classes, but the file lacks per-row lineage and authoritative class direction/threshold. Do not map class to blocker/nonblocker.
- **ToxiMol hERGCentral:** 60 structures/images but no measured hERG label or potency. Reference only.
- **Bulk hERGCentral/integrated commercial-source datasets:** rights and record-level provenance are unresolved. Do not use scale claims to bypass those gates.

## Non-additivity and evidence-lineage rules

1. A structure appearing in multiple files is not automatically independent evidence.
2. A mirror, wrapper, or later database import does not create a new assay lineage.
3. Deduplicate within a source by source record and standardized structure, but retain legitimate replicates and distinct experiments in the observation table.
4. Never majority-vote across patch clamp, binding, flux, fixed-dose inhibition, mutant channels, and clinical outcomes.
5. The barebones table may contain one call per structure; the observation table must remain the authoritative scientific product.
6. AID 720551 supplies public reported hERG evidence only. It does not by itself establish preclinical validation, clinical validation, or clinical-trial reporting tiers.

## Recommended training sequence

1. **Weak-label pretraining:** conflict-free, conclusive AID 720551 structures; group splits by standardized parent/scaffold and use imbalance-aware objectives.
2. **Quantitative fine-tuning:** Zenodo 8359714 pIC50, then exact ChEMBL IC50 subsets, with source/modality-aware evaluation.
3. **Assay-aware auxiliary heads:** AID 588834/Tox21 flux AC50 and ChEMBL inhibition/binding modalities kept separate.
4. **External stress tests:** HERGAI and TDC wrappers only after all overlapping structures/scaffolds are removed from evaluation.
5. **Evidence hierarchy analysis:** link public hERG observations to independent preclinical and human/trial evidence without changing the molecular hERG label.

## Arithmetic and artifact validation

All headline arithmetic was independently recomputed from local artifacts:

- 1,267 + 341,912 + 730 = **343,909** AID 720551 raw rows.
- 1,263 + 341,669 + 729 + 5 conflict CIDs = **343,666** CIDs.
- 1,263 + 341,669 = **342,932** binary candidates.
- 13,428 + 125 + 264 = **13,817** pIC50 >= 5.
- 4,869 + 89 + 113 = **5,071** pIC50 <= 4.522879.
- 3,949 + 36 + 96 = **4,081** gray-zone rows.
- 13,817 + 5,071 + 4,081 = **22,969** quantitative rows.
- 343,909 + 299,927 + 41,078 + 22,969 + 5,381 = **713,264** gross inventory rows, explicitly non-additive.

Evidence files used for this profile include the official AID summaries/descriptions, the exact-coverage PubChem structure manifest, the ChEMBL specialized view, version-pinned repository metadata/licenses, Zenodo record metadata, and earlier hash-bound acquisition/overlap audits in `research/reports/platform/herg_expansion/`.

