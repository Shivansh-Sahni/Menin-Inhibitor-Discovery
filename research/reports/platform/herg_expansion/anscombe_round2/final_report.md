# HREG-3 public hERG gap audit and bounded acquisition

Date: 2026-08-06  
Scope: new public-source discovery beyond the existing `herg_expansion` inventory. No existing files were changed; no data were canonicalized or admitted to training.

## Result

Three meaningful bounded artifacts were added under `anscombe_round2`:

1. **Tox21 PubChem AID 1671200:** a source-grade human hERG FluxOR II qHTS assay with **9,667 official SIDs**, **7,671 CIDs/structures**, 704 Active, 8,297 Inactive, and 666 Inconclusive. The full 2,049-column depositor export preserves up to 51 replicate series, curve fits, concentration responses, and QC. Of its structures, **6,515 are absent from local ChEMBL 37**.
2. **Current BindingDB Q12809 API snapshot:** **18,759 affinity rows**, 14,864 monomer IDs and **14,832 RDKit-valid structures**. Endpoint counts are 15,753 IC50, 2,847 Ki, 110 EC50, and 49 Kd; 5,437 values are explicitly censored. **1,469 structures are absent from local ChEMBL 37**, but they remain delta candidates because the public API omits assay conditions, row curator, and inherited source-license fields.
3. **Zenodo ToxTree release 5807719:** CC BY 4.0, 8,879 pIC50 rows. Its published MD5 was verified. It is mostly redundant: 8,068/8,877 structures overlap local ChEMBL and 8,617 overlap the later Zenodo 8359714 release; only 260 structures are absent from that later release.

## Complete PubChem KCNH2 universe

Official target-centric PubChem queries found **4,067 AIDs by GeneSymbol KCNH2** and **4,059 by GeneID 3757**. All 4,067 summaries were retained in 28 batches. The source breakdown is:

- ChEMBL mirrors: 3,878 AIDs.
- BindingDB mirrors: 127 AIDs.
- Other: 62 AIDs.

The 62 “other” records are dominated by genome-wide RNAi assays where KCNH2 is merely one screened gene and broad counter-screen panels where hERG is one target. The only large, clear, previously missing compound-hERG assay was Tox21 AID 1671200. AID 652075 is a one-compound hERG counterscreen; AID 1346735 is three rat Kv11.1 records. These were retained as metadata, not inflated into new corpus claims. PubChem access followed its documented target-centric PUG REST interface ([official documentation](https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest)).

## Overlap warnings

- Tox21 AID 1671200 shares 2,992 CIDs with AID 588834 but only eight SIDs, showing why SID-only reconciliation is insufficient. It shares 1,510 CIDs and zero SIDs with AID 720551.
- BindingDB overlaps local ChEMBL on 13,363 standardized structures. The apparent 1,469-structure delta cannot be admitted from the compact API alone because assay descriptions, measurement provenance, and curator/license lineage are missing.
- ToxTree's raw SMILES appear almost different from Zenodo 8359714, but RDKit canonicalization exposes 8,617 shared structures. Exact-string deduplication would badly overstate novelty.
- Tox21 and BindingDB share only 97 standardized structures, so the two new acquisitions largely add different chemical coverage.

## Rights and blockers

- PubChem provides depositor records openly but does not transfer contributor rights. Preserve Tox21/NCATS attribution and keep raw assay semantics.
- BindingDB's live export policy distinguishes BindingDB-curated data from inherited ChEMBL data. The compact public API does not expose that distinction per row. A registered assay-rich TSV/SDF export or curator-provenance mapping is required before admission.
- The original hERGCentral database still lacks a located authoritative current snapshot with an explicit dataset license. HERGAI and TDC/hERGCentral-style files remain derivative wrappers, not substitutes for cleared source provenance ([original hERGCentral paper](https://doi.org/10.1089/adt.2011.0425)).
- The 2018 integrated database includes commercial GOSTAR data, so the article's open license does not clear redistribution of the underlying mixed dataset ([integration study](https://doi.org/10.1371/journal.pone.0199348)).
- Figshare, Dryad, Harvard Dataverse, OSF/institutional repositories, Zenodo, GitHub, and Hugging Face were searched. No additional dedicated source-grade release justified acquisition beyond the three above; the ledger records wrappers, existing sources, rights blocks, and negative searches.

## Verification and next admission gate

The round contains 41,686,934 bytes. The Tox21 counts reconcile exactly to PubChem's official 9,667-SID summary; the BindingDB row count reconciles to its live Q12809 target page; Zenodo's MD5 matches exactly. JSON reports parse, and the sorted raw-file hash list is bound by SHA-256 `1bf0660f306a248e8bb21cb8289129b57a32d1ad067408390bd593b57cc298f8`.

Before canonical use: obtain the full BindingDB curator/assay export, standardize structures without overwriting raw strings, reconcile Tox21/NCATS by CID and parent structure, retain replicate/curve-class evidence, and partition fluorescence qHTS separately from patch-clamp, binding, and fixed-dose endpoints.
