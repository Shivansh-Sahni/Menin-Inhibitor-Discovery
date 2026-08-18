# EPA invitrodb v4.3 hERG/KCNH2 reconciliation (HREG-4)

Date: 2026-08-07  
Disposition: metadata admitted to the audit only; no bioactivity rows admitted to canonical/training data.

## Decision

EPA invitrodb v4.3 contains **three** human KCNH2 endpoints. It is not a single new hERG dataset:

| Source lineage | aid / acid / aeid | Exact endpoint | Experimental evidence | Relationship to the existing inventory | Disposition |
|---|---:|---|---|---|---|
| NovaScreen (NVS) | 237 / 423 / 686 | `NVS_IC_hKhERGCh` | Cell-free CHO extract, `[3H]-Astemizole` radioligand displacement, filter radiodetection; single-concentration screen in duplicate at 25 uM, with selected actives/negatives followed in an eight-point series | Independent assay lineage from Tox21. No exact named ToxCast/NVS assay was found in local ChEMBL 37. ChEMBL and BindingDB do contain many astemizole-binding hERG records, so row/value novelty is **not** established from protocol similarity alone. | Valuable independent binding endpoint; quarantine until exact rows, provenance and structure-normalized overlap are counted. |
| Eurofins Taiwan (ERF) | 897 / 2914 / 3184 | `ERF_TW_IC_hKCNH2` | Cell-free membranes from recombinant human HEK-293, 3 nM `[3H]-Dofetilide`, 60 min at 25 C, filter/scintillation binding; nominal single screen at 10 uM and dose response where present | EPA's v4.3 release note explicitly lists this endpoint among both the **38 new multi-concentration** and **58 new single-concentration** endpoints. ChEMBL 37 contains small Eurofins/Panlabs SafetyScreen hERG assays CHEMBL5442367 and CHEMBL5463762 (10 and 7 local activity rows); their identity with EPA-screened samples is unproven. | Strongest likely new measurement source; exact record delta remains fail-closed pending database extraction. |
| Tox21 | 910 / 2939 / 3210 | `TOX21_hERG_U2OS_Antagonist` | hERG-U2OS, FluxOR II thallium-flux fluorescence, 1536 wells, 10 min exposure, FDSS 7000EX, triplicate; EPA metadata says 7,871 unique chemicals | Same experimental lineage as already acquired PubChem AID 1671200 (`HERG915`): same U2OS cell line, FluxOR II, thallium surrogate, astemizole control, 10 min exposure, 1536-well format and FDSS reader. PubChem has 9,667 SIDs/7,671 CIDs because sample/substance representations differ from the publication's 7,871 unique chemicals. EPA tcpl reprocessing is not a new experiment. | Treat as a cross-pipeline mirror; retain only for fit/QC comparison and DSSTox mapping, not as independent evidence. |

Therefore, the defensible answer is **yes, invitrodb adds two non-Tox21 hERG assay lineages (NVS and ERF), but only ERF is newly released in v4.3; exact novel compounds and measurements cannot be claimed from metadata alone**. The Tox21 endpoint is a duplicate measurement lineage already captured at source grade from PubChem.

## Why this is defensible

The official v4.3 annotation workbook has 1,647 endpoint rows and only three rows mapped by the official target workbook to human Entrez Gene 3757 / `KCNH2`. All three are marked export-ready, usable, percent-activity, loss-of-signal endpoints. EPA's release note says v4.3 added 77 endpoints and 873 chemicals overall, and specifically exposes ERF hKCNH2 in both its new-MC and new-SC tables. It does not list NVS or Tox21 hERG as new endpoints.

The Tox21 identity match is experimental, not name-based. The official PubChem description for AID 1671200 and EPA's endpoint documentation independently specify the same platform and protocol. Different counts, identifiers, curve fits or activity calls between NCATS/PubChem and EPA/tcpl must be treated as representation/analysis differences until proven otherwise.

The two binding assays are biologically complementary to the Tox21 functional flux assay, but neither is equivalent to patch clamp. They should remain separate assay strata in modeling and evaluation. A binding hit may not imply functional current block at the same potency, and the legacy NVS design selectively dose-responses primary hits rather than testing every negative in a complete curve.

## Exact post-download gate

Run [`epa_invitrodb_v4_3_postload.sql`](./epa_invitrodb_v4_3_postload.sql) only after the official 17,651,807,605-byte gzip finishes and passes integrity validation. It will:

1. verify the schema and the three identity triplets;
2. count multi- and single-concentration rows, SPIDs, chemical IDs and DTXSIDs per endpoint;
3. count EPA active/inactive/unintended-direction calls using the v4.3 rules;
4. export all endpoint-result fields plus sample/chemical identifiers;
5. quantify within-EPA DTXSID overlap.

For the cross-source delta, resolve exported DTXSIDs to standardized parent structures, compare parent InChIKey to PubChem AID 1671200, local ChEMBL 37 and the BindingDB Q12809 acquisition, and separately match measurement lineage (source, protocol, endpoint, concentration/value and reference). Do not deduplicate by CAS, name, raw SMILES, vendor name or protocol alone. Do not count EPA's Tox21 rows as independent labels.

## Rights and provenance

The official Figshare v14 record is CC0 and identifies EPA CCTE as author. The Clowder file pages do not repeat a file-level license; preserve the Figshare release citation and source lineage with every derivative. EPA publishes the release as invitrodb v4.3/August 2025; Figshare version 14 was posted 2025-09-03. The assay workbooks are named `AUG2024` but were uploaded 2025-08-27 as v4.3 summary artifacts; record both artifact name and release version to avoid date ambiguity.

## Acquired evidence and verification

- Official annotation workbook SHA-256: `eecbdac47fe4b4c3db765d412d15526331c269d36c7d79dde938f59ea60461f9` (matches Clowder's published hash).
- Official target-mapping workbook SHA-256: `bf45c47dc383c16aea9c4cf92993f7034a3141a252aa02d9f7aa7216ac84c394` (matches Clowder's published hash).
- Official v4.3 release note SHA-256: `7dd695e855cd385ff9afe0a6b6f544418554bf87dd1ed774c96fb8b008c05d13`.
- Official MySQL README SHA-256: `2cc2387b246743e2c4e7e7d83e766aa3c3ed6b32b107f9174dde30cfd158ee29`; both pages were rendered and visually checked. It states the dump exceeds half a billion records and should be loaded into MySQL; it does not provide hERG counts.
- The downloaded workbooks' misleading `A1` dimensions were handled by resetting worksheet dimensions before iterating; physical XML contains all 1,647 endpoint records.

## Official sources

- [EPA Exploring ToxCast Data](https://www.epa.gov/comptox-tools/exploring-toxcast-data) - release scope, August 2025 version, downloads and API status.
- [EPA Figshare invitrodb v4.3, version 14](https://epa.figshare.com/articles/dataset/ToxCast_Database_invitroDB_/6062623/14) - authoritative release record and CC0 declaration.
- [EPA invitrodb v4.3 Clowder space](https://clowder.edap-cluster.com/spaces/687e388ce4b02565bc3e28e4) - database, summary, assay and release-note artifacts.
- [EPA ctxR Bioactivity vignette](https://usepa.github.io/ctxR/articles/Bioactivity.html) - definition of AEID/M4ID/SPID retrieval and active/total summaries.
- [PubChem AID 1671200](https://pubchem.ncbi.nlm.nih.gov/bioassay/1671200) - source-grade Tox21 hERG assay already acquired.
- [Tox21 hERG study](https://pmc.ncbi.nlm.nih.gov/articles/PMC8869358/) - published FluxOR/U2OS experiment and 7,871-chemical study description.

## Fail-closed conclusion

Admit no EPA bioactivity records yet. After the dump is complete, prioritize ERF 3184, then NVS 686, and use Tox21 3210 only to reconcile EPA/NCATS processing. Report exact new structures and measurement rows only after parent-structure and source-lineage matching; until then, describe ERF and NVS as **candidate independent evidence**, not a quantified corpus expansion.
