# EPA invitrodb v4.3 hERG exact streaming extraction (HREG-5)

Date: 2026-08-07  
Status: extraction and independent validation passed, with eight explicitly unmapped control SPIDs  
Disposition: audit-only; zero canonical rows and zero training labels admitted

## Outcome

The complete 17,651,807,605-byte EPA invitrodb v4.3 dump was streamed without loading MySQL or modifying the lead archive. The parser consumed the entire gzip, reproduced its publisher SHA-256 `ee159e1cdd28996f85db13e742700d8d76ef9d5baf31e3b5e00d249899529c7b`, and therefore also exercised the gzip CRC. It recovered source column order from each `CREATE TABLE` statement, parsed every `chemical`, `sample`, `mc4`, `mc5`, and `sc2` tuple with exact field-count checks, and retained only AEIDs 686, 3184, and 3210 plus their linked identities.

## Exact endpoint counts

“Mapped rows” below have a matching EPA `sample` record; activity categories are sample-row hit calls, not unique-compound labels. Negative hit calls are responses in the unintended direction under v4.3 processing.

| Endpoint | MC rows / mapped | Mapped MC active / inactive / negative | SC rows / mapped | Mapped SC active / inactive / negative | Distinct DTXSIDs, MC / SC / union |
|---|---:|---:|---:|---:|---:|
| NVS `NVS_IC_hKhERGCh` (686) | 95 / 93 | 37 / 37 / 19 | 1,124 / 1,120 | 52 / 1,040 / 28 | 93 / 1,077 / **1,077** |
| ERF `ERF_TW_IC_hKCNH2` (3184) | 22 / 22 | 10 / 11 / 1 | 119 / 117 | 20 / 97 / 0 | 22 / 111 / **131** |
| Tox21 `TOX21_hERG_U2OS_Antagonist` (3210) | 9,670 / 9,667 | 1,280 / 6,077 / 2,310 | 0 / 0 | 0 / 0 / 0 | 7,871 / 0 / **7,871** |

All source-row arithmetic reconciles. Including controls, MC activity counts are 38/38/19 for NVS, 10/11/1 for ERF, and 1,281/6,079/2,310 for Tox21. NVS has 28 negative single-concentration calls; these must not be silently collapsed into ordinary inactive labels. Every one of the 9,787 retained `mc4` rows has exactly one `mc5` row: no missing, duplicate, or orphan target M4IDs were found.

Within an endpoint, all 93 NVS MC DTXSIDs also occur in its SC screen. ERF MC and SC share two DTXSIDs, yielding 22 + 111 - 2 = 131. Tox21 has no SC rows.

## Within-EPA chemical overlap

| Endpoint membership | Distinct DTXSIDs |
|---|---:|
| Tox21 only | 6,785 |
| NVS + Tox21 | 1,030 |
| ERF + Tox21 | 22 |
| NVS + ERF + Tox21 | 34 |
| NVS only | 13 |
| ERF only | 75 |
| NVS + ERF but not Tox21 | 0 |
| **Union** | **7,959** |

Pairwise overlap is 1,064 DTXSIDs for NVS–Tox21, 56 for ERF–Tox21, and 34 for NVS–ERF; all 34 NVS–ERF overlaps are also in Tox21. Thus NVS covers 98.79% of its 1,077 DTXSIDs in the EPA Tox21 assay, while ERF covers 42.75% of its 131 DTXSIDs there. Relative to the EPA Tox21 chemical set, NVS contributes 13 and ERF contributes 75 additional resolved DTXSIDs, for an exact within-EPA increment of **88**.

The full 7,959-row membership matrix is `endpoint_dtxsid_membership.csv`; it makes these inclusion/exclusion calculations reproducible.

## PubChem and ChEMBL reconciliation

The Tox21 result closes the main lineage question. EPA has 9,670 MC rows: three named controls (`Astemizole`, `Blank`, and `DMSO`) lack `sample` rows, leaving exactly **9,667 mapped sample rows**. That equals PubChem AID 1671200’s 9,667 SIDs exactly. EPA’s **7,871 DTXSIDs** also equals the published Tox21 study’s 7,871 unique chemicals. The EPA and PubChem activity classifications differ because EPA/tcpl and NCATS/PubChem apply different fitting and hit-call logic; they are not independent experiments. Therefore AEID 3210 contributes **zero independent experimental rows** beyond the already acquired PubChem assay.

EPA DTXSID-to-Tox21 overlap is exact internally, but the dump’s `chemical` table contains DTXSID, CAS, and name rather than structure. It is not defensible to claim that the 13 NVS-only or 75 ERF-only DTXSIDs are absent from ChEMBL 37 or BindingDB using CAS/name matching. Local ChEMBL has related Eurofins/Panlabs hERG assays CHEMBL5442367 and CHEMBL5463762 with 10 and 7 activity rows, respectively, but their identity with EPA’s 131 ERF DTXSIDs remains unproven. Cross-source novelty therefore stays fail-closed pending a structure-resolved DTXSID crosswalk and measurement-lineage comparison.

## Source identity gaps

Eleven retained endpoint rows use eight SPIDs that do not occur in EPA’s `sample` table. They are all recognizable assay controls, not silently missing test chemicals:

- NVS: `DMSO`, `Terfenadine`, `5-iodotubercidin`, and `NSC-0012` across six rows.
- ERF: `Dofetilide` and `DMSO - Maximum Control` across two rows.
- Tox21: `Astemizole`, `Blank`, and `DMSO` across three rows.

The rows and hit calls are preserved in `mc4`/`mc5`/`sc2`, their identities are not imputed, and they are excluded from DTXSID overlap. The exact occurrences are in `unresolved_sample_rows.csv`.

## Parser and validation proof

The stream inspected 10,487 chemical rows, 295,445 sample rows, 3,527,295 `mc4` rows, 3,527,295 `mc5` rows, and 523,472 `sc2` rows. It parsed 1,179 MC4, 471 MC5, and 78 SC2 extended-INSERT statements, retaining 9,787 MC4, 9,787 MC5, and 1,243 SC2 rows. There were zero tuple, schema, Unicode, field-count, or gzip errors.

An independent validator then:

1. verified all nine extractor-produced artifact hashes;
2. compared every CSV cell and row against the SQLite audit database;
3. reconciled all active/inactive/negative arithmetic;
4. confirmed zero unresolved CHIDs and the 11 control-row identity exceptions;
5. reconciled the three-way DTXSID union by inclusion–exclusion to 7,959;
6. confirmed the exact extractor source hash recorded at run time.

## Reproducible artifacts

- `stream_extract_invitrodb_herg.py`: one-pass, fail-closed extractor.
- `validate_invitrodb_herg_extract.py`: independent content, hash, and arithmetic validator.
- `invitrodb_v4_3_herg_audit.sqlite`: compact relational audit database.
- `herg_mc4.csv`, `herg_mc5.csv`, `herg_sc2.csv`: exact retained result rows.
- `herg_sample.csv`, `herg_chemical.csv`: linked identity rows only.
- `endpoint_dtxsid_membership.csv`: exact three-endpoint DTXSID membership.
- `audit_summary.json`, `independent_validation.json`: machine-readable extraction and validation results.
- `SHA256SUMS.txt`: extractor-produced artifact hashes.

Raw outputs are under `research/data/platform/raw/external_public/herg_expansion/anscombe_round2/epa_invitrodb_v4_3_stream/`. No file in `high_value_expansion/lead`, any canonical dataset, registry, split, or training directory was changed.

## Sources

- [EPA invitrodb v4.3 release record, version 14](https://epa.figshare.com/articles/dataset/ToxCast_Database_invitroDB_/6062623/14)
- [EPA ToxCast downloadable-data documentation](https://www.epa.gov/comptox-tools/exploring-toxcast-data)
- [EPA ctxR Bioactivity documentation](https://usepa.github.io/ctxR/articles/Bioactivity.html)
- [PubChem Tox21 hERG AID 1671200](https://pubchem.ncbi.nlm.nih.gov/bioassay/1671200)
- [Published Tox21 hERG FluxOR/U2OS study](https://pmc.ncbi.nlm.nih.gov/articles/PMC8869358/)

## Admission decision

Do not admit rows yet. Treat NVS and ERF as source-grade candidate independent binding evidence after structure-resolved ChEMBL/BindingDB reconciliation. Treat Tox21 as an analysis mirror of PubChem AID 1671200, useful for EPA-specific hit calls, QC, and DTXSID mapping but not an independent label source. Preserve assay modality, SC/MC regime, negative hit-call state, sample lineage, and controls separately.
