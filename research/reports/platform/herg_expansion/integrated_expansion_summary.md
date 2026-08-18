# Integrated public hERG expansion summary

Date: 2026-08-06  
Status: raw acquisition and source audit complete; canonical admission pending.

## Outcome

The project is no longer limited to the 41,078-row local ChEMBL hERG view. The
raw, version-pinned inventory now contains more than 700,000 wild-type or
standard hERG result rows across ChEMBL, HERGAI, official PubChem/NCATS assays,
and smaller quantitative/benchmark releases. A separate PubChem mutant-channel
assay contributes 342,311 additional rows and is not pooled with wild type.

These counts are deliberately not added together as unique compounds. HERGAI,
PubChem, ChEMBL, hERGCentral-derived releases, and TDC wrappers overlap heavily.
The best current planning estimate is approximately 300,000–350,000 unique
compound structures after standardization and deduplication.

## Principal inventories

| Inventory | Rows | Important detail |
| --- | ---: | --- |
| Local ChEMBL 37 hERG | 41,078 | 26,254 unique InChIKeys; 4,829 assays |
| HERGAI | 299,927 | 298,402 exact-SMILES union; 1,937 active, 297,990 inactive |
| PubChem AID 720551, wild type | 343,909 | 343,666 unique CIDs; 1,267 active |
| PubChem AID 588834 | 5,381 | 4,743 unique CIDs; thallium-flux assay |
| Quantitative three-channel release | 22,969 | pIC50 records across supplied nonoverlapping splits |
| PubChem AID 720553, mutant | 342,311 | Separate biology; never pooled with wild type |

Smaller TDC and Zenodo files were retained for benchmarking, provenance, and
overlap analysis. They must not be represented as independent new evidence.

## Lead reconciliation decision

1. Raw artifacts, official descriptions, licenses, revisions, URLs, and hashes
   are retained under `research/data/platform/raw/external_public/herg_expansion`.
2. No incoming row was silently added to canonical labels or model splits.
3. Wild-type, mutant, patch-clamp/electrophysiology, flux, binding, fixed-dose
   inhibition, and sparse binary calls remain separate evidence modalities.
4. HERGAI provides immediate scale for internal benchmarking, but upstream
   rights and extreme class imbalance require explicit treatment.
5. PubChem AID 720551 is the largest official source-grade acquisition. Its
   overlap with HERGAI and local ChEMBL must be measured using standardized
   parent structures before a canonical count can be claimed.
6. The next mechanical step is a deterministic structure-normalization,
   source-overlap, label-conflict, and assay-modality admission pipeline.

## Verification

All three nonoverlapping acquisition workstreams were reviewed by the lead.
Their JSON manifests parse, reported row totals reconcile, official PubChem
outcome subtotals sum to the acquired result counts, and the retained files are
hash-bound. The inventory remains evidence-grade raw data rather than a clean
training set.
