# Public hERG expansion: acquisition and admission audit

Date: 2026-08-06  
Scope: public hERG artifacts beyond the local ChEMBL collection. All files remain in raw external storage. Nothing was added to canonical data, labels, splits, or training.

## Outcome

Four version-pinned candidates were acquired with source metadata and license evidence. The useful raw footprint is 27,785,866 bytes. The largest source, HERGAI, contributes 299,927 labeled records; the most scientifically usable quantitative source contributes 22,969 nonoverlapping hERG pIC50 records. None is approved for automatic canonical admission yet.

## 1. HERGAI

Source: [immutable GitHub revision](https://github.com/vktrannguyen/HERGAI/tree/21f1b0ef34ab8c818015f1ac6bdfd6c8e1bff351); paper: [Tran-Nguyen et al., 2025](https://doi.org/10.1186/s13321-025-01063-8).

- Training: 224,945 rows; 224,945 unique SIDs; 223,590 unique exact SMILES; 1,453 Active and 223,492 Inactive.
- Test: 74,982 rows; 74,982 unique SIDs; 74,813 unique exact SMILES; 484 Active and 74,498 Inactive.
- Combined: 299,927 rows/SIDs, 298,402 exact-SMILES union, 1,937 Active and 297,990 Inactive (0.646% active). This is extremely imbalanced.
- Potency is the literal string `NA` for 223,157 training and 74,430 test rows. Only 2,340/299,927 rows have numeric potency; 403 of those are labeled Inactive. Treat potency as sparse, not complete regression ground truth.
- Internal duplication: training contains 1,262 repeated-SMILES groups (2,617 rows), including two groups with conflicting activity labels; test contains 163 repeated-SMILES groups (332 rows).
- Split leakage: exactly one SMILES appears in both splits, `C1CCN(C1)C(=S)[S-]`. It is Inactive in training (SID 57264559, potency `NA`) but Active in test (SID 50106934, potency 5.45000414846935). Any benchmark must remove or reconcile it.
- Byte integrity: both local Git-object SHA-1 values exactly match GitHub's pinned tree; SHA-256 values are in `artifact_manifest.json`.
- Rights: the repository includes an MIT LICENSE, while its README says the data and source code are “subject to copyright.” The paper/repository do not itemize the rights lineage of every upstream hERGCentral/PubChem record. Safe status: internal raw audit only until attribution and redistribution policy are reviewed.

## 2. CC BY ChEMBL + hERGCentral union

Source: [Zenodo 6334912](https://doi.org/10.5281/zenodo.6334912), the latest version resolved from concept DOI 10.5281/zenodo.6334721. Zenodo metadata explicitly declares CC BY 4.0.

- 7,879 rows, 7,879 unique exact SMILES, no missing cells or exact duplicate rows.
- Class balance: class 0 = 4,127; class 1 = 3,752.
- Blocker: only `smiles`, `class`, and `MW` are retained. The record says the file unites ChEMBL and hERGCentral, but provides neither per-row source nor the class threshold/direction. Class 0/1 must not be mapped to inactive/active without authoritative documentation.
- Published MD5 `34e7ade08e88dedcbe69ff544404e376` was verified exactly.

## 3. Curated quantitative three-channel release

Source: [Zenodo 8359714](https://doi.org/10.5281/zenodo.8359714), CC BY 4.0; associated paper: [Arab et al., JCIM](https://doi.org/10.1021/acs.jcim.3c01301).

- hERG development set: 22,246 rows, all unique by InChIKey and exact SMILES; Train 17,796, Validation 4,450; no missing pIC50.
- hERG external sets: 250 rows at the stated 0.60 similarity limit and 473 at 0.70; each has unique InChIKeys/SMILES and no missing pIC50.
- No exact SMILES or InChIKey overlap among the three supplied hERG splits.
- Development sources: ChEMBL 15,624; Didziapetris et al. 2,372; PubChem 2,081; US Patent 1,238; Konda et al. 463; Doddareddy et al. 417; Munwar et al. 36; BindingDB 15.
- Strong source duplication: 7,591/7,879 (96.35%) exact SMILES from the smaller Zenodo union reappear in this development set. They must not be counted as independent observations.
- Archive MD5 exactly matches the published value. Admission still requires assay-level provenance, censoring/relation handling, unit verification, and deduplication against local ChEMBL/BindingDB.

## 4. ToxiMol hERGCentral subset

Source: [pinned Hugging Face revision](https://huggingface.co/datasets/DeepYoke/ToxiMol-benchmark/tree/dedfdb2f898c0d8f30b35b03d604159179655c5f/herg_central), MIT license.

- 60 rows and 60 unique SMILES; schema is `task`, `id`, `smiles`, `image`.
- It contains molecular images but no measured hERG label, potency, assay, or source-row identifier. It is useful only as a benchmark/reference set, not as training evidence.
- The dataset card describes 660 molecules across 11 tasks; the hERGCentral configuration is the 60-row subset inspected here.

## Scientific admission rules justified by the literature

- hERGCentral was designed to combine compound-channel measurements and cross-links, not to guarantee one homogeneous assay ([Du et al., 2011](https://doi.org/10.1089/adt.2011.0425)).
- Integrated hERG studies explicitly report large assay/source heterogeneity and warn that protocol differences can change blocker/non-blocker classification ([Sato et al., 2018](https://doi.org/10.1371/journal.pone.0199348)).
- Therefore retain source, assay technology, cell system, endpoint, relation/censoring, units, concentration, and original identifier before harmonization. Keep continuous measurements separate from threshold-derived binary labels, deduplicate by standardized parent structure plus measurement provenance, and construct structure/scaffold/time splits only after conflict resolution.

## Final blockers and next safe action

1. Do not concatenate these files. Exact duplication is already proven, and standardized-structure overlap will be larger.
2. Obtain authoritative class semantics for Zenodo 6334912 and upstream rights clarification for HERGAI.
3. Normalize structures without overwriting raw strings; preserve salts/stereochemistry mappings and standardization version.
4. Map each quantitative row to assay/source provenance, convert units with relation-aware censoring, then resolve conflicts explicitly.
5. Rebuild leakage-resistant splits after deduplication. Keep all three downloaded source-provided splits as provenance fields, not immutable model partitions.

The machine-readable hashes, versions, row counts, and admission statuses are in `artifact_manifest.json`. Exact upstream metadata and license/README snapshots are stored beside each raw artifact.
