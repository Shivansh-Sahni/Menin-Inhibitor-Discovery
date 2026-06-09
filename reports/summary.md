# Menin Inhibitor Discovery Preliminary Work

Generated: 2026-06-09 00:46 UTC

## Executive summary

This repo builds a reproducible public-data foundation for menin inhibitor modeling. It anchors target activity on human Menin/MEN1 (CHEMBL1615381, UniProt O00255), adds BindingDB Menin and Menin/KMT2A TSV exports, searches PubChem BioAssay for MEN1/menin assays, and builds baseline ML-ready tables for activity, PK/ADMET observations, and hERG liability.

## Current dataset snapshot

- Menin measurement rows: 3,929
- Curated unique Menin SMILES strings: 1,634
- hERG/KCNH2 measurement rows: 41,078
- hERG/KCNH2 curated unique SMILES strings: 11,278
- Menin-molecule PK/ADMET observation rows: 283

### Source coverage

| source | n_measurements | n_compounds |
| --- | --- | --- |
| ChEMBL | 1776 | 689 |
| BindingDB | 1126 | 936 |
| PubChem | 1027 | 416 |

### Potency label counts

| potency_class | count |
| --- | --- |
| high_potency_<=100nM | 751 |
| weak_1uM_to_10uM | 303 |
| moderate_100nM_to_1uM | 292 |
| low_or_inactive_>10uM | 288 |

## Highest-potency public compounds in the curated table

| compound_ids | value_nm_median | p_activity_median | n_measurements | endpoints | sources |
| --- | --- | --- | --- | --- | --- |
| 71549453.0 | 0.05 | 10.3 | 1 | Ki | PubChem |
| 689587 | 0.1 | 10 | 2 | IC50 | BindingDB |
| CHEMBL5267644 | 0.104 | 9.98 | 1 | Ki | ChEMBL |
| 689589;689590 | 0.139 | 9.88 | 2 | IC50 | BindingDB |
| CHEMBL6053681 | 1 | 9 | 1 | IC50 | ChEMBL |
| 506357 | 1 | 9 | 1 | IC50 | BindingDB |
| 50511905 | 1 | 9 | 1 | IC50 | BindingDB |
| CHEMBL4530300 | 1 | 9 | 1 | IC50 | ChEMBL |
| CHEMBL4211366 | 1 | 9 | 1 | IC50 | ChEMBL |
| 50454121 | 1 | 9 | 1 | IC50 | BindingDB |
| 71549452.0 | 1.2 | 8.92 | 1 | Ki | PubChem |
| 656643 | 1.25 | 8.9 | 1 | Ki | BindingDB |
| 656820 | 1.25 | 8.9 | 1 | Ki | BindingDB |
| 656647 | 1.25 | 8.9 | 1 | Ki | BindingDB |
| 656824 | 1.25 | 8.9 | 1 | Ki | BindingDB |

## Baseline model status

### Menin activity regression

```json
{
  "status": "trained",
  "model": "hashed-SMILES Ridge regression",
  "n_compounds": 1634,
  "n_train": 1307,
  "n_test": 327,
  "best_alpha": 0.1,
  "test_mae_pchembl": 0.6558959494072406,
  "test_rmse_pchembl": 0.80483036146578,
  "test_r2": 0.6894210227369341,
  "interpretation": "A fast baseline for triage only; RDKit fingerprints and scaffold split are next-step upgrades."
}
```

### hERG liability classifier

```json
{
  "status": "trained",
  "model": "hashed-SMILES logistic regression",
  "n_compounds": 8808,
  "n_train": 7046,
  "n_test": 1762,
  "positive_label": "hERG activity <=10 uM",
  "negative_label": "hERG activity >=30 uM",
  "best_C": 3.0,
  "test_roc_auc": 0.825543120473996,
  "test_balanced_accuracy": 0.7529401246759158,
  "interpretation": "Coarse liability screen; experimental hERG assay labels and applicability-domain checks are needed before decision use."
}
```

## How to interpret the models

The current models are intentionally dependency-light baselines using hashed SMILES character n-grams plus simple string descriptors. They are useful for pipeline validation, triage, and demonstrating the full workflow. They should not be treated as final medicinal chemistry decision models until the next steps are complete: RDKit descriptors/fingerprints, scaffold/time splits, assay-family stratification, uncertainty estimates, and prospective validation.

## Immediate next steps for the Wang lab

1. Add internal compound IDs, SD files, and assay metadata under the same schema.
2. Confirm which endpoint should be the primary model target: biochemical FP/HTRF IC50, Kd/Ki, cell EC50, or a weighted multitask formulation.
3. Replace string-features with RDKit Morgan fingerprints, physicochemical descriptors, and optional graph neural network embeddings.
4. Build scaffold-split and time-split evaluations to estimate prospective performance.
5. Expand hERG labels with lab-preferred thresholds and assay-specific annotations.
6. Turn PK/ADMET observations into endpoint-specific models only when each endpoint has enough clean, comparable rows.

## Questions to clarify with the lab

- Are internal menin compounds available as SDF/SMILES with exact batch IDs?
- Which assay should define the official activity label when multiple public measurements disagree?
- Should censored values such as `>10000 nM` be retained for classification, regression with censoring, or excluded from the first model?
- Does the lab want prospective design suggestions, virtual-screening rank lists, or only data/model infrastructure first?

## Key public-source anchors

- ChEMBL Menin target: CHEMBL1615381 / UniProt O00255
- ChEMBL hERG/KCNH2 target: CHEMBL240 / UniProt Q12809
- BindingDB target-index TSVs for Menin and Menin/KMT2A complex
- PubChem BioAssay search terms in `src/menin_discovery/config.py`
