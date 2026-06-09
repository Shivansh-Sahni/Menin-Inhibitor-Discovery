# Methodology

## Project Interpretation

The project is treated as a computational drug-discovery workflow for Menin/MEN1 inhibitors. The repository is organized around three modeling surfaces:

1. Menin activity prediction from public biochemical and interaction assays.
2. hERG/KCNH2 liability prediction from public hERG ChEMBL activity data.
3. PK/ADMET observation extraction for Menin-associated molecules, with endpoint-specific modeling deferred until there is enough clean data.

## Target Anchors

- Menin/MEN1: ChEMBL `CHEMBL1615381`, UniProt `O00255`
- hERG/KCNH2: ChEMBL `CHEMBL240`, UniProt `Q12809`

These identifiers are used instead of only text search so the collection can be rerun reproducibly.

## Public Data Acquisition

ChEMBL:

- Target activity collection uses the ChEMBL REST API.
- Menin target rows are saved to `data/raw/chembl/chembl_menin_activities.csv`.
- hERG target rows are saved to `data/raw/chembl/chembl_herg_activities.csv`.
- All ChEMBL activities for Menin-associated ChEMBL molecules are saved to `data/raw/chembl/chembl_menin_molecule_all_activities.csv` and filtered for PK/ADMET keywords.

BindingDB:

- Menin and Menin/KMT2A TSV exports are downloaded from BindingDB target-index files.
- Ki, IC50, Kd, and EC50 columns are melted into the common measurement schema.

PubChem:

- PubChem BioAssay is searched using MEN1/menin/menin-MLL terms.
- Assay metadata, download status, and available assay CSVs are saved under `data/raw/pubchem/`.
- PubChem unit handling is explicit: `Standard Value` plus `Standard Units` is preferred; endpoint-specific values such as `IC50` use the PubChem `RESULT_UNIT` metadata; `PubChem Standard Value` is treated as micromolar when used as fallback.

## Curation Rules

Measurement-level rows are normalized into:

- `source`
- `source_record_id`
- `compound_id`
- `smiles`
- `target_name`
- `target_id`
- `endpoint`
- `relation`
- `value_raw`
- `standard_units`
- `value_nm`
- `p_value`
- assay and document metadata

Core bioactivity endpoints are `IC50`, `Ki`, `Kd`, and `EC50`.

The p-value transform is:

```text
pActivity = 9 - log10(value_nM)
```

Compound-level Menin aggregation uses exact values when exact values are available. For each SMILES string, the table stores median pActivity, best pActivity, median nM value, best nM value, sources, endpoints, and potency labels.

Potency labels:

- `high_potency_<=100nM`
- `moderate_100nM_to_1uM`
- `weak_1uM_to_10uM`
- `low_or_inactive_>10uM`

## hERG Liability Labels

The hERG compound table is built from ChEMBL hERG/KCNH2 activities.

Coarse label policy:

- Positive hERG blocker: median hERG activity `<= 10 uM`
- Negative/non-blocker: median hERG activity `>= 30 uM`
- Ambiguous middle band `10-30 uM` is omitted from classifier training

This is a first-pass public-data policy, not a substitute for the lab's preferred hERG assay and decision threshold.

## Baseline Models

The current environment does not include RDKit, so the baseline uses:

- hashed SMILES character n-grams
- simple SMILES string descriptors
- Ridge regression for Menin pActivity
- logistic regression for hERG liability

This is intentionally portable. The next serious modeling iteration should add RDKit Morgan fingerprints, physicochemical descriptors, scaffold splits, and uncertainty estimates.
