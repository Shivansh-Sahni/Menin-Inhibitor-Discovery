# Data Dictionary

## `data/processed/menin_activity_measurements.csv`

Measurement-level public Menin activity data.

| Column | Meaning |
| --- | --- |
| `source` | Public database source: ChEMBL, BindingDB, or PubChem |
| `source_record_id` | Source-specific activity or assay-row identifier |
| `compound_id` | Source-specific molecule identifier |
| `compound_name` | Source molecule name when available |
| `smiles` | Structure string used for modeling |
| `inchi_key` | InChIKey when available from source |
| `target_name` | Source target name |
| `target_id` | Source target identifier |
| `endpoint` | Normalized endpoint, usually IC50, Ki, Kd, or EC50 |
| `relation` | Assay qualifier such as `=`, `<`, or `>` |
| `value_raw` | Original parsed numeric-like value |
| `standard_units` | Units used before nM conversion |
| `assay_description` | Source assay description |
| `assay_type` | Source assay type or activity outcome |
| `document_id` | Source document, patent, DOI, or AID identifier |
| `document_year` | Publication/document year when available |
| `reference` | Source reference field |
| `source_detail` | Additional provenance detail |
| `value_nm` | Activity converted to nM where possible |
| `p_value` | `9 - log10(value_nm)` |
| `is_exact` | True for exact values rather than censored values |
| `is_core_endpoint` | True for IC50, Ki, Kd, or EC50 |

## `data/processed/menin_compounds_curated.csv`

Compound-level Menin modeling table aggregated by SMILES.

| Column | Meaning |
| --- | --- |
| `smiles` | Compound structure |
| `n_measurements` | Number of exact core measurements contributing to the aggregate |
| `n_sources` | Number of source databases represented |
| `p_activity_median` | Median pActivity for the compound |
| `p_activity_best` | Highest pActivity for the compound |
| `value_nm_median` | Median nM activity |
| `value_nm_best` | Best/lower nM activity |
| `endpoints` | Endpoints represented |
| `sources` | Sources represented |
| `compound_ids` | Source compound IDs |
| `target_names` | Target names represented |
| `document_years` | Source document years when available |
| `potency_class` | Coarse potency label |
| `active_100nM` | Median activity `<=100 nM` |
| `active_1uM` | Median activity `<=1000 nM` |

## `data/processed/herg_compounds_curated.csv`

Compound-level hERG liability table aggregated by SMILES.

Important columns:

- `p_herg_median`
- `herg_value_nm_median`
- `herg_blocker_label`
- `herg_label_policy`

`herg_blocker_label` is `1` for `<=10 uM`, `0` for `>=30 uM`, and blank for the ambiguous middle range.

## `data/processed/pk_admet_observations.csv`

Observed ChEMBL PK/ADMET rows for Menin-associated ChEMBL molecules. These are not yet endpoint-specific model labels.

Important columns:

- `molecule_chembl_id`
- `smiles`
- `standard_type`
- `standard_value`
- `standard_units`
- `assay_description`
- `target_pref_name`
- `document_chembl_id`
- `document_year`

## `reports/menin_with_predicted_herg_risk.csv`

Menin compound table with added hERG model predictions.

Added columns:

- `predicted_herg_blocker_probability`
- `predicted_herg_risk`: `low`, `medium`, or `high`
