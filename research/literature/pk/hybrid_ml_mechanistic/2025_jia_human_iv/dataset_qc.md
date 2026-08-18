# Dataset QC: Jia human-IV modeling workbook

## File status

`supplementary_dataset.xlsx` is a valid Office Open XML workbook (6,005,940 bytes) with nine worksheets. It is preserved as publisher evidence. The QC below is descriptive; no source cell was edited.

## Sheet inventory

| Sheet | Data rows | Columns | Primary role |
|---|---:|---:|---|
| `106_cmp_test_set` | 106 | 25 | common external test compounds and observed/predicted model inputs |
| `pKas_modeling_set` | 9,208 | 12 | acidic/basic pKa modeling records and endpoint-specific splits |
| `Fu_VDss_CL_modeling_set` | 5,620 | 23 | fraction-unbound, Vdss, and CL records and splits |
| `hyperparameter for models` | 27 | 11 | selected RF/SVM/XGBoost settings |
| `ChEMBL_pKas` | 11,488 | 34 | source-oriented ChEMBL property export |
| `Lombardo_Trend` | 1,352 | 21 | human IV summary-PK source table |
| `eDrug3D` | 2,083 | 17 | source drug/property table |
| `watanabe` | 2,319 | 8 | fraction-unbound source/split table |
| `Opera` | 2,723 | 14 | OPERA fraction-unbound source/split table |

## Test-set checks

The test sheet contains 106 unique `ID_trend` values and 106 unique SMILES, with no duplicate SMILES. Its RDKit MW range is 136.04-810.42 Da. Five compounds are at least 650 Da and three are at least 700 Da. None of the 106 rows is missing the observed acidic pKa, basic pKa, fraction unbound, final CL, final Vdss, or infusion-time field.

Predictions and reference values are stored side by side (`pred_*` versus pKa, `fu`, `CL_final`, and `VD_final`). Ingestion must preserve the semantic distinction and must not train on predicted fields as if they were experimental labels.

## Endpoint modeling sets

The pKa sheet contains 9,208 unique parent SMILES. Acidic-pKa split counts are 5,306 train and 776 test, with 3,126 rows lacking an acidic-pKa split because the endpoint is absent. Basic-pKa counts are 5,687 train and 815 test, with 2,706 endpoint-missing rows. Split missingness is therefore endpoint-specific, not evidence that the whole compound is unassigned.

The PK-property sheet contains 5,620 rows and 5,619 unique parent SMILES. Endpoint coverage is 1,464 rows for Vdss, 1,464 for CL, and 4,675 for fraction unbound. Vdss and CL have 1,287 train and 177 test labels; fraction unbound has 4,042 train and 633 test labels.

The sole repeated parent SMILES is the normalized parent of a quaternary-ammonium compound represented once without bromide counterions and once as the dibromide. The two rows carry complementary source endpoints. It should be resolved through state/source-aware standardization, not deleted as a generic duplicate.

## Split isolation

All 106 common-test SMILES occur in both modeling sheets and are marked through endpoint-specific test flags. This layout is safe only if:

1. each endpoint uses its own split column;
2. the 106-compound rows never enter fitting, feature selection, or calibration;
3. parent/salt aliases remain grouped; and
4. source sheets are not independently appended after the curated modeling sheets, which would duplicate evidence.

## What the workbook does not contain

No sheet is a row-level concentration-time table: there are no sampling-time/concentration pairs, LLOQ fields, subject identifiers, or infusion/dose events sufficient to reconstruct a `PKStudy`/`PKSample` hierarchy. The file contains summary inputs, source tables, split assignments, and hyperparameters. Raw/digitized profiles, curve-level prediction outputs, code, and fitted model artifacts remain missing.

## Fitness for use

The workbook is fit for property-model reproduction, split audits, and external methodological comparison. It is not fit for learning a neural ODE, recalculating AUC/Cmax from observations, or calibrating rat IV/PO exposure. Any large-molecule analysis must report the sparse 650+ Da test stratum separately and treat the workbook as out-of-species external evidence.
