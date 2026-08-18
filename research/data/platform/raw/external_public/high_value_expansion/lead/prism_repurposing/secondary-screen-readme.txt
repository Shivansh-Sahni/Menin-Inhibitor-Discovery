prism_repurposing_secondary

The secondary PRISM Repurposing dataset contains the results of pooled-cell line chemical-perturbation viability screens for 1,448 compounds screened against 489 cell lines in an 8-step, 4-fold dilution, starting from 10uM. Technical redos for 147 previously screened oncology compounds have been added as of 11/22/2019. This data is identified using a `screen_id` of "MTS010".  It is recommended to use MTS010 when available. MTS010 was not included in the analysis for Corsello et al. 2019, DOI:10.1101/730119.


Data are processed according to the following protocol:

- Calculate the median fluorescence intensity (MFI) of each bead-replicate pair.
- Remove outlier-pools. MFI values are log-transformed and centered to the median logMFI for each cell line on each detection plate. For each well on a plate, the median of the centered values is standardized (using median and MAD) against all other wells in the same position in the same screen. Data from wells with a standardized score greater than 5 or less than -5 is filtered.
- Remove cell line-plates with low control separation. For each cell line-detection plate combination, the SSMD is calculated between the logMFI of positive and negative control treatments. Cell line-detection plates with an SSMD > -2 are removed.
- Divide data from each well-detection plate by the median of control barcodes from the same well-detection plate.
- Calculate log2-fold-change for data relative to negative-control wells from the same cell line on the same detection plate.
- Positive-control and negative-control treatments are removed.
- Run ComBat for each treatment condition (compound plate-well combination), considering log-viability values as probes and cell line pool-replicate combinations as batches.
- Median-collapse each drug-dose-cell line combination
- Calculate viability data as two to the power of replicate-level logfold change data.
- Fit dose-response curves to viability data using drc R package.


****************
Dataset contents
****************
## Shared metadata:

cell_line_info - table 
Cell line and control barcode metadata for rows of all PRISM Repurposing data matrices
Columns:
row_name - ID used to identify cell line or control barcode in data matrices
depmap_id - ID to identify a cell line
ccle_name - ID to identify a cell line
primary_tissue - 
secondary_tissue - 
tertiary_tissue - 
passed_str_profiling - indicator for problems identified by STR profiling
str_profiling_notes - summary of problems identified by STR profiling
originally_assigned_depmap_id - depmap_ID assigned to cell line before identification of problems by STR profiling
originally_assigned_ccle_name - ccle_name assigned to cell line before identification of problems by STR profiling


pooling_info - table 
Pooling information for all profiled cell lines and control barcodes
Columns:
row_name - ID used to identify cell line or control barcode in data matrices
depmap_id - ID to identify a cell line
ccle_name - ID to identify a cell line
screen_id - ID to identify a screen
analyte_id - ID used to identify the luminex bead coupled to the cell line barcode
detection_pool - ID used to identify the lysate detection pool of each cell line
minipool_id - ID used to identify the mini pool that a cell line is thawed in before pooling
pool_id - ID used to identify the pool that a cell line is assayed in
passed_str_profiling - indicator for problems identified by STR profiling


## Replicate-level files:

secondary_replicate_treatment_info - Table 
Drug treatment metadata at the replicate level
Columns:
column_name - ID used to identify the column of replicate-level data matrices
broad_id - ID used to identify drug-batch combinations
name - name of drug
dose - dose of drug(uM)
perturbation_type - “positive_control” treatments are MG-132 or bortezomib and should have strong cytotoxic effects on all cell lines. “vehicle_control” treatments have been treated with DMSO. “experimental_treatment” includes all other treatments
screen_id - ID to identify a screen
detection_plate - ID of plates used to detect luminex barcodes for a single replicate of a compound plate
compound_plate - ID of plate used to seed compounds
well - plate-position of treatment
moa - Broad Repurposing Hub mechanism of action annotation
target- Broad Repurposing Hub gene target annotation
disease.area - Broad Repurposing Hub disease area annotation
indication - Broad Repurposing Hub indication annotation
smiles - simplified molecular-input line-entry system molecular identifier
phase - Broad Repurposing Hub latest-clinical phase annotation


secondary_MFI - NumericalMatrix
Median fluorescence intensities
Columns: IDs identifying sample-replicate combinations. See column_name from primary_replicate_treatment_info
Rows: IDs identifying cell lines or control barcodes. See row_name from cell_line_info

secondary_logfold_change - NumericalMatrix
Logfold change values relative to DMSO, corrected for experimental confounders using ComBat
Columns: IDs identifying experimental condition-replicate combinations. See column_name from primary_replicate_treatment_info
Rows: IDs identifying cell lines. See row_name from cell_line_info

## Replicate-collapsed files:

secondary_replicate_collapsed_treatment_info - Table
Drug treatment metadata at drug level, collapsed across replicates within a screen
column_name - ID used to identify the column of replicate-collapsed data matrices
broad_id - ID used to identify drug-batch combinations
name - name of drug
dose - dose of drug(uM)
screen_id - ID to identify a screen
moa - Broad Repurposing Hub mechanism of action annotation
target- Broad Repurposing Hub gene target annotation
disease.area - Broad Repurposing Hub disease area annotation
indication - Broad Repurposing Hub indication annotation
smiles - simplified molecular-input line-entry system molecular identifier
phase - Broad Repurposing Hub latest-clinical phase annotation

secondary_replicate_collapsed_logfold_change - NumericalMatrix
Replicate collapsed logfold change values relative to DMSO, corrected for experimental confounders using ComBat
Columns: IDs identifying experimental conditions. See column name from primary_replicate_collapsed_treatment_info
Rows: IDs identifying cell lines. See row_name from cell_line_info


## Dose-response files:
secondary_dose_response_curve_parameters - Table
Parameters of dose-response curves fit to replicate-level viability data using a four-parameter log-logistic function with the drc R package.
broad_id - ID used to identify drug-batch combinations
name - name of drug

depmap_id - ID to identify a cell line
ccle_name - ID to identify a cell line
screen_id - ID to identify a screen
upper_limit - limit of curve as dose approaches infinite
lower_limit - limit of curve as dose approaches zero
slope - 
convergence - indicates if the dose-response curve fitting procedure converged
r2 - goodness of fit measure between dose response curve and replicate viability data
auc - area under the dose response curve
ec50 - dose of inflection point of dose response curve
ic50 - dose that corresponds to a viability of 50 on the dose response curve
moa - Broad Repurposing Hub mechanism of action annotation
target- Broad Repurposing Hub gene target annotation
disease.area - Broad Repurposing Hub disease area annotation
indication - Broad Repurposing Hub indication annotation
smiles - simplified molecular-input line-entry system molecular identifier
phase - Broad Repurposing Hub latest-clinical phase annotation
passed_str_profiling - indicator for problems identified by STR profiling


