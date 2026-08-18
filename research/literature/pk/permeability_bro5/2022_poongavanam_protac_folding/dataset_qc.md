# Dataset QC: Poongavanam PROTAC companion CSV

## File status

`assay_data.csv` is a verbatim copy of the publisher companion CSV. It contains three scientific rows, 23 columns, and one trailing empty row. Do not edit this source file to make it look cleaner; all harmonization should occur in a derived ingestion layer with explicit provenance.

## Schema observations

- The first header is `PROTAC ` with trailing whitespace.
- Unit notation uses `uM` rather than a machine-standard micromolar symbol.
- PAMPA for PROTAC 2 is missing.
- PAMPA for PROTAC 3 is censored as `>7.37` in negative-log permeability units; it is not an exact value.
- Means and SEM values are present, but replicate-level results and replicate counts are not stored row by row.
- The last row is entirely blank.

The ingestion schema should trim header whitespace while retaining the original field name, normalize units in a separate field, type missing and censored values, and preserve source strings.

There is also a protocol-text discrepancy: the main-article Table 2 footnote describes the passive Caco-2 calculation at pH 7.4, while supporting Table S1 states that the directional Caco-2 permeabilities were determined at pH 7.5. The exact apical and basolateral buffer conditions should be requested and stored separately; a pooled `pH` field should not be guessed.

## Caco-2 inconsistency

The supporting PDF Table S1 reports for PROTAC 1:

- AB permeability: 2.6 +/- 0.40 nm/s;
- BA permeability: 370 +/- 99 nm/s; and
- efflux ratio: 309 +/- 72.

The CSV reports:

- AB permeability: 2.58 +/- 0.40 nm/s;
- BA permeability: 370 +/- 99.7 nm/s; and
- efflux ratio: 165 +/- 72.3.

The ratio of the CSV means, 370 / 2.58, is approximately 143.41. The main article defines passive permeability as the geometric mean of AB and BA; the geometric mean of the CSV means is approximately 30.90 nm/s, close to but not identical to the CSV value 29.6 nm/s or the rounded article value 30 nm/s. These differences can arise when ratios and geometric means are calculated per replicate before averaging, but replicate-level data are absent and the 309-versus-165 conflict cannot be resolved from the attachments.

Required handling:

1. preserve PDF and CSV efflux ratios as distinct source observations;
2. set a conflict flag on PROTAC 1 efflux ratio;
3. do not overwrite, average, or choose one value;
4. do not recompute an authoritative efflux ratio from rounded means; and
5. request replicate-level AB/BA paired measurements and the exact aggregation method.

Rows 2 and 3 are consistent between PDF and CSV within rounding.

## Censoring and missingness

`PAMPA (-LogPe, cm/s)` is a negative logarithm. Larger values imply lower permeability. PROTAC 3's `>7.37` means its permeability is below the corresponding assay limit; it should enter a censored likelihood or ordinal comparison, not a point-regression table. PROTAC 2 has no PAMPA measurement and should remain missing rather than imputed from Caco-2.

## Fitness for use

With three related molecules, this dataset can validate parsing, units, censorship handling, and reproduction of published descriptive plots. It cannot support cross-validation, feature selection, a cutoff search, or a general permeability model. Any correlation coefficient calculated on these three rows is unstable and should not be treated as evidence.

The appropriate analytical use is a mechanistic case study with exact provenance. A larger matched-series dataset with replicate-level assays is required before estimating effects of folding, exposed polarity, or linker chemistry.
