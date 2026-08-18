# Source-data QC: Lau et al. (2024)

## Package integrity and inventory

- Canonical archive: `source_data.zip`.
- Archive integrity test: passed.
- Archive inventory: 79 entries comprising six directories, 72 raw Axon Binary Format (`.abf`) electrophysiology files, and one Excel workbook.
- The workbook was inspected with a spreadsheet parser, formula-error scan, and rendered visual review of every sheet.

## Workbook contents

The workbook, `Data for Nat Comms paper summary.xlsx`, has seven sheets:

1. `Summary`;
2. `WT`;
3. `V625T`;
4. `F627Y`;
5. `S620T`;
6. `S620T_V625T`; and
7. `S620T_F627Y`.

It contains fitted quantities and recording-level values associated with WT hERG and five mutant conditions shown in Figure 2E. File-name fragments link many columns to the raw `.abf` records. No spreadsheet formula errors were detected by the parser.

## Interpretation limits

- The workbook uses sparse, presentation-oriented headers; endpoint units and fitting conventions must be resolved from the article and supplementary information rather than inferred from cell position.
- Highlight colors identify selected values in the source workbook but are not treated as analytical labels.
- The `.abf` files require pClamp/Clampfit-compatible or independently validated ABF readers and the voltage protocol before reanalysis.
- The archive contains experimental electrophysiology source data, not MD inputs, prepared systems, trajectories, umbrella windows, or free-energy analysis code.
- The package is therefore useful for filter-state and mutant-mechanism validation, but it is not a compound-level hERG potency dataset and must not be merged into the internal liability labels.
