# Dataset QC: Kim degrader supplementary dataset

## File status

`dataset.pdf` is a valid four-page PDF (57,543 bytes). Its first line identifies the source as `JCIM_PermPred_Degraders_SupplementaryData.docx`, but the local asset is only a PDF rendering. It should remain evidence, not be silently converted into an authoritative machine-readable source table.

`supplementary_information.pdf` is a valid 48-page PDF (4,016,062 bytes). SI Table S3 provides an independently readable 32-row permeability summary.

## Record and field coverage

The dataset table contains 32 compound identifiers with:

- E3 ligase and target;
- ensemble-average 3D-PSA, IMHB, and Rgyr;
- full PROTAC SMILES;
- Papp AB, Papp BA, and reported passive/net Papp;
- E3-ligand, linker, and warhead fragment representations.

The SI confirms 32 compounds: 19 VHL, 12 CRBN, and one MDM2-based molecule. Units are explicit in the SI for Rgyr (angstrom), 3D-PSA (square angstrom), IMHB (count), and permeability (nm/s). The dataset PDF's narrow columns and wrapped SMILES make automatic text extraction lossy, so the original DOCX/CSV remains necessary for reliable structure ingestion.

## Derived permeability

Reported passive/net Papp is generally consistent with a geometric combination of directional AB and BA values, as in the underlying PROTAC precedent, but it should be treated as derived. Rounded directional values do not always reproduce the reported net value exactly. Two BA entries display as 0.0 while their net Papp remains nonzero, implying hidden precision, an assay floor, or a source-specific convention. Do not replace the reported value with a calculation from rounded PDF cells.

Required handling:

1. retain AB, BA, and net Papp as separate fields;
2. record net Papp lineage and formula only when confirmed from the article/source;
3. preserve censored or rounded-to-zero directional values as relations, not exact zeros; and
4. retain the literature-reference identifier as protocol/source evidence.

## Provenance and comparability

The 32 rows originate from 14 publications. The PDF table does not provide a row-complete assay protocol, replicate count, uncertainty, cell line, pH, recovery, free concentration, or transporter-control schema. Source effects must therefore be modeled or stratified; the values should not be treated as one uniform experiment.

The SI text says seven compounds have solubility data, while Table S9 visibly lists six. Those listed records are not comparable without condition fields: buffers span pH 5.8-7.4, and one value uses DMSO/Cremophor/PBS. Preserve the count conflict, and retain the values only as descriptive source-specific measurements.

## Modeling constraints

The reported 100 random 50/50 splits are not a scaffold- or series-held-out validation design. Related VHL/BRD4 and other linked analogs can appear on both sides. Any local reproduction must group structure, series, E3 system, and assay source; preserve an untouched evaluation; and avoid selecting 3D features on the final folds.

## Missing reproducibility assets

Still missing are the main article, original DOCX or CSV, row-level assay metadata and replicates, exact split assignments, metadynamics starting systems/inputs/trajectories, saved conformer ensembles, ANI refinement workflow, descriptor-generation code, and fitted model artifacts.

## Fitness for use

This package is fit for qualitative mechanism review, manual cross-checking of a 32-compound benchmark, and defining a future ingestion contract. It is not yet fit for lossless automated structure ingestion or strong external-validation claims.
