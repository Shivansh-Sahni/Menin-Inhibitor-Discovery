# Evidence review: Price et al. (2025)

## Citation and project role

Price et al., "Explainable Machine Learning for ETR and Drug Chameleonicity," *Journal of Medicinal Chemistry* (2025). DOI: `10.1021/acs.jmedchem.5c00536`.

Only the four-page supporting-information PDF is local. The main article, definition/data table for ETR, structures, split assignments, and modeling code are absent. The attachment is therefore a promising precedent map, not a reproducible dataset.

## Evidence visible in the SI

The SI contains seven figures covering:

- train/test evaluation and chemical-space visualization for a machine-learned pEPSA model;
- categorical prediction results across PROTAC, bRo5, macrocycle, and Ro5 strata;
- ETR prediction error as a function of MW and rotatable-bond count;
- relationships among ETR, solubility, Caco-2 permeability, and absorption in polar compounds;
- measured and predicted ETR distributions for high/low permeability and absorption;
- explicit-water MD distributions of polar surface area versus Rgyr for KT-474 and two analogues; and
- IMHB frequencies for KT-474 in decadiene.

Together, the figures support the central project idea that chameleonicity is an environment-dependent response rather than a fixed descriptor. They also connect experimental/learned exposure of polarity to downstream permeability and absorption and show a matched-series path from MD distributions to a compact endpoint.

## Mechanistic value

The strongest contribution for the present program is the bridge between two levels:

`environment-conditioned conformer distribution -> exposed polarity response -> permeability/absorption phenotype`.

That bridge can guide validation of water-to-low-dielectric polarity shifts, folded-state populations, IMHB compensation, and rare transport-compatible conformers. The KT-474 figures also reinforce that joint PSA-Rgyr distributions contain information that a mean compactness value loses.

The SI alone does not define the ETR equation or experimental protocol with enough detail to reproduce it. The acronym must not be reverse-engineered or assigned a local formula from the figures. Until the main text and supporting data are acquired, ETR should be treated as an external named endpoint with unknown measurement and aggregation uncertainty.

## Limits and confounders

- Figure-level summaries do not expose compound-level labels, structures, predictions, or errors.
- The train/test topology and independence of chemical series cannot be audited.
- Permeability and absorption combine heterogeneous biological processes and may differ by assay/source.
- The figures do not provide the MD systems, force fields, simulation lengths, convergence checks, or trajectory data.
- The local package cannot establish whether ETR adds information beyond size, flexibility, ionization, lipophilicity, or assay-specific covariates.

## Project use

Retain ETR as a high-priority experimental/ML comparator for chameleonic response. Once the article and data arrive, reproduce its definition, stratify by chemical class and assay source, compare it with mechanistic ensemble features, and test whether it mediates rather than merely correlates with permeability. Do not use the current figure-only SI for training, threshold selection, or novelty claims.

## Bottom line

Price et al. appear to provide an experimentally anchored framework for connecting chameleonicity to permeability and absorption across difficult chemical classes. The current SI strengthens the research hypothesis but is insufficient for quantitative reproduction; the article and row-level supporting data remain essential.
