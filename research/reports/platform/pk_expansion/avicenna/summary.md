# PK/ADME public expansion — Avicenna workstream

## Result

- Acquired and checksum-bound **27 TDC tables (112,841 rows)**, **8 official NCATS/PubChem assays (23,994 rows)**, and **12 OpenADMET tables (139,770 physical file rows)** without credentials or HPC.
- The only published systemic-PK benchmark labels acquired are **2,437 TDC rows**: bioavailability 640, half-life 667, and Vdss 1,130. They lack explicit route, dose, and unit columns, so **zero rows are context-complete genuine PK observations**.
- Upstream ADME/safety acquisition totals **274,168 physical file rows** (110,404 TDC + 23,994 NCATS + 139,770 OpenADMET). This is not a unique-observation count because paired X/y files, splits, wells, and cross-source overlap are present.
- A full ChEMBL 37 exact-type census found **330,261 candidate activity rows across 85,441 molecules**. These remain candidates: route and dose are not structured, `Cl` is mostly chloride, lowercase `t1/2` is mostly nonnumeric, `CL` mixes systemic and tissue-normalized clearance, and several endpoints have incompatible units.
- PK-DB reports **138,411 outputs**, but the anonymous endpoint returned **0** and the API declares Basic authentication. No PK-DB label was fabricated or admitted.
- DailyMed follow-up parsed **54,672/54,672 SPL XML documents with 0 errors**, indexing **43,162 quantitative PK candidate sections and 26,610 tables** across 41,960 document versions. Exact approval IDs joined 19,652 candidate versions to Drugs@FDA. These remain evidence candidates and contribute **0 normalized PK quantities**.
- **Canonical rows admitted: 0. Training labels admitted: 0.** This is intentional: raw acquisition, provenance, rights, context, and endpoint-tier separation are complete; scientific admission still requires source-level normalization and review.

## Endpoint hierarchy

1. **Systemic PK quantities** — CL, t1/2, AUC, Cmax, Tmax, Vd/Vdss, F. Preserve species, route, matrix, dose, formulation, sampling/time basis, units, and study provenance.
2. **Upstream ADME** — solubility, permeability, protein binding, CYP/transporter behavior, metabolic stability, microsomal/hepatocyte CLint. These may explain PK but are not interchangeable with it.
3. **Regulatory evidence** — DailyMed clinical-pharmacology prose/tables and Drugs@FDA stage metadata. These are product- and time-specific until extracted and mapped.
4. **Blocked/conditional sources** — PK-DB credentials/rights, DrugBank license, and one authenticated OpenADMET model repository.

## Quality and overlap limits

- Counts are labeled as physical file rows, benchmark rows, raw type hits, or official-but-inaccessible outputs; they are never presented as one deduplicated dataset.
- No source was merged across structure, salt, stereochemistry, study, species, route, dose, or assay. That avoids silent leakage and false biological equivalence.
- TDC underlying dataset rights are source-specific; OpenADMET repositories are Apache-2.0 or CC-BY-4.0 with noted conflicts; ChEMBL is CC BY-SA 3.0; regulator and NLM terms still require institutional review.

## Next CPU-feasible work

1. Review ChEMBL documents/assays and units for the 330,261 hits; extract route/dose from primary evidence where available.
2. Obtain authorized PK-DB access and record-level rights; it is the highest-value context-rich source.
3. Human-review the indexed DailyMed evidence spans/tables and normalize only claims with product/formulation/dose/route/matrix/time provenance.
4. Create structure- and study-aware overlap groups before any split or model training.

Official entry points: [TDC](https://tdcommons.ai/single_pred_tasks/overview/), [NCATS OpenData](https://opendata.ncats.nih.gov/public/adme/data/public_datasets/), [OpenADMET](https://openadmet.org/datasetsmodels/), [ChEMBL](https://www.ebi.ac.uk/chembl/), [PK-DB](https://pk-db.com/), [DailyMed](https://dailymed.nlm.nih.gov/dailymed/), and [Drugs@FDA](https://www.fda.gov/drugs/drug-approvals-and-databases/drugsfda-data-files).
