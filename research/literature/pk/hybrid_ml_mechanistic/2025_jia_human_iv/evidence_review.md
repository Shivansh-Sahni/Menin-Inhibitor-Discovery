# Evidence review: Jia et al. (2025)

## Citation and project role

Jia et al., "Application of Machine Learning and Mechanistic Modeling to Predict Intravenous Pharmacokinetic Profiles in Humans," *Journal of Medicinal Chemistry* (2025). DOI: `10.1021/acs.jmedchem.5c00340`.

The local package contains the public supporting-information PDF and modeling workbook, but not the main article. It is a useful successor to the Mavroudis rat-IV architecture and a human external-comparison resource. It is not direct calibration evidence for the present large-molecule rat IV/PO program.

## What is locally supported

The SI describes two second-stage profile models:

1. a hybrid workflow that predicts physicochemical/PK inputs and passes them to a physiological model; and
2. a hierarchical ML workflow that predicts log concentration as a function of time, infusion time, CL, Vdss, fraction unbound, and acidic/basic pKa.

The hybrid model uses in vivo CL together with logP, fraction unbound, MW, acidic/basic pKa, infusion time, and halogen counts. The hierarchical model uses CL and Vdss directly. These are modular and interpretable compared with a single structure-to-AUC model, but CL and Vdss are still downstream summaries of the same disposition process. They are not primitive causes of clearance or distribution.

The SI reports a 106-compound test set. For profile-derived AUC, the hybrid model has R2 = -0.38, GMFE = 3.22, and 43% within two-fold, while the hierarchical model has R2 = 0.10, GMFE = 1.93, and 60% within two-fold. Cmax R2 is negative for both frameworks (-0.29 and -0.26), despite 39% and 59% within two-fold, respectively. This combination is important: acceptable aggregate fold-error counts can coexist with poor between-compound variance explanation.

The SI also treats alternative tissue-distribution hypotheses as structural uncertainty and shows applicability-domain analyses based on nearest-neighbor fingerprint similarity. Those are useful precedents for the program's decision/discovery separation.

## Workbook evidence

The workbook contains nine sheets and explicit endpoint-specific train/test labels:

- 106 unique test compounds with observed and predicted pKa, fraction unbound, CL, Vdss, infusion time, and structure fields;
- 9,208 unique parent structures in the pKa modeling set;
- 5,620 rows in the combined fraction-unbound/Vdss/CL modeling set;
- fitted hyperparameter tables; and
- source-oriented ChEMBL pKa, Lombardo/Trend, eDrug3D, Watanabe, and OPERA sheets.

The 106 test structures span 136.04-810.42 Da, but only five are at least 650 Da and only three are at least 700 Da. The property-modeling sheets contain more high-MW entries, but their presence does not establish that the human IV profile model was validated on a representative bRo5-like chemical regime.

All 106 test SMILES also occur in the pKa and PK-property modeling sheets, where the endpoint split fields identify them as test observations. This is expected for a common test set, but any ingestion that drops the endpoint-specific split columns would create severe leakage.

## Mechanistic interpretation

This study supports a layered architecture, not a claim that learned CL and Vdss reveal causal PK mechanisms. A deeper model should replace or supplement those summaries with separately measured or latent processes:

- blood/plasma partition and unbound fraction;
- hepatic uptake, intrinsic metabolism, and blood-flow limitation;
- renal filtration, secretion, and reabsorption;
- tissue-specific binding and permeability limitation; and
- for PO dosing, dissolution, intestinal microstate populations, membrane crossing, efflux, gut metabolism, and first pass.

The workbook is especially useful for testing endpoint-specific split handling, property-model baselines, applicability-domain methods, and uncertainty propagation across alternative distribution models. It cannot validate the present oral process graph because it contains no PO process measurements.

## Reproducibility and limitations

- The local package does not include the main article.
- No sheet stores sampling time and concentration as row-level observations; the digitized concentration-time profiles needed to independently reconstruct curves, AUC, or Cmax are absent.
- The workbook combines source datasets with different provenance and endpoint coverage. Source, endpoint, and split columns must remain intact.
- Predicted and observed fields coexist in the 106-compound sheet and must never be conflated.
- One standardized-parent duplicate in the PK-property set represents counterion/parent normalization with complementary endpoints, not an observation to delete blindly.
- The study is human IV, whereas the immediate internal target is rat IV and PO.

## Project use

Use the public workbook to reproduce conventional property-model baselines and to audit grouped, endpoint-specific validation. Use the SI's alternative-distribution comparison as a structural-uncertainty template. Do not use its CL/Vdss-driven profile predictions as evidence that primitive clearance or distribution mechanisms have been identified, and do not describe its human predictions as calibrated for the internal rat series.

## Bottom line

Jia et al. provide a valuable public hybrid/hierarchical PK benchmark and explicit train/test resources. The attached package supports reproduction of property and summary-driven profile modeling, but not independent reconstruction of time-resolved human PK or direct validation in the intended large-molecule rat domain.
