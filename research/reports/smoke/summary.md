# Menin discovery data and modeling audit

Content build: `build-6237a81ba03d0ccb`

## Bottom line

This build is incomplete: one or more primary models or the analysis-eligible quality gate is unavailable. Treat the repository as an auditable data pipeline, not as validated predictive evidence, until the readiness matrix is satisfied.

## Dataset inventory

| dataset | rows | unique_structures | modeling_eligible_rows |
| --- | --- | --- | --- |
| menin_measurements | 8176 | 2123 | 3.8e+03 |
| menin_endpoint_assay_tasks | 2104 | 1184 |  |
| herg_measurements | 41078 | 26060 | 1.77e+04 |
| herg_endpoint_assay_tasks | 11549 | 11053 |  |
| pk_admet_observations | 204 | 40 |  |

## Menin endpoint and assay coverage

| endpoint | assay_family | measurements | unique_structures | exact_measurements | modeling_eligible | median_p_activity |
| --- | --- | --- | --- | --- | --- | --- |
| IC50 | biochemical_binding | 1722 | 991 | 1631 | 1389 | 6.66 |
| nan | unclassified | 1505 | 124 | 1505 | 0 |  |
| nan | biochemical_binding | 1221 | 445 | 1221 | 0 |  |
| IC50 | biochemical_inhibition | 1103 | 912 | 1034 | 1101 | 6.85 |
| Ki | biochemical_binding | 696 | 370 | 196 | 695 | 9 |
| k_off | biochemical_binding | 437 | 429 | 437 | 0 |  |
| kon | biochemical_binding | 437 | 429 | 437 | 0 |  |
| IC50 | in_vivo | 416 | 184 | 326 | 416 | 6.65 |
| Activity | in_vivo | 112 | 23 | 110 | 0 |  |
| IC50 | cellular_functional | 99 | 54 | 92 | 97 | 5.89 |
| Activity | biochemical_binding | 59 | 22 | 59 | 0 |  |
| Kd | biochemical_binding | 51 | 20 | 51 | 50 | 7.62 |
| Ki | cellular_functional | 38 | 38 | 37 | 38 | 7.96 |
| Inhibition | unclassified | 35 | 28 | 22 | 0 |  |
| GI50 | in_vivo | 33 | 29 | 32 | 0 |  |
| FC | in_vivo | 21 | 8 | 18 | 0 |  |
| Inhibition | biochemical_binding | 19 | 16 | 6 | 0 |  |
| Inhibition | in_vivo | 19 | 5 | 11 | 0 |  |
| Activity | unclassified | 15 | 3 | 15 | 0 |  |
| CL | in_vivo | 13 | 2 | 13 | 0 |  |
| Tm | biochemical_binding | 12 | 2 | 12 | 0 |  |
| GI | unclassified | 10 | 2 | 10 | 0 |  |
| Cp | in_vivo | 9 | 3 | 9 | 0 |  |
| T1/2 | in_vivo | 9 | 3 | 9 | 0 |  |
| Solubility | in_vivo | 9 | 9 | 7 | 0 |  |
| Activity | cellular_functional | 9 | 2 | 9 | 0 |  |
| AUC | in_vivo | 8 | 3 | 8 | 0 |  |
| Kd | in_vivo | 6 | 6 | 6 | 6 | 7.64 |
| Kd | cellular_functional | 6 | 6 | 6 | 6 | 8 |
| GI50 | unclassified | 6 | 2 | 6 | 0 |  |

## Critical-field completeness

| dataset | column | rows | missing_rows | missing_fraction |
| --- | --- | --- | --- | --- |
| menin | smiles | 8176 | 1382 | 0.169 |
| menin | target_name | 8176 | 3884 | 0.475 |
| menin | target_id | 8176 | 3884 | 0.475 |
| menin | assay_description | 8176 | 1126 | 0.138 |
| menin | assay_type | 8176 | 1126 | 0.138 |
| menin | document_year | 8176 | 1126 | 0.138 |
| menin | standard_units | 8176 | 3300 | 0.404 |
| menin | p_value | 8176 | 3999 | 0.489 |
| herg | smiles | 41078 | 102 | 0.00248 |
| herg | target_name | 41078 | 0 | 0 |
| herg | target_id | 41078 | 0 | 0 |
| herg | assay_description | 41078 | 0 | 0 |
| herg | assay_type | 41078 | 0 | 0 |
| herg | document_year | 41078 | 3354 | 0.0816 |
| herg | standard_units | 41078 | 6563 | 0.16 |
| herg | p_value | 41078 | 17780 | 0.433 |
| pk_admet | smiles | 204 | 0 | 0 |
| pk_admet | admet_endpoint | 204 | 0 | 0 |
| pk_admet | species | 204 | 14 | 0.0686 |
| pk_admet | matrix | 204 | 137 | 0.672 |
| pk_admet | administration_route | 204 | 85 | 0.417 |
| pk_admet | experimental_context | 204 | 0 | 0 |
| pk_admet | standard_units | 204 | 0 | 0 |

## Cross-source mirror linkage

The curation layer linked 863 potential cross-source mirror groups. Same-source replicates remain intact; a separate no-collapse model sensitivity analysis quantifies the heuristic's effect.

## Applicability-domain performance

| inside_applicability_domain | holdout_rows | mae | median_absolute_error | mean_max_training_tanimoto |
| --- | --- | --- | --- | --- |
| False | 25 | 0.929 | 1 | 0.553 |
| True | 144 | 0.716 | 0.64 | 0.789 |

## hERG task populations

Primary functional-electrophysiology task:

| scope | endpoint | assay_family | label | task_rows |
| --- | --- | --- | --- | --- |
| primary | IC50 | electrophysiology_functional | blocker | 2412 |
| primary | IC50 | electrophysiology_functional | non_blocker | 365 |
| primary | IC50 | electrophysiology_functional | ambiguous_10_to_30_uM | 816 |

Broader pooled sensitivity population (not the headline safety estimate):

| scope | endpoint | assay_family | label | task_rows |
| --- | --- | --- | --- | --- |
| pooled_sensitivity | pooled | pooled | blocker | 7945 |
| pooled_sensitivity | pooled | pooled | non_blocker | 1068 |
| pooled_sensitivity | pooled | pooled | ambiguous_10_to_30_uM | 2536 |

## Validation summary

| task | requested_split | actual_split | status | model | n_compounds | n_train | n_test | mae | rmse | r2 | spearman_r | roc_auc | pr_auc | balanced_accuracy | brier_score | expected_calibration_error_10bin |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| menin_IC50_biochemical_binding | scaffold | scaffold | trained | extra_trees | 849 | 680 | 169 | 0.747 | 0.926 | 0.55 | 0.736 |  |  |  |  |  |
| herg_IC50_electrophysiology_functional | scaffold | scaffold | trained | extra_trees | 2777 | 2223 | 554 |  |  |  |  | 0.864 | 0.967 | 0.691 | 0.0808 | 0.0411 |

Random-split results are included as a sensitivity analysis, not the headline estimate. Scaffold, chemical-cluster, and temporal results better probe prospective chemical generalization; any requested strategy fallback is recorded in the `actual_split` column and split manifest.

## High-potency public tasks

| structure_id | standard_inchi_key | smiles | compound_ids | endpoint | assay_family | p_activity_median | value_nm_median | n_measurements | sources | activity_range_log10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| STR-510CF8EB9D92CA4FC9A9 | KJEUORQIESDSSV-UHFFFAOYSA-N | O=S(=O)(NCB(O)O)c1ccc(-c2nn\[nH\]n2)cc1C(F)(F)F | 71549453 | Ki | biochemical_binding | 10.3 | 0.05 | 1 | PubChem | 0 |
| STR-7E1F0EE1EE4F776E8704 | PFCVKXYYUQIAHS-AREMUKBSSA-N | CCN(C(=O)c1cc(F)ccc1Oc1nncnc1N1CCC2(C1)CN(\[C@H\](CCCNCCOC)C(C)C)C2)C(C)C | 689587 | IC50 | biochemical_inhibition | 10 | 0.1 | 2 | BindingDB | 0.0872 |
| STR-B2AA5A0FB80A562D611E | ADHHOUXZPBYYSU-YOCNBXQISA-N | CC(C)N(C(=O)c1cc(F)ccc1Oc1cncnc1N1CC2(CCN(C\[C@H\]3CC\[C@H\](NS(C)(=O)=O)CC3)CC2)C1)C(C)C | CHEMBL5267644 | Ki | biochemical_binding | 9.98 | 0.104 | 1 | ChEMBL | 0 |
| STR-CAD50D46562C0D29D331 | PDUGAXSIWNMIBQ-HHHXNRCGSA-N | CCN(C(=O)c1cc(F)ccc1Oc1nncnc1N1CCC2(C1)CN(\[C@H\](CCCN(C)CCOC)C(C)C)C2)C(C)C | 689589;689590 | IC50 | biochemical_inhibition | 9.88 | 0.139 | 2 | BindingDB | 0.264 |
| STR-BD9FCFA15F8B311D296A | XVEVAEBDYQDZRJ-UHFFFAOYSA-N | CNc1nc(NC2CCN(Cc3ccc4c(cc(C#N)n4CC45CC(NC=O)(C4)C5)c3C)CC2)c2cc(CCC(F)(F)F)sc2n1 | 172447626 | IC50 | biochemical_inhibition | 9.3 | 0.5 | 1 | PubChem | 0 |
| STR-DC80B2DAE37FF4BEDB8F | MBFYNGBMAACLAT-UHFFFAOYSA-N | CNc1nc(NC2CCN(Cc3ccc4c(cc(C#N)n4CC45CC(NC=O)(C4)C5)c3C)CC2)c2cc(CC(F)(F)F)sc2n1 | 130443890 | IC50 | biochemical_inhibition | 9.3 | 0.5 | 1 | PubChem | 0 |
| STR-073AD85CF5EE1E9C87AB | MIDBFVUJIVFIOM-UHFFFAOYSA-N | CCN1Cc2ccccc2C(C2CCCC2)(C2CCN(CC3CN(c4ccc(S(=O)(=O)c5ccncc5)cc4)C3)CC2)C1 | CHEMBL4585198 | Ki | biochemical_binding | 9 | 1 | 1 | BindingDB;ChEMBL | 0 |
| STR-073AD85CF5EE1E9C87AB | MIDBFVUJIVFIOM-UHFFFAOYSA-N | CCN1Cc2ccccc2C(C2CCCC2)(C2CCN(CC3CN(c4ccc(S(=O)(=O)c5ccncc5)cc4)C3)CC2)C1 | 132090541 | Ki | cellular_functional | 9 | 1 | 1 | PubChem | 0 |
| STR-34DA364C524542BAF54B | LAHJBSUOBULURD-HHKHNSFJSA-N | COC(=O)N\[C@H\]1CCC\[C@@H\]1\[C@\](CN1CCC1)(c1cccc(F)c1)C1CCN(CC2CN(c3ccc(S(=O)(=O)C4CN(C(=O)/C=C/CN5CC(F)(F)C5)C4)cc3)C2)CC1 | CHEMBL4530300 | IC50 | biochemical_binding | 9 | 1 | 1 | ChEMBL | 0 |
| STR-34DA364C524542BAF54B | LAHJBSUOBULURD-HHKHNSFJSA-N | COC(=O)N\[C@H\]1CCC\[C@@H\]1\[C@\](CN1CCC1)(c1cccc(F)c1)C1CCN(CC2CN(c3ccc(S(=O)(=O)C4CN(C(=O)/C=C/CN5CC(F)(F)C5)C4)cc3)C2)CC1 | 50511905 | IC50 | biochemical_inhibition | 9 | 1 | 1 | BindingDB | 0 |
| STR-34FE1D5BDCFE7B2EB7FE | MCMLLYNXDSNKTO-PALXJHBUSA-N | O=c1cccc2n1\[C@@H\]1C\[C@H\](C2)CN(CC(O)CCCc2ccccc2)C1 | CHEMBL4211366 | IC50 | biochemical_binding | 9 | 1 | 1 | ChEMBL | 0 |
| STR-34FE1D5BDCFE7B2EB7FE | MCMLLYNXDSNKTO-PALXJHBUSA-N | O=c1cccc2n1\[C@@H\]1C\[C@H\](C2)CN(CC(O)CCCc2ccccc2)C1 | 50454121 | IC50 | biochemical_inhibition | 9 | 1 | 1 | BindingDB;PubChem | 0 |
| STR-723FB71024F547E2C9C5 | QJWOBZCZCISQBL-VXSFFCHMSA-N | C\[C@@\]12CN(CC(O)CCCc3ccccc3)C\[C@@\](C)(C1)c1cccc(=O)n1C2 | 171354227 | IC50 | biochemical_inhibition | 9 | 1 | 1 | PubChem | 0 |
| STR-BB3373616F96D1625E90 | BCTIJYSCGDNKEE-OKOZNKHASA-N | CCc1nccn1C\[C@@\](c1cccc(F)c1)(C1CCN(CC2(F)CN(c3ccc(S(=O)(=O)C4CN(C(=O)/C=C/CN(C)C)C4)cc3)C2)CC1)\[C@H\]1CCC\[C@@H\]1NC(=O)OC | CHEMBL6053681 | IC50 | biochemical_binding | 9 | 1 | 1 | ChEMBL;PubChem | 0 |
| STR-BB3373616F96D1625E90 | BCTIJYSCGDNKEE-OKOZNKHASA-N | CCc1nccn1C\[C@@\](c1cccc(F)c1)(C1CCN(CC2(F)CN(c3ccc(S(=O)(=O)C4CN(C(=O)/C=C/CN(C)C)C4)cc3)C2)CC1)\[C@H\]1CCC\[C@@H\]1NC(=O)OC | 506357 | IC50 | biochemical_inhibition | 9 | 1 | 1 | BindingDB | 0 |
| STR-ECA04B71ABBECE5B5C4C | SHNAIZGISNDQJQ-UHFFFAOYSA-N | O=S(=O)(NCB(O)O)c1ccc(-c2nn\[nH\]n2)cc1 | 71549452 | Ki | biochemical_binding | 8.92 | 1.2 | 1 | PubChem | 0 |
| STR-052D31B0B659EBB9473E | LPPRDZVYQRJOOO-SXOMAYOGSA-N | CCN(C(=O)c1cc(F)ccc1Oc1cncnc1N1CC2(CCN(C\[C@@H\]3CC\[C@@H\](NC(=O)NCC4CC4)CO3)CC2)C1)C(C)C | 656647 | Ki | biochemical_binding | 8.9 | 1.25 | 1 | BindingDB;PubChem | 0 |
| STR-2D8843254F92027CB324 | QURMPJZYPCSXKN-SXOMAYOGSA-N | CCN(C(=O)c1cc(F)ccc1Oc1cncnc1N1CC2(CCN(C\[C@@H\]3CC\[C@@H\](NS(=O)(=O)CCN(C)C)CO3)CC2)C1)C(C)C | 656824 | Ki | biochemical_binding | 8.9 | 1.25 | 1 | BindingDB;PubChem | 0 |
| STR-495577E2F7FD7CFA736E | LRUVGLUJXLOGAW-FTJBHMTQSA-N | CCN(C(=O)c1cc(F)ccc1Oc1cnc(C)nc1N1CC2(CCN(C\[C@@H\]3CC\[C@@H\](NS(=O)(=O)CC)CO3)CC2)C1)C(C)C | 656820 | Ki | biochemical_binding | 8.9 | 1.25 | 1 | BindingDB;PubChem | 0 |
| STR-5FD5B7D1231F811AD204 | MLNCDQCPBWERBE-FTJBHMTQSA-N | CCCNC(=O)N\[C@@H\]1CC\[C@@H\](CN2CCC3(CC2)CN(c2ncncc2Oc2ccc(F)cc2C(=O)N(CC)C(C)C)C3)OC1 | 656643 | Ki | biochemical_binding | 8.9 | 1.25 | 1 | BindingDB;PubChem | 0 |

Rows above are endpoint–assay tasks, not interchangeable efficacy claims. Review the original records, replicate spread, assay context, censoring, and source rights before experimental prioritization.

## Quality-gate status

```json
{}
```

Quality findings are not all fatal errors: the detailed CSVs distinguish missing public metadata, quarantined measurements, repeated-measurement conflicts, and schema violations. Modeling uses only rows that pass the explicit eligibility policy.

## Publication readiness (4/13 infrastructure criteria currently satisfied)

| criterion | satisfied | evidence_or_action |
| --- | --- | --- |
| Defined biological endpoint | True | Primary tasks are IC50 × biochemical_binding and scoped hERG IC50 × electrophysiology_functional. |
| Unambiguous RDKit algorithm | True | Both trained primary manifests must record RDKit Morgan features and selected algorithms. |
| Complete requested holdouts | False | Every configured split must train and must not silently fall back. |
| Applicability domain | True | Both primary tasks record nearest-neighbor domain policy and coverage. |
| Uncertainty | True | Primary Menin metrics include scaffold-group bootstrap intervals and conformal prediction intervals. |
| Build-linked provenance | False | Processed, model, and report manifests plus model provenance must share a content build. |
| Upstream manifest verification | False | Raw, processed, software, and model inputs must verify before report generation; the final report bundle is verified as the release's last step. |
| Data quality audit | False | The analysis-eligible quality gate must pass; source-inventory exclusions remain reported. |
| Environment lock | False | The frozen environment lock must be included in the verified software manifest. |
| Clean source revision | False | A publication release must be rebuilt from a clean committed revision. |
| Independent external validation | False | No independent Menin or hERG external test set has been reserved. |
| Prospective experimental validation | False | Requires new lab measurements after model lock. |
| Authorship and licensing approval | False | Requires project-owner approval. |

## Figures

- [Data attrition](figures/data_quality_attrition.png)
- [Endpoint coverage](figures/menin_endpoint_counts.png)
- [Source × endpoint coverage](figures/menin_source_endpoint_heatmap.png)
- [Chemical space](figures/menin_chemical_space.png)
- [Observed vs predicted Menin activity](figures/menin_observed_vs_predicted.png)
- [Menin validation sensitivity](figures/menin_validation_comparison.png)
- [hERG discrimination](figures/herg_discrimination_curves.png)
- [hERG calibration](figures/herg_calibration.png)
- [Potency–hERG triage](figures/menin_potency_herg_triage.png)

## Interpretation boundaries

- Public assay values remain heterogeneous across biochemical, biophysical, and cellular contexts.
- Censored values are preserved as bounds but excluded from point-regression labels; no censored-likelihood model is claimed.
- hERG thresholds define a screening label, not clinical cardiotoxicity, and probabilities require external calibration checks.
- PK/ADMET observations are coverage evidence only until endpoint, species, matrix, route, and units support separate validated tasks.
- Model artifacts must be loaded only from this trusted build and matched to their manifests.

## Target anchors

- Menin/MEN1: CHEMBL1615381; UniProt O00255.
- hERG/KCNH2: CHEMBL240; UniProt Q12809.

The complete methods, data dictionary, architecture, limitations, reproducibility instructions, and internal-data intake contract are maintained under `docs/`.
