# Menin discovery data and modeling audit

Content build: `build-6237a81ba03d0ccb`

## Bottom line

The repository contains a traceable public-data evidence base and a chemically grouped, endpoint-scoped baseline QSAR evaluation. Menin labels are isolated by endpoint before aggregation; hERG labels are resolved at structure level; uncertainty, calibration, applicability domain, quarantine, and provenance outputs are explicit. These models remain hypothesis-generation tools until independent and prospective lab validation is complete.

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
|  | unclassified | 1505 | 124 | 1505 | 0 |  |
|  | biochemical_binding | 1221 | 445 | 1221 | 0 |  |
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
| False | 26 | 0.868 | 0.826 | 0.532 |
| True | 143 | 0.739 | 0.709 | 0.78 |

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
| menin_IC50_biochemical_binding | scaffold | scaffold | trained | extra_trees | 849 | 680 | 169 | 0.759 | 0.935 | 0.541 | 0.733 |  |  |  |  |  |
| herg_IC50_electrophysiology_functional | scaffold | scaffold | trained | extra_trees | 2777 | 2.22e+03 | 554 |  |  |  |  | 0.869 | 0.974 | 0.731 | 0.0799 | 0.0405 |
| menin_IC50_biochemical_binding | chemical | chemical | trained | ridge_alpha_10 | 849 | 678 | 171 | 0.879 | 1.1 | 0.521 | 0.75 |  |  |  |  |  |
| herg_IC50_electrophysiology_functional | chemical | chemical | trained | logistic_c0p3 | 2777 | 2.21e+03 | 566 |  |  |  |  | 0.709 | 0.931 | 0.512 | 0.112 | 0.024 |
| menin_IC50_biochemical_binding | temporal | temporal | trained | ridge_alpha_10 | 849 | 654 | 195 | 1.3 | 1.54 | -2.74 | 0.159 |  |  |  |  |  |
| herg_IC50_electrophysiology_functional | temporal | temporal | trained | logistic_c0p3 | 2777 | 2.27e+03 | 503 |  |  |  |  | 0.641 | 0.896 | 0.551 | 0.206 | 0.178 |
| menin_IC50_biochemical_binding | random | random | trained | ridge_alpha_10 | 849 | 679 | 170 | 0.636 | 0.806 | 0.649 | 0.797 |  |  |  |  |  |
| herg_IC50_electrophysiology_functional | random | random | trained | extra_trees | 2777 | 2.22e+03 | 556 |  |  |  |  | 0.917 | 0.985 | 0.731 | 0.0669 | 0.0346 |
| menin_Ki_biochemical_binding | scaffold | scaffold | trained | extra_trees | 117 | 94 | 23 | 0.718 | 0.869 | 0.561 | 0.686 |  |  |  |  |  |
| menin_Kd_biochemical_binding | scaffold |  | insufficient_data |  | 20 |  |  |  |  |  |  |  |  |  |  |  |
| menin_EC50_cellular_functional | scaffold |  | insufficient_data |  | 0 |  |  |  |  |  |  |  |  |  |  |  |
| menin_IC50_biochemical_inhibition | scaffold | scaffold | trained | ridge_alpha_10 | 849 | 679 | 170 | 0.736 | 0.932 | 0.547 | 0.746 |  |  |  |  |  |
| menin_IC50_in_vivo | scaffold | scaffold | trained | extra_trees | 173 | 140 | 33 | 0.356 | 0.444 | 0.719 | 0.916 |  |  |  |  |  |
| menin_cross_source_mirrors_retained_sensitivity | scaffold | scaffold | trained | extra_trees | 849 | 680 | 169 | 0.761 | 0.939 | 0.537 | 0.732 |  |  |  |  |  |
| menin_clean_label_sensitivity | scaffold | scaffold | trained | extra_trees | 847 | 678 | 169 | 0.757 | 0.933 | 0.544 | 0.738 |  |  |  |  |  |
| herg_pooled_sensitivity | scaffold | scaffold | trained | extra_trees | 8621 | 6.9e+03 | 1.72e+03 |  |  |  |  | 0.851 | 0.974 | 0.626 | 0.0817 | 0.0231 |

Random-split results are included as a sensitivity analysis, not the headline estimate. Scaffold, chemical-cluster, and temporal results better probe prospective chemical generalization; any requested strategy fallback is recorded in the `actual_split` column and split manifest.

## Generalization and safety stress test

- Menin temporal holdout: MAE 1.302 pActivity, R² -2.737, and Spearman r 0.159. This is the most prospective-like public stress test and should govern expectations under drift.
- hERG temporal holdout: ROC AUC 0.641, balanced accuracy 0.551, and Brier score 0.206.
- The primary hERG training population is 86.9% blockers. Precision–recall AUC must therefore be interpreted with the class prevalence; ROC AUC, balanced accuracy, specificity, and calibration are reported alongside it.
- Of 849 Menin structures scored for hERG, 419 (49.4%) are inside the hERG applicability domain and only 1 structure has an observed primary hERG record. Communication-band counts are high=836, medium=13; these are liability flags, not experimentally validated rankings.

## Chemical intelligence and experimental prioritization

The primary chemical-intelligence population contains 849 unique structures across 252 Bemis–Murcko/exact-acyclic series and 126 Butina clusters.
Configured high-similarity analysis found 502 fingerprint activity cliffs; single-cut MMP analysis found 3,899 pairs, including 498 ≥100-fold potency cliffs. These are SAR review targets, not causal transformations.
Evidence-first tier counts are priority 2 potent safety data gap=162, priority 3 potent liability flag=135, priority 4 chemistry review=31, priority 5 context only=521. No out-of-domain hERG prediction receives safety credit, structural alerts are review flags rather than automatic exclusions, and the safety-free discovery score is labeled separately from the complete-evidence score.

Approved Menin-inhibitor reference coverage:

| name | regulatory_status | has_exact_primary_task_structure | has_any_public_menin_measurement | public_menin_assay_families | maximum_primary_achiral_tanimoto | maximum_primary_chiral_tanimoto |
| --- | --- | --- | --- | --- | --- | --- |
| revumenib | FDA approved | False | True | in_vivo | 0.538 | 0.553 |
| ziftomenib | FDA approved | False | False |  | 0.68 | 0.68 |

The approved-reference panel is a dated coverage benchmark, not a comparator efficacy analysis. Regulatory status does not make public assay contexts interchangeable, and absent primary-task coverage is reported rather than imputed.

Largest primary-task chemical series:

| series_id | series_size | median_p_activity | best_p_activity | activity_span_log10 |
| --- | --- | --- | --- | --- |
| SER-F7B9115420359440 | 176 | 6.56 | 8.53 | 4.48 |
| SER-FBD8412D03D2F209 | 47 | 5.96 | 8.4 | 5.36 |
| SER-B92687DE56E5CCCB | 44 | 6.05 | 8.22 | 4.44 |
| SER-6F3389DAFB00746E | 25 | 5.77 | 8.77 | 4.34 |
| SER-317158EDAD8598EE | 25 | 6.35 | 7.7 | 4.17 |
| SER-D0CCC78DDD1F06F9 | 22 | 7.85 | 8.7 | 3.84 |
| SER-20F67B855744F824 | 19 | 8.1 | 8.52 | 1.99 |
| SER-8B11BACADDC8B4B4 | 15 | 7.92 | 8.52 | 2.29 |
| SER-AABF4AB999A5F725 | 15 | 7.59 | 8 | 0.919 |
| SER-789386810AA6A1B8 | 13 | 5.74 | 7.51 | 4.19 |
| SER-BF392468AE9FDA19 | 12 | 6.59 | 7.73 | 3.29 |
| SER-647213101879D186 | 10 | 8.26 | 8.52 | 2.25 |

Top same-context activity cliffs:

| structure_id_a | structure_id_b | absolute_delta_pactivity | achiral_morgan_tanimoto | chiral_morgan_tanimoto | evidence_context_grade |
| --- | --- | --- | --- | --- | --- |
| STR-8A96AEE3549DB26F33DD | STR-A03DAF6611859C5C8983 | 4.76 | 0.863 | 0.691 | same_assay |
| STR-72BF5FE38870FA3B6280 | STR-7E1F0EE1EE4F776E8704 | 4.48 | 0.8 | 0.719 | same_assay |
| STR-2CFC8301293C2B82E4A6 | STR-7E1F0EE1EE4F776E8704 | 4.46 | 0.83 | 0.731 | same_assay |
| STR-09E54D8A9B952AC86D2A | STR-A723964BC2F481BF1509 | 4.44 | 0.835 | 0.682 | same_assay |
| STR-377B42435916FCE77AA8 | STR-A723964BC2F481BF1509 | 4.44 | 0.835 | 0.682 | same_assay |
| STR-3BD3AAFB21D6F0AE245A | STR-7E1F0EE1EE4F776E8704 | 4.38 | 0.814 | 0.714 | same_assay |
| STR-7C604423619052A15AC8 | STR-A723964BC2F481BF1509 | 4.37 | 0.857 | 0.706 | same_assay |
| STR-990708AC2AD0842F3BCB | STR-A723964BC2F481BF1509 | 4.26 | 0.833 | 0.686 | same_assay |
| STR-79D75B6A88F0457FA1C2 | STR-A723964BC2F481BF1509 | 4.22 | 0.835 | 0.69 | same_assay |
| STR-241F2E6DC7BEF8AE2212 | STR-7E1F0EE1EE4F776E8704 | 4.18 | 0.811 | 0.716 | same_assay |
| STR-94C191645BE5B0A1978A | STR-A723964BC2F481BF1509 | 3.96 | 0.823 | 0.678 | same_assay |
| STR-13418EBC6BF2EEA6DE7C | STR-2FF2DB436296505D61BF | 3.87 | 0.838 | 0.837 | same_assay |

The analysis also exports 498 MMP cliffs and 153 connectivity-equivalent variant groups for source-level review.

Experimental follow-up tiers:

| structure_id | p_activity_median | qed | property_window_violation_count | herg_evidence_status | experimental_followup_tier | discovery_rank_without_safety | complete_evidence_rank |
| --- | --- | --- | --- | --- | --- | --- | --- |
| STR-32FEF1FCB73DC9ADA20D | 8.52 | 0.383 | 0 | unknown_outside_applicability_domain | priority_2_potent_safety_data_gap | 3 |  |
| STR-E4F36A98E00A8B27E8BD | 8.7 | 0.241 | 1 | unknown_outside_applicability_domain | priority_2_potent_safety_data_gap | 5 |  |
| STR-DA4F708C5ED8977F505E | 8.5 | 0.322 | 1 | unknown_outside_applicability_domain | priority_2_potent_safety_data_gap | 6 |  |
| STR-213F84F38B4A2B9413E0 | 7.83 | 0.464 | 0 | unknown_outside_applicability_domain | priority_2_potent_safety_data_gap | 11 |  |
| STR-972288F8D4D50B5E44FE | 7.8 | 0.416 | 0 | unknown_outside_applicability_domain | priority_2_potent_safety_data_gap | 12 |  |
| STR-971462EFAE13318C0E1F | 8.52 | 0.376 | 0 | unknown_outside_applicability_domain | priority_2_potent_safety_data_gap | 22 |  |
| STR-34DA364C524542BAF54B | 9 | 0.273 | 1 | unknown_outside_applicability_domain | priority_2_potent_safety_data_gap | 23 |  |
| STR-070A2DB3A5A9041ED5EA | 8.4 | 0.419 | 0 | unknown_outside_applicability_domain | priority_2_potent_safety_data_gap | 24 |  |
| STR-4DD88D5839EF83BB667C | 8.4 | 0.528 | 0 | unknown_outside_applicability_domain | priority_2_potent_safety_data_gap | 25 |  |
| STR-F10D2D8306F0D924B76B | 8.7 | 0.241 | 1 | unknown_outside_applicability_domain | priority_2_potent_safety_data_gap | 27 |  |
| STR-70C7272DAD1CFA014D53 | 8.15 | 0.56 | 0 | unknown_outside_applicability_domain | priority_2_potent_safety_data_gap | 28 |  |
| STR-33ECA7BC02C166AF1729 | 7.6 | 0.543 | 0 | unknown_outside_applicability_domain | priority_2_potent_safety_data_gap | 33 |  |
| STR-718AB1E455060CA519A1 | 7.81 | 0.384 | 0 | unknown_outside_applicability_domain | priority_2_potent_safety_data_gap | 34 |  |
| STR-5C740F62A4861DEC230E | 7.34 | 0.729 | 0 | unknown_outside_applicability_domain | priority_2_potent_safety_data_gap | 35 |  |
| STR-FF8C3D50A964C03AC597 | 7.19 | 0.793 | 0 | unknown_outside_applicability_domain | priority_2_potent_safety_data_gap | 37 |  |
| STR-DC0F93BC28332B505B19 | 7.58 | 0.379 | 1 | unknown_outside_applicability_domain | priority_2_potent_safety_data_gap | 38 |  |
| STR-8053B83BB63EAAF32F61 | 8.39 | 0.437 | 0 | unknown_outside_applicability_domain | priority_2_potent_safety_data_gap | 52 |  |
| STR-7A321275AB46AC4BC765 | 8.7 | 0.237 | 1 | unknown_outside_applicability_domain | priority_2_potent_safety_data_gap | 53 |  |
| STR-6F8F59E84BD75C74A25C | 8.52 | 0.373 | 1 | unknown_outside_applicability_domain | priority_2_potent_safety_data_gap | 54 |  |
| STR-72BC2AEC0C9B5FB0C4A6 | 8.7 | 0.324 | 1 | unknown_outside_applicability_domain | priority_2_potent_safety_data_gap | 56 |  |

These are transparent experimental follow-up categories, not validated drug candidates. Priority 2 explicitly means a potent public-data profile with a safety evidence gap; Priority 3 explicitly carries a modeled or observed liability flag.

Prospective experimental design quotas:

| selection_category | requested_quota | selected_structures | shortfall |
| --- | --- | --- | --- |
| potent_safety_gap | 12 | 12 | 0 |
| liability_characterization | 8 | 8 | 0 |
| novel_scaffold_exploration | 8 | 8 | 0 |
| activity_cliff_confirmation | 8 | 8 | 0 |
| negative_control | 6 | 6 | 0 |
| pk_bridge | 6 | 6 | 0 |

| selection_order | selection_category | structure_id | p_activity_median | series_id | herg_evidence_status | selection_rationale |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | potent_safety_gap | STR-32FEF1FCB73DC9ADA20D | 8.52 | SER-8B11BACADDC8B4B4 | unknown_outside_applicability_domain | potent/property-qualified public evidence; obtain direct hERG and compatible PK before progression |
| 2 | potent_safety_gap | STR-E4F36A98E00A8B27E8BD | 8.7 | SER-034771967D4F78CF | unknown_outside_applicability_domain | potent/property-qualified public evidence; obtain direct hERG and compatible PK before progression |
| 3 | potent_safety_gap | STR-DA4F708C5ED8977F505E | 8.5 | SER-0A8207C373DE3400 | unknown_outside_applicability_domain | potent/property-qualified public evidence; obtain direct hERG and compatible PK before progression |
| 4 | potent_safety_gap | STR-213F84F38B4A2B9413E0 | 7.83 | SER-8766DF06D1ECE89F | unknown_outside_applicability_domain | potent/property-qualified public evidence; obtain direct hERG and compatible PK before progression |
| 5 | potent_safety_gap | STR-972288F8D4D50B5E44FE | 7.8 | SER-3BBBF6CEAC67F2A5 | unknown_outside_applicability_domain | potent/property-qualified public evidence; obtain direct hERG and compatible PK before progression |
| 6 | potent_safety_gap | STR-971462EFAE13318C0E1F | 8.52 | SER-20F67B855744F824 | unknown_outside_applicability_domain | potent/property-qualified public evidence; obtain direct hERG and compatible PK before progression |
| 7 | potent_safety_gap | STR-34DA364C524542BAF54B | 9 | SER-B212B95A94D293C5 | unknown_outside_applicability_domain | potent/property-qualified public evidence; obtain direct hERG and compatible PK before progression |
| 8 | potent_safety_gap | STR-070A2DB3A5A9041ED5EA | 8.4 | SER-8B11BACADDC8B4B4 | unknown_outside_applicability_domain | potent/property-qualified public evidence; obtain direct hERG and compatible PK before progression |
| 9 | potent_safety_gap | STR-4DD88D5839EF83BB667C | 8.4 | SER-0CFBA55DF63719F3 | unknown_outside_applicability_domain | potent/property-qualified public evidence; obtain direct hERG and compatible PK before progression |
| 10 | potent_safety_gap | STR-F10D2D8306F0D924B76B | 8.7 | SER-01C7163C23A2198D | unknown_outside_applicability_domain | potent/property-qualified public evidence; obtain direct hERG and compatible PK before progression |
| 11 | potent_safety_gap | STR-70C7272DAD1CFA014D53 | 8.15 | SER-A61E4F29FD542291 | unknown_outside_applicability_domain | potent/property-qualified public evidence; obtain direct hERG and compatible PK before progression |
| 12 | potent_safety_gap | STR-33ECA7BC02C166AF1729 | 7.6 | SER-88BF110C8AC128BC | unknown_outside_applicability_domain | potent/property-qualified public evidence; obtain direct hERG and compatible PK before progression |
| 13 | liability_characterization | STR-34FE1D5BDCFE7B2EB7FE | 9 | SER-87DF4ACBBC600AE2 | predicted_high_concern | potent chemistry with modeled/observed hERG concern; characterize liability and exposure margin |
| 14 | liability_characterization | STR-9E4FCF8CA68285C7D759 | 8.4 | SER-B210E620CAC4DB11 | predicted_high_concern | potent chemistry with modeled/observed hERG concern; characterize liability and exposure margin |
| 15 | liability_characterization | STR-768449AFD0A1BD14A41D | 8.52 | SER-647213101879D186 | predicted_high_concern | potent chemistry with modeled/observed hERG concern; characterize liability and exposure margin |
| 16 | liability_characterization | STR-0B18427EF6CEB8CC075B | 8.4 | SER-FBD8412D03D2F209 | predicted_high_concern | potent chemistry with modeled/observed hERG concern; characterize liability and exposure margin |
| 17 | liability_characterization | STR-4A902F7AC5A334C3F331 | 8.77 | SER-6F3389DAFB00746E | predicted_high_concern | potent chemistry with modeled/observed hERG concern; characterize liability and exposure margin |
| 18 | liability_characterization | STR-33EC56FC5D9EE8140CC4 | 8.7 | SER-D0CCC78DDD1F06F9 | predicted_high_concern | potent chemistry with modeled/observed hERG concern; characterize liability and exposure margin |
| 19 | liability_characterization | STR-BAFF2FF7D3D8F1D2F783 | 8.44 | SER-E8FAB356E73C31D0 | predicted_high_concern | potent chemistry with modeled/observed hERG concern; characterize liability and exposure margin |
| 20 | liability_characterization | STR-1F39727F90F76C636B4B | 8.4 | SER-6F3389DAFB00746E | predicted_high_concern | potent chemistry with modeled/observed hERG concern; characterize liability and exposure margin |

This configurable selection is a pre-experimental design spanning exploitation, liability characterization, novelty, cliff confirmation, negative controls, and PK bridges. It must be frozen before testing and does not count as prospective validation until new blinded measurements are returned.

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
{
  "gate_scope": "analysis-eligible rows; source-inventory findings remain visible",
  "generated_at": "2026-07-14T22:41:32Z",
  "passed": true,
  "source_inventory": {
    "herg": {
      "finding_count": 25241,
      "passed": false,
      "row_count": 41078,
      "severity_counts": {
        "error": 13668,
        "warning": 11573
      }
    },
    "menin": {
      "finding_count": 24186,
      "passed": false,
      "row_count": 8176,
      "severity_counts": {
        "error": 8099,
        "info": 9,
        "warning": 16078
      }
    },
    "pk": {
      "finding_count": 191,
      "passed": false,
      "row_count": 306,
      "severity_counts": {
        "error": 113,
        "info": 2,
        "warning": 76
      }
    }
  },
  "tables": {
    "herg": {
      "finding_count": 5300,
      "passed": true,
      "row_count": 17654,
      "severity_counts": {
        "warning": 5300
      }
    },
    "menin": {
      "finding_count": 6343,
      "passed": true,
      "row_count": 3798,
      "severity_counts": {
        "warning": 6343
      }
    },
    "pk": {
      "finding_count": 25,
      "passed": true,
      "row_count": 204,
      "severity_counts": {
        "info": 2,
        "warning": 23
      }
    }
  }
}
```

Quality findings are not all fatal errors: the detailed CSVs distinguish missing public metadata, quarantined measurements, repeated-measurement conflicts, and schema violations. Modeling uses only rows that pass the explicit eligibility policy.

## Publication readiness (9/14 infrastructure criteria currently satisfied)

| criterion | satisfied | evidence_or_action |
| --- | --- | --- |
| Defined biological endpoint | True | Primary tasks are IC50 × biochemical_binding and scoped hERG IC50 × electrophysiology_functional. |
| Unambiguous RDKit algorithm | True | Both trained primary manifests must record RDKit Morgan features and selected algorithms. |
| Complete requested holdouts | True | Every configured split must train and must not silently fall back. |
| Applicability domain | True | Both primary tasks record nearest-neighbor domain policy and coverage. |
| Uncertainty | True | Primary Menin metrics include scaffold-group bootstrap intervals and conformal prediction intervals. |
| Build-linked provenance | False | Processed, model, and report manifests plus model provenance must share a content build. |
| Upstream manifest verification | True | Raw, processed, software, and model inputs must verify before report generation; the final report bundle is verified as the release's last step. |
| Data quality audit | True | The analysis-eligible quality gate must pass; source-inventory exclusions remain reported. |
| Environment lock | True | The frozen environment lock must be included in the verified software manifest. |
| Chemical-intelligence decision trace | True | Primary-task properties, series, cliffs, evidence gaps, Pareto fronts, and applicability-aware tiers must be content-linked. |
| Clean source revision | False | A publication release must be rebuilt from a clean committed revision. |
| Independent external validation | False | No independent Menin or hERG external test set has been reserved. |
| Prospective experimental validation | False | Requires new lab measurements after model lock. |
| Authorship and licensing approval | False | Requires explicit project-owner and Wang-lab decisions. |

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
- [Medicinal-chemistry property landscape](figures/menin_medchem_property_landscape.png)
- [Chemical-series coverage](figures/menin_chemical_series_sizes.png)
- [Activity-cliff landscape](figures/menin_activity_cliff_landscape.png)
- [Experimental follow-up tiers](figures/menin_followup_tier_counts.png)
- [Complete-evidence Pareto frontier](figures/menin_complete_evidence_frontier.png)

## Interpretation boundaries

- Public assay values remain heterogeneous across biochemical, biophysical, and cellular contexts.
- Censored values are preserved as bounds but excluded from point-regression labels; no censored-likelihood model is claimed.
- hERG thresholds define a screening label, not clinical cardiotoxicity, and probabilities require external calibration checks.
- Out-of-domain hERG probabilities are retained for audit but classified as unknown and receive no safety credit.
- PAINS, Brenk, and NIH substructure matches are review alerts, not proof of interference, toxicity, or inactivity.
- Apparent ligand-efficiency and lipophilic-efficiency values use IC50-derived pActivity and are not binding free energies.
- PK/ADMET observations are coverage evidence only until endpoint, species, matrix, route, and units support separate validated tasks.
- Model artifacts must be loaded only from this trusted build and matched to their manifests.

## Target anchors

- Menin/MEN1: CHEMBL1615381; UniProt O00255.
- hERG/KCNH2: CHEMBL240; UniProt Q12809.

The complete methods, data dictionary, architecture, limitations, reproducibility instructions, and internal-data intake contract are maintained under `docs/`.
