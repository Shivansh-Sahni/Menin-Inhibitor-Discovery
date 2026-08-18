# Current wild-type hERG data analysis

## Executive result

The present hierarchy supports **339,373 Q0 structures**, **23,186 Q1 quantitative records**, and **3,557 eligible Q2 functional records**. The core scientific advantage is not row count alone: target scope, evidence quality, assay technology, automation, dose design, scaffold split, clinical development, and QT/QTc context remain separately addressable.

This analysis establishes data assets and testable hypotheses. It does **not** establish causality, clinical validity, or superiority over published models. Superiority must be shown on identical locked and prospective challenges.

## Quality and prevalence

| Layer | Eligible rows | Structures | Positive prevalence | Meaning |
|---|---:|---:|---:|---|
| C0 | 3,056 | 3,056 | NA | clinical-development context only |
| C1 | 221 | 95 | NA | QT/QTc context only; never a hERG label |
| Q0 | 339,373 | 339,373 | 0.00365 | large weak fixed-dose binary screen |
| Q1 | 23,186 | 22,081 | 0.44377 | quantitative pIC50 potency task |
| Q2 | 3,557 | 2,785 | NA | functional assay-aware task |

Q0's low natural prevalence makes accuracy unsuitable as a headline metric. PR-AUC, calibration, false-negative rate, enrichment, and recall at a fixed testing budget are required. Q1 prevalence is conditional on the blocker/non-blocker zones; its 13,947 gray-zone records are excluded from that denominator and remain available for ordinal/regression analysis.

## Measurement landscape

| Modality | Observations | Structures |
|---|---:|---:|
| high_throughput_thallium_flux | 344,029 | 340,212 |
| unresolved | 24,654 | 23,347 |
| patch_clamp_electrophysiology | 16,200 | 8,943 |
| binding_unspecified | 10,064 | 7,952 |
| radioligand_binding | 9,787 | 7,017 |
| functional_electrophysiology | 1,667 | 1,216 |
| functional_unspecified | 816 | 715 |
| functional_ion_flux | 444 | 397 |
| clinical_qt_in_vivo | 37 | 35 |

The strongest categorical entanglement is **source_family × endpoint_grouped** (bias-corrected Cramér's V 0.998; n=407,698). This is evidence that naive pooling can confound source, method, or protocol—not evidence that either variable causes potency.

## Matched-structure disagreement

Only exact structure matches are compared. Binary values are within-group majority consensuses; ties are excluded. Quantitative comparisons use exact pIC50 means and remain protocol-confounded.

| Axis | Pair | Outcome | Matched | Agreement | Mean absolute ΔpIC50 | Spearman |
|---|---|---|---:|---:|---:|---:|
| measurement_modality | high_throughput_thallium_flux vs unresolved | binary_consensus | 591 | 0.345 | NA | NA |
| measurement_modality | functional_unspecified vs unresolved | binary_consensus | 49 | 0.980 | NA | NA |
| measurement_modality | patch_clamp_electrophysiology vs unresolved | binary_consensus | 33 | 0.939 | NA | NA |
| measurement_modality | functional_unspecified vs high_throughput_thallium_flux | binary_consensus | 29 | 0.379 | NA | NA |
| measurement_modality | functional_electrophysiology vs unresolved | binary_consensus | 18 | 1.000 | NA | NA |
| source_family | pubchem_aid720551 vs quantitative_pic50_release | binary_consensus | 591 | 0.345 | NA | NA |
| source_family | chembl_herg_specialized_view vs quantitative_pic50_release | binary_consensus | 97 | 0.979 | NA | NA |
| source_family | chembl_herg_specialized_view vs pubchem_aid720551 | binary_consensus | 33 | 0.424 | NA | NA |
| measurement_modality | functional_unspecified vs unresolved | exact_pic50_mean | 55 | NA | 0.289 | 0.942 |
| measurement_modality | patch_clamp_electrophysiology vs unresolved | exact_pic50_mean | 42 | NA | 0.511 | 0.895 |
| measurement_modality | functional_electrophysiology vs unresolved | exact_pic50_mean | 18 | NA | 0.127 | 0.948 |
| source_family | chembl_herg_specialized_view vs quantitative_pic50_release | exact_pic50_mean | 112 | NA | 0.327 | 0.919 |

## Quantitative replicate stability

There are **843 structures with at least two exact pIC50 records**. Large ranges identify audit priorities, not automatic outliers: protocol, source, cell context, and temperature can create real measurement differences.

| Structure | Records | Sources | pIC50 range | SD |
|---|---:|---:|---:|---:|
| HSTR-0688D35877B5D910E17A315D | 4 | 2 | 5.000 | 2.424 |
| HSTR-5D8C164A5C3492D70E69E774 | 8 | 2 | 3.944 | 1.364 |
| HSTR-2E08D22E4E8E60AF6E5FDC0F | 2 | 1 | 3.530 | 2.496 |
| HSTR-3A0D2BB0E217024AEF10DE4E | 6 | 2 | 3.440 | 1.374 |
| HSTR-90520B8F9DF9527BED683CA7 | 3 | 2 | 3.400 | 1.963 |
| HSTR-42B474FF9DE1D4755F5C5ED0 | 3 | 2 | 3.217 | 1.829 |
| HSTR-2F89843AAE968BEC70FD1EC6 | 2 | 1 | 3.003 | 2.124 |
| HSTR-35D83AD6A7E92B5976838BBE | 2 | 1 | 3.000 | 2.122 |
| HSTR-0B7C19573F6958A90B3CB55A | 2 | 1 | 3.000 | 2.121 |
| HSTR-1E3DC246860CDFFFEFCDDA00 | 2 | 1 | 3.000 | 2.121 |

## Fundamental molecular features

The strongest univariate separator in the weak Q0 screen is **logp** (active-higher AUC 0.630, standardized mean difference 0.468). These are exploratory structure–label associations and may partly reflect library composition or screen selection.

The largest prespecified two-feature additive-logit residual is **logp × tpsa**, tertiles 1/1 (residual -0.381, prevalence lift 0.565; n=21,830). It is a hypothesis for matched-series and assay-aware confirmation, not mechanistic proof.

Descriptor deciles, all prespecified interaction cells, and the full structure-level compact descriptor matrix are retained as machine-readable artifacts.

## Scaffold-split shift

The largest absolute standardized train-versus-validation/test shift among class prevalence and compact descriptors is **0.389 SD**. Small marginal shifts do not make the task easy: scaffold separation can preserve global property distributions while removing close analogs.

## Clinical and QT/QTc coverage

Clinical-development and QT/QTc records are context/evaluation layers only. They are never promoted into direct molecular hERG potency labels. Their overlap tables quantify how much of the molecular hierarchy can currently support downstream exposure/QT translation.

## What this project can credibly emphasize

1. Larger public coverage is combined with explicit WT scope and immutable provenance.
2. Weak fixed-dose, quantitative potency, functional assays, clinical context, and QT/QTc are separate tasks instead of pooled labels.
3. Measurement modality, automation, dose design, source, and scaffold are measurable evaluation axes.
4. Natural-prevalence analyses expose the false-positive problem hidden by balanced benchmarks.
5. Fundamental descriptors and prespecified interactions generate falsifiable hypotheses.
6. These are design superiorities. Predictive superiority remains a future empirical result until competitors are reproduced on identical locked/prospective data.

## Required next analyses

- Reproduce published comparators on the frozen split and modality/source holdouts.
- Add temperature, voltage protocol, cell line, incubation, and platform fields where source text supports them.
- Validate feature interactions in matched molecular pairs and quantitative Q1/Q2 data.
- Add unbound exposure, metabolites, and multi-ion-channel data before claiming QT-risk prediction.
- Reserve a blinded multi-laboratory manual-patch panel for prospective validation.

## Artifact contract

Schema `platform-herg-current-analysis/1.0`; RDKit `2026.03.3`. All Parquet outputs and their SHA-256 digests are recorded in the manifest. Inputs are content-bound; upstream artifacts are not modified.
