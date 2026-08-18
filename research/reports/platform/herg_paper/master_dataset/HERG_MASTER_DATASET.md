# Comprehensive wild-type hERG master dataset (v1.3)

This release is the paper-facing join layer. It does not replace or mutate any native source artifact.

## Scale

- 407,698 admitted wild-type-scope observations.
- 369,546 standardized molecular structures with CPU-feasible RDKit 2D features.
- 446 structure entities retain multiple reported standardized representations for sensitivity analysis.
- 4,782 source-aware assay catalog entries.
- 370,167 quality/clinical task memberships.
- 3,277 clinical-development or QT/QTc context rows.
- 1,041 explicit scope or task-specific exclusions retained in quarantine.

## Established design advantages over existing hERG model datasets

- Explicit mutants are excluded; confirmed WT and WT-or-unspecified evidence are never conflated.
- Native endpoint, relation, value, unit, assay, source row, auxiliary metadata, and lineage remain recoverable.
- Fixed-dose activity, exact/censored pIC50, other endpoints, measurement modality, and clinical QT/QTc are separate axes.
- Manual/automated evidence, assay technology, source lineage, scaffold partition, and quality-task membership support transport audits that most pooled benchmarks omit.
- Natural censoring is represented with valid pIC50 bounds; no threshold value is fabricated.
- Clinical QT/QTc remains label-disabled, preventing a downstream exposure-dependent phenotype from leaking into a direct hERG target.
These are established data-design superiorities, not a claim of predictive superiority; model superiority still requires locked external and prospective comparisons.

## Measurement coverage

- high_throughput_thallium_flux: 344,029 observations.
- unresolved: 24,654 observations.
- patch_clamp_electrophysiology: 16,200 observations.
- binding_unspecified: 10,094 observations.
- radioligand_binding: 9,787 observations.
- functional_electrophysiology: 1,667 observations.
- functional_unspecified: 816 observations.
- functional_ion_flux: 444 observations.
- clinical_qt_in_vivo: 7 observations.

## Endpoint coverage

- categorical_activity_call: 343,909 observations.
- potency_pic50: 23,099 observations.
- potency_ic50: 19,102 observations.
- inhibition_or_effect_level: 8,166 observations.
- binding_or_effect_kinetics: 5,943 observations.
- binding_affinity_ki: 3,720 observations.
- potency_ac50: 1,823 observations.
- potency_other_or_derived: 922 observations.
- other_reported_endpoint: 649 observations.
- effect_ec50: 253 observations.
- ratio_or_fold_change: 61 observations.
- binding_affinity_kd: 47 observations.
- clinical_qt_qtc_phenotype: 4 observations.

The modality and endpoint totals answer different questions: seven rows belong to curated clinical-QT phenotype assays, while only four natively report a `QT interval` endpoint. The other three retain their source-reported EC10 endpoint class and are still clinical-context-only, never direct hERG labels.

## Standardized potency status

- not_standardized: 367,657 observations.
- exact_standardized: 34,957 observations.
- censored_standardized: 5,074 observations.
- value_available_relation_unresolved: 6 observations.
- approximate_standardized: 4 observations.

## Structure features and intentional omissions

The structure table contains only deterministic 2D physicochemical descriptors plus identifiers and frozen split metadata. `fundamental_feature_summary.parquet` reports coverage and empirical ranges. The manifest lists the only feature-eligible columns. Evidence-density summaries are isolated and explicitly label-ineligible. No pKa, protonation ensemble, docking pose, channel state, 3D conformer, membrane partition coefficient, or binding free energy was invented.

## Analysis-ready audit tables

`method_endpoint_summary.parquet` cross-tabulates target certainty, measurement technology, automation, dose design, endpoint semantics, structure coverage, and potency standardization. It supports prespecified method-impact analyses without altering labels. `structure_evidence_summary.parquet` supports coverage audits but is explicitly prohibited as a molecular feature because measurement density can encode outcome and selection bias.

`assay_protocol_index.parquet` preserves raw assay text and normalizes only
explicit host system, voltage, temperature, time, recording configuration,
named platform, and manual-operation evidence. Raw-text evidence and curated
source-contract evidence are stored in separate JSON columns. In particular,
the frozen AID720551 source contract supplies U2OS/FluxOR/thallium-qHTS
metadata that is absent from its abbreviated ledger title; this produces
343,910 U2OS-linked and 343,957 FluxOR-linked observations without pretending
those terms were text-mined. Every normalized field carries evidence and
confidence; missing protocol details remain unresolved.

## Use

Use `observation_master.parquet` for source-faithful endpoint and method analysis, `structure_master.parquet` for leakage-controlled molecular inputs, `task_membership.parquet` for labels, and `clinical_context_master.parquet` only for clinical stratification or held-out translation analysis.

No model was trained by this build.
