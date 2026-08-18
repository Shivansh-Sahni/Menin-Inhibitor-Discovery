# Data dictionary

This document describes the processed and analytical contracts. Source-native files under `research/data/raw/` retain their upstream schemas and are inventoried by the raw manifest.

Blank strings and nulls are meaningful: they generally mean the source did not provide the field or the pipeline could not resolve it. They must not be replaced with assumed values.

## Measurement-level bioactivity

Files:

- `research/data/processed/menin_activity_measurements.csv`
- `research/data/processed/herg_activity_measurements.csv`

### Provenance and observation identity

| Column | Meaning |
| --- | --- |
| `source` | Source database, normally `ChEMBL`, `BindingDB`, or `PubChem`. |
| `source_record_id` | Source-specific observation identifier. It may be absent when the source export lacks a stable row ID. |
| `measurement_id` | Deterministic `MEA-…` digest of source, record, compound, assay, endpoint, relation, value, unit, and document fields. |
| `is_duplicate_measurement` | Whether this row has a nonblank source record ID and repeats an earlier row with the same measurement identity fields. Rows without a source record ID are not marked by this flag and require separate duplicate review. |
| `compound_id` | Source-specific compound or substance identifier. |
| `compound_name` | Source name, when provided. |
| `parent_compound_id` | Optional source parent-molecule identifier. |
| `source_record_parent_id` | Optional parent record ID. |
| `source_id` | Optional source-within-source identifier. |
| `document_id` | Source document, patent, publication, or BioAssay identifier. |
| `document_year` | ChEMBL/BindingDB source document year when provided; for PubChem, the assay deposit year. It is not generally the assay/test date. |
| `date_provenance` | Time-field origin when explicitly known. PubChem rows use `pubchem_assay_deposit_date`; blank source rows retain only their source-native document-year semantics. |
| `source_deposit_date`, `source_modify_date` | PubChem assay catalog deposit and modification dates. The deposit year can support the public temporal sensitivity; modification date is audit metadata only. |
| `reference` | Source reference text or stable source citation. |
| `source_detail` | Collector/target/export context. |

### Structure identity

| Column | Meaning |
| --- | --- |
| `original_smiles` | Structure string submitted by the source; never overwritten by standardization. |
| `smiles` | Modeling representation; normally the standardized parent SMILES. |
| `canonical_smiles` | Canonical cleaned full-structure SMILES before fragment-parent selection. |
| `standardized_smiles` | Canonical isomeric parent SMILES after cleanup, fragment-parent selection, and neutralization. |
| `original_inchi_key` | Source InChIKey, retained unchanged. |
| `inchi_key` | Generated standard InChIKey when available, otherwise the source key. |
| `standard_inchi_key` | InChIKey generated from the standardized parent. |
| `structure_id` | Stable `STR-…` digest of the standardized parent plus standardization namespace. |
| `full_structure_id` | Stable `FULL-…` digest of the cleaned full representation. |
| `structure_valid` | RDKit parse/standardization result: true, false, or unresolved when RDKit is unavailable. |
| `structure_standardization_status` | `standardized`, `missing_structure`, `invalid_smiles`, `standardization_failed`, `rdkit_unavailable`, or `not_requested`. |
| `structure_error` | Parse/standardization error text, if any. |
| `structure_standardization_version` | Versioned policy/identity namespace encoding cleanup, neutralization, fragment-parent selection, and tautomer-canonicalization settings. |
| `rdkit_version` | RDKit version used for the row. |
| `fragment_count` | Number of disconnected fragments in the submitted molecule. |
| `formal_charge` | Formal charge of the standardized parent. |

### Target and assay context

| Column | Meaning |
| --- | --- |
| `target_name` | Source target name. |
| `target_id` | Source target identifier/accession. |
| `target_organism` | Optional organism from the source. |
| `target_relevance` | Normalized relevance status, such as confirmed target, manual include/exclude, or unresolved. |
| `is_target_relevant` | Boolean gate indicating that the target is accepted for the current table. |
| `target_relevance_reason` | Evidence or reason for the relevance decision. |
| `assay_id` | Source assay identifier. |
| `assay_name` | Optional assay name. |
| `assay_description` | Source assay description or catalog description. |
| `assay_type` | Source assay type or activity outcome. |
| `assay_format` | Optional source assay/BAO format label. |
| `bao_format_id` | Optional BAO format identifier. |
| `assay_variant_accession` | Optional target-variant accession. |
| `assay_variant_mutation` | Optional mutation text; populated variants are excluded by the default policy. |
| `assay_family` | Normalized context such as `biochemical_binding`, `biochemical_inhibition`, `cellular_functional`, `electrophysiology_functional`, biophysical, in vivo, or unclassified. |
| `assay_family_override` | Optional reviewed PubChem registry override. |
| `assay_relevance` | Optional PubChem assay-level relevance result. |
| `activity_comment` | Optional source activity annotation. |

### Endpoint and numeric semantics

| Column | Meaning |
| --- | --- |
| `endpoint_original` | Endpoint text before normalization. |
| `endpoint` | Normalized endpoint, commonly `IC50`, `Ki`, `Kd`, or `EC50`. |
| `endpoint_family` | Normalized family, for example binding or functional. |
| `is_core_endpoint` | Whether the endpoint is eligible under the configured core endpoint set. |
| `relation_original` | Relation as received. |
| `relation` | Normalized `=`, `~`, `<`, `<=`, `>`, or `>=`. |
| `value_raw` | Source value before numeric conversion. |
| `value_numeric` | Parsed numeric value in the original unit. |
| `standard_units_original` | Unit text before normalization. |
| `standard_units` | Source/adaptor unit field. |
| `unit_normalized` | Canonical unit token. |
| `unit_conversion_status` | `converted`, `missing_unit`, or `unsupported_unit`. |
| `value_nm` | Positive concentration converted to nM; null if conversion is not defensible. |
| `p_value` | `9 - log10(value_nm)` for a valid positive converted value. For censored data this is the transformed threshold, not an exact label. |
| `is_exact` | Relation is `=`. |
| `is_censored` | Relation is one of `<`, `<=`, `>`, `>=`. |
| `censoring_direction` | `left`, `right`, `approximate`, `none`, or `invalid` in concentration space. |
| `value_nm_lower_bound` | Concentration lower bound implied by the relation. |
| `value_nm_upper_bound` | Concentration upper bound implied by the relation. |
| `p_activity_lower_bound` | pActivity lower bound implied by the relation. |
| `p_activity_upper_bound` | pActivity upper bound implied by the relation. |
| `p_value_semantics` | `point_estimate`, `approximate_point`, `censoring_threshold`, or `invalid`. |
| `measurement_origin` | Field/rule used to obtain the value, especially for PubChem. |

### Source quality and eligibility

| Column | Meaning |
| --- | --- |
| `standard_flag` | ChEMBL standard measurement flag. |
| `data_validity_comment` | ChEMBL data-validity annotation. |
| `data_validity_description` | Longer ChEMBL validity text. |
| `potential_duplicate` | ChEMBL potential-duplicate flag. |
| `reported_pchembl_value` | pChEMBL supplied by ChEMBL. |
| `pchembl_delta` | Absolute difference between reported pChEMBL and the pipeline conversion. |
| `quality_flags` | Semicolon-separated review/exclusion indicators. |
| `exclusion_reason` | Semicolon-separated reasons the row is not eligible for the ordinary modeling table. Empty means eligible. |
| `is_modeling_eligible` | Authoritative boolean eligibility gate. |
| `requires_review` | Whether any quality flag is present. |

### Cross-source mirror linkage

Potential cross-source mirrors are annotated after normalization. The measurement rows are never removed.

| Column | Meaning |
| --- | --- |
| `cross_source_mirror_group_id` | Stable `MIR-…` identifier for exact normalized rows sharing structure, endpoint, assay family, relation, and pActivity across two or more sources. Blank means the row was not linked. |
| `is_cross_source_mirror_candidate` | Whether the row belongs to such a multi-source exact-match group. This is heuristic evidence, not proof of common experimental origin. |
| `cross_source_mirror_preferred_source` | Source retained for the central aggregate under the fixed priority ChEMBL → BindingDB → PubChem. |
| `is_cross_source_mirror_redundant` | Lower-priority linked row omitted from the central aggregate. It remains in the measurement/provenance table and is restored by the no-collapse Menin sensitivity analysis. |

Same-source replicates are not marked redundant by this policy.

Not every source populates every optional column. Source-specific columns are retained to prevent loss of provenance.

## Menin compound task table

File: `research/data/processed/menin_compounds_curated.csv`

Rows are endpoint/assay-family task observations, not necessarily one row per unique structure. `structure_id` must therefore remain the split grouping key.

| Column | Meaning |
| --- | --- |
| `structure_id` | Standardized parent grouping key. |
| `endpoint` | Endpoint stratum when enabled. |
| `assay_family` | Assay-family stratum when enabled. |
| `smiles`, `original_smiles`, `canonical_smiles`, `standardized_smiles` | Representative identity fields from contributing measurements. |
| `inchi_key`, `standard_inchi_key` | Representative standardized chemical keys. |
| `structure_standardization_status` | Representative standardization status. |
| `n_measurements` | Number of contributing measurements after the selected censoring and central mirror-collapse policies. |
| `n_source_rows` | Eligible source rows represented before cross-source mirror collapse. |
| `n_cross_source_mirror_rows_collapsed` | Linked lower-priority rows omitted from the central aggregate. |
| `n_sources` | Number of source labels represented. |
| `n_exact_measurements` | Exact contributing values. |
| `n_censored_measurements` | Censored contributing thresholds. |
| `n_approximate_measurements` | Approximate contributing values. |
| `p_activity_median` | Median transformed value/threshold in the group. Primary regression label only for exact-point groups. |
| `p_activity_best` | Maximum transformed value. |
| `p_activity_min` | Minimum transformed value. |
| `value_nm_median` | Median nM value/threshold. |
| `value_nm_best` | Lowest nM value/threshold. |
| `activity_range_log10` | `p_activity_best - p_activity_min`. |
| `is_activity_heterogeneous` | True when the activity range exceeds the configured threshold (2.0 log units by default) or the group mixes endpoint/assay families. |
| `endpoints`, `endpoint_families`, `assay_families` | Semicolon-delimited contributing categories. |
| `n_endpoints`, `n_endpoint_families`, `n_assay_families` | Counts of contributing categories. |
| `sources`, `compound_ids`, `target_names`, `document_years`, `relations` | Semicolon-delimited provenance summaries, capped where necessary. |
| `censoring_policy` | `strict_exact`, `prefer_exact_per_compound`, or `include_all`. |
| `aggregation_value_semantics` | Exact points, censoring thresholds only, approximate points only, or mixed. |
| `potency_class` | Descriptive bin based on median nM value. |
| `active_100nM`, `active_1uM` | Convenience thresholds based on the median value; not independent measurements. |

## hERG compound task table

File: `research/data/processed/herg_compounds_curated.csv`

It has the same identity, stratification, count, provenance, and aggregation columns as the Menin task table, with these renamed or added fields:

| Column | Meaning |
| --- | --- |
| `p_herg_median` | Median hERG pActivity in the compatible group. |
| `p_herg_best` | Highest hERG pActivity. |
| `herg_value_nm_median` | Median hERG concentration. |
| `herg_value_nm_best` | Lowest hERG concentration. |
| `herg_blocker_label` | `1` at `<=10,000 nM`, `0` at `>=30,000 nM`, null in the ambiguous band. |
| `herg_label_policy` | Human-readable threshold policy used for the row. |

## Quarantine and curation summaries

| File | Contract |
| --- | --- |
| `menin_activity_quarantine.csv` | Menin measurement rows where `is_modeling_eligible` is false. |
| `herg_activity_quarantine.csv` | hERG measurement rows where `is_modeling_eligible` is false. |
| `source_summary.csv` | `source`, measurement count, structure count, and eligible count where available. |
| `data_quality_summary.csv` | Dataset, exclusion reason, affected row count, and affected structure count. One row can contribute to multiple reasons. |
| `build_summary.json` | Processed schema version and row count for each returned processed table. It is operational metadata, not the content-addressed manifest. |

Quarantine membership does not mean a source record is false. It means the record does not meet the current ordinary-modeling policy.

## PK/ADMET observations

Files:

- `research/data/processed/pk_admet_observations.csv`: analysis-ready rows with usable structure, numeric value, and units;
- `research/data/processed/pk_admet_observations_all.csv`: the complete explicitly classified observation inventory; and
- `research/data/processed/pk_admet_quarantine.csv`: classified rows that are not analysis-ready, with reason-coded quality flags.

| Column | Meaning |
| --- | --- |
| `source`, `activity_id`, `molecule_chembl_id` | Upstream observation and molecule identities. |
| `structure_id`, `smiles`, `original_smiles`, `inchi_key` | Standardized and submitted structure identity. |
| `standard_type`, `standard_relation`, `standard_value`, `standard_units` | Backward-compatible ChEMBL measurement fields. |
| `relation`, `value_raw`, `value_numeric` | Common relation/value fields. |
| `admet_category` | Broad class such as pharmacokinetics, metabolic stability, distribution, or permeability. |
| `admet_endpoint` | Explicit endpoint classification, such as intrinsic clearance or fraction unbound. |
| `admet_directionality` | Contextual interpretation such as `higher_is_more_stable`; not a universal optimization objective. |
| `admet_inclusion_rule` | Rule/evidence that caused inclusion. |
| `species`, `matrix`, `administration_route`, `experimental_context` | Context parsed from source metadata. Blank means unresolved. |
| `assay_description`, `assay_type`, `target_chembl_id`, `target_pref_name` | Assay and target context. |
| `document_chembl_id`, `document_year` | Source document provenance. |
| `is_admet_analysis_ready` | Structure, numeric value, and unit are present. It does not assert cross-row comparability. |
| `admet_quality_flags` | Missing structure/value/unit or structure-validation warnings. |
| `structure_standardization_status` | Standardization outcome. |

## Quality audit outputs

For each analysis-eligible table, `research/reports/quality/` contains:

- `<table>.json`: report version, timestamp, input schema, pass/fail, severity counts, summaries, and detailed findings;
- `<table>_findings.csv`: `table`, `code`, `severity`, `scope`, message, row number, column, identifier, value, and JSON context; and
- `<table>_summary.csv`: counts grouped by table, severity, code, and scope.

`<table>` is `menin`, `herg`, or `pk`. Parallel `<table>_inventory*` files audit every classified source row, including the reason-coded quarantine population. `quality_gate.json` gates the analysis-eligible tables and embeds separate source-inventory summaries so expected source exclusions remain visible without making the release gate permanently fail.

`passed` means no error-severity findings in the analysis-eligible population under the selected `QualityConfig`. It does not mean the raw source inventory is clean, nor does it waive review of warnings, information, quarantines, or assay comparability.

## Private intake outputs

These artifacts are written only when `menin_discovery.internal_data.ingest_internal_data` is invoked with an approved private output directory. They are not part of the public pipeline and must not be committed.

| File | Contract |
| --- | --- |
| `internal_measurements.csv` | Accepted, pseudonymized, standardized internal rows. |
| `internal_quarantine.csv` | Rows with one or more error-level validation codes. |
| `internal_validation_issues.csv` | Pseudonymous row ID, severity, code, field, and non-secret message. |
| `internal_validation_summary.json` | Intake/input/configuration/standardization versions and hashes, row/unique pseudonymous entity counts, and issue counts. |

Important accepted/quarantine columns include:

| Column | Meaning |
| --- | --- |
| `internal_row_id` | `IROW-…` HMAC pseudonym when a submitted row ID exists; otherwise the deterministic content-derived `IREC-…` pseudonym. |
| `row_id_basis` | Whether the pseudonym derives from a supplied row ID or record content. |
| `internal_compound_id` | `ICMP-…` HMAC pseudonym of the standardized `structure_id`. |
| `internal_source_compound_id` | `ISRCMP-…` HMAC pseudonym of the supplied compound ID. |
| `internal_batch_id` | `IBAT-…` HMAC pseudonym of supplied batch ID. |
| `internal_assay_id` | `IASSAY-…` HMAC pseudonym of supplied assay ID. |
| `submitted_smiles`, `submitted_value`, `submitted_units`, `submitted_relation`, `submitted_endpoint` | Submitted evidence needed for private review. These fields remain confidential. |
| structure identity fields | The same original/canonical/standardized SMILES, InChIKey, structure IDs, status, policy, and RDKit fields used by public curation. |
| `endpoint`, `endpoint_family`, `target_name`, `target_id`, `assay_family` | Values resolved through the registered endpoint/assay metadata and submitted context. |
| `standard_units`, `unit_source` | Normalized unit and whether it was submitted or explicitly supplied by the assay registry. |
| `relation`, `value`, `value_nm`, `p_value`, `is_censored`, `lower_bound_nm`, `upper_bound_nm` | Normalized numerical/censoring semantics. |
| `measurement_date`, `replicate` | Optional submitted temporal and replicate context. |
| `cohort_role` | Required governed role: `development`, `locked_external`, or `prospective_blind`. Intake validates the vocabulary and reports accepted counts by role. |
| `validation_codes`, `validation_status`, `quality_eligible` | Row outcome and reason codes. |
| `source_record_id`, `compound_id`, `batch_id`, `assay_id` | Common-contract aliases containing only pseudonyms. |
| `source`, `intake_version` | `InternalLab` and the intake-policy version. |

HMAC pseudonyms protect source identifiers from direct disclosure but do not anonymize structure, measurement, target, date, or context fields.

Only data-owner-approved `development` rows may be used for fitting or calibration. `locked_external` rows are evaluation-only. `prospective_blind` labels must remain sealed until the model and analysis plan are locked. The intake library records these roles but does not automatically connect private outputs to the public model CLI, so downstream private orchestration must enforce the separation.

## Model evidence

| Pattern | Important fields |
| --- | --- |
| `research/reports/menin_activity*_model_metrics.json` | Status, task, selected model, population, split/CV metadata, feature backend, holdout metrics, scaffold-group bootstrap intervals, conformal interval, applicability domain, and artifact metadata. |
| `research/reports/herg_classifier_metrics.json` | Primary `IC50 × electrophysiology_functional` status, class counts, label policy, split/CV metadata, calibration method, calibrated and uncalibrated metrics, scaffold-group bootstrap intervals, applicability domain, and artifact metadata. |
| `*_model_comparison.csv` | Candidate, training-CV selection metric/mean/std/fold count, holdout diagnostics, and `selected_from_training_cv`. |
| `*_split_assignments.csv` | Structure/InChIKey, target, task context, source/date context, structure and Bemis–Murcko groups, grouping method, requested/actual strategy, split/dataset digests, and partition assignment. |
| `*_model_test_predictions.csv` or `herg_classifier_test_predictions.csv` | Identity/task context, observed and predicted holdout values, errors or probabilities, uncertainty, nearest training structure, similarity, and domain flag. |
| `herg_classifier_calibration_curve.csv` | Mean predicted probability and observed positive fraction per quantile bin. |
| `research/models/*_manifest.json` | Model schema, artifact class/hash/format/trust note, dataset digest and declared digest-column list, split/CV hashes, feature metadata, selection metric, and package environment. Endpoint/evaluation manifests can be nested below `research/models/`. |
| `model_validation_summary.csv` | One row per task/requested split with actual split, status, selected model, population, and core holdout metrics. |
| `model_validation_summary.json` | Nested metrics for all configured split evaluations, configured endpoint models, additional eligible endpoint × assay-family tasks above minimum support, cross-source-mirror/clean-label Menin sensitivities, and pooled hERG sensitivity. |

The central Menin task is `IC50 × biochemical_binding`. Cross-source-mirror-retained and clean-label sensitivity artifacts are under `research/models/sensitivity/` and `research/reports/sensitivity/`. The headline hERG task is `IC50 × electrophysiology_functional`; pooled hERG artifacts are under `research/models/herg_sensitivity/pooled/` and `research/reports/herg_sensitivity/pooled/` and must not be presented as the primary safety estimate.

## Menin compounds scored for hERG

File: `research/reports/menin_with_predicted_herg_risk.csv`

It retains the Menin task columns and adds:

| Column | Meaning |
| --- | --- |
| `scored_menin_endpoint`, `scored_menin_assay_family` | Menin primary task whose unique standardized structures were scored. |
| `has_observed_primary_herg_record` | Whether the structure already has a curated record in the same primary hERG endpoint/assay scope. |
| `predicted_herg_blocker_probability` | Calibrated-model probability for the defined binary hERG label. |
| `predicted_herg_probability_entropy_bits` | Binary entropy of that probability. |
| `herg_max_training_tanimoto` | Maximum fingerprint similarity to the hERG training set. |
| `herg_nearest_training_smiles` | Nearest hERG training structure. |
| `herg_inside_applicability_domain` | Whether similarity meets the model's recorded domain threshold. |
| `predicted_herg_risk` | `low` at probability `<=0.30`, `high` at `>=0.70`, otherwise `medium`; `unscored` for missing structures. |

The risk band is a communication label, not a measured property or optimized safety threshold.

## Chemical-intelligence outputs

When `analysis.enabled` is true, the `analyze` stage transactionally replaces the configured `research/analysis/` root. The schema is `chemical-intelligence-v1` and is scoped to one row per structure in the configured primary Menin task (`IC50 × biochemical_binding` by default). Empty result tables retain their schemas; absence of a particular tier, series, cliff, matched pair, or approved reference in the primary task is valid evidence rather than an execution error.

| File | Content and important fields |
| --- | --- |
| `research/analysis/medicinal_chemistry_profiles.csv` | Primary-task identity/potency/context; RDKit molecular weight, exact weight, logP, TPSA, H-bond, rotatable-bond, ring, fraction-sp3, heavy-atom, and charge descriptors; QED; property-window and Lipinski/Veber counts; PAINS/Brenk/NIH alert counts/descriptions; Bemis–Murcko `series_id`; achiral/chiral nearest neighbors; `local_novelty_achiral`; and apparent ligand efficiency/LLE. |
| `research/analysis/candidate_priorities.csv` | Chemistry-gate fields, applicability-aware `herg_evidence_status`, observed/predicted evidence used, `discovery_score_without_safety`, nullable `complete_evidence_score`, Pareto/deterministic ranks, sensitivity-rank min/median/max/span, and `experimental_followup_tier`. Missing or out-of-domain hERG evidence has no safety desirability or complete score. |
| `research/analysis/priority_decision_trace.csv` | Long-form objective name/desirability column, desirability, configured weight, complete-score inclusion flag, normalized weighted contribution, and missing policy for each structure. The missing policy is `unknown_no_credit`. |
| `research/analysis/priority_data_gaps.csv` | Reason-coded gaps such as unknown safety, no PK coverage, limited Menin replication/source support, property review, and PAINS review. |
| `research/analysis/priority_sensitivity.csv` | Base, leave-one-objective-out, and emphasized-objective scenario eligibility, active objectives, scores, and ranks. Cross-scenario rank-stability fields are merged into `candidate_priorities.csv`. |
| `research/analysis/priority_frontier.csv` | Chemistry-gated structures on complete-evidence Pareto rank 1. Unknown safety cannot enter this complete-evidence frontier. |
| `research/analysis/chemical_series_members.csv` | Structure-to-Bemis–Murcko series mapping; acyclic structures use exact canonical identity. |
| `research/analysis/chemical_series_summary.csv` | Series size, potency span, median QED/molecular weight/logP, source count, representative structure, and whether the configured minimum series size is met. |
| `research/analysis/similarity_cluster_members.csv` | Deterministic Butina cluster membership and representative under achiral Morgan fingerprints. |
| `research/analysis/similarity_cluster_summary.csv` | Cluster sizes, representatives, and activity summaries. |
| `research/analysis/activity_cliffs.csv` | Fingerprint pairs meeting configured achiral similarity and absolute ΔpActivity thresholds, with chiral similarity, SALI where defined, connectivity/series flags, shared assay/document context, and evidence grade. |
| `research/analysis/matched_molecular_pairs.csv` | Conservative RDKit MMPA exactly-one-cut pairs satisfying core/variable constraints, with deterministic core/transform/variable fragments, potency delta, similarity, context grade, and cliff flag. |
| `research/analysis/matched_molecular_pair_cliffs.csv` | The matched-pair subset that also meets the configured activity-cliff rule. |
| `research/analysis/connectivity_variant_members.csv` | Members grouped by first InChIKey connectivity block for stereochemistry/protonation/source-identity review. |
| `research/analysis/connectivity_variant_summary.csv` | Connectivity-group size, identity, potency, and context summary. |
| `research/analysis/connectivity_variant_cliffs.csv` | Same-connectivity pairs meeting the configured potency-difference review threshold. |
| `research/analysis/approved_reference_coverage.csv` | Dated, configured coverage benchmark for approved reference structures: name/CID/status/`source_checked_at`/source URLs, standardized-reference identity, exact-primary and any-public-MENIN coverage, endpoint/assay inventory, nearest primary achiral/chiral similarity, scaffold coverage, descriptors/QED, and alert counts. It does not compare efficacy or establish equivalence. |
| `research/analysis/prospective_selection_plan.csv` | Configurable, scaffold-series-capped public-data experiment template with `selection_order`, category/rationale, optional paired cliff ID, structure/potency/series, QED/property, hERG status, PK coverage, local novelty, and follow-up tier. Categories cover potent safety gaps, liability characterization, novel-scaffold exploration, activity-cliff confirmation, negative controls, and PK bridges. It is not a lead list or prospective result. |
| `research/analysis/prospective_selection_summary.csv` | Requested quota, selected structures, and explicit shortfall by selection category. |
| `research/analysis/analysis_summary.json` | Task, schema/status, output counts, tier/evidence-status counts, algorithm contract, RDKit version, fingerprint/scaffold/cluster/cliff policies, unknown-safety policy, and SHA-256 of every direct input. |
| `research/analysis/analysis_build_metadata.json` | Build ID plus processed, software, models, analysis, and resolved-settings SHA-256 values used by lineage validation. It is excluded from the analysis content digest to avoid self-reference. |

`apparent_ligand_efficiency = 1.364 × pActivity / heavy_atom_count` and `apparent_lle = pActivity - logP`. Both use an IC50-derived pActivity and are not thermodynamic binding-efficiency measurements. Alert columns are review signals, not automatic exclusions. The configured approved-reference panel is a time-stamped coverage control whose PubChem structures and FDA status links must be rechecked for a new release; it is not an efficacy, safety, or ranking benchmark.

## Data manifests

`research/reports/manifests/` contains six release-level manifests when chemical intelligence is enabled (five when it is disabled):

| Manifest | Root and upstream contract |
| --- | --- |
| `raw_manifest.json` | Source snapshot. |
| `processed_manifest.json` | Curated snapshot linked to the raw digest. |
| `software_manifest.json` | Package source, scripts, pipeline configuration, project metadata, requirements/lock, and environment specification. |
| `models_manifest.json` | Complete model root linked to processed and software digests. |
| `analysis_manifest.json` | Release analysis artifacts linked to processed, models, and software digests, including validation of the direct input SHA-256 values in `analysis_summary.json`. `analysis_build_metadata.json` and `smoke/` are excluded. Present only when analysis is enabled. |
| `reports_manifest.json` | Report bundle linked to processed, models, software, and analysis digests when enabled; manifest/verification/run-metadata files are excluded to avoid recursive identity. |

Each contains, as applicable:

- `manifest_version`, `stage`, `created_at`, `build_id`, and `root`;
- `file_count`, `total_size_bytes`, and content-derived `dataset_sha256`;
- `upstream` stage/digest links;
- sorted file entries with relative path, SHA-256, size, media type, and tabular row/schema metadata where readable; and
- `manifest_sha256`, which protects the manifest document itself.

Absolute local paths and file modification times are intentionally excluded from content identity.

Verification writes `research/reports/verification/<stage>_verification.json` and `<stage>_verification_issues.csv`, with validity, files checked/expected, and any missing/hash/size/schema/linkage issue. `stage` is `raw`, `processed`, `software`, `models`, `analysis`, or `reports` when analysis is enabled. Release verification also checks shared build IDs, upstream digests, model lineage, analysis direct-input/manifest lineage, and report-build lineage.

## Run metadata

`research/reports/run_metadata.json` records the latest invocation, while `research/reports/run_metadata/<stage>.json` preserves the latest invocation for each requested stage. Each record includes start/finish time, completion status (and a typed failure message when applicable), command, requested stage, Git revision when available, Python/platform, fully resolved settings, and a stable YAML settings snapshot. These files help reconstruct command execution but are not a substitute for the six enabled-stage release manifests or per-model manifests.

`research/data/raw/collection_metadata.json` records collection time, ChEMBL status payload, collected row/file/assay counts, request limits, target/query policy, and the all-source staging/promotion policy for a network refresh. Source-specific search, catalog, status, and target-search files provide the detailed acquisition evidence.

## Generated publication tables

The report stage writes `research/reports/tables/`:

| File | Content |
| --- | --- |
| `dataset_inventory.csv` | Rows, unique structures, and eligible rows across core datasets. |
| `menin_endpoint_assay_summary.csv` | Measurement/structure/exact/eligible counts and median pActivity by endpoint and assay family. |
| `menin_source_endpoint_summary.csv` | Measurements and structures by source and endpoint. |
| `menin_replicate_consistency.csv` | Repeated exact measurements, source count, min/median/max/SD, and log-range by structure/endpoint. |
| `herg_label_summary.csv` | Blocker, non-blocker, and ambiguous task-row counts. |
| `herg_primary_label_summary.csv` | Label population for the primary `IC50 × electrophysiology_functional` hERG task. |
| `herg_pooled_label_summary.csv` | Broader pooled hERG label population, explicitly sensitivity-only. |
| `pk_admet_coverage.csv` | Observation/compound/unit coverage by endpoint, species, matrix, route, and available experimental context. |
| `top_menin_tasks.csv` | Review table of high-potency Menin task rows with identity, context, provenance, and heterogeneity. |
| `critical_field_missingness.csv` | Field-level completeness for core source and analytical tables. |
| `assay_context_completeness.csv` | Availability of target, endpoint, assay, document/date, and protocol-context fields. |
| `curation_attrition_by_source.csv` | Source rows, eligible rows, excluded rows, and structures by source. |
| `quarantine_reasons_by_source.csv` | Reason-coded Menin, hERG, and PK/ADMET quarantine counts by source. |
| `menin_cross_source_mirror_links.csv` | Row-level membership and preferred/redundant status for potential cross-source mirror groups. |
| `menin_model_domain_performance.csv` | Menin holdout error inside versus outside the applicability domain. |
| `menin_model_source_performance.csv` | Menin holdout error by contributing source label. |
| `menin_model_temporal_performance.csv` | Menin holdout error by five-year/undated band. |
| `menin_model_scaffold_performance.csv` | Menin holdout error by Bemis–Murcko/exact acyclic grouping. |
| `menin_model_failure_cases.csv` | Highest-error Menin holdout cases with prediction and domain context. |
| `menin_split_chemical_coverage.csv` | Train/test structure/scaffold coverage and requested/actual split metadata. |
| `herg_model_domain_performance.csv` | Primary hERG accuracy/Brier/probability summaries inside versus outside domain. |
| `chemical_medicinal_profiles.csv`, `chemical_candidate_priorities.csv`, `chemical_priority_frontier.csv` | Report copies of the primary medicinal-chemistry, transparent prioritization, and complete-evidence frontier tables. |
| `chemical_priority_data_gaps.csv`, `chemical_priority_sensitivity.csv` | Report copies of reason-coded evidence gaps and scoring/rank sensitivity scenarios. |
| `chemical_series_summary.csv`, `chemical_similarity_cluster_summary.csv` | Report copies of series and deterministic Butina cluster summaries. |
| `chemical_activity_cliffs.csv`, `chemical_mmp_cliffs.csv` | Report copies of fingerprint and conservative single-cut matched-pair cliffs. |
| `chemical_connectivity_variant_summary.csv`, `chemical_connectivity_variant_cliffs.csv` | Report copies of connectivity-variant audit summaries and flagged potency pairs. |
| `chemical_approved_reference_coverage.csv` | Report copy of the dated approved-reference coverage benchmark; not an efficacy comparison. |
| `chemical_prospective_selection_plan.csv`, `chemical_prospective_selection_summary.csv` | Report copies of the pre-experiment public-data selection template and its quota shortfalls; not prospective validation. |
| `publication_readiness_matrix.csv` | Semantic checks for trained primary tasks, RDKit features, requested holdouts without fallback, uncertainty/domain evidence, linked/verified manifests, quality, environment, and clean revision, plus explicit false blockers for external/prospective validation and approvals. It is not scientific or governance sign-off. |

When analysis is enabled, the report also generates `menin_medchem_property_landscape.png`, `menin_chemical_series_sizes.png`, `menin_activity_cliff_landscape.png`, `menin_followup_tier_counts.png`, and `menin_complete_evidence_frontier.png` under `research/reports/figures/` when their required data are available.
## Wild-type hERG first-paper layers

The paper-facing layers are additive to `herg_hierarchy/v1` and preserve its
observation-native values.

- `herg_paper/wildtype_scope_v1/wildtype_observation_index.parquet` contains
  admitted observation identifiers, original target-variant status,
  `wildtype_scope`, admission reason, pIC50 when present, and a source-derived
  binary label when present. `wild_type_or_unspecified` is never equivalent to
  confirmed wild type.
- `herg_paper/wildtype_scope_v1/explicit_mutant_exclusions.parquet` is the
  complete 258-row mutant/variant audit surface. It is not a modeling table.
- `herg_hierarchy/v1_2_quality_tasks/` contains Q0 weak fixed-dose binary, Q1
  quantitative pIC50, Q2 functional assay-aware, C0 development-context, and
  C1 QT/QTc-context artifacts. `eligible`, `eligibility_reason`,
  `exclusion_reason`, `clinical_context_only`, `direct_herg_label`, and
  `use_as_training_label` must be honored together.
- `herg_hierarchy/v1_2_modality_qt/herg_measurement_modality_index.parquet`
  separates measurement technology, automation evidence, and dose design.
  `unresolved` is a valid reported state rather than an imputed method.
- `herg_hierarchy/v1_2_modality_qt/qt_clinical_phenotype_index.parquet`
  preserves clinical QT/QTc values, denominators, correction semantics, trial
  identifiers, and source pointers. `herg_potency_derived` and
  `qt_used_as_herg_label` are required to remain false.
- `herg_paper/cpu_baseline_v1/` contains a molecule-only diagnostic model,
  validation-selected threshold, locked scaffold-test predictions, and a
  manifest that explicitly prohibits interpreting the run as prospective,
  clinical, or superiority evidence.
- `herg_paper/quality_baselines_v1/` contains structure-median Q1 pIC50
  targets, a fixed Morgan ridge baseline, Q2 IC50/pIC50 censoring-aware
  structure targets, exact-only and penalized Gaussian/Tobit descriptor ridge
  baselines, locked validation/test predictions, strict-JSON metrics, serialized
  lightweight models, and a content-bound manifest. Q1 and Q2 remain separate
  endpoints; the small Q2 result is a pipeline diagnostic rather than a model
  claim.
- `herg_hierarchy/v1_3_master/observation_master.parquet` is the paper-facing
  one-row-per-observation table. It joins, without overwriting native values,
  WT scope, task membership, source/assay identity, endpoint and relation
  semantics, modality, automation, dose design, clinical-context flags, and
  provenance.
- `herg_hierarchy/v1_3_master/structure_master.parquet` and
  `structure_evidence_summary.parquet` provide one-row-per-structure molecular
  descriptors and evidence coverage. `task_membership.parquet` remains the
  authoritative observation-to-task mapping; task grains must not be inferred
  from structure-level counts.
- `herg_hierarchy/v1_3_master/assay_protocol_index.parquet` is an assay-level
  evidence index for reported cell system, temperature, voltage, time/incubation,
  recording configuration, and named automated platform. Raw supporting text,
  separately labeled curated source-contract evidence, normalization status,
  and confidence are retained. Blank or `unresolved` fields mean the source did
  not support a normalized value; they are not imputations.
- `herg_hierarchy/v1_3_current_analysis/` contains descriptive inventories,
  source/modality/automation disagreement tables, exact-pIC50 replicate
  dispersion, categorical-confounding measures, Q0 fundamental descriptors,
  prespecified descriptor bins/interactions, split-shift profiles, and
  clinical/QT coverage. These outputs are hypothesis-generating and explicitly
  non-causal.
- `herg_hierarchy/v1_4_pre_hpc_assets/` contains three human-review queues:
  quantitative functional evaluation candidates, standardized-potency
  replicate conflicts, and assay-protocol enrichment priorities. Candidate
  rows are not an adjudicated or sealed gold standard. The broader conflict
  queue uses every exact standardized IC50/pIC50 interpretation in the master;
  it is intentionally larger than the source-native exact-pIC50 replicate table
  in `v1_3_current_analysis`.
- `herg_hierarchy/v1_4_benchmark_freeze/` contains label-blind challenge
  membership and a machine-readable registry. It never embeds target values or
  classes. Blocked temporal, low-similarity external, and prospective
  manual-patch challenges remain explicit rather than being approximated.
- `herg_hierarchy/v1_5_candidate_adjudication/` contains automated source,
  target, document, relation/unit, duplicate/mirror, and measurement-lineage
  evidence for all 226 evaluation candidates and 1,340 conflict structures.
  Its 1,566-row human packet has blank decision fields; zero rows are gold or WT
  promoted.
- `herg_hierarchy/v1_5_benchmark_freeze/` contains seven supplemental Q2
  challenge definitions (assay, document, two temporal, automated-patch, and
  two Morgan-similarity surfaces) and six reason-coded blockers. Memberships
  contain routing/context only—no outcome columns—and preserve structure,
  scaffold, and applicable context-group exclusivity.
- `herg_hierarchy/v1_5_wt_reference/` freezes the reviewed human canonical
  Q12809/KCNH2 sequence (1,159 amino acids) as JSON and FASTA. This is a
  sequence identity contract, not an experimental construct or receptor.
- `herg_hierarchy/v1_5_mmp_analysis/` contains 48,988 label-free Q1 single-cut
  pair definitions and 43,824 exploratory training-only effect rows. Pair
  definitions are structure/scaffold split-contained; validation/test values
  are not returned or retained, although physical Parquet page nonaccess is not
  asserted. The analysis fields are not a production feature store.
- `herg_hierarchy/v1_5_qt_exposure_prep/` contains 95-structure/143-trial
  collection templates, 143 gap priorities, and 1,072 source-review candidates.
  All adjudicated dose/exposure/hERG/margin fields are null; QT remains phenotype
  context and every hERG, clinical-risk, margin, and training-label flag is zero.
- `herg_hierarchy/v1_5_hpc_preflight/` binds the accepted pre-HPC releases and
  records the present packages, storage, seven future stages, 25 unresolved
  production contract fields, and readiness gates. Only 2/7 blocking gates
  pass; no smoke test, model, production feature job, or HPC job ran.
