# Changelog

All notable changes to this project are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow [Semantic Versioning](https://semver.org/) where practical for code and data-contract changes.

## [Unreleased]

### Added

- Publication-oriented report tables and figure expansion.
- Final release evidence and independent/prospective validation as they are completed.

## [0.3.0] - 2026-07-14

### Added

- A first-class, transactional `analyze` stage with its own content-addressed manifest and complete processed/software/model lineage.
- Primary-task medicinal-chemistry profiles, structural-alert review flags, scaffold series, deterministic Butina clusters, chiral/achiral novelty, activity cliffs, conservative single-cut matched molecular pairs, connectivity-variant audits, Pareto fronts, evidence gaps, and deterministic rank-sensitivity scenarios.
- Applicability-aware experimental follow-up tiers that treat missing or out-of-domain hERG evidence as unknown and never as favorable safety evidence.

### Changed

- The enabled release graph now verifies six stages: raw, processed, software, models, analysis, and reports.
- Descriptive potency and chemical-space figures are scoped to the configured primary Menin endpoint and assay family.
- Publication reports now include chemical-series, SAR, medicinal-chemistry, and experimental-prioritization diagnostics with explicit claim boundaries.

## [0.2.0] - 2026-07-14

### Added

- Versioned YAML configuration for curation, hERG labeling, modeling, paths, and reporting.
- Resilient HTTP sessions, retries, atomic writes, ChEMBL status capture, PubChem target-gene search, and stale-file-safe PubChem loading.
- RDKit structure cleanup, fragment-parent selection, neutralization, canonical identities, InChIKeys, and stable parent/full structure IDs.
- Strict unit handling, relation bounds, endpoint/assay families, target relevance, measurement IDs, eligibility flags, and quarantine outputs.
- Explicit PK/ADMET endpoint/context classification.
- Configurable data-quality auditing with JSON and detailed/summary CSV evidence.
- Portable SHA-256 raw/processed manifests with schema/row metadata, linked build IDs, and verification.
- Offline CSV/TSV/SDF proprietary-data intake with declarative assay/endpoint registries, runtime-key HMAC pseudonyms, conflict quarantine, deterministic outputs, and tests that prevent plaintext source identifiers from escaping.
- RDKit Morgan fingerprints and physicochemical descriptors with a disclosed hashed-SMILES fallback.
- Compound-grouped scaffold, chemical-cluster, temporal, and random holdouts and compatible cross-validation.
- Dummy, linear, and Extra Trees candidate models selected by training cross-validation.
- hERG probability calibration, expanded regression/classification metrics, bootstrap confidence intervals, regression conformal intervals, and Tanimoto applicability-domain diagnostics.
- Safer `skops` model serialization with explicit `joblib` trust warning fallback and model manifests.
- Development/test dependencies, pre-commit configuration, CI matrix, and offline pipeline smoke test.
- Architecture, methodology, data dictionary, reproducibility, limitations, source/citation, publication, licensing, contribution, and proprietary-data intake documentation.

### Changed

- Modeling tables are keyed by standardized parent structure and stratified by endpoint and assay family rather than raw SMILES-only pooling.
- Missing/unknown units and unresolved/off-target PubChem assays are quarantined instead of assigned default values or endpoints.
- Repeated structures are kept together for every split strategy, including random and temporal evaluation.
- Menin and hERG models now use chemical features and defensible validation evidence rather than only random hashed-SMILES baselines.
- The project is described as publication-oriented research software with explicit claim and confidentiality boundaries.

### Security

- Documented model-artifact trust boundaries and prohibition on loading untrusted pickle/joblib files.
- Added a controlled proprietary-data intake and disclosure-review protocol.
- Clarified that internal data and derivative models/reports must remain outside public Git and unapproved services.

## [0.1.0] - 2026-06-09

### Added

- Initial ChEMBL, BindingDB, and PubChem collection scripts.
- Measurement and compound-level Menin/hERG tables and a broad PK/ADMET extract.
- Portable hashed-SMILES Ridge/logistic baseline models.
- Initial summary report, figures, tests, and project documentation.
