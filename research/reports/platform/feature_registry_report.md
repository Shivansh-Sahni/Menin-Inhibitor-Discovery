# Feature registry report

Status: implemented and materialized on 2026-08-04; dataset-level coverage analysis remains tied to the canonical public task build.

The feature registry contains 19 versioned admission rules. Its canonical digest is
`ee9ec1c59bb6790cac45595f179bfbfd9eccca725c72f4dbadbdce71d13fc674`; the materialized CSV digest is
`f907bd3d01cfcc370c671ac7d54707bf6ab8f0ec9b3b0292109d004cb953523b`.

## Default-admitted input families

- standardized/canonical SMILES;
- deterministic molecular graphs;
- versioned Morgan fingerprints;
- compact RDKit descriptors; and
- traceable protein sequence.

These are pre-outcome representations. Deterministic structure and sequence preparation may run globally, but any learned transform, imputer, scaler, vocabulary, category encoding, selector, or calibration step must be fit on the frozen training partition only.

## Fail-closed exclusions

- Outcomes and transformations of the same outcome are targets, never features.
- IDs and source/year fields are grouping or audit metadata, not default predictors.
- Free text is disabled by default even when a label-like scan is clean. It requires a separate admission decision and manual leakage policy.
- Cross-endpoint measurements require proof that the measurement existed before the prediction time and was not derived from the same target assay.
- Upstream computational predictions require out-of-fold or locked-prospective lineage, including upstream model, dataset, split, and prediction-time identity.
- Unregistered fields fail closed.

## Materialized artifacts

- `research/data/platform/features/feature_registry.csv`
- `research/data/platform/features/feature_registry_metadata.json`

Dataset-specific descriptor failure rates, input-length distributions, and truncation impact are not claimed here. They must be computed from the completed public canonical task artifacts and reported separately; absence of those artifacts is not represented as zero failure or zero truncation.
