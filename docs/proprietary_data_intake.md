# Proprietary data intake protocol

This protocol is for future Wang lab or collaborator data. It is intentionally stricter than the public-data workflow. The repository includes an offline validation and pseudonymization library, `menin_discovery.internal_data`, but does not grant permission, provide a secure upload channel, configure protected storage, or wire private data into the public CLI/model pipeline.

## Non-negotiable rule

Do not copy proprietary files into this Git working tree, commit them, attach them to an issue, paste them into a prompt, or upload them to an unapproved cloud service. “De-identified” compound structures can still be highly identifying intellectual property.

Use an institution-approved, access-controlled storage location outside the repository. Treat intermediate tables, caches, reports, nearest-neighbor outputs, model artifacts, and manifests as proprietary until a release review says otherwise.

On POSIX systems, the implemented intake writer tightens the private output directory to mode `0700` and accepted, quarantine, and summary files to `0600`. This is defense in depth, not a replacement for approved encrypted storage, access control, and audit logging; equivalent controls must be configured explicitly on non-POSIX platforms.

## 1. Authorization before transfer

The data owner or steward should document:

- owner and originating laboratory;
- governing NDA, data-use agreement, collaboration agreement, sponsor terms, patent strategy, and publication embargo;
- approved people, systems, geographic/storage restrictions, and retention period;
- whether structures, compound IDs, assay protocols, raw curves, and derived models can each be used and disclosed;
- whether any human-derived, animal, clinical, export-controlled, or regulated data are present; and
- who can approve a combined public/private model and any external release.

If these answers are absent, pause intake. This protocol is operational guidance, not legal, regulatory, or institutional approval.

## 2. Secure transfer and storage

- Use the lab/institution's approved encrypted transfer mechanism.
- Store the encrypted-at-rest source outside the Git root with least-privilege access and audit logging.
- Disable consumer sync, public links, automatic notebook publishing, and unapproved backup destinations.
- Keep an immutable received copy; perform transformations on a versioned working copy.
- Scan archives before extraction and reject path traversal, executables, macros, or unexpected file types.
- Record receipt time, sender, transfer channel, file count, byte sizes, and SHA-256 in a **private** inventory.
- Do not publish a manifest whose filenames, row counts, hashes, or schema reveal confidential program details unless approved.

Before intake, add and test a repository-level ignore rule for the chosen private path or, preferably, keep the path entirely outside the repository. A documentation statement is not a substitute for a tested technical control.

## Implemented offline intake boundary

The library supports CSV, TSV, and SDF input and requires a declarative `InternalDataConfig` with:

- source-column/property mappings;
- an assay registry defining endpoint, target, assay family, units, and allowed units;
- an endpoint registry defining canonical names, families, and allowed units;
- explicit defaults and required fields; and
- RDKit and structure-parent policy controls.

It validates structures, positive concentration values, units, relations, assay/endpoint registration, target metadata, assay family, cohort role, duplicate rows, one source row ID mapping to conflicting records, and one source compound ID mapping to conflicting structures. Accepted and quarantined rows remain separate.

The generic library defaults require structure, value, unit, relation, endpoint, batch, and assay. The checked-in template additionally requires `cohort_role`. A Wang lab production profile should also make source `compound_id`, source `row_id`, and the appropriate assay/date/replicate fields required. The current canonical input vocabulary does not yet carry lot/form, protocol version, curve QC, campaign, or release classification; add and test those fields before relying on them.

[`pipeline/config/internal_data_template.yaml`](../pipeline/config/internal_data_template.yaml) is a placeholder contract with no lab data. Copy it to approved private storage before adding real column names or assay codes; do not turn the checked-in template into the live project registry.

Source row, compound, batch, and assay identifiers are replaced with deterministic, namespace-separated HMAC pseudonyms. The secret must contain at least 16 bytes, is supplied only at runtime, and is rejected if placed in configuration. The secret and plaintext source identifiers are not written to returned metadata or validation messages. **This is pseudonymization, not anonymization:** submitted/standardized structures, measurements, dates, replicates, and analytical context remain sensitive.

When given a private output directory, the library atomically writes:

- `internal_measurements.csv`;
- `internal_quarantine.csv`;
- `internal_validation_issues.csv`; and
- `internal_validation_summary.json`.

The summary includes input/configuration hashes, intake and standardization versions, accepted/quarantined counts, and issue counts. Outputs are deterministic and compatible with the provenance manifest API, but private manifests must stay private unless approved.

### Cohort-role contract

Every production row must be assigned exactly one approved role:

| Role | Permitted use | Prohibited influence |
| --- | --- | --- |
| `development` | Training, training-fold model selection, and calibration only when the data owner and analysis plan approve it. | It must not be relabeled as external/prospective evidence after inspection. |
| `locked_external` | One-time evaluation of a locked pipeline with compatible endpoints/protocols. | Training, calibration, feature or threshold selection, curation-policy tuning, and iterative stopping decisions. |
| `prospective_blind` | Time-forward/blinded evaluation after acquisition and the pre-specified unlock event. | Label access or any use in training, calibration, selection, thresholding, policy tuning, or stopping before lock/unblinding. |

The intake library rejects unknown nonblank role values and records accepted counts by role. It does not wire private rows into the public modeling CLI or provide a secure label vault. Private orchestration must filter roles before any model process can read a table, maintain a separate sealed location for prospective labels, and record the lock/unblind event. A missing role is a production-profile error even if a custom configuration omits it from the generic required-field list.

Use the Python API inside the approved environment. Prompt for the runtime secret instead of putting it in source, YAML, command history, or process arguments:

```python
from getpass import getpass
from menin_discovery.internal_data import ingest_internal_data

result = ingest_internal_data(
    "/approved/private/incoming/menin_assays.sdf",
    config="/approved/private/config/menin_intake.yaml",
    pseudonymization_key=getpass("Private intake key: "),
    output_directory="/approved/private/processed/build-001",
)
```

Do not use the example paths until the data steward has approved a real storage root. Do not print `result.accepted`, because it contains structures and measurements.

## 3. Minimum measurement contract

One row should represent one observed result or one explicitly defined replicate aggregate. Required fields:

| Field | Requirement |
| --- | --- |
| `compound_id` | Stable source registration ID; configure it as required and emit only its HMAC pseudonym (`internal_source_compound_id`). Do not overload a name or plate position. |
| `row_id` | Stable source measurement/record ID; configure it as required when the source has one. |
| `structure` | SMILES or an SDF record, with the submitted representation retained. |
| `target_id`, `target_name` | Exact target/construct identity. |
| `endpoint` | IC50, Ki, Kd, EC50, or a separately defined endpoint. |
| `relation` | `=`, `~`, `<`, `<=`, `>`, or `>=`. |
| `value_raw` | Observed or reported value without pre-conversion. |
| `standard_units` | Explicit unit for every numeric result. |
| `assay_id` | Stable internal assay/protocol identifier. |
| `measurement_date` | Actual assay/test date or controlled temporal-split field. |
| `cohort_role` | Exactly `development`, `locked_external`, or `prospective_blind`; required in the production intake profile. |
| private build metadata | Record data scope, project, dataset version, release class, and approval record outside the row file until these are added to the canonical input vocabulary. |

Strongly recommended fields:

- lot/batch and salt/form identifiers;
- full SDF structure, stereochemistry, and registration parent relationship;
- replicate ID and whether the row is raw replicate, curve fit, or approved aggregate;
- protocol version, construct/mutation, binding partner/substrate, detection technology, incubation time, temperature, pH, and operator/site;
- cell line, species, tissue/matrix, dose, route, formulation, and time point when applicable;
- curve-fit model, top/bottom, Hill slope, fit uncertainty, plate controls, QC decision, and reason for rejection;
- sample concentration range and limit of quantitation/detection;
- source instrument/file ID and notebook/ELN reference;
- synthesis/receipt/test dates and campaign/batch identifiers; and
- publication/release status.

Preserve raw replicate and curve-level data when available. Do not reduce the only retained evidence to a single potency spreadsheet.

## 4. Private staging and validation

Perform these steps inside the approved environment:

1. Validate file format and schema without changing the immutable received copy.
2. Generate a private content manifest and assign a dataset version/build ID.
3. Detect duplicate file deliveries and duplicate measurement identities.
4. Validate structures with the same versioned RDKit policy while retaining submitted structures.
5. Reconcile registration parent, salt/form, stereochemistry, and lot identity with the lab registry.
6. Validate endpoint vocabulary, units, relations, bounds, dates, target/construct, and required assay metadata.
7. Retain failed rows in a private quarantine with actionable reasons.
8. Produce an exception report for data-owner review; never “repair” values or units silently.
9. Freeze the approved curated snapshot and its private manifest before modeling.

If a unit or endpoint must be inferred from a protocol, record the inference as a steward-approved mapping with the protocol version and reviewer. Do not embed one-off assumptions in notebooks.

## 5. Mapping to the project contract

The private adapter should emit the same normalized concepts described in the [data dictionary](data_dictionary.md) plus:

- `data_scope` and private dataset/build ID (a required extension to the current v1 row contract, or controlled build metadata in the interim);
- controlled `cohort_role`, retained on every accepted row and in private role-count metadata;
- pseudonymous internal compound, lot, assay, replicate, batch, and campaign IDs;
- actual assay/test date separate from document year;
- protocol and QC version;
- registration-parent relationship; and
- release classification.

The adapter must preserve a one-way trace from a normalized `measurement_id` to the controlled source record. Public reports should not expose the reverse mapping.

Recommended build modes:

The public `menin-pipeline` CLI, including `--stage analyze`, is not an approved intake route for confidential data. Any private analogue must run in separately reviewed orchestration and storage. Give each governed role an isolated build root and manifest: a `development` build may fit and refine the pre-specified workflow; a `locked_external` build may evaluate the frozen workflow only; and a `prospective_blind` build must keep labels sealed until the model, thresholds, tiers, experiment template, and analysis plan are locked. Never reuse a development output root for either evaluation cohort.

| Mode | Purpose | Release default |
| --- | --- | --- |
| Public-only | Fully reproducible public baseline. | Candidate for release after source review. |
| Internal-only | Estimate lab-specific signal and protocol effects. | Private. |
| Combined | Evaluate whether internal evidence improves prediction. | Private unless explicitly approved. |
| Public-trained/private-test | Honest internal external validation when endpoint/protocol mapping is compatible. | Report only approved aggregate metrics. |
| Time-forward internal | Prospective-like lab evaluation by assay date/campaign. | Private; preferred decision evidence. |

Do not retain only the combined table. Separate modes are necessary to quantify domain shift, source effects, and the incremental value of proprietary data.

## 6. Modeling safeguards

- Define the prediction endpoint and assay protocol before inspecting final test outcomes.
- Group all forms/replicates of a registered parent consistently unless the scientific question explicitly concerns form/lot differences.
- Keep structures from one registered compound out of both training and test, including public duplicates.
- Search public/private cross-dataset overlap using standardized structures and source/document provenance.
- Reserve a true internal holdout by scaffold and, when feasible, by later campaign/date.
- Physically and logically exclude `locked_external` and `prospective_blind` rows before constructing any development matrix; do not rely on an analyst remembering to drop them later.
- Freeze code, structure policy, feature generation, model candidates, calibration, thresholds, split, and analysis plan before opening locked-external or prospective-blind labels.
- Freeze the chemical-intelligence configuration, scoring/tier policy, reference panel, proposed experiment selection, and outcome-analysis plan at the same lock point; a public-data `prospective_selection_plan.csv` is only a design template until new blinded outcomes are collected.
- Keep model selection and calibration inside the training partition.
- Report public-only, internal-only, and combined performance with confidence intervals.
- Analyze residuals by source, protocol version, batch, assay family, and applicability domain.
- Require prospective confirmation for compounds selected with the model.

If private data are too small for an independent test set, state that limitation rather than converting every row into training data.

## 7. Disclosure review

Before sharing code, a report, manuscript supplement, model, or presentation, inspect all of the following:

- raw and processed files, temporary files, notebook outputs, caches, logs, stack traces, and shell history;
- structures, InChIKeys, identifiers, source filenames, assay dates, batch counts, and nearest-neighbor examples;
- figures with labels, hover text, embedded metadata, or small groups;
- train/test assignments and prediction tables;
- model artifacts and embeddings/fingerprints;
- manifests, hashes, schema names, row counts, and build metadata; and
- git history, branches, releases, issues, CI artifacts, and remote object storage.

Use a two-person review by the technical owner and data steward. Release only explicitly approved aggregate results. Small cell counts, exact extrema, or structure depictions may still identify a confidential series.

## 8. Incident response

If proprietary data enter Git or an unapproved service:

1. Stop further syncing and notify the data steward/security contact immediately.
2. Restrict or disable repository/service access.
3. Preserve incident facts without further redistributing the data.
4. Rotate exposed credentials or links.
5. Follow institutional incident-response and legal guidance for history rewriting, deletion, notification, and documentation.
6. Assume deleting the current working-tree file is insufficient; copies may remain in Git history, forks, caches, CI artifacts, backups, or service logs.

Do not improvise destructive cleanup before the steward/security team determines the required preservation and remediation steps.

## Intake approval record

For each private dataset, retain a private sign-off with:

- dataset name/version and manifest digest;
- data owner and steward;
- approved purpose and systems;
- allowed build modes and users;
- retention/review date;
- fields removed or pseudonymized;
- known limitations and unresolved exceptions;
- release classification for data, aggregates, code, and models; and
- final approver and date.
