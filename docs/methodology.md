# Methodology

## Scope and intended use

This workflow supports four related research questions:

1. What public measurements provide defensible evidence of Menin/MEN1 activity?
2. How well can structure-based baselines predict endpoint-specific Menin potency under chemical holdout?
3. Which Menin-associated structures merit follow-up for possible hERG/KCNH2 liability?
4. Which observed PK/ADMET measurements exist for Menin-associated molecules and are sufficiently contextualized for endpoint-specific analysis?

The workflow does not establish efficacy, safety, clinical relevance, or a synthesis decision. Menin potency, cellular response, hERG activity, and PK/ADMET are separate endpoints with different assay semantics.

## Target anchors

| Surface | ChEMBL target | UniProt | Preferred name |
| --- | --- | --- | --- |
| Menin | `CHEMBL1615381` | `O00255` | Menin |
| hERG | `CHEMBL240` | `Q12809` | hERG / KCNH2 |

These identifiers anchor target collection and target-relevance checks. Text search is retained for discovery and audit, especially in BindingDB and PubChem, but a textual hit alone is not treated as confirmed target evidence.

## Public data acquisition

### ChEMBL

The collector records the ChEMBL service status, target-search results, target activities, and all ChEMBL activities for Menin-associated ChEMBL molecule IDs when PK/ADMET collection is enabled. Paged CSV writes use a temporary partial file and replace the prior snapshot only after successful completion.

The curation layer retains `standard_flag`, `data_validity_comment`, `potential_duplicate`, assay identifiers/formats, assay variants, document provenance, and the reported pChEMBL value. These fields are required to apply the quality guidance described in the [ChEMBL data FAQ](https://chembl.gitbook.io/chembl-interface-documentation/frequently-asked-questions/chembl-data-questions).

### BindingDB

Target-index TSV exports for Menin and Menin/KMT2A are downloaded. Ki, IC50, Kd, and EC50 fields are converted into the common long-form schema while retaining source identifiers, target text, patent/article provenance, and the original qualified value.

BindingDB can include measurements sourced from other databases or patents. Cross-source duplication must therefore be examined at the measurement and document levels rather than inferred only from the `source` label.

### PubChem BioAssay

Collection combines PUG REST gene-target lookup for `MEN1` with ranked text searches such as MEN1, menin, and menin–MLL/KMT2A. The pipeline stores search provenance, assay summaries, assay descriptions, data files, and download status. Only successfully downloaded assay files in the current status table are loaded, preventing stale files from silently re-entering a build.

PubChem rows do not receive a default IC50 or default unit. The parser prefers explicit standard value/unit fields, then explicit endpoint columns and their unit metadata, then a PubChem standard-value field only when its unit is available. A reviewed assay registry can explicitly override endpoint, unit, assay family, or include/exclude status; unresolved assay metadata remain excluded.

PubChem assay `deposit_date` and `modify_date` are retained from the assay catalog. The deposit year is exposed as `document_year` with `date_provenance=pubchem_assay_deposit_date`; the modification date is audit metadata and is not used as the temporal label. ChEMBL and BindingDB years generally describe a publication, patent, or source document. A temporal split can therefore mix source-document and assay-deposit clocks. It is a disclosed public-data sensitivity analysis, not a reconstruction of assay chronology, and should be replaced by actual test/campaign dates for governed internal validation.

## Common measurement contract

Each source adapter maps observations into a long-form contract containing source and record identity, compound identity, submitted structure, target, endpoint, relation, raw value and unit, assay context, document provenance, and source-specific quality fields. Normalization adds:

- deterministic `measurement_id`;
- standardized chemical identities;
- normalized endpoint and unit;
- numeric value, nM conversion, and pActivity;
- censoring direction and numeric bounds;
- endpoint and assay families;
- target relevance;
- quality and exclusion reasons; and
- explicit modeling eligibility.

The raw value, original unit, original relation, and original structure remain available after normalization.

## Chemical structure policy

The RDKit policy identifier encodes cleanup/neutralization plus the configured fragment-parent and tautomer options (for example, `rdkit-cleanup-neutralize-v2-fragment-parent-1-tautomer-0`):

1. Parse and sanitize the submitted SMILES.
2. Apply RDKit cleanup.
3. Preserve a canonical cleaned full-structure representation.
4. Select the fragment parent to remove salts and disconnected minor fragments.
5. Neutralize the parent where possible.
6. Sanitize and generate canonical isomeric SMILES and a standard InChIKey.
7. Hash both the parent representation (`structure_id`) and cleaned full representation (`full_structure_id`) under explicit namespaces.

Tautomers are **not** canonicalized by default because tautomer collapsing can merge experimentally relevant states. The original SMILES and original source InChIKey are never overwritten. If RDKit is unavailable, records can be inspected with an explicitly prefixed unvalidated raw key, but publication modeling should require the RDKit backend.

Parent standardization is not chemical registration. Stereoisomers, isotopologues, covalent states, mixtures, and active salt forms require project-specific review.

## Endpoint, units, and censoring

The default core bioactivity endpoints are IC50, Ki, Kd, and EC50. They are normalized but not assumed interchangeable. Binding constants, biochemical functional activity, cellular functional activity, biophysical measurements, and in-vivo observations are assigned distinct families when context allows.

Supported concentration units are converted to nM with explicit factors. A missing or unknown unit produces no `value_nm` or `p_value` and an exclusion reason. There is no nM fallback.

For a positive concentration value:

```text
pActivity = 9 - log10(value_nM)
```

Relations retain their scientific meaning:

| Relation | Concentration semantics | pActivity semantics |
| --- | --- | --- |
| `=` | exact point | exact point |
| `~` | approximate point | approximate point |
| `<`, `<=` | upper concentration bound | lower pActivity bound |
| `>`, `>=` | lower concentration bound | upper pActivity bound |

The measurement table retains all parseable relations. The current default modeling tables use exact, eligible values. The aggregation API also supports a versioned policy that retains censored-only compounds as thresholds; those threshold aggregates must not be presented as exact labels or mixed into ordinary point-regression without a censoring-aware method.

## Quality and eligibility gates

A row is ineligible for ordinary model training when any applicable gate fails. Gates include:

- missing, invalid, or unvalidated-required structure;
- missing, non-numeric, non-positive, or unconvertible concentration;
- unsupported endpoint or relation;
- unresolved or mismatched target;
- duplicate measurement identity;
- a nonblank source record ID that maps to incompatible values or structures within the same source/endpoint;
- ChEMBL non-standard measurement, validity warning, or potential-duplicate flag;
- excluded assay variant; and
- disagreement between computed and reported pChEMBL beyond tolerance.

`quality_flags` can include review information that does not always exclude a row, while `exclusion_reason` is the authoritative training gate. Quarantine files contain excluded rows; they are evidence for review, not data to delete.

The separate quality audit checks required fields, data types, ranges, known units, relations, identifiers, target consistency, assay ambiguity, repeated measurements, and within-compound conflicts. It produces table-level, group-level, and row-level findings in JSON and CSV.

## Compound aggregation

Eligible measurements are grouped by standardized `structure_id` and, by default, endpoint and assay family. Each group records median/best/min pActivity, median/best nM value, exact/censored/approximate counts, sources, endpoints, assay families, documents, activity range, and heterogeneity flags.

The primary regression label is median pActivity within a compatible endpoint/assay-family group. The best value is retained for inspection but is not the default label. An activity range above the configured log-spread threshold (2.0 log units by default), multiple endpoint families, or multiple assay families marks a group as heterogeneous.

Before central-label aggregation, the pipeline links a deliberately conservative class of potential cross-source mirrors: exact normalized observations with the same standardized structure, endpoint, assay family, relation, and pActivity appearing in more than one public source. Every source row remains in the measurement table with a stable `MIR-…` group ID. Same-source replicates are never collapsed. Within a linked group the default label retains the highest-priority source (`ChEMBL`, then `BindingDB`, then `PubChem`) and marks lower-priority rows as redundant because ChEMBL usually carries the richest structured assay/document metadata. This is a heuristic, not proof that two database rows arose from the same experiment.

A preconfigured Menin no-collapse sensitivity retrains the primary `IC50 × biochemical_binding` task with linked mirror rows retained. A separate clean-label sensitivity excludes structures whose exact labels exceed the configured within-structure spread threshold. Central and sensitivity populations, split assignments, and metrics must be compared; a favorable sensitivity result must not be substituted for the pre-specified primary analysis.

This aggregation controls obvious mixing; it cannot remove protocol, construct, incubation-time, substrate, cell-line, or laboratory effects that are absent from source metadata.

## hERG labels

The hERG table uses target-anchored ChEMBL observations and the same structure/unit/quality controls. Its communication label is:

- blocker: median hERG activity `<= 10,000 nM`;
- non-blocker: median hERG activity `>= 30,000 nM`; and
- ambiguous: `10,000–30,000 nM`, retained in data but omitted from binary training.

The positive class therefore means activity at or below 10 µM under the curated public assay context; it is not a clinical QT-risk label. The thresholds and 0.30/0.70 predicted-risk communication bands are adjustable research policies, not optimized decision thresholds.

The primary hERG model is restricted to `IC50 × electrophysiology_functional` observations, which prioritizes functional channel-current evidence and avoids silently pooling binding, expression, and heterogeneous endpoint semantics. A broader endpoint/assay-family pooled model is generated only as an explicitly labeled sensitivity analysis. Neither task is a substitute for a dedicated cardiac-safety assay or prospective validation.

Before classification, endpoint/assay-family task rows are collapsed to one row per standardized structure. Structures whose compatible task rows produce both blocker and non-blocker labels are excluded and counted in task metadata rather than duplicated or assigned an arbitrary majority label. Ambiguous-only structures remain outside the binary task.

## PK/ADMET observations

The PK/ADMET step applies explicit endpoint-and-context rules to molecule-wide ChEMBL activity rows. Current categories include pharmacokinetics, metabolic stability, distribution, permeability, physicochemical properties, drug interaction, safety pharmacology, and toxicity. It extracts species, matrix, administration route, experimental context, directionality, and the rule responsible for inclusion.

PK/ADMET rows remain observations. Different species, matrices, routes, units, and protocols are not pooled into a single label. Endpoint-specific modeling requires a dedicated harmonization protocol and sufficient comparable support.

## Proprietary-data boundary

`menin_discovery.internal_data` provides an offline, deliberately separate intake path for approved CSV, TSV, or SDF lab data. A declarative configuration maps source fields and defines registered assays/endpoints, target context, assay families, defaults, and permitted units. The validator standardizes structures, preserves submitted values/units/relations, checks registry conflicts and duplicate identities, and separates accepted rows from a reason-coded quarantine.

Source row, compound, batch, and assay IDs are replaced with deterministic, namespace-separated HMAC pseudonyms using a key supplied only at runtime. The key cannot be stored in the intake configuration. The summary records input and configuration hashes, policy versions, counts, and issue categories; deterministic outputs can be privately manifested.

Every accepted row carries one controlled cohort role: `development`, `locked_external`, or `prospective_blind`. The intake validates the vocabulary and reports role counts, but the public model CLI does not consume private tables. Private orchestration must allow only approved `development` rows into training and calibration. `locked_external` rows are evaluation-only, while `prospective_blind` labels remain sealed until the model, preprocessing, thresholds, and analysis plan are locked. Neither role may influence feature engineering, hyperparameter selection, calibration, thresholds, curation-policy choice, or stopping decisions.

This boundary does not anonymize structures or results and is not part of the public `all` stage. Use it only after data-owner approval in access-controlled storage, following [the proprietary-data intake protocol](proprietary_data_intake.md). Public-only, internal-only, and combined analyses must remain distinguishable.

Future approved cohorts must be evaluated as separate private builds rather than appended to the public CLI inputs. A `development` build may fit models and develop a private analysis specification. A `locked_external` build may apply that frozen specification for evaluation only. A `prospective_blind` build must keep outcome labels sealed until code, preprocessing, models, hERG policy, tiers, and the analysis plan are locked. Each build needs its own private roots, manifests, build ID, access policy, and disclosure review.

## Features

The preferred feature representation is an RDKit Morgan bit fingerprint (radius 2, 2,048 bits by default) concatenated with a compact descriptor panel: molecular weights, logP, TPSA, H-bond donors/acceptors, rotatable bonds, ring counts, fraction sp3, heavy atoms, and formal charge. Descriptors are scaled using training-partition statistics.

An explicit hashed-SMILES plus string-descriptor fallback keeps validation and smoke tests runnable without RDKit. It is recorded in model metadata and is not chemically equivalent. Publication runs should request `rdkit` rather than `auto` so a missing dependency cannot silently change the representation.

## Chemical-intelligence analysis

The optional `analyze` stage operates only on the configured primary Menin task, by default `IC50 × biochemical_binding`, and resolves one row per `structure_id`. It reads the curated primary-task measurements, PK/ADMET observation inventory, and the model-generated Menin/hERG scoring table. Conflicting duplicate primary rows are an error rather than an implicit aggregation. Thresholds, property windows, alert catalogs, fingerprint settings, scoring weights, and sensitivity scenarios are all resolved from configuration. The resulting tables are descriptive decision support, not validated candidate selections.

### Medicinal-chemistry profiles and series

For a valid standardized structure, RDKit computes molecular weight, exact molecular weight, logP, TPSA, hydrogen-bond donors/acceptors, rotatable bonds, total/aromatic rings, fraction sp3, heavy atoms, formal charge, Lipinski/Veber counts, configured property-window violations, and property desirability. QED is reported as a composite drug-likeness descriptor. Two potency-normalized quantities are deliberately labeled **apparent** because they use the observed IC50-derived `pActivity`, not a binding free energy:

```text
apparent ligand efficiency = 1.364 × pActivity / heavy_atom_count
apparent LLE = pActivity - logP
```

RDKit FilterCatalog matches from the configured PAINS, Brenk, and NIH catalogs are emitted as counts and descriptions. They are review alerts, not proof of interference or toxicity, and no row is deleted because it matches. When `require_no_pains` is configured, a PAINS match fails the chemistry gate and routes an otherwise potent structure to the chemistry-review tier; the source observation remains present.

Bemis–Murcko frameworks define stable chemical-series identifiers when a scaffold is available; acyclic structures fall back to exact identity rather than receiving a fabricated shared scaffold. Series summaries are emitted above the configured minimum size. Deterministic Butina clustering uses achiral Morgan fingerprints and the configured Tanimoto similarity cutoff, with stable member ordering and representative choice. These analysis clusters are distinct from the MiniBatchKMeans groups used for model split sensitivity.

Each structure receives nearest-neighbor identifiers/similarities under both achiral and chiral Morgan fingerprints. `local_novelty_achiral` is one minus the nearest achiral similarity **within the analyzed public primary-task set**. It is a local dataset statistic, not a patentability, global novelty, or freedom-to-operate conclusion.

### Activity cliffs, matched pairs, and connectivity variants

Fingerprint activity cliffs are pairs in the primary task whose achiral Morgan Tanimoto similarity and absolute pActivity difference meet configured thresholds. The table also records chiral similarity, identical-achiral-fingerprint and same-connectivity flags, series membership, shared document/assay context, an evidence grade (`same_assay`, `same_document`, or `cross_context`), and SALI when it is numerically defined. These pairs require source and assay review because representation choice, measurement error, and protocol differences can create apparent cliffs.

The matched-molecular-pair analysis uses RDKit MMPA fragmentation with exactly one cut and configurable core/variable heavy-atom constraints. It retains a deterministic best qualifying core per pair and records the transform, core, variable fragments, potency delta, fingerprint similarity, cliff status, and context grade. This intentionally conservative enumeration is descriptive; an observed transform is not a causal substituent effect and pairwise rows are not independent observations.

The connectivity audit groups structures by the first InChIKey block and reports members, summaries, and potency-difference pairs above the configured threshold. Same connectivity can still mask stereochemical, protonation, tautomeric, salt/form, source-identity, or registration differences. Every flagged group is a review queue, not permission to merge records.

### hERG evidence and transparent prioritization

Observed primary hERG labels override a model probability and produce `observed_non_blocker` or `observed_blocker`. For unobserved structures, the calibrated prediction is used only inside its recorded applicability domain. In-domain probabilities are mapped to `predicted_lower_concern`, `predicted_indeterminate`, or `predicted_high_concern` using configured bounds. A missing prediction is `unknown_missing_prediction`; an out-of-domain prediction is `unknown_outside_applicability_domain`. Both unknown states receive no safety desirability or score credit. They are not treated as safe, low risk, or zero liability.

The chemistry gate combines valid structure, non-heterogeneous primary evidence, configured potency/spread/property limits, and the configured PAINS policy. It feeds five auditable follow-up tiers:

1. `priority_1_balanced_public_evidence`: passes chemistry and has observed non-blocker or in-domain predicted-lower-concern evidence;
2. `priority_2_potent_safety_data_gap`: passes chemistry but hERG evidence is unknown;
3. `priority_3_potent_liability_flag`: passes chemistry but has blocker, high-concern, or indeterminate hERG evidence;
4. `priority_4_chemistry_review`: is strong on potency but fails the chemistry gate; and
5. `priority_5_context_only`: all remaining structures.

A valid build can contain no `priority_1_balanced_public_evidence` structures. That result describes the evidence gaps and is not a pipeline failure. It must not be repaired by weakening thresholds after inspecting the output.

`discovery_score_without_safety` explicitly omits safety. `complete_evidence_score`, its rank, and the complete-evidence Pareto analysis exist only when a usable hERG desirability is available. The output includes objective-by-objective decision traces, base/leave-one-objective-out/emphasized-objective sensitivity scenarios, rank stability, chemistry-gated Pareto/frontier membership, and reason-coded data gaps. Missing inputs follow the recorded `unknown_no_credit` policy. PK/ADMET contributes only endpoint/species/matrix observation coverage and a missing-coverage flag; it is not converted into desirability or added to either score.

### Approved references and prospective experiment template

The configured approved-reference panel uses source-linked PubChem structures and dated FDA status/indication text for revumenib and ziftomenib. `approved_reference_coverage.csv` asks only whether each reference appears in the exact primary task or any public Menin measurement, how similar its achiral/chiral fingerprints and scaffold are to the primary set, and what descriptors/alerts the same pipeline computes. This is a coverage benchmark for the assembled public evidence—not an efficacy, safety, superiority, or therapeutic-equivalence comparison. Regulatory status, labels, indications, and database structures are time-sensitive and must be rechecked for a frozen release.

When enabled, `analysis.prospective_selection` produces a diversity-capped public-data experiment-design template with configurable quotas for potent safety gaps, liability characterization, novel-scaffold exploration, activity-cliff pair confirmation, lower-potency negative controls, and PK bridges. A configured maximum per scaffold series limits concentration in a single series; the summary exposes requested quotas, selected counts, and shortfalls rather than silently backfilling categories. The plan should be frozen before testing. It is not prospective validation (no new outcome has yet been observed), a lead recommendation, or evidence that any listed compound is safe, efficacious, available, synthesizable, or legally usable.

## Splitting and leakage control

All split strategies first resolve a registered-structure grouping key. Repeated rows for one `structure_id` remain in a single partition.

| Strategy | Grouping and use |
| --- | --- |
| `scaffold` | Bemis–Murcko scaffolds; acyclic compounds use exact canonical-structure groups. Preferred primary chemical-generalization estimate. |
| `chemical` | MiniBatchKMeans groups over fingerprints. Sensitivity analysis for chemical-region holdout. |
| `temporal` | Earliest known year per structure; later dated structures form the holdout and undated structures remain in training. Requires at least 50% dated structures and a viable holdout. Public years mix publication/patent document years with PubChem assay-deposit years, so use only as a disclosed sensitivity. |
| `random` | Random holdout of whole compound groups. Optimistic comparator, never the only publication estimate. |

Cross-validation uses compatible group-aware folds inside the training partition. If the requested split is impossible, the implementation records a fallback reason and uses compound-grouped random splitting. A fallback must be disclosed; it is not equivalent to the requested design.

## Models and selection

The default primary Menin task is IC50 × biochemical binding because it has the strongest assay-context and public source-year coverage in the current snapshot. Outside `--fast` mode, the model stage also enumerates every observed eligible endpoint × assay-family pair and trains each additional task that meets the configured minimum compound count. Configured endpoint tasks below that threshold still emit a machine-readable insufficient-data status. This breadth is an analysis inventory, not permission to treat task metrics as interchangeable or select the most favorable task after seeing its holdout.

Menin regression candidates are a dummy median baseline, Ridge models at two regularization strengths, and Extra Trees. Selection minimizes training cross-validation MAE. The untouched holdout reports MAE, RMSE, median absolute error, R², Pearson and Spearman correlation, signed error, fractions within 0.5 and 1 log unit, and calibration slope/intercept.

hERG classification candidates are a dummy prior baseline, class-balanced logistic models at two regularization strengths, and class-balanced Extra Trees. Selection maximizes training cross-validation precision–recall AUC. The selected estimator is refit and calibrated with Platt sigmoid using the same audited group-aware training folds when support allows. The holdout reports ROC-AUC, PR-AUC, balanced accuracy, sensitivity, specificity, precision, F1, Matthews correlation, Brier score, log loss, expected calibration error, and a confusion matrix.

Candidate holdout metrics are diagnostic only; selecting a final candidate from them would leak holdout information. The `selected_from_training_cv` field identifies the valid selection path.

## Uncertainty and applicability domain

- Core holdout metrics receive paired nonparametric 95% bootstrap intervals by resampling whole Bemis–Murcko scaffold groups (acyclic structures retain their exact-structure scaffold grouping). Row resampling is used only when a usable scaffold grouping cannot be formed, and the recorded `resampling_unit` discloses the fallback.
- Menin predictions receive a split-conformal-style symmetric interval whose radius is a finite-sample quantile of training out-of-fold absolute residuals. Its nominal and observed holdout coverage are reported.
- hERG probabilities report entropy as a communication measure of predictive ambiguity and receive calibration diagnostics.
- Each query is compared with training compounds using maximum fingerprint Tanimoto similarity. The domain threshold is the fifth percentile of sampled training nearest-neighbor similarities. Results include the nearest structure and an inside/outside flag.

These diagnostics are conditional on the curated dataset and model family. The applicability domain is a heuristic, and ordinary conformal guarantees can be weakened by scaffold or temporal distribution shift.

## Artifacts and reproducibility

Model manifests record the input-table digest and its declared digest-column list, split and cross-validation hashes, feature backend, estimator, selection metric, package environment, artifact hash, trust boundary, code revision/dirty status, resolved settings, and processed-build linkage. `skops` is preferred for serialization; a `joblib` fallback is explicitly labeled unsafe for untrusted files.

When chemical intelligence is enabled, six release-level manifests cover raw data, processed data, software/configuration/environment files, model artifacts, analysis artifacts, and reports. They contain relative paths, SHA-256 hashes, byte sizes, tabular schemas/row counts where readable, and a shared processed-data build ID. Upstream links form a verifiable DAG: raw → processed; processed + software → models; processed + models + software → analysis; and processed + models + software + analysis → reports. Verification checks content and metadata, direct analysis-input hashes, and those build/digest relationships. Disabling analysis leaves the five-manifest raw/processed/software/models/reports chain.

Public collection, processed curation, model generation, chemical intelligence, and reporting use staging before promotion. A failed collection does not replace the prior all-source raw snapshot. Curation completes every output in staging before whole-directory promotion with backup/rollback. A failed model-stage promotion restores the prior model/report evidence, while failed analysis and report promotions restore their prior roots. `--fast` isolates reduced artifacts under `research/models/smoke/`, `research/analysis/smoke/`, and `research/reports/smoke/`; it is an interface test, not release evidence, and it still reads the configured raw/processed snapshot.

## Alignment with QSAR validation principles

The workflow is designed around the [OECD QSAR validation principles](https://www.oecd.org/en/publications/guidance-document-on-the-validation-of-quantitative-structure-activity-relationship-q-sar-models_9789264085442-en.html): a defined endpoint, an unambiguous algorithm, a defined applicability domain, appropriate measures of fit/robustness/predictivity, and mechanistic interpretation where possible. It addresses the first four procedurally; mechanistic interpretation, true external validation, and prospective confirmation remain required before strong scientific claims.
