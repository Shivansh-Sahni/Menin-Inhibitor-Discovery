# Bias, missingness, selection, and subgroup plan

Status: `partially_executed` analysis contract. The frozen ChEMBL-only corpus
now has source/task missingness, attrition, composition, concentration,
temporal/stage, compatible-stratum, and support artifacts. Corrective weighting,
post-model performance comparisons, and fairness claims remain unexecuted.

## 1. Target populations

Every task must name its target population before modeling. Examples include “small-molecule biochemical `Kd` measurements against reviewed human single-protein constructs,” “room/physiological-temperature functional hERG `IC50` measurements,” or “human oral parent-drug concentration-time studies.” “All drug-like molecules” and “all proteins” are not operational populations without sampling frames.

Report four populations separately:

1. **source universe**: all records returned by the frozen source/query;
2. **resolvable universe**: records with usable entity identity;
3. **analysis eligible**: records satisfying the endpoint/protocol policy;
4. **model ready**: aggregated labels surviving split, feature, and minimum-support rules.

Attrition tables must link every excluded record to one or more non-overwriting reason codes. The model-ready population must never be used as the denominator for source coverage.

## 2. Selection diagram

The main selection process is:

`scientific interest -> experiment conducted -> result quantified -> result disclosed -> source indexed -> query retrieved -> entity resolved -> metadata/unit accepted -> label policy accepted -> feature computed -> split admitted`

Each arrow can depend on target family, molecule series, potency, toxicity, sponsor, assay technology, time, and documentation quality. Therefore model-ready labels are not an iid sample of underlying molecular behavior, and missingness is not assumed at random.

## 3. Bias register and required diagnostics

| Bias mechanism | Likely consequence | Required diagnostic | Mitigation/claim boundary |
|---|---|---|---|
| Publication and patent bias | Potent or novel compounds overrepresented; failed programs and negatives underreported | Source/year/endpoint value distributions; disclosed positive/negative ratio; patent-family concentration | Source-stratified results; do not infer prevalence or success probability |
| Target popularity | Kinases/GPCRs and selected safety targets dominate | Records/compounds/assays per target and family; Lorenz/Gini and top-k share | Macro-average by target; family and low-support holdouts; disclose unsupported families |
| Medicinal-chemistry series bias | Near analogs inflate effective sample size and random-split performance | Scaffold/cluster/patent-family sizes; nearest-neighbor distributions; effective cluster count | Cluster-aware splits/bootstrap; series cap only as prespecified sensitivity, not silent deletion |
| Assay/laboratory bias | Protocol and lab effects masquerade as structure effects | Endpoint-by-assay/source/lab distributions; within-compound conflicts; hierarchical variance components | Endpoint/context-specific tasks; assay/source holdout; random effects where identifiable |
| Quantification/censoring bias | Exact-only labels overrepresent measurable middle ranges | Relation (`=`, `<`, `>`, interval) by value/source/task; exact versus censored distributions | Censored likelihood or separate exact-only analysis; never substitute limits as exact |
| Documentation-quality selection | Valid units/structures/context concentrated in curated sources/eras | Eligibility rate by source/year/endpoint/target/family; standardized differences | Report attrition; source leave-one-out; avoid claiming representativeness |
| Aggregator duplication | Evidence count and train/test performance inflated | Provenance graph, exact/near mirror candidates, publication/assay IDs | Collapse only linked mirrors; cluster all plausible mirrors across splits; no-collapse sensitivity |
| Chemical identity resolution | Salts/stereo/mixtures lost or selectively excluded | Resolution status by source/task; parent-vs-submitted collision counts | Preserve submitted forms; identity-policy sensitivities; quarantine ambiguity |
| Temporal availability | Later records or model-pretraining knowledge leak into tests | Earliest disclosure, source ingestion, structure release, and model cutoff distributions | Use conservative max-known cutoff and embargo; post-cutoff evaluation |
| hERG threshold/ambiguous selection | Extreme classes and high blocker prevalence make task easier/nonrepresentative | Full continuous distribution, ambiguous fraction, prevalence by source/protocol | Continuous model where possible; threshold grid; report specificity, calibration and prevalence |
| PK context selection | Dose/route/species/formulation determine observed endpoints and availability | Endpoint missingness by context and chemical descriptors; study-level sampling patterns | Context-specific tasks; concentration-time preference; no naïve pooling or intrinsic PK labels |
| Clinical registry/result availability | Registered and positive studies easier to find than unpublished outcomes | Results-posted fraction by sponsor/phase/status/time; mapping confidence | Registration is not outcome; retain `not_reported`; no negative label from missing results |
| External model training overlap | Benchmark performance inflated by memorized proteins/ligands/assays | Training-cutoff, sequence/ligand/pocket similarity, source and assay overlap | Remove/stratify overlaps; label overlap-unknown; avoid universal superiority claims |

## 4. Missing-data analysis

### 4.1 Inventory

For every release, generate missingness counts and fractions by field, source, source release, endpoint, assay family, target/family, evidence stage, year, relation/censoring type, and eligibility state. Include explicit “not collected,” “not reported,” “not applicable,” “failed parse,” and “redacted” categories; do not collapse them into null.

Required field groups:

- molecule identity: structure, stereo, salt/material, molecular form;
- protein identity: accession/version, sequence, species, isoform, construct, mutation, complex;
- assay context: method, system, cell line, temperature, pH, time, substrate, concentration;
- result: endpoint, statistic, relation, value, unit, uncertainty, replicate count;
- PK context: species, route, dose, formulation, matrix, analyte, time course;
- clinical context: study version, phase, status, arms, dose, outcome, results, population;
- provenance/rights: source record, publication, version, retrieval time, license class.

Publish clustered missingness heat maps and co-missingness patterns, but retain underlying tabular counts. Compare included versus excluded populations on source, time, endpoints, chemical descriptors, target families, and values available before exclusion. Use standardized mean/proportion differences with cluster-aware intervals; p-values alone are not evidence of practical balance.

### 4.2 Mechanisms and analyses

- Do not claim MCAR from a nonsignificant test.
- Primary analyses should avoid imputing labels and avoid imputing scientific context that defines a task.
- Feature imputation is fitted inside training folds and includes missingness indicators where meaningful; molecules or proteins lacking the minimum representation may be assigned a separate modality rather than a fabricated embedding.
- If covariate-based inverse-probability weighting is attempted, define the observation/eligibility estimand, fit weights on training data, trim extreme weights by a prespecified rule, report effective sample size, and treat unmeasured-selection assumptions as unverified.
- Pattern-mixture or delta-adjustment sensitivity analyses must test departures from missing at random. They are not corrections that recover unknown truth.
- Multiple imputation is allowed only for compatible auxiliary covariates, never for held-out endpoint labels or source-reported assay conditions that define the endpoint.

### 4.3 Negative evidence

Missing measurement, failed retrieval, no clinical result, no label mention, and out-of-domain prediction are not negative labels. A negative requires an explicit experimental/curated definition, appropriate controls, and recorded detection threshold. Screening “inactive” labels must preserve tested concentration, assay quality, and source definition.

## 5. Subgroup evaluation

Subgroups are scientific performance slices, not claims of biological fairness unless a human population and ethically appropriate demographic data are actually represented.

Mandatory axes when available:

- target family, organism, sequence novelty, pocket similarity, construct/mutation status;
- endpoint, assay family/technology, source/lab, year, relation/censoring, evidence quality;
- molecule scaffold/cluster, molecular weight, charge/basicity, stereo completeness, modality, alerts;
- in/out of applicability domain and distance/similarity bins;
- PK species, route, formulation, dose band, matrix, analyte;
- hERG protocol/temperature/cell system and concentration range;
- clinical phase, indication, sponsor class, results-posted status, and mapping confidence;
- sex, age, race/ethnicity and geography only when definitions, consent/governance, denominators, and missingness support responsible analysis.

Rules:

1. Predefine primary subgroups and directionally plausible interactions before inspecting the locked test set.
2. Report subgroup sample count, compound count, target count, cluster count, event count, prevalence/range, and uncertainty.
3. Suppress performance claims for fewer than 30 independent groups or fewer than 20 events/non-events for classification; show coverage only and label `insufficient_support`. Higher thresholds may be required for complex models.
4. Use hierarchical partial pooling for estimation when justified, but never use shrinkage to conceal unsupported raw counts.
5. Test interactions directly rather than comparing “significant” in one subgroup and “not significant” in another.
6. Control multiplicity within the prespecified subgroup family. All post-hoc slices are exploratory.
7. Evaluate intersectional subgroups only when disclosure risk and support allow; do not publish small clinical cells.

## 6. Representation and measurement bias

Fingerprint, graph, sequence-language, and structure representations encode different notions of similarity. Performance and applicability domains must therefore be recomputed per representation. Feature attribution is not a causal explanation and may be unstable under correlated descriptors.

Chemical standardization can create group-dependent failures (metals, covalent species, mixtures, peptides, macrocycles). Report parser/feature failure rates by modality. A tool cannot claim broad molecular coverage after silently dropping hard modalities.

Protein language and structure models may have uneven accuracy across disorder, membrane proteins, complexes, cofactors, mutations, and low-homology sequences. Report these strata and abstain when required inputs or confidence are insufficient.

## 7. Release outputs

Each release must contain:

- source-to-model attrition table with reason codes;
- missingness matrix and co-missingness report;
- source/target/family/scaffold concentration report;
- exact/censored and positive/negative/ambiguous distributions;
- selection-model diagnostics and effective sample sizes if weighting is used;
- subgroup support and performance table with suppressed unstable cells;
- parser/feature failure rates by modality;
- a narrative statement of which target population the results do and do not represent.

Those descriptive outputs now exist for the frozen ChEMBL-only platform corpus
and were reproduced byte-identically. Bias/fairness **model-performance**
readiness remains `planned`: the corpus is one admitted source, no final model
or external lockbox is accepted, and no representation result is treated as
demonstrated fairness.
