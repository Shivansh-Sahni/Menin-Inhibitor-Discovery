# Evidence and endpoint ontology

Audit status: `locally_evidenced` as the normative platform contract and for the
ChEMBL 37 canonical implementation. External source bundles remain acquisition-
only until source-specific rights, identity, endpoint, and admission gates pass;
the ontology is not evidence that every declared modality is populated.

## 1. Observation model

One row is one source-supported observation, never a compound-level fact. The minimum composite identity is:

`molecule_entity + protein_entity + biological_system + assay_or_study + endpoint + statistic + relation + value + unit + provenance`

Replicate, aggregate, and prediction rows are distinct observation kinds. Aggregation never overwrites raw observations.

Required orthogonal axes are:

| Axis | Controlled values or required semantics |
|---|---|
| Observation kind | `experimental_raw`, `experimental_summary`, `curated_assertion`, `derived`, `prediction` |
| Evidence stage | `preclinical_in_vitro`, `preclinical_ex_vivo`, `preclinical_in_vivo`, `clinical_registry`, `clinical_results`, `regulatory_label`, `postmarketing` |
| Development stage | `discovery`, `explicit_preclinical`, `ind_enabling`, `phase_1`, `phase_2`, `phase_3`, `phase_4`, `approved`, `withdrawn`, `unknown`; never inferred from missing trial evidence |
| Result status | `reported`, `not_reported`, `pending`, `terminated`, `withdrawn`, `not_applicable`; trial registration is not a result |
| Access class | `public_redistributable`, `public_access_restricted`, `licensed`, `confidential`, `unknown` |
| Evidence quality | `raw`, `parsable`, `identity_resolved`, `protocol_sufficient`, `gold`, `quarantined`, plus explicit reasons; never inferred solely from evidence stage |
| Inclusion status | `included`, `review`, `quarantined`; exclusion/review reasons remain explicit |
| Numeric relation | `=`, `~`, `<`, `<=`, `>`, `>=`, `interval`, `not_reported`; an interval requires explicit lower and upper bounds |
| Provenance | Source, release/version, stable record identifier, retrieval time, original row/record locator, transformation lineage, checksum, citation, and license snapshot |

The public, preclinical, clinical-development, clinical-results, regulatory, and postmarketing views are filters over one provenance-preserving evidence graph. They are not successively “truer” copies of the same table.

## 2. Molecule entity

The canonical molecule record must preserve both the submitted and standardized forms. Required fields include source identifier, submitted representation, standardized parent, salt/solvate components, formal charge, isotope state, stereochemistry completeness, tautomer policy/version, canonical SMILES, InChI, InChIKey, and standardization status/reason.

Entity-resolution keys are use-case dependent:

- exact submitted material for assay reconciliation;
- parent identity for some cross-source analyses;
- stereo-aware identity for modeling where stereochemistry is specified;
- stereo-unspecified parent only as an explicitly labeled sensitivity view.

Salts, stereoisomers, mixtures, prodrugs, active metabolites, and covalent warheads must not be silently collapsed. If structure resolution is ambiguous, retain the observation with an unresolved identity status rather than fabricating a canonical molecule.

## 3. Protein and biological entity

Target identity is sequence- and construct-aware. Required fields include stable protein accession and accession version, organism and taxonomy identifier, canonical sequence checksum, isoform, construct boundaries, mutations, post-translational state when known, complex partners, target role, and source target identifier.

The following are not interchangeable:

- gene, canonical protein, isoform, mutant, construct, domain, multimer, and protein complex;
- orthologs from different species;
- intended target, assay target, off-target, carrier, transporter, and metabolizing enzyme;
- a sequence supplied to a model and a structure actually used by an assay.

Unknown construct details remain missing. Mapping an assay to a canonical sequence is a derived link with method and confidence, not a source fact.

## 4. Binding and functional potency

Endpoint identity is preserved before any transformation or aggregation.

| Endpoint | Meaning | Permitted primary transform | Non-equivalence boundary |
|---|---|---|---|
| `Kd` | Equilibrium dissociation constant | `pKd = -log10(Kd in mol/L)` | Not `Ki`, `IC50`, or `EC50` |
| `Ki` | Inhibition constant under a specified model | `pKi = -log10(Ki in mol/L)` | Requires inhibition model/context; not automatically `Kd` |
| `IC50` | Concentration producing 50% inhibition in an assay | `pIC50 = -log10(IC50 in mol/L)` | Assay- and substrate-dependent; not thermodynamic affinity |
| `EC50` | Concentration producing 50% effect | `pEC50 = -log10(EC50 in mol/L)` | Functional potency; not binding affinity |
| `kon`, `koff` | Association and dissociation rates | log transform with explicit units | Do not infer one from endpoint potency |
| Residence time | Kinetic persistence under a stated definition | definition-specific | Often derived from `koff`; retain derivation |
| Fraction/percent effect | Response at a stated concentration/time | none by default | Not an `IC50` without curve fitting |

Relations (`=`, `~`, `<`, `<=`, `>`, `>=`, `interval`, `not_reported`) are part of the label. An unequivocal reported range is represented as `interval` with both bounds; an ambiguous range remains raw and quarantined. Censored values must be modeled as censored or analyzed in a separate exact-only view; replacing a bound or one side of an interval by its numeric endpoint as if exact is prohibited.

Standard binding free energy may be derived only from a compatible equilibrium constant:

`Delta G standard = R * T * ln(K / C standard)`

The derivation must store endpoint type, temperature, standard state, unit conversion, sign convention, and uncertainty. `IC50` or `EC50` must never be relabeled as binding free energy. If temperature is unavailable, a reference-temperature calculation is a clearly labeled approximation, not a measured value.

Assay context must retain method, detection technology, substrate/competitor, concentration, incubation time, pH, temperature, buffer, construct, cell line, and organism when reported. Missing context must not be imputed from another record.

## 5. Pharmacokinetics and ADME

In-vitro ADME, animal PK, and human PK are separate task families. The minimum PK context includes species, strain, biological matrix, route, dose, dose unit, formulation, regimen, analyte, sampling schedule, fed/fasted status when relevant, and statistic type.

| Family | Examples | Required boundary |
|---|---|---|
| In-vitro ADME | microsomal/hepatocyte stability, permeability, solubility, plasma-protein binding, CYP inhibition | System, species, method, concentrations, and units are part of the endpoint |
| Concentration-time | time, concentration, matrix, individual/summary | Preserve raw time course and subject/animal grouping where available |
| Noncompartmental PK | `Cmax`, `Tmax`, `AUC`, terminal half-life, clearance, volume, bioavailability | Derived method, dose, route, sampling adequacy, and uncertainty required |
| Clinical exposure | steady-state/trough/peak/AUC and variability | Population, formulation, regimen, and analysis population required |

`Cmax` and `AUC` are dose-, route-, and formulation-dependent and are not intrinsic molecular labels. Oral and intravenous clearance, apparent and absolute volume, total and unbound exposure, and parent and metabolite measurements must remain distinct. Dose normalization is a derived exploratory view unless dose proportionality is supported.

## 6. hERG, QT, and cardiac safety

Cardiac evidence is a hierarchy of related but non-equivalent observations:

1. hERG/Kv11.1 biochemical binding;
2. static functional current block (including protocol, temperature, cell system, pulse pattern, and endpoint);
3. dynamic or state-dependent hERG assays;
4. multichannel ion-current panels;
5. cellular electrophysiology or action-potential assays;
6. in-vivo ECG/QT observations;
7. human concentration-QTc analysis and thorough-QT evidence;
8. adjudicated clinical arrhythmia outcomes and postmarketing evidence.

A hERG model estimates only the endpoint and assay domain on which it was trained. It is not a clinical cardiotoxicity, QT-prolongation, torsades-de-pointes, or patient-risk model. The ICH E14/S7B framework explicitly combines nonclinical and clinical evidence; it does not authorize replacing that evidence with a hERG classifier ([ICH E14/S7B Q&As, Step 4, 2022](https://database.ich.org/sites/default/files/E14-S7B_QAs_Step4_2022_0221.pdf)).

Project-defined blocker/nonblocker thresholds must be stored as versioned analysis policies. An ambiguous interval is neither a negative class nor missing at random.

## 7. Clinical and regulatory evidence

Clinical-development evidence requires stable study identifier, registry version/data timestamp, intervention identity and mapping confidence, sponsor, phase, status, condition, arms, dose/regimen where reported, dates, outcome definitions, analysis population, results-posted status, and citations.

The following must remain separate:

- trial registration versus posted results;
- recruitment status versus efficacy or safety outcome;
- trial phase versus probability of success;
- investigator attribution versus regulatory finding;
- regulatory approval for one indication/formulation versus broad molecule safety;
- adverse-event reports versus causal effects.

ClinicalTrials.gov refreshes its data on weekdays and exposes a `dataTimestamp`; every extract therefore requires a dated snapshot ([ClinicalTrials.gov API](https://clinicaltrials.gov/data-api/api)). Drugs@FDA and DailyMed supply complementary application and current-label evidence, not identical drug records ([Drugs@FDA data files](https://www.fda.gov/drugs/drug-approvals-and-databases/drugsfda-data-files), [DailyMed API](https://dailymed.nlm.nih.gov/dailymed/app-support-web-services.cfm)).

## 8. Aggregation and conflicts

Aggregation keys must include endpoint, assay family, biological target/construct, species/system, and relevant protocol dimensions. Report count, median, range, dispersion, censoring fraction, source count, and conflict flags. Cross-source duplicates may be linked into mirror groups, but equality of values is insufficient proof that records are the same experiment.

Conflicts are preserved as data. A precedence policy may select an analysis view, but must not delete contrary evidence. Model-ready labels are materialized artifacts linked to all included and excluded observations and to a versioned policy.

## 9. Claim record

Every reported result should be accompanied by:

- claim identifier and exact text;
- endpoint, population, evidence stage, and observation kind;
- dataset build and split identifiers;
- estimand, metric, confidence interval, and multiplicity family;
- applicability-domain and abstention policy;
- supporting artifact checksums;
- limitations and prohibited extrapolations;
- responsible reviewer and review date.

The platform may expose a broad evidence graph while still requiring endpoint-specific training tasks. Breadth is not permission to pool scientifically distinct labels.
