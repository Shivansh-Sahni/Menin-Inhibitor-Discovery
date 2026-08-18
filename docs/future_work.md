# Remaining roadmap

The rebuilt pipeline now covers strict unit/target curation, RDKit parent identity, endpoint/assay stratification, cross-source mirror linkage and no-collapse sensitivity, quarantines, quality audits, compound-grouped splits, baseline model comparison, calibration, scaffold-group bootstrap uncertainty, applicability-domain checks, primary-task chemical intelligence, staged promotion, and a six-manifest content-addressed DAG when analysis is enabled. The implemented chemical layer includes profiles/QED/apparent efficiencies, review alerts, Bemis–Murcko series, deterministic Butina clusters, achiral/chiral novelty, fingerprint and matched-pair cliffs, connectivity review, applicability-aware hERG evidence, transparent tiers/Pareto scoring, sensitivity traces, approved-reference coverage, and a diversity-capped public-data experiment template. The priorities below are the remaining scientific and operational work, not promises that every method will improve performance.

## Priority 0: release and governance gates

- Approve authorship/contributor, repository license, model-artifact license, and third-party data redistribution decisions.
- Complete an independent clean-environment reproduction of a frozen public-only build.
- Freeze resolved configuration, environment lock/container, code tag, all six enabled-stage verified release manifests, and source citations.
- Review all PubChem assay inclusion decisions and high-impact ChEMBL/BindingDB records against primary source material.
- Complete the [publication checklist](publication_checklist.md) and document any unmet gate in the manuscript.

## Priority 1: controlled internal-data integration

- Obtain Wang lab data-owner authorization and deploy the existing offline intake library only inside the controls in [proprietary data intake](proprietary_data_intake.md).
- Have the data steward review the HMAC key lifecycle, assay/endpoint registries, private output root, backup/audit policy, and disclosure threat model.
- Extend the current CSV/TSV/SDF intake contract beyond its existing governed cohort role for registration parent, lot/form, protocol version, curve-level QC, operator/site, campaign, and release-classification fields before those data are needed.
- Produce separate private roots/manifests for development, locked-external evaluation, and prospective-blind unblinding, as well as distinguishable public-only, internal-only, public-trained/private-test, and combined builds; do not route them through the public CLI.
- Detect public/private structure and document overlap before splitting.
- Enforce `development`, `locked_external`, and `prospective_blind` isolation outside the public model CLI; reserve scaffold- and time-forward internal holdouts and do not convert all private data into training labels.

## Priority 2: curation sensitivity and endpoint definitions

- Extend the implemented cross-source-mirror and clean-label sensitivities to quantify validity flags, assay variants, structure-parent rules, and exact/censored policies; source-review material mirror groups rather than treating the heuristic as ground truth.
- Add a reviewed PubChem assay registry with protocol-level endpoint/unit/target decisions.
- Scientifically review the automatically enumerated endpoint × assay-family tasks, freeze manuscript task definitions before holdout interpretation, and suppress strata whose protocol compatibility is inadequate despite meeting a numeric support threshold.
- Implement interval/censored regression or survival-style methods for threshold data; compare with exact-only selection bias.
- Add source-document review annotations for the implemented fingerprint/MMP/connectivity cliffs and conflicts above the configured within-structure spread threshold.

## Priority 3: stronger evaluation

- Pre-specify repeated scaffold or chemical-region splits to estimate split variance without tuning to the final holdout.
- Define a defensible temporal cutoff using actual assay/campaign dates when internal data arrive; keep the mixed public document/deposit-year analysis labeled as sensitivity-only.
- Build an external Menin evaluation set with non-overlapping structures and compatible assay protocols.
- Evaluate the TDC hERG benchmark in an isolated compatible environment after label reconciliation and overlap removal.
- Run prospective experiments on model-selected and uncertainty-selected compounds, including deliberately out-of-domain controls.
- Measure experimental utility: hit rate, potency improvement, hERG exposure margin, and decision cost—not only retrospective metrics.

## Priority 4: model and chemical-space analysis

- Add repeated/nested group-aware hyperparameter evaluation where data volume justifies it.
- Compare additional chemically appropriate baselines, such as count fingerprints, MACCS, alternative radii, gradient boosting, and carefully regularized graph models.
- Add uncertainty ensembles and compare empirical coverage of the current scaffold-group bootstrap and conformal intervals under scaffold and temporal shift.
- Validate the implemented fingerprint-cliff, conservative single-cut MMP, scaffold-series, Butina-cluster, and nearest-neighbor analyses across pre-specified representations and thresholds; source-review high-impact pairs instead of treating them as causal SAR.
- Stress-test QED/property windows, review-alert policy, apparent ligand-efficiency/LLE, hERG evidence bands, tier gates, objective weights, Pareto membership, and rank sensitivity against prospectively declared alternatives.
- Add interpretable substructure diagnostics with stability checks; avoid causal interpretation of attribution maps.
- Periodically revalidate the configured revumenib/ziftomenib PubChem structures and FDA status/indication links; retain approved-reference coverage as a benchmark of public-dataset representation, never as an efficacy comparison.
- Have Menin chemistry, assay, safety, and statistics reviewers pre-register and source-check the implemented diversity-capped experiment template before testing; then measure category-specific yield and retain all negative/shortfall outcomes.
- Evaluate multi-task learning only after labels, assay contexts, and leakage controls are harmonized.

## Priority 5: endpoint-specific ADMET

- Harmonize candidate endpoints separately: microsomal/hepatocyte clearance, metabolic half-life, solubility, permeability, plasma protein binding/fraction unbound, CYP inhibition, exposure, and bioavailability.
- Require compatible unit, species, matrix, route, dose, time point, and protocol metadata before aggregation.
- Establish endpoint-specific minimum support, censoring, transformation, split, metric, uncertainty, and applicability-domain policies.
- Integrate measured exposure and free concentration before interpreting hERG margins.
- Keep a missing-evidence state distinct from a favorable prediction.

## Priority 6: operational maturity

- Add an approved secrets-free private-data interface outside the public repository.
- Add schema migration/versioning when processed contracts change.
- Add release automation that archives verification, configuration, environment, and artifact checksums.
- Add performance/resource tests for full public refreshes and deterministic retry/resume behavior.
- Add an access-controlled review dashboard only after threat modeling, authentication, audit logging, and release policy are approved.
- Monitor upstream schema, API, terms, and package changes.

## Stop/go criteria for a decision model

Do not use a model as a compound-selection gate until it has:

- a fixed, biologically meaningful endpoint;
- sufficient protocol-compatible support;
- zero registered-structure leakage;
- performance above a declared baseline on a chemical and/or time-forward holdout;
- acceptable calibration/interval behavior in the intended domain;
- an external or prospective evaluation;
- a defined experimental action threshold and cost analysis; and
- domain-expert and data-owner approval.

If these criteria are not met, the model remains an exploratory prioritization aid.
