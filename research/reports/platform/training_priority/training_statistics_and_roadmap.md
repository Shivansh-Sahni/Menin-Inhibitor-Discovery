# Verified inventory, training priorities, and continuation roadmap

**Evidence date:** 2026-08-07  
**Scope:** lead reconciliation of hERG, PK/ADME, affinity, canonical ChEMBL tasks, and adjacent million-scale modalities.  
**Boundary:** no substantive model has been trained in this expanded platform run; the newly acquired external sources have added zero canonical rows and zero training labels so far.

## Executive decision

The project has achieved multi-million to billion-scale **raw evidence**, but not a single homogeneous dataset with millions of interchangeable labels. The strongest conditional development sequence is **Kd pilot → Ki powered benchmark → IC50 separate potency benchmark**:

1. **Exact Kd controlled endpoint-semantics pilot:** 62,334 measurements, 25,573 molecules, and 3,610 proteins. It is manageable on modest hardware and has the most direct equilibrium endpoint semantics. It is not proven empirically cleaner: assay, construct, pH, temperature, and long-tail target heterogeneity remain.
2. **Exact Ki powered benchmark:** 407,926 measurements, 205,293 molecules, and 2,608 proteins. If only one substantive first model is allowed, Ki is probably stronger because 508 proteins have at least 100 rows and 91 have at least 1,000, versus 121 and three for Kd.
3. **Exact IC50 potency benchmark:** 1,107,075 measurements, 664,970 molecules, and 5,395 proteins. It is the largest immediately integrated task, but it is a separate assay-context-dependent potency quantity, not simply a third tier of equilibrium affinity.

The previously selected 137-row exact hERG task remains useful only as an end-to-end engineering smoke test. It is not a serious scientific training set. Expanded hERG, systemic PK, PRISM, LINCS, ToxCast, and JUMP require distinct admission or harmonization work before they can support the claims their raw sizes appear to promise.

## 1. What is actually ready now

The accepted ChEMBL 37 observation layer contains 3,938,372 records: 3,870,524 unique source activities plus 67,848 explicitly derived binding-free-energy records. The source observations comprise 2,690,547 included, 734,448 quarantined, and 445,529 review records. The layer also contains 1,304,011 molecules, 430,776 assays, 9,411 proteins, 2,548,198 default-task rows, 67,839 derived-sensitivity task rows, and 28 registered task views. “Accepted platform” describes a verified build/snapshot; it does not mean every row has accepted inclusion status or is a training-eligible experimental label. Twenty-two task views are mechanically integrated and six were support-skipped. Eleven task-level capped diagnostic suites completed, each with two simple estimators; these were pipeline checks, not substantive models. Scientific task-claim readiness and substantive-training authorization remain false until the claim-specific gates below pass.

### Exact binding tasks

| Task | Rows | Molecules | Proteins | Assays | Documents | Existing molecule-grouped engineering split: train / validation / test-label-omitted artifact | Target support |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Kd | 62,334 | 25,573 | 3,610 | 13,816 | 3,753 | 45,965 / 8,677 / 7,692 | 121 proteins have ≥100 rows; 3 have ≥1,000 |
| Ki | 407,926 | 205,293 | 2,608 | 39,733 | 14,209 | 286,048 / 61,257 / 60,621 | 508 proteins have ≥100 rows; 91 have ≥1,000 |
| IC50 | 1,107,075 | 664,970 | 5,395 | 96,569 | 31,776 | 776,752 / 166,411 / 163,912 | 1,238 proteins have ≥100 rows; 241 have ≥1,000 |
| EC50 | 89,725 | 61,213 | 1,314 | 7,860 | 3,729 | 62,972 / 13,401 / 13,352 | 164 proteins have ≥100 rows; 15 have ≥1,000 |

All values are positive, but raw nanomolar labels span up to nine orders of magnitude. Medians are 260 nM for Kd, 70 nM for Ki, 167 nM for IC50, and 202 nM for EC50. The displayed split counts are from the current deterministic molecule-grouped engineering split; “test-label-omitted” describes the split artifact, not an independently held prospective test or a physical access-control claim. The approximate hash fractions are not a claim-ready primary split. For all four large exact binding tasks, mandatory claim candidates are not fully materialized: scaffold splitting skips where an RDKit exception would require an exact-SMILES proxy, source holdout is inapplicable because the canonical layer has one source, and the relevant chemical-neighbor/protein-homology evidence is sampled or capped. The existing molecule-only ridge diagnostics on raw nM values perform poorly, which is useful negative evidence: the first real protocol should use log-molar/pActivity labels, include the protein input, and retain endpoint identity. A raw-nM molecule-only regression across thousands of proteins is not an adequate baseline.

### Exact versus censored affinity

The narrower integrated ChEMBL binding tasks contain 2,117,384 Kd/Ki/IC50/EC50 rows: 1,667,060 exact (78.732%) and 450,324 censored (21.268%). Exact-only data are appropriate for a bounded engineering start, but may introduce reporting/selection bias. A full-population endpoint claim requires a prespecified censor-aware likelihood or an explicitly exact-reported scope plus an exact-versus-censored distribution analysis. Censored values must not be converted to exact threshold values.

## 2. Affinity: the strongest large-scale foundation

The checksum-verified BindingDB 2026-08 archive has 3,234,499 physical rows, 2,481,305 source-defined distinct monomer–target keys, and 3,237,384 populated endpoint cells. “Monomer–target” retains BindingDB's source unit and does not imply parent/salt/stereo normalization. Of the endpoint cells, 2,589,275 are exact (79.98%) and 648,109 are censored or range-like (20.02%). IC50 dominates with 2,209,905 cells, followed by Ki 627,186, EC50 270,125, and Kd 130,168.

The new exact overlap audit proves why simple addition is wrong:

- 1,658,927 BindingDB physical rows—51.289% of the archive—are explicitly tagged as originating from ChEMBL.
- Of 1,658,153 eligible tagged measurements, 1,229,978 (74.178%) have an exact ChEMBL 37 match using ligand ChEMBL ID or exact InChIKey, normalized UniProt set, endpoint, censor relation, and numeric nM value.
- Compound identity resolves for 1,443,944 of 1,658,927 tagged rows (87.041%). However, 455,907 rows have a current ChEMBL ligand ID but a different exact InChIKey, exposing a major parent/salt/stereo/release-drift reconciliation queue.
- The explicitly ChEMBL-tagged subdomain supports a defensible union of 2,864,273–3,266,356 distinct standardized measurement keys, with 2,988,469 as the explicit-ID observed union. This is not yet the final full-BindingDB union because non-ChEMBL BindingDB lineages require their own admission and rights review.

**Training implication:** use the already integrated ChEMBL tasks first. Build an independent external validation set only from origin-audited, non-ChEMBL, non-mirror BindingDB literature or later-dated records after exact and near-overlap removal. Do not call the BindingDB mirror an external test.

## 3. hERG: large raw scale, small canonical truth

The gross local inventory now contains 773,908 wild-type/standard candidate rows plus 342,311 explicitly mutant-channel rows, for 1,116,219 physical hERG-related rows. The 1,360-row increase is the exact NVS/ERF EPA result inventory (1,219 NVS MC/SC rows plus 141 ERF MC/SC rows); the EPA Tox21 processing mirror is not added again. This gross accounting includes mirrors, repeated structures, assay replicates, controls, derived binary wrappers, and quarantined records. The current planning estimate remains 300,000–350,000 unique standardized structures, implying roughly 2.2–2.6 wild-type physical rows per eventual structure even before mutant rows are considered.

The saturation audit found 4,067 PubChem KCNH2 AIDs; 4,005 (98.476%) are ChEMBL or BindingDB mirrors. Tox21 AID 1671200 contributes 7,671 structures, of which 6,515 (84.93%) are absent from local ChEMBL, but it is a fluorescence thallium-flux functional proxy rather than patch clamp.

EPA invitrodb v4.3 contains exactly three human KCNH2 endpoints:

- NVS endpoint 686: legacy astemizole radioligand binding, an independent lineage.
- ERF endpoint 3184: dofetilide radioligand binding, newly released in v4.3 and the strongest likely new EPA evidence.
- Tox21 endpoint 3210: the same experimental lineage as PubChem AID 1671200; retain for processing/QC comparison, not as an independent experiment.

The completed full-archive streaming extraction retained 9,787 MC and 1,243 single-concentration result rows. After identity resolution, the three endpoints cover 7,959 DTXSIDs: 1,077 for NVS, 131 for ERF, and 7,871 for Tox21. Tox21 has exactly 9,667 mapped sample rows, matching PubChem AID 1671200, so it adds no independent experiment. Within EPA, NVS and ERF contribute only 88 DTXSIDs beyond Tox21 (13 NVS-only and 75 ERF-only). Eleven endpoint rows correspond to eight named controls lacking `sample` rows; they are preserved without identity imputation. Cross-source novelty versus ChEMBL and BindingDB remains unproven until a structure-resolved DTXSID crosswalk is applied.

Only 137 exact hERG IC50 rows and 120 binary rows have passed the current strict canonical contract. Therefore:

- use the 137-row task only to test code, not to select a production model;
- admit expanded evidence into separate manual patch-clamp, automated patch-clamp, radioligand-binding, fluorescence/proxy, wild-type, and mutant task strata;
- use PR-AUC, calibration, and prespecified operating points for classification;
- never interpret hERG predictions as QTc prolongation or torsade risk.

## 4. PK and ADME: the bottleneck is extraction quality, not document count

The currently acquired systemic-PK benchmarks total only 2,437 context-incomplete rows. Another 274,168 physical rows are upstream ADME/safety observations, and 330,261 ChEMBL hits across 85,441 molecules are reclassification candidates. PK-DB advertises 138,411 outputs, but anonymous access yielded zero and rights/access remain unresolved.

The completed DailyMed scan materially improves the evidence layer:

- 54,672 of 54,672 SPL packages parsed, with zero errors;
- 41,960 candidate document versions and 43,162 candidate sections;
- 26,610 tables and 313,014 bounded evidence spans;
- 29,664 Tier-A, 4,324 Tier-B, and 9,174 Tier-C sections; Tier A is 68.727% and contains machine-detected human/population, route, dose, matrix, and unit context in the same section;
- the latest-version view contains 40,247 SETIDs and 41,395 sections;
- 19,652 candidate versions (46.835%) join exactly to Drugs@FDA;
- only 10,976 candidate versions (26.158%) expose an active ingredient through the strict XML path.

Endpoint keywords are nonexclusive and indicate candidate-document coverage, not extracted measurements: half-life occurs in 39,116 sections, Cmax in 32,235, clearance in 30,445, AUC in 27,863, bioavailability in 24,959, volume of distribution in 20,665, and Tmax in 14,191.

These are excellent curation candidates, not labels. Context-complete admitted systemic-PK measurements remain **zero**. The immediate PK task is table-aware extraction plus ingredient/structure resolution and human adjudication. Upstream solubility, permeability, protein binding, CYP/transporter, metabolic-stability, and intrinsic-clearance models may be trained sooner, but they must remain separate and must not be advertised as clinical PK.

## 5. Other high-value scale

| Modality | Verified scale | Current value | Why it is not first |
| --- | ---: | --- | --- |
| PRISM | 8,506,400 observed viability values; 93.792% combined matrix coverage | Drug–cell response and repurposing | Must harmonize parent compound, dose/sample, cell line, screen phase, and leakage groups |
| LINCS L1000 | 1,665,114 profiles; 591,697 signatures; 1,628,481,492 measured profile–gene positions | Mechanism-of-action and cell-state representation | Replicate/signature semantics and cell/dose/time effects; full modeling is compute-heavy |
| EPA ToxCast | >500 million relational records in a 17,651,807,605-byte compressed database dump | Multitask mechanistic toxicity | Gzip/checksums and the targeted hERG stream extraction pass, but the full source is not expanded into a local database; relational records are not labels; broad curve collapse, sample identity, and endpoint lineage remain required |
| JUMP Cell Painting | ~1.540 billion cells; 58.267 million TIFFs; 358.4 TB remote payload | Understudied morphology and mechanism | Requires object storage, batch correction, image preprocessing, and HPC |

The best non-HPC adjacent-model candidate is PRISM after identity harmonization. LINCS, ToxCast-wide multitask learning, and JUMP should wait until their data contracts are stable and HPC/object storage are approved.

## 6. Training sequence

### Stage A — final pre-training decisions

1. Professor approves intended use: research decision support, not clinical safety or dosing.
2. Freeze separate endpoint contracts for Kd, Ki, IC50, EC50, hERG modality, each ADME quantity, and every systemic-PK quantity.
3. Approve source rights, top-level code license, checkpoint policy, and prohibited claims.
4. Freeze molecule, analog/scaffold, protein, target-family, source/assay, temporal, and double-cold splits. Keep the prospective test labels sealed.

### Stage B — first CPU/modest-hardware models

1. **Kd controlled pilot:** transform exact nM values to log-molar/pKd for numerical stability while retaining raw nM. Compare target-median and nearest-neighbor controls, ECFP ridge/tree baselines, a D-MPNN, and a small protein-aware fusion model using frozen ESM-2 35M/150M embeddings. Treat derived standard free energy only as a deterministic Kd view using a declared 1 M standard state and recorded temperature or an explicit temperature assumption—not as new evidence or independent validation.
2. **Ki powered benchmark:** proceed only if the chosen generalization split has adequate target/family support and accepted near-leakage evidence. Reuse the fixed protocol on 407,926 rows. Report both aggregate and per-target results; do not let abundant targets dominate the scientific conclusion, and distinguish direct from derived Ki where source semantics allow.
3. **IC50 potency benchmark:** scale to 1,107,075 rows only after the controls behave sensibly. Keep IC50 distinct from equilibrium affinity and stratify by assay context. Handle EC50 independently as a smaller functional-potency task; do not pool it with IC50 or omit it by implication.
4. Add censored rows with an interval/censor-aware loss as a prespecified primary extension or limit claims explicitly to exact-reported measurements.
5. Run the 137-row hERG task only as a pipeline smoke. Begin scientific hERG training only after the expanded modality-specific corpus is admitted.

Primary regression metrics should be MAE on the log scale with cluster bootstrap confidence intervals, supported by RMSE and Spearman. Results must be stratified by target support, assay family, source, evidence tier, time, and maximum training similarity. Classification should emphasize PR-AUC, calibration, and fixed operating points. Every run needs label-permutation, identifier-only, target-only, and nearest-neighbor controls.

### Stage C — external validation

1. Create an origin-audited non-ChEMBL, non-mirror BindingDB source holdout after exact and near-neighbor removal.
2. Add temporal evaluation using the earliest defensible publication/patent date, not database modification time.
3. For hERG, reserve a high-quality independent patch-clamp set; use proxy assays only as separate auxiliary tasks.
4. For PK, reserve study- or document-level groups and keep clinical outcomes distinct from in-vitro ADME.
5. Use external/source-temporal data for tuning only when explicitly designated as validation. Keep a final external/prospective test sealed and open it once after all choices freeze. Report performance decay by ligand similarity, protein identity, target family, assay/source, and time; abstain on unsupported/OOD regions.

### Stage D — multimodal expansion

1. Harmonize a shared parent-molecule registry across ChEMBL, BindingDB, PRISM, LINCS, ToxCast, JUMP, DailyMed, and Drugs@FDA while retaining salts/formulations as linked entities.
2. Train PRISM drug–cell response with cell-line and compound cold splits.
3. Add endpoint-specific ADME models and use them as features only when causal/temporal leakage is excluded.
4. Under HPC authorization, add LINCS signatures, ToxCast multitask assay representations, and JUMP morphology embeddings.
5. Evaluate Boltz-2, RFAA, Chai-1, and Protenix only as frozen structure/affinity comparators after exact checkpoint, license, cutoff, and training-overlap audits. Structure predictions are auxiliary evidence, not labels.

## 7. Concrete go/no-go gates

- **Data gate:** every training row has resolved molecule, protein/target, endpoint, unit, relation, assay/source, provenance, and permitted use.
- **Primary-claim support gate:** declare one intended generalization claim and require prespecified minimum molecule, target, and family support in every partition. Kd cold-target/double-cold claims may be underpowered despite adequate total rows.
- **Leakage gate:** zero exact entity overlap for the declared cold split. The current chemical-neighbor and protein-homology audits are capped/non-exhaustive for several large tasks, so the chosen primary split needs an exhaustive or explicitly accepted near-duplicate/homology audit before scientific model selection.
- **Assay-quality gate:** quantify replicate disagreement, assay/source/construct completeness, direct-versus-derived Ki, Kd temperature/pH availability, unit validity, and outlier sensitivity before making endpoint-quality claims.
- **Baseline/effect-size gate:** a protein-aware model must show a prespecified, confidence-interval-supported improvement over target-aware simple and nearest-neighbor controls on tuning validation and at least one independent source/temporal validation before architecture scaling. A numerically higher score alone is insufficient.
- **Censor gate:** no full-population claim until a censor-aware analysis is complete; otherwise the claim must be limited to exact-reported measurements and selection bias analyzed.
- **PK gate:** no systemic-PK model until value-level endpoint/context links are human-verified; a proposed acceptance target is ≥95% field accuracy on a blinded audit with strong inter-annotator agreement.
- **hERG gate:** no broad safety claim without assay-stratified calibration and an independent patch-clamp evaluation. QT/QTc and clinical events remain separate endpoints.
- **HPC gate:** approve large compute only after non-HPC experiments demonstrate stable external signal, reproducible data loading, checkpoint/resume behavior, and a measured hardware budget.
- **Release gate:** institutional rights/license approval, model and dataset cards, full provenance manifests, and explicit research-only limitations.

## 8. Paper and product path

The strongest near-term paper is a data/resource and benchmarking contribution: exact source overlap, assay disagreement, censoring bias, missingness/selection into clinical reporting, and performance decay by chemical/protein/source/time similarity. A second paper can benchmark protein-aware affinity models on the frozen splits. A later multimodal paper can test whether PRISM, LINCS, ToxCast, and morphology add value beyond structure and sequence.

The eventual software should expose evidence and uncertainty, not a single unexplained score: molecule and protein inputs, endpoint-specific predictions, provenance-backed supporting observations, assay/evidence tier, OOD similarity, calibrated uncertainty, and abstention. The durable contribution is the evidence graph and evaluation discipline; model architecture can evolve behind that interface.

## Evidence bindings

The counts and decisions above are bound to the canonical build manifest and corpus acceptance, the exact BindingDB–ChEMBL overlap audit, the hERG integrated and exact EPA extraction reports, the full DailyMed report/validation replay, and the EPA acquisition manifest. The scientific/model ordering is grounded in `research/reports/platform/literature_decisions/comprehensive_literature_review.md` and its decision recommendations. Machine-readable paths are enumerated in `training_statistics.json`, and hashes for the upstream evidence are frozen in `source_binding_manifest.json`; the shared reconciliation history is in `docs/project/pretraining_readiness_ledger.md`.

## Final honest status

The platform is mechanically mature and now holds genuinely large evidence sources. It can support an approved, tightly scoped Kd endpoint-semantics pilot after claim-specific gates pass; Ki is the stronger candidate for the first statistically powered protein–ligand benchmark. It is not yet ready for a credible expanded hERG model, a systemic-PK model, a clinical cardiotoxicity claim, or billion-scale multimodal pretraining. The next highest-value work is canonical admission and identity/leakage reconciliation, followed conditionally by Kd pilot → Ki powered benchmark → separately stratified IC50 potency modeling.
