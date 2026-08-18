# hERG pre-HPC completion and handoff — 2026-08-09

## Outcome

All currently machine-executable preparation supported by the frozen local evidence in this continuation has been pushed to a reproducible handoff. Substantial human curation, new-source acquisition, and prospective experiments remain. The project now has source-bound candidate adjudication evidence, stronger label-blind evaluation challenges, a frozen WT sequence, matched-pair chemical analysis, an exposure-aware QT collection layer, corrected future feature/smoke contracts, and a fail-closed runtime/storage preflight.

This establishes a stronger data and experimental-design foundation. It does **not** establish predictive superiority, a gold standard, clinical safety, a QT-risk model, a completed exposure margin, or production-HPC readiness.

## Lead-reconciled releases

| Release | Manifest self-hash | Main result | Hard boundary |
|---|---|---|---|
| v1.4 review assets | `cca60c80eb4b817806e770d88d403854173129fb9256104da83827bb691a4dab` | 226 evaluation candidates; 1,340 real standardized-potency conflicts; 4,779 protocol priorities | candidate/review queues only |
| v1.5 candidate evidence | `98c58f301fe084da5d989a37c48927f5f93b682d2ed2696436fd3c3fa69fbbf8` | 4,601 local bindings; 951 lineage hypotheses; 1,566 blank human decisions | zero human decisions/gold promotion |
| v1.5 benchmark | `fa64ea392adb166c8a0511054c96396c5d397e6ea27ee70af3a07e6a381de3ea` | seven materialized Q2 challenges/15,435 memberships; six explicit blockers | label-blind, internal sensitivity surfaces |
| v1.5 WT reference | `502b70e765237b615ba3f622eabf284f7321e95d8d7234dc84b7c0fb7874807d` | reviewed human Q12809/KCNH2, 1,159 aa, zero mutants | sequence only; no construct/receptor |
| v1.5 MMP analysis | `9298e68e6ae2841be5ed466a86f693bed64e18ff84304899f4e494d4d1d2ace7` | 48,988 pairs; 43,824 training effects; 6,161 threshold-defined ≥1-pIC50 activity-cliff candidates | exploratory analysis, not feature store/mechanism |
| v1.5 QT/exposure prep | `33e81b5fbd0d1cc071c91282f83633104649f74e393be58fb3d2e630569f1995` | 95 structures, 143 trial contexts, 1,072 source-review rows | zero adjudicated exposure/margins/labels |
| v1.5 HPC preflight | `21bc07fe14f6cc9653630d923c857c9efcb8ab2dc4585babc53cab5c4199cc30` | 14-package/seven-stage inventory; only 2/7 gates pass | zero smoke/features/models/HPC jobs |

Every canonical release above preserves source/artifact hashes and the zero-training boundary. Superseded builds were retained under explicitly named directories; no purge was performed.

## Scientific results available now

### Candidate and lineage evidence

All 226 quantitative evaluation candidates are ChEMBL records. Target metadata separates 88 direct human KCNH2 relationships from 138 homologue relationships; none has explicit WT wording. Null variant IDs were correctly treated as “not annotated,” not WT confirmation. DOI metadata is present on 226 candidate rows but represents 17 unique DOIs; PubMed metadata is present on 224 rows but represents 16 unique PubMed IDs. This makes a bounded primary-source review feasible without inflating document counts. The automated lineage layer finds 617 conflict structures with cross-source mirror candidates and 33 with source-key reuse, but never equates heuristic similarity with experimental identity.

The current candidate split is not a sealed panel: 200 candidate observations share documents across model partitions and 68 share assays. Any accepted human decisions therefore require a complete structure/scaffold/assay/document/measurement-lineage refreeze.

### Harder benchmark surfaces

Seven Q2 challenges are now materialized without reading outcome columns: assay-group, document-group, broad document-year temporal, narrow exact-task temporal, automated-patch holdout, Morgan `<0.60`, and no-near-duplicate Morgan `<0.80`. Structure/scaffold and applicable assay/document/year groups are exclusive. These are internal stress tests, not external validation. The six honest blockers are Q1 cross-source power, manual-vs-automated power, a second Q2 source, an external low-similarity panel, real assay dates, and a prospective manual-patch gold panel.

### Matched-pair insight

Training-only Q1 matched pairs support a real analog-series analysis. Potency change has its largest signed rank relationships with logP (`rho=0.294`) and TPSA (`rho=-0.245`), followed by hydrogen-bond donors (`-0.156`) and acceptors (`-0.148`). The previously interesting standardized logP × TPSA aggregate interaction is weak within pairs (`rho=0.027`). That is useful negative evidence: it should remain a model hypothesis, not a claimed mechanism. Pairs are correlated through repeated structures/transformations and nominal p-values are not confirmatory.

### QT/exposure bridge

The clinical layer now distinguishes QTcF (136 memberships), QTcB (54), QTcI (1), and unresolved correction context (74). It identifies trial dose text for 77 structure-trials, route text for 76, DailyMed Cmax sections for 31 structures, human ChEMBL Cmax candidates for 13, and human FU/PPB candidates for 49; only nine structures have both candidate types. All 27 adjudication/margin fields remain null across 143 structure-trial rows. The interval contract requires compatible human exposure, analyte/matrix/context, and explicit human WT functional hERG potency before computing `IC50 / unbound Cmax`.

## Corrected future-compute contract

Feature contract `1.1.0` now distinguishes `structure_id` as a ligand parent join key from the composite feature-row key. It defines seven legal levels and level-specific nullable identities for ligand-only, protein-only, pose, and complex rows; protein embeddings are stored once rather than duplicated per molecule. WT protein input is frozen to Q12809. Twelve production input families remain distinct from assay covariates, exploratory MMP/alert analysis, and post-fit uncertainty/applicability outputs.

The machine runtime block now exposes 25 unresolved exact pins/schema/provider/receptor/physics fields. The local Python 3.13.7 and PyTorch 2.13.0 do not satisfy the future Python `3.11.*` and PyTorch `2.7.*` constraints. PyG, Transformers, OpenMM, Meeko, and Vina are missing. The required S0 smoke is therefore correctly blocked rather than described as runnable.

## Ordered continuation plan

1. Complete the 1,566-row primary-source adjudication packet; do not promote null-variant or homologue records to WT.
2. Obtain/freeze an independent prospective manual-patch WT panel and at least one independently governed Q2 source; then refreeze every candidate by structure, scaffold, assay, document, and lineage.
3. Resolve the 4,779 assay-protocol queue using source text for host, temperature, voltage, time, recording configuration, and platform—never imputation.
4. Adjudicate analyte, dose, route, schedule, population, matrix, Cmax, fu/PPB, metabolite, and compatible WT functional IC50 for the prioritized QT contexts; only then compute interval margins.
5. Freeze exact Python/package/model/provider revisions, lockfile and container digest, enumerate receptor/construct choices, and provision external/HPC storage.
6. Run the 10–100 valid-molecule smoke twice, including quarantined/synthetic negative controls, and require composite-key, determinism, provenance, failure, aggregation, and checkpoint-resume assertions.
7. Generate production features only after S0 passes. Start with the 2048-bit Morgan/RDKit baseline, then controlled graph/SMILES/protein/3D/physics increments.
8. Train on frozen train data, tune on validation, and evaluate once across the seven materialized and future external/prospective challenges. Compare all 15 competitor systems under identical data, preprocessing, budget, and coverage rules.
9. Claim model superiority only for a named frozen challenge whose paired interval excludes zero without materially worse calibration, recall, or applicability coverage.

## Verification state

The lead reviewed and reconciled every nonoverlapping workstream, corrected the QT manifest/WT-margin contract, MMP label-access/provenance/scaffold contract, feature hierarchy/composite keys, v1.5 preflight bindings, smoke dependencies, and shared YAML parity.

- All eight canonical production validators, the full contract validator, and the model-landscape validator pass.
- All eight canonical manifest self-hashes and 135 unique declared physical bindings independently replay; every declared Parquet row count and Arrow schema matches.
- Ruff and Ruff formatting across 238 files, mypy across 110 source files, `pip check`, and `git diff --check` pass.
- Pipeline regression: 752 passed, with one environment CPU-count warning.
- Editor-package regression: 53 passed.
- Independent focused regression: 25 passed.

## Independent final disposition

The read-only verifier independently replayed all eight canonical manifests and 135 unique declared bindings, including byte/hash checks and every declared Parquet row/schema contract. It then passed 82/82 semantic assertions covering WT/mutant exclusion, blank candidate decisions, Q2 eligibility and group leakage, train-only matched-pair effects and scaffold separation, null QT/exposure adjudication and hard margin blocks, and the absence of new v1.5 production-feature generation, substantive model fitting, or HPC execution in this continuation. All eight production validators plus the feature-contract and model-landscape validators passed. The focused regression passed 25 tests and the final full regressions passed 752 pipeline plus 53 editor-package tests. Ruff/format (238 files), mypy (110 source files), `pip check`, and `git diff --check` passed. The only diagnostic was the nonblocking joblib CPU-count warning. Final independent disposition: **PASS, with no release-blocking pre-HPC defect**.
