# hERG first-paper final reconciliation — 2026-08-07

## Outcome

The non-HPC implementation of Dr. Wang's wild-type hERG direction is complete.
The accepted release is a provenance-preserving, assay-aware data and
experimental-design resource. It does **not** establish predictive superiority,
clinical replacement, or prospective validity.

## Workstream ledger and lead disposition

| Workstream | Nonoverlapping responsibility | Lead reconciliation |
|---|---|---|
| `recent_models_v2` | 2025–2026 model/method landscape and reproducibility/adapter priorities | Accepted after validator replay: 15 systems and 14 priorities; unresolved published details remain explicitly `NR` |
| `herg_master_data` | Observation/structure/assay/protocol/task/clinical/exclusion master compiler | Accepted at release 3 after relation-direction repair, protocol-index addition, source-contract separation, two rebuilds, and canonical validation |
| `herg_current_analysis` | Descriptive quality, disagreement, feature, replicate, split, and clinical-overlap analyses | Rebuilt by lead against final quality/modality inputs; accepted at schema 1.1 after automation-axis addition |
| `independent_verifier` | Read-only semantic, lineage, hash, schema, leakage, feature, protocol, and documentation audit | Final PASS on release 3; no release-blocking defects |
| `/root` lead | Scope, integration, conflict resolution, corrective patches, baselines, configuration, documentation, and final gates | Every workstream reviewed and reconciled; shared entry points changed only by the lead |

## Accepted content-bound releases

| Layer | Schema | Manifest self-hash |
|---|---|---|
| Wild-type scope | `platform-herg-wildtype-scope/1.0` | `8366be01525219a5efbc88627eccfcc0509aa25f7bf8c4eb276f7022dfe2290f` |
| Modality and QT | `platform-herg-modality-qt/1.1` | `ddab5c5c95f5b8030d0cbb8fffd0a24bdb3aa4e8edf7b1e06f86f9573a0721e6` |
| Quality tasks | `platform-herg-quality-tasks/1.2` | `6d67a13578605187ce92c8e76fd5244f1699b6221ccf84b33d51e7cb97efdca8` |
| Master dataset | `platform-herg-master-dataset/1.0` | `5495fc151b4fd2dfafeceb0c327596524e8c734a369df211134961ff901e2ec4` |
| Current analysis | `platform-herg-current-analysis/1.1` | `4aa2dd8ec7c842ce77338f331e63a061c53fbc5ca2099773a8e51a205bfe64de` |
| Q0 CPU floor | `platform-herg-paper-baseline/1.0` | `1384c2b738c195436e4fa9f5fead5d2e2a185b75f5621d44907f0ef2cb3d527d` |
| Q1/Q2 CPU floors | `platform-herg-quality-baselines/1.0` | `d7acf47d5e76649f46dfbf240c6bbd772af941f684b9862e7f59c0ebbe0553b7` |

## Final scientific reconciliation

- Scope: 407,698 admitted observations across 369,546 structures; 343,909 are
  confirmed WT, 63,789 remain visibly WT-or-unspecified, and all 258 explicit
  mutants are excluded.
- Tasks: Q0 has 339,373 eligible structures; Q1 has 23,186 observations across
  22,081 structures; Q2 retains 3,597 rows with 3,548 eligible across 2,776
  structures; C0 has 3,056 structures; C1 has 221 endpoints/95 structures and
  5,605 result-or-denominator records.
- Q2 exclusions: 49 exactly—seven clinical-QT phenotype rows, six IC50 rows
  missing native relation, 13 missing numeric endpoints, and 23 unusable
  structures. No clinical row is a direct or training hERG label.
- Measurement: 344,029 thallium-flux, 16,200 patch-clamp, 10,094
  binding-unspecified, 9,787 radioligand-binding, and seven curated clinical-QT
  phenotype observations. Thirty generic-QT-text overcalls were corrected.
- Protocols: 4,782 assay rows retain raw text and separately labeled curated
  source-contract evidence. The AID720551 contract supports U2OS/FluxOR/qHTS
  metadata, giving 343,910 U2OS-linked and 343,957 FluxOR-linked observations.
  Missing protocol fields remain unresolved rather than imputed.
- Features: all 369,546 structures have 23 complete, finite, deterministic 2D
  features. Unsupported pKa, microstates, docking poses, 3D conformers, channel
  states, membrane partitioning, and binding free energy were not fabricated.
- Leakage: zero cross-partition conflicts by structure, scaffold, canonical
  SMILES, or InChIKey; zero task-level structure/scaffold conflicts.
- Standardization: all exact, approximate, and censored IC50/pIC50 relations
  replay correctly. pIC50 relations remain native; concentration relations are
  inverted monotonically; six missing-relation IC50 rows remain unresolved.

## Analyses completed now

- Source and endpoint are nearly perfectly entangled (bias-corrected Cramér's
  V 0.998), motivating source-balanced and modality-aware evaluation.
- Exact-structure flux-versus-quantitative binary agreement is 34.5% across 591
  structures; ChEMBL-versus-quantitative agreement is 97.9% across 97.
- Automation comparisons are possible but limited: only 28 structures support
  automated-versus-manual binary comparison, so the result is descriptive and
  confounded rather than causal.
- There are 843 replicated exact-pIC50 structures; the largest within-structure
  range is 5.0 log units and is retained as an audit priority.
- Q0's strongest single prespecified descriptor association is logP (AUC 0.630;
  standardized mean difference 0.468). The largest marginal split shift is
  0.389 SD.
- CPU floors are intentionally modest: Q0 AP 0.00806 at 0.00362 prevalence;
  Q1 locked-test R² 0.184/Spearman 0.432; Q2 censored-regression locked-test R²
  0.094/Spearman 0.255. They validate plumbing, not superiority.

## Final verification

- 720 pipeline tests passed; 53 Menin-Edit tests passed.
- Ruff lint and formatting passed over 167 files; mypy passed over 87 source
  files; `git diff --check` passed.
- All production hERG validators, 25 independent focused tests, the 15-system
  model-landscape validator, and the paper-configuration claim gates passed.
- The independent verifier replayed every declared master input/artifact path,
  row count, byte count, SHA-256, Arrow schema, native column, relation/bound,
  task membership, split identity, feature, protocol evidence field, and live
  documentation count. Final result: PASS with no release-blocking defects.

## Storage and remaining boundaries

No purge was performed. Approximately 7.13 GiB of failed/superseded payloads is
documented as reclaimable but retained under the explicit no-purge instruction.
The hERGBoost downloadable development checksum remains unrecovered; exact
CToxPred2 labeled counts and MultiCTox headline metrics remain not reported in
the located primary/official evidence. The curated AID720551 protocol contract
is clearly labeled and hashed in the master artifact; its corroborating
source-profile report is narrative rather than a separately declared master
manifest input. These are disclosed limitations, not hidden assumptions.
