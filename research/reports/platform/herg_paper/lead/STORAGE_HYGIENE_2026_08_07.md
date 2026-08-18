# Storage hygiene record — 2026-08-07

## Decision rule

Raw source evidence, the accepted canonical build, the independent determinism
build, current versioned paper releases, and compact forensic explanations are
retained. Never-promoted failed payloads and superseded generated releases with
verified replacements are merely marked reclaimable; deletion requires a new,
explicit authorization because the current instruction is **no purge**.

## Reclaimable but retained under the no-purge instruction

| Artifact | Current size | Why it is reclaimable | Replacement/evidence |
|---|---:|---|---|
| `quarantine/full_chembl37_failed_attempt_2_numeric_dtype_collision` binary payload | 3,723,272 KiB | Never promoted; failed on a corrected integer/float serialization defect | Existing README; accepted canonical build plus regression tests |
| `quarantine/full_chembl37_failed_attempt_3_arrow_schema_unification` binary payload | 3,720,600 KiB | Never promoted; failed on a corrected all-null Arrow schema defect | Existing README with inventory and hashes; accepted canonical build plus regression tests |
| `v1_2_quality_tasks_superseded_entity_leak` | 35,132 KiB | Eight Q1 structure entities crossed partitions | Replaced by schema 1.1 release with entity-exclusive validation; old manifest SHA-256 `20c626ababb6d4843a3f2d7ddd9f9b37b794945c82e2760c79cbd3767c202fc4` |
| `quality_baselines_v1_superseded_nonstandard_json` | 1,468 KiB | Manifest contained non-standard JSON `NaN` values for constant-baseline correlations | Rebuilt with strict JSON and explicit zero correlation; old manifest SHA-256 `fe602b34fba81a361c595e29893fca83d831acb76998ee150b26497e87440945` |
| Later compact `v1_2_quality_tasks_superseded_*` releases | Small relative to source archives | Preserve corrected relation, clinical-QT, and split semantics during final integration | Canonical schema 1.2 validator and independent audit |
| `v1_2_modality_qt_superseded_qt_keyword_overmatch` | Small relative to source archives | Keyword-only QT screening overclassified 30 in-vitro hERG rows | Canonical schema 1.1 validator and curated phenotype registry |
| Superseded hERG current-analysis and quantitative-baseline releases | Small relative to source archives | Bound to pre-correction Q2 or modality semantics | Canonical content-bound rebuilds and validators |
| Superseded quality-task report | 4 KiB | Describes the defective split release | Replaced by current quality-task report |

The reclaimable total is approximately 7.13 GiB. No deletion was performed:
the earlier explicit no-purge instruction takes precedence until the owner
specifically authorizes these exact targets. All listed artifacts are excluded
from accepted data and current scientific use.

## Explicitly retained

- Approximately 74 GiB of unchanged raw evidence: required for provenance and reproducibility.
- The 1.1 GiB accepted canonical corpus.
- The 1.1 GiB determinism build: required evidence for the established reproducibility advantage.
- Small failed/interrupted build payloads totaling under 23 MiB: useful compact transaction-boundary evidence.
- Every current hERG hierarchy, model-ready, analysis, and lightweight-baseline release.
