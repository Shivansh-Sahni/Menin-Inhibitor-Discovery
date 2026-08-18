# Final 25-section pretraining-readiness status matrix

Assessment date: 2026-08-04. This matrix reconciles the governing 25-section
specification to the final physical artifacts. The integration gate passed for
artifact integrity and source rebinding. It did **not** make a scientific task
claim-ready or authorize substantive large-model training.

This is a transparent post-gate reconciliation document. It does not mutate or
replace the immutable machine final report.

## Status language

- `executed_real`: executed against authentic repository or real-corpus bytes.
- `executed_real_bounded`: executed on the real corpus, but deliberately capped
  or descriptive and therefore not final model-performance evidence.
- `executed_interface`: executable and tested, but a model/checkpoint/hardware or
  human decision still prevents a final scientific execution.
- `documented_plan`: prespecified later work; no result is claimed.
- `human_or_external_blocked`: cannot responsibly be closed by code or the
  currently admitted data.
- `substantive_training_deferred`: intentionally stops before prohibited
  large-model training.

Legacy Menin/hERG outputs remain prior target-specific evidence. They are not
used to inflate the expanded platform results below.

## Governing crosswalk

| Section | Final audited status | Physical evidence and limitation |
|---:|---|---|
| 1. Full project audit | `executed_real` | Repository, migration, access-class, discrepancy, risk, changed-file, and workstream inventories were reviewed. The mechanical final report passed; the dirty migration and staged release inventory still require human release review. |
| 2. Data collection and provenance | `executed_real` for frozen acquisition; canonical admission remains separate | Five external bundles pass recursive verification: 404 declared source artifacts/18,913,867,042 bytes, 853 bundle artifacts/18,994,847,039 bytes, and ten archive inventories. Nine normalized artifacts total 60,377,234 bytes and 464,123 Parquet rows. They intentionally admit zero canonical observations and zero labels. |
| 3. Data schema and dictionary | `executed_real` | The canonical schema, ontology, dictionary, explicit Arrow fingerprints, allowed values, roles, units, lineage, and fail-closed path/schema rules are bound to the accepted build. |
| 4. Data integrity and QC | `executed_real` | Canonical A QC SHA `d481c8…` and B QC SHA `acc14b…` pass against their exact manifests; 260 canonical components and 1,154,513,167 bytes are content-equivalent. QC records zero issue records and zero error-finding rows for the promoted build. |
| 5. Deduplication and entity resolution | `executed_real` within admitted evidence | The build contains 1,304,011 molecules and aliases, 430,776 assays, 9,411 proteins, 9,071 constructs, 3,938,372 observations, and matching lineage. Repeated/conflicting measurements are retained and reported. Cross-source canonical identity/admission remains blocked rather than guessed. |
| 6. Normalization and standardization | `executed_real` | The ChEMBL 37 transaction binds 162 Parquets and 31,402,047 full-plus-overlapping specialized/development rows; external normalization is deterministic and preserves source dispositions. Original values, conversions, relations, and derived-label lineage remain distinct. |
| 7. Missing-data analysis | `executed_real_bounded` | Real `missingness.csv` (281 rows), `attrition.csv` (207), exclusions, compatible strata, and source/task/context denominators were generated twice byte-identically. No target imputation was performed. |
| 8. Exploratory analysis | `executed_real_bounded` | Eighteen real statistical artifacts per build cover composition, concentration, coverage, temporal/stage strata, conflicts, hERG thresholds, free-energy sensitivity, associations, and two SVGs. This is a single-source descriptive census, not causal or clinical inference. |
| 9. Target construction and label validation | `executed_real` | Twenty-eight task datasets are registered: 23 default and five derived-sensitivity tasks. Default candidates 2,548,252 yield 2,548,198 included rows; sensitivity candidates 67,848 yield 67,839. Observed, censored, categorical, and derived labels remain separated; all exclusions are reasoned missing-sequence cases. |
| 10. Preprocessing pipeline | `executed_real` | Raw-to-interim, external normalization, canonical A/B, bound QC, and content-equivalence verification pass. A/B manifests are `1ace39…` and `85bcbe…`; determinism report SHA is `ab5fc4…`. Failed attempts remain quarantined and were never promoted. |
| 11. Domain-specific features | `executed_real_bounded` | Real label-blind feature projections and modality support were generated for all 28 tasks. Scaffold extraction failures are explicit exact-SMILES proxies; 214 exception-proxy rows occur across 12 overlapping task views and make those scaffold candidates inapplicable. Broader expensive feature families remain future model choices. |
| 12. Representations and embeddings | `executed_real_bounded`; no expensive embedding generation | Corpus readiness integrates 22 tasks and support-skips six, binds 524 components, performs train/validation-only loader and diagnostic interfaces, and never opens or hashes test lockboxes. No large embedding corpus or substantive training is claimed. |
| 13. Splits and leakage | `executed_real_bounded` | Split A/B are byte-identical: 480 files, 225 directories, 564,604,068 bytes, inventory SHA `0307c2…`. Both were directly source-reverified. Across 168 decisions, 85 strategies materialized and 83 skipped. Exact overlap is exhaustive; near-similarity evidence is deterministic capped sampling and remains explicitly not claim-ready. |
| 14. Imbalance and sampling | `executed_real_bounded` | Real support/prevalence and fixed-seed feasibility are recorded. Twenty-two molecule-grouped tasks materialize; six are support-skipped without seed search. Weighting/sampling alternatives are registered, but no final task/model strategy has been scientifically selected. |
| 15. Statistical analysis | `executed_real_bounded` | Statistical A/B are source-reverified and byte-identical: 19 files, 1,374,306 bytes, inventory SHA `70702e…`. The only inferential panel with adequate cells is QT decade×evidence stage; it is multiplicity-adjusted and explicitly descriptive. |
| 16. Baselines and sanity checks | `executed_real_bounded` | Eleven capped train/validation diagnostics completed and 11 were explicitly skipped; six tasks were preflight-skipped. Caps are 10,000 train and 2,500 validation rows. Censored tasks are not midpoint-imputed, and test labels are not used. These are diagnostics, not accepted platform performance. |
| 17. Fine-tuning preparation | `executed_interface` + `substantive_training_deferred` | Candidate, feature, metric, robustness, loader, collator, loss, and resume contracts exist. Exact checkpoint/revision/hash/license/cutoff/overlap, compute, and human authorization remain unresolved. |
| 18. Metrics and evaluation | `executed_interface` with bounded real diagnostics | Metric contracts and tests pass, and completed diagnostics emit development metrics. No sealed-test, external, prospective, calibration-under-shift, or headline model evaluation is accepted. |
| 19. Robustness and sensitivity | `documented_plan` with partial real execution | Fixed split strategies, exact-only/censored separation, derived free-energy sensitivity, and skip states are executed. The full representation/model/threshold/identity ablation matrix awaits a frozen task and model. |
| 20. Bias, fairness, and subgroups | `executed_real_bounded` descriptively; post-model fairness blocked | Attrition, missingness, target/source concentration, stage/temporal composition, compatible strata, and subgroup support are real. No model fairness or equitable clinical-performance claim is made. |
| 21. Error analysis | `executed_real_bounded` for completed diagnostics | Eleven completed diagnostics bind predictions/errors and lineage without test use; skipped tasks have explicit reasons. Broader error taxonomy, blinded review, and external-error analysis await an accepted model. |
| 22. Reproducibility and software quality | `executed_real` mechanically; release hygiene blocked | Final gate SHA `f27ceb…` passes; 2,117 generated regular files have zero inode aliases. Pipeline tests: 579 passed; Menin-Edit: 53 passed; Ruff and 177-file format checks pass; mypy passes 80 source files; pip check passes; the 53-pin core lock has zero known vulnerabilities. A dirty migration, missing repository license, staged disclosure, and clean-clone reproduction remain release blockers. |
| 23. Documentation and reporting | `executed_real` | Platform methods, ontology, claim boundaries, sources/models, risk, status matrix, reproduction contract, and independent reconciliation now agree with the physical final report and preserve actual-versus-deferred boundaries. |
| 24. Output management | `executed_real` | Raw, interim, canonical, QC, statistical, split, corpus, model, report, and quarantine namespaces are separated. Transactional publication, exact inventories, portable paths, no symlinks/special files, and no hardlink aliases passed. Redistribution/storage approval remains human work. |
| 25. Final pretraining-readiness review | mechanical `executed_real`; substantive readiness `human_or_external_blocked` | `platform_final_artifact_verification.json` is 20,573 bytes, SHA `f27ceb…`, scans 1,090 JSON documents, and reports artifact/source integrity true while task claim readiness, training readiness, and authorization are false. Eight human/external blockers remain. |

## Final decision

The repository is mechanically artifact-ready for the frozen ChEMBL-only
platform snapshot and has a working, source-bound pretraining interface. It is
**NOT READY for substantive large-model training**. The blockers are: external
canonical admission and real clinical/PK/QT outcomes; release hygiene/storage;
repository and redistribution licenses; a frozen intended task and leakage
thresholds; checkpoint/license/corpus-overlap review; authorized compute and
operations; independent external/prospective validation; and identification of
the meeting-referenced approximately 100,000-structure model.

No substantive large-model pretraining or fine-tuning was initiated.
