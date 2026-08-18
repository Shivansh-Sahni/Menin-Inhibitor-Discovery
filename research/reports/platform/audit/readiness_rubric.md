# Independent pretraining-readiness rubric

Final assessment snapshot: 2026-08-04. Mechanical artifact readiness is
separate from scientific task readiness, release readiness, and authorization
to train.

This rubric is a post-gate reconciliation of the immutable machine report and
does not alter that report's bytes.

## Verdict

**Mechanical artifact verification: PASS.**

**Substantive large-model training: NOT READY and NOT AUTHORIZED.**

The final verifier reports artifact integrity and source rebinding true, while
scientific task claim readiness, substantive-training readiness, and training
authorization are false. A numeric score cannot waive a failed hard gate.

## Hard gates

| Gate | Pass condition | Final status |
|---|---|---|
| G1 Scope truthfulness | Legacy, synthetic, bounded diagnostic, and real platform results are separated | `passed` |
| G2 Rights | Repository/content/model licenses selected; every training row permitted | `blocked_human`: no repository license and conditional redistribution/model terms unresolved |
| G3 Confidentiality/release | Public dependency trace plus staged disclosure, secrets/PII/private-correspondence, and storage review | `partial`: generated platform integrity passes; staged release review and large-artifact disposition remain |
| G4 Provenance | Immutable source/version/query/lineage for every admitted row | `passed_for_chembl_only`; external normalized candidates have zero canonical admission |
| G5 Ontology | Canonical entity/evidence vocabulary; endpoint separation; predictions excluded from labels | `passed_mechanical` |
| G6 Scale/coverage | Large representative multisource corpus including the intended scientific evidence layers | `failed_scope`: large ChEMBL corpus exists, but no admitted external clinical/PK/QT/outcome layer |
| G7 QC | Identity, units, censoring, duplicates, missingness, conflicts, and count reconciliation pass | `passed_for_frozen_corpus` |
| G8 Leakage | Intended-use exact and near ligand/protein/source/assay/time thresholds accepted | `partial`: exhaustive exact checks pass; capped near-similarity and human threshold acceptance remain |
| G9 Evaluation | Adequate baselines, uncertainty, calibration, difficult splits, sensitivities, and sealed/external evaluation | `partial`: real capped diagnostics only; no accepted final or external model evaluation |
| G10 Reproducibility | Exact environment, tests/lint/types, deterministic rebuild, clean release and independent reconstruction | `partial`: mechanical suite passes; dirty migration, missing license, and independent clean-clone run remain |
| G11 External validity | Compatible independent lockbox and prospective evidence for broad claims | `blocked_external` |
| G12 Compute/governance | Approved compute, monitoring, checkpoint/resume, budget, retention, and human authorization | `blocked_human_hardware` |

## Evidence-weighted rubric

Scores are 0–4: absent, specified, fixture/interface, real locally evidenced,
or independently reproduced and accepted. Points equal weight×score/4. This is
diagnostic only; the conjunctive hard-gate verdict controls.

| Dimension | Weight | Score | Evidence | Points |
|---|---:|---:|---|---:|
| Governance, rights, confidentiality | 15 | 1 | Strong local containment and explicit blockers; licenses and staged approval unresolved | 3.75 |
| Source versioning, provenance, citation | 15 | 3 | Five raw bundles, normalized evidence, and ChEMBL row lineage verified; external canonical admission unresolved | 11.25 |
| Entity/endpoint/evidence ontology | 12 | 4 | Machine schema, data dictionary, ontology, task signatures, and fail-closed tests accepted | 12.00 |
| Real scale, diversity, clinical layers | 14 | 2 | 3,938,372 ChEMBL observations; still single-source admitted and no genuine clinical/PK/QT outcome corpus | 7.00 |
| QC, deduplication, missingness, bias | 10 | 4 | Bound QC plus byte-identical real statistical census and lineage/count closure | 10.00 |
| Splits and leakage prevention | 12 | 3 | Directly source-reverified A/B, exhaustive exact checks; near-similarity remains capped/sample-based | 9.00 |
| Representations and training interface | 8 | 3 | Twenty-two tasks integrated, 524 bound components, unopened test lockboxes; no expensive embeddings/training | 6.00 |
| Baselines, metrics, statistics, robustness | 8 | 3 | Real statistics and 11 capped development diagnostics; full model/ablation/external evaluation absent | 6.00 |
| Software reproducibility and release | 6 | 3 | Final gate, tests, Ruff, format, mypy, pip check, determinism and core audit pass; clean release remains | 4.50 |
| **Total** | **100** |  | Hard gates still control | **69.5/100** |

The score rose because the real canonical, statistical, split, and corpus
artifacts closed successfully. It does not compensate for rights, clinical
coverage, intended-use, external validity, checkpoint/overlap, compute, or
release failures.

## Readiness levels

| Level | Definition | Final result |
|---|---|---|
| L0 Design-ready | Ontology, source plan, risks, claim boundaries, and acceptance criteria exist | **Yes** |
| L1 Interface-ready | Deterministic schema/features/splits/corpus loaders and bounded diagnostics pass | **Yes, for the frozen ChEMBL-only snapshot** |
| L2 Public-release-ready | Rights, staged disclosure, clean commit/clone, storage, and notices pass | **No** |
| L3 Broad corpus-ready | Rights-cleared representative multisource/clinical corpus with accepted leakage and bias audit | **No** |
| L4 Substantive-pretraining-ready | L3 plus checkpoint, objective, compute, monitoring, resume, and authorization | **No** |
| L5 Paper/product claim-ready | Independent external/prospective validation and accepted claim ledger | **No** |

## Exact blockers to a changed verdict

1. Resolve rights and canonical admission for external evidence; complete
   molecule/product/intervention linkage, ambiguity quarantine, duplicate and
   conflict review, and genuine clinical, PK, QT/QTc, and outcome extraction.
2. Complete staged release, secrets/PII/private-correspondence, redistribution,
   large-artifact storage, clean-commit, and independent clean-clone reviews.
3. Select repository/content/model licenses and approve conditional source and
   model-weight terms.
4. Freeze one intended task and claim, minimum support, label policy, and exact
   ligand/protein/source/assay/temporal leakage thresholds; replace or formally
   accept the capped near-similarity audit.
5. Freeze model checkpoint/revision/weight and tokenizer hashes, training
   cutoff, license, and corpus-overlap evidence. Identify the meeting-referenced
   model reportedly trained on roughly 100,000 structures.
6. Approve accelerator/HPC allocation, measured budget/throughput, monitoring,
   checkpoint/resume, failure recovery, retention, and responsible-use plan.
7. Obtain independent external and appropriately designed prospective
   validation before broad translational, safety, or product claims.
8. Only after 1–7: run a capped dry run on approved hardware, re-run the final
   verifier on the newly frozen bundle, and obtain explicit human authorization.

## Explicit non-claims

- No substantive large model was trained.
- No production HPC simulation campaign was run or verified.
- No platform clinical cardiotoxicity, QT, PK, binding-free-energy, or efficacy
  model was externally validated.
- hERG assay evidence is not QT prolongation, torsades, or patient safety.
- Capped diagnostics and loader smoke are engineering evidence, not final model
  performance.
