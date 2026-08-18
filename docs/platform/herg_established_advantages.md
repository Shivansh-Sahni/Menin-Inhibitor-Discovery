# hERG project advantages and claim rules

This document is the required claim preamble for every hERG paper, model card,
benchmark report, and presentation produced from this repository.

## Established now

The project is already superior to the audited comparator set as a **data and
experimental-design asset**, in the following specific, locally verified ways:

1. **Scale with honest grain:** 407,698 admitted observations across 369,546
   structures, rather than presenting repeated measurements as unique drugs.
2. **Wild-type governance:** 343,909 confirmed-WT observations are separated
   from 63,789 WT-or-unspecified observations, and 258 explicit mutants are
   physically excluded and auditable.
3. **Assay semantics:** endpoint, units, relation/censoring, modality,
   automation evidence, dose design, source, and native values are retained
   instead of being silently collapsed into one binary label.
4. **Quality-specific science:** Q0, Q1, and Q2 are distinct training targets;
   clinical-development and QT/QTc evidence are label-disabled context layers.
5. **Leakage control:** fixed whole-scaffold routing is reinforced by exact
   structure-entity exclusivity, closing the alternate-SMILES leakage defect
   found during lead integration.
6. **Auditability:** deterministic schemas, stable identifiers, input hashes,
   exclusion ledgers, raw pointers, validators, and versioned releases allow a
   claim to be traced back to its source evidence.
7. **Direct response to Dr. Wang's questions:** the data can test evidence
   quality, automatic versus manual/unknown measurement, cross-method
   disagreement, and later exposure-aware hERG-to-QT translation.

These advantages are materially stronger than merely training another graph or
SMILES model on a pooled public label. They are the first-paper foundation and
should be emphasized even if a later architecture is deliberately simple.

## Not established yet

The project has not yet shown higher prospective AUC, average precision, MCC,
calibration, new-scaffold transfer, laboratory replacement, clinical QT/TdP
prediction, or causal mechanism than existing models. “State of the art” and
unqualified “superior hERG predictor” are therefore prohibited.

The correct wording is:

> We established a larger, wild-type-governed, assay-aware, quality-tiered,
> provenance-preserving hERG evidence and evaluation resource. We will test
> predictive superiority through matched frozen benchmarks; we do not infer it
> from dataset size or incomparable published headline metrics.

## Rule for a future predictive-superiority claim

A claim must name one frozen challenge and comparator. It requires identical
structures, labels, preprocessing, partitions, tuning budget, test access,
coverage accounting, and primary metrics; repeated seeds; paired uncertainty;
calibration and applicability-domain evidence; and no material failure in the
scientifically important subgroup. A win on Q0 does not imply a win on Q1, Q2,
manual patch clamp, new chemistry, or QT translation.
