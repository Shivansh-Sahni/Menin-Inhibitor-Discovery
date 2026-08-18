# ClinicalTrials.gov cardiac-cohort result candidate analysis

**Source snapshot:** ClinicalTrials.gov API 2.0.5, data timestamp `2026-08-04T09:00:05`  
**Scope:** four manifest-bound raw pages in the frozen cardiac-safety heuristic cohort  
**Disposition:** candidate inventory only; zero canonical observations, zero model labels, and no training

## Outcome

The four source pages were reconciled byte-for-byte to the frozen acquisition manifest, parsed independently, normalized into six deterministic gzip-compressed CSV inventories, verified, and rebuilt twice from scratch. Replay A, replay B, and the retained bundle were byte-identical for every file.

| Inventory | Rows | SHA-256 |
|---|---:|---|
| Studies | 3,879 | `f8f45a4afcdd43bb1564e62f43b6716e6cbac6f9200f7e7d8d7a6a7d0bb10239` |
| Interventions | 9,831 | `97dc8a7633aef8bcac254f160d8004790617be4a955c9c60d3eca7b15fa30cb5` |
| Reported outcome-measure groups | 97,745 | `a40a738b59cb1b085abc0ce57bf5d4e3c1e6209c5f2c9007d2d3241feba7cd20` |
| Posted outcome measures | 31,810 | `3810030bd0a468dfcc7c45b54d4c17a4edb4913ba9e45fffc10159f0abedbcb6` |
| Posted adverse-event groups | 5,772 | `71900a948d1feba4a8d21e7763d12a0c34130af236436beb5dfcbd2773cb967f` |
| QT/QTc and PK endpoint candidates | 10,144 | `a7181ded0a4774adc8b886cf46bd00ecb438fa297b23348d06d4395494c82d02` |

The compressed inventory payload is 16,081,158 bytes (about 16 MB), versus roughly 258 MB for an equivalent plain-CSV build. Compression is deterministic (`mtime=0`, blank gzip filename), so it does not weaken replayability.

## Source and code binding

- Acquisition manifest physical SHA-256: `42707517b8afd0d74b8c3ad1abcd4457fec273596ac7001d710ba37a80d9a6ea`.
- Acquisition manifest internal SHA-256: `5b33bcc2ae99fd0bae32e79a8ec4706555436441d7439476ddfdccafbda2d48c`.
- Concatenated four-page SHA-256: `b871e04eed762244d580a11df914633add6fd3cc19b5921eedddc446af6509aa`.
- Page SHA-256 values, in order: `f21ee560...90bc`, `f4332ee7...5771`, `f18a821f...d409`, `9dd80902...05c` (full values are retained in the candidate manifest and every row carries its own page SHA-256).
- Parser source SHA-256 `eaac80f156291712d5660d1bde79ed6bd426d4e939c9fb70d65a7520c3f1d7c7` and byte length are frozen in the retained candidate manifest. The verifier recomputes both rather than trusting the declaration.
- Every row preserves NCT ID, source page path/hash/index, study index, exact raw JSON Pointer, source API/data version, unknown or missing fields, and an explicit candidate-only status.

## Candidate findings

Of 3,879 studies, 1,343 have posted result modules in this snapshot and 2,536 do not. Absence is always encoded as unknown—not as a negative outcome. Conservative phrase rules identified 10,144 target candidates across 1,280 studies:

| Classification | Candidates |
|---|---:|
| PK explicit metric candidate | 6,788 |
| PK quantitative concentration candidate | 443 |
| PK context requiring review | 556 |
| PK participant/safety count, not a genuine PK metric | 27 |
| QT/QTc event or threshold candidate | 1,617 |
| QT/QTc interval-measure candidate | 644 |
| QT/QTc context requiring review | 69 |

Domain totals are 7,814 PK candidates and 2,330 QT/QTc candidates. Rule-based candidate status marks 7,231 PK and 2,261 QT/QTc records as explicit quantitative/event candidates; this is **not confirmation of scientific validity or canonical admissibility**. All 10,144 rows remain manual-review candidates. There are 9,944 outcome-measure candidates, 47 serious adverse-event-term candidates, and 153 other adverse-event-term candidates. At study level, 792 studies contain both domains, 450 only QT/QTc, and 38 only PK.

Candidate rows retain 84,835 reported measurement/statistic records plus their units, timeframes, group IDs, and denominators. Zero adverse-event counts are preserved as reported group-level values and explicitly prohibited from becoming study-level negative labels.

## Data-quality findings

- Among 31,810 posted outcomes, 1,641 omit a unit, 377 have no reported denominator records, 1,707 have no measurement records, and three omit a timeframe.
- Sixty-three studies have posted result modules but no target endpoint detected by the conservative rules. This means “not identified,” never a negative QT or PK result.
- All 1,343 studies with result modules have posted outcome and adverse-event modules in the selected snapshot; the adverse-event inventory contains 17 groups missing a serious-at-risk count and 22 missing an other-at-risk count.
- The selected API fields did not include protocol arm-group records. Therefore the 97,745 arm/group rows are exact outcome-measure group instances, not reconstructed protocol arms. No outcome-group, adverse-event-group, arm, or intervention linkage is inferred across modules.
- There are 9,831 reported interventions and 5,716 distinct case-folded names. The inventory preserves raw registry text only. The cohort requires at least one drug intervention, but its returned study records also list 691 non-`DRUG` co-interventions. No intervention name is treated as an exact chemical identity.

## Verification and replay

Commands used:

```bash
.venv/bin/python -m pytest -q pipeline/tests/test_platform_clinical_results.py
PYTHONPATH=pipeline/src .venv/bin/python -m menin_discovery.platform_clinical_results build \
  --source-root research/data/platform/raw/external_public/clinicaltrials_gov_v2 \
  --output-root research/data/platform/interim/clinical_results_candidates
PYTHONPATH=pipeline/src .venv/bin/python -m menin_discovery.platform_clinical_results verify \
  --source-root research/data/platform/raw/external_public/clinicaltrials_gov_v2 \
  --output-root research/data/platform/interim/clinical_results_candidates \
  --report research/reports/platform/clinical_results_analysis/verification_report.json
```

Focused tests cover conservative QT ambiguity, exclusion of creatinine clearance from drug-PK classification, absent-result semantics, reported zero adverse-event counts, denominator preservation, source tampering, output tampering, symlink rejection, verification-report hash separation, and deterministic replay. Result: **6 passed**.

For the required real-data replay, two fresh output directories were built independently and compared recursively against each other and against the retained output. All comparisons were empty (no byte differences), and the current candidate-manifest internal hash is `82370be84f3ad8192bb7d12c1d70fde38d2826fb92c81eaff7f55d330585503c` (physical file SHA-256 `81295086d9838f1cb58c201089d2570169c14b91613d0a4aaf9788e0ed560a5a`).

## Hard limitations and next gate

This cohort was selected by heuristic cardiac-safety text and contains false positives and contextual mentions. Exact evidence phrases are retained so every classification can be audited. Result values are aggregate registry reports, not verified individual-level data. No source text was interpreted beyond explicit phrases, and no unreported result was inferred.

Before any canonical admission, a domain reviewer must inspect candidate evidence, endpoint semantics, units, correction method, timepoint, population, denominator, group linkage, intervention identity, and source history. Conflicts and missing fields must remain unresolved rather than imputed.
