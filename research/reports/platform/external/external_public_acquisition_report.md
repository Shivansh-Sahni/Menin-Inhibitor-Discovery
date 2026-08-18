# External public acquisition freeze report

Status: **PASS — five immutable source bundles are recursively reconciled.**

This workstream acquired and froze public evidence only. It admitted **0 canonical rows**, admitted **0 model labels**, performed no cross-source normalization, and started no substantive training. In particular, missing registry records or missing posted results are not negative safety, activity, or efficacy observations.

## Reconciled scope

- 5 source bundles
- 404 declared source artifacts totaling 18,913,867,042 bytes
- 853 exactly bound bundle artifacts totaling 18,994,847,039 bytes
- 10 streamed archive member inventories
- Every declared file was rehashed from disk; exact recursive membership, sidecars, manifest self-identities, archive inventories, and zero-admission boundaries passed.

## Frozen identities

The internal digest is the non-circular canonical-JSON identity stored in the manifest. The file digest is the SHA-256 of the physical indented JSON file.

| Source | Internal manifest SHA-256 | Physical file SHA-256 | Source artifacts / bytes | Bundle artifacts / bytes | Archive inventories |
| --- | --- | --- | ---: | ---: | ---: |
| BindingDB curated 202608 | `5a7425f680e6820ea665212e051bd11a2ece2e03d916534456640bdd67297406` | `f430fdf4f5740708f5ac77089b9c0091604010338020c8ed83f80ee541d5c189` | 11 / 44,266,829 | 25 / 44,280,360 | 3 |
| ClinicalTrials.gov API v2 | `5b33bcc2ae99fd0bae32e79a8ec4706555436441d7439476ddfdccafbda2d48c` | `42707517b8afd0d74b8c3ad1abcd4457fec273596ac7001d710ba37a80d9a6ea` | 231 / 434,571,993 | 488 / 469,081,796 | 0 |
| DailyMed human prescription SPL | `d29a97bb83d39a6117c9c8d22a577eba8e9adf69b35291d93a02e3a4541fcb65` | `f2a2c2fca158bfd524c3d1d8acd4edb7bcf8c2db3a4212bc6595cf7cef359886` | 7 / 17,767,301,645 | 26 / 17,788,919,602 | 6 |
| Drugs@FDA bulk | `d995f3309c5665fea9f6b4d74d64989b0ce20f3266fb1b802a3695a21c6c74dd` | `406f87317a0f16b7c99a38dd5ec34aa86f0174f4158b6c97a6a17413528bcab8` | 3 / 6,220,407 | 7 / 6,228,028 | 1 |
| UniProtKB targeted 2026_02 | `d4000c69e3c51ff1fbd7b68b0c32e78274fdbbdac40b6b1938f462fef816017a` | `781072c1959bf41dcfeccf3ebf6a66bbfc8ab3fba53624fbd551ac52d373d3cf` | 152 / 661,506,168 | 307 / 686,337,253 | 0 |

## Scientific acquisition audits

BindingDB preserves 93,712 physical measurement rows without admission. Origin isolation found 93,023 BindingDB literature-curated candidates, 429 ChEMBL mirror rows excluded from later independent-evidence consideration, and 260 Taylor Research Group rows quarantined for origin/rights review. There are 93,522 unique Reactant Set IDs, 190 duplicate-ID rows retained, and no missing IDs. Nonblank endpoints are Ki 25,790; IC50 63,525; Kd 2,650; EC50 4,230; kon 15; and koff 29.

ClinicalTrials.gov freezes the version identity `apiVersion=2.0.5`, `dataTimestamp=2026-08-04T09:00:05`. The alias-independent all-DRUG cohort contains 210,644 unique NCT records over 211 token-chained pages. A separate high-recall cardiac-safety heuristic cohort contains 3,879 unique NCT records over 4 pages, including 1,343 records with posted outcome modules and 1,343 with posted adverse-event modules. Its text membership is explicitly unreviewed and may contain false-positive, context-ambiguous, co-intervention, and unmapped-drug records.

DailyMed freezes all six current human-prescription release archives: 17,767,198,122 transferred archive bytes and 54,672 CRC- and stream-verified members. XML section extraction, SETID history reconstruction, product/molecule mapping, and label interpretation remain deliberately unstarted.

Drugs@FDA preserves all 12 published relational tables and 959,447 source rows. One malformed-width source row, 16 blank composite primary-key rows, and 6,906 missing foreign-key references are retained and reported; there are no duplicate primary-key rows. No row was repaired or coerced.

UniProtKB queried 14,983 requested accessions against release 2026_02. It resolved 14,980, quarantined 3 ambiguous multi-maps plus 46 non-UniProt-syntax identifiers, and found no unresolved missing accession. Of 14,983 returned primary entries, 13,976 are sequence-ready with source MD5/length verification. The 1,007 Inactive entries without sequence/name are retained but quarantined from identity or sequence features. Duplicate-sequence auditing found 955 sequence-hash groups covering 1,958 entries.

## Known human-review boundaries

- BindingDB redistribution and origin-specific rights require institutional review before admission.
- ClinicalTrials.gov OpenAPI endpoint failure evidence is preserved; the captured terms page is an SPA shell and its human-readable terms/disclaimer still require review.
- DailyMed terms/copyright/API handling, attribution, and SETID/version semantics require review before derived evidence admission.
- UniProt CC BY 4.0 attribution and third-party disclaimer obligations require review.
- Drugs@FDA attribution, disclaimers, and any document-specific third-party restrictions require review.

## Verification and code gates

Generator code SHA-256: `6b337c3d27df113daeb4baa4f541d82aacbd12233530f13df16655c5d40bfa04`

Focused test file SHA-256: `ed6f4702b21df7bad15d49827b53bc4334b2a7292c7473d0480ac499b76a4587`

```text
.venv/bin/pytest -q pipeline/tests/test_platform_external_acquisition.py
38 passed

.venv/bin/ruff check pipeline/src/menin_discovery/platform_external_acquisition.py pipeline/tests/test_platform_external_acquisition.py
All checks passed!

.venv/bin/mypy pipeline/src/menin_discovery/platform_external_acquisition.py
Success: no issues found in 1 source file
```

Reproduce all five recursive source verifications from the repository root:

```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path
from menin_discovery.platform_external_acquisition import verify_source_acquisition_manifest

paths = [
    Path("research/data/platform/raw/external_public/bindingdb_curated_202608/bindingdb_curated_202608_manifest.json"),
    Path("research/data/platform/raw/external_public/clinicaltrials_gov_v2/clinicaltrials_gov_v2_manifest.json"),
    Path("research/data/platform/raw/external_public/dailymed_spl_v2_human_rx/dailymed_spl_v2_human_rx_manifest.json"),
    Path("research/data/platform/raw/external_public/drugs_at_fda_bulk/drugs_at_fda_bulk_manifest.json"),
    Path("research/data/platform/raw/external_public/uniprotkb_targeted_2026_02/uniprotkb_targeted_2026_02_manifest.json"),
]
for path in paths:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    print(verify_source_acquisition_manifest(path.parent, manifest))
PY
```

The machine-readable summary is `external_public_acquisition_summary.json`; the complete verification ledger is `external_public_acquisition_verification.json`.
