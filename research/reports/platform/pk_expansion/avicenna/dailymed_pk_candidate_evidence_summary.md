# DailyMed quantitative PK candidate evidence inventory

## Outcome

- Stream-scanned **54,672/54,672** frozen DailyMed human-prescription SPL packages across all six release archives; **54,672 XML documents parsed and 0 errors**.
- Identified **41,960 candidate document versions**, containing **43,162 quantitative Pharmacokinetics/Clinical Pharmacology candidate sections** and **26,610 tables**.
- The version-aware latest-available view contains **40,247 SETIDs** and **41,395 candidate sections**. It is only the latest version present in this frozen corpus—not a marketed/current-label assertion.
- Exact SPL approval-number joins linked **19,652 candidate document versions** to Drugs@FDA application/product metadata.
- Every candidate retains outer archive/member/CRC, inner XML member/SHA-256, document ID, SETID, version, effective time, author, product/NDC, approval identifiers, section ID/code/title/hash, bounded evidence spans, table hashes, and Drugs@FDA join evidence.
- **Canonical PK rows: 0. Training labels: 0.** No number was converted into CL, t1/2, AUC, Cmax, Tmax, Vd, or F.

## Context tiers

- **Tier A — 29,664 sections:** human/population, route, dose, matrix, and unit-like terms all detected in the same section. These are strong review candidates, but term presence does not prove every term applies to every value.
- **Tier B — 4,324 sections:** endpoint, units, matrix, human context, and either route or dose detected.
- **Tier C — 9,174 sections:** quantitative PK endpoint candidate with a major context gap.

Endpoint hits are nonexclusive: half-life 39,116; Cmax 32,235; clearance 30,445; AUC 27,863; bioavailability 24,959; volume of distribution 20,665; Tmax 14,191.

## Verification

- All six archive byte sizes, SHA-256 hashes, and member counts match the frozen DailyMed manifest.
- Complete reads validated every outer member CRC and every selected inner XML CRC.
- All document locators, 43,162 candidate IDs, and 26,610 table keys are unique.
- Every section links to a parsed candidate document; every table links to a candidate section; declared and observed table counts agree.
- All table XML and normalized-text hashes are present. All validation checks passed.

### Reproducible implementation

- The exact executed scanner and merge/validator programs are preserved byte-for-byte under `scripts/`; their SHA-256 hashes are bound in the JSON report, validation, and report manifest.
- `scripts/dailymed_pk_candidate_cli.py` exposes bounded scanning and read-only validation replay with explicit input/output arguments and nonzero fail-closed exits.
- A 10-package smoke scan passed with zero parse errors. A full validation-only replay rehashed all six declared evidence artifacts, reparsed every JSONL row, reproduced the core counts and links, and changed zero raw evidence rows.
- The exact historical programs retain run timestamps and elapsed-time metadata, so those metadata fields are not byte-stable; evidence rows, candidate IDs, hashes, ordering, counts, and the new validation replay are deterministic.

## Honest limitations

- This is a deterministic evidence index, not a scientific measurement extraction. Regex selection can miss unusual language and can retain unrelated numbers inside an otherwise relevant section.
- Tier A is machine-detected and unverified; genuine context-complete PK remains **zero** until each value is linked to its endpoint, unit, species/population, route, dose, matrix, formulation, time basis, and study evidence.
- Only 10,976 candidate versions exposed active ingredients through the strict ACTIM parser. Product names/codes are nearly complete, but ingredient/structure resolution remains required.
- Bounded excerpts reduce text duplication; the immutable XML locator and hash are authoritative.

Raw index: `research/data/platform/raw/external_public/pk_expansion/avicenna/dailymed_pk_candidate_evidence/`.
