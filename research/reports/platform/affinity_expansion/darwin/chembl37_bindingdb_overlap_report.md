# ChEMBL 37 ↔ BindingDB 2026-08 overlap audit

## Bottom line

BindingDB contains **1,658,927 physical rows explicitly tagged `Curation/DataSource=ChEMBL`**. Of these, 1,658,155 contain a populated Kd, Ki, IC50, or EC50 field. This is direct proof that BindingDB and ChEMBL cannot be added together as independent datasets.

Using a strict standardized key—ligand identity, sorted UniProt component set, endpoint, censor relation, and numeric nM value—**1,229,978 of 1,658,153 eligible BindingDB measurement rows (74.18%) have an exact ChEMBL 37 key hit** through either an explicit ChEMBL ligand ID or an exact InChIKey. The comparison preserved Kd/Ki/IC50/EC50 and `=`, `<`, and `>` semantics; zero populated values failed parsing.

## What was matched

Two deliberately separate identity routes were measured:

- **Explicit ChEMBL-ID route:** 1,064,845 BindingDB measurements were eligible; 929,186 matched (87.26%). At unique-key level, 857,134 of 981,330 keys matched.
- **Exact InChIKey route:** 1,658,152 measurements were eligible; 831,340 matched.
- **Combined route:** 530,548 measurement rows matched through both routes, 398,638 through ID only, and 300,792 through InChIKey only. The InChIKey-only recovery is important because 593,439 explicitly ChEMBL-tagged BindingDB rows do not carry a ligand ChEMBL ID.

Endpoint-level combined matches were:

| Endpoint | BindingDB measurements | Exact key matches | Match rate | Eligible unmatched |
|---|---:|---:|---:|---:|
| EC50 | 143,693 | 98,651 | 68.65% | 45,042 |
| IC50 | 1,013,258 | 759,801 | 74.99% | 253,456 |
| Kd | 90,909 | 67,214 | 73.94% | 23,695 |
| Ki | 410,295 | 304,312 | 74.17% | 105,982 |

## Compound and target coverage

- 420,948 of 427,518 distinct explicit ligand ChEMBL IDs occur in ChEMBL 37; 6,570 do not.
- 417,955 of 653,927 distinct exact BindingDB InChIKeys occur in the ChEMBL 37 structure table.
- At physical-row level, 1,443,944 of 1,658,927 rows (87.04%) resolve through at least one compound route; 214,983 do not.
- 8,719 of 8,942 normalized UniProt target-component sets occur exactly in ChEMBL 37; 223 do not. Target accessions were complete relative to BindingDB's declared chain count for 1,658,921 rows; only six were partial/undeclared and none lacked all accessions.
- 455,907 rows have an explicit ChEMBL ID that exists in ChEMBL 37 but a different InChIKey. This is an audit queue, not proof of bad data: parent/salt handling, protonation, stereochemistry, or release drift can legitimately change the exact structure key.

## Defensible union range

ChEMBL 37 contains 4,870,628 nM activity rows for the four endpoints across all assay and target types, collapsing to **2,864,273 distinct resolved standardized measurement keys**. BindingDB's ChEMBL-tagged subset contains 1,539,331 distinct combined-identity keys, of which 1,137,248 match ChEMBL 37.

The defensible identity-reconciled union range is therefore:

- **Lower bound: 2,864,273 keys.** This assumes every currently unmatched BindingDB key is ultimately an alias or release-drift duplicate.
- **Explicit-ID observed union: 2,988,469 keys** (`2,864,273 + 981,330 - 857,134`). This is the highest-confidence directly identified subdomain but excludes rows without a ligand ChEMBL ID.
- **Combined-identity upper bound: 3,266,356 keys** (`2,864,273 + 1,539,331 - 1,137,248`). This treats all 402,083 currently unmatched combined keys as distinct.

This is a range of standardized measurement keys, not assays, citations, physical source rows, or merely ligand-target pairs. It should not be reported as a final training-row total until structure-aware and activity-ID-level reconciliation is complete.

## Unresolved boundary and honest limitations

- BindingDB does not expose the originating ChEMBL activity ID in this TSV. These are exact key-existence matches, not one-to-one source-record matches.
- Two endpoint rows lack both usable ligand identity routes. More importantly, 402,083 distinct combined keys have no exact ChEMBL 37 hit and require structure/target/version reconciliation.
- The audit intentionally compares against all ChEMBL 37 nM rows for Kd/Ki/IC50/EC50, rather than the platform's narrower 2,117,384-row binding-task subset. BindingDB's ChEMBL import spans assay and target categories; restricting ChEMBL prematurely would falsely label legitimate overlaps as novel.
- No endpoint was converted, no censor relation was made exact, no measurements were averaged, and no canonical files or models were modified.

The complete exact metrics are in `chembl37_bindingdb_overlap_audit.json` and `chembl37_bindingdb_overlap_metrics.csv`. The 1.2 GB SQLite audit database preserves row-level matches and passed `PRAGMA quick_check`.
