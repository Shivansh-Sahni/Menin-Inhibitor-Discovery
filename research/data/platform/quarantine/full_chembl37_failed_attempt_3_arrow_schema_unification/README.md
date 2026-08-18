# Quarantined canonical build attempt 3

This directory is a failed-closed, never-promoted ChEMBL 37 canonical build.
It is retained as forensic evidence only and is excluded from every accepted
canonical, analysis, split, corpus-readiness, and model input.

- Failure date: 2026-08-04
- Failure gate: final logical-partition Arrow schema normalization
- Exception: `Partition Arrow schemas cannot be safely unified`
- Root cause: `derived_observations/part-00006.parquet` physically inferred its
  entirely null `evidence_stage` column as `double`, while other shards used
  string types. Physical-schema unification rejected that difference before the
  declared normative `large_string` rewrite could run.
- Live offending column evidence: 45 rows, 45 nulls; no numeric value was
  reinterpreted as text.
- Promotion state: absent `build_manifest.json`, absent accepted QC report, and
  no canonical-directory promotion.
- Recovery decision: direct promotion or ad hoc continuation is prohibited.
  The in-memory artifact records and counters were lost when the process
  exited, and the public builder intentionally refuses an existing staging
  directory. A clean rebuild is required.

Pre-move inventory:

| Dataset | Parquet parts | Rows | Bytes |
| --- | ---: | ---: | ---: |
| `observations` | 23 | 3,938,372 | 280,190,289 |
| `observation_lineage` | 23 | 3,938,372 | 100,385,819 |
| `derived_observations` | 6 | 67,848 | 7,668,333 |
| `views` | 6 | 67,848 | 7,018,010 |
| `tasks` | 141 | 2,616,037 | 442,346,147 |
| `task_exclusions` | 4 | 63 | 47,703 |

The directory contained 203 Parquet files and three SQLite files. The registry
was 2,970,304,512 bytes; its WAL was zero bytes at preservation time.

Pre-move SHA-256 evidence:

- `.canonical_registry.sqlite`:
  `5ec618e58e59f3eb00a75f231d9ce9ea28d999e531d63bc1638779bbaf0a9ac4`
- `derived_observations/part-00006.parquet`:
  `6893195d8fc1dee01073d92816caf8fc93b04eeb323f7aa34d0b122650caf266`
- `observations/part-00006.parquet`:
  `4732ad99f8fc45b6da83d741cda0423bac5d2f49e084e85d22943a1f0fd90b5e`

The code that failed had module SHA-256 `3028c980c9663425e5e1e791ba00cbd51d198d210b98c2787af4ea9b04e9ac6d`.
The final repaired module (`240b52193bf8d275c2401e13632d708810daab764112d2037c0c74376d2b980d`)
unions field membership independently of inferred physical types, applies the
normative schema, prohibits non-null incompatible physical families, and
stages and verifies every rewrite before replacing any source shard. Its test
SHA-256 is `e3a601b175ae6edf1c7d9e460b2ed7770fba0b8a0bb5b2ea1bea40fb317a6c29`.
It additionally rejects noncanonical, linked, and special staging paths and
revalidates source and temporary paths immediately before writes and commits.
The repair passed 44 combined canonical/QC/determinism tests, Ruff check, Ruff
format check, mypy, and two independent audits before the clean rebuild.

Do not rename this directory to an accepted canonical build or supply it to
downstream tooling.
