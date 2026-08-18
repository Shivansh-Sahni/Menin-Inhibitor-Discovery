# Interrupted canonical build attempt 4A

This primary build was deliberately interrupted by the lead on 2026-08-04
after an independent audit rejected the then-current schema-normalization
transaction boundary. It had completed only source shard 0. It was never
promoted, has no build manifest or QC acceptance, and is retained only as
diagnostic evidence.

- Stop reason: add symlink/path rejection and prove that an interrupted
  multi-file normalization commit remains inside an unpublishable,
  non-resumable `.building` directory.
- Size at preservation: approximately 9.5 MiB.
- SQLite registry size: 9,162,752 bytes.
- `observations/part-00000.parquet` SHA-256:
  `1ad6ff10056b9bca7c8992687b9f6a5ee98e9deb96bf588e9c25fb52a8496d77`.

This directory is excluded from all downstream inputs and must not be promoted
or resumed.
