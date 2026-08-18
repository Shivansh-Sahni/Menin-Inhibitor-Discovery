# Interrupted canonical build attempt 4B

This independent determinism build was deliberately interrupted by the lead
on 2026-08-04 at the same audit gate as attempt 4A. It used a disjoint output
root, had completed only source shard 0, was never promoted, and has no build
manifest or QC acceptance. It is retained only as diagnostic evidence.

- Stop reason: add symlink/path rejection and prove the private `.building`
  publication boundary before restarting either deterministic build.
- Size at preservation: approximately 9.5 MiB.
- SQLite registry size: 9,162,752 bytes.
- `observations/part-00000.parquet` SHA-256:
  `1ad6ff10056b9bca7c8992687b9f6a5ee98e9deb96bf588e9c25fb52a8496d77`.

This directory is excluded from all downstream inputs and must not be promoted
or resumed.
