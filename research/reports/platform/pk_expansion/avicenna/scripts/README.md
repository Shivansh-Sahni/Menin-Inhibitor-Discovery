# DailyMed PK candidate scanner reproducibility

This directory preserves the exact programs used for the completed DailyMed
candidate-evidence build:

- `dailymed_pk_candidate_scanner_exact.py` — byte-identical to the final
  `/private/tmp/extract_dailymed_pk_candidates.py` used to scan each archive.
- `dailymed_pk_candidate_merge_validator_exact.py` — byte-identical to the
  final `/private/tmp/merge_dailymed_pk_candidates.py` used to merge the six
  part outputs and construct the original validation artifacts.
- `dailymed_pk_candidate_cli.py` — fail-closed wrapper and independent
  validation-only replay.

The exact programs intentionally retain their original absolute source paths
and temporary part paths because changing them would no longer preserve the
executed implementation. Use the wrapper for current CLI operation.

The historical scanner and merger also retain their executed run-timestamp and
elapsed-time fields. Those run metadata values vary, while JSONL evidence rows,
candidate IDs, hashes, ordering, computed counts, and the validation-only replay
are deterministic.

## Bounded smoke scan

```bash
python research/reports/platform/pk_expansion/avicenna/scripts/dailymed_pk_candidate_cli.py scan \
  --archive dm_spl_release_human_rx_part6.zip \
  --limit 10 \
  --out /private/tmp/dailymed-pk-smoke \
  --receipt /private/tmp/dailymed-pk-smoke-receipt.json
```

The output directory must be absent or empty. Any scanner error, parse error,
manifest mismatch, incomplete XML parse, or nonzero admission count produces a
nonzero exit status.

## Validation-only full replay

```bash
python research/reports/platform/pk_expansion/avicenna/scripts/dailymed_pk_candidate_cli.py validate \
  --evidence-dir research/data/platform/raw/external_public/pk_expansion/avicenna/dailymed_pk_candidate_evidence \
  --source-manifest research/data/platform/raw/external_public/dailymed_spl_v2_human_rx/dailymed_spl_v2_human_rx_manifest.json \
  --output research/reports/platform/pk_expansion/avicenna/dailymed_pk_candidate_validation_replay.json
```

This replay is read-only with respect to raw evidence rows. It rehashes all six
declared evidence artifacts, reparses every JSONL record, recomputes core
counts and links, checks uniqueness and table hashes, verifies zero admitted
rows/labels, and confirms the frozen source-manifest binding. Any false check
or malformed input returns exit status 2 or 3.
