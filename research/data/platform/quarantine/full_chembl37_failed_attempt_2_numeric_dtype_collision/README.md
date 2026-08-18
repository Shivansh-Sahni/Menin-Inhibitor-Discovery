# Quarantined canonical build attempt 2

This directory is a failed-closed, never-promoted ChEMBL 37 canonical build.
It is retained for audit evidence only and is excluded from all accepted corpus,
analysis, split, and model-readiness inputs.

- Failure time: 2026-08-04T17:20:36-04:00
- Failure gate: `_DiskRegistry.assert_consistent()`
- Reported conflict: 777 `molecules` records
- Root cause: pandas promoted nullable integer `fragment_count` values to float
  in some shards, so semantically identical values serialized as `1` versus
  `1.0` and triggered the exact-payload collision guard.
- Evidence: all 777 conflicted identities were independently rescanned across
  all 23 specialized activity parts; each had exactly one source compound,
  canonical SMILES, and standard InChIKey tuple. Replaying the first two parts
  reproduced the issue and showed no field-value difference after JSON parsing.
- Resolution: integral finite floats are now normalized to integers only for
  registry identity serialization. A regression proves `1` equals `1.0` while
  `1.5` remains a true conflict. The collision guard remains fail closed.
- Patched module SHA-256: `3028c980c9663425e5e1e791ba00cbd51d198d210b98c2787af4ea9b04e9ac6d`
- Patched test SHA-256: `b2cc97d6d3efd75090a8a1c82451e0a7e224373894a01e50517f27a2050f1986`

The directory must not be renamed to an accepted canonical build or supplied
to downstream tooling.
