# Failed canonical attempt 1

This directory is diagnostic evidence from the fail-closed first real canonical attempt on 2026-08-04. It is quarantined and is not an input, accepted canonical corpus, or model-ready artifact.

The attempt stopped on the first specialized shard before promotion because 44 of 271 nominal task candidates lacked the protein sequence required by the molecule-plus-protein task contract. All 2,498 source observations and lineage rows written before the check are retained here. A read-only replay under the repaired policy reconciled 271 candidates to 227 eligible rows plus 44 explicit `missing_protein_sequence` exclusions, all for `CHEMBL612545` (`Unchecked`).

Pre-move SHA-256 values:

- `.canonical_registry.sqlite`: `b6e79bd434bd7530779f132cf87ae8d7cc8252b00db6101c4a5b3564fd8f491b`
- `observations/part-00000.parquet`: `1ad6ff10056b9bca7c8992687b9f6a5ee98e9deb96bf588e9c25fb52a8496d77`
- `observation_lineage/part-00000.parquet`: `1a52964d739b727ac2626276c521c3f9911192ea5184a708943276f1e9026567`

The accepted rebuild must be read only from `research/data/platform/canonical/full_chembl37/` and must pass manifest-bound QC independently.
