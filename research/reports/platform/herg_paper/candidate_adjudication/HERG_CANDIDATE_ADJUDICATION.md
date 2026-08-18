# hERG candidate and conflict adjudication evidence

## Outcome

This release adds deterministic local evidence for all 226 evaluation candidates and 1,340 pIC50 conflict structures. It records **zero human decisions** and promotes **zero rows to gold**.

## Candidate evidence

- Candidate observations: 226 across 123 structures.
- Direct human KCNH2 single-protein target metadata with no variant annotation: 88.
- Homologue relationship to human KCNH2 with no variant annotation: 138.
- Explicit wild-type wording in the locally bound assay description: 0.
- Known ChEMBL document IDs: 226; DOI available: 226; PubMed ID available: 224.
- Candidates participating in at least one automated lineage group: 113.
- Automated relation/unit inconsistencies: 0.
- Target evidence classes: `{"direct_human_kcnh2_single_protein_no_variant_annotation":88,"homologue_relationship_to_human_kcnh2_no_variant_annotation":138}`.
- Existing model partitions: `{"test":18,"train":180,"validation":28}`.

A null ChEMBL variant identifier is only an absence of variant annotation. It is not explicit wild-type confirmation. Likewise, relationship type `H` is a homologue relationship and must be resolved against the primary document before any evaluation use.

## Conflict and lineage evidence

- Conflict structures with at least one automated duplicate/mirror/source-reuse group: 696.
- Conflict structures with cross-source exact-value mirror candidates: 617.
- Conflict structures with source-primary-key reuse: 33.
- Conflict observation bindings replayed to manifest-bound local sources: 4,392.
- Lineage groups: 951; groups touching a queued observation: 936.
- Lineage group kinds: `{"exact_native_measurement_signature":54,"source_record_reuse":37,"standardized_equal_value_cluster":860}`.
- Automated evidence strengths: `{"moderate":193,"strong":609,"weak":149}`.

Equal standardized values, shared source keys, shared assay/document context, and cross-database matches are evidence for review—not proof of a common experiment. No rows were collapsed, averaged, corrected, excluded, or relabeled.

## Human packet

The packet contains 1,566 pending items. Every decision, reviewer, review date, and note field is intentionally blank. The accompanying decision contract defines allowable dispositions and requires primary-source verification, target/construct resolution, protocol comparison, and a new structure/scaffold/assay/document/measurement-lineage freeze before any accepted candidate can enter an evaluation panel.

## Limits

- This is local automated evidence only; no primary paper was manually read in this build.
- Distinct sources, documents, assays, or rows are not assumed independent.
- Absence of an automated duplicate signal is not proof of independence.
- Missing document or protocol fields may be recoverable from primary sources, but are not inferred here.
- Current train/validation/test labels are cautions, not authorization to move a row into a sealed test set.
- Any future gold release must be separately versioned after completed human decisions and leakage-safe refreezing.

No model, feature generator, smoke test, or HPC job was run.
