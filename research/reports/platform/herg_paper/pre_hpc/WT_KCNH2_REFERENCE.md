# Frozen wild-type KCNH2 reference

The future protein-aware hERG pipeline now has one immutable human WT sequence input. It is the reviewed UniProt canonical entry `Q12809` / `KCNH2_HUMAN` from release `2026_02`: 1,159 amino acids, sequence SHA-256 `287332153da38b59cc1be9554cc3a29f14d3b9e2a33150b4d54137773b22d1f7`.

The build validates the frozen UniProt acquisition page against its acquisition manifest, requires one reviewed human Q12809 record, independently recomputes sequence length/MD5/SHA-256, and proves JSON/FASTA identity. The output manifest self-hash is `502b70e765237b615ba3f622eabf284f7321e95d8d7234dc84b7c0fb7874807d`; it admits zero mutant sequences.

This is deliberately a sequence contract, not a receptor claim. It does not select a truncated experimental construct, structure, membrane state, conformation, or docking receptor. Those remain blocked until explicit scientific selection and residue mapping are recorded.

Machine-readable artifacts are under `research/data/platform/processed/herg_hierarchy/v1_5_wt_reference/`.
