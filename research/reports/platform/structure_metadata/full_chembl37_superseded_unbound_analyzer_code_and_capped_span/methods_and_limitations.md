# PDB structure-metadata coverage

## Result

- Exact SIFTS accession mapping candidates exist for 5,428/9,411
  frozen canonical protein rows and 7,176/14,983 frozen returned
  UniProt entries.
- This is **metadata coverage**, not coordinate readiness, construct equivalence, or
  experimentally observed-residue coverage.
- Zero coordinate files and zero predicted structure files were downloaded. Zero labels
  were created and no model was trained.

## Sources and method

- PDBe SIFTS `pdb_chain_uniprot.tsv.gz`: https://ftp.ebi.ac.uk/pub/databases/msd/sifts/flatfiles/tsv/pdb_chain_uniprot.tsv.gz
- wwPDB `pdb_entry_type.txt`: https://files.wwpdb.org/pub/pdb/derived_data/pdb_entry_type.txt
- Release: PDB 31.26; UniProt 2026.03;
  generated 2026/08/03 16:01.
- Exact raw bytes, HTTP response headers, local SHA-256 hashes, citations, and the PDBe
  public-data statement were preserved. The endpoints did not expose cryptographic
  sidecar checksums; HTTP ETags were preserved and local SHA-256 was computed.
- Every SIFTS segment was parsed into a deterministic Parquet table. Frozen canonical
  and external-UniProt rows were reconciled only by exact accession string.
- `pdb_entry_type.txt` supplies coarse diffraction/NMR/EM/other archive method classes.
  These PDB archive mappings are explicitly separate from predicted-structure resources.

## Limits and next gates

- SIFTS uses UniProt 2026.03 while the frozen external input is 2026_02. Exact accession
  agreement does not prove sequence-version identity.
- The reported span fraction is the outer mapped UniProt range, not observed residues;
  gaps and unobserved residues can occur.
- Chain-level SIFTS mapping cannot prove exact construct boundaries, variants, mutations,
  expression tags, biological assembly, ligand state, resolution, or validation quality.
- The 9,071 canonical construct
  records therefore remain unreconciled.
- Coordinate/validation retrieval should be limited to a task-specific, frozen subset
  after construct, ligand, method, quality, and leakage policies are approved.
