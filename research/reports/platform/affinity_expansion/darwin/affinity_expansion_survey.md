# Binding-affinity expansion: acquisition, evidence, and decisions

## Result

The non-HPC affinity expansion now has a reproducible full BindingDB snapshot, the current Guide to PHARMACOLOGY interaction/ligand exports, the BioLiP2 structure-affinity linkage table, and official Papyrus release metadata. The main scientific conclusion is that raw source row totals cannot be summed: BindingDB already contains 1,658,927 ChEMBL-tagged rows and substantial PubChem/patent/PDSP content; BioLiP2 links rather than creates many BindingDB/MOAD measurements; GPCRdb and Papyrus are principally integrated views of upstream sources.

The largest defensible local label base before identity-level reconciliation is therefore the **union** of local ChEMBL 37 (2,117,384 endpoint rows) and BindingDB 2026-08 (3,234,499 physical rows), plus curated Guide to PHARMACOLOGY evidence. It is not their arithmetic sum. A multi-million corpus is realistic, but only after source-aware compound/target mapping and measurement-level deduplication.

## Acquired evidence

- **BindingDB 2026-08 full TSV:** 592,986,963 compressed bytes; 8,978,755,790 uncompressed bytes; upstream MD5 and local SHA-256 verified. Exact scan found 3,234,499 rows and 2,481,305 distinct monomer-target keys: 130,168 rows with Kd, 627,186 with Ki, 2,209,905 with IC50, and 270,125 with EC50. Relations were retained: 648,109 of those populated endpoint cells were censor/range-like rather than plain exact values.
- **Guide to PHARMACOLOGY 2026.2:** 24,599 interaction rows; the publisher reports 22,316 quantitative interactions and 20,258 unique target-ligand pairs. Endpoint labels include pKd, pKi, pIC50, and pEC50, plus pKB/pA2 that must remain distinct.
- **BioLiP2 affinity linkage:** 68,890 rows, 19,947 PDB IDs, 49,802 unique structural-site keys, and 6,520 nonblank UniProt IDs. Its BindingDB, MOAD, and manual source fields overlap; these are not 68,890 independent new measurements.
- **Papyrus 05.7 metadata:** official Zenodo record resolves to dataset version 2024.2, published 2024-10-24 under CC BY-SA 4.0. The normalized no-stereochemistry activity archive alone is 751,521,856 compressed bytes. It was not downloaded because its dominant ChEMBL source is older than local ChEMBL 37, so it is a delta/reference candidate rather than an immediate primary source.

All exact counts, checksums, paths, and source URLs are recorded in `verified_counts_and_overlap.json` and the raw `acquisition_manifest.json`.

## Endpoint semantics that must survive ingestion

1. Keep **Kd, Ki, IC50, and EC50 as separate endpoint types**. Kd/Ki are affinity constants; IC50/EC50 depend on assay design and are not interchangeable free energies.
2. Preserve the original relation (`=`, `<`, `<=`, `>`, `>=`, approximate/range), value, unit, assay, target construct/species, source, and citation. A censored result must not be trained as an exact point.
3. Normalize to molar and pX only in derived columns. Retain original text and never infer Kd from IC50 unless an explicit, justified assay-specific conversion is stored as a transformation.
4. Do not average measurements across assays during ingestion. Create a later consensus view only after identity mapping and disagreement analysis; keep every supporting measurement addressable.

## Deduplication design

Use a two-level representation:

- **Evidence row key:** source + source record/assay ID + ligand identity + target identity/construct + endpoint + relation + value/unit + citation.
- **Biological interaction key:** standardized parent compound/InChIKey + target UniProt/complex identity + endpoint. This enables overlap and disagreement analysis without deleting assay-level evidence.

Deduplication order should be: validate values/units → standardize compounds while preserving salts/stereochemistry → map targets and complexes → map citations/assays → exact evidence deduplication → near-duplicate/source-lineage clustering → optional consensus. BindingDB rows tagged ChEMBL must link to, not duplicate, the corresponding ChEMBL measurement. PubChem AIDs in BindingDB should drive a targeted PubChem confirmatory-assay delta instead of a second bulk import.

## Rights and redistribution boundary

- BindingDB-curated records are reported as CC BY 4.0; ChEMBL-imported records retain ChEMBL share-alike requirements. ChEMBL's official FAQ states CC BY-SA 3.0. Use per-row lineage and ship attribution/license manifests.
- Guide to PHARMACOLOGY uses ODbL for the database and CC BY-SA 4.0 for content.
- BioLiP says data are freely available but does not expose a clear standardized database license on the inspected pages. Its January 2025 removal of 24,809 PDBbind-CN records for licensing reasons is strong evidence that source-level rights cannot be flattened.
- PDBbind redistribution is not permitted without explicit permission. Do not place PDBbind data in a public corpus.
- PLINDER's curated portion is Apache 2.0, but upstream BindingDB/ChEMBL terms still attach. Its public repository also documents a BindingDB parsing defect and disables `ligand_binding_affinity` queries; that label should not be used.
- CrossDocked docking scores are predictions, not experimental affinity, and should never be mixed into Kd/Ki/IC50/EC50 labels.

## Recommended next ingestion sequence

1. Parse BindingDB in chunks to Parquet with lossless endpoint/relation/source columns; separately quarantine Taylor-lab rows and tag ChEMBL-imported lineage.
2. Reconcile ChEMBL 37 ↔ BindingDB using ChEMBL IDs, InChIKey/structure, UniProt/target-complex IDs, assay/citation, endpoint, relation, and normalized value—not structure alone.
3. Add Guide to PHARMACOLOGY as high-trust curated evidence, preserving pKB/pA2 separately and using its references to audit disagreements.
4. Use BioLiP2 only to attach PDB/site context and MOAD/manual provenance; do not count structural repetitions as new affinities.
5. Query PubChem only for confirmatory AIDs absent from both primary sources. Evaluate Papyrus source-specific deltas, not its full ChEMBL-derived body.
6. Generate frozen train/validation/test splits by scaffold, target-family/sequence similarity, and publication time. Deduplicate before splitting to prevent leakage.

## Primary evidence reviewed

- BindingDB current download and 2024 update: `https://www.bindingdb.org/rwd/bind/chemsearch/marvin/Download.jsp` and DOI `10.1093/nar/gkae1075`.
- ChEMBL license FAQ: `https://chembl.gitbook.io/chembl-interface-documentation/frequently-asked-questions/general-questions`.
- Guide to PHARMACOLOGY downloads/content and 2026.2 release: `https://www.guidetopharmacology.org/download.jsp`.
- BioLiP2 current database/download: `https://zhanggroup.org/BioLiP/` and DOI `10.1093/nar/gkad630`.
- Papyrus paper and official 05.7 record: DOI `10.1186/s13321-022-00672-x`, `https://zenodo.org/records/13987985`.
- PLINDER repository: `https://github.com/plinder-org/plinder`.
- BigBind repository/paper: `https://github.com/molecularmodelinglab/bigbind` and DOI `10.1021/acs.jcim.3c01208`.
- CrossDocked2020 archive: `https://bits.csb.pitt.edu/files/crossdock2020/v1.1/`.
- Binding MOAD historical scope: DOI `10.1093/nar/gku1088`.
- PDBbind rights boundary and historical description: DOI `10.1093/bioinformatics/btu626`; BindingDB's 2024 update explicitly explains why it does not collect PDBbind data.
- GPCRdb data/license documentation: `https://gpcrdb-data.readthedocs.io/en/latest/legal_notice.html` and `https://docs.gpcrdb.org/ligands.html`.

## Brutally honest limitations

- No identity-level BindingDB↔ChEMBL overlap count has yet been produced; source-row counts prove extensive overlap but are not a substitute for record matching.
- BioLiP's reuse language is less explicit than a conventional database license, so public redistribution should await institutional review or written confirmation.
- BindingDB's live page and downloaded archive differed by 4,828 measurements, consistent with the live database advancing after the archive snapshot. Reproducible analyses must cite the archive hash and its 3,234,499 scanned rows.
- No claimed “binding free energy” was manufactured from assay endpoints. Experimental Kd can be transformed to standard free energy only with stated temperature/standard-state assumptions; IC50/EC50 cannot simply be relabeled as ΔG.
