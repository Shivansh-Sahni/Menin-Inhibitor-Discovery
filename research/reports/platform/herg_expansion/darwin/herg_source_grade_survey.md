# hERG source-grade expansion survey

Status: completed 2026-08-06. Scope: source discovery, rights, assay semantics, counts, and overlap. This work does **not** modify the canonical corpus or train a model.

## Executive decision

The project already possesses a much larger hERG corpus than its current strict task views expose: the local ChEMBL 37 CHEMBL240 extract contains **41,078 measurements, 26,254 unique non-null Standard InChIKeys, 4,829 assays, and 3,494 documents**. The immediate opportunity is therefore not indiscriminate aggregation. It is a provenance-preserving, assay-aware reconstruction of multiple hERG tasks.

Recommended source order:

1. **ChEMBL 37 is the authoritative backbone.** Retain raw relations, units, assay descriptions, source tags, and document identifiers. Create separate IC50, percent-inhibition, binding, kinetic, and assay-modality views.
2. **Add PubChem AID 588834 as a source-grade qHTS modality**, after an assay-specific PUG REST acquisition and duplicate reconciliation. It is a thallium-flux functional screen, not patch clamp.
3. **Query BindingDB only for post-ChEMBL or BindingDB-curated deltas.** Do not import its ChEMBL-origin records again.
4. **Treat TDC hERG and hERG Karim as frozen benchmarks/audit sets**, not new experimental sources. Their label wrappers lose assay provenance and substantially overlap ChEMBL.
5. **Do not bulk-ingest hERGCentral/TDC hERGCentral until licensing and label-sign semantics are resolved.** Its scale is useful, but its one-dose inactive majority must remain a separate weak/single-dose task.
6. **Use clinical QT/TdP, DILI, and ECG resources only as separate downstream annotations or evaluation tasks.** They are not hERG measurements.

The full machine-readable decision table is [source_decision_matrix.csv](source_decision_matrix.csv), and verified counts are in [verified_counts_and_overlap.json](verified_counts_and_overlap.json).

## What the local ChEMBL data actually contain

The 41,078 records are not one homogeneous endpoint:

- IC50: 19,130 rows / 15,995 unique keys / 2,658 assays.
- Percent inhibition: 7,808 / 6,453 / 1,623.
- Ki: 3,689 / 2,942 / 195.
- `kon` and `k_off`: 2,946 rows each / 1,665 keys / 68 assays each.
- AC50: 1,823 / 1,792 / 2.
- Remaining records include potency, activity, EC50, Kd, and small heterogeneous endpoint groups.

The source tags also overlap. ChEMBL labels 28,126 rows as literature, 9,210 as BindingDB, 1,742 as DrugMatrix, and 803 as PubChem BioAssay. The BindingDB tag contributes only 1,855 unique keys, and 104 of those keys also occur in ChEMBL's literature tag. Therefore, source tags are provenance fields, not independent datasets.

Quality flags are material: 2,223 rows are marked potential duplicates, 482 have a data-validity comment, 11,908 have `standard_flag=0`, and 13,198 records are censored or have no standard relation rather than exact equality. These records may still be useful, but only in censored/weak-label tasks.

Description heuristics find approximately 16,430 electrophysiology/patch-clamp rows, 10,723 radioligand/binding rows, and 2,976 fluorescence/membrane-potential/flux rows. These are heuristic strata requiring assay-level review, not definitive modality labels. The integrated hERG database paper independently shows why this matters: after exclusions it found only 209 compounds measured in both binding and electrophysiology modes and reported deviations exceeding 100-fold for some molecules ([PLOS One, 2018](https://doi.org/10.1371/journal.pone.0199348)).

## Direct verification of the two small TDC files

Two small official TDC-hosted Harvard Dataverse files were downloaded solely for audit; hashes and URLs are recorded in the raw [acquisition manifest](../../../../data/platform/raw/external_public/herg_expansion/darwin/tdc/acquisition_manifest.json). No hERGCentral bulk file was downloaded.

### TDC `herg`

- The file has **655 rows**, despite the TDC page describing 648 drugs.
- It has 648 unique raw/canonical SMILES, 451 positive rows, and 204 negative rows.
- Seven rows duplicate a canonical structure; three canonical structures have conflicting labels.
- RDKit produced 644 unique full-structure InChIKeys; 177 (27.5%) overlap the local ChEMBL hERG keys.

### TDC `herg_karim`

- Exactly 13,445 rows and 13,445 unique canonical SMILES.
- 6,718 positives and 6,727 negatives.
- RDKit produced 13,267 unique InChIKeys; 9,131 (68.8%) overlap local ChEMBL hERG.
- It shares 458 InChIKeys with TDC `herg`; 77 of those shared keys have disagreeing binary labels.

This is direct evidence that blindly concatenating benchmark wrappers would introduce duplicates and label contradictions. The Karim/CardioTox paper says its source collection combined BindingDB, ChEMBL, and literature and reduced 30,000 input structures to 12,620 training molecules after removing inconsistent labels ([Journal of Cheminformatics, 2021](https://doi.org/10.1186/s13321-021-00541-z)). TDC's 13,445-row wrapper is consequently a useful benchmark snapshot, not 13,445 provenance-resolved new experiments.

## Rights and availability findings

- **ChEMBL:** CC BY-SA 3.0; adaptations must preserve attribution and share-alike ([official ChEMBL FAQ](https://chembl.gitbook.io/chembl-interface-documentation/frequently-asked-questions/general-questions)).
- **BindingDB:** BindingDB-curated records are CC BY 4.0; records imported from ChEMBL remain under ChEMBL's CC BY-SA 3.0, and BindingDB marks provenance for compliance ([BindingDB in 2024](https://doi.org/10.1093/nar/gkae1075)).
- **TDC:** the toxicity page displays “Dataset License: Not Specified” alongside “CC BY 4.0.” The TDC GitHub repository is MIT-licensed code, but its own documentation says individual dataset licenses govern data. Treat both downloaded files as quarantined until source-specific rights are confirmed ([TDC toxicity datasets](https://tdcommons.ai/single_pred_tasks/tox/), [TDC repository](https://github.com/mims-harvard/TDC)).
- **hERGCentral:** the primary paper describes a repository of more than 300,000 compounds with annotated structures, inhibition data, and raw traces, but a current explicit dataset license was not located ([primary paper](https://doi.org/10.1089/adt.2011.0425)). Do not infer data rights from article access.
- **Integrated hERG DB:** the article is CC BY, but the integration included commercial GOSTAR plus ChEMBL, NCGC/PubChem, and hERGCentral. Article license does not clear redistribution of all underlying records ([paper](https://doi.org/10.1371/journal.pone.0199348)).
- **OCHEM:** official documentation requires a source for every measurement and supports record-level access/filters, but no blanket hERG dataset license or authoritative current hERG count was found ([OCHEM database manual](https://docs.ochem.eu/x/JAFe.html)). Use it as a discovery/index layer unless every exported record has compatible provenance and permissions.
- **CredibleMeds:** QTdrugs is copyrighted; registration is required and commercial/software embedding requires a license ([registration/rights](https://www.crediblemeds.org/register)). It is reference-only absent a suitable license.
- **PhysioNet QTDB:** 105 ECG records, licensed ODC Attribution 1.0, but it contains waveform/QT-boundary annotations rather than molecule exposures ([QTDB v1.0.0](https://physionet.org/content/qtdb/1.0.0/)).
- **FDA DILIrank 2.0:** 1,336 FDA-approved drugs annotated for liver-injury concern. It is valuable as a clinical safety covariate, never as a cardiac label ([FDA DILIrank 2.0](https://www.fda.gov/science-research/liver-toxicity-knowledge-base-ltkb/drug-induced-liver-injury-rank-dilirank-20-dataset)).

## Assay semantics that must become schema, not prose

Every hERG record should retain:

- source database, source record/assay ID, document DOI/PMID/patent, and source-specific license;
- molecule form as reported plus standardized parent, with mapping history;
- endpoint (`IC50`, `Ki`, `Kd`, `AC50`, percent inhibition, kinetic rate, categorical call);
- value, relation/censoring operator, unit, tested concentration, and concentration range;
- functional versus binding; manual patch, automated patch, flux/fluorescence, radioligand, or unresolved modality;
- cell line/species/channel construct, voltage protocol, temperature, incubation, and assay controls where available;
- quality/curve class, replicate/aggregation rule, and whether the label is measured or derived;
- development/clinical annotation kept in a separate linked table.

Critical exclusions:

- PubChem AID 1511 is a screen for compounds that **protect hERG from dofetilide block**, not a conventional blocker assay ([AID 1511](https://pubchem.ncbi.nlm.nih.gov/bioassay/1511)).
- PubChem AID 588834 uses FluxOR thallium flux and curve-fitted AC50 values; it must not be renamed patch-clamp IC50 ([AID 588834](https://pubchem.ncbi.nlm.nih.gov/bioassay/588834)).
- A single-dose non-inhibitor does not prove `IC50 > tested concentration` unless the assay and threshold rule support that inference.
- hERG block, QT prolongation, and torsades de pointes are related but distinct biological/clinical outcomes.

## Non-HPC execution plan

1. Freeze ChEMBL 37 raw records and generate a deterministic assay dictionary keyed by `assay_chembl_id`.
2. Manually adjudicate the highest-volume assays first; propagate only controlled modality labels.
3. Build five separate data products: exact/censored IC50, fixed-dose inhibition, binding affinity, kinetic rates, and categorical literature/benchmark calls.
4. Pull PubChem AID 588834 with complete result fields and reconcile by source assay ID, Standard InChIKey, and concentration rather than by structure alone.
5. Query a current BindingDB hERG export; admit only BindingDB-curated records absent from the ChEMBL snapshot, preserving original rights.
6. Run pairwise identity/label-conflict reports for every incoming source before aggregation. Never silently majority-vote across modalities.
7. Freeze TDC hERG and hERG Karim as external audit sets with canonical-molecule grouped splits; prevent all overlapping ChEMBL molecules/scaffolds from leaking into their evaluation.
8. Keep hERGCentral, CredibleMeds, and any commercial-source integration behind explicit rights gates.
9. Report performance by assay modality, source, chemical scaffold, temporal group, and censoring status; a single pooled AUROC would conceal the central scientific problem.

## Brutally honest conclusion

The project does not yet have a clean “very large hERG training set.” It has a strong raw ChEMBL foundation plus several tempting but heavily overlapping secondary collections. The scientifically defensible expansion is to expose more of the 41,078 local ChEMBL measurements through explicit task definitions and then add only source-cleared, genuinely novel measurements. The most publishable analyses are likely the cross-assay/method disagreement, benchmark-label contradictions, censoring-aware performance, and source/domain shift—not a claim that hundreds of thousands of heterogeneous records are equivalent IC50 observations.
