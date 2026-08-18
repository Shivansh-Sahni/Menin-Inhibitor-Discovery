# External sources and references

Record the source release/status and access date from the actual build. The references below describe the resources and methods; they do not replace a build manifest or source-specific citation.

## Data resources

### ChEMBL

- Collection interface: [ChEMBL REST API](https://www.ebi.ac.uk/chembl/api/data/docs)
- Target anchors: Menin `CHEMBL1615381`; hERG `CHEMBL240`
- Data-quality guidance: [ChEMBL assay and activity questions](https://chembl.gitbook.io/chembl-interface-documentation/frequently-asked-questions/chembl-data-questions)
- Release-specific citation/DOI: [ChEMBL downloads](https://chembl.gitbook.io/chembl-interface-documentation/downloads)
- Licensing: the [ChEMBL about page](https://chembl.gitbook.io/chembl-interface-documentation/about) states CC BY-SA 3.0 for ChEMBL data; EMBL-EBI's [terms of use](https://www.ebi.ac.uk/about/terms-of-use/) also require attention to contributed-data rights and attribution.

Recommended database citation:

> Zdrazil B, Felix E, Hunter F, et al. The ChEMBL Database in 2023: a drug discovery platform spanning multiple bioactivity data types and time periods. *Nucleic Acids Research*. 2024;52(D1):D1180–D1192. [doi:10.1093/nar/gkad1004](https://doi.org/10.1093/nar/gkad1004).

Recommended web-services citation:

> Davies M, Nowotka M, Papadatos G, et al. ChEMBL web services: streamlining access to drug discovery data and utilities. *Nucleic Acids Research*. 2015;43(W1):W612–W620. [doi:10.1093/nar/gkv352](https://doi.org/10.1093/nar/gkv352).

The analysis must also cite the exact ChEMBL release or service status captured by the build. Preserve `research/data/raw/chembl/chembl_status.json` when present.

### BindingDB

- Resource and current recommended publications: [BindingDB information](https://www.bindingdb.org/rwd/bind/info.jsp)
- Downloads and TSV documentation: [BindingDB downloads](https://www.bindingdb.org/rwd/bind/chemsearch/marvin/SDFdownload.jsp?all_download=yes)
- The BindingDB information page distinguishes BindingDB-curated CC BY 3.0 data from imported ChEMBL CC BY-SA 3.0 data. Check the provenance of the actual exported rows and the current provider terms before redistribution.

Recommended current citation:

> Liu T, Hwang L, Burley SK, et al. BindingDB in 2024: a FAIR knowledgebase of protein-small molecule binding data. *Nucleic Acids Research*. 2025;53(D1):D1633–D1644. [doi:10.1093/nar/gkae1075](https://doi.org/10.1093/nar/gkae1075).

BindingDB includes records derived from literature, patents, PubChem, and ChEMBL. Deduplicate by structure, measurement, document, and original provenance rather than assuming each database label is independent.

### PubChem BioAssay

- Programmatic access: [PUG REST](https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest) and [PUG REST tutorial](https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest-tutorial)
- Provider citation guidance: [PubChem citation guidelines](https://pubchem.ncbi.nlm.nih.gov/docs/citation-guidelines)

Recommended current citation:

> Kim S, Chen J, Cheng T, et al. PubChem 2025 update. *Nucleic Acids Research*. 2025;53(D1):D1516–D1525. [doi:10.1093/nar/gkae1059](https://doi.org/10.1093/nar/gkae1059).

Also cite or identify each key BioAssay AID and depositor/source used in a manuscript. PubChem aggregates contributed records; the database citation alone does not describe an assay protocol or confer rights to every contributed element.

### UniProt target anchors

- Menin: [UniProt O00255](https://www.uniprot.org/uniprotkb/O00255/entry)
- hERG/KCNH2: [UniProt Q12809](https://www.uniprot.org/uniprotkb/Q12809/entry)

Use accessions to disambiguate targets, but retain construct, mutation, organism, and complex context from the assay.

## Biological and translational context

These papers motivate the target and help frame a manuscript; they are not automatically the provenance of any activity row. Map a record to a paper, patent, assay, compound, or structure only when the source metadata establishes that link. “MLL” is retained below where it appears in a historical article title; current gene nomenclature is **KMT2A**.

> Grembecka J, He S, Shi A, et al. Menin-MLL inhibitors reverse oncogenic activity of MLL fusion proteins in leukemia. *Nature Chemical Biology*. 2012;8(3):277–284. [doi:10.1038/nchembio.773](https://doi.org/10.1038/nchembio.773).

> Xu S, Aguilar A, Xu T, et al. Design of the first-in-class, highly potent irreversible inhibitor targeting the Menin-MLL protein-protein interaction. *Angewandte Chemie International Edition*. 2018;57(6):1601–1605. [doi:10.1002/anie.201711828](https://doi.org/10.1002/anie.201711828).

> Issa GC, Aldoss I, DiPersio J, et al. The menin inhibitor revumenib in KMT2A-rearranged or NPM1-mutant leukaemia. *Nature*. 2023;615:920–924. [doi:10.1038/s41586-023-05812-3](https://doi.org/10.1038/s41586-023-05812-3).

> Perner F, Stein EM, Wenge DV, et al. MEN1 mutations mediate clinical resistance to menin inhibition. *Nature*. 2023;615:913–919. [doi:10.1038/s41586-023-05755-9](https://doi.org/10.1038/s41586-023-05755-9).

Together, these studies establish biological rationale, chemical tractability, clinical translation, and a known resistance mechanism. They do not validate this repository's endpoint harmonization, model performance, candidate ranking, or clinical utility; those claims require the project-specific evidence described in [methodology](methodology.md) and [limitations](limitations.md).

## Computational methods and software

### RDKit

- Project citation guidance: [RDKit overview](https://www.rdkit.org/docs/Overview.html)
- Molecule standardization API: [RDKit MolStandardize documentation](https://www.rdkit.org/docs/source/rdkit.Chem.MolStandardize.html)
- Version archive DOI family: [doi:10.5281/zenodo.591637](https://doi.org/10.5281/zenodo.591637)

Cite the exact RDKit version/Zenodo release recorded by the model manifest. The RDKit project recommends citing “RDKit: Open-source cheminformatics” with its project URL and version DOI.

### Fingerprints and scaffolds

> Rogers D, Hahn M. Extended-connectivity fingerprints. *Journal of Chemical Information and Modeling*. 2010;50(5):742–754. [doi:10.1021/ci100050t](https://doi.org/10.1021/ci100050t).

> Bemis GW, Murcko MA. The properties of known drugs. 1. Molecular frameworks. *Journal of Medicinal Chemistry*. 1996;39(15):2887–2893. [doi:10.1021/jm9602928](https://doi.org/10.1021/jm9602928).

The implementation uses RDKit Morgan fingerprints and Bemis–Murcko scaffold extraction; it does not claim bit-level identity with every implementation described in the original papers.

### Chemical intelligence and review alerts

> Bickerton GR, Paolini GV, Besnard J, Muresan S, Hopkins AL. Quantifying the chemical beauty of drugs. *Nature Chemistry*. 2012;4:90–98. [doi:10.1038/nchem.1243](https://doi.org/10.1038/nchem.1243).

> Baell JB, Holloway GA. New substructure filters for removal of pan assay interference compounds (PAINS) from screening libraries and for their exclusion in bioassays. *Journal of Medicinal Chemistry*. 2010;53(7):2719–2740. [doi:10.1021/jm901137j](https://doi.org/10.1021/jm901137j).

> Hu X, Hu Y, Vogt M, Stumpfe D, Bajorath J. MMP-cliffs: systematic identification of activity cliffs on the basis of matched molecular pairs. *Journal of Chemical Information and Modeling*. 2012;52(5):1138–1145. [doi:10.1021/ci3001138](https://doi.org/10.1021/ci3001138).

- Official implementation reference: [RDKit `FilterCatalog` documentation](https://www.rdkit.org/docs/source/rdkit.Chem.FilterCatalog.html), including PAINS, Brenk, and NIH catalog enumerations.

QED, catalog alerts, activity cliffs, and matched pairs are implemented as reproducible descriptive diagnostics. Filter matches are review flags, not assay-interference or toxicity determinations; matched-pair transforms are not causal effects. Cite the exact RDKit version because catalog contents and chemistry algorithms are implementation-dependent.

### Approved-reference coverage panel

The versioned analysis configuration currently records the following primary public sources for its dated coverage controls:

- revumenib: [PubChem CID 132212657](https://pubchem.ncbi.nlm.nih.gov/compound/132212657), the FDA [November 15, 2024 KMT2A-translocated approval summary](https://www.fda.gov/drugs/drug-approvals-and-databases/drug-trials-snapshots-revuforj), and the FDA [October 24, 2025 NPM1-mutant AML approval notice](https://www.fda.gov/drugs/resources-information-approved-drugs/fda-approves-revumenib-relapsed-or-refractory-acute-myeloid-leukemia-susceptible-npm1-mutation);
- ziftomenib: [PubChem CID 138497449](https://pubchem.ncbi.nlm.nih.gov/compound/138497449), the FDA [November 13, 2025 oncology approval notice](https://www.fda.gov/drugs/resources-information-approved-drugs/oncology-cancerhematologic-malignancies-approval-notifications), and the [NDA 220305 prescribing information](https://www.accessdata.fda.gov/drugsatfda_docs/label/2025/220305s000lbl.pdf).

Recheck these structures, status statements, indications, labels, and URLs on the release date. The pipeline uses them to benchmark whether approved-reference chemistry is represented in the assembled public Menin evidence; it does not compare clinical efficacy or establish therapeutic equivalence.

### scikit-learn and model persistence

- [Cross-validation guide](https://scikit-learn.org/stable/modules/cross_validation.html)
- [`GroupKFold` API](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html)
- [Probability calibration guide](https://scikit-learn.org/stable/modules/calibration.html)
- [Model persistence security and compatibility](https://scikit-learn.org/stable/model_persistence.html)

Cite the scikit-learn version recorded in the model manifest and the project's recommended publication for that release. The persistence guidance explains why `skops` is preferred and why `joblib`/pickle-derived artifacts must not be loaded from untrusted sources.

### QSAR validation and reporting

- OECD. [Guidance Document on the Validation of (Quantitative) Structure–Activity Relationship Models](https://www.oecd.org/en/publications/guidance-document-on-the-validation-of-quantitative-structure-activity-relationship-q-sar-models_9789264085442-en.html). OECD Series on Testing and Assessment, No. 69.

The five commonly cited OECD principles are a defined endpoint, an unambiguous algorithm, a defined applicability domain, appropriate measures of goodness-of-fit/robustness/predictivity, and mechanistic interpretation where possible. A checklist mapping is in [publication readiness](publication_checklist.md).

### Conformal uncertainty

> Angelopoulos AN, Bates S. Conformal prediction: a gentle introduction. *Foundations and Trends in Machine Learning*. 2023;16(4):494–591. [doi:10.1561/2200000101](https://doi.org/10.1561/2200000101).

The project uses out-of-fold absolute residuals for a symmetric regression interval and reports empirical holdout coverage. It does not claim unconditional coverage under chemical or temporal distribution shift.

## Optional external hERG benchmark

Therapeutics Data Commons exposes a [TDC hERG benchmark](https://tdcommons.ai/benchmark/admet_group/20herg/) and [toxicity-task documentation](https://tdcommons.ai/single_pred_tasks/tox/). It can be useful as an independent sensitivity analysis only after:

1. reconciling its endpoint and label definition with this project's `<=10 µM`/`>=30 µM` task;
2. standardizing structures under the same policy;
3. removing all overlap with training and model-selection data;
4. preserving the benchmark's intended scaffold split; and
5. recording dataset version, license, and source citation.

TDC is not a current core dependency. In the environment tested during this revision, PyTDC's dependency constraints did not resolve cleanly with Python 3.13 and the installed NumPy generation, so use a dedicated compatible environment rather than downgrading the primary release environment. Re-check current package metadata before use. Do not compare leaderboard AUROC directly unless population, label, split, and overlap are matched.

## Source citation record for a build

For each manuscript or archived run, complete this table in the release bundle:

| Source | Release/status | Accessed UTC | Retrieval query/target | Raw manifest digest | Citation/DOI | Redistribution review |
| --- | --- | --- | --- | --- | --- | --- |
| ChEMBL |  |  |  |  |  |  |
| BindingDB |  |  |  |  |  |  |
| PubChem BioAssay | Live service |  |  |  |  |  |
| Any internal dataset | Private version only |  | Approved purpose | Private digest | Access statement | Data-steward approval |

Do not copy private manifest digests, filenames, counts, or dataset versions into a public record without approval.
