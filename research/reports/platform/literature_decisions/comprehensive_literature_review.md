# Comprehensive literature and model decision review

**Review type:** decision-focused rapid evidence review (not a formal PRISMA systematic review)  
**Evidence cutoff and access date:** 2026-08-05  
**Scope:** protein–ligand structure and affinity models; protein and molecular encoders; hERG/QT/PK evidence; leakage and benchmark overlap; licensing, checkpoints, inputs, outputs, compute, and suitability for this platform.  
**Source policy:** only primary papers, official repositories, official model cards, official database documentation, regulator guidance, and source licenses were used. Secondary reviews, vendor comparisons, news, Reddit, and Wikipedia were not used as evidence.

## Executive findings

1. **The most plausible identity of the meeting-referenced “model trained on ~100,000 structures” is RoseTTAFold All-Atom (RFAA).** Its peer-reviewed Science paper reports 121,800 protein–small-molecule structures in 5,662 clusters after 30% protein-sequence clustering, plus 112,546 protein–metal complexes and 12,689 covalently modified structures [S01]. This is a close quantitative and semantic match. Confidence is **moderate-high**, not certain, because the meeting note lacks a name, date, modality, or exact count. AlphaFold 2 is a weaker alternative if “structures” meant the roughly 170,000 PDB structures then available rather than protein–ligand complexes [S06].
2. **Do not make a cofolding model the first production predictor.** A 2026 peer-reviewed evaluation of 2,600 post-cutoff systems concluded that current all-atom cofolding methods largely memorize ligand poses from training data [S14]. PoseBusters had already shown that low RMSD alone does not guarantee chemically or physically valid poses [S15]. Structure predictions should therefore be auxiliary hypotheses with similarity, confidence, and physical-validity annotations—not labels or clinical evidence.
3. **Boltz-2 is the best current open candidate for a later structure-plus-affinity comparator, not an immediately trusted oracle.** It accepts biomolecular complexes and can return structure plus affinity outputs; code and weights are MIT licensed [S07, S08]. However, the current official repository still warns that updated Boltz-2 evaluation and training code are forthcoming, and the headline affinity claims originate in a 2025 preprint [S07, S08]. Its exact training-corpus overlap with this platform's ChEMBL-derived examples must be audited before evaluation.
4. **Use simple, leakage-resistant baselines first.** For molecule-only endpoints such as hERG block and PK/ADME properties, frozen fingerprints plus logistic/ridge/boosted-tree baselines and Chemprop v2.2.3 are appropriate [S25]. For protein–molecule interaction tasks, a sequence-based comparator such as ConPLex can be useful, but it predicts interaction propensity rather than calibrated affinity and its BindingDB checkpoint is overlap-prone [S26].
5. **The current platform does not yet contain a defensible PK or clinical QT task.** The local task registry has large binding tasks, but its `pk_adme` rows are tiny binding-measure subsets (IC50/Ki/Kd/EC50), not genuine PK quantities such as clearance, half-life, AUC, bioavailability, or volume of distribution. It contains 120 hERG binary rows, 137 exact hERG IC50 rows, eight censored hERG rows, and no clinical QT/QTc endpoint. These facts make new source admission and endpoint semantics more important than model selection.
6. **hERG block, QT/QTc prolongation, and torsade de pointes are related but distinct outcomes.** FDA/ICH guidance requires an integrated assessment spanning high-quality hERG work, other ion channels/in silico modeling, in vivo evidence, and clinical exposure–response; a molecule-only hERG classifier cannot support a clinical arrhythmia claim [S29, S30].
7. **Licensing must be evaluated at the individual source and artifact level.** PubChem is an aggregator and explicitly directs users to contributor-specific terms [S38]. AlphaFold 3 weights and outputs have noncommercial and downstream-training restrictions [S05]. PDBbind forbids redistribution without permission according to the primary BindingDB resource paper [S35]. Open code does not imply open training data.

## Review method and limitations

The search covered literature and official artifacts available through 2026-08-05. Each decision-relevant claim was checked against a primary paper, official repository/model card, regulator page, database documentation, or license. Peer-reviewed papers were preferred; recent preprints were retained when they describe models whose official weights are already available. Exact versions were recorded when an official release exposed them. Where a repository's moving `main` branch was the only discoverable reference, it is labeled unpinned.

This is broad and current, but it is not a registered systematic review, does not claim exhaustive recall of every model, and did not reproduce model benchmarks or download multi-gigabyte checkpoints. Reported benchmark values are authors' results, not independently verified results. Licenses were read for research planning only; this review is not legal advice.

## What the current local task registry actually supports

The frozen ChEMBL 37 registry was inspected directly at `research/data/platform/canonical/full_chembl37/task_registry.csv`.

| Domain | Current local evidence | Honest interpretation |
|---|---:|---|
| Binding | Large exact/censored IC50, Ki, Kd, and EC50 task families; largest exact IC50 task has 1,107,075 rows | Strong basis for endpoint-specific retrospective modeling, subject to assay/source/target leakage controls |
| Derived standard binding free energy | 62,334 exact Kd-derived rows in the binding assay family and 5,458 in `other_bioactivity`; tiny other subsets | Valid only as a declared transformation from exact equilibrium Kd under an explicit standard-state/temperature convention; not a new experimental measurement |
| hERG | 137 exact IC50 rows, 120 binary rows, eight censored rows; tiny derived subsets | Far too small for broad hERG claims; likely reflects a strict canonical subset rather than all public hERG evidence |
| `pk_adme` family | IC50 1 row, Ki 15, EC50 4, Kd 2; derived free energy 2 | These are binding-style measures attached to assays classified as PK/ADME, not usable PK endpoints |
| Clinical QT/QTc or torsade | No registered task | No clinical cardiotoxicity prediction claim is currently supportable |

The immediate scientific problem is therefore not “choose the largest model.” It is to create valid endpoint contracts and admit fit-for-purpose evidence without mixing incompatible quantities.

## Identification of the ~100,000-structure model

### Leading identification: RoseTTAFold All-Atom

RFAA is the best match because:

- Its Science paper uses the exact kind of language the note suggests: a general biomolecular structure model trained with **121,800 protein–small-molecule structures**, reduced to **5,662 clusters** at 30% sequence identity [S01].
- It predicts assemblies containing proteins, nucleic acids, small molecules, metals, and covalent modifications [S01, S02].
- Its official code and referenced weights are available under a BSD license that explicitly covers both code and weights [S02, S03].
- The meeting note connects the remembered structure count with “binding affinity and things,” which is consistent with discussing a protein–ligand all-atom model while planning a broader platform, even though RFAA itself is fundamentally a structure model rather than a calibrated affinity predictor.

**Important correction:** RFAA should not be described as “trained on 100,000 structures” without qualification. The paper reports multiple training subsets; the most relevant is 121,800 protein–small-molecule structures, alongside other complex types [S01]. It also requires large sequence/template databases (the official setup lists roughly 46 GB UniRef30, 272 GB BFD, and 81 GB structure templates) and GPU inference [S02].

### Alternatives and why they are weaker matches

- **AlphaFold 2:** its historical training context involved the PDB at roughly 170,000 structures, which someone could round informally to “about 100,000.” But it is primarily a protein-structure model and is a weaker fit to the note's protein–molecule/affinity context [S06].
- **Chai-1 / AlphaFold 3 / Boltz / Protenix:** these train on PDB-scale multimodal structures, but their official materials do not foreground an approximately 100,000-structure count matching the note as closely [S04, S07, S09, S10].
- **PDBbind-family affinity models:** common releases contain tens of thousands, not approximately 100,000, protein–ligand affinity complexes; this is quantitatively less likely [S35].
- **PLINDER:** it contains more than 400,000 protein–ligand systems and is a dataset/evaluation resource, not the likely remembered model [S16].

**Decision:** record RFAA as `probable_meeting_reference`, confidence `moderate_high`, and ask the professor to confirm the name or paper before citing this identification publicly.

## Protein–ligand structure and affinity model landscape

### AlphaFold 3

AlphaFold 3 predicts complexes containing proteins, nucleic acids, small molecules, ions, and modifications [S04]. Version 3.0.2 was the latest official GitHub release found. Its paper used a standard structural cutoff of 30 September 2021 and an earlier 30 September 2019 model for PoseBusters evaluation [S04]. It outputs structures and confidence, **not experimental binding affinity**. Local inference requires a GPU and substantial databases [S05].

It is unsuitable as the platform's core distributable model because, although the current inference source code is Apache-2.0, the model parameters are governed by separate gated terms. Those terms restrict parameter and output use to qualifying noncommercial work, prohibit commercial activity, restrict parameter redistribution, and prohibit using AlphaFold 3 output to train similar biomolecular structure-prediction technology [S05]. It remains a valuable noncommercial comparator if institutional use and output handling are approved.

### RFAA

RFAA is scientifically important and likely the meeting reference. Its strengths are explicit all-atom multimodality, public weights, and permissive BSD licensing [S01-S03]. Weaknesses are an older architecture than current AF3-like models, a heavy local data setup, no native calibrated affinity output, and documented limitations on difficult cases in its own README [S02]. Use it as a historical/provenance comparator, not the default platform backbone.

### Chai-1

Chai-1 v0.6.1 accepts protein/nucleic-acid sequences and ligand SMILES, can use MSAs, templates, restraints, and covalent bonds, and produces structure predictions [S09, S27]. Code and weights are Apache 2.0. The official repository recommends A100/H100 80 GB or L40S 48 GB, while noting smaller complexes can run on A10/A30/RTX 4090 [S09]. It has no native experimental-affinity head and its paper is a technical report/preprint. It is a strong open structure comparator, not an affinity ground truth.

### Boltz-1 and Boltz-2

Boltz code and weights are MIT licensed. Boltz-2 jointly models structure and small-molecule–protein affinity, accepting YAML-described biomolecules with optional MSAs, templates, method conditioning, contacts, and an affinity request [S07, S08]. It returns an affinity value and a binary binding-probability-like output. Inference can technically run on CPU, GPU, or TPU, but the official repository says CPU is significantly slower; defaults include 200 structure sampling steps and five affinity diffusion samples [S08].

Boltz-2 is the most relevant future comparator because it is the only reviewed open all-atom foundation model with a first-party affinity head. It should not be used to generate labels. The 2025 affinity claims are not yet a substitute for independent, post-cutoff, similarity-stratified testing, and the official repository still marks updated Boltz-2 evaluation/training support as forthcoming [S07, S08]. Pin release v2.2.1 plus exact downloaded weight hashes before any run.

### Protenix and OpenFold3-preview

Protenix v2.0.0 provides Apache-2.0 code and weights. `protenix-v2` is a 464.44M-parameter structure model with a documented 30 September 2021 cutoff; the repository also exposes a 2025-06-30 applied model, which must not be used on an evaluation set containing structures through that date [S10]. Protenix predicts structures, not binding affinity. Its explicit cutoffs and permissive artifacts make it a good later structure comparator.

OpenFold3-preview v0.4.1 is Apache 2.0, supplies weights, training code, and a published training-data route, and supports proteins, nucleic acids, and small molecules [S11]. It is still labeled a research preview with the final model under development. This transparency is attractive for method research, but it is not an affinity predictor and should not displace simpler task baselines.

### NeuralPLexer and DiffDock

NeuralPLexer predicts protein–ligand complex structures from protein sequence and ligand graph [S12]. Official published checkpoints are CC BY-NC-SA 4.0 for noncommercial use [S12]. DiffDock v1.1.3 generates ligand poses against protein structures and releases code/weights under MIT [S13]. Neither is a calibrated affinity model. Both are useful as pose-generation comparators only after applying physical-validity checks, similarity stratification, and an unbound/predicted-receptor benchmark rather than privileged holo redocking.

### Generalization warning

Runs N' Poses includes 2,600 high-resolution systems released after common training cutoffs. The peer-reviewed 2026 analysis reports that current cofolding methods largely memorize ligand poses, with accuracy tied to similarity to training systems [S14]. This directly changes the project decision: a high aggregate RMSD success rate on a familiar benchmark is insufficient. Every structural result must carry:

- exact PDB/release-date overlap;
- maximum ligand similarity to the model's training structures;
- maximum protein-sequence and pocket similarity;
- template/MSA provenance;
- confidence and sample rank;
- PoseBusters physical-validity results;
- performance stratified by similarity and novelty.

## Molecular and protein encoders

### Protein encoders

**ESM-2** is the preferred frozen protein representation baseline. The official 650M model card is MIT licensed and lists checkpoints from 8M to 15B parameters [S19]. For a non-HPC start, use 35M or 150M for pipeline validation; use 650M only when the value of larger embeddings is measured. The original Meta repository is archived, so pin the Hugging Face model revision and file hashes rather than relying on a floating name [S18, S19]. ESM-2 embeddings are not structures and do not prove binding.

**ProtT5** is a reasonable sensitivity encoder with readily available checkpoints, but its official repository states the pretrained models use Academic Free License v3.0 [S20]. This should receive a license review before redistribution. It is not clearly superior enough to justify becoming the first dependency.

### Molecular encoders

**Morgan/ECFP fingerprints** remain essential because they are cheap, deterministic, interpretable, and expose whether a deep model adds value [S46, S47]. Fingerprint generation itself is not fitted; any feature filtering, scaling, selection, or downstream model fitting must use training data only.

**Chemprop v2.2.3** is the preferred trainable molecular graph baseline. It implements D-MPNNs, is MIT licensed, supports explicit train/validation/test files and frozen encoders, and is actively maintained [S25]. It should be evaluated per endpoint, with no test-driven hyperparameter search.

**Uni-Mol** provides 3D molecular and pocket representations. The original model used 209M conformations and 3M candidate pockets; Uni-Mol2 scales to 800M conformations and 1.1B parameters [S21, S22]. Code is MIT licensed and weights are public. The original 181 MB molecular checkpoint is practical, but requiring generated 3D conformers introduces seed/conformer-policy sensitivity. Use it as a later frozen-embedding sensitivity analysis, not as the sole molecular representation.

**MoLFormer** is a SMILES transformer pretrained on more than 1.1 billion PubChem/ZINC molecules [S23, S24]. It has broad chemical coverage, but that same coverage creates substantial benchmark-overlap risk and its official repository does not make a crisp code/weight licensing statement in the reviewed README. Do not adopt until code, weights, training corpus versions, and redistribution terms are separately documented.

### Protein–molecule pair encoders

**ConPLex** combines pretrained protein language-model features with molecular representations and contrastive learning. The official repository supplies a BindingDB checkpoint under MIT and accepts protein sequence plus SMILES [S26]. It predicts interaction propensity/co-embedding distance, not Kd/Ki/IC50. Because the checkpoint is trained on BindingDB, it is likely contaminated for any evaluation derived from BindingDB/ChEMBL overlaps unless exact molecule–target pairs and near neighbors are removed. It is a useful throughput comparator, not the platform's affinity authority.

## Binding affinity and binding free energy semantics

The platform must retain **Kd, Ki, IC50, EC50, percent inhibition, and qualitative activity as separate endpoint types**. ChEMBL's pChEMBL field intentionally places several molar potency types on a common negative-log scale, but that convenience does not make the assay quantities mechanistically identical [S37, S48].

For a simple binding equilibrium, the standard Gibbs energy is related to the dimensionless equilibrium constant by the IUPAC relation `K° = exp(-ΔrG°/RT)` [S40]. Consequently, a Kd-derived standard binding free energy can be represented under a declared 1 M standard state as `ΔG°bind = RT ln(Kd / 1 M)`. This transformation is defensible only when:

- the input is an exact equilibrium dissociation constant Kd;
- units are verified and converted to molar;
- temperature is measured or a clearly labeled assumption is recorded;
- relation/censoring is propagated;
- stoichiometry, construct, species, assay conditions, and source remain linked;
- the result is labeled **derived**, not experimental.

Do **not** convert IC50, EC50, or generic potency to ΔG as if they were Kd. Ki can relate to thermodynamics under an appropriate inhibition model but should remain separate unless the mechanistic assumptions are explicit. Boltz-2 affinity output likewise must not be conflated with measured Kd or physical FEP ΔG without model-specific calibration.

## hERG, QT/QTc, and PK evidence

### hERG

The public evidence landscape is much larger than the current 120–137-row local tasks:

- hERGCentral reported electrophysiology screening of more than 300,000 compounds, but the original site later became unavailable and downstream copies need explicit rights/provenance review [S31, S32].
- PubChem AID 588834 is an official qHTS hERG assay record and should be ingested by depositor and assay identity, not treated as generic PubChem truth [S33].
- HERGAI released 224,945 training molecules and a 74,982-molecule test set assembled from PubChem/ChEMBL, with only about 2,000 confirmed blockers in the broader nearly-300,000 set [S28]. The repository is MIT licensed but archived as read-only from 2026; its test is not independent of the same public sources and its structure-based PLEC features require docking poses. It is useful for overlap analysis and a reproducibility comparator, not a clean external validation set.
- TDC's hERG benchmark has only 648 examples and a scaffold split [S34]. It is useful for tool compatibility, not for claiming comprehensive safety prediction.

Required hERG fields include exact compound identity and salt policy; KCNH2 construct; species; cell line; manual versus automated patch clamp or proxy assay; voltage protocol; temperature; exposure time; measured versus nominal concentration; concentration verification; endpoint definition; relation/censoring; replicate count; positive/negative controls; and source version. FDA/ICH specifically emphasizes high-quality patch clamp practice and exposure verification [S29, S30].

The first hERG model should therefore have two separate outputs:

1. exact/censored potency modeling for comparable functional patch-clamp IC50 evidence;
2. threshold classification with a prespecified gray zone and assay-stratified calibration.

Report PR-AUC, recall at a prespecified false-positive rate, calibration, and uncertainty in addition to AUROC because class imbalance is material. Never interpret a hERG prediction as QTc or torsade prediction.

### Clinical QT/QTc and arrhythmia

FDA's integrated framework explicitly combines multiple cardiac ion channels, in silico cardiomyocyte modeling, human induced pluripotent stem-cell cardiomyocytes, and phase-1 clinical ECG biomarkers [S29, S30]. Clinical QT data must therefore be a different evidence tier with different units and context:

- QT correction method (QTcF, QTcB, individualized);
- baseline and placebo correction;
- dose, route, formulation, sampling time, and concentration;
- study design, population, concomitant drugs, electrolytes, and disease status;
- point estimate and confidence interval for exposure–response;
- whether the record is a trial result, regulatory review, label warning, or spontaneous report.

ClinicalTrials.gov v2, Drugs@FDA reviews, and DailyMed v2 are appropriate source interfaces, but names must be normalized through stable identifiers and every extracted claim must retain document section, version/date, and a verbatim evidence span [S41, S42]. ClinicalTrials.gov absence of a reported outcome is not a negative result. DailyMed labels are regulatory text, not randomized labels. FAERS-style spontaneous reports can support signal detection but cannot supply incidence or causal negatives.

The PhysioNet QT Database is openly licensed and contains 105 ECG records with expert waveform annotations [S36]. It is appropriate for validating QT-interval measurement algorithms, **not** for molecule-level drug safety training because it does not provide the required drug-exposure labeling.

### Pharmacokinetics

PK endpoints require distinct tasks by species, route, matrix, dose regime, and quantity. At minimum separate:

- clearance (total, renal, hepatic, intrinsic; observed versus scaled);
- half-life and terminal-phase definition;
- volume of distribution and Vdss;
- AUC, Cmax, Tmax;
- absolute/relative bioavailability;
- fraction unbound/plasma protein binding;
- permeability, solubility, metabolic stability, and transporter endpoints as upstream ADME—not interchangeable clinical PK.

PK-DB is the best reviewed open structured clinical-PK source. It retains studies, subjects/groups, interventions, outputs, time courses, units, errors, and provenance through a REST API [S43, S44]. It also warns that some studies are private or access-limited because of copyright/raw-data constraints [S43]. Admit only public records with a per-study rights decision. TDC can supply small molecule benchmark endpoints (half-life, clearance, Vdss, protein binding), but its scaffold benchmarks are comparison sets, not necessarily source-grade canonical truth [S34].

## Leakage, cutoff, and benchmark-overlap policy

Random row splits are unacceptable for the main claims. MoleculeNet established scaffold splitting as a stronger default than random splitting for molecular property benchmarks [S17], but current evidence shows scaffold splitting alone is insufficient for protein–ligand systems. DataSAIL formalizes similarity-aware one- and two-dimensional splits, including simultaneous drug and target separation [S15A]. PLINDER provides ligand, protein, pocket, and interaction similarities plus curated splits [S16]. Runs N' Poses shows why training-system similarity must be reported explicitly [S14].

### Required split hierarchy

For every task, materialize and freeze the following independently when support allows:

1. **Exact molecule cold:** no standardized parent identity across partitions.
2. **Scaffold/analog cold:** Bemis–Murcko grouping plus explicit nearest-neighbor ECFP similarity bins; acyclic and scaffold-exception molecules remain a separately reported group.
3. **Protein cold:** no protein identity overlap; report maximum train/test sequence identity and coverage.
4. **Target/family cold:** group target concept and relevant protein family, not merely sequence record.
5. **Pocket/interaction cold:** for structure tasks, separate by pocket and shared protein–ligand interaction similarity using PLINDER-style metrics.
6. **Source/assay cold:** keep documents, patents, depositor assays, and laboratory/assay families together.
7. **Temporal:** partition by the earliest defensible public evidence date; do not use current database modification timestamps as experimental dates.
8. **Double cold:** hold out both molecule and protein/target clusters for interaction claims.
9. **Prospective lockbox:** new post-freeze evidence, never used for architecture or threshold selection.

### Pretrained-model overlap ledger

Before a pretrained encoder or structure model is evaluated, record:

- model name and exact release/commit;
- weight filenames, byte sizes, SHA-256 hashes, and download URL;
- license/terms snapshot and access date;
- structural and sequence training cutoff;
- named training corpora and versions;
- exact overlap with evaluation molecule IDs, protein IDs, target pairs, assay records, and PDB entries;
- near overlap by ligand, protein, pocket, and interaction similarity;
- template and MSA databases used at inference;
- whether the evaluation result is admissible, sensitivity-only, or contaminated.

Unknown training data is not “no overlap.” It is **unknown overlap** and limits the claim. Test labels must remain sealed; overlap audits operate on identifiers/structures/sequences and routing metadata, not outcomes.

## Licensing and redistribution decisions

| Resource/artifact | Reviewed terms | Decision |
|---|---|---|
| RCSB PDB structure/API data | CC0 per official policy [S39] | Generally admissible with provenance; preserve citations and source versions |
| ChEMBL | Data presented under CC BY-SA terms in official documentation [S37] | Admissible only with attribution/share-alike plan and versioned source manifest |
| BindingDB-curated data | CC BY 4.0 stated in PLINDER and BindingDB primary materials [S16, S35] | Candidate for admission after record-level origin and overlap checks |
| PDBbind | Redistribution forbidden without permission according to BindingDB's primary 2024 resource paper [S35] | Do not redistribute; obtain written terms before use |
| PubChem | NCBI adds no blanket restriction, but contributor-specific rights may apply [S38] | Admit by depositor/source license; never label the whole aggregate “public domain” |
| UniProt | CC BY 4.0, with patent/other-right caveat [S45] | Admissible with attribution and release pinning |
| AlphaFold 3 | Apache-2.0 inference code; gated, noncommercial parameter/output restrictions [S05] | Comparator only after institutional review; exclude parameters from distributable core |
| Boltz-2 | MIT code and weights [S07] | Candidate after exact release/weight and corpus-overlap audit |
| Chai-1 | Apache 2.0 code and weights [S09] | Candidate structure comparator |
| RFAA | BSD terms explicitly cover code and referenced weights [S03] | Candidate historical structure comparator |
| Protenix | Apache 2.0 code and weights [S10] | Candidate structure comparator; cutoff-specific model selection required |
| OpenFold3-preview | Apache 2.0 [S11] | Research comparator; preserve preview status |
| NeuralPLexer published checkpoints | CC BY-NC-SA 4.0 [S12] | Noncommercial comparator only |
| DiffDock | MIT code and weights [S13] | Candidate pose comparator |
| PLINDER | Mixed notices by derived/source component [S16] | Preserve per-field/source license; do not flatten into one license |
| ClinicalTrials.gov / DailyMed | Official APIs and terms; records contain submitter/regulatory text [S41, S42] | Extract with document provenance; legal review before redistributing large text corpora |

## Compute and operational fit without HPC

The project can make major progress without HPC, but it should not pretend CPU feasibility means practical high-throughput structure inference.

### Feasible now

- endpoint registry and evidence-tier redesign;
- source/right/admission manifests;
- molecule identity, scaffold, assay, target, and temporal overlap analyses;
- ECFP/RDKit/tabular baselines;
- Chemprop single-task and carefully scoped multitask baselines on local GPU or modest cloud GPU later;
- frozen small ESM-2 embeddings in batches;
- metadata-only PLINDER/PK-DB/HERGAI overlap audits;
- calibration, bootstrap confidence intervals, subgroup/error analysis;
- exact checkpoint and benchmark registries.

### Defer until authorized GPU/HPC

- corpus-wide Boltz-2, Chai-1, Protenix, RFAA, OpenFold3, NeuralPLexer, or DiffDock inference;
- training or fine-tuning all-atom structure models;
- 1.1B Uni-Mol2 or multi-billion protein encoder sweeps;
- large MSA/template generation and local database installation;
- FEP campaigns or molecular-dynamics ensembles.

For eventual structure runs, use a small, predeclared benchmark panel first: exact known complexes; recent post-cutoff systems; low ligand/protein/pocket similarity systems; apo/predicted receptor cases; and negative/nonbinding controls where scientifically valid. Measure wall time, peak GPU memory, failure rate, physical validity, confidence calibration, and novelty-stratified accuracy before scaling.

## Model decision summary

1. **Primary molecule-only baseline:** Morgan/ECFP plus regularized linear model, random forest/gradient boosting, and Chemprop v2.2.3. This is the honest starting point for hERG and genuine PK endpoints.
2. **Primary protein representation:** frozen ESM-2, initially 35M/150M for pipeline validation and 650M only if justified by validation under cold-protein splits.
3. **Primary interaction baseline:** simple ligand and protein features with an auditable fusion head; ConPLex as a sensitivity comparator after BindingDB overlap removal.
4. **Future structure-plus-affinity comparator:** Boltz-2 v2.2.1, exact weights pinned, evaluated only on post-cutoff and similarity-stratified lockboxes. Treat outputs as predictions, never labels.
5. **Future structure comparators:** Protenix v2 with the 2021-09-30 cutoff, Chai-1 v0.6.1, DiffDock v1.1.3, and RFAA as the probable meeting-reference model.
6. **Exclude from the distributable core:** AlphaFold 3 and NeuralPLexer checkpoints unless noncommercial restrictions are explicitly compatible with the intended use.
7. **Do not adopt yet:** MoLFormer as a required dependency, any PDBbind-derived redistributed corpus, or an unversioned floating checkpoint.

## What this review does not justify

- It does not justify saying the platform predicts clinical cardiotoxicity.
- It does not justify converting all potency values into binding free energy.
- It does not establish that any foundation-model checkpoint is free of benchmark overlap.
- It does not show that a structure predictor improves affinity or hERG prediction.
- It does not authorize downloading or redistributing restricted data or model artifacts.
- It does not replace prospective experimental validation.

The scientifically defensible near-term paper is a **data-quality, endpoint-semantics, leakage, and evidence-tier study with transparent baselines**, not a claim that a universal drug-discovery foundation model has already been built.
