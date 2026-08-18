# Decision recommendations: maximum useful work before HPC

**Decision date:** 2026-08-05  
**Boundary:** no substantive large-model training, no mass structure inference, no clinical claims, and no restricted-data redistribution.

## Decisions to lock now

### P0 — scientific meaning and intended use

1. **Declare the product a research decision-support platform.** Initial outputs rank evidence and generate testable hypotheses. They are not diagnoses, clinical safety determinations, dosing advice, or substitutes for hERG/QT/PK experiments.
2. **Keep endpoint contracts separate.** Kd, Ki, IC50, EC50, percent inhibition, binary activity, PK quantities, QT/QTc, and clinical arrhythmia outcomes cannot share one label column merely because all can be numeric.
3. **Restrict binding-free-energy derivation to exact Kd.** Use `delta_g_standard_kcal_mol = R*T*ln(Kd_molar)` with a declared 1 M standard state and recorded temperature or a clearly labeled temperature assumption. Propagate uncertainty and provenance. Do not transform IC50/EC50/generic potency; keep Ki separate unless a mechanistic policy is approved.
4. **Adopt five evidence tiers:**
   - T0: raw public/source records;
   - T1: curated experimental in vitro evidence;
   - T2: preclinical/in vivo or regulator-reviewed nonclinical evidence;
   - T3: clinical biomarker/exposure–response evidence such as QTcF;
   - T4: clinical event, approved-label, or postmarket signal evidence.
   Predictions and derived values must never be silently promoted to an experimental tier.

### P0 — data gaps exposed by the local registry

1. **hERG:** investigate why the canonical task has only 120 binary and 137 exact IC50 rows while primary public resources describe hundreds of thousands of screened compounds. Preserve assay modality and do not bulk-import proxy/low-quality negatives.
2. **PK:** replace the current pseudo-PK task concept. The registered `pk_adme` tasks contain only tiny binding-style endpoints and no genuine PK quantity. Create separate schemas for clearance, intrinsic clearance, half-life, Vd/Vdss, AUC, Cmax, Tmax, bioavailability, fraction unbound, permeability, solubility, and metabolic stability, with species/route/matrix/dose.
3. **QT/QTc:** create a clinical evidence schema before extraction. Required fields include correction method, baseline/placebo adjustment, dose, route, exposure, sampling time, study population/design, point estimate, interval, source section, and document version.
4. **External admission order:** PK-DB public studies; source-specific PubChem hERG assays; ChEMBL hERG re-query; regulator-reviewed Drugs@FDA/DailyMed evidence; ClinicalTrials.gov results. Use HERGAI/TDC/PLINDER first for overlap and benchmark metadata, not automatic canonical labels.

### P0 — leakage and checkpoint governance

1. Freeze exact molecule, scaffold/analog, protein, target/family, source/assay, temporal, and double-cold splits independently.
2. Add protein-sequence, ligand-fingerprint, pocket, and shared interaction similarity bins. Report performance versus maximum training similarity instead of one aggregate result.
3. Use earliest defensible evidence dates for temporal splitting. Database modification dates are not experimental dates.
4. Keep the prospective test lockbox label-sealed. Identifiers and sequences may be used for overlap routing; outcomes may not be used for model selection.
5. Create one checkpoint record per model with release/tag/commit, source URL, weight filenames, sizes, SHA-256, terms snapshot, access date, structural/sequence cutoff, named corpora, templates/MSAs, and overlap results.
6. Treat undisclosed training data as `overlap_unknown`, not `overlap_absent`.

## Baseline model order

### Stage 1 — CPU/modest hardware

1. ECFP/Morgan + regularized logistic/ridge models.
2. Random forest or gradient boosting with fixed, train-only preprocessing.
3. Chemprop v2.2.3 per endpoint, with the exact frozen split files supplied directly.
4. Frozen ESM-2 35M or 150M embeddings plus a small fusion head for protein–molecule tasks; advance to 650M only if cold-protein validation improves materially.
5. Censored regression or interval-aware likelihoods for censored tasks. Never coerce `<`, `>`, or intervals into exact values for the primary analysis.

### Stage 2 — limited authorized GPU pilot

1. ConPLex only after exact and near BindingDB overlap analysis; interpret output as interaction propensity.
2. Uni-Mol original 181 MB molecular encoder as a conformer-sensitive representation check.
3. A small structure panel using Boltz-2 v2.2.1, Protenix-v2 with the 2021-09-30 cutoff, Chai-1 v0.6.1, DiffDock v1.1.3, and RFAA.
4. Every predicted pose must pass PoseBusters-style chemical/physical checks and be stratified by ligand/protein/pocket similarity.

### Stage 3 — defer until HPC authorization

- corpus-scale cofolding or docking;
- all-atom model training/fine-tuning;
- billion-parameter encoder sweeps;
- local full MSA/template database builds;
- FEP or molecular-dynamics campaigns.

## Model decisions

| Model/family | Decision | Reason |
|---|---|---|
| RFAA | Preserve as probable meeting reference; later comparator | Exact reported 121,800 protein–small-molecule structures is the strongest ~100k match; permissive weights; structure-only and operationally heavy |
| Boltz-2 v2.2.1 | Preferred later structure+affinity comparator | MIT code/weights and native affinity head; still requires independent post-cutoff validation and overlap audit |
| Protenix-v2 | Preferred open structure comparator | Apache 2.0, explicit 2021 cutoff, current model; no affinity output |
| Chai-1 v0.6.1 | Secondary structure comparator | Apache 2.0 code/weights and flexible inputs; requires substantial GPU and has no affinity head |
| AlphaFold 3 v3.0.2 | Noncommercial comparator only | Apache-2.0 inference code, but gated parameters and restrictive output terms; no native affinity; exclude parameters from distributable core |
| OpenFold3-preview v0.4.1 | Methods/reproducibility comparator | Apache 2.0 and unusually transparent training path; still a preview and structure-only |
| DiffDock v1.1.3 | Pose comparator | MIT; no calibrated affinity; evaluate on apo/predicted receptors, not only holo redocking |
| NeuralPLexer | Noncommercial sensitivity only | Published checkpoints are CC BY-NC-SA 4.0 |
| ESM-2 | Primary frozen protein encoder | MIT checkpoints across sizes; pin Hugging Face revision because original repository is archived |
| Chemprop v2.2.3 | Primary learned molecule baseline | Active, MIT, auditable, compatible with fixed external splits |
| ConPLex | Interaction sensitivity comparator | Fast and open, but BindingDB overlap risk and output is not affinity |
| Uni-Mol | Later molecule/pocket representation sensitivity | Public MIT weights; 3D conformer dependence adds preprocessing uncertainty |
| MoLFormer | Hold | Huge pretraining overlap risk and reviewed official README did not make all artifact terms sufficiently explicit |

## Evaluation contract

### Regression

- Primary: MAE plus bootstrap confidence interval.
- Secondary: RMSE, Spearman, Pearson only when scientifically appropriate.
- Report per target, assay family, source, evidence tier, relation type, and similarity bin.
- Compare against mean/median, nearest-neighbor, and simple molecular baselines.

### Classification

- Primary for imbalanced hERG: PR-AUC and prespecified recall/specificity operating points.
- Secondary: AUROC, balanced accuracy, MCC, Brier score, calibration slope/intercept, and reliability plots.
- Choose thresholds using validation data only; freeze before test.

### Structure

- Pose RMSD and LDDT-PLI are insufficient alone.
- Add PoseBusters physical validity, stereochemistry, clashes, interaction recovery, pocket accuracy, confidence calibration, and failure rate.
- Report performance against training-system similarity and inference receptor state.

### Uncertainty and robustness

- Bootstrap by the unit of intended generalization (molecule/scaffold/target/source), not rows.
- Repeat model seeds where stochastic.
- Run label permutation and identifier-only controls.
- Test sensitivity to replicate aggregation, salts/tautomers, censor handling, and disputed records.
- Record abstention behavior for low-support/OOD cases.

## Licensing actions before public release

1. Add a top-level repository license selected by the project owner/institution.
2. Create a source-by-source rights manifest; do not label all PubChem data public domain.
3. Preserve ChEMBL attribution/share-alike obligations, UniProt CC BY attribution, RCSB PDB provenance, and component-level PLINDER notices.
4. Do not redistribute PDBbind without written permission.
5. Do not place AlphaFold 3 weights in shared storage or use its outputs to train similar structure models.
6. Review publication PDFs, label text, and trial narratives separately from structured facts; an accessible document is not automatically redistributable training text.

## Paper-ready analyses that do not need HPC

1. Quantify endpoint disagreement for molecules measured by multiple assay types or sources.
2. Model missingness and selection into high-quality assays, preclinical evidence, and clinical reporting.
3. Measure performance decay across scaffold, ligand-similarity, protein-identity, target-family, source, and time strata.
4. Compare exact-only, censor-aware, and naive censor-coercion analyses to demonstrate bias.
5. Compare molecule-only models against protein-aware baselines; if protein features do not improve cold-target performance, report that honestly.
6. Audit hERG-to-QT mismatches without treating either as the other's label.
7. Produce calibration and abstention analyses for OOD groups.
8. Publish the evidence-tier and provenance framework as a reusable contribution.

## Go/no-go gates before any substantive training

Training remains **NO-GO** until all are true:

- intended use and prohibited claims approved;
- task semantics and minimum support thresholds approved;
- source rights and canonical admission decided;
- leakage thresholds and lockboxes frozen;
- exact checkpoints and training-corpus overlap documented;
- compute budget, monitoring, checkpoint/resume, and responsible-compute plan approved;
- external/prospective validation design approved;
- repository license and release storage plan in place.

The best near-term scientific outcome is a rigorous, reusable data and evaluation paper. Large-model training should follow evidence quality—not substitute for it.
