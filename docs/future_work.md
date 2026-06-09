# Future Work Roadmap

## Phase 1: Make the Dataset Lab-Grade

- Add internal Wang lab compound IDs and structures as SDF/SMILES.
- Add exact assay names, assay dates, batch IDs, and curve-quality flags.
- Harmonize public and internal endpoint names.
- Decide official activity priority when multiple measurements exist for the same compound.
- Preserve censored measurements, but model them separately or with censoring-aware methods.

## Phase 2: Upgrade Menin Activity Modeling

- Install RDKit and add Morgan fingerprints, MACCS keys, and physicochemical descriptors.
- Evaluate scaffold split, time split, and random split side by side.
- Train endpoint-specific models for biochemical IC50, Kd/Ki, and cellular EC50 if enough labels exist.
- Add conformal prediction or model ensembles for uncertainty.
- Track applicability domain so the model can flag out-of-distribution proposed compounds.

## Phase 3: Build ADMET and hERG Triage

- Confirm the lab's hERG threshold and preferred assay format.
- Add experimental hERG results for internal Menin compounds if available.
- Split hERG data by assay type where possible.
- Turn PK/ADMET observations into endpoint-specific models only where each endpoint has enough consistent rows.
- Candidate PK endpoints: microsomal clearance, hepatocyte clearance, solubility, permeability, plasma protein binding, half-life, and bioavailability.

## Phase 4: Use Models Prospectively

- Score internal and proposed Menin analogs for activity and hERG risk.
- Create a rank table balancing potency, hERG probability, and observed/predicted PK liabilities.
- Add nearest-neighbor explanations: similar public compounds, source assays, and measured endpoints.
- Add active-learning suggestions for which compounds would be most informative to synthesize or test next.

## Phase 5: Repository and Collaboration Polish

- Add GitHub Actions for unit tests and pipeline smoke tests.
- Add `data/external/` instructions for internal files that should not be committed.
- Add a small Streamlit or notebook dashboard for assay filtering and compound triage.
- Add reproducible environment files once the lab chooses conda, pip, or Docker.
