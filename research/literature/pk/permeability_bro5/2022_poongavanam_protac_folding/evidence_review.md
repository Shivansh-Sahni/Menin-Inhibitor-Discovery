# Evidence review: Poongavanam et al. (2022)

## Citation and study role

Poongavanam et al., "Linker-Dependent Folding Rationalizes PROTAC Cell Permeability," *Journal of Medicinal Chemistry* 65 (2022), 13029-13040. DOI: `10.1021/acs.jmedchem.2c00877`.

This is the clearest attached precedent that a large molecule's permeability depends on an environment-conditioned conformational ensemble rather than a single 2D descriptor vector. It supplies a mechanistic hypothesis and a very small matched series; it does not supply enough compounds to train or validate a general model.

## Experimental system

The study compares three closely related CRBN/BRD4 PROTACs with common warhead and ligase-binding components but different linker length and chemistry. All lie in bRo5 space. Standard calculated properties do not explain their permeability order: the most lipophilic compound is the least permeable, and the most permeable compound has somewhat higher molecular weight and rotatable-bond count than the others.

Multiple readouts agree on the order 1 > 2 > 3:

- CRBN cellular-to-biochemical binding ratio: 4, 12, and 27, where lower is interpreted as more permeable;
- passive Caco-2 permeability: approximately 30 +/- 1.5, 11 +/- 1.7, and 6 +/- 1.4 nm/s; and
- PAMPA: measured for 1 and 3, with 1 more permeable; 2 was not determined.

Passive Caco-2 permeability is defined as the geometric mean of apical-to-basolateral and basolateral-to-apical apparent permeability. This reduces but does not eliminate transporter and monolayer context. The raw direction-specific values and an internal conflict are documented in `dataset_qc.md`.

## Conformational evidence

### NMR/NAMFIS

NMR is performed in CDCl3 because its dielectric constant is used as a rough proxy for the membrane interior. NOE build-up measurements across seven mixing times and multiple force-field/implicit-solvent conformational searches feed NAMFIS ensemble fitting. Ensembles are resolved for PROTACs 1 and 2; compound 3 lacks enough long- and medium-range NOEs for a comparable ensemble and is inferred to be more extended.

For the S,S stereoisomers, the reported population-weighted ensemble means are:

- PROTAC 1: Rgyr 5.42 angstrom and solvent-accessible 3D PSA 209 square angstrom;
- PROTAC 2: Rgyr 5.58 angstrom and solvent-accessible 3D PSA 246 square angstrom.

PROTAC 1's ensemble is more consistently folded. PROTAC 2 spans folded, semifolded, and nearly linear states. Intramolecular hydrogen bonds, pi stacking, and van der Waals contacts stabilize subsets of the folded states. Compound 3's sparse long-range NOEs are consistent with a more elongated ensemble.

### Molecular dynamics

Each S,S PROTAC is simulated in explicit chloroform in three independent 100 ns replicas after quantum-chemical geometry and charge preparation with GAFF-compatible parameters. The trajectories are analyzed for Rgyr, solvent-accessible 3D PSA, IMHBs, principal moments, PCA clusters, and folding class.

The most populated solvent-accessible 3D PSA regions increase from roughly 190 square angstrom for 1 to 265 for 2 and two regions near 290 and 330 for 3. Higher exposed polarity is inversely associated with IMHB count. Compound 1 primarily samples folded and semifolded states; compound 2 samples a broader mixture; compound 3 is dominated by linear states. The NMR and MD ensembles differ in detailed Rgyr populations, but independently preserve the mechanistic rank: easier access to compact, polarity-masked conformations tracks higher permeability.

### Proposed physical cause

The linker changes more than flexibility. PEG-like bonds favor gauche turns, while the alkyl linker favors extended anti conformations. Linker length controls whether distant groups can meet. IMHB, pi-stacking, and dispersion then stabilize folded states. Folding reduces the polar surface presented to the low-dielectric environment and can reduce the effective dehydration penalty during membrane entry. The relevant causal object is therefore a joint distribution:

`environment -> conformer population -> exposed polarity/shape/IMHB state -> insertion and crossing free energies -> passive flux`.

This is not equivalent to saying that lower average Rgyr is always better. The study itself shows that the relation between folding class and solvent-accessible 3D PSA is complex; conformers with the same coarse fold can expose very different polarity.

## Direct dataset audit

The companion CSV contains three compound rows and 23 columns plus a trailing blank row. It preserves structures, solubility, binding, PAMPA, directional Caco-2, efflux-ratio, and passive-permeability fields. It is useful as a worked example, not a learnable sample.

For PROTAC 1, the PDF supplement reports Caco-2 efflux ratio 309 +/- 72, while the CSV reports 165 +/- 72.3. The underlying direction values are approximately 2.6 +/- 0.40 nm/s AB and 370 +/- 99 nm/s BA; their simple ratio is about 143, so neither published summary is arithmetically recovered from the rounded means. Both provenance-specific records must be retained and the efflux-ratio field quarantined until replicate-level calculations are available.

## Limits

- Sample size is three and all compounds belong to one closely related series.
- NMR ensemble determination fails for compound 3, so part of the mechanistic order is inferred rather than equally observed.
- CDCl3 resembles a membrane interior only in limited dielectric terms; it lacks water-lipid interfaces, lipid hydrogen-bond partners, lateral pressure, and membrane deformation.
- Three 100 ns trajectories do not establish converged folding thermodynamics or transition kinetics for all slow modes.
- The study correlates ensemble features with permeability but does not directly compute membrane insertion/desorption barriers or a transbilayer PMF.
- Directional Caco-2 measurements show strong efflux, making the passive reconstruction dependent on assay assumptions.
- Raw NMR intensities, NAMFIS coordinates, MD systems/trajectories, and replicate-level assay data are unavailable.

## How to extend the mechanism

For each large molecule, generate ensembles in water, a low-polarity solvent, and an explicit membrane/interface. Describe distributions and transitions, not one minimized structure:

- exposed 3D polar surface and its low-tail probability;
- Rgyr, anisotropy, end-to-end distance, and shape-state populations;
- IMHB network occupancy and solvent/lipid competition;
- conformational free-energy cost to reach membrane-compatible states;
- interfacial orientation and partition free energy;
- local water defects and membrane deformation;
- state-to-state transition rates and kinetic bottlenecks.

Test these features first within matched linker series, then across held-out chemotypes. Compare against 2D and generic 3D baselines under nested splits. Mechanistic validation should include environment-sensitive NMR, permeability under multiple membrane compositions, transporter controls, and prospective linker edits predicted to shift a specific ensemble state.

## Bottom line

The paper supports folding-dependent polarity masking as a plausible cause of permeability differences in one PROTAC series. It justifies a conformational physics layer, but not a universal cutoff or a direct claim that compactness alone causes permeability. The project should preserve the joint, environment-dependent ensemble and test its intervention-level predictions on a much larger prospective series.
