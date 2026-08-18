# Evidence review: Miyashita et al. (2024)

## Citation and assets

Miyashita et al., "Improved higher resolution cryo-EM structures reveal the binding modes of hERG channel inhibitors," *Structure* 32 (2024), 1926-1935. DOI: `10.1016/j.str.2024.08.021`.

The canonical local package separates the 14-page article and 10-page supplementary information. The source file `mmc2.pdf` was independently verified as a 24-page concatenation whose first 14 extracted-text pages exactly match the article and whose last 10 exactly match the supplementary file. It is therefore a redundant combined copy, not a third scientific asset.

## Reported structural evidence

Four structures were deposited:

- 8ZYN / EMD-60573: apo hERG;
- 8ZYO / EMD-60574: astemizole-bound;
- 8ZYP / EMD-60575: E-4031-bound;
- 8ZYQ / EMD-60576: pimozide-bound.

All four deposited coordinate models are now local as RCSB PDBx/mmCIF records. Entry/data-block identity, complete `_atom_site` parsing, unique atom IDs, and finite Cartesian coordinates were validated (6,640; 6,238; 6,676; and 6,329 atoms, respectively). The associated EM maps remain nonlocal. These checks establish raw-record integrity only; assembly review, protonation, missing-residue handling, membrane placement, and simulation preparation remain incomplete.

The reported C1 reconstructions are approximately 3.27, 3.29, 3.19, and 3.18 angstrom global resolution, respectively. Use of digitonin is presented as an important experimental change that mitigated preferred particle orientation and enabled improved cavity interpretation.

Across the inhibitor-bound structures, a positively charged or protonatable center occupies the central cavity below the selectivity filter. Interactions with S624 and Y652 recur. T623 and S649 are more ligand-specific. Y652 side chains do not behave as four identical static copies: inhibitor binding can break apparent fourfold symmetry and support alternate rotamers or near-equivalent ligand poses. Astemizole is modeled in two similar poses, which itself is evidence against treating one docked orientation as uniquely known.

F656, historically central in hERG pharmacology, appears to respond partly through indirect packing and allosteric rearrangement rather than always making the same direct ligand contact. That observation is mechanistically important: a mutational effect at F656 need not imply a single direct pairwise interaction in every ligand complex.

## What is mechanistically supported

The structures support a receptor-ensemble account of hERG recognition:

1. a protonation/microstate ensemble supplies ligand species;
2. a membrane-access and cavity-entry process positions the ligand below the filter;
3. the ligand selects and induces combinations of Y652 and neighboring side-chain states;
4. recurrent electrostatic, hydrogen-bond, aromatic, and shape-complementarity interactions stabilize one or more bound poses;
5. remote packing changes, including around F656, can transmit ligand-specific effects;
6. channel state and experimental timing determine whether this occupancy becomes measured block or trapping.

This is substantially richer than a flat count of positive charge and aromatic rings. Those descriptors are useful as priors, but the structural hypothesis lives in distances, orientations, water exposure, residue rotamers, symmetry breaking, and the probabilities of joint ligand-receptor states.

## What is not established

- Four structures and three inhibitor chemotypes cannot determine a universal hERG affinity function.
- Cryo-EM density identifies populated structural states under the preparation conditions; it does not directly provide on-rates, off-rates, residence times, state-dependent access, or trapping probabilities.
- The models do not by themselves establish the physiological gating state distribution under a voltage-clamp protocol.
- A static pose cannot distinguish conformational selection from induced fit without kinetics or perturbational evidence.
- The reported MOE GBVI/WSA dG calculations after Amber10:EHT minimization are scoring approximations. They must not be described as rigorous binding free energies or used as ground-truth energetic labels.
- Local side-chain and ligand-pose uncertainty remains even when global map resolution is high. Map-level validation and alternate-state modeling are required before generating precise interaction features.

## Feature and experiment implications

The immediate structure-aware feature layer should enumerate, for each ligand microstate and each curated receptor state:

- distance/orientation distributions to T623, S624, S649, Y652, and F656;
- cation-pi and aromatic stacking geometry rather than binary contact flags;
- hydrogen-bond occupancy, desolvation demand, and retained water states;
- Y652/F656 rotamer combinations and fourfold-symmetry breaking;
- ligand-pose entropy and near-degenerate pose count;
- cavity fit, strain, and receptor reorganization terms; and
- agreement across the apo and three bound-state templates.

Validation should use ligand-series perturbations and targeted electrophysiology, including residue mutations interpreted cautiously, state- and time-dependent protocols, onset/washout kinetics, and external compounds not used to choose the receptor ensemble. A feature is mechanistically credible when it predicts perturbational direction across multiple ligands and protocols, not merely when it correlates with pooled IC50.

## Project use

These four structures are the canonical cavity component of the bounded six-coordinate hERG ensemble: one recent matched experimental series provides an apo hypothesis and three chemically distinct inhibitor-bound hypotheses at approximately 3.18-3.29 angstrom reported global resolution. The initial implementation can generate transparent interaction hypotheses and compare them with conventional QSAR residuals. Rigorous thermodynamic or kinetic calculations should be reserved for a small, information-rich prospective panel after protonation, receptor-state preparation, and map validation are complete. The structures advance causal modeling, but they do not remove the need for assay context, channel-state physics, or uncertainty.
