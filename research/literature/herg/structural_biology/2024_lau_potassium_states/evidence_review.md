# Evidence review: Lau et al. (2024)

## Citation and scientific question

Lau et al., "Potassium dependent structural changes in the selectivity filter of HERG potassium channels," *Nature Communications* 15 (2024), 7470. DOI: `10.1038/s41467-024-51208-w`.

This paper asks how extracellular potassium occupancy changes the hERG selectivity filter and stabilizes nonconducting states. It is directly relevant because the receptor seen by a blocker is not one fixed structure: ion occupancy, filter conformation, gating, and voltage can alter the accessible binding environment and the measured current response.

## Structural states and deposited assets

The study reports high-potassium (300 mM) and low-potassium (3 mM) conditions in both C4 and C1 reconstructions:

- 9CHP / EMDB-45597: high K+, C4;
- 9CHQ / EMDB-45598: low K+, C4;
- 9CHR / EMDB-45599: high K+, C1;
- 9CHS / EMDB-45600: low K+, C1.

The article also references prior structures 5VA1 and 5VA3. Reported global resolutions are approximately 3.3 and 3.0 angstrom for the C4 high- and low-potassium maps and 3.5 and 3.4 angstrom for the corresponding C1 reconstructions.

Only 9CHP and 9CHQ are retained as canonical local RCSB PDBx/mmCIF records. They provide the matched C4 high-/low-potassium contrast at the best reported resolution in this study, and each contains 19,104 atoms across author chains A-D in one model. Both passed entry/data-block identity, complete `_atom_site` parsing, unique atom-ID, and finite-coordinate checks. The lower-resolution C1 models 9CHR and 9CHS are intentionally nonlocal and excluded from routine modeling because adding correlated symmetry-relaxed reconstructions would multiply receptor hypotheses without independent chemical evidence. They are reserved only for a preregistered asymmetry-sensitivity analysis. The associated EM maps remain nonlocal, and neither retained coordinate has undergone ion-occupancy, missing-residue, protonation, membrane, or force-field preparation.

The meeting screenshot records the article's data-availability statement and these accession codes. The publisher-hosted supplementary information, file description, movie, reporting summary, and source-data archive have now been acquired and registered. The source archive contains 72 raw Axon Binary Format (`.abf`) electrophysiology files plus a seven-sheet summary workbook covering WT hERG and five mutant conditions. It is experimental source evidence for the Figure 2 mechanism, not a molecular-simulation reproducibility package. `Data Sheet 1.pdf` remains correctly assigned to the Mavroudis PK paper and is not Lau source data.

## Proposed filter mechanism

The combined structures and simulations support the following sequence:

1. potassium departure changes selectivity-filter occupancy;
2. the V625 carbonyl can flip away from a conductive arrangement;
3. S620 helps stabilize the flipped configuration;
4. multiple subunits can enter flipped states;
5. three or four coordinated flips are proposed to produce a strongly nonconducting filter.

The calculations report barriers on the order of 4-5 kcal/mol for relevant carbonyl-flip transitions and small free-energy differences, generally under about 1-2 kcal/mol depending on ion occupancy and state definition. These energy scales imply an ensemble rather than a permanently locked conformation. Modest changes in ions, force field, voltage, ligand binding, mutation, or membrane environment could shift state populations materially.

## Simulation evidence

The reported workflow uses NAMD, CHARMM36 lipids, and a CHARMM22 protein description. The paper and acquired supplementary information describe ten 50 ns simulations for each of nine ion configurations, paired 500 ns conduction simulations, a 72-window umbrella-sampling calculation, and two 500 ns, 16-replica REST2 simulations. WHAM and MBAR are used for free-energy reconstruction/analysis.

This is substantially more mechanistic than representing hERG as a single crystal-like pocket. It provides candidate state coordinates:

- potassium occupancy pattern;
- number and identity of flipped V625 carbonyls;
- S620-V625 stabilizing geometry;
- C1 versus C4 symmetry;
- filter hydration and conduction state; and
- transition barriers among those states.

These variables can condition structure-based ligand features and can also act as latent states in an electrophysiology observation model.

## Limits and unresolved causes

The paper does not settle the rate-limiting step for filter inactivation or recovery. It does not conclusively establish whether three flipped subunits are sufficient or four are required under physiological voltage and ionic conditions. Force-field choice, finite sampling, applied voltage, membrane composition, construct, and map interpretation can change relative state stability. The simulations describe selected pathways and collective variables; unmodeled pathways can exist.

The reported low barriers and small state free-energy differences make point estimates fragile. A downstream hERG model should therefore propagate receptor-state uncertainty instead of selecting one high-K or low-K structure as truth. It should also distinguish:

- equilibrium filter-state populations;
- transition kinetics between filter states;
- ligand affinity within each state;
- ligand-induced changes to transition barriers; and
- the voltage-clamp protocol that maps these processes to observed current.

The study is not a ligand-bound hERG liability model. Any claim that a compound prefers a filter state is a prospective hypothesis until supported by state-dependent binding, electrophysiology, or perturbational experiments.

## Implications for the hERG physics layer

The bounded six-coordinate receptor ensemble should cross the four Miyashita cavity structures with the two retained Lau C4 filter conditions rather than treating them as unrelated papers or mechanically multiplying every deposited reconstruction. A tractable initial state grid is:

`ligand microstate x cavity side-chain state x potassium/filter state x gating/protocol state`.

For each grid point, compute or estimate transparent quantities: accessibility, steric compatibility, electrostatic potential, hydration/desolvation, interaction occupancy, receptor strain, and uncertainty. A learned model can then combine these state-specific features through physically constrained population weights. Later high-cost calculations can target only states with substantial posterior mass or high decision value.

Prospective validation should vary extracellular potassium, pulse timing, holding potential, and washout while keeping chemistry matched. If a proposed filter-dependent feature is causal, it should predict not only pooled IC50 but directional changes under those perturbations.

## Bottom line

Lau et al. supply essential evidence that hERG liability is conditioned by a dynamic potassium-dependent filter ensemble. The work motivates explicit receptor-state variables and perturbational validation. Publisher-hosted source evidence and the selected 9CHP/9CHQ coordinate pair are local; 9CHR/9CHS are intentionally excluded/nonlocal, while EM maps, prepared simulation systems, trajectories, numerical free-energy outputs, and analysis code remain absent. The work therefore does not yet support independently reproduced quantitative filter-state energetics as model inputs.
