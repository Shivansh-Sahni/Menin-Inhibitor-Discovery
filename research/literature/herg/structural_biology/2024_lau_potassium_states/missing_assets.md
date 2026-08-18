# Missing assets: Lau et al. potassium-dependent hERG states

## Attached now

- Main article PDF.
- Supplementary information PDF.
- Supplementary file description and Supplementary Movie 1.
- Nature Portfolio reporting summary.
- Publisher source-data archive containing 72 raw `.abf` electrophysiology files and a seven-sheet summary workbook for WT hERG and five mutants.
- Deposited RCSB PDBx/mmCIF coordinate models 9CHP and 9CHQ, selected as the canonical matched C4 high-/low-potassium pair and validated at the entry and coordinate-table levels.
- The reported lower-resolution C1 models 9CHR (3.5 angstrom) and 9CHS (3.4 angstrom) are deliberately excluded and nonlocal; they are optional asymmetry-sensitivity assets, not missing routine inputs.
- Project-direction screenshot quoting the article's data-availability section, stored with the dated meeting notes.

The archive passed an integrity test. The workbook rendered without formula errors; it contains fitted electrophysiology summaries and per-recording values, but sparse labels and implicit units make the article/SI essential for interpretation. The movie is a qualitative visualization rather than a simulation trajectory suitable for reanalysis.

## Public repository assets not yet local

1. EMDB maps 45597 and 45598 for the selected 9CHP/9CHQ pair.

The selected map accessions are public but have not been downloaded into this literature package. EMDB-45599/45600 should be acquired only if a specific C1-asymmetry sensitivity is justified. The local coordinate models are raw repository evidence, not prepared systems; article attachments, deposited coordinates, maps, and computational inputs remain distinct asset classes.

## Reproducibility assets requiring author contact or repository search

- prepared protein/membrane systems and protonation choices;
- CHARMM parameter/topology versions and any modifications;
- starting ion configurations;
- production trajectories;
- 72 umbrella windows, restraints, and convergence histories;
- REST2 inputs and replica-exchange histories;
- WHAM/MBAR numeric outputs and uncertainty calculations;
- conduction simulation inputs;
- modified or in-house analysis code; and
- bulk supporting data described as available on request.

## Why these matter

The proposed mechanism depends on small state free-energy differences and barriers of only several kcal/mol. The acquired experimental source data make the mutant electrophysiology auditable, but without numerical simulation profiles and provenance it is not possible to determine how much of a computed population shift arises from sampling, collective-variable choice, force field, ion configuration, or the physical mechanism itself.

## Retrieval order

1. Acquire and validate EMDB-45597/45598 only if map-level refinement of the selected C4 pair is prioritized.
2. Request simulation inputs, free-energy outputs, trajectories, and analysis code.
3. Reproduce one high-K/low-K contrast before expanding to a ligand-conditioned receptor ensemble.
