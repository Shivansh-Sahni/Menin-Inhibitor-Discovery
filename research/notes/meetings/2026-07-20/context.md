# Menin-Edit meeting context - 2026-07-20

## Provenance note

This transcription was reconstructed from the meeting-note screenshot supplied in the conversation. The original screenshot path was a temporary macOS capture location and was no longer readable when the literature package was organized, so no binary image was copied. Spelling has been normalized only where the intended term was clear; ambiguous phrases remain paraphrased rather than invented.

A detailed user-supplied text summary was reviewed on 2026-07-22 and corroborated the
same direction: decompose PK into solubility, permeability, metabolism, distribution, and
other ADME processes; treat folding, solvent response, and chameleonicity dynamically;
use receptor-state physics for hERG; retain standard descriptors only as controls; and
automate the workflow without substituting descriptor volume for mechanistic evidence.

## Direction agreed in the notes

- Pause Menin-Edit-specific work and prioritize PK and hERG.
- Emphasize physics and underlying processes rather than superficial correlations.
- Treat conventional QSAR as a baseline with poor expected out-of-class behavior, not as the final scientific answer.
- For PK, decompose solubility, permeability, metabolism, distribution, and other physicochemical/biological processes instead of relying on one structure-to-exposure mapping.
- Reuse the current dataset where defensible, but improve the model with substantially more meaningful features.
- Investigate molecular fields, solvent response, shape, and conformational behavior.
- A protein/receptor model may transfer across compound classes if the relevant binding receptor and state are represented appropriately.
- PK and hERG are the first targets.
- Mouse/rodent PK is comparatively feasible to obtain but existing coverage is sparse.
- Focus on large molecules because conventional sub-500-Da space is already heavily studied. The notes mention roughly 650, 700, and related candidate cutoffs but explicitly call for finding a meaningful boundary rather than assuming one.
- A paper/chapter from Dr. Aguilar was expected.
- Use molecular dynamics and other methods to understand how a molecule folds and changes shape.
- Chameleonic/adaptable molecules may present differently in hydrophilic and hydrophobic environments.
- Go far beyond basic RDKit physicochemical descriptors and automate the resulting workflow.
- Determine how to improve the model by adding causally meaningful parameters grounded in literature and physics.
- A central unresolved question is folding in hydrophilic versus hydrophobic environments.
- Different physical processes may need different models; PK should be divided into layers and each process understood separately.
- For hERG, review new publications and define how the work will improve on existing models.
- Positive charge and aromatic rings are useful starting descriptors only when connected to receptor interactions and state-specific physics.
- The eventual design problem must balance PK and hERG.

## Follow-up communication visible in the supplied screenshot

A related email thread titled "Menin Project Direction Idea" records that a Dropbox folder containing the then-current internal and external project data was shared with Dr. Aguilar. The message asked that additional useful material be placed in the Internal Data folder and requested the paper and chapter discussed during the meeting. The screenshot itself was in an expired temporary capture location and could not be copied, but these action items are preserved here for continuity.

## Research interpretation

The notes define a process-resolved research program, not permission to maximize model complexity without validation. The practical standard is: every added state or parameter must correspond to a physical hypothesis, expose a failure mode, support a sensitivity or intervention test, and carry uncertainty when the current data cannot identify it.

“Important” is not assigned merely because a process appears in the graph. A causal
process may be negligible for a particular molecule. Its importance must be learned from
rate limitation, sensitivity, reactive flux, or a controlled perturbation. Stored proxies,
assay observations, baseline descriptors, and QC quantities are not relabeled as
fundamental parameters.

The literature package created from the subsequent attachments directly addresses the named themes:

- Poongavanam: environment-dependent PROTAC folding, exposed polarity, and permeability;
- Qi: microstate-weighted membrane crossing and membrane deformation;
- Mavroudis: learned PK inputs coupled to compartmental/PBPK equations;
- Miyashita: inhibitor-bound hERG cavity ensembles;
- Lau: potassium-dependent hERG filter states; and
- Sun: conventional hERG prediction and a public-data comparator.
