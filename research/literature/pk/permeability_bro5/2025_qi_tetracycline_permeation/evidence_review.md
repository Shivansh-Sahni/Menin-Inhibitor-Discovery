# Evidence review: Qi, Chipot, and Wang (2025)

## Citation and project role

Qi, Chipot, and Wang, "Probing Passive Permeation of Tetracycline: Are Simulations Ready for beyond-Rule-of-Five Drug Permeability Calculation?" *The Journal of Physical Chemistry B* 129 (2025), 10810-10823. DOI: `10.1021/acs.jpcb.5c05445`.

This paper provides the deepest attached physical decomposition of passive permeability. Its central lesson is not a new scalar descriptor: microstate populations, translocation free energy, diffusivity, membrane remodeling, and sampling coordinates interact nonlinearly. It is also explicit that numerical agreement with an assay does not guarantee a correct mechanism.

## Molecular states and population model

Tetracycline has three ionizable groups, four charge macrostates (+1, 0, -1, and -2), and eight enumerated microstates. Two zero-net-charge tautomers are simulated explicitly:

- `TCN`: neutral at each site; and
- `TCZ`: internally charge-separated zwitterion.

An alternate zwitterion, `TCZ'`, and the charged macrostates are not simulated separately; their specific permeabilities are approximated or bounded using TCZ.

At pH 6, the reported fractions are:

| State | Fraction |
|---|---:|
| +1 macrostate | 2.10e-3 |
| TCZ | 9.65e-1 |
| TCN | 1.40e-3 |
| TCZ' | 1.43e-2 |
| -1 macrostate | 1.74e-2 |
| -2 macrostate | 4.30e-6 |

Thus TCN is only about one-thousandth as abundant as TCZ in bulk solution. Effective permeability is assembled with both pH-partitioning and Boltzmann-weighted-average-potential schemes. The two schemes agree numerically here because one rare state overwhelmingly dominates crossing.

## Simulation workflow

Systems use POPC bilayers built at 32, 64, 128, and 256 lipids; 128 POPC is the default. Simulations use NAMD 3, CHARMM36 lipids, TIP3P water, tetracycline parameters from Aleksandrov and Simonson, 310 K, and 1.01325 bar.

The membrane-normal center-of-mass coordinate is sampled with well-tempered metadynamics extended-system adaptive biasing force (WTM-eABF) in three overlapping windows with 0.2 angstrom bins. Inverse-PMF ABF trajectories flatten the free-energy surface for inference of position-dependent diffusivity with DiffusionFusion. The inhomogeneous solubility-diffusion equation integrates the PMF and diffusivity into a state-specific permeability.

Reported WTM-eABF sampling is approximately 5, 9, 8, and 6.8 microseconds for neutral tetracycline in the 32, 64, 128, and 256 POPC systems, respectively, and 9 microseconds for the zwitterion in 128 POPC. Additional inverse-PMF diffusivity trajectories are 4 and 2 microseconds for TCN and TCZ. Convergence checks include last-approximately-0.5-microsecond PMF changes under 0.5 kcal/mol, leaflet asymmetry around or below 1 kcal/mol, and PMF-gradient RMSD.

## Quantitative mechanism

In 128 POPC, TCN and TCZ have similar diffusivity profiles but radically different PMFs:

- TCN barrier: about 4.4-4.5 kcal/mol;
- TCZ barrier: 13.6 kcal/mol;
- TCN specific permeability: 2.22e-2 cm/s;
- TCZ specific permeability: 1.63e-8 cm/s.

The approximately 9.1 kcal/mol barrier difference produces about six orders of magnitude difference in state-specific permeability because free energy enters exponentially. At pH 6, the combined effective permeability is 3.17e-5 cm/s (log10 = -4.50). This is close to a reported PAMPA value of 1.8e-5 cm/s, but about 1.36 log units above a POPC-liposome fluorescence result of log Papp = -5.86.

The physical interpretation is state-specific:

- TCN behaves approximately amphiphilically, adopts a more horizontal interfacial orientation, flip-flops during crossing, carries about 3-5 nearby waters, and has a favorable interfacial well around -4.9 kcal/mol.
- TCZ adopts a more vertical orientation to expose its separated charges, retains about 6-8 waters, and pays a much larger cost to transport charge and solvation into the bilayer core.

Although TCN is rare, its specific permeability is so much higher that it contributes essentially all effective flux. Mean formal charge or dominant-microstate descriptors would miss the mechanism.

## Membrane patch size is part of the physics

For TCN, the reported barrier changes strongly with patch size: 8.1, 7.0, 4.4, and 2.2 kcal/mol for 32, 64, 128, and 256 POPC. Methanol, urea, and 2',3'-dideoxyadenosine controls vary by at most about 9% across the same sizes.

The smallest bilayer is too stiff to supply the deformation, water, and lipid hydrogen-bond partners that compensate tetracycline's polarity, so it produces an artificially high barrier. The largest bilayer deforms readily but introduces serious hysteresis: a global membrane-normal coordinate does not uniquely identify the ligand's local position when the two leaflets undulate or cave in. Membrane deformation becomes an additional slow degree of freedom. Longer simulation along the same degenerate coordinate is not guaranteed to fix the problem.

The 64-POPC calculation illustrates the resulting uncertainty. Substituting its TCN permeability into the population model gives effective permeability 3.72e-7 cm/s (log10 -6.43), far below the 128-POPC estimate and nearer the liposome result. A single "converged" PMF is therefore not sufficient evidence unless collective-variable completeness and finite-size effects are challenged.

## Critical uncertainties

### Microscopic pKa

The rare-neutral-state weight comes from microscopic pKa values inferred using analog compounds, not direct microstate-resolved tetracycline measurements. Alternate analog choices shift log effective permeability from -4.50 toward -4.65, -4.83, or -5.06. Incorrectly assigning macroscopic pKa values to individual sites yields approximately -6.07 or -7.55, errors up to three orders of magnitude.

### State and force-field approximations

Only two microstates are explicitly simulated. Other states are proxied to TCZ. The model uses fixed protonation and fixed-charge force fields, not constant-pH dynamics or explicit electronic polarization. It uses POPC rather than the decane/soy-lecithin PAMPA membrane and cannot assume assay equivalence.

### Domain

Tetracycline has molecular weight about 444.4 and only one formal Lipinski-rule violation. It is a valuable borderline-bRo5 stress test, not direct validation for compounds above 650-750 Da. The rare-state mechanism may be general or tetracycline-specific.

### Data availability

The attached SI gives convergence figures and sensitivity tables but not raw PMFs, diffusivity profiles, prepared systems, force-field files, trajectories, input configurations, or analysis code.

## Model implications

The physics layer should compute a distribution over pathways, not one permeability descriptor:

1. enumerate plausible tautomers/protomers with uncertain microscopic pKa;
2. estimate bulk population distributions as a function of pH;
3. sample environment-conditioned conformers for each relevant microstate;
4. compute interfacial affinity, orientation, dehydration, and core-crossing barriers;
5. include local membrane deformation, hydration defects, and lipid interactions as coupled coordinates;
6. infer position-dependent diffusion and propagate its covariance with the PMF;
7. combine state-specific permeabilities through alternative protonation-kinetic limits; and
8. validate against multiple experimental membranes and pH conditions.

Before scaling to many compounds, run finite-size, force-field, coordinate, sampling, and microstate sensitivity studies on a designed panel. A mechanistic claim should predict interventions: pH shifts, microstate stabilization, donor/acceptor edits, membrane-composition changes, and linker changes. Agreement with one aggregate permeability value is insufficient.

## Bottom line

Qi et al. show that a rare neutral tautomer can dominate observed permeability and that the permeant can remodel the membrane that defines its barrier. The study supplies the correct conceptual depth for this project while also demonstrating why first-principles results require uncertainty, alternate hypotheses, and perturbational validation. Its exact numbers should not be extrapolated to the intended much larger molecules.
