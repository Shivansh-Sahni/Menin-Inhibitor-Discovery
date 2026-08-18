# Aguilar evidence integration: defensible mechanistic parameter program

## Bottom line

The new material supports the process-centered program, but it also prevents an easy
mistake: ETR, eHBD, AB-MPS, compactness, lipophilicity, and generic hERG pharmacophores
must not become the project's supposedly novel parameter set. They are valuable
measurements, controls, or falsifiers of narrower hypotheses.

The proposed contribution is a **coupled State-Path-Flux representation**: identify the
chemical and biological states, estimate the rates and free-energy bottlenecks connecting
them, and weight local molecular properties only by the paths that carry productive flux.

## Evidence-driven corrections

| Evidence | What survives | What does not survive | Project action |
|---|---|---|---|
| Di/Kerns barriers chapter | Exposure is a coupled sequence of physical and biochemical barriers | A flat list of drug-like properties as independent causes | Build a causal state graph and mass-balanced competing fluxes |
| Di/Kerns PK chapter | AUC is observed from the curve; CL and F are derived from dose/AUC families | Training on AUC, CL, and F as independent labels | Enforce lineage and use CL/F as closure diagnostics |
| Di/Kerns hERG chapter | Unbound exposure, channel gating, cavity recognition, and trapping matter | Historical scalar alerts or the chapter's internally contradictory oxygen-HBA advice | Use modern structures and dynamic protocols; retain old rules only as controls |
| Price 2024 | Experimental polarity can diverge from TPSA and chemical classes use different absorption strategies | ETR as a universal chameleonicity mechanism or portable cutoff | Treat ETR as a comparator; resolve local shielding and path flux instead |
| Schade 2024 | Site-specific donor shielding and stereochemical preorganization are experimentally measurable | Global folding/compactness as a necessary condition for oral exposure | Resolve local donor/acceptor compensation independently of global Rg |
| Le Manach 2026 | Cross-portfolio falsification and efflux can dominate oral absorption | Assuming a published chameleonicity trend transfers to this series | Add enterocyte efflux/re-entry/metabolism competition and preregister negative controls |
| Current hERG primary evidence | Ligand, microstate, receptor state, voltage protocol, access, binding, and trapping interact | Treating IC50 as protocol-independent affinity | Predict protocol-integrated bound/trapped occupancy once kinetic assays exist |

## Ten causal modules

These modules are stable scientific questions, not ten immutable columns. A fast proxy is
retired when a more direct observable becomes available. The tenth module is an
observation/uncertainty layer required for inference, not a molecular-physics parameter.

1. **Chemical-state thermodynamics and exchange** - microscopic protomer/tautomer free
   energies, pH populations, uncertainty, and switching rates.
2. **Local hydration and compensation** - site-resolved water loss and the specific IMHB,
   lipid, transporter, or receptor contact replacing it; donors and acceptors remain separate.
3. **Environment-conditioned conformational response** - joint basin changes between
   water, interface, membrane core, and receptor rather than one vacuum conformer.
4. **Conformational gating** - MFPT, barrier, committor, and flux into a competent state;
   global compaction is only one possible coordinate.
5. **Membrane bottleneck profile** - entry, interface trapping, core passage, and release,
   including diffusivity, orientation, deformation, and hysteresis.
6. **Enterocyte fate competition** - basolateral escape versus apical efflux, re-entry,
   intracellular sequestration, and gut metabolism.
7. **Systemic disposition exchange** - unbound blood/tissue exchange, hepatic uptake and
   metabolism, biliary/renal loss, and slow tissue release.
8. **hERG access-state conversion** - whether a membrane-accessible state converts into a
   cavity-binding state during its local residence time.
9. **hERG receptor-state binding and trapping** - receptor-conditioned affinity, contact
   reorganization, escape, trapping, and recovery using at most two preregistered receptor
   hypotheses per compound from the six-structure library.
10. **Observation and uncertainty model** - formulation, species, assay pH, voltage,
    temperature, free concentration, recovery, censoring, and measurement/derivation error.

## The new enterocyte quantities

For state (s) inside an enterocyte, the simplest competing-hazard limit is

\[
P_{escape}(s)=\frac{k_{BL}(s)}
{k_{BL}(s)+k_{efflux}(s)+k_{gut-met}(s)+k_{seq}(s)},
\]

where basolateral escape competes with apical efflux, gut metabolism, and sequestration.
The full model includes apical re-entry, so the useful outputs are an absorbing Markov
network's systemic-escape probability, metabolism-before-escape probability, expected
number of efflux/re-entry cycles, and cumulative intracellular residence time.

These quantities are more fundamental than an efflux ratio: the ratio remains an assay
observation used to constrain rates, whereas the latent fate probabilities explain why a
given ratio changes exposure under a particular concentration, inhibitor, or protocol.

## Why this is relatively novel without overclaiming

None of the following is individually novel: TPSA/EPSA, eHBD/eHBA, IMHBs, Rg, PMFs,
efflux ratios, Markov models, PBPK, or hERG state dependence. The potentially distinctive
method is their restricted causal coupling:

- polarity is weighted by reactive membrane or binding flux rather than averaged over all
  generated conformers;
- shielding is site- and path-specific and need not imply global folding;
- transporter recycling is coupled to membrane re-entry and enterocyte metabolism;
- membrane access is coupled to chemical-state conversion and hERG receptor-state kinetics;
- summary PK and IC50 observations are generated by the state model rather than reused as
  upstream physical features; and
- every mechanistic quantity carries a competing explanation and a predeclared falsifier.

This is a research hypothesis, not a universal novelty or patentability claim. It becomes a
scientific contribution only if the coupled quantities converge, survive matched-pair and
cross-series tests, and explain residual patterns that simpler models cannot.

## Efficient validation sequence

1. Preserve conventional baselines and published comparators, including ETR/eHBD when
   experimentally available.
2. Use the internal matched pairs to test local shielding versus global compactness.
3. Measure efflux with recovery and transporter inhibitors and fit the enterocyte fate model.
4. Run HPC environment and membrane calculations only for mechanism-discriminating pairs.
5. Add hERG access and receptor-state kinetics only after free-concentration onset/recovery
   data exist.
6. Promote a quantity to the decision track only after convergence, prospective calibration,
   and non-inferiority to the retained baseline.
