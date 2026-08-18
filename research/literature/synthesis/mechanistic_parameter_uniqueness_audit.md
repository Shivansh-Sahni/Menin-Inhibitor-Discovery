# Mechanistic parameter uniqueness audit

## Direct answer

No claim should be made that every stored numeric column is a unique fundamental
parameter. The repository intentionally contains four different kinds of quantities:

1. **mechanistic parameters**: free energies, populations, rates, diffusivities, and
   state occupancies attached to a specific causal node or edge;
2. **observations**: assay- and protocol-conditioned measurements that constrain those
   parameters;
3. **proxies and conventional controls**: inexpensive correlates used for baselines or
   one-at-a-time discovery ablations; and
4. **QC and uncertainty quantities**: convergence, missingness, applicability, and
   sensitivity diagnostics that must never be interpreted as biological causes.

Only the first category is the intended fundamental layer. A process can be fundamental
without being rate-limiting for every compound. Its importance must be estimated from
sensitivity, resistance, reactive flux, or intervention evidence rather than asserted in
advance.

## Uniqueness of the eight pre-HPC observables

The eight observables in `pre_hpc_scientific_contract.json` are a selected HPC physics
subset, not the entire PK model and not eight arbitrary descriptors.

| Observable | Unique causal question | Boundary from neighboring quantities |
|---|---|---|
| Environment-conditioned state shift | Which chemical/conformational basins change thermodynamic stability between environments? | Population/free-energy response, not the rate of reaching a basin |
| Reactive-flux-weighted polar exposure | Which local polarity states actually carry productive transport or binding flux? | Path contribution, not equilibrium SASA or an arbitrary distribution tail |
| Conformational-gating time | Can a competent state form before membrane/receptor residence ends? | Transition kinetics/MFPT, not state population |
| Site-resolved desolvation compensation | What energetic price is paid when individual hydration contacts are lost, and which contacts replace them? | Local energy accounting, not global polarity or IMHB counts |
| Membrane bottleneck profile | Is transport limited by entry, interface trapping, core crossing, or release? | Spatial translocation resistance and diffusivity, not conformational gating or partition alone |
| Enterocyte efflux/recycling | Does intracellular material escape systemically, recycle apically, become sequestered, or undergo gut metabolism? | Competing cellular fates, not passive membrane crossing |
| Microstate-switching competition | Does protonation/tautomer exchange occur fast enough during local residence to alter the productive path? | Chemical-state exchange kinetics, not equilibrium pKa population |
| hERG state binding/trapping | Which channel states bind, release, or trap each accessible ligand state under a voltage protocol? | Receptor/channel kinetics, not systemic exposure or membrane access |

These quantities are separately falsifiable and have different governing equations,
units, experimental perturbations, and failure signatures. Their values cannot be inferred
by renaming one another.

## Full PK and hERG coverage outside that subset

The meeting notes correctly require PK to remain decomposed. The complete causal model
therefore also contains the following nonredundant modules:

- administered phase, dissolution, precipitation, aggregation, and free-monomer access;
- hepatic uptake, metabolism, and biliary export;
- plasma/blood binding and organ-specific distribution/exchange;
- renal filtration, secretion, and reabsorption; and
- the observation model for formulation, species, sampling, recovery, censoring, and
  assay protocol.

The observation model is essential for valid inference but is not itself molecular
physics. Likewise, AUC, CL, F, Vdss, IC50, ETR, efflux ratio, MW, TPSA, and Rg are not
independent fundamental processes. They are downstream summaries, observations,
comparators, controls, or diagnostics.

hERG is connected to PK through unbound cardiac exposure and membrane access, then
separated into channel-state recognition, binding/unbinding, trapping/recovery, and
protocol-integrated current inhibition. Generic positive-charge and aromatic features
remain conventional controls unless they are resolved into a ligand microstate, receptor
state, interaction geometry, and kinetic consequence.

## Admission rule

A new mechanistic parameter is admitted only when it has:

1. one named causal node or edge and no duplicate causal role;
2. an equation, units, environmental/protocol conditions, and uncertainty;
3. a distinct perturbation or negative control;
4. a convergence or assay-identifiability gate;
5. a predicted residual or intervention signature;
6. a falsification criterion and a rule for retirement or replacement; and
7. incremental explanatory value tested one module at a time against the conventional
   baseline.

This implements the July 20 meeting direction: physics before correlation, separately
modeled PK processes, large-molecule folding and solvent response, protein-state physics
for hERG, automation, and a small mechanistic model rather than hundreds of descriptors.

