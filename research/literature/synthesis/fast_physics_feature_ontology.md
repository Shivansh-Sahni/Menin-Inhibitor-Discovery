# Fast-physics feature ontology and sampling defense

## Decision

The fast-physics table is an evidence store, not a model matrix. The historical
smoke summary exposed 105 numeric scientific/QC columns at pH 7.4, and an early
allowlist admitted 11 columns representing ten proposed phenomena. A subsequent
primary-evidence and causal audit showed that several exact constructions were
QC quantities or unvalidated algebraic hypotheses. The fail-closed provisional
model allowlist therefore contains only two columns: a low-polarity tail and a
mean nonpolar contact surface. Even these are discovery
proxies rather than fundamental observables and require controlled ablation.

The number two is not a new scientific constant. It is the current count of
proxies that survived the stated admission rule. Unknown future columns fail
closed until they receive a causal interpretation, confounder list, permitted
role, redundancy decision, and falsification test.

No fast-physics feature is decision-track eligible. These screening geometries
use approximate state weights, ETKDG, and MMFF94s/UFF without explicit solvent,
membrane, receptor relaxation, or calibrated transfer free energies.

The provisional probes are not universal chameleonicity rules. Inganäs et al.
(2025, DOI `10.1021/acs.jmedchem.5c01499`) found environment-dependent polarity
shielding across a large PROTAC collection but also favorable extended
membrane conformations and multiple possible rate-limiting membrane steps. Le
Manach et al. (2026, DOI `10.1021/acsmedchemlett.6c00043`) did not reproduce
general chameleonicity guidance in an AstraZeneca 68-PROTAC set and found
efflux ratio more informative. Compactness, shielding, interface trapping,
core crossing, release, and cellular efflux must therefore remain separable
causal hypotheses.

## Post-audit provisional discovery set

| Selected column | Physical phenomenon | Biological event / causal location | Principal hidden variables | Why this representation survives |
|---|---|---|---|---|
| `polar_sasa_ang2__q05` | rare low-polar-exposure tail | desolvation during membrane insertion/cavity entry | state populations, conformer free energies, solvent competition | tests a transport-compatible tail instead of five redundant central summaries |
| `nonpolar_sasa_ang2__mean` | accessible nonpolar contact area | membrane partition and hydrophobic cavity encounter | lipid composition, dispersion, aggregation | distinct contact opportunity from polar dehydration |

The model layer may use the conventional 2D controls plus these two columns
only in discovery-track ablations. Correlation, convergence, matched-pair
residual tests, and explicit-environment falsification determine whether any
one survives. Their fundamental replacements are reactive-flux-weighted
polarity, interface/core/release free energies, position-dependent diffusivity,
and path-conditioned shape/orientation.

Radius of gyration remains in the ensemble evidence store and conformer-bag
representation, but no Rg summary is automatically admitted to the model.
Schade et al. 2024 observed globally extended oral clinical PROTACs with local
donor shielding, demonstrating that compactness is neither necessary nor
sufficient for transport. Path-conditioned shape, local shielding, and
orientation must be tested together.

The removed quantities remain available for audit and one-at-a-time
falsification: signed charge is speciation context; absolute net charge cannot
detect zwitterionic burden; mean NPR destroys multimodal shape information;
normalized enumerated-weight entropy is sampling QC; donor and acceptor
exposure must be separated; and the IMHB/polar-SASA and charge-centroid/Rg
ratios are unvalidated hypotheses. Membrane- and receptor-state predictors are
absent until converged PMF/MD or compatible experiments exist.

The conformer multiple-instance experiment is separately fail-closed to five
raw physical primitives: polar and nonpolar SASA, donor and acceptor exposed
SASA, and radius of gyration. It cannot ingest conformer rank, cluster ID,
minimization status, force-field energy, exact aliases, algebraic surface
closure, raw PMI values, mean shape coordinates, distance-only IMHB counts,
fixed-charge centroids, or any new numeric output merely because it exists.

## Features retained outside the model

- `sa_3d_psa_ang2` is an exact implementation alias of `polar_sasa_ang2` and is
  excluded. In the 110-compound smoke table, all five summaries are exactly
  equal.
- Mean, median, upper tail, lower tail, and standard deviation are not five
  mechanisms. Only the prespecified aggregation above enters the model; the
  others diagnose distribution width, tail stability, and sampling failure.
- `total_sasa = polar_sasa + nonpolar_sasa` is a closure/size diagnostic, not a
  third independent surface mechanism.
- Exposed HBD/HBA counts and donor/acceptor SASA remain component-level
  explanations, but are not entered beside their combined burden.
- Raw IMHB count, raw charge-centroid distance, and the Gasteiger dipole remain
  sensitivity diagnostics; their normalized selected representations replace
  them in the model.
- The polarity-response, shape-response, and folded-shift features are three
  readouts of one fixed descriptor-reweighting axis. None is treated as
  independent solvent physics. The polarity response can prioritize an
  explicit-solvent falsification pilot; the other two are companion diagnostics.
- Hydration-shedding compensation algebraically recombines selected component
  features. Rare-state transport dominance uses an uncalibrated descriptor
  propensity rather than permeability. Both are one-at-a-time falsification
  hypotheses, not predictors.
- pKa-sensitivity spans are uncertainty/abstention inputs, never independent
  causal predictors. `joint_weight_sum` and effective sampled count are QC and
  convergence quantities.

The machine-readable ontology is generated as
`fast_physics_feature_ontology.csv` beside the canonical physics contracts. It
records the physical phenomenon, biological event, causal location, hidden
variables, permissible role, redundancy group, rationale, and falsification
test for every current feature family.

## Can approximately 18 microstates be defended?

Only as an observed enumeration breadth, not as a universal or equilibrium
microstate count. In the current 110-structure smoke registry, retained state
count is mean 17.48, median 15, range 11–24. The number emerged from the current
rules: pH 2.0/5.0/6.5/7.4 plus hERG assay pH, nominal pKa and +/-1-unit
sensitivity, the historical smoke run's 0.5% population screen, retention of
reference tautomers and charge/exposure-changing states, and a 24-state cap.
That historical count is not the release rule. The release configuration uses
0.1%, because Qi et al. report a roughly 0.14% neutral tetracycline tautomer
whose approximately million-fold higher state-specific permeability makes it
the dominant transport path.

Two limitations prevent a stronger defense now. First, the current enumerator
creates reference tautomers and one-site protonation/deprotonation transforms;
it does not establish all coupled multi-site microstates. Second, reference
tautomer weights do not come from tautomer free energies. Thus “18
microstates” must be reported as approximately 18 retained chemical-state
hypotheses, with approximate Henderson–Hasselbalch sensitivity weights.

The scientific gate is threshold convergence, not the number 18. Enumerate
candidates down to 0.01% and compare retained ensembles at 1%, 0.1%, and 0.01%,
while always preserving plausible rare neutral, zwitterionic, or charge-shielded
transport states. Accept the 0.1% working threshold only when omitted mass is
below 0.5% at every pH, weighted physical means change by no more than 3%, and
folded/IMHB fractions change by no more than 0.05 absolute versus 0.01%. HPC
enumeration has no fixed state-count cap; any future resource truncation must
expose omitted chemistry and makes the compound inadmissible. Experimental microscopic pKas or a validated coupled-site engine are
required before these weights can be interpreted as equilibrium populations.

## Can 250 conformers per retained state be defended?

Not yet as a fixed scientific requirement. It is a deep-tier sampling ceiling,
not an equation-derived answer. Local all-series execution is deferred. The
future selected-state workflow starts at 25; 50/100/250/500 are adaptive
escalation rungs. This preserves broad
state coverage without pretending that 250 correlated ETKDG starts are 250
independent equilibrium samples.

The required nested convergence experiment is (N=25, 50, 100, 250, 500),
using deterministic nested seeds. At every level, compare minimum minimized
energy, selected ensemble means and q05 tails, radius of gyration, exposed polar
surface, IMHB-network occupancy, compact-cluster occupancy, cluster count, and
new-cluster discovery rate. For an observable (X), record

\[
\Delta X_N = \frac{|X_N-X_{2N}|}{\max(|X_{2N}|,\epsilon)}.
\]

The 250 setting is adequate for a particular state and observable only if both
independent seeds pass: relative mean changes no greater than 5%, standardized
changes no greater than 0.10, distribution Jensen-Shannon divergence no greater
than 0.05, folded/IMHB fractions changing no more than 0.05 absolute, new
torsional-cluster mass no greater than 2%, and minimum-energy improvement below
0.5 kcal/mol as a diagnostic. At least 90% of important pilot states must pass,
with no systematic failure in the flexible or charge-sensitive strata, before
250 is called broadly adequate.
Sampling must be adaptive: stop earlier for demonstrably converged rigid states;
escalate flexible, high-population, matched-pair-discordant, or mechanism-critical
states to 250/500 or explicit-solvent enhanced sampling. A single global count
cannot be defended before these tests.

## Evidence basis

- Gunner et al. (2020), DOI `10.1007/s10822-020-00280-7`: tautomer/protomer
  populations require state free energies; macroscopic pKa values are not
  sufficient microstate weights.
- Harris, Chipot, and Roux (2024), DOI `10.1021/acs.jpcb.3c06765`: effective
  permeability depends on state-specific permeability and protonation kinetics,
  not population alone.
- Qi, Chipot, and Wang (2025), DOI `10.1021/acs.jpcb.5c05445`: a roughly 0.14%
  neutral tetracycline tautomer dominates modeled permeability because its
  state-specific permeability is about six orders of magnitude higher.
- Riniker and Landrum (2015), DOI `10.1021/acs.jcim.5b00654`, and Wang et al.
  (2020), DOI `10.1021/acs.jcim.0c00025`: ETKDG is stochastic, knowledge-guided
  candidate generation whose required depth grows with conformational space.
- Kanal, Keith, and Hutchison (2018), DOI `10.1002/qua.25512`: MMFF94, UFF, and
  GAFF conformer energies/ranks correlate poorly with higher-level results, so
  minimized force-field weights cannot be called equilibrium populations.
- Poongavanam et al. (2022), DOI `10.1021/acs.jmedchem.2c00877`: the attached
  PROTAC study used five force fields, two solvent models, 50,000 Monte Carlo
  steps per search, and triplicate 100 ns MD, illustrating why 250 ETKDG starts
  are a screen rather than exhaustive bRo5 sampling.
