"""Causal ontology and fail-closed selection for fast-physics features.

The fast-physics tables intentionally retain rich distributions for audit and
falsification.  They are *not* a license to give every numeric column to a
model.  This module assigns each feature family one physical interpretation
and one permissible role, then exposes a deliberately small discovery-model
set.  Unknown future columns are excluded until they receive an ontology row.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd


@dataclass(frozen=True)
class PhysicsFeatureConcept:
    """One physical concept represented by one or more output columns."""

    feature_family: str
    physical_phenomenon: str
    biological_event: str
    causal_location: str
    hidden_variables_and_confounders: str
    permissible_model_roles: str
    status: str
    redundancy_group: str
    selected_aggregation: str
    rationale: str
    falsification_test: str


@dataclass(frozen=True)
class ConventionalFeatureConcept:
    """Role of a conventional molecular input in the claim hierarchy."""

    feature: str
    calculation_class: str
    permissible_role: str
    mechanistic_status: str
    redundancy_or_bias: str
    production_action: str
    fundamental_replacement: str
    rationale: str


# This compact descriptor panel is a conventional empirical control, not a
# mechanistic parameter set. Exact molecular weight and heavy-atom count are
# removed because MolWt already supplies the same size axis at this sample
# size; submitted-state formal charge is constant after parent
# standardization. Ring and topology descriptors remain only so the standard
# baseline is not artificially weakened.
CONVENTIONAL_DESCRIPTOR_COLUMNS: tuple[str, ...] = (
    "mol_wt",
    "logp",
    "tpsa",
    "h_bond_donors",
    "h_bond_acceptors",
    "rotatable_bonds",
    "ring_count",
    "aromatic_ring_count",
    "fraction_csp3",
)


_CONVENTIONAL_CONCEPTS: tuple[ConventionalFeatureConcept, ...] = (
    ConventionalFeatureConcept(
        "mol_wt",
        "exact structural calculation",
        "domain and size control",
        "not_a_mechanism",
        "correlated with heavy-atom count and exact molecular weight",
        "retain_conventional_control",
        "explicit accommodation, diffusion, partition, and free-energy terms",
        "Size defines the chemical domain but does not identify a biological event.",
    ),
    ConventionalFeatureConcept(
        "exact_mol_wt",
        "exact structural calculation",
        "redundancy audit only",
        "not_a_mechanism",
        "duplicates the molecular-size axis represented by mol_wt",
        "remove_from_internal_model_matrix",
        "none; retain mol_wt as the single conventional size control",
        "The isotopic-mass convention is not a second biological process.",
    ),
    ConventionalFeatureConcept(
        "heavy_atom_count",
        "exact structural count",
        "redundancy audit only",
        "not_a_mechanism",
        "strongly overlaps molecular weight in the current narrow chemistry",
        "remove_from_internal_model_matrix",
        "explicit process-specific steric and accommodation quantities",
        "Atom count cannot be interpreted as a distinct process beside molecular weight.",
    ),
    ConventionalFeatureConcept(
        "logp",
        "calculated partition proxy",
        "conventional empirical control",
        "not_a_mechanism",
        "neutral-parent and model dependent; conflates entry and interfacial trapping",
        "retain_conventional_control",
        "pH-specific state transfer free energies and membrane/tissue partition",
        "A strong baseline control is required precisely because lipophilicity has established hERG and PK associations.",
    ),
    ConventionalFeatureConcept(
        "tpsa",
        "2D polarity proxy",
        "conventional empirical control",
        "not_a_mechanism",
        "ignores state, exposure, environment, hydration, and transition path",
        "retain_conventional_control",
        "state-, site-, environment-, and path-conditioned hydration/desolvation",
        "TPSA tests whether proposed physics adds value beyond established polarity heuristics.",
    ),
    ConventionalFeatureConcept(
        "h_bond_donors",
        "typed atom count",
        "conventional empirical control",
        "not_a_mechanism",
        "donor exposure and compensation are absent",
        "retain_conventional_control",
        "site-specific donor hydration, exposure, and compensation free energy",
        "Donor and acceptor counts remain separate controls because their transfer penalties differ.",
    ),
    ConventionalFeatureConcept(
        "h_bond_acceptors",
        "typed atom count",
        "conventional empirical control",
        "not_a_mechanism",
        "acceptor exposure and compensation are absent",
        "retain_conventional_control",
        "site-specific acceptor hydration, exposure, and compensation free energy",
        "Donor and acceptor counts remain separate controls because their transfer penalties differ.",
    ),
    ConventionalFeatureConcept(
        "rotatable_bonds",
        "topological count",
        "conventional flexibility control",
        "not_a_mechanism",
        "does not encode basins, barriers, rates, or productive pathways",
        "retain_conventional_control",
        "metastable basins, transition rates, MFPT, committor, and reactive flux",
        "It is an established control against which transition dynamics must be tested.",
    ),
    ConventionalFeatureConcept(
        "ring_count",
        "topological count",
        "conventional topology control only",
        "not_a_mechanism",
        "overlaps rigidity, size, and aromatic-ring count",
        "retain_conventional_control_not_mechanistic_layer",
        "path-conditioned conformation and steric accommodation",
        "It prevents weakening the empirical comparator but receives no causal interpretation.",
    ),
    ConventionalFeatureConcept(
        "aromatic_ring_count",
        "topological pharmacophore count",
        "conventional hERG/rigidity control",
        "not_a_mechanism",
        "does not encode receptor state, geometry, energy, or persistence",
        "retain_conventional_control",
        "receptor-state-specific aromatic contact free energy and persistence",
        "Established hERG pharmacophore content must be controlled before claiming a receptor mechanism.",
    ),
    ConventionalFeatureConcept(
        "fraction_csp3",
        "topological composition ratio",
        "conventional topology comparator",
        "not_a_mechanism",
        "non-specific mixture of saturation, shape, and scaffold effects",
        "retain_conventional_control",
        "explicit conformational basin and path-conditioned shape quantities",
        "It is a comparator, not a fundamental causal variable.",
    ),
    ConventionalFeatureConcept(
        "formal_charge",
        "submitted-parent state calculation",
        "structure/state QC only",
        "not_an_assay_state",
        "all standardized parents are neutral despite predominantly basic pKa evidence",
        "remove_from_internal_model_matrix",
        "microstate free energies, populations, local charge exposure, and switching rates",
        "A constant neutral-parent value erases rather than represents assay speciation.",
    ),
    ConventionalFeatureConcept(
        "invalid_structure",
        "quality-control flag",
        "quality control only",
        "never_a_predictor",
        "a parsing failure can become a source or batch identifier",
        "exclude_from_all_models",
        "source correction or explicit exclusion",
        "Data validity is not molecular biology.",
    ),
    ConventionalFeatureConcept(
        "morgan_fingerprint",
        "associative substructure representation",
        "strong conventional residual baseline",
        "not_a_mechanistic_parameter",
        "bits have no unique causal interpretation and encode series proximity",
        "retain_as_separate_control",
        "explicit free energies, rates, and observation-model parameters",
        "The fingerprint establishes how much can be recovered from analogue similarity alone.",
    ),
    ConventionalFeatureConcept(
        "d_mpnn_embedding",
        "learned atom/bond representation",
        "conventional learned comparator",
        "not_a_mechanistic_parameter",
        "latent dimensions are non-identifiable and data hungry",
        "retain_as_separate_control_when_series_support_is_adequate",
        "explicit process latents constrained by orthogonal measurements",
        "A graph model is a predictive null, not a physical ontology.",
    ),
    ConventionalFeatureConcept(
        "scaffold_or_series",
        "group/context label",
        "splitting and hierarchical context only",
        "not_a_molecular_mechanism",
        "computed scaffolds are not medicinal-chemistry series",
        "exclude_from_molecular_predictors",
        "chemist-validated series metadata for evaluation only",
        "Group identity controls dependence; it does not cause PK or hERG.",
    ),
    ConventionalFeatureConcept(
        "mw_bin_or_cutoff",
        "analysis stratification",
        "domain analysis only",
        "not_a_mechanism",
        "post hoc thresholds create researcher degrees of freedom",
        "exclude_from_primary_predictors",
        "continuous physical-state variables and preregistered change-point tests",
        "No 650/700/750-Da biological boundary is currently supported.",
    ),
)


_PRIMITIVE_CONCEPTS: dict[str, PhysicsFeatureConcept] = {
    "sa_3d_psa_ang2": PhysicsFeatureConcept(
        "sa_3d_psa_ang2",
        "solvent-accessible polar surface",
        "dehydration before membrane insertion or cavity entry",
        "aqueous conformer ensemble before membrane/receptor access",
        "identical implementation to polar_sasa_ang2; SASA radii; protonation; solvent and force field absent",
        "legacy output compatibility only",
        "remove_duplicate",
        "polar_surface_alias",
        "none",
        "This is an exact alias of polar_sasa_ang2, so retaining both creates duplicate evidence.",
        "Exact numerical equality with polar_sasa_ang2 is required.",
    ),
    "polar_sasa_ang2": PhysicsFeatureConcept(
        "polar_sasa_ang2",
        "accessible polar surface and its low-exposure tail",
        "hydration shedding during membrane insertion and desolvation during hERG cavity entry",
        "aqueous-to-interface transition",
        "microstate population; solvent competition; atomic radii; conformer free energies; aggregation",
        "q05 as discovery predictor; other summaries as distribution diagnostics",
        "provisional_discovery_proxy",
        "polar_exposure",
        "q05",
        "The low tail tests whether rare polarity-shielded conformers exist; the mean and median are strongly redundant in current smoke data.",
        "Passive-permeability ranking and solvent-dependent NMR/SASA should change coherently across matched pairs.",
    ),
    "nonpolar_sasa_ang2": PhysicsFeatureConcept(
        "nonpolar_sasa_ang2",
        "solvent-accessible nonpolar contact area",
        "membrane/headgroup partition and hydrophobic hERG-cavity contact opportunity",
        "membrane partition and receptor encounter",
        "lipid composition; dispersion; aromatic topology; aggregation; exposed area is not partition free energy",
        "mean as discovery predictor; tails as descriptive diagnostics",
        "provisional_discovery_proxy",
        "nonpolar_contact_surface",
        "mean",
        "Mean nonpolar area represents contact capacity distinct from polar dehydration burden.",
        "Matched liposome partitioning or cavity-contact persistence should track the predicted direction.",
    ),
    "total_sasa_ang2": PhysicsFeatureConcept(
        "total_sasa_ang2",
        "total accessible molecular surface",
        "gross diffusional and accommodation demand",
        "solution, membrane, and tissue encounter",
        "algebraically equals polar plus nonpolar SASA and is strongly confounded by MW/heavy-atom count",
        "descriptive size control only",
        "mechanistic_diagnostic_only",
        "surface_closure",
        "none",
        "It is retained for surface-area closure but excluded from models containing its components and 2D size controls.",
        "Verify total SASA equals polar plus nonpolar SASA within numerical tolerance.",
    ),
    "exposed_hbd_count_proxy": PhysicsFeatureConcept(
        "exposed_hbd_count_proxy",
        "thresholded exposed hydrogen-bond donors",
        "donor desolvation and solvent/lipid competition",
        "aqueous-to-membrane and ligand-to-cavity transfer",
        "arbitrary per-atom SASA threshold; donor geometry; protonation; redundant with continuous donor SASA",
        "interpretation diagnostic only",
        "mechanistic_diagnostic_only",
        "hbond_exposure_components",
        "none",
        "Thresholded counts discard exposure magnitude; the continuous combined burden is preferred.",
        "Vary the exposed-atom threshold and require stable matched-pair direction.",
    ),
    "exposed_hba_count_proxy": PhysicsFeatureConcept(
        "exposed_hba_count_proxy",
        "thresholded exposed hydrogen-bond acceptors",
        "acceptor desolvation and solvent/lipid competition",
        "aqueous-to-membrane and ligand-to-cavity transfer",
        "arbitrary per-atom SASA threshold; acceptor typing; protonation; redundant with continuous acceptor SASA",
        "interpretation diagnostic only",
        "mechanistic_diagnostic_only",
        "hbond_exposure_components",
        "none",
        "Thresholded counts discard exposure magnitude; the continuous combined burden is preferred.",
        "Vary the exposed-atom threshold and require stable matched-pair direction.",
    ),
    "exposed_hbd_sasa_ang2": PhysicsFeatureConcept(
        "exposed_hbd_sasa_ang2",
        "continuous exposed donor surface",
        "donor dehydration burden",
        "aqueous-to-membrane and ligand-to-cavity transfer",
        "donor typing; protonation; solvent competition; overlap with polar SASA",
        "component audit for the combined H-bond burden",
        "mechanistic_diagnostic_only",
        "hbond_exposure_components",
        "none",
        "Donor and acceptor components remain visible for explanation but are not entered beside their combined burden.",
        "Donor-selective matched edits should alter this component without an equivalent acceptor change.",
    ),
    "exposed_hba_sasa_ang2": PhysicsFeatureConcept(
        "exposed_hba_sasa_ang2",
        "continuous exposed acceptor surface",
        "acceptor dehydration burden",
        "aqueous-to-membrane and ligand-to-cavity transfer",
        "acceptor typing; protonation; solvent competition; overlap with polar SASA",
        "component audit for the combined H-bond burden",
        "mechanistic_diagnostic_only",
        "hbond_exposure_components",
        "none",
        "Donor and acceptor components remain visible for explanation but are not entered beside their combined burden.",
        "Acceptor-selective matched edits should alter this component without an equivalent donor change.",
    ),
    "radius_of_gyration_angstrom": PhysicsFeatureConcept(
        "radius_of_gyration_angstrom",
        "global conformational compactness and its compact tail",
        "diagnosis of conformational basins that may affect insertion or cavity accommodation",
        "solution-to-membrane/cavity conformational selection",
        "molecular size; ETKDG coverage; minimization force field; true solvent-conditioned populations",
        "shape and sampling diagnostic only; not an automatic summary-model predictor",
        "mechanistic_diagnostic_only",
        "compactness",
        "none",
        "Oral PROTACs can remain globally extended while locally shielding donors, so compactness is neither necessary nor sufficient for productive transport.",
        "Test path-conditioned shape jointly with local shielding and orientation; matched-pair direction must survive size normalization and explicit environment sampling.",
    ),
    "npr1": PhysicsFeatureConcept(
        "npr1",
        "first coordinate of mass-shape anisotropy",
        "orientation and steric accommodation at membrane or hERG cavity",
        "membrane orientation and receptor entry/binding",
        "conformer sampling; atom masses; environment; meaningful interpretation requires NPR2 jointly",
        "ensemble visualization and orientation-hypothesis diagnostic only",
        "mechanistic_diagnostic_only",
        "shape_anisotropy_pair",
        "mean",
        "A mean NPR coordinate can describe an unpopulated shape and erases multimodality, orientation, and transition barriers.",
        "Replace with environment/path-conditioned shape basins and membrane/receptor orientation distributions.",
    ),
    "npr2": PhysicsFeatureConcept(
        "npr2",
        "second coordinate of mass-shape anisotropy",
        "orientation and steric accommodation at membrane or hERG cavity",
        "membrane orientation and receptor entry/binding",
        "conformer sampling; atom masses; environment; meaningful interpretation requires NPR1 jointly",
        "ensemble visualization and orientation-hypothesis diagnostic only",
        "mechanistic_diagnostic_only",
        "shape_anisotropy_pair",
        "mean",
        "NPR1 and NPR2 are one visualization coordinate, not two mechanisms or a path-resolved steric observable.",
        "Replace with environment/path-conditioned shape basins and membrane/receptor orientation distributions.",
    ),
    "gasteiger_dipole_proxy_debye": PhysicsFeatureConcept(
        "gasteiger_dipole_proxy_debye",
        "charge-weighted conformer dipole proxy",
        "electrostatic orientation in an interface or cavity field",
        "membrane interface and hERG cavity encounter",
        "Gasteiger charges; origin/ion treatment; polarization; dielectric response; correlated with charge-centroid separation",
        "diagnostic and force-field-sensitivity comparison only",
        "mechanistic_diagnostic_only",
        "electrostatic_topology",
        "none",
        "A fixed-charge proxy is too model-dependent to enter beside the normalized charge-separation feature.",
        "Compare against quantum/force-field charge models and require stable matched-pair ordering.",
    ),
    "charge_centroid_separation_angstrom": PhysicsFeatureConcept(
        "charge_centroid_separation_angstrom",
        "spatial separation of positive and negative partial-charge centroids",
        "electrostatic orientation and complementary receptor contacts",
        "membrane interface and hERG cavity encounter",
        "Gasteiger charges; molecular size; formal charge; dielectric polarization",
        "component audit for size-normalized charge topology",
        "mechanistic_diagnostic_only",
        "electrostatic_topology",
        "none",
        "The raw distance is size-confounded; the gyration-normalized composite is the selected representation.",
        "Charge-model sensitivity must not reverse matched-pair direction.",
    ),
    "imhb_count_proxy": PhysicsFeatureConcept(
        "imhb_count_proxy",
        "geometric intramolecular donor-acceptor contact network",
        "temporary polarity masking during low-dielectric transfer",
        "solution-to-membrane conformational selection",
        "distance-only definition; no angle or solvent competition; donor/acceptor typing; conformer kinetics",
        "component audit for surface-normalized shielding",
        "mechanistic_diagnostic_only",
        "intramolecular_shielding",
        "none",
        "A raw count scales with available polar groups; the surface-normalized shielding candidate is preferred.",
        "Explicit-solvent IMHB occupancy and NMR should confirm persistence rather than geometric opportunity.",
    ),
    "formal_charge": PhysicsFeatureConcept(
        "formal_charge",
        "population-weighted signed charge tendency",
        "aqueous speciation controls membrane access and cationic hERG recognition",
        "solution speciation before membrane/receptor encounter",
        "micro-pKa; tautomer free energy; ionic strength; assay pH; current enumeration omits coupled multi-site equilibria",
        "speciation context, QC, and pH-stratified sensitivity only",
        "mechanistic_diagnostic_only",
        "speciation_charge_moments",
        "mean",
        "Signed mean distinguishes net cationic from anionic tendency and is not interchangeable with charge magnitude.",
        "pH-dependent permeability and hERG potency should move consistently with measured micro-pKas.",
    ),
    "absolute_formal_charge": PhysicsFeatureConcept(
        "absolute_formal_charge",
        "population-weighted magnitude of net formal charge",
        "speciation bookkeeping; not total local charge or zwitterionic burden",
        "solution speciation before membrane/receptor encounter",
        "micro-pKa; tautomer free energy; net-zero zwitterions are indistinguishable from nonionic states; ionic strength; assay pH",
        "quality-control and sensitivity context only",
        "quality_control_only",
        "speciation_charge_moments",
        "none",
        "Absolute net charge is zero for both a nonionic state and a net-zero zwitterion, so it cannot estimate the intended charge burden.",
        "Replace with separate positive/negative formal-charge burdens and accessible charge surfaces; require neutral-zwitterion discrimination.",
    ),
}


_SPECIAL_CONCEPTS: dict[str, PhysicsFeatureConcept] = {
    "joint_weight_sum": PhysicsFeatureConcept(
        "joint_weight_sum",
        "probability normalization",
        "none; numerical closure only",
        "ensemble bookkeeping",
        "floating-point and join errors",
        "quality control only",
        "quality_control_only",
        "normalization",
        "none",
        "A constant normalization check contains no compound mechanism.",
        "Require a value of one within numerical tolerance.",
    ),
    "effective_joint_state_conformer_count": PhysicsFeatureConcept(
        "effective_joint_state_conformer_count",
        "inverse-Simpson effective sampled support",
        "none directly; indicates whether conclusions depend on few generated hypotheses",
        "sampling diagnostics",
        "requested conformer count; state cap; force-field weights; clustering",
        "convergence and escalation diagnostic only",
        "quality_control_only",
        "ensemble_diversity",
        "none",
        "The value is driven by algorithmic sampling depth and cannot be treated as a molecular property.",
        "It must stabilize under nested conformer counts and state-retention thresholds.",
    ),
    "joint_conformational_entropy_nats": PhysicsFeatureConcept(
        "joint_conformational_entropy_nats",
        "unnormalized entropy of approximate joint weights",
        "conformational multiplicity",
        "solution ensemble",
        "state/conformer count; MMFF/UFF minima are not free energies",
        "sampling diagnostic only",
        "mechanistic_diagnostic_only",
        "ensemble_diversity",
        "none",
        "Unnormalized entropy increases with enumeration size and is not comparable across capped state sets.",
        "Nested 25/50/100/250/500 sampling and explicit-solvent entropy ordering must agree.",
    ),
    "joint_conformational_entropy_normalized": PhysicsFeatureConcept(
        "joint_conformational_entropy_normalized",
        "normalized dispersion of approximate state-conformer weights",
        "generated-ensemble diversity diagnostic, not conformational-route availability",
        "sampling ensemble",
        "state/conformer cap; MMFF/UFF energies lack solvent and entropic basin volumes; tautomer free energies absent",
        "sampling convergence and escalation diagnostic only",
        "quality_control_only",
        "ensemble_diversity",
        "none",
        "Normalizing by log(N) does not turn enumerated ETKDG/MMFF/UFF weights into equilibrium basin probabilities or thermodynamic entropy.",
        "Replace with converged metastable-basin populations and transition rates validated against independent enhanced sampling or NMR exchange.",
    ),
}


_COMPOSITE_CONCEPTS: dict[str, PhysicsFeatureConcept] = {
    "folded_low_polarity_fraction": PhysicsFeatureConcept(
        "folded_low_polarity_fraction",
        "within-ensemble coupling of compactness and low polarity",
        "formation of membrane-compatible conformers",
        "solution-to-membrane selection",
        "thresholds are each compound's own medians; no absolute folded-state definition; force-field weights",
        "matched-pair diagnostic only",
        "mechanistic_diagnostic_only",
        "folding_response",
        "none",
        "Self-normalized medians measure shape-polarity coupling, not an absolute folded population.",
        "Use common physical thresholds or explicit-solvent state definitions and compare to NMR/permeability.",
    ),
    "exposure_adjusted_hbond_burden": PhysicsFeatureConcept(
        "exposure_adjusted_hbond_burden",
        "fractional solvent exposure of donor and acceptor surface",
        "hydrogen-bond dehydration during membrane insertion or cavity entry",
        "aqueous-to-membrane and ligand-to-cavity transfer",
        "donor/acceptor chemistry is combined; solvent/lipid competition absent; SASA radii and protonation",
        "component audit and one-at-a-time falsification only",
        "hypothesis_falsification_only",
        "hbond_exposure_burden",
        "exact",
        "Combining donor and acceptor exposure erases their experimentally demonstrated asymmetry, while total-SASA normalization can dilute an unchanged absolute burden in larger molecules.",
        "Matched donor-to-acceptor swaps must justify any future combination; otherwise keep site-resolved donor and acceptor hydration terms separate.",
    ),
    "intramolecular_shielding_candidate": PhysicsFeatureConcept(
        "intramolecular_shielding_candidate",
        "IMHB opportunity normalized by accessible polar area",
        "intramolecular compensation for polar desolvation",
        "solution-to-low-dielectric transfer",
        "distance-only IMHB proxy; no occupancy, geometry, exchange kinetics, or solvent competition",
        "one-at-a-time hypothesis with explicit falsification",
        "hypothesis_falsification_only",
        "intramolecular_shielding",
        "exact",
        "Normalization separates shielding opportunity from merely having more polar atoms.",
        "Explicit-solvent IMHB occupancy/NMR must confirm that the proposed contacts persist and mask polarity.",
    ),
    "charge_separation_per_gyration_candidate": PhysicsFeatureConcept(
        "charge_separation_per_gyration_candidate",
        "size-normalized electrostatic charge topology",
        "orientation at membrane and complementary hERG cavity recognition",
        "membrane interface and receptor encounter",
        "Gasteiger charges; dielectric polarization; formal charge; coordinate uncertainty",
        "one-at-a-time hypothesis with charge-model sensitivity",
        "hypothesis_falsification_only",
        "electrostatic_topology",
        "exact",
        "It replaces the size-confounded raw centroid distance and the correlated dipole proxy.",
        "Matched-pair ordering must survive alternative charge models and receptor/membrane environments.",
    ),
    "environment_conditioned_polarity_response_surrogate": PhysicsFeatureConcept(
        "environment_conditioned_polarity_response_surrogate",
        "descriptor-axis polarity response",
        "hypothesized chameleonic polarity hiding in low dielectric",
        "solution-to-membrane selection",
        "no solvent free energy; fixed arbitrary axis coefficients; reuses polar exposure, charge, IMHB and Rg",
        "pilot selection and one-at-a-time falsification only",
        "hypothesis_falsification_only",
        "descriptor_reweighting_axis",
        "none",
        "It is a sensitivity coordinate, not an independently calculated solvent-conditioned population.",
        "Explicit water/low-dielectric MD or NMR must reproduce sign and rank before model use.",
    ),
    "environment_conditioned_shape_response_surrogate": PhysicsFeatureConcept(
        "environment_conditioned_shape_response_surrogate",
        "descriptor-axis shape response",
        "hypothesized low-dielectric compaction",
        "solution-to-membrane selection",
        "same uncalibrated reweighting axis as polarity response; no explicit solvent; Rg size coupling",
        "diagnostic companion to the polarity-response surrogate",
        "mechanistic_diagnostic_only",
        "descriptor_reweighting_axis",
        "none",
        "It is not independent evidence from the polarity response and therefore is not co-modeled.",
        "Explicit environment MD must reproduce sign and relative magnitude.",
    ),
    "water_to_low_dielectric_folded_fraction_shift_surrogate": PhysicsFeatureConcept(
        "water_to_low_dielectric_folded_fraction_shift_surrogate",
        "descriptor-axis folded-mask response",
        "hypothesized environment-triggered folding",
        "solution-to-membrane selection",
        "same uncalibrated reweighting axis; self-relative median mask; no solvent thermodynamics",
        "diagnostic companion only",
        "mechanistic_diagnostic_only",
        "descriptor_reweighting_axis",
        "none",
        "It is a third view of the same perturbation and must not be counted as an independent mechanism.",
        "Explicit environment MD must reproduce the direction with a prespecified folded-state definition.",
    ),
    "hydration_shedding_imhb_compensation_surrogate": PhysicsFeatureConcept(
        "hydration_shedding_imhb_compensation_surrogate",
        "algebraic dehydration burden minus shielding compensation",
        "hypothesized net cost of shedding hydration",
        "aqueous-to-membrane transfer",
        "recombines selected charge, polar exposure, H-bond exposure, and IMHB components; not hydration free energy",
        "one-at-a-time hypothesis ablation only",
        "hypothesis_falsification_only",
        "hydration_component_recombination",
        "none",
        "Co-modeling it with its components would duplicate the same evidence and obscure causal interpretation.",
        "Hydration/transfer free energies and passive permeability must support incremental value over components.",
    ),
    "rare_state_transport_dominance_surrogate": PhysicsFeatureConcept(
        "rare_state_transport_dominance_surrogate",
        "coupling of rare state population to low-dielectric descriptor propensity",
        "rare neutral or shielded states may dominate membrane flux",
        "solution speciation to membrane crossing",
        "state populations are approximate; propensity is not permeability; 5% rare threshold arbitrary; transition kinetics absent",
        "pilot prioritization and one-at-a-time falsification only",
        "hypothesis_falsification_only",
        "rare_state_transport",
        "none",
        "The causal idea is important, but the present P_s surrogate is not strong enough for the predictive layer.",
        "pH-dependent permeability plus state-resolved PMF/diffusivity must validate f_s*P_s contributions.",
    ),
}


# Compatibility name: these are the small, fail-closed *provisional proxy*
# matrix, not a final or decision-track physical parameter set.  The former
# 11-column allowlist was reduced after the primary-evidence/causal audit to
# two distinct, explicitly provisional exposure/contact hypotheses; context,
# hypothesis-only, and QC quantities remain in the evidence tables.
MODEL_PHYSICS_FEATURES: tuple[str, ...] = (
    "polar_sasa_ang2__q05",
    "nonpolar_sasa_ang2__mean",
)

MODEL_PHYSICS_FEATURE_BLOCKS: dict[str, tuple[str, ...]] = {
    # Retain the stable public layer name while narrowing its contents.
    "physics_conformation_and_exposure": (
        "polar_sasa_ang2__q05",
        "nonpolar_sasa_ang2__mean",
    ),
}

# Raw conformer bags need their own fail-closed contract. These columns are
# physical primitives with distinct roles; algorithmic identifiers, force-field
# energies, exact aliases, algebraic surface closure, thresholded counts, raw
# PMI values, and correlated charge proxies are deliberately excluded.
MODEL_CONFORMER_FEATURES: tuple[str, ...] = (
    "polar_sasa_ang2",
    "nonpolar_sasa_ang2",
    "exposed_hbd_sasa_ang2",
    "exposed_hba_sasa_ang2",
    "radius_of_gyration_angstrom",
)


def classify_physics_feature(feature: str) -> PhysicsFeatureConcept | None:
    """Return the single ontology concept covering a physics-summary column."""

    if feature in _SPECIAL_CONCEPTS:
        return _SPECIAL_CONCEPTS[feature]
    if feature.startswith("composite_pka_sensitivity_span__"):
        name = feature.removeprefix("composite_pka_sensitivity_span__")
        base = _COMPOSITE_CONCEPTS.get(name)
        if base is None:
            return None
        return PhysicsFeatureConcept(
            feature_family=f"pka_sensitivity_span::{name}",
            physical_phenomenon="sensitivity of the feature to the declared +/-1 pKa scenarios",
            biological_event="uncertainty propagation, not a separate biological event",
            causal_location="microstate-speciation uncertainty",
            hidden_variables_and_confounders=base.hidden_variables_and_confounders,
            permissible_model_roles="uncertainty flag, abstention, and experiment prioritization only",
            status="uncertainty_only",
            redundancy_group=f"uncertainty::{base.redundancy_group}",
            selected_aggregation="none",
            rationale="Uncertainty magnitude must not be presented as an independent causal predictor.",
            falsification_test="Replace approximate pKas with measured micro-pKas and verify interval contraction.",
        )
    if feature.startswith("composite__"):
        return _COMPOSITE_CONCEPTS.get(feature.removeprefix("composite__"))
    if "__" in feature:
        primitive, _aggregation = feature.rsplit("__", 1)
        return _PRIMITIVE_CONCEPTS.get(primitive)
    return None


def selected_model_physics_features(columns: list[str] | pd.Index) -> list[str]:
    """Select only predeclared, nonredundant discovery predictors.

    The order is fixed for deterministic model matrices.  A new numeric column
    is ignored until it receives a causal interpretation and an explicit
    selection decision here.
    """

    available = set(str(column) for column in columns)
    return [feature for feature in MODEL_PHYSICS_FEATURES if feature in available]


def selected_model_conformer_features(columns: list[str] | pd.Index) -> list[str]:
    """Select the predeclared physical primitives allowed in conformer MIL."""

    available = set(str(column) for column in columns)
    return [feature for feature in MODEL_CONFORMER_FEATURES if feature in available]


def feature_ontology_frame() -> pd.DataFrame:
    """Return the reviewable ontology for primitives, composites, and controls."""

    records: list[dict[str, str]] = []
    for output_kind, concepts in (
        ("distribution_primitive", _PRIMITIVE_CONCEPTS),
        ("ensemble_or_qc", _SPECIAL_CONCEPTS),
        ("mechanism_composite", _COMPOSITE_CONCEPTS),
    ):
        for name, concept in concepts.items():
            record = asdict(concept)
            record["output_kind"] = output_kind
            record["column_selector"] = (
                name
                if output_kind == "ensemble_or_qc"
                else f"composite__{name}"
                if output_kind == "mechanism_composite"
                else f"{name}__(mean|sd|q05|q50|q95)"
            )
            selected = [
                feature
                for feature in MODEL_PHYSICS_FEATURES
                if feature == name or feature.startswith(f"{name}__") or feature == f"composite__{name}"
            ]
            record["selected_model_columns"] = ";".join(selected)
            records.append(record)
    records.append(
        {
            "feature_family": "composite_pka_sensitivity_span::*",
            "physical_phenomenon": "declared +/-1 pKa sensitivity",
            "biological_event": "uncertainty propagation, not a separate event",
            "causal_location": "microstate-speciation uncertainty",
            "hidden_variables_and_confounders": "unknown microscopic pKas and coupled-site equilibria",
            "permissible_model_roles": "uncertainty flag, abstention, and experiment prioritization only",
            "status": "uncertainty_only",
            "redundancy_group": "uncertainty",
            "selected_aggregation": "none",
            "rationale": "Sensitivity spans are not independent molecular mechanisms.",
            "falsification_test": "Measured micro-pKas should contract the sensitivity span.",
            "output_kind": "uncertainty",
            "column_selector": "composite_pka_sensitivity_span__*",
            "selected_model_columns": "",
        }
    )
    return (
        pd.DataFrame(records)
        .sort_values(["output_kind", "feature_family"], kind="stable")
        .reset_index(drop=True)
    )


def conventional_feature_ontology_frame() -> pd.DataFrame:
    """Return the claim-safe roles of all conventional model inputs."""

    frame = pd.DataFrame(asdict(concept) for concept in _CONVENTIONAL_CONCEPTS)
    frame["selected_internal_descriptor"] = frame["feature"].isin(CONVENTIONAL_DESCRIPTOR_COLUMNS)
    return frame.reset_index(drop=True)


__all__ = [
    "CONVENTIONAL_DESCRIPTOR_COLUMNS",
    "MODEL_CONFORMER_FEATURES",
    "MODEL_PHYSICS_FEATURES",
    "MODEL_PHYSICS_FEATURE_BLOCKS",
    "ConventionalFeatureConcept",
    "PhysicsFeatureConcept",
    "classify_physics_feature",
    "conventional_feature_ontology_frame",
    "feature_ontology_frame",
    "selected_model_conformer_features",
    "selected_model_physics_features",
]
