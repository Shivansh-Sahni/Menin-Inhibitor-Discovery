# Evidence review: Price et al. 2024 polarity reducers

## Source and role

Edward Price et al., *Beyond Rule of Five and PROTACs in Modern Drug Discovery:
Polarity Reducers, Chameleonicity, and the Evolving Physicochemical Landscape*,
J. Med. Chem. 2024, DOI 10.1021/acs.jmedchem.3c02332.

This is an important industrial-scale precedent for experimental exposed polarity and
intestinal absorption. It is not direct calibration evidence for the internal series and
does not establish that its scalar thresholds are causal or portable.

## What was actually measured or derived

- Approximately 1,000 literature human fraction-absorbed records, more than 10,000
  internal rodent `faFg` records, and about 3,000 EPSA measurements were analyzed; the
  underlying compound-level internal data are not released.
- EPSA was derived from calibrated supercritical-fluid chromatography retention. ETR
  was defined as `EPSA/TPSA`; it is therefore a ratio of an experimental assay response
  to a 2D calculated maximum-polarity proxy, not a microscopic free energy or a direct
  conformational population.
- Internal `faFg` was not directly observed. Bioavailability came from dose-normalized
  PO/IV exposure, hepatic availability was approximated as `Fh = 1 - CL/QH`, and
  `faFg = F/Fh`. Error and biological misspecification in clearance, liver flow, or
  extrahepatic clearance therefore propagate into the label.
- Rat and mouse `faFg` values were averaged even though the reported cross-species
  correlation was only about 0.5. The analysis assumed no solubility limitation because
  fully solubilized low-dose formulations were used.
- The reported ETR boundaries (about 0.8 for MW 500-800 and 0.6 for MW 800-1000 among
  polar compounds) are empirical, bin- and portfolio-conditioned enrichment rules.

## Mechanistic contribution

The study supports three defensible propositions:

1. nominal TPSA can substantially overstate the polarity presented by some bRo5
   molecules in a nonaqueous assay environment;
2. the magnitude of useful polarity response depends jointly on size, baseline polarity,
   lipophilicity, and chemical class; and
3. different bRo5 subsets can reach absorption through different mechanisms.

It does **not** show that low ETR uniquely means dynamic chameleonic folding. The same
ratio may reflect static shielding, persistent IMHBs, steric occlusion, assay chemistry,
or other effects. It also does not identify dissolution, passive membrane flux, efflux,
gut metabolism, or hepatic first pass separately.

## External contradiction and project decision

Le Manach et al. 2026 (DOI 10.1021/acsmedchemlett.6c00043) evaluated published
chameleonicity guidelines in a different AstraZeneca PROTAC portfolio and did not find
the Price ETR trend. In that data, efflux ratio was more useful for enriching oral
absorption. This is strong evidence against treating ETR as a universal causal parameter.

Accordingly:

- ETR, EPSA, AB-MPS, TPSA, and their published cutoffs remain external comparators and
  assay hypotheses, not novel project features or decision rules.
- The project replaces the scalar ratio with state-, site-, and path-conditioned polarity,
  desolvation compensation, and reactive-flux weighting.
- The oral model adds explicit competition among basolateral escape, apical efflux,
  enterocyte metabolism, and re-entry. The observation to reproduce is protocol-specific
  efflux/recovery, not a single transporter-free permeability number.
- Any ETR-like relationship must be re-estimated inside a training fold, stratified by
  series and assay, and must survive the internal matched-pair and prospective gates.

## Missing evidence

The supporting information, row-level structures, rodent studies, formulation metadata,
EPSA replicates, and derivation uncertainty for `faFg` are not local. They are required
for quantitative reproduction. Until then, this article informs causal architecture and
falsification, not parameter calibration.

