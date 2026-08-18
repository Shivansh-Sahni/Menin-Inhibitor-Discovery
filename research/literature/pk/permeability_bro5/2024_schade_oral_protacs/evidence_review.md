# Evidence review: Schade et al. 2024 oral PROTACs

## Source and role

Markus Schade et al., *Structural and Physicochemical Features of Oral PROTACs*,
J. Med. Chem. 2024, DOI 10.1021/acs.jmedchem.4c01017.

This is unusually relevant because its proprietary series overlaps the project's molecular
weight regime and combines NMR, physicochemical assays, and mouse/rat/dog PK. It remains
a selected oral-PROTAC data set rather than a general large-molecule law.

## Strong evidence

- The authors measured solvent exposure of individual donors and acceptors using
  variable-temperature and HBD-acidity NMR in DMSO and chloroform rather than inferring
  shielding from a minimized structure.
- Four clinical-stage oral PROTACs and thirty advanced preclinical compounds selected for
  oral behavior had one or two solvent-exposed donors in an apolar environment. Within
  a related series, compounds assigned three exposed donors showed much poorer oral
  bioavailability, with donor capping providing an intervention rather than a mere
  cross-sectional correlation.
- Shielding was site-specific. Some donors were hidden by conventional IMHB geometry;
  others were sterically or weakly shielded only in chloroform and would be missed by
  static IMHB counts.
- The clinical compounds were predominantly extended free in solution. Useful local
  shielding therefore did not require global end-to-end collapse. This directly warns
  against equating low radius of gyration with permeability.
- Bioactive preorganization and stereochemistry affected target/E3 recognition, showing
  that the same conformational landscape can influence PK and binding through distinct
  local coordinates.

## Boundaries and confounders

- The thirty-compound extension was enriched for compounds already showing favorable
  oral bioavailability across species; it cannot estimate an unbiased probability of
  oral success.
- Oral bioavailability varied by as much as tenfold across mouse, rat, and dog. A universal
  property threshold cannot substitute for a species- and protocol-aware process model.
- Dosing used 1 mg/kg and a uniform solubility-enabling cyclodextrin formulation. These
  results do not identify dissolution-limited behavior under other formulations.
- Several high-lipophilicity PROTACs had unreliable free-fraction measurements. Total
  exposure is not interchangeable with free tissue or hERG exposure.
- Exposed-donor thresholds depend on solvent, temperature, NMR criterion, tautomer and
  protonation state. `eHBD <= 2` is a useful experimental boundary for this domain, not a
  universal biological constant.
- The authors explicitly noted that conventional Caco-2/PAMPA measurements can be
  unreliable for very lipophilic PROTACs; later portfolio evidence nevertheless found
  valid efflux ratios informative. Recovery, censoring, inhibitors, and assay validity must
  therefore be modeled rather than accepting or rejecting the entire assay class.

## Project consequences

1. Replace global compactness as the main hypothesis with **local, site-resolved shielding
   conditional on environment and path position**.
2. Preserve donor and acceptor identities separately. A donor-water contact lost at the
   membrane must be paired with the specific intramolecular, lipid, or receptor contact
   that compensates it.
3. Measure both the probability of a transport-competent state and the time required to
   form it. An extended but locally shielded state can be competent.
4. Use NMR eHBD/eHBA and EPSA as future experimental validators of simulated exposure,
   not as replacements for the underlying transition model.
5. Treat formulation, species, free fraction, efflux, and clearance as competing causal
   explanations for oral exposure.

The distinctive project proposal is not another eHBD count. It is a path-conditioned
desolvation-compensation network linked to membrane flux, transporter recycling, systemic
exposure, and hERG access, with each link separately falsifiable.

## Missing evidence

The article supporting information, NMR peak tables/spectra, molecular ensembles, raw
PK profiles, per-animal data, assay recovery, and proprietary structures are not local.
They are required for quantitative reproduction and uncertainty estimation.

