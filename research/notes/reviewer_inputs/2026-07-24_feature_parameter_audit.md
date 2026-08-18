# Reviewer input: feature and parameter semantics

## Source

User-supplied critique integrated on 2026-07-24 from:

Original Codex attachment ID: `fa7f905d-ee2c-4b48-b32f-639713734cb8` (`pasted-text.txt`).

This note records the directives applied to the project; the authoritative
integrated interpretation is
`../../reports/pk_herg/reviewer_audit/integrated_feature_and_parameter_ontology.md`.

## Directives incorporated

- Do not treat conventional descriptors or learned representations as
  fundamental parameters.
- Define independence by causal node/transition, not statistical
  decorrelation.
- Separate conventional controls, direct measurements, fundamental free
  energies/rates, derived functionals, and QC/uncertainty.
- Remove redundant exact-mass/heavy-atom axes and constant submitted-state
  charge from the internal descriptor matrix.
- Demote SASA, compactness, charge, IMHB, entropy, environment-response, and
  rare-state surrogate features to declared proxy/diagnostic/QC roles.
- Replace static descriptor composites with state-, path-, and flux-resolved
  free energies, rates, PMF/diffusivity, and protocol observation models.
- Treat global chameleonicity/ETR guidance as portfolio-conditioned and
  falsifiable, not universal.
- Separate PK and hERG causal parent graphs and prohibit parent/derived double
  counting.
- Center novelty on a State–Path–Flux integration with intervention evidence,
  not on any individual descriptor.

## Implementation trace

- Conventional role contract:
  `pipeline/src/menin_discovery/research_feature_ontology.py`
- Fundamental/derived parent graph:
  `pipeline/src/menin_discovery/research_parameter_ontology.py`
- Quantitative regeneration:
  `pipeline/src/menin_discovery/research_reviewer_audit.py`
- Claim-controlled reviewer bundle:
  `research/reports/pk_herg/reviewer_audit/`
