# Supplemental context-split candidates

These are label-blind **candidate** assignments, not replacements for the
official split suite and not authorization to train. The exact hERG functional
IC50 pilot and exact binding Kd scale task were evaluated with assay-group,
document-group, and strict whole-year temporal rules.

- Materialized: 6
- Skipped at the fixed rule: 0
- Missing assay/document/year values are `excluded_unknown`, never silently assigned.
- Group exclusion and temporal chronology are recomputed exhaustively from every candidate row.
- Test features are used only for leakage auditing. No label column or model-ready lockbox was opened.

Context splits expose a different generalization question from molecular or
scaffold splits. They do not eliminate chemical similarity, protein-family
similarity, source dependence, publication bias, or prospective-validation
requirements. Near-similarity evidence remains in the separately bound deep
leakage report; this package reports exact identity/context overlap only.
