# Drugs@FDA record-level candidate normalization

## Result

- Normalized 29,253 applications, 51,619
  products, 193,242 submissions, 208,723
  submission-action links, 80,714 document pointers,
  and 60,157 semicolon-projected ingredient-name candidates.
- Preserved 7,220 explicit malformed, orphan,
  blank-key, or duplicate-key quarantine records.
- Approval, marketing status, submission status, action category, document presence, and
  ingredient names remain regulatory metadata—not efficacy, safety, QT, PK, binding,
  causality, molecular activity, negative outcomes, or model labels.

## Method

- Reverified the frozen FDA acquisition manifest and every recursive source-bundle hash.
- Parsed the exact 12-table archive under fixed member, column, encoding, width, and row-count
  contracts. One malformed ApplicationDocs row was quarantined rather than repaired.
- Joined only exact FDA relational keys. Missing keys are retained as source orphans.
- Split `ActiveIngredient` only on source semicolons and trimmed whitespace. Commas, `AND`,
  salts, mixtures, stereochemistry, and names were not chemically interpreted.
- Document URLs were retained as pointers; document content was not downloaded.
- DailyMed remains archive-inventory only. Its 17.8 GB source archives were not opened.

## Limits and gates

- Ingredient strings require curated structure/identifier resolution before molecule linkage.
- Regulatory actions and status require indication, formulation, jurisdiction, and temporal
  context and cannot support biomedical labels by themselves.
- Source rights/site-policy and institutional review remain required before redistribution.
- Application-document malformed/orphan records require source-steward adjudication.
- No missing record is interpreted as a negative outcome.
