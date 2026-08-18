# hERG model landscape v2 (2025-2026 audit)

This directory is a current, reproducible comparison of recent hERG prediction systems and directly useful method adapters. It is deliberately separate from the earlier landscape so that claims can be reconciled before integration.

## Deliverables

- `CURRENT_HERG_MODEL_LANDSCAPE_2025_2026.md`: critical review and project positioning.
- `model_comparison_matrix.csv` and `.json`: machine-readable comparator facts.
- `benchmark_adapter_priority_matrix.csv` and `.json`: implementation and benchmarking priorities.
- `SOURCE_AND_REPRODUCIBILITY_LEDGER.md`: primary sources, repository snapshots, and exact audit commands.
- `validate_landscape.py`: structural and semantic checks for the machine-readable artifacts.

## Claim discipline

The local project has established **data-asset superiority** over the audited public model datasets in several dimensions: explicit wild-type/variant disposition, observation-level lineage, assay modality and automation fields, quality tiers, and a 369,546-unique-structure admitted hERG universe. It has **not established predictive superiority**. The only local baseline located during the parent audit had ROC-AUC 0.662 and average precision 0.00806 at 0.00362 prevalence; those values are not directly comparable to papers using different cohorts and splits.

Headline metrics in this directory are transcribed as reported. They are not ranked across incompatible splits. `NR` means that an exact value was not recoverable from a primary source or official artifact during this audit; it is not an estimate.

## Validation

From the repository root:

```bash
python3 research/reports/platform/herg_paper/model_landscape_v2/validate_landscape.py
```

Last literature/repository audit: 2026-08-07.
