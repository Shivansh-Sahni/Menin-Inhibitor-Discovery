# hERG quality-specific CPU baseline results

These are fixed diagnostic baselines on the entity-exclusive scaffold split. They do not establish predictive superiority, prospective validity, or clinical utility.

## Data contract

- Q1: 22,081 structure-level exact pIC50 targets.
- Q2: 123 IC50/pIC50 structures; 9 are censoring-only constraints.
- Test partitions were evaluated once and never used for fitting or model selection.
- Q2 is deliberately separate from AC50, fixed-dose inhibition, QT, and other endpoint semantics.

## Locked-test metrics

| Task | Model | n | MAE | RMSE | R2 | Spearman |
|---|---|---:|---:|---:|---:|---:|
| Q1 | train_mean | 2,275 | 0.6526 | 0.8659 | -0.0211 | 0.0000 |
| Q1 | morgan_ridge | 2,275 | 0.5817 | 0.7741 | 0.1839 | 0.4318 |
| Q2 | train_mean | 11 | 1.4416 | 1.9193 | -0.2787 | 0.0000 |
| Q2 | descriptor_ridge_exact_only | 11 | 1.2670 | 1.7263 | -0.0345 | 0.1777 |
| Q2 | descriptor_censored_gaussian_ridge | 11 | 1.1418 | 1.6159 | 0.0936 | 0.2551 |

## Interpretation boundary

The project already has an established data-engineering advantage in WT scope, scale, assay semantics, evidence levels, entity-exclusive splitting, and provenance. Model-performance superiority remains unestablished until like-for-like external and prospective comparisons are run.

The censored Gaussian model is a penalized Tobit/interval likelihood, not a Cox survival model. Its present Q2 result is low-power and primarily verifies that bounded IC50 values can be retained instead of discarded or falsely treated as exact.
