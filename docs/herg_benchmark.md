# Private/public hERG benchmark

This workflow compares three training regimes against the same confidential-domain evaluation folds:

1. `confidential_only`: only decisive lab IC50 labels.
2. `confidential_prioritized`: public primary hERG data plus lab rows weighted five-fold by default.
3. `equal_importance`: public and lab rows receive equal source weight.

The binary policy is configurable. By default, IC50 at or below 10 µM is a blocker, IC50 at or above 30 µM is a nonblocker, and the intermediate interval is not assigned a binary label. One-sided measurements such as `<0.37` and `>30` are used only when their interval guarantees the class.

## Run

```bash
.venv/bin/python pipeline/scripts/run_herg_benchmark.py \
  --workbook /absolute/path/to/lab_workbook.xlsx \
  --profile quick \
  --output research/benchmarks/herg/quick
```

The `quick` profile evaluates two complexity levels for logistic regression, random forest, SVM, KNN, XGBoost, LightGBM, clustering-plus-logistic, and a character-level recurrent network over three chemical feature sets. The `full` profile expands molecular representations, parameter grids, folds, repeats, and scaffold-grouped evaluation. All choices are controlled in `config/herg_benchmark.yaml`.

## Outputs

- `calculated_molecular_parameters.csv`: source rows plus calculated RDKit parameters.
- `feature_registry.json`: exact feature dimensions and descriptor stabilization metadata.
- `model_parameter_grid.csv`: every requested model/feature/parameter combination.
- `fold_results.csv`: fold-level metrics and failures.
- `oof_private_predictions.csv`: held-out confidential predictions used for comparison.
- `cv_results.csv`: aggregate candidate ranking.
- `best_models.csv`: production-ensemble membership.
- `private_compound_predictions.csv`: molecule-level probabilities for all lab rows.
- `research/models/`: fitted production estimators.
- `run_manifest.json`: data hashes, software versions, counts, and execution status.

The quick profile uses repeated stratified structure folds for rapid iteration. It is an optimization screen, not the final publication estimate. Publication claims should use the scaffold-grouped full profile, nested selection, confidence intervals, and—when available—a later locked prospective cohort.

## Nested scaffold validation

```bash
.venv/bin/python pipeline/scripts/run_herg_validation.py \
  --workbook /absolute/path/to/lab_workbook.xlsx \
  --output research/benchmarks/herg/nested_scaffold
```

This second phase uses five outer confidential-scaffold folds. Inside every outer training partition, three additional scaffold folds drive successive-halving model/feature/complexity selection, sigmoid calibration, decision-threshold selection, and diverse-family ensemble selection. Outer confidential scaffolds remain untouched until all those choices are frozen. Outputs include fold selections, every inner fit and prediction, calibrated outer predictions, scaffold-bootstrap intervals, and private/public applicability-domain diagnostics.
