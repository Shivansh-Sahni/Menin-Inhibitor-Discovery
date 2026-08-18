# Likely Technical Questions and Answers

## What exactly is the best result?

The strongest unbiased internal evidence is the five-fold nested scaffold OOF V9 stack: MAE 0.4328, RMSE 0.6311, Spearman 0.6907, 69.8% within 0.5, and 91.8% within 1.0 pIC50. It is not an external result.

## Is the improvement statistically supported?

Yes internally. V9 minus V8 MAE is -0.0114, with a 10,000-replicate scaffold-bootstrap 95% CI of -0.0141 to -0.0086. Every outer fold improved. This supports internal robustness, not external superiority.

## Why are published metrics sometimes much better?

Many published tools solve easier or different tasks: binary classification at selected thresholds, random or chemically closer splits, duplicate-rich datasets, or threshold-tuned evaluation. Pred-hERG 5.0's continuous regression is closer, but its dataset and split still differ. Our scaffold-held-out, structure-collapsed, train-only nested evaluation is intentionally harder.

## Did the fundamental/physics features help?

Generic ligand-only 3D families did not add stable aggregate accuracy beyond RDKit2D and Morgan fingerprints. This indicates redundancy or noisy conformer/force-field approximations. It does not test prepared receptor states, membranes, kinetics, or high-quality microstate populations, which remain blocked/deferred.

## Are heavy compounds handled?

Reasonably through 500-700 Da, but not uniformly. The >=700 Da subgroup has only 218 structures and V9 MAE 0.518; treat that estimate as uncertain and do not claim universal macromolecule coverage.

## What is the biggest current error source?

No single source explains all errors. The strongest reproducible patterns are potency-tail regression, lower train-set similarity, assay/source heterogeneity, activity-cliff membership, and measurement disagreement. The case atlas identifies the exact compounds driving each pattern.

## Why not optimize directly for the most safety-relevant potency tails?

We tested that in V7. The safety/tail-selected model reduced tail MAE from 1.425 to 1.354 and improved equal-potency-bin MAE from 0.790 to 0.768, but global MAE worsened by 0.011 (95% CI 0.008 to 0.014). The defensible solution is to report both operating objectives rather than present the tail-weighted model as universally better.

## Does MMP analysis provide a mechanistic discovery?

Not yet. Analog-assisted level predictions improve on covered structures, but activity-cliff delta errors remain large and direction accuracy is weak. The MMP result supports local-context modeling and targeted experimental pairs, not causal transformation rules.

## Can this predict clinical QT risk?

No. The model predicts molecular hERG potency. Clinical QT/QTc depends on exposure, protein binding, metabolites, other ion channels, patient factors, and dosing. The clinical context surface is explicitly separate and contains no training labels.

## Why not train on all 339,373 fixed-dose labels?

That surface is a highly imbalanced binary endpoint with different measurement semantics. It is useful as a separate auxiliary task, but pooling it directly into continuous pIC50 would corrupt the target.

## What should be tested next?

The highest-information next test is a frozen external series with explicit human-WT functional patch-clamp protocols. Computationally, prioritize assay-conditioned models, potency-tail correction, activity-cliff/local analog models, and uncertainty-aware deployment before expensive receptor physics.

## Where is each number?

Use `RESULT_TRACEABILITY.csv` for headline values, `tables/` for the supporting calculations, `figures/` for presentation graphics, and `tables/top_150_error_cases.csv` for compound-level questions.
