# Literature and Tool Comparability

## The central rule

Do not compare a published binary recall, accuracy, AUROC, or random-split result directly with our continuous scaffold-held-out MAE. Report the task, endpoint threshold, data source, split, duplicate handling, and validation type beside every metric.

## Relevant primary sources

- **Pred-hERG 5.0** used ChEMBL30 (>14,000 compounds; 7,609 regression records) and reported regression MAE 0.35 and RMSE 0.44 on its test design, alongside classification metrics. It is the closest public continuous comparator, but dataset curation and split are not identical to ours: https://pmc.ncbi.nlm.nih.gov/articles/PMC11187631/
- **HERGAI** is primarily a highly imbalanced binary classifier. Its approximately 300,000-molecule surface contains 1,937 blockers versus 297,990 nonblockers, and its 86.4%/94.29% headline values are recall at selected blocker thresholds, not continuous potency MAE: https://pmc.ncbi.nlm.nih.gov/articles/PMC12291323/
- **HergSPred** is a consensus binary classifier using fingerprints and multiple learners, again a different endpoint and evaluation target: https://pubs.acs.org/doi/10.1021/acs.jcim.2c00256
- **hERGBoost** is a relevant quantitative XGBoost publication, but a fair comparison requires its exact external dataset, curation, and split rather than copying a headline metric: https://doi.org/10.1016/j.compbiomed.2024.109416
- A recent hERG AutoML study reported markedly lower MCC under scaffold cross-validation than random splitting, directly illustrating how split choice changes apparent performance: https://pmc.ncbi.nlm.nih.gov/articles/PMC12756696/
- The Step Forward validation paper similarly shows random cross-validation can look substantially better than more out-of-domain assessments: https://pmc.ncbi.nlm.nih.gov/articles/PMC11245006/

## Defensible positioning

Our contribution is not the largest-looking metric. It is a broad, structure-collapsed, scaffold-held-out, uncertainty-aware continuous potency evaluation with explicit assay, source, similarity, mass, cliff, and label-disagreement boundaries. Superiority remains unclaimed until every comparator is replayed on the same frozen structures and endpoints or a common prospective series.
