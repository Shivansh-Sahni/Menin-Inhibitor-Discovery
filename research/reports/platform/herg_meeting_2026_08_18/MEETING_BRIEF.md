# hERG Project Meeting Brief

## Headline

The V9 domain-mixture campaign produced the strongest honest internal result so far: **MAE 0.4328 pIC50** across **18,801 structures and 8,455 scaffold groups**. It improved over V8 by **0.0114 pIC50 MAE** in paired scaffold analysis (95% CI **-0.0141 to -0.0086** for V9 minus V8), and the improvement occurred in all five outer folds.

## Major findings

1. **Accuracy improved without relaxing the evaluation.** The V9 honest nested stack reached 69.8% within 0.5 pIC50 and 91.8% within 1.0 pIC50. Repository validation and test outcomes remained sealed.
2. **The deployable molecular model is not the stack.** The frozen deployable candidate is XGBoost depth-10 (MAE 0.4370); the stack is the best internal cross-fitted evidence and depends on multiple specialists.
3. **2D chemistry carries the robust unique signal.** Removing RDKit2D worsened MAE by 0.0250 and removing Morgan fingerprints worsened it by 0.0065 in held-out scaffold analysis. Generic ligand-only 3D, shape, WHIM, energy/flexibility, and polarity/charge blocks did not show a stable independent aggregate gain.
4. **Uncertainty is operationally useful, but only with the governed domain flag.** The cross-fitted 90% interval covered 91.3% of outcomes. The label-blind abstention flag separates an easier unflagged domain (MAE 0.379) from flagged extrapolative predictions (MAE 0.560). Ranking by interval width alone was nearly flat and is retained as a negative sensitivity result.
5. **Local analog evidence helps where it exists.** MMP analog support covers 7,847/18,801 structures (41.7%) and improves level prediction versus the broad anchor on that covered subset. It does **not** yet solve activity-cliff direction or establish causal transformations.
6. **The remaining ceiling is structured.** Errors concentrate in potency extremes, low-to-moderate similarity, assay/source heterogeneity, activity-cliff members, highly flexible chemistry, and molecules above 700 Da. The 500-700 Da region remains competitive; the small >=700 Da subgroup degrades.
7. **Optimizing for the tails creates a real tradeoff.** V7's safety/tail-selected model reduced tail MAE from 1.425 to 1.354, but worsened global MAE by 0.011 (95% CI 0.008 to 0.014). A safety objective should therefore remain a separately reported operating mode, not replace the accuracy model.

## What was added for this meeting

- Independent replay of all headline metrics and paired scaffold-bootstrap improvement.
- Threshold views at 20, 10, and 1 micromolar for comparison with classification tools.
- Applicability-domain, interval-calibration, and negative interval-width-ranking sensitivity analyses.
- Observation-level label disagreement audit across 27,728 exact measurements.
- Heavy-molecule, potency-tail, similarity, flexibility, modality, and source sensitivity analyses.
- Safety/tail-objective versus global-accuracy tradeoff audit.
- MMP-covered performance audit and a 150-case error atlas.
- Literature/task-comparability review so unlike metrics are not presented as head-to-head superiority.
- Traceable figures, result index, and technical Q&A.

## Important contradictions and limits

- More generic ligand-only 3D features did **not** improve aggregate scaffold transfer; selective physics also performed worse than V8. This is a useful negative result, not evidence that receptor-aware physics is irrelevant.
- The target is **wild-type-or-unspecified hERG quantitative potency**, not fully adjudicated explicit human WT in every record.
- These are internal nested scaffold results, not prospective, external, clinical-QT, or superiority validation.
- Published hERG tools often report binary recall/AUC on different thresholds, datasets, and splits. Those values are not directly comparable to continuous scaffold-held-out MAE.
- The most potent and least potent tails are strongly regressed toward the mean; assay disagreement and heterogeneous protocols remain plausible contributors.
- A tail-weighted objective improves tail balance but significantly worsens overall error, so there is no single metric-free definition of the "best" model.

## Recommended next steps

1. Freeze a truly external, protocol-resolved functional patch-clamp series and evaluate once.
2. Adjudicate explicit human-WT construct and protocol metadata for the highest-value difficult cases.
3. Train assay-conditioned or hierarchical measurement models rather than treating all modalities as interchangeable.
4. Develop a separate activity-cliff/local-delta model and acquire targeted matched pairs.
5. Add microstate and receptor-state physics only after receptor preparation, ligand-state, and software-environment blockers are resolved.
6. Use uncertainty/abstention in deployment and communicate both prediction and applicability domain.
