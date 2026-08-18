# Evidence review: Mavroudis et al. (2023)

## Citation and project role

Mavroudis, Teutonico, Abos, and Pillai, "Application of machine learning in combination with mechanistic modeling to predict plasma exposure of small molecules," *Frontiers in Systems Biology* 3 (2023), 1180948. DOI: `10.3389/fsysb.2023.1180948`.

This is an important hybrid-model precedent: molecular structure is mapped to pharmacokinetic or physicochemical inputs with machine learning, and mechanistic mass-balance models map those inputs to full concentration-time profiles. The paper is a useful architectural baseline, not a direct solution for large-molecule PK.

## Reported study design

The target experiment is rat intravenous dosing at 1 mg/kg. Historical data were generated at one Sanofi site to reduce intersite assay variability. The combined property and PK-profile collections contain 637 unique compounds; 61 compounds have all required properties and a concentration-time profile and are held out as a common test set across models.

Available endpoint counts differ substantially:

- 530 compounds with in vivo rat clearance and steady-state volume of distribution from noncompartmental analysis;
- 451 with liver-microsome intrinsic clearance;
- 459 with a most-acidic pKa value;
- 324 with a most-basic pKa value;
- 188 with fraction unbound; and
- 397 with rat concentration-time profiles.

After holding out the common 61-compound set, the non-pKa training counts are 469 for clearance and Vdss, 390 for CLint, and 127 for fraction unbound. The pKa documentation is internally inconsistent: the availability paragraph implies 398 most-acidic and 263 most-basic training compounds, while Table 2 assigns 398 to most-basic and 263 to most-acidic. The endpoint labels or counts appear reversed, and the attachments do not resolve which mapping reached the fitted models.

### Learned property models

The paper compares random forest, XGBoost, support-vector regression, and a message-passing neural network across 1,024-bit Morgan fingerprints, approximately 200 RDKit descriptors, and molecular graphs. Ten-fold cross-validation within the training data is used for hyperparameter selection. Molecular weight and logP are generated with RDKit rather than fitted as experimental-property models.

The selected models and reported common-test performance are:

| Property | Selected model/representation | MAPE | RMSE |
|---|---|---:|---:|
| In vivo CL | XGBoost/fingerprints | 0.82 | 15.74 mL/min/kg |
| Vdss | XGBoost/descriptors | 0.57 | 961 mL/kg |
| CLint | XGBoost/descriptors | 0.75 | 30.6 microliter/min/mg |
| Row labeled most-basic pKa in Table 2 | SVR/fingerprints | 0.28 | 1.47 |
| Row labeled most-acidic pKa in Table 2 | random forest/fingerprints | 0.054 | 1.2 |
| Fraction unbound | random forest/descriptors | 1.11 | 0.052 |

Deep graph models underperformed the classical approaches in this small-data setting and tended toward the mean. This is useful evidence against assuming that model complexity alone produces mechanistic or predictive improvement.

### Exposure models

The learned inputs feed either:

1. a one-compartment IV model parameterized by CL and Vdss; or
2. a whole-body PK-Sim PBPK model parameterized with either in vivo CL or microsomal CLint plus molecular weight, logP, acidic/basic pKa, fraction unbound, and halogen count.

The PBPK workflow compares Berezhkovskiy, PK-Sim standard, Poulin-Theil, Rodgers-Rowland, and Schmidt tissue-distribution models. With in vivo CL, reported median fold differences in observed versus predicted AUClast are generally between one- and two-fold depending on distribution model. Cmax varies less across distribution models. When microsomal CLint is the only clearance route, AUC is systematically overpredicted because renal clearance, transport, and other elimination mechanisms are absent. Cmax remains within roughly two- to three-fold at the median but is variable.

## What the architecture gets right

- It predicts a time course through conservation laws instead of predicting AUC or Cmax as isolated labels.
- It exposes intermediate failure points: a poor exposure profile can be traced to CL, Vdss, fu, pKa, a distribution model, or an omitted pathway.
- It compares alternative distribution hypotheses instead of reporting one PBPK curve as certain.
- It demonstrates that an apparently more mechanistic input, microsomal CLint, can worsen exposure prediction if relevant clearance mechanisms are missing.
- It shows why endpoint-specific data volume and noise matter: fraction unbound is the sparsest and least stable learned input.

## Fundamental limitations for the present project

### Derived endpoints can create circularity

For the one-compartment model, CL and Vdss are derived by noncompartmental analysis of the same kind of IV concentration-time profiles the model reconstructs. Predicting those summaries and inserting them into an IV equation can yield a good curve without discovering the biochemical or physiological causes of clearance and distribution. It is a valid predictive shortcut, but not a primitive-process explanation.

The next model should instead represent latent causal quantities such as unbound microsomal clearance, blood/plasma partition, renal filtration/secretion/reabsorption, transporter uptake and efflux, tissue binding, membrane permeability, and organ blood flow. Observed CL, Vdss, AUC, Cmax, and half-life should be consequences with their own observation error, not interchangeable input features.

### IV dosing omits the largest oral-PK challenges

The study bypasses dissolution, luminal speciation, passive permeation, transporter competition, enterocyte metabolism, and first-pass bioavailability. Those processes are precisely where large, flexible, chameleonic molecules can depart from small-molecule rules. The architecture transfers; the process graph and data do not.

### Chemical and validation domain

The study is described as small-molecule rat PK, uses only 61 common test compounds, and does not provide a prospective large-molecule stratum. Public reproducibility is limited by proprietary structures/profiles and absent split/code artifacts. The supplied supporting PDF contains figures and summary tables, not a machine-readable compound-level dataset. The pKa count/label reversal described above and duplicate rows in supporting Table S2 reinforce the need for executable manifests and typed endpoint definitions.

## Project adaptation

Use this work as the Tier-0 hybrid baseline, then deepen it in stages:

1. fit transparent property models and a compartmental mass-balance model exactly as a comparator;
2. replace derived CL/Vdss inputs with separately measured primitive processes where available;
3. introduce latent-process inference from raw concentration-time data with partial pooling across assays and species;
4. add oral absorption states only when data support them, explicitly separating dissolution, microstate-weighted membrane crossing, efflux, gut metabolism, and hepatic first pass;
5. propagate input uncertainty through the ODE model to exposure intervals;
6. compare time-resolved residuals, not only AUC and Cmax; and
7. perform mechanism-removal and pathway-misspecification tests to learn why a model succeeds or fails.

The minimum metrics are input-property MAE/RMSE/bias/calibration, concentration-time weighted error, log-scale residuals by sampling time, AUClast and Cmax fold error, half-life and MRT error, interval coverage, and mass-balance checks. Performance must be stratified by molecular size, ionization, flexibility, exposed-polarity behavior, clearance regime, and applicability distance.

## Bottom line

Mavroudis et al. demonstrate that learned molecular properties can be productively coupled to mechanistic PK equations and that missing pathways produce interpretable exposure errors. The present project should retain that modularity while moving causality one level deeper: fit primitive physical and biological processes, treat summary PK parameters as observations, and validate on raw time courses in the intended large-molecule domain.
