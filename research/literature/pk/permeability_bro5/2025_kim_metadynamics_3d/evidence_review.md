# Evidence review: Kim et al. (2025)

## Citation and project role

Kim et al., "Enhancing Permeability Prediction of Heterobifunctional Degraders Using Machine Learning and Metadynamics-Informed 3D Molecular Descriptors," *Journal of Chemical Information and Modeling* (2025). DOI: `10.1021/acs.jcim.5c01600`.

The local package contains the SI and a four-page PDF rendering of the supplementary dataset, but not the main article or original machine-readable data. This is directly relevant mechanistic precedent for degrader permeability and environment-conditioned 3D features. Its small, literature-assembled dataset is not a definitive predictive benchmark.

## Study design supported by the SI

The study analyzes 32 heterobifunctional degraders with literature-derived directional apparent permeability. Well-tempered metadynamics in chloroform is reported for 100 ns per compound, yielding 10,000 conformations per molecule. Ensemble-averaged 3D polar surface area, intramolecular hydrogen-bond count, and radius of gyration are combined with conventional descriptor sets.

Random forest, partial least squares, and linear SVM models are evaluated over 100 randomized 50/50 train-test splits. In the SI's standalone regression screen, 3D-only descriptors give the strongest reported mean R2 for PLS (0.442) and RF (0.366), compared with 0.263 and 0.176 for the referenced 2D descriptor set. For median-threshold classification at 7.0 nm/s, the RF 3D model reports ROC-AUC 0.773 +/- 0.110 and accuracy 0.668 +/- 0.117.

Those results support information in conformation-dependent shape and polarity, but random half-splits among closely related degrader series are optimistic evidence for scaffold generalization. The large split-to-split variance and n = 32 must remain visible.

## Physical interpretation

The SI shows why one minimized 3D structure is insufficient:

- ANI optimization changes ensemble descriptors and removes high-energy distorted conformers; reported energy reductions are often 70-130 kcal/mol and can be much larger.
- AMBER- and ANI-weighted ensembles agree unevenly across observables; Rgyr is less stable than IMHB and 3D-PSA.
- Across all 32 compounds, pairwise correlations among Rgyr, IMHB, and 3D-PSA are only moderate (Pearson r = 0.42-0.56), so they are related but not interchangeable.
- Two Rgyr outliers have no IMHB, much higher 3D-PSA, and substantially lower log permeability than the main set despite lower mean MW.
- PEG versus mixed linkers differs in median Papp (10.55 versus 3.7 nm/s, p = 0.029), while the study's rigidity split is not significant for Papp.

These observations are consistent with a causal chain in which linker chemistry controls accessible conformers, intramolecular compensation, and exposed polarity, which in turn alter membrane entry. They do not prove that Rgyr or 3D-PSA alone causes permeability. Chloroform captures a low-dielectric environment but not the membrane interface, lipid accommodation, water defects, transporters, or transbilayer kinetics.

## Dataset and assay limitations

The dataset spans 32 compounds from 14 literature references, multiple targets, and VHL, CRBN, or MDM2 ligase systems. The SI supplies AB, BA, and derived/passive Papp values, but no replicate-level measurements or unified assay-protocol table. Apparent inter-series effects can therefore reflect assay source, cell system, pH, transporter activity, or calculation conventions.

The SI narrative says seven compounds have solubility measurements, but Table S9 lists only six compounds. The listed conditions are nonuniform and range from aqueous buffers to a cosolvent formulation. The values are appropriately excluded from quantitative modeling, and the seven-versus-six discrepancy must remain unresolved until the source data are obtained.

The local dataset is a PDF table, not the named original DOCX or a CSV. Raw metadynamics trajectories, conformer coordinates, force-field inputs, ANI workflow, descriptor code, exact model splits, and fitted artifacts are absent.

## Project adaptation

Use the 32-compound set for mechanistic feature precedent and as a small external stress test. Reproduce it only with grouped series/scaffold/source splits, source-stratified residuals, and uncertainty that reflects n = 32. The most useful extensions are:

1. distributions and low-tail exposed-polarity probabilities rather than ensemble means alone;
2. paired water/low-dielectric/membrane ensembles to measure environmental response;
3. explicit microstate and force-field sensitivity;
4. matched-pair tests within linker series;
5. membrane partition/deformation observables; and
6. prospective falsification through linker edits predicted to change a specific conformational state.

## Bottom line

Kim et al. materially broaden the PROTAC folding precedent from three compounds to 32 and show that metadynamics-informed 3D observables carry useful signal. The current attachments justify the physics feature layer, but their random-split performance, mixed assay provenance, and missing raw workflows do not establish a transferable decision model for the internal series.
