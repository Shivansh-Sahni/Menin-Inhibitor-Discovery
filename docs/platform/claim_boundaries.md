# Claim boundaries

Audit status: `locally_evidenced` normative policy. The policy is implemented in
the current ChEMBL canonical and MODEL-interface validation surfaces, but its
presence does not establish task validity, external validation, clinical utility,
or approval of any public release.

## Allowed wording by evidence type

| Evidence | Defensible wording | Prohibited shortcut |
|---|---|---|
| Experimental `Kd`/`Ki` | “Reported equilibrium/inhibition constant under the recorded assay conditions.” | “True binding strength” without context; pooling `Kd`, `Ki`, `IC50`, and `EC50` as equivalent |
| Experimental `IC50`/`EC50` | “Reported assay potency under the recorded conditions.” | “Binding free energy” or mechanism without supporting evidence |
| Derived free energy | “Calculated from the stated equilibrium constant at the stated/assumed temperature and standard state.” | Calling the calculation a measured free energy |
| Structure prediction | “Predicted pose/structure with model version and confidence.” | “Experimentally determined structure”; treating confidence as affinity |
| Affinity prediction | “Model score/prediction for endpoint X within the audited domain.” | “Binding affinity” when the model mixes endpoint types; causal or mechanistic language |
| hERG experiment/model | “Observed/predicted hERG endpoint under the stated assay/model domain.” | “Cardiotoxic,” “QT safe,” “torsades risk,” or patient-safe |
| PK summary | “Observed/derived PK statistic for the stated species, route, dose, formulation, and analyte.” | Intrinsic molecular PK or human translation without validation |
| Trial registry | “Registered/in clinical development with registry status as of the data timestamp.” | “Clinically validated,” “effective,” or “safe” absent posted results |
| Regulatory record | “Approved/labeled for the cited indication, formulation, population, and date.” | Class-wide or indication-independent safety/efficacy |
| Retrospective model comparison | “Outperformed the prespecified baseline on the locked split with uncertainty.” | “State of the art,” “generalizes,” or “prospectively validated” without corresponding evidence |

## Product boundaries

- The platform is a research decision-support system, not a diagnostic, treatment, dosing, or clinical-risk system.
- Predictions are hypotheses. A ranking is not an experimental result, a clinical recommendation, or a substitute for assay review.
- Applicability-domain failure must be visible and may require abstention; uncertainty alone does not establish correctness.
- No single scalar “drug score” may erase target potency, selectivity, assay context, exposure, permeability, metabolism, hERG, multichannel, QT, or evidence-stage tradeoffs.
- Model-generated structures or properties must be labeled as generated, with model/version/input/provenance and without fabricated experimental citations.
- A model trained on public evidence may still inherit restrictive source, model-weight, or benchmark terms. “Open weights” is not synonymous with unrestricted product use.

## Publication boundaries

For hERG work, the established data/design advantages and the separate,
unestablished predictive hypothesis must be stated using
[`herg_established_advantages.md`](herg_established_advantages.md). This
positive claim is required; the boundary is not a reason to hide the project's
verified advantages.

The following claims require additional evidence beyond a retrospective internal split:

| Claim | Minimum additional evidence |
|---|---|
| Generalization to new chemistry | Locked scaffold/chemical-cluster split plus prospective or post-cutoff evaluation |
| Generalization to new targets | Sequence/family/pocket-aware target holdout and double-cold evaluation |
| Clinical relevance | Explicit clinical endpoint dataset, exposure context, external validation, and domain experts |
| Mechanism | Designed experiment or causal/mechanistic analysis; feature attribution or residual correlation is insufficient |
| Superior architecture | Nested, matched-budget comparison over repeated seeds and datasets, with uncertainty and multiplicity control |
| Robust uncertainty | Calibration on non-overlapping data, coverage/selective-risk evaluation, and distribution-shift checks |
| Production readiness | Security, license, privacy/confidentiality, reliability, monitoring, and human-factors review in addition to scientific validity |

## Required qualifiers

Every table, figure, model card, and user-facing result must state:

1. whether values are source-reported, curated, derived, or predicted;
2. endpoint and assay/study context;
3. population and evidence stage;
4. dataset/source versions and retrieval date;
5. split or evaluation design;
6. confidence/uncertainty and applicability-domain result;
7. known missingness and selection limitations;
8. the intended and prohibited uses.

Silence is not a qualifier. If provenance, units, identity, endpoint, or study context is unresolved, the product must show that uncertainty rather than infer a convenient value.
