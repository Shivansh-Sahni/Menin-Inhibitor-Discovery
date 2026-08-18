# Future feature contract

Machine contract version `1.1.0`, dated 2026-08-09. This version replaces the earlier ambiguous row-key/level schema; it does not authorize execution.

## Non-execution statement

This is a contract for later CPU/HPC computation. No feature values were computed. Every produced array must be traceable to a standardized input identity, software/container digest, parameter set, and atomic output manifest.

## Identity and grain

The immutable canonical **parent join key** for ligand-bearing rows is `structure_id`; it is not a globally unique feature-row primary key. `standard_inchi_key` is a standardized-structure attribute, not the database primary key. Salts/mixtures are standardized before splitting. `standardized_smiles` identifies the normalized parent representation used for 2D features. Protonation, tautomer, conformer, protein construct/conformation, complex, and pose records receive separate IDs and never replace their parent identities.

Required hierarchy:

Allowed levels are exactly `parent`, `protomer_tautomer`, `conformer`, `protein_construct`, `protein_conformation`, `pose`, and `complex`. Ligand hierarchy is `parent -> protomer/tautomer -> conformer`; protein hierarchy is `protein construct -> protein conformation`; pose/complex rows join both branches.

The actual row key is the composite of feature family/version/level and all applicable ligand/protein hierarchy IDs, with canonical null normalization and uniqueness enforced. Ligand-only rows require `structure_id`, `standard_inchi_key`, and `standardized_smiles`. Protein-only rows require `protein_construct_id`, sequence accession, and sequence checksum; molecule fields are null and protein embeddings are stored once, never duplicated per molecule. Pose/complex rows require both ligand and protein identities. Hierarchy IDs are nullable only according to the machine mapping. Null is database/Arrow null, never an empty string or sentinel.

Labels remain observation-level and are joined only through frozen split manifests. Feature rows contain no label, activity, source outcome, quality tier, clinical status, or post-split aggregate.

## Planned families

| Family | Shape/unit | Default level | Purpose |
|---|---|---|---|
| Morgan radius 2 | `[2048]`, binary | parent | strong classical baseline |
| MACCS | `[167]`, binary | parent | reproducible compact fingerprint |
| RDKit 2D descriptors | fixed versioned vector, descriptor-native units | parent | interpretable physicochemical baseline |
| molecular graph | variable nodes/edges; categorical/numeric tensors | parent/protomer | GNN input |
| SMILES tokens | variable integer sequence plus mask | parent/protomer | sequence model input |
| pretrained molecular embedding | `[D]`, unitless | parent/protomer | transfer representation; exact model revision required |
| predicted pKa/logD | fixed scalar/vector, pKa/log10 partition units | protomer/parent | ionization/exposure-relevant context |
| 3D conformer geometry | `[N_atoms,3]`, angstrom | conformer | geometry-aware models |
| conformer energy | scalar, kcal/mol | conformer | Boltzmann aggregation |
| hERG protein embedding | `[D_protein]`, unitless | protein construct/conformation | protein-aware extension; WT sequence pinned |
| docking/PLEC contacts | fixed sparse/vector schema, contact-distance parameters pinned | pose | interaction evidence |
| physics calculations | named scalars with explicit physical units | conformer/complex | incremental mechanistic features only |
Exact dimensions that depend on a library/model are resolved in a versioned `feature_schema.json` emitted by the future job; unresolved dimensions are `NR`, never guessed.

The WT protein input is now frozen to the reviewed human UniProt canonical sequence `Q12809` / `KCNH2_HUMAN`, release `2026_02`: 1,159 amino acids, SHA-256 `287332153da38b59cc1be9554cc3a29f14d3b9e2a33150b4d54137773b22d1f7`. Its hash-bound local reference manifest is `502b70e765237b615ba3f622eabf284f7321e95d8d7234dc84b7c0fb7874807d`. This resolves the sequence identity only; no truncated experimental construct, protein conformation, membrane state, or docking receptor has been selected.

### Post-fit outputs—not molecular input features

Uncertainty, calibration, applicability-domain, and risk/coverage values form a separate `post_fit_outputs` artifact keyed by `structure_id`, `run_id`, and `split_role`. They are not one of the 12 planned input-feature families. If any post-fit value is ever used by a downstream learner, it must be generated through training-only cross-fitting and versioned as a derived prediction, never inserted into the molecular feature store.

## Provenance envelope

Every feature shard must record: contract version, feature family/version, parent/protomer/tautomer/conformer, protein-construct/protein-conformation, complex/pose identities as applicable, standardized input checksum, split-manifest checksum, source raw-record IDs, generation timestamp, command/config checksum, Git commit, container digest, OS/architecture, software/package versions, random seed, worker/job IDs, success/failure code, warning list, dtype, shape, units, and output checksum.

## Software policy

The contract pins RDKit to the compatible actual release `2026.03.3`. Execution must replace every other unresolved revision with an immutable container/package lock before production: Python `3.11.*` (`exact_patch=NR`), PyTorch `2.7.*` (`exact_patch=NR`), PyTorch Geometric compatible release (`NR`), Transformers `4.*` (`exact=NR`), OpenMM `8.*` (`exact=NR`), and docking engine/receptor structure (`NR`). Production is forbidden while any required pin is `NR`.

## Missingness and failure

- Never impute silently and never replace failure with zero vectors.
- Use explicit `feature_status`: `ok`, `not_applicable`, `input_invalid`, `enumeration_failed`, `conformer_failed`, `docking_failed`, `timeout`, `oom`, `software_error`, or `quality_rejected`.
- Retain error class and bounded diagnostic text; never retain secrets or full scheduler environments.
- Models receive a missingness mask or are restricted to complete cases according to a preregistered analysis.
- Retry only deterministic/transient categories under the HPC policy. Permanent chemistry failures remain data.

## Leakage rules

1. Standardize and assign parent/scaffold/source/temporal splits before any learned preprocessing.
2. Fit scalers, vocabularies, PCA, feature selection, imputation, calibration, and thresholds on training only.
3. Pretrained models must disclose pretraining corpus and suspected overlap; overlap is measured and reported, not assumed absent.
4. No activity, quality, clinical stage, assay result, source outcome, or test-derived statistic enters molecular features.
5. Protein inputs use the pinned WT KCNH2 sequence/construct; mutant structures/sequences are forbidden for the WT task.
6. Docking receptor/pose selection cannot use target labels. Any learned pose selector is frozen before test inference.
7. Uncertainty or applicability features used downstream must be cross-fitted on training and generated once on validation/test.

## Enumeration and aggregation

- Preserve one `structure_id` prediction as the primary output.
- Default physiologic aggregation conditions are pH 7.4, temperature 298.15 K, and ionic strength 0.15 mol/L. The future enumerator/provider and exact version remain `NR` and must be pinned before execution.
- Enumerate protomers/tautomers deterministically at pH 7.4 with maximum absolute formal charge 2, maximum 16 retained states, and predicted population cutoff 0.01. Renormalize retained populations to sum to one and store original/retained population mass. States below the cutoff contribute only to the discarded-mass field. Any changed pH grid or population rule creates a new contract version.
- Generate a bounded, deterministic conformer ensemble with pinned seed, RMSD pruning, energy method, and maximum count.
- Compute conformer Boltzmann weights **within the same protomer/tautomer state only**: set that state's minimum valid conformer energy to zero, `delta_E_i = E_i - min(E_state)`, and use `w_i = exp(-delta_E_i / (R*T))` with `R = 0.00198720425864083 kcal mol^-1 K^-1` and `T = 298.15 K`, then normalize within state.
- Never compare or combine raw conformer energies across different protonation/tautomer states. Aggregate conformers within state first; aggregate states to `structure_id` using the pinned pH-specific predicted protomer/tautomer populations—not energy differences between states.
- Also retain minimum, maximum, standard deviation, effective ensemble size, valid/attempted counts, and dominant member ID.
- If within-state energies are unavailable, use a preregistered uniform conformer mean and flag `conformer_weight=uniform`. If state populations are unavailable, the primary aggregated prediction is missing; an optional uniform-state sensitivity result may be stored separately with `state_weight=uniform_sensitivity`, never substituted silently.
- Pose-level interaction features aggregate per conformer using prespecified mean/max/contact-frequency summaries, then follow the conformer/protomer hierarchy.
- Stereochemistry-unknown parents remain explicitly unknown; enumerated stereoisomers may support sensitivity analysis but cannot be called observed stereochemistry.

## Production acceptance

A family may enter training only after the smoke test passes schema, determinism, row/ID conservation, leakage, failure-code, checksum, dtype/shape, finite-value, unit, and aggregation assertions. Scientific value is then tested as an ablation against the frozen Morgan/classical baseline.
