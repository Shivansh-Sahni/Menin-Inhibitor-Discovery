# hERG HPC execution preflight

## Result

The accepted data and benchmark inputs validate, and the present CPU environment
is sufficient for lightweight preparation. It is **not** ready for production
fancy-feature generation or HPC training.

## Current environment

- Python 3.13.7 on macOS arm64.
- Present: RDKit 2026.3.3, NumPy 2.5.1, pandas 3.0.3, PyArrow 23.0.1,
  SciPy 1.18.0, scikit-learn 1.9.0, XGBoost 3.3.0, LightGBM 4.7.0, and
  PyTorch 2.13.0.
- Missing future-stage packages: `openmm`, `torch-geometric`, `transformers`,
  `meeko`, and `vina`.
- The intended production runtime is not the current shell: Python 3.13.7 does
  not satisfy the frozen `3.11.*` constraint and PyTorch 2.13.0 does not satisfy
  `2.7.*`. No environment was silently repinned.
- The feature contract still has 25 intentionally unresolved production fields,
  including model/provider revisions, receptor/construct choices, docking schema,
  exact runtime pins, container/lock digests, and physics definitions. The WT protein sequence itself is no longer unresolved:
  reviewed human KCNH2 `Q12809`, 1,159 amino acids, is frozen by SHA-256.
- Approximately 41.41 GB is currently available on the workspace volume.

## Stage disposition

| Stage | Local software | Local storage with 20% headroom | Decision |
|---|---|---|---|
| S0 10–100 molecule smoke | PyG/Transformers missing | Fits | Blocked until the required cheap-family runtime is pinned and installed |
| S1 parent 2D features | Present | Fits minimum estimate | Defer until feature schema/version freeze |
| S2 protomer/conformer enumeration | OpenMM missing | Does not fit | Requires HPC/external storage |
| S3 classical baselines | Present | Fits minimum estimate | Training remains outside this pre-HPC release |
| S4 GNN/SMILES comparators | PyG/Transformers missing | Does not fit | Requires pinned environment and accelerator |
| S5 docking/PLEC | OpenMM/Meeko/Vina missing | Does not fit | Requires receptor/docking freeze and TB-scale storage |
| S6 multitask/pretraining | PyG/Transformers missing | Does not fit | Requires HPC |

Only two of seven blocking readiness gates currently pass: accepted inputs and
current dependency consistency. Target-runtime incompatibility, unresolved
production versions, missing future packages, inadequate local storage for all
stages, and the unexecuted two-run smoke test remain blocking by design.

Machine-readable package, stage, readiness, environment, input-binding, and
artifact manifests are under
`research/data/platform/processed/herg_hierarchy/v1_5_hpc_preflight/`.
Manifest self-hash:
`21bc07fe14f6cc9653630d923c857c9efcb8ab2dc4585babc53cab5c4199cc30`.

The preflight is also physically bound to the WT-reference manifest
`502b70e765237b615ba3f622eabf284f7321e95d8d7234dc84b7c0fb7874807d`.
That sequence asset authorizes future WT sequence features only; it does not
select an experimental construct or docking receptor.

The accepted-input gate validates and binds the master, corrected v1.4 review
assets, v1.5 candidate evidence, v1.5 label-blind benchmark, WT reference,
matched-pair analysis, and QT/exposure preparation. It binds the feature/HPC/smoke
contracts and the preflight implementation; the separate full contract validator
also passes.

No feature, model, smoke test, or HPC job was executed by this preflight.
