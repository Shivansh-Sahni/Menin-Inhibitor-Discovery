# Repository structure

The repository has four visible content categories. New files should be placed
inside one of these roots instead of creating additional top-level groups.

```text
Menin/
├── pipeline/                 # Main software, configuration, tests, and environments
├── research/                 # Data, analyses, models, reports, and benchmarks
├── packages/menin-edit/      # Standalone molecular-editing companion package
├── docs/                     # Scientific and project documentation
├── README.md                 # Project entry point
├── pyproject.toml            # Python build and tool configuration
├── Makefile                  # Common developer commands
└── CITATION.cff              # Citation metadata
```

## `pipeline/`

| Path | Purpose |
| --- | --- |
| `pipeline/src/menin_discovery/` | Main data-curation, modeling, analysis, and reporting package. |
| `pipeline/config/` | Versioned pipeline, validation, and intake policies. |
| `pipeline/scripts/` | Executable wrappers and benchmark/release utilities. |
| `pipeline/tests/` | Main-package tests. |
| `pipeline/environments/` | Portable requirements, lock file, and Conda environment definition. |

## `research/`

| Path | Purpose |
| --- | --- |
| `research/data/raw/` | Source snapshots grouped by provider. |
| `research/data/interim/` | Temporary pipeline intermediates. |
| `research/data/processed/` | Curated, normalized, and quarantined analytical tables. |
| `research/data/internal/` | Internal source workbooks with clear, date-qualified names. |
| `research/analysis/` | Primary chemical-intelligence outputs. |
| `research/models/` | Release models, endpoint variants, evaluations, and sensitivity runs. |
| `research/reports/` | Summaries, figures, publication tables, validation evidence, and manifests. |
| `research/benchmarks/herg/` | hERG benchmark experiments and fitted benchmark artifacts. |
| `research/literature/` | Papers, supplementary assets, structures, and the literature catalog. |
| `research/simulations/` | Simulation protocols, inputs, audits, and HPC bundles. |
| `research/notes/` | Dated research and meeting notes. |
| `research/outputs/` | Final user-facing research deliverables such as assay workbooks. |

Analysis tables mirrored under `research/reports/tables/` and smoke artifacts
mirrored under `research/reports/smoke/` are intentional pipeline contracts.
Different provider IDs can likewise have byte-identical exports while remaining
distinct source records.

The paths above describe the complete local workspace, not the set of files
uploaded to GitHub. Large data, feature, run, and model artifacts remain in
governed artifact storage; Git retains their compact manifests, checksums,
schemas, reports, and build instructions. See [the repository and artifact
storage policy](artifact_storage.md).

## `packages/` and `docs/`

`packages/menin-edit/` is independently installable but resolves selected model
and evidence artifacts from `research/`. Its code, tests, configuration, examples,
and documentation remain together.

Scientific documentation stays directly under `docs/`. Research-program policies
live under `docs/research/`, while the changelog and contribution guide live under
`docs/project/`.

## Naming and cleanup rules

- Use lowercase `snake_case` for maintained files and directories unless an
  upstream identifier, product name, or tool convention requires another form.
- Use ISO dates (`YYYY-MM-DD`) in filenames.
- Do not manually rename generated model, analysis, or report artifacts; their
  stable names are part of code and manifest contracts.
- Do not commit local environments, caches, coverage data, Python bytecode,
  editable-install metadata, or operating-system metadata.
- Do not commit raw downloads, processed tables, feature matrices, local run
  directories, or serialized models. Never bypass their ignore rules with
  `git add -f`; commit compact provenance and validation evidence instead.
