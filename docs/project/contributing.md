# Contributing

Thank you for improving the Menin discovery workflow. Contributions should preserve scientific traceability, chemical-identity controls, and the separation between public and proprietary work.

## License and contribution status

The repository does not yet have an approved software license or public contribution agreement. Before submitting an external contribution, contact the repository owner to confirm that the contribution can be accepted and under what terms. Submitting code does not by itself change the repository's license status.

Do not contribute material you do not have the right to share.

## Development setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

The repository includes `.pre-commit-config.yaml`. If you use pre-commit, install that tool separately in your development environment and run `pre-commit install`.

Run before requesting review:

```bash
ruff check .
pytest
menin-pipeline --stage all --skip-network --fast
menin-pipeline --stage verify
```

For changes to curation, splitting, models, or reports, also run the non-fast offline pipeline and inspect generated tables/figures. Do not commit refreshed data or model artifacts unless their scope and redistribution have been approved.

## Change principles

- Preserve source values and provenance; add normalized fields rather than overwriting evidence.
- Fail closed for unknown units, endpoints, targets, structures, or assay context.
- Keep observations, exclusions, aggregates, and predictions in separate fields/files.
- Group registered structures across train/test and cross-validation boundaries.
- Select models inside training-only cross-validation and preserve an untouched holdout.
- Record configuration, environment, hashes, fallback behavior, and limitations.
- Prefer small, reviewable changes with tests over notebook-only transformations.

## Code changes

- Target Python 3.10+ and follow the configured Ruff rules and line length.
- Add type annotations and concise docstrings for public APIs.
- Use `pathlib.Path`, deterministic ordering, explicit seeds, and atomic writes where files can be partially downloaded or generated.
- Do not make network calls in unit tests. Use small synthetic fixtures and dependency injection/mocking.
- Avoid unsafe model loading. Prefer `skops`; never load an untrusted pickle/joblib artifact.
- Add or update tests for every bug fix, schema change, curation rule, split invariant, and fallback.
- Update `docs/project/changelog.md`, README, methodology, architecture, and data dictionary when behavior or contracts change.

## Data and curation changes

A curation change should include:

- scientific rationale and source/reference;
- before/after row and structure counts by source, endpoint, assay family, and exclusion reason;
- a schema/data-dictionary update;
- unit tests for accepted, excluded, missing, malformed, censored, and duplicate examples;
- sensitivity of core model metrics when the affected population is material; and
- a new processed build/manifest rather than silent replacement of a reported dataset.

Never infer nM, IC50, Menin relevance, or an exact value from missing metadata. Manual PubChem decisions belong in a reviewed, versioned registry with reviewer evidence—not an undocumented code branch.

## Model changes

A new model or feature representation should preserve:

- dummy and simple baseline comparators;
- compound-grouped split and CV invariants;
- training-only selection/calibration;
- split assignments and digests;
- holdout predictions and comprehensive metrics;
- confidence intervals, uncertainty, and applicability-domain diagnostics;
- feature/model/environment manifests; and
- a clear fallback/trust policy.

Do not promote a model because it wins on the final holdout. Add it to a pre-specified candidate set and compare through training cross-validation or a new independent evaluation.

## Proprietary, sensitive, and regulated data

Do not open an issue or pull request containing internal structures, assay results, compound IDs, protocols, file hashes, screenshots, snippets, logs, or links. Do not run proprietary data through public CI or an unapproved AI/cloud service.

Follow [the proprietary-data intake protocol](docs/proprietary_data_intake.md). If private data are exposed, stop sharing and notify the data steward/security contact; deleting the visible file is not sufficient because Git and CI retain history/artifacts.

## Pull request description

Include:

- problem and scientific/software rationale;
- files/contracts affected;
- exact verification commands and results;
- data/metric deltas, if applicable;
- backward-compatibility or migration impact;
- security, confidentiality, and third-party-data impact; and
- remaining limitations.

Keep unrelated changes out of the same pull request. Generated files should be clearly identified and traceable to their generation command.

## Documentation links

- [Architecture](docs/architecture.md)
- [Methodology](docs/methodology.md)
- [Data dictionary](docs/data_dictionary.md)
- [Reproducibility](docs/reproducibility.md)
- [Limitations](docs/limitations.md)
- [Publication checklist](docs/publication_checklist.md)
