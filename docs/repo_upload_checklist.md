# Repository sharing checklist

This short checklist covers repository publication. It does not replace the scientific [publication readiness checklist](publication_checklist.md).

## Blockers

- [ ] The rights holder has approved a repository license, or the repository is explicitly shared without a reuse license and that limitation is understood.
- [ ] Authorship/contributor metadata and `CITATION.cff` are approved. Do not add collaborators or mentors without their approval.
- [ ] Third-party data files have been reviewed against current source terms; files that should be regenerated rather than redistributed are removed.
- [ ] No proprietary structures, compound IDs, assay data, protocols, manifests, logs, models, caches, or Git history are present.
- [ ] A data steward has reviewed any branch/history that ever touched non-public data.
- [ ] Secrets, tokens, credentials, private URLs, personal information, and local paths are absent.
- [ ] Model files and generated analysis/prioritization outputs are approved for release; models include artifact hashes and trust warnings.

## Reproducibility and quality

```bash
pytest
ruff check .
menin-pipeline --stage all --skip-network
menin-pipeline --stage verify
git status --short
```

- [ ] Commands pass in a clean environment.
- [ ] Raw, processed, software, models, analysis, and reports manifests verify with valid build/upstream/direct-input links when analysis is enabled.
- [ ] README commands and links are current.
- [ ] Generated counts/metrics are reported from the current frozen build, not copied from an older README.
- [ ] Release configuration, environment, source access/status, and code revision are archived.
- [ ] Quarantine and quality summaries are included or accurately described.
- [ ] Limitations and source notices are visible from the README.

## Git hosting review

- [ ] Read and follow the [repository and artifact storage policy](artifact_storage.md).
- [ ] Confirm no staged added/modified file exceeds 10 MiB without explicit artifact review.
- [ ] Confirm raw/processed data, feature stores, local runs, checkpoints, model binaries, and caches remain outside Git history.
- [ ] Inspect tracked files: `git ls-files`.
- [ ] Inspect objects/history with an approved secret/data scanner.
- [ ] Review CI logs/artifacts, releases, issues, pull requests, wikis, caches, packages, and branch protections.
- [ ] Ensure CI never tries to access private storage or external credentials for the offline smoke test.
- [ ] Configure security reporting and least-privilege maintainers.
- [ ] Create the release from the reviewed tag and archive its checksums.

Suggested repository description:

> Reproducible curation, QSAR evaluation, and safety-triage workflow for Menin/MEN1 inhibitor research using public bioactivity data.

Avoid calling the project “publication-ready,” “validated,” or a “safety predictor” until the corresponding gates in [publication readiness](publication_checklist.md) are actually complete.
