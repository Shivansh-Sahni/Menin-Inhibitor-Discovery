# Repository and artifact storage policy

This private GitHub repository is the collaboration surface for the project. It
contains source code, configuration, tests, documentation, schemas, compact
manifests and checksums, concise reports, and selected presentation files.

It deliberately does **not** contain the large scientific payloads generated or
consumed by the workflows. The following stay outside ordinary Git history:

- raw database downloads and source archives;
- normalized or processed observation tables;
- feature matrices and conformer shards;
- local campaign directories and logs;
- fitted models, checkpoints, prediction caches, and temporary build products;
- private, proprietary, rights-restricted, or collaborator-provided material.

Their absence from a clone is intentional and does not mean the project is
incomplete. Versioned manifests, source metadata, row and schema counts, and
SHA-256 bindings describe the governed artifacts needed to reproduce or verify a
release. The corresponding bytes remain in approved local or content-addressed
artifact storage and are shared separately only after rights and disclosure
review.

## Contributor rules

1. Do not use `git add -f` to bypass an artifact ignore rule.
2. Treat any new file larger than 10 MiB as an artifact requiring explicit
   review, even when GitHub would technically accept it.
3. Do not commit local runs, generated feature stores, serialized models, or raw
   database payloads. Commit their compact manifest or report instead.
4. Before every push, inspect staged files and confirm that no added file exceeds
   the repository threshold:

   ```bash
   git diff --cached --name-only --diff-filter=ACMR -z \
     | xargs -0 stat -f '%z %N' \
     | sort -nr \
     | head -20
   ```

5. A private repository is access-controlled, not exempt from data licenses,
   collaborator agreements, confidentiality obligations, or secret scanning.

See [the data-source notice](data_source_notice.md) for rights requirements and
[the repository sharing checklist](repo_upload_checklist.md) before widening
access or publishing a release.
