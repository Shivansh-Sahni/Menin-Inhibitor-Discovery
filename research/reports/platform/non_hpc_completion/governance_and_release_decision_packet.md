# Governance and release decision packet

Decision snapshot: 2026-08-05. This packet makes the remaining human decisions
answerable; it does not make them on behalf of the rights holder, institution,
scientific owner, or compute approver.

## Current automated evidence

- The Git migration is deliberately dirty and has zero staged files. A public
  release must be reviewed from an explicit staged candidate, not the working
  tree.
- No nonignored file larger than 50 MiB was found. The 36 larger local files
  are ignored raw, quarantine, split, literature, or corpus artifacts.
- The current Git-visible documentation/code/configuration scan reports zero
  personal-home paths and zero high-confidence secret patterns.
- No active `.building` or `.partial` artifact and no special file was found.
- No top-level `LICENSE`, `COPYING`, or `NOTICE` exists. Public distribution is
  therefore not approved.
- `.tmp/meeting_deck_2026_08_03` is preserved user scratch, excluded from the
  release, and is not part of any final artifact root.

The machine-readable evidence is in `non_hpc_governance_report.json` and
`release_inventory.csv`. These scans reduce review effort but cannot replace a
human review of the exact staged bytes.

## Recommended rights architecture

This is a technical recommendation, not legal advice or a license grant.

| Surface | Recommended decision for owner/counsel review | Why |
|---|---|---|
| Original code | Prefer **Apache-2.0** if institutional and contributor ownership permits | It is a recognized permissive license with an express patent grant and defensive termination, useful for a biotech/software project. MIT or BSD-3-Clause remain reasonable if counsel prefers simpler obligations. |
| Documentation and original figures | Consider **CC BY 4.0**, separately and explicitly | Documentation is content, not software. A separate notice avoids implying that upstream data inherit the code license. |
| ChEMBL-derived data | Preserve ChEMBL 37 attribution, release identity, and **CC BY-SA 3.0** review | The code license cannot relicense database content. Distributed derivatives need source-specific share-alike analysis. |
| BindingDB candidates | Keep row-level curation origin and use only rights-cleared rows | BindingDB-curated, imported ChEMBL, and Taylor-origin rows have different dispositions. |
| UniProt candidates | Preserve accession/release/version and CC BY attribution; retain inactive/ambiguous quarantine | Sequence identity and database rights are separate from model fitness. |
| ClinicalTrials.gov, Drugs@FDA, DailyMed | Preserve official record/version/section lineage; approve derived redistributable views separately | Government/public access does not make every normalized interpretation a validated molecular outcome. |
| Third-party model code/weights | Store exact license snapshots and checkpoint hashes per model | Repository, package, weight, output, and commercial-use terms can differ. |
| Project-trained weights | Choose a separate model-artifact license only after source-data and institution review | Training-data conditions may constrain weights or public use even if code is permissive. |

Primary references for the decision include the
[OSI discussion of express patent grants](https://opensource.org/blog/patents-and-open-source-understanding-the-risks-and-available-solutions-2),
the [official ChEMBL release/license page](https://www.ebi.ac.uk/chembl/beta/),
and each source/model's official terms captured in the platform literature and
source reviews.

## Required approval record

The following must be completed before a license file or public release is
created:

| Decision | Required signer/evidence | Status |
|---|---|---|
| Code copyright holder and institutional ownership | Principal investigator/institution or technology-transfer office | `pending` |
| Contributor authority and third-party code compatibility | Repository owner plus dependency/notice inventory | `pending` |
| Code license | Exact SPDX license and canonical text | `pending` |
| Documentation/content license | Exact license and scope statement | `pending` |
| Per-source redistribution | Source-by-source decision bound to release and artifact paths | `pending` |
| Model code/weights/output terms | Exact model revision, immutable terms snapshot, intended use | `pending` |
| Large-artifact storage | Content-addressed store, retention, access, checksums, restore test | `pending` |
| Staged disclosure | Human review of every staged add/modify/delete, private correspondence, PII, secrets, and binaries | `pending` |
| Clean-clone reproduction | Independent checkout using only approved release inputs | `pending` |

## Artifact-storage recommendation

- Keep Git for code, configuration, small reports, schemas, and manifests.
- Store large immutable raw and generated artifacts in a content-addressed
  institutional object store with SHA-256, source/access class, retention,
  version, and restore-test metadata.
- Do not publish the current 67.35 GB platform-data tree as an undifferentiated
  archive. Separate source redistributability, canonical derivatives,
  deterministic replicas, and quarantined failures.
- Retain failed canonical attempts as internal forensic evidence, but exclude
  them from the public release package unless there is a specific reproducibility
  reason and rights approval.
- Publish lightweight manifests that allow authorized users to verify retrieved
  objects. Never replace a source-specific license with the repository license.

## Honest boundary

The automatable hygiene surface is substantially prepared. Release remains
blocked by the dirty migration, the absent license, the absence of an approved
artifact store, the lack of a staged human review, and the lack of an
independent clean-clone reconstruction. None of these blockers is evidence that
the scientific artifacts are mechanically invalid; they are separate release
and governance gates.
